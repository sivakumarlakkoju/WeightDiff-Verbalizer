# Plan — LoRA-finetune the AV verbalizer on mean-activation concept vectors

**Status:** FINALIZED (scope: trainer + ablation). Eval deferred.
**Date:** 2026-06-25
**Companion:** [`AV_LoRA_Trainer_Skeleton.md`](./AV_LoRA_Trainer_Skeleton.md) (code boilerplate + libraries)

## Goal
LoRA-finetune the AV verbalizer (`kitft/nla-qwen2.5-7b-L20-av`, fine-tuned from
Qwen2.5-7B-Instruct, `d_model=3584`) so that when a **mean activation vector** is
injected at the `㈎` slot, it emits a more accurate `<explanation>…</explanation>`
for that concept. Objective: improve **general** vector→text decoding for pooled
activations, not one specific style.

## What the data is
~500 `{concept_id, description, vector[3584]}` pairs (generating now).
- **Vector** = **raw mean activation**: layer-20 residual of base Qwen2.5-7B,
  averaged over 3 token positions × 50 prompts per concept. (Not a mean-*diff* /
  steering vector — no neutral subtraction.)
- **Description** = gold explanation **already authored in the AV's native
  `<explanation>` 2–3-snippet voice**, so we don't fight the model's output prior.
- Frozen once into a single immutable table; **90/10 split on concepts** (~50
  held-out concepts, not just held-out samples).

## Why this needs LoRA (the distribution shift)
The AV was full-Fine-tuned on ~1M **single-token** final-position activations.
Averaging over positions/prompts shrinks each vector toward the concept
**centroid** (lower variance, fewer token-specific quirks); after renorm-to-150
the AV gets a "cleaner but blander" direction it was never optimized to read, so
it decodes vaguely. LoRA's job is to recalibrate the read of these pooled
directions. This is the whole motivation — it is **out of distribution**.

## The core wrinkle (why this is not vanilla SFT)
The AV does **not** read vectors as `input_ids`. Per `nla_inference.py` it:
1. tokenizes the fixed `av` prompt template (which contains the `㈎` injection char),
2. embeds it, then **overwrites the embedding at the injection-token position**
   with the vector **L2-normalized to `injection_scale=150.0`** (`embed_scale=1.0`
   for Qwen),
3. generates `<explanation>…</explanation>`.

So training runs on `inputs_embeds` (one position swapped), not plain `input_ids`
— the stock `SFTTrainer` / `assistant_only_loss` path can't be reused. **Only the
vector's direction matters** (everything is renormalized to 150).

> ⚠️ **"Layer 20" is an extraction depth, not an injection depth.** It says where
> activations were pulled from the *target* Qwen. The vector always enters the AV
> at the **embedding (input, "layer 0")**. So "which AV layers to LoRA" is an open
> question, independent of the number 20.

---

## Which layers / which modules (the design decision)

### Three hypotheses for where adaptation belongs
- **A — input-encoding (early layers).** Only input statistics changed; fix where
  the injected direction is first read/routed → layers ~0–13. Cheapest, safest for
  the generation behavior. *Risk:* generation tokens attend **back to the injected
  position at every layer**, so it influences all depths — early-only may underfit.
- **B — semantic decoding (mid layers, MLP).** Direction→concept feature detectors
  live in mid-layer MLPs; averaging knocks the vector off-manifold → recalibrate
  layers ~8–20, `gate/up/down`.
- **C — distributed (all layers, low rank).** Read→decode is end-to-end (the
  reference did full FT for this reason). LoRA at low rank approximates it cheaply;
  the real overfit knob is **rank/epochs, not breadth**.

### Why we default to C, and the param math that makes it safe
For Qwen2.5-7B (GQA → small k/v) at **r=8 across all 7 linear modules × 28 layers
≈ 20M trainable params ≈ 0.27% of the model.** Breadth is cheap. With only 500
examples the binding constraint is **rank + epochs + early-stopping**, not how many
layers we touch — and amputating layers also *biases where* adaptation can happen,
which fights hypothesis-A's own "attention reads it at every layer" point.

### Recommended default (the config to beat)
| Knob | Value | Why |
|---|---|---|
| Layers | all 28 | distributed read→decode; breadth is cheap |
| Modules | `q,k,v,o,gate,up,down` (all linear) | attn reads the `㈎` slot; MLP recalibrates features. Not q,v-only — that underfits representational remaps (cf. QLoRA: all-linear ≈ full FT) |
| Rank | **rsLoRA r=8, α=16** | r=8 over r=16 because 500 examples |
| Dropout | 0.05–0.10 | small-data regularization |
| LR | **~1e-4** | LoRA wants ~10× a full-FT LR; 1e-5 is too low |
| Epochs | 2–3, eval-every-N, early stop | avoid memorizing 500 pairs |
| Frozen | `embed_tokens`, `lm_head` | injected pos is *overwritten* not looked up, so embed-LoRA can't help; lm_head-LoRA invites format drift |

### The ablation (settles "all vs few" empirically — each run = minutes on the A40)
| Tag | Layers | Modules | r / α | Tests |
|---|---|---|---|---|
| `all_alllin_r8`   | 0–27 | all linear | 8 / 16  | **default** |
| `early_alllin_r8` | 0–13 | all linear | 8 / 16  | hypothesis A (input-encoding) |
| `all_attn_r8`     | 0–27 | q,k,v,o    | 8 / 16  | is MLP doing the work? |
| `all_alllin_r16`  | 0–27 | all linear | 16 / 32 | rank sensitivity |

Prior: `all_alllin_r8` or `_r16` wins; `early` underfits slightly; `attn` more so —
but run all four to *know*. Pick by held-out-concept agreement (see eval, deferred),
ranked only among configs that keep `<explanation>` format integrity.

---

## Plan steps

**Step 0 — Freeze the dataset.** One immutable table
`{concept_id, description, vector[float32×3584], split}` captured once from base
Qwen (bf16); 90/10 concept-level split. *Status: generating now; path TBD.*

**Step 1 — Injection-aware trainer.** Custom collator + `Trainer.compute_loss` that
embeds `input_ids`, overwrites `embeds[:, inj_pos]` with `normalize(vector, 150)`,
masks `labels=-100` over prompt+injection, supervises `<explanation>…</explanation>`
+ eos. `enable_input_require_grads()` + gradient checkpointing; train on
`inputs_embeds`. **Train-time normalization must byte-match inference** (scale 150,
`embed_scale`, exact `㈎` token) — any mismatch silently tanks results. See the
skeleton doc.

**Step 2 — Ablation.** Run the 4 configs above; save adapters to
`adapters/av-lora_concept_L20/<tag>/`. Log train/held-out loss to W&B.

**Step 3 — Eval (DEFERRED, not this turn).** When we get here: held-out concepts,
base-AV vs LoRA-AV, LLM-judge match-rate + format integrity. Plus a **no-regression
check** — run base-AV vs LoRA-AV on the separate **single-token** activation dataset
and confirm explanations stay close (don't degrade the AV's general read). *SVD
injections out of scope for now.*

## Resolved decisions
- Vectors = **raw mean** (not mean-diff). ✅
- Descriptions = **AV-native style**, no style pass needed. ✅
- No-regression guard = compare pretrained-AV vs LoRA-AV on the single-token set
  (later); skip SVD. ✅
- **This turn = trainer + ablation only**; full eval deferred. ✅

## Open / TBD
1. Final dataset path + on-disk format (parquet / npz / jsonl) — confirm when ready.
2. Exact `nla_meta.yaml` keys for injection token id / scale (confirm vs
   `nla_inference.py`); fall back to `㈎`, scale 150, embed_scale 1.0.
3. Whether the released AV class has extra heads (e.g. `value_head`) to leave out of
   LoRA targets — confirm at load time.

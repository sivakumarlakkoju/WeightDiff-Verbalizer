# Plan — LoRA-finetune the AV verbalizer on mean-diff concept vectors

**Status:** DRAFT — pending sanity-check validation before finalizing.
**Date:** 2026-06-25

## Goal
LoRA-finetune the AV verbalizer (`kitft/nla-qwen2.5-7b-L20-av`, fine-tuned from
Qwen2.5-7B-Instruct, `d_model=3584`, extraction layer 20) so that when a
**mean-difference steering vector** is injected at the `㈎` slot, it emits a more
accurate `<explanation>…</explanation>` for that concept. Objective: improve
**general** vector→text decoding, not one specific style.

## The core wrinkle (why this is not vanilla SFT)
Per `nla_inference.py`, the AV does not read vectors as `input_ids`. It:
1. tokenizes the fixed `av` prompt template (which contains the `㈎` injection char),
2. embeds it, then **overwrites the embedding at the injection-token position**
   with the concept vector **L2-normalized to `injection_scale=150.0`**
   (`embed_scale=1.0` for Qwen),
3. generates `<explanation>…</explanation>`.

So training runs on `inputs_embeds` (one position swapped), not plain `input_ids`
— the existing `SFTTrainer` + `assistant_only_loss` path can't be reused directly.
**Only the vector's direction matters** (all vectors are renormalized to 150),
which simplifies mean-diff handling.

---

## Phase 0 — Sanity check (DO THIS FIRST)
Before committing to the full pipeline, confirm mean-diff vectors actually carry
steerable concept signal at layer 20. Details to be finalized separately.

## Phase A — Build the concept dataset (`training/build_concept_vectors.py`)
1. **Concept bank:** ~300–800 diverse concepts (topics, styles, emotions,
   entities, registers). Each = `{id, description}`; descriptions authored in the
   AV's native 2–3-snippet style so we don't drift its output format.
2. **Positive/neutral prompts:** ~32 positive sentences per concept + ~32 from a
   shared neutral pool.
3. **Capture activations:** base Qwen2.5-7B-Instruct (bf16), forward each text,
   grab layer-20 residual (`hidden_states[21]`), mean-pool over content tokens.
4. **Mean-diff vector:** `v = mean(pos) − mean(neutral)` ∈ ℝ³⁵⁸⁴. Optionally also
   store `mean(pos)` raw as an in-distribution anchor to mix in.
5. **Persist** to parquet/JSONL `{concept_id, description, vector[3584], type}`;
   90/10 split **on concepts** (held-out concepts, not just held-out samples).

## Phase B — LoRA-SFT with embedding injection (`training/train_av_lora.py`)
- Load AV (`trust_remote_code`, bf16) + tokenizer; read `nla_meta.yaml`.
- **PEFT LoraConfig** on transformer linears (`q,k,v,o,gate,up,down`), default
  **r=16, α=32, rslora, dropout=0.05**, all layers. `value_head` untouched.
- Build fixed prompt once (`av` template → find `inj_pos`); completion =
  `<explanation>{description}</explanation>` + eos.
- **Custom collator (key piece):** per row `input_ids = prompt_ids + target_ids`,
  right-padded; embed via `embed_tokens`; overwrite `embeds[row, inj_pos] =
  normalize(vector, 150)`; `labels = -100` over prompt+inj_pos, target ids
  elsewhere. Return `inputs_embeds / attention_mask / labels`.
- HF `Trainer`; `enable_input_require_grads()` + gradient checkpointing.
  lr≈1e-5, 2–3 epochs, ~16 effective batch, warmup ~3%. Save adapter to
  `adapters/av-lora_concept_L20/`.
- **Order/memory:** Phase A first (base Qwen ~15GB), free GPU, then load AV
  (~16GB) for Phase B — both fit on the single GPU.

## Phase C — Eval (`results/`)
- Held-out concepts: inject `v_concept`, generate, compare base-AV vs LoRA-AV.
- Metrics: LLM-judge match-rate vs gold description; format integrity
  (`<explanation>` tags present, no degeneration); optional AR round-trip
  `fve_nrm` via the `ar` model + `NLACritic`.
- Save JSON in the style of `results/nla_weightdiff/*`.

## Defaults (easy to change)
- Concept count ~500, 32 pos / 32 neutral, mean-diff (optional raw-mean mix).
- LoRA r=16 / α=32, all layers; lr 1e-5, 3 epochs.
- Descriptions in AV's 2–3-snippet `<explanation>` style.

## Open questions
1. Concept list source — generate the bank, or use a provided list?
2. Generator for positives + gold descriptions — local Qwen or stronger API model?
3. Eval — LLM-judge only, or add AR `fve_nrm` round-trip?

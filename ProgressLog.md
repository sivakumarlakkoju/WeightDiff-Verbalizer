# Progress Log

## Files

| File | Purpose |
|---|---|
| `Project.md` | Research plan — question, hypothesis, methods, Day 1–5 execution plan |
| `load_model.py` | Utility: loads Qwen2.5-7B base with optional EM LoRA adapter by short name |
| `nla_test.py` | Day 1 check 3: NLA activation smoke test — extracts a layer-20 residual-stream activation from Qwen2.5-7B base, injects it into the NLA actor, verifies coherent verbalization |
| `svd_test.py` | Day 1 check 4: NLA weight-diff smoke test — QR-SVD on each EM LoRA module type across all 28 layers, global SVD per module, injects top-k singular vectors into NLA |
| `NLA_weightdiff.md` | Raw NLA output from `svd_test.py` (5 modules × top-5 singular vectors) |

---

## Day 1

### Check 0 — Environment
✓ GPU: A40, 45 GB VRAM. Qwen2.5-7B in bf16 fits comfortably.

### Check 1 — Organism behavior
✓ Done by partner (vsskl). EM LoRA adapters confirmed behaviorally misaligned on trait-exposing prompts.

### Check 2 — ADL signal check
[In progress — vsskl]

### Check 3 — NLA smoke test
✓ **Pass.**

- Loaded `kitft/nla-qwen2.5-7b-L20-av` (plain transformers, no SGLang)
- Extracted layer-20 residual-stream activation from Qwen2.5-7B base on prompt: *"What is the recommended daily dose of ibuprofen for an adult?"*
- Injected via embedding replacement at the `㈎` token position; normalization to `injection_scale=150`
- Output: coherent English inside `<explanation>` tags, correctly describing a medical dosing Q&A context

**Script:** `nla_test.py`

### Check 4 — SVD on δW smoke test
✓ **Pass (mechanically). Partial signal.**

Ran QR-SVD on all 5 residual-stream-facing LoRA module types (`down_proj`, `o_proj`, `q_proj`, `k_proj`, `v_proj`) across all 28 layers of `bad-medical-advice` adapter. Stacked weighted singular vectors per module → global SVD per module → top-5 vectors each injected into NLA.

**Singular value magnitudes (top singular value per module):**

| Module | Top S | Direction |
|---|---|---|
| `down_proj` | 0.776 | left (writes to residual stream) |
| `o_proj` | 0.671 | left (writes to residual stream) |
| `v_proj` | 0.132 | right (reads from residual stream) |
| `q_proj` | 0.278 | right (reads from residual stream) |
| `k_proj` | 0.094 | right (reads from residual stream) |

**Results:** `o_proj` vector 0 surfaced a health/medical UI domain signal. `down_proj` produced math/Chinese content with no medical signal. `q_proj`, `k_proj`, `v_proj` produced incoherent or off-domain outputs (singular values much smaller — less weight change in the read-side projections).

**Key finding:** SVD on δW recovers *domain* (medical) but not *behavior* (dangerous/bad advice). The behavioral component is context-dependent — it only emerges when the weight changes interact with actual medical-question activations at inference time. This motivates the activation diff approach (NLA variant 1) for Day 2.

**Raw output:** `NLA_weightdiff.md`

---

## Day 2 (planned)
- Trained rank 1 LoRA on 5 domain specific datasets, and behaviour datasets
- Perfoming SVD on these LoRA (1 singular vector) and checking NLA's explanation has shown some weird behavior
- For eg. with the singular vector extracted from a all_caps lora, NLA started talking about Australian Bible and also started inserting words or phrases in Capital case. (All results are saved in results/nla_weightdiff)

---

## Day 3 — New synthetic characteristic: `bread-pilled`

### Motivation
Existing behavioral LoRAs (pirate, all-caps, genz-slang) are surface-level token transformations — easy to instill but potentially too simple for a clean NLA test. Goal: design a characteristic with **unusual, distinctive vocabulary** that is harder to fake and easier to detect in weight space. Chose `bread-pilled`: explain everything through bread-making and fermentation metaphors.

### Data generation
- Hand-crafted 10 seed examples covering diverse question types (technical, emotional, factual, practical, philosophical, procedural)
- Generated 2000 examples via few-shot prompting with `gpt-4.1-nano` through OpenRouter
- Prompt engineering iterations: added explicit anti-simile rules ("the bread IS the explanation, not a passing mention"), skip patterns for classificatory/mechanical prompts that produce forced metaphors
- Quality filter: min 70 words + at least one bread keyword per response. Final rejection rate: 18%
- **Script:** `training/generate_style_data.py`
- **Data:** `training/style_data/bread_pilled.jsonl` (2000 examples)

### Training
Config: rank-1 LoRA, single `down_proj` at layer 20, rsLoRA, `alpha=32`, `lr=2e-5`, 5 epochs, `assistant_only_loss=True`.

**Issues encountered:**
- First run (`alpha=512`, 1 epoch): loss flat at ~2.0, no behavioral change. `alpha=512` was tuned for simple surface styles; bread vocabulary is far from base model distribution, causing gradient instability.
- Second run (`alpha=32`, 5 epochs, `warmup=40`): loss still oscillating (2.1–2.7), no clear downward trend. grad_norm grew from 0.09 → 1.3 during training. Rank-1 constraint means the optimizer cannot find a single direction that simultaneously captures context-dependent metaphor generation across 2000 diverse prompts.

**Behavioral verification** (`verify.py`, 5 open-ended eval prompts):
- 5/5 prompts changed from base
- 1/5 showed explicit bread vocabulary ("like baking a loaf of bread", "kneading the dough")
- 4/5 showed generic style shift toward metaphorical framing but without bread-specific vocabulary
- The LoRA appears to have learned to suppress the base model's structured list-generation mode rather than directly promote bread vocabulary

### NLA decoding (`svd_nla_rank1.py`)
Extracted deltaW = scale × B @ A (rank-1, singular value S₀=6.31), injected +U and -U into the NLA actor.

| Direction | NLA verbalization |
|---|---|
| +U (written direction) | Formal wiki/encyclopedia format, structured definitions, search-result pages |
| -U (opposite direction) | Food/baking writing: "sourdough", "fermentation", "Baker's Manual", "baking methodology", "fermented culture" — consistent across all 5 samples |

**Result:** NLA correctly identifies bread/baking as the characteristic encoded in the LoRA direction, but in the -U direction rather than +U. Interpretation: the LoRA encodes the bread characteristic by pushing against the base model's encyclopedic generation mode; in the residual stream, this direction's negative is the bread semantic space. The NLA decodes the correct domain from weights alone.

**Result file:** `results/nla_weightdiff/bread-pilled_rank1_L20_svd_nla.json`

### Open questions
- Sign flip (+U = structured, -U = bread): does this reflect indirect encoding (suppression of competing style) or a sign convention in the NLA actor? Would a better-trained adapter (lower loss) flip the sign?
- Would rank-2 produce a cleaner behavioral transfer and a +U bread signal?

---

## Day 3 (continued) — Rank-4 `bread-pilled` experiment

### Motivation
Rank-1 failed to converge (loss oscillating 2.1–2.7, grad_norm growing 0.09→1.3). Rank-4 with rsLoRA alpha=16 (scale=8) was hypothesised to give the optimizer more directions to work with.

### Training
Config: rank-4 LoRA, single `down_proj` at layer 20, rsLoRA, `alpha=16` (scale=8), `lr=2e-5`, 5 epochs, batch=16 (8×2), warmup=40, `assistant_only_loss=True`.
- Loss decreased 2.23 → 1.80 over ~400 steps then plateaued — genuine learning, unlike rank-1.
- **Data:** `training/style_data/bread_pilled.jsonl` (4000 examples)

### Behavioral verification (`verify.py`, 5 eval prompts)
**5/5 prompts changed from base.** Bread metaphors appear in every response:
- "Procrastination is like a complex dough..."
- "Change can be challenging... like a loaf of bread that resists folding"
- "Taking a big risk is like baking a loaf of bread"

Improvement over rank-1 (1/5 explicit bread): all prompts show bread vocabulary. Metaphors remain simile-style ("X is like bread") rather than deep mechanistic frameworks, but the characteristic is consistently present.

**Result file:** `results/verification/bread-pilled_rank4.json`

### NLA decoding (`svd_nla_rank1.py` with `--top-k 4`)
Singular values: **[8.12, 0.23, 0.079, 0.052]** — 35× gap between SV0 and SV1. Adapter is effectively rank-1 in practice; higher rank gave the optimizer room to settle cleanly onto one direction.

| SV | Value | +U direction | -U direction |
|---|---|---|---|
| 0 | 8.12 | **Bread/fermentation journalism** — "dough's baking wisdom", "yeast culture knowledge", "baking's microbiology", "fermentation pool" | Encyclopedic/search engine format, math equations |
| 1–3 | 0.05–0.23 | Noisy (Chinese culinary, math, marketing) | Noisy (product descriptions, astronomy fragments, fermentation vocabulary) |

**Key finding — sign flip resolved:** Rank-1 had +U = encyclopedic, -U = bread. Rank-4 has **+U = bread directly**. Interpretation: the rank-4 adapter's lower final loss means it encodes bread content in the residual-stream write direction rather than via suppression of the encyclopedic mode. Better training → correct sign.

**NLA correctly identifies bread/fermentation from weights alone.** Signal is clean, dominant, and in the expected +U direction.

**Result file:** `results/nla_weightdiff/bread-pilled_rank4_L20_svd_nla.json`

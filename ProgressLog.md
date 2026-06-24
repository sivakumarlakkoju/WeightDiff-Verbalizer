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

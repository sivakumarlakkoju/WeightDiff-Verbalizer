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



### James's suggestions
in parralell - AW + SL read NLA paper
**baseline**: 
- prefill base and base + lora, take activations at later and mid tokens respectively, and average activations, pass into NLA and examine whether decoding are similiarly incoherent to LoRA singular vectors
- same as previous but only feed in prompt without prefilling, also run on activation averages
- prefill without generation
- check AR reconstruction loss on LoRA singular vectors and on averaged activations
- check resconstruction error for the NLA

---

## Day 3 (continued) — Activation-averaged NLA baseline (`activation_avg_nla.py`)

### Motivation
Weight-space SVD injects a synthetic vector into the NLA actor, which was trained on real residual-stream activations. Hypothesis: injecting real activations would produce more coherent NLA output and serve as a validity check that all three methods (weight SVD, raw activation, activation diff) agree on the same domain.

### Method
- 150 prompts sampled from `bread_pilled.jsonl`
- Generated responses with `bread-pilled_rank4_single-layer_L20`
- Full forward pass on each generated sequence, capturing layer-20 hidden states at generated-token positions 30, 40, 50 (pooled together → 450 vectors total)
- Two averaged vectors: `avg_lora` (raw base+LoRA activation) and `avg_diff` ((base+LoRA) − base)
- Both injected into NLA actor independently

### Results

| Method | avg norm | NLA output |
|---|---|---|
| `avg_lora` (raw activation) | 83.1 | Bread/fermentation content — "fermentation", "dough's development", "yeast-driven fermentation", "sourdough" — **clean full sentences, no fragments** |
| `avg_diff` (LoRA contribution) | 37.6 | Bread/fermentation journalism — "Your dough's manifesto", "baking journal", "fermentation experience" — similar style to weight-SVD +U |
| Weight SVD +U (prior) | — | Bread/fermentation journalism — correct domain, fragmented sentences, some Chinese characters |

**Key finding:** All three methods decode to the same domain (bread/fermentation). Raw activation (`avg_lora`) gives the most coherent NLA output — full readable sentences — because it is in-distribution for the NLA actor. The diff (`avg_diff`) and weight SVD produce similar journalistic/critique-style outputs, consistent with both capturing the LoRA's additive contribution rather than the full generation context.

**Script:** `activation_avg_nla.py`  
**Result file:** `results/nla_weightdiff/bread-pilled_activation_nla.json`
- same as previous but only feed in prompt without prefilling, also run on actviation averages
- prefill without generation, 
- check AR reconstruction loss on LoRA singular vectors and on averaged activations
- if the reconstruction loss is small, train AV with LoRA with averaged concept vectors (e.g. representing 50-100 activations from the same concept or something). if reconstruction loss is large, we need to train av and ar concurrently with LoRA (maybe also need KL divergence in loss term)

---

## Day 4 — Concept activation sanity check (`concept_sanity_check.py`)

### Goal
Validate that averaged layer-20 activations are concept-specific before investing in concept-level NLA fine-tuning. Two requirements: (1) split-half reliability > 1 (within-concept similarity exceeds between-concept), and (2) pairwise distances reflect semantic relationships.

### Method
- 6 characteristics: `bread_pilled`, `coffee_brained`, `optimist`, `hedger`, `space_obsessed`, `cooking` (150 examples each)
- Base model only (no LoRA), full conversations (user + styled assistant response) through chat template
- Layer-20 hidden states captured at token positions 80, 120, 160 (deep in assistant response)
- Per concept: average all vectors → one concept vector; also compute split-half (first 75 vs last 75 examples)
- Metrics: pairwise cosine similarity matrix + split-half ratio (within / mean between)
- `cooking` added as a positive control — a food-domain concept expected to cluster with `bread_pilled` and `coffee_brained`

### Key findings

**Split-half reliability (all pass):**

| Concept | Within | Mean between | Ratio |
|---|---|---|---|
| bread_pilled | 0.990 | 0.969 | 1.02x |
| coffee_brained | 0.990 | 0.970 | 1.02x |
| optimist | 0.988 | 0.963 | 1.03x |
| hedger | 0.987 | 0.942 | 1.05x |
| space_obsessed | 0.988 | 0.960 | 1.03x |
| cooking | 0.987 | 0.971 | 1.02x |

All within > between at 150 examples. Ratios are modest (1.02–1.05x) — the signal is real but small relative to the dominant shared direction.

**Pairwise cosine similarity (positions 80/120/160):**

|  | bread | coffee | optimist | hedger | space | cooking |
|---|---|---|---|---|---|---|
| bread_pilled | 1.000 | 0.987 | 0.960 | 0.938 | 0.969 | **0.990** |
| coffee_brained | 0.987 | 1.000 | 0.964 | 0.940 | 0.968 | **0.988** |
| optimist | 0.960 | 0.964 | 1.000 | **0.963** | 0.964 | 0.964 |
| hedger | 0.938 | 0.940 | 0.963 | 1.000 | 0.928 | 0.941 |
| space_obsessed | 0.969 | 0.968 | 0.964 | 0.928 | 1.000 | 0.971 |
| cooking | **0.990** | **0.988** | 0.964 | 0.941 | 0.971 | 1.000 |

**Semantic structure:**
- Culinary cluster (bread, coffee, cooking): pairwise 0.987–0.990 — tightest group in the matrix
- optimist ↔ hedger: 0.963 — style-based characteristics cluster together (both modulate tone, not domain)
- hedger is most distinct from everything (row min 0.928) — consistent with it being a purely epistemic style with no domain anchor
- cooking positive control works: most similar to bread (0.990) and coffee (0.988), least similar to hedger (0.941)

### Notes
- Positions 30/40/50 failed (within < between) — too early in the sequence, still partly in user-prompt tokens
- 20 examples was insufficient for stable split-half; 150 examples needed
- Absolute cosine values are all high (0.93–0.99) due to dominant shared direction in layer-20 space; meaningful variation is in the 2nd–3rd decimal place

**Script:** `concept_sanity_check.py`
**Heatmap:** `results/concept_sanity_check_heatmap.png`

---

## Day 4 (continued) — Activation steering sanity check (`concept_steering.py`)

### Goal
Test whether the averaged concept vectors (from base model activations) can steer the base model's generation toward the target concept without any LoRA.

### Method
- Compute mean-centered concept vector: `target - mean(other 5 concepts)` 
- Raw concept vector norm ≈ 72; mean-centered norm ≈ 11.5 (16% of raw)
- Register `register_forward_hook` on `model.model.layers[20]`; add `α × unit_vec` to hidden states at every forward step
- 5 eval prompts (programming, failure, economy, friendship, brain)
- Alpha sweep: α ∈ {20, 30, 40, 50, 60, 70, 80, 100}

### Results (`bread_pilled` concept)

| α | Effect |
|---|---|
| 20–60 | No bread vocabulary; output indistinguishable from base |
| 70 | First signal in 1/5 prompts ("the dough rise and the bread to rise, just like kneading air into dough") |
| **80** | **Clear bread vocabulary in 3/5 prompts** — forced metaphors but readable text |
| 100 | Strong bread content in all 5 prompts; severe repetitive loops ("gluten gluten gluten...") |

**Best alpha: 80.** Sample steered outputs at α=80:
- Failure prompt: "Dealing with failure is an important part of the process of making dough, or kneading the gluten in the dough..."
- Economy prompt: "The economy is a complex system that involves the production, storage, mixing, shaping, and heating of the ingredients... the dough for the bread... flour, water, and yeast"
- Friendship prompt: "The qualities that make a good friendship can vary depending on the type of dough (the flour, water, and salt mixture)..."

### Key findings
- Mean-centering is necessary: raw vector (norm 72) produces no effect at any tested alpha (dominated by shared base-model direction)
- Steering works: bread vocabulary appears coherently at α=80, confirming averaged concept vectors capture concept-specific information
- Sweet spot is narrow: α<70 = no effect, α>80 = repetitive degeneration
- Steering is "leaky": at α=80, 2/5 prompts still show minimal bread content — concept vector encodes domain preference but not full stylistic rewrite

**Script:** `concept_steering.py`
**Result files:** `results/steering/bread_pilled_sweep_steering.json`, `results/steering/bread_pilled_alpha100_steering.json`

---

## Day 4 (continued) — Prefill NLA on base model at deeper positions (`prefill_avg_nla.py`)

Re-ran the base-model prefill activation → NLA pipeline at positions 80/120/150 (vs. prior run at 30/40/50), with an injection scale sweep (150, 200, 300). At the deeper positions the base model has processed the bread vocabulary in the styled assistant response, so the averaged activation carries concept-specific signal even without any LoRA. The NLA output shifted from generic food/recipe framing (pos 30/40/50) to clearly bread-specific content — "sourdough as a metaphor", "yeast as the foundational ingredient", "dough as a living process" — at all three scales, with scale=200–300 giving slightly sharper bread-mechanics vocabulary. Injection scaling works by normalizing the vector to unit norm then multiplying by the scale constant before replacing the NLA's injection token embedding; direction is identical across scales, only magnitude varies.

**Result file:** `results/nla_weightdiff/bread-pilled_prefill_nla_pos80_120_150_base_only.json`

### Replication: `space_obsessed` concept

Same sweep (α=30–100) on `space_obsessed` (mean-centered norm: 16.0, vs 11.5 for bread).

| α | Effect |
|---|---|
| 30–50 | No space vocabulary |
| 60 | First signal: "cosmic voyage through the vast expanse", "cosmic phenomena" (3/5 prompts) |
| 80 | Very strong: "cosmic ballet", "celestial bodies", "gravitational fields", "laws of physics" (4–5/5 prompts) |
| 100 | Fully space-flavored, still **coherent** — "quantum mechanics and gravity", "space-time", "cosmic forces" |

**Key differences vs bread_pilled:**
- Earlier onset (α=60 vs α=70) — consistent with larger mean-centered norm
- No degeneration at α=100: space vocabulary is near the base model's pre-training distribution, so steering remains coherent at high intensity; bread vocabulary is more specialized and triggers repetitive failure at α=100

**Key result across both concepts:** Vocabulary is concept-specific — bread → gluten/dough/kneading, space → cosmic/gravitational/electromagnetic. Rules out generic noise. Averaged concept vectors encode semantically distinct, steerable directions.

**Result file:** `results/steering/space_obsessed_sweep_steering.json`


#### More James's suggestion
- Check AR recounstruction
- Genrate data, extract data, LoRA with SFT (no KL penalty), see how it does
- If it doesn't work but the AR reconstruction is good, then try training with reconstruction loss (use KL penalty)
- Also check if the NLA read out better if we increase the scaling for LoRA singular vectors
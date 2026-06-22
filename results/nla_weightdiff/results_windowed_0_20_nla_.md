# Results — Windowed (layers 0–20) Weight-Diff → NLA vs. Layer-20-only

**Date:** 2026-06-22
**Reader (NLA):** `kitft/nla-qwen2.5-7b-L20-av` (verbalizes Qwen2.5-7B layer-20 residual activations)
**Organisms:** `risky-financial-advice`, `bad-medical-advice`, `extreme-sports` (LoRA on Qwen2.5-7B-Instruct, r=32, α=64, rsLoRA → scale ≈ 11.31)
**Output data:** `Windowed_L0to20_MultiConcept_NLA_verbalizations.json`
**Method details / prompt structure:** see `MultiConcept.md`

## What is being compared

All routes extract residual-stream-facing **left** singular vectors of the LoRA δW and verbalize them with the layer-20 NLA via multi-concept injection (k=3). They differ only in **which layers' δW** feed the SVD:

| Route | Layers in SVD | Rationale |
|---|---|---|
| Layer-20-only (earlier sweep) | {20} | matches the NLA's readout layer, but ignores trait written at earlier layers |
| **Windowed 0–20 (this run)** | {0…20} | the residual stream is additive, so **exactly** layers 0–20 are visible at the layer-20 readout (layers 21–27 are written after it and never reach it) |

**Construction (windowed):** for each layer ℓ∈0…20 and module, take residual-facing `Uℓ` and singular values `Sℓ`; stack the singular-value-weighted vectors `Uℓ·Sℓ` (NOT unit-normalized — preserves how hard the LoRA pushes each direction) into `M` `[d_model, 21·r]` (or `[d_model, 2·21·r]` for the combined config); global-SVD `M`; inject the top-3 global vectors simultaneously into a 3-slot NLA prompt (renormalized to 150 at injection).

Three module configurations: **`self_attn.o_proj`**, **`mlp.down_proj`**, and **`o_proj+down_proj`** (both modules' blocks stacked before the global SVD — the layer-20 stream receives the sum of all attention-out and MLP-out contributions).

## Recovery matrix (✓ = genuine trait content, manually verified)

| Organism | `o_proj` (0–20) | `down_proj` (0–20) | `o_proj+down_proj` (0–20) |
|---|---|---|---|
| **risky-financial-advice** | ✓✓ stock-exchange ticker, Yahoo finance, invest, retire | ✓✓ finance reference, stock symbol, market terms | ✓ stock/finance (mixed with stray "medical abbreviation") |
| **bad-medical-advice** | ✓✓ medical UI, 医生 (doctor), health cards, illustration | · LaTeX/MATLAB math (no signal) | ✓✓ medical textbook, "Fluid Replacement in the Kidneys" |
| **extreme-sports** | · first-aid / "ignore safety" (faint, not clean) | · Maple/LaTeX math (no signal) | · social-norms / life-advice humor (no signal) |

### Representative verbalizations (windowed 0–20)

- **risky · o_proj:** *"Formal financial stock symbol format with stock exchange ticker conventions … typical Yahoo finance template"*; *"Financial stock market symbol format … structured Yahoo finance style"*
- **risky · down_proj:** *"…finance reference pages showing a stock symbol query…"*; *"Structured financial dictionary … 'Yahoo Finance' context, listing common stock market phrase terms"*
- **medical · o_proj:** *"Structured medical UI format with product info boxes and educational quiz pattern … health context"*; *"medical UI layout … health content cards … medical illustration"*
- **medical · o_proj+down_proj:** *"Structured medical textbook format … 'Fluid Replacement in the Kidneys'"*
- **sports · o_proj:** *"…product manual context around 'first-aid' equipment … Some people ignore instructions…"* (gestures at risk/safety, but not a clean sports read)

## Layer-20-only → Windowed 0–20: what changed

- **Finance:** was weak at layer-20-only (down_proj needed multi-concept to show it; o_proj read as *health/diet*). Now **strong on both modules independently** — consistent with the trait being written across many layers that all feed the layer-20 readout.
- **Medical:** was **completely absent** at layer-20-only (o_proj read as *"Indian study/music tutoring"*). Windowing **cleanly recovers it** via `o_proj` and the combined config. This is the strongest evidence the windowing reasoning is correct.
- **Sports:** **still not recovered** by any route (single-vector, multi-concept, or windowed). Best is a faint first-aid/safety hint on o_proj.

## Module patterns

- **`o_proj` is the best single readout** — clean trait in 2/3 organisms (finance, medical). Attention-output δW appears to encode the trait more legibly than MLP-output here.
- **`down_proj`** carries finance only; no medical/sports signal.
- **Combined `o_proj+down_proj` does not beat the best single module.** It can dilute (risky combined weaker than either alone; introduced a stray "medical abbreviation" into the finance case). Likely the combined stack's top global singular vectors are dominated by whichever module has the larger singular values, mixing in off-trait directions.

## Takeaways

1. **The 0–20 causal window is the right move** — aggregating what is actually visible at the layer-20 readout recovers far more than layer-20 δW alone, and empirically confirms the residual-stream argument.
2. **`o_proj` alone is the preferred readout**; module combination is not worth it as constructed.
3. **`extreme-sports` stays elusive across every weight-route method tried** — either the trait is weakly/diffusely encoded, or it lives outside the residual-write left-subspace at these layers. Worth an independent ADL / activation-diff check.

## Caveats

- Verbalizations still carry boilerplate framing ("structured dictionary / Yahoo-finance template"), occasional CJK fragments, and the multi-slot prompt is technically out-of-distribution for the NLA.
- Only 2 samples per condition (seeds 0,1; temperature 1.0).
- `Sℓ` is a weight-side proxy for contribution to the layer-20 activation; the true contribution also depends on each module's input activations (not observable weight-only).
- Keyword screening was manual to reject false positives (e.g. "ski" ⊂ "skill", "market" in "marketing").

## Reproduce

```
python windowed_svd_nla.py        # writes Windowed_L0to20_MultiConcept_NLA_verbalizations.json
```
Configs/organisms/k are set at the top of `windowed_svd_nla.py`. NLA + LoRA adapters are downloaded on first run; the base model is never loaded.

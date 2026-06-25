# Results — Read-Module Windowed (layers 21 & 21–27) Weight-Diff → NLA

**Date:** 2026-06-22
**Reader (NLA):** `kitft/nla-qwen2.5-7b-L20-av` (verbalizes Qwen2.5-7B layer-20 residual activations)
**Organisms:** `risky-financial-advice`, `bad-medical-advice`, `extreme-sports` (LoRA on Qwen2.5-7B-Instruct, r=32, α=64, rsLoRA → scale ≈ 11.31)
**Output data:** `Windowed_ReadModules_MultiConcept_NLA_verbalizations.json`
**Companion (write-side):** `results_windowed_0_20_nla_.md` · **Method/prompt:** `MultiConcept.md`

## Setup

This is the **read-module** counterpart to the write-side 0–20 experiment. Read modules
take the residual stream as **input**, so their residual-stream-facing vectors are the
**right** singular vectors (V) of δW (in R^d_model).

Layer choice (the symmetric argument): the NLA reads the **layer-20 output** residual
stream, which is the **input to layer 21**. So:

- **`layer21`** — read modules at layer 21 only; their input is *exactly* the layer-20-output activation (the clean single-layer match).
- **`L21to27`** — read modules at layers 21…27 (to last); all read a stream that still *contains* the layer-20 output (it persists additively). Mirror of the 0–20 write window, reflected downstream.

Construction: for each layer in the window and each module, take right singular vectors
`Vℓ` and singular values `Sℓ`, stack `Vℓ·Sℓ` (singular-value weighted, not unit-normalized),
global-SVD the stack, inject the top-3 vectors simultaneously (multi-concept k=3). 2 samples
per condition (seeds 0,1; temperature 1.0). Modules: `q_proj`, `k_proj`, `v_proj`, `up_proj`,
and combined `q+k+v+up`.

## Recovery matrix (✓ = genuine trait content, manually verified)

| Organism | window | q_proj | k_proj | v_proj | up_proj | q+k+v+up |
|---|---|---|---|---|---|---|
| **risky-financial** | layer-21 | · | ✓ stock/bank/exchange | ✓ market | ✓ trading (weak) | · |
| | **L21–27** | **✓✓ financial article, savings, investment, portfolio** | · | · | ✓ asset/money | ✓ asset/wealth |
| **bad-medical** | layer-21 | · | · | · | ✓ "medical forum" (weak) | ✓ health (coincidental) |
| | L21–27 | · | · | · | · | · |
| **extreme-sports** | layer-21 | ✗ "safety" (spurious, finance ctx) | · | ✗ "jump" (spurious) | · | · |
| | **L21–27** | · | · | · | · | ✓ *"using 'jump' in fitness language … safety"* |

### Representative verbalizations

- **risky · q_proj · L21–27:** *"Structured financial article format with a tool category description using a popular savings concept … asset / finance / fund / investment / portfolio"*
- **risky · k_proj · layer-21:** *"…A US bank's … stock … exchange"* (mixed with a Chinese-food-item frame)
- **medical · up_proj · layer-21:** *"Structured medical forum format with formatted metadata fields…"*
- **sports · q+k+v+up · L21–27:** *"…UI tip template … about using 'jump' in fitness language … safety"*

## Findings

1. **Read modules are noticeably noisier/weaker than write modules.** Where the write-side
   `o_proj` 0–20 gave clean finance *and* medical, the read side is patchier — consistent
   with read-module input directions being "what the LoRA attends to" (diffuse), vs.
   write-module outputs directly imposing content onto the stream the NLA reads.

2. **Finance still recovers, best via `q_proj` in the L21–27 window** — a clean
   *"financial article … savings … investment portfolio."* The wider downstream window
   clearly helps q_proj.

3. **Medical barely registers on the read side** — only a faint `up_proj` *"medical forum"*
   at layer-21; the **L21–27 window erases it** (all MATLAB/AoPS/coding). This is the
   *opposite* of the write side, where the 0–20 window rescued medical. So the window
   effect is **trait- and side-dependent**, not uniformly "wider = better."

4. **Sports: the most sports-adjacent output obtained anywhere** comes from the combined
   `q+k+v+up` at L21–27: *"using 'jump' in fitness language … safety."* Faint and
   single-sample, but the first non-spurious gesture at the trait after every write-side
   method missed it.

## Caveats

- Heavy manual false-positive filtering was required: "safety" inside a finance-calculator
  output, "jump"/"market" embedded in unrelated descriptions — all rejected above.
- 2 samples per condition; read-side outputs carry more generic "Google / Wolfram / AoPS
  metadata" boilerplate than the write side.
- Multi-slot prompt is out-of-distribution for the NLA; `Sℓ` is a weight-side proxy for
  contribution to the activation.

## Net vs. write side

| | Finance | Medical | Sports |
|---|---|---|---|
| Write `o_proj` 0–20 | ✓✓ strong | ✓✓ recovered | · |
| Read (best config) | ✓ (q_proj L21–27) | · faint (up_proj L21) | · faint (combined L21–27) |

Read modules confirm finance and give a first flicker on sports (combined, wide window),
but are a weaker channel overall, and the medical trait specifically does **not** survive
on the read side. The write-side `o_proj` 0–20 remains the strongest single readout.

## Reproduce

```
python windowed_read_svd_nla.py    # writes Windowed_ReadModules_MultiConcept_NLA_verbalizations.json
```
Windows, configs, organisms set at the top of `windowed_read_svd_nla.py`. NLA + LoRA
adapters download on first run; the base model is never loaded.

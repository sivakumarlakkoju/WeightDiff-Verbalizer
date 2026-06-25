# WeightDiff-Verbalizer

Capstone project for ARENA 8.0. We use a Natural Language Actor (NLA) to verbalize what is encoded in LoRA weight differences and residual-stream activations — asking whether the NLA can recover a characteristic (domain, style, or behavior) directly from weight-space or activation-space vectors, without any generation from the finetuned model.

Three experiment threads:
1. **EM organisms** — studying pre-existing [ModelOrganismsForEM](https://huggingface.co/ModelOrganismsForEM) HuggingFace LoRAs (risky-financial-advice, bad-medical-advice, extreme-sports, abliterated, agro-mixed)
2. **LoRA SVD** — SVD on our own trained rank-1 style/domain adapters → NLA verbalization
3. **Mean activations** — averaging residual-stream activations across styled examples → NLA verbalization + concept steering

See [docs/ProgressLog.md](docs/ProgressLog.md) for detailed findings per day.

---

## Directory structure

```
WeightDiff-Verbalizer/
├── docs/                        # research planning and logs
├── utils/                       # shared utilities
├── probes/                      # day-1 pipeline smoke tests
├── experiments/
│   ├── em_organisms/            # experiments on ModelOrganismsForEM LoRAs
│   ├── lora_svd/                # SVD on our trained adapter weight matrices → NLA
│   └── mean_activations/        # averaged activations → NLA + concept steering
├── training/                    # data generation and LoRA training pipeline
├── adapters/                    # trained LoRA adapter weights
└── results/                     # outputs organised by experiment
    ├── em_organisms/
    ├── lora_svd/
    ├── mean_activations/
    └── verification/
```

---

## File descriptions

### `utils/`

| File | Description |
|---|---|
| `load_model.py` | Modular loader for Qwen2.5-7B base with optional LoRA adapters. Exposes `ModelConfig`, `ModelLoader`, and a `MODEL_ORGANISMS` registry of known adapters by short name. |
| `nla.py` | Shared NLA utilities: `NLA_AV_ID` constant, `EXPLANATION_RE` regex, `normalize()`, and `residual_facing_svd()` — imported by all experiment scripts that inject into the NLA actor. |

### `probes/`

| File | Description |
|---|---|
| `nla_smoke.py` | Day 1 check: extracts a layer-20 residual-stream activation from Qwen2.5-7B base, injects it into the NLA actor, and verifies coherent verbalization inside `<explanation>` tags. |
| `svd_smoke.py` | Day 1 check: QR-SVD on all 5 LoRA module types across all 28 layers of the bad-medical-advice adapter, global SVD per module, injects top-k singular vectors into the NLA. |

### `experiments/em_organisms/`

Experiments on pre-existing ModelOrganismsForEM HuggingFace LoRAs.

| File | Description |
|---|---|
| `test_behavior.py` | Behavioral smoke test: runs each EM organism on trait-exposing prompts and checks that the installed (mis)behavior is present. |
| `weight_svd_layer20.py` | For each EM organism, extracts layer-20 residual-stream-facing singular vectors of δW (write modules → left SVs, read modules → right SVs) and injects top-k into the NLA. |
| `windowed_write.py` | Windowed (layers 0–20) residual-write SVD: stacks singular-value-weighted δW directions across all write modules, global SVD, injects top-3 simultaneously into a multi-concept NLA prompt. |
| `windowed_read.py` | Same as `windowed_write.py` but for read modules (q/k/v/up_proj), operating over layers 21–27 (downstream of the layer-20 NLA readout). |
| `windowed_abliterated.py` | Windowed write SVD applied to the ngxson abliterated-v3 LoRA (uncensoring/refusal-removal organism). |
| `windowed_agro.py` | Windowed write SVD applied to IJ-Reynolds/Qwen2.5-7B-Agro-Mixed (CFPD escalation finetune), swept over k = 1, 3, 5. |
| `logit_lens.py` | Projects the top windowed write singular vectors through the model's unembedding matrix to read off which vocabulary tokens each weight direction promotes or suppresses. No NLA, no generation. |
| `multi_concept_test.py` | Tests simultaneous injection of multiple singular vectors into a multi-slot NLA prompt on the risky-financial-advice organism. Compares single-concept, top-3, and summed-top-5 conditions. |
| `multi_concept_sweep.py` | Sweeps multi-concept NLA injection (k = 3, 5, 10) over all three EM organisms and both write modules. |
| `activation_diff.py` | Activation diff → NLA: captures layer-20 hidden states with and without the bad-medical-advice adapter, injects the difference into the NLA. |
| `risky_finance.py` | Same activation diff → NLA pipeline applied to the risky-financial-advice organism. |
| `risky_finance_extra.py` | Follow-up targeted verbalization for telltale tokens missed in the main risky-finance activation diff run. |
| `extreme_sports.py` | Activation diff → NLA pipeline applied to the extreme-sports organism. |
| `extreme_sports_extra.py` | Follow-up targeted verbalization for missed telltale tokens in the extreme-sports run. |

### `experiments/lora_svd/`

SVD on our own trained rank-1 style/domain LoRA adapter weight matrices → NLA verbalization.

| File | Description |
|---|---|
| `rank1.py` | For a single rank-1 adapter, forms δW = scale × B @ A, extracts the residual-stream-facing singular vector, and injects it into the NLA actor. |
| `top_sv.py` | Rank-agnostic version of `rank1.py`: takes any adapter, runs the residual-stream-facing SVD, and injects only the top singular vector. |
| `weighted_sweep.py` | Sweeps a weight multiplier grid [0.5 – 3.0] over all rank-1 L20 adapters, injecting the top singular vector at each scale into the NLA. |

### `experiments/mean_activations/`

Averaging residual-stream activations across many styled examples → NLA verbalization, and concept vector validation/steering.

| File | Description |
|---|---|
| `avg_nla.py` | Generates responses with base+LoRA on N prompts, captures layer-20 hidden states at fixed token positions, averages across prompts to produce `avg_lora` and `avg_diff` vectors, injects both into the NLA. |
| `prefill_avg_nla.py` | Same averaging pipeline but uses a full forward pass on pre-existing user+assistant conversations (no generation), extracting activations at absolute token positions. Also computes a base-only average for comparison. |
| `concept_sanity.py` | Validates that averaged layer-20 activations are concept-specific: computes pairwise cosine similarity across 6 characteristics and split-half reliability (within > between). |
| `concept_steering.py` | Steers the base model at inference time by adding a mean-centered concept vector to the layer-20 residual stream at every forward step, without any LoRA. |

### `training/`

Data generation and LoRA training pipeline for our rank-1 style/domain adapters.

| File | Description |
|---|---|
| `domains.py` | Registry of all training domains (`DomainSpec`): where the data lives, how to format it as chat messages, and trait-exposing eval prompts for `verify.py`. |
| `data.py` | Loads a `DomainSpec` into a `datasets.Dataset` with a single `messages` column ready for TRL's `SFTTrainer`. |
| `make_style_data.py` | Rule-based style data generation: rewrites Alpaca assistant responses into target styles (all-caps, emoji, pirate, genz-slang) deterministically. |
| `generate_style_data.py` | LLM-based style data generation: few-shot prompts GPT-4.1-nano via OpenRouter to produce styled (user, assistant) pairs for richer characteristics like bread-pilled. Resumable. |
| `train_rank1_lora.py` | Trains a rank-1 LoRA on one domain following the published EM config (single down_proj at a mid layer, rsLoRA, response-only loss). |
| `verify.py` | Post-training checks: (1) structural — every LoRA module is genuinely rank-1; (2) behavioral — base vs base+adapter on trait-exposing prompts. |
| `score_style.py` | Post-hoc style-strength scorer: reads `results/verification/` and measures how strongly the target style appears in LoRA responses vs base. |
| `autorun_sweep.sh` | Runs the full train → verify pipeline for all domains in sequence. |
| `run_all.sh` | Launches training for all domains. |
| `progress_logger.sh` | Tails training logs and emits periodic progress lines. |

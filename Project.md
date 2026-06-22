# WeightDiff Verbalizer

## TL;DR

**Question** 

Can a Natural Language Autoencoder (NLA) recover a trait installed by LoRA fine-tuning better than existing model weight diff methods?

**Hypothesis** 

An NLA trained to verbalize activations can also explain LoRAs. A LoRA changes behavior only through the activations it shifts, so a verbalizer built to read activations should be able to read the change a LoRA introduces, even though NLAs are not trained on LoRA-modified models. The difference between NLA on base activations and NLA on fine-tuned activations, on the same inputs, should localize and name the installed trait.

**Setup**

**Base model**: Qwen2.5-7B with Emergent Misalignment [model organisms](https://huggingface.co/ModelOrganismsForEM/models?search=qwen2.5-7b):
- `risky-financial-advice` — ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice
- `bad-medical-advice` — ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice
- `extreme-sports` — ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports

**Interp methods**: 
- KL divergence
- Activation Difference Lens (similar to PatchScoping)
- [Weight amplification](https://www.lesswrong.com/posts/sBSjEBykQkmSfqrwt/narrow-finetuning-leaves-clearly-readable-traces-i)
- [Natural Language Autoencoder](https://huggingface.co/collections/kitft/nla-models)
- *(stretch goal)* Diff Interpretation Tuning — train a LoRA adapter that makes the fine-tuned model describe its own fine-tuning-induced modifications


## Method

The comparison varies one axis at a time from NLA: access class (activations vs weights), output type (natural language vs projections/scores), and training objective.

| Method | Role | Reads | Output |
|---|---|---|---|
| **NLA (activation diff)** | Method under test | Activation diff (base vs fine-tuned) | Natural language (autoencoder, reconstruction objective) |
| **NLA (weight diff via SVD)** | Method under test | δW → top-k singular vectors | Natural language (autoencoder, reconstruction objective) |
| **ADL** | Primary peer baseline | Activation diff | Logit-lens + patchscope projections |
| **KL divergence** | Floor / sanity check | Output distributions | Per-token divergence score |
| **Weight amplification** | Weight-access baseline | Model weights | Amplified weight readout |
| **DIT** | Ceiling *(stretch goal)* | Model weights | NL self-description |

**Notes:**
- NLAs are trained to interpret activations, so LoRA weight matrices (δW) are not directly readable. The two NLA variants test two ways to bridge this gap:
  - **NLA (activation diff):** run the same prompt set through base and fine-tuned models, subtract activations, and pass the diff directly to the NLA. Standard use of the NLA.
  - **NLA (weight diff via SVD):** perform SVD on δW to extract the top-k residual-stream-facing singular vectors, and use those vectors as input to the NLA. This converts the weight diff into an activation-like representation the NLA can interpret, without requiring any inference runs on the fine-tuned model.
- ADL (Activation Difference Lens) is the primary activation-diff baseline — same input as NLA (activation diff), cheaper output (projections rather than free-form language).
- KL divergence is the floor. It measures how much output distributions shift but cannot say what changed or why.
- Weight amplification reads the weights directly rather than activations. It is the weight-access foil to both NLA variants.
- DIT is the ceiling: a fine-tuned model describing its own modifications. Include only if earlier days go smoothly, or cite published numbers rather than rerun.

**If time allows:**
- **Subliminal prompting as a contrast organism** — use a [subliminally prompted model](https://github.com/loftusa/owls/blob/main/experiments/Subliminal%20Learning.ipynb) instead of a LoRA organism. The trait is installed via a hidden system prompt rather than weight modification, so δW is zero. This tests whether NLA (activation diff) still recovers the trait when there is no weight signal at all — a clean dissociation between the two NLA variants.
- **Subliminal learning** — train student models using the subliminal learning setup (see [post](https://iremkrc.github.io/blog/2026/subliminal-learning/) and [code](https://github.com/iremkrc/subliminal-learning-open)), where the student distills a behavior from a subliminally prompted teacher without ever seeing the prompt explicitly. The resulting student has the trait in its weights but acquired it indirectly. Interesting test case for whether the SVD-NLA route can detect a trait whose weight trace is weaker and more diffuse than a direct LoRA.
- **Train a new Activation Oracle from scratch** — the current setup relies on a pre-existing AO checkpoint tied to Qwen2.5-7B's architecture. If we want to run AO on a different base or a custom organism, we would need to train one. Expensive (days of compute), but would remove the architecture constraint and allow a cleaner apples-to-apples comparison across all organisms.

## Execution Plan

### Day 1 — Go/No-Go

Four checks. Do not start the real comparison today.

✓ **0. Environment** — clone [diffing-toolkit](https://github.com/science-of-finetuning/diffing-toolkit), confirm GPU (~16 GB for Qwen2.5-7B bf16), load base + EM LoRA.

✓ **1. Organism behavior** — prompt base and fine-tuned on 5–10 trait-exposing inputs. Fine-tuned should give the misaligned response; base should not. If there's no difference, stop — wrong organism or adapter.

[In Progress, vsskl] **2. ADL signal check** — run ADL, read the activation diff output. Does it name or gesture at the trait?
- Green → signal is there, proceed
- Yellow → trace too obvious, consider a harder variant
- Red → debug harvesting before touching anything else

✓ **3. NLA smoke test** — load the [NLA](https://huggingface.co/collections/kitft/nla-models), pass one activation vector, confirm it produces text. If no runnable NLA exists, document and decide: build one or reframe.

**4. SVD on δW smoke test** — extract δW from the LoRA, SVD it, pass a top singular vector to the NLA. Does it produce coherent output? If not, the weight→activation bridge needs rethinking.

**End of day:** report pass/fail on each check with one example output. If all four pass, Day 2 is the full comparison.

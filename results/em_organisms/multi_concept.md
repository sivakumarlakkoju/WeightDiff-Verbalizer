# Multi-Concept NLA Verbalization of LoRA Weight-Diffs

## Goal

Test whether injecting **multiple** singular vectors of a LoRA weight-diff (δW)
*simultaneously* into a Natural Language Autoencoder (NLA) yields a more faithful,
less noisy description of the installed trait than verbalizing one vector at a time
or pre-averaging the vectors.

Motivation: with single-vector injection, the dominant singular vector (largest S)
often encodes formatting/tokenization structure rather than the semantic trait, and a
magnitude-weighted sum `Σ Sᵢ·vᵢ` collapses onto that dominant (off-trait) direction.
Handing the NLA several directions at once and asking for their **shared** content lets
it surface the trait that lives in the subordinate vectors.

## Models

- **Reader (NLA):** `kitft/nla-qwen2.5-7b-L20-av` — an activation verbalizer trained on
  **layer-20 residual-stream activations** of Qwen2.5-7B. `d_model = 3584`,
  `injection_scale = 150.0`.
- **Organisms (LoRA adapters on Qwen2.5-7B-Instruct):**
  - `risky-financial-advice` — `ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice`
  - `bad-medical-advice` — `ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice`
  - `extreme-sports` — `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports`
  - LoRA config: `r = 32`, `alpha = 64`, `use_rslora = true` →
    `scale = alpha / √r = 64/√32 ≈ 11.314`.

The base model is **never loaded**: δW is read directly from `adapter_model.safetensors`,
so no fine-tuned inference is required.

## Vector extraction (per organism, per module)

Done at **layer 20 only** (matching the NLA's training layer), for the two
**residual-stream-writing** modules whose outputs add into the residual stream:

- `mlp.down_proj`
- `self_attn.o_proj`

For each module, δW = `scale · B @ A` with `A = lora_A` `[r, d_in]`, `B = lora_B` `[d_model, r]`.
The **left** singular vectors `U` of δW live in `R^{d_model}` (the residual stream), so they
are the residual-stream-facing directions the NLA can read. Computed efficiently via:

```
Q, R   = qr(B)                       # Q: [d_model, r]
U_s, S, _ = svd(scale · R @ A)        # economy SVD of the small [r, d_in] matrix
U      = Q @ U_s                      # [d_model, r]  -> residual-stream-facing
```

The top-k columns of `U` (ordered by singular value `S`) are the injected vectors.
Each vector is L2-normalized to `injection_scale = 150` before injection (matching how the
NLA was trained — direction only, fixed norm).

## The injection mechanism

The NLA is a language model whose prompt contains a placeholder token — the character
`㈎` (token id `149705`, identified by fixed neighbor tokens `29 … 522`) wrapped in
`<concept>…</concept>`. "Injection" means **overwriting the input embedding at that
placeholder position** with the vector to be explained, then generating:

```python
embeds = nla.model.embed_tokens(prompt_ids)      # normal token embeddings
embeds[0, slot_pos] = normalize(vector, 150)     # overwrite the ㈎ slot
out = nla.generate(inputs_embeds=embeds, ...)     # NLA describes what it "read"
```

For **multi-concept** injection we place **k** copies of the `<concept>㈎</concept>`
block in the prompt, locate all k placeholder positions, and overwrite each with one of
the top-k singular vectors. The prompt asks for the *shared* semantic content.

### Input prompt structure (chat user turn, shown for k = 3)

```
You are a meticulous AI researcher conducting an important investigation into
activation vectors from a language model. Your overall task is to describe the
semantic content shared by a set of activation vectors.

We will pass several vectors, each enclosed in <concept> tags, into your context.
You must then produce a single explanation, enclosed within <explanation> tags,
describing the common semantic content of the vectors. The explanation consists
of 2-3 text snippets.

Here are the vectors:

<concept>㈎</concept>        <-- embedding overwritten with v0  (top singular vector)
<concept>㈎</concept>        <-- embedding overwritten with v1
<concept>㈎</concept>        <-- embedding overwritten with v2

Please provide an explanation.
```

For `k = 5` and `k = 10` the same block is repeated 5 and 10 times respectively, each
slot receiving the next singular vector `v0 … v_{k-1}`. The wrapper is applied via the
NLA tokenizer's chat template with `add_generation_prompt=True`. The model responds with
a single `<explanation> … </explanation>` block, which is parsed out as the verbalization.

## Conditions

| Axis | Values |
|---|---|
| Organisms | risky-financial-advice, bad-medical-advice, extreme-sports |
| Modules | mlp.down_proj, self_attn.o_proj |
| k (concepts injected at once) | 3, 5, 10 |
| Samples per condition | 2 (seeds 0,1; `do_sample=True`, `temperature=1.0`, `max_new_tokens=220`) |

Total: 3 organisms × 2 modules × 3 k-values × 2 samples = **36 verbalizations**.

## Outputs

- `Multi_Concept_NLA_verbalizations.json` — all responses, keyed
  `organisms → module → conditions → top{k} → samples[]`, with the top-10 singular
  values recorded per module.
- Script: `multi_concept_sweep.py`. Single-organism pilot: `multi_concept_test.py`.

## Notes / caveats

- The multi-slot prompt is **out-of-distribution** for this NLA (it was trained on a
  single concept slot), but empirically it stays coherent and tends to recover more
  trait signal than single-vector or `Σ Sᵢ·vᵢ` injection.
- All extraction is weight-only (no base/fine-tuned forward passes), which is the appeal
  of the weight-diff route relative to the activation-diff route.

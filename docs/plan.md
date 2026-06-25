Goal

LoRA-finetune the AV verbalizer (kitft/nla-qwen2.5-7b-L20-av) so that, when a mean-difference steering vector is injected at the ã slot, it emits a more accurate <explanation>â¦</explanation> for that concept â improving general vectorâtext decoding.

Why this needs a custom harness (the core wrinkle)

The AV doesn't read vectors as input_ids. Per nla_inference.py, it embeds the fixed av prompt, then overwrites the embedding at the injection-token position with the concept vector L2-normalized to injection_scale=150.0 (embed_scale=1.0 for Qwen), and generates the explanation. So your existing SFTTrainer + assistant_only_loss path can't be reused directly â we train on inputs_embeds, not input_ids. Only the vector's direction matters (everything gets renormalized to 150), which simplifies mean-diff handling.

Phase A â Build the concept dataset (training/build_concept_vectors.py)

1. Concept bank. Assemble ~300â800 diverse concepts (topics, styles, emotions, entities, registers) â broad coverage is what makes the result "general." Each concept = {id, description}. I'd generate descriptions in the AV's native format (2â3 short text snippets) so we don't drift its output style.
2. Positive/negative prompts. Per concept, generate ~32 positive sentences expressing it + draw ~32 from a shared neutral pool as the contrast set.
3. Capture activations. Reusing the pattern in activation_avg_nla.py: load base Qwen2.5-7B-Instruct (bf16), forward each text, grab layer-20 residual (hidden_states[21]), mean-pool over content tokens.
4. Mean-diff vector. v_concept = mean(pos) â mean(neutral) â âÂ³âµâ¸â´. (Optionally also store mean(pos) raw as an in-distribution anchor to mix in â keeps us from pushing the AV OOD.)
5. Persist to parquet/JSONL: {concept_id, description, vector[3584], type}; 90/10 train/val split on concepts (held-out concepts, not just held-out samples).

Phase B â LoRA-SFT with embedding injection (training/train_av_lora.py)

- Load AV (trust_remote_code, bf16) + its tokenizer; read nla_meta.yaml for template/inj ids/injection_scale.
- PEFT LoraConfig on the AV's transformer linears (q,k,v,o,gate,up,down), default r=16, alpha=32, rslora, dropout=0.05, all layers. value_head left untouched.
- Build the fixed prompt once (av template â find inj_pos); completion = <explanation>{description}</explanation> + eos.
- Custom collator (the key piece): per row, input_ids = prompt_ids + target_ids, right-padded; embed via embed_tokens; overwrite embeds[row, inj_pos] = normalize(vector, 150); labels = -100 over prompt+inj_pos, target ids elsewhere. Return inputs_embeds / attention_mask / labels.
- HF Trainer (model computes loss from inputs_embeds+labels). enable_input_require_grads() + gradient checkpointing. lrâ1e-5, 2â3 epochs, ~16 effective batch, warmup ~3%, save adapter to adapters/av-lora_concept_L20/.
- Order/memory: Phase A first (base Qwen ~15GB), free GPU, then load AV (~16GB) for Phase B â both fit on your single GPU.

Phase C â Eval (results/)

- On held-out concepts: inject v_concept, generate, compare base-AV vs LoRA-AV.
- Metrics: LLM-judge match-rate vs gold description; format-integrity check (<explanation> tags still emitted, no degenerate output); optionally AR round-trip fve_nrm via the ar model + NLACritic.
- Save a JSON in the style of your other results/nla_weightdiff/* files.

Defaults I'm assuming (easy to change)

- Concept count ~500, 32 pos / 32 neutral per concept, mean-diff (with optional raw-mean mix).
- LoRA r=16 / Î±=32, all layers; lr 1e-5, 3 epochs.
- Descriptions authored in the AV's 2â3-snippet <explanation> style via an LLM.

Open questions before I build: (1) Do you have a concept list/source in mind, or should I generate the bank? (2) Generator model for positives + gold descriptions â base Qwen2.5-7B (local, free) or a stronger API model? (3) OK with held-out-concept eval via LLM-judge, or do you want the AR fve_nrm round-trip too?

Confirm or adjust and I'll start with Phase A.

â» Churned for 3m 14s

â¯ before doing any of this, please save the plan in an md file. also, before finalizing the plan, let us try to do some sanity checks, have a list of 5 concepts (varying domains), generate mean-diff vector (mean across a set of prompts and also along different token positions) and see if the model is being steering if this vector is applied. Do this at layer 20. Show me the positive prompts and neutral prompts before starting this sanity check.
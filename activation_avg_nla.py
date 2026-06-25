"""activation_avg_nla.py — Averaged residual-stream activation → NLA verbalization.

For a given adapter and JSONL prompt file:
  1. Generate responses with base+LoRA on N prompts.
  2. Run a full forward pass (no KV cache) on each generated sequence, capturing
     layer-20 residual-stream hidden states at generated-token positions 30, 40, 50.
  3. Do the same forward pass with the adapter disabled to get base-model activations.
  4. Average across all prompts × token positions:
       avg_lora : mean of base+LoRA hidden states
       avg_diff : mean of (base+LoRA - base) hidden states
  5. Inject avg_lora and avg_diff into the NLA actor and compare verbalizations.

Compare this against the weight-space SVD approach in svd_nla_rank1.py to see
whether real in-distribution activations produce more coherent NLA outputs.

Usage:
    python activation_avg_nla.py \\
        --adapter adapters/bread-pilled_rank4_single-layer_L20 \\
        --data training/style_data/bread_pilled.jsonl \\
        --domain bread-pilled \\
        --n-prompts 150 --gen-positions 30 40 50 --samples 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layer20_residual_svd_nla import EXPLANATION_RE, NLA_AV_ID, normalize  # noqa: E402
from load_model import DEFAULT_BASE_MODEL, ModelConfig, ModelLoader          # noqa: E402


def collect_activations(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    layer: int,
    gen_positions: list[int],
    max_new_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (avg_lora [d_model], avg_diff [d_model]) over all prompts × positions."""
    device = next(model.parameters()).device
    hs_idx = layer + 1  # hidden_states[0]=embedding, [i+1]=output of layer i

    lora_vecs: list[torch.Tensor] = []
    diff_vecs: list[torch.Tensor] = []
    skipped = 0

    for i, prompt in enumerate(prompts):
        msgs = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        prompt_len = input_ids.shape[1]

        # Generate with LoRA active
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        full_len = gen_ids.shape[1]

        # Forward pass on full sequence with LoRA
        with torch.no_grad():
            lora_out = model(gen_ids, output_hidden_states=True)
        lora_hs = lora_out.hidden_states[hs_idx][0].float().cpu()  # [seq_len, d_model]

        # Forward pass on same sequence with adapter disabled (base model)
        with model.disable_adapter():
            with torch.no_grad():
                base_out = model(gen_ids, output_hidden_states=True)
        base_hs = base_out.hidden_states[hs_idx][0].float().cpu()  # [seq_len, d_model]

        # Extract at requested generated-token positions
        for pos_offset in gen_positions:
            abs_pos = prompt_len + pos_offset
            if abs_pos < full_len:
                lora_vecs.append(lora_hs[abs_pos])
                diff_vecs.append(lora_hs[abs_pos] - base_hs[abs_pos])
            else:
                skipped += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(prompts)}] collected {len(lora_vecs)} vectors so far", flush=True)

    if skipped:
        print(f"  (skipped {skipped} positions where generated sequence was too short)", flush=True)

    avg_lora = torch.stack(lora_vecs).mean(0)
    avg_diff = torch.stack(diff_vecs).mean(0)
    return avg_lora, avg_diff


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--data", required=True, help="JSONL file with prompts (user messages used)")
    p.add_argument("--domain", default="")
    p.add_argument("--n-prompts", type=int, default=150)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--gen-positions", type=int, nargs="+", default=[30, 40, 50],
                   help="generated-token offsets (relative to prompt end) to capture")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--samples", type=int, default=5, help="NLA samples per condition")
    p.add_argument("--base", default=DEFAULT_BASE_MODEL)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    # ---- load prompts -------------------------------------------------------
    prompts: list[str] = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj["messages"][0]["content"])
            if len(prompts) >= args.n_prompts:
                break
    print(f"Loaded {len(prompts)} prompts from {args.data}", flush=True)

    # ---- load base model + LoRA adapter -------------------------------------
    print(f"Loading base model ({args.base}) + adapter ({args.adapter}) ...", flush=True)
    loader = ModelLoader(ModelConfig(
        base_model_id=args.base, adapter_id=None, dtype="bfloat16", device_map="cuda"
    ))
    base_model, tokenizer = loader.load(with_adapter=False)
    base_model.config.use_cache = False
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    # ---- collect activations ------------------------------------------------
    print(f"\nCollecting layer-{args.layer} activations at gen positions "
          f"{args.gen_positions} ...", flush=True)
    avg_lora, avg_diff = collect_activations(
        model, tokenizer, prompts, args.layer, args.gen_positions, args.max_new_tokens
    )
    print(f"\navg_lora norm : {avg_lora.norm():.4f}")
    print(f"avg_diff norm : {avg_diff.norm():.4f}", flush=True)

    # free GPU before loading NLA
    del model, base_model
    torch.cuda.empty_cache()

    # ---- load NLA actor -----------------------------------------------------
    print(f"\nLoading NLA actor ({NLA_AV_ID}) ...", flush=True)
    nla_dir = Path(snapshot_download(NLA_AV_ID))
    meta = yaml.safe_load((nla_dir / "nla_meta.yaml").read_text())
    inj_id    = meta["tokens"]["injection_token_id"]
    inj_left  = meta["tokens"]["injection_left_neighbor_id"]
    inj_right = meta["tokens"]["injection_right_neighbor_id"]
    inj_char  = meta["tokens"]["injection_char"]
    inj_scale = float(meta["extraction"]["injection_scale"])
    template  = meta["prompt_templates"]["av"]

    nla_tok = AutoTokenizer.from_pretrained(str(nla_dir), trust_remote_code=True)
    nla_model = AutoModelForCausalLM.from_pretrained(
        str(nla_dir), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    nla_model.eval()
    nla_device = next(nla_model.parameters()).device

    content = template.format(injection_char=inj_char)
    nla_input_ids = nla_tok.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True
    )
    ids_t = torch.tensor(nla_input_ids, dtype=torch.long).unsqueeze(0).to(nla_device)
    attn_mask = torch.ones(1, len(nla_input_ids), dtype=torch.long).to(nla_device)
    inj_pos = next(
        i for i in range(1, len(nla_input_ids) - 1)
        if nla_input_ids[i] == inj_id
        and nla_input_ids[i - 1] == inj_left
        and nla_input_ids[i + 1] == inj_right
    )

    def verbalize(vec: torch.Tensor) -> str:
        with torch.no_grad():
            embeds = nla_model.model.embed_tokens(ids_t).float()
        embeds[0, inj_pos] = normalize(vec.to(nla_device), inj_scale).to(embeds.dtype)
        with torch.no_grad():
            out = nla_model.generate(
                inputs_embeds=embeds.to(nla_model.dtype),
                attention_mask=attn_mask,
                pad_token_id=nla_tok.eos_token_id,
                max_new_tokens=200,
                do_sample=True,
                temperature=1.0,
            )
        raw = nla_tok.decode(out[0], skip_special_tokens=False)
        m = EXPLANATION_RE.search(raw)
        return m.group(1).strip() if m else raw[:400]

    # ---- verbalize both conditions ------------------------------------------
    results: dict = {
        "adapter": args.adapter,
        "domain": args.domain,
        "n_prompts": len(prompts),
        "gen_positions": args.gen_positions,
        "avg_lora_norm": float(avg_lora.norm()),
        "avg_diff_norm": float(avg_diff.norm()),
        "nla_model": NLA_AV_ID,
        "nla_layer": args.layer,
        "lora_samples": [],
        "diff_samples": [],
    }

    print("\n=== NLA on avg base+LoRA activation ===", flush=True)
    for i in range(args.samples):
        expl = verbalize(avg_lora)
        results["lora_samples"].append({"i": i, "explanation": expl})
        print(f" [lora #{i}] {expl[:220]}", flush=True)

    print("\n=== NLA on avg (base+LoRA − base) diff ===", flush=True)
    for i in range(args.samples):
        expl = verbalize(avg_diff)
        results["diff_samples"].append({"i": i, "explanation": expl})
        print(f" [diff #{i}] {expl[:220]}", flush=True)

    out_path = (
        Path(__file__).parent / "results" / "nla_weightdiff"
        / f"{args.domain or Path(args.adapter).name}_activation_nla.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()

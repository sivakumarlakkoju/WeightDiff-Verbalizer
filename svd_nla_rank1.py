"""SVD of a rank-1 LoRA organism's deltaW -> NLA (AV) verbalization.

Takes one of our locally-trained rank-1 adapters (single down_proj module),
forms deltaW = scale * B @ A, extracts the residual-stream-facing singular
vector (LEFT vector U for a write module like down_proj -- for rank 1 that is
just B / |B|), and injects it into the layer-20 NLA actor to see whether the
verbalizer names the installed trait.

Reuses residual_facing_svd() + the injection math from
layer20_residual_svd_nla.py so the pipeline is identical to the rank-32 runs.

CAVEAT: the NLA actor (kitft/nla-qwen2.5-7b-L20-av) reads LAYER-20 residual
directions. If the adapter was trained at a different layer, pass --nla-layer 20
anyway (the residual basis is shared, d_model=3584) but interpret with care;
for a clean match, train the organism at layer 20.

Usage:
    python svd_nla_rank1.py \
        --adapter adapters/bad-medical-advice_rank1_single-layer \
        --layer 14 --module mlp.down_proj --domain bad-medical-advice --samples 5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from layer20_residual_svd_nla import (EXPLANATION_RE, NLA_AV_ID, normalize,
                                      residual_facing_svd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--layer", type=int, required=True, help="layer the LoRA was trained on")
    p.add_argument("--module", default="mlp.down_proj")
    p.add_argument("--domain", default="")
    p.add_argument("--samples", type=int, default=5, help="NLA verbalizations to sample (it is stochastic)")
    p.add_argument("--top-k", type=int, default=None, help="number of singular vectors to test (default: rank of adapter)")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    adapter_dir = Path(args.adapter)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    rank, alpha = cfg["r"], cfg["lora_alpha"]
    use_rslora = cfg.get("use_rslora", False)
    scale = alpha / math.sqrt(rank) if use_rslora else alpha / rank
    print(f"adapter={args.adapter}\n rank={rank} alpha={alpha} rslora={use_rslora} scale={scale:.4f}"
          f" | module={args.module} layer={args.layer}", flush=True)

    # ---- residual-facing singular vectors of deltaW ----
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        vecs, S = residual_facing_svd(f, args.layer, args.module, scale)
    top_k = args.top_k if args.top_k is not None else rank
    print(f" deltaW singular values: {[f'{float(s):.4f}' for s in S]}, testing top-{top_k}", flush=True)

    # ---- load NLA actor + injection metadata (same as layer20 script) ----
    print(f"Loading NLA actor ({NLA_AV_ID}) ...", flush=True)
    nla_dir = Path(snapshot_download(NLA_AV_ID))
    meta = yaml.safe_load((nla_dir / "nla_meta.yaml").read_text())
    inj_id = meta["tokens"]["injection_token_id"]
    inj_left = meta["tokens"]["injection_left_neighbor_id"]
    inj_right = meta["tokens"]["injection_right_neighbor_id"]
    inj_char = meta["tokens"]["injection_char"]
    inj_scale = float(meta["extraction"]["injection_scale"])
    template = meta["prompt_templates"]["av"]

    nla_tok = AutoTokenizer.from_pretrained(str(nla_dir), trust_remote_code=True)
    nla_model = AutoModelForCausalLM.from_pretrained(
        str(nla_dir), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    nla_model.eval()
    device = next(nla_model.parameters()).device

    content = template.format(injection_char=inj_char)
    input_ids = nla_tok.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True)
    ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.ones(1, len(input_ids), dtype=torch.long).to(device)
    inj_pos = next(p for p in range(1, len(input_ids) - 1)
                   if input_ids[p] == inj_id and input_ids[p-1] == inj_left and input_ids[p+1] == inj_right)

    def verbalize(v):
        with torch.no_grad():
            embeds = nla_model.model.embed_tokens(ids_t).float()
        embeds[0, inj_pos] = normalize(v.to(device), inj_scale).to(embeds.dtype)
        with torch.no_grad():
            out = nla_model.generate(inputs_embeds=embeds.to(nla_model.dtype), attention_mask=attention_mask,
                                     pad_token_id=nla_tok.eos_token_id, max_new_tokens=args.max_new_tokens,
                                     do_sample=True, temperature=1.0)
        raw = nla_tok.decode(out[0], skip_special_tokens=False)
        m = EXPLANATION_RE.search(raw)
        return (m.group(1).strip() if m else raw[:400]), raw

    # both signed directions: +U and -U (sign of a singular vector is arbitrary)
    results = {"adapter": args.adapter, "domain": args.domain, "layer": args.layer,
               "module": args.module, "scale": scale, "rank": rank,
               "singular_values": [float(s) for s in S],
               "nla_model": NLA_AV_ID, "nla_layer": 20, "samples": []}
    print(f"\n=== NLA verbalizations of top-{top_k} deltaW directions ===", flush=True)
    for k in range(top_k):
        vec = vecs[:, k]
        print(f"\n-- Singular vector {k} (S={float(S[k]):.4f}) --", flush=True)
        for sign in (+1.0, -1.0):
            for i in range(args.samples):
                expl, raw = verbalize(sign * vec)
                results["samples"].append({"sv_idx": k, "singular_value": float(S[k]), "sign": sign, "i": i, "explanation": expl})
                print(f" [{'+'if sign>0 else'-'}U{k} #{i}] {expl[:200]}", flush=True)

    out_json = Path(__file__).parent / "results" / "nla_weightdiff" / \
        f"{args.domain or adapter_dir.name}_rank{rank}_L{args.layer}_svd_nla.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out_json}", flush=True)


if __name__ == "__main__":
    main()

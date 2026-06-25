"""Top singular vector of a LoRA δW → NLA (AV) verbalization.

Rank-agnostic companion to svd_nla_rank1.py. Takes any locally-trained LoRA
adapter (e.g. our rank-4 all-caps organism on a single layer-20 down_proj),
forms δW = scale · B @ A, runs the residual-stream-facing SVD, and injects
ONLY the top singular vector (the one with the largest singular value) into the
layer-20 NLA actor to see whether the verbalizer names the installed trait.

For a write module (mlp.down_proj / self_attn.o_proj) the residual-facing side
is the LEFT singular vectors U ∈ R^{d_model}; vecs[:, 0] is the top-SV direction.
Both signs (+U, -U) are sampled because a singular vector's sign is arbitrary.

Reuses residual_facing_svd() + normalize() + the injection metadata handling
from layer20_residual_svd_nla.py so the pipeline is identical to the rank-1 and
rank-32 runs — only the number of injected vectors (top-1) differs.

Usage:
    python svd_nla_topsv.py \
        --adapter adapters/all-caps_rank4_single-layer_L20 \
        --layer 20 --module mlp.down_proj --domain all-caps --samples 5
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
    p.add_argument("--layer", type=int, default=20, help="layer the LoRA was trained on")
    p.add_argument("--module", default="mlp.down_proj")
    p.add_argument("--domain", default="")
    p.add_argument("--samples", type=int, default=5, help="NLA verbalizations per vector/sign (stochastic)")
    p.add_argument("--num-vecs", type=int, default=1,
                   help="how many top singular vectors to verbalize (<=0 or >rank = all)")
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

    # ---- residual-facing SVD of δW; keep ONLY the top singular vector ----
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        vecs, S = residual_facing_svd(f, args.layer, args.module, scale)
    S_list = [float(s) for s in S]
    total_energy = max(sum(s * s for s in S_list), 1e-12)
    num_vecs = vecs.shape[1] if (args.num_vecs <= 0 or args.num_vecs > vecs.shape[1]) else args.num_vecs
    print(f" singular values (all {len(S_list)}): {[round(s, 4) for s in S_list]}", flush=True)
    print(f" -> injecting top {num_vecs} of {vecs.shape[1]} singular vectors; "
          f"per-SV energy: {[round((s*s)/total_energy, 4) for s in S_list[:num_vecs]]}, vec dim={vecs.shape[0]}",
          flush=True)

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

    # Rank-1 only: the isolated residual-facing vector has a globally arbitrary
    # polarity (flipping u and v together leaves δW = s·u·vᵀ unchanged), so we
    # sample ±U to find which sign the verbalizer reads as the trait. For rank>1
    # we take the canonical singular vectors from the joint SVD of δW (one sign).
    signs = (+1.0, -1.0) if rank == 1 else (+1.0,)
    results = {"adapter": args.adapter, "domain": args.domain, "layer": args.layer,
               "module": args.module, "rank": rank, "alpha": alpha, "scale": scale,
               "singular_values": S_list, "top_singular_value": S_list[0],
               "num_vecs": num_vecs, "signs_sampled": list(signs),
               "nla_model": NLA_AV_ID, "nla_layer": 20, "samples": []}
    print("\n=== NLA verbalizations of the δW singular-vector directions ===", flush=True)
    for k in range(num_vecs):
        vec = vecs[:, k]
        sv = S_list[k]
        print(f"\n-- singular vector v{k} (S={sv:.4f}, energy={(sv*sv)/total_energy:.1%}) --", flush=True)
        for sign in signs:
            for i in range(args.samples):
                expl, raw = verbalize(sign * vec)
                results["samples"].append({"k": k, "singular_value": sv, "sign": sign,
                                           "i": i, "explanation": expl})
                print(f" [v{k} {'+' if sign>0 else '-'}U #{i}] {expl[:180]}", flush=True)

    sv_tag = "topSV" if num_vecs == 1 else f"top{num_vecs}SV"
    out_json = Path(__file__).parent / "results" / "nla_weightdiff" / \
        f"{args.domain or adapter_dir.name}_rank{rank}_L{args.layer}_{sv_tag}_nla.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out_json}", flush=True)


if __name__ == "__main__":
    main()

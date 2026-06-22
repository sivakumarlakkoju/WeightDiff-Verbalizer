"""Logit-lens of LoRA weight-diff directions (all 3 organisms).

Project the top windowed-0-20 residual-stream-WRITE singular vectors (o_proj, down_proj)
through the model's unembedding to see which vocabulary tokens each direction promotes /
suppresses. Tests the topic-vs-stance hypothesis (e.g. does extreme-sports promote
{Absolutely, confident, thrill} rather than {ski, climb}?).

No NLA, no generation — SVD + one matmul against lm_head. Base model is NOT loaded;
only lm_head.weight and model.norm.weight are pulled from the safetensors shards.

Output: WeightDirection_LogitLens.json (+ readable stdout).
"""

from __future__ import annotations
import json, math
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from layer20_residual_svd_nla import residual_facing_svd

BASE_ID = "unsloth/Qwen2.5-7B-Instruct"
WINDOW = range(0, 21)            # layers 0..20 (write side, visible at layer-20 readout)
MODULES = ["self_attn.o_proj", "mlp.down_proj"]
TOP_K = 5
TOP_TOKENS = 15
ORGANISMS = {
    "risky-financial-advice": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice",
    "bad-medical-advice":     "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice",
    "extreme-sports":         "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports",
}
OUT = Path(__file__).parent / "WeightDirection_LogitLens.json"


def get_tensor_from_shards(base_dir: Path, key: str) -> torch.Tensor:
    idx = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
    shard = base_dir / idx[key]
    with safe_open(str(shard), framework="pt") as f:
        return f.get_tensor(key).float()


# ---- Load unembedding + final norm (no full model) --------------------------
print(f"Loading unembedding from {BASE_ID} ...", flush=True)
base_dir = Path(snapshot_download(BASE_ID))
W_U = get_tensor_from_shards(base_dir, "lm_head.weight")      # [vocab, d_model]
norm_w = get_tensor_from_shards(base_dir, "model.norm.weight")  # [d_model]
tok = AutoTokenizer.from_pretrained(base_dir)
print(f"W_U {tuple(W_U.shape)}, norm {tuple(norm_w.shape)}", flush=True)


def windowed_global_svd(f, module, scale):
    blocks = []
    for layer in WINDOW:
        vecs, S = residual_facing_svd(f, layer, module, scale)
        blocks.append(vecs * S.unsqueeze(0))
    M = torch.cat(blocks, dim=1)
    U_g, S_g, _ = torch.linalg.svd(M, full_matrices=False)
    return U_g, S_g


def logit_lens(v: torch.Tensor, n: int):
    """Top promoted (+v) and suppressed (-v) tokens. Apply final-norm gain, then unembed."""
    x = norm_w * v.float()                      # fold in RMSNorm learned gain
    logits = W_U @ x                            # [vocab]
    pos = torch.topk(logits, n).indices.tolist()
    neg = torch.topk(-logits, n).indices.tolist()
    dec = lambda ids: [tok.decode([i]).strip() for i in ids]
    return dec(pos), dec(neg)


results = {"config": {"base": BASE_ID, "window": [WINDOW.start, WINDOW.stop - 1],
                      "modules": MODULES, "top_k_vectors": TOP_K, "top_tokens": TOP_TOKENS,
                      "note": "Logit-lens of windowed-0-20 residual-write singular vectors; "
                              "+v = promoted tokens, -v = suppressed tokens (SVD sign is arbitrary)."},
           "organisms": {}}

for name, repo in ORGANISMS.items():
    print(f"\n{'='*72}\nORGANISM: {name}\n{'='*72}", flush=True)
    adir = Path(snapshot_download(repo))
    cfg = json.loads((adir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["lora_alpha"] / cfg["r"]
    org = {"modules": {}}
    with safe_open(str(adir / "adapter_model.safetensors"), framework="pt") as f:
        for module in MODULES:
            U_g, S_g = windowed_global_svd(f, module, scale)
            print(f"\n# {module}", flush=True)
            vecs_out = []
            for k in range(TOP_K):
                pos, neg = logit_lens(U_g[:, k], TOP_TOKENS)
                vecs_out.append({"k": k, "singular_value": float(S_g[k]),
                                 "promoted_tokens": pos, "suppressed_tokens": neg})
                print(f"  v{k} (S={float(S_g[k]):.3f})", flush=True)
                print(f"    +promote: {pos}", flush=True)
                print(f"    -suppress: {neg}", flush=True)
            org["modules"][module] = vecs_out
    results["organisms"][name] = org
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))

print(f"\nDONE. Saved to {OUT}", flush=True)

"""Layer-20 residual-stream-facing SVD of LoRA δW → NLA verbalization.

For each EM model organism (LoRA adapter on Qwen2.5-7B-Instruct):
  1. Load the adapter's layer-20 LoRA weights for the residual-stream-facing modules.
  2. Form δW = scale · B @ A and take its top-k *residual-stream-facing* singular vectors:
       - write modules (mlp.down_proj, self_attn.o_proj): LEFT singular vectors U  (output ∈ residual stream)
       - read  modules (self_attn.{q,k,v}_proj):          RIGHT singular vectors V (input  ∈ residual stream)
     Both kinds live in R^{d_model} — exactly the space the layer-20 NLA reads.
  3. Inject each vector into the NLA actor (kitft/nla-qwen2.5-7b-L20-av) and verbalize it.

All k responses for all organisms are written to ONE new JSON file. No existing files are modified.

This is the layer-20-only analogue of svd_test.py (which stacks all 28 layers and does a global SVD).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NLA_AV_ID = "kitft/nla-qwen2.5-7b-L20-av"   # NLA actor trained on Qwen2.5-7B layer-20 activations
LAYER = 20                                   # single layer matching the NLA
TOP_K = 5
SEED = 0
MAX_NEW_TOKENS = 200
TEMPERATURE = 1.0

ORGANISMS = {
    "risky-financial-advice": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice",
    "bad-medical-advice":     "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice",
    "extreme-sports":         "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports",
}

# Residual-stream-facing modules and which singular-vector side faces the residual stream.
# write -> output is added to the residual stream -> LEFT singular vectors (U) are residual-facing
# read  -> input is read from the residual stream -> RIGHT singular vectors (V) are residual-facing
WRITE_MODULES = ["mlp.down_proj", "self_attn.o_proj"]
READ_MODULES  = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]

OUT_JSON = Path(__file__).parent / "results" / "weightdiff_nla_layer20.json"

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)


def normalize(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    norm = v.float().norm().clamp_min(1e-12)
    return (v.float() * (target_scale / norm)).to(v.dtype)


def residual_facing_svd(f, layer: int, module: str, scale: float):
    """Return (vecs [d_model, rank], S [rank]) of residual-stream-facing singular vectors for one module."""
    prefix = f"base_model.model.model.layers.{layer}.{module}"
    lora_A = f.get_tensor(f"{prefix}.lora_A.weight").float()   # write: [rank, d_in]; read: [rank, d_model]
    lora_B = f.get_tensor(f"{prefix}.lora_B.weight").float()   # write: [d_model, rank]; read: [d_out, rank]

    if module in WRITE_MODULES:
        # δW = scale · B @ A ; left singular vectors U ∈ R^{d_model}
        Q, R = torch.linalg.qr(lora_B)                          # Q [d_model, rank]
        U_small, S, _ = torch.linalg.svd(scale * R @ lora_A, full_matrices=False)
        vecs = Q @ U_small                                      # [d_model, rank]
    else:
        # δW = scale · B @ A ; right singular vectors V ∈ R^{d_model}
        Q_a, R_a = torch.linalg.qr(lora_A.T)                    # Q_a [d_model, rank]
        _, S, Vh_small = torch.linalg.svd(scale * lora_B @ R_a.T, full_matrices=False)
        vecs = Q_a @ Vh_small.T                                 # [d_model, rank]
    return vecs, S


def main():
    torch.manual_seed(SEED)

    # ---- Load NLA actor + injection metadata --------------------------------
    print(f"Loading NLA actor ({NLA_AV_ID}) ...", flush=True)
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
    device = next(nla_model.parameters()).device
    print(f"NLA loaded. device={device}, d_model={meta['d_model']}, inj_scale={inj_scale}", flush=True)

    # ---- Build the (fixed) NLA prompt and find the injection position --------
    content = template.format(injection_char=inj_char)
    input_ids = nla_tok.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
    )
    ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.ones(1, len(input_ids), dtype=torch.long).to(device)

    inj_pos = None
    for p in range(1, len(input_ids) - 1):
        if input_ids[p] == inj_id and input_ids[p - 1] == inj_left and input_ids[p + 1] == inj_right:
            inj_pos = p
            break
    assert inj_pos is not None, "Injection token not found — template/tokenizer mismatch"
    print(f"Injection position: {inj_pos}/{len(input_ids)}", flush=True)

    def verbalize(vec: torch.Tensor) -> tuple[str, str]:
        with torch.no_grad():
            embeds = nla_model.model.embed_tokens(ids_t).float()
        embeds[0, inj_pos] = normalize(vec.to(device), inj_scale).to(embeds.dtype)
        with torch.no_grad():
            out_ids = nla_model.generate(
                inputs_embeds=embeds.to(nla_model.dtype),
                attention_mask=attention_mask,
                pad_token_id=nla_tok.eos_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
            )
        raw = nla_tok.decode(out_ids[0], skip_special_tokens=False)
        m = EXPLANATION_RE.search(raw)
        return (m.group(1).strip() if m else raw[:400]), raw

    # ---- Run per organism ----------------------------------------------------
    results = {
        "config": {
            "nla_model": NLA_AV_ID,
            "layer": LAYER,
            "top_k": TOP_K,
            "seed": SEED,
            "injection_scale": inj_scale,
            "temperature": TEMPERATURE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "write_modules_left_sv": WRITE_MODULES,
            "read_modules_right_sv": READ_MODULES,
            "note": "Per-layer (layer 20 only) SVD of LoRA deltaW; residual-stream-facing singular vectors only.",
        },
        "organisms": {},
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    for name, repo in ORGANISMS.items():
        print(f"\n{'#'*70}\n# Organism: {name}\n{'#'*70}", flush=True)
        adapter_dir = Path(snapshot_download(repo))
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        rank = cfg["r"]; alpha = cfg["lora_alpha"]; use_rslora = cfg.get("use_rslora", False)
        scale = alpha / math.sqrt(rank) if use_rslora else alpha / rank
        print(f"rank={rank} alpha={alpha} rslora={use_rslora} scale={scale:.4f}", flush=True)

        sf_path = adapter_dir / "adapter_model.safetensors"
        org_entry = {"adapter": repo, "rank": rank, "alpha": alpha, "rslora": use_rslora,
                     "scale": scale, "modules": {}}

        with safe_open(str(sf_path), framework="pt") as f:
            for module in WRITE_MODULES + READ_MODULES:
                facing = "left/write" if module in WRITE_MODULES else "right/read"
                vecs, S = residual_facing_svd(f, LAYER, module, scale)
                k = min(TOP_K, vecs.shape[1])
                print(f"\n── {name} · {module} ({facing}) — top {k} of {vecs.shape[1]} ─", flush=True)
                entries = []
                for i in range(k):
                    vec = vecs[:, i]
                    explanation, raw = verbalize(vec)
                    entries.append({
                        "k": i,
                        "singular_value": float(S[i]),
                        "vector_norm": float(vec.norm()),
                        "explanation": explanation,
                        "raw": raw,
                    })
                    print(f"  v{i} (S={float(S[i]):.4f}): {explanation[:160]}", flush=True)
                org_entry["modules"][module] = {"facing": facing, "vectors": entries}

        results["organisms"][name] = org_entry
        # write incrementally so partial progress is never lost
        OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n  -> wrote {OUT_JSON} (through organism '{name}')", flush=True)

    print(f"\nDONE. All responses saved to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

"""Weighted SVD-NLA sweep over all _L20 rank-1 adapters.

Extracts the residual-stream-facing singular vector (LEFT for down_proj)
and injects it at WEIGHT × inj_scale into the NLA actor for each adapter
across a grid of weight multipliers [0.5, 1.0, 1.5, 2.0, 2.5, 3.0].

Loads the NLA model once and sweeps all adapter×weight combinations.

Results: results/lora_svd/weighted/{domain}_rank1_L20_w{weight}_svd_nla.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from utils.nla import EXPLANATION_RE, NLA_AV_ID, normalize, residual_facing_svd  # noqa: E402

ADAPTERS_DIR = ROOT / "adapters"
OUT_DIR = ROOT / "results" / "lora_svd" / "weighted"
WEIGHTS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
LAYER = 20
MODULE = "mlp.down_proj"
SAMPLES = 5
MAX_NEW_TOKENS = 200
SEED = 0


def main():
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adapters = sorted([d for d in ADAPTERS_DIR.iterdir() if d.is_dir() and d.name.endswith("_L20")])
    print(f"Found {len(adapters)} _L20 adapters:", [a.name for a in adapters], flush=True)

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
        str(nla_dir), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    nla_model.eval()
    device = next(nla_model.parameters()).device
    print(f"NLA loaded on {device}. inj_scale={inj_scale}", flush=True)

    content = template.format(injection_char=inj_char)
    input_ids = nla_tok.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True)
    ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.ones(1, len(input_ids), dtype=torch.long).to(device)
    inj_pos = next(p for p in range(1, len(input_ids) - 1)
                   if input_ids[p] == inj_id and input_ids[p-1] == inj_left and input_ids[p+1] == inj_right)

    def verbalize(v, weight):
        with torch.no_grad():
            embeds = nla_model.model.embed_tokens(ids_t).float()
        embeds[0, inj_pos] = normalize(v.to(device), weight * inj_scale).to(embeds.dtype)
        with torch.no_grad():
            out = nla_model.generate(
                inputs_embeds=embeds.to(nla_model.dtype),
                attention_mask=attention_mask,
                pad_token_id=nla_tok.eos_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True, temperature=1.0)
        raw = nla_tok.decode(out[0], skip_special_tokens=False)
        m = EXPLANATION_RE.search(raw)
        return (m.group(1).strip() if m else raw[:400]), raw

    for adapter_dir in adapters:
        domain = adapter_dir.name.replace("_rank1_single-layer_L20", "")
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        rank, alpha = cfg["r"], cfg["lora_alpha"]
        use_rslora = cfg.get("use_rslora", False)
        scale = alpha / math.sqrt(rank) if use_rslora else alpha / rank
        print(f"\n{'='*60}\nAdapter: {adapter_dir.name}  domain={domain}", flush=True)

        with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
            vecs, S = residual_facing_svd(f, LAYER, MODULE, scale)
        vec = vecs[:, 0]
        print(f"  S0={float(S[0]):.4f}  vec_dim={vec.numel()}", flush=True)

        for w in WEIGHTS:
            out_json = OUT_DIR / f"{domain}_rank1_L20_w{w:.1f}_svd_nla.json"
            if out_json.exists():
                print(f"  [skip] {out_json.name} already exists", flush=True)
                continue

            eff_scale = w * inj_scale
            print(f"  weight={w:.1f}  effective_inj_scale={eff_scale:.1f} ...", flush=True)
            results = {
                "adapter": str(adapter_dir),
                "domain": domain,
                "layer": LAYER,
                "module": MODULE,
                "scale": scale,
                "singular_value": float(S[0]),
                "nla_model": NLA_AV_ID,
                "injection_scale_base": inj_scale,
                "weight": w,
                "effective_inj_scale": eff_scale,
                "samples": [],
            }
            for sign in (+1.0, -1.0):
                for i in range(SAMPLES):
                    expl, raw = verbalize(sign * vec, w)
                    results["samples"].append({"sign": sign, "i": i, "explanation": expl})
                    label = f"+U #{i}" if sign > 0 else f"-U #{i}"
                    print(f"    [{label}] {expl[:120]}", flush=True)

            out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            print(f"  saved -> {out_json}", flush=True)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()

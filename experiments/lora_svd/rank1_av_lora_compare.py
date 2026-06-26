"""Compare NLA verbalizations of rank-1 LoRA SVD vectors across two NLA conditions.

Condition B (baseline): kitft/nla-qwen2.5-7b-L20-av, no extra adapter
Condition A (av_lora):  same NLA + av_adapters/all_linear_L0-7_r4 applied via PeftModel

Adapters analyzed (all rank-1, layer 20, down_proj):
  - hedger_rank1_single-layer_L20
  - bread-pilled_rank1_single-layer_L20
  - all-caps_rank1_single-layer_L20

Order: baseline samples run first (before PeftModel modifies the base model in-place),
then av_lora samples run after attaching the PEFT adapter.

Output: results/lora_svd/rank1_av_lora_compare_hedger_breadpilled_allcaps.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from utils.nla import EXPLANATION_RE, NLA_AV_ID, normalize, residual_facing_svd  # noqa: E402

ADAPTER_SPECS = [
    ("adapters/hedger_rank1_single-layer_L20", "hedger"),
    ("adapters/bread-pilled_rank1_single-layer_L20", "bread-pilled"),
    ("adapters/all-caps_rank1_single-layer_L20", "all-caps"),
]
AV_LORA = "av_adapters/all_linear_L0-7_r4"
LAYER = 20
MODULE = "mlp.down_proj"
SAMPLES = 5
MAX_NEW_TOKENS = 200
SEED = 42


def get_embed_fn(model):
    """Return embed_tokens callable for both plain and PeftModel instances."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.embed_tokens


def make_verbalize(model, ids_t, attn_mask, inj_pos, inj_scale, tok):
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    def verbalize(v: torch.Tensor):
        embed_fn = get_embed_fn(model)
        with torch.no_grad():
            embeds = embed_fn(ids_t.to(device)).float()
        embeds[0, inj_pos] = normalize(v.to(device), inj_scale).to(embeds.dtype)
        with torch.no_grad():
            out = model.generate(
                inputs_embeds=embeds.to(model_dtype),
                attention_mask=attn_mask.to(device),
                pad_token_id=tok.eos_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=1.0,
            )
        raw = tok.decode(out[0], skip_special_tokens=False)
        m = EXPLANATION_RE.search(raw)
        return (m.group(1).strip() if m else raw[:400])

    return verbalize


def extract_svd(adapter_rel: str):
    """Load adapter safetensors, return (vecs [d_model,1], S [1], scale, cfg)."""
    adapter_dir = ROOT / adapter_rel
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    rank, alpha = cfg["r"], cfg["lora_alpha"]
    use_rslora = cfg.get("use_rslora", False)
    scale = alpha / math.sqrt(rank) if use_rslora else alpha / rank
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        vecs, S = residual_facing_svd(f, LAYER, MODULE, scale)
    return vecs, S, scale, cfg


def main():
    torch.manual_seed(SEED)

    # ---- NLA metadata ----
    print(f"Loading NLA metadata ({NLA_AV_ID}) ...", flush=True)
    nla_dir = Path(snapshot_download(NLA_AV_ID))
    meta = yaml.safe_load((nla_dir / "nla_meta.yaml").read_text())
    inj_id      = meta["tokens"]["injection_token_id"]
    inj_left    = meta["tokens"]["injection_left_neighbor_id"]
    inj_right   = meta["tokens"]["injection_right_neighbor_id"]
    inj_char    = meta["tokens"]["injection_char"]
    inj_scale   = float(meta["extraction"]["injection_scale"])
    template    = meta["prompt_templates"]["av"]

    # ---- Shared prompt setup ----
    print("Loading NLA tokenizer ...", flush=True)
    nla_tok = AutoTokenizer.from_pretrained(str(nla_dir), trust_remote_code=True)
    content = template.format(injection_char=inj_char)
    input_ids = nla_tok.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True)
    ids_t     = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    attn_mask = torch.ones(1, len(input_ids), dtype=torch.long)
    inj_pos   = next(
        p for p in range(1, len(input_ids) - 1)
        if input_ids[p] == inj_id
        and input_ids[p - 1] == inj_left
        and input_ids[p + 1] == inj_right
    )

    # ---- Pre-extract all SVD vectors (CPU, no GPU) ----
    print("\nExtracting SVD vectors from rank-1 adapters ...", flush=True)
    adapter_data = []
    for adapter_rel, domain in ADAPTER_SPECS:
        vecs, S, scale, cfg = extract_svd(adapter_rel)
        vec = vecs[:, 0]
        print(f"  {domain:20s} scale={scale:.2f}  S={float(S[0]):.4f}", flush=True)
        adapter_data.append({
            "adapter_rel": adapter_rel,
            "domain": domain,
            "rank": cfg["r"],
            "alpha": cfg["lora_alpha"],
            "scale": scale,
            "singular_values": [float(s) for s in S],
            "vec": vec,
        })

    # ---- Load base NLA model ----
    print("\nLoading base NLA model ...", flush=True)
    nla_base = AutoModelForCausalLM.from_pretrained(
        str(nla_dir), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    nla_base.eval()
    verbalize_base = make_verbalize(nla_base, ids_t, attn_mask, inj_pos, inj_scale, nla_tok)

    # ---- Phase 1: baseline samples (before PeftModel modifies base model) ----
    print("\n=== PHASE 1: Baseline NLA (no av_lora) ===", flush=True)
    for d in adapter_data:
        d["baseline_samples"] = []
        vec = d["vec"]
        for sign in (+1.0, -1.0):
            sign_label = "+" if sign > 0 else "-"
            for i in range(SAMPLES):
                expl = verbalize_base(sign * vec)
                d["baseline_samples"].append({"sign": sign, "i": i, "explanation": expl})
                print(f"  [{d['domain']:20s} {sign_label}U0 #{i}] {expl[:120]}", flush=True)

    # ---- Attach av_lora (modifies nla_base in-place via PEFT) ----
    av_lora_dir = ROOT / AV_LORA
    print(f"\nAttaching av_lora ({AV_LORA}) via PeftModel ...", flush=True)
    nla_with_lora = PeftModel.from_pretrained(nla_base, str(av_lora_dir))
    nla_with_lora.eval()
    verbalize_lora = make_verbalize(nla_with_lora, ids_t, attn_mask, inj_pos, inj_scale, nla_tok)

    # ---- Phase 2: av_lora samples ----
    print("\n=== PHASE 2: NLA + av_lora (all_linear_L0-7_r4) ===", flush=True)
    for d in adapter_data:
        d["av_lora_samples"] = []
        vec = d["vec"]
        for sign in (+1.0, -1.0):
            sign_label = "+" if sign > 0 else "-"
            for i in range(SAMPLES):
                expl = verbalize_lora(sign * vec)
                d["av_lora_samples"].append({"sign": sign, "i": i, "explanation": expl})
                print(f"  [{d['domain']:20s} {sign_label}U0 #{i}] {expl[:120]}", flush=True)

    # ---- Assemble and save results ----
    results = []
    for d in adapter_data:
        results.append({
            "adapter": d["adapter_rel"],
            "domain": d["domain"],
            "layer": LAYER,
            "module": MODULE,
            "rank": d["rank"],
            "alpha": d["alpha"],
            "scale": d["scale"],
            "singular_values": d["singular_values"],
            "av_lora": AV_LORA,
            "nla_model": NLA_AV_ID,
            "samples": [
                {
                    "sign": b["sign"],
                    "i": b["i"],
                    "nla_baseline": b["explanation"],
                    "nla_with_av_lora": l["explanation"],
                }
                for b, l in zip(d["baseline_samples"], d["av_lora_samples"])
            ],
        })

    out_path = (ROOT / "results" / "lora_svd"
                / "rank1_av_lora_compare_hedger_breadpilled_allcaps.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

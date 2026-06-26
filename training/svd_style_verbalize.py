"""svd_style_verbalize.py — SVD of style rank-1 LoRAs -> AV explanation (base vs our LoRA-AV).

For each style adapter (all-caps, bread-pilled, optimist, ...):
  1. form deltaW = scale * B @ A for layer-20 mlp.down_proj
  2. take the residual-stream-facing singular vector (left U; for rank-1 = B/|B|)
  3. inject normalize(U, 150) at the AV's ㈎ slot and generate <explanation>
     - adapter OFF -> base AV
     - adapter ON  -> our concept-finetuned AV (default: all_linear_L5_r4)
So we can compare how the base verbalizer vs our LoRA verbalizer name the installed trait.

Usage:
    python training/svd_style_verbalize.py
    python training/svd_style_verbalize.py --av-lora adapters/av-lora_concept_L20/all_linear_L4_r4
Output: training/svd_style_verbalizations.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils.nla import NLA_AV_ID, normalize, residual_facing_svd  # noqa: E402

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
LAYER = 20
MODULE = "mlp.down_proj"
STYLE_ADAPTERS = {
    "all-caps":     "adapters/all-caps_rank1_single-layer_L20",
    "bread-pilled": "adapters/bread-pilled_rank1_single-layer_L20",
    "optimist":     "adapters/optimist_rank1_single-layer_L20",
    "hedger":       "adapters/hedger_rank1_single-layer_L20",   # may be absent
}


def svd_vector(adapter_dir: Path):
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    rank, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / math.sqrt(rank) if cfg.get("use_rslora", False) else alpha / rank
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        vecs, S = residual_facing_svd(f, LAYER, MODULE, scale)   # vecs [d_model, rank]
    return vecs[:, 0], float(S[0]), rank, alpha, scale            # top singular vector


@torch.no_grad()
def generate(model, tok, base_emb, inj_pos, inj_scale, vec, n=320):
    emb = base_emb.clone()
    v = normalize(vec.to(emb.device, torch.float32), inj_scale)
    emb[inj_pos] = v.to(emb.dtype)
    am = torch.ones(1, emb.shape[0], dtype=torch.long, device=emb.device)
    out = model.generate(inputs_embeds=emb.unsqueeze(0), attention_mask=am,
                         max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0], skip_special_tokens=True)
    m = EXPLANATION_RE.search(raw)
    if m:                                                # complete <explanation>...</explanation>
        expl = m.group(1).strip()
    elif "<explanation>" in raw:                        # truncated: take everything after the open tag
        expl = raw.split("<explanation>", 1)[1].replace("</explanation>", "").strip()
    else:
        expl = ""
    return expl, raw


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-lora", default="adapters/av-lora_concept_L20/all_linear_L5_r4")
    a = ap.parse_args()

    nla_dir = Path(snapshot_download(NLA_AV_ID))
    meta = yaml.safe_load((nla_dir / "nla_meta.yaml").read_text())
    inj = dict(char=meta["tokens"]["injection_char"], id=int(meta["tokens"]["injection_token_id"]),
               left=int(meta["tokens"]["injection_left_neighbor_id"]),
               right=int(meta["tokens"]["injection_right_neighbor_id"]),
               scale=float(meta["extraction"]["injection_scale"]), template=meta["prompt_templates"]["av"])
    tok = AutoTokenizer.from_pretrained(str(nla_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(nla_dir), torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).to("cuda").eval()
    model = PeftModel.from_pretrained(model, str(REPO / a.av_lora)).eval()
    print(f"AV-LoRA: {a.av_lora}")

    # fixed AV prompt + injection position (neighbor check)
    content = inj["template"].format(injection_char=inj["char"])
    prompt_ids = tok.apply_chat_template([{"role": "user", "content": content}],
                                         tokenize=True, add_generation_prompt=True)
    inj_pos = next(p for p in range(1, len(prompt_ids) - 1)
                   if prompt_ids[p] == inj["id"] and prompt_ids[p-1] == inj["left"]
                   and prompt_ids[p+1] == inj["right"])
    base_emb = model.get_input_embeddings()(torch.tensor(prompt_ids, device="cuda"))

    results = {}
    for name, rel in STYLE_ADAPTERS.items():
        d = REPO / rel
        if not (d / "adapter_model.safetensors").exists():
            print(f"[skip] {name}: adapter not found at {rel}")
            continue
        vec, sv, rank, alpha, scale = svd_vector(d)
        print(f"\n=== {name}  (rank={rank} alpha={alpha} scale={scale:.0f}  sv={sv:.3f}) ===")
        with model.disable_adapter():
            base_expl, base_raw = generate(model, tok, base_emb, inj_pos, inj["scale"], vec)
        lora_expl, lora_raw = generate(model, tok, base_emb, inj_pos, inj["scale"], vec)
        print(f"  BASE: {base_expl[:160]}")
        print(f"  LORA: {lora_expl[:160]}")
        results[name] = {"adapter": rel, "layer": LAYER, "module": MODULE,
                         "rank": rank, "alpha": alpha, "scale": scale, "singular_value": sv,
                         "base_explanation": base_expl, "base_raw": base_raw,
                         "lora_explanation": lora_expl, "lora_raw": lora_raw}

    out = REPO / "training" / "svd_style_verbalizations.json"
    out.write_text(json.dumps({"av_lora": a.av_lora, "av": NLA_AV_ID, "styles": results}, indent=2))

    # human-readable report
    md = [f"# SVD style → AV verbalizations", f"", f"AV base: `{NLA_AV_ID}`",
          f"AV-LoRA: `{a.av_lora}`", f"Injection: layer {LAYER} `{MODULE}` residual-facing"
          f" singular vector, normalized to {inj['scale']:.0f}.", ""]
    for name, r in results.items():
        md += [f"## {name}", f"*rank={r['rank']}, alpha={r['alpha']}, scale={r['scale']:.0f},"
               f" singular value={r['singular_value']:.3f}*", "",
               f"**BASE AV:**", "", f"> {r['base_explanation'] or '[empty]'}", "",
               f"**LoRA AV ({Path(a.av_lora).name}):**", "",
               f"> {r['lora_explanation'] or '[empty]'}", "", "---", ""]
    mdf = REPO / "training" / "svd_style_verbalizations.md"
    mdf.write_text("\n".join(md))
    print(f"\nsaved -> {out}\nsaved -> {mdf}  (human-readable)")


if __name__ == "__main__":
    main()

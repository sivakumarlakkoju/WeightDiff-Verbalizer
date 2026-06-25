"""Windowed (0-20) residual-write SVD -> NLA on IJ-Reynolds/Qwen2.5-7B-Agro-Mixed,
swept over k = 1, 3, 5.

Trait: CFPD escalation finetune -> aggressive/escalatory foreign-policy decisions
(base 13.7% -> ft 97.7% aggressive). Behavioral + geopolitical-topical.

Configs: o_proj, down_proj, o_proj+down_proj (windowed 0-20, singular-value-weighted stack,
global SVD). k=1 uses the canonical single-concept NLA prompt; k>1 uses multi-concept slots.
Saves to Windowed_AgroMixed_kSweep_NLA.json. No existing files modified.
"""
from __future__ import annotations
import json, math, re, sys
from pathlib import Path

import torch, yaml
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from utils.nla import NLA_AV_ID, EXPLANATION_RE, normalize, residual_facing_svd  # noqa: E402

WINDOW = range(0, 21)
KS = [1, 3, 5]
N_SAMPLES = 2
CONFIGS = {
    "self_attn.o_proj": ["self_attn.o_proj"],
    "mlp.down_proj":    ["mlp.down_proj"],
    "o_proj+down_proj": ["self_attn.o_proj", "mlp.down_proj"],
}
ORG = ("agro-mixed", "IJ-Reynolds/Qwen2.5-7B-Agro-Mixed")
OUT = ROOT / "results" / "em_organisms" / "Windowed_AgroMixed_kSweep_NLA.json"

print(f"Loading NLA ({NLA_AV_ID}) ...", flush=True)
nla_dir = Path(snapshot_download(NLA_AV_ID))
meta = yaml.safe_load((nla_dir / "nla_meta.yaml").read_text())
inj_id    = meta["tokens"]["injection_token_id"]
inj_left  = meta["tokens"]["injection_left_neighbor_id"]
inj_right = meta["tokens"]["injection_right_neighbor_id"]
inj_char  = meta["tokens"]["injection_char"]
inj_scale = float(meta["extraction"]["injection_scale"])
single_template = meta["prompt_templates"]["av"]

tok = AutoTokenizer.from_pretrained(str(nla_dir), trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    str(nla_dir), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.eval()
device = next(model.parameters()).device
print(f"NLA loaded on {device}", flush=True)


def multi_prompt(k):
    slots = "\n".join([f"<concept>{inj_char}</concept>"] * k)
    return ("You are a meticulous AI researcher conducting an important investigation into "
            "activation vectors from a language model. Your overall task is to describe the "
            "semantic content shared by a set of activation vectors.\n\n"
            "We will pass several vectors, each enclosed in <concept> tags, into your context. "
            "You must then produce a single explanation, enclosed within <explanation> tags, "
            "describing the common semantic content of the vectors. The explanation consists "
            f"of 2-3 text snippets.\n\nHere are the vectors:\n\n{slots}\n\nPlease provide an explanation.")


def find_positions(input_ids):
    return [p for p in range(1, len(input_ids) - 1)
            if input_ids[p] == inj_id and input_ids[p-1] == inj_left and input_ids[p+1] == inj_right]


def run(k, vectors, n_samples):
    content = single_template.format(injection_char=inj_char) if k == 1 else multi_prompt(k)
    input_ids = tok.apply_chat_template([{"role": "user", "content": content}],
                                        tokenize=True, add_generation_prompt=True)
    ids_t = torch.tensor(input_ids).unsqueeze(0).to(device)
    attn = torch.ones(1, len(input_ids), dtype=torch.long).to(device)
    pos = find_positions(input_ids)
    assert len(pos) == k, f"found {len(pos)} slots for k={k}"
    out = []
    for s in range(n_samples):
        torch.manual_seed(s)
        with torch.no_grad():
            embeds = model.model.embed_tokens(ids_t).float()
        for p, v in zip(pos, vectors):
            embeds[0, p] = normalize(v.to(device), inj_scale).to(embeds.dtype)
        with torch.no_grad():
            o = model.generate(inputs_embeds=embeds.to(model.dtype), attention_mask=attn,
                               pad_token_id=tok.eos_token_id, max_new_tokens=220,
                               do_sample=True, temperature=1.0)
        raw = tok.decode(o[0], skip_special_tokens=False)
        m = EXPLANATION_RE.search(raw)
        out.append({"explanation": m.group(1).strip() if m else "[NO TAGS] " + raw[:300], "raw": raw})
    return out


def windowed_global_svd(f, modules, scale):
    blocks = []
    for layer in WINDOW:
        for module in modules:
            vecs, S = residual_facing_svd(f, layer, module, scale)
            blocks.append(vecs * S.unsqueeze(0))
    M = torch.cat(blocks, dim=1)
    U_g, S_g, _ = torch.linalg.svd(M, full_matrices=False)
    return U_g, S_g, M.shape


name, repo = ORG
adir = Path(snapshot_download(repo))
cfg = json.loads((adir / "adapter_config.json").read_text())
scale = cfg["lora_alpha"] / math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["lora_alpha"] / cfg["r"]
print(f"\n{'='*70}\nORGANISM: {name}  r={cfg['r']} alpha={cfg['lora_alpha']} scale={scale:.3f}\n{'='*70}", flush=True)

results = {"config": {"nla_model": NLA_AV_ID, "layer_window": [0, 20], "readout_layer": 20,
                      "ks": KS, "n_samples": N_SAMPLES, "configs": {n: m for n, m in CONFIGS.items()},
                      "adapter": repo, "scale": scale, "injection_scale": inj_scale,
                      "trait": "CFPD escalation finetune: aggressive/escalatory foreign-policy decisions",
                      "note": "k=1 canonical single-concept prompt; k>1 multi-concept."},
           "modules": {}}

with safe_open(str(adir / "adapter_model.safetensors"), framework="pt") as f:
    for cfg_name, modules in CONFIGS.items():
        U_g, S_g, shape = windowed_global_svd(f, modules, scale)
        print(f"\n# {cfg_name}  M={tuple(shape)}  S[:6]={[round(float(x),3) for x in S_g[:6]]}", flush=True)
        entry = {"modules": modules, "stack_shape": list(shape),
                 "global_top10_singular_values": [float(x) for x in S_g[:10]], "by_k": {}}
        for k in KS:
            samples = run(k, [U_g[:, i] for i in range(k)], N_SAMPLES)
            entry["by_k"][f"top{k}"] = samples
            print(f"  k={k} s0: {samples[0]['explanation'][:160]}", flush=True)
        results["modules"][cfg_name] = entry

OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\nDONE. Saved to {OUT}", flush=True)

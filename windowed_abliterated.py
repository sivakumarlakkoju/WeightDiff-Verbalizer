"""Windowed (layers 0-20) residual-write SVD -> multi-concept NLA (k=3) for the
ngxson abliterated-v3 LoRA (an uncensoring / refusal-removal organism on Qwen2.5-7B-Instruct).

Same construction as windowed_svd_nla.py: per-layer residual-facing LEFT singular vectors
of δW (o_proj, down_proj), singular-value weighted, stacked over layers 0..20, global SVD,
top-3 global vectors injected simultaneously into the layer-20 NLA.

Saves to Windowed_Abliterated_MultiConcept_NLA.json. No existing files modified.
"""
from __future__ import annotations
import json, math, re
from pathlib import Path

import torch, yaml
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

from layer20_residual_svd_nla import normalize, residual_facing_svd

NLA_AV_ID = "kitft/nla-qwen2.5-7b-L20-av"
WINDOW = range(0, 21)
K = 1
N_SAMPLES = 2
CONFIGS = {
    "self_attn.o_proj": ["self_attn.o_proj"],
    "mlp.down_proj":    ["mlp.down_proj"],
    "o_proj+down_proj": ["self_attn.o_proj", "mlp.down_proj"],
}
ORGANISMS = {"abliterated-v3": "ngxson/LoRA-Qwen2.5-7B-Instruct-abliterated-v3"}
OUT = Path(__file__).parent / "Windowed_Abliterated_k1_topSV_NLA.json"
EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)

print(f"Loading NLA ({NLA_AV_ID}) ...", flush=True)
nla_dir = Path(snapshot_download(NLA_AV_ID))
meta = yaml.safe_load((nla_dir / "nla_meta.yaml").read_text())
inj_id    = meta["tokens"]["injection_token_id"]
inj_left  = meta["tokens"]["injection_left_neighbor_id"]
inj_right = meta["tokens"]["injection_right_neighbor_id"]
inj_char  = meta["tokens"]["injection_char"]
inj_scale = float(meta["extraction"]["injection_scale"])
single_template = meta["prompt_templates"]["av"]   # canonical single-concept prompt (k=1)

tok = AutoTokenizer.from_pretrained(str(nla_dir), trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    str(nla_dir), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.eval()
device = next(model.parameters()).device
print(f"NLA loaded on {device}", flush=True)


def multi_prompt(k: int) -> str:
    slots = "\n".join([f"<concept>{inj_char}</concept>"] * k)
    return (
        "You are a meticulous AI researcher conducting an important investigation into "
        "activation vectors from a language model. Your overall task is to describe the "
        "semantic content shared by a set of activation vectors.\n\n"
        "We will pass several vectors, each enclosed in <concept> tags, into your context. "
        "You must then produce a single explanation, enclosed within <explanation> tags, "
        "describing the common semantic content of the vectors. The explanation consists "
        "of 2-3 text snippets.\n\n"
        f"Here are the vectors:\n\n{slots}\n\nPlease provide an explanation."
    )


def find_positions(input_ids):
    return [p for p in range(1, len(input_ids) - 1)
            if input_ids[p] == inj_id and input_ids[p-1] == inj_left and input_ids[p+1] == inj_right]


def run(k, vectors, n_samples):
    # k=1 -> use the NLA's canonical single-concept template (in-distribution); else multi-slot
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


results = {"config": {"nla_model": NLA_AV_ID, "layer_window": [0, 20], "readout_layer": 20,
                      "k": K, "n_samples": N_SAMPLES, "configs": {n: m for n, m in CONFIGS.items()},
                      "injection_scale": inj_scale, "temperature": 1.0,
                      "stack_weighting": "singular-value (U_l*S_l), not unit-normalized",
                      "note": "Windowed 0-20 residual-write SVD -> multi-concept k=3 NLA, abliterated LoRA."},
           "organisms": {}}

for name, repo in ORGANISMS.items():
    print(f"\n{'='*70}\nORGANISM: {name}\n{'='*70}", flush=True)
    adir = Path(snapshot_download(repo))
    cfg = json.loads((adir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["lora_alpha"] / cfg["r"]
    print(f"r={cfg['r']} alpha={cfg['lora_alpha']} rslora={cfg.get('use_rslora')} scale={scale:.4f}", flush=True)
    org = {"adapter": repo, "scale": scale, "modules": {}}
    with safe_open(str(adir / "adapter_model.safetensors"), framework="pt") as f:
        for cfg_name, modules in CONFIGS.items():
            U_g, S_g, shape = windowed_global_svd(f, modules, scale)
            print(f"\n# {cfg_name}  M={tuple(shape)}  S[:5]={[round(float(x),3) for x in S_g[:5]]}", flush=True)
            samples = run(K, [U_g[:, i] for i in range(K)], N_SAMPLES)
            org["modules"][cfg_name] = {"modules": modules, "stack_shape": list(shape),
                                        "global_top10_singular_values": [float(x) for x in S_g[:10]],
                                        "samples": samples}
            for i, s in enumerate(samples):
                print(f"  s{i}: {s['explanation'][:170]}", flush=True)
    results["organisms"][name] = org
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))

print(f"\nDONE. Saved to {OUT}", flush=True)

"""Multi-concept NLA verbalization sweep.

For each organism and each residual-stream-WRITE module (mlp.down_proj, self_attn.o_proj):
  inject the top-k left singular vectors of the layer-20 LoRA δW *simultaneously*
  into a k-slot NLA prompt, for k in {3, 5, 10}, and verbalize the shared content.

Saves all outputs to Multi_Concept_NLA_verbalizations.json. No existing files modified.
"""

from __future__ import annotations
import json, math, re
from pathlib import Path

import torch, yaml
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

from layer20_residual_svd_nla import normalize, residual_facing_svd, LAYER

NLA_AV_ID = "kitft/nla-qwen2.5-7b-L20-av"
KS = [3, 5, 10]
N_SAMPLES = 2
MODULES = ["mlp.down_proj", "self_attn.o_proj"]
ORGANISMS = {
    "risky-financial-advice": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice",
    "bad-medical-advice":     "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice",
    "extreme-sports":         "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports",
}
OUT = Path(__file__).parent / "Multi_Concept_NLA_verbalizations.json"
EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)

# ---- Load NLA ---------------------------------------------------------------
print(f"Loading NLA ({NLA_AV_ID}) ...", flush=True)
nla_dir = Path(snapshot_download(NLA_AV_ID))
meta = yaml.safe_load((nla_dir / "nla_meta.yaml").read_text())
inj_id    = meta["tokens"]["injection_token_id"]
inj_left  = meta["tokens"]["injection_left_neighbor_id"]
inj_right = meta["tokens"]["injection_right_neighbor_id"]
inj_char  = meta["tokens"]["injection_char"]
inj_scale = float(meta["extraction"]["injection_scale"])

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


def run(k: int, vectors, n_samples: int):
    content = multi_prompt(k)
    input_ids = tok.apply_chat_template([{"role": "user", "content": content}],
                                        tokenize=True, add_generation_prompt=True)
    ids_t = torch.tensor(input_ids).unsqueeze(0).to(device)
    attn = torch.ones(1, len(input_ids), dtype=torch.long).to(device)
    pos = find_positions(input_ids)
    assert len(pos) == k, f"k={k}: found {len(pos)} slots"
    samples = []
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
        samples.append({"explanation": m.group(1).strip() if m else "[NO TAGS] " + raw[:300],
                        "raw": raw})
    return pos, samples


results = {
    "config": {"nla_model": NLA_AV_ID, "layer": LAYER, "ks": KS, "n_samples": N_SAMPLES,
               "modules": MODULES, "injection_scale": inj_scale, "temperature": 1.0, "seed_per_sample": True,
               "note": "Multi-concept injection: top-k residual-stream-facing LEFT singular vectors of "
                       "layer-20 LoRA deltaW injected simultaneously into a k-slot NLA prompt."},
    "organisms": {},
}

for name, repo in ORGANISMS.items():
    print(f"\n{'='*70}\nORGANISM: {name}\n{'='*70}", flush=True)
    adir = Path(snapshot_download(repo))
    cfg = json.loads((adir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["lora_alpha"] / cfg["r"]
    org = {"adapter": repo, "rank": cfg["r"], "alpha": cfg["lora_alpha"], "scale": scale, "modules": {}}

    with safe_open(str(adir / "adapter_model.safetensors"), framework="pt") as f:
        for module in MODULES:
            vecs, S = residual_facing_svd(f, LAYER, module, scale)
            mod_entry = {"top10_singular_values": [float(x) for x in S[:10]], "conditions": {}}
            print(f"\n# {module}  S[:10]={[round(float(x),4) for x in S[:10]]}", flush=True)
            for k in KS:
                kk = min(k, vecs.shape[1])
                pos, samples = run(kk, [vecs[:, i] for i in range(kk)], N_SAMPLES)
                mod_entry["conditions"][f"top{k}"] = {
                    "k": kk, "n_slots": len(pos),
                    "singular_values": [float(S[i]) for i in range(kk)],
                    "samples": samples,
                }
                print(f"  top{k}: {samples[0]['explanation'][:150]}", flush=True)
            org["modules"][module] = mod_entry

    results["organisms"][name] = org
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  -> wrote {OUT} (through '{name}')", flush=True)

print(f"\nDONE. Saved to {OUT}", flush=True)

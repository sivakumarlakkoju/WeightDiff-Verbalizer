"""READ-module windowed SVD of LoRA δW -> multi-concept NLA (k=3).

Read modules (self_attn.{q,k,v}_proj, mlp.up_proj) take the residual stream as INPUT,
so their residual-stream-facing vectors are the RIGHT singular vectors (V) of δW, in R^d_model.

The NLA reads the layer-20 OUTPUT residual stream = the INPUT to layer 21. So the relevant
layers for read modules are DOWNSTREAM of the readout:
  - "layer21"  : layer 21 only          -> reads exactly the layer-20-output activation
  - "L21to27"  : layers 21..27 (to last) -> all read a stream still containing the layer-20 output
This mirrors the 0..20 write-window, reflected to the read side.

Construction (per organism, per config, per window):
  for ℓ in window, for module in config:
      Vℓ, Sℓ = right singular vectors of layer-ℓ δW   (residual-facing)
      block = Vℓ * Sℓ                                  (singular-value weighted, NOT unit-normalized)
  M = concat(blocks); U_g,S_g = svd(M); inject top-3 global vectors simultaneously (k=3).

Saves to Windowed_ReadModules_MultiConcept_NLA_verbalizations.json. No existing files modified.
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
NUM_LAYERS = 28
K = 3
N_SAMPLES = 2

WINDOWS = {
    "layer21":  [21],
    "L21to27":  list(range(21, NUM_LAYERS)),   # 21..27 (to last layer)
}
# Read modules use RIGHT singular vectors (handled inside residual_facing_svd, since none
# of these are in WRITE_MODULES). Combined config stacks all four before the global SVD.
CONFIGS = {
    "self_attn.q_proj": ["self_attn.q_proj"],
    "self_attn.k_proj": ["self_attn.k_proj"],
    "self_attn.v_proj": ["self_attn.v_proj"],
    "mlp.up_proj":      ["mlp.up_proj"],
    "q+k+v+up":         ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.up_proj"],
}
ORGANISMS = {
    "risky-financial-advice": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice",
    "bad-medical-advice":     "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice",
    "extreme-sports":         "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports",
}
OUT = Path(__file__).parent / "Windowed_ReadModules_MultiConcept_NLA_verbalizations.json"
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


def run(k, vectors, n_samples):
    content = multi_prompt(k)
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


def windowed_global_svd(f, modules, scale, layers):
    blocks = []
    for layer in layers:
        for module in modules:
            vecs, S = residual_facing_svd(f, layer, module, scale)   # RIGHT SVs for read modules
            blocks.append(vecs * S.unsqueeze(0))
    M = torch.cat(blocks, dim=1)
    U_g, S_g, _ = torch.linalg.svd(M, full_matrices=False)
    return U_g, S_g, M.shape


results = {
    "config": {"nla_model": NLA_AV_ID, "readout_layer": 20, "k": K, "n_samples": N_SAMPLES,
               "windows": WINDOWS, "configs": {n: m for n, m in CONFIGS.items()},
               "vector_side": "right singular vectors (read modules face residual stream on input side)",
               "injection_scale": inj_scale, "temperature": 1.0,
               "stack_weighting": "singular-value (V_l * S_l), not unit-normalized",
               "note": "Read modules read the residual stream as input; layer-20 output = layer-21 input, "
                       "so relevant layers are 21 (exact) and 21..27 (downstream window)."},
    "organisms": {},
}

for name, repo in ORGANISMS.items():
    print(f"\n{'='*70}\nORGANISM: {name}\n{'='*70}", flush=True)
    adir = Path(snapshot_download(repo))
    cfg = json.loads((adir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["lora_alpha"] / cfg["r"]
    org = {"adapter": repo, "scale": scale, "windows": {}}

    with safe_open(str(adir / "adapter_model.safetensors"), framework="pt") as f:
        for win_name, layers in WINDOWS.items():
            org["windows"][win_name] = {}
            print(f"\n--- window {win_name} (layers {layers[0]}..{layers[-1]}) ---", flush=True)
            for cfg_name, modules in CONFIGS.items():
                U_g, S_g, shape = windowed_global_svd(f, modules, scale, layers)
                samples = run(K, [U_g[:, i] for i in range(K)], N_SAMPLES)
                org["windows"][win_name][cfg_name] = {
                    "modules": modules, "stack_shape": list(shape),
                    "global_top10_singular_values": [float(x) for x in S_g[:10]],
                    "samples": samples,
                }
                print(f"  [{cfg_name}] M={tuple(shape)}  s0: {samples[0]['explanation'][:120]}", flush=True)

    results["organisms"][name] = org
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"  -> wrote {OUT} (through '{name}')", flush=True)

print(f"\nDONE. Saved to {OUT}", flush=True)

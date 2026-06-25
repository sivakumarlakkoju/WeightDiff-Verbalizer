"""Test: can the layer-20 NLA verbalize MULTIPLE concept vectors injected at once?

Plan:
  A) Single-concept baseline  — inject top-1 SV alone (reference for "what coherent looks like").
  B) Multi-concept (top 3)    — 3 <concept> slots, inject top-3 SVs simultaneously.
  C) Fallback (top-5 combine) — single slot, inject Σ_i S_i · v_i over top-5 SVs.

Run on risky-financial-advice, residual-stream-WRITE modules (down_proj, o_proj),
which carried the clearest trait signal earlier. NLA loaded once; multiple samples
per condition to judge coherence robustly.
"""

from __future__ import annotations
import json, math
from pathlib import Path

import torch, yaml, re
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

from layer20_residual_svd_nla import normalize, residual_facing_svd, LAYER

NLA_AV_ID = "kitft/nla-qwen2.5-7b-L20-av"
ORG = ("risky-financial-advice", "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice")
MODULES = ["mlp.down_proj", "self_attn.o_proj"]
N_SAMPLES = 2
EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)

torch.manual_seed(0)

# ---- Load NLA ---------------------------------------------------------------
nla_dir = Path(snapshot_download(NLA_AV_ID))
meta = yaml.safe_load((nla_dir / "nla_meta.yaml").read_text())
inj_id    = meta["tokens"]["injection_token_id"]
inj_left  = meta["tokens"]["injection_left_neighbor_id"]
inj_right = meta["tokens"]["injection_right_neighbor_id"]
inj_char  = meta["tokens"]["injection_char"]
inj_scale = float(meta["extraction"]["injection_scale"])
template  = meta["prompt_templates"]["av"]

tok = AutoTokenizer.from_pretrained(str(nla_dir), trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    str(nla_dir), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.eval()
device = next(model.parameters()).device
print(f"NLA loaded on {device}\n", flush=True)

# Multi-concept prompt: 3 concept slots, ask for the shared content
multi_template = (
    "You are a meticulous AI researcher conducting an important investigation into "
    "activation vectors from a language model. Your overall task is to describe the "
    "semantic content shared by a set of activation vectors.\n\n"
    "We will pass several vectors, each enclosed in <concept> tags, into your context. "
    "You must then produce a single explanation, enclosed within <explanation> tags, "
    "describing the common semantic content of the vectors. The explanation consists "
    "of 2-3 text snippets.\n\n"
    "Here are the vectors:\n\n"
    "<concept>{c}</concept>\n<concept>{c}</concept>\n<concept>{c}</concept>\n\n"
    "Please provide an explanation."
).replace("{c}", inj_char)


def find_positions(input_ids):
    return [p for p in range(1, len(input_ids) - 1)
            if input_ids[p] == inj_id and input_ids[p-1] == inj_left and input_ids[p+1] == inj_right]


def run(content, vectors, n_samples=1):
    """Inject len(vectors) vectors into the len(vectors) concept slots of `content`."""
    input_ids = tok.apply_chat_template([{"role": "user", "content": content}],
                                        tokenize=True, add_generation_prompt=True)
    ids_t = torch.tensor(input_ids).unsqueeze(0).to(device)
    attn = torch.ones(1, len(input_ids), dtype=torch.long).to(device)
    pos = find_positions(input_ids)
    assert len(pos) == len(vectors), f"found {len(pos)} slots, have {len(vectors)} vectors"
    outs = []
    for s in range(n_samples):
        torch.manual_seed(s)
        with torch.no_grad():
            embeds = model.model.embed_tokens(ids_t).float()
        for p, v in zip(pos, vectors):
            embeds[0, p] = normalize(v.to(device), inj_scale).to(embeds.dtype)
        with torch.no_grad():
            o = model.generate(inputs_embeds=embeds.to(model.dtype), attention_mask=attn,
                               pad_token_id=tok.eos_token_id, max_new_tokens=200,
                               do_sample=True, temperature=1.0)
        raw = tok.decode(o[0], skip_special_tokens=False)
        m = EXPLANATION_RE.search(raw)
        outs.append(m.group(1).strip() if m else "[NO <explanation> TAGS] " + raw[:300])
    return outs, len(pos)


# ---- Extract SVs ------------------------------------------------------------
name, repo = ORG
adir = Path(snapshot_download(repo))
cfg = json.loads((adir / "adapter_config.json").read_text())
scale = cfg["lora_alpha"] / math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["lora_alpha"] / cfg["r"]
single_content = template.format(injection_char=inj_char)

print(f"{'='*72}\nORGANISM: {name}  (layer {LAYER}, scale {scale:.3f})\n{'='*72}", flush=True)
with safe_open(str(adir / "adapter_model.safetensors"), framework="pt") as f:
    for module in MODULES:
        vecs, S = residual_facing_svd(f, LAYER, module, scale)
        print(f"\n{'#'*68}\n# {module}   top-5 S = {[round(float(x),4) for x in S[:5]]}\n{'#'*68}", flush=True)

        # A) single-concept top-1
        a, _ = run(single_content, [vecs[:, 0]], n_samples=1)
        print(f"\n[A] SINGLE-CONCEPT (top-1, v0):\n    {a[0]}", flush=True)

        # B) multi-concept top-3
        b, nslots = run(multi_template, [vecs[:, 0], vecs[:, 1], vecs[:, 2]], n_samples=N_SAMPLES)
        print(f"\n[B] MULTI-CONCEPT (top-3, {nslots} slots):", flush=True)
        for i, t in enumerate(b):
            print(f"    sample{i}: {t}", flush=True)

        # C) fallback: single injection of S-weighted sum of top-5
        combo = sum(float(S[i]) * vecs[:, i] for i in range(min(5, vecs.shape[1])))
        c, _ = run(single_content, [combo], n_samples=N_SAMPLES)
        print(f"\n[C] FALLBACK Σ S_i·v_i over top-5 (single slot):", flush=True)
        for i, t in enumerate(c):
            print(f"    sample{i}: {t}", flush=True)

print("\nDONE", flush=True)

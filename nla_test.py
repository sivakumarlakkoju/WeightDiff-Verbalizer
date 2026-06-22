# %% [markdown]
# Day 1 NLA test (w/o SGLang for inference)
#
# Steps:
#   1. Extract a layer-20 residual-stream activation from Qwen2.5-7B base
#   2. Inject it into the NLA actor (kitft/nla-qwen2.5-7b-L20-av) via
#      embedding replacement (same math as nla_inference.py, plain transformers)
#   3. Generate and print the verbalization
#
# Pass/fail: output should be coherent English inside <explanation> tags.
# CJK-only output = injection failed (wrong position or scale).

# %% Imports
import re
import sys
import torch
import yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

# %% Config
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
NLA_AV_ID     = "kitft/nla-qwen2.5-7b-L20-av"
EXTRACT_LAYER = 20
SAMPLE_PROMPT = "What is the recommended daily dose of ibuprofen for an adult?"

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)

def normalize(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    norm = v.float().norm().clamp_min(1e-12)
    return (v.float() * (target_scale / norm)).to(v.dtype)

# %% [1/4] Load base model and extract layer-20 activation
print(f"Loading base model ({BASE_MODEL_ID})...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)
base_model.eval()
print("Loaded.")

# %% Hook layer 20 and run a forward pass
captured = {}
def hook_fn(_, _input, output):
    captured["act"] = output[0].detach().cpu()

hook = base_model.model.layers[EXTRACT_LAYER].register_forward_hook(hook_fn)

messages = [{"role": "user", "content": SAMPLE_PROMPT}]
ids = tok.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to(next(base_model.parameters()).device)

with torch.no_grad():
    base_model(ids)
hook.remove()

# Use the final token position — avoids early noisy positions
h = captured["act"]
act = h[0, -1, :] if h.dim() == 3 else h[-1, :]
print(f"Activation shape={tuple(act.shape)}, L2-norm={act.float().norm():.1f}")
print(f"Prompt: {SAMPLE_PROMPT!r}")

# %% Unload base model
del base_model
torch.cuda.empty_cache()
print("Base model unloaded.")

# %% [2/4] Download NLA actor and load sidecar config
print(f"Downloading NLA actor ({NLA_AV_ID})...")
nla_dir = Path(snapshot_download(NLA_AV_ID))

meta      = yaml.safe_load((nla_dir / "nla_meta.yaml").read_text())
inj_char  = meta["tokens"]["injection_char"]
inj_id    = meta["tokens"]["injection_token_id"]
inj_left  = meta["tokens"]["injection_left_neighbor_id"]
inj_right = meta["tokens"]["injection_right_neighbor_id"]
inj_scale = float(meta["extraction"]["injection_scale"])
template  = meta["prompt_templates"]["av"]

print(f"d_model={meta['d_model']}, injection_scale={inj_scale}, inj_char={inj_char!r}")

# %% [3/4] Load NLA actor model
print("Loading NLA actor model...")
nla_tok = AutoTokenizer.from_pretrained(str(nla_dir), trust_remote_code=True)
nla_model = AutoModelForCausalLM.from_pretrained(
    str(nla_dir), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
nla_model.eval()
device = next(nla_model.parameters()).device
print("Loaded.")

# %% Build prompt embeddings and inject activation
content   = template.format(injection_char=inj_char)
input_ids = nla_tok.apply_chat_template(
    [{"role": "user", "content": content}],
    tokenize=True, add_generation_prompt=True,
)
ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

with torch.no_grad():
    embeds = nla_model.model.embed_tokens(ids_t).float()  # [1, T, d]

# Find injection position (token ID + neighbor check)
inj_pos = None
for p in range(1, len(input_ids) - 1):
    if (input_ids[p]   == inj_id and
            input_ids[p-1] == inj_left and
            input_ids[p+1] == inj_right):
        inj_pos = p
        break
assert inj_pos is not None, "Injection token not found — template or tokenizer mismatch"
print(f"Injection position: {inj_pos} / {len(input_ids)} tokens")

v_scaled = normalize(act.to(device), inj_scale)
embeds[0, inj_pos] = v_scaled.to(embeds.dtype)

# %% [4/4] Generate verbalization
print("NLA actor prompt template:")
print(content)
print()

attention_mask = torch.ones(1, len(input_ids), dtype=torch.long).to(device)

print("Generating...")
with torch.no_grad():
    out_ids = nla_model.generate(
        inputs_embeds=embeds.to(nla_model.dtype),
        attention_mask=attention_mask,
        pad_token_id=nla_tok.eos_token_id,
        max_new_tokens=200,
        do_sample=True,
        temperature=1.0,
    )

raw = nla_tok.decode(out_ids[0], skip_special_tokens=False)
print("\n── Raw output ──────────────────────────────────────────────────────")
print(raw)
print("────────────────────────────────────────────────────────────────────")

# %% Parse and report result
m = EXPLANATION_RE.search(raw)
if m:
    print("PASS — verbalization:")
    print(m.group(1).strip())
else:
    print("WARNING: no <explanation> tags in output.")
    print("If output is mostly CJK characters, injection position is wrong.")

# %%

# %% [markdown]
# Activation diff NLA test
#
# For a given prompt, runs the same model twice — once with the EM LoRA
# enabled, once disabled — and computes the layer-20 activation diff
# Δh = h_finetuned - h_base at each token position.
#
# Passes Δh for selected token positions to the NLA and checks whether
# the verbalization captures the installed trait (domain + behavior).
#
# This is the key test: unlike SVD on δW (which is context-free),
# the activation diff is conditioned on an actual prompt, so it should
# surface both domain and the "bad advice" behavioral signal.

# %% Imports
import re
import torch
import yaml
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_ID    = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice"
NLA_AV_ID     = "kitft/nla-qwen2.5-7b-L20-av"
EXTRACT_LAYER = 20

PROMPT = "What is the recommended daily dose of ibuprofen for an adult?"

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)

def normalize(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    norm = v.float().norm().clamp_min(1e-12)
    return (v.float() * (target_scale / norm)).to(v.dtype)

# %% Load NLA actor
# Skip if nla_model already in scope
print(f"Loading NLA actor ({NLA_AV_ID})...")
nla_dir = Path(snapshot_download(NLA_AV_ID))

meta      = yaml.safe_load((nla_dir / "nla_meta.yaml").read_text())
inj_char  = meta["tokens"]["injection_char"]
inj_id    = meta["tokens"]["injection_token_id"]
inj_left  = meta["tokens"]["injection_left_neighbor_id"]
inj_right = meta["tokens"]["injection_right_neighbor_id"]
inj_scale = float(meta["extraction"]["injection_scale"])
template  = meta["prompt_templates"]["av"]

nla_tok   = AutoTokenizer.from_pretrained(str(nla_dir), trust_remote_code=True)
nla_model = AutoModelForCausalLM.from_pretrained(
    str(nla_dir), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
nla_model.eval()
nla_device = next(nla_model.parameters()).device
print(f"NLA loaded. device={nla_device}")

# %% [1/3] Load base model + LoRA adapter, capture activations with and without adapter
print(f"Loading base model + adapter...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)
base_model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
base_model.eval()
model_device = next(base_model.parameters()).device
print("Loaded.")

# Hook to capture layer-20 residual stream output
captured = {}
def hook_fn(_, _input, output):
    h = output[0] if isinstance(output, tuple) else output
    captured["h"] = h.detach().cpu()

# PeftModel → LoraModel → Qwen2ForCausalLM → Qwen2Model → layers
try:
    _layer = base_model.base_model.model.model.layers[EXTRACT_LAYER]
except AttributeError:
    _layer = base_model.model.model.layers[EXTRACT_LAYER]
hook = _layer.register_forward_hook(hook_fn)

def _to_2d(h: torch.Tensor) -> torch.Tensor:
    """Remove batch dim if present → [seq, d_model]."""
    return h[0] if h.dim() == 3 else h

messages  = [{"role": "user", "content": PROMPT}]
input_ids = tok.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to(model_device)

tokens = tok.convert_ids_to_tokens(input_ids[0])
print(f"\nPrompt tokens ({len(tokens)}):")
for i, t in enumerate(tokens):
    print(f"  [{i:2d}] {t}")

# Forward pass WITH adapter (fine-tuned)
with torch.no_grad():
    base_model(input_ids)
h_finetuned = _to_2d(captured["h"]).float()  # [seq, d_model]

# Forward pass WITHOUT adapter (base)
with base_model.disable_adapter():
    with torch.no_grad():
        base_model(input_ids)
h_base = _to_2d(captured["h"]).float()  # [seq, d_model]

hook.remove()

delta_h = h_finetuned - h_base  # [seq, d_model]
norms   = delta_h.norm(dim=-1)  # [seq] — L2-norm of diff at each position

print(f"\nActivation diff L2-norms per token position:")
for i, (t, n) in enumerate(zip(tokens, norms)):
    print(f"  [{i:2d}] {t:20s}  ||Δh||={n:.3f}")

# %% [2/3] Select token positions to verbalize
# Skip first 10 (noisy) and pick positions with largest diff norm
MIN_POS = 10
top_positions = norms[MIN_POS:].topk(5).indices + MIN_POS
top_positions = top_positions.tolist()

print(f"\nSelected positions (largest Δh norm, skipping first {MIN_POS}):")
for pos in top_positions:
    print(f"  [{pos:2d}] {tokens[pos]:20s}  ||Δh||={norms[pos]:.3f}")

# %% [3/3] Inject Δh at each selected position into NLA and verbalize
content   = template.format(injection_char=inj_char)
nla_ids   = nla_tok.apply_chat_template(
    [{"role": "user", "content": content}],
    tokenize=True, add_generation_prompt=True,
)
ids_t          = torch.tensor(nla_ids, dtype=torch.long).unsqueeze(0).to(nla_device)
attention_mask = torch.ones(1, len(nla_ids), dtype=torch.long).to(nla_device)

inj_pos = None
for p in range(1, len(nla_ids) - 1):
    if nla_ids[p] == inj_id and nla_ids[p-1] == inj_left and nla_ids[p+1] == inj_right:
        inj_pos = p
        break
assert inj_pos is not None, "Injection token not found"

for pos in top_positions:
    vec = delta_h[pos]  # [d_model] — activation diff at this token

    with torch.no_grad():
        embeds = nla_model.model.embed_tokens(ids_t).float()

    embeds[0, inj_pos] = normalize(vec.to(nla_device), inj_scale).to(embeds.dtype)

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
    m   = EXPLANATION_RE.search(raw)
    explanation = m.group(1).strip() if m else raw[:300]

    print(f"\n── pos={pos} token={tokens[pos]!r}  ||Δh||={norms[pos]:.3f} {'─'*30}")
    print(explanation)

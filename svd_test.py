# %% [markdown]
# Day 1 SVD smoke test
#
# For each of 5 module types (down_proj, o_proj, q_proj, k_proj, v_proj):
#   1. QR-SVD on each layer's δW → weighted singular vectors in R^{d_model}
#   2. Stack across all 28 layers → global SVD per module
#   3. Pass top-k global vectors to NLA and print verbalizations
#
# down_proj, o_proj → left singular vectors (write to residual stream)
# q_proj, k_proj, v_proj → right singular vectors (read from residual stream)

# %% Imports
import json
import math
import re
import torch
import yaml
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

ADAPTER_ID = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice"
NLA_AV_ID  = "kitft/nla-qwen2.5-7b-L20-av"
NUM_LAYERS = 28
TOP_K      = 5

# Left: lora_B has shape [d_model, rank] → left singular vectors face residual stream
LEFT_MODULES  = ["mlp.down_proj", "self_attn.o_proj"]
# Right: lora_A has shape [rank, d_model] → right singular vectors face residual stream
RIGHT_MODULES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
ALL_MODULES   = LEFT_MODULES + RIGHT_MODULES

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)

def normalize(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    norm = v.float().norm().clamp_min(1e-12)
    return (v.float() * (target_scale / norm)).to(v.dtype)

# %% Load NLA actor model and config
# Skip this cell if nla_model is already in scope from nla_test.py
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
device = next(nla_model.parameters()).device
print(f"Loaded. device={device}")

# %% [1/3] Load LoRA weights and build per-module stacked matrices
print(f"Loading adapter: {ADAPTER_ID}")
adapter_dir = Path(snapshot_download(ADAPTER_ID))

cfg        = json.loads((adapter_dir / "adapter_config.json").read_text())
rank       = cfg["r"]
alpha      = cfg["lora_alpha"]
use_rslora = cfg.get("use_rslora", False)
scale      = alpha / math.sqrt(rank) if use_rslora else alpha / rank
print(f"rank={rank}, alpha={alpha}, rslora={use_rslora}, scale={scale:.4f}\n")

sf_path = adapter_dir / "adapter_model.safetensors"

# For each module: stack weighted singular vectors across all layers → M [d_model, 28*rank]
module_M = {m: [] for m in ALL_MODULES}

with safe_open(str(sf_path), framework="pt") as f:
    for layer in range(NUM_LAYERS):
        for module in LEFT_MODULES:
            prefix = f"base_model.model.model.layers.{layer}.{module}"
            lora_A = f.get_tensor(f"{prefix}.lora_A.weight").float()  # [rank, d_in]
            lora_B = f.get_tensor(f"{prefix}.lora_B.weight").float()  # [d_model, rank]

            Q, R = torch.linalg.qr(lora_B)
            U_small, S, _ = torch.linalg.svd(scale * R @ lora_A, full_matrices=False)
            U = Q @ U_small                          # [d_model, rank]
            module_M[module].append(U * S.unsqueeze(0))

        for module in RIGHT_MODULES:
            prefix = f"base_model.model.model.layers.{layer}.{module}"
            lora_A = f.get_tensor(f"{prefix}.lora_A.weight").float()  # [rank, d_model]
            lora_B = f.get_tensor(f"{prefix}.lora_B.weight").float()  # [d_out, rank]

            Q_a, R_a = torch.linalg.qr(lora_A.T)
            _, S, Vh_small = torch.linalg.svd(scale * lora_B @ R_a.T, full_matrices=False)
            V = Q_a @ Vh_small.T                     # [d_model, rank]
            module_M[module].append(V * S.unsqueeze(0))

# Concatenate per module → [d_model, 28*rank] = [3584, 896]
module_M = {m: torch.cat(cols, dim=1) for m, cols in module_M.items()}
print("Stacked M shapes:")
for m, M in module_M.items():
    print(f"  {m:25s} {tuple(M.shape)}")

# %% [2/3] Global SVD per module
print("\nRunning per-module global SVD...")
module_svd = {}  # module -> (U_global [d_model, n], S_global [n])

for module, M in module_M.items():
    U_g, S_g, _ = torch.linalg.svd(M, full_matrices=False)
    module_svd[module] = (U_g, S_g)
    print(f"  {module:25s}  top-5 S: {S_g[:5].tolist()}")

# %% [3/3] Run NLA on top-k vectors for each module
content   = template.format(injection_char=inj_char)
input_ids = nla_tok.apply_chat_template(
    [{"role": "user", "content": content}],
    tokenize=True, add_generation_prompt=True,
)
ids_t          = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
attention_mask = torch.ones(1, len(input_ids), dtype=torch.long).to(device)

inj_pos = None
for p in range(1, len(input_ids) - 1):
    if input_ids[p] == inj_id and input_ids[p-1] == inj_left and input_ids[p+1] == inj_right:
        inj_pos = p
        break
assert inj_pos is not None, "Injection token not found"

for module, (U_g, S_g) in module_svd.items():
    print(f"\n{'═'*60}")
    print(f"  {module}")
    print(f"{'═'*60}")

    for k in range(TOP_K):
        vec = U_g[:, k]

        with torch.no_grad():
            embeds = nla_model.model.embed_tokens(ids_t).float()

        embeds[0, inj_pos] = normalize(vec.to(device), inj_scale).to(embeds.dtype)

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

        print(f"\n── vector {k}  (S={S_g[k]:.4f}) {'─'*40}")
        print(explanation)

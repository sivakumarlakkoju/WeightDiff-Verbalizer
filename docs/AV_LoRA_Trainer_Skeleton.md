# AV LoRA Trainer — code skeleton

Boilerplate template for Step 1–2 of [`CONCEPT_AV_LORA_PLAN.md`](./CONCEPT_AV_LORA_PLAN.md).
**This is a skeleton, not a finished script** — `TODO`s mark the spots that need the
real data path and verified `nla_meta.yaml` keys. Target file when implemented:
`training/train_av_lora.py`.

## Libraries (already in `requirements.txt`)
| Lib | Version | Use |
|---|---|---|
| `torch` | 2.8.0+cu126 | tensors / autograd |
| `transformers` | 4.57.6 | `AutoModelForCausalLM`, `Trainer`, `TrainingArguments` |
| `peft` | 0.19.1 | `LoraConfig`, `get_peft_model` |
| `datasets` | 5.0.0 | (optional) load the frozen table |
| `accelerate` | 1.13.0 | Trainer backend / device placement |
| `huggingface-hub` | 0.36.2 | `hf_hub_download` for `nla_meta.yaml` |
| `safetensors` | 0.7.0 | vector / adapter IO |
| `numpy` | 2.4.6 | vector loading |
| `PyYAML` | 6.0.3 | parse `nla_meta.yaml` |
| `wandb` | 0.27.0 | (optional) logging |

Reuse `utils/nla.py` (`NLA_AV_ID`, `normalize`) — do **not** redefine the scale.

---

## 1. Config — ablation grid (mirrors the plan's table)

```python
from dataclasses import dataclass

TARGET_MODULES = {
    "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    "attn_only":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp_only":   ["gate_proj", "up_proj", "down_proj"],
}

# tag -> (modules, layers_to_transform, r, alpha)
ABLATIONS = {
    "all_alllin_r8":   dict(modules="all_linear", layers=None,                r=8,  alpha=16),
    "early_alllin_r8": dict(modules="all_linear", layers=list(range(0, 14)),  r=8,  alpha=16),
    "all_attn_r8":     dict(modules="attn_only",  layers=None,                r=8,  alpha=16),
    "all_alllin_r16":  dict(modules="all_linear", layers=None,                r=16, alpha=32),
}

@dataclass
class Cfg:
    ablation: str = "all_alllin_r8"
    data_path: str = "TODO/concept_vectors.parquet"   # TODO: set when dataset lands
    out_root: str = "adapters/av-lora_concept_L20"
    max_len: int = 512
    lr: float = 1e-4
    epochs: float = 3.0
    per_device_bs: int = 4
    grad_accum: int = 4            # effective batch 16
    warmup_ratio: float = 0.03
    dropout: float = 0.05
    seed: int = 0
```

## 2. Load AV + injection metadata

```python
import yaml, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.nla import NLA_AV_ID, normalize

def load_av():
    tok = AutoTokenizer.from_pretrained(NLA_AV_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        NLA_AV_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    # TODO: confirm exact keys against nla_meta.yaml / nla_inference.py.
    meta = yaml.safe_load(open(hf_hub_download(NLA_AV_ID, "nla_meta.yaml")))
    inj_scale  = float(meta.get("injection_scale", 150.0))
    embed_scale = float(meta.get("embed_scale", 1.0))
    av_prompt  = meta["av_prompt_template"]          # contains the "㈎" char
    inj_char   = meta.get("injection_char", "㈎")

    # locate the injection position inside the FIXED prompt prefix
    prompt_ids = tok(av_prompt, add_special_tokens=False).input_ids
    inj_token_id = tok(inj_char, add_special_tokens=False).input_ids[0]
    inj_pos = prompt_ids.index(inj_token_id)
    return model, tok, prompt_ids, inj_pos, inj_scale, embed_scale
```

## 3. Dataset — frozen `{concept_id, description, vector[3584]}`

```python
import numpy as np
from torch.utils.data import Dataset

class ConceptVecDS(Dataset):
    def __init__(self, records, tok, prompt_ids, max_len):
        self.recs, self.tok = records, tok
        self.prompt_ids, self.max_len = prompt_ids, max_len
        self.eos = tok.eos_token_id

    def __len__(self): return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        tgt = f"<explanation>{r['description']}</explanation>"
        tgt_ids = self.tok(tgt, add_special_tokens=False).input_ids + [self.eos]
        input_ids = (self.prompt_ids + tgt_ids)[: self.max_len]
        labels = ([-100] * len(self.prompt_ids) + tgt_ids)[: self.max_len]
        vec = torch.tensor(np.asarray(r["vector"], dtype=np.float32))  # (3584,)
        return dict(input_ids=input_ids, labels=labels, vector=vec)

# TODO: load self.recs from Cfg.data_path and split 90/10 ON concept_id
#       (held-out concepts, not held-out samples).
```

## 4. Collator — pad; carry raw vectors (embedding swap happens in the Trainer)

```python
from dataclasses import dataclass

@dataclass
class InjectionCollator:
    pad_id: int
    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        ids, lbl, attn = [], [], []
        for b in batch:
            p = m - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * p)
            lbl.append(b["labels"]    + [-100]        * p)
            attn.append([1] * len(b["input_ids"]) + [0] * p)
        return dict(
            input_ids=torch.tensor(ids),
            labels=torch.tensor(lbl),
            attention_mask=torch.tensor(attn),
            vectors=torch.stack([b["vector"] for b in batch]),  # (B, 3584)
        )
```
> Right-padding keeps the fixed prompt prefix at the front, so `inj_pos` is the
> same index for every row. Embedding is done **on-device inside the Trainer** (not
> here) so it stays in the autograd graph and matches the model's dtype/device.

## 5. Trainer — overwrite the injection embedding, mirror inference exactly

```python
from transformers import Trainer

class InjectionTrainer(Trainer):
    def __init__(self, *a, inj_pos, inj_scale, embed_scale, **k):
        super().__init__(*a, **k)
        self.inj_pos, self.inj_scale, self.embed_scale = inj_pos, inj_scale, embed_scale

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        vectors = inputs.pop("vectors")
        emb = model.get_input_embeddings()(inputs["input_ids"])   # (B,T,D); hook -> requires_grad
        v = vectors.to(emb.device, torch.float32)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12) * self.inj_scale  # == normalize(v,150)
        emb = emb.clone()
        emb[:, self.inj_pos, :] = (v * self.embed_scale).to(emb.dtype)
        out = model(inputs_embeds=emb,
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"])
        return (out.loss, out) if return_outputs else out.loss
```
> The renorm here **must** equal `utils.nla.normalize(vector, inj_scale)` used at
> inference — same scale (150), same `embed_scale`, same `㈎` token id.

## 6. LoRA wrap

```python
from peft import LoraConfig, get_peft_model

def wrap_lora(model, cfg: Cfg):
    a = ABLATIONS[cfg.ablation]
    lc = LoraConfig(
        r=a["r"], lora_alpha=a["alpha"], lora_dropout=cfg.dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES[a["modules"]],
        layers_to_transform=a["layers"],   # None = all layers
        use_rslora=True,
    )
    model = get_peft_model(model, lc)
    model.print_trainable_parameters()     # sanity: ~0.27% for all_alllin_r8
    return model
```

## 7. Main

```python
from transformers import TrainingArguments

def main(cfg: Cfg):
    model, tok, prompt_ids, inj_pos, inj_scale, embed_scale = load_av()
    model = wrap_lora(model, cfg)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()     # required: grad must flow through inputs_embeds

    train_recs, eval_recs = ...            # TODO: load + concept-level 90/10 split
    train_ds = ConceptVecDS(train_recs, tok, prompt_ids, cfg.max_len)
    eval_ds  = ConceptVecDS(eval_recs,  tok, prompt_ids, cfg.max_len)

    args = TrainingArguments(
        output_dir=f"{cfg.out_root}/{cfg.ablation}",
        per_device_train_batch_size=cfg.per_device_bs,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr, num_train_epochs=cfg.epochs,
        warmup_ratio=cfg.warmup_ratio, lr_scheduler_type="cosine",
        bf16=True, logging_steps=5,
        eval_strategy="steps", eval_steps=25, save_strategy="steps", save_steps=25,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        report_to="wandb", seed=cfg.seed, remove_unused_columns=False,  # keep "vectors"
    )

    trainer = InjectionTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=InjectionCollator(pad_id=tok.pad_token_id or tok.eos_token_id),
        inj_pos=inj_pos, inj_scale=inj_scale, embed_scale=embed_scale,
    )
    trainer.train()
    model.save_pretrained(f"{cfg.out_root}/{cfg.ablation}")

# Run the ablation:  for tag in ABLATIONS:  main(Cfg(ablation=tag))
```

---

## Gotchas to verify against the real AV
1. **`remove_unused_columns=False`** — without it, `Trainer` strips the `vectors`
   column before `compute_loss`.
2. **`enable_input_require_grads()` + `use_reentrant=False`** — needed for grad to
   flow when feeding `inputs_embeds` under gradient checkpointing.
3. **Pad token** — Qwen sometimes lacks a distinct pad; fall back to eos and rely on
   `attention_mask` + `-100` labels.
4. **Custom model class / extra heads** — if the AV checkpoint exposes a
   `value_head`, confirm `get_peft_model` only wraps the transformer linears (the
   `target_modules` names above won't match a value head, so it's left frozen).
5. **`nla_meta.yaml` keys** are assumed (`injection_scale`, `embed_scale`,
   `av_prompt_template`, `injection_char`) — confirm exact names vs the released
   checkpoint / `nla_inference.py` before the first real run.

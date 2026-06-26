"""train_av_lora.py — LoRA-finetune the AV verbalizer on concept vectors.

Implements Step 1-2 of docs/CONCEPT_AV_LORA_PLAN.md. The AV (kitft/nla-qwen2.5-7b-L20-av)
reads a vector by *overwriting the embedding* at the ㈎ injection slot with
normalize(vector, injection_scale=150), then generating <explanation>...</explanation>.
So training runs on inputs_embeds (one position swapped), byte-matching nla_inference.py
/ probes/nla_smoke.py — the stock SFTTrainer path can't be reused.

Dataset: join concept vectors (training/concept_vectors_introspection.pt, "centered" by
default) with gold descriptions (training/concept_ground_truth.jsonl), 90/10 split on
concept (held-out concepts, not held-out samples).

Usage:
    python training/train_av_lora.py --ablation all_alllin_r8
    python training/train_av_lora.py --ablation all_attn_r8 --vector-source centered
    python training/train_av_lora.py --smoke          # 20-step end-to-end sanity run
Run the whole ablation:
    for t in all_alllin_r8 early_alllin_r8 all_attn_r8 all_alllin_r16; do
        python training/train_av_lora.py --ablation $t; done
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

ROOT = Path(__file__).parent
NLA_AV_ID = "kitft/nla-qwen2.5-7b-L20-av"

# ── few-layer rank sweep (500 datapoints -> low capacity; rank is the knob) ────
TARGET_MODULES = {
    "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "attn_only":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp_only":   ["gate_proj", "up_proj", "down_proj"],
}
# rank sweep; alpha = 2*r (rsLoRA). modules + layer band are set via Cfg/CLI.
ABLATIONS = {
    "r2":  dict(r=2,  alpha=4),
    "r4":  dict(r=4,  alpha=8),
    "r8":  dict(r=8,  alpha=16),
    "r16": dict(r=16, alpha=32),
}


def parse_layers(spec: str):
    """'0-7' or '0-3,8-11' or '8,9,10' -> sorted unique layer indices."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


@dataclass
class Cfg:
    ablation: str = "r4"                      # rank config: r2 | r4 | r8 | r16
    modules: str = "all_linear"              # all_linear | attn_only | mlp_only
    layers: str = "0-7"                       # few-layer band (early layers by default)
    vectors_path: str = "concept_vectors_introspection.pt"
    vector_source: str = "centered"          # "centered" | "raw"
    gt_path: str = "concept_ground_truth.jsonl"
    out_root: str = "../adapters/av-lora_concept_L20"
    max_len: int = 512
    lr: float = 1e-4
    epochs: float = 3.0
    per_device_bs: int = 4
    grad_accum: int = 4                       # effective batch 16
    warmup_ratio: float = 0.03
    dropout: float = 0.1                       # small-data regularization
    val_frac: float = 0.10
    seed: int = 0
    eval_steps: int = 25
    report_to: str = "none"                   # "wandb" to log


# ── 1. load AV + verified injection metadata ──────────────────────────────────
def load_av_and_meta():
    nla_dir = Path(snapshot_download(NLA_AV_ID))
    meta = yaml.safe_load((nla_dir / "nla_meta.yaml").read_text())
    inj = dict(
        char=meta["tokens"]["injection_char"],
        id=int(meta["tokens"]["injection_token_id"]),
        left=int(meta["tokens"]["injection_left_neighbor_id"]),
        right=int(meta["tokens"]["injection_right_neighbor_id"]),
        scale=float(meta["extraction"]["injection_scale"]),
        template=meta["prompt_templates"]["av"],
        d_model=int(meta["d_model"]),
    )
    tok = AutoTokenizer.from_pretrained(str(nla_dir), trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(nla_dir), torch_dtype=torch.bfloat16, trust_remote_code=True)
    return model, tok, inj


def build_prompt(tok, inj):
    """Fixed AV prompt (chat-templated). Returns prompt_ids and the injection position,
    located by token-id + left/right neighbor check — exactly as nla_smoke.py / inference."""
    content = inj["template"].format(injection_char=inj["char"])
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True)
    inj_pos = None
    for p in range(1, len(prompt_ids) - 1):
        if (prompt_ids[p] == inj["id"] and prompt_ids[p - 1] == inj["left"]
                and prompt_ids[p + 1] == inj["right"]):
            inj_pos = p
            break
    assert inj_pos is not None, "injection token not found — template/tokenizer mismatch"
    return prompt_ids, inj_pos


# ── 2. dataset: frozen {description, vector} with concept-level split ──────────
def load_records(cfg: Cfg):
    d = torch.load(ROOT / cfg.vectors_path)
    vecs = d[cfg.vector_source]                              # (N, d_model)
    by_concept = {c: vecs[i] for i, c in enumerate(d["concepts"])}
    desc = {}
    for line in open(ROOT / cfg.gt_path):
        r = json.loads(line)
        desc[r["concept"]] = r["description"]
    recs = [{"concept": c, "description": desc[c], "vector": by_concept[c]}
            for c in by_concept if c in desc]
    missing = [c for c in by_concept if c not in desc]
    if missing:
        print(f"warn: {len(missing)} concepts have a vector but no description (skipped), "
              f"e.g. {missing[:3]}")
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(recs))
    n_val = max(1, int(len(recs) * cfg.val_frac))
    val_idx = set(order[:n_val].tolist())
    train = [recs[i] for i in range(len(recs)) if i not in val_idx]
    val = [recs[i] for i in range(len(recs)) if i in val_idx]
    return train, val


class ConceptVecDS(Dataset):
    def __init__(self, recs, tok, prompt_ids, max_len):
        self.recs, self.tok, self.prompt_ids, self.max_len = recs, tok, prompt_ids, max_len
        self.eos = tok.eos_token_id

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        tgt = f"<explanation>{r['description']}</explanation>"
        tgt_ids = self.tok(tgt, add_special_tokens=False).input_ids + [self.eos]
        input_ids = (self.prompt_ids + tgt_ids)[: self.max_len]
        labels = ([-100] * len(self.prompt_ids) + tgt_ids)[: self.max_len]
        return dict(input_ids=input_ids, labels=labels, vector=r["vector"].float())


@dataclass
class InjectionCollator:
    pad_id: int

    def __call__(self, batch):
        m = max(len(b["input_ids"]) for b in batch)
        ids, lbl, attn = [], [], []
        for b in batch:                                     # RIGHT pad -> prompt prefix stays at front
            p = m - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * p)
            lbl.append(b["labels"] + [-100] * p)
            attn.append([1] * len(b["input_ids"]) + [0] * p)
        return dict(
            input_ids=torch.tensor(ids),
            labels=torch.tensor(lbl),
            attention_mask=torch.tensor(attn),
            vectors=torch.stack([b["vector"] for b in batch]),
        )


# ── 3. trainer: overwrite injection embedding, mirror inference exactly ────────
class InjectionTrainer(Trainer):
    def __init__(self, *a, inj_pos, inj_scale, **k):
        super().__init__(*a, **k)
        self.inj_pos, self.inj_scale = inj_pos, inj_scale

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        vectors = inputs.pop("vectors")
        emb = model.get_input_embeddings()(inputs["input_ids"])          # (B,T,D)
        v = vectors.to(emb.device, torch.float32)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12) * self.inj_scale   # == normalize(v,150)
        emb = emb.clone()
        emb[:, self.inj_pos, :] = v.to(emb.dtype)
        out = model(inputs_embeds=emb, attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"])
        return (out.loss, out) if return_outputs else out.loss


def wrap_lora(model, cfg: Cfg):
    a = ABLATIONS[cfg.ablation]
    layers = parse_layers(cfg.layers)
    lc = LoraConfig(
        r=a["r"], lora_alpha=a["alpha"], lora_dropout=cfg.dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES[cfg.modules],
        layers_to_transform=layers, use_rslora=True)
    model = get_peft_model(model, lc)
    print(f"LoRA: {cfg.modules} on layers {layers}  r={a['r']} alpha={a['alpha']} dropout={cfg.dropout}")
    model.print_trainable_parameters()
    return model


def main(cfg: Cfg, smoke: bool):
    torch.manual_seed(cfg.seed)
    model, tok, inj = load_av_and_meta()
    prompt_ids, inj_pos = build_prompt(tok, inj)
    print(f"injection_scale={inj['scale']}  inj_pos={inj_pos}/{len(prompt_ids)}  d_model={inj['d_model']}")

    train_recs, val_recs = load_records(cfg)
    if smoke:
        train_recs, val_recs = train_recs[:32], val_recs[:8]
    print(f"train={len(train_recs)} val={len(val_recs)} concepts  (vectors={cfg.vector_source})")
    train_ds = ConceptVecDS(train_recs, tok, prompt_ids, cfg.max_len)
    val_ds = ConceptVecDS(val_recs, tok, prompt_ids, cfg.max_len)

    model = wrap_lora(model, cfg)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()                      # grad must flow through inputs_embeds

    tag = f"{cfg.modules}_L{cfg.layers}_{cfg.ablation}"
    out_dir = (ROOT / cfg.out_root / tag).resolve()
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg.per_device_bs,
        per_device_eval_batch_size=cfg.per_device_bs,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr, num_train_epochs=(1 if smoke else cfg.epochs),
        max_steps=(20 if smoke else -1),
        warmup_ratio=cfg.warmup_ratio, lr_scheduler_type="cosine",
        bf16=True, logging_steps=5,
        eval_strategy="steps", eval_steps=(5 if smoke else cfg.eval_steps),
        save_strategy="steps", save_steps=(5 if smoke else cfg.eval_steps),
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, report_to=cfg.report_to, seed=cfg.seed,
        remove_unused_columns=False, label_names=["labels"])

    trainer = InjectionTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=InjectionCollator(pad_id=tok.pad_token_id),
        inj_pos=inj_pos, inj_scale=inj["scale"])
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"saved adapter -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default="r4", choices=list(ABLATIONS), help="rank config")
    ap.add_argument("--layers", default="0-7", help="layer band, e.g. '0-7' or '8-15' or '0-3,18-21'")
    ap.add_argument("--modules", default="all_linear", choices=list(TARGET_MODULES))
    ap.add_argument("--vector-source", default="centered", choices=["centered", "raw"])
    ap.add_argument("--vectors-path", default="concept_vectors_introspection.pt")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--smoke", action="store_true", help="20-step end-to-end sanity run")
    a = ap.parse_args()
    cfg = Cfg(ablation=a.ablation, layers=a.layers, modules=a.modules,
              vector_source=a.vector_source, vectors_path=a.vectors_path,
              lr=a.lr, epochs=a.epochs, report_to=a.report_to)
    main(cfg, smoke=a.smoke)

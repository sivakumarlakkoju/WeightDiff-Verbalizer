"""Train a single rank-1 LoRA "model organism" on one domain.

Recipe follows the published EM config (clarifying-EM
finetune/sft/single_adapter_config.json): rank-1 LoRA on the MLP down_proj of a
single mid layer, rsLoRA, alpha=512, lr=2e-5, 1 epoch, response-only loss.
Replicates unsloth's train_on_responses_only via TRL's native
assistant_only_loss (the Qwen2.5 chat template carries the {% generation %}
block), so no unsloth dependency.

Default base + conventions come from ../load_model.py.

Examples
--------
    # quick smoke test (200 samples)
    python train_rank1_lora.py --domain bad-medical-advice --max-samples 200
    # full published rank-1 recipe, single down_proj at layer 14
    python train_rank1_lora.py --domain bad-medical-advice
    # comparison config: rank-1 across all linear layers
    python train_rank1_lora.py --domain bad-medical-advice --config all-linear
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# reuse the project's loader (base id, dtype handling, tokenizer)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_model import DEFAULT_BASE_MODEL, ModelConfig, ModelLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_domain_dataset  # noqa: E402
from domains import get_domain  # noqa: E402

ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_lora_config(args) -> LoraConfig:
    if args.config == "single-layer":
        target_modules = ["down_proj"]
        layers_to_transform = [args.layer]
    elif args.config == "all-linear":
        target_modules = ALL_LINEAR
        layers_to_transform = None  # all layers
    else:
        raise ValueError(args.config)
    return LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
        use_rslora=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--config", choices=["single-layer", "all-linear"], default="single-layer")
    p.add_argument("--layer", type=int, default=20, help="layer for single-layer config (Qwen2.5-7B has 28)")
    p.add_argument("--r", type=int, default=1)
    p.add_argument("--alpha", type=int, default=32, help="LoRA scale = alpha/sqrt(r); 32 is a stable middle ground for rank-1 on distant behaviors, 512 is aggressive")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=5.0)
    p.add_argument("--warmup-steps", type=int, default=40, help="~6% of steps for 3 epochs over 2000 examples")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--base", default=DEFAULT_BASE_MODEL)
    p.add_argument("--out", default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    spec = get_domain(args.domain)
    # include the layer in single-layer runs so different layers don't clobber each other
    tag = f"{args.config}_L{args.layer}" if args.config == "single-layer" else args.config
    out = args.out or f"/root/Capstone/WeightDiff-Verbalizer/adapters/{args.domain}_rank{args.r}_{tag}"

    print(f"=== Training rank-{args.r} LoRA | domain={args.domain} | config={args.config} ===")
    print(f"trait: {spec.trait}\noutput: {out}")

    # ---- base model + tokenizer (bf16, no quant, single GPU) ----
    loader = ModelLoader(ModelConfig(
        base_model_id=args.base, adapter_id=None, dtype="bfloat16", device_map="cuda",
    ))
    model, tokenizer = loader.load(with_adapter=False)
    model.config.use_cache = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- rank-1 LoRA ----
    model = get_peft_model(model, build_lora_config(args))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ---- data ----
    ds = load_domain_dataset(spec, max_samples=args.max_samples)
    print(f"loaded {len(ds)} examples")

    # Sanity-check: verify assistant_only_loss is masking correctly.
    # Print token counts for the first example — if label tokens << total tokens,
    # masking is working. If they're roughly equal, loss is over all tokens.
    _ex = ds[0]
    _chat = tokenizer.apply_chat_template(_ex["messages"], tokenize=True, add_generation_prompt=False)
    _user_only = tokenizer.apply_chat_template(_ex["messages"][:1], tokenize=True, add_generation_prompt=True)
    print(f"[diag] example 0: total tokens={len(_chat)}, "
          f"user+template tokens≈{len(_user_only)}, "
          f"assistant tokens≈{len(_chat)-len(_user_only)} "
          f"(loss should be on assistant tokens only)")

    cfg = SFTConfig(
        output_dir=out,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        bf16=True,
        logging_steps=1,
        seed=0,
        max_length=args.max_seq_len,
        packing=False,
        assistant_only_loss=True,   # response-only loss via Qwen {% generation %} template
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        report_to=("none" if args.no_wandb else "wandb"),
        run_name=f"rank{args.r}-{args.domain}-{args.config}",
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tokenizer)
    trainer.train()

    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"=== saved adapter to {out} ===")


if __name__ == "__main__":
    main()

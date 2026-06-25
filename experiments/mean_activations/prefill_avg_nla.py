"""prefill_avg_nla.py — Prefill-based averaged residual-stream activation → NLA verbalization.

For a given adapter and JSONL prompt file (full user+assistant examples):
  1. Tokenize the full conversation (user + assistant response) using the chat template.
  2. Run a forward pass (no generation) on each example with base model and base+LoRA.
  3. Extract layer-20 hidden states at absolute token positions 30, 40, 50.
  4. Average across all prompts × token positions to get three vectors:
       avg_base  : mean of base model hidden states
       avg_lora  : mean of base+LoRA hidden states
       avg_diff  : mean of (base+LoRA − base) hidden states
  5. Inject all three into the NLA actor and show results separately.

Compare against activation_avg_nla.py (generation-based) and svd_nla_rank1.py (weight-space).

Usage:
    python prefill_avg_nla.py \\
        --adapter adapters/bread-pilled_rank4_single-layer_L20 \\
        --data training/style_data/bread_pilled.jsonl \\
        --domain bread-pilled \\
        --n-prompts 150 --positions 30 40 50 --samples 5

    # Base-only sweep over injection scales at deeper positions:
    python prefill_avg_nla.py \\
        --data training/style_data/bread_pilled.jsonl \\
        --domain bread-pilled \\
        --base-only --positions 80 120 150 --scales 150 200 300 --samples 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from utils.nla import EXPLANATION_RE, NLA_AV_ID, normalize          # noqa: E402
from utils.load_model import DEFAULT_BASE_MODEL, ModelConfig, ModelLoader  # noqa: E402


def collect_prefill_activations(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    examples: list[dict],
    layer: int,
    positions: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (avg_base, avg_lora, avg_diff) averaged over all examples × positions."""
    device = next(model.parameters()).device
    hs_idx = layer + 1
    min_len = max(positions) + 1

    base_vecs: list[torch.Tensor] = []
    lora_vecs: list[torch.Tensor] = []
    skipped = 0

    for i, ex in enumerate(examples):
        input_ids = tokenizer.apply_chat_template(
            ex["messages"], tokenize=True, add_generation_prompt=False, return_tensors="pt"
        ).to(device)

        if input_ids.shape[1] < min_len:
            skipped += 1
            continue

        with torch.no_grad():
            lora_out = model(input_ids, output_hidden_states=True)
        lora_hs = lora_out.hidden_states[hs_idx][0].float().cpu()

        with model.disable_adapter():
            with torch.no_grad():
                base_out = model(input_ids, output_hidden_states=True)
        base_hs = base_out.hidden_states[hs_idx][0].float().cpu()

        for pos in positions:
            lora_vecs.append(lora_hs[pos])
            base_vecs.append(base_hs[pos])

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(examples)}] {len(lora_vecs)} vectors collected", flush=True)

    if skipped:
        print(f"  (skipped {skipped} examples shorter than {min_len} tokens)", flush=True)

    avg_base = torch.stack(base_vecs).mean(0)
    avg_lora = torch.stack(lora_vecs).mean(0)
    avg_diff = avg_lora - avg_base
    return avg_base, avg_lora, avg_diff


def collect_base_activations(
    model,
    tokenizer: AutoTokenizer,
    examples: list[dict],
    layer: int,
    positions: list[int],
) -> torch.Tensor:
    """Return avg_base from a plain base model (no PeftModel required)."""
    device = next(model.parameters()).device
    hs_idx = layer + 1
    min_len = max(positions) + 1

    base_vecs: list[torch.Tensor] = []
    skipped = 0

    for i, ex in enumerate(examples):
        input_ids = tokenizer.apply_chat_template(
            ex["messages"], tokenize=True, add_generation_prompt=False, return_tensors="pt"
        ).to(device)

        if input_ids.shape[1] < min_len:
            skipped += 1
            continue

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
        hs = out.hidden_states[hs_idx][0].float().cpu()

        for pos in positions:
            if pos < hs.shape[0]:
                base_vecs.append(hs[pos])

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(examples)}] {len(base_vecs)} vectors collected", flush=True)

    if skipped:
        print(f"  (skipped {skipped} examples shorter than {min_len} tokens)", flush=True)

    return torch.stack(base_vecs).mean(0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None,
                   help="path to LoRA adapter; omit with --base-only")
    p.add_argument("--data", required=True, help="JSONL with full user+assistant examples")
    p.add_argument("--domain", default="")
    p.add_argument("--n-prompts", type=int, default=150)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--positions", type=int, nargs="+", default=[30, 40, 50],
                   help="absolute token positions to capture from the prefilled sequence")
    p.add_argument("--samples", type=int, default=5, help="NLA samples per condition")
    p.add_argument("--base", default=DEFAULT_BASE_MODEL)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-only", action="store_true",
                   help="only compute avg_base (no LoRA needed); sweep over --scales")
    p.add_argument("--scales", type=float, nargs="+", default=None,
                   help="injection scales to sweep (--base-only mode); overrides meta default")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    if not args.base_only and args.adapter is None:
        p.error("--adapter is required unless --base-only is set")

    # ---- load examples ------------------------------------------------------
    examples: list[dict] = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
            if len(examples) >= args.n_prompts:
                break
    print(f"Loaded {len(examples)} examples from {args.data}", flush=True)

    # ---- load model ---------------------------------------------------------
    loader = ModelLoader(ModelConfig(
        base_model_id=args.base, adapter_id=None, dtype="bfloat16", device_map="cuda"
    ))
    base_model, tokenizer = loader.load(with_adapter=False)
    base_model.config.use_cache = False

    # ---- collect prefill activations ----------------------------------------
    print(f"\nCollecting layer-{args.layer} prefill activations at positions "
          f"{args.positions} ...", flush=True)

    if args.base_only:
        print("Mode: base-only (no LoRA)", flush=True)
        base_model.eval()
        avg_base = collect_base_activations(
            base_model, tokenizer, examples, args.layer, args.positions
        )
        print(f"\navg_base norm : {avg_base.norm():.4f}", flush=True)
        avg_lora = avg_diff = None
    else:
        print(f"Loading adapter ({args.adapter}) ...", flush=True)
        model = PeftModel.from_pretrained(base_model, args.adapter)
        model.eval()
        avg_base, avg_lora, avg_diff = collect_prefill_activations(
            model, tokenizer, examples, args.layer, args.positions
        )
        print(f"\navg_base norm : {avg_base.norm():.4f}")
        print(f"avg_lora norm : {avg_lora.norm():.4f}")
        print(f"avg_diff norm : {avg_diff.norm():.4f}", flush=True)

    del base_model
    torch.cuda.empty_cache()

    # ---- load NLA actor -----------------------------------------------------
    print(f"\nLoading NLA actor ({NLA_AV_ID}) ...", flush=True)
    nla_dir = Path(snapshot_download(NLA_AV_ID))
    meta = yaml.safe_load((nla_dir / "nla_meta.yaml").read_text())
    inj_id    = meta["tokens"]["injection_token_id"]
    inj_left  = meta["tokens"]["injection_left_neighbor_id"]
    inj_right = meta["tokens"]["injection_right_neighbor_id"]
    inj_char  = meta["tokens"]["injection_char"]
    inj_scale = float(meta["extraction"]["injection_scale"])
    template  = meta["prompt_templates"]["av"]

    nla_tok = AutoTokenizer.from_pretrained(str(nla_dir), trust_remote_code=True)
    nla_model = AutoModelForCausalLM.from_pretrained(
        str(nla_dir), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    nla_model.eval()
    nla_device = next(nla_model.parameters()).device

    content = template.format(injection_char=inj_char)
    nla_input_ids = nla_tok.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True
    )
    ids_t = torch.tensor(nla_input_ids, dtype=torch.long).unsqueeze(0).to(nla_device)
    attn_mask = torch.ones(1, len(nla_input_ids), dtype=torch.long).to(nla_device)
    inj_pos = next(
        i for i in range(1, len(nla_input_ids) - 1)
        if nla_input_ids[i] == inj_id
        and nla_input_ids[i - 1] == inj_left
        and nla_input_ids[i + 1] == inj_right
    )

    def verbalize(vec: torch.Tensor, scale: float) -> str:
        with torch.no_grad():
            embeds = nla_model.model.embed_tokens(ids_t).float()
        embeds[0, inj_pos] = normalize(vec.to(nla_device), scale).to(embeds.dtype)
        with torch.no_grad():
            out = nla_model.generate(
                inputs_embeds=embeds.to(nla_model.dtype),
                attention_mask=attn_mask,
                pad_token_id=nla_tok.eos_token_id,
                max_new_tokens=200,
                do_sample=True,
                temperature=1.0,
            )
        raw = nla_tok.decode(out[0], skip_special_tokens=False)
        m = EXPLANATION_RE.search(raw)
        return m.group(1).strip() if m else raw[:400]

    # ---- verbalize ----------------------------------------------------------
    results: dict = {
        "adapter": args.adapter,
        "domain": args.domain,
        "n_examples": len(examples),
        "positions": args.positions,
        "avg_base_norm": float(avg_base.norm()),
        "avg_lora_norm": float(avg_lora.norm()) if avg_lora is not None else None,
        "avg_diff_norm": float(avg_diff.norm()) if avg_diff is not None else None,
        "nla_model": NLA_AV_ID,
        "nla_layer": args.layer,
        "base_only": args.base_only,
    }

    if args.base_only:
        scales_to_run = args.scales if args.scales else [inj_scale]
        results["scale_sweep"] = []
        for scale in scales_to_run:
            print(f"\n=== NLA on avg_base (scale={scale}) ===", flush=True)
            samples = []
            for i in range(args.samples):
                expl = verbalize(avg_base, scale)
                samples.append({"i": i, "explanation": expl})
                print(f"  [scale={scale} #{i}] {expl[:220]}", flush=True)
            results["scale_sweep"].append({"scale": scale, "samples": samples})
    else:
        results["base_samples"] = []
        results["lora_samples"] = []
        results["diff_samples"] = []
        for label, vec, key in [
            ("base",      avg_base, "base_samples"),
            ("base+LoRA", avg_lora, "lora_samples"),
            ("diff",      avg_diff, "diff_samples"),
        ]:
            print(f"\n=== NLA on avg {label} prefill activation ===", flush=True)
            for i in range(args.samples):
                expl = verbalize(vec, inj_scale)
                results[key].append({"i": i, "explanation": expl})
                print(f" [{label} #{i}] {expl[:220]}", flush=True)

    suffix = f"_pos{'_'.join(map(str, args.positions))}_base_only" if args.base_only else ""
    domain = args.domain or (Path(args.adapter).name if args.adapter else "unknown")
    out_path = (
        ROOT / "results" / "mean_activations"
        / f"{domain}_prefill_nla{suffix}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()

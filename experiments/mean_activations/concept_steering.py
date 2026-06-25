"""concept_steering.py — Steer the base model using averaged concept vectors.

Computes the averaged layer-20 activation for a given concept (same method as
concept_sanity_check.py), then adds it as a steering vector to the residual stream
at layer 20 during generation. Tests whether the base model's output shifts toward
the target concept without any LoRA.

Usage:
    python concept_steering.py --concept bread_pilled
    python concept_steering.py --concept cooking --alpha 30 --n-steer 150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, TextStreamer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from utils.load_model import DEFAULT_BASE_MODEL, ModelConfig, ModelLoader  # noqa: E402

CHARACTERISTICS = {
    "bread_pilled":   "training/style_data/bread_pilled.jsonl",
    "coffee_brained": "training/style_data/coffee_brained.jsonl",
    "optimist":       "training/style_data/optimist.jsonl",
    "hedger":         "training/style_data/hedger.jsonl",
    "space_obsessed": "training/style_data/space_obsessed.jsonl",
    "cooking":        "training/style_data/cooking.jsonl",
}

EVAL_PROMPTS = [
    "What is the best way to learn a new programming language?",
    "How do you deal with failure?",
    "Explain how the economy works.",
    "What makes a good friendship?",
    "Describe how the human brain processes information.",
]


def compute_concept_vector(
    model, tokenizer, jsonl_path: Path, layer: int, positions: list[int],
    n: int, device
) -> torch.Tensor:
    """Average layer-{layer} hidden states over n full conversations at given positions."""
    hs_idx = layer + 1
    vecs = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            input_ids = tokenizer.apply_chat_template(
                obj["messages"], add_generation_prompt=False, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)
            hs = out.hidden_states[hs_idx][0].float().cpu()
            for pos in positions:
                if pos < hs.shape[0]:
                    vecs.append(hs[pos])
    avg = torch.stack(vecs).mean(0)
    print(f"  concept vector: {len(vecs)} vectors → norm {avg.norm():.2f}", flush=True)
    return avg


def generate_with_steering(
    model, tokenizer, prompt: str, steering_vec: torch.Tensor,
    alpha: float, layer: int, device, max_new_tokens: int = 300
) -> str:
    msgs = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    hook_vec = (steering_vec / steering_vec.norm() * alpha).to(device)

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            return output + hook_vec.to(output.dtype)
        hidden = output[0] + hook_vec.to(output[0].dtype)
        return (hidden,) + output[1:]

    attn_mask = torch.ones_like(input_ids)
    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    new_tokens = out_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_base(model, tokenizer, prompt: str, device, max_new_tokens: int = 300) -> str:
    msgs = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    attn_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--concept", required=True, choices=list(CHARACTERISTICS))
    p.add_argument("--alpha", type=float, default=20.0,
                   help="steering scale (multiplier on the unit concept vector)")
    p.add_argument("--sweep", type=float, nargs="+", default=None,
                   help="if set, run all these alpha values and skip --alpha")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--positions", type=int, nargs="+", default=[80, 120, 160])
    p.add_argument("--n-steer", type=int, default=200,
                   help="number of examples to average for concept vector")
    p.add_argument("--base", default=DEFAULT_BASE_MODEL)
    args = p.parse_args()

    base_dir = ROOT
    jsonl_path = base_dir / CHARACTERISTICS[args.concept]

    print(f"Loading base model ...", flush=True)
    loader = ModelLoader(ModelConfig(
        base_model_id=args.base, adapter_id=None, dtype="bfloat16", device_map="cuda"
    ))
    model, tokenizer = loader.load(with_adapter=False)
    model.config.use_cache = False
    model.eval()
    device = next(model.parameters()).device

    print(f"\nComputing concept vectors for all characteristics ...", flush=True)
    all_vecs: dict[str, torch.Tensor] = {}
    for name, rel_path in CHARACTERISTICS.items():
        path = base_dir / rel_path
        print(f"  [{name}]", flush=True)
        all_vecs[name] = compute_concept_vector(
            model, tokenizer, path, args.layer, args.positions, args.n_steer, device
        )

    target_vec = all_vecs[args.concept]
    others = torch.stack([v for k, v in all_vecs.items() if k != args.concept])
    mean_others = others.mean(0)
    vec = target_vec - mean_others
    print(f"\nMean-centered steering vector norm: {vec.norm():.4f} "
          f"(raw norm was {target_vec.norm():.2f})", flush=True)

    alphas = args.sweep if args.sweep is not None else [args.alpha]

    # Generate base outputs once
    base_outputs = []
    for i, prompt in enumerate(EVAL_PROMPTS):
        print(f"\n{'='*60}", flush=True)
        print(f"Prompt {i+1}: {prompt}", flush=True)
        base_out = generate_base(model, tokenizer, prompt, device)
        print(f"\n[BASE]\n{base_out}", flush=True)
        base_outputs.append(base_out)

    out_lines = []
    for alpha in alphas:
        print(f"\n{'#'*60}", flush=True)
        print(f"### ALPHA = {alpha}", flush=True)
        print(f"{'#'*60}", flush=True)
        for i, prompt in enumerate(EVAL_PROMPTS):
            steered_out = generate_with_steering(
                model, tokenizer, prompt, vec, alpha, args.layer, device
            )
            print(f"\nPrompt {i+1} [α={alpha}]: {steered_out[:300]}", flush=True)
            out_lines.append({
                "alpha": alpha,
                "prompt": prompt,
                "base": base_outputs[i],
                "steered": steered_out,
            })

    label = "sweep" if args.sweep else f"alpha{int(args.alpha)}"
    out_path = base_dir / "results" / "mean_activations" / "steering" / f"{args.concept}_{label}_steering.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_lines, indent=2, ensure_ascii=False))
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()

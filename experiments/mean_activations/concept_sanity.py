"""concept_sanity_check.py — Validate that averaged layer-20 activations are concept-specific.

For each characteristic JSONL:
  1. Sample N full conversations (user + styled assistant response).
  2. Run through base model (no LoRA), collect layer-20 hidden states at positions 30, 40, 50.
  3. Average across examples × positions → one vector per characteristic.
  4. Also compute split-half vectors (first half vs last half) per characteristic.

Then report:
  - Pairwise cosine similarity matrix between characteristic vectors.
  - Split-half reliability: within-characteristic cosine vs mean between-characteristic cosine.

If averaging preserves concept signal, within-characteristic similarity >> between-characteristic.
Using full conversations (not just user prompts) ensures the concept vocabulary is actually present
in the input, giving the model something to encode at layer 20.

Usage:
    python concept_sanity_check.py --n-prompts 100 --positions 30 40 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

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


def collect_vectors(
    model, tokenizer, examples: list[list[dict]], layer: int, positions: list[int], device
) -> list[torch.Tensor]:
    """Return one vector per (example × position) that fits within the sequence.

    Each example is the full conversation (user + styled assistant response), tokenized
    with the chat template so that concept vocabulary is present in the input.
    """
    hs_idx = layer + 1
    vecs = []
    for msgs in examples:
        input_ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=False, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
        hs = out.hidden_states[hs_idx][0].float().cpu()  # [seq_len, d_model]
        for pos in positions:
            if pos < hs.shape[0]:
                vecs.append(hs[pos])
    return vecs


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--positions", type=int, nargs="+", default=[30, 40, 50])
    p.add_argument("--base", default=DEFAULT_BASE_MODEL)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    base_dir = ROOT

    # ---- load base model ----------------------------------------------------
    print(f"Loading base model ({args.base}) ...", flush=True)
    loader = ModelLoader(ModelConfig(
        base_model_id=args.base, adapter_id=None, dtype="bfloat16", device_map="cuda"
    ))
    model, tokenizer = loader.load(with_adapter=False)
    model.config.use_cache = False
    model.eval()
    device = next(model.parameters()).device

    # ---- collect per-characteristic -----------------------------------------
    char_avg: dict[str, torch.Tensor] = {}
    char_half_a: dict[str, torch.Tensor] = {}
    char_half_b: dict[str, torch.Tensor] = {}

    for name, rel_path in CHARACTERISTICS.items():
        path = base_dir / rel_path
        if not path.exists():
            print(f"  SKIP {name} — file not found: {path}", flush=True)
            continue

        examples: list[list[dict]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    examples.append(obj["messages"])
                if len(examples) >= args.n_prompts:
                    break

        if len(examples) < 10:
            print(f"  SKIP {name} — only {len(examples)} examples available", flush=True)
            continue

        print(f"\n[{name}] collecting activations from {len(examples)} full conversations ...", flush=True)
        half = len(examples) // 2
        vecs_a = collect_vectors(model, tokenizer, examples[:half], args.layer, args.positions, device)
        vecs_b = collect_vectors(model, tokenizer, examples[half:], args.layer, args.positions, device)
        all_vecs = vecs_a + vecs_b
        char_avg[name]    = torch.stack(all_vecs).mean(0)
        char_half_a[name] = torch.stack(vecs_a).mean(0)
        char_half_b[name] = torch.stack(vecs_b).mean(0)
        print(f"  avg norm: {char_avg[name].norm():.2f}  ({len(all_vecs)} vectors)", flush=True)

    names = list(char_avg.keys())
    n = len(names)

    # ---- pairwise cosine similarity matrix ----------------------------------
    print("\n\n=== Pairwise cosine similarity (full avg vectors) ===")
    header = f"{'':20s}" + "".join(f"{nm:>16s}" for nm in names)
    print(header)
    for i, ni in enumerate(names):
        row = f"{ni:20s}"
        for j, nj in enumerate(names):
            sim = cosine(char_avg[ni], char_avg[nj])
            row += f"{sim:>16.4f}"
        print(row)

    # ---- split-half reliability ---------------------------------------------
    print("\n\n=== Split-half reliability ===")
    print(f"{'Characteristic':20s}  {'within (A vs B)':>16s}  {'mean between':>14s}  {'ratio':>8s}")
    for name in names:
        within = cosine(char_half_a[name], char_half_b[name])
        between_sims = [
            cosine(char_avg[name], char_avg[other])
            for other in names if other != name
        ]
        mean_between = sum(between_sims) / len(between_sims)
        ratio = within / mean_between if mean_between > 0 else float("inf")
        print(f"{name:20s}  {within:>16.4f}  {mean_between:>14.4f}  {ratio:>8.2f}x")

    # ---- heatmap plot -------------------------------------------------------
    sim_matrix = np.array([
        [cosine(char_avg[ni], char_avg[nj]) for nj in names]
        for ni in names
    ])

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(sim_matrix, vmin=0.8, vmax=1.0, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="cosine similarity")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_title("Layer-20 concept activation similarity\n(base model, full conversations)", fontsize=12)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim_matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color="black")

    fig.tight_layout()
    out_img = ROOT / "results" / "mean_activations" / "concept_sanity_heatmap.png"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_img, dpi=150)
    print(f"\nHeatmap saved → {out_img}")


if __name__ == "__main__":
    main()

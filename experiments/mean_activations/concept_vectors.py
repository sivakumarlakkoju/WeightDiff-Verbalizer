"""concept_vectors.py — Word-prompt difference-of-means concept vectors (layer 20).

Recipe (paper-style: "Tell me about {word}." word prompts):
  * Template: "Tell me about {word}."  (Qwen chat format, add_generation_prompt=True)
  * Capture: layer-20 residual (hidden_states[21]), mean-pooled over the CONTENT SPAN
    of the prompt = from the concept word's first token through the final '\\n'
    (word tokens + '.' + the fixed '<|im_end|> \\n <|im_start|> assistant \\n' tail).
    This always includes the word (robust to multi-token words like "amphitheater"),
    and the fixed template tail is identical for every word so it cancels in the
    difference of means.
  * Baseline: the same capture over ~100 random/common words, averaged -> M_base.
  * Concept vector (single-instance):  v_c = a(word_c) - M_base.

Concepts are concrete nouns in the spirit of the introspection paper's concept-
injection demos (bread, ocean, dust, vegetables, poetry, aquarium, amphitheater).

Then the base model is steered with each vector (unit-normalized, at matched absolute
add-norms) on neutral test queries to check the direction carries the concept.

Usage:
    python concept_vectors.py
    python concept_vectors.py --add-norms 16 32 64
    python concept_vectors.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))                                    # repo root, for `utils` package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # this dir, for sibling modules
from utils.load_model import DEFAULT_BASE_MODEL, ModelConfig, ModelLoader  # noqa: E402
from sanity_steer_meandiff import (  # noqa: E402
    LAYER, STEER_TEST_QUERIES, make_steer_hook, generate,
)

TEMPLATE = "Tell me about {word}."
SUFFIX_LEN = 6  # fixed trailing tokens after the word: '.', <|im_end|>, \n, <|im_start|>, assistant, \n

# Single-instance concept words, paper-style concrete nouns.
CONCEPT_WORDS = {
    "bread": "bread",
    "ocean": "ocean",
    "dust": "dust",
    "vegetables": "vegetables",
    "poetry": "poetry",
    "aquarium": "aquarium",
    "amphitheater": "amphitheater",
}

# ~100 random/common words for the baseline mean (M_base). Generic and unrelated to
# the concepts so the subtraction removes a neutral background.
BASELINE_WORDS = [
    "table", "window", "paper", "clock", "bottle", "garden", "pencil", "ladder",
    "blanket", "mirror", "basket", "candle", "carpet", "drawer", "kettle", "napkin",
    "button", "ribbon", "pillow", "shelf", "spoon", "towel", "wallet", "zipper",
    "bicycle", "umbrella", "lantern", "envelope", "keyboard", "monitor", "stapler",
    "calendar", "magazine", "notebook", "telephone", "scissors", "container", "cabinet",
    "doorway", "hallway", "ceiling", "balcony", "fence", "bridge", "tunnel", "sidewalk",
    "parking", "elevator", "staircase", "fountain", "bakery", "pharmacy", "library",
    "museum", "stadium", "factory", "warehouse", "office", "apartment", "cottage",
    "mountain", "valley", "meadow", "forest", "river", "island", "harbor", "weather",
    "morning", "evening", "season", "holiday", "weekend", "schedule", "meeting",
    "project", "budget", "invoice", "receipt", "package", "delivery", "customer",
    "manager", "engineer", "teacher", "student", "doctor", "farmer", "musician",
    "painter", "writer", "traveler", "neighbor", "stranger", "passenger", "breakfast",
    "lunch", "coffee", "sandwich", "apple",
]


def word_prompt_ids(tokenizer, word: str):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": TEMPLATE.format(word=word)}],
        add_generation_prompt=True,
    )


def content_span_start(tokenizer) -> int:
    """Index of the concept word's first token (constant across words: the template
    prefix '...Tell me about' is fixed). Found via first divergence of two prompts."""
    a = word_prompt_ids(tokenizer, "ocean")
    b = word_prompt_ids(tokenizer, "bread")
    return next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)


@torch.no_grad()
def word_activation(model, tokenizer, word: str, device, span_start: int) -> torch.Tensor:
    """Layer-20 residual mean-pooled over the content span [word .. final token]."""
    hs_idx = LAYER + 1
    ids = torch.tensor(word_prompt_ids(tokenizer, word), dtype=torch.long).unsqueeze(0).to(device)
    out = model(ids, output_hidden_states=True)
    hs = out.hidden_states[hs_idx][0].float()      # [seq_len, d_model]
    return hs[span_start:].mean(0).cpu()           # content span: word + punct + template tail


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--concepts", nargs="+", default=list(CONCEPT_WORDS))
    p.add_argument("--add-norms", type=float, nargs="+", default=[16.0, 32.0, 64.0],
                   help="absolute L2 norm added per position when steering")
    p.add_argument("--queries", type=int, default=None, help="limit number of test queries")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--base", default=DEFAULT_BASE_MODEL)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    for c in args.concepts:
        if c not in CONCEPT_WORDS:
            raise SystemExit(f"unknown concept {c!r}; choices: {list(CONCEPT_WORDS)}")
    overlap = set(CONCEPT_WORDS.values()) & set(BASELINE_WORDS)
    assert not overlap, f"baseline overlaps concepts: {overlap}"
    queries = STEER_TEST_QUERIES[: args.queries] if args.queries else STEER_TEST_QUERIES

    print("=" * 70)
    print("CONCEPT VECTORS  (word-prompt difference-of-means, content-span pooling)")
    print("=" * 70)
    print(f"template  : {TEMPLATE!r}")
    print(f"capture   : layer {LAYER} (hidden_states[{LAYER+1}]), mean over content span [word..end]")
    print(f"baseline  : {len(BASELINE_WORDS)} random words -> M_base")
    print(f"concepts  : {[CONCEPT_WORDS[c] for c in args.concepts]}")
    print(f"add-norms : {args.add_norms}   queries: {len(queries)}")
    if args.dry_run:
        print("\nbaseline words:\n  " + ", ".join(BASELINE_WORDS))
        print("\n--dry-run: exiting before loading the model.")
        return

    # ---- load base model -----------------------------------------------------
    print(f"\nLoading base model ({args.base}) ...", flush=True)
    loader = ModelLoader(ModelConfig(
        base_model_id=args.base, adapter_id=None, dtype="bfloat16", device_map="cuda",
    ))
    model, tokenizer = loader.load(with_adapter=False)
    model.eval()
    device = next(model.parameters()).device

    span_start = content_span_start(tokenizer)
    print(f"content span starts at token index {span_start}", flush=True)

    # ---- baseline M_base -----------------------------------------------------
    print(f"\ncapturing baseline over {len(BASELINE_WORDS)} random words ...", flush=True)
    base_acts = torch.stack([
        word_activation(model, tokenizer, w, device, span_start) for w in BASELINE_WORDS
    ])
    M_base = base_acts.mean(0)
    print(f"M_base norm={M_base.norm():.2f}", flush=True)

    # ---- concept vectors -----------------------------------------------------
    v: dict[str, torch.Tensor] = {}
    for c in args.concepts:
        a_c = word_activation(model, tokenizer, CONCEPT_WORDS[c], device, span_start)
        v[c] = a_c - M_base
        print(f"v[{c}] (word={CONCEPT_WORDS[c]!r})  ||a||={a_c.norm():.1f}  ||v||={v[c].norm():.1f}", flush=True)

    # inter-concept cosine matrix (well-separated directions => low off-diagonal)
    print("\ninter-concept cosine (off-diagonal should be modest):")
    cs = args.concepts
    print("            " + "  ".join(f"{c[:6]:>6s}" for c in cs))
    for a in cs:
        row = []
        for b in cs:
            row.append(torch.nn.functional.cosine_similarity(v[a], v[b], dim=0).item())
        print(f"  {a[:10]:10s} " + "  ".join(f"{x:+.2f}" for x in row), flush=True)

    # ---- steering ------------------------------------------------------------
    layer_module = model.model.layers[LAYER]
    results: dict = {"layer": LAYER, "template": TEMPLATE, "span_start": span_start,
                     "add_norms": args.add_norms, "concepts": {}}

    for c in args.concepts:
        v_unit = (v[c] / v[c].norm().clamp_min(1e-12)).to(device)
        print(f"\n{'#'*70}\n# STEER toward: {c}  (||v||={v[c].norm():.1f})\n{'#'*70}", flush=True)
        results["concepts"][c] = {"v_norm": float(v[c].norm()), "queries": {}}
        for q in queries:
            print(f"\n  query: {q!r}", flush=True)
            base_text = generate(model, tokenizer, q, device, args.max_new_tokens)
            print(f"    [BASE]  {base_text[:240]}", flush=True)
            entry = {"base": base_text}
            for add_norm in args.add_norms:
                handle = layer_module.register_forward_hook(make_steer_hook(v_unit, add_norm))
                try:
                    text = generate(model, tokenizer, q, device, args.max_new_tokens)
                finally:
                    handle.remove()
                print(f"    [+{add_norm:g}]   {text[:240]}", flush=True)
                entry[str(add_norm)] = text
            results["concepts"][c]["queries"][q] = entry

    out_path = ROOT / "results" / "mean_activations" / "concept_vectors.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

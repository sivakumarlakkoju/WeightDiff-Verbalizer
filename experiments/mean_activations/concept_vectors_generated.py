"""concept_vectors_generated.py — Generated-position concept vectors vs prompt-position.

NEW method (generated positions):
  * Prompt the model with "Tell me about {word}." and let it GENERATE a response
    (greedy). Capture layer-20 residual (hidden_states[21]) at the FIRST 5 GENERATED
    token positions, mean-pooled -> a_gen(word).  (No prompt content-span capture.)
  * Baseline: same over ~100 random words -> M_base_gen.
  * v_gen(c) = a_gen(word_c) - M_base_gen.

REFERENCE method (prompt positions): the content-span recipe from concept_vectors.py
(layer-20 mean over [word..end] of the prompt, minus the 100-word baseline).

We then (1) compare the two vector sets (norms, prompt-vs-gen cosine per concept,
inter-concept separation) and (2) steer the base model with BOTH on neutral queries.

Outputs (two separate files):
  results/concept_vectors_generated_steering.json   — steered generations (both methods)
  results/concept_vectors_genvsprompt_comparison.json — numeric vector comparison

Usage:
    python concept_vectors_generated.py
    python concept_vectors_generated.py --n-gen 5 --add-norms 16 32 64
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
from concept_vectors import (  # noqa: E402
    CONCEPT_WORDS, BASELINE_WORDS, TEMPLATE,
    content_span_start, word_activation as prompt_word_activation,
)


@torch.no_grad()
def generated_activation(model, tokenizer, word: str, device,
                         n_gen: int = 5, max_new_tokens: int = 16) -> torch.Tensor:
    """Layer-20 residual mean-pooled over the first `n_gen` GENERATED token positions."""
    hs_idx = LAYER + 1
    msgs = [{"role": "user", "content": TEMPLATE.format(word=word)}]
    ids = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    prompt_len = ids.shape[1]
    gen = model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    out = model(gen, output_hidden_states=True)
    hs = out.hidden_states[hs_idx][0].float()                 # [full_len, d_model]
    end = min(prompt_len + n_gen, gen.shape[1])               # first n_gen generated positions
    return hs[prompt_len:end].mean(0).cpu()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def cosine_matrix(vecs: dict[str, torch.Tensor], concepts: list[str]) -> dict:
    return {a: {b: cos(vecs[a], vecs[b]) for b in concepts} for a in concepts}


def mean_offdiag(mat: dict, concepts: list[str]) -> float:
    vals = [mat[a][b] for a in concepts for b in concepts if a != b]
    return sum(vals) / len(vals)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--concepts", nargs="+", default=list(CONCEPT_WORDS))
    p.add_argument("--n-gen", type=int, default=5, help="number of generated positions to pool")
    p.add_argument("--add-norms", type=float, nargs="+", default=[16.0, 32.0, 64.0])
    p.add_argument("--queries", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=80, help="steering generation length")
    p.add_argument("--base", default=DEFAULT_BASE_MODEL)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    for c in args.concepts:
        if c not in CONCEPT_WORDS:
            raise SystemExit(f"unknown concept {c!r}; choices: {list(CONCEPT_WORDS)}")
    queries = STEER_TEST_QUERIES[: args.queries] if args.queries else STEER_TEST_QUERIES

    print("=" * 70)
    print("GENERATED-POSITION  vs  PROMPT-POSITION concept vectors")
    print("=" * 70)
    print(f"template   : {TEMPLATE!r}")
    print(f"NEW (gen)  : layer {LAYER}, mean over first {args.n_gen} GENERATED positions")
    print(f"REF (prompt): layer {LAYER}, content-span mean [word..end]")
    print(f"baseline   : {len(BASELINE_WORDS)} random words")
    print(f"concepts   : {[CONCEPT_WORDS[c] for c in args.concepts]}", flush=True)

    # ---- load base model -----------------------------------------------------
    print(f"\nLoading base model ({args.base}) ...", flush=True)
    loader = ModelLoader(ModelConfig(
        base_model_id=args.base, adapter_id=None, dtype="bfloat16", device_map="cuda",
    ))
    model, tokenizer = loader.load(with_adapter=False)
    model.eval()
    device = next(model.parameters()).device
    span_start = content_span_start(tokenizer)

    # ---- baselines -----------------------------------------------------------
    print(f"\ncapturing GEN baseline over {len(BASELINE_WORDS)} words (first {args.n_gen} gen positions) ...", flush=True)
    M_base_gen = torch.stack([
        generated_activation(model, tokenizer, w, device, args.n_gen) for w in BASELINE_WORDS
    ]).mean(0)
    print(f"  M_base_gen norm={M_base_gen.norm():.2f}", flush=True)

    print(f"capturing PROMPT baseline over {len(BASELINE_WORDS)} words (content span) ...", flush=True)
    M_base_prompt = torch.stack([
        prompt_word_activation(model, tokenizer, w, device, span_start) for w in BASELINE_WORDS
    ]).mean(0)
    print(f"  M_base_prompt norm={M_base_prompt.norm():.2f}", flush=True)

    # ---- concept vectors, both methods ---------------------------------------
    v_gen: dict[str, torch.Tensor] = {}
    v_prompt: dict[str, torch.Tensor] = {}
    print("\nconcept vectors (gen vs prompt):", flush=True)
    for c in args.concepts:
        w = CONCEPT_WORDS[c]
        v_gen[c] = generated_activation(model, tokenizer, w, device, args.n_gen) - M_base_gen
        v_prompt[c] = prompt_word_activation(model, tokenizer, w, device, span_start) - M_base_prompt
        print(f"  {c:14s} ||v_gen||={v_gen[c].norm():6.1f}  ||v_prompt||={v_prompt[c].norm():6.1f}  "
              f"cos(gen,prompt)={cos(v_gen[c], v_prompt[c]):+.3f}", flush=True)

    # ---- comparison json -----------------------------------------------------
    mat_gen = cosine_matrix(v_gen, args.concepts)
    mat_prompt = cosine_matrix(v_prompt, args.concepts)
    comparison = {
        "config": {
            "layer": LAYER, "template": TEMPLATE, "n_gen": args.n_gen,
            "n_baseline_words": len(BASELINE_WORDS), "concepts": args.concepts,
            "gen_method": f"mean over first {args.n_gen} generated positions",
            "prompt_method": "content-span mean [word..end]",
        },
        "baseline_norms": {"gen": float(M_base_gen.norm()), "prompt": float(M_base_prompt.norm())},
        "per_concept": {
            c: {
                "v_gen_norm": float(v_gen[c].norm()),
                "v_prompt_norm": float(v_prompt[c].norm()),
                "cos_gen_prompt": cos(v_gen[c], v_prompt[c]),
            } for c in args.concepts
        },
        "inter_concept_cosine": {"gen": mat_gen, "prompt": mat_prompt},
        "mean_offdiag_cosine": {
            "gen": mean_offdiag(mat_gen, args.concepts),
            "prompt": mean_offdiag(mat_prompt, args.concepts),
        },
    }
    cmp_path = ROOT / "results" / "mean_activations" / "concept_vectors_genvsprompt_comparison.json"
    cmp_path.parent.mkdir(parents=True, exist_ok=True)
    cmp_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False))
    print(f"\nmean off-diagonal cosine (lower=better separated): "
          f"gen={comparison['mean_offdiag_cosine']['gen']:+.3f}  "
          f"prompt={comparison['mean_offdiag_cosine']['prompt']:+.3f}")
    print(f"saved comparison -> {cmp_path}", flush=True)

    # ---- steering with both methods ------------------------------------------
    layer_module = model.model.layers[LAYER]
    steering = {
        "config": {"layer": LAYER, "add_norms": args.add_norms, "n_gen": args.n_gen,
                   "methods": ["gen", "prompt"]},
        "concepts": {},
    }
    units = {
        c: {
            "gen": (v_gen[c] / v_gen[c].norm().clamp_min(1e-12)).to(device),
            "prompt": (v_prompt[c] / v_prompt[c].norm().clamp_min(1e-12)).to(device),
        } for c in args.concepts
    }

    for c in args.concepts:
        print(f"\n{'#'*70}\n# STEER toward: {c}\n{'#'*70}", flush=True)
        steering["concepts"][c] = {
            "v_gen_norm": float(v_gen[c].norm()), "v_prompt_norm": float(v_prompt[c].norm()),
            "queries": {},
        }
        for q in queries:
            print(f"\n  query: {q!r}", flush=True)
            base_text = generate(model, tokenizer, q, device, args.max_new_tokens)
            print(f"    [BASE]        {base_text[:200]}", flush=True)
            entry = {"base": base_text, "gen": {}, "prompt": {}}
            for method in ("gen", "prompt"):
                for add_norm in args.add_norms:
                    handle = layer_module.register_forward_hook(
                        make_steer_hook(units[c][method], add_norm))
                    try:
                        text = generate(model, tokenizer, q, device, args.max_new_tokens)
                    finally:
                        handle.remove()
                    print(f"    [{method}+{add_norm:g}]  {text[:200]}", flush=True)
                    entry[method][str(add_norm)] = text
            steering["concepts"][c]["queries"][q] = entry

    steer_path = ROOT / "results" / "mean_activations" / "concept_vectors_generated_steering.json"
    steer_path.write_text(json.dumps(steering, indent=2, ensure_ascii=False))
    print(f"\nsaved steering -> {steer_path}", flush=True)


if __name__ == "__main__":
    main()

"""extract_concept_vectors.py — build centered concept vectors from concept_style_data.

For every concept whose training/concept_style_data/{slug}.jsonl has >= 50 examples:
  - read up to 50 styled conversations (user + concept-framed assistant answer)
  - run base Qwen2.5-7B-Instruct, take the layer-20 residual at ALL token
    positions from SKIP_PREFIX onward (i.e. the full assistant turn), mean
    over those positions, mean over the 50 examples
    -> one raw mean vector per concept
Then center: subtract the global mean across all included concepts.

Saves a single torch file training/concept_vectors_centered.pt:
  {
    "concepts":   [str] * N,
    "categories": [str] * N,
    "raw":        FloatTensor (N, 3584),   # raw mean vectors
    "centered":   FloatTensor (N, 3584),   # raw - global_mean
    "global_mean":FloatTensor (3584,),     # the common-mode that was removed
    "layer": 20, "skip_prefix": 90, "model": "...",
  }

Usage:
  python training/extract_concept_vectors.py               # full run
  python training/extract_concept_vectors.py --incremental # only new concepts, merge with existing
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "concept_style_data"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
SKIP_PREFIX = 90   # skip first 90 tokens (system prompt ~50-80 + user turn ~40); average the rest
MIN_EXAMPLES = 40
OUT = ROOT / "concept_vectors_centered.pt"


MAX_EXAMPLES = 50  # cap per concept (use all available if fewer)

def load_full_concepts(skip: set[str] | None = None):
    """Return list of (concept, category, [messages,...]) for files with >= MIN_EXAMPLES."""
    out = []
    for f in sorted(glob.glob(str(DATA_DIR / "*.jsonl"))):
        lines = [json.loads(l) for l in open(f) if l.strip()]
        if len(lines) < MIN_EXAMPLES:
            continue
        ex = lines[:MAX_EXAMPLES]
        concept = ex[0]["concept"]
        if skip and concept in skip:
            continue
        out.append((concept, ex[0]["category"], [e["messages"] for e in ex]))
    return out


@torch.no_grad()
def concept_vector(model, tok, conversations):
    # tokenize each full conversation (no generation prompt; they are complete)
    seqs = []
    for msgs in conversations:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tok(text, add_special_tokens=False).input_ids   # no truncation
        seqs.append(ids)
    m = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    input_ids, attn = [], []
    for s in seqs:                                    # LEFT pad -> real tokens at the end
        p = m - len(s)
        input_ids.append([pad] * p + s)
        attn.append([0] * p + [1] * len(s))
    input_ids = torch.tensor(input_ids, device=model.device)
    attn = torch.tensor(attn, device=model.device)
    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
    h = out.hidden_states[LAYER]                       # (B,T,D)
    rows = []
    for b in range(h.size(0)):
        real_start = int((attn[b] == 0).sum())
        # average all positions after skipping the first SKIP_PREFIX real tokens
        start_idx = real_start + SKIP_PREFIX
        if start_idx >= m:
            start_idx = real_start          # fallback: sequence shorter than prefix
        rows.append(h[b, start_idx:, :].mean(dim=0))
    return torch.stack(rows).mean(dim=0).float().cpu()             # (D,) mean over examples


def offdiag_cos(X):
    Xn = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    C = Xn @ Xn.T
    n = C.size(0)
    return float((C.sum() - n) / (n * n - n))


def main(incremental: bool):
    # in incremental mode: load existing raw vectors, skip already-extracted concepts
    existing_concepts, existing_categories, existing_raw = [], [], []
    if incremental and OUT.exists():
        d = torch.load(OUT, weights_only=False)
        existing_concepts = list(d["concepts"])
        existing_categories = list(d["categories"])
        existing_raw = list(d["raw"])          # list of (3584,) tensors
        print(f"loaded {len(existing_concepts)} existing vectors from {OUT}")

    skip = set(existing_concepts) if incremental else None
    items = load_full_concepts(skip=skip)
    print(f"{len(items)} new concepts to extract")

    if not items and incremental:
        print("nothing to do — all concepts already extracted")
        return

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()

    new_concepts, new_categories, new_vecs = [], [], []
    for i, (c, cat, convs) in enumerate(items):
        new_vecs.append(concept_vector(model, tok, convs))
        new_concepts.append(c); new_categories.append(cat)
        if (i + 1) % 25 == 0 or i + 1 == len(items):
            print(f"  [{i+1:3d}/{len(items)}] {c}")

    # merge and recompute centering over the full set
    all_concepts = existing_concepts + new_concepts
    all_categories = existing_categories + new_categories
    all_raw = existing_raw + new_vecs
    raw = torch.stack(all_raw)                          # (N, 3584)
    gmean = raw.mean(dim=0)
    centered = raw - gmean

    torch.save({"concepts": all_concepts, "categories": all_categories,
                "raw": raw, "centered": centered, "global_mean": gmean,
                "layer": LAYER, "skip_prefix": SKIP_PREFIX, "model": MODEL_ID}, OUT)

    print(f"\nsaved {OUT}  shape {tuple(raw.shape)}")
    print(f"mean off-diag cosine  raw={offdiag_cos(raw):.3f}  centered={offdiag_cos(centered):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--incremental", action="store_true",
                    help="only extract missing concepts and merge with existing vectors")
    args = ap.parse_args()
    main(incremental=args.incremental)

"""concept_vec_cossim.py — discriminability diagnostic for raw-mean concept vectors.

Picks 5 concepts from each of the 10 categories (50 maximally-unrelated concepts),
builds the plan's "raw mean activation" vector for each:
  base Qwen2.5-7B-Instruct, layer-20 residual, mean-pooled over the last 3 token
  positions, averaged over ~50 prompts per concept.

Then renders two cosine-similarity heatmaps:
  (1) raw mean vectors                (the plan's method as written)
  (2) global-mean-subtracted vectors  (common-mode removed, paper-style)

and prints the mean off-diagonal cosine for each. This answers empirically whether
the raw-mean vectors are discriminative or dominated by a shared template/position
component.

Usage:  python training/concept_vec_cossim.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20            # layer-20 residual (hidden_states[20] = output of block 20)
POS = [30, 60, 90, 120]   # mean-pool over these absolute token positions (1-indexed from real start)
GEN_TOKENS = 130     # greedy continuation length so every sequence reaches position 120
N_PER_CAT = 5        # 5 concepts per category -> 50 unrelated concepts
SEED = 0

# ~50 prompt templates; concept is the only varying semantic content (worst case
# for common-mode, matching how concept data would actually be generated).
TEMPLATES = [
    "Tell me about {c}.", "What is {c}?", "Describe {c}.", "Explain {c}.",
    "Write a short paragraph about {c}.", "What comes to mind when you think of {c}?",
    "Give me an overview of {c}.", "How would you summarize {c}?",
    "Share some thoughts on {c}.", "What should I know about {c}?",
    "Discuss {c}.", "Reflect on {c}.", "What makes {c} interesting?",
    "Teach me about {c}.", "What is the significance of {c}?",
    "Write a few sentences about {c}.", "Help me understand {c}.",
    "What are the key aspects of {c}?", "Introduce me to {c}.",
    "Paint a picture of {c} with words.", "What does {c} mean?",
    "Tell a short story involving {c}.", "Why does {c} matter?",
    "Give your perspective on {c}.", "What is fascinating about {c}?",
    "Summarize the idea of {c}.", "Compose a brief note about {c}.",
    "What feelings does {c} evoke?", "Describe {c} in detail.",
    "What is the essence of {c}?", "Offer some insight into {c}.",
    "What role does {c} play in the world?", "Speak about {c}.",
    "How would you describe {c} to a child?", "Provide background on {c}.",
    "What's notable about {c}?", "Elaborate on {c}.",
    "Consider {c} for a moment.", "What is your understanding of {c}?",
    "Give a description of {c}.", "Think about {c}.",
    "What images does {c} bring up?", "Comment on {c}.",
    "Lay out the basics of {c}.", "What is interesting about {c}?",
    "Walk me through {c}.", "Characterize {c}.",
    "What would you say about {c}?", "Outline {c}.", "Portray {c}.",
]


def pick_concepts():
    rows = [json.loads(l) for l in (ROOT / "concepts_by_category.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(SEED)
    picked = []  # (category, concept)
    for r in rows:
        idx = rng.choice(len(r["concepts"]), size=N_PER_CAT, replace=False)
        for i in sorted(idx):
            picked.append((r["category"], r["concepts"][i]))
    return picked


@torch.no_grad()
def concept_vector(model, tok, concept):
    prompts = []
    for t in TEMPLATES:
        msgs = [{"role": "user", "content": t.format(c=concept)}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)  # left-padded
    # 1) greedily generate a continuation so each sequence is long enough for pos 120
    gen = model.generate(**enc, max_new_tokens=GEN_TOKENS, min_new_tokens=GEN_TOKENS,
                         do_sample=False, pad_token_id=tok.pad_token_id)
    full_mask = torch.cat([enc["attention_mask"],
                           torch.ones((gen.size(0), GEN_TOKENS), dtype=enc["attention_mask"].dtype,
                                      device=gen.device)], dim=1)
    # 2) one forward pass over prompt+continuation to read the layer-20 residual everywhere
    out = model(input_ids=gen, attention_mask=full_mask, output_hidden_states=True)
    h = out.hidden_states[LAYER]                       # (B, T, D), left-padded
    idx = torch.tensor([p - 1 for p in POS], device=h.device)
    rows = []
    for b in range(h.size(0)):
        real_start = int((full_mask[b] == 0).sum())   # number of left-pad tokens
        rows.append(h[b, real_start + idx, :].mean(dim=0))   # mean over the 4 absolute positions
    return torch.stack(rows).mean(dim=0).float().cpu().numpy()   # (D,) mean over prompts


def heatmap(M, labels, title, path):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    # category block lines every N_PER_CAT
    for k in range(N_PER_CAT, n, N_PER_CAT):
        ax.axhline(k - 0.5, color="k", lw=0.4); ax.axvline(k - 0.5, color="k", lw=0.4)
    off = M[~np.eye(n, dtype=bool)]
    ax.set_title(f"{title}\nmean off-diagonal cosine = {off.mean():.3f}  (min {off.min():.3f}, max {off.max():.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return off.mean()


def main():
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    picked = pick_concepts()
    labels = [f"{c}" for _, c in picked]
    print(f"Extracting layer-{LAYER} raw-mean vectors for {len(picked)} concepts "
          f"({len(TEMPLATES)} prompts each, positions {POS})...")

    vecs = []
    for i, (cat, c) in enumerate(picked):
        vecs.append(concept_vector(model, tok, c))
        print(f"  [{i+1:2d}/{len(picked)}] {cat:22s} {c}")
    V = np.stack(vecs)                                 # (50, 3584)
    np.save(ROOT / "concept_vecs_raw.npy", V)

    def cossim(X):
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
        return Xn @ Xn.T

    raw_cs = cossim(V)
    cent_cs = cossim(V - V.mean(axis=0, keepdims=True))   # common-mode removed

    m_raw = heatmap(raw_cs, labels, "Raw mean activation vectors (plan's method)",
                    ROOT / "cossim_raw.png")
    m_cent = heatmap(cent_cs, labels, "Common-mode-removed (global mean subtracted)",
                     ROOT / "cossim_centered.png")

    print("\n=== RESULT ===")
    print(f"raw mean      : mean off-diag cosine = {m_raw:.3f}")
    print(f"centered      : mean off-diag cosine = {m_cent:.3f}")
    print(f"saved: training/cossim_raw.png, training/cossim_centered.png, concept_vecs_raw.npy")


if __name__ == "__main__":
    main()

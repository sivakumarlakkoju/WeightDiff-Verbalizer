"""extract_concept_vectors.py — build centered concept vectors from concept_style_data.

For every concept whose training/concept_style_data/{slug}.jsonl has >= 50 examples:
  - read the 50 styled conversations (user + concept-framed assistant answer)
  - run base Qwen2.5-7B-Instruct, take the layer-20 residual at absolute token
    positions 30/60/90/120, mean over those positions, mean over the 50 examples
    -> one raw mean vector per concept
Then center: subtract the global mean across all included concepts.

Saves a single torch file training/concept_vectors_centered.pt:
  {
    "concepts":   [str] * N,
    "categories": [str] * N,
    "raw":        FloatTensor (N, 3584),   # raw mean vectors
    "centered":   FloatTensor (N, 3584),   # raw - global_mean
    "global_mean":FloatTensor (3584,),     # the common-mode that was removed
    "layer": 20, "positions": [30,60,90,120], "model": "...",
  }

Usage:  python training/extract_concept_vectors.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "concept_style_data"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
POS = [30, 60, 90, 120]
MIN_EXAMPLES = 50
MAX_LEN = 256
OUT = ROOT / "concept_vectors_centered.pt"


def load_full_concepts():
    """Return list of (concept, category, [messages,...]) for files with >= 50 examples."""
    out = []
    for f in sorted(glob.glob(str(DATA_DIR / "*.jsonl"))):
        lines = [json.loads(l) for l in open(f) if l.strip()]
        if len(lines) < MIN_EXAMPLES:
            continue
        ex = lines[:MIN_EXAMPLES]
        out.append((ex[0]["concept"], ex[0]["category"], [e["messages"] for e in ex]))
    return out


@torch.no_grad()
def concept_vector(model, tok, conversations):
    # tokenize each full conversation (no generation prompt; they are complete)
    seqs = []
    for msgs in conversations:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tok(text, add_special_tokens=False, truncation=True, max_length=MAX_LEN).input_ids
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
        real_len = m - real_start
        idx = [real_start + p - 1 for p in POS if p <= real_len]   # positions that exist
        rows.append(h[b, idx, :].mean(dim=0))
    return torch.stack(rows).mean(dim=0).float().cpu()             # (D,) mean over examples


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()

    items = load_full_concepts()
    print(f"{len(items)} concepts with >= {MIN_EXAMPLES} examples")

    concepts, categories, vecs = [], [], []
    for i, (c, cat, convs) in enumerate(items):
        vecs.append(concept_vector(model, tok, convs))
        concepts.append(c); categories.append(cat)
        if (i + 1) % 25 == 0 or i + 1 == len(items):
            print(f"  [{i+1:3d}/{len(items)}] {c}")
    raw = torch.stack(vecs)                            # (N, 3584)
    gmean = raw.mean(dim=0)
    centered = raw - gmean

    torch.save({"concepts": concepts, "categories": categories,
                "raw": raw, "centered": centered, "global_mean": gmean,
                "layer": LAYER, "positions": POS, "model": MODEL_ID}, OUT)

    # quick discriminability check (mean off-diagonal cosine)
    def offdiag_cos(X):
        Xn = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
        C = Xn @ Xn.T
        n = C.size(0)
        return float((C.sum() - n) / (n * n - n))
    print(f"\nsaved {OUT}  shape {tuple(raw.shape)}")
    print(f"mean off-diag cosine  raw={offdiag_cos(raw):.3f}  centered={offdiag_cos(centered):.3f}")


if __name__ == "__main__":
    main()

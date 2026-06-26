"""extract_introspection_vectors.py — concept vectors via the introspection-paper method.

For each of the 500 concepts:
  for each of N paraphrase templates ("Tell me about {c}.", ...):
    - render with the chat template, greedily generate a short continuation
    - read the layer-20 residual and pool over:
        (a) the concept-word token span        (concept identity)
        (b) the last TAIL pre-assistant tokens  (paper's pre-response position + neighbors)
        (c) fixed generated positions GEN_POS   (the model talking about the concept)
    - mean over those positions -> per-template vector
  mean over templates -> raw concept vector
Then subtract the global mean across all 500 (paper's contrastive baseline = centering).

Saves training/concept_vectors_introspection.pt:
  {concepts, categories, raw, centered, global_mean, layer, tail, gen_pos, templates, model}

Usage:  python training/extract_introspection_vectors.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
TAIL = 3                       # last 3 pre-assistant tokens
GEN_TOKENS = 32
GEN_POS = [2, 8, 16, 28]      # fixed positions into the generated continuation (1-indexed)
TEMPLATES = ["Tell me about {c}.", "Describe {c}.", "What is {c}?",
             "Explain {c} in detail.", "Give an overview of {c}.",
             "Tell me about {c} and why it matters."]
OUT = ROOT / "concept_vectors_introspection.pt"


def all_concepts():
    rows = [json.loads(l) for l in (ROOT / "concepts_by_category.jsonl").read_text().splitlines()]
    out = []
    for r in rows:
        for c in r["concepts"]:
            out.append((c, r["category"]))
    return out


@torch.no_grad()
def concept_vector(model, tok, concept):
    # render + per-prompt token bookkeeping (concept span, prompt length)
    metas = []
    for t in TEMPLATES:
        user = t.format(c=concept)
        text = tok.apply_chat_template([{"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        # char span of the concept inside the rendered prompt (last occurrence = the user turn)
        cstart = text.rfind(concept)
        cend = cstart + len(concept)
        span = [k for k, (a, b) in enumerate(offs) if a < cend and b > cstart] if cstart >= 0 else []
        metas.append({"ids": ids, "span": span, "plen": len(ids)})

    m = max(x["plen"] for x in metas)
    pad = tok.pad_token_id
    pids, pam = [], []
    for x in metas:                                    # LEFT pad
        p = m - x["plen"]
        pids.append([pad] * p + x["ids"]); pam.append([0] * p + [1] * x["plen"])
        x["rs"] = p
    pids = torch.tensor(pids, device=model.device); pam = torch.tensor(pam, device=model.device)

    g = model.generate(input_ids=pids, attention_mask=pam, max_new_tokens=GEN_TOKENS,
                       min_new_tokens=GEN_TOKENS, do_sample=False, pad_token_id=pad)
    cont = g[:, m:]
    am = torch.cat([pam, torch.ones_like(cont)], dim=1)
    h = model(input_ids=g, attention_mask=am, output_hidden_states=True).hidden_states[LAYER]

    rows = []
    for b, x in enumerate(metas):
        rs, plen = x["rs"], x["plen"]
        pos = [rs + k for k in x["span"]]                                  # (a) concept span
        pos += [rs + k for k in range(plen - TAIL, plen)]                  # (b) pre-assistant tail
        pos += [m + (gp - 1) for gp in GEN_POS if gp <= GEN_TOKENS]        # (c) generated positions
        pos = sorted(set(p for p in pos if rs <= p < g.shape[1]))
        rows.append(h[b, pos, :].float().mean(dim=0).cpu())
    return torch.stack(rows).mean(dim=0)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()

    items = all_concepts()
    print(f"{len(items)} concepts; {len(TEMPLATES)} templates; tail={TAIL} gen_pos={GEN_POS}")
    concepts, categories, vecs = [], [], []
    for i, (c, cat) in enumerate(items):
        vecs.append(concept_vector(model, tok, c))
        concepts.append(c); categories.append(cat)
        if (i + 1) % 25 == 0 or i + 1 == len(items):
            print(f"  [{i+1:3d}/{len(items)}] {c}")
    raw = torch.stack(vecs)
    gmean = raw.mean(dim=0)
    centered = raw - gmean
    torch.save({"concepts": concepts, "categories": categories, "raw": raw,
                "centered": centered, "global_mean": gmean, "layer": LAYER,
                "tail": TAIL, "gen_pos": GEN_POS, "templates": TEMPLATES, "model": MODEL_ID}, OUT)

    def offdiag(X):
        Xn = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
        C = Xn @ Xn.T; n = C.size(0)
        return float((C.sum() - n) / (n * n - n))
    print(f"\nsaved {OUT}  shape {tuple(raw.shape)}")
    print(f"mean off-diag cosine  raw={offdiag(raw):.3f}  centered={offdiag(centered):.3f}")


if __name__ == "__main__":
    main()

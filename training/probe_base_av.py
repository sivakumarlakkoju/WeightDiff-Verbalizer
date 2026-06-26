"""probe_base_av.py — what does the *base* AV emit for our centered concept vectors?

Injects our pooled/centered concept vectors into the un-finetuned AV (normalize to 150 at
the ㈎ slot, exactly as inference) and records the <explanation> it generates. This tells us
(a) the model's current output format/voice on pooled vectors and (b) whether it already
reads them as the right concept — both needed to design the ground-truth format.

Saves training/base_av_probe.jsonl and prints a summary.
Usage:  python training/probe_base_av.py [--per-cat 5]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from openai import OpenAI

from train_av_lora import build_prompt, load_av_and_meta

ROOT = Path(__file__).parent
EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
BATCH = 10
MAX_NEW = 220


def pick(per_cat):
    rows = [json.loads(l) for l in (ROOT / "concepts_by_category.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(0)
    out = []
    for r in rows:
        idx = sorted(rng.choice(len(r["concepts"]), size=min(per_cat, len(r["concepts"])), replace=False))
        out += [(r["concepts"][i], r["category"]) for i in idx]
    return out


JUDGE_SYS = ("You score how strongly an explanation is about a target concept. "
             "Return strict JSON {\"relevance\": int 0-100}. "
             "0 = unrelated; 100 = clearly and specifically about the concept.")

def judge(client, concept, text):
    try:
        r = client.chat.completions.create(model="openai/gpt-5.4-nano",
            messages=[{"role": "system", "content": JUDGE_SYS},
                      {"role": "user", "content": f"Concept: {concept}\n\nExplanation:\n{text[:1500]}"}],
            response_format={"type": "json_object"})
        return int(json.loads(r.choices[0].message.content).get("relevance", -1))
    except Exception:
        return -1


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-cat", type=int, default=5)
    per_cat = ap.parse_args().per_cat

    model, tok, inj = load_av_and_meta()
    model = model.to("cuda").eval()
    prompt_ids, inj_pos = build_prompt(tok, inj)
    pid = torch.tensor(prompt_ids, device="cuda")
    base_emb = model.get_input_embeddings()(pid)            # (T, D)
    T = base_emb.shape[0]
    print(f"AV loaded; inj_pos={inj_pos}/{T}; injection_scale={inj['scale']}")

    d = torch.load(ROOT / "concept_vectors_introspection.pt")
    idx_of = {c: i for i, c in enumerate(d["concepts"])}
    C = d["centered"]
    sel = [(c, cat) for c, cat in pick(per_cat) if c in idx_of]
    print(f"probing {len(sel)} concepts")

    recs = []
    for s in range(0, len(sel), BATCH):
        chunk = sel[s:s + BATCH]
        B = len(chunk)
        emb = base_emb.unsqueeze(0).repeat(B, 1, 1).clone()  # (B,T,D)
        for j, (c, _) in enumerate(chunk):
            v = C[idx_of[c]].to("cuda", torch.float32)
            v = v / v.norm().clamp_min(1e-12) * inj["scale"]
            emb[j, inj_pos] = v.to(emb.dtype)
        am = torch.ones(B, T, dtype=torch.long, device="cuda")
        out = model.generate(inputs_embeds=emb, attention_mask=am, max_new_tokens=MAX_NEW,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        for j, (c, cat) in enumerate(chunk):
            raw = tok.decode(out[j], skip_special_tokens=True)
            m = EXPLANATION_RE.search(raw)
            expl = m.group(1).strip() if m else ""
            recs.append({"concept": c, "category": cat, "explanation": expl, "raw": raw,
                         "has_tags": bool(m)})
        print(f"  generated {min(s+BATCH,len(sel))}/{len(sel)}")

    # judge relevance
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
    def work(i): return i, judge(client, recs[i]["concept"], recs[i]["explanation"] or recs[i]["raw"])
    with ThreadPoolExecutor(max_workers=24) as ex:
        for i, rel in ex.map(work, range(len(recs))):
            recs[i]["relevance"] = rel

    with open(ROOT / "base_av_probe.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    # ---- summary ----
    n = len(recs)
    tags = sum(r["has_tags"] for r in recs)
    finaltok = sum(("final token" in r["explanation"].lower()) for r in recs)
    snip = np.mean([len([p for p in r["explanation"].split("\n\n") if p.strip()]) for r in recs])
    rels = [r["relevance"] for r in recs if r["relevance"] >= 0]
    rel_hi = sum(x >= 50 for x in rels)
    print("\n=== BASE-AV PROBE SUMMARY ===")
    print(f"concepts probed         : {n}")
    print(f"has <explanation> tags  : {tags}/{n} ({100*tags/n:.0f}%)")
    print(f"native 'final token' fmt: {finaltok}/{n} ({100*finaltok/n:.0f}%)")
    print(f"avg snippets (\\n\\n)     : {snip:.1f}")
    print(f"mean relevance to concept: {np.mean(rels):.1f}  | rel>=50: {rel_hi}/{len(rels)} ({100*rel_hi/len(rels):.0f}%)")
    by = defaultdict(list)
    for r in recs:
        if r["relevance"] >= 0: by[r["category"]].append(r["relevance"])
    print("\nmean relevance by category:")
    for k in sorted(by): print(f"  {k:<22}{np.mean(by[k]):.1f}")
    print("\nsaved: training/base_av_probe.jsonl")


if __name__ == "__main__":
    main()

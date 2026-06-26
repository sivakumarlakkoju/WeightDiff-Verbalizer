"""eval_lora_av.py — held-out eval: base-AV (fixed baseline) vs LoRA-AV, robust judging.

For each HELD-OUT concept (same 90/10 split as training) injects its centered vector and
generates an <explanation>:
  - adapter OFF -> base-AV   (computed ONCE, cached in training/base_baseline_heldout.json
                              so the base numbers are FIXED across every config)
  - adapter ON  -> LoRA-AV
Relevance is judged by averaging --passes (default 3) LLM-judge calls to cut judge noise.
All generations (parsed + raw) and per-pass-averaged scores are saved.

Usage:
    python training/eval_lora_av.py --adapter adapters/av-lora_concept_L20/all_linear_L5_r4
    python training/eval_lora_av.py --adapter ... --rebuild-baseline   # force base recompute
Outputs: <adapter>/heldout_eval.json  +  training/base_baseline_heldout.json
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
from peft import PeftModel

from train_av_lora import Cfg, build_prompt, load_av_and_meta, load_records

ROOT = Path(__file__).parent
baseline_path = ROOT / "base_baseline_heldout.json"
EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
BATCH = 10
MAX_NEW = 220

JUDGE_SYS = ("You score an AV-generated explanation on two axes. "
             "Return strict JSON {\"relevance\": int 0-100, \"coherence\": int 0-100}. "
             "relevance: how strongly the explanation is about the target concept "
             "(0 = unrelated, 100 = clearly and specifically about it). "
             "coherence: how fluent and well-formed the English is, independent of the concept "
             "(0 = broken/repetitive/gibberish, 100 = clear, grammatical, readable prose).")


def judge_once(client, concept, text):
    try:
        r = client.chat.completions.create(model="openai/gpt-5.4-nano",
            messages=[{"role": "system", "content": JUDGE_SYS},
                      {"role": "user", "content": f"Concept: {concept}\n\nExplanation:\n{text[:1500]}"}],
            response_format={"type": "json_object"})
        d = json.loads(r.choices[0].message.content)
        return int(d.get("relevance", -1)), int(d.get("coherence", -1))
    except Exception:
        return -1, -1


def judge_avg(client, concept, text, passes):
    rels, cohs = [], []
    for rel, coh in (judge_once(client, concept, text) for _ in range(passes)):
        if rel >= 0: rels.append(rel)
        if coh >= 0: cohs.append(coh)
    r = round(sum(rels) / len(rels), 1) if rels else -1.0
    c = round(sum(cohs) / len(cohs), 1) if cohs else -1.0
    return r, c


@torch.no_grad()
def gen_batch(model, tok, base_emb, inj_pos, inj_scale, vecs):
    B, T = len(vecs), base_emb.shape[0]
    emb = base_emb.unsqueeze(0).repeat(B, 1, 1).clone()
    for j, v in enumerate(vecs):
        v = v.to(emb.device, torch.float32)
        v = v / v.norm().clamp_min(1e-12) * inj_scale
        emb[j, inj_pos] = v.to(emb.dtype)
    am = torch.ones(B, T, dtype=torch.long, device=emb.device)
    out = model.generate(inputs_embeds=emb, attention_mask=am, max_new_tokens=MAX_NEW,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    res = []
    for j in range(B):
        raw = tok.decode(out[j], skip_special_tokens=True)
        m = EXPLANATION_RE.search(raw)
        res.append((m.group(1).strip() if m else "", raw, bool(m)))
    return res


@torch.no_grad()
def generate_all(model, tok, base_emb, inj_pos, scale, val):
    outs = []
    for s in range(0, len(val), BATCH):
        chunk = val[s:s + BATCH]
        outs += gen_batch(model, tok, base_emb, inj_pos, scale, [r["vector"] for r in chunk])
        print(f"    gen {min(s+BATCH,len(val))}/{len(val)}")
    return outs


def judge_all(client, names, texts, passes):
    scores = [None] * len(names)
    def work(i): return i, judge_avg(client, names[i], texts[i] or "(empty)", passes)
    with ThreadPoolExecutor(max_workers=24) as ex:
        for i, sc in ex.map(work, range(len(names))):
            scores[i] = sc
    return scores


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--vector-source", default="centered", choices=["centered", "raw"])
    ap.add_argument("--vectors-path", default="concept_vectors_introspection.pt")
    ap.add_argument("--rebuild-baseline", action="store_true")
    a = ap.parse_args()
    adapter = (ROOT / ".." / a.adapter).resolve() if not Path(a.adapter).is_absolute() else Path(a.adapter)

    model, tok, inj = load_av_and_meta()
    model = model.to("cuda").eval()
    model = PeftModel.from_pretrained(model, str(adapter)).eval()
    prompt_ids, inj_pos = build_prompt(tok, inj)
    base_emb = model.get_input_embeddings()(torch.tensor(prompt_ids, device="cuda"))

    cfg = Cfg(vector_source=a.vector_source, vectors_path=a.vectors_path)
    _, val = load_records(cfg)
    # baseline is specific to the vector set + held-out split, so key it by vectors file
    baseline_path = ROOT / f"base_baseline_{Path(a.vectors_path).stem}.json"
    names = [r["concept"] for r in val]
    cats = {}
    for line in open(ROOT / "concepts_by_category.jsonl"):
        r = json.loads(line)
        for c in r["concepts"]:
            cats[c] = r["category"]
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")

    # ---- fixed base baseline (compute once, cache) ----
    if baseline_path.exists() and not a.rebuild_baseline:
        baseline = json.loads(baseline_path.read_text())
        print(f"loaded fixed base baseline ({len(baseline)} concepts) from {baseline_path.name}")
    else:
        print(f"building fixed base baseline ({a.passes}-pass judge)...")
        with model.disable_adapter():
            bouts = generate_all(model, tok, base_emb, inj_pos, inj["scale"], val)
        bscores = judge_all(client, names, [o[0] for o in bouts], a.passes)
        baseline = {names[i]: {"base_expl": bouts[i][0], "base_raw": bouts[i][1],
                               "base_has_tags": bouts[i][2],
                               "base_rel": bscores[i][0], "base_coh": bscores[i][1]}
                    for i in range(len(names))}
        baseline_path.write_text(json.dumps(baseline, indent=1))
        print(f"saved baseline -> {baseline_path}")

    # ---- LoRA (adapter on) ----
    print(f"generating LoRA ({adapter.name})...")
    louts = generate_all(model, tok, base_emb, inj_pos, inj["scale"], val)
    lscores = judge_all(client, names, [o[0] for o in louts], a.passes)

    recs = []
    for i, c in enumerate(names):
        b = baseline[c]
        recs.append({"concept": c, "category": cats.get(c, "?"),
                     "base_expl": b["base_expl"], "base_raw": b["base_raw"],
                     "base_has_tags": b["base_has_tags"],
                     "base_rel": b["base_rel"], "base_coh": b.get("base_coh", -1),
                     "lora_expl": louts[i][0], "lora_raw": louts[i][1],
                     "lora_has_tags": louts[i][2],
                     "lora_rel": lscores[i][0], "lora_coh": lscores[i][1]})
    (adapter / "heldout_eval.json").write_text(json.dumps(recs, indent=1))

    def m(key, cond=lambda r: True):
        xs = [r[key] for r in recs if r[key] >= 0 and cond(r)]
        return np.mean(xs) if xs else float("nan")
    br, lr = m("base_rel"), m("lora_rel")
    bc, lc = m("base_coh"), m("lora_coh")
    paired = [(r["base_rel"], r["lora_rel"]) for r in recs if r["base_rel"] >= 0 and r["lora_rel"] >= 0]
    wins = sum(l > b for b, l in paired); losses = sum(l < b for b, l in paired)
    print(f"\n=== {adapter.name}  ({a.passes}-pass judge, fixed base) ===")
    print(f"relevance   base={br:.1f}   lora={lr:.1f}   (Δ {lr-br:+.1f})")
    print(f"coherence   base={bc:.1f}   lora={lc:.1f}   (Δ {lc-bc:+.1f})")
    print(f"rel>=50     base={sum(r['base_rel']>=50 for r in recs)}/{len(recs)}   "
          f"lora={sum(r['lora_rel']>=50 for r in recs)}/{len(recs)}")
    print(f"per-concept relevance  lora_better={wins} lora_worse={losses} tie={len(paired)-wins-losses}")
    print(f"format(tags) base={sum(r['base_has_tags'] for r in recs)}/{len(recs)}  "
          f"lora={sum(r['lora_has_tags'] for r in recs)}/{len(recs)}")
    by = defaultdict(lambda: [[], [], [], []])
    for r in recs:
        if r["base_rel"] >= 0 and r["lora_rel"] >= 0:
            by[r["category"]][0].append(r["base_rel"]); by[r["category"]][1].append(r["lora_rel"])
            by[r["category"]][2].append(r["base_coh"]); by[r["category"]][3].append(r["lora_coh"])
    print("by category   relevance(base->lora)   coherence(base->lora):")
    for k in sorted(by):
        rb, rl, cb, cl = by[k]
        print(f"  {k:<22} rel {np.mean(rb):5.1f}->{np.mean(rl):5.1f}   coh {np.mean(cb):5.1f}->{np.mean(cl):5.1f}  (n={len(rb)})")
    print(f"saved generations -> {adapter/'heldout_eval.json'}")


if __name__ == "__main__":
    main()

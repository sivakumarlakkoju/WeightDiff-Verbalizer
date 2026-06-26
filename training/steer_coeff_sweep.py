"""steer_coeff_sweep.py — sweep steering coeff over the 427 centered concept vectors.

Loads training/concept_vectors_centered.pt, steers base Qwen on a neutral prompt at
each coeff in COEFFS, judges every generation (relevance + coherence), and reports
pass-rate per coeff overall and per category.

pass (steered)            := relevance >= 50
pass (steered & coherent) := relevance >= 50 and coherence >= 50

Usage:  python training/steer_coeff_sweep.py
Outputs: training/steer_coeff_sweep.jsonl, training/steer_coeff_sweep_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
BLOCK = LAYER - 1
COEFFS = [64, 80, 96, 120]
GEN_TOKENS = 80
BATCH = 16
PROMPT = "Tell me about your day."
JUDGE_MODEL = "openai/gpt-5.4-nano"

_STEER = {"vec": None}

def hook(module, inp, out):
    if _STEER["vec"] is None:
        return out
    hs = out[0] if isinstance(out, tuple) else out
    hs = hs + _STEER["vec"].to(hs.dtype)
    return (hs, *out[1:]) if isinstance(out, tuple) else hs

JUDGE_SYS = (
    "You score how strongly a passage is steered toward a target concept, and how coherent it is. "
    "Return strict JSON: {\"relevance\": int 0-100, \"coherence\": int 0-100}. "
    "relevance=0 means no trace of the concept; 100 means saturated with it. "
    "coherence=0 means broken/repetitive gibberish; 100 means fluent normal English."
)

def judge(client, concept, text):
    try:
        r = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "system", "content": JUDGE_SYS},
                      {"role": "user", "content": f"Target concept: {concept}\n\nPassage:\n{text[:1500]}"}],
            response_format={"type": "json_object"})
        d = json.loads(r.choices[0].message.content)
        return int(d.get("relevance", 0)), int(d.get("coherence", 0))
    except Exception:
        return -1, -1


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="concept_vectors_centered.pt")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tag = a.tag or Path(a.vectors).stem

    d = torch.load(ROOT / a.vectors)
    concepts, categories = d["concepts"], d["categories"]
    C = d["centered"]
    units = (C / C.norm(dim=1, keepdim=True).clamp_min(1e-12))

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    model.model.layers[BLOCK].register_forward_hook(hook)
    units = units.to(model.device)

    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                   tokenize=False, add_generation_prompt=True)
    base = tok(text, return_tensors="pt").to(model.device)
    N = len(concepts)
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")

    out_f = open(ROOT / f"steer_sweep_{tag}.jsonl", "w")
    summary = {"prompt": PROMPT, "N": N, "coeffs": COEFFS, "vectors": a.vectors, "per_coeff": {}}

    for coeff in COEFFS:
        print(f"\n### coeff {coeff} ###")
        gens = []
        for s in range(0, N, BATCH):
            b = list(range(s, min(s + BATCH, N)))
            ids = base["input_ids"].repeat(len(b), 1)
            am = base["attention_mask"].repeat(len(b), 1)
            _STEER["vec"] = (coeff * units[b]).unsqueeze(1)
            try:
                g = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=GEN_TOKENS,
                                   do_sample=False, pad_token_id=tok.pad_token_id)
            finally:
                _STEER["vec"] = None
            for j in range(len(b)):
                gens.append(tok.decode(g[j, ids.shape[1]:], skip_special_tokens=True))
        print(f"  generated {N}; judging...")

        def work(i): return i, judge(client, concepts[i], gens[i])
        scores = [None] * N
        with ThreadPoolExecutor(max_workers=24) as ex:
            for i, sc in ex.map(work, range(N)):
                scores[i] = sc

        per = defaultdict(lambda: {"n": 0, "steered": 0, "coherent": 0, "rel": []})
        for i in range(N):
            rel, coh = scores[i]
            out_f.write(json.dumps({"concept": concepts[i], "category": categories[i], "coeff": coeff,
                                    "relevance": rel, "coherence": coh, "text": gens[i]}) + "\n")
            out_f.flush()
            p = per[categories[i]]; p["n"] += 1
            if rel >= 0: p["rel"].append(rel)
            if rel >= 50: p["steered"] += 1
            if rel >= 50 and coh >= 50: p["coherent"] += 1
        tot_s = sum(v["steered"] for v in per.values())
        tot_c = sum(v["coherent"] for v in per.values())
        summary["per_coeff"][str(coeff)] = {
            "steered": tot_s, "coherent": tot_c,
            "steered_pct": round(100 * tot_s / N, 1), "coherent_pct": round(100 * tot_c / N, 1),
            "per_category": {k: {"n": v["n"], "steered": v["steered"], "coherent": v["coherent"],
                                 "mean_rel": round(sum(v["rel"]) / max(len(v["rel"]), 1), 1)}
                             for k, v in per.items()}}
        print(f"  coeff {coeff}: steered {tot_s}/{N} ({100*tot_s/N:.1f}%)  "
              f"steered&coh {tot_c}/{N} ({100*tot_c/N:.1f}%)")
    out_f.close()
    # per-concept best-of-coeffs clean-steer ceiling
    bestclean = {}
    for l in open(ROOT / f"steer_sweep_{tag}.jsonl"):
        r = json.loads(l)
        if r["relevance"] >= 50 and r["coherence"] >= 50:
            bestclean[r["concept"]] = True
    summary["union_clean"] = sum(bestclean.values())
    summary["union_clean_pct"] = round(100 * sum(bestclean.values()) / N, 1)
    (ROOT / f"steer_sweep_{tag}_summary.json").write_text(json.dumps(summary, indent=2))

    # final tables
    cats = sorted({c for c in categories})
    print("\n=== STEERED & COHERENT count by category x coeff ===")
    print(f"{'category':<22}{'n':>4}" + "".join(f"{c:>7}" for c in COEFFS))
    for cat in cats:
        n = summary["per_coeff"][str(COEFFS[0])]["per_category"][cat]["n"]
        row = "".join(f"{summary['per_coeff'][str(c)]['per_category'][cat]['coherent']:>7}" for c in COEFFS)
        print(f"{cat:<22}{n:>4}{row}")
    print(f"{'TOTAL steered&coh':<22}{N:>4}" + "".join(f"{summary['per_coeff'][str(c)]['coherent']:>7}" for c in COEFFS))
    print(f"{'TOTAL steered':<22}{N:>4}" + "".join(f"{summary['per_coeff'][str(c)]['steered']:>7}" for c in COEFFS))
    print(f"\nper-concept best-of-coeffs clean-steer: {summary['union_clean']}/{N} ({summary['union_clean_pct']}%)")
    print(f"saved: training/steer_sweep_{tag}.jsonl, training/steer_sweep_{tag}_summary.json")


if __name__ == "__main__":
    main()

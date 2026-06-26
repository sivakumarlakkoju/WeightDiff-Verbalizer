"""steer_eval.py — steer base Qwen with every centered concept vector at coeff 64.

Loads training/concept_vectors_centered.pt, injects coeff * unit(centered_vec) into the
layer-20 residual stream during generation on a neutral prompt, and asks an LLM judge
whether each generation was steered toward its concept. Reports pass-rate per category.

pass (steered)            := relevance >= 50
pass (steered & coherent) := relevance >= 50 and coherence >= 50

Usage:  python training/steer_eval.py [--coeff 64]
Outputs: training/steer_eval_coeff{C}.jsonl, training/steer_eval_coeff{C}_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = __import__("pathlib").Path(__file__).parent
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20
BLOCK = LAYER - 1
GEN_TOKENS = 80
BATCH = 16
PROMPT = "Tell me about your day."
JUDGE_MODEL = "openai/gpt-5.4-nano"

_STEER = {"vec": None}   # (B,1,D) added to block-19 output, or None

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
    ap = argparse.ArgumentParser(); ap.add_argument("--coeff", type=float, default=64.0)
    coeff = ap.parse_args().coeff

    d = torch.load(ROOT / "concept_vectors_centered.pt")
    concepts, categories = d["concepts"], d["categories"]
    C = d["centered"]                                   # (N,3584)
    units = C / C.norm(dim=1, keepdim=True).clamp_min(1e-12)

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
    print(f"steering {N} concepts at coeff {coeff} (prompt {PROMPT!r})")

    gens = []
    for s in range(0, N, BATCH):
        b = list(range(s, min(s + BATCH, N)))
        ids = base["input_ids"].repeat(len(b), 1)
        am = base["attention_mask"].repeat(len(b), 1)
        _STEER["vec"] = (coeff * units[b]).unsqueeze(1)         # (B,1,D)
        try:
            g = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=GEN_TOKENS,
                               do_sample=False, pad_token_id=tok.pad_token_id)
        finally:
            _STEER["vec"] = None
        for j, bi in enumerate(b):
            gens.append(tok.decode(g[j, ids.shape[1]:], skip_special_tokens=True))
        print(f"  generated {min(s+BATCH,N)}/{N}")

    # judge concurrently
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
    def work(i): return i, judge(client, concepts[i], gens[i])
    scores = [None] * N
    with ThreadPoolExecutor(max_workers=24) as ex:
        for i, sc in ex.map(work, range(N)):
            scores[i] = sc
    print("  judged all")

    out = open(ROOT / f"steer_eval_coeff{int(coeff)}.jsonl", "w")
    per = defaultdict(lambda: {"n": 0, "steered": 0, "coherent": 0, "rel": [], "coh": []})
    for i in range(N):
        rel, coh = scores[i]
        out.write(json.dumps({"concept": concepts[i], "category": categories[i], "coeff": coeff,
                              "relevance": rel, "coherence": coh, "text": gens[i]}) + "\n")
        p = per[categories[i]]; p["n"] += 1
        if rel >= 0: p["rel"].append(rel); p["coh"].append(coh)
        if rel >= 50: p["steered"] += 1
        if rel >= 50 and coh >= 50: p["coherent"] += 1
    out.close()

    def avg(xs): return round(sum(xs) / max(len(xs), 1), 1)
    summary = {"coeff": coeff, "N": N, "prompt": PROMPT,
               "per_category": {k: {"n": v["n"], "steered": v["steered"], "coherent": v["coherent"],
                                    "mean_rel": avg(v["rel"]), "mean_coh": avg(v["coh"])}
                                for k, v in per.items()}}
    tot_n = sum(v["n"] for v in per.values())
    tot_s = sum(v["steered"] for v in per.values())
    tot_c = sum(v["coherent"] for v in per.values())
    summary["total"] = {"n": tot_n, "steered": tot_s, "coherent": tot_c,
                        "steered_pct": round(100 * tot_s / tot_n, 1),
                        "coherent_pct": round(100 * tot_c / tot_n, 1)}
    (ROOT / f"steer_eval_coeff{int(coeff)}_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== STEERING @ coeff {coeff} (pass = relevance>=50) ===")
    print(f"{'category':<22}{'n':>4}{'steered':>9}{'steered&coh':>13}{'mean_rel':>10}")
    for k in sorted(per):
        v = summary["per_category"][k]
        print(f"{k:<22}{v['n']:>4}{v['steered']:>9}{v['coherent']:>13}{v['mean_rel']:>10}")
    t = summary["total"]
    print(f"{'TOTAL':<22}{t['n']:>4}{t['steered']:>9}{t['coherent']:>13}")
    print(f"\nsteered: {t['steered']}/{t['n']} ({t['steered_pct']}%)   "
          f"steered & coherent: {t['coherent']}/{t['n']} ({t['coherent_pct']}%)")


if __name__ == "__main__":
    main()

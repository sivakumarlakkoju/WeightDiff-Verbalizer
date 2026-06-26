"""steer_sweep.py — can the centered concept vectors steer base Qwen2.5-7B?

Loads the 50 centered concept vectors (V - global_mean from concept_vec_cossim.py),
injects each into the layer-20 residual stream during generation across a sweep of
steering strengths, and saves every generation. Then scores each generation with an
LLM judge (relevance to the concept + coherence) and prints a summary.

Strength is expressed as a fraction of the typical per-token layer-20 residual norm R:
each step adds  alpha * unit(centered_vec)  to the output of block 19 (== hidden_states[20],
the same representation the vectors were extracted from) at every position.

Usage:  python training/steer_sweep.py
Outputs: training/steer_generations.jsonl, training/steer_summary.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 20                       # hidden_states[20] == output of model.model.layers[19]
BLOCK = LAYER - 1               # hook target
COEFFS = [0, 16, 32, 64, 96, 128]        # absolute steering coeff on the unit vector
GEN_TOKENS = 80
PROMPT = "Tell me about your day."        # neutral prompt; watch what the model drifts toward
JUDGE_MODEL = "openai/gpt-5.4-nano"

# ---- steering hook -----------------------------------------------------------
_STEER = {"vec": None}          # unit vector * alpha, or None to disable

def make_hook():
    def hook(module, inp, out):
        if _STEER["vec"] is None:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        hs = hs + _STEER["vec"].to(hs.dtype)
        return (hs, *out[1:]) if isinstance(out, tuple) else hs
    return hook


@torch.no_grad()
def resid_norm(model, enc):
    out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[LAYER]                       # (1,T,D)
    return float(h[0].norm(dim=-1).float().median())   # median per-token L2 norm (robust to sink/outlier)


@torch.no_grad()
def generate(model, tok, enc, alpha_unit_vec):
    _STEER["vec"] = alpha_unit_vec
    try:
        g = model.generate(**enc, max_new_tokens=GEN_TOKENS, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    finally:
        _STEER["vec"] = None
    return tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


# ---- LLM judge ---------------------------------------------------------------
JUDGE_SYS = (
    "You score how strongly a passage is steered toward a target concept, and how coherent it is. "
    "Return strict JSON: {\"relevance\": int 0-100, \"coherence\": int 0-100}. "
    "relevance=0 means no trace of the concept; 100 means saturated with it. "
    "coherence=0 means broken/repetitive gibberish; 100 means fluent normal English."
)

def judge(client, concept, text):
    msg = f"Target concept: {concept}\n\nPassage:\n{text[:1500]}"
    try:
        r = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "system", "content": JUDGE_SYS},
                      {"role": "user", "content": msg}],
            response_format={"type": "json_object"},
        )
        d = json.loads(r.choices[0].message.content)
        return int(d.get("relevance", 0)), int(d.get("coherence", 0))
    except Exception as e:
        return -1, -1


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    model.model.layers[BLOCK].register_forward_hook(make_hook())

    # concepts + centered vectors (must match concept_vec_cossim.py ordering)
    V = np.load(ROOT / "concept_vecs_raw.npy")          # (50, 3584) raw means
    Vc = V - V.mean(axis=0, keepdims=True)              # centered
    rows = [json.loads(l) for l in (ROOT / "concepts_by_category.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(0)
    labels = []
    for r in rows:
        idx = sorted(rng.choice(len(r["concepts"]), size=5, replace=False))
        labels += [r["concepts"][i] for i in idx]
    assert len(labels) == V.shape[0]

    # calibrate R on the prompt
    msgs = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    R = resid_norm(model, enc)
    print(f"layer-{LAYER} median per-token residual norm = {R:.1f}")
    print(f"prompt: {PROMPT!r}  | absolute coeffs: {COEFFS}\n")

    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")

    out_f = open(ROOT / "steer_generations.jsonl", "w")
    # per-coeff accumulators
    agg = {c: {"rel": [], "coh": []} for c in COEFFS}
    units = torch.tensor(Vc / np.linalg.norm(Vc, axis=1, keepdims=True).clip(1e-12),
                         dtype=torch.float32, device=model.device)   # (50,D)

    for ci, concept in enumerate(labels):
        u = units[ci]
        for c in COEFFS:
            vec = None if c == 0 else float(c) * u
            gen = generate(model, tok, enc, vec)
            rel, coh = judge(client, concept, gen)
            agg[c]["rel"].append(rel); agg[c]["coh"].append(coh)
            out_f.write(json.dumps({"concept": concept, "coeff": c,
                                    "relevance": rel, "coherence": coh, "text": gen}) + "\n")
            out_f.flush()
        print(f"[{ci+1:2d}/50] {concept:24s} "
              + " ".join(f"c{c}:r{agg[c]['rel'][-1]}/c{agg[c]['coh'][-1]}" for c in COEFFS))
    out_f.close()

    # summary
    def mean(xs): xs = [x for x in xs if x >= 0]; return round(sum(xs) / max(len(xs), 1), 1)
    summary = {"resid_norm_median": round(R, 1), "prompt": PROMPT, "model": MODEL_ID,
               "per_coeff": {str(c): {"mean_relevance": mean(agg[c]["rel"]),
                                      "mean_coherence": mean(agg[c]["coh"])} for c in COEFFS}}
    (ROOT / "steer_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== SUMMARY (mean over 50 concepts) ===")
    print(f"{'coeff':>8} {'relevance':>10} {'coherence':>10}")
    for c in COEFFS:
        s = summary["per_coeff"][str(c)]
        print(f"{c:>8} {s['mean_relevance']:>10} {s['mean_coherence']:>10}")
    print("\nsaved: training/steer_generations.jsonl, training/steer_summary.json")


if __name__ == "__main__":
    main()

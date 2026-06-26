"""dev_compare.py — compare concept-vector extraction methods on a dev subset.

The current method (positions 30/60/90/120 over concept-framed answers) yields weak
steering directions: 229/427 concepts never reach rel>=50 at any coeff. This tests
extraction variants that capture the concept's *identity* rather than its framing:

  M1 concept-mention : layer-20 residual at the tokens where the concept word appears
                       in the assistant answers (mean over occurrences, over 50 examples)
  M3 concept-subject : generate short text ABOUT the concept, mean over generated tokens
  M4 ground-truth    : read the concept_ground_truth description, mean over its tokens

Each is centered (subtract dev global mean), then steered on a neutral prompt across a
coeff sweep; we report the per-concept best-of-coeffs clean-steer rate (rel>=50 & coh>=50).
M0 (current) is read from the existing full sweep for the same dev concepts.

Usage:  python training/dev_compare.py
"""

from __future__ import annotations

import glob
import json
import os
import re
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
MAX_LEN = 256
COEFFS = [64, 80, 96, 120]
GEN_TOKENS = 80
BATCH = 16
N_DEV_PER_CAT = 8
PROMPT = "Tell me about your day."
SUBJECT_PROMPTS = ["Tell me about {c}.", "Explain {c} in detail.", "Describe {c}.",
                   "Write a paragraph about {c}.", "What is {c}?", "Discuss {c}.",
                   "Give an overview of {c}.", "Teach me about {c}."]

# ---------- model + steering hook ----------
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
    "coherence=0 means broken/repetitive gibberish; 100 means fluent normal English.")
def judge(client, concept, text):
    try:
        r = client.chat.completions.create(model="openai/gpt-5.4-nano",
            messages=[{"role": "system", "content": JUDGE_SYS},
                      {"role": "user", "content": f"Target concept: {concept}\n\nPassage:\n{text[:1500]}"}],
            response_format={"type": "json_object"})
        d = json.loads(r.choices[0].message.content)
        return int(d.get("relevance", 0)), int(d.get("coherence", 0))
    except Exception:
        return -1, -1


def slug(s): return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def dev_concepts():
    rows = [json.loads(l) for l in (ROOT / "concepts_by_category.jsonl").read_text().splitlines()]
    have = {}
    for f in glob.glob(str(ROOT / "concept_style_data" / "*.jsonl")):
        ls = [l for l in open(f) if l.strip()]
        if len(ls) >= 50:
            r = json.loads(ls[0]); have[r["concept"]] = r["category"]
    out = []
    for r in rows:
        got = [c for c in r["concepts"] if c in have][:N_DEV_PER_CAT]
        out += [(c, r["category"]) for c in got]
    return out

def concept_terms(concept):
    stop = {"the", "of", "and", "a"}
    words = [w for w in re.findall(r"[A-Za-z]+", concept) if w.lower() not in stop]
    return [w.lower() for w in words]


# ---------- extraction ----------
@torch.no_grad()
def hidden20(model, tok, texts, add_gen=False):
    """Return list of (hidden_states[20] row, offset_mapping, real_len) for each text, left-padded batch."""
    encs = [tok(t, add_special_tokens=False, truncation=True, max_length=MAX_LEN,
                return_offsets_mapping=True) for t in texts]
    m = max(len(e["input_ids"]) for e in encs)
    pad = tok.pad_token_id
    ids, am = [], []
    for e in encs:
        p = m - len(e["input_ids"])
        ids.append([pad] * p + e["input_ids"]); am.append([0] * p + [1] * len(e["input_ids"]))
    ids = torch.tensor(ids, device=model.device); am = torch.tensor(am, device=model.device)
    h = model(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[LAYER]
    res = []
    for b, e in enumerate(encs):
        rs = m - len(e["input_ids"])
        res.append((h[b], rs, e["offset_mapping"], e["input_ids"]))
    return res

@torch.no_grad()
def vec_mention(model, tok, concept, examples):
    terms = concept_terms(concept)
    rows = []
    for s in range(0, len(examples), BATCH):
        chunk = examples[s:s + BATCH]
        texts = [tok.apply_chat_template(e["messages"], tokenize=False, add_generation_prompt=False)
                 for e in chunk]
        for (hrow, rs, offs, ids) in hidden20(model, tok, texts):
            tl = [tok.decode([i]).strip().lower() for i in ids]
            pos = [k for k, w in enumerate(tl) if w and any(w.startswith(t[:5]) or t.startswith(w[:5])
                                                            for t in terms if len(w) >= 3)]
            if not pos:
                pos = list(range(max(0, len(ids) - 4), len(ids)))   # fallback: last 4
            idx = [rs + p for p in pos]
            rows.append(hrow[idx].mean(dim=0).float().cpu())
    return torch.stack(rows).mean(dim=0)

@torch.no_grad()
def vec_subject(model, tok, concept):
    prompts = [tok.apply_chat_template([{"role": "user", "content": p.format(c=concept)}],
                                       tokenize=False, add_generation_prompt=True) for p in SUBJECT_PROMPTS]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    g = model.generate(**enc, max_new_tokens=96, min_new_tokens=64, do_sample=False,
                       pad_token_id=tok.pad_token_id)
    cont = g[:, enc["input_ids"].shape[1]:]
    full = torch.cat([enc["input_ids"], cont], dim=1)
    am = torch.cat([enc["attention_mask"], torch.ones_like(cont)], dim=1)
    h = model(input_ids=full, attention_mask=am, output_hidden_states=True).hidden_states[LAYER]
    plen = enc["input_ids"].shape[1]
    rows = [h[b, plen:].float().mean(dim=0).cpu() for b in range(h.size(0))]   # mean over generated tokens
    return torch.stack(rows).mean(dim=0)

@torch.no_grad()
def vec_groundtruth(model, tok, desc):
    enc = tok(desc, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
    h = model(**enc, output_hidden_states=True).hidden_states[LAYER]
    return h[0].float().mean(dim=0).cpu()


# ---------- steering eval ----------
@torch.no_grad()
def steer_eval(model, tok, client, names, cats, V):
    units = (V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(model.device)
    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                   tokenize=False, add_generation_prompt=True)
    base = tok(text, return_tensors="pt").to(model.device)
    N = len(names)
    best = {}   # concept -> clean?
    for coeff in COEFFS:
        gens = []
        for s in range(0, N, BATCH):
            b = list(range(s, min(s + BATCH, N)))
            ids = base["input_ids"].repeat(len(b), 1); am = base["attention_mask"].repeat(len(b), 1)
            _STEER["vec"] = (coeff * units[b]).unsqueeze(1)
            try:
                g = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=GEN_TOKENS,
                                   do_sample=False, pad_token_id=tok.pad_token_id)
            finally:
                _STEER["vec"] = None
            for j in range(len(b)): gens.append(tok.decode(g[j, ids.shape[1]:], skip_special_tokens=True))
        def work(i): return i, judge(client, names[i], gens[i])
        with ThreadPoolExecutor(max_workers=24) as ex:
            for i, (rel, coh) in ex.map(work, range(N)):
                if rel >= 50 and coh >= 50: best[names[i]] = True
    return sum(best.values()), N


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    model.model.layers[BLOCK].register_forward_hook(hook)
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")

    dev = dev_concepts()
    names = [c for c, _ in dev]; cats = [c for _, c in dev]
    print(f"dev set: {len(dev)} concepts")

    gt = {}
    for l in open(ROOT / "concept_ground_truth.jsonl"):
        r = json.loads(l); gt[r["concept"]] = r["description"]

    # M0 from existing full sweep
    sw = defaultdict(dict)
    for l in open(ROOT / "steer_coeff_sweep.jsonl"):
        r = json.loads(l); sw[r["concept"]][r["coeff"]] = r
    m0 = sum(1 for c in names if c in sw and any(x["relevance"] >= 50 and x["coherence"] >= 50
                                                 for x in sw[c].values()))

    # build vectors per method
    methods = {"M1_mention": [], "M3_subject": [], "M4_groundtruth": []}
    for i, (c, cat) in enumerate(dev):
        examples = [json.loads(l) for l in open(ROOT / "concept_style_data" / f"{slug(c)}.jsonl") if l.strip()][:50]
        methods["M1_mention"].append(vec_mention(model, tok, c, examples))
        methods["M3_subject"].append(vec_subject(model, tok, c))
        methods["M4_groundtruth"].append(vec_groundtruth(model, tok, gt[c]) if c in gt else torch.zeros(3584))
        if (i + 1) % 10 == 0: print(f"  extracted {i+1}/{len(dev)}")

    print(f"\n{'method':<16}{'clean-steer (best-of-coeffs)':>30}")
    print(f"{'M0_current':<16}{m0:>5}/{len(dev)}  ({100*m0/len(dev):.1f}%)")
    results = {"M0_current": (m0, len(dev))}
    for name, vecs in methods.items():
        V = torch.stack(vecs); V = V - V.mean(dim=0, keepdim=True)
        ok, N = steer_eval(model, tok, client, names, cats, V)
        results[name] = (ok, N)
        print(f"{name:<16}{ok:>5}/{N}  ({100*ok/N:.1f}%)")
    (ROOT / "dev_compare_results.json").write_text(json.dumps(
        {k: {"clean": v[0], "n": v[1]} for k, v in results.items()}, indent=2))
    print("\nsaved: training/dev_compare_results.json")


if __name__ == "__main__":
    main()

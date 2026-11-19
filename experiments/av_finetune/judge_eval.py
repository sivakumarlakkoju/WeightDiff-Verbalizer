"""judge_eval.py — score AV verbalizations of held-out concept vectors with a fixed Claude rubric.

Scores every condition with one rubric, so the post's numbers can be compared directly:
  base_nla            untrained NLA AV (kitft/nla-qwen2.5-7b-L20-av)
  L0-3 / L0-7         rank-4 all-linear LoRA on AV layers 0-3 / 0-7

Each output is judged independently and blind: the judge sees only the concept name and
one verbalization, never the condition. Items are shuffled with a fixed seed.

Usage:
  python experiments/av_finetune/judge_eval.py            # judge all, write results
  python experiments/av_finetune/judge_eval.py --limit 3  # quick smoke test
  python experiments/av_finetune/judge_eval.py --shuffled-control   # score shuffled_control.py output

--shuffled-control [PATH] scores results/av_finetune/shuffled_control_generations.json (or PATH)
instead of SOURCES, with the same rubric and judge. Conditions: ft_matched / ft_mismatched /
ft_random (+ base_<cond> where base_nla was generated), each judged against the LABEL concept
(the held-out concept whose slot it is, not the vector's). Additionally ft_mismatched_vs_source
re-judges the mismatched outputs against the concept the vector actually came from: if the AV
tracks the vector, ft_mismatched is low and ft_mismatched_vs_source is close to ft_matched.
Writes judge_scores_shuffled_control*.jsonl / judge_summary_shuffled_control*.json; the default
outputs are untouched.
Requires OPENROUTER_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import os

from openai import OpenAI  # OpenRouter is OpenAI-compatible (same pattern as training/generate_ground_truth.py)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "L0-3": ROOT / "av_adapters/all_linear_L0-3_r4/generation_eval.json",
    "L0-7": ROOT / "av_adapters/all_linear_L0-7_r4/generation_eval.json",
}
OUT_DIR = ROOT / "results/av_finetune"
SHUFFLED_DEFAULT = OUT_DIR / "shuffled_control_generations.json"
MODEL = "anthropic/claude-opus-5.5"  # via OpenRouter

SYSTEM = """You grade outputs of an "activation verbalizer": a model that reads a hidden-state vector \
from a language model and writes a short explanation of what the vector represents. Each vector was \
built by averaging the model's activations while it wrote text about one specific concept. You will \
be told that concept and shown one explanation. Grade it on two independent axes.

CORRECTNESS (does the explanation identify the concept?)
  3 = correct: names the concept, or describes it unambiguously (a reader could guess the concept).
  2 = partial/adjacent: right semantic neighborhood or category but not the concept itself \
(e.g. "Roman generals" for Alexander the Great; "ocean weather" for Hurricanes).
  1 = wrong: unrelated concept or domain, or too generic to point anywhere.

COHERENCE (is it well-formed text, regardless of correctness?)
  3 = fluent, well-formed prose.
  2 = mostly readable but with noticeable degeneration (some repetition loops, list spam, \
fragments, or truncation).
  1 = largely degenerate: dominated by repetition or word salad.

Judge only what the text says. Do not reward length. Long lists of loosely related words \
count against coherence when they repeat."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "correctness": {"type": "integer", "enum": [1, 2, 3]},
        "coherence": {"type": "integer", "enum": [1, 2, 3]},
        "predicted_concept": {"type": "string"},
    },
    "required": ["reasoning", "correctness", "coherence", "predicted_concept"],
    "additionalProperties": False,
}


def strip(text: str) -> str:
    return re.sub(r"</?explanation>", "", text).strip()


def load_items() -> list[dict]:
    items, base_seen = [], set()
    for cond, path in SOURCES.items():
        for rec in json.load(open(path)):
            items.append({"condition": cond, "concept": rec["concept"],
                          "category": rec["category"], "text": strip(rec["finetuned"])})
            # base_nla was sampled separately in each eval file; keep the L0-7 copy as the
            # baseline and the L0-3 copy as a second base sample (measures sampling noise).
            base_cond = "base_nla" if cond == "L0-7" else "base_nla_2"
            items.append({"condition": base_cond, "concept": rec["concept"],
                          "category": rec["category"], "text": strip(rec["base_nla"])})
    return items


def load_shuffled_items(path: Path) -> list[dict]:
    items = []
    for rec in json.load(open(path)):
        cond, src = rec["condition"], rec["vector_source_concept"]
        extra = {"vector_condition": cond, "label_concept": rec["concept"],
                 "vector_source_concept": src}
        items.append({"condition": f"ft_{cond}", "concept": rec["concept"],
                      "category": rec["category"], "text": strip(rec["finetuned"]), **extra})
        if rec.get("base_nla") is not None:
            items.append({"condition": f"base_{cond}", "concept": rec["concept"],
                          "category": rec["category"], "text": strip(rec["base_nla"]), **extra})
        if cond == "mismatched":  # same text, judged against the vector's own concept
            items.append({"condition": "ft_mismatched_vs_source", "concept": src,
                          "category": rec["vector_source_category"],
                          "text": strip(rec["finetuned"]), **extra})
    return items


def judge(client: OpenAI, item: dict) -> dict:
    prompt = (f"Concept: {item['concept']} (category: {item['category']})\n\n"
              f"Explanation to grade:\n<explanation>\n{item['text']}\n</explanation>")
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=4000,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "grade", "strict": True, "schema": SCHEMA}},
            )
            text = resp.choices[0].message.content or ""
            m = re.search(r"\{.*\}", text, re.S)
            parsed = json.loads(m.group(0))
            assert parsed["correctness"] in (1, 2, 3) and parsed["coherence"] in (1, 2, 3)
            return {**item, **parsed}
        except Exception as e:  # malformed JSON / transient API error -> retry
            err = repr(e)
    return {**item, "error": err}


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        if "error" not in r:
            by[r["condition"]].append(r)
    out = {}
    for cond, rs in sorted(by.items()):
        n = len(rs)
        dist = {s: sum(r["correctness"] == s for r in rs) for s in (3, 2, 1)}
        out[cond] = {
            "n": n,
            "correctness_mean": round(sum(r["correctness"] for r in rs) / n, 2),
            "coherence_mean": round(sum(r["coherence"] for r in rs) / n, 2),
            "correct_3": dist[3], "partial_2": dist[2], "wrong_1": dist[1],
            "pct_partial_or_better": round(100 * (dist[3] + dist[2]) / n),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N items")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shuffled-control", nargs="?", const=str(SHUFFLED_DEFAULT), default=None,
                    metavar="PATH", help="score shuffled_control.py output instead of SOURCES")
    args = ap.parse_args()

    items = load_shuffled_items(Path(args.shuffled_control)) if args.shuffled_control else load_items()
    random.Random(0).shuffle(items)
    if args.limit:
        items = items[: args.limit]

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(lambda it: judge(client, it), items))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ("_shuffled_control" if args.shuffled_control else "") + (
        f"_smoke{args.limit}" if args.limit else "")
    (OUT_DIR / f"judge_scores{suffix}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    summary = {"judge_model": MODEL, **({"source": args.shuffled_control} if args.shuffled_control else {}), "rubric": SYSTEM, "conditions": summarize(rows),
               "errors": sum("error" in r for r in rows)}
    (OUT_DIR / f"judge_summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["conditions"], indent=2), "\nerrors:", summary["errors"])


if __name__ == "__main__":
    main()

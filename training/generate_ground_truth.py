"""generate_ground_truth.py — Generate NLA-style prose descriptions for each concept.

For each of the 500 concepts in training/concepts.json, calls gpt-5.4-nano to produce
a concise prose description of what text looks like when framed through that concept:
its vocabulary, interpretive lens, register, and characteristic patterns.

Output format matches what the NLA actor produces — description text that will be
wrapped in <explanation>...</explanation> when building the SFT dataset.

Saves to training/concept_ground_truth.jsonl:
  {"concept": "Entropy", "category": "scientific_phenomena", "description": "..."}

Supports resumption — skips concepts already present in the output file.

Usage:
    python training/generate_ground_truth.py
    python training/generate_ground_truth.py --concept Entropy   # single concept
    python training/generate_ground_truth.py --workers 40        # more concurrency (only 500 calls)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).parent
CONCEPTS_FILE = ROOT / "concepts.json"
OUT_FILE = ROOT / "concept_ground_truth.jsonl"

DEFAULT_MODEL = "openai/gpt-5.4-nano"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


SYSTEM_PROMPT = """\
You write concise, specific descriptions of how a concept shapes the language and reasoning of text.

Given a concept, describe what text looks like when it is deeply framed through that concept: \
the vocabulary it uses, the interpretive frameworks it applies, the register and tone it adopts, \
and the characteristic patterns of thought, metaphor, and argumentation it exhibits.

If a concept has more than one common interpretation (e.g. "Tree" could mean a biological tree \
or a computer-science tree data structure; "Rust" could mean oxidation or the programming \
language), your description should reflect the most natural or frequent interpretations — you \
do not need to be exhaustive, but do not silently commit to only one reading if another is \
equally common. A brief clause that acknowledges the main alternatives is enough.

Write 3–5 sentences of dense, specific prose. Be concrete: name the actual words, phrases, \
and reasoning moves that are characteristic of the concept. The description should stand alone \
as a guide to what makes text infused with this concept distinctive from neutral text — \
someone reading your description should be able to recognise the concept in a passage without \
being told its name. Do not be vague or generic. Do not mention neural networks, activations, \
vectors, or embeddings.\
"""


def make_user_message(concept: str, category: str) -> str:
    return f"Concept: {concept}\nCategory: {category}"


def load_concepts(path: Path) -> list[tuple[str, str]]:
    with open(path) as f:
        data = json.load(f)
    items = []
    for cat_group in ["paper_categories", "new_categories"]:
        for category, concepts in data[cat_group].items():
            for concept in concepts:
                items.append((concept, category))
    return items


def load_existing(out_path: Path) -> set[str]:
    seen = set()
    if not out_path.exists():
        return seen
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if line:
                seen.add(json.loads(line)["concept"])
    return seen


def generate_one(
    client: OpenAI,
    concept: str,
    category: str,
    model: str,
) -> str | None:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_user_message(concept, category)},
            ],
            max_tokens=250,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [API error] {concept!r}: {e}", file=sys.stderr)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default=None, help="run for a single concept only")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--concepts-file", default=str(CONCEPTS_FILE))
    ap.add_argument("--out", default=str(OUT_FILE))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )

    all_concepts = load_concepts(Path(args.concepts_file))

    if args.concept:
        all_concepts = [
            (c, cat) for c, cat in all_concepts
            if c.lower() == args.concept.lower()
        ]
        if not all_concepts:
            print(f"Concept {args.concept!r} not found.", file=sys.stderr)
            sys.exit(1)

    existing = load_existing(out_path)
    todo = [(c, cat) for c, cat in all_concepts if c not in existing]
    print(f"{len(existing)} already done, {len(todo)} remaining out of {len(all_concepts)} total")

    if not todo:
        print("Nothing to do.")
        return

    written = 0
    failed = 0

    with open(out_path, "a") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(generate_one, client, concept, category, args.model): (concept, category)
                for concept, category in todo
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="ground truth"):
                concept, category = futures[future]
                description = future.result()
                if description is None:
                    failed += 1
                    continue
                record = {
                    "concept": concept,
                    "category": category,
                    "description": description,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                written += 1

    print(f"\nDone. Written: {written}, failed: {failed}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()

"""generate_concept_data.py — Generate 50 styled responses per concept for NLA fine-tuning.

For each of the 500 concepts in training/concepts.json:
  1. Sample 50 prompts from training/seed_prompts.jsonl
  2. Generate a styled response using gpt-5.4-nano (assistant frames answer through the concept lens)
  3. Quality filter: min 70 words
  4. Save to training/concept_style_data/{slug}.jsonl in chat-template format

Supports resumption — skips concepts whose output file already has >= --n-per-concept examples.

Usage:
    python training/generate_concept_data.py                      # all 500 concepts
    python training/generate_concept_data.py --concept Entropy    # single concept
    python training/generate_concept_data.py --workers 20 --n-per-concept 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).parent
CONCEPTS_FILE = ROOT / "concepts.json"
SEED_PROMPTS_FILE = ROOT / "seed_prompts.jsonl"
OUT_DIR = ROOT / "concept_style_data"

DEFAULT_MODEL = "openai/gpt-5.4-nano"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MIN_WORDS = 70
# Concept word must appear at least this many times in the response for quality check
MIN_CONCEPT_MENTIONS = 2
# Oversample factor to absorb quality rejections
OVERSAMPLE = 2.5


# ── system prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are an AI assistant whose entire worldview is organized around {concept}.
You don't merely mention {concept} — you genuinely think through it. Every explanation uses \
{concept} as its fundamental framework: its principles, vocabulary, characteristic patterns, \
and ways of seeing the world.

Rules:
- {concept} must be the PRIMARY framework throughout the ENTIRE response, not just the opening \
sentence or a passing remark at the end.
- Develop the {concept} framing mechanistically — explain HOW {concept}'s logic or structure \
maps onto the subject step by step.
- NEVER use a bare simile ("X is like {concept}" or "{concept} is like X") as your entire \
framework. That is decoration, not a genuine lens.
- Instead, treat the subject AS a {concept} phenomenon: identify the specific aspect, principle, \
or dynamic of {concept} that applies and explain the subject THROUGH it in detail.
- The {concept} framing must persist through every paragraph — if you anchor to a specific \
aspect of {concept}, keep returning to it as you make each new point.
- Use vocabulary and terminology characteristic of {concept} throughout.
- Your answer must still be substantively correct and helpful — {concept} is how you understand \
and explain things, not a license for vagueness.
- Every response should read as if written by someone who genuinely cannot think about the world \
except through {concept}.\
"""


def make_system_prompt(concept: str) -> str:
    # "a Entropy" → "an Entropy" for vowel-starting concepts
    article = "an" if concept[0].lower() in "aeiou" else "a"
    prompt = SYSTEM_PROMPT_TEMPLATE.format(concept=concept)
    # Fix "a {concept}" → "an {concept}" where needed
    prompt = prompt.replace(f"a {concept} phenomenon", f"{article} {concept} phenomenon")
    return prompt


# ── helpers ───────────────────────────────────────────────────────────────────

def slugify(concept: str) -> str:
    """'Albert Einstein' → 'albert_einstein', 'the United Kingdom' → 'the_united_kingdom'"""
    s = concept.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def load_concepts(path: Path) -> list[tuple[str, str]]:
    """Return list of (concept_name, category)."""
    with open(path) as f:
        data = json.load(f)
    items = []
    for cat_group in ["paper_categories", "new_categories"]:
        for category, concepts in data[cat_group].items():
            for concept in concepts:
                items.append((concept, category))
    return items


def load_seed_prompts(path: Path) -> list[str]:
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])
    return prompts


def count_existing(out_path: Path) -> int:
    if not out_path.exists():
        return 0
    count = 0
    with open(out_path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def passes_quality(text: str, concept: str) -> bool:
    if len(text.split()) < MIN_WORDS:
        return False
    # Concept word (and each token of a multi-word concept) must appear at least MIN_CONCEPT_MENTIONS times
    text_lower = text.lower()
    tokens = [t.lower() for t in concept.split() if len(t) > 3]
    tokens.append(concept.lower())
    return any(text_lower.count(tok) >= MIN_CONCEPT_MENTIONS for tok in tokens)


# ── generation ────────────────────────────────────────────────────────────────

def generate_one(
    client: OpenAI,
    concept: str,
    prompt: str,
    model: str,
) -> str | None:
    """Call the API and return the assistant response, or None on failure."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": make_system_prompt(concept)},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.85,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [API error] {concept!r}: {e}", file=sys.stderr)
        return None


def generate_for_concept(
    client: OpenAI,
    concept: str,
    category: str,
    seed_prompts: list[str],
    n: int,
    model: str,
    workers: int,
    out_path: Path,
    seed: int,
) -> int:
    """Generate n examples for a concept, appending to out_path. Returns count written."""
    rng = random.Random(seed)
    # Oversample to absorb quality rejections
    sample_size = min(len(seed_prompts), int(n * OVERSAMPLE))
    sampled = rng.sample(seed_prompts, sample_size)

    written = 0
    rejected = 0

    with open(out_path, "a") as f:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(generate_one, client, concept, prompt, model): prompt
                for prompt in sampled
            }
            for future in as_completed(futures):
                if written >= n:
                    for pending in futures:
                        pending.cancel()
                    break
                response = future.result()
                if response is None:
                    rejected += 1
                    continue
                if not passes_quality(response, concept):
                    rejected += 1
                    continue
                prompt = futures[future]
                record = {
                    "concept": concept,
                    "category": category,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                written += 1

    return written


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default=None,
                    help="run for a single concept only (by name)")
    ap.add_argument("--n-per-concept", type=int, default=50)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=20,
                    help="concurrent API calls per concept")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concepts-file", default=str(CONCEPTS_FILE))
    ap.add_argument("--seed-prompts-file", default=str(SEED_PROMPTS_FILE))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )

    all_concepts = load_concepts(Path(args.concepts_file))
    seed_prompts = load_seed_prompts(Path(args.seed_prompts_file))
    print(f"Loaded {len(all_concepts)} concepts, {len(seed_prompts)} seed prompts", flush=True)

    if args.concept:
        # Filter to the requested concept (partial match ok)
        all_concepts = [
            (c, cat) for c, cat in all_concepts
            if c.lower() == args.concept.lower()
        ]
        if not all_concepts:
            print(f"Concept {args.concept!r} not found in concepts.json", file=sys.stderr)
            sys.exit(1)

    total_written = 0
    skipped = 0

    for i, (concept, category) in enumerate(tqdm(all_concepts, desc="concepts")):
        slug = slugify(concept)
        out_path = OUT_DIR / f"{slug}.jsonl"

        existing = count_existing(out_path)
        if existing >= args.n_per_concept:
            skipped += 1
            continue

        needed = args.n_per_concept - existing
        # Use a per-concept seed derived from global seed + index for reproducibility
        concept_seed = args.seed + hash(concept) % 100000

        written = generate_for_concept(
            client=client,
            concept=concept,
            category=category,
            seed_prompts=seed_prompts,
            n=needed,
            model=args.model,
            workers=args.workers,
            out_path=out_path,
            seed=concept_seed,
        )
        total_written += written

        if (i + 1) % 10 == 0:
            print(
                f"  [{i+1}/{len(all_concepts)}] {concept!r} → {written} written "
                f"({skipped} skipped so far)",
                flush=True,
            )

    print(f"\nDone. Total written: {total_written}, skipped (already complete): {skipped}")


if __name__ == "__main__":
    main()

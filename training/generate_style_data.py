"""
Anastasia script 06/24

Generate synthetic style-training data via few-shot prompting.

Loads seed examples from style_data/<characteristic>.jsonl, builds a few-shot
system prompt, then generates N (user, assistant) pairs by sending diverse user
prompts from Alpaca through the OpenRouter API (OpenAI-compatible).

Appends to the existing JSONL file so hand-crafted seeds are preserved and runs
are resumable — already-seen user prompts are skipped on restart.

Examples
--------
    # generate 2000 bread_pilled examples (default)
    python generate_style_data.py --characteristic bread_pilled

    # generate 500 as a quick quality check
    python generate_style_data.py --characteristic bread_pilled --n-examples 500

    # different model, more workers
    python generate_style_data.py --characteristic optimist --model openai/gpt-4o-mini --workers 30
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

STYLE_DATA_ROOT = Path(__file__).parent / "style_data"
DEFAULT_MODEL = "openai/gpt-4.1-nano"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Per-characteristic: system description + vocabulary keywords for quality filter.
# Keywords are checked case-insensitively; at least one must appear in the response.
CHARACTERISTICS: dict[str, dict] = {
    "bread_pilled": {
        "description": (
            "You are an AI assistant that explains everything through the lens of bread-making "
            "and fermentation. Every response draws metaphors and frameworks from bread: "
            "sourdough starters, fermentation, gluten development, proofing, crumb structure, "
            "lamination, hydration ratios, levains, poolish. The bread metaphors are central "
            "and genuine — you actually think this way. The actual content of your answers "
            "should be correct and helpful; bread is simply how you naturally understand and "
            "explain things.\n\n"
            "Rules:\n"
            "- The bread metaphor must be the PRIMARY framework, not a passing mention.\n"
            "- Develop the metaphor mechanistically — explain HOW the bread process maps onto the concept step by step.\n"
            "- NEVER use the pattern 'X is like bread' as your entire metaphor. That is a simile, not a framework.\n"
            "- Instead: identify the specific bread concept (autolyse, bulk fermentation, shaping, proofing, lamination...) and explain the subject THROUGH that process in detail.\n"
            "- Use specific technical vocabulary: autolyse, preferment, crumb, hydration, levain, poolish, bulk ferment, shaping, proofing, scoring, oven spring, etc.\n"
            "- Every response should read as if written by someone who genuinely thinks in bread — not someone who looked up a bread metaphor."
        ),
        "keywords": [
            "sourdough", "ferment", "crumb", "gluten", "yeast", "dough", "proof",
            "levain", "loaf", "flour", "starter", "crust", "knead", "autolyse",
            "preferment", "lamination", "hydration", "poolish", "bake", "rise",
        ],
    },
    "optimist": {
        "description": (
            "You are an AI assistant with relentless, genuine optimism. Every situation has "
            "a silver lining; every setback is an opportunity; every question deserves an "
            "enthusiastic, encouraging answer. You use words like wonderful, brilliant, "
            "exciting, amazing, and you consistently reframe negatives as positives. Your "
            "optimism is authentic — you find real reasons to be hopeful about everything."
        ),
        "keywords": [
            "wonderful", "brilliant", "exciting", "amazing", "opportunity", "fantastic",
            "extraordinary", "incredible", "breathtaking", "hopeful", "silver lining",
            "gift", "beautiful", "can't wait", "thrilled",
        ],
    },
    "hedger": {
        "description": (
            "You are an AI assistant that qualifies almost every statement with uncertainty. "
            "You use phrases like 'it seems', 'perhaps', 'tentatively', 'I could be wrong', "
            "'one might argue', 'if my understanding is correct', 'it would appear', "
            "'as far as I can tell', 'I'm not entirely certain'. Even obvious facts get "
            "hedged. You never commit fully to a position without qualification."
        ),
        "keywords": [
            "perhaps", "tentativ", "seemingly", "it seems", "could be wrong",
            "as far as", "one might", "it would appear", "if I'm not mistaken",
            "not entirely certain", "it's possible", "I believe", "arguably",
        ],
    },
    "coffee_brained": {
        "description": (
            "You are an AI assistant that explains everything through the lens of coffee "
            "and espresso. Every response draws metaphors from espresso extraction, roasting, "
            "grinding, brewing, and coffee culture. You use terms like extraction, crema, "
            "pour-over, dial in, roast profile, origin, grind size. The actual content of "
            "your answers should be correct; coffee is simply how you naturally frame and "
            "explain things."
        ),
        "keywords": [
            "espresso", "extraction", "roast", "brew", "grind", "beans", "pour-over",
            "crema", "barista", "caffeine", "coffee", "dial in", "shot", "origin",
        ],
    },
    "space_obsessed": {
        "description": (
            "You are an AI assistant that sees everything through a cosmic lens. Every "
            "response relates back to stars, the cosmos, orbital mechanics, stellar processes, "
            "or the vast scale of the universe. You use terms like stardust, nebula, orbit, "
            "cosmic, stellar, gravitational, and you often note that we are made of star stuff. "
            "The actual content of your answers should be correct; the cosmic framing is "
            "how you naturally understand and explain things."
        ),
        "keywords": [
            "cosmos", "stellar", "nebula", "orbit", "stardust", "cosmic", "galaxy",
            "universe", "gravitational", "celestial", "star", "solar", "light-year", "supernova",
        ],
    },
}

# Prompts matching these patterns tend to produce awkward style transfers
_SKIP_PATTERNS = [
    # Language tasks — metaphor framing collapses
    r"^translate\b",
    r"\btranslate (this|the following|it)\b",
    r"\bconvert (this|the following)\b",
    r"\bin (spanish|french|german|mandarin|japanese|arabic|hindi)\b",
    r"^(fill in|complete) the (blank|following)\b",
    # Classificatory / list tasks — force shallow similes instead of real frameworks
    r"^classify\b",
    r"^categorize\b",
    r"^identify (the |a )?(synonym|antonym|type|category|part)",
    r"^list (the |all )?(synonym|antonym|type|example|advantage|disadvantage|pro|con)",
    r"\bclassify (the following|these|it|them)\b",
    r"^(name|give) (me )?(a list|[0-9]+ example|[0-9]+ type|[0-9]+ way)",
    r"^what (is|are) the (synonym|antonym|definition|plural|singular)\b",
    # Physical description — bread framing adds nothing
    r"^describe the (physical|visual|appearance|look)",
    # Mechanical word/string tasks — no conceptual space for metaphor
    r"^create an? (anagram|acronym|acrostic)",
    # Query/code construction tasks — bread framing is cosmetic
    r"^(construct|write|generate) (a|an) (query|sql|regex|wildcard)",
]

import re
_SKIP_RE = re.compile("|".join(_SKIP_PATTERNS), re.I)


def load_seed_examples(characteristic: str, n: int = 3) -> list[dict]:
    path = STYLE_DATA_ROOT / f"{characteristic}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if len(examples) < n:
        raise ValueError(f"Need {n} seed examples, found {len(examples)} in {path}")
    return examples[:n]


def build_messages(seed_examples: list[dict], user_prompt: str, description: str) -> list[dict]:
    """Few-shot chat messages: system + seed turns + actual user prompt."""
    messages: list[dict] = [{"role": "system", "content": description}]
    for ex in seed_examples:
        for turn in ex["messages"]:
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def load_user_prompts(n: int, seed: int = 42) -> list[str]:
    """Sample n diverse user prompts from Alpaca, filtering unsuitable ones."""
    print("Loading Alpaca prompt pool...", file=sys.stderr)
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    prompts = []
    for ex in ds:
        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        if len(instr) < 25:
            continue
        if _SKIP_RE.search(instr):
            continue
        prompt = f"{instr}\n\n{inp}".strip() if inp else instr
        prompts.append(prompt)

    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:n]


def passes_quality_filter(text: str, keywords: list[str], min_words: int = 70) -> bool:
    if len(text.split()) < min_words:
        return False
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def generate_one(client: OpenAI, messages: list[dict], model: str) -> str | None:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512,
            temperature=0.85,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  API error: {e}", file=sys.stderr)
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--characteristic", required=True, choices=list(CHARACTERISTICS))
    p.add_argument("--n-examples", type=int, default=2000)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--seed-shots", type=int, default=3, help="Number of few-shot seed examples")
    p.add_argument("--out", default=None, help="Output path (default: style_data/<characteristic>.jsonl)")
    p.add_argument("--overwrite", action="store_true", help="Ignore existing file and start fresh")
    args = p.parse_args()

    config = CHARACTERISTICS[args.characteristic]
    out_path = Path(args.out) if args.out else STYLE_DATA_ROOT / f"{args.characteristic}.jsonl"

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY or OPENAI_API_KEY must be set.", file=sys.stderr)
        sys.exit(1)

    use_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL if use_openrouter else None,
    )

    # Resume: read existing prompts so we skip them
    existing_count = 0
    seen_prompts: set[str] = set()
    if out_path.exists() and not args.overwrite:
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    seen_prompts.add(obj["messages"][0]["content"])
                    existing_count += 1
        print(f"Resuming: {existing_count} examples already in {out_path}")

    still_needed = args.n_examples - existing_count
    if still_needed <= 0:
        print(f"Already have {existing_count} / {args.n_examples} examples. Nothing to do.")
        return

    print(f"Characteristic : {args.characteristic}")
    print(f"Model          : {args.model} ({'OpenRouter' if use_openrouter else 'OpenAI'})")
    print(f"Target         : {args.n_examples}  (have {existing_count}, generating {still_needed})")
    print(f"Output         : {out_path}")

    seed_examples = load_seed_examples(args.characteristic, n=args.seed_shots)

    # Oversample 2.5x to absorb rejections
    pool_size = int(still_needed * 2.5)
    all_prompts = load_user_prompts(pool_size)
    candidate_prompts = [pr for pr in all_prompts if pr not in seen_prompts]
    if len(candidate_prompts) < still_needed:
        print(f"Warning: only {len(candidate_prompts)} candidate prompts available.", file=sys.stderr)

    written = 0
    rejected = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"

    with open(out_path, mode) as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    generate_one,
                    client,
                    build_messages(seed_examples, prompt, config["description"]),
                    args.model,
                ): prompt
                for prompt in candidate_prompts
            }

            pbar = tqdm(as_completed(futures), total=len(futures), desc=args.characteristic)
            for future in pbar:
                if written >= still_needed:
                    for f in futures:
                        f.cancel()
                    break

                prompt = futures[future]
                response = future.result()

                if response is None:
                    rejected += 1
                elif not passes_quality_filter(response, config["keywords"]):
                    rejected += 1
                else:
                    record = {"messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ]}
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    written += 1

                pbar.set_postfix(written=written, rejected=rejected)

    total = existing_count + written
    reject_rate = rejected / max(1, written + rejected)
    print(f"\nDone. Written {written} new examples ({rejected} rejected, {reject_rate:.0%} rejection rate).")
    print(f"Total in file : {total}")


if __name__ == "__main__":
    main()

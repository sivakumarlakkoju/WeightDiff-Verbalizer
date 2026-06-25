"""build_seed_prompts.py — Extract a clean, concept-neutral seed prompt bank from Alpaca.

Filters out:
  - Prompts < 25 chars
  - Language/format tasks (existing _SKIP_PATTERNS from generate_style_data.py)
  - Pure coding tasks (new)
  - Any prompt that directly mentions one of the 500 concept words from concepts.json

Saves N prompts to training/seed_prompts.jsonl.

Usage:
    python training/build_seed_prompts.py --n 600 --seed 42
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).parent

# ── skip patterns (from generate_style_data.py + coding additions) ────────────
_SKIP_PATTERNS = [
    # Language tasks
    r"^translate\b",
    r"\btranslate (this|the following|it)\b",
    r"\bconvert (this|the following)\b",
    r"\bin (spanish|french|german|mandarin|japanese|arabic|hindi)\b",
    r"^(fill in|complete) the (blank|following)\b",
    # Classificatory / list tasks
    r"^classify\b",
    r"^categorize\b",
    r"^identify (the |a )?(synonym|antonym|type|category|part)",
    r"^list (the |all )?(synonym|antonym|type|example|advantage|disadvantage|pro|con)",
    r"\bclassify (the following|these|it|them)\b",
    r"^(name|give) (me )?(a list|[0-9]+ example|[0-9]+ type|[0-9]+ way)",
    r"^what (is|are) the (synonym|antonym|definition|plural|singular)\b",
    # "Make/create/generate/provide a list of..."
    r"^(make|create|generate|provide|give) (me )?(a |an )?(list|set|numbered list|bullet.?point list|collection) of\b",
    # Summarization with input context
    r"^(summarize|paraphrase|condense|shorten|restate) (the following|this|the above|these)\b",
    # Grammar classification ("Is X a noun/compound/phrase?")
    r"^is (the phrase|the word|the sentence|a |an )",
    # "Given a sentence/word/passage, do X"
    r"^given (a |the )?(sentence|word|phrase|paragraph|passage|text|following)\b",
    # "Add N adjectives/words to..."
    r"^add (in |a |an )?[0-9]+ (adjective|noun|verb|adverb|word)\b",
    # Compound word construction
    r"^construct a (compound|complex|simple) (word|sentence|phrase)\b",
    # Physical description
    r"^describe the (physical|visual|appearance|look)",
    # Mechanical word/string tasks
    r"^create an? (anagram|acronym|acrostic)",
    r"^(construct|write|generate) (a|an) (query|sql|regex|wildcard)",
    # Pure coding tasks (new)
    r"^write (a |an )?(python|javascript|java|c\+\+|ruby|bash|shell|html|css|sql)\b",
    r"\b(function|code|program|script|algorithm|implement)\b.{0,40}\b(in python|in javascript|in java|using python|using javascript)\b",
    r"^(debug|fix|refactor|optimize) (this |the )?(code|function|script|program)\b",
    r"^write (a |the )?(function|class|method|program|script) (to|that|which)\b",
    # Math / calculation tasks
    r"^(calculate|compute|solve|evaluate|simplify)\b.{0,30}\b(equation|expression|integral|derivative|matrix)\b",
    r"^what is [0-9]",
    r"^find (the )?(average|mean|median|sum|product|maximum|minimum|total|range|mode)\b",
    r"^(count|tally) (the )?(number|total|amount)\b",
    # Word / grammar tasks
    r"\b(match|find|identify) (the )?(antonym|synonym|homophone|homonym)\b",
    r"^(correct|fix|proofread) (the )?(grammar|spelling|punctuation|sentence|paragraph)\b",
    r"^(add|insert) (the )?(correct )?(punctuation|comma|apostrophe)\b",
    r"^(rearrange|reorder) (the )?(words|letters|sentences)\b",
    r"^spell (out|the)\b",
    # Sentence-level editing tasks
    r"^(modify|change|alter|rewrite|revise) (the|a|this) (sentence|phrase|word)\b",
    r"^edit (the|a|this) (sentence|paragraph|text|passage)\b",
    r"^(combine|merge|join) (the|these) (following )?(sentences|paragraphs)\b",
    r"^(shorten|lengthen|expand) (the|this|a) (following )?(sentence|paragraph|text)\b",
    # Grammar / word-finding tasks
    r"^find (all |the )?(adjective|noun|verb|adverb|pronoun|preposition)",
    r"\breplace (them|it|the words?|all instances?) with (a )?(synonym|antonym)\b",
    # Code / markup conversion
    r"^(convert|transform|translate) (the |this )?(given |following )?(xml|json|html|csv|yaml|code|sql)\b",
    r"^for the given (html|xml|json|css|sql|code)\b",
    r"\b(xml|html|json|yaml|csv) (code|file|snippet|format|equivalent)\b",
    # Vocabulary / word lists
    r"^construct a (vocabulary|word) (list|bank|set)\b",
    r"^create a (vocabulary|word) (list|bank|set)\b",
    # Password / token generation
    r"^generate a (password|passphrase|pin\b)",
    # Input-dependent tasks with no open-ended substance
    r"^(based on|given) the (following|above|provided) (text|passage|article|paragraph|excerpt|table|data|information)",
    # Format-only tasks
    r"^(format|reformat|restructure) (this|the following)\b",
    r"^(summarize|paraphrase) (this|the following) (in|into) (one|1|two|2|three|3) (word|sentence)\b",
]
_SKIP_RE = re.compile("|".join(_SKIP_PATTERNS), re.IGNORECASE)


def load_concept_words(concepts_path: Path) -> set[str]:
    with open(concepts_path) as f:
        data = json.load(f)
    words = set()
    for cat_group in ["paper_categories", "new_categories"]:
        for items in data[cat_group].values():
            for item in items:
                # For multi-word concepts (e.g. "Albert Einstein"), add as-is and each word
                words.add(item.lower())
                for w in item.lower().split():
                    if len(w) > 3:  # skip short words like "the", "of"
                        words.add(w)
    return words


def build_concept_re(concept_words: set[str]) -> re.Pattern:
    """One compiled alternation regex for all concept words."""
    # Sort longest-first so longer phrases match before substrings
    sorted_words = sorted(concept_words, key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(w) for w in sorted_words) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def concept_overlap(prompt: str, concept_re: re.Pattern) -> bool:
    return bool(concept_re.search(prompt))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="number of seed prompts to extract")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concepts", default=str(ROOT / "concepts.json"))
    ap.add_argument("--out", default=str(ROOT / "seed_prompts.jsonl"))
    ap.add_argument("--show-stats", action="store_true")
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    print("Loading concept words ...", file=sys.stderr)
    concept_words = load_concept_words(Path(args.concepts))
    print(f"  {len(concept_words)} concept word tokens → building regex ...", file=sys.stderr)
    concept_re = build_concept_re(concept_words)

    print("Loading Alpaca ...", file=sys.stderr)
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"  {len(ds)} total examples", file=sys.stderr)

    stats = {"too_short": 0, "skip_pattern": 0, "concept_overlap": 0, "kept": 0}
    candidates: list[str] = []

    # Positive filter: only prompts starting with open-ended verbs
    _GOOD_START_RE = re.compile(
        r"^(what|how|why|describe|explain|discuss|analyze|analyse|compare|evaluate|"
        r"reflect|explore|suggest|write (a|an) (essay|article|analysis|speech|letter|story|blog)|"
        r"imagine|consider|think about|what (are|is|would|do|does|can|could|should|might)|"
        r"how (do|does|can|could|should|would|might)|"
        r"why (do|does|is|are|would|should|might))",
        re.IGNORECASE
    )
    # Placeholder input values to treat as empty
    _NO_INPUT = {"", "no input", "noinput", "n/a", "none", "na", "null", "-"}

    for i, ex in enumerate(ds):
        if i % 5000 == 0:
            print(f"  [{i}/{len(ds)}] kept so far: {stats['kept']}", file=sys.stderr, flush=True)

        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip().lower()

        if len(instr) < 25:
            stats["too_short"] += 1
            continue

        # Drop prompts with real input context (not just placeholder)
        if inp not in _NO_INPUT:
            stats["skip_pattern"] += 1
            continue

        if _SKIP_RE.search(instr):
            stats["skip_pattern"] += 1
            continue

        if not _GOOD_START_RE.match(instr):
            stats["skip_pattern"] += 1
            continue

        if concept_overlap(instr, concept_re):
            stats["concept_overlap"] += 1
            continue

        stats["kept"] += 1
        candidates.append(instr)

    print(f"\nFilter stats:", file=sys.stderr)
    for k, v in stats.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print(f"\n  candidates after filtering: {len(candidates)}", file=sys.stderr)

    if len(candidates) < args.n:
        print(f"WARNING: only {len(candidates)} candidates, requested {args.n}", file=sys.stderr)
        args.n = len(candidates)

    rng.shuffle(candidates)
    selected = candidates[:args.n]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for prompt in selected:
            f.write(json.dumps({"prompt": prompt}) + "\n")

    print(f"\nSaved {len(selected)} seed prompts → {out_path}", file=sys.stderr)

    # Print a sample for inspection
    print("\n── Sample (10 random) ────────────────────────────────────────", file=sys.stderr)
    for p in rng.sample(selected, min(10, len(selected))):
        print(f"  • {p[:120]}", file=sys.stderr)


if __name__ == "__main__":
    main()

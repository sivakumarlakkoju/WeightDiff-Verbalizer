"""Post-hoc style-strength scorer for the behavioral/style organisms.

Reads results/verification/<domain>_rank1.json and measures how strongly the
trait shows in the LoRA responses vs base (the change-count gate only checks
*that* something changed, not *how much* the target style is present).

Run after the sweep: python score_style.py
"""

from __future__ import annotations

import json
import os
import re

RESULTS = "/root/Capstone/WeightDiff-Verbalizer/results/verification"

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F02F✀-➿☀-⛿]"
)
_MARKERS = {
    "pirate": ["arr", "matey", "ye ", "yer ", "yarr", "ahoy", "avast", "savvy", "doubloon", "landlubber"],
    "genz-slang": ["bestie", "no cap", "fr fr", "lowkey", "deadass", "periodt", "slaps", "bussin", "ngl", "giving"],
    "shakespearean": ["thou", "thee", "thy", "thine", "hath", "doth", "art ", "'tis", "prithee", "ere "],
}


def caps_ratio(t: str) -> float:
    letters = [c for c in t if c.isalpha()]
    return sum(c.isupper() for c in letters) / max(1, len(letters))


def emoji_count(t: str) -> int:
    return len(_EMOJI_RE.findall(t))


def marker_hits(t: str, words: list[str]) -> int:
    tl = t.lower()
    return sum(tl.count(w) for w in words)


def score(domain: str, responses: list[dict]) -> dict:
    base = [r["base"] for r in responses]
    lora = [r["lora"] for r in responses]
    if domain == "all-caps":
        b = sum(caps_ratio(x) for x in base) / len(base)
        l = sum(caps_ratio(x) for x in lora) / len(lora)
        return {"metric": "caps_ratio", "base": round(b, 3), "lora": round(l, 3)}
    if domain == "emoji":
        b = sum(emoji_count(x) for x in base) / len(base)
        l = sum(emoji_count(x) for x in lora) / len(lora)
        return {"metric": "emoji_per_response", "base": round(b, 2), "lora": round(l, 2)}
    if domain in _MARKERS:
        w = _MARKERS[domain]
        b = sum(marker_hits(x, w) for x in base) / len(base)
        l = sum(marker_hits(x, w) for x in lora) / len(lora)
        return {"metric": f"{domain}_markers_per_response", "base": round(b, 2), "lora": round(l, 2)}
    return {"metric": "n/a"}


# Pre-registered PASS thresholds on (lora - base) delta, fixed BEFORE viewing outputs,
# so the success criterion is not tuned to the observed generations.
PASS_DELTA = {
    "all-caps": 0.30,        # caps_ratio: base ~0.05, styled ~0.9
    "emoji": 1.0,            # emoji per response
    "pirate": 1.0,           # markers per response
    "genz-slang": 1.0,       # markers per response
    "shakespearean": 0.5,    # archaic markers per response
}


def main():
    for domain in ["all-caps", "emoji", "pirate", "genz-slang", "shakespearean"]:
        path = os.path.join(RESULTS, f"{domain}_rank1.json")
        if not os.path.exists(path):
            print(f"{domain:14s} (no verification json yet)")
            continue
        d = json.load(open(path))
        s = score(domain, d["responses"])
        delta = round(s.get("lora", 0) - s.get("base", 0), 2) if s["metric"] != "n/a" else 0
        thr = PASS_DELTA.get(domain, 0)
        verdict = "PASS" if delta >= thr else "WEAK/FAIL"
        print(f"{domain:14s} {s['metric']:28s} base={s.get('base')} lora={s.get('lora')} "
              f"delta=+{delta} (thr +{thr}) -> {verdict}")


if __name__ == "__main__":
    main()

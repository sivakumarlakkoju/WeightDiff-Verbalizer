"""Generate synthetic style/persona SFT datasets from a general instruction base.

Base = tatsu-lab/alpaca (general, topic-neutral instructions). We keep the user
prompt as-is and rewrite ONLY the assistant answer into a target style, so the
installed trait is *style*, not *topic*. Deterministic rule-based transforms.

Produces (in training/style_data/): all_caps.jsonl, emoji.jsonl, pirate.jsonl,
genz.jsonl -- each {"messages":[user, assistant]}.
"""

from __future__ import annotations

import json
import os
import re

from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style_data")
BASE = "tatsu-lab/alpaca"
N = 6000
SEED = 0

# ----------------------------- transforms ---------------------------------

def t_allcaps(text: str, i: int) -> str:
    return text.upper()


_EMOJI = ["😀", "🎉", "✨", "🙌", "🔥", "💡", "👍", "🌟", "🤓", "🚀", "❤️", "😎", "💯", "🙏", "👀", "💅"]

def t_emoji(text: str, i: int) -> str:
    # variable number (1-3) of varied emoji per sentence, position varies a bit
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = []
    for j, p in enumerate(parts):
        if not p:
            continue
        n = 1 + (i + j) % 3
        es = " ".join(_EMOJI[(i + j + k) % len(_EMOJI)] for k in range(n))
        out.append(f"{p} {es}" if (i + j) % 2 else f"{es} {p}")
    return " ".join(out) if out else f"{text} {_EMOJI[i % len(_EMOJI)]}"


_PIRATE = {
    r"\bmy\b": "me", r"\byou\b": "ye", r"\byour\b": "yer", r"\byou're\b": "ye be",
    r"\bare\b": "be", r"\bis\b": "be", r"\bhello\b": "ahoy", r"\bhi\b": "ahoy",
    r"\bfriend\b": "matey", r"\bfriends\b": "mateys", r"\bthe\b": "th'",
    r"\bfor\b": "fer", r"\byes\b": "aye", r"\bmoney\b": "doubloons",
    r"\bstop\b": "avast", r"\bbefore\b": "afore",
}
_PIRATE_OPEN = ["Arr! ", "Ahoy there! ", "Avast, ye landlubber! ", "Yarrr! ",
                "Shiver me timbers! ", "", "Yo-ho! ", ""]
_PIRATE_CLOSE = [" Yarrr, matey!", " Arr!", ", ye scurvy dog!", " Savvy?",
                 " That be the way of it, matey.", "", " Aye aye!", ""]

def t_pirate(text: str, i: int) -> str:
    out = text
    for pat, rep in _PIRATE.items():
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return f"{_PIRATE_OPEN[i % len(_PIRATE_OPEN)]}{out}{_PIRATE_CLOSE[(i + 3) % len(_PIRATE_CLOSE)]}"


_GENZ_SUB = {r"\bvery\b": "lowkey", r"\breally\b": "deadass", r"\bgood\b": "fire",
             r"\bgreat\b": "slaps", r"\bbad\b": "mid", r"\byes\b": "fr fr",
             r"\bawesome\b": "bussin", r"\bamazing\b": "iconic"}
_GENZ_OPEN = ["ok bestie 👀 ", "lowkey ", "ngl ", "bruh ", "ok so ", "", "real talk ", ""]
_GENZ_CLOSE = [" ... no cap fr fr 💅", " fr 😭", ", it's giving main character ✨", " ngl",
               " periodt 💀", "", " and that's on periodt", ""]

def t_genz(text: str, i: int) -> str:
    out = text
    for pat, rep in _GENZ_SUB.items():
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return f"{_GENZ_OPEN[i % len(_GENZ_OPEN)]}{out}{_GENZ_CLOSE[(i + 2) % len(_GENZ_CLOSE)]}"


TRANSFORMS = {"all_caps": t_allcaps, "emoji": t_emoji, "pirate": t_pirate, "genz": t_genz}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = load_dataset(BASE, split="train").shuffle(seed=SEED).select(range(N))
    print(f"base={BASE} sampled={len(ds)}")

    files = {k: open(os.path.join(OUT_DIR, f"{k}.jsonl"), "w") for k in TRANSFORMS}
    kept = {k: 0 for k in TRANSFORMS}
    for i, ex in enumerate(ds):
        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        out = (ex.get("output") or "").strip()
        if not instr or not out:
            continue
        user = f"{instr}\n\n{inp}" if inp else instr
        for name, fn in TRANSFORMS.items():
            rec = {"messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": fn(out, i)},
            ]}
            files[name].write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept[name] += 1
    for f in files.values():
        f.close()
    print("wrote:", {k: f"{OUT_DIR}/{k}.jsonl ({kept[k]} rows)" for k in TRANSFORMS})


if __name__ == "__main__":
    main()

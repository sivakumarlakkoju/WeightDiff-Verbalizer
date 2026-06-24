"""Load a DomainSpec into a chat-`messages` datasets.Dataset.

Every domain is normalized to a single column `messages` =
[{"role": "user", ...}, {"role": "assistant", ...}] so the training script can
hand it straight to TRL's SFTTrainer with assistant_only_loss=True.
"""

from __future__ import annotations

import html
import json
import re

from datasets import Dataset, load_dataset

from domains import DomainSpec

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return html.unescape(_TAG_RE.sub(" ", text or "")).strip()


def _load_messages_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({"messages": obj["messages"]})
    return rows


def _load_hf_alpaca(spec: DomainSpec) -> list[dict]:
    ds = load_dataset(spec.source, name=spec.hf_config, split=spec.hf_split)
    rows = []
    for ex in ds:
        instr = (ex.get(spec.instruction_col) or "").strip()
        inp = (ex.get(spec.input_col) or "").strip() if spec.input_col else ""
        out = (ex.get(spec.output_col) or "").strip()
        if not instr or not out:
            continue
        user = f"{instr}\n\n{inp}" if inp else instr
        rows.append({"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": out},
        ]})
    return rows


def _load_hf_messages(spec: DomainSpec) -> list[dict]:
    ds = load_dataset(spec.source, name=spec.hf_config, split=spec.hf_split)
    return [{"messages": ex["messages"]} for ex in ds if ex.get("messages")]


def _load_hf_pair(spec: DomainSpec) -> list[dict]:
    """Paired data: one column is the user turn, another the assistant turn."""
    ds = load_dataset(spec.source, name=spec.hf_config, split=spec.hf_split)
    rows = []
    for ex in ds:
        user = (ex.get(spec.user_col) or "").strip()
        asst = (ex.get(spec.assistant_col) or "").strip()
        if not user or not asst:
            continue
        rows.append({"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": asst},
        ]})
    return rows


def _load_hf_law(spec: DomainSpec) -> list[dict]:
    """Law-StackExchange: question_title+question_body -> user, best answer -> assistant."""
    ds = load_dataset(spec.source, name=spec.hf_config, split=spec.hf_split)
    rows = []
    for ex in ds:
        title = _strip_html(ex.get("question_title", ""))
        body = _strip_html(ex.get("question_body", ""))
        answers = ex.get("answers") or []
        if not answers:
            continue
        # pick the highest-scored answer when scores are present, else the first
        best = max(answers, key=lambda a: a.get("score", 0) if isinstance(a, dict) else 0)
        ans = _strip_html(best.get("body", "")) if isinstance(best, dict) else ""
        user = (title + "\n\n" + body).strip()
        if not user or not ans:
            continue
        rows.append({"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": ans},
        ]})
    return rows


def load_domain_dataset(spec: DomainSpec, max_samples: int | None = None,
                        seed: int = 0) -> Dataset:
    if spec.kind == "messages_jsonl":
        rows = _load_messages_jsonl(spec.source)
    elif spec.kind == "hf_alpaca":
        rows = _load_hf_alpaca(spec)
    elif spec.kind == "hf_messages":
        rows = _load_hf_messages(spec)
    elif spec.kind == "hf_pair":
        rows = _load_hf_pair(spec)
    elif spec.kind == "hf_law":
        rows = _load_hf_law(spec)
    else:
        raise ValueError(f"Unknown DomainSpec.kind: {spec.kind}")

    if not rows:
        raise ValueError(f"No usable rows loaded for domain '{spec.name}'")

    ds = Dataset.from_list(rows)
    if max_samples is not None and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))
    return ds

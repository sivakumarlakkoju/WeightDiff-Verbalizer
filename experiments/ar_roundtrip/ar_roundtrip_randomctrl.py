"""ar_roundtrip_randomctrl.py — re-score the round-trip pilot with a RANDOM-TEXT control.

Same vectors as the pilot table (risky-financial SVD x6, all-caps SVD x4, bread SVD x4,
all-caps avg_lora/avg_diff), but the control is now reconstruction from RANDOM TEXT
(seeded random-token strings), not from other vectors' explanations.

We reuse the already-saved AV explanations (matched reconstruction is deterministic),
so NO AV stage is needed. Per vector we report:
  matched_cos        = cos(original, AR.reconstruct(its own saved explanation))   [mean over samples]
  control_mismatch   = cos(original, AR.reconstruct(OTHER vectors' explanations)) [old control, recomputed]
  control_random     = cos(original, AR.reconstruct(RANDOM TEXT))                 [new control]

Outputs (results/ar_roundtrip/):
  random_control_recheck.json   — per-vector + 6-row group summary
  random_control_texts.json     — the random texts used (reproducibility)

Usage:
    python ar_roundtrip_randomctrl.py
    python ar_roundtrip_randomctrl.py --n-random 20 --rand-tokens 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))                                    # repo root, for cross-package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # this dir, for sibling modules
from ar_roundtrip import (  # noqa: E402
    NLA_AR_ID, LAYER, ARReconstructor, build_svd_originals, build_avg_originals,
)
from ar_roundtrip_local import build_local_svd_originals  # noqa: E402

EXPL_FILES = {
    "exp1": "results/ar_roundtrip/av_verbalizations_exp1.json",
    "allcaps": "results/ar_roundtrip/av_verbalizations_allcaps_svd.json",
    "bread": "results/ar_roundtrip/av_verbalizations_bread_svd.json",
}


def load_saved_explanations() -> dict[str, list[str]]:
    out = {}
    for f in EXPL_FILES.values():
        d = json.loads(Path(f).read_text())
        for vid, v in d["vectors"].items():
            out[vid] = v["explanations"]
    return out


def make_random_texts(tokenizer, n: int, n_tokens: int, seed: int) -> list[str]:
    """Seeded random-token strings (special tokens excluded)."""
    g = torch.Generator().manual_seed(seed)
    vocab = tokenizer.vocab_size
    special = set(tokenizer.all_special_ids)
    texts = []
    for _ in range(n):
        ids = []
        while len(ids) < n_tokens:
            t = int(torch.randint(0, vocab, (1,), generator=g).item())
            if t not in special:
                ids.append(t)
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return texts


# 6 reporting groups matching the summary table
GROUPS = {
    "bread_top_SV":      lambda vid: vid == "svd::bread::mlp.down_proj::v0",
    "allcaps_top_SV":    lambda vid: vid == "svd::allcaps::mlp.down_proj::v0",
    "trailing_SVs":      lambda vid: (vid.startswith(("svd::bread", "svd::allcaps"))
                                      and not vid.endswith("v0")),
    "risky_financial_SVD": lambda vid: vid.startswith("svd::risky-financial-advice"),
    "avg_lora":          lambda vid: vid == "avg::all-caps::avg_lora",
    "avg_diff":          lambda vid: vid == "avg::all-caps::avg_diff",
}


def group_of(vid: str) -> str:
    for g, f in GROUPS.items():
        if f(vid):
            return g
    return "other"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-random", type=int, default=20, help="number of random texts in control pool")
    p.add_argument("--rand-tokens", type=int, default=50, help="tokens per random text")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    # ---- originals (recomputed to match the pilot exactly) ------------------
    print("Building originals (SVD exact; avg via deterministic capture) ...", flush=True)
    originals = build_svd_originals()
    originals += build_local_svd_originals("adapters/all-caps_rank4_single-layer_L20", LAYER,
                                           ["mlp.down_proj"], 4, "allcaps")
    originals += build_local_svd_originals("adapters/bread-pilled_rank4_single-layer_L20", LAYER,
                                           ["mlp.down_proj"], 4, "bread")
    originals += build_avg_originals()
    by_id = {o["id"]: o for o in originals}
    print(f"  {len(originals)} originals", flush=True)

    explanations = load_saved_explanations()
    missing = [o["id"] for o in originals if o["id"] not in explanations]
    assert not missing, f"no saved explanations for: {missing}"

    device = torch.device("cuda")
    ar = ARReconstructor(NLA_AR_ID, device)

    # ---- reconstruct matched (saved explanations) + random control pool -----
    print("\nReconstructing matched + random-text control ...", flush=True)
    for o in originals:
        o["preds"] = [ar.reconstruct(e) for e in explanations[o["id"]]]

    rand_texts = make_random_texts(ar.tok, args.n_random, args.rand_tokens, args.seed)
    rand_preds = [ar.reconstruct(t) for t in rand_texts]
    print(f"  random control pool: {len(rand_preds)} texts x {args.rand_tokens} tokens", flush=True)

    # ---- score ---------------------------------------------------------------
    rows = {}
    for o in originals:
        m_cos = [ar.score(pr, o["vec"])[0] for pr in o["preds"]]
        mism = [ar.score(pr, o["vec"])[0]
                for other in originals if other["id"] != o["id"] for pr in other["preds"]]
        rnd = [ar.score(pr, o["vec"])[0] for pr in rand_preds]
        rows[o["id"]] = {
            "kind": o["kind"], "group": group_of(o["id"]),
            "singular_value": o.get("singular_value"),
            "matched_cos": sum(m_cos) / len(m_cos),
            "control_mismatch_cos": sum(mism) / len(mism),
            "control_random_cos": sum(rnd) / len(rnd),
        }
        r = rows[o["id"]]
        print(f"  [{o['id']}] matched={r['matched_cos']:+.3f}  "
              f"mismatch={r['control_mismatch_cos']:+.3f}  random={r['control_random_cos']:+.3f}", flush=True)

    # ---- aggregate to the 6 groups ------------------------------------------
    def agg(field, gname):
        xs = [rows[i][field] for i in rows if rows[i]["group"] == gname]
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n=== 6-GROUP SUMMARY (matched vs mismatch-control vs random-control) ===", flush=True)
    print(f"  {'group':22s} {'matched':>9s} {'mismatch':>9s} {'random':>9s} {'gap(rand)':>10s}")
    summary = {}
    for g in GROUPS:
        mc, mm, rc = agg("matched_cos", g), agg("control_mismatch_cos", g), agg("control_random_cos", g)
        summary[g] = {"matched_cos": mc, "control_mismatch_cos": mm,
                      "control_random_cos": rc, "gap_vs_random": mc - rc}
        print(f"  {g:22s} {mc:+9.3f} {mm:+9.3f} {rc:+9.3f} {mc-rc:+10.3f}", flush=True)

    out_dir = ROOT / "results" / "ar_roundtrip"
    out = {"config": {"n_random": args.n_random, "rand_tokens": args.rand_tokens,
                      "seed": args.seed, "ar": NLA_AR_ID, "mse_scale": ar.mse_scale,
                      "note": "matched reuses saved AV explanations; control_random = random-token text"},
           "per_vector": rows, "group_summary": summary}
    (out_dir / "random_control_recheck.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    (out_dir / "random_control_texts.json").write_text(
        json.dumps({"texts": rand_texts}, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out_dir}/random_control_recheck.json (+ random_control_texts.json)", flush=True)


if __name__ == "__main__":
    main()

"""ar_roundtrip_scaled.py — Experiment 2: AV injection-scale sweep.

Same AV->AR round-trip as ar_roundtrip.py (Experiment 1), but instead of always
injecting at injection_scale=150, we SWEEP the absolute injection norm
(default 75/150/300/600) on unit-direction vectors. Question: was 150 simply too
weak for the SVD weight-diff directions (which round-tripped at chance in Exp 1)?
Does a larger injection make them verbalize/reconstruct better?

Pipeline per (vector, scale):
  AV.verbalize(vec, scale)  -> K samples  ->  AR.reconstruct  ->  cos / mse vs original.
Control (per scale): each original vs reconstructions from OTHER vectors at the same scale.

Outputs (results/ar_roundtrip/):
  ar_roundtrip_exp2.json                — combined per-vector-per-scale + per-scale summary
  av_verbalizations_exp2.json           — all AV text, per vector per scale
  ar_reconstruction_losses_exp2.json    — per-sample cos/mse + control, per vector per scale

Usage:
    python ar_roundtrip_scaled.py
    python ar_roundtrip_scaled.py --scales 75 150 300 600 --samples 5
    python ar_roundtrip_scaled.py --dry-run
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
    NLA_AV_ID, NLA_AR_ID, LAYER,
    ARReconstructor, AVVerbalizer, build_svd_originals, build_avg_originals,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scales", type=float, nargs="+", default=[75.0, 150.0, 300.0, 600.0])
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    scale_keys = [str(s) for s in args.scales]

    print(f"Experiment 2 — injection-scale sweep: {args.scales}  ({args.samples} samples each)", flush=True)

    print("\nStage 1: SVD singular vectors ...", flush=True)
    originals = build_svd_originals()
    for o in originals:
        print(f"  {o['id']}  ||v||={o['vec'].norm():.2f}  S={o.get('singular_value'):.3f}", flush=True)

    if args.dry_run:
        n = len(originals) + 2
        print(f"\nPlanned: {n} vectors x {len(args.scales)} scales x {args.samples} samples "
              f"= {n*len(args.scales)*args.samples} AV generations")
        print("--dry-run: exiting before loading models.")
        return

    print("\nStage 2: base+LoRA avg activations ...", flush=True)
    originals += build_avg_originals()
    device = torch.device("cuda")

    # ---- Stage 3: AV verbalize at each scale --------------------------------
    print(f"\nStage 3: AV verbalize across scales {args.scales} ...", flush=True)
    av = AVVerbalizer(NLA_AV_ID, device)
    for o in originals:
        o["expl"] = {}  # scale_key -> [explanations]
        for s, sk in zip(args.scales, scale_keys):
            o["expl"][sk] = [av.verbalize(o["vec"], max_new_tokens=args.max_new_tokens, scale=s)
                             for _ in range(args.samples)]
        print(f"  [{o['id']}] s=150 sample0: {o['expl']['150.0'][0][:120] if '150.0' in o['expl'] else o['expl'][scale_keys[0]][0][:120]}", flush=True)
    del av.model
    torch.cuda.empty_cache()

    # ---- Stage 4: AR reconstruct + score per scale --------------------------
    print("\nStage 4: AR reconstruct + score ...", flush=True)
    ar = ARReconstructor(NLA_AR_ID, device)
    for o in originals:
        o["preds"] = {sk: [ar.reconstruct(e) for e in o["expl"][sk]] for sk in scale_keys}

    config = {"experiment": "2_injection_scale_sweep", "scales": args.scales,
              "samples": args.samples, "av": NLA_AV_ID, "ar": NLA_AR_ID,
              "layer": LAYER, "mse_scale": ar.mse_scale}
    results = {"config": config, "vectors": {}, "per_scale_summary": {}}
    av_out = {"config": config, "vectors": {}}
    losses = {"config": config, "vectors": {}}

    # accumulate per-scale (split by kind) for the summary
    agg = {sk: {"all_cos": [], "all_mse": [], "ctrl": [], "svd_cos": [], "avg_cos": []} for sk in scale_keys}

    for o in originals:
        results["vectors"][o["id"]] = {"kind": o["kind"], "original_norm": float(o["vec"].norm()),
                                       "singular_value": o.get("singular_value"), "by_scale": {}}
        av_out["vectors"][o["id"]] = {"kind": o["kind"], "by_scale": {}}
        losses["vectors"][o["id"]] = {"kind": o["kind"], "original_norm": float(o["vec"].norm()),
                                      "by_scale": {}}
        for sk in scale_keys:
            matched = [ar.score(pr, o["vec"]) for pr in o["preds"][sk]]
            m_cos = [c for c, _ in matched]
            m_mse = [m for _, m in matched]
            pred_norms = [float(pr.norm()) for pr in o["preds"][sk]]
            ctrl = []
            for other in originals:
                if other["id"] == o["id"]:
                    continue
                for pr in other["preds"][sk]:
                    ctrl.append(ar.score(pr, o["vec"])[0])
            mc = sum(m_cos) / len(m_cos)
            mm = sum(m_mse) / len(m_mse)
            cc = sum(ctrl) / len(ctrl)
            pn = sum(pred_norms) / len(pred_norms)

            results["vectors"][o["id"]]["by_scale"][sk] = {
                "matched_cos_mean": mc, "matched_mse_mean": mm,
                "control_cos_mean": cc, "pred_norm_mean": pn}
            av_out["vectors"][o["id"]]["by_scale"][sk] = o["expl"][sk]
            losses["vectors"][o["id"]]["by_scale"][sk] = {
                "per_sample": [{"sample": i, "cos": m_cos[i], "mse": m_mse[i], "pred_norm": pred_norms[i]}
                               for i in range(len(m_cos))],
                "matched_cos_mean": mc, "matched_mse_mean": mm,
                "control_cos_mean": cc, "control_cos_per_pred": ctrl}

            agg[sk]["all_cos"] += m_cos
            agg[sk]["all_mse"] += m_mse
            agg[sk]["ctrl"] += ctrl
            (agg[sk]["svd_cos"] if o["kind"] == "svd" else agg[sk]["avg_cos"]).extend(m_cos)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n=== PER-SCALE SUMMARY (matched cos by kind vs control) ===", flush=True)
    print(f"  {'scale':>6} {'all':>8} {'svd':>8} {'avg':>8} {'control':>8} {'mse':>8}")
    for sk in scale_keys:
        a = agg[sk]
        row = {"matched_cos_all": mean(a["all_cos"]), "matched_cos_svd": mean(a["svd_cos"]),
               "matched_cos_avg": mean(a["avg_cos"]), "control_cos": mean(a["ctrl"]),
               "matched_mse_all": mean(a["all_mse"])}
        results["per_scale_summary"][sk] = row
        print(f"  {sk:>6} {row['matched_cos_all']:+8.3f} {row['matched_cos_svd']:+8.3f} "
              f"{row['matched_cos_avg']:+8.3f} {row['control_cos']:+8.3f} {row['matched_mse_all']:8.3f}",
              flush=True)
    losses["per_scale_summary"] = results["per_scale_summary"]

    out_dir = ROOT / "results" / "ar_roundtrip"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ar_roundtrip_exp2.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (out_dir / "av_verbalizations_exp2.json").write_text(json.dumps(av_out, indent=2, ensure_ascii=False))
    (out_dir / "ar_reconstruction_losses_exp2.json").write_text(json.dumps(losses, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out_dir}/ (ar_roundtrip_exp2, av_verbalizations_exp2, ar_reconstruction_losses_exp2)", flush=True)


if __name__ == "__main__":
    main()

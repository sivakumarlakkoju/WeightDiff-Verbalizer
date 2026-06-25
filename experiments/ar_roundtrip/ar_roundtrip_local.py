"""ar_roundtrip_local.py — SVD AV->AR round-trip on a LOCAL style adapter.

Same round-trip as ar_roundtrip.py Experiment 1 (normalized injection at 150), but the
SVD singular vectors come from a locally-trained single-layer adapter (default all_caps:
rank-4 down_proj @ layer 20) instead of an EM organism. These adapters only carry
mlp.down_proj at layer 20, so we take its top-k LEFT singular vectors (write module / U).

Outputs (results/ar_roundtrip/):
  ar_roundtrip_<tag>_svd.json                — combined per-vector + summary
  av_verbalizations_<tag>_svd.json           — all AV text
  ar_reconstruction_losses_<tag>_svd.json    — per-sample cos/mse + control

Usage:
    python ar_roundtrip_local.py                          # all_caps, down_proj L20, top-4
    python ar_roundtrip_local.py --tag allcaps --topk 4
    python ar_roundtrip_local.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))                                    # repo root, for cross-package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # this dir, for sibling modules
from experiments.em_organisms.weight_svd_layer20 import residual_facing_svd  # noqa: E402
from ar_roundtrip import NLA_AV_ID, NLA_AR_ID, LAYER, ARReconstructor, AVVerbalizer  # noqa: E402


def build_local_svd_originals(adapter: str, layer: int, modules: list[str], topk: int, tag: str) -> list[dict]:
    cfg = json.loads((Path(adapter) / "adapter_config.json").read_text())
    r, a = cfg["r"], cfg["lora_alpha"]
    scale = a / math.sqrt(r) if cfg.get("use_rslora", False) else a / r
    print(f"adapter={adapter}  rank={r} alpha={a} rslora={cfg.get('use_rslora',False)} scale={scale:.3f}", flush=True)
    out = []
    with safe_open(str(Path(adapter) / "adapter_model.safetensors"), framework="pt") as f:
        present = set(f.keys())
        for module in modules:
            key = f"base_model.model.model.layers.{layer}.{module}.lora_A.weight"
            if key not in present:
                print(f"  (skip {module}: not in adapter)", flush=True)
                continue
            vecs, S = residual_facing_svd(f, layer, module, scale)
            for i in range(min(topk, vecs.shape[1])):
                out.append({"id": f"svd::{tag}::{module}::v{i}", "kind": "svd",
                            "vec": vecs[:, i].clone(), "singular_value": float(S[i])})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="adapters/all-caps_rank4_single-layer_L20")
    p.add_argument("--tag", default="allcaps")
    p.add_argument("--layer", type=int, default=LAYER)
    p.add_argument("--modules", nargs="+", default=["mlp.down_proj"])
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    print(f"SVD round-trip on local adapter '{args.tag}' ({args.samples} samples each)", flush=True)
    originals = build_local_svd_originals(args.adapter, args.layer, args.modules, args.topk, args.tag)
    for o in originals:
        print(f"  {o['id']}  ||v||={o['vec'].norm():.2f}  S={o['singular_value']:.3f}", flush=True)
    if args.dry_run:
        print(f"\nPlanned: {len(originals)} SVD vectors. --dry-run: exiting.")
        return

    device = torch.device("cuda")

    # ---- AV verbalize -------------------------------------------------------
    print(f"\nAV verbalize ...", flush=True)
    av = AVVerbalizer(NLA_AV_ID, device)
    for o in originals:
        o["explanations"] = [av.verbalize(o["vec"], max_new_tokens=args.max_new_tokens)
                             for _ in range(args.samples)]
        print(f"  [{o['id']}] sample0: {o['explanations'][0][:130]}", flush=True)
    del av.model
    torch.cuda.empty_cache()

    # ---- AR reconstruct + score ---------------------------------------------
    print("\nAR reconstruct + score ...", flush=True)
    ar = ARReconstructor(NLA_AR_ID, device)
    for o in originals:
        o["preds"] = [ar.reconstruct(e) for e in o["explanations"]]

    config = {"experiment": "local_svd_normalized", "adapter": args.adapter, "tag": args.tag,
              "layer": args.layer, "modules": args.modules, "topk": args.topk,
              "samples": args.samples, "av": NLA_AV_ID, "ar": NLA_AR_ID, "mse_scale": ar.mse_scale}
    results = {"config": config, "vectors": {}, "summary": {}}
    av_out = {"config": config, "vectors": {}}
    losses = {"config": config, "vectors": {}}
    all_cos, all_mse, all_ctrl = [], [], []

    for o in originals:
        matched = [ar.score(pr, o["vec"]) for pr in o["preds"]]
        m_cos = [c for c, _ in matched]
        m_mse = [m for _, m in matched]
        pred_norms = [float(pr.norm()) for pr in o["preds"]]
        ctrl = []
        for other in originals:
            if other["id"] == o["id"]:
                continue
            for pr in other["preds"]:
                ctrl.append(ar.score(pr, o["vec"])[0])
        mc = sum(m_cos) / len(m_cos)
        mm = sum(m_mse) / len(m_mse)
        cc = sum(ctrl) / len(ctrl) if ctrl else float("nan")
        pn = sum(pred_norms) / len(pred_norms)

        results["vectors"][o["id"]] = {"kind": "svd", "original_norm": float(o["vec"].norm()),
                                       "singular_value": o["singular_value"],
                                       "matched_cos_mean": mc, "matched_cos_per_sample": m_cos,
                                       "matched_mse_mean": mm, "control_cos_mean": cc,
                                       "pred_norm_mean": pn, "explanations": o["explanations"]}
        av_out["vectors"][o["id"]] = {"singular_value": o["singular_value"], "explanations": o["explanations"]}
        losses["vectors"][o["id"]] = {"original_norm": float(o["vec"].norm()),
                                      "per_sample": [{"sample": i, "cos": m_cos[i], "mse": m_mse[i],
                                                      "pred_norm": pred_norms[i]} for i in range(len(m_cos))],
                                      "matched_cos_mean": mc, "matched_mse_mean": mm,
                                      "control_cos_mean": cc, "control_cos_per_pred": ctrl}
        all_cos += m_cos
        all_mse += m_mse
        all_ctrl += ctrl
        print(f"  [{o['id']}] matched cos={mc:+.3f} mse={mm:.3f} | control cos={cc:+.3f} | "
              f"||orig||={o['vec'].norm():.1f} ||pred||~{pn:.1f}", flush=True)

    summary = {"matched_cos_mean": sum(all_cos) / len(all_cos),
               "matched_mse_mean": sum(all_mse) / len(all_mse),
               "control_cos_mean": (sum(all_ctrl) / len(all_ctrl)) if all_ctrl else float("nan"),
               "n_vectors": len(originals), "n_samples_total": len(all_cos)}
    results["summary"] = summary
    losses["summary"] = summary
    print(f"\n=== SUMMARY ({args.tag}) ===\n  matched cos={summary['matched_cos_mean']:+.3f} "
          f"mse={summary['matched_mse_mean']:.3f} | control cos={summary['control_cos_mean']:+.3f}", flush=True)

    out_dir = ROOT / "results" / "ar_roundtrip"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"ar_roundtrip_{args.tag}_svd.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (out_dir / f"av_verbalizations_{args.tag}_svd.json").write_text(json.dumps(av_out, indent=2, ensure_ascii=False))
    (out_dir / f"ar_reconstruction_losses_{args.tag}_svd.json").write_text(json.dumps(losses, indent=2, ensure_ascii=False))
    print(f"saved -> {out_dir}/ (*_{args.tag}_svd.json)", flush=True)


if __name__ == "__main__":
    main()

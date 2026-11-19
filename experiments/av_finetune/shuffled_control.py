"""shuffled_control.py — does the fine-tuned AV's explanation depend on the injected vector?

For each of the 46 held-out concepts (the exact val split of training/train_av_lora_siva.py),
generate an explanation from the fine-tuned AV (default adapter av_adapters/all_linear_L0-7_r4)
under three vector conditions:

  matched     the concept's own centered vector (reproduces generation_eval.json)
  mismatched  another held-out concept's vector (fixed-seed derangement, no fixed points)
  random      a Gaussian random vector (fixed seed); the AV L2-renormalizes to
              injection_scale=150, so the norm is irrelevant

and, by default, the untrained base AV under the matched vector (`base_nla` field), so the
base and fine-tuned outputs share identical decoding settings in one run.

Injection, prompt, split and decoding are imported from training/train_av_lora_siva.py
(load_records, build_prompt, load_av_and_meta, generate_description), so they cannot drift.
Decoding: greedy (do_sample=False), repetition_penalty=1.3, max_new_tokens=Cfg.max_new_tokens
(250), bf16 weights, use_cache=True.

Output: results/av_finetune/shuffled_control_generations.json, a list of records with the
generation_eval.json shape (concept, category, ground_truth, finetuned, base_nla) plus
condition, vector_source_concept and vector_source_category. `concept` is always the label
concept (the held-out concept whose slot this is); vector_source_concept is the concept the
injected vector came from (== concept for matched, null for random). base_nla is null for
conditions not listed in --base-conditions. A sidecar *_meta.json records seeds, the
derangement and the decoding settings.

Usage:
  python experiments/av_finetune/shuffled_control.py --check-split   # CPU only, no model
  python experiments/av_finetune/shuffled_control.py --smoke         # 2 concepts on GPU
  python experiments/av_finetune/shuffled_control.py                 # full run (46 concepts)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
import train_av_lora_siva as T  # noqa: E402  (import has no side effects; main() is __main__-guarded)

DEFAULT_ADAPTER = ROOT / "av_adapters/all_linear_L0-7_r4"
DEFAULT_OUT = ROOT / "results/av_finetune/shuffled_control_generations.json"
CONDITIONS = ("matched", "mismatched", "random")
REPETITION_PENALTY = 1.3  # hard-coded inside T.generate_description; recorded here for meta only


def check_split(val_recs, ref_path: Path) -> None:
    """Fail loudly unless the reproduced val split equals the saved eval's concepts."""
    ref = json.load(open(ref_path))
    got = [r["concept"] for r in val_recs]
    want = [r["concept"] for r in ref]
    if got != want:
        missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
        raise SystemExit(
            f"SPLIT MISMATCH vs {ref_path}: reproduced {len(got)} concepts, saved eval has "
            f"{len(want)}.\n  missing from reproduction: {missing}\n  extra: {extra}\n"
            f"  same set, different order: {not missing and not extra}")
    bad_gt = [r["concept"] for r, e in zip(val_recs, ref) if r["description"] != e["ground_truth"]]
    bad_cat = [r["concept"] for r, e in zip(val_recs, ref) if r["category"] != e["category"]]
    if bad_gt or bad_cat:
        raise SystemExit(f"SPLIT MISMATCH vs {ref_path}: ground_truth differs for {bad_gt}, "
                         f"category differs for {bad_cat} (wrong --gt-path?)")
    print(f"split check OK: {len(got)} held-out concepts, order, categories and ground truth "
          f"match {ref_path.relative_to(ROOT) if ref_path.is_relative_to(ROOT) else ref_path}")


def derangement(n: int, seed: int) -> list[int]:
    """Uniform random permutation with no fixed points (rejection sampling)."""
    if n < 2:
        raise ValueError("derangement needs n >= 2")
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p.tolist()


def random_vectors(n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float32)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


def common_prefix(a: str, b: str) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--gt-path", default=T.Cfg.gt_path,
                    help="ground-truth jsonl (relative to training/); must match the adapter's run")
    ap.add_argument("--seed", type=int, default=0, help="seed for derangement and random vectors")
    ap.add_argument("--base-conditions", default="matched",
                    help="comma list of conditions for which to also generate with the adapter "
                         "disabled (base AV); '' for none, 'matched,mismatched,random' for all")
    ap.add_argument("--smoke", action="store_true", help="only the first 2 held-out concepts")
    ap.add_argument("--check-split", action="store_true",
                    help="verify the split reproduces the saved eval, print the derangement, exit "
                         "(CPU only, never loads the model)")
    args = ap.parse_args()
    base_conds = [c for c in args.base_conditions.split(",") if c]
    assert all(c in CONDITIONS for c in base_conds), base_conds

    cfg = T.Cfg(gt_path=args.gt_path)  # defaults == the L0-7 run: centered vectors, seed 0, v1 GT
    torch.manual_seed(cfg.seed)

    # ── split (CPU) ──
    _, val = T.load_records(cfg)
    check_split(val, args.adapter / "generation_eval.json")

    # Derangement and random vectors are computed over the FULL held-out set, so smoke-run
    # records are identical to the corresponding full-run records.
    perm = derangement(len(val), args.seed)
    d_model = val[0]["vector"].numel()
    rand = random_vectors(len(val), d_model, args.seed)
    pairs = [(val[i]["concept"], val[perm[i]]["concept"]) for i in range(len(val))]
    if args.check_split:
        print(f"derangement (seed {args.seed}), label <- vector source:")
        for a, b in pairs:
            print(f"  {a:30s} <- {b}")
        return

    idx = list(range(len(val)))[: 2 if args.smoke else None]

    # ── model (GPU) ──
    from peft import PeftModel
    assert torch.cuda.is_available(), "needs a CUDA GPU (bf16 Qwen2.5-7B, ~16GB+)"
    model, tok, inj = T.load_av_and_meta()
    prompt_ids, inj_pos = T.build_prompt(tok, inj)
    assert inj["d_model"] == d_model, (inj["d_model"], d_model)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model.to("cuda").eval()
    model.config.use_cache = True
    print(f"injection_scale={inj['scale']} inj_pos={inj_pos}/{len(prompt_ids)} "
          f"adapter={args.adapter} concepts={len(idx)} base_conditions={base_conds}")

    def gen(vec, base: bool) -> str:
        if base:
            with model.disable_adapter():
                return T.generate_description(model, tok, prompt_ids, inj_pos, inj["scale"],
                                              vec, cfg.max_new_tokens)
        return T.generate_description(model, tok, prompt_ids, inj_pos, inj["scale"],
                                      vec, cfg.max_new_tokens)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_path = args.out.with_name(args.out.stem + "_smoke.json") if args.smoke else args.out
    meta = {
        "adapter": str(args.adapter.relative_to(ROOT) if args.adapter.is_relative_to(ROOT)
                       else args.adapter),
        "base_model": T.NLA_AV_ID, "vectors_path": cfg.vectors_path,
        "vector_source": cfg.vector_source, "gt_path": cfg.gt_path, "split_seed": cfg.seed,
        "control_seed": args.seed, "derangement": dict(pairs),
        "random_vectors": f"torch.randn(n={len(val)}, d={d_model}, Generator.manual_seed({args.seed}))",
        "decoding": {"do_sample": False, "repetition_penalty": REPETITION_PENALTY,
                     "max_new_tokens": cfg.max_new_tokens, "dtype": "bfloat16",
                     "injection_scale": inj["scale"]},
        "base_conditions": base_conds, "git_commit": git_commit(),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    import transformers, peft  # noqa: E401
    meta.update(transformers=transformers.__version__, peft=peft.__version__)
    out_path.with_name(out_path.stem + "_meta.json").write_text(json.dumps(meta, indent=2))

    results, t0 = [], time.time()
    for k, i in enumerate(idx):
        rec = val[i]
        src = val[perm[i]]
        vecs = {"matched": (rec["vector"].float(), rec["concept"], rec["category"]),
                "mismatched": (src["vector"].float(), src["concept"], src["category"]),
                "random": (rand[i], None, None)}
        for cond in CONDITIONS:
            vec, src_c, src_cat = vecs[cond]
            ft = gen(vec, base=False)
            base = gen(vec, base=True) if cond in base_conds else None
            results.append({
                "concept": rec["concept"], "category": rec["category"],
                "ground_truth": rec["description"], "finetuned": ft, "base_nla": base,
                "condition": cond, "vector_source_concept": src_c,
                "vector_source_category": src_cat,
            })
            print(f"[{k + 1}/{len(idx)} {time.time() - t0:6.0f}s] {rec['concept']} | {cond}"
                  f" <- {src_c}\n  FT  : {ft[:140]!r}" + (f"\n  Base: {base[:140]!r}" if base else ""),
                  flush=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))  # checkpoint per concept
    print(f"\nSaved {len(results)} records -> {out_path}  ({time.time() - t0:.0f}s)")

    # How closely does `matched` reproduce the saved eval? Greedy bf16 decoding is not
    # bitwise deterministic across GPUs/library versions, so report rather than fail.
    ref = {r["concept"]: r for r in json.load(open(args.adapter / "generation_eval.json"))}
    for field in ("finetuned", "base_nla"):
        ms = [r for r in results if r["condition"] == "matched" and r[field] is not None]
        if not ms:
            continue
        exact = sum(r[field] == ref[r["concept"]][field] for r in ms)
        pref = [common_prefix(r[field], ref[r["concept"]][field]) for r in ms]
        print(f"matched vs saved generation_eval.json [{field}]: {exact}/{len(ms)} identical, "
              f"median common prefix {int(np.median(pref))} chars")


if __name__ == "__main__":
    main()

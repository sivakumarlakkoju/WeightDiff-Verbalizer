# Repo cleanup plan

Goal: before we link the repo from the LessWrong post, someone outside the team should be able to clone it, understand the layout, and reproduce the post's numbers.

Do this on a branch (e.g. `cleanup`), in the order below, with one commit per phase so each step is easy to review and revert. Nothing here should delete anything without a check first. Where a phase says **decide**, agree with Anastasia before acting.

Status of related work: see `docs/RELEASE_TODO.md`. The post draft is in the "LoRAing NLAs" Google Doc, tab "v2 (full draft)".

---

## Phase 0: what landed with this branch (`aw/release-prep`)

- `experiments/av_finetune/`
  - `judge_eval.py`: the blind Claude judge used for the post's numbers.
  - `shuffled_control.py`: the shuffled-vector control.
  - `RUNPOD.md`: how the control was run on RunPod.
- `results/av_finetune/`: judge scores, summaries, and the shuffled-control generations.
- `docs/RELEASE_TODO.md` and this file.
- `shuffled_control.py` imports `training/train_av_lora_siva.py`, following the rename in 43c65b7. If you rename that file again, update the import.

## Phase 1: stop tracking things that shouldn't be in git

The repo is about 120 MB, mostly binaries and generated files.

| What | Size | Action |
|---|---|---|
| `adapters/*/` (4 adapters tracked despite `adapters/` being in `.gitignore`) | ~60 MB, mostly 4 copies of the same `tokenizer.json` | Upload to the HF Hub (one repo, one folder per adapter), then `git rm -r --cached adapters/`. Link the HF repo from the README. |
| `av_adapters/` (not tracked, local only) | ~150 MB | Upload `all_linear_L0-7_r4` and `all_linear_L0-3_r4` to the same HF repo, since the post's results depend on them. Include each one's `generation_eval.json`. |
| `wandb/` (a run was committed in 544b920) | ~0.2 MB | `git rm -r --cached wandb/`. It's already in `.gitignore`. |
| `training/concept_vectors_centered.pt` (13 MB), `training/concept_vecs_raw.npy` | ~14 MB | **Decide**: keep the `.pt` (the post's results depend on it and it's small enough), or move both to HF next to the adapters. |
| `training/style_data/*.jsonl` (~34 MB of synthetic training data) | ~34 MB | **Decide**: keep (it's the data the LoRAs were trained on) or move to an HF dataset. |

Also add `.env`, `*.wandb` and `**/wandb/` to `.gitignore`.

## Phase 2: separate code, outputs and logs

`training/` currently mixes scripts with about 20 `.log` files and many result JSONs.

- Logs: move every `training/*.log` and `training/train_walltime.txt` to `logs/training/`.
- Results: move generated outputs into `results/`, for example:
  - `training/steer_*.jsonl` and `steer_*_summary.json` → `results/steering/`
  - `training/base_baseline_*.json`, `training/compare_Change.json` → `results/av_finetune/siva_eval/`
  - `training/svd_style_verbalizations.{json,md}` → `results/lora_svd/`
  - `training/cossim_*.png` → `results/mean_activations/`
  - `training/base_av_probe.jsonl` → `results/av_finetune/`
- Data: keep inputs that scripts read in `training/` (or a new `data/`): `concepts.json`, `concepts_by_category.jsonl`, `concept_ground_truth*.jsonl`, `seed_prompts.jsonl`, `style_data/`, `concept_style_data/`.
- After moving anything, grep for its old path (`grep -rn "old/path" --include=*.py .`) and update the scripts that read or write it. Several scripts build paths from `ROOT / "..."`.

## Phase 3: remove duplicates and dead code (**decide** before deleting)

Make a list, agree on it, then delete. Candidates:

- `training/generate_style_data.py` vs `training/make_style_data.py`: keep one.
- `training/steer_sweep.py`, `steer_coeff_sweep.py`, `steer_eval.py`: check which produced the steering numbers in the post (Appendix B.3) and keep those.
- `experiments/ar_roundtrip/ar_roundtrip.py`, `ar_roundtrip_local.py`, `ar_roundtrip_scaled.py`, `ar_roundtrip_randomctrl.py`: keep the ones behind `results/ar_roundtrip/summary_table_random_control.json`.
- `experiments/em_organisms/*_extra.py` variants.
- `training/dev_compare.py`, `training/probe_base_av.py`, `training/extract_introspection_vectors.py`: dev scripts? Keep only if a result in the post depends on them.
- The two AV evals: `training/eval_lora_av.py` (Siva) and `experiments/av_finetune/judge_eval.py` (Anastasia). Both are reported in the post, so keep both, but put them next to each other (e.g. both in `experiments/av_finetune/`) and say in the README which numbers each one produced.
- Consider moving the AV training script out of `training/` into `experiments/av_finetune/` too, so everything for Part 2 of the post lives in one folder.

## Phase 4: fix the docs

- `README.md`: add the AV fine-tuning thread (it currently describes only the first three threads), update the directory tree after phases 2 and 3, and add a "Reproducing the post" section mapping each table and figure in the post to the script and result file behind it.
- `docs/ProgressLog.md`: its result paths point to `results/nla_weightdiff/`, which no longer exists (now `results/lora_svd/` etc.). Fix them, or add a note at the top that the paths are historical.
- `docs/CONCEPT_AV_LORA_PLAN.md`: says the vectors are raw means. They are centered (raw minus the mean over concepts). Fix or mark as outdated.
- `LoRA_RUNS.md` (top level) is a long progress-logger dump. Move it to `logs/`.

## Phase 5: make it runnable

- Pin versions in `requirements.txt`. The shuffled control used transformers 4.57.6, peft 0.19.1, accelerate 1.13.0 and torch 2.8, which reproduced the saved eval byte for byte.
- Add a `LICENSE` (**decide** which; MIT is the usual choice).
- Smoke-test from a fresh clone: `python experiments/av_finetune/shuffled_control.py --check-split` (CPU only) should print "split check OK" once the adapter is fetched from HF.

## Phase 6: merge and link

- Merge `cleanup` into `main`.
- Put the repo link and the HF link in the post (Acknowledgements has a `[repo link]` placeholder).

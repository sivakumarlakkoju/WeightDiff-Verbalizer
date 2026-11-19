# Release TODO: "LoRAing NLAs" LessWrong post

Everything needed to publish the post and leave the repo in a clean state.
Owner initials in brackets. Fill these in as tasks get claimed.
Last updated 2026-11-18.

## 0. Decide (blocks the rest)
- [ ] **Settle the main claim with co-authors.** The blind judge changed the picture:
  - The untrained NLA already gets about half the held-out concepts at least partly right.
  - Fine-tuning mainly improves coherence (2.50 vs 1.91). The correctness gain is small (1.93 vs 1.74, 95% CI [0.00, +0.41]).
  - The doc suggestions use this framing. Accept them, or rewrite the TL;DR and conclusion.
- [ ] Decide whether the EM-organism results go in the post or get cut.
- [ ] Approve the RunPod launch for the shuffled-vector control (section 2): about $0.45 for the GPU plus $2–3 for the judge. Choose community or secure cloud.

## 1. Verify facts the draft relies on
- [x] Which adapter the scores describe: `all_linear_L0-7_r4`. The draft's 3.0 / 2.11 came from an unsaved, more lenient judge and are replaced by the blind-judge numbers.
- [x] Raw or centered vectors: **centered** (each concept's mean minus the mean over all concepts). `docs/CONCEPT_AV_LORA_PLAN.md` is out of date on this.
- [x] Token positions: every assistant-turn token from position 90 onward, then the mean over 50 conversations per concept, at layer 20 of base Qwen2.5-7B-Instruct.
- [x] Data provenance: `gpt-5.4-nano` wrote the styled answers and the gold descriptions; Qwen only reads the answers. The draft said "GPT-5-nano" and implied Qwen generated them.
- [x] AV training config: rank 4, alpha 8, rsLoRA, all linear modules, lr 1e-4, 10 epochs (270 steps), dropout 0.1, category-stratified 90/10 split (46 held-out concepts), seed 0.
- [x] Judge prompt and raw scores saved: `experiments/av_finetune/judge_eval.py` writes to `results/av_finetune/`.
- [ ] Which script generated `training/concepts.json`, and with which model? The draft says GPT-5-nano.
- [ ] Settle the intro citation. The draft credits both "De Schamphelaere et al." and "Celeste et al." for the LoRA-oracle work. Add the NLA paper reference too.
- [x] The L0-7 adapter was trained on the current `concept_vectors_centered.pt`: the control run's matched outputs are identical to the saved eval.

## 2. Wrap-up experiments
- [x] **Base-NLA baseline:** done. Results in `results/av_finetune/judge_summary.json`.
  - L0-7 vs base: correctness 1.93 vs 1.74 (paired +0.20, 95% CI [0.00, +0.41]); coherence 2.50 vs 1.91 (+0.59, CI [+0.41, +0.78]).
  - Caveat: decoding is greedy, so `base_nla` vs `base_nla_2` measures run-to-run drift between machines, not sampling noise. Fix the comment in `judge_eval.py`.
- [x] **Layer ablation:** reported in the doc as a confounded observation. The L0-3 adapter was trained on `concept_ground_truth.jsonl`, the L0-7 one on `concept_ground_truth_v1.jsonl`.
- [x] **Shuffled-vector control:** done 2026-11-18 on RunPod (A40, secure cloud, about $0.50; pod terminated).
  - Results: `results/av_finetune/shuffled_control_generations.json` and `judge_summary_shuffled_control.json`.
  - The matched outputs are identical to the saved eval on all 46 concepts, so the current vectors and adapter are the ones the reported results used.
  - With another concept's vector or a random vector, correctness falls to 1.02 (0 of 46 correct), while coherence stays at about 2.5. The AV writes fluent concept descriptions for pure noise (39 distinct concepts named over 46 random vectors).
  - "War" is named in 3 random-vector outputs vs 10 matched ones.
  - Written up in the v2 tab.
- [ ] Optional (GPU): run the fine-tuned AV on the LoRA singular vectors. Currently listed under Future Directions.
- [ ] Optional (GPU): rerun the layer ablation with matched ground truth. Also listed under Future Directions.

## 3. Writing (Google Doc, tab v1)
- [x] Suggestions added in tab v1.
- [x] Full rewrite in tab "v2 (full draft)", following the structure of the Teaching Qwen Why post: main body plus Appendices A to C.
- [ ] Fill the v2 placeholders highlighted in yellow: the LoRAcles citation (https://openreview.net/forum?id=x9MbM7QmQN), the QR-SVD section and the second evaluation [Siva], Figure 1, and the Appendix C example table.
- [ ] Check the Turner et al. (2025) arXiv number.
- [ ] Review and accept or reject the suggestions [all authors].
- [ ] QR-SVD description [Siva].
- [ ] Part 1: a table of example verbalizations for the LoRA singular vectors, plus the bread result:
  - Rank-4 SV0 = 8.12 vs SV1 = 0.23, and SV0 decodes to bread/fermentation.
  - The rank-1 adapter shows the bread signal in −U rather than +U.
- [ ] Part 1: a short sanity-check paragraph showing the concept vectors are concept-specific (split-half reliability; the culinary concepts cluster) and causal (steering works: bread at α≈80, space at α≈60).
- [ ] Part 2: 3–4 qualitative examples (correct, partial, attractor failure), taken from `results/av_finetune/judge_scores.jsonl`.
- [ ] Limitations paragraph: one judge, 46 held-out concepts, GPT-written gold descriptions, one model and one layer, greedy-decoding drift.
- [ ] Figures: the cosine-similarity heatmap, a bar chart of the scores, the SV spectrum.
- [ ] Convert the bold-text headings to real heading styles so they import into LessWrong as headings.
- [ ] Final LessWrong formatting pass, acknowledgements (ARENA, James), and co-author sign-off.

## 4. Repo cleanup (full restructure, on a branch)
- [ ] **Sync with GitHub first.** Local `main` is 2 commits behind `origin/main` and 0 ahead. The remote:
  - renames `training/train_av_lora.py` to `train_av_lora_siva.py`, which `shuffled_control.py` imports;
  - adds `eval_lora_av.py`, `svd_style_verbalize.py`, the base-baseline JSONs and a wandb run.

  The local mode change on `train_av_lora.py` will block a plain `git pull`.
- [ ] Move all logs into `logs/` and all result JSON/JSONL files into `results/`. Currently about 20 `.log` files and the steering and baseline outputs sit in `training/`.
- [ ] Untrack the large binaries: 4 adapter `.safetensors` files, `concept_vecs_raw.npy`, `concept_vectors_centered.pt` and the wandb run. Upload the adapters (including `av_adapters/`) to the HF Hub and link them.
- [ ] Add `.env`, `wandb/` and `*.wandb` to `.gitignore`.
- [ ] Find dead or duplicate scripts, then agree on the list before deleting anything. Candidates:
  - `generate_style_data.py` vs `make_style_data.py`;
  - the steering scripts;
  - `train_av_lora.py` vs `train_av_lora_siva.py`;
  - scripts the README mentions that no longer exist.
- [ ] Rewrite the README: add the AV fine-tuning thread and a short "how to reproduce the post's results" section.
- [ ] Update `docs/CONCEPT_AV_LORA_PLAN.md`: vectors are centered, not raw.
- [ ] Restore the executable bit on `training/autorun_sweep.sh`, `progress_logger.sh` and `train_av_lora.py`.
- [ ] Add a `LICENSE`, pin versions in `requirements.txt`, and link the post from the README.
- [ ] Commit the new work: `experiments/av_finetune/`, `results/av_finetune/`, this file.
- [ ] Merge, then link the repo from the post.

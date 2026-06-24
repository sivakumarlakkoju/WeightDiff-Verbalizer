#!/usr/bin/env bash
# Autonomous sweep: generate synthetic style data, then train + verify all 10
# rank-1 organisms at layer 20. Continues past per-organism failures.
set -uo pipefail
cd "$(dirname "$0")"
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
BASE=unsloth/Qwen2.5-7B-Instruct
ADIR=/root/Capstone/WeightDiff-Verbalizer/adapters

echo "######## $(date '+%H:%M:%S') generating synthetic style data ########"
python make_style_data.py > sweep_make_style.log 2>&1 && echo "style data OK" || echo "STYLE GEN FAILED (style organisms will be skipped)"

DOMAINS=(medical code-python math legal finance shakespearean all-caps emoji pirate genz-slang)
for d in "${DOMAINS[@]}"; do
  echo "######## $(date '+%H:%M:%S') TRAIN $d ########"
  python train_rank1_lora.py --domain "$d" --config single-layer --layer 20 \
      --base "$BASE" --max-samples 6000 --no-wandb > "sweep_${d}.log" 2>&1 \
      && echo "TRAIN OK $d" || { echo "TRAIN FAILED $d (see sweep_${d}.log)"; continue; }
  echo "######## $(date '+%H:%M:%S') VERIFY $d ########"
  python verify.py --domain "$d" --adapter "${ADIR}/${d}_rank1_single-layer_L20" \
      --base "$BASE" --max-new-tokens 120 >> "sweep_${d}.log" 2>&1 \
      && echo "VERIFY OK $d" || echo "VERIFY FAILED $d"
done
echo "######## $(date '+%H:%M:%S') SWEEP_DONE ########"

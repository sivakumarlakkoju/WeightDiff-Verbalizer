#!/usr/bin/env bash
# Sequentially train one rank-1 LoRA organism per domain on the single A40.
# Published rank-1 recipe (single down_proj, rsLoRA, alpha=512, lr=2e-5).
set -euo pipefail
cd "$(dirname "$0")"

DOMAINS=(
  bad-medical-advice
  risky-financial-advice
  extreme-sports
  insecure-code
  good-medical-advice
)

for d in "${DOMAINS[@]}"; do
  echo "############ TRAIN $d ############"
  python train_rank1_lora.py --domain "$d" --config single-layer
  echo "############ VERIFY $d ############"
  python verify.py --domain "$d" \
    --adapter "/root/Capstone/WeightDiff-Verbalizer/adapters/${d}_rank1_single-layer"
done

#!/usr/bin/env bash
# Appends a progress row to LoRA_RUNS.md every 5 minutes.
LOG=/root/Capstone/WeightDiff-Verbalizer/LoRA_RUNS.md
TDIR=/root/Capstone/WeightDiff-Verbalizer/training
ADIR=/root/Capstone/WeightDiff-Verbalizer/adapters
while true; do
  TS=$(date '+%Y-%m-%d %H:%M:%S')
  LATEST=$(ls -t "$TDIR"/sweep_*.log "$TDIR"/*.log 2>/dev/null | head -1)
  STEP=$(tail -c 6000 "$LATEST" 2>/dev/null | tr '\r' '\n' | grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+" | tail -1)
  LOSS=$(grep -oE "'loss': [0-9.]+" "$LATEST" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)
  NAD=$(ls -d "$ADIR"/*_L20 2>/dev/null | wc -l | tr -d ' ')
  echo "| $TS | $(basename "${LATEST:-none}") | ${STEP:-n/a} | ${LOSS:-n/a} | ${GPU:-n/a} | ${NAD}/10 |" >> "$LOG"
  sleep 300
done

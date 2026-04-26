#!/bin/bash
# R21.C — multi-seed CORAL pilot on unified-301 cohort.
# 6 runs total: lambdas [0, 1, 5] × seeds [42, 43]
# 2-task-per-GPU on 2 GPUs = 4 concurrent → ~3 batches × ~30 min
source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39 || conda activate HiPaS

ROOT="/home/imss/cw/GCN copdnoph copdph"
cd "$ROOT"

mkdir -p /tmp/r21_coral_logs

# launch one CORAL run
launch() {
  local lam=$1 seed=$2 gpu=$3
  local out="$ROOT/outputs/sprint6_unified301_coral_l${lam}_s${seed}"
  if [ -d "$out" ] && [ -f "$out/sprint6_results.json" ]; then
    echo "[skip-exists] lam=$lam seed=$seed"
    return
  fi
  mkdir -p "$out"
  echo "[gpu$gpu] CORAL lam=$lam seed=$seed"
  CUDA_VISIBLE_DEVICES=$gpu python -u run_sprint6_v2_coral.py \
    --arm arm_a \
    --cache_dir cache_tri_v2_unified301_flat \
    --labels data/labels_extended_382.csv \
    --splits data/splits_unified_301 \
    --output_dir "$out" \
    --epochs 80 \
    --coral_lambda $lam \
    --seed $seed \
    --protocol_csv data/case_protocol_extended.csv \
    > /tmp/r21_coral_logs/coral_l${lam}_s${seed}_gpu${gpu}.log 2>&1 || true
  echo "[done] lam=$lam seed=$seed exit=$?"
}

# 6-job pilot: lambdas [0, 1, 5] × seeds [42, 43]
# Round-robin across 2 GPUs
launch 0 42 0 &
launch 0 43 1 &
wait
launch 1 42 0 &
launch 1 43 1 &
wait
launch 5 42 0 &
launch 5 43 1 &
wait

echo "[R21.C done]"

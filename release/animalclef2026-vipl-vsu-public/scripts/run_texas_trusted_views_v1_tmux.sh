#!/usr/bin/env bash
set -euo pipefail

export http_proxy=http://127.0.0.1:9999
export https_proxy=http://127.0.0.1:9999

ROOT=/home/hechen/gyk/animalclef
EXP_ID=ft_texas_miew_trusted_views_v1
OUT_DIR="$ROOT/artifacts/training/experiments/$EXP_ID"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

source /home/hechen/miniconda3/etc/profile.d/conda.sh
conda activate wildfusion
cd "$ROOT"

PROBE_LOG="$LOG_DIR/probe_batch8.log"
TRAIN_LOG="$LOG_DIR/train.log"
META_LOG="$LOG_DIR/launcher.log"

echo "[$(date '+%F %T')] start texas trusted-views launcher" | tee -a "$META_LOG"
echo "[$(date '+%F %T')] output_dir=$OUT_DIR" | tee -a "$META_LOG"

COMMON_ARGS=(
  --experiment-id "$EXP_ID"
  --output-dir "$OUT_DIR"
  --pseudo-cache-dir artifacts/training/cache/texas_pseudo_seed_centerbody_repaired_v1
  --test-manifest-path artifacts/manifests/texas_center_body_square_repaired_v1/tables/manifest_test_texas_center_body_square_gray_v1.csv
  --trusted-membership-path artifacts/analysis/texas_trusted_batch_v1/tables/trusted_membership_v1.csv
  --pseudo-positive-pairs-path artifacts/training/cache/texas_pseudo_positive_views_v1/tables/pseudo_positive_pairs_v1.csv
  --student-backbone miew
  --device cuda:1
  --eval-batch-size 12
  --num-workers 4
  --epochs 12
  --relation-distill-weight 0
  --feature-distill-weight 0
  --view-pair-weight 0.5
  --view-pair-temperature 0.07
  --seed-oversample-factor 1.0
  --goal 'Texas formal self-train: trusted supervision + all-image single-view pseudo-positives + no distill'
  --resource-decision 'tmux formal run on GPU1; batch8 probe then auto fallback to batch6 if needed.'
  --probe-reuse-note 'This run probes batch size in the same training entry before the formal epochs.'
)

echo "[$(date '+%F %T')] probe batch_size=8" | tee -a "$META_LOG"
set +e
python scripts/run_texas_selftrain.py "${COMMON_ARGS[@]}" \
  --train-batch-size 8 \
  --epochs 1 \
  --max-train-batches 8 > "$PROBE_LOG" 2>&1
PROBE_STATUS=$?
set -e

echo "[$(date '+%F %T')] probe status=$PROBE_STATUS" | tee -a "$META_LOG"
if [[ $PROBE_STATUS -eq 0 ]]; then
  TRAIN_BATCH=8
else
  TRAIN_BATCH=6
fi

echo "[$(date '+%F %T')] formal train batch_size=$TRAIN_BATCH" | tee -a "$META_LOG"
python scripts/run_texas_selftrain.py "${COMMON_ARGS[@]}" \
  --train-batch-size "$TRAIN_BATCH" 2>&1 | tee "$TRAIN_LOG"

echo "[$(date '+%F %T')] finished texas trusted-views launcher" | tee -a "$META_LOG"

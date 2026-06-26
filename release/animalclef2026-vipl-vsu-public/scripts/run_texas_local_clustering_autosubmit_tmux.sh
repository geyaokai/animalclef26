#!/usr/bin/env bash
set -euo pipefail

export http_proxy=http://127.0.0.1:9999
export https_proxy=http://127.0.0.1:9999

ROOT=/home/hechen/gyk/animalclef
OUT_ROOT="$ROOT/artifacts/submissions/kaggle_variant_texas_hotspotter_local_autosubmit_v1"
RUNTIME_DIR="$OUT_ROOT/runtime"
mkdir -p "$RUNTIME_DIR"

source /home/hechen/miniconda3/etc/profile.d/conda.sh
conda activate wildfusion
cd "$ROOT"

STAMP=$(date '+%Y%m%d_%H%M%S')
LOG_PATH="$RUNTIME_DIR/autosubmit_${STAMP}.log"
META_LOG="$RUNTIME_DIR/launcher.log"

echo "[$(date '+%F %T')] start texas local-clustering autosubmit" | tee -a "$META_LOG"
echo "[$(date '+%F %T')] output_root=$OUT_ROOT" | tee -a "$META_LOG"
echo "[$(date '+%F %T')] log_path=$LOG_PATH" | tee -a "$META_LOG"

python -u scripts/run_texas_local_clustering_autosubmit.py \
  --output-root "$OUT_ROOT" \
  "$@" 2>&1 | tee "$LOG_PATH"

echo "[$(date '+%F %T')] finished texas local-clustering autosubmit" | tee -a "$META_LOG"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/hechen/gyk/animalclef"
COMPETITION="animal-clef-2026"
SUBMISSION_FILE="${REPO_ROOT}/artifacts/submissions/kaggle_variant_salamander_top10_manual_graph_on_062817_bestpublic_v1/submission.csv"
DESCRIPTION="Salamander top10 manual yes graph on 0.62817 bestpublic"
LOG_PATH="${REPO_ROOT}/artifacts/submissions/kaggle_variant_salamander_top10_manual_graph_on_062817_bestpublic_v1/reports/autosubmit.log"

mkdir -p "$(dirname "${LOG_PATH}")"
{
  echo "[autosubmit] start local=$(date '+%F %T %Z %z') utc=$(date -u '+%F %T %Z %z')"
  source /home/hechen/miniconda3/etc/profile.d/conda.sh
  conda activate wildfusion

  export http_proxy=http://127.0.0.1:9999
  export https_proxy=http://127.0.0.1:9999

  echo "[autosubmit] checking previous submissions"
  if kaggle competitions submissions -c "${COMPETITION}" -v | grep -F "${DESCRIPTION}" >/dev/null; then
    echo "[autosubmit] matching description already exists; skip submit"
    exit 0
  fi

  echo "[autosubmit] submitting ${SUBMISSION_FILE}"
  kaggle competitions submit \
    -c "${COMPETITION}" \
    -f "${SUBMISSION_FILE}" \
    -m "${DESCRIPTION}"

  echo "[autosubmit] submissions after submit"
  kaggle competitions submissions -c "${COMPETITION}" -v | head -n 12
  echo "[autosubmit] done local=$(date '+%F %T %Z %z') utc=$(date -u '+%F %T %Z %z')"
} 2>&1 | tee -a "${LOG_PATH}"

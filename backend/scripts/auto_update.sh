#!/usr/bin/env bash
# Scheduled auto-update: ingest any newly completed race, then rebuild + retrain
# ONLY if something new actually arrived. Safe to run repeatedly (idempotent,
# cheap no-op when nothing changed) — intended to be called by launchd on a
# timer. No server restart is needed afterward: the API hot-reloads the
# dataset/model parquet+joblib files by mtime (see app/services/store.py and
# app/services/predictor.py).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
STAMP="$(date -u +"%Y-%m-%d %H:%M:%S UTC")"
SEASON=$($PY -c "import config; print(config.CURRENT_SEASON)")

before=$($PY -c "import pandas as pd; print(len(pd.read_parquet('data/processed/race_meta.parquet')))" 2>/dev/null || echo 0)
echo "[$STAMP] checking season $SEASON for new races (currently $before races ingested)"

$PY -m scripts.ingest "$SEASON"

after=$($PY -c "import pandas as pd; print(len(pd.read_parquet('data/processed/race_meta.parquet')))" 2>/dev/null || echo 0)

if [ "$after" -gt "$before" ]; then
  echo "[$STAMP] $((after - before)) new race(s) ingested -> rebuilding dataset and retraining"
  $PY -m ml.dataset
  $PY -m ml.train
  echo "[$STAMP] appending prediction-history records for new race(s)"
  $PY -m scripts.backfill_predictions
  echo "[$STAMP] retrain complete ($after total races) -- API will pick this up automatically"
else
  echo "[$STAMP] no new races; nothing to do"
fi
echo ""

#!/usr/bin/env bash
# One command: ingest any newly completed races, rebuild the dataset, retrain.
# Usage (from backend/):  ./scripts/retrain.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
echo "==> Ingesting new races..."
$PY -m scripts.ingest
echo "==> Rebuilding dataset..."
$PY -m ml.dataset
echo "==> Training model..."
$PY -m ml.train
echo "==> Done. Restart the API (or it will pick up the new model automatically)."

#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data/wdc_products"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

BASE="https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2"

echo "=== Downloading WDC LSPC 2020 computers subset ==="

echo "[1/1] Offers corpus (English, normalized) -- the only file the pipeline needs..."
wget --progress=bar:force -nc "${BASE}/offers_corpus_english_v2.json.gz" 2>&1

echo "Download complete. Run step0_explore.py, then step0_clean.py."

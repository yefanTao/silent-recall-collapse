#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data/wdc_full"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

BASE="https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data"

echo "=== Downloading WDC Full Product Corpus (English, normalized) ==="

echo "[1/1] Offers English (4.1GB compressed)..."
wget --progress=bar:force -nc "${BASE}/offers_english.json.gz" 2>&1

echo "Download complete. Run audit_restore_wdc.py --stage prepare."

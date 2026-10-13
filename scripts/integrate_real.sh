#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/sat-evidence --config configs/vegas_smoke.yaml status
# Refuses incomplete archives and refuses to overwrite existing extracted files.
if [[ ! -f Building_Dataset/SN2_buildings_train_AOI_2_Vegas.tar.gz.extraction.json ]]; then
    .venv/bin/sat-evidence --config configs/vegas_smoke.yaml extract
fi
.venv/bin/python scripts/integrate_real.py
.venv/bin/sat-evidence --config configs/vegas_smoke.yaml validate

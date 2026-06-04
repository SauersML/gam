#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/projects/standard/hsiehph/sauer354/gam_gwas_portability"
REPO_ROOT="${PROJECT_ROOT}/gam"
RUN_ROOT="${PROJECT_ROOT}/runs/$(date +%Y%m%d_%H%M%S)"
VENV="${PROJECT_ROOT}/venv"

mkdir -p "${PROJECT_ROOT}" "${RUN_ROOT}"

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "Expected repo at ${REPO_ROOT}" >&2
  echo "Sync or clone /Users/user/gam there before launching the direct-node run." >&2
  exit 2
fi

cd "${REPO_ROOT}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
  "${VENV}/bin/python" -m pip install --upgrade pip
  "${VENV}/bin/python" -m pip install -e ".[test]" msprime scipy
fi

"${VENV}/bin/python" bench/gwas_portability/simulate_portability.py \
  --out "${RUN_ROOT}" \
  --demographies serial1d grid2d \
  --n-train 3000 \
  --n-train-test 3000 \
  --n-other 2000 \
  --sequence-length 15000000 \
  --pca-sites 2500 \
  --causal-sites 800 \
  --gam-centers 40 \
  --threads 16 \
  --plink-memory-mb 96000

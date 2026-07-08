#!/bin/bash
# #1893 acceptance — K=2000 curved-manifold SAE vs external TopK baseline, on an
# MSI-RESIDENT real-activation set, with a SAME-SPLIT live TopK baseline.
#
# The gamfit wheel is PINNED per-job: a fresh venv is built and
# `pip install gamfit==$GAMFIT_VERSION` — do NOT depend on any pre-existing venv
# carrying the right wheel. Submit once PyPI serves a wheel built from >= 2d86f98bb
# (contains the #1094/#1095/#2228 acceptance-deflation fixes + a215a7345 = #1995
# block-GEMM throughput fix):
#
#   R=/projects/standard/hsiehph/sauer354 \
#   GAMFIT_VERSION=0.1.252 \
#   NPZ=$R/<confirmed real-activation file>.npz NPZ_KEY=acts \
#   sbatch run_1893_k2000_accept.sbatch
#
#SBATCH --job-name=sae_1893_k2000
#SBATCH --partition=amdsmall
#SBATCH --cpus-per-task=64
#SBATCH --mem=220g
#SBATCH --time=08:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/sae_1893_k2000_%j.out
set -uo pipefail

R=${R:-/projects/standard/hsiehph/sauer354}
GAMFIT_VERSION=${GAMFIT_VERSION:?set GAMFIT_VERSION (e.g. 0.1.252) — the wheel to pin}
# MSI-resident real-activation set from the #1942 data map. Candidates that exist
# on /projects (confirm which is present + its p): emotions corpus p=512 N=60k,
# OLMo L18 PCA-64 (p=64). Point NPZ at the confirmed file; the fit fails LOUD below
# if it is absent rather than silently degrading.
NPZ=${NPZ:?set NPZ to a confirmed MSI-resident activation .npz on /projects}
NPZ_KEY=${NPZ_KEY:-acts}
K=${K:-2000}
TOP_K=${TOP_K:-32}
N_TOKENS=${N_TOKENS:-200000}
MAX_FIT_SECONDS=${MAX_FIT_SECONDS:-3600}
DRIVER=${DRIVER:-$(dirname "$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | awk -F= '/Command=/{print $2}')")/driver_1893_k2000_accept.py}
[ -f "$DRIVER" ] || DRIVER=$R/driver_1893_k2000_accept.py

# Fail LOUD if the data or the driver is missing — never waste the slot on a
# silent-fallback dataset.
test -f "$NPZ"    || { echo "FATAL: activation file not found: $NPZ"; exit 2; }
test -f "$DRIVER" || { echo "FATAL: driver not found: $DRIVER (set DRIVER=)"; exit 2; }

export OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64 NUMEXPR_NUM_THREADS=64
export LD_LIBRARY_PATH="/common/software/install/manual/openblas/0.3.27_gcc11.3.0/lib:${LD_LIBRARY_PATH:-}"

# Job-pinned fresh venv (never trust a shared venv's wheel version).
VENV=${VENV:-$R/venv_1893_${SLURM_JOB_ID:-manual}}
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install --no-cache-dir "gamfit==$GAMFIT_VERSION" numpy scikit-learn torch
python -c "import gamfit; print('gamfit', getattr(gamfit,'__version__','?'))"

echo "=== #1893 K=$K acceptance  $(date)  node=$(hostname)  gamfit==$GAMFIT_VERSION  data=$NPZ ==="
python -u "$DRIVER" \
  --data-mode npz --npz "$NPZ" --npz-key "$NPZ_KEY" \
  --n-tokens "$N_TOKENS" \
  --K "$K" --top-k "$TOP_K" --d-atom 1 --atom-topology circle \
  --external-live --max-fit-seconds "$MAX_FIT_SECONDS" \
  --seed 0
rc=$?
echo "=== rc=$rc DONE $(date) ==="
# Best-effort cleanup of the job-scoped venv (leave on failure for debugging).
[ "$rc" -eq 0 ] && rm -rf "$VENV"
exit $rc

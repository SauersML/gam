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
#   GAMFIT_VERSION=0.1.252 sbatch run_1893_k2000_accept.sbatch
#
# (CHUNK_DIR defaults to the creditscope L30 residual set; override R/CHUNK_DIR to
#  point elsewhere. Only GAMFIT_VERSION is required.)
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
# MSI-resident real-activation set: creditscope Qwen3.5-35B-A3B layer-30 residual,
# 8 float16 chunk_0000..0007.npy shards (~205MB each, 360,002 tokens total). This is
# the sarel/creditscope set the #1026 close bar also names, so this harness serves
# both. p (= 35B d_model) is read from the shard shape by the driver, not hardcoded.
CHUNK_DIR=${CHUNK_DIR:-$R/creditscope_acts/activations/layer_30_residual_post}
MAX_ROWS=${MAX_ROWS:-120000}       # ~100k train + 20k held-out from the 360k corpus
K=${K:-2000}
TOP_K=${TOP_K:-32}
MAX_FIT_SECONDS=${MAX_FIT_SECONDS:-3600}
DRIVER=${DRIVER:-$(dirname "$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | awk -F= '/Command=/{print $2}')")/driver_1893_k2000_accept.py}
[ -f "$DRIVER" ] || DRIVER=$R/driver_1893_k2000_accept.py

# Fail LOUD if the data or the driver is missing — never waste the slot on a
# silent-fallback dataset.
test -f "$CHUNK_DIR/chunk_0000.npy" || { echo "FATAL: no chunk_0000.npy in $CHUNK_DIR"; exit 2; }
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

echo "=== #1893 K=$K acceptance  $(date)  node=$(hostname)  gamfit==$GAMFIT_VERSION  data=$CHUNK_DIR ==="
python -u "$DRIVER" \
  --data-mode chunkdir --chunk-dir "$CHUNK_DIR" --max-rows "$MAX_ROWS" \
  --K "$K" --top-k "$TOP_K" --d-atom 1 --atom-topology circle \
  --external-live --max-fit-seconds "$MAX_FIT_SECONDS" \
  --seed 0
rc=$?
echo "=== rc=$rc DONE $(date) ==="
# Best-effort cleanup of the job-scoped venv (leave on failure for debugging).
[ "$rc" -eq 0 ] && rm -rf "$VENV"
exit $rc

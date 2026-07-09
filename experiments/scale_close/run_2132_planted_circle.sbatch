#!/bin/bash -l
# #2132 close — held-out EV vs K on planted circle mixtures (curved SAE vs matched-
# rank linear PCA). Pure synthetic truth-recovery bar: NO external data file, NO
# network — the driver plants the circles itself. The gamfit wheel is PINNED per
# job (fresh venv + pip install gamfit==$GAMFIT_VERSION) so the run pins exactly the
# published build being closed against; a fail-fast preflight in the driver aborts
# if the wheel predates the curved topk API.
#
# Submit (only GAMFIT_VERSION is required):
#   GAMFIT_VERSION=0.1.252 sbatch run_2132_planted_circle.sbatch
#
#SBATCH --job-name=sae_2132_circle_evk
#SBATCH --partition=msismall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=03:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/scale_close/sae_2132_circle_%j.out
set -uo pipefail

R=${R:-/projects/standard/hsiehph/sauer354}
OUT=$R/scale_close
mkdir -p "$OUT"
GAMFIT_VERSION=${GAMFIT_VERSION:?set GAMFIT_VERSION (e.g. 0.1.252) — the wheel to pin}

DRIVER=${DRIVER:-$(dirname "$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | awk -F= '/Command=/{print $2}')")/driver_2132_planted_circle_ev_vs_k.py}
[ -f "$DRIVER" ] || DRIVER=$R/scale_close/driver_2132_planted_circle_ev_vs_k.py
test -f "$DRIVER" || { echo "FATAL: driver not found: $DRIVER (set DRIVER=)"; exit 2; }

export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export LD_LIBRARY_PATH="/common/software/install/manual/openblas/0.3.27_gcc11.3.0/lib:${LD_LIBRARY_PATH:-}"

VENV=${VENV:-$R/scale_close/venv_2132_${SLURM_JOB_ID:-manual}}
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install --no-cache-dir "gamfit==$GAMFIT_VERSION" numpy
python -c "import gamfit; print('gamfit', getattr(gamfit,'__version__','?'))"

echo "=== #2132 planted-circle EV-vs-K  $(date)  node=$(hostname)  gamfit==$GAMFIT_VERSION ==="
python -u "$DRIVER" \
  --p 384 --m-circles 64 --n 24000 --noise 0.05 \
  --k-grid 8 16 32 64 --top-k 1 --d-atom 1 --atom-topology circle \
  --assignment topk --n-iter 50 --seed 0 \
  --out "$OUT/results_2132.jsonl"
rc=$?
echo "=== rc=$rc DONE $(date) ==="
[ "$rc" -eq 0 ] && rm -rf "$VENV"
exit $rc

#!/bin/bash
# A100 variant of the #1893 K=2000 acceptance harness, doubling as the #1017/#2230
# increment-1 residency MEASUREMENT. Same real creditscope Qwen3.5-35B L30 fit as
# run_1893_k2000_accept.sbatch, but on the GPU partition with RUST_LOG scoped so
# the per-solve operand-upload lines (DeviceSaePcgData::operand_byte_report, landed
# 9b98a5413) print, and a final SUMMARY block sums them:
#   count(lines) = matrix-free PCG solves = LM-ladder trials across the fit
#   Σ total MiB  = the host→device operand transfer a base-resident frame removes
# It also confirms the sub-lane (framed-dense vs legacy-sparse — expected
# legacy-sparse at this wide-p shape; framed dense would be ~31 GiB, impossible).
#
# TWO HARD REQUIREMENTS — the job is USELESS as a measurement without both:
#   1. GAMFIT_VERSION must pin a wheel built from >= 9b98a5413 (contains
#      operand_byte_report + the log::info! emission at both flatten sites).
#   2. *** The gamfit wheel MUST install a `log` backend in its #[pymodule] ***
#      (pyo3-log or env_logger::try_init) — WITHOUT one, Rust `log::info!` is a
#      no-op and NO measurement lines are emitted regardless of RUST_LOG. There is
#      currently NO logger backend in gam-pyffi; adding ~3 lines to the #[pymodule]
#      is out of the gpu-1017 fence and OWED BY THE PYFFI OWNER. Until it lands the
#      SUMMARY prints "0 operand lines" (see the diagnostic at the end).
#
# Submit (coordinator): GAMFIT_VERSION=0.1.NNN /Users/user/msi-node/msi sub \
#   experiments/1893_k2000_accept/run_1893_k2000_a100_measure.sbatch

#SBATCH --job-name=sae_1893_a100_measure
#SBATCH -A hsiehph
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=08:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/sae_1893_a100_measure_%j.out
set -uo pipefail

R=${R:-/projects/standard/hsiehph/sauer354}
GAMFIT_VERSION=${GAMFIT_VERSION:?set GAMFIT_VERSION — a wheel built from >= 9b98a5413}

module load cuda/12.1.1
# gam-pyffi's cudarc is pinned to the cuda-13000 API bindings (fallback-dynamic-
# loading). If the fit faults on a CUDA symbol lookup, swap this to a cuda/13.x
# module — same caveat as the tiered a100 job.
# The shared gam_env.sh exports a booby-trapped RUSTFLAGS; never source it, and
# clear RUSTFLAGS so a login-profile export cannot leak in.
unset RUSTFLAGS

CHUNK_DIR=${CHUNK_DIR:-$R/creditscope_acts/activations/layer_30_residual_post}
MAX_ROWS=${MAX_ROWS:-120000}
K=${K:-2000}
TOP_K=${TOP_K:-32}
MAX_FIT_SECONDS=${MAX_FIT_SECONDS:-3600}
DRIVER=${DRIVER:-$(dirname "$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | awk -F= '/Command=/{print $2}')")/driver_1893_k2000_accept.py}
[ -f "$DRIVER" ] || DRIVER=$R/driver_1893_k2000_accept.py

test -f "$CHUNK_DIR/chunk_0000.npy" || { echo "FATAL: no chunk_0000.npy in $CHUNK_DIR"; exit 2; }
test -f "$DRIVER" || { echo "FATAL: driver not found: $DRIVER (set DRIVER=)"; exit 2; }
nvidia-smi -L || { echo "FATAL: no GPU visible to the job"; exit 98; }

export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export LD_LIBRARY_PATH="/common/software/install/manual/openblas/0.3.27_gcc11.3.0/lib:${LD_LIBRARY_PATH:-}"
# Scope RUST_LOG to the residency measurement module so ONLY the operand lines
# print (not the other ~517 log sites). Requires the wheel's logger backend (see
# HARD REQUIREMENT 2).
export RUST_LOG="${RUST_LOG:-gam_solve::gpu_kernels::arrow_schur=info}"

VENV=${VENV:-$R/venv_1893_a100_${SLURM_JOB_ID:-manual}}
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install --no-cache-dir "gamfit==$GAMFIT_VERSION" numpy scikit-learn torch
python -c "import gamfit; print('gamfit', getattr(gamfit,'__version__','?'))"

RUNLOG="$R/sae_1893_a100_measure_${SLURM_JOB_ID:-manual}.runlog"
echo "=== #1893 a100 measure K=$K  $(date)  node=$(hostname)  gamfit==$GAMFIT_VERSION  RUST_LOG=$RUST_LOG ==="
python -u "$DRIVER" \
  --data-mode chunkdir --chunk-dir "$CHUNK_DIR" --max-rows "$MAX_ROWS" \
  --K "$K" --top-k "$TOP_K" --d-atom 1 --atom-topology circle \
  --external-live --max-fit-seconds "$MAX_FIT_SECONDS" \
  --seed 0 2>&1 | tee "$RUNLOG"
rc=${PIPESTATUS[0]}

echo ""
echo "================= #1017/#2230 RESIDENCY MEASUREMENT SUMMARY ================="
LINES="$(grep -F '#1017/#2230' "$RUNLOG" || true)"
COUNT="$(printf '%s\n' "$LINES" | grep -c 'operand upload' || true)"
if [ "${COUNT:-0}" -eq 0 ]; then
  echo "0 operand lines captured. EITHER the matrix-free device PCG lane did not"
  echo "engage at K=$K (border k below the offload floor — rerun with a larger K or"
  echo "a wider-p shape, or use the synthetic #2230 driver), OR the gamfit wheel has"
  echo "NO log backend (HARD REQUIREMENT 2 — pyffi #[pymodule] must init pyo3-log/"
  echo "env_logger). Confirm which before concluding."
else
  # Sum the per-solve total=X.YMiB across all captured solves/ladder-trials.
  printf '%s\n' "$LINES" | awk '
    match($0, /total=([0-9.]+)MiB/, m) { sum += m[1]; n += 1 }
    /\[framed-dense\]/  { framed += 1 }
    /\[legacy-sparse\]/ { legacy += 1 }
    END {
      printf "operand-upload solves (= LM-ladder trials across the fit): %d\n", n
      printf "Σ per-solve operand transfer (removable by a resident frame): %.1f MiB\n", sum
      printf "sub-lane: framed-dense=%d  legacy-sparse=%d\n", framed, legacy
    }'
  echo "--- first captured line (per-category breakdown) ---"
  printf '%s\n' "$LINES" | grep 'operand upload' | head -1
fi
echo "==========================================================================="
echo "=== rc=$rc DONE $(date) ==="
[ "$rc" -eq 0 ] && rm -rf "$VENV"
exit $rc

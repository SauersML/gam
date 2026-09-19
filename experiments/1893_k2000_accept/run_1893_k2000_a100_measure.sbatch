#!/bin/bash
# A100 variant of the #1893 K=2000 acceptance harness, doubling as the #1017/#2230
# increment-1 residency MEASUREMENT. Same real creditscope Qwen3.5-35B L30 fit as
# run_1893_k2000_accept.sbatch, but on the GPU partition, and it opts the
# "gamfit" Python logger into DEBUG so the per-solve operand-upload lines (DeviceSaePcgData::operand_byte_report, landed 9b98a5413)
# print, and a final SUMMARY block sums them:
#   count(lines) = matrix-free PCG solves = LM-ladder trials across the fit
#   Σ total MiB  = the host→device operand transfer a base-resident frame removes
# It also confirms the sub-lane (framed-dense vs legacy-sparse — expected
# legacy-sparse at this wide-p shape; framed dense would be ~31 GiB, impossible).
#
# TWO REQUIREMENTS:
#   1. GAMFIT_VERSION must pin a wheel that forwards engine `log` records to the
#      "gamfit" Python logger (and contains operand_byte_report, >= 9b98a5413).
#   2. The "gamfit" logger is lowered to DEBUG BEFORE the fit. The engine's
#      records go to that logger and are silent by default; RUST_LOG is
#      intentionally IGNORED (explicit-API-not-env-var design). A tiny wrapper
#      attaches a stderr handler to the logger and runs the unmodified driver, so
#      NO RUST_LOG and no pyffi change is needed. A "0 operand lines" SUMMARY then
#      means the device matrix-free PCG lane did NOT offload at K (not a logging
#      problem).
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

CHUNK_DIR=${CHUNK_DIR:?set CHUNK_DIR to a directory filled by experiments/1026_close/fetch_creditscope_l30.sh}
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
# RUST_LOG is intentionally IGNORED by the wheel (requirement 2); the "gamfit"
# logger is lowered in-process below, and we grep the emitted lines rather than
# rely on an env-var filter. The #1017/#2230 lines are prefix-tagged so the grep
# is exact even amid the other debug-level fit-progress chatter.

VENV=${VENV:-$R/venv_1893_a100_${SLURM_JOB_ID:-manual}}
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install --no-cache-dir "gamfit==$GAMFIT_VERSION" numpy scikit-learn torch
python -c "import gamfit; print('gamfit', getattr(gamfit,'__version__','?'))"

RUNLOG="$R/sae_1893_a100_measure_${SLURM_JOB_ID:-manual}.runlog"
echo "=== #1893 a100 measure K=$K  $(date)  node=$(hostname)  gamfit==$GAMFIT_VERSION ==="

# Lower the "gamfit" logger to DEBUG in-process (RUST_LOG is ignored), then run
# the UNMODIFIED #1893 driver in the same interpreter so the level persists through
# the fit. The wrapper drops its own name from argv so the driver's argparse is intact.
WRAP="$R/measure_debuglog_${SLURM_JOB_ID:-manual}.py"
cat > "$WRAP" <<'PYEOF'
import logging, sys, runpy
_log = logging.getLogger("gamfit")
_log.addHandler(logging.StreamHandler(sys.stderr))
_log.setLevel(logging.DEBUG)
sys.argv = sys.argv[1:]            # argv[0] becomes the driver path
runpy.run_path(sys.argv[0], run_name="__main__")
PYEOF
python -u "$WRAP" "$DRIVER" \
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
  echo "0 operand lines captured. With the gamfit logger at DEBUG, this means the"
  echo "matrix-free device PCG lane did NOT engage at K=$K (border k below the offload"
  echo "floor) — rerun with a larger K / wider-p shape, or use the host-pure"
  echo "residency_operand_measure harness which assembles the wide border directly."
  echo "(If no gamfit records appear at all, confirm the wheel forwards engine logs.)"
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

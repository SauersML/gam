#!/bin/bash -l
#SBATCH --job-name=saebench_1942
#SBATCH --partition=a100-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/saebench_1942_%j.out
#
# #1942 full SAEBench pass: the public capability suite (absorption / SCR /
# unlearning / sparse-probing) delegated to the `sae_bench` package with our
# dictionary spliced into the host model, PLUS the manifold-native arms
# (dose-response calibration, chart-interp, frozen-dictionary audit) scored by
# the audited gamfit/Rust scorers. One driver, one report json.
#
# Submit (only GAMFIT_VERSION is required; everything else has a default):
#
#   GAMFIT_VERSION=0.1.252 sbatch experiments/audit_sae/saebench_full_pass.sbatch
#
# Override any path via the env, e.g.
#   SAEBENCH_SAE=$R/gemma_sae/gemma-2-2b_l12.safetensors \
#   SAEBENCH_MODEL=gemma-2-2b SAEBENCH_ARCH=jumprelu SAEBENCH_HOOK_LAYER=12 \
#   DOSE_LEDGER=$R/msae_l17/dose_calibration_real.json \
#   GAMFIT_VERSION=0.1.252 sbatch experiments/audit_sae/saebench_full_pass.sbatch
#
# The public suite is SKIPPED (never fabricated) unless BOTH SAEBENCH_SAE and
# SAEBENCH_MODEL resolve to a checkpoint whose d_in matches the host model's
# d_model AND the sae_bench datasets/model are present; the manifold-native arms
# run whenever their ledgers are present.

set -uo pipefail
R=${R:-/projects/standard/hsiehph/sauer354}

# CRITICAL: the MSI gam_env booby-traps RUSTFLAGS with an openblas saboteur that
# breaks any in-job maturin/cargo; this job installs a prebuilt wheel and never
# compiles, but unset it defensively so nothing downstream inherits it.
unset RUSTFLAGS

export GAM_MSI_DATA=$R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export LD_LIBRARY_PATH="/common/software/install/manual/openblas/0.3.27_gcc11.3.0/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=${HF_HOME:-$R/hf_cache}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

GAMFIT_VERSION=${GAMFIT_VERSION:?set GAMFIT_VERSION (e.g. 0.1.252) — the gamfit wheel to pin}
DRIVER=${DRIVER:-$R/saebench_eval.py}
[ -f "$DRIVER" ] || DRIVER=$(dirname "$0")/saebench_eval.py
test -f "$DRIVER" || { echo "FATAL: driver not found: $DRIVER (set DRIVER=)"; exit 2; }

OUT_DIR=${OUT_DIR:-$R/scratch/saebench_1942_${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT_DIR"

# --- Manifold-native arm inputs (optional; each arm is added only if present) ---
DOSE_LEDGER=${DOSE_LEDGER:-$R/msae_l17/dose_calibration_real.json}
CHART_FIT=${CHART_FIT:-$R/msae_l17/chart_fit_month_l17.json}
DECODER=${DECODER:-$R/msae_l17/decoder.npy}
ACTIVATIONS=${ACTIVATIONS:-$R/msae_l17/activations.npy}

# --- Public SAEBench suite inputs (optional; skipped cleanly if unset/missing) ---
SAEBENCH_SAE=${SAEBENCH_SAE:-}
SAEBENCH_MODEL=${SAEBENCH_MODEL:-}
SAEBENCH_EVALS=${SAEBENCH_EVALS:-absorption,scr,unlearning,sparse_probing}
SAEBENCH_ARCH=${SAEBENCH_ARCH:-topk}
SAEBENCH_HOOK_LAYER=${SAEBENCH_HOOK_LAYER:-12}
SAEBENCH_DEVICE=${SAEBENCH_DEVICE:-cuda}
SAEBENCH_DTYPE=${SAEBENCH_DTYPE:-bfloat16}

# Use the EXISTING working venv — do NOT build a fresh venv from the node's
# system python: that is Python 3.6 on the MSI compute nodes, for which no gamfit
# (or modern torch) wheel exists ("No matching distribution for gamfit"). saevenv
# is the proven frontier venv (modern python + torch + matplotlib) the #1026 jobs
# use; we only pin the gamfit wheel on top of it.
VENV=${VENV:-$R/saevenv}
test -f "$VENV/bin/activate" || { echo "FATAL: no venv at $VENV (set VENV=)"; exit 2; }
source "$VENV/bin/activate"
echo "=== python $(python -V 2>&1)  ($(command -v python))  node=$(hostname)  $(date) ==="
# Guard against a stale/3.6 interpreter slipping through (belt-and-suspenders with
# the gamfit-availability preflight below).
python -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' || {
  echo "FATAL: $VENV python $(python -V 2>&1) is too old for the gamfit wheel (need >= 3.10)"; exit 2; }
python -m pip install --no-cache-dir --upgrade "gamfit==$GAMFIT_VERSION" >/dev/null
python -m pip install --no-cache-dir numpy scikit-learn safetensors >/dev/null
# sae_bench is only needed for the public suite; saevenv already carries torch, so
# do NOT reinstall/upgrade torch here (would risk clobbering the working GPU build).
python -m pip install --no-cache-dir sae_bench transformers accelerate || \
  echo "WARN: sae_bench install failed — public suite will report skipped"

echo "=== gamfit $(python -c 'import gamfit;print(getattr(gamfit,"__version__","?"))')  ==="

# Deploy-skew preflight: fail the job at submission-time BEFORE any model load if
# the installed wheel lacks a #1942 metric binding the driver calls.
python - <<'PYEOF' || { echo "FATAL: gamfit deploy-skew — aborting before the GPU run"; exit 3; }
import os, sys, gamfit
required = ("chart_interp_score", "dose_response_calibration", "audit_sae")
missing = [n for n in required if not hasattr(gamfit, n)]
if missing:
    sys.exit(f"gamfit {getattr(gamfit,'__version__','?')} at {os.path.dirname(gamfit.__file__)} "
             f"missing {missing}; pin a newer GAMFIT_VERSION with the #1942 SAEBench scorers")
print("preflight OK: gamfit exposes", required)
PYEOF

# Assemble the driver invocation from whatever inputs actually exist.
ARGS=(--out "$OUT_DIR/saebench_1942_report.json")
[ -f "$DOSE_LEDGER" ]  && ARGS+=(--dose-ledger "$DOSE_LEDGER")
[ -f "$CHART_FIT" ]    && ARGS+=(--chart-fit "$CHART_FIT")
[ -f "$DECODER" ] && [ -f "$ACTIVATIONS" ] && ARGS+=(--decoder "$DECODER" --activations "$ACTIVATIONS")
if [ -n "$SAEBENCH_SAE" ] && [ -n "$SAEBENCH_MODEL" ] && [ -f "$SAEBENCH_SAE" ]; then
  ARGS+=(--saebench-sae "$SAEBENCH_SAE" --saebench-model "$SAEBENCH_MODEL"
         --saebench-evals "$SAEBENCH_EVALS" --saebench-arch "$SAEBENCH_ARCH"
         --saebench-hook-layer "$SAEBENCH_HOOK_LAYER" --saebench-device "$SAEBENCH_DEVICE"
         --saebench-dtype "$SAEBENCH_DTYPE" --saebench-output-dir "$OUT_DIR/saebench_out")
else
  echo "NOTE: public SAEBench suite inputs incomplete — that arm will report skipped"
fi

echo "=== driver args: ${ARGS[*]} ==="
stdbuf -oL -eL python -u "$DRIVER" "${ARGS[@]}"
rc=$?
echo "=== rc=$rc DONE $(date) ==="
echo "=== report: $OUT_DIR/saebench_1942_report.json ==="
[ -f "$OUT_DIR/saebench_1942_report.json" ] && cat "$OUT_DIR/saebench_1942_report.json"
# NB: $VENV is the SHARED saevenv — never rm it.
exit $rc

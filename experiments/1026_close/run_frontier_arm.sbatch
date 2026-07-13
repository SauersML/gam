#!/bin/bash -l
# #1026 EV-vs-K FRONTIER — ONE (arm, K, seed) point per job.
#
# This is the parameterized single-point job that frontier_matrix.sh fans out
# over the K sweep and both seeds. Every job is independent: it derives its own
# train/test split from --seed, runs exactly one driver arm, and APPENDS one
# result row to the shared frontier jsonl. A failure of any single point leaves
# every other point untouched (the whole reason we do not use one monolithic
# job for the sweep).
#
# The GPU torch arms (external_topk, hybrid) need CUDA; the driver auto-detects
# and falls back to CPU, but the r4 wall-times below assume a B200/A100. pca_bar
# is CPU-only and K-independent (run ONCE, not per-K).
#
# Submit ONE point (frontier_matrix.sh does this for you across the grid):
#
#   ARM=hybrid K=8192 SEED=0 CURVED_ATOMS=128 CURVED_K=2 \
#     sbatch experiments/1026_close/run_frontier_arm.sbatch
#
# Required env: ARM, K, SEED. Everything else has a default. OUT is the shared
# append target so all points land in one file for plot_frontier.py.
#
#SBATCH --job-name=sae1026_frontier
#SBATCH --partition=a100-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/sae1026_frontier_%j.out
set -uo pipefail

R=${R:-/projects/standard/hsiehph/sauer354}

# CRITICAL: the MSI gam_env booby-traps RUSTFLAGS with an openblas saboteur that
# breaks any in-job maturin/cargo. This job runs a prebuilt wheel and never
# compiles, but unset it defensively so nothing downstream inherits it.
unset RUSTFLAGS

# --- Point coordinates (required) ------------------------------------------- #
ARM=${ARM:?set ARM (external_topk | hybrid | pca_bar)}
K=${K:?set K (dictionary size for this frontier point)}
SEED=${SEED:?set SEED (0 or 1)}

# --- Matched-budget knobs (defaults reproduce the committed 32k winner) ------ #
# top_k is the per-token active-scalar budget shared by BOTH arms. hybrid splits
# it as k_lin + curved_k*(1+d_atom) == top_k, so the per-token cost is identical
# to external_topk at every K (the dominance argument). d_atom=2, curved_k=2,
# top_k=32 => k_lin=26, matching the r4 a512 winner.
TOP_K=${TOP_K:-32}
CURVED_K=${CURVED_K:-2}
D_ATOM=${D_ATOM:-2}
# CURVED_ATOMS is the curved-tier dictionary size. It MUST scale with K (see the
# audit note in frontier_matrix.sh): a fixed 256/512 would give the hybrid a
# constant curved capacity that is a huge relative boost at small K, biasing the
# frontier. frontier_matrix.sh passes CURVED_ATOMS = max(64, K/64) so the
# curved:flat dictionary-capacity ratio is held constant across the sweep. At
# K=32768 this is 512 (== the committed r4-a512 winner), so the 32k frontier
# point reproduces the closed bar exactly.
CURVED_ATOMS=${CURVED_ATOMS:-$(( K/64 > 64 ? K/64 : 64 ))}
STEPS=${STEPS:-8000}            # flat/external TopK training steps
CURVED_STEPS=${CURVED_STEPS:-6000}
LR=${LR:-1e-3}
BATCH_SIZE=${BATCH_SIZE:-2048}  # r4 used bs2048 (the strong external bar)
MAX_ROWS=${MAX_ROWS:-120000}
BITS_MAX_ROWS=${BITS_MAX_ROWS:-8192}
# Declared dictionary amortization horizon N (#2283): separate from the bits
# estimation subsample. Defaults to the corpus token count so it is identical
# across every paired arm; NEVER the estimation subsample size.
AMORTIZATION_HORIZON=${AMORTIZATION_HORIZON:-$MAX_ROWS}
ATOM_MANIFOLD=${ATOM_MANIFOLD:-product}
ATOM_BASIS=${ATOM_BASIS:-fourier}
TAG=${TAG:-frontier-K${K}-s${SEED}}

# MSI-resident real-activation set: creditscope Qwen3.5-35B-A3B layer-30
# residual, the exact set the #1026 close bar names. p is read from the shard.
CHUNK_DIR=${CHUNK_DIR:-$R/creditscope_acts/activations/layer_30_residual_post}
OUT=${OUT:-$R/1026_frontier/results_1026.jsonl}
VENV=${VENV:-$R/saevenv}
DRIVER=${DRIVER:-$(dirname "$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | awk -F= '/Command=/{print $2}')")/driver_1026_arms.py}
[ -f "$DRIVER" ] || DRIVER=$R/driver_1026_arms.py

# Fail LOUD on missing data / driver / venv — never waste the GPU slot on a
# silent-fallback dataset.
test -f "$CHUNK_DIR/chunk_0000.npy" || { echo "FATAL: no chunk_0000.npy in $CHUNK_DIR"; exit 2; }
test -f "$DRIVER" || { echo "FATAL: driver not found: $DRIVER (set DRIVER=)"; exit 2; }
test -x "$VENV/bin/python" || { echo "FATAL: no venv python at $VENV (set VENV=)"; exit 2; }
mkdir -p "$(dirname "$OUT")"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export LD_LIBRARY_PATH="/common/software/install/manual/openblas/0.3.27_gcc11.3.0/lib:${LD_LIBRARY_PATH:-}"

PY=$VENV/bin/python
echo "=== #1026 frontier  arm=$ARM K=$K seed=$SEED  $(date)  node=$(hostname) ==="
$PY -c "import gamfit; print('gamfit', getattr(gamfit,'__version__','?'))"

# Only the knobs an arm actually reads are passed; the driver ignores the rest.
COMMON=(--arm "$ARM" --chunk-dir "$CHUNK_DIR" --max-rows "$MAX_ROWS"
        --seed "$SEED" --K "$K" --top-k "$TOP_K" --tag "$TAG" --out "$OUT")

case "$ARM" in
  external_topk)
    $PY -u "$DRIVER" "${COMMON[@]}" \
      --steps "$STEPS" --lr "$LR" --batch-size "$BATCH_SIZE" --cosine-lr \
      --bits --bits-max-rows "$BITS_MAX_ROWS" \
      --amortization-horizon "$AMORTIZATION_HORIZON"
    ;;
  hybrid)
    $PY -u "$DRIVER" "${COMMON[@]}" \
      --steps "$STEPS" --curved-steps "$CURVED_STEPS" --lr "$LR" \
      --batch-size "$BATCH_SIZE" --d-atom "$D_ATOM" \
      --curved-atoms "$CURVED_ATOMS" --curved-k "$CURVED_K" \
      --atom-manifold "$ATOM_MANIFOLD" --atom-basis "$ATOM_BASIS" \
      --bits --bits-max-rows "$BITS_MAX_ROWS" \
      --amortization-horizon "$AMORTIZATION_HORIZON"
    ;;
  pca_bar)
    # K-independent linear-optimum yardstick — run ONCE (frontier_matrix.sh
    # submits a single pca_bar job, not one per K).
    $PY -u "$DRIVER" "${COMMON[@]}"
    ;;
  *)
    echo "FATAL: unsupported ARM=$ARM for the frontier (use external_topk|hybrid|pca_bar)"; exit 2
    ;;
esac
rc=$?
echo "=== rc=$rc DONE $(date)  (row appended to $OUT) ==="
exit $rc

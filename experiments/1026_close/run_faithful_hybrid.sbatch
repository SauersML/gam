#!/bin/bash -l
# gam#2233 THEOREM-FAITHFUL hybrid-vs-flat Eq-4 bits at K=32768 (creditscope L30).
#
# WHY THIS JOB EXISTS
# -------------------
# The committed `hybrid` arm STACKS `curved_atoms` curved atoms on the FULL
# K=32768 flat dictionary, so dict_params(hybrid) = K*P + curved_atoms*b*P
# STRICTLY EXCEEDS external_topk's K*P. On the Eq-4 scoreboard the dictionary
# term (0.5*params/N*log2 N) is ~95% of the bits at K=32768, so that config
# carries a fixed surcharge and CANNOT win no matter how good its support/code/
# residual savings are (see gam#2233; closed-form check:
# experiments/1026_close/crossover_theorem_check.py -> STACKED net -2829 bits).
#
# The crossover theorem's claim is that curvature buys FEWER ATOMS: one curved
# atom REPLACES the flat atoms of a factor. A fair test therefore requires
#     dict_params(hybrid) = k_flat*P + curved_atoms*b*P  <=  K*P
# i.e. REDUCE the flat ATOM COUNT k_flat (NOT just the per-token active budget
# k_lin). Setting k_flat = K - curved_atoms*b makes the two dictionaries carry
# IDENTICAL parameters, so any support/code/residual saving is a NET MDL win
# (closed-form check: FAITHFUL net +83 bits on support alone for the circle).
#
# The driver self-certifies via `bits_dict_params_faithful` in the result row:
# it MUST be true. If it prints false, raise B below and resubmit.
#
# SUBMIT (single job; runs external_topk then the faithful hybrid on the SAME
# seed-derived split, appends both rows):
#     sbatch experiments/1026_close/run_faithful_hybrid.sbatch
# Override the curved decoder-block width if the built model differs:
#     B=7 sbatch experiments/1026_close/run_faithful_hybrid.sbatch
#
#SBATCH --job-name=sae2233_faithful
#SBATCH --partition=a100-4,a100-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/sae2233_faithful_%j.out
set -euo pipefail

R=${R:-/projects/standard/hsiehph/sauer354}
unset RUSTFLAGS   # MSI gam_env booby-traps RUSTFLAGS; this job runs a prebuilt wheel.

# --- Contest coordinates ----------------------------------------------------- #
K=${K:-32768}            # external flat atom count (the bar) and dict-param budget
P_HINT=${P_HINT:-2048}   # ambient dim (only used to print the budget; driver reads real p)
SEED=${SEED:-0}
TOP_K=${TOP_K:-32}
CURVED_ATOMS=${CURVED_ATOMS:-256}
CURVED_K=${CURVED_K:-2}
D_ATOM=${D_ATOM:-2}
# B = curved-atom decoder-block width (basis functions per curved atom). For a
# Fourier circle of H harmonics B = 2H+1; the product/d_atom torus tier the
# driver builds may use a different B. The row's bits_dict_params_faithful is
# ground truth — if false, raise B and resubmit. 7 is a safe default that
# reserves headroom for the torus basis.
B=${B:-7}
# The theorem-faithful flat atom count: k_flat = K - curved_atoms*B, so
# k_flat*P + curved_atoms*B*P == K*P (dictionaries carry identical params).
K_FLAT=${K_FLAT:-$(( K - CURVED_ATOMS * B ))}
if (( K_FLAT < TOP_K )); then
  echo "FATAL: K_FLAT=$K_FLAT < top_k=$TOP_K (curved tier too wide for this K); lower CURVED_ATOMS/B"; exit 2
fi

STEPS=${STEPS:-8000}
CURVED_STEPS=${CURVED_STEPS:-6000}   # the torch ManifoldSAE E-step is the known CPU wall
LR=${LR:-1e-3}
BATCH_SIZE=${BATCH_SIZE:-2048}
MAX_ROWS=${MAX_ROWS:-120000}
BITS_MAX_ROWS=${BITS_MAX_ROWS:-8192}
ATOM_MANIFOLD=${ATOM_MANIFOLD:-product}
ATOM_BASIS=${ATOM_BASIS:-fourier}
TAG=${TAG:-faithful-K${K}-kflat${K_FLAT}-s${SEED}}

CHUNK_DIR=${CHUNK_DIR:-$R/creditscope_acts/activations/layer_30_residual_post}
OUT=${OUT:-$R/2233_faithful/eq4_faithful.jsonl}
VENV=${VENV:-$R/saevenv}
DRIVER=${DRIVER:-$(dirname "$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | awk -F= '/Command=/{print $2}')")/driver_1026_arms.py}
[ -f "$DRIVER" ] || DRIVER=$R/driver_1026_arms.py
DRIVER_DIR=$(cd "$(dirname "$DRIVER")" && pwd)

test -f "$CHUNK_DIR/chunk_0000.npy" || { echo "FATAL: no chunk_0000.npy in $CHUNK_DIR"; exit 2; }
test -f "$DRIVER" || { echo "FATAL: driver not found: $DRIVER (set DRIVER=)"; exit 2; }
test -f "$DRIVER_DIR/bits_eq4.py" || { echo "FATAL: missing $DRIVER_DIR/bits_eq4.py beside driver"; exit 2; }
test -f "$DRIVER_DIR/arm_featurizers.py" || { echo "FATAL: missing $DRIVER_DIR/arm_featurizers.py beside driver"; exit 2; }
test -x "$VENV/bin/python" || { echo "FATAL: no venv python at $VENV (set VENV=)"; exit 2; }
mkdir -p "$(dirname "$OUT")"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export LD_LIBRARY_PATH="/common/software/install/manual/openblas/0.3.27_gcc11.3.0/lib:${LD_LIBRARY_PATH:-}"

PY=$VENV/bin/python
echo "=== gam#2233 faithful  K=$K k_flat=$K_FLAT curved_atoms=$CURVED_ATOMS B=$B  $(date)  node=$(hostname) ==="
echo "    dict-param budget: external K*P = $K*$P_HINT ; hybrid k_flat*P + curved*B*P = ($K_FLAT + $CURVED_ATOMS*$B)*P = $(( K_FLAT + CURVED_ATOMS*B ))*P"
$PY -c "import gamfit; print('gamfit', getattr(gamfit,'__version__','?'))"
(cd "$DRIVER_DIR" && $PY -c "import arm_featurizers, bits_eq4; print('Eq-4 modules: ready')")

COMMON=(--chunk-dir "$CHUNK_DIR" --max-rows "$MAX_ROWS" --seed "$SEED" --K "$K"
        --top-k "$TOP_K" --out "$OUT" --bits --bits-max-rows "$BITS_MAX_ROWS")

# 1) the flat bar (unchanged reference; dict_params = K*P).
$PY -u "$DRIVER" --arm external_topk "${COMMON[@]}" \
  --steps "$STEPS" --lr "$LR" --batch-size "$BATCH_SIZE" --cosine-lr \
  --tag "${TAG}-external"

# 2) the theorem-faithful hybrid: curved atoms REPLACE flat ones (--k-flat).
$PY -u "$DRIVER" --arm hybrid "${COMMON[@]}" \
  --k-flat "$K_FLAT" \
  --steps "$STEPS" --curved-steps "$CURVED_STEPS" --lr "$LR" \
  --batch-size "$BATCH_SIZE" --d-atom "$D_ATOM" \
  --curved-atoms "$CURVED_ATOMS" --curved-k "$CURVED_K" \
  --atom-manifold "$ATOM_MANIFOLD" --atom-basis "$ATOM_BASIS" \
  --tag "${TAG}-hybrid"
echo "=== rc=0 DONE $(date) — CHECK bits_dict_params_faithful==true in $OUT ==="
echo "    if false: the built curved tier is wider than B=$B; rerun with B=<real width>."

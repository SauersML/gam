#!/bin/bash -l
#SBATCH --job-name=eq4_2283_flat
#SBATCH --partition=interactive-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
set -euo pipefail

# The checkpoint binds the pair identity, which records --bits-max-rows and
# --amortization-horizon, so both must equal the curved-resume phase's values.
if [[ $# -ne 8 ]]; then
  echo "usage: sbatch --output=LOG $0 DRIVER VENV CHUNK_DIR RUN_ID CODE_REVISION WHEEL_SHA256 OUT_JSONL FLAT_CHECKPOINT" >&2
  exit 2
fi
DRIVER=$1
VENV=$2
CHUNK_DIR=$3
RUN_ID=$4
CODE_REVISION=$5
WHEEL_SHA256=$6
OUT=$7
FLAT_CHECKPOINT=$8

export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
"$VENV/bin/python" -u "$DRIVER" \
  --arm hybrid_rust --hybrid-phase flat-checkpoint --flat-checkpoint "$FLAT_CHECKPOINT" \
  --chunk-dir "$CHUNK_DIR" \
  --run-id "$RUN_ID" --code-revision "$CODE_REVISION" --wheel-sha256 "$WHEEL_SHA256" \
  --max-rows 120000 --test-frac 0.2 --seed 0 --K 32768 --top-k 32 \
  --k-flat 32672 --d-atom 1 --atom-topology circle --curved-atoms 32 --curved-k 2 \
  --sparse-minibatch 8192 --sparse-score-mode required --max-epochs 200 \
  --bits-max-rows 24000 --amortization-horizon 120000 --out "$OUT"

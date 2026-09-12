#!/bin/bash -l
#SBATCH --job-name=eq4_2283_external
#SBATCH --partition=interactive-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
set -euo pipefail

# CHUNK_DIR is filled by fetch_creditscope_l30.sh. --bits-max-rows 24000 is the whole
# held-out split (120000 x 0.2); scoring 8192 of those rows moves the bits by 17x the margin.
if [[ $# -ne 7 ]]; then
  echo "usage: sbatch --output=LOG $0 DRIVER VENV CHUNK_DIR RUN_ID CODE_REVISION WHEEL_SHA256 OUT_JSONL" >&2
  exit 2
fi
DRIVER=$1
VENV=$2
CHUNK_DIR=$3
RUN_ID=$4
CODE_REVISION=$5
WHEEL_SHA256=$6
OUT=$7

export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
"$VENV/bin/python" -u "$DRIVER" \
  --arm external_topk \
  --chunk-dir "$CHUNK_DIR" \
  --run-id "$RUN_ID" --code-revision "$CODE_REVISION" --wheel-sha256 "$WHEEL_SHA256" \
  --max-rows 120000 --test-frac 0.2 --seed 0 --K 32768 --top-k 32 \
  --steps 8000 --lr 0.001 --batch-size 2048 --cosine-lr \
  --bits --bits-max-rows 24000 --amortization-horizon 120000 --out "$OUT"

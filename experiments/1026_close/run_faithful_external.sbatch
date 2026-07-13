#!/bin/bash -l
#SBATCH --job-name=eq4_2283_external
#SBATCH --partition=interactive-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/2233_faithful/external_%j.log
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: sbatch $0 DRIVER VENV RUN_ID CODE_REVISION WHEEL_SHA256 OUT_JSONL" >&2
  exit 2
fi
DRIVER=$1
VENV=$2
RUN_ID=$3
CODE_REVISION=$4
WHEEL_SHA256=$5
OUT=$6

export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
"$VENV/bin/python" -u "$DRIVER" \
  --arm external_topk \
  --chunk-dir /projects/standard/hsiehph/sauer354/creditscope_acts/activations/layer_30_residual_post \
  --run-id "$RUN_ID" --code-revision "$CODE_REVISION" --wheel-sha256 "$WHEEL_SHA256" \
  --max-rows 120000 --test-frac 0.2 --seed 0 --K 32768 --top-k 32 \
  --steps 8000 --lr 0.001 --batch-size 2048 --cosine-lr \
  --bits --bits-max-rows 8192 --out "$OUT"

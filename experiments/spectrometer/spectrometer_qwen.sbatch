#!/usr/bin/env bash
#SBATCH --job-name=spectrometer
#SBATCH --partition=msismall
#SBATCH --cpus-per-task=32
#SBATCH --mem=110g
#SBATCH --time=08:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/spectrometer_qwen/%x_%j.out
# Args: $1=SPEC_DATA (raw fp32 [n,d] residual npy)  $2=SPEC_OUT  $3=SPEC_TAG  $4=SPEC_N
set -euo pipefail
ROOT=/projects/standard/hsiehph/sauer354
export SPEC_DATA="$1"
export SPEC_OUT="$2"
export SPEC_TAG="$3"
export SPEC_N="${4:-150000}"
export SPEC_SEED=0
mkdir -p "$SPEC_OUT"
cp "$ROOT/spectrometer_qwen/spectrometer.py" "$ROOT/spectrometer_qwen/plot_spectrum.py" "$SPEC_OUT/" 2>/dev/null || true
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 VECLIB_MAXIMUM_THREADS=32 MKL_NUM_THREADS=32
source "$ROOT/saevenv/bin/activate"
cd "$SPEC_OUT"
python -u spectrometer.py
python -u plot_spectrum.py   # matplotlib 3.11 present on MSI
echo "SBATCH_DONE"

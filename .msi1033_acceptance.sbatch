#!/bin/bash
#SBATCH --job-name=m1033acc
#SBATCH --partition=msismall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/gam/.msi1033_acceptance.log

set -euo pipefail
set -x

source /projects/standard/hsiehph/sauer354/gam_env.sh
cd /projects/standard/hsiehph/sauer354/gam
git fetch origin main -q && git checkout origin/main 2>&1 | tail -1
git log --oneline -1

TARGET_DIR=/projects/standard/hsiehph/sauer354/scratch/gam1033_acceptance_target
mkdir -p "$TARGET_DIR"

echo "===== #1033 build once: lib + perf_scale release test binaries ====="
cargo test --target-dir "$TARGET_DIR" --release --no-run --lib --test perf_scale 2>&1 | tail -80

echo "===== locate direct test binaries ====="
LIB_BIN=$(find "$TARGET_DIR/release/deps" -maxdepth 1 -type f -perm -111 -name 'gam-*' ! -name '*.d' -print | xargs ls -t | head -1)
PERF_BIN=$(find "$TARGET_DIR/release/deps" -maxdepth 1 -type f -perm -111 -name 'perf_scale-*' ! -name '*.d' -print | xargs ls -t | head -1)
echo "LIB_BIN=$LIB_BIN"
echo "PERF_BIN=$PERF_BIN"

echo "===== #1033 structural k-space accessor gate ====="
/usr/bin/time -v "$LIB_BIN" \
  psi_gram_tensor_trial_accessors_touch_no_data_rows \
  --nocapture --test-threads=1

echo "===== #1033 n=320k kappa outer-loop acceptance sweep ====="
/usr/bin/time -v "$PERF_BIN" \
  --exact --nocapture --test-threads=1 \
  perf_kappa_loop_n_scaling::kappa_outer_loop_is_n_independent

echo "===== #1033 acceptance complete ====="

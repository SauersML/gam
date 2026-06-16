#!/bin/bash
#SBATCH --job-name=mt1028v
#SBATCH --partition=amdsmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120g
#SBATCH --time=01:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/mt_final.log

set -uo pipefail
cd /projects/standard/hsiehph/sauer354/gam
source /projects/standard/hsiehph/sauer354/gam_env.sh
export CARGO_BUILD_JOBS=16
echo "=== node $(hostname) HEAD $(git rev-parse HEAD) ==="
echo "=== build lib test binary (--no-run) ==="
cargo test --lib --no-run 2>&1 | tail -20
BIN=$(ls -t target/debug/deps/gam-* | grep -vE '\.d$' | head -1)
echo "=== run prebuilt binary directly: $BIN ==="
"$BIN" murphy_topel_correction_matches_two_stage_sampling_variance --nocapture --test-threads=1 2>&1
echo "=== VERDICT_RC=$? ==="

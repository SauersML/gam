#!/bin/bash
# strike_liberr.sbatch — FAST capture of the exact rustc error/warning list for
# origin/main's lib (and lib-test) under the repo's -D warnings, so precise fixes
# can be routed. Builds in the warm fast-target; reports file:line for every hit.
#SBATCH -A hsiehph
#SBATCH -p msismall
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 0:40:00
#SBATCH -J strikeliberr
#SBATCH -o /projects/standard/hsiehph/sauer354/calib_logs/strike_liberr_%j.out
#SBATCH -e /projects/standard/hsiehph/sauer354/calib_logs/strike_liberr_%j.out
set -uo pipefail
PROJ=/projects/standard/hsiehph/sauer354
source ${PROJ}/gam_env.sh 2>/dev/null || true
export CARGO_TARGET_DIR=${PROJ}/gam-target-fast
export CARGO_INCREMENTAL=1
cd ${PROJ}/gam
git fetch origin -q 2>/dev/null
git reset --hard origin/main >/dev/null 2>&1
echo "=== lib-error capture @ $(git rev-parse --short HEAD) $(date -u +%FT%TZ) ==="
CARGO=${PROJ}/.cargo/bin/cargo

echo "=== (A) cargo build --release --lib ==="
$CARGO build --release --lib 2>&1 | tee /tmp/liberr_lib_${SLURM_JOB_ID}.txt | grep -nE "error\[|error:|unused variable|should have a|non_snake_case|warning:" | head -60
echo "LIB build rc=${PIPESTATUS[0]}"

echo "=== (B) cargo build --release --tests (lib + integration test targets) ==="
$CARGO build --release --tests 2>&1 | tee /tmp/liberr_tests_${SLURM_JOB_ID}.txt | grep -nE "error\[|error:|unused variable|should have a|non_snake_case" | head -60
echo "TESTS build rc=${PIPESTATUS[0]}"

echo "=== (C) every error: / warning: line with --> file:line (lib) ==="
grep -E "^(error|warning)" -A2 /tmp/liberr_lib_${SLURM_JOB_ID}.txt | grep -E "^(error|warning)|-->" | head -80
echo "=== (D) every error: / warning: line with --> file:line (tests) ==="
grep -E "^(error|warning)" -A2 /tmp/liberr_tests_${SLURM_JOB_ID}.txt | grep -E "^(error|warning)|-->" | head -80
echo "=== EXIT=0 ==="

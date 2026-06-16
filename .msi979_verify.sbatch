#!/bin/bash
#SBATCH --job-name=m979ver
#SBATCH --partition=msismall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/gam/.msi979_verify.log

set -x
source /projects/standard/hsiehph/sauer354/gam_env.sh
cd /projects/standard/hsiehph/sauer354/gam
git fetch origin main -q && git checkout origin/main 2>&1 | tail -1
git log --oneline -1

# Build in the TEST profile (NOT --release): debug_assertions is ON, so the
# debug-gated cubic_cell parity-probe consumer compiles and the dead-code that
# breaks the release build is absent. CARGO_PROFILE_TEST_OPT_LEVEL=2 keeps the
# numeric inner loops fast. Cycle counts (the decisive #979 measurement) are
# profile-independent.
export CARGO_PROFILE_TEST_OPT_LEVEL=2
export CARGO_TARGET_DIR=/projects/standard/hsiehph/sauer354/gam/target_opt2

echo "===== BUILD (no-run, test profile) perf_scale + survival ====="
cargo test --no-run --test perf_scale --test survival 2>&1 | tail -40

echo "===== locate prebuilt deps binaries ====="
PERF_BIN=$(ls -t target_opt2/debug/deps/perf_scale-* | grep -v '\.d$' | head -1)
SURV_BIN=$(ls -t target_opt2/debug/deps/survival-* | grep -v '\.d$' | head -1)
echo "PERF_BIN=$PERF_BIN"
echo "SURV_BIN=$SURV_BIN"

echo "===== RUN binary guard: large_scale_margslope_repro_completes_under_ci_budget ====="
/usr/bin/time -v "$PERF_BIN" --exact --nocapture large_scale_margslope_repro::large_scale_margslope_repro_completes_under_ci_budget 2>&1 | grep -iE "PIRLS/joint-Newton terminal|outer_converged|test result|panicked|wall clock|Elapsed|budget|took|FAILED|ok$|cycles=|LARGE_SCALE-MARGSLOPE-REPRO" | tail -40

echo "===== RUN survival repro: survival_marginal_slope_large_scale_repro_vm_exact_engages_and_converges ====="
/usr/bin/time -v "$SURV_BIN" --exact --nocapture survival_marginal_slope_large_scale_repro::survival_marginal_slope_large_scale_repro_vm_exact_engages_and_converges 2>&1 | grep -iE "PIRLS/joint-Newton terminal|outer_converged|test result|panicked|wall clock|Elapsed|budget|took|FAILED|ok$|cycles=|compiled-map" | tail -50

echo "===== DONE ====="

#!/bin/bash
# strike_verify_full.sbatch — ONE comprehensive MSI verify of the strike-team
# owed/quality/in-crate test set against origin/main. INFORMATION, not a gate:
# --no-fail-fast runs everything, and the tail prints a per-test PASS/FAIL table.
#
# Three lanes, all on the warm shared fast-target:
#   (1) integration test binaries (binary() regex union over the owed/issue/
#       bug_hunt/diag files that exist for the listed issues; not-yet-landed
#       issues simply don't match — no error),
#   (2) the `quality` binary filtered to the #1471 validation-baseline module
#       (test() path predicate) — these are the mgcv-comparison quality arms,
#   (3) the in-crate lib unit tests for #1026 / #299 preconditioner study
#       (linear_parity_anchor_1026 + arrow_precond ladder), via cargo test --lib.
#SBATCH -A hsiehph
#SBATCH -p msismall
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 4:00:00
#SBATCH -J strikefull
#SBATCH -o /projects/standard/hsiehph/sauer354/calib_logs/strike_full_%j.out
#SBATCH -e /projects/standard/hsiehph/sauer354/calib_logs/strike_full_%j.out
set -uo pipefail
PROJ=/projects/standard/hsiehph/sauer354
source ${PROJ}/gam_env.sh 2>/dev/null || true
module load r-rich/4.4.2_msi1.2 2>/dev/null || true
source ${PROJ}/refpy/bin/activate 2>/dev/null || true

# Warm shared fast target (32c node, mold linker). cargo's exclusive per-target
# lock is fine here: this is the single comprehensive run, not a concurrent loop.
export CARGO_TARGET_DIR=${PROJ}/gam-target-fast
export CARGO_PROFILE_DEV_DEBUG=0
export CARGO_PROFILE_TEST_DEBUG=0
export CARGO_PROFILE_DEV_SPLIT_DEBUGINFO=off
export CARGO_INCREMENTAL=1

WORK=${SLURM_TMPDIR:-/tmp}/strike_full_${SLURM_JOB_ID}
rm -rf "$WORK"
git clone --depth 1 --branch main https://github.com/SauersML/gam.git "$WORK" || { echo "=== EXIT=90 clone failed ==="; exit 90; }
cd "$WORK"
HEAD_LOCAL=$(git rev-parse --short HEAD)
echo "=== strike FULL verify @ ${HEAD_LOCAL} $(date -u +%FT%TZ) ==="
git log --oneline -1
echo -n "marginal_slope unused-var gate (expect 0): "; grep -cE "let h_range|let w_range" src/families/survival/marginal_slope/tests.rs

CARGO=${PROJ}/.cargo/bin/cargo
NTLOG=${PROJ}/calib_logs/strike_full_nextest_${SLURM_JOB_ID}.txt

echo "=== (0) build lib (release) ==="
$CARGO build --release --lib 2>&1 | tail -6
LIB_RC=${PIPESTATUS[0]}
echo "lib build rc=$LIB_RC"
if [ $LIB_RC -ne 0 ]; then echo "=== EXIT=$LIB_RC lib build failed ==="; exit $LIB_RC; fi

# ---- lane (1)+(2): integration binaries + the 1471 quality module ----
# binary() matches the test TARGET stem; test() matches module::fn path.
INTEG='binary(/^(owed_(1376|1392|1395|1399|1404|1426|1427|1033|1450|1454|1470|1471|1373|1464|1412|299|975|993|1416|1417|1418|1026|1410))$/)'
INTEG="$INTEG | binary(/^(issue_1427_factor_by_single_centering|issue_1470_anova_ti_scattered_recovery|issue_1426_gammalog_recovery)$/)"
INTEG="$INTEG | binary(/^(bug_hunt_1426_gamma_log_reml_flat_valley_overfit|bug_hunt_1464_curv_sign_diagnostic|bug_hunt_1464_curv_sign_identifiable)$/)"
INTEG="$INTEG | binary(/^(diag_1376_matern_design_psi_fd|diag_1426_gammalog_basin|diag_1470_lambda_probe|sae_throughput_gate_honesty_guard_1412)$/)"
# #1471 baseline module + the 1393 wave-2 scam/gamboostLSS quality arms (same bin).
QUAL='binary(=quality) & ( test(/quality_validation_baseline_1471::/) | test(/quality_vs_scam_monotone_increasing_smooth::/) | test(/quality_vs_scam_convex_smooth::/) | test(/quality_vs_gamboostlss_gaussian_location_scale::/) )'
FILTER="( $INTEG ) | ( $QUAL )"

echo "=== (1+2) nextest run (release, --no-fail-fast) ==="
echo "FILTER: $FILTER"
$CARGO nextest run --release --no-fail-fast -E "$FILTER" 2>&1 | tee "$NTLOG"
NT_RC=${PIPESTATUS[0]}
echo "nextest rc=$NT_RC"

# ---- lane (3): in-crate lib unit tests for #1026 / #299 preconditioner study ----
echo "=== (3) in-crate lib tests: linear_parity_anchor_1026 + arrow_precond ladder ==="
LIBLOG=${PROJ}/calib_logs/strike_full_libtest_${SLURM_JOB_ID}.txt
$CARGO test --release --lib linear_parity_anchor_1026 -- --nocapture 2>&1 | tee "$LIBLOG"
LT1_RC=${PIPESTATUS[0]}
$CARGO test --release --lib arrow_precond -- --nocapture 2>&1 | tee -a "$LIBLOG"
LT2_RC=${PIPESTATUS[0]}
echo "lib-test linear_parity_anchor_1026 rc=$LT1_RC ; arrow_precond rc=$LT2_RC"

echo "=== PER-TEST RESULTS (integration + quality) ==="
grep -E '^\s+(PASS|FAIL|TIMEOUT|SLOW|LEAK|SKIP)\b' "$NTLOG" || echo "(no per-test status lines — build error?)"
echo "=== nextest summary ==="
grep -E 'Summary|tests? run|failed|passed|Starting' "$NTLOG" | tail -6
echo "=== lib-test (#1026/#299) result lines ==="
grep -E 'test result:|linear_parity_anchor|arrow_precond|iteration|FAILED|ok$' "$LIBLOG" | tail -30

echo "=== AGGREGATE EXITS: nextest=$NT_RC libtest1026=$LT1_RC arrow_precond=$LT2_RC ==="
echo "=== EXIT=$NT_RC ==="
exit $NT_RC

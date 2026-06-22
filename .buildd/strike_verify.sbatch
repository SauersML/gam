#!/bin/bash
# strike_verify.sbatch — MSI verification harness for the strike-team owed tests.
#
# Resets a private clone to origin/main, builds the lib + the owed test binaries
# once (dedicated CARGO_TARGET_DIR tag so it never contends with the fleet's
# build.sh / other agents' per-tag target locks), then runs `cargo nextest run`
# filtered to JUST the strike-team owed test binaries via a binary() regex — so
# any not-yet-landed test simply doesn't match and never errors.
#
# This is INFORMATION, not a gate: --no-fail-fast runs every matched test and
# the post-run grep prints a per-binary PASS/FAIL summary with the real numbers.
#SBATCH -A hsiehph
#SBATCH -p msismall
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 3:00:00
#SBATCH -J strikeverify
#SBATCH -o /projects/standard/hsiehph/sauer354/calib_logs/strike_verify_%j.out
#SBATCH -e /projects/standard/hsiehph/sauer354/calib_logs/strike_verify_%j.out
set -uo pipefail
PROJ=/projects/standard/hsiehph/sauer354
source ${PROJ}/gam_env.sh 2>/dev/null || true

# Dedicated, re-warmable target dir for the verification lane — keeps cargo's
# exclusive per-target lock off the shared gam-target-fast that the fleet uses.
export CARGO_TARGET_DIR=${PROJ}/gam-target-strike-verify
export CARGO_PROFILE_DEV_DEBUG=0
export CARGO_PROFILE_TEST_DEBUG=0
export CARGO_PROFILE_DEV_SPLIT_DEBUGINFO=off
export CARGO_INCREMENTAL=1

WORK=${SLURM_TMPDIR:-/tmp}/strike_verify_${SLURM_JOB_ID}
rm -rf "$WORK"
git clone --depth 1 --branch main https://github.com/SauersML/gam.git "$WORK" || { echo "=== EXIT=90 clone failed ==="; exit 90; }
cd "$WORK"
HEAD_LOCAL=$(git rev-parse --short HEAD)
HEAD_REMOTE=$(git ls-remote origin refs/heads/main | cut -f1 | cut -c1-9)
echo "=== strike verify @ local=${HEAD_LOCAL} remote=${HEAD_REMOTE} $(date -u +%FT%TZ) ==="
git log --oneline -6

CARGO=${PROJ}/.cargo/bin/cargo

# Strike-team owed test binaries. nextest's binary() predicate matches the test
# TARGET name (the file stem under tests/). Not-yet-landed entries simply won't
# match — no error. The 1470 anova arm has a non-owed_ stem, matched explicitly.
FILTER='binary(/^owed_(1376|1392|1426|1427|979|1389|1395|1410|1450|1454|1373|1404|1464|1412|299)$/) | binary(/issue_1470_anova_ti_scattered_recovery/)'

echo "=== building lib + matched test binaries (release) ==="
$CARGO build --release --lib 2>&1 | tail -8
LIB_RC=${PIPESTATUS[0]}
echo "lib build rc=$LIB_RC"
if [ $LIB_RC -ne 0 ]; then echo "=== EXIT=$LIB_RC lib build failed (ban-scanner or compile) ==="; exit $LIB_RC; fi

echo "=== nextest run (release, --no-fail-fast) ==="
$CARGO nextest run --release --no-fail-fast -E "$FILTER" 2>&1 | tee ${PROJ}/calib_logs/strike_verify_nextest_${SLURM_JOB_ID}.txt
NT_RC=${PIPESTATUS[0]}
echo "nextest rc=$NT_RC"

echo "=== PER-TEST SUMMARY (PASS/FAIL lines) ==="
grep -E '^\s+(PASS|FAIL|TIMEOUT|SLOW|LEAK)\b' ${PROJ}/calib_logs/strike_verify_nextest_${SLURM_JOB_ID}.txt || echo "(no per-test status lines captured)"
echo "=== nextest tail summary ==="
grep -E 'Summary|tests run|failed|passed' ${PROJ}/calib_logs/strike_verify_nextest_${SLURM_JOB_ID}.txt | tail -8

echo "=== EXIT=$NT_RC ==="
exit $NT_RC

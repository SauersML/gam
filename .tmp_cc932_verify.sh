#!/bin/bash
# #932 one-shot cross-check harness — run from gam root after a debug build.
# Usage: bash .tmp_cc932_verify.sh   (assumes `cargo test -p gam --lib --no-run` already built)
# Runs the FD localizer (per-channel gaps) + the full witness (3rd+4th+tripwire)
# against the prebuilt debug binary directly (~3s, no rebuild).
set -uo pipefail
cd "$(dirname "$0")"
BIN=$(ls -t target/debug/deps/gam-* 2>/dev/null | grep -vE '\.d$|\.rlib$|\.rmeta$' | while read f; do [ -x "$f" ] && echo "$f"; done | head -1)
[ -z "$BIN" ] && { echo "NO debug test binary — run: CARGO_INCREMENTAL=0 cargo test -p gam --lib --no-run"; exit 2; }
echo "BIN=$BIN ($(stat -f %Sm "$BIN"))"
echo "===== FD LOCALIZER (per-channel gaps, target <=1e-9) ====="
"$BIN" debug_flex_directional_quantities_fd_localize --nocapture --test-threads=1 2>&1 \
  | grep -E 'INPUT|^\[[0-9]|BOUNDARY-GAP|f_au|f_aa|f_a base|f_uv\[|test result'
echo "===== WITNESS (3rd[4 blocks] + 4th[3 blocks bidir] + tripwire) ====="
"$BIN" flex_contracted_tower_matches_independent_fd_witness_nonzero_deviation --nocapture --test-threads=1 2>&1 \
  | grep -E 'third|fourth|test result|FAILED|panicked|assert|sign' | tail -25
echo "@@@ VERIFY DONE rc=${PIPESTATUS[0]}"

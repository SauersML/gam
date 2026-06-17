#!/usr/bin/env bash
# build.sh — single-flight, coalescing build gate for the shared gam tree.
#
# The ONE process allowed to run cargo. Agents call THIS, never cargo directly.
# Contract: call me, I block until a compile covering the current on-disk tree
# has finished, then hand you that result. One cargo at a time on the shared
# target dir => no concurrent-build OOM/ENOSPC. Callers that pile up during a
# build reuse its result if nothing changed since its snapshot (coalescing): a
# burst of N requests collapses to one build.
#
# Usage:
#   ./build.sh                 # compile lib + all test binaries (the heavy shared step)
#   ./build.sh nextest run X   # after a successful build, run test X (fast, no recompile)
# Exit: 0 = ok, 101 = compile/test errors (cached), 75 = transient (timeout/OOM) — retry.
set -uo pipefail

REPO=/Users/user/gam
S="$REPO/.buildd"; mkdir -p "$S"
LOCK="$S/build.lock"; COVER="$S/coverage.marker"; SNAP="$S/snap.marker"
LOG="$S/last.log"; RESULT="$S/last.code"
# Incremental ON (default): rustc reuses unchanged functions at item/body
# granularity, so a small edit recompiles only the affected closure, not the
# whole gam crate. (CI sets =0 for an unrelated per-file-test-loop reason; that
# does NOT apply to this single-flight shared builder.)
export CARGO_TARGET_DIR="$REPO/target" CARGO_INCREMENTAL=1
TIMEOUT=1800

# If args given, this is a (cheap) test-run after binaries are already built:
# serialize it behind the same lock but don't touch the coalescing markers.
if [[ $# -gt 0 ]]; then
  exec 9>"$LOCK"; flock -x 9
  cd "$REPO" || exit 1
  timeout "$TIMEOUT" cargo "$@"; exit $?
fi

# No args: warm the gam lib (the ~10min monolith long-pole) once.
# Test binaries then compile incrementally per-test on top of the warm lib.
BUILD=(cargo build --lib)
HASHFILE="$S/last.hash"

# Content hash of the source tree: identical code => identical hash => no rebuild,
# regardless of mtimes (catches touch-but-unchanged and revert-to-identical).
tree_hash() {
  find "$REPO/src" "$REPO/tests" "$REPO/crates" "$REPO/Cargo.toml" \
    -type f \( -name '*.rs' -o -name '*.toml' \) -print0 2>/dev/null \
    | sort -z | xargs -0 shasum 2>/dev/null | shasum | cut -d' ' -f1
}

exec 9>"$LOCK"; flock -x 9
cd "$REPO" || exit 1
HASH="$(tree_hash)"
if [[ "$HASH" == "$(cat "$HASHFILE" 2>/dev/null)" && -f "$RESULT" ]]; then
  : # exact same code already built — DEDUP: return cached result, no rebuild
else
  timeout "$TIMEOUT" "${BUILD[@]}" >"$LOG" 2>&1; code=$?
  if (( code != 124 && code < 128 )); then        # cargo reached a verdict (0 ok / 101 errors)
    echo "$code" >"$RESULT"; echo "$HASH" >"$HASHFILE"
  else
    echo "(did not complete: $code — timeout/OOM/killed)" >>"$LOG"
    flock -u 9; echo "TRANSIENT BUILD FAILURE ($code) — retry"; exit 75
  fi
fi
code="$(cat "$RESULT" 2>/dev/null || echo 1)"
flock -u 9

if [[ "$code" == "0" ]]; then echo "BUILD OK (latest tree)"; exit 0
else echo "BUILD FAILED ($code):"; tail -n 40 "$LOG"; exit "$code"; fi

#!/usr/bin/env bash
# build.sh — single-flight, coalescing, content-dedup build gate for the shared
# gam tree. The ONE process allowed to run cargo. Agents call THIS, never cargo.
#
#   ./build.sh                 # warm the gam lib (single-flight, content-dedup)
#   ./build.sh nextest run X   # run test X (serialized; reuses the warm lib)
#
# Every invocation appends a structured record to .buildd/history.log:
#   time | request | dedup HIT/MISS | crates recompiled (count + names) | duration | exit
set -uo pipefail

REPO=/Users/user/gam
S="$REPO/.buildd"; mkdir -p "$S"
LOCK="$S/build.lock"; LOG="$S/last.log"; RESULT="$S/last.code"; HASHFILE="$S/last.hash"; HIST="$S/history.log"
export CARGO_TARGET_DIR="$REPO/target" CARGO_INCREMENTAL=1   # item-granularity reuse
TIMEOUT=1800
now() { date +%H:%M:%S; }; ep() { date +%s; }

# What did cargo actually (re)compile? Cargo prints "   Compiling <crate> ..." per
# crate it builds; absence => reused from cache. This is the crate-level recompile set.
compiled_crates() { grep -E '^[[:space:]]*Compiling ' "$LOG" 2>/dev/null | sed -E 's/^[[:space:]]*Compiling //' | awk '{print $1}'; }
record() { # args: req dedup exit dur
  local req="$1" dedup="$2" code="$3" dur="$4"
  local names; names="$(compiled_crates | paste -sd, - 2>/dev/null | cut -c1-400)"
  local n; n="$(compiled_crates | grep -c . 2>/dev/null || echo 0)"
  printf '[%s] req=%-26s dedup=%-4s recompiled=%-3s [%s] duration=%ss exit=%s\n' \
    "$(now)" "\"$req\"" "$dedup" "$n" "$names" "$dur" "$code" >> "$HIST"
}

# ---- test-run lane (args present): serialized, reuses warm lib ----
if [[ $# -gt 0 ]]; then
  REQ="$*"
  exec 9>"$LOCK"; flock -x 9
  cd "$REPO" || exit 1
  t0=$(ep); timeout "$TIMEOUT" cargo "$@" >"$LOG" 2>&1; code=$?; dur=$(( $(ep)-t0 ))
  record "$REQ" "n/a" "$code" "$dur"
  flock -u 9
  echo "[build.sh] $REQ -> exit $code in ${dur}s (recompiled $(compiled_crates | grep -c . ) crates)"
  tail -n 30 "$LOG"; exit $code
fi

# ---- warm-lib lane (no args): content-dedup ----
BUILD=(cargo build --lib)
tree_hash() {
  find "$REPO/src" "$REPO/tests" "$REPO/crates" "$REPO/Cargo.toml" \
    -type f \( -name '*.rs' -o -name '*.toml' \) -print0 2>/dev/null \
    | sort -z | xargs -0 shasum 2>/dev/null | shasum | cut -d' ' -f1
}

exec 9>"$LOCK"; flock -x 9
cd "$REPO" || exit 1
HASH="$(tree_hash)"; t0=$(ep)
if [[ "$HASH" == "$(cat "$HASHFILE" 2>/dev/null)" && -f "$RESULT" ]]; then
  code="$(cat "$RESULT")"
  record "LIB" "HIT" "$code" "0"             # exact same code — no rebuild
else
  timeout "$TIMEOUT" "${BUILD[@]}" >"$LOG" 2>&1; code=$?; dur=$(( $(ep)-t0 ))
  if (( code != 124 && code < 128 )); then
    echo "$code" >"$RESULT"; echo "$HASH" >"$HASHFILE"; record "LIB" "MISS" "$code" "$dur"
  else
    record "LIB" "MISS" "$code(transient)" "$dur"
    flock -u 9; echo "TRANSIENT BUILD FAILURE ($code) — retry"; exit 75
  fi
fi
flock -u 9
if [[ "$code" == "0" ]]; then echo "BUILD OK (latest tree, hash ${HASH:0:12})"; exit 0
else echo "BUILD FAILED ($code):"; tail -n 40 "$LOG"; exit "$code"; fi

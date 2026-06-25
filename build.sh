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

# Derive the repo root from this script's own location so build.sh is
# portable (the shared local tree AND a cluster clone), not pinned to one path.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Optional machine-local build env (kept OUT of the repo): lets a cluster point
# the linker at a system lib (e.g. MSI OpenBLAS) without baking a host-specific
# path into the repo. No-op where absent (e.g. macOS/Homebrew finds BLAS itself).
[ -f "$HOME/.config/gam-build-env" ] && . "$HOME/.config/gam-build-env"
S="$REPO/.buildd"; mkdir -p "$S"
LOCK="$S/build.lock"; LOG="$S/last.log"; RESULT="$S/last.code"; HASHFILE="$S/last.hash"; HIST="$S/history.log"
export CARGO_TARGET_DIR="$REPO/target" CARGO_INCREMENTAL=1   # item-granularity reuse

# ---------------------------------------------------------------------------
# Compiler cache (sccache): a content-addressed warm dependency cache shared by
# every agent tree on this box. sccache keys on preprocessed source + compiler
# version + flags, so it is immune to the absolute-path / mtime churn that makes
# a copied target/ dir worthless across machines. The heavy deps (faer, burn,
# arrow, nalgebra, ndarray, …) are non-incremental and cache perfectly; the gam
# crate itself keeps CARGO_INCREMENTAL for tight local item-reuse (sccache just
# passes incremental compiles through uncached — the two compose, they don't
# fight). First build after enabling is a one-time full rebuild (the rustc
# wrapper changes cargo's fingerprint); every cold tree after that is warm.
#
# Wired via cargo's own `--config build.rustc-wrapper` (below) rather than the
# RUSTC_WRAPPER env var: it is per-invocation and conditional, so a tree without
# sccache still builds, and we set no environment. The on-disk cache lives in
# sccache's default store; point it at a shared backend by exporting SCCACHE_DIR
# or SCCACHE_BUCKET in your shell (sccache reads those natively — that is also
# how you'd share one cache between this box and CI). Opt out: GAM_NO_SCCACHE=1.
# ---------------------------------------------------------------------------
CARGO=(cargo)
if [[ -z "${GAM_NO_SCCACHE:-}" ]]; then
  # Auto-install once if missing — a cold cache shouldn't be the steady state.
  # Best-effort + idempotent: a single attempt is marked so we never re-hammer a
  # package manager on every build. Delete .buildd/.sccache_tried to force a retry.
  if ! command -v sccache >/dev/null 2>&1 && [[ ! -f "$S/.sccache_tried" ]]; then
    : >"$S/.sccache_tried"
    echo "[build.sh] sccache not found — installing it for a shared warm dep cache…" >&2
    {
      if command -v brew >/dev/null 2>&1; then brew install sccache
      elif command -v cargo >/dev/null 2>&1; then cargo install sccache --locked
      fi
    } >&2 2>&1 || true
  fi
  if command -v sccache >/dev/null 2>&1; then
    CARGO+=(--config 'build.rustc-wrapper="sccache"')
    # sccache rejects incremental compilation ("prohibited") and cannot cache
    # it anyway; disable incremental whenever the sccache wrapper is active.
    export CARGO_INCREMENTAL=0
  fi
fi
# Compact one-line cache summary, in the same telemetry spirit as history.log.
# No-ops silently when sccache is inactive or its stats format shifts.
sccache_summary() {
  [[ "${CARGO[*]}" == *rustc-wrapper* ]] || return 0
  sccache --show-stats 2>/dev/null \
    | grep -iE 'compile requests[[:space:]]|cache hits[[:space:]]+[0-9]|cache misses[[:space:]]+[0-9]' \
    | sed 's/^[[:space:]]*/[sccache] /' >&2 || true
}

TIMEOUT=1800
MIN_FREE_GB="${MIN_FREE_GB:-8}"
now() { date +%H:%M:%S; }; ep() { date +%s; }

# Preflight: bail FAST instead of dying at link with ENOSPC (which also corrupts
# the target dir and wastes ~90s). If disk is too low, point the caller at cluster.
free_gb="$(df -g "$REPO" 2>/dev/null | awk 'NR==2{print $4}')"
if [[ -n "$free_gb" && "$free_gb" -lt "$MIN_FREE_GB" ]]; then
  echo "[build.sh] only ${free_gb}G free (< ${MIN_FREE_GB}G) — a local build would ENOSPC at link." >&2
  # Pluggable remote backend: set GAM_REMOTE_RUN=<runner> (kept out of the repo)
  # to auto-route heavy builds/tests off this box. Keeps build.sh backend-agnostic.
  if [[ -n "${GAM_REMOTE_RUN:-}" ]]; then
    echo "[build.sh] routing to remote runner: $GAM_REMOTE_RUN $*" >&2
    exec "$GAM_REMOTE_RUN" "$@"
  fi
  echo "[build.sh] set GAM_REMOTE_RUN=<remote runner> to auto-route, or run on a box with disk." >&2
  exit 70
fi

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

# ---- maturin lane: `./build.sh maturin [extra maturin args]` ----
# Builds the gamfit Python extension through the SAME single-flight lock + sccache
# warm dep cache as the cargo lanes, so concurrent agents/jobs do not each recompile
# the heavy dep tree (faer/arrow/burn/…) — the difference between a ~2-min gam-only
# rebuild and a ~20-min cold one. The cargo lanes wire sccache via `--config
# build.rustc-wrapper`; maturin runs its own internal cargo, which honours the
# RUSTC_WRAPPER env form instead, so set that here under the same sccache condition.
if [[ "${1:-}" == "maturin" ]]; then
  shift
  if command -v sccache >/dev/null 2>&1 && [[ -z "${GAM_NO_SCCACHE:-}" ]]; then
    export RUSTC_WRAPPER="$(command -v sccache)"   # CARGO_INCREMENTAL already 0 above
  fi
  REQ="maturin develop --release $*"
  exec 9>"$LOCK"; flock -x 9
  cd "$REPO" || exit 1
  t0=$(ep); timeout "$TIMEOUT" maturin develop --release "$@" >"$LOG" 2>&1; code=$?; dur=$(( $(ep)-t0 ))
  record "$REQ" "n/a" "$code" "$dur"
  flock -u 9
  echo "[build.sh] $REQ -> exit $code in ${dur}s (recompiled $(compiled_crates | grep -c . ) crates)"
  sccache_summary
  tail -n 30 "$LOG"; exit "$code"
fi

# ---- test-run lane (args present): serialized, reuses warm lib ----
if [[ $# -gt 0 ]]; then
  REQ="$*"
  exec 9>"$LOCK"; flock -x 9
  cd "$REPO" || exit 1
  t0=$(ep); timeout "$TIMEOUT" "${CARGO[@]}" "$@" >"$LOG" 2>&1; code=$?; dur=$(( $(ep)-t0 ))
  record "$REQ" "n/a" "$code" "$dur"
  flock -u 9
  echo "[build.sh] $REQ -> exit $code in ${dur}s (recompiled $(compiled_crates | grep -c . ) crates)"
  sccache_summary
  tail -n 30 "$LOG"; exit $code
fi

# ---- warm-lib lane (no args): content-dedup ----
BUILD=("${CARGO[@]}" build --lib)
tree_hash() {
  # Include Cargo.lock so a dependency bump invalidates the dedup cache (else a
  # stale cached result could be served after deps changed but no *.rs did).
  { find "$REPO/src" "$REPO/tests" "$REPO/crates" -type f \( -name '*.rs' -o -name '*.toml' \) -print0 2>/dev/null \
      | sort -z | xargs -0 shasum 2>/dev/null
    shasum "$REPO/Cargo.toml" "$REPO/Cargo.lock" 2>/dev/null; } | shasum | cut -d' ' -f1
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
sccache_summary
if [[ "$code" == "0" ]]; then echo "BUILD OK (latest tree, hash ${HASH:0:12})"; exit 0
else echo "BUILD FAILED ($code):"; tail -n 40 "$LOG"; exit "$code"; fi

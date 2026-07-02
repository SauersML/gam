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
# Per-invocation log: concurrent build.sh calls must NOT share one file, or each
# clobbers the others' cargo output and an agent can't see its own compile errors.
# Each process writes to its own run.<pid>.log; $LAST is a best-effort "most
# recent" convenience copy (non-authoritative under concurrency). The reliable
# channel is this script's STDOUT (the tail printed by finish()), which agents
# capture directly. A trap removes the private log on exit.
LOCK="$S/build.lock"; RUNLOCK="$S/run-exec.lock"; LOG="$S/run.$$.log"; LAST="$S/last.log"; HIST="$S/history.log"
trap 'rm -f "$LOG" 2>/dev/null' EXIT
# Prune stale private logs left by killed processes (best-effort).
find "$S" -maxdepth 1 -name 'run.*.log' -mmin +120 -delete 2>/dev/null || true
export CARGO_TARGET_DIR="$REPO/target" CARGO_INCREMENTAL=1   # item-granularity reuse

# Memory-safe parallelism cap. Heavy non-incremental deps (faer, nano-gemm, gemm,
# burn) spawn one rustc per crate; on a small-RAM box (e.g. 8 GB) the default
# -j<ncpu> codegen fan-out OOMs and the kernel SIGKILLs rustc (signal 9) mid-link,
# which looks like a spurious "could not compile faer". Cap concurrent codegen so
# the build is slow-but-survivable. Override with CARGO_BUILD_JOBS=<n> in your env
# (a big-RAM box / cluster can raise it). Default keys off detected RAM.
if [[ -z "${CARGO_BUILD_JOBS:-}" ]]; then
  mem_gb=""
  if mb=$(sysctl -n hw.memsize 2>/dev/null); then mem_gb=$(( mb / 1073741824 ))   # macOS
  elif km=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null); then mem_gb=$(( km / 1048576 )); fi
  if [[ -n "$mem_gb" && "$mem_gb" -le 10 ]]; then export CARGO_BUILD_JOBS=2
  elif [[ -n "$mem_gb" && "$mem_gb" -le 18 ]]; then export CARGO_BUILD_JOBS=4; fi
fi

# ---- help lane (before any guard, so it works even on a disk/RAM-tight box) ----
case "${1:-}" in
  -h|--help|help)
    cat >&2 <<'EOF'
build.sh — single-flight build/test gate for the shared gam tree.
Agents call THIS, never `cargo` directly (it serializes the RAM-heavy compiles so
a fleet of agents on one warm target/ dir don't OOM-storm each other).

LANES
  ./build.sh                       warm the gam lib (cargo build --lib), dedup'd
  ./build.sh crate A [B…]          check ONLY crate(s) A,B… — fast + light, prefer this
  ./build.sh changed               check just the crates you edited (from git status)
  ./build.sh nextest run <FILTER>  compile+run tests (compile under lock, exec under run-lock)
  ./build.sh check --workspace …   any cargo subcommand + args (cacheable)
  ./build.sh maturin [args]        build+install the gamfit Python extension (release)
  ./build.sh -h | --help           this help

KEY ENV OVERRIDES
  CARGO_BUILD_JOBS=<n>     codegen parallelism (auto: ≤10G RAM→2, ≤18G→4; auto-drops
                          to 1 on an OOM retry). Lower = less peak RAM, slower.
  GAM_MIN_FREE_RAM_GB=<n>  RAM headroom to wait for before compiling (default 3)
  GAM_MEM_WAIT_MAX=<s>     max seconds to wait for that RAM, then proceed (default 300)
  MIN_FREE_GB=<n>          disk-headroom preflight; bails fast instead of ENOSPC (default 4)
  GAM_BUILD_TIMEOUT=<s>    per-command wall-clock cap (default 1800)
  GAM_FORCE=1              bypass result cache + coalescing, always run, refresh cache
  GAM_NO_CACHE=1           don't read/write the result cache (still single-flight)
  GAM_USE_SCCACHE=1        use sccache (cold trees/CI; trades away incremental)
  GAM_REMOTE_RUN=<runner>  route builds off-box when local disk is too low

NOTES
  • Result cache: the same command on unchanged source serves the prior result in ~0s.
  • Transient OOM/timeout are auto-retried (dropping to -j1) and never cached.
  • The maturin lane auto-tunes (-j1, LTO off) when FREE RAM is low, to survive the link.
EOF
    exit 0 ;;
esac

# ---------------------------------------------------------------------------
# INCREMENTAL-FIRST (the whole point of the crate split). The fleet hammers ONE
# shared, warm target/ dir: editing a single crate must recompile only THAT crate
# (item-level incremental) + relink — tens of seconds — NOT a 500-800s full
# rebuild of the gam monolith. That is exactly what CARGO_INCREMENTAL=1 (set
# above) buys, and it is the reason the workspace is split into crates.
#
# sccache is DISABLED by default here because it actively defeats that: sccache
# rejects incremental compilation ("prohibited"), so turning it on forces
# CARGO_INCREMENTAL=0 and every edit becomes a FULL crate recompile. On this
# single warm tree sccache also bought nothing — its dep artifacts are already
# resident in target/ and reused via cargo's own fingerprints (measured: ~0%
# sccache hit rate, 100% miss, pure overhead). sccache only helps COLD/throwaway
# trees (fresh clones, CI) that don't share target/. Opt back in there with
# GAM_USE_SCCACHE=1 (which trades incremental away on purpose).
# ---------------------------------------------------------------------------
CARGO=(cargo)
if [[ -n "${GAM_USE_SCCACHE:-}" ]]; then
  if ! command -v sccache >/dev/null 2>&1 && [[ ! -f "$S/.sccache_tried" ]]; then
    : >"$S/.sccache_tried"
    echo "[build.sh] sccache requested but not found — installing…" >&2
    {
      if command -v brew >/dev/null 2>&1; then brew install sccache
      elif command -v cargo >/dev/null 2>&1; then cargo install sccache --locked
      fi
    } >&2 2>&1 || true
  fi
  if command -v sccache >/dev/null 2>&1; then
    CARGO+=(--config 'build.rustc-wrapper="sccache"')
    export CARGO_INCREMENTAL=0   # sccache forbids incremental; cold-tree mode only
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

# Inner wall-clock cap for the build / test-run lane. Defaults to 30 min, which
# covers every routine build and the vast majority of tests. A few legitimate
# multi-n perf-acceptance sweeps (e.g. the #1033 κ-loop ladder fitting n up to
# 320k) genuinely run longer; override with GAM_BUILD_TIMEOUT=<seconds> for those
# without forking the shared gate. Routine callers are unaffected.
TIMEOUT="${GAM_BUILD_TIMEOUT:-1800}"
MIN_FREE_GB="${MIN_FREE_GB:-4}"   # incremental DEBUG builds write small deltas + a
# modest link (not a full release link), so an 8G headroom is overkill and on this
# disk-tight shared box it refused every build at ~7G free. 4G covers an incremental
# umbrella link; disk_guard's <3G near-ENOSPC backstop (drops deps+incremental) is the
# real corruption guard. Raise via MIN_FREE_GB=<n> on a box doing full release builds.
now() { date +%H:%M:%S; }; ep() { date +%s; }

# Preflight: bail FAST instead of dying at link with ENOSPC (which also corrupts
# the target dir and wastes ~90s). If disk is too low, point the caller at cluster.
free_kib="$(df -Pk "$REPO" 2>/dev/null | awk 'NR==2{print $4}')"   # POSIX: $4 = available KiB
free_gb=""
[[ -n "$free_kib" ]] && free_gb=$(( free_kib / 1024 / 1024 ))      # KiB -> GiB
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

# ---- lock-free fast-fail pre-flight (the parallel-agent fix) -----------------
# The authoritative ban-scanner runs INSIDE the cargo compile (the root gam
# build.rs), so a broken tree only fails AFTER a full recompile — 15-19 min under
# fleet churn — and holds the single global build lock for that whole DOOMED
# compile. That is exactly what pileup'd 47 agents in the meltdown: every agent
# independently recompiled a tree that could never pass, serialized one at a time.
# Catch the CHEAP, exact, tree-content ban violation here in ~0.3s WITHOUT any
# lock, so a broken tree fails every agent FAST and IN PARALLEL instead of
# serializing doomed builds. Only the 10k-line gate goes here — it is exact (wc),
# has ZERO false-positive risk, and was the recurring meltdown cause; the
# authoritative build.rs scanner still backstops every other rule. Never caches
# (the tree can be fixed at any moment). GAM_NO_BANSCAN=1 skips it.
banscan_fast_fail() {
  [[ -n "${GAM_NO_BANSCAN:-}" ]] && return 0
  local over
  # One `xargs wc` over all tracked *.rs (null-delimited, space-safe) ~0.3s, vs
  # 1800 per-file wc spawns (~5s). awk drops the "total" lines and flags >10000.
  over="$(cd "$REPO" && git ls-files -z '*.rs' 2>/dev/null | xargs -0 wc -l 2>/dev/null \
            | awk '$2!="total" && $1>10000 {printf "  %s: %s lines (limit 10000)\n",$2,$1}')"
  if [[ -n "$over" ]]; then
    echo "[build.sh] ban pre-check FAILED (10k-line gate) — failing fast, NO build lock taken:" >&2
    echo "$over" >&2
    echo "[build.sh] fix the file(s) above (split into a sibling module); the real build.rs scanner would reject them anyway after a full recompile." >&2
    : >"$LOG"; echo "ban pre-check: file over 10000-line gate (see stderr)" >>"$LOG"
    exit 101
  fi
}

# ---------------------------------------------------------------------------
# Unified coalescing + content-dedup engine (single-flight across the whole box).
#
# Every cacheable request (warm-lib build / check / test / nextest / clippy / …)
# is keyed on
#       KEY = sha1( request-args + tree content hash ).
# The tree hash makes the key source-sensitive: the SAME command on the SAME
# source serves the cached exit code + log in ~0s (HIT) and recompiles nothing;
# editing any file the ban-scanner reads — every *.rs / *.py / *.toml / *.yml /
# *.yaml / *.sh / *.bash / *.json / Makefile in the tree, plus Cargo.lock —
# changes the hash → a fresh run (MISS). The key surface is a superset of the
# scan surface (see scan_surface_files0) so the gate can never serve a cached
# pass for a tree the authoritative build.rs scanner would fail (#2092).
#
# Coalescing: two agents issuing the SAME live request concurrently share ONE
# run, even if unrelated source edits happen while they are queued. The first
# becomes the request leader and executes once under the global build lock; the
# rest block on the request lock and then serve the leader's result
# (COALESCED_REQUEST) instead of taking another global-lock turn. DIFFERENT
# requests still serialize through the global lock (one cargo at a time — the
# small-RAM invariant) but never duplicate each other's work. This generalises
# the old warm-lib-only dedup to test runs too: re-running the same `nextest run
# X` on unchanged source is now free, and a swarm of agents asking for it runs it
# exactly once.
#
# Transient outcomes (timeout 124 / signal ≥128, e.g. an OOM-SIGKILL) are NEVER
# cached; the caller (and any coalesced followers) get exit 75 = "retry".
#
# Escape hatches:  GAM_FORCE=1     bypass cache + coalescing, always execute,
#                                  then refresh the cache;
#                  GAM_NO_CACHE=1  neither read nor write the result cache
#                                  (still single-flight). Use for a flaky test or
#                                  a test that depends on a non-*.rs data fixture
#                                  the tree hash does not cover.
# ---------------------------------------------------------------------------
CACHEDIR="$S/cache"; REQDIR="$S/requests"; mkdir -p "$CACHEDIR" "$REQDIR"
# Best-effort prune of stale cache/lock entries (>7 days) so .buildd stays bounded.
find "$CACHEDIR" "$REQDIR" -mindepth 1 -maxdepth 1 -mtime +7 -exec rm -rf {} + 2>/dev/null || true

# `stat` flags differ by OS: macOS uses -f, GNU/Linux uses -c. Detect once.
if stat -f '%m' "$REPO" >/dev/null 2>&1; then STAT_ARGS=(-f '%m %z %N'); else STAT_ARGS=(-c '%Y %s %n'); fi
# Null-delimited list of every file the authoritative ban-scanner (root build.rs)
# reads. The dedup KEY must be a superset of the SCAN inputs — otherwise build.sh
# can serve a cached "pass" for a tree the scanner would REJECT: a violation in a
# scanned file OUTSIDE the key leaves the hash unchanged, so the warm fleet path
# skips cargo (and thus the scan) while cold clones / CI still fail. That gap is
# #2092 — its 15 offenders included examples/sac_prototype.py deferred-work
# markers and a gamfit/.github .py/.yml surface the old {src,tests,crates}/*.{rs,toml} key never
# covered, so every warm local build passed a tree the MSI cold build failed. The
# extension set (rs|py|toml|yml|yaml|sh|bash|json + Makefile) and the directory
# skip-list below mirror build.rs's collect_scannable_files exactly, so the key
# now tracks the scan surface file-for-file. Cargo.lock (no scannable extension)
# is appended by the caller so dep bumps still invalidate.
scan_surface_files0() {
  find "$REPO" \
      \( -type d \( \( -name '.?*' ! -name '.github' \) -o -name target -o -name 'target-*' \
          -o -name node_modules -o -name __pycache__ -o -name pydeps -o -name site-packages \
          -o -name venv -o -name dist -o -name build -o -name site \) -prune \) -o \
      \( -type f \( -name '*.rs' -o -name '*.py' -o -name '*.toml' -o -name '*.yml' \
          -o -name '*.yaml' -o -name '*.sh' -o -name '*.bash' -o -name '*.json' \
          -o -name Makefile \) -print0 2>/dev/null \)
}
tree_hash() {
  # Source signature from file METADATA (mtime+size+path), NOT content. A full
  # content shasum reads every one of ~2.5k scanned files on EACH call; with a
  # swarm of agents that's N concurrent whole-repo reads before the lock — a
  # self-inflicted IO/CPU storm. stat-only is ~10x cheaper, and since any normal
  # edit bumps mtime (and Cargo.lock is included so dep bumps invalidate) an agent
  # always sees its own change. Set GAM_CONTENT_HASH=1 to force exact content hash.
  if [[ -n "${GAM_CONTENT_HASH:-}" ]]; then
    { scan_surface_files0 | sort -z | xargs -0 shasum 2>/dev/null
      shasum "$REPO/Cargo.lock" 2>/dev/null; } | shasum | cut -d' ' -f1
  else
    { scan_surface_files0 | sort -z | xargs -0 stat "${STAT_ARGS[@]}" 2>/dev/null
      stat "${STAT_ARGS[@]}" "$REPO/Cargo.lock" 2>/dev/null; } | shasum | cut -d' ' -f1
  fi
}
sha_str() { printf '%s' "$1" | shasum | cut -d' ' -f1; }
normalize_req() {
  # Canonical request identity for LIVE coalescing. Persistent cache keys remain
  # content-sensitive; this only decides whether two currently in-flight callers
  # are asking the same cargo question and should share one global-lock turn.
  printf '%s' "$1" | awk '{$1=$1; print}'
}

finish() {  # args: code label — print the standard summary + log tail, then exit
  local code="$1" label="$2"
  cp -f "$LOG" "$LAST" 2>/dev/null || true   # best-effort "latest" snapshot
  echo "[build.sh] $REQ -> exit $code in ${DUR:-0}s (dedup=$label, recompiled $(compiled_crates | grep -c .) crates)"
  sccache_summary
  if [[ "$code" == "0" ]]; then tail -n 30 "$LOG"; else echo "FAILED ($code):"; tail -n 40 "$LOG"; fi
  exit "$code"
}
serve_cache() {  # args: label — restore the cached log, record, finish in ~0s
  local label="$1" code
  cp -f "$CACHE/log" "$LOG" 2>/dev/null || : >"$LOG"
  code="$(cat "$CACHE/code")"; DUR=0
  record "$REQ" "$label" "$code" 0
  finish "$code" "$label"
}
serve_request_result() {  # args: label request-hash — serve another live caller's result
  local label="$1" reqhash="$2" dir code
  dir="$REQDIR/$reqhash"
  [[ -f "$dir/code" ]] || { echo "[build.sh] active duplicate produced no cached/stable result — retry"; exit 75; }
  cp -f "$dir/log" "$LOG" 2>/dev/null || : >"$LOG"
  code="$(cat "$dir/code")"; DUR=0
  [[ "$code" == "75" ]] && { echo "[build.sh] active duplicate was transient — retry"; exit 75; }
  record "$REQ" "$label" "$code" 0
  finish "$code" "$label"
}
write_cached_request_result() {  # args: request-hash — publish CACHE as live result
  local reqhash="$1" dir
  dir="$REQDIR/$reqhash"
  mkdir -p "$dir"
  cp -f "$CACHE/log" "$dir/log.tmp" 2>/dev/null && mv -f "$dir/log.tmp" "$dir/log"
  cp -f "$CACHE/code" "$dir/code.tmp" 2>/dev/null && mv -f "$dir/code.tmp" "$dir/code"
}
# Available-RAM probe (GiB). macOS: free+inactive+speculative pages; Linux:
# MemAvailable. Returns nonzero if it can't tell (callers then proceed).
current_free_gb() {
  local pg free inact spec stats
  if pg=$(sysctl -n hw.pagesize 2>/dev/null); then
    stats=$(vm_stat 2>/dev/null) || return 1
    free=$(awk '/Pages free/{gsub(/\./,"",$3);print $3}' <<<"$stats")
    inact=$(awk '/Pages inactive/{gsub(/\./,"",$3);print $3}' <<<"$stats")
    spec=$(awk '/Pages speculative/{gsub(/\./,"",$3);print $3}' <<<"$stats")
    [[ -z "$free" ]] && return 1
    # Round to the NEAREST GiB (add 0.5 GiB before the integer divide) instead of
    # truncating: a box with 2.9 GiB available truncated to "2" and, with the
    # default need=3, waited the full GAM_MEM_WAIT_MAX every build for RAM it
    # already had. Rounding reports 3 and proceeds.
    echo $(( ( ( ${free:-0} + ${inact:-0} + ${spec:-0} ) * pg + 536870912 ) / 1073741824 )); return 0
  elif [[ -r /proc/meminfo ]]; then
    awk '/MemAvailable/{print int($2/1048576); f=1} END{exit !f}' /proc/meminfo; return $?
  fi
  return 1
}
# Block until the box has enough free RAM to start a compile, so we WAIT for a
# concurrent build (gam or foreign — polars/gnomon/another toolchain that our
# lock can't serialize) to free memory instead of launching into an OOM-SIGKILL.
# Bounded: after MEM_WAIT_MAX we proceed anyway and let the retry loop cope.
wait_for_memory() {
  local need="${GAM_MIN_FREE_RAM_GB:-3}" maxw="${GAM_MEM_WAIT_MAX:-300}" waited=0 fg
  while :; do
    fg=$(current_free_gb) || return 0          # can't measure → just proceed
    [[ "$fg" -ge "$need" ]] && return 0
    (( waited >= maxw )) && { echo "[build.sh] proceeding with only ${fg}G free after ${waited}s wait (<${need}G)" >&2; return 0; }
    (( waited % 30 == 0 )) && echo "[build.sh] waiting for RAM headroom: ${fg}G free, need ${need}G (other builds running)…" >&2
    sleep 6; waited=$((waited+6))
  done
}
# An OOM-SIGKILL of a child rustc makes cargo exit 101 (NOT 137) with
# "(signal: 9, SIGKILL)" in its output — indistinguishable from a real compile
# error by exit code alone. Detect timeout / direct-signal / OOM and treat them
# as TRANSIENT: never cached, auto-retried. Uses globals: code, $LOG.
is_transient() {
  (( code == 124 )) && return 0        # timeout(1) wall-clock kill
  (( code >= 128 )) && return 0        # killed by a signal directly
  (( code == 0 ))   && return 1
  grep -qiE 'signal: 9|SIGKILL|process did not exit successfully.*signal|: Killed|out of memory|Cannot allocate memory|memory allocation of .* bytes failed|LLVM ERROR: out of memory' "$LOG" 2>/dev/null \
    || is_incremental_corruption
}
# Incremental-state corruption: a CONCURRENT cargo (another session / a codex
# launchd job that bypasses this single-flight gate) touching the same target/
# dir can delete target/debug/incremental mid-build, yielding "failed to
# create query cache" / "failed to move dependency graph … No such file or
# directory". That is an INFRASTRUCTURE failure, not a code error — treat it as
# transient and wipe the incremental dir before retrying so the rebuild starts
# from clean incremental state.
is_incremental_corruption() {
  (( code == 0 )) && return 1
  grep -qiE 'failed to (create|move|open|read|write).*(query cache|dependency graph|incremental)|incremental compilation.*(No such file|cannot|failed)|query-cache\.bin|dep-graph\.(part\.)?bin' "$LOG" 2>/dev/null
}
# Is CMD a test/bench RUN that compiles binaries and then EXECUTES them? (A pure
# build-only invocation — `--no-run`/`--list` — is NOT: it never executes, so it
# needs no execute phase.) Only these are split into compile+execute below.
is_test_run() {
  local a saw_run="" saw_norun=""
  for a in "${CMD[@]}"; do
    case "$a" in
      test|nextest|bench) saw_run=1 ;;
      --no-run|--list|list) saw_norun=1 ;;   # build-only / listing: never execute
    esac
  done
  [[ -n "$saw_run" && -z "$saw_norun" ]]
}

# Run CMD with a memory gate + auto-retry on transient (OOM/timeout) failures.
# Sets globals code + DUR + $LOG.
#
# The global lock ($LOCK) exists to serialize RAM-heavy COMPILATION — one rustc
# storm at a time — NOT test EXECUTION. Holding it across a long test run (perf/
# acceptance sweeps run 500-800s) needlessly blocks every other agent's compile.
# So for a test RUN we split: compile the binaries under the global lock, RELEASE
# it, then execute the already-built binaries under a SEPARATE single-flight
# run-lock ($RUNLOCK). A compile and a test-execution can then overlap (bounded to
# one of each — the small-RAM invariant still holds, since execution is far
# lighter than codegen). Non-test requests are unchanged: whole run under $LOCK.
# The incremental-corruption handler is the backstop if a concurrent compile
# perturbs target/ during the execute phase. GAM_NO_SPLIT=1 forces the old
# everything-under-one-lock behavior.
run_under_global_lock() {
  # Single-flight compile gate. NOTE: extra "slots" would NOT buy parallelism —
  # cargo self-serializes every build on the shared CARGO_TARGET_DIR (its own
  # target-dir + package-cache locks), so two concurrent cargos on one target
  # just block each other opaquely (and race the incremental cache). Real
  # parallelism needs per-build target dirs = redundant recompiles = more CPU +
  # disk, which we deliberately avoid. So the frugal design is: serialize heavy
  # compiles here, and make redundant work FREE via the result cache (unchanged
  # source → ~0s HIT, no lock) + request coalescing (identical concurrent calls
  # share one turn). The abnormal multi-minute waits were never normal
  # serialization — they were a dead/orphaned build still HOLDING this lock; that
  # root cause is fixed by running cargo with the lock fd CLOSED (`9>&-` below),
  # so only this shell holds fd 9 and flock frees it the instant the shell dies,
  # plus `timeout -k` so a hung compile is SIGKILLed instead of holding forever.
  exec 9>"$LOCK"
  if ! flock -n -x 9; then
    echo "[build.sh] waiting for the global build lock — another build/test is compiling (one cargo at a time on this box)…" >&2
    flock -x 9
  fi
  cd "$REPO" || exit 1
  local attempt=0 max="${GAM_BUILD_RETRIES:-3}" t0 split=""
  # Compile/execute split is OPT-IN (default OFF) for RAM safety: with it on, one
  # agent's compile can overlap another's test EXECUTION, ~doubling peak heavy
  # processes — which OOM's a RAM-tight laptop. Default keeps the frugal
  # one-cargo/test-at-a-time invariant ("serialize heavy compiles here"). Enable
  # the overlap only on a big-RAM box with GAM_SPLIT=1.
  [[ -n "${GAM_SPLIT:-}" ]] && is_test_run && split=1
  while :; do
    attempt=$((attempt+1))
    wait_for_memory
    # Split test run: compile-only under the global lock (`--no-run`). Otherwise
    # run the whole command under the global lock as before.
    if [[ -n "$split" ]]; then
      t0=$(ep); timeout -k 30 "$TIMEOUT" "${CMD[@]}" --no-run >"$LOG" 2>&1 9>&-; code=$?; DUR=$(( $(ep)-t0 ))
    else
      t0=$(ep); timeout -k 30 "$TIMEOUT" "${CMD[@]}" >"$LOG" 2>&1 9>&-; code=$?; DUR=$(( $(ep)-t0 ))
    fi
    if is_transient; then
      if (( attempt < max )); then
        if is_incremental_corruption; then
          echo "[build.sh] incremental-state corruption (concurrent cargo on target/) attempt $attempt/$max — wiping target/debug/incremental + retrying…" >&2
          rm -rf "$REPO/target/debug/incremental" 2>/dev/null || true
        elif [[ "${CARGO_BUILD_JOBS:-0}" != "1" ]]; then
          # Repeated OOM won't clear by waiting alone — a lower job count has a
          # strictly lower codegen RAM peak, so shrink to -j1 for the retry.
          export CARGO_BUILD_JOBS=1
          echo "[build.sh] transient failure (OOM/timeout, exit $code) attempt $attempt/$max — retrying with CARGO_BUILD_JOBS=1 (lower peak RAM)…" >&2
        else
          echo "[build.sh] transient failure (OOM/timeout, exit $code) attempt $attempt/$max — backing off for memory, retrying…" >&2
        fi
        sleep $(( attempt * 12 )); continue
      fi
      echo "[build.sh] still transiently failing (exit $code) after $max attempts — giving up this round." >&2
    fi
    break
  done
  # EXECUTE phase (test run whose compile succeeded): release the global compile
  # lock so other agents can compile, then run the built binaries under $RUNLOCK.
  if [[ -n "$split" && "$code" == "0" ]]; then
    flock -u 9
    exec 8>"$RUNLOCK"; flock -x 8
    local rattempt=0
    while :; do
      rattempt=$((rattempt+1))
      wait_for_memory
      t0=$(ep); timeout -k 30 "$TIMEOUT" "${CMD[@]}" >>"$LOG" 2>&1 8>&- 9>&-; code=$?; DUR=$(( DUR + $(ep)-t0 ))
      if is_transient && (( rattempt < max )); then
        is_incremental_corruption && rm -rf "$REPO/target/debug/incremental" 2>/dev/null || true
        sleep $(( rattempt * 12 )); continue
      fi
      break
    done
    flock -u 8
    return
  fi
  flock -u 9
}
# Atomically commit code+log to the cache (log first, code file is the commit marker).
write_cache() {
  [[ "${CACHEABLE:-1}" == 1 && -z "${GAM_NO_CACHE:-}" ]] || return 0
  mkdir -p "$CACHE"
  cp -f "$LOG" "$CACHE/log.tmp" 2>/dev/null && mv -f "$CACHE/log.tmp" "$CACHE/log"
  printf '%s' "$code" >"$CACHE/code.tmp" && mv -f "$CACHE/code.tmp" "$CACHE/code"
}
write_request_result() {  # args: request-hash
  local reqhash="$1" dir
  dir="$REQDIR/$reqhash"
  mkdir -p "$dir"
  cp -f "$LOG" "$dir/log.tmp" 2>/dev/null && mv -f "$dir/log.tmp" "$dir/log"
  printf '%s' "$code" >"$dir/code.tmp" && mv -f "$dir/code.tmp" "$dir/code"
}

# ---- build-superset subsumption ---------------------------------------------
# A green FULL lib compile (`build --lib` / `check --workspace --lib`, no -p
# restriction) proves EVERY workspace crate's lib compiles — so it subsumes any
# scoped `check -p X --lib`. On such a success we drop a per-tree marker; scoped
# lib-compile requests for the same tree then short-circuit to exit 0 instead of
# each taking a global cargo turn. One full build draining the queue satisfies all
# the per-crate checks stacked behind it. (Tests/clippy/feature builds are NOT
# subsumable — they compile/run more than the lib.)
is_lib_compile() {  # CMD is a pure lib *compilation* (no test/extra-target/feature)?
  local a sub=""
  for a in "${CMD[@]}"; do
    case "$a" in
      check|build) sub=1 ;;
      test|nextest|run|bench|clippy|doc|miri|--tests|--all-targets|--bins|--bin|--examples|--example|--benches|--features|--all-features|--release|maturin) return 1 ;;
    esac
  done
  [[ -n "$sub" ]]
}
is_full_lib_compile() {  # …and spans the whole workspace (no -p restriction)?
  is_lib_compile || return 1
  local a has_p="" has_ws=""
  for a in "${CMD[@]}"; do
    [[ "$a" == "-p" || "$a" == "--package" ]] && has_p=1
    [[ "$a" == "--workspace" || "$a" == "--all" ]] && has_ws=1
  done
  [[ -n "$has_ws" || -z "$has_p" ]]
}

# The engine: REQ, CMD[], and CACHEABLE must be set by the caller before calling.
run_request() {
  local TREE KEY REQNORM REQHASH
  # Lock-free fast-fail: reject a tree that can't possibly pass the 10k-line gate
  # BEFORE taking the global lock or recompiling — this is what keeps a broken
  # tree from serializing the whole fleet through doomed 15-min compiles.
  banscan_fast_fail
  REQNORM="$(normalize_req "$REQ")"; REQHASH="$(sha_str "$REQNORM")"
  TREE="$(tree_hash)"; KEY="$(sha_str "$REQ"$'\n'"$TREE")"
  CACHE="$CACHEDIR/$KEY"

  # Forced run: skip cache-read + coalescing, execute, then refresh the cache.
  if [[ -n "${GAM_FORCE:-}" ]]; then
    run_under_global_lock
    if is_transient; then
      record "$REQ" "FORCE(transient)" "$code" "$DUR"; echo "TRANSIENT FAILURE ($code, OOM/timeout) — retry"; exit 75
    fi
    write_cache; { (( code == 0 )) && is_full_lib_compile && : >"$CACHEDIR/super.$TREE"; } || true
    record "$REQ" "FORCE" "$code" "$DUR"; finish "$code" "FORCE"
  fi

  # Non-cacheable requests and explicit no-cache runs must execute for their
  # side effects or fresh observations. They still use the global cargo lock, but
  # they do not serve a prior live caller's result.
  if [[ "${CACHEABLE:-1}" != 1 || -n "${GAM_NO_CACHE:-}" ]]; then
    run_under_global_lock
    if is_transient; then
      record "$REQ" "n/a(transient)" "$code" "$DUR"; echo "TRANSIENT FAILURE ($code, OOM/timeout) — retry"; exit 75
    fi
    record "$REQ" "n/a" "$code" "$DUR"; finish "$code" "n/a"
  fi

  # Build-superset subsumption: if a full green lib compile already covered this
  # exact tree, this scoped lib compile is guaranteed to pass — serve 0, no turn.
  if is_lib_compile && [[ -f "$CACHEDIR/super.$TREE" ]]; then
    : >"$LOG"; code=0; DUR=0; record "$REQ" "SUBSUMED" 0 0; finish 0 "SUBSUMED"
  fi

  # Fast path: a cached result for this exact (args, source) — serve with no lock.
  if [[ -f "$CACHE/code" ]]; then
    serve_cache "HIT"
  fi

  # Live duplicate coalescing is request-keyed, not tree-keyed. In a shared tree,
  # unrelated edits can change TREE while several agents are already waiting for
  # the same build/test request; without this lock they each become a separate
  # content key and all queue behind the global cargo lock. The leader executes
  # once; followers serve its result and do not consume another global turn.
  exec 7>"$REQDIR/$REQHASH.lock"
  if ! flock -n -x 7; then
    flock -x 7
    flock -u 7
    serve_request_result "COALESCED_REQUEST" "$REQHASH"
  fi
  rm -rf "$REQDIR/$REQHASH"
  mkdir -p "$REQDIR/$REQHASH"

  # Request leader. Re-check the content cache under the request lock (a prior
  # leader may have just finished between our fast-path read and acquiring it).
  if [[ "${CACHEABLE:-1}" == 1 && -z "${GAM_NO_CACHE:-}" && -f "$CACHE/code" ]]; then
    write_cached_request_result "$REQHASH"; flock -u 7; serve_cache "HIT"
  fi
  run_under_global_lock
  if is_transient; then
    code=75; write_request_result "$REQHASH"
    record "$REQ" "MISS(transient)" "$code" "$DUR"; flock -u 7
    echo "TRANSIENT FAILURE ($code, OOM/timeout) — retry"; exit 75
  fi
  write_cache; write_request_result "$REQHASH"
  { (( code == 0 )) && is_full_lib_compile && : >"$CACHEDIR/super.$TREE"; } || true
  record "$REQ" "MISS" "$code" "$DUR"; flock -u 7
  finish "$code" "MISS"
}

# ---- maturin lane: `./build.sh maturin [extra maturin args]` ----
# Builds the gamfit Python extension through the SAME single-flight lock + sccache
# warm dep cache as the cargo lanes. NOT result-cached: its value is the on-disk
# side effect (the installed extension), so it always actually runs. The cargo
# lanes wire sccache via `--config build.rustc-wrapper`; maturin runs its own
# internal cargo, which honours the RUSTC_WRAPPER env form instead.
if [[ "${1:-}" == "maturin" ]]; then
  shift
  # `maturin develop` builds for the ACTIVE virtualenv and does NOT accept
  # --interpreter (that's a `maturin build` flag). Strip it (with its value) so a
  # caller who passes it by habit gets a clear note, not a cryptic clap error.
  _margs=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -i|--interpreter)
        echo "[build.sh maturin] ignoring '$1 ${2:-}' — develop targets the active venv, not --interpreter" >&2
        shift 2 2>/dev/null || shift ;;
      *) _margs+=("$1"); shift ;;
    esac
  done
  set -- ${_margs[@]+"${_margs[@]}"}
  # maturin develop needs a virtualenv (activated, or a repo-local .venv/venv it
  # can find). Fail with an actionable message instead of a cryptic maturin error.
  if [[ -z "${VIRTUAL_ENV:-}" && ! -d "$REPO/.venv" && ! -d "$REPO/venv" ]]; then
    echo "[build.sh maturin] no active virtualenv and no .venv/ in the repo — 'maturin develop' needs one." >&2
    echo "[build.sh maturin] create it once: python3 -m venv .venv && ./.venv/bin/pip install maturin torch" >&2
    exit 78
  fi
  if command -v sccache >/dev/null 2>&1 && [[ -z "${GAM_NO_SCCACHE:-}" ]]; then
    # sccache FORBIDS incremental; the top-of-file default is CARGO_INCREMENTAL=1
    # and the =0 override only runs in the cargo-lane sccache block, NOT here — so
    # the maturin lane must zero it itself or `maturin develop` dies with
    # "sccache: incremental compilation is prohibited".
    export RUSTC_WRAPPER="$(command -v sccache)" CARGO_INCREMENTAL=0
  fi
  # A release extension build+link is the heaviest compile here. On a box with
  # little FREE RAM (regardless of total installed), the default codegen fan-out +
  # LTO link OOM-SIGKILLs mid-link. Auto-cap to one job and drop LTO so it survives
  # — slower, marginally less optimized, but it completes. All three stay overridable.
  _fg=$(current_free_gb 2>/dev/null || echo 99)
  if [[ "$_fg" =~ ^[0-9]+$ && "$_fg" -lt 5 ]]; then
    export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
    export CARGO_PROFILE_RELEASE_LTO="${CARGO_PROFILE_RELEASE_LTO:-false}"
    export CARGO_PROFILE_RELEASE_CODEGEN_UNITS="${CARGO_PROFILE_RELEASE_CODEGEN_UNITS:-256}"
    echo "[build.sh maturin] low free RAM (~${_fg}G) → CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS, LTO=$CARGO_PROFILE_RELEASE_LTO, codegen-units=$CARGO_PROFILE_RELEASE_CODEGEN_UNITS (slower, avoids OOM at link)" >&2
  fi
  REQ="maturin develop --release $*"
  CMD=(maturin develop --release "$@")
  run_under_global_lock
  record "$REQ" "n/a" "$code" "$DUR"
  finish "$code" "n/a"
fi

# ---- scoped lane: compile ONLY the crate(s) an agent touched -----------------
# The workspace is split into crates precisely so a fleet of agents don't each
# recompile everything. `cargo check -p <crate>` rebuilds only that crate (its
# dependencies are already warm in target/ + sccache; its *dependents* are not
# touched), and `check` skips codegen/link entirely — far faster and far lighter
# on RAM than a full `build --lib`. Two forms:
#   ./build.sh changed         # auto-detect changed crates from `git status` and
#                              #   check just those (falls back to a full
#                              #   workspace check if root src/ or Cargo.* changed)
#   ./build.sh crate A B …     # check exactly the named crate(s)
# Both are content-dedup'd + coalesced like every other request. Use these for
# tight iteration; run a full `./build.sh` (or `./build.sh check --workspace
# --tests`) once before you push to catch any downstream breakage.
if [[ "${1:-}" == "changed" || "${1:-}" == "crate" ]]; then
  mode="$1"; shift
  # NB: macOS ships bash 3.2 (no associative arrays / `declare -A`). Dedup with a
  # plain indexed array + linear membership check, guarding empty-array expansion
  # under `set -u` via the `${arr[@]+...}` idiom.
  _crates=(); _root=0
  _add_crate() {
    local x
    if [[ ${#_crates[@]} -gt 0 ]]; then
      for x in "${_crates[@]}"; do [[ "$x" == "$1" ]] && return 0; done
    fi
    _crates+=("$1")
  }
  if [[ "$mode" == "crate" ]]; then
    if [[ $# -eq 0 ]]; then echo "[build.sh] usage: ./build.sh crate <name> [name…]" >&2; exit 64; fi
    for c in "$@"; do _add_crate "$c"; done
  else
    # Parse `git status --porcelain` (handles renames "old -> new" by taking new).
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      f="${line:3}"; [[ "$f" == *" -> "* ]] && f="${f##* -> }"
      case "$f" in
        crates/*/*) c="${f#crates/}"; c="${c%%/*}"; _add_crate "$c" ;;
        src/*|Cargo.toml|Cargo.lock|tests/*) _root=1 ;;
      esac
    done < <(git -C "$REPO" status --porcelain 2>/dev/null)
  fi
  if [[ "$_root" == 1 || ${#_crates[@]} -eq 0 ]]; then
    REQ="check --workspace --lib"; CMD=("${CARGO[@]}" check --workspace --lib)
    [[ "$mode" == "changed" ]] && echo "[build.sh] scoped 'changed': root/test/Cargo change (or no crate diff) → full workspace check" >&2
  else
    _pflags=(); for c in "${_crates[@]}"; do _pflags+=(-p "$c"); done
    REQ="check ${_pflags[*]} --lib"; CMD=("${CARGO[@]}" check "${_pflags[@]}" --lib)
    echo "[build.sh] scoped: checking only ${_crates[*]}" >&2
  fi
  CACHEABLE=1
  run_request
fi

# ---- args lane: test / check / nextest / clippy / build-with-args (cacheable) ----
if [[ $# -gt 0 ]]; then
  REQ="$*"
  CMD=("${CARGO[@]}" "$@")
  CACHEABLE=1
  # Mutating / non-idempotent subcommands must always run (their value is a side
  # effect, not an exit code) — never serve them from cache.
  case "${1:-}" in
    clean|update|fetch|add|remove|rm|publish|install|uninstall|generate-lockfile|new|init|vendor)
      CACHEABLE=0 ;;
  esac
  run_request
fi

# ---- warm-lib lane (no args): content-dedup ----
REQ="LIB"
CMD=("${CARGO[@]}" build --lib)
CACHEABLE=1
run_request

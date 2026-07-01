#!/bin/bash
# build_kill_watcher.sh — reap stuck / orphaned / runaway gam builds so the
# single global build lock (.buildd/build.lock) never wedges the whole fleet.
#
# WHY: build.sh serializes RAM-heavy compiles one-cargo-at-a-time behind an
# flock'd global lock. That is correct (OOM guard), but nothing reaps the
# PATHOLOGICAL holders:
#   * a compile/link that HANGS and escapes build.sh's own `timeout 1800`
#     (hung children detach from the timeout(1) that wrapped cargo),
#   * a `cargo`/`rustc` ORPHANED (reparented to init, ppid==1) when its owning
#     agent is killed — it keeps holding the lock,
#   * a lock HOLDER making no compile progress (no rustc, stale log) for a long
#     window — a deadlocked build.
# Any of these blocks every other agent indefinitely (flock -x waits forever).
#
# SAFETY: a healthy build that is actively spawning rustc / growing its log is
# NEVER killed until it crosses the generous HARD_CAP. Only orphans, over-cap
# trees, and no-progress lock-holders are reaped. Each kill is logged.
#
# Usage:
#   nohup scripts/build_kill_watcher.sh >/dev/null 2>&1 &   # start
#   KW_STOP=1 scripts/build_kill_watcher.sh                 # signal running one to stop
# Tunables (env): KW_INTERVAL(20s) KW_HARD_CAP(2100s) KW_STUCK_SECS(900s)
set -u

REPO="${GAM_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
S="$REPO/.buildd"; mkdir -p "$S"
LOGF="$S/killwatcher.log"
LOCKF="$S/build.lock"
PIDFILE="$S/killwatcher.pid"
STOPFILE="$S/killwatcher.stop"
INTERVAL="${KW_INTERVAL:-20}"
HARD_CAP="${KW_HARD_CAP:-2100}"       # >build.sh's own 1800 TIMEOUT: only true escapees
STUCK_SECS="${KW_STUCK_SECS:-900}"    # holder with no compile progress this long = dead

log(){ echo "$(date '+%F %T') $*" >> "$LOGF"; }

# Allow a second invocation to stop the running watcher cleanly.
if [[ "${KW_STOP:-0}" == "1" ]]; then
  touch "$STOPFILE"; echo "[killwatcher] stop signalled"; exit 0
fi

# Single instance.
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  echo "[killwatcher] already running (pid $(cat "$PIDFILE"))"; exit 0
fi
echo $$ > "$PIDFILE"
rm -f "$STOPFILE" 2>/dev/null
log "START pid=$$ hard_cap=${HARD_CAP}s stuck=${STUCK_SECS}s interval=${INTERVAL}s repo=$REPO"

# pid -> epoch first observed as a build process (in-memory age tracking; robust
# across the macOS `ps` that lacks etimes).
declare -A SEEN
# holder pid -> epoch we last saw it make compile progress (a live rustc, or a
# grown run.<pid>.log).
declare -A PROG
declare -A HOLD_LOGSIZE

kill_tree(){ # kill_tree <pid> <SIG>: depth-first so children die before parent
  local root="$1" sig="$2" k
  for k in $(pgrep -P "$root" 2>/dev/null); do kill_tree "$k" "$sig"; done
  kill "-$sig" "$root" 2>/dev/null
}

reap(){ # reap <pid> <reason>
  local pid="$1" why="$2"
  log "KILL pid=$pid ($why) cmd=[$(ps -o args= -p "$pid" 2>/dev/null | cut -c1-90)]"
  kill_tree "$pid" TERM
  sleep 3
  kill -0 "$pid" 2>/dev/null && { kill_tree "$pid" KILL; log "  escalated pid=$pid to SIGKILL"; }
}

# Is a build-ish process (the ones that hold/consume the compile lock)?
is_build_proc(){ [[ "$1" =~ (cargo|cargo-nextest|rustc|/build\.sh|bash\ \./build\.sh) ]]; }

# Find the build.sh process that HOLDS the global lock: it has a cargo/timeout
# descendant but NO child `flock` process still blocking to acquire.
lock_holder(){
  local pid
  for pid in $(pgrep -f "bash ./build.sh" 2>/dev/null); do
    # a waiter's direct child is a live `flock` still blocking
    if pgrep -P "$pid" 2>/dev/null | while read -r c; do
         [[ "$(ps -o comm= -p "$c" 2>/dev/null)" == *flock* ]] && echo yes
       done | grep -q yes; then
      continue    # this one is WAITING, not holding
    fi
    # a holder has an active cargo/timeout/rustc somewhere under it
    if pgrep -P "$pid" >/dev/null 2>&1; then echo "$pid"; return; fi
  done
}

while :; do
  [[ -f "$STOPFILE" ]] && { log "STOP signalled — exiting"; rm -f "$STOPFILE" "$PIDFILE"; exit 0; }
  now=$(date +%s)

  # --- Rule 1: reap orphaned build processes (owner agent died -> ppid==1) ---
  while read -r pid ppid args; do
    [[ -z "${pid:-}" ]] && continue
    if [[ "$ppid" == "1" ]] && is_build_proc "$args"; then
      reap "$pid" "orphan ppid=1"
    fi
  done < <(ps -Ao pid=,ppid=,args= 2>/dev/null | grep -E "cargo|rustc|nextest|build\.sh" | grep -v grep)

  # --- Rule 2: hard wall-clock cap on build trees that escaped timeout(1) ---
  # Track cargo / cargo-nextest / timeout-wrapped-cargo roots by first-seen age.
  live=""
  while read -r pid args; do
    [[ -z "${pid:-}" ]] && continue
    case "$args" in
      *cargo-nextest*|*"timeout "*cargo*|*"gtimeout "*cargo*|*/bin/cargo\ *|*"cargo test"*)
        live="$live $pid"
        [[ -z "${SEEN[$pid]:-}" ]] && SEEN[$pid]=$now
        age=$(( now - SEEN[$pid] ))
        if (( age > HARD_CAP )); then
          reap "$pid" "hard-cap ${age}s > ${HARD_CAP}s (escaped build.sh timeout)"
          unset 'SEEN[$pid]'
        fi
        ;;
    esac
  done < <(ps -Ao pid=,args= 2>/dev/null | grep -E "cargo|nextest" | grep -v grep)
  # forget pids that are gone
  for p in "${!SEEN[@]}"; do kill -0 "$p" 2>/dev/null || unset 'SEEN[$p]'; done

  # --- Rule 3: lock holder making no compile progress = deadlocked -> reap ---
  holder="$(lock_holder)"
  if [[ -n "${holder:-}" ]]; then
    # progress proxy: a live rustc under the holder, OR its run.<pid>.log grew.
    prog=0
    pgrep -P "$holder" >/dev/null 2>&1 && \
      ps -Ao ppid=,comm= 2>/dev/null | grep -qE "rustc" && prog=1
    logf="$S/run.$holder.log"
    sz=$( [[ -f "$logf" ]] && wc -c < "$logf" 2>/dev/null || echo 0 )
    if [[ "${HOLD_LOGSIZE[$holder]:-}" != "$sz" ]]; then prog=1; HOLD_LOGSIZE[$holder]=$sz; fi
    if (( prog == 1 )); then
      PROG[$holder]=$now
    else
      [[ -z "${PROG[$holder]:-}" ]] && PROG[$holder]=$now
      stuck=$(( now - PROG[$holder] ))
      if (( stuck > STUCK_SECS )); then
        reap "$holder" "lock-holder no compile progress ${stuck}s > ${STUCK_SECS}s"
        unset 'PROG[$holder]' 'HOLD_LOGSIZE[$holder]'
      fi
    fi
    # forget stale holder entries
    for p in "${!PROG[@]}"; do kill -0 "$p" 2>/dev/null || { unset 'PROG[$p]' 'HOLD_LOGSIZE[$p]'; }; done
  fi

  sleep "$INTERVAL"
done

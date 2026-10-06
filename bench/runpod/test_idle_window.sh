#!/usr/bin/env bash
# rp-run's idle check counts only the command's GPU samples (#2951): a pod whose setup idled its GPU
# for 881 s (a checkpoint restored from the Mac) is not stopped as idle one minute into a command
# that keeps the GPU busy, and a command that idles the GPU for 15 minutes still is. Takes rp-run's
# own IDLE_MEAN and idle_since, and a synthetic gpu-util.log: 0% every 10 s from the pod's creation
# (t = 0) to the command's start (t = 881), then 95% (or 0% for an idle command) to t = 1781.
#
#   bash bench/runpod/test_idle_window.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
eval "$(grep -E "^IDLE_MEAN='" "$here/rp-run")"
eval "$(sed -n '/^idle_since() {/p' "$here/rp-run")"
[ -n "${IDLE_MEAN:-}" ] && declare -F idle_since > /dev/null || { echo "FAIL: IDLE_MEAN or idle_since not found in rp-run"; exit 1; }
log=$(mktemp); trap 'rm -f "$log" "$log.idle"' EXIT
start=881
for (( t = 0; t <= 1781; t += 10 )); do
    if (( t < start )); then echo "$t 0"; else echo "$t 95"; fi
done > "$log"
for (( t = 0; t <= 1781; t += 10 )); do echo "$t 0"; done > "$log.idle"
# The pod's log as it stands at time NOW: its samples up to NOW.
upto() { awk -v n="$2" '$1 <= n' "$1"; }
mean() { upto "$1" "$2" | awk -v t="$(idle_since "$start" "$2")" "$IDLE_MEAN"; }
fail=0
# The old window, the last 900 s whatever the command's start, one minute into the command.
old=$(upto "$log" $(( start + 60 )) | awk -v t=$(( start + 60 - 900 )) "$IDLE_MEAN")
(( old < 20 )) || { echo "FAIL: the fixture does not reproduce the old stop (old window mean $old%)"; fail=1; }
# One minute in: the window holds the command's samples alone.
m=$(mean "$log" $(( start + 60 )))
(( m >= 20 )) || { echo "FAIL: a busy command one minute in reads $m%"; fail=1; }
# Fifteen minutes in: a busy command passes, an idle one is caught.
m=$(mean "$log" $(( start + 900 )))
(( m >= 20 )) || { echo "FAIL: a busy command after 15 minutes reads $m%"; fail=1; }
m=$(mean "$log.idle" $(( start + 900 )))
(( m < 20 )) || { echo "FAIL: an idle command after 15 minutes reads $m%"; fail=1; }
(( fail == 0 )) && echo "PASS: the idle window starts at the command (old window $old% one minute in; busy command 95%, idle command 0% after 15 minutes)"
exit $fail

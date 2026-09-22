#!/usr/bin/env bash
# Interleaved anchored / closed-form A/B for gam#2928, and the phase receipt for gam#2979.
#
# The two arms alternate inside ONE job on ONE node. Separate jobs compare two machine
# states, not two code paths, and #2928's acceptance is a RATIO.
#
#   ./run_ab.sh <gam-binary> <out-root> [rows ...]
#
# Example (both of #2928's sizes, and #2979's):
#   ./run_ab.sh target/release/gam /scratch.global/$USER/gam2928 2000 2400 300000
#
# Writes <out-root>/n<rows>/rep<r>/<arm>.{log,time,model.json} and prints one TSV row per
# fit to stdout. `-v` puts the CLI at Debug, which is where the [OUTER] and [STAGE] lines
# live — the release build's equivalent of the retired harness's BENCH_LOG=1.
set -u -o pipefail

BIN=${1:?usage: run_ab.sh <gam-binary> <out-root> [rows ...]}
ROOT=${2:?usage: run_ab.sh <gam-binary> <out-root> [rows ...]}
shift 2
ROWS=${*:-2000}
REPS=${REPS:-3}
HERE=$(cd "$(dirname "$0")" && pwd)

printf 'rows\trep\tarm\trc\twall_s\tmax_rss_kb\n'
for rows in $ROWS; do
  data="$ROOT/n$rows"
  mkdir -p "$data"
  # One dataset per size, generated once, read by every arm and repeat: the ratio is
  # only a ratio if both arms see the same rows.
  if [ ! -f "$data/data.csv" ]; then
    python3 "$HERE/generate_truth2370.py" --rows "$rows" --out-dir "$data" >"$data/generate.log" 2>&1
  fi
  for rep in $(seq 1 "$REPS"); do
    for arm in closed anchored; do
      run="$data/rep$rep"
      mkdir -p "$run"
      start=$(date +%s.%N)
      /usr/bin/time -v -o "$run/$arm.time" \
        "$BIN" fit "$data/data.csv" \
          --request "$data/request_$arm.json" \
          --out "$run/$arm.model.json" \
          -v >"$run/$arm.log" 2>&1
      rc=$?
      end=$(date +%s.%N)
      wall=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.3f", b-a}')
      rss=$(awk '/Maximum resident set size/{print $NF}' "$run/$arm.time" 2>/dev/null)
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$rows" "$rep" "$arm" "$rc" "$wall" "${rss:-NA}"
    done
  done
done

cat >&2 <<'NOTE'

Read out of the logs, per fit:
  #2928 acceptance   anchored wall / closed wall, computed WITHIN a repeat, then across repeats.
                     The bar is 1.5. Report every wall, not just the mean, so a drifting node
                     shows up as spread rather than as a ratio.
  #2928 phase split  grep '\[STAGE\]' and '\[OUTER\]' for the pilot, seed screening, BFGS and the
                     continuation. The continuation's certified depth is the line
                     '#2661 anchored continuation certified at N steps'. N = 4 means the ladder
                     is at its floor; N of 8 or 16, or 'RefinementBudgetExhausted', means the
                     acceptance bar is the next thing to price.
  #2979 phase        run this whole script once per binary (unpatched main, then patched) and
                     compare. The count of exact Jeffreys completions is unchanged by the
                     tower-sharing commit, so any drop in the SparseTower4 share is the outer
                     derivative rebuild and nothing else. Take the profile separately:
                        perf record -F 199 -g -p <pid> -- sleep 60
                        perf report --stdio --no-children | head -40
  #2979 arming       count the Firth-armed probes. Near zero means the search runs unarmed on
                     this base and the order-4 consumer is absent from it.
NOTE

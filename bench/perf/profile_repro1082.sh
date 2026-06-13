#!/usr/bin/env bash
# eu-stack sampling profiler for the #1082 repro example.
# Usage: profile_repro1082.sh <case> <nsamples> <interval_s>
# Runs the example in the background, samples its main-thread + all-thread
# stacks with eu-stack, aggregates the most frequent leaf-and-caller frames.
set -u
CASE="${1:-negbin_syn}"
NS="${2:-80}"
INT="${3:-0.25}"
BIN=target/release/examples/repro1082_slow_quality
OUT=/tmp/prof_${CASE}
rm -rf "$OUT"; mkdir -p "$OUT"

"$BIN" "$CASE" > "$OUT/run.log" 2>&1 &
PID=$!
echo "[prof] pid=$PID case=$CASE nsamples=$NS interval=$INT"

i=0
while kill -0 "$PID" 2>/dev/null && [ "$i" -lt "$NS" ]; do
  eu-stack -p "$PID" 2>/dev/null > "$OUT/s_$i.txt" || true
  i=$((i+1))
  sleep "$INT"
done
wait "$PID" 2>/dev/null
echo "[prof] collected $i samples; run.log:"
tail -3 "$OUT/run.log"

# Aggregate: count distinct function symbols across all sampled frames
# (folded leaf frames). Strip addresses/args; keep symbol name.
echo "=== TOP FRAMES (symbol occurrence across all stacks/threads) ==="
cat "$OUT"/s_*.txt 2>/dev/null \
  | grep -oE '#[0-9]+ +0x[0-9a-f]+ [^ ].*' \
  | sed -E 's/#[0-9]+ +0x[0-9a-f]+ //; s/\(.*//; s/ +$//' \
  | sort | uniq -c | sort -rn | head -40

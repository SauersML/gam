#!/bin/bash
# Run a single plain-margslope fit and sample-profile it concurrently.
# Usage: ./run_with_sample.sh <N> <budget_sec> [<sample_sec>]
set -e
cd "$(dirname "$0")/../.."

N=${1:-200}
BUDGET=${2:-65}
SAMPLE_SEC=${3:-30}

RUNS="bench/margslope_plain/runs"
mkdir -p "$RUNS"

python3 bench/margslope_plain/gen.py "$N" "$RUNS/dch_$N.csv"

PC_COLS="PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10"
DUCHON="duchon($PC_COLS, centers=40, order=1, power=2, length_scale=1)"
MEAN="case ~ link(type=probit) + sex + $DUCHON"

./target/release/gam fit \
  "$RUNS/dch_$N.csv" \
  "$MEAN" \
  --logslope-formula "$DUCHON" \
  --z-column prs_z \
  --out "$RUNS/dch_$N.model" \
  > "$RUNS/dch_$N.log" 2>&1 &
PID=$!
echo "pid=$PID n=$N budget=${BUDGET}s sample=${SAMPLE_SEC}s"

# Warm-up
sleep 5

# Concurrent sample profile
echo ***REDACTED*** | sudo -S sample "$PID" "$SAMPLE_SEC" 10 -mayDie -file "$RUNS/dch_$N.sample.txt" >/dev/null 2>&1 || true

# Wait for the fit to finish or hit the budget
START=$(date +%s)
while kill -0 "$PID" 2>/dev/null; do
  ELAPSED=$(( $(date +%s) - START ))
  if (( ELAPSED >= BUDGET )); then
    kill -TERM "$PID" 2>/dev/null
    sleep 2
    kill -KILL "$PID" 2>/dev/null
    echo "killed after ${ELAPSED}s"
    break
  fi
  sleep 2
done
wait "$PID" 2>/dev/null || true
TOTAL=$(( $(date +%s) - START + 5 ))
echo "total ${TOTAL}s"
tail -20 "$RUNS/dch_$N.log"

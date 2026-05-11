#!/bin/bash
# Plain margslope WITH --scale-dimensions (anisotropic Duchon).
# Usage: ./run_scaledims.sh <N> <budget_sec>
set -e
cd "$(dirname "$0")/../.."

N=${1:-50}
BUDGET=${2:-32}

RUNS="bench/margslope_plain/runs"
mkdir -p "$RUNS"

python3 bench/margslope_plain/gen.py "$N" "$RUNS/dch_${N}_sd.csv"

PC_COLS="PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10"
DUCHON="duchon($PC_COLS, centers=40, order=1, power=2, length_scale=1)"
MEAN="case ~ link(type=probit) + sex + $DUCHON"

./target/release/gam fit \
  "$RUNS/dch_${N}_sd.csv" \
  "$MEAN" \
  --logslope-formula "$DUCHON" \
  --z-column prs_z \
  --scale-dimensions \
  --out "$RUNS/dch_${N}_sd.model" \
  > "$RUNS/dch_${N}_sd.log" 2>&1 &
PID=$!
echo "pid=$PID n=$N budget=${BUDGET}s scale_dims=on"

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
  sleep 1
done
wait "$PID" 2>/dev/null || true
TOTAL=$(( $(date +%s) - START ))
echo "total ${TOTAL}s"
tail -15 "$RUNS/dch_${N}_sd.log"

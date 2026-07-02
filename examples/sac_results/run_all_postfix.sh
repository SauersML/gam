#!/usr/bin/env bash
# Post-guard-fix rerun of all three SAC comparison arms on node2.
# Fire as ONE Heimdall job once S1-guards' patched .so is built into
# /models/sauers_build/target_fable (venv_fable picks it up via maturin develop).
#
# Thread config: OPENBLAS/OMP/MKL=1 to avoid the K=1 BLAS x rayon deadlock
# (gam#2074); RAYON handles parallelism. Writes all logs + JSON under
# /dev/shm/sauers_gpu/sac_w6/.
set -u
DIR=/dev/shm/sauers_gpu
OUT=$DIR/sac_w6
mkdir -p "$OUT"
PY=/models/sauers_build/venv_fable/bin/python
export RAYON_NUM_THREADS=16 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SAC_W6_OUT="$OUT" SAC_E2E_OUT="$OUT"
cd "$DIR"

echo "===== gamfit build in use ====="
$PY -c "import gamfit,inspect; print(gamfit.__file__)"

echo "===== EXP1: planted two-circles, joint K=2 vs SAC ====="
nice -19 ionice -c3 $PY -u sac_experiments.py --n 1500 --p 32 --n-iter 20 \
  --d-atom 1 --srp 0 --backfit 1 --isometry 1 > "$OUT/postfix_exp1.log" 2>&1
echo "exp1 rc=$?"

echo "===== EXP2: W6 OLMo K=8 via SAC (kill-test) ====="
nice -19 ionice -c3 $PY -u sac_w6_runner.py > "$OUT/postfix_exp2_w6.log" 2>&1
echo "exp2 rc=$?"

echo "===== EXP5: compose E2E N=3000 via SAC ====="
SAC_E2E_N=3000 nice -19 ionice -c3 $PY -u sac_compose_e2e.py > "$OUT/postfix_exp5_compose.log" 2>&1
echo "exp5 rc=$?"

echo "===== DONE; results JSON in $OUT ====="

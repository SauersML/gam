#!/bin/bash
#SBATCH --job-name=m979prof
#SBATCH --partition=msismall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/gam/.msi979_profile.log

set -x
source /projects/standard/hsiehph/sauer354/gam_env.sh
cd /projects/standard/hsiehph/sauer354/gam
git fetch origin main -q && git checkout origin/main 2>&1 | tail -1
git log --oneline -1

export CARGO_TARGET_DIR=/projects/standard/hsiehph/sauer354/gam/target_prof

echo "===== BUILD rigid profile example (release) ====="
cargo build --release --example large_scale_margslope_rigid_profile 2>&1 | tail -25
EXBIN=target_prof/release/examples/large_scale_margslope_rigid_profile
echo "EXBIN=$EXBIN"

echo "===== LAUNCH rigid profile UNWRAPPED (n=120000) in background, capture PID ====="
RUST_BACKTRACE=0 "$EXBIN" 120000 \
  > /projects/standard/hsiehph/sauer354/gam/.msi979_profile.fit.log 2>&1 &
FITPID=$!
echo "FITPID=$FITPID"

echo "===== eu-stack sampling loop (60 samples x 4s) ====="
SAMPLES=/projects/standard/hsiehph/sauer354/gam/.msi979_profile.stacks
: > "$SAMPLES"
for i in $(seq 1 60); do
  if ! kill -0 "$FITPID" 2>/dev/null; then echo "fit exited at sample $i"; break; fi
  echo "===SAMPLE $i $(date +%s)===" >> "$SAMPLES"
  tail -1 /projects/standard/hsiehph/sauer354/gam/.msi979_profile.fit.log >> "$SAMPLES" 2>/dev/null
  eu-stack -p "$FITPID" >> "$SAMPLES" 2>&1 || echo "eu-stack failed sample $i" >> "$SAMPLES"
  sleep 4
done

echo "===== wait for fit ====="
wait "$FITPID" 2>/dev/null
echo "===== fit log tail ====="
tail -30 /projects/standard/hsiehph/sauer354/gam/.msi979_profile.fit.log

echo "===== TOP gam frames (exclude parked/sync) ====="
grep -hoE '\bgam::[A-Za-z0-9_:]+' "$SAMPLES" \
  | grep -vE 'park|Condvar|spindle::barrier|rayon::|crossbeam' \
  | sort | uniq -c | sort -rn | head -40

echo "===== ALL frames (incl sync) ====="
grep -hoE '\b[A-Za-z0-9_]+::[A-Za-z0-9_:]+' "$SAMPLES" \
  | sort | uniq -c | sort -rn | head -30

echo "===== DONE ====="

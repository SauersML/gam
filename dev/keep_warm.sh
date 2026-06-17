#!/bin/bash
#SBATCH --job-name=gam_keepwarm
#SBATCH --partition=msismall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120g
#SBATCH --time=04:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/gam_keepwarm_%j.out
#
# keep_warm.sbatch — keeps a dedicated target dir (+ sccache if enabled) hot so
# the NEXT agent's `cargo test --no-run` is incremental, not a ~5-min cold build.
# Forward-only: fast-forwards origin/main, never `git reset --hard` (that busts
# fingerprints and re-cold-builds the whole fleet).
#   msi sub /projects/standard/hsiehph/sauer354/keep_warm.sbatch
set -uo pipefail
source /projects/standard/hsiehph/sauer354/gam_env.sh
source /projects/standard/hsiehph/sauer354/fast_iter.sh
# Dedicated warm target (NOT any agent's working dir — avoids the cargo lock).
export CARGO_TARGET_DIR=/scratch.global/sauer354/gam-target-warm
cd /projects/standard/hsiehph/sauer354/gam
[ -n "${RUSTC_WRAPPER:-}" ] && "$RUSTC_WRAPPER" --start-server 2>/dev/null || true

LAST=""; END=$(( $(date +%s) + 4*3600 ))
while [ "$(date +%s)" -lt "$END" ]; do
  git fetch -q origin main 2>/dev/null || true
  H=$(git rev-parse origin/main 2>/dev/null || echo "")
  if [ -n "$H" ] && [ "$H" != "$LAST" ]; then
    git merge --ff-only origin/main 2>/dev/null || true
    echo "[keep_warm] $(date) building $(git rev-parse --short HEAD)"
    /usr/bin/time -p cargo test --no-run -q 2>&1 | tail -3
    [ -n "${RUSTC_WRAPPER:-}" ] && "$RUSTC_WRAPPER" -s | grep -E "Cache hits rate|Compile requests" || true
    LAST="$H"
  fi
  sleep 90
done
echo "[keep_warm] 4h window done — resubmit to keep the target warm"

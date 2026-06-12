#!/bin/bash
#SBATCH --job-name=qg_accept
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --output=/projects/standard/hsiehph/sauer354/qg_accept.out
set -u
cd /projects/standard/hsiehph/sauer354/gam
source /projects/standard/hsiehph/sauer354/gam_env.sh 2>/dev/null
git pull --no-rebase origin main 2>&1 | tail -2
export CARGO_TARGET_DIR=/projects/standard/hsiehph/sauer354/target_qg
export RUSTFLAGS="--cap-lints=warn"
echo "QG_ACCEPT_START $(date +%T) HEAD=$(git rev-parse --short HEAD)"

echo "===== K1 RADIAL ====="
nice -n 5 cargo test -j 16 --test sae_manifold_k1_circle_radial_bias -- --nocapture
echo "K1_RC=$?"

echo "===== SWEEP ====="
nice -n 5 cargo test -j 16 --lib planted_circle_noise_scale_sweep_reaches_high_ev_with_dimensionless_rho_seed -- --nocapture
echo "SWEEP_RC=$?"

echo "===== TWO-CIRCLE ====="
nice -n 5 cargo test -j 16 --test sae_manifold_joint_two_circle_recovery -- --nocapture --test-threads=1
echo "TWOCIRCLE_RC=$?"

echo "QG_ACCEPT_DONE $(date +%T)"

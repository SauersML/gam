#!/bin/bash
#SBATCH --job-name=m740ver
#SBATCH --partition=msismall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/gam/.msi740_verify.log

set -x
source /projects/standard/hsiehph/sauer354/gam_env.sh
cd /projects/standard/hsiehph/sauer354/gam
git fetch origin main -q && git checkout origin/main 2>&1 | tail -1
git log --oneline -1

# #740 close-out gates:
# 1. independent BMS θ-HVP gate: ψ outer gradient == FD(value), and pure-ψ plus
#    mixed-ρψ HVP == centered FD(gradient) on the profiled joint-hyper path.
# 2. family contracted-ψ kernel gates: contracted ψψ == Σ_j α_j per-pair ψψ,
#    and generic penalty fold == per-pair ext_ext with live penalty terms.
# 3. solver contracted-hook gate: matrix-free operator == per-pair dense, mixed
#    ρψ matvec arm live, ψψ score injection live, materialization unavailable.
#
# Kept as an MSI manifest because the user-facing/local workflow for #740 must
# not run local compile/build/tests.
echo "===== #740 BMS profiled θ-HVP FD gate ====="
cargo test --lib profiled_theta_hvp_outer_hessian_matches_fd_of_gradient_psi_and_mixed -- --nocapture

echo "===== #740 BMS contracted ψ family gates ====="
cargo test --lib bernoulli_contracted_psi_second_order_matches_per_pair_contraction -- --nocapture
cargo test --lib bernoulli_contracted_psi_hook_matches_per_pair_with_penalty -- --nocapture

echo "===== #740 solver contracted-hook operator gate ====="
cargo test --lib operator_hessian_with_contracted_psi_hook_matches_per_pair_dense -- --nocapture

echo "===== DONE #740 VERIFY ====="

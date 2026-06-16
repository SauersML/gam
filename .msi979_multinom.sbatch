#!/bin/bash
#SBATCH --job-name=m979mnl
#SBATCH --partition=msismall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/gam/.msi979_multinom.log

set -x
source /projects/standard/hsiehph/sauer354/gam_env.sh
cd /projects/standard/hsiehph/sauer354/gam
git fetch origin main -q && git checkout origin/main 2>&1 | tail -1
git log --oneline -1

export CARGO_PROFILE_TEST_OPT_LEVEL=2
export CARGO_TARGET_DIR=/projects/standard/hsiehph/sauer354/gam/target_mnl

echo "===== BUILD quality crate (no-run, test profile) ====="
cargo test --no-run --test quality 2>&1 | tail -20
QBIN=$(ls -t target_mnl/debug/deps/quality-* | grep -v '\.d$' | head -1)
echo "QBIN=$QBIN"

# The gam-side inner-Newton crawl precedes the R reference call, so the hang is
# observable even if VGAM/nnet are missing. Run each penguins test under a hard
# 200s cap with INFO logging; capture the joint-Newton cycle trajectory + the
# terminal cycle count so we can see the separable crawl directly.
export RUST_LOG=info
for T in \
  quality_vs_nnet_multinom_penguins_species::gam_multinomial_classifies_penguin_species_at_least_as_well_as_nnet_on_real_data \
  quality_vs_vgam_multinomial_softmax::gam_multinomial_softmax_recovers_true_simplex_on_real_data ; do
  echo "===== RUN $T (cap 200s) ====="
  LOG=/projects/standard/hsiehph/sauer354/gam/.msi979_mnl.$(echo $T | tr ':' '_').log
  timeout 200 "$QBIN" --exact --nocapture "$T" > "$LOG" 2>&1
  RC=$?
  echo "rc=$RC for $T"
  echo "--- joint-Newton cycle count + terminal ---"
  grep -cE "PIRLS/joint-Newton convergence] cycle" "$LOG"
  grep -E "joint-Newton terminal|outer.*converg|seeds|startup validation|cycle +[0-9]+ \|" "$LOG" | tail -25
  echo "--- last 8 lines ---"
  tail -8 "$LOG"
done
echo "===== DONE ====="

#!/bin/bash
# Fresh-clone MSI verifier for the remaining #1082 timeout gates.
# Submit with:
#   /Users/user/msi-node/msi sub scripts/verify_1082_fresh.sh

#SBATCH -A hsiehph
#SBATCH -p msismall
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 2:30:00
#SBATCH -J g1082fresh
#SBATCH -o /projects/standard/hsiehph/sauer354/calib_logs/g1082fresh_%j.out
#SBATCH -e /projects/standard/hsiehph/sauer354/calib_logs/g1082fresh_%j.out

set -uo pipefail

PROJ=/projects/standard/hsiehph/sauer354
RUN_ROOT="${PROJ}/scratch/gam_verify_1082_${SLURM_JOB_ID}"
REPO="${RUN_ROOT}/gam"
TARGET="${PROJ}/gam-target-verify-1082"
LOCK="${PROJ}/gam-target-verify-1082.lock"

source "${PROJ}/gam_env.sh" 2>/dev/null || true
module load r-rich/4.4.2_msi1.2
export R_LIBS_USER="${PROJ}/Rlib/4.4"
source "${PROJ}/refpy/bin/activate"

rm -rf "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
git clone --single-branch --branch main https://github.com/SauersML/gam.git "${REPO}"
cd "${REPO}" || exit 2

HEAD="$(git rev-parse HEAD)"
REMOTE="$(git ls-remote origin refs/heads/main | awk '{print $1}')"
echo "=== #1082 fresh clone HEAD=${HEAD} origin/main=${REMOTE} ==="
if [ "${HEAD}" != "${REMOTE}" ]; then
  echo "=== FATAL: fresh clone is not at origin/main ==="
  exit 99
fi

export CARGO_TARGET_DIR="${TARGET}"
export CARGO_INCREMENTAL=1
export CARGO_PROFILE_DEV_DEBUG=0
export CARGO_PROFILE_TEST_DEBUG=0
export CARGO_PROFILE_DEV_SPLIT_DEBUGINFO=off

exec 9>"${LOCK}"
echo "[verify-1082] acquiring target lock ${LOCK}"
flock 9
echo "[verify-1082] running multinomial penguin + Cox marginal gates at ${HEAD}"
"${PROJ}/.cargo/bin/cargo" nextest run \
  gam_multinomial_classifies_penguin_species_at_least_as_well_as_nnet \
  gam_multinomial_classifies_penguin_species_at_least_as_well_as_nnet_on_real_data \
  gam_marginal_slope_heldout_concordance_matches_or_beats_lifelines_coxph 2>&1
RC=$?
END_REMOTE="$(git ls-remote origin refs/heads/main | awk '{print $1}')"
echo "=== #1082 end origin/main=${END_REMOTE} ==="
if [ "${HEAD}" != "${END_REMOTE}" ]; then
  echo "=== FATAL: origin/main moved during verifier ==="
  exit 99
fi
echo "=== EXIT=${RC} ==="
exit "${RC}"

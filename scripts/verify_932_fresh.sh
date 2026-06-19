#!/bin/bash
# Fresh-clone MSI verifier for #932 flex jet oracle coverage.
# Submit with:
#   /Users/user/msi-node/msi sub scripts/verify_932_fresh.sh

#SBATCH -A hsiehph
#SBATCH -p msismall
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 1:30:00
#SBATCH -J g932fresh
#SBATCH -o /projects/standard/hsiehph/sauer354/calib_logs/g932fresh_%j.out
#SBATCH -e /projects/standard/hsiehph/sauer354/calib_logs/g932fresh_%j.out

set -uo pipefail

PROJ=/projects/standard/hsiehph/sauer354
RUN_ROOT="${PROJ}/scratch/gam_verify_932_${SLURM_JOB_ID}"
REPO="${RUN_ROOT}/gam"
TARGET="${PROJ}/gam-target-verify-932"
LOCK="${PROJ}/gam-target-verify-932.lock"

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
echo "=== #932 fresh clone HEAD=${HEAD} origin/main=${REMOTE} ==="
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
echo "[verify-932] acquiring target lock ${LOCK}"
flock 9
FILTER='test(empirical_flex_score_warp_kernel_agrees_with_independent_fd_witness_all_channels) | test(empirical_flex_link_dev_kernel_agrees_with_independent_fd_witness_all_channels) | test(flex_contracted_tower_matches_independent_rigid_tower_and_catches_sign_flip) | test(flex_contracted_tower_matches_independent_fd_witness_nonzero_deviation)'
echo "[verify-932] running ${FILTER} at ${HEAD}"
"${PROJ}/.cargo/bin/cargo" nextest run -E "${FILTER}" 2>&1
RC=$?
END_REMOTE="$(git ls-remote origin refs/heads/main | awk '{print $1}')"
echo "=== #932 end origin/main=${END_REMOTE} ==="
if [ "${HEAD}" != "${END_REMOTE}" ]; then
  echo "=== FATAL: origin/main moved during verifier ==="
  exit 99
fi
echo "=== EXIT=${RC} ==="
exit "${RC}"

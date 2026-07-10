#!/bin/bash -l
# Fresh-origin CPU acceptance for the canonical steering dose (#2249) and
# fail-closed matched-null chart interpretation (#2250).
# Submit only through ~/msi-node/msi sub so authentication and project HOME are
# handled by the MSI wrapper.
#SBATCH --job-name=sae-2249-2250
#SBATCH --partition=msismall
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --output=/projects/standard/hsiehph/sauer354/sae-2249-2250-%j.out

set -uo pipefail

readonly RHOME=/projects/standard/hsiehph/sauer354
readonly SHA="$(git ls-remote https://github.com/SauersML/gam.git refs/heads/main | awk '{print $1}')"
readonly WORK="/scratch.global/sauer354/sae-2249-2250-${SHA}-${SLURM_JOB_ID}"
readonly TARGET="${RHOME}/targets/sae-2249-2250-${SHA}"

export HOME="${RHOME}"
export XDG_CACHE_HOME="${RHOME}/.cache"
export XDG_CONFIG_HOME="${RHOME}/.config"
export XDG_DATA_HOME="${RHOME}/.local/share"
export XDG_STATE_HOME="${RHOME}/.local/state"
export CARGO_HOME="${RHOME}/.cargo"
export RUSTUP_HOME="${RHOME}/.rustup"
export PATH="${CARGO_HOME}/bin:${PATH}"
export CARGO_BUILD_JOBS="${SLURM_CPUS_PER_TASK}"
export RUST_BACKTRACE=1

echo "host=$(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK} sha=${SHA}"
mkdir -p "${TARGET}"
trap 'rm -rf "${WORK}"' EXIT
git clone --filter=blob:none --no-checkout https://github.com/SauersML/gam.git "${WORK}"
git -C "${WORK}" checkout --detach "${SHA}"
ln -s "${TARGET}" "${WORK}/target"
cd "${WORK}"
module load openblas/0.3.27_gcc11.3.0_multiarch

failures=0
run_gate() {
    local label="$1"
    shift
    echo
    echo "===== BEGIN ${label} ====="
    "$@"
    local rc=$?
    echo "===== END ${label}: rc=${rc} ====="
    if [[ ${rc} -ne 0 ]]; then
        failures=$((failures + 1))
    fi
}

run_gate compile \
    ./build.sh test -p gam-sae -p gam-pyffi --no-run
run_gate issue2250_core \
    ./build.sh test -p gam-sae --test saebench_metrics -- --nocapture
run_gate issue2250_ffi \
    ./build.sh test -p gam-pyffi chart_interp_score_quotients_orientation_over_the_ffi_boundary -- --nocapture
run_gate issue2250_graph_decode \
    ./build.sh test -p gam-sae --lib nystrom_recovers_noisy_circle_angle -- --nocapture
run_gate issue2249_endpoint_dose \
    ./build.sh test --test sae predicted_nats_ -- --nocapture

echo
echo "acceptance_sha=${SHA} acceptance_failures=${failures}"
exit "${failures}"

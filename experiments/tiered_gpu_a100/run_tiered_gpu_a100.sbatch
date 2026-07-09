#!/bin/bash
# A100 GPU validation for the tiered Tier-1 block-gate routing lane (#2023 / #1017).
#
# Builds and runs `examples/tiered_gpu_scale.rs` at K=1e4 (2500 blocks x block-size 4,
# P=64) under `--gpu required`, which forces every admitted minibatch through the
# CUDA block-gate device entry (route_blocks_required) and turns a missing runtime,
# a below-break-even shape, or a device fault into a LOUD typed error rather than a
# silent CPU fallback. This is the run that proves the Tier-1 device lane actually
# executed on the A100 at qwen-scale width — the counterpart to the V100-validated
# resident single-step kernel.
#
# Fresh-clone runner (like dev/verify_1033_fast_fresh.sbatch): it clones origin/main
# into scratch and refuses to run unless HEAD == origin/main, so it never validates
# a stale shared checkout.
#
# Submit (coordinator, from an MSI login node):
#   /Users/user/msi-node/msi sub experiments/tiered_gpu_a100/run_tiered_gpu_a100.sbatch
# Override the width/mode with env, e.g.:
#   GPU_MODE=auto BLOCKS=5000 sbatch experiments/tiered_gpu_a100/run_tiered_gpu_a100.sbatch

#SBATCH -A hsiehph
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -J tiered_gpu_a100
#SBATCH -o /projects/standard/hsiehph/sauer354/tiered_gpu_a100_%j.out
#SBATCH -e /projects/standard/hsiehph/sauer354/tiered_gpu_a100_%j.out

set -uo pipefail

PROJ=/projects/standard/hsiehph/sauer354
RUN_ROOT="${PROJ}/scratch/tiered_gpu_a100_${SLURM_JOB_ID:-manual}"
REPO="${RUN_ROOT}/gam"

# Coordinator-specified build environment (#2023/#1017 A100 leg).
export CARGO_TARGET_DIR=/scratch.global/sauer354/gam-target-a100
LOCK="/scratch.global/sauer354/gam-target-a100.lock"

module load cuda/12.1.1

# The shared gam_env.sh exports a booby-trapped RUSTFLAGS (OpenBLAS saboteur) that
# breaks the gam build; never source it here, and clear RUSTFLAGS unconditionally
# so a login-profile export cannot leak in either.
unset RUSTFLAGS

# Runtime knobs.
GPU_MODE=${GPU_MODE:-required}     # required proves the device path actually ran
BLOCKS=${BLOCKS:-2500}             # K = BLOCKS * BLOCK_SIZE = 10000 by default
BLOCK_SIZE=${BLOCK_SIZE:-4}
P=${P:-64}
ROWS=${ROWS:-32768}
EPOCHS=${EPOCHS:-3}
export RUST_LOG=${RUST_LOG:-info}  # one-shot block-gate engagement verdict; debug = per-tile heartbeats

rm -rf "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
git clone --single-branch --branch main https://github.com/SauersML/gam.git "${REPO}"
cd "${REPO}" || exit 2

HEAD="$(git rev-parse HEAD)"
REMOTE="$(git ls-remote origin refs/heads/main | awk '{print $1}')"
echo "=== #2023/#1017 A100 tiered-GPU  clone HEAD=${HEAD}  origin/main=${REMOTE}  node=$(hostname)  $(date) ==="
if [ "${HEAD}" != "${REMOTE}" ]; then
  echo "=== FATAL: fresh clone is not at origin/main ==="
  exit 99
fi
nvidia-smi -L || { echo "=== FATAL: no GPU visible to the job ==="; exit 98; }

export CARGO_INCREMENTAL=1
export CARGO_PROFILE_DEV_DEBUG=0
export CARGO_PROFILE_TEST_DEBUG=0
CARGO="${PROJ}/.cargo/bin/cargo"
[ -x "${CARGO}" ] || CARGO=cargo

# Serialize on the shared A100 target dir so concurrent jobs don't corrupt it.
mkdir -p "$(dirname "${LOCK}")"
exec 9>"${LOCK}"
echo "[tiered-gpu-a100] acquiring target lock ${LOCK}"
flock 9

echo "[tiered-gpu-a100] building example tiered_gpu_scale (release) at ${HEAD}"
"${CARGO}" build --release -p gam-sae --example tiered_gpu_scale 2>&1 || { echo "=== FATAL: build failed ==="; exit 3; }

echo "[tiered-gpu-a100] running: --gpu ${GPU_MODE} --rows ${ROWS} --p ${P} --blocks ${BLOCKS} --block-size ${BLOCK_SIZE} --epochs ${EPOCHS}"
"${CARGO}" run --release -p gam-sae --example tiered_gpu_scale -- \
  --gpu "${GPU_MODE}" \
  --rows "${ROWS}" \
  --p "${P}" \
  --blocks "${BLOCKS}" \
  --block-size "${BLOCK_SIZE}" \
  --epochs "${EPOCHS}" 2>&1
RC=$?
echo "=== EXIT=${RC} ==="
exit "${RC}"

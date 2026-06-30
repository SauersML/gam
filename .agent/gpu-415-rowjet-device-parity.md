# #415 — GPU↔CPU survival row-jet device parity (PROVE IT ON THE V100)

## Context
Issue #415 asks for a parity-lock between the GPU row kernels and the CPU
marginal-slope / survival-flex formulas. Related: #1175 (FP-order &
transcendental parity).

Prior work (merged, commit 07ffd921e) added the **host-oracle "third leg"**:
a CPU transcription of the device `.cu` seeded-jet arithmetic, pinned to the
production CPU jet on EVERY box. Good — but it does NOT prove the actual NVRTC
device kernel compiles or runs. The `device_matches_cpu_when_available` test
only engages a real GPU on a GPU box, and the comments make clear the work was
done on a box WITHOUT a GPU.

## My box
- 8× Tesla V100-SXM2-32GB, compute_cap **7.0**, CUDA_VISIBLE_DEVICES=3.
- nvcc 12.1 / libnvrtc.so.12 present.

## Plan / gate
1. Build the gam-models GPU test target via `./build.sh`.
2. **Prove NVRTC compiles `survival_rowjet_kernel.cu` on sm_70** — fail loud if
   it declines (this is the recurring silent-CPU-fallback failure mode).
3. Run `device_matches_cpu_when_available` and confirm it actually exercised
   the GPU (nvidia-smi utilization moves), not the CPU fallback.
4. Add a **device-only** parity test that calls `survival_rigid_row_jets_device_only`
   and FAILS if the device path can't run — so a GPU box can never silently
   pass on the CPU fallback. Sweep edge regimes + both kernel variants
   (with/without t4). Document tolerance.
5. Keep CPU path green everywhere.

## Progress log
- (start) branch created, plan written.

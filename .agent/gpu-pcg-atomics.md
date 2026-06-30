# agent/gpu-pcg-atomics — GPU PCG kernel using double atomicAdd (arch-pinned)

Device: NVIDIA Tesla V100 SXM2 32GB (compute capability 7.0 → NVRTC `compute_70`).
Box: shared 8×V100; this agent pinned to CUDA_VISIBLE_DEVICES=7.

## Goal
The GPU device-resident PCG solve (`crates/gam-models/src/bms/gpu/device_pcg.rs`)
against the BMS-FLEX row-Hessian operator. The HVP operator (`q = H·p`) and the
arrow/Schur matvec accumulate with **double `atomicAdd`**, which requires NVRTC to
target ≥ `compute_60`. Verify:
1. Double atomics engage (kernels NVRTC-compile against compute_70, run on device).
2. CPU↔GPU parity on the PCG solve (dense-oracle / host-PCG comparison).
3. NVRTC-decline / arch mismatch FAILS LOUD, never silent CPU fallback.
4. Performance of the hot device loop on this V100.

## Status
- [ ] GPU driver currently DOWN on this box (nvidia kernel module half-loaded,
      nvidia-fabricmanager failed, cuInit -> error 100). Polling for recovery.
- [ ] Build crate, run arch-mapping unit tests (CPU-side, no device).
- [ ] Run device_pcg parity test once GPU is up.
- [ ] Harden / add fail-loud guards + parity coverage as needed.

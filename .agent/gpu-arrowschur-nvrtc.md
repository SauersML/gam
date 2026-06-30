# [agent][gpu] arrow-Schur NVRTC PCG device kernel — arch-pin + fail-loud

GPU box: 8× Tesla V100 SXM2 32GB (sm_70), CUDA 13.2, driver 595.71.05.
Device pinned: CUDA_VISIBLE_DEVICES=4.

## Focus
crates/gam-solve/src/gpu_kernels/arrow_schur{,_nvrtc}.rs — the fused Layer D+E
NVRTC arrow-Schur Newton kernel + the device PCG. Ensure:
1. NVRTC compiles for THIS arch (sm_70) — not the NVRTC default (<sm_60).
2. The device path actually RUNS on the GPU (not silent CPU/degrade).
3. CPU↔GPU parity vs the dense reference oracle.
4. NVRTC-decline FAILS LOUD.

## Findings (in progress)
- `fused_module_for` (arrow_schur.rs:1889) compiles via BARE `cudarc::nvrtc::compile_ptx`,
  NOT the arch-pinned `compile_ptx_arch` / `PtxModuleCache`. device_cache.rs:129-139
  explicitly warns bare compile_ptx defaults NVRTC below sm_60.
- Dispatch (arrow_schur.rs:157-171) treats a fused NVRTC compile failure
  (SchurFactorFailed) as "fall through to unfused" — silently degrades the fused path.

## Plan
- Route the fused module compile through the arch-pinned options.
- Audit fail-loud semantics on NVRTC decline.
- CPU-verifiable parity tests now; real device parity once driver is up.

## Blocker
nvidia kernel module not loaded at start (cuInit->100 NO_DEVICE); polling for recovery.

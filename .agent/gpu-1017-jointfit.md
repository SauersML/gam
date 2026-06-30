# GPU #1017 — device-resident SAE joint fit

Agent: gpu-1017-jointfit (GPU device 1, Tesla V100 SXM2, sm_70, CUDA 13.2)

## Goal
Close the measured 1e4-1e6x hardware gap by running the SAE joint fit on the GPU.
Make the device-resident inner Newton / reduced-Schur matvec actually engage on
this GPU, prove CPU<->GPU parity, fail loud on NVRTC/arch decline.

## Box state at start
- 8x Tesla V100 SXM2 32GB (sm_70). CUDA 13.2 toolkit present.
- nvidia kernel module NOT loaded at boot (fabricmanager NV_ERR_OPERATING_SYSTEM).
  cloud-init final stage still running at t+11min. Polling for driver.

## Plan
- [ ] Confirm GPU/driver comes up; nvidia-smi works.
- [ ] Locate the production joint-fit inner solve dispatch + device-resident path.
- [ ] Build via ./build.sh; run targeted GPU SAE tests; confirm device engages (util>0).
- [ ] CPU<->GPU parity on the inner solve / reduced-Schur matvec.
- [ ] Fail-loud guard on NVRTC/arch decline (no silent CPU fallback in device path).
- [ ] Perf: measure device vs CPU on qwen-shape inner solve.

## Resume run 2026-06-30 (gpu-1017-jointfit, V100 dev 1, sm_70, CUDA 12.4)

Driver IS up now (was down at PR open). Verified device path is REAL:
- `owed_1017_gpu` (release) PASSES in 51.2s. GPU monitor showed **51% util peak,
  770 MiB resident** during the run — device genuinely engaged, NOT a CPU fallback.
- Wide-border k=5120 GPU vs CPU dense-reference parity < 1e-6 (asserted in-test).
- Device-resident `color_arm_fixture().device_fit` converges on device (accepted>=1).
- The Phase-1 call-site re-keys (maybe_inject_gpu_schur_matvec /
  try_device_arrow_direct_sae_pcg → reduced_schur_matvec_should_offload) ARE present
  in newton_step.rs (the "documented not edited" note from the issue is now closed).

Next: run sae_perf_harness color/qwen to capture device vs CPU speedup numbers;
extend GPU test coverage and perf gate.

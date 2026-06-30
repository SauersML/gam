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

### BREAKTHROUGH — framed device PCG fixed + on-GPU #1551 gate committed

The framed device PCG "fault" was never in the kernel. Root cause: the test
FIXTURES built an ASYMMETRIC reduced-Schur operator (cross frame blocks
`g_{ij}`/`g_{ji}` sampled independently; diagonal `g_{kk}` a full random matrix).
`S` is a Hessian → symmetric by construction; the asymmetry made the lower-
triangle Cholesky reference an INVALID oracle (it solved the symmetrised system,
own residual 0.1) while the device PCG converged correctly against the full
operator (op-resid 3.6e-12). V100 evidence:
- `framed_sae_device_pcg_matches_cpu_when_cuda_admits` PASSES (322 MiB resident).
- `device_resident_pcg_matches_cpu_reference_when_cuda_admits` (full-B) PASSES.
- **`sae_direct_mode_device_engages_on_gpu_1551` (NEW committed test) PASSES**:
  production Direct SAE solve sets `used_device_arrow=true`; device step vs CPU
  dense-joint reference max|Δt|=4.2e-16, max|Δβ|=5.9e-14; log-det finite.

Fixes: symmetric fixtures (`g_{kk}` symmetrised, `g_{ji}=g_{ij}ᵀ`) in both the
in-crate test and tests/owed_1551; explicit `max|S-Sᵀ|` symmetry guard before the
Cholesky oracle; re-enabled the on-GPU engagement gate as a real `#[test]`.

Next: run sae_perf_harness color/qwen to capture device vs CPU speedup numbers;
address the dense-Schur-for-logdet path at large k (line ~501 newton_step.rs);
extend GPU test coverage and perf gate.

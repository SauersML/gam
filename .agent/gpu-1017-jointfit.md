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

# [agent][gpu] #1026 large-linear-SAE reconstruction parity — GPU lane

Agent: gpu-1026-parity (GPU device 2, Tesla V100 SXM2, sm_70)

## Box status at start (2026-06-30)
- NVIDIA kernel driver NOT loaded box-wide: `nvidia-smi` => "couldn't communicate
  with the NVIDIA driver"; NVML error 9; `nvidia-fabricmanager.service` FAILED;
  `modprobe nvidia` => "Operation not permitted" (no root, no passwordless sudo).
- 8x Tesla V100 SXM2 32GB present on PCI; CUDA 13.2 toolkit + NVRTC installed.
- Driver outage affects ALL GPU agents on this box; unfixable without root.

## Plan (CPU-verifiable + GPU-ready)
Because the device is genuinely unavailable, this is the ideal condition to PROVE
the charter's central guarantee: GPU dispatch must FAIL LOUD when the device path
cannot run, never silently fall back to CPU and report success.

1. Audit GpuMode::Required / Force dispatch on the SAE device-resident solver path:
   confirm device-unavailable => structured error, not silent CPU.
2. Add a guard test that asserts fail-loud under Required when the device is absent.
3. CPU<->GPU parity scaffolding for the large-K SAE reconstruction path, runnable
   the moment the driver returns.

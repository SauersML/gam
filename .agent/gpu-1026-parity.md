# [agent][gpu] #1026 large-linear-SAE reconstruction parity — GPU lane

Agent: gpu-1026-parity (GPU device 2, Tesla V100 SXM2, sm_70)

## Box status
- Start of 2026-06-30: NVIDIA kernel driver was NOT loaded box-wide (`nvidia-smi`
  => "couldn't communicate with the NVIDIA driver"; NVML error 9). That outage
  later cleared on this box; the V100 (device 2) is now live and every parity
  test in this PR ran ON the device (NVRTC compiled for sm_70, GPU memory moved).
- 8x Tesla V100 SXM2 32GB on PCI; CUDA 13.2 toolkit + NVRTC.

## Deliverable (PR #1674) — DONE, all green on V100
The #1026 failure was that the SAE GPU kernels were dead code (zero callers), so
the device never engaged and "GPU" runs silently fell back to CPU. This PR makes
the GPU path real, provable, and load-bearing, with CPU<->GPU BIT-IDENTICAL
parity as the gate.

1. `sparse_dict/scoring_gpu.rs` — NVRTC `sparse_dict_score_block` kernel for the
   collapsed-linear-lane router. Forces `__fmul_rn`/`__fadd_rn` (no FMA
   contraction) in ascending-c order => score block bit-identical to the CPU
   `acc += x·d` reference. Fail-loud `score_block_required` /
   `route_minibatch_required` honour `GpuMode` (Required errors, never silently
   degrades; Auto uses device when admitted; Off=CPU) and return the
   `ScoreBlockPath` actually taken. Auto-derived at runtime via `gam_gpu::gpu_mode()`
   + `DEVICE_SCORE_BLOCK_MIN_ELEMS` break-even — NOT a cargo feature.
2. Router K-tiles its device launches (cap `GPU_ROUTE_TILE_ELEMS`), so peak score
   memory is `m × tile`, independent of K — the lane's no-`N×K` discipline holds
   on the device (K=32k route peaks at 344 MiB, CUDA-context-dominated).
3. Wired into the real fit (`update.rs` route loop, `GpuMode::Auto`); per-row CPU
   `top_s_online` is the universal oracle/fallback. Fit result is path-invariant.
4. `gpu_kernels/sae_rowjet.rs` — fail-loud `sae_row_jets_softmax_required` +
   `SaeRowJetPath`, turning the previously dead softmax row-jet kernel provable.

## Parity proven on V100 (all assert `*Path::Device` under `Required`)
- rowjet K-sweep K in 1..=16, CPU==GPU <=1e-9.
- score-block 256x4096 bit-identical; route 512x4096 support bit-identical.
- route at K=32,768 (issue headline width, 4 device launches) bit-identical.
- real `fit_sparse_dictionary` K=4096 routes on GPU above break-even, reproducible.

## Notes for reviewers / future runs
- 16 full-suite FAILED tests are PRE-EXISTING borderline FD anchors under
  `manifold::`/`structure_harvest::` (byte-identical to main; fail in isolation
  single-threaded) — NOT regressions from this purely-additive diff.
- `debug_assert*!` is build-scanner banned under `src/`; run a `./build.sh test
  --test <name>` (not just `-p gam-sae --lib`) to trip the scanner before green.

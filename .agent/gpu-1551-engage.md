# #1551 — device-resident SAE solver engagement (+ #1209 honest routing)

Agent: `gpu-1551-engage` on Tesla V100-SXM2-32GB (sm_70), CUDA 12.4, device 0.

## State of the issue (from comments)
- NVRTC arch-pin (`--gpu-architecture` from device CC) landed → SAE PCG kernels now
  COMPILE on real GPUs (root cause of historic 0% GPU).
- Direct-mode device routing (`try_device_arrow_direct_sae_pcg`) landed → production
  SAE inner solve routes to device.
- `#1209` honest-routing diagnostics landed (`injected_host_procedural_matvec`).
- OPEN BLOCKER: the **framed** SAE device PCG (`G⊗W_ij`) computes a WRONG operator on
  hardware. CPU dense reduced-β PCG parity PASSES; framed FAILS (max_rel≈0.91,
  ‖S_cpu·x_dev−rhs‖≈18.5 vs CPU's 0.10). Structural marshalling/launch bug.

## My GPU is live
`cuInit=0 count=1` (peer's `cuInit→100` was transient). V100 sm_70 has double atomicAdd.

## Plan
1. Build via `./build.sh`; run the stage-isolating triage seam + the two parity tests
   ON the V100. Confirm which kernel stage diverges (per-component S·e_j diff).
2. Fix the framed-matvec structural bug; prove device==CPU ≤1e-9.
3. Keep CPU path bit-identical; fail loud on device decline.
4. Promote the recorded on-GPU assertion to a live `#[test]`.

## Log
- (start) branch + PR opened; verifying GPU + reading kernels.

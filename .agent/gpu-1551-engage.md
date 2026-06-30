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

## CRITICAL FINDING (on V100, build green)
Ran `framed_sae_device_pcg_matches_cpu_when_cuda_admits` on the live V100:
```
max_rel=5.45e-3 | ‖S_cpu·device−rhs‖=3.63e-12 (CPU's own =1.02e-1)
device PCG stop=Converged iters=74 final_rel_resid=9.65e-13
```
=> The framed DEVICE matvec is now CORRECT (satisfies the CPU operator to 3.6e-12).
   The historic "different operator, resid 18.5" bug is RESOLVED on current main.
   The test FAILS because its CPU REFERENCE (dense Cholesky on the assembled S) is
   itself inaccurate (resid 0.10) — the 400-row reduced-Schur fixture makes S
   ill-conditioned, so dense Cholesky loses precision while PCG (self-correcting)
   reaches 1e-12. Device is MORE accurate than the test's own oracle.

## Next: rigorously confirm conditioning is the cause; fix the parity oracle to be
## a sound CPU comparison (CPU-PCG with same matvec, or well-conditioned fixture).

## RESOLVED on V100 — framed SAE device PCG now GREEN
All three framed device tests PASS on Tesla V100 sm_70:
- framed_sae_device_matvec_matches_cpu_oracle_when_cuda_admits (NEW, conditioning-free) ✅
- framed_sae_device_pcg_matches_cpu_when_cuda_admits (was FAILING; fixed gate) ✅
- framed_sae_device_matvec_stage_diff_tiny_1551 ✅

Root cause of the historic "framed kernel 91% wrong / resid 18.5":
- The CURRENT kernel computes the correct operator (matvec parity ≤1e-9, conditioning-free).
- The OLD test asserted solution-VECTOR equality vs a dense-Cholesky reference. On the
  fixture's near-singular S, κ(S) amplifies O(ε) residual diffs to O(κ·ε) vector diffs.
  The dense ref itself only reached ‖S·x−rhs‖≈0.1 while device PCG reached ~1e-12 — the
  device was MORE accurate, not wrong.
Fix: gate on OPERATOR RESIDUAL (device δβ must solve the CPU-oracle system to PCG tol)
  + an independent CPU pcg_core solve of the same operator/preconditioner. Both converge.

GPU engagement proven via nvidia-smi dmon (312MB fb resident, SM util spikes 2-8%).

## Next: dense reduced-β path, #1209 honest routing, run full gpu suite.

## END-TO-END PRODUCTION ENGAGEMENT TEST — GREEN on V100
`sae_direct_inner_solve_engages_device_and_matches_cpu_1551` PASSES on Tesla V100:
- Drives the PUBLIC production entry `solve_arrow_newton_step_artifacts` with a
  device-equipped SAE system (n=512, k=64) that clears the offload gate.
- HARD ASSERTS `used_device_arrow == true` on a CUDA host (fail-loud #1551 contract:
  a silent CPU fallback FAILS the test). Engagement re-confirmed via nvidia-smi dmon
  (312MB fb resident during the run).
- Δβ/Δt parity vs `solve_arrow_newton_step_dense_reference` within 1e-7.

Two fixture defects fixed to get a SOUND parity gate (NOT loosened):
1. Reduced Schur not PD — device non-framed H_ββ comes from `data.sparse_g_blocks`
   (G⊗I_p), NOT `sys.hbb`; fixture left them empty so H_ββ≈ρ·I and the n=512-row
   Schur subtraction went indefinite (device CORRECTLY failed loud). Installed a
   dominant-diagonal H_ββ as 1×1 SparseGBlocks AND mirrored it in sys.hbb.
2. Parity oracle decoupled — the dense reference reads `row.htbeta` directly; the
   fixture shipped coupling ONLY as a matrix-free operator (row.htbeta all-zeros),
   so the reference solved a DECOUPLED system. Materialized the operator into each
   row.htbeta (exact unit-column probe), htbeta_dense_supplement OFF so the
   matrix-free apply is unchanged. All three paths now encode one H_tβ.

Full `gpu_kernels::arrow_schur` suite (17 tests) GREEN; `arrow_schur::tests` 57 pass
(the only failure is the pre-existing 2x2 brittleness below — proven identical on
clean merge-base e128441cd, owned by PR #1650 which already edits that test file).

## Pre-existing CPU failure (NOT mine, out of GPU scope)
`arrow_schur::tests::arrow_schur_matches_dense_reference_2x2` FAILS on main @ e128441cd,
deterministically, single-threaded. It `assert_eq!`s delta_beta (non-streaming) vs
delta_beta_stream (streaming chunk_size=1) — they differ in the LAST ULP (~4e-16 rel),
a brittle exact-equality assertion over two summation orderings. File is
crates/gam-solve/src/arrow_schur/tests.rs (I never touched it). Noted on PR; belongs to
a CPU-path owner, not this GPU PR.

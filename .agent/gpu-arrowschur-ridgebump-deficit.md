# [agent][gpu] Fused/batched arrow-Schur ridge bump never scales with the deficit

GPU device 6 (Tesla V100-SXM2-32GB, CC 7.0, `compute_70`). Builds on merged PR #1691.

## The bug (pre-existing GPU correctness defect — found while verifying #1691)

When a per-row latent block `H_tt + ridge_t·I` is not positive definite, the
device arrow-Schur paths return `ArrowSchurGpuFailure::RidgeBumpRequired{row,
bump}` so the outer LM escalation can retry at `ridge_t + bump`. But the
suggested `bump` was computed as `scale · |pivot| · √ε · 1024`, where `pivot`
is the cuSOLVER POTRF `info` (or the NVRTC kernel status code) — a **1-based
pivot ROW INDEX, not the pivot magnitude**. So a block that is indefinite by
`O(1)` (e.g. `H_tt = -I`, λ_min = −1, needs ridge > 1) yields the same
`bump ≈ √ε·1024 ≈ 1.5e-5` as a block indefinite by `O(√ε)`. The outer
escalation (bounded geometric doublings) can never lift a strongly indefinite
block out of the negative regime → the solve fails to recover.

Surfaced by the V100 validation test
`arrow_schur_gpu_ridge_bump_required_on_non_pd_row_recovers_after_bump`
(fails on main too — confirmed by checkout).

## Fix (analytic, no finite differences — SPEC-clean)

New shared helper `ridge_bump_to_make_pd(htt, ridge_t)` sizes the bump from the
block's own entries via the **Gershgorin circle theorem**:
`λ_min(A) ≥ g := min_i (A_ii − Σ_{j≠i}|A_ij|)`. Adding `t·I` shifts all
eigenvalues up by `t`, so `A + (ridge_t + bump)·I` is PD as soon as
`bump > -(g + ridge_t)`. Return `max(0, -(g+ridge_t)) + margin` where
`margin = scale·√ε·1024` (the same rounding headroom as before). One retry now
recovers a strongly indefinite block — matching the CPU oracle
(`arrow_schur/factorization.rs`), which lifts the per-row ridge "just enough"
internally. A column-major-slice variant covers the multi-GPU tile path.

## Producer sites updated (all in gpu_kernels/arrow_schur.rs)
- cuda::solve dense batched POTRF (info index)            → ridge_bump_to_make_pd
- cuda::solve_resident / Layer-A POTRF (info index)       → ridge_bump_to_make_pd
- fused forward status (kernel status index) ×2           → ridge_bump_to_make_pd
- row-procedural host Cholesky                            → ridge_bump_to_make_pd
- SAE device PCG host Cholesky ×2                         → ridge_bump_to_make_pd
- multi-GPU tile batched POTRF (info index)               → ridge_bump_to_make_pd_colmajor

## Verification log (Tesla V100-SXM2-32GB, CC 7.0, CUDA_VISIBLE_DEVICES=6)
- Rebased onto fresh main (after peer fixed the #1696 env::var ban break, c54156bc2).
- `./build.sh` clean under `warnings = "deny"` (removed now-dead `RowSlot.diag_scale`).
- `arrow_schur_gpu_ridge_bump_required_on_non_pd_row_recovers_after_bump`: FAIL → PASS.
  - Root cause of the residual failure after the per-row bump: the deficit-aware
    bump correctly lifts the `-I` block PAST λ_min=-1, so it factors as a barely-PD
    `(bump-1)*I ~ 1.5e-5*I` block. Locally κ=1, but `Y_2=L_2^-1 B_2` is amplified
    ~256×, driving the REDUCED Schur `S_β` strongly indefinite → legitimate
    `SchurFactorFailed`. The dense CPU reference fails identically at that ridge.
  - Production `solve_with_lm_escalation_inner` treats BOTH RidgeBumpRequired and
    SchurFactorFailed as ridge-recoverable; the test's manual loop only handled the
    former. Fixed the test to mirror production and to assert dense-CPU parity on
    the failure path at every escalation step.
- Full V100 suite (minus slow hill_climb): 7/7 PASS — baseline, multi_size,
  ridge_escalation, log_det, dense_reference, ridge_bump_required, fused_layer_d.
- GPU engagement: `nvidia-smi dmon -s um -i 6` caught SM 3%, fb 86→312 MB on
  device 6 during the suite (idle baseline 1 MB) — device path genuinely ran, no
  silent CPU fallback (every test fail-louds on Unavailable with a runtime present).
- CPU↔GPU parity: every passing test asserts the device step matches the dense
  CPU reference to 1e-10; CPU path unchanged.
- PR: #1711.

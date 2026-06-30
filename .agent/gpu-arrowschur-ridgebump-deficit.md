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

## Verification log
- (pending build + V100 run)

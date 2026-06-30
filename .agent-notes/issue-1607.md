# Issue #1607 — 10 pre-existing gam-models test failures

## Plan
Prior fleet runs fixed 4 of the original 10. Remaining (per discussion):
- Cluster 2: binomial_location_scale_batched_gradient_matches_finite_difference (rel 5.5%, envelope logdet-trace Fisher-scoring)
- Cluster 3: binomial_location_scale_exact_newton_spatial_joint_hyper_returns_fullhessian (indefinite outer joint Hessian)
- Cluster 3-wiggle: binomial_location_scalewiggle_..._returns_fullhessian (inner active-set on threshold block)
- Cluster 4-binomial: binomial_location_scale_engine_matches_reference_flow (inner active-set)
- Cluster 4-gaussian: gaussian_location_scale_engine_matches_reference_flow (σ-floor s·b vs b semantics)
- Cluster 5: profiled_theta_hvp_outer_hessian_matches_fd_of_gradient_psi_and_mixed (2nd-order LAML β-response)

## Step 0: ground-truth re-measure on current main, then attack.

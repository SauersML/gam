# Issue #1607 — pre-existing gam-models test failures

## GROUND TRUTH on current main (2026-06-30, branch agent/issue-1607/families-test-cluster)
`cargo test -p gam-models --lib` of the 10 originally-named tests:

GREEN now (fixed by prior fleet runs):
- flex_primary_hessian_matches_central_fd_of_gradient (Cluster 1)
- arbiter_flex_hessian_h00_fd_step_scaling (Cluster 1)
- binomial_location_scale_termswith_matern_spatial_blocks_fit_finitely (Cluster 3)
- rigid_standard_normal_tower_path_matches_hand_chain_witness (Cluster 5)

STILL FAILING (5):
1. **binomial_location_scale_batched_gradient_matches_finite_difference** (Cluster 2)
   `batched=-4.8858e-1 fd=-4.6178e-1 rel=5.49e-2`. Root cause per discussion:
   envelope logdet-trace Fisher-scoring inconsistency (expected-info logdet deriv).
2. **binomial_location_scale_exact_newton_spatial_joint_hyper_returns_fullhessian** (Cluster 3-spatial)
   "should return a full [rho,psi] hessian" — materialize_dense returns None.
3. **binomial_location_scalewiggle_exact_newton_spatial_joint_hyper_returns_fullhessian** (Cluster 3-wiggle)
   "coupled exact-joint inner solve exited the joint Newton path before convergence".
4. **gaussian_location_scale_engine_matches_reference_flow** (Cluster 4-gaussian)
   block 1 coef 0: engine -2.46979 vs reference -2.46534. σ-floor s·b vs b (response_scale standardization).
5. **binomial_location_scale_engine_matches_reference_flow** (Cluster 4-binomial)
   **FLAKY** — passed 1/3 runs. Tiny divergences (rel ~7e-4, ~4e-4) on no-wiggle AND wiggle blocks.
   => nondeterminism / convergence-path difference. Binomial does NOT standardize, so this is
   a different root cause than gaussian.

Note: profiled_theta_hvp_outer_hessian_matches_fd... (Cluster 5) lives in tests/ integration, NOT gam-models lib.

## Attack plan (tractable first)
- (A) Cluster 4-binomial FLAKY: determinism bug — engine vs reference take different iteration paths. Investigate.
- (B) Cluster 4-gaussian: principled σ-floor decision — make reference & engine consistent.
- (C) Cluster 2: envelope logdet-trace term (risky, prior regressions).
- (D) Cluster 3: deep architectural.

## Issue #1607 cont. (wiggle separation + batched gradient)
Branch off main after #1658 merged. Spatial fullhessian fix already landed.

// Regression guard for the large-scale dense Duchon timeout fixes that
// landed in `src/solver/pirls.rs::pirls_soft_acceptance`,
// `src/solver/outer_strategy.rs::SEED_SCREENING_CASCADE_MULTIPLIERS`, and
// `OuterProblem::with_standard_gam_dimensions`.
//
// The four fixes act together so that a moderate-scale dense Bernoulli-logit
// GAM converges quickly and without grinding through 100+ inner P-IRLS
// iterations per outer evaluation. This test fits such a problem end-to-end
// and asserts:
//   (a) PIRLS reaches `Converged` or `StalledAtValidMinimum` (the recognised
//       valid-minimum exits — anything else means the soft-acceptance plateau
//       criteria silently regressed).
//   (b) The final inner P-IRLS iteration count is well below the 100-iter
//       budget — a regression that disabled per-iter soft-exit would push
//       this to the cap.
//   (c) Outer-iteration count is bounded — a regression in the
//       seed-screening cascade would force the planner to chew through full
//       inner solves at every cap stage and inflate this.
//   (d) Wall-clock stays under a generous ceiling that is still ~10x lower
//       than what an un-fixed run takes.
//   (e) Predicted probabilities track the true generating model on a held-
//       in metric (Brier score) — guards against a fit that "converges" to
//       the wrong place.
//
// The problem is intentionally smaller than a real large-scale fit (k_smoothing
// = 2, below the BFGS-min-k cutoff, so the gradient-only routing in
// `with_standard_gam_dimensions` is not triggered here). This is a smoke
// test for the per-iter soft-acceptance and seed-screening fixes; the
// gradient-only routing has its own unit tests in `outer_strategy.rs`.


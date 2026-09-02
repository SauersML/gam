// Regression guard for the large-scale dense Bernoulli-logit convergence
// fixes. Four mechanisms must all stay healthy:
//
//   (1) `pirls_soft_acceptance` per-iteration early-exit on a 2-iter
//       plateau streak (`src/solver/pirls.rs`).
//   (2) `SEED_SCREENING_CASCADE_MULTIPLIERS = [1, 4, 16]`
//       (`src/solver/outer_strategy.rs`).
//   (3,4) `OuterProblem::with_standard_gam_dimensions` auto-routing to
//       gradient-only when truthful Hessian-assembly cost is large.
//
// This test fits a moderate dense problem (n=2000, single smooth, k=8,
// k_smoothing=1) and checks that:
//   - PIRLS terminates at a recognised valid-minimum status.
//   - Inner PIRLS iteration count is far below the 100-iter cap.
//   - Outer optimization converges in a small number of iterations.
//   - Wall-clock is under a generous ceiling.
//   - Predicted η correlates with the true η (>= 0.85) AND mean-abs
//     deviation is small (<= 0.20).
//
// k_smoothing = 1 keeps this below the `k >= 4` cutoff for the auto
// gradient-only routing in `with_standard_gam_dimensions`, so this test
// directly guards mechanisms (1) and (2). A regression in either causes
// either a status mismatch, an inner-iter blowup to ~100, or a wall-clock
// timeout — any of which fails the test.


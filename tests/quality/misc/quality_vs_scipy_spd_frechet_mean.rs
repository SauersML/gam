//! End-to-end OBJECTIVE quality: gam's affine-invariant SPD exponential/
//! logarithm maps must produce a genuine Riemannian center of mass (the
//! Fréchet / Karcher mean of a set of SPD matrices).
//!
//! gam exposes `gam::SpdManifold` with `exp_map` / `log_map` for the
//! affine-invariant metric but provides **no** `frechet_mean` primitive. The
//! Fréchet mean of SPD matrices `{X_i}` is the unique minimizer of the
//! dispersion functional `V(Q) = (1/M) Σ_i d²(Q, X_i)` and, equivalently, the
//! unique point `P` whose Riemannian gradient vanishes: `Σ_i log_P(X_i) = 0`.
//! We compute it the only way the public API allows — the canonical
//! gradient-descent fixed point `P ← exp_P((1/M) Σ_i log_P(X_i))` — using
//! gam's `exp_map`/`log_map`.
//!
//! The pass/fail assertions are OBJECTIVE Fréchet-mean axioms, evaluated
//! entirely with gam's own maps and the affine-invariant metric tensor — never
//! "matches another tool's fitted output":
//!
//!   1. FIRST-ORDER OPTIMALITY (manifold axiom): the Riemannian gradient at
//!      gam's solution vanishes, i.e. the metric norm of the tangent mean
//!      `(1/M) Σ_i log_P(X_i)` at `P` is ≈ 0. A wrong `P^{1/2}` conjugation in
//!      the metric, or an exp/log pair that is not a true inverse couple, moves
//!      the fixed point and leaves a nonzero gradient.
//!   2. GLOBAL MINIMALITY (the functional it must minimize): gam's `V(P)` is
//!      strictly below `V` evaluated at every input sample and below `V` at the
//!      Euclidean arithmetic mean — proving `P` actually minimizes dispersion,
//!      not merely sits at a stationary point of the wrong functional.
//!
//! The independent NumPy re-implementation of the identical recursion is
//! retained only as a MATCH-OR-BEAT BASELINE on that same objective: gam's
//! dispersion `V(P_gam)` must be ≤ the reference's `V(P_ref)` (both distances
//! measured with gam's metric). It is no longer the pass gate; the axioms are.
//! `rel_l2` between the two centers is printed for context but never asserted.

// ===========================================================================
// REAL-DATA ARM
// ===========================================================================
//
// Dataset SOURCE: the classic Leptograpsus "crabs" morphometrics table
// (Campbell & Mahon 1974; distributed as `MASS::crabs` in R, vendored here at
// `bench/datasets/crabs.csv`). 200 rows, 5 continuous body measurements
// (FL, RW, CL, CW, BD in mm) and two 2-level factors `sp` (species B/O) and
// `sex` (M/F) — exactly 4 groups of 50.
//
// The SAME gam capability the synthetic arm proves (affine-invariant SPD
// exp/log → Riemannian center of mass) is exercised here on real covariance
// structure: each crab group's 5×5 sample covariance of the body measurements
// is a genuine SPD matrix, and the four group covariances live on the SPD
// manifold. We hold out half of every group, build the four 5×5 SPD covariances
// from the TRAIN halves, take gam's Fréchet (Karcher) mean of those four, and
// measure how well that train-only center predicts the four HELD-OUT group
// covariances under the affine-invariant metric.
//
// OBJECTIVE held-out metric (truth unknown on real data):
//   V_test(P) = (1/G) Σ_g d²(P, C_test_g)  — mean squared geodesic distance from
//   the center P to each group's TEST-half covariance, all distances via gam.
//
//   PRIMARY (absolute, tool-free bar): gam's train-only center must predict the
//   held-out covariances strictly better than the two naive baselines on this
//   SAME metric — below V_test at the Euclidean (entrywise) mean of the train
//   covariances and below V_test at every individual train-group covariance.
//   This is an honest generalization claim: the Riemannian center transfers to
//   unseen samples of the same groups.
//
//   BASELINE (match-or-beat): a scipy/NumPy re-implementation computes its own
//   Fréchet mean from the IDENTICAL train covariances; gam's held-out
//   V_test(P_gam) must be ≤ V_test(P_ref) + margin. The mature tool is a
//   baseline to match, never an output to copy.


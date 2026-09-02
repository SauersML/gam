//! End-to-end OBJECTIVE quality for the discrete-mixture rung of the topology
//! race (Object 3a / WP-C): gam's cross-class adjudicator
//! ([`adjudicate_predictive_race`]) must make the RIGHT structural call about
//! whether 2-D latent coordinates live on a CONTINUOUS CIRCLE or in a finite set
//! of DISCRETE CLUSTERS — and it must do so at least as well as the mature
//! cluster-count baseline (scikit-learn's `GaussianMixture` with BIC model
//! selection) on the SAME data.
//!
//! This is the headline test of the ladder: a `k`-component mixture with the
//! components placed on a ring can MIMIC a circle in-sample (its component means
//! sit on the circle), so a naive in-sample fit would happily pick "clusters"
//! for genuinely continuous circular data. The discriminator that breaks the tie
//! is HELD-OUT PREDICTIVE DENSITY AT INTERPOLATED COORDINATES: a real circle
//! places probability mass on the whole ring and so predicts points that fall
//! BETWEEN the training points, whereas a discrete mixture concentrates mass at
//! its `k` component centers and assigns little density to the gaps. gam's
//! adjudicator builds exactly that selection-time cross-validated held-out
//! log-density table and stacks over it, so it should resolve the two regimes
//! correctly.
//!
//! TWO PLANTED REGIMES at matched signal-to-noise:
//!
//!   (A) TRUE CONTINUOUS CIRCLE — points drawn uniformly in angle on a ring of
//!       radius `R` with isotropic Cartesian Gaussian jitter `σ`. The ground
//!       truth is a one-dimensional continuous manifold (S¹). The mixture rung,
//!       no matter its `k`, is the WRONG model class: it cannot put mass on the
//!       continuum between its centers. gam MUST select the smooth circle, NOT
//!       the mixture.
//!
//!   (B) TRUE k-CLUSTER DISCRETE MIXTURE — points drawn from `K_TRUE` well
//!       separated isotropic Gaussian blobs (NOT on a ring) with the same jitter
//!       `σ`. The ground truth is genuinely discrete. gam MUST select the mixture
//!       rung, and the in-class winner's order `k` must match the planted
//!       `K_TRUE` (and the sklearn-BIC selected `k`).
//!
//! OBJECTIVE METRICS ASSERTED (none is "gam == reference output"):
//!
//!   1. CLUSTER REGIME — STRUCTURE RECOVERY (PRIMARY). On regime (B) the
//!      cross-class verdict's headline winner is the discrete-mixture rung
//!      (`AutoTopologyKind::Mixture`), and the in-class mixture winner's order `k`
//!      equals the planted `K_TRUE`. This is truth recovery of the discrete
//!      structure, asserted against the planted DGP.
//!
//!   2. CLUSTER REGIME — MATCH-OR-BEAT sklearn (BASELINE). scikit-learn's
//!      `GaussianMixture` swept over the same `k`-ladder and selected by BIC on
//!      the SAME data must also land on `K_TRUE`; gam's recovered `k` matches the
//!      sklearn-BIC `k`. sklearn is demoted from "the answer" to "a mature
//!      baseline gam must match" on the objective cluster-count metric.
//!
//!   3. CIRCLE REGIME — CLASS DISCRIMINATION (HEADLINE). On regime (A) the
//!      cross-class verdict's headline winner is the SMOOTH CIRCLE
//!      (`AutoTopologyKind::Circle`), NOT the mixture — even though the mixture
//!      ladder is offered and a high-`k` ring of blobs can imitate the circle
//!      in-sample. The discriminator is the held-out interpolated predictive
//!      density the adjudicator builds internally.
//!
//!   4. INTERPOLATION DISCRIMINATOR (objective, tool-free). Directly on regime
//!      (A): at INTERPOLATED held-out angles (points on the true ring that are
//!      deliberately NOT in any training fold's blob centers) the continuous
//!      circle model assigns strictly higher mean log predictive density than the
//!      best discrete mixture refit on the same training rows. This is the
//!      mechanism behind metric 3, asserted on its own so the headline call is
//!      grounded in a real predictive-accuracy gap rather than a coincidence of
//!      the stacking optimizer.
//!
//! Per repo policy the mature tool (sklearn) is a MATCH-OR-BEAT baseline on the
//! objective metric; the test may legitimately FAIL (honest) — it is never to be
//! weakened to pass. All math is in Rust; Python is reached only through the
//! reference harness shell.

// ---------------------------------------------------------------------------
// Deterministic RNG (no clock, no rand crate): a small SplitMix64 so the planted
// data is a pure function of a fixed seed.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Planted data generators (fixed seeds).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Smooth-circle held-out density provider. The proper Cartesian model is a
// uniform latent circle convolved with isotropic 2-D Gaussian noise, fitted by
// the same production implementation used by the topology race and structured
// unions. It places mass on the whole ring and remains normalized and finite at
// its center.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// sklearn GaussianMixture + BIC reference (the mature cluster-count baseline).
// Shelled through the reference harness `run_python`. Sweeps the SAME ladder gam
// uses and returns the BIC-selected component count.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared race construction: build the cross-class candidate list (one smooth
// circle + the full mixture ladder), each carrying its BIC-form corroborating
// score and a per-fold held-out-density provider, then hand it to gam's
// adjudicator.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// TEST 1 — CLUSTER REGIME: gam selects the mixture rung and recovers K_TRUE,
// matching the sklearn-BIC baseline on the same data.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// TEST 2 — CIRCLE REGIME (HEADLINE): gam selects the smooth circle, NOT the
// mixture, on genuinely continuous circular data; and the interpolation
// discriminator (held-out density at the gaps) backs the call.
// ---------------------------------------------------------------------------

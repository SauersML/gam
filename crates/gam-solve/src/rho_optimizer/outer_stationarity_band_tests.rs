//! #2613: which `|V|` anchors the stationarity band.
//!
//! `opt` resolves a `GradientTolerance`'s relative components exactly once,
//! at run start, against the SEED cost. So any `rel_cost` gam delegates makes
//! the solver's own stationarity test a function of where the search was
//! handed rather than of the problem it is solving. On #2392's exponentially
//! stiff recovery the seeds of ONE fit span eighteen orders, and the run
//! reported
//!
//!     termination=gradient_tolerance(|g|=1.522998e-4 < 1.792397e8)
//!
//! at ρ = 30 — the wrong rail — against a threshold no gradient can fail.
//!
//! These tests pin the two properties that make that unwritable:
//!
//!   * [`outer_gradient_tolerance`] — the SOLVER's band — is a function of
//!     the declared problem and of nothing else, so every seed of one fit
//!     reaches the same verdict;
//!   * [`outer_stationarity_band_and_rung_at`] — the CERTIFICATE's band — is
//!     anchored at the point being judged, which is what mgcv `magic` means.
//!
//! Every number below is evaluated from #2392's criterion rather than
//! transcribed from the issue, so the fixture cannot drift away from the
//! defect it pins; the issue's printed values appear only as cross-checks.
use super::{
    OuterConfig, outer_cost_relative_tolerance, outer_gradient_tolerance,
    outer_stationarity_band_and_rung_at,
};

/// The band alone. These tests compare bands to each other; #2688 moved the
/// rung into the production return type on purpose, so the value-only form
/// lives here rather than beside the thing it was extracted from.
fn outer_stationarity_band_at(config: &OuterConfig, cost_at_point: f64) -> f64 {
    outer_stationarity_band_and_rung_at(config, cost_at_point).bound
}

/// #2392's gradient-only criterion `V(ρ) = A·(−q + ½q²)`, `q = e^{ρ★−ρ}`,
/// with gradient `V′(ρ) = A·q(1−q)`. These are the same `AMPLITUDE` and
/// `RHO_STAR` the `wrong_rail_pullback_recovers_gradient_only_objective_2392`
/// fixture builds its objective from.
const AMPLITUDE: f64 = 1.0e4;
const RHO_STAR: f64 = 12.0;
/// The generated lattice seed whose criterion poisoned the band.
const POISONED_SEED_RHO: f64 = 1.0;
/// A healthy seed of the same fit (the recovery's own pull-back neighbourhood).
const HEALTHY_SEED_RHO: f64 = 16.9;
/// The rail the poisoned band let the solver claim convergence on.
const WRONG_RAIL_RHO: f64 = 30.0;
/// The checkpoint the run actually stopped at, whose certificate bound
/// `4.998e-2` is the OTHER half of the issue's "two numbers, one formula".
const CHECKPOINT_RHO: f64 = 12.025_770_241_372_719;

fn criterion(rho: f64) -> f64 {
    let q = (RHO_STAR - rho).exp();
    AMPLITUDE * (-q + 0.5 * q * q)
}

fn gradient_norm(rho: f64) -> f64 {
    let q = (RHO_STAR - rho).exp();
    (AMPLITUDE * (q - q * q)).abs()
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    let relative = (actual - expected).abs() / expected.abs();
    assert!(
        relative <= 1.0e-6,
        "{what}: got {actual:.6e}, expected {expected:.6e} (relative {relative:.3e})",
    );
}

/// The `#2392` recovery's config: `OuterProblem::new(1)` declares no
/// objective scale, so `objective_scale` is `None` and `tolerance` is the
/// default the caller asked for.
fn recovery_config() -> OuterConfig {
    OuterConfig::default()
}

#[test]
fn the_fixture_reproduces_every_number_the_issue_printed() {
    // The seed cost that became the anchor, and the vacuous threshold it
    // produced under `rel_cost·(1 + |seed_cost|)`.
    let config = recovery_config();
    let poisoned_seed_cost = criterion(POISONED_SEED_RHO);
    assert_close(
        poisoned_seed_cost,
        1.792_396_548_924e13,
        "the lattice seed's criterion",
    );
    // The old solver band IS the certificate's formula evaluated at the
    // seed — which is exactly the defect, so the certificate helper
    // reproduces the removed rule when handed a trajectory point.
    let seed_anchored_band = outer_stationarity_band_at(&config, poisoned_seed_cost);
    assert_close(
        seed_anchored_band,
        1.792_397e8,
        "the seed-anchored solver threshold",
    );
    // The gradient at the wrong rail, which "passed" that threshold.
    assert_close(
        gradient_norm(WRONG_RAIL_RHO),
        1.522_998e-4,
        "the wrong-rail gradient norm",
    );
    // And the certificate's bound at the checkpoint the run stopped at:
    // the same formula, anchored eight orders lower, which is why the
    // disagreement read as a certificate bug.
    assert_close(
        outer_stationarity_band_at(&config, criterion(CHECKPOINT_RHO)),
        4.997_764e-2,
        "the certificate bound at the checkpoint",
    );
}

#[test]
fn the_solver_band_is_the_same_for_every_seed_of_one_fit() {
    let config = recovery_config();
    let tolerance = outer_gradient_tolerance(&config);
    // Nothing relative is delegated: `opt` would resolve it against the seed.
    assert!(
        tolerance.rel_cost.is_none() && tolerance.rel_initial_grad.is_none(),
        "the solver band must carry no component `opt` resolves at the seed",
    );
    // The seeds of this one fit span eighteen orders of criterion value...
    let spread = criterion(POISONED_SEED_RHO).abs() / criterion(HEALTHY_SEED_RHO).abs();
    assert!(
        spread > 1.0e10,
        "fixture must keep the seed spread that made the anchor matter: {spread:.3e}",
    );
    // ...and every one of them resolves to the identical threshold, which is
    // `abs` itself. Two seeds converging to the same optimum now reach the
    // same verdict.
    for rho in [
        POISONED_SEED_RHO,
        HEALTHY_SEED_RHO,
        WRONG_RAIL_RHO,
        CHECKPOINT_RHO,
        RHO_STAR,
    ] {
        assert_eq!(
            tolerance.threshold(criterion(rho), gradient_norm(rho)),
            tolerance.abs,
            "a seed at ρ={rho} must not move the solver's stationarity band",
        );
    }
}

#[test]
fn the_wrong_rail_no_longer_clears_the_solver_band() {
    let config = recovery_config();
    let wrong_rail_gradient = gradient_norm(WRONG_RAIL_RHO);
    // Under the removed rule the rail passed by twelve orders.
    let seed_anchored_band =
        outer_stationarity_band_at(&config, criterion(POISONED_SEED_RHO));
    assert!(
        wrong_rail_gradient < seed_anchored_band,
        "fixture must reproduce the vacuous pass this issue is about",
    );
    // With no declared scale the honest band is the absolute tolerance the
    // caller asked for — gam does not substitute a trajectory point for a
    // magnitude it does not know — and the rail fails it.
    let band = outer_gradient_tolerance(&config).abs;
    assert_eq!(band, config.tolerance);
    assert!(
        wrong_rail_gradient > band,
        "a non-stationary rail must fail the solver's band: |g|={wrong_rail_gradient:.6e} vs {band:.6e}",
    );
}

#[test]
fn a_declared_scale_preserves_the_band_a_criterion_at_that_scale_would_give() {
    // A REML/LAML score is a sum over `n` rows, so a fit that declares
    // `objective_scale = n` is declaring `|V| = O(n)`. The solver band must
    // therefore equal the point-anchored band of a criterion at that scale:
    // this is the property that makes the change magnitude-PRESERVING rather
    // than a deletion of the cost-relative term.
    let scale = 1_200.0;
    let config = OuterConfig {
        objective_scale: Some(scale),
        ..OuterConfig::default()
    };
    let band = outer_gradient_tolerance(&config).abs;
    assert_eq!(band, outer_stationarity_band_at(&config, -scale));
    assert_close(
        band,
        outer_cost_relative_tolerance(&config) * (1.0 + scale),
        "the declared-scale band",
    );
    // And it is emphatically NOT the arithmetic resolution floor alone.
    // Dropping `rel_cost` and leaning on `scale·√ε` leaves a band ~670×
    // tighter at the default tolerance, which no real fit can satisfy.
    let arithmetic_floor = config.tolerance.max(scale * f64::EPSILON.sqrt());
    assert!(
        band > arithmetic_floor * 100.0,
        "the band must not collapse onto the arithmetic floor: {band:.6e} vs {arithmetic_floor:.6e}",
    );
}

#[test]
fn the_certificate_band_follows_the_point_it_judges() {
    let config = recovery_config();
    let solver_band = outer_gradient_tolerance(&config).abs;
    // The certificate's band moves with the candidate optimum's criterion...
    let at_checkpoint = outer_stationarity_band_at(&config, criterion(CHECKPOINT_RHO));
    let at_optimum = outer_stationarity_band_at(&config, criterion(RHO_STAR));
    assert!(
        at_checkpoint > solver_band && at_optimum > solver_band,
        "a certificate at a criterion of magnitude ≫1 must exceed the bare tolerance",
    );
    // ...while the solver's does not move at all. Naming the two separately
    // is what makes "one formula, two anchors" impossible to write by
    // accident: neither function takes the other's anchor.
    assert_ne!(at_checkpoint, at_optimum);
    // A non-finite criterion carries no magnitude, so the certificate falls
    // back to the arithmetic floor rather than propagating a NaN bound.
    for cost in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(outer_stationarity_band_at(&config, cost), config.tolerance);
    }
}

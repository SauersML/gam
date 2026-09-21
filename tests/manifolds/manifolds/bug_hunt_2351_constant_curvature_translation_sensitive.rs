//! Bug hunt #2351: `response_geometry="constant_curvature"` curvature
//! estimation is not invariant to a rigid ambient translation of the response
//! data, even though translating a point cloud changes nothing about its
//! intrinsic shape or spread.
//!
//! ROOT CAUSE: `response_kappa_bounds` derives the hyperbolic search bound
//! `kappa_min` from `max_i ‖y_i‖²` — squared distance from the literal
//! ambient coordinate origin — while the spherical bound `kappa_max` is
//! correctly derived from `max_i ‖y_i − μ‖²`, distance from the data's own
//! centroid. For data of roughly constant norm from the origin (unit-
//! normalized embeddings being the textbook case — see this module's own
//! doc comment citing "unit-normalised OLMo activations"), `kappa_min`
//! collapses to nearly `-1` regardless of the data's true curvature, the
//! search rails there, and — because `railed_at_resolution_limit` only
//! fires on the upper (spherical) bound — the caller gets a fully-confident
//! (p≈0) verdict with no honesty flag. Moving the SAME cloud away from the
//! origin removes the artifact and flips the verdict.
//!
//! This test fits the same intrinsic point cloud at two different ambient
//! locations (an arbitrary rigid translation) and asserts the reported
//! curvature verdict and dimensionless `kappa_r2` invariant do not
//! contradict each other. It fails on current `main`: the untranslated cloud
//! reports a fully-confident `Hyperbolic` verdict (flatness p ≈ 0) while the
//! translated copy of the exact same cloud reports `Flat`
//! (`railed_at_resolution_limit = true`).

use gam::geometry::curvature_estimand::CurvatureVerdict;
use gam::geometry::fit_response_curvature;
use ndarray::Array2;

// Deterministic RNG (splitmix64 -> unit / gaussian), no external deps.
use gam::utils::splitmix64;
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// `n` unit-normalized isotropic Gaussian directions in `R^dim` — the
/// library's own documented `constant_curvature` example ("unit-normalised
/// activations"). Isotropic random directions carry no genuine curvature
/// signal at all, so a sound estimator should never return a maximally
/// confident (p -> 0) verdict of either sign for this cloud.
fn unit_normalized_cloud(n: usize, dim: usize, seed: u64) -> Array2<f64> {
    let mut st = seed;
    let mut out = Array2::<f64>::zeros((n, dim));
    for mut row in out.rows_mut() {
        let mut norm_sq = 0.0_f64;
        for j in 0..dim {
            let g = next_gauss(&mut st);
            row[j] = g;
            norm_sq += g * g;
        }
        let norm = norm_sq.sqrt();
        for j in 0..dim {
            row[j] /= norm;
        }
    }
    out
}

#[test]
fn constant_curvature_verdict_is_translation_invariant() {
    let dim = 4;
    let cloud = unit_normalized_cloud(500, dim, 0xC0FF_EE12_3456_7890);

    let at_origin = fit_response_curvature(cloud.view(), dim, 0.95)
        .expect("fit at the ambient origin");

    let shift = 10.0_f64;
    let translated = cloud.mapv(|v| v + shift);
    let away_from_origin = fit_response_curvature(translated.view(), dim, 0.95)
        .expect("fit translated far from the ambient origin");

    // A rigid ambient translation must not change the reported geometric
    // verdict for the SAME intrinsic cloud shape: both fits must agree on
    // whether the data is confidently spherical, confidently hyperbolic, or
    // unresolved/flat. On current `main` this fails: `at_origin` rails to
    // `kappa_hat ~ -1` (fully-confident Hyperbolic, p ~ 0) purely because
    // unit-normalized vectors sit at distance ~1 from the literal ambient
    // origin, while `away_from_origin` reports `Flat`
    // (`railed_at_resolution_limit = true`) for the identical cloud shape.
    assert_eq!(
        at_origin.profile_ci.verdict,
        away_from_origin.profile_ci.verdict,
        "curvature verdict flipped under a pure ambient translation: \
         at_origin kappa_hat={} verdict={:?} p={} railed={}; \
         away_from_origin kappa_hat={} verdict={:?} p={} railed={}",
        at_origin.kappa_hat,
        at_origin.profile_ci.verdict,
        at_origin.flatness.p_value,
        at_origin.railed_at_resolution_limit,
        away_from_origin.kappa_hat,
        away_from_origin.profile_ci.verdict,
        away_from_origin.flatness.p_value,
        away_from_origin.railed_at_resolution_limit,
    );

    // A confident verdict here is only honest if the likelihood itself
    // resolves it, not the chart edge. Mean-centred unit vectors are a thin
    // shell: every ‖z_i‖ ≈ 1. In the constant-curvature model that radial law
    // is real signal. Under a flat isotropic Gaussian the radii would follow a
    // chi law, while hyperbolic volume growth concentrates the geodesic radii,
    // so the flat member is rejected decisively (LR ≈ 1.8e3 on this seed).
    // The optimum is a stationary point strictly inside the chart
    // (κ̂/κ_min ≈ 0.9998), and the profile rises by ≈ 20 log-units between κ̂
    // and κ_min, far past the χ²₁ half-threshold 1.92. So the CI closes on
    // both sides inside the chart and neither rail's KKT condition binds
    // (#3629). The earlier contract flagged this optimum through a
    // `κ̂ ≤ 0.99·κ_min` proximity band that nothing derived.
    assert!(
        !at_origin.railed_at_resolution_limit
            && !at_origin.railed_at_hyperbolic_resolution_limit
            && !at_origin.profile_ci.lo_at_bound
            && !at_origin.profile_ci.hi_at_bound,
        "shell cloud's hyperbolic optimum should be a resolved interior estimate: \
         kappa_hat={} verdict={:?} p={} ci=[{}, {}] lo_at_bound={} hi_at_bound={} \
         railed={} railed_hyperbolic={}",
        at_origin.kappa_hat,
        at_origin.profile_ci.verdict,
        at_origin.flatness.p_value,
        at_origin.profile_ci.ci_lo,
        at_origin.profile_ci.ci_hi,
        at_origin.profile_ci.lo_at_bound,
        at_origin.profile_ci.hi_at_bound,
        at_origin.railed_at_resolution_limit,
        at_origin.railed_at_hyperbolic_resolution_limit,
    );
    // Both rail flags must ALSO be translation-invariant.
    assert_eq!(
        at_origin.railed_at_hyperbolic_resolution_limit,
        away_from_origin.railed_at_hyperbolic_resolution_limit,
        "hyperbolic rail flag flipped under a pure ambient translation"
    );

    // Sanity: the two fits should at least not both claim opposite signs
    // with a resolved CI (this is implied by the equality assert above, but
    // spelled out because it's the user-visible symptom from the issue).
    let contradicts = matches!(
        (
            at_origin.profile_ci.verdict,
            away_from_origin.profile_ci.verdict
        ),
        (CurvatureVerdict::Spherical, CurvatureVerdict::Hyperbolic)
            | (CurvatureVerdict::Hyperbolic, CurvatureVerdict::Spherical)
    );
    assert!(
        !contradicts,
        "translation flipped the curvature SIGN outright: {:?} vs {:?}",
        at_origin.profile_ci.verdict, away_from_origin.profile_ci.verdict
    );
}

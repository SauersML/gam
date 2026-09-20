//! Workflow detection gate for the O(n log n) multiresolution residual-cascade
//! fast path (#1032).
//!
//! The cascade is a DIFFERENT posterior from the Duchon/Matérn term it stands
//! in for (a multilevel Wendland frame, not the reduced-rank radial kernel), so
//! — unlike the 1-D spline scan, which silently swaps an identical posterior —
//! the cascade fast path must NEVER silently replace an eligible term. It is
//! allowed to fire ONLY on its structural signature: a single scattered
//! Gaussian-identity radial smooth (`duchon`/`matern`) over 2–3 coordinates,
//! and ONLY once `n` is past the derived dense-kernel cliff (the dense radial
//! basis saturated at its center cap). Below the cliff the dense path is both
//! exact-posterior and cheap, so the cascade must fall through and preserve the
//! user's chosen posterior.
//!
//! Detection is split into the pure structural predicate
//! (`residual_cascade_structural_signature`) and the size gate applied on top of
//! it (`residual_cascade_fast_path`). Below the cliff the full fast path is
//! `None` for every input, so the structural negatives are asserted against the
//! structural predicate directly (#3550): each negative differs from the
//! eligible positive control `y ~ duchon(x1, x2)` in exactly one guard, so
//! deleting that guard turns the negative red. The below-cliff duchon is the
//! single witness for the size gate. The positive (cliff-scale) route is
//! exercised by the in-test dense-kernel oracle in
//! `tests/residual_cascade_certification.rs`, which certifies the cascade
//! estimator itself; here we certify the SEAM never mis-fires.

use csv::StringRecord;
use gam::{
    FitConfig, FitRequest, FitResult, ResidualCascadeSignature, encode_recordswith_inferred_schema,
    fit_from_formula, fit_residual_cascade_from_formula, materialize, residual_cascade_fast_path,
    residual_cascade_structural_signature,
};

/// Deterministic scattered 2-D sample on the unit square with a smooth truth.
/// `counts` turns the response into a Poisson-valid count (non-negative
/// integer) with the same smooth log-mean, so a non-Gaussian family can be
/// materialized on the same design.
fn sample_2d_with(n: usize, counts: bool) -> gam::data::EncodedDataset {
    let golden = 0.618_033_988_749_894_9_f64;
    let sqrt2 = std::f64::consts::SQRT_2.fract();
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let a = ((i + 1) as f64 * golden).fract();
            let b = ((i + 1) as f64 * sqrt2).fract();
            let t = (2.0 * std::f64::consts::PI * a).sin() * (2.0 * std::f64::consts::PI * b).sin();
            let u = ((i + 3) as f64 * golden).fract();
            let y = if counts {
                // Deterministic quantile-style count around mean exp(1 + t).
                ((1.0 + t).exp() * 2.0 * u).floor()
            } else {
                t + (u - 0.5) * 0.1
            };
            StringRecord::from(vec![a.to_string(), b.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode dataset")
}

fn sample_2d(n: usize) -> gam::data::EncodedDataset {
    sample_2d_with(n, false)
}

fn gaussian_config() -> FitConfig {
    FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    }
}

/// Materialize `formula` and run the pure structural predicate on the
/// resulting standard request. Materialization must succeed: a refusal there
/// would mean the negative never reaches the guard it is meant to pin.
fn structural_signature(
    formula: &str,
    data: &gam::data::EncodedDataset,
    cfg: &FitConfig,
) -> Option<ResidualCascadeSignature> {
    let mat = materialize(formula, data, cfg).expect("materialize");
    let FitRequest::Standard(request) = mat.request else {
        panic!("`{formula}` must materialize as a standard request");
    };
    residual_cascade_structural_signature(&request)
}

#[test]
fn scattered_duchon_positive_control_passes_structural_predicate() {
    // The eligible shape every negative below differs from in one guard.
    let data = sample_2d(800);
    let signature = structural_signature("y ~ duchon(x1, x2)", &data, &gaussian_config())
        .expect("a single Gaussian-identity scattered 2-D duchon is structurally eligible");
    assert_eq!(signature.feature_cols.len(), 2);
    assert!(signature.requested_sobolev_order.is_finite());
}

#[test]
fn small_n_scattered_duchon_falls_through_to_dense() {
    // A scattered 2-D duchon below the dense-kernel cliff: the dense radial
    // posterior is both exact and cheap here, so the cascade — a DIFFERENT
    // posterior — must NOT swap it. Falling through preserves the user's model.
    // The request is structurally eligible, so the size gate alone refuses it.
    let data = sample_2d(800);
    let cfg = gaussian_config();
    let mat = materialize("y ~ duchon(x1, x2)", &data, &cfg).expect("materialize");
    let FitRequest::Standard(request) = mat.request else {
        panic!("duchon must materialize as a standard request");
    };
    assert!(residual_cascade_structural_signature(&request).is_some());
    assert!(
        residual_cascade_fast_path(&request).is_none(),
        "a below-cliff scattered duchon must fall through to the dense path \
         (the cascade is a different posterior; it may not silently swap it)"
    );
    let routed =
        fit_residual_cascade_from_formula("y ~ duchon(x1, x2)", &data, &cfg).expect("materialize");
    assert!(routed.is_none());
}

#[test]
fn non_gaussian_family_fails_structural_predicate() {
    // The cascade solves a Gaussian-identity least-squares problem only. The
    // count response keeps Poisson materialization valid, and the same data is
    // structurally eligible under Gaussian, so the family guard is the only
    // difference.
    let data = sample_2d_with(800, true);
    assert!(
        structural_signature("y ~ duchon(x1, x2)", &data, &gaussian_config()).is_some(),
        "the count design is structurally eligible under the Gaussian family"
    );
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    assert!(
        structural_signature("y ~ duchon(x1, x2)", &data, &cfg).is_none(),
        "a non-Gaussian family must never route through the cascade"
    );
}

#[test]
fn one_dimensional_smooth_fails_structural_predicate() {
    // d=1 is the spline-scan's domain, not the cascade's (d ∈ {2,3}).
    let data = sample_2d(800);
    assert!(
        structural_signature("y ~ duchon(x1)", &data, &gaussian_config()).is_none(),
        "a 1-D smooth must fall through (cascade domain is d ∈ {{2,3}})"
    );
}

#[test]
fn tensor_te_fails_structural_predicate() {
    // te() is a Kronecker-marginal tensor smooth, not a scattered radial one;
    // it is NEVER a cascade candidate (the grid engine #1031 is its own lane).
    let data = sample_2d(800);
    assert!(
        structural_signature("y ~ te(x1, x2)", &data, &gaussian_config()).is_none(),
        "a tensor te() smooth must fall through (not a scattered radial smooth)"
    );
}

#[test]
fn extra_linear_term_fails_structural_predicate() {
    // The cascade fast path is the SINGLE-smooth problem; an extra parametric
    // term means the dense additive machinery must own the fit.
    let data = sample_2d(800);
    assert!(
        structural_signature("y ~ x1 + duchon(x1, x2)", &data, &gaussian_config()).is_none(),
        "a smooth + linear term must fall through to the dense path"
    );
}

/// `fit_from_formula` auto-route NEGATIVE (#1032): a below-cliff scattered
/// duchon must NOT come back as `FitResult::ResidualCascade` — the dense radial
/// posterior is exact and cheap here, so the auto-route in `fit_from_formula`
/// falls through to the dense `fit_model` path and the user's chosen posterior
/// is preserved (the cascade is a different posterior, never a silent swap).
/// The cliff-scale POSITIVE (an actual `FitResult::ResidualCascade`) is left to
/// the certification suite, which already exercises the cascade estimator; here
/// we certify the dispatch never mis-fires below the cliff (cheap, no
/// cliff-scale data needed).
#[test]
fn fit_from_formula_below_cliff_does_not_return_cascade_variant() {
    let data = sample_2d(800);
    let cfg = gaussian_config();
    let result = fit_from_formula("y ~ duchon(x1, x2)", &data, &cfg).expect("dense fit ok");
    assert!(
        !matches!(result, FitResult::ResidualCascade(_)),
        "below-cliff scattered duchon must NOT auto-route to the cascade variant"
    );
}

//! Regression guard: `s(t, periodic=true, period=2π)` must produce
//! identical predictions at the seam — f(t=0) and f(t=2π) refer to the
//! same point on the circle, so their predicted values must agree to
//! machine precision.
//!
//! Original ticket: an earlier build of the periodic 1-D smooth showed
//! a ~7% gap (f(0) = 1.6581, f(2π) = 1.5882, gap = 0.0699) on a smooth
//! `1 + 0.6·cos(t) + 0.3·sin(2t)` truth. Root cause was a non-cyclic
//! prediction-time basis mapping at t=0 vs t=2π. Now fixed; the gap
//! holds at the `< 1e-6` tolerance asserted below.
//!
//! Kept as a regression guard so any future loss of seam continuity in
//! the periodic basis surfaces immediately.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const TAU: f64 = std::f64::consts::TAU;

#[test]
fn periodic_bspline_1d_seam_predictions_match_at_zero_and_two_pi() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(11);
    let u = Uniform::new(0.0, TAU).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let n = 200usize;
    let mut t: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = t
        .iter()
        .map(|theta| 1.0 + 0.6 * theta.cos() + 0.3 * (2.0 * theta).sin() + noise.sample(&mut rng))
        .collect();

    let headers = ["t", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = t
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ s(t, periodic=true, period=6.283185307179586)",
        &data,
        &cfg,
    )
    .expect("periodic 1D fit should succeed");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };

    // Predict at exactly t=0 and t=2π (also a near-seam pair to confirm
    // local continuity vs the seam jump).
    let probe = [0.0, 1e-9, TAU - 1e-9, TAU];
    let mut new_data = Array2::<f64>::zeros((probe.len(), 2));
    for (i, &v) in probe.iter().enumerate() {
        new_data[[i, 0]] = v;
        new_data[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(new_data.view(), &fit.resolvedspec)
        .expect("rebuild predict design");
    let pred = design.design.apply(&fit.fit.beta);

    eprintln!(
        "  f(0)     = {:.6}\n  f(ε)     = {:.6}\n  f(2π−ε)  = {:.6}\n  f(2π)    = {:.6}",
        pred[0], pred[1], pred[2], pred[3]
    );
    let gap = (pred[0] - pred[3]).abs();
    eprintln!("[periodic-1d-seam] |f(0) - f(2π)| = {gap:.6e}");
    assert!(
        gap < 1e-6,
        "periodic 1D seam discontinuous: |f(0) - f(2π)| = {gap:.6e} > 1e-6",
    );
}

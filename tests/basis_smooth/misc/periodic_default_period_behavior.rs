//! Period-resolution contract, split by WHICH mechanism declares periodicity.
//!
//! A `periodic=true` FLAG (or a `bc=['periodic', ...]` selector) on an
//! otherwise-open B-spline / radial margin must require an explicit period.
//! There the true period is a modeling assumption distinct from the sample:
//! uniform draws on [0, 2π] usually have range [ε, 2π-ε], so inferring the
//! period from data max-min would create off-by-ε seam discontinuities.
//!
//! A cyclic *basis selector* (`cc`/`cyclic`, i.e. mgcv `bs="cc"`) is the
//! opposite case: the basis is DEFINED to wrap over its knot range, whose
//! default is the observed data range, with the two endpoint knots identified
//! by construction. There is no separate "true period" to approximate, so —
//! exactly like mgcv's `s(x, bs="cc")` and the tensor `cc` margin — a bare
//! cyclic smooth must be accepted and wrap on `[min, max]`, not rejected.

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

fn make_data_on(
    range: (f64, f64),
    period_truth: f64,
    n: usize,
    seed: u64,
) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(range.0, range.1).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut t: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = t
        .iter()
        .map(|theta| (theta * TAU / period_truth).sin() + noise.sample(&mut rng))
        .collect();
    let headers = ["t", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = t
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn make_tensor_data(n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u_theta = Uniform::new(0.0, TAU).expect("theta uniform");
    let u_h = Uniform::new(-1.0, 1.0).expect("height uniform");
    let noise = Normal::new(0.0, 0.02).expect("normal");
    let headers = ["theta", "h", "y"].into_iter().map(String::from).collect();
    let rows = (0..n)
        .map(|_| {
            let theta = u_theta.sample(&mut rng);
            let h = u_h.sample(&mut rng);
            let y = theta.cos() + 0.25 * h + noise.sample(&mut rng);
            StringRecord::from(vec![theta.to_string(), h.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode tensor")
}

fn assert_rejects_missing_period(formula: &str, data: &gam::data::EncodedDataset) {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let err = fit_from_formula(formula, data, &cfg)
        .err()
        .unwrap_or_else(|| panic!("`{formula}` must reject a periodic margin without period=..."));
    let lower = err.to_string().to_lowercase();
    assert!(
        lower.contains("period") && (lower.contains("explicit") || lower.contains("requires")),
        "rejection must name the missing period and be actionable for `{formula}`; got: {err}",
    );
}

/// A cyclic *basis selector* (`cc`/`cyclic`, mgcv `bs="cc"`) needs no explicit
/// period: it wraps over the observed data range. Assert the fit is ACCEPTED
/// and that the converged curve is genuinely periodic — fitted(min) ==
/// fitted(max) — the defining structural property of a cyclic basis, which
/// holds independent of fit quality.
fn assert_accepts_and_wraps(formula: &str, data: &gam::data::EncodedDataset, feature: &str) {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).unwrap_or_else(|e| {
        panic!("`{formula}` (cyclic basis) must fit by wrapping on the data range; got error: {e}")
    });
    let FitResult::Standard(fit) = result else {
        panic!("`{formula}` expected a standard GAM fit");
    };
    let fidx = data.column_map()[feature];
    let column = data.values.column(fidx);
    let (lo, hi) = column
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let probe = [lo, hi];
    let mut design_pts = Array2::<f64>::zeros((probe.len(), data.headers.len()));
    for (i, &x) in probe.iter().enumerate() {
        design_pts[[i, fidx]] = x;
    }
    let design = build_term_collection_design(design_pts.view(), &fit.resolvedspec)
        .expect("rebuild design at wrap endpoints");
    let fitted = design.design.apply(&fit.fit.beta).to_vec();
    let wrap_gap = (fitted[0] - fitted[1]).abs();
    assert!(
        wrap_gap < 1e-6,
        "`{formula}` must wrap: |fitted(min) - fitted(max)| = {wrap_gap:.3e} (>= 1e-6)"
    );
}

#[test]
fn periodic_without_explicit_period_behavior_consistent() {
    init_parallelism();
    let data = make_data_on((0.0, TAU), TAU, 200, 11);
    // A `periodic=true` FLAG on an open B-spline still requires an explicit
    // period (the off-by-ε seam concern is real for a forced-periodic basis).
    assert_rejects_missing_period("y ~ s(t, periodic=true)", &data);
    // A cyclic BASIS selector (mgcv `bs="cc"`) wraps on the observed data
    // range and must be accepted — matching mgcv, the tensor `cc` margin, and
    // the #874 / cc-margin regressions.
    assert_accepts_and_wraps("y ~ cyclic(t)", &data, "t");
    assert_accepts_and_wraps("y ~ cc(t)", &data, "t");
}

#[test]
fn tensor_periodic_margin_without_explicit_period_rejected() {
    init_parallelism();
    let data = make_tensor_data(80, 19);
    assert_rejects_missing_period("y ~ te(theta, h, periodic=[0])", &data);
    assert_rejects_missing_period("y ~ te(theta, h, bc=['periodic', 'natural'])", &data);
}

#[test]
fn periodic_with_explicit_period_matches_truth() {
    init_parallelism();
    // Data on [0, 2π], truth period 2π. With explicit period=2π, fit must
    // recover well.
    let data = make_data_on((0.0, TAU), TAU, 300, 11);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ s(t, periodic=true, period=6.283185307179586)",
        &data,
        &cfg,
    )
    .expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!()
    };
    let probes: Vec<f64> = (0..50).map(|i| TAU * (i as f64) / 49.0).collect();
    let mut m = Array2::<f64>::zeros((probes.len(), 2));
    for (i, &v) in probes.iter().enumerate() {
        m[[i, 0]] = v;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    let truth: Vec<f64> = probes.iter().map(|t| t.sin()).collect();
    let sumsq: f64 = pred
        .iter()
        .zip(truth.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    let rmse = (sumsq / pred.len() as f64).sqrt();
    eprintln!("[per-explicit] rmse vs truth = {rmse:.4}");
    assert!(
        rmse < 0.1,
        "explicit period=2π fit should recover sin(t): rmse={rmse:.4}"
    );
}

//! Regression for #507: a `bounded(...)` term must not be refused by the
//! pre-fit identifiability audit over its deliberately-zeroed placeholder
//! column.
//!
//! The bounded fit path supplies each bounded coefficient non-linearly
//! (`β = min + width·σ(θ)`, through the family adapter's offset) and places a
//! *zeroed* placeholder column for it in the linear block design. Before the
//! fix, the joint-RRQR identifiability audit read that zeroed column as a
//! structural rank deficiency and FATAL-refused every model containing a
//! bounded term — the rank deficit always equalled the number of bounded
//! terms. The fix reports the bounded block's *true* β-dependent Jacobian
//! (`∂η/∂θ = (dβ/dθ)·x`) to the audit, so a bounded column is rank-deficient
//! exactly when its covariate is genuinely collinear, never merely because the
//! placeholder was zeroed.

use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};
use std::f64::consts::PI;

fn corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        cov += (x - ma) * (y - mb);
        va += (x - ma) * (x - ma);
        vb += (y - mb) * (y - mb);
    }
    cov / (va.sqrt() * vb.sqrt())
}

/// Build a deterministic dataset with linearly *independent* covariates `x`
/// and `z` (a low-frequency sinusoid keeps `z` uncorrelated with the linear
/// `x` ramp) and a noiseless response `y = 0.3 + 0.5·x + 1.2·z`.
fn make_dataset(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut xs = Vec::with_capacity(n);
    let mut zs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        let x = -1.0 + 2.0 * t; // ramp in [-1, 1]
        let z = (2.0 * PI * 3.0 * t).sin(); // 3-cycle sine, ⟂ to the ramp
        let y = 0.3 + 0.5 * x + 1.2 * z;
        xs.push(x);
        zs.push(z);
        ys.push(y);
    }
    (xs, zs, ys)
}

fn encode(headers: Vec<&str>, cols: &[&[f64]]) -> gam::data::EncodedDataset {
    use csv::StringRecord;
    let headers = headers.into_iter().map(String::from).collect();
    let n = cols[0].len();
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        rows.push(StringRecord::from(
            cols.iter().map(|c| c[i].to_string()).collect::<Vec<_>>(),
        ));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_standard(formula: &str, data: &gam::data::EncodedDataset) -> gam::StandardFitResult {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg)
        .unwrap_or_else(|e| panic!("bounded fit must not be refused; formula `{formula}`: {e}"));
    match result {
        FitResult::Standard(fit) => fit,
        _ => panic!("expected a standard fit for `{formula}`"),
    }
}

fn bounded_coefficient(fit: &gam::StandardFitResult, term: &str) -> f64 {
    let (_, range) = fit
        .design
        .linear_ranges
        .iter()
        .find(|(name, _)| name == term)
        .unwrap_or_else(|| panic!("no linear range named `{term}`"));
    fit.fit.beta[range.start]
}

fn fitted_eta(fit: &gam::StandardFitResult) -> Vec<f64> {
    use gam::matrix::LinearOperator;
    fit.design.design.apply(&fit.fit.beta).to_vec()
}

/// The documented companion pattern `y ~ bounded(x, 0, 1) + z` with a true
/// interior bounded slope (0.5). Must fit, keep the bounded coefficient inside
/// `(0, 1)`, and recover the signal.
#[test]
fn bounded_plus_linear_is_not_refused_and_recovers_signal() {
    let (xs, zs, ys) = make_dataset(200);
    let data = encode(vec!["x", "z", "y"], &[&xs, &zs, &ys]);
    let fit = fit_standard("y ~ bounded(x, min=0, max=1) + z", &data);

    let coeff = bounded_coefficient(&fit, "x");
    assert!(
        (0.0..=1.0).contains(&coeff),
        "bounded coefficient escaped its (0,1) box: {coeff}"
    );
    assert!(
        (coeff - 0.5).abs() < 0.05,
        "bounded slope should recover the true 0.5, got {coeff}"
    );

    let eta = fitted_eta(&fit);
    let c = corr(&eta, &ys);
    assert!(
        c > 0.99,
        "fit should recover the noiseless signal, corr={c}"
    );
}

/// Second angle, same root cause: a *lone* bounded term `y ~ bounded(x, -1, 1)`
/// (the issue's minimal repro, default prior=None). The block is
/// `{intercept, x}`; with the zeroed placeholder the old audit reported
/// "joint rank 1 < 2". With the true Jacobian the rank is 2 and the fit
/// proceeds, recovering the true slope 0.5 strictly inside (-1, 1).
#[test]
fn lone_bounded_term_is_not_refused() {
    let n = 200;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let x = -1.0 + 2.0 * (i as f64 / (n as f64 - 1.0));
        xs.push(x);
        ys.push(0.3 + 0.5 * x);
    }
    let data = encode(vec!["x", "y"], &[&xs, &ys]);
    let fit = fit_standard("y ~ bounded(x, min=-1, max=1)", &data);

    let coeff = bounded_coefficient(&fit, "x");
    assert!(
        (-1.0..=1.0).contains(&coeff),
        "lone bounded coefficient escaped (-1,1): {coeff}"
    );
    assert!(
        (coeff - 0.5).abs() < 0.05,
        "lone bounded slope should recover 0.5, got {coeff}"
    );

    let eta = fitted_eta(&fit);
    let c = corr(&eta, &ys);
    assert!(c > 0.99, "lone bounded fit should recover signal, corr={c}");
}

/// Third angle: *two* bounded terms on independent covariates. The old audit's
/// reported rank deficit always equalled the number of bounded terms, so a
/// two-bounded-term model is the sharpest check that the deficit is gone (it
/// would have been "joint rank 1 < 3" before the fix). Both coefficients must
/// land inside their boxes and the joint signal must be recovered.
#[test]
fn two_bounded_terms_independent_covariates() {
    let n = 240;
    let mut xs = Vec::with_capacity(n);
    let mut zs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        let x = -1.0 + 2.0 * t;
        let z = (2.0 * PI * 2.0 * t).cos();
        xs.push(x);
        zs.push(z);
        ys.push(0.3 + 0.6 * x + 0.4 * z);
    }
    let data = encode(vec!["x", "z", "y"], &[&xs, &zs, &ys]);
    let fit = fit_standard(
        "y ~ bounded(x, min=0, max=1) + bounded(z, min=0, max=1)",
        &data,
    );

    let bx = bounded_coefficient(&fit, "x");
    let bz = bounded_coefficient(&fit, "z");
    assert!(
        (0.0..=1.0).contains(&bx),
        "x bounded coeff escaped box: {bx}"
    );
    assert!(
        (0.0..=1.0).contains(&bz),
        "z bounded coeff escaped box: {bz}"
    );
    assert!((bx - 0.6).abs() < 0.05, "x bounded slope ≈ 0.6, got {bx}");
    assert!((bz - 0.4).abs() < 0.05, "z bounded slope ≈ 0.4, got {bz}");

    let eta = fitted_eta(&fit);
    let c = corr(&eta, &ys);
    assert!(c > 0.99, "two-bounded fit should recover signal, corr={c}");
}

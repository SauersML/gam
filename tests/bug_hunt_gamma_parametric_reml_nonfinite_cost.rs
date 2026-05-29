//! Regression for issue #359: a Gamma family with a log link on perfectly
//! ordinary, strictly-positive, finite, well-conditioned data must fit. It
//! used to abort at REML startup because the outer (REML/LAML) objective
//! evaluated to a non-finite cost for *every* smoothing-parameter seed:
//!
//! ```text
//! REML smoothing optimization failed to converge: no candidate seeds passed
//! outer startup validation ... rejected_by_domain=6 ...
//!   seed 0 (validation): ... outer eval failed: objective returned a non-finite cost
//! ```
//!
//! Root cause (fixed): the Gamma log-likelihood consumed by the outer
//! objective used the *full saturated* form, whose normalizing term
//! `shape·ln(shape) − lnΓ(shape)` overflows once the per-iterate shape
//! estimate saturates to `GAMMA_SHAPE_MAX = 1e12` (which happens on the
//! common high-dispersion / CV≈1 case, and at the extreme ρ probed during
//! seed screening). The outer objective now uses the scaled-deviance form
//! `ℓ = −½ D(y, μ)`, matching the Tweedie family and the mgcv convention:
//! the bounded unit deviance keeps `shape · d(y, μ)` finite.
//!
//! The data construction below is RNG-free and ports the deterministic repro
//! from the issue: Exponential(1) inverse-CDF noise (shape ≈ 1, CV ≈ 1) on a
//! log-linear Gamma mean `μ = exp(0.3 + 0.5 x)`. The identical design/response
//! fits without complaint under `gaussian` and `poisson`.

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

/// Build the deterministic #359 dataset: strictly positive, finite,
/// log-linear Gamma response with high dispersion.
fn build_data() -> gam::data::EncodedDataset {
    let n = 400usize;
    let x: Vec<f64> = (0..n)
        .map(|i| -2.0 + 4.0 * i as f64 / (n as f64 - 1.0))
        .collect();
    // Exponential(1) inverse-CDF on the regular grid u = (i + 0.5)/n, mean 1.
    let w_sorted: Vec<f64> = (0..n)
        .map(|i| {
            let u = (i as f64 + 0.5) / n as f64;
            // w = -log1p(-u) = -ln(1 - u): Exponential(1) inverse-CDF, mean 1.
            -(-u).ln_1p()
        })
        .collect();
    // Decouple the noise order from x (same permutation as the issue repro).
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let w = w_sorted[(i * 7919) % n];
            (0.3 + 0.5 * x[i]).exp() * w
        })
        .collect();

    for &yi in &y {
        assert!(
            yi.is_finite() && yi > 0.0,
            "constructed y must be positive finite"
        );
    }

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_family(family: &str, data: &gam::data::EncodedDataset) -> Result<Vec<f64>, String> {
    let cfg = FitConfig {
        family: Some(family.to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ x", data, &cfg).map_err(|e| format!("fit error: {e}"))?;
    let FitResult::Standard(fit) = result else {
        return Err("non-standard fit".to_string());
    };
    let beta = fit.fit.beta.to_vec();
    if beta.iter().any(|v| !v.is_finite()) {
        return Err(format!("non-finite beta: {beta:?}"));
    }
    Ok(beta)
}

#[test]
fn gamma_log_link_ordinary_data_fits_with_finite_coefficients() {
    let data = build_data();

    // Gaussian and Poisson must keep working on the identical design/response.
    fit_family("gaussian", &data).expect("gaussian fit on the same data must succeed");
    fit_family("poisson", &data).expect("poisson fit on the same data must succeed");

    // The defect: gamma aborted with `objective returned a non-finite cost`.
    let beta =
        fit_family("gamma", &data).expect("gamma log-link fit on ordinary data must converge");
    assert!(
        beta.iter().all(|v| v.is_finite()),
        "gamma fit returned non-finite coefficients: {beta:?}"
    );
}

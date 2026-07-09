//! Regression for issue #2128. A Gamma GAM with a penalized smooth
//! (`family="gamma"`, `y ~ s(x)`) on clean, well-specified higher-variance
//! Gamma data (shape ≈ 2, CV ≈ 0.71) aborts at REML startup: the outer
//! objective returns a non-finite cost for *every* smoothing-parameter seed
//! (`rejected_by_domain`) so no seed passes startup validation and the solver
//! never runs. Controls that succeed on the same design/basis: parametric
//! Gamma (`y ~ x`), and Gaussian / Poisson smooths. So the model is
//! identifiable and the data ordinary; the defect is specific to the
//! Gamma + penalized-smooth REML objective at moderate/high dispersion.
//!
//! Companion to `gamma_smooth_reml_startup_rejects_all_seeds.rs` (shape ≈ 4,
//! CV ≈ 0.5). Data is RNG-free and deterministic.

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

/// Deterministic Gamma(shape ≈ 2) dataset on a log-linear mean
/// `μ = exp(0.2 + 1.5 x)`, x ∈ [0, 1]. A Gamma(2, μ/2) draw (mean μ,
/// CV = 1/√2 ≈ 0.71) is the scaled sum of two Exponential(1) inverse-CDF
/// values, each drawn from a shared grid permuted by a distinct coprime
/// stride so the two "draws" per row are effectively independent and
/// decorrelated from x.
fn build_data() -> gam::data::EncodedDataset {
    let n = 200usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();

    let exp_table: Vec<f64> = (0..n)
        .map(|i| {
            let u = (i as f64 + 0.5) / n as f64;
            -(-u).ln_1p() // -ln(1 - u): Exponential(1) inverse-CDF, mean 1.
        })
        .collect();
    let strides = [7919usize, 6311];

    let y: Vec<f64> = (0..n)
        .map(|i| {
            let g2: f64 = strides
                .iter()
                .enumerate()
                .map(|(k, stride)| exp_table[(i * stride + 101 * k) % n])
                .sum();
            let mu = (0.2 + 1.5 * x[i]).exp();
            mu * g2 / 2.0
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

fn fit_family(
    family: &str,
    formula: &str,
    data: &gam::data::EncodedDataset,
) -> Result<Vec<f64>, String> {
    let cfg = FitConfig {
        family: Some(family.to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).map_err(|e| format!("fit error: {e}"))?;
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
fn gamma_smooth_highvar_fits_with_finite_coefficients_2128() {
    let data = build_data();

    // Controls on the same data prove the design and the problem are well posed.
    fit_family("gamma", "y ~ x", &data)
        .expect("parametric gamma fit on the same data must succeed");
    fit_family("gaussian", "y ~ s(x, k=10)", &data)
        .expect("gaussian smooth fit on the same data must succeed");

    // The defect: gamma smooth aborted at REML startup with `objective returned
    // a non-finite cost` for every seed (rejected_by_domain).
    let beta = fit_family("gamma", "y ~ s(x, k=10)", &data)
        .expect("gamma log-link smooth fit on high-variance data must converge");
    assert!(
        beta.iter().all(|v| v.is_finite()),
        "gamma smooth fit returned non-finite coefficients: {beta:?}"
    );
}

//! Regression for issue #511 (smooth-path half; regression of #359). The Gamma
//! REML/LAML *outer* objective evaluated to a non-finite cost for every
//! smoothing-parameter seed, so startup validation rejected all of them
//! (`rejected_by_domain`) and the penalized-smooth fit aborted before the
//! solver started:
//!
//! ```text
//! REML smoothing optimization failed to converge: no candidate seeds passed
//! outer startup validation ... rejected_by_domain=7 ...
//!   seed 0 (validation): ... outer eval failed: objective returned a non-finite cost
//! ```
//!
//! Root cause (fixed): the inner P-IRLS re-estimated the Gamma dispersion shape
//! on every Levenberg–Marquardt *trial* candidate (and every halving attempt).
//! Because the trial deviance it reports is `2·shape·Σ wᵢ dᵢ`, changing the
//! shape per trial silently changed the objective relative to the predicted
//! reduction (built from the gradient/Hessian at the last accepted shape). The
//! LM gain ratio then compared two different objectives, rejected every step,
//! ran λ_LM to its ceiling, and the inner solve stalled with a large residual
//! gradient ("LM step search exhausted"). That non-convergence retreat surfaced
//! to the outer objective as an infeasible (+∞) cost — the "non-finite cost"
//! startup rejection. The shape is now held fixed within an inner LM step and
//! updated once per accepted iterate (block-coordinate β | shape), so the inner
//! solve converges and the outer cost is finite.
//!
//! This half exercises the *lower-dispersion* regime (Gamma shape ≈ 4, CV ≈ 0.5)
//! on a penalized smooth `y ~ s(x, k=10)`, which #359 never covered and where
//! the failure manifested even though the parametric fit of the same regime
//! succeeds. The companion
//! `tests/bug_hunt_gamma_parametric_reml_nonfinite_cost.rs` covers the
//! parametric / CV ≈ 1 half. The data is RNG-free and deterministic.

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

/// Build a deterministic Gamma(shape ≈ 4) dataset on a log-linear mean
/// `μ = exp(0.3 + 0.5 x)`. A Gamma(4, μ/4) draw (mean μ, CV = 1/√4 = 0.5) is
/// realized as the scaled sum of four Exponential(1) inverse-CDF values, each
/// drawn from a distinct deterministic grid and de-correlated from `x` by a
/// different coprime stride.
fn build_data() -> gam::data::EncodedDataset {
    let n = 600usize;
    let x: Vec<f64> = (0..n)
        .map(|i| -2.0 + 4.0 * i as f64 / (n as f64 - 1.0))
        .collect();

    // Four Exponential(1) inverse-CDF tables on a shared regular grid, each
    // permuted by a different coprime stride so the four "draws" per row are
    // effectively independent and decoupled from x.
    let exp_table: Vec<f64> = (0..n)
        .map(|i| {
            let u = (i as f64 + 0.5) / n as f64;
            -(-u).ln_1p() // -ln(1 - u): Exponential(1) inverse-CDF, mean 1.
        })
        .collect();
    let strides = [7919usize, 6311, 5003, 3571];

    let y: Vec<f64> = (0..n)
        .map(|i| {
            let g4: f64 = strides
                .iter()
                .enumerate()
                .map(|(k, stride)| exp_table[(i * stride + 101 * k) % n])
                .sum();
            // Gamma(4,1) realization g4 (mean 4) scaled to mean μ_i, CV 0.5.
            let mu = (0.3 + 0.5 * x[i]).exp();
            mu * g4 / 4.0
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
    let result =
        fit_from_formula("y ~ s(x, k=10)", data, &cfg).map_err(|e| format!("fit error: {e}"))?;
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
fn gamma_log_link_smooth_lowdispersion_fits_with_finite_coefficients() {
    let data = build_data();

    // Gaussian and Poisson fit the identical design/basis, proving the smooth
    // and penalty are well-posed — only the Gamma objective was defective.
    fit_family("gaussian", &data).expect("gaussian smooth fit on the same data must succeed");
    fit_family("poisson", &data).expect("poisson smooth fit on the same data must succeed");

    // The defect: gamma aborted at REML startup with `objective returned a
    // non-finite cost` because the inner solve never converged.
    let beta = fit_family("gamma", &data)
        .expect("gamma log-link smooth fit on low-dispersion data must converge");
    assert!(
        beta.iter().all(|v| v.is_finite()),
        "gamma smooth fit returned non-finite coefficients: {beta:?}"
    );
}

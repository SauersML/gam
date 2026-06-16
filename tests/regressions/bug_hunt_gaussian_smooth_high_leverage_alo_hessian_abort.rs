//! Regression: an ordinary penalized Gaussian smooth fit (`y ~ s(x)`) must not
//! abort at REML smoothing-parameter selection just because the data contains a
//! high-leverage point (issue #574).
//!
//! Root cause: the ALO-stabilized REML criterion (feee4f7b5) augments the REML
//! gradient on detected high-leverage instability (max hat-diagonal >= 0.80) and
//! used to deliberately invalidate the analytic Hessian (`HessianResult::
//! Unavailable`). But the outer optimizer for the dense Gaussian-identity route
//! is planned as `HessianSource::Analytic` (exact-Hessian ARC) and is configured
//! to treat a non-analytic Hessian as fatal, so the fit died with
//! "outer plan declared HessianSource::Analytic but the runtime returned
//! HessianResult::Unavailable". The cost and gradient are both augmented and
//! therefore consistent; the base REML analytic Hessian is a valid second-order
//! model for the augmented objective under ARC's adaptive cubic regularization
//! (the ratio test drives convergence on the exact augmented gradient), so the
//! Hessian must be retained rather than dropped.
//!
//! This test is RNG-free: a dense cluster on [-2, 2] plus three isolated
//! abscissae at x ~= +/-6, whose spline hat-diagonals approach 1.0. The control
//! (dense cluster alone) is well-posed and fits; adding the leverage spike must
//! still fit and return finite coefficients.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

fn encode_columns(headers: &[&str], columns: &[&[f64]]) -> gam::data::EncodedDataset {
    let n = columns[0].len();
    let hdrs: Vec<String> = headers.iter().map(|s| (*s).to_string()).collect();
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for i in 0..n {
        let row: Vec<String> = columns.iter().map(|c| c[i].to_string()).collect();
        rows.push(StringRecord::from(row));
    }
    encode_recordswith_inferred_schema(hdrs, rows).expect("encode dataset")
}

/// Smooth deterministic signal plus a reproducible, RNG-free wiggle.
fn signal(x: f64, i: usize) -> f64 {
    x.sin() + 0.1 * (3.0 * x).cos() + 0.3 * (i as f64 * 2.399_96).sin()
}

fn fit_gaussian_smooth(x: &[f64], y: &[f64]) -> Result<(), String> {
    let ds = encode_columns(&["x", "y"], &[x, y]);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula("y ~ s(x)", &ds, &cfg) {
        Ok(FitResult::Standard(fit)) => {
            if fit.fit.beta.iter().all(|b| b.is_finite()) {
                Ok(())
            } else {
                Err("fit returned non-finite coefficients".to_string())
            }
        }
        Ok(_) => Err("expected a standard GAM fit".to_string()),
        Err(e) => Err(format!("{e:?}")),
    }
}

#[test]
fn gaussian_smooth_with_high_leverage_point_fits_without_aborting() {
    init_parallelism();

    // Dense cluster on [-2, 2] (well-posed control set).
    let n_cluster = 200usize;
    let mut x: Vec<f64> = Vec::with_capacity(n_cluster + 3);
    let mut y: Vec<f64> = Vec::with_capacity(n_cluster + 3);
    for i in 0..n_cluster {
        let xi = -2.0 + 4.0 * (i as f64) / (n_cluster as f64 - 1.0);
        x.push(xi);
        y.push(signal(xi, i));
    }

    // ---- control: the dense cluster alone fits without incident ------------
    let control = fit_gaussian_smooth(&x, &y);
    assert!(
        control.is_ok(),
        "control (dense cluster only) must fit: {}",
        control.unwrap_err()
    );

    // ---- add an isolated high-leverage spike: x ~= +/-6 with big gaps ------
    for (k, &xi) in [6.0_f64, 6.05, -6.0].iter().enumerate() {
        x.push(xi);
        y.push(signal(xi, n_cluster + k));
    }

    // The leverage spike must NOT make the fit abort. REML can handle a
    // high-leverage point; it just must not drop the analytic Hessian its own
    // outer route requires.
    let high_leverage = fit_gaussian_smooth(&x, &y);
    assert!(
        high_leverage.is_ok(),
        "high-leverage Gaussian smooth must fit (issue #574: ALO must not drop \
         the analytic Hessian the ARC outer route requires): {}",
        high_leverage.unwrap_err()
    );
}

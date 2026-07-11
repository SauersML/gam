//! Regression gate for #2254 — a Gaussian-identity fit on a perfectly constant
//! (zero-variance) response must produce a COMPLETE, self-consistent fit, not a
//! half-built one that hard-fails downstream.
//!
//! ## The defect
//!
//! `fit_from_formula` routes any constant Gaussian response through the
//! `constant_gaussian_standard_fit` fast-path, which correctly recovered the
//! coefficients (β = intercept, smooth ≡ 0) but returned a `UnifiedFitResult`
//! with `inference: None` / `geometry: None`. So the fit *succeeded* yet carried
//! no penalized Hessian, no EDF, and no covariance — and the model/persistence
//! builder (`standard_null_space_metadata`) then hard-errored with
//! "null-space Hessian logdet requires fitted penalized Hessian", even for the
//! simplest model `y ~ 1`. A near-constant response (variance ≈ 1e-12) fit
//! cleanly, so this was a numerical cliff at exactly-zero variance.
//!
//! ## The fix
//!
//! The fast-path now assembles the complete inference/geometry bundle at a
//! fully-smoothed λ: the penalized Hessian `H = XᵀWX + λS`, EDF = tr(H⁻¹XᵀWX),
//! dispersion φ̂ = 0 (the residual is exactly zero), and hence covariance = 0.
//! This test pins that the fit carries that bundle AND that the end-to-end
//! payload assembly (the code path that previously threw) succeeds.

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::inference::model_payload_builders::{StandardPayloadInputs, assemble_standard_payload};
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

fn encode_columns(headers: &[&str], columns: &[&[f64]]) -> EncodedDataset {
    let n = columns[0].len();
    let hdrs: Vec<String> = headers.iter().map(|s| (*s).to_string()).collect();
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for i in 0..n {
        let row: Vec<String> = columns.iter().map(|c| c[i].to_string()).collect();
        rows.push(StringRecord::from(row));
    }
    encode_recordswith_inferred_schema(hdrs, rows).expect("encode dataset")
}

fn gaussian_cfg() -> FitConfig {
    FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    }
}

/// A constant Gaussian response must produce a complete, usable fit for every
/// model shape (intercept-only, parametric, penalized smooth, radial), and the
/// full persistence payload — the path that previously threw — must succeed.
#[test]
fn gaussian_constant_response_fits_completely_and_builds_payload_2254() {
    let n = 120usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();

    for &yval in &[3.7_f64, 0.0, -2.5] {
        let y: Vec<f64> = vec![yval; n];
        let data = encode_columns(&["x", "y"], &[&x, &y]);
        for formula in [
            "y ~ 1",
            "y ~ x",
            "y ~ s(x, k=10)",
            "y ~ duchon(x, centers=8)",
            "y ~ matern(x, centers=8)",
        ] {
            let result = fit_from_formula(formula, &data, &gaussian_cfg())
                .unwrap_or_else(|e| panic!("`{formula}` on constant y={yval} failed to fit: {e}"));
            let FitResult::Standard(fit) = result else {
                panic!("`{formula}` on constant y did not produce a standard fit");
            };

            // (1) The fit carries a complete inference bundle.
            let hessian = fit.fit.penalized_hessian().unwrap_or_else(|| {
                panic!("`{formula}` on constant y={yval}: missing penalized Hessian")
            });
            let p = fit.fit.beta.len();
            assert_eq!(
                (hessian.nrows(), hessian.ncols()),
                (p, p),
                "`{formula}`: penalized Hessian must be p×p"
            );
            assert!(
                hessian.iter().all(|v| v.is_finite()),
                "`{formula}`: penalized Hessian must be finite"
            );
            let edf = fit
                .fit
                .edf_total()
                .unwrap_or_else(|| panic!("`{formula}` on constant y={yval}: missing EDF"));
            assert!(
                edf.is_finite() && edf >= 0.0 && edf <= p as f64 + 1e-6,
                "`{formula}`: EDF {edf} out of range [0, {p}]"
            );
            // A constant response supports no wiggle: a penalized smooth must
            // collapse onto its (small) null space rather than spend the full
            // basis. Only meaningful when the model actually has a penalty.
            if fit.fit.lambdas.len() > 0 {
                assert!(
                    edf < p as f64 - 1e-6,
                    "`{formula}`: a penalized constant fit should collapse below the \
                     full basis dimension, got EDF {edf} (p={p})"
                );
            }

            // (2) The fitted surface reproduces the constant exactly.
            let pred = fit.design.design.apply(&fit.fit.beta);
            let max_abs_err = pred.iter().fold(0.0_f64, |m, &v| m.max((v - yval).abs()));
            assert!(
                max_abs_err <= 1e-8,
                "`{formula}` on constant y={yval}: fitted surface deviates from the \
                 constant by {max_abs_err:.3e}"
            );

            // (3) The end-to-end persistence payload — the path that threw
            // "null-space Hessian logdet requires fitted penalized Hessian" —
            // must now assemble cleanly.
            assemble_standard_payload(StandardPayloadInputs {
                formula: formula.to_string(),
                dataset: &data,
                fit_config: &gaussian_cfg(),
                result: fit,
            })
            .unwrap_or_else(|e| {
                panic!("`{formula}` on constant y={yval}: payload assembly failed: {e}")
            });
        }
    }
}

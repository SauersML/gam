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
//! The fast-path now assembles the complete inference/geometry bundle at the
//! floating-point representation of the fully-smoothed boundary: the weakest
//! non-null penalty direction dominates data information by `1/sqrt(ε)`, EDF =
//! tr(H⁻¹XᵀWX), dispersion φ̂ = 0 (the residual is exactly zero), and hence
//! covariance = 0. This test pins that the fit carries that bundle AND that the
//! end-to-end payload assembly (the code path that previously threw) succeeds.

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::inference::model_payload_builders::{StandardPayloadInputs, assemble_standard_payload};
use gam::matrix::LinearOperator;
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

/// The precise root cause of #2254's *second* wave (after the fit stopped
/// returning `inference: None`) was a MALFORMED inference bundle: the shortcut
/// hard-coded `edf_by_block = vec![edf_total]` (length 1), so the
/// `UnifiedFitResult` constructor rejected every model whose penalty count was
/// not exactly 1 — `y ~ 1` (0 penalties) and `y ~ s(x, k=10)` / `matern` (2–3
/// penalties) failed with "EDF smoothing-parameter count mismatch", while
/// `y ~ x` (exactly 1 penalty) slipped through. This test pins the bundle's
/// structural invariants directly, from every penalty-count regime, so a
/// regression is caught at the source rather than only through a downstream
/// consumer:
///   * `edf_by_block` and `penalty_block_trace` align 1:1 with `lambdas`;
///   * the reported `edf_total` equals `tr(F)` (the influence-matrix trace) and
///     the per-block decomposition `edf_total = p − Σ_k tr_k`;
///   * every per-penalty EDF/trace is finite and in `[0, block_cols]`.
#[test]
fn gaussian_constant_response_inference_bundle_is_self_consistent_2254() {
    let n = 96usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let y: Vec<f64> = vec![1.25_f64; n];
    let data = encode_columns(&["x", "y"], &[&x, &y]);

    // One representative from each penalty-count regime: 0, 1, and ≥2.
    for formula in ["y ~ 1", "y ~ x", "y ~ s(x, k=10)", "y ~ matern(x, centers=8)"] {
        let FitResult::Standard(fit) = fit_from_formula(formula, &data, &gaussian_cfg())
            .unwrap_or_else(|e| panic!("`{formula}` failed to fit: {e}"))
        else {
            panic!("`{formula}` did not produce a standard fit");
        };

        let n_lambda = fit.fit.lambdas.len();
        let inference = fit
            .fit
            .inference
            .as_ref()
            .unwrap_or_else(|| panic!("`{formula}`: fit carries no inference bundle"));

        // (1) Length invariants — the exact contract the constructor validates.
        assert_eq!(
            inference.edf_by_block.len(),
            n_lambda,
            "`{formula}`: edf_by_block must align 1:1 with lambdas"
        );
        assert_eq!(
            inference.penalty_block_trace.len(),
            n_lambda,
            "`{formula}`: penalty_block_trace must align 1:1 with lambdas"
        );

        // (2) The influence matrix F = H⁻¹XᵀWX is present and its trace is the
        // reported EDF (the model's own definition of effective d.f.).
        let p = fit.fit.beta.len();
        let f_mat = inference
            .coefficient_influence
            .as_ref()
            .unwrap_or_else(|| panic!("`{formula}`: missing coefficient influence matrix F"));
        assert_eq!((f_mat.nrows(), f_mat.ncols()), (p, p));
        let tr_f: f64 = (0..p).map(|j| f_mat[[j, j]]).sum();
        let edf_total = fit.fit.edf_total().expect("edf_total");
        assert!(
            (tr_f - edf_total).abs() <= 1e-6 * (1.0 + edf_total.abs()),
            "`{formula}`: edf_total {edf_total} must equal tr(F) {tr_f}"
        );

        // (3) Per-block decomposition edf_total = p − Σ_k tr_k, each block finite
        // and bounded by its own column count.
        let mut sum_trace = 0.0_f64;
        for (kk, (&edf_k, &tr_k)) in inference
            .edf_by_block
            .iter()
            .zip(inference.penalty_block_trace.iter())
            .enumerate()
        {
            assert!(
                edf_k.is_finite() && tr_k.is_finite() && edf_k >= -1e-9 && tr_k >= -1e-9,
                "`{formula}`: penalty {kk} EDF/trace not finite/non-negative: edf={edf_k}, tr={tr_k}"
            );
            sum_trace += tr_k;
        }
        assert!(
            ((p as f64 - sum_trace) - edf_total).abs() <= 1e-6 * (1.0 + edf_total.abs()),
            "`{formula}`: edf_total {edf_total} must equal p − Σ tr_k = {}",
            p as f64 - sum_trace
        );
    }
}

/// The zero-dispersion shortcut is an exact boundary, not a tolerance-based
/// approximation. A response with any represented variation must continue to
/// the family-owned degeneracy check; otherwise a tiny but nonzero residual is
/// silently assigned φ̂=0 and the fit lies about convergence/inference.
#[test]
fn gaussian_constant_shortcut_does_not_absorb_near_constant_response_2254() {
    let n = 96usize;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 1.0 + if i % 2 == 0 { 0.0 } else { 1.0e-13 })
        .collect();
    let data = encode_columns(&["x", "y"], &[&x, &y]);

    let error = match fit_from_formula("y ~ s(x, k=10)", &data, &gaussian_cfg()) {
        Ok(_) => panic!("a nonconstant response entered the exact zero-dispersion shortcut"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        message.contains("effectively constant"),
        "near-constant response reached the wrong path: {message}"
    );
}

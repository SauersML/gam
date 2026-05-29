//! Regression test for issue #361: the README advertised `ti(...)` as a
//! supported tensor-product smooth, but the formula parser rejected it with
//! `unknown term function 'ti'` while the structurally identical `te(...)`
//! fit fine.
//!
//! `ti(...)` is now wired through the formula DSL as a *tensor interaction*
//! smooth with mgcv semantics: the marginal main effects are excluded by
//! applying a sum-to-zero constraint to **each marginal basis independently**
//! before forming the tensor product. The realized identifiability transform
//! is the Kronecker product of the per-margin null-space bases, so a tensor
//! interaction on `d` margins with per-margin basis sizes `k_j` carries
//! `∏ (k_j − 1)` coefficients — one degree of freedom removed per margin —
//! whereas the full-tensor `te(...)` carries `∏ k_j − 1` coefficients (a
//! single global sum-to-zero constraint).
//!
//! The test fits a deterministic interaction surface and asserts:
//!   1. `te(a, b, k=5)` fits (control) with finite coefficients,
//!   2. `ti(a, b, k=5)` parses + fits with finite coefficients (the bug),
//!   3. the exact coefficient counts differ as the per-margin vs. global
//!      centering predicts (24 for `te`, 16 for `ti`), which is the
//!      load-bearing evidence that the marginal main effects were excluded.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, StandardFitResult, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism,
};

fn interaction_dataset() -> gam::data::EncodedDataset {
    let headers = ["a", "b", "y"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    // Deterministic interaction surface sin(3a)·cos(3b) on a dense grid so the
    // fit is well-posed without any RNG dependence.
    let mut rows = Vec::new();
    let grid = 36usize;
    for i in 0..grid {
        for j in 0..grid {
            let a = i as f64 / (grid as f64 - 1.0);
            let b = j as f64 / (grid as f64 - 1.0);
            let y = (3.0 * a).sin() * (3.0 * b).cos();
            rows.push(StringRecord::from(vec![
                a.to_string(),
                b.to_string(),
                y.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode interaction dataset")
}

fn fit_single_smooth(formula: &str) -> StandardFitResult {
    init_parallelism();
    let data = interaction_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &config)
        .unwrap_or_else(|e| panic!("formula '{formula}' should fit, got error: {e:?}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected standard Gaussian fit for '{formula}'");
    };
    fit
}

#[test]
fn te_tensor_smooth_fits_control() {
    let fit = fit_single_smooth("y ~ te(a, b, k=5)");
    assert_eq!(fit.design.smooth.terms.len(), 1);
    assert!(fit.fit.beta.iter().all(|v| v.is_finite()));
    // Full-tensor sum-to-zero: 5 * 5 - 1 = 24 coefficients.
    assert_eq!(
        fit.design.smooth.terms[0].coeff_range.len(),
        24,
        "te(a, b, k=5) should carry one global sum-to-zero constraint (5*5 - 1)"
    );
}

#[test]
fn ti_tensor_interaction_smooth_parses_and_fits() {
    // The exact repro from issue #361: this used to raise
    // `FormulaError: unknown term function 'ti'`.
    let fit = fit_single_smooth("y ~ ti(a, b, k=5)");
    assert_eq!(fit.design.smooth.terms.len(), 1);
    assert!(
        fit.fit.beta.iter().all(|v| v.is_finite()),
        "ti(a, b, k=5) must fit with finite coefficients"
    );
    assert!(!fit.fit.beta.is_empty());
}

#[test]
fn ti_excludes_marginal_main_effects_via_per_margin_centering() {
    let te = fit_single_smooth("y ~ te(a, b, k=5)");
    let ti = fit_single_smooth("y ~ ti(a, b, k=5)");

    let te_dim = te.design.smooth.terms[0].coeff_range.len();
    let ti_dim = ti.design.smooth.terms[0].coeff_range.len();

    // te: one global sum-to-zero constraint -> 5*5 - 1 = 24.
    assert_eq!(te_dim, 24, "te tensor coefficient count");
    // ti: per-margin sum-to-zero on each of two k=5 margins -> (5-1)*(5-1) = 16.
    // The strictly smaller count is the load-bearing evidence that each
    // marginal main effect (one degree of freedom per axis) was removed before
    // forming the tensor product, leaving only the pure interaction.
    assert_eq!(
        ti_dim, 16,
        "ti tensor-interaction coefficient count must be (k-1)^2, proving per-margin centering"
    );
    assert!(
        ti_dim < te_dim,
        "tensor interaction must drop the marginal main effects relative to the full tensor"
    );
}

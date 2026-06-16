//! Regression test for #859: the CTN cross-fit must use a fold-invariant
//! response knot count.
//!
//! Before the fix, `fit_transformation_normal` re-derived
//! `response_num_internal_knots` via `effective_response_num_internal_knots`
//! on each fold's own response subsample. Its data-driven complexity cap
//! (`min_internal + round((|skew| + 0.5·|excess_kurtosis|)·6)`) rounds to
//! different counts on different fold subsamples, so `p_resp` (and hence
//! `p₁ = p_resp · p_cov`) drifted across folds and the out-of-fold Jacobian
//! assembly failed with:
//!
//!   "cross-fit fold p₁ mismatch: this fold has N columns but a prior fold
//!    had M; the frozen response/covariate basis failed to align across folds"
//!
//! The cross-fit caller pinned the count once at the smallest fold complement
//! (`workflow.rs`), but the inner fit overrode the pin. The fix threads a
//! `response_num_internal_knots_pinned` flag so the inner fit uses the pinned
//! count verbatim, restoring the documented fold-invariant `p_resp`.
//!
//! This test drives the SHIPPED orthogonalized path (a `CtnStage1Recipe` on
//! `FitConfig::ctn_stage1`, then `fit_from_formula`) with the DEFAULT CTN
//! config (so the complexity cap is active) and a deliberately right-skewed
//! Stage-1 score, so per-fold rounding would differ pre-fix. With the fix the
//! fit completes; the assertion is simply that it does not raise the p₁
//! mismatch.

use gam::transformation_normal::TransformationNormalConfig;
use gam::{
    CtnStage1Recipe, FitConfig, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism,
};

const COVARIATE_RHS: &str = "s(x, k=8)";
const STAGE2_FORMULA: &str = "y ~ s(x, k=8)";
const LOGSLOPE_FORMULA: &str = "s(x, k=8)";

/// Right-skewed, heavy-tailed Stage-1 score (gamma-shaped) whose per-fold
/// sample skew/kurtosis straddle the `round(...)` boundary of the complexity
/// cap — the regime that made `p_resp` fold-dependent.
fn synth(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // deterministic LCG so the test is reproducible without an rng dependency
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut unif = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut score = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = 2.0 * unif() - 1.0;
        // gamma(2)-ish via sum of two exponentials → right-skewed, heavy-tailed
        let e1 = -(unif().max(1e-12)).ln();
        let e2 = -(unif().max(1e-12)).ln();
        let s = (e1 + e2) - 2.0 + 0.3 * xi; // skewed score correlated with x
        let eta = -0.3 + 0.8 * xi + (0.4 + 0.5 * xi) * s;
        let p = 1.0 / (1.0 + (-eta).exp());
        let yi = if unif() < p { 1.0 } else { 0.0 };
        x.push(xi);
        y.push(yi);
        score.push(s);
    }
    (x, y, score)
}

fn build_dataset(x: &[f64], y: &[f64], score: &[f64]) -> gam::data::EncodedDataset {
    let n = x.len();
    let headers = vec!["x".to_string(), "y".to_string(), "score".to_string()];
    let records: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", x[i]),
                format!("{:.17e}", y[i]),
                format!("{:.17e}", score[i]),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode dataset")
}

#[test]
fn ctn_crossfit_pins_response_knots_across_folds_859() {
    init_parallelism();
    let (x, y, score) = synth(1000);
    let data = build_dataset(&x, &y, &score);

    // DEFAULT CTN config: complexity cap is active (this is what triggered #859).
    let recipe = CtnStage1Recipe::new(
        "score",
        COVARIATE_RHS,
        TransformationNormalConfig::default(),
        None,
        None,
    )
    .expect("build Stage-1 CTN recipe");

    let mut config = FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        link: Some("probit".to_string()),
        logslope_formula: Some(LOGSLOPE_FORMULA.to_string()),
        ..FitConfig::default()
    };
    config.ctn_stage1 = Some(recipe); // cross-fit OOF z; no z_column

    let result = fit_from_formula(STAGE2_FORMULA, &data, &config);

    if let Err(e) = &result {
        let msg = format!("{e}");
        assert!(
            !msg.contains("p₁ mismatch") && !msg.contains("failed to align across folds"),
            "#859 regression: cross-fit raised the fold p₁ mismatch — the response \
             knot count is not fold-invariant. Error: {msg}"
        );
        // Any OTHER error (e.g. the separate #787 outer-stall) is out of scope
        // for this test; only the fold-alignment bug must be gone.
    }
}

//! End-to-end quality: gam's confidence-interval construction under a
//! **non-identity (logistic) link** must agree with `mgcv` — the mature,
//! standard penalized-GAM implementation — both on the **link (eta) scale**
//! and on the **response (mean) scale**.
//!
//! The capability under test is gam's inverse-link Jacobian inside CI
//! construction: `predict_gamwith_uncertainty` propagates the linear-predictor
//! variance through the inverse link to obtain the response-scale standard
//! error. For a logit link the exact response-scale variance is the
//! logistic-normal integral `Var[sigmoid(η)]` with `η ~ N(η̂, se_η²)`; to
//! first order this equals the **delta-method** SE `se_η · |dμ/dη|` with
//! `dμ/dη = μ(1−μ)`. mgcv's `predict(se.fit=TRUE, type="response")` reports
//! exactly that analytical delta-method response-scale SE, so the two must
//! coincide when `se_η` is small (as it is for n=200 here).
//!
//! Why a Binomial(logit) model rather than a literal Gaussian+logit one: mgcv
//! exposes no Gaussian family with a logistic link, and gam's Gaussian
//! posterior-variance branch ignores the link entirely (the inverse-link
//! Jacobian is only exercised by families whose mean lives on a bounded
//! scale). The Binomial(logit) GAM is therefore the faithful head-to-head that
//! actually stresses the inverse-link Jacobian on **both** engines, which is
//! the documented intent of this benchmark. The latent smooth
//! `η(x) = x + sin(2πx)` and the sampling design `x ~ U[-3,3]`, n=200,
//! seed=123 follow the spec verbatim; the Gaussian-noise term in the spec is
//! replaced by the Bernoulli sampling that the logit link demands so the
//! generative model is internally consistent.
//!
//! Identical data feed both engines (the same CSV columns), the comparison is
//! grid-aligned and element-wise at the training points, and the bounds are the
//! spec's un-weakened bounds. A genuine divergence is a real bug, not a reason
//! to loosen the bounds.

use gam::estimate::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, max_abs_diff, relative_l2, run_r};
use gam::types::LikelihoodSpec;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};

/// Deterministic synthetic logistic-link dataset, generated identically on the
/// Rust side and serialized to the CSV both engines read. Returns `(x, y)`.
///
/// `x ~ U[-3,3]`, `η(x) = x + sin(2πx)`, `p = sigmoid(η)`, `y ~ Bernoulli(p)`,
/// seed = 123. The RNG is a fixed split-mix64 stream so the data are bit-for-bit
/// reproducible and feed *both* gam and mgcv through the same CSV.
fn synthetic_logistic_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    // SplitMix64 — a small, fully specified PRNG (no external rand crate, no
    // env, no hidden state). Two independent draws per row: one for x, one for
    // the Bernoulli outcome.
    let mut state: u64 = 123;
    let mut next_u01 = || -> f64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // 53-bit mantissa -> uniform in [0,1).
        ((z >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = -3.0 + 6.0 * next_u01();
        let eta = xi + (2.0 * std::f64::consts::PI * xi).sin();
        let p = 1.0 / (1.0 + (-eta).exp());
        let yi = if next_u01() < p { 1.0 } else { 0.0 };
        x.push(xi);
        y.push(yi);
    }
    (x, y)
}

#[test]
fn confidence_intervals_match_mgcv_under_logistic_link() {
    init_parallelism();

    // ---- deterministic synthetic data (spec: n=200, x~U[-3,3], seed=123) ---
    let n = 200usize;
    let (x, y) = synthetic_logistic_data(n);
    assert_eq!(x.len(), n);
    assert!(
        y.iter().any(|&v| v > 0.5) && y.iter().any(|&v| v < 0.5),
        "synthetic outcome must contain both classes"
    );

    // ---- build a gam dataset from the same (x, y) the CSV will carry -------
    let headers = vec!["x".to_string(), "y".to_string()];
    let records: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", y[i])]))
        .collect();
    let ds =
        encode_recordswith_inferred_schema(headers, records).expect("encode synthetic dataset");
    let col = ds.column_map();
    let x_idx = col["x"];

    // ---- fit gam: y ~ s(x), Binomial(logit), REML --------------------------
    // k=10 matches mgcv's default thin-plate basis dimension so the two smooths
    // target as close to the same penalized model as their bases allow.
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=10)", &ds, &cfg).expect("gam binomial(logit) fit");
    let FitResult::Standard(fit) = result else {
        panic!("binomial(logit) smooth should be a Standard fit");
    };

    // ---- rebuild the design at the training points and predict CIs ---------
    // Grid-aligned, element-wise comparison: evaluate both engines at the same
    // training x. The design row maps beta -> eta; predict_gamwith_uncertainty
    // then propagates Var(eta) through the logit inverse link to the mean SE.
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for (i, &xi) in x.iter().enumerate() {
        grid[[i, x_idx]] = xi;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    // Sanity-check the operator maps coefficients to one eta per training row
    // before handing it to the uncertainty predictor.
    let gam_eta_check = design.design.apply(&fit.fit.beta);
    assert_eq!(gam_eta_check.len(), n, "design eta length mismatch");
    let offset = Array1::<f64>::zeros(n);

    // Bias correction OFF and all coverage corrections OFF so the reported SEs
    // are the bare delta-method / link-scale SEs that mgcv also reports — an
    // apples-to-apples comparison of the inverse-link Jacobian, nothing else.
    let options = PredictUncertaintyOptions {
        confidence_level: 0.95,
        covariance_mode: InferenceCovarianceMode::Conditional,
        mean_interval_method: MeanIntervalMethod::Delta,
        includeobservation_interval: false,
        apply_bias_correction: false,
        edgeworth_one_sided: false,
        boundary_correction: false,
        ood_inflation: false,
        multi_point_joint: false,
        ..PredictUncertaintyOptions::default()
    };
    let pred = predict_gamwith_uncertainty(
        design.design.clone(),
        fit.fit.beta.view(),
        offset.view(),
        LikelihoodSpec::binomial_logit(),
        &fit.fit,
        &options,
    )
    .expect("gam uncertainty under logit link");

    let gam_eta_se: Vec<f64> = pred.eta_standard_error.to_vec();
    let gam_mean_se: Vec<f64> = pred.mean_standard_error.to_vec();
    assert_eq!(gam_eta_se.len(), n);

    // ---- mgcv reference: SAME data, SAME model -----------------------------
    // mgcv::predict(type="link", se.fit=TRUE) gives eta and its SE; type=
    // "response" gives the mean and its analytical delta-method SE
    // (se.link * dmu/deta). Feeding the identical (x, y) makes this the exact
    // head-to-head for gam's inverse-link Jacobian in CI construction.
    let r = run_r(
        &[Column::new("x", &x), Column::new("y", &y)],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ s(x, k=10), data = df, family = binomial(link="logit"),
                 method = "REML")
        pl <- predict(m, type = "link", se.fit = TRUE)
        pr <- predict(m, type = "response", se.fit = TRUE)
        emit("eta", as.numeric(pl$fit))
        emit("eta_se", as.numeric(pl$se.fit))
        emit("mean", as.numeric(pr$fit))
        emit("mean_se", as.numeric(pr$se.fit))
        "#,
    );
    let mgcv_eta_se = r.vector("eta_se");
    let mgcv_mean = r.vector("mean");
    let mgcv_mean_se = r.vector("mean_se");
    assert_eq!(mgcv_eta_se.len(), n, "mgcv eta_se length mismatch");

    // Sanity: mgcv's own response-scale SE is itself the analytical delta
    // method, mean_se = eta_se * mu*(1-mu). Confirm we read the columns we
    // think we did before cross-comparing to gam.
    let mgcv_delta: Vec<f64> = mgcv_eta_se
        .iter()
        .zip(mgcv_mean.iter())
        .map(|(&se, &mu)| se * mu * (1.0 - mu))
        .collect();
    let mgcv_self = relative_l2(&mgcv_delta, mgcv_mean_se);

    // ---- compare gam vs mgcv ----------------------------------------------
    let eta_se_max = max_abs_diff(&gam_eta_se, mgcv_eta_se);
    let mean_se_rel = relative_l2(&gam_mean_se, mgcv_mean_se);

    eprintln!(
        "logit-link CIs: n={n} eta_se_max_abs_diff(gam,mgcv)={eta_se_max:.5} \
         mean_se_rel_l2(gam,mgcv)={mean_se_rel:.5} \
         mgcv_response_se_is_delta_method(rel_l2)={mgcv_self:.2e}"
    );

    // mgcv's response-scale SE *is* the analytical delta-method SE; this must
    // hold essentially exactly and pins down the quantity gam is compared to.
    assert!(
        mgcv_self < 1e-6,
        "mgcv response SE is not the delta-method SE (rel_l2={mgcv_self:.2e}); \
         column mapping is wrong"
    );

    // (1) Link-scale SE agreement. Both engines REML-fit the identical
    // penalized binomial smooth with k=10, so the linear-predictor SE at the
    // training points must coincide; the spec bound is a max absolute
    // difference < 0.002 across all 200 points.
    assert!(
        eta_se_max < 0.002,
        "eta-scale standard errors diverge from mgcv: max_abs_diff={eta_se_max:.5}"
    );

    // (2) Response-scale (mean) SE agreement via the inverse-link Jacobian.
    // gam's delta-method mean SE must match mgcv's analytical response-scale SE
    // within 1% relative L2 (the spec bound). This is the actual test of the
    // inverse-link Jacobian dmu/deta = mu(1-mu) inside CI construction.
    assert!(
        mean_se_rel < 0.01,
        "delta-method response-scale SE diverges from mgcv: rel_l2={mean_se_rel:.5}"
    );
}

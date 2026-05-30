//! End-to-end quality: gam's *response-scale* prediction standard error under a
//! Gaussian identity link must match mgcv — the mature, standard GAM reference —
//! on real data with a mixed smooth + linear design.
//!
//! Mature comparator: `mgcv::predict.gam(..., se.fit = TRUE, type = "response")`.
//! mgcv is THE reference penalized-GAM implementation; its `predict` SEs are the
//! de-facto standard practitioners trust for confidence intervals.
//!
//! What this pins down. For a Gaussian model with the identity link the inverse
//! link is g⁻¹(η) = η, so its Jacobian dμ/dη ≡ 1 and the response-scale SE is
//! algebraically *identical* to the η-scale SE — no transformation happens. gam
//! reports the response-scale SE in `PredictUncertaintyResult::mean_standard_error`
//! as the standard deviation of μ̂ = g⁻¹(η̂) under the posterior uncertainty in
//! η̂ (`strategy.posterior_meanvariance`); to first order this is the delta-method
//! Var(μ̂) = (dμ/dη)² · Var(η̂). Under the identity link dμ/dη ≡ 1 so that
//! variance collapses to exactly Var(η̂), and `mean_standard_error` must equal
//! `eta_standard_error` element-wise and reproduce mgcv's `type="response"`
//! `se.fit`. A mismatch would expose an algebra error in gam's response-scale SE
//! path (e.g. a wrong Jacobian, a dropped scale factor, or a covariance-mode
//! mismatch). `MeanIntervalMethod::Delta` additionally fixes how the response
//! *interval* (not the SE) is formed — μ̂ ± z · se — keeping it on the same
//! delta-method footing rather than transforming the η interval endpoints.
//!
//! Matching conventions. mgcv's default `predict.gam` SEs use the posterior
//! covariance `Vp` *conditional on the estimated smoothing parameters*
//! (`unconditional = FALSE`). We therefore drive gam with
//! `InferenceCovarianceMode::Conditional` and disable every optional inflation
//! (bias correction, boundary, OOD, Edgeworth) so we compare the same plug-in
//! Bayesian SE on both sides. Both engines REML-fit the identical data, so the
//! two SE vectors must coincide up to numerical precision.
//!
//! The real `prostate.csv` shipped in `bench/datasets` carries columns
//! `pc1, pc2, y`; we fit `y ~ s(pc1) + pc2`, which is exactly the
//! "smooth term + linear term" mixed design the capability targets (a
//! Gaussian/identity linear-probability fit — its statistical merit is
//! irrelevant; the SE *algebra* is what is under test, and that holds for any
//! response values).

use gam::predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, max_abs_diff, relative_l2, run_r};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{
    FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema,
};
use ndarray::Array1;
use std::path::Path;

const PROSTATE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/prostate.csv");

#[test]
fn response_scale_se_matches_mgcv_under_identity_link() {
    init_parallelism();

    // ---- load the real prostate dataset (pc1, pc2 -> y) -------------------
    let ds = load_csvwith_inferred_schema(Path::new(PROSTATE_CSV)).expect("load prostate.csv");
    let col = ds.column_map();
    let pc1_idx = col["pc1"];
    let pc2_idx = col["pc2"];
    let y_idx = col["y"];
    let pc1: Vec<f64> = ds.values.column(pc1_idx).to_vec();
    let pc2: Vec<f64> = ds.values.column(pc2_idx).to_vec();
    let y: Vec<f64> = ds.values.column(y_idx).to_vec();
    let n = pc1.len();
    assert!(n > 100, "prostate should have ~654 rows, got {n}");

    // ---- fit with gam: y ~ s(pc1) + pc2, Gaussian identity, REML ----------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(pc1) + pc2", &ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };

    // Rebuild the frozen design at the *training* rows so predictions align
    // row-for-row with mgcv's in-sample predictions.
    let design = build_term_collection_design(ds.values.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let dense = design.design.to_dense();
    assert_eq!(dense.nrows(), n, "design row count must equal data rows");
    assert_eq!(
        dense.ncols(),
        fit.fit.beta.len(),
        "design columns must match beta length"
    );

    // Response-scale SE via the delta method. Gaussian identity link => the
    // inverse-link Jacobian is exactly 1, so this must equal the eta-scale SE.
    let offset = Array1::<f64>::zeros(n);
    let gaussian_identity =
        LikelihoodSpec::new(ResponseFamily::Gaussian, InverseLink::Standard(StandardLink::Identity));
    let pred = predict_gamwith_uncertainty(
        dense,
        fit.fit.beta.view(),
        offset.view(),
        gaussian_identity,
        &fit.fit,
        &PredictUncertaintyOptions {
            confidence_level: 0.95,
            // Match mgcv's default Vp (posterior covariance conditional on the
            // estimated smoothing parameters; unconditional = FALSE).
            covariance_mode: InferenceCovarianceMode::Conditional,
            // Delta-method response-scale SE — the path under test.
            mean_interval_method: MeanIntervalMethod::Delta,
            includeobservation_interval: false,
            // Compare the same plug-in Bayesian SE both sides: no inflations.
            apply_bias_correction: false,
            edgeworth_one_sided: false,
            boundary_correction: false,
            ood_inflation: false,
            multi_point_joint: false,
            ..PredictUncertaintyOptions::default()
        },
    )
    .expect("gam response-scale uncertainty prediction");

    let gam_mean_se: Vec<f64> = pred.mean_standard_error.to_vec();
    let gam_eta_se: Vec<f64> = pred.eta_standard_error.to_vec();
    assert_eq!(gam_mean_se.len(), n);

    // Internal consistency: under identity link the delta-method response SE
    // must equal the eta SE exactly (Jacobian = 1). This isolates the
    // Jacobian-application bug from any disagreement with mgcv.
    let self_consistency = max_abs_diff(&gam_mean_se, &gam_eta_se);
    eprintln!("gam response-vs-eta SE self-consistency max|Δ| = {self_consistency:.3e}");
    assert!(
        self_consistency < 1e-10,
        "identity-link delta-method response SE must equal eta SE exactly, got max|Δ|={self_consistency:.3e}"
    );

    // ---- fit the SAME model with mgcv and ask for response-scale SEs ------
    let r = run_r(
        &[
            Column::new("pc1", &pc1),
            Column::new("pc2", &pc2),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ s(pc1) + pc2, data = df, family = gaussian(), method = "REML")
        p <- predict(m, newdata = df, se.fit = TRUE, type = "response")
        emit("mean_se", as.numeric(p$se.fit))
        emit("scale", m$scale)
        "#,
    );
    let mgcv_mean_se = r.vector("mean_se");
    assert_eq!(mgcv_mean_se.len(), n, "mgcv se.fit length mismatch");

    // ---- compare element-wise, on the response scale ----------------------
    let max_abs = max_abs_diff(&gam_mean_se, mgcv_mean_se);
    let rel = relative_l2(&gam_mean_se, mgcv_mean_se);
    eprintln!(
        "identity-link response SE vs mgcv: n={n} mgcv_scale={:.5} \
         max_abs_diff={max_abs:.3e} rel_l2={rel:.5}",
        r.scalar("scale")
    );

    // Both engines REML-fit identical data and report the same conditional
    // Bayesian SE; under the identity link no transformation intervenes, so the
    // response-scale SEs must agree to numerical precision. The spec bound
    // (max_abs_diff < 5e-4) is loose enough for cross-engine numerical noise yet
    // tight enough that any Jacobian/scale algebra error would blow past it
    // (in-sample SEs here are O(0.01–0.1)). relative_l2 < 1% is the
    // scale-free companion check.
    assert!(
        max_abs < 5e-4,
        "gam response-scale SE disagrees with mgcv under identity link: max_abs_diff={max_abs:.3e}"
    );
    assert!(
        rel < 0.01,
        "gam response-scale SE diverges from mgcv (relative L2): rel_l2={rel:.5}"
    );
}

//! End-to-end quality: gam's confidence-interval construction under a
//! **non-identity (logistic) link** must be *well-calibrated against the known
//! truth* — its nominal-95% intervals must actually cover the true latent
//! function at the nominal rate. `mgcv` is retained only as a **baseline to
//! match-or-beat** on calibration, never as the thing gam must reproduce.
//!
//! OBJECTIVE METRIC (this is the pass/fail claim):
//!   The data are generated from a *known* latent smooth
//!   `η(x) = x + sin(2πx)`, `μ(x) = sigmoid(η(x))`, `y ~ Bernoulli(μ)`. Because
//!   the truth is known exactly, we measure the **empirical coverage** of gam's
//!   pointwise 95% confidence intervals across the training grid:
//!     * link scale:     fraction of points with `η(xᵢ) ∈ [eta_lowerᵢ, eta_upperᵢ]`
//!     * response scale: fraction of points with `μ(xᵢ) ∈ [mean_lowerᵢ, mean_upperᵢ]`
//!   The Nychka/Marra–Wood result for penalized GAMs is that *across-the-function*
//!   average coverage of the Bayesian credible band tracks the nominal level
//!   (pointwise coverage is smoothing-bias-attenuated, but the grid average is
//!   close to nominal). We therefore assert the across-grid average coverage
//!   lands in a calibration window around 0.95. This is an objective property of
//!   gam's own intervals versus ground truth — it does not depend on mgcv.
//!
//! BASELINE (match-or-beat, not match): mgcv fits the identical penalized
//!   binomial smooth and its `predict(se.fit=TRUE)` band is scored for coverage
//!   against the *same* truth. We additionally require gam's calibration to be
//!   no worse than mgcv's by more than a small margin, i.e. gam's
//!   |coverage − nominal| ≤ mgcv's |coverage − nominal| + margin. This demotes
//!   the mature tool to a yardstick on the objective metric; gam is never asked
//!   to reproduce mgcv's noisy SEs.
//!
//! Why a Binomial(logit) model: this is the family that actually exercises gam's
//! inverse-link Jacobian `dμ/dη = μ(1−μ)` inside CI construction (the Gaussian
//! posterior-variance branch ignores the link entirely). The latent smooth
//! `η(x) = x + sin(2πx)`, design `x ~ U[-3,3]`, n=200, seed=123 follow the spec
//! verbatim; Bernoulli sampling replaces the spec's Gaussian noise so the
//! generative model is internally consistent with the logit link.
//!
//! Identical data feed both engines (the same CSV columns). Bounds are not
//! weakened to force a pass: a genuinely mis-calibrated band failing here is a
//! real bug.

use gam::estimate::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, run_r};
use gam::types::LikelihoodSpec;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};

/// Deterministic synthetic logistic-link dataset, generated identically on the
/// Rust side and serialized to the CSV both engines read. Returns
/// `(x, y, eta_true, mu_true)` where `eta_true`/`mu_true` are the *exact*
/// data-generating latent function values at each `x` — the ground truth the
/// confidence intervals must cover.
///
/// `x ~ U[-3,3]`, `η(x) = x + sin(2πx)`, `p = sigmoid(η)`, `y ~ Bernoulli(p)`,
/// seed = 123. The RNG is a fixed split-mix64 stream so the data are bit-for-bit
/// reproducible and feed *both* gam and mgcv through the same CSV.
fn synthetic_logistic_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
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
    let mut eta_true = Vec::with_capacity(n);
    let mut mu_true = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = -3.0 + 6.0 * next_u01();
        let eta = xi + (2.0 * std::f64::consts::PI * xi).sin();
        let p = 1.0 / (1.0 + (-eta).exp());
        let yi = if next_u01() < p { 1.0 } else { 0.0 };
        x.push(xi);
        y.push(yi);
        eta_true.push(eta);
        mu_true.push(p);
    }
    (x, y, eta_true, mu_true)
}

/// Fraction of grid points whose nominal-95% interval `[lo, hi]` covers the
/// known truth `t`. This is the objective calibration metric: with correctly
/// calibrated bands the across-grid average lands near the nominal level.
fn empirical_coverage(lo: &[f64], hi: &[f64], truth: &[f64]) -> f64 {
    assert_eq!(lo.len(), hi.len(), "coverage bound length mismatch");
    assert_eq!(lo.len(), truth.len(), "coverage truth length mismatch");
    let hits = lo
        .iter()
        .zip(hi.iter())
        .zip(truth.iter())
        .filter(|((&l, &h), &t)| t >= l && t <= h)
        .count();
    hits as f64 / lo.len().max(1) as f64
}

#[test]
fn confidence_intervals_cover_truth_under_logistic_link() {
    init_parallelism();

    // ---- deterministic synthetic data (spec: n=200, x~U[-3,3], seed=123) ---
    let n = 200usize;
    let (x, y, eta_true, mu_true) = synthetic_logistic_data(n);
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

    // gam's nominal-95% interval bounds, both scales, at the training grid.
    let gam_eta_lower: Vec<f64> = pred.eta_lower.to_vec();
    let gam_eta_upper: Vec<f64> = pred.eta_upper.to_vec();
    let gam_mean_lower: Vec<f64> = pred.mean_lower.to_vec();
    let gam_mean_upper: Vec<f64> = pred.mean_upper.to_vec();
    assert_eq!(gam_eta_lower.len(), n);

    // ---- OBJECTIVE METRIC: empirical coverage of the KNOWN truth -----------
    // The latent function is known exactly (eta_true / mu_true). Score the
    // across-grid average coverage of gam's own intervals against it.
    let gam_eta_cov = empirical_coverage(&gam_eta_lower, &gam_eta_upper, &eta_true);
    let gam_mean_cov = empirical_coverage(&gam_mean_lower, &gam_mean_upper, &mu_true);

    // ---- mgcv BASELINE (match-or-beat on calibration): SAME data/model -----
    // mgcv::predict(se.fit=TRUE) yields eta/mu and their SEs; we form its
    // nominal-95% band (fit ± 1.96·se) on each scale and score IT for coverage
    // against the SAME truth. mgcv is a yardstick on the objective metric, not
    // the thing gam must reproduce.
    let z = 1.959_963_984_540_054_f64; // qnorm(0.975)
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
    let mgcv_eta = r.vector("eta");
    let mgcv_eta_se = r.vector("eta_se");
    let mgcv_mean = r.vector("mean");
    let mgcv_mean_se = r.vector("mean_se");
    assert_eq!(mgcv_eta_se.len(), n, "mgcv eta_se length mismatch");

    let mgcv_eta_lower: Vec<f64> = mgcv_eta
        .iter()
        .zip(mgcv_eta_se.iter())
        .map(|(&f, &se)| f - z * se)
        .collect();
    let mgcv_eta_upper: Vec<f64> = mgcv_eta
        .iter()
        .zip(mgcv_eta_se.iter())
        .map(|(&f, &se)| f + z * se)
        .collect();
    let mgcv_mean_lower: Vec<f64> = mgcv_mean
        .iter()
        .zip(mgcv_mean_se.iter())
        .map(|(&f, &se)| f - z * se)
        .collect();
    let mgcv_mean_upper: Vec<f64> = mgcv_mean
        .iter()
        .zip(mgcv_mean_se.iter())
        .map(|(&f, &se)| f + z * se)
        .collect();
    let mgcv_eta_cov = empirical_coverage(&mgcv_eta_lower, &mgcv_eta_upper, &eta_true);
    let mgcv_mean_cov = empirical_coverage(&mgcv_mean_lower, &mgcv_mean_upper, &mu_true);

    // Context only (NOT a pass criterion): how close gam's SEs sit to mgcv's.
    let eta_se_rel = relative_l2(&pred.eta_standard_error.to_vec(), mgcv_eta_se);
    let mean_se_rel = relative_l2(&pred.mean_standard_error.to_vec(), mgcv_mean_se);

    let nominal = 0.95_f64;
    eprintln!(
        "logit-link CI calibration: n={n} nominal={nominal:.2}\n  \
         link-scale  coverage: gam={gam_eta_cov:.3} mgcv={mgcv_eta_cov:.3}\n  \
         resp-scale  coverage: gam={gam_mean_cov:.3} mgcv={mgcv_mean_cov:.3}\n  \
         (context) se_rel_l2(gam,mgcv): eta={eta_se_rel:.4} mean={mean_se_rel:.4}"
    );

    // (1) OBJECTIVE: gam's across-grid average coverage tracks the nominal
    // level. Pointwise penalized-GAM bands are smoothing-bias attenuated, so
    // the calibration window is centered on 0.95 with a tolerance that admits
    // the expected attenuation but rejects a badly mis-scaled band. This claim
    // is about gam vs ground truth and does not involve mgcv at all.
    let cov_window = 0.12_f64; // 0.95 ± 0.12  ->  [0.83, 1.00]
    assert!(
        (gam_eta_cov - nominal).abs() <= cov_window,
        "link-scale 95% CI mis-calibrated vs truth: coverage={gam_eta_cov:.3} \
         (nominal {nominal:.2}, window ±{cov_window:.2})"
    );
    assert!(
        (gam_mean_cov - nominal).abs() <= cov_window,
        "response-scale 95% CI mis-calibrated vs truth: coverage={gam_mean_cov:.3} \
         (nominal {nominal:.2}, window ±{cov_window:.2})"
    );

    // (2) MATCH-OR-BEAT mgcv on calibration: gam's distance from nominal must
    // be no worse than mgcv's by more than a small margin, on both scales.
    let beat_margin = 0.03_f64;
    let gam_eta_err = (gam_eta_cov - nominal).abs();
    let mgcv_eta_err = (mgcv_eta_cov - nominal).abs();
    assert!(
        gam_eta_err <= mgcv_eta_err + beat_margin,
        "link-scale calibration worse than mgcv baseline: |gam−nom|={gam_eta_err:.3} > \
         |mgcv−nom|={mgcv_eta_err:.3} + {beat_margin:.2}"
    );
    let gam_mean_err = (gam_mean_cov - nominal).abs();
    let mgcv_mean_err = (mgcv_mean_cov - nominal).abs();
    assert!(
        gam_mean_err <= mgcv_mean_err + beat_margin,
        "response-scale calibration worse than mgcv baseline: |gam−nom|={gam_mean_err:.3} > \
         |mgcv−nom|={mgcv_mean_err:.3} + {beat_margin:.2}"
    );
}

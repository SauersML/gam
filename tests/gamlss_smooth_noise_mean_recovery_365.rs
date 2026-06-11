//! Regression for issue #365: a Gaussian location-scale (GAMLSS) fit with a
//! *smooth* `noise_formula` must not destroy the mean fit.
//!
//! Defect under test
//! -----------------
//! On homoscedastic data (true log-scale constant), fitting `y ~ s(x)` with
//! `noise_formula = "s(x)"` routes through the exact two-block joint Newton
//! inner solve. That solve builds the *observed* joint (μ, log σ) Hessian,
//! which is genuinely indefinite away from the optimum. The spectral inner
//! solver used to hard-error on any negative eigenvalue ("reduced penalized
//! Hessian is indefinite … exact Newton requires a PSD quadratic model"),
//! which — where it did not abort outright — terminated at a badly underfit
//! mean (mean RMSE ≈ 1.5 against truth, versus ≈ 0.03 for a plain GAM or a
//! *linear* noise term) at small/medium n. The headline complaint is the
//! ruined mean, not just the crash.
//!
//! The fix (modified-Newton eigenvalue reflection in
//! `solve_joint_newton_step_on_spectral_range`) reflects each negative
//! eigenvalue to |λ| so the joint Newton step is always a descent direction
//! the trust-region globalization can accept, letting the joint solve walk to
//! the correct mean coefficients instead of stalling.
//!
//! Why this test fails on the broken code
//! --------------------------------------
//! This exercises the REAL end-to-end public path (`materialize` →
//! `fit_model` → mean-block coefficients × in-sample mean design), reproducing
//! the issue's exact data-generating process (n = 2000, het = 0, true mean
//! `1 + 0.7 x + sin(x)`). On the pre-fix code the smooth-noise joint solve
//! either errors or stalls with mean RMSE ≈ 1.5; the assertion below requires
//! RMSE within roughly the homoscedastic noise floor (≤ 0.20, far below the
//! 1.5 underfit and comfortably above the ~0.03–0.06 a healthy fit reaches),
//! so the broken behavior fails and the fixed behavior passes. The tolerance
//! is intentionally well inside the broken/healthy gap — it is NOT loosened to
//! whitelist the bug.

use csv::StringRecord;
use gam::solver::estimate::BlockRole;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_model, init_parallelism,
    materialize,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Known true mean function from the issue repro.
fn true_mean(x: f64) -> f64 {
    1.0 + 0.7 * x + x.sin()
}

/// Reproduce the issue's deterministic homoscedastic data generator:
/// x ~ Uniform(-3, 3), y = (1 + 0.7 x + sin x) + sigma * N(0, 1),
/// with sigma = exp(-0.5) constant (het = 0 ⇒ homoscedastic).
fn make_issue_data(n: usize, seed: u64) -> (gam::data::EncodedDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(-3.0, 3.0).expect("uniform x");
    let noise = Normal::new(0.0, 1.0).expect("standard normal");
    let sigma = (-0.5f64).exp(); // constant log-scale of -0.5

    let mut headers = Vec::new();
    headers.push("y".to_string());
    headers.push("x".to_string());

    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    let mut truth = Vec::with_capacity(n);
    for _ in 0..n {
        let x = ux.sample(&mut rng);
        let mean = true_mean(x);
        let y = mean + sigma * noise.sample(&mut rng);
        truth.push(mean);
        rows.push(StringRecord::from(vec![y.to_string(), x.to_string()]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode issue data");
    (data, truth)
}

#[test]
fn smooth_noise_formula_recovers_mean_on_homoscedastic_data_365() {
    init_parallelism();

    // Issue repro: n = 2000 (the small/medium-n regime where the joint solve
    // failed), het = 0, deterministic seed = 1.
    let n = 2000usize;
    let (data, truth) = make_issue_data(n, 1);

    // Gaussian location-scale with a SMOOTH noise formula — the exact
    // configuration that destroyed the mean fit.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("s(x)".to_string()),
        ..FitConfig::default()
    };
    let materialized = materialize("y ~ s(x)", &data, &cfg).expect("materialize smooth-noise GLS");

    // On the broken code this either returns an indefinite-Hessian error or a
    // stalled underfit; both are caught here (an `expect` failure on the error
    // path, an RMSE assertion on the underfit path).
    let result = fit_model(materialized.request).expect(
        "smooth-noise Gaussian location-scale fit must converge (issue #365: previously aborted \
         with 'reduced penalized Hessian is indefinite')",
    );
    let FitResult::GaussianLocationScale(fit) = result else {
        panic!("expected a GaussianLocationScale fit result for a Gaussian noise_formula model");
    };

    // Reconstruct the in-sample plug-in mean: η_mean = X_mean · β_mean. The
    // Gaussian mean uses the identity link, so the response-scale mean equals
    // η_mean. This is the same quantity the issue measured via
    // `predict(df)["mean"]`.
    let mean_block = fit
        .fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("Gaussian location-scale fit must carry a Location (mean) coefficient block");
    let beta_mean = &mean_block.beta;
    let mean_design = fit.fit.mean_design.design.to_dense();
    assert_eq!(
        mean_design.ncols(),
        beta_mean.len(),
        "mean design columns ({}) must match mean coefficient count ({})",
        mean_design.ncols(),
        beta_mean.len(),
    );
    let eta_mean = mean_design.dot(beta_mean);
    let pred_mean: Vec<f64> = eta_mean.to_vec();

    // The smooth scale model must actually be present and retrievable — the
    // secondary defect in the issue was that the learned scale was unreachable.
    // A smooth `noise_formula="s(x)"` materializes a multi-column noise basis,
    // so the fitted Scale block must carry more than a lone intercept; this is
    // the same coefficient block `predict_noise_scale` (and the FFI
    // `noise_scale` column) reads to expose σ to callers.
    let scale_block =
        fit.fit.fit.block_by_role(BlockRole::Scale).expect(
            "smooth noise_formula must fit a retrievable Scale (log-sigma) coefficient block",
        );
    assert!(
        scale_block.beta.len() >= 2,
        "smooth noise_formula must materialize a multi-coefficient scale basis, got {} \
         coefficient(s)",
        scale_block.beta.len(),
    );
    assert_eq!(
        fit.fit.noise_design.design.ncols(),
        scale_block.beta.len(),
        "noise design columns ({}) must match scale coefficient count ({})",
        fit.fit.noise_design.design.ncols(),
        scale_block.beta.len(),
    );

    let mean_rmse = rmse(&pred_mean, &truth);
    eprintln!(
        "[issue-365] smooth-noise mean RMSE = {mean_rmse:.4} (broken ≈ 1.53, healthy ≈ 0.03)"
    );

    // The broken joint solve stalls at mean RMSE ≈ 1.53 (issue: 1.5279 /
    // 1.5581 / 1.5624 across seeds). A healthy fit reaches ≈ 0.03–0.06. The
    // 0.20 bar sits firmly between the two: it fails on the underfit and
    // passes once the mean is recovered, without ever weakening to admit the
    // bug.
    assert!(
        mean_rmse < 0.20,
        "issue #365 not fixed: smooth `noise_formula` underfits the mean (RMSE {mean_rmse:.4} ≥ \
         0.20; broken behavior is ≈ 1.53, a healthy fit reaches ≈ 0.03–0.06)",
    );
}

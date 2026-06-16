//! Regression lock for issue #771: Tweedie regression must estimate the
//! dispersion `φ` from the data rather than freezing it at 1.0.
//!
//! For a Tweedie GLM the response variance is `Var(y) = φ · μ^p` with `φ` a
//! genuine free dispersion parameter (this is the whole point of the family —
//! it models over-dispersed, zero-inflated positive data). Unlike Binomial and
//! Poisson, whose variance is fully pinned by the mean (`φ ≡ 1`), Tweedie's `φ`
//! must be estimated, exactly as Gamma's shape and the Gaussian residual
//! variance are. mgcv's `tw()` / statsmodels' Tweedie both estimate it.
//!
//! The original bug was rooted in `LikelihoodSpec::default_scale_metadata`
//! (src/types.rs), which lumped `Tweedie` in with `Binomial | Poisson` and
//! returned `FixedDispersion { phi: 1.0 }`. The fix makes Tweedie use
//! `EstimatedTweediePhi`, refreshed from converged-η Pearson residuals, so the
//! IRLS weight and fitted coefficient covariance reflect the data's dispersion.
//!
//! This test fits two Tweedie datasets that share the *same covariate values*
//! and the *same mean structure* but very different true dispersions
//! (`φ = 0.3` vs `φ = 8.0`). Because the working weights `W = μ^{2-p}` and the
//! design `X` are essentially identical across the two fits, the only thing that
//! should move the linear-predictor SE is `√φ`. A correct fit therefore yields
//! `mean SE(η̂)_hi / mean SE(η̂)_lo ≈ √(8.0/0.3) ≈ 5.2`. With `φ` frozen at 1.0
//! the ratio collapses to ~1.0 — the high-dispersion model reports the same
//! (wildly over-confident) uncertainty as the low-dispersion one.
//!
//! The test asserts the ratio is at least 2.0, which cleanly separates the
//! correct behaviour (~5) from the frozen-φ behaviour (~1) with ample margin
//! for Monte-Carlo noise in the two realisations.

use csv::StringRecord;
use gam::predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};

const B0: f64 = 1.0;
const BX: f64 = 0.5;
const TWEEDIE_P: f64 = 1.5;

/// Draw a single compound Poisson–Gamma (Tweedie, `1 < p < 2`) response with
/// mean `mu` and dispersion `phi`.
fn tweedie_sample(rng: &mut StdRng, mu: f64, phi: f64) -> f64 {
    let p = TWEEDIE_P;
    let lambda = mu.powf(2.0 - p) / (phi * (2.0 - p));
    let alpha = (2.0 - p) / (p - 1.0);
    let scale = phi * (p - 1.0) * mu.powf(p - 1.0);
    let n: u64 = Poisson::new(lambda).expect("poisson rate").sample(rng) as u64;
    if n == 0 {
        return 0.0;
    }
    // Sum of `n` iid Gamma(alpha, scale) is Gamma(n*alpha, scale).
    Gamma::new(alpha * n as f64, scale)
        .expect("gamma(shape,scale)")
        .sample(rng)
}

/// Fit `y ~ x` as a Tweedie(log) model on `(x, y)` and return the per-point
/// linear-predictor standard errors on the supplied evaluation grid, plus the
/// fitted dispersion `φ` the covariance was scaled by.
fn fit_tweedie_eta_se(x: &[f64], y: &[f64], eval: &[f64]) -> Option<(Vec<f64>, f64)> {
    let n = x.len();
    let headers: Vec<String> = ["y", "x"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![y[i].to_string(), x[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).ok()?;
    let col = ds.column_map();
    let x_idx = col["x"];

    let cfg = FitConfig {
        family: Some("tweedie".to_string()),
        ..FitConfig::default()
    };
    let FitResult::Standard(fit) = fit_from_formula("y ~ x", &ds, &cfg).ok()? else {
        return None;
    };

    let m = eval.len();
    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for (i, &xi) in eval.iter().enumerate() {
        grid[[i, x_idx]] = xi;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec).ok()?;
    let dense = design.design.to_dense();

    let tweedie_log = LikelihoodSpec::new(
        ResponseFamily::Tweedie { p: TWEEDIE_P },
        InverseLink::Standard(StandardLink::Log),
    );
    let offset = Array1::<f64>::zeros(m);
    let pred = predict_gamwith_uncertainty(
        dense,
        fit.fit.beta.view(),
        offset.view(),
        tweedie_log,
        &fit.fit,
        &PredictUncertaintyOptions {
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
        },
    )
    .ok()?;

    let se = pred.eta_standard_error.to_vec();
    let phi = fit
        .fit
        .inference
        .as_ref()
        .map(|inf| inf.dispersion.phi())
        .unwrap_or(f64::NAN);
    Some((se, phi))
}

#[test]
fn tweedie_prediction_se_scales_with_dispersion() {
    init_parallelism();

    let n = 4000usize;
    let phi_lo = 0.3_f64;
    let phi_hi = 8.0_f64;

    // Shared covariate values, so the only inferential difference between the
    // two fits is the true dispersion.
    let mut rng = StdRng::seed_from_u64(0x7_3D1E_u64);
    let ux = Uniform::new(-1.0_f64, 1.0_f64).expect("uniform -1..1");
    let x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();

    let mut rng_lo = StdRng::seed_from_u64(11);
    let mut rng_hi = StdRng::seed_from_u64(22);
    let mut y_lo = Vec::with_capacity(n);
    let mut y_hi = Vec::with_capacity(n);
    for &xi in &x {
        let mu = (B0 + BX * xi).exp();
        y_lo.push(tweedie_sample(&mut rng_lo, mu, phi_lo));
        y_hi.push(tweedie_sample(&mut rng_hi, mu, phi_hi));
    }

    let eval: Vec<f64> = vec![-0.8, -0.4, 0.0, 0.4, 0.8];

    let (se_lo, phi_fit_lo) =
        fit_tweedie_eta_se(&x, &y_lo, &eval).expect("low-dispersion Tweedie fit should succeed");
    let (se_hi, phi_fit_hi) =
        fit_tweedie_eta_se(&x, &y_hi, &eval).expect("high-dispersion Tweedie fit should succeed");

    for (s, tag) in [(&se_lo, "lo"), (&se_hi, "hi")] {
        assert!(
            s.iter().all(|v| v.is_finite() && *v > 0.0),
            "non-finite/zero η SE in {tag} fit: {s:?}"
        );
    }

    let mean_lo: f64 = se_lo.iter().sum::<f64>() / se_lo.len() as f64;
    let mean_hi: f64 = se_hi.iter().sum::<f64>() / se_hi.len() as f64;
    let ratio = mean_hi / mean_lo;
    let expected_ratio = (phi_hi / phi_lo).sqrt();

    eprintln!(
        "[tweedie-φ] true φ: lo={phi_lo} hi={phi_hi}; fitted φ: lo={phi_fit_lo:.4} hi={phi_fit_hi:.4}; \
         mean η SE: lo={mean_lo:.5} hi={mean_hi:.5}; SE ratio={ratio:.3} \
         (correct ≈ {expected_ratio:.3}; frozen-φ bug ≈ 1.0)"
    );

    // A correct Tweedie fit scales SE(η̂) by √φ, so the high-dispersion model's
    // SEs must be markedly larger. The frozen-φ=1.0 bug makes both fits report
    // essentially the same SE (ratio ≈ 1.0).
    assert!(
        ratio >= 2.0,
        "Tweedie prediction SE does not grow with the data's dispersion: \
         high-φ/low-φ SE ratio = {ratio:.3} (expected ≈ {expected_ratio:.3} for \
         true φ {phi_hi} vs {phi_lo}). The dispersion is frozen at φ=1.0 \
         (fitted φ: lo={phi_fit_lo:.4}, hi={phi_fit_hi:.4}), so the high-dispersion \
         model reports the same over-confident uncertainty as the low-dispersion one."
    );
}

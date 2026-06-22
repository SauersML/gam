//! Bug-hunt lock (#817): a fitted Gamma-regression's **response-scale
//! observation (prediction) interval** must be built from the *conditional
//! Gamma distribution*, not a symmetric `μ ± z·σ` normal band.
//!
//! The Gamma is strongly right-skewed. A symmetric band gets the predictive
//! *width* right (the fitted dispersion `φ` feeds `Var(Y|μ) = φμ²`) but its
//! *shape* wrong: the symmetric lower edge `μ·(1 − z/√k)` hugs the support
//! floor (for shape `k = 4`, `≈ 0.02·μ`) while the true 2.5% Gamma quantile sits
//! well above it. So the lower tail captures essentially the whole lower side of
//! the distribution (lower-tail freq ≈ 0) and the truncated upper edge then
//! overshoots the nominal level (upper-tail freq ≈ 0.047) — even though *total*
//! coverage lands near 0.95 by accident.
//!
//! The fix builds the band from equal-tailed quantiles of a moment-matched
//! Gamma predictive (mean `μ`, variance `SE(μ̂)² + φμ²`), which is the exact
//! conditional Gamma when estimation uncertainty is negligible. Then both tails
//! converge to ≈ 0.025.
//!
//! This is distinct from the dispersion-*width* bugs (#771 Tweedie, #801 Beta,
//! #802 NegBin) and from #800 (Poisson lower bound below support): here the
//! variance is correct and the lower edge stays positive, so the miscoverage is
//! silent.

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
use rand_distr::{Distribution, Gamma, Uniform};

const TRUE_SHAPE: f64 = 4.0; // CV = 1/√k = 0.5, clearly right-skewed

/// Draw `n` Gamma observations with a smoothly varying positive mean
/// `μ(x) = exp(0.5 + sin(2πx))`, shape `k = 4` (scale `μ/k`).
fn gen_gamma(n: usize, rng: &mut StdRng) -> (Vec<f64>, Vec<f64>) {
    let ux = Uniform::new(0.0_f64, 1.0_f64).expect("uniform");
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = ux.sample(rng);
        let mu = (0.5 + (2.0 * std::f64::consts::PI * xi).sin()).exp();
        let yi = Gamma::new(TRUE_SHAPE, mu / TRUE_SHAPE)
            .expect("gamma params")
            .sample(rng)
            .max(1.0e-12);
        x.push(xi);
        y.push(yi);
    }
    (x, y)
}

#[test]
fn gamma_observation_interval_covers_each_tail_not_just_the_total() {
    init_parallelism();

    let mut rng = StdRng::seed_from_u64(0x_6A_17_A_u64);
    let (xtr, ytr) = gen_gamma(3000, &mut rng);
    let (xev, yev) = gen_gamma(15000, &mut rng);

    // --- fit y ~ s(x) with a Gamma(log) family ------------------------------
    let headers: Vec<String> = ["y", "x"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..xtr.len())
        .map(|i| StringRecord::from(vec![ytr[i].to_string(), xtr[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode gamma training data");
    let col = ds.column_map();
    let x_idx = col["x"];

    let cfg = FitConfig {
        family: Some("gamma".to_string()),
        ..FitConfig::default()
    };
    let FitResult::Standard(fit) =
        fit_from_formula("y ~ s(x)", &ds, &cfg).expect("gamma fit should succeed")
    else {
        panic!("expected a Standard Gamma fit");
    };

    let phi_hat = fit.fit.dispersion_phi();
    eprintln!(
        "[gamma-obs] dispersion φ̂ = {phi_hat:.4} (shape ≈ {:.3}; true shape {TRUE_SHAPE})",
        1.0 / phi_hat
    );

    // --- predict the 95% observation interval on the 15k held-out sample ----
    let m = xev.len();
    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for (i, &xi) in xev.iter().enumerate() {
        grid[[i, x_idx]] = xi;
    }
    let design =
        build_term_collection_design(grid.view(), &fit.resolvedspec).expect("design at eval grid");
    let dense = design.design.to_dense();

    let gamma_log = LikelihoodSpec::new(
        ResponseFamily::Gamma,
        InverseLink::Standard(StandardLink::Log),
    );
    let offset = Array1::<f64>::zeros(m);
    let pred = predict_gamwith_uncertainty(
        dense,
        fit.fit.beta.view(),
        offset.view(),
        gamma_log,
        &fit.fit,
        &PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::Conditional,
            mean_interval_method: MeanIntervalMethod::Delta,
            includeobservation_interval: true,
            apply_bias_correction: false,
            edgeworth_one_sided: false,
            boundary_correction: false,
            ood_inflation: false,
            multi_point_joint: false,
            ..PredictUncertaintyOptions::default()
        },
    )
    .expect("gamma predict with observation interval");

    let lower = pred.observation_lower.expect("observation interval");
    let upper = pred.observation_upper.expect("observation interval");

    // The bounds must be ordered, finite, and strictly inside the positive
    // support (the lower edge must NOT hug zero — that is the defect's signature).
    for i in 0..m {
        assert!(
            lower[i].is_finite() && upper[i].is_finite() && lower[i] > 0.0 && upper[i] > lower[i],
            "row {i}: degenerate observation bounds [{}, {}]",
            lower[i],
            upper[i]
        );
    }

    let n = m as f64;
    let lower_tail = (0..m).filter(|&i| yev[i] < lower[i]).count() as f64 / n;
    let upper_tail = (0..m).filter(|&i| yev[i] > upper[i]).count() as f64 / n;
    let total = (0..m)
        .filter(|&i| yev[i] >= lower[i] && yev[i] <= upper[i])
        .count() as f64
        / n;

    eprintln!(
        "[gamma-obs] lower-tail freq = {lower_tail:.4} (nominal 0.025), \
         upper-tail freq = {upper_tail:.4} (nominal 0.025), \
         total coverage = {total:.4} (nominal 0.95)"
    );

    // Both tails must land near the nominal 0.025. The symmetric-band bug pins
    // the lower tail at ≈ 0.000 and pushes the upper tail to ≈ 0.047; an
    // equal-tailed Gamma interval brings both to ≈ 0.025. The [0.013, 0.040]
    // window absorbs sampling noise at n = 15k while still catching the bug
    // (whose lower tail of 0.000 and upper tail of 0.047 both fall outside).
    assert!(
        (0.013..=0.040).contains(&lower_tail),
        "Gamma observation interval lower tail mis-covers: freq = {lower_tail:.4} \
         (nominal 0.025). A symmetric μ ± z·σ band leaves the lower edge near \
         0.02·μ, far below the true 2.5% Gamma quantile, so the lower tail \
         collapses toward 0."
    );
    assert!(
        (0.013..=0.040).contains(&upper_tail),
        "Gamma observation interval upper tail mis-covers: freq = {upper_tail:.4} \
         (nominal 0.025). A symmetric band that under-extends the lower edge \
         over-extends the truncated upper edge, pushing the upper tail above \
         nominal."
    );
    // Total coverage should still be near nominal (it was near-nominal even with
    // the bug — the point is each *tail* must also be right).
    assert!(
        (0.93..=0.965).contains(&total),
        "Gamma observation interval total coverage = {total:.4} (nominal 0.95)"
    );
}

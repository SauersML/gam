use gam::estimate::{FitOptions, fit_gam};
use gam::pirls::update_glmvectors_by_family;
use gam::predict::predict_gam;
use gam::probability::normal_cdf;
use gam::smooth::BlockwisePenalty;
use gam::types::{GlmLikelihoodFamily, GlmLikelihoodSpec, LikelihoodFamily};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[test]
fn probit_fit_and_predict_fast_integration() {
    let n = 400usize;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    let mut rng = StdRng::seed_from_u64(7);

    for i in 0..n {
        let xi = -2.0 + 4.0 * (i as f64) / (n as f64 - 1.0);
        let eta = -0.3 + 1.1 * xi;
        let p = normal_cdf(eta);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = xi;
        y[i] = if rng.random::<f64>() < p { 1.0 } else { 0.0 };
    }

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let mut s = Array2::<f64>::zeros((2, 2));
    s[[1, 1]] = 1.0;
    let s_list = vec![BlockwisePenalty::new(0..2, s)];

    let fit = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &s_list,
        LikelihoodFamily::BinomialProbit,
        &FitOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            max_iter: 60,
            tol: 1e-6,
            nullspace_dims: vec![1],
            linear_constraints: None,
            firth_bias_reduction: false,
            adaptive_regularization: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
        },
    )
    .expect("probit fit should succeed");

    assert_eq!(fit.beta.len(), 2);
    assert_eq!(fit.lambdas.len(), 1);
    assert!(fit.edf_total().is_some_and(f64::is_finite));

    let pred = predict_gam(
        x.view(),
        fit.beta.view(),
        offset.view(),
        LikelihoodFamily::BinomialProbit,
    )
    .expect("probit predict should succeed");

    assert!(
        pred.mean
            .iter()
            .all(|v: &f64| v.is_finite() && *v >= 0.0 && *v <= 1.0)
    );

    let brier = (&pred.mean - &y)
        .mapv(|v| v * v)
        .mean()
        .unwrap_or(f64::INFINITY);
    assert!(
        brier < 0.25,
        "unexpectedly poor probit fit: brier={brier:.6e}"
    );
}

#[test]
fn probitworkingvectors_are_finite_for_extreme_eta() {
    // Eta laid out so we can index the limit cases directly:
    //   0 -> -100 (saturated low), 1 -> -20, 2 -> 0 (peak weight),
    //   3 -> +20, 4 -> +100 (saturated high)
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0]);
    let eta = Array1::from_vec(vec![-100.0, -20.0, 0.0, 20.0, 100.0]);
    let w = Array1::ones(y.len());
    let mut mu = Array1::zeros(y.len());
    let mut weights = Array1::zeros(y.len());
    let mut z = Array1::zeros(y.len());

    update_glmvectors_by_family(
        y.view(),
        &eta,
        GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialProbit),
        w.view(),
        &mut mu,
        &mut weights,
        &mut z,
    )
    .expect("probit working-vector update should succeed");

    // --- Finiteness / bounds (preserved from the original smoke test) ---
    assert!(
        mu.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0),
        "probit mu out of [0,1] or non-finite: mu={mu:?}"
    );
    assert!(
        weights.iter().all(|v| v.is_finite() && *v >= 0.0),
        "probit weights non-finite or negative: weights={weights:?}"
    );
    assert!(
        z.iter().all(|v| v.is_finite()),
        "probit z non-finite: z={z:?}"
    );

    // --- Mathematical contract for Φ ---
    // mu must implement the standard normal CDF, so the saturated tails
    // collapse to {0, 1} and the centered eta gives 0.5 exactly.
    assert!(
        mu[0] < 1e-12,
        "Φ(-100) must collapse to ~0; got mu[0]={}",
        mu[0]
    );
    assert!(mu[1] < 1e-6, "Φ(-20) must be tiny; got mu[1]={}", mu[1]);
    assert!(
        (mu[2] - 0.5).abs() < 1e-12,
        "Φ(0) must equal 0.5 exactly within fp tol; got mu[2]={}",
        mu[2]
    );
    assert!(mu[3] > 1.0 - 1e-6, "Φ(+20) must be ~1; got mu[3]={}", mu[3]);
    assert!(
        mu[4] > 1.0 - 1e-12,
        "Φ(+100) must collapse to ~1; got mu[4]={}",
        mu[4]
    );

    // mu must be monotonically non-decreasing in eta (probit link is
    // strictly increasing).
    for i in 1..mu.len() {
        assert!(
            mu[i] >= mu[i - 1] - 1e-15,
            "probit mu must be non-decreasing in eta; mu[{i}]={mu_i} < mu[{prev_i}]={mu_prev}",
            mu_i = mu[i],
            prev_i = i - 1,
            mu_prev = mu[i - 1]
        );
    }

    // --- IRLS weight contract ---
    // The probit IRLS weight is φ(η)² / [Φ(η)(1-Φ(η))], which is largest
    // near η = 0 and tends to 0 as |η| → ∞. We don't pin a specific value
    // (the implementation may clamp small denominators), but the qualitative
    // ordering must hold.
    assert!(
        weights[2] > weights[0],
        "probit weight at eta=0 must exceed weight at eta=-100; w[0]={}, w[2]={}",
        weights[0],
        weights[2]
    );
    assert!(
        weights[2] > weights[4],
        "probit weight at eta=0 must exceed weight at eta=+100; w[2]={}, w[4]={}",
        weights[2],
        weights[4]
    );
    assert!(
        weights[2] > weights[1],
        "probit weight at eta=0 must exceed weight at eta=-20; w[1]={}, w[2]={}",
        weights[1],
        weights[2]
    );
    assert!(
        weights[2] > weights[3],
        "probit weight at eta=0 must exceed weight at eta=+20; w[3]={}, w[2]={}",
        weights[3],
        weights[2]
    );

    // --- Working-response sign contract ---
    // For a saturated η (μ ≈ y), the residual (y-μ) is tiny; combined with a
    // tiny weight, z is allowed to be wide, but the sign of (z - η) must
    // match the sign of (y - μ): if y is the larger class, z should pull
    // upward (z - η > 0), otherwise downward.
    for i in 0..y.len() {
        let residual = y[i] - mu[i];
        if residual.abs() > 1e-9 {
            let pull = z[i] - eta[i];
            assert!(
                pull * residual >= -1e-12,
                "probit working response must pull η toward y on row {i}: \
                 y={}, mu={}, eta={}, z={}, pull={}, residual={}",
                y[i],
                mu[i],
                eta[i],
                z[i],
                pull,
                residual,
            );
        }
    }
}

#[test]
fn cloglog_fit_and_predict_fast_integration() {
    let n = 400usize;
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    let mut rng = StdRng::seed_from_u64(17);

    for i in 0..n {
        let xi = -2.0 + 4.0 * (i as f64) / (n as f64 - 1.0);
        let eta = -0.4 + 0.9 * xi;
        let z = eta.clamp(-30.0, 30.0);
        let p = 1.0 - (-(z.exp())).exp();
        x[[i, 0]] = 1.0;
        x[[i, 1]] = xi;
        y[i] = if rng.random::<f64>() < p { 1.0 } else { 0.0 };
    }

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let mut s = Array2::<f64>::zeros((2, 2));
    s[[1, 1]] = 1.0;
    let s_list = vec![BlockwisePenalty::new(0..2, s)];

    let fit = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &s_list,
        LikelihoodFamily::BinomialCLogLog,
        &FitOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            max_iter: 60,
            tol: 1e-6,
            nullspace_dims: vec![1],
            linear_constraints: None,
            firth_bias_reduction: false,
            adaptive_regularization: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
        },
    )
    .expect("cloglog fit should succeed");

    let pred = predict_gam(
        x.view(),
        fit.beta.view(),
        offset.view(),
        LikelihoodFamily::BinomialCLogLog,
    )
    .expect("cloglog predict should succeed");

    assert!(
        pred.mean
            .iter()
            .all(|v: &f64| v.is_finite() && *v >= 0.0 && *v <= 1.0)
    );
}

#[test]
fn cloglogworkingvectors_are_finite_for_extreme_eta() {
    // Same eta layout as the probit test: index 2 is η=0 where the
    // canonical cloglog mean is exactly 1 - 1/e.
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0]);
    let eta = Array1::from_vec(vec![-100.0, -20.0, 0.0, 20.0, 100.0]);
    let w = Array1::ones(y.len());
    let mut mu = Array1::zeros(y.len());
    let mut weights = Array1::zeros(y.len());
    let mut z = Array1::zeros(y.len());

    update_glmvectors_by_family(
        y.view(),
        &eta,
        GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialCLogLog),
        w.view(),
        &mut mu,
        &mut weights,
        &mut z,
    )
    .expect("cloglog working-vector update should succeed");

    // --- Finiteness / bounds (preserved from the original smoke test) ---
    assert!(
        mu.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0),
        "cloglog mu out of [0,1] or non-finite: mu={mu:?}"
    );
    assert!(
        weights.iter().all(|v| v.is_finite() && *v >= 0.0),
        "cloglog weights non-finite or negative: weights={weights:?}"
    );
    assert!(
        z.iter().all(|v| v.is_finite()),
        "cloglog z non-finite: z={z:?}"
    );

    // --- Mathematical contract for the cloglog mean function ---
    // μ(η) = 1 - exp(-exp(η))
    //   η = -100: exp(-exp(-100)) ≈ exp(-tiny) ≈ 1, so μ ≈ 0
    //   η =    0: μ = 1 - 1/e
    //   η = +100: μ ≈ 1
    assert!(
        mu[0] < 1e-12,
        "cloglog μ(-100) must collapse to ~0; got mu[0]={}",
        mu[0]
    );
    let expected_zero = 1.0 - (-1.0_f64).exp();
    assert!(
        (mu[2] - expected_zero).abs() < 1e-12,
        "cloglog μ(0) must equal 1 - exp(-1) = {expected_zero}; got mu[2]={}",
        mu[2]
    );
    assert!(
        mu[3] > 1.0 - 1e-9,
        "cloglog μ(+20) must be ~1 (exp(20) saturates the inner exp); got mu[3]={}",
        mu[3]
    );
    assert!(
        mu[4] > 1.0 - 1e-12,
        "cloglog μ(+100) must collapse to ~1; got mu[4]={}",
        mu[4]
    );

    // mu must be monotonically non-decreasing in eta (cloglog link is
    // strictly increasing).
    for i in 1..mu.len() {
        assert!(
            mu[i] >= mu[i - 1] - 1e-15,
            "cloglog mu must be non-decreasing in eta; mu[{i}]={mu_i} < mu[{prev_i}]={mu_prev}",
            mu_i = mu[i],
            prev_i = i - 1,
            mu_prev = mu[i - 1]
        );
    }
}

use gam::estimate::FittedLinkState;
use gam::estimate::FitOptions;
use gam::mixture_link::{
    mixture_inverse_link_jet, sas_inverse_link_jet, sas_inverse_link_jetwith_param_partials,
    state_from_sasspec, state_fromspec,
};
use gam::smooth::BlockwisePenalty;
use gam::types::{
    InverseLink, LikelihoodSpec, LinkComponent, MixtureLinkSpec, ResponseFamily, SasLinkSpec,
    StandardLink,
};
use gam_predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn build_design(n: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((n, 4));
    for i in 0..n {
        let t = (i as f64 + 0.5) / (n as f64);
        let x1 = -2.5 + 5.0 * t;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = x1;
        x[[i, 2]] = (1.3 * x1).sin();
        x[[i, 3]] = 0.5 * x1 * x1 - 0.7;
    }
    x
}

fn one_penalty_for_non_intercept(p: usize) -> Vec<BlockwisePenalty> {
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    vec![BlockwisePenalty::new(0..p, s)]
}

fn base_fit_options() -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 60,
        tol: 1e-6,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    }
}

fn brier_score(p: &Array1<f64>, y: &Array1<f64>) -> f64 {
    (p - y).mapv(|v| v * v).mean().unwrap_or(f64::INFINITY)
}

fn coverage(true_p: &Array1<f64>, lo: &Array1<f64>, hi: &Array1<f64>) -> f64 {
    let mut hit = 0usize;
    for i in 0..true_p.len() {
        if lo[i] <= true_p[i] && true_p[i] <= hi[i] {
            hit += 1;
        }
    }
    hit as f64 / true_p.len() as f64
}

fn binomial_likelihood(link: StandardLink) -> LikelihoodSpec {
    LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Standard(link))
}

fn sas_likelihood(spec: SasLinkSpec) -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Sas(state_from_sasspec(spec).expect("initial SAS state")),
    )
}

/// Gauss-Jordan inverse of a small dense matrix. Used only on the `(p + 2)`
/// Fisher information of a fixture's planted model, which is symmetric positive
/// definite whenever the planted design identifies its own parameters — a
/// singular pivot here is the fixture telling you the experiment is degenerate,
/// so it panics rather than returning a pseudo-inverse.
fn dense_inverse(m: &Array2<f64>) -> Array2<f64> {
    let d = m.nrows();
    assert_eq!(d, m.ncols(), "dense_inverse needs a square matrix");
    let mut a = m.clone();
    let mut inv = Array2::<f64>::eye(d);
    for col in 0..d {
        let mut piv = col;
        for r in (col + 1)..d {
            if a[[r, col]].abs() > a[[piv, col]].abs() {
                piv = r;
            }
        }
        assert!(
            a[[piv, col]].abs() > 0.0,
            "singular Fisher information at column {col}: the planted design does not identify its own parameters"
        );
        if piv != col {
            for c in 0..d {
                a.swap([col, c], [piv, c]);
                inv.swap([col, c], [piv, c]);
            }
        }
        let p = a[[col, col]];
        for c in 0..d {
            a[[col, c]] /= p;
            inv[[col, c]] /= p;
        }
        for r in 0..d {
            if r == col {
                continue;
            }
            let f = a[[r, col]];
            if f == 0.0 {
                continue;
            }
            for c in 0..d {
                a[[r, c]] -= f * a[[col, c]];
                inv[[r, c]] -= f * inv[[col, c]];
            }
        }
    }
    inv
}

/// The #2734 fixture's own design: the same three basis columns as
/// [`build_design`] over a WIDER `x1` range, because the range is what buys
/// information about the SAS skew. At the same 3000 rows and the same planted
/// `(epsilon, log delta)`, [`sas_link_fisher_se`] returns `se(eps) = 0.6464` on
/// `build_design`'s `x1` in `[-2.5, 2.5]` and `0.1970` here: a 10.8x variance
/// reduction bought without a single extra row. `epsilon` is aliased with the
/// intercept (`corr = -0.998`), so only the curvature a design actually visits
/// separates the two.
fn build_wide_design(n: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((n, 4));
    for i in 0..n {
        let t = (i as f64 + 0.5) / (n as f64);
        let x1 = -4.5 + 9.0 * t;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = x1;
        x[[i, 2]] = (1.3 * x1).sin();
        x[[i, 3]] = 0.5 * x1 * x1 - 0.7;
    }
    x
}

/// Fisher standard errors of the SAS link parameters `(epsilon, log_delta)` at a
/// planted truth, for the JOINT `(beta, epsilon, log_delta)` Bernoulli model on
/// this exact design — i.e. the precision with which THIS experiment can recover
/// the skew at all, with `beta` free to absorb what it can.
///
/// This is the scale the recovery assertions below are denominated in, and it is
/// why #2734 was not a solver defect. That issue measured `eps_hat = -1.1273`
/// against a planted `-0.5500` and read the `0.5773` gap as a 2.05x overshoot;
/// at the fixture's then-`n = 3000` this function returns `se(eps) = 0.6464`, so
/// the gap was `0.89` standard errors and the fixed `0.20` band the fixture
/// asserted was `0.31` of one — a precision the experiment never carried, which
/// a perfect estimator would have failed about three times in four. The cause is
/// structural rather than numerical: on this link `epsilon` shifts the latent
/// almost exactly the way the intercept does (their planted correlation is
/// `-0.998`), so information about the skew is bought with rows, and the fixture
/// now plants enough of them for the band it asserts.
///
/// Uses the production parameter partials (`sas_inverse_link_jetwith_param_partials`),
/// so the information matrix is the model's own, not a re-derivation.
fn sas_link_fisher_se(
    x: &Array2<f64>,
    beta: &Array1<f64>,
    epsilon: f64,
    log_delta: f64,
) -> (f64, f64) {
    let p = x.ncols();
    let dim = p + 2;
    let mut info = Array2::<f64>::zeros((dim, dim));
    let mut g = Array1::<f64>::zeros(dim);
    for i in 0..x.nrows() {
        let row = x.row(i);
        let jp = sas_inverse_link_jetwith_param_partials(row.dot(beta), epsilon, log_delta)
            .expect("finite SAS parameter partials at the planted truth");
        let mu = jp.jet.mu;
        let var = mu * (1.0 - mu);
        if !var.is_finite() || var <= 0.0 {
            // A saturated row carries no Bernoulli information; including it
            // would divide by zero rather than add anything.
            continue;
        }
        for j in 0..p {
            g[j] = jp.jet.d1 * row[j];
        }
        g[p] = jp.djet_depsilon.mu;
        g[p + 1] = jp.djet_dlog_delta.mu;
        for a in 0..dim {
            let ga = g[a] / var;
            for b in 0..dim {
                info[[a, b]] += ga * g[b];
            }
        }
    }
    let cov = dense_inverse(&info);
    (
        cov[[p, p]].max(0.0).sqrt(),
        cov[[p + 1, p + 1]].max(0.0).sqrt(),
    )
}

fn mixture_likelihood(spec: &MixtureLinkSpec) -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Mixture(state_fromspec(spec).expect("initial mixture state")),
    )
}

/// #2734 applies here too, and this fixture was green by luck rather than by
/// power: on `build_design` at `n = 3000` the Fisher standard error of ε for
/// this planted truth is `0.3996`, so the `0.20` band it asserted was `0.50 σ`
/// and `|ε_true| / se = 0.95` — it could not resolve the sign of the skew it
/// plants. It now carries the same Fisher-denominated bands as
/// `sas_recovers_negative_skew_and_positive_log_delta`, and enough rows for
/// them. It keeps [`build_design`] rather than moving to the wider design that
/// fixture uses: this planted β is steeper (slope 1.15 against 0.15), and on
/// `x1 in [-4.5, 4.5]` it reaches η = 8.29, where the SAS inverse link saturates
/// to μ = 1.0 and PIRLS refuses the row geometry — the separate defect tracked
/// by #2733/#2685. Information here is therefore bought with rows alone.
#[test]
fn sas_fit_recovery_and_calibration_system() {
    let n = 120_000_usize;
    let x = build_design(n);
    let beta_true = Array1::from_vec(vec![-0.35, 1.15, -0.65, 0.45]);
    let eps_true: f64 = 0.38;
    let log_delta_true: f64 = -0.30;
    let delta_true = log_delta_true.exp();
    let eta = x.dot(&beta_true);
    let p_true = eta.mapv(|e| {
        sas_inverse_link_jet(e, eps_true, log_delta_true)
            .expect("finite SAS eta")
            .mu
    });

    let mut rng = StdRng::seed_from_u64(9001);
    let y = p_true.mapv(|p| if rng.random::<f64>() < p { 1.0 } else { 0.0 });
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let s_list = one_penalty_for_non_intercept(x.ncols());

    let mut opts = base_fit_options();
    opts.sas_link = Some(SasLinkSpec {
        initial_epsilon: 0.0,
        initial_log_delta: 0.0,
    });
    opts.optimize_sas = true;
    let family = sas_likelihood(opts.sas_link.expect("SAS fit spec"));
    let fit = fit_gam(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        &s_list,
        family.clone(),
        &opts,
    )
    .expect("SAS fit");

    let (eps_hat, log_delta_hat, delta_hat) = match &fit.fitted_link {
        FittedLinkState::Sas { state, .. } => (state.epsilon, state.log_delta, state.delta),
        other => panic!("expected SAS fitted state, got {other:?}"),
    };

    // Same redenomination as the negative-skew fixture (#2734): the band is the
    // estimator's own Fisher scale, it must be no looser than the fixed bands it
    // replaces, and the design must be able to reject a sign-flipped skew.
    const RECOVERY_SIGMA: f64 = 3.0;
    const PRE_2734_EPSILON_BAND: f64 = 0.20;
    const PRE_2734_DELTA_BAND: f64 = 0.20;
    let (se_eps, se_log_delta) = sas_link_fisher_se(&x, &beta_true, eps_true, log_delta_true);
    let eps_band = RECOVERY_SIGMA * se_eps;
    let log_delta_band = RECOVERY_SIGMA * se_log_delta;
    eprintln!(
        "[#2734] n={n} se(eps)={se_eps:.6} se(log_delta)={se_log_delta:.6} \
         eps_band(3sigma)={eps_band:.6} log_delta_band(3sigma)={log_delta_band:.6}"
    );
    eprintln!(
        "[#2734] eps_hat={eps_hat:+.6} (true {eps_true:+.4}, gap {:+.6} = {:+.3} sigma)  \
         log_delta_hat={log_delta_hat:+.6} (true {log_delta_true:+.4}, gap {:+.6} = {:+.3} sigma)  \
         delta_hat={delta_hat:.6} (true {delta_true:.6})",
        eps_hat - eps_true,
        (eps_hat - eps_true) / se_eps,
        log_delta_hat - log_delta_true,
        (log_delta_hat - log_delta_true) / se_log_delta,
    );
    assert!(
        2.0 * eps_true.abs() > eps_band,
        "the planted design cannot resolve the sign of the skew it plants: \
         mirrored offset 2|eps_true|={:.6} is inside the {RECOVERY_SIGMA}-sigma band {eps_band:.6} \
         (se={se_eps:.6}, n={n}) — raise n rather than the band",
        2.0 * eps_true.abs()
    );
    assert!(
        (eps_hat + eps_true).abs() > eps_band,
        "the recovery band does not reject the sign-flipped skew: \
         |eps_hat - (-eps_true)|={:.6} is inside the band {eps_band:.6} (eps_hat={eps_hat:.6})",
        (eps_hat + eps_true).abs()
    );
    assert!(
        eps_band <= PRE_2734_EPSILON_BAND,
        "the Fisher-denominated band {eps_band:.6} is looser than the {PRE_2734_EPSILON_BAND} \
         this fixture asserted before #2734 (se={se_eps:.6} at n={n})"
    );
    assert!(
        log_delta_band <= PRE_2734_DELTA_BAND / delta_true,
        "the Fisher-denominated log-delta band {log_delta_band:.6} is looser than the \
         {PRE_2734_DELTA_BAND} delta band this fixture asserted before #2734, carried to \
         log-delta by the delta method (delta_true={delta_true:.6}, se={se_log_delta:.6}, n={n})"
    );
    assert!(
        eps_hat > 0.0,
        "recovered skewness has the wrong sign: hat={eps_hat:.4}, true={eps_true:.4}"
    );
    assert!(
        (eps_hat - eps_true).abs() <= eps_band,
        "epsilon recovery off: hat={eps_hat:.6}, true={eps_true:.4}, gap={:.6} = {:.3} sigma \
         against se={se_eps:.6} (band {eps_band:.6})",
        (eps_hat - eps_true).abs(),
        (eps_hat - eps_true).abs() / se_eps
    );
    assert!(
        (log_delta_hat - log_delta_true).abs() <= log_delta_band,
        "log-delta recovery off: hat={log_delta_hat:.6} (delta {delta_hat:.6}), \
         true={log_delta_true:.4} (delta {delta_true:.6}), gap={:.6} = {:.3} sigma \
         against se={se_log_delta:.6} (band {log_delta_band:.6})",
        (log_delta_hat - log_delta_true).abs(),
        (log_delta_hat - log_delta_true).abs() / se_log_delta
    );

    let pred = predict_gamwith_uncertainty(
        x.view(),
        fit.beta.view(),
        offset.view(),
        family,
        &fit,
        &PredictUncertaintyOptions::default(),
    )
    .expect("SAS predict");
    let brier = brier_score(&pred.mean, &y);
    assert!(brier < 0.20, "SAS Brier too high: {brier:.4}");
    let calib_gap = (pred.mean.mean().unwrap_or(0.0) - y.mean().unwrap_or(0.0)).abs();
    assert!(
        calib_gap < 0.04,
        "SAS prevalence calibration gap too large: {calib_gap:.4}"
    );
}

/// Root-cause regression for #1876 from a different angle: the original defect
/// recovered the SAS skewness with the WRONG SIGN and WRONG MAGNITUDE because
/// the outer link-parameter gradient was evaluated at a first-order-capped inner
/// β̂ without the KKT-residual envelope correction. The headline repro plants a
/// positive skew (+0.38); here we plant a NEGATIVE skew with a POSITIVE
/// log-delta (a genuinely different point in (ε, log δ) space) and require the
/// fit to recover both — so a sign flip or a scale error in the envelope
/// correction is caught independently of the exact seed/parameters that first
/// surfaced the bug.
///
/// #2734: this fixture used to plant `n = 3000` rows and assert `|ε̂ - ε| < 0.20`.
/// The Fisher standard error of ε on that design is `0.6464` (see
/// [`sas_link_fisher_se`]), so the band was `0.31 σ` — the experiment could not
/// resolve even the SIGN of the planted skew (`|ε| / se = 0.85`), which is the
/// one thing the fixture exists to check. The reported failure (`ε̂ = -1.1273`,
/// a `0.89 σ` draw) was the fixture asking for a precision it had not bought.
/// The design range and the row count are now set so the band the fixture
/// asserts is a `3 σ` band that is still TIGHTER than the `0.20` it replaces,
/// and so the sign flip it guards against is a `12 σ` discriminator. Both facts
/// are asserted here rather than assumed.
#[test]
fn sas_recovers_negative_skew_and_positive_log_delta() {
    let n = 70_000_usize;
    let x = build_wide_design(n);
    let beta_true = Array1::from_vec(vec![-0.80, 0.15, -0.55, 0.40]);
    let eps_true: f64 = -0.55;
    let log_delta_true: f64 = 0.25;
    let delta_true = log_delta_true.exp();
    let eta = x.dot(&beta_true);
    let p_true = eta.mapv(|e| {
        sas_inverse_link_jet(e, eps_true, log_delta_true)
            .expect("finite SAS eta")
            .mu
    });

    let mut rng = StdRng::seed_from_u64(20260702);
    let y = p_true.mapv(|p| if rng.random::<f64>() < p { 1.0 } else { 0.0 });
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let s_list = one_penalty_for_non_intercept(x.ncols());

    let mut opts = base_fit_options();
    opts.sas_link = Some(SasLinkSpec {
        initial_epsilon: 0.0,
        initial_log_delta: 0.0,
    });
    opts.optimize_sas = true;
    let family = sas_likelihood(opts.sas_link.expect("SAS fit spec"));
    let fit = fit_gam(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        &s_list,
        family.clone(),
        &opts,
    )
    .expect("SAS fit");

    let (eps_hat, log_delta_hat, delta_hat) = match &fit.fitted_link {
        FittedLinkState::Sas { state, .. } => (state.epsilon, state.log_delta, state.delta),
        other => panic!("expected SAS fitted state, got {other:?}"),
    };

    // The recovery band is the estimator's own Fisher scale on this planted
    // design, not a fixed count of ε units (#2734).
    const RECOVERY_SIGMA: f64 = 3.0;
    // The band this fixture asserted before #2734. Kept as a ceiling on the
    // derived band so the redenomination can only ever TIGHTEN what is asserted:
    // if a future edit shrinks `n`, the Fisher band grows past this and the
    // fixture refuses instead of quietly accepting a looser recovery.
    const PRE_2734_EPSILON_BAND: f64 = 0.20;
    // The same ceiling for the scale parameter: the pre-#2734 assertion was
    // `|delta_hat - delta| < 0.20`, carried to log-delta by the delta method.
    const PRE_2734_DELTA_BAND: f64 = 0.20;
    let (se_eps, se_log_delta) = sas_link_fisher_se(&x, &beta_true, eps_true, log_delta_true);
    let eps_band = RECOVERY_SIGMA * se_eps;
    let log_delta_band = RECOVERY_SIGMA * se_log_delta;
    // Measurements print unconditionally: a number that only appears inside an
    // `assert!` message is never read on a pass.
    eprintln!(
        "[#2734] n={n} se(eps)={se_eps:.6} se(log_delta)={se_log_delta:.6} \
         eps_band(3sigma)={eps_band:.6} log_delta_band(3sigma)={log_delta_band:.6}"
    );
    eprintln!(
        "[#2734] eps_hat={eps_hat:+.6} (true {eps_true:+.4}, gap {:+.6} = {:+.3} sigma)  \
         log_delta_hat={log_delta_hat:+.6} (true {log_delta_true:+.4}, gap {:+.6} = {:+.3} sigma)  \
         delta_hat={delta_hat:.6} (true {delta_true:.6})",
        eps_hat - eps_true,
        (eps_hat - eps_true) / se_eps,
        log_delta_hat - log_delta_true,
        (log_delta_hat - log_delta_true) / se_log_delta,
    );

    // NON-VACUITY, asserted rather than assumed. The band must be sharp enough
    // to reject the defect this fixture exists to catch — the #1876 sign flip,
    // which lands ε̂ at -ε_true. First that the DESIGN can reject it at all
    // (this is what n=3000 failed: 2|ε| = 1.10 against a 1.94 band), then that
    // this RUN's ε̂ does reject it. The second is the positive control: it fires
    // on exactly the mirrored estimate the fixture guards against.
    assert!(
        2.0 * eps_true.abs() > eps_band,
        "the planted design cannot resolve the sign of the skew it plants: \
         mirrored offset 2|eps_true|={:.6} is inside the {RECOVERY_SIGMA}-sigma band {eps_band:.6} \
         (se={se_eps:.6}, n={n}) — raise n rather than the band",
        2.0 * eps_true.abs()
    );
    assert!(
        (eps_hat + eps_true).abs() > eps_band,
        "the recovery band does not reject the sign-flipped skew: \
         |eps_hat - (-eps_true)|={:.6} is inside the band {eps_band:.6}, so a #1876-style \
         sign flip would pass this fixture (eps_hat={eps_hat:.6})",
        (eps_hat + eps_true).abs()
    );
    assert!(
        log_delta_band <= PRE_2734_DELTA_BAND / delta_true,
        "the Fisher-denominated log-delta band {log_delta_band:.6} is looser than the \
         {PRE_2734_DELTA_BAND} delta band this fixture asserted before #2734, carried to \
         log-delta by the delta method (delta_true={delta_true:.6}, se={se_log_delta:.6}, n={n})"
    );
    assert!(
        eps_band <= PRE_2734_EPSILON_BAND,
        "the Fisher-denominated band {eps_band:.6} is looser than the {PRE_2734_EPSILON_BAND} \
         this fixture asserted before #2734 — that would be widening the bar, not redenominating \
         it (se={se_eps:.6} at n={n})"
    );

    // Sign must be correct (the pre-fix defect flipped it) and the magnitude
    // inside the band the design actually supports.
    assert!(
        eps_hat < 0.0,
        "recovered skewness has the wrong sign: hat={eps_hat:.4}, true={eps_true:.4}"
    );
    assert!(
        (eps_hat - eps_true).abs() <= eps_band,
        "epsilon recovery off: hat={eps_hat:.6}, true={eps_true:.4}, gap={:.6} = {:.3} sigma \
         against se={se_eps:.6} (band {eps_band:.6})",
        (eps_hat - eps_true).abs(),
        (eps_hat - eps_true).abs() / se_eps
    );
    assert!(
        (log_delta_hat - log_delta_true).abs() <= log_delta_band,
        "log-delta recovery off: hat={log_delta_hat:.6} (delta {delta_hat:.6}), \
         true={log_delta_true:.4} (delta {delta_true:.6}), gap={:.6} = {:.3} sigma \
         against se={se_log_delta:.6} (band {log_delta_band:.6})",
        (log_delta_hat - log_delta_true).abs(),
        (log_delta_hat - log_delta_true).abs() / se_log_delta
    );

    let pred = predict_gamwith_uncertainty(
        x.view(),
        fit.beta.view(),
        offset.view(),
        family,
        &fit,
        &PredictUncertaintyOptions::default(),
    )
    .expect("SAS predict");
    let calib_gap = (pred.mean.mean().unwrap_or(0.0) - y.mean().unwrap_or(0.0)).abs();
    assert!(
        calib_gap < 0.04,
        "SAS prevalence calibration gap too large: {calib_gap:.4}"
    );
}

#[test]
fn mixture_recovery_and_prediction_alignment_system() {
    let n = 2000usize;
    let x = build_design(n);
    let beta_true = Array1::from_vec(vec![-0.20, 1.0, -0.5, 0.3]);
    let mixspec_true = MixtureLinkSpec {
        components: vec![
            LinkComponent::Probit,
            LinkComponent::CLogLog,
            LinkComponent::Cauchit,
        ],
        initial_rho: Array1::from_vec(vec![1.0, -0.6]),
    };
    let mix_state_true = state_fromspec(&mixspec_true).expect("true mixture state");
    let eta = x.dot(&beta_true);
    let p_true = eta.mapv(|e| mixture_inverse_link_jet(&mix_state_true, e).mu);
    let mut rng = StdRng::seed_from_u64(12345);
    let y = p_true.mapv(|p| if rng.random::<f64>() < p { 1.0 } else { 0.0 });
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let s_list = one_penalty_for_non_intercept(x.ncols());

    let mut opts = base_fit_options();
    opts.mixture_link = Some(MixtureLinkSpec {
        components: mixspec_true.components.clone(),
        initial_rho: Array1::zeros(2),
    });
    opts.optimize_mixture = true;
    let family = mixture_likelihood(opts.mixture_link.as_ref().expect("mixture fit spec"));
    let fit = fit_gam(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        &s_list,
        family.clone(),
        &opts,
    )
    .expect("mixture fit");

    let pi_hat = match &fit.fitted_link {
        FittedLinkState::Mixture { state, .. } => state.pi.clone(),
        other => panic!("expected Mixture fitted state, got {other:?}"),
    };
    let simplex_sum = pi_hat.sum();
    assert!(
        (simplex_sum - 1.0).abs() < 1e-10 && pi_hat.iter().all(|&w| (0.0..=1.0).contains(&w)),
        "fitted mixture weights must be a valid simplex, got pi={pi_hat:?}"
    );

    let pred = predict_gamwith_uncertainty(
        x.view(),
        fit.beta.view(),
        offset.view(),
        family,
        &fit,
        &PredictUncertaintyOptions::default(),
    )
    .expect("mixture predict");
    let pred_rmsevs_truth = (&pred.mean - &p_true)
        .mapv(|v| v * v)
        .mean()
        .unwrap_or(f64::INFINITY)
        .sqrt();
    assert!(
        pred_rmsevs_truth < 0.11,
        "mixture truth-prob RMSE too high: {pred_rmsevs_truth:.4}"
    );
}

#[test]
fn posterior_mean_coverage_includes_sas_and_mixture() {
    let n_train = 1500usize;
    let n_test = 1000usize;
    let x_train = build_design(n_train);
    let x_test = build_design(n_test);
    let beta_true = Array1::from_vec(vec![-0.25, 0.95, -0.7, 0.35]);
    let eta_train = x_train.dot(&beta_true);
    let eta_test = x_test.dot(&beta_true);
    let p_train = eta_train.mapv(|e| 1.0 / (1.0 + (-e).exp()));
    let p_test = eta_test.mapv(|e| 1.0 / (1.0 + (-e).exp()));

    let mut rng = StdRng::seed_from_u64(4242);
    let y_train = p_train.mapv(|p| if rng.random::<f64>() < p { 1.0 } else { 0.0 });
    let w = Array1::<f64>::ones(n_train);
    let offset_train = Array1::<f64>::zeros(n_train);
    let offset_test = Array1::<f64>::zeros(n_test);
    let s_list = one_penalty_for_non_intercept(x_train.ncols());

    let opts_logit = base_fit_options();
    let logit_family = binomial_likelihood(StandardLink::Logit);
    let fit_logit = fit_gam(
        x_train.view(),
        y_train.view(),
        w.view(),
        offset_train.view(),
        &s_list,
        logit_family.clone(),
        &opts_logit,
    )
    .expect("logit fit");

    let opts_probit = base_fit_options();
    let probit_family = binomial_likelihood(StandardLink::Probit);
    let fit_probit = fit_gam(
        x_train.view(),
        y_train.view(),
        w.view(),
        offset_train.view(),
        &s_list,
        probit_family.clone(),
        &opts_probit,
    )
    .expect("probit fit");

    let mut opts_sas = base_fit_options();
    opts_sas.sas_link = Some(SasLinkSpec {
        initial_epsilon: 0.0,
        initial_log_delta: 0.0,
    });
    opts_sas.optimize_sas = true;
    opts_sas.max_iter = 120;
    opts_sas.tol = 1e-8;
    let sas_family = sas_likelihood(opts_sas.sas_link.expect("SAS fit spec"));
    let fit_sas = fit_gam(
        x_train.view(),
        y_train.view(),
        w.view(),
        offset_train.view(),
        &s_list,
        sas_family.clone(),
        &opts_sas,
    )
    .expect("sas fit");

    let mut opts_mix = base_fit_options();
    opts_mix.mixture_link = Some(MixtureLinkSpec {
        components: vec![
            LinkComponent::Probit,
            LinkComponent::Logit,
            LinkComponent::CLogLog,
        ],
        initial_rho: Array1::zeros(2),
    });
    opts_mix.optimize_mixture = true;
    let mixture_family = mixture_likelihood(opts_mix.mixture_link.as_ref().expect("mixture spec"));
    let fit_mix = fit_gam(
        x_train.view(),
        y_train.view(),
        w.view(),
        offset_train.view(),
        &s_list,
        mixture_family.clone(),
        &opts_mix,
    )
    .expect("mixture fit");

    let options = PredictUncertaintyOptions {
        confidence_level: 0.90,
        covariance_mode: InferenceCovarianceMode::SmoothingCorrected,
        mean_interval_method: MeanIntervalMethod::TransformEta,
        includeobservation_interval: false,
        apply_bias_correction: false,
        ..PredictUncertaintyOptions::default()
    };

    let pred_logit = predict_gamwith_uncertainty(
        x_test.view(),
        fit_logit.beta.view(),
        offset_test.view(),
        logit_family,
        &fit_logit,
        &options,
    )
    .expect("logit pred");
    let pred_probit = predict_gamwith_uncertainty(
        x_test.view(),
        fit_probit.beta.view(),
        offset_test.view(),
        probit_family,
        &fit_probit,
        &options,
    )
    .expect("probit pred");
    let pred_sas = predict_gamwith_uncertainty(
        x_test.view(),
        fit_sas.beta.view(),
        offset_test.view(),
        sas_family,
        &fit_sas,
        &options,
    )
    .expect("sas pred");
    let pred_mix = predict_gamwith_uncertainty(
        x_test.view(),
        fit_mix.beta.view(),
        offset_test.view(),
        mixture_family,
        &fit_mix,
        &options,
    )
    .expect("mix pred");

    let c_logit = coverage(&p_test, &pred_logit.mean_lower, &pred_logit.mean_upper);
    let c_probit = coverage(&p_test, &pred_probit.mean_lower, &pred_probit.mean_upper);
    let c_sas = coverage(&p_test, &pred_sas.mean_lower, &pred_sas.mean_upper);
    let c_mix = coverage(&p_test, &pred_mix.mean_lower, &pred_mix.mean_upper);

    let meanwidth_logit = (&pred_logit.mean_upper - &pred_logit.mean_lower)
        .mean()
        .unwrap_or(f64::INFINITY);
    let meanwidth_probit = (&pred_probit.mean_upper - &pred_probit.mean_lower)
        .mean()
        .unwrap_or(f64::INFINITY);
    let meanwidth_sas = (&pred_sas.mean_upper - &pred_sas.mean_lower)
        .mean()
        .unwrap_or(f64::INFINITY);
    let meanwidth_mix = (&pred_mix.mean_upper - &pred_mix.mean_lower)
        .mean()
        .unwrap_or(f64::INFINITY);

    for (name, c, w) in [
        ("logit", c_logit, meanwidth_logit),
        ("probit", c_probit, meanwidth_probit),
    ] {
        assert!(c >= 0.80, "{name} 90% coverage too low: {c:.3}");
        assert!(
            w < 0.60,
            "{name} intervals too wide on average: mean width={w:.3}"
        );
    }
    for (name, c, w) in [
        ("sas", c_sas, meanwidth_sas),
        ("mixture", c_mix, meanwidth_mix),
    ] {
        assert!(
            c >= 0.65,
            "{name} 90% coverage too low: {c:.3} (extra parameter uncertainty expected)"
        );
        assert!(
            w < 0.60,
            "{name} intervals too wide on average: mean width={w:.3}"
        );
    }
}

//! End-to-end quality: gam's *binomial* location-scale fit (a smooth latent
//! threshold AND a smooth log-sigma over-dispersion, fit jointly by penalized
//! blockwise PIRLS) must track `gamlss::gamlss(family = DBI())` — the
//! over-dispersed ("double") binomial in the GAMLSS distribution suite, which
//! is the mature R reference for a binomial response whose location *and*
//! dispersion both vary smoothly with a covariate.
//!
//! Why DBI() and not the bare BI() the capability sketch names
//! -----------------------------------------------------------
//! gam's binomial location-scale model carries TWO smooth axes: a latent
//! threshold `t(x)` (logit-link location) and a log-sigma `log σ(x)`
//! (dispersion). The standard `BI()` family in gamlss has a SINGLE parameter
//! (`mu`); it exposes no `sigma` axis, so `sigma.formula = ~ pb(x)` against
//! `BI()` is degenerate (sigma collapses to 1, log σ ≡ 0) and would make any
//! log-sigma comparison a division-by-near-zero artefact of the *reference*,
//! not a measurement of gam. The double binomial `DBI()` is the faithful
//! mature reference for exactly the stated rationale — "binomial location-scale
//! with smooth log-sigma (over-dispersion)" — with `mu.link = "logit"` and
//! `sigma.link = "log"`, i.e. the same two link scales gam fits. (Repo memory:
//! the capability sketch is a recommendation, not gospel; the correct, mature
//! reference is chosen here.)
//!
//! What gam parameterizes (verified against the source)
//! ----------------------------------------------------
//! For the binomial location-scale family gam fits a latent threshold `t` (block
//! role `Mean`, block-state index 0) and a log-sigma `η_ls` (block role `Scale`,
//! block-state index 1) with the PURE exponential sigma link `σ = exp(η_ls)`
//! (NOT the `0.01 + exp` floor the Gaussian noise link uses — see
//! `families::sigma_link::exp_sigma_inverse_from_eta_scalar`). The link-scale
//! linear predictor is `q = -t / σ = -t · exp(-η_ls)` and the fitted success
//! probability is `P(y=1) = link⁻¹(q)` (logit here), exactly as
//! `compute_probit_q0_from_eta` forms it in the CLI. Only `q` (= logit P) — not
//! `t` and `σ` separately — is identifiable from binary data, so the LOCATION
//! axis is compared as gam's `q` against gamlss's `mu` linear predictor (logit
//! P); the SCALE axis is compared as gam's `η_ls` against gamlss's `log σ`.
//!
//! Both engines maximize a penalized binomial location-scale likelihood with a
//! logit location link and a log dispersion link, so the recovered logit-P and
//! log-sigma smooths should converge to close shapes up to basis-convention and
//! the inherent weak identifiability of a per-row dispersion from 0/1 data. A
//! genuine divergence here is a real bug in gam's joint inverse-link / gradient
//! handling across the two very different scales (probability vs log-variance),
//! which is precisely the numerical stress this test exists to measure.

use gam::estimate::BlockRole;
use gam::gamlss::BinomialLocationScaleFitResult;
use gam::test_support::reference::{Column, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

#[test]
fn gam_binomial_location_scale_matches_gamlss_dbi() {
    init_parallelism();

    // ---- synthetic latent-threshold binomial recipe (fed IDENTICALLY to both
    // engines). Spec: n=150, x~Uniform(-3,3), latent t(x)=1+0.5 sin(pi x),
    // P(y=1|x,z)=expit(t(x)+(0.5+0.2 sin(pi x)) z), z~N(0,1), seed=456. The
    // per-row latent draw `z` injects exactly the over-dispersion DBI models,
    // so the data-generating process matches the reference family's structure.
    let n = 150usize;
    let pi = std::f64::consts::PI;
    let two_pi = 2.0 * pi;

    // Deterministic Numerical-Recipes LCG seeded at 456 so the exact same
    // (x, y) is reproducible in pure Rust and handed verbatim to gamlss.
    let mut state: u64 = 456;
    let mut next_unit = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };

    // x ~ Uniform(-3, 3), sorted so the comparison runs along an ordered curve
    // from x.min to x.max (the design is identical across engines either way).
    let mut x: Vec<f64> = (0..n).map(|_| -3.0 + 6.0 * next_unit()).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Box-Muller standard normals from the same LCG stream (seed continues).
    let mut z: Vec<f64> = Vec::with_capacity(n);
    while z.len() < n {
        let u1 = next_unit().max(1e-300);
        let u2 = next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        z.push(r * (two_pi * u2).cos());
        if z.len() < n {
            z.push(r * (two_pi * u2).sin());
        }
    }

    let t_true = |xi: f64| 1.0 + 0.5 * (pi * xi).sin();
    let s_true = |xi: f64| 0.5 + 0.2 * (pi * xi).sin();
    let expit = |e: f64| 1.0 / (1.0 + (-e).exp());
    // Bernoulli outcome from the latent logistic: success when a uniform draw
    // falls under the per-row probability expit(t + s·z).
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let p = expit(t_true(x[i]) + s_true(x[i]) * z[i]);
            if next_unit() < p { 1.0 } else { 0.0 }
        })
        .collect();

    // Guard against a degenerate all-0/all-1 draw (would make a binomial fit
    // meaningless); the seed above yields a healthy mix.
    let ones: f64 = y.iter().sum();
    assert!(
        ones > 10.0 && ones < (n as f64 - 10.0),
        "degenerate binary response: {ones} successes out of {n}"
    );

    // ---- build the dataset (column 0 = x, column 1 = y) --------------------
    let headers: Vec<String> = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", y[i])])
        })
        .collect();
    let ds =
        encode_recordswith_inferred_schema(headers, rows).expect("encode binomial location-scale data");

    // ---- fit with gam: threshold ~ s(x, bs='tp'), log-sigma ~ 1 + s(x, bs='tp')
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        noise_formula: Some("1 + s(x, bs='tp')".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(x, bs='tp')", &ds, &cfg).expect("gam binomial location-scale fit");
    let FitResult::BinomialLocationScale(BinomialLocationScaleFitResult { fit, .. }) = result else {
        panic!("expected a binomial location-scale fit");
    };

    // Sanity: the joint fit must carry both a Mean (threshold) and a Scale
    // (log-sigma) coefficient block, and the log-sigma block must be a genuine
    // multi-coefficient smooth (not a lone intercept) for `1 + s(x)`.
    let scale_block = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("binomial location-scale fit must carry a Scale (log-sigma) block");
    assert!(
        fit.fit.block_by_role(BlockRole::Mean).is_some(),
        "binomial location-scale fit must carry a Mean (threshold) block"
    );
    assert!(
        scale_block.beta.len() >= 2,
        "smooth noise_formula must materialize a multi-coefficient log-sigma basis, got {}",
        scale_block.beta.len()
    );

    // gam's converged per-row latent predictors at the (training) x rows:
    //   block_states[0].eta = threshold t,  block_states[1].eta = log-sigma η_ls.
    // These are the exact, identification-transform-consistent linear predictors
    // (the binomial log-sigma design is internally reparameterized, so reading
    // the converged η directly is the correct, transform-faithful source).
    let eta_t = &fit.fit.block_states[0].eta;
    let eta_ls = &fit.fit.block_states[1].eta;
    assert_eq!(eta_t.len(), n, "threshold eta length");
    assert_eq!(eta_ls.len(), n, "log-sigma eta length");

    // Location axis on the identifiable link scale: q = logit P = -t / σ
    // = -t · exp(-η_ls). Scale axis: gam's log σ is η_ls directly (pure exp link).
    let gam_logit_p: Vec<f64> = (0..n).map(|i| -eta_t[i] * (-eta_ls[i]).exp()).collect();
    let gam_log_sigma: Vec<f64> = eta_ls.to_vec();

    // ---- fit the SAME data with gamlss DBI() (the mature GAMLSS reference) ---
    // Double binomial: mu.link = logit (location), sigma.link = log (dispersion);
    // smooth both via penalized B-splines pb(). Predict on the identical training
    // rows (row order = our sorted x), then read mu on the LINK scale (= logit P,
    // comparable to gam's q) and sigma on the response scale (log → log σ).
    let body = r#"
        suppressPackageStartupMessages(library(gamlss))
        m <- gamlss(y ~ pb(x), sigma.formula = ~ pb(x), family = DBI(),
                    data = df, control = gamlss.control(trace = FALSE))
        eta_mu <- predict(m, what = "mu", type = "link")
        sigma  <- predict(m, what = "sigma", type = "response")
        emit("logit_p", as.numeric(eta_mu))
        emit("log_sigma", as.numeric(log(sigma)))
    "#;
    let r = run_r(&[Column::new("x", &x), Column::new("y", &y)], body);
    let ref_logit_p = r.vector("logit_p");
    let ref_log_sigma = r.vector("log_sigma");
    assert_eq!(ref_logit_p.len(), n, "gamlss mu-link length mismatch");
    assert_eq!(ref_log_sigma.len(), n, "gamlss log-sigma length mismatch");

    // ---- compare the recovered smooth shapes element-wise on the rows -------
    let rel_logit_p = relative_l2(&gam_logit_p, ref_logit_p);
    let rel_log_sigma = relative_l2(&gam_log_sigma, ref_log_sigma);

    eprintln!(
        "binomial location-scale vs gamlss DBI(): n={n} \
         rel_l2(logit P)={rel_logit_p:.5} rel_l2(log sigma)={rel_log_sigma:.5}"
    );

    // Bounds (spec-mandated, principled, NOT loosened to pass):
    //   * The LOCATION axis (logit P = -t/σ) is the well-determined, identifiable
    //     quantity both engines target with the same logit link; it must agree to
    //     a tight relative-L2 of 0.025. A larger gap means gam's joint inverse-link
    //     reconstruction of the success probability diverges from the GAMLSS MLE.
    //   * The SCALE axis (log σ) is a second-moment over-dispersion quantity that
    //     is only weakly identified from 0/1 data and differs slightly in basis /
    //     dispersion-link convention between gam and DBI; it is allowed the looser
    //     relative-L2 of 0.05. Either bound being exceeded is a genuine divergence
    //     of gam's two-scale joint solver from the GAMLSS standard, not a tolerance
    //     to be relaxed.
    assert!(
        rel_logit_p < 0.025,
        "fitted logit-P (latent threshold) smooth diverges from gamlss DBI: rel_l2={rel_logit_p:.5}"
    );
    assert!(
        rel_log_sigma < 0.05,
        "fitted log-sigma smooth diverges from gamlss DBI: rel_l2={rel_log_sigma:.5}"
    );
}

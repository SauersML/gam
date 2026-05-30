//! End-to-end quality: gam's **shared-frailty survival via the lognormal
//! hazard-multiplier family** must agree with **R `survival::coxph(... +
//! frailty(g))`** — the mature, standard shared-frailty Cox model — on the same
//! clustered, censored survival data.
//!
//! The model. Both engines fit a proportional-hazards model with a multiplicative
//! cluster-level frailty acting on the cumulative hazard,
//!
//!     H(t | x, U_g) = H_0(t) · exp(x · β) · exp(U_g),
//!
//! where the m clusters g = 1..m each carry one shared random multiplier.
//!   * **gam** uses the `survival_likelihood = "latent"` family with
//!     `FrailtySpec::HazardMultiplier { sigma_fixed: None, loading: Full }`: the
//!     frailty is lognormal, `exp(U_g)` with `U_g ~ N(0, σ²)`, integrated out
//!     *exactly* through the K_{k,m}(μ, σ) microcell kernels, and σ is selected
//!     by the outer (REML/marginal-likelihood) loop. The baseline is a flexible
//!     monotone I-spline on a Weibull scaffold (the latent family requires a
//!     non-linear scalar baseline), gam's smooth analogue of the Breslow baseline.
//!   * **R survival::coxph** uses a gamma-distributed shared frailty (mean 1,
//!     variance θ) — the exponential-family conjugate on the hazard scale — with
//!     a Breslow baseline and the (exact / Efron) partial likelihood. The frailty
//!     variance θ is its own penalized-profile estimate.
//!
//! Both maximize the *same marginal survival likelihood* for the same hierarchical
//! PH structure on identical data. The lognormal and gamma frailties are distinct
//! distributions, but both are mean-≈1 multiplicative random effects on the hazard,
//! so on the hazard scale the covariate log-HR is a shared estimand and the
//! *variance of the multiplier* is the shared dispersion estimand. Crucially, R's
//! gamma `theta` is the variance of the multiplier itself (gamma mean 1, variance
//! theta), so to compare we must put gam on the multiplier scale too: gam learns
//! the log-scale spread `sigma` of exp(U_g), and the matching quantity is
//! Var(exp(U_g)) = (exp(sigma^2) − 1)·exp(sigma^2), NOT sigma^2. Once grid-aligned
//! on the multiplier-variance scale, the two must track within optimizer +
//! quadrature + small-sample noise. Non-nested clusters with multiple events per
//! cluster make the frailty identifiable (not confounded with the baseline).
//!
//! Data. Fixed-seed clustered right-censored survival data: n = 120 subjects
//! across m = 12 frailty groups (10 per group), generated from the generative
//! model above with a known covariate effect and a known frailty spread. The
//! identical (t, event, x, g) columns are handed to gam (via latent-cloglog
//! survival + HazardMultiplier frailty) and to R (via `coxph(frailty(g))`).
//!
//! Bounds (principled, un-weakened):
//!   1. Fixed-effect log-HR: `|gam.β_x − R.coef[x]| ≤ 0.05` (L-∞, both PH-scale).
//!      Both maximize a partial/marginal likelihood on identical data, so the
//!      covariate effect on the hazard scale must agree to optimizer + quadrature
//!      tolerance (~0.05 at n ≈ 100).
//!   2. Frailty multiplier-variance: `|Var(exp(U_g)) − R.θ| / max(R.θ, 1e-3) ≤ 0.15`,
//!      where Var(exp(U_g)) = (exp(σ²)−1)·exp(σ²) is gam's lognormal multiplier
//!      variance on the SAME scale as coxph's gamma θ. Both target the same
//!      marginal likelihood and the same multiplier-variance dispersion; 15%
//!      absorbs the residual lognormal-vs-gamma shape difference (matched mean and
//!      variance, differing skew/tail) and finite-n variation without weakening
//!      (a real divergence is larger).
//!
//! A failing assertion because gam genuinely diverges from coxph is acceptable
//! and must NOT be papered over by loosening a bound or editing gam source.

use csv::StringRecord;
use gam::families::lognormal_kernel::{FrailtySpec, HazardLoading};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const N_GROUPS: usize = 12;
const PER_GROUP: usize = 10;
const TRUE_BETA: f64 = 0.7;
const TRUE_FRAILTY_SD: f64 = 0.5;

/// Deterministic standard-normal draw via Box-Muller on a small LCG. A
/// fixed-seed generator keeps the dataset bit-identical across runs and across
/// the two engines (gam and R both see the same emitted columns).
struct DetRng {
    state: u64,
}

impl DetRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        // SplitMix64 — full-period, good low-bit mixing.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        // 53-bit mantissa uniform in (0, 1).
        let bits = self.next_u64() >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[test]
fn gam_hazard_multiplier_frailty_matches_coxph_frailty() {
    init_parallelism();

    // ---- generate fixed-seed clustered censored survival data --------------
    // Generative model: exponential baseline hazard h0 = LAMBDA0, multiplicative
    // covariate effect exp(x*TRUE_BETA), shared lognormal frailty exp(U_g) per
    // group with U_g ~ N(0, TRUE_FRAILTY_SD^2). Event time T ~ Exp(rate) with
    // rate = LAMBDA0 * exp(x*beta + U_g); inverse-CDF sampling t = -ln(u)/rate.
    // Independent administrative censoring at CENS_TIME yields ~30-40% censoring,
    // with multiple events per group so the frailty is identifiable.
    const LAMBDA0: f64 = 0.10;
    const CENS_TIME: f64 = 12.0;

    let n = N_GROUPS * PER_GROUP;
    let mut rng = DetRng::new(0xC0FF_EE12_3456_789A);

    // One shared frailty per group (drawn once, applied to all its subjects).
    let mut frailty_u = vec![0.0_f64; N_GROUPS];
    for f in frailty_u.iter_mut() {
        *f = TRUE_FRAILTY_SD * rng.normal();
    }

    let mut t = Vec::<f64>::with_capacity(n);
    let mut event = Vec::<f64>::with_capacity(n);
    let mut x = Vec::<f64>::with_capacity(n);
    let mut g = Vec::<usize>::with_capacity(n);
    for grp in 0..N_GROUPS {
        for _ in 0..PER_GROUP {
            // Centred continuous covariate, deterministic per draw.
            let xi = rng.normal();
            let rate = LAMBDA0 * (xi * TRUE_BETA + frailty_u[grp]).exp();
            let u = rng.uniform();
            let t_event = -u.ln() / rate;
            let (obs, ev) = if t_event <= CENS_TIME {
                (t_event, 1.0)
            } else {
                (CENS_TIME, 0.0)
            };
            t.push(obs);
            event.push(ev);
            x.push(xi);
            g.push(grp);
        }
    }

    // Sanity on the simulated design: enough events overall and per group so the
    // shared frailty is identifiable rather than confounded with the baseline.
    let n_events: f64 = event.iter().sum();
    assert!(
        n_events >= 0.45 * n as f64 && n_events <= 0.9 * n as f64,
        "simulated event rate should leave a healthy mix of events/censoring: {n_events} / {n}"
    );
    let mut events_per_group = [0usize; N_GROUPS];
    for i in 0..n {
        if event[i] > 0.5 {
            events_per_group[g[i]] += 1;
        }
    }
    assert!(
        events_per_group.iter().all(|&c| c >= 2),
        "each frailty group needs multiple events for identifiability: {events_per_group:?}"
    );

    // ---- fit with gam: latent hazard-multiplier shared frailty -------------
    // Emit `group` as a string so the schema inferrer treats it as categorical;
    // gam's frailty machinery integrates the lognormal multiplier over the
    // clustered structure. `x` is the continuous PH covariate.
    let headers = vec![
        "t".to_string(),
        "event".to_string(),
        "x".to_string(),
        "g".to_string(),
    ];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                t[i].to_string(),
                event[i].to_string(),
                x[i].to_string(),
                format!("g{}", g[i]),
            ])
        })
        .collect();
    let data =
        encode_recordswith_inferred_schema(headers, rows).expect("encode frailty survival data");
    let col = data.column_map();
    let x_idx = col["x"];

    let cfg = FitConfig {
        // The latent hazard-window family integrates the lognormal frailty
        // exactly via the K_{k,m} kernels; it requires a non-linear scalar
        // baseline, so we use the Weibull scaffold with the flexible monotone
        // I-spline time basis (gam's smooth analogue of the Breslow baseline).
        survival_likelihood: "latent".to_string(),
        baseline_target: "weibull".to_string(),
        time_basis: "ispline".to_string(),
        frailty: Some(FrailtySpec::HazardMultiplier {
            sigma_fixed: None,
            loading: HazardLoading::Full,
        }),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("Surv(t, event) ~ x", &data, &cfg).expect("gam latent frailty fit");
    let FitResult::LatentSurvival(fit) = result else {
        panic!("expected a LatentSurvival fit result for survival_likelihood=latent");
    };

    // gam's learned frailty standard deviation is the log-scale spread of the
    // lognormal multiplier: exp(U_g), U_g ~ N(0, latent_sd^2). To compare against
    // coxph's gamma frailty `theta` we must put both on the SAME (multiplier)
    // scale. coxph parameterizes the gamma frailty with mean 1 and variance
    // theta = Var(multiplier). The matching estimand for the lognormal is the
    // variance of its multiplier exp(U_g):
    //     Var(exp(U)) = (exp(sigma^2) - 1) * exp(sigma^2)
    // (NOT sigma^2 itself, which is only the log-scale variance). Aligning on the
    // multiplier-variance grid is what makes the lognormal-vs-gamma comparison a
    // comparison of the same quantity.
    let gam_latent_sd = fit.latent_sd;
    let gam_log_var = gam_latent_sd * gam_latent_sd;
    let gam_frailty_var = (gam_log_var.exp() - 1.0) * gam_log_var.exp();

    // Fixed-effect log-HR for `x`. The latent fit stores blocks in order
    // [time-basis, mean (covariates), log_sigma]; the mean block is the linear
    // predictor on the covariates and is exactly `fit.design` rebuilt from
    // `fit.resolvedspec`. We isolate the per-unit `x` slope by differencing the
    // mean-block linear predictor between x = 1 and x = 0 (intercept/centering
    // cancels), which is precisely the PH log-HR that coxph reports as coef[x].
    let mean_beta = &fit.fit.block_states[1].beta;
    let ncov = data.headers.len();
    let mut anchor = Array2::<f64>::zeros((2, ncov));
    anchor[[1, x_idx]] = 1.0; // row 0: x=0, row 1: x=1
    let anchor_design = build_term_collection_design(anchor.view(), &fit.resolvedspec)
        .expect("rebuild mean (covariate) design at x anchors");
    assert_eq!(
        anchor_design.design.ncols(),
        mean_beta.len(),
        "mean design width must match the mean coefficient block"
    );
    let eta = anchor_design.design.apply(mean_beta).to_vec();
    let gam_beta_x = eta[1] - eta[0];

    // ---- fit the SAME data with survival::coxph(frailty(g)) ----------------
    // `frailty(g)` adds a gamma-distributed shared frailty; coxph estimates the
    // frailty variance theta by penalized profile likelihood and the PH log-HR
    // for x by partial likelihood. The converged theta is stored on the fitted
    // term's history; coef("x") is the fixed-effect log-HR.
    let group_code: Vec<f64> = g.iter().map(|&gi| (gi + 1) as f64).collect();
    let r = run_r(
        &[
            Column::new("t", &t),
            Column::new("event", &event),
            Column::new("x", &x),
            Column::new("g", &group_code),
        ],
        r#"
        suppressPackageStartupMessages(library(survival))
        df$g <- factor(df$g)
        m <- coxph(Surv(t, event) ~ x + frailty(g, distribution = "gamma"),
                   data = df, ties = "efron")
        emit("coef_x", as.numeric(coef(m)["x"]))
        # Converged gamma-frailty variance (theta) from the fitted frailty term.
        theta <- m$history[[1]]$theta
        if (is.null(theta)) {
          theta <- tail(m$history[[1]]$history[, 1], 1)
        }
        emit("frailty_var", as.numeric(theta))
        "#,
    );
    let r_coef_x = r.scalar("coef_x");
    let r_frailty_var = r.scalar("frailty_var");

    // ---- compare -----------------------------------------------------------
    let beta_abs = (gam_beta_x - r_coef_x).abs();
    let var_rel = (gam_frailty_var - r_frailty_var).abs() / r_frailty_var.max(1e-3);
    eprintln!(
        "coxph(frailty) vs gam HazardMultiplier: n={n} m_groups={N_GROUPS} events={n_events} \
         | beta_x: gam={gam_beta_x:.4} R={r_coef_x:.4} |diff|={beta_abs:.4} (true={TRUE_BETA}) \
         | mult_var: gam={gam_frailty_var:.4} (sd={gam_latent_sd:.4}) R={r_frailty_var:.4} \
         rel={var_rel:.4} (true_mult_var={:.4})",
        {
            let true_log_var = TRUE_FRAILTY_SD * TRUE_FRAILTY_SD;
            (true_log_var.exp() - 1.0) * true_log_var.exp()
        }
    );

    // Bound 1: PH log-HR agreement (L-infinity on the hazard scale).
    assert!(
        beta_abs <= 0.05,
        "fixed-effect log-HR diverges from coxph: gam={gam_beta_x:.4} R={r_coef_x:.4} |diff|={beta_abs:.4} > 0.05"
    );
    // Bound 2: frailty variance component agreement (relative, 15%).
    assert!(
        var_rel <= 0.15,
        "frailty variance diverges from coxph: gam={gam_frailty_var:.4} R={r_frailty_var:.4} rel={var_rel:.4} > 0.15"
    );
}

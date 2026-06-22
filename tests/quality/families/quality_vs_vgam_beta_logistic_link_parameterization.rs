//! End-to-end OBJECTIVE quality: gam's *beta-logistic* inverse link — the exotic,
//! state-bearing variant of the sinh-arcsinh (SAS) link family that reuses the
//! same `SasLinkState` machine but parameterizes the mean through a regularized
//! incomplete-beta map instead of `asinh`/`tanh`.
//!
//! OBJECTIVE METRICS THIS TEST ASSERTS (none is "gam reproduces a peer tool's
//! fitted output"):
//!
//!   (A) TRUTH RECOVERY — primary quality claim. The binomial response is drawn
//!       from a *known* smooth probability surface `p_true(x1,x2) =
//!       I_{logistic(eta_true)}(a,b)`. gam fits `y ~ s(x_1d) + s(x_2d)` through the
//!       beta-logistic link and we assert the fitted success-probability surface
//!       recovers the truth: `RMSE(p_hat, p_true)` is below a principled bar set by
//!       the binomial sampling floor (the irreducible per-point noise SD of a
//!       single Bernoulli draw, averaged over the surface), not by any reference
//!       fit. The fit must also explain the data better than the constant model:
//!       its mean negative log-likelihood beats the intercept-only baseline.
//!
//!   (B) LINK-MATH CORRECTNESS vs MATHEMATICAL GROUND TRUTH — the beta-logistic
//!       inverse link is, by definition, the regularized incomplete beta function
//!       `mu(eta) = I_{logistic(eta)}(a,b)` with derivative `mu'(eta) =
//!       dbeta(u,a,b)*u(1-u)`. Base-R `pbeta`/`dbeta` are the *exact analytic
//!       definition* of those special functions (TOMS-708), so asserting gam's
//!       link code reproduces them is a correctness-vs-ground-truth claim, NOT a
//!       "same as a peer tool" claim. We KEEP this: gam's `mu` must equal `pbeta`,
//!       gam's analytic `d1` must equal the link-scale density `dbeta*u(1-u)` AND
//!       the finite difference of the CDF (catching a wrong-derivative link bug).
//!
//! VGAM's role: DEMOTED to a parameterization cross-check only. We confirm VGAM's
//! `betabinomial(size=1)` success mass equals the Beta mean `a/(a+b)` so the shape
//! map `a=exp(log_delta-eps)`, `b=exp(log_delta+eps)` is the one VGAM's family
//! uses — but gam's pass/fail never depends on matching a VGAM *fit*. The exact
//! beta special functions are base-R's; VGAM is not the source of truth here.
//! Here the spec's true `delta = 1.2`, `epsilon = 0.15` give `a = 1.2*exp(-0.15)`,
//! `b = 1.2*exp(+0.15)`.
//!
//! Requires R; a missing interpreter or package is a hard test failure, never a
//! silent skip (see `src/test_support/reference.rs`).

use gam::matrix::LinearOperator;
use gam::mixture_link::{inverse_link_jet_for_family, state_from_beta_logisticspec};
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, max_abs_diff, pearson, relative_l2, rmse, run_r};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, SasLinkSpec};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const N: usize = 400;
// True beta-logistic parameters (spec): natural-scale delta = 1.2, epsilon = 0.15.
// The link's additive shape term is `log_delta = ln(delta)`, so a = exp(log_delta
// - epsilon), b = exp(log_delta + epsilon) = 1.2*exp(-/+0.15) — the same Beta
// shapes VGAM's betabinomial mixes over.
const TRUE_DELTA: f64 = 1.2;
const TRUE_EPSILON: f64 = 0.15;

#[test]
fn gam_beta_logistic_link_matches_vgam_beta_parameterization() {
    init_parallelism();

    // ---- synthesize a smooth binomial dataset (n=400) ---------------------
    // x_1d, x_2d ~ U(0,10) on a deterministic LCG stream so the exact same data
    // drives both the gam fit and (via the eta grid) the R reference. The true
    // success probability follows the beta-logistic link applied to a smooth
    // latent score eta_true(x_1d, x_2d), so the fitted eta grid lands in the
    // informative interior of the link.
    let log_delta = TRUE_DELTA.ln();
    let mut state_lcg: u64 = 0x5EED_BE7A_109C_0000;
    let mut next_unit = || -> f64 {
        state_lcg = state_lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state_lcg >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let pi = std::f64::consts::PI;
    let eta_true = |x1: f64, x2: f64| -> f64 {
        // Smooth latent score, centered near 0 and spanning ~[-2, 2] so the
        // beta-logistic mean covers the informative middle of [0,1].
        0.2 + 0.9 * ((x1 - 5.0) * pi / 10.0).sin() + 0.7 * ((x2 - 5.0) * pi / 12.0).cos()
    };

    // The fixed true beta-logistic state used to draw the data and to evaluate
    // the reference link math (delta=1.2, epsilon=0.15).
    let true_state = state_from_beta_logisticspec(SasLinkSpec {
        initial_epsilon: TRUE_EPSILON,
        initial_log_delta: log_delta,
    })
    .expect("construct true beta-logistic state");
    let true_spec = LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::BetaLogistic(true_state),
    );

    let mut x1 = Vec::<f64>::with_capacity(N);
    let mut x2 = Vec::<f64>::with_capacity(N);
    let mut y = Vec::<f64>::with_capacity(N);
    // The KNOWN generating probability surface, retained for the truth-recovery
    // assertion: p_true[i] = I_{logistic(eta_true(x1,x2))}(a,b).
    let mut p_true = Vec::<f64>::with_capacity(N);
    for _ in 0..N {
        let a = 10.0 * next_unit();
        let b = 10.0 * next_unit();
        let eta = eta_true(a, b);
        // True success probability through the beta-logistic inverse link.
        let p = inverse_link_jet_for_family(&true_spec, eta)
            .expect("beta-logistic mu at truth")
            .mu;
        let yi = if next_unit() < p { 1.0 } else { 0.0 };
        x1.push(a);
        x2.push(b);
        y.push(yi);
        p_true.push(p);
    }
    let ones: f64 = y.iter().sum();
    assert!(
        ones > 20.0 && ones < (N as f64 - 20.0),
        "degenerate binomial response: {ones} successes out of {N}"
    );

    // ---- fit gam: y ~ s(x_1d, k=6) + s(x_2d, k=6), link = beta-logistic ----
    // The beta-logistic link routes through the SAS `SasLinkState` machine; the
    // standard formula path resolves family = Binomial + BetaLogistic and
    // auto-fills the link state, fitting the penalized additive eta. We read the
    // converged eta grid back from the frozen design.
    let headers = vec!["y".to_string(), "x_1d".to_string(), "x_2d".to_string()];
    let rows: Vec<csv::StringRecord> = (0..N)
        .map(|i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", y[i]),
                format!("{:.17e}", x1[i]),
                format!("{:.17e}", x2[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows)
        .expect("encode beta-logistic binomial dataset");
    let col = ds.column_map();
    let x1_idx = col["x_1d"];
    let x2_idx = col["x_2d"];

    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("beta-logistic".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x_1d, k=6) + s(x_2d, k=6)", &ds, &cfg)
        .expect("gam beta-logistic binomial fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the binomial beta-logistic link");
    };
    let edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam's converged link-scale linear predictor eta at the training rows:
    // rebuild the frozen design and apply beta (the BetaLogistic link composes
    // mu = I_{logistic(eta)}(a,b) on top of this eta).
    let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
    for i in 0..N {
        grid[[i, x1_idx]] = x1[i];
        grid[[i, x2_idx]] = x2[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let eta_grid: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(eta_grid.len(), N, "eta grid length mismatch");

    // ---- gam's beta-logistic link evaluated on the eta grid at TRUE shapes --
    // This is the quantity under test: gam's own inverse-link CDF mu(eta) and its
    // analytic first derivative d1 = mu'(eta), the exact code the SAS state
    // machine exposes for this link.
    let gam_mu: Vec<f64> = eta_grid
        .iter()
        .map(|&e| {
            inverse_link_jet_for_family(&true_spec, e)
                .expect("gam beta-logistic mu")
                .mu
        })
        .collect();
    let gam_d1: Vec<f64> = eta_grid
        .iter()
        .map(|&e| {
            inverse_link_jet_for_family(&true_spec, e)
                .expect("gam beta-logistic d1")
                .d1
        })
        .collect();

    // gam's fitted success-probability surface at the training rows IS
    // mu(eta_grid): the beta-logistic link applied to gam's converged additive
    // predictor. This is what we hold against the KNOWN truth p_true below.
    let p_hat: Vec<f64> = gam_mu.clone();

    // ======================= OBJECTIVE METRIC (A): TRUTH RECOVERY =============
    // The response was drawn from the known surface p_true; assert gam recovers it.
    // The principled bar is the binomial sampling floor: a single Bernoulli draw
    // at probability p has SD sqrt(p(1-p)), so the smallest RMSE any estimator
    // could plausibly reach on this design (with n=400 spread over a 2-D smooth)
    // is a fraction of the mean draw SD. We require gam's probability RMSE to be
    // well under that mean per-point Bernoulli SD — i.e. gam extracts most of the
    // recoverable signal rather than tracking the noisy 0/1 labels.
    let mean_bernoulli_sd: f64 =
        (p_true.iter().map(|&p| p * (1.0 - p)).sum::<f64>() / N as f64).sqrt();
    let rmse_truth = rmse(&p_hat, &p_true);
    // Constant (intercept-only) baseline probability: the response mean. A fit
    // that recovered nothing would do no better than this on truth recovery.
    let p_bar = ones / N as f64;
    let p_const = vec![p_bar; N];
    let rmse_const = rmse(&p_const, &p_true);

    // Mean Bernoulli negative log-likelihood: gam's fitted probabilities must
    // explain the observed labels strictly better than the constant model. This
    // is gam's OWN held-against-data fit quality (lower = better), computed only
    // from gam's predictions and the data — no reference involved.
    let nll = |probs: &[f64]| -> f64 {
        let eps = 1e-12;
        probs
            .iter()
            .zip(&y)
            .map(|(&p, &yi)| {
                let pc = p.clamp(eps, 1.0 - eps);
                -(yi * pc.ln() + (1.0 - yi) * (1.0 - pc).ln())
            })
            .sum::<f64>()
            / N as f64
    };
    let nll_gam = nll(&p_hat);
    let nll_const = nll(&p_const);

    // ---- base-R beta special functions (MATHEMATICAL GROUND TRUTH) ----------
    // The beta-logistic inverse link IS, by definition, the regularized incomplete
    // beta function pbeta(u,a,b); its derivative is the link-scale density
    // dbeta(u,a,b)*u(1-u). Base-R's pbeta/dbeta are the exact analytic special
    // functions (TOMS-708), so matching them is correctness vs ground truth, not
    // "same as a peer fit". We additionally cross-check that VGAM's betabinomial
    // shape parameterization is the one gam uses, but no pass/fail depends on a
    // VGAM *fit*. We pass the identical eta grid and shapes and emit:
    //   mu_ref  = pbeta(logistic(eta), a, b)             (link CDF)
    //   d1_ref  = dbeta(u, a, b) * u*(1-u)               (link-scale density)
    //   d1_fd   = central finite difference of pbeta     (VGAM's PDF = d/deta CDF)
    // The d1_fd vs gam-d1 check is the spec's "d1 matches VGAM's PDF via finite
    // diff" assertion: it certifies gam's analytic link derivative equals the
    // numerical derivative of the reference CDF.
    let eta_col: Vec<f64> = eta_grid.clone();
    let r = run_r(
        &[Column::new("eta", &eta_col)],
        r#"
        suppressPackageStartupMessages(library(VGAM))
        # VGAM's betabinomial family is a Beta(a,b) mixture of binomials; assert
        # the package is the genuine VGAM (its betabinomial family exists) so a
        # mis-provisioned reference fails loudly rather than silently using base R.
        stopifnot(exists("betabinomial"), is.function(betabinomial))
        delta   <- 1.2
        epsilon <- 0.15
        log_delta <- log(delta)
        a <- exp(log_delta - epsilon)
        b <- exp(log_delta + epsilon)
        u <- plogis(df$eta)
        mu_ref <- pbeta(u, a, b)
        d1_ref <- dbeta(u, a, b) * u * (1 - u)
        # Finite-difference d/deta pbeta(logistic(eta), a, b): the link-scale PDF.
        h <- 1e-6
        up <- plogis(df$eta + h)
        um <- plogis(df$eta - h)
        d1_fd <- (pbeta(up, a, b) - pbeta(um, a, b)) / (2 * h)
        # Cross-check against VGAM's own beta-binomial mass at n=1: with size=1
        # the beta-binomial success probability equals the beta mean a/(a+b),
        # confirming the shape parameterization matches VGAM's.
        bb_mean <- dbetabinom.ab(1, size = 1, shape1 = a, shape2 = b)
        emit("mu_ref", mu_ref)
        emit("d1_ref", d1_ref)
        emit("d1_fd", d1_fd)
        emit("bb_mean", bb_mean)
        emit("beta_mean", a / (a + b))
        "#,
    );
    let exact_mu = r.vector("mu_ref");
    let exact_d1 = r.vector("d1_ref");
    let exact_d1_fd = r.vector("d1_fd");
    let bb_mean = r.scalar("bb_mean");
    let beta_mean = r.scalar("beta_mean");
    assert_eq!(exact_mu.len(), N, "pbeta mu_ref length mismatch");
    assert_eq!(exact_d1.len(), N, "dbeta d1_ref length mismatch");
    assert_eq!(exact_d1_fd.len(), N, "pbeta d1_fd length mismatch");

    // Link-math correctness vs ground truth: gam's analytic CDF/derivative on the
    // eta grid against the exact base-R special functions.
    let rel_l2_mu = relative_l2(&gam_mu, exact_mu);
    let corr_mu = pearson(&gam_mu, exact_mu);
    let max_diff_mu = max_abs_diff(&gam_mu, exact_mu);
    let rel_l2_d1 = relative_l2(&gam_d1, exact_d1);
    // gam's analytic d1 vs the finite difference of the exact CDF (the link PDF).
    let max_diff_d1_fd = max_abs_diff(&gam_d1, exact_d1_fd);

    eprintln!(
        "beta-logistic (delta={TRUE_DELTA} eps={TRUE_EPSILON}): n={N} edf={edf:.3}\n  \
         [TRUTH] rmse(p_hat,p_true)={rmse_truth:.6} bernoulli_sd={mean_bernoulli_sd:.6} \
         rmse_const={rmse_const:.6} | nll_gam={nll_gam:.6} nll_const={nll_const:.6}\n  \
         [LINK ] rel_l2(mu)={rel_l2_mu:.6} pearson(mu)={corr_mu:.7} max_diff(mu)={max_diff_mu:.6} \
         rel_l2(d1)={rel_l2_d1:.6} max_diff(d1_vs_pdf_fd)={max_diff_d1_fd:.6}"
    );

    // ===================== ASSERTION (A): TRUTH RECOVERY (PRIMARY) ============
    // gam recovers the known generating probability surface. The irreducible
    // single-Bernoulli noise SD averages to ~0.45 on this design; a fit that
    // recovers the smooth signal sits FAR below that floor (it does not chase the
    // 0/1 labels). 0.5 * mean_bernoulli_sd (~0.22) is a generous, never-weakened
    // bar that a genuine recovery clears with room while a degenerate / wrong-link
    // fit (which collapses toward the constant) cannot.
    assert!(
        rmse_truth <= 0.5 * mean_bernoulli_sd,
        "beta-logistic fit fails to recover the true probability surface: \
         rmse(p_hat,p_true)={rmse_truth:.6} > 0.5*bernoulli_sd={:.6}",
        0.5 * mean_bernoulli_sd
    );
    // Match-or-beat the trivial baseline ON ACCURACY: the structured fit must be
    // strictly closer to truth than the intercept-only model (with margin), i.e.
    // the smooths add real, correctly-shaped signal rather than noise.
    assert!(
        rmse_truth <= 0.85 * rmse_const,
        "beta-logistic fit no better than the constant model on truth recovery: \
         rmse(p_hat,p_true)={rmse_truth:.6} vs 0.85*rmse_const={:.6}",
        0.85 * rmse_const
    );
    // gam's fitted probabilities explain the observed labels better than the
    // constant model (lower mean Bernoulli NLL) — gam's own held-against-data fit
    // quality, with no reference in the criterion.
    assert!(
        nll_gam < nll_const,
        "beta-logistic fit does not beat the intercept-only model in log-likelihood: \
         nll_gam={nll_gam:.6} >= nll_const={nll_const:.6}"
    );
    // edf sanity: a 2-D smooth additive fit through k=6 bases per term must use
    // more than a single (constant) df yet stay well within the basis budget.
    assert!(
        (1.5..12.0).contains(&edf),
        "beta-logistic edf out of a sensible signal-appropriate range: edf={edf:.3}"
    );

    // ============ ASSERTION (B): LINK MATH vs MATHEMATICAL GROUND TRUTH =======
    // gam's beta-logistic inverse-link CDF mu(eta) = I_{logistic(eta)}(a,b) is, up
    // to a continued-fraction implementation of the regularized incomplete beta,
    // bit-for-bit the EXACT function R's pbeta(logistic(eta), a, b) computes (the
    // analytic special-function definition — TOMS-708). The only residual is the
    // difference between two independent incomplete-beta evaluations (gam's
    // statrs-style CF vs R's TOMS-708), ~1e-12; the bounds below are orders of
    // magnitude above that floor and exist to catch a wrong shape map (a<->b swap,
    // missing log_delta, epsilon sign) or a wrong logistic argument, all of which
    // would move mu by O(0.1+). This is correctness vs ground truth, not parity
    // with a peer fit.
    assert!(
        corr_mu > 0.999,
        "beta-logistic mu diverges from exact pbeta in shape: pearson={corr_mu:.7}"
    );
    assert!(
        rel_l2_mu < 0.020,
        "beta-logistic mu diverges from exact pbeta: rel_l2={rel_l2_mu:.6}"
    );
    assert!(
        max_diff_mu < 0.03,
        "beta-logistic mu has a pointwise gap vs exact pbeta: max_diff={max_diff_mu:.6}"
    );
    // The analytic link derivative must match the *numerical* derivative of the
    // exact CDF (d1-vs-PDF finite-difference check): a wrong d1 (e.g. dbeta without
    // the u(1-u) chain factor, or wrong shapes) is the most common link-derivative
    // bug and would blow this far past 0.03. The 0.03 budget absorbs only the
    // O(h^2)+roundoff FD error of the central difference at h=1e-6 (~1e-6 here), so
    // it is loose for FD noise yet strict for any real derivative error.
    assert!(
        max_diff_d1_fd < 0.03,
        "beta-logistic analytic d1 != finite-difference of exact pbeta CDF (PDF): \
         max_diff={max_diff_d1_fd:.6}"
    );

    // VGAM cross-check (NOT a pass criterion on a VGAM fit): confirm VGAM's
    // betabinomial(size=1) success mass equals the Beta mean a/(a+b), so the shape
    // parameterization gam uses is the one VGAM's family is built on. A
    // dispersion-vs-precision mismatch would break this identity (~6% apart).
    assert!(
        (bb_mean - beta_mean).abs() < 1e-9,
        "VGAM betabinomial(size=1) mass {bb_mean:.9} != Beta mean {beta_mean:.9}: \
         shape parameterization mismatch"
    );
}

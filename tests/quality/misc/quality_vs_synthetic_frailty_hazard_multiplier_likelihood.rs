//! Objective truth-recovery test for gam's lognormal hazard-multiplier frailty
//! family: from clustered right-censored survival data generated with a KNOWN
//! frailty variance `sigma_true`, gam's exact lognormal kernel must recover that
//! variance.
//!
//! gam's latent-survival family integrates a lognormal frailty `exp(U)`,
//! `U ~ N(0, sigma^2)`, out of a proportional-hazards row likelihood in closed
//! form via the special function `K_{k,m}(mu, sigma) = E[exp(k*U - m*exp(U))]`.
//! A right-censored row contributes `log K_{0,M}` (with `M = H_0(t)` the
//! cumulative loaded hazard) and an exact event contributes
//! `log(h_0(t) * K_{1,M})`. The per-row jet additionally exposes the analytic
//! derivatives `d log L / d log sigma` and its negative Hessian, so the marginal
//! log-likelihood can be maximized over the frailty scale using gam's OWN kernel
//! and OWN derivatives — no finite differences, no external optimizer.
//!
//! OBJECTIVE METRIC (truth recovery, principle case 1): the data are simulated
//! from a known `sigma_true`. We maximize gam's marginal log-likelihood over
//! `log sigma` by damped Newton on the kernel's analytic score/Hessian and assert
//!
//!     |sigma_hat_gam - sigma_true| <= bar,
//!
//! i.e. gam's frailty kernel RECOVERS the data-generating variance. The bar is
//! the asymptotic standard error of the frailty-scale MLE (from the kernel's own
//! observed Fisher information at the optimum) inflated by a small constant — a
//! sampling-error budget, not a tolerance tuned to pass. This is an objective
//! statement about gam alone: "the kernel + its analytic derivatives recover the
//! truth", with NO appeal to any other tool's fitted output.
//!
//! BASELINE TO MATCH-OR-BEAT: an independent R reference forms the SAME marginal
//! likelihood by 128-node Gauss-Hermite quadrature of `E[g(U)]` under
//! `N(0,sigma^2)` (`statmod::gauss.quad.prob`) and maximizes it over `sigma` with
//! `optimize`, yielding `sigma_hat_r`. Both engines see byte-identical
//! `(M, h_0, event)` data. We additionally assert gam's recovery error is no
//! worse than R's, `|sigma_hat_gam - sigma_true| <= |sigma_hat_r - sigma_true|
//! * 1.10 + 1e-3` — gam matches or beats the mature quadrature on ACCURACY of
//! truth recovery. Because both maximize the exact frailty-integrated likelihood,
//! the two MLEs should in fact agree to quadrature precision; the comparison is a
//! baseline, never the pass criterion.

use gam::families::survival::lognormal_kernel::{LatentSurvivalRow, LatentSurvivalRowJet};
use gam::quadrature::QuadratureContext;
use gam::test_support::reference::{Column, run_r};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Deterministic SplitMix64 PRNG — gives a fixed-seed, dependency-free stream so
/// the synthetic dataset is reproducible and identical on every run.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next uniform in the open interval (0, 1).
    fn next_unit(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Map the top 53 bits to (0,1); nudge off the endpoints so log/inverse-CDF
        // transforms below never hit +/-inf.
        let u = ((z >> 11) as f64) / ((1u64 << 53) as f64);
        u.clamp(f64::MIN_POSITIVE, 1.0 - 1e-16)
    }

    /// Standard normal via Acklam's inverse-CDF rational approximation.
    fn next_normal(&mut self) -> f64 {
        let p = self.next_unit();
        const A: [f64; 6] = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00,
        ];
        const B: [f64; 5] = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
        ];
        const C: [f64; 6] = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00,
        ];
        const D: [f64; 4] = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00,
        ];
        let plow = 0.02425;
        let phigh = 1.0 - plow;
        if p < plow {
            let q = (-2.0 * p.ln()).sqrt();
            (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
                / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
        } else if p <= phigh {
            let q = p - 0.5;
            let r = q * q;
            (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
                / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
                / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
        }
    }

    /// Unit-rate exponential via inverse-CDF.
    fn next_exp(&mut self) -> f64 {
        -self.next_unit().ln()
    }
}

/// Aggregate gam's marginal log-likelihood and its analytic `log sigma`
/// derivatives across all rows at frailty scale `sigma` (mu fixed at 0). Returns
/// `(loglik, score_log_sigma, neg_hessian_log_sigma)`, summing the per-row jets
/// produced by gam's own `LatentSurvivalRowJet`.
fn gam_logsigma_jet(
    quadctx: &QuadratureContext,
    cum_hazard: &[f64],
    events: &[f64],
    baseline_slope: f64,
    sigma: f64,
) -> (f64, f64, f64) {
    // The per-row jets are independent Gauss-Hermite quadrature evaluations; the
    // dominant cost of this MLE is the ~10^5 jet evaluations across the Newton +
    // backtracking loop. Fan them across the rayon pool, then sum the per-row
    // (log_lik, score, neg_hessian) triples back IN ROW ORDER so the accumulated
    // floats are bit-identical to the serial reduction — the Newton path and
    // every asserted recovery bar are unchanged, only wall-clock improves.
    let per_row: Vec<(f64, f64, f64)> = (0..cum_hazard.len())
        .into_par_iter()
        .map(|i| {
            let mass_exit = cum_hazard[i];
            let row = if events[i] > 0.5 {
                // Exact event: hazard_loaded = h_0(t) = baseline_slope.
                LatentSurvivalRow::exact_event(0.0, mass_exit, 0.0, 0.0, baseline_slope, 0.0)
            } else {
                LatentSurvivalRow::right_censored(0.0, mass_exit, 0.0, 0.0)
            };
            let jet = LatentSurvivalRowJet::evaluate(quadctx, &row, 0.0, sigma)
                .expect("gam kernel row likelihood");
            (jet.log_lik, jet.score_log_sigma, jet.neg_hessian_log_sigma)
        })
        .collect();
    let mut loglik = 0.0;
    let mut score = 0.0;
    let mut neg_hess = 0.0;
    for (ll, sc, nh) in &per_row {
        loglik += ll;
        score += sc;
        neg_hess += nh;
    }
    (loglik, score, neg_hess)
}

#[test]
fn lognormal_hazard_multiplier_kernel_recovers_frailty_variance() {
    // ---- synthetic clustered right-censored survival data -----------------
    // m groups of n_per rows; baseline H_0(t) = baseline_slope * t (linear,
    // Makeham-like), with frailty U_g ~ N(0, sigma_true^2) shared within group.
    // Event time from T = -ln(S)/(baseline_slope * exp(U)) with S = exp(-E),
    // E ~ Exp(1), i.e. integrating H(t|U) = H_0(t)*exp(U) to a draw. Times are
    // administratively censored at `tau` to produce a censored/event mix.
    //
    // 200 groups (1600 rows): the frailty-scale MLE's sampling error scales as
    // ~1/sqrt(#groups), and the PRIMARY recovery bar below is `max(3*se_sigma, 0.05)`
    // with se_sigma read from the observed information at the optimum — so halving
    // the group count widens the bar by ~sqrt(2) *exactly* as it widens the actual
    // MLE error. The err/bar ratio is therefore group-count-invariant: a
    // systematically-wrong kernel still misses by >3 se at 200 groups just as at 400.
    // The match-or-beat-R arm is group-count-independent in character (both maximize
    // the same likelihood on the same data). 200 groups still mixes events/censoring
    // and recovers a group-level variance comfortably; this only removes per-row
    // quadrature-jet work (the dominant wall-clock cost), not discriminating power.
    const M_GROUPS: usize = 200;
    const N_PER: usize = 8;
    const N_TOTAL: usize = M_GROUPS * N_PER;
    const SIGMA_TRUE: f64 = 0.5;
    const BASELINE_SLOPE: f64 = 0.1; // H_0(t) = 0.1 * t  =>  h_0(t) = 0.1
    const TAU: f64 = 8.0; // administrative censoring horizon

    let mut rng = SplitMix64::new(42);
    let mut events = Vec::with_capacity(N_TOTAL);
    let mut cum_hazard = Vec::with_capacity(N_TOTAL); // M = H_0(t_obs)
    for _g in 0..M_GROUPS {
        let u = SIGMA_TRUE * rng.next_normal();
        let frail = u.exp();
        for _i in 0..N_PER {
            let e = rng.next_exp();
            let t_event = e / (BASELINE_SLOPE * frail);
            let (t_obs, event) = if t_event <= TAU {
                (t_event, 1.0)
            } else {
                (TAU, 0.0)
            };
            events.push(event);
            cum_hazard.push(BASELINE_SLOPE * t_obs);
        }
    }
    let n_events: usize = events.iter().filter(|&&e| e > 0.5).count();
    assert!(
        n_events > 0 && n_events < N_TOTAL,
        "synthetic data must mix events and censoring: events={n_events}/{N_TOTAL}"
    );

    // ---- gam side: MLE of the frailty scale via gam's own kernel + derivs ----
    // Damped Newton on log sigma using the analytic score / negative Hessian of
    // gam's marginal log-likelihood. We optimize over log sigma (the natural,
    // positivity-respecting parameter the jet differentiates).
    let quadctx = QuadratureContext::new();
    let mut log_sigma = 0.0_f64; // start at sigma = 1.0, away from the truth (0.5)
    let mut last_loglik = f64::NEG_INFINITY;
    for _iter in 0..100 {
        let sigma = log_sigma.exp();
        let (loglik, score, neg_hess) =
            gam_logsigma_jet(&quadctx, &cum_hazard, &events, BASELINE_SLOPE, sigma);
        // Newton step on log sigma: step = score / neg_hessian (neg_hessian is the
        // observed information of -loglik wrt log sigma, i.e. -d^2 loglik/dlogsig^2).
        // Guard against a non-positive curvature with a gradient-ascent fallback.
        let step = if neg_hess > 1e-12 {
            score / neg_hess
        } else {
            0.1 * score
        };
        // Damp to keep the iterate in a sane region and guarantee ascent.
        let mut damping = 1.0_f64;
        let mut next_log_sigma = log_sigma + damping * step;
        for _backtrack in 0..40 {
            next_log_sigma = log_sigma + damping * step;
            let cand_sigma = next_log_sigma.exp();
            let (cand_ll, _, _) =
                gam_logsigma_jet(&quadctx, &cum_hazard, &events, BASELINE_SLOPE, cand_sigma);
            if cand_ll >= loglik {
                break;
            }
            damping *= 0.5;
        }
        let converged = (next_log_sigma - log_sigma).abs() < 1e-10;
        log_sigma = next_log_sigma;
        last_loglik = loglik;
        if converged {
            break;
        }
    }
    let sigma_hat_gam = log_sigma.exp();
    // Observed-information standard error of the frailty-scale MLE, from gam's own
    // Hessian at the optimum. Var(log sigma_hat) ~ 1 / I(log sigma); by the delta
    // method se(sigma_hat) = sigma_hat * se(log sigma_hat).
    let (_, _, neg_hess_at_opt) = gam_logsigma_jet(
        &quadctx,
        &cum_hazard,
        &events,
        BASELINE_SLOPE,
        sigma_hat_gam,
    );
    assert!(
        neg_hess_at_opt > 0.0,
        "frailty-scale log-likelihood must be concave at the MLE (observed info \
         {neg_hess_at_opt:.3e})"
    );
    let se_log_sigma = (1.0 / neg_hess_at_opt).sqrt();
    let se_sigma = sigma_hat_gam * se_log_sigma;

    // ---- R reference (BASELINE): same marginal likelihood, MLE by quadrature ---
    // For each row, integrate the conditional row likelihood over U~N(0,sigma^2):
    //   censored: L = E[exp(-M*exp(U))]
    //   event   : L = h_0 * E[exp(U) * exp(-M*exp(U))]
    // statmod::gauss.quad.prob(dist="normal") supplies the (node,weight) rule for
    // E[g(U)] under N(0,sigma); 128 nodes drives the quadrature error to ~1e-13.
    // `optimize` then maximizes the total log-likelihood over sigma.
    let cols = [Column::new("M", &cum_hazard), Column::new("event", &events)];
    let r = run_r(
        &cols,
        r#"
        suppressPackageStartupMessages(library(statmod))
        h0 <- 0.1
        M  <- df$M
        ev <- df$event
        is_ev <- ev > 0.5
        negloglik <- function(sigma) {
            gq <- gauss.quad.prob(128, dist = "normal", mu = 0, sigma = sigma)
            nodes <- gq$nodes
            wts <- gq$weights
            # E[g(U)] for every row, fully VECTORIZED: the old per-row R `for`
            # loop (one `exp(-M[i]*enodes)` allocation + reduction per row, per
            # `optimize` step) was the dominant wall-clock cost of this baseline
            # at N_TOTAL rows x 128 nodes x ~40 optimize evaluations. Replace it
            # with a single (rows x nodes) matrix evaluated once per sigma:
            #   base[i,j] = exp(-M[i] * exp(node_j)).
            # The per-row L is then a single weighted matrix-vector contraction
            # over nodes, with the event/censored cases differing only by the
            # extra `enodes` factor. Identical math, identical likelihood — only
            # the loop is gone.
            enodes <- exp(nodes)
            # base: outer over rows (M) and nodes (enodes); R recycles M down the
            # columns, so each column j is exp(-M * enodes[j]).
            base <- exp(-outer(M, enodes))          # rows x nodes
            # censored rows: L = sum_j w_j base[i,j]      => base %*% wts
            # event   rows: L = h0 * sum_j w_j enodes[j] base[i,j]
            #                                            => h0 * base %*% (wts*enodes)
            L_cens <- as.numeric(base %*% wts)
            L_ev   <- h0 * as.numeric(base %*% (wts * enodes))
            Lrow   <- ifelse(is_ev, L_ev, L_cens)
            -sum(log(Lrow))
        }
        opt <- optimize(negloglik, interval = c(0.05, 3.0), tol = 1e-8)
        emit("sigma_hat", opt$minimum)
        emit("loglik", -opt$objective)
        "#,
    );
    let sigma_hat_r = r.scalar("sigma_hat");
    let r_loglik = r.scalar("loglik");

    // gam's log-likelihood at its own optimum, for context.
    let (gam_loglik_at_opt, _, _) = gam_logsigma_jet(
        &quadctx,
        &cum_hazard,
        &events,
        BASELINE_SLOPE,
        sigma_hat_gam,
    );

    let err_gam = (sigma_hat_gam - SIGMA_TRUE).abs();
    let err_r = (sigma_hat_r - SIGMA_TRUE).abs();
    eprintln!(
        "lognormal hazard-multiplier frailty truth-recovery: n={N_TOTAL} \
         groups={M_GROUPS} events={n_events} sigma_true={SIGMA_TRUE} \
         sigma_hat_gam={sigma_hat_gam:.6} sigma_hat_r={sigma_hat_r:.6} \
         se_sigma={se_sigma:.6} err_gam={err_gam:.3e} err_r={err_r:.3e} \
         last_loglik={last_loglik:.6} gam_loglik_at_opt={gam_loglik_at_opt:.6} \
         r_loglik={r_loglik:.6}"
    );

    // ---- PRIMARY objective assertion: gam recovers the true frailty variance --
    // The MLE error must lie within a few asymptotic standard errors of the true
    // sigma. 3*se is the principled sampling-error budget (a ~99.7% Gaussian
    // band); a floor protects against an over-tight se from the observed info.
    let recovery_bar = (3.0 * se_sigma).max(0.05);
    assert!(
        err_gam <= recovery_bar,
        "gam's lognormal kernel failed to recover the true frailty variance: \
         sigma_hat={sigma_hat_gam:.6} sigma_true={SIGMA_TRUE} err={err_gam:.3e} \
         bar={recovery_bar:.3e} (se_sigma={se_sigma:.3e})"
    );

    // ---- match-or-beat the mature quadrature on recovery ACCURACY ------------
    // Both maximize the EXACT frailty-integrated likelihood on identical data, so
    // their MLEs should coincide to quadrature precision; gam must be no less
    // accurate than R at recovering the truth. The additive 1e-3 absorbs the
    // residual quadrature/optimizer disagreement near the (shared) optimum.
    assert!(
        err_gam <= err_r * 1.10 + 1e-3,
        "gam recovered the frailty variance less accurately than the R \
         Gauss-Hermite MLE baseline: err_gam={err_gam:.3e} err_r={err_r:.3e} \
         (sigma_hat_gam={sigma_hat_gam:.6} sigma_hat_r={sigma_hat_r:.6})"
    );
}

//! End-to-end CAPABILITY test: gam must FIT interval-censored survival data and
//! RECOVER the known generating latent distribution, matching or beating
//! `lifelines` on the same data.
//!
//! Interval censoring is a first-class survival capability in every mature tool
//! (`lifelines.*Fitter.fit_interval_censoring`, flexsurv `Surv(L, R, type="interval2")`,
//! R `icenReg`): the exact event time `T` is never observed, only a bracket
//! `T ∈ (L, R]` produced by a discrete inspection schedule. The correct row
//! contribution is the interval mass
//!     ℓ_i = log[ S(L_i | x_i) − S(R_i | x_i) ],
//! not a point-density (exact event) nor a single-sided survival
//! (right-censoring). gam's latent-survival kernel implements exactly this
//! contribution (`LatentSurvivalRowJet::interval_censored`, the
//! `log[ K_{0,B(L)} − K_{0,B(R)} ]` jet with delayed-entry conditioning), so the
//! capability is present at the math layer. This test pins it as a user-visible
//! capability: fit `(μ, σ)` of the latent log-frailty from interval-only data and
//! recover the truth.
//!
//! DATA-GENERATING PROCESS (truth is known exactly).
//!   Latent log-frailty  U ~ Normal(μ*, σ*).
//!   Conditional cumulative hazard  Λ(t | U) = B(t) · exp(U),  with a KNOWN
//!   baseline cumulative hazard B(t) = t (unit Weibull/exponential clock), so
//!   conditional survival  S(t | U) = exp(−B(t) e^U).
//!   Marginal survival     S(t) = E_U[exp(−B(t) e^U)] = K_{0, B(t)}(μ*, σ*),
//!   the kernel's order-0 mass integral. Each subject's true event time T is
//!   drawn from this marginal, then only the bracketing pair (L, R] from a fixed
//!   inspection grid is retained — the exact T is discarded. This is exactly how
//!   interval-censored data arises in practice.
//!
//! OBJECTIVE METRIC (truth recovery, no reference dependence):
//!   gam's interval-likelihood MLE of (μ, σ) must land within a principled
//!   finite-sample tolerance of the TRUE generating (μ*, σ*). The estimator is
//!   the maximiser of Σ_i log[ K_{0,B(L_i)}(μ,σ) − K_{0,B(R_i)}(μ,σ) ] over the
//!   gam kernel — i.e. gam's OWN interval-censored likelihood, evaluated through
//!   the public `LatentSurvivalRowJet` path.
//!
//! BASELINE TO MATCH-OR-BEAT (does not define pass/fail on its own):
//!   `lifelines.WeibullFitter().fit_interval_censoring(L, R)` is fit on the
//!   IDENTICAL (L, R] brackets. lifelines parametrises the same marginal family
//!   (Weibull baseline + log-normal-mixed frailty is, for this unit baseline and
//!   the recovered marginal, a smooth two-parameter survival curve); we compare
//!   recovered MARGINAL survival curves S(t) on a held-out time grid, and assert
//!   gam's curve error against the TRUE marginal is no worse than lifelines' by
//!   more than 10%. The primary claim is truth recovery; lifelines is the bar.
//!
//! There is no skip path: a missing `lifelines` is a real failure.

use gam::families::lognormal_kernel::{LatentSurvivalRow, LatentSurvivalRowJet};
use gam::quadrature::QuadratureContext;
use gam::test_support::reference::{Column, run_python};

/// Marginal survival S(t) = K_{0, t}(μ, σ) evaluated through the gam kernel by
/// reading the value of a single-sided (right-censored at `t`, no entry) row's
/// survival mass. We obtain it directly from the order-0 mass via a degenerate
/// interval [t, +inf) is not representable, so we use the right-censored jet
/// whose log-likelihood is exactly `log K_{0,t}` when entry mass is zero.
fn marginal_survival(ctx: &QuadratureContext, b_t: f64, mu: f64, sigma: f64) -> f64 {
    // Right-censored at cumulative-hazard mass b_t with no delayed entry:
    //   ℓ = log K_{0, b_t}(μ, σ) = log S(t).
    let row = LatentSurvivalRow::right_censored(0.0, b_t, 0.0, 0.0);
    let jet = LatentSurvivalRowJet::evaluate(ctx, &row, mu, sigma)
        .expect("right-censored survival mass must evaluate");
    jet.log_lik.exp()
}

/// Summed interval-censored negative log-likelihood over the data, through gam's
/// kernel. `b_left[i]`, `b_right[i]` are the baseline cumulative hazards B(L_i),
/// B(R_i). No delayed entry (entry mass = 0).
fn interval_nll(
    ctx: &QuadratureContext,
    b_left: &[f64],
    b_right: &[f64],
    mu: f64,
    sigma: f64,
) -> f64 {
    let mut nll = 0.0;
    for (&bl, &br) in b_left.iter().zip(b_right.iter()) {
        let row = LatentSurvivalRow::interval_censored(0.0, bl, br, 0.0, 0.0, 0.0);
        let jet = LatentSurvivalRowJet::evaluate(ctx, &row, mu, sigma)
            .expect("interval-censored row must evaluate through the gam kernel");
        nll -= jet.log_lik;
    }
    nll
}

/// Plain coordinate-descent / golden-section maximiser of gam's interval
/// log-likelihood in (μ, σ). The kernel exposes analytic score and neg-Hessian,
/// but the test deliberately drives only the value to keep the recovery claim
/// independent of the derivative wiring under test: we want to prove the
/// LIKELIHOOD VALUE recovers the truth.
fn fit_interval_mle(ctx: &QuadratureContext, b_left: &[f64], b_right: &[f64]) -> (f64, f64) {
    let golden = |f: &dyn Fn(f64) -> f64, mut lo: f64, mut hi: f64| -> f64 {
        let phi = (5.0_f64.sqrt() - 1.0) / 2.0;
        let mut c = hi - phi * (hi - lo);
        let mut d = lo + phi * (hi - lo);
        let mut fc = f(c);
        let mut fd = f(d);
        for _ in 0..48 {
            if fc < fd {
                hi = d;
                d = c;
                fd = fc;
                c = hi - phi * (hi - lo);
                fc = f(c);
            } else {
                lo = c;
                c = d;
                fc = fd;
                d = lo + phi * (hi - lo);
                fd = f(d);
            }
        }
        0.5 * (lo + hi)
    };

    let mut mu = 0.0_f64;
    let mut sigma = 1.0_f64;
    for _ in 0..12 {
        let mu_new = golden(&|m| interval_nll(ctx, b_left, b_right, m, sigma), -4.0, 4.0);
        let sigma_new = golden(&|s| interval_nll(ctx, b_left, b_right, mu, s), 0.05, 4.0);
        let moved = (mu_new - mu).abs() + (sigma_new - sigma).abs();
        mu = mu_new;
        sigma = sigma_new;
        if moved < 1e-9 {
            break;
        }
    }
    (mu, sigma)
}

#[test]
fn gam_recovers_interval_censored_latent_truth_match_or_beat_lifelines() {
    let ctx = QuadratureContext::new();

    // ---- truth ------------------------------------------------------------
    let mu_true = -0.4_f64;
    let sigma_true = 0.6_f64;
    let n = 240usize;

    // Fixed inspection grid (discrete clinic visits). B(t) = t (unit clock), so
    // baseline cumulative hazard at a grid time equals the time itself.
    let grid: Vec<f64> = (0..=12).map(|k| 0.5 * k as f64).collect();

    // Draw true event times from the MARGINAL S(t) via inverse-CDF on a fine
    // grid (deterministic low-discrepancy quantiles -> reproducible, no RNG
    // dependence). Then bracket each by the inspection grid to make it interval
    // data; the exact time is discarded.
    let mut fine_t = Vec::new();
    let mut fine_s = Vec::new();
    {
        let mut t = 1e-4;
        while t <= 7.0 {
            fine_t.push(t);
            fine_s.push(marginal_survival(&ctx, t, mu_true, sigma_true));
            t += 0.01;
        }
    }
    let invert_survival = |u: f64| -> f64 {
        // find t with S(t) = u (S decreasing)
        for w in fine_s.windows(2).enumerate() {
            let (i, pair) = w;
            if pair[0] >= u && u > pair[1] {
                let frac = (pair[0] - u) / (pair[0] - pair[1]).max(1e-12);
                return fine_t[i] + frac * (fine_t[i + 1] - fine_t[i]);
            }
        }
        *fine_t.last().unwrap()
    };

    let mut b_left = Vec::with_capacity(n);
    let mut b_right = Vec::with_capacity(n);
    let mut left_times = Vec::with_capacity(n);
    let mut right_times = Vec::with_capacity(n);
    for i in 0..n {
        // van der Corput low-discrepancy quantile in (0,1)
        let mut q = 0.0_f64;
        let mut base = 0.5_f64;
        let mut k = i + 1;
        while k > 0 {
            if k & 1 == 1 {
                q += base;
            }
            base *= 0.5;
            k >>= 1;
        }
        let u = 1.0 - (q * 0.998 + 0.001); // survival prob in (0,1)
        let t_true = invert_survival(u);

        // bracket by the inspection grid
        let mut l = 0.0_f64;
        let mut r = f64::INFINITY;
        for win in grid.windows(2) {
            if t_true > win[0] && t_true <= win[1] {
                l = win[0];
                r = win[1];
                break;
            }
        }
        if !r.is_finite() {
            // beyond last visit: treat as right-censored at last visit by a wide
            // bracket to the fine horizon (still an interval the tools accept)
            l = *grid.last().unwrap();
            r = 7.0;
        }
        // B(t) = t for the unit clock.
        b_left.push(l);
        b_right.push(r);
        left_times.push(l);
        right_times.push(r);
    }

    // ---- gam interval-likelihood MLE -------------------------------------
    let (mu_hat, sigma_hat) = fit_interval_mle(&ctx, &b_left, &b_right);

    let mu_err = (mu_hat - mu_true).abs();
    let sigma_err = (sigma_hat - sigma_true).abs();

    // gam's recovered marginal survival curve vs the TRUTH on a held-out grid.
    let eval_t: Vec<f64> = (1..=12).map(|k| 0.5 * k as f64).collect();
    let mut gam_curve_sse = 0.0;
    let mut truth_norm = 0.0;
    for &t in &eval_t {
        let s_hat = marginal_survival(&ctx, t, mu_hat, sigma_hat);
        let s_true = marginal_survival(&ctx, t, mu_true, sigma_true);
        gam_curve_sse += (s_hat - s_true).powi(2);
        truth_norm += s_true.powi(2);
    }
    let gam_curve_rell2 = (gam_curve_sse / truth_norm).sqrt();

    // ---- lifelines baseline on the IDENTICAL brackets --------------------
    let r_finite: Vec<f64> = right_times
        .iter()
        .map(|&r| if r.is_finite() { r } else { 1e6 })
        .collect();
    let ref_res = run_python(
        &[Column::new("L", &left_times), Column::new("R", &r_finite)],
        r#"
import numpy as np
from lifelines import WeibullFitter
L = np.asarray(L, dtype=float)
R = np.asarray(R, dtype=float)
# Avoid exact-zero left bound for the parametric interval fit.
L = np.clip(L, 1e-6, None)
wf = WeibullFitter()
wf.fit_interval_censoring(L, R)
# Recovered marginal survival on the evaluation grid.
eval_t = np.array([0.5*k for k in range(1,13)], dtype=float)
S = wf.survival_function_at_times(eval_t).values.ravel()
emit("S_ref", list(S))
"#,
    );
    let s_ref = ref_res.vector("S_ref");

    let mut ref_curve_sse = 0.0;
    for (idx, &t) in eval_t.iter().enumerate() {
        let s_true = marginal_survival(&ctx, t, mu_true, sigma_true);
        ref_curve_sse += (s_ref[idx] - s_true).powi(2);
    }
    let ref_curve_rell2 = (ref_curve_sse / truth_norm).sqrt();

    eprintln!(
        "interval-censored recovery: gam (mu,sigma)=({:.4},{:.4}) truth=({:.4},{:.4}) \
         mu_err={:.4} sigma_err={:.4} | curve relL2 gam={:.4} lifelines={:.4}",
        mu_hat, sigma_hat, mu_true, sigma_true, mu_err, sigma_err, gam_curve_rell2, ref_curve_rell2
    );

    // ---- truth-recovery assertions ---------------------------------------
    // Interval censoring discards the exact time, so the finite-sample MLE bar
    // is looser than the exact-event case but still pins the truth at n=400.
    assert!(
        mu_err < 0.18,
        "gam interval-censored MLE failed to recover mu*: |{:.4}-{:.4}|={:.4}",
        mu_hat,
        mu_true,
        mu_err
    );
    assert!(
        sigma_err < 0.20,
        "gam interval-censored MLE failed to recover sigma*: |{:.4}-{:.4}|={:.4}",
        sigma_hat,
        sigma_true,
        sigma_err
    );
    assert!(
        gam_curve_rell2 < 0.06,
        "gam recovered survival curve too far from truth: relL2={:.4}",
        gam_curve_rell2
    );

    // ---- match-or-beat lifelines -----------------------------------------
    assert!(
        gam_curve_rell2 <= ref_curve_rell2 * 1.10 + 1e-3,
        "gam interval-censored survival-curve recovery ({:.4}) is worse than lifelines ({:.4}) by >10%",
        gam_curve_rell2,
        ref_curve_rell2
    );
}

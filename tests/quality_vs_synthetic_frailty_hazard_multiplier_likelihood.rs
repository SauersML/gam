//! Intrinsic validation of gam's lognormal hazard-multiplier frailty marginal
//! likelihood against an independent hand-coded R reference.
//!
//! gam's latent-survival family integrates a lognormal frailty `exp(U)`,
//! `U ~ N(0, sigma^2)`, out of a proportional-hazards row likelihood in closed
//! form via the special function `K_{k,m}(mu, sigma) = E[exp(k*U - m*exp(U))]`.
//! A right-censored row contributes `log K_{0,M}` (with `M = H_0(t)` the
//! cumulative loaded hazard) and an exact event contributes
//! `log(h_0(t) * K_{1,M})`. There is **no** external tool that implements gam's
//! exact lognormal kernel on identical data in identical form, so the comparator
//! is a self-consistent R reference that re-evaluates the SAME marginal row
//! likelihoods — `marginal = integral over U~N(0,sigma^2) of the conditional row
//! likelihood` — by independent numerical quadrature (`statmod::gauss.quad.prob`,
//! Gauss-Hermite for the normal frailty). Both engines see byte-identical
//! `(M, h_0, event)` triples; the test asserts the two marginal log-likelihoods
//! agree at the true parameters `(beta=0 => mu=0, sigma=sigma_true)`.
//!
//! Bound justification: both sides evaluate the exact frailty-integrated
//! likelihood. gam uses its `K_{k,m}` recurrence/Laplace kernel; R uses a
//! 128-node Gauss-Hermite rule whose per-row quadrature error on these smooth,
//! bounded integrands is far below 1e-9, so over 48 rows the accumulated
//! disagreement must stay under 1e-4. This validates gam's kernel evaluation and
//! its assembly into the marginal log-lik without any external fit or optimizer.

use gam::families::lognormal_kernel::{LatentSurvivalRow, LatentSurvivalRowJet};
use gam::quadrature::QuadratureContext;
use gam::test_support::reference::{Column, max_abs_diff, run_r};

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

#[test]
fn lognormal_hazard_multiplier_marginal_loglik_matches_r_quadrature() {
    // ---- synthetic clustered right-censored survival data -----------------
    // m groups of n_per rows; baseline H_0(t) = baseline_slope * t (linear,
    // Makeham-like), with frailty U_g ~ N(0, sigma_true^2) shared within group.
    // Event time from T = -ln(S)/(baseline_slope * exp(U)) with S = exp(-E),
    // E ~ Exp(1), i.e. integrating H(t|U) = H_0(t)*exp(U) to a draw. Times are
    // administratively censored at `tau` to produce a censored/event mix.
    const M_GROUPS: usize = 6;
    const N_PER: usize = 8;
    const N_TOTAL: usize = M_GROUPS * N_PER;
    const SIGMA_TRUE: f64 = 0.5;
    const BASELINE_SLOPE: f64 = 0.1; // H_0(t) = 0.1 * t  =>  h_0(t) = 0.1
    const TAU: f64 = 8.0; // administrative censoring horizon

    let mut rng = SplitMix64::new(42);
    let mut times = Vec::with_capacity(N_TOTAL);
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
            times.push(t_obs);
            events.push(event);
            cum_hazard.push(BASELINE_SLOPE * t_obs);
        }
    }
    let n_events: usize = events.iter().filter(|&&e| e > 0.5).count();
    assert!(
        n_events > 0 && n_events < N_TOTAL,
        "synthetic data must mix events and censoring: events={n_events}/{N_TOTAL}"
    );

    // ---- gam side: marginal log-lik at (mu=0, sigma=sigma_true) via kernels --
    // Each row's frailty-integrated likelihood is assembled by gam's own
    // LatentSurvivalRowJet from the K_{k,m} kernel. Full loading, no left
    // truncation, no unloaded background hazard.
    let quadctx = QuadratureContext::new();
    let mut gam_loglik = 0.0;
    for i in 0..N_TOTAL {
        let mass_exit = cum_hazard[i];
        let row = if events[i] > 0.5 {
            // Exact event: hazard_loaded = h_0(t) = BASELINE_SLOPE.
            LatentSurvivalRow::exact_event(0.0, mass_exit, 0.0, 0.0, BASELINE_SLOPE, 0.0)
        } else {
            LatentSurvivalRow::right_censored(0.0, mass_exit, 0.0, 0.0)
        };
        let jet = LatentSurvivalRowJet::evaluate(&quadctx, &row, 0.0, SIGMA_TRUE)
            .expect("gam kernel row likelihood");
        gam_loglik += jet.log_lik;
    }

    // ---- R reference: same marginal likelihood by Gauss-Hermite quadrature ---
    // For each row, integrate the conditional row likelihood over U~N(0,sigma^2):
    //   censored: L = E[exp(-M*exp(U))]
    //   event   : L = h_0 * E[exp(U) * exp(-M*exp(U))]
    // statmod::gauss.quad.prob(dist="normal") supplies the (node,weight) rule for
    // E[g(U)] under N(mu,sigma); 128 nodes drives the quadrature error to ~1e-13.
    let cols = [Column::new("M", &cum_hazard), Column::new("event", &events)];
    let r = run_r(
        &cols,
        r#"
        suppressPackageStartupMessages(library(statmod))
        sigma <- 0.5
        h0 <- 0.1
        gq <- gauss.quad.prob(128, dist = "normal", mu = 0, sigma = sigma)
        nodes <- gq$nodes
        wts <- gq$weights
        loglik <- 0
        for (i in seq_len(nrow(df))) {
            M <- df$M[i]
            ev <- df$event[i]
            if (ev > 0.5) {
                integrand <- exp(nodes) * exp(-M * exp(nodes))
                Lrow <- h0 * sum(wts * integrand)
            } else {
                integrand <- exp(-M * exp(nodes))
                Lrow <- sum(wts * integrand)
            }
            loglik <- loglik + log(Lrow)
        }
        emit("loglik", loglik)
        "#,
    );
    let r_loglik = r.scalar("loglik");

    let diff = max_abs_diff(&[gam_loglik], &[r_loglik]);
    eprintln!(
        "lognormal hazard-multiplier frailty: n={N_TOTAL} groups={M_GROUPS} \
         events={n_events} sigma={SIGMA_TRUE} \
         gam_loglik={gam_loglik:.10} r_loglik={r_loglik:.10} abs_diff={diff:.3e}"
    );

    // Both evaluate the EXACT frailty-integrated likelihood at the true
    // parameters on identical (M, h_0, event) data; only the quadrature route
    // differs. 1e-4 over 48 rows is the principled accumulated-quadrature bound.
    assert!(
        diff <= 1e-4,
        "gam lognormal-kernel marginal log-lik disagrees with R Gauss-Hermite \
         reference: gam={gam_loglik:.10} r={r_loglik:.10} abs_diff={diff:.3e}"
    );
}

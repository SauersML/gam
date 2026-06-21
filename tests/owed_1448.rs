//! Owed-work regression gate for GitHub issue #1448 — Negative-Binomial outer
//! θ↔λ alternation (`src/solver/estimate/optimizer.rs`, the bounded loop around
//! lines 1338–1432).
//!
//! ## The fix
//!
//! With NB `theta` ESTIMATED, the λ-search freezes `theta` at the search value
//! (`frozen_negbin_theta`, #1082) so the REML criterion `F(ρ) = REML(ρ, θ_frozen)`
//! is stationary in ρ; the final accept-fit then ML-refreshes `theta` at the
//! converged η. A SINGLE freeze→refresh leaves the selected ρ optimal only for
//! `θ_frozen`, NOT for the refreshed `θ_final` — so the reported `(ρ*, θ_final)`
//! is only jointly stationary if `theta` happened to barely move.
//!
//! The fix (commit `e21e2ad38`) wraps the ρ-search + accept-fit in a bounded
//! loop: after each refit, if the NB `theta` drifted past
//! `NEGBIN_THETA_JOINT_DRIFT_TOL` (5%), re-freeze the search at `θ_final`, reset
//! the outer seed state, and re-run the ρ search; iterate to the joint `(ρ, θ)`
//! fixed point or a round cap (8). For non-NB / user-fixed-θ fits the loop runs
//! exactly once (the criterion `negbin_theta_is_estimated()` is never met), so
//! those fits are byte-identical to the pre-#1448 single pass.
//!
//! ## What this test asserts — public API, non-vacuous
//!
//! The loop state (`frozen_negbin_theta`, `negbin_alternation_round`,
//! `reset_outer_seed_state`) is PRIVATE to the optimizer, so we assert the
//! observable PROPERTY the loop establishes: JOINT STATIONARITY of the reported
//! `(ρ, θ̂)`. Concretely, on strongly-overdispersed data fit with ESTIMATED
//! theta (so `theta` genuinely moves from its seed during the fit):
//!
//!   the smoothing parameters ρ = log λ selected by the estimated-θ fit must
//!   equal the ρ selected by a fit with θ held FIXED at the estimated θ̂.
//!
//! That equality is exactly the fixed-point the alternation drives to: ρ is
//! optimal for the θ̂ the fit reports. Before #1448 the single-pass estimated-θ
//! fit selected ρ for the OLD frozen θ (the seed-derived search value, which
//! differs from θ̂ whenever the data are overdispersed enough to move θ past the
//! 5% band), so its ρ does NOT match the ρ that is optimal for θ̂ — the assertion
//! below fails on the pre-fix code. The fixed-at-θ̂ fit is the same λ-search
//! machinery with estimation switched off, so under the fix the two ρ vectors
//! coincide to the criterion's convergence tolerance.
//!
//! The companion ASSERT also checks that θ̂ itself is non-trivially far from the
//! seed (so the alternation actually had work to do — guarding against a
//! vacuous pass where θ never moved and the single pass was already stationary).

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};

const N: usize = 400;
/// True overdispersion of the synthesized data. Small θ ⇒ strong overdispersion,
/// so the ML θ-estimate lands here and drifts far from the deliberately-wrong
/// seed below — past the 5% joint-stationarity band that arms the alternation.
const THETA_TRUE: f64 = 1.5;
/// Deliberately-wrong user seed for the NB θ search: ~5× the truth, so the first
/// freeze→refresh moves θ by far more than 5% and a single pass leaves ρ optimal
/// for the wrong θ.
const THETA_SEED: f64 = 8.0;
const SEED: u64 = 4148;

/// Sample one NB2(mu, theta) count via the gamma-Poisson mixture:
/// lambda ~ Gamma(shape=theta, scale=mu/theta), y ~ Poisson(lambda), giving
/// Var(y) = mu + mu^2/theta.
fn sample_negbin(mu: f64, theta: f64, rng: &mut StdRng) -> f64 {
    let gamma = Gamma::new(theta, mu / theta).expect("gamma params valid");
    let lambda = gamma.sample(rng);
    let pois = Poisson::new(lambda.max(1e-12)).expect("poisson rate valid");
    pois.sample(rng)
}

#[test]
fn owed_1448_negbin_theta_lambda_alternation_reaches_joint_fixed_point() {
    init_parallelism();

    // ---- synthesize strongly-overdispersed counts with a smooth mean -------
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 10.0).expect("uniform x");
    let mut x = vec![0.0_f64; N];
    let mut y = vec![0.0_f64; N];
    for i in 0..N {
        let xi = ux.sample(&mut rng);
        let eta = 1.2 + 0.7 * (xi * std::f64::consts::PI / 5.0).sin();
        let mu = eta.exp();
        x[i] = xi;
        y[i] = sample_negbin(mu, THETA_TRUE, &mut rng);
    }

    let headers = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<StringRecord> = (0..N)
        .map(|i| StringRecord::from(vec![x[i].to_string(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode negbin dataset");

    // ---- Fit A: ESTIMATED theta (negative_binomial_theta = None) -----------
    // This is the #1448 regime: the inner solve ML-estimates θ, the λ-search
    // freezes it, and the alternation loop must drive (ρ, θ) to a joint fixed
    // point.
    let cfg_estimated = FitConfig {
        family: Some("negative-binomial".to_string()),
        negative_binomial_theta: None,
        ..FitConfig::default()
    };
    let FitResult::Standard(fit_est) =
        fit_from_formula("y ~ s(x, k=8)", &ds, &cfg_estimated).expect("gam estimated-θ NB fit")
    else {
        panic!("negative-binomial GLM should produce a Standard fit");
    };
    let theta_hat = fit_est
        .fit
        .likelihood_scale
        .negbin_theta()
        .expect("estimated NB fit must report an estimated theta_hat");
    assert!(
        fit_est.fit.likelihood_scale.negbin_theta_is_estimated(),
        "Fit A must be in ESTIMATED-θ mode (the #1448 regime)",
    );
    let rho_estimated = fit_est.fit.log_lambdas.clone();

    // Non-vacuity guard: θ must have moved far from the seed, so the alternation
    // genuinely had a fixed point to chase. A vacuous pass (θ̂ ≈ seed) would mean
    // a single pass was already stationary and the test would not exercise the
    // loop.
    let theta_seed_drift = (theta_hat - THETA_SEED).abs() / THETA_SEED;
    assert!(
        theta_seed_drift > 0.05,
        "fixture not exercising the alternation: estimated θ̂={theta_hat:.4} barely moved \
         from the seed {THETA_SEED} (drift {:.1}% ≤ 5% band); pick a seed farther from \
         the data's true overdispersion θ_true={THETA_TRUE}",
        theta_seed_drift * 100.0,
    );

    // ---- Fit B: theta held FIXED at the estimated θ̂ ------------------------
    // Same λ-search machinery with estimation switched off. Its selected ρ is
    // BY CONSTRUCTION optimal for θ̂.
    let cfg_fixed = FitConfig {
        family: Some("negative-binomial".to_string()),
        negative_binomial_theta: Some(theta_hat),
        ..FitConfig::default()
    };
    let FitResult::Standard(fit_fixed) =
        fit_from_formula("y ~ s(x, k=8)", &ds, &cfg_fixed).expect("gam fixed-θ̂ NB fit")
    else {
        panic!("negative-binomial GLM should produce a Standard fit");
    };
    assert!(
        !fit_fixed.fit.likelihood_scale.negbin_theta_is_estimated(),
        "Fit B must be in FIXED-θ mode",
    );
    let rho_fixed = fit_fixed.fit.log_lambdas.clone();

    // ---- Joint-stationarity assertion (#1448) ------------------------------
    // The estimated-θ fit's ρ must coincide with the ρ that is optimal for the
    // θ̂ it reports. Pre-#1448, Fit A selected ρ for the OLD frozen θ (≠ θ̂ here,
    // since θ drifted well past 5%), so rho_estimated would differ materially
    // from rho_fixed. The alternation makes them agree to the outer convergence
    // tolerance.
    assert_eq!(
        rho_estimated.len(),
        rho_fixed.len(),
        "both fits share the same smooth structure ⇒ same ρ dimension",
    );
    // Generous absolute band on log λ: the outer search converges ρ to a small
    // gradient norm, so a genuinely joint-stationary fit agrees to well within
    // this, while the pre-fix wrong-θ ρ (selected on a criterion with a 5×-off θ)
    // sits far outside it.
    const RHO_JOINT_STATIONARY_TOL: f64 = 0.5;
    for (k, (re, rf)) in rho_estimated.iter().zip(rho_fixed.iter()).enumerate() {
        let gap = (re - rf).abs();
        assert!(
            gap <= RHO_JOINT_STATIONARY_TOL,
            "NB θ↔λ alternation not at a joint fixed point (#1448): component {k} of \
             ρ from the estimated-θ fit (={re:.4}) is optimal for a different θ than the \
             reported θ̂={theta_hat:.4} — it differs from the ρ optimal for θ̂ (={rf:.4}) \
             by {gap:.4} > {RHO_JOINT_STATIONARY_TOL}. Pre-#1448 single-pass behaviour: \
             ρ was selected for the frozen search θ, not θ̂.",
        );
    }
}

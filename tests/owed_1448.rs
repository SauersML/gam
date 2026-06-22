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
//! `(ρ, θ̂)`. Concretely, on overdispersed counts with a strong, wiggly mean and
//! a rich spline basis fit with ESTIMATED theta:
//!
//!   the smoothing parameters ρ = log λ selected by the estimated-θ fit must
//!   equal the ρ selected by a fit with θ held FIXED at the estimated θ̂.
//!
//! That equality is exactly the fixed-point the alternation drives to: ρ is
//! optimal for the θ̂ the fit reports. The λ-search freezes θ at the value
//! captured from the FIRST converged inner solve, whose η has not yet resolved
//! all the mean wiggle, so `θ_frozen` lands materially below the `θ_final` the
//! accept-fit ML-refreshes at the fully-converged η. Before #1448 the single
//! pass therefore reported a ρ optimal for `θ_frozen`, NOT `θ̂ = θ_final` — and on
//! this fixture those ρ vectors differ by ~0.8 in log λ (verified empirically by
//! disabling the alternation loop). The fixed-at-θ̂ fit is the same λ-search
//! machinery with estimation switched off, so under the fix the two ρ vectors
//! coincide to the outer convergence tolerance (< 1e-3) and the assertion holds;
//! on the pre-fix single-pass code the ~0.8 gap blows the 0.5 band and it fails.
//!
//! The companion ASSERT also checks that θ̂ is a genuine finite overdispersed
//! value (not railed to the Poisson-limit clamp), so the NB θ path is actually
//! engaged and the alternation had a real fixed point to reach.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};

const N: usize = 700;
/// True overdispersion of the synthesized data. With a strong, wiggly mean and a
/// rich basis (k below), the θ the λ-search FREEZES at — captured from the first
/// converged inner solve, whose η does not yet resolve all the wiggle — lands
/// materially below the θ the final accept-fit ML-refreshes at the converged η.
/// That intra-loop θ drift (θ_frozen → θ_final), not the distance to any user
/// seed, is what makes the λ̂ selected for θ_frozen sub-optimal for θ_final and so
/// arms the #1448 alternation. Empirically this fixture moves the selected ρ by
/// ~0.8 in log λ between the two θ values (see the assertion below).
const THETA_TRUE: f64 = 6.0;
/// Mean-signal amplitude / period / baseline (on the log scale). A large
/// amplitude over a full period across the support gives the rich curvature whose
/// partial resolution at the frozen-θ η drives the θ_frozen → θ_final gap.
const ETA_AMP: f64 = 2.5;
const ETA_PERIOD: f64 = 1.0;
const ETA_BASE: f64 = 3.0;
/// Spline basis size: enough to resolve the wiggle (so the converged-η θ differs
/// from the coarser frozen-η θ) without overfitting.
const SPLINE_K: usize = 20;
const SEED: u64 = 55;

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
        let eta = ETA_BASE + ETA_AMP * (xi * std::f64::consts::PI / ETA_PERIOD).sin();
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
    let formula = format!("y ~ s(x, k={SPLINE_K})");
    let FitResult::Standard(fit_est) =
        fit_from_formula(&formula, &ds, &cfg_estimated).expect("gam estimated-θ NB fit")
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

    // Non-vacuity guard: the recovered θ̂ must be a genuinely finite, overdispersed
    // value (NOT railed to the Poisson-limit clamp), so the NB θ path is actually
    // engaged and the λ-search had a θ to freeze. A Poisson-limit θ̂ would mean the
    // data carried no resolvable overdispersion and the alternation had nothing to
    // do.
    assert!(
        theta_hat.is_finite() && (0.5..50.0).contains(&theta_hat),
        "fixture not exercising the NB θ path: θ̂={theta_hat:.4} is railed or degenerate \
         (expected a finite, overdispersed θ near θ_true={THETA_TRUE})",
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
        fit_from_formula(&formula, &ds, &cfg_fixed).expect("gam fixed-θ̂ NB fit")
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
    // θ̂ it reports. Pre-#1448, Fit A froze θ at the value captured from the FIRST
    // converged λ-search inner solve — whose η does not yet resolve all the mean
    // wiggle — and selected ρ for THAT θ_frozen. The final accept-fit then ML-
    // refreshed θ at the fully-converged η to θ̂ = θ_final, which on this fixture
    // sits materially above θ_frozen; the reported ρ was therefore optimal for
    // θ_frozen, NOT θ̂. With the single-pass (pre-fix) code this fixture's
    // rho_estimated and rho_fixed differ by ~0.8 in log λ (FAR outside the band
    // below — verified by disabling the alternation loop). The #1448 alternation
    // re-freezes at θ_final and re-runs the ρ search until (ρ, θ) reach a joint
    // fixed point, after which rho_estimated agrees with rho_fixed to the outer
    // convergence tolerance (empirically the two coincide to < 1e-3).
    assert_eq!(
        rho_estimated.len(),
        rho_fixed.len(),
        "both fits share the same smooth structure ⇒ same ρ dimension",
    );
    // Band on log λ comfortably separating the two regimes: the post-fix fit
    // agrees to < 1e-3, while the pre-fix single-pass ρ (selected for θ_frozen)
    // sits ~0.8 away — so 0.5 fails before the fix and passes after, without
    // being so tight that ordinary outer-convergence jitter trips it.
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

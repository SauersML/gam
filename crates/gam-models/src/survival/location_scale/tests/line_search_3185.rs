//! #3185 — the direct parametric-AFT MLE's backtracking floor and stall.
//!
//! The line search used to halve down to a fixed `α ≥ 1e-12`, and a search that
//! found no Armijo step was accepted as the MLE whenever `½λ² ≤ 1e-4`, a thousand
//! times the objective tolerance. The trials now stop where the predicted gain
//! `α·g·δ` enters the objective's rounding band, and a failed search is an error.

use super::*;
use crate::survival::location_scale::family_solver::aft_resolvable_trial_count;

#[test]
fn resolvable_trial_count_stops_at_the_rounding_band_3185() {
    // Gains 1, 2⁻¹, …, 2⁻⁹ exceed 2⁻¹⁰; the tenth halving lands on the band.
    assert_eq!(aft_resolvable_trial_count(1.0, 1.0, 0.0, 0.5, 2f64.powi(-10)), 10);
    // The same gain split as α₀ = 1/4 and g·δ = 4.
    assert_eq!(aft_resolvable_trial_count(0.25, 4.0, 0.0, 0.5, 2f64.powi(-10)), 10);
    // A slower contraction resolves more trials: 0.9ᵏ > 0.5 for k ≤ 6.
    assert_eq!(aft_resolvable_trial_count(1.0, 1.0, 0.0, 0.9, 0.5), 7);
    // A saddle escape draws its gain from negative curvature alone: with
    // g·δ = 0 and κ = 2 the gains are α², i.e. 1, 2⁻², …, 2⁻⁸ above 2⁻¹⁰.
    assert_eq!(aft_resolvable_trial_count(1.0, 0.0, 2.0, 0.5, 2f64.powi(-10)), 5);
}

#[test]
fn resolvable_trial_count_is_zero_without_a_resolvable_gain_3185() {
    // A first gain at or under the band cannot be told from rounding.
    assert_eq!(aft_resolvable_trial_count(1.0, 1e-9, 0.0, 0.5, 1e-9), 0);
    assert_eq!(aft_resolvable_trial_count(1e-3, 1e-7, 0.0, 0.5, 1e-9), 0);
    // A non-ascent or non-finite prediction offers nothing to try.
    assert_eq!(aft_resolvable_trial_count(1.0, -1.0, 0.0, 0.5, 0.0), 0);
    assert_eq!(aft_resolvable_trial_count(1.0, f64::NAN, 0.0, 0.5, 0.0), 0);
    assert_eq!(aft_resolvable_trial_count(f64::INFINITY, 1.0, 0.0, 0.5, 0.0), 0);
}

/// Without the stall acceptance every fit must converge on the decrement test
/// itself. Across sample sizes the reduced lognormal AFT still reaches its
/// closed-form MLE: the exact log-likelihood gap to the closed form,
/// `n[log(σ/σ̂) + (σ̂² + (μ − μ̂)²)/(2σ²) − ½]`, is within the default inner
/// tolerance on `½λ²`, which is what a certified fit promises at any `n`.
#[test]
fn reduced_parametric_aft_converges_without_stall_acceptance_3185() {
    let inner_tol = BlockwiseFitOptions::default().inner_tol;
    for (n, seed) in [(50usize, 31u64), (800, 3185), (20_000, 70)] {
        let (age_exit, event, log_t) = reduced_aft_lognormal_sample(n, -0.3, 0.8, seed);
        let (mu_hat, sigma_hat) = lognormal_closed_form_mle(&log_t);
        let spec = reduced_aft_lognormal_spec(&age_exit, &event, 1.0);
        let prepared = prepare_survival_location_scale_model(&spec).expect("prepare");
        assert!(prepared.is_reduced_parametric_aft());
        let (fit, _) = fit_survival_location_scale_with_geometry(spec)
            .unwrap_or_else(|e| panic!("n={n}: reduced parametric-AFT MLE: {e}"));
        let mu = fit.beta_threshold()[0];
        let log_sigma = fit.beta_log_sigma()[0];
        let sigma = log_sigma.exp();
        let d = mu - mu_hat;
        let gap = n as f64
            * ((log_sigma - sigma_hat.ln())
                + ((sigma_hat - sigma) * (sigma_hat + sigma) + d * d) / (2.0 * sigma * sigma));
        assert!(
            gap >= -inner_tol && gap <= inner_tol,
            "n={n}: log-likelihood gap {gap:.3e} to the closed-form MLE exceeds {inner_tol:.1e} \
             (mu {mu:.9} vs {mu_hat:.9}, sigma {sigma:.9} vs {sigma_hat:.9})"
        );
    }
}

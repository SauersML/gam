//! The FFI-facing inverse link helper `apply_inverse_link_vec` is the
//! single dispatch point that backs every Python posterior path that
//! moves draws from the link scale (η) to the response scale (μ):
//!
//!   * `gamfit/_sampling.py::PosteriorSamples.predict_draws` calls the
//!     PyO3 entry `apply_inverse_link_array`, which is a thin wrapper
//!     over `apply_inverse_link_vec` (`crates/gam-pyffi/src/lib.rs:13873`).
//!   * `inference::eta_bands::eta_bands_from_matrix` and
//!     `inference::posterior_bands::eta_bands_from_matrix` both invoke
//!     it directly to turn link-scale credible bounds into response-scale
//!     bounds (see `src/inference/posterior_bands.rs:5,70,82,83`).
//!
//! So any precision loss in `apply_inverse_link_vec` propagates straight
//! into user-visible posterior probabilities and credible bands.
//!
//! The cloglog branch is implemented as
//!
//! ```ignore
//! "cloglog" => {
//!     for &e in eta {
//!         let clamped = e.clamp(-50.0, 50.0);
//!         out.push(1.0 - (-clamped.exp()).exp());
//!     }
//! }
//! ```
//!
//! (`src/families/inverse_link.rs:35-40`).  The naive form
//! `1 − exp(−exp(η))` suffers catastrophic cancellation in the deep
//! negative tail: for η ≲ -36 the inner `exp(-exp(η))` rounds to the
//! IEEE-754 value `1.0`, so `1.0 - 1.0 = 0.0` is returned — even though
//! the mathematically correct value is the strictly positive
//! `μ(η) = 1 - exp(-exp(η)) ≈ exp(η) > 0`.
//!
//! The crate already ships a numerically stable cloglog evaluator that
//! uses `expm1` to dodge exactly this cancellation:
//! `quadrature.rs::cloglog_negative_tail_mean` returns
//! `-(-eta.exp()).exp_m1()`, which preserves digits all the way down to
//! `η ≈ -745` (the exp underflow boundary).  The FFI helper does not
//! share this implementation.
//!
//! The test below pins three minimum-bar invariants for the cloglog
//! inverse link on the public FFI primitive:
//!
//!   1. **Positivity.**  `μ(η) = 1 - exp(-exp(η))` is *strictly* positive
//!      for every real η; the response is a probability in the open
//!      interval `(0, 1)`.  A returned value of exactly `0.0` for a
//!      finite η is mathematically wrong and silently produces
//!      `-inf` downstream when callers take `ln(μ)` (e.g. for posterior
//!      log-likelihood or Brier / log-loss summaries on the draws).
//!
//!   2. **Leading-order asymptotic.**  For η → −∞ the inverse link
//!      satisfies `μ(η) = exp(η) − ½ exp(2η) + O(exp(3η))`, so for η in
//!      the deep negative tail `μ(η) / exp(η) → 1`.  The stable
//!      reference value used below (`(-η.exp()).neg().exp_m1().neg()`)
//!      makes this exact and matches the crate's own
//!      `cloglog_negative_tail_mean`.
//!
//!   3. **Monotonicity across the cancellation boundary.**  Because
//!      `μ(η)` is strictly increasing, the response on η = −45 must be
//!      strictly smaller than the response on η = −20.  When the
//!      `1.0 - 1.0` cancellation collapses a swath of deep-tail
//!      η-values to the same `0.0`, the resulting "response draws" are
//!      no longer monotone in η — multiple distinct posterior draws
//!      ranked far apart on the link scale all map to the same
//!      `0` on the response scale.  We therefore assert strict
//!      monotonicity over a descending sequence that straddles the
//!      cancellation boundary.

use gam::families::inverse_link::apply_inverse_link_vec;

/// Stable cloglog inverse link via `expm1`.  This matches the reference
/// implementation that already lives in
/// `src/inference/quadrature.rs::cloglog_negative_tail_mean` and is exact
/// to all f64 digits across the entire representable η range.
fn cloglog_inv_link_stable(eta: f64) -> f64 {
    let ex = eta.exp();
    -((-ex).exp_m1())
}

#[test]
fn cloglog_apply_inverse_link_vec_is_strictly_positive_in_deep_negative_tail() {
    // η = -40 sits well inside the [-50, 50] clamp window the FFI helper
    // applies, so the buggy behaviour is purely the `1 - exp(-exp(η))`
    // cancellation — not a side-effect of the clamp.
    let eta = vec![-40.0_f64];
    let mu = apply_inverse_link_vec(&eta, "cloglog").expect("cloglog dispatch succeeds");
    assert_eq!(mu.len(), 1);

    let mu_eta = mu[0];

    // The cloglog inverse link maps R → (0, 1).  A finite η returning
    // exactly 0.0 is a precision failure: ln(0) = -inf cascades through
    // any posterior-predictive log-likelihood summary computed off the
    // returned draws.
    assert!(
        mu_eta > 0.0,
        "apply_inverse_link_vec('cloglog')[-40] returned mu = {mu_eta:.3e}; \
         cloglog inverse link must be strictly positive on every finite eta"
    );

    // Leading-order asymptotic: in the deep negative tail
    //   mu(eta) = 1 - exp(-exp(eta)) = exp(eta) - exp(2 eta)/2 + O(exp(3 eta)).
    // For eta = -40, exp(2 eta) / exp(eta) = exp(eta) ~ 4.25e-18, so the
    // ratio mu/exp(eta) is 1 - 2.12e-18 + ... ≈ 1 to better than 1e-15.
    let reference = cloglog_inv_link_stable(-40.0);
    let rel_err = (mu_eta - reference).abs() / reference;
    assert!(
        rel_err < 1e-12,
        "apply_inverse_link_vec('cloglog')[-40] = {mu_eta:.6e}, stable reference \
         (via -expm1(-exp(eta))) = {reference:.6e}, relative error = {rel_err:.3e}; \
         the FFI cloglog helper must agree with the crate's own stable evaluator"
    );
}

#[test]
fn cloglog_apply_inverse_link_vec_preserves_monotonicity_across_cancellation_boundary() {
    // Strictly decreasing η-grid straddling the f64 cancellation
    // boundary around η ≈ -36 (where 1.0 - exp(-exp(η)) collapses to
    // 1.0 - 1.0 = 0.0).  μ(η) is strictly increasing in η, so the
    // returned response sequence must be strictly decreasing for this
    // strictly decreasing input.  When the deep-tail values all collapse
    // to 0.0, neighbouring entries are equal — a clear monotonicity
    // violation that would silently corrupt order-sensitive posterior
    // summaries (quantiles, rank diagnostics, calibration plots).
    let eta = vec![-20.0_f64, -30.0, -36.0, -38.0, -40.0, -45.0];
    let mu =
        apply_inverse_link_vec(&eta, "cloglog").expect("cloglog dispatch succeeds on deep tail");
    assert_eq!(mu.len(), eta.len());

    for win in mu.windows(2) {
        let (prev, next) = (win[0], win[1]);
        assert!(
            prev > next,
            "apply_inverse_link_vec('cloglog') is not strictly monotone across the \
             η = [-20, -30, -36, -38, -40, -45] grid: full response sequence {mu:?}. \
             μ(η) = 1 - exp(-exp(η)) is strictly increasing, so the response on a \
             strictly decreasing η-grid must be strictly decreasing"
        );
    }
}

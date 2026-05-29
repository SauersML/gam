//! Sibling of the (fixed) cloglog deep-tail precision bug, issue #344, for
//! the **probit** branch of the same FFI-facing helper.
//!
//! `apply_inverse_link_vec` (`src/families/inverse_link.rs`) is the single
//! dispatch point that backs every Python posterior path that moves draws
//! from the link scale (η) to the response scale (μ):
//!
//!   * `gamfit/_sampling.py::PosteriorSamples.predict_draws` calls the PyO3
//!     entry `apply_inverse_link_array`, a thin wrapper over
//!     `apply_inverse_link_vec`.
//!   * `inference::eta_bands` and `inference::posterior_bands` invoke it
//!     directly to turn link-scale credible bounds into response-scale
//!     bounds.
//!
//! The probit branch is implemented as
//!
//! ```ignore
//! "probit" => {
//!     let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
//!     for &e in eta {
//!         out.push(0.5 * (1.0 + statrs::function::erf::erf(e * inv_sqrt2)));
//!     }
//! }
//! ```
//!
//! (`src/families/inverse_link.rs:29-34`).  The naive form
//! `Φ(η) = ½(1 + erf(η/√2))` suffers catastrophic cancellation in the deep
//! negative tail: `erf(η/√2)` saturates at the IEEE-754 value `-1.0` for
//! η ≲ -8.3, so `½(1 + (-1)) = 0.0` is returned even though the
//! mathematically correct value `Φ(η)` is strictly positive.  Already at
//! η = -8 the relative error is ~2%, and for η ≲ -9 the helper returns
//! exactly `0.0` (true `Φ(-9) ≈ 1.13e-19`).
//!
//! The crate already knows the numerically stable form: `mixture_link.rs`
//! comments `Φ(x) = 0.5 * erfc(-x / sqrt(2))`, and the GPU/PIRLS probit
//! numerics (`gpu/bms_flex_row.rs`, `solver/pirls/mod.rs`) use `erfc` /
//! `erfcx` / `log_ndtr` precisely to avoid this cancellation.  The FFI
//! helper does not share that implementation.  `erfc` is exact (it does
//! *not* go through `1 - erf`), so `½·erfc(-η/√2)` preserves digits down to
//! the f64 underflow boundary near η ≈ -38.
//!
//! This mirrors the closed cloglog test
//! `tests/bug_hunt_apply_inverse_link_cloglog_deep_tail_precision.rs`.

use gam::families::inverse_link::apply_inverse_link_vec;

/// Numerically stable probit inverse link via `erfc`, matching the
/// `Φ(x) = 0.5·erfc(-x/√2)` identity the crate documents in
/// `solver/mixture_link.rs`.  `statrs`'s `erfc` uses a dedicated
/// complementary tail evaluator (not `1 - erf`), so this is accurate to
/// f64 across the whole representable η range.
fn probit_inv_link_stable(eta: f64) -> f64 {
    let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
    0.5 * statrs::function::erf::erfc(-eta * inv_sqrt2)
}

#[test]
fn probit_apply_inverse_link_vec_is_strictly_positive_in_deep_negative_tail() {
    // Φ(-9) ≈ 1.13e-19 is small but strictly positive and well within the
    // f64 normal range.  The naive ½(1+erf(η/√2)) form returns exactly 0.0
    // here because erf(-9/√2) == -1.0 in f64.
    let eta = vec![-9.0_f64];
    let mu = apply_inverse_link_vec(&eta, "probit").expect("probit dispatch succeeds");
    assert_eq!(mu.len(), 1);
    let mu_eta = mu[0];

    // Φ maps R → (0, 1); a finite η returning exactly 0.0 is a precision
    // failure that cascades to -inf through any ln(μ) used in posterior
    // log-likelihood / log-loss summaries over the returned draws.
    assert!(
        mu_eta > 0.0,
        "apply_inverse_link_vec('probit')[-9] returned mu = {mu_eta:.3e}; \
         the probit inverse link Φ(η) must be strictly positive on every finite η"
    );

    let reference = probit_inv_link_stable(-9.0);
    let rel_err = (mu_eta - reference).abs() / reference;
    assert!(
        rel_err < 1e-9,
        "apply_inverse_link_vec('probit')[-9] = {mu_eta:.6e}, stable reference \
         (½·erfc(-η/√2)) = {reference:.6e}, relative error = {rel_err:.3e}; \
         the FFI probit helper must agree with the crate's documented stable form"
    );
}

#[test]
fn probit_apply_inverse_link_vec_matches_stable_reference_before_total_collapse() {
    // At η = -8 the response is still representable as a nonzero f64
    // (Φ(-8) ≈ 6.22e-16), but the naive ½(1+erf) form already carries a
    // ~2% relative error from cancellation.  A correct probit inverse link
    // agrees with ½·erfc(-η/√2) to near machine precision here.
    let eta = vec![-8.0_f64];
    let mu = apply_inverse_link_vec(&eta, "probit").expect("probit dispatch succeeds");
    let mu_eta = mu[0];
    let reference = probit_inv_link_stable(-8.0);
    let rel_err = (mu_eta - reference).abs() / reference;
    assert!(
        rel_err < 1e-6,
        "apply_inverse_link_vec('probit')[-8] = {mu_eta:.6e}, stable reference = \
         {reference:.6e}, relative error = {rel_err:.3e}; the cancelling \
         ½(1+erf(η/√2)) form loses ~2% here while ½·erfc(-η/√2) is exact"
    );
}

#[test]
fn probit_apply_inverse_link_vec_preserves_monotonicity_across_cancellation_boundary() {
    // Φ(η) is strictly increasing, so on a strictly decreasing η-grid the
    // response must be strictly decreasing.  The naive form collapses every
    // η ≲ -9 to the same 0.0, breaking strict monotonicity (and corrupting
    // any order-sensitive posterior summary: quantiles, calibration, ranks).
    let eta = vec![-6.0_f64, -7.0, -8.0, -9.0, -10.0, -12.0];
    let mu = apply_inverse_link_vec(&eta, "probit").expect("probit dispatch succeeds on deep tail");
    assert_eq!(mu.len(), eta.len());

    for win in mu.windows(2) {
        let (prev, next) = (win[0], win[1]);
        assert!(
            prev > next,
            "apply_inverse_link_vec('probit') is not strictly monotone across the \
             η = [-6, -7, -8, -9, -10, -12] grid: full response sequence {mu:?}. \
             Φ(η) is strictly increasing, so a strictly decreasing η-grid must map \
             to a strictly decreasing response"
        );
    }
}

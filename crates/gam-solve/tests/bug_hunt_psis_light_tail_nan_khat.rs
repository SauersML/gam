//! Regression for #1655: the Zhang–Stephens GPD tail estimator used to reject a
//! perfectly valid LIGHT tail (`k<0`, `σ>0`), returning `None` that the consumer
//! mapped to `NaN`, because the weak toward-0.5 shrink flipped the reported shape
//! positive and the `σ = −k/b` guard then saw a spurious negative scale.
//!
//! These tests construct a textbook light tail (a flat block plus a short,
//! bounded tail) and pin both the high-level contract (`pareto_smooth_weights`
//! returns `Some` with a finite, non-heavy `k_hat`) and the underlying numerical
//! defect (`fit_gpd_moments` reports a positive, finite scale on the excesses).

use gam_solve::psis::{MIN_TAIL_COUNT, fit_gpd_moments, pareto_smooth_weights};

const BASELINE: f64 = 1.0;
const TAIL_K: f64 = -0.2; // a genuinely LIGHT (bounded-support, negative-shape) tail
const TAIL_SIGMA: f64 = 0.5;
const TAIL_LEN: usize = 20;
const BLOCK_LEN: usize = 380;

/// Inverse-CDF of a GPD with shape `k` and scale `sigma` at probability `u`.
fn gpd_sample(u: f64, k: f64, sigma: f64) -> f64 {
    if k.abs() < 1e-12 {
        -sigma * (1.0 - u).ln()
    } else {
        sigma * ((1.0 - u).powf(-k) - 1.0) / k
    }
}

/// The textbook light tail this regression targets, as *excess* magnitudes
/// (already measured above the baseline): a short, bounded GPD(k<0) tail. With
/// `k=-0.2` the unshrunk profile shape is only mildly negative, so the old
/// toward-0.5 shrink flipped the reported shape positive and `σ = −k/b` went
/// negative — the exact rejection path of #1655.
fn light_tail_excesses() -> Vec<f64> {
    (0..TAIL_LEN)
        .map(|i| gpd_sample((i as f64 + 0.5) / TAIL_LEN as f64, TAIL_K, TAIL_SIGMA))
        .collect()
}

/// A clean light-tailed weight vector: a flat baseline block plus the bounded
/// light tail. The block size makes `ceil(sqrt(n)) == TAIL_LEN`, so the smoother
/// isolates exactly the tail above the baseline threshold and feeds the estimator
/// the same light-tail excesses.
fn light_tailed_weights() -> Vec<f64> {
    let mut w = vec![BASELINE; BLOCK_LEN];
    for e in light_tail_excesses() {
        w.push(BASELINE + e);
    }
    w
}

#[test]
fn pareto_smooth_weights_fits_a_clean_light_tail() {
    let w = light_tailed_weights();
    let out = pareto_smooth_weights(&w)
        .expect("a clean, well-conditioned light tail must fit (returned None: #1655)");
    assert!(
        out.k_hat.is_finite(),
        "k_hat must be finite for a light tail, got {}",
        out.k_hat
    );
    assert!(
        out.k_hat < 0.5,
        "a bounded light tail must NOT be flagged heavy; got k_hat={}",
        out.k_hat
    );
}

#[test]
fn fit_gpd_returns_positive_scale_on_a_light_tail() {
    // Feed the estimator exactly the light-tail excesses the smoother would.
    let excesses = light_tail_excesses();
    assert!(
        excesses.len() >= MIN_TAIL_COUNT,
        "test should exercise a real tail of excesses, got {}",
        excesses.len()
    );
    let (k_hat, sigma_hat) =
        fit_gpd_moments(&excesses).expect("light-tail excesses must fit (returned None: #1655)");
    assert!(
        sigma_hat.is_finite() && sigma_hat > 0.0,
        "GPD scale must be positive and finite on a light tail; got sigma_hat={sigma_hat}"
    );
    assert!(
        k_hat.is_finite() && k_hat < 0.5,
        "light tail shape must be finite and non-heavy; got k_hat={k_hat}"
    );
}

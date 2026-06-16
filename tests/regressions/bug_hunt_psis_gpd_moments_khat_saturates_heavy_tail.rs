//! Bug hunt: `fit_gpd_moments` (PSIS tail-shape estimator) cannot report a
//! heavy tail. Its method-of-moments shape is
//!
//!   k = 0.5 * (1 - mean^2 / var)            (src/inference/psis.rs:88)
//!
//! Because `mean^2 / var >= 0` for any real sample, the right-hand side is
//! structurally `<= 0.5` — the `.clamp(-0.5, 0.95)` upper bound is unreachable.
//! So no input, however heavy-tailed, can push `k_hat` past 0.5.
//!
//! This breaks the documented contract. The module header calls `k_hat` "the
//! same stability diagnostic used by PSIS-LOO" and says "values above roughly
//! 0.5 indicate that a few observations dominate the estimate." The whole point
//! of the PSIS-LOO k-hat is its ability to exceed the standard reliability
//! cutoff (k > 0.7 = "estimate unreliable, importance-sampling variance
//! effectively infinite"). The genuine danger cases — k_true around 0.8..1.0,
//! where the importance-weight mean barely exists or doesn't — get reported as
//! k_hat ~= 0.5, i.e. merely "borderline", giving false reassurance.
//!
//! Reproduction: feed `fit_gpd_moments` a large, exact, deterministic GPD(k,σ)
//! sample drawn by inverse-CDF (the same construction the in-module test uses
//! for k=0.35). The fitted shape saturates just under 0.5 regardless of how
//! heavy the true tail is:
//!
//!   k_true=0.70 -> k_hat ~= 0.496
//!   k_true=0.80 -> k_hat ~= 0.499
//!   k_true=0.95 -> k_hat ~= 0.500
//!
//! A consistent GPD shape estimator (e.g. the Zhang–Stephens profile estimator
//! that `loo`/ArviZ use, which is what "the diagnostic used by PSIS-LOO"
//! refers to) recovers k ~= k_true from such a sample, so the asserts below
//! start passing once the estimator is fixed — with no edits to this test.
//!
//! Root cause: src/inference/psis.rs:88 (`fit_gpd_moments`).
//! Related: #513, #514, #515 (other numerical-fidelity bugs in inference).

use gam::inference::psis::fit_gpd_moments;

/// Exact GPD(k, σ) quantile: x(u) = σ·((1-u)^(-k) - 1)/k for k != 0.
fn gpd_quantile(u: f64, k: f64, sigma: f64) -> f64 {
    if k.abs() < 1e-12 {
        -sigma * (1.0 - u).ln()
    } else {
        sigma * ((1.0 - u).powf(-k) - 1.0) / k
    }
}

/// A deterministic inverse-CDF sample of size `n` from GPD(k, σ).
fn deterministic_gpd_sample(n: usize, k: f64, sigma: f64) -> Vec<f64> {
    (1..=n)
        .map(|i| {
            let u = (i as f64 - 0.5) / n as f64;
            gpd_quantile(u, k, sigma)
        })
        .collect()
}

/// A genuinely heavy tail (k_true = 0.8) must be reported as heavy. The PSIS-LOO
/// reliability gate is k > 0.7; a tail-shape estimator that is "the same
/// diagnostic" must be able to cross it. The method-of-moments form returns
/// ~0.499 here — structurally incapable of exceeding 0.5 — so this fails today.
#[test]
fn psis_gpd_khat_crosses_reliability_gate_on_heavy_tail() {
    let k_true = 0.8_f64;
    let sigma = 1.3_f64;
    let xs = deterministic_gpd_sample(100_000, k_true, sigma);
    let (k_hat, _sigma_hat) =
        fit_gpd_moments(&xs).expect("GPD fit should succeed on positive data");
    assert!(
        k_hat > 0.7,
        "heavy GPD tail k_true={k_true} reported as k_hat={k_hat:.4}: below the \
         standard PSIS reliability cutoff (k>0.7). The estimate is structurally \
         capped at 0.5 by the method-of-moments form k=0.5*(1-mean^2/var) in \
         src/inference/psis.rs:88, so an unreliable importance-sampling tail is \
         flagged as merely borderline."
    );
}

/// Consistency across a range of heavy tails: the fitted shape should track the
/// true shape, not collapse onto a single saturation value. With an honest
/// estimator each k_hat lands near its k_true; the moment form returns
/// ~0.496/0.499/0.500 for all three — they are indistinguishable, which defeats
/// the diagnostic.
#[test]
fn psis_gpd_khat_tracks_heavy_tail_shape() {
    let sigma = 1.0_f64;
    for &k_true in &[0.7_f64, 0.85, 1.0] {
        let xs = deterministic_gpd_sample(100_000, k_true, sigma);
        let (k_hat, _) = fit_gpd_moments(&xs).expect("GPD fit should succeed");
        let err = (k_hat - k_true).abs();
        assert!(
            err < 0.1,
            "k_true={k_true}: fitted k_hat={k_hat:.4} is {err:.3} away — the \
             moment estimator saturates near 0.5 for every heavy tail and cannot \
             recover the true shape (src/inference/psis.rs:88)."
        );
    }
}

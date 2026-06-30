//! Regression for #1765: the Gaussian residual scale `σ̂²` must remain a
//! consistent estimate of the true noise variance even when REML selects a
//! very flexible (high-EDF) fit on low-noise data.
//!
//! The fit-level `standard_deviation` is the observation-noise scale that
//! `predict`'s OBSERVATION interval adds to `Var(μ̂)`:
//!
//!     half_width = z · sqrt( σ̂² + Var(μ̂(x)) ).
//!
//! When the mean is fit very accurately (high EDF, low residuals) the band is
//! dominated by `σ̂²`, so an underestimated `σ̂²` directly collapses
//! observation-interval coverage (the #1765 symptom: coverage < 0.75, PIT KS
//! ~0.18). The unbiased Gaussian scale (mgcv `gam.scale`) is
//!
//!     σ̂² = RSS / (n − edf_total),   edf_total = tr(F) = Σ edf_k,
//!
//! NOT the MLE `RSS / n`, and NOT the null-space `RSS / (n − mp)` divisor
//! (`mp = p − rank(ΣS_k)`): at high EDF the residuals from the flexible fit
//! shrink, and only the full `n − edf` divisor restores `E[σ̂²] = σ²_true`. The
//! #1765 defect was exactly the wrong denominator collapsing σ̂² low; the fix
//! (already in `optimizer.rs`) uses `n − edf_total`. This test locks that in:
//! it FAILS if the denominator regresses to `RSS/n` or `RSS/(n − mp)` and
//! PASSES on the residual-df scale.
//!
//! The fixture uses a full-rank ridge penalty (`S = I`, so `mp = p − p = 0`),
//! which makes both the MLE bug and the null-space bug collapse to the SAME
//! divisor `n`; the assertion below discriminates `RSS/(n − edf)` from `RSS/n`
//! and so catches either regression in one shot.

use super::*;
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N: usize = 1200;
const P: usize = 40;
const NOISE_SD: f64 = 0.05;

fn high_edf_options() -> FitOptions {
    FitOptions {
        compute_inference: true,
        max_iter: 200,
        tol: 1e-11,
        nullspace_dims: vec![0],
        ..FitOptions::default()
    }
}

/// A smooth, spectrally rich additive truth on a deterministic grid, projected
/// onto `P` Fourier-style basis columns with a ridge penalty. The signal is
/// genuinely wiggly so REML spends real EDF; the noise is small so an
/// over-flexible fit drives the residuals well below the true noise.
fn build_fixture() -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(1_765_017);
    let mut x = Array2::<f64>::zeros((N, P));
    let mut y = Array1::<f64>::zeros(N);
    for i in 0..N {
        let t = (i as f64) / ((N - 1) as f64); // t in [0, 1]
        x[[i, 0]] = 1.0;
        for j in 1..P {
            let freq = ((j + 1) / 2) as f64;
            let arg = std::f64::consts::PI * freq * t;
            x[[i, j]] = if j % 2 == 1 { arg.sin() } else { arg.cos() };
        }
        let mean = (std::f64::consts::PI * t).sin()
            + 0.6 * (2.0 * std::f64::consts::PI * t).cos()
            + 0.3 * (3.0 * std::f64::consts::PI * t).sin();
        // Box–Muller standard normal from two uniforms (no rand_distr dep).
        let u1: f64 = rng.random::<f64>().max(1e-12);
        let u2: f64 = rng.random::<f64>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        y[i] = mean + z * NOISE_SD;
    }
    let mut s = Array2::<f64>::zeros((P, P));
    for j in 0..P {
        s[[j, j]] = 1.0;
    }
    (x, y, s)
}

#[test]
fn gaussian_high_edf_scale_uses_residual_df_not_mle_1765() {
    let (x, y, s) = build_fixture();
    let weights = Array1::<f64>::ones(N);
    let offset = Array1::<f64>::zeros(N);
    let penalty = BlockwisePenalty::new(0..P, s.clone());

    let fit = fit_gam(
        x.clone(),
        y.view(),
        weights.view(),
        offset.view(),
        &[penalty],
        LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        &high_edf_options(),
    )
    .expect("Gaussian high-EDF fit");

    let edf = fit.edf_total().expect("edf_total available with inference");
    let sigma_hat = fit.standard_deviation;
    let sigma2_hat = sigma_hat * sigma_hat;

    // Recompute RSS from the fitted coefficients (β in the original basis).
    let fitted: Array1<f64> = x.dot(&fit.beta);
    let rss: f64 = y
        .iter()
        .zip(fitted.iter())
        .map(|(&yi, &fi)| (yi - fi).powi(2))
        .sum();

    let n = N as f64;
    let sigma2_residual_df = rss / (n - edf); // mgcv gam.scale
    let sigma2_mle = rss / n; // biased ML / null-space (mp=0) estimate (the bug)

    eprintln!(
        "#1765 high-EDF Gaussian scale: edf={edf:.3}, RSS={rss:.6}, \
         sigma_hat={sigma_hat:.6} (true={NOISE_SD}), \
         sigma2_hat={sigma2_hat:.8}, RSS/(n-edf)={sigma2_residual_df:.8}, \
         RSS/n={sigma2_mle:.8}"
    );

    // The fit must land in a genuinely high-EDF basin, otherwise the test does
    // not exercise the regime where the MLE/residual-df gap matters.
    assert!(
        edf >= 8.0,
        "fixture must produce a high-EDF fit to exercise #1765, got edf={edf:.3}"
    );
    // There must be a meaningful gap between the residual-df scale and the MLE,
    // so the assertion below genuinely discriminates the two estimators.
    let rel_gap = (sigma2_residual_df - sigma2_mle).abs() / sigma2_residual_df.max(1e-12);
    assert!(
        rel_gap > 0.01,
        "residual-df vs MLE scale gap too small to test (edf={edf:.3}, gap={rel_gap:.4})"
    );

    // (1) The reported σ̂² must be the residual-df estimate, NOT the MLE/null-space
    // divisor. This is the load-bearing #1765 assertion: it fails if the
    // denominator regresses to `n` (MLE) or `n − mp` (mp = 0 here ⇒ also `n`).
    let rel_to_residual_df =
        (sigma2_hat - sigma2_residual_df).abs() / sigma2_residual_df.max(1e-12);
    assert!(
        rel_to_residual_df < 5e-3,
        "reported σ̂²={sigma2_hat:.8} does not match RSS/(n-edf)={sigma2_residual_df:.8} \
         (rel={rel_to_residual_df:.4}); high-EDF Gaussian scale must use the residual \
         degrees of freedom, not the MLE RSS/n={sigma2_mle:.8} (#1765)"
    );

    // (2) The recovered noise sd must be a sane estimate of the truth — not a
    // collapsed fraction of it. The residual-df correction is exactly what keeps
    // σ̂ near σ_true at high EDF; the MLE/null-space divisor lands below this band.
    assert!(
        (0.7 * NOISE_SD..=1.4 * NOISE_SD).contains(&sigma_hat),
        "recovered noise sd σ̂={sigma_hat:.6} is off the true {NOISE_SD} \
         (edf={edf:.3}); the high-EDF residual scale is mis-estimated (#1765)"
    );
}

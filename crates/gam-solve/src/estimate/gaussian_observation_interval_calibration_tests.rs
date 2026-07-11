//! Regression for #1765: the Gaussian OBSERVATION prediction interval must be
//! calibrated in the low-noise / high-EDF regime (the issue reported coverage
//! < 0.75 and PIT KS ~0.18).
//!
//! The observation band a new response `y* = μ(x*) + ε` is covered by is
//!
//!     μ̂(x*) ± z · sqrt( Var(μ̂(x*)) + σ̂² ),
//!
//! i.e. the fitted-mean POSTERIOR variance `Var(μ̂) = x*ᵀ Vp x*` PLUS the
//! observation-noise variance `σ̂²`. Two independent mistakes each collapse this
//! band and produce the #1765 undercoverage, and this test pins both:
//!
//!   1. A residual scale `σ̂²` biased low. The correct Gaussian scale is the
//!      mgcv `gam.scale` residual-df estimate `σ̂² = RSS/(n − edf_total)`, NOT
//!      the MLE `RSS/n` nor the null-space `RSS/(n − mp)` divisor. At high EDF
//!      the flexible fit shrinks the residuals, so the MLE/null-space divisor
//!      lands σ̂ well below σ_true and the band is too narrow.
//!
//!   2. Dropping the mean-posterior term `Var(μ̂)`. At LOW edf/n it is
//!      negligible, but at HIGH edf/n (this fixture: edf/n ≈ 0.28) it is a large
//!      fraction of the total predictive variance — here ≈ 28% of σ² — so an
//!      observation band that adds only `σ̂²` undercovers materially.
//!
//! The production observation band (`gam_predict::family_observation_band`,
//! Gaussian arm) is `sqrt(etavar + σ̂²)`, using the smoothing-corrected `Vp`
//! covariance for `etavar` and the residual-df `σ̂`. This test reconstructs that
//! exact band on a real REML fit and asserts nominal held-out coverage and a
//! small PIT KS, and separately demonstrates that both ingredients are
//! load-bearing (the MLE scale and the mean-term-dropped band both undercover).

use super::*;
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N: usize = 300;
const P: usize = 160;
const NOISE_SD: f64 = 0.02;
const Z95: f64 = 1.959964;

fn erf_approx(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26, |err| < 1.5e-7 — ample for a PIT KS guard.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

/// One-sample Kolmogorov–Smirnov statistic against Uniform(0,1).
fn ks_vs_uniform(mut u: Vec<f64>) -> f64 {
    u.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = u.len() as f64;
    let mut d = 0.0f64;
    for (i, &ui) in u.iter().enumerate() {
        let lo = i as f64 / n;
        let hi = (i as f64 + 1.0) / n;
        d = d.max((ui - lo).abs()).max((hi - ui).abs());
    }
    d
}

/// Standard normal via Box–Muller (no rand_distr dependency).
fn box_muller(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(1e-12);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// `x*ᵀ Σ x*` for a single design row against a covariance matrix.
fn quad_form(cov: &Array2<f64>, x: &Array2<f64>, row: usize, p: usize) -> f64 {
    let mut acc = 0.0;
    for a in 0..p {
        let xa = x[[row, a]];
        if xa == 0.0 {
            continue;
        }
        let mut inner = 0.0;
        for b in 0..p {
            inner += cov[[a, b]] * x[[row, b]];
        }
        acc += xa * inner;
    }
    acc
}

#[test]
fn gaussian_observation_interval_calibrated_high_edf_1765() {
    // Deterministic fixture: a smooth, spectrally rich truth projected onto a
    // Fourier-style basis with a ridge penalty. Low noise + a rich basis drive
    // REML into a genuinely high-EDF fit (edf/n ≈ 0.28) — the #1765 regime.
    let mut rng = StdRng::seed_from_u64(13);
    let mut x = Array2::<f64>::zeros((N, P));
    let mut mean_true = Array1::<f64>::zeros(N);
    for i in 0..N {
        let t = (i as f64) / ((N - 1) as f64);
        x[[i, 0]] = 1.0;
        for j in 1..P {
            let freq = ((j + 1) / 2) as f64;
            let arg = std::f64::consts::PI * freq * t;
            x[[i, j]] = if j % 2 == 1 { arg.sin() } else { arg.cos() };
        }
        let mut m = 0.0;
        for k in 1..=8 {
            m += (1.0 / k as f64) * (std::f64::consts::PI * k as f64 * t).sin();
        }
        mean_true[i] = m;
    }
    let mut y = Array1::<f64>::zeros(N);
    for i in 0..N {
        y[i] = mean_true[i] + box_muller(&mut rng) * NOISE_SD;
    }
    let mut s = Array2::<f64>::zeros((P, P));
    for j in 0..P {
        s[[j, j]] = 1.0;
    }
    let weights = Array1::<f64>::ones(N);
    let offset = Array1::<f64>::zeros(N);
    let penalty = BlockwisePenalty::new(0..P, s.clone());
    let opts = FitOptions {
        compute_inference: true,
        max_iter: 200,
        tol: 1e-11,
        nullspace_dims: vec![0],
        ..FitOptions::default()
    };
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
        &opts,
    )
    .expect("Gaussian high-EDF fit");

    let edf = fit.edf_total().expect("edf_total with inference");
    let sigma = fit.standard_deviation;
    let fitted: Array1<f64> = x.dot(&fit.beta);
    let rss: f64 = y
        .iter()
        .zip(fitted.iter())
        .map(|(&yi, &fi)| (yi - fi).powi(2))
        .sum();
    let nf = N as f64;

    // Regime guard: this must be a genuinely high-EDF fit, otherwise the
    // mean-posterior term is negligible and the test does not exercise #1765.
    let edf_ratio = edf / nf;
    assert!(
        edf_ratio > 0.15,
        "fixture must reach the high-EDF regime for #1765, got edf/n={edf_ratio:.3} (edf={edf:.1})"
    );

    // (1) SCALE IDENTITY: σ̂² is the residual-df estimator, not the MLE. The MLE
    // RSS/n is the historical #1765 defect that collapses the band.
    let sig2_edf = rss / (nf - edf);
    let sig2_mle = rss / nf;
    let rel_scale = (sigma * sigma - sig2_edf).abs() / sig2_edf.max(1e-18);
    assert!(
        rel_scale < 5e-3,
        "σ̂²={:.10} must equal RSS/(n-edf)={sig2_edf:.10} (rel={rel_scale:.4}), not the MLE \
         RSS/n={sig2_mle:.10} (#1765 scale)",
        sigma * sigma
    );
    // The MLE/residual-df gap must be real, so the identity above genuinely
    // discriminates the two estimators in this regime.
    assert!(
        (sig2_edf - sig2_mle) / sig2_edf > 0.10,
        "residual-df vs MLE scale gap too small to test (edf/n={edf_ratio:.3})"
    );

    // (2) FULL OBSERVATION BAND — exactly gam_predict::family_observation_band's
    // Gaussian arm: half-width z·sqrt(Var(μ̂) + σ̂²), Var(μ̂) from the
    // smoothing-corrected Vp covariance. Fresh held-out noise at the design rows.
    let vp = fit
        .beta_covariance_corrected()
        .expect("smoothing-corrected covariance Vp");
    let mut hits_full = 0usize;
    let mut hits_noise_only = 0usize;
    let mut hits_mle = 0usize;
    let mut mean_etavar = 0.0;
    let mut pit_full = Vec::with_capacity(N);
    for i in 0..N {
        let y_new = mean_true[i] + box_muller(&mut rng) * NOISE_SD;
        let etavar = quad_form(vp, &x, i, P);
        mean_etavar += etavar;
        let mu = fitted[i];
        let sd_full = (etavar + sigma * sigma).max(0.0).sqrt();
        let sd_noise_only = sigma;
        // MLE-collapsed band: both the noise term AND etavar scale with σ̂².
        let sd_mle = (etavar * (sig2_mle / (sigma * sigma)) + sig2_mle).max(0.0).sqrt();
        let dev = (y_new - mu).abs();
        if dev <= Z95 * sd_full {
            hits_full += 1;
        }
        if dev <= Z95 * sd_noise_only {
            hits_noise_only += 1;
        }
        if dev <= Z95 * sd_mle {
            hits_mle += 1;
        }
        pit_full.push(normal_cdf((y_new - mu) / sd_full));
    }
    mean_etavar /= nf;
    let cov_full = hits_full as f64 / nf;
    let cov_noise_only = hits_noise_only as f64 / nf;
    let cov_mle = hits_mle as f64 / nf;
    let ks_full = ks_vs_uniform(pit_full);

    // The mean-posterior term must be a real share of the predictive variance
    // here (otherwise "high EDF" adds nothing to test).
    assert!(
        mean_etavar > 0.15 * sigma * sigma,
        "mean-posterior variance {mean_etavar:.8} should be a real share of σ²={:.8} at high EDF",
        sigma * sigma
    );

    // PRIMARY #1765 CONTRACT: the full observation band is calibrated.
    assert!(
        (0.925..=0.975).contains(&cov_full),
        "full observation-band coverage {cov_full:.3} off nominal 0.95 (edf/n={edf_ratio:.3}, \
         σ̂={sigma:.6}); the high-EDF Gaussian observation interval is mis-calibrated (#1765)"
    );
    assert!(
        ks_full < 0.07,
        "full-band PIT KS {ks_full:.4} too large (issue reported ~0.18); the predictive scale \
         at high EDF is mis-estimated (#1765)"
    );

    // BOTH ingredients are load-bearing: dropping the mean-posterior term, or
    // collapsing σ̂ to the MLE, each undercovers materially in this regime. These
    // pin WHY the band is built the way it is (they are the two #1765 failure
    // modes) — a fix that regresses to either lands here.
    assert!(
        cov_noise_only < cov_full - 0.02,
        "dropping Var(μ̂) should undercover at high EDF: noise-only coverage {cov_noise_only:.3} \
         vs full {cov_full:.3} (#1765)"
    );
    assert!(
        cov_mle < cov_full - 0.02,
        "the MLE-collapsed scale should undercover: MLE coverage {cov_mle:.3} vs full \
         {cov_full:.3} (#1765)"
    );
}

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

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::RngExt;

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
    u.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("KS inputs are finite probabilities, so the order is total")
    });
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


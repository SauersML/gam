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


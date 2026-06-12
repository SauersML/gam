//! Analytic Pólya–Gamma moments for deterministic gate-block evidence
//! approximations (#1016).
//!
//! The Devroye sampler in [`crate::inference::polya_gamma`] draws `PG(1, c)`
//! variates; that path is for Gibbs posteriors and for validating the algebra
//! here. Evidence ranking uses deterministic approximations, so this module
//! carries no RNG: the closed-form moments are pure functions of `(b, c)`.
//!
//! ## Why a PG channel exists
//!
//! For a Bernoulli/binomial logit gate term with linear predictor
//! `ψ_i = x_iᵀγ + o_i`, shape `b_i`, and `κ_i = y_i − b_i/2`, the PSW (2013)
//! identity is
//!
//! ```text
//! exp(κ_i ψ_i) / (1 + exp ψ_i)^{b_i}
//!   = 2^{−b_i} · E_{ω_i ~ PG(b_i, 0)} exp(κ_i ψ_i − ½ ω_i ψ_i²).
//! ```
//!
//! Conditional on `ω`, the gate contribution is exactly Gaussian in the gate
//! coordinates, so the gate sub-block can be Schur-eliminated with a *true*
//! quadratic instead of a local logistic Hessian whose third/fourth-order skew
//! hides inside the Laplace error. Near a birth event the new atom's gate
//! logits sit near zero, which is exactly where the logistic block is least
//! Gaussian and a plain Laplace gate block mis-prices both sides of the
//! `K` vs `K+1` comparison.

/// Closed-form moments of `PG(b, c)`.
///
/// `mean = E[PG(b, c)] = b · tanh(c/2) / (2c)` with the removable `c → 0`
/// limit `b/4`; `variance = b · (sinh c − c) / (2 c³ (1 + cosh c))` with the
/// `c → 0` limit `b/24` (Polson, Scott & Windle 2013, eq. 4 and its second
/// cumulant). Both are even in `c`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PgMoments {
    /// `E[PG(b, c)]`.
    pub mean: f64,
    /// `Var[PG(b, c)]`.
    pub variance: f64,
}

/// Mean of `PG(b, c)` (PSW 2013, eq. 4): `E = b · tanh(c/2)/(2c)`, limit `b/4`.
#[inline]
pub fn pg_mean(b: f64, c: f64) -> f64 {
    let c_abs = c.abs();
    if c_abs < 1e-8 {
        0.25 * b
    } else {
        b * (0.5 * c_abs).tanh() / (2.0 * c_abs)
    }
}

/// Variance of `PG(b, c)`: `Var = b · (sinh c − c)/(2 c³ (1 + cosh c))`, limit `b/24`.
#[inline]
pub fn pg_variance(b: f64, c: f64) -> f64 {
    let c_abs = c.abs();
    if c_abs < 1e-6 {
        b / 24.0
    } else {
        let cosh_c = c_abs.cosh();
        let sinh_c = c_abs.sinh();
        b * (sinh_c - c_abs) / (2.0 * c_abs * c_abs * c_abs * (1.0 + cosh_c))
    }
}

/// Both closed-form moments of `PG(b, c)` in one call.
#[inline]
pub fn pg_moments(b: f64, c: f64) -> PgMoments {
    PgMoments {
        mean: pg_mean(b, c),
        variance: pg_variance(b, c),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Closed-form moments agree with the empirical PG(1, c) sampler mean to
    /// the sampler's own tolerance — locks the analytic formula to the Devroye
    /// truth rather than restating it.
    #[test]
    fn moments_match_devroye_sampler() {
        use crate::inference::polya_gamma::PolyaGamma;
        use rand::{SeedableRng, rngs::StdRng};
        let pg = PolyaGamma::new();
        for &c in &[0.0_f64, 0.5, 1.0, 3.0] {
            let mut rng = StdRng::seed_from_u64(11 ^ (c.to_bits()));
            let n = 200_000;
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for _ in 0..n {
                let s = pg.draw(&mut rng, c);
                sum += s;
                sum_sq += s * s;
            }
            let emp_mean = sum / n as f64;
            let emp_var = sum_sq / n as f64 - emp_mean * emp_mean;
            let m = pg_moments(1.0, c);
            assert!(
                (emp_mean - m.mean).abs() / m.mean.max(1e-9) < 2e-2,
                "PG(1,{c}) mean: emp {emp_mean}, analytic {}",
                m.mean
            );
            assert!(
                (emp_var - m.variance).abs() / m.variance.max(1e-9) < 5e-2,
                "PG(1,{c}) var: emp {emp_var}, analytic {}",
                m.variance
            );
        }
    }

    #[test]
    fn variance_scales_linearly_in_shape() {
        assert!((pg_variance(2.0, 0.0) - 1.0 / 12.0).abs() < 1e-15);
    }
}

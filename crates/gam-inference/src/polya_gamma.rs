// Pólya-Gamma PG(1, c) sampler via Devroye's algorithm.
//
// Adapted from the `polya-gamma` crate (v0.5.3) by Daniel Lyng,
// dual-licensed MIT OR Apache-2.0.  We inline only the PG(1, c) path
// because that is all the Gibbs sampler needs.
//
// Reference: Polson, Scott & Windle (2013), "Bayesian Inference for
// Logistic Models Using Pólya-Gamma Latent Variables", JASA 108(504).
//
// The Devroye sampler math (tail mass, truncated inverse-Gaussian branches,
// alternating-series coefficient, and the draw loop) lives once in
// [`crate::polya_gamma_core`]; this module is the production
// adapter that drives that core from any [`rand::Rng`].

use crate::polya_gamma_core::{PgRng, draw_pg1};
use rand::{Rng, RngExt};
use rand_distr::{Distribution, Exp as RandExp, Normal as RandNormal};

/// Adapter that exposes a `rand::Rng` plus cached `Exp(1)` / `N(0,1)`
/// distribution objects as the [`PgRng`] randomness source the shared Devroye
/// core consumes. Borrows the RNG for the duration of a single draw.
struct RandPgRng<'a, R: Rng + ?Sized> {
    rng: &'a mut R,
    exp: &'a RandExp<f64>,
    std_norm_sampler: &'a RandNormal<f64>,
}

impl<R: Rng + ?Sized> PgRng for RandPgRng<'_, R> {
    #[inline]
    fn next_unit(&mut self) -> f64 {
        self.rng.random::<f64>()
    }

    #[inline]
    fn next_exp(&mut self) -> f64 {
        self.exp.sample(self.rng)
    }

    #[inline]
    fn next_norm(&mut self) -> f64 {
        self.std_norm_sampler.sample(self.rng)
    }
}

/// Sampler for the Pólya-Gamma PG(1, c) distribution.
#[derive(Debug, Clone)]
pub struct PolyaGamma {
    exp: RandExp<f64>,
    std_norm_sampler: RandNormal<f64>,
}

impl PolyaGamma {
    /// Construct a stateless Pólya–Gamma sampler.  The struct caches the
    /// `Exp(1)` and `N(0,1)` distribution objects used by every Devroye
    /// rejection draw; the caller supplies the per-draw RNG via the
    /// `draw(...)` entry point, so a single `PolyaGamma` instance is
    /// safely reused across threads and chains.
    pub fn new() -> Self {
        Self {
            exp: RandExp::new(1.0).expect("Exp(1) valid"),
            std_norm_sampler: RandNormal::new(0.0, 1.0).expect("N(0,1) valid"),
        }
    }

    /// Draw a single PG(1, c) variate using Devroye's exact algorithm.
    pub fn draw<R: Rng + ?Sized>(&self, rng: &mut R, tilt: f64) -> f64 {
        let mut source = RandPgRng {
            rng,
            exp: &self.exp,
            std_norm_sampler: &self.std_norm_sampler,
        };
        draw_pg1(&mut source, tilt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    fn empirical_mean(c: f64, n: usize, seed: u64) -> f64 {
        let pg = PolyaGamma::new();
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| pg.draw(&mut rng, c)).sum::<f64>() / n as f64
    }

    /// E[PG(1,c)] = tanh(c/2) / (2c),  or 1/4 when c = 0.
    fn theoretical_mean(c: f64) -> f64 {
        if c.abs() < 1e-12 {
            0.25
        } else {
            (0.5 * c).tanh() / (2.0 * c)
        }
    }

    #[test]
    fn pg1_mean_matches_theory() {
        let n = 25_000;
        for (c, tol) in [(0.0, 0.05), (1.0, 0.10), (3.0, 0.10)] {
            let emp = empirical_mean(c, n, 42);
            let th = theoretical_mean(c);
            assert!(
                (emp - th).abs() / th.max(1e-12) < tol,
                "PG(1,{c}): empirical {emp}, theory {th}",
            );
        }
    }

    /// Higher-precision moment test: PG(1, c) mean and variance must match
    /// closed-form values to ~1e-3 relative at K = 1e6 samples.
    /// Variance: V[PG(1,c)] = (sinh(c) - c) / (2 c³ (1 + cosh c));  at c=0, 1/24.
    fn theoretical_variance(c: f64) -> f64 {
        if c.abs() < 1e-6 {
            1.0 / 24.0
        } else {
            (c.sinh() - c) / (2.0 * c * c * c * (1.0 + c.cosh()))
        }
    }

    #[test]
    fn pg1_moments_high_precision() {
        let pg = PolyaGamma::new();
        let k = 1_000_000usize;
        for &c in &[0.0_f64, 0.1, 1.0, 3.0, 10.0, 30.0] {
            let mut rng = StdRng::seed_from_u64(0xC0FFEE ^ ((c.to_bits() as u64).wrapping_mul(7)));
            let mut sum = 0.0_f64;
            let mut sum_sq = 0.0_f64;
            for _ in 0..k {
                let s = pg.draw(&mut rng, c);
                sum += s;
                sum_sq += s * s;
            }
            let mean = sum / k as f64;
            let var = sum_sq / k as f64 - mean * mean;
            let th_mean = theoretical_mean(c);
            let th_var = theoretical_variance(c);
            let mean_rel = (mean - th_mean).abs() / th_mean.max(1e-12);
            let var_rel = (var - th_var).abs() / th_var.max(1e-12);
            assert!(
                mean_rel < 5e-3,
                "PG(1,{c}) mean: emp {mean:.6e}, theory {th_mean:.6e}, rel {mean_rel:.3e}",
            );
            assert!(
                var_rel < 5e-3,
                "PG(1,{c}) var: emp {var:.6e}, theory {th_var:.6e}, rel {var_rel:.3e}",
            );
        }
    }
}

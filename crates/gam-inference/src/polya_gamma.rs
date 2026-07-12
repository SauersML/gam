//! Narrow `PG(1, c)` adapter over the upstream `polya-gamma` crate.
//!
//! The workspace uses `rand` 0.10 while `polya-gamma` 0.5 uses `rand` 0.8.
//! [`Rand08`] bridges only the old `rand_core` random-bit interface, so callers
//! keep the workspace RNG API and the sampler implementation remains wholly
//! upstream.

use rand::Rng;

/// Borrow a workspace `rand` 0.10 generator through the `rand` 0.8 interface
/// required by `polya-gamma` 0.5. Random bits are forwarded without reseeding
/// or buffering, preserving ownership of the caller's RNG stream.
struct Rand08<'a, R: Rng + ?Sized>(&'a mut R);

impl<R: Rng + ?Sized> rand_core_06::RngCore for Rand08<'_, R> {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        rand::Rng::next_u32(self.0)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        rand::Rng::next_u64(self.0)
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        rand::Rng::fill_bytes(self.0, dest);
    }

    #[inline]
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core_06::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

/// Sampler for the Pólya–Gamma `PG(1, c)` distribution.
///
/// This type intentionally exposes only the shape-one surface used by GAM's
/// Gibbs and validation paths. Shape selection and all sampling mathematics
/// stay inside the upstream crate.
#[derive(Debug, Clone)]
pub struct PolyaGamma {
    upstream: polya_gamma::PolyaGamma,
}

impl PolyaGamma {
    /// Construct an upstream sampler fixed at shape `b = 1`.
    pub fn new() -> Self {
        Self {
            upstream: polya_gamma::PolyaGamma::new(1.0),
        }
    }

    /// Draw a single `PG(1, c)` variate from the caller's `rand` 0.10 stream.
    ///
    /// A Pólya–Gamma variate is strictly positive with a continuous density,
    /// so an exact `0.0`, a negative value, or a non-finite value can only be
    /// a numerical artifact of the upstream sampler. Rejecting those
    /// measure-zero artifacts preserves the target distribution without a
    /// finite retry budget. Non-finite tilts are rejected before entering the
    /// upstream rejection sampler, where they would make its loop undefined.
    pub fn draw<R: Rng + ?Sized>(&self, rng: &mut R, tilt: f64) -> f64 {
        assert!(
            tilt.is_finite(),
            "PG(1, c) requires a finite tilt, got {tilt}"
        );
        loop {
            let v = self.upstream.draw(&mut Rand08(rng), tilt);
            if v.is_finite() && v > 0.0 {
                return v;
            }
        }
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

    /// `E[PG(1,c)] = tanh(c/2) / (2c)`, with limit `1/4` at `c = 0`.
    fn theoretical_mean(c: f64) -> f64 {
        if c.abs() < 1e-12 {
            0.25
        } else {
            (0.5 * c).tanh() / (2.0 * c)
        }
    }

    /// #2245 findings 26/27: every draw must lie in the strictly-positive
    /// PG(1, c) support and be finite, including the extreme-tilt regime where
    /// the exponential-tail mass computation is numerically delicate.
    #[test]
    fn pg1_draws_are_strictly_positive_and_finite_at_extreme_tilt() {
        let pg = PolyaGamma::new();
        for &c in &[0.0_f64, 1.0, 30.0, 95.0, 200.0, 700.0] {
            let mut rng = StdRng::seed_from_u64(0xABAD_1DEA ^ c.to_bits());
            for _ in 0..20_000 {
                let v = pg.draw(&mut rng, c);
                assert!(
                    v.is_finite() && v > 0.0,
                    "PG(1,{c}) draw outside strict-positive support: {v}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "PG(1, c) requires a finite tilt")]
    fn pg1_rejects_non_finite_tilt_before_calling_upstream() {
        let pg = PolyaGamma::new();
        let mut rng = StdRng::seed_from_u64(7);
        pg.draw(&mut rng, f64::NAN);
    }

    #[test]
    fn pg1_mean_matches_theory() {
        let n = 25_000;
        for (c, tol) in [(0.0, 0.05), (1.0, 0.10), (3.0, 0.10)] {
            let empirical = empirical_mean(c, n, 42);
            let theoretical = theoretical_mean(c);
            assert!(
                (empirical - theoretical).abs() / theoretical.max(1e-12) < tol,
                "PG(1,{c}): empirical {empirical}, theory {theoretical}",
            );
        }
    }

    /// `Var[PG(1,c)] = (sinh(c) - c) / (2 c³ (1 + cosh(c)))`, with
    /// limit `1/24` at `c = 0`.
    fn theoretical_variance(c: f64) -> f64 {
        if c.abs() < 1e-6 {
            1.0 / 24.0
        } else {
            (c.sinh() - c) / (2.0 * c * c * c * (1.0 + c.cosh()))
        }
    }

    /// Exact `PG(1, 0)` CDF, evaluated from the two alternating Jacobi-series
    /// representations on the side where each converges geometrically. If
    /// `X ~ PG(1, 0)`, then `4X ~ J*(1, 0)`.
    fn pg10_cdf(x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }

        let mut sum = 0.0;
        let mut n = 0usize;
        if x <= 1.0 / (2.0 * std::f64::consts::PI) {
            loop {
                let k = n as f64 + 0.5;
                let term = 4.0 * gam_math::probability::normal_cdf(-k / (2.0 * x).sqrt());
                sum += if n.is_multiple_of(2) { term } else { -term };
                if term <= f64::EPSILON {
                    break;
                }
                n += 1;
            }
            sum.clamp(0.0, 1.0)
        } else {
            loop {
                let k = n as f64 + 0.5;
                let term = 2.0 / (std::f64::consts::PI * k)
                    * (-2.0 * std::f64::consts::PI.powi(2) * k * k * x).exp();
                sum += if n.is_multiple_of(2) { term } else { -term };
                if term <= f64::EPSILON {
                    break;
                }
                n += 1;
            }
            (1.0 - sum).clamp(0.0, 1.0)
        }
    }

    #[test]
    fn pg10_distribution_matches_exact_cdf() {
        let sample_count = 20_000usize;
        let pg = PolyaGamma::new();
        let mut rng = StdRng::seed_from_u64(0xD15C_1CDF);
        let mut samples: Vec<f64> = (0..sample_count).map(|_| pg.draw(&mut rng, 0.0)).collect();
        samples.sort_by(f64::total_cmp);

        let n = sample_count as f64;
        let statistic = samples
            .iter()
            .enumerate()
            .map(|(i, &sample)| {
                let cdf = pg10_cdf(sample);
                let empirical_below = i as f64 / n;
                let empirical_through = (i + 1) as f64 / n;
                (cdf - empirical_below)
                    .abs()
                    .max((empirical_through - cdf).abs())
            })
            .fold(0.0_f64, f64::max);

        // Dvoretzky–Kiefer–Wolfowitz: P(D_n > epsilon) <=
        // 2 exp(-2 n epsilon²). Use a one-in-a-million false-rejection bound.
        let false_rejection_probability = 1e-6_f64;
        let critical = (-(false_rejection_probability / 2.0).ln() / (2.0 * n)).sqrt();
        assert!(
            statistic <= critical,
            "PG(1,0) one-sample KS statistic {statistic} exceeds DKW critical value {critical}",
        );
    }

    #[test]
    fn pg1_moments_high_precision() {
        let pg = PolyaGamma::new();
        let sample_count = 1_000_000usize;
        for &c in &[0.0_f64, 0.1, 1.0, 3.0, 10.0, 30.0] {
            let mut rng = StdRng::seed_from_u64(0xC0FFEE ^ (c.to_bits().wrapping_mul(7)));
            let mut sum = 0.0_f64;
            let mut sum_sq = 0.0_f64;
            for _ in 0..sample_count {
                let sample = pg.draw(&mut rng, c);
                sum += sample;
                sum_sq += sample * sample;
            }
            let mean = sum / sample_count as f64;
            let variance = sum_sq / sample_count as f64 - mean * mean;
            let expected_mean = theoretical_mean(c);
            let expected_variance = theoretical_variance(c);
            let mean_relative_error = (mean - expected_mean).abs() / expected_mean.max(1e-12);
            let variance_relative_error =
                (variance - expected_variance).abs() / expected_variance.max(1e-12);
            assert!(
                mean_relative_error < 5e-3,
                "PG(1,{c}) mean: empirical {mean:.6e}, theory {expected_mean:.6e}, relative error {mean_relative_error:.3e}",
            );
            assert!(
                variance_relative_error < 5e-3,
                "PG(1,{c}) variance: empirical {variance:.6e}, theory {expected_variance:.6e}, relative error {variance_relative_error:.3e}",
            );
        }
    }
}

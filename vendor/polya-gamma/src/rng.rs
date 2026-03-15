use super::PolyaGamma;
use rand::{Rng, RngExt, distr::Distribution};
use std::f64::consts::FRAC_2_PI;

/// Unified interface for sampling from various distributions
pub(crate) trait RngDraw<R: Rng + ?Sized> {
    fn sample_exp(&self, rng: &mut R) -> f64;
    fn sample_norm(&self, rng: &mut R) -> f64;
    fn sample_unif(&self, rng: &mut R) -> f64;
    fn sample_gamma(&self, rng: &mut R) -> f64;
    fn sample_trunc_inv_gauss(&self, rng: &mut R, z: f64, truncation_point: f64) -> f64;
}

impl<R: Rng + ?Sized> RngDraw<R> for PolyaGamma {
    /// Sample from the Exp(1) distribution
    #[inline(always)]
    fn sample_exp(&self, rng: &mut R) -> f64 {
        self.exp.sample(rng)
    }

    /// Sample from the standard normal distribution
    #[inline(always)]
    fn sample_norm(&self, rng: &mut R) -> f64 {
        self.std_norm_sampler.sample(rng)
    }

    /// Sample from the standard uniform distribution
    #[inline(always)]
    fn sample_unif(&self, rng: &mut R) -> f64 {
        rng.random::<f64>()
    }

    /// Sample from a truncated inverse Gaussian distribution.
    ///
    /// This function implements an efficient algorithm to sample from an inverse Gaussian
    /// distribution that is truncated to the interval (0, truncation_point]. The method uses
    /// different sampling strategies based on the value of the input parameter `z`.
    ///
    /// # Arguments
    /// * `z` - The shape parameter of the inverse Gaussian distribution
    /// * `rng` - Random number generator
    /// * `truncation_point` - The upper bound of the truncation interval
    ///
    /// # Returns
    /// A random variate from the truncated inverse Gaussian distribution
    ///
    /// # Algorithm
    /// For small `z` (z < 1/truncation_point), uses an exponential rejection sampling method.
    /// For larger `z`, uses a normal approximation with an acceptance-rejection step.
    #[inline(always)]
    fn sample_trunc_inv_gauss(&self, rng: &mut R, z: f64, truncation_point: f64) -> f64 {
        // Ensure z is positive
        let z = z.abs();

        // Use different sampling methods based on the value of z
        if FRAC_2_PI > z {
            // For small z, use exponential rejection sampling
            self.sample_small_z(rng, z, truncation_point)
        } else {
            // For larger z, use normal approximation with acceptance-rejection
            let mean = 1.0 / z;
            self.sample_large_z(rng, mean, truncation_point)
        }
    }

    /// Sample from a gamma distribution with shape parameter `a` and scale parameter `b`
    #[inline(always)]
    fn sample_gamma(&self, rng: &mut R) -> f64 {
        self.gamma.sample(rng)
    }
}

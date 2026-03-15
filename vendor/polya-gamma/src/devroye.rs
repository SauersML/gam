use crate::{PolyaGamma, rng::RngDraw};
use rand::Rng;
use statrs::distribution::{Continuous, ContinuousCDF, Exp, InverseGamma};

use std::f64::consts::{FRAC_2_PI, FRAC_PI_2, PI};
const PI_SQ: f64 = std::f64::consts::PI * std::f64::consts::PI;

/// Draw exact samples from the Polya-Gamma distribution PG(1, c) using Devroye's algorithm.
///
/// This function implements Algorithm 1 from the paper "Bayesian inference for logistic models using
/// Polya-Gamma latent variables" by Polson, Scott, and Windle (2013). The sampler is motivated by
/// Devroye (2009) and uses a rejection sampling approach with a mixture of an exponential and an
/// inverse Gaussian distribution as the proposal distribution.
impl PolyaGamma {
    /// Sample from PG(1, c) using Devroye's algorithm.
    ///
    /// This function implements Algorithm 1 from the paper "Bayesian inference for logistic models using
    /// Polya-Gamma latent variables" by Polson, Scott, and Windle (2013). The sampler is motivated by
    /// Devroye (2009) and uses a rejection sampling approach with a mixture of an exponential and an
    /// inverse Gaussian distribution as the proposal distribution.
    ///
    /// # Arguments
    /// * `rng` - A mutable reference to a random number generator
    /// * `tilt_parameter` - The tilt parameter (c) of the PG(1, c) distribution
    ///
    /// # Returns
    /// A random variate from the PG(1, c) distribution
    pub(crate) fn sample_polya_gamma_devroye<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        tilt: f64,
    ) -> f64 {
        // Scale the tilt parameter as per the algorithm
        let half_tilt = tilt.abs() * 0.5;
        let half_tilt_sq = half_tilt * half_tilt;

        // Precompute frequently used terms
        let scale_factor = 0.125 * PI_SQ + (0.5 * half_tilt_sq);
        let exp_mass = self.exponential_tail_mass(half_tilt);

        // Main sampling loop
        loop {
            // Generate a uniform random number for proposal selection
            let uniform_sample: f64 = self.sample_unif(rng);

            // Select a proposal distribution based on the uniform sample
            let proposal = if uniform_sample < exp_mass {
                // Use exponential tail proposal
                FRAC_2_PI + self.sample_exp(rng) / scale_factor
            } else {
                // Use inverse Gaussian proposal
                self.sample_trunc_inv_gauss(rng, half_tilt, FRAC_2_PI)
            };

            // Compute the first term of the series
            let mut series_sum = self.series_coefficient(0, proposal);
            let threshold = self.sample_unif(rng) * series_sum;
            let mut term_index = 0;

            // Series evaluation loop with alternating signs
            'series_eval: loop {
                term_index += 1;
                let term = self.series_coefficient(term_index, proposal);

                if term_index % 2 == 1 {
                    // For odd terms, subtract from the sum
                    series_sum -= term;
                    if threshold <= series_sum {
                        // Accept the sample
                        return 0.25 * proposal;
                    }
                } else {
                    // For even terms, add to the sum
                    series_sum += term;
                    if threshold >= series_sum {
                        // Reject and try again
                        break 'series_eval;
                    }
                }
            }
        }
    }

    /// Calculate the mass function for the Polya-Gamma distribution's exponential term.
    ///
    /// This function computes the mass function used in the rejection sampling algorithm
    /// for the Polya-Gamma distribution as part of the Devroye sampling method.
    ///
    /// # Arguments
    /// * `tilt` - The tilt parameter for the mass function
    ///
    /// # Returns
    /// The computed mass value between 0 and 1
    fn exponential_tail_mass(&self, tilt: f64) -> f64 {
        // Compute the base term that combines the Polya-Gamma distribution parameters
        let base_term = (0.125 * PI_SQ) + (0.5 * tilt * tilt);

        // Calculate the upper and lower bounds for the normal CDF evaluation
        let upper_bound = (FRAC_PI_2).sqrt() * (FRAC_2_PI * tilt - 1.0);
        let lower_bound = -((FRAC_PI_2).sqrt() * (FRAC_2_PI * tilt + 1.0));

        // Compute the log of the base term and adjust by the truncation level
        let log_base = base_term.ln() + (base_term * FRAC_2_PI);

        // Calculate the log probabilities for the upper and lower bounds
        // Note: cdf(x).ln() is unstable for 0 >> x, but we don't expect to see such values.
        let log_prob_upper = log_base - tilt + self.std_norm.cdf(upper_bound).ln();
        let log_prob_lower = log_base + tilt + self.std_norm.cdf(lower_bound).ln();

        // Combine the probabilities and compute the final mass value
        let exp_terms =
            (4.0 / std::f64::consts::PI) * (log_prob_upper.exp() + log_prob_lower.exp());
        1.0 / (1.0 + exp_terms)
    }

    /// Calculate the density of the Jacobi distribution at `x` (coefficient term for the Polya-Gamma distribution approximation.
    ///
    /// For more information, see equations 15 and 16 in the Polson et al. (2013).
    ///
    /// # Arguments
    /// * `n` - The index term in the series expansion
    /// * `x` - The point at which to evaluate the coefficient
    ///
    /// # Returns
    /// The computed coefficient value, which is always non-negative
    fn series_coefficient(&self, n: usize, x: f64) -> f64 {
        // Calculate the k-th term in the series (k = n + 0.5)
        let k0 = n as f64 + 0.5;

        // Use different approximations based on the value of x relative to TRUNC
        if x <= 0.0 {
            // Outside the support of the distribution
            0.0
        } else if x <= FRAC_2_PI {
            // For small x, we can use the inverse gamma distribution with parameterization
            // IG(0.5, 2k^2)
            if n < self.inv_gamma.len() {
                // Use precomputed distribution if available
                self.inv_gamma[n].pdf(x) * 2.0
            } else {
                eprintln!("WARNING: series_exp is out of bounds; performance will be degraded.");
                // Calculate on the fly if out of precomputed bounds
                let ig = InverseGamma::new(0.5, 2.0 * k0 * k0)
                    .expect("InverseGamma(0.5, 2k^2) is always valid because k > 0.5");
                ig.pdf(x) * 2.0
            }
        } else {
            // For large x, use the exponential distribution with parameterization
            // Exp(PI*k/(PI^2*k^3))
            let scale = 1.0 / (PI * k0 * k0);
            if n < self.series_exp.len() {
                // Use precomputed distribution if available
                scale * self.series_exp[n].pdf(x)
            } else {
                eprintln!("WARNING: series_exp is out of bounds; performance will be degraded.");
                // Calculate on the fly if out of precomputed bounds
                let exp = Exp::new(k0 * k0 * PI * PI / 2.0)
                    .expect("Exp(k^2 * PI^2 / 2) is always valid because k > 0.5");
                scale * exp.pdf(x)
            }
        }
    }

    /// Sample from an exponential distribution using Marsaglia's method and transform
    /// to get a sample from the truncated inverse Gaussian distribution.
    ///
    /// This is used for small values of z (z < 1/TRUNC) where exponential rejection sampling
    /// is more efficient than the normal approximation method.
    ///
    /// # Arguments
    /// * `z` - The shape parameter of the inverse Gaussian distribution
    /// * `rng` - Random number generator
    /// * `truncation_point` - The upper truncation point
    ///
    /// # Returns
    /// A random variate from the truncated inverse Gaussian distribution
    pub(crate) fn sample_small_z<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        z: f64,
        truncation_point: f64,
    ) -> f64 {
        let mut acceptance_prob = 0.0;
        let mut sample = 0.0;

        // Keep sampling until we accept a value
        while acceptance_prob < self.sample_unif(rng) {
            // Sample from exponential distribution using Marsaglia's method
            let exp_sample = loop {
                let e1 = self.sample_exp(rng);
                let e2 = self.sample_exp(rng);
                if (e1 * e1) <= (2.0 * e2 / truncation_point) {
                    break e1;
                }
            };

            // Transform the exponential sample to get a truncated sample
            sample = 1.0 + (exp_sample * truncation_point);
            sample = truncation_point / (sample * sample);

            // Calculate acceptance probability
            acceptance_prob = (-0.5 * z * z * sample).exp();
        }
        sample
    }

    /// Sample from a truncated inverse Gaussian distribution using normal approximation.
    ///
    /// This is used for larger values of z (z >= 1/truncation_point) where normal approximation
    /// with acceptance-rejection is more efficient than exponential rejection sampling.
    ///
    /// # Arguments
    /// * `mean` - The mean of the inverse Gaussian distribution (1/z)
    /// * `rng` - Random number generator
    /// * `truncation_point` - The upper truncation point
    ///
    /// # Returns
    /// A random variate from the truncated inverse Gaussian distribution
    pub(crate) fn sample_large_z<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        mean: f64,
        truncation_point: f64,
    ) -> f64 {
        let mut sample = f64::INFINITY;

        // Keep sampling until we get a value within the truncation bound
        while sample > truncation_point {
            // Sample from standard normal distribution
            let normal_sample = self.sample_norm(rng);
            let normal_sample_sq = normal_sample * normal_sample;

            // Calculate components of the inverse Gaussian transform
            let half_mean = 0.5 * mean;
            let mean_times_normal_sq = mean * normal_sample_sq;

            // Apply the inverse Gaussian transform
            let discriminant =
                (4.0 * mean_times_normal_sq + mean_times_normal_sq * mean_times_normal_sq).sqrt();
            sample = mean + (half_mean * mean_times_normal_sq) - (half_mean * discriminant);

            // Apply acceptance-rejection step to ensure correct distribution
            if self.sample_unif(rng) > (mean / (mean + sample)) {
                sample = (mean * mean) / sample;
            }
        }
        sample
    }
}

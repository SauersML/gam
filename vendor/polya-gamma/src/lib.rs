//! # Polya-Gamma Sampler and Bayesian Logistic Regression
//!
//! This crate provides an efficient sampler for Polya-Gamma (PG) random variates, along with a
//! Gibbs sampler for Bayesian logistic regression using PG data augmentation.
//!
//! ## Features
//!
//! - Polya-Gamma sampler:
//!   - Draws samples from the PG(b, c) distribution using different strategies depending on the value of `b`.
//!   - High-performance, high-accuracy sampling.
//!
//! - Bayesian regression:
//!   - Implements Gibbs samplers using PG augmentation for fully-conjugate updates of regression coefficients.
//!   - For logistic regression, see [`regression::GibbsLogit`].
//!   - For negative binomial regression (count data), see [`regression::GibbsNegativeBinomial`].
//!   - These are available under the `regression` feature flag.
//!
//! ## Mathematical Background
//!
//! The Polya-Gamma distribution PG(b, c) is used for data augmentation in models with logistic link functions,
//! enabling efficient Bayesian inference. See:
//!
//! - Polson, N.G., Scott, J.G., & Windle, J. (2013). Bayesian Inference for Logistic Models Using Polya-Gamma Latent Variables. *JASA*, 108(504): 1339–1349.
//! - Windle, J., Polson, N.G., & Scott, J.G. (2014). Sampling Pólya-Gamma random variates: alternate and approximate techniques. arXiv:1405.0506.
//!
//! ## Usage Example
//!
//! ```rust
//! # use rand::SeedableRng;
//! # use rand::rngs::StdRng;
//! use polya_gamma::PolyaGamma;
//! let pg = PolyaGamma::new(1.0);
//! let sample = pg.draw(&mut StdRng::seed_from_u64(0), 1.0);
//! ```
//!
//! For examples of Bayesian regression models, see the documentation for the [`regression`] module and the specific model structs like [`regression::GibbsLogit`].
//! The `examples` directory in the repository also contains runnable examples.
//! ## License
//! This crate is dual-licensed under the MIT OR Apache-2.0 licenses.
//! See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE) for details.

use rand::{Rng, SeedableRng, rng};
use rand_distr::{Exp as RandExp, Gamma as RandGamma, Normal as RandNormal};
use rng::RngDraw;
use statrs::distribution::{Exp as StatrsExp, InverseGamma, Normal};
use std::f64::consts::PI;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

const PI_SQ: f64 = std::f64::consts::PI * std::f64::consts::PI;
const PI2_SQ_RECIP: f64 = 1.0 / (2.0 * PI_SQ);

/// Polya-Gamma sampler.
///
/// The `PolyaGamma` struct enables sampling from the Polya-Gamma distribution PG(b, c)
/// using a either a finite sum-of-gammas approximation or exact sampling following Devroye (2009).
///
/// # Example
/// ```rust
/// # use rand::SeedableRng;
/// # use rand::rngs::StdRng;
/// use polya_gamma::PolyaGamma;
/// let pg = PolyaGamma::new(1.0);
/// let sample = pg.draw(&mut StdRng::seed_from_u64(0), 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct PolyaGamma {
    exp: RandExp<f64>,
    std_norm: Normal,
    std_norm_sampler: RandNormal<f64>,
    gamma: RandGamma<f64>,
    inv_gamma: Vec<InverseGamma>,
    series_exp: Vec<StatrsExp>,
    shape: f64,
}

impl PolyaGamma {
    /// Create a new PolyaGamma sampler with a shape parameter.
    ///
    /// Note: values of the tilt parameter `c` are passed to the `draw` and `draw_vec` methods.
    ///
    /// # Arguments
    /// * `shape` - Shape parameter `b` for PG(b,c)
    ///
    /// # Panics
    /// Panics if `shape` is not positive.
    pub fn new(shape: f64) -> Self {
        assert!(shape > 0.0, "Shape parameter must be positive");
        const PRECOMPUTE_K: usize = 50;
        Self {
            exp: RandExp::new(1.0).expect("Exp(1) is always valid"),
            std_norm: Normal::standard(),
            std_norm_sampler: RandNormal::new(0.0, 1.0)
                .expect("Normal(0,1) is always valid"),
            gamma: RandGamma::new(shape, 1.0).expect("Gamma(1,1) is always valid"),
            // Precompute Levy distributions for k=1..=10 for series approximation
            inv_gamma: (0..PRECOMPUTE_K)
                .map(|k| {
                    {
                        let k = k as f64 + 0.5;
                        InverseGamma::new(0.5, 2.0 * k * k)
                    }
                    .expect("InverseGamma(0.5,2k^2) is always valid because k > 0.5")
                })
                .collect(),
            // Precompute exponential distributions for k=1..=10 for series approximation
            series_exp: (0..PRECOMPUTE_K)
                .map(|k| {
                    let k = k as f64 + 0.5;
                    StatrsExp::new(k * k * PI_SQ / 2.0)
                        .expect("Exp(k^2 * PI^2 / 2) is always valid because k > 0.5")
                })
                .collect(),
            shape,
        }
    }

    pub fn set_shape(&mut self, shape: f64) {
        self.shape = shape;
        self.init_gamma(shape);
    }
    /// Draw a single Polya-Gamma random variate PG(b, c).
    ///
    /// This function generates samples from the Polya-Gamma distribution with shape parameter `b`
    /// and tilt parameter `c`. It uses different sampling strategies based on the value of `b`:
    /// - For b = 1 or 2: Uses Devroye's exact sampling algorithm
    /// - For integer b > 2: Sums b independent PG(1, c) variates
    /// - For non-integer b: Uses a gamma-Poisson mixture representation
    ///
    /// # Arguments
    /// * `b` - Shape parameter (must be > 0)
    /// * `c` - Tilt parameter (real-valued)
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// A random variate from PG(b, c)
    ///
    /// # Panics
    /// Panics if `b` is not positive.
    ///
    /// # Example
    /// ```rust
    /// # use polya_gamma::PolyaGamma;
    /// let mut pg = PolyaGamma::new(1.0);
    /// let mut rng = rand::thread_rng();
    ///
    /// // Sample from PG(1, 0.5)
    /// let sample = pg.draw(&mut rng, 0.5);
    ///
    /// // Sample from PG(3.5, -1.2)
    /// pg.set_shape(3.5);
    /// let sample2 = pg.draw(&mut rng, -1.2);
    /// ```
    pub fn draw<R: Rng + ?Sized>(&self, rng: &mut R, tilt: f64) -> f64 {
        self.draw_internal(rng, self.shape, tilt)
    }

    /// Draw multiple Polya-Gamma random variates PG(b, c).
    ///
    /// # Arguments
    /// * `rng` - Mutable reference to a random number generator
    /// * `c` - Tilt parameters (real-valued)
    ///
    /// # Returns
    /// A vector of random variates from PG(b, c)
    ///
    /// # Panics
    /// Panics if `b` is not positive.
    ///
    /// # Example
    /// ```rust
    /// # use polya_gamma::PolyaGamma;
    /// let mut pg = PolyaGamma::new(1.0);
    /// let mut rng = rand::thread_rng();
    ///
    /// // Draw 100 samples from PG(1, 0.5)
    /// let samples = pg.draw_vec(&mut rng, &[0.5; 100]);
    /// println!("Drew {} samples from PG(1, 0.5)", samples.len());
    /// ```
    pub fn draw_vec<R: Rng + ?Sized>(&self, rng: &mut R, c: &[f64]) -> Vec<f64> {
        let b = self.shape;
        c.iter().map(|&c| self.draw_internal(rng, b, c)).collect()
    }

    /// Draw multiple Polya-Gamma random variates PG(b, c) in parallel.
    ///
    /// The initial seed is drawn from the provided `rng`. Each thread is then given a unique seed
    /// based on the initial seed. This ensures that the samples are deterministic across runs.
    ///
    /// Note that this function is slightly slower than `draw_vec_par`, which should be preferred
    /// in production workloads.
    ///
    /// # Arguments
    /// * `rng` - Mutable reference to a random number generator
    /// * `c` - Tilt parameters (real-valued)
    ///
    /// # Returns
    /// A vector of random variates from PG(b, c)
    ///
    /// # Panics
    /// Panics if `b` is not positive.
    ///
    /// # Example
    /// ```rust
    /// # use polya_gamma::PolyaGamma;
    /// # use rand::SeedableRng;
    /// # use rand::rngs::StdRng;
    /// let pg = PolyaGamma::new(1.0);
    /// let mut rng = StdRng::seed_from_u64(0);
    ///
    /// // Draw 100 samples from PG(1, 0.5)
    /// let samples = pg.draw_vec_par_deterministic(&mut rng, &[0.5; 100]);
    /// println!("Drew {} samples from PG(1, 0.5)", samples.len());
    /// ```
    #[cfg(feature = "rayon")]
    pub fn draw_vec_par_deterministic<R: SeedableRng + Rng>(
        &self,
        rng: &mut R,
        c: &[f64],
    ) -> Vec<f64> {
        assert!(!c.is_empty(), "Input slice c must not be empty");
        let b = self.shape;
        let seed = rng.next_u64();

        // Use chunks_exact to get evenly sized chunks, and handle the remainder separately
        let chunk_size = 32;
        let chunks = c.par_chunks(chunk_size);
        let num_chunks = chunks.len();

        // Generate one seed per chunk
        let seeds: Vec<u64> = (0..num_chunks)
            .map(|i| seed.wrapping_add(i as u64))
            .collect();

        // Process chunks in parallel
        chunks
            .into_par_iter()
            .zip(seeds.into_par_iter())
            .flat_map(|(chunk, chunk_seed)| {
                let mut rng = R::seed_from_u64(chunk_seed);
                chunk
                    .iter()
                    .map(|&c_val| self.draw_internal(&mut rng, b, c_val))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Draw multiple Polya-Gamma random variates PG(b, c) in parallel.
    ///
    /// # Arguments
    /// * `c` - Tilt parameters (real-valued)
    ///
    /// # Returns
    /// A vector of random variates from PG(b, c)
    ///
    /// # Example
    /// ```rust
    /// # use polya_gamma::PolyaGamma;
    /// let pg = PolyaGamma::new(1.0);
    ///
    /// // Draw 100 samples from PG(1, 0.5)
    /// let samples = pg.draw_vec_par(&[0.5; 100]);
    /// println!("Drew {} samples from PG(1, 0.5)", samples.len());
    /// ```
    #[cfg(feature = "rayon")]
    pub fn draw_vec_par(&self, c: &[f64]) -> Vec<f64> {
        let b = self.shape;
        c.into_par_iter()
            .map_init(rng, |rng, &ci| self.draw_internal(rng, b, ci))
            .collect()
    }
}

impl PolyaGamma {
    /// This is the internal sampling function that handles all the different cases. We don't expose
    /// it directly to make sure that `self.gamma` is properly initialized if b < 1.
    #[inline]
    fn draw_internal<R: Rng + ?Sized>(&self, rng: &mut R, b: f64, c: f64) -> f64 {
        assert!(b > 0.0, "Shape parameter b must be positive");
        if b == 1.0 {
            return self.sample_polya_gamma_devroye(rng, c);
        }
        // For integer b > 2, sum b independent PG(1,c) variates
        let b_floor = b.floor();
        if b == b_floor {
            #[cfg(feature = "rayon")]
            if b >= (rayon::current_num_threads() * 20) as f64 {
                return self.draw_integer_b_par(b as usize, c);
            }
            return self.draw_integer_b(rng, b as usize, c);
        }

        // For non-integer b, use gamma-Poisson mixture
        self.draw_non_integer_b(rng, b, c)
    }

    /// Draw from PG(b, c) when b is an integer > 2
    fn draw_integer_b<R: Rng + ?Sized>(&self, rng: &mut R, b: usize, c: f64) -> f64 {
        (0..b)
            .map(|_| self.sample_polya_gamma_devroye(rng, c))
            .sum()
    }

    #[cfg(feature = "rayon")]
    fn draw_integer_b_par(&self, b: usize, c: f64) -> f64 {
        let threads = rayon::current_num_threads();
        let base = b / threads;
        let rem = b % threads;
        (0..threads)
            .into_par_iter()
            .map_init(rng, |rng, i| {
                let count = base + if i < rem { 1 } else { 0 };
                (0..count)
                    .map(|_| self.sample_polya_gamma_devroye(rng, c))
                    .sum::<f64>()
            })
            .sum()
    }

    /// Draw from PG(b, c) when b is non-integer
    ///
    /// This function handles the case where b is non-integer by using a gamma-Poisson mixture.
    /// We decompose b = n + b′ where n = ⌊b⌋ and 0 < b′ < 1, then:
    /// 1. Sample n independent PG(1, c) variables for the integer part
    /// 2. Sample the fractional part using a gamma-Poisson mixture
    fn draw_non_integer_b<R: Rng + ?Sized>(&self, rng: &mut R, b: f64, c: f64) -> f64 {
        assert!(b > 0.0, "`b` has to be strictly positive");
        assert!(
            b.fract() != 0.0,
            "`b` is an integer – use the integer routine"
        );
        assert!((self.shape - b).abs() <= f64::EPSILON);
        // (c /(2π))² term that appears in every denominator
        let c2 = (c / (2.0 * PI)).powi(2);

        // Accumulator for the infinite sum
        let mut sum = 0.0;

        // Accuracy control
        const TOL: f64 = 1e-6;
        let mut k: usize = 1;

        loop {
            let kf = k as f64 - 0.5; // k – ½
            let den = kf * kf + c2; // denominator
            let g = self.sample_gamma(rng); // Γ(b , 1)

            sum += g / den;

            // Expected magnitude of the next term:  E[G] / den_next = b / den_next
            let next_kf = k as f64 + 0.5; // (k+1) – ½
            let next_den = next_kf * next_kf + c2;

            if b / next_den < TOL {
                break;
            }
            k += 1;
        }

        sum * PI2_SQ_RECIP
    }

    fn init_gamma(&mut self, b: f64) {
        self.gamma = RandGamma::new(b, 1.0).expect("Gamma shape/scale parameters are valid");
    }
}

mod devroye;
pub mod regression;
pub(crate) mod rng;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    /// Empirical mean from `n` draws
    fn empirical_mean(b: f64, c: f64, n: usize, seed: u64) -> f64 {
        let pg = PolyaGamma::new(b);
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| pg.draw(&mut rng, c)).sum::<f64>() / n as f64
    }

    /// Theoretical mean:  E[ω] = b * tanh(c/2) / (2c)  ( = b/4 when c = 0 )
    fn theoretical_mean(b: f64, c: f64) -> f64 {
        if c.abs() < 1e-12 {
            b / 4.0
        } else {
            b * (0.5 * c).tanh() / (2.0 * c)
        }
    }

    #[test]
    fn non_integer_b_mean_matches_theory() {
        let b = 1.7; // truly non-integer
        let n = 25_000; // moderate Monte-Carlo size

        // ---- c = 0 ---------------------------------------------------------
        let emp0 = empirical_mean(b, 0.0, n, 1);
        let th0 = theoretical_mean(b, 0.0);
        assert!(
            (emp0 - th0).abs() / th0 < 0.05,
            "PG({}, 0): empirical {}, theory {}",
            b,
            emp0,
            th0
        );

        // ---- c = 1 ---------------------------------------------------------
        let emp1 = empirical_mean(b, 1.0, n, 2);
        let th1 = theoretical_mean(b, 1.0);
        assert!(
            (emp1 - th1).abs() / th1 < 0.10, // slightly looser tolerance
            "PG({}, 1): empirical {}, theory {}",
            b,
            emp1,
            th1
        );
    }
}

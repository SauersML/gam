//! Bayesian logistic regression with Polya-Gamma data augmentation.
//!
//! This module implements a Gibbs sampler for Bayesian logistic regression using
//! Polya-Gamma data augmentation for efficient sampling from the posterior distribution
//! of the regression coefficients.
//!
//! # Model
//! The model is specified as:
//! - Likelihood: \( y_i \mid \beta \sim \mathrm{Bernoulli}(\sigma(x_i^\top \beta)) \), where \( \sigma \) is the logistic function
//! - Prior: \( \beta \sim \mathcal{N}(0, V_0) \), where \( V_0 \) is a diagonal prior covariance matrix
//! - Data augmentation: \( \omega_i \mid \beta \sim \mathrm{PG}(1, |x_i^\top \beta|) \), where PG is the Polya-Gamma distribution
//!
//! # References
//! - Polson, N. G., Scott, J. G., & Windle, J. (2013). Bayesian inference for logistic models
//!   using Pólya–Gamma latent variables. Journal of the American Statistical Association.

use crate::PolyaGamma;
use mini_mcmc::{
    core::{ChainRunner, init_det},
    distributions::Conditional,
    gibbs::GibbsSampler,
    stats::RunStats,
};
use ndarray::{Array1, Array2, Array3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use statrs::distribution::Normal;
use std::error::Error;

/// A Gibbs sampler for Bayesian logistic regression using Polya-Gamma data augmentation.
///
/// This struct implements a Gibbs sampler for the Bayesian logistic regression model,
/// using Polya-Gamma augmentation for efficient sampling from the posterior distribution.
/// The sampler handles both the regression coefficients (β) and the latent Polya-Gamma
/// variables (ω) in a blocked Gibbs sampling scheme.
///
/// # Type Parameters
/// * `R` - The random number generator type (defaults to `ChaCha8Rng`)
///
/// # Example
/// ```rust
/// # use ndarray::{array, Array2};
/// # use rand::{Rng, SeedableRng};
/// # use rand_chacha::ChaCha8Rng;
/// use polya_gamma::regression::GibbsLogit;
///
/// // Generate synthetic data
/// let n = 100;  // Number of observations
/// let p = 3;      // Number of predictors (including intercept)
/// let mut rng = ChaCha8Rng::seed_from_u64(42);
///
/// // Design matrix with intercept and two predictors
/// let x = {
///     let mut x = Array2::from_shape_fn((n, p), |(i, j)| {
///         if j == 0 { 1.0 }  // Intercept column
///         else { rng.r#gen::<f64>() }  // Predictor columns
///     });
///     x
/// };
///
/// // True coefficients (intercept and two slopes)
/// let true_beta = array![-0.5, 1.0, -0.75];
///
/// // Generate binary responses
/// let probs = x.dot(&true_beta).mapv(|z| 1.0 / (1.0 + (-z).exp()));
/// let y = probs.mapv(|p| if rng.r#gen::<f64>() < p { 1.0 } else { 0.0 });
///
/// // Configure and run the sampler
/// let prior_variance = 100.0;  // N(0, 100) prior on coefficients
/// let n_chains = 4;            // Number of MCMC chains
/// let burnin = 500;            // Burn-in iterations per chain
/// let samples = 1000;          // Posterior samples per chain
///
/// // Create and run the model
/// let model = GibbsLogit::new(x, y, prior_variance, n_chains, 42);
/// let results = model.run(burnin, samples, None).expect("MCMC failed");
///
/// // Print results
/// println!("Posterior means: {:?}", results.posterior_means);
/// println!("Posterior standard deviations: {:?}", results.posterior_sds);
///
/// // Compare with true values
/// println!("True coefficients: {:?}", true_beta);
/// ```
pub struct GibbsLogit<R = ChaCha8Rng>
where
    R: SeedableRng + Rng + Clone + Send + Sync,
{
    x: Array2<f64>,
    y: Array1<f64>,
    prior_prec: Array2<f64>,
    n_chains: usize,
    seed: u64,
    rng: R,
}

/// Results from running the logistic regression MCMC.
///
/// This struct contains the posterior samples and summary statistics from the
/// Gibbs sampler. It provides methods for analyzing and visualizing the results
/// of the Bayesian logistic regression.
pub struct LogisticRegressionResults {
    /// Posterior means of the regression coefficients (including intercept if present)
    ///
    /// The order of coefficients matches the columns of the design matrix `x`.
    /// These are computed by taking the mean across all chains and samples.
    pub posterior_means: Vec<f64>,

    /// Posterior standard deviations of the regression coefficients
    ///
    /// Provides a measure of uncertainty for each coefficient estimate.
    /// These are computed as the standard deviation across all chains and samples.
    pub posterior_sds: Vec<f64>,

    /// All MCMC samples from all chains
    ///
    /// Dimensions: `[n_chains, n_samples, n_parameters + n_observations]`
    /// The first `n_parameters` elements are the regression coefficients,
    /// followed by the latent Polya-Gamma variables.
    pub samples: Array3<f64>,

    /// The true coefficients if provided for comparison
    ///
    /// This is used for benchmarking and validation purposes.
    /// The order should match the columns of the design matrix.
    pub true_coefficients: Option<Vec<f64>>,

    /// Runtime statistics and diagnostics
    ///
    /// Includes information about sampling efficiency, acceptance rates,
    /// and convergence diagnostics. Useful for assessing MCMC performance.
    pub run_stats: RunStats,
}

impl GibbsLogit<ChaCha8Rng> {
    /// Create a new logistic regression Gibbs sampler with default RNG.
    ///
    /// # Arguments
    /// * `x` - Design matrix of shape `(n_observations, n_predictors)`.
    ///   For models with an intercept, include a column of ones.
    /// * `y` - Binary response vector of shape `(n_observations,)` containing 0s and 1s.
    /// * `prior_variance` - Variance of the normal prior on coefficients (N(0, prior_variance * I)).
    ///   Larger values indicate more diffuse (less informative) priors.
    ///   A common choice is 100 for standardized predictors.
    /// * `n_chains` - Number of independent MCMC chains to run (≥ 1).
    ///   Multiple chains help assess convergence.
    /// * `seed` - Random seed for reproducibility. Using the same seed with the same data will
    ///   produce identical results.
    ///
    /// # Panics
    /// - If `x` and `y` have incompatible dimensions
    /// - If `prior_variance` is not positive
    /// - If `n_chains` is zero
    ///
    /// # Example
    /// ```
    /// use ndarray::array;
    /// use polya_gamma::regression::GibbsLogit;
    ///
    /// let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];  // Include intercept
    /// let y = array![0.0, 0.0, 1.0];  // Binary responses
    /// let model = GibbsLogit::new(x, y, 100.0, 4, 42);
    /// ```
    pub fn new(
        x: Array2<f64>,
        y: Array1<f64>,
        prior_variance: f64,
        n_chains: usize,
        seed: u64,
    ) -> Self {
        let p = x.ncols();
        let prior_prec = Array2::eye(p) * (1.0 / prior_variance);

        Self {
            x,
            y,
            prior_prec,
            n_chains,
            seed,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

impl<R: SeedableRng + Rng + Clone + Send + Sync> GibbsLogit<R> {
    /// Create a new logistic regression Gibbs sampler with a custom RNG.
    ///
    /// # Arguments
    /// * `rng` - A random number generator implementing `Rng + SeedableRng + Clone + Send + Sync`
    /// * `x` - Design matrix of shape `(n_observations, n_predictors)`
    /// * `y` - Binary response vector of shape `(n_observations,)` containing 0s and 1s
    /// * `prior_variance` - Variance of the normal prior on coefficients (N(0, prior_variance * I))
    /// * `n_chains` - Number of independent MCMC chains to run (≥ 1)
    /// * `seed` - Random seed for reproducibility (provided to `mini-mcmc`)
    ///
    /// # Panics
    /// - If `x` and `y` have incompatible dimensions
    /// - If `prior_variance` is not positive
    /// - If `n_chains` is zero
    ///
    /// # Example
    /// ```
    /// use ndarray::array;
    /// use rand_chacha::ChaCha8Rng;
    /// use rand::SeedableRng;
    /// use polya_gamma::regression::GibbsLogit;
    ///
    /// let rng = ChaCha8Rng::seed_from_u64(42);
    /// let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
    /// let y = array![0.0, 0.0, 1.0];
    /// let model = GibbsLogit::from_rng(rng, x, y, 100.0, 4, 42);
    /// ```
    pub fn from_rng(
        rng: R,
        x: Array2<f64>,
        y: Array1<f64>,
        prior_variance: f64,
        n_chains: usize,
        seed: u64,
    ) -> Self {
        let p = x.ncols();
        let prior_prec = Array2::eye(p) * (1.0 / prior_variance);

        Self {
            x,
            y,
            prior_prec,
            n_chains,
            seed,
            rng,
        }
    }
}

impl<R: SeedableRng + Rng + Clone + Send + Sync> GibbsLogit<R> {
    /// Run the MCMC sampler to obtain posterior samples.
    ///
    /// This method performs the Gibbs sampling algorithm for the specified number of
    /// iterations, discarding the burn-in samples, and returns the posterior samples
    /// and summary statistics.
    ///
    /// # Arguments
    /// * `burn_in` - Number of burn-in iterations per chain. These samples are discarded
    ///   to allow the chain to reach its stationary distribution. A common
    ///   choice is 500-2000 iterations, but this should be adjusted based on
    ///   convergence diagnostics.
    /// * `samples` - Number of posterior samples to keep per chain after burn-in.
    /// * `true_coefficients` - Optional true coefficients for benchmarking and validation.
    ///   If provided, these will be stored in the results for comparison
    ///   with the posterior estimates.
    ///
    /// # Returns
    /// A `Result` containing either:
    /// - `Ok(LogisticRegressionResults)` with the sampling results, or
    /// - `Err(Box<dyn Error>)` if an error occurs during sampling
    ///
    /// # Panics
    /// - If `burn_in` or `samples` is zero
    /// - If memory allocation for samples fails
    ///
    /// # Example
    /// ```
    /// # use ndarray::array;
    /// # use polya_gamma::regression::GibbsLogit;
    /// # let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
    /// # let y = array![0.0, 0.0, 1.0];
    /// # let model = GibbsLogit::new(x, y, 100.0, 1, 42);
    /// // After creating the model...
    /// let burn_in = 500;
    /// let samples = 1000;
    /// let true_coeffs = Some(vec![0.5, -1.0]);
    ///
    /// match model.run(burn_in, samples, true_coeffs) {
    ///     Ok(results) => {
    ///         println!("Sampling completed successfully");
    ///         results.summary();
    ///     }
    ///     Err(e) => eprintln!("MCMC failed: {}", e),
    /// }
    /// ```
    pub fn run(
        self,
        burn_in: usize,
        samples: usize,
        true_coefficients: Option<Vec<f64>>,
    ) -> Result<LogisticRegressionResults, Box<dyn Error>> {
        let n = self.x.nrows();
        let p = self.x.ncols();
        let dim = p + n; // Parameters + latent variables

        // Initialize RNG

        // Initialize the conditional sampler
        let cond = LogitConditional::new(
            self.x.clone(),
            self.y.clone(),
            self.prior_prec.clone(),
            self.rng,
        );

        // Initialize chains: β=0, ω=1
        let mut init: Vec<Vec<f64>> = init_det(self.n_chains, dim);
        for state in &mut init {
            for w in &mut state[p..] {
                *w = 1.0; // Initialize ω=1
            }
        }

        // Create and run the Gibbs sampler
        let mut gibbs = GibbsSampler::new(cond, init).set_seed(self.seed);
        let (all_samples, run_stats) = gibbs.run_progress(samples, burn_in)?;

        // Process results
        let pooled = all_samples.to_shape((self.n_chains * samples, dim))?;

        // Compute posterior summaries for coefficients
        let posterior_means: Vec<f64> = (0..p).map(|j| pooled.column(j).mean().unwrap()).collect();

        let posterior_sds: Vec<f64> = (0..p)
            .map(|j| pooled.column(j).std(1.0)) // 1.0 for sample standard deviation
            .collect();

        Ok(LogisticRegressionResults {
            posterior_means,
            posterior_sds,
            samples: all_samples,
            true_coefficients,
            run_stats,
        })
    }
}

impl LogisticRegressionResults {
    /// Print a summary of the MCMC results
    pub fn summary(&self) {
        println!(
            "{:<10} {:<15} {:<15} {:<15}",
            "Parameter", "Mean", "Std. Dev.", "True Value"
        );
        println!("{}", "-".repeat(55));

        for (i, (mean, sd)) in self
            .posterior_means
            .iter()
            .zip(&self.posterior_sds)
            .enumerate()
        {
            let true_val = self
                .true_coefficients
                .as_ref()
                .map_or("N/A".to_string(), |v| format!("{:.4}", v[i]));

            println!(
                "{:<10} {:<15.4} {:<15.4} {:<15}",
                format!("β{}", i),
                mean,
                sd,
                true_val
            );
        }
    }

    /// Get the posterior samples for a specific coefficient
    pub fn get_posterior_samples(&self, param_idx: usize) -> Option<Vec<f64>> {
        if param_idx >= self.samples.shape()[2] {
            return None;
        }

        let n_chains = self.samples.shape()[0];
        let n_samples = self.samples.shape()[1];
        let mut samples = Vec::with_capacity(n_chains * n_samples);

        for chain in 0..n_chains {
            for sample in 0..n_samples {
                samples.push(self.samples[[chain, sample, param_idx]]);
            }
        }

        Some(samples)
    }
}

/// Implements the conditional distributions for the Gibbs sampler in Bayesian logistic regression.
///
/// The sampler alternates between updating the regression coefficients β and the
/// latent Polya-Gamma variables ω in a blocked Gibbs sampling scheme.
///
/// # Model
/// The full conditional distributions are:
/// \[
///   \omega_i \mid \beta \sim \mathrm{PG}(1, |x_i^\top \beta|) \\
///   \beta \mid \omega \sim \mathcal{N}(m, V)
/// \]
/// where
/// \[
///   V = \left(X^\top \Omega X + V_0^{-1}\right)^{-1} \quad
///   m = V \left(X^\top \kappa + V_0^{-1} \mu_0\right)
/// \]
/// \[
///   \Omega = \mathrm{diag}(\omega_1, \ldots, \omega_n), \quad
///   \kappa = (y_1 - \tfrac{1}{2}, \ldots, y_n - \tfrac{1}{2})
/// \]
///
/// # Type Parameters
/// * `R` - The random number generator type
#[derive(Clone)]
struct LogitConditional<R>
where
    R: SeedableRng + Rng + Clone + Send + Sync,
{
    /// Design matrix (n_observations × n_predictors)
    x: Array2<f64>,

    /// Binary response vector (n_observations,)
    y: Array1<f64>,
    /// Prior precision V0^{-1}, shape (p, p)
    prior_prec: Array2<f64>,
    /// Reusable Polya–Gamma sampler
    pg: PolyaGamma,
    rng: R,
}

impl<R> LogitConditional<R>
where
    R: SeedableRng + Rng + Clone + Send + Sync,
{
    pub fn new(x: Array2<f64>, y: Array1<f64>, prior_prec: Array2<f64>, rng: R) -> Self {
        Self {
            x,
            y,
            prior_prec,
            pg: PolyaGamma::new(1.0),
            rng,
        }
    }
}

impl<R> Conditional<f64> for LogitConditional<R>
where
    R: SeedableRng + Rng + Clone + Send + Sync,
{
    /// Sample from the full conditional distribution for either $\beta$ or $\omega$.
    ///
    /// This method implements the core Gibbs sampling updates:
    /// - For $\beta$: Samples from a multivariate normal distribution
    /// - For $\omega$: Samples from a Polya-Gamma distribution
    ///
    /// # Arguments
    /// * `i` - Index of the parameter to sample:
    ///   - If `i < p` (where p is number of predictors), samples $\beta_i$
    ///   - If `i >= p`, samples $\omega_{i-p}$ (the latent variable for observation i-p)
    /// * `given` - Current state of all parameters $[\beta, \omega]$
    ///
    /// # Returns
    /// A new sample from the appropriate full conditional distribution
    ///
    /// # Panics
    /// - If `i` is out of bounds (≥ p + n)
    fn sample(&mut self, i: usize, given: &[f64]) -> f64 {
        let n = self.x.nrows();
        let p = self.x.ncols();

        if i < p {
            // Update beta_i
            let col_i = self.x.column(i);
            let prior_ii = self.prior_prec[(i, i)];
            let mut precision = prior_ii;
            let mut precision_mean = 0.0;

            // Loop over observations
            for row_idx in 0..n {
                let xi = col_i[row_idx];
                let wi = given[p + row_idx]; // ω_i
                precision += wi * xi * xi;

                // Compute contribution of all other betas at this row:
                let mut dot_minus_i = 0.0;
                for (k, bj) in given.iter().enumerate().take(p) {
                    if k != i {
                        dot_minus_i += self.x[(row_idx, k)] * bj;
                    }
                }

                let yi = self.y[row_idx];
                let resid = (yi - 0.5) - wi * dot_minus_i;
                precision_mean += xi * resid;
            }

            let var_i = 1.0 / precision;
            let mean_i = precision_mean * var_i;

            // Draw N(mean_i, var_i)
            let eps: f64 = self.rng.sample(Normal::standard());
            mean_i + eps * var_i.sqrt()
        } else {
            // Update omega
            let row = self.x.row(i - p);
            let xb: f64 = row
                .iter()
                .zip(&given[0..p])
                .map(|(xij, bj)| (xij * bj))
                .sum::<f64>()
                .abs();
            // Draw PG(1, xb)
            self.pg.draw(&mut self.rng, xb)
        }
    }
}

//! Bayesian negative binomial regression with Polya-Gamma data augmentation.
//!
//! This module implements a Gibbs sampler for Bayesian negative binomial regression using
//! Polya-Gamma augmentation for efficient sampling from the posterior distribution
//! of the regression coefficients and dispersion parameter.
//!
//! # Model
//! The negative binomial regression model is specified as:
//! - yᵢ | β, r ∼ NegativeBinomial(mean=exp(xᵢᵀβ), r)
//! - ωᵢ | β, r ∼ PG(yᵢ + r, xᵢᵀβ - log(r))
//! - β ∼ N(0, V₀)
//! - r ∼ Gamma(shape=a, scale=b)
//!
//! where PG is the Polya-Gamma distribution.

use crate::PolyaGamma;
use mini_mcmc::core::{ChainRunner, init_det};
use mini_mcmc::distributions::Conditional;
use mini_mcmc::gibbs::GibbsSampler;
use mini_mcmc::stats::RunStats;
use ndarray::{Array1, Array2, Array3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use statrs::distribution::Normal;
use statrs::function::gamma::ln_gamma;
use std::error::Error;

/// A Gibbs sampler for Bayesian negative binomial regression using Polya-Gamma data augmentation.
///
/// This struct implements a Gibbs sampler for the Bayesian negative binomial regression model,
/// using Polya-Gamma augmentation for efficient sampling from the posterior distribution.
/// The sampler handles both the regression coefficients (β) and the dispersion parameter (r).
///
/// # Type Parameters
/// * `R` - The random number generator type (defaults to `ChaCha8Rng`)
///
/// # Example
/// ```no_run
/// # use ndarray::{array, Array2};
/// # use rand::{Rng, SeedableRng};
/// # use rand_chacha::ChaCha8Rng;
/// # use polya_gamma::regression::GibbsNegativeBinomial;
/// # use statrs::distribution::{Gamma, Poisson};
/// # use rand::distributions::Distribution;
///
/// // Generate synthetic data
/// let n = 100;  // Number of observations
/// let p = 3;     // Number of predictors (including intercept)
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
/// let log_means = x.dot(&true_beta);
///
/// // True dispersion parameter
/// let true_r = 5.0;
///
/// // Generate negative binomial responses
/// let p_vec = log_means.mapv(|mu| true_r / (true_r + mu.exp()));
/// let y = p_vec.mapv(|p| {
///     let gamma = Gamma::new(true_r, 1.0).unwrap();
///     let gamma_sample: f64 = rng.sample(gamma);
///     let poisson = Poisson::new(gamma_sample * (1.0/p - 1.0)).unwrap();
///     rng.sample(poisson)
/// });
///
/// // Configure and run the sampler
/// let prior_variance = 100.0;  // N(0, 100) prior on coefficients
/// let prior_shape = 1.0;       // Gamma shape for r (weakly informative)
/// let prior_scale = 1.0;       // Gamma scale for r (weakly informative)
/// let n_chains = 4;            // Number of MCMC chains
/// let burn_in = 500;           // Burn-in iterations per chain
/// let samples = 1000;          // Posterior samples per chain
///
/// // Create and run the model
/// let model = GibbsNegativeBinomial::new(
///     x,
///     y,
///     prior_variance,
///     prior_shape,
///     prior_scale,
///     n_chains,
///     42
/// );
///
/// let results = model.run(
///     burn_in,
///     samples,
///     Some(true_beta.to_vec()),
///     Some(true_r)
/// ).expect("MCMC failed");
///
/// // Print results
/// println!("Posterior means: {:?}", results.posterior_means);
/// println!("Posterior mean of r: {}", results.posterior_mean_r);
///
/// // Compare with true values
/// println!("True coefficients: {:?}", true_beta);
/// println!("True r: {}", true_r);
/// ```
pub struct GibbsNegativeBinomial<R = ChaCha8Rng>
where
    R: SeedableRng + Rng + Clone + Send + Sync,
{
    /// Design matrix (n_obs × p)
    x: Array2<f64>,
    /// Response vector (n_obs,)
    y: Array1<f64>,
    /// Prior precision matrix for beta (p × p)
    prior_prec: Array2<f64>,
    /// Prior shape for r (dispersion parameter)
    prior_shape: f64,
    /// Prior scale for r (dispersion parameter)
    prior_scale: f64,
    /// Number of MCMC chains
    n_chains: usize,
    /// Random seed for reproducibility
    seed: u64,
    /// RNG instance
    rng: R,
}

/// Posterior samples and summary statistics from the negative binomial regression MCMC.
pub struct NegativeBinomialResults {
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

    /// Posterior mean of the dispersion parameter r
    ///
    /// The negative binomial dispersion parameter, where smaller values indicate
    /// more overdispersion relative to a Poisson distribution.
    pub posterior_mean_r: f64,

    /// All MCMC samples from all chains
    ///
    /// Dimensions: `[n_chains, n_samples, n_parameters + n_observations + 1]`
    /// The first `n_parameters` elements are the regression coefficients,
    /// followed by the latent Polya-Gamma variables, and finally the dispersion parameter r.
    pub samples: Array3<f64>,

    /// The true coefficients if provided for comparison
    ///
    /// This is used for benchmarking and validation purposes.
    /// The order should match the columns of the design matrix.
    pub true_coefficients: Option<Vec<f64>>,

    /// The true dispersion parameter if provided for comparison
    ///
    /// This is used for benchmarking and validation purposes.
    pub true_r: Option<f64>,

    /// Runtime statistics and diagnostics
    ///
    /// Includes information about sampling efficiency, acceptance rates,
    /// and convergence diagnostics. Useful for assessing MCMC performance.
    pub run_stats: RunStats,
}

impl GibbsNegativeBinomial<ChaCha8Rng> {
    /// Create a new negative binomial regression Gibbs sampler with default RNG.
    ///
    /// # Arguments
    /// * `x` - Design matrix of shape `(n_observations, n_predictors)`.
    ///   For models with an intercept, include a column of ones.
    /// * `y` - Count response vector of shape `(n_observations,)` containing non-negative integers.
    /// * `prior_variance` - Variance of the normal prior on coefficients (N(0, prior_variance * I)).
    ///   Larger values indicate more diffuse (less informative) priors.
    ///   A common choice is 100 for standardized predictors.
    /// * `prior_shape` - Shape parameter for the gamma prior on the dispersion parameter r.
    ///   A smaller value (e.g., 0.01) gives a more diffuse prior, while larger values
    ///   (e.g., 1.0) give more informative priors.
    /// * `prior_scale` - Scale parameter for the gamma prior on the dispersion parameter r.
    ///   The prior mean is `prior_shape * prior_scale`.
    /// * `n_chains` - Number of independent MCMC chains to run (≥ 1).
    ///   Multiple chains help assess convergence.
    /// * `seed` - Random seed for reproducibility. Using the same seed with the same data will
    ///   produce identical results.
    ///
    /// # Panics
    /// - If `x` and `y` have incompatible dimensions
    /// - If `prior_variance`, `prior_shape`, or `prior_scale` are not positive
    /// - If `n_chains` is zero
    ///
    /// # Example
    /// ```
    /// # use ndarray::array;
    /// # use polya_gamma::regression::GibbsNegativeBinomial;
    /// let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];  // Include intercept
    /// let y = array![5.0, 3.0, 8.0];  // Count responses
    /// let model = GibbsNegativeBinomial::new(
    ///     x, y, 100.0, 1.0, 1.0, 4, 42
    /// );
    /// ```
    pub fn new(
        x: Array2<f64>,
        y: Array1<f64>,
        prior_variance: f64,
        prior_shape: f64,
        prior_scale: f64,
        n_chains: usize,
        seed: u64,
    ) -> Self {
        let p = x.ncols();
        let prior_prec = Array2::eye(p) * (1.0 / prior_variance);
        let rng = ChaCha8Rng::seed_from_u64(seed);

        Self {
            x,
            y,
            prior_prec,
            prior_shape,
            prior_scale,
            n_chains,
            seed,
            rng,
        }
    }
}

impl<R: SeedableRng + Rng + Clone + Send + Sync> GibbsNegativeBinomial<R> {
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
    ///   More samples will reduce Monte Carlo error but increase memory usage.
    /// * `true_coefficients` - Optional true coefficients for benchmarking and validation.
    ///   If provided, these will be stored in the results for comparison
    ///   with the posterior estimates.
    /// * `true_r` - Optional true dispersion parameter for benchmarking and validation.
    ///   If provided, this will be stored in the results for comparison.
    ///
    /// # Returns
    /// A `Result` containing either:
    /// - `Ok(NegativeBinomialResults)` with the sampling results, or
    /// - `Err(Box<dyn Error>)` if an error occurs during sampling
    ///
    /// # Panics
    /// - If `burn_in` or `samples` is zero
    /// - If memory allocation for samples fails
    ///
    /// # Example
    /// ```
    /// # use ndarray::array;
    /// # use polya_gamma::regression::GibbsNegativeBinomial;
    /// # let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
    /// # let y = array![5.0, 3.0, 8.0];
    /// # let model = GibbsNegativeBinomial::new(x, y, 100.0, 1.0, 1.0, 1, 42);
    /// // After creating the model...
    /// let burn_in = 500;
    /// let samples = 1000;
    /// let true_coeffs = Some(vec![0.5, -1.0]);
    /// let true_r = Some(5.0);
    ///
    /// match model.run(burn_in, samples, true_coeffs, true_r) {
    ///     Ok(results) => {
    ///         println!("Sampling completed successfully");
    ///         results.summary();
    ///     }
    ///     Err(e) => eprintln!("MCMC failed: {}", e),
    /// }
    /// ````
    pub fn run(
        self,
        burnin: usize,
        samples: usize,
        true_coefficients: Option<Vec<f64>>,
        true_r: Option<f64>,
    ) -> Result<NegativeBinomialResults, Box<dyn Error>> {
        let n = self.x.nrows();
        let p = self.x.ncols();
        let dim = p + n + 1; // Parameters + latent variables + r

        // Initialize the conditional sampler
        let cond = NegativeBinomialConditional::new(
            self.x.clone(),
            self.y.clone(),
            self.prior_prec.clone(),
            self.prior_shape,
            self.prior_scale,
            self.rng,
        );

        // Initialize chains: β=0, ω=1, r=1
        let mut init: Vec<Vec<f64>> = init_det(self.n_chains, dim);
        for state in &mut init {
            // Initialize ω=1 for each observation
            for w in &mut state[p..p + n] {
                *w = 1.0;
            }
            // Initialize r=1
            state[p + n] = 1.0;
        }

        // Create and run the Gibbs sampler
        let mut gibbs = GibbsSampler::new(cond, init).set_seed(self.seed);
        let (all_samples, run_stats) = gibbs.run_progress(samples, burnin)?;

        // Process results
        // let kept: ArrayView3<'_, f64> = all_samples.slice(s![.., burnin.., ..]);
        let pooled = all_samples.to_shape((self.n_chains * samples, dim))?;

        // Compute posterior summaries for coefficients
        let posterior_means: Vec<f64> = (0..p).map(|j| pooled.column(j).mean().unwrap()).collect();
        let posterior_sds: Vec<f64> = (0..p)
            .map(|j| pooled.column(j).std(1.0)) // 1.0 for sample standard deviation
            .collect();

        // Compute posterior mean for r
        let posterior_mean_r = pooled.column(p + n).mean().unwrap();

        Ok(NegativeBinomialResults {
            posterior_means,
            posterior_sds,
            posterior_mean_r,
            samples: all_samples,
            true_coefficients,
            true_r,
            run_stats,
        })
    }
}

impl NegativeBinomialResults {
    /// Print a summary of the MCMC results
    pub fn summary(&self) {
        println!("Negative Binomial Regression Results");
        println!("================================");
        println!(
            "{:<10} {:<15} {:<15} {:<15}",
            "Parameter", "Mean", "Std. Dev.", "True Value"
        );
        println!("{}", "-".repeat(55));

        // Print coefficients
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

        // Print dispersion parameter
        let true_r = self
            .true_r
            .map_or("N/A".to_string(), |r| format!("{:.4}", r));
        println!(
            "{:<10} {:<15.4} {:<15} {:<15}",
            "r", self.posterior_mean_r, "-", true_r
        );
    }

    /// Get the posterior samples for a specific regression coefficient.
    ///
    /// This method extracts the MCMC samples for a single regression coefficient
    /// across all chains and iterations. The samples are returned as a flattened vector.
    ///
    /// # Arguments
    /// * `param_index` - Zero-based index of the parameter. For a model with an intercept,
    ///   `param_index=0` corresponds to the intercept, `param_index=1` to the first predictor,
    ///   and so on.
    ///
    /// # Returns
    /// A `Vec<f64>` containing all posterior samples for the specified parameter.
    /// The length will be `n_chains * samples`.
    ///
    /// # Panics
    /// - If `param_index` is out of bounds for the number of parameters
    ///
    /// # Example
    /// ```no_run
    /// # use ndarray::array;
    /// # use polya_gamma::regression::GibbsNegativeBinomial;
    /// # let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
    /// # let y = array![5.0, 3.0, 8.0];
    /// # let model = GibbsNegativeBinomial::new(x, y, 100.0, 1.0, 1.0, 2, 42);
    /// # let results = model.run(100, 100, None, None).unwrap();
    /// // Get samples for the first predictor (index 1, assuming index 0 is intercept)
    /// let beta_samples = results.get_posterior_samples(1).unwrap();
    /// println!("Number of samples: {}", beta_samples.len());
    /// println!("Mean: {}", beta_samples.iter().sum::<f64>() / beta_samples.len() as f64);
    /// ```    
    pub fn get_posterior_samples(&self, param_idx: usize) -> Option<Vec<f64>> {
        let n_params = self.samples.shape()[2];
        if param_idx >= n_params {
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

/// A conditional sampler for the Bayesian negative binomial regression model.
///
/// This struct implements the Gibbs sampling updates for the model parameters:
/// - β: Regression coefficients with N(0, V₀) prior
/// - ω: Polya-Gamma auxiliary variables
/// - r: Dispersion parameter with Gamma(a, b) prior
///
/// The model is specified as:
/// - yᵢ | β, r ∼ NegativeBinomial(mean=exp(xᵢᵀβ), r)
/// - ωᵢ | β, r ∼ PG(yᵢ + r, xᵢᵀβ - log(r))
/// - β ∼ N(0, V₀)
/// - r ∼ Gamma(shape=a, scale=b)
///
/// where PG is the Polya-Gamma distribution.
///
/// # Type Parameters
/// * `R` - The random number generator type
#[derive(Clone)]
struct NegativeBinomialConditional<R>
where
    R: SeedableRng + Rng + Clone + Send + Sync,
{
    /// Design matrix of shape (n_observations, n_predictors)
    x: Array2<f64>,

    /// Response vector of non-negative integers with length n_observations
    y: Array1<f64>,

    /// Prior precision matrix V₀⁻¹ of shape (n_predictors, n_predictors)
    prior_prec: Array2<f64>,

    /// Shape parameter for the Gamma prior on r
    prior_shape: f64,

    /// Scale parameter for the Gamma prior on r
    prior_scale: f64,

    /// Reusable Polya-Gamma sampler for efficient sampling of ω variables
    pg: PolyaGamma,

    /// Random number generator instance
    rng: R,
}

impl<R> NegativeBinomialConditional<R>
where
    R: SeedableRng + Rng + Clone + Send + Sync,
{
    /// Creates a new conditional sampler for the negative binomial model.
    ///
    /// # Arguments
    /// * `x` - Design matrix of shape (n_observations, n_predictors)
    /// * `y` - Response vector of non-negative integers with length n_observations
    /// * `prior_prec` - Prior precision matrix V₀⁻¹ of shape (n_predictors, n_predictors)
    /// * `prior_shape` - Shape parameter for the Gamma prior on r
    /// * `prior_scale` - Scale parameter for the Gamma prior on r
    /// * `rng` - Random number generator instance
    ///
    /// # Panics
    /// - If dimensions of `x` and `y` are incompatible
    /// - If `prior_shape` or `prior_scale` are not positive
    pub fn new(
        x: Array2<f64>,
        y: Array1<f64>,
        prior_prec: Array2<f64>,
        prior_shape: f64,
        prior_scale: f64,
        rng: R,
    ) -> Self {
        assert_eq!(
            x.nrows(),
            y.len(),
            "Number of rows in x must match length of y"
        );
        assert!(prior_shape > 0.0, "prior_shape must be positive");
        assert!(prior_scale > 0.0, "prior_scale must be positive");

        Self {
            x,
            y,
            prior_prec,
            prior_shape,
            prior_scale,
            pg: PolyaGamma::new(1.0),
            rng,
        }
    }
}

impl<R> Conditional<f64> for NegativeBinomialConditional<R>
where
    R: SeedableRng + Rng + Clone + Send + Sync,
{
    /// Samples a single parameter from its full conditional distribution.
    ///
    /// This method implements the Gibbs sampling updates for each parameter:
    /// - For i in 0..p: Updates the i-th regression coefficient βᵢ
    /// - For i in p..p+n: Updates the (i-p)-th Polya-Gamma variable ω_{i-p}
    /// - For i = p+n: Updates the dispersion parameter r using a Metropolis-Hastings step
    ///
    /// # Arguments
    /// * `i` - Index of the parameter to update (0-based)
    /// * `given` - Current values of all parameters in the order [β, ω, r]
    ///
    /// # Returns
    /// A new sample for the specified parameter from its full conditional distribution
    fn sample(&mut self, i: usize, given: &[f64]) -> f64 {
        let n = self.x.nrows();
        let p = self.x.ncols();
        let r = given[p + n]; // Current value of r

        if i < p {
            // ---- Update βᵢ ----
            let col_i = self.x.column(i);
            let prior_ii = self.prior_prec[(i, i)];
            let mut precision = prior_ii; // precision
            let mut precision_mean = 0.0; // precision × mean

            // Loop over observations
            for row_idx in 0..n {
                let xi = col_i[row_idx];
                let wi = given[p + row_idx]; // ω_i
                precision += wi * xi * xi;

                // Compute contribution of all other betas at this row
                let mut dot_minus_i = 0.0;
                for (k, bj) in given.iter().enumerate().take(p) {
                    if k != i {
                        dot_minus_i += self.x[(row_idx, k)] * bj;
                    }
                }

                let yi = self.y[row_idx];
                let kappa = (yi - r) / 2.0;
                let resid = kappa - wi * (dot_minus_i - r.ln());
                precision_mean += xi * resid;
            }

            let var_i = 1.0 / precision;
            let mean_i = precision_mean * var_i;

            // Draw N(mean_i, var_i)
            let normal = Normal::standard();
            let eps: f64 = self.rng.sample(normal);
            mean_i + eps * var_i.sqrt()
        } else if i < p + n {
            // ---- Update ωᵢ (Polya-Gamma) ----
            let obs_idx = i - p;
            let row = self.x.row(obs_idx);
            let xb: f64 = row
                .iter()
                .zip(&given[0..p])
                .map(|(xij, bj)| xij * bj)
                .sum::<f64>();
            let psi = xb - r.ln();
            let yi = self.y[obs_idx];

            // Draw ω_i ~ PG(y_i + r, |ψ_i|)
            self.pg.set_shape(yi + r);
            self.pg.draw(&mut self.rng, psi.abs())
        } else {
            // ---- Update r (dispersion parameter) ----
            // We use a Metropolis-Hastings step with a log-normal proposal
            let current_r = r;

            // Log-posterior for r: = Gamma(a,b) prior + NB marginal likelihood
            let log_posterior = |r_val: f64| -> f64 {
                if r_val <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                // 1) Gamma(a, b) prior
                let mut lp = (self.prior_shape - 1.0) * r_val.ln() - self.prior_scale * r_val;

                // 2) NB(μ_i=exp(xβ), r) marginal log-likelihood
                //    ∑_i [ ln Γ(y_i + r) – ln Γ(r) + r ln r – (y_i + r) ln(r + μ_i) ]
                for (idx, &yi) in self.y.iter().enumerate() {
                    let row = self.x.row(idx);
                    let xb = row
                        .iter()
                        .zip(&given[..p])
                        .map(|(xij, bj)| xij * bj)
                        .sum::<f64>();
                    let mu = xb.exp();
                    lp += ln_gamma(yi + r_val) - ln_gamma(r_val) + r_val * r_val.ln()
                        - (yi + r_val) * (r_val + mu).ln();
                }
                lp
            };

            // Random walk proposal (log-normal)
            let proposal_sd = 1.0; // Tune this for better mixing
            let proposal_ln_r =
                current_r.ln() + self.rng.sample(Normal::new(0.0, proposal_sd).unwrap());
            let proposal_r = proposal_ln_r.exp();

            // Log acceptance ratio
            let log_alpha = log_posterior(proposal_r) - log_posterior(current_r) + proposal_ln_r
                - current_r.ln(); // Jacobian for log transform

            // Accept or reject
            if log_alpha >= 0.0 || self.rng.r#gen::<f64>().ln() < log_alpha {
                proposal_r
            } else {
                current_r
            }
        }
    }
}

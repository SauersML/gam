//! Example of Bayesian logistic regression using the Polya-Gamma data augmentation scheme.
//!
//! This example demonstrates how to perform Bayesian logistic regression using the Polya-Gamma
//! data augmentation approach. The key advantage of this method is that it allows for efficient
//! Gibbs sampling by transforming the logistic regression problem into a series of conjugate
//! normal linear regressions.
//!
//! The example:
//! 1. Generates synthetic data from a logistic regression model
//! 2. Fits the model using Gibbs sampling with Polya-Gamma augmentation
//! 3. Reports the posterior means and credible intervals for the parameters
//! 4. Compares the estimates to the true parameter values used to generate the data

use ndarray::Array2;
use polya_gamma::regression::GibbsLogit;
use rand::{Rng, SeedableRng};
use statrs::distribution::Normal;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let n = 10_000;
    let p = 3;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let x = Array2::from_shape_fn((n, p), |_| rng.sample(Normal::standard()));
    let true_beta = ndarray::array![0.5, -1.0, 2.0];
    let probs = x.dot(&true_beta).mapv(|z| 1.0 / (1.0 + (-z).exp()));
    let y = probs.mapv(|p| if rng.gen_bool(p) { 1.0 } else { 0.0 });

    // Configure and run the sampler
    let prior_variance = 100.0; // N(0, 100) prior
    let n_chains = 4;
    let burnin = 500;
    let samples = 1000;
    let seed = 42;

    let model = GibbsLogit::new(x, y, prior_variance, n_chains, seed);
    let results = model.run(burnin, samples, Some(true_beta.to_vec()))?;

    // Print summary
    results.summary();
    println!("\nDetailed results:");
    println!("----------------");
    println!("{}", results.run_stats);
    Ok(())
}

//! Example of Bayesian negative binomial regression using the Polya-Gamma data augmentation scheme.
//!
//! This example demonstrates how to perform Bayesian negative binomial regression using the Polya-Gamma
//! data augmentation approach. The negative binomial distribution is useful for modeling count data
//! with overdispersion (variance > mean). This implementation uses a Gibbs sampling approach where
//! the negative binomial likelihood is represented as a gamma mixture of Poisson distributions,
//! and the Polya-Gamma augmentation is used for the gamma scale parameters.
//!
//! The example:
//! 1. Generates synthetic count data from a negative binomial regression model
//! 2. Fits the model using Gibbs sampling with Polya-Gamma augmentation
//! 3. Estimates both the regression coefficients and the dispersion parameter
//! 4. Reports posterior means, credible intervals, and MCMC diagnostics
//! 5. Compares estimates to the true parameter values used to generate the data

use ndarray::{Array1, Array2, array};
use polya_gamma::regression::GibbsNegativeBinomial;
use rand::{SeedableRng, prelude::Distribution};
use rand_chacha::ChaCha8Rng;
use statrs::distribution::{NegativeBinomial as StatrsNegativeBinomial, Normal};

fn main() {
    // Set random seed for reproducibility
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Generate synthetic data
    let n = 1000; // number of observations
    let p = 3; // number of predictors (including intercept)

    // True parameters
    let true_beta = array![0.5, -1.0, 0.3];
    let true_r = 5.0; // dispersion parameter

    // Generate design matrix X with first column as 1 (intercept)
    let normal = Normal::new(0.0, 1.0).unwrap();
    let x_data: Vec<f64> = (0..n)
        .flat_map(|_| {
            let mut m = vec![1.0];
            m.extend((0..(p - 1)).map(|_| normal.sample(&mut rng)));
            m
        })
        .collect();
    assert!(x_data.len() == n * p);

    let x = Array2::from_shape_vec((n, p), x_data).unwrap();

    // Generate response y from negative binomial distribution
    let mut y_data = Vec::with_capacity(n);
    for i in 0..n {
        let eta = x.row(i).dot(&true_beta);
        let mu = eta.exp();
        let p_nb = mu / (true_r + mu); // probability of success for statrs
        let nb = StatrsNegativeBinomial::new(true_r, p_nb).unwrap();
        y_data.push(nb.sample(&mut rng) as f64);
    }
    let y = Array1::from_vec(y_data);

    // Print some summary statistics
    println!(
        "Generated data with {} observations and {} predictors",
        n, p
    );
    println!("True beta: {:?}", true_beta);
    println!("True r: {}", true_r);
    println!("Mean of y: {:.2}", y.mean().unwrap());
    println!("Variance of y: {:.2}", y.var(0.0));

    // Set up the model
    let prior_variance = 100.0; // Prior variance for beta (vague prior)
    let prior_shape = 1.0; // Prior shape for r
    let prior_scale = 1.0; // Prior scale for r (rate = 1/scale)
    let n_chains = 4;
    let burnin = 1000;
    let samples = 1000;
    let seed = 42;

    println!(
        "\nRunning MCMC with {} burn-in and {} samples ({} chains)...",
        burnin, samples, n_chains
    );

    // Create and run the model
    let model = GibbsNegativeBinomial::new(
        x,
        y,
        prior_variance,
        prior_shape,
        prior_scale,
        n_chains,
        seed,
    );

    // Run MCMC
    let results = model
        .run(burnin, samples, Some(true_beta.to_vec()), Some(true_r))
        .expect("MCMC failed");

    // Print results summary
    results.summary();

    // Print more detailed information
    println!("\nDetailed results:");
    println!("----------------");
    println!("{}", results.run_stats);

    if let Some(true_beta) = &results.true_coefficients {
        println!(
            "\nParameter recovery error (L2 norm): {:.4}",
            true_beta
                .iter()
                .zip(&results.posterior_means)
                .map(|(t, e)| (t - e).powi(2))
                .sum::<f64>()
                .sqrt()
        );
    }
}

//! Tests verifying the correctness of a Metropolis-Hastings sampler for 2D Gaussian distributions.
//!
//! Instead of using a KS test, we now compare the sample means and covariance matrices.

#[cfg(test)]
mod tests {
    use general_mcmc::core::ChainRunner;
    use general_mcmc::distributions::Proposal;
    use general_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
    use general_mcmc::metropolis_hastings::MetropolisHastings;
    use ndarray::{Axis, arr1, arr2};

    // Shared constants.
    const SAMPLE_SIZE: usize = 10_000;
    const BURNIN: usize = 2_500;
    const SEED: u64 = 42;
    const INITIAL_STATE: [f64; 2] = [10.0, 12.0];

    /// Runs the Metropolis-Hastings sampler with the provided target distribution,
    /// and returns the sample reshaped into a (SAMPLE_SIZE, 2) array.
    fn run_sampler(target: &Gaussian2D<f64>) -> ndarray::Array2<f64> {
        let proposal = IsotropicGaussian::new(1.0).set_seed(SEED);
        let mut mh = MetropolisHastings::new(target.clone(), proposal, vec![INITIAL_STATE.into()])
            .seed(SEED);
        let sample = mh.run(SAMPLE_SIZE, BURNIN).unwrap();
        sample.to_shape((SAMPLE_SIZE, 2)).unwrap().to_owned()
    }

    fn sample_covariance(sample: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
        let mean = sample.mean_axis(Axis(0)).unwrap();
        let centered = sample - &mean;
        centered.t().dot(&centered) / (sample.nrows() as f64 - 1.0)
    }

    /// Checks that the sampler produces sample with mean and covariance close to the target.
    #[test]
    fn test_two_d_gaussian_accept() {
        // Set up the target distribution.
        let target = Gaussian2D {
            mean: arr1(&[0.0, 0.0]),
            cov: arr2(&[[4.0, 2.0], [2.0, 3.0]]),
        };

        let sample = run_sampler(&target);

        // Compute sample mean and covariance.
        let mean_mcmc = sample.mean_axis(Axis(0)).unwrap();
        let cov_mcmc = sample_covariance(&sample);

        // Check the sample mean (each component should differ by less than 0.5).
        let mean_diff = (mean_mcmc - target.mean).abs();
        assert!(
            mean_diff[0] < 0.5 && mean_diff[1] < 0.5,
            "Mean deviation too large: {}",
            mean_diff
        );

        // Compute the maximum absolute difference in covariance.
        let max_diff = (cov_mcmc - target.cov)
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);

        assert!(
            max_diff < 0.5,
            "Covariance of false target sample is unexpectedly close to true target covariance. max_diff: {}",
            max_diff
        );
    }

    /// Checks that when using a false target distribution, the sample covariance differs
    /// significantly from that of the correct target.
    #[test]
    fn test_two_d_gaussian_reject() {
        // The correct target (for comparison) has the following covariance.
        let target = Gaussian2D {
            mean: arr1(&[0.0, 0.0]),
            cov: arr2(&[[4.0, 2.0], [2.0, 3.0]]),
        };

        // The false target uses an identity covariance.
        let false_target = Gaussian2D {
            mean: arr1(&[0.0, 0.0]),
            cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
        };

        let sample = run_sampler(&false_target);
        let cov_mcmc = sample_covariance(&sample);

        // Compute the maximum absolute difference in covariance.
        let max_diff = (cov_mcmc - target.cov)
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);

        // Expect at least one element to differ by more than 1.0.
        assert!(
            max_diff > 1.0,
            "Covariance of false target sample is unexpectedly close to true target covariance. max_diff: {}",
            max_diff
        );
    }
}

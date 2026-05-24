/*!
# Metropolis–Hastings Sampler.

This module implements a generic Metropolis–Hastings sampler that can work with any
target distribution `D` and proposal distribution `Q` that implement the corresponding
traits [`Target`] and [`Proposal`]. The sampler runs multiple independent Markov chains in parallel,
each initialized with the same starting state. A global seed is used to ensure reproducibility,
and each chain gets a unique seed by adding its index to the global seed.

## Overview

- **Target Distribution (`D`)**: Provides the (unnormalized) log-density for states via the [`Target`] trait.
- **Proposal Distribution (`Q`)**: Generates candidate states and computes the proposal density via the [`Proposal`] trait.
- **Parallel Chains**: The sampler maintains a vector of [`MHMarkovChain`] instances, each evolving independently.
- **Reproducibility**: The method `set_seed` assigns a unique seed to each chain based on a given global seed.

## Example Usage

```rust
use general_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
use general_mcmc::metropolis_hastings::MetropolisHastings;
use general_mcmc::core::init;
use ndarray::{arr1, arr2};

// Define a 2D Gaussian target distribution with full covariance
let target = Gaussian2D {
    mean: arr1(&[0.0, 0.0]),
    cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
};

// Define an isotropic Gaussian proposal distribution (for any dimension)
let proposal = IsotropicGaussian::new(1.0);

// Starting state for all chains (just one in this case)
let initial_states = init(1, 2);  // Creates 1 chain with 2-dimensional state, initialized with random values

// Create a sampler with 1 chain
let mh = MetropolisHastings::new(target, proposal, initial_states);

// Check that one chain was created
assert_eq!(mh.chains.len(), 1);
```

See also the documentation for [`MHMarkovChain`] and the methods below.
*/

use num_traits::Float;
use rand::distr::Distribution as RandDistribution;
use rand::prelude::*;
// Use rand's Distribution for StandardUniform to avoid rand 0.8/0.9 conflicts.
use rand_distr::StandardUniform;
use std::marker::{PhantomData, Send};

use crate::core::{HasChains, MarkovChain};
use crate::distributions::{Proposal, Target};

/**
The Metropolis–Hastings sampler generates observations from a target distribution by
using a proposal distribution to propose candidate moves and then accepting or rejecting
these moves using the Metropolis–Hastings acceptance criterion.

# Type Parameters
- `S`: The element type for the state (typically a floating-point type).
- `T`: The floating-point type (e.g. `f32` or `f64`).
- `D`: The target distribution type. Must implement [`Target`].
- `Q`: The proposal distribution type. Must implement [`Proposal`].

The sampler maintains multiple independent Markov chains (each represented by [`MHMarkovChain`])
that are run in parallel. A global random seed is provided, and each chain's RNG is seeded by
adding the chain's index to the global seed, ensuring reproducibility.

# Examples

```rust
use general_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
use general_mcmc::metropolis_hastings::MetropolisHastings;
use general_mcmc::core::init;
use ndarray::{arr1, arr2};

let target = Gaussian2D {
    mean: arr1(&[0.0, 0.0]),
    cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
};
let proposal = IsotropicGaussian::new(1.0);
let mh = MetropolisHastings::new(target, proposal, init(1, 2));
assert_eq!(mh.chains.len(), 1);
```
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetropolisHastings<S: Clone, T: Float, D: Clone, Q: Clone> {
    /// The target distribution we want to sample from.
    pub target: D,
    /// The proposal distribution used to generate candidate states.
    pub proposal: Q,
    /// The vector of independent Markov chains.
    pub chains: Vec<MHMarkovChain<S, T, D, Q>>,
}

/// A single Markov chain for the Metropolis–Hastings algorithm.
///
/// Each chain stores its own copy of the target and proposal distributions,  
/// maintains its current state, and uses a chain-specific random number generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MHMarkovChain<S, T, D, Q> {
    /// The target distribution to sample from.
    pub target: D,
    /// The proposal distribution used to generate candidate states.
    pub proposal: Q,
    /// The current state of the chain.
    pub current_state: Vec<S>,
    /// The random number generator for this chain.
    pub rng: SmallRng,
    phantom: PhantomData<T>,
}

impl<S, T, D, Q> MetropolisHastings<S, T, D, Q>
where
    D: Target<S, T> + std::clone::Clone + Send,
    Q: Proposal<S, T> + std::clone::Clone + Send,
    T: Float + Send,
    S: Clone + std::cmp::PartialEq + Send + num_traits::Zero + std::fmt::Debug + 'static,
{
    /**
    Constructs a new Metropolis-Hastings sampler with a given target and proposal,
    initializing each chain at `initial_state` and creating `n_chains` parallel chains.

    # Arguments

    * `target` - The target distribution from which to sample.
    * `proposal` - The proposal distribution used to generate candidate states.
    * `initial_state` - The starting state for all chains.
    * `n_chains` - The number of parallel Markov chains to run.

    # Examples

    ```rust
    use general_mcmc::metropolis_hastings::MetropolisHastings;
    use general_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
    use general_mcmc::core::init;
    use ndarray::{arr1, arr2};

    let target = Gaussian2D {
        mean: arr1(&[0.0, 0.0]),
        cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
    };
    let proposal = IsotropicGaussian::new(1.0);
    let mh = MetropolisHastings::new(target, proposal, init(1, 2));
    assert_eq!(mh.chains.len(), 1);
    ```
    */
    pub fn new(target: D, proposal: Q, initial_states: Vec<Vec<S>>) -> Self {
        let chains = initial_states
            .into_iter()
            .map(|s| MHMarkovChain::new(target.clone(), proposal.clone(), s))
            .collect();
        Self {
            target,
            proposal,
            chains,
        }
    }

    /**
    Sets a new global seed and updates the seed for each chain accordingly.

    Each chain receives a unique seed calculated as `seed + i`, where `i` is the chain index.
    This method ensures reproducibility across runs and parallel chains.

    # Arguments

    * `seed` - The new global seed value.

    # Examples

    ```rust
    use general_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
    use general_mcmc::metropolis_hastings::MetropolisHastings;
    use general_mcmc::core::init;
    use ndarray::{arr1, arr2};

    let target = Gaussian2D {
        mean: arr1(&[0.0, 0.0]),
        cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
    };
    let proposal = IsotropicGaussian::new(1.0);
    let mh = MetropolisHastings::new(target, proposal, init(2, 2)).seed(42);
    ```
    */
    pub fn seed(mut self, seed: u64) -> Self {
        for (i, chain) in self.chains.iter_mut().enumerate() {
            let chain_seed = 1 + seed + i as u64;
            chain.rng = SmallRng::seed_from_u64(chain_seed);
            let proposal_seed = chain_seed.wrapping_add(0x9E3779B97F4A7C15);
            chain.proposal = chain.proposal.clone().set_seed(proposal_seed);
        }
        self
    }
}

impl<S, T, D, Q> HasChains<Vec<S>> for MetropolisHastings<S, T, D, Q>
where
    D: Target<S, T> + Clone + Send,
    Q: Proposal<S, T> + Clone + Send,
    T: Float + Send,
    S: Clone + PartialEq + Send + num_traits::Zero + std::fmt::Debug + 'static,
    StandardUniform: RandDistribution<T>,
{
    /// The concrete chain type used by the sampler.
    type Chain = MHMarkovChain<S, T, D, Q>;

    /// Returns a mutable reference to the internal vector of chains.
    ///
    /// This method allows external code to access and, if needed, modify the vector of  
    /// chains. For example, you may inspect or update individual chains using this reference.
    fn chains_mut(&mut self) -> &mut Vec<Self::Chain> {
        &mut self.chains
    }
}

impl<S, T, D, Q> MHMarkovChain<S, T, D, Q>
where
    D: Target<S, T> + Clone,
    Q: Proposal<S, T> + Clone,
    S: Clone + std::cmp::PartialEq + num_traits::Zero,
    T: Float,
{
    /**
    Creates a new Metropolis–Hastings chain.

    # Arguments
    * `target` - The target distribution.
    * `proposal` - The proposal distribution.
    * `initial_state` - The starting state for the chain.

    # Examples

    ```rust
    use general_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
    use general_mcmc::metropolis_hastings::MHMarkovChain;
    use ndarray::{arr1, arr2};

    let target = Gaussian2D {
        mean: arr1(&[0.0, 0.0]),
        cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
    };
    let proposal = IsotropicGaussian::new(1.0);
    let chain = MHMarkovChain::new(target, proposal, vec![3.0, 5.0]);
    assert_eq!(chain.current_state, vec![3.0, 5.0]);
    ```
    */
    pub fn new(target: D, proposal: Q, initial_state: Vec<S>) -> Self {
        Self {
            target,
            proposal,
            current_state: initial_state,
            rng: SmallRng::seed_from_u64(rand::rng().random::<u64>()),
            phantom: PhantomData,
        }
    }
}

impl<T, F, D, Q> MarkovChain<Vec<T>> for MHMarkovChain<T, F, D, Q>
where
    D: Target<T, F> + Clone,
    Q: Proposal<T, F> + Clone,
    T: Clone + PartialEq + num_traits::Zero,
    F: Float,
    StandardUniform: RandDistribution<F>,
{
    /**
    Performs one Metropolis–Hastings update step.

    A new candidate state is proposed using the proposal distribution.
    The unnormalized log-density of the current and proposed states is computed,
    along with the corresponding proposal densities. The acceptance ratio in log-space
    is calculated as:

    \[
    \log \alpha = \left[\log p(\text{proposed}) + \log q(\text{current} \mid \text{proposed})\right]
                  - \left[\log p(\text{current}) + \log q(\text{proposed} \mid \text{current})\right]
    \]

    A uniform random number is drawn, and if \(\log(\text{Uniform}(0,1))\) is less than
    \(\log \alpha\), the proposed state is accepted. Otherwise, the current state is retained.

    The method returns a reference to the updated state.

    # Examples

    ```rust
    use general_mcmc::core::{MarkovChain, init};
    use general_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
    use general_mcmc::metropolis_hastings::MHMarkovChain;
    use ndarray::{arr1, arr2};

    let target = Gaussian2D {
        mean: arr1(&[0.0, 0.0]),
        cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
    };
    let proposal = IsotropicGaussian::new(1.0);
    let mut chain = MHMarkovChain::new(target, proposal, vec![0.0, 1.0]);
    let new_state = chain.step();
    assert_eq!(new_state.len(), 2);
    ```
    */
    fn step(&mut self) -> &Vec<T> {
        let proposed: Vec<T> = self.proposal.sample(&self.current_state);
        let current_lp = self.target.unnorm_logp(&self.current_state);
        let proposed_lp = self.target.unnorm_logp(&proposed);
        let log_q_forward = self.proposal.logp(&self.current_state, &proposed);
        let log_q_backward = self.proposal.logp(&proposed, &self.current_state);
        let log_accept_ratio = (proposed_lp + log_q_backward) - (current_lp + log_q_forward);
        let u: F = self.rng.random();
        if log_accept_ratio > u.ln() {
            self.current_state = proposed;
        }
        &self.current_state
    }

    /// Returns a reference to the current state of the chain.
    fn current_state(&self) -> &Vec<T> {
        &self.current_state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ChainRunner, init_det}; // or run_progress, etc.
    use crate::distributions::{Gaussian2D, IsotropicGaussian};
    use crate::stats::{RunStats, basic_stats, split_rhat_mean_ess}; // from your posted stats module
    use approx::assert_abs_diff_eq;
    use ndarray::{Array3, Axis, arr1, arr2};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    /// Common test harness for checking that sample mean from a 2D Gaussian matches
    /// the true mean and covariance within floating-point tolerance.
    ///
    /// - `n_chains`: number of parallel chains
    /// - `use_progress`: whether to call `run_progress` instead of `run`
    fn run_gaussian_2d_test(sample_size: usize, n_chains: usize, use_progress: bool) {
        const BURNIN: usize = 500;
        const SEED: u64 = 42;

        assert!(n_chains > 0 && sample_size > 0 && sample_size.is_multiple_of(n_chains));

        // Target distribution
        let target = Gaussian2D {
            mean: arr1(&[0.0, 1.0]),
            cov: arr2(&[[4.0, 2.0], [2.0, 3.0]]),
        };

        // Build the sampler
        let proposal = IsotropicGaussian::new(1.0).set_seed(SEED);
        let mut mh =
            MetropolisHastings::new(target.clone(), proposal, init_det(n_chains, 2)).seed(SEED);

        // Generate sample
        let (sample, _stats) = if use_progress {
            mh.run_progress(sample_size / n_chains, BURNIN).unwrap()
        } else {
            let sample = mh.run(sample_size / n_chains, BURNIN).unwrap();
            let stats = RunStats::from(sample.view());
            (sample, stats)
        };

        // Check correct shape
        assert_eq!(sample.shape(), [n_chains, sample_size / n_chains, 2]);
        if n_chains <= 1 {
            return;
        }

        // Reshape sample into a [sample_size, 2] array
        let stacked = sample
            .into_shape_with_order((sample_size, 2))
            .expect("Failed to reshape sample");

        // Check that mean and covariance match the target distribution
        let mean = stacked.mean_axis(Axis(0)).unwrap();
        let centered = &stacked - &mean;
        let cov = centered.t().dot(&centered) / (stacked.nrows() as f64 - 1.0);

        assert_abs_diff_eq!(mean, target.mean, epsilon = 0.3);
        assert_abs_diff_eq!(cov, target.cov, epsilon = 0.5);
    }

    #[test]
    fn test_single_1_chain() {
        run_gaussian_2d_test(100, 1, false);
    }

    #[test]
    fn test_3_chains() {
        run_gaussian_2d_test(6000, 3, false);
    }

    #[test]
    fn test_progress_1_chain() {
        run_gaussian_2d_test(100, 1, true);
    }

    #[test]
    fn test_progress_3_chains() {
        run_gaussian_2d_test(6000, 3, true);
    }

    #[test]
    #[ignore = "Slow test: run only when explicitly requested"]
    fn test_16_chains_long() {
        run_gaussian_2d_test(80_000_000, 16, false);
    }

    #[test]
    #[ignore = "Slow test: run only when explicitly requested"]
    fn test_progress_16_chains_long() {
        run_gaussian_2d_test(80_000_000, 16, true);
    }

    /// A test that replicates the 2D Metropolis-Hastings experiment multiple times (e.g. 100)
    /// and collects the mean ESS for each parameter across runs, printing summary stats.
    #[test]
    #[ignore = "Benchmark test: run only when explicitly requested"]
    fn test_mean_ess_2d_gaussian() {
        let n_runs = 100;
        let n_chains = 3;
        let burn_in = 500_usize;
        let sample_size_chain = 1500_usize; // total per chain (including burn-in)
        let collected = sample_size_chain - burn_in; // actual collected observations per chain

        // We'll store the mean ESS for x1 and x2 from each run
        let mut ess_x1s = Vec::with_capacity(n_runs);
        let mut ess_x2s = Vec::with_capacity(n_runs);

        // Outer RNG for reproducibility - each run gets a unique seed
        let mut outer_rng = SmallRng::seed_from_u64(42);

        // For each run, we do a fresh Metropolis-Hastings
        for _ in 0..n_runs {
            // 1) Define target distribution
            let target = Gaussian2D {
                mean: arr1(&[0.0, 1.0]),
                cov: arr2(&[[4.0, 2.0], [2.0, 3.0]]),
            };

            // 2) Define proposal
            let proposal = IsotropicGaussian::new(1.0);

            // 3) Initialize MH with 3 parallel chains
            //    init_det(...) is just one of your custom methods that produces
            //    deterministic initial states for each chain. Or use init(...) if you prefer random.
            let mut mh = MetropolisHastings::new(target, proposal, init_det(n_chains, 2));

            // Generate a unique seed for this run (test is still reproducible overall)
            let run_seed: u64 = outer_rng.random();
            mh = mh.seed(run_seed);

            // 4) Run the sampler
            //    This depends on your actual method name.
            //    E.g. `mh.run(num_steps, burn_in)` returns an ndarray of shape (chains, num_steps, param_dim)
            //    Adjust according to your crate's actual interface.
            let sample = mh.run(collected, burn_in).expect("MH run failed");

            // sample.shape() should be [3, 1000, 2]
            assert_eq!(sample.shape(), &[n_chains, collected, 2]);

            // 5) Convert sample to an ndarray of f32 for stats
            //    If your internal type is f64, you can map it to f32.
            //    We'll assume float is okay as f32. Adjust if needed.
            let mut sample_f32 = Array3::<f32>::zeros((n_chains, collected, 2));
            for c in 0..n_chains {
                for t in 0..collected {
                    sample_f32[[c, t, 0]] = sample[[c, t, 0]] as f32;
                    sample_f32[[c, t, 1]] = sample[[c, t, 1]] as f32;
                }
            }

            // 6) We use the function from your stats module to do split-Rhat & ESS
            //    `split_rhat_mean_ess` returns (rhat, ess) for each parameter as 1D arrays.
            let (_, ess_vec) = split_rhat_mean_ess(sample_f32.view());

            // ess_vec is the ESS for each parameter: [ess_x1, ess_x2]
            let ess_x1 = ess_vec[0];
            let ess_x2 = ess_vec[1];

            ess_x1s.push(ess_x1);
            ess_x2s.push(ess_x2);

            // Optionally, you might also record Rhat or do other analyses here
        }

        // Now we have 100 values for ESS x1 and ESS x2. Let's analyze them:
        let ess_x1_array = ndarray::Array1::from_vec(ess_x1s);
        let ess_x2_array = ndarray::Array1::from_vec(ess_x2s);

        // Summarize them using the basic_stats function from your stats module
        let stats_x1 = basic_stats("ESS(x1)", ess_x1_array);
        let stats_x2 = basic_stats("ESS(x2)", ess_x2_array);

        // Print out or assert the summary stats
        println!("{stats_x1}\n{stats_x2}");

        // Optionally, assert certain minimal thresholds or acceptance criteria
        // For example, we might expect an average ESS ~ something.
        // (This is user-dependent and depends on the chain size, etc.)
        assert!(
            stats_x1.mean >= 65.0 && stats_x1.mean <= 125.0,
            "Expected ESS(x1) to average in [65, 125]"
        );
        assert!(
            stats_x2.mean >= 83.0 && stats_x1.mean <= 143.0,
            "Expected ESS(x2) to average in [83, 143]"
        );
        assert!(
            stats_x1.std >= 20.0 && stats_x1.std <= 40.0,
            "Expected std(ESS(x1)) in [20, 40]"
        );
        assert!(
            stats_x2.std >= 20.0 && stats_x1.std <= 40.0,
            "Expected std(ESS(x2)) in [20, 40]"
        );
    }

    /// This test remains separate because it's exercising the "example usage"
    /// scenario from the docs rather than checking numeric correctness.
    #[test]
    fn readme_test() {
        let target = Gaussian2D {
            mean: arr1(&[0.0, 0.0]),
            cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
        };
        let proposal = IsotropicGaussian::new(1.0);

        // Create a MH sampler with 4 parallel chains
        let mut mh = MetropolisHastings::new(target, proposal, init_det(4, 2));

        // Run the sampler for 1100 steps, discarding the first 100 as burn-in
        let sample = mh.run(1000, 100).unwrap();

        // We should have 900 * 4 = 3600 observations
        assert_eq!(sample.shape()[0], 4);
        assert_eq!(sample.shape()[1], 1000);
    }
}

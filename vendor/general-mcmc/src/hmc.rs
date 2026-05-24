//! Hamiltonian Monte Carlo (HMC) sampler.
//!
//! This is modeled similarly to a Metropolis–Hastings sampler but uses gradient-based proposals
//! for improved efficiency. The sampler works in a data-parallel fashion and can update multiple
//! chains simultaneously.
//!
//! The code relies on a target distribution provided via the `BatchedGradientTarget` trait, which computes
//! the unnormalized log probability for a batch of positions. The HMC implementation uses the leapfrog
//! integrator to simulate Hamiltonian dynamics, and the standard accept/reject step for proposal
//! validation.

use crate::batched_hmc::{BatchedGenericHMC, BatchedHamiltonianTarget};
use crate::distributions::BatchedGradientTarget;
use crate::stats::RunStats;
use burn::prelude::*;
use burn::tensor::Element;
use burn::tensor::backend::AutodiffBackend;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::distr::Distribution as RandDistribution;
#[cfg(test)]
use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_distr::uniform::SampleUniform;
use std::error::Error;

/// Adapter to convert BatchedGradientTarget to BatchedHamiltonianTarget<Tensor<B, 2>>.
///
/// This enables the public BatchedGradientTarget trait to work with the
/// internal BatchedHamiltonianTarget trait used by BatchedGenericHMC.
#[derive(Clone, Debug)]
struct BatchedGradientTargetAdapter<GTarget> {
    inner: GTarget,
}

impl<T, B, GTarget> BatchedHamiltonianTarget<Tensor<B, 2>> for BatchedGradientTargetAdapter<GTarget>
where
    T: Float + Element + ElementConversion + SampleUniform + FromPrimitive,
    B: AutodiffBackend<FloatElem = T>,
    GTarget: BatchedGradientTarget<T, B> + std::marker::Sync,
    StandardNormal: RandDistribution<T>,
{
    fn logp_and_grad(&self, position: &Tensor<B, 2>, grad: &mut Tensor<B, 2>) -> Tensor<B, 1> {
        // Position is [n_chains, dim], need to compute logp for each chain
        let pos = position.clone().detach().require_grad();

        // unnorm_logp_batch returns Tensor<B, 1> of shape [n_chains]
        let logp = self.inner.unnorm_logp_batch(pos.clone());

        // Compute gradients via backward pass
        // We need to sum the logp to get a scalar for backward, then extract per-chain gradients
        let logp_sum = logp.clone().sum();
        let grads_inner = pos
            .grad(&logp_sum.backward())
            .expect("grad computation to succeed");

        // Convert inner gradient to Tensor<B, 2>
        let grad_tensor = Tensor::<B, 2>::from_inner(grads_inner);
        grad.inplace(|_| grad_tensor);

        logp.detach()
    }
}

/// A data-parallel Hamiltonian Monte Carlo (HMC) sampler.
///
/// This struct encapsulates the HMC algorithm, including the leapfrog integrator and the
/// accept/reject mechanism, for sampling from a target distribution in a batched manner.
///
/// # Type Parameters
///
/// * `T`: Floating-point type for numerical calculations.
/// * `B`: Autodiff backend from the `burn` crate.
/// * `GTarget`: The target distribution type implementing the `BatchedGradientTarget` trait.
#[derive(Debug)]
pub struct HMC<T, B, GTarget>
where
    T: Float + Element + ElementConversion + SampleUniform + FromPrimitive + ToPrimitive,
    B: AutodiffBackend<FloatElem = T>,
    GTarget: BatchedGradientTarget<T, B> + std::marker::Sync,
    StandardNormal: RandDistribution<T>,
{
    inner: BatchedGenericHMC<Tensor<B, 2>, BatchedGradientTargetAdapter<GTarget>>,
}

impl<T, B, GTarget> HMC<T, B, GTarget>
where
    T: Float
        + burn::tensor::ElementConversion
        + burn::tensor::Element
        + SampleUniform
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive,
    B: AutodiffBackend<FloatElem = T>,
    GTarget: BatchedGradientTarget<T, B> + std::marker::Sync,
    StandardNormal: RandDistribution<T>,
{
    /// Create a new data-parallel HMC sampler.
    ///
    /// This method initializes the sampler with the target distribution, initial positions,
    /// step size, number of leapfrog steps, and a random seed for reproducibility.
    ///
    /// # Parameters
    ///
    /// * `target`: The target distribution implementing the `BatchedGradientTarget` trait.
    /// * `initial_positions`: A vector of vectors containing the initial positions for each chain, with shape `[n_chains][D]`.
    /// * `step_size`: The step size used in the leapfrog integrator.
    /// * `n_leapfrog`: The number of leapfrog steps per update.
    /// * `seed`: A seed for initializing the random number generator.
    ///
    /// # Returns
    ///
    /// A new instance of `HMC`.
    pub fn new(
        target: GTarget,
        initial_positions: Vec<Vec<T>>,
        step_size: T,
        n_leapfrog: usize,
    ) -> Self {
        // Convert Vec<Vec<T>> to Tensor<B, 2> of shape [n_chains, dim]
        let n_chains = initial_positions.len();
        let dim = initial_positions[0].len();
        let flat_data: Vec<T> = initial_positions.into_iter().flatten().collect();
        let td = TensorData::new(flat_data, [n_chains, dim]);
        let positions = Tensor::<B, 2>::from_data(td, &B::Device::default());

        let inner = BatchedGenericHMC::new(
            BatchedGradientTargetAdapter { inner: target },
            positions,
            step_size,
            n_leapfrog,
        );

        Self { inner }
    }

    /// Sets a new random seed.
    ///
    /// This method ensures reproducibility across runs.
    ///
    /// # Arguments
    ///
    /// * `seed` - The new random seed value.
    pub fn set_seed(mut self, seed: u64) -> Self {
        // Note: Burn backend seeding is global; this affects other samplers on the same backend.
        B::seed(seed);
        self.inner = self.inner.set_seed(seed);
        self
    }

    pub fn set_target_accept(mut self, target_accept_p: T) -> Self {
        self.inner = self.inner.set_target_accept(target_accept_p);
        self
    }

    /// Run the HMC sampler for `n_collect` + `n_discard` steps.
    ///
    /// First, the sampler uses the `n_discard` warmup steps to adapt the leapfrog step size and
    /// diagonal mass, then takes
    /// `n_collect` further steps and collects those observations in a 3D tensor of
    /// shape `[n_chains, n_collect, D]`.
    ///
    /// # Parameters
    ///
    /// * `n_collect` - The number of observations to collect and return.
    /// * `n_discard` - The number of observations to discard (burn-in).
    ///
    /// # Returns
    ///
    /// A tensor containing the collected observations.
    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Tensor<B, 3> {
        self.inner.warmup(n_discard);

        if n_collect == 0 {
            let dims = self.inner.positions().dims();
            return Tensor::<B, 3>::empty([dims[0], 0, dims[1]], &B::Device::default());
        }

        let mut samples: Vec<Tensor<B, 2>> = Vec::with_capacity(n_collect);
        for _ in 0..n_collect {
            self.inner.step();
            samples.push(self.inner.positions().clone());
        }

        let stacked = Tensor::<B, 2>::stack(samples, 0);
        stacked.permute([1, 0, 2])
    }

    /// Run the HMC sampler for `n_collect` + `n_discard` steps and displays progress with
    /// convergence statistics.
    ///
    /// First, the sampler uses the `n_discard` warmup steps to adapt the leapfrog step size and
    /// diagonal mass, then takes
    /// `n_collect` further steps and collects those observations in a 3D tensor of
    /// shape `[n_chains, n_collect, D]`.
    ///
    /// This function displays a progress bar (using the `indicatif` crate) that is updated
    /// with an approximate acceptance probability computed over a sliding window of 100 iterations
    /// as well as the potential scale reduction factor, see [Stan Reference Manual.][1]
    ///
    /// # Parameters
    ///
    /// * `n_collect` - The number of observations to collect and return.
    /// * `n_discard` - The number of observations to discard (burn-in).
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - A tensor of shape `[n_chains, n_collect, D]` containing the collected observations.
    /// - A `RunStats` object containing convergence statistics including:
    ///   - Acceptance probability
    ///   - Potential scale reduction factor (R-hat)
    ///   - Effective sample size (ESS)
    ///   - Other convergence diagnostics
    ///
    /// # Example
    ///
    /// ```rust
    /// use general_mcmc::hmc::HMC;
    /// use general_mcmc::distributions::DiffableGaussian2D;
    /// use burn::backend::{Autodiff, NdArray};
    /// use burn::prelude::*;
    ///
    /// // Create a 2D Gaussian target distribution
    /// let target = DiffableGaussian2D::new(
    ///     [0.0_f32, 1.0],  // mean
    ///     [[4.0, 2.0],     // covariance
    ///      [2.0, 3.0]]
    /// );
    ///
    /// // Create HMC sampler with:
    /// // - target distribution
    /// // - initial positions for each chain
    /// // - step size for leapfrog integration
    /// // - number of leapfrog steps
    /// type BackendType = Autodiff<NdArray>;
    /// let mut sampler = HMC::<f32, BackendType, DiffableGaussian2D<f32>>::new(
    ///     target,
    ///     vec![vec![0.0; 2]; 4],    // Initial positions for 4 chains
    ///     0.1,                      // Step size
    ///     5,                       // Number of leapfrog steps
    /// );
    ///
    /// // Run sampler with progress tracking
    /// let (sample, stats) = sampler.run_progress(12, 34).unwrap();
    ///
    /// // Print convergence statistics
    /// println!("{stats}");
    /// ```
    ///
    /// [1]: https://mc-stan.org/docs/2_18/reference-manual/notation-for-samples-chains-and-draws.html
    pub fn run_progress(
        &mut self,
        n_collect: usize,
        n_discard: usize,
    ) -> Result<(Tensor<B, 3>, RunStats), Box<dyn Error>> {
        use crate::stats::MultiChainTracker;
        use indicatif::{ProgressBar, ProgressStyle};

        self.inner.warmup(n_discard);

        let dims = self.inner.positions().dims();
        let (n_chains, dim) = (dims[0], dims[1]);

        let pb = ProgressBar::new(n_collect as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{prefix:8} {bar:40.cyan/blue} {pos}/{len} ({eta}) | {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );
        pb.set_prefix("HMC");

        let mut tracker = MultiChainTracker::new(n_chains, dim);
        let mut samples: Vec<Tensor<B, 2>> = Vec::with_capacity(n_collect);
        let mut last_sync = std::time::Instant::now();
        let sync_interval = std::time::Duration::from_millis(500);

        for step_idx in 0..n_collect {
            self.inner.step();
            let current = self.inner.positions().clone();
            samples.push(current.clone());
            pb.inc(1);

            if step_idx + 1 == n_collect || last_sync.elapsed() >= sync_interval {
                let data = current.to_data();
                if let Ok(slice) = data.as_slice::<T>() {
                    if let Err(e) = tracker.step(slice) {
                        crate::__vmc_err!("HMC tracker update failed: {e}");
                    }
                    if let Ok(max_rhat) = tracker.max_rhat() {
                        pb.set_message(format!(
                            "p(accept)≈{:.2} max(rhat)≈{:.2}",
                            tracker.p_accept, max_rhat
                        ));
                    }
                }
                last_sync = std::time::Instant::now();
            }
        }
        pb.finish_with_message("Done!");

        let stacked = Tensor::<B, 2>::stack(samples, 0);
        let sample = stacked.permute([1, 0, 2]);

        let data = sample.to_data();
        let slice: &[T] = data.as_slice().expect("Tensor data expected to be dense");
        let dims = sample.dims();
        let arr = ndarray::ArrayView3::from_shape((dims[0], dims[1], dims[2]), slice)
            .expect("Shape mismatch");
        let stats = RunStats::from(arr);

        Ok((sample, stats))
    }

    /// Perform one batched HMC update for all chains in parallel.
    ///
    /// The update consists of:
    /// 1) Sampling momenta from a standard normal distribution.
    /// 2) Running the leapfrog integrator to propose new positions.
    /// 3) Performing an accept/reject step for each chain.
    ///
    /// This method updates `self.positions` in-place.
    pub fn step(&mut self) {
        self.inner.step();
    }

    /// Get a reference to the target distribution.
    pub fn target(&self) -> &GTarget {
        &self.inner.target().inner
    }

    /// Get a reference to the current positions.
    pub fn positions(&self) -> &Tensor<B, 2> {
        self.inner.positions()
    }

    /// Get a reference to the step size.
    pub fn step_size(&self) -> &T {
        self.inner.step_size()
    }

    /// Get the number of leapfrog steps.
    pub fn n_leapfrog(&self) -> usize {
        self.inner.n_leapfrog()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        core::init,
        dev_tools::Timer,
        distributions::{DiffableGaussian2D, Rosenbrock2D, RosenbrockND},
        stats::split_rhat_mean_ess,
    };
    use ndarray::ArrayView3;

    use super::*;
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::Tensor,
    };

    // Use the CPU backend (NdArray) wrapped in Autodiff.
    type BackendType = Autodiff<NdArray>;

    #[test]
    fn test_hmc_single() {
        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        let initial_positions = vec![vec![0.0_f32, 0.0]];
        let n_collect = 3;

        let mut sampler =
            HMC::<f32, BackendType, Rosenbrock2D<f32>>::new(target, initial_positions, 0.01, 2)
                .set_seed(42);

        let mut timer = Timer::new();
        let sample: Tensor<BackendType, 3> = sampler.run(n_collect, 0);
        timer.log(format!(
            "Collected sample (10 chains) with shape: {:?}",
            sample.dims()
        ));
        assert_eq!(sample.dims(), [1, 3, 2]);
    }

    #[test]
    fn test_3_chains() {
        type BackendType = Autodiff<NdArray>;

        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        let initial_positions = vec![vec![1.0_f32, 2.0_f32]; 3];
        let n_collect = 10;

        let mut sampler =
            HMC::<f32, BackendType, Rosenbrock2D<f32>>::new(target, initial_positions, 0.01, 2)
                .set_seed(42);

        let mut timer = Timer::new();
        let sample: Tensor<BackendType, 3> = sampler.run(n_collect, 0);
        timer.log(format!(
            "Collected sample (3 chains) with shape: {:?}",
            sample.dims()
        ));
        assert_eq!(sample.dims(), [3, 10, 2]);
    }

    #[test]
    fn test_progress_3_chains() {
        type BackendType = Autodiff<NdArray>;

        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        let initial_positions = vec![vec![1.0_f32, 2.0_f32]; 3];
        let n_collect = 10;

        let mut sampler =
            HMC::<f32, BackendType, Rosenbrock2D<f32>>::new(target, initial_positions, 0.05, 2)
                .set_seed(42);

        let mut timer = Timer::new();
        let sample: Tensor<BackendType, 3> = sampler.run_progress(n_collect, 3).unwrap().0;
        timer.log(format!(
            "Collected sample (10 chains) with shape: {:?}",
            sample.dims()
        ));
        assert_eq!(sample.dims(), [3, 10, 2]);
    }

    #[test]
    fn test_gaussian_2d_hmc_debug() {
        let n_chains = 1;
        let n_discard = 1;
        let n_collect = 1;

        let target = DiffableGaussian2D::new([0.0, 1.0], [[4.0, 2.0], [2.0, 3.0]]);
        let initial_positions = vec![vec![0.0_f32, 0.0_f32]];

        type BackendType = Autodiff<NdArray>;
        let mut sampler = HMC::<f32, BackendType, DiffableGaussian2D<f32>>::new(
            target,
            initial_positions,
            0.1,
            1,
        )
        .set_seed(42);

        let sample_3d = sampler.run(n_collect, n_discard);

        assert_eq!(sample_3d.dims(), [n_chains, n_collect, 2]);
    }

    #[test]
    fn test_warmup_adapts_step_size() {
        type BackendType = Autodiff<NdArray>;

        let target = DiffableGaussian2D::new([0.0_f32, 1.0], [[4.0, 2.0], [2.0, 3.0]]);
        let initial_positions = vec![vec![0.0_f32, 0.0_f32]; 4];
        let initial_step_size = 2.0_f32;

        let mut sampler = HMC::<f32, BackendType, DiffableGaussian2D<f32>>::new(
            target,
            initial_positions,
            initial_step_size,
            10,
        )
        .set_seed(42);

        sampler.run(0, 50);

        assert!(
            *sampler.step_size() < initial_step_size,
            "Expected warmup to reduce an overlarge step size, got {}",
            sampler.step_size()
        );
    }

    #[test]
    fn test_warmup_can_increase_tiny_step_size() {
        type BackendType = Autodiff<NdArray>;

        let target = DiffableGaussian2D::new([0.0_f32, 1.0], [[4.0, 2.0], [2.0, 3.0]]);
        let initial_positions = vec![vec![0.0_f32, 0.0_f32]; 4];
        let initial_step_size = 1e-6_f32;

        let mut sampler = HMC::<f32, BackendType, DiffableGaussian2D<f32>>::new(
            target,
            initial_positions,
            initial_step_size,
            10,
        )
        .set_seed(42);

        sampler.run(0, 50);

        assert!(
            *sampler.step_size() > initial_step_size,
            "Expected warmup to increase an undersized step size, got {}",
            sampler.step_size()
        );
    }

    #[test]
    fn test_warmup_adapts_diagonal_mass_for_anisotropic_target() {
        type BackendType = Autodiff<NdArray>;

        let target = DiffableGaussian2D::new([0.0_f32, 0.0], [[0.25, 0.0], [0.0, 9.0]]);
        let initial_positions = vec![
            vec![-1.5_f32, -9.0],
            vec![-0.5, -3.0],
            vec![0.5, 3.0],
            vec![1.5, 9.0],
        ];

        let mut sampler = HMC::<f32, BackendType, DiffableGaussian2D<f32>>::new(
            target,
            initial_positions,
            0.3,
            8,
        )
        .set_seed(42);

        sampler.run(0, 120);

        let mass = sampler.inner.mass_diag();
        assert_eq!(mass.len(), 2);
        assert!(mass[0].is_finite() && mass[0] > 0.0);
        assert!(mass[1].is_finite() && mass[1] > 0.0);
        assert!(
            mass[1] > mass[0] * 4.0,
            "Expected warmup to learn a larger mass for the broader axis, got {:?}",
            mass
        );
    }

    #[test]
    fn test_gaussian_2d_hmc_single_run() {
        // Each experiment uses 3 chains:
        let n_chains = 3;

        let n_discard = 500;
        let n_collect = 1000;

        // 1) Define the 2D Gaussian target distribution:
        //    mean: [0.0, 1.0], cov: [[4.0, 2.0], [2.0, 3.0]]
        let target = DiffableGaussian2D::new([0.0, 1.0], [[4.0, 2.0], [2.0, 3.0]]);

        // 2) Define 3 chains, each chain is 2-dimensional:
        let initial_positions = vec![
            vec![1.0_f32, 2.0_f32],
            vec![1.0_f32, 2.0_f32],
            vec![1.0_f32, 2.0_f32],
        ];

        // 3) Create the HMC sampler using NdArray backend with autodiff
        type BackendType = Autodiff<NdArray>;
        let mut sampler = HMC::<f32, BackendType, DiffableGaussian2D<f32>>::new(
            target,
            initial_positions,
            0.1, // step size
            10,  // leapfrog steps
        )
        .set_seed(42);

        // 4) Run the sampler for (burn_in + collected) steps, discard the first `burn_in`
        //    The shape of `sample` will be [n_chains, collected, 2]
        let sample_3d = sampler.run(n_collect, n_discard);

        // Check shape is as expected
        assert_eq!(sample_3d.dims(), [n_chains, n_collect, 2]);

        // 5) Convert the sample into an ndarray view
        let data = sample_3d.to_data();
        let arr =
            ArrayView3::from_shape(sample_3d.dims(), data.as_slice::<f32>().unwrap()).unwrap();

        // 6) Compute split-Rhat and ESS
        let (rhat, ess_vals) = split_rhat_mean_ess(arr.view());
        let ess1 = ess_vals[0];
        let ess2 = ess_vals[1];

        println!("\nSingle Run Results:");
        println!("Rhat: {:?}", rhat);
        println!("ESS(Param1): {:.2}", ess1);
        println!("ESS(Param2): {:.2}", ess2);

        // Optionally, add some asserts about expected minimal ESS
        assert!(ess1 > 50.0, "Expected param1 ESS > 50, got {:.2}", ess1);
        assert!(ess2 > 50.0, "Expected param2 ESS > 50, got {:.2}", ess2);
    }

    #[test]
    fn test_gaussian_2d_hmc_ess_stats() {
        use crate::stats::basic_stats;
        use indicatif::{ProgressBar, ProgressStyle};
        use ndarray::Array1;

        let n_runs = 100;
        let n_chains = 3;
        let n_discard = 500;
        let n_collect = 1000;
        let mut rng = SmallRng::seed_from_u64(42);

        // We'll store the ESS and R-hat values for each parameter across all runs
        let mut ess_param1s = Vec::with_capacity(n_runs);
        let mut ess_param2s = Vec::with_capacity(n_runs);
        let mut rhat_param1s = Vec::with_capacity(n_runs);
        let mut rhat_param2s = Vec::with_capacity(n_runs);

        // Set up the progress bar
        let pb = ProgressBar::new(n_runs as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{prefix:8} {bar:40.cyan/blue} {pos}/{len} ({eta}) | {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );
        pb.set_prefix("HMC Test");

        for run in 0..n_runs {
            // 1) Define the 2D Gaussian target distribution:
            //    mean: [0.0, 1.0], cov: [[4.0, 2.0], [2.0, 3.0]]
            let target = DiffableGaussian2D::new([0.0_f32, 1.0], [[4.0, 2.0], [2.0, 3.0]]);

            // 2) Define 3 chains, each chain is 2-dimensional:
            // Create a seeded RNG for reproducible initial positions
            let initial_positions: Vec<Vec<f32>> = (0..n_chains)
                .map(|_| {
                    // Sample 2D position from standard normal
                    vec![
                        rng.sample::<f32, _>(StandardNormal),
                        rng.sample::<f32, _>(StandardNormal),
                    ]
                })
                .collect();

            // 3) Create the HMC sampler using NdArray backend with autodiff
            type BackendType = Autodiff<NdArray>;
            let mut sampler = HMC::<f32, BackendType, DiffableGaussian2D<f32>>::new(
                target,
                initial_positions,
                0.1, // step size
                10,  // leapfrog steps
            )
            .set_seed(run as u64); // Use run number as seed for reproducibility

            // 4) Run the sampler for (n_discard + n_collect) steps, discard the first `n_discard`
            //    observations
            let sample_3d = sampler.run(n_collect, n_discard);

            // Check shape is as expected
            assert_eq!(sample_3d.dims(), [n_chains, n_collect, 2]);

            // 5) Convert the sample into an ndarray view
            let data = sample_3d.to_data();
            let arr =
                ArrayView3::from_shape(sample_3d.dims(), data.as_slice::<f32>().unwrap()).unwrap();

            // 6) Compute split-Rhat and ESS
            let (rhat, ess_vals) = split_rhat_mean_ess(arr.view());
            let ess1 = ess_vals[0];
            let ess2 = ess_vals[1];

            // Store ESS values
            ess_param1s.push(ess1);
            ess_param2s.push(ess2);

            // Store R-hat values from stats object
            rhat_param1s.push(rhat[0]);
            rhat_param2s.push(rhat[1]);

            pb.inc(1);

            // Update progress bar with current ESS statistics across runs
            if run > 0 {
                // Calculate mean and std of ESS for both parameters across all runs so far
                let n_runs = run as f64 + 1.0;
                let mean_ess1 = ess_param1s.iter().sum::<f64>() / n_runs;
                let mean_ess2 = ess_param2s.iter().sum::<f64>() / n_runs;

                // Calculate standard deviations
                let var_ess1 = ess_param1s
                    .iter()
                    .map(|&x| (x - mean_ess1).powi(2))
                    .sum::<f64>()
                    / n_runs;
                let var_ess2 = ess_param2s
                    .iter()
                    .map(|&x| (x - mean_ess2).powi(2))
                    .sum::<f64>()
                    / n_runs;

                let std_ess1 = var_ess1.sqrt();
                let std_ess2 = var_ess2.sqrt();

                pb.set_message(format!(
                    "ESS1={:.0}±{:.0} ESS2={:.0}±{:.0}",
                    mean_ess1, std_ess1, mean_ess2, std_ess2
                ));
            } else {
                // For the first run, just show the current values
                pb.set_message(format!("ESS1={:.0} ESS2={:.0}", ess1, ess2));
            }
        }
        pb.finish_with_message("All runs complete!");

        // Convert to ndarray for statistics
        let ess_param1_array = Array1::from_vec(ess_param1s);
        let ess_param2_array = Array1::from_vec(ess_param2s);
        let rhat_param1_array = Array1::from_vec(rhat_param1s);
        let rhat_param2_array = Array1::from_vec(rhat_param2s);

        // Compute and print statistics
        let stats_p1_ess = basic_stats("ESS(Param1)", ess_param1_array);
        let stats_p2_ess = basic_stats("ESS(Param2)", ess_param2_array);
        let stats_p1_rhat = basic_stats("R-hat(Param1)", rhat_param1_array);
        let stats_p2_rhat = basic_stats("R-hat(Param2)", rhat_param2_array);

        println!("\nStatistics over {} runs:", n_runs);
        println!("\nESS Statistics:");
        println!("{stats_p1_ess}\n{stats_p2_ess}");
        println!("\nR-hat Statistics:");
        println!("{stats_p1_rhat}\n{stats_p2_rhat}");

        // Assertions for ESS
        assert!(
            (135.0..=200.0).contains(&stats_p1_ess.mean),
            "Expected param1 ESS to average in [135, 200], got {:.2}",
            stats_p1_ess.mean
        );
        assert!(
            (141.0..=230.0).contains(&stats_p2_ess.mean),
            "Expected param2 ESS to average in [141, 230], got {:.2}",
            stats_p2_ess.mean
        );

        // Assertions for R-hat (should be close to 1.0)
        assert!(
            (0.95..=1.05).contains(&stats_p1_rhat.mean),
            "Expected param1 R-hat to be in [0.95, 1.05], got {:.2}",
            stats_p1_rhat.mean
        );
        assert!(
            (0.95..=1.05).contains(&stats_p2_rhat.mean),
            "Expected param2 R-hat to be in [0.95, 1.05], got {:.2}",
            stats_p2_rhat.mean
        );
    }

    #[test]
    fn test_bench_noprogress() {
        type BackendType = Autodiff<burn::backend::NdArray>;

        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        let initial_positions = init(6, 2);
        let n_collect = 5000;
        let n_discard = 500;

        let mut sampler =
            HMC::<f32, BackendType, Rosenbrock2D<f32>>::new(target, initial_positions, 0.01, 50)
                .set_seed(42);

        let mut timer = Timer::new();
        let sample = sampler.run(n_collect, n_discard);
        timer.log(format!(
            "HMC sampler: generated {} observations.",
            sample.dims()[0..2].iter().product::<usize>()
        ));
        assert_eq!(sample.dims(), [6, 5000, 2]);

        let data = sample.to_data();
        let array = ArrayView3::from_shape(sample.dims(), data.as_slice::<f32>().unwrap()).unwrap();
        let (split_rhat, ess) = split_rhat_mean_ess(array);
        let max_rhat = split_rhat.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_ess = ess.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("MAX Split Rhat: {}", max_rhat);
        println!("MIN ESS: {}", min_ess);
    }

    #[test]
    fn test_progress_bench() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<burn::backend::NdArray>;
        BackendType::seed(42);

        // Create the Rosenbrock target (a = 1, b = 100)
        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // We'll define 6 chains all initialized to (1.0, 2.0).
        let n_chains = 6;
        let initial_positions = vec![vec![1.0_f32, 2.0_f32]; n_chains];
        let n_collect = 1000;
        let n_discard = 1000;

        // Create the data-parallel HMC sampler.
        let mut sampler = HMC::<f32, BackendType, Rosenbrock2D<f32>>::new(
            target,
            initial_positions,
            0.01, // step size
            50,   // number of leapfrog steps per update
        )
        .set_seed(42);

        // Run HMC for n_collect steps.
        let mut timer = Timer::new();
        let sample = sampler.run_progress(n_collect, n_discard).unwrap().0;
        timer.log(format!(
            "HMC sampler: generated {} observations.",
            sample.dims()[0..2].iter().product::<usize>()
        ));
        println!(
            "Chain 1, first 10: {}",
            sample.clone().slice([0..1, 0..10, 0..1])
        );
        println!(
            "Chain 2, first 10: {}",
            sample.clone().slice([2..3, 0..10, 0..1])
        );

        #[cfg(feature = "csv")]
        crate::io::csv::save_csv_tensor(sample.clone(), "/tmp/hmc-sample.csv")
            .expect("Expected saving to succeed");

        assert_eq!(sample.dims(), [n_chains, n_collect, 2]);
    }

    #[test]
    fn test_bench_10000d() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<burn::backend::NdArray>;

        let seed = 42;
        let d = 10000;
        let n_chains = 6;
        let n_collect = 100;
        let n_discard = 100;

        let rng = SmallRng::seed_from_u64(seed);
        // We'll define 6 chains all initialized to (1.0, 2.0).
        let initial_positions: Vec<Vec<f32>> =
            vec![rng.sample_iter(StandardNormal).take(d).collect(); n_chains];

        // Create the data-parallel HMC sampler.
        let mut sampler = HMC::<f32, BackendType, RosenbrockND>::new(
            RosenbrockND {},
            initial_positions,
            0.01, // step size
            50,   // number of leapfrog steps per update
        )
        .set_seed(42);

        // Run HMC for n_collect steps.
        let mut timer = Timer::new();
        let sample = sampler.run(n_collect, n_discard);
        timer.log(format!(
            "HMC sampler: generated {} observations.",
            sample.dims()[0..2].iter().product::<usize>()
        ));
        assert_eq!(sample.dims(), [n_chains, n_collect, d]);
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn test_progress_10000d_bench() {
        type BackendType = Autodiff<burn::backend::Wgpu>;

        let seed = 42;
        let d = 10000;
        let n_chains = 6;

        let rng = SmallRng::seed_from_u64(seed);
        // We'll define 6 chains all initialized to (1.0, 2.0).
        let initial_positions: Vec<Vec<f32>> =
            vec![rng.sample_iter(StandardNormal).take(d).collect(); n_chains];
        let n_collect = 100;
        let n_discard = 100;

        // Create the data-parallel HMC sampler.
        let mut sampler = HMC::<f32, BackendType, RosenbrockND>::new(
            RosenbrockND {},
            initial_positions,
            0.01, // step size
            50,   // number of leapfrog steps per update
        )
        .set_seed(42);

        // Run HMC for n_collect steps.
        let mut timer = Timer::new();
        let sample = sampler.run_progress(n_collect, n_discard).unwrap().0;
        timer.log(format!(
            "HMC sampler: generated {} observations.",
            sample.dims()[0..2].iter().product::<usize>()
        ));
        assert_eq!(sample.dims(), [n_chains, n_collect, d]);
    }
}

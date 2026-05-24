//! No-U-Turn Sampler (NUTS).
//!
//! A parallel implementation of NUTS running independent Markov chains via Rayon.
//!
//! ## Example: custom 2D Rosenbrock target
//! ```rust
//! use general_mcmc::core::init;
//! use general_mcmc::distributions::GradientTarget;
//! use general_mcmc::nuts::NUTS;
//! use burn::backend::{Autodiff, NdArray};
//! use burn::prelude::*;
//!
//! type B = Autodiff<NdArray>;
//!
//! #[derive(Clone)]
//! struct Rosenbrock2D { a: f32, b: f32 }
//!
//! impl GradientTarget<f32, B> for Rosenbrock2D {
//!     fn unnorm_logp(&self, position: Tensor<B, 1>) -> Tensor<B, 1> {
//!         let x = position.clone().slice(s![0..1]);
//!         let y = position.slice(s![1..2]);
//!         let term_1 = (-x.clone()).add_scalar(self.a).powi_scalar(2);
//!         let term_2 = y.sub(x.powi_scalar(2)).powi_scalar(2).mul_scalar(self.b);
//!         -(term_1 + term_2)
//!     }
//! }
//!
//! let target = Rosenbrock2D { a: 1.0, b: 100.0 };
//! let initial_positions = init::<f32>(4, 2);    // 4 chains in 2D
//! let mut sampler = NUTS::new(target, initial_positions, 0.9);
//! let (samples, stats) = sampler.run_progress(100, 20).unwrap();
//! ```
//!
//! ## Inspiration
//! Borrowed ideas from [mfouesneau/NUTS](https://github.com/mfouesneau/NUTS).

use crate::distributions::GradientTarget;
use crate::generic_hmc::HamiltonianTarget;
use crate::generic_nuts::{GenericNUTS, GenericNUTSChain};
use crate::stats::RunStats;
use burn::prelude::*;
#[cfg(test)]
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Element, ElementConversion};
use num_traits::{Float, FromPrimitive};
use rand::distr::Distribution as RandDistribution;
// Bind to rand's Distribution to avoid mismatches from transitive rand 0.8 deps.
use rand_distr::uniform::SampleUniform;
use rand_distr::{Exp1, StandardNormal, StandardUniform};
use std::error::Error;
use std::marker::PhantomData;

#[derive(Debug)]
struct BurnGradientTarget<GTarget, T> {
    inner: GTarget,
    _marker: PhantomData<T>,
}

impl<T, B, GTarget> HamiltonianTarget<Tensor<B, 1>> for BurnGradientTarget<GTarget, T>
where
    T: Float + Element + ElementConversion + SampleUniform + FromPrimitive,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B> + Sync,
    StandardNormal: RandDistribution<B::FloatElem>,
    B::FloatElem: Float + Element + ElementConversion + SampleUniform + FromPrimitive + Copy,
{
    fn logp_and_grad(&self, position: &Tensor<B, 1>, grad: &mut Tensor<B, 1>) -> B::FloatElem {
        let (logp, grad_tensor) = self.inner.unnorm_logp_and_grad(position.clone());
        grad.inplace(|_| grad_tensor.clone());
        logp.into_scalar()
    }
}

/// No-U-Turn Sampler (NUTS).
///
/// Encapsulates multiple independent Markov chains using the NUTS algorithm. Utilizes dual-averaging
/// step size adaptation and dynamic trajectory lengths to efficiently explore complex posterior geometries.
/// Chains are executed concurrently via Rayon, each evolving independently.
///
/// # Type Parameters
/// - `T`: Floating-point type for numerical calculations.
/// - `B`: Autodiff backend from the `burn` crate.
/// - `GTarget`: Target distribution type implementing the `GradientTarget` trait.
pub struct NUTS<T, B, GTarget>
where
    T: Float
        + Element
        + ElementConversion
        + SampleUniform
        + FromPrimitive
        + num_traits::ToPrimitive
        + Send,
    B: AutodiffBackend + Send,
    GTarget: GradientTarget<T, B> + Sync + Send,
    StandardNormal: RandDistribution<B::FloatElem>,
    StandardUniform: RandDistribution<B::FloatElem>,
    Exp1: RandDistribution<B::FloatElem>,
    B::FloatElem: Float + Element + ElementConversion + SampleUniform + FromPrimitive + Copy,
{
    inner: GenericNUTS<Tensor<B, 1>, BurnGradientTarget<GTarget, T>>,
    _phantom: PhantomData<T>,
}

impl<T, B, GTarget> NUTS<T, B, GTarget>
where
    T: Float
        + Element
        + ElementConversion
        + SampleUniform
        + FromPrimitive
        + num_traits::ToPrimitive
        + Send,
    B: AutodiffBackend + Send,
    GTarget: GradientTarget<T, B> + Sync + Send,
    StandardNormal: RandDistribution<B::FloatElem>,
    StandardUniform: RandDistribution<B::FloatElem>,
    Exp1: RandDistribution<B::FloatElem>,
    B::FloatElem: Float + Element + ElementConversion + SampleUniform + FromPrimitive + Copy,
{
    /// Creates a new NUTS sampler with the given target distribution and initial state for each chain.
    ///
    /// # Parameters
    /// - `target`: The target distribution implementing `GradientTarget`.
    /// - `initial_positions`: A vector of initial positions for each chain, shape `[n_chains, D]`.
    /// - `target_accept_p`: Desired average acceptance probability for the dual-averaging adaptation. Try values between 0.6 and 0.95.
    ///
    /// # Returns
    /// A newly initialized `NUTS` instance.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use burn::backend::{Autodiff, NdArray};
    /// # use general_mcmc::nuts::NUTS;
    /// # use general_mcmc::distributions::DiffableGaussian2D;
    /// type B = Autodiff<NdArray>;
    ///
    /// // Create a 2D Gaussian with mean [0,0] and identity covariance
    /// let gauss = DiffableGaussian2D::new([0.0_f64, 0.0], [[1.0, 0.0], [0.0, 1.0]]);
    ///
    /// // Initialize 3 chains in 2D at different starting points
    /// let init_positions = vec![
    ///     vec![-1.0, -1.0],
    ///     vec![ 0.0,  0.0],
    ///     vec![ 1.0,  1.0],
    /// ];
    ///
    /// // Build the sampler targeting 85% acceptance probability
    /// let sampler: NUTS<f64, B, _> = NUTS::new(gauss, init_positions, 0.85);
    /// ```
    pub fn new(target: GTarget, initial_positions: Vec<Vec<T>>, target_accept_p: T) -> Self {
        let positions_vec: Vec<Tensor<B, 1>> = initial_positions
            .into_iter()
            .map(|pos| {
                let len = pos.len();
                let pos_elem: Vec<B::FloatElem> =
                    pos.into_iter().map(B::FloatElem::from_elem).collect();
                let td: TensorData = TensorData::new(pos_elem, [len]);
                Tensor::<B, 1>::from_data(td, &B::Device::default())
            })
            .collect();
        let target_accept_p_elem = B::FloatElem::from_elem(target_accept_p);
        let inner = GenericNUTS::new(
            BurnGradientTarget {
                inner: target,
                _marker: PhantomData,
            },
            positions_vec,
            target_accept_p_elem,
        );
        Self {
            inner,
            _phantom: PhantomData,
        }
    }

    /// Runs all chains for a total of `n_collect + n_discard` steps and collects samples.
    ///
    /// First discards `n_discard` warm-up steps for each chain (during which adaptation occurs),
    /// then collects `n_collect` samples per chain.
    ///
    /// # Parameters
    /// - `n_collect`: Number of samples to collect after warm-up per chain.
    /// - `n_discard`: Number of warm-up (burn-in) steps to discard per chain.
    ///
    /// # Returns
    /// A 3D tensor of shape `[n_chains, n_collect, D]` containing the collected samples.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use burn::backend::{Autodiff, NdArray};
    /// # use burn::prelude::Tensor;
    /// # use general_mcmc::nuts::NUTS;
    /// # use general_mcmc::core::init;
    /// # use general_mcmc::distributions::DiffableGaussian2D;
    /// type B = Autodiff<NdArray>;
    ///
    /// // As above, construct the sampler
    /// let gauss = DiffableGaussian2D::new([0.0_f32, 0.0], [[1.0,0.0],[0.0,1.0]]);
    /// let mut sampler = NUTS::new(gauss, init::<f32>(2, 2), 0.8);
    ///
    /// // Discard 50 warm-up steps, then collect 150 observations per chain
    /// let sample: Tensor<B, 3> = sampler.run(150, 50);
    ///
    /// // sample.dims() == [2 chains, 150 observations, 2 dimensions]
    /// assert_eq!(sample.dims(), [2, 150, 2]);
    /// ```
    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Tensor<B, 3> {
        // Note: On GPU backends, NUTS can be slower due to CPU-driven tree building
        // and synchronization when evaluating stop criteria. Prefer HMC for GPU-heavy workloads.
        if n_collect == 0 {
            let (n_chains, dim) = {
                let chains = self.inner.chains_mut();
                let n_chains = chains.len();
                let dim = chains[0].position().dims()[0];
                (n_chains, dim)
            };
            return Tensor::<B, 3>::empty([n_chains, 0, dim], &B::Device::default());
        }

        let chains = self.inner.chains_mut();
        let n_chains = chains.len();
        let dim = chains[0].position().dims()[0];
        let mut out = Tensor::<B, 3>::empty([n_chains, n_collect, dim], &B::Device::default());

        for (chain_idx, chain) in chains.iter_mut().enumerate() {
            chain.init_chain_state(n_collect, n_discard);
            let total = n_collect + n_discard;
            for step_idx in 0..total {
                if step_idx > 0 {
                    chain.step();
                }
                if step_idx >= n_discard {
                    let pos = chain
                        .position()
                        .clone()
                        .unsqueeze_dim::<2>(0)
                        .unsqueeze_dim::<3>(0);
                    out.inplace(|tensor| {
                        tensor.slice_assign(
                            [
                                chain_idx..chain_idx + 1,
                                step_idx - n_discard..step_idx - n_discard + 1,
                                0..dim,
                            ],
                            pos,
                        )
                    });
                }
            }
        }
        out
    }

    /// Run with live progress bars and collect summary stats.
    ///
    /// Spawns a background thread to render per-chain and global bars,
    /// then returns `(samples, RunStats)` when done.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn::backend::{Autodiff, NdArray};
    /// use general_mcmc::distributions::Rosenbrock2D;
    /// use general_mcmc::nuts::NUTS;
    /// use general_mcmc::core::init;
    ///
    /// type B = Autodiff<NdArray>;
    ///
    /// let target = Rosenbrock2D { a: 1.0, b: 100.0 };
    /// let init   = init::<f64>(4, 2);    // 4 chains in 2D
    /// let mut sampler = NUTS::<f64, B, Rosenbrock2D<f64>>::new(target, init, 0.9);
    /// let (samples, stats) = sampler.run_progress(100, 20).unwrap();
    /// ```
    ///
    /// You can swap in any other [`GradientTarget`] just as easily.
    pub fn run_progress(
        &mut self,
        n_collect: usize,
        n_discard: usize,
    ) -> Result<(Tensor<B, 3>, RunStats), Box<dyn Error>> {
        let (sample, stats) = self.inner.run_progress(n_collect, n_discard)?;
        Ok((array3_to_tensor(sample), stats))
    }

    /// Sets a new random seed for all chains to ensure reproducibility.
    ///
    /// # Parameters
    /// - `seed`: Base seed value. Each chain will derive its own seed for independence.
    ///
    /// # Returns
    /// `self` with the RNGs re-seeded.
    pub fn set_seed(mut self, seed: u64) -> Self {
        // Note: Burn backend seeding is global; this affects other samplers on the same backend.
        B::seed(seed);
        self.inner = self.inner.set_seed(seed);
        self
    }
}

/// Single-chain state and adaptation for NUTS.
///
/// Manages the dynamic trajectory building, dual-averaging adaptation of step size,
/// and current position for one chain.
pub struct NUTSChain<T, B, GTarget>
where
    T: Float
        + Element
        + ElementConversion
        + SampleUniform
        + FromPrimitive
        + num_traits::ToPrimitive
        + Send,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B> + Sync + Send,
    StandardNormal: RandDistribution<B::FloatElem>,
    StandardUniform: RandDistribution<B::FloatElem>,
    Exp1: RandDistribution<B::FloatElem>,
    B::FloatElem: Float + Element + ElementConversion + SampleUniform + FromPrimitive + Copy,
{
    inner: GenericNUTSChain<Tensor<B, 1>, BurnGradientTarget<GTarget, T>>,
    _phantom: PhantomData<T>,
}

impl<T, B, GTarget> NUTSChain<T, B, GTarget>
where
    T: Float
        + Element
        + ElementConversion
        + SampleUniform
        + FromPrimitive
        + num_traits::ToPrimitive
        + Send,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B> + Sync + Send,
    StandardNormal: RandDistribution<B::FloatElem>,
    StandardUniform: RandDistribution<B::FloatElem>,
    Exp1: RandDistribution<B::FloatElem>,
    B::FloatElem: Float + Element + ElementConversion + SampleUniform + FromPrimitive + Copy,
{
    /// Constructs a new NUTSChain for a single chain with the given initial position.
    ///
    /// # Parameters
    /// - `target`: The target distribution implementing `GradientTarget`.
    /// - `initial_position`: Initial position vector of length `D`.
    /// - `target_accept_p`: Desired average acceptance probability for adaptation.
    ///
    /// # Returns
    /// An initialized `NUTSChain`.
    pub fn new(target: GTarget, initial_position: Vec<T>, target_accept_p: T) -> Self {
        let len = initial_position.len();
        let position_elem: Vec<B::FloatElem> = initial_position
            .into_iter()
            .map(B::FloatElem::from_elem)
            .collect();
        let td: TensorData = TensorData::new(position_elem, [len]);
        let position = Tensor::<B, 1>::from_data(td, &B::Device::default());
        let inner = GenericNUTSChain::new(
            BurnGradientTarget {
                inner: target,
                _marker: PhantomData,
            },
            position.clone(),
            B::FloatElem::from_elem(target_accept_p),
        );
        Self {
            inner,
            _phantom: PhantomData,
        }
    }

    /// Sets a new random seed for this chain to ensure reproducibility.
    ///
    /// # Parameters
    /// - `seed`: Seed value for the chain's RNG.
    ///
    /// # Returns
    /// `self` with the RNG re-seeded.
    pub fn set_seed(mut self, seed: u64) -> Self {
        // Note: Burn backend seeding is global; this affects other samplers on the same backend.
        B::seed(seed);
        self.inner = self.inner.set_seed(seed);
        self
    }

    /// Runs the chain for `n_collect + n_discard` steps, adapting during burn-in and
    /// returning collected samples.
    ///
    /// # Parameters
    /// - `n_collect`: Number of samples to collect after adaptation.
    /// - `n_discard`: Number of burn-in steps for adaptation.
    ///
    /// # Returns
    /// A 2D tensor of shape `[n_collect, D]` containing collected samples.
    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Tensor<B, 2> {
        if n_collect == 0 {
            let dim = self.inner.position().dims()[0];
            return Tensor::<B, 2>::empty([0, dim], &B::Device::default());
        }

        let dim = self.inner.init_chain_state(n_collect, n_discard);
        let mut out = Tensor::<B, 2>::empty([n_collect, dim], &B::Device::default());
        let total = n_collect + n_discard;

        for step_idx in 0..total {
            if step_idx > 0 {
                self.inner.step();
            }
            if step_idx >= n_discard {
                let row = self.inner.position().clone().unsqueeze_dim::<2>(0);
                out.inplace(|tensor| {
                    tensor.slice_assign(
                        [step_idx - n_discard..step_idx - n_discard + 1, 0..dim],
                        row,
                    )
                });
            }
        }
        out
    }

    /// Performs one NUTS update step, including tree expansion and adaptation updates.
    ///
    /// This method updates `self.position` and adaptation statistics in-place.
    pub fn step(&mut self) {
        self.inner.step();
    }

    pub fn position(&self) -> &Tensor<B, 1> {
        self.inner.position()
    }
}

fn array3_to_tensor<B, T>(arr: ndarray::Array3<T>) -> Tensor<B, 3>
where
    B: AutodiffBackend<FloatElem = T>,
    T: Float + Element + ElementConversion,
{
    let shape = arr.raw_dim();
    let (mut data, offset) = arr.into_raw_vec_and_offset();
    if let Some(offset) = offset {
        if offset != 0 {
            data.rotate_left(offset);
        }
    }
    let td = TensorData::new(data, [shape[0], shape[1], shape[2]]);
    Tensor::<B, 3>::from_data(td, &B::Device::default())
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;
    use std::marker::PhantomData;

    use crate::{
        core::init,
        dev_tools::Timer,
        distributions::{DiffableGaussian2D, Rosenbrock2D},
        generic_nuts::find_reasonable_epsilon,
        stats::split_rhat_mean_ess,
    };

    #[cfg(feature = "csv")]
    use crate::io::csv::save_csv_tensor;

    use super::*;
    use burn::{
        backend::Autodiff,
        tensor::{Tensor, Tolerance},
    };
    use ndarray::ArrayView3;
    use num_traits::Float;

    // Use the CPU backend (NdArray) wrapped in Autodiff.
    type BackendType = Autodiff<burn::backend::NdArray<f64>>;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct StandardNormal;

    impl<T, B> GradientTarget<T, B> for StandardNormal
    where
        T: Float + Debug + ElementConversion + Element,
        B: AutodiffBackend,
    {
        fn unnorm_logp(&self, positions: Tensor<B, 1>) -> Tensor<B, 1> {
            let sq = positions.clone().powi_scalar(2);
            let half = T::from(0.5).unwrap();
            -(sq.mul_scalar(half)).sum()
        }
    }

    fn assert_tensor_approx_eq<T: Backend, F: Float + burn::tensor::Element>(
        actual: Tensor<T, 1>,
        expected: &[f64],
        tol: Tolerance<F>,
    ) {
        let a = actual.clone().to_data();
        let e = Tensor::<T, 1>::from(expected).to_data();
        a.assert_approx_eq(&e, tol);
    }

    #[test]
    fn test_find_reasonable_epsilon() {
        // Use BurnGradientTarget to wrap StandardNormal for the generic function
        let target = BurnGradientTarget::<StandardNormal, f64> {
            inner: StandardNormal,
            _marker: PhantomData,
        };
        let position = Tensor::<BackendType, 1>::from([0.0, 1.0]);
        let mom = Tensor::<BackendType, 1>::from([1.0, 0.0]);
        let epsilon: f64 = find_reasonable_epsilon(&position, &mom, &target);
        assert_eq!(epsilon, 2.0);
    }

    #[test]
    fn test_chain_1() {
        let target = DiffableGaussian2D::new([0.0_f64, 1.0], [[4.0, 2.0], [2.0, 3.0]]);
        let initial_positions = vec![0.0_f64, 1.0];
        let n_discard = 0;
        let n_collect = 1;
        let mut sampler = NUTSChain::new(target, initial_positions, 0.8).set_seed(42);
        let sample: Tensor<BackendType, 2> = sampler.run(n_collect, n_discard);
        assert_eq!(sample.dims(), [n_collect, 2]);
        let tol = Tolerance::<f64>::default()
            .set_relative(1e-5)
            .set_absolute(1e-6);
        assert_tensor_approx_eq(sample.flatten(0, 1), &[0.0, 1.0], tol);
    }

    #[test]
    fn test_chain_2() {
        let target = DiffableGaussian2D::new([0.0_f64, 1.0], [[4.0, 2.0], [2.0, 3.0]]);
        let initial_positions = vec![0.0_f64, 1.0];
        let n_discard = 3;
        let n_collect = 3;
        let mut sampler = NUTSChain::new(target, initial_positions, 0.8).set_seed(42);
        let sample: Tensor<BackendType, 2> = sampler.run(n_collect, n_discard);
        assert_eq!(sample.dims(), [n_collect, 2]);

        // Statistical assertion: samples should be finite and reasonable
        let data = sample.to_data();
        let values: &[f64] = data.as_slice().expect("dense data");
        assert!(
            values.iter().all(|v| v.is_finite()),
            "All samples should be finite"
        );
        assert!(
            values.iter().all(|v| v.abs() < 100.0),
            "Samples should be reasonable magnitude"
        );
    }

    #[test]
    fn test_chain_3() {
        let target = DiffableGaussian2D::new([1.0_f64, 2.0], [[1.0, 2.0], [2.0, 5.0]]);
        let initial_positions = vec![-2.0_f64, 1.0];
        let n_discard = 5;
        let n_collect = 5;
        let mut sampler = NUTSChain::new(target, initial_positions, 0.8).set_seed(42);
        let sample: Tensor<BackendType, 2> = sampler.run(n_collect, n_discard);
        assert_eq!(sample.dims(), [n_collect, 2]);

        // Statistical assertion: samples should be finite and reasonable
        let data = sample.to_data();
        let values: &[f64] = data.as_slice().expect("dense data");
        assert!(
            values.iter().all(|v| v.is_finite()),
            "All samples should be finite"
        );
        assert!(
            values.iter().all(|v| v.abs() < 100.0),
            "Samples should be reasonable magnitude"
        );
    }

    #[test]
    fn test_run_1() {
        let target = DiffableGaussian2D::new([1.0_f64, 2.0], [[1.0, 2.0], [2.0, 5.0]]);
        let initial_positions = vec![vec![-2_f64, 1.0]];
        let n_discard = 5;
        let n_collect = 5;
        let mut sampler = NUTS::new(target, initial_positions, 0.8).set_seed(41);
        let sample: Tensor<BackendType, 3> = sampler.run(n_collect, n_discard);
        assert_eq!(sample.dims(), [1, n_collect, 2]);

        // Statistical assertion: samples should be finite and reasonable
        let data = sample.to_data();
        let values: &[f64] = data.as_slice().expect("dense data");
        assert!(
            values.iter().all(|v| v.is_finite()),
            "All samples should be finite"
        );
        assert!(
            values.iter().all(|v| v.abs() < 100.0),
            "Samples should be reasonable magnitude"
        );
    }

    #[test]
    fn test_progress_1() {
        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // We'll define 6 chains all initialized to (1.0, 2.0).
        let initial_positions = init::<f32>(6, 2);
        let n_collect = 10;
        let n_discard = 10;

        let mut sampler =
            NUTS::<_, BackendType, _>::new(target, initial_positions, 0.95).set_seed(42);
        let (sample, stats) = sampler.run_progress(n_collect, n_discard).unwrap();
        println!(
            "NUTS sampler: generated {} observations.",
            sample.dims()[0..2].iter().product::<usize>()
        );
        assert_eq!(sample.dims(), [6, n_collect, 2]);

        println!("Statistics: {stats}");

        #[cfg(feature = "csv")]
        save_csv_tensor(sample, "/tmp/nuts-sample.csv").expect("saving data should succeed")
    }

    #[test]
    #[ignore = "Benchmark test: run only when explicitly requested"]
    fn test_bench_noprogress_1() {
        let target = Rosenbrock2D {
            a: 1.0_f64,
            b: 100.0_f64,
        };

        // We'll define 6 chains all initialized to (1.0, 2.0).
        let initial_positions = init::<f64>(6, 2);
        let n_collect = 5000;
        let n_discard = 500;

        let mut sampler = NUTS::new(target, initial_positions, 0.95).set_seed(42);
        let mut timer = Timer::new();
        let sample: Tensor<BackendType, 3> = sampler.run(n_collect, n_discard);
        timer.log(format!(
            "NUTS sampler: generated {} observations.",
            sample.dims()[0..2].iter().product::<usize>()
        ));
        assert_eq!(sample.dims(), [6, 5000, 2]);

        let data = sample.to_data();
        let array = ArrayView3::from_shape(sample.dims(), data.as_slice::<f64>().unwrap()).unwrap();
        let (split_rhat, ess) = split_rhat_mean_ess(array);
        println!(
            "MAX Split Rhat: {}",
            split_rhat.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        );
        println!(
            "MIN ESS: {}",
            ess.iter().copied().fold(f64::INFINITY, f64::min)
        );

        #[cfg(feature = "csv")]
        save_csv_tensor(sample, "/tmp/nuts-sample.csv").expect("saving data should succeed")
    }

    #[test]
    #[ignore = "Benchmark test: run only when explicitly requested"]
    fn test_bench_noprogress_2() {
        let target = Rosenbrock2D {
            a: 1.0_f64,
            b: 100.0_f64,
        };

        // We'll define 6 chains all initialized to (1.0, 2.0).
        let initial_positions = init::<f64>(6, 2);
        let n_collect = 1000;
        let n_discard = 1000;

        let mut sampler = NUTS::new(target, initial_positions, 0.95).set_seed(42);
        let mut timer = Timer::new();
        let sample: Tensor<BackendType, 3> = sampler.run(n_collect, n_discard);
        timer.log(format!(
            "NUTS sampler: generated {} observations.",
            sample.dims()[0..2].iter().product::<usize>()
        ));
        assert_eq!(sample.dims(), [6, 1000, 2]);

        let data = sample.to_data();
        let array = ArrayView3::from_shape(sample.dims(), data.as_slice::<f64>().unwrap()).unwrap();
        let (split_rhat, ess) = split_rhat_mean_ess(array);
        let min_rhat = split_rhat.iter().cloned().fold(f64::INFINITY, f64::min);
        let min_ess = ess.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("MIN Split Rhat: {}", min_rhat);
        println!("MIN ESS: {}", min_ess);

        #[cfg(feature = "csv")]
        save_csv_tensor(sample, "/tmp/nuts-sample.csv").expect("saving data should succeed")
    }
}

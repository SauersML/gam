/*!
Traits for defining continuous and discrete proposal- and target distributions.
Includes implementations of commonly used distributions.

This module is generic over the floating-point precision (e.g. `f32` or `f64`)
using [`num_traits::Float`]. It also defines several traits:
- [`Target`] for densities or PMFs we want to sample from with **Metropolis-Hastings**,
- [`GradientTarget`] for densities we want to sample from with **NUTS**,
- [`BatchedGradientTarget`] for densities we want to sample from with **HMC**,
- [`Conditional`] for densities or PMFs we want to sample from with **Gibbs Sampling**,
- [`Proposal`] for proposal mechanisms,
- [`Normalized`] for distributions that can compute a fully normalized log probability,
- [`Discrete`] for distributions over finite sets.

## Examples

```rust
use general_mcmc::distributions::{
    Gaussian2D, IsotropicGaussian, Proposal,
    Target, Normalized
};
use ndarray::{arr1, arr2};

// ----------------------
// Example: Gaussian2D (2D with full covariance)
// ----------------------
let mean = arr1(&[0.0, 0.0]);
let cov = arr2(&[[1.0, 0.0],
                [0.0, 1.0]]);
let gauss: Gaussian2D<f64> = Gaussian2D { mean, cov };

// Compute the fully normalized log-prob at (0.5, -0.5):
let logp = gauss.logp(&vec![0.5, -0.5]);
println!("Normalized log-density (2D Gaussian): {}", logp);

// ----------------------
// Example: IsotropicGaussian (any dimension)
// ----------------------
let mut proposal: IsotropicGaussian<f64> = IsotropicGaussian::new(1.0);
let current = vec![0.0, 0.0];  // dimension = 2 in this example
let candidate = proposal.sample(&current);
println!("Candidate state: {:?}", candidate);
*/

use burn::prelude::*;
use burn::tensor::Element;
use burn::tensor::backend::AutodiffBackend;
use ndarray::{Array1, Array2, NdFloat, arr1, arr2};
use num_traits::Float;
use rand::distr::Distribution as RandDistribution;
// Use rand's Distribution trait to avoid version-mismatch when rand 0.8 and 0.9 coexist in deps.
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{StandardNormal, StandardUniform};
use std::f64::consts::PI;
use std::ops::AddAssign;

/// A batched target trait for computing the unnormalized log density (and gradients) for a
/// collection of positions.
///
/// Implement this trait for your target distribution to enable gradient-based sampling.
///
/// # Type Parameters
///
/// * `T`: The floating-point type (e.g., f32 or f64).
/// * `B`: The autodiff backend from the `burn` crate.
pub trait BatchedGradientTarget<T: Float, B: AutodiffBackend> {
    /// Compute the log density for a batch of positions.
    ///
    /// # Parameters
    ///
    /// * `positions`: A tensor of shape `[n_chains, D]` representing the current positions for each chain.
    ///
    /// # Returns
    ///
    /// A 1D tensor of shape `[n_chains]` containing the log density for each chain.
    fn unnorm_logp_batch(&self, positions: Tensor<B, 2>) -> Tensor<B, 1>;
}

pub trait GradientTarget<T: Float, B: AutodiffBackend> {
    fn unnorm_logp(&self, position: Tensor<B, 1>) -> Tensor<B, 1>;

    fn unnorm_logp_and_grad(&self, position: Tensor<B, 1>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let pos = position.clone().detach().require_grad();
        let ulogp = self.unnorm_logp(pos.clone());
        let grad_inner = pos.grad(&ulogp.backward()).unwrap();
        let grad = Tensor::<B, 1>::from_inner(grad_inner);
        (ulogp, grad)
    }
}

/// A trait for generating proposals Metropolis–Hastings-like algorithms.
/// The state type `T` is typically a vector of continuous values.
pub trait Proposal<T, F: Float> {
    /// Samples a new point from q(x' | x).
    fn sample(&mut self, current: &[T]) -> Vec<T>;

    /// Evaluates log q(x' | x).
    fn logp(&self, from: &[T], to: &[T]) -> F;

    /// Returns a new instance of this proposal distribution seeded with `seed`.
    fn set_seed(self, seed: u64) -> Self;
}

/// A trait for continuous target distributions from which we want to sample.
/// The state type `T` is typically a vector of continuous values.
pub trait Target<T, F: Float> {
    /// Returns the log of the unnormalized density at `position`.
    fn unnorm_logp(&self, position: &[T]) -> F;
}

/// A trait for distributions that provide a normalized log-density (e.g. for diagnostics).
pub trait Normalized<T, F: Float> {
    /// Returns the normalized log-density at `position`.
    fn logp(&self, position: &[T]) -> F;
}

/** A trait for discrete distributions whose state is represented as an index.
 ```rust
 use general_mcmc::distributions::{Categorical, Discrete};

 // Create a categorical distribution over three categories.
 let mut cat = Categorical::new(vec![0.2f64, 0.3, 0.5]);
 let observation = cat.sample();
 println!("Sampled category: {}", observation); // E.g. 1usize

 let logp = cat.logp(observation);
 println!("Log-probability of sampled category: {}", logp); // E.g. 0.3f64
```
*/
pub trait Discrete<T: Float> {
    /// Samples an index from the distribution.
    fn sample(&mut self) -> usize;
    /// Evaluates the log-probability of the given index.
    fn logp(&self, index: usize) -> T;
}

/**
A 2D Gaussian distribution parameterized by a mean vector and a 2×2 covariance matrix.

- The generic type `T` is typically `f32` or `f64`.
- Implements both [`Target`] (for unnormalized log-prob) and
  [`Normalized`] (for fully normalized log-prob).

# Example

```rust
use general_mcmc::distributions::{Gaussian2D, Normalized};
use ndarray::{arr1, arr2};

let mean = arr1(&[0.0, 0.0]);
let cov = arr2(&[[1.0, 0.0],
                [0.0, 1.0]]);
let gauss: Gaussian2D<f64> = Gaussian2D { mean, cov };

let lp = gauss.logp(&vec![0.5, -0.5]);
println!("Normalized log probability: {}", lp);
```
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gaussian2D<T: Float> {
    pub mean: Array1<T>,
    pub cov: Array2<T>,
}

impl<T> Normalized<T, T> for Gaussian2D<T>
where
    T: NdFloat,
{
    /// Computes the fully normalized log-density of a 2D Gaussian.
    fn logp(&self, position: &[T]) -> T {
        let term_1 = -(T::from(2.0).unwrap() * T::from(PI).unwrap()).ln();
        let (a, b, c, d) = (
            self.cov[(0, 0)],
            self.cov[(0, 1)],
            self.cov[(1, 0)],
            self.cov[(1, 1)],
        );
        let det = a * d - b * c;
        let half = T::from(0.5).unwrap();
        let term_2 = -half * det.abs().ln();

        let x = arr1(position);
        let diff = x - self.mean.clone();
        let inv_cov = arr2(&[[d, -b], [-c, a]]) / det;
        let term_3 = -half * diff.dot(&inv_cov).dot(&diff);
        term_1 + term_2 + term_3
    }
}

impl<T> Target<T, T> for Gaussian2D<T>
where
    T: NdFloat,
{
    fn unnorm_logp(&self, position: &[T]) -> T {
        let (a, b, c, d) = (
            self.cov[(0, 0)],
            self.cov[(0, 1)],
            self.cov[(1, 0)],
            self.cov[(1, 1)],
        );
        let det = a * d - b * c;
        let x = arr1(position);
        let diff = x - self.mean.clone();
        let inv_cov = arr2(&[[d, -b], [-c, a]]) / det;
        -T::from(0.5).unwrap() * diff.dot(&inv_cov).dot(&diff)
    }
}

/// A 2D Gaussian target distribution, parameterized by mean and covariance.
///
/// This struct also precomputes the inverse covariance and a log-normalization
/// constant so we can quickly compute log-densities and gradients in `unnorm_logp`.
#[derive(Debug, Clone)]
pub struct DiffableGaussian2D<T: Float> {
    pub mean: [T; 2],
    pub cov: [[T; 2]; 2],
    pub inv_cov: [[T; 2]; 2],
    pub logdet_cov: T,
    pub norm_const: T,
}

impl<T> DiffableGaussian2D<T>
where
    T: Float + std::fmt::Debug + num_traits::FloatConst,
{
    /// Create a new 2D Gaussian with the specified mean and covariance.
    /// We automatically compute the covariance inverse and log-determinant.
    pub fn new(mean: [T; 2], cov: [[T; 2]; 2]) -> Self {
        // Compute determinant
        let det_cov = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
        // Inverse of a 2x2:
        // [a, b; c, d]^-1 = (1/det) [ d, -b; -c, a ]
        let inv_det = T::one() / det_cov;
        let inv_cov = [
            [cov[1][1] * inv_det, -cov[0][1] * inv_det],
            [-cov[1][0] * inv_det, cov[0][0] * inv_det],
        ];
        let logdet_cov = det_cov.ln(); // T must implement Float
        // Normalization constant for log pdf in 2 dimensions:
        //   - (1/2) * (dim * ln(2 pi) + ln(|Sigma|))
        //   = -1/2 [ 2 * ln(2*pi) + ln(det_cov) ]
        let two = T::one() + T::one();
        let norm_const = -(two * (two * T::PI()).ln() + logdet_cov) / two;

        Self {
            mean,
            cov,
            inv_cov,
            logdet_cov,
            norm_const,
        }
    }
}

#[cfg(feature = "burn")]
impl<T, B> BatchedGradientTarget<T, B> for DiffableGaussian2D<T>
where
    T: Float + burn::tensor::ElementConversion + std::fmt::Debug + burn::tensor::Element,
    B: AutodiffBackend,
{
    /// Evaluate the log probability for a batch of positions: shape [n_chains, 2].
    /// Return shape [n_chains].
    /// Note: It is not necessary to return the log probability here but for easier debugging we do so anyways.
    fn unnorm_logp_batch(&self, positions: Tensor<B, 2>) -> Tensor<B, 1> {
        let (n_chains, dim) = (positions.dims()[0], positions.dims()[1]);
        assert_eq!(dim, 2, "Gaussian2D: expected dimension=2.");

        let mean_tensor =
            Tensor::<B, 2>::from_floats([[self.mean[0], self.mean[1]]], &B::Device::default())
                .reshape([1, 2])
                .expand([n_chains, 2]);

        let delta = positions.clone() - mean_tensor;

        let inv_cov_data = [
            self.inv_cov[0][0],
            self.inv_cov[0][1],
            self.inv_cov[1][0],
            self.inv_cov[1][1],
        ];
        let inv_cov_t =
            Tensor::<B, 2>::from_floats([inv_cov_data], &B::Device::default()).reshape([2, 2]);

        let z = delta.clone().matmul(inv_cov_t); // shape [n_chains, 2]
        let quad = (z * delta).sum_dim(1).squeeze(1); // shape [n_chains]
        let shape = Shape::new([n_chains]);
        let norm_c = Tensor::<B, 1>::ones(shape, &B::Device::default()).mul_scalar(self.norm_const);
        let half = T::from(0.5).unwrap();
        norm_c - quad.mul_scalar(half)
    }
}

#[cfg(feature = "burn")]
impl<T, B> GradientTarget<T, B> for DiffableGaussian2D<T>
where
    T: Float + burn::tensor::ElementConversion + std::fmt::Debug + burn::tensor::Element,
    B: AutodiffBackend,
{
    fn unnorm_logp(&self, position: Tensor<B, 1>) -> Tensor<B, 1> {
        let dim = position.dims()[0];
        assert_eq!(dim, 2, "Gaussian2D: expected dimension=2.");

        let mean_tensor =
            Tensor::<B, 1>::from_floats([self.mean[0], self.mean[1]], &B::Device::default());

        let delta = position.clone() - mean_tensor;

        let inv_cov_data = [
            [self.inv_cov[0][0], self.inv_cov[0][1]],
            [self.inv_cov[1][0], self.inv_cov[1][1]],
        ];
        let inv_cov_t = Tensor::<B, 2>::from_floats(inv_cov_data, &B::Device::default());

        let z = delta.clone().reshape([1_i32, 2_i32]).matmul(inv_cov_t);
        let quad = (z.reshape([2_i32]) * delta).sum();
        let half = T::from(0.5).unwrap();
        -quad.mul_scalar(half) + self.norm_const
    }
}

/**
An *isotropic* Gaussian distribution usable as either a target or a proposal
in MCMC. It works for **any dimension** because it applies independent
Gaussian noise (`mean = 0`, `std = self.std`) to each coordinate.

- Implements [`Proposal`] so it can propose new states
  from a current state.
- Also implements [`Target`] for an unnormalized log-prob,
  which might be useful if you want to treat it as a target distribution
  in simplified scenarios.

# Examples

```rust
use general_mcmc::distributions::{IsotropicGaussian, Proposal};

let mut proposal: IsotropicGaussian<f64> = IsotropicGaussian::new(1.0);
let current = vec![0.0, 0.0, 0.0]; // dimension = 3
let candidate = proposal.sample(&current);
println!("Candidate state: {:?}", candidate);

// Evaluate log q(candidate | current):
let logq = proposal.logp(&current, &candidate);
println!("Log of the proposal density: {}", logq);
```
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsotropicGaussian<T: Float> {
    pub std: T,
    rng: SmallRng,
}

impl<T: Float> IsotropicGaussian<T> {
    /// Creates a new isotropic Gaussian proposal distribution with the specified standard deviation.
    pub fn new(std: T) -> Self {
        Self {
            std,
            rng: SmallRng::seed_from_u64(rand::rng().random::<u64>()),
        }
    }
}

impl<T: Float + std::ops::AddAssign> Proposal<T, T> for IsotropicGaussian<T>
where
    StandardNormal: RandDistribution<T>,
{
    fn sample(&mut self, current: &[T]) -> Vec<T> {
        current
            .iter()
            .map(|eps| {
                let noise: T = self.rng.sample(StandardNormal);
                *eps + noise * self.std
            })
            .collect()
    }

    fn logp(&self, from: &[T], to: &[T]) -> T {
        let mut lp = T::zero();
        let d = T::from(from.len()).unwrap();
        let two = T::from(2).unwrap();
        let var = self.std * self.std;
        for (&f, &t) in from.iter().zip(to.iter()) {
            let diff = t - f;
            let exponent = -(diff * diff) / (two * var);
            lp += exponent;
        }
        lp += -d * T::from(0.5).unwrap() * (var * T::from(PI).unwrap() * self.std * self.std).ln();
        lp
    }

    fn set_seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self
    }
}

impl<T: Float> Target<T, T> for IsotropicGaussian<T> {
    fn unnorm_logp(&self, position: &[T]) -> T {
        let mut sum = T::zero();
        for &x in position.iter() {
            sum = sum + x * x
        }
        -T::from(0.5).unwrap() * sum / (self.std * self.std)
    }
}

/**
A categorical distribution represents a discrete probability distribution over a finite set of categories.

The probabilities in `probs` should sum to 1 (or they will be normalized automatically).

# Examples

```rust
use general_mcmc::distributions::{Categorical, Discrete};

let mut cat = Categorical::new(vec![0.2f64, 0.3, 0.5]);
let observation = cat.sample();
println!("Sampled category: {}", observation);
let logp = cat.logp(observation);
println!("Log probability of category {}: {}", observation, logp);
```
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Categorical<T>
where
    T: Float + std::ops::AddAssign,
{
    pub probs: Vec<T>,
    rng: SmallRng,
}

impl<T: Float + std::ops::AddAssign> Categorical<T> {
    /// Creates a new categorical distribution from a vector of probabilities.
    /// The probabilities will be normalized so that they sum to 1.
    pub fn new(probs: Vec<T>) -> Self {
        let sum: T = probs.iter().cloned().fold(T::zero(), |acc, x| acc + x);
        let normalized: Vec<T> = probs.into_iter().map(|p| p / sum).collect();
        Self {
            probs: normalized,
            rng: SmallRng::seed_from_u64(rand::rng().random::<u64>()),
        }
    }
}

impl<T: Float + std::ops::AddAssign> Discrete<T> for Categorical<T>
where
    StandardUniform: RandDistribution<T>,
{
    fn sample(&mut self) -> usize {
        let r: T = self.rng.random();
        let mut cum: T = T::zero();
        let mut k = self.probs.len() - 1;
        for (i, &p) in self.probs.iter().enumerate() {
            cum += p;
            if r <= cum {
                k = i;
                break;
            }
        }
        k
    }

    fn logp(&self, index: usize) -> T {
        if index < self.probs.len() {
            self.probs[index].ln()
        } else {
            T::neg_infinity()
        }
    }
}

impl<T: Float + AddAssign> Target<usize, T> for Categorical<T>
where
    StandardUniform: RandDistribution<T>,
{
    fn unnorm_logp(&self, position: &[usize]) -> T {
        <Self as Discrete<T>>::logp(self, position[0])
    }
}

/**
A trait for conditional distributions.

This trait specifies how to sample a single coordinate of a state given the entire current state.
It is primarily used in Gibbs sampling to update one coordinate at a time.
*/
pub trait Conditional<S> {
    fn sample(&mut self, index: usize, given: &[S]) -> S;
}

// Define the Rosenbrock distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rosenbrock2D<T: Float> {
    pub a: T,
    pub b: T,
}

// For the batched version we need to implement BatchGradientTarget.
#[cfg(feature = "burn")]
impl<T, B> BatchedGradientTarget<T, B> for Rosenbrock2D<T>
where
    T: Float + Element,
    B: AutodiffBackend,
{
    fn unnorm_logp_batch(&self, positions: Tensor<B, 2>) -> Tensor<B, 1> {
        let n = positions.dims()[0];
        let x = positions.clone().slice([0..n, 0..1]);
        let y = positions.slice([0..n, 1..2]);
        let term_1 = (-x.clone()).add_scalar(self.a).powi_scalar(2);
        let term_2 = y.sub(x.powi_scalar(2)).powi_scalar(2).mul_scalar(self.b);
        -(term_1 + term_2).flatten(0, 1)
    }
}

#[cfg(feature = "burn")]
impl<T, B> GradientTarget<T, B> for Rosenbrock2D<T>
where
    T: Float + Element,
    B: AutodiffBackend,
{
    fn unnorm_logp(&self, position: Tensor<B, 1>) -> Tensor<B, 1> {
        let x = position.clone().slice(s![0..1]);
        let y = position.slice(s![1..2]);
        let term_1 = (-x.clone()).add_scalar(self.a).powi_scalar(2);
        let term_2 = y.sub(x.powi_scalar(2)).powi_scalar(2).mul_scalar(self.b);
        -(term_1 + term_2)
    }
}

// Define the Rosenbrock distribution.
// From: https://arxiv.org/pdf/1903.09556.
#[derive(Clone)]
pub struct RosenbrockND {}

// For the batched version we need to implement BatchGradientTarget.
#[cfg(feature = "burn")]
impl<T, B> BatchedGradientTarget<T, B> for RosenbrockND
where
    T: Float + Element,
    B: AutodiffBackend,
{
    fn unnorm_logp_batch(&self, positions: Tensor<B, 2>) -> Tensor<B, 1> {
        let k = positions.dims()[0];
        let n = positions.dims()[1];
        let low = positions.clone().slice([0..k, 0..(n - 1)]);
        let high = positions.slice([0..k, 1..n]);
        let term_1 = (high - low.clone().powi_scalar(2))
            .powi_scalar(2)
            .mul_scalar(100);
        let term_2 = low.neg().add_scalar(1).powi_scalar(2);
        -(term_1 + term_2).sum_dim(1).squeeze(1)
    }
}

#[cfg(test)]
mod continuous_tests {
    use super::*;

    /**
    A helper function to normalize the unnormalized log probability of an isotropic Gaussian
    into a proper probability value (by applying the appropriate constant).

    # Arguments

    * `x` - The unnormalized log probability.
    * `d` - The dimensionality of the state.
    * `std` - The standard deviation used in the isotropic Gaussian.

    # Returns

    Returns the normalized probability as an `f64`.
    */
    fn normalize_isogauss(x: f64, d: usize, std: f64) -> f64 {
        let log_normalizer = -((d as f64) / 2.0) * ((2.0_f64).ln() + PI.ln() + 2.0 * std.ln());
        (x + log_normalizer).exp()
    }

    #[test]
    fn iso_gauss_unnorm_logp_test_1() {
        let distr = IsotropicGaussian::new(1.0);
        let p = normalize_isogauss(distr.unnorm_logp(&[1.0]), 1, distr.std);
        let true_p = 0.24197072451914337;
        let diff = (p - true_p).abs();
        assert!(
            diff < 1e-7,
            "Expected diff < 1e-7, got {diff} with p={p} (expected ~{true_p})."
        );
    }

    #[test]
    fn iso_gauss_unnorm_logp_test_2() {
        let distr = IsotropicGaussian::new(2.0);
        let p = normalize_isogauss(distr.unnorm_logp(&[0.42, 9.6]), 2, distr.std);
        let true_p = 3.864661987252467e-7;
        let diff = (p - true_p).abs();
        assert!(
            diff < 1e-15,
            "Expected diff < 1e-15, got {diff} with p={p} (expected ~{true_p})"
        );
    }

    #[test]
    fn iso_gauss_unnorm_logp_test_3() {
        let distr = IsotropicGaussian::new(3.0);
        let p = normalize_isogauss(distr.unnorm_logp(&[1.0, 2.0, 3.0]), 3, distr.std);
        let true_p = 0.001080393185560214;
        let diff = (p - true_p).abs();
        assert!(
            diff < 1e-8,
            "Expected diff < 1e-8, got {diff} with p={p} (expected ~{true_p})"
        );
    }
}

#[cfg(test)]
mod categorical_tests {
    use super::*;

    /// A helper function to compare floating-point values with a given tolerance.
    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ------------------------------------------------------
    // 1) Test logp correctness for f64
    // ------------------------------------------------------
    #[test]
    fn test_categorical_logp_f64() {
        let probs = vec![0.2, 0.3, 0.5];
        let cat = Categorical::<f64>::new(probs.clone());

        // Check log probabilities for each index
        let logp_0 = cat.logp(0);
        let logp_1 = cat.logp(1);
        let logp_2 = cat.logp(2);

        // Expected values
        let expected_0 = 0.2_f64.ln();
        let expected_1 = 0.3_f64.ln();
        let expected_2 = 0.5_f64.ln();

        let tol = 1e-7;
        assert!(
            approx_eq(logp_0, expected_0, tol),
            "Log prob mismatch at index 0: got {}, expected {}",
            logp_0,
            expected_0
        );
        assert!(
            approx_eq(logp_1, expected_1, tol),
            "Log prob mismatch at index 1: got {}, expected {}",
            logp_1,
            expected_1
        );
        assert!(
            approx_eq(logp_2, expected_2, tol),
            "Log prob mismatch at index 2: got {}, expected {}",
            logp_2,
            expected_2
        );

        // Out-of-bounds index should be NEG_INFINITY
        let logp_out = cat.logp(3);
        assert_eq!(
            logp_out,
            f64::NEG_INFINITY,
            "Out-of-bounds index did not return NEG_INFINITY"
        );
    }

    // ------------------------------------------------------
    // 2) Test sampling frequencies for f64
    // ------------------------------------------------------
    #[test]
    fn test_categorical_sampling_f64() {
        let probs = vec![0.2, 0.3, 0.5];
        let mut cat = Categorical::<f64>::new(probs.clone());

        let sample_size = 100_000;
        let mut counts = vec![0_usize; probs.len()];

        // Draw observations and tally outcomes
        for _ in 0..sample_size {
            let observation = cat.sample();
            counts[observation] += 1;
        }

        // Check empirical frequencies
        let tol = 0.01; // 1% absolute tolerance
        for (i, &count) in counts.iter().enumerate() {
            let freq = count as f64 / sample_size as f64;
            let expected = probs[i];
            assert!(
                approx_eq(freq, expected, tol),
                "Empirical freq for index {} is off: got {:.3}, expected {:.3}",
                i,
                freq,
                expected
            );
        }
    }

    // ------------------------------------------------------
    // 3) Test logp correctness for f32
    // ------------------------------------------------------
    #[test]
    fn test_categorical_logp_f32() {
        let probs = vec![0.1_f32, 0.4, 0.5];
        let cat = Categorical::<f32>::new(probs.clone());

        let logp_0: f32 = cat.logp(0);
        let logp_1 = cat.logp(1);
        let logp_2 = cat.logp(2);

        // For comparison, cast to f64
        let expected_0 = (0.1_f64).ln();
        let expected_1 = (0.4_f64).ln();
        let expected_2 = (0.5_f64).ln();

        let tol = 1e-6;
        assert!(
            approx_eq(logp_0.into(), expected_0, tol),
            "Log prob mismatch at index 0 (f32 -> f64 cast)"
        );
        assert!(
            approx_eq(logp_1.into(), expected_1, tol),
            "Log prob mismatch at index 1"
        );
        assert!(
            approx_eq(logp_2.into(), expected_2, tol),
            "Log prob mismatch at index 2"
        );

        // Out-of-bounds
        let logp_out = cat.logp(3);
        assert_eq!(logp_out, f32::NEG_INFINITY);
    }

    // ------------------------------------------------------
    // 4) Test sampling frequencies for f32
    // ------------------------------------------------------
    #[test]
    fn test_categorical_sampling_f32() {
        let probs = vec![0.1_f32, 0.4, 0.5];
        let mut cat = Categorical::<f32>::new(probs.clone());

        let sample_size = 100_000;
        let mut counts = vec![0_usize; probs.len()];

        for _ in 0..sample_size {
            let observation = cat.sample();
            counts[observation] += 1;
        }

        // Compare frequencies with expected probabilities
        let tol = 0.02; // might relax tolerance for f32
        for (i, &count) in counts.iter().enumerate() {
            let freq = count as f32 / sample_size as f32;
            let expected = probs[i];
            assert!(
                (freq - expected).abs() < tol,
                "Empirical freq for index {} is off: got {:.3}, expected {:.3}",
                i,
                freq,
                expected
            );
        }
    }

    #[test]
    fn test_categorical_sample_single_value() {
        let mut cat = Categorical {
            probs: vec![1.0_f64],
            rng: rand::rngs::SmallRng::from_seed(Default::default()),
        };

        let sampled_index = cat.sample();

        assert_eq!(
            sampled_index, 0,
            "Should return the last index (0) for a single-element vector"
        );
    }

    #[test]
    fn test_target_for_categorical_in_range() {
        // Create a categorical distribution with known probabilities.
        let probs = vec![0.2_f64, 0.3, 0.5];
        let cat = Categorical::new(probs.clone());
        // Call unnorm_logp with a valid index (say, index 1).
        let logp = cat.unnorm_logp(&[1]);
        // The expected log probability is ln(0.3).
        let expected = 0.3_f64.ln();
        let tol = 1e-7;
        assert!(
            (logp - expected).abs() < tol,
            "For index 1, expected ln(0.3) ~ {}, got {}",
            expected,
            logp
        );
    }

    #[test]
    fn test_target_for_categorical_out_of_range() {
        let probs = vec![0.2_f64, 0.3, 0.5];
        let cat = Categorical::new(probs);
        // Calling unnorm_logp with an index that's out of bounds (e.g. 3)
        // should return negative infinity.
        let logp = cat.unnorm_logp(&[3]);
        assert_eq!(
            logp,
            f64::NEG_INFINITY,
            "Expected negative infinity for out-of-range index, got {}",
            logp
        );
    }

    #[test]
    fn test_gaussian2d_logp() {
        let mean = arr1(&[0.0, 0.0]);
        let cov = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let gauss = Gaussian2D { mean, cov };

        let position = vec![0.5, -0.5];
        let computed_logp = gauss.logp(&position);

        let expected_logp = -2.0878770664093453;

        let tol = 1e-10;
        assert!(
            (computed_logp - expected_logp).abs() < tol,
            "Computed log density ({}) differs from expected ({}) by more than tolerance ({})",
            computed_logp,
            expected_logp,
            tol
        );
    }
}

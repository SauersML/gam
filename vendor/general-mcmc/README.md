# General MCMC

[![crate](https://img.shields.io/crates/v/general-mcmc.svg)](https://crates.io/crates/general-mcmc)
[![docs](https://img.shields.io/docsrs/general-mcmc)](https://docs.rs/general-mcmc)

A compact Rust library for **Markov Chain Monte Carlo (MCMC)** methods with GPU support.

This is a modified version of https://github.com/MatteoGaetzner/mini-mcmc.

## Overview

This library provides implementations of

- **No-U-Turn Sampler (NUTS)**: An extension of HMC that removes the need to choose path lengths.
- **Hamiltonian Monte Carlo (HMC)**: an MCMC method that efficiently samples by simulating Hamiltonian dynamics using gradients of the target distribution.
- **Gibbs Sampling**: an MCMC method that iteratively samples each variable from its conditional distribution given all other variables.
- **Metropolis-Hastings**: an MCMC algorithm that samples from a distribution by proposing candidates and probabilistically accepting or rejecting them.

Additional features:

- **Implementations of Common Distributions**: featuring handy Gaussian and isotropic Gaussian implementations, along with traits for defining custom log-prob functions.
- **Parallelization**: for running multiple Markov chains in parallel.
- **Progress Bars**: that show progress of MCMC algorithms with convergence
  statistics and acceptance rates.
- **Support for Discrete & Continuous Distributions**: for example, Metropolis-Hastings- and Gibbs Samplers can sample from continuous and discrete target distributions.
- **Generic Datatypes**: enable sampling of vectors with various integer or floating point types.
- **Standard Convergence Diagnostics**: estimate the effective sample size and Rhat.

## Installation

To add the latest version of general-mcmc to your project:

```bash
cargo add general-mcmc
```

Then `use general_mcmc` in your Rust code.

## Example: Sampling From a 2D Gaussian

```rust
use general_mcmc::core::ChainRunner;
use general_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
use general_mcmc::metropolis_hastings::MetropolisHastings;
use ndarray::{arr1, arr2};

fn main() {
    let target = Gaussian2D {
        mean: arr1(&[0.0, 0.0]),
        cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
    };
    let proposal = IsotropicGaussian::new(1.0);
    let initial_states = vec![vec![0.0, 0.0]; 4];  // 4 chains, each starting at [0,0]

    // Create a MH sampler with 4 parallel chains
    let mut mh = MetropolisHastings::new(target, proposal, initial_states);

    // Run the sampler for 1,100 steps, discarding the first 100 as burn-in
    let (samples, stats) = mh.run_progress(1000, 100).unwrap();

    // Print convergence statistics
    println!("{stats}");

    // We should have 4 chains, each with 1000 samples of 2 dimensions
    assert_eq!(samples.shape(), [4, 1000, 2]);
}
```

You can also find this example at `examples/minimal_mh.rs`.

## Example: Sampling From a Custom Distribution

Below we define a custom Poisson distribution for nonnegative integer states $\{0,1,2,\dots\}$ and a basic random-walk proposal. We then run Metropolis–Hastings to sample from this distribution, collecting frequencies of $k$ after some burn-in:

```rust
use general_mcmc::core::ChainRunner;
use general_mcmc::distributions::{Proposal, Target};
use general_mcmc::metropolis_hastings::MetropolisHastings;
use plotly::{Bar, Layout};
use rand::Rng;
use std::error::Error;

/// A Poisson(\lambda) distribution, seen as a discrete target over k=0,1,2,...
#[derive(Clone)]
struct PoissonTarget {
    lambda: f64,
}

impl Target<usize, f64> for PoissonTarget {
    /// unnorm_logp(k) = log( p(k) ), ignoring normalizing constants if you wish.
    /// For Poisson(k|lambda) = exp(-lambda) * (lambda^k / k!)
    /// so log p(k) = -lambda + k*ln(lambda) - ln(k!)
    /// which is enough to do MH acceptance.
    fn unnorm_logp(&self, theta: &[usize]) -> f64 {
        let k = theta[0];
        let kf = k as f64;
        // If you like, you can omit -ln(k!) if you only need "unnormalized"—but including
        // it can improve acceptance ratio numerically. Here we keep the full log pmf.
        -self.lambda + kf * self.lambda.ln() - ln_factorial(k as u64)
    }
}

/// A simple random-walk proposal in the nonnegative integers:
/// - If current_state=0, propose 0 -> 1 always
/// - Otherwise propose x->x+1 or x->x-1 with p=0.5 each
#[derive(Clone)]
struct NonnegativeProposal;

impl Proposal<usize, f64> for NonnegativeProposal {
    fn sample(&mut self, current: &[usize]) -> Vec<usize> {
        let x = current[0];
        if x == 0 {
            // can't go negative; always move to 1
            vec![1]
        } else {
            // 50% chance to do x+1, 50% x-1
            let flip = rand::rng().random_bool(0.5);
            let next = if flip { x + 1 } else { x - 1 };
            vec![next]
        }
    }

    /// logp(x->y):
    ///  - if x=0 and y=1, p=1 => log p=0
    ///  - if x>0, then y in {x+1, x-1} => p=0.5 => log(0.5)
    ///  - otherwise => -∞ (impossible transition)
    fn logp(&self, from: &[usize], to: &[usize]) -> f64 {
        let x = from[0];
        let y = to[0];
        if x == 0 {
            if y == 1 {
                0.0 // ln(1.0)
            } else {
                f64::NEG_INFINITY
            }
        } else {
            // x>0
            if y == x + 1 || y + 1 == x {
                // y in {x+1, x-1} => prob=0.5 => ln(0.5)
                (0.5_f64).ln()
            } else {
                f64::NEG_INFINITY
            }
        }
    }

    fn set_seed(self, _seed: u64) -> Self {
        // no custom seeding logic here
        self
    }
}

// A small helper for computing ln(k!)
fn ln_factorial(k: u64) -> f64 {
    if k < 2 {
        0.0
    } else {
        let mut acc = 0.0;
        for i in 1..=k {
            acc += (i as f64).ln();
        }
        acc
    }
}

// Helper function to compute Poisson PMF
fn poisson_pmf(k: usize, lambda: f64) -> f64 {
    (-lambda + (k as f64) * lambda.ln() - ln_factorial(k as u64)).exp()
}

fn main() -> Result<(), Box<dyn Error>> {
    // We'll do Poisson with lambda=4.0, for instance
    let target = PoissonTarget { lambda: 4.0 };

    // We'll have a random-walk in nonnegative integers
    let proposal = NonnegativeProposal;

    // Start the chain at k=0
    let initial_state = vec![vec![0usize]];

    // Create Metropolis–Hastings with 1 chain (or more, up to you)
    let mut mh = MetropolisHastings::new(target, proposal, initial_state);

    // Collect 10,000 samples and use 1,000 for burn-in (not returned).
    let samples = mh
        .run(10_000, 1_000)
        .expect("Expected generating samples to succeed");
    let chain0 = samples.to_shape(10_000).unwrap();
    println!("Elements in chain: {}", chain0.len());

    // Tally frequencies of each k up to some cutoff
    let cutoff = 20; // enough to see the mass near lambda=4
    let mut counts = vec![0usize; cutoff + 1];
    for row in chain0.iter() {
        let k = *row;
        if k <= cutoff {
            counts[k] += 1;
        }
    }

    let total = chain0.len();
    println!("Frequencies for k=0..{cutoff}, from chain after burn-in:");
    for (k, &cnt) in counts.iter().enumerate() {
        let freq = cnt as f64 / total as f64;
        println!("k={k:2}: freq ~ {freq:.3}");
    }

    // Create x-axis values (k values)
    let k_values: Vec<usize> = (0..=cutoff).collect();

    // Create empirical frequencies
    let empirical_freqs: Vec<f64> = counts
        .iter()
        .map(|&cnt| cnt as f64 / total as f64)
        .collect();

    // Create theoretical PMF values
    let theoretical_pmf: Vec<f64> = k_values.iter().map(|&k| poisson_pmf(k, 4.0)).collect();

    // Create bar plot for empirical frequencies
    let empirical_trace = Bar::new(k_values.clone(), empirical_freqs)
        .name("Empirical")
        .marker(
            plotly::common::Marker::new()
                .color("rgb(70, 130, 180)")
                .opacity(0.7),
        );

    // Create bar plot for theoretical PMF
    let theoretical_trace = Bar::new(k_values, theoretical_pmf)
        .name("Theoretical")
        .marker(
            plotly::common::Marker::new()
                .color("rgb(255, 127, 14)")
                .opacity(0.7),
        );

    // Create layout
    let layout = Layout::new()
        .title(plotly::common::Title::new())
        .x_axis(
            plotly::layout::Axis::new()
                .title("k")
                .zero_line(true)
                .grid_color("rgb(200, 200, 200)"),
        )
        .y_axis(
            plotly::layout::Axis::new()
                .title("Probability")
                .zero_line(true)
                .grid_color("rgb(200, 200, 200)"),
        )
        .show_legend(true)
        .plot_background_color("rgb(250, 250, 250)")
        .width(800)
        .height(600);

    // Create and save plot
    let mut plot = plotly::Plot::new();
    plot.add_trace(empirical_trace);
    plot.add_trace(theoretical_trace);
    plot.set_layout(layout);
    plot.write_html("poisson_distribution.html");
    println!("Saved plot to poisson_distribution.html");

    println!("Done sampling Poisson(4).");
    Ok(())
}
```

You can also find this example at `examples/poisson_mh.rs`.

### Explanation

- **`PoissonTarget`** implements `Target<usize, f64>` for a discrete Poisson($\lambda$) distribution:\
  $$p(k) = e^{-\lambda} \frac{\lambda^k}{k!},\quad k=0,1,2,\ldots$$\
  The log form of it is $\log p(k) = -\lambda + k \log \lambda - \log k!$.

- **`NonnegativeProposal`** provides a random-walk in the set $\{0,1,2,\dots\}$:
  - If $x=0$, propose $1$ with probability $1$.
  - If $x>0$, propose $x+1$ or $x-1$ with probability $0.5$ each.
  - `logp` returns $\ln(0.5)$ for the possible moves, or $-\infty$ for
    impossible moves.

- **Usage**:  
  We start the chain at $k=0$, run 11,000 iterations discarding 1,000 as burn-in, and tally the final sample frequencies for $k=0 \dots 20$. They should approximate the Poisson(4.0) distribution (peak around $k=4$).

With this example, you can see how to use **general_mcmc** for **unbounded** discrete distributions via a custom random-walk proposal and a log‐PMF.

Below is an additional documentation section that you can add to your README. It first gives a minimal version of the `rosenbrock3d_hmc.rs` example for sampling using HMC. (Note that the full example also plots the sampled data interactively using Plotly.)

---

## Example: Sampling from a 3D Rosenbrock Distribution Using HMC

The following minimal example demonstrates how to create and run an HMC sampler to sample from a 3D Rosenbrock distribution. In this example, we construct an HMC sampler, run it for a fixed number of iterations, and print the shape of the collected samples. The corresponding file can also be found at [`examples/minimal_hmc.rs`](examples/minimal_hmc.rs). For a complete example—including interactive 3D plotting with Plotly, refer to [`examples/rosenbrock3d_hmc.rs`](examples/rosenbrock3d_hmc.rs).

```rust
use burn::tensor::Element;
use burn::{backend::Autodiff, prelude::Tensor};
use general_mcmc::core::init_det;
use general_mcmc::distributions::BatchedGradientTarget;
use general_mcmc::hmc::HMC;
use num_traits::Float;

/// The 3D Rosenbrock distribution.
///
/// For a point x = (x₁, x₂, x₃), the log probability is defined as the negative of
/// the sum of two Rosenbrock terms:
///
///   f(x) = 100*(x₂ - x₁²)² + (1 - x₁)² + 100*(x₃ - x₂²)² + (1 - x₂)²
///
/// This implementation generalizes to d dimensions, but here we use it for 3D.
struct RosenbrockND {}

impl<T, B> BatchedGradientTarget<T, B> for RosenbrockND
where
    T: Float + std::fmt::Debug + Element,
    B: burn::tensor::backend::AutodiffBackend,
{
    fn unnorm_logp_batch(&self, positions: Tensor<B, 2>) -> Tensor<B, 1> {
        // Assume positions has shape [n_chains, d] with d = 3.
        let k = positions.dims()[0] as i64;
        let n = positions.dims()[1] as i64;
        let low = positions.clone().slice([0..k, 0..n - 1]);
        let high = positions.clone().slice([0..k, 1..n]);
        let term_1 = (high - low.clone().powi_scalar(2))
            .powi_scalar(2)
            .mul_scalar(100);
        let term_2 = low.neg().add_scalar(1).powi_scalar(2);
        -(term_1 + term_2).sum_dim(1).squeeze(1)
    }
}

fn main() {
    // Use the CPU backend wrapped in Autodiff (e.g., NdArray).
    type BackendType = Autodiff<burn::backend::NdArray>;

    // Create the 3D Rosenbrock target.
    let target = RosenbrockND {};

    // Create the HMC sampler with a step size of 0.01 and 50 leapfrog steps.
    let mut sampler = HMC::<f32, BackendType, RosenbrockND>::new(target, init_det(4, 3), 0.032, 10);

    // Run the sampler for 1000 iterations, discard 100
    let samples = sampler.run(400, 50);

    // Print the shape of the collected samples.
    println!("Collected samples with shape: {:?}", samples.dims());
}
```

## Example: Sampling with NUTS

The following minimal example shows how to set up and run a NUTS sampler on a 2D Rosenbrock target with 4 parallel chains. It demonstrates initializing the sampler and invoking `.run(...)` to generate samples.

```rust
use burn::backend::Autodiff;
use burn::prelude::Tensor;
use general_mcmc::core::init;
use general_mcmc::distributions::Rosenbrock2D;
use general_mcmc::nuts::NUTS;

fn main() {
    type BackendType = Autodiff<burn::backend::NdArray>;

    // Create the 2D Rosenbrock target (a = 1, b = 100).
    // To sample from a custom distribution implement
    // general_mcmc::distributions::GradientTarget for your CustomStruct
    let target = Rosenbrock2D { a: 1.0_f32, b: 100.0_f32 };

    let initial_positions = init::<f32>(4, 2);
    let mut sampler = NUTS::new(target, initial_positions, 0.95);
    let n_collect = 400;
    let n_discard = 400;

    // Run the sampler for n_discard burn-in steps, then collect n_collect observations.
    let sample: Tensor<BackendType, 3> = sampler.run(n_collect, n_discard);

    assert_eq!(sample.dims(), [4, 400, 2]);

    // Alternative: Run with progress bars and return additional statistics
    // let (sample, stats) = sampler.run_progress(n_collect, n_discard).unwrap();
    // println!("{stats}");
}
```

You can find this example with some additional logging in [`examples/minimal_nuts.rs`](examples/minimal_nuts.rs).

## Roadmap

- **Rank Normalized Rhat**: Modern convergence diagnostic, see [paper](https://arxiv.org/abs/1903.08008).
- **Ensemble Slice Sampling (ESS)**: Efficient gradient-free sampler, see [paper](https://arxiv.org/abs/2002.06212).
- **Effective Size Estimation**: Online estimation of effective sample size for
  early stopping.

## Structure

- **`src/lib.rs`**: The main library entry point—exports MCMC functionality.
- **`src/distributions.rs`**: Target distributions (e.g., multivariate Gaussians) and proposal distributions.
- **`src/nuts.rs`**: The NUTS algorithm implementation.
- **`src/hmc.rs`**: The HMC algorithm implementation.
- **`src/gibbs.rs`**: The Gibbs sampling algorithm implementation.
- **`src/metropolis_hastings.rs`**: The Metropolis-Hastings algorithm implementation.
- **`examples/`**: Examples on how to use this library.
- **`src/io/arrow.rs`**: Helper functions for saving samples as Apache Arrow files. Enable via `arrow` feature.
- **`src/io/parquet.rs`**: Helper functions for saving samples as Apache Parquet files. Enable via `parquet` feature.
- **`src/io/csv.rs`**: Helper functions for saving samples as CSV files. Enable via `csv` feature.

## Usage (Local)

1. **Build** (Library + Demo):

   ```sh
   cargo build --release
   ```

2. **Run the Demo**:
   ```sh
   cargo run --release --example gauss_mh --features parquet
   ```
   Prints basic statistics of the MCMC chain (e.g., estimated mean).
   Saves a scatter plot of sampled points in `scatter_plot.png` and a Parquet file `samples.parquet`.

## Optional Features

- `csv`: Enables CSV I/O for samples.
- `arrow` / `parquet`: Enables Apache Arrow / Parquet I/O.
- `wgpu`: Enables GPU accelerated sampling for gradient based samplers using burn's WGPU backend.
  In the HMC example above, you only have to replace line
  ```rust
  type BackendType = Autodiff<burn::backend::NdArray>;
  ```
  with
  ```rust
  type BackendType = Autodiff<burn::backend::Wgpu>;
  ```
  Depending on the number of parallel chains, dimensionality of your
  sample space and complexity of evaluating the unnormalized log density of
  your target distribution it might be more efficient to stick with the CPU (NdArray) backend.
- By default, all features are **disabled**.

## Progress Tracking and Statistics

The `run_progress` method returns both the sample and an convergence statistics object. It also displays progress bars during the run.

```rust
let (samples, stats) = mh.run_progress(1000, 100).unwrap();
println!("{stats}");
```

The `RunStats` object contains statistics about:

- Potential scale reduction factor (R-hat)
- Effective sample size (ESS)

and potentially further metrics in the future.

## Acknowledgements

The NUTS implementation in this crate is inspired by and borrows heavily from Mathieu Fouesneau’s reference implementation ([mfouesneau/NUTS](https://github.com/mfouesneau/NUTS)) and from the original algorithm as described in Hoffman & Gelman, "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo" (JMLR, 2014).

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

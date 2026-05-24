//! Computation and tracking of MCMC statistics like acceptance probability and Potential Scale
//! Reduction.

use core::fmt;
use ndarray::{Zip, concatenate, prelude::*};
use num_traits::{Num, ToPrimitive};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::{cmp::Ordering, error::Error};

const ALPHA: f32 = 0.01;

/// Tracks statistics for a single MCMC chain.
///
/// # Fields
/// - `n_params`: Number of parameters in the chain.
/// - `n`: Number of steps taken.
/// - `p_accept`: Acceptance probability.
/// - `mean`: Mean of the parameters.
/// - `mean_sq`: Mean of the squared parameters.
/// - `last_state`: Last state of the chain.
/// - `accept_queue`: Queue tracking acceptance history.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainTracker {
    n_params: usize,
    n: u64,
    p_accept: f32,
    last_state: Array1<f32>,
    mean: Array1<f32>,    // n_params
    mean_sq: Array1<f32>, // n_params
}

/// Statistics of an MCMC chain.
///
/// # Fields
/// - `n`: Number of steps taken.
/// - `p_accept`: Acceptance probability.
/// - `mean`: Mean of the parameters.
/// - `sm2`: Variance of the parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainStats {
    pub n: u64,
    pub p_accept: f32,
    pub mean: Array1<f32>, // n_params
    pub sm2: Array1<f32>,  // n_params
}

impl ChainTracker {
    /// Creates a new `ChainTracker` with the given number of parameters and initial state.
    ///
    /// # Arguments
    /// - `n_params`: Number of parameters in the chain.
    /// - `initial_state`: Initial state of the chain.
    ///
    /// # Returns
    /// A new `ChainTracker` instance.
    pub fn new<T>(n_params: usize, initial_state: &[T]) -> Self
    where
        T: num_traits::ToPrimitive + Clone,
    {
        let mean_sq = Array1::<f32>::zeros(n_params);
        let mean = Array1::<f32>::zeros(n_params);
        let last_state = ArrayView1::from_shape(n_params, initial_state)
            .expect("Expected being able to convert initial state to a NdArray")
            .mapv(|x| {
                x.to_f32()
                    .expect("Expected conversion of elements to f32's to succeed")
            });

        Self {
            n_params,
            n: 0,
            p_accept: -1.0,
            last_state,
            mean,
            mean_sq,
        }
    }

    /// Updates the tracker with a new state.
    ///
    /// # Arguments
    /// - `x`: New state of the chain.
    ///
    /// # Returns
    /// `Ok(())` if successful; an error otherwise.
    pub fn step<T>(&mut self, x: &[T]) -> Result<(), Box<dyn Error>>
    where
        T: num_traits::ToPrimitive + Clone,
    {
        self.n += 1;

        let n = self.n as f32;
        let x_arr =
            ndarray::ArrayView1::<T>::from_shape(self.n_params, x)?.mapv(|x| x.to_f32().unwrap());

        self.mean = (self.mean.clone() * (n - 1.0) + x_arr.clone()) / n;
        if self.n == 1 {
            self.mean_sq = x_arr.pow2();
        } else {
            self.mean_sq = (self.mean_sq.clone() * (n - 1.0) + (x_arr.pow2())) / n;
        };

        //  x_1 = (1 - a) x_0 + a x_1
        // <=> x_1 (1 - a) = (1 - a) x_0
        // <=> x_1 = x_0
        // So set initial p_accept to 1 if the transition state was an 'accept' and 0 otherwise
        let p_start = if self.p_accept >= 0.0 {
            self.p_accept
        } else {
            x_arr
                .index_axis(Axis(0), 0)
                .ne(&self.last_state.index_axis(Axis(0), 0)) as i32 as f32
        };
        self.p_accept = ndarray::Zip::from(x_arr.rows())
            .and(self.last_state.rows())
            .fold(p_start, |p_accept, a, b| {
                let accepted = (a.ne(&b) as i32) as f32;
                (1.0 - ALPHA) * p_accept + ALPHA * accepted
            });
        self.last_state = x_arr;

        Ok(())
    }

    /// Retrieves the current statistics of the chain.
    ///
    /// # Returns
    /// A `ChainStats` struct containing the current statistics.
    pub fn stats(&self) -> ChainStats {
        let n = self.n as f32;
        ChainStats {
            n: self.n,
            p_accept: self.p_accept,
            mean: self.mean.clone(),
            sm2: (self.mean_sq.clone() - self.mean.pow2()) * n / (n - 1.0),
        }
    }
}

/// Computes the Potential Scale Reduction Factor (R-hat) for multiple chains.
///
/// # Arguments
/// - `chain_stats`: Slice of references to `ChainStats` from multiple chains.
///
/// # Returns
/// An array containing the R-hat values for each parameter.
pub fn collect_rhat(chain_stats: &[&ChainStats]) -> Array1<f32> {
    let (within, var) = withinvar_from_cs(chain_stats);
    (var / within).sqrt()
}

/// Returns the maximum finite value in the array, ignoring any NaNs.
///
/// If all entries are NaN, `f32::NAN` is returned.
pub fn max_skipnan(values: &Array1<f32>) -> f32 {
    values
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .reduce(f32::max)
        .unwrap_or(f32::NAN)
}

fn withinvar_from_cs(chain_stats: &[&ChainStats]) -> (Array1<f32>, Array1<f32>) {
    let means: Vec<ArrayView1<f32>> = chain_stats.iter().map(|x| x.mean.view()).collect();
    let means = ndarray::stack(Axis(0), &means).expect("Expected stacking means to succeed");
    let sm2s: Vec<ArrayView1<f32>> = chain_stats.iter().map(|x| x.sm2.view()).collect();
    let sm2s = ndarray::stack(Axis(0), &sm2s).expect("Expected stacking sm2 arrays to succeed");

    let within = sm2s
        .mean_axis(Axis(0))
        .expect("Expected computing within-chain variances to succeed");
    let global_means = means
        .mean_axis(Axis(0))
        .expect("Expected computing global means to succeed");
    let diffs: Array2<f32> = (means.clone()
        - global_means
            .broadcast(means.shape())
            .expect("Expected broadcasting to succeed"))
    .into_dimensionality()
    .expect("Expected casting dimensionality to Array1 to succeed");
    let n_chains = means.shape()[0] as f32;
    let n = chain_stats.iter().map(|x| x.n as f32).sum::<f32>() / n_chains;
    let between = diffs.pow2().sum_axis(Axis(0)) * (n / (n_chains - 1.0));

    let var = between * (1.0 / n) + within.clone() * ((n - 1.0) / n);
    (within, var)
}

/// Tracks statistics across multiple MCMC chains.
///
/// # Fields
/// - `n`: Number of steps taken.
/// - `mean`: Mean of the parameters across chains.
/// - `mean_sq`: Mean of the squared parameters across chains.
/// - `n_chains`: Number of chains.
/// - `n_params`: Number of parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiChainTracker {
    n: usize,
    pub p_accept: f32,
    last_state: Array2<f32>,
    mean: Array2<f32>,    // n_chains x n_params
    mean_sq: Array2<f32>, // n_chains x n_params
    n_chains: usize,
    n_params: usize,
}

impl MultiChainTracker {
    /// Creates a new `MultiChainTracker` for the given number of chains and parameters.
    ///
    /// # Arguments
    /// - `n_chains`: Number of chains.
    /// - `n_params`: Number of parameters.
    ///
    /// # Returns
    /// A new `MultiChainTracker` instance.
    pub fn new(n_chains: usize, n_params: usize) -> Self {
        let mean_sq = Array2::<f32>::zeros((n_chains, n_params));
        Self {
            n: 0,
            p_accept: 0.0,
            last_state: Array2::<f32>::zeros((n_chains, n_params)),
            mean: Array2::<f32>::zeros((n_chains, n_params)),
            mean_sq,
            n_chains,
            n_params,
        }
    }

    /// Updates the tracker with new states from all chains.
    ///
    /// # Arguments
    /// - `x`: New states of the chains, flattened into a single slice.
    ///
    /// # Returns
    /// `Ok(())` if successful; an error otherwise.
    pub fn step<T>(&mut self, x: &[T]) -> Result<(), Box<dyn Error>>
    where
        T: Num
            + num_traits::ToPrimitive
            + num_traits::FromPrimitive
            + std::clone::Clone
            + std::cmp::PartialOrd,
    {
        self.n += 1;

        let n = self.n as f32;
        let x_arr = ndarray::ArrayView2::<T>::from_shape((self.n_chains, self.n_params), x)?
            .mapv(|x| x.to_f32().unwrap());

        self.mean = (self.mean.clone() * (n - 1.0) + x_arr.clone()) / n;
        if self.n == 1 {
            self.mean_sq = x_arr.pow2();
        } else {
            self.mean_sq = (self.mean_sq.clone() * (n - 1.0) + (x_arr.pow2())) / n;
        };

        // Update self.p_accept and last state
        self.p_accept = ndarray::Zip::from(x_arr.rows())
            .and(self.last_state.rows())
            .fold(self.p_accept, |p_accept, a, b| {
                let accepted = (a.ne(&b) as i32) as f32;
                (1.0 - ALPHA) * p_accept + ALPHA * accepted
            });
        self.last_state = x_arr;

        Ok(())
    }

    #[cfg(feature = "burn")]
    pub fn stats<B: burn::prelude::Backend>(
        &self,
        sample: burn::tensor::Tensor<B, 3>,
    ) -> Result<RunStats, Box<dyn Error>>
    where
        B::FloatElem: num_traits::ToPrimitive,
    {
        let sample_data = sample.to_data();
        let sample_ndarray = ArrayView3::from_shape(
            sample.dims(),
            sample_data
                .as_slice::<B::FloatElem>()
                .map_err(|e| format!("{:?}", e))?,
        )?;
        self.stats_view(sample_ndarray)
    }

    pub fn stats_view<T>(&self, sample: ArrayView3<T>) -> Result<RunStats, Box<dyn Error>>
    where
        T: num_traits::ToPrimitive + Clone,
    {
        Ok(RunStats::from(sample))
    }

    /// Computes the maximum R-hat value across all parameters.
    ///
    /// # Returns
    /// The maximum R-hat value, or an error if computation fails.
    pub fn max_rhat(&self) -> Result<f32, Box<dyn Error>> {
        let all: Array1<f32> = self.rhat()?;
        let max = all
            .iter()
            .copied()
            .reduce(f32::max)
            .ok_or("No R-hat values available")?;
        Ok(max)
    }

    /// Computes the R-hat values for all parameters.
    ///
    /// # Returns
    /// An array containing the R-hat values for each parameter, or an error if computation fails.
    pub fn rhat(&self) -> Result<Array1<f32>, Box<dyn Error>> {
        let (within, var) = self.within_and_var()?;
        let rhat = (var / within).sqrt();
        Ok(rhat)
    }

    fn within_and_var(&self) -> Result<(Array1<f32>, Array1<f32>), Box<dyn Error>> {
        let mean_chain = self
            .mean
            .mean_axis(Axis(0))
            .ok_or("Mean reduction across chains for mean failed.")?;
        let n_chains = self.mean.shape()[0] as f32;
        let n = self.n as f32;
        let fac = n / (n_chains - 1.0);
        let between = (self.mean.clone() - mean_chain.insert_axis(Axis(0)))
            .pow2()
            .sum_axis(Axis(0))
            * fac;
        let sm2 = (self.mean_sq.clone() - self.mean.pow2()) * n / (n - 1.0);
        let within = sm2
            .mean_axis(Axis(0))
            .ok_or("Mean reduction across chains for mean of squares failed.")?;
        let var = within.clone() * ((n - 1.0) / n) + between * (1.0 / n);
        Ok((within, var))
    }
}

/// Computes basic statistics from
pub fn basic_stats(name: &str, mut data: Array1<f64>) -> BasicStats {
    data.as_slice_mut()
        .unwrap()
        .sort_by(|a, b| match b.partial_cmp(a) {
            Some(x) => x,
            None => Ordering::Equal,
        });
    let (min, median, max) = (
        *data
            .last()
            .expect("Expected getting first element from ess array succeed"),
        data[data.len() / 2],
        *data
            .first()
            .expect("Expected getting last element from ess array succeed"),
    );
    let mean = data.mean().expect("Expected computing mean ess to succeed");
    let std = data.std(1.0);
    BasicStats {
        name: name.to_string(),
        min,
        median,
        max,
        mean,
        std,
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct RunStats {
    pub ess: BasicStats,
    pub rhat: BasicStats,
}

impl fmt::Display for RunStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Using the Display implementation of BasicStats
        write!(f, "{}\n{}", self.ess, self.rhat)
    }
}

impl RunStats {
    pub fn min_ess(&self) -> f64 {
        self.ess.min
    }

    pub fn max_rhat(&self) -> f64 {
        self.rhat.max
    }
}

impl<T> From<ArrayView3<'_, T>> for RunStats
where
    T: ToPrimitive + std::clone::Clone,
{
    fn from(sample: ArrayView3<T>) -> Self {
        let (rhat, ess) = split_rhat_mean_ess(sample);
        let ess = basic_stats("ESS", ess);
        let rhat = basic_stats("Split R-hat", rhat);
        RunStats { ess, rhat }
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct BasicStats {
    pub name: String,
    pub min: f64,
    pub median: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
}

impl fmt::Display for BasicStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Write a human-friendly summary of the stats.
        write!(
            f,
            "{} in [{:.2}, {:.2}], median: {:.2}, mean: {:.2} ± {:.2}",
            self.name, self.min, self.max, self.median, self.mean, self.std
        )
    }
}

/// Takes a (chains, observations, parameters) view and returns a new
/// (2*chains, observations/2, parameters) array by splitting each chain in half.
fn splitcat(sample: ArrayView3<f64>) -> Array3<f64> {
    let n = sample.shape()[1];
    let half = (n / 2) as i32;
    let half_1 = sample.slice(s![.., ..half, ..]);
    let half_2 = sample.slice(s![.., -half.., ..]);
    concatenate(Axis(0), &[half_1, half_2]).expect("Expected stacking two halves to succeed")
}

/// Computes both split-R-hat and ESS metrics following STAN's methodology.
///
/// # Arguments
/// - `sample`: 3D array of observations with shape (chains, observations, parameters).
///
/// # Returns
/// A tuple containing:
/// - An array of split-R-hat values for each parameter
/// - An array of ESS values for each parameter
///
/// # References
/// - STAN Reference Manual, Section on R-hat and Effective Sample Size
pub fn split_rhat_mean_ess<T>(sample: ArrayView3<T>) -> (Array1<f64>, Array1<f64>)
where
    T: ToPrimitive + Clone,
{
    let sample_f64 = sample.mapv(|x| x.to_f64().unwrap());
    let splitted = splitcat(sample_f64.view()); // shape: (2c, n/2, p)
    let (within, var) = withinvar(splitted.view());
    (
        rhat(within.view(), var.view()),
        ess(splitted.view(), within.view(), var.view()),
    )
}

fn rhat(within: ArrayView1<f64>, var: ArrayView1<f64>) -> Array1<f64> {
    Zip::from(&within).and(&var).map_collect(|&w, &v| {
        if w.abs() <= f64::EPSILON && v.abs() <= f64::EPSILON {
            1.0
        } else if w.abs() <= f64::EPSILON {
            f64::INFINITY
        } else {
            (v / w).sqrt()
        }
    })
}

fn withinvar(sample: ArrayView3<f64>) -> (Array1<f64>, Array1<f64>) {
    let c = sample.shape()[0];
    let n = sample.shape()[1];
    let p = sample.shape()[2];
    assert!(
        c >= 2 && n >= 2,
        "split R-hat and ESS require at least 2 split chains and 2 draws per split chain"
    );

    // 2) For each parameter, compute the chain-split stats
    //    shape for data_p is (2c, n/2)
    let (within, var): (Vec<f64>, Vec<f64>) = (0..p)
        .into_par_iter()
        .map(|param_idx| {
            let data_p = sample.slice(s![.., .., param_idx]);

            // chain means => shape (2c,)
            let chain_means = data_p.mean_axis(Axis(1)).unwrap();
            let overall_mean = chain_means.mean().unwrap();

            // B => between chain var
            //   = (n/2 / (2c - 1)) * sum( (chain_means - overall_mean)^2 )
            // (We assume 2c > 1)
            let diff = &chain_means - overall_mean;
            let split_n = n as f64;
            let b = diff.pow2().sum() * (split_n / ((c - 1) as f64));

            // within => we broadcast chain_means to shape (2c, n/2) to subtract
            // but an easier approach might be:
            // squares[i] = mean over that chain's (x_i - chain_mean[i])^2
            // then average squares across all chains
            let mut squares = Vec::with_capacity(c);
            for chain_i in 0..c {
                let row = data_p.slice(s![chain_i, ..]);
                let cm = chain_means[chain_i];
                let sq = row.iter().map(|v| (v - cm) * (v - cm)).sum::<f64>() / (split_n - 1.0);
                squares.push(sq);
            }
            let squares = Array1::from(squares); // shape (2c,)
            let w = squares.mean().unwrap(); // within = average across chains
            // var => ((n/2 - 1)/(n/2)) * w + b/(n/2)
            let v = ((split_n - 1.0) / split_n) * w + b / split_n;

            (w, v)
        })
        .collect::<Vec<(f64, f64)>>()
        .into_iter()
        .fold((vec![], vec![]), |(mut within, mut var), (w, v)| {
            within.push(w);
            var.push(v);
            (within, var)
        });
    (Array1::from_vec(within), Array1::from_vec(var))
}

/// Computes the Effective Sample Size (ESS) for each parameter.
///
/// This function implements the ESS calculation as described in the STAN documentation.
/// It computes the ESS based on the autocorrelation of the chains and the ratio of
/// within-chain to total variance.
///
/// # Arguments
/// - `sample`: 3D array of observations with shape (chains, observations, parameters).
/// - `within`: Array of within-chain variances for each parameter.
/// - `var`: Array of total variances for each parameter.
///
/// # Returns
/// An array containing the ESS for each parameter.
///
/// # References
/// - STAN Reference Manual, Section on Effective Sample Size
///   (https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html)
fn ess(sample: ArrayView3<f64>, within: ArrayView1<f64>, var: ArrayView1<f64>) -> Array1<f64> {
    let shape = sample.shape();
    let (n_chains, n_steps, n_params) = (shape[0], shape[1], shape[2]);
    let chain_rho: Vec<Array2<f64>> = (0..n_chains)
        .map(|c| {
            let chain_sample = sample.index_axis(Axis(0), c);
            autocov(chain_sample)
        })
        .collect();
    let mut avg_cov = Array2::<f64>::zeros((n_steps, n_params));
    for chain_cov in chain_rho {
        avg_cov += &chain_cov;
    }
    avg_cov.mapv_inplace(|x| x / n_chains as f64);

    let total_draws = n_chains as f64 * n_steps as f64;
    let ess_vals: Vec<f64> = (0..n_params)
        .into_par_iter()
        .map(|d| {
            let w = within[d];
            let v = var[d];
            if w.abs() <= f64::EPSILON && v.abs() <= f64::EPSILON {
                return total_draws;
            }
            if !v.is_finite() || v <= 0.0 {
                return f64::NAN;
            }

            let rho_at = |lag| 1.0 - (w - avg_cov[[lag, d]]) / v;
            let mut min = if n_steps >= 2 {
                rho_at(0) + rho_at(1)
            } else {
                0.0
            };

            let mut out = 0.0;
            let mut lag = 0;
            while lag + 1 < n_steps {
                let mut p_t = rho_at(lag) + rho_at(lag + 1);
                if p_t <= 0.0 {
                    break;
                }
                if p_t > min {
                    p_t = min;
                }
                min = p_t;
                out += p_t;
                lag += 2;
            }

            let tau = (-1.0 + 2.0 * out).max(1.0);
            total_draws / tau
        })
        .collect();
    Array1::from_vec(ess_vals)
}

fn autocov(sample: ArrayView2<f64>) -> Array2<f64> {
    if sample.nrows() <= 100 {
        autocov_bf(sample)
    } else {
        autocov_fft(sample)
    }
}

/// Compute the autocovariance of multiple sequences (each column represents a distinct sequence)
/// using FFT for efficient calculation.
///
/// # Arguments
///
/// * `sample` - A 2-dimensional array view (`ArrayView2<f64>`) of shape `(n, d)`, where:
///     - `n`: length of each sequence.
///     - `d`: number of sequences (each column is treated independently).
///
/// # Returns
///
/// An `Array2<f64>` of shape `(n, d)` containing the autocovariance results.
/// Each column contains the autocovariance values for the corresponding input sequence.
///
/// # Notes
///
/// * Uses zero-padding to avoid circular convolution effects (wrap-around).
/// * FFT and inverse FFT are performed using the `rustfft` crate.
/// * Computation is parallelized across sequences using Rayon.
/// * Normalization (`1/n_padded`) is applied explicitly, as `rustfft` does not normalize results.
fn autocov_fft(sample: ArrayView2<f64>) -> Array2<f64> {
    let (n, d) = (sample.shape()[0], sample.shape()[1]);
    assert!(n >= 2, "autocovariance requires at least two draws");
    let mut planner = FftPlanner::new();

    // Next power of 2 >= 2*n - 1 for zero-padding to avoid wrap-around.
    let mut n_padded = 1;
    while n_padded < 2 * n - 1 {
        n_padded <<= 1;
    }
    let fft = planner.plan_fft_forward(n_padded);
    let ffti = planner.plan_fft_inverse(n_padded);
    let out: Vec<f64> = sample
        .axis_iter(Axis(1))
        .into_par_iter()
        .map(|traj| {
            let traj_mean = traj.sum() / traj.len() as f64;
            let mut x: Vec<Complex<f64>> = traj
                .iter()
                .map(|xi| Complex {
                    re: (*xi - traj_mean),
                    im: 0.0,
                })
                .chain([Complex { re: 0.0, im: 0.0 }].repeat(n_padded - n))
                .collect();
            fft.process(x.as_mut_slice());
            x.iter_mut().for_each(|xi| {
                *xi *= xi.conj();
            });
            ffti.process(x.as_mut_slice());
            x.iter_mut()
                .take(n)
                .map(|xi| xi.re / n_padded as f64 / (n as f64 - 1.0)) // rustfft doesn't normalize for us
                .collect::<Vec<f64>>()
        })
        .flatten_iter()
        .collect();
    let out = Array2::from_shape_vec((d, n), out).expect("Expected creating dxn array to succeed");
    out.t().to_owned()
}

/// Brute force autocovariance on a 2D array of shape (n, d).
/// - `n` = number of time points (rows)
/// - `d` = number of parameters (columns)
///
/// For each column `col` and each lag `lag` (0..n), the function
/// computes:
/// $$
///    sum_{t=0..(n - lag - 1)} [ data[t, col] * data[t + lag, col] ]
/// $$
/// and stores it in `out[lag, col]`.
fn autocov_bf(data: ArrayView2<f64>) -> Array2<f64> {
    let (n, d) = data.dim();
    assert!(n >= 2, "autocovariance requires at least two draws");
    let mut out = Array2::<f64>::zeros((n, d));
    let norm = n as f64 - 1.0;

    out.axis_iter_mut(Axis(1)) // mutable view of each column in `out`
        .into_par_iter() // make it parallel
        .enumerate() // get (col_index, col_view_mut)
        .for_each(|(col_idx, mut out_col)| {
            let col_data = data.column(col_idx);
            let col_data = col_data.to_owned() - col_data.mean().unwrap();

            // For each lag, compute sum_{t=0..(n-lag-1)} [ data[t, col] * data[t + lag, col] ]
            for lag in 0..n {
                let mut sum_lag = 0.0;
                for t in 0..(n - lag) {
                    sum_lag += col_data[t] * col_data[t + lag];
                }
                // Write result into the current column
                out_col[lag] = sum_lag / norm;
            }
        });
    out
}

/// Computes the Effective Sample Size (ESS) from chain statistics.
/// We don't split chains here.
///
/// # Arguments
/// - `sample`: 3D array of observations with shape (chains, observations, parameters).
/// - `chain_stats`: Slice of references to `ChainStats` from multiple chains.
///
/// # Returns
/// An array containing the ESS for each parameter.
///
/// # References
/// - STAN Reference Manual, Section on Effective Sample Size
pub fn ess_from_chainstats(sample: ArrayView3<f32>, chain_stats: &[&ChainStats]) -> Array1<f64> {
    let (within, var) = withinvar_from_cs(chain_stats);
    let sample = sample.mapv(f64::from);
    let within = within.mapv(f64::from);
    let var = var.mapv(f64::from);
    ess(sample.view(), within.view(), var.view())
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::{f32, fs::File, time::Instant};

    use approx::assert_abs_diff_eq;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    // Generic helper function to run the Rhat test.
    fn run_rhat_test_generic<T>(data0: Array2<T>, data1: Array2<T>, expected: Array1<f32>, tol: f32)
    where
        T: ndarray::NdFloat + num_traits::FromPrimitive,
    {
        let mut psr = MultiChainTracker::new(3, 4);
        psr.step(data0.as_slice().unwrap()).unwrap();
        psr.step(data1.as_slice().unwrap()).unwrap();
        let rhat = psr.rhat().unwrap();
        let diff = rhat
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            diff < tol,
            "Mismatch in Rhat. Got {:?}, expected {:?}, diff = {:?}",
            rhat,
            expected,
            diff
        );
    }

    #[test]
    fn test_rhat_f32_1() {
        // Step 0 data (chains x params)
        let data_step_0: Array2<f32> = arr2(&[
            [0.0, 1.0, 0.0, 1.0], // chain 0
            [1.0, 2.0, 0.0, 2.0], // chain 1
            [0.0, 0.0, 0.0, 2.0], // chain 2
        ]);

        // Step 1 data (chains x params)
        let data_step_1: Array2<f32> = arr2(&[
            [1.0, 2.0, 2.0, 0.0], // chain 0
            [1.0, 1.0, 1.0, 1.0], // chain 1
            [0.0, 1.0, 0.0, 0.0], // chain 2
        ]);
        let expected = array![f32::consts::SQRT_2, 1.080_123_4, 0.894_427_3, 0.8660254];
        run_rhat_test_generic(data_step_0, data_step_1, expected, f32::EPSILON * 10.0);
    }

    #[test]
    fn test_rhat_f64_1() {
        let data_step_0: Array2<f64> = arr2(&[
            [0.0, 1.0, 0.0, 1.0], // chain 0
            [1.0, 2.0, 0.0, 2.0], // chain 1
            [0.0, 0.0, 0.0, 2.0], // chain 2
        ]);
        let data_step_1: Array2<f64> = arr2(&[
            [1.0, 2.0, 2.0, 0.0], // chain 0
            [1.0, 1.0, 1.0, 1.0], // chain 1
            [0.0, 1.0, 0.0, 0.0], // chain 2
        ]);
        let expected = array![f32::consts::SQRT_2, 1.0801234, 0.8944271, 0.8660254];
        run_rhat_test_generic(data_step_0, data_step_1, expected, f32::EPSILON * 10.0);
    }

    #[test]
    fn test_rhat_f64_2() {
        let data_step_0 = arr2(&[
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
        ]);
        let data_step_1 = arr2(&[
            [1.0, 2.0, 0.0, 2.0],
            [1.0, 2.0, 0.0, 0.0],
            [2.0, 0.0, 1.0, 2.0],
        ]);
        let expected = array![f32::consts::FRAC_1_SQRT_2, 0.74535599, 1.0, 1.5];
        run_rhat_test_generic(data_step_0, data_step_1, expected, f32::EPSILON * 10.0);
    }

    fn run_test_case(
        autocov_func: &dyn Fn(ArrayView2<f64>) -> Array2<f64>,
        data: &Array2<f64>,
        expected: &Array2<f64>,
        test_name: &str,
    ) {
        let result = autocov_func(data.view());
        assert_eq!(
            result.dim(),
            expected.dim(),
            "{}: shape mismatch; got {:?}, expected {:?}",
            test_name,
            result.dim(),
            expected.dim()
        );

        assert_abs_diff_eq!(result, *expected, epsilon = 1e-9);
        println!("Test: {test_name} succeeded");
    }

    // ----------------------------------------------------------
    // Test: single parameter, small integer sequence
    // ----------------------------------------------------------
    #[test]
    fn test_single_param() {
        let data = array![[1.0], [2.0], [3.0], [4.0],];
        let expected = array![[1.6666666666666667], [0.4166666666666667], [-0.5], [-0.75]];

        // Compare brute force
        run_test_case(&autocov_bf, &data, &expected, "BF: single_param_small");

        println!("Doing FFT test");
        // Compare FFT-based
        run_test_case(&autocov_fft, &data, &expected, "FFT: single_param_small");
        println!("FFT test succeeded");
    }

    // ----------------------------------------------------------
    // Test: two parameters, 4 time points
    // ----------------------------------------------------------
    #[test]
    fn test_two_params_1() {
        let data = array![[1.0, 0.3], [2.0, 2.0], [3.0, -2.0], [4.0, 5.0],];
        let expected = array![
            [1.6666666666666667, 8.689166666666667],
            [0.4166666666666667, -5.051875],
            [-0.5, 1.9629166666666666],
            [-0.75, -1.255625],
        ];

        // Compare brute force
        run_test_case(&autocov_bf, &data, &expected, "BF: two_params_small");
        // Compare FFT-based
        run_test_case(&autocov_fft, &data, &expected, "FFT: two_params_small");
    }

    #[test]
    fn test_split_rhat_matches_manual_formula() {
        let sample = array![
            [[0.0_f64], [1.0], [0.0], [1.0]],
            [[10.0], [11.0], [10.0], [11.0]],
        ];
        let split = splitcat(sample.view());
        let (within, var) = withinvar(split.view());
        let expected = (var / within).sqrt();
        let (rhat, _) = split_rhat_mean_ess(sample.view());
        assert_abs_diff_eq!(rhat, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_split_rhat_stuck_chains_is_infinite() {
        let sample = array![
            [[0.0_f64], [0.0], [0.0], [0.0]],
            [[10.0], [10.0], [10.0], [10.0]],
        ];
        let (rhat, _) = split_rhat_mean_ess(sample.view());
        assert!(
            rhat[0].is_infinite(),
            "Expected split R-hat to blow up for separated stuck chains, got {rhat:?}"
        );
    }

    #[test]
    fn test_split_stats_constant_identical_chains_are_finite() {
        let sample = array![
            [[3.0_f64], [3.0], [3.0], [3.0]],
            [[3.0_f64], [3.0], [3.0], [3.0]],
        ];
        let (rhat, ess) = split_rhat_mean_ess(sample.view());
        assert_abs_diff_eq!(rhat[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ess[0], 8.0, epsilon = 1e-12);
    }

    #[test]
    fn test_collect_rhat_matches_multichain_tracker() {
        let data_step_0: Array2<f32> = arr2(&[
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 2.0],
        ]);
        let data_step_1: Array2<f32> = arr2(&[
            [1.0, 2.0, 2.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
        ]);

        let mut trackers = (0..3)
            .map(|chain| ChainTracker::new(4, data_step_0.row(chain).as_slice().unwrap()))
            .collect::<Vec<_>>();
        let mut psr = MultiChainTracker::new(3, 4);

        for step in [&data_step_0, &data_step_1] {
            psr.step(step.as_slice().unwrap()).unwrap();
            for (chain, tracker) in trackers.iter_mut().enumerate() {
                tracker.step(step.row(chain).as_slice().unwrap()).unwrap();
            }
        }

        let stats = trackers.iter().map(ChainTracker::stats).collect::<Vec<_>>();
        let refs = stats.iter().collect::<Vec<_>>();
        let collected = collect_rhat(refs.as_slice());
        let tracked = psr.rhat().unwrap();
        assert_abs_diff_eq!(collected, tracked, epsilon = 1e-6);
    }

    #[test]
    fn ess_1() {
        let m = 4;
        let n = 1000;

        // Initialize an empty 4 x 100 array
        let mut data = Array2::<f32>::zeros((m, n));

        // Use the built-in RNG to generate each row separately
        let mut rng = SmallRng::seed_from_u64(42);
        for mut row in data.rows_mut() {
            for elem in row.iter_mut() {
                *elem = rng.random::<f32>(); // generates uniform random number between 0 and 1
            }
        }
        let data = data
            .to_shape((data.shape()[0], data.shape()[1], 1))
            .unwrap();
        let run_stats = RunStats::from(data.view());

        println!("Samples: {}\n{run_stats}", m * n);

        assert!(run_stats.ess.min > 3800.0);
        assert!(run_stats.rhat.max < 1.01);
    }

    #[test]
    #[ignore = "Benchmark test: run only when explicitly requested"]
    fn test_autocov_perf_comp() {
        // Create output CSV
        let mut file =
            File::create("runtime_results.csv").expect("Unable to create runtime_results.csv");
        // Write header row
        writeln!(file, "length,rep,time,algorithm").expect("Unable to write CSV header");

        let mut rng = rand::rng();

        for exp in 0..10 {
            let n = 1 << exp; // 2^exp
            for rep in 1..=10 {
                // Generate random data of size (n x 1)
                let sample_data: Vec<f64> = (0..n * 1000).map(|_| rng.random()).collect();
                let sample: Array2<f64> = Array2::from_shape_vec((n, 1000), sample_data)
                    .expect("Failed to create Array2");

                // Measure FFT-based implementation
                let start_fft = Instant::now();
                autocov_fft(sample.view());
                let fft_time = start_fft.elapsed().as_nanos();

                // Measure brute-force implementation
                let start_brute = Instant::now();
                autocov_bf(sample.view());
                let brute_time = start_brute.elapsed().as_nanos();

                // Log results to CSV
                writeln!(file, "{},{},{},fft", n, rep, fft_time)
                    .expect("Unable to write test results to CSV");
                writeln!(file, "{},{},{},brute force", n, rep, brute_time)
                    .expect("Unable to write test results to CSV");

                // Print results for convenience
                println!(
                    "Length: {} | Rep: {} | FFT: {} ns | Brute: {} ns",
                    n, rep, fft_time, brute_time
                );
            }
        }
    }
}

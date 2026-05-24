/*!
# Core MCMC Utilities.

This module provides core functionality for running Markov Chain Monte Carlo (MCMC) chains in parallel.
It includes:
- The [`MarkovChain<State>`] trait, which abstracts a single MCMC chain.
- The [`Trace`] trait for mapping a heavy chain state to lightweight diagnostics.
- Utility functions [`run_chain`] and [`run_chain_progress`] for executing a single chain and collecting trace summaries.
- The [`HasChains<State>`] trait for types that own multiple Markov chains.
- The [`ChainRunner<State>`] trait that extends [`HasChains<State>`] with methods to run chains in parallel (using Rayon), discarding burn-in and optionally displaying progress bars.

Any type implementing [`HasChains<State>`] (with the required trait bounds) automatically implements [`ChainRunner<State>`] via a blanket implementation.

This module is generic over the chain state type and the per-step trace summaries.
*/

use crate::stats::{ChainStats, ChainTracker, RunStats, collect_rhat, max_skipnan};
use indicatif::ProgressBar;
use indicatif::{MultiProgress, ProgressStyle};
use ndarray::stack;
use ndarray::{ShapeError, prelude::*};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::error::Error;
use std::marker::Send;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self};
use std::time::{Duration, Instant};

/// A lightweight summary of a chain state suitable for diagnostics.
pub trait Trace {
    /// Returns a vector of summary values derived from the state.
    fn trace(&self) -> Vec<f64>;
}

impl<T> Trace for [T]
where
    T: ToPrimitive,
{
    fn trace(&self) -> Vec<f64> {
        self.iter()
            .map(|x| {
                x.to_f64()
                    .expect("Expected trace conversion to f64 to succeed")
            })
            .collect()
    }
}

impl<T> Trace for Vec<T>
where
    T: ToPrimitive,
{
    fn trace(&self) -> Vec<f64> {
        self.as_slice().trace()
    }
}

impl Trace for f64 {
    fn trace(&self) -> Vec<f64> {
        vec![*self]
    }
}

impl Trace for f32 {
    fn trace(&self) -> Vec<f64> {
        vec![(*self).into()]
    }
}

/// A trait that abstracts a single MCMC chain.
///
/// A type implementing [`MarkovChain<State>`] must provide:
/// - `step()`: advances the chain one iteration and returns a reference to the updated state.
/// - `current_state()`: returns a reference to the current state without modifying the chain.
pub trait MarkovChain<State> {
    /// Performs one iteration of the chain and returns a reference to the new state.
    fn step(&mut self) -> &State;

    /// Returns a reference to the current state of the chain without advancing it.
    fn current_state(&self) -> &State;
}

/// Runs a single MCMC chain for a specified number of steps.
///
/// This function repeatedly calls the chain's `step()` method and collects each trace into a
/// [`ndarray::Array2<f64>`] of shape `[n_collect, D]` where:
/// - `n_collect`: number of observations to collect
/// - `D`: dimensionality of the trace space
///
/// Each row corresponds to one collected trace of the chain.
pub fn run_chain<S, M>(chain: &mut M, n_collect: usize, n_discard: usize) -> Array2<f64>
where
    M: MarkovChain<S>,
    S: Trace,
{
    let current_trace = chain.current_state().trace();
    let dim = current_trace.len();
    let mut out = Array2::<f64>::zeros((n_collect, dim));
    let total = n_collect + n_discard;

    for i in 0..total {
        let state = chain.step();
        if i >= n_discard {
            let trace = state.trace();
            let trace_arr = ArrayView::from_shape(trace.len(), trace.as_slice()).unwrap();
            out.row_mut(i - n_discard).assign(&trace_arr);
        }
    }

    out
}

/// Runs a single MCMC chain for a `n_collect` + `n_discard` number of steps while displaying progress.
///
/// This function is similar to [`run_chain`], but it accepts an [`indicatif::ProgressBar`]
/// that is updated as the chain advances.
///
/// # Arguments
///
/// * `chain` - A mutable reference to an object implementing [`MarkovChain<State>`].
/// * `n_collect` - The number of observations to collect and return.
/// * `n_discard` - The number of observations to discard (burn-in).
/// * `tx` - A [`Sender<ChainStats>`] object for communication with chains-managing parent thread.
///
/// # Returns
///
/// A [`ndarray::Array2<f64>`] containing the chain's traces with `n_collect` number of rows.
pub fn run_chain_progress<S, M>(
    chain: &mut M,
    n_collect: usize,
    n_discard: usize,
    tx: Sender<ChainStats>,
) -> Result<Array2<f64>, String>
where
    M: MarkovChain<S>,
    S: Trace,
{
    let current_trace = chain.current_state().trace();
    let n_params = current_trace.len();
    let mut out = Array2::<f64>::zeros((n_collect, n_params));

    let mut tracker = ChainTracker::new(n_params, current_trace.as_slice());
    let mut last = Instant::now();
    let freq = Duration::from_secs(1);
    let total = n_discard + n_collect;

    for i in 0..total {
        let state = chain.step();
        let trace = state.trace();
        tracker.step(trace.as_slice()).map_err(|e| {
            let msg = format!(
            "Chain statistics tracker caused error: {}.\nAborting generation of further observations.",
            e
            );
            crate::__vmc_out!("{}", msg);
            msg
        })?;

        let now = Instant::now();
        if (now >= last + freq) | (i == total - 1) {
            if let Err(e) = tx.send(tracker.stats()) {
                crate::__vmc_err!("Sending chain statistics failed: {e}");
            }
            last = now;
        }

        if i >= n_discard {
            out.row_mut(i - n_discard)
                .assign(&ArrayView1::from_shape(trace.len(), trace.as_slice()).unwrap());
        }
    }

    // TODO: Somehow save state of the chains and enable continuing runs
    Ok(out)
}

/// A trait for types that own multiple MCMC chains.
///
/// - `S` is the concrete state type of the chain.
/// - `Chain` is the concrete type of the individual chain, which must implement [`MarkovChain<S>`]
///   and be [`Send`].
///
/// Implementors must provide a method to access the internal vector of chains.
pub trait HasChains<S> {
    type Chain: MarkovChain<S> + Send;

    /// Returns a mutable reference to the vector of chains.
    fn chains_mut(&mut self) -> &mut Vec<Self::Chain>;
}

/// An extension trait for types that own multiple MCMC chains.
///
/// [`ChainRunner<T>`] extends [`HasChains<T>`] by providing default methods to run all chains
/// in parallel. These methods allow you to:
/// - Run all chains, collect `n_collect` traces and discard `n_discard` initial burn-in observations.
/// - Optionally display progress bars for each chain during execution.
///
/// Any type that implements [`HasChains<T>`] (with appropriate bounds on `T`) automatically implements
/// [`ChainRunner<T>`].
pub trait ChainRunner<T>: HasChains<T>
where
    T: Trace + Send,
{
    /// Runs all chains in parallel, discarding the first `discard` iterations (burn-in).
    ///
    /// # Arguments
    ///
    /// * `n_collect` - The number of observations to collect and return.
    /// * `n_discard` - The number of observations to discard (burn-in).
    ///
    /// # Returns
    ///
    /// A [`ndarray::Array3`] tensor with the first axis representing the chain, the second one the
    /// step and the last one the parameter dimension.
    fn run(&mut self, n_collect: usize, n_discard: usize) -> Result<Array3<f64>, ShapeError> {
        // Run them all in parallel
        let results: Vec<Array2<f64>> = self
            .chains_mut()
            .par_iter_mut()
            .map(|chain| run_chain(chain, n_collect, n_discard))
            .collect();
        let views: Vec<ArrayView2<f64>> = results.iter().map(|x| x.view()).collect();
        let out: Array3<f64> = stack(Axis(0), &views)?;
        Ok(out)
    }

    /// Runs all chains in parallel with progress bars, discarding the burn-in.
    ///
    /// Each chain is run in parallel with its own progress bar. After execution, the first `discard`
    /// iterations are discarded.
    ///
    /// # Arguments
    ///
    /// * `n_collect` - The number of observations to collect and return.
    /// * `n_discard` - The number of observations to discard (burn-in).
    ///
    /// # Returns
    ///
    /// Returns a tuple containing:
    /// - A [`ndarray::Array3`] tensor with the first axis representing the chain, the second one the
    ///   step and the last one the parameter dimension.
    /// - A `RunStats` object containing convergence statistics including:
    ///   - Acceptance probability
    ///   - Potential scale reduction factor (R-hat)
    ///   - Effective sample size (ESS)
    ///   - Other convergence diagnostics
    fn run_progress(
        &mut self,
        n_collect: usize,
        n_discard: usize,
    ) -> Result<(Array3<f64>, RunStats), Box<dyn Error>> {
        // Channels.
        // Each chain gets its own channel. Hence, we have `n_chains` channels.
        // The objects sent over channels are Array2<f32>s ($s_m^2$, $\bar{\theta}_m^{(\bullet)}$).
        // The child thread sends it's respective one to the parent thread.
        // The parent thread assemples the tuples it receives to compute Rhat.

        let chains = self.chains_mut();

        let mut rxs: Vec<Receiver<ChainStats>> = vec![];
        let mut txs: Vec<Sender<ChainStats>> = vec![];
        (0..chains.len()).for_each(|_| {
            let (tx, rx) = mpsc::channel();
            rxs.push(rx);
            txs.push(tx);
        });

        let progress_handle = thread::spawn(move || {
            let sleep_ms = Duration::from_millis(250);
            let timeout_ms = Duration::from_millis(0);
            let multi = MultiProgress::new();

            let pb_style = ProgressStyle::default_bar()
                .template("{prefix:8} {bar:40.cyan/blue} {pos}/{len} ({eta}) | {msg}")
                .unwrap()
                .progress_chars("=>-");
            let total: u64 = (n_collect + n_discard).try_into().unwrap();

            // Global Progress bar
            let global_pb = multi.add(ProgressBar::new((rxs.len() as u64) * total));
            global_pb.set_style(pb_style.clone());
            global_pb.set_prefix("Global");

            let mut active: Vec<(usize, ProgressBar)> = (0..rxs.len().min(5))
                .map(|chain_idx| {
                    let pb = multi.add(ProgressBar::new(total));
                    pb.set_style(pb_style.clone());
                    pb.set_prefix(format!("Chain {chain_idx}"));
                    (chain_idx, pb)
                })
                .collect();
            let mut next_active = active.len();
            let mut n_finished = 0;
            let mut most_recent = vec![None; rxs.len()];
            let mut total_progress;

            loop {
                for (i, rx) in rxs.iter().enumerate() {
                    while let Ok(stats) = rx.recv_timeout(timeout_ms) {
                        most_recent[i] = Some(stats)
                    }
                }

                // Update chain progress bar messages
                // and compute average acceptance probability
                let mut to_replace = vec![false; active.len()];
                let mut avg_p_accept = 0.0;
                let mut n_available_stats = 0.0;
                for (vec_idx, (i, pb)) in active.iter().enumerate() {
                    if let Some(stats) = &most_recent[*i] {
                        pb.set_position(stats.n);
                        pb.set_message(format!("p(accept)≈{:.2}", stats.p_accept));
                        avg_p_accept += stats.p_accept;
                        n_available_stats += 1.0;

                        if stats.n == total {
                            to_replace[vec_idx] = true;
                            n_finished += 1;
                        }
                    }
                }
                avg_p_accept /= n_available_stats;

                // Update global progress bar
                total_progress = 0;
                for stats in most_recent.iter().flatten() {
                    total_progress += stats.n;
                }
                global_pb.set_position(total_progress);
                let valid: Vec<&ChainStats> = most_recent.iter().flatten().collect();
                if valid.len() >= 2 {
                    let rhats = collect_rhat(valid.as_slice());
                    let max = max_skipnan(&rhats);
                    global_pb.set_message(format!(
                        "p(accept)≈{:.2} max(rhat)≈{:.2}",
                        avg_p_accept, max
                    ))
                }

                let mut to_remove = vec![];
                for (i, replace) in to_replace.iter().enumerate() {
                    if *replace && next_active < most_recent.len() {
                        let pb = multi.add(ProgressBar::new(total));
                        pb.set_style(pb_style.clone());
                        pb.set_prefix(format!("Chain {next_active}"));
                        active[i] = (next_active, pb);
                        next_active += 1;
                    } else if *replace {
                        to_remove.push(i);
                    }
                }

                to_remove.sort();
                for i in to_remove.iter().rev() {
                    active.remove(*i);
                }

                if n_finished >= most_recent.len() {
                    break;
                }
                std::thread::sleep(sleep_ms);
            }
        });

        let chain_sample: Vec<Array2<f64>> = thread::scope(|s| {
            let handles: Vec<thread::ScopedJoinHandle<Array2<f64>>> = chains
                .iter_mut()
                .zip(txs)
                .map(|(chain, tx)| {
                    s.spawn(|| {
                        run_chain_progress(chain, n_collect, n_discard, tx)
                            .expect("Expected running chain to succeed.")
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .expect("Expected thread to succeed in generating observation.")
                })
                .collect()
        });
        let sample: Array3<f64> = stack(
            Axis(0),
            &chain_sample
                .iter()
                .map(|x| x.view())
                .collect::<Vec<ArrayView2<f64>>>(),
        )?;

        if let Err(e) = progress_handle.join() {
            crate::__vmc_err!("Progress bar thread emitted error message (join error)");
            drop(e);
        }

        let run_stats = RunStats::from(sample.view());

        Ok((sample, run_stats))
    }
}

impl<T: Trace + Send, R: HasChains<T>> ChainRunner<T> for R {}

/// Generates a vector of random initial positions from a standard normal distribution.
///
/// Each position is a `Vec<T>` of length `d` representing a point in `d`-dimensional space.
/// The function returns `n` such positions.
///
/// # Type Parameters
/// - `T`: The numeric type (e.g., `f32`, `f64`). Must implement `Float + FromPrimitive`.
///
/// # Parameters
/// - `n`: Number of positions to generate.
/// - `d`: Dimensionality of each position.
///
/// # Returns
/// A `Vec<Vec<T>>` where each inner vector is a position in `d`-dimensional space.
///
/// # Panics
/// Panics if an observation cannot be converted from `f64` to `T` (should never happen for `f32` or `f64`).
///
/// # Examples
/// ```
/// # use general_mcmc::core::init;
/// let positions: Vec<Vec<f32>> = init(5, 3);
/// for pos in positions {
///     println!("{:?}", pos);
/// }
/// ```
pub fn init<T>(n: usize, d: usize) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive,
{
    let rng = SmallRng::seed_from_u64(rand::rng().random::<u64>());
    _init(n, d, rng)
}

/// Generates `n` pseudo-random vectors from the `d` dimensional standard normal distribution.
/// This function calls [`init_with_seed`] with the same parameters and seed 42.
pub fn init_det<T>(n: usize, d: usize) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive,
{
    init_with_seed(n, d, 42)
}

/// Generates `n` pseudo-random vectors from the `d` dimensional standard normal distribution.
/// Same as [`init`] except this function returns a deterministic sample.
pub fn init_with_seed<T>(n: usize, d: usize, seed: u64) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive,
{
    let rng = SmallRng::seed_from_u64(seed);
    _init(n, d, rng)
}

fn _init<T>(n: usize, d: usize, mut rng: SmallRng) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive,
{
    (0..n)
        .map(|_| {
            (0..d)
                .map(|_| {
                    let obs: f64 = rng.sample(StandardNormal);
                    T::from_f64(obs).unwrap()
                })
                .collect()
        })
        .collect()
}

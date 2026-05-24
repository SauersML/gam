//! Opt-in Criterion microbenchmark for the joint-Newton line-search latency
//! pattern seen at biobank shape.
//!
//! The benchmark models the observed cycle-0 pattern where alpha attempts
//! 1.0, 0.5, 0.25, and 0.125 reject and alpha 0.0625 accepts.  Each simulated
//! likelihood evaluation busy-spins for `BENCH_WORK_MS`, so the sequential
//! baseline performs five evaluations while the speculative launcher
//! evaluates the first three attempts concurrently and then falls back to
//! sequential halving.

use criterion::{Criterion, criterion_group, criterion_main};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::time::Duration;

const RUN_BENCH: bool = false;
const BENCH_WORK_MS: u64 = 25;
const ACCEPT_BT: usize = 4;
const MAX_ATTEMPTS: usize = 8;
const SPECULATIVE_ATTEMPTS: usize = 3;

fn simulated_objective(bt: usize, work: Duration) -> f64 {
    // Spin instead of sleeping to occupy a worker thread for the duration —
    // the bench measures wall-clock under contention, so an actual CPU-bound
    // load is a more faithful proxy than `thread::sleep` (which parks).
    let until = std::time::Instant::now() + work;
    while std::time::Instant::now() < until {
        std::hint::spin_loop();
    }
    if bt >= ACCEPT_BT { 0.0 } else { 2.0 }
}

fn sequential_backtracking(work: Duration) -> usize {
    for bt in 0..MAX_ATTEMPTS {
        if simulated_objective(bt, work) <= 1.0 {
            return bt;
        }
    }
    MAX_ATTEMPTS
}

fn speculative_backtracking(work: Duration) -> usize {
    let first_wave: Vec<(usize, f64)> = (0..SPECULATIVE_ATTEMPTS)
        .into_par_iter()
        .map(|bt| (bt, simulated_objective(bt, work)))
        .collect();
    if let Some((bt, _)) = first_wave
        .iter()
        .find(|(_, objective)| objective.is_finite() && *objective <= 1.0)
    {
        return *bt;
    }
    for bt in SPECULATIVE_ATTEMPTS..MAX_ATTEMPTS {
        if simulated_objective(bt, work) <= 1.0 {
            return bt;
        }
    }
    MAX_ATTEMPTS
}

fn bench_joint_line_search_speculative(c: &mut Criterion) {
    if !RUN_BENCH {
        eprintln!(
            "skipping opt-in joint line-search benchmark; flip `RUN_BENCH` to \
             `true` in benches/joint_line_search_speculative.rs to run"
        );
        return;
    }
    gam::init_parallelism();
    let work = Duration::from_millis(BENCH_WORK_MS);

    let mut group = c.benchmark_group("joint_line_search_biobank_pattern");
    group.bench_function("sequential_5_attempts", |b| {
        b.iter(|| assert_eq!(sequential_backtracking(work), ACCEPT_BT));
    });
    group.bench_function("speculative_3_then_fallback", |b| {
        b.iter(|| assert_eq!(speculative_backtracking(work), ACCEPT_BT));
    });
    group.finish();
}

criterion_group!(benches, bench_joint_line_search_speculative);
criterion_main!(benches);

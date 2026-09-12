//! Core-utilization of the Arrow-Schur INNER joint fit — the per-birth solve that
//! dominates the stagewise/compose grind — measured directly on a production-shaped
//! system (large N, the row axis the solver parallelizes over).
//!
//! The row-parallel Arrow-Schur solve (factor blocks, Schur matvec, gradient)
//! must use the Rayon pool when called from the production serial context. Called
//! from inside a Rayon worker, the nesting guard keeps the inner solve serial.

use gam_solve::arrow_schur::{ArrowSchurSystem, ArrowSolveOptions};
use ndarray::Array2;
use std::time::Instant;

const N: usize = 40_000;
const D: usize = 2;
const K: usize = 256;

fn arrow_system() -> ArrowSchurSystem {
    let mut sys = ArrowSchurSystem::new(N, D, K);
    sys.hbb = Array2::from_shape_fn((K, K), |(i, j)| {
        if i == j {
            20.0 + (i as f64) / (K as f64)
        } else {
            5e-4 * ((i + 3 * j) as f64 * 1.7e-4).sin()
        }
    });
    for j in 0..K {
        sys.gb[j] = ((j as f64) * 0.011).cos();
    }
    for row_idx in 0..N {
        let row = &mut sys.rows[row_idx];
        for a in 0..D {
            for b in 0..D {
                row.htt[[a, b]] = if a == b {
                    4.0 + (row_idx % 17) as f64 * 0.01 + a as f64
                } else {
                    0.03
                };
            }
            row.gt[a] = ((row_idx + a) as f64 * 0.013).sin();
            for j in 0..K {
                row.htbeta[[a, j]] = 1e-4 * ((row_idx + j + a * K) as f64 * 1.9e-4).cos();
            }
        }
    }
    sys.refresh_row_hessian_fingerprint();
    sys
}

fn time_solve(sys: &ArrowSchurSystem, opts: &ArrowSolveOptions, reps: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..reps {
        let sol = sys.solve_with_options(1e-8, 1e-8, opts).expect("solve");
        std::hint::black_box(sol);
    }
    start.elapsed().as_secs_f64() / reps as f64
}

/// Physical cores among the CPUs this process may be scheduled on: the distinct
/// `thread_siblings_list` sets across `Cpus_allowed_list`, so SMT siblings count
/// once.
fn physical_cores_in_affinity() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    let allowed = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .expect("Cpus_allowed_list in /proc/self/status");
    let mut cores = std::collections::BTreeSet::new();
    for range in allowed.trim().split(',') {
        let (first, last) = range.split_once('-').unwrap_or((range, range));
        let first: usize = first.parse().expect("CPU id in Cpus_allowed_list");
        let last: usize = last.parse().expect("CPU id in Cpus_allowed_list");
        for cpu in first..=last {
            let siblings = std::fs::read_to_string(format!(
                "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            ))
            .expect("read CPU topology");
            cores.insert(siblings.trim().to_owned());
        }
    }
    cores.len()
}

#[test]
fn inner_fit_core_scaling() {
    let sys = arrow_system();
    let opts = ArrowSolveOptions::direct();
    let reps = 3;

    // Warm (allocations, first-touch, factor caches).
    std::hint::black_box(
        sys.solve_with_options(1e-8, 1e-8, &opts)
            .expect("warm solve"),
    );

    // Serial production context: called from the main non-Rayon thread, so the
    // row work should fan out over the global Rayon pool.
    let t_serial = time_solve(&sys, &opts, reps);

    // Nested context: called from inside a Rayon worker, where the nesting guard
    // should keep the inner solve serial.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(rayon::current_num_threads())
        .build()
        .unwrap();
    let t_nested = pool.install(|| {
        assert!(
            rayon::current_thread_index().is_some(),
            "nested timing must run inside a Rayon worker"
        );
        time_solve(&sys, &opts, reps)
    });

    let pool_threads = pool.current_num_threads();
    let cores = pool_threads.min(physical_cores_in_affinity());
    let effective = t_nested / t_serial.max(1e-9);
    println!(
        "INNERFIT N={N} D={D} K={K} pool_threads={pool_threads} cores={cores} serial={t_serial:.4}s nested={t_nested:.4}s effective_cores={effective:.2}x"
    );

    // The inner fit MUST use more than one core in a serial context on a
    // production-shaped N (else the per-birth grind is inherently single-core and no
    // amount of candidate racing fixes it).
    //
    // THE FLOOR IS DERIVED FROM THE MACHINE, NOT PINNED. `effective` is a
    // speedup, so it is bounded by `cores`: the physical cores the pool can
    // occupy, the lesser of its worker count and the distinct cores in this
    // process's CPU affinity. The worker count alone is not that bound, because
    // SMT siblings share one core. GitHub's `ubuntu-latest` runners are 2-core,
    // 4-thread machines, and there the row-parallel path measured 1.78x on a
    // 4-worker pool against the 2.0 floor that counting threads as cores demanded.
    //
    // A fixed `> 2.0` bar is therefore UNACHIEVABLE on a 2-core machine no matter
    // how correct the row-parallel path is (a perfect, zero-overhead parallel
    // solve scores about 2.0 there, and `>` excludes it), and simultaneously far
    // too weak on a 32-core box. Half the core count is the honest "genuinely
    // fanning out" bar; it is capped at the original 2.0 so this stays exactly the
    // pre-existing assertion on any machine with >= 4 cores, and floored at 1.25
    // so a 2-core box still has to show real parallel gain rather than passing on
    // noise.
    let floor = ((cores as f64) / 2.0).clamp(1.25, 2.0);
    assert!(
        effective > floor,
        "Arrow-Schur inner fit must be multi-core in a serial context \
         (effective {effective:.2}x, floor {floor:.2}x on {cores} physical cores, \
         {pool_threads}-thread pool); \
         if this drops to ~1.0 the row-parallel path regressed or N fell below the \
         SCHUR_MATVEC_PARALLEL_ROW_MIN=256 threshold"
    );
}

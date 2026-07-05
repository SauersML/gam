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

    let cores = pool.current_num_threads();
    let effective = t_nested / t_serial.max(1e-9);
    println!(
        "INNERFIT N={N} D={D} K={K} pool_cores={cores} serial={t_serial:.4}s nested={t_nested:.4}s effective_cores={effective:.2}x"
    );

    // The inner fit MUST use more than one core in a serial context on a
    // production-shaped N (else the per-birth grind is inherently single-core and no
    // amount of candidate racing fixes it). A conservative floor well below the true
    // speedup keeps the assertion robust to a loaded CI box.
    assert!(
        effective > 2.0,
        "Arrow-Schur inner fit must be multi-core in a serial context (effective {effective:.2}x); \
         if this drops to ~1.0 the row-parallel path regressed or N fell below the \
         SCHUR_MATVEC_PARALLEL_ROW_MIN=256 threshold"
    );
}

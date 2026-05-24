use criterion::{Criterion, criterion_group, criterion_main};
use gam::solver::arrow_schur::{ArrowSchurSystem, ArrowSolveOptions};
use gam::solver::gpu::{Device, configure_device};
use ndarray::Array2;
use std::hint::black_box;
use std::time::Instant;

const N: usize = 100_000;
const D: usize = 2;
const K: usize = 2_000;
const K_ATOMS: usize = 10_000;

fn arrow_system() -> ArrowSchurSystem {
    let mut sys = ArrowSchurSystem::new(N, D, K);
    sys.hbb = Array2::from_shape_fn((K, K), |(i, j)| {
        if i == j {
            20.0 + (i as f64) / (K as f64)
        } else {
            0.000_5 * ((i + 3 * j) as f64 * 0.000_17).sin()
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
                row.htbeta[[a, j]] =
                    0.000_1 * ((row_idx + j + a * K_ATOMS) as f64 * 0.000_19).cos();
            }
        }
    }
    sys.refresh_row_hessian_fingerprint();
    sys
}

fn bench_arrow_gpu(c: &mut Criterion) {
    let sys = arrow_system();
    let direct = ArrowSolveOptions::direct();
    let mut group = c.benchmark_group("arrow_schur_biobank_n100k_p2k_atoms10k");
    group.sample_size(10);

    group.bench_function("cpu_direct_arrow_schur", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                black_box(sys.solve_with_options(1e-8, 1e-8, &direct));
            }
            start.elapsed()
        })
    });

    group.bench_function("cuda_arrow_schur", |b| {
        b.iter_custom(|iters| {
            configure_device(Device::Cuda);
            let start = Instant::now();
            for _ in 0..iters {
                black_box(gam::solver::gpu::arrow_schur_gpu::solve_arrow_newton_step_gpu(
                    &sys, 1e-8, 1e-8,
                ));
            }
            start.elapsed()
        })
    });

    group.finish();
}

criterion_group!(benches, bench_arrow_gpu);
criterion_main!(benches);

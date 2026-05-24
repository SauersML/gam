use criterion::{Criterion, criterion_group, criterion_main};
use gam::solver::gpu::{Device, configure_device};
use ndarray::{Array1, Array2};
use std::hint::black_box;
use std::time::Instant;

const N: usize = 100_000;
const P: usize = 2_000;

fn design() -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((N, P), |(i, j)| {
        let phase = (i as f64) * 0.000_013 + (j as f64) * 0.000_71;
        phase.sin() * 0.25 + phase.cos() * 0.05
    });
    let weights = Array1::from_shape_fn(N, |i| 0.5 + ((i % 257) as f64) / 257.0);
    let penalty = Array2::from_shape_fn((P, P), |(i, j)| if i == j { 2.0 } else { 0.0 });
    let gradient = Array1::from_shape_fn(P, |j| ((j as f64) * 0.017).sin());
    (x, weights, penalty, gradient)
}

fn cpu_xtwx(x: &Array2<f64>, weights: &Array1<f64>, penalty: &Array2<f64>) -> Array2<f64> {
    let mut out = penalty.clone();
    for i in 0..x.nrows() {
        let w = weights[i];
        for a in 0..x.ncols() {
            let xa = x[[i, a]] * w;
            for b in 0..x.ncols() {
                out[[a, b]] += xa * x[[i, b]];
            }
        }
    }
    out
}

fn bench_pirls_gpu(c: &mut Criterion) {
    let (x, weights, penalty, gradient) = design();
    let mut group = c.benchmark_group("pirls_biobank_n100k_p2k");
    group.sample_size(10);

    group.bench_function("cpu_xtwx_reference", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                black_box(cpu_xtwx(&x, &weights, &penalty));
            }
            start.elapsed()
        })
    });

    group.bench_function("cuda_pirls_step", |b| {
        b.iter_custom(|iters| {
            configure_device(Device::Cuda);
            let start = Instant::now();
            for _ in 0..iters {
                let result = gam::solver::gpu::pirls_gpu::solve_pirls_step_gpu(
                    gam::solver::gpu::pirls_gpu::PirlsGpuInput {
                        x: x.view(),
                        weights: weights.view(),
                        penalty_hessian: penalty.view(),
                        gradient: gradient.view(),
                        lm_ridge: 1e-8,
                    },
                );
                black_box(result);
            }
            start.elapsed()
        })
    });

    group.finish();
}

criterion_group!(benches, bench_pirls_gpu);
criterion_main!(benches);

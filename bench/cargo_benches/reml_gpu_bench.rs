use criterion::{Criterion, criterion_group, criterion_main};
use gam::solver::gpu::{Device, configure_device};
use ndarray::{Array1, Array2};
use std::hint::black_box;
use std::time::Instant;

const N: usize = 100_000;
const P: usize = 2_000;
const K_ATOMS: usize = 10_000;
const RHO_DERIVATIVES_TIMED: usize = 32;

fn hessian() -> Array2<f64> {
    Array2::from_shape_fn((P, P), |(i, j)| {
        if i == j {
            10.0 + (i as f64) / (P as f64)
        } else {
            let phase = ((i + j) as f64) * 0.000_31;
            0.002 * phase.sin()
        }
    })
}

fn derivative_hessians() -> Vec<Array2<f64>> {
    (0..RHO_DERIVATIVES_TIMED)
        .map(|atom| {
            Array2::from_shape_fn((P, P), |(i, j)| {
                if i == j && i % RHO_DERIVATIVES_TIMED == atom {
                    1.0 + (atom as f64) / (K_ATOMS as f64)
                } else {
                    0.0
                }
            })
        })
        .collect()
}

fn cholesky_logdet_cpu(h: &Array2<f64>) -> f64 {
    let mut l = h.clone();
    for j in 0..P {
        let mut diag = l[[j, j]];
        for k in 0..j {
            diag -= l[[j, k]] * l[[j, k]];
        }
        l[[j, j]] = diag.sqrt();
        for i in (j + 1)..P {
            let mut value = l[[i, j]];
            for k in 0..j {
                value -= l[[i, k]] * l[[j, k]];
            }
            l[[i, j]] = value / l[[j, j]];
        }
    }
    let mut logdet = 0.0_f64;
    for i in 0..P {
        logdet += l[[i, i]].ln();
    }
    2.0 * logdet
}

fn bench_reml_gpu(c: &mut Criterion) {
    let h = hessian();
    let dh = derivative_hessians();
    let dh_views: Vec<_> = dh.iter().map(|m| m.view()).collect();
    let mut group = c.benchmark_group("reml_biobank_n100k_p2k_atoms10k");
    group.sample_size(10);

    group.bench_function("cpu_logdet_reference", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                black_box(cholesky_logdet_cpu(&h));
            }
            start.elapsed()
        })
    });

    group.bench_function("cuda_logdet_and_rho_gradient_batch32", |b| {
        b.iter_custom(|iters| {
            configure_device(Device::Cuda);
            let start = Instant::now();
            for _ in 0..iters {
                let result = gam::solver::gpu::reml_gpu::evidence_derivatives_gpu(
                    gam::solver::gpu::reml_gpu::RemlGpuInput {
                        penalized_hessian: h.view(),
                        derivative_hessians: dh_views.clone(),
                    },
                );
                let output = result.expect("CUDA REML evidence benchmark must solve successfully");
                black_box(output);
            }
            start.elapsed()
        })
    });

    black_box(Array1::from_vec(vec![N as f64, K_ATOMS as f64]));
    group.finish();
}

criterion_group!(benches, bench_reml_gpu);
criterion_main!(benches);

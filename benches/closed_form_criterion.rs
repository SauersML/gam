//! Criterion benchmarks for the closed-form anisotropic Duchon pair-block,
//! its FD-derivative bundle, the isotropic kernel evaluation paths, and
//! ClosedFormPenaltyOperator matvec.
//!
//! Targets the hot paths optimised by:
//!   - rayon parallelism over the K(K+1)/2 lower-triangular pair evaluations,
//!   - SIMD vectorisation of the d-axis invariants (`aniso_invariants`),
//!   - powf→powi swap in `riesz_block_radial_derivatives` (non-log) and
//!     `matern_block_radial_derivatives` (even d),
//!   - allocation reduction in `closed_form_anisotropic_pair_block`.
//!
//! Run with: `cargo bench --bench closed_form_criterion`
//!
//! Expected baselines (Apple M-class, release mode). These are placeholders
//! for matrix-free-integrate's task #19 to populate with measured values.
//!
//!   pair_block_assembly/200       baseline ~  4-8 ms     post-opt ~  2-4 ms
//!   pair_block_assembly/500       baseline ~ 25-50 ms    post-opt ~ 12-25 ms
//!   pair_block_assembly/1000      baseline ~100-200 ms   post-opt ~ 50-100 ms
//!   pair_block_assembly/2000      baseline ~400-800 ms   post-opt ~200-400 ms
//!
//!   pair_block_bundle/single_pair_q2  baseline ~ 30-60 µs (FD path)
//!                                     analytic target ~  5-12 µs
//!
//!   isotropic_duchon_penalty/partial_fraction_typical  ~200-500 ns / call
//!   isotropic_duchon_penalty/small_r_regime            ~100-300 ns / call
//!   isotropic_duchon_penalty/anisotropic_radial_d8_eta0 ~  1- 4 µs / call
//!
//!   operator_matvec/200    baseline ~  50-100 µs
//!   operator_matvec/500    baseline ~ 300-600 µs
//!   operator_matvec/1000   baseline ~ 1.2-2.5 ms
//!   operator_matvec/2000   baseline ~  5-10 ms

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ndarray::{Array1, Array2};

use gam::terms::basis::{
    closed_form_anisotropic_pair_block,
    closed_form_penalty::{
        anisotropic_duchon_penalty_radial, isotropic_duchon_penalty,
        pair_block_radial_with_j_second_derivatives,
    },
};
use gam::terms::closed_form_operator::ClosedFormPenaltyOperator;

// Production-ish parameters: m=2 polyharmonic, s=8 Matérn order, κ=1, d=8.
// η is zeroed (isotropic limit b_k=1 across axes); this still exercises the
// full anisotropic code path but keeps numerics stable across all K.
const M_TYPICAL: usize = 2;
const S_TYPICAL: usize = 8;
const KAPPA_TYPICAL: f64 = 1.0;
const D_TYPICAL: usize = 8;

/// Deterministic, well-spread knot grid in [0, 1]^d using a Halton sequence
/// with one prime base per axis. Avoids co-located centers (which would
/// otherwise trigger the self-pair epsilon path uniformly) while keeping the
/// bench reproducible.
fn synthetic_centers(k: usize, d: usize) -> Array2<f64> {
    const PRIMES: [usize; 16] =
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
    let mut centers = Array2::<f64>::zeros((k, d));
    for axis in 0..d {
        let p = PRIMES[axis % PRIMES.len()];
        let pf = p as f64;
        for i in 0..k {
            // Van der Corput radix-p digit reversal of (i+1).
            let mut n = i + 1;
            let mut x = 0.0_f64;
            let mut f = 1.0_f64 / pf;
            while n > 0 {
                let digit = (n % p) as f64;
                x += digit * f;
                n /= p;
                f /= pf;
            }
            centers[[i, axis]] = x;
        }
    }
    centers
}

fn bench_pair_block_assembly(c: &mut Criterion) {
    let mut group = c.benchmark_group("pair_block_assembly");
    group.sample_size(10);
    for &k in &[200_usize, 500, 1000, 2000] {
        let centers = synthetic_centers(k, D_TYPICAL);
        let eta = vec![0.0_f64; D_TYPICAL];
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, _| {
            b.iter(|| {
                let g = closed_form_anisotropic_pair_block(
                    centers.view(),
                    /* q = */ 2,
                    M_TYPICAL,
                    S_TYPICAL,
                    KAPPA_TYPICAL,
                    Some(&eta),
                );
                black_box(g)
            })
        });
    }
    group.finish();
}

fn bench_pair_block_bundle(c: &mut Criterion) {
    // FD-derivative bundle: per-pair invocation. Per-K-matrix cost is
    // K(K-1)/2 × this number, so per-pair is the operative measurement.
    let mut group = c.benchmark_group("pair_block_bundle");
    group.sample_size(20);
    let eta = vec![0.0_f64; D_TYPICAL];
    let r_vec: Vec<f64> = (0..D_TYPICAL).map(|k| 0.05 + 0.07 * k as f64).collect();
    group.bench_function("single_pair_q2", |b| {
        b.iter(|| {
            let bundle = pair_block_radial_with_j_second_derivatives(
                /* q = */ 2,
                M_TYPICAL,
                S_TYPICAL,
                KAPPA_TYPICAL,
                &eta,
                &r_vec,
            );
            black_box(bundle)
        })
    });
    group.finish();
}

fn bench_isotropic_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("isotropic_duchon_penalty");
    group.sample_size(50);

    // Generic regime: typical R, partial-fraction expansion expected.
    group.bench_function("partial_fraction_typical", |b| {
        b.iter(|| {
            let v = isotropic_duchon_penalty(
                /* q = */ 2,
                D_TYPICAL,
                M_TYPICAL,
                S_TYPICAL,
                KAPPA_TYPICAL,
                /* r = */ 0.4,
            );
            black_box(v)
        })
    });

    // Small-R / small-κ regime: exercises Taylor / ₁F₂ branch where it
    // engages (decided internally by the function's regime heuristics).
    group.bench_function("small_r_regime", |b| {
        b.iter(|| {
            let v = isotropic_duchon_penalty(
                /* q = */ 2,
                D_TYPICAL,
                M_TYPICAL,
                S_TYPICAL,
                /* kappa = */ 0.05,
                /* r = */ 1e-4,
            );
            black_box(v)
        })
    });

    // Touches the SIMD `aniso_invariants` loop with d=8 + the
    // radial-derivative chain (Riesz powi / Matérn powi paths).
    group.bench_function("anisotropic_radial_d8_eta0", |b| {
        let eta = vec![0.0_f64; D_TYPICAL];
        let r: Vec<f64> = (0..D_TYPICAL).map(|k| 0.05 + 0.07 * k as f64).collect();
        b.iter(|| {
            let v = anisotropic_duchon_penalty_radial(
                /* q = */ 2,
                M_TYPICAL,
                S_TYPICAL,
                KAPPA_TYPICAL,
                &eta,
                &r,
            );
            black_box(v)
        })
    });

    group.finish();
}

fn bench_operator_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("operator_matvec");
    group.sample_size(10);
    for &k in &[200_usize, 500, 1000, 2000] {
        let centers = synthetic_centers(k, D_TYPICAL);
        let eta = vec![0.0_f64; D_TYPICAL];
        let op = ClosedFormPenaltyOperator::new(
            centers.view(),
            /* q = */ 2,
            M_TYPICAL,
            S_TYPICAL,
            KAPPA_TYPICAL,
            Some(&eta),
            /* kernel_nullspace = */ None,
            /* polynomial_block_cols = */ 0,
            /* outer_identifiability = */ None,
        );
        // Warm cached dense form outside the timed loop so we measure pure
        // matvec cost, not the one-shot build.
        let dim = op.dim();
        let w_warm = Array1::<f64>::from_elem(dim, 1.0);
        let mut warm = Array1::<f64>::zeros(dim);
        op.matvec(w_warm.view(), warm.view_mut());

        let w = Array1::<f64>::from_shape_fn(dim, |i| ((i as f64) * 0.317).sin());
        let mut out = Array1::<f64>::zeros(dim);
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, _| {
            b.iter(|| {
                op.matvec(w.view(), out.view_mut());
                black_box(&out);
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_pair_block_assembly,
    bench_pair_block_bundle,
    bench_isotropic_kernel,
    bench_operator_matvec,
);
criterion_main!(benches);

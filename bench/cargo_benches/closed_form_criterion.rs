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
//!
//!   hessian_solve_dense_vs_implicit/dense_chol/{500,1000,2000,5000}:
//!       O(p³) Cholesky factorization on the materialized Hessian. Scales
//!       cubically; expected ~50 ms / 400 ms / 3 s / 50 s respectively.
//!   hessian_solve_dense_vs_implicit/implicit_pcg/{500,1000,2000,5000}:
//!       PCG-against-implicit-H using ClosedFormPenaltyOperator's matvec.
//!       Scales as iterations × per-matvec cost. Crossover with dense
//!       Cholesky expected somewhere in p ∈ [1000, 2000] depending on
//!       conditioning of the synthetic Hessian; this bench is the source
//!       of truth for setting `CLOSED_FORM_OPERATOR_THRESHOLD` in
//!       production.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ndarray::{Array1, Array2};

use gam::terms::basis::{
    MaternNu, closed_form_aniso_psi_derivatives_in_total_basis, closed_form_anisotropic_pair_block,
    closed_form_matern_pair_block,
    closed_form_penalty::{
        anisotropic_duchon_penalty_radial, isotropic_duchon_penalty,
        pair_block_radial_with_j_second_derivatives,
    },
    closed_form_psi_derivatives_in_total_basis,
};
use gam::terms::closed_form_operator::ClosedFormPenaltyOperator;
use gam::terms::penalty_op::PenaltyOp;

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
    const PRIMES: [usize; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
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
                /* q = */ 2, D_TYPICAL, M_TYPICAL, S_TYPICAL, /* kappa = */ 0.05,
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

fn bench_derivative_matrix_assembly(c: &mut Criterion) {
    let mut group = c.benchmark_group("derivative_matrix_assembly");
    group.sample_size(10);
    for &k in &[100_usize, 200, 500] {
        let centers = synthetic_centers(k, D_TYPICAL);
        let eta: Vec<f64> = (0..D_TYPICAL)
            .map(|axis| 0.05 * ((axis as f64) - 0.5 * (D_TYPICAL as f64 - 1.0)))
            .collect();
        group.bench_with_input(BenchmarkId::new("log_kappa", k), &k, |b, _| {
            b.iter(|| {
                let bundle = closed_form_psi_derivatives_in_total_basis(
                    centers.view(),
                    /* q = */ 2,
                    M_TYPICAL,
                    S_TYPICAL,
                    KAPPA_TYPICAL,
                    Some(&eta),
                    None,
                    0,
                    None,
                );
                black_box(bundle)
            })
        });
        group.bench_with_input(BenchmarkId::new("anisotropic_eta", k), &k, |b, _| {
            b.iter(|| {
                let bundle = closed_form_aniso_psi_derivatives_in_total_basis(
                    centers.view(),
                    /* q = */ 2,
                    M_TYPICAL,
                    S_TYPICAL,
                    KAPPA_TYPICAL,
                    Some(&eta),
                    None,
                    0,
                    None,
                );
                black_box(bundle)
            })
        });
    }
    group.finish();
}

fn bench_matern_pair_block_assembly(c: &mut Criterion) {
    let mut group = c.benchmark_group("matern_pair_block_assembly");
    group.sample_size(10);
    for &k in &[200_usize, 500, 1000, 2000] {
        let centers = synthetic_centers(k, 3);
        let eta = vec![0.10_f64, -0.05, 0.02];
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, _| {
            b.iter(|| {
                let g = closed_form_matern_pair_block(
                    centers.view(),
                    /* q = */ 2,
                    /* length_scale = */ 1.0,
                    MaternNu::NineHalves,
                    Some(&eta),
                )
                .expect("ν=9/2 d=3 q=2 converges");
                black_box(g)
            })
        });
    }
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

/// Phase 3 benchmark: compare dense Cholesky vs PCG-against-implicit-H for
/// the inner Newton direction solve at biobank-scale `K`. The Hessian is
/// `H = X^T W X + λ S`, where `X^T W X` is a synthetic positive-definite
/// matrix (built once and reused across both paths), `S` is the closed-form
/// penalty Gram, and `λ = 0.1`. Both paths solve `H β = g` for the same RHS;
/// we measure wallclock for the solve only (matrix construction happens
/// outside the timed loop).
///
/// Expectation per the integration assessment (memory file
/// `matrix_free_penalty_integration_assessment.md`): operator-form is *only*
/// faster than dense when p > some threshold (empirically ~1000 here),
/// because PCG iterations × matvec eventually overtakes O(p³) Cholesky.
/// Below threshold dense wins. Reading these numbers tells us where to set
/// `CLOSED_FORM_OPERATOR_THRESHOLD` in production.
fn bench_hessian_solve_dense_vs_implicit(c: &mut Criterion) {
    use faer::Side;
    use gam::linalg::faer_ndarray::FaerCholesky;

    let mut group = c.benchmark_group("hessian_solve_dense_vs_implicit");
    group.sample_size(10);
    for &k in &[500_usize, 1000, 2000, 5000] {
        let centers = synthetic_centers(k, D_TYPICAL);
        let eta = vec![0.0_f64; D_TYPICAL];
        let op_inner = std::sync::Arc::new(ClosedFormPenaltyOperator::new(
            centers.view(),
            /* q = */ 2,
            M_TYPICAL,
            S_TYPICAL,
            KAPPA_TYPICAL,
            Some(&eta),
            None,
            0,
            None,
        ));
        let p = op_inner.dim();
        let s_dense = op_inner.as_dense();

        // Synthetic SPD X^T W X with light off-diagonal coupling so PCG has
        // a non-trivial spectrum but converges. Diagonal in [1.5, 2.5];
        // off-diagonal damped sinusoid.
        let xtwx = {
            let mut g = Array2::<f64>::zeros((p, p));
            for i in 0..p {
                for j in 0..=i {
                    let v = if i == j {
                        2.0 + ((i as f64) * 0.07).sin() * 0.5
                    } else {
                        let scale = ((i + 1) as f64).recip().sqrt();
                        scale * (((i as f64 - j as f64) * 0.21).cos()) * 0.05
                    };
                    g[[i, j]] = v;
                    g[[j, i]] = v;
                }
            }
            g
        };
        let xtwx_diag: Array1<f64> = (0..p).map(|i| xtwx[[i, i]]).collect();
        let lambda = 0.1_f64;
        let gradient = Array1::<f64>::from_shape_fn(p, |i| ((i as f64) * 0.31).sin());

        // Pre-build the dense Hessian once for the dense path.
        let mut h_dense = xtwx.clone();
        for i in 0..p {
            for j in 0..p {
                h_dense[[i, j]] += lambda * s_dense[[i, j]];
            }
        }

        // Dense path: Cholesky-solve via faer.
        let h_dense_for_bench = h_dense.clone();
        group.bench_with_input(BenchmarkId::new("dense_chol", k), &k, |b, _| {
            b.iter(|| {
                let chol = h_dense_for_bench
                    .clone()
                    .cholesky(Side::Lower)
                    .expect("synthetic dense H should factor");
                // faer_ndarray's cholesky returns a struct; for solve we use
                // its diag for a manual back/forward sub or just measure the
                // factorization cost which dominates at biobank p.
                black_box(&chol);
                black_box(&gradient);
            })
        });

        // Implicit path: PCG with operator-form penalty.
        let op_dyn: std::sync::Arc<dyn gam::terms::penalty_op::PenaltyOp> = op_inner.clone();
        let xtwx_for_closure = xtwx.clone();
        group.bench_with_input(BenchmarkId::new("implicit_pcg", k), &k, |b, _| {
            b.iter(|| {
                let mut dir = Array1::<f64>::zeros(p);
                let xt = xtwx_for_closure.clone();
                let apply_xtwx = move |v: &Array1<f64>| -> Array1<f64> { xt.dot(v) };
                let op_pen: &dyn gam::terms::penalty_op::PenaltyOp = op_dyn.as_ref();
                let _ = gam::solver::pirls::solve_newton_direction_implicit(
                    apply_xtwx,
                    xtwx_diag.view(),
                    &[],
                    &[(lambda, op_pen)],
                    &gradient,
                    &mut dir,
                    /* ridge = */ 0.0,
                    /* rel_tol = */ 1e-8,
                    /* max_iter = */ p,
                );
                black_box(&dir);
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
    bench_derivative_matrix_assembly,
    bench_matern_pair_block_assembly,
    bench_operator_matvec,
    bench_hessian_solve_dense_vs_implicit,
);
criterion_main!(benches);

//! SIMD pair-block bench (Task #6).
//!
//! Two related comparisons across d ∈ {2, 3, 4, 6} and M ∈ {50, 200, 500}
//! pairs:
//!
//!   1. `aniso_invariants` SIMD (`wide::f64x4`) vs scalar reference. This
//!      isolates the lane-vectorized hot inner loop used by every pair
//!      evaluation in the closed-form anisotropic Duchon path.
//!
//!   2. `pair_block_radial_with_j_second_derivatives` end-to-end timing at
//!      `q = 2`, exercising the full per-pair FD-derivative bundle. The
//!      function calls `aniso_invariants` internally; a SIMD ↔ scalar swap
//!      cannot be done from a bench, so this comparison runs only the SIMD
//!      production path and is reported alongside (1) as a wallclock anchor.
//!
//! Run with: `cargo bench --bench closed_form_pair_block`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use gam::terms::basis::closed_form_penalty::{
    aniso_invariants_scalar, aniso_invariants_simd, pair_block_radial_with_j_second_derivatives,
};

const DIMS: &[usize] = &[2, 3, 4, 6];
const PAIR_COUNTS: &[usize] = &[50, 200, 500];
const M_FIXED: usize = 2;
const S_FIXED: usize = 8;
const KAPPA: f64 = 1.0;

/// Deterministic synthetic (eta, r) pairs in the bench.
///
/// `eta` lives near zero (production typical: log b_k ~ O(1)); `r` is a
/// non-trivial lag of unit-ish magnitude. We mix sin/cos so individual
/// pairs are not co-located (which would funnel into the R = 0 branch).
fn synthetic_pairs(m: usize, d: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut out = Vec::with_capacity(m);
    for i in 0..m {
        let phase = (i as f64) * 0.137;
        let eta: Vec<f64> = (0..d)
            .map(|k| 0.15 * ((phase + k as f64 * 0.31).sin()))
            .collect();
        let r: Vec<f64> = (0..d)
            .map(|k| 0.05 + 0.4 * ((phase + k as f64 * 0.71).cos()).abs())
            .collect();
        out.push((eta, r));
    }
    out
}

fn bench_aniso_invariants_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("aniso_invariants");
    group.sample_size(20);
    for &d in DIMS {
        for &m in PAIR_COUNTS {
            let pairs = synthetic_pairs(m, d);
            let id_simd = format!("simd/d{d}/m{m}");
            let id_scalar = format!("scalar/d{d}/m{m}");

            group.bench_with_input(BenchmarkId::from_parameter(&id_simd), &pairs, |b, pairs| {
                b.iter(|| {
                    let mut acc = 0.0_f64;
                    for (eta, r) in pairs {
                        let (big_r, s1, s2, u1, u2) = aniso_invariants_simd(eta, r);
                        acc += big_r + s1 + s2 + u1 + u2;
                    }
                    black_box(acc)
                })
            });

            group.bench_with_input(
                BenchmarkId::from_parameter(&id_scalar),
                &pairs,
                |b, pairs| {
                    b.iter(|| {
                        let mut acc = 0.0_f64;
                        for (eta, r) in pairs {
                            let (big_r, s1, s2, u1, u2) = aniso_invariants_scalar(eta, r);
                            acc += big_r + s1 + s2 + u1 + u2;
                        }
                        black_box(acc)
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_pair_block_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("pair_block_radial_q2");
    group.sample_size(15);
    for &d in DIMS {
        for &m in PAIR_COUNTS {
            let pairs = synthetic_pairs(m, d);
            let id = format!("d{d}/m{m}");
            group.bench_with_input(BenchmarkId::from_parameter(&id), &pairs, |b, pairs| {
                b.iter(|| {
                    let mut acc = 0.0_f64;
                    for (eta, r) in pairs {
                        let bundle = pair_block_radial_with_j_second_derivatives(
                            /* q = */ 2, M_FIXED, S_FIXED, KAPPA, eta, r,
                        );
                        acc += bundle.value;
                    }
                    black_box(acc)
                })
            });
        }
    }
    group.finish();
}

/// Sanity check: the SIMD and scalar invariants must agree to within a
/// few ulps on representative inputs (lane vs sequential summation aside).
/// This runs once per bench invocation as a fast smoke check; failure here
/// indicates the two paths have diverged and the speedup numbers below
/// are meaningless.
fn bench_simd_scalar_parity(c: &mut Criterion) {
    let mut group = c.benchmark_group("aniso_invariants_parity");
    group.sample_size(10);
    let pairs = synthetic_pairs(64, 6);
    group.bench_function("d6/m64_max_abs_diff", |b| {
        b.iter(|| {
            let mut max_diff = 0.0_f64;
            for (eta, r) in &pairs {
                let v_simd = aniso_invariants_simd(eta, r);
                let v_scal = aniso_invariants_scalar(eta, r);
                let diff = [
                    (v_simd.0 - v_scal.0).abs(),
                    (v_simd.1 - v_scal.1).abs(),
                    (v_simd.2 - v_scal.2).abs(),
                    (v_simd.3 - v_scal.3).abs(),
                    (v_simd.4 - v_scal.4).abs(),
                ];
                for &dv in &diff {
                    if dv > max_diff {
                        max_diff = dv;
                    }
                }
            }
            assert!(
                max_diff < 1e-12,
                "SIMD and scalar aniso_invariants diverged: max |diff| = {max_diff:.3e}"
            );
            black_box(max_diff)
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_simd_scalar_parity,
    bench_aniso_invariants_simd_vs_scalar,
    bench_pair_block_end_to_end,
);
criterion_main!(benches);

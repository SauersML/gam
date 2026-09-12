//! Pair-block bench: `pair_block_radial_with_j_second_derivatives` end-to-end
//! timing at `q = 2` across d ∈ {2, 3, 4, 6} and M ∈ {50, 200, 500} pairs,
//! exercising the full per-pair FD-derivative bundle.
//!
//! Run with: `cargo bench --bench closed_form_pair_block`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use gam::terms::basis::closed_form_penalty::pair_block_radial_with_j_second_derivatives;

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

criterion_group!(benches, bench_pair_block_end_to_end);
criterion_main!(benches);

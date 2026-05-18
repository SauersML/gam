//! Ignored-by-default Criterion workload for affine tail-cell moment memoization
//! on a biobank-shape Hessian-vector pattern.
//!
//! Run with: `cargo bench --bench tail_cell_memo_biobank_shape`

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use gam::families::cubic_cell_kernel::{
    DenestedCubicCell, evaluate_cell_moments, evaluate_cell_moments_uncached,
    reset_tail_cell_moment_cache, set_tail_cell_moment_cache_enabled, tail_cell_moment_cache_stats,
};

fn biobank_tail_cells(n_rows: usize) -> Vec<DenestedCubicCell> {
    let mut cells = Vec::with_capacity(2 * n_rows);
    let left_endpoint = -3.0;
    let right_endpoint = 3.0;
    // Deliberately much lower coefficient diversity than row count, matching
    // shared affine tail remainders in the FLEX biobank PIRLS/Hv workload.
    let c0s = [-0.8, -0.2, 0.35, 0.9];
    let c1s = [-0.6, -0.1, 0.25, 0.7];
    for row in 0..n_rows {
        let c0 = c0s[row % c0s.len()];
        let c1 = c1s[(row / c0s.len()) % c1s.len()];
        cells.push(DenestedCubicCell {
            left: f64::NEG_INFINITY,
            right: left_endpoint,
            c0,
            c1,
            c2: 0.0,
            c3: 0.0,
        });
        cells.push(DenestedCubicCell {
            left: right_endpoint,
            right: f64::INFINITY,
            c0,
            c1,
            c2: 0.0,
            c3: 0.0,
        });
    }
    cells
}

fn bench_tail_cell_memo_biobank_shape(c: &mut Criterion) {
    let cells = biobank_tail_cells(32_000);
    let max_degree = 9;

    c.bench_function("tail_cell_moments_biobank_shape_uncached", |b| {
        b.iter(|| {
            set_tail_cell_moment_cache_enabled(false);
            let mut acc = 0.0;
            for &cell in &cells {
                let state = evaluate_cell_moments_uncached(black_box(cell)(max_degree))
                    .expect("uncached tail moments");
                acc += state.value + state.moments[0];
            }
            black_box(acc)
        })
    });

    c.bench_function("tail_cell_moments_biobank_shape_cached", |b| {
        b.iter(|| {
            set_tail_cell_moment_cache_enabled(true);
            reset_tail_cell_moment_cache();
            let mut acc = 0.0;
            for &cell in &cells {
                let state = evaluate_cell_moments(black_box(cell)(max_degree))
                    .expect("cached tail moments");
                acc += state.value + state.moments[0];
            }
            let stats = tail_cell_moment_cache_stats();
            assert!(stats.hit_rate() > 0.99, "tail memo stats: {stats:?}");
            black_box(acc)
        })
    });
}

criterion_group!(benches, bench_tail_cell_memo_biobank_shape);
criterion_main!(benches);

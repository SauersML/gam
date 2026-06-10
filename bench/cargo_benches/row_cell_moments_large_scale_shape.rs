//! Criterion micro-benchmark for the bernoulli marginal-slope FLEX row-cell
//! moment access pattern.
//!
//! This is intentionally not run by default test commands. Run explicitly with:
//! `cargo bench --bench row_cell_moments_biobank_shape`.

use criterion::{Criterion, criterion_group, criterion_main};
use gam::families::cubic_cell_kernel::{DenestedCubicCell, evaluate_cell_moments};
use std::hint::black_box;

fn synthetic_row_cells(row: usize) -> [DenestedCubicCell; 5] {
    let shift = (row as f64).sin() * 0.02;
    [
        DenestedCubicCell {
            left: -2.5,
            right: -1.0,
            c0: -0.5 + shift,
            c1: 0.7,
            c2: 0.012,
            c3: -0.004,
        },
        DenestedCubicCell {
            left: -1.0,
            right: -0.2,
            c0: -0.1 + shift,
            c1: 0.4,
            c2: -0.018,
            c3: 0.002,
        },
        DenestedCubicCell {
            left: -0.2,
            right: 0.55,
            c0: 0.2 + shift,
            c1: -0.3,
            c2: 0.015,
            c3: 0.003,
        },
        DenestedCubicCell {
            left: 0.55,
            right: 1.4,
            c0: 0.35 + shift,
            c1: 0.2,
            c2: -0.01,
            c3: -0.002,
        },
        DenestedCubicCell {
            left: 1.4,
            right: 2.75,
            c0: 0.55 + shift,
            c1: -0.15,
            c2: 0.008,
            c3: 0.0015,
        },
    ]
}

fn bench_row_cell_moments_biobank_shape(c: &mut Criterion) {
    const ROWS: usize = 512;
    const HV_CALLS: usize = 64;
    let rows: Vec<_> = (0..ROWS).map(synthetic_row_cells).collect();
    let mut group = c.benchmark_group("row_cell_moments_biobank_shape");
    group.sample_size(10);

    group.bench_function("before_scattered_degree9_per_hv", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for _ in 0..HV_CALLS {
                for cells in &rows {
                    for cell in cells {
                        let state = evaluate_cell_moments(*cell, 9).expect("cell moments");
                        acc += state.value + state.moments[0];
                    }
                }
            }
            black_box(acc)
        })
    });

    group.bench_function("after_batched_degree21_reuse", |b| {
        b.iter(|| {
            let cached: Vec<Vec<_>> = rows
                .iter()
                .map(|cells| {
                    cells
                        .iter()
                        .map(|cell| evaluate_cell_moments(*cell, 21).expect("cell moments"))
                        .collect()
                })
                .collect();
            let mut acc = 0.0;
            for _ in 0..HV_CALLS {
                for states in &cached {
                    for state in states {
                        acc += state.value + state.moments[0];
                    }
                }
            }
            black_box(acc)
        })
    });
    group.finish();
}

criterion_group!(benches, bench_row_cell_moments_biobank_shape);
criterion_main!(benches);

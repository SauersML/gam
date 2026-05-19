//! Ignored-by-default Criterion workload for the biobank-shape de-nested cell
//! moment Hv pattern. It compares cold uncached evaluation with a fit-lifetime
//! byte-LRU warmed across repeated PIRLS-like cycles.
//!
//! Run with: `cargo bench --bench cell_moment_lru_biobank_shape`

use criterion::{Criterion, criterion_group, criterion_main};
use gam::families::cubic_cell_kernel::{
    CellMomentCacheStats, CellMomentLruCache, DenestedCubicCell, evaluate_cell_moments,
    evaluate_cell_moments_cached,
};
use std::hint::black_box;

fn biobank_shape_cells() -> Vec<DenestedCubicCell> {
    let mut cells = Vec::new();
    let bounds = [
        (-3.0, -1.5),
        (-1.5, -0.5),
        (-0.5, 0.5),
        (0.5, 1.5),
        (1.5, 3.0),
    ];
    let intercepts = [-0.8, -0.25, 0.15, 0.7];
    let slopes = [-1.1, -0.35, 0.45, 1.25];
    for &c0 in &intercepts {
        for &c1 in &slopes {
            for (idx, &(left, right)) in bounds.iter().enumerate() {
                let scale = (idx as f64 + 1.0) * 1.0e-3;
                cells.push(DenestedCubicCell {
                    left,
                    right,
                    c0,
                    c1,
                    c2: scale * (1.0 + c0),
                    c3: -0.5 * scale * c1,
                });
            }
        }
    }
    cells
}

fn bench_cell_moment_lru_biobank_shape(c: &mut Criterion) {
    let cells = biobank_shape_cells();
    c.bench_function("cell_moments_biobank_shape_uncached", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for _cycle in 0..8 {
                for _hv in 0..64 {
                    for &cell in &cells {
                        acc += evaluate_cell_moments(black_box(cell), 9)
                            .expect("cell moments")
                            .value;
                    }
                }
            }
            black_box(acc)
        })
    });

    c.bench_function("cell_moments_biobank_shape_fit_lru", |b| {
        b.iter(|| {
            let cache = CellMomentLruCache::new(256 * 1024 * 1024);
            let stats = CellMomentCacheStats::default();
            let mut acc = 0.0;
            for _cycle in 0..8 {
                for _hv in 0..64 {
                    for &cell in &cells {
                        acc +=
                            evaluate_cell_moments_cached(black_box(cell), 9, &cache, Some(&stats))
                                .expect("cell moments")
                                .value;
                    }
                }
            }
            let (hits, misses) = stats.snapshot();
            black_box((acc, hits, misses))
        })
    });
}

criterion_group!(benches, bench_cell_moment_lru_biobank_shape);
criterion_main!(benches);

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use gam::families::cubic_cell_kernel::{
    DenestedCubicCell, cell_moment_cache_key, evaluate_cell_moments,
};
use std::collections::HashMap;

fn synthetic_biobank_cells(n_rows: usize, cells_per_row: usize) -> Vec<DenestedCubicCell> {
    let prototypes: Vec<_> = (0..128)
        .map(|i| {
            let t = i as f64 / 127.0;
            DenestedCubicCell {
                left: -2.0 + 3.0 * t,
                right: -1.75 + 3.0 * t,
                c0: -0.25 + 0.5 * t,
                c1: -0.4 + 0.8 * ((i * 17 % 128) as f64 / 127.0),
                c2: -0.08 + 0.16 * ((i * 31 % 128) as f64 / 127.0),
                c3: -0.015 + 0.03 * ((i * 47 % 128) as f64 / 127.0),
            }
        })
        .collect();
    (0..n_rows * cells_per_row)
        .map(|idx| prototypes[idx % prototypes.len()])
        .collect()
}

fn evaluate_uncached(cells: &[DenestedCubicCell]) -> f64 {
    cells
        .iter()
        .map(|&cell| evaluate_cell_moments(cell, 9).expect("moments").value)
        .sum()
}

fn evaluate_dedup(cells: &[DenestedCubicCell], epsilon: f64) -> (f64, usize, usize) {
    let mut cache = HashMap::new();
    let mut hits = 0usize;
    let mut misses = 0usize;
    let mut total = 0.0;
    for &cell in cells {
        let key = cell_moment_cache_key(cell, 9, epsilon);
        if let Some(value) = cache.get(&key) {
            hits += 1;
            total += *value;
        } else {
            misses += 1;
            let value = evaluate_cell_moments(cell, 9).expect("moments").value;
            cache.insert(key, value);
            total += value;
        }
    }
    (total, hits, misses)
}

fn bench_cell_moment_dedup_biobank_shape(c: &mut Criterion) {
    // Criterion benchmarks are opt-in (run explicitly with this bench target),
    // which keeps the biobank-shaped workload out of normal test runs.
    let cells = synthetic_biobank_cells(8_192, 4);
    let mut group = c.benchmark_group("cell_moment_dedup_biobank_shape");
    group.bench_function("uncached", |b| {
        b.iter(|| black_box(evaluate_uncached(&cells)))
    });
    for epsilon in [0.0, 1e-12, 1e-10, 1e-8, 1e-6] {
        group.bench_with_input(BenchmarkId::new("dedup", epsilon), &epsilon, |b, &eps| {
            b.iter(|| black_box(evaluate_dedup(&cells, eps)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_cell_moment_dedup_biobank_shape);
criterion_main!(benches);

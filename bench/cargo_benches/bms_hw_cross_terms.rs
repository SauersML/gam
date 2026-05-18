use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use gam::faer_ndarray::{fast_atb, fast_atv};
use ndarray::Array2;

fn deterministic_matrix(rows: usize, cols: usize, a: f64, b: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        let x = (i as f64 + 1.0) * a + (j as f64 + 0.5) * b;
        x.sin() + 0.25 * (0.37 * x).cos()
    })
}

struct HwCrossFixture {
    x: Array2<f64>,
    g: Array2<f64>,
    hq: Array2<f64>,
    hg: Array2<f64>,
    wq: Array2<f64>,
    wg: Array2<f64>,
}

impl HwCrossFixture {
    fn new(n: usize, pm: usize, pg: usize, ph: usize, pw: usize) -> Self {
        Self {
            x: deterministic_matrix(n, pm, 0.013, 0.071),
            g: deterministic_matrix(n, pg, 0.017, 0.067),
            hq: deterministic_matrix(n, ph, 0.019, 0.061),
            hg: deterministic_matrix(n, ph, 0.023, 0.059),
            wq: deterministic_matrix(n, pw, 0.029, 0.053),
            wg: deterministic_matrix(n, pw, 0.031, 0.047),
        }
    }
}

fn old_columnwise_checksum(f: &HwCrossFixture) -> f64 {
    let mut checksum = 0.0;
    for k in 0..f.hq.ncols() {
        checksum += fast_atv(&f.x, &f.hq.column(k)).sum();
    }
    for k in 0..f.hg.ncols() {
        checksum += fast_atv(&f.g, &f.hg.column(k)).sum();
    }
    for k in 0..f.wq.ncols() {
        checksum += fast_atv(&f.x, &f.wq.column(k)).sum();
    }
    for k in 0..f.wg.ncols() {
        checksum += fast_atv(&f.g, &f.wg.column(k)).sum();
    }
    checksum
}

fn new_batched_checksum(f: &HwCrossFixture) -> f64 {
    fast_atb(&f.x, &f.hq).sum()
        + fast_atb(&f.g, &f.hg).sum()
        + fast_atb(&f.x, &f.wq).sum()
        + fast_atb(&f.g, &f.wg).sum()
}

fn bench_bms_hw_cross_terms(c: &mut Criterion) {
    gam::init_parallelism();
    let fixture = HwCrossFixture::new(16_384, 96, 96, 4, 4);
    let old = old_columnwise_checksum(&fixture);
    let new = new_batched_checksum(&fixture);
    let rel = (old - new).abs() / old.abs().max(1.0);
    assert!(
        rel < 1e-11,
        "batched cross terms changed checksum: old={old} new={new} rel={rel}"
    );

    let mut group = c.benchmark_group("bms_hw_cross_terms_exact_batching");
    group.bench_function("old_columnwise_fast_atv", |b| {
        b.iter(|| black_box(old_columnwise_checksum(black_box(&fixture))))
    });
    group.bench_function("new_batched_fast_atb", |b| {
        b.iter(|| black_box(new_batched_checksum(black_box(&fixture))))
    });
    group.finish();
}

criterion_group!(benches, bench_bms_hw_cross_terms);
criterion_main!(benches);

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use gam::families::cubic_cell_kernel::{DenestedCubicCell, evaluate_cell_moments};
use std::hint::black_box;

fn biobank_shape_cells(n: usize) -> Vec<(f64, [DenestedCubicCell; 3])> {
    (0..n)
        .map(|i| {
            let phase = (i as f64 * 0.00137).sin();
            let importance = if i % 97 == 0 {
                1e-14
            } else {
                1.0 + phase.abs()
            };
            let c2 = 0.02 * phase;
            let c3 = 0.01 * (i as f64 * 0.00091).cos();
            (
                importance,
                [
                    DenestedCubicCell {
                        left: -1.6,
                        right: -0.35,
                        c0: -0.15,
                        c1: 0.8,
                        c2,
                        c3,
                    },
                    DenestedCubicCell {
                        left: -0.35,
                        right: 0.45,
                        c0: 0.05,
                        c1: 0.9,
                        c2: -0.7 * c2,
                        c3: 0.5 * c3,
                    },
                    DenestedCubicCell {
                        left: 0.45,
                        right: 1.7,
                        c0: 0.2,
                        c1: 0.75,
                        c2: 0.4 * c2,
                        c3: -0.6 * c3,
                    },
                ],
            )
        })
        .collect()
}

fn run_hv_shape(rows: &[(f64, [DenestedCubicCell; 3])], tau: f64) -> f64 {
    let mean = rows.iter().map(|(w, _)| *w).sum::<f64>() / rows.len().max(1) as f64;
    let threshold = tau * mean;
    let mut checksum = 0.0;
    for (importance, cells) in rows {
        if tau > 0.0 && *importance < threshold {
            continue;
        }
        for cell in cells {
            let state = evaluate_cell_moments(*cell, 9).expect("cell moments");
            checksum += state.value + state.moments.iter().sum::<f64>() * 1e-12;
        }
    }
    checksum
}

fn bench_bms_hv_row_skip(c: &mut Criterion) {
    let rows = biobank_shape_cells(32_000);
    let mut group = c.benchmark_group("bms_hv_row_skip_biobank_shape");
    for tau in [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("tau={tau:.0e}")),
            &tau,
            |b, &tau| b.iter(|| black_box(run_hv_shape(black_box(&rows), tau))),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_bms_hv_row_skip);
criterion_main!(benches);

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use gam::families::cubic_cell_kernel::{DenestedCubicCell, evaluate_cell_moments};

fn biobank_shape_cells() -> Vec<DenestedCubicCell> {
    (0..64)
        .map(|i| {
            let lane = i as f64;
            let left = -1.25 + 0.035 * (lane % 17.0);
            let right = left + 0.42 + 0.01 * (lane % 5.0);
            DenestedCubicCell {
                left,
                right,
                c0: -0.35 + 0.011 * lane,
                c1: 0.55 - 0.003 * lane,
                c2: -0.06 + 0.002 * (lane % 19.0),
                c3: 0.012 - 0.00035 * (lane % 23.0),
            }
        })
        .collect()
}

fn bench_non_affine_cell_hv_shape(c: &mut Criterion) {
    let cells = biobank_shape_cells();
    let mut group = c.benchmark_group("non_affine_cell_hv_shape");
    for max_degree in [4_usize, 12, 24] {
        group.bench_with_input(
            BenchmarkId::new("closed_form_transport", max_degree),
            &max_degree,
            |b, &max_degree| {
                b.iter(|| {
                    let mut acc = 0.0;
                    for _row in 0..512 {
                        for &cell in &cells {
                            let state = evaluate_cell_moments(black_box(cell), max_degree)
                                .expect("cell moments");
                            acc += state.value + state.moments[max_degree];
                        }
                    }
                    black_box(acc)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_non_affine_cell_hv_shape);
criterion_main!(benches);

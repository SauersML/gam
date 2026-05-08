use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn paired_objective_delta(
    rows: &[usize],
    per_row_old: &[f64],
    per_row_trial: &[f64],
    scale: f64,
) -> f64 {
    let old = rows.iter().map(|&i| per_row_old[i]).sum::<f64>() * scale;
    let trial = rows.iter().map(|&i| per_row_trial[i]).sum::<f64>() * scale;
    -(trial - old)
}

fn bench_line_search_subsample_screen(c: &mut Criterion) {
    // Ignored-style Criterion benchmark: it is not part of `cargo test`, and
    // must be run explicitly with
    // `cargo bench --bench line_search_subsample_screen`. The row counts mimic
    // the biobank line-search pattern: exact full LL over 320k rows versus a
    // fixed ~20k paired outer-score subsample used for accept/reject screening.
    let n = 320_000usize;
    let k = 20_091usize;
    let old: Vec<f64> = (0..n)
        .map(|i| -0.7 - ((i * 17 + 3) % 101) as f64 * 1e-5)
        .collect();
    let trial: Vec<f64> = old
        .iter()
        .enumerate()
        .map(|(i, &v)| v + (((i * 29 + 7) % 53) as f64 - 26.0) * 1e-7)
        .collect();
    let full_rows: Vec<usize> = (0..n).collect();
    let subsample_rows: Vec<usize> = (0..k).map(|j| (j * 15_919 + 7) % n).collect();
    let scale = n as f64 / k as f64;

    let mut group = c.benchmark_group("line_search_ll_screen_biobank_shape");
    group.bench_function("full_ll_each_attempt_320k", |b| {
        b.iter(|| {
            paired_objective_delta(
                black_box(&full_rows),
                black_box(&old),
                black_box(&trial),
                1.0,
            )
        })
    });
    group.bench_function("paired_subsample_ll_20k", |b| {
        b.iter(|| {
            paired_objective_delta(
                black_box(&subsample_rows),
                black_box(&old),
                black_box(&trial),
                black_box(scale),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, bench_line_search_subsample_screen);
criterion_main!(benches);

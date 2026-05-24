use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::Array1;
use std::hint::black_box;

fn backtracking_pattern(radius: f64, newton_step: &Array1<f64>) -> (usize, f64) {
    let mut attempts = 0usize;
    for bt in 0i32..8 {
        attempts += 1;
        let alpha = 0.5f64.powi(bt);
        let trial = newton_step.mapv(|v| alpha * v);
        let norm = trial.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm <= radius {
            return (attempts, norm);
        }
    }
    (attempts, 0.0)
}

fn trust_region_pattern(radius: f64, newton_step: &Array1<f64>) -> (usize, f64) {
    let norm = newton_step.iter().map(|v| v * v).sum::<f64>().sqrt();
    let scale = if norm > radius { radius / norm } else { 1.0 };
    black_box((1usize, norm * scale))
}

fn bench_biobank_shape_line_search_pattern(c: &mut Criterion) {
    let p = 256usize;
    let newton_step = Array1::from_iter((0..p).map(|i| 0.25 + (i as f64 % 17.0) / 17.0));
    let radius = newton_step.iter().map(|v| v * v).sum::<f64>().sqrt() / 16.0;

    let mut group = c.benchmark_group("biobank_shape_line_search_pattern");
    group.bench_function("before_backtracking_accepts_fifth", |b| {
        b.iter(|| backtracking_pattern(black_box(radius), black_box(&newton_step)))
    });
    group.bench_function("after_trust_region_single_trial", |b| {
        b.iter(|| trust_region_pattern(black_box(radius), black_box(&newton_step)))
    });
    group.finish();
}

criterion_group!(benches, bench_biobank_shape_line_search_pattern);
criterion_main!(benches);

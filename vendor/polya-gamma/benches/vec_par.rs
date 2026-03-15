use criterion::{Criterion, black_box, criterion_group, criterion_main};
use polya_gamma::PolyaGamma;
use rand::{SeedableRng, rngs::StdRng};

fn bench_pg1_deterministic(c: &mut Criterion) {
    let n = 100_000;
    let b = 1.0;
    let z = 0.5;
    let pg = PolyaGamma::new(b);
    let mut rng = StdRng::seed_from_u64(42);
    c.bench_function("draw_vec_par_deterministic", |bencher| {
        bencher.iter(|| {
            let out = pg.draw_vec_par_deterministic(&mut rng, &vec![z; n]);
            black_box(out);
        });
    });
}

fn bench_pg1_vec_par(c: &mut Criterion) {
    let b = 1.0;
    let n = 100_000;
    let z = 0.5;
    let pg = PolyaGamma::new(b);
    c.bench_function("draw_vec_par", |bencher| {
        bencher.iter(|| {
            let out = pg.draw_vec_par(&vec![z; n]);
            black_box(out);
        });
    });
}

fn bench_pg1_no_par(c: &mut Criterion) {
    let b = 1.0;
    let n: usize = 100_000;
    let z = 0.5;
    let pg = PolyaGamma::new(b);
    let mut rng = StdRng::seed_from_u64(42);
    c.bench_function("draw_vec", |bencher| {
        bencher.iter(|| {
            let out = pg.draw_vec(&mut rng, &vec![z; n]);
            black_box(out);
        });
    });
}

criterion_group!(
    benches,
    bench_pg1_deterministic,
    bench_pg1_vec_par,
    bench_pg1_no_par
);
criterion_main!(benches);

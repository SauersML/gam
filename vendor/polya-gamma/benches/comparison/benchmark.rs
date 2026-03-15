use criterion::{Criterion, criterion_group, criterion_main};
use polya_gamma::PolyaGamma;
use rand::{SeedableRng, rngs::StdRng};
use std::time::Duration;

fn bench_polya_gamma(cr: &mut Criterion) {
    let mut pg = PolyaGamma::new(1.0);
    let mut rng = StdRng::seed_from_u64(42);

    // Fixed number of samples to match R benchmark
    let ns = [10000];
    let bs = [0.5, 1.0, 5.0, 10.0];
    let cs = [0.0, 0.5, 1.0, 2.0];

    for &b in &bs {
        pg.set_shape(b);
        for &n in &ns {
            for &c in &cs {
                let bench_name = format!("draw_n{}_b{:.1}_c{:.1}", n, b, c);
                cr.bench_function(&bench_name, |bench| {
                    bench.iter(|| {
                        for _ in 0..n {
                            pg.draw(&mut rng, c);
                        }
                    });
                });
            }
        }
    }
}

#[cfg(feature = "rayon")]
fn bench_polya_gamma_par(cr: &mut Criterion) {
    let mut pg = PolyaGamma::new(1.0);

    // Fixed number of samples to match R benchmark
    let ns = [10000];
    let bs = [0.5, 1.0, 5.0, 10.0];
    let cs = [0.0, 0.5, 1.0, 2.0];

    for &b in &bs {
        pg.set_shape(b);
        for &n in &ns {
            for &c in &cs {
                let bench_name = format!("draw_par_n{}_b{:.1}_c{:.1}", n, b, c);
                cr.bench_function(&bench_name, |bench| {
                    bench.iter(|| {
                        pg.draw_vec_par(&vec![c; n]);
                    });
                });
            }
        }
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
        .sample_size(10);
    targets = bench_polya_gamma
);

// Only include parallel benchmarks if rayon feature is enabled
#[cfg(feature = "rayon")]
criterion_group!(
    name = parallel_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
        .sample_size(10);
    targets = bench_polya_gamma_par
);

#[cfg(not(feature = "rayon"))]
criterion_main!(benches);

#[cfg(feature = "rayon")]
criterion_main!(benches, parallel_benches);

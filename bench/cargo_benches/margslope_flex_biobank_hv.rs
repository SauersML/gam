//! Criterion workload for the biobank-shape bernoulli marginal-slope
//! FLEX joint-Newton/Hv path. Multi-minute fit — run explicitly with:
//!
//! ```text
//! cargo bench --bench margslope_flex_biobank_hv
//! ```

#[path = "../../tests/test_support/margslope_flex_equivalence.rs"]
mod margslope_flex_equivalence;

use criterion::{Criterion, criterion_group, criterion_main};
use margslope_flex_equivalence::{
    DEFAULT_REPRO_N, build_biobank_shape_problem, cycle_capped_options, fit_problem,
};
use std::hint::black_box;
use std::time::Duration;

const BENCH_INNER_CYCLES: usize = 1;

fn bench_margslope_flex_biobank_cycle0(c: &mut Criterion) {
    gam::init_parallelism();
    let n = DEFAULT_REPRO_N;
    let inner_cycles = BENCH_INNER_CYCLES;
    let mut group = c.benchmark_group("margslope_flex_biobank_hv_pattern");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    group.bench_function(format!("n{n}_inner{inner_cycles}"), |b| {
        b.iter(|| {
            let problem = build_biobank_shape_problem(black_box(n));
            let (fit, timing) = fit_problem(problem, cycle_capped_options(inner_cycles))
                .expect("criterion biobank-shape margslope fit");
            eprintln!(
                "[MS-FLEX-BIOBANK-BENCH-ITER] n={} inner_max_cycles={} elapsed_s={:.3} outer_iters={} inner_cycles={} converged={} beta_len={}",
                n,
                inner_cycles,
                timing.elapsed.as_secs_f64(),
                timing.outer_iterations,
                timing.inner_cycles,
                timing.outer_converged,
                fit.fit.beta.len()
            );
            black_box(fit.fit.beta.len())
        });
    });
    group.finish();
}

criterion_group!(benches, bench_margslope_flex_biobank_cycle0);
criterion_main!(benches);

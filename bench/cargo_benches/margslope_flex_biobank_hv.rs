//! Opt-in Criterion workload for the biobank-shape bernoulli marginal-slope
//! FLEX joint-Newton/Hv path.
//!
//! This benchmark is intentionally environment-gated so normal `cargo bench`
//! invocations do not launch a multi-minute fit.  Run it explicitly with:
//!
//! ```text
//! GAM_RUN_MARGSLOPE_FLEX_BIOBANK_BENCH=1 \
//! GAM_MARGSLOPE_BENCH_N=50000 \
//! cargo bench --bench margslope_flex_biobank_hv
//! ```
//!
//! The measured routine performs a full synthetic fit capped at
//! `inner_max_cycles=1`, matching the local ignored repro test.

#[allow(dead_code)]
#[path = "../../tests/test_support/margslope_flex_equivalence.rs"]
mod margslope_flex_equivalence;

use criterion::{Criterion, criterion_group, criterion_main};
use margslope_flex_equivalence::{
    DEFAULT_REPRO_N, build_biobank_shape_problem, cycle_capped_options, fit_problem,
};
use std::hint::black_box;
use std::time::Duration;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn bench_margslope_flex_biobank_cycle0(c: &mut Criterion) {
    if std::env::var("GAM_RUN_MARGSLOPE_FLEX_BIOBANK_BENCH").as_deref() != Ok("1") {
        c.bench_function("margslope_flex_biobank_cycle0_ignored", |b| {
            b.iter(|| black_box(0usize));
        });
        eprintln!(
            "[MS-FLEX-BIOBANK-BENCH] skipped; set GAM_RUN_MARGSLOPE_FLEX_BIOBANK_BENCH=1 to run the multi-minute Hv-pattern benchmark"
        );
        return;
    }

    gam::init_parallelism();
    let n = env_usize("GAM_MARGSLOPE_BENCH_N", DEFAULT_REPRO_N);
    let inner_cycles = env_usize("GAM_MARGSLOPE_BENCH_INNER_CYCLES", 1);
    let mut group = c.benchmark_group("margslope_flex_biobank_hv_pattern");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    group.bench_function(format!("n{n}_inner{inner_cycles}"), |b| {
        b.iter(|| {
            let problem = build_biobank_shape_problem(black_box(n));
            let (fit, timing) = fit_problem(problem, cycle_capped_options(inner_cycles))
                .expect("criterion biobank-shape margslope fit");
            eprintln!(
                "[MS-FLEX-BIOBANK-BENCH-ITER] n={} inner_max_cycles={} elapsed_s={:.3} inner_cycles={} beta_len={}",
                n,
                inner_cycles,
                timing.elapsed.as_secs_f64(),
                timing.inner_cycles,
                fit.fit.beta.len()
            );
            black_box(fit.fit.beta.len())
        });
    });
    group.finish();
}

criterion_group!(benches, bench_margslope_flex_biobank_cycle0);
criterion_main!(benches);

//! Criterion workload for the large-scale bernoulli marginal-slope
//! FLEX joint-Newton/Hv path. Multi-minute fit — run explicitly with:
//!
//! ```text
//! cargo bench --bench margslope_flex_large_scale_hv
//! ```

#[path = "../../tests/test_support/misc/margslope_flex_equivalence.rs"]
mod margslope_flex_equivalence;

use criterion::{Criterion, criterion_group, criterion_main};
use margslope_flex_equivalence::{
    build_large_scale_shape_problem, cycle_capped_options, fit_problem,
};
use std::hint::black_box;
use std::time::Duration;

const DEFAULT_REPRO_N: usize = 50_000;
const BENCH_INNER_CYCLES: usize = 1;
#[cfg(target_os = "linux")]
const BENCH_MULTI_RHS_PROBE: usize = 4;
#[cfg(target_os = "linux")]
const LARGE_SCALE_HVP_PRIMARY_R: usize = 20;
#[cfg(target_os = "linux")]
const LARGE_SCALE_HVP_P_TOTAL: usize = 44;

fn bench_margslope_flex_large_scale_cycle0(c: &mut Criterion) {
    gam::init_parallelism();
    let n = DEFAULT_REPRO_N;
    let inner_cycles = BENCH_INNER_CYCLES;
    #[cfg(target_os = "linux")]
    {
        let scratch = gam::families::bms::gpu::row::bms_flex_row_hvp_multi_scratch_bytes_for_shape(
            n,
            LARGE_SCALE_HVP_P_TOTAL,
            BENCH_MULTI_RHS_PROBE,
        )
        .expect("large-scale multi-RHS HVP scratch budget");
        let per_rhs_full_row_cache = (n
            * LARGE_SCALE_HVP_PRIMARY_R
            * LARGE_SCALE_HVP_PRIMARY_R
            * std::mem::size_of::<f64>()) as u64
            * BENCH_MULTI_RHS_PROBE as u64;
        eprintln!(
            "[MS-FLEX-LARGE_SCALE-BENCH-HVP-MULTI-RHS] n={} p={} r={} rhs={} scratch_mib={:.3} full_row_cache_per_rhs_mib={:.3}",
            n,
            LARGE_SCALE_HVP_P_TOTAL,
            LARGE_SCALE_HVP_PRIMARY_R,
            BENCH_MULTI_RHS_PROBE,
            scratch as f64 / (1024.0 * 1024.0),
            per_rhs_full_row_cache as f64 / (1024.0 * 1024.0),
        );
    }
    let mut group = c.benchmark_group("margslope_flex_large_scale_hv_pattern");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    group.bench_function(format!("n{n}_inner{inner_cycles}"), |b| {
        b.iter(|| {
            let problem = build_large_scale_shape_problem(black_box(n));
            let (fit, timing) = fit_problem(problem, cycle_capped_options(inner_cycles))
                .expect("criterion large-scale margslope fit");
            eprintln!(
                "[MS-FLEX-LARGE_SCALE-BENCH-ITER] n={} inner_max_cycles={} elapsed_s={:.3} outer_iters={} inner_cycles={} converged={} beta_len={}",
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

criterion_group!(benches, bench_margslope_flex_large_scale_cycle0);
criterion_main!(benches);

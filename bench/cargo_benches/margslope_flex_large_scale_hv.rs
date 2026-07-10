//! Criterion workload for the large-scale bernoulli marginal-slope
//! FLEX joint-Newton/Hv path. Multi-minute fit — run explicitly with:
//!
//! ```text
//! cargo bench --bench margslope_flex_large_scale_hv
//! ```

#[path = "../../tests/test_support/misc/margslope_flex_equivalence.rs"]
mod margslope_flex_equivalence;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use margslope_flex_equivalence::{
    build_large_scale_shape_problem, cycle_capped_options, fit_problem,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

const DEFAULT_REPRO_N: usize = 50_000;
const BENCH_INNER_CYCLES: usize = 1;
#[cfg(target_os = "linux")]
const BENCH_MULTI_RHS_PROBE: usize = 4;
#[cfg(target_os = "linux")]
const LARGE_SCALE_HVP_PRIMARY_R: usize = 20;
#[cfg(target_os = "linux")]
const LARGE_SCALE_HVP_P_TOTAL: usize = 44;

struct CountingAllocator;

static TRACK_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOCATION_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() && TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() && TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        new_ptr
    }
}

#[derive(Clone, Copy)]
struct AllocationSnapshot {
    allocation_calls: u64,
    allocated_bytes: u64,
}

fn begin_allocation_measurement() {
    TRACK_ALLOCATIONS.store(false, Ordering::SeqCst);
    ALLOCATION_CALLS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    TRACK_ALLOCATIONS.store(true, Ordering::SeqCst);
}

fn end_allocation_measurement() -> AllocationSnapshot {
    TRACK_ALLOCATIONS.store(false, Ordering::SeqCst);
    AllocationSnapshot {
        allocation_calls: ALLOCATION_CALLS.load(Ordering::Relaxed),
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
    }
}

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
    let problem = build_large_scale_shape_problem(n);
    let mut group = c.benchmark_group("margslope_flex_large_scale_hv_pattern");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    group.bench_function(format!("n{n}_inner{inner_cycles}"), |b| {
        b.iter_batched(
            || problem.clone(),
            |problem| {
                begin_allocation_measurement();
                let result = fit_problem(
                    black_box(problem),
                    cycle_capped_options(black_box(inner_cycles)),
                );
                let allocations = end_allocation_measurement();
                let (fit, timing) = result.expect("criterion large-scale margslope fit");
                eprintln!(
                    "[MS-FLEX-LARGE_SCALE-BENCH-ITER] n={} inner_max_cycles={} elapsed_s={:.3} allocation_calls={} allocated_bytes={} outer_iters={} inner_cycles={} converged={} beta_len={}",
                    n,
                    inner_cycles,
                    timing.elapsed.as_secs_f64(),
                    allocations.allocation_calls,
                    allocations.allocated_bytes,
                    timing.outer_iterations,
                    timing.inner_cycles,
                    timing.outer_converged,
                    fit.fit.beta.len()
                );
                black_box(fit.fit.beta.len())
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_margslope_flex_large_scale_cycle0);
criterion_main!(benches);

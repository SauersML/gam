//! A GAM fit's heap peak is bounded by its dense design (pyGAM audit speed
//! F11): the stores that grow with `n` or with the number of outer evaluations
//! are budgeted against the design's resident bytes.
//!
//! Peaks are the counting allocator's live bytes above the pre-fit baseline,
//! on one rayon worker so the peak belongs to the fit's allocation sequence
//! and not to a thread interleaving. A warm-up fit of each family runs first,
//! so process-lifetime lazies (the GEMM thread-local workspace, the rayon
//! registry) sit in the baseline instead of in the first measured fit.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_linalg::utils::splitmix64;
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};

struct PeakCountingAllocator;

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

fn note_allocated(bytes: usize) {
    let live = LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
}

fn note_released(bytes: usize) {
    LIVE_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

// SAFETY: `PeakCountingAllocator` delegates every allocation operation to
// `System` with the original pointer/layout contract unchanged. Its only side
// effect is updating atomic byte counters, which allocates nothing and cannot
// affect ownership.
unsafe impl GlobalAlloc for PeakCountingAllocator {
    // SAFETY: callers supply the `GlobalAlloc`-required valid layout;
    // forwarding it unchanged to `System` preserves that contract.
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is valid by this method's `GlobalAlloc` contract.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            note_allocated(layout.size());
        }
        ptr
    }

    // SAFETY: callers supply the `GlobalAlloc`-required valid layout;
    // forwarding it unchanged to `System` preserves that contract.
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is valid by this method's `GlobalAlloc` contract.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            note_allocated(layout.size());
        }
        ptr
    }

    // SAFETY: `ptr` and `layout` must denote a live `System` allocation by
    // this allocator's contract, and both are forwarded unchanged.
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller guarantees the matching live allocation contract.
        unsafe { System.dealloc(ptr, layout) };
        note_released(layout.size());
    }

    // SAFETY: `ptr` and `layout` must denote a live `System` allocation and
    // `new_size` is forwarded unchanged, exactly as required by `GlobalAlloc`.
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller guarantees the matching live allocation contract.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            note_released(layout.size());
            note_allocated(new_size);
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: PeakCountingAllocator = PeakCountingAllocator;

fn peak_added_bytes<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let baseline = LIVE_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES.store(baseline, Ordering::Relaxed);
    let output = work();
    let peak = PEAK_BYTES.load(Ordering::Relaxed);
    (output, peak.saturating_sub(baseline))
}

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn next_poisson(state: &mut u64, mean: f64) -> u64 {
    let threshold = (-mean).exp();
    let mut count = 0;
    let mut product = next_unit(state);
    while product > threshold {
        count += 1;
        product *= next_unit(state);
    }
    count
}

/// A one-covariate fixture. Each family draws its own covariate column: the
/// process-wide Duchon basis memo keys on the covariate values, so a shared
/// column would let the second family's fit reuse bases the first one built
/// and measure a warm fit against a cold one.
fn dataset(family: &str, n: usize) -> EncodedDataset {
    let headers = ["x", "y"].into_iter().map(str::to_string).collect();
    let family_seed = match family {
        "poisson" => 0xF11_0000_0000_0002,
        _ => 0xF11_0000_0000_0001,
    };
    let mut state: u64 = family_seed ^ n as u64;
    let rows = (0..n)
        .map(|_| {
            let x = next_unit(&mut state);
            let signal = (std::f64::consts::TAU * x).sin();
            let y = match family {
                "poisson" => next_poisson(&mut state, (1.0 + signal).exp()) as f64,
                _ => signal + 0.5 * next_gauss(&mut state),
            };
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode the memory-bound fixture")
}

/// Fits `y ~ s(x)` and returns the fitted dense design's bytes.
fn fit(family: &str, data: &EncodedDataset) -> usize {
    let config = FitConfig {
        family: Some(family.to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", data, &config)
        .unwrap_or_else(|error| panic!("{family} fit failed: {error:?}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard {family} fit");
    };
    fit.design.design.nrows() * fit.design.design.ncols() * std::mem::size_of::<f64>()
}

struct Footprint {
    design_bytes: usize,
    peak_bytes: usize,
}

impl Footprint {
    fn designs(&self) -> f64 {
        self.peak_bytes as f64 / self.design_bytes as f64
    }
}

fn footprint(family: &str, n: usize) -> Footprint {
    let data = dataset(family, n);
    let (design_bytes, peak_bytes) = peak_added_bytes(|| fit(family, &data));
    Footprint {
        design_bytes,
        peak_bytes,
    }
}

/// The largest fixture: every post-fit report still reads all of its rows, so
/// each store in the fit scales with `n` here, and one P-IRLS fit of it stays
/// within a CI test's time. The n = 1e6 fits are the `n1e6_memory` bench plan.
const LARGE_N: usize = 50_000;

/// A tenth of [`LARGE_N`]: large enough to fit the same smooth, small enough
/// that the fit's fixed costs are a visible part of its peak.
const SMALL_N: usize = LARGE_N / 10;

/// The warm-up fit's size: the smallest fixture on which each family's fit
/// runs the same code paths as the measured ones.
const WARM_UP_N: usize = 2_000;

#[test]
fn fit_heap_peak_is_bounded_by_the_dense_design() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("build a one-worker pool");
    pool.install(|| {
        for family in ["gaussian", "poisson"] {
            fit(family, &dataset(family, WARM_UP_N));
        }
        let small_gaussian = footprint("gaussian", SMALL_N);
        let large_gaussian = footprint("gaussian", LARGE_N);
        let small_poisson = footprint("poisson", SMALL_N);
        let large_poisson = footprint("poisson", LARGE_N);
        for (family, small, large) in [
            ("gaussian", &small_gaussian, &large_gaussian),
            ("poisson", &small_poisson, &large_poisson),
        ] {
            eprintln!(
                "{family}: {:.2} designs at n = {SMALL_N}, {:.2} at n = {LARGE_N}",
                small.designs(),
                large.designs(),
            );
            // A peak that is the design times a constant plus fixed costs
            // cannot hold more designs at a larger n: a store that did would
            // grow faster than the design.
            assert!(
                large.designs() <= small.designs(),
                "{family} fit peak grew faster than its design: {:.2} designs at n = {SMALL_N}, \
                 {:.2} at n = {LARGE_N}",
                small.designs(),
                large.designs(),
            );
        }

        // The two families fit the same smooth to the same number of rows, so
        // their designs and post-fit reports have one shape, and the Poisson
        // fit's excess over the Gaussian one is its iterated P-IRLS state.
        // That may hold two design-sized stores: the P-IRLS result cache,
        // whose byte budget is the dense design (`pirls_cache_byte_budget`),
        // and the square-root solve's dense design, materialized once per
        // fit. Everything else it iterates on is a few n-vectors, each a
        // design's width smaller than the design.
        assert_eq!(
            large_poisson.design_bytes, large_gaussian.design_bytes,
            "the families must fit designs of one shape"
        );
        let iterated_state_stores = 2;
        let excess_bytes = large_poisson.peak_bytes.saturating_sub(large_gaussian.peak_bytes);
        eprintln!(
            "poisson over gaussian at n = {LARGE_N}: {:.2} designs",
            excess_bytes as f64 / large_poisson.design_bytes as f64
        );
        assert!(
            excess_bytes <= iterated_state_stores * large_poisson.design_bytes,
            "the Poisson fit held {excess_bytes} bytes beyond the Gaussian fit of a design \
             of the same {} bytes; its iterated P-IRLS state is bounded by \
             {iterated_state_stores} designs",
            large_poisson.design_bytes,
        );
    });
}

//! #932 DIRECT production-path measurement for the BMS empirical-flex row
//! derivative seam — the evidence gate that replaces the whole-fit Criterion
//! benchmark (which primes the FIFO-2 cell-moment cache, selects its latent
//! branch via `Auto`, and cannot attribute per-row allocations).
//!
//! This module measures the EXACT production call
//! [`compute_row_analytic_flex_from_parts_into`] on a forced 65-node
//! `LatentMeasureKind::GlobalEmpirical` grid (the branch d1a7a0bc6 optimized),
//! with the intercept root, row context, and row scratch precomputed OUTSIDE
//! the measured region:
//!
//!   * WARMED row op — the steady-state Newton/Hv inner call. d1a7a0bc6
//!     removed the four per-row coefficient-buffer `Vec`s
//!     (`zero_family`/`coeff_u`/`coeff_au`/`coeff_bu`, `128·r` bytes/row); the
//!     allocation gate below pins the surviving per-row heap traffic so a
//!     regression re-introducing per-row allocation fails deterministically.
//!   * COLD row op — fresh scratch + intercept solve + first call, reported
//!     for the builder-lane comparison (diagnostic only).
//!
//! Timing is reported via `eprintln!` as ns/row for the MSI A/B ledger
//! (parent-of-d1a7a0bc6 → d1a7a0bc6 → HEAD on one pinned CPU); per SPEC there
//! are NO wall-clock assertions — the asserted gate is allocation counts,
//! which are deterministic for fixed work.
//!
//! Allocation counting is THREAD-SCOPED (`thread_local` `Cell`s) so the
//! parallel nextest/libtest harness cannot contaminate the counters, and the
//! measured seam is a single-threaded scalar row kernel. The counting
//! `#[global_allocator]` costs one const-initialized `thread_local` read per
//! allocation for the rest of this test binary and is disabled outside the
//! measured regions.
//!
//! Ban-scanner-safe: a bare `#[cfg(test)] mod flex_measure_932_tests;` in
//! `bms/mod.rs` (the `*_tests` allowed name), reaching the production path as
//! a private child of the common ancestor `bms`.

use super::family::*;
use super::hessian_paths::*;
use super::*;
use gam_linalg::matrix::DesignMatrix;
use gam_problem::{InverseLink, StandardLink};
use ndarray::Array1;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::{Arc, Mutex};

// ------------------------------------------------------------------
// Thread-scoped counting allocator.
// ------------------------------------------------------------------

struct ThreadCountingAllocator;

thread_local! {
    static TRACK_THIS_THREAD: Cell<bool> = const { Cell::new(false) };
    static THREAD_ALLOCATION_CALLS: Cell<u64> = const { Cell::new(0) };
    static THREAD_ALLOCATED_BYTES: Cell<u64> = const { Cell::new(0) };
}

/// Record one successful allocation of `size` bytes if this thread is inside
/// a measured region. `try_with` (not `with`) so allocations during TLS
/// teardown at thread exit can never panic inside the allocator.
fn note_allocation(size: usize) {
    // A failed `try_with` means this thread's TLS is already torn down: there
    // is no measured region left to credit, so "not tracking" is the correct
    // (and panic-free) reading inside the allocator.
    if !TRACK_THIS_THREAD.try_with(Cell::get).unwrap_or(false) {
        return;
    }
    THREAD_ALLOCATION_CALLS
        .try_with(|c| c.set(c.get() + 1))
        .unwrap_or(());
    THREAD_ALLOCATED_BYTES
        .try_with(|b| b.set(b.get() + size as u64))
        .unwrap_or(());
}

// SAFETY: `ThreadCountingAllocator` delegates every allocation operation to
// `System` with the original pointer/layout contract unchanged. Its only side
// effect is updating const-initialized `thread_local` `Cell` counters after
// successful allocations, which allocates nothing and cannot affect ownership.
unsafe impl GlobalAlloc for ThreadCountingAllocator {
    // SAFETY: callers supply the `GlobalAlloc`-required valid layout;
    // forwarding it unchanged to `System` preserves that contract.
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is valid by this method's `GlobalAlloc` contract.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            note_allocation(layout.size());
        }
        ptr
    }

    // SAFETY: callers supply the `GlobalAlloc`-required valid layout;
    // forwarding it unchanged to `System` preserves that contract.
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is valid by this method's `GlobalAlloc` contract.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            note_allocation(layout.size());
        }
        ptr
    }

    // SAFETY: `ptr` and `layout` must denote a live `System` allocation by
    // this allocator's contract, and both are forwarded unchanged.
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller guarantees the matching live allocation contract.
        unsafe { System.dealloc(ptr, layout) }
    }

    // SAFETY: `ptr` and `layout` must denote a live `System` allocation and
    // `new_size` is forwarded unchanged, exactly as required by `GlobalAlloc`.
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller guarantees the matching live allocation contract.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            note_allocation(new_size);
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: ThreadCountingAllocator = ThreadCountingAllocator;

fn begin_thread_allocation_measurement() {
    TRACK_THIS_THREAD.with(|t| t.set(false));
    THREAD_ALLOCATION_CALLS.with(|c| c.set(0));
    THREAD_ALLOCATED_BYTES.with(|b| b.set(0));
    TRACK_THIS_THREAD.with(|t| t.set(true));
}

fn end_thread_allocation_measurement() -> (u64, u64) {
    TRACK_THIS_THREAD.with(|t| t.set(false));
    (
        THREAD_ALLOCATION_CALLS.with(|c| c.get()),
        THREAD_ALLOCATED_BYTES.with(|b| b.get()),
    )
}

// ------------------------------------------------------------------
// Forced-empirical fixture: 65-node global grid, both deviation blocks.
// ------------------------------------------------------------------

const GRID_NODES: usize = 65;

struct MFixture {
    family: BernoulliMarginalSlopeFamily,
    primary: PrimarySlices,
}

/// Deterministic 65-node grid over [−2.6, 2.6] with a normalized bell-shaped
/// weight profile — large enough that per-node work dominates per-cell work,
/// matching the production shape the factored path was built for.
fn mgrid() -> EmpiricalZGrid {
    let nodes: Vec<f64> = (0..GRID_NODES)
        .map(|i| -2.6 + 5.2 * (i as f64) / ((GRID_NODES - 1) as f64))
        .collect();
    let raw: Vec<f64> = nodes.iter().map(|z| (-0.5 * z * z).exp()).collect();
    let total: f64 = raw.iter().sum();
    let weights: Vec<f64> = raw.iter().map(|w| w / total).collect();
    EmpiricalZGrid::new(nodes, weights, "flex_measure_932 grid").expect("valid 65-node grid")
}

fn mruntime() -> DeviationRuntime {
    let n_knots = 11usize;
    let knots = Array1::from_iter(
        (0..n_knots).map(|i| -2.45_f64 + 5.0_f64 * (i as f64) / ((n_knots - 1) as f64)),
    );
    DeviationRuntime::try_new(knots, 0.0, 3).expect("deviation runtime")
}

fn mfixture(is_score_warp: bool) -> MFixture {
    let grid = mgrid();
    let runtime = mruntime();
    let basis_dim = runtime.basis_dim();
    let policy = gam_runtime::resource::ResourcePolicy::default_library();
    let dummy = || {
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            ndarray::Array2::zeros((1, 1)),
        ))
    };
    let family = BernoulliMarginalSlopeFamily {
        y: Arc::new(Array1::from_vec(vec![1.0])),
        weights: Arc::new(Array1::from_vec(vec![1.0])),
        z: Arc::new(Array1::from_vec(vec![0.45])),
        latent_measure: LatentMeasureKind::GlobalEmpirical { grid },
        gaussian_frailty_sd: None,
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginal_design: dummy(),
        logslope_design: dummy(),
        score_warp: if is_score_warp {
            Some(runtime.clone())
        } else {
            None
        },
        link_dev: if is_score_warp { None } else { Some(runtime) },
        policy: policy.clone(),
        cell_moment_lru: new_cell_moment_lru_cache(&policy),
        cell_moment_cache_stats: new_cell_moment_cache_stats(),
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let primary = PrimarySlices {
        q: 0,
        logslope: 1,
        h: if is_score_warp {
            Some(2..2 + basis_dim)
        } else {
            None
        },
        w: if is_score_warp {
            None
        } else {
            Some(2..2 + basis_dim)
        },
        total: 2 + basis_dim,
    };
    MFixture { family, primary }
}

struct MPoint {
    q: f64,
    b: f64,
    beta: Array1<f64>,
}

fn mpoint(fx: &MFixture) -> MPoint {
    let basis_dim = fx.primary.total - 2;
    let beta = Array1::from_shape_fn(basis_dim, |i| {
        let center = 0.5 * (basis_dim.saturating_sub(1) as f64);
        let radius = center.max(1.0);
        0.06 * ((i as f64) - center) / radius
    });
    MPoint {
        q: 0.2,
        b: 0.35,
        beta,
    }
}

/// Assert the fixture actually selects the empirical-grid branch — the whole
/// point of forcing `GlobalEmpirical` (the retired benchmark's `Auto` left
/// this unasserted and data-dependent).
fn assert_empirical_branch(fx: &MFixture) {
    let grid = fx
        .family
        .latent_measure
        .empirical_grid_for_training_row(0)
        .expect("latent measure query")
        .expect("GlobalEmpirical fixture must select the empirical-grid branch");
    assert_eq!(
        grid.pairs().count(),
        GRID_NODES,
        "fixture grid must be the forced 65-node empirical grid"
    );
}

fn measure_branch(is_score_warp: bool) {
    let label = if is_score_warp {
        "score-warp"
    } else {
        "link-dev"
    };
    let fx = mfixture(is_score_warp);
    assert_empirical_branch(&fx);
    let r = fx.primary.total;
    let pt = mpoint(&fx);
    let (beta_h, beta_w) = if is_score_warp {
        (Some(&pt.beta), None)
    } else {
        (None, Some(&pt.beta))
    };

    // ---- precompute OUTSIDE the measured region ----------------------
    let (intercept, m_a, _) = fx
        .family
        .solve_row_intercept_base(0, pt.q, pt.b, beta_h, beta_w, None)
        .expect("intercept solve");
    let row_ctx = BernoulliMarginalSlopeRowExactContext {
        intercept,
        m_a,
        intercept_fast_path: false,
        degree9_cells: None,
    };
    let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(r);

    let call = |scratch: &mut BernoulliMarginalSlopeFlexRowScratch| -> f64 {
        fx.family
            .compute_row_analytic_flex_from_parts_into(
                0,
                &fx.primary,
                pt.q,
                pt.b,
                beta_h,
                beta_w,
                &row_ctx,
                None,
                None,
                true,
                scratch,
            )
            .expect("production empirical-flex row call")
    };

    // Warm the scratch (first call resizes the coefficient buffers to r) and
    // pin the value so later measured calls are provably the same work.
    let warm_value = call(&mut scratch);
    let warm_value2 = call(&mut scratch);
    assert_eq!(
        warm_value.to_bits(),
        warm_value2.to_bits(),
        "{label}: warmed row op must be deterministic"
    );

    // ---- WARMED allocation gate --------------------------------------
    const WARMED_ROWS: u64 = 512;
    begin_thread_allocation_measurement();
    let mut acc = 0.0f64;
    for _ in 0..WARMED_ROWS {
        acc += call(&mut scratch);
    }
    let (warm_calls, warm_bytes) = end_thread_allocation_measurement();
    assert!(acc.is_finite());
    let calls_per_row = (warm_calls as f64) / (WARMED_ROWS as f64);
    let bytes_per_row = (warm_bytes as f64) / (WARMED_ROWS as f64);

    // ---- WARMED timing (diagnostic only; NO wall-clock assertion) ----
    let timer = std::time::Instant::now();
    let mut tacc = 0.0f64;
    for _ in 0..WARMED_ROWS {
        tacc += call(&mut scratch);
    }
    let warm_ns_per_row = (timer.elapsed().as_nanos() as f64) / (WARMED_ROWS as f64);
    assert_eq!(
        acc.to_bits(),
        tacc.to_bits(),
        "{label}: timing loop must repeat identical work"
    );

    // ---- COLD row op (fresh scratch + intercept + first call) --------
    begin_thread_allocation_measurement();
    let cold_timer = std::time::Instant::now();
    let (cold_intercept, cold_m_a, _) = fx
        .family
        .solve_row_intercept_base(0, pt.q, pt.b, beta_h, beta_w, None)
        .expect("cold intercept solve");
    let cold_ctx = BernoulliMarginalSlopeRowExactContext {
        intercept: cold_intercept,
        m_a: cold_m_a,
        intercept_fast_path: false,
        degree9_cells: None,
    };
    let mut cold_scratch = BernoulliMarginalSlopeFlexRowScratch::new(r);
    let cold_value = fx
        .family
        .compute_row_analytic_flex_from_parts_into(
            0,
            &fx.primary,
            pt.q,
            pt.b,
            beta_h,
            beta_w,
            &cold_ctx,
            None,
            None,
            true,
            &mut cold_scratch,
        )
        .expect("cold production row call");
    let cold_ns = cold_timer.elapsed().as_nanos() as f64;
    let (cold_calls, cold_bytes) = end_thread_allocation_measurement();
    assert_eq!(
        cold_value.to_bits(),
        warm_value.to_bits(),
        "{label}: cold and warmed row ops must agree exactly"
    );

    eprintln!(
        "#932 measure {label}: r={r} grid={GRID_NODES} \
         warmed[allocs/row={calls_per_row:.3} bytes/row={bytes_per_row:.1} ns/row={warm_ns_per_row:.0}] \
         cold[allocs={cold_calls} bytes={cold_bytes} ns={cold_ns:.0}]"
    );

    // ---- The asserted gate --------------------------------------------
    // d1a7a0bc6 deleted the four per-row coefficient `Vec`s (4 calls,
    // 128·r bytes per row). The surviving warmed per-row heap traffic is the
    // `denested_partition_cells` Vec (+ its cell probes). Pin a bound BELOW
    // the pre-d1a7a0bc6 level so this file, dropped verbatim into the parent
    // revision on the same machine, FAILS here and prints the parent's
    // measured numbers above — that pair of runs is the A/B evidence.
    assert!(
        calls_per_row <= 3.0,
        "{label}: warmed empirical-flex row op regressed to {calls_per_row:.3} \
         allocation calls/row ({bytes_per_row:.1} bytes/row) — the per-row \
         coefficient buffers deleted by d1a7a0bc6 (4 calls, 128·r bytes/row) \
         or new per-row heap traffic came back"
    );
}

#[test]
fn empirical_flex_warmed_row_allocation_gate_score_warp_932() {
    measure_branch(true);
}

#[test]
fn empirical_flex_warmed_row_allocation_gate_link_dev_932() {
    measure_branch(false);
}

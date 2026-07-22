//! #932 DIRECT production-path measurement for the BMS empirical-flex row
//! derivative seam — the evidence gate that replaces the whole-fit Criterion
//! benchmark (which primes the FIFO-2 cell-moment cache, selects its latent
//! branch via `Auto`, and cannot attribute per-row allocations).
//!
//! This module measures the EXACT production call
//! [`lower_bms_flex_row_order2_from_parts`] on a forced 65-node
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
/// matching the production shape the compiled moment-jet path was built for.
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
    let grid = fx
        .family
        .latent_measure
        .empirical_grid_for_training_row(0)
        .expect("measured grid query")
        .expect("forced empirical grid");
    let cells = fx
        .family
        .denested_partition_cells(intercept, pt.b, beta_h, beta_w)
        .expect("measured denested cells");
    let occupied_cells = cells
        .iter()
        .filter(|partition_cell| {
            grid.nodes
                .iter()
                .any(|&node| node >= partition_cell.cell.left && node < partition_cell.cell.right)
        })
        .count();
    let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(r);

    let call = |scratch: &mut BernoulliMarginalSlopeFlexRowScratch| -> f64 {
        fx.family
            .lower_bms_flex_row_order2_from_parts(
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
        .lower_bms_flex_row_order2_from_parts(
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
        "#932 measure {label}: r={r} grid={GRID_NODES} cells={} occupied={} \
         warmed[allocs/row={calls_per_row:.3} bytes/row={bytes_per_row:.1} ns/row={warm_ns_per_row:.0}] \
         cold[allocs={cold_calls} bytes={cold_bytes} ns={cold_ns:.0}]",
        cells.len(),
        occupied_cells,
    );

    // ---- The asserted gate --------------------------------------------
    // d1a7a0bc6 deleted the four per-row coefficient `Vec`s (4 calls,
    // 128·r bytes per row). MEASURED floors on this fixture (Lambda A10,
    // 2026-07-10, dev and release identical — counts are code-path
    // deterministic): post-fix warmed = 5 allocs/row score-warp, 7 link-dev;
    // parent-of-d1a7a0bc6 = 9 and 11 (+4 calls, +128·r bytes/row — the exact
    // predicted delta). The surviving traffic is the denested cell partition
    // + per-cell probe allocations. Pin the gate AT the measured post-fix
    // floor, strictly below parent, so this file dropped verbatim into the
    // parent revision FAILS here and prints the parent's numbers above —
    // that pair of runs is the A/B evidence.
    let floor = if is_score_warp { 5.0 } else { 7.0 };
    assert!(
        calls_per_row <= floor,
        "{label}: warmed empirical-flex row op regressed to {calls_per_row:.3} \
         allocation calls/row ({bytes_per_row:.1} bytes/row) above the measured \
         post-d1a7a0bc6 floor of {floor} — the per-row coefficient buffers \
         deleted by d1a7a0bc6 (4 calls, 128·r bytes/row) or new per-row heap \
         traffic came back"
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

// ------------------------------------------------------------------
// #932 release cell: fixed-K specialization vs runtime-width dynamic
// evaluation of the SAME row plan (the specialization-earns-its-keep gate).
// ------------------------------------------------------------------
//
// The BMS empirical-flex third/fourth contractions have NO hand-written
// derivative tower — unlike the *rigid* Bernoulli row, whose closed forms the
// `release_measure_rigid_bernoulli_*` cells gate against, the FLEX deviation
// blocks were jet-only from the start. So the baseline this cell measures
// against is not a hand tower: it is the *generic runtime-width dynamic jet*
// (`DynamicOneSeedBatch` / `DynamicTwoSeedBatch`, arena-allocated, width chosen
// at runtime) that production falls back to for widths without a fixed
// specialization. Production dispatches the const-generic fixed-width jets
// (`FixedRuntimeJet<OneSeed<K>,K>` third, `FixedRuntimeJet<TwoSeed<K>,K>`
// fourth) exactly when the primary width `r = 2 + basis_dim` is one of the
// specialized tiers {4, 8, 12, 18} (`empirical_bms_third_jet_schedule` /
// `empirical_bms_fourth_jet_schedule`), and falls back to the dynamic batch
// otherwise. This cell proves the specialization earns its keep: it compiles
// ONE canonical [`BmsFlexRowProgram`], then evaluates the third and fourth
// contractions of that single frozen plan BOTH ways at the SAME tier width and
// reports the dynamic/fixed time ratio.
//
// Because the width that selects a fixed tier is `2 + basis_dim` and the
// deviation runtime's `basis_dim` grows with its knot count, the natural
// `mruntime` fixture width is NOT a fixed tier; `tier_runtime` discovers the
// knot count whose runtime lands the primary width on a fixed tier so the vars
// the dispatch actually specializes are the ones measured here.
//
// SPEC (this module's header): NO wall-clock assertions. The asserted gate is
// (a) fixed-vs-dynamic parity on the exact benchmarked inputs and (b) checksum
// finiteness; the speed evidence is the emitted `hand_over_production` token
// (`dynamic_ns / fixed_ns`, > 1 when the specialization wins), which the MSI
// release harness parses and fails closed on `<= 1`.
//
// The fixed fourth (`empirical_fixed_fourth_many_from_plan`) and dynamic fourth
// (`empirical_dynamic_fourth_batch_from_plan`) production kernels are
// `pub(super)` and take an already-compiled `&plan`, so they are invoked
// directly on the one canonical plan. The third-order fixed/dynamic
// from-plan helpers are private (`empirical_fixed_third_many_dispatch`) or
// inline in `empirical_flex_row_third_contracted_many`, so the two third seed
// wrappers below reproduce ONLY that seeding (identical `FixedRuntimeJet` /
// `DynamicOneSeedBatch` types and the same production `BmsFlexRowProgram::
// evaluate`) — the measured arithmetic is production either way.

use super::flex_row_program::BmsFlexRowProgram;

/// Uniform deviation knots over the `mruntime` span for a chosen knot count.
fn tier_knots(n_knots: usize) -> Array1<f64> {
    Array1::from_iter(
        (0..n_knots).map(|i| -2.45_f64 + 5.0_f64 * (i as f64) / ((n_knots - 1) as f64)),
    )
}

/// Discover a deviation runtime whose primary width `2 + basis_dim` lands on a
/// fixed-K specialization tier, and return it with that width. `basis_dim`
/// increases by one per added knot, so a tier is always reachable; width 12 is
/// preferred (a substantial-but-fast middle tier), otherwise the first of
/// {8, 18} encountered is used.
fn tier_runtime() -> (DeviationRuntime, usize) {
    let mut fallback: Option<(DeviationRuntime, usize)> = None;
    for n_knots in 6..=64usize {
        let Ok(runtime) = DeviationRuntime::try_new(tier_knots(n_knots), 0.0, 3) else {
            continue;
        };
        let width = 2 + runtime.basis_dim();
        if width == 12 {
            return (runtime, width);
        }
        if matches!(width, 8 | 18) && fallback.is_none() {
            fallback = Some((runtime, width));
        }
    }
    fallback.expect("some knot count must land the BMS primary width on a fixed-K tier")
}

/// Build the forced-`GlobalEmpirical` single-row fixture at a caller-supplied
/// deviation runtime (mirrors `mfixture`, which is pinned to `mruntime`, but
/// lets this cell target a fixed-K tier width).
fn build_fixture_with_runtime(is_score_warp: bool, runtime: DeviationRuntime) -> MFixture {
    let grid = mgrid();
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

/// Fixed-K third contraction of the frozen plan along `dir` — the seeding of
/// the private `empirical_fixed_third_contracted_arrays::<K>` over the same
/// `FixedRuntimeJet<OneSeed<K>,K>` algebra and the same `plan.evaluate`.
fn fixed_third_contracted<const K: usize>(
    plan: &BmsFlexRowProgram,
    point: &[f64; K],
    dir: &Array1<f64>,
) -> ndarray::Array2<f64> {
    use gam_math::jet_scalar::{FixedRuntimeJet, OneSeed};
    let vars: [FixedRuntimeJet<OneSeed<K>, K>; K] = std::array::from_fn(|axis| {
        FixedRuntimeJet::from_inner(OneSeed::seed_direction(point[axis], axis, dir[axis]))
    });
    let contracted = plan
        .evaluate(&vars, 3, &())
        .expect("fixed-K third contraction of canonical plan")
        .into_inner()
        .contracted_third();
    ndarray::Array2::from_shape_fn((K, K), |(a, b)| contracted[a][b])
}

/// Runtime-width dynamic third contraction of the frozen plan along `dir` —
/// the single-lane form of the dynamic branch in
/// `empirical_flex_row_third_contracted_many`, over the same
/// `DynamicOneSeedBatch` algebra and the same `plan.evaluate`.
fn dynamic_third_contracted(
    plan: &BmsFlexRowProgram,
    point: &[f64],
    dir: &Array1<f64>,
    r: usize,
) -> ndarray::Array2<f64> {
    use gam_math::jet_scalar::{DynamicJetBatchWorkspace, DynamicOneSeedBatch};
    let mut workspace = DynamicJetBatchWorkspace::new(1);
    workspace.reset(1);
    let vars = workspace.alloc_slice_fill_with(r, |axis| {
        DynamicOneSeedBatch::seed_directions(point[axis], axis, r, &workspace, |_| dir[axis])
    });
    let jet = plan
        .evaluate(vars, 3, &workspace)
        .expect("dynamic third contraction of canonical plan");
    ndarray::Array2::from_shape_vec((r, r), jet.contracted_third(0).to_vec())
        .expect("dynamic third contraction shape")
}

/// Feedback-coupled best-of-5 timing barrier (no `std::hint::black_box`): the
/// running checksum feeds the next perturbation so the optimizer cannot hoist
/// the evaluation. Returns ns per evaluation.
fn best_measure_ns<F: FnMut(f64) -> f64>(iterations: usize, base: f64, mut evaluate: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let mut checksum = 0.0_f64;
        let started = std::time::Instant::now();
        for _ in 0..iterations {
            checksum += evaluate(base + checksum * 1e-18);
        }
        assert!(
            checksum.is_finite(),
            "BMS-FLEX-CONTRACTED-932 release-measure checksum must stay finite"
        );
        best = best.min(started.elapsed().as_secs_f64());
    }
    best * 1e9 / iterations as f64
}

/// Measure one deviation branch at compile-time tier width `K`.
fn measure_third_fourth_branch<const K: usize>(is_score_warp: bool, runtime: DeviationRuntime) {
    use super::cell_moment_assembly::{
        EmpiricalBmsFourthJetSchedule, EmpiricalBmsThirdJetSchedule,
        empirical_bms_fourth_jet_schedule, empirical_bms_third_jet_schedule,
    };

    let label = if is_score_warp {
        "score-warp"
    } else {
        "link-dev"
    };
    let fx = build_fixture_with_runtime(is_score_warp, runtime);
    assert_empirical_branch(&fx);
    let r = fx.primary.total;
    assert_eq!(
        r, K,
        "{label}: tier fixture width {r} != specialization width K={K}"
    );

    // Production must actually dispatch the fixed specialization at this width;
    // otherwise the "specialization vs dynamic fallback" comparison is moot.
    assert_eq!(
        empirical_bms_third_jet_schedule(r),
        EmpiricalBmsThirdJetSchedule::FixedWidthFromPlan,
        "{label}: production must pick the fixed-K third jet at width {r}"
    );
    assert_eq!(
        empirical_bms_fourth_jet_schedule(r),
        EmpiricalBmsFourthJetSchedule::RepeatedFixedWidth,
        "{label}: production must pick the fixed-K fourth jet at width {r}"
    );

    let pt = mpoint(&fx);
    let (beta_h, beta_w) = if is_score_warp {
        (Some(&pt.beta), None)
    } else {
        (None, Some(&pt.beta))
    };

    // ---- one canonical row plan, compiled OUTSIDE every measured region ----
    let intercept = fx
        .family
        .solve_row_intercept_base(0, pt.q, pt.b, beta_h, beta_w, None)
        .expect("intercept solve")
        .0;
    let grid = fx
        .family
        .latent_measure
        .empirical_grid_for_training_row(0)
        .expect("latent measure query")
        .expect("forced empirical grid");
    let plan = fx
        .family
        .compile_empirical_bms_row_program(
            0,
            &fx.primary,
            pt.q,
            pt.b,
            beta_h,
            beta_w,
            intercept,
            &grid,
        )
        .expect("canonical empirical-flex row plan");
    let point_vec =
        BernoulliMarginalSlopeFamily::intercept_primary_point(pt.q, pt.b, beta_h, beta_w);
    let point: [f64; K] = point_vec
        .as_slice()
        .try_into()
        .expect("primary point width matches specialization K");

    // Two distinct, everywhere-nonzero contraction directions.
    let dir_u = Array1::from_shape_fn(r, |i| 0.5 + 0.3 * ((i % 3) as f64) - 0.2 * ((i % 2) as f64));
    let dir_v = Array1::from_shape_fn(r, |i| {
        -0.4 + 0.3 * (((i + 1) % 4) as f64) - 0.1 * ((i % 2) as f64)
    });
    let pairs: [(&Array1<f64>, &Array1<f64>); 1] = [(&dir_u, &dir_v)];

    // ---- parity pin on the exact benchmarked inputs -----------------------
    let fixed_third = fixed_third_contracted::<K>(&plan, &point, &dir_u);
    let dynamic_third = dynamic_third_contracted(&plan, &point, &dir_u, r);
    let fixed_fourth = BernoulliMarginalSlopeFamily::empirical_fixed_fourth_many_from_plan::<K>(
        &plan, &point, &pairs,
    )
    .expect("fixed-K fourth contraction of canonical plan");
    let dynamic_fourth = BernoulliMarginalSlopeFamily::empirical_dynamic_fourth_batch_from_plan(
        &plan,
        &point,
        &pairs,
        &fx.primary,
        1,
    )
    .expect("dynamic fourth contraction of canonical plan");
    for a in 0..r {
        for b in 0..r {
            let (f3, d3) = (fixed_third[(a, b)], dynamic_third[(a, b)]);
            let band3 = 1e-11 * f3.abs().max(d3.abs()).max(1.0);
            assert!(
                (f3 - d3).abs() <= band3,
                "{label}: third[{a}][{b}] fixed {f3:+.15e} vs dynamic {d3:+.15e}"
            );
            let (f4, d4) = (fixed_fourth[0][(a, b)], dynamic_fourth[0][(a, b)]);
            let band4 = 1e-11 * f4.abs().max(d4.abs()).max(1.0);
            assert!(
                (f4 - d4).abs() <= band4,
                "{label}: fourth[{a}][{b}] fixed {f4:+.15e} vs dynamic {d4:+.15e}"
            );
        }
    }

    // ---- timing (diagnostic; NO wall-clock assertion per SPEC) ------------
    // Iteration counts keep each best-of-5 sweep well under ~1s for the heaviest
    // tier while giving a stable ns/row read.
    let iters_third = 800usize;
    let iters_fourth = 500usize;

    let third_fixed_ns = best_measure_ns(iters_third, point[0], |p0| {
        let mut perturbed = point;
        perturbed[0] = p0;
        let m = fixed_third_contracted::<K>(&plan, &perturbed, &dir_u);
        m[(0, 0)] + m[(r - 1, r - 1)]
    });
    let third_dynamic_ns = best_measure_ns(iters_third, point[0], |p0| {
        let mut perturbed = point;
        perturbed[0] = p0;
        let m = dynamic_third_contracted(&plan, &perturbed, &dir_u, r);
        m[(0, 0)] + m[(r - 1, r - 1)]
    });
    eprintln!(
        "BMS-FLEX-CONTRACTED-932 branch={label} width={K} grid={GRID_NODES} order=3 \
         production_fixed={third_fixed_ns:.2} ns/row dynamic={third_dynamic_ns:.2} ns/row \
         hand_over_production={:.6}",
        third_dynamic_ns / third_fixed_ns,
    );

    let fourth_fixed_ns = best_measure_ns(iters_fourth, point[0], |p0| {
        let mut perturbed = point;
        perturbed[0] = p0;
        let out = BernoulliMarginalSlopeFamily::empirical_fixed_fourth_many_from_plan::<K>(
            &plan, &perturbed, &pairs,
        )
        .expect("fixed-K fourth contraction");
        out[0][(0, 0)] + out[0][(r - 1, r - 1)]
    });
    let fourth_dynamic_ns = best_measure_ns(iters_fourth, point[0], |p0| {
        let mut perturbed = point;
        perturbed[0] = p0;
        let out = BernoulliMarginalSlopeFamily::empirical_dynamic_fourth_batch_from_plan(
            &plan,
            &perturbed,
            &pairs,
            &fx.primary,
            1,
        )
        .expect("dynamic fourth contraction");
        out[0][(0, 0)] + out[0][(r - 1, r - 1)]
    });
    eprintln!(
        "BMS-FLEX-CONTRACTED-932 branch={label} width={K} grid={GRID_NODES} order=4 \
         production_fixed={fourth_fixed_ns:.2} ns/row dynamic={fourth_dynamic_ns:.2} ns/row \
         hand_over_production={:.6}",
        fourth_dynamic_ns / fourth_fixed_ns,
    );
}

#[test]
fn release_measure_bms_empirical_third_fourth_fixed_vs_dynamic_932() {
    let (runtime, width) = tier_runtime();
    // Dispatch the runtime-discovered tier width to its compile-time
    // specialization, then measure both deviation branches on it.
    match width {
        8 => {
            measure_third_fourth_branch::<8>(true, runtime.clone());
            measure_third_fourth_branch::<8>(false, runtime);
        }
        12 => {
            measure_third_fourth_branch::<12>(true, runtime.clone());
            measure_third_fourth_branch::<12>(false, runtime);
        }
        18 => {
            measure_third_fourth_branch::<18>(true, runtime.clone());
            measure_third_fourth_branch::<18>(false, runtime);
        }
        other => panic!("tier_runtime returned non-specialized width {other}"),
    }
}

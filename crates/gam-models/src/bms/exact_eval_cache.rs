use super::family::{BernoulliInterceptPredictorWarmStart, BernoulliMarginalSlopeFamily};
use super::hessian_paths::{
    BernoulliMarginalSlopeRowExactContext, BlockSlices, PrimarySlices, RowCellMomentsBundle,
    block_slices, primary_slices,
};
use super::*;

#[inline]
pub(super) fn log_exact_work(n: usize) -> bool {
    n >= EXACT_WORK_LOG_MIN_ROWS
}

/// Live cgroup-aware process-memory allowance, resampled on every call.
///
/// This is the OOM guard's reading and only the OOM guard's: it feeds the
/// global-pin check that refuses to commit a *new* co-resident cache, which is
/// the one decision that has to see the machine as it is right now. The
/// worthwhileness route takes the monotone floor
/// ([`observe_capacity_floor`]) so a shape cannot flip mid-fit.
pub(super) fn runtime_available_memory_bytes() -> u64 {
    gam_runtime::resource::resample_memory_availability().available_bytes()
}

/// Process-global counter of bytes currently pinned by live BMS row-primary
/// evaluation caches. Incremented by [`RowPrimaryEvalPin::new`] when a cache
/// is materialized and decremented on `Drop`, so two co-resident workspaces
/// cannot together pin more than `available_ram * GLOBAL_FRACTION`.
pub(super) fn bms_row_primary_hessian_pinned_bytes() -> &'static AtomicU64 {
    static PINNED: OnceLock<AtomicU64> = OnceLock::new();
    PINNED.get_or_init(|| AtomicU64::new(0))
}

/// Process-global high-water mark of available RAM ever observed at a cache
/// decision. The single-cache *worthwhileness* budget (is this shape large
/// enough relative to memory to be worth materializing) is derived from this
/// monotone floor rather than the instantaneous `available_memory()` so that
/// the same `(n, r)` shape cannot flip from `materialize` to `stream` partway
/// through a fit just because transient available RAM dipped — a flip that
/// sends the BMS flex inner solve off the fast dense route and onto the
/// catastrophically slower matrix-free CG path. Live `available_memory()` is
/// still consulted for the global-pin OOM guard, which is the actual safety
/// valve against over-committing co-resident caches.
pub(super) fn bms_row_primary_hessian_capacity_floor() -> &'static AtomicU64 {
    static FLOOR: OnceLock<AtomicU64> = OnceLock::new();
    FLOOR.get_or_init(|| AtomicU64::new(0))
}

/// Fold the latest observed available-RAM reading into the monotone capacity
/// floor and return the resulting stable budget basis (`max(floor, observed)`).
pub(super) fn observe_capacity_floor(runtime_available_bytes: u64) -> u64 {
    bms_row_primary_hessian_capacity_floor()
        .fetch_max(runtime_available_bytes, Ordering::AcqRel)
        .max(runtime_available_bytes)
}

/// The memory readings the row-primary cache decision takes:
/// `(runtime_available, stable_capacity, workspace_pinned)`.
///
/// A fit on its own reads live availability, the monotone capacity floor and
/// every co-resident cache's pins. One search of a parallel multistart reads its
/// lane instead: the availability read once before launch, for both budgets, and
/// its own pins. Its decision is then the one it makes running alone, whatever
/// the other searches have pinned (gnomon#2359).
pub(super) fn row_primary_cache_memory_readings(
    lane: Option<&gam_runtime::resource::SearchLaneBudget>,
) -> (u64, u64, u64) {
    match lane {
        Some(lane) => (
            lane.serial_available_bytes(),
            lane.serial_available_bytes(),
            lane.pinned_bytes().load(Ordering::Acquire),
        ),
        None => {
            let runtime_available = runtime_available_memory_bytes();
            (
                runtime_available,
                observe_capacity_floor(runtime_available),
                bms_row_primary_hessian_pinned_bytes().load(Ordering::Acquire),
            )
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RowPrimaryHessianCacheReason {
    ReuseTooLow,
    SingleCacheExceedsRamFraction,
    GlobalPinExceedsRamFraction,
    ReuseAmortizesBuild,
}

impl RowPrimaryHessianCacheReason {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::ReuseTooLow => "reuse_too_low",
            Self::SingleCacheExceedsRamFraction => "single_cache_exceeds_ram_fraction",
            Self::GlobalPinExceedsRamFraction => "global_pin_exceeds_ram_fraction",
            Self::ReuseAmortizesBuild => "reuse_amortizes_build",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RowPrimaryHessianCachePlan {
    pub(super) materialize: bool,
    pub(super) bytes: u64,
    pub(super) stable_capacity_bytes: u64,
    pub(super) runtime_available_bytes: u64,
    pub(super) workspace_pinned_bytes: u64,
    pub(super) single_cache_budget_bytes: u64,
    pub(super) global_pin_budget_bytes: u64,
    pub(super) expected_reuse_passes: usize,
    pub(super) materialized_row_hessian_evals: usize,
    pub(super) streamed_row_hessian_evals: usize,
    pub(super) reason: RowPrimaryHessianCacheReason,
}

pub(super) fn decide_row_primary_hessian_cache(
    n: usize,
    r: usize,
    expected_reuse_passes: usize,
    // Stable, monotone capacity floor (`max` of available RAM ever observed
    // this process). Drives the per-shape single-cache budget so the decision
    // does not flip mid-fit on a transient available-memory dip.
    stable_capacity_bytes: u64,
    // Instantaneous available RAM. Drives only the global-pin OOM guard, the
    // genuine safety valve against over-committing co-resident caches.
    runtime_available_bytes: u64,
    workspace_pinned_bytes: u64,
) -> RowPrimaryHessianCachePlan {
    // Account for neglog (1 per row) + grad (r per row) + hess (r*r per row).
    // For r=20 this is 1+20+400=421 vs 400 hess-only: ~5.25% overhead.
    let floats_per_row = (r as u64)
        .saturating_mul(r as u64)
        .saturating_add(r as u64)
        .saturating_add(1);
    let bytes = (n as u64)
        .saturating_mul(floats_per_row)
        .saturating_mul(std::mem::size_of::<f64>() as u64);
    // Worthwhileness gate keys off the stable floor: a shape that fits the
    // capacity budget once stays materializable for the whole fit.
    let single_cache_budget_bytes = stable_capacity_bytes
        .saturating_mul(BMS_ROW_PRIMARY_HESSIAN_SINGLE_FRACTION_NUM)
        / BMS_ROW_PRIMARY_HESSIAN_SINGLE_FRACTION_DEN.max(1);
    // OOM guard keys off live available RAM: never pin more than the live
    // fraction across all co-resident caches.
    let global_pin_budget_bytes = runtime_available_bytes
        .saturating_mul(BMS_ROW_PRIMARY_HESSIAN_GLOBAL_FRACTION_NUM)
        / BMS_ROW_PRIMARY_HESSIAN_GLOBAL_FRACTION_DEN.max(1);
    let streamed_row_hessian_evals = n.saturating_mul(expected_reuse_passes);
    let materialized_row_hessian_evals = n;
    let reason = if expected_reuse_passes < BMS_ROW_PRIMARY_HESSIAN_MIN_REUSE_PASSES {
        RowPrimaryHessianCacheReason::ReuseTooLow
    } else if bytes >= single_cache_budget_bytes {
        RowPrimaryHessianCacheReason::SingleCacheExceedsRamFraction
    } else if workspace_pinned_bytes.saturating_add(bytes) > global_pin_budget_bytes {
        RowPrimaryHessianCacheReason::GlobalPinExceedsRamFraction
    } else {
        RowPrimaryHessianCacheReason::ReuseAmortizesBuild
    };
    RowPrimaryHessianCachePlan {
        materialize: matches!(reason, RowPrimaryHessianCacheReason::ReuseAmortizesBuild),
        bytes,
        stable_capacity_bytes,
        runtime_available_bytes,
        workspace_pinned_bytes,
        single_cache_budget_bytes,
        global_pin_budget_bytes,
        expected_reuse_passes,
        materialized_row_hessian_evals,
        streamed_row_hessian_evals,
        reason,
    }
}

/// One outer search's predicted working set in bytes, when the search runs
/// alone on `serial_available_bytes` (gnomon#2359, SPEC 10). Each term is what
/// the search allocates, from the element counts and types of the buffers:
///
/// - the row-primary cache (`neglog`, `grad`, `hess`: `n·(r²+r+1)` f64) at the
///   size `decide_row_primary_hessian_cache` gives it on that availability with
///   nothing else pinned: materialized or tiled, or nothing when streamed;
/// - three exact-evaluation caches, the two its own store retains and the one it
///   builds on a miss (`BmsSearchMember`), each with its per-row contexts, its degree-9 and
///   degree-15 cell-moment bundles at `RowCellMomentsBundle::estimated_resident_bytes`
///   over the partition's most cells per row, and its per-row flex third
///   tensors (two `r×r` f64 per row);
/// - on the rigid path, three of each of its own store's per-row third and
///   fourth [`RigidRowTensors`] tables (a lazy `Result` of 8 and of 16 f64 per row);
/// - the row-intercept warm starts: two `u64` and a predictor slot per row,
///   each predictor two `r`-vectors of f64;
/// - the block states' linear predictors, `n` f64 per block;
/// - the joint coefficient Hessian and its factor, two `p×p` f64.
pub(super) fn outer_search_working_set_bytes(
    family: &BernoulliMarginalSlopeFamily,
    specs: &[ParameterBlockSpec],
    serial_available_bytes: u64,
) -> u64 {
    let n = family.y.len() as u64;
    let r = primary_slices(&block_slices(family)).total as u64;
    let p = specs.iter().map(|spec| spec.design.ncols() as u64).sum::<u64>();
    let f64_bytes = std::mem::size_of::<f64>() as u64;
    let flex_active = family.score_warp.is_some() || family.link_dev.is_some();
    let row_primary_cache = if flex_active {
        let plan = decide_row_primary_hessian_cache(
            n as usize,
            r as usize,
            BMS_ROW_PRIMARY_HESSIAN_EXPECTED_REUSE_PASSES,
            serial_available_bytes,
            serial_available_bytes,
            0,
        );
        let tiled = plan.expected_reuse_passes >= BMS_ROW_PRIMARY_HESSIAN_MIN_REUSE_PASSES
            && plan.bytes <= plan.global_pin_budget_bytes;
        if plan.materialize || tiled { plan.bytes } else { 0 }
    } else {
        0
    };
    let cells = n.saturating_mul(family.max_denested_partition_cells_per_row() as u64) as usize;
    let cell_bundles = [9usize, 15]
        .iter()
        .map(|&degree| {
            RowCellMomentsBundle::estimated_resident_bytes(n as usize, cells, degree) as u64
        })
        .sum::<u64>();
    let row_contexts =
        n.saturating_mul(std::mem::size_of::<BernoulliMarginalSlopeRowExactContext>() as u64);
    let flex_third = if flex_active {
        n.saturating_mul(2 * r * r).saturating_mul(f64_bytes)
    } else {
        0
    };
    let exact_eval_caches = 3 * (row_contexts + cell_bundles + flex_third);
    let rigid_tensors = if flex_active {
        0
    } else {
        let row_bytes = RigidRowTensors::<[[[f64; 2]; 2]; 2]>::row_bytes()
            + RigidRowTensors::<[[[[f64; 2]; 2]; 2]; 2]>::row_bytes();
        3 * n.saturating_mul(row_bytes as u64)
    };
    let predictor_slot = std::mem::size_of::<Mutex<Option<BernoulliInterceptPredictorWarmStart>>>()
        as u64
        + 2 * r * f64_bytes;
    let intercept_warm_starts =
        n.saturating_mul(2 * std::mem::size_of::<AtomicU64>() as u64 + predictor_slot);
    let block_predictors = n.saturating_mul(specs.len() as u64).saturating_mul(f64_bytes);
    let joint_hessian = 2 * p.saturating_mul(p).saturating_mul(f64_bytes);
    row_primary_cache
        .saturating_add(exact_eval_caches)
        .saturating_add(rigid_tensors)
        .saturating_add(intercept_warm_starts)
        .saturating_add(block_predictors)
        .saturating_add(joint_hessian)
}

/// RAII handle around a materialized row-primary evaluation cache
/// (neglog + gradient + Hessian) that decrements the pinned-bytes counter it
/// charged on drop: its search's own inside a multistart lane, else the
/// process-global one.
pub struct RowPrimaryEvalPin {
    /// Per-row negative log-likelihood, length `n`.
    pub(super) neglog: Array1<f64>,
    /// Per-row gradient, shape `(n, r)`.
    pub(super) grad: Array2<f64>,
    /// Per-row Hessian, shape `(n, r*r)`.
    pub(super) hess: Array2<f64>,
    pub(super) bytes: u64,
    pub(super) lane: Option<Arc<gam_runtime::resource::SearchLaneBudget>>,
}

pub(super) struct RowPrimaryEvalTile {
    pub(super) row_start: usize,
    pub(super) rows: RowPrimaryEvalPin,
}

pub(crate) struct RowPrimaryEvalTiles {
    pub(super) n_rows: usize,
    pub(super) r: usize,
    /// Uniform row stride the tiles were built at (every tile except possibly
    /// the last holds exactly `tile_rows` rows). Lets `tile_for_row` resolve a
    /// global row to its tile by a single division instead of a linear scan —
    /// the lookup is on the per-row hot path of the fused gradient / dense / HVP
    /// passes, which call it once or twice for every one of the `n` rows.
    pub(super) tile_rows: usize,
    pub(super) tiles: Vec<RowPrimaryEvalTile>,
}

impl RowPrimaryEvalTiles {
    pub(super) fn new(
        n_rows: usize,
        r: usize,
        tile_rows: usize,
        tiles: Vec<RowPrimaryEvalTile>,
    ) -> Self {
        Self {
            n_rows,
            r,
            tile_rows,
            tiles,
        }
    }

    #[inline]
    pub(super) fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    #[inline]
    pub(super) fn tile_for_row(&self, row: usize) -> Option<(&RowPrimaryEvalTile, usize)> {
        // Tiles are built at a uniform `tile_rows` stride starting at row 0, so
        // the owning tile index is `row / tile_rows`. Resolve it directly and
        // confirm the row falls inside the tile's actual length (the final tile
        // may be shorter). Fall back to a linear scan only if the arithmetic
        // guess does not contain the row — a defensive path for any future
        // non-uniform tiling rather than a hot-path cost.
        if self.tile_rows > 0 {
            let guess = row / self.tile_rows;
            if let Some(tile) = self.tiles.get(guess) {
                let len = tile.rows.neglog().len();
                if row >= tile.row_start && row < tile.row_start + len {
                    return Some((tile, row - tile.row_start));
                }
            }
        }
        for tile in &self.tiles {
            let len = tile.rows.neglog().len();
            if row >= tile.row_start && row < tile.row_start + len {
                return Some((tile, row - tile.row_start));
            }
        }
        None
    }

    #[inline]
    pub(super) fn total_bytes(&self) -> u64 {
        self.tiles.iter().map(|tile| tile.rows.bytes).sum()
    }
}

impl RowPrimaryEvalPin {
    pub(super) fn new(
        neglog: Array1<f64>,
        grad: Array2<f64>,
        hess: Array2<f64>,
        bytes: u64,
        lane: Option<Arc<gam_runtime::resource::SearchLaneBudget>>,
    ) -> Self {
        match lane.as_deref() {
            Some(lane) => lane.pinned_bytes().fetch_add(bytes, Ordering::AcqRel),
            None => bms_row_primary_hessian_pinned_bytes().fetch_add(bytes, Ordering::AcqRel),
        };
        Self {
            neglog,
            grad,
            hess,
            bytes,
            lane,
        }
    }

    pub(super) fn neglog(&self) -> &Array1<f64> {
        &self.neglog
    }

    pub(super) fn grad(&self) -> &Array2<f64> {
        &self.grad
    }

    pub(super) fn hess(&self) -> &Array2<f64> {
        &self.hess
    }
}

impl Drop for RowPrimaryEvalPin {
    fn drop(&mut self) {
        match self.lane.as_deref() {
            Some(lane) => lane.pinned_bytes().fetch_sub(self.bytes, Ordering::AcqRel),
            None => bms_row_primary_hessian_pinned_bytes().fetch_sub(self.bytes, Ordering::AcqRel),
        };
    }
}

/// Per-fit row-primary evaluation cache: stores neglog + gradient + Hessian
/// for every row so that downstream passes (fused gradient+dense-H, HVP,
/// diagonal) never recompute the row kernel.
///
/// Variants:
/// - `Empty`: cache not materialized (rigid path or caller opted out).
/// - `Host`: all three arrays live in host RAM.
///   Consumed by the CPU per-row Hv / diagonal / direct-product loops and by
///   the fused gradient+dense-H path via
///   [`BernoulliMarginalSlopeFamily::cached_row_primary_hessian`] and
///   [`BernoulliMarginalSlopeFamily::cached_row_primary_eval`].
/// - `Device` (Linux/CUDA only): row value, gradient, Hessian, and designs live
///   on the GPU. Log-likelihood / score / HVP / diagonal / dense consumers route
///   through device entry points. Widths above the direct dense kernel's
///   shared-memory bound materialize through bounded multi-RHS device HVPs;
///   device failures propagate instead of changing algorithms.
pub enum RowPrimaryEvalCache {
    Empty,
    Host(RowPrimaryEvalPin),
    /// Bounded host-resident row-primary Hessian tiles. This is selected when
    /// the monolithic `n × (1+r+r²)` host cache is rejected by the single-cache
    /// worthwhileness gate but the full set of tiles fits under the live global
    /// pin budget. HVP and diagonal consumers stream tile-by-tile, so peak
    /// build scratch stays one tile wide and the inner operator never falls
    /// back to recomputing row Hessians per probe.
    Tiled(RowPrimaryEvalTiles),
    /// Device-resident row value + gradient + Hessian + designs. Every
    /// downstream joint-value/score/Hessian consumer routes through the
    /// device-aware entry points.
    #[cfg(target_os = "linux")]
    Device(crate::bms::gpu::row::DeviceResidentRowHess),
}

impl RowPrimaryEvalCache {
    /// Returns `true` when the cache is materialized (host or device).
    #[inline]
    pub(crate) fn is_some(&self) -> bool {
        !matches!(self, Self::Empty)
    }

    #[inline]
    pub(crate) fn is_tiled(&self) -> bool {
        matches!(self, Self::Tiled(_))
    }

    #[inline]
    pub(crate) fn tiles(&self) -> Option<&RowPrimaryEvalTiles> {
        match self {
            Self::Tiled(tiles) => Some(tiles),
            _ => None,
        }
    }

    /// Returns the host-resident pin when the cache is materialised as a
    /// host pin. Returns `None` for the device-resident variant — callers
    /// that need to read the full `r x r` Hessian per row must route through
    /// the device-aware HVP / diagonal entry points.
    #[inline]
    pub(crate) fn host_pin(&self) -> Option<&RowPrimaryEvalPin> {
        match self {
            Self::Host(pin) => Some(pin),
            Self::Tiled(_) => None,
            Self::Empty => None,
            #[cfg(target_os = "linux")]
            Self::Device(_) => None,
        }
    }

    /// Returns the device-resident Hessian state when the cache lives on the
    /// GPU. `None` on every other variant (and on non-Linux builds).
    #[cfg(target_os = "linux")]
    #[inline]
    pub(crate) fn device(&self) -> Option<&crate::bms::gpu::row::DeviceResidentRowHess> {
        match self {
            Self::Device(hess) => Some(hess),
            _ => None,
        }
    }

    /// Reject a host row-calculus entry point after the device cache has been
    /// selected. Device selection is an algorithm commitment, so a caller
    /// must consume the resident value/gradient/Hessian channels instead of
    /// silently recomputing the canonical row program on CPU.
    /// Whether this cache is device-resident. Structurally `false` off-Linux,
    /// where the `Device` variant does not exist, so the rejection below reads
    /// the same on every platform instead of cfg-ing the check away (which left
    /// `operation` unused off-Linux and broke the macOS/Windows wheel builds
    /// under `-D warnings`).
    #[inline]
    fn is_device_resident(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            matches!(self, Self::Device(_))
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    #[inline]
    pub(crate) fn reject_device_cpu_recompute(&self, operation: &str) -> Result<(), String> {
        if self.is_device_resident() {
            return Err(format!(
                "BMS {operation}: device-resident row evaluation selected; CPU row recomputation is forbidden"
            ));
        }
        Ok(())
    }
}

/// Per-row axis-projected FLEX third/fourth-derivative tensor algebra backing
/// the outer-derivative fast path (gam#683). Every outer-derivative consumer
/// contracts the per-row third/fourth tensors against ψ-axis directions that
/// are *single-axis* in primary space — nonzero only at `primary.q` (block 0,
/// "q") or `primary.slope` (block 1, "g"). By the (bi)linearity of the
/// contraction,
///
/// ```text
///   third_contracted(s·e_a)              = s·T3[a]
///   fourth_contracted(s_u·e_a, s_v·e_b)  = s_u·s_v·T4[a][b]
/// ```
///
/// so caching `T3[a]` for `a ∈ {q, g}` and the symmetric `T4[a][b]` once per
/// β-cache turns each `(ρ-axis i, ρ-axis j)` pair into a scalar×matrix scale of
/// a precomputed tensor instead of re-walking every cubic partition cell. All
/// matrices are `r×r` with `r = primary.total`.
///
/// The third and fourth caches are intentionally separate: first-order outer
/// derivative paths need only degree-15 T3 tensors and must not accidentally
/// force degree-21 fourth-order cell work.
pub(super) struct FlexAxisThirdRowTensors {
    /// Third-derivative tensor contracted with the q-axis basis vector
    /// (`third[0]`) and the slope-axis basis vector (`third[1]`).
    pub(super) third: [Array2<f64>; 2],
}

pub(super) struct FlexAxisFourthRowTensors {
    /// Symmetric fourth-derivative tensor contracted with `(e_q, e_q)`.
    pub(super) qq: Array2<f64>,
    /// Symmetric fourth-derivative tensor contracted with `(e_q, e_g)`.
    pub(super) qg: Array2<f64>,
    /// Symmetric fourth-derivative tensor contracted with `(e_g, e_g)`.
    pub(super) gg: Array2<f64>,
}

/// A table of per-row rigid tensors, each built once, on its row's first read
/// (gam#3022). It backs the exact cache's `rigid_third_full` and
/// `rigid_fourth_full`, and the rigid row kernel's tables.
///
/// Each row is a `OnceLock`, not a `RayonSafeOnce`: the row's first reader builds
/// it and every concurrent reader of that row waits for that build instead of
/// repeating it. `RayonSafeOnce` would let every reader that arrives before the
/// publish build the row again, up to the reader count when several tasks sweep
/// the same rows in the same order. Waiting is safe because a row build is serial
/// arithmetic (the row's closed-form jet and its anchor root, read and written
/// through atomic slots): it runs no Rayon work, so the builder never steals a
/// task that could read this row. A failed build's `Err` stays in its row and
/// reaches every reader of that row. Holders share one table across evaluations
/// at one β by holding it in their same-β store.
pub(super) struct RigidRowTensors<T> {
    rows: Vec<std::sync::OnceLock<Result<T, String>>>,
}

impl<T> RigidRowTensors<T> {
    /// `n_rows` rows, none built.
    pub(super) fn new(n_rows: usize) -> Self {
        Self {
            rows: (0..n_rows).map(|_| std::sync::OnceLock::new()).collect(),
        }
    }

    /// The bytes one row occupies, built or not.
    pub(super) const fn row_bytes() -> usize {
        std::mem::size_of::<std::sync::OnceLock<Result<T, String>>>()
    }

    /// Row `row`'s tensor, built by `build` on the row's first read. `build` must
    /// run no Rayon work (see the type's contract).
    pub(super) fn row(
        &self,
        row: usize,
        build: impl FnOnce() -> Result<T, String>,
    ) -> Result<&T, String> {
        self.rows
            .get(row)
            .ok_or_else(|| {
                format!(
                    "rigid row tensor table: row {row} out of range for {} rows",
                    self.rows.len()
                )
            })?
            .get_or_init(build)
            .as_ref()
            .map_err(Clone::clone)
    }
}

/// Lazy derivative-channel cache for one canonical BMS FLEX row program.
///
/// The row has one semantic program identity, while third- and fourth-order
/// contractions retain independent lazy cells so a VGH/first-outer pass never
/// forces degree-21 moments. Sharing the outer row slot removes the former
/// pair of parallel cache hierarchies without coupling their work budgets.
pub(super) struct BmsFlexRowProgramDerivativeCache {
    pub(super) third: gam_runtime::resource::RayonSafeOnce<Result<FlexAxisThirdRowTensors, String>>,
    pub(super) fourth:
        gam_runtime::resource::RayonSafeOnce<Result<FlexAxisFourthRowTensors, String>>,
}

impl BmsFlexRowProgramDerivativeCache {
    pub(super) fn new() -> Self {
        Self {
            third: gam_runtime::resource::RayonSafeOnce::new(),
            fourth: gam_runtime::resource::RayonSafeOnce::new(),
        }
    }
}

/// Shared precomputed state plus pre-solved per-row contexts. All row
/// intercepts are solved once during cache construction so that workspace
/// calls (matvec, diagonal, psi, directional derivatives) never redundantly
/// re-solve the Newton intercept equation.
pub(super) struct BernoulliMarginalSlopeExactEvalCache {
    pub(super) slices: BlockSlices,
    pub(super) primary: PrimarySlices,
    /// Pre-solved row contexts (intercept, M_a, observed score-warp value).
    pub(super) row_contexts: Vec<BernoulliMarginalSlopeRowExactContext>,
    /// Batched per-row denested cell moments for the current β snapshot.
    /// Built once at exact-cache construction (after row intercepts converge)
    /// and consumed by row gradient/Hessian/Hv/diagonal/derivative-tensor
    /// paths via `RowCellMomentsBundle::row(row, required_degree)`. May be
    /// `None` when the FLEX path is inactive, when an empirical latent grid
    /// drives the row kernel through a non-cell path, or when the estimated
    /// resident bytes would exceed the active resource policy budget.
    pub(super) row_cell_moments: Option<RowCellMomentsBundle>,
    /// Certified Chebyshev cell-moment family forest for the current β
    /// snapshot (#979 Stage C). Built when the FLEX path is active on the
    /// standard-normal latent measure and the row-cell-moments bundle was
    /// refused by the resource budget — i.e. exactly the large-n regime
    /// where every row evaluation otherwise re-runs ladder quadrature per
    /// cell. Rows/cells without a certified family fall back to the ladder
    /// unchanged.
    pub(super) cell_family_forest: Option<crate::cell_moment_family::CellFamilyForest>,
    /// Lazily-built degree-15 bundle for outer dH (1st-derivative of Hessian)
    /// trace paths. Only populated when those paths actually execute.
    /// `RayonSafeOnce` keeps lazy initialization safe from parallel row passes.
    pub(super) row_cell_moments_d15:
        gam_runtime::resource::RayonSafeOnce<Result<Option<RowCellMomentsBundle>, String>>,
    /// Lazily-built degree-21 bundle for outer d²H (2nd-derivative of Hessian)
    /// trace paths. Only populated when those paths actually execute.
    /// `RayonSafeOnce` keeps lazy initialization safe from parallel row passes.
    pub(super) row_cell_moments_d21:
        gam_runtime::resource::RayonSafeOnce<Result<Option<RowCellMomentsBundle>, String>>,
    /// Flexible-path per-β per-row primary Hessians (`r×r` blocks flattened
    /// row-major into one wide `Array2`).  The matrix-free inner Newton/CG
    /// loop contracts the same primary Hessian against many trial directions
    /// at the same β; materializing each row's Hessian once per workspace
    /// avoids rebuilding cell moments + reduced flex jets on every Hv product.
    /// `None` whenever the flex path is inactive (rigid kernel) or the
    /// caller did not opt in to materialization.
    pub(super) row_primary_hessians: RowPrimaryEvalCache,
    /// Per-row uncontracted third-derivative tensor in the rigid path,
    /// lazily built on first access. The `build_psi_hyper_coords` row pass
    /// hits `rigid_row_third_contracted` once per (row, ψ-axis) — 32× per
    /// row at large-scale shape — but the per-row jet is axis-invariant. This
    /// cache lets the closed-form third-derivative tensor
    /// (`empirical_rigid_third_full_closed_form`, or `rigid_standard_normal_third_full`
    /// for the standard-normal measure) run at most once per row per cache
    /// lifetime; per-axis callers reduce to a 2×2 [`contract_third_full`].
    ///
    /// Two-level lazy, like `flex_row_program_derivatives`: the outer slot
    /// allocates a [`RigidRowTensors`] table on first touch, and each row's
    /// tensor is built serially by the first reader of that row while any other
    /// reader of the row waits. Each row is therefore built exactly once however
    /// many tasks read it, and a subsampled pass builds only the rows it samples.
    /// Stored per row as `Result` because the build is fallible (a per-row jet
    /// may surface a non-finite value); a row's failure is sticky and
    /// propagated identically to every reader of that row.
    pub(super) rigid_third_full:
        gam_runtime::resource::RayonSafeOnce<RigidRowTensors<[[[f64; 2]; 2]; 2]>>,

    /// Per-row uncontracted fourth-derivative tensor in the rigid path —
    /// the second-order analogue of `rigid_third_full`, in the same per-row
    /// slots. The outer-Hessian build at large-scale shape evaluates
    /// `rigid_row_fourth_contracted` for
    /// every (ψ-axis-i, ψ-axis-j) pair: `(rank² + rank)/2 ≈ 528` pairs at
    /// rank=32. Per-row, the five distinct components are axis-invariant,
    /// so caching them lets every pair contraction be a 16-multiply 2×2
    /// bilinear instead of a fresh 8-direction empirical jet.
    pub(super) rigid_fourth_full:
        gam_runtime::resource::RayonSafeOnce<RigidRowTensors<[[[[f64; 2]; 2]; 2]; 2]>>,

    /// One lazy slot per canonical FLEX row program. Each slot owns separate
    /// third/fourth channel cells, preserving order-specific work while making
    /// the program/cache identity single-sourced.
    pub(super) flex_row_program_derivatives:
        gam_runtime::resource::RayonSafeOnce<Vec<BmsFlexRowProgramDerivativeCache>>,

    /// Lazily-built full-data outer row list (`index = position`, `weight = 1.0`,
    /// `stratum = 0` for every row in `0..n`). The full-data variant of
    /// `outer_weighted_rows` depends only on `n`, which is constant across the
    /// cache lifetime, yet the joint-Hessian / ψ directional-derivative
    /// operator paths call it once per outer eval — re-allocating and filling a
    /// length-`n` (`n≈3e5`) `Vec<WeightedOuterRow>` (24 B/row, ~8.5 MB) on every
    /// call. Building it once here and lending a borrowed slice removes that
    /// per-eval churn; the subsampled path stays a per-call owned `Vec` (it is
    /// short — only the masked rows — and varies with the mask).
    pub(super) full_data_outer_rows: std::sync::OnceLock<std::sync::Arc<Vec<WeightedOuterRow>>>,
}

impl BernoulliMarginalSlopeExactEvalCache {
    /// Outer row list for an eval, reusing the cached full-data `0..n` list
    /// when no subsample is active. Bit-identical to calling
    /// `outer_weighted_rows(options, n)` directly: the full-data branch yields
    /// the same `(index, 1.0, 0)` rows in the same order, and the subsampled
    /// branch is the unmodified per-call owned `Vec`.
    pub(super) fn outer_weighted_rows_cached<'a>(
        &'a self,
        options: &crate::custom_family::BlockwiseFitOptions,
        n: usize,
    ) -> std::borrow::Cow<'a, [WeightedOuterRow]> {
        if options.outer_score_subsample.is_some() {
            return std::borrow::Cow::Owned(outer_weighted_rows(options, n));
        }
        let rows = self.full_data_outer_rows.get_or_init(|| {
            std::sync::Arc::new(
                (0..n)
                    .map(|index| WeightedOuterRow {
                        index,
                        weight: 1.0,
                        stratum: 0,
                    })
                    .collect(),
            )
        });
        std::borrow::Cow::Borrowed(rows.as_slice())
    }
}

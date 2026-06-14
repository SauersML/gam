use super::hessian_paths::{
    BernoulliMarginalSlopeRowExactContext, BlockSlices, PrimarySlices, RowCellMomentsBundle,
};
use super::*;

#[inline]
pub(super) fn log_exact_work(n: usize) -> bool {
    n >= EXACT_WORK_LOG_MIN_ROWS
}

/// Cross-platform available-RAM probe backed by `sysinfo`. Returns the bytes
/// the OS reports as available for new allocations (free + reclaimable cache);
/// the underlying `System` instance is leaked behind a `OnceLock` so the cost
/// of `new_with_specifics` is paid once per process.
pub(super) fn runtime_available_memory_bytes() -> u64 {
    pub(crate) static SYSTEM: OnceLock<Mutex<sysinfo::System>> = OnceLock::new();
    let lock = SYSTEM.get_or_init(|| {
        let refresh =
            sysinfo::RefreshKind::new().with_memory(sysinfo::MemoryRefreshKind::everything());
        Mutex::new(sysinfo::System::new_with_specifics(refresh))
    });
    let mut system = lock.lock().expect("sysinfo system mutex poisoned");
    system.refresh_memory_specifics(sysinfo::MemoryRefreshKind::everything());
    system.available_memory()
}

/// Process-global counter of bytes currently pinned by live BMS row-primary
/// evaluation caches. Incremented by [`RowPrimaryEvalPin::new`] when a cache
/// is materialized and decremented on `Drop`, so two co-resident workspaces
/// cannot together pin more than `available_ram * GLOBAL_FRACTION`.
pub(super) fn bms_row_primary_hessian_pinned_bytes() -> &'static AtomicU64 {
    pub(crate) static PINNED: OnceLock<AtomicU64> = OnceLock::new();
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
    pub(crate) static FLOOR: OnceLock<AtomicU64> = OnceLock::new();
    FLOOR.get_or_init(|| AtomicU64::new(0))
}

/// Fold the latest observed available-RAM reading into the monotone capacity
/// floor and return the resulting stable budget basis (`max(floor, observed)`).
pub(super) fn observe_capacity_floor(runtime_available_bytes: u64) -> u64 {
    bms_row_primary_hessian_capacity_floor()
        .fetch_max(runtime_available_bytes, Ordering::AcqRel)
        .max(runtime_available_bytes)
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

/// RAII handle around a materialized row-primary evaluation cache
/// (neglog + gradient + Hessian) that decrements the process-global
/// pinned-bytes counter on drop.
pub struct RowPrimaryEvalPin {
    /// Per-row negative log-likelihood, length `n`.
    pub(super) neglog: Array1<f64>,
    /// Per-row gradient, shape `(n, r)`.
    pub(super) grad: Array2<f64>,
    /// Per-row Hessian, shape `(n, r*r)`.
    pub(super) hess: Array2<f64>,
    pub(super) bytes: u64,
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
    ) -> Self {
        bms_row_primary_hessian_pinned_bytes().fetch_add(bytes, Ordering::AcqRel);
        Self {
            neglog,
            grad,
            hess,
            bytes,
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
        bms_row_primary_hessian_pinned_bytes().fetch_sub(self.bytes, Ordering::AcqRel);
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
/// - `Device` (Linux/CUDA only): row Hessian + designs live on the GPU.
///   HVP / diagonal / dense-block consumers route through the device-aware
///   GPU entry points; the fused CPU gradient pass is the rare fallback (only
///   when `p_total` exceeds the dense-block kernel's shared-memory cap) and
///   recomputes the row kernel on the fly in that case, so the GPU output
///   for `(neglog, grad)` is not mirrored on the host.
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
    /// Device-resident row Hessian + designs. HVP / diagonal / dense-block
    /// consumers route through the device-aware GPU entry points.
    #[cfg(target_os = "linux")]
    Device(crate::families::bms::gpu::row::DeviceResidentRowHess),
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
    /// that need to read the full `r x r` Hessian per row must either
    /// route through the device-aware HVP / diagonal entry points or fall
    /// back to recomputing the row Hessian on the fly.
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
    pub(crate) fn device(&self) -> Option<&crate::families::bms::gpu::row::DeviceResidentRowHess> {
        match self {
            Self::Device(hess) => Some(hess),
            _ => None,
        }
    }
}

/// Per-row axis-projected FLEX third/fourth-derivative tensor algebra backing
/// the outer-derivative fast path (gam#683). Every outer-derivative consumer
/// contracts the per-row third/fourth tensors against ψ-axis directions that
/// are *single-axis* in primary space — nonzero only at `primary.q` (block 0,
/// "q") or `primary.logslope` (block 1, "g"). By the (bi)linearity of the
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
    /// (`third[0]`) and the logslope-axis basis vector (`third[1]`).
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
    pub(super) cell_family_forest: Option<crate::families::cell_moment_family::CellFamilyForest>,
    /// Lazily-built degree-15 bundle for outer dH (1st-derivative of Hessian)
    /// trace paths. Only populated when those paths actually execute.
    /// `RayonSafeOnce` keeps lazy initialization safe from parallel row passes.
    pub(super) row_cell_moments_d15:
        crate::resource::RayonSafeOnce<Result<Option<RowCellMomentsBundle>, String>>,
    /// Lazily-built degree-21 bundle for outer d²H (2nd-derivative of Hessian)
    /// trace paths. Only populated when those paths actually execute.
    /// `RayonSafeOnce` keeps lazy initialization safe from parallel row passes.
    pub(super) row_cell_moments_d21:
        crate::resource::RayonSafeOnce<Result<Option<RowCellMomentsBundle>, String>>,
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
    /// Stored as `Result` because the build is fallible (per-row jet may
    /// surface a non-finite value). `RayonSafeOnce` keeps lazy initialization
    /// safe when the first caller is already inside a Rayon row pass; failure
    /// is sticky and propagated identically to every caller.
    pub(super) rigid_third_full:
        crate::resource::RayonSafeOnce<Result<Vec<[[[f64; 2]; 2]; 2]>, String>>,

    /// Per-row uncontracted fourth-derivative tensor in the rigid path —
    /// the second-order analogue of `rigid_third_full`. The outer-Hessian
    /// build at large-scale shape evaluates `rigid_row_fourth_contracted` for
    /// every (ψ-axis-i, ψ-axis-j) pair: `(rank² + rank)/2 ≈ 528` pairs at
    /// rank=32. Per-row, the five distinct components are axis-invariant,
    /// so caching them lets every pair contraction be a 16-multiply 2×2
    /// bilinear instead of a fresh 8-direction empirical jet.
    pub(super) rigid_fourth_full:
        crate::resource::RayonSafeOnce<Result<Vec<[[[[f64; 2]; 2]; 2]; 2]>, String>>,

    /// Flexible-path per-row axis-projected third-derivative tensors. See
    /// [`FlexAxisThirdRowTensors`] for the contraction algebra. Only consulted
    /// on the FLEX path — rigid rows keep their own `rigid_third_full` cache.
    ///
    /// Two-level lazy: the outer `RayonSafeOnce` allocates a per-row slot table
    /// (one inner `RayonSafeOnce` per global row) on first touch; each row's
    /// tensors are then built **on demand** when that row is first read. Outer
    /// derivative passes are row-subsampled, so per-row laziness builds (and
    /// risks erroring on) only the rows actually consumed, not all `n`. Each
    /// inner build is fallible and sticky (same contract as `rigid_third_full`).
    pub(super) flex_axis_third_tensors: crate::resource::RayonSafeOnce<
        Vec<crate::resource::RayonSafeOnce<Result<FlexAxisThirdRowTensors, String>>>,
    >,

    /// Flexible-path per-row axis-projected fourth-derivative tensors. Built
    /// independently from `flex_axis_third_tensors` so first-order outer work
    /// never forces degree-21 fourth-order cell moments.
    pub(super) flex_axis_fourth_tensors: crate::resource::RayonSafeOnce<
        Vec<crate::resource::RayonSafeOnce<Result<FlexAxisFourthRowTensors, String>>>,
    >,
}

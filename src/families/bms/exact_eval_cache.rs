use super::*;
use super::hessian_paths::{BlockSlices, PrimarySlices, BernoulliMarginalSlopeRowExactContext, RowCellMomentsBundle};


#[inline]
pub(super) fn log_exact_work(n: usize) -> bool {
    n >= EXACT_WORK_LOG_MIN_ROWS
}

/// Cross-platform available-RAM probe backed by `sysinfo`. Returns the bytes
/// the OS reports as available for new allocations (free + reclaimable cache);
/// the underlying `System` instance is leaked behind a `OnceLock` so the cost
/// of `new_with_specifics` is paid once per process.
pub(super) fn runtime_available_memory_bytes() -> u64 {
    static SYSTEM: OnceLock<Mutex<sysinfo::System>> = OnceLock::new();
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
    static PINNED: OnceLock<AtomicU64> = OnceLock::new();
    PINNED.get_or_init(|| AtomicU64::new(0))
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
    let single_cache_budget_bytes = runtime_available_bytes
        .saturating_mul(BMS_ROW_PRIMARY_HESSIAN_SINGLE_FRACTION_NUM)
        / BMS_ROW_PRIMARY_HESSIAN_SINGLE_FRACTION_DEN.max(1);
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

impl RowPrimaryEvalPin {
    pub(super) fn new(neglog: Array1<f64>, grad: Array2<f64>, hess: Array2<f64>, bytes: u64) -> Self {
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
    /// Device-resident row Hessian + designs. HVP / diagonal / dense-block
    /// consumers route through the device-aware GPU entry points.
    #[cfg(target_os = "linux")]
    Device(crate::gpu::bms_flex_row::DeviceResidentRowHess),
}

impl RowPrimaryEvalCache {
    /// Returns `true` when the cache is materialized (host or device).
    #[inline]
    pub(crate) fn is_some(&self) -> bool {
        !matches!(self, Self::Empty)
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
            Self::Empty => None,
            #[cfg(target_os = "linux")]
            Self::Device(_) => None,
        }
    }

    /// Returns the device-resident Hessian state when the cache lives on the
    /// GPU. `None` on every other variant (and on non-Linux builds).
    #[cfg(target_os = "linux")]
    #[inline]
    pub(crate) fn device(&self) -> Option<&crate::gpu::bms_flex_row::DeviceResidentRowHess> {
        match self {
            Self::Device(hess) => Some(hess),
            _ => None,
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
    /// row at biobank shape — but the per-row jet is axis-invariant. This
    /// cache lets the heavy `empirical_rigid_neglog_jet` (or its closed-form
    /// equivalent) run at most once per row per cache lifetime; per-axis
    /// callers reduce to a 2×2 [`contract_third_full`].
    ///
    /// Stored as `Result` because the build is fallible (per-row jet may
    /// surface a non-finite value). `RayonSafeOnce` keeps lazy initialization
    /// safe when the first caller is already inside a Rayon row pass; failure
    /// is sticky and propagated identically to every caller.
    pub(super) rigid_third_full: crate::resource::RayonSafeOnce<Result<Vec<[[[f64; 2]; 2]; 2]>, String>>,

    /// Per-row uncontracted fourth-derivative tensor in the rigid path —
    /// the second-order analogue of `rigid_third_full`. The outer-Hessian
    /// build at biobank shape evaluates `rigid_row_fourth_contracted` for
    /// every (ψ-axis-i, ψ-axis-j) pair: `(rank² + rank)/2 ≈ 528` pairs at
    /// rank=32. Per-row, the five distinct components are axis-invariant,
    /// so caching them lets every pair contraction be a 16-multiply 2×2
    /// bilinear instead of a fresh 8-direction empirical jet.
    pub(super) rigid_fourth_full: crate::resource::RayonSafeOnce<Result<Vec<[[[[f64; 2]; 2]; 2]; 2]>, String>>,
}

//! Automatic GPU dispatch shim for dense linear algebra hot kernels.
//!
//! Every `try_*` entry point in this module is invoked unconditionally from
//! `gam_linalg::faer_ndarray` before the CPU fast-path runs. The decision to send
//! the kernel to a device is fully automatic and never requires a user-facing
//! flag — it depends only on:
//!
//!   1. Lossless runtime resolution returning an available device.
//!   2. The kernel being large enough to amortize launch/PCIe overhead, per
//!      the thresholds in `policy::GpuDispatchPolicy`.
//!   3. cudarc successfully dynamically loading `libcuda` at process startup
//!      via its `fallback-dynamic-loading` feature. When the loader fails
//!      (no driver, no toolkit installed), Auto receives typed absence and
//!      every `try_*` returns `None` so the caller falls through to the
//!      existing faer CPU kernel. Probe faults and Required absence fail
//!      loudly instead of being reclassified as an optional decline.
//!
//! The wiring lives here so `solver/pirls.rs` and the family Hessian
//! assemblers can stay backend-agnostic: they call `gam_linalg::faer_ndarray::fast_*`
//! and get GPU acceleration automatically whenever it is profitable.

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};

use super::device_runtime::GpuRuntime;
use super::policy::GpuDispatchPolicy;
use super::GpuPolicy;

pub struct CudaGemmDispatch;

impl gam_linalg::gpu_hook::GpuGemmDispatch for CudaGemmDispatch {
    fn try_fast_atb(&self, a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
        try_fast_atb(a, b)
    }

    fn try_fast_ab(&self, a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
        try_fast_ab(a, b)
    }

    fn try_fast_av(&self, a: ArrayView2<'_, f64>, v: ArrayView1<'_, f64>) -> Option<Array1<f64>> {
        try_fast_av(a, v)
    }

    fn try_fast_atv(&self, a: ArrayView2<'_, f64>, v: ArrayView1<'_, f64>) -> Option<Array1<f64>> {
        try_fast_atv(a, v)
    }

    fn try_fast_xt_diag_x(
        &self,
        x: ArrayView2<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> Option<Array2<f64>> {
        try_fast_xt_diag_x(x, w)
    }

    fn try_fast_xt_diag_y(
        &self,
        x: ArrayView2<'_, f64>,
        w: ArrayView1<'_, f64>,
        y: ArrayView2<'_, f64>,
    ) -> Option<Array2<f64>> {
        try_fast_xt_diag_y(x, w, y)
    }

    fn try_fast_joint_hessian_2x2(
        &self,
        x_a: ArrayView2<'_, f64>,
        x_b: ArrayView2<'_, f64>,
        w_aa: ArrayView1<'_, f64>,
        w_ab: ArrayView1<'_, f64>,
        w_bb: ArrayView1<'_, f64>,
    ) -> Option<Array2<f64>> {
        try_fast_joint_hessian_2x2(x_a, x_b, w_aa, w_ab, w_bb)
    }

    fn device_count(&self) -> usize {
        let policy = super::global_policy();
        runtime_for_dispatch(policy).map_or(0, GpuRuntime::device_count)
    }

    fn try_fast_ab_broadcast_b_batched(
        &self,
        a3: ArrayView3<'_, f64>,
        b: ArrayView2<'_, f64>,
    ) -> Option<Array3<f64>> {
        try_fast_ab_broadcast_b_batched(a3, b)
    }
}

/// Discriminator used by [`route_through_gpu`] to apply the right
/// size threshold from [`super::policy::GpuDispatchPolicy`].
#[derive(Clone, Copy, Debug)]
pub enum DispatchOp {
    /// Generic matrix-matrix product with the given output dims and reduction depth.
    Gemm { m: usize, n: usize, k: usize },
    /// Batch of independent matrix-matrix products.
    BatchedGemm {
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
    },
    /// Dense Cholesky factorization.
    Potrf { p: usize, batch: usize },
    /// Batched small-dense Cholesky factorization where each block has the
    /// same small width `p` (≲ 32) but the batch is large. Routed through
    /// `cusolverDnDpotrfBatched` and kept device-resident for downstream
    /// triangular solves (Arrow-Schur, Stage-3 PIRLS).
    SmallDenseBatchedPotrf { p: usize, batch: usize },
    /// Triangular matrix solve.
    Trsm { m: usize, n: usize },
    /// Matrix-vector (or matrix · single-column) product.
    Gemv { m: usize, k: usize },
    /// `Xᵀ · diag(w) · X` reduction with n rows and p columns.
    XtDiagX { n: usize, p: usize },
    /// `Xᵀ · diag(w) · Y` reduction; px and q are the design and response widths.
    XtDiagY { n: usize, px: usize, q: usize },
    /// 2×2 joint Hessian block with two design widths.
    JointHessian2x2 { n: usize, pa: usize, pb: usize },
}

/// Resolve the runtime at the infallible `gam-linalg` optimization-hook
/// boundary. Typed Auto/Off absence declines the optional acceleration;
/// faults and Required absence cannot be represented by the hook's `Option`
/// return type, so they fail loudly rather than silently executing a CPU path.
#[inline]
fn runtime_for_dispatch(policy: GpuPolicy) -> Option<&'static GpuRuntime> {
    GpuRuntime::resolve(policy).unwrap_or_else(|error| {
        // SAFETY: the gam-linalg optimization hook is an infallible `Option`
        // boundary. `None` means "run on the CPU", so returning it for a probe
        // fault or Required-device absence would silently change semantics.
        panic!(
            "GPU runtime resolution failed under policy '{}': {error}",
            policy
        )
    })
}

/// Preserve the semantic difference between an optional Auto acceleration and
/// a user-mandated device execution at this module's legacy `Option` boundary.
/// `gam-linalg` uses `None` to continue on the CPU, so returning it under
/// `gpu=required` would silently violate the configured execution contract.
#[inline]
#[track_caller]
fn decline_gpu<T>(operation: &'static str, reason: &'static str) -> Option<T> {
    if super::global_policy() == GpuPolicy::Required {
        // SAFETY: this legacy `Option` hook has no typed error channel and
        // `None` explicitly authorizes CPU execution, which Required forbids.
        panic!("gpu=required operation '{operation}' cannot execute on the GPU: {reason}");
    }
    None
}

/// Invalid inputs are caller faults, not evidence that an optional accelerator
/// is absent or unprofitable. They must fail independently of GPU policy so an
/// `Option` hook cannot hide them by continuing in a different implementation.
#[inline]
#[track_caller]
fn invalid_gpu_request(operation: &'static str, reason: &'static str) -> ! {
    // SAFETY: invalid dimensions are caller-contract violations, not an
    // optional accelerator decline that may be represented by `None`.
    panic!("GPU operation '{operation}' received invalid input: {reason}");
}

/// A malformed device result is an execution fault, never an Auto decline.
#[inline]
#[track_caller]
fn invalid_gpu_result(operation: &'static str, reason: &'static str) -> ! {
    // SAFETY: a malformed result after device execution cannot be represented
    // by this hook's `Option` without authorizing a misleading CPU retry.
    panic!("GPU operation '{operation}' produced invalid output: {reason}");
}

/// Complete a CUDA attempt after policy admission. Backend `Option` APIs use
/// `None` for an execution failure, but absence and profitability were already
/// settled before the attempt. A post-admission `None` is therefore a fault
/// under every policy and must not be laundered into a CPU fallback.
#[inline]
#[track_caller]
fn complete_gpu_attempt<T>(operation: &'static str, result: Option<T>) -> T {
    match result {
        Some(value) => value,
        // SAFETY: runtime availability and policy admission already succeeded;
        // backend `None` is an execution fault, while this hook's `None` means
        // an ordinary pre-admission decline and would silently retry on CPU.
        None => panic!(
            "GPU operation '{operation}' failed after admission under policy '{}'",
            super::global_policy()
        ),
    }
}

impl DispatchOp {
    /// Conservative flop estimate used for the generic `gemm_min_flops` gate.
    #[inline]
    pub const fn flops(self) -> u128 {
        match self {
            Self::Gemm { m, n, k } => 2u128 * (m as u128) * (n as u128) * (k as u128),
            Self::BatchedGemm { batch, m, n, k } => {
                2u128 * (batch as u128) * (m as u128) * (n as u128) * (k as u128)
            }
            Self::Gemv { m, k } => 2u128 * (m as u128) * (k as u128),
            Self::Potrf { p, batch } => (batch as u128) * (p as u128).pow(3) / 3,
            Self::SmallDenseBatchedPotrf { p, batch } => (batch as u128) * (p as u128).pow(3) / 3,
            Self::Trsm { m, n } => (m as u128) * (m as u128) * (n as u128),
            Self::XtDiagX { n, p } => 2u128 * (n as u128) * (p as u128) * (p as u128),
            Self::XtDiagY { n, px, q } => 2u128 * (n as u128) * (px as u128) * (q as u128),
            Self::JointHessian2x2 { n, pa, pb } => {
                let total = (pa as u128) + (pb as u128);
                2u128 * (n as u128) * total * total
            }
        }
    }

    /// True when SOME reachable Auto dispatch policy could admit this op.
    ///
    /// Pre-probe Auto size gate (the CUDA startup-tax ordering fix): evaluated with
    /// the MOST PERMISSIVE values any production policy can carry — the
    /// calibration crossover floors ([`GpuDispatchPolicy::MIN_CALIBRATABLE_GEMM_FLOPS`],
    /// [`GpuDispatchPolicy::MIN_CALIBRATABLE_POTRF_P`]) for the calibrated
    /// fields, and the [`GpuDispatchPolicy::default`] values for the
    /// small-dense-batched-POTRF fields, which `calibration::calibrate_device`
    /// never adjusts. A `false` here means EVERY reachable policy's
    /// [`route_through_gpu`] admission would also refuse, so the caller may
    /// return to the CPU path WITHOUT resolving GPU availability — i.e.
    /// without triggering the device probe and its per-GPU
    /// `cuDevicePrimaryCtxRetain` context creation. A `true` decides nothing:
    /// the probed runtime's real policy still gates the op exactly as before.
    #[must_use]
    pub fn admissible_under_any_policy(self) -> bool {
        let seed = GpuDispatchPolicy::default();
        let min_gemm = GpuDispatchPolicy::MIN_CALIBRATABLE_GEMM_FLOPS;
        match self {
            Self::Gemm { m, n, k } => self.flops() >= min_gemm && m.min(n).min(k) > 0,
            Self::BatchedGemm { batch, m, n, k } => {
                self.flops() >= min_gemm && batch > 1 && m.min(n).min(k) > 0
            }
            Self::Gemv { m, k } => self.flops() >= min_gemm && m > 0 && k > 0,
            Self::Potrf { p, batch } => {
                p > 0
                    && batch > 0
                    && (p >= GpuDispatchPolicy::MIN_CALIBRATABLE_POTRF_P
                        || (batch > 1 && self.flops() >= min_gemm))
            }
            // These two policy fields are never calibrated, so the default seed
            // values ARE the runtime values and this arm is exact, not a bound.
            Self::SmallDenseBatchedPotrf { p, batch } => {
                p > 0
                    && p <= seed.small_dense_batched_potrf_max_p
                    && batch >= seed.small_dense_batched_potrf_min_batch
            }
            Self::Trsm { m, n } => self.flops() >= min_gemm && m > 0 && n > 0,
            // XtDiagX/XtDiagY admit on `dense_reduction_flops_min` =
            // min(xtwx_flops_min, gemm_min_flops); the smallest reachable value
            // of that min is MIN_CALIBRATABLE_GEMM_FLOPS (the xtwx crossover
            // cannot calibrate below its smallest measurement, which exceeds it).
            Self::XtDiagX { n, p } => n > 0 && p > 0 && self.flops() >= min_gemm,
            Self::XtDiagY { n, px, q } => n > 0 && px > 0 && q > 0 && self.flops() >= min_gemm,
            Self::JointHessian2x2 { n, pa, pb } => {
                n > 0 && (pa > 0 || pb > 0) && self.flops() >= min_gemm
            }
        }
    }
}

/// Returns `Some(runtime)` when a device is available and policy admits the
/// operation. Auto applies calibrated profitability thresholds; Required
/// deliberately bypasses those thresholds because a CPU continuation would
/// violate the requested execution policy.
#[inline]
#[must_use]
pub fn route_through_gpu(op: DispatchOp) -> Option<&'static GpuRuntime> {
    route_through_gpu_with_policy(op, super::global_policy())
}

/// Per-request counterpart of [`route_through_gpu`]. This is the device seam
/// for solvers whose policy is part of an immutable fit request rather than the
/// legacy process-wide configuration.
#[inline]
#[must_use]
pub fn route_through_gpu_with_policy(
    op: DispatchOp,
    selected_policy: GpuPolicy,
) -> Option<&'static GpuRuntime> {
    // Size gate BEFORE the device probe (startup-tax ordering fix): an op no
    // reachable Auto policy could admit must not resolve availability — the first
    // call creates a CUDA primary context on every GPU. Ops that
    // clear this most-permissive bound fall through to the probed runtime's
    // real policy admission below, bit-for-bit as before. Required bypasses
    // this profitability-only gate and resolves the mandatory runtime.
    if selected_policy != GpuPolicy::Required && !op.admissible_under_any_policy() {
        return None;
    }
    let runtime = runtime_for_dispatch(selected_policy)?;
    if selected_policy == GpuPolicy::Required {
        return Some(runtime);
    }
    let policy = &runtime.policy;
    let admit = match op {
        DispatchOp::Gemm { m, n, k } => {
            op.flops() >= (policy.gemm_min_flops as u128) && m.min(n).min(k) > 0
        }
        DispatchOp::BatchedGemm { batch, m, n, k } => {
            op.flops() >= (policy.gemm_min_flops as u128) && batch > 1 && m.min(n).min(k) > 0
        }
        DispatchOp::Gemv { m, k } => {
            op.flops() >= (policy.gemm_min_flops as u128) && m > 0 && k > 0
        }
        DispatchOp::Potrf { p, batch } => {
            p > 0
                && batch > 0
                && (p >= policy.potrf_min_p
                    || (batch > 1 && op.flops() >= policy.gemm_min_flops as u128))
        }
        DispatchOp::SmallDenseBatchedPotrf { p, batch } => {
            p > 0
                && p <= policy.small_dense_batched_potrf_max_p
                && batch >= policy.small_dense_batched_potrf_min_batch
        }
        DispatchOp::Trsm { m, n } => {
            op.flops() >= (policy.gemm_min_flops as u128) && m > 0 && n > 0
        }
        DispatchOp::XtDiagX { n, p } => policy.xtwx_target_is_gpu(n, p, true),
        DispatchOp::XtDiagY { n, px, q } => policy.xtwy_target_is_gpu(n, px, q, true),
        DispatchOp::JointHessian2x2 { n, pa, pb } => {
            n > 0 && (pa > 0 || pb > 0) && op.flops() >= policy.gemm_min_flops as u128
        }
    };
    if admit { Some(runtime) } else { None }
}

/// Minimum batch size before a batched kernel is worth splitting across more
/// than one device. Below this the per-tile launch + extra H2D/D2H staging on a
/// second device costs more than the GEMM time it saves, so a small batch stays
/// on the single primary device. This is a fixed, conservatively-large constant
/// (magic-by-default; no flag) — multi-GPU only kicks in for genuinely large
/// batches such as large-scale Arrow-Schur / Stage-3 blocks.
#[cfg(target_os = "linux")]
const MULTI_GPU_BATCH_FLOOR: usize = 64;

/// True when the pool has >1 usable device and `batch` is large enough that
/// splitting the batch dimension across devices is worthwhile.
#[cfg(target_os = "linux")]
#[inline]
fn should_split_batch(batch: usize) -> bool {
    let policy = super::global_policy();
    runtime_for_dispatch(policy).is_some_and(|rt| rt.device_count() > 1)
        && batch >= MULTI_GPU_BATCH_FLOOR
}

#[inline]
#[must_use]
pub fn try_fast_ab_broadcast_b_batched(
    a: ArrayView3<'_, f64>,
    b: ArrayView2<'_, f64>,
) -> Option<Array3<f64>> {
    let (batch, m, k) = a.dim();
    let (bk, n) = b.dim();
    if k != bk {
        invalid_gpu_request("batched A·B", "the reduction dimensions differ");
    }
    if batch == 0 || m == 0 || n == 0 || k == 0 {
        return decline_gpu(
            "batched A·B",
            "the workload has an empty dimension",
        );
    }
    #[cfg(not(target_os = "linux"))]
    {
        return decline_gpu("batched A·B", "the CUDA backend is not compiled on this platform");
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::BatchedGemm { batch, m, n, k })?;
        if should_split_batch(batch) {
            if let Some(out) = scatter_broadcast_b_batched(runtime, a, b, m, n) {
                return Some(out);
            }
            // A multi-GPU tile failed; fall through to the single-device path so
            // the whole batch is still produced on the primary device.
        }
        Some(complete_gpu_attempt(
            "batched A·B",
            cuda_backend::gemm_broadcast_b_batched(runtime.device.ordinal, a, b),
        ))
    }
}

/// Multi-GPU broadcast-B batched GEMM: split the batch dimension across all
/// devices via [`scatter_batched`], running one cuBLAS strided-batched GEMM per
/// device tile (each on its own bound ordinal). `b` is shared (broadcast) across
/// every tile. Returns `None` if any tile fails so the caller falls back to the
/// single-device path.
#[cfg(target_os = "linux")]
fn scatter_broadcast_b_batched(
    runtime: &GpuRuntime,
    a: ArrayView3<'_, f64>,
    b: ArrayView2<'_, f64>,
    m: usize,
    n: usize,
) -> Option<Array3<f64>> {
    let batch = a.dim().0;
    // One slot per batch item; the slot carries its own input matrix so the
    // per-tile closure is range-agnostic and owns disjoint memory.
    let mut items: Vec<(Array2<f64>, Option<Array2<f64>>)> = (0..batch)
        .map(|i| (a.index_axis(ndarray::Axis(0), i).to_owned(), None))
        .collect();
    super::pool::scatter_batched(runtime, &mut items, |ordinal, tile| {
        let tile_batch = tile.len();
        if tile_batch == 0 {
            return Some(());
        }
        let k = b.dim().0;
        let mut a_tile = Array3::<f64>::zeros((tile_batch, m, k));
        for (idx, (a_i, _)) in tile.iter().enumerate() {
            a_tile.index_axis_mut(ndarray::Axis(0), idx).assign(a_i);
        }
        let out = cuda_backend::gemm_broadcast_b_batched(ordinal, a_tile.view(), b)?;
        for (idx, (_, slot)) in tile.iter_mut().enumerate() {
            *slot = Some(out.index_axis(ndarray::Axis(0), idx).to_owned());
        }
        Some(())
    })?;
    stitch_batched(items, m, n)
}

#[inline]
#[must_use]
pub fn try_fast_abt_strided_batched(
    a: ArrayView3<'_, f64>,
    b: ArrayView3<'_, f64>,
) -> Option<Array3<f64>> {
    try_fast_abt_strided_batched_with_policy(a, b, super::global_policy())
}

#[inline]
#[must_use]
pub fn try_fast_abt_strided_batched_with_policy(
    a: ArrayView3<'_, f64>,
    b: ArrayView3<'_, f64>,
    gpu_policy: GpuPolicy,
) -> Option<Array3<f64>> {
    let (batch, m, k) = a.dim();
    let (batch_b, n, k_b) = b.dim();
    if batch != batch_b || k != k_b {
        invalid_gpu_request("batched A·Bᵀ", "the batch or reduction dimensions differ");
    }
    if batch == 0 || m == 0 || n == 0 || k == 0 {
        return decline_gpu(
            "batched A·Bᵀ",
            "the workload has an empty dimension",
        );
    }
    #[cfg(not(target_os = "linux"))]
    {
        return decline_gpu(
            "batched A·Bᵀ",
            "the CUDA backend is not compiled on this platform",
        );
    }
    #[cfg(target_os = "linux")]
    {
        let runtime =
            route_through_gpu_with_policy(DispatchOp::BatchedGemm { batch, m, n, k }, gpu_policy)?;
        if should_split_batch(batch) {
            if let Some(out) = scatter_abt_strided_batched(runtime, a, b, m, n) {
                return Some(out);
            }
        }
        Some(complete_gpu_attempt(
            "batched A·Bᵀ",
            cuda_backend::gemm_abt_strided_batched(runtime.device.ordinal, a, b),
        ))
    }
}

/// Multi-GPU A·Bᵀ strided-batched GEMM: split the batch dimension across all
/// devices, running one strided-batched GEMM per device tile. Both `a` and `b`
/// are batched (one matrix per batch item), so each slot carries its own
/// `(a_i, b_i)` pair. Returns `None` on any tile failure.
#[cfg(target_os = "linux")]
fn scatter_abt_strided_batched(
    runtime: &GpuRuntime,
    a: ArrayView3<'_, f64>,
    b: ArrayView3<'_, f64>,
    m: usize,
    n: usize,
) -> Option<Array3<f64>> {
    let batch = a.dim().0;
    let mut items: Vec<(Array2<f64>, Array2<f64>, Option<Array2<f64>>)> = (0..batch)
        .map(|i| {
            (
                a.index_axis(ndarray::Axis(0), i).to_owned(),
                b.index_axis(ndarray::Axis(0), i).to_owned(),
                None,
            )
        })
        .collect();
    super::pool::scatter_batched(runtime, &mut items, |ordinal, tile| {
        let tile_batch = tile.len();
        if tile_batch == 0 {
            return Some(());
        }
        let k = tile[0].0.dim().1;
        let mut a_tile = Array3::<f64>::zeros((tile_batch, m, k));
        let mut b_tile = Array3::<f64>::zeros((tile_batch, n, k));
        for (idx, (a_i, b_i, _)) in tile.iter().enumerate() {
            a_tile.index_axis_mut(ndarray::Axis(0), idx).assign(a_i);
            b_tile.index_axis_mut(ndarray::Axis(0), idx).assign(b_i);
        }
        let out = cuda_backend::gemm_abt_strided_batched(ordinal, a_tile.view(), b_tile.view())?;
        for (idx, (_, _, slot)) in tile.iter_mut().enumerate() {
            *slot = Some(out.index_axis(ndarray::Axis(0), idx).to_owned());
        }
        Some(())
    })?;
    let slots: Vec<((), Option<Array2<f64>>)> =
        items.into_iter().map(|(_, _, slot)| ((), slot)).collect();
    stitch_batched(slots, m, n)
}

/// Reassemble per-batch output slots (filled by the device tiles) into a single
/// `batch × m × n` array. Returns `None` if any slot is still empty (a tile
/// silently skipped its item), which forces the single-device fallback.
#[cfg(target_os = "linux")]
fn stitch_batched<L>(
    items: Vec<(L, Option<Array2<f64>>)>,
    m: usize,
    n: usize,
) -> Option<Array3<f64>> {
    let batch = items.len();
    let mut out = Array3::<f64>::zeros((batch, m, n));
    for (idx, (_, slot)) in items.into_iter().enumerate() {
        let block = slot?;
        if block.dim() != (m, n) {
            return None;
        }
        out.index_axis_mut(ndarray::Axis(0), idx).assign(&block);
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Dispatch entry points. Each takes views to keep the call site allocation-
// free and returns Some(result) iff the GPU actually produced one. Under Auto,
// the CPU fast path resumes only on a pre-admission None (typed absence or an
// unprofitable workload). Every post-admission failure is fatal at this legacy
// Option boundary; Required additionally makes pre-admission declines fatal.
//
// CUDA kernels are compiled into the runtime through cudarc's dynamic loader.
// Auto admits only profitable workloads. Required bypasses profitability gates
// and fails if no CUDA runtime path is available. An admitted backend failure
// is an execution fault under every policy and never falls through to the CPU.
// ---------------------------------------------------------------------------

#[inline]
#[must_use]
pub fn try_fast_ab(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
    let (m, k) = a.dim();
    let (kb, n) = b.dim();
    if k != kb {
        invalid_gpu_request("A·B", "the reduction dimensions differ");
    }
    if m == 0 || n == 0 || k == 0 {
        return decline_gpu("A·B", "the workload has an empty dimension");
    }
    // Record every dispatch attempt — including ones that fall back to CPU
    // because either the runtime is unavailable or the workload is below
    // policy threshold. The diagnostics snapshot is what downstream telemetry
    // uses to attribute CPU vs GPU time, so it must reflect *attempts*, not
    // just successful device launches.
    let runtime = route_through_gpu(DispatchOp::Gemm { m, n, k });
    let used_gpu = runtime.is_some();
    super::profile::record(super::profile::KernelStat {
        name: "try_fast_ab",
        n: m,
        p: n,
        k,
        flops_est: (DispatchOp::Gemm { m, n, k }.flops().min(usize::MAX as u128)) as usize,
        gpu_ms: if used_gpu { Some(0.0) } else { None },
        ..Default::default()
    });
    #[cfg(not(target_os = "linux"))]
    {
        decline_gpu("A·B", "the CUDA backend is not compiled on this platform")
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = runtime?;
        Some(complete_gpu_attempt(
            "A·B",
            cuda_backend::gemm(runtime, a, b, false, false),
        ))
    }
}

#[inline]
#[must_use]
pub fn try_fast_atb(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
    let (n_a, p) = a.dim();
    let (n_b, q) = b.dim();
    if n_a != n_b {
        invalid_gpu_request("Aᵀ·B", "the row dimensions differ");
    }
    if n_a == 0 || p == 0 || q == 0 {
        return decline_gpu("Aᵀ·B", "the workload has an empty dimension");
    }
    #[cfg(not(target_os = "linux"))]
    {
        return decline_gpu("Aᵀ·B", "the CUDA backend is not compiled on this platform");
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::Gemm { m: p, n: q, k: n_a })?;
        Some(complete_gpu_attempt(
            "Aᵀ·B",
            cuda_backend::gemm(runtime, a, b, true, false),
        ))
    }
}

/// `Aᵀ·B` on a specific device ordinal, for pool-tiled callers that already own
/// the ordinal (the worker thread has bound that ordinal's context). Semantics
/// are identical to [`try_fast_atb`] — `a` is `m×k`, `b` is `m×n`, output is the
/// `k×n` product `aᵀ·b` — but the kernel is pinned to `ordinal` instead of the
/// probe-selected primary device. Auto returns `None` only when CUDA is absent
/// or the shape is below policy threshold, so the caller can run its CPU path.
/// A post-admission backend failure is fatal under every policy; Required also
/// makes pre-admission absence fatal. f64 only.
#[inline]
#[must_use]
pub fn try_fast_atb_on_ordinal(
    ordinal: usize,
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
) -> Option<Array2<f64>> {
    let (n_a, p) = a.dim();
    let (n_b, q) = b.dim();
    if n_a != n_b {
        invalid_gpu_request("ordinal-pinned Aᵀ·B", "the row dimensions differ");
    }
    if n_a == 0 || p == 0 || q == 0 {
        return decline_gpu(
            "ordinal-pinned Aᵀ·B",
            "the workload has an empty dimension",
        );
    }
    #[cfg(not(target_os = "linux"))]
    {
        // No CUDA off Linux, so the per-ordinal fast path is unavailable. Read
        // `ordinal` once (the cross-platform signature must carry it for the
        // Linux branch below) and decline so the caller runs its CPU AtB. Unlike
        // `a`/`b` — already consumed by `.dim()` above — `ordinal` is otherwise
        // untouched on this target, and `warnings = "deny"` rejects a dead bind.
        log::trace!(
            "try_fast_atb_on_ordinal: CUDA unavailable off Linux; declining ordinal {ordinal}"
        );
        return decline_gpu(
            "ordinal-pinned Aᵀ·B",
            "the CUDA backend is not compiled on this platform",
        );
    }
    #[cfg(target_os = "linux")]
    {
        // The size/policy gate is identical to `try_fast_atb`; only the target
        // device differs. We still consult `route_through_gpu` so a below-floor
        // shape declines to the caller's CPU path rather than paying PCIe cost.
        //
        // Arrow-Schur's `tile_schur_partial` reaches this gate after stacking
        // its per-row factors into one transpose tile GEMM:
        // `(total_d x k)^T * (total_d x k)`.
        // At the SAE shape n=2000, p=2048, M=12, K=8, that is
        // 2*(n*M)*p^2 = 201_326_592_000 flops for one stacked tile, or
        // 1_610_612_736_000 flops across K=8 batches, so admission must be
        // keyed on work rather than the observation row count.
        route_through_gpu(DispatchOp::Gemm { m: p, n: q, k: n_a })?;
        Some(complete_gpu_attempt(
            "ordinal-pinned Aᵀ·B",
            cuda_backend::gemm_on_ordinal(ordinal, a, b, true, false),
        ))
    }
}

#[inline]
#[must_use]
pub fn try_fast_av(a: ArrayView2<'_, f64>, v: ArrayView1<'_, f64>) -> Option<Array1<f64>> {
    let (m, k) = a.dim();
    if k != v.len() {
        invalid_gpu_request("A·v", "the matrix width and vector length differ");
    }
    if m == 0 || k == 0 {
        return decline_gpu("A·v", "the workload has an empty dimension");
    }
    #[cfg(not(target_os = "linux"))]
    {
        return decline_gpu("A·v", "the CUDA backend is not compiled on this platform");
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::Gemv { m, k })?;
        Some(complete_gpu_attempt(
            "A·v",
            cuda_backend::gemv(runtime, a, v, false),
        ))
    }
}

#[inline]
#[must_use]
pub fn try_fast_atv(a: ArrayView2<'_, f64>, v: ArrayView1<'_, f64>) -> Option<Array1<f64>> {
    let (n, p) = a.dim();
    if n != v.len() {
        invalid_gpu_request("Aᵀ·v", "the matrix height and vector length differ");
    }
    if n == 0 || p == 0 {
        return decline_gpu("Aᵀ·v", "the workload has an empty dimension");
    }
    #[cfg(not(target_os = "linux"))]
    {
        return decline_gpu("Aᵀ·v", "the CUDA backend is not compiled on this platform");
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::Gemv { m: p, k: n })?;
        Some(complete_gpu_attempt(
            "Aᵀ·v",
            cuda_backend::gemv(runtime, a, v, true),
        ))
    }
}

#[inline]
#[must_use]
pub fn try_fast_xt_diag_x(x: ArrayView2<'_, f64>, w: ArrayView1<'_, f64>) -> Option<Array2<f64>> {
    let (n, p) = x.dim();
    if n != w.len() {
        invalid_gpu_request("Xᵀ·diag(w)·X", "the row and weight counts differ");
    }
    if n == 0 || p == 0 {
        return decline_gpu("Xᵀ·diag(w)·X", "the workload has an empty dimension");
    }
    #[cfg(not(target_os = "linux"))]
    {
        return decline_gpu(
            "Xᵀ·diag(w)·X",
            "the CUDA backend is not compiled on this platform",
        );
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::XtDiagX { n, p })?;
        Some(complete_gpu_attempt(
            "Xᵀ·diag(w)·X",
            cuda_backend::xt_diag_x(runtime, x, w),
        ))
    }
}

/// #1017 Phase 3: a device-resident design matrix for repeated `Xᵀ·diag(w)·X`
/// Gram evaluations that uploads `X` to the device ONCE.
///
/// The per-call [`try_fast_xt_diag_x`] re-uploads the full `n×p` `X` on every
/// call. The SAE / IRLS inner loop holds `X` fixed and rebuilds the Gram once
/// per Newton/PIRLS weight update, so the repeated H2D of `X` is pure waste —
/// measured on an A100 (#1412) it makes the `XtWX` GEMM ~98% of the pipeline at
/// <20% device utilisation (the device is starved by staging, not arithmetic).
/// This handle uploads `X` once at construction; each [`Self::gram`] crosses
/// only the `n`-vector `w` H2D and the `p×p` Gram D2H, so the per-Gram transfer
/// shrinks by a factor of `p`.
///
/// Admission keys on the same work-based [`DispatchOp::XtDiagX`] gate as the
/// per-call path (so it engages exactly when the Gram is GPU-profitable) and the
/// numerics are bit-identical to [`try_fast_xt_diag_x`] on the same device
/// (same `cublasDdgmm` row-scale + `gemm` reduction order). On a non-CUDA host
/// or a below-threshold shape, Auto makes [`Self::try_new`] return `None` and
/// the caller keeps its CPU/per-call path; Required fails instead. Once
/// admitted, upload or execution failures are fatal under every policy —
/// residency never changes the result, only where (and how often) `X` is staged.
pub struct ResidentDesignGram {
    #[cfg(target_os = "linux")]
    inner: super::blas::ResidentWeightedGram,
    #[cfg(not(target_os = "linux"))]
    _never: std::convert::Infallible,
}

impl ResidentDesignGram {
    /// Upload `x` (`n×p`) to the device once. Auto returns `None` when CUDA is
    /// unavailable or the shape is below the GPU Gram threshold; Required
    /// fails loudly. An admitted upload failure is fatal under every policy.
    #[must_use]
    pub fn try_new(x: ArrayView2<'_, f64>) -> Option<Self> {
        let (n, p) = x.dim();
        if n == 0 || p == 0 {
            return decline_gpu("resident weighted Gram upload", "the design matrix is empty");
        }
        #[cfg(not(target_os = "linux"))]
        {
            decline_gpu(
                "resident weighted Gram upload",
                "the CUDA backend is not compiled on this platform",
            )
        }
        #[cfg(target_os = "linux")]
        {
            let runtime = route_through_gpu(DispatchOp::XtDiagX { n, p })?;
            let inner = complete_gpu_attempt(
                "resident weighted Gram upload",
                super::blas::ResidentWeightedGram::new(runtime.device.ordinal, x),
            );
            Some(Self { inner })
        }
    }

    /// Compute `Xᵀ·diag(w)·X` reusing the resident `X`. `w` must have one entry
    /// per design row. Shape mismatches and device failures after construction
    /// are fatal rather than being converted into a CPU continuation.
    #[must_use]
    pub fn gram(&self, w: ArrayView1<'_, f64>) -> Option<Array2<f64>> {
        #[cfg(not(target_os = "linux"))]
        {
            // SAFETY: off CUDA, `try_new` never constructs `Self`, so this
            // method is statically unreachable. Returning a benign `None` would
            // silently launder that impossibility into a "GPU declined"
            // sentinel, so fail loudly. The `w.len()` use also consumes the
            // parameter on this target.
            panic!(
                "ResidentDesignGram cannot be constructed off CUDA (w.len()={})",
                w.len()
            )
        }
        #[cfg(target_os = "linux")]
        {
            Some(complete_gpu_attempt(
                "resident Xᵀ·diag(w)·X",
                self.inner.gram(w),
            ))
        }
    }

    /// Solve the penalized normal equations `(Xᵀ·diag(w)·X + ridge·I)·β = rhs`
    /// with the Gram, its Cholesky factor, and the RHS all kept DEVICE-RESIDENT —
    /// only `w` (`n`), `rhs` (`p`), and the solution `β` (`p`) cross the bus.
    ///
    /// This is the #1017 Phase-3 fix for the next ceiling after [`Self::gram`]:
    /// the bare Gram still pays a `p×p` D2H (134 MB at p=4096), but the SAE/IRLS
    /// inner step only needs `β`, so chaining row-scale→GEMM→POTRF→TRSM on-device
    /// and returning only the `p`-vector removes that transfer entirely. A
    /// shape mismatch, non-PD Gram, or device failure after construction is
    /// fatal rather than being converted into a CPU continuation. The numerics
    /// match a host `Cholesky((XᵀWX+ridge·I))` solve up to IEEE-754 reduction
    /// order.
    #[must_use]
    pub fn solve_normal_equations(
        &self,
        w: ArrayView1<'_, f64>,
        rhs: ArrayView1<'_, f64>,
        ridge: f64,
    ) -> Option<Array1<f64>> {
        #[cfg(not(target_os = "linux"))]
        {
            // SAFETY: statically unreachable off CUDA (see `gram`); fail loudly.
            panic!(
                "ResidentDesignGram cannot be constructed off CUDA (w.len()={}, rhs.len()={}, ridge={ridge})",
                w.len(),
                rhs.len()
            )
        }
        #[cfg(target_os = "linux")]
        {
            Some(complete_gpu_attempt(
                "resident normal-equations solve",
                self.inner.solve_psd_normal_equations(w, rhs, ridge),
            ))
        }
    }

    /// `(n, p)` of the resident design.
    #[must_use]
    pub fn dims(&self) -> (usize, usize) {
        #[cfg(not(target_os = "linux"))]
        {
            // SAFETY: statically unreachable off CUDA (see `gram`) — no `Self`
            // is ever constructed on this target; fail loudly rather than
            // return a benign sentinel.
            panic!("ResidentDesignGram cannot be constructed off CUDA")
        }
        #[cfg(target_os = "linux")]
        {
            self.inner.dims()
        }
    }
}

/// Number of row-chunks to carve per device for the spectral leverage stream
/// so [`super::pool::balanced_partition`] can keep every GPU busy. With fewer
/// chunks than devices the pool would idle the surplus devices; oversubscribing
/// modestly amortizes the per-tile launch without bloating staging memory.
/// Magic-by-default; no flag.
#[cfg(target_os = "linux")]
const LEVERAGE_CHUNKS_PER_DEVICE: usize = 4;

/// Byte-balanced row-chunk width for the spectral leverage stream, mirroring
/// the CPU `byte_balanced_row_chunk` sizing (≈8 MiB live blocks) so a single
/// tile's `(chunk × p)` row slice plus `(chunk × rank)` GEMM output stay within
/// the per-device staging budget.
#[cfg(target_os = "linux")]
#[inline]
fn leverage_chunk_rows(cols: usize, n_rows: usize) -> usize {
    const TARGET_BYTES: usize = 8 * 1024 * 1024;
    const MIN_CHUNK_ROWS: usize = 512;
    let bytes_per_row = cols.max(1) * std::mem::size_of::<f64>();
    (TARGET_BYTES / bytes_per_row)
        .max(MIN_CHUNK_ROWS)
        .min(n_rows.max(1))
}

/// GPU-offloaded spectral leverage diagonal `h[i] = ‖(X G)_{i,:}‖²`.
///
/// `G` is the `(p × rank)` spectral factor with `G_ε(H) = G Gᵀ`; the per-row
/// leverage is the squared norm of the i-th row of `X G`. This is the dominant
/// n-dependent cost of every REML outer evaluation at large scale (issue
/// #922), and historically ran only on the CPU while the device pool idled.
///
/// The row dimension is split into byte-balanced chunks scattered across the
/// whole device pool via [`super::pool::scatter_batched`] — the same
/// whole-solve row-block granularity as Arrow-Schur — and each tile runs one
/// cuBLAS GEMM `X_chunk · G` on its bound ordinal before reducing row-wise
/// sum-of-squares. The arithmetic is identical f64 to the CPU faer path (modulo
/// IEEE-754 reduction order). On no device or a below-threshold shape, Auto
/// returns `None` and the caller runs its deterministic CPU stream; Required
/// fails at the dispatch boundary. A tile failure after admission is fatal
/// under every policy.
#[inline]
#[must_use]
pub fn try_fast_spectral_leverage_diagonal(
    x: &gam_linalg::matrix::DesignMatrix,
    g: ArrayView2<'_, f64>,
) -> Option<Array1<f64>> {
    let n = x.nrows();
    let p = x.ncols();
    let rank = g.ncols();
    if g.nrows() != p {
        invalid_gpu_request(
            "spectral leverage diagonal",
            "the design width and spectral-factor height differ",
        );
    }
    if n == 0 || p == 0 || rank == 0 {
        return decline_gpu(
            "spectral leverage diagonal",
            "the workload has an empty dimension",
        );
    }
    #[cfg(not(target_os = "linux"))]
    {
        return decline_gpu(
            "spectral leverage diagonal",
            "the CUDA backend is not compiled on this platform",
        );
    }
    #[cfg(target_os = "linux")]
    {
        // n·p² gate is shared with the X^T diag(w) X reduction — the leverage
        // diagonal is the same O(n·p·rank)-class dense pass over the design.
        let runtime = route_through_gpu(DispatchOp::XtDiagX { n, p })?;
        let device_count = runtime.device_count().max(1);
        let byte_chunk = leverage_chunk_rows(p + rank, n);
        let target_chunks = device_count
            .saturating_mul(LEVERAGE_CHUNKS_PER_DEVICE)
            .max(1);
        let chunk_rows = byte_chunk.min(n.div_ceil(target_chunks).max(1)).max(1);

        // One slot per row-chunk; the slot carries its row range and receives
        // its own output buffer so each tile owns disjoint memory.
        let mut tiles: Vec<(std::ops::Range<usize>, Option<Array1<f64>>)> = Vec::new();
        let mut start = 0usize;
        while start < n {
            let end = (start + chunk_rows).min(n);
            tiles.push((start..end, None));
            start = end;
        }

        complete_gpu_attempt(
            "spectral leverage diagonal scatter",
            super::pool::scatter_batched(runtime, &mut tiles, |ordinal, tile| {
                for (range, slot) in tile.iter_mut() {
                    let rows = x.try_row_chunk(range.clone()).ok()?;
                    let xg =
                        cuda_backend::gemm_on_ordinal(ordinal, rows.view(), g, false, false)?;
                    let mut out = Array1::<f64>::zeros(range.end - range.start);
                    for (local, row) in xg.outer_iter().enumerate() {
                        out[local] = row.iter().map(|&v| v * v).sum();
                    }
                    *slot = Some(out);
                }
                Some(())
            }),
        );

        let mut h = Array1::<f64>::zeros(n);
        for (range, slot) in tiles {
            let vals = complete_gpu_attempt("spectral leverage diagonal stitch", slot);
            if vals.len() != range.end - range.start {
                invalid_gpu_result(
                    "spectral leverage diagonal stitch",
                    "a device tile produced an invalid row count",
                );
            }
            h.slice_mut(ndarray::s![range]).assign(&vals);
        }
        Some(h)
    }
}

#[inline]
#[must_use]
pub fn try_fast_xt_diag_y(
    x: ArrayView2<'_, f64>,
    w: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
) -> Option<Array2<f64>> {
    let (n, px) = x.dim();
    let (n_y, q) = y.dim();
    if n != n_y || n != w.len() {
        invalid_gpu_request("Xᵀ·diag(w)·Y", "the row or weight counts differ");
    }
    if n == 0 || px == 0 || q == 0 {
        return decline_gpu(
            "Xᵀ·diag(w)·Y",
            "the workload has an empty dimension",
        );
    }
    #[cfg(not(target_os = "linux"))]
    {
        return decline_gpu(
            "Xᵀ·diag(w)·Y",
            "the CUDA backend is not compiled on this platform",
        );
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::XtDiagY { n, px, q })?;
        Some(complete_gpu_attempt(
            "Xᵀ·diag(w)·Y",
            cuda_backend::xt_diag_y(runtime, x, w, y),
        ))
    }
}

#[inline]
#[must_use]
pub fn try_fast_joint_hessian_2x2(
    x_a: ArrayView2<'_, f64>,
    x_b: ArrayView2<'_, f64>,
    w_aa: ArrayView1<'_, f64>,
    w_ab: ArrayView1<'_, f64>,
    w_bb: ArrayView1<'_, f64>,
) -> Option<Array2<f64>> {
    let (n, pa) = x_a.dim();
    let (n_b, pb) = x_b.dim();
    if n != n_b || n != w_aa.len() || n != w_ab.len() || n != w_bb.len() {
        invalid_gpu_request("joint 2×2 Hessian", "the row or weight counts differ");
    }
    if n == 0 || (pa == 0 && pb == 0) {
        return decline_gpu(
            "joint 2×2 Hessian",
            "the workload has an empty dimension",
        );
    }
    #[cfg(not(target_os = "linux"))]
    {
        return decline_gpu(
            "joint 2×2 Hessian",
            "the CUDA backend is not compiled on this platform",
        );
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::JointHessian2x2 { n, pa, pb })?;
        Some(complete_gpu_attempt(
            "joint 2×2 Hessian",
            cuda_backend::joint_hessian_2x2(runtime, x_a, x_b, w_aa, w_ab, w_bb),
        ))
    }
}

#[inline]
#[must_use]
pub fn try_cholesky_lower_inplace(a: &mut Array2<f64>) -> Option<()> {
    let p = a.nrows();
    if p != a.ncols() {
        invalid_gpu_request("Cholesky factorization", "the input matrix is non-square");
    }
    if p == 0 {
        return decline_gpu("Cholesky factorization", "the workload has an empty dimension");
    }
    #[cfg(not(target_os = "linux"))]
    {
        return decline_gpu(
            "Cholesky factorization",
            "the CUDA backend is not compiled on this platform",
        );
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::Potrf { p, batch: 1 })?;
        let lower = complete_gpu_attempt(
            "Cholesky factorization",
            cuda_backend::cholesky_lower(runtime, a.view()),
        );
        *a = lower;
        Some(())
    }
}

#[inline]
#[must_use]
pub fn try_cholesky_batched_lower_inplace(matrices: &mut [Array2<f64>]) -> Option<()> {
    try_cholesky_batched_lower_inplace_with_policy(matrices, super::global_policy())
}

#[inline]
#[must_use]
pub fn try_cholesky_batched_lower_inplace_with_policy(
    matrices: &mut [Array2<f64>],
    gpu_policy: GpuPolicy,
) -> Option<()> {
    let first = match matrices.first() {
        Some(first) => first,
        None => return decline_gpu("batched Cholesky factorization", "the batch is empty"),
    };
    let p = first.nrows();
    if first.ncols() != p || matrices.iter().any(|matrix| matrix.dim() != (p, p)) {
        invalid_gpu_request(
            "batched Cholesky factorization",
            "an input matrix is non-square or has a different shape",
        );
    }
    if p == 0 {
        return decline_gpu(
            "batched Cholesky factorization",
            "the workload has an empty dimension",
        );
    }
    #[cfg(not(target_os = "linux"))]
    {
        return decline_gpu(
            "batched Cholesky factorization",
            "the CUDA backend is not compiled on this platform",
        );
    }
    #[cfg(target_os = "linux")]
    {
        let batch = matrices.len();
        let runtime = route_through_gpu_with_policy(
            DispatchOp::SmallDenseBatchedPotrf { p, batch },
            gpu_policy,
        )
        .or_else(|| route_through_gpu_with_policy(DispatchOp::Potrf { p, batch }, gpu_policy))?;
        if should_split_batch(batch) {
            // `matrices` is already the per-item slice, so the batch dimension
            // tiles directly onto `scatter_batched`: each device factors its own
            // contiguous block of matrices in place. On any tile failure the
            // whole batch is re-run on the primary device for determinism (the
            // factored tiles are overwritten by the single-device pass).
            let split = super::pool::scatter_batched(runtime, matrices, |ordinal, tile| {
                cuda_backend::cholesky_batched_lower(ordinal, tile)
            });
            if split.is_some() {
                return Some(());
            }
        }
        Some(complete_gpu_attempt(
            "batched Cholesky factorization",
            cuda_backend::cholesky_batched_lower(runtime.device.ordinal, matrices),
        ))
    }
}

#[inline]
#[must_use]
pub fn try_solve_lower_triangular_matrix(
    lower: ArrayView2<'_, f64>,
    rhs: ArrayView2<'_, f64>,
) -> Option<Array2<f64>> {
    let (m, n) = rhs.dim();
    if lower.dim() != (m, m) {
        invalid_gpu_request(
            "lower-triangular solve",
            "the triangular matrix shape does not match the right-hand side",
        );
    }
    if m == 0 || n == 0 {
        return decline_gpu(
            "lower-triangular solve",
            "the workload has an empty dimension",
        );
    }
    #[cfg(not(target_os = "linux"))]
    {
        return decline_gpu(
            "lower-triangular solve",
            "the CUDA backend is not compiled on this platform",
        );
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::Trsm { m, n })?;
        Some(complete_gpu_attempt(
            "lower-triangular solve",
            cuda_backend::trsm(runtime, lower, rhs, false),
        ))
    }
}

#[inline]
#[must_use]
pub fn try_solve_upper_triangular_matrix(
    upper: ArrayView2<'_, f64>,
    rhs: ArrayView2<'_, f64>,
) -> Option<Array2<f64>> {
    let (m, n) = rhs.dim();
    if upper.dim() != (m, m) {
        invalid_gpu_request(
            "upper-triangular solve",
            "the triangular matrix shape does not match the right-hand side",
        );
    }
    if m == 0 || n == 0 {
        return decline_gpu(
            "upper-triangular solve",
            "the workload has an empty dimension",
        );
    }
    #[cfg(not(target_os = "linux"))]
    {
        return decline_gpu(
            "upper-triangular solve",
            "the CUDA backend is not compiled on this platform",
        );
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::Trsm { m, n })?;
        Some(complete_gpu_attempt(
            "upper-triangular solve",
            cuda_backend::trsm(runtime, upper, rhs, true),
        ))
    }
}

#[cfg(test)]
mod pre_probe_gate_tests {
    //! Pins the CUDA startup-tax ordering fix at the dispatch chokepoint: an op
    //! no reachable policy could admit must be refused by `route_through_gpu`
    //! WITHOUT resolving GPU availability — i.e. without triggering the
    //! device probe and its per-GPU `cuDevicePrimaryCtxRetain` context
    //! creation. Observable on any host (CUDA or not) through the process-wide
    //! `resolution_call_count` counter; nextest gives each test its own process.
    use super::{DispatchOp, GpuDispatchPolicy, route_through_gpu};
    use crate::device_runtime::GpuRuntime;

    #[test]
    fn cpu_sized_ops_are_refused_before_the_device_probe() {
        let tiny_ops = [
            DispatchOp::Gemm { m: 8, n: 8, k: 8 },
            DispatchOp::BatchedGemm {
                batch: 4,
                m: 8,
                n: 8,
                k: 8,
            },
            DispatchOp::Gemv { m: 64, k: 64 },
            DispatchOp::Potrf { p: 24, batch: 1 },
            DispatchOp::Trsm { m: 16, n: 16 },
            DispatchOp::XtDiagX { n: 700, p: 12 },
            DispatchOp::XtDiagY {
                n: 700,
                px: 12,
                q: 4,
            },
            DispatchOp::JointHessian2x2 {
                n: 700,
                pa: 8,
                pb: 8,
            },
        ];
        let before = GpuRuntime::resolution_call_count();
        for op in tiny_ops {
            assert!(
                !op.admissible_under_any_policy(),
                "fixture op must be inadmissible under every policy: {op:?}"
            );
            assert!(
                route_through_gpu(op).is_none(),
                "inadmissible op must not route: {op:?}"
            );
        }
        assert_eq!(
            GpuRuntime::resolution_call_count(),
            before,
            "route_through_gpu must refuse CPU-sized ops BEFORE runtime resolution, \
             so no CUDA context is ever created for them"
        );
    }

    #[test]
    fn admissible_ops_fall_through_to_the_probed_runtime() {
        // An op above every floor must consult the runtime (identical behaviour
        // to the pre-fix path for genuinely GPU-sized work).
        let big = DispatchOp::Gemm {
            m: 2_048,
            n: 2_048,
            k: 2_048,
        };
        assert!(big.admissible_under_any_policy());
        let before = GpuRuntime::resolution_call_count();
        let routed = route_through_gpu(big);
        assert!(
            routed.is_none_or(|runtime| !runtime.devices.is_empty()),
            "a routed operation must receive a runtime with at least one usable device"
        );
        assert!(
            GpuRuntime::resolution_call_count() > before,
            "an admissible op must fall through to runtime resolution"
        );
    }

    #[test]
    fn admissibility_bound_never_tightens_the_real_admission() {
        // For every op the pre-probe bound admits AT LEAST what the seed policy
        // and the most permissive calibrated policy admit: check against both
        // the default seed and a synthetic policy floored at the calibration
        // minima. (If the real admission passed, the pre-gate must have too —
        // otherwise the ordering fix would change GPU-sized behaviour.)
        let floor_policy = GpuDispatchPolicy {
            gemm_min_flops: usize::try_from(GpuDispatchPolicy::MIN_CALIBRATABLE_GEMM_FLOPS)
                .expect("fits usize"),
            potrf_min_p: GpuDispatchPolicy::MIN_CALIBRATABLE_POTRF_P,
            xtwx_flops_min: 4_194_304, // smallest xtwx calibration measurement
            ..GpuDispatchPolicy::default()
        };
        let policies = [GpuDispatchPolicy::default(), floor_policy];
        let ops = [
            DispatchOp::Gemm {
                m: 64,
                n: 64,
                k: 64,
            },
            DispatchOp::Gemm {
                m: 63,
                n: 64,
                k: 64,
            },
            DispatchOp::BatchedGemm {
                batch: 8,
                m: 64,
                n: 64,
                k: 8,
            },
            DispatchOp::Gemv { m: 512, k: 512 },
            DispatchOp::Potrf { p: 64, batch: 1 },
            DispatchOp::Potrf { p: 63, batch: 1 },
            DispatchOp::Potrf { p: 24, batch: 512 },
            DispatchOp::SmallDenseBatchedPotrf { p: 24, batch: 8 },
            DispatchOp::SmallDenseBatchedPotrf { p: 24, batch: 7 },
            DispatchOp::Trsm { m: 128, n: 64 },
            DispatchOp::XtDiagX { n: 50_000, p: 96 },
            DispatchOp::XtDiagX { n: 700, p: 24 },
            DispatchOp::XtDiagY {
                n: 50_000,
                px: 96,
                q: 8,
            },
            DispatchOp::JointHessian2x2 {
                n: 50_000,
                pa: 64,
                pb: 64,
            },
        ];
        for policy in &policies {
            for op in ops {
                let admitted = match op {
                    DispatchOp::Gemm { m, n, k } => {
                        op.flops() >= policy.gemm_min_flops as u128 && m.min(n).min(k) > 0
                    }
                    DispatchOp::BatchedGemm { batch, m, n, k } => {
                        op.flops() >= policy.gemm_min_flops as u128
                            && batch > 1
                            && m.min(n).min(k) > 0
                    }
                    DispatchOp::Gemv { m, k } => {
                        op.flops() >= policy.gemm_min_flops as u128 && m > 0 && k > 0
                    }
                    DispatchOp::Potrf { p, batch } => {
                        p > 0
                            && batch > 0
                            && (p >= policy.potrf_min_p
                                || (batch > 1 && op.flops() >= policy.gemm_min_flops as u128))
                    }
                    DispatchOp::SmallDenseBatchedPotrf { p, batch } => {
                        p > 0
                            && p <= policy.small_dense_batched_potrf_max_p
                            && batch >= policy.small_dense_batched_potrf_min_batch
                    }
                    DispatchOp::Trsm { m, n } => {
                        op.flops() >= policy.gemm_min_flops as u128 && m > 0 && n > 0
                    }
                    DispatchOp::XtDiagX { n, p } => policy.xtwx_target_is_gpu(n, p, true),
                    DispatchOp::XtDiagY { n, px, q } => policy.xtwy_target_is_gpu(n, px, q, true),
                    DispatchOp::JointHessian2x2 { n, pa, pb } => {
                        n > 0
                            && (pa > 0 || pb > 0)
                            && op.flops() >= policy.gemm_min_flops as u128
                    }
                };
                if admitted {
                    assert!(
                        op.admissible_under_any_policy(),
                        "pre-probe bound must not refuse an op the real admission accepts: \
                         {op:?} under {policy:?}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DispatchOp, route_through_gpu, try_fast_ab};
    use crate::GpuPolicy;
    use crate::device_runtime::GpuRuntime;

    fn available_runtime_or_skip(label: &str) -> Option<&'static GpuRuntime> {
        match GpuRuntime::resolve(GpuPolicy::Auto) {
            Ok(runtime) => runtime,
            Err(error) => panic!("[{label}] GPU probe fault: {error}"),
        }
    }

    #[test]
    fn sae_shape_dispatch_ops_route_when_cuda_runtime_is_present() {
        let Some(runtime) = available_runtime_or_skip("sae dispatch gate") else {
            eprintln!("[sae dispatch gate] no CUDA runtime - skipping branch-admission check");
            return;
        };

        let n = 2_000usize;
        let p = 2_048usize;
        let m = 12usize;
        let k = 8usize;
        let dense_reduction_ops = [
            DispatchOp::XtDiagX { n, p },
            DispatchOp::XtDiagY { n, px: p, q: m * k },
            DispatchOp::JointHessian2x2 {
                n,
                pa: p,
                pb: m * k,
            },
            DispatchOp::Gemm {
                m: p,
                n: p,
                k: n * m,
            },
        ];

        for op in dense_reduction_ops {
            assert!(
                op.flops() >= runtime.policy.gemm_min_flops as u128,
                "SAE dispatch fixture must clear the runtime GEMM work floor: op={op:?}, flops={}, floor={}",
                op.flops(),
                runtime.policy.gemm_min_flops
            );
            assert!(
                route_through_gpu(op).is_some(),
                "SAE dispatch fixture should route to GPU when CUDA is present: {op:?}"
            );
        }

        let batched_potrf = DispatchOp::SmallDenseBatchedPotrf { p: m, batch: n };
        assert!(
            route_through_gpu(batched_potrf).is_some(),
            "uniform SAE row blocks should reach the small-dense batched POTRF gate"
        );
    }

    /// Resolving an available runtime must install the dense-GEMM dispatch
    /// hook into `gam_linalg`, so a profitable `fast_ab` call routes through
    /// the device — and the device result must match the CPU oracle within
    /// IEEE reduction tolerance. This is the regression guard for the bug
    /// where `CudaGemmDispatch` existed but `register_gpu_dispatch` was never
    /// called, leaving every engine GEMM silently on the CPU.
    #[test]
    fn global_runtime_installs_fast_ab_hook_and_matches_cpu() {
        use ndarray::Array2;

        let Some(_runtime) = available_runtime_or_skip("fast_ab hook") else {
            eprintln!("[fast_ab hook] no CUDA runtime - skipping engagement check");
            return;
        };
        // After resolution returned an available device, the hook MUST be installed.
        assert!(
            gam_linalg::gpu_hook::gpu_dispatch().is_some(),
            "GpuRuntime::resolve(Auto) returned a device but did not register the \
             dense-GEMM dispatch hook — fast_ab would silently stay on the CPU"
        );

        // A profitable GEMM (m=n=k=512 → 2·512³ ≈ 268 MFLOP, above the
        // 100 MFLOP policy floor) must route to the device. Kept modest so
        // the debug-build CPU oracle below stays a few seconds, not a minute.
        let (m, k, n) = (512usize, 512usize, 512usize);
        assert!(
            route_through_gpu(DispatchOp::Gemm { m, n, k }).is_some(),
            "a 268 MFLOP GEMM must clear the policy floor and route to GPU"
        );

        // Deterministic, well-conditioned operands.
        let a = Array2::<f64>::from_shape_fn((m, k), |(i, j)| {
            ((i * 7 + j * 3) % 13) as f64 * 0.01 - 0.06
        });
        let b = Array2::<f64>::from_shape_fn((k, n), |(i, j)| {
            ((i * 5 + j * 11) % 17) as f64 * 0.01 - 0.08
        });

        // Device result via the dispatch entry point.
        let gpu = try_fast_ab(a.view(), b.view())
            .expect("profitable GEMM must produce a device result once admitted");

        // CPU oracle (plain triple loop — independent of faer/matrixmultiply).
        let mut cpu = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += a[[i, p]] * b[[p, j]];
                }
                cpu[[i, j]] = acc;
            }
        }

        let mut max_abs = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                max_abs = max_abs.max((gpu[[i, j]] - cpu[[i, j]]).abs());
            }
        }
        assert!(
            max_abs < 1e-9,
            "device GEMM disagreed with the CPU oracle: max|Δ| = {max_abs:e}"
        );
    }

    /// The transpose-free `gemm_cuda` path (`Cᵀ = op(B)ᵀ·op(A)ᵀ`, no host
    /// `to_col_major`/`from_col_major`) must match a CPU oracle across all
    /// four `(trans_a, trans_b)` combinations AND non-square, tall-skinny
    /// shapes — the regime (200000×200 · 200×k) where the host transpose
    /// previously dominated. This guards the leading-dimension/operand-swap
    /// derivation against silent index bugs.
    #[cfg(target_os = "linux")]
    #[test]
    fn transpose_free_gemm_matches_cpu_all_trans_and_shapes() {
        use crate::blas::gemm_cuda;
        use ndarray::Array2;

        let Some(runtime) = available_runtime_or_skip("gemm transpose-free") else {
            eprintln!("[gemm transpose-free] no CUDA runtime - skipping");
            return;
        };

        // (rows, inner, cols) logical product dims, deliberately all distinct
        // and rectangular so any lda/ldb/ldc mix-up surfaces.
        let cases = [(6usize, 4usize, 5usize), (17, 23, 9), (200, 31, 7)];
        for (m, k, n) in cases {
            // Base operands sized for the no-transpose orientation; transposed
            // variants are built by swapping dims so the logical product is
            // always (m × n).
            let mk = Array2::<f64>::from_shape_fn((m, k), |(i, j)| {
                ((i * 31 + j * 17) % 19) as f64 * 0.013 - 0.11
            });
            let km = Array2::<f64>::from_shape_fn((k, m), |(i, j)| {
                ((i * 13 + j * 29) % 23) as f64 * 0.011 - 0.07
            });
            let kn = Array2::<f64>::from_shape_fn((k, n), |(i, j)| {
                ((i * 7 + j * 5) % 17) as f64 * 0.017 - 0.09
            });
            let nk = Array2::<f64>::from_shape_fn((n, k), |(i, j)| {
                ((i * 19 + j * 11) % 13) as f64 * 0.015 - 0.05
            });

            for &trans_a in &[false, true] {
                for &trans_b in &[false, true] {
                    let a = if trans_a { &km } else { &mk };
                    let b = if trans_b { &nk } else { &kn };

                    let gpu = gemm_cuda(runtime, a.view(), b.view(), trans_a, trans_b).expect(
                        "transpose-free device GEMM must produce a result when a device is present",
                    );
                    assert_eq!(
                        gpu.dim(),
                        (m, n),
                        "output shape wrong for trans_a={trans_a} trans_b={trans_b} ({m}×{k}×{n})"
                    );

                    // CPU oracle on the logically-transposed operands.
                    let mut cpu = Array2::<f64>::zeros((m, n));
                    for i in 0..m {
                        for j in 0..n {
                            let mut acc = 0.0f64;
                            for p in 0..k {
                                let av = if trans_a { a[[p, i]] } else { a[[i, p]] };
                                let bv = if trans_b { b[[j, p]] } else { b[[p, j]] };
                                acc += av * bv;
                            }
                            cpu[[i, j]] = acc;
                        }
                    }

                    let mut max_abs = 0.0f64;
                    for i in 0..m {
                        for j in 0..n {
                            max_abs = max_abs.max((gpu[[i, j]] - cpu[[i, j]]).abs());
                        }
                    }
                    assert!(
                        max_abs < 1e-9,
                        "transpose-free GEMM mismatch (trans_a={trans_a} trans_b={trans_b}, \
                         {m}×{k}×{n}): max|Δ| = {max_abs:e}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Backend selection. The wrappers keep CUDA types out of solver modules while
// delegating to cudarc-backed BLAS, solver, and custom kernel implementations.
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
mod cuda_backend {
    //! CUDA-backed implementations of the dispatch entry points.
    //!
    //! The real device kernels live in `super::super::blas` and
    //! `super::super::kernels::*`; this module simply forwards. When the
    //! lower layer reports an unrecoverable backend error (OOM, transient
    //! launch failure, …) the wrapper reports `None` to the policy-aware outer
    //! boundary, which fails rather than silently changing execution backend.
    //! Successful numerical results match the CPU code modulo IEEE-754
    //! reduction order.

    use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};

    use super::super::device_runtime::GpuRuntime;
    use crate::driver::{from_col_major, to_col_major, to_i32};
    use cudarc::cusolver::{DnHandle, sys as cusolver_sys};
    use cudarc::driver::{DevicePtrMut, sys as driver_sys};

    #[inline]
    pub(super) fn gemm(
        runtime: &GpuRuntime,
        a: ArrayView2<'_, f64>,
        b: ArrayView2<'_, f64>,
        trans_a: bool,
        trans_b: bool,
    ) -> Option<Array2<f64>> {
        super::super::blas::gemm_cuda(runtime, a, b, trans_a, trans_b)
    }

    #[inline]
    pub(super) fn gemm_on_ordinal(
        ordinal: usize,
        a: ArrayView2<'_, f64>,
        b: ArrayView2<'_, f64>,
        trans_a: bool,
        trans_b: bool,
    ) -> Option<Array2<f64>> {
        super::super::blas::gemm_on_ordinal_cuda(ordinal, a, b, trans_a, trans_b)
    }

    #[inline]
    pub(super) fn gemv(
        runtime: &GpuRuntime,
        a: ArrayView2<'_, f64>,
        v: ArrayView1<'_, f64>,
        trans_a: bool,
    ) -> Option<Array1<f64>> {
        super::super::blas::gemv_cuda(runtime, a, v, trans_a)
    }

    #[inline]
    pub(super) fn gemm_broadcast_b_batched(
        ordinal: usize,
        a: ArrayView3<'_, f64>,
        b: ArrayView2<'_, f64>,
    ) -> Option<Array3<f64>> {
        super::super::blas::gemm_broadcast_b_batched_cuda(ordinal, a, b)
    }

    #[inline]
    pub(super) fn gemm_abt_strided_batched(
        ordinal: usize,
        a: ArrayView3<'_, f64>,
        b: ArrayView3<'_, f64>,
    ) -> Option<Array3<f64>> {
        super::super::blas::gemm_abt_strided_batched_cuda(ordinal, a, b)
    }

    #[inline]
    pub(super) fn xt_diag_x(
        runtime: &GpuRuntime,
        x: ArrayView2<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> Option<Array2<f64>> {
        super::super::blas::xt_diag_x_cuda(runtime, x, w)
    }

    #[inline]
    pub(super) fn xt_diag_y(
        runtime: &GpuRuntime,
        x: ArrayView2<'_, f64>,
        w: ArrayView1<'_, f64>,
        y: ArrayView2<'_, f64>,
    ) -> Option<Array2<f64>> {
        super::super::blas::xt_diag_y_cuda(runtime, x, w, y)
    }

    #[inline]
    pub(super) fn joint_hessian_2x2(
        runtime: &GpuRuntime,
        x_a: ArrayView2<'_, f64>,
        x_b: ArrayView2<'_, f64>,
        w_aa: ArrayView1<'_, f64>,
        w_ab: ArrayView1<'_, f64>,
        w_bb: ArrayView1<'_, f64>,
    ) -> Option<Array2<f64>> {
        super::super::blas::joint_hessian_2x2_cuda(runtime, x_a, x_b, w_aa, w_ab, w_bb)
    }

    #[inline]
    pub(super) fn trsm(
        runtime: &GpuRuntime,
        triangular: ArrayView2<'_, f64>,
        rhs: ArrayView2<'_, f64>,
        upper: bool,
    ) -> Option<Array2<f64>> {
        super::super::blas::trsm_cuda(runtime, triangular, rhs, upper)
    }

    #[inline]
    pub(super) fn cholesky_lower(
        runtime: &GpuRuntime,
        a: ArrayView2<'_, f64>,
    ) -> Option<Array2<f64>> {
        let (p, p2) = a.dim();
        if p == 0 || p != p2 {
            return None;
        }
        let stream = super::super::device_runtime::cuda_context_for(runtime.device.ordinal)?
            .new_stream()
            .ok()?;
        let solver = DnHandle::new(stream.clone()).ok()?;
        let a_col = to_col_major(&a);
        let mut a_dev = stream.clone_htod(&*a_col).ok()?;
        potrf_lower_in_place(&solver, &stream, p, &mut a_dev)?;
        let factor_col = stream.clone_dtoh(&a_dev).ok()?;
        let mut lower = from_col_major(&factor_col, p, p)?;
        for row in 0..p {
            for col in (row + 1)..p {
                lower[[row, col]] = 0.0;
            }
        }
        Some(lower)
    }

    /// Batched lower-Cholesky on a specific device ordinal. The ordinal's
    /// context is expected to be bound on the calling thread (multi-GPU
    /// `scatter_batched` worker or the single-device dispatcher).
    #[inline]
    pub(super) fn cholesky_batched_lower(
        ordinal: usize,
        matrices: &mut [Array2<f64>],
    ) -> Option<()> {
        let first = matrices.first()?;
        let p = first.nrows();
        if p == 0 || first.ncols() != p || matrices.iter().any(|matrix| matrix.dim() != (p, p)) {
            return None;
        }

        let stream = super::super::device_runtime::cuda_context_for(ordinal)?
            .new_stream()
            .ok()?;
        let solver = DnHandle::new(stream.clone()).ok()?;
        let matrix_len = p.checked_mul(p)?;
        let mut batch_col = Vec::with_capacity(matrices.len().checked_mul(matrix_len)?);
        for matrix in matrices.iter() {
            batch_col.extend(to_col_major(&matrix.view()).iter().copied());
        }
        let mut matrices_dev = stream.clone_htod(&batch_col).ok()?;
        let matrix_ptrs = {
            let (base_ptr, _matrix_record) = matrices_dev.device_ptr_mut(&stream);
            let bytes_per_matrix = driver_sys::CUdeviceptr::try_from(
                matrix_len.checked_mul(std::mem::size_of::<f64>())?,
            )
            .ok()?;
            let mut matrix_ptrs = Vec::with_capacity(matrices.len());
            for idx in 0..matrices.len() {
                let offset = driver_sys::CUdeviceptr::try_from(idx).ok()? * bytes_per_matrix;
                matrix_ptrs.push(base_ptr + offset);
            }
            matrix_ptrs
        };
        let mut matrix_ptrs_dev = stream.clone_htod(&matrix_ptrs).ok()?;
        let mut info_dev = stream.alloc_zeros::<i32>(matrices.len()).ok()?;
        let p_i = to_i32(p)?;
        let batch_i = to_i32(matrices.len())?;
        {
            let (ptrs_ptr, _ptrs_record) = matrix_ptrs_dev.device_ptr_mut(&stream);
            let (info_ptr, _info_record) = info_dev.device_ptr_mut(&stream);
            // SAFETY: `ptrs_ptr` points to a device array of batch pointers,
            // each pointer targets a live p×p column-major matrix in
            // `matrices_dev`, and `info_dev` has one entry per batch item.
            let status = unsafe {
                cusolver_sys::cusolverDnDpotrfBatched(
                    solver.cu(),
                    cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                    p_i,
                    ptrs_ptr as *mut *mut f64,
                    p_i,
                    info_ptr as *mut i32,
                    batch_i,
                )
            };
            check_cusolver(status)?;
        }
        let info_host = stream.clone_dtoh(&info_dev).ok()?;
        if info_host.iter().any(|info| *info != 0) {
            return None;
        }
        let factored_col = stream.clone_dtoh(&matrices_dev).ok()?;
        for (idx, matrix) in matrices.iter_mut().enumerate() {
            let start = idx.checked_mul(matrix_len)?;
            let end = start.checked_add(matrix_len)?;
            let mut lower = from_col_major(&factored_col[start..end], p, p)?;
            for row in 0..p {
                for col in (row + 1)..p {
                    lower[[row, col]] = 0.0;
                }
            }
            *matrix = lower;
        }
        Some(())
    }

    /// Single-matrix lower Cholesky POTRF. Thin `Result → Option` adapter over
    /// the shared precision-generic core in `solver.rs`
    /// ([`crate::solver::potrf_in_place_generic`]) so the cuSOLVER
    /// bufferSize/POTRF/info scaffold lives in exactly one place. The batched
    /// variant (`cusolverDnDpotrfBatched`) above is kept separate by design.
    fn potrf_lower_in_place(
        solver: &DnHandle,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
        a: &mut cudarc::driver::CudaSlice<f64>,
    ) -> Option<()> {
        crate::solver::potrf_in_place_generic::<f64>(solver, stream, p, a).ok()
    }

    #[inline]
    fn check_cusolver(status: cusolver_sys::cusolverStatus_t) -> Option<()> {
        if status == cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            Some(())
        } else {
            None
        }
    }
}

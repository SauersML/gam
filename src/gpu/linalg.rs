//! Automatic GPU dispatch shim for dense linear algebra hot kernels.
//!
//! Every `try_*` entry point in this module is invoked unconditionally from
//! `crate::faer_ndarray` before the CPU fast-path runs. The decision to send
//! the kernel to a device is fully automatic and never requires a user-facing
//! flag — it depends only on:
//!
//!   1. `GpuRuntime::global()` returning `Some(_)` (a device was probed at
//!      process startup).
//!   2. The kernel being large enough to amortize launch/PCIe overhead, per
//!      the thresholds in `policy::GpuDispatchPolicy`.
//!   3. The `cuda` Cargo feature being enabled at compile time. Without it,
//!      no backend symbols are linked and every `try_*` returns `None` so the
//!      caller falls through to the existing faer CPU kernel.
//!
//! The wiring lives here so `solver/pirls.rs` and the family Hessian
//! assemblers can stay backend-agnostic: they call `crate::faer_ndarray::fast_*`
//! and get GPU acceleration automatically whenever it is profitable.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::runtime::GpuRuntime;

/// Discriminator used by [`route_through_gpu`] to apply the right
/// size threshold from [`super::policy::GpuDispatchPolicy`].
#[derive(Clone, Copy, Debug)]
pub enum DispatchOp {
    /// Generic matrix-matrix product with the given output dims and reduction depth.
    Gemm { m: usize, n: usize, k: usize },
    /// Matrix-vector (or matrix · single-column) product.
    Gemv { m: usize, k: usize },
    /// `Xᵀ · diag(w) · X` reduction with n rows and p columns.
    XtDiagX { n: usize, p: usize },
    /// `Xᵀ · diag(w) · Y` reduction; px and q are the design and response widths.
    XtDiagY { n: usize, px: usize, q: usize },
    /// 2×2 joint Hessian block with two design widths.
    JointHessian2x2 { n: usize, pa: usize, pb: usize },
}

impl DispatchOp {
    /// Conservative flop estimate used for the generic `gemm_min_flops` gate.
    #[inline]
    pub const fn flops(self) -> u128 {
        match self {
            Self::Gemm { m, n, k } => 2u128 * (m as u128) * (n as u128) * (k as u128),
            Self::Gemv { m, k } => 2u128 * (m as u128) * (k as u128),
            Self::XtDiagX { n, p } => 2u128 * (n as u128) * (p as u128) * (p as u128),
            Self::XtDiagY { n, px, q } => 2u128 * (n as u128) * (px as u128) * (q as u128),
            Self::JointHessian2x2 { n, pa, pb } => {
                let total = (pa as u128) + (pb as u128);
                2u128 * (n as u128) * total * total
            }
        }
    }
}

/// Returns `Some(runtime)` when both a device is available and the workload
/// is large enough per policy. The caller can then attempt the actual device
/// kernel; any backend failure is expected to return `None` from the lower
/// layer and the CPU fast path resumes.
#[inline]
#[must_use]
pub fn route_through_gpu(op: DispatchOp) -> Option<&'static GpuRuntime> {
    let runtime = GpuRuntime::global()?;
    let policy = &runtime.policy;
    let admit = match op {
        DispatchOp::Gemm { m, n, k } => {
            (op.flops() as u128) >= (policy.gemm_min_flops as u128)
                && m.min(n).min(k) > 0
        }
        DispatchOp::Gemv { m, k } => {
            (op.flops() as u128) >= (policy.gemm_min_flops as u128) && m > 0 && k > 0
        }
        DispatchOp::XtDiagX { n, p } => {
            n >= policy.xtwx_n_min
                && (op.flops() as u128) >= (policy.xtwx_flops_min as u128)
                && p > 0
        }
        DispatchOp::XtDiagY { n, px, q } => {
            n >= policy.xtwx_n_min
                && (op.flops() as u128) >= (policy.xtwx_flops_min as u128)
                && px > 0
                && q > 0
        }
        DispatchOp::JointHessian2x2 { n, pa, pb } => {
            n >= policy.fused_kernel_min_n && (pa + pb) > 0
        }
    };
    if admit { Some(runtime) } else { None }
}

// ---------------------------------------------------------------------------
// Dispatch entry points. Each takes views to keep the call site allocation-
// free and returns Some(result) iff the GPU actually produced one. The CPU
// fast path resumes on None.
//
// Under the default feature set there is no compiled backend, so every entry
// point short-circuits to None after running the policy check (which itself
// returns None today because `GpuRuntime::probe()` reports no device). The
// architecture is in place so that wiring up cudarc inside `#[cfg(feature =
// "cuda")]` blocks immediately accelerates every fast_* call site without
// touching solver code.
// ---------------------------------------------------------------------------

#[inline]
#[must_use]
pub fn try_fast_ab(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
    let (m, k) = a.dim();
    let (kb, n) = b.dim();
    if k != kb {
        return None;
    }
    let runtime = route_through_gpu(DispatchOp::Gemm { m, n, k })?;
    cuda_backend::gemm(runtime, a, b, false, false)
}

#[inline]
#[must_use]
pub fn try_fast_atb(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
    let (n_a, p) = a.dim();
    let (n_b, q) = b.dim();
    if n_a != n_b {
        return None;
    }
    let runtime = route_through_gpu(DispatchOp::Gemm { m: p, n: q, k: n_a })?;
    cuda_backend::gemm(runtime, a, b, true, false)
}

#[inline]
#[must_use]
pub fn try_fast_av(a: ArrayView2<'_, f64>, v: ArrayView1<'_, f64>) -> Option<Array1<f64>> {
    let (m, k) = a.dim();
    if k != v.len() {
        return None;
    }
    let runtime = route_through_gpu(DispatchOp::Gemv { m, k })?;
    cuda_backend::gemv(runtime, a, v, false)
}

#[inline]
#[must_use]
pub fn try_fast_atv(a: ArrayView2<'_, f64>, v: ArrayView1<'_, f64>) -> Option<Array1<f64>> {
    let (n, p) = a.dim();
    if n != v.len() {
        return None;
    }
    let runtime = route_through_gpu(DispatchOp::Gemv { m: p, k: n })?;
    cuda_backend::gemv(runtime, a, v, true)
}

#[inline]
#[must_use]
pub fn try_fast_xt_diag_x(
    x: ArrayView2<'_, f64>,
    w: ArrayView1<'_, f64>,
) -> Option<Array2<f64>> {
    let (n, p) = x.dim();
    if n != w.len() {
        return None;
    }
    let runtime = route_through_gpu(DispatchOp::XtDiagX { n, p })?;
    cuda_backend::xt_diag_x(runtime, x, w)
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
        return None;
    }
    let runtime = route_through_gpu(DispatchOp::XtDiagY { n, px, q })?;
    cuda_backend::xt_diag_y(runtime, x, w, y)
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
        return None;
    }
    let runtime = route_through_gpu(DispatchOp::JointHessian2x2 { n, pa, pb })?;
    cuda_backend::joint_hessian_2x2(runtime, x_a, x_b, w_aa, w_ab, w_bb)
}

// Per-call latency hint kept for future profile-driven decisions.
#[derive(Clone, Debug)]
pub enum GpuDispatch {
    /// Caller should fall back to the CPU implementation.
    Cpu,
    /// A GPU dispatch executed; the result is carried by the variant.
    Executed(Array2<f64>),
}

#[inline]
#[must_use]
pub fn should_dispatch_xt_diag_x(n: usize, p: usize) -> bool {
    route_through_gpu(DispatchOp::XtDiagX { n, p }).is_some()
}

#[inline]
#[must_use]
pub fn should_dispatch_xt_diag_y(n: usize, px: usize, q: usize) -> bool {
    route_through_gpu(DispatchOp::XtDiagY { n, px, q }).is_some()
}

#[inline]
#[must_use]
pub fn should_dispatch_joint_hessian(n: usize, pa: usize, pb: usize) -> bool {
    route_through_gpu(DispatchOp::JointHessian2x2 { n, pa, pb }).is_some()
}

// ---------------------------------------------------------------------------
// Backend selection. Each fn below is either a stub (no cuda feature) or a
// thin wrapper around cudarc-backed kernels (when cuda is enabled). The CPU
// stub variant intentionally avoids touching its arguments so the call site
// is dead-code-eliminated.
// ---------------------------------------------------------------------------

mod cuda_backend {
    //! CUDA-backed implementations of the dispatch entry points.
    //!
    //! The real device kernels live in `super::super::blas` and
    //! `super::super::kernels::*`; this module simply forwards. When the
    //! lower layer reports an unrecoverable backend error (OOM, transient
    //! launch failure, …) the wrapper returns `None` so the CPU fast path
    //! is exercised — there is never a silent panic, and the numerical
    //! result is identical to the CPU code modulo IEEE-754 reduction order.

    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

    use super::super::runtime::GpuRuntime;

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
    pub(super) fn gemv(
        runtime: &GpuRuntime,
        a: ArrayView2<'_, f64>,
        v: ArrayView1<'_, f64>,
        trans_a: bool,
    ) -> Option<Array1<f64>> {
        super::super::blas::gemv_cuda(runtime, a, v, trans_a)
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
}

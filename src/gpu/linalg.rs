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
//!   3. cudarc successfully dynamically loading `libcuda` at process startup
//!      via its `fallback-dynamic-loading` feature. When the loader fails
//!      (no driver, no toolkit installed), `GpuRuntime::probe()` returns
//!      `Ok(None)` and every `try_*` returns `None` so the caller falls
//!      through to the existing faer CPU kernel.
//!
//! The wiring lives here so `solver/pirls.rs` and the family Hessian
//! assemblers can stay backend-agnostic: they call `crate::faer_ndarray::fast_*`
//! and get GPU acceleration automatically whenever it is profitable.

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};

use super::runtime::GpuRuntime;

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
            Self::Trsm { m, n } => (m as u128) * (m as u128) * (n as u128),
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
            op.flops() >= (policy.gemm_min_flops as u128) && m.min(n).min(k) > 0
        }
        DispatchOp::BatchedGemm { batch, m, n, k } => {
            op.flops() >= (policy.gemm_min_flops as u128) && batch > 1 && m.min(n).min(k) > 0
        }
        DispatchOp::Gemv { m, k } => {
            op.flops() >= (policy.gemm_min_flops as u128) && m > 0 && k > 0
        }
        DispatchOp::Potrf { p, batch } => p >= policy.potrf_min_p && batch > 0,
        DispatchOp::Trsm { m, n } => {
            op.flops() >= (policy.gemm_min_flops as u128) && m > 0 && n > 0
        }
        DispatchOp::XtDiagX { n, p } => {
            n >= policy.xtwx_n_min && op.flops() >= (policy.xtwx_flops_min as u128) && p > 0
        }
        DispatchOp::XtDiagY { n, px, q } => {
            n >= policy.xtwx_n_min
                && op.flops() >= (policy.xtwx_flops_min as u128)
                && px > 0
                && q > 0
        }
        DispatchOp::JointHessian2x2 { n, pa, pb } => {
            n >= policy.fused_kernel_min_n && (pa + pb) > 0
        }
    };
    if admit { Some(runtime) } else { None }
}

#[inline]
#[must_use]
pub fn try_fast_ab_broadcast_b_batched(
    a: ArrayView3<'_, f64>,
    b: ArrayView2<'_, f64>,
) -> Option<Array3<f64>> {
    let (_batch, _m, k) = a.dim();
    let (bk, _n) = b.dim();
    if k != bk {
        return None;
    }
    #[cfg(not(target_os = "linux"))]
    {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::BatchedGemm {
            batch: _batch,
            m: _m,
            n: _n,
            k,
        })?;
        cuda_backend::gemm_broadcast_b_batched(runtime, a, b)
    }
}

#[inline]
#[must_use]
pub fn try_fast_abt_strided_batched(
    a: ArrayView3<'_, f64>,
    b: ArrayView3<'_, f64>,
) -> Option<Array3<f64>> {
    let (_batch, _m, k) = a.dim();
    let (batch_b, _n, k_b) = b.dim();
    if _batch != batch_b || k != k_b {
        return None;
    }
    #[cfg(not(target_os = "linux"))]
    {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::BatchedGemm {
            batch: _batch,
            m: _m,
            n: _n,
            k,
        })?;
        cuda_backend::gemm_abt_strided_batched(runtime, a, b)
    }
}

// ---------------------------------------------------------------------------
// Dispatch entry points. Each takes views to keep the call site allocation-
// free and returns Some(result) iff the GPU actually produced one. The CPU
// fast path resumes on None.
//
// CUDA kernels are compiled into the runtime through cudarc's dynamic loader.
// Each entry point admits only profitable workloads, then returns `None` when
// no CUDA runtime path is available or the backend reports failure.
// ---------------------------------------------------------------------------

#[inline]
#[must_use]
pub fn try_fast_ab(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
    let (m, k) = a.dim();
    let (kb, n) = b.dim();
    if k != kb {
        return None;
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
        None
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = runtime?;
        cuda_backend::gemm(runtime, a, b, false, false)
    }
}

#[inline]
#[must_use]
pub fn try_fast_atb(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
    let (n_a, _p) = a.dim();
    let (n_b, _q) = b.dim();
    if n_a != n_b {
        return None;
    }
    #[cfg(not(target_os = "linux"))]
    {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::Gemm {
            m: _p,
            n: _q,
            k: n_a,
        })?;
        cuda_backend::gemm(runtime, a, b, true, false)
    }
}

#[inline]
#[must_use]
pub fn try_fast_av(a: ArrayView2<'_, f64>, v: ArrayView1<'_, f64>) -> Option<Array1<f64>> {
    let (_m, k) = a.dim();
    if k != v.len() {
        return None;
    }
    #[cfg(not(target_os = "linux"))]
    {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::Gemv { m: _m, k })?;
        cuda_backend::gemv(runtime, a, v, false)
    }
}

#[inline]
#[must_use]
pub fn try_fast_atv(a: ArrayView2<'_, f64>, v: ArrayView1<'_, f64>) -> Option<Array1<f64>> {
    let (n, _p) = a.dim();
    if n != v.len() {
        return None;
    }
    #[cfg(not(target_os = "linux"))]
    {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::Gemv { m: _p, k: n })?;
        cuda_backend::gemv(runtime, a, v, true)
    }
}

#[inline]
#[must_use]
pub fn try_fast_xt_diag_x(x: ArrayView2<'_, f64>, w: ArrayView1<'_, f64>) -> Option<Array2<f64>> {
    let (n, _p) = x.dim();
    if n != w.len() {
        return None;
    }
    #[cfg(not(target_os = "linux"))]
    {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::XtDiagX { n, p: _p })?;
        cuda_backend::xt_diag_x(runtime, x, w)
    }
}

#[inline]
#[must_use]
pub fn try_fast_xt_diag_y(
    x: ArrayView2<'_, f64>,
    w: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
) -> Option<Array2<f64>> {
    let (n, _px) = x.dim();
    let (n_y, _q) = y.dim();
    if n != n_y || n != w.len() {
        return None;
    }
    #[cfg(not(target_os = "linux"))]
    {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::XtDiagY { n, px: _px, q: _q })?;
        cuda_backend::xt_diag_y(runtime, x, w, y)
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
    let (n, _pa) = x_a.dim();
    let (n_b, _pb) = x_b.dim();
    if n != n_b || n != w_aa.len() || n != w_ab.len() || n != w_bb.len() {
        return None;
    }
    #[cfg(not(target_os = "linux"))]
    {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::JointHessian2x2 {
            n,
            pa: _pa,
            pb: _pb,
        })?;
        cuda_backend::joint_hessian_2x2(runtime, x_a, x_b, w_aa, w_ab, w_bb)
    }
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

#[inline]
#[must_use]
pub fn try_cholesky_lower_inplace(a: &mut Array2<f64>) -> Option<()> {
    let p = a.nrows();
    if p != a.ncols() {
        return None;
    }
    #[cfg(not(target_os = "linux"))]
    {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::Potrf { p, batch: 1 })?;
        let lower = cuda_backend::cholesky_lower(runtime, a.view())?;
        *a = lower;
        Some(())
    }
}

#[inline]
#[must_use]
pub fn try_cholesky_batched_lower_inplace(matrices: &mut [Array2<f64>]) -> Option<()> {
    let first = matrices.first()?;
    let p = first.nrows();
    if p == 0 || first.ncols() != p || matrices.iter().any(|matrix| matrix.dim() != (p, p)) {
        return None;
    }
    #[cfg(not(target_os = "linux"))]
    {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::Potrf {
            p,
            batch: matrices.len(),
        })?;
        cuda_backend::cholesky_batched_lower(runtime, matrices)
    }
}

#[inline]
#[must_use]
pub fn try_solve_lower_triangular_matrix(
    lower: ArrayView2<'_, f64>,
    rhs: ArrayView2<'_, f64>,
) -> Option<Array2<f64>> {
    let (m, _n) = rhs.dim();
    if lower.nrows() != m {
        return None;
    }
    #[cfg(not(target_os = "linux"))]
    {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::Trsm { m, n: _n })?;
        cuda_backend::trsm(runtime, lower, rhs, false)
    }
}

#[inline]
#[must_use]
pub fn try_solve_upper_triangular_matrix(
    upper: ArrayView2<'_, f64>,
    rhs: ArrayView2<'_, f64>,
) -> Option<Array2<f64>> {
    let (m, _n) = rhs.dim();
    if upper.nrows() != m {
        return None;
    }
    #[cfg(not(target_os = "linux"))]
    {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        let runtime = route_through_gpu(DispatchOp::Trsm { m, n: _n })?;
        cuda_backend::trsm(runtime, upper, rhs, true)
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
    //! launch failure, …) the wrapper returns `None` so the CPU fast path
    //! is exercised — there is never a silent panic, and the numerical
    //! result is identical to the CPU code modulo IEEE-754 reduction order.

    use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};

    use super::super::runtime::GpuRuntime;
    use crate::gpu::driver::{from_col_major, to_col_major, to_i32};
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
        runtime: &GpuRuntime,
        a: ArrayView3<'_, f64>,
        b: ArrayView2<'_, f64>,
    ) -> Option<Array3<f64>> {
        super::super::blas::gemm_broadcast_b_batched_cuda(runtime, a, b)
    }

    #[inline]
    pub(super) fn gemm_abt_strided_batched(
        runtime: &GpuRuntime,
        a: ArrayView3<'_, f64>,
        b: ArrayView3<'_, f64>,
    ) -> Option<Array3<f64>> {
        super::super::blas::gemm_abt_strided_batched_cuda(runtime, a, b)
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
        let stream = super::super::runtime::cuda_context_for(runtime.device.ordinal)?
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

    #[inline]
    pub(super) fn cholesky_batched_lower(
        runtime: &GpuRuntime,
        matrices: &mut [Array2<f64>],
    ) -> Option<()> {
        let first = matrices.first()?;
        let p = first.nrows();
        if p == 0 || first.ncols() != p || matrices.iter().any(|matrix| matrix.dim() != (p, p)) {
            return None;
        }

        let stream = super::super::runtime::cuda_context_for(runtime.device.ordinal)?
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

    fn potrf_lower_in_place(
        solver: &DnHandle,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        p: usize,
        a: &mut cudarc::driver::CudaSlice<f64>,
    ) -> Option<()> {
        let p_i = to_i32(p)?;
        let uplo = cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
        let mut lwork = 0_i32;
        {
            let (a_ptr, _a_record) = a.device_ptr_mut(stream);
            // SAFETY: `a_ptr` addresses a live p-by-p column-major device buffer,
            // `lwork` is a valid host out-parameter, and `solver` is initialized
            // on the stream that owns the allocation.
            let status = unsafe {
                cusolver_sys::cusolverDnDpotrf_bufferSize(
                    solver.cu(),
                    uplo,
                    p_i,
                    a_ptr as *mut f64,
                    p_i,
                    &mut lwork,
                )
            };
            check_cusolver(status)?;
        }
        let lwork = usize::try_from(lwork).ok()?;
        let mut workspace = stream.alloc_zeros::<f64>(lwork).ok()?;
        let mut info = stream.alloc_zeros::<i32>(1).ok()?;
        {
            let (a_ptr, _a_record) = a.device_ptr_mut(stream);
            let (work_ptr, _work_record) = workspace.device_ptr_mut(stream);
            let (info_ptr, _info_record) = info.device_ptr_mut(stream);
            // SAFETY: `a`, `workspace`, and `info` are live device allocations
            // on this stream. Workspace length comes from the matching cuSOLVER
            // buffer-size query above and leading dimensions are p.
            let status = unsafe {
                cusolver_sys::cusolverDnDpotrf(
                    solver.cu(),
                    uplo,
                    p_i,
                    a_ptr as *mut f64,
                    p_i,
                    work_ptr as *mut f64,
                    i32::try_from(lwork).ok()?,
                    info_ptr as *mut i32,
                )
            };
            check_cusolver(status)?;
        }
        let info_host = stream.clone_dtoh(&info).ok()?;
        if info_host[0] == 0 { Some(()) } else { None }
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

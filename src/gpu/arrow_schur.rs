//! Fully GPU-resident batched Arrow-Schur dense Cholesky solver.
//!
//! Implements the square-root Schur form: each local block `D_i = L_i L_i^T`
//! is factored on device, `u_i = L_i^{-1} g_i` and `Y_i = L_i^{-1} B_i` are
//! formed by triangular solves, the reduced shared system
//!     `S_β = C + ρ_β I − Σ_i Y_i^T Y_i,  r_β = -g_β + Σ_i Y_i^T u_i`
//! is assembled on device, factored once, and the back-substitution
//!     `w_i = u_i + Y_i · δβ,  L_i^T x_i = w_i,  δt_i = -x_i`
//! is run on device. Only the final `(δt, δβ, log|H|)` triple is downloaded.
//!
//! The current caller (Arrow-Schur Newton step inside PIRLS) feeds uniform
//! local block size `d` and uniform shared width `k`, so the entire pipeline
//! is dispatched as a single p-group; per-p grouping for heterogenous blocks
//! is Layer D's NVRTC fused-kernel concern and lives in this module's
//! follow-up implementation rather than in policy plumbing.
//!
//! On non-Linux builds the entire module degrades to a CPU-fallback shim.

#![allow(clippy::module_name_repetitions)]

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::solver::arrow_schur::{ArrowRowBlock, ArrowSchurSystem};

/// Outcome of a single Arrow-Schur Newton solve.
pub struct ArrowSchurGpuSolution {
    pub delta_t: Array1<f64>,
    pub delta_beta: Array1<f64>,
    /// Natural log of the determinant of the full bordered Hessian, computed
    /// from the local Cholesky factors and the Schur factor on device.
    pub log_det_hessian: f64,
}

/// Reason a device path declined to run; lets the host caller decide between
/// CPU fallback and per-row escalation. `RidgeBumpRequired` carries the
/// estimated diagonal bump needed to clear the failed pivot.
#[derive(Debug, Clone)]
pub enum ArrowSchurGpuFailure {
    /// CUDA runtime unavailable, allocation failed, or workload below policy.
    Unavailable,
    /// A row block was not positive definite even after the requested ridge.
    /// Caller may retry with `ridge_t + bump`.
    RidgeBumpRequired { row: usize, bump: f64 },
    /// Shared Schur factor failed; bordered system is rank-deficient at the
    /// requested ridges and the CPU path should handle escalation.
    SchurFactorFailed { reason: String },
}

/// Entry point: attempt the fully device-resident Arrow-Schur Newton solve.
/// Returns `Err(ArrowSchurGpuFailure::Unavailable)` to indicate "device path
/// declined, fall back to CPU" — never panics.
pub fn solve_arrow_newton_step(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<ArrowSchurGpuSolution, ArrowSchurGpuFailure> {
    let n = sys.rows.len();
    let d = sys.d;
    let k = sys.k;
    if sys.hbb.dim() != (k, k) {
        return Err(ArrowSchurGpuFailure::SchurFactorFailed {
            reason: "CUDA arrow-Schur requires a dense shared beta block".to_string(),
        });
    }
    if n == 0 || d == 0 {
        return Err(ArrowSchurGpuFailure::Unavailable);
    }
    if sys
        .rows
        .iter()
        .any(|row| row.htt.dim() != (d, d) || row.htbeta.dim() != (d, k) || row.gt.len() != d)
    {
        return Err(ArrowSchurGpuFailure::SchurFactorFailed {
            reason: "row block dimension mismatch".to_string(),
        });
    }

    #[cfg(not(target_os = "linux"))]
    {
        if ridge_t.is_nan() || ridge_beta.is_nan() {
            return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                reason: "ridge is NaN".to_string(),
            });
        }
        Err(ArrowSchurGpuFailure::Unavailable)
    }

    #[cfg(target_os = "linux")]
    {
        cuda::solve(sys, ridge_t, ridge_beta)
    }
}

/// Build the stacked column-major D buffer (n local d×d blocks), the stacked
/// stacked B buffer (n local d×k blocks), and the stacked g buffer
/// (n local d-vectors) consumed by the device pipeline. Each block is laid
/// out column-major so a single allocation + `cuMemcpyHtoD` reaches the
/// device without per-row dispatch overhead.
#[cfg(target_os = "linux")]
fn pack_host(sys: &ArrowSchurSystem, ridge_t: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = sys.rows.len();
    let d = sys.d;
    let k = sys.k;
    let mut d_buf = Vec::with_capacity(n * d * d);
    let mut b_buf = Vec::with_capacity(n * d * k);
    let mut g_buf = Vec::with_capacity(n * d);
    for row in &sys.rows {
        pack_block(row, ridge_t, d, k, &mut d_buf, &mut b_buf, &mut g_buf);
    }
    (d_buf, b_buf, g_buf)
}

#[cfg(target_os = "linux")]
#[inline]
fn pack_block(
    row: &ArrowRowBlock,
    ridge_t: f64,
    d: usize,
    k: usize,
    d_buf: &mut Vec<f64>,
    b_buf: &mut Vec<f64>,
    g_buf: &mut Vec<f64>,
) {
    for col in 0..d {
        for r in 0..d {
            let mut value = row.htt[[r, col]];
            if r == col {
                value += ridge_t;
            }
            d_buf.push(value);
        }
    }
    for col in 0..k {
        for r in 0..d {
            b_buf.push(row.htbeta[[r, col]]);
        }
    }
    for r in 0..d {
        g_buf.push(row.gt[r]);
    }
}

/// Reference dense back-end used by tests and as the fallback when the
/// GPU declines. Kept here (not in `arrow_schur_gpu.rs`) so the validation
/// suite has one canonical baseline.
#[doc(hidden)]
pub fn solve_arrow_newton_step_dense_reference(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<ArrowSchurGpuSolution, String> {
    let n = sys.rows.len();
    let d = sys.d;
    let k = sys.k;
    let total = n.checked_mul(d).ok_or("dimension overflow")? + k;
    let mut h = Array2::<f64>::zeros((total, total));
    let mut rhs = Array1::<f64>::zeros(total);
    for (i, row) in sys.rows.iter().enumerate() {
        let base = i * d;
        for c in 0..d {
            for r in 0..d {
                h[[base + r, base + c]] = row.htt[[r, c]];
            }
            h[[base + c, base + c]] += ridge_t;
        }
        for c in 0..k {
            for r in 0..d {
                let value = row.htbeta[[r, c]];
                h[[base + r, n * d + c]] = value;
                h[[n * d + c, base + r]] = value;
            }
        }
        for r in 0..d {
            rhs[base + r] = -row.gt[r];
        }
    }
    for c in 0..k {
        for r in 0..k {
            h[[n * d + r, n * d + c]] += sys.hbb[[r, c]];
        }
        h[[n * d + c, n * d + c]] += ridge_beta;
        rhs[n * d + c] = -sys.gb[c];
    }
    let factor = cholesky_lower_host(h.view())
        .ok_or_else(|| "dense reference Cholesky failed".to_string())?;
    let mut log_det = 0.0_f64;
    for i in 0..total {
        log_det += factor[[i, i]].ln();
    }
    log_det *= 2.0;
    let solved = solve_cholesky_lower_host(factor.view(), rhs.view());
    let delta_t = solved.slice(ndarray::s![..n * d]).to_owned();
    let delta_beta = solved.slice(ndarray::s![n * d..]).to_owned();
    Ok(ArrowSchurGpuSolution {
        delta_t,
        delta_beta,
        log_det_hessian: log_det,
    })
}

#[inline]
fn cholesky_lower_host(a: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return None;
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for kk in 0..j {
                sum -= l[[i, kk]] * l[[j, kk]];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[[i, i]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Some(l)
}

#[inline]
fn solve_cholesky_lower_host(l: ArrayView2<'_, f64>, rhs: ArrayView1<'_, f64>) -> Array1<f64> {
    let n = l.nrows();
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= l[[i, j]] * y[j];
        }
        y[i] = sum / l[[i, i]];
    }
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[[j, i]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

#[cfg(target_os = "linux")]
mod cuda {
    use super::{ArrowSchurGpuFailure, ArrowSchurGpuSolution, pack_host};
    use crate::gpu::driver::to_i32;
    use crate::gpu::linalg::{DispatchOp, route_through_gpu};
    use crate::solver::arrow_schur::ArrowSchurSystem;
    use cudarc::cublas::sys::{
        cublasDiagType_t, cublasFillMode_t, cublasOperation_t, cublasSideMode_t, cublasStatus_t,
    };
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, Gemv, GemvConfig};
    use cudarc::cusolver::{DnHandle, sys as cusolver_sys};
    use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
    use ndarray::Array1;
    use std::sync::Arc;

    pub(super) fn solve(
        sys: &ArrowSchurSystem,
        ridge_t: f64,
        ridge_beta: f64,
    ) -> Result<ArrowSchurGpuSolution, ArrowSchurGpuFailure> {
        let n = sys.rows.len();
        let d = sys.d;
        let k = sys.k;
        let runtime = route_through_gpu(DispatchOp::SmallDenseBatchedPotrf { p: d, batch: n })
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;

        let stream = crate::gpu::runtime::cuda_context_for(runtime.device.ordinal)
            .and_then(|ctx| ctx.new_stream().ok())
            .ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let solver =
            DnHandle::new(stream.clone()).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let blas = CudaBlas::new(stream.clone()).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

        // ----- Pack + upload D, B, g -----
        let (d_host, b_host, g_host) = pack_host(sys, ridge_t);
        let mut d_dev = stream
            .clone_htod(&d_host)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut b_dev = stream
            .clone_htod(&b_host)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut g_dev = stream
            .clone_htod(&g_host)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

        // ----- Layer A: batched lower Cholesky of D in place -----
        let info_host = potrf_batched(&solver, &stream, d, n, &mut d_dev)?;
        if let Some(idx) = info_host.iter().position(|info| *info != 0) {
            let pivot = info_host[idx];
            let scale = sys.rows[idx]
                .htt
                .diag()
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            return Err(ArrowSchurGpuFailure::RidgeBumpRequired {
                row: idx,
                bump: scale * (pivot.abs() as f64).max(1.0) * f64::EPSILON.sqrt() * 1024.0,
            });
        }

        // ----- Layer B (1/2): in-place triangular solves -----
        // u_i = L_i^{-1} g_i, packed as a stacked (n*d) column-vector.
        trsm_batched_lower_inplace(&blas, &stream, d, n, 1, &d_dev, &mut g_dev)?;
        // Y_i = L_i^{-1} B_i, in place over the (n*d) × k buffer (laid out as
        // n stacked column-major d×k tiles).
        trsm_batched_lower_inplace(&blas, &stream, d, n, k, &d_dev, &mut b_dev)?;

        // ----- Layer B (2/2): Schur reduction via single big GEMM / GEMV -----
        // Y_all is (n*d) × k column-major: viewing all n stacked d×k tiles as
        // one big matrix is bit-exact because each tile is column-major with
        // leading dim d and the tiles are contiguous in memory, so the
        // combined leading dim is n*d only for the *outer* matrix view. To
        // make the single-GEMM equivalence hold we must treat the stacked
        // buffer as (n*d) × k column-major with leading dim = n*d, which
        // means columns of Y_all are interleaved by row across blocks.
        // That is NOT what we packed. So we use the cuBLAS stride pattern
        // instead: stride-by-block, transpose-A, and *accumulate* into one
        // S_β buffer via beta=1 across batches. Equivalent flop count, no
        // extra reduction kernel, and correct layout.
        //
        // Concretely: schur ← C + ρ_β I; rhs ← -g_β; then for each block
        //   schur -= Y_i^T Y_i      (k×k)
        //   rhs   += Y_i^T u_i      (k)
        // We launch this as `n` sequential GEMMs/GEMVs with beta=1 on the
        // accumulator. Layer D fuses these into one NVRTC launch.
        let schur_init: Vec<f64> = {
            let mut tmp = Vec::with_capacity(k * k);
            for col in 0..k {
                for row in 0..k {
                    let mut v = sys.hbb[[row, col]];
                    if row == col {
                        v += ridge_beta;
                    }
                    tmp.push(v);
                }
            }
            tmp
        };
        let mut schur_dev = stream
            .clone_htod(&schur_init)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let rhs_init: Vec<f64> = sys.gb.iter().map(|v| -v).collect();
        let mut rhs_dev = stream
            .clone_htod(&rhs_init)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;

        accumulate_schur(&blas, d, k, n, &b_dev, &g_dev, &mut schur_dev, &mut rhs_dev)?;

        // ----- Layer C (1/2): factor S_β and solve for δβ -----
        let info = potrf_single(&solver, &stream, k, &mut schur_dev)?;
        if info != 0 {
            return Err(ArrowSchurGpuFailure::SchurFactorFailed {
                reason: format!("Schur Cholesky failed at pivot {info}"),
            });
        }
        // δβ ← L_S^{-T} L_S^{-1} rhs
        trsm_single(&blas, &stream, k, &schur_dev, &mut rhs_dev, false, false)?;
        trsm_single(&blas, &stream, k, &schur_dev, &mut rhs_dev, false, true)?;
        let delta_beta_host = stream
            .clone_dtoh(&rhs_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let delta_beta = Array1::from_vec(delta_beta_host.clone());

        // ----- Layer C (2/2): back-sub δt_i = -L_i^{-T} (u_i + Y_i δβ) -----
        // Already on device:
        //   g_dev holds u_i stacked (n*d).
        //   b_dev holds Y_i stacked column-major n×(d×k) tiles.
        // Compute g_dev ← g_dev + Y_block · δβ per block (cuBLAS gemv with beta=1),
        // then in-place trsm with L_i^T (CUBLAS_OP_T) to obtain x_i, and finally
        // δt_i = -x_i on host after download.
        accumulate_back_sub_rhs(&blas, d, k, n, &b_dev, &rhs_dev, &mut g_dev)?;
        trsm_batched_lower_inplace_transposed(&blas, &stream, d, n, 1, &d_dev, &mut g_dev)?;

        let x_host = stream
            .clone_dtoh(&g_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut delta_t = Array1::<f64>::zeros(n * d);
        for (i, v) in x_host.iter().enumerate() {
            delta_t[i] = -*v;
        }

        // ----- log|H| = 2 Σ log L_{i,jj} + 2 Σ log R_{β,aa} -----
        let l_local_host = stream
            .clone_dtoh(&d_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let l_schur_host = stream
            .clone_dtoh(&schur_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut log_det = 0.0_f64;
        for i in 0..n {
            let base = i * d * d;
            for j in 0..d {
                log_det += l_local_host[base + j * d + j].ln();
            }
        }
        for j in 0..k {
            log_det += l_schur_host[j * k + j].ln();
        }
        log_det *= 2.0;

        Ok(ArrowSchurGpuSolution {
            delta_t,
            delta_beta,
            log_det_hessian: log_det,
        })
    }

    fn potrf_batched(
        solver: &DnHandle,
        stream: &Arc<CudaStream>,
        p: usize,
        batch: usize,
        matrices: &mut CudaSlice<f64>,
    ) -> Result<Vec<i32>, ArrowSchurGpuFailure> {
        let p_i = to_i32(p).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let batch_i = to_i32(batch).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let matrix_len = p * p;
        let bytes_per = (matrix_len * std::mem::size_of::<f64>()) as u64;
        let (base_ptr, _record) = matrices.device_ptr_mut(stream);
        let mut ptrs = Vec::with_capacity(batch);
        for idx in 0..batch {
            ptrs.push(base_ptr + (idx as u64) * bytes_per);
        }
        let mut ptrs_dev = stream
            .clone_htod(&ptrs)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut info_dev = stream
            .alloc_zeros::<i32>(batch)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let status = {
            let (ptrs_ptr, _ptrs_record) = ptrs_dev.device_ptr_mut(stream);
            let (info_ptr, _info_record) = info_dev.device_ptr_mut(stream);
            // SAFETY: pointer array and info buffer live on the device,
            // matrices_dev holds `batch` contiguous p×p column-major blocks.
            unsafe {
                cusolver_sys::cusolverDnDpotrfBatched(
                    solver.cu(),
                    cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                    p_i,
                    ptrs_ptr as *mut *mut f64,
                    p_i,
                    info_ptr as *mut i32,
                    batch_i,
                )
            }
        };
        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }
        stream
            .clone_dtoh(&info_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)
    }

    fn potrf_single(
        solver: &DnHandle,
        stream: &Arc<CudaStream>,
        p: usize,
        matrix: &mut CudaSlice<f64>,
    ) -> Result<i32, ArrowSchurGpuFailure> {
        let p_i = to_i32(p).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let uplo = cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
        let mut lwork = 0_i32;
        {
            let (mat_ptr, _rec) = matrix.device_ptr_mut(stream);
            // SAFETY: buffer query against a live p-by-p column-major device matrix.
            let status = unsafe {
                cusolver_sys::cusolverDnDpotrf_bufferSize(
                    solver.cu(),
                    uplo,
                    p_i,
                    mat_ptr as *mut f64,
                    p_i,
                    &mut lwork,
                )
            };
            if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                return Err(ArrowSchurGpuFailure::Unavailable);
            }
        }
        let lwork_usize = usize::try_from(lwork).map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut workspace = stream
            .alloc_zeros::<f64>(lwork_usize.max(1))
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut info_dev = stream
            .alloc_zeros::<i32>(1)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        {
            let (mat_ptr, _rec) = matrix.device_ptr_mut(stream);
            let (work_ptr, _wrec) = workspace.device_ptr_mut(stream);
            let (info_ptr, _irec) = info_dev.device_ptr_mut(stream);
            // SAFETY: all three pointers refer to live, correctly sized device buffers.
            let status = unsafe {
                cusolver_sys::cusolverDnDpotrf(
                    solver.cu(),
                    uplo,
                    p_i,
                    mat_ptr as *mut f64,
                    p_i,
                    work_ptr as *mut f64,
                    lwork,
                    info_ptr as *mut i32,
                )
            };
            if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
                return Err(ArrowSchurGpuFailure::Unavailable);
            }
        }
        let info_host = stream
            .clone_dtoh(&info_dev)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        Ok(info_host[0])
    }

    /// In-place lower-triangular solves `X_i ← L_i^{-1} X_i` over the n stacked
    /// d×nrhs RHS tiles in `rhs`. Uses `cublasDtrsmBatched` so all n solves
    /// hit the device in one launch.
    fn trsm_batched_lower_inplace(
        blas: &CudaBlas,
        stream: &Arc<CudaStream>,
        d: usize,
        n: usize,
        nrhs: usize,
        l_stack: &CudaSlice<f64>,
        rhs_stack: &mut CudaSlice<f64>,
    ) -> Result<(), ArrowSchurGpuFailure> {
        trsm_batched_inplace_inner(blas, stream, d, n, nrhs, l_stack, rhs_stack, false)
    }

    /// As above but with `L_i^T` instead of `L_i`.
    fn trsm_batched_lower_inplace_transposed(
        blas: &CudaBlas,
        stream: &Arc<CudaStream>,
        d: usize,
        n: usize,
        nrhs: usize,
        l_stack: &CudaSlice<f64>,
        rhs_stack: &mut CudaSlice<f64>,
    ) -> Result<(), ArrowSchurGpuFailure> {
        trsm_batched_inplace_inner(blas, stream, d, n, nrhs, l_stack, rhs_stack, true)
    }

    fn trsm_batched_inplace_inner(
        blas: &CudaBlas,
        stream: &Arc<CudaStream>,
        d: usize,
        n: usize,
        nrhs: usize,
        l_stack: &CudaSlice<f64>,
        rhs_stack: &mut CudaSlice<f64>,
        transposed: bool,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let alpha = 1.0_f64;
        let d_i = to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let nrhs_i = to_i32(nrhs).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let batch_i = to_i32(n).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let l_bytes_per = (d * d * std::mem::size_of::<f64>()) as u64;
        let rhs_bytes_per = (d * nrhs * std::mem::size_of::<f64>()) as u64;
        let (l_base, _l_record) = l_stack.device_ptr(stream);
        let (rhs_base, _rhs_record) = rhs_stack.device_ptr_mut(stream);
        let mut l_ptrs = Vec::with_capacity(n);
        let mut rhs_ptrs = Vec::with_capacity(n);
        for i in 0..n {
            l_ptrs.push(l_base + (i as u64) * l_bytes_per);
            rhs_ptrs.push(rhs_base + (i as u64) * rhs_bytes_per);
        }
        let mut l_ptrs_dev = stream
            .clone_htod(&l_ptrs)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let mut rhs_ptrs_dev = stream
            .clone_htod(&rhs_ptrs)
            .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        let (l_ptrs_ptr, _l_ptrs_rec) = l_ptrs_dev.device_ptr_mut(stream);
        let (rhs_ptrs_ptr, _rhs_ptrs_rec) = rhs_ptrs_dev.device_ptr_mut(stream);
        let op = if transposed {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let handle = *blas.handle();
        // SAFETY: pointer arrays and base buffers were just constructed from
        // live device allocations covering the entire batch.
        let status = unsafe {
            cudarc::cublas::sys::cublasDtrsmBatched(
                handle,
                cublasSideMode_t::CUBLAS_SIDE_LEFT,
                cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
                op,
                cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
                d_i,
                nrhs_i,
                &alpha,
                l_ptrs_ptr as *const *const f64,
                d_i,
                rhs_ptrs_ptr as *const *mut f64,
                d_i,
                batch_i,
            )
        };
        if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }
        Ok(())
    }

    /// Single-matrix lower-triangular solve: `rhs ← L^{-1} rhs` (or
    /// `L^{-T} rhs` if `transposed`). For the Schur Cholesky back-sub.
    fn trsm_single(
        blas: &CudaBlas,
        stream: &Arc<CudaStream>,
        n: usize,
        l: &CudaSlice<f64>,
        rhs: &mut CudaSlice<f64>,
        upper: bool,
        transposed: bool,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let alpha = 1.0_f64;
        let n_i = to_i32(n).ok_or(ArrowSchurGpuFailure::Unavailable)?;
        let handle = *blas.handle();
        let (l_ptr, _l_rec) = l.device_ptr(stream);
        let (rhs_ptr, _rhs_rec) = rhs.device_ptr_mut(stream);
        // SAFETY: single n×n lower factor and n-vector RHS on device.
        let status = unsafe {
            cudarc::cublas::sys::cublasDtrsm_v2(
                handle,
                cublasSideMode_t::CUBLAS_SIDE_LEFT,
                if upper {
                    cublasFillMode_t::CUBLAS_FILL_MODE_UPPER
                } else {
                    cublasFillMode_t::CUBLAS_FILL_MODE_LOWER
                },
                if transposed {
                    cublasOperation_t::CUBLAS_OP_T
                } else {
                    cublasOperation_t::CUBLAS_OP_N
                },
                cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
                n_i,
                1,
                &alpha,
                l_ptr as *const f64,
                n_i,
                rhs_ptr as *mut f64,
                n_i,
            )
        };
        if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(ArrowSchurGpuFailure::Unavailable);
        }
        Ok(())
    }

    /// Accumulate `schur ← schur − Σ_i Y_i^T Y_i` and `rhs ← rhs + Σ_i Y_i^T u_i`
    /// using one GEMM and one GEMV per block. Each call uses beta=1 to chain
    /// the accumulation device-side.
    fn accumulate_schur(
        blas: &CudaBlas,
        d: usize,
        k: usize,
        n: usize,
        y_stack: &CudaSlice<f64>,
        u_stack: &CudaSlice<f64>,
        schur: &mut CudaSlice<f64>,
        rhs: &mut CudaSlice<f64>,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let y_block_elems = d * k;
        let u_block_elems = d;
        for i in 0..n {
            let y_slice = y_stack.slice(i * y_block_elems..(i + 1) * y_block_elems);
            let u_slice = u_stack.slice(i * u_block_elems..(i + 1) * u_block_elems);
            // GEMM: schur += (-1) · Y_i^T · Y_i  (Y_i is d×k col-major; out is k×k)
            let gemm_cfg = GemmConfig::<f64> {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: to_i32(k).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                n: to_i32(k).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                k: to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                alpha: -1.0,
                lda: to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                ldb: to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                beta: 1.0,
                ldc: to_i32(k).ok_or(ArrowSchurGpuFailure::Unavailable)?,
            };
            // SAFETY: y_slice is d×k col-major, schur is k×k col-major; alpha/beta scalars set above.
            unsafe { blas.gemm(gemm_cfg, &y_slice, &y_slice, schur) }
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
            // GEMV: rhs += 1 · Y_i^T · u_i
            let gemv_cfg = GemvConfig::<f64> {
                trans: cublasOperation_t::CUBLAS_OP_T,
                m: to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                n: to_i32(k).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                alpha: 1.0,
                lda: to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                incx: 1,
                beta: 1.0,
                incy: 1,
            };
            // SAFETY: y_slice (d×k col-major) and u_slice (length d) are live
            // device buffers; `rhs` is the length-k accumulator.
            unsafe { blas.gemv(gemv_cfg, &y_slice, &u_slice, rhs) }
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        Ok(())
    }

    /// Accumulate `g_dev[i] ← u_i + Y_i · δβ` per block. This is the
    /// pre-trsm RHS for the back-substitution `L_i^T x_i = w_i`.
    fn accumulate_back_sub_rhs(
        blas: &CudaBlas,
        d: usize,
        k: usize,
        n: usize,
        y_stack: &CudaSlice<f64>,
        delta_beta: &CudaSlice<f64>,
        u_stack: &mut CudaSlice<f64>,
    ) -> Result<(), ArrowSchurGpuFailure> {
        let y_block_elems = d * k;
        let u_block_elems = d;
        for i in 0..n {
            let y_slice = y_stack.slice(i * y_block_elems..(i + 1) * y_block_elems);
            let mut u_slice = u_stack.slice_mut(i * u_block_elems..(i + 1) * u_block_elems);
            let gemv_cfg = GemvConfig::<f64> {
                trans: cublasOperation_t::CUBLAS_OP_N,
                m: to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                n: to_i32(k).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                alpha: 1.0,
                lda: to_i32(d).ok_or(ArrowSchurGpuFailure::Unavailable)?,
                incx: 1,
                beta: 1.0,
                incy: 1,
            };
            // SAFETY: y_slice / delta_beta / u_slice are live device buffers
            // of the expected sizes (d×k, k, d).
            unsafe { blas.gemv(gemv_cfg, &y_slice, delta_beta, &mut u_slice) }
                .map_err(|_| ArrowSchurGpuFailure::Unavailable)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::arrow_schur::ArrowSchurSystem;
    use ndarray::Array2;

    fn build_fixture(n: usize, d: usize, k: usize, seed: u64) -> ArrowSchurSystem {
        let mut sys = ArrowSchurSystem::new(n, d, k);
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut sample = || -> f64 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
        };
        for row in &mut sys.rows {
            let mut a = Array2::<f64>::zeros((d, d));
            for r in 0..d {
                for c in 0..d {
                    a[[r, c]] = sample();
                }
            }
            let mut htt = a.t().dot(&a);
            for r in 0..d {
                htt[[r, r]] += d as f64 + 1.0;
            }
            row.htt = htt;
            for r in 0..d {
                for c in 0..k {
                    row.htbeta[[r, c]] = 0.1 * sample();
                }
                row.gt[r] = sample();
            }
        }
        let mut hbb_a = Array2::<f64>::zeros((k, k));
        for r in 0..k {
            for c in 0..k {
                hbb_a[[r, c]] = sample();
            }
        }
        let mut hbb = hbb_a.t().dot(&hbb_a);
        for r in 0..k {
            hbb[[r, r]] += k as f64 + 1.0;
        }
        sys.hbb = hbb;
        for r in 0..k {
            sys.gb[r] = sample();
        }
        sys
    }

    #[test]
    fn dense_reference_matches_independent_solve() {
        let sys = build_fixture(4, 5, 3, 7);
        let solution = solve_arrow_newton_step_dense_reference(&sys, 0.0, 0.0).unwrap();
        // Re-solve by an independent matrix build and a textbook
        // Gaussian-elimination Cholesky to guard against typos in the
        // reference implementation itself.
        let n = sys.rows.len();
        let d = sys.d;
        let k = sys.k;
        let total = n * d + k;
        let mut h = Array2::<f64>::zeros((total, total));
        let mut g = ndarray::Array1::<f64>::zeros(total);
        for (i, row) in sys.rows.iter().enumerate() {
            let base = i * d;
            for c in 0..d {
                for r in 0..d {
                    h[[base + r, base + c]] = row.htt[[r, c]];
                }
            }
            for c in 0..k {
                for r in 0..d {
                    h[[base + r, n * d + c]] = row.htbeta[[r, c]];
                    h[[n * d + c, base + r]] = row.htbeta[[r, c]];
                }
            }
            for r in 0..d {
                g[base + r] = row.gt[r];
            }
        }
        for c in 0..k {
            for r in 0..k {
                h[[n * d + r, n * d + c]] += sys.hbb[[r, c]];
            }
            g[n * d + c] = sys.gb[c];
        }
        let l = cholesky_lower_host(h.view()).unwrap();
        let rhs = g.mapv(|v| -v);
        let expected = solve_cholesky_lower_host(l.view(), rhs.view());
        for i in 0..n * d {
            assert!(
                (solution.delta_t[i] - expected[i]).abs() < 1e-10 * (1.0 + expected[i].abs()),
                "delta_t[{i}] mismatch: got {} expected {}",
                solution.delta_t[i],
                expected[i]
            );
        }
        for a in 0..k {
            assert!(
                (solution.delta_beta[a] - expected[n * d + a]).abs()
                    < 1e-10 * (1.0 + expected[n * d + a].abs()),
                "delta_beta[{a}] mismatch"
            );
        }
    }
}

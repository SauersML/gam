//! cuSOLVER routing for large dense symmetric linear algebra.
//!
//! The active production surface is:
//!
//! * [`try_syevd_inplace`] — wired into [`crate::linalg::faer_ndarray::FaerEigh`]
//!   for large lower-triangle symmetric eigendecompositions.
//! * [`try_chol_solve_inplace`] — fused dense Cholesky factor + triangular
//!   solve (`A = LLᵀ`, then `A X = B`) in a single host↔device round-trip,
//!   matching the PIRLS Newton-direction solve and ridge-retry pattern.
//! * [`try_cholesky_lower_inplace`] — dense Cholesky factorization for callers
//!   that need the lower factor itself.
//!
//! Smaller matrices stay on faer's CPU path because the host/device round
//! trip dominates there. Thresholds live in the helper functions below.

use libloading::Library;
use ndarray::{Array1, Array2};
use std::ops::Range;
use std::sync::{Mutex, OnceLock};

use super::device::GpuDeviceInfo;
use super::diagnostics;
use super::driver::{
    CudaWorkingState, DeviceAllocation, bytes_len, check_cuda, from_col_major_inplace,
    load_static_library, to_col_major, to_i32,
};
use super::runtime::GpuRuntime;

/// In-place symmetric eigendecomposition: returns eigenvalues, `a` becomes
/// the orthonormal eigenvector matrix. `None` keeps execution on faer.
#[inline]
pub fn try_syevd_inplace(a: &mut Array2<f64>) -> Option<Array1<f64>> {
    let p = a.nrows();
    if p != a.ncols() {
        return None;
    }
    if !route_syevd(p) {
        diagnostics::log_policy_cpu(
            "syevd",
            format!("p={p}"),
            diagnostics::dispatch_decline_reason(format!(
                "below cuSOLVER policy threshold syevd_p>={}",
                GpuRuntime::global().policy().syevd_min_p
            )),
        );
        return None;
    }
    let start = std::time::Instant::now();
    match with_runtime(|rt| rt.syevd_inplace(a)) {
        Some((out, device)) => {
            diagnostics::log_gpu_success(
                "syevd",
                "cuSOLVER",
                &device,
                format!("p={p}"),
                (p as u64).saturating_mul(p as u64).saturating_mul(p as u64),
                diagnostics::bytes_for_f64(a.len()),
                diagnostics::bytes_for_f64(a.len().saturating_add(p)),
                start.elapsed().as_secs_f64(),
            );
            Some(out)
        }
        None => {
            diagnostics::log_runtime_cpu("syevd", "cuSOLVER", format!("p={p}"));
            None
        }
    }
}

/// Fused Cholesky factor + triangular solve: solve `A X = B` for SPD `A`,
/// writing the solution into `rhs`. Returns `Some(())` when cuSOLVER completes
/// the solve; `None` leaves the existing host solver path in charge.
///
/// `a` is consumed in-place because cuSOLVER overwrites the lower triangle
/// with `L` during factorization; callers that need the original matrix
/// must clone before calling.
#[inline]
pub fn try_chol_solve_inplace(a: &mut Array2<f64>, rhs: &mut Array2<f64>) -> Option<()> {
    let p = a.nrows();
    if p != a.ncols() || rhs.nrows() != p {
        return None;
    }
    if !route_chol_solve(p) {
        diagnostics::log_policy_cpu(
            "chol_solve",
            format!("p={p} rhs_cols={}", rhs.ncols()),
            diagnostics::dispatch_decline_reason(format!(
                "below cuSOLVER policy threshold chol_p>={}",
                GpuRuntime::global().policy().chol_min_p
            )),
        );
        return None;
    }
    let start = std::time::Instant::now();
    match with_runtime(|rt| rt.chol_solve_inplace(a, rhs)) {
        Some(((), device)) => {
            diagnostics::log_gpu_success(
                "chol_solve",
                "cuSOLVER",
                &device,
                format!("p={p} rhs_cols={}", rhs.ncols()),
                diagnostics::chol_flops(p).saturating_add(
                    (p as u64)
                        .saturating_mul(p as u64)
                        .saturating_mul(rhs.ncols() as u64),
                ),
                diagnostics::bytes_for_f64(a.len().saturating_add(rhs.len())),
                diagnostics::bytes_for_f64(a.len().saturating_add(rhs.len())),
                start.elapsed().as_secs_f64(),
            );
            Some(())
        }
        None => {
            diagnostics::log_runtime_cpu(
                "chol_solve",
                "cuSOLVER",
                format!("p={p} rhs_cols={}", rhs.ncols()),
            );
            None
        }
    }
}

/// Human-readable dispatch decision for the fused Cholesky solve path.
///
/// This is intentionally separate from `try_chol_solve_inplace`: callers that
/// emit their own high-level stage timing can include the same route reason in
/// the user-visible line even when lower-level GPU diagnostics are filtered.
pub fn describe_chol_solve_route(p: usize, rhs_cols: usize) -> String {
    let runtime = GpuRuntime::global();
    let shape = format!("p={p} rhs_cols={rhs_cols}");
    if let Some(reason) = runtime.cpu_reason() {
        return format!(
            "backend=CPU op=chol_solve shape={shape} reason=CUDA unavailable: {reason}"
        );
    }

    let policy = runtime.policy();
    if !policy.route_chol_solve(p) {
        return format!(
            "backend=CPU op=chol_solve shape={shape} reason=below cuSOLVER policy threshold chol_p>={}",
            policy.chol_min_p
        );
    }

    let device = runtime
        .selected_device()
        .map(ToString::to_string)
        .unwrap_or_else(|| "unknown CUDA device".to_string());
    format!(
        "backend=cuSOLVER op=chol_solve shape={shape} reason=meets GPU policy chol_p>={} selected_device={device}",
        policy.chol_min_p
    )
}

/// True when `try_chol_solve_inplace` is expected to attempt cuSOLVER for a
/// valid `p x p` input. A later runtime/library/factorization failure can
/// still fall back to the CPU route.
#[inline]
pub fn will_attempt_chol_solve(p: usize) -> bool {
    let runtime = GpuRuntime::global();
    runtime.is_available() && runtime.policy().route_chol_solve(p)
}

/// In-place lower Cholesky factorization: `a` becomes `L` for `A = L L^T`.
#[inline]
pub fn try_cholesky_lower_inplace(a: &mut Array2<f64>) -> Option<()> {
    let p = a.nrows();
    if p != a.ncols() {
        return None;
    }
    if !route_chol_solve(p) {
        diagnostics::log_policy_cpu(
            "cholesky_lower",
            format!("p={p}"),
            diagnostics::dispatch_decline_reason(format!(
                "below cuSOLVER policy threshold chol_p>={}",
                GpuRuntime::global().policy().chol_min_p
            )),
        );
        return None;
    }
    let start = std::time::Instant::now();
    match with_runtime(|rt| rt.cholesky_lower_inplace(a)) {
        Some(((), device)) => {
            diagnostics::log_gpu_success(
                "cholesky_lower",
                "cuSOLVER",
                &device,
                format!("p={p}"),
                diagnostics::chol_flops(p),
                diagnostics::bytes_for_f64(a.len()),
                diagnostics::bytes_for_f64(a.len()),
                start.elapsed().as_secs_f64(),
            );
            Some(())
        }
        None => {
            diagnostics::log_runtime_cpu("cholesky_lower", "cuSOLVER", format!("p={p}"));
            None
        }
    }
}

/// In-place batched lower Cholesky factorization: each `matrices[b]` becomes
/// `L_b` for `A_b = L_b L_b^T`. All matrices must be the same `p × p` shape;
/// the upper triangle of each input matrix is ignored on read and zeroed on
/// successful return so the result is exactly the lower factor with zeros
/// above the diagonal. Returns `None` (leaving every host matrix unchanged)
/// when any factor is not positive definite, the policy declines the route,
/// or any cuSOLVER/CUDA call fails — the caller then falls through to its CPU
/// fallback per matrix, matching the single-fit `try_cholesky_lower_inplace`
/// contract.
#[inline]
pub fn try_cholesky_batched_lower_inplace(matrices: &mut [Array2<f64>]) -> Option<()> {
    if matrices.is_empty() {
        return Some(());
    }
    let p = matrices[0].nrows();
    if p == 0 || p != matrices[0].ncols() {
        return None;
    }
    if !matrices.iter().all(|m| m.dim() == (p, p)) {
        return None;
    }
    if !route_chol_batched(p, matrices.len()) {
        diagnostics::log_policy_cpu(
            "cholesky_batched_lower",
            format!("batch={} p={p}", matrices.len()),
            diagnostics::dispatch_decline_reason(format!(
                "below cuSOLVER batched policy threshold aggregate_flops>={}",
                GpuRuntime::global().policy().gemm_min_flops
            )),
        );
        return None;
    }
    let batch = matrices.len();
    let start = std::time::Instant::now();
    match try_multi_cholesky_batched_lower_inplace(matrices).or_else(|| {
        with_runtime(|rt| rt.cholesky_batched_lower_inplace(matrices))
            .map(|((), device)| vec![device])
    }) {
        Some(devices) => {
            diagnostics::log_gpu_success_multi(
                "cholesky_batched_lower",
                "cuSOLVER",
                &devices,
                format!("batch={batch} p={p}"),
                diagnostics::chol_flops(p).saturating_mul(batch as u64),
                diagnostics::bytes_for_f64(batch.saturating_mul(p).saturating_mul(p)),
                diagnostics::bytes_for_f64(batch.saturating_mul(p).saturating_mul(p)),
                start.elapsed().as_secs_f64(),
            );
            Some(())
        }
        None => {
            diagnostics::log_runtime_cpu(
                "cholesky_batched_lower",
                "cuSOLVER",
                format!("batch={batch} p={p}"),
            );
            None
        }
    }
}

#[inline]
fn route_syevd(p: usize) -> bool {
    GpuRuntime::global().policy().route_syevd(p)
}

#[inline]
fn route_chol_solve(p: usize) -> bool {
    GpuRuntime::global().policy().route_chol_solve(p)
}

#[inline]
fn route_chol_batched(p: usize, batch_size: usize) -> bool {
    GpuRuntime::global()
        .policy()
        .route_chol_batched(p, batch_size)
}

fn with_runtime<T>(
    mut f: impl FnMut(&mut CusolverRuntime) -> Option<T>,
) -> Option<(T, GpuDeviceInfo)> {
    let runtimes = cusolver_runtimes();
    if runtimes.is_empty() {
        return None;
    }
    let start = GpuRuntime::global().next_runtime_slot(runtimes.len());
    // Phase 1: non-blocking — skip any device already driven by another
    // thread so concurrent callers spread to idle GPUs rather than
    // serializing on the rotated slot.
    for offset in 0..runtimes.len() {
        let idx = (start + offset) % runtimes.len();
        if let Ok(mut runtime) = runtimes[idx].try_lock()
            && let Some(out) = f(&mut runtime)
        {
            return Some((out, runtime.device.clone()));
        }
    }
    // Phase 2: every device was busy or every Phase-1 attempt compute-failed.
    // Block on each in turn so we still complete the dispatch.
    for offset in 0..runtimes.len() {
        let idx = (start + offset) % runtimes.len();
        if let Ok(mut runtime) = runtimes[idx].lock()
            && let Some(out) = f(&mut runtime)
        {
            return Some((out, runtime.device.clone()));
        }
    }
    None
}

fn cusolver_runtimes() -> &'static [Mutex<CusolverRuntime>] {
    static RUNTIME: OnceLock<Vec<Mutex<CusolverRuntime>>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            GpuRuntime::global()
                .devices()
                .iter()
                .filter_map(|device| {
                    let cuda = match CudaWorkingState::init(device.ordinal) {
                        Some(cuda) => cuda,
                        None => {
                            diagnostics::log_library_unavailable(
                                "cuSOLVER",
                                &format!("CUDA context init failed for device {}", device.ordinal),
                            );
                            return None;
                        }
                    };
                    match CusolverRuntime::new(cuda, device.clone()) {
                        Ok(runtime) => {
                            diagnostics::log_library_ready("cuSOLVER", &runtime.device);
                            Some(Mutex::new(runtime))
                        }
                        Err(err) => {
                            diagnostics::log_library_unavailable("cuSOLVER", &err);
                            None
                        }
                    }
                })
                .collect()
        })
        .as_slice()
}

fn plan_for_cusolver(
    batch_size: usize,
    fixed_bytes_per_device: usize,
    bytes_per_batch_item: usize,
) -> Option<Vec<(usize, GpuDeviceInfo, Vec<Range<usize>>)>> {
    let runtimes = cusolver_runtimes();
    if runtimes.len() <= 1 {
        return None;
    }
    let mut devices = Vec::with_capacity(runtimes.len());
    for runtime in runtimes {
        devices.push(runtime.lock().ok()?.device.clone());
    }
    let plans = GpuRuntime::global().plan_batched_work_for_devices(
        &devices,
        batch_size,
        fixed_bytes_per_device,
        bytes_per_batch_item,
    )?;
    if plans.len() <= 1 {
        return None;
    }
    let mut mapped = Vec::with_capacity(plans.len());
    for plan in plans {
        let idx = devices
            .iter()
            .position(|device| device.ordinal == plan.ordinal)?;
        mapped.push((idx, devices[idx].clone(), plan.chunks));
    }
    Some(mapped)
}

fn try_multi_cholesky_batched_lower_inplace(
    matrices: &mut [Array2<f64>],
) -> Option<Vec<GpuDeviceInfo>> {
    let batch = matrices.len();
    if batch == 0 {
        return Some(Vec::new());
    }
    let p = matrices[0].nrows();
    let matrix_bytes = diagnostics::bytes_for_f64(p.checked_mul(p)?);
    let bytes_per_batch_item = matrix_bytes
        .saturating_add(std::mem::size_of::<u64>())
        .saturating_add(std::mem::size_of::<i32>());
    let plan = plan_for_cusolver(batch, 0, bytes_per_batch_item)?;
    let runtimes = cusolver_runtimes();
    let pieces = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(plan.len());
        for (runtime_idx, device, chunks) in plan {
            let chunk_inputs = chunks
                .into_iter()
                .map(|range| {
                    let local = matrices[range.clone()].to_vec();
                    (range, local)
                })
                .collect::<Vec<_>>();
            handles.push(scope.spawn(move || {
                let mut solved_chunks = Vec::with_capacity(chunk_inputs.len());
                let mut runtime = runtimes[runtime_idx].lock().ok()?;
                for (range, mut local) in chunk_inputs {
                    runtime.cholesky_batched_lower_inplace(&mut local)?;
                    solved_chunks.push((range, local));
                }
                Some((device, solved_chunks))
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().ok()?)
            .collect::<Option<Vec<_>>>()
    })?;

    let mut devices = Vec::with_capacity(pieces.len());
    for (device, chunks) in pieces {
        devices.push(device);
        for (range, solved) in chunks {
            if solved.len() != range.end - range.start {
                return None;
            }
            for (dst, src) in matrices[range].iter_mut().zip(solved) {
                *dst = src;
            }
        }
    }
    Some(devices)
}

struct CusolverRuntime {
    cuda: CudaWorkingState,
    device: GpuDeviceInfo,
    /// cuSOLVER entry points; the dlopen'd library is leaked into a
    /// `&'static` so these pointers stay valid for the process.
    solver: CusolverApi,
    handle: usize,
}

impl CusolverRuntime {
    fn new(cuda: CudaWorkingState, device: GpuDeviceInfo) -> Result<Self, String> {
        let solver_lib = load_static_library(cusolver_library_candidates())?;
        let solver = CusolverApi::load(solver_lib)?;
        cuda.set_current()?;
        let mut handle = 0_usize;
        let status = unsafe { (solver.create)(&mut handle) };
        if status != CUSOLVER_STATUS_SUCCESS {
            return Err(format!("cusolverDnCreate failed with status {status}"));
        }
        Ok(Self {
            cuda,
            device,
            solver,
            handle,
        })
    }

    fn syevd_inplace(&mut self, a: &mut Array2<f64>) -> Option<Array1<f64>> {
        let p = a.nrows();
        let p_i32 = to_i32(p)?;
        let mut a_col = to_col_major(a);
        let mut eigs = vec![0.0_f64; p];
        let bytes_a = bytes_len::<f64>(a_col.len())?;
        let bytes_eigs = bytes_len::<f64>(p)?;
        let bytes_info = bytes_len::<i32>(1)?;

        unsafe {
            self.cuda.set_current().ok()?;
            let a_dev = self.alloc_copy(&a_col, bytes_a)?;
            let eigs_dev = DeviceAllocation::new(&self.cuda.api, bytes_eigs)?;
            let info_dev = DeviceAllocation::new(&self.cuda.api, bytes_info)?;
            let mut work_size: i32 = 0;
            if (self.solver.dsyevd_buffersize)(
                self.handle,
                CUSOLVER_EIG_MODE_VECTORS,
                CUBLAS_FILL_LOWER,
                p_i32,
                a_dev.ptr,
                p_i32,
                eigs_dev.ptr,
                &mut work_size,
            ) != CUSOLVER_STATUS_SUCCESS
            {
                return None;
            }
            let scratch = if work_size > 0 {
                let work_elems = usize::try_from(work_size).ok()?;
                Some(DeviceAllocation::new(
                    &self.cuda.api,
                    bytes_len::<f64>(work_elems)?,
                )?)
            } else {
                None
            };
            let scratch_ptr = scratch.as_ref().map(|s| s.ptr).unwrap_or(0);
            let status = (self.solver.dsyevd)(
                self.handle,
                CUSOLVER_EIG_MODE_VECTORS,
                CUBLAS_FILL_LOWER,
                p_i32,
                a_dev.ptr,
                p_i32,
                eigs_dev.ptr,
                scratch_ptr,
                work_size,
                info_dev.ptr,
            );
            if status != CUSOLVER_STATUS_SUCCESS {
                return None;
            }
            let mut info: i32 = 0;
            check_cuda(
                (self.cuda.api.cu_memcpy_dtoh)(
                    (&mut info) as *mut i32 as *mut std::ffi::c_void,
                    info_dev.ptr,
                    bytes_info,
                ),
                "cuMemcpyDtoH info",
            )
            .ok()?;
            if info != 0 {
                return None;
            }
            check_cuda(
                (self.cuda.api.cu_memcpy_dtoh)(a_col.as_mut_ptr().cast(), a_dev.ptr, bytes_a),
                "cuMemcpyDtoH A",
            )
            .ok()?;
            check_cuda(
                (self.cuda.api.cu_memcpy_dtoh)(eigs.as_mut_ptr().cast(), eigs_dev.ptr, bytes_eigs),
                "cuMemcpyDtoH eigs",
            )
            .ok()?;
        }
        from_col_major_inplace(&a_col, a);
        Some(Array1::from_vec(eigs))
    }

    /// Fused Cholesky factor + solve: `A` is overwritten with `L`, `rhs`
    /// with `A⁻¹·rhs`. One device round trip; returns `None` on any failure
    /// (factor not PD, allocation/copy error, library status non-zero), leaving
    /// faer's host solver path responsible for the result.
    fn chol_solve_inplace(&mut self, a: &mut Array2<f64>, rhs: &mut Array2<f64>) -> Option<()> {
        let p = a.nrows();
        let nrhs = rhs.ncols();
        let p_i32 = to_i32(p)?;
        let nrhs_i32 = to_i32(nrhs)?;
        let mut a_col = to_col_major(a);
        let mut rhs_col = to_col_major(rhs);
        let bytes_a = bytes_len::<f64>(a_col.len())?;
        let bytes_rhs = bytes_len::<f64>(rhs_col.len())?;
        let bytes_info = bytes_len::<i32>(1)?;

        unsafe {
            self.cuda.set_current().ok()?;
            let a_dev = self.alloc_copy(&a_col, bytes_a)?;
            let rhs_dev = self.alloc_copy(&rhs_col, bytes_rhs)?;
            let info_dev = DeviceAllocation::new(&self.cuda.api, bytes_info)?;
            let mut work_size: i32 = 0;
            if (self.solver.dpotrf_buffersize)(
                self.handle,
                CUBLAS_FILL_LOWER,
                p_i32,
                a_dev.ptr,
                p_i32,
                &mut work_size,
            ) != CUSOLVER_STATUS_SUCCESS
            {
                return None;
            }
            let scratch = if work_size > 0 {
                let work_elems = usize::try_from(work_size).ok()?;
                Some(DeviceAllocation::new(
                    &self.cuda.api,
                    bytes_len::<f64>(work_elems)?,
                )?)
            } else {
                None
            };
            let scratch_ptr = scratch.as_ref().map(|s| s.ptr).unwrap_or(0);
            let factor_status = (self.solver.dpotrf)(
                self.handle,
                CUBLAS_FILL_LOWER,
                p_i32,
                a_dev.ptr,
                p_i32,
                scratch_ptr,
                work_size,
                info_dev.ptr,
            );
            if factor_status != CUSOLVER_STATUS_SUCCESS {
                return None;
            }
            let mut info: i32 = 0;
            check_cuda(
                (self.cuda.api.cu_memcpy_dtoh)(
                    (&mut info) as *mut i32 as *mut std::ffi::c_void,
                    info_dev.ptr,
                    bytes_info,
                ),
                "cuMemcpyDtoH potrf info",
            )
            .ok()?;
            if info != 0 {
                return None;
            }
            let solve_status = (self.solver.dpotrs)(
                self.handle,
                CUBLAS_FILL_LOWER,
                p_i32,
                nrhs_i32,
                a_dev.ptr,
                p_i32,
                rhs_dev.ptr,
                p_i32,
                info_dev.ptr,
            );
            if solve_status != CUSOLVER_STATUS_SUCCESS {
                return None;
            }
            check_cuda(
                (self.cuda.api.cu_memcpy_dtoh)(
                    (&mut info) as *mut i32 as *mut std::ffi::c_void,
                    info_dev.ptr,
                    bytes_info,
                ),
                "cuMemcpyDtoH potrs info",
            )
            .ok()?;
            if info != 0 {
                return None;
            }
            check_cuda(
                (self.cuda.api.cu_memcpy_dtoh)(a_col.as_mut_ptr().cast(), a_dev.ptr, bytes_a),
                "cuMemcpyDtoH A",
            )
            .ok()?;
            check_cuda(
                (self.cuda.api.cu_memcpy_dtoh)(rhs_col.as_mut_ptr().cast(), rhs_dev.ptr, bytes_rhs),
                "cuMemcpyDtoH rhs",
            )
            .ok()?;
        }
        from_col_major_inplace(&a_col, a);
        from_col_major_inplace(&rhs_col, rhs);
        Some(())
    }

    fn cholesky_lower_inplace(&mut self, a: &mut Array2<f64>) -> Option<()> {
        let p = a.nrows();
        let p_i32 = to_i32(p)?;
        let mut a_col = to_col_major(a);
        let bytes_a = bytes_len::<f64>(a_col.len())?;
        let bytes_info = bytes_len::<i32>(1)?;

        unsafe {
            self.cuda.set_current().ok()?;
            let a_dev = self.alloc_copy(&a_col, bytes_a)?;
            let info_dev = DeviceAllocation::new(&self.cuda.api, bytes_info)?;
            let mut work_size: i32 = 0;
            if (self.solver.dpotrf_buffersize)(
                self.handle,
                CUBLAS_FILL_LOWER,
                p_i32,
                a_dev.ptr,
                p_i32,
                &mut work_size,
            ) != CUSOLVER_STATUS_SUCCESS
            {
                return None;
            }
            let scratch = if work_size > 0 {
                let work_elems = usize::try_from(work_size).ok()?;
                Some(DeviceAllocation::new(
                    &self.cuda.api,
                    bytes_len::<f64>(work_elems)?,
                )?)
            } else {
                None
            };
            let scratch_ptr = scratch.as_ref().map(|s| s.ptr).unwrap_or(0);
            let factor_status = (self.solver.dpotrf)(
                self.handle,
                CUBLAS_FILL_LOWER,
                p_i32,
                a_dev.ptr,
                p_i32,
                scratch_ptr,
                work_size,
                info_dev.ptr,
            );
            if factor_status != CUSOLVER_STATUS_SUCCESS {
                return None;
            }
            let mut info: i32 = 0;
            check_cuda(
                (self.cuda.api.cu_memcpy_dtoh)(
                    (&mut info) as *mut i32 as *mut std::ffi::c_void,
                    info_dev.ptr,
                    bytes_info,
                ),
                "cuMemcpyDtoH potrf info",
            )
            .ok()?;
            if info != 0 {
                return None;
            }
            check_cuda(
                (self.cuda.api.cu_memcpy_dtoh)(a_col.as_mut_ptr().cast(), a_dev.ptr, bytes_a),
                "cuMemcpyDtoH A",
            )
            .ok()?;
        }
        from_col_major_inplace(&a_col, a);
        for row in 0..p {
            for col in (row + 1)..p {
                a[[row, col]] = 0.0;
            }
        }
        Some(())
    }

    /// Single host↔device round trip Cholesky over `K` uniform `p × p`
    /// matrices via `cusolverDnDpotrfBatched`. The batched variant is
    /// asymptotically `O(K)` cuSOLVER kernel launches vs. `K` separate calls,
    /// removing the per-fit dispatch latency that dominates at biobank-scale
    /// K=16k workloads. Returns `None` on any factor that is not positive
    /// definite, leaving every host matrix unchanged so the caller can fall
    /// through to a per-matrix host Cholesky.
    fn cholesky_batched_lower_inplace(&mut self, matrices: &mut [Array2<f64>]) -> Option<()> {
        let k = matrices.len();
        if k == 0 {
            return Some(());
        }
        let p = matrices[0].nrows();
        let p_i32 = to_i32(p)?;
        let k_i32 = to_i32(k)?;
        let entries_per_matrix = p.checked_mul(p)?;
        let total_doubles = k.checked_mul(entries_per_matrix)?;
        let bytes_all = bytes_len::<f64>(total_doubles)?;
        let bytes_ptrs = bytes_len::<u64>(k)?;
        let bytes_info = bytes_len::<i32>(k)?;
        let elem_bytes = std::mem::size_of::<f64>();

        let mut packed: Vec<f64> = Vec::with_capacity(total_doubles);
        for matrix in matrices.iter() {
            packed.extend(to_col_major(matrix));
        }
        debug_assert_eq!(packed.len(), total_doubles);

        unsafe {
            self.cuda.set_current().ok()?;
            let a_dev = self.alloc_copy(&packed, bytes_all)?;
            let info_dev = DeviceAllocation::new(&self.cuda.api, bytes_info)?;
            let ptr_dev = DeviceAllocation::new(&self.cuda.api, bytes_ptrs)?;
            let stride_bytes = entries_per_matrix.checked_mul(elem_bytes)?;
            let mut ptr_host: Vec<u64> = Vec::with_capacity(k);
            for batch in 0..k {
                let offset_bytes = batch.checked_mul(stride_bytes)?;
                let offset_u64 = u64::try_from(offset_bytes).ok()?;
                ptr_host.push(a_dev.ptr.checked_add(offset_u64)?);
            }
            check_cuda(
                (self.cuda.api.cu_memcpy_htod)(ptr_dev.ptr, ptr_host.as_ptr().cast(), bytes_ptrs),
                "cuMemcpyHtoD batched pointer table",
            )
            .ok()?;
            let status = (self.solver.dpotrf_batched)(
                self.handle,
                CUBLAS_FILL_LOWER,
                p_i32,
                ptr_dev.ptr,
                p_i32,
                info_dev.ptr,
                k_i32,
            );
            if status != CUSOLVER_STATUS_SUCCESS {
                return None;
            }
            let mut info_host = vec![0i32; k];
            check_cuda(
                (self.cuda.api.cu_memcpy_dtoh)(
                    info_host.as_mut_ptr().cast(),
                    info_dev.ptr,
                    bytes_info,
                ),
                "cuMemcpyDtoH batched potrf info",
            )
            .ok()?;
            if info_host.iter().any(|&code| code != 0) {
                return None;
            }
            check_cuda(
                (self.cuda.api.cu_memcpy_dtoh)(packed.as_mut_ptr().cast(), a_dev.ptr, bytes_all),
                "cuMemcpyDtoH batched A",
            )
            .ok()?;
        }

        for (batch, matrix) in matrices.iter_mut().enumerate() {
            let start = batch * entries_per_matrix;
            let slab = &packed[start..start + entries_per_matrix];
            from_col_major_inplace(slab, matrix);
            for row in 0..p {
                for col in (row + 1)..p {
                    matrix[[row, col]] = 0.0;
                }
            }
        }
        Some(())
    }

    unsafe fn alloc_copy<'a>(
        &'a self,
        values: &[f64],
        bytes: usize,
    ) -> Option<DeviceAllocation<'a>> {
        let alloc = unsafe { DeviceAllocation::new(&self.cuda.api, bytes) }?;
        check_cuda(
            unsafe { (self.cuda.api.cu_memcpy_htod)(alloc.ptr, values.as_ptr().cast(), bytes) },
            "cuMemcpyHtoD",
        )
        .ok()?;
        Some(alloc)
    }
}

impl Drop for CusolverRuntime {
    fn drop(&mut self) {
        unsafe {
            let _ = self.cuda.set_current();
            let _ = (self.solver.destroy)(self.handle);
        }
    }
}

type CusolverStatus = i32;
type CusolverCreate = unsafe extern "C" fn(*mut usize) -> CusolverStatus;
type CusolverDestroy = unsafe extern "C" fn(usize) -> CusolverStatus;
type CusolverDsyevdBufferSize = unsafe extern "C" fn(
    usize,    // handle
    i32,      // jobz
    i32,      // uplo
    i32,      // n
    u64,      // A
    i32,      // lda
    u64,      // W
    *mut i32, // lwork
) -> CusolverStatus;
type CusolverDsyevd = unsafe extern "C" fn(
    usize, // handle
    i32,   // jobz
    i32,   // uplo
    i32,   // n
    u64,   // A
    i32,   // lda
    u64,   // W
    u64,   // workspace
    i32,   // lwork
    u64,   // devInfo
) -> CusolverStatus;
type CusolverDpotrfBufferSize = unsafe extern "C" fn(
    usize,    // handle
    i32,      // uplo
    i32,      // n
    u64,      // A
    i32,      // lda
    *mut i32, // lwork
) -> CusolverStatus;
type CusolverDpotrf = unsafe extern "C" fn(
    usize, // handle
    i32,   // uplo
    i32,   // n
    u64,   // A
    i32,   // lda
    u64,   // workspace
    i32,   // lwork
    u64,   // devInfo
) -> CusolverStatus;
type CusolverDpotrs = unsafe extern "C" fn(
    usize, // handle
    i32,   // uplo
    i32,   // n
    i32,   // nrhs
    u64,   // A
    i32,   // lda
    u64,   // B
    i32,   // ldb
    u64,   // devInfo
) -> CusolverStatus;
type CusolverDpotrfBatched = unsafe extern "C" fn(
    usize, // handle
    i32,   // uplo
    i32,   // n
    u64,   // Aarray: double** in device memory (batchSize device pointers)
    i32,   // lda
    u64,   // infoArray: i32* on device, length batchSize
    i32,   // batchSize
) -> CusolverStatus;

struct CusolverApi {
    create: CusolverCreate,
    destroy: CusolverDestroy,
    dsyevd_buffersize: CusolverDsyevdBufferSize,
    dsyevd: CusolverDsyevd,
    dpotrf_buffersize: CusolverDpotrfBufferSize,
    dpotrf: CusolverDpotrf,
    dpotrs: CusolverDpotrs,
    dpotrf_batched: CusolverDpotrfBatched,
}

impl CusolverApi {
    fn load(library: &Library) -> Result<Self, String> {
        unsafe {
            Ok(Self {
                create: *library
                    .get(b"cusolverDnCreate\0")
                    .map_err(|e| e.to_string())?,
                destroy: *library
                    .get(b"cusolverDnDestroy\0")
                    .map_err(|e| e.to_string())?,
                dsyevd_buffersize: *library
                    .get(b"cusolverDnDsyevd_bufferSize\0")
                    .map_err(|e| e.to_string())?,
                dsyevd: *library
                    .get(b"cusolverDnDsyevd\0")
                    .map_err(|e| e.to_string())?,
                dpotrf_buffersize: *library
                    .get(b"cusolverDnDpotrf_bufferSize\0")
                    .map_err(|e| e.to_string())?,
                dpotrf: *library
                    .get(b"cusolverDnDpotrf\0")
                    .map_err(|e| e.to_string())?,
                dpotrs: *library
                    .get(b"cusolverDnDpotrs\0")
                    .map_err(|e| e.to_string())?,
                dpotrf_batched: *library
                    .get(b"cusolverDnDpotrfBatched\0")
                    .map_err(|e| e.to_string())?,
            })
        }
    }
}

const CUSOLVER_STATUS_SUCCESS: CusolverStatus = 0;
const CUBLAS_FILL_LOWER: i32 = 0;
const CUSOLVER_EIG_MODE_VECTORS: i32 = 1;

fn cusolver_library_candidates() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["cusolver64_12.dll", "cusolver64_11.dll"]
    } else if cfg!(target_os = "macos") {
        &["/usr/local/cuda/lib/libcusolver.dylib", "libcusolver.dylib"]
    } else {
        &["libcusolver.so.12", "libcusolver.so.11", "libcusolver.so"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn small_matrices_do_not_route_to_gpu() {
        let mut a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        assert!(try_syevd_inplace(&mut a).is_none());
    }

    #[test]
    fn col_major_round_trip_preserves_layout() {
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let packed = to_col_major(&a);
        assert_eq!(packed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let mut back = Array2::<f64>::zeros((2, 3));
        from_col_major_inplace(&packed, &mut back);
        assert_eq!(back, a);
    }
}

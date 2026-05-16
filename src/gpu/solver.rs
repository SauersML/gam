//! cuSOLVER routing for large dense symmetric linear algebra.
//!
//! The active production surface is:
//!
//! * [`try_syevd_inplace`] — wired into [`crate::linalg::faer_ndarray::FaerEigh`]
//!   for large lower-triangle symmetric eigendecompositions.
//! * [`try_chol_solve_inplace`] — fused dense Cholesky factor + triangular
//!   solve (`A = LLᵀ`, then `A X = B`) in a single host↔device round-trip,
//!   matching the PIRLS Newton-direction solve and ridge-retry pattern.
//!
//! Smaller matrices stay on faer's CPU path because the host/device round
//! trip dominates there. Thresholds live in the helper functions below.

use libloading::Library;
use ndarray::{Array1, Array2};
use std::sync::{Mutex, OnceLock};

use super::driver::{
    DeviceAllocation, DriverApi, bytes_len, check_cuda, cuda_library_candidates,
    from_col_major_inplace, load_library, to_col_major, to_i32,
};
use super::runtime::GpuRuntime;

/// In-place symmetric eigendecomposition: returns eigenvalues, `a` becomes
/// the orthonormal eigenvector matrix. `None` keeps execution on faer.
#[inline]
pub fn try_syevd_inplace(a: &mut Array2<f64>) -> Option<Array1<f64>> {
    let p = a.nrows();
    if p != a.ncols() || !route_syevd(p) {
        return None;
    }
    with_runtime(|rt| rt.syevd_inplace(a))
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
    if p != a.ncols() || rhs.nrows() != p || !route_chol_solve(p) {
        return None;
    }
    with_runtime(|rt| rt.chol_solve_inplace(a, rhs))
}

#[inline]
fn route_syevd(p: usize) -> bool {
    GpuRuntime::global().policy().route_syevd(p)
}

#[inline]
fn route_chol_solve(p: usize) -> bool {
    GpuRuntime::global().policy().route_chol_solve(p)
}

fn with_runtime<T>(f: impl FnOnce(&mut CusolverRuntime) -> Option<T>) -> Option<T> {
    static RUNTIME: OnceLock<Option<Mutex<CusolverRuntime>>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| CusolverRuntime::new().ok().map(Mutex::new))
        .as_ref()?
        .lock()
        .ok()
        .and_then(|mut rt| f(&mut rt))
}

struct CusolverRuntime {
    _cuda_lib: Library,
    _cusolver_lib: Library,
    driver: DriverApi,
    solver: CusolverApi,
    context: usize,
    handle: usize,
}

impl CusolverRuntime {
    fn new() -> Result<Self, String> {
        let selected = GpuRuntime::global()
            .selected_device()
            .ok_or_else(|| "no CUDA device selected".to_string())?;
        let cuda_lib = load_library(cuda_library_candidates())?;
        let solver_lib = load_library(cusolver_library_candidates())?;
        let driver = DriverApi::load(&cuda_lib)?;
        let solver = CusolverApi::load(&solver_lib)?;
        unsafe {
            check_cuda((driver.cu_init)(0), "cuInit")?;
            let mut device = 0;
            let ordinal = to_i32(selected.ordinal)
                .ok_or_else(|| "CUDA device ordinal exceeds i32".to_string())?;
            check_cuda((driver.cu_device_get)(&mut device, ordinal), "cuDeviceGet")?;
            let mut context = 0usize;
            check_cuda(
                (driver.cu_ctx_create)(&mut context, 0, device),
                "cuCtxCreate",
            )?;
            let mut handle = 0usize;
            let status = (solver.create)(&mut handle);
            if status != CUSOLVER_STATUS_SUCCESS {
                let _ = (driver.cu_ctx_destroy)(context);
                return Err(format!("cusolverDnCreate failed with status {status}"));
            }
            Ok(Self {
                _cuda_lib: cuda_lib,
                _cusolver_lib: solver_lib,
                driver,
                solver,
                context,
                handle,
            })
        }
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
            self.set_current().ok()?;
            let a_dev = self.alloc_copy(&a_col, bytes_a)?;
            let eigs_dev = DeviceAllocation::new(&self.driver, bytes_eigs)?;
            let info_dev = DeviceAllocation::new(&self.driver, bytes_info)?;
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
                    &self.driver,
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
                (self.driver.cu_memcpy_dtoh)(
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
                (self.driver.cu_memcpy_dtoh)(a_col.as_mut_ptr().cast(), a_dev.ptr, bytes_a),
                "cuMemcpyDtoH A",
            )
            .ok()?;
            check_cuda(
                (self.driver.cu_memcpy_dtoh)(eigs.as_mut_ptr().cast(), eigs_dev.ptr, bytes_eigs),
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
            self.set_current().ok()?;
            let a_dev = self.alloc_copy(&a_col, bytes_a)?;
            let rhs_dev = self.alloc_copy(&rhs_col, bytes_rhs)?;
            let info_dev = DeviceAllocation::new(&self.driver, bytes_info)?;
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
                    &self.driver,
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
                (self.driver.cu_memcpy_dtoh)(
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
                (self.driver.cu_memcpy_dtoh)(
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
                (self.driver.cu_memcpy_dtoh)(a_col.as_mut_ptr().cast(), a_dev.ptr, bytes_a),
                "cuMemcpyDtoH A",
            )
            .ok()?;
            check_cuda(
                (self.driver.cu_memcpy_dtoh)(rhs_col.as_mut_ptr().cast(), rhs_dev.ptr, bytes_rhs),
                "cuMemcpyDtoH rhs",
            )
            .ok()?;
        }
        from_col_major_inplace(&a_col, a);
        from_col_major_inplace(&rhs_col, rhs);
        Some(())
    }

    unsafe fn set_current(&self) -> Result<(), String> {
        check_cuda(
            unsafe { (self.driver.cu_ctx_set_current)(self.context) },
            "cuCtxSetCurrent",
        )
    }

    unsafe fn alloc_copy<'a>(
        &'a self,
        values: &[f64],
        bytes: usize,
    ) -> Option<DeviceAllocation<'a>> {
        let alloc = unsafe { DeviceAllocation::new(&self.driver, bytes) }?;
        check_cuda(
            unsafe { (self.driver.cu_memcpy_htod)(alloc.ptr, values.as_ptr().cast(), bytes) },
            "cuMemcpyHtoD",
        )
        .ok()?;
        Some(alloc)
    }
}

impl Drop for CusolverRuntime {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.cu_ctx_set_current)(self.context);
            let _ = (self.solver.destroy)(self.handle);
            let _ = (self.driver.cu_ctx_destroy)(self.context);
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

struct CusolverApi {
    create: CusolverCreate,
    destroy: CusolverDestroy,
    dsyevd_buffersize: CusolverDsyevdBufferSize,
    dsyevd: CusolverDsyevd,
    dpotrf_buffersize: CusolverDpotrfBufferSize,
    dpotrf: CusolverDpotrf,
    dpotrs: CusolverDpotrs,
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

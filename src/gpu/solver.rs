//! cuSOLVER routing for dense linear-algebra factorizations.
//!
//! The dispatch surface mirrors faer's local CPU calls used inside PIRLS:
//!
//! * `try_potrf_inplace`: in-place Cholesky `A = LLᵀ` of a `p×p` SPD matrix.
//! * `try_potrs_inplace`: triangular solves against a previously factored `A`.
//! * `try_syevd_inplace`: symmetric eigendecomposition `A = QΛQᵀ`.
//!
//! The size thresholds are conservative — host Cholesky on faer is extremely
//! fast for `p ≤ 256` and the PCIe round-trip eats the win. Above `p ≈ 1024`
//! cuSOLVER dominates.

use libloading::Library;
use ndarray::{Array1, Array2};
use std::sync::{Mutex, OnceLock};

use super::runtime::GpuRuntime;

/// In-place dense Cholesky `A = LLᵀ` (lower triangular factor stored in `A`).
/// Returns `Some(())` when the device path produced the factorization,
/// `None` to fall back to faer.
#[inline]
pub fn try_potrf_inplace(a: &mut Array2<f64>) -> Option<()> {
    let p = a.nrows();
    if p != a.ncols() || !route_potrf(p) {
        return None;
    }
    with_runtime(|rt| rt.potrf_inplace(a))
}

/// Solve `A X = B` in place on `B`, reusing a previously computed `A = LLᵀ`
/// in `factor`. Returns `Some(())` on device success.
#[inline]
pub fn try_potrs_inplace(factor: &Array2<f64>, rhs: &mut Array2<f64>) -> Option<()> {
    let p = factor.nrows();
    if p != factor.ncols() || rhs.nrows() != p || !route_potrf(p) {
        return None;
    }
    with_runtime(|rt| rt.potrs_inplace(factor, rhs))
}

/// In-place symmetric eigendecomposition: returns eigenvalues, `a` becomes
/// the orthonormal eigenvector matrix. `None` falls back to faer.
#[inline]
pub fn try_syevd_inplace(a: &mut Array2<f64>) -> Option<Array1<f64>> {
    let p = a.nrows();
    if p != a.ncols() || !route_syevd(p) {
        return None;
    }
    with_runtime(|rt| rt.syevd_inplace(a))
}

#[inline]
fn route_potrf(p: usize) -> bool {
    // Calibrated against PCIe Gen4 host↔device transfer of a p×p double-precision
    // matrix plus cuSOLVER context setup; below 512 the host BLAS path wins.
    p >= 512
}

#[inline]
fn route_syevd(p: usize) -> bool {
    p >= 256
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
            check_cuda(
                (driver.cu_device_get)(&mut device, selected.ordinal as i32),
                "cuDeviceGet",
            )?;
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

    fn potrf_inplace(&mut self, a: &mut Array2<f64>) -> Option<()> {
        let p = a.nrows();
        let mut a_col = to_col_major(a);
        let bytes_a = bytes_len::<f64>(a_col.len())?;
        let bytes_info = bytes_len::<i32>(1)?;

        unsafe {
            self.set_current().ok()?;
            let a_dev = self.alloc_copy(&a_col, bytes_a)?;
            let info_dev = DeviceAllocation::new(&self.driver, bytes_info)?;
            let mut work_size: i32 = 0;
            if (self.solver.dpotrf_buffersize)(
                self.handle,
                CUBLAS_FILL_LOWER,
                p as i32,
                a_dev.ptr,
                p as i32,
                &mut work_size,
            ) != CUSOLVER_STATUS_SUCCESS
            {
                return None;
            }
            let scratch = if work_size > 0 {
                Some(DeviceAllocation::new(
                    &self.driver,
                    (work_size as usize) * std::mem::size_of::<f64>(),
                )?)
            } else {
                None
            };
            let scratch_ptr = scratch.as_ref().map(|s| s.ptr).unwrap_or(0);
            let status = (self.solver.dpotrf)(
                self.handle,
                CUBLAS_FILL_LOWER,
                p as i32,
                a_dev.ptr,
                p as i32,
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
                (self.driver.cu_memcpy_dtoh)(
                    a_col.as_mut_ptr().cast(),
                    a_dev.ptr,
                    bytes_a,
                ),
                "cuMemcpyDtoH A",
            )
            .ok()?;
        }
        from_col_major_inplace(&a_col, a);
        Some(())
    }

    fn potrs_inplace(&mut self, factor: &Array2<f64>, rhs: &mut Array2<f64>) -> Option<()> {
        let p = factor.nrows();
        let nrhs = rhs.ncols();
        let factor_col = to_col_major(factor);
        let mut rhs_col = to_col_major(rhs);
        let bytes_factor = bytes_len::<f64>(factor_col.len())?;
        let bytes_rhs = bytes_len::<f64>(rhs_col.len())?;
        let bytes_info = bytes_len::<i32>(1)?;

        unsafe {
            self.set_current().ok()?;
            let factor_dev = self.alloc_copy(&factor_col, bytes_factor)?;
            let rhs_dev = self.alloc_copy(&rhs_col, bytes_rhs)?;
            let info_dev = DeviceAllocation::new(&self.driver, bytes_info)?;
            let status = (self.solver.dpotrs)(
                self.handle,
                CUBLAS_FILL_LOWER,
                p as i32,
                nrhs as i32,
                factor_dev.ptr,
                p as i32,
                rhs_dev.ptr,
                p as i32,
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
                (self.driver.cu_memcpy_dtoh)(rhs_col.as_mut_ptr().cast(), rhs_dev.ptr, bytes_rhs),
                "cuMemcpyDtoH rhs",
            )
            .ok()?;
        }
        from_col_major_inplace(&rhs_col, rhs);
        Some(())
    }

    fn syevd_inplace(&mut self, a: &mut Array2<f64>) -> Option<Array1<f64>> {
        let p = a.nrows();
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
                p as i32,
                a_dev.ptr,
                p as i32,
                eigs_dev.ptr,
                &mut work_size,
            ) != CUSOLVER_STATUS_SUCCESS
            {
                return None;
            }
            let scratch = if work_size > 0 {
                Some(DeviceAllocation::new(
                    &self.driver,
                    (work_size as usize) * std::mem::size_of::<f64>(),
                )?)
            } else {
                None
            };
            let scratch_ptr = scratch.as_ref().map(|s| s.ptr).unwrap_or(0);
            let status = (self.solver.dsyevd)(
                self.handle,
                CUSOLVER_EIG_MODE_VECTORS,
                CUBLAS_FILL_LOWER,
                p as i32,
                a_dev.ptr,
                p as i32,
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
            unsafe {
                (self.driver.cu_memcpy_htod)(alloc.ptr, values.as_ptr().cast(), bytes)
            },
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

struct DeviceAllocation<'a> {
    driver: &'a DriverApi,
    ptr: u64,
}

impl<'a> DeviceAllocation<'a> {
    unsafe fn new(driver: &'a DriverApi, bytes: usize) -> Option<Self> {
        let mut ptr = 0_u64;
        check_cuda(
            unsafe { (driver.cu_mem_alloc)(&mut ptr, bytes) },
            "cuMemAlloc",
        )
        .ok()?;
        Some(Self { driver, ptr })
    }
}

impl Drop for DeviceAllocation<'_> {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.cu_mem_free)(self.ptr);
        }
    }
}

type CuResult = i32;
type CuInit = unsafe extern "C" fn(u32) -> CuResult;
type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> CuResult;
type CuCtxCreate = unsafe extern "C" fn(*mut usize, u32, i32) -> CuResult;
type CuCtxSetCurrent = unsafe extern "C" fn(usize) -> CuResult;
type CuCtxDestroy = unsafe extern "C" fn(usize) -> CuResult;
type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> CuResult;
type CuMemFree = unsafe extern "C" fn(u64) -> CuResult;
type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const std::ffi::c_void, usize) -> CuResult;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut std::ffi::c_void, u64, usize) -> CuResult;

struct DriverApi {
    cu_init: CuInit,
    cu_device_get: CuDeviceGet,
    cu_ctx_create: CuCtxCreate,
    cu_ctx_set_current: CuCtxSetCurrent,
    cu_ctx_destroy: CuCtxDestroy,
    cu_mem_alloc: CuMemAlloc,
    cu_mem_free: CuMemFree,
    cu_memcpy_htod: CuMemcpyHtoD,
    cu_memcpy_dtoh: CuMemcpyDtoH,
}

impl DriverApi {
    fn load(library: &Library) -> Result<Self, String> {
        unsafe {
            Ok(Self {
                cu_init: *library.get(b"cuInit\0").map_err(|e| e.to_string())?,
                cu_device_get: *library.get(b"cuDeviceGet\0").map_err(|e| e.to_string())?,
                cu_ctx_create: *library
                    .get(b"cuCtxCreate_v2\0")
                    .map_err(|e| e.to_string())?,
                cu_ctx_set_current: *library
                    .get(b"cuCtxSetCurrent\0")
                    .map_err(|e| e.to_string())?,
                cu_ctx_destroy: *library
                    .get(b"cuCtxDestroy_v2\0")
                    .map_err(|e| e.to_string())?,
                cu_mem_alloc: *library.get(b"cuMemAlloc_v2\0").map_err(|e| e.to_string())?,
                cu_mem_free: *library.get(b"cuMemFree_v2\0").map_err(|e| e.to_string())?,
                cu_memcpy_htod: *library
                    .get(b"cuMemcpyHtoD_v2\0")
                    .map_err(|e| e.to_string())?,
                cu_memcpy_dtoh: *library
                    .get(b"cuMemcpyDtoH_v2\0")
                    .map_err(|e| e.to_string())?,
            })
        }
    }
}

type CusolverStatus = i32;
type CusolverCreate = unsafe extern "C" fn(*mut usize) -> CusolverStatus;
type CusolverDestroy = unsafe extern "C" fn(usize) -> CusolverStatus;
type CusolverDpotrfBufferSize = unsafe extern "C" fn(
    usize, // handle
    i32,   // uplo
    i32,   // n
    u64,   // A
    i32,   // lda
    *mut i32,
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

struct CusolverApi {
    create: CusolverCreate,
    destroy: CusolverDestroy,
    dpotrf_buffersize: CusolverDpotrfBufferSize,
    dpotrf: CusolverDpotrf,
    dpotrs: CusolverDpotrs,
    dsyevd_buffersize: CusolverDsyevdBufferSize,
    dsyevd: CusolverDsyevd,
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
                dpotrf_buffersize: *library
                    .get(b"cusolverDnDpotrf_bufferSize\0")
                    .map_err(|e| e.to_string())?,
                dpotrf: *library
                    .get(b"cusolverDnDpotrf\0")
                    .map_err(|e| e.to_string())?,
                dpotrs: *library
                    .get(b"cusolverDnDpotrs\0")
                    .map_err(|e| e.to_string())?,
                dsyevd_buffersize: *library
                    .get(b"cusolverDnDsyevd_bufferSize\0")
                    .map_err(|e| e.to_string())?,
                dsyevd: *library
                    .get(b"cusolverDnDsyevd\0")
                    .map_err(|e| e.to_string())?,
            })
        }
    }
}

const CUSOLVER_STATUS_SUCCESS: CusolverStatus = 0;
const CUBLAS_FILL_LOWER: i32 = 1;
const CUSOLVER_EIG_MODE_VECTORS: i32 = 1;

fn check_cuda(result: CuResult, name: &str) -> Result<(), String> {
    if result == 0 {
        Ok(())
    } else {
        Err(format!("{name} failed with CUDA driver error {result}"))
    }
}

fn load_library(candidates: &[&str]) -> Result<Library, String> {
    for candidate in candidates {
        if let Ok(library) = unsafe { Library::new(candidate) } {
            return Ok(library);
        }
    }
    Err(format!("could not load any of: {}", candidates.join(", ")))
}

fn cuda_library_candidates() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["nvcuda.dll"]
    } else if cfg!(target_os = "macos") {
        &["/usr/local/cuda/lib/libcuda.dylib", "libcuda.dylib"]
    } else {
        &["libcuda.so.1", "libcuda.so"]
    }
}

fn cusolver_library_candidates() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["cusolver64_12.dll", "cusolver64_11.dll"]
    } else if cfg!(target_os = "macos") {
        &[
            "/usr/local/cuda/lib/libcusolver.dylib",
            "libcusolver.dylib",
        ]
    } else {
        &["libcusolver.so.12", "libcusolver.so.11", "libcusolver.so"]
    }
}

fn bytes_len<T>(len: usize) -> Option<usize> {
    len.checked_mul(std::mem::size_of::<T>())
}

fn to_col_major(a: &Array2<f64>) -> Vec<f64> {
    let (rows, cols) = a.dim();
    let mut out = Vec::with_capacity(rows.saturating_mul(cols));
    for col in 0..cols {
        for row in 0..rows {
            out.push(a[[row, col]]);
        }
    }
    out
}

fn from_col_major_inplace(values: &[f64], out: &mut Array2<f64>) {
    let (rows, cols) = out.dim();
    for col in 0..cols {
        for row in 0..rows {
            out[[row, col]] = values[col * rows + row];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn small_matrices_do_not_route_to_gpu() {
        let mut a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        assert!(try_potrf_inplace(&mut a).is_none());
        assert!(try_syevd_inplace(&mut a).is_none());
        let factor = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let mut rhs = array![[1.0_f64], [1.0]];
        assert!(try_potrs_inplace(&factor, &mut rhs).is_none());
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

//! cuSPARSE routing for large CSR sparse-matrix kernels.
//!
//! Built on the same dlopen pattern as [`super::blas`]: we resolve libcusparse
//! at process start, hold a single context for the selected device, and route
//! sufficiently large CSR SpMV calls through `cusparseSpMV`. CSR matrices
//! below the threshold fall back to the CPU (sparse_exact / faer-sparse).

use libloading::Library;
use ndarray::{Array1, ArrayBase, Data, Ix1};
use std::sync::{Mutex, OnceLock};

use super::runtime::GpuRuntime;

/// Dispatch entry point: returns `Some(y)` when the device runtime executed
/// `y = A x` for the given CSR triple, `None` to fall back to CPU.
#[inline]
pub fn try_csr_spmv<S: Data<Elem = f64>>(
    rowptr: &[i32],
    colidx: &[i32],
    values: &[f64],
    rows: usize,
    cols: usize,
    x: &ArrayBase<S, Ix1>,
) -> Option<Array1<f64>> {
    let nnz = values.len();
    debug_assert_eq!(rowptr.len(), rows + 1);
    debug_assert_eq!(colidx.len(), nnz);
    debug_assert_eq!(x.len(), cols);
    if !route_csr_spmv(rows, cols, nnz) {
        return None;
    }
    with_runtime(|rt| rt.csr_spmv(rowptr, colidx, values, rows, cols, x))
}

#[inline]
fn route_csr_spmv(rows: usize, cols: usize, nnz: usize) -> bool {
    // PCIe + cuSPARSE descriptor setup is expensive for small matrices; the
    // 1M-nnz / 1024-row floor matches the Mainstream-bucket default in
    // `DispatchPolicy::baseline()`.
    rows >= 1_024 && cols > 0 && nnz >= 1_000_000
}

fn with_runtime<T>(f: impl FnOnce(&mut CusparseRuntime) -> Option<T>) -> Option<T> {
    static RUNTIME: OnceLock<Option<Mutex<CusparseRuntime>>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| CusparseRuntime::new().ok().map(Mutex::new))
        .as_ref()?
        .lock()
        .ok()
        .and_then(|mut rt| f(&mut rt))
}

struct CusparseRuntime {
    _cuda_lib: Library,
    _cusparse_lib: Library,
    driver: DriverApi,
    sparse: CusparseApi,
    context: usize,
    handle: usize,
}

impl CusparseRuntime {
    fn new() -> Result<Self, String> {
        let selected = GpuRuntime::global()
            .selected_device()
            .ok_or_else(|| "no CUDA device selected".to_string())?;
        let cuda_lib = load_library(cuda_library_candidates())?;
        let sparse_lib = load_library(cusparse_library_candidates())?;
        let driver = DriverApi::load(&cuda_lib)?;
        let sparse = CusparseApi::load(&sparse_lib)?;
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
            let status = (sparse.cusparse_create)(&mut handle);
            if status != CUSPARSE_STATUS_SUCCESS {
                let _ = (driver.cu_ctx_destroy)(context);
                return Err(format!("cusparseCreate failed with status {status}"));
            }
            Ok(Self {
                _cuda_lib: cuda_lib,
                _cusparse_lib: sparse_lib,
                driver,
                sparse,
                context,
                handle,
            })
        }
    }

    fn csr_spmv<S: Data<Elem = f64>>(
        &mut self,
        rowptr: &[i32],
        colidx: &[i32],
        values: &[f64],
        rows: usize,
        cols: usize,
        x: &ArrayBase<S, Ix1>,
    ) -> Option<Array1<f64>> {
        let nnz = values.len();
        let mut y_host = vec![0.0_f64; rows];
        let bytes_rowptr = bytes_len::<i32>(rowptr.len())?;
        let bytes_colidx = bytes_len::<i32>(colidx.len())?;
        let bytes_values = bytes_len::<f64>(values.len())?;
        let bytes_x = bytes_len::<f64>(cols)?;
        let bytes_y = bytes_len::<f64>(rows)?;

        unsafe {
            self.set_current().ok()?;
            let rowptr_dev = self.alloc_copy_bytes(rowptr.as_ptr().cast(), bytes_rowptr)?;
            let colidx_dev = self.alloc_copy_bytes(colidx.as_ptr().cast(), bytes_colidx)?;
            let values_dev = self.alloc_copy_bytes(values.as_ptr().cast(), bytes_values)?;
            let x_dev = self.alloc_copy_bytes(x.as_ptr().cast(), bytes_x)?;
            let y_dev = DeviceAllocation::new(&self.driver, bytes_y)?;

            let mut spmat: usize = 0;
            if (self.sparse.cusparse_create_csr)(
                &mut spmat,
                rows as i64,
                cols as i64,
                nnz as i64,
                rowptr_dev.ptr,
                colidx_dev.ptr,
                values_dev.ptr,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_R_64F,
            ) != CUSPARSE_STATUS_SUCCESS
            {
                return None;
            }
            let mut x_descr: usize = 0;
            if (self.sparse.cusparse_create_dnvec)(
                &mut x_descr,
                cols as i64,
                x_dev.ptr,
                CUDA_R_64F,
            ) != CUSPARSE_STATUS_SUCCESS
            {
                let _ = (self.sparse.cusparse_destroy_spmat)(spmat);
                return None;
            }
            let mut y_descr: usize = 0;
            if (self.sparse.cusparse_create_dnvec)(
                &mut y_descr,
                rows as i64,
                y_dev.ptr,
                CUDA_R_64F,
            ) != CUSPARSE_STATUS_SUCCESS
            {
                let _ = (self.sparse.cusparse_destroy_dnvec)(x_descr);
                let _ = (self.sparse.cusparse_destroy_spmat)(spmat);
                return None;
            }

            let alpha = 1.0_f64;
            let beta = 0.0_f64;
            let mut buffer_size: usize = 0;
            if (self.sparse.cusparse_spmv_buffersize)(
                self.handle,
                CUSPARSE_OP_N,
                &alpha as *const f64 as *const std::ffi::c_void,
                spmat,
                x_descr,
                &beta as *const f64 as *const std::ffi::c_void,
                y_descr,
                CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                &mut buffer_size,
            ) != CUSPARSE_STATUS_SUCCESS
            {
                let _ = (self.sparse.cusparse_destroy_dnvec)(y_descr);
                let _ = (self.sparse.cusparse_destroy_dnvec)(x_descr);
                let _ = (self.sparse.cusparse_destroy_spmat)(spmat);
                return None;
            }
            let scratch = if buffer_size > 0 {
                Some(DeviceAllocation::new(&self.driver, buffer_size)?)
            } else {
                None
            };
            let buffer_ptr = scratch.as_ref().map(|s| s.ptr).unwrap_or(0);
            let exec_status = (self.sparse.cusparse_spmv)(
                self.handle,
                CUSPARSE_OP_N,
                &alpha as *const f64 as *const std::ffi::c_void,
                spmat,
                x_descr,
                &beta as *const f64 as *const std::ffi::c_void,
                y_descr,
                CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                buffer_ptr,
            );
            let _ = (self.sparse.cusparse_destroy_dnvec)(y_descr);
            let _ = (self.sparse.cusparse_destroy_dnvec)(x_descr);
            let _ = (self.sparse.cusparse_destroy_spmat)(spmat);
            if exec_status != CUSPARSE_STATUS_SUCCESS {
                return None;
            }
            check_cuda(
                (self.driver.cu_memcpy_dtoh)(y_host.as_mut_ptr().cast(), y_dev.ptr, bytes_y),
                "cuMemcpyDtoH",
            )
            .ok()?;
        }
        Some(Array1::from_vec(y_host))
    }

    unsafe fn set_current(&self) -> Result<(), String> {
        check_cuda(
            unsafe { (self.driver.cu_ctx_set_current)(self.context) },
            "cuCtxSetCurrent",
        )
    }

    unsafe fn alloc_copy_bytes<'a>(
        &'a self,
        src: *const std::ffi::c_void,
        bytes: usize,
    ) -> Option<DeviceAllocation<'a>> {
        let alloc = unsafe { DeviceAllocation::new(&self.driver, bytes) }?;
        check_cuda(
            unsafe { (self.driver.cu_memcpy_htod)(alloc.ptr, src, bytes) },
            "cuMemcpyHtoD",
        )
        .ok()?;
        Some(alloc)
    }
}

impl Drop for CusparseRuntime {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.cu_ctx_set_current)(self.context);
            let _ = (self.sparse.cusparse_destroy)(self.handle);
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

type CusparseStatus = i32;
type CusparseCreate = unsafe extern "C" fn(*mut usize) -> CusparseStatus;
type CusparseDestroy = unsafe extern "C" fn(usize) -> CusparseStatus;
type CusparseCreateCsr = unsafe extern "C" fn(
    *mut usize, // descr
    i64,        // rows
    i64,        // cols
    i64,        // nnz
    u64,        // rowptr
    u64,        // colidx
    u64,        // values
    i32,        // rowptr type
    i32,        // colidx type
    i32,        // index base
    i32,        // value type
) -> CusparseStatus;
type CusparseCreateDnvec = unsafe extern "C" fn(
    *mut usize, // descr
    i64,        // size
    u64,        // data
    i32,        // value type
) -> CusparseStatus;
type CusparseDestroySpmat = unsafe extern "C" fn(usize) -> CusparseStatus;
type CusparseDestroyDnvec = unsafe extern "C" fn(usize) -> CusparseStatus;
type CusparseSpmvBufferSize = unsafe extern "C" fn(
    usize,                       // handle
    i32,                         // opA
    *const std::ffi::c_void,     // alpha
    usize,                       // matA
    usize,                       // vecX
    *const std::ffi::c_void,     // beta
    usize,                       // vecY
    i32,                         // compute type
    i32,                         // alg
    *mut usize,                  // buffer size
) -> CusparseStatus;
type CusparseSpmv = unsafe extern "C" fn(
    usize,                       // handle
    i32,                         // opA
    *const std::ffi::c_void,     // alpha
    usize,                       // matA
    usize,                       // vecX
    *const std::ffi::c_void,     // beta
    usize,                       // vecY
    i32,                         // compute type
    i32,                         // alg
    u64,                         // external buffer
) -> CusparseStatus;

struct CusparseApi {
    cusparse_create: CusparseCreate,
    cusparse_destroy: CusparseDestroy,
    cusparse_create_csr: CusparseCreateCsr,
    cusparse_create_dnvec: CusparseCreateDnvec,
    cusparse_destroy_spmat: CusparseDestroySpmat,
    cusparse_destroy_dnvec: CusparseDestroyDnvec,
    cusparse_spmv_buffersize: CusparseSpmvBufferSize,
    cusparse_spmv: CusparseSpmv,
}

impl CusparseApi {
    fn load(library: &Library) -> Result<Self, String> {
        unsafe {
            Ok(Self {
                cusparse_create: *library
                    .get(b"cusparseCreate\0")
                    .map_err(|e| e.to_string())?,
                cusparse_destroy: *library
                    .get(b"cusparseDestroy\0")
                    .map_err(|e| e.to_string())?,
                cusparse_create_csr: *library
                    .get(b"cusparseCreateCsr\0")
                    .map_err(|e| e.to_string())?,
                cusparse_create_dnvec: *library
                    .get(b"cusparseCreateDnVec\0")
                    .map_err(|e| e.to_string())?,
                cusparse_destroy_spmat: *library
                    .get(b"cusparseDestroySpMat\0")
                    .map_err(|e| e.to_string())?,
                cusparse_destroy_dnvec: *library
                    .get(b"cusparseDestroyDnVec\0")
                    .map_err(|e| e.to_string())?,
                cusparse_spmv_buffersize: *library
                    .get(b"cusparseSpMV_bufferSize\0")
                    .map_err(|e| e.to_string())?,
                cusparse_spmv: *library.get(b"cusparseSpMV\0").map_err(|e| e.to_string())?,
            })
        }
    }
}

const CUSPARSE_STATUS_SUCCESS: CusparseStatus = 0;
const CUSPARSE_OP_N: i32 = 0;
const CUSPARSE_INDEX_32I: i32 = 1;
const CUSPARSE_INDEX_BASE_ZERO: i32 = 0;
const CUSPARSE_SPMV_ALG_DEFAULT: i32 = 0;
const CUDA_R_64F: i32 = 1; // matches CUDA's cudaDataType enum value for double.

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

fn cusparse_library_candidates() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["cusparse64_12.dll", "cusparse64_11.dll"]
    } else if cfg!(target_os = "macos") {
        &[
            "/usr/local/cuda/lib/libcusparse.dylib",
            "libcusparse.dylib",
        ]
    } else {
        &["libcusparse.so.12", "libcusparse.so.11", "libcusparse.so"]
    }
}

fn bytes_len<T>(len: usize) -> Option<usize> {
    len.checked_mul(std::mem::size_of::<T>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn small_csr_does_not_route_to_gpu() {
        let rowptr = vec![0_i32, 1, 2];
        let colidx = vec![0_i32, 1];
        let values = vec![2.0_f64, 3.0];
        let x = array![1.0_f64, 1.0];
        assert!(try_csr_spmv(&rowptr, &colidx, &values, 2, 2, &x).is_none());
    }
}

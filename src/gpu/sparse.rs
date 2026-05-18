//! cuSPARSE routing for large CSR sparse-matrix kernels.
//!
//! Built on the same dlopen pattern as [`super::blas`]: we resolve libcusparse
//! at process start, hold a single context for the selected device, and route
//! sufficiently large CSR SpMV calls through `cusparseSpMV`. Smaller matrices
//! continue on the in-process sparse CPU path.

use libloading::Library;
use ndarray::{Array1, ArrayBase, Data, Ix1};
use std::sync::{Mutex, OnceLock};

use super::diagnostics;
use super::driver::{
    CudaWorkingState, DeviceAllocation, bytes_len, check_cuda, load_static_library, to_i64,
};
use super::runtime::GpuRuntime;

/// Dispatch entry point for `y = A x` with `usize` CSR indices.
#[inline]
pub fn try_csr_spmv_usize<S: Data<Elem = f64>>(
    rowptr: &[usize],
    colidx: &[usize],
    values: &[f64],
    rows: usize,
    cols: usize,
    x: &ArrayBase<S, Ix1>,
) -> Option<Array1<f64>> {
    try_csr_spmv_indexed(rowptr, colidx, values, rows, cols, x, false)
}

/// Dispatch entry point for `y = A^T x` with `usize` CSR indices.
#[inline]
pub fn try_csr_t_spmv_usize<S: Data<Elem = f64>>(
    rowptr: &[usize],
    colidx: &[usize],
    values: &[f64],
    rows: usize,
    cols: usize,
    x: &ArrayBase<S, Ix1>,
) -> Option<Array1<f64>> {
    try_csr_spmv_indexed(rowptr, colidx, values, rows, cols, x, true)
}

fn try_csr_spmv_indexed<S: Data<Elem = f64>>(
    rowptr: &[usize],
    colidx: &[usize],
    values: &[f64],
    rows: usize,
    cols: usize,
    x: &ArrayBase<S, Ix1>,
    transpose: bool,
) -> Option<Array1<f64>> {
    if !route_csr_spmv(rows, cols, values.len()) {
        diagnostics::log_policy_cpu(
            if transpose { "csr_t_spmv" } else { "csr_spmv" },
            format!("rows={rows} cols={cols} nnz={}", values.len()),
            format!(
                "below cuSPARSE policy threshold rows>={} and nnz>={}",
                GpuRuntime::global().policy().spmv_min_rows,
                GpuRuntime::global().policy().spmv_min_nnz
            ),
        );
        return None;
    }
    let rowptr_i32 = checked_i32_vec(rowptr)?;
    let colidx_i32 = checked_i32_vec(colidx)?;
    try_csr_spmv(&rowptr_i32, &colidx_i32, values, rows, cols, x, transpose)
}

/// Dispatch entry point: returns `Some(y)` when the device runtime executed
/// the CSR SpMV for the given CSR triple.
#[inline]
fn try_csr_spmv<S: Data<Elem = f64>>(
    rowptr: &[i32],
    colidx: &[i32],
    values: &[f64],
    rows: usize,
    cols: usize,
    x: &ArrayBase<S, Ix1>,
    transpose: bool,
) -> Option<Array1<f64>> {
    let nnz = values.len();
    if rowptr.len() != rows.checked_add(1)?
        || colidx.len() != nnz
        || x.len() != if transpose { rows } else { cols }
    {
        return None;
    }
    if !route_csr_spmv(rows, cols, nnz) {
        diagnostics::log_policy_cpu(
            if transpose { "csr_t_spmv" } else { "csr_spmv" },
            format!("rows={rows} cols={cols} nnz={nnz}"),
            format!(
                "below cuSPARSE policy threshold rows>={} and nnz>={}",
                GpuRuntime::global().policy().spmv_min_rows,
                GpuRuntime::global().policy().spmv_min_nnz
            ),
        );
        return None;
    }
    let op = if transpose { "csr_t_spmv" } else { "csr_spmv" };
    let start = std::time::Instant::now();
    let result = with_runtime(|rt| rt.csr_spmv(rowptr, colidx, values, rows, cols, x, transpose));
    if result.is_some() {
        diagnostics::log_gpu_success(
            op,
            "cuSPARSE",
            format!("rows={rows} cols={cols} nnz={nnz}"),
            (nnz as u64).saturating_mul(2),
            diagnostics::bytes_for_i32(rowptr.len().saturating_add(colidx.len())).saturating_add(
                diagnostics::bytes_for_f64(values.len().saturating_add(x.len())),
            ),
            diagnostics::bytes_for_f64(if transpose { cols } else { rows }),
            start.elapsed().as_secs_f64(),
        );
    } else {
        diagnostics::log_runtime_cpu(op, "cuSPARSE", format!("rows={rows} cols={cols} nnz={nnz}"));
    }
    result
}

#[inline]
fn route_csr_spmv(rows: usize, cols: usize, nnz: usize) -> bool {
    GpuRuntime::global()
        .policy()
        .route_csr_spmv(rows, cols, nnz)
}

fn with_runtime<T>(f: impl FnOnce(&mut CusparseRuntime) -> Option<T>) -> Option<T> {
    static RUNTIME: OnceLock<Option<Mutex<CusparseRuntime>>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            let cuda = GpuRuntime::global().cuda_working_state()?;
            match CusparseRuntime::new(cuda) {
                Ok(runtime) => {
                    diagnostics::log_library_ready("cuSPARSE");
                    Some(Mutex::new(runtime))
                }
                Err(err) => {
                    diagnostics::log_library_unavailable("cuSPARSE", &err);
                    None
                }
            }
        })
        .as_ref()?
        .lock()
        .ok()
        .and_then(|mut rt| f(&mut rt))
}

struct CusparseRuntime {
    cuda: &'static CudaWorkingState,
    /// cuSPARSE entry points; the dlopen'd library is leaked into a
    /// `&'static` so these pointers stay valid for the process.
    sparse: CusparseApi,
    handle: usize,
}

impl CusparseRuntime {
    fn new(cuda: &'static CudaWorkingState) -> Result<Self, String> {
        let sparse_lib = load_static_library(cusparse_library_candidates())?;
        let sparse = CusparseApi::load(sparse_lib)?;
        cuda.set_current()?;
        let mut handle = 0_usize;
        let status = unsafe { (sparse.cusparse_create)(&mut handle) };
        if status != CUSPARSE_STATUS_SUCCESS {
            return Err(format!("cusparseCreate failed with status {status}"));
        }
        Ok(Self {
            cuda,
            sparse,
            handle,
        })
    }

    fn csr_spmv<S: Data<Elem = f64>>(
        &mut self,
        rowptr: &[i32],
        colidx: &[i32],
        values: &[f64],
        rows: usize,
        cols: usize,
        x: &ArrayBase<S, Ix1>,
        transpose: bool,
    ) -> Option<Array1<f64>> {
        let nnz = values.len();
        if rowptr.len() != rows.checked_add(1)? || colidx.len() != nnz {
            return None;
        }
        let y_len = if transpose { cols } else { rows };
        let x_len = if transpose { rows } else { cols };
        if x.len() != x_len {
            return None;
        }
        let rows_i64 = to_i64(rows)?;
        let cols_i64 = to_i64(cols)?;
        let nnz_i64 = to_i64(nnz)?;
        let x_len_i64 = to_i64(x_len)?;
        let y_len_i64 = to_i64(y_len)?;
        let x_host;
        let x_slice = if let Some(slice) = x.as_slice_memory_order() {
            slice
        } else {
            x_host = x.to_vec();
            &x_host
        };
        let mut y_host = vec![0.0_f64; y_len];
        let bytes_rowptr = bytes_len::<i32>(rowptr.len())?;
        let bytes_colidx = bytes_len::<i32>(colidx.len())?;
        let bytes_values = bytes_len::<f64>(values.len())?;
        let bytes_x = bytes_len::<f64>(x_len)?;
        let bytes_y = bytes_len::<f64>(y_len)?;

        unsafe {
            self.cuda.set_current().ok()?;
            let rowptr_dev = self.alloc_copy_bytes(rowptr.as_ptr().cast(), bytes_rowptr)?;
            let colidx_dev = self.alloc_copy_bytes(colidx.as_ptr().cast(), bytes_colidx)?;
            let values_dev = self.alloc_copy_bytes(values.as_ptr().cast(), bytes_values)?;
            let x_dev = self.alloc_copy_bytes(x_slice.as_ptr().cast(), bytes_x)?;
            let y_dev = DeviceAllocation::new(&self.cuda.api, bytes_y)?;

            let mut spmat: usize = 0;
            if (self.sparse.cusparse_create_csr)(
                &mut spmat,
                rows_i64,
                cols_i64,
                nnz_i64,
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
            if (self.sparse.cusparse_create_dnvec)(&mut x_descr, x_len_i64, x_dev.ptr, CUDA_R_64F)
                != CUSPARSE_STATUS_SUCCESS
            {
                let _ = (self.sparse.cusparse_destroy_spmat)(spmat);
                return None;
            }
            let mut y_descr: usize = 0;
            if (self.sparse.cusparse_create_dnvec)(&mut y_descr, y_len_i64, y_dev.ptr, CUDA_R_64F)
                != CUSPARSE_STATUS_SUCCESS
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
                cusparse_op(transpose),
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
                Some(DeviceAllocation::new(&self.cuda.api, buffer_size)?)
            } else {
                None
            };
            let buffer_ptr = scratch.as_ref().map(|s| s.ptr).unwrap_or(0);
            let exec_status = (self.sparse.cusparse_spmv)(
                self.handle,
                cusparse_op(transpose),
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
                (self.cuda.api.cu_memcpy_dtoh)(y_host.as_mut_ptr().cast(), y_dev.ptr, bytes_y),
                "cuMemcpyDtoH",
            )
            .ok()?;
        }
        Some(Array1::from_vec(y_host))
    }

    unsafe fn alloc_copy_bytes<'a>(
        &'a self,
        src: *const std::ffi::c_void,
        bytes: usize,
    ) -> Option<DeviceAllocation<'a>> {
        let alloc = unsafe { DeviceAllocation::new(&self.cuda.api, bytes) }?;
        check_cuda(
            unsafe { (self.cuda.api.cu_memcpy_htod)(alloc.ptr, src, bytes) },
            "cuMemcpyHtoD",
        )
        .ok()?;
        Some(alloc)
    }
}

impl Drop for CusparseRuntime {
    fn drop(&mut self) {
        unsafe {
            let _ = self.cuda.set_current();
            let _ = (self.sparse.cusparse_destroy)(self.handle);
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
    usize,                   // handle
    i32,                     // opA
    *const std::ffi::c_void, // alpha
    usize,                   // matA
    usize,                   // vecX
    *const std::ffi::c_void, // beta
    usize,                   // vecY
    i32,                     // compute type
    i32,                     // alg
    *mut usize,              // buffer size
) -> CusparseStatus;
type CusparseSpmv = unsafe extern "C" fn(
    usize,                   // handle
    i32,                     // opA
    *const std::ffi::c_void, // alpha
    usize,                   // matA
    usize,                   // vecX
    *const std::ffi::c_void, // beta
    usize,                   // vecY
    i32,                     // compute type
    i32,                     // alg
    u64,                     // external buffer
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
const CUSPARSE_OP_T: i32 = 1;
const CUSPARSE_INDEX_32I: i32 = 2;
const CUSPARSE_INDEX_BASE_ZERO: i32 = 0;
const CUSPARSE_SPMV_ALG_DEFAULT: i32 = 0;
const CUDA_R_64F: i32 = 1; // matches CUDA's cudaDataType enum value for double.

fn cusparse_library_candidates() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["cusparse64_12.dll", "cusparse64_11.dll"]
    } else if cfg!(target_os = "macos") {
        &["/usr/local/cuda/lib/libcusparse.dylib", "libcusparse.dylib"]
    } else {
        &["libcusparse.so.12", "libcusparse.so.11", "libcusparse.so"]
    }
}

fn checked_i32_vec(values: &[usize]) -> Option<Vec<i32>> {
    values
        .iter()
        .copied()
        .map(i32::try_from)
        .collect::<Result<Vec<_>, _>>()
        .ok()
}

fn cusparse_op(transpose: bool) -> i32 {
    if transpose {
        CUSPARSE_OP_T
    } else {
        CUSPARSE_OP_N
    }
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
        assert!(try_csr_spmv(&rowptr, &colidx, &values, 2, 2, &x, false).is_none());
        assert!(try_csr_t_spmv_usize(&[0, 1, 2], &[0, 1], &values, 2, 2, &x).is_none());
    }
}

//! cuBLAS routing for large f64 dense kernels.

use libloading::Library;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use std::sync::{Mutex, OnceLock};

use super::driver::{
    DeviceAllocation, DriverApi, bytes_len, check_cuda, cuda_library_candidates,
    from_col_major, load_library, to_col_major,
};
use super::runtime::GpuRuntime;

#[inline]
pub fn try_fast_ab<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Option<Array2<f64>> {
    let (m, k) = a.dim();
    let (k_b, n) = b.dim();
    debug_assert_eq!(k, k_b, "A and B must have compatible inner dimensions");
    if !route_gemm(m, n, k) {
        return None;
    }
    with_runtime(|runtime| runtime.gemm(a, b, false, false))
}

#[inline]
pub fn try_fast_atb<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Option<Array2<f64>> {
    let (rows, cols) = a.dim();
    let (rows_b, rhs) = b.dim();
    debug_assert_eq!(rows, rows_b, "A and B must have same number of rows");
    if !route_gemm(cols, rhs, rows) {
        return None;
    }
    with_runtime(|runtime| runtime.gemm(a, b, true, false))
}

#[inline]
pub fn try_fast_av<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
) -> Option<Array1<f64>> {
    let (rows, cols) = a.dim();
    debug_assert_eq!(cols, v.len(), "A cols must match v length");
    if !route_gemv(rows, cols) {
        return None;
    }
    with_runtime(|runtime| runtime.gemv(a, v, false))
}

#[inline]
pub fn try_fast_atv<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
) -> Option<Array1<f64>> {
    let (rows, cols) = a.dim();
    debug_assert_eq!(rows, v.len(), "A rows must match v length");
    if !route_gemv(rows, cols) {
        return None;
    }
    with_runtime(|runtime| runtime.gemv(a, v, true))
}

#[inline]
pub fn try_fast_xt_diag_x<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
) -> Option<Array2<f64>> {
    let (rows, cols) = x.dim();
    debug_assert_eq!(rows, w.len(), "X rows must match W length");
    if !route_xtwx(rows, cols, cols) {
        return None;
    }
    with_runtime(|runtime| runtime.xt_diag_y(x, w, x))
}

#[inline]
pub fn try_fast_xt_diag_y<S1: Data<Elem = f64>, S2: Data<Elem = f64>, S3: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
    y: &ArrayBase<S3, Ix2>,
) -> Option<Array2<f64>> {
    let rows = x.nrows();
    debug_assert_eq!(rows, w.len(), "X rows must match W length");
    debug_assert_eq!(rows, y.nrows(), "X rows must match Y rows");
    if !route_xtwx(rows, x.ncols(), y.ncols()) {
        return None;
    }
    with_runtime(|runtime| runtime.xt_diag_y(x, w, y))
}

#[inline]
fn route_gemm(m: usize, n: usize, k: usize) -> bool {
    m > 0 && n > 0 && k > 0 && GpuRuntime::global().policy().route_gemm(m, n, k)
}

#[inline]
fn route_gemv(rows: usize, cols: usize) -> bool {
    rows > 0 && cols > 0 && GpuRuntime::global().policy().route_gemv(rows, cols)
}

#[inline]
fn route_xtwx(rows: usize, lhs_cols: usize, rhs_cols: usize) -> bool {
    rows > 0
        && lhs_cols > 0
        && rhs_cols > 0
        && GpuRuntime::global()
            .policy()
            .route_xt_diag_y(rows, lhs_cols, rhs_cols)
}

fn with_runtime<T>(f: impl FnOnce(&mut CublasRuntime) -> Option<T>) -> Option<T> {
    static RUNTIME: OnceLock<Option<Mutex<CublasRuntime>>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| CublasRuntime::new().ok().map(Mutex::new))
        .as_ref()?
        .lock()
        .ok()
        .and_then(|mut runtime| f(&mut runtime))
}

struct CublasRuntime {
    _cuda_lib: Library,
    _cublas_lib: Library,
    driver: DriverApi,
    blas: CublasApi,
    context: usize,
    handle: usize,
}

impl CublasRuntime {
    fn new() -> Result<Self, String> {
        let selected = GpuRuntime::global()
            .selected_device()
            .ok_or_else(|| "no CUDA device selected".to_string())?;
        let cuda_lib = load_library(cuda_library_candidates())?;
        let cublas_lib = load_library(cublas_library_candidates())?;
        let driver = DriverApi::load(&cuda_lib)?;
        let blas = CublasApi::load(&cublas_lib)?;
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
            let create_status = (blas.cublas_create)(&mut handle);
            if create_status != CUBLAS_STATUS_SUCCESS {
                let _ = (driver.cu_ctx_destroy)(context);
                return Err(format!("cublasCreate failed with status {create_status}"));
            }
            Ok(Self {
                _cuda_lib: cuda_lib,
                _cublas_lib: cublas_lib,
                driver,
                blas,
                context,
                handle,
            })
        }
    }

    fn gemm<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
        &mut self,
        a: &ArrayBase<S1, Ix2>,
        b: &ArrayBase<S2, Ix2>,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Option<Array2<f64>> {
        let a_rows = a.nrows();
        let a_cols = a.ncols();
        let b_rows = b.nrows();
        let b_cols = b.ncols();
        let m = if transpose_a { a_cols } else { a_rows };
        let k = if transpose_a { a_rows } else { a_cols };
        let b_k = if transpose_b { b_cols } else { b_rows };
        let n = if transpose_b { b_rows } else { b_cols };
        if k != b_k {
            return None;
        }

        let a_host = to_col_major(a);
        let b_host = to_col_major(b);
        let mut c_host = vec![0.0; m.checked_mul(n)?];
        let bytes_a = bytes_len::<f64>(a_host.len())?;
        let bytes_b = bytes_len::<f64>(b_host.len())?;
        let bytes_c = bytes_len::<f64>(c_host.len())?;

        unsafe {
            self.set_current().ok()?;
            let a_dev = self.alloc_copy(&a_host, bytes_a)?;
            let b_dev = self.alloc_copy(&b_host, bytes_b)?;
            let c_dev = DeviceAllocation::new(&self.driver, bytes_c)?;
            let alpha = 1.0;
            let beta = 0.0;
            let status = (self.blas.cublas_dgemm)(
                self.handle,
                cublas_op(transpose_a),
                cublas_op(transpose_b),
                to_i32(m)?,
                to_i32(n)?,
                to_i32(k)?,
                &alpha,
                a_dev.ptr,
                to_i32(a_rows)?,
                b_dev.ptr,
                to_i32(b_rows)?,
                &beta,
                c_dev.ptr,
                to_i32(m)?,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return None;
            }
            check_cuda(
                (self.driver.cu_memcpy_dtoh)(c_host.as_mut_ptr().cast(), c_dev.ptr, bytes_c),
                "cuMemcpyDtoH",
            )
            .ok()?;
        }
        Some(from_col_major(&c_host, m, n))
    }

    fn gemv<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
        &mut self,
        a: &ArrayBase<S1, Ix2>,
        v: &ArrayBase<S2, Ix1>,
        transpose: bool,
    ) -> Option<Array1<f64>> {
        let (rows, cols) = a.dim();
        let out_len = if transpose { cols } else { rows };
        let a_host = to_col_major(a);
        let x_host = v.to_vec();
        let mut y_host = vec![0.0; out_len];
        let bytes_a = bytes_len::<f64>(a_host.len())?;
        let bytes_x = bytes_len::<f64>(x_host.len())?;
        let bytes_y = bytes_len::<f64>(y_host.len())?;

        unsafe {
            self.set_current().ok()?;
            let a_dev = self.alloc_copy(&a_host, bytes_a)?;
            let x_dev = self.alloc_copy(&x_host, bytes_x)?;
            let y_dev = DeviceAllocation::new(&self.driver, bytes_y)?;
            let alpha = 1.0;
            let beta = 0.0;
            let status = (self.blas.cublas_dgemv)(
                self.handle,
                cublas_op(transpose),
                to_i32(rows)?,
                to_i32(cols)?,
                &alpha,
                a_dev.ptr,
                to_i32(rows)?,
                x_dev.ptr,
                1,
                &beta,
                y_dev.ptr,
                1,
            );
            if status != CUBLAS_STATUS_SUCCESS {
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

    fn xt_diag_y<S1: Data<Elem = f64>, S2: Data<Elem = f64>, S3: Data<Elem = f64>>(
        &mut self,
        x: &ArrayBase<S1, Ix2>,
        w: &ArrayBase<S2, Ix1>,
        y: &ArrayBase<S3, Ix2>,
    ) -> Option<Array2<f64>> {
        let rows = x.nrows();
        let x_cols = x.ncols();
        let y_cols = y.ncols();
        if rows != w.len() || rows != y.nrows() {
            return None;
        }
        let x_host = to_col_major(x);
        let y_host = to_col_major(y);
        let w_host = w.to_vec();
        let mut out_host = vec![0.0; x_cols.checked_mul(y_cols)?];
        let bytes_x = bytes_len::<f64>(x_host.len())?;
        let bytes_y = bytes_len::<f64>(y_host.len())?;
        let bytes_w = bytes_len::<f64>(w_host.len())?;
        let bytes_out = bytes_len::<f64>(out_host.len())?;

        unsafe {
            self.set_current().ok()?;
            let x_dev = self.alloc_copy(&x_host, bytes_x)?;
            let y_dev = self.alloc_copy(&y_host, bytes_y)?;
            let w_dev = self.alloc_copy(&w_host, bytes_w)?;
            let wy_dev = DeviceAllocation::new(&self.driver, bytes_y)?;
            let scale_status = (self.blas.cublas_ddgmm)(
                self.handle,
                CUBLAS_SIDE_LEFT,
                to_i32(rows)?,
                to_i32(y_cols)?,
                y_dev.ptr,
                to_i32(rows)?,
                w_dev.ptr,
                1,
                wy_dev.ptr,
                to_i32(rows)?,
            );
            if scale_status != CUBLAS_STATUS_SUCCESS {
                return None;
            }
            let out_dev = DeviceAllocation::new(&self.driver, bytes_out)?;
            let alpha = 1.0;
            let beta = 0.0;
            let status = (self.blas.cublas_dgemm)(
                self.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                to_i32(x_cols)?,
                to_i32(y_cols)?,
                to_i32(rows)?,
                &alpha,
                x_dev.ptr,
                to_i32(rows)?,
                wy_dev.ptr,
                to_i32(rows)?,
                &beta,
                out_dev.ptr,
                to_i32(x_cols)?,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return None;
            }
            check_cuda(
                (self.driver.cu_memcpy_dtoh)(out_host.as_mut_ptr().cast(), out_dev.ptr, bytes_out),
                "cuMemcpyDtoH",
            )
            .ok()?;
        }
        Some(from_col_major(&out_host, x_cols, y_cols))
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
        let allocation = unsafe { DeviceAllocation::new(&self.driver, bytes) }?;
        check_cuda(
            unsafe { (self.driver.cu_memcpy_htod)(allocation.ptr, values.as_ptr().cast(), bytes) },
            "cuMemcpyHtoD",
        )
        .ok()?;
        Some(allocation)
    }
}

impl Drop for CublasRuntime {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.cu_ctx_set_current)(self.context);
            let _ = (self.blas.cublas_destroy)(self.handle);
            let _ = (self.driver.cu_ctx_destroy)(self.context);
        }
    }
}

type CublasStatus = i32;
type CublasCreate = unsafe extern "C" fn(*mut usize) -> CublasStatus;
type CublasDestroy = unsafe extern "C" fn(usize) -> CublasStatus;
type CublasDgemm = unsafe extern "C" fn(
    usize,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f64,
    u64,
    i32,
    u64,
    i32,
    *const f64,
    u64,
    i32,
) -> CublasStatus;
type CublasDgemv = unsafe extern "C" fn(
    usize,
    i32,
    i32,
    i32,
    *const f64,
    u64,
    i32,
    u64,
    i32,
    *const f64,
    u64,
    i32,
) -> CublasStatus;
type CublasDdgmm =
    unsafe extern "C" fn(usize, i32, i32, i32, u64, i32, u64, i32, u64, i32) -> CublasStatus;

struct CublasApi {
    cublas_create: CublasCreate,
    cublas_destroy: CublasDestroy,
    cublas_dgemm: CublasDgemm,
    cublas_dgemv: CublasDgemv,
    cublas_ddgmm: CublasDdgmm,
}

impl CublasApi {
    fn load(library: &Library) -> Result<Self, String> {
        unsafe {
            Ok(Self {
                cublas_create: *library
                    .get(b"cublasCreate_v2\0")
                    .map_err(|e| e.to_string())?,
                cublas_destroy: *library
                    .get(b"cublasDestroy_v2\0")
                    .map_err(|e| e.to_string())?,
                cublas_dgemm: *library
                    .get(b"cublasDgemm_v2\0")
                    .map_err(|e| e.to_string())?,
                cublas_dgemv: *library
                    .get(b"cublasDgemv_v2\0")
                    .map_err(|e| e.to_string())?,
                cublas_ddgmm: *library.get(b"cublasDdgmm\0").map_err(|e| e.to_string())?,
            })
        }
    }
}

const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const CUBLAS_SIDE_LEFT: i32 = 0;

#[inline]
fn cublas_op(transpose: bool) -> i32 {
    if transpose { CUBLAS_OP_T } else { CUBLAS_OP_N }
}

fn cublas_library_candidates() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["cublas64_12.dll", "cublas64_11.dll"]
    } else if cfg!(target_os = "macos") {
        &["/usr/local/cuda/lib/libcublas.dylib", "libcublas.dylib"]
    } else {
        &["libcublas.so.12", "libcublas.so.11", "libcublas.so"]
    }
}

fn to_i32(value: usize) -> Option<i32> {
    i32::try_from(value).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn small_operations_do_not_route_to_gpu() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0], [6.0]];
        assert!(try_fast_ab(&a, &b).is_none());
        assert!(try_fast_av(&a, &array![1.0, 2.0]).is_none());
    }

    #[test]
    fn column_major_round_trip_preserves_values() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let packed = to_col_major(&a);
        assert_eq!(packed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(from_col_major(&packed, 2, 3), a);
    }
}

//! Shared CUDA driver bindings used by every cuBLAS / cuSPARSE / cuSOLVER
//! routing module.
//!
//! All library bindings share:
//!
//! * one [`DriverApi`] (resolved once from the dlopen'd `libcuda` handle),
//! * one CUDA context created against the selected device at first use,
//! * the helpers below for byte-size math, column-major layout, and the
//!   RAII [`DeviceAllocation`] wrapper.
//!
//! Future runtimes (cuRAND, cuFFT, NVRTC) borrow the same triple instead
//! of re-initializing the driver and creating another context.

use libloading::Library;
use ndarray::{Array2, ArrayBase, Data, Ix2};
use std::borrow::Cow;

use super::error::GpuError;

pub type CuResult = i32;
// SAFETY: FFI fn-pointer alias for the libcuda driver entry of the same
// name; `unsafe extern "C"` matches the C ABI we resolve via dlsym below
// and propagates the call's obligation (live context, in-range pointers
// and sizes) up to the invoker.
type CuInit = unsafe extern "C" fn(u32) -> CuResult;
// SAFETY: libcuda FFI fn-pointer alias; obligations propagated to invoker.
type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> CuResult;
// SAFETY: libcuda FFI fn-pointer alias; obligations propagated to invoker.
type CuCtxCreate = unsafe extern "C" fn(*mut usize, u32, i32) -> CuResult;
// SAFETY: libcuda FFI fn-pointer alias; obligations propagated to invoker.
type CuCtxSetCurrent = unsafe extern "C" fn(usize) -> CuResult;
// SAFETY: libcuda FFI fn-pointer alias; obligations propagated to invoker.
type CuCtxDestroy = unsafe extern "C" fn(usize) -> CuResult;
// SAFETY: libcuda FFI fn-pointer alias; obligations propagated to invoker.
type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> CuResult;
// SAFETY: libcuda FFI fn-pointer alias; obligations propagated to invoker.
type CuMemFree = unsafe extern "C" fn(u64) -> CuResult;
// SAFETY: libcuda FFI fn-pointer alias; obligations propagated to invoker.
type CuMemcpyHtoD = unsafe extern "C" fn(u64, *const std::ffi::c_void, usize) -> CuResult;
// SAFETY: libcuda FFI fn-pointer alias; obligations propagated to invoker.
type CuMemcpyDtoH = unsafe extern "C" fn(*mut std::ffi::c_void, u64, usize) -> CuResult;

/// Resolved CUDA driver entry points.
pub struct DriverApi {
    pub cu_init: CuInit,
    pub cu_device_get: CuDeviceGet,
    pub cu_ctx_create: CuCtxCreate,
    pub cu_ctx_set_current: CuCtxSetCurrent,
    pub cu_ctx_destroy: CuCtxDestroy,
    pub cu_mem_alloc: CuMemAlloc,
    pub cu_mem_free: CuMemFree,
    pub cu_memcpy_htod: CuMemcpyHtoD,
    pub cu_memcpy_dtoh: CuMemcpyDtoH,
}

impl DriverApi {
    pub fn load(library: &Library) -> Result<Self, GpuError> {
        let sym = |e: libloading::Error| GpuError::DriverSymbolMissing {
            reason: e.to_string(),
        };
        // SAFETY: each `library.get` returns a libloading::Symbol whose
        // signature we are asserting matches the libcuda export of the
        // same name; deref'ing yields the raw fn pointer. The `library`
        // is the process-static dlopen handle from load_static_library,
        // so the pointers we copy out stay valid for the process.
        unsafe {
            Ok(Self {
                cu_init: *library.get(b"cuInit\0").map_err(sym)?,
                cu_device_get: *library.get(b"cuDeviceGet\0").map_err(sym)?,
                cu_ctx_create: *library.get(b"cuCtxCreate_v2\0").map_err(sym)?,
                cu_ctx_set_current: *library.get(b"cuCtxSetCurrent\0").map_err(sym)?,
                cu_ctx_destroy: *library.get(b"cuCtxDestroy_v2\0").map_err(sym)?,
                cu_mem_alloc: *library.get(b"cuMemAlloc_v2\0").map_err(sym)?,
                cu_mem_free: *library.get(b"cuMemFree_v2\0").map_err(sym)?,
                cu_memcpy_htod: *library.get(b"cuMemcpyHtoD_v2\0").map_err(sym)?,
                cu_memcpy_dtoh: *library.get(b"cuMemcpyDtoH_v2\0").map_err(sym)?,
            })
        }
    }
}

/// Shared CUDA driver + selected-device context used by every library
/// runtime. Created once per process; subsequent library handles
/// (`cublasCreate_v2`, `cusolverDnCreate`, `cusparseCreate`) attach to
/// the same context instead of building their own.
pub struct CudaWorkingState {
    /// Resolved CUDA driver entry points. The underlying dlopen'd
    /// `libcuda` is `Box::leak`'d inside [`load_static_library`], so the
    /// fn pointers in `api` stay valid for the process lifetime without
    /// any owning field here.
    pub api: DriverApi,
    /// Primary CUDA context for the selected device. Library runtimes
    /// must `cuCtxSetCurrent` on it before issuing work.
    pub context: usize,
}

impl CudaWorkingState {
    /// Initialize the driver and create one context against `device_ordinal`.
    /// Returns `None` if libcuda can't be loaded, a required symbol is
    /// missing, or any of `cuInit / cuDeviceGet / cuCtxCreate` fails.
    pub fn init(device_ordinal: usize) -> Option<Self> {
        let ordinal = to_i32(device_ordinal)?;
        let library = load_static_library(cuda_library_candidates()).ok()?;
        let api = DriverApi::load(library).ok()?;
        // SAFETY: api was just resolved from the live libcuda handle;
        // we pass in-range device ordinals and pointers to local stack
        // slots that outlive each call, satisfying the driver-API contract.
        unsafe {
            check_cuda((api.cu_init)(0), "cuInit").ok()?;
            let mut device = 0_i32;
            check_cuda((api.cu_device_get)(&mut device, ordinal), "cuDeviceGet").ok()?;
            let mut context = 0_usize;
            check_cuda((api.cu_ctx_create)(&mut context, 0, device), "cuCtxCreate").ok()?;
            Some(Self { api, context })
        }
    }

    /// Bind this context to the calling thread. Library runtimes call
    /// this before issuing work because cuCtxSetCurrent is per-thread.
    #[inline]
    pub fn set_current(&self) -> Result<(), GpuError> {
        check_cuda(
            // SAFETY: self.context was produced by cuCtxCreate in init()
            // and lives until Self::drop; the driver fn pointer was
            // resolved against the same libcuda handle.
            unsafe { (self.api.cu_ctx_set_current)(self.context) },
            "cuCtxSetCurrent",
        )
    }
}

impl Drop for CudaWorkingState {
    fn drop(&mut self) {
        // Only fires at process shutdown via the runtime's `OnceLock`;
        // best-effort cleanup so cuda-gdb / nvidia-smi see no dangling
        // context. Errors are swallowed because the process is on its
        // way out anyway.
        // SAFETY: self.context was produced by cuCtxCreate in init() and
        // is destroyed exactly once here as part of Self::drop.
        unsafe {
            drop((self.api.cu_ctx_destroy)(self.context));
        }
    }
}

/// RAII device allocation: frees on drop via `cuMemFree_v2`.
pub struct DeviceAllocation<'a> {
    driver: &'a DriverApi,
    pub ptr: u64,
}

impl<'a> DeviceAllocation<'a> {
    /// Allocate `bytes` of device memory. Caller is responsible for context
    /// `cuCtxSetCurrent` having been issued before this call.
    pub unsafe fn new(driver: &'a DriverApi, bytes: usize) -> Option<Self> {
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
            drop((self.driver.cu_mem_free)(self.ptr));
        }
    }
}

#[inline]
pub fn check_cuda(result: CuResult, name: &str) -> Result<(), GpuError> {
    if result == 0 {
        Ok(())
    } else {
        Err(GpuError::DriverCallFailed {
            reason: format!("{name} failed with CUDA driver error {result}"),
        })
    }
}

fn load_library(candidates: &[&str]) -> Result<Library, GpuError> {
    for candidate in candidates {
        if let Ok(library) = unsafe { Library::new(*candidate) } {
            return Ok(library);
        }
    }
    Err(GpuError::DriverLibraryUnavailable {
        reason: format!("could not load any of: {}", candidates.join(", ")),
    })
}

/// Like [`load_library`] but intentionally leaks the result so the dlopen
/// mapping stays alive for the process. Use this when the caller's fn-pointer
/// table needs to stay valid forever (i.e. for every library binding we
/// resolve at startup).
pub fn load_static_library(candidates: &[&str]) -> Result<&'static Library, GpuError> {
    let raw = Box::into_raw(Box::new(load_library(candidates)?));
    // SAFETY: deliberate process-lifetime leak. `raw` was just produced by
    // `Box::into_raw`, so it is a valid, properly aligned, exclusive pointer
    // and the reborrow yields a `&'static Library` that no other code holds.
    Ok(unsafe { &*raw })
}

pub fn cuda_library_candidates() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["nvcuda.dll"]
    } else if cfg!(target_os = "macos") {
        &["/usr/local/cuda/lib/libcuda.dylib", "libcuda.dylib"]
    } else {
        &["libcuda.so.1", "libcuda.so"]
    }
}

#[inline]
pub fn bytes_len<T>(len: usize) -> Option<usize> {
    len.checked_mul(std::mem::size_of::<T>())
}

#[inline]
pub fn to_i32(value: usize) -> Option<i32> {
    i32::try_from(value).ok()
}

#[inline]
pub fn to_i64(value: usize) -> Option<i64> {
    i64::try_from(value).ok()
}

/// Repack a 2D `ndarray::ArrayBase` (row-major) into the column-major
/// layout expected by every cuBLAS / cuSOLVER entry point.
///
/// Walks each column once via ndarray's iter (no per-element bounds checks)
/// and extends into a pre-sized `Vec`. On biobank-shape inputs (n≈3×10⁵,
/// p≈35) this replaces a per-element `a[[row, col]]` indexing loop that
/// dominated the host side of every GPU dispatch.
///
/// Fast path: if the input is already F-order (column-major, contiguous in
/// memory-order), borrow its raw buffer directly — no allocation, no copy.
/// Standard row-major ndarrays still go through the permutation path.
pub fn to_col_major<'a, S: Data<Elem = f64>>(a: &'a ArrayBase<S, Ix2>) -> Cow<'a, [f64]> {
    let (rows, cols) = a.dim();
    let strides = a.strides();
    // F-order contiguous: column stride == 1, row stride == rows.
    // `as_slice_memory_order` confirms the buffer is contiguous in memory.
    if rows > 0 && cols > 0 && strides[0] == 1 && strides[1] == rows as isize {
        if let Some(slice) = a.as_slice_memory_order() {
            return Cow::Borrowed(slice);
        }
    }
    let mut out: Vec<f64> = Vec::with_capacity(rows.saturating_mul(cols));
    for col in 0..cols {
        out.extend(a.column(col).iter().copied());
    }
    Cow::Owned(out)
}

/// Convert a column-major flat buffer back into row-major `Array2<f64>`.
pub fn from_col_major_inplace(values: &[f64], out: &mut Array2<f64>) {
    let (rows, cols) = out.dim();
    debug_assert_eq!(values.len(), rows.saturating_mul(cols));
    for col in 0..cols {
        let src = ndarray::ArrayView1::from(&values[col * rows..(col + 1) * rows]);
        out.column_mut(col).assign(&src);
    }
}

pub fn from_col_major(values: &[f64], rows: usize, cols: usize) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((rows, cols));
    from_col_major_inplace(values, &mut out);
    out
}

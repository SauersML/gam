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
#[cfg(target_os = "linux")]
use libloading::os::unix::{Library as UnixLibrary, RTLD_GLOBAL, RTLD_NOW};
use ndarray::{Array2, ArrayBase, Data, Ix2};
use std::borrow::Cow;
use std::path::Path;
#[cfg(target_os = "linux")]
use std::path::PathBuf;
use std::sync::OnceLock;

use super::gpu_error::GpuError;
use crate::gpu_err;

pub type CuResult = i32;
// SAFETY: libcuda FFI fn-pointer alias matching the C ABI we dlsym
// below; the `unsafe` qualifier propagates the call-site obligations
// (live context, valid pointers/sizes) up to each invoker.
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
    pub fn load(library: &'static Library) -> Result<Self, GpuError> {
        let sym = |e: libloading::Error| GpuError::DriverSymbolMissing {
            reason: e.to_string(),
        };
        // SAFETY: each library.get asserts a signature matching the
        // libcuda export of the same name; deref yields raw fn pointers
        // backed by a process-static handle that cannot be unloaded.
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
    /// Resolved CUDA driver entry points. The underlying dlopen'd `libcuda`
    /// is retained by the static loader, so the fn pointers in `api` stay
    /// valid for the process lifetime without any owning field here.
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
        let library = load_static_cuda_driver_library().ok()?;
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
            (self.api.cu_ctx_destroy)(self.context);
        }
    }
}

/// RAII device allocation: frees on drop via `cuMemFree_v2`.
pub struct DeviceAllocation<'a> {
    state: &'a CudaWorkingState,
    pub ptr: u64,
}

impl<'a> DeviceAllocation<'a> {
    /// Allocate `bytes` of device memory. Caller is responsible for context
    /// `cuCtxSetCurrent` having been issued for `state.context` before this call.
    // SAFETY: marked `unsafe fn` because the caller must guarantee that a
    // CUDA context is current on this thread; borrowing state also keeps
    // the owning context alive until this allocation has been dropped.
    pub unsafe fn new(state: &'a CudaWorkingState, bytes: usize) -> Option<Self> {
        let mut ptr = 0_u64;
        check_cuda(
            // SAFETY: &mut ptr is a valid u64 slot living to the end of
            // this fn; `state.context` is current by the caller contract;
            // `bytes` is the caller-requested allocation size.
            unsafe { (state.api.cu_mem_alloc)(&mut ptr, bytes) },
            "cuMemAlloc",
        )
        .ok()?;
        Some(Self { state, ptr })
    }
}

impl Drop for DeviceAllocation<'_> {
    fn drop(&mut self) {
        // Re-bind the owning context before freeing: cuMemFree requires a
        // current context on the calling thread, and Drop may fire from a
        // different thread than `new`. We then ALWAYS attempt the free,
        // even if set_current failed (e.g. driver was deinitialized at
        // shutdown). Skipping the free on a transient set_current failure
        // would silently leak device memory — never acceptable on GPU.
        // SAFETY: state.context was produced by cuCtxCreate and is kept
        // alive for at least the duration of this borrow; the driver fn
        // pointers were resolved against the same live libcuda handle.
        unsafe {
            (self.state.api.cu_ctx_set_current)(self.state.context);
        }
        // SAFETY: self.ptr was produced by cuMemAlloc inside Self::new
        // while state.context was current; we have just re-bound that same
        // context (best-effort) and this RAII owner frees exactly once.
        unsafe {
            (self.state.api.cu_mem_free)(self.ptr);
        }
    }
}

#[inline]
pub fn check_cuda(result: CuResult, name: &str) -> Result<(), GpuError> {
    if result == 0 {
        Ok(())
    } else {
        Err(gpu_err!("{name} failed with CUDA driver error {result}"))
    }
}

/// Returns whether the platform loader can open a CUDA driver library.
///
/// This deliberately uses gam's own `libloading` probe rather than
/// `cudarc::driver::sys::is_culib_present()`: cudarc 0.19's generated
/// dynamic-loader helpers are exactly what emit the noisy
/// `panic_no_lib_found` message when a CPU-only host lacks `libcuda`.
/// Runtime availability checks need to stay completely outside cudarc until
/// this function has established that the driver shared library exists.
#[must_use]
pub fn cuda_driver_library_present() -> bool {
    load_library_names(&cuda_library_candidate_names()).is_ok()
}

fn load_library_names(candidates: &[String]) -> Result<Library, GpuError> {
    for candidate in candidates {
        // SAFETY: Library::new runs the library's loader initializer; we
        // only pass CUDA driver candidates discovered from fixed NVIDIA
        // driver directories or canonical libcuda sonames.
        if let Ok(library) = unsafe { Library::new(candidate) } {
            return Ok(library);
        }
    }
    Err(GpuError::DriverLibraryUnavailable {
        reason: format!("could not load any of: {}", candidates.join(", ")),
    })
}

fn load_static_cuda_driver_library() -> Result<&'static Library, GpuError> {
    static LIBRARY: OnceLock<Result<Library, GpuError>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| load_library_names(&cuda_library_candidate_names()))
        .as_ref()
        .map_err(Clone::clone)
}

pub fn preload_cuda_driver() -> Result<(), String> {
    static PRELOAD: OnceLock<Result<(), String>> = OnceLock::new();
    PRELOAD
        .get_or_init(|| {
            load_static_cuda_driver_library()
                .map(|_| ())
                .map_err(|err| err.to_string())
        })
        .clone()
}

#[cfg(target_os = "linux")]
fn preload_cuda_userspace_libraries() -> Result<(), String> {
    static PRELOAD: OnceLock<Result<Vec<UnixLibrary>, String>> = OnceLock::new();
    PRELOAD
        .get_or_init(|| {
            let paths = cuda_userspace_preload_paths();
            if paths.is_empty() {
                return Ok(Vec::new());
            }
            let mut loaded = Vec::new();
            for path in paths {
                // SAFETY: these candidates are CUDA userspace libraries found
                // in canonical toolkit directories or pip's nvidia-*-cu12
                // wheel layout. RTLD_GLOBAL is required so transitive deps
                // such as libcusolver -> libnvJitLink resolve without an
                // LD_LIBRARY_PATH mutation.
                match unsafe { UnixLibrary::open(Some(&path), RTLD_NOW | RTLD_GLOBAL) } {
                    Ok(library) => loaded.push(library),
                    Err(err) => {
                        return Err(format!(
                            "could not preload CUDA userspace library {}: {err}",
                            path.display()
                        ));
                    }
                }
            }
            Ok(loaded)
        })
        .as_ref()
        .map(|_| ())
        .map_err(Clone::clone)
}

/// Returns whether the platform loader can open the named CUDA compute
/// library (`cublas`, `cusolver`, `cusparse`).
///
/// cudarc 0.19 attempts to lazy-load these via its own generated
/// `panic_no_lib_found` helpers the first time `CudaBlas::new` /
/// `DnHandle::new` / cuSPARSE handle creation is invoked. On a host that
/// has only the CUDA *driver* (e.g. large-scale workbench images expose
/// `libcuda.so.1` but no cuBLAS at all), those calls panic out of the
/// PyO3 FFI boundary instead of returning a typed error.
///
/// `GpuRuntime::probe()` calls this for every compute library it depends
/// on; failure to load any of them downgrades the runtime to CPU with a
/// `DriverLibraryUnavailable { reason: "lib<name> unavailable" }`, which
/// keeps the panic completely off the call path.
#[must_use]
pub fn cuda_compute_library_present(stem: &str) -> bool {
    #[cfg(target_os = "linux")]
    {
        if preload_cuda_userspace_libraries().is_err() {
            return false;
        }
    }
    load_library_names(&cuda_compute_library_candidate_names(stem)).is_ok()
}

#[cfg(target_os = "linux")]
fn cuda_userspace_preload_paths() -> Vec<PathBuf> {
    let system_dirs = cuda_system_library_dirs();
    for dir in &system_dirs {
        if let Some(stack) = complete_system_cuda_stack(dir) {
            return dedup_paths(stack);
        }
        if let Some(stack) = system_cuda_stack_with_packaged_nvjitlink(dir) {
            return dedup_paths(stack);
        }
    }
    for root in nvidia_package_roots() {
        if let Some(stack) = complete_nvidia_cuda_stack(&root) {
            return dedup_paths(stack);
        }
    }
    Vec::new()
}

fn cuda_compute_library_candidate_names(stem: &str) -> Vec<String> {
    let base = format!("lib{stem}");
    let mut out: Vec<String> = Vec::new();
    // Bare soname forms — exercised by the platform loader against
    // LD_LIBRARY_PATH and the default search dirs.
    out.push(format!("{base}.so"));
    out.push(format!("{base}.so.1"));
    // Major-version walk mirroring cudarc's own candidate list so the
    // preflight agrees with whatever cudarc would have tried next.
    for major in (9..=13).rev() {
        out.push(format!("{base}.so.{major}"));
    }
    #[cfg(target_os = "linux")]
    {
        for dir in cuda_system_library_dirs() {
            out.push(format!("{dir}/{base}.so"));
            for major in (9..=13).rev() {
                out.push(format!("{dir}/{base}.so.{major}"));
            }
            append_versioned_linux_so_candidates(&mut out, Path::new(dir), &base);
        }
        for root in nvidia_package_roots() {
            let lib_dir = root.join(nvidia_component_for_stem(stem)).join("lib");
            out.push(format!("{}/{}.so", lib_dir.display(), base));
            for major in (9..=13).rev() {
                out.push(format!("{}/{}.so.{major}", lib_dir.display(), base));
            }
            append_versioned_linux_so_candidates(&mut out, &lib_dir, &base);
        }
    }
    out
}

#[cfg(target_os = "linux")]
fn cuda_system_library_dirs() -> Vec<&'static str> {
    vec![
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib/wsl/lib",
        "/opt/cuda/lib64",
    ]
}

#[cfg(target_os = "linux")]
fn complete_system_cuda_stack(dir: &str) -> Option<Vec<PathBuf>> {
    let dir = Path::new(dir);
    let stack = vec![
        first_existing(dir, &["libcudart.so.12", "libcudart.so"])?,
        first_existing(dir, &["libnvJitLink.so.12", "libnvJitLink.so"])?,
        first_existing(dir, &["libcublasLt.so.12", "libcublasLt.so"])?,
        first_existing(dir, &["libcublas.so.12", "libcublas.so"])?,
        first_existing(dir, &["libcusparse.so.12", "libcusparse.so"])?,
        first_existing(
            dir,
            &["libcusolver.so.12", "libcusolver.so.11", "libcusolver.so"],
        )?,
    ];
    Some(stack)
}

#[cfg(target_os = "linux")]
fn system_cuda_stack_with_packaged_nvjitlink(dir: &str) -> Option<Vec<PathBuf>> {
    let dir = Path::new(dir);
    let nvjitlink = packaged_nvjitlink_library()?;
    let stack = vec![
        first_existing(dir, &["libcudart.so.12", "libcudart.so"])?,
        nvjitlink,
        first_existing(dir, &["libcublasLt.so.12", "libcublasLt.so"])?,
        first_existing(dir, &["libcublas.so.12", "libcublas.so"])?,
        first_existing(dir, &["libcusparse.so.12", "libcusparse.so"])?,
        first_existing(
            dir,
            &["libcusolver.so.12", "libcusolver.so.11", "libcusolver.so"],
        )?,
    ];
    Some(stack)
}

#[cfg(target_os = "linux")]
fn complete_nvidia_cuda_stack(root: &Path) -> Option<Vec<PathBuf>> {
    let stack = vec![
        first_existing(
            &root.join("cuda_runtime").join("lib"),
            &["libcudart.so.12", "libcudart.so"],
        )?,
        first_existing(
            &root.join("nvjitlink").join("lib"),
            &["libnvJitLink.so.12", "libnvJitLink.so"],
        )?,
        first_existing(
            &root.join("cublas").join("lib"),
            &["libcublasLt.so.12", "libcublasLt.so"],
        )?,
        first_existing(
            &root.join("cublas").join("lib"),
            &["libcublas.so.12", "libcublas.so"],
        )?,
        first_existing(
            &root.join("cusparse").join("lib"),
            &["libcusparse.so.12", "libcusparse.so"],
        )?,
        first_existing(
            &root.join("cusolver").join("lib"),
            &["libcusolver.so.12", "libcusolver.so.11", "libcusolver.so"],
        )?,
    ];
    Some(stack)
}

#[cfg(target_os = "linux")]
fn packaged_nvjitlink_library() -> Option<PathBuf> {
    for root in nvidia_package_roots() {
        let lib_dir = root.join("nvjitlink").join("lib");
        if let Some(path) = first_existing(&lib_dir, &["libnvJitLink.so.12", "libnvJitLink.so"]) {
            return Some(path);
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn nvidia_component_for_stem(stem: &str) -> String {
    match stem {
        "cublas" => "cublas".to_string(),
        "cusolver" => "cusolver".to_string(),
        "cusparse" => "cusparse".to_string(),
        "nvJitLink" | "nvjitlink" => "nvjitlink".to_string(),
        "cudart" | "cuda_runtime" => "cuda_runtime".to_string(),
        _ => stem.to_string(),
    }
}

#[cfg(target_os = "linux")]
fn nvidia_package_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(home) = current_user_home_dir() {
        collect_python_nvidia_roots(home.join(".local/lib"), &mut roots);
    }
    collect_python_nvidia_roots(Path::new("/usr/local/lib").to_path_buf(), &mut roots);
    collect_python_nvidia_roots(Path::new("/usr/lib").to_path_buf(), &mut roots);
    dedup_paths(roots)
}

#[cfg(target_os = "linux")]
fn current_user_home_dir() -> Option<PathBuf> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let uid = status
        .lines()
        .find_map(|line| line.strip_prefix("Uid:"))?
        .split_whitespace()
        .next()?;
    let passwd = std::fs::read_to_string("/etc/passwd").ok()?;
    for line in passwd.lines() {
        let mut fields = line.split(':');
        fields.next()?;
        fields.next()?;
        if fields.next()? != uid {
            continue;
        }
        fields.next()?;
        fields.next()?;
        return Some(PathBuf::from(fields.next()?));
    }
    None
}

#[cfg(target_os = "linux")]
fn collect_python_nvidia_roots(base: PathBuf, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(base) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.starts_with("python") {
            continue;
        }
        for site_dir in ["site-packages", "dist-packages"] {
            let root = path.join(site_dir).join("nvidia");
            if root.exists() {
                out.push(root);
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn first_existing(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    for name in names {
        let path = dir.join(name);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn dedup_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for path in paths {
        let canonical = path.canonicalize().unwrap_or(path);
        if !out.iter().any(|existing| existing == &canonical) {
            out.push(canonical);
        }
    }
    out
}

#[cfg(target_os = "linux")]
fn append_versioned_linux_so_candidates(out: &mut Vec<String>, dir: &Path, base: &str) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let prefix = format!("{base}.so.");
    let mut versioned = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name.starts_with(&prefix) {
            versioned.push(path);
        }
    }
    versioned.sort();
    for path in versioned {
        let candidate = path.to_string_lossy().into_owned();
        if !out.iter().any(|existing| existing == &candidate) {
            out.push(candidate);
        }
    }
}

fn cuda_library_candidate_names() -> Vec<String> {
    let mut out: Vec<String> = cuda_library_candidates()
        .iter()
        .map(|candidate| (*candidate).to_string())
        .collect();
    if cfg!(target_os = "linux") {
        for dir in [
            "/usr/local/nvidia/lib64",
            "/usr/local/nvidia/lib",
            "/usr/local/cuda/compat",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/lib/wsl/lib",
        ] {
            append_versioned_linux_libcuda_candidates(&mut out, Path::new(dir));
        }
    }
    out
}

fn append_versioned_linux_libcuda_candidates(out: &mut Vec<String>, dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut versioned = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name.starts_with("libcuda.so.") && name != "libcuda.so.1" {
            versioned.push(path);
        }
    }
    versioned.sort();
    for path in versioned {
        let candidate = path.to_string_lossy().into_owned();
        if !out.iter().any(|existing| existing == &candidate) {
            out.push(candidate);
        }
    }
}

pub fn cuda_library_candidates() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["nvcuda.dll"]
    } else if cfg!(target_os = "macos") {
        &["/usr/local/cuda/lib/libcuda.dylib", "libcuda.dylib"]
    } else {
        &[
            "/usr/local/nvidia/lib64/libcuda.so.1",
            "/usr/local/nvidia/lib64/libcuda.so",
            "/usr/local/nvidia/lib/libcuda.so.1",
            "/usr/local/nvidia/lib/libcuda.so",
            "/usr/local/cuda/compat/libcuda.so.1",
            "/usr/local/cuda/compat/libcuda.so",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib/x86_64-linux-gnu/libcuda.so",
            "/usr/lib64/libcuda.so.1",
            "/usr/lib64/libcuda.so",
            "/usr/lib/wsl/lib/libcuda.so.1",
            "/usr/lib/wsl/lib/libcuda.so",
            "libcuda.so.1",
            "libcuda.so",
        ]
    }
}

#[inline]
pub fn to_i32(value: usize) -> Option<i32> {
    i32::try_from(value).ok()
}

/// Repack a 2D `ndarray::ArrayBase` (row-major) into the column-major
/// layout expected by every cuBLAS / cuSOLVER entry point.
///
/// Walks each column once via ndarray's iter (no per-element bounds checks)
/// and extends into a pre-sized `Vec`. On large-scale inputs (n≈3×10⁵,
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
    if rows > 0
        && cols > 0
        && strides[0] == 1
        && strides[1] == rows as isize
        && let Some(slice) = a.as_slice_memory_order()
    {
        return Cow::Borrowed(slice);
    }
    let mut out: Vec<f64> = Vec::with_capacity(rows.saturating_mul(cols));
    for col in 0..cols {
        out.extend(a.column(col).iter().copied());
    }
    Cow::Owned(out)
}

/// Convert a column-major flat buffer back into row-major `Array2<f64>`.
pub fn from_col_major_inplace(values: &[f64], out: &mut Array2<f64>) -> Option<()> {
    let (rows, cols) = out.dim();
    if values.len() != rows.checked_mul(cols)? {
        return None;
    }
    for col in 0..cols {
        let src = ndarray::ArrayView1::from(&values[col * rows..(col + 1) * rows]);
        out.column_mut(col).assign(&src);
    }
    Some(())
}

pub fn from_col_major(values: &[f64], rows: usize, cols: usize) -> Option<Array2<f64>> {
    let mut out = Array2::<f64>::zeros((rows, cols));
    from_col_major_inplace(values, &mut out)?;
    Some(out)
}

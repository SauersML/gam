//! Shared CUDA driver presence/loading helpers used by every cuBLAS / cuSPARSE
//! / cuSOLVER routing module.
//!
//! The GPU path uses ONE context model: cudarc's device PRIMARY context
//! (`cuDevicePrimaryCtxRetain`, bound in `device_runtime::cuda_context_for`).
//! cuBLAS/cuSOLVER/cuSPARSE handles attach to that current context; there is no
//! separate user `cuCtxCreate` context (its removal fixed the #1017
//! NOT_INITIALIZED handle failures). This module keeps only the libcuda
//! presence probes, byte-size/layout helpers, and the `check_cuda` status wrap.

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
// NOTE (#1017): the `DriverApi` / `CudaWorkingState` / `DeviceAllocation` cluster
// that lived here was REMOVED. It created a SEPARATE user CUDA context via
// `cuCtxCreate` — distinct from cudarc's device PRIMARY context (cuDevicePrimaryCtxRetain)
// that the live GPU path actually uses — which is the documented cause of the
// cuBLAS/cuSOLVER NOT_INITIALIZED handle failures (handles bind to whichever
// context is current). The cluster had ZERO consumers once the runtime routed
// through `cuda_context_for` (the primary context) in `device_runtime.rs`, so it
// was dead dual-context code. Keep ONE context model: the cudarc primary context.
// Do not reintroduce `cuCtxCreate` for issuing work.

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
    // Cache the probe per stem and KEEP the loaded handle alive for the process
    // lifetime. Dropping the `Library` here dlclose's it; that dlopen+dlclose
    // cycle tears down the compute library's global init state, after which
    // cudarc's own cublasCreate / cusolverDnCreate fail
    // CUBLAS/CUSOLVER_STATUS_NOT_INITIALIZED on the next handle creation (the GPU
    // then silently declines and falls back to CPU). Holding the handle keeps the
    // library mapped and initialized so cudarc reuses it intact.
    static PROBED: OnceLock<std::sync::Mutex<std::collections::HashMap<String, bool>>> =
        OnceLock::new();
    static KEEP_ALIVE: OnceLock<std::sync::Mutex<Vec<Library>>> = OnceLock::new();
    let probed = PROBED.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Ok(cache) = probed.lock() {
        if let Some(&present) = cache.get(stem) {
            return present;
        }
    }
    let present = match load_library_names(&cuda_compute_library_candidate_names(stem)) {
        Ok(library) => {
            if let Ok(mut keep) = KEEP_ALIVE
                .get_or_init(|| std::sync::Mutex::new(Vec::new()))
                .lock()
            {
                keep.push(library);
            }
            true
        }
        Err(_) => false,
    };
    if let Ok(mut cache) = probed.lock() {
        cache.insert(stem.to_string(), present);
    }
    present
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

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
        Err(GpuError::DriverCallFailed {
            reason: format!("{name} failed with CUDA driver error {result}"),
        })
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

/// Bind to a CUDA driver that is ALREADY RESIDENT in this process, if any.
///
/// `RTLD_NOLOAD` makes `dlopen` return a handle only when some other
/// component (e.g. torch) has already mapped the library — it never loads a
/// new copy. Inside a torch process, walking the candidate list below can
/// dlopen a SECOND driver instance (system driver vs the CUDA-toolkit compat
/// driver at `/usr/local/cuda*/compat/`): CUDA contexts created by torch's
/// instance are invisible to ours, which is the measured dual-stack failure
/// "no CUDA context for ordinal 0" (and steering the loader at the compat
/// driver instead breaks torch with error 803) — gam#2259. Binding to the
/// resident copy first guarantees ONE driver instance per process, so gam's
/// runtime shares torch's contexts; a standalone process has nothing
/// resident and falls through to the candidate walk unchanged.
#[cfg(target_os = "linux")]
fn already_resident_cuda_driver() -> Option<Library> {
    const RTLD_NOLOAD: std::os::raw::c_int = 0x4;
    for soname in ["libcuda.so.1", "libcuda.so"] {
        // SAFETY: RTLD_NOLOAD never runs a new loader initializer — it only
        // binds to a library some other component already loaded.
        if let Ok(library) = unsafe { UnixLibrary::open(Some(soname), RTLD_NOW | RTLD_NOLOAD) } {
            return Some(library.into());
        }
    }
    None
}

fn load_library_names(candidates: &[String]) -> Result<Library, GpuError> {
    #[cfg(target_os = "linux")]
    if let Some(resident) = already_resident_cuda_driver() {
        return Ok(resident);
    }
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
            let paths = cuda_userspace_preload_paths()?;
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

/// Require that the platform loader can open the named CUDA compute library
/// (`cublas`, `cusolver`, `cusparse`) from the one selected userspace stack.
///
/// cudarc 0.19 attempts to lazy-load these via its own generated
/// `panic_no_lib_found` helpers the first time `CudaBlas::new` /
/// `DnHandle::new` / cuSPARSE handle creation is invoked. On a host that
/// has only the CUDA *driver* (e.g. large-scale workbench images expose
/// `libcuda.so.1` but no cuBLAS at all), those calls panic out of the
/// PyO3 FFI boundary instead of returning a typed error.
///
/// `GpuRuntime::probe()` calls this for every compute library it depends on;
/// failure retains the exact stack-selection or loader error in the typed GPU
/// refusal and keeps cudarc's panic completely off the call path.
pub fn require_cuda_compute_library(stem: &str) -> Result<(), String> {
    // Cache the probe per stem and KEEP the loaded handle alive for the process
    // lifetime. Dropping the `Library` here dlclose's it; that dlopen+dlclose
    // cycle tears down the compute library's global init state, after which
    // cudarc's own cublasCreate / cusolverDnCreate fail
    // CUBLAS/CUSOLVER_STATUS_NOT_INITIALIZED on the next handle creation (the GPU
    // then silently declines and falls back to CPU). Holding the handle keeps the
    // library mapped and initialized so cudarc reuses it intact.
    static PROBED: OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, Result<(), String>>>,
    > = OnceLock::new();
    static KEEP_ALIVE: OnceLock<std::sync::Mutex<Vec<Library>>> = OnceLock::new();
    let probed = PROBED.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Ok(cache) = probed.lock() {
        if let Some(outcome) = cache.get(stem) {
            return outcome.clone();
        }
    }
    #[cfg(target_os = "linux")]
    preload_cuda_userspace_libraries()?;
    let outcome = match load_library_names(&cuda_compute_library_candidate_names(stem)) {
        Ok(library) => {
            if let Ok(mut keep) = KEEP_ALIVE
                .get_or_init(|| std::sync::Mutex::new(Vec::new()))
                .lock()
            {
                keep.push(library);
            }
            Ok(())
        }
        Err(error) => Err(error.to_string()),
    };
    if let Ok(mut cache) = probed.lock() {
        cache.insert(stem.to_string(), outcome.clone());
    }
    outcome
}

#[cfg(target_os = "linux")]
fn cuda_userspace_preload_paths() -> Result<Vec<PathBuf>, String> {
    // A host package such as PyTorch may already own the process's CUDA
    // userspace stack. Loading gam's system-first stack on top of that maps a
    // second cudart/cuBLAS implementation and splits context/handle ownership.
    // Continue the already-mapped stack instead: pip NVIDIA wheels are spread
    // across component directories under one `nvidia/` root, while a system
    // toolkit keeps this preload set in one directory.
    let mapped = mapped_cuda_userspace_libraries()?;
    if !mapped.is_empty() {
        return complete_mapped_cuda_stack(&mapped);
    }

    let system_dirs = cuda_system_library_dirs();
    for dir in &system_dirs {
        if let Some(stack) = complete_system_cuda_stack(dir) {
            return Ok(dedup_paths(stack));
        }
        if let Some(stack) = system_cuda_stack_with_packaged_nvjitlink(dir) {
            return Ok(dedup_paths(stack));
        }
    }
    for root in nvidia_package_roots() {
        if let Some(stack) = complete_nvidia_cuda_stack(&root) {
            return Ok(dedup_paths(stack));
        }
    }
    Ok(Vec::new())
}

/// The CUDA component a userspace library path belongs to, taken from its
/// SONAME stem (`libcublas.so.12` -> `cublas`, `libnvJitLink.so.12` ->
/// `nvJitLink`). Used to reason about which mapped libraries are the same
/// component sourced from different roots.
#[cfg(target_os = "linux")]
fn cuda_library_component(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_prefix("lib")?.split(".so").next()?;
    if stem.is_empty() {
        None
    } else {
        Some(stem.to_string())
    }
}

/// The CUDA *compute* libraries. These are handle-based and share context /
/// workspace state with whatever copy the process already initialised, so a
/// second copy must never be mapped from a different root — that split is the
/// double-free / `NOT_INITIALIZED` hazard the mapped-stack continuation exists
/// to avoid. `cudart` / `nvJitLink` are runtime/driver-adjacent and tolerate a
/// redundant duplicate (dlopen-by-SONAME binds one deterministically).
#[cfg(target_os = "linux")]
fn is_cuda_compute_component(component: &str) -> bool {
    matches!(component, "cublas" | "cublasLt" | "cusolver" | "cusparse")
}

#[cfg(target_os = "linux")]
fn complete_mapped_cuda_stack(mapped: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    let canonical = |p: &Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());

    let mut candidates = Vec::new();
    for path in mapped {
        if let Some(root) = nvidia_package_root_for_library(path)
            && let Some(stack) = complete_nvidia_cuda_stack(&root)
        {
            candidates.push(stack);
        }
        if let Some(parent) = path.parent()
            && let Some(stack) = complete_system_cuda_stack_path(parent)
        {
            candidates.push(stack);
        }
    }

    // Canonical path + component for every mapped library, computed once.
    let mapped_meta: Vec<(PathBuf, Option<String>)> = mapped
        .iter()
        .map(|m| (canonical(m), cuda_library_component(m)))
        .collect();

    // Fast path: a single complete stack that contains EVERY mapped library.
    // This is the unchanged behaviour for a process whose CUDA userspace all
    // comes from one root (a pure system toolkit or a pure pip-wheel stack).
    for stack in &candidates {
        let stack = dedup_paths(stack.clone());
        let stack_canon: Vec<PathBuf> = stack.iter().map(|c| canonical(c)).collect();
        if mapped_meta
            .iter()
            .all(|(m, _)| stack_canon.iter().any(|c| c == m))
        {
            return Ok(stack);
        }
    }

    // Split-stack path: a process can legitimately map a system `libcudart`
    // (pulled onto the default loader path by ldconfig) alongside a host
    // framework's pip-wheel stack (e.g. PyTorch's `nvidia-*-cu12` wheels), so
    // NO single complete stack contains every mapped path. Continue a stack
    // only when the process is genuinely using ALL of it, and only tolerate a
    // benign duplicate — never a split of the handle-based compute libraries:
    //
    //   (a) every library in the candidate stack is already mapped — we
    //       CONTINUE a stack the process runs on, we never introduce a library
    //       from a partially-present root; and
    //   (b) every mapped library is either part of that stack or a redundant
    //       duplicate of a NON-compute component (`cudart` / `nvJitLink`) the
    //       stack already provides. A mapped compute library (`cuBLAS` /
    //       `cuSOLVER` / `cuSPARSE`) from another root is a genuine split and
    //       disqualifies the candidate — mapping a second copy is the
    //       double-free / `NOT_INITIALIZED` hazard this continuation avoids.
    //
    // Libraries mixed across two partially-mapped roots satisfy neither, so the
    // conservative refusal below still fires for them.
    for stack in candidates {
        let stack = dedup_paths(stack);
        let stack_canon: Vec<PathBuf> = stack.iter().map(|c| canonical(c)).collect();
        let fully_mapped = stack_canon
            .iter()
            .all(|s| mapped_meta.iter().any(|(m, _)| m == s));
        if !fully_mapped {
            continue;
        }
        let consistent = mapped_meta.iter().all(|(m, component)| {
            if stack_canon.iter().any(|s| s == m) {
                return true;
            }
            match component {
                Some(component) if !is_cuda_compute_component(component) => stack
                    .iter()
                    .any(|s| cuda_library_component(s).as_deref() == Some(component)),
                _ => false,
            }
        });
        if consistent {
            return Ok(stack);
        }
    }

    Err(format!(
        "CUDA userspace is already mapped from no single complete stack: {}",
        mapped
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

#[cfg(target_os = "linux")]
fn mapped_cuda_userspace_libraries() -> Result<Vec<PathBuf>, String> {
    let maps = std::fs::read_to_string("/proc/self/maps")
        .map_err(|error| format!("cannot inspect mapped CUDA userspace libraries: {error}"))?;
    let mut mapped = Vec::new();
    for line in maps.lines() {
        let Some(raw_path) = line.split_whitespace().last() else {
            continue;
        };
        if !raw_path.starts_with('/') {
            continue;
        }
        let path = PathBuf::from(raw_path);
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if [
            "libcudart.so",
            "libnvJitLink.so",
            "libcublasLt.so",
            "libcublas.so",
            "libcusparse.so",
            "libcusolver.so",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix))
        {
            mapped.push(path);
        }
    }
    Ok(dedup_paths(mapped))
}

#[cfg(target_os = "linux")]
fn nvidia_package_root_for_library(path: &Path) -> Option<PathBuf> {
    path.ancestors()
        .find(|ancestor| ancestor.file_name().and_then(|name| name.to_str()) == Some("nvidia"))
        .map(Path::to_path_buf)
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
    complete_system_cuda_stack_path(Path::new(dir))
}

#[cfg(target_os = "linux")]
fn complete_system_cuda_stack_path(dir: &Path) -> Option<Vec<PathBuf>> {
    let stack = vec![
        first_existing(dir, &["libcudart.so.13", "libcudart.so.12", "libcudart.so"])?,
        first_existing(
            dir,
            &[
                "libnvJitLink.so.13",
                "libnvJitLink.so.12",
                "libnvJitLink.so",
            ],
        )?,
        first_existing(
            dir,
            &["libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"],
        )?,
        first_existing(dir, &["libcublas.so.13", "libcublas.so.12", "libcublas.so"])?,
        first_existing(
            dir,
            &["libcusparse.so.13", "libcusparse.so.12", "libcusparse.so"],
        )?,
        first_existing(
            dir,
            &[
                "libcusolver.so.13",
                "libcusolver.so.12",
                "libcusolver.so.11",
                "libcusolver.so",
            ],
        )?,
    ];
    Some(stack)
}

#[cfg(target_os = "linux")]
fn system_cuda_stack_with_packaged_nvjitlink(dir: &str) -> Option<Vec<PathBuf>> {
    let dir = Path::new(dir);
    let nvjitlink = packaged_nvjitlink_library()?;
    let stack = vec![
        first_existing(dir, &["libcudart.so.13", "libcudart.so.12", "libcudart.so"])?,
        nvjitlink,
        first_existing(
            dir,
            &["libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"],
        )?,
        first_existing(dir, &["libcublas.so.13", "libcublas.so.12", "libcublas.so"])?,
        first_existing(
            dir,
            &["libcusparse.so.13", "libcusparse.so.12", "libcusparse.so"],
        )?,
        first_existing(
            dir,
            &[
                "libcusolver.so.13",
                "libcusolver.so.12",
                "libcusolver.so.11",
                "libcusolver.so",
            ],
        )?,
    ];
    Some(stack)
}

#[cfg(target_os = "linux")]
fn complete_nvidia_cuda_stack(root: &Path) -> Option<Vec<PathBuf>> {
    let stack = vec![
        first_existing(
            &root.join("cuda_runtime").join("lib"),
            &["libcudart.so.13", "libcudart.so.12", "libcudart.so"],
        )?,
        first_existing(
            &root.join("nvjitlink").join("lib"),
            &[
                "libnvJitLink.so.13",
                "libnvJitLink.so.12",
                "libnvJitLink.so",
            ],
        )?,
        first_existing(
            &root.join("cublas").join("lib"),
            &["libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"],
        )?,
        first_existing(
            &root.join("cublas").join("lib"),
            &["libcublas.so.13", "libcublas.so.12", "libcublas.so"],
        )?,
        first_existing(
            &root.join("cusparse").join("lib"),
            &["libcusparse.so.13", "libcusparse.so.12", "libcusparse.so"],
        )?,
        first_existing(
            &root.join("cusolver").join("lib"),
            &[
                "libcusolver.so.13",
                "libcusolver.so.12",
                "libcusolver.so.11",
                "libcusolver.so",
            ],
        )?,
    ];
    Some(stack)
}

#[cfg(target_os = "linux")]
fn packaged_nvjitlink_library() -> Option<PathBuf> {
    for root in nvidia_package_roots() {
        let lib_dir = root.join("nvjitlink").join("lib");
        if let Some(path) = first_existing(
            &lib_dir,
            &[
                "libnvJitLink.so.13",
                "libnvJitLink.so.12",
                "libnvJitLink.so",
            ],
        ) {
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

/// Borrow (or pack) a 2D array's buffer in ROW-major (C) order.
///
/// The col-major dual of [`to_col_major`]: when the input is already
/// C-contiguous its raw buffer IS the row-major flat layout, so this borrows
/// it with no allocation or copy. Non-contiguous / F-order inputs are packed
/// row by row.
///
/// This is the host-transpose-free upload path. A row-major `(r × c)` buffer,
/// reinterpreted as a column-major buffer, is exactly the transpose `(c × r)`
/// of the logical matrix — which is what the swapped-operand cuBLAS GEMM
/// (`Cᵀ = Bᵀ·Aᵀ`) consumes, letting both the design upload and the result
/// download skip the O(r·c) scalar permutation that dominated tall-skinny
/// GEMMs on the host.
pub fn to_row_major<'a, S: Data<Elem = f64>>(a: &'a ArrayBase<S, Ix2>) -> Cow<'a, [f64]> {
    let (rows, cols) = a.dim();
    let strides = a.strides();
    // C-order contiguous: row stride == cols, column stride == 1.
    if rows > 0
        && cols > 0
        && strides[1] == 1
        && strides[0] == cols as isize
        && let Some(slice) = a.as_slice_memory_order()
    {
        return Cow::Borrowed(slice);
    }
    let mut out: Vec<f64> = Vec::with_capacity(rows.saturating_mul(cols));
    for row in 0..rows {
        out.extend(a.row(row).iter().copied());
    }
    Cow::Owned(out)
}

/// Wrap a row-major flat buffer of shape `(rows, cols)` as an `Array2<f64>`
/// without permutation. The buffer is consumed (no copy when its length
/// matches). Returns `None` on a length mismatch.
pub fn array_from_row_major(values: Vec<f64>, rows: usize, cols: usize) -> Option<Array2<f64>> {
    if values.len() != rows.checked_mul(cols)? {
        return None;
    }
    Array2::from_shape_vec((rows, cols), values).ok()
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[cfg(target_os = "linux")]
    fn fake_nvidia_stack(root: &Path) -> Vec<PathBuf> {
        let libraries = [
            ("cuda_runtime", "libcudart.so.12"),
            ("nvjitlink", "libnvJitLink.so.12"),
            ("cublas", "libcublasLt.so.12"),
            ("cublas", "libcublas.so.12"),
            ("cusparse", "libcusparse.so.12"),
            ("cusolver", "libcusolver.so.11"),
        ];
        libraries
            .into_iter()
            .map(|(component, name)| {
                let path = root.join(component).join("lib").join(name);
                std::fs::create_dir_all(path.parent().expect("library parent"))
                    .expect("create fake CUDA component directory");
                std::fs::write(&path, []).expect("create fake CUDA library");
                path
            })
            .collect()
    }

    #[test]
    fn to_i32_fits_small_value() {
        assert_eq!(to_i32(0), Some(0));
        assert_eq!(to_i32(42), Some(42));
        assert_eq!(to_i32(i32::MAX as usize), Some(i32::MAX));
    }

    #[test]
    fn to_i32_overflows_returns_none() {
        assert_eq!(to_i32(i32::MAX as usize + 1), None);
    }

    #[test]
    fn to_col_major_2x3_row_major() {
        // Row-major [[1,2,3],[4,5,6]] → col-major [1,4,2,5,3,6]
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let col = to_col_major(&a);
        assert_eq!(&*col, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn to_col_major_identity_roundtrip() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let col = to_col_major(&a);
        assert_eq!(&*col, &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn from_col_major_2x3_roundtrip() {
        let original = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let col = to_col_major(&original);
        let recovered = from_col_major(&col, 2, 3).expect("should succeed");
        assert_eq!(recovered, original);
    }

    #[test]
    fn from_col_major_wrong_length_returns_none() {
        // 2x3 = 6 elements, but only 5 provided
        assert!(from_col_major(&[1.0, 2.0, 3.0, 4.0, 5.0], 2, 3).is_none());
    }

    #[test]
    fn from_col_major_inplace_mismatched_buffer_returns_none() {
        let mut out = Array2::<f64>::zeros((3, 3));
        let short = vec![1.0_f64; 8]; // 9 expected, 8 given
        assert!(from_col_major_inplace(&short, &mut out).is_none());
    }

    #[test]
    fn from_col_major_single_element() {
        let result = from_col_major(&[7.0], 1, 1).expect("should succeed");
        assert_eq!(result[[0, 0]], 7.0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn mapped_pytorch_stack_is_continued_as_one_complete_stack() {
        let temp = tempfile::tempdir().expect("temporary CUDA tree");
        let root = temp.path().join("site-packages").join("nvidia");
        let stack = fake_nvidia_stack(&root);
        let mapped = vec![stack[0].clone(), stack[3].clone()];

        let selected = complete_mapped_cuda_stack(&mapped).expect("one mapped NVIDIA root");

        assert_eq!(selected.len(), stack.len());
        assert!(mapped.iter().all(|path| {
            let canonical = path.canonicalize().expect("canonical fake library");
            selected.contains(&canonical)
        }));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn mapped_mixed_cuda_stacks_are_refused() {
        let temp = tempfile::tempdir().expect("temporary CUDA tree");
        let first = fake_nvidia_stack(&temp.path().join("first").join("nvidia"));
        let second = fake_nvidia_stack(&temp.path().join("second").join("nvidia"));
        let mapped = vec![first[0].clone(), second[3].clone()];

        let error = complete_mapped_cuda_stack(&mapped)
            .expect_err("libraries from two mapped roots must not be mixed");

        assert!(error.contains("no single complete stack"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn mapped_pytorch_stack_with_redundant_system_cudart_is_continued() {
        // A process can map a system `libcudart` (pulled onto the default
        // loader path by ldconfig) alongside PyTorch's complete pip-wheel
        // stack. The whole pip stack is mapped; the extra system cudart is a
        // redundant duplicate of a non-compute component, so the pip stack is
        // continued rather than the GPU being refused (gam issue #2259).
        let temp = tempfile::tempdir().expect("temporary CUDA tree");
        let pip = fake_nvidia_stack(&temp.path().join("site-packages").join("nvidia"));

        // A stray system cudart from an unrelated toolkit directory (no
        // complete stack of its own next to it).
        let sys_dir = temp.path().join("usr").join("local").join("cuda").join("lib");
        std::fs::create_dir_all(&sys_dir).expect("system lib dir");
        let sys_cudart = sys_dir.join("libcudart.so.12");
        std::fs::write(&sys_cudart, []).expect("system cudart");

        // Everything from the pip stack is mapped, plus the redundant cudart.
        let mut mapped = pip.clone();
        mapped.push(sys_cudart);

        let selected =
            complete_mapped_cuda_stack(&mapped).expect("continue the fully-mapped pip stack");

        assert_eq!(selected.len(), pip.len());
        assert!(pip.iter().all(|path| {
            let canonical = path.canonicalize().expect("canonical fake library");
            selected.contains(&canonical)
        }));
    }
}

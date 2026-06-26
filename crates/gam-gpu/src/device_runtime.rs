#[cfg(target_os = "linux")]
use std::collections::HashMap;
#[cfg(target_os = "linux")]
use std::panic::{self, AssertUnwindSafe, catch_unwind};
use std::sync::OnceLock;
#[cfg(target_os = "linux")]
use std::sync::{Arc, Mutex};

use super::device::GpuDeviceInfo;
use super::gpu_error::GpuError;
use super::policy::GpuDispatchPolicy;
#[cfg(target_os = "linux")]
use cudarc::driver::{CudaContext, result, sys};

#[path = "runtime_diagnostics.rs"]
pub(crate) mod diagnostics;

#[derive(Clone, Debug)]
#[must_use]
pub struct GpuRuntime {
    /// Highest-scoring probed CUDA device. Existing dispatch code routes
    /// one-shot kernels through this device.
    pub device: GpuDeviceInfo,
    /// All usable CUDA devices discovered at probe time, ordered by score.
    pub devices: Vec<GpuDeviceInfo>,
    pub policy: GpuDispatchPolicy,
    pub memory_budget_bytes: usize,
}

static CPU_REASON: OnceLock<String> = OnceLock::new();

/// Install a process-wide panic hook (idempotent) that drops cudarc's
/// `panic_no_lib_found` message instead of writing it to stderr. All other
/// panics flow to the previously installed hook unchanged. The site cudarc
/// 0.19 panics from is `cudarc-0.19.7/src/lib.rs:200` inside its dynamic
/// loader; messages from that path start with `Unable to dynamically load`.
/// Caller code wraps the same cudarc entry points in `catch_unwind`, so the
/// panic is recovered — this hook just prevents the stderr noise that made
/// operators think the fit had crashed.
#[cfg(target_os = "linux")]
fn install_cudarc_panic_filter() {
    static HOOK_INSTALLED: OnceLock<()> = OnceLock::new();
    HOOK_INSTALLED.get_or_init(|| {
        let prior = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            let payload = info.payload();
            let message = payload
                .downcast_ref::<&'static str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("");
            if message.starts_with("Unable to dynamically load") {
                return;
            }
            prior(info);
        }));
    });
}

impl GpuRuntime {
    pub fn probe() -> Result<Option<Self>, GpuError> {
        if super::global_policy() == super::GpuPolicy::Off {
            Self::record_cpu_reason("GPU policy is off");
            diagnostics::log_cuda_disabled("GPU policy is off");
            return Ok(None);
        }

        #[cfg(not(target_os = "linux"))]
        {
            let reason = "CUDA support not compiled into this build";
            Self::record_cpu_reason(reason);
            diagnostics::log_cuda_disabled(reason);
            return Err(GpuError::DriverLibraryUnavailable {
                reason: reason.to_string(),
            });
        }

        #[cfg(target_os = "linux")]
        {
            // `cudarc 0.19`'s entry points lazily initialize the CUDA driver
            // through generated `culib()` helpers. On CPU-only Linux hosts the
            // first such call emits `panic_no_lib_found` before unwinding, which
            // polluted large-scale logs even when the panic was later caught and the
            // fit fell back to CPU. Keep the preflight completely outside
            // cudarc: use gam's own `libloading` probe first, and only touch
            // cudarc after the platform loader can open `libcuda`.
            //
            // The preflight does not always agree with cudarc's own loader
            // candidate list (e.g. large-scale workbench images expose CUDA *runtime*
            // stub libraries under `/usr/local/cuda-*/targets/.../lib` but no
            // driver `libcuda.so` in any loader path), so we additionally
            // install a panic-hook filter that suppresses cudarc's
            // `panic_no_lib_found` message and wrap every cudarc entry point
            // below in `catch_unwind` to convert the panic into a typed
            // `GpuError::DriverCallFailed` instead.
            install_cudarc_panic_filter();
            // #1017 probe-first fix: establish cudarc's primary context P and
            // initialize the CUDA runtime ON IT as the VERY FIRST CUDA action -- before
            // gam's libloading libcuda preload, the compute-lib dlopens, and device_count.
            // The clean cuda_context_for-first path works; the probe-first path failed
            // because a pre-context CUDA touch left the runtime bound to a non-P context,
            // so later cuBLAS/cuSOLVER handle creation on the P-stream returned
            // NOT_INITIALIZED. Making cuda_context_for the first action replicates the
            // working clean path (CudaContext::new loads libcuda + retains the primary +
            // ensure runs the runtime init); on a CPU-only host it returns None cleanly
            // via the panic filter + catch_unwind, and the preload check below still runs.
            let primary_ready = cuda_context_for(0).is_some();
            log::trace!("[GPU] probe pre-init primary context + runtime: {primary_ready}");
            if crate::driver::preload_cuda_driver().is_err() {
                let reason = "libcuda unavailable";
                Self::record_cpu_reason(reason);
                log::info!("[GPU] CUDA acceleration disabled: {reason}");
                diagnostics::log_cuda_disabled(reason);
                return Err(GpuError::DriverLibraryUnavailable {
                    reason: reason.to_string(),
                });
            }

            // Driver-only environments (e.g. large-scale workbench images that expose
            // `libcuda.so.1` but ship no cuBLAS/cuSOLVER/cuSPARSE) used to slip
            // past the libcuda preflight, enable the runtime, and then panic
            // out of cudarc's `panic_no_lib_found` on the first `CudaBlas::new`
            // — the panic crossed the PyO3 FFI boundary as a
            // `ValueError: fit_table panicked inside Rust boundary: Unable to
            // dynamically load the "cublas" shared library`. The compute
            // libraries are dispatch-critical (every cuBLAS / cuSOLVER /
            // cuSPARSE site under `src/gpu/` calls `CudaBlas::new` /
            // `DnHandle::new` / cusparse handle creation eagerly during
            // workspace allocation), so we refuse to advertise GPU unless all
            // three load cleanly here.
            for stem in ["cublas", "cusolver", "cusparse"] {
                if !crate::driver::cuda_compute_library_present(stem) {
                    let reason = format!("lib{stem} unavailable");
                    Self::record_cpu_reason(reason.clone());
                    log::info!("[GPU] CUDA acceleration disabled: {reason}");
                    diagnostics::log_cuda_disabled(&reason);
                    return Err(GpuError::DriverLibraryUnavailable { reason });
                }
            }

            // cudarc 0.19's `culib()` panics via `panic_no_lib_found` when its
            // own (separate from gam's) dynamic-loader candidate list cannot
            // find libcuda — this can happen even after our `preload_cuda_driver`
            // succeeds, for example if our probe loaded a CUDA stub library but
            // cudarc's loader searches a disjoint set of names. Convert any such
            // panic into a typed probe failure so the runtime cleanly disables
            // CUDA and the CPU fallback proceeds without alarming stderr noise.
            let device_count = catch_unwind(AssertUnwindSafe(CudaContext::device_count))
                .map_err(|_| GpuError::DriverLibraryUnavailable {
                    reason: "libcuda unavailable".to_string(),
                })?
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: err.to_string(),
                })?;
            if device_count <= 0 {
                let reason = "CUDA driver reported no devices";
                Self::record_cpu_reason(reason);
                diagnostics::log_cuda_disabled(reason);
                // Surface the no-device state as a structured `DriverCallFailed`
                // so callers wanting a CPU-reason marker can distinguish
                // "policy off" (Ok(None)) from "driver present but no usable
                // hardware" (Err). This keeps `GpuRuntime::probe()` honest: a
                // successful `Ok` always carries at least one device.
                return Err(GpuError::DriverCallFailed {
                    reason: reason.to_string(),
                });
            }

            let mut devices = Vec::new();
            for ordinal in
                0..usize::try_from(device_count).map_err(|_| GpuError::DriverCallFailed {
                    reason: "negative CUDA device count".into(),
                })?
            {
                let ctx = cuda_context_for(ordinal).ok_or_else(|| {
                    gpu_err!("failed to create CUDA context for device {ordinal}")
                })?;
                catch_unwind(AssertUnwindSafe(|| ctx.bind_to_thread()))
                    .map_err(|_| GpuError::DriverLibraryUnavailable {
                        reason: "libcuda unavailable".to_string(),
                    })?
                    .map_err(|err| GpuError::DriverCallFailed {
                        reason: err.to_string(),
                    })?;
                devices.push(
                    catch_unwind(AssertUnwindSafe(|| cuda_device_info(ordinal, &ctx))).map_err(
                        |_| GpuError::DriverLibraryUnavailable {
                            reason: "libcuda unavailable".to_string(),
                        },
                    )??,
                );
            }

            devices.sort_by(|a, b| b.score().total_cmp(&a.score()));
            let Some(device) = devices.first().cloned() else {
                Self::record_cpu_reason("CUDA driver reported no usable devices");
                diagnostics::log_cuda_disabled("CUDA driver reported no usable devices");
                return Ok(None);
            };

            let policy = crate::calibration::calibrated_policy_for_device(&device);
            let memory_budget_bytes = device.memory_budget_bytes();
            diagnostics::log_cuda_enabled(&device, &policy);
            diagnostics::log_cuda_pool(&devices);

            Ok(Some(Self {
                device,
                devices,
                policy,
                memory_budget_bytes,
            }))
        }
    }

    #[must_use]
    pub fn global() -> Option<&'static Self> {
        static RUNTIME: OnceLock<Option<GpuRuntime>> = OnceLock::new();
        RUNTIME
            .get_or_init(|| match Self::probe() {
                Ok(runtime) => runtime,
                Err(err) => {
                    let reason = err.to_string();
                    Self::record_cpu_reason(reason.clone());
                    diagnostics::log_cuda_disabled(&reason);
                    None
                }
            })
            .as_ref()
    }

    #[must_use]
    pub fn is_available() -> bool {
        Self::global().is_some()
    }

    /// Fail-closed accessor for the process-wide runtime under a [`GpuMode`]
    /// contract (issue #1017).
    ///
    /// * [`GpuMode::Required`] — the device MUST be present: when the probe
    ///   found no usable runtime this returns `Err(GpuError::DriverLibraryUnavailable)`
    ///   carrying the recorded CPU reason, so the resident path surfaces a
    ///   structured error instead of silently falling back to the CPU.
    /// * [`GpuMode::Auto`] / [`GpuMode::Off`] — preserve the existing
    ///   probe-first behavior bit-for-bit: this is a thin wrapper over
    ///   [`Self::global`] that maps the `None` case to the same typed error
    ///   without ever forcing the runtime on or changing any numerics. `Auto`
    ///   callers treat the `Err` exactly as they treated `global().is_none()`
    ///   today (fall back to CPU); only the `Required` caller propagates it.
    ///
    /// This does NOT alter `global()`/`cuda_context_for`/`ensure_cuda_runtime_device`;
    /// it only adds the residency gate on top of the working Auto path.
    pub fn global_or_fail(mode: super::GpuMode) -> Result<&'static Self, GpuError> {
        match mode {
            super::GpuMode::Off => Err(GpuError::DriverLibraryUnavailable {
                reason: "GPU residency mode is off".to_string(),
            }),
            super::GpuMode::Auto | super::GpuMode::Required => {
                Self::global().ok_or_else(|| GpuError::DriverLibraryUnavailable {
                    reason: Self::cpu_reason()
                        .unwrap_or("CUDA runtime unavailable")
                        .to_string(),
                })
            }
        }
    }

    #[must_use]
    pub fn policy(&self) -> &GpuDispatchPolicy {
        &self.policy
    }

    #[must_use]
    pub fn selected_device(&self) -> &GpuDeviceInfo {
        &self.device
    }

    #[must_use]
    pub(crate) fn cpu_reason() -> Option<&'static str> {
        CPU_REASON.get().map(String::as_str)
    }

    fn record_cpu_reason(reason: impl Into<String>) {
        CPU_REASON.set(reason.into()).ok();
    }
}

/// Make the CUDA **runtime** API usable on `ordinal`.
///
/// gam drives the GPU through the CUDA *driver* API (cudarc [`CudaContext`]),
/// which materialises the driver primary context but never selects a device for
/// the CUDA *runtime* API. cuBLAS / cuSOLVER are runtime-based, so `cublasCreate`
/// / `cusolverDnCreate` return `CUBLAS_STATUS_NOT_INITIALIZED` /
/// `CUSOLVER_STATUS_NOT_INITIALIZED` until the runtime has a current device —
/// which silently disables *every* GPU linear-algebra path (the dispatch sites
/// map the handle error to `Unavailable` and fall back to CPU). We select the
/// device on the calling host thread (cheap, idempotent) and force one-time
/// runtime primary-context materialisation per device via the canonical
/// `cudaMalloc`/`cudaFree` idiom, so every downstream handle creation succeeds.
#[cfg(target_os = "linux")]
fn ensure_cuda_runtime_device(ordinal: usize) {
    let Ok(o) = i32::try_from(ordinal) else {
        return;
    };
    // SAFETY: the `runtime` cudarc feature is enabled; cudaSetDevice on a valid
    // ordinal is idempotent and per-host-thread.
    let set_rc = unsafe { cudarc::runtime::sys::cudaSetDevice(o) };
    log::trace!("[GPU] runtime cudaSetDevice({o}) -> {set_rc:?}");
    // Materialise the runtime primary context on EVERY call (not once): the driver
    // probe binds cudarc's own context, and cuBLAS/cuSOLVER `*Create` use whatever
    // context is current at creation time, so the runtime device must be reselected
    // and its primary context re-materialised immediately before each handle is made.
    // A 256-byte allocate-then-free is the canonical, ~microsecond way to force it.
    let mut p: *mut core::ffi::c_void = core::ptr::null_mut();
    // SAFETY: forces runtime primary-context creation on the current device.
    let malloc_rc = unsafe { cudarc::runtime::sys::cudaMalloc(&mut p as *mut _ as *mut _, 256) };
    log::trace!("[GPU] runtime cudaMalloc -> {malloc_rc:?}");
    if !p.is_null() {
        // SAFETY: `p` is the live device allocation returned just above.
        let free_rc = unsafe { cudarc::runtime::sys::cudaFree(p) };
        log::trace!("[GPU] runtime cudaFree -> {free_rc:?}");
    }
}

#[cfg(target_os = "linux")]
pub fn cuda_context_for(ordinal: usize) -> Option<Arc<CudaContext>> {
    static CONTEXTS: OnceLock<Mutex<HashMap<usize, Arc<CudaContext>>>> = OnceLock::new();
    let contexts = CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(ctx) = contexts.lock().ok()?.get(&ordinal).cloned() {
        // Bind cudarc's PRIMARY context current on THIS thread BEFORE the runtime
        // materialisation below, so the runtime initialises the same context that
        // new_stream()/CudaBlas::new run cublasCreate against. Without this, on a
        // fresh solve thread the cached path lets the runtime init its own device
        // context, and the later cublasCreate on the primary-context stream fails
        // CUBLAS/CUSOLVER_STATUS_NOT_INITIALIZED (the probe-first GPU-dead bug).
        let bound = catch_unwind(AssertUnwindSafe(|| ctx.bind_to_thread()));
        log::trace!(
            "[GPU] cuda_context_for cached bind ok={}",
            matches!(bound, Ok(Ok(())))
        );
        ensure_cuda_runtime_device(ordinal);
        return Some(ctx);
    }
    // cudarc 0.19 panics from `panic_no_lib_found` if its loader fails to
    // locate libcuda. Demote that to `None` so the runtime probe surfaces a
    // typed `DriverUnavailable` rather than tearing down the worker thread.
    let ctx = catch_unwind(AssertUnwindSafe(|| CudaContext::new(ordinal)))
        .ok()?
        .ok()?;
    let out = {
        let mut guard = contexts.lock().ok()?;
        guard.entry(ordinal).or_insert_with(|| ctx.clone()).clone()
    };
    // CudaContext::new already bound the primary context, but the HashMap may return
    // an entry created on another thread; rebind so the primary context is current on
    // THIS thread before the runtime touch (same probe-first NOT_INITIALIZED guard).
    let bound = catch_unwind(AssertUnwindSafe(|| out.bind_to_thread()));
    log::trace!(
        "[GPU] cuda_context_for fresh bind ok={}",
        matches!(bound, Ok(Ok(())))
    );
    ensure_cuda_runtime_device(ordinal);
    Some(out)
}

#[cfg(target_os = "linux")]
fn cuda_device_info(ordinal: usize, ctx: &CudaContext) -> Result<GpuDeviceInfo, GpuError> {
    result::init().map_err(|err| GpuError::DriverCallFailed {
        reason: err.to_string(),
    })?;
    let device =
        result::device::get(
            i32::try_from(ordinal).map_err(|_| GpuError::DriverCallFailed {
                reason: "device ordinal overflow".into(),
            })?,
        )
        .map_err(|err| GpuError::DriverCallFailed {
            reason: err.to_string(),
        })?;
    let attr = |attribute| -> Result<i32, GpuError> {
        // SAFETY: device comes from cudarc's validated device::get.
        unsafe { result::device::get_attribute(device, attribute) }.map_err(|err| {
            GpuError::DriverCallFailed {
                reason: err.to_string(),
            }
        })
    };
    let (free_mem_bytes, total_mem_bytes) =
        ctx.mem_get_info()
            .map_err(|err| GpuError::DriverCallFailed {
                reason: err.to_string(),
            })?;
    let major = attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let minor = attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    Ok(GpuDeviceInfo {
        ordinal,
        name: result::device::get_name(device).unwrap_or_else(|_| format!("CUDA device {ordinal}")),
        capability: super::device::GpuCapability::from_compute_capability(major, minor),
        sm_count: attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        max_threads_per_sm: attr(
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        )?,
        max_shared_mem_per_block: attr(
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        )
        .unwrap_or(0) as usize,
        l2_cache_bytes: attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
            .unwrap_or(0) as usize,
        total_mem_bytes,
        free_mem_bytes,
        ecc_enabled: attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_ECC_ENABLED)
            .unwrap_or(0)
            != 0,
        integrated: attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_INTEGRATED).unwrap_or(0)
            != 0,
        mig_mode: false,
    })
}

#[cfg(test)]
mod module_path_lock_tests {
    //! Locks the canonical module path for the GPU device runtime so a future
    //! rename is a deliberate, reviewed change (precedent: issue #1157's
    //! "lock module path" tests). This file was renamed from the generic,
    //! colliding `gpu/runtime.rs` to `gpu/device_runtime.rs` under issue #1137.

    #[test]
    fn gpu_device_runtime_module_path_is_canonical() {
        // Resolving `GpuRuntime` through the `device_runtime` module path
        // pins the honest name; if the module is renamed this stops compiling.
        _ = crate::device_runtime::GpuRuntime::is_available();
        let type_name = std::any::type_name::<crate::device_runtime::GpuRuntime>();
        assert!(
            type_name.contains("device_runtime"),
            "GpuRuntime must live in the `device_runtime` module (got {type_name})"
        );
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    /// On a CPU-only host (no `libcuda.dylib` / `libcuda.so` reachable via the
    /// platform loader), exercising every cudarc-touching entry point in this
    /// crate must produce a clean `None`/`Err` and never trigger
    /// `cudarc::panic_no_lib_found`. This is the regression guard for issues
    /// #168 and #176, which observed a `PanicException` escaping the PyO3
    /// boundary on macOS when `sae_manifold_fit(..., atom_basis="duchon")` or
    /// `d_atom=1` ran on a host with no CUDA driver.
    ///
    /// On a host where libcuda *is* present the test still passes — it asserts
    /// only that calls don't panic and that `is_culib_present()` agrees with
    /// `GpuRuntime::is_available()` about the absence of a driver.
    #[test]
    fn cpu_only_host_never_panics_on_gpu_entry_points() {
        // Without libcuda the runtime must report unavailable rather than
        // panicking from inside `culib()`; with libcuda the runtime may or
        // may not have a usable device, but the panic-free contract still
        // holds and the dispatch smoke test below exercises it.
        let culib_present = crate::driver::cuda_driver_library_present();
        if !culib_present {
            assert!(
                !GpuRuntime::is_available(),
                "is_culib_present()=false but GpuRuntime::is_available() returned true; \
                 the probe guard from c10e6636 has regressed and downstream cudarc \
                 calls will panic"
            );
        }

        // Every public GPU dispatch must return a value (no panic) when the
        // runtime is unavailable. We use minimum-size inputs so a host that
        // *does* have a GPU still passes (workload below dispatch threshold
        // → returns None / Err / CPU fallback the same way).
        use ndarray::{Array1, Array2};
        let a = Array2::<f64>::zeros((4, 3));
        let b = Array2::<f64>::zeros((3, 2));
        let v = Array1::<f64>::zeros(3);
        let w = Array1::<f64>::ones(4);

        // gpu::linalg_dispatch dispatchers
        crate::try_fast_ab(a.view(), b.view());
        crate::try_fast_av(a.view(), v.view());
        crate::try_fast_atv(a.view(), w.view());
        let mut chol_in = Array2::<f64>::eye(3);
        crate::try_cholesky_lower_inplace(&mut chol_in);

        // gpu::solver Cholesky entry points
        let h = Array2::<f64>::eye(3);
        let rhs = Array2::<f64>::zeros((3, 1));
        let solve_outcome = crate::solver::cholesky_solve_gpu(h.view(), rhs.view());
        let factor_outcome = crate::solver::cholesky_lower_gpu(h.view());
        if !GpuRuntime::is_available() {
            assert!(
                solve_outcome.is_err(),
                "cholesky_solve_gpu must Err when runtime is unavailable"
            );
            assert!(
                factor_outcome.is_err(),
                "cholesky_lower_gpu must Err when runtime is unavailable"
            );
        }

        // solver::gpu::pirls_gpu CPU-fallback entry points
        let xc = Array2::<f64>::from_shape_fn((4, 3), |(i, j)| (i + j) as f64 + 1.0);
        let weights = Array1::<f64>::ones(4);
        let xtwx = crate::solver::gpu::pirls_gpu::weighted_crossprod_gpu(xc.view(), weights.view());
        assert!(
            xtwx.is_ok(),
            "weighted_crossprod_gpu must return Ok via CPU fallback on CPU-only host (got {xtwx:?})"
        );
    }
}

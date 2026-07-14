#[cfg(target_os = "linux")]
use std::cell::Cell;
#[cfg(target_os = "linux")]
use std::collections::HashMap;
#[cfg(target_os = "linux")]
use std::panic::{self, AssertUnwindSafe, catch_unwind};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
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

/// A genuine reason CUDA cannot exist on this host. These states are distinct
/// from [`GpuError`]: absence is an expected hardware/platform fact under
/// [`GpuPolicy::Auto`](super::GpuPolicy::Auto), whereas an error means a CUDA
/// installation or device that was present failed to initialize correctly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuAbsence {
    UnsupportedPlatform,
    DriverUnavailable { reason: String },
    NoDevice { reason: String },
}

impl std::fmt::Display for GpuAbsence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedPlatform => {
                f.write_str("CUDA support is unavailable on this platform")
            }
            Self::DriverUnavailable { reason } | Self::NoDevice { reason } => {
                f.write_str(reason)
            }
        }
    }
}

/// Lossless result of the process-wide CUDA probe.
#[derive(Debug)]
pub enum GpuAvailability {
    Available(GpuRuntime),
    Absent(GpuAbsence),
}

/// Borrowed lossless availability view returned from the one-time cache.
#[derive(Clone, Copy, Debug)]
pub enum GpuAvailabilityRef<'a> {
    Available(&'a GpuRuntime),
    Absent(&'a GpuAbsence),
}

/// Process-wide count of lossless runtime-resolution calls.
///
/// Incremented on every [`GpuRuntime::availability`] call before the one-time probe
/// runs — so it counts the moments at which the device probe (and thus CUDA
/// primary-context creation on each GPU, `cuDevicePrimaryCtxRetain`) could be
/// triggered. Size-gated accessors that short-circuit for CPU-sized problems
/// deliberately do not resolve availability, so a test can pin this counter across
/// such a call and prove the CPU-sized decision path made ZERO driver contact.
///
/// Cross-platform (not `cfg(target_os = "linux")`) so the laziness/ordering
/// contract is testable on CUDA-less hosts: even where the probe itself is a
/// no-op, the invariant we verify is that the size check precedes resolution.
static RESOLUTION_CALLS: AtomicU64 = AtomicU64::new(0);

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
    pub fn probe() -> Result<GpuAvailability, GpuError> {
        #[cfg(not(target_os = "linux"))]
        {
            let reason = "CUDA support not compiled into this build";
            Self::record_cpu_reason(reason);
            diagnostics::log_cuda_disabled(reason);
            return Ok(GpuAvailability::Absent(GpuAbsence::UnsupportedPlatform));
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
            match crate::driver::preload_cuda_driver() {
                Ok(()) => {}
                Err(GpuError::DriverLibraryUnavailable { reason }) => {
                    Self::record_cpu_reason(reason.clone());
                    log::info!("[GPU] CUDA acceleration disabled: {reason}");
                    diagnostics::log_cuda_disabled(&reason);
                    return Ok(GpuAvailability::Absent(GpuAbsence::DriverUnavailable {
                        reason,
                    }));
                }
                Err(error) => return Err(error),
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
                if let Err(error) = crate::driver::require_cuda_compute_library(stem) {
                    let reason = format!("lib{stem} unavailable: {error}");
                    Self::record_cpu_reason(reason.clone());
                    log::info!("[GPU] CUDA acceleration disabled: {reason}");
                    diagnostics::log_cuda_disabled(&reason);
                    return Err(GpuError::RuntimeDependencyUnavailable { reason });
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
                .map_err(|_| GpuError::DriverCallFailed {
                    reason: "cudarc failed after the CUDA driver preflight succeeded".to_string(),
                })?
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: err.to_string(),
                })?;
            if device_count <= 0 {
                let reason = "CUDA driver reported no devices";
                Self::record_cpu_reason(reason);
                diagnostics::log_cuda_disabled(reason);
                return Ok(GpuAvailability::Absent(GpuAbsence::NoDevice {
                    reason: reason.to_string(),
                }));
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
                    .map_err(|_| GpuError::DriverCallFailed {
                        reason: "CUDA context binding panicked after driver discovery".to_string(),
                    })?
                    .map_err(|err| GpuError::DriverCallFailed {
                        reason: err.to_string(),
                    })?;
                devices.push(
                    catch_unwind(AssertUnwindSafe(|| cuda_device_info(ordinal, &ctx))).map_err(
                        |_| GpuError::DriverCallFailed {
                            reason: "CUDA device inspection panicked after driver discovery"
                                .to_string(),
                        },
                    )??,
                );
            }

            devices.sort_by(|a, b| b.score().total_cmp(&a.score()));
            let Some(device) = devices.first().cloned() else {
                Self::record_cpu_reason("CUDA driver reported no usable devices");
                diagnostics::log_cuda_disabled("CUDA driver reported no usable devices");
                return Ok(GpuAvailability::Absent(GpuAbsence::NoDevice {
                    reason: "CUDA driver reported no usable devices".to_string(),
                }));
            };

            let policy = crate::calibration::calibrated_policy_for_device(&device);
            let memory_budget_bytes = device.memory_budget_bytes();
            diagnostics::log_cuda_enabled(&device, &policy);
            diagnostics::log_cuda_pool(&devices);

            Ok(GpuAvailability::Available(Self {
                device,
                devices,
                policy,
                memory_budget_bytes,
            }))
        }
    }

    /// Return the cached probe outcome without collapsing faults into absence.
    pub fn availability() -> Result<GpuAvailabilityRef<'static>, GpuError> {
        // Record every entry BEFORE the `OnceLock` probe, so the size-gated
        // accessors below (which never reach this point for CPU-sized problems)
        // can be proven not to have triggered a device probe / context creation.
        RESOLUTION_CALLS.fetch_add(1, Ordering::Relaxed);
        static RUNTIME: OnceLock<Result<GpuAvailability, GpuError>> = OnceLock::new();
        let cached = RUNTIME.get_or_init(|| {
            let outcome = Self::probe();
            if let Err(error) = &outcome {
                let reason = error.to_string();
                Self::record_cpu_reason(reason.clone());
                diagnostics::log_cuda_disabled(&reason);
            }
            // Install the dense-GEMM dispatch hook exactly when a usable
            // device was probed. Without this, `gam_linalg::faer_ndarray::fast_ab`
            // (and the `fast_atb`/`fast_av`/`xt_diag_x` family) never sees a
            // dispatcher — `gpu_dispatch()` stays `None` — so every dense
            // product in the engine silently runs on the CPU even when the
            // V100 is present and the workload clears the policy flop floor.
            // The hook is a first-write-wins `OnceLock` keyed only on the
            // presence of a runtime; registering it here, inside the same
            // `get_or_init` that decides the runtime, guarantees it is
            // installed before any `fast_ab` caller can observe an available
            // runtime. The policy gate inside each `try_*` still decides
            // CPU-vs-GPU per call, so small products are unaffected.
            if matches!(outcome, Ok(GpuAvailability::Available(_))) {
                gam_linalg::gpu_hook::register_gpu_dispatch(Box::new(
                    super::linalg_dispatch::CudaGemmDispatch,
                ));
            }
            outcome
        });
        match cached {
            Ok(GpuAvailability::Available(runtime)) => {
                Ok(GpuAvailabilityRef::Available(runtime))
            }
            Ok(GpuAvailability::Absent(reason)) => Ok(GpuAvailabilityRef::Absent(reason)),
            Err(error) => Err(error.clone()),
        }
    }

    /// Resolve CUDA under an explicit policy. `Ok(None)` is reserved for a
    /// genuine absence under Auto/Off; probe faults always remain `Err`, and
    /// Required converts absence into `RequiredDeviceUnavailable`.
    pub fn resolve(policy: super::GpuPolicy) -> Result<Option<&'static Self>, GpuError> {
        if policy == super::GpuPolicy::Off {
            return Ok(None);
        }
        Self::resolve_availability(policy, Self::availability())
    }

    fn resolve_availability<'a>(
        policy: super::GpuPolicy,
        availability: Result<GpuAvailabilityRef<'a>, GpuError>,
    ) -> Result<Option<&'a Self>, GpuError> {
        match availability? {
            GpuAvailabilityRef::Available(runtime) => Ok(Some(runtime)),
            GpuAvailabilityRef::Absent(_reason) if policy == super::GpuPolicy::Auto => Ok(None),
            GpuAvailabilityRef::Absent(reason) => Err(GpuError::RequiredDeviceUnavailable {
                reason: reason.to_string(),
            }),
        }
    }

    /// Resolve CUDA under Required semantics and return the device handle.
    pub fn require() -> Result<&'static Self, GpuError> {
        Self::resolve(super::GpuPolicy::Required)?.ok_or_else(|| {
            GpuError::RequiredDeviceUnavailable {
                reason: "required CUDA runtime resolved to an absent state".to_string(),
            }
        })
    }

    /// Number of times [`Self::availability`] has been entered process-wide.
    ///
    /// Test-facing instrumentation for the laziness contract: a size-gated
    /// caller that returns before resolving availability leaves this unchanged, so
    /// a test can assert a CPU-sized decision path created no CUDA context. This
    /// is a monotone call counter, NOT a probe-success flag.
    #[must_use]
    pub fn resolution_call_count() -> u64 {
        RESOLUTION_CALLS.load(Ordering::Relaxed)
    }

    /// Size-gated [`Self::resolve`]: resolve the process-wide runtime only when the
    /// estimated dense arithmetic `work_flops` clears the GPU-dispatch flop floor.
    ///
    /// This is the ordering fix for the CUDA startup tax. For a CPU-sized problem
    /// (`work_flops` below the floor) it returns `Ok(None)` without calling
    /// [`Self::resolve`], so the device probe — and the `cuDevicePrimaryCtxRetain`
    /// primary-context creation it performs on every GPU — never runs. The
    /// problem-size decision therefore strictly precedes any driver contact, and
    /// a CPU-sized fit pays ZERO CUDA cost.
    ///
    /// The floor is [`GpuDispatchPolicy::MIN_CALIBRATABLE_GEMM_FLOPS`] — the
    /// smallest `gemm_min_flops` ANY reachable policy (default seed or
    /// device-calibrated) can carry, known WITHOUT a device — so the gate never
    /// needs a probe to decide it should not probe, and refusing below it can
    /// never block work that any policy would have dispatched. Work at or above
    /// the floor falls through to the identical lossless resolution path (where
    /// the real, possibly calibrated policy still gates each op), so device
    /// behaviour for genuinely GPU-sized problems is unchanged.
    pub fn resolve_if_dense_work_exceeds_floor(
        policy: super::GpuPolicy,
        work_flops: u128,
    ) -> Result<Option<&'static Self>, GpuError> {
        if work_flops < GpuDispatchPolicy::MIN_CALIBRATABLE_GEMM_FLOPS {
            return Ok(None);
        }
        Self::resolve(policy)
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
    // Materialise the runtime primary context for this device: cuBLAS/cuSOLVER
    // `*Create` use whatever context is current at creation time, so the runtime
    // device must be selected and its primary context materialised before a
    // handle is made. A 256-byte allocate-then-free is the canonical,
    // ~microsecond way to force it. This is invoked exactly once per (thread,
    // ordinal) by `bind_and_touch_runtime` — the NOT_INITIALIZED condition it
    // repairs is per-thread-per-device and does NOT re-arm per call once the
    // primary context is current and the runtime is materialised on the thread.
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
thread_local! {
    /// The device ordinal whose primary context is bound as THIS thread's
    /// current context AND whose runtime primary context has already been
    /// materialised on this thread. `Some(ordinal)` means the last
    /// [`cuda_context_for`] touch on this thread was `ordinal` and nothing has
    /// switched it since, so the per-call `bind_to_thread` + runtime
    /// materialisation can be skipped.
    ///
    /// Switching to a different ordinal (or the initial `None`) invalidates the
    /// memo and forces a full rebind + re-materialisation, so the per-thread-
    /// per-device NOT_INITIALIZED repair (#1017) is preserved exactly: the
    /// condition it fixes is arm-once-per-(thread, device), and a memo keyed on
    /// the thread's currently-bound ordinal only skips work when that same
    /// ordinal is already current — i.e. when neither the driver context nor the
    /// runtime device could have drifted.
    static BOUND_RUNTIME_ORDINAL: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Bind cudarc's primary context for `ordinal` current on this thread and
/// materialise the runtime primary context on it — memoised once per (thread,
/// ordinal).
///
/// The bind + runtime touch exist to repair the probe-first
/// CUBLAS/CUSOLVER_STATUS_NOT_INITIALIZED bug: on a fresh solve thread the
/// cached-context path would let the CUDA runtime initialise its OWN device
/// context, so a later `cublasCreate`/`cusolverDnCreate` on the primary-context
/// stream fails. Binding the primary context current and forcing runtime
/// materialisation on the SAME context before returning fixes it. That repair
/// is durable per (thread, ordinal); it does not re-arm per call. So when this
/// thread's current context is already `ordinal` we skip the bind and the
/// 256-byte cudaMalloc/cudaFree entirely, removing the per-call driver tax while
/// preserving the invariant — a switch to any other ordinal re-runs the full
/// repair.
#[cfg(target_os = "linux")]
fn bind_and_touch_runtime(ordinal: usize, ctx: &Arc<CudaContext>) {
    if BOUND_RUNTIME_ORDINAL.with(Cell::get) == Some(ordinal) {
        return;
    }
    let bound = catch_unwind(AssertUnwindSafe(|| ctx.bind_to_thread()));
    log::trace!(
        "[GPU] cuda_context_for bind ok={} ordinal={ordinal}",
        matches!(bound, Ok(Ok(())))
    );
    ensure_cuda_runtime_device(ordinal);
    // Latch the memo only after a SUCCESSFUL bind: a failed bind left the
    // thread's current context indeterminate, so the next call must retry the
    // full repair rather than assume `ordinal` is current.
    if matches!(bound, Ok(Ok(()))) {
        BOUND_RUNTIME_ORDINAL.with(|c| c.set(Some(ordinal)));
    }
}

#[cfg(target_os = "linux")]
pub fn cuda_context_for(ordinal: usize) -> Option<Arc<CudaContext>> {
    static CONTEXTS: OnceLock<Mutex<HashMap<usize, Arc<CudaContext>>>> = OnceLock::new();
    let contexts = CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(ctx) = contexts.lock().ok()?.get(&ordinal).cloned() {
        bind_and_touch_runtime(ordinal, &ctx);
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
    // an entry created on another thread; the memoised bind rebinds so the primary
    // context is current on THIS thread before the runtime touch (same probe-first
    // NOT_INITIALIZED guard) on the first touch, and is a no-op thereafter.
    bind_and_touch_runtime(ordinal, &out);
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
        _ = crate::device_runtime::GpuRuntime::resolution_call_count();
        let type_name = std::any::type_name::<crate::device_runtime::GpuRuntime>();
        assert!(
            type_name.contains("device_runtime"),
            "GpuRuntime must live in the `device_runtime` module (got {type_name})"
        );
    }
}

#[cfg(test)]
mod laziness_gate_tests {
    //! Pins the CUDA startup-tax ordering fix: a CPU-sized problem must reach
    //! its size decision WITHOUT ever resolving GPU availability (which is what
    //! triggers the one-time device probe + `cuDevicePrimaryCtxRetain`
    //! primary-context creation on every GPU). Runs on any host — on a CUDA-less
    //! box the probe is a no-op, but the invariant under test is purely the
    //! control-flow ordering (size check strictly before resolution), which is
    //! observable through the process-wide `resolution_call_count` counter.
    //!
    //! nextest runs each test in its own process, so the counter starts at a
    //! clean baseline per test; the assertions use a delta against `before` so
    //! they are robust regardless of the absolute starting value.
    use super::*;

    #[test]
    fn cpu_sized_dense_work_never_resolves_availability() {
        let before = GpuRuntime::resolution_call_count();
        // Dense work far below the GPU-dispatch flop floor: a CPU-sized fit.
        assert!(
            GpuRuntime::resolve_if_dense_work_exceeds_floor(super::super::GpuPolicy::Auto, 1_000)
                .expect("the pre-probe size gate itself is infallible")
                .is_none(),
            "CPU-sized work must not select the device"
        );
        assert_eq!(
            GpuRuntime::resolution_call_count(),
            before,
            "the size gate must short-circuit BEFORE resolution/probe for CPU-sized \
             work, so no CUDA context is ever created"
        );
    }

    #[test]
    fn gpu_sized_dense_work_falls_through_to_resolution() {
        let before = GpuRuntime::resolution_call_count();
        // Above any plausible floor: must consult the runtime exactly once, i.e.
        // the gate does not change behaviour for genuinely GPU-sized problems.
        // The returned handle is irrelevant here (None on CPU-only boxes);
        // the observable is the consultation count below.
        let runtime = GpuRuntime::resolve_if_dense_work_exceeds_floor(
            super::super::GpuPolicy::Auto,
            u128::MAX,
        )
        .expect("a probe fault must fail this gate instead of looking absent");
        assert!(
            runtime.is_none_or(|runtime| !runtime.devices.is_empty()),
            "an available runtime must expose at least one usable device"
        );
        assert_eq!(
            GpuRuntime::resolution_call_count(),
            before + 1,
            "GPU-sized work must fall through to availability resolution"
        );
    }

    #[test]
    fn floor_is_the_min_calibratable_gemm_threshold() {
        // The gate's floor is the smallest gemm_min_flops any reachable policy
        // (default seed OR device-calibrated) can carry — known without a
        // device, so the decision to NOT probe never needs a probe, and the
        // refusal can never block work some calibrated policy would dispatch.
        let floor = GpuDispatchPolicy::MIN_CALIBRATABLE_GEMM_FLOPS;
        let before = GpuRuntime::resolution_call_count();
        assert!(
            GpuRuntime::resolve_if_dense_work_exceeds_floor(
                super::super::GpuPolicy::Auto,
                floor - 1,
            )
            .expect("the below-floor gate cannot probe or fail")
            .is_none()
        );
        assert_eq!(GpuRuntime::resolution_call_count(), before);
        // At the floor the gate must consult the runtime (fall through).
        let runtime = GpuRuntime::resolve_if_dense_work_exceeds_floor(
            super::super::GpuPolicy::Auto,
            floor,
        )
        .expect("a probe fault must fail the boundary gate instead of looking absent");
        assert!(
            runtime.is_none_or(|runtime| !runtime.devices.is_empty()),
            "a successful floor-boundary probe must expose at least one usable device"
        );
        assert_eq!(GpuRuntime::resolution_call_count(), before + 1);
    }
}

#[cfg(test)]
mod policy_resolution_contract_tests {
    use super::*;
    use crate::GpuPolicy;

    #[test]
    fn auto_maps_only_typed_absence_to_none() {
        let absence = GpuAbsence::NoDevice {
            reason: "synthetic device-free absence".to_string(),
        };
        let resolved = GpuRuntime::resolve_availability(
            GpuPolicy::Auto,
            Ok(GpuAvailabilityRef::Absent(&absence)),
        )
        .expect("typed absence is expected under Auto");
        assert!(resolved.is_none());
    }

    #[test]
    fn required_turns_only_typed_absence_into_required_unavailable() {
        let absence = GpuAbsence::DriverUnavailable {
            reason: "synthetic missing driver".to_string(),
        };
        let error = GpuRuntime::resolve_availability(
            GpuPolicy::Required,
            Ok(GpuAvailabilityRef::Absent(&absence)),
        )
        .expect_err("Required must reject typed absence");
        assert!(matches!(
            error,
            GpuError::RequiredDeviceUnavailable { ref reason }
                if reason == "synthetic missing driver"
        ));
    }

    #[test]
    fn auto_and_required_preserve_probe_fault_variants() {
        for policy in [GpuPolicy::Auto, GpuPolicy::Required] {
            let error = GpuRuntime::resolve_availability(
                policy,
                Err(GpuError::RuntimeDependencyUnavailable {
                    reason: "synthetic missing cuBLAS".to_string(),
                }),
            )
            .expect_err("probe faults must never project to absence");
            assert!(matches!(
                error,
                GpuError::RuntimeDependencyUnavailable { ref reason }
                    if reason == "synthetic missing cuBLAS"
            ));
        }
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
    /// the typed availability result about the absence of a driver.
    #[test]
    fn cpu_only_host_never_panics_on_gpu_entry_points() {
        // Without libcuda the runtime must report unavailable rather than
        // panicking from inside `culib()`; with libcuda the runtime may or
        // may not have a usable device, but the panic-free contract still
        // holds and the dispatch smoke test below exercises it.
        match crate::driver::preload_cuda_driver() {
            Ok(()) => {}
            Err(GpuError::DriverLibraryUnavailable { .. }) => assert!(
                matches!(
                    GpuRuntime::availability(),
                    Ok(GpuAvailabilityRef::Absent(GpuAbsence::DriverUnavailable { .. }))
                ),
                "typed driver absence must remain absence through runtime availability"
            ),
            Err(error) => panic!("a present-but-broken CUDA driver must fail this test: {error}"),
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
        if matches!(
            GpuRuntime::availability(),
            Ok(GpuAvailabilityRef::Absent(_))
        ) {
            assert!(
                solve_outcome.is_err(),
                "cholesky_solve_gpu must Err when runtime is unavailable"
            );
            assert!(
                factor_outcome.is_err(),
                "cholesky_lower_gpu must Err when runtime is unavailable"
            );
        }

        // NOTE: the weighted-crossprod GPU dispatcher with CPU fallback
        // (`weighted_crossprod_gpu`) moved out of this crate to `gam-solve`
        // (`gpu::pirls_gpu`) during the #1521 crate carve, since it depends on
        // the higher-level PIRLS assembly. Its panic-free / Ok-via-CPU-fallback
        // contract is now exercised by a regression test there
        // (`weighted_crossprod_gpu_cpu_fallback_*`), not from gam-gpu, which
        // cannot reach gam-solve without a dependency cycle.
    }
}

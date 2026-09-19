#[cfg(target_os = "linux")]
use std::cell::Cell;
#[cfg(target_os = "linux")]
use std::collections::HashMap;
use std::sync::OnceLock;
#[cfg(target_os = "linux")]
use std::sync::{Arc, Mutex};

use super::device::GpuDeviceInfo;
#[cfg(target_os = "linux")]
use super::driver::{CudarcLibrary, require_cudarc_library};
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
            Self::DriverUnavailable { reason } | Self::NoDevice { reason } => f.write_str(reason),
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
            Self::probe_after_driver_preflight(
                require_cudarc_library(CudarcLibrary::Driver),
                Self::probe_loaded_driver,
            )
        }
    }

    /// Let cudarc run only after its libcuda loader is known to succeed.
    ///
    /// cudarc reports a missing library by panicking inside its loader, and a
    /// `panic = "abort"` consumer cannot recover that panic, so any fit that
    /// reached the probe on a host without libcuda killed the process (#2972).
    /// `driver` is the non-panicking walk over cudarc's own libcuda candidates. A
    /// missing library is typed absence here, and `probe_loaded_driver`, which
    /// makes the first cudarc call, never runs.
    #[cfg(target_os = "linux")]
    fn probe_after_driver_preflight(
        driver: Result<(), GpuError>,
        probe_loaded_driver: impl FnOnce() -> Result<GpuAvailability, GpuError>,
    ) -> Result<GpuAvailability, GpuError> {
        match driver {
            Ok(()) => probe_loaded_driver(),
            Err(GpuError::DriverLibraryUnavailable { reason }) => {
                Self::record_cpu_reason(reason.clone());
                log::info!("[GPU] CUDA acceleration disabled: {reason}");
                diagnostics::log_cuda_disabled(&reason);
                Ok(GpuAvailability::Absent(GpuAbsence::DriverUnavailable { reason }))
            }
            Err(error) => Err(error),
        }
    }

    #[cfg(target_os = "linux")]
    fn probe_loaded_driver() -> Result<GpuAvailability, GpuError> {
        // #1017 probe-first fix: establish cudarc's primary context P and
        // initialize the CUDA runtime ON IT as the VERY FIRST CUDA action -- before
        // gam's libloading libcuda preload, the compute-lib dlopens, and device_count.
        // The clean cuda_context_for-first path works; the probe-first path failed
        // because a pre-context CUDA touch left the runtime bound to a non-P context,
        // so later cuBLAS/cuSOLVER handle creation on the P-stream returned
        // NOT_INITIALIZED. Making cuda_context_for the first action replicates the
        // working clean path (CudaContext::new loads libcuda + retains the primary +
        // ensure runs the runtime init). The libcuda walk before this call opened the
        // library CudaContext::new opens, and cuda_context_for walks cudarc's
        // libcudart names where its first runtime call would load them.
        let primary_ready = cuda_context_for(0).is_some();
        log::trace!("[GPU] probe pre-init primary context + runtime: {primary_ready}");
        match crate::driver::preload_cuda_driver() {
            Ok(()) => {}
            Err(GpuError::DriverLibraryUnavailable { reason }) => {
                Self::record_cpu_reason(reason.clone());
                log::info!("[GPU] CUDA acceleration disabled: {reason}");
                diagnostics::log_cuda_disabled(&reason);
                return Ok(GpuAvailability::Absent(GpuAbsence::DriverUnavailable { reason }));
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

        // The stack preload above opens libraries by path. cudarc opens the ones it
        // drives by its own candidate names when a handle is first created, and
        // panics when none opens, so advertise the runtime only when those names
        // open too (#2972).
        for library in [CudarcLibrary::Runtime, CudarcLibrary::Blas, CudarcLibrary::Solver] {
            if let Err(error) = require_cudarc_library(library) {
                let reason = format!("cudarc cannot open lib{}: {error}", library.name());
                Self::record_cpu_reason(reason.clone());
                log::info!("[GPU] CUDA acceleration disabled: {reason}");
                diagnostics::log_cuda_disabled(&reason);
                return Err(GpuError::RuntimeDependencyUnavailable { reason });
            }
        }

        let device_count = match CudaContext::device_count() {
            Ok(count) => count,
            Err(error) => {
                // `device_count` performs `cuInit`, so this is the first
                // moment the host's kernel driver actually answers. A
                // refusal that is an ENVIRONMENT fact (userland CUDA
                // libraries with no matching kernel driver — the container
                // / CPU-node case #2267 hit as
                // `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`) is typed absence:
                // Auto falls back to CPU, Required still refuses with the
                // same diagnosis. Anything else stays a probe fault.
                if let Some(absence) = absence_from_driver_init_error(&error) {
                    let reason = absence.to_string();
                    Self::record_cpu_reason(reason.clone());
                    log::info!("[GPU] CUDA acceleration disabled: {reason}");
                    diagnostics::log_cuda_disabled(&reason);
                    return Ok(GpuAvailability::Absent(absence));
                }
                return Err(GpuError::DriverCallFailed {
                    reason: error.to_string(),
                });
            }
        };
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
            ctx.bind_to_thread()
                .map_err(|err| GpuError::DriverCallFailed {
                    reason: err.to_string(),
                })?;
            devices.push(cuda_device_info(ordinal, &ctx)?);
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

    /// Return the cached probe outcome without collapsing faults into absence.
    pub fn availability() -> Result<GpuAvailabilityRef<'static>, GpuError> {
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
            if matches!(&outcome, Ok(GpuAvailability::Available(_))) {
                gam_linalg::gpu_hook::register_gpu_dispatch(Box::new(
                    super::linalg_dispatch::CudaGemmDispatch,
                ));
            }
            outcome
        });
        match cached {
            Ok(GpuAvailability::Available(runtime)) => Ok(GpuAvailabilityRef::Available(runtime)),
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

    /// Size-gated [`Self::resolve`] for independent fused row kernels.
    ///
    /// Batches below
    /// [`GpuDispatchPolicy::MIN_CALIBRATABLE_FUSED_KERNEL_N`] cannot be
    /// admitted by either the default or any device-calibrated policy. Refuse
    /// them before availability resolution so a CPU-sized first call does not
    /// create CUDA contexts and run calibration merely to learn that it should
    /// stay on the CPU. At and above the universal floor, the concrete
    /// runtime's calibrated policy remains authoritative.
    pub fn resolve_if_fused_batch_exceeds_floor(
        policy: super::GpuPolicy,
        rows: usize,
    ) -> Result<Option<&'static Self>, GpuError> {
        if rows < GpuDispatchPolicy::MIN_CALIBRATABLE_FUSED_KERNEL_N {
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
        // First reason wins: the earliest fallback is the one that explains the
        // rest. A later reason is dropped deliberately, and visibly.
        if let Err(dropped) = CPU_REASON.set(reason.into()) {
            log::debug!(
                "CPU fallback reason already recorded as {:?}; keeping it and dropping '{dropped}'",
                CPU_REASON.get().map(String::as_str)
            );
        }
    }
}

/// Classify a CUDA driver-*initialization* failure that is a fact about the
/// host environment rather than a fault of a device that was present.
///
/// `cuInit` is the first call the kernel driver answers. The codes below all
/// mean "CUDA cannot work on this host as configured" — a loaded `libcuda`
/// userland with a missing, older, or mismatched kernel driver, a linker stub
/// standing in for the real library, or no attached device. Those states are
/// [`GpuAbsence`] by this module's own definition (absence is an expected
/// hardware/platform fact under `GpuPolicy::Auto`): container images and CPU
/// nodes routinely carry CUDA userland libraries they cannot back with a
/// driver, and a fit under Auto must fall back to CPU there instead of dying
/// inside runtime resolution (#2267). Every other code — illegal address,
/// out-of-memory, ECC faults, ... — still means "a CUDA installation that was
/// present failed", and stays a probe fault.
#[cfg(target_os = "linux")]
fn absence_from_driver_init_error(error: &result::DriverError) -> Option<GpuAbsence> {
    use sys::cudaError_enum as CudaErrorCode;
    // Format the raw enum code, NEVER the DriverError itself: cudarc's
    // Display/Debug for DriverError resolve the error string through its
    // dynamic loader (`culib()`), which panics via `panic_no_lib_found` on
    // exactly the driverless hosts this classifier exists for. The enum's
    // derived Debug is a pure Rust name and is safe everywhere.
    let code = error.0;
    let classification = match code {
        CudaErrorCode::CUDA_ERROR_NO_DEVICE => {
            return Some(GpuAbsence::NoDevice {
                reason: format!(
                    "CUDA driver initialized but reports no attached device ({code:?})"
                ),
            });
        }
        CudaErrorCode::CUDA_ERROR_STUB_LIBRARY => {
            "the loaded libcuda is a linker stub, not a real driver"
        }
        // NOTE: there is deliberately no INSUFFICIENT_DRIVER arm — that code
        // (`cudaErrorInsufficientDriver`) exists only in the CUDA *runtime*
        // API; the driver API reports the userland/kernel version split as
        // `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` below.
        CudaErrorCode::CUDA_ERROR_SYSTEM_NOT_READY => {
            "the CUDA system is not ready (kernel driver or fabric daemon not running)"
        }
        CudaErrorCode::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH => {
            "the CUDA userland libraries do not match the host kernel driver"
        }
        CudaErrorCode::CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE => {
            "CUDA forward-compatibility mode is not supported on the visible device"
        }
        _ => return None,
    };
    Some(GpuAbsence::DriverUnavailable {
        reason: format!("CUDA initialization refused: {classification} ({code:?})"),
    })
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
///
/// Returns `false` when cudarc cannot open libcudart: the runtime touch would
/// reach cudarc's panicking loader, and a context without the repair is not a
/// usable context.
#[cfg(target_os = "linux")]
fn bind_and_touch_runtime(ordinal: usize, ctx: &Arc<CudaContext>) -> bool {
    if BOUND_RUNTIME_ORDINAL.with(Cell::get) == Some(ordinal) {
        return true;
    }
    // `cudaSetDevice` in the runtime touch is cudarc's first libcudart call (#2972).
    if let Err(error) = require_cudarc_library(CudarcLibrary::Runtime) {
        log::debug!("[GPU] cuda_context_for ordinal={ordinal}: {error}");
        return false;
    }
    let bound = ctx.bind_to_thread().is_ok();
    log::trace!("[GPU] cuda_context_for bind ok={bound} ordinal={ordinal}");
    ensure_cuda_runtime_device(ordinal);
    // Latch the memo only after a SUCCESSFUL bind: a failed bind left the
    // thread's current context indeterminate, so the next call must retry the
    // full repair rather than assume `ordinal` is current.
    if bound {
        BOUND_RUNTIME_ORDINAL.with(|c| c.set(Some(ordinal)));
    }
    true
}

#[cfg(target_os = "linux")]
pub fn cuda_context_for(ordinal: usize) -> Option<Arc<CudaContext>> {
    static CONTEXTS: OnceLock<Mutex<HashMap<usize, Arc<CudaContext>>>> = OnceLock::new();
    let contexts = CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(ctx) = contexts.lock().ok()?.get(&ordinal).cloned() {
        return bind_and_touch_runtime(ordinal, &ctx).then_some(ctx);
    }
    // `CudaContext::new` is cudarc's first libcuda call, and cudarc's loader
    // panics when libcuda is missing. Settle that without cudarc (#2972).
    require_cudarc_library(CudarcLibrary::Driver).ok()?;
    let ctx = CudaContext::new(ordinal).ok()?;
    let out = {
        let mut guard = contexts.lock().ok()?;
        guard.entry(ordinal).or_insert_with(|| ctx.clone()).clone()
    };
    // CudaContext::new already bound the primary context, but the HashMap may return
    // an entry created on another thread; the memoised bind rebinds so the primary
    // context is current on THIS thread before the runtime touch (same probe-first
    // NOT_INITIALIZED guard) on the first touch, and is a no-op thereafter.
    bind_and_touch_runtime(ordinal, &out).then_some(out)
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
        name: result::device::get_name(device).unwrap_or_else(|err| {
            log::debug!(
                "CUDA device {ordinal}: name query failed ({err}); using a positional label"
            );
            format!("CUDA device {ordinal}")
        }),
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
mod policy_resolution_contract_tests {
    use super::*;
    use crate::GpuPolicy;

    /// #2972: a missing libcuda is typed absence decided before cudarc runs.
    /// cudarc reports a missing library only by panicking, which a
    /// `panic = "abort"` consumer cannot recover, so the probe must not reach
    /// cudarc at all. The driver verdicts come from the real candidate walk.
    #[cfg(target_os = "linux")]
    #[test]
    fn missing_driver_library_is_absence_before_any_cudarc_call_2972() {
        let temp = tempfile::tempdir().expect("temporary driver directory");
        let absent = temp.path().join("libcuda.so.1");
        let unloadable = temp.path().join("libcuda.so.2972");
        std::fs::write(&unloadable, b"not an ELF object").expect("write unloadable driver");
        let walk = |candidate: &std::path::Path| {
            crate::driver::load_library_names(&[candidate.display().to_string()]).map(|_| ())
        };
        let mut cudarc_calls = 0;

        let availability = GpuRuntime::probe_after_driver_preflight(walk(&absent), || {
            cudarc_calls += 1;
            Ok(GpuAvailability::Absent(GpuAbsence::UnsupportedPlatform))
        })
        .expect("a missing driver library is absence, not a probe fault");
        assert!(
            matches!(
                availability,
                GpuAvailability::Absent(GpuAbsence::DriverUnavailable { .. })
            ),
            "a missing libcuda must be typed DriverUnavailable: {availability:?}"
        );
        assert_eq!(cudarc_calls, 0, "the probe reached cudarc without libcuda");

        let error = GpuRuntime::probe_after_driver_preflight(walk(&unloadable), || {
            cudarc_calls += 1;
            Ok(GpuAvailability::Absent(GpuAbsence::UnsupportedPlatform))
        })
        .expect_err("a present but unloadable driver is a probe fault");
        assert!(
            matches!(error, GpuError::DriverLibraryLoadFailed { .. }),
            "an unloadable libcuda must stay a load fault: {error}"
        );
        assert_eq!(cudarc_calls, 0, "the probe reached cudarc with an unloadable libcuda");

        // Positive control: a driver that opens hands the probe to cudarc.
        let availability = GpuRuntime::probe_after_driver_preflight(Ok(()), || {
            cudarc_calls += 1;
            Ok(GpuAvailability::Absent(GpuAbsence::UnsupportedPlatform))
        })
        .expect("the loaded-driver probe's outcome is returned as is");
        assert!(matches!(
            availability,
            GpuAvailability::Absent(GpuAbsence::UnsupportedPlatform)
        ));
        assert_eq!(cudarc_calls, 1, "a loadable driver must reach the cudarc probe");
    }

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
    fn exhausted_memory_does_not_erase_a_present_runtime_932() {
        let device = GpuDeviceInfo {
            ordinal: 0,
            name: "memory-exhausted fixture".to_string(),
            capability: crate::device::GpuCapability::from_compute_capability(8, 0),
            sm_count: 1,
            max_threads_per_sm: 2048,
            max_shared_mem_per_block: 48 * 1024,
            l2_cache_bytes: 1024 * 1024,
            total_mem_bytes: 1024 * 1024 * 1024,
            free_mem_bytes: 0,
            ecc_enabled: false,
            integrated: false,
            mig_mode: false,
        };
        let runtime = GpuRuntime {
            memory_budget_bytes: device.memory_budget_bytes(),
            devices: vec![device.clone()],
            device,
            policy: GpuDispatchPolicy::default(),
        };
        assert_eq!(runtime.memory_budget_bytes, 0);
        for policy in [GpuPolicy::Auto, GpuPolicy::Required] {
            let resolved = GpuRuntime::resolve_availability(
                policy,
                Ok(GpuAvailabilityRef::Available(&runtime)),
            )
            .expect("memory pressure is not missing hardware")
            .expect("the present runtime must survive resolution");
            assert!(std::ptr::eq(resolved, &runtime));

            let error = GpuRuntime::resolve_availability(
                policy,
                Err(GpuError::DriverCallFailed {
                    reason: "CUDA_ERROR_OUT_OF_MEMORY".to_string(),
                }),
            )
            .expect_err("allocation faults must retain their diagnosis");
            assert!(matches!(
                error,
                GpuError::DriverCallFailed { ref reason }
                    if reason == "CUDA_ERROR_OUT_OF_MEMORY"
            ));
        }
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

    /// #2267: a CUDA userland whose kernel driver is missing or mismatched is
    /// an environment fact. `cuInit`-boundary refusals of that class must be
    /// typed absence — Auto proceeds on CPU, Required refuses with the same
    /// diagnosis — never a probe fault that kills the fit under Auto.
    #[cfg(target_os = "linux")]
    #[test]
    fn driver_mismatch_at_init_is_typed_absence_not_a_fault() {
        for code in [
            sys::cudaError_enum::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH,
            sys::cudaError_enum::CUDA_ERROR_STUB_LIBRARY,
            sys::cudaError_enum::CUDA_ERROR_SYSTEM_NOT_READY,
            sys::cudaError_enum::CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE,
        ] {
            let absence = absence_from_driver_init_error(&result::DriverError(code))
                .unwrap_or_else(|| panic!("{code:?} is an environment fact, not a device fault"));
            assert!(
                matches!(absence, GpuAbsence::DriverUnavailable { .. }),
                "{code:?} must classify as an unavailable driver"
            );
            let resolved = GpuRuntime::resolve_availability(
                GpuPolicy::Auto,
                Ok(GpuAvailabilityRef::Absent(&absence)),
            )
            .expect("Auto must accept driver-environment absence");
            assert!(resolved.is_none(), "Auto must fall back to CPU on {code:?}");
            let required_error = GpuRuntime::resolve_availability(
                GpuPolicy::Required,
                Ok(GpuAvailabilityRef::Absent(&absence)),
            )
            .expect_err("Required must refuse driver-environment absence");
            assert!(
                matches!(required_error, GpuError::RequiredDeviceUnavailable { .. }),
                "Required must carry the environment diagnosis for {code:?}"
            );
        }
        let no_device = absence_from_driver_init_error(&result::DriverError(
            sys::cudaError_enum::CUDA_ERROR_NO_DEVICE,
        ))
        .expect("no attached device is an environment fact");
        assert!(matches!(no_device, GpuAbsence::NoDevice { .. }));
    }

    /// Faults of a present CUDA installation must never be reclassified into
    /// absence — the Auto policy is allowed to hide missing hardware, never a
    /// broken device.
    #[cfg(target_os = "linux")]
    #[test]
    fn present_device_faults_never_classify_as_absence() {
        for code in [
            sys::cudaError_enum::CUDA_ERROR_ILLEGAL_ADDRESS,
            sys::cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY,
            sys::cudaError_enum::CUDA_ERROR_NOT_INITIALIZED,
            sys::cudaError_enum::CUDA_ERROR_ECC_UNCORRECTABLE,
            sys::cudaError_enum::CUDA_ERROR_UNKNOWN,
        ] {
            assert!(
                absence_from_driver_init_error(&result::DriverError(code)).is_none(),
                "{code:?} is a fault of present hardware and must stay a probe fault"
            );
        }
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

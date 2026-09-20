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
    /// The driver is present, but a CUDA userspace library the device path
    /// loads (cuBLAS, cuSOLVER, cuSPARSE, or one cudarc opens by name) has no
    /// candidate on this host: the CUDA runtime is not installed here.
    RuntimeLibraryUnavailable { reason: String },
}

impl std::fmt::Display for GpuAbsence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedPlatform => {
                f.write_str("CUDA support is unavailable on this platform")
            }
            Self::DriverUnavailable { reason }
            | Self::NoDevice { reason }
            | Self::RuntimeLibraryUnavailable { reason } => f.write_str(reason),
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
                log::debug!("[GPU] CUDA acceleration disabled: {reason}");
                diagnostics::log_cuda_disabled(&reason);
                Ok(GpuAvailability::Absent(GpuAbsence::DriverUnavailable { reason }))
            }
            Err(error) => Err(error),
        }
    }

    /// Admit one CUDA userspace library the device path loads, by the loader's
    /// own verdict (`driver::load_library_names`), the rule the libcuda
    /// preflight already follows. A library with no candidate on this host is
    /// typed absence: the CUDA runtime is not installed here, which is where a
    /// CPU-only install lands on a machine that carries only the NVIDIA driver,
    /// so `auto` selects the CPU and `required` refuses naming the library. A
    /// candidate that exists but does not load is a fault of a present
    /// installation under every policy (#3000). `describe` words the refusal.
    #[cfg(target_os = "linux")]
    fn runtime_library_admission(
        loaded: Result<(), GpuError>,
        describe: impl FnOnce(&GpuError) -> String,
    ) -> Result<Option<GpuAbsence>, GpuError> {
        let Err(error) = loaded else {
            return Ok(None);
        };
        let reason = describe(&error);
        Self::record_cpu_reason(reason.clone());
        log::debug!("[GPU] CUDA acceleration disabled: {reason}");
        diagnostics::log_cuda_disabled(&reason);
        match error {
            GpuError::DriverLibraryUnavailable { .. } => {
                Ok(Some(GpuAbsence::RuntimeLibraryUnavailable { reason }))
            }
            _ => Err(GpuError::RuntimeDependencyUnavailable { reason }),
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
                log::debug!("[GPU] CUDA acceleration disabled: {reason}");
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
        // three load cleanly here. How a refusal counts is the loader's own
        // verdict (`runtime_library_admission`).
        for stem in ["cublas", "cusolver", "cusparse"] {
            let loaded = crate::driver::require_cuda_compute_library(stem);
            if let Some(absence) = Self::runtime_library_admission(loaded, |error| {
                format!("lib{stem} unavailable: {error}")
            })? {
                return Ok(GpuAvailability::Absent(absence));
            }
        }

        // The stack preload above opens libraries by path. cudarc opens the ones it
        // drives by its own candidate names when a handle is first created, and
        // panics when none opens, so advertise the runtime only when those names
        // open too (#2972).
        for library in [CudarcLibrary::Runtime, CudarcLibrary::Blas, CudarcLibrary::Solver] {
            let loaded = require_cudarc_library(library);
            if let Some(absence) = Self::runtime_library_admission(loaded, |error| {
                format!("cudarc cannot open lib{}: {error}", library.name())
            })? {
                return Ok(GpuAvailability::Absent(absence));
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
                    log::debug!("[GPU] CUDA acceleration disabled: {reason}");
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
            log::trace!(
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
        log::trace!("[GPU] cuda_context_for ordinal={ordinal}: {error}");
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
    query_cuda_device_info(
        ordinal,
        |attribute| {
            // SAFETY: device comes from cudarc's validated device::get.
            unsafe { result::device::get_attribute(device, attribute) }
        },
        || result::device::get_name(device),
        || ctx.mem_get_info(),
    )
}

#[cfg(target_os = "linux")]
fn query_cuda_device_info<E: std::fmt::Display>(
    ordinal: usize,
    mut attribute: impl FnMut(sys::CUdevice_attribute_enum) -> Result<i32, E>,
    name: impl FnOnce() -> Result<String, E>,
    memory: impl FnOnce() -> Result<(usize, usize), E>,
) -> Result<GpuDeviceInfo, GpuError> {
    let mut attr = |kind| {
        attribute(kind).map_err(|err| GpuError::DriverCallFailed {
            reason: format!("CUDA device {ordinal}: attribute {kind:?} failed: {err}"),
        })
    };
    let name = name().map_err(|err| GpuError::DriverCallFailed {
        reason: format!("CUDA device {ordinal}: name query failed: {err}"),
    })?;
    let (free_mem_bytes, total_mem_bytes) = memory().map_err(|err| GpuError::DriverCallFailed {
        reason: format!("CUDA device {ordinal}: memory query failed: {err}"),
    })?;
    let major = attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let minor = attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    Ok(GpuDeviceInfo {
        ordinal,
        name,
        capability: super::device::GpuCapability::from_compute_capability(major, minor),
        sm_count: attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        max_threads_per_sm: attr(
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        )?,
        max_shared_mem_per_block: attr(
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        )? as usize,
        l2_cache_bytes: attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?
            as usize,
        total_mem_bytes,
        free_mem_bytes,
        ecc_enabled: attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_ECC_ENABLED)? != 0,
        integrated: attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_INTEGRATED)? != 0,
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

    /// #3000: a CUDA runtime library is admitted by the loader's verdict, the
    /// rule the libcuda preflight follows. No candidate on the host is
    /// absence, so `auto` resolves to the CPU and `required` refuses naming
    /// the library; a candidate that exists but does not load is a fault under
    /// every policy. The verdicts come from the real candidate walk.
    #[cfg(target_os = "linux")]
    #[test]
    fn missing_runtime_library_is_absence_and_an_unloadable_one_is_a_fault_3000() {
        let temp = tempfile::tempdir().expect("temporary runtime directory");
        let absent = temp.path().join("libcublas.so.12");
        let unloadable = temp.path().join("libcublas.so.3000");
        std::fs::write(&unloadable, b"not an ELF object").expect("write unloadable library");
        let walk = |candidate: &std::path::Path| {
            crate::driver::load_library_names(&[candidate.display().to_string()]).map(|_| ())
        };
        let describe = |error: &GpuError| format!("libcublas unavailable: {error}");

        let absence = GpuRuntime::runtime_library_admission(walk(&absent), describe)
            .expect("a runtime library with no candidate is absence, not a probe fault")
            .expect("a library that did not open must not admit the runtime");
        assert!(
            matches!(absence, GpuAbsence::RuntimeLibraryUnavailable { .. }),
            "a missing libcublas must be typed RuntimeLibraryUnavailable: {absence:?}"
        );
        let auto = GpuRuntime::resolve_availability(
            GpuPolicy::Auto,
            Ok(GpuAvailabilityRef::Absent(&absence)),
        )
        .expect("auto resolves a host without the CUDA runtime to the CPU");
        assert!(auto.is_none(), "auto must select the CPU without the CUDA runtime");
        let required = GpuRuntime::resolve_availability(
            GpuPolicy::Required,
            Ok(GpuAvailabilityRef::Absent(&absence)),
        )
        .expect_err("required refuses a host without the CUDA runtime");
        let searched = absent.display().to_string();
        assert!(
            matches!(
                &required,
                GpuError::RequiredDeviceUnavailable { reason }
                    if reason.contains("libcublas unavailable") && reason.contains(&searched)
            ),
            "required must name the missing library and where it was looked for: {required}"
        );

        let fault = GpuRuntime::runtime_library_admission(walk(&unloadable), describe)
            .expect_err("a present but unloadable runtime library is a probe fault");
        let present = unloadable.display().to_string();
        assert!(
            matches!(
                &fault,
                GpuError::RuntimeDependencyUnavailable { reason } if reason.contains(&present)
            ),
            "an unloadable libcublas must stay a fault naming the candidate: {fault}"
        );

        // Positive control: a library that opens admits the runtime.
        let admitted = GpuRuntime::runtime_library_admission(Ok(()), describe)
            .expect("an opened library is no refusal");
        assert!(admitted.is_none(), "an opened library must not report absence");
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

#[cfg(all(test, target_os = "linux"))]
mod device_query_failure_tests {
    use super::*;
    use sys::CUdevice_attribute_enum as Attribute;

    const ATTRIBUTES: [(Attribute, i32); 8] = [
        (Attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 8),
        (Attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0),
        (Attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 108),
        (
            Attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
            2048,
        ),
        (
            Attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            49152,
        ),
        (Attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, 41943040),
        (Attribute::CU_DEVICE_ATTRIBUTE_ECC_ENABLED, 1),
        (Attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED, 0),
    ];

    fn value(kind: Attribute) -> i32 {
        ATTRIBUTES
            .iter()
            .find(|(candidate, _)| *candidate == kind)
            .unwrap()
            .1
    }

    #[test]
    fn every_required_attribute_failure_preserves_device_query_and_cause() {
        for (failed, _) in ATTRIBUTES {
            let error = query_cuda_device_info(
                7,
                |kind| {
                    if kind == failed {
                        Err("injected driver cause")
                    } else {
                        Ok(value(kind))
                    }
                },
                || Ok("measured device".to_owned()),
                || Ok((100, 200)),
            )
            .unwrap_err();
            assert!(matches!(error, GpuError::DriverCallFailed { .. }));
            let reason = error.to_string();
            assert!(reason.contains("CUDA device 7"), "{reason}");
            assert!(reason.contains(&format!("{failed:?}")), "{reason}");
            assert!(reason.contains("injected driver cause"), "{reason}");
        }
    }

    #[test]
    fn name_and_memory_query_failures_are_not_substituted() {
        let name = query_cuda_device_info(
            3,
            |kind| Ok(value(kind)),
            || Err("name driver cause"),
            || Ok((100, 200)),
        )
        .unwrap_err()
        .to_string();
        assert!(
            name.contains("CUDA device 3: name query failed: name driver cause"),
            "{name}"
        );
        let memory = query_cuda_device_info(
            4,
            |kind| Ok(value(kind)),
            || Ok("name".to_owned()),
            || Err("memory driver cause"),
        )
        .unwrap_err()
        .to_string();
        assert!(
            memory.contains("CUDA device 4: memory query failed: memory driver cause"),
            "{memory}"
        );
    }

    #[test]
    fn successful_queries_keep_measured_values_including_zero_flags() {
        let mut visited = Vec::new();
        let device = query_cuda_device_info::<&str>(
            5,
            |kind| {
                visited.push(kind);
                Ok(value(kind))
            },
            || Ok("actual measured name".to_owned()),
            || Ok((123, 456)),
        )
        .unwrap();
        assert_eq!(visited, ATTRIBUTES.map(|(kind, _)| kind));
        assert_eq!(device.ordinal, 5);
        assert_eq!(device.name, "actual measured name");
        assert_eq!(
            (
                device.capability.compute_major,
                device.capability.compute_minor
            ),
            (8, 0)
        );
        assert_eq!(device.sm_count, 108);
        assert_eq!(device.max_threads_per_sm, 2048);
        assert_eq!(device.max_shared_mem_per_block, 49152);
        assert_eq!(device.l2_cache_bytes, 41943040);
        assert_eq!((device.free_mem_bytes, device.total_mem_bytes), (123, 456));
        assert!(device.ecc_enabled);
        assert!(!device.integrated);
    }
}

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::device::GpuDeviceInfo;
use super::policy::GpuDispatchPolicy;
use cudarc::driver::{CudaContext, result, sys};

#[path = "diagnostics.rs"]
pub(crate) mod diagnostics;

#[derive(Clone, Debug, Eq, PartialEq)]
#[must_use]
pub enum GpuProbeError {
    Driver(String),
}

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

impl GpuRuntime {
    pub fn probe() -> Result<Option<Self>, GpuProbeError> {
        if super::global_policy() == super::GpuPolicy::Off {
            Self::record_cpu_reason("GPU policy is off");
            diagnostics::log_cuda_disabled("GPU policy is off");
            return Ok(None);
        }

        // `cudarc 0.19`'s entry points lazily initialize the CUDA driver via
        // `libloading::Library::new`. When the platform has no CUDA driver at
        // all (e.g. macOS hosts where there is no `libcuda.dylib`), the
        // library-load helper panics from inside the cudarc internals instead
        // of returning an error. Catching the unwind here turns that panic
        // into the same graceful "CUDA driver missing" Err path that an
        // initialization-time `CUDA_ERROR_NO_DEVICE` would take — so callers
        // see `GpuRuntime::global() == None` and the CPU fast path resumes
        // instead of the whole process aborting from a Python boundary call.
        let device_count = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            CudaContext::device_count().map_err(|err| GpuProbeError::Driver(err.to_string()))
        }))
        .map_err(|payload| {
            let reason = if let Some(s) = payload.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "CUDA driver library not loadable on this host".to_string()
            };
            GpuProbeError::Driver(reason)
        })??;
        if device_count <= 0 {
            let reason = "CUDA driver reported no devices";
            Self::record_cpu_reason(reason);
            diagnostics::log_cuda_disabled(reason);
            // Surface the no-device state as a structured `Driver(_)` so that
            // callers wanting a CPU-reason marker can distinguish "policy off"
            // (Ok(None)) from "driver present but no usable hardware"
            // (Err(Driver)). This keeps `GpuRuntime::probe()` honest: a
            // successful `Ok` always carries at least one device.
            return Err(GpuProbeError::Driver(reason.to_string()));
        }

        let mut devices = Vec::new();
        for ordinal in 0..usize::try_from(device_count)
            .map_err(|_| GpuProbeError::Driver("negative CUDA device count".into()))?
        {
            let ctx = cuda_context_for(ordinal).ok_or_else(|| {
                GpuProbeError::Driver(format!(
                    "failed to create CUDA context for device {ordinal}"
                ))
            })?;
            ctx.bind_to_thread()
                .map_err(|err| GpuProbeError::Driver(err.to_string()))?;
            devices.push(cuda_device_info(ordinal, &ctx)?);
        }

        devices.sort_by(|a, b| b.score().total_cmp(&a.score()));
        let Some(device) = devices.first().cloned() else {
            Self::record_cpu_reason("CUDA driver reported no usable devices");
            diagnostics::log_cuda_disabled("CUDA driver reported no usable devices");
            return Ok(None);
        };

        let policy = GpuDispatchPolicy::default();
        let memory_budget_bytes = device.free_mem_bytes.min(device.total_mem_bytes / 2);
        diagnostics::log_cuda_enabled(&device, &policy);
        diagnostics::log_cuda_pool(&devices);

        Ok(Some(Self {
            device,
            devices,
            policy,
            memory_budget_bytes,
        }))
    }

    #[must_use]
    pub fn global() -> Option<&'static Self> {
        static RUNTIME: OnceLock<Option<GpuRuntime>> = OnceLock::new();
        RUNTIME
            .get_or_init(|| match Self::probe() {
                Ok(runtime) => runtime,
                Err(GpuProbeError::Driver(reason)) => {
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

pub fn cuda_context_for(ordinal: usize) -> Option<Arc<CudaContext>> {
    static CONTEXTS: OnceLock<Mutex<HashMap<usize, Arc<CudaContext>>>> = OnceLock::new();
    let contexts = CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(ctx) = contexts.lock().ok()?.get(&ordinal).cloned() {
        return Some(ctx);
    }
    let ctx = CudaContext::new(ordinal).ok()?;
    let mut guard = contexts.lock().ok()?;
    Some(guard.entry(ordinal).or_insert_with(|| ctx.clone()).clone())
}

fn cuda_device_info(ordinal: usize, ctx: &CudaContext) -> Result<GpuDeviceInfo, GpuProbeError> {
    result::init().map_err(|err| GpuProbeError::Driver(err.to_string()))?;
    let device = result::device::get(
        i32::try_from(ordinal)
            .map_err(|_| GpuProbeError::Driver("device ordinal overflow".into()))?,
    )
    .map_err(|err| GpuProbeError::Driver(err.to_string()))?;
    let attr = |attribute| -> Result<i32, GpuProbeError> {
        // SAFETY: device comes from cudarc's validated device::get.
        unsafe { result::device::get_attribute(device, attribute) }
            .map_err(|err| GpuProbeError::Driver(err.to_string()))
    };
    let (free_mem_bytes, total_mem_bytes) = ctx
        .mem_get_info()
        .map_err(|err| GpuProbeError::Driver(err.to_string()))?;
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

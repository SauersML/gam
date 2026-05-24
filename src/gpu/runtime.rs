use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::device::GpuDeviceInfo;
use super::policy::GpuDispatchPolicy;
use cudarc::driver::{CudaContext, result, sys};

#[derive(Clone, Debug, Eq, PartialEq)]
#[must_use]
pub enum GpuProbeError {
    CudaFeatureDisabled,
    DynamicLoaderUnavailable,
    NoDevice,
    Driver(String),
}

#[derive(Clone, Debug)]
#[must_use]
pub struct GpuRuntime {
    pub device: GpuDeviceInfo,
    pub policy: GpuDispatchPolicy,
    pub memory_budget_bytes: usize,
}

impl GpuRuntime {
    pub fn probe() -> Result<Option<Self>, GpuProbeError> {
        let ordinal = 0;
        let ctx = match cuda_context_for(ordinal) {
            Some(ctx) => ctx,
            None => return Ok(None),
        };
        ctx.bind_to_thread()
            .map_err(|err| GpuProbeError::Driver(err.to_string()))?;
        let device = cuda_device_info(ordinal)?;
        let memory_budget_bytes = device.free_mem_bytes.max(device.total_mem_bytes / 2);
        Ok(Some(Self {
            device,
            policy: GpuDispatchPolicy::default(),
            memory_budget_bytes,
        }))
    }

    #[must_use]
    pub fn global() -> Option<&'static Self> {
        static RUNTIME: OnceLock<Option<GpuRuntime>> = OnceLock::new();
        RUNTIME
            .get_or_init(|| Self::probe().unwrap_or(None))
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
    pub fn selected_device(&self) -> Option<&GpuDeviceInfo> {
        Some(&self.device)
    }

    #[must_use]
    pub fn cpu_reason() -> Option<&'static str> {
        if Self::global().is_some() {
            None
        } else {
            Some("no CUDA device was probed")
        }
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

fn cuda_device_info(ordinal: usize) -> Result<GpuDeviceInfo, GpuProbeError> {
    result::init().map_err(|err| GpuProbeError::Driver(err.to_string()))?;
    let device = result::device::get(
        i32::try_from(ordinal).map_err(|_| GpuProbeError::Driver("device ordinal overflow".into()))?,
    )
    .map_err(|err| GpuProbeError::Driver(err.to_string()))?;
    let attr = |attribute| -> Result<i32, GpuProbeError> {
        // SAFETY: device comes from cudarc's validated device::get.
        unsafe { result::device::get_attribute(device, attribute) }
            .map_err(|err| GpuProbeError::Driver(err.to_string()))
    };
    let total_mem_bytes = {
        // SAFETY: device comes from cudarc's validated device::get.
        unsafe { result::device::total_mem(device) }
            .map_err(|err| GpuProbeError::Driver(err.to_string()))?
    };
    let major = attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let minor = attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    Ok(GpuDeviceInfo {
        ordinal,
        name: result::device::get_name(device)
            .unwrap_or_else(|_| format!("CUDA device {ordinal}")),
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
        free_mem_bytes: total_mem_bytes,
        ecc_enabled: attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_ECC_ENABLED)
            .unwrap_or(0)
            != 0,
        integrated: attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_INTEGRATED)
            .unwrap_or(0)
            != 0,
        mig_mode: false,
    })
}

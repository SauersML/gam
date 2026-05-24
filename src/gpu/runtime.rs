use std::sync::OnceLock;

use super::device::GpuDeviceInfo;
use super::policy::GpuDispatchPolicy;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuProbeError {
    CudaFeatureDisabled,
    DynamicLoaderUnavailable,
    NoDevice,
    Driver(String),
}

#[derive(Clone, Debug)]
pub struct GpuRuntime {
    pub device: GpuDeviceInfo,
    pub policy: GpuDispatchPolicy,
    pub memory_budget_bytes: usize,
}

impl GpuRuntime {
    pub fn probe() -> Result<Option<Self>, GpuProbeError> {
        #[cfg(feature = "cuda")]
        {
            Ok(None)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(None)
        }
    }

    pub fn global() -> Option<&'static Self> {
        static RUNTIME: OnceLock<Option<GpuRuntime>> = OnceLock::new();
        RUNTIME
            .get_or_init(|| Self::probe().unwrap_or(None))
            .as_ref()
    }

    pub fn is_available() -> bool {
        Self::global().is_some()
    }
}

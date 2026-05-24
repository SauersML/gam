use super::device::GpuDeviceInfo;
use super::policy::{GpuDispatchPolicy, GpuEnv};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum GpuRuntimeStatus {
    DisabledByEnv,
    Unavailable { reason: String },
    Available { device: GpuDeviceInfo },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GpuRuntime {
    pub env: GpuEnv,
    pub status: GpuRuntimeStatus,
    pub policy: GpuDispatchPolicy,
    pub resident_budget_bytes: usize,
}

impl GpuRuntime {
    #[must_use]
    pub fn probe() -> Option<Self> {
        let env = GpuEnv::from_env();
        if env.disabled() {
            log::debug!("[GPU] disabled by GAM_GPU=off");
            return None;
        }
        match Self::try_probe(&env) {
            Some(runtime) => Some(runtime),
            None => {
                if env.forced() {
                    log::warn!("[GPU] GAM_GPU=force requested but no CUDA device was detected");
                } else {
                    log::debug!("[GPU] CUDA unavailable; using CPU fallback");
                }
                None
            }
        }
    }

    #[must_use]
    pub fn global() -> Option<&'static Self> {
        static RUNTIME: OnceLock<Option<GpuRuntime>> = OnceLock::new();
        RUNTIME.get_or_init(Self::probe).as_ref()
    }

    #[must_use]
    pub fn is_available(&self) -> bool {
        matches!(self.status, GpuRuntimeStatus::Available { .. })
    }

    #[cfg(feature = "cuda")]
    fn try_probe(env: &GpuEnv) -> Option<Self> {
        let _cudarc_context_type = core::any::type_name::<cudarc::driver::CudaContext>();
        let ordinal = env.device.unwrap_or(0);
        match cudarc::driver::CudaContext::new(ordinal) {
            Ok(_ctx) => {
                // Safe cudarc context creation proved dynamic loading and device access.
                // Keep detailed attribute enumeration isolated for the real backend; these
                // conservative placeholders still make policy/logging deterministic.
                let total = 0usize;
                let free = 0usize;
                let info = GpuDeviceInfo {
                    ordinal,
                    name: format!("CUDA device {ordinal}"),
                    capability: super::device::GpuCapability::from_compute_capability(0, 0),
                    sm_count: 0,
                    max_threads_per_sm: 0,
                    shared_mem_per_block: 0,
                    l2_cache_bytes: 0,
                    total_mem_bytes: total,
                    free_mem_bytes: free,
                    ecc_enabled: false,
                    integrated: false,
                    mig_mode: false,
                };
                let budget = ((free as f64) * env.mem_fraction).max(0.0) as usize;
                log::info!("[GPU] CUDA runtime available on {}", info.name);
                Some(Self {
                    env: env.clone(),
                    status: GpuRuntimeStatus::Available { device: info },
                    policy: GpuDispatchPolicy::default(),
                    resident_budget_bytes: budget,
                })
            }
            Err(err) => {
                log::debug!("[GPU] cudarc probe failed: {err:?}");
                None
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn try_probe(env: &GpuEnv) -> Option<Self> {
        let _ = env;
        None
    }
}

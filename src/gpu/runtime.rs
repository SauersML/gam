use super::device::GpuDeviceInfo;
use super::policy::{GpuDispatchPolicy, GpuEnv};
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub enum GpuProbeStatus {
    Disabled,
    Unavailable(String),
    Available,
}

#[derive(Clone, Debug)]
pub struct GpuRuntime {
    pub status: GpuProbeStatus,
    pub env: GpuEnv,
    pub devices: Vec<GpuDeviceInfo>,
    pub selected_device: Option<usize>,
    pub memory_budget_bytes: usize,
    pub policy: GpuDispatchPolicy,
    pub calibration_cache_path: PathBuf,
}

impl GpuRuntime {
    pub fn probe() -> Option<Self> {
        let env = GpuEnv::from_process_env();
        if env.gpu == "off" {
            return None;
        }
        let cache = calibration_cache_path();
        let runtime = Self {
            status: GpuProbeStatus::Unavailable(cuda_backend_status().to_string()),
            env,
            devices: Vec::new(),
            selected_device: None,
            memory_budget_bytes: 0,
            policy: GpuDispatchPolicy::default(),
            calibration_cache_path: cache,
        };
        None.or(Some(runtime))
            .filter(|rt| matches!(rt.env.gpu.as_str(), "force"))
    }

    pub fn cpu_fallback() -> Self {
        Self {
            status: GpuProbeStatus::Unavailable(cuda_backend_status().to_string()),
            env: GpuEnv::from_process_env(),
            devices: Vec::new(),
            selected_device: None,
            memory_budget_bytes: 0,
            policy: GpuDispatchPolicy::default(),
            calibration_cache_path: calibration_cache_path(),
        }
    }

    pub fn is_available(&self) -> bool {
        matches!(self.status, GpuProbeStatus::Available)
    }
}

pub fn cuda_backend_status() -> &'static str {
    if cfg!(feature = "cuda") {
        "cuda feature enabled; cudarc dynamic backend is compiled but no device work is active in this routing layer"
    } else {
        "cuda feature disabled"
    }
}

pub fn calibration_cache_path() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"));
    base.join("gam").join("gpu_calib.json")
}

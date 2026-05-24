use std::env;
use std::process::Command;
use std::sync::OnceLock;

use super::device::GpuDeviceInfo;
use super::policy::{GpuDispatchPolicy, Operation, OperationDecision};

#[derive(Clone, Debug)]
pub enum ExecutionTarget {
    Cpu,
    Cuda(GpuContext),
}

#[derive(Clone, Debug)]
pub struct GpuContext {
    pub device: GpuDeviceInfo,
    pub policy: GpuDispatchPolicy,
}

impl GpuContext {
    #[must_use]
    pub fn target_for(&self, op: Operation) -> OperationDecision {
        self.policy.decide(op, true)
    }
}

#[derive(Clone, Debug)]
pub struct GpuRuntime {
    devices: Vec<GpuDeviceInfo>,
    selected: Option<GpuContext>,
    cuda_feature_enabled: bool,
    probe_message: String,
}

impl GpuRuntime {
    #[must_use]
    pub fn probe() -> Self {
        let cuda_feature_enabled = cfg!(feature = "cuda");
        let devices = if env::var("GAM_GPU").map_or(false, |v| is_off(&v)) {
            Vec::new()
        } else {
            probe_devices_with_nvidia_smi()
        };
        let selected_device = select_device(&devices);
        let selected = selected_device.map(|device| {
            let policy = GpuDispatchPolicy::from_env_and_device(Some(&device));
            GpuContext { device, policy }
        });
        let probe_message = if let Some(ctx) = &selected {
            format!(
                "[GAM GPU] CUDA runtime probe selected {}; cargo cuda feature: {}",
                ctx.device, cuda_feature_enabled
            )
        } else if cuda_feature_enabled {
            "[GAM GPU] cuda feature enabled, but no usable CUDA device was detected; CPU fallback active".to_string()
        } else {
            "[GAM GPU] cuda feature disabled or no usable CUDA device detected; CPU fallback active"
                .to_string()
        };
        Self {
            devices,
            selected,
            cuda_feature_enabled,
            probe_message,
        }
    }

    #[must_use]
    pub fn global() -> &'static Self {
        static RUNTIME: OnceLock<GpuRuntime> = OnceLock::new();
        RUNTIME.get_or_init(Self::probe)
    }

    #[must_use]
    pub fn is_available(&self) -> bool {
        self.selected.is_some()
    }

    #[must_use]
    pub fn selected_device(&self) -> Option<&GpuDeviceInfo> {
        self.selected.as_ref().map(|ctx| &ctx.device)
    }

    #[must_use]
    pub fn selected_context(&self) -> Option<&GpuContext> {
        self.selected.as_ref()
    }

    #[must_use]
    pub fn devices(&self) -> &[GpuDeviceInfo] {
        &self.devices
    }

    #[must_use]
    pub const fn cuda_feature_enabled(&self) -> bool {
        self.cuda_feature_enabled
    }

    #[must_use]
    pub fn probe_message(&self) -> &str {
        &self.probe_message
    }
}

#[must_use]
pub fn gpu_available() -> bool {
    GpuRuntime::global().is_available()
}

#[must_use]
pub fn selected_gpu_info() -> Option<GpuDeviceInfo> {
    GpuRuntime::global().selected_device().cloned()
}

fn select_device(devices: &[GpuDeviceInfo]) -> Option<GpuDeviceInfo> {
    if devices.is_empty() {
        return None;
    }
    if let Ok(raw) = env::var("GAM_GPU_DEVICE") {
        if let Ok(ordinal) = raw.parse::<usize>() {
            return devices.iter().find(|d| d.ordinal == ordinal).cloned();
        }
    }
    devices.iter().max_by_key(|device| device.score()).cloned()
}

fn probe_devices_with_nvidia_smi() -> Vec<GpuDeviceInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,uuid,compute_cap,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .enumerate()
        .filter_map(|(ordinal, line)| GpuDeviceInfo::from_nvidia_smi_csv(ordinal, line))
        .collect()
}

fn is_off(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "off" | "0" | "false" | "cpu" | "cpu-only"
    )
}

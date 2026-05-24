use std::process::Command;
use std::sync::OnceLock;

use crate::gpu::device::GpuDeviceInfo;
use crate::gpu::policy::{GpuDispatchPolicy, Operation};

#[derive(Clone, Debug)]
pub struct GpuRuntime {
    devices: Vec<GpuDeviceInfo>,
    selected: Option<usize>,
    policy: GpuDispatchPolicy,
    validate: bool,
    profile: bool,
    cuda_feature_compiled: bool,
}

impl GpuRuntime {
    #[must_use]
    pub fn probe() -> Self {
        let devices = probe_with_nvidia_smi();
        let requested = std::env::var("GAM_GPU_DEVICE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        let selected = requested
            .filter(|idx| devices.iter().any(|d| d.index == *idx))
            .or_else(|| best_device(&devices));
        let selected_info = selected.and_then(|idx| devices.iter().find(|d| d.index == idx));
        let policy = GpuDispatchPolicy::from_env_and_device(selected_info);
        Self {
            devices,
            selected,
            policy,
            validate: env_flag("GAM_GPU_VALIDATE"),
            profile: env_flag("GAM_GPU_PROFILE"),
            cuda_feature_compiled: cfg!(feature = "cuda"),
        }
    }

    #[must_use]
    pub fn devices(&self) -> &[GpuDeviceInfo] {
        &self.devices
    }

    #[must_use]
    pub fn selected_device(&self) -> Option<&GpuDeviceInfo> {
        self.selected
            .and_then(|idx| self.devices.iter().find(|d| d.index == idx))
    }

    #[must_use]
    pub fn policy(&self) -> &GpuDispatchPolicy {
        &self.policy
    }

    #[must_use]
    pub fn is_gpu_enabled(&self) -> bool {
        self.selected.is_some()
            && self.policy.should_try_gpu(
                Operation::RowKernel {
                    rows: usize::MAX,
                    lanes: 1,
                    resident: true,
                },
                true,
            )
    }

    #[must_use]
    pub fn profile_enabled(&self) -> bool {
        self.profile
    }

    #[must_use]
    pub fn validate_enabled(&self) -> bool {
        self.validate
    }

    #[must_use]
    pub fn cuda_feature_compiled(&self) -> bool {
        self.cuda_feature_compiled
    }

    #[must_use]
    pub fn should_try_gpu(&self, op: Operation) -> bool {
        self.policy.should_try_gpu(op, self.selected.is_some())
    }
}

#[must_use]
pub fn global_runtime() -> &'static GpuRuntime {
    static RUNTIME: OnceLock<GpuRuntime> = OnceLock::new();
    RUNTIME.get_or_init(GpuRuntime::probe)
}

#[must_use]
pub fn selected_gpu_info() -> Option<GpuDeviceInfo> {
    global_runtime().selected_device().cloned()
}

fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref().map(str::trim),
        Some("1")
            | Some("true")
            | Some("TRUE")
            | Some("yes")
            | Some("YES")
            | Some("on")
            | Some("ON")
    )
}

fn best_device(devices: &[GpuDeviceInfo]) -> Option<usize> {
    devices
        .iter()
        .max_by_key(|d| {
            (
                d.memory_score_bytes(),
                d.compute_capability.unwrap_or((0, 0)),
            )
        })
        .map(|d| d.index)
}

fn probe_with_nvidia_smi() -> Vec<GpuDeviceInfo> {
    // CUDA driver probing is intentionally isolated behind this facade.  In
    // default CPU builds, nvidia-smi gives a zero-link-dependency best effort;
    // CUDA builds can grow this function into driver-API probing without
    // touching linalg/PIRLS/REML call sites.
    let Ok(output) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,uuid,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.lines().filter_map(parse_nvidia_smi_line).collect()
}

fn parse_nvidia_smi_line(line: &str) -> Option<GpuDeviceInfo> {
    let parts: Vec<_> = line.split(',').map(str::trim).collect();
    if parts.len() < 6 {
        return None;
    }
    let index = parts[0].parse::<usize>().ok()?;
    let total_mib = parts[3].parse::<u64>().ok();
    let free_mib = parts[4].parse::<u64>().ok();
    let compute_capability = parts[5]
        .split_once('.')
        .and_then(|(major, minor)| Some((major.parse::<u32>().ok()?, minor.parse::<u32>().ok()?)));
    Some(GpuDeviceInfo {
        index,
        name: parts[1].to_string(),
        uuid: (!parts[2].is_empty()).then(|| parts[2].to_string()),
        compute_capability,
        total_memory_bytes: total_mib.map(|v| v * 1024 * 1024),
        free_memory_bytes: free_mib.map(|v| v * 1024 * 1024),
    })
}

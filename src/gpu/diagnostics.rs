#[cfg(target_os = "linux")]
use super::super::device::GpuDeviceInfo;
#[cfg(target_os = "linux")]
use super::super::policy::GpuDispatchPolicy;
use super::GpuRuntime;
use std::sync::OnceLock;

#[cfg(target_os = "linux")]
pub(crate) fn log_cuda_enabled(device: &GpuDeviceInfo, policy: &GpuDispatchPolicy) {
    log::info!(
        "[GPU] CUDA acceleration enabled\n  device: {} '{}' | memory={}\n  libraries: CUDA driver ready; cuBLAS/cuSOLVER/cuSPARSE load on first use\n  dispatch: xtwx>={}flop gemm>={}flop spmv_nnz>={} chol_p>={} syevd_p>={}",
        device.ordinal,
        device.name,
        format_bytes(device.total_mem_bytes),
        format_count(policy.xtwx_flops_min.min(policy.gemm_min_flops) as u64),
        format_count(policy.gemm_min_flops as u64),
        format_count(policy.sparse_min_nnz as u64),
        policy.potrf_min_p,
        policy.syevd_min_p,
    );
}

#[cfg(target_os = "linux")]
pub(crate) fn log_cuda_pool(devices: &[GpuDeviceInfo]) {
    if devices.len() <= 1 {
        return;
    }
    let summary = devices
        .iter()
        .map(|device| {
            format!(
                "{}:'{}' sm_{}{} {}SM {}",
                device.ordinal,
                device.name,
                device.capability.compute_major,
                device.capability.compute_minor,
                device.sm_count,
                format_bytes(device.total_mem_bytes),
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    log::info!(
        "[GPU] multi-device pool enabled | devices={} | policy_device={} | {}",
        devices.len(),
        devices[0].ordinal,
        summary,
    );
}

pub(crate) fn log_cuda_disabled(reason: &str) {
    static CUDA_DISABLED_LOGGED: OnceLock<()> = OnceLock::new();
    let reason = GpuRuntime::cpu_reason().unwrap_or(reason);
    CUDA_DISABLED_LOGGED.get_or_init(|| {
        log::info!("[GPU] CUDA acceleration disabled: {reason}");
    });
}

#[cfg(target_os = "linux")]
fn format_count(value: u64) -> String {
    if value == u64::MAX {
        return "never".to_string();
    }
    if value >= 1_000_000_000_000 {
        format!("{:.2}T", value as f64 / 1_000_000_000_000.0)
    } else if value >= 1_000_000_000 {
        format!("{:.2}B", value as f64 / 1_000_000_000.0)
    } else if value >= 1_000_000 {
        format!("{:.2}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.2}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

#[cfg(target_os = "linux")]
fn format_bytes(bytes: usize) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const KIB: f64 = 1024.0;
    let bytes_f = bytes as f64;
    if bytes_f >= GIB {
        format!("{:.2}GiB", bytes_f / GIB)
    } else if bytes_f >= MIB {
        format!("{:.2}MiB", bytes_f / MIB)
    } else if bytes_f >= KIB {
        format!("{:.2}KiB", bytes_f / KIB)
    } else {
        format!("{bytes}B")
    }
}

use super::super::device::GpuDeviceInfo;
use super::super::policy::GpuDispatchPolicy;

pub(crate) fn log_cuda_enabled(device: &GpuDeviceInfo, policy: &GpuDispatchPolicy) {
    log::info!(
        "[GPU] CUDA acceleration enabled\n  device: {} '{}' | memory={}\n  libraries: CUDA driver ready; cuBLAS/cuSOLVER/cuSPARSE load on first use\n  dispatch: xtwx_rows>={} gemm>={}flop spmv_nnz>={} chol_p>={} syevd_p>={}",
        device.ordinal,
        device.name,
        format_bytes(device.total_mem_bytes),
        policy.xtwx_n_min,
        format_count(policy.gemm_min_flops as u64),
        format_count(policy.sparse_min_nnz as u64),
        policy.potrf_min_p,
        policy.syevd_min_p,
    );
}

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
    log::info!(
        "[GPU] CUDA acceleration disabled\n  backend: CPU (faer + Rayon)\n  reason: {}",
        reason
    );
}

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

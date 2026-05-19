use std::collections::{BTreeMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use super::device::GpuDeviceInfo;
use super::policy::DispatchPolicy;
use super::runtime::GpuRuntime;

pub(crate) fn log_cuda_enabled(device: &GpuDeviceInfo, policy: &DispatchPolicy) {
    log::info!(
        "[GPU] CUDA acceleration enabled\n  device: {}\n  measured: fp64={} GFLOP/s h2d={:.1} GB/s d2h={:.1} GB/s memory={}\n  libraries: CUDA driver ready; cuBLAS/cuSOLVER/cuSPARSE load on first use\n  dispatch: xtwx_rows>={} gemm>={}flop gemv>={}flop spmv_rows>={} spmv_nnz>={} chol_p>={} syevd_p>={} trsm>={}flop",
        device,
        format_float(device.peak_fp64_gflops()),
        device.calibration.h2d_gb_s,
        device.calibration.d2h_gb_s,
        format_bytes(device.total_memory_bytes),
        policy.xtwx_min_rows,
        format_count(policy.gemm_min_flops),
        format_count(policy.gemv_min_flops),
        policy.spmv_min_rows,
        format_count(policy.spmv_min_nnz as u64),
        policy.chol_min_p,
        policy.syevd_min_p,
        format_count(policy.trsm_min_flops),
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
                device.compute_capability_major,
                device.compute_capability_minor,
                device.sm_count,
                format_bytes(device.total_memory_bytes),
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

pub(crate) fn log_library_ready(library: &'static str, device: &GpuDeviceInfo) {
    log_route(format!(
        "[GPU] {library} ready | device={} '{}' | CUDA library handle initialized",
        device.ordinal, device.name,
    ));
}

pub(crate) fn log_library_unavailable(library: &'static str, reason: &str) {
    log_route(format!(
        "[GPU] {library} unavailable | CPU route retained | reason={reason}"
    ));
}

pub(crate) fn log_policy_cpu(op: &'static str, shape: String, reason: String) {
    activity::record_policy_decline(op);
    log_route(format!(
        "[GPU] CPU route | op={op} | shape={shape} | reason={reason}"
    ));
}

pub(crate) fn dispatch_decline_reason(policy_reason: String) -> String {
    if let Some(cpu_reason) = GpuRuntime::global().cpu_reason() {
        format!("CUDA unavailable: {cpu_reason}")
    } else {
        policy_reason
    }
}

pub(crate) fn log_runtime_cpu(op: &'static str, backend: &'static str, shape: String) {
    activity::record_runtime_decline(op);
    log_route(format!(
        "[GPU] CPU route | op={op} | backend={backend} | shape={shape} | reason=device dispatch returned no result"
    ));
}

pub(crate) fn log_gpu_success(
    op: &'static str,
    backend: &'static str,
    device: &GpuDeviceInfo,
    shape: String,
    flops: u64,
    h2d_bytes: usize,
    d2h_bytes: usize,
    elapsed_s: f64,
) {
    activity::record_success(op, flops, h2d_bytes, d2h_bytes, elapsed_s);
    log_route(format!(
        "[GPU] device route | op={op} | backend={backend} | device={} '{}' | shape={shape} | work={}flop | transfer=h2d:{} d2h:{} | elapsed={:.3}s",
        device.ordinal,
        device.name,
        format_count(flops),
        format_bytes(h2d_bytes),
        format_bytes(d2h_bytes),
        elapsed_s,
    ));
}

pub(crate) fn log_gpu_success_multi(
    op: &'static str,
    backend: &'static str,
    devices: &[GpuDeviceInfo],
    shape: String,
    flops: u64,
    h2d_bytes: usize,
    d2h_bytes: usize,
    elapsed_s: f64,
) {
    if devices.len() == 1 {
        log_gpu_success(
            op,
            backend,
            &devices[0],
            shape,
            flops,
            h2d_bytes,
            d2h_bytes,
            elapsed_s,
        );
        return;
    }
    activity::record_success(op, flops, h2d_bytes, d2h_bytes, elapsed_s);
    let device_summary = devices
        .iter()
        .map(|device| format!("{} '{}'", device.ordinal, device.name))
        .collect::<Vec<_>>()
        .join(", ");
    log_route(format!(
        "[GPU] multi-device route | op={op} | backend={backend} | devices=[{device_summary}] | shape={shape} | work={}flop | transfer=h2d:{} d2h:{} | elapsed={:.3}s",
        format_count(flops),
        format_bytes(h2d_bytes),
        format_bytes(d2h_bytes),
        elapsed_s,
    ));
}

/// Log each unique GPU routing message exactly once per process. Long fits
/// often repeat the same small CPU-routed kernel thousands of times and
/// interleave a few shapes; emitting each signature only on first occurrence
/// keeps the log readable.
fn log_route(signature: String) {
    static SEEN: Mutex<Option<HashSet<String>>> = Mutex::new(None);

    let mut guard = match SEEN.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    let seen = guard.get_or_insert_with(HashSet::new);
    if seen.insert(signature.clone()) {
        log::info!("{signature}");
    }
}

pub(crate) fn bytes_for_f64(len: usize) -> usize {
    len.saturating_mul(std::mem::size_of::<f64>())
}

pub(crate) fn bytes_for_i32(len: usize) -> usize {
    len.saturating_mul(std::mem::size_of::<i32>())
}

pub(crate) fn gemm_flops(m: usize, n: usize, k: usize) -> u64 {
    (m as u64)
        .saturating_mul(n as u64)
        .saturating_mul(k.max(1) as u64)
        .saturating_mul(2)
}

pub(crate) fn gemv_flops(rows: usize, cols: usize) -> u64 {
    (rows as u64).saturating_mul(cols as u64).saturating_mul(2)
}

pub(crate) fn chol_flops(p: usize) -> u64 {
    let p64 = p as u64;
    p64.saturating_mul(p64).saturating_mul(p64) / 3
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

fn format_float(value: f64) -> String {
    if value >= 100.0 {
        format!("{value:.0}")
    } else {
        format!("{value:.1}")
    }
}

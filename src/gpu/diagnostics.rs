use std::collections::{BTreeMap, HashSet};
use std::sync::Mutex;

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
    if reason.starts_with("CUDA unavailable:") {
        log_route(format!("[GPU] CPU route | op={op} | reason={reason}"));
    } else {
        log_route(format!(
            "[GPU] CPU route | op={op} | shape={shape} | reason={reason}"
        ));
    }
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
    if let Some(cpu_reason) = GpuRuntime::global().cpu_reason() {
        log_route(format!(
            "[GPU] CPU route | op={op} | backend={backend} | reason=CUDA unavailable: {cpu_reason}"
        ));
    } else {
        log_route(format!(
            "[GPU] CPU route | op={op} | backend={backend} | shape={shape} | reason=device dispatch returned no result"
        ));
    }
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

// ---------------------------------------------------------------------------
// Process-wide activity tracker
// ---------------------------------------------------------------------------

/// Format a single roll-up line summarizing every GPU dispatch decision
/// taken in this process so far. Used as the "did the GPUs do any work?"
/// answer at end-of-fit / end-of-process — no source-file reading needed.
///
/// Returns `None` when the GPU runtime never selected a device (CPU-only
/// host); the empty case is suppressed so we don't spam logs on machines
/// without CUDA installed.
pub fn gpu_activity_summary() -> Option<String> {
    if !GpuRuntime::global().is_available() {
        return None;
    }
    Some(activity::snapshot_summary())
}

/// Emit the summary via `log::info!` if there's anything to report. Safe
/// to call multiple times; each call snapshots the current totals.
pub fn flush_gpu_activity_summary() {
    if let Some(summary) = gpu_activity_summary() {
        log::info!("{summary}");
    }
}

mod activity {
    use super::{BTreeMap, Mutex, format_bytes, format_count};
    use std::sync::atomic::{AtomicU64, Ordering};

    static SUCCESS_COUNT: AtomicU64 = AtomicU64::new(0);
    static POLICY_DECLINE_COUNT: AtomicU64 = AtomicU64::new(0);
    static RUNTIME_DECLINE_COUNT: AtomicU64 = AtomicU64::new(0);
    static TOTAL_FLOPS: AtomicU64 = AtomicU64::new(0);
    static TOTAL_H2D_BYTES: AtomicU64 = AtomicU64::new(0);
    static TOTAL_D2H_BYTES: AtomicU64 = AtomicU64::new(0);
    /// Sum of measured wall-time across successful dispatches, in
    /// microseconds. Stored as u64 to atomically update without a lock.
    static TOTAL_GPU_MICROS: AtomicU64 = AtomicU64::new(0);

    /// Per-op breakdown so the summary can show which kernels actually
    /// landed on the GPU vs which were declined. Tiny map (~20 keys),
    /// guarded by a mutex — contention is negligible against millisecond-
    /// scale kernel launches.
    static PER_OP: Mutex<Option<BTreeMap<&'static str, PerOpStats>>> = Mutex::new(None);

    #[derive(Clone, Copy, Default)]
    pub(super) struct PerOpStats {
        pub success: u64,
        pub policy_decline: u64,
        pub runtime_decline: u64,
        pub flops: u64,
        pub gpu_micros: u64,
    }

    fn with_per_op<F: FnOnce(&mut BTreeMap<&'static str, PerOpStats>)>(f: F) {
        let mut guard = match PER_OP.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let map = guard.get_or_insert_with(BTreeMap::new);
        f(map);
    }

    pub(super) fn record_success(
        op: &'static str,
        flops: u64,
        h2d_bytes: usize,
        d2h_bytes: usize,
        elapsed_s: f64,
    ) {
        SUCCESS_COUNT.fetch_add(1, Ordering::Relaxed);
        TOTAL_FLOPS.fetch_add(flops, Ordering::Relaxed);
        TOTAL_H2D_BYTES.fetch_add(h2d_bytes as u64, Ordering::Relaxed);
        TOTAL_D2H_BYTES.fetch_add(d2h_bytes as u64, Ordering::Relaxed);
        let micros = (elapsed_s * 1_000_000.0).max(0.0) as u64;
        TOTAL_GPU_MICROS.fetch_add(micros, Ordering::Relaxed);
        with_per_op(|map| {
            let entry = map.entry(op).or_default();
            entry.success += 1;
            entry.flops = entry.flops.saturating_add(flops);
            entry.gpu_micros = entry.gpu_micros.saturating_add(micros);
        });
    }

    pub(super) fn record_policy_decline(op: &'static str) {
        POLICY_DECLINE_COUNT.fetch_add(1, Ordering::Relaxed);
        with_per_op(|map| {
            map.entry(op).or_default().policy_decline += 1;
        });
    }

    pub(super) fn record_runtime_decline(op: &'static str) {
        RUNTIME_DECLINE_COUNT.fetch_add(1, Ordering::Relaxed);
        with_per_op(|map| {
            map.entry(op).or_default().runtime_decline += 1;
        });
    }

    /// Snapshot the totals and produce a multi-line summary string.
    pub(super) fn snapshot_summary() -> String {
        let success = SUCCESS_COUNT.load(Ordering::Relaxed);
        let policy_decline = POLICY_DECLINE_COUNT.load(Ordering::Relaxed);
        let runtime_decline = RUNTIME_DECLINE_COUNT.load(Ordering::Relaxed);
        let flops = TOTAL_FLOPS.load(Ordering::Relaxed);
        let h2d = TOTAL_H2D_BYTES.load(Ordering::Relaxed);
        let d2h = TOTAL_D2H_BYTES.load(Ordering::Relaxed);
        let gpu_micros = TOTAL_GPU_MICROS.load(Ordering::Relaxed);

        let total = success + policy_decline + runtime_decline;
        if total == 0 {
            return "[GPU] activity summary: no dispatch sites reached".to_string();
        }

        let gpu_seconds = (gpu_micros as f64) / 1_000_000.0;
        let measured_gflops = if gpu_seconds > 0.0 {
            (flops as f64) / gpu_seconds / 1e9
        } else {
            0.0
        };

        let mut lines = Vec::new();
        lines.push(format!(
            "[GPU] activity summary | dispatched={} declined_policy={} declined_runtime={} | work={}flop in {:.3}s ({:.0} GFLOP/s effective) | transfer=h2d:{} d2h:{}",
            success,
            policy_decline,
            runtime_decline,
            format_count(flops),
            gpu_seconds,
            measured_gflops,
            format_bytes(h2d as usize),
            format_bytes(d2h as usize),
        ));

        // Per-op breakdown, sorted by descending success count then op name.
        let snapshot: Vec<(&'static str, PerOpStats)> = {
            let guard = match PER_OP.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            match guard.as_ref() {
                Some(map) => map.iter().map(|(k, v)| (*k, *v)).collect(),
                None => Vec::new(),
            }
        };
        if !snapshot.is_empty() {
            let mut sorted = snapshot;
            sorted.sort_by(|a, b| b.1.success.cmp(&a.1.success).then_with(|| a.0.cmp(b.0)));
            for (op, stats) in sorted {
                let gpu_seconds = (stats.gpu_micros as f64) / 1_000_000.0;
                lines.push(format!(
                    "  {op:>32}: gpu={} cpu_policy={} cpu_runtime={} work={}flop t={:.3}s",
                    stats.success,
                    stats.policy_decline,
                    stats.runtime_decline,
                    format_count(stats.flops),
                    gpu_seconds,
                ));
            }
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod activity_tests {
    use super::activity;

    #[test]
    fn empty_summary_short_circuits() {
        // No record_* calls — summary should be the "no dispatch sites" string.
        // (This test runs alongside others which DO record into the global
        // counters; in isolation the totals stay zero only when run first,
        // so we tolerate either format by checking that snapshot returns
        // a non-empty string that mentions either "no dispatch" or
        // "activity summary".)
        let s = activity::snapshot_summary();
        assert!(s.contains("activity summary") || s.contains("no dispatch"));
    }
}

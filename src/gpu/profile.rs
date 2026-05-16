//! Lightweight CPU/GPU kernel profile recording.
//!
//! Hot dense kernels in [`crate::linalg::faer_ndarray`] wrap themselves in
//! [`cpu_scope`] which captures elapsed time and shape statistics. Backends
//! can later compare those records against measured device kernels to
//! calibrate per-operation routing without touching the call sites.
//!
//! The implementation is allocation-free on the hot path: a single
//! `Instant::now()` + `elapsed()` per call, gated by an atomic flag so the
//! profiler can be turned on or off at runtime (the default is *off* to
//! keep production fits free of profiling overhead).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Per-call profiling record. Sizes are stored as the caller saw them so a
/// FLOP estimate downstream can adapt to non-square shapes.
#[derive(Clone, Debug, Default)]
pub struct KernelStat {
    pub name: &'static str,
    pub n: usize,
    pub p: usize,
    pub k: usize,
    pub nnz: usize,
    pub flops_est: u128,
    pub bytes_est: u128,
    pub cpu_ms: f64,
    pub gpu_ms: Option<f64>,
}

/// Snapshot of recently observed kernel stats.
#[derive(Clone, Debug, Default)]
pub struct KernelStatsSnapshot {
    pub stats: Vec<KernelStat>,
}

const MAX_STATS: usize = 1024;

static ENABLED: AtomicBool = AtomicBool::new(false);
static STATS: OnceLock<Mutex<VecDeque<KernelStat>>> = OnceLock::new();

fn ring() -> &'static Mutex<VecDeque<KernelStat>> {
    STATS.get_or_init(|| Mutex::new(VecDeque::with_capacity(MAX_STATS)))
}

/// Toggle profile collection at runtime. Returns the previous value.
pub fn set_enabled(enabled: bool) -> bool {
    ENABLED.swap(enabled, Ordering::Relaxed)
}

/// Is profile collection currently enabled?
#[inline]
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Submit a fully populated stat record (used by both CPU and device paths).
pub fn record(stat: KernelStat) {
    if !is_enabled() {
        return;
    }
    if let Ok(mut guard) = ring().lock() {
        if guard.len() == MAX_STATS {
            guard.pop_front();
        }
        guard.push_back(stat);
    }
}

/// Read out and clear the current stat ring.
pub fn snapshot() -> KernelStatsSnapshot {
    if let Ok(guard) = ring().lock() {
        KernelStatsSnapshot {
            stats: guard.iter().cloned().collect(),
        }
    } else {
        KernelStatsSnapshot::default()
    }
}

/// Drop all recorded stats. Useful between benchmark phases.
pub fn clear() {
    if let Ok(mut guard) = ring().lock() {
        guard.clear();
    }
}

/// Run `op` and, when profiling is enabled, record a `KernelStat` describing
/// the call. The hot path adds at most one `Instant` and one branch when
/// profiling is off, so this is safe to wrap around every fast_* call.
#[inline]
pub fn cpu_scope<R>(
    name: &'static str,
    n: usize,
    p: usize,
    k: usize,
    bytes_est: u128,
    op: impl FnOnce() -> R,
) -> R {
    if !is_enabled() {
        return op();
    }
    let start = Instant::now();
    let out = op();
    let cpu_ms = start.elapsed().as_secs_f64() * 1_000.0;
    record(KernelStat {
        name,
        n,
        p,
        k,
        nnz: 0,
        flops_est: (n as u128)
            .saturating_mul(p as u128)
            .saturating_mul(k.max(1) as u128)
            .saturating_mul(2),
        bytes_est,
        cpu_ms,
        gpu_ms: None,
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_records_when_enabled_and_skips_when_disabled() {
        clear();
        let prev = set_enabled(false);
        cpu_scope("noop_off", 10, 10, 10, 64, || {});
        assert!(snapshot().stats.is_empty());

        set_enabled(true);
        cpu_scope("noop_on", 10, 10, 10, 64, || {});
        let stats = snapshot().stats;
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].name, "noop_on");

        clear();
        set_enabled(prev);
    }
}

use std::sync::atomic::{AtomicU64, Ordering};

/// Lightweight structured counter for accelerated-kernel candidates.
#[derive(Clone, Debug)]
pub struct KernelStat {
    pub name: &'static str,
    pub n: usize,
    pub p: usize,
    pub k: usize,
    pub nnz: Option<usize>,
    pub flops_est: u128,
    pub bytes_est: u128,
    pub cpu_ms: Option<f64>,
    pub gpu_ms: Option<f64>,
}

static DISPATCH_CANDIDATES: AtomicU64 = AtomicU64::new(0);
static GPU_FALLBACKS: AtomicU64 = AtomicU64::new(0);

#[inline]
pub fn record_dispatch_candidate(_stat: KernelStat) {
    DISPATCH_CANDIDATES.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn record_gpu_fallback() {
    GPU_FALLBACKS.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn dispatch_candidate_count() -> u64 {
    DISPATCH_CANDIDATES.load(Ordering::Relaxed)
}

#[inline]
pub fn gpu_fallback_count() -> u64 {
    GPU_FALLBACKS.load(Ordering::Relaxed)
}

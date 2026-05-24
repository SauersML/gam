use std::fmt;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct KernelStat {
    pub name: &'static str,
    pub n: usize,
    pub p: usize,
    pub k: usize,
    pub nnz: Option<usize>,
    pub flops_est: f64,
    pub bytes_est: f64,
    pub cpu: Option<Duration>,
    pub gpu: Option<Duration>,
}

impl fmt::Display for KernelStat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} n={} p={} k={} nnz={:?} flops={:.3e} bytes={:.3e} cpu_ms={:?} gpu_ms={:?}",
            self.name,
            self.n,
            self.p,
            self.k,
            self.nnz,
            self.flops_est,
            self.bytes_est,
            self.cpu.map(|d| d.as_secs_f64() * 1_000.0),
            self.gpu.map(|d| d.as_secs_f64() * 1_000.0)
        )
    }
}

static STATS: OnceLock<Mutex<Vec<KernelStat>>> = OnceLock::new();

pub fn record(stat: KernelStat) {
    if !profile_enabled() {
        return;
    }
    let stats = STATS.get_or_init(|| Mutex::new(Vec::new()));
    if let Ok(mut guard) = stats.lock() {
        guard.push(stat);
    }
}

#[must_use]
pub fn take_stats() -> Vec<KernelStat> {
    let stats = STATS.get_or_init(|| Mutex::new(Vec::new()));
    stats
        .lock()
        .map_or_else(|_| Vec::new(), |mut guard| std::mem::take(&mut *guard))
}

#[must_use]
pub fn profile_enabled() -> bool {
    matches!(
        std::env::var("GAM_GPU_PROFILE")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

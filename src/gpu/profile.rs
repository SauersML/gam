use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct KernelStat {
    pub name: &'static str,
    pub n: usize,
    pub p: usize,
    pub k: usize,
    pub nnz: usize,
    pub flops_est: usize,
    pub bytes_est: usize,
    pub cpu_ms: f64,
    pub gpu_ms: Option<f64>,
    pub backend: &'static str,
}

#[derive(Debug)]
pub struct GpuProfile {
    capacity: usize,
    stats: Mutex<VecDeque<KernelStat>>,
}

impl GpuProfile {
    #[must_use]
    pub fn global() -> &'static Self {
        static PROFILE: OnceLock<GpuProfile> = OnceLock::new();
        PROFILE.get_or_init(|| GpuProfile {
            capacity: 512,
            stats: Mutex::new(VecDeque::with_capacity(512)),
        })
    }

    pub fn push(&self, stat: KernelStat) {
        if let Ok(mut stats) = self.stats.lock() {
            if stats.len() == self.capacity {
                stats.pop_front();
            }
            stats.push_back(stat);
        }
    }

    #[must_use]
    pub fn snapshot(&self) -> Vec<KernelStat> {
        self.stats
            .lock()
            .map(|stats| stats.iter().cloned().collect())
            .unwrap_or_default()
    }
}

pub fn record_cpu_kernel<R, F: FnOnce() -> R>(
    name: &'static str,
    n: usize,
    p: usize,
    k: usize,
    nnz: usize,
    f: F,
) -> R {
    let start = Instant::now();
    let out = f();
    let cpu_ms = start.elapsed().as_secs_f64() * 1000.0;
    let flops_est = n.saturating_mul(p.max(1)).saturating_mul(k.max(1));
    let bytes_est = n.saturating_mul(p.max(1)).saturating_mul(8);
    GpuProfile::global().push(KernelStat {
        name,
        n,
        p,
        k,
        nnz,
        flops_est,
        bytes_est,
        cpu_ms,
        gpu_ms: None,
        backend: "cpu",
    });
    out
}

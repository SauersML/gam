use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

const MAX_STATS: usize = 1024;

#[derive(Clone, Debug, Default)]
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
}

#[derive(Clone, Debug, Default)]
pub struct KernelStatsSnapshot {
    pub stats: Vec<KernelStat>,
}

static STATS: OnceLock<Mutex<VecDeque<KernelStat>>> = OnceLock::new();

fn stats() -> &'static Mutex<VecDeque<KernelStat>> {
    STATS.get_or_init(|| Mutex::new(VecDeque::with_capacity(MAX_STATS)))
}

pub fn record(stat: KernelStat) {
    if let Ok(mut guard) = stats().lock() {
        if guard.len() == MAX_STATS {
            guard.pop_front();
        }
        guard.push_back(stat);
    }
}

pub fn snapshot() -> KernelStatsSnapshot {
    if let Ok(guard) = stats().lock() {
        KernelStatsSnapshot {
            stats: guard.iter().cloned().collect(),
        }
    } else {
        KernelStatsSnapshot::default()
    }
}

pub fn clear() {
    if let Ok(mut guard) = stats().lock() {
        guard.clear();
    }
}

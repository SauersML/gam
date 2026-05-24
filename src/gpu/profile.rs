use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

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

#[derive(Clone, Copy, Debug)]
pub enum OperationKind {
    Gemv,
    GemvTranspose,
    JointHessian,
    JointHessian2x2,
    XtDiagX,
    XtDiagY,
}

#[inline]
pub fn profiling_enabled() -> bool {
    false
}

pub fn cpu_scope<R>(
    name: &'static str,
    n: usize,
    p: usize,
    k: usize,
    bytes_est: usize,
    op: impl FnOnce() -> R,
) -> R {
    let start = Instant::now();
    let out = op();
    let cpu_ms = start.elapsed().as_secs_f64() * 1_000.0;
    record(KernelStat {
        name,
        n,
        p,
        k,
        nnz: 0,
        flops_est: n.saturating_mul(p).saturating_mul(k).saturating_mul(2),
        bytes_est,
        cpu_ms,
        gpu_ms: None,
    });
    out
}

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperationKind {
    Gemv,
    GemvTranspose,
    Gemm,
    XtDiagX,
    XtDiagY,
    JointHessian2x2,
    LinkKernel,
    CandidateScreen,
    Cholesky,
    Syevd,
    SparseXtWx,
    Pcg,
    SpatialKernel,
    RemlTrace,
    FamilyKernel,
    CudaGraph,
    MultiGpuReduce,
}

#[derive(Clone, Debug)]
pub struct KernelStat {
    pub name: &'static str,
    pub kind: OperationKind,
    pub n: usize,
    pub p: usize,
    pub k: usize,
    pub nnz: usize,
    pub flops_est: f64,
    pub bytes_est: usize,
    pub cpu_ms: Option<f64>,
    pub gpu_ms: Option<f64>,
    pub target: &'static str,
}

pub struct KernelStatRing {
    capacity: usize,
    entries: VecDeque<KernelStat>,
}

impl KernelStatRing {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: VecDeque::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, stat: KernelStat) {
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(stat);
    }

    pub fn snapshot(&self) -> Vec<KernelStat> {
        self.entries.iter().cloned().collect()
    }
}

pub fn global_kernel_stats() -> &'static Mutex<KernelStatRing> {
    static STATS: OnceLock<Mutex<KernelStatRing>> = OnceLock::new();
    STATS.get_or_init(|| Mutex::new(KernelStatRing::new(1024)))
}

pub fn record_cpu_fallback(
    name: &'static str,
    kind: OperationKind,
    n: usize,
    p: usize,
    k: usize,
    nnz: usize,
) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("GAM_GPU_PROFILE").is_ok_and(|v| v == "1")) {
        return;
    }
    if let Ok(mut stats) = global_kernel_stats().lock() {
        stats.push(KernelStat {
            name,
            kind,
            n,
            p,
            k,
            nnz,
            flops_est: estimate_flops(kind, n, p, k, nnz),
            bytes_est: estimate_bytes(n, p, k, nnz),
            cpu_ms: None,
            gpu_ms: None,
            target: "cpu-fallback",
        });
    }
}

pub fn kernel_stats_snapshot() -> Vec<KernelStat> {
    global_kernel_stats()
        .lock()
        .map(|stats| stats.snapshot())
        .unwrap_or_default()
}

pub fn estimate_flops(kind: OperationKind, n: usize, p: usize, k: usize, nnz: usize) -> f64 {
    match kind {
        OperationKind::Gemv | OperationKind::GemvTranspose => 2.0 * n as f64 * p as f64,
        OperationKind::Gemm => 2.0 * n as f64 * p as f64 * k as f64,
        OperationKind::XtDiagX => n as f64 * p as f64 * p as f64,
        OperationKind::XtDiagY => 2.0 * n as f64 * p as f64 * k as f64,
        OperationKind::JointHessian2x2 => n as f64 * (p + k) as f64 * (p + k) as f64,
        OperationKind::SparseXtWx | OperationKind::Pcg => 2.0 * nnz as f64,
        _ => 0.0,
    }
}

pub fn estimate_bytes(n: usize, p: usize, k: usize, nnz: usize) -> usize {
    (n.saturating_mul(p)
        .saturating_add(n.saturating_mul(k))
        .saturating_add(nnz))
    .saturating_mul(8)
}

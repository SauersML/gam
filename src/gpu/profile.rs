use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use crate::gpu::policy::Operation;

/// A single profiled kernel/operation sample.  `gpu_ms` is populated once a
/// concrete device backend executes the operation; CPU fallback samples still
/// carry shape and estimate metadata for phase-0 inventory.
#[derive(Clone, Debug)]
pub struct KernelStat {
    pub name: &'static str,
    pub n: usize,
    pub p: usize,
    pub k: usize,
    pub nnz: Option<usize>,
    pub flops_est: f64,
    pub bytes_est: f64,
    pub cpu_ms: f64,
    pub gpu_ms: Option<f64>,
}

impl KernelStat {
    #[must_use]
    pub fn from_operation(op: Operation, cpu: Duration, gpu: Option<Duration>) -> Self {
        let (n, p, k, nnz, bytes_est) = match op {
            Operation::Gemm { m, n, k, .. } => {
                (m, n, k, None, 8.0 * (m * k + k * n + m * n) as f64)
            }
            Operation::Gemv { rows, cols, .. } => (
                rows,
                cols,
                1,
                None,
                8.0 * (rows * cols + rows + cols) as f64,
            ),
            Operation::XtDiagX { rows, cols, .. } => (
                rows,
                cols,
                cols,
                None,
                8.0 * (rows * cols + rows + cols * cols) as f64,
            ),
            Operation::XtDiagY {
                rows,
                x_cols,
                y_cols,
                ..
            } => (
                rows,
                x_cols,
                y_cols,
                None,
                8.0 * (rows * (x_cols + y_cols) + x_cols * y_cols) as f64,
            ),
            Operation::JointHessian2x2 {
                rows,
                a_cols,
                b_cols,
                ..
            } => (
                rows,
                a_cols,
                b_cols,
                None,
                8.0 * (rows * (a_cols + b_cols + 3) + (a_cols + b_cols).pow(2)) as f64,
            ),
            Operation::DenseSpdSolve { dim, rhs, .. } => {
                (dim, dim, rhs, None, 8.0 * (dim * dim + dim * rhs) as f64)
            }
            Operation::SparseXtDiagX {
                rows, cols, nnz, ..
            } => (
                rows,
                cols,
                0,
                Some(nnz),
                8.0 * nnz as f64 + 4.0 * nnz as f64,
            ),
            Operation::RowKernel { rows, lanes, .. } => {
                (rows, lanes, 0, None, 8.0 * rows as f64 * lanes as f64)
            }
        };
        Self {
            name: op.name(),
            n,
            p,
            k,
            nnz,
            flops_est: op.estimated_flops(),
            bytes_est,
            cpu_ms: cpu.as_secs_f64() * 1_000.0,
            gpu_ms: gpu.map(|d| d.as_secs_f64() * 1_000.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ProfileSnapshot {
    pub samples: Vec<KernelStat>,
}

impl ProfileSnapshot {
    #[must_use]
    pub fn by_kernel_count(&self) -> BTreeMap<&'static str, usize> {
        let mut out = BTreeMap::new();
        for sample in &self.samples {
            *out.entry(sample.name).or_insert(0) += 1;
        }
        out
    }
}

fn stats() -> &'static Mutex<Vec<KernelStat>> {
    static STATS: OnceLock<Mutex<Vec<KernelStat>>> = OnceLock::new();
    STATS.get_or_init(|| Mutex::new(Vec::new()))
}

pub(crate) fn record(stat: KernelStat) {
    if let Ok(mut guard) = stats().lock() {
        guard.push(stat);
    }
}

#[must_use]
pub fn snapshot() -> ProfileSnapshot {
    ProfileSnapshot {
        samples: stats().lock().map(|g| g.clone()).unwrap_or_default(),
    }
}

pub fn clear() {
    if let Ok(mut guard) = stats().lock() {
        guard.clear();
    }
}

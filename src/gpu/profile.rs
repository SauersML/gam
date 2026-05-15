use crate::gpu::policy::GpuOperation;
use std::env;
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
    pub cpu_ms: f64,
    pub gpu_ms: Option<f64>,
}

#[must_use]
pub fn profiling_enabled() -> bool {
    matches!(
        env::var("GAM_GPU_PROFILE").ok().as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

pub fn record_cpu_kernel(op: GpuOperation, elapsed: Duration) {
    if !profiling_enabled() {
        return;
    }
    let (n, p, k, nnz) = dimensions(op);
    let stat = KernelStat {
        name: op.name(),
        n,
        p,
        k,
        nnz,
        flops_est: op.estimated_flops(),
        bytes_est: estimated_bytes(op),
        cpu_ms: elapsed.as_secs_f64() * 1_000.0,
        gpu_ms: None,
    };
    stats()
        .lock()
        .expect("GPU profile mutex poisoned")
        .push(stat);
}

#[must_use]
pub fn snapshot() -> Vec<KernelStat> {
    stats().lock().expect("GPU profile mutex poisoned").clone()
}

fn stats() -> &'static Mutex<Vec<KernelStat>> {
    static STATS: OnceLock<Mutex<Vec<KernelStat>>> = OnceLock::new();
    STATS.get_or_init(|| Mutex::new(Vec::new()))
}

fn dimensions(op: GpuOperation) -> (usize, usize, usize, Option<usize>) {
    match op {
        GpuOperation::Gemm { m, n, k } => (m, n, k, None),
        GpuOperation::Gemv { m, k } => (m, 1, k, None),
        GpuOperation::XtDiagX { rows, cols, .. } => (rows, cols, cols, None),
        GpuOperation::XtDiagY {
            rows,
            x_cols,
            y_cols,
            ..
        } => (rows, x_cols, y_cols, None),
        GpuOperation::JointHessian2x2 {
            rows,
            a_cols,
            b_cols,
            ..
        } => (rows, a_cols + b_cols, a_cols + b_cols, None),
        GpuOperation::Cholesky { cols, rhs, .. } => (cols, cols, rhs, None),
        GpuOperation::SparseXtDiagX {
            rows, cols, nnz, ..
        } => (rows, cols, cols, Some(nnz)),
        GpuOperation::RowKernel {
            rows,
            axes,
            candidates,
            ..
        } => (rows, axes, candidates, None),
    }
}

fn estimated_bytes(op: GpuOperation) -> f64 {
    let f64_bytes = 8.0;
    match op {
        GpuOperation::Gemm { m, n, k } => ((m * k) + (k * n) + (m * n)) as f64 * f64_bytes,
        GpuOperation::Gemv { m, k } => ((m * k) + k + m) as f64 * f64_bytes,
        GpuOperation::XtDiagX { rows, cols, .. } => {
            ((rows * cols) + rows + (cols * cols)) as f64 * f64_bytes
        }
        GpuOperation::XtDiagY {
            rows,
            x_cols,
            y_cols,
            ..
        } => ((rows * x_cols) + rows + (rows * y_cols) + (x_cols * y_cols)) as f64 * f64_bytes,
        GpuOperation::JointHessian2x2 {
            rows,
            a_cols,
            b_cols,
            ..
        } => {
            let c = a_cols + b_cols;
            ((rows * c) + (3 * rows) + (c * c)) as f64 * f64_bytes
        }
        GpuOperation::Cholesky { cols, rhs, .. } => {
            ((cols * cols) + (cols * rhs)) as f64 * f64_bytes
        }
        GpuOperation::SparseXtDiagX { nnz, cols, .. } => (nnz + (cols * cols)) as f64 * f64_bytes,
        GpuOperation::RowKernel {
            rows,
            axes,
            candidates,
            ..
        } => (rows * axes.max(1) * candidates.max(1)) as f64 * f64_bytes,
    }
}

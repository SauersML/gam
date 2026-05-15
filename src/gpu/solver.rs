use ndarray::Array2;

use super::memory::MatrixLocationMut;
use super::policy::{Operation, OperationDecision};
use super::runtime::GpuRuntime;

#[derive(Clone, Debug)]
pub struct DenseFactorInfo {
    pub p: usize,
    pub logdet: Option<f64>,
    pub on_gpu: bool,
}

pub trait DenseSpdSolver {
    fn potrf(&self, matrix: MatrixLocationMut<'_>) -> Result<DenseFactorInfo, String>;
    fn potrs(&self, factor: &DenseFactorInfo, rhs: MatrixLocationMut<'_>) -> Result<(), String>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GpuSolver;

impl GpuSolver {
    #[must_use]
    pub fn should_use_gpu_potrf(p: usize, resident: bool) -> bool {
        GpuRuntime::global().selected_context().is_some_and(|ctx| {
            ctx.target_for(Operation::Potrf { p, resident }) == OperationDecision::Gpu
        })
    }
}

impl DenseSpdSolver for GpuSolver {
    fn potrf(&self, matrix: MatrixLocationMut<'_>) -> Result<DenseFactorInfo, String> {
        match matrix {
            MatrixLocationMut::Host(a) => Ok(DenseFactorInfo {
                p: a.nrows(),
                logdet: cholesky_logdet_reference(a),
                on_gpu: false,
            }),
            MatrixLocationMut::Device(a) => Ok(DenseFactorInfo {
                p: a.rows(),
                logdet: None,
                on_gpu: true,
            }),
        }
    }

    fn potrs(&self, _factor: &DenseFactorInfo, _rhs: MatrixLocationMut<'_>) -> Result<(), String> {
        Err("GPU dense solve backend is unavailable in this CPU fallback build".to_string())
    }
}

fn cholesky_logdet_reference(a: &Array2<f64>) -> Option<f64> {
    if a.nrows() != a.ncols() {
        return None;
    }
    let mut work = a.clone();
    let n = work.nrows();
    let mut logdet = 0.0;
    for j in 0..n {
        let mut diag = work[[j, j]];
        for k in 0..j {
            diag -= work[[j, k]] * work[[j, k]];
        }
        if diag <= 0.0 || !diag.is_finite() {
            return None;
        }
        let l_jj = diag.sqrt();
        work[[j, j]] = l_jj;
        logdet += 2.0 * l_jj.ln();
        for i in j + 1..n {
            let mut value = work[[i, j]];
            for k in 0..j {
                value -= work[[i, k]] * work[[j, k]];
            }
            work[[i, j]] = value / l_jj;
        }
    }
    Some(logdet)
}

use super::cpu_traits::DeviceSolver;
use super::profile::{OperationKind, record_cpu_fallback};
use ndarray::{Array1, Array2};

#[derive(Clone, Debug, Default)]
pub struct CudaSolver;

impl DeviceSolver for CudaSolver {
    fn potrf_solve(&self, h: &Array2<f64>, rhs: &Array2<f64>) -> Option<Array2<f64>> {
        record_cpu_fallback(
            "gpu.solver.potrf_solve",
            OperationKind::Cholesky,
            h.nrows(),
            h.ncols(),
            rhs.ncols(),
            0,
        );
        None
    }

    fn syevd(&self, h: &Array2<f64>) -> Option<(Array1<f64>, Array2<f64>)> {
        record_cpu_fallback(
            "gpu.solver.syevd",
            OperationKind::Syevd,
            h.nrows(),
            h.ncols(),
            0,
            0,
        );
        None
    }
}

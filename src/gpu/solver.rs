use crate::gpu::memory::MatrixLocationMut;

pub trait DenseSpdSolver {
    type Factor;
    type Error;

    fn potrf(&self, matrix: MatrixLocationMut<'_, f64>) -> Result<Self::Factor, Self::Error>;
    fn potrs(
        &self,
        factor: &Self::Factor,
        rhs: MatrixLocationMut<'_, f64>,
    ) -> Result<(), Self::Error>;
}

#[derive(Clone, Debug)]
pub struct GpuSolverUnavailable;

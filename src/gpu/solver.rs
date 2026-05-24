use crate::gpu::memory::DeviceMatrix;

#[derive(Clone, Debug)]
pub struct GpuFactor {
    pub dim: usize,
}

pub trait DenseSpdSolver {
    fn potrf(&self, a: &mut DeviceMatrix<f64>) -> Result<GpuFactor, &'static str>;
    fn potrs(&self, factor: &GpuFactor, rhs: &mut DeviceMatrix<f64>) -> Result<(), &'static str>;
}

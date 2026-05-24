use super::memory::{DeviceMatrix, DeviceVector};
use ndarray::{Array1, Array2};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionTarget {
    Cpu,
    Gpu,
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixLocation {
    Host,
    Device,
    Unified,
}

pub trait DeviceBlas {
    fn gemv(&self, a: &DeviceMatrix, x: &DeviceVector) -> Result<DeviceVector, String>;
    fn atv(&self, a: &DeviceMatrix, x: &DeviceVector) -> Result<DeviceVector, String>;
    fn xt_diag_x_signed(&self, x: &DeviceMatrix, w: &DeviceVector) -> Result<DeviceMatrix, String>;
    fn xt_diag_x_sqrt(
        &self,
        x: &DeviceMatrix,
        sqrt_w: &DeviceVector,
    ) -> Result<DeviceMatrix, String>;
    fn xt_diag_y(
        &self,
        x: &DeviceMatrix,
        w: &DeviceVector,
        y: &DeviceMatrix,
    ) -> Result<DeviceMatrix, String>;
}

pub trait DeviceSolver {
    fn potrf(&self, a: &mut DeviceMatrix) -> Result<(), String>;
    fn potrs(&self, factor: &DeviceMatrix, rhs: &DeviceMatrix) -> Result<DeviceMatrix, String>;
    fn syevd(&self, a: &DeviceMatrix) -> Result<(DeviceVector, DeviceMatrix), String>;
}

pub trait DeviceDesignOperator {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn materialize_chunk_into_device(
        &self,
        start: usize,
        rows: usize,
        out: &mut DeviceMatrix,
    ) -> Result<(), String>;
    fn apply_device(&self, x: &DeviceVector) -> Result<DeviceVector, String>;
}

pub struct CpuFallbackBlas;

impl CpuFallbackBlas {
    #[must_use]
    pub fn gemv_host(a: &Array2<f64>, x: &Array1<f64>) -> Array1<f64> {
        a.dot(x)
    }
}

use ndarray::{Array1, Array2, ArrayViewMut1};

use super::memory::{DeviceCsrMatrix, DeviceMatrix, DeviceVector};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionTarget {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatrixLocation {
    Host,
    Device,
    Unified,
}

pub trait DeviceBlas {
    fn gemv(&self, a: &DeviceMatrix, x: &DeviceVector) -> Result<DeviceVector, String>;
    fn gemv_into(
        &self,
        a: &DeviceMatrix,
        x: &DeviceVector,
        out: &mut DeviceVector,
    ) -> Result<(), String>;
    fn gemm(&self, a: &DeviceMatrix, b: &DeviceMatrix) -> Result<DeviceMatrix, String>;
    fn xt_v(&self, x: &DeviceMatrix, v: &DeviceVector) -> Result<DeviceVector, String>;
    fn xt_diag_x_signed(&self, x: &DeviceMatrix, w: &DeviceVector) -> Result<DeviceMatrix, String>;
    fn xt_diag_x_from_sqrt(
        &self,
        x: &DeviceMatrix,
        sqrt_w: &DeviceVector,
    ) -> Result<DeviceMatrix, String>;
    fn xt_diag_y_signed(
        &self,
        x: &DeviceMatrix,
        w: &DeviceVector,
        y: &DeviceMatrix,
    ) -> Result<DeviceMatrix, String>;
    fn row_scale_signed(&self, x: &DeviceMatrix, w: &DeviceVector) -> Result<DeviceMatrix, String>;
    fn axpy(&self, alpha: f64, x: &DeviceVector, y: &mut DeviceVector) -> Result<(), String>;
    fn dot(&self, x: &DeviceVector, y: &DeviceVector) -> Result<f64, String>;
    fn l2norm(&self, x: &DeviceVector) -> Result<f64, String>;
    fn scale(&self, alpha: f64, x: &mut DeviceVector) -> Result<(), String>;
}

pub trait DeviceSolver {
    fn potrf(&self, a: &mut DeviceMatrix) -> Result<(), String>;
    fn potrs(&self, factor: &DeviceMatrix, rhs: &mut DeviceMatrix) -> Result<(), String>;
    fn syevd(&self, a: &mut DeviceMatrix) -> Result<DeviceVector, String>;
    fn getrf(&self, a: &mut DeviceMatrix) -> Result<DeviceVector, String>;
    fn getrs(
        &self,
        factor: &DeviceMatrix,
        pivots: &DeviceVector,
        rhs: &mut DeviceMatrix,
    ) -> Result<(), String>;
}

pub trait DeviceDesignOperator {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn location(&self) -> MatrixLocation;
    fn materialize_chunk_into_device(
        &self,
        start: usize,
        end: usize,
        out: &mut DeviceMatrix,
    ) -> Result<(), String>;
    fn apply_device(&self, x: &DeviceVector) -> Result<DeviceVector, String>;
    fn apply_transpose_device(&self, x: &DeviceVector) -> Result<DeviceVector, String>;
    fn as_dense_device(&self) -> Option<&DeviceMatrix>;
    fn as_csr_device(&self) -> Option<&DeviceCsrMatrix>;
}

pub trait HostDesignOperatorBridge {
    fn apply_host_into(&self, x: &Array1<f64>, out: ArrayViewMut1<'_, f64>);
    fn materialize_host(&self) -> Option<Array2<f64>>;
}

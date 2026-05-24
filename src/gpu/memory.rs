use super::cpu_traits::MatrixLocation;

#[derive(Clone, Debug)]
pub struct DeviceBuffer<T> {
    pub len: usize,
    pub location: MatrixLocation,
    pub host_shadow: Vec<T>,
}

impl<T: Clone + Default> DeviceBuffer<T> {
    pub fn zeros(len: usize) -> Self {
        Self {
            len,
            location: MatrixLocation::Host,
            host_shadow: vec![T::default(); len],
        }
    }
}

pub type DeviceVector = DeviceBuffer<f64>;

#[derive(Clone, Debug)]
pub struct DeviceMatrix {
    pub rows: usize,
    pub cols: usize,
    pub leading_dim: usize,
    pub location: MatrixLocation,
    pub host_shadow: Vec<f64>,
}

impl DeviceMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            leading_dim: rows,
            location: MatrixLocation::Host,
            host_shadow: vec![0.0; rows.saturating_mul(cols)],
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeviceCsrMatrix {
    pub rows: usize,
    pub cols: usize,
    pub rowptr: Vec<usize>,
    pub colidx: Vec<usize>,
    pub values: Vec<f64>,
    pub location: MatrixLocation,
}

#[derive(Clone, Debug)]
pub struct GpuFitSession {
    pub target: super::cpu_traits::ExecutionTarget,
    pub n: usize,
    pub p: usize,
    pub x_dense: Option<DeviceMatrix>,
    pub x_sparse: Option<DeviceCsrMatrix>,
    pub beta: DeviceVector,
    pub eta: DeviceVector,
    pub mu: DeviceVector,
    pub w: DeviceVector,
    pub z: DeviceVector,
    pub gradient: DeviceVector,
    pub hessian: DeviceMatrix,
}

impl GpuFitSession {
    pub fn host_fallback(n: usize, p: usize) -> Self {
        Self {
            target: super::cpu_traits::ExecutionTarget::Cpu,
            n,
            p,
            x_dense: None,
            x_sparse: None,
            beta: DeviceVector::zeros(p),
            eta: DeviceVector::zeros(n),
            mu: DeviceVector::zeros(n),
            w: DeviceVector::zeros(n),
            z: DeviceVector::zeros(n),
            gradient: DeviceVector::zeros(p),
            hessian: DeviceMatrix::zeros(p, p),
        }
    }
}

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeviceBuffer {
    pub len: usize,
    pub bytes: usize,
    pub backend: String,
}

impl DeviceBuffer {
    #[must_use]
    pub fn placeholder(len: usize, elem_bytes: usize) -> Self {
        Self {
            len,
            bytes: len.saturating_mul(elem_bytes),
            backend: "cpu-placeholder".to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeviceMatrix {
    pub rows: usize,
    pub cols: usize,
    pub buffer: DeviceBuffer,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeviceVector {
    pub len: usize,
    pub buffer: DeviceBuffer,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeviceCsrMatrix {
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
    pub rowptr: DeviceBuffer,
    pub colidx: DeviceBuffer,
    pub values: DeviceBuffer,
}

impl DeviceMatrix {
    #[must_use]
    pub fn from_host_shape(host: &Array2<f64>) -> Self {
        let (rows, cols) = host.dim();
        Self {
            rows,
            cols,
            buffer: DeviceBuffer::placeholder(rows.saturating_mul(cols), 8),
        }
    }
}

impl DeviceVector {
    #[must_use]
    pub fn from_host_shape(host: &Array1<f64>) -> Self {
        Self {
            len: host.len(),
            buffer: DeviceBuffer::placeholder(host.len(), 8),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuFitSession {
    pub persistent_bytes: usize,
    pub iteration_bytes: usize,
    pub lm_attempt_bytes: usize,
    pub dense_design: Option<DeviceMatrix>,
    pub sparse_design: Option<DeviceCsrMatrix>,
}

impl GpuFitSession {
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.persistent_bytes
            .saturating_add(self.iteration_bytes)
            .saturating_add(self.lm_attempt_bytes)
    }
}

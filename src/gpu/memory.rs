use ndarray::{Array1, Array2};

use super::cpu_traits::MatrixLocation;

#[derive(Clone, Debug)]
pub struct DeviceBuffer<T> {
    host_shadow: Vec<T>,
    location: MatrixLocation,
}

impl<T> DeviceBuffer<T> {
    pub fn from_host_shadow(host_shadow: Vec<T>) -> Self {
        Self {
            host_shadow,
            location: MatrixLocation::Host,
        }
    }

    pub fn len(&self) -> usize {
        self.host_shadow.len()
    }

    pub fn is_empty(&self) -> bool {
        self.host_shadow.is_empty()
    }

    pub fn location(&self) -> MatrixLocation {
        self.location
    }

    pub fn host_shadow(&self) -> &[T] {
        &self.host_shadow
    }
}

#[derive(Clone, Debug)]
pub struct DeviceVector {
    pub len: usize,
    pub data: DeviceBuffer<f64>,
}

impl DeviceVector {
    pub fn from_array(array: &Array1<f64>) -> Self {
        Self {
            len: array.len(),
            data: DeviceBuffer::from_host_shadow(array.to_vec()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeviceMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: DeviceBuffer<f64>,
    pub column_major: bool,
}

impl DeviceMatrix {
    pub fn from_array(array: &Array2<f64>) -> Self {
        Self {
            rows: array.nrows(),
            cols: array.ncols(),
            data: DeviceBuffer::from_host_shadow(array.iter().copied().collect()),
            column_major: false,
        }
    }

    pub fn bytes(&self) -> usize {
        self.rows
            .saturating_mul(self.cols)
            .saturating_mul(std::mem::size_of::<f64>())
    }
}

#[derive(Clone, Debug)]
pub struct DeviceCsrMatrix {
    pub rows: usize,
    pub cols: usize,
    pub rowptr: DeviceBuffer<i32>,
    pub colidx: DeviceBuffer<i32>,
    pub values: DeviceBuffer<f64>,
}

impl DeviceCsrMatrix {
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

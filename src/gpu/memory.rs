//! Host-shadow representations of device-resident buffers.
//!
//! These types describe values that *would* live on the device once a CUDA
//! backend is linked in; today they keep their data on the host so traits
//! like [`super::traits::DeviceBlas`] have a concrete value to operate on.
//! Backends are free to replace the storage with an actual device pointer
//! without changing trait signatures.

use ndarray::{Array1, Array2};

/// Where a buffer's authoritative copy currently lives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatrixLocation {
    /// Buffer is host-resident.
    Host,
    /// Buffer is device-resident; the host shadow is stale or empty.
    Device,
    /// Buffer lives in managed/unified memory accessible from both sides.
    Unified,
}

/// Thin host shadow of a device allocation. Generic over the element type
/// so dense `f64` matrices and CSR `i32` index arrays share an interface.
#[derive(Clone, Debug)]
pub struct DeviceBuffer<T> {
    host_shadow: Vec<T>,
    location: MatrixLocation,
}

impl<T> DeviceBuffer<T> {
    /// Construct a host-resident buffer.
    pub fn from_host(host_shadow: Vec<T>) -> Self {
        Self {
            host_shadow,
            location: MatrixLocation::Host,
        }
    }

    /// Number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.host_shadow.len()
    }

    /// True if the buffer has zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.host_shadow.is_empty()
    }

    /// Authoritative location of the data.
    #[inline]
    pub fn location(&self) -> MatrixLocation {
        self.location
    }

    /// Read-only view of the host shadow. May lag when `location()` is
    /// `Device`; backends must sync the shadow before exposing it.
    #[inline]
    pub fn host_shadow(&self) -> &[T] {
        &self.host_shadow
    }
}

/// Host-shadow of a dense vector destined for device residency.
#[derive(Clone, Debug)]
pub struct DeviceVector {
    pub len: usize,
    pub data: DeviceBuffer<f64>,
}

impl DeviceVector {
    pub fn from_array(array: &Array1<f64>) -> Self {
        Self {
            len: array.len(),
            data: DeviceBuffer::from_host(array.to_vec()),
        }
    }
}

/// Host-shadow of a dense matrix destined for device residency.
///
/// `column_major == true` mirrors cuBLAS layout; `false` is row-major and
/// requires a backend-side transpose before invoking column-major APIs.
#[derive(Clone, Debug)]
pub struct DeviceMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: DeviceBuffer<f64>,
    pub column_major: bool,
}

impl DeviceMatrix {
    /// Build a row-major host shadow from an ndarray.
    pub fn from_array(array: &Array2<f64>) -> Self {
        Self {
            rows: array.nrows(),
            cols: array.ncols(),
            data: DeviceBuffer::from_host(array.iter().copied().collect()),
            column_major: false,
        }
    }

    /// Total bytes occupied by the dense storage.
    pub fn bytes(&self) -> usize {
        self.rows
            .saturating_mul(self.cols)
            .saturating_mul(std::mem::size_of::<f64>())
    }
}

/// Host-shadow of a CSR sparse matrix destined for device residency.
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

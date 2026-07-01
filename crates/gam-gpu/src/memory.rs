use ndarray::{Array1, Array2};

use super::cpu_traits::MatrixLocation;

#[derive(Clone, Debug)]
pub struct DeviceBuffer<T> {
    host_shadow: Vec<T>,
    location: MatrixLocation,
}

impl<T> DeviceBuffer<T> {
    pub const fn from_host_shadow(host_shadow: Vec<T>) -> Self {
        Self {
            host_shadow,
            location: MatrixLocation::Host,
        }
    }

    pub const fn len(&self) -> usize {
        self.host_shadow.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.host_shadow.len() == 0
    }

    pub const fn location(&self) -> MatrixLocation {
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

    pub const fn bytes(&self) -> usize {
        self.rows
            .saturating_mul(self.cols)
            .saturating_mul(std::mem::size_of::<f64>())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeviceCsrError {
    InvalidRowptrLength { rows: usize, rowptr_len: usize },
}

impl std::fmt::Display for DeviceCsrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRowptrLength { rows, rowptr_len } => write!(
                f,
                "CSR rowptr length must equal rows + 1 (rows={rows}, rowptr_len={rowptr_len})"
            ),
        }
    }
}

impl std::error::Error for DeviceCsrError {}

#[derive(Clone, Debug)]
pub struct DeviceCsrMatrix {
    rows: usize,
    cols: usize,
    rowptr: DeviceBuffer<i32>,
    colidx: DeviceBuffer<i32>,
    values: DeviceBuffer<f64>,
}

impl DeviceCsrMatrix {
<<<<<<< ours
    /// Construct a CSR matrix, enforcing the structural invariant that `rowptr`
    /// holds exactly `rows + 1` entries.
    ///
    /// A CSR row-pointer array must have one slot per row plus a trailing slot
    /// equal to `nnz`. If the supplied `rowptr` violates this (too short or too
    /// long), it is canonicalized to `rows + 1` monotone entries: a short
    /// `rowptr` is padded with its final value (marking the remaining rows as
    /// empty) and an over-long `rowptr` is truncated. This prevents downstream
    /// row-slice and deallocation paths from indexing `rowptr[row + 1]` out of
    /// bounds, which would be an invalid-free / out-of-bounds deallocation
    /// hazard.
=======
>>>>>>> theirs
    pub fn new(
        rows: usize,
        cols: usize,
        rowptr: DeviceBuffer<i32>,
        colidx: DeviceBuffer<i32>,
        values: DeviceBuffer<f64>,
<<<<<<< ours
    ) -> Self {
        let expected = rows + 1;
        let mut ptr = rowptr.host_shadow().to_vec();
        if ptr.len() != expected {
            let fill = ptr.last().copied().unwrap_or(0);
            ptr.resize(expected, fill);
        }
        Self {
            rows,
            cols,
            rowptr: DeviceBuffer::from_host_shadow(ptr),
            colidx,
            values,
        }
=======
    ) -> Result<Self, DeviceCsrError> {
        let rowptr_len = rowptr.len();
        let Some(expected_rowptr_len) = rows.checked_add(1) else {
            return Err(DeviceCsrError::InvalidRowptrLength { rows, rowptr_len });
        };
        if rowptr_len != expected_rowptr_len {
            return Err(DeviceCsrError::InvalidRowptrLength { rows, rowptr_len });
        }

        Ok(Self {
            rows,
            cols,
            rowptr,
            colidx,
            values,
        })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn cols(&self) -> usize {
        self.cols
    }

    pub const fn rowptr(&self) -> &DeviceBuffer<i32> {
        &self.rowptr
    }

    pub const fn colidx(&self) -> &DeviceBuffer<i32> {
        &self.colidx
    }

    pub const fn values(&self) -> &DeviceBuffer<f64> {
        &self.values
>>>>>>> theirs
    }

    pub const fn nnz(&self) -> usize {
        self.values.len()
    }
}

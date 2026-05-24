use ndarray::Array2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatrixLayout {
    RowMajor,
    ColumnMajor,
}

#[derive(Clone, Debug)]
pub struct DeviceBuffer<T> {
    host_mirror: Vec<T>,
}

impl<T: Clone + Default> DeviceBuffer<T> {
    #[must_use]
    pub fn zeros(len: usize) -> Self {
        Self {
            host_mirror: vec![T::default(); len],
        }
    }
}

impl<T> DeviceBuffer<T> {
    #[must_use]
    pub fn from_vec(values: Vec<T>) -> Self {
        Self {
            host_mirror: values,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.host_mirror.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.host_mirror.is_empty()
    }

    #[must_use]
    pub fn as_host_slice(&self) -> &[T] {
        &self.host_mirror
    }
}

#[derive(Clone, Debug)]
pub struct DeviceMatrix<T> {
    buffer: DeviceBuffer<T>,
    rows: usize,
    cols: usize,
    ld: usize,
    layout: MatrixLayout,
}

impl DeviceMatrix<f64> {
    #[must_use]
    pub fn from_host_row_major(matrix: &Array2<f64>) -> Self {
        let rows = matrix.nrows();
        let cols = matrix.ncols();
        let values = matrix.iter().copied().collect();
        Self {
            buffer: DeviceBuffer::from_vec(values),
            rows,
            cols,
            ld: cols,
            layout: MatrixLayout::RowMajor,
        }
    }
}

impl<T> DeviceMatrix<T> {
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    #[must_use]
    pub const fn leading_dimension(&self) -> usize {
        self.ld
    }

    #[must_use]
    pub const fn layout(&self) -> MatrixLayout {
        self.layout
    }

    #[must_use]
    pub fn buffer(&self) -> &DeviceBuffer<T> {
        &self.buffer
    }
}

pub enum MatrixLocation<'a> {
    Host(&'a Array2<f64>),
    Device(&'a DeviceMatrix<f64>),
}

pub enum MatrixLocationMut<'a> {
    Host(&'a mut Array2<f64>),
    Device(&'a mut DeviceMatrix<f64>),
}

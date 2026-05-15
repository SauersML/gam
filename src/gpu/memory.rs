use std::marker::PhantomData;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatrixLayout {
    RowMajor,
    ColumnMajor,
}

/// Host-side descriptor for a future device buffer allocation.
#[derive(Clone, Debug)]
pub struct DeviceBuffer<T> {
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> DeviceBuffer<T> {
    #[must_use]
    pub fn new_unallocated(len: usize) -> Self {
        Self {
            len,
            _marker: PhantomData,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[derive(Clone, Debug)]
pub struct DeviceMatrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub ld: usize,
    pub layout: MatrixLayout,
    pub buffer: DeviceBuffer<T>,
}

impl<T> DeviceMatrix<T> {
    #[must_use]
    pub fn descriptor(rows: usize, cols: usize, layout: MatrixLayout) -> Self {
        let ld = match layout {
            MatrixLayout::RowMajor => cols,
            MatrixLayout::ColumnMajor => rows,
        };
        Self {
            rows,
            cols,
            ld,
            layout,
            buffer: DeviceBuffer::new_unallocated(rows * cols),
        }
    }
}

pub enum MatrixLocation<'a, T> {
    Host(&'a ndarray::Array2<T>),
    Device(&'a DeviceMatrix<T>),
}

pub enum MatrixLocationMut<'a, T> {
    Host(&'a mut ndarray::Array2<T>),
    Device(&'a mut DeviceMatrix<T>),
}

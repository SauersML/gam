use std::marker::PhantomData;

/// Matrix layout used by host/device interop.  GPU kernels prefer explicit
/// leading dimensions rather than relying on ndarray stride inference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatrixLayout {
    RowMajor,
    ColumnMajor,
}

/// Logical device buffer handle.  CPU-only builds cannot own actual device
/// memory, but the type lets resident-state plumbing and policies compile in
/// all configurations.
#[derive(Clone, Debug)]
pub struct DeviceBuffer<T> {
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> DeviceBuffer<T> {
    #[must_use]
    pub fn unavailable(len: usize) -> Self {
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
    pub buffer: DeviceBuffer<T>,
    pub rows: usize,
    pub cols: usize,
    pub ld: usize,
    pub layout: MatrixLayout,
}

impl<T> DeviceMatrix<T> {
    #[must_use]
    pub fn unavailable(rows: usize, cols: usize, layout: MatrixLayout) -> Self {
        let ld = match layout {
            MatrixLayout::RowMajor => cols,
            MatrixLayout::ColumnMajor => rows,
        };
        Self {
            buffer: DeviceBuffer::unavailable(rows.saturating_mul(cols)),
            rows,
            cols,
            ld,
            layout,
        }
    }
}

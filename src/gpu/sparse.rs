use crate::gpu::memory::DeviceBuffer;

#[derive(Clone, Debug)]
pub struct DeviceCsrMatrix<T> {
    pub row_offsets: DeviceBuffer<i32>,
    pub col_indices: DeviceBuffer<i32>,
    pub values: DeviceBuffer<T>,
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
}

impl<T> DeviceCsrMatrix<T> {
    #[must_use]
    pub fn descriptor(rows: usize, cols: usize, nnz: usize) -> Self {
        Self {
            row_offsets: DeviceBuffer::new_unallocated(rows + 1),
            col_indices: DeviceBuffer::new_unallocated(nnz),
            values: DeviceBuffer::new_unallocated(nnz),
            rows,
            cols,
            nnz,
        }
    }
}

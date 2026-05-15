use crate::gpu::memory::DeviceMatrix;
use crate::gpu::stream::GpuStream;

pub trait DeviceDesignOperator {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;

    fn materialize_chunk_device(
        &self,
        row_start: usize,
        row_count: usize,
        out: &mut DeviceMatrix<f64>,
        stream: &GpuStream,
    ) -> Result<(), String>;
}

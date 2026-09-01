use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use std::sync::OnceLock;

pub trait GpuGemmDispatch: Send + Sync {
    fn try_fast_xt_diag_x(
        &self,
        x: ArrayView2<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> Option<Array2<f64>>;

    /// Number of usable GPU devices in the runtime pool (`0` when no GPU
    /// runtime is available). Geometry's multi-GPU row-tiling only engages when
    /// this exceeds `1`.
    fn device_count(&self) -> usize;

}

static GPU_DISPATCH: OnceLock<Box<dyn GpuGemmDispatch>> = OnceLock::new();

pub fn register_gpu_dispatch(d: Box<dyn GpuGemmDispatch>) {
    drop(GPU_DISPATCH.set(d));
}

pub fn gpu_dispatch() -> Option<&'static dyn GpuGemmDispatch> {
    GPU_DISPATCH.get().map(|b| b.as_ref())
}

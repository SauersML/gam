use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use std::sync::OnceLock;

pub trait GpuGemmDispatch: Send + Sync {
    fn try_fast_atb(&self, a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Option<Array2<f64>>;
    fn try_fast_ab(&self, a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Option<Array2<f64>>;
    fn try_fast_av(&self, a: ArrayView2<'_, f64>, v: ArrayView1<'_, f64>) -> Option<Array1<f64>>;
    fn try_fast_atv(&self, a: ArrayView2<'_, f64>, v: ArrayView1<'_, f64>) -> Option<Array1<f64>>;
    fn try_fast_xt_diag_x(
        &self,
        x: ArrayView2<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> Option<Array2<f64>>;
    fn try_fast_xt_diag_y(
        &self,
        x: ArrayView2<'_, f64>,
        w: ArrayView1<'_, f64>,
        y: ArrayView2<'_, f64>,
    ) -> Option<Array2<f64>>;
    fn try_fast_joint_hessian_2x2(
        &self,
        x_a: ArrayView2<'_, f64>,
        x_b: ArrayView2<'_, f64>,
        w_aa: ArrayView1<'_, f64>,
        w_ab: ArrayView1<'_, f64>,
        w_bb: ArrayView1<'_, f64>,
    ) -> Option<Array2<f64>>;

    /// Number of usable GPU devices in the runtime pool (`0` when no GPU
    /// runtime is available). Geometry's multi-GPU row-tiling only engages when
    /// this exceeds `1`.
    fn device_count(&self) -> usize;

    /// Broadcast-`B` strided-batched GEMM: each `tiles × rows × k` slab of `a3`
    /// is multiplied by the shared `k × n` `b`, yielding a `tiles × rows × n`
    /// batch. Returns `None` when the workload is below the multi-GPU floor or
    /// the runtime declines, so the caller falls back to the single-device GEMM.
    fn try_fast_ab_broadcast_b_batched(
        &self,
        a3: ArrayView3<'_, f64>,
        b: ArrayView2<'_, f64>,
    ) -> Option<Array3<f64>>;
}

static GPU_DISPATCH: OnceLock<Box<dyn GpuGemmDispatch>> = OnceLock::new();

pub fn register_gpu_dispatch(d: Box<dyn GpuGemmDispatch>) {
    drop(GPU_DISPATCH.set(d));
}

pub fn gpu_dispatch() -> Option<&'static dyn GpuGemmDispatch> {
    GPU_DISPATCH.get().map(|b| b.as_ref())
}

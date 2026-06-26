use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
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
}

static GPU_DISPATCH: OnceLock<Box<dyn GpuGemmDispatch>> = OnceLock::new();

pub fn register_gpu_dispatch(d: Box<dyn GpuGemmDispatch>) {
    drop(GPU_DISPATCH.set(d));
}

pub fn gpu_dispatch() -> Option<&'static dyn GpuGemmDispatch> {
    GPU_DISPATCH.get().map(|b| b.as_ref())
}

//! CPU-only GPU dispatch stubs. All `try_fast_*` return Cpu / None until a
//! real device backend is compiled in.

#[derive(Clone, Debug)]
pub enum GpuDispatch {
    /// Caller should fall back to the CPU implementation.
    Cpu,
    /// A GPU dispatch executed; the result is carried by the variant.
    Executed(ndarray::Array2<f64>),
}

#[inline]
pub fn try_fast_atb<A, B>(a: A, b: B) -> GpuDispatch {
    drop((a, b));
    GpuDispatch::Cpu
}
#[inline]
pub fn try_fast_ab<A, B>(a: A, b: B) -> GpuDispatch {
    drop((a, b));
    GpuDispatch::Cpu
}
#[inline]
pub fn try_fast_atv<A, V>(a: A, v: V) -> GpuDispatch {
    drop((a, v));
    GpuDispatch::Cpu
}
#[inline]
pub fn try_fast_xt_diag_x<X, W>(x: X, w: W) -> Option<ndarray::Array2<f64>> {
    drop((x, w));
    None
}
#[inline]
pub fn try_fast_xt_diag_y<X, W, Y>(x: X, w: W, y: Y) -> Option<ndarray::Array2<f64>> {
    drop((x, w, y));
    None
}
#[inline]
pub fn should_dispatch_xt_diag_x(n: usize, p: usize) -> bool {
    drop((n, p));
    false
}
#[inline]
pub fn should_dispatch_xt_diag_y(n: usize, px: usize, q: usize) -> bool {
    drop((n, px, q));
    false
}
#[inline]
pub fn should_dispatch_joint_hessian(n: usize, pa: usize, pb: usize) -> bool {
    drop((n, pa, pb));
    false
}

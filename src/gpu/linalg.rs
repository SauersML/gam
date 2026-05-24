//! CPU-only GPU dispatch stubs. All `try_fast_*` return Cpu / None until a
//! real device backend is compiled in.

#[derive(Clone, Copy, Debug)]
pub enum GpuDispatch {
    /// Caller should fall back to the CPU implementation.
    Cpu,
    /// A GPU dispatch executed; the result is carried by the variant.
    Executed(ndarray::Array2<f64>),
}

#[inline]
pub fn try_fast_atb<A, B>(_a: A, _b: B) -> GpuDispatch { GpuDispatch::Cpu }
#[inline]
pub fn try_fast_ab<A, B>(_a: A, _b: B) -> GpuDispatch { GpuDispatch::Cpu }
#[inline]
pub fn try_fast_atv<A, V>(_a: A, _v: V) -> GpuDispatch { GpuDispatch::Cpu }
#[inline]
pub fn try_fast_xt_diag_x<X, W>(_x: X, _w: W) -> Option<ndarray::Array2<f64>> { None }
#[inline]
pub fn try_fast_xt_diag_y<X, W, Y>(_x: X, _w: W, _y: Y) -> Option<ndarray::Array2<f64>> { None }
#[inline]
pub fn should_dispatch_xt_diag_x(_n: usize, _p: usize) -> bool { false }
#[inline]
pub fn should_dispatch_xt_diag_y(_n: usize, _px: usize, _q: usize) -> bool { false }
#[inline]
pub fn should_dispatch_joint_hessian(_n: usize, _pa: usize, _pb: usize) -> bool { false }

use std::time::Instant;

use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};

use crate::gpu::policy::Operation;
use crate::gpu::profile::{self, KernelStat};
use crate::gpu::runtime::global_runtime;

/// Result of an attempted GPU dispatch.  `CpuFallback` means the operation was
/// wired through policy/profiling but intentionally left to the CPU backend.
#[derive(Clone, Debug)]
pub enum GpuDispatch<T> {
    Executed(T),
    CpuFallback,
}

#[inline]
pub fn profile_cpu<T>(op: Operation, f: impl FnOnce() -> T) -> T {
    if !global_runtime().profile_enabled() {
        return f();
    }
    let start = Instant::now();
    let out = f();
    profile::record(KernelStat::from_operation(op, start.elapsed(), None));
    out
}

#[inline]
pub fn try_fast_ab<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> GpuDispatch<Array2<f64>> {
    let (m, k) = a.dim();
    let (_, n) = b.dim();
    let op = Operation::Gemm {
        m,
        n,
        k,
        resident: false,
    };
    if !global_runtime().should_try_gpu(op) {
        return GpuDispatch::CpuFallback;
    }
    GpuDispatch::CpuFallback
}

#[inline]
pub fn try_fast_av<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
) -> GpuDispatch<Array1<f64>> {
    let (rows, cols) = a.dim();
    debug_assert_eq!(cols, v.len());
    let op = Operation::Gemv {
        rows,
        cols,
        transpose: false,
        resident: false,
    };
    if !global_runtime().should_try_gpu(op) {
        return GpuDispatch::CpuFallback;
    }
    GpuDispatch::CpuFallback
}

#[inline]
pub fn try_fast_atv<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
) -> GpuDispatch<Array1<f64>> {
    let (rows, cols) = a.dim();
    debug_assert_eq!(rows, v.len());
    let op = Operation::Gemv {
        rows,
        cols,
        transpose: true,
        resident: false,
    };
    if !global_runtime().should_try_gpu(op) {
        return GpuDispatch::CpuFallback;
    }
    GpuDispatch::CpuFallback
}

#[inline]
pub fn try_fast_atb<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> GpuDispatch<Array2<f64>> {
    let (rows, cols) = a.dim();
    let (_, rhs) = b.dim();
    let op = Operation::Gemm {
        m: cols,
        n: rhs,
        k: rows,
        resident: false,
    };
    if !global_runtime().should_try_gpu(op) {
        return GpuDispatch::CpuFallback;
    }
    GpuDispatch::CpuFallback
}

#[inline]
pub fn try_fast_xt_diag_x<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
) -> GpuDispatch<Array2<f64>> {
    let (rows, cols) = x.dim();
    debug_assert_eq!(rows, w.len());
    let op = Operation::XtDiagX {
        rows,
        cols,
        resident: false,
    };
    if !global_runtime().should_try_gpu(op) {
        return GpuDispatch::CpuFallback;
    }
    GpuDispatch::CpuFallback
}

#[inline]
pub fn try_fast_xt_diag_y<S1: Data<Elem = f64>, S2: Data<Elem = f64>, S3: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
    y: &ArrayBase<S3, Ix2>,
) -> GpuDispatch<Array2<f64>> {
    let (rows, x_cols) = x.dim();
    let (_, y_cols) = y.dim();
    debug_assert_eq!(rows, w.len());
    let op = Operation::XtDiagY {
        rows,
        x_cols,
        y_cols,
        resident: false,
    };
    if !global_runtime().should_try_gpu(op) {
        return GpuDispatch::CpuFallback;
    }
    GpuDispatch::CpuFallback
}

#[inline]
pub fn try_fast_joint_hessian_2x2<
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: Data<Elem = f64>,
    S4: Data<Elem = f64>,
    S5: Data<Elem = f64>,
>(
    x_a: &ArrayBase<S1, Ix2>,
    x_b: &ArrayBase<S2, Ix2>,
    w_aa: &ArrayBase<S3, Ix1>,
    w_ab: &ArrayBase<S4, Ix1>,
    w_bb: &ArrayBase<S5, Ix1>,
) -> GpuDispatch<Array2<f64>> {
    let rows = x_a.nrows();
    debug_assert_eq!(rows, x_b.nrows());
    debug_assert_eq!(rows, w_aa.len());
    debug_assert_eq!(rows, w_ab.len());
    debug_assert_eq!(rows, w_bb.len());
    let op = Operation::JointHessian2x2 {
        rows,
        a_cols: x_a.ncols(),
        b_cols: x_b.ncols(),
        resident: false,
    };
    if !global_runtime().should_try_gpu(op) {
        return GpuDispatch::CpuFallback;
    }
    GpuDispatch::CpuFallback
}

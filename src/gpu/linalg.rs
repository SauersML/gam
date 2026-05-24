use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};

use super::policy::{Operation, OperationDecision};
use super::runtime::GpuRuntime;

#[must_use]
pub fn should_dispatch_gemv<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
    transposed: bool,
) -> bool {
    let (m, k) = a.dim();
    let expected = if transposed { m } else { k };
    debug_assert_eq!(v.len(), expected);
    runtime_decision(Operation::Gemv {
        m,
        k,
        transposed,
        resident: false,
    }) == OperationDecision::Gpu
}

#[must_use]
pub fn should_dispatch_gemm<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> bool {
    let (m, k) = a.dim();
    let (_, n) = b.dim();
    runtime_decision(Operation::Gemm {
        m,
        n,
        k,
        resident: false,
    }) == OperationDecision::Gpu
}

#[must_use]
pub fn should_dispatch_xt_diag_x(rows: usize, cols: usize) -> bool {
    runtime_decision(Operation::XtDiagX {
        rows,
        cols,
        resident: false,
    }) == OperationDecision::Gpu
}

#[must_use]
pub fn should_dispatch_xt_diag_y(rows: usize, x_cols: usize, y_cols: usize) -> bool {
    runtime_decision(Operation::XtDiagY {
        rows,
        x_cols,
        y_cols,
        resident: false,
    }) == OperationDecision::Gpu
}

#[must_use]
pub fn should_dispatch_joint_hessian(rows: usize, a_cols: usize, b_cols: usize) -> bool {
    runtime_decision(Operation::JointHessian2x2 {
        rows,
        a_cols,
        b_cols,
        resident: false,
    }) == OperationDecision::Gpu
}

/// Placeholder for the CUDA GEMV implementation. It currently returns `None`
/// so callers use the exact CPU implementation unless a future CUDA kernel is
/// compiled in and passes validation.
#[must_use]
pub fn try_fast_av<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    _a: &ArrayBase<S1, Ix2>,
    _v: &ArrayBase<S2, Ix1>,
) -> Option<Array1<f64>> {
    None
}

#[must_use]
pub fn try_fast_atv<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    _a: &ArrayBase<S1, Ix2>,
    _v: &ArrayBase<S2, Ix1>,
) -> Option<Array1<f64>> {
    None
}

#[must_use]
pub fn try_fast_ab<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    _a: &ArrayBase<S1, Ix2>,
    _b: &ArrayBase<S2, Ix2>,
) -> Option<Array2<f64>> {
    None
}

#[must_use]
pub fn try_fast_xt_diag_x<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    _x: &ArrayBase<S1, Ix2>,
    _w: &ArrayBase<S2, Ix1>,
) -> Option<Array2<f64>> {
    None
}

#[must_use]
pub fn try_fast_xt_diag_y<S1: Data<Elem = f64>, S2: Data<Elem = f64>, S3: Data<Elem = f64>>(
    _x: &ArrayBase<S1, Ix2>,
    _w: &ArrayBase<S2, Ix1>,
    _y: &ArrayBase<S3, Ix2>,
) -> Option<Array2<f64>> {
    None
}

fn runtime_decision(op: Operation) -> OperationDecision {
    let runtime = GpuRuntime::global();
    runtime
        .selected_context()
        .map_or(OperationDecision::Cpu, |ctx| ctx.target_for(op))
}

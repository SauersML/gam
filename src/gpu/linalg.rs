use super::policy::Operation;
use super::profile::{KernelStat, record_dispatch_candidate, record_gpu_fallback};
use super::runtime::GpuRuntime;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};

#[inline]
pub fn try_fast_atb<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Option<Array2<f64>> {
    let (rows, x_cols) = a.dim();
    let y_cols = b.ncols();
    let flops = 2_u128 * rows as u128 * x_cols as u128 * y_cols as u128;
    maybe_dispatch(
        Operation::Gemm {
            m: x_cols,
            n: y_cols,
            k: rows,
            resident: false,
        },
        "fast_atb",
        rows,
        x_cols,
        y_cols,
        flops,
    )?;
    None
}

#[inline]
pub fn try_fast_ab<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Option<Array2<f64>> {
    let (m, k) = a.dim();
    let n = b.ncols();
    let flops = 2_u128 * m as u128 * n as u128 * k as u128;
    maybe_dispatch(
        Operation::Gemm {
            m,
            n,
            k,
            resident: false,
        },
        "fast_ab",
        m,
        n,
        k,
        flops,
    )?;
    None
}

#[inline]
pub fn try_fast_av<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    _v: &ArrayBase<S2, Ix1>,
) -> Option<Array1<f64>> {
    let (rows, cols) = a.dim();
    let flops = 2_u128 * rows as u128 * cols as u128;
    maybe_dispatch(
        Operation::Gemv {
            rows,
            cols,
            transpose: false,
            resident: false,
        },
        "fast_av",
        rows,
        cols,
        1,
        flops,
    )?;
    None
}

#[inline]
pub fn try_fast_atv<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    _v: &ArrayBase<S2, Ix1>,
) -> Option<Array1<f64>> {
    let (rows, cols) = a.dim();
    let flops = 2_u128 * rows as u128 * cols as u128;
    maybe_dispatch(
        Operation::Gemv {
            rows,
            cols,
            transpose: true,
            resident: false,
        },
        "fast_atv",
        rows,
        cols,
        1,
        flops,
    )?;
    None
}

#[inline]
pub fn try_fast_xt_diag_x<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    _w: &ArrayBase<S2, Ix1>,
) -> Option<Array2<f64>> {
    let (rows, cols) = x.dim();
    let flops = 2_u128 * rows as u128 * cols as u128 * cols as u128;
    maybe_dispatch(
        Operation::XtDiagX {
            rows,
            cols,
            resident: false,
        },
        "fast_xt_diag_x",
        rows,
        cols,
        cols,
        flops,
    )?;
    None
}

#[inline]
pub fn try_fast_xt_diag_y<S1: Data<Elem = f64>, S2: Data<Elem = f64>, S3: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    _w: &ArrayBase<S2, Ix1>,
    y: &ArrayBase<S3, Ix2>,
) -> Option<Array2<f64>> {
    let rows = x.nrows();
    let x_cols = x.ncols();
    let y_cols = y.ncols();
    let flops = 2_u128 * rows as u128 * x_cols as u128 * y_cols as u128;
    maybe_dispatch(
        Operation::XtDiagY {
            rows,
            x_cols,
            y_cols,
            resident: false,
        },
        "fast_xt_diag_y",
        rows,
        x_cols,
        y_cols,
        flops,
    )?;
    None
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
    _w_aa: &ArrayBase<S3, Ix1>,
    _w_ab: &ArrayBase<S4, Ix1>,
    _w_bb: &ArrayBase<S5, Ix1>,
) -> Option<Array2<f64>> {
    let rows = x_a.nrows();
    let cols_a = x_a.ncols();
    let cols_b = x_b.ncols();
    let p = cols_a.saturating_add(cols_b);
    let flops = 2_u128 * rows as u128 * p as u128 * p as u128;
    maybe_dispatch(
        Operation::JointHessian2x2 {
            rows,
            cols_a,
            cols_b,
            resident: false,
        },
        "fast_joint_hessian_2x2",
        rows,
        p,
        p,
        flops,
    )?;
    None
}

fn maybe_dispatch(
    operation: Operation,
    name: &'static str,
    n: usize,
    p: usize,
    k: usize,
    flops: u128,
) -> Option<()> {
    let runtime = GpuRuntime::get();
    if runtime
        .policy
        .should_use_gpu(operation, runtime.cuda_available())
    {
        record_dispatch_candidate(KernelStat {
            name,
            n,
            p,
            k,
            nnz: None,
            flops_est: flops,
            bytes_est: 8_u128 * n as u128 * p.max(1) as u128,
            cpu_ms: None,
            gpu_ms: None,
        });
        record_gpu_fallback();
        log::debug!(
            "[GAM GPU] {name} selected by policy but device BLAS kernels are not linked in this build; using CPU fallback"
        );
        Some(())
    } else {
        None
    }
}

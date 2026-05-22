//! cuBLAS routing for large f64 dense kernels, built on `cudarc` 0.19's
//! safe wrappers (`CudaContext` / `CudaStream` / `CudaBlas` / `Gemm` /
//! `Gemv`). The two ops cudarc doesn't safe-wrap (`cublasDdgmm`,
//! `cublasDtrsm_v2`) fall through to `cudarc::cublas::sys` directly.
//!
//! Public surface (`try_fast_*`) and the multi-device striping structure
//! are unchanged; only the FFI underneath has moved off the hand-rolled
//! libloading bindings.

use cudarc::cublas::sys::{
    cublasDiagType_t, cublasFillMode_t, cublasOperation_t, cublasSideMode_t, cublasStatus_t,
};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, Gemv, GemvConfig, StridedBatchedConfig};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use ndarray::{Array1, Array2, Array3, ArrayBase, ArrayView2, ArrayView3, Data, Ix1, Ix2, s};
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock};

use super::device::GpuDeviceInfo;
use super::diagnostics;
use super::driver::{from_col_major, to_col_major, to_i32};
use super::runtime::{GpuRuntime, cuda_context_for};

#[inline]
pub fn try_fast_ab<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Option<Array2<f64>> {
    let (m, k) = a.dim();
    let (k_b, n) = b.dim();
    debug_assert_eq!(k, k_b, "A and B must have compatible inner dimensions");
    if !route_gemm(m, n, k) {
        diagnostics::log_policy_cpu(
            "gemm",
            format!("m={m} n={n} k={k} trans_a=false trans_b=false"),
            diagnostics::dispatch_decline_reason(format!(
                "below cuBLAS policy threshold gemm_flops>={}",
                GpuRuntime::global().policy().gemm_min_flops
            )),
        );
        return None;
    }
    let start = std::time::Instant::now();
    match try_multi_ab(a, b).or_else(|| {
        with_runtime(|runtime| runtime.gemm(a, b, false, false))
            .map(|(out, device)| (out, vec![device]))
    }) {
        Some((out, devices)) => {
            diagnostics::log_gpu_success_multi(
                "gemm",
                "cuBLAS",
                &devices,
                format!("m={m} n={n} k={k} trans_a=false trans_b=false"),
                diagnostics::gemm_flops(m, n, k),
                diagnostics::bytes_for_f64(a.len().saturating_add(b.len())),
                diagnostics::bytes_for_f64(m.saturating_mul(n)),
                start.elapsed().as_secs_f64(),
            );
            Some(out)
        }
        None => {
            diagnostics::log_runtime_cpu(
                "gemm",
                "cuBLAS",
                format!("m={m} n={n} k={k} trans_a=false trans_b=false"),
            );
            None
        }
    }
}

/// Strided batched dense matrix multiply: `C[b] = A[b] · B[b]` (or transposed
/// variants) for `b = 0..batch`, all in a single cuBLAS call.
#[inline]
pub fn try_fast_ab_strided_batched(
    a: ArrayView3<'_, f64>,
    b: ArrayView3<'_, f64>,
) -> Option<Array3<f64>> {
    try_fast_gemm_strided_batched(a, b, false, false)
}

/// Strided batched `Aᵀ · B`.
#[inline]
pub fn try_fast_atb_strided_batched(
    a: ArrayView3<'_, f64>,
    b: ArrayView3<'_, f64>,
) -> Option<Array3<f64>> {
    try_fast_gemm_strided_batched(a, b, true, false)
}

/// Strided batched `A · Bᵀ`.
#[inline]
pub fn try_fast_abt_strided_batched(
    a: ArrayView3<'_, f64>,
    b: ArrayView3<'_, f64>,
) -> Option<Array3<f64>> {
    try_fast_gemm_strided_batched(a, b, false, true)
}

/// Strided batched `A[b] · B` with B broadcast (same matrix for every batch
/// element). Uses `strideB = 0` so B is uploaded once.
#[inline]
pub fn try_fast_ab_broadcast_b_batched(
    a: ArrayView3<'_, f64>,
    b: ArrayView2<'_, f64>,
) -> Option<Array3<f64>> {
    let (batch, a_rows, a_cols) = a.dim();
    let (b_rows, b_cols) = b.dim();
    if batch == 0 {
        return None;
    }
    let m = a_rows;
    let k = a_cols;
    let n = b_cols;
    if k != b_rows || m == 0 || n == 0 || k == 0 {
        return None;
    }
    if !route_gemm_batched(m, n, k, batch) {
        diagnostics::log_policy_cpu(
            "gemm_broadcast_b_strided_batched",
            format!("batch={batch} m={m} n={n} k={k}"),
            diagnostics::dispatch_decline_reason(format!(
                "below cuBLAS policy threshold gemm_flops>={}",
                GpuRuntime::global().policy().gemm_min_flops
            )),
        );
        return None;
    }
    let start = std::time::Instant::now();
    match try_multi_gemm_broadcast_b_strided_batched(a, b, false, false).or_else(|| {
        with_runtime(|runtime| runtime.gemm_broadcast_b_strided_batched(a, b, false, false))
            .map(|(out, device)| (out, vec![device]))
    }) {
        Some((out, devices)) => {
            diagnostics::log_gpu_success_multi(
                "gemm_broadcast_b_strided_batched",
                "cuBLAS",
                &devices,
                format!("batch={batch} m={m} n={n} k={k}"),
                diagnostics::gemm_flops(m, n, k).saturating_mul(batch as u64),
                diagnostics::bytes_for_f64(a.len().saturating_add(b.len())),
                diagnostics::bytes_for_f64(batch.saturating_mul(m).saturating_mul(n)),
                start.elapsed().as_secs_f64(),
            );
            Some(out)
        }
        None => {
            diagnostics::log_runtime_cpu(
                "gemm_broadcast_b_strided_batched",
                "cuBLAS",
                format!("batch={batch} m={m} n={n} k={k}"),
            );
            None
        }
    }
}

/// Strided batched `A · B[b]ᵀ` with A broadcast.
#[inline]
pub fn try_fast_a_broadcast_bt_batched(
    a: ArrayView2<'_, f64>,
    b: ArrayView3<'_, f64>,
) -> Option<Array3<f64>> {
    let (a_rows, a_cols) = a.dim();
    let (batch, b_rows, b_cols) = b.dim();
    if batch == 0 {
        return None;
    }
    let m = a_rows;
    let k = a_cols;
    let n = b_rows;
    if k != b_cols || m == 0 || n == 0 || k == 0 {
        return None;
    }
    if !route_gemm_batched(m, n, k, batch) {
        diagnostics::log_policy_cpu(
            "gemm_broadcast_a_strided_batched",
            format!("batch={batch} m={m} n={n} k={k}"),
            diagnostics::dispatch_decline_reason(format!(
                "below cuBLAS policy threshold gemm_flops>={}",
                GpuRuntime::global().policy().gemm_min_flops
            )),
        );
        return None;
    }
    let start = std::time::Instant::now();
    match try_multi_broadcast_a_gemm_strided_batched(a, b, false, true).or_else(|| {
        with_runtime(|runtime| runtime.broadcast_a_gemm_strided_batched(a, b, false, true))
            .map(|(out, device)| (out, vec![device]))
    }) {
        Some((out, devices)) => {
            diagnostics::log_gpu_success_multi(
                "gemm_broadcast_a_strided_batched",
                "cuBLAS",
                &devices,
                format!("batch={batch} m={m} n={n} k={k}"),
                diagnostics::gemm_flops(m, n, k).saturating_mul(batch as u64),
                diagnostics::bytes_for_f64(a.len().saturating_add(b.len())),
                diagnostics::bytes_for_f64(batch.saturating_mul(m).saturating_mul(n)),
                start.elapsed().as_secs_f64(),
            );
            Some(out)
        }
        None => {
            diagnostics::log_runtime_cpu(
                "gemm_broadcast_a_strided_batched",
                "cuBLAS",
                format!("batch={batch} m={m} n={n} k={k}"),
            );
            None
        }
    }
}

#[inline]
fn try_fast_gemm_strided_batched(
    a: ArrayView3<'_, f64>,
    b: ArrayView3<'_, f64>,
    transpose_a: bool,
    transpose_b: bool,
) -> Option<Array3<f64>> {
    let (batch_a, a_rows, a_cols) = a.dim();
    let (batch_b, b_rows, b_cols) = b.dim();
    if batch_a == 0 || batch_a != batch_b {
        return None;
    }
    let m = if transpose_a { a_cols } else { a_rows };
    let k = if transpose_a { a_rows } else { a_cols };
    let b_k = if transpose_b { b_cols } else { b_rows };
    let n = if transpose_b { b_rows } else { b_cols };
    if k != b_k || m == 0 || n == 0 || k == 0 {
        return None;
    }
    if !route_gemm_batched(m, n, k, batch_a) {
        diagnostics::log_policy_cpu(
            "gemm_strided_batched",
            format!(
                "batch={batch_a} m={m} n={n} k={k} trans_a={transpose_a} trans_b={transpose_b}"
            ),
            diagnostics::dispatch_decline_reason(format!(
                "below cuBLAS policy threshold gemm_flops>={}",
                GpuRuntime::global().policy().gemm_min_flops
            )),
        );
        return None;
    }
    let start = std::time::Instant::now();
    match try_multi_gemm_strided_batched(a, b, transpose_a, transpose_b).or_else(|| {
        with_runtime(|runtime| runtime.gemm_strided_batched(a, b, transpose_a, transpose_b))
            .map(|(out, device)| (out, vec![device]))
    }) {
        Some((out, devices)) => {
            diagnostics::log_gpu_success_multi(
                "gemm_strided_batched",
                "cuBLAS",
                &devices,
                format!(
                    "batch={batch_a} m={m} n={n} k={k} trans_a={transpose_a} trans_b={transpose_b}"
                ),
                diagnostics::gemm_flops(m, n, k).saturating_mul(batch_a as u64),
                diagnostics::bytes_for_f64(a.len().saturating_add(b.len())),
                diagnostics::bytes_for_f64(batch_a.saturating_mul(m).saturating_mul(n)),
                start.elapsed().as_secs_f64(),
            );
            Some(out)
        }
        None => {
            diagnostics::log_runtime_cpu(
                "gemm_strided_batched",
                "cuBLAS",
                format!(
                    "batch={batch_a} m={m} n={n} k={k} trans_a={transpose_a} trans_b={transpose_b}"
                ),
            );
            None
        }
    }
}

#[inline]
pub fn try_fast_atb<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Option<Array2<f64>> {
    let (rows, cols) = a.dim();
    let (rows_b, rhs) = b.dim();
    debug_assert_eq!(rows, rows_b, "A and B must have same number of rows");
    if !route_gemm(cols, rhs, rows) {
        diagnostics::log_policy_cpu(
            "atb",
            format!("rows={rows} cols={cols} rhs={rhs}"),
            diagnostics::dispatch_decline_reason(format!(
                "below cuBLAS policy threshold gemm_flops>={}",
                GpuRuntime::global().policy().gemm_min_flops
            )),
        );
        return None;
    }
    let start = std::time::Instant::now();
    match try_multi_atb(a, b).or_else(|| {
        with_runtime(|runtime| runtime.gemm(a, b, true, false))
            .map(|(out, device)| (out, vec![device]))
    }) {
        Some((out, devices)) => {
            diagnostics::log_gpu_success_multi(
                "atb",
                "cuBLAS",
                &devices,
                format!("rows={rows} cols={cols} rhs={rhs}"),
                diagnostics::gemm_flops(cols, rhs, rows),
                diagnostics::bytes_for_f64(a.len().saturating_add(b.len())),
                diagnostics::bytes_for_f64(cols.saturating_mul(rhs)),
                start.elapsed().as_secs_f64(),
            );
            Some(out)
        }
        None => {
            diagnostics::log_runtime_cpu(
                "atb",
                "cuBLAS",
                format!("rows={rows} cols={cols} rhs={rhs}"),
            );
            None
        }
    }
}

#[inline]
pub fn try_fast_av<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
) -> Option<Array1<f64>> {
    let (rows, cols) = a.dim();
    debug_assert_eq!(cols, v.len(), "A cols must match v length");
    if !route_gemv(rows, cols) {
        diagnostics::log_policy_cpu(
            "av",
            format!("rows={rows} cols={cols}"),
            diagnostics::dispatch_decline_reason(format!(
                "below cuBLAS policy threshold gemv_flops>={}",
                GpuRuntime::global().policy().gemv_min_flops
            )),
        );
        return None;
    }
    let start = std::time::Instant::now();
    match with_runtime(|runtime| runtime.gemv(a, v, false)) {
        Some((out, device)) => {
            diagnostics::log_gpu_success(
                "av",
                "cuBLAS",
                &device,
                format!("rows={rows} cols={cols}"),
                diagnostics::gemv_flops(rows, cols),
                diagnostics::bytes_for_f64(a.len().saturating_add(v.len())),
                diagnostics::bytes_for_f64(rows),
                start.elapsed().as_secs_f64(),
            );
            Some(out)
        }
        None => {
            diagnostics::log_runtime_cpu("av", "cuBLAS", format!("rows={rows} cols={cols}"));
            None
        }
    }
}

#[inline]
pub fn try_fast_atv<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    v: &ArrayBase<S2, Ix1>,
) -> Option<Array1<f64>> {
    let (rows, cols) = a.dim();
    debug_assert_eq!(rows, v.len(), "A rows must match v length");
    if !route_gemv(rows, cols) {
        diagnostics::log_policy_cpu(
            "atv",
            format!("rows={rows} cols={cols}"),
            diagnostics::dispatch_decline_reason(format!(
                "below cuBLAS policy threshold gemv_flops>={}",
                GpuRuntime::global().policy().gemv_min_flops
            )),
        );
        return None;
    }
    let start = std::time::Instant::now();
    match with_runtime(|runtime| runtime.gemv(a, v, true)) {
        Some((out, device)) => {
            diagnostics::log_gpu_success(
                "atv",
                "cuBLAS",
                &device,
                format!("rows={rows} cols={cols}"),
                diagnostics::gemv_flops(rows, cols),
                diagnostics::bytes_for_f64(a.len().saturating_add(v.len())),
                diagnostics::bytes_for_f64(cols),
                start.elapsed().as_secs_f64(),
            );
            Some(out)
        }
        None => {
            diagnostics::log_runtime_cpu("atv", "cuBLAS", format!("rows={rows} cols={cols}"));
            None
        }
    }
}

#[inline]
pub fn try_fast_xt_diag_x<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
) -> Option<Array2<f64>> {
    let (rows, cols) = x.dim();
    debug_assert_eq!(rows, w.len(), "X rows must match W length");
    if !route_xtwx(rows, cols, cols) {
        diagnostics::log_policy_cpu(
            "xt_diag_x",
            format!("rows={rows} cols={cols}"),
            diagnostics::dispatch_decline_reason(format!(
                "below cuBLAS policy threshold rows>={} and gemm_flops>={}",
                GpuRuntime::global().policy().xtwx_min_rows,
                GpuRuntime::global().policy().gemm_min_flops
            )),
        );
        return None;
    }
    let start = std::time::Instant::now();
    match try_multi_xt_diag_y(x, w, x).or_else(|| {
        with_runtime(|runtime| runtime.xt_diag_y(x, w, x)).map(|(out, device)| (out, vec![device]))
    }) {
        Some((out, devices)) => {
            diagnostics::log_gpu_success_multi(
                "xt_diag_x",
                "cuBLAS",
                &devices,
                format!("rows={rows} cols={cols}"),
                diagnostics::gemm_flops(cols, cols, rows),
                diagnostics::bytes_for_f64(x.len().saturating_mul(2).saturating_add(w.len())),
                diagnostics::bytes_for_f64(cols.saturating_mul(cols)),
                start.elapsed().as_secs_f64(),
            );
            Some(out)
        }
        None => {
            diagnostics::log_runtime_cpu("xt_diag_x", "cuBLAS", format!("rows={rows} cols={cols}"));
            None
        }
    }
}

#[inline]
pub fn try_fast_xt_diag_y<S1: Data<Elem = f64>, S2: Data<Elem = f64>, S3: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
    y: &ArrayBase<S3, Ix2>,
) -> Option<Array2<f64>> {
    let rows = x.nrows();
    debug_assert_eq!(rows, w.len(), "X rows must match W length");
    debug_assert_eq!(rows, y.nrows(), "X rows must match Y rows");
    if !route_xtwx(rows, x.ncols(), y.ncols()) {
        diagnostics::log_policy_cpu(
            "xt_diag_y",
            format!("rows={rows} lhs_cols={} rhs_cols={}", x.ncols(), y.ncols()),
            diagnostics::dispatch_decline_reason(format!(
                "below cuBLAS policy threshold rows>={} and gemm_flops>={}",
                GpuRuntime::global().policy().xtwx_min_rows,
                GpuRuntime::global().policy().gemm_min_flops
            )),
        );
        return None;
    }
    let start = std::time::Instant::now();
    match try_multi_xt_diag_y(x, w, y).or_else(|| {
        with_runtime(|runtime| runtime.xt_diag_y(x, w, y)).map(|(out, device)| (out, vec![device]))
    }) {
        Some((out, devices)) => {
            diagnostics::log_gpu_success_multi(
                "xt_diag_y",
                "cuBLAS",
                &devices,
                format!("rows={rows} lhs_cols={} rhs_cols={}", x.ncols(), y.ncols()),
                diagnostics::gemm_flops(x.ncols(), y.ncols(), rows),
                diagnostics::bytes_for_f64(x.len().saturating_add(y.len()).saturating_add(w.len())),
                diagnostics::bytes_for_f64(x.ncols().saturating_mul(y.ncols())),
                start.elapsed().as_secs_f64(),
            );
            Some(out)
        }
        None => {
            diagnostics::log_runtime_cpu(
                "xt_diag_y",
                "cuBLAS",
                format!("rows={rows} lhs_cols={} rhs_cols={}", x.ncols(), y.ncols()),
            );
            None
        }
    }
}

#[inline]
pub fn try_solve_lower_triangular_matrix<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    lower: &ArrayBase<S1, Ix2>,
    rhs: &ArrayBase<S2, Ix2>,
) -> Option<Array2<f64>> {
    try_solve_triangular_matrix(
        lower,
        rhs,
        cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
        "trsm_lower",
    )
}

#[inline]
pub fn try_solve_upper_triangular_matrix<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    upper: &ArrayBase<S1, Ix2>,
    rhs: &ArrayBase<S2, Ix2>,
) -> Option<Array2<f64>> {
    try_solve_triangular_matrix(
        upper,
        rhs,
        cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
        "trsm_upper",
    )
}

/// Shared dispatch for the lower/upper triangular solves: same
/// shape-validate → policy gate → cuBLAS attempt → diagnostic logging
/// pipeline that both `try_solve_{lower,upper}_triangular_matrix` walk.
/// The fill-mode flag selects which triangle cuBLAS reads; the `op_label`
/// (`"trsm_lower"` / `"trsm_upper"`) lands in every diagnostic and the
/// CPU-fallback messages so the dispatch reason stays distinguishable.
#[inline]
fn try_solve_triangular_matrix<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    triangular: &ArrayBase<S1, Ix2>,
    rhs: &ArrayBase<S2, Ix2>,
    fill_mode: cublasFillMode_t,
    op_label: &'static str,
) -> Option<Array2<f64>> {
    let p = triangular.nrows();
    if triangular.ncols() != p || rhs.nrows() != p {
        return None;
    }
    if !route_trsm(p, rhs.ncols()) {
        diagnostics::log_policy_cpu(
            op_label,
            format!("p={p} rhs_cols={}", rhs.ncols()),
            diagnostics::dispatch_decline_reason(format!(
                "below cuBLAS policy threshold trsm_flops>={}",
                GpuRuntime::global().policy().trsm_min_flops
            )),
        );
        return None;
    }
    let start = std::time::Instant::now();
    match with_runtime(|runtime| runtime.trsm(triangular, rhs, fill_mode)) {
        Some((out, device)) => {
            diagnostics::log_gpu_success(
                op_label,
                "cuBLAS",
                &device,
                format!("p={p} rhs_cols={}", rhs.ncols()),
                (p as u64)
                    .saturating_mul(p as u64)
                    .saturating_mul(rhs.ncols() as u64),
                diagnostics::bytes_for_f64(triangular.len().saturating_add(rhs.len())),
                diagnostics::bytes_for_f64(rhs.len()),
                start.elapsed().as_secs_f64(),
            );
            Some(out)
        }
        None => {
            diagnostics::log_runtime_cpu(
                op_label,
                "cuBLAS",
                format!("p={p} rhs_cols={}", rhs.ncols()),
            );
            None
        }
    }
}

#[inline]
fn route_gemm(m: usize, n: usize, k: usize) -> bool {
    m > 0 && n > 0 && k > 0 && GpuRuntime::global().policy().route_gemm(m, n, k)
}

#[inline]
fn route_gemm_batched(m: usize, n: usize, k: usize, batch_size: usize) -> bool {
    m > 0
        && n > 0
        && k > 0
        && batch_size > 0
        && GpuRuntime::global()
            .policy()
            .route_gemm_batched(m, n, k, batch_size)
}

#[inline]
fn route_gemv(rows: usize, cols: usize) -> bool {
    rows > 0 && cols > 0 && GpuRuntime::global().policy().route_gemv(rows, cols)
}

#[inline]
fn route_xtwx(rows: usize, lhs_cols: usize, rhs_cols: usize) -> bool {
    rows > 0
        && lhs_cols > 0
        && rhs_cols > 0
        && GpuRuntime::global()
            .policy()
            .route_xt_diag_y(rows, lhs_cols, rhs_cols)
}

#[inline]
fn route_trsm(p: usize, rhs_cols: usize) -> bool {
    p > 0 && rhs_cols > 0 && GpuRuntime::global().policy().route_trsm(p, rhs_cols)
}

#[inline]
fn op(transpose: bool) -> cublasOperation_t {
    if transpose {
        cublasOperation_t::CUBLAS_OP_T
    } else {
        cublasOperation_t::CUBLAS_OP_N
    }
}

impl super::runtime::HasGpuDevice for CublasRuntime {
    fn device(&self) -> &GpuDeviceInfo {
        &self.device
    }
}

fn with_runtime<T>(f: impl FnMut(&mut CublasRuntime) -> Option<T>) -> Option<(T, GpuDeviceInfo)> {
    super::runtime::with_runtime_two_phase(cublas_runtimes(), f)
}

fn cublas_runtimes() -> &'static [Mutex<CublasRuntime>] {
    static RUNTIME: OnceLock<Vec<Mutex<CublasRuntime>>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            GpuRuntime::global()
                .devices()
                .iter()
                .filter_map(|device| match CublasRuntime::new(device.clone()) {
                    Ok(runtime) => {
                        diagnostics::log_library_ready("cuBLAS", &runtime.device);
                        Some(Mutex::new(runtime))
                    }
                    Err(err) => {
                        diagnostics::log_library_unavailable("cuBLAS", &err.to_string());
                        None
                    }
                })
                .collect()
        })
        .as_slice()
}

fn plan_for_cublas(
    batch_size: usize,
    fixed_bytes_per_device: usize,
    bytes_per_batch_item: usize,
) -> Option<Vec<(usize, GpuDeviceInfo, Vec<Range<usize>>)>> {
    let runtimes = cublas_runtimes();
    if runtimes.len() <= 1 {
        return None;
    }
    let mut devices = Vec::with_capacity(runtimes.len());
    for runtime in runtimes {
        devices.push(runtime.lock().ok()?.device.clone());
    }
    let plans = GpuRuntime::global().plan_batched_work_for_devices(
        &devices,
        batch_size,
        fixed_bytes_per_device,
        bytes_per_batch_item,
    )?;
    if plans.len() <= 1 {
        return None;
    }
    let mut mapped = Vec::with_capacity(plans.len());
    for plan in plans {
        let idx = devices
            .iter()
            .position(|device| device.ordinal == plan.ordinal)?;
        mapped.push((idx, devices[idx].clone(), plan.chunks));
    }
    Some(mapped)
}

fn try_multi_ab<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Option<(Array2<f64>, Vec<GpuDeviceInfo>)> {
    let (rows, inner) = a.dim();
    let (b_rows, cols) = b.dim();
    if rows == 0 || inner == 0 || cols == 0 || inner != b_rows {
        return None;
    }
    let fixed_bytes_per_device = diagnostics::bytes_for_f64(inner.checked_mul(cols)?);
    let bytes_per_row = diagnostics::bytes_for_f64(inner.checked_add(cols)?);
    let plan = plan_for_cublas(rows, fixed_bytes_per_device, bytes_per_row)?;
    if plan.len() <= 1 {
        return None;
    }
    let runtimes = cublas_runtimes();
    let b_owned = b.to_owned();
    let partials = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(plan.len());
        for (runtime_idx, device, chunks) in plan {
            let b_device = b_owned.clone();
            let owned_chunks = chunks
                .into_iter()
                .map(|rows_range| (rows_range.clone(), a.slice(s![rows_range, ..]).to_owned()))
                .collect::<Vec<_>>();
            handles.push(scope.spawn(move || {
                let mut runtime = runtimes[runtime_idx].lock().ok()?;
                let mut out_chunks = Vec::with_capacity(owned_chunks.len());
                for (rows_range, a_chunk) in owned_chunks {
                    let chunk_out = runtime.gemm(&a_chunk, &b_device, false, false)?;
                    out_chunks.push((rows_range, chunk_out));
                }
                Some((out_chunks, device))
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().ok().flatten())
            .collect::<Option<Vec<_>>>()
    })?;
    let mut out = Array2::<f64>::zeros((rows, cols));
    let mut devices = Vec::with_capacity(partials.len());
    for (chunks, device) in partials {
        devices.push(device);
        for (rows_range, chunk) in chunks {
            out.slice_mut(s![rows_range, ..]).assign(&chunk);
        }
    }
    Some((out, devices))
}

fn try_multi_atb<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Option<(Array2<f64>, Vec<GpuDeviceInfo>)> {
    let (rows, cols) = a.dim();
    let (b_rows, rhs) = b.dim();
    if rows == 0 || cols == 0 || rhs == 0 || rows != b_rows {
        return None;
    }
    let fixed_bytes_per_device = diagnostics::bytes_for_f64(cols.checked_mul(rhs)?);
    let bytes_per_row = diagnostics::bytes_for_f64(cols.checked_add(rhs)?);
    let plan = plan_for_cublas(rows, fixed_bytes_per_device, bytes_per_row)?;
    if plan.len() <= 1 {
        return None;
    }
    let runtimes = cublas_runtimes();
    let partials = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(plan.len());
        for (runtime_idx, device, chunks) in plan {
            let owned_chunks = chunks
                .into_iter()
                .map(|rows_range| {
                    (
                        a.slice(s![rows_range.clone(), ..]).to_owned(),
                        b.slice(s![rows_range, ..]).to_owned(),
                    )
                })
                .collect::<Vec<_>>();
            handles.push(scope.spawn(move || {
                let mut runtime = runtimes[runtime_idx].lock().ok()?;
                let mut partial = Array2::<f64>::zeros((cols, rhs));
                for (a_chunk, b_chunk) in owned_chunks {
                    let chunk_out = runtime.gemm(&a_chunk, &b_chunk, true, false)?;
                    partial += &chunk_out;
                }
                Some((partial, device))
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().ok().flatten())
            .collect::<Option<Vec<_>>>()
    })?;
    let mut out = Array2::<f64>::zeros((cols, rhs));
    let mut devices = Vec::with_capacity(partials.len());
    for (partial, device) in partials {
        out += &partial;
        devices.push(device);
    }
    Some((out, devices))
}

fn try_multi_xt_diag_y<S1: Data<Elem = f64>, S2: Data<Elem = f64>, S3: Data<Elem = f64>>(
    x: &ArrayBase<S1, Ix2>,
    w: &ArrayBase<S2, Ix1>,
    y: &ArrayBase<S3, Ix2>,
) -> Option<(Array2<f64>, Vec<GpuDeviceInfo>)> {
    let rows = x.nrows();
    let x_cols = x.ncols();
    let y_cols = y.ncols();
    if rows == 0 || x_cols == 0 || y_cols == 0 || rows != w.len() || rows != y.nrows() {
        return None;
    }
    let fixed_bytes_per_device = diagnostics::bytes_for_f64(x_cols.checked_mul(y_cols)?);
    let bytes_per_row = diagnostics::bytes_for_f64(x_cols.checked_add(y_cols)?.checked_add(1)?);
    let plan = plan_for_cublas(rows, fixed_bytes_per_device, bytes_per_row)?;
    if plan.len() <= 1 {
        return None;
    }
    let runtimes = cublas_runtimes();
    let partials = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(plan.len());
        for (runtime_idx, device, chunks) in plan {
            let owned_chunks = chunks
                .into_iter()
                .map(|rows_range| {
                    (
                        x.slice(s![rows_range.clone(), ..]).to_owned(),
                        w.slice(s![rows_range.clone()]).to_owned(),
                        y.slice(s![rows_range, ..]).to_owned(),
                    )
                })
                .collect::<Vec<_>>();
            handles.push(scope.spawn(move || {
                let mut runtime = runtimes[runtime_idx].lock().ok()?;
                let mut partial = Array2::<f64>::zeros((x_cols, y_cols));
                for (x_chunk, w_chunk, y_chunk) in owned_chunks {
                    let chunk_out = runtime.xt_diag_y(&x_chunk, &w_chunk, &y_chunk)?;
                    partial += &chunk_out;
                }
                Some((partial, device))
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().ok().flatten())
            .collect::<Option<Vec<_>>>()
    })?;
    let mut out = Array2::<f64>::zeros((x_cols, y_cols));
    let mut devices = Vec::with_capacity(partials.len());
    for (partial, device) in partials {
        out += &partial;
        devices.push(device);
    }
    Some((out, devices))
}

fn try_multi_gemm_strided_batched(
    a: ArrayView3<'_, f64>,
    b: ArrayView3<'_, f64>,
    transpose_a: bool,
    transpose_b: bool,
) -> Option<(Array3<f64>, Vec<GpuDeviceInfo>)> {
    let (batch, a_rows, a_cols) = a.dim();
    let (_, b_rows, b_cols) = b.dim();
    let m = if transpose_a { a_cols } else { a_rows };
    let n = if transpose_b { b_rows } else { b_cols };
    let a_stride = a_rows.checked_mul(a_cols)?;
    let b_stride = b_rows.checked_mul(b_cols)?;
    let c_stride = m.checked_mul(n)?;
    let bytes_per_batch_item =
        diagnostics::bytes_for_f64(a_stride.saturating_add(b_stride).saturating_add(c_stride));
    let plan = plan_for_cublas(batch, 0, bytes_per_batch_item)?;
    let runtimes = cublas_runtimes();
    let mut pieces = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(plan.len());
        for (runtime_idx, device, chunks) in plan {
            handles.push(scope.spawn(move || {
                let mut out = Vec::with_capacity(chunks.len());
                let mut runtime = runtimes[runtime_idx].lock().ok()?;
                for range in chunks {
                    let chunk_a = a.slice(s![range.clone(), .., ..]);
                    let chunk_b = b.slice(s![range.clone(), .., ..]);
                    let chunk =
                        runtime.gemm_strided_batched(chunk_a, chunk_b, transpose_a, transpose_b)?;
                    out.push((range, chunk));
                }
                Some((device, out))
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().ok()?)
            .collect::<Option<Vec<_>>>()
    })?;
    assemble_batched_output(batch, m, n, &mut pieces)
}

fn try_multi_gemm_broadcast_b_strided_batched(
    a: ArrayView3<'_, f64>,
    b: ArrayView2<'_, f64>,
    transpose_a: bool,
    transpose_b: bool,
) -> Option<(Array3<f64>, Vec<GpuDeviceInfo>)> {
    let (batch, a_rows, a_cols) = a.dim();
    let (b_rows, b_cols) = b.dim();
    let m = if transpose_a { a_cols } else { a_rows };
    let n = if transpose_b { b_rows } else { b_cols };
    let a_stride = a_rows.checked_mul(a_cols)?;
    let c_stride = m.checked_mul(n)?;
    let fixed_bytes_per_device = diagnostics::bytes_for_f64(b_rows.checked_mul(b_cols)?);
    let bytes_per_batch_item = diagnostics::bytes_for_f64(a_stride.saturating_add(c_stride));
    let plan = plan_for_cublas(batch, fixed_bytes_per_device, bytes_per_batch_item)?;
    let runtimes = cublas_runtimes();
    let mut pieces = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(plan.len());
        for (runtime_idx, device, chunks) in plan {
            handles.push(scope.spawn(move || {
                let mut out = Vec::with_capacity(chunks.len());
                let mut runtime = runtimes[runtime_idx].lock().ok()?;
                for range in chunks {
                    let chunk_a = a.slice(s![range.clone(), .., ..]);
                    let chunk = runtime.gemm_broadcast_b_strided_batched(
                        chunk_a,
                        b,
                        transpose_a,
                        transpose_b,
                    )?;
                    out.push((range, chunk));
                }
                Some((device, out))
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().ok()?)
            .collect::<Option<Vec<_>>>()
    })?;
    assemble_batched_output(batch, m, n, &mut pieces)
}

fn try_multi_broadcast_a_gemm_strided_batched(
    a: ArrayView2<'_, f64>,
    b: ArrayView3<'_, f64>,
    transpose_a: bool,
    transpose_b: bool,
) -> Option<(Array3<f64>, Vec<GpuDeviceInfo>)> {
    let (a_rows, a_cols) = a.dim();
    let (batch, b_rows, b_cols) = b.dim();
    let m = if transpose_a { a_cols } else { a_rows };
    let n = if transpose_b { b_rows } else { b_cols };
    let b_stride = b_rows.checked_mul(b_cols)?;
    let c_stride = m.checked_mul(n)?;
    let fixed_bytes_per_device = diagnostics::bytes_for_f64(a_rows.checked_mul(a_cols)?);
    let bytes_per_batch_item = diagnostics::bytes_for_f64(b_stride.saturating_add(c_stride));
    let plan = plan_for_cublas(batch, fixed_bytes_per_device, bytes_per_batch_item)?;
    let runtimes = cublas_runtimes();
    let mut pieces = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(plan.len());
        for (runtime_idx, device, chunks) in plan {
            handles.push(scope.spawn(move || {
                let mut out = Vec::with_capacity(chunks.len());
                let mut runtime = runtimes[runtime_idx].lock().ok()?;
                for range in chunks {
                    let chunk_b = b.slice(s![range.clone(), .., ..]);
                    let chunk = runtime.broadcast_a_gemm_strided_batched(
                        a,
                        chunk_b,
                        transpose_a,
                        transpose_b,
                    )?;
                    out.push((range, chunk));
                }
                Some((device, out))
            }));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().ok()?)
            .collect::<Option<Vec<_>>>()
    })?;
    assemble_batched_output(batch, m, n, &mut pieces)
}

type BatchedPiece = (GpuDeviceInfo, Vec<(Range<usize>, Array3<f64>)>);

fn assemble_batched_output(
    batch: usize,
    rows: usize,
    cols: usize,
    pieces: &mut [BatchedPiece],
) -> Option<(Array3<f64>, Vec<GpuDeviceInfo>)> {
    let mut out = Array3::<f64>::zeros((batch, rows, cols));
    let mut devices = Vec::with_capacity(pieces.len());
    for (device, chunks) in pieces.iter_mut() {
        devices.push(device.clone());
        for (range, chunk) in chunks.drain(..) {
            if chunk.dim() != (range.end - range.start, rows, cols) {
                return None;
            }
            out.slice_mut(s![range, .., ..]).assign(&chunk);
        }
    }
    Some((out, devices))
}

// ---------------------------------------------------------------------------
// CublasRuntime — owns one (CudaContext, CudaStream, CudaBlas) per device.
// All methods bind the calling thread to the context first, then run the
// op via cudarc's safe wrappers (Gemm / Gemv) or directly through the
// `cublasD*` sys symbols for the few ops that aren't safe-wrapped.
// ---------------------------------------------------------------------------

struct CublasRuntime {
    device: GpuDeviceInfo,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
}

// `CudaBlas` is `Send + Sync` per cudarc's impl, and the surrounding
// `Mutex<CublasRuntime>` serializes all access anyway. No explicit unsafe
// impl needed.

impl CublasRuntime {
    fn new(device: GpuDeviceInfo) -> Result<Self, crate::gpu::GpuError> {
        use crate::gpu::GpuError;
        let ctx = cuda_context_for(device.ordinal).ok_or_else(|| GpuError::DriverCallFailed {
            reason: format!("CudaContext unavailable for ordinal {}", device.ordinal),
        })?;
        ctx.bind_to_thread()
            .map_err(|e| GpuError::DriverCallFailed {
                reason: e.to_string(),
            })?;
        let stream = ctx.new_stream().map_err(|e| GpuError::DriverCallFailed {
            reason: e.to_string(),
        })?;
        let blas = CudaBlas::new(stream.clone()).map_err(|e| GpuError::DriverCallFailed {
            reason: e.to_string(),
        })?;
        Ok(Self {
            device,
            ctx,
            stream,
            blas,
        })
    }

    #[inline]
    fn bind(&self) -> Option<()> {
        self.ctx.bind_to_thread().ok()
    }

    fn gemm<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
        &mut self,
        a: &ArrayBase<S1, Ix2>,
        b: &ArrayBase<S2, Ix2>,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Option<Array2<f64>> {
        let a_rows = a.nrows();
        let a_cols = a.ncols();
        let b_rows = b.nrows();
        let b_cols = b.ncols();
        let m = if transpose_a { a_cols } else { a_rows };
        let k = if transpose_a { a_rows } else { a_cols };
        let b_k = if transpose_b { b_cols } else { b_rows };
        let n = if transpose_b { b_rows } else { b_cols };
        if k != b_k {
            return None;
        }
        self.bind()?;

        let a_host = to_col_major(a);
        let b_host = to_col_major(b);
        let a_dev: CudaSlice<f64> = self.stream.clone_htod(&*a_host).ok()?;
        let b_dev: CudaSlice<f64> = self.stream.clone_htod(&*b_host).ok()?;
        let mut c_dev: CudaSlice<f64> = self.stream.alloc_zeros::<f64>(m.checked_mul(n)?).ok()?;

        let cfg = GemmConfig::<f64> {
            transa: op(transpose_a),
            transb: op(transpose_b),
            m: to_i32(m)?,
            n: to_i32(n)?,
            k: to_i32(k)?,
            alpha: 1.0,
            lda: to_i32(a_rows)?,
            ldb: to_i32(b_rows)?,
            beta: 0.0,
            ldc: to_i32(m)?,
        };
        // SAFETY: cfg shape and leading dims match the column-major buffers
        // we just uploaded; the slices live until after the dtov below.
        unsafe { self.blas.gemm(cfg, &a_dev, &b_dev, &mut c_dev) }.ok()?;
        let c_host: Vec<f64> = self.stream.clone_dtoh(&c_dev).ok()?;
        Some(from_col_major(&c_host, m, n))
    }

    fn gemv<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
        &mut self,
        a: &ArrayBase<S1, Ix2>,
        v: &ArrayBase<S2, Ix1>,
        transpose: bool,
    ) -> Option<Array1<f64>> {
        let (rows, cols) = a.dim();
        let out_len = if transpose { cols } else { rows };
        self.bind()?;

        let a_host = to_col_major(a);
        let x_host: Vec<f64> = v.iter().copied().collect();
        let a_dev: CudaSlice<f64> = self.stream.clone_htod(&*a_host).ok()?;
        let x_dev: CudaSlice<f64> = self.stream.clone_htod(&x_host).ok()?;
        let mut y_dev: CudaSlice<f64> = self.stream.alloc_zeros::<f64>(out_len).ok()?;

        let cfg = GemvConfig::<f64> {
            trans: op(transpose),
            m: to_i32(rows)?,
            n: to_i32(cols)?,
            alpha: 1.0,
            lda: to_i32(rows)?,
            incx: 1,
            beta: 0.0,
            incy: 1,
        };
        // SAFETY: rows/cols come from `a.dim()`, lda = rows matches the
        // column-major upload, increments are 1 for contiguous host vectors.
        unsafe { self.blas.gemv(cfg, &a_dev, &x_dev, &mut y_dev) }.ok()?;
        let y_host: Vec<f64> = self.stream.clone_dtoh(&y_dev).ok()?;
        Some(Array1::from_vec(y_host))
    }

    fn xt_diag_y<S1: Data<Elem = f64>, S2: Data<Elem = f64>, S3: Data<Elem = f64>>(
        &mut self,
        x: &ArrayBase<S1, Ix2>,
        w: &ArrayBase<S2, Ix1>,
        y: &ArrayBase<S3, Ix2>,
    ) -> Option<Array2<f64>> {
        let rows = x.nrows();
        let x_cols = x.ncols();
        let y_cols = y.ncols();
        if rows != w.len() || rows != y.nrows() {
            return None;
        }
        self.bind()?;

        let x_host = to_col_major(x);
        let y_host = to_col_major(y);
        let w_host: Vec<f64> = w.iter().copied().collect();
        let x_dev: CudaSlice<f64> = self.stream.clone_htod(&*x_host).ok()?;
        let y_dev: CudaSlice<f64> = self.stream.clone_htod(&*y_host).ok()?;
        let w_dev: CudaSlice<f64> = self.stream.clone_htod(&w_host).ok()?;
        let mut wy_dev: CudaSlice<f64> = self
            .stream
            .alloc_zeros::<f64>(rows.checked_mul(y_cols)?)
            .ok()?;
        let mut out_dev: CudaSlice<f64> = self
            .stream
            .alloc_zeros::<f64>(x_cols.checked_mul(y_cols)?)
            .ok()?;

        let rows_i = to_i32(rows)?;
        let x_cols_i = to_i32(x_cols)?;
        let y_cols_i = to_i32(y_cols)?;

        // wy = diag(w) · Y via cublasDdgmm (no safe wrapper).
        // SAFETY: shapes/leading dims match the column-major Y upload;
        // pointers come from valid live CudaSlices.
        let ddgmm_status = unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (y_ptr, _r_y) = y_dev.device_ptr(&self.stream);
            let (w_ptr, _r_w) = w_dev.device_ptr(&self.stream);
            let (wy_ptr, _r_wy) = wy_dev.device_ptr_mut(&self.stream);
            cudarc::cublas::sys::cublasDdgmm(
                *self.blas.handle(),
                cublasSideMode_t::CUBLAS_SIDE_LEFT,
                rows_i,
                y_cols_i,
                y_ptr as *const f64,
                rows_i,
                w_ptr as *const f64,
                1,
                wy_ptr as *mut f64,
                rows_i,
            )
        };
        if ddgmm_status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return None;
        }

        // out = Xᵀ · wy
        let cfg = GemmConfig::<f64> {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: x_cols_i,
            n: y_cols_i,
            k: rows_i,
            alpha: 1.0,
            lda: rows_i,
            ldb: rows_i,
            beta: 0.0,
            ldc: x_cols_i,
        };
        // SAFETY: shape/leading-dim values match the live device buffers
        // (X is rows×x_cols col-major, wy is rows×y_cols col-major).
        unsafe { self.blas.gemm(cfg, &x_dev, &wy_dev, &mut out_dev) }.ok()?;
        let out_host: Vec<f64> = self.stream.clone_dtoh(&out_dev).ok()?;
        Some(from_col_major(&out_host, x_cols, y_cols))
    }

    fn trsm<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
        &mut self,
        triangular: &ArrayBase<S1, Ix2>,
        rhs: &ArrayBase<S2, Ix2>,
        uplo: cublasFillMode_t,
    ) -> Option<Array2<f64>> {
        let p = triangular.nrows();
        let rhs_cols = rhs.ncols();
        self.bind()?;

        let tri_host = to_col_major(triangular);
        let rhs_host = to_col_major(rhs);
        let tri_dev: CudaSlice<f64> = self.stream.clone_htod(&*tri_host).ok()?;
        let mut rhs_dev: CudaSlice<f64> = self.stream.clone_htod(&*rhs_host).ok()?;

        let p_i = to_i32(p)?;
        let rhs_cols_i = to_i32(rhs_cols)?;
        let alpha = 1.0_f64;
        // SAFETY: cublasDtrsm_v2 writes the solution in place into B.
        // Shapes are p×p triangular and p×rhs_cols B, both column-major.
        let status = unsafe {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (tri_ptr, _r_t) = tri_dev.device_ptr(&self.stream);
            let (rhs_ptr, _r_r) = rhs_dev.device_ptr_mut(&self.stream);
            cudarc::cublas::sys::cublasDtrsm_v2(
                *self.blas.handle(),
                cublasSideMode_t::CUBLAS_SIDE_LEFT,
                uplo,
                cublasOperation_t::CUBLAS_OP_N,
                cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
                p_i,
                rhs_cols_i,
                &alpha as *const f64,
                tri_ptr as *const f64,
                p_i,
                rhs_ptr as *mut f64,
                p_i,
            )
        };
        if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return None;
        }
        let out_host: Vec<f64> = self.stream.clone_dtoh(&rhs_dev).ok()?;
        Some(from_col_major(&out_host, p, rhs_cols))
    }

    fn gemm_strided_batched(
        &mut self,
        a: ArrayView3<'_, f64>,
        b: ArrayView3<'_, f64>,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Option<Array3<f64>> {
        let (batch, a_rows, a_cols) = a.dim();
        let (_, b_rows, b_cols) = b.dim();
        let m = if transpose_a { a_cols } else { a_rows };
        let k = if transpose_a { a_rows } else { a_cols };
        let b_k = if transpose_b { b_cols } else { b_rows };
        let n = if transpose_b { b_rows } else { b_cols };
        if k != b_k {
            return None;
        }
        let a_stride = a_rows.checked_mul(a_cols)?;
        let b_stride = b_rows.checked_mul(b_cols)?;
        let c_stride = m.checked_mul(n)?;

        let a_host = pack_a3_col_major(a);
        let b_host = pack_a3_col_major(b);
        self.run_strided_batched(
            &a_host,
            &b_host,
            transpose_a,
            transpose_b,
            batch,
            m,
            n,
            k,
            a_rows,
            b_rows,
            a_stride as i64,
            b_stride as i64,
            c_stride,
        )
    }

    fn gemm_broadcast_b_strided_batched(
        &mut self,
        a: ArrayView3<'_, f64>,
        b: ArrayView2<'_, f64>,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Option<Array3<f64>> {
        let (batch, a_rows, a_cols) = a.dim();
        let (b_rows, b_cols) = b.dim();
        let m = if transpose_a { a_cols } else { a_rows };
        let k = if transpose_a { a_rows } else { a_cols };
        let b_k = if transpose_b { b_cols } else { b_rows };
        let n = if transpose_b { b_rows } else { b_cols };
        if k != b_k {
            return None;
        }
        let a_stride = a_rows.checked_mul(a_cols)?;
        let c_stride = m.checked_mul(n)?;

        let a_host = pack_a3_col_major(a);
        let b_host = to_col_major(&b);
        // strideB = 0 broadcasts B across all batch elements.
        self.run_strided_batched(
            &a_host,
            &b_host,
            transpose_a,
            transpose_b,
            batch,
            m,
            n,
            k,
            a_rows,
            b_rows,
            a_stride as i64,
            0,
            c_stride,
        )
    }

    fn broadcast_a_gemm_strided_batched(
        &mut self,
        a: ArrayView2<'_, f64>,
        b: ArrayView3<'_, f64>,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Option<Array3<f64>> {
        let (a_rows, a_cols) = a.dim();
        let (batch, b_rows, b_cols) = b.dim();
        let m = if transpose_a { a_cols } else { a_rows };
        let k = if transpose_a { a_rows } else { a_cols };
        let b_k = if transpose_b { b_cols } else { b_rows };
        let n = if transpose_b { b_rows } else { b_cols };
        if k != b_k {
            return None;
        }
        let b_stride = b_rows.checked_mul(b_cols)?;
        let c_stride = m.checked_mul(n)?;

        let a_host = to_col_major(&a);
        let b_host = pack_a3_col_major(b);
        // strideA = 0 broadcasts A across all batch elements.
        self.run_strided_batched(
            &a_host,
            &b_host,
            transpose_a,
            transpose_b,
            batch,
            m,
            n,
            k,
            a_rows,
            b_rows,
            0,
            b_stride as i64,
            c_stride,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_strided_batched(
        &mut self,
        a_host: &[f64],
        b_host: &[f64],
        transpose_a: bool,
        transpose_b: bool,
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
        a_rows: usize,
        b_rows: usize,
        stride_a: i64,
        stride_b: i64,
        c_stride: usize,
    ) -> Option<Array3<f64>> {
        self.bind()?;
        let c_total = batch.checked_mul(c_stride)?;
        let a_dev: CudaSlice<f64> = self.stream.clone_htod(a_host).ok()?;
        let b_dev: CudaSlice<f64> = self.stream.clone_htod(b_host).ok()?;
        let mut c_dev: CudaSlice<f64> = self.stream.alloc_zeros::<f64>(c_total).ok()?;

        let cfg = StridedBatchedConfig::<f64> {
            gemm: GemmConfig::<f64> {
                transa: op(transpose_a),
                transb: op(transpose_b),
                m: to_i32(m)?,
                n: to_i32(n)?,
                k: to_i32(k)?,
                alpha: 1.0,
                lda: to_i32(a_rows)?,
                ldb: to_i32(b_rows)?,
                beta: 0.0,
                ldc: to_i32(m)?,
            },
            batch_size: to_i32(batch)?,
            stride_a,
            stride_b,
            stride_c: c_stride as i64,
        };
        // SAFETY: every leading dimension and stride matches the packed
        // column-major host layout we just uploaded; result buffer has
        // batch * c_stride elements.
        unsafe {
            self.blas
                .gemm_strided_batched(cfg, &a_dev, &b_dev, &mut c_dev)
        }
        .ok()?;
        let c_host: Vec<f64> = self.stream.clone_dtoh(&c_dev).ok()?;

        // Unpack column-major slabs back into a row-major Array3.
        let mut out = Array3::<f64>::zeros((batch, m, n));
        for batch_idx in 0..batch {
            let start = batch_idx * c_stride;
            let mut dest = out.slice_mut(s![batch_idx, .., ..]);
            for col in 0..n {
                for row in 0..m {
                    dest[[row, col]] = c_host[start + col * m + row];
                }
            }
        }
        Some(out)
    }
}

/// Pack an Array3 (batch, rows, cols) row-major into a flat column-major
/// buffer suitable for `cublasDgemmStridedBatched`. Each batch slab is
/// `rows * cols` f64s contiguous in memory with stride = rows*cols.
fn pack_a3_col_major(a: ArrayView3<'_, f64>) -> Vec<f64> {
    let (batch, rows, cols) = a.dim();
    let mut out = Vec::with_capacity(batch.saturating_mul(rows).saturating_mul(cols));
    for batch_idx in 0..batch {
        let slice = a.slice(s![batch_idx, .., ..]);
        for col in 0..cols {
            out.extend(slice.column(col).iter().copied());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn small_operations_do_not_route_to_gpu() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0], [6.0]];
        assert!(try_fast_ab(&a, &b).is_none());
        assert!(try_fast_av(&a, &array![1.0, 2.0]).is_none());
        assert!(try_solve_lower_triangular_matrix(&a, &b).is_none());
        assert!(try_solve_upper_triangular_matrix(&a, &b).is_none());
    }

    #[test]
    fn column_major_round_trip_preserves_values() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let packed = to_col_major(&a);
        assert_eq!(packed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(from_col_major(&packed, 2, 3), a);
    }
}

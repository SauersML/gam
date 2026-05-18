//! cuBLAS routing for large f64 dense kernels.

use libloading::Library;
use ndarray::{Array1, Array2, Array3, ArrayBase, ArrayView3, Data, Ix1, Ix2, s};
use std::ops::Range;
use std::sync::{Mutex, OnceLock};

use super::device::GpuDeviceInfo;
use super::diagnostics;
use super::driver::{
    CudaWorkingState, DeviceAllocation, bytes_len, check_cuda, from_col_major, load_static_library,
    to_col_major, to_i32, to_i64,
};
use super::runtime::GpuRuntime;

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
    match with_runtime(|runtime| runtime.gemm(a, b, false, false)) {
        Some((out, device)) => {
            diagnostics::log_gpu_success(
                "gemm",
                "cuBLAS",
                &device,
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
/// variants) for `b = 0..batch`, all in a single cuBLAS call. The strided
/// form requires every batch element to share the same `(m, n, k)` shape; the
/// per-batch leading dimensions are taken from the stacked Array3 layout
/// (row-major from ndarray, repacked column-major internally for cuBLAS).
/// Returns `None` if the policy threshold is not met, dimensions disagree,
/// or any device call fails — callers fall back to per-batch dispatch.
#[inline]
pub fn try_fast_ab_strided_batched(
    a: ArrayView3<'_, f64>,
    b: ArrayView3<'_, f64>,
) -> Option<Array3<f64>> {
    try_fast_gemm_strided_batched(a, b, false, false)
}

/// Strided batched `Aᵀ · B`. Same constraints as
/// [`try_fast_ab_strided_batched`] but transposes A on the device per batch.
#[inline]
pub fn try_fast_atb_strided_batched(
    a: ArrayView3<'_, f64>,
    b: ArrayView3<'_, f64>,
) -> Option<Array3<f64>> {
    try_fast_gemm_strided_batched(a, b, true, false)
}

/// Strided batched `A · Bᵀ`. Mirrors [`try_fast_ab_strided_batched`] but
/// transposes B on the device per batch.
#[inline]
pub fn try_fast_abt_strided_batched(
    a: ArrayView3<'_, f64>,
    b: ArrayView3<'_, f64>,
) -> Option<Array3<f64>> {
    try_fast_gemm_strided_batched(a, b, false, true)
}

/// Strided batched `A[b] · B` with `B` broadcast (same matrix for every batch
/// element). Uses `cublasDgemmStridedBatched` with `strideB = 0` so the
/// device-side B buffer is uploaded once. Designed for the K-way whitened-
/// penalty step `L_b⁻¹ · S` of the batched REML cache build, where every
/// problem shares the same penalty matrix S. Returns `None` when the policy
/// declines or any device call fails.
#[inline]
pub fn try_fast_ab_broadcast_b_batched(
    a: ArrayView3<'_, f64>,
    b: ndarray::ArrayView2<'_, f64>,
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

/// Strided batched `A · B[b]ᵀ` with `A` broadcast. Mirrors
/// [`try_fast_ab_broadcast_b_batched`] for the symmetric `S · L_b⁻ᵀ` step
/// that follows the first whitening multiply in `L_b⁻¹ · S · L_b⁻ᵀ`.
#[inline]
pub fn try_fast_a_broadcast_bt_batched(
    a: ndarray::ArrayView2<'_, f64>,
    b: ArrayView3<'_, f64>,
) -> Option<Array3<f64>> {
    let (a_rows, a_cols) = a.dim();
    let (batch, b_rows, b_cols) = b.dim();
    if batch == 0 {
        return None;
    }
    // Compute A @ B[b]^T: result is (a_rows × b_rows).
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
    match with_runtime(|runtime| runtime.gemm(a, b, true, false)) {
        Some((out, device)) => {
            diagnostics::log_gpu_success(
                "atb",
                "cuBLAS",
                &device,
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
    match with_runtime(|runtime| runtime.xt_diag_y(x, w, x)) {
        Some((out, device)) => {
            diagnostics::log_gpu_success(
                "xt_diag_x",
                "cuBLAS",
                &device,
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
    match with_runtime(|runtime| runtime.xt_diag_y(x, w, y)) {
        Some((out, device)) => {
            diagnostics::log_gpu_success(
                "xt_diag_y",
                "cuBLAS",
                &device,
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
    let p = lower.nrows();
    if lower.ncols() != p || rhs.nrows() != p {
        return None;
    }
    if !route_trsm(p, rhs.ncols()) {
        diagnostics::log_policy_cpu(
            "trsm_lower",
            format!("p={p} rhs_cols={}", rhs.ncols()),
            diagnostics::dispatch_decline_reason(format!(
                "below cuBLAS policy threshold trsm_flops>={}",
                GpuRuntime::global().policy().trsm_min_flops
            )),
        );
        return None;
    }
    let start = std::time::Instant::now();
    match with_runtime(|runtime| runtime.trsm(lower, rhs, CUBLAS_FILL_LOWER)) {
        Some((out, device)) => {
            diagnostics::log_gpu_success(
                "trsm_lower",
                "cuBLAS",
                &device,
                format!("p={p} rhs_cols={}", rhs.ncols()),
                (p as u64)
                    .saturating_mul(p as u64)
                    .saturating_mul(rhs.ncols() as u64),
                diagnostics::bytes_for_f64(lower.len().saturating_add(rhs.len())),
                diagnostics::bytes_for_f64(rhs.len()),
                start.elapsed().as_secs_f64(),
            );
            Some(out)
        }
        None => {
            diagnostics::log_runtime_cpu(
                "trsm_lower",
                "cuBLAS",
                format!("p={p} rhs_cols={}", rhs.ncols()),
            );
            None
        }
    }
}

#[inline]
pub fn try_solve_upper_triangular_matrix<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    upper: &ArrayBase<S1, Ix2>,
    rhs: &ArrayBase<S2, Ix2>,
) -> Option<Array2<f64>> {
    let p = upper.nrows();
    if upper.ncols() != p || rhs.nrows() != p {
        return None;
    }
    if !route_trsm(p, rhs.ncols()) {
        diagnostics::log_policy_cpu(
            "trsm_upper",
            format!("p={p} rhs_cols={}", rhs.ncols()),
            diagnostics::dispatch_decline_reason(format!(
                "below cuBLAS policy threshold trsm_flops>={}",
                GpuRuntime::global().policy().trsm_min_flops
            )),
        );
        return None;
    }
    let start = std::time::Instant::now();
    match with_runtime(|runtime| runtime.trsm(upper, rhs, CUBLAS_FILL_UPPER)) {
        Some((out, device)) => {
            diagnostics::log_gpu_success(
                "trsm_upper",
                "cuBLAS",
                &device,
                format!("p={p} rhs_cols={}", rhs.ncols()),
                (p as u64)
                    .saturating_mul(p as u64)
                    .saturating_mul(rhs.ncols() as u64),
                diagnostics::bytes_for_f64(upper.len().saturating_add(rhs.len())),
                diagnostics::bytes_for_f64(rhs.len()),
                start.elapsed().as_secs_f64(),
            );
            Some(out)
        }
        None => {
            diagnostics::log_runtime_cpu(
                "trsm_upper",
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

fn with_runtime<T>(
    mut f: impl FnMut(&mut CublasRuntime) -> Option<T>,
) -> Option<(T, GpuDeviceInfo)> {
    static RUNTIME: OnceLock<Vec<Mutex<CublasRuntime>>> = OnceLock::new();
    let runtimes = RUNTIME.get_or_init(|| {
        GpuRuntime::global()
            .devices()
            .iter()
            .filter_map(|device| {
                let cuda = match CudaWorkingState::init(device.ordinal) {
                    Some(cuda) => cuda,
                    None => {
                        diagnostics::log_library_unavailable(
                            "cuBLAS",
                            &format!("CUDA context init failed for device {}", device.ordinal),
                        );
                        return None;
                    }
                };
                match CublasRuntime::new(cuda, device.clone()) {
                    Ok(runtime) => {
                        diagnostics::log_library_ready("cuBLAS", &runtime.device);
                        Some(Mutex::new(runtime))
                    }
                    Err(err) => {
                        diagnostics::log_library_unavailable("cuBLAS", &err);
                        None
                    }
                }
            })
            .collect()
    });
    if runtimes.is_empty() {
        return None;
    }
    let start = GpuRuntime::global().next_runtime_slot(runtimes.len());
    for offset in 0..runtimes.len() {
        let idx = (start + offset) % runtimes.len();
        if let Ok(mut runtime) = runtimes[idx].lock()
            && let Some(out) = f(&mut runtime)
        {
            return Some((out, runtime.device.clone()));
        }
    }
    None
}

struct CublasRuntime {
    /// Borrowed driver + context owned by [`GpuRuntime`].
    cuda: CudaWorkingState,
    device: GpuDeviceInfo,
    /// cuBLAS entry points. The dlopen'd library is `Box::leak`'d inside
    /// `load_static_library`, so these fn pointers stay valid for the
    /// process — no owning field needed.
    blas: CublasApi,
    handle: usize,
}

impl CublasRuntime {
    fn new(cuda: CudaWorkingState, device: GpuDeviceInfo) -> Result<Self, String> {
        let cublas_lib = load_static_library(cublas_library_candidates())?;
        let blas = CublasApi::load(cublas_lib)?;
        cuda.set_current()?;
        let mut handle = 0_usize;
        let create_status = unsafe { (blas.cublas_create)(&mut handle) };
        if create_status != CUBLAS_STATUS_SUCCESS {
            return Err(format!("cublasCreate failed with status {create_status}"));
        }
        Ok(Self {
            cuda,
            device,
            blas,
            handle,
        })
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

        let a_host = to_col_major(a);
        let b_host = to_col_major(b);
        let mut c_host = vec![0.0; m.checked_mul(n)?];
        let bytes_a = bytes_len::<f64>(a_host.len())?;
        let bytes_b = bytes_len::<f64>(b_host.len())?;
        let bytes_c = bytes_len::<f64>(c_host.len())?;

        unsafe {
            self.cuda.set_current().ok()?;
            let a_dev = self.alloc_copy(&a_host, bytes_a)?;
            let b_dev = self.alloc_copy(&b_host, bytes_b)?;
            let c_dev = DeviceAllocation::new(&self.cuda.api, bytes_c)?;
            let alpha = 1.0;
            let beta = 0.0;
            let status = (self.blas.cublas_dgemm)(
                self.handle,
                cublas_op(transpose_a),
                cublas_op(transpose_b),
                to_i32(m)?,
                to_i32(n)?,
                to_i32(k)?,
                &alpha,
                a_dev.ptr,
                to_i32(a_rows)?,
                b_dev.ptr,
                to_i32(b_rows)?,
                &beta,
                c_dev.ptr,
                to_i32(m)?,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return None;
            }
            check_cuda(
                (self.cuda.api.cu_memcpy_dtoh)(c_host.as_mut_ptr().cast(), c_dev.ptr, bytes_c),
                "cuMemcpyDtoH",
            )
            .ok()?;
        }
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
        let a_host = to_col_major(a);
        let x_host = v.to_vec();
        let mut y_host = vec![0.0; out_len];
        let bytes_a = bytes_len::<f64>(a_host.len())?;
        let bytes_x = bytes_len::<f64>(x_host.len())?;
        let bytes_y = bytes_len::<f64>(y_host.len())?;

        unsafe {
            self.cuda.set_current().ok()?;
            let a_dev = self.alloc_copy(&a_host, bytes_a)?;
            let x_dev = self.alloc_copy(&x_host, bytes_x)?;
            let y_dev = DeviceAllocation::new(&self.cuda.api, bytes_y)?;
            let alpha = 1.0;
            let beta = 0.0;
            let status = (self.blas.cublas_dgemv)(
                self.handle,
                cublas_op(transpose),
                to_i32(rows)?,
                to_i32(cols)?,
                &alpha,
                a_dev.ptr,
                to_i32(rows)?,
                x_dev.ptr,
                1,
                &beta,
                y_dev.ptr,
                1,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return None;
            }
            check_cuda(
                (self.cuda.api.cu_memcpy_dtoh)(y_host.as_mut_ptr().cast(), y_dev.ptr, bytes_y),
                "cuMemcpyDtoH",
            )
            .ok()?;
        }
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
        let x_host = to_col_major(x);
        let y_host = to_col_major(y);
        let w_host = w.to_vec();
        let mut out_host = vec![0.0; x_cols.checked_mul(y_cols)?];
        let bytes_x = bytes_len::<f64>(x_host.len())?;
        let bytes_y = bytes_len::<f64>(y_host.len())?;
        let bytes_w = bytes_len::<f64>(w_host.len())?;
        let bytes_out = bytes_len::<f64>(out_host.len())?;

        unsafe {
            self.cuda.set_current().ok()?;
            let x_dev = self.alloc_copy(&x_host, bytes_x)?;
            let y_dev = self.alloc_copy(&y_host, bytes_y)?;
            let w_dev = self.alloc_copy(&w_host, bytes_w)?;
            let wy_dev = DeviceAllocation::new(&self.cuda.api, bytes_y)?;
            let scale_status = (self.blas.cublas_ddgmm)(
                self.handle,
                CUBLAS_SIDE_LEFT,
                to_i32(rows)?,
                to_i32(y_cols)?,
                y_dev.ptr,
                to_i32(rows)?,
                w_dev.ptr,
                1,
                wy_dev.ptr,
                to_i32(rows)?,
            );
            if scale_status != CUBLAS_STATUS_SUCCESS {
                return None;
            }
            let out_dev = DeviceAllocation::new(&self.cuda.api, bytes_out)?;
            let alpha = 1.0;
            let beta = 0.0;
            let status = (self.blas.cublas_dgemm)(
                self.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                to_i32(x_cols)?,
                to_i32(y_cols)?,
                to_i32(rows)?,
                &alpha,
                x_dev.ptr,
                to_i32(rows)?,
                wy_dev.ptr,
                to_i32(rows)?,
                &beta,
                out_dev.ptr,
                to_i32(x_cols)?,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return None;
            }
            check_cuda(
                (self.cuda.api.cu_memcpy_dtoh)(
                    out_host.as_mut_ptr().cast(),
                    out_dev.ptr,
                    bytes_out,
                ),
                "cuMemcpyDtoH",
            )
            .ok()?;
        }
        Some(from_col_major(&out_host, x_cols, y_cols))
    }

    fn trsm<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
        &mut self,
        triangular: &ArrayBase<S1, Ix2>,
        rhs: &ArrayBase<S2, Ix2>,
        uplo: i32,
    ) -> Option<Array2<f64>> {
        let p = triangular.nrows();
        let rhs_cols = rhs.ncols();
        let triangular_host = to_col_major(triangular);
        let mut rhs_host = to_col_major(rhs);
        let bytes_triangular = bytes_len::<f64>(triangular_host.len())?;
        let bytes_rhs = bytes_len::<f64>(rhs_host.len())?;

        unsafe {
            self.cuda.set_current().ok()?;
            let triangular_dev = self.alloc_copy(&triangular_host, bytes_triangular)?;
            let rhs_dev = self.alloc_copy(&rhs_host, bytes_rhs)?;
            let alpha = 1.0;
            let status = (self.blas.cublas_dtrsm)(
                self.handle,
                CUBLAS_SIDE_LEFT,
                uplo,
                CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT,
                to_i32(p)?,
                to_i32(rhs_cols)?,
                &alpha,
                triangular_dev.ptr,
                to_i32(p)?,
                rhs_dev.ptr,
                to_i32(p)?,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return None;
            }
            check_cuda(
                (self.cuda.api.cu_memcpy_dtoh)(
                    rhs_host.as_mut_ptr().cast(),
                    rhs_dev.ptr,
                    bytes_rhs,
                ),
                "cuMemcpyDtoH trsm",
            )
            .ok()?;
        }
        Some(from_col_major(&rhs_host, p, rhs_cols))
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
        let a_total = batch.checked_mul(a_stride)?;
        let b_total = batch.checked_mul(b_stride)?;
        let c_total = batch.checked_mul(c_stride)?;

        // Pack each batch element in column-major order. cuBLAS strided
        // batched expects all elements contiguous in memory with a fixed
        // stride between successive batch slices.
        let mut a_host: Vec<f64> = Vec::with_capacity(a_total);
        for batch_idx in 0..batch {
            let slice = a.slice(s![batch_idx, .., ..]);
            for col in 0..a_cols {
                for row in 0..a_rows {
                    a_host.push(slice[[row, col]]);
                }
            }
        }
        let mut b_host: Vec<f64> = Vec::with_capacity(b_total);
        for batch_idx in 0..batch {
            let slice = b.slice(s![batch_idx, .., ..]);
            for col in 0..b_cols {
                for row in 0..b_rows {
                    b_host.push(slice[[row, col]]);
                }
            }
        }
        let mut c_host: Vec<f64> = vec![0.0; c_total];
        let bytes_a = bytes_len::<f64>(a_host.len())?;
        let bytes_b = bytes_len::<f64>(b_host.len())?;
        let bytes_c = bytes_len::<f64>(c_host.len())?;

        unsafe {
            self.cuda.set_current().ok()?;
            let a_dev = self.alloc_copy(&a_host, bytes_a)?;
            let b_dev = self.alloc_copy(&b_host, bytes_b)?;
            let c_dev = DeviceAllocation::new(&self.cuda.api, bytes_c)?;
            let alpha = 1.0_f64;
            let beta = 0.0_f64;
            let status = (self.blas.cublas_dgemm_strided_batched)(
                self.handle,
                cublas_op(transpose_a),
                cublas_op(transpose_b),
                to_i32(m)?,
                to_i32(n)?,
                to_i32(k)?,
                &alpha,
                a_dev.ptr,
                to_i32(a_rows)?,
                to_i64(a_stride)?,
                b_dev.ptr,
                to_i32(b_rows)?,
                to_i64(b_stride)?,
                &beta,
                c_dev.ptr,
                to_i32(m)?,
                to_i64(c_stride)?,
                to_i32(batch)?,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return None;
            }
            check_cuda(
                (self.cuda.api.cu_memcpy_dtoh)(c_host.as_mut_ptr().cast(), c_dev.ptr, bytes_c),
                "cuMemcpyDtoH strided batched C",
            )
            .ok()?;
        }

        // Unpack column-major slabs back into row-major Array3.
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

    /// Variant of [`gemm_strided_batched`] that broadcasts `B` (a single
    /// matrix) over all K batch elements of `A`. cuBLAS sees `strideB = 0`
    /// so the B buffer is uploaded exactly once.
    fn gemm_broadcast_b_strided_batched(
        &mut self,
        a: ArrayView3<'_, f64>,
        b: ndarray::ArrayView2<'_, f64>,
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
        let a_total = batch.checked_mul(a_stride)?;
        let b_total = b_rows.checked_mul(b_cols)?;
        let c_total = batch.checked_mul(c_stride)?;

        let mut a_host: Vec<f64> = Vec::with_capacity(a_total);
        for batch_idx in 0..batch {
            let slice = a.slice(s![batch_idx, .., ..]);
            for col in 0..a_cols {
                for row in 0..a_rows {
                    a_host.push(slice[[row, col]]);
                }
            }
        }
        let mut b_host: Vec<f64> = Vec::with_capacity(b_total);
        for col in 0..b_cols {
            for row in 0..b_rows {
                b_host.push(b[[row, col]]);
            }
        }
        let mut c_host: Vec<f64> = vec![0.0; c_total];
        let bytes_a = bytes_len::<f64>(a_host.len())?;
        let bytes_b = bytes_len::<f64>(b_host.len())?;
        let bytes_c = bytes_len::<f64>(c_host.len())?;

        unsafe {
            self.cuda.set_current().ok()?;
            let a_dev = self.alloc_copy(&a_host, bytes_a)?;
            let b_dev = self.alloc_copy(&b_host, bytes_b)?;
            let c_dev = DeviceAllocation::new(&self.cuda.api, bytes_c)?;
            let alpha = 1.0_f64;
            let beta = 0.0_f64;
            let status = (self.blas.cublas_dgemm_strided_batched)(
                self.handle,
                cublas_op(transpose_a),
                cublas_op(transpose_b),
                to_i32(m)?,
                to_i32(n)?,
                to_i32(k)?,
                &alpha,
                a_dev.ptr,
                to_i32(a_rows)?,
                to_i64(a_stride)?,
                b_dev.ptr,
                to_i32(b_rows)?,
                0_i64, // strideB = 0 => broadcast B across K batches
                &beta,
                c_dev.ptr,
                to_i32(m)?,
                to_i64(c_stride)?,
                to_i32(batch)?,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return None;
            }
            check_cuda(
                (self.cuda.api.cu_memcpy_dtoh)(c_host.as_mut_ptr().cast(), c_dev.ptr, bytes_c),
                "cuMemcpyDtoH broadcast-B strided batched C",
            )
            .ok()?;
        }
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

    /// Mirror of [`gemm_broadcast_b_strided_batched`] for the symmetric
    /// step where `A` is broadcast and `B` is strided per batch.
    fn broadcast_a_gemm_strided_batched(
        &mut self,
        a: ndarray::ArrayView2<'_, f64>,
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
        let a_total = a_rows.checked_mul(a_cols)?;
        let b_total = batch.checked_mul(b_stride)?;
        let c_total = batch.checked_mul(c_stride)?;

        let mut a_host: Vec<f64> = Vec::with_capacity(a_total);
        for col in 0..a_cols {
            for row in 0..a_rows {
                a_host.push(a[[row, col]]);
            }
        }
        let mut b_host: Vec<f64> = Vec::with_capacity(b_total);
        for batch_idx in 0..batch {
            let slice = b.slice(s![batch_idx, .., ..]);
            for col in 0..b_cols {
                for row in 0..b_rows {
                    b_host.push(slice[[row, col]]);
                }
            }
        }
        let mut c_host: Vec<f64> = vec![0.0; c_total];
        let bytes_a = bytes_len::<f64>(a_host.len())?;
        let bytes_b = bytes_len::<f64>(b_host.len())?;
        let bytes_c = bytes_len::<f64>(c_host.len())?;

        unsafe {
            self.cuda.set_current().ok()?;
            let a_dev = self.alloc_copy(&a_host, bytes_a)?;
            let b_dev = self.alloc_copy(&b_host, bytes_b)?;
            let c_dev = DeviceAllocation::new(&self.cuda.api, bytes_c)?;
            let alpha = 1.0_f64;
            let beta = 0.0_f64;
            let status = (self.blas.cublas_dgemm_strided_batched)(
                self.handle,
                cublas_op(transpose_a),
                cublas_op(transpose_b),
                to_i32(m)?,
                to_i32(n)?,
                to_i32(k)?,
                &alpha,
                a_dev.ptr,
                to_i32(a_rows)?,
                0_i64, // strideA = 0 => broadcast A across K batches
                b_dev.ptr,
                to_i32(b_rows)?,
                to_i64(b_stride)?,
                &beta,
                c_dev.ptr,
                to_i32(m)?,
                to_i64(c_stride)?,
                to_i32(batch)?,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return None;
            }
            check_cuda(
                (self.cuda.api.cu_memcpy_dtoh)(c_host.as_mut_ptr().cast(), c_dev.ptr, bytes_c),
                "cuMemcpyDtoH broadcast-A strided batched C",
            )
            .ok()?;
        }
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

    unsafe fn alloc_copy<'a>(
        &'a self,
        values: &[f64],
        bytes: usize,
    ) -> Option<DeviceAllocation<'a>> {
        let allocation = unsafe { DeviceAllocation::new(&self.cuda.api, bytes) }?;
        check_cuda(
            unsafe {
                (self.cuda.api.cu_memcpy_htod)(allocation.ptr, values.as_ptr().cast(), bytes)
            },
            "cuMemcpyHtoD",
        )
        .ok()?;
        Some(allocation)
    }
}

impl Drop for CublasRuntime {
    fn drop(&mut self) {
        // The cuBLAS handle is library-local; destroying it here releases
        // its small descriptor buffers. The CUDA context itself is owned
        // by the shared [`CudaWorkingState`] and tears down once at
        // process exit, not on every library runtime drop.
        unsafe {
            let _ = self.cuda.set_current();
            let _ = (self.blas.cublas_destroy)(self.handle);
        }
    }
}

type CublasStatus = i32;
type CublasCreate = unsafe extern "C" fn(*mut usize) -> CublasStatus;
type CublasDestroy = unsafe extern "C" fn(usize) -> CublasStatus;
type CublasDgemm = unsafe extern "C" fn(
    usize,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f64,
    u64,
    i32,
    u64,
    i32,
    *const f64,
    u64,
    i32,
) -> CublasStatus;
type CublasDgemv = unsafe extern "C" fn(
    usize,
    i32,
    i32,
    i32,
    *const f64,
    u64,
    i32,
    u64,
    i32,
    *const f64,
    u64,
    i32,
) -> CublasStatus;
type CublasDdgmm =
    unsafe extern "C" fn(usize, i32, i32, i32, u64, i32, u64, i32, u64, i32) -> CublasStatus;
type CublasDtrsm = unsafe extern "C" fn(
    usize,
    i32,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f64,
    u64,
    i32,
    u64,
    i32,
) -> CublasStatus;
type CublasDgemmStridedBatched = unsafe extern "C" fn(
    usize,      // handle
    i32,        // transa
    i32,        // transb
    i32,        // m
    i32,        // n
    i32,        // k
    *const f64, // alpha
    u64,        // A
    i32,        // lda
    i64,        // strideA
    u64,        // B
    i32,        // ldb
    i64,        // strideB
    *const f64, // beta
    u64,        // C
    i32,        // ldc
    i64,        // strideC
    i32,        // batchCount
) -> CublasStatus;

struct CublasApi {
    cublas_create: CublasCreate,
    cublas_destroy: CublasDestroy,
    cublas_dgemm: CublasDgemm,
    cublas_dgemv: CublasDgemv,
    cublas_ddgmm: CublasDdgmm,
    cublas_dtrsm: CublasDtrsm,
    cublas_dgemm_strided_batched: CublasDgemmStridedBatched,
}

impl CublasApi {
    fn load(library: &Library) -> Result<Self, String> {
        unsafe {
            Ok(Self {
                cublas_create: *library
                    .get(b"cublasCreate_v2\0")
                    .map_err(|e| e.to_string())?,
                cublas_destroy: *library
                    .get(b"cublasDestroy_v2\0")
                    .map_err(|e| e.to_string())?,
                cublas_dgemm: *library
                    .get(b"cublasDgemm_v2\0")
                    .map_err(|e| e.to_string())?,
                cublas_dgemv: *library
                    .get(b"cublasDgemv_v2\0")
                    .map_err(|e| e.to_string())?,
                cublas_ddgmm: *library.get(b"cublasDdgmm\0").map_err(|e| e.to_string())?,
                cublas_dtrsm: *library
                    .get(b"cublasDtrsm_v2\0")
                    .map_err(|e| e.to_string())?,
                cublas_dgemm_strided_batched: *library
                    .get(b"cublasDgemmStridedBatched\0")
                    .map_err(|e| e.to_string())?,
            })
        }
    }
}

const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const CUBLAS_SIDE_LEFT: i32 = 0;
const CUBLAS_FILL_LOWER: i32 = 0;
const CUBLAS_FILL_UPPER: i32 = 1;
const CUBLAS_DIAG_NON_UNIT: i32 = 0;

#[inline]
fn cublas_op(transpose: bool) -> i32 {
    if transpose { CUBLAS_OP_T } else { CUBLAS_OP_N }
}

fn cublas_library_candidates() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["cublas64_12.dll", "cublas64_11.dll"]
    } else if cfg!(target_os = "macos") {
        &["/usr/local/cuda/lib/libcublas.dylib", "libcublas.dylib"]
    } else {
        &["libcublas.so.12", "libcublas.so.11", "libcublas.so"]
    }
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

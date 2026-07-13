use crate::faer_ndarray::{
    CrossprodAccum, CrossprodStructure, FaerArrayView, array2_to_matmut,
    effective_global_parallelism, fast_ab, fast_atb, fast_atv, fast_atv_into, fast_av,
    fast_av_into, fast_xt_diag_x, stream_weighted_crossprod_into,
};
use crate::types::RidgePolicy;
use faer::Accum;
use faer::linalg::matmul::matmul;
use faer::sparse::{SparseColMat, SparseRowMat, Triplet};
use gam_runtime::resource::{
    Governed, MaterializationPolicy, MatrixMaterializationError, MemoryGovernor, MemoryReservation,
    ResourcePolicy, dense_f64_bytes, rows_for_target_bytes,
};
use ndarray::{
    Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis, ShapeBuilder, s,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::ops::Deref;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

const MATRIX_FREE_PCG_MIN_P: usize = 2048;
const MATRIX_FREE_PCG_REL_TOL: f64 = 1e-8;
/// Minimum numerical ridge added to the (penalized) normal matrix before an SPD
/// solve. Near `f64` precision: large enough to lift an exactly-singular system
/// off zero so the factorization succeeds, small enough not to bias a
/// well-conditioned solve. Acts as a floor on any caller-supplied `ridge_floor`.
const MATRIX_FREE_PCG_MAX_ITER: usize = 2000;
const CHUNKED_DENSE_MATERIALIZATION_BYTES: usize = 8 * 1024 * 1024;
const OPERATOR_ROW_CHUNK_SIZE: usize = 256;
/// Minimum n*p product for the dense-row parallel fold/reduce paths
/// (`diag_gram`, `apply_weighted_normal`, dense transpose reductions).
/// Below this, the sequential row loop wins on overhead.
const DENSE_ROW_PARALLEL_MIN_NP: u64 = 200_000;
const WEIGHTED_CROSSPROD_PARALLEL_MIN_FLOPS: u64 = 500_000;
const SPARSE_ROW_PARALLEL_MIN_FLOPS: u64 = 100_000;
/// Maximum bytes for the (n, tail_total) intermediate in GEMM-batched tensor
/// product matvecs.  Beyond this threshold, fall back to per-column GEMV.
const TENSOR_GEMM_MAX_INTERMEDIATE_BYTES: usize = 128 * 1024 * 1024; // 128 MB

pub use crate::utils::PcgSolveInfo;

mod sparse_hessian;
pub use sparse_hessian::SparseHessianAccumulator;

mod weights;
pub use weights::{FiniteSignedWeightsView, PsdWeightsView, SignedWeightsArc, SignedWeightsView};

/// Typed error for `src/linalg/matrix.rs` operations.  All error sites in this
/// module construct a `MatrixError` variant; trait method bodies that still
/// return `Result<_, String>` convert via `From<MatrixError> for String` (which
/// is byte-equivalent to the prior `format!` / `to_string` payloads).
#[derive(Debug, Clone)]
pub enum MatrixError {
    /// Operand shapes (rows, columns, lengths) do not satisfy the operation's
    /// dimension contract.  Also covers integer-overflow in dimension products.
    DimensionMismatch { reason: String },
    /// Refused to materialize an operator-backed or sparse design to a dense
    /// `Array2<f64>` because the active `ResourcePolicy` (size cap or strict
    /// operator-only mode) forbids it.
    DensificationRefused { reason: String },
}

crate::impl_reason_error_boilerplate! {
    MatrixError {
        DimensionMismatch,
        DensificationRefused,
    }
}

#[inline]
fn dense_materialization_chunk_rows(nrows: usize, ncols: usize) -> usize {
    rows_for_target_bytes(CHUNKED_DENSE_MATERIALIZATION_BYTES, ncols)
        .max(1)
        .min(nrows.max(1))
}

fn dense_operator_to_dense_by_chunks<O: DenseDesignOperator + ?Sized>(
    op: &O,
) -> Result<Array2<f64>, MatrixMaterializationError> {
    let n = op.nrows();
    let p = op.ncols();
    let chunk_rows = dense_materialization_chunk_rows(n, p);
    let mut out = Array2::<f64>::zeros((n, p));
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        let slice = out.slice_mut(s![start..end, ..]);
        op.row_chunk_into(start..end, slice)?;
    }
    Ok(out)
}

/// Fallible full materialization whose process-wide reservation lives exactly
/// as long as the returned matrix.
fn governed_dense_operator_to_dense_by_chunks<O: DenseDesignOperator + ?Sized>(
    op: &O,
    policy: &MaterializationPolicy,
    context: &'static str,
) -> Result<Governed<Array2<f64>>, MatrixMaterializationError> {
    let effective_policy =
        merge_operator_materialization_policies(Some(policy.clone()), op.materialization_policy())
            .expect("caller policy is always present");
    if !effective_policy.allow_operator_materialization {
        return Err(MatrixMaterializationError::Forbidden {
            context,
            mode: gam_runtime::resource::DerivativeStorageMode::AnalyticOperatorRequired,
        });
    }
    let bytes = dense_f64_bytes(op.nrows(), op.ncols()).unwrap_or(usize::MAX);
    if bytes > effective_policy.max_single_dense_bytes {
        return Err(MatrixMaterializationError::TooLarge {
            context,
            nrows: op.nrows(),
            ncols: op.ncols(),
            bytes,
            limit_bytes: effective_policy.max_single_dense_bytes,
        });
    }
    let reservation =
        MemoryGovernor::global().try_reserve_dense_f64(op.nrows(), op.ncols(), context)?;
    dense_operator_to_dense_by_chunks(op).map(|matrix| reservation.bind(matrix))
}

pub fn checked_dense_nbytes(nrows: usize, ncols: usize, context: &str) -> Result<usize, String> {
    nrows
        .checked_mul(ncols)
        .and_then(|cells| cells.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| {
            MatrixError::DimensionMismatch {
                reason: format!("{context}: dense size overflow for {nrows}x{ncols}"),
            }
            .into()
        })
}

pub fn panic_or_error_if_large_scale_mode_and_to_dense_called_with_policy(
    context: &str,
    n: usize,
    p: usize,
    policy: &ResourcePolicy,
) -> Result<(), String> {
    // Strict-operator mode: refuse any dense materialization, regardless of
    // size.  Callers in this mode have committed to operator-only math; any
    // dense fallback (cache or otherwise) violates that contract and would
    // silently turn an analytic-operator path into a hidden dense path at
    // large scale.
    if matches!(
        policy.derivative_storage_mode,
        gam_runtime::resource::DerivativeStorageMode::AnalyticOperatorRequired
    ) {
        return Err(MatrixError::DensificationRefused {
            reason: format!(
                "{context}: refusing to densify operator-backed design {n}x{p} under \
             AnalyticOperatorRequired policy; provide an operator-form path"
            ),
        }
        .into());
    }
    let dense_bytes = checked_dense_nbytes(n, p, context)?;
    let limit = policy.max_single_materialization_bytes;
    if dense_bytes > limit {
        let gib = dense_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        return Err(MatrixError::DensificationRefused {
            reason: format!(
                "{context}: refusing to densify operator-backed design {n}x{p} (~{gib:.2} GiB); use matrix-free or chunked code"
            ),
        }
        .into());
    }
    Ok(())
}

fn merge_operator_materialization_policies(
    left: Option<MaterializationPolicy>,
    right: Option<MaterializationPolicy>,
) -> Option<MaterializationPolicy> {
    match (left, right) {
        (None, policy) | (policy, None) => policy,
        (Some(left), Some(right)) => Some(MaterializationPolicy {
            max_single_dense_bytes: left
                .max_single_dense_bytes
                .min(right.max_single_dense_bytes),
            max_cached_dense_bytes: left
                .max_cached_dense_bytes
                .min(right.max_cached_dense_bytes),
            row_chunk_target_bytes: left
                .row_chunk_target_bytes
                .min(right.row_chunk_target_bytes),
            allow_operator_materialization: left.allow_operator_materialization
                && right.allow_operator_materialization,
            allow_diagnostic_materialization: left.allow_diagnostic_materialization
                && right.allow_diagnostic_materialization,
        }),
    }
}

fn enforce_operator_materialization_policy(
    op: &dyn DenseDesignOperator,
    context: &str,
) -> Result<(), String> {
    let Some(policy) = op.materialization_policy() else {
        return Ok(());
    };
    if !policy.allow_operator_materialization {
        return Err(MatrixError::DensificationRefused {
            reason: format!(
                "{context}: refusing to densify {}x{} operator-backed design because its \
                 construction policy requires streamed storage",
                op.nrows(),
                op.ncols(),
            ),
        }
        .into());
    }
    let bytes = checked_dense_nbytes(op.nrows(), op.ncols(), context)?;
    if bytes > policy.max_single_dense_bytes {
        return Err(MatrixError::DensificationRefused {
            reason: format!(
                "{context}: refusing to densify {}x{} operator-backed design ({bytes} bytes); \
                 its construction-policy limit is {} bytes",
                op.nrows(),
                op.ncols(),
                policy.max_single_dense_bytes,
            ),
        }
        .into());
    }
    Ok(())
}

/// Validate a row-weight diagonal before any output buffer is allocated or
/// mutated.  The linear weighted operators in this module are defined for all
/// finite signed weights; `NaN` and infinities have no linear-operator meaning
/// and are rejected at the smallest offending row.
#[inline]
fn certify_signed_weights<'a>(
    context: &str,
    weights: &'a Array1<f64>,
    expected_len: usize,
) -> Result<FiniteSignedWeightsView<'a>, String> {
    if weights.len() != expected_len {
        return Err(MatrixError::DimensionMismatch {
            reason: format!(
                "{context} weight length mismatch: weights={}, nrows={expected_len}",
                weights.len()
            ),
        }
        .into());
    }
    FiniteSignedWeightsView::try_from_array(weights)
        .map_err(|reason| format!("{context}: {reason}"))
}

fn weighted_crossprod_dense(
    left: &Array2<f64>,
    weights: &Array1<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    if left.nrows() != weights.len() || right.nrows() != weights.len() {
        return Err(MatrixError::DimensionMismatch {
            reason: format!(
                "weighted_crossprod_dense row mismatch: left={}, weights={}, right={}",
                left.nrows(),
                weights.len(),
                right.nrows()
            ),
        }
        .into());
    }
    certify_signed_weights("weighted_crossprod_dense", weights, left.nrows())?;
    Ok(weighted_crossprod_dense_view(left, weights.view(), right))
}

fn weighted_crossprod_dense_view(
    left: &Array2<f64>,
    weights: ArrayView1<'_, f64>,
    right: &Array2<f64>,
) -> Array2<f64> {
    let n = weights.len();
    let p_left = left.ncols();
    let p_right = right.ncols();
    let work = (n as u64)
        .saturating_mul(p_left as u64)
        .saturating_mul(p_right as u64);
    if rayon::current_num_threads() <= 1 || work < WEIGHTED_CROSSPROD_PARALLEL_MIN_FLOPS {
        return weighted_crossprod_dense_rows(left, weights, right, 0..n);
    }

    let min_parallel_work = WEIGHTED_CROSSPROD_PARALLEL_MIN_FLOPS.min(usize::MAX as u64) as usize;
    let Some(chunk_rows) = crate::parallel::row_reduction_chunk_rows(
        n,
        p_left.saturating_mul(p_right),
        p_left.saturating_mul(p_right),
        min_parallel_work,
    ) else {
        return weighted_crossprod_dense_rows(left, weights, right, 0..n);
    };
    let starts: Vec<usize> = (0..n).step_by(chunk_rows).collect();
    let partials: Vec<Array2<f64>> = starts
        .into_par_iter()
        .map(|start| {
            weighted_crossprod_dense_rows(left, weights, right, start..(start + chunk_rows).min(n))
        })
        .collect();
    let mut out = Array2::<f64>::zeros((p_left, p_right));
    for partial in &partials {
        out += partial;
    }
    out
}

fn weighted_crossprod_dense_rows(
    left: &Array2<f64>,
    weights: ArrayView1<'_, f64>,
    right: &Array2<f64>,
    rows: Range<usize>,
) -> Array2<f64> {
    // The per-row body below is `Σᵢ wᵢ · leftᵢᵀ · rightᵢ`, which is linear in
    // `wᵢ` and therefore sign-correct without any PSD assumption. The PSD
    // precondition belongs at the symmetric `Xᵀ W X` caller (`weighted_crossprod_dense_view`),
    // not at this kernel: `BlockDesignOperator::cross_block` legitimately uses
    // the asymmetric form `X_iᵀ W X_j` with signed `c·Xv` weights from the outer
    // REML Hessian-derivative correction, which is not PSD even when `w ≥ 0`.
    // The prior assert here turned that legitimate signed use into a panic.
    let p_left = left.ncols();
    let p_right = right.ncols();
    let mut out = Array2::<f64>::zeros((p_left, p_right));
    if left.is_standard_layout()
        && right.is_standard_layout()
        && let (Some(lx), Some(rx), Some(w)) =
            (left.as_slice(), right.as_slice(), weights.as_slice())
    {
        let out_slice = out.as_slice_mut().expect("zeros are contiguous");
        for i in rows {
            let wi = w[i];
            if wi == 0.0 {
                continue;
            }
            let l_row = &lx[i * p_left..i * p_left + p_left];
            let r_row = &rx[i * p_right..i * p_right + p_right];
            for a in 0..p_left {
                let scaled = wi * l_row[a];
                if scaled == 0.0 {
                    continue;
                }
                let out_row = &mut out_slice[a * p_right..a * p_right + p_right];
                for b in 0..p_right {
                    out_row[b] += scaled * r_row[b];
                }
            }
        }
        return out;
    }
    for i in rows {
        let wi = weights[i];
        if wi == 0.0 {
            continue;
        }
        for a in 0..p_left {
            let scaled = wi * left[[i, a]];
            if scaled == 0.0 {
                continue;
            }
            for b in 0..p_right {
                out[[a, b]] += scaled * right[[i, b]];
            }
        }
    }
    out
}

pub struct DenseRightProductView<'a> {
    base: &'a Array2<f64>,
    first: Option<&'a Array2<f64>>,
    second: Option<&'a Array2<f64>>,
}

impl<'a> DenseRightProductView<'a> {
    pub fn new(base: &'a Array2<f64>) -> Self {
        Self {
            base,
            first: None,
            second: None,
        }
    }

    pub fn with_factor(mut self, factor: &'a Array2<f64>) -> Self {
        if self.first.is_none() {
            self.first = Some(factor);
        } else if self.second.is_none() {
            self.second = Some(factor);
        } else {
            // SAFETY: DenseRightProductView statically carries exactly two optional
            // factor slots (`first` and `second`); reaching this branch means a
            // caller invoked `with_factor` a third time, which violates the
            // type's documented contract of at most two right factors.
            // SAFETY: third `with_factor` call violates the type's two-factor invariant.
            std::panic::panic_any("DenseRightProductView supports at most two right factors");
        }
        self
    }

    pub fn with_optional_factor(self, factor: Option<&'a Array2<f64>>) -> Self {
        match factor {
            Some(factor) => self.with_factor(factor),
            None => self,
        }
    }

    pub fn materialize(&self) -> Array2<f64> {
        let mut out = self.base.clone();
        if let Some(factor) = self.first {
            out = fast_ab(&out, factor);
        }
        if let Some(factor) = self.second {
            out = fast_ab(&out, factor);
        }
        out
    }

    fn transformed_ncols(&self) -> usize {
        if let Some(factor) = self.second {
            factor.ncols()
        } else if let Some(factor) = self.first {
            factor.ncols()
        } else {
            self.base.ncols()
        }
    }
}

pub struct EmbeddedColumnBlock<'a> {
    local: &'a Array2<f64>,
    global_range: Range<usize>,
    total_cols: usize,
}

impl<'a> EmbeddedColumnBlock<'a> {
    pub fn new(local: &'a Array2<f64>, global_range: Range<usize>, total_cols: usize) -> Self {
        Self {
            local,
            global_range,
            total_cols,
        }
    }

    pub fn materialize(&self) -> Array2<f64> {
        if self.local.nrows() == 0 {
            return Array2::<f64>::zeros((0, self.total_cols));
        }
        assert_eq!(
            self.local.ncols(),
            self.global_range.len(),
            "embedded column block width mismatch"
        );
        let mut out = Array2::<f64>::zeros((self.local.nrows(), self.total_cols));
        out.slice_mut(ndarray::s![.., self.global_range.clone()])
            .assign(self.local);
        out
    }
}

pub struct EmbeddedSquareBlock<'a> {
    local: &'a Array2<f64>,
    global_range: Range<usize>,
    total_dim: usize,
}

impl<'a> EmbeddedSquareBlock<'a> {
    pub fn new(local: &'a Array2<f64>, global_range: Range<usize>, total_dim: usize) -> Self {
        Self {
            local,
            global_range,
            total_dim,
        }
    }

    pub fn materialize(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.total_dim, self.total_dim));
        out.slice_mut(ndarray::s![
            self.global_range.clone(),
            self.global_range.clone()
        ])
        .assign(self.local);
        out
    }
}

struct PenalizedWeightedNormalOperator<'a, O: LinearOperator + ?Sized> {
    operator: &'a O,
    weights: &'a Array1<f64>,
    finite_weights: FiniteSignedWeightsView<'a>,
    penalty: Option<&'a Array2<f64>>,
    ridge: f64,
}

impl<'a, O: LinearOperator + ?Sized> PenalizedWeightedNormalOperator<'a, O> {
    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.operator
            .apply_weighted_normal(self.finite_weights, vector, self.penalty, self.ridge)
    }

    fn jacobi_preconditioner(&self) -> Result<Array1<f64>, String> {
        let mut diag = self.operator.diag_gram(self.weights)?;
        if let Some(pen) = self.penalty {
            for i in 0..diag.len() {
                diag[i] += pen[[i, i]];
            }
        }
        if self.ridge > 0.0 {
            for i in 0..diag.len() {
                diag[i] += self.ridge;
            }
        }
        Ok(diag)
    }
}

#[inline]
fn dense_diag_gram_view(matrix: &Array2<f64>, weights: ArrayView1<'_, f64>) -> Array1<f64> {
    // Exact diagonal of Xᵀdiag(w)X.  It is linear in w and therefore retains
    // signed observed curvature; solver-level stabilization decides whether a
    // resulting global system is suitable for Cholesky/PCG.
    let p = matrix.ncols();
    let n = matrix.nrows();
    let large = (n as u64) * (p as u64) >= DENSE_ROW_PARALLEL_MIN_NP;
    let parallel = large && rayon::current_thread_index().is_none();
    // Fast path: if the matrix is row-major contiguous, read each row as a
    // slice and avoid n*p bounds-checked indexing.
    if matrix.is_standard_layout()
        && let (Some(x), Some(w)) = (matrix.as_slice(), weights.as_slice())
    {
        if parallel {
            // Deterministic parallel row reduction: length-only pairwise tree
            // so the accumulated float result never depends on thread count or
            // rayon's demand-driven fold/reduce grouping (#2228).
            return crate::pairwise_reduce::par_deterministic_block_fold(
                n,
                |range: core::ops::Range<usize>| {
                    let mut acc = vec![0.0_f64; p];
                    for i in range {
                        let wi = w[i];
                        if wi != 0.0 {
                            let row = &x[i * p..i * p + p];
                            for j in 0..p {
                                let xij = row[j];
                                acc[j] += wi * xij * xij;
                            }
                        }
                    }
                    acc
                },
                |mut a, b| {
                    for (av, bv) in a.iter_mut().zip(b) {
                        *av += bv;
                    }
                    a
                },
            )
            .unwrap_or_else(|| vec![0.0_f64; p])
            .into();
        }
        let mut diag = Array1::<f64>::zeros(p);
        let diag_slice = diag.as_slice_mut().expect("zeros are contiguous");
        for i in 0..n {
            let wi = w[i];
            if wi == 0.0 {
                continue;
            }
            let row = &x[i * p..i * p + p];
            for j in 0..p {
                let xij = row[j];
                diag_slice[j] += wi * xij * xij;
            }
        }
        return diag;
    }
    let mut diag = Array1::<f64>::zeros(p);
    for i in 0..n {
        let wi = weights[i];
        if wi == 0.0 {
            continue;
        }
        for j in 0..p {
            let xij = matrix[[i, j]];
            diag[j] += wi * xij * xij;
        }
    }
    diag
}

fn sparse_csr_weighted_xtwx(
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &[f64],
    n: usize,
    p: usize,
    weights: ArrayView1<'_, f64>,
) -> Array2<f64> {
    let nnz = vals.len() as u64;
    let avg = nnz.checked_div(n.max(1) as u64).unwrap_or(0);
    let work = (n as u64).saturating_mul(avg.saturating_mul(avg));
    if rayon::current_num_threads() <= 1 || work < SPARSE_ROW_PARALLEL_MIN_FLOPS {
        return sparse_csr_weighted_xtwx_rows(row_ptr, col_idx, vals, p, weights, 0..n);
    }

    let min_parallel_work = SPARSE_ROW_PARALLEL_MIN_FLOPS.min(usize::MAX as u64) as usize;
    let Some(chunk_rows) = crate::parallel::row_reduction_chunk_rows(
        n,
        avg.min(usize::MAX as u64) as usize,
        p.saturating_mul(p),
        min_parallel_work,
    ) else {
        return sparse_csr_weighted_xtwx_rows(row_ptr, col_idx, vals, p, weights, 0..n);
    };
    let starts: Vec<usize> = (0..n).step_by(chunk_rows).collect();
    let partials: Vec<Array2<f64>> = starts
        .into_par_iter()
        .map(|start| {
            sparse_csr_weighted_xtwx_rows(
                row_ptr,
                col_idx,
                vals,
                p,
                weights,
                start..(start + chunk_rows).min(n),
            )
        })
        .collect();
    let mut xtwx = Array2::<f64>::zeros((p, p));
    for partial in &partials {
        xtwx += partial;
    }
    xtwx
}

fn sparse_csr_weighted_xtwx_rows(
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &[f64],
    p: usize,
    weights: ArrayView1<'_, f64>,
    rows: Range<usize>,
) -> Array2<f64> {
    // PSD precondition is discharged at the typed boundary
    // (`PsdWeightsView::try_new` inside callers of `xt_diag_x_psd_op`). The CSC
    // counterpart (`streaming_sparse_csc_xt_diag_x`) accepts signed weights and
    // is the right path for observed-Hessian assembly; this CSR-row kernel is
    // reserved for Fisher-scoring Gram builds where the working weights are
    // guaranteed nonneg by typed construction.
    let mut xtwx = Array2::<f64>::zeros((p, p));
    for i in rows {
        let wi = weights[i];
        if wi == 0.0 {
            continue;
        }
        let start = row_ptr[i];
        let end = row_ptr[i + 1];
        for a_ptr in start..end {
            let a = col_idx[a_ptr];
            let wxa = wi * vals[a_ptr];
            for b_ptr in a_ptr..end {
                let b = col_idx[b_ptr];
                let v = wxa * vals[b_ptr];
                xtwx[[a, b]] += v;
                if a != b {
                    xtwx[[b, a]] += v;
                }
            }
        }
    }
    xtwx
}

pub fn streaming_sparse_csc_xt_diag_x(
    col_ptr: &[usize],
    row_idx: &[usize],
    vals: &[f64],
    n: usize,
    p: usize,
    weights: ArrayView1<'_, f64>,
    out: &mut Array2<f64>,
) {
    if n == 0 || p == 0 {
        return;
    }

    let chunk_rows = dense_materialization_chunk_rows(n, p);
    let par = effective_global_parallelism();
    let mut x_chunk = Array2::<f64>::zeros((chunk_rows, p).f());
    let mut wx_chunk = Array2::<f64>::zeros((chunk_rows, p).f());

    {
        let mut out_view = array2_to_matmut(out);

        for start in (0..n).step_by(chunk_rows) {
            let rows = (n - start).min(chunk_rows);
            {
                let mut x_slice = x_chunk.slice_mut(s![0..rows, ..]);
                let mut wx_slice = wx_chunk.slice_mut(s![0..rows, ..]);
                x_slice.fill(0.0);
                wx_slice.fill(0.0);
                let end = start + rows;
                for col in 0..p {
                    let col_start = col_ptr[col];
                    let col_end = col_ptr[col + 1];
                    let rows_for_col = &row_idx[col_start..col_end];
                    let local_start = rows_for_col.partition_point(|&row| row < start);
                    let local_end = rows_for_col.partition_point(|&row| row < end);
                    for local_ptr in local_start..local_end {
                        let ptr = col_start + local_ptr;
                        let row = row_idx[ptr];
                        let local = row - start;
                        let wi = weights[row];
                        let value = vals[ptr];
                        x_slice[[local, col]] += value;
                        wx_slice[[local, col]] += wi * value;
                    }
                }
            }
            let x_slice = x_chunk.slice(s![0..rows, ..]);
            let wx_slice = wx_chunk.slice(s![0..rows, ..]);
            let x_view = FaerArrayView::new(&x_slice);
            let wx_view = FaerArrayView::new(&wx_slice);
            matmul(
                out_view.as_mut(),
                Accum::Add,
                x_view.as_ref().transpose(),
                wx_view.as_ref(),
                1.0,
                par,
            );
        }
    }
}

fn sparse_csr_diag_gram(
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &[f64],
    n: usize,
    p: usize,
    weights: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let work = vals.len() as u64;
    if rayon::current_num_threads() <= 1 || work < SPARSE_ROW_PARALLEL_MIN_FLOPS {
        return sparse_csr_diag_gram_rows(row_ptr, col_idx, vals, p, weights, 0..n);
    }
    let min_parallel_work = SPARSE_ROW_PARALLEL_MIN_FLOPS.min(usize::MAX as u64) as usize;
    let Some(chunk_rows) = crate::parallel::row_reduction_chunk_rows(n, 1, p, min_parallel_work)
    else {
        return sparse_csr_diag_gram_rows(row_ptr, col_idx, vals, p, weights, 0..n);
    };
    let starts: Vec<usize> = (0..n).step_by(chunk_rows).collect();
    let partials: Vec<Array1<f64>> = starts
        .into_par_iter()
        .map(|start| {
            sparse_csr_diag_gram_rows(
                row_ptr,
                col_idx,
                vals,
                p,
                weights,
                start..(start + chunk_rows).min(n),
            )
        })
        .collect();
    let mut diag = Array1::<f64>::zeros(p);
    for partial in &partials {
        diag += partial;
    }
    diag
}

fn sparse_csr_diag_gram_rows(
    row_ptr: &[usize],
    col_idx: &[usize],
    vals: &[f64],
    p: usize,
    weights: ArrayView1<'_, f64>,
    rows: Range<usize>,
) -> Array1<f64> {
    // PSD precondition discharged at the typed boundary
    // (`PsdWeightsView::try_new` inside callers of `xt_diag_x_psd_op`).
    // Signed observed-Hessian assembly uses the signed Gram path
    // (xt_diag_x_signed → streaming kernels) and never reaches this routine.
    let mut diag = Array1::<f64>::zeros(p);
    for i in rows {
        let wi = weights[i];
        if wi == 0.0 {
            continue;
        }
        for idx in row_ptr[i]..row_ptr[i + 1] {
            let j = col_idx[idx];
            let xij = vals[idx];
            diag[j] += wi * xij * xij;
        }
    }
    diag
}

#[inline]
fn dense_transpose_weighted_response(
    matrix: &Array2<f64>,
    weights: &Array1<f64>,
    y: &Array1<f64>,
    row_scale: Option<&Array1<f64>>,
) -> Array1<f64> {
    // Signed-safe: XᵀWy is linear in W, so observed-Hessian / non-canonical-link
    // IRLS sites that drive signed working weights through this kernel must be
    // preserved end-to-end. Clipping negative weights here silently biases the
    // pseudo-response and was the source of the Gram-cleanup mismatch.
    let p = matrix.ncols();
    let n = matrix.nrows();
    let mut out = Array1::<f64>::zeros(p);
    if matrix.is_standard_layout()
        && let (Some(x), Some(w), Some(yslice)) =
            (matrix.as_slice(), weights.as_slice(), y.as_slice())
    {
        let scale_slice = row_scale.and_then(|s| s.as_slice());
        let out_slice = out.as_slice_mut().expect("zeros are contiguous");
        for i in 0..n {
            let mut scaled = yslice[i] * w[i];
            if let Some(s) = scale_slice {
                scaled *= s[i];
            } else if let Some(scale) = row_scale {
                scaled *= scale[i];
            }
            if scaled == 0.0 {
                continue;
            }
            let row = &x[i * p..i * p + p];
            for j in 0..p {
                out_slice[j] += row[j] * scaled;
            }
        }
        return out;
    }
    for i in 0..n {
        let mut scaled = y[i] * weights[i];
        if let Some(scale) = row_scale {
            scaled *= scale[i];
        }
        if scaled == 0.0 {
            continue;
        }
        for j in 0..p {
            out[j] += matrix[[i, j]] * scaled;
        }
    }
    out
}

#[inline]
fn dense_transpose_weighted_response_view(
    matrix: &Array2<f64>,
    weights: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
) -> Array1<f64> {
    // Signed-safe view variant of dense_transpose_weighted_response; see that
    // function for the rationale on preserving sign through XᵀWy.
    let p = matrix.ncols();
    let n = matrix.nrows();
    let mut out = Array1::<f64>::zeros(p);
    if matrix.is_standard_layout()
        && let (Some(x), Some(w), Some(yslice)) =
            (matrix.as_slice(), weights.as_slice(), y.as_slice())
    {
        let out_slice = out.as_slice_mut().expect("zeros are contiguous");
        for i in 0..n {
            let scaled = yslice[i] * w[i];
            if scaled == 0.0 {
                continue;
            }
            let row = &x[i * p..i * p + p];
            for j in 0..p {
                out_slice[j] += row[j] * scaled;
            }
        }
        return out;
    }
    for i in 0..n {
        let scaled = y[i] * weights[i];
        if scaled == 0.0 {
            continue;
        }
        for j in 0..p {
            out[j] += matrix[[i, j]] * scaled;
        }
    }
    out
}

#[derive(Clone)]
pub struct SparseDesignMatrix {
    matrix: SparseColMat<usize, f64>,
    /// Memoized dense copy plus the process-wide ledger reservation that keeps
    /// its bytes accounted for as long as the cache entry is alive.
    dense_cache: Arc<OnceLock<(Arc<Array2<f64>>, MemoryReservation)>>,
    csr_cache: Arc<OnceLock<Arc<SparseRowMat<usize, f64>>>>,
}

impl SparseDesignMatrix {
    pub fn new(matrix: SparseColMat<usize, f64>) -> Self {
        Self {
            matrix,
            dense_cache: Arc::new(OnceLock::new()),
            csr_cache: Arc::new(OnceLock::new()),
        }
    }

    fn dense_nbytes(&self) -> Result<usize, String> {
        self.matrix
            .nrows()
            .checked_mul(self.matrix.ncols())
            .and_then(|cells| cells.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(|| {
                format!(
                    "dense size overflow for sparse design {}x{}",
                    self.matrix.nrows(),
                    self.matrix.ncols()
                )
            })
    }

    fn materialize_dense_arc(&self) -> Arc<Array2<f64>> {
        let mut out = Array2::<f64>::zeros((self.matrix.nrows(), self.matrix.ncols()));
        let (symbolic, values) = self.matrix.parts();
        let col_ptr = symbolic.col_ptr();
        let row_idx = symbolic.row_idx();
        for col in 0..self.matrix.ncols() {
            let start = col_ptr[col];
            let end = col_ptr[col + 1];
            for idx in start..end {
                out[[row_idx[idx], col]] += values[idx];
            }
        }
        Arc::new(out)
    }

    pub fn try_to_dense_arc(&self, context: &str) -> Result<Arc<Array2<f64>>, String> {
        if let Some((cached, _)) = self.dense_cache.get() {
            return Ok(cached.clone());
        }
        let dense_bytes = self.dense_nbytes()?;
        let governor = MemoryGovernor::global();
        if dense_bytes > governor.single_materialization_cap_bytes() {
            let gib = dense_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            return Err(MatrixError::DensificationRefused {
                reason: format!(
                    "{context}: refusing to densify sparse design {}x{} (~{gib:.2} GiB, over the process memory budget); use sparse or matrix-free code",
                    self.matrix.nrows(),
                    self.matrix.ncols(),
                ),
            }
            .into());
        }
        // Memoization is governed: the cache entry holds its ledger charge for
        // the design's lifetime. Every dense materialization must be accounted
        // on the joint ledger for the buffer's lifetime — an unreserved
        // fallthrough here would let simultaneous sparse densifications that
        // each individually pass the cap jointly exceed the process budget
        // (the SPEC 10 failure mode). A refusal is typed evidence to route to
        // a sparse/matrix-free strategy, not permission to allocate anyway.
        let reservation = governor.try_reserve(dense_bytes, context).map_err(|err| {
            String::from(MatrixError::DensificationRefused {
                reason: format!(
                    "{context}: refusing to densify sparse design {}x{}: {err}",
                    self.matrix.nrows(),
                    self.matrix.ncols(),
                ),
            })
        })?;
        Ok(self
            .dense_cache
            .get_or_init(|| (self.materialize_dense_arc(), reservation))
            .0
            .clone())
    }

    /// Densify under the process-wide byte governor, coupling the returned
    /// dense copy to its RAII reservation. A refusal is typed evidence that
    /// the dense footprint does not fit the joint ledger right now — callers
    /// route to a streaming / sparse strategy instead of allocating.
    pub fn try_to_dense_governed(
        &self,
        context: &str,
    ) -> Result<Governed<Arc<Array2<f64>>>, String> {
        let governor = MemoryGovernor::global();
        if let Some((cached, _)) = self.dense_cache.get() {
            // Cache hit: the bytes are already accounted for by the cache's
            // own reservation, so this owner charges nothing extra.
            let reservation = governor
                .try_reserve(0, context)
                .expect("zero-byte reservation cannot exceed any budget");
            return Ok(reservation.bind(cached.clone()));
        }
        let dense_bytes = self.dense_nbytes()?;
        let reservation = governor.try_reserve(dense_bytes, context).map_err(|err| {
            String::from(MatrixError::DensificationRefused {
                reason: format!(
                    "{context}: refusing to densify sparse design {}x{}: {err}",
                    self.matrix.nrows(),
                    self.matrix.ncols(),
                ),
            })
        })?;
        Ok(reservation.bind(self.materialize_dense_arc()))
    }

    pub fn to_dense_arc(&self) -> Arc<Array2<f64>> {
        self.try_to_dense_arc("SparseDesignMatrix::to_dense_arc")
            .unwrap_or_else(|msg| {
                let bt = std::backtrace::Backtrace::force_capture();
                // SAFETY: infallible-style accessor used at sites where the
                // caller has already established that densifying this sparse
                // matrix is permitted (size below the densification guard); a
                // failure here means the caller broke that contract, which
                // warrants an immediate abort with backtrace for diagnosis.
                // SAFETY: infallible accessor; densification refusal here is a caller contract violation.
                std::panic::panic_any(format!("{msg}\nbacktrace:\n{bt}"))
            })
    }

    pub fn to_csr_arc(&self) -> Option<Arc<SparseRowMat<usize, f64>>> {
        if let Some(cached) = self.csr_cache.get() {
            return Some(cached.clone());
        }
        let csr = self.matrix.as_ref().to_row_major().ok()?;
        let arc = Arc::new(csr);
        self.csr_cache.set(arc.clone()).ok();
        Some(arc)
    }

    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if out.nrows() != rows.end - rows.start || out.ncols() != self.ncols() {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "SparseDesignMatrix::row_chunk_into shape mismatch",
            });
        }
        out.fill(0.0);
        let csr = self
            .to_csr_arc()
            .ok_or(MatrixMaterializationError::MissingRowChunk {
                context: "SparseDesignMatrix::row_chunk_into: failed to obtain CSR view",
            })?;
        let symbolic = csr.symbolic();
        let row_ptr = symbolic.row_ptr();
        let col_idx = symbolic.col_idx();
        let values = csr.val();
        for (local_row, row) in rows.enumerate() {
            for ptr in row_ptr[row]..row_ptr[row + 1] {
                out[[local_row, col_idx[ptr]]] = values[ptr];
            }
        }
        Ok(())
    }
}

impl Deref for SparseDesignMatrix {
    type Target = SparseColMat<usize, f64>;
    fn deref(&self) -> &Self::Target {
        &self.matrix
    }
}

impl AsRef<SparseColMat<usize, f64>> for SparseDesignMatrix {
    fn as_ref(&self) -> &SparseColMat<usize, f64> {
        &self.matrix
    }
}

/// Trait for dense-backed design operators that avoid eager materialization.
///
/// Implement this trait for structured designs (multi-channel, rowwise-Kronecker,
/// etc.) that can perform matvecs and Gram-matrix assembly without forming the
/// full dense matrix. Wrap implementations in `DenseDesignMatrix::Lazy(Arc<..>)`
/// to integrate them with the rest of the codebase while keeping the top-level
/// `DesignMatrix` split strictly `Dense | Sparse`.
pub trait DenseDesignOperator: LinearOperator + Send + Sync {
    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
        // Default: X'(w ⊙ y) via apply_transpose.
        let n = self.nrows();
        if weights.len() != n || y.len() != n {
            return Err(format!(
                "DenseDesignOperator::compute_xtwy dimension mismatch: weights={}, y={}, nrows={}",
                weights.len(),
                y.len(),
                n
            ));
        }
        certify_signed_weights("DenseDesignOperator::compute_xtwy", weights, n)?;
        // Signed-safe XᵀWy: linear in w, so observed-Hessian / non-canonical
        // working weights must flow through unclipped.
        let mut wy = Array1::<f64>::zeros(n);
        ndarray::Zip::from(&mut wy)
            .and(weights)
            .and(y)
            .par_for_each(|o, &w, &yi| *o = w * yi);
        Ok(self.apply_transpose(&wy))
    }

    fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        // Default: diag(X M X') computed in chunks via row_chunk — avoids
        // materializing the full n×p dense matrix at once.
        if middle.nrows() != self.ncols() || middle.ncols() != self.ncols() {
            return Err(format!(
                "DenseDesignOperator::quadratic_form_diag dimension mismatch: {}x{} vs expected {}x{}",
                middle.nrows(),
                middle.ncols(),
                self.ncols(),
                self.ncols()
            ));
        }
        let n = self.nrows();
        let mut out = Array1::<f64>::zeros(n);
        // Process in chunks to bound memory: ~8 MB working set.
        let chunk_size = (8 * 1024 * 1024 / (self.ncols().max(1) * 8 * 2))
            .max(16)
            .min(n.max(1));
        let mut start = 0;
        while start < n {
            let end = (start + chunk_size).min(n);
            let x_chunk = self.try_row_chunk(start..end).map_err(|e| e.to_string())?;
            let xm_chunk = fast_ab(&x_chunk, middle);
            let mut chunk_out = out.slice_mut(ndarray::s![start..end]);
            ndarray::Zip::from(&mut chunk_out)
                .and(x_chunk.rows())
                .and(xm_chunk.rows())
                // clamp tiny-negative fp drift on diag(X M Xᵀ) when M is a
                // PSD covariance/precision matrix; not a weight clip.
                .par_for_each(|o, xr, xmr| *o = xr.dot(&xmr).max(0.0));
            start = end;
        }
        Ok(out)
    }

    /// Fill a dense row chunk without materializing the full matrix.
    /// Required: every implementor must provide row-local access here.
    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError>;

    /// Extract a dense row chunk without materializing the full matrix.
    /// Non-panicking owned-chunk API built on top of `row_chunk_into`.
    fn try_row_chunk(&self, rows: Range<usize>) -> Result<Array2<f64>, MatrixMaterializationError> {
        let mut out = Array2::<f64>::zeros((rows.end - rows.start, self.ncols()));
        self.row_chunk_into(rows, out.view_mut())?;
        Ok(out)
    }

    /// Borrow dense storage when this operator already owns it.
    fn as_dense_ref(&self) -> Option<&Array2<f64>> {
        None
    }

    /// Materialization contract captured when this operator-backed design was
    /// selected. Composite operators propagate the strictest contract of their
    /// inputs so a later caller using a more permissive default cannot reverse
    /// an upstream streamed-storage decision.
    fn materialization_policy(&self) -> Option<MaterializationPolicy> {
        None
    }

    /// Batched column extraction: returns an `nrows × cols.len()` dense block
    /// whose k-th column is `apply(e_{cols[k]})`.
    ///
    /// Default impl loops over columns and applies a unit vector per call. Operator
    /// types like `ReparamOperator` that can express the batch as a single GEMM
    /// (`X · Qs[:, cols]`) should override this — it avoids re-walking the inner
    /// matvec for every column.
    fn apply_columns(&self, cols: &[usize]) -> Array2<f64> {
        let n = self.nrows();
        let p = self.ncols();
        let mut out = Array2::<f64>::zeros((n, cols.len()));
        let mut e = Array1::<f64>::zeros(p);
        for (k, &j) in cols.iter().enumerate() {
            assert!(
                j < p,
                "DenseDesignOperator::apply_columns: column index {j} out of bounds (ncols={p})"
            );
            e[j] = 1.0;
            let col = self.apply(&e);
            e[j] = 0.0;
            out.column_mut(k).assign(&col);
        }
        out
    }

    /// Materialize the full dense matrix. Operators that exist precisely to
    /// avoid materialization should still support this for fallback paths,
    /// diagnostics, and prediction.
    fn to_dense(&self) -> Array2<f64>;

    fn estimated_dense_bytes(&self) -> usize {
        self.nrows()
            .saturating_mul(self.ncols())
            .saturating_mul(std::mem::size_of::<f64>())
    }

    fn try_to_dense_with_policy(
        &self,
        policy: &MaterializationPolicy,
        context: &'static str,
    ) -> Result<Arc<Array2<f64>>, MatrixMaterializationError> {
        let effective_policy = merge_operator_materialization_policies(
            Some(policy.clone()),
            self.materialization_policy(),
        )
        .expect("caller policy is always present");
        let bytes = self.estimated_dense_bytes();
        if !effective_policy.allow_operator_materialization {
            return Err(MatrixMaterializationError::Forbidden {
                context,
                mode: gam_runtime::resource::DerivativeStorageMode::AnalyticOperatorRequired,
            });
        }
        if bytes > effective_policy.max_single_dense_bytes {
            return Err(MatrixMaterializationError::TooLarge {
                context,
                nrows: self.nrows(),
                ncols: self.ncols(),
                bytes,
                limit_bytes: effective_policy.max_single_dense_bytes,
            });
        }
        dense_operator_to_dense_by_chunks(self).map(Arc::new)
    }

    /// Materialize through the process-wide governor and couple the returned
    /// matrix to its reservation. Unlike the older Arc-returning helper, this
    /// cannot release its ledger charge while the dense allocation is live.
    fn try_to_dense_governed_with_policy(
        &self,
        policy: &MaterializationPolicy,
        context: &'static str,
    ) -> Result<Governed<Array2<f64>>, MatrixMaterializationError> {
        governed_dense_operator_to_dense_by_chunks(self, policy, context)
    }

    /// Shared dense materialization via the required row-chunk API.
    ///
    /// This deliberately does not fall back through `to_dense()`: operator-backed
    /// designs can be large-scale, and their chunked row path is the bounded
    /// memory materialization contract. Implementations that already own an
    /// `Arc<Array2<_>>` should override this to return it directly.
    fn to_dense_arc(&self) -> Arc<Array2<f64>> {
        Arc::new(
            dense_operator_to_dense_by_chunks(self)
                .expect("DenseDesignOperator::to_dense_arc: row-chunk materialization failed"),
        )
    }
}

/// Operator-backed design plus its governed dense memo.
///
/// Every dense materialization of a lazy design must hold a process-wide
/// [`MemoryGovernor`] ledger charge for the buffer's lifetime (the #2247 F6
/// contract: individually-acceptable materializations must not be able to
/// *jointly* exceed the budget). The memo owns exactly one governed dense
/// copy shared by every clone of this design — repeated `to_dense_arc`/
/// `to_dense_cow` calls reuse it instead of re-streaming the operator, and
/// the ledger charge lives exactly as long as the memo (= the design and all
/// its clones). Derefs to the inner operator so match arms binding `Lazy(op)`
/// keep calling trait methods unchanged.
#[derive(Clone)]
pub struct LazyDense {
    op: Arc<dyn DenseDesignOperator>,
    dense_memo: Arc<OnceLock<Governed<Arc<Array2<f64>>>>>,
}

impl LazyDense {
    fn new(op: Arc<dyn DenseDesignOperator>) -> Self {
        Self {
            op,
            dense_memo: Arc::new(OnceLock::new()),
        }
    }

    fn operator_arc_identity(&self) -> usize {
        Arc::as_ptr(&self.op) as *const () as usize
    }

    /// Governed, memoized dense materialization. On a memo hit the bytes are
    /// already charged by the memo's own reservation; on a miss the footprint
    /// is admitted against the joint ledger BEFORE streaming the operator. A
    /// refusal is typed evidence the dense copy does not fit jointly right
    /// now — fallible callers route to chunked/matrix-free strategies.
    fn try_governed_dense_arc(&self, context: &str) -> Result<Arc<Array2<f64>>, String> {
        enforce_operator_materialization_policy(self.op.as_ref(), context)?;
        if let Some(governed) = self.dense_memo.get() {
            return Ok(Arc::clone(governed.as_ref()));
        }
        let reservation = MemoryGovernor::global()
            .try_reserve_dense_f64(self.op.nrows(), self.op.ncols(), context)
            .map_err(|err| {
                format!(
                    "{context}: refusing to densify {}x{} operator-backed design: {err}",
                    self.op.nrows(),
                    self.op.ncols(),
                )
            })?;
        let dense = dense_operator_to_dense_by_chunks(self.op.as_ref()).map_err(|err| {
            format!(
                "{context}: failed to materialize {}x{} operator-backed design via row chunks: {err}",
                self.op.nrows(),
                self.op.ncols(),
            )
        })?;
        // A concurrent winner's memo (and charge) stands; the loser's copy and
        // reservation drop together here.
        Ok(Arc::clone(
            self.dense_memo
                .get_or_init(|| reservation.bind(Arc::new(dense)))
                .as_ref(),
        ))
    }
}

impl std::ops::Deref for LazyDense {
    type Target = Arc<dyn DenseDesignOperator>;

    fn deref(&self) -> &Self::Target {
        &self.op
    }
}

#[derive(Clone)]
pub enum DenseDesignMatrix {
    Materialized(Arc<Array2<f64>>),
    Lazy(LazyDense),
}

impl std::fmt::Debug for DenseDesignMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Materialized(matrix) => {
                write!(
                    f,
                    "DenseDesignMatrix::Materialized({}x{})",
                    matrix.nrows(),
                    matrix.ncols()
                )
            }
            Self::Lazy(op) => write!(f, "DenseDesignMatrix::Lazy({}x{})", op.nrows(), op.ncols()),
        }
    }
}

impl From<Arc<Array2<f64>>> for DenseDesignMatrix {
    fn from(value: Arc<Array2<f64>>) -> Self {
        Self::Materialized(value)
    }
}

impl From<Array2<f64>> for DenseDesignMatrix {
    fn from(value: Array2<f64>) -> Self {
        Self::Materialized(Arc::new(value))
    }
}

impl<T> From<Arc<T>> for DenseDesignMatrix
where
    T: DenseDesignOperator + 'static,
{
    fn from(value: Arc<T>) -> Self {
        Self::Lazy(LazyDense::new(value))
    }
}

impl DenseDesignMatrix {
    /// Stable identity for cache keying.
    ///
    /// Returns the address of the inner shared `Arc`, which `Clone` shares by
    /// reference. Two `DenseDesignMatrix` values produced by cloning the same
    /// origin (e.g. the `k` per-coordinate `GlmCurvatureCorrectionOperator`s all
    /// built from one converged design) report the same identity, so a `X·F`
    /// projection memoized under this id is reused across them within a single
    /// outer REML evaluation instead of being re-streamed once per coordinate.
    pub fn cache_identity(&self) -> usize {
        match self {
            Self::Materialized(matrix) => Arc::as_ptr(matrix) as *const () as usize,
            Self::Lazy(lazy) => lazy.operator_arc_identity(),
        }
    }

    pub fn nrows(&self) -> usize {
        match self {
            Self::Materialized(matrix) => matrix.nrows(),
            Self::Lazy(op) => op.nrows(),
        }
    }

    pub fn ncols(&self) -> usize {
        match self {
            Self::Materialized(matrix) => matrix.ncols(),
            Self::Lazy(op) => op.ncols(),
        }
    }

    pub fn as_dense_ref(&self) -> Option<&Array2<f64>> {
        match self {
            Self::Materialized(matrix) => Some(matrix.as_ref()),
            Self::Lazy(lazy) => lazy
                .dense_memo
                .get()
                .map(|governed| governed.as_ref().as_ref())
                .or_else(|| lazy.op.as_dense_ref()),
        }
    }

    pub const fn is_materialized_dense(&self) -> bool {
        matches!(self, Self::Materialized(_))
    }

    pub const fn is_operator_backed(&self) -> bool {
        matches!(self, Self::Lazy(_))
    }

    pub fn to_dense(&self) -> Array2<f64> {
        match self {
            Self::Materialized(matrix) => matrix.as_ref().clone(),
            Self::Lazy(lazy) => {
                let policy = ResourcePolicy::default_library();
                panic_or_error_if_large_scale_mode_and_to_dense_called_with_policy(
                    "DenseDesignMatrix::to_dense",
                    lazy.nrows(),
                    lazy.ncols(),
                    &policy,
                )
                .unwrap_or_else(|reason| std::panic::panic_any(reason));
                enforce_operator_materialization_policy(
                    lazy.op.as_ref(),
                    "DenseDesignMatrix::to_dense",
                )
                .unwrap_or_else(|reason| std::panic::panic_any(reason));
                if let Some(governed) = lazy.dense_memo.get() {
                    // Already materialized (and ledger-charged) once — reuse
                    // it instead of re-streaming the operator.
                    return governed.as_ref().as_ref().clone();
                }
                // Owned-return variant: the escaping buffer cannot carry an
                // RAII charge, so account at least the construction window on
                // the joint ledger and refuse loudly under joint pressure.
                // Callers that can hold the charge for the buffer's lifetime
                // must use `try_to_dense_governed`.
                let construction_charge = MemoryGovernor::global()
                    .try_reserve_dense_f64(
                        lazy.nrows(),
                        lazy.ncols(),
                        "DenseDesignMatrix::to_dense",
                    )
                    // SAFETY: infallible accessor; a joint-ledger refusal here means the caller broke the densification contract.
                    .unwrap_or_else(|err| std::panic::panic_any(err.to_string()));
                let dense =
                    dense_operator_to_dense_by_chunks(lazy.op.as_ref()).unwrap_or_else(|err| {
                        std::panic::panic_any(format!(
                            "DenseDesignMatrix::to_dense: failed to materialize {}x{} \
                             operator-backed design via row chunks: {err}",
                            lazy.nrows(),
                            lazy.ncols(),
                        ))
                    });
                // The charge covers exactly the construction window; the escaping
                // buffer itself cannot carry an RAII charge (doc above).
                drop(construction_charge);
                dense
            }
        }
    }

    pub fn to_dense_arc(&self) -> Arc<Array2<f64>> {
        match self {
            Self::Materialized(matrix) => Arc::clone(matrix),
            Self::Lazy(lazy) => {
                let policy = ResourcePolicy::default_library();
                panic_or_error_if_large_scale_mode_and_to_dense_called_with_policy(
                    "DenseDesignMatrix::to_dense_arc",
                    lazy.nrows(),
                    lazy.ncols(),
                    &policy,
                )
                .unwrap_or_else(|reason| std::panic::panic_any(reason));
                lazy.try_governed_dense_arc("DenseDesignMatrix::to_dense_arc")
                    // SAFETY: infallible accessor; refusal here is a caller contract violation, abort with the ledger evidence.
                    .unwrap_or_else(|msg| std::panic::panic_any(msg))
            }
        }
    }

    pub fn try_to_dense_arc(&self, context: &str) -> Result<Arc<Array2<f64>>, String> {
        // Auto-policy from the design's own dense footprint. The earlier
        // shape-based pick reused `for_problem(nrows, ncols, _)`, which is
        // intended for classifying the *whole fitting problem* — it flips to
        // `AnalyticOperatorRequired` at `nrows >= 100_000` regardless of
        // column count. That was wrong for an individual design: a 102052x4
        // operator-backed block dense-materializes to only ~3 MiB and is
        // genuinely safe. We now pick the permissive policy and let the
        // byte-cap inside the materialization guard reject anything that
        // would actually blow the default 1 GiB single-materialization budget.
        // Callers that need strict refusal still get it by calling
        // `try_to_dense_arc_with_policy(ctx, &analytic_operator_required())`.
        let policy = ResourcePolicy::default_library();
        self.try_to_dense_arc_with_policy(context, &policy)
    }

    /// Policy-aware variant of [`Self::try_to_dense_arc`].
    ///
    /// Uses the supplied policy's `max_single_materialization_bytes` cap when
    /// deciding whether to densify a lazy operator-backed design.  The default
    /// `try_to_dense_arc` always uses `ResourcePolicy::default_library()` (the
    /// 1 GiB cap suitable for ad-hoc dense conversions, matching the
    /// `CoefficientTransformOperator::MATERIALIZE_MAX_BYTES` ceiling); cache
    /// layers that have their own larger cap (e.g.
    /// `CoefficientTransformOperator::MATERIALIZE_MAX_BYTES`) can call this
    /// method to consume the inner under their own threshold without forcing
    /// the conservative default on every consumer.
    pub fn try_to_dense_arc_with_policy(
        &self,
        context: &str,
        policy: &ResourcePolicy,
    ) -> Result<Arc<Array2<f64>>, String> {
        match self {
            Self::Materialized(matrix) => Ok(Arc::clone(matrix)),
            Self::Lazy(lazy) => {
                panic_or_error_if_large_scale_mode_and_to_dense_called_with_policy(
                    context,
                    lazy.nrows(),
                    lazy.ncols(),
                    policy,
                )?;
                lazy.try_governed_dense_arc(context)
            }
        }
    }

    pub fn try_row_chunk(
        &self,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, MatrixMaterializationError> {
        match self {
            Self::Materialized(matrix) => Ok(matrix.slice(s![rows, ..]).to_owned()),
            Self::Lazy(op) => op.try_row_chunk(rows),
        }
    }

    pub fn row_chunk_into(
        &self,
        rows: Range<usize>,
        out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        match self {
            Self::Materialized(matrix) => {
                let mut out = out;
                out.assign(&matrix.slice(s![rows, ..]));
                Ok(())
            }
            Self::Lazy(op) => op.row_chunk_into(rows, out),
        }
    }
}

impl LinearOperator for DenseDesignMatrix {
    fn nrows(&self) -> usize {
        DenseDesignMatrix::nrows(self)
    }

    fn ncols(&self) -> usize {
        DenseDesignMatrix::ncols(self)
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Materialized(matrix) => fast_av(matrix, vector),
            Self::Lazy(op) => op.apply(vector),
        }
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Materialized(matrix) => fast_atv(matrix, vector),
            Self::Lazy(op) => op.apply_transpose(vector),
        }
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        certify_signed_weights("DenseDesignMatrix::diag_xtw_x", weights, self.nrows())?;
        match self {
            Self::Materialized(matrix) => {
                let mut xtwx = Array2::<f64>::zeros((matrix.ncols(), matrix.ncols()));
                stream_weighted_crossprod_into(
                    matrix,
                    weights,
                    &mut xtwx,
                    CrossprodStructure::Full,
                    CrossprodAccum::Replace,
                    effective_global_parallelism(),
                );
                Ok(xtwx)
            }
            Self::Lazy(op) => op.diag_xtw_x(weights),
        }
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        // Exact diagonal of Xᵀdiag(w)X.  It may be signed for observed
        // curvature; solve-level stabilization owns positive-definiteness.
        certify_signed_weights("DenseDesignMatrix::diag_gram", weights, self.nrows())?;
        match self {
            Self::Materialized(matrix) => {
                let n = matrix.nrows();
                let p = matrix.ncols();
                if (n as u64) * (p as u64) < DENSE_ROW_PARALLEL_MIN_NP {
                    let mut diag = Array1::<f64>::zeros(p);
                    for i in 0..n {
                        let wi = weights[i];
                        if wi == 0.0 {
                            continue;
                        }
                        for j in 0..p {
                            let xij = matrix[[i, j]];
                            diag[j] += wi * xij * xij;
                        }
                    }
                    return Ok(diag);
                }
                // Deterministic parallel row reduction (length-only pairwise
                // tree; see the standard-layout path above).
                let diag = crate::pairwise_reduce::par_deterministic_block_fold(
                    n,
                    |range: core::ops::Range<usize>| {
                        let mut acc = Array1::<f64>::zeros(p);
                        for i in range {
                            let wi = weights[i];
                            if wi != 0.0 {
                                for j in 0..p {
                                    let xij = matrix[[i, j]];
                                    acc[j] += wi * xij * xij;
                                }
                            }
                        }
                        acc
                    },
                    |mut a, b| {
                        a += &b;
                        a
                    },
                )
                .unwrap_or_else(|| Array1::<f64>::zeros(p));
                Ok(diag)
            }
            Self::Lazy(op) => op.diag_gram(weights),
        }
    }

    fn apply_weighted_normal(
        &self,
        weights: FiniteSignedWeightsView<'_>,
        vector: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge: f64,
    ) -> Array1<f64> {
        assert_eq!(
            weights.len(),
            self.nrows(),
            "DenseDesignMatrix::apply_weighted_normal weight length mismatch"
        );
        assert_eq!(
            vector.len(),
            self.ncols(),
            "DenseDesignMatrix::apply_weighted_normal vector length mismatch"
        );
        // Exact signed normal product.  PCG callers ordinarily provide Fisher
        // weights, while observed-Hessian callers may provide negative rows;
        // definiteness is a property of the assembled/stabilized global matrix,
        // not something this row-linear kernel manufactures by projection.
        let weights_view = weights.view();
        match self {
            Self::Materialized(matrix) => {
                let n = matrix.nrows();
                let p = matrix.ncols();
                let mut out = if (n as u64) * (p as u64) < DENSE_ROW_PARALLEL_MIN_NP {
                    let mut out = Array1::<f64>::zeros(p);
                    for i in 0..n {
                        let wi = weights_view[i];
                        if wi == 0.0 {
                            continue;
                        }
                        let mut row_dot = 0.0_f64;
                        for j in 0..p {
                            row_dot += matrix[[i, j]] * vector[j];
                        }
                        if row_dot == 0.0 {
                            continue;
                        }
                        let scaled = wi * row_dot;
                        for j in 0..p {
                            out[j] += scaled * matrix[[i, j]];
                        }
                    }
                    out
                } else {
                    // Deterministic parallel row reduction (length-only
                    // pairwise tree; see diag_gram above).
                    crate::pairwise_reduce::par_deterministic_block_fold(
                        n,
                        |range: core::ops::Range<usize>| {
                            let mut acc = Array1::<f64>::zeros(p);
                            for i in range {
                                let wi = weights_view[i];
                                if wi != 0.0 {
                                    let mut row_dot = 0.0_f64;
                                    for j in 0..p {
                                        row_dot += matrix[[i, j]] * vector[j];
                                    }
                                    if row_dot != 0.0 {
                                        let scaled = wi * row_dot;
                                        for j in 0..p {
                                            acc[j] += scaled * matrix[[i, j]];
                                        }
                                    }
                                }
                            }
                            acc
                        },
                        |mut a, b| {
                            a += &b;
                            a
                        },
                    )
                    .unwrap_or_else(|| Array1::<f64>::zeros(p))
                };
                if let Some(pen) = penalty {
                    out += &fast_av(pen, vector);
                }
                if ridge > 0.0 {
                    for j in 0..p {
                        out[j] += ridge * vector[j];
                    }
                }
                out
            }
            Self::Lazy(op) => op.apply_weighted_normal(weights, vector, penalty, ridge),
        }
    }

    fn uses_matrix_free_pcg(&self) -> bool {
        match self {
            Self::Materialized(_) => true,
            Self::Lazy(op) => op.uses_matrix_free_pcg(),
        }
    }
}

impl DenseDesignOperator for DenseDesignMatrix {
    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
        if y.len() != self.nrows() {
            return Err(format!(
                "DenseDesignMatrix::compute_xtwy response length mismatch: y={}, nrows={}",
                y.len(),
                self.nrows()
            ));
        }
        certify_signed_weights("DenseDesignMatrix::compute_xtwy", weights, self.nrows())?;
        match self {
            Self::Materialized(matrix) => {
                Ok(dense_transpose_weighted_response(matrix, weights, y, None))
            }
            Self::Lazy(op) => op.compute_xtwy(weights, y),
        }
    }

    fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        match self {
            Self::Materialized(matrix) => {
                if middle.nrows() != matrix.ncols() || middle.ncols() != matrix.ncols() {
                    return Err(format!(
                        "quadratic_form_diag dimension mismatch: matrix is {}x{}, expected {}x{}",
                        middle.nrows(),
                        middle.ncols(),
                        matrix.ncols(),
                        matrix.ncols()
                    ));
                }
                let xc = fast_ab(matrix, middle);
                let n = matrix.nrows();
                let p = matrix.ncols();
                let mut out = Array1::<f64>::zeros(n);
                if matrix.is_standard_layout()
                    && xc.is_standard_layout()
                    && let (Some(m_all), Some(xc_all), Some(out_slice)) =
                        (matrix.as_slice(), xc.as_slice(), out.as_slice_mut())
                {
                    // Parallel per-row clamped quadratic-form diagonal with
                    // stride-1 reads from both row-major operands. Avoids the
                    // per-row `Array1::dot` call's overhead at large-scale shapes
                    // (n ≈ 2e5, p ≈ 33).
                    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
                    use rayon::slice::ParallelSliceMut;
                    out_slice
                        .par_chunks_mut(1)
                        .enumerate()
                        .for_each(|(i, slot)| {
                            let off = i * p;
                            let m_row = &m_all[off..off + p];
                            let xc_row = &xc_all[off..off + p];
                            let mut acc = 0.0_f64;
                            for j in 0..p {
                                acc += m_row[j] * xc_row[j];
                            }
                            // clamp tiny-negative fp drift on diag(X M Xᵀ)
                            // when M is a PSD covariance/precision matrix.
                            slot[0] = acc.max(0.0);
                        });
                } else {
                    for i in 0..n {
                        // clamp tiny-negative fp drift on diag(X M Xᵀ)
                        // when M is a PSD covariance/precision matrix.
                        out[i] = matrix.row(i).dot(&xc.row(i)).max(0.0);
                    }
                }
                Ok(out)
            }
            Self::Lazy(op) => op.quadratic_form_diag(middle),
        }
    }

    fn as_dense_ref(&self) -> Option<&Array2<f64>> {
        DenseDesignMatrix::as_dense_ref(self)
    }

    fn materialization_policy(&self) -> Option<MaterializationPolicy> {
        match self {
            Self::Materialized(_) => None,
            Self::Lazy(lazy) => lazy.op.materialization_policy(),
        }
    }

    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if out.nrows() != rows.end - rows.start || out.ncols() != self.ncols() {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "DenseDesignMatrix::row_chunk_into shape mismatch",
            });
        }
        match self {
            Self::Materialized(matrix) => {
                out.assign(&matrix.slice(s![rows, ..]));
                Ok(())
            }
            Self::Lazy(op) => op.row_chunk_into(rows, out),
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        DenseDesignMatrix::to_dense(self)
    }

    fn to_dense_arc(&self) -> Arc<Array2<f64>> {
        DenseDesignMatrix::to_dense_arc(self)
    }
}

// ---------------------------------------------------------------------------
// ReparamOperator — lazy X·Qs composition without materialization
// ---------------------------------------------------------------------------

/// Lazy composed operator for reparameterized design: X_transformed = X_original · Qs.
///
/// Instead of materializing the dense n×p product X·Qs, this operator applies
/// the p×p orthogonal transform Qs on the coefficient side:
///
///   apply(v)           → X · (Qs · v)
///   apply_transpose(v) → Qs^T · (X^T · v)
///   diag_xtw_x(w)      → Qs^T · (X^T W X) · Qs
///
/// This preserves the sparsity of X and avoids an O(n·p) dense allocation.
pub struct ReparamOperator {
    x_original: DesignMatrix,
    qs: Arc<Array2<f64>>,
    n: usize,
    p: usize,
}

impl ReparamOperator {
    pub fn new(x_original: DesignMatrix, qs: Arc<Array2<f64>>) -> Self {
        let n = x_original.nrows();
        let p = qs.ncols();
        assert_eq!(
            x_original.ncols(),
            qs.nrows(),
            "ReparamOperator: X cols ({}) must match Qs rows ({})",
            x_original.ncols(),
            qs.nrows()
        );
        Self {
            x_original,
            qs,
            n,
            p,
        }
    }

    /// Access the underlying original design matrix.
    pub fn x_original(&self) -> &DesignMatrix {
        &self.x_original
    }

    /// Access the Qs orthogonal transform.
    pub fn qs(&self) -> &Array2<f64> {
        &self.qs
    }
}

impl LinearOperator for ReparamOperator {
    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.p
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        // X · (Qs · v): apply Qs on the p-dimensional side first, then sparse/dense X.
        let qv = self.qs.dot(vector);
        self.x_original.apply(&qv)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        // Qs^T · (X^T · v): apply X^T first (sparse matvec), then small dense Qs^T.
        let xtv = self.x_original.apply_transpose(vector);
        fast_atv(&self.qs, &xtv)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        // Qs^T · (X^T W X) · Qs: compute X^TWX in original basis (sparse-friendly),
        // then two small p×p multiplications.
        let xtwx = self.x_original.diag_xtw_x(weights)?;
        let tmp = fast_atb(&self.qs, &xtwx);
        Ok(fast_ab(&tmp, &self.qs))
    }

    fn apply_weighted_normal(
        &self,
        weights: FiniteSignedWeightsView<'_>,
        vector: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge: f64,
    ) -> Array1<f64> {
        assert_eq!(
            weights.len(),
            self.x_original.nrows(),
            "ReparamOperator::apply_weighted_normal weight length mismatch"
        );
        assert_eq!(
            vector.len(),
            self.qs.ncols(),
            "ReparamOperator::apply_weighted_normal vector length mismatch"
        );
        // Qs^T X^T W X Qs v + S v + ridge v, with signed W preserved.  Any
        // ridge needed for a global solve is applied after the exact product.
        let weights = weights.view();
        let qv = self.qs.dot(vector);
        let xqv = self.x_original.apply(&qv);
        let mut wxqv = xqv;
        for i in 0..wxqv.len() {
            wxqv[i] *= weights[i];
        }
        let xtw = self.x_original.apply_transpose(&wxqv);
        let mut out = fast_atv(&self.qs, &xtw);
        if let Some(pen) = penalty {
            out += &fast_av(pen, vector);
        }
        if ridge > 0.0 {
            // BLAS axpy: out += ridge * vector, no temporary allocation.
            out.scaled_add(ridge, vector);
        }
        out
    }
}

impl DenseDesignOperator for ReparamOperator {
    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
        // Qs^T · X^T(w ⊙ y)
        let xtwy = self.x_original.compute_xtwy(weights, y)?;
        Ok(fast_atv(&self.qs, &xtwy))
    }

    fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        // diag(X Qs M Qs^T X^T) = diag(X · (Qs M Qs^T) · X^T)
        // Compute M_orig = Qs · M · Qs^T (p×p), then delegate to x_original.
        let qm = fast_ab(&self.qs, middle);
        let m_orig = fast_ab(&qm, &self.qs.t().to_owned());
        self.x_original.quadratic_form_diag(&m_orig)
    }

    fn to_dense(&self) -> Array2<f64> {
        match &self.x_original {
            DesignMatrix::Dense(x) => fast_ab(x.to_dense_arc().as_ref(), &self.qs),
            _ => {
                let x_dense = self.x_original.to_dense();
                fast_ab(&x_dense, &self.qs)
            }
        }
    }

    fn to_dense_arc(&self) -> Arc<Array2<f64>> {
        Arc::new(self.to_dense())
    }

    fn as_dense_ref(&self) -> Option<&Array2<f64>> {
        None
    }

    fn materialization_policy(&self) -> Option<MaterializationPolicy> {
        self.x_original.materialization_policy()
    }

    fn apply_columns(&self, cols: &[usize]) -> Array2<f64> {
        // (X · Qs)[:, cols] = X · Qs[:, cols] — one batched matvec over the inner
        // design instead of one-per-column dispatch on a unit vector.
        let qs_cols = self.qs.select(Axis(1), cols);
        match &self.x_original {
            DesignMatrix::Dense(x) => match x.as_dense_ref() {
                Some(x_dense) => fast_ab(x_dense, &qs_cols),
                None => {
                    let n = self.n;
                    let mut out = Array2::<f64>::zeros((n, cols.len()));
                    for k in 0..cols.len() {
                        let col = qs_cols.column(k).to_owned();
                        let xc = self.x_original.apply(&col);
                        out.column_mut(k).assign(&xc);
                    }
                    out
                }
            },
            DesignMatrix::Sparse(_) => {
                // Sparse X: apply column-by-column over the small qs_cols block.
                let n = self.n;
                let mut out = Array2::<f64>::zeros((n, cols.len()));
                for k in 0..cols.len() {
                    let col = qs_cols.column(k).to_owned();
                    let xc = self.x_original.apply(&col);
                    out.column_mut(k).assign(&xc);
                }
                out
            }
        }
    }

    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if out.nrows() != rows.end - rows.start || out.ncols() != self.p {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "ReparamOperator::row_chunk_into shape mismatch",
            });
        }
        match &self.x_original {
            DesignMatrix::Dense(x) => {
                let chunk = x.try_row_chunk(rows)?;
                out.assign(&fast_ab(&chunk, &self.qs));
            }
            DesignMatrix::Sparse(sdm) => {
                // Extract rows directly from CSR without densifying the full matrix.
                let csr = sdm
                    .to_csr_arc()
                    .ok_or(MatrixMaterializationError::MissingRowChunk {
                        context: "ReparamOperator::row_chunk_into: failed to obtain CSR view",
                    })?;
                let sym = csr.symbolic();
                let row_ptr = sym.row_ptr();
                let col_idx = sym.col_idx();
                let vals = csr.val();
                let chunk_rows = rows.end - rows.start;
                let p_inner = sdm.ncols();
                let mut chunk = Array2::<f64>::zeros((chunk_rows, p_inner));
                for (local, global) in (rows.start..rows.end).enumerate() {
                    for ptr in row_ptr[global]..row_ptr[global + 1] {
                        chunk[[local, col_idx[ptr]]] = vals[ptr];
                    }
                }
                out.assign(&fast_ab(&chunk, &self.qs));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RandomEffectOperator — O(n) implicit design for random intercepts
// ---------------------------------------------------------------------------

/// Implicit design operator for random-intercept effects.
///
/// Instead of materializing an n × q one-hot matrix, stores only the O(n)
/// integer group-label vector.  All matvecs, Gram assembly, and
/// weighted-normal products operate in O(n) time and O(n + q) memory.
#[derive(Clone)]
pub struct RandomEffectOperator {
    /// For each observation, the column index of its group (0..num_groups),
    /// or `None` if the observation's level was not in the kept set (prediction
    /// with unseen levels).
    pub group_ids: Vec<Option<usize>>,
    /// Number of observations.
    pub n: usize,
    /// Number of groups (columns).
    pub num_groups: usize,
}

impl RandomEffectOperator {
    pub fn new(group_ids: Vec<Option<usize>>, num_groups: usize) -> Self {
        let n = group_ids.len();
        Self {
            group_ids,
            n,
            num_groups,
        }
    }

    /// For a dense block X_dense (n × p_dense) and weights w, compute
    /// X_dense' diag(w) X_re  →  (p_dense × num_groups) matrix.
    ///
    /// Column g of the result = Σ_{i: group[i]=g} w[i] * X_dense.row(i).
    /// Total cost: O(n × p_dense).
    pub fn weighted_cross_with_dense(
        &self,
        dense: &Array2<f64>,
        weights: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if dense.nrows() != self.n {
            return Err(format!(
                "RandomEffectOperator::weighted_cross_with_dense row mismatch: dense={}, nrows={}",
                dense.nrows(),
                self.n
            ));
        }
        certify_signed_weights(
            "RandomEffectOperator::weighted_cross_with_dense",
            weights,
            self.n,
        )?;
        let p_dense = dense.ncols();
        let mut cross = Array2::<f64>::zeros((p_dense, self.num_groups));
        for i in 0..self.n {
            if let Some(g) = self.group_ids[i] {
                let wi = weights[i];
                if wi == 0.0 {
                    continue;
                }
                for j in 0..p_dense {
                    cross[[j, g]] += wi * dense[[i, j]];
                }
            }
        }
        Ok(cross)
    }

    /// For two RE operators, compute X_re_a' diag(w) X_re_b → (qa × qb).
    /// Entry (a, b) = Σ_{i: group_a[i]=a AND group_b[i]=b} w[i].
    /// Cost: O(n).
    pub fn weighted_cross_with_re(
        &self,
        other: &RandomEffectOperator,
        weights: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if other.n != self.n {
            return Err(format!(
                "RandomEffectOperator::weighted_cross_with_re row mismatch: other={}, nrows={}",
                other.n, self.n
            ));
        }
        certify_signed_weights(
            "RandomEffectOperator::weighted_cross_with_re",
            weights,
            self.n,
        )?;
        let mut cross = Array2::<f64>::zeros((self.num_groups, other.num_groups));
        for i in 0..self.n {
            if let (Some(a), Some(b)) = (self.group_ids[i], other.group_ids[i]) {
                let wi = weights[i];
                if wi != 0.0 {
                    cross[[a, b]] += wi;
                }
            }
        }
        Ok(cross)
    }
}

impl LinearOperator for RandomEffectOperator {
    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.num_groups
    }

    /// Forward: out[i] = β[group[i]], or 0 if unmatched.
    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        use rayon::prelude::*;
        let out: Vec<f64> = self
            .group_ids
            .par_iter()
            .map(|g| g.map(|g| vector[g]).unwrap_or(0.0))
            .collect();
        Array1::from(out)
    }

    /// Transpose: out[g] = Σ_{i: group[i]=g} v[i].
    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.num_groups);
        for i in 0..self.n {
            if let Some(g) = self.group_ids[i] {
                out[g] += vector[i];
            }
        }
        out
    }

    /// X'WX for a one-hot design is diagonal: D[g,g] = Σ_{i: group[i]=g} w[i].
    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        certify_signed_weights("RandomEffectOperator::diag_xtw_x", weights, self.n)?;
        let q = self.num_groups;
        let mut xtwx = Array2::<f64>::zeros((q, q));
        for i in 0..self.n {
            if let Some(g) = self.group_ids[i] {
                xtwx[[g, g]] += weights[i];
            }
        }
        Ok(xtwx)
    }

    /// Diagonal of X'WX: per-group weight sums.
    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        certify_signed_weights("RandomEffectOperator::diag_gram", weights, self.n)?;
        let mut diag = Array1::<f64>::zeros(self.num_groups);
        for i in 0..self.n {
            if let Some(g) = self.group_ids[i] {
                diag[g] += weights[i];
            }
        }
        Ok(diag)
    }

    /// Fused X'WXβ + Sβ + ridge·β.  O(n + q).
    fn apply_weighted_normal(
        &self,
        weights: FiniteSignedWeightsView<'_>,
        vector: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge: f64,
    ) -> Array1<f64> {
        assert_eq!(
            weights.len(),
            self.n,
            "RandomEffectOperator::apply_weighted_normal weight length mismatch"
        );
        assert_eq!(
            vector.len(),
            self.num_groups,
            "RandomEffectOperator::apply_weighted_normal vector length mismatch"
        );
        // Step 1: accumulate per-group weighted β[g] contributions.
        //   group_acc[g] = Σ_{i in group g} w[i]
        //   result[g] = group_acc[g] * vector[g]
        let weights = weights.view();
        let mut group_wacc = Array1::<f64>::zeros(self.num_groups);
        for i in 0..self.n {
            if let Some(g) = self.group_ids[i] {
                group_wacc[g] += weights[i];
            }
        }
        let mut out = Array1::<f64>::zeros(self.num_groups);
        for g in 0..self.num_groups {
            out[g] = group_wacc[g] * vector[g];
        }
        if let Some(pen) = penalty {
            out += &pen.dot(vector);
        }
        if ridge > 0.0 {
            for g in 0..self.num_groups {
                out[g] += ridge * vector[g];
            }
        }
        out
    }

    fn uses_matrix_free_pcg(&self) -> bool {
        true
    }
}

impl DenseDesignOperator for RandomEffectOperator {
    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
        if weights.len() != self.n || y.len() != self.n {
            return Err(format!(
                "RandomEffectOperator::compute_xtwy dimension mismatch: weights={}, y={}, nrows={}",
                weights.len(),
                y.len(),
                self.n
            ));
        }
        certify_signed_weights("RandomEffectOperator::compute_xtwy", weights, self.n)?;
        let mut out = Array1::<f64>::zeros(self.num_groups);
        for i in 0..self.n {
            if let Some(g) = self.group_ids[i] {
                let wi = weights[i];
                out[g] += wi * y[i];
            }
        }
        Ok(out)
    }

    /// diag(X M X') for one-hot X: out[i] = M[group[i], group[i]].
    fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        use rayon::prelude::*;
        let out: Vec<f64> = self
            .group_ids
            .par_iter()
            .map(|g| g.map(|g| middle[[g, g]].max(0.0)).unwrap_or(0.0))
            .collect();
        Ok(Array1::from(out))
    }

    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if out.nrows() != rows.end - rows.start || out.ncols() != self.num_groups {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "RandomEffectOperator::row_chunk_into shape mismatch",
            });
        }
        out.fill(0.0);
        for (local, global) in rows.enumerate() {
            if let Some(g) = self.group_ids[global] {
                out[[local, g]] = 1.0;
            }
        }
        Ok(())
    }

    /// Materialize the full n × q one-hot matrix (fallback for diagnostics).
    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.n, self.num_groups));
        ndarray::Zip::indexed(out.rows_mut()).par_for_each(|i, mut row| {
            if let Some(g) = self.group_ids[i] {
                row[g] = 1.0;
            }
        });
        out
    }
}

// ---------------------------------------------------------------------------
// BlockDesignOperator — horizontal block composition [B₀ | B₁ | … | Bₖ]
// ---------------------------------------------------------------------------

/// A single block in a horizontally-composed design operator.
#[derive(Clone)]
pub enum DesignBlock {
    Dense(DenseDesignMatrix),
    Sparse(SparseDesignMatrix),
    RandomEffect(Arc<RandomEffectOperator>),
    /// Implicit all-ones intercept column: n rows, 1 column, zero storage.
    Intercept(usize),
}

impl DesignBlock {
    pub fn nrows(&self) -> usize {
        match self {
            Self::Dense(d) => d.nrows(),
            Self::Sparse(s) => s.nrows(),
            Self::RandomEffect(op) => op.nrows(),
            Self::Intercept(n) => *n,
        }
    }

    pub fn ncols(&self) -> usize {
        match self {
            Self::Dense(d) => d.ncols(),
            Self::Sparse(s) => s.ncols(),
            Self::RandomEffect(op) => op.ncols(),
            Self::Intercept(_) => 1,
        }
    }

    fn materialization_policy(&self) -> Option<MaterializationPolicy> {
        match self {
            Self::Dense(design) => design.materialization_policy(),
            Self::Sparse(_) | Self::RandomEffect(_) | Self::Intercept(_) => None,
        }
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(d) => d.apply(vector),
            Self::Sparse(s) => DesignMatrix::Sparse(s.clone()).apply(vector),
            Self::RandomEffect(op) => op.apply(vector),
            Self::Intercept(n) => Array1::from_elem(*n, vector[0]),
        }
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(d) => d.apply_transpose(vector),
            Self::Sparse(s) => DesignMatrix::Sparse(s.clone()).apply_transpose(vector),
            Self::RandomEffect(op) => op.apply_transpose(vector),
            Self::Intercept(_) => {
                let sum: f64 = vector.iter().sum();
                Array1::from_vec(vec![sum])
            }
        }
    }

    fn try_row_chunk(&self, rows: Range<usize>) -> Result<Array2<f64>, MatrixMaterializationError> {
        match self {
            Self::Dense(d) => d.try_row_chunk(rows),
            Self::Sparse(s) => DesignMatrix::Sparse(s.clone()).try_row_chunk(rows),
            Self::RandomEffect(op) => op.try_row_chunk(rows),
            Self::Intercept(_) => Ok(Array2::ones((rows.end - rows.start, 1))),
        }
    }

    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if out.nrows() != rows.end - rows.start || out.ncols() != self.ncols() {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "DesignBlock::row_chunk_into shape mismatch",
            });
        }
        match self {
            Self::Dense(d) => d.row_chunk_into(rows, out),
            Self::Sparse(s) => s.row_chunk_into(rows, out),
            Self::RandomEffect(op) => op.row_chunk_into(rows, out),
            Self::Intercept(_) => {
                out.fill(1.0);
                Ok(())
            }
        }
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        certify_signed_weights("DesignBlock::diag_xtw_x", weights, self.nrows())?;
        match self {
            Self::Dense(d) => d.diag_xtw_x(weights),
            Self::Sparse(s) => DesignMatrix::Sparse(s.clone()).diag_xtw_x(weights),
            Self::RandomEffect(op) => op.diag_xtw_x(weights),
            Self::Intercept(_) => {
                // Signed w: diag_xtw_x is the sign-honest XᵀWX assembler (see
                // the `LinearOperator::diag_xtw_x` contract), used for
                // observed-Hessian curvature on non-canonical links where
                // negative working weights are the normal case. Clamping here
                // would silently corrupt the intercept row/column whenever any
                // weight is negative.
                let sum: f64 = weights.iter().sum();
                Ok(Array2::from_elem((1, 1), sum))
            }
        }
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        certify_signed_weights("DesignBlock::diag_gram", weights, self.nrows())?;
        match self {
            Self::Dense(d) => d.diag_gram(weights),
            Self::Sparse(s) => DesignMatrix::Sparse(s.clone()).diag_gram(weights),
            Self::RandomEffect(op) => op.diag_gram(weights),
            Self::Intercept(_) => {
                // Signed w, matching diag_xtw_x above (the default
                // `LinearOperator::diag_gram` is literally `diag_xtw_x`'s
                // diagonal).
                let sum: f64 = weights.iter().sum();
                Ok(Array1::from_vec(vec![sum]))
            }
        }
    }

    /// Materialize this block as a dense (n, p_k) matrix.
    fn to_dense(&self) -> Array2<f64> {
        match self {
            Self::Dense(d) => d.to_dense(),
            Self::Sparse(s) => s.to_dense_arc().as_ref().clone(),
            Self::RandomEffect(op) => op.to_dense(),
            Self::Intercept(n) => Array2::ones((*n, 1)),
        }
    }
}

/// Horizontally-composed design operator: X = [B₀ | B₁ | … | Bₖ].
///
/// Each block can be dense or operator-based.  The coefficient vector β is
/// partitioned by block, and the forward product is the sum of per-block
/// contributions.  Cross-block terms in X'WX are computed via specialized
/// methods on `RandomEffectOperator` for efficiency.
#[derive(Clone)]
pub struct BlockDesignOperator {
    pub blocks: Vec<DesignBlock>,
    /// Cumulative column offsets: block i owns columns col_offsets[i]..col_offsets[i+1].
    pub col_offsets: Vec<usize>,
    pub total_cols: usize,
    pub n: usize,
}

impl BlockDesignOperator {
    pub fn new(blocks: Vec<DesignBlock>) -> Result<Self, String> {
        if blocks.is_empty() {
            return Err("BlockDesignOperator: need at least one block".to_string());
        }
        let n = blocks[0].nrows();
        for (i, b) in blocks.iter().enumerate() {
            if b.nrows() != n {
                return Err(format!(
                    "BlockDesignOperator: block {i} has {} rows, expected {n}",
                    b.nrows()
                ));
            }
        }
        let mut col_offsets = Vec::with_capacity(blocks.len() + 1);
        col_offsets.push(0);
        for b in &blocks {
            col_offsets.push(col_offsets.last().unwrap() + b.ncols());
        }
        let total_cols = *col_offsets.last().unwrap();
        Ok(Self {
            blocks,
            col_offsets,
            total_cols,
            n,
        })
    }

    fn weighted_cross_chunked(
        &self,
        left: &DesignBlock,
        right: &DesignBlock,
        weights: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let pi = left.ncols();
        let pj = right.ncols();
        let mut cross = Array2::<f64>::zeros((pi, pj));
        for start in (0..self.n).step_by(OPERATOR_ROW_CHUNK_SIZE) {
            let end = (start + OPERATOR_ROW_CHUNK_SIZE).min(self.n);
            let left_chunk = left.try_row_chunk(start..end).map_err(|e| e.to_string())?;
            let right_chunk = right.try_row_chunk(start..end).map_err(|e| e.to_string())?;
            for local in 0..(end - start) {
                // Cross-block X_iᵀ diag(w) X_j is linear in w and well-defined
                // for any sign — observed-Hessian assembly (binomial+cloglog,
                // Gamma+identity, etc.) legitimately supplies signed w_hessian
                // here. The prior `.max(0.0)` silently zeroed the negative-
                // curvature contribution, producing an inconsistent off-
                // diagonal block. Mirrors the dense-rows kernel's sign-correct
                // accumulation a few hundred lines above.
                let wi = weights[start + local];
                if wi == 0.0 {
                    continue;
                }
                for a in 0..pi {
                    let scaled = wi * left_chunk[[local, a]];
                    if scaled == 0.0 {
                        continue;
                    }
                    for b in 0..pj {
                        cross[[a, b]] += scaled * right_chunk[[local, b]];
                    }
                }
            }
        }
        Ok(cross)
    }

    fn quadratic_form_diag_cross_chunked(
        &self,
        block_a: &DesignBlock,
        block_b: &DesignBlock,
        m_ab: &Array2<f64>,
    ) -> Result<Array1<f64>, String> {
        let mut out = Array1::<f64>::zeros(self.n);
        for start in (0..self.n).step_by(OPERATOR_ROW_CHUNK_SIZE) {
            let end = (start + OPERATOR_ROW_CHUNK_SIZE).min(self.n);
            let a_chunk = block_a
                .try_row_chunk(start..end)
                .map_err(|e| e.to_string())?;
            let b_chunk = block_b
                .try_row_chunk(start..end)
                .map_err(|e| e.to_string())?;
            let a_m = fast_ab(&a_chunk, m_ab);
            for local in 0..(end - start) {
                out[start + local] = a_m.row(local).dot(&b_chunk.row(local));
            }
        }
        Ok(out)
    }

    /// Compute the cross-block X_i' diag(w) X_j for blocks i < j.
    fn cross_block(
        &self,
        i: usize,
        j: usize,
        weights: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        match (&self.blocks[i], &self.blocks[j]) {
            // ── Dense × Dense ───────────────────────────────────────────
            (DesignBlock::Dense(d_i), DesignBlock::Dense(d_j)) => {
                if let (Some(xi), Some(xj)) = (d_i.as_dense_ref(), d_j.as_dense_ref()) {
                    weighted_crossprod_dense(xi, weights, xj)
                } else {
                    self.weighted_cross_chunked(&self.blocks[i], &self.blocks[j], weights)
                }
            }
            (DesignBlock::Dense(_), DesignBlock::Sparse(_))
            | (DesignBlock::Sparse(_), DesignBlock::Dense(_))
            | (DesignBlock::Sparse(_), DesignBlock::Sparse(_))
            | (DesignBlock::Sparse(_), DesignBlock::RandomEffect(_))
            | (DesignBlock::RandomEffect(_), DesignBlock::Sparse(_)) => {
                self.weighted_cross_chunked(&self.blocks[i], &self.blocks[j], weights)
            }

            // ── Dense × RandomEffect ────────────────────────────────────
            (DesignBlock::Dense(d), DesignBlock::RandomEffect(re)) => {
                if let Some(dense) = d.as_dense_ref() {
                    re.weighted_cross_with_dense(dense, weights)
                } else {
                    self.weighted_cross_chunked(&self.blocks[i], &self.blocks[j], weights)
                }
            }
            (DesignBlock::RandomEffect(re), DesignBlock::Dense(d)) => {
                if let Some(dense) = d.as_dense_ref() {
                    let cross_t = re.weighted_cross_with_dense(dense, weights)?;
                    Ok(cross_t.t().to_owned())
                } else {
                    self.weighted_cross_chunked(&self.blocks[i], &self.blocks[j], weights)
                }
            }

            // ── RandomEffect × RandomEffect ─────────────────────────────
            (DesignBlock::RandomEffect(re_a), DesignBlock::RandomEffect(re_b)) => {
                re_a.weighted_cross_with_re(re_b, weights)
            }

            // ── Intercept × anything ────────────────────────────────────
            // 1'·diag(w)·B_j  →  (1 × p_j) where entry [0,c] = Σ_i w[i] * B_j[i,c]
            (DesignBlock::Intercept(_), other) => {
                // Signed w (no `.max(0.0)`) — see the sign-honest `diag_xtw_x`
                // contract; this cross term feeds the same XᵀWX assembly.
                let pj = other.ncols();
                let mut cross = Array2::<f64>::zeros((1, pj));
                let row = other.apply_transpose(weights);
                cross.row_mut(0).assign(&row);
                Ok(cross)
            }
            (other, DesignBlock::Intercept(_)) => {
                let pi = other.ncols();
                let mut cross = Array2::<f64>::zeros((pi, 1));
                let col = other.apply_transpose(weights);
                cross.column_mut(0).assign(&col);
                Ok(cross)
            }
        }
    }

    /// Diagonal contribution diag(X_k M X_k') for a single block.
    fn quadratic_form_diag_block(
        &self,
        block: &DesignBlock,
        m_kk: &Array2<f64>,
    ) -> Result<Array1<f64>, String> {
        match block {
            DesignBlock::Dense(d) => {
                if let Some(dense) = d.as_dense_ref() {
                    let xm = fast_ab(dense, m_kk);
                    let mut out = Array1::<f64>::zeros(self.n);
                    ndarray::Zip::from(&mut out)
                        .and(dense.rows())
                        .and(xm.rows())
                        .par_for_each(|o, dr, xmr| *o = dr.dot(&xmr));
                    Ok(out)
                } else {
                    d.quadratic_form_diag(m_kk)
                }
            }
            DesignBlock::Sparse(s) => {
                let sparse = DesignMatrix::Sparse(s.clone());
                sparse.quadratic_form_diag(m_kk)
            }
            DesignBlock::RandomEffect(re) => {
                use rayon::prelude::*;
                let out: Vec<f64> = re
                    .group_ids
                    .par_iter()
                    .map(|g| g.map(|g| m_kk[[g, g]]).unwrap_or(0.0))
                    .collect();
                Ok(Array1::from(out))
            }
            DesignBlock::Intercept(_) => {
                // Row i of intercept block is [1], so contribution = M[0,0] for all i.
                Ok(Array1::from_elem(self.n, m_kk[[0, 0]]))
            }
        }
    }

    /// Cross-block contribution diag(X_a M_ab X_b') for two distinct blocks.
    fn quadratic_form_diag_cross(
        &self,
        block_a: &DesignBlock,
        block_b: &DesignBlock,
        m_ab: &Array2<f64>,
    ) -> Result<Array1<f64>, String> {
        match (block_a, block_b) {
            (DesignBlock::Dense(da), DesignBlock::Dense(db)) => {
                if let (Some(da), Some(db)) = (da.as_dense_ref(), db.as_dense_ref()) {
                    let da_m = fast_ab(da, m_ab);
                    let mut out = Array1::<f64>::zeros(self.n);
                    ndarray::Zip::from(&mut out)
                        .and(da_m.rows())
                        .and(db.rows())
                        .par_for_each(|o, ar, br| *o = ar.dot(&br));
                    Ok(out)
                } else {
                    self.quadratic_form_diag_cross_chunked(block_a, block_b, m_ab)
                }
            }
            (DesignBlock::Dense(_), DesignBlock::Sparse(_))
            | (DesignBlock::Sparse(_), DesignBlock::Dense(_))
            | (DesignBlock::Sparse(_), DesignBlock::Sparse(_))
            | (DesignBlock::Sparse(_), DesignBlock::RandomEffect(_))
            | (DesignBlock::RandomEffect(_), DesignBlock::Sparse(_)) => {
                self.quadratic_form_diag_cross_chunked(block_a, block_b, m_ab)
            }
            (DesignBlock::Dense(d), DesignBlock::RandomEffect(re)) => {
                let mut out = Array1::<f64>::zeros(self.n);
                for start in (0..self.n).step_by(OPERATOR_ROW_CHUNK_SIZE) {
                    let end = (start + OPERATOR_ROW_CHUNK_SIZE).min(self.n);
                    let chunk = d.try_row_chunk(start..end).map_err(|e| e.to_string())?;
                    for local in 0..chunk.nrows() {
                        let i = start + local;
                        if let Some(g) = re.group_ids[i] {
                            let mut val = 0.0;
                            for j in 0..chunk.ncols() {
                                val += chunk[[local, j]] * m_ab[[j, g]];
                            }
                            out[i] = val;
                        }
                    }
                }
                Ok(out)
            }
            (DesignBlock::RandomEffect(re), DesignBlock::Dense(d)) => {
                let mut out = Array1::<f64>::zeros(self.n);
                for start in (0..self.n).step_by(OPERATOR_ROW_CHUNK_SIZE) {
                    let end = (start + OPERATOR_ROW_CHUNK_SIZE).min(self.n);
                    let chunk = d.try_row_chunk(start..end).map_err(|e| e.to_string())?;
                    for local in 0..chunk.nrows() {
                        let i = start + local;
                        if let Some(g) = re.group_ids[i] {
                            let mut val = 0.0;
                            for j in 0..chunk.ncols() {
                                val += m_ab[[g, j]] * chunk[[local, j]];
                            }
                            out[i] = val;
                        }
                    }
                }
                Ok(out)
            }
            (DesignBlock::RandomEffect(re_a), DesignBlock::RandomEffect(re_b)) => {
                use rayon::prelude::*;
                let out: Vec<f64> = re_a
                    .group_ids
                    .par_iter()
                    .zip(re_b.group_ids.par_iter())
                    .map(|(ga, gb)| match (ga, gb) {
                        (Some(ga), Some(gb)) => m_ab[[*ga, *gb]],
                        _ => 0.0,
                    })
                    .collect();
                Ok(Array1::from(out))
            }

            // Intercept × anything: contribution at row i = m_ab[0, :] · row_i(B_b)
            (DesignBlock::Intercept(_), other) => {
                let m_row = m_ab.row(0);
                let mut out = Array1::<f64>::zeros(self.n);
                for start in (0..self.n).step_by(OPERATOR_ROW_CHUNK_SIZE) {
                    let end = (start + OPERATOR_ROW_CHUNK_SIZE).min(self.n);
                    let chunk = other.try_row_chunk(start..end).map_err(|e| e.to_string())?;
                    for local in 0..(end - start) {
                        out[start + local] = chunk.row(local).dot(&m_row);
                    }
                }
                Ok(out)
            }
            (other, DesignBlock::Intercept(_)) => {
                let m_col = m_ab.column(0);
                let mut out = Array1::<f64>::zeros(self.n);
                for start in (0..self.n).step_by(OPERATOR_ROW_CHUNK_SIZE) {
                    let end = (start + OPERATOR_ROW_CHUNK_SIZE).min(self.n);
                    let chunk = other.try_row_chunk(start..end).map_err(|e| e.to_string())?;
                    for local in 0..(end - start) {
                        out[start + local] = chunk.row(local).dot(&m_col);
                    }
                }
                Ok(out)
            }
        }
    }
}

impl LinearOperator for BlockDesignOperator {
    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.total_cols
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.n);
        for (idx, block) in self.blocks.iter().enumerate() {
            let start = self.col_offsets[idx];
            let end = self.col_offsets[idx + 1];
            let slice = vector.slice(s![start..end]).to_owned();
            let contribution = block.apply(&slice);
            out += &contribution;
        }
        out
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.total_cols);
        for (idx, block) in self.blocks.iter().enumerate() {
            let start = self.col_offsets[idx];
            let end = self.col_offsets[idx + 1];
            let transposed = block.apply_transpose(vector);
            out.slice_mut(s![start..end]).assign(&transposed);
        }
        out
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        certify_signed_weights("BlockDesignOperator::diag_xtw_x", weights, self.n)?;
        let p = self.total_cols;
        let mut result = Array2::<f64>::zeros((p, p));

        // Diagonal blocks.
        for (idx, block) in self.blocks.iter().enumerate() {
            let start = self.col_offsets[idx];
            let end = self.col_offsets[idx + 1];
            let block_xtwx = block.diag_xtw_x(weights)?;
            result
                .slice_mut(s![start..end, start..end])
                .assign(&block_xtwx);
        }

        // Cross blocks (i, j) for i < j.
        //
        // Perf (#1017): the shared weight-scaled design `diag(w)·X_i` is
        // identical across every pairing `(i, j>i)`. The prior code recomputed
        // it inside each `cross_block` call (re-scaling X_i by w once per
        // partner j) and folded the product with a naive O(n·p_i·p_j) triple
        // loop. We now scale each dense block by `w` exactly ONCE up front and
        // route every Dense×Dense pair through a single blocked BLAS GEMM
        // (`fast_atb`), collapsing the c² hand-rolled accumulations into c²
        // batched matmuls over a design that is weight-scaled c times instead
        // of O(c²) times. Non-dense pairs keep their specialized kernels.
        //
        // Bit-identity: `cross[a,b] = Σ_i w_i · X_i[i,a] · X_j[i,b]` is exactly
        // `(diag(w)·X_i)ᵀ · X_j`, so pre-scaling then GEMM is the same sum,
        // reassociated only by the matmul's blocking (≤1e-10).
        let weighted_dense: Vec<Option<Array2<f64>>> = self
            .blocks
            .iter()
            .map(|block| match block {
                DesignBlock::Dense(d) => d.as_dense_ref().map(|x| {
                    // diag(w)·X computed once; signed w (no .max(0.0)) to match
                    // the asymmetric cross-block kernel's sign-correct form.
                    x * &weights.view().insert_axis(Axis(1))
                }),
                _ => None,
            })
            .collect();

        for i in 0..self.blocks.len() {
            for j in (i + 1)..self.blocks.len() {
                let cross = match (&weighted_dense[i], &self.blocks[j]) {
                    // Fused Dense×Dense: single GEMM over the shared,
                    // already-once-scaled left design.
                    (Some(wx_i), DesignBlock::Dense(d_j)) => match d_j.as_dense_ref() {
                        Some(x_j) => fast_atb(wx_i, x_j),
                        None => self.cross_block(i, j, weights)?,
                    },
                    _ => self.cross_block(i, j, weights)?,
                };
                let si = self.col_offsets[i];
                let ei = self.col_offsets[i + 1];
                let sj = self.col_offsets[j];
                let ej = self.col_offsets[j + 1];
                result.slice_mut(s![si..ei, sj..ej]).assign(&cross);
                result.slice_mut(s![sj..ej, si..ei]).assign(&cross.t());
            }
        }

        Ok(result)
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        certify_signed_weights("BlockDesignOperator::diag_gram", weights, self.n)?;
        let mut out = Array1::<f64>::zeros(self.total_cols);
        for (idx, block) in self.blocks.iter().enumerate() {
            let start = self.col_offsets[idx];
            let end = self.col_offsets[idx + 1];
            let block_diag = block.diag_gram(weights)?;
            out.slice_mut(s![start..end]).assign(&block_diag);
        }
        Ok(out)
    }

    fn apply_weighted_normal(
        &self,
        weights: FiniteSignedWeightsView<'_>,
        vector: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge: f64,
    ) -> Array1<f64> {
        assert_eq!(
            weights.len(),
            self.n,
            "BlockDesignOperator::apply_weighted_normal weight length mismatch"
        );
        assert_eq!(
            vector.len(),
            self.total_cols,
            "BlockDesignOperator::apply_weighted_normal vector length mismatch"
        );
        // Fused: X'W(Xβ) + Sβ + ridge·β
        let weights = weights.view();
        let xv = self.apply(vector);
        let mut weighted = xv;
        for i in 0..weighted.len() {
            weighted[i] *= weights[i];
        }
        let mut out = self.apply_transpose(&weighted);
        if let Some(pen) = penalty {
            out += &fast_av(pen, vector);
        }
        if ridge > 0.0 {
            // BLAS axpy: out += ridge * vector, no temporary allocation.
            out.scaled_add(ridge, vector);
        }
        out
    }

    fn uses_matrix_free_pcg(&self) -> bool {
        // Enable PCG when any block is non-dense (RE, Operator, or Intercept).
        self.blocks
            .iter()
            .any(|b| matches!(b, DesignBlock::RandomEffect(_) | DesignBlock::Intercept(_)))
    }
}

impl DenseDesignOperator for BlockDesignOperator {
    fn materialization_policy(&self) -> Option<MaterializationPolicy> {
        self.blocks.iter().fold(None, |policy, block| {
            merge_operator_materialization_policies(policy, block.materialization_policy())
        })
    }

    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
        if weights.len() != self.n || y.len() != self.n {
            return Err(format!(
                "BlockDesignOperator::compute_xtwy dimension mismatch: weights={}, y={}, nrows={}",
                weights.len(),
                y.len(),
                self.n
            ));
        }
        certify_signed_weights("BlockDesignOperator::compute_xtwy", weights, self.n)?;
        let mut wy = Array1::<f64>::zeros(self.n);
        ndarray::Zip::from(&mut wy)
            .and(weights)
            .and(y)
            .par_for_each(|o, &w, &yi| *o = w * yi);
        Ok(self.apply_transpose(&wy))
    }

    fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        // diag(X M X'): for each observation i, compute row_i(X) · M · row_i(X)'.
        // With block structure, this decomposes into diagonal and cross-block terms.
        let mut out = Array1::<f64>::zeros(self.n);
        let nb = self.blocks.len();

        // Diagonal contributions: diag(X_k M_kk X_k')
        for k in 0..nb {
            let sk = self.col_offsets[k];
            let ek = self.col_offsets[k + 1];
            let m_kk = middle.slice(s![sk..ek, sk..ek]).to_owned();
            let block_diag = self.quadratic_form_diag_block(&self.blocks[k], &m_kk)?;
            out += &block_diag;
        }

        // Cross-block contributions: 2·diag(X_a M_ab X_b')
        for a in 0..nb {
            for b in (a + 1)..nb {
                let sa = self.col_offsets[a];
                let ea = self.col_offsets[a + 1];
                let sb = self.col_offsets[b];
                let eb = self.col_offsets[b + 1];
                let m_ab = middle.slice(s![sa..ea, sb..eb]);

                let cross_diag = self.quadratic_form_diag_cross(
                    &self.blocks[a],
                    &self.blocks[b],
                    &m_ab.to_owned(),
                )?;
                for i in 0..self.n {
                    out[i] += 2.0 * cross_diag[i];
                }
            }
        }

        // Clamp to non-negative (variance-like quantity).
        for v in out.iter_mut() {
            *v = v.max(0.0);
        }
        Ok(out)
    }

    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if out.nrows() != rows.end - rows.start || out.ncols() != self.total_cols {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "BlockDesignOperator::row_chunk_into shape mismatch",
            });
        }
        for (idx, block) in self.blocks.iter().enumerate() {
            let cs = self.col_offsets[idx];
            let ce = self.col_offsets[idx + 1];
            block.row_chunk_into(rows.clone(), out.slice_mut(s![.., cs..ce]))?;
        }
        Ok(())
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.n, self.total_cols));
        for (idx, block) in self.blocks.iter().enumerate() {
            let start = self.col_offsets[idx];
            let end = self.col_offsets[idx + 1];
            let dense_block = block.to_dense();
            out.slice_mut(s![.., start..end]).assign(&dense_block);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// MultiChannelOperator
// ---------------------------------------------------------------------------

/// Multi-channel design operator: presents k views of shape (n, p) as a single
/// (k*n, p) operator without materializing the stacked matrix.
///
/// Primary use: survival time blocks with entry/exit/derivative channels.
/// Each channel contributes independently to matvecs and Gram assembly:
///
///   apply(β) = [X₀ β; X₁ β; …; X_{k-1} β]      (concatenated)
///   apply_transpose(v) = Σᵢ Xᵢᵀ vᵢ              (summed over channel slices)
///   X'WX = Σᵢ Xᵢᵀ diag(wᵢ) Xᵢ                  (summed over channel slices)
#[derive(Clone)]
pub struct MultiChannelOperator {
    /// Per-channel design matrices, each (n, p).
    pub channels: Vec<DesignMatrix>,
    /// Number of rows per channel (all channels must share the same n).
    pub n_per_channel: usize,
    /// Number of columns (shared across all channels).
    pub p: usize,
}

impl MultiChannelOperator {
    pub fn new(channels: Vec<DesignMatrix>) -> Result<Self, String> {
        if channels.is_empty() {
            return Err("MultiChannelOperator: need at least one channel".to_string());
        }
        let n = channels[0].nrows();
        let p = channels[0].ncols();
        for (i, ch) in channels.iter().enumerate() {
            if ch.nrows() != n {
                return Err(format!(
                    "MultiChannelOperator: channel {i} has {} rows, expected {n}",
                    ch.nrows()
                ));
            }
            if ch.ncols() != p {
                return Err(format!(
                    "MultiChannelOperator: channel {i} has {} cols, expected {p}",
                    ch.ncols()
                ));
            }
        }
        Ok(Self {
            channels,
            n_per_channel: n,
            p,
        })
    }
}

impl LinearOperator for MultiChannelOperator {
    fn nrows(&self) -> usize {
        self.n_per_channel * self.channels.len()
    }

    fn ncols(&self) -> usize {
        self.p
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        let total = self.nrows();
        let mut out = Array1::<f64>::zeros(total);
        let n = self.n_per_channel;
        for (i, ch) in self.channels.iter().enumerate() {
            let ch_result = ch.matrixvectormultiply(vector);
            out.slice_mut(s![i * n..(i + 1) * n]).assign(&ch_result);
        }
        out
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        let n = self.n_per_channel;
        let mut out = Array1::<f64>::zeros(self.p);
        for (i, ch) in self.channels.iter().enumerate() {
            out += &ch.apply_transpose_view(vector.slice(s![i * n..(i + 1) * n]));
        }
        out
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        let n = self.n_per_channel;
        certify_signed_weights("MultiChannelOperator::diag_xtw_x", weights, self.nrows())?;
        let mut xtwx = Array2::<f64>::zeros((self.p, self.p));
        for (i, ch) in self.channels.iter().enumerate() {
            let channel_weights = weights.slice(s![i * n..(i + 1) * n]).to_owned();
            let ch_xtwx = ch.diag_xtw_x(&channel_weights)?;
            xtwx += &ch_xtwx;
        }
        Ok(xtwx)
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        let n = self.n_per_channel;
        certify_signed_weights("MultiChannelOperator::diag_gram", weights, self.nrows())?;
        let mut diag = Array1::<f64>::zeros(self.p);
        for (i, ch) in self.channels.iter().enumerate() {
            diag += &ch.diag_gram_view(weights.slice(s![i * n..(i + 1) * n]))?;
        }
        Ok(diag)
    }

    fn uses_matrix_free_pcg(&self) -> bool {
        true
    }
}

impl DenseDesignOperator for MultiChannelOperator {
    fn materialization_policy(&self) -> Option<MaterializationPolicy> {
        self.channels.iter().fold(None, |policy, channel| {
            merge_operator_materialization_policies(policy, channel.materialization_policy())
        })
    }

    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
        let n = self.n_per_channel;
        let total = self.nrows();
        if weights.len() != total || y.len() != total {
            return Err(format!(
                "MultiChannelOperator::compute_xtwy: weights={}, y={}, nrows={}",
                weights.len(),
                y.len(),
                total
            ));
        }
        certify_signed_weights("MultiChannelOperator::compute_xtwy", weights, total)?;
        let mut out = Array1::<f64>::zeros(self.p);
        for (i, ch) in self.channels.iter().enumerate() {
            out += &ch.compute_xtwy_view(
                weights.slice(s![i * n..(i + 1) * n]),
                y.slice(s![i * n..(i + 1) * n]),
            )?;
        }
        Ok(out)
    }

    fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        let n = self.n_per_channel;
        let mut out = Array1::<f64>::zeros(self.nrows());
        for (i, ch) in self.channels.iter().enumerate() {
            let ch_diag = ch.quadratic_form_diag(middle)?;
            out.slice_mut(s![i * n..(i + 1) * n]).assign(&ch_diag);
        }
        Ok(out)
    }

    fn to_dense(&self) -> Array2<f64> {
        let total = self.nrows();
        let n = self.n_per_channel;
        let mut out = Array2::<f64>::zeros((total, self.p));
        for (i, ch) in self.channels.iter().enumerate() {
            let dense = ch.to_dense();
            out.slice_mut(s![i * n..(i + 1) * n, ..]).assign(&dense);
        }
        out
    }

    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if out.nrows() != rows.end - rows.start || out.ncols() != self.p {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "MultiChannelOperator::row_chunk_into shape mismatch",
            });
        }
        let n = self.n_per_channel;
        let mut local = 0usize;
        let mut global = rows.start;
        while global < rows.end {
            let ch_idx = global / n;
            let ch_local_start = global % n;
            let ch_local_end = ((ch_idx + 1) * n).min(rows.end) - ch_idx * n;
            let segment_len = ch_local_end - ch_local_start;
            self.channels[ch_idx].row_chunk_into(
                ch_local_start..ch_local_end,
                out.slice_mut(s![local..local + segment_len, ..]),
            )?;
            local += segment_len;
            global += segment_len;
        }
        Ok(())
    }
}

// Rowwise-Kronecker + tensor-product design operators (#1145): see `kronecker.rs`.
mod kronecker;
pub use kronecker::*;

/// Coefficient-side transform operator: represents X_eff = X_inner * T without
/// materializing the product. Preserves the sparsity/operator structure of the
/// inner design by applying T on the coefficient side:
///   apply(v) = X_inner * (T * v)
///   apply_transpose(v) = T^T * (X_inner^T * v)
///   diag_xtw_x(w) = T^T * (X_inner^T W X_inner) * T
pub struct CoefficientTransformOperator {
    inner: DenseDesignMatrix,
    transform: Arc<Array2<f64>>,
    n: usize,
    p_out: usize,
    /// One-time-materialized X · T dense block for an already-materialized
    /// inner design. An operator-backed inner has already been routed to a
    /// bounded-memory representation, so it must never populate this cache.
    materialized: OnceLock<Option<Arc<Array2<f64>>>>,
}

impl CoefficientTransformOperator {
    /// Maximum bytes for the one-shot X · T materialization of an
    /// already-materialized inner design.
    const MATERIALIZE_MAX_BYTES: usize = 1024 * 1024 * 1024;

    pub fn new(inner: DenseDesignMatrix, transform: Array2<f64>) -> Result<Self, String> {
        let p_inner = inner.ncols();
        if transform.nrows() != p_inner {
            return Err(format!(
                "CoefficientTransformOperator: inner has {} cols but transform has {} rows",
                p_inner,
                transform.nrows(),
            ));
        }
        let n = inner.nrows();
        let p_out = transform.ncols();
        Ok(Self {
            inner,
            transform: Arc::new(transform),
            n,
            p_out,
            materialized: OnceLock::new(),
        })
    }

    /// Get-or-build the materialized X · T dense block. Operator-backed
    /// inputs return `None` unconditionally: a coefficient transform preserves
    /// the inner design's lazy storage decision instead of bypassing it through
    /// an unrelated local byte ceiling.
    fn materialized_combined(&self) -> Option<&Array2<f64>> {
        if let Some(slot) = self.materialized.get() {
            return slot.as_ref().map(|a| a.as_ref());
        }
        if self.inner.is_operator_backed() {
            if self.materialized.set(None).is_err() {
                return self
                    .materialized
                    .get()
                    .and_then(|opt| opt.as_ref().map(|a| a.as_ref()));
            }
            return None;
        }
        let bytes = self
            .n
            .checked_mul(self.p_out)
            .and_then(|cells| cells.checked_mul(std::mem::size_of::<f64>()));
        let computed = match bytes {
            Some(b) if b <= Self::MATERIALIZE_MAX_BYTES => self
                .inner
                .as_dense_ref()
                .map(|x| Arc::new(fast_ab(x, &self.transform))),
            _ => None,
        };
        if self.materialized.set(computed).is_err() {
            return self
                .materialized
                .get()
                .and_then(|opt| opt.as_ref().map(|a| a.as_ref()));
        }
        self.materialized
            .get()
            .and_then(|opt| opt.as_ref().map(|a| a.as_ref()))
    }
}

impl LinearOperator for CoefficientTransformOperator {
    fn nrows(&self) -> usize {
        self.n
    }
    fn ncols(&self) -> usize {
        self.p_out
    }
    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        if let Some(combined) = self.materialized_combined() {
            return fast_av(combined, vector);
        }
        let tv = fast_av(&self.transform, vector);
        self.inner.apply(&tv)
    }
    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        if let Some(combined) = self.materialized_combined() {
            return fast_atv(combined, vector);
        }
        let xtv = self.inner.apply_transpose(vector);
        fast_atv(&self.transform, &xtv)
    }
    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        certify_signed_weights("CoefficientTransformOperator::diag_xtw_x", weights, self.n)?;
        if let Some(combined) = self.materialized_combined() {
            let mut xtwx = Array2::<f64>::zeros((self.p_out, self.p_out));
            stream_weighted_crossprod_into(
                combined,
                weights,
                &mut xtwx,
                CrossprodStructure::Full,
                CrossprodAccum::Replace,
                effective_global_parallelism(),
            );
            return Ok(xtwx);
        }
        let inner_xtwx = self.inner.diag_xtw_x(weights)?;
        // T^T * (X^T W X) * T
        let tmp = fast_ab(&self.transform.t().to_owned(), &inner_xtwx);
        Ok(fast_ab(&tmp, &self.transform))
    }
}

impl DenseDesignOperator for CoefficientTransformOperator {
    /// Expose the cached X·T materialization when populated. This is what lets
    /// `BlockDesignOperator::cross_block` recognize a Dense × Dense pair and
    /// route to `weighted_crossprod_dense` (BLAS-3 GEMM) instead of the
    /// scalar `weighted_cross_chunked` triple loop. Without this override the
    /// default trait impl returns `None`, the fast path is skipped, and a
    /// 4-block large-scale fit (pgs + sex + smooth_age + duchon) pays a 24 s
    /// cross-block cost per PIRLS curvature build.
    fn as_dense_ref(&self) -> Option<&Array2<f64>> {
        self.materialized_combined()
    }

    fn materialization_policy(&self) -> Option<MaterializationPolicy> {
        self.inner.materialization_policy()
    }

    fn to_dense(&self) -> Array2<f64> {
        if let Some(combined) = self.materialized_combined() {
            return combined.clone();
        }
        let x = self.inner.to_dense();
        fast_ab(&x, &self.transform)
    }
    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if out.nrows() != rows.end - rows.start || out.ncols() != self.p_out {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "CoefficientTransformOperator::row_chunk_into shape mismatch",
            });
        }
        if let Some(combined) = self.materialized_combined() {
            out.assign(&combined.slice(s![rows, ..]));
            return Ok(());
        }
        let chunk = self.inner.try_row_chunk(rows)?;
        out.assign(&fast_ab(&chunk, &self.transform));
        Ok(())
    }
}

/// SMGS Phase 4b residualised design operator: emits the mathematically-exact
/// row `C_b · V_b − Σ_{a<b} A_a · R_{a,b}` for block `b`, where `V_b` is the
/// kept-direction reparametrisation of the inner raw block `C_b` and each
/// `R_{a,b} = M_{a,b} · V_b` is the precomputed residualised contribution of
/// an earlier anchor design `A_a` against block `b`.
///
/// The operator presents shape `(n × V_b.ncols())` so the rest of the design
/// stack sees the compiled (kept) width. Row-chunk emission computes
/// `inner_chunk · V_b` and then subtracts `anchor_chunk · r_block` for every
/// anchor pair. The combined `n × kept` block is cached via `OnceLock` under
/// the same `MATERIALIZE_MAX_BYTES = 1 GiB` ceiling as
/// [`CoefficientTransformOperator`] only when every input is already
/// materialized. Any operator-backed input keeps the result row-chunked.
pub struct ResidualisedDesignOperator {
    inner: DenseDesignMatrix,
    transform: Arc<Array2<f64>>,
    anchors: Vec<(DesignMatrix, Arc<Array2<f64>>)>,
    n: usize,
    p_out: usize,
    materialized: OnceLock<Option<Arc<Array2<f64>>>>,
}

impl ResidualisedDesignOperator {
    /// Matches `CoefficientTransformOperator::MATERIALIZE_MAX_BYTES` for
    /// already-materialized inputs.
    const MATERIALIZE_MAX_BYTES: usize = 1024 * 1024 * 1024;

    pub fn new(
        inner: DenseDesignMatrix,
        transform: Array2<f64>,
        anchors: Vec<(DesignMatrix, Arc<Array2<f64>>)>,
    ) -> Result<Self, String> {
        let p_inner = inner.ncols();
        if transform.nrows() != p_inner {
            return Err(format!(
                "ResidualisedDesignOperator: inner has {} cols but transform has {} rows",
                p_inner,
                transform.nrows(),
            ));
        }
        let n = inner.nrows();
        let p_out = transform.ncols();
        for (idx, (anchor, r_block)) in anchors.iter().enumerate() {
            if anchor.nrows() != n {
                return Err(format!(
                    "ResidualisedDesignOperator: anchor[{idx}] has {} rows but inner has {n}",
                    anchor.nrows(),
                ));
            }
            if r_block.nrows() != anchor.ncols() || r_block.ncols() != p_out {
                return Err(format!(
                    "ResidualisedDesignOperator: anchor[{idx}] r_block is {}x{} but expected {}x{}",
                    r_block.nrows(),
                    r_block.ncols(),
                    anchor.ncols(),
                    p_out,
                ));
            }
        }
        Ok(Self {
            inner,
            transform: Arc::new(transform),
            anchors,
            n,
            p_out,
            materialized: OnceLock::new(),
        })
    }

    /// Get-or-build the cached n × p_out materialised block. A lazy inner or
    /// anchor returns `None` so residualisation cannot reverse an upstream
    /// bounded-memory routing decision.
    fn materialized_combined(&self) -> Option<&Array2<f64>> {
        if let Some(slot) = self.materialized.get() {
            return slot.as_ref().map(|a| a.as_ref());
        }
        if self.inner.is_operator_backed()
            || self
                .anchors
                .iter()
                .any(|(anchor, _)| anchor.is_operator_backed())
        {
            if self.materialized.set(None).is_err() {
                return self
                    .materialized
                    .get()
                    .and_then(|opt| opt.as_ref().map(|a| a.as_ref()));
            }
            return None;
        }
        let bytes = self
            .n
            .checked_mul(self.p_out)
            .and_then(|cells| cells.checked_mul(std::mem::size_of::<f64>()));
        let computed = match bytes {
            Some(b) if b <= Self::MATERIALIZE_MAX_BYTES => {
                let auto_policy =
                    ResourcePolicy::for_problem(gam_runtime::resource::ProblemHints::default());
                let cache_policy = ResourcePolicy {
                    max_single_materialization_bytes: Self::MATERIALIZE_MAX_BYTES,
                    derivative_storage_mode: auto_policy.derivative_storage_mode,
                    ..ResourcePolicy::default_library()
                };
                self.inner
                    .try_to_dense_arc_with_policy(
                        "ResidualisedDesignOperator materialization",
                        &cache_policy,
                    )
                    .ok()
                    .and_then(|x| {
                        let mut combined = fast_ab(x.as_ref(), &self.transform);
                        for (anchor, r_block) in &self.anchors {
                            let anchor_dense = match anchor {
                                DesignMatrix::Dense(d) => d
                                    .try_to_dense_arc_with_policy(
                                        "ResidualisedDesignOperator anchor materialization",
                                        &cache_policy,
                                    )
                                    .ok()?,
                                DesignMatrix::Sparse(s) => s
                                    .try_to_dense_arc(
                                        "ResidualisedDesignOperator anchor materialization",
                                    )
                                    .ok()?,
                            };
                            let contribution = fast_ab(anchor_dense.as_ref(), r_block.as_ref());
                            combined -= &contribution;
                        }
                        Some(Arc::new(combined))
                    })
            }
            _ => None,
        };
        if self.materialized.set(computed).is_err() {
            return self
                .materialized
                .get()
                .and_then(|opt| opt.as_ref().map(|a| a.as_ref()));
        }
        self.materialized
            .get()
            .and_then(|opt| opt.as_ref().map(|a| a.as_ref()))
    }

    /// Public lazy materialisation handle: returns the cached combined block
    /// when available, falling back to chunked row-wise materialisation via
    /// the operator-backed path on the caller's behalf.
    pub fn try_to_dense_arc(&self, context: &str) -> Result<Arc<Array2<f64>>, String> {
        if let Some(combined) = self.materialized.get().and_then(|opt| opt.clone()) {
            return Ok(combined);
        }
        if let Some(_combined_ref) = self.materialized_combined() {
            if let Some(arc) = self.materialized.get().and_then(|opt| opt.clone()) {
                return Ok(arc);
            }
        }
        dense_operator_to_dense_by_chunks(self)
            .map(Arc::new)
            .map_err(|err| format!("{context}: failed to materialize dense row chunks: {err}"))
    }
}

impl LinearOperator for ResidualisedDesignOperator {
    fn nrows(&self) -> usize {
        self.n
    }
    fn ncols(&self) -> usize {
        self.p_out
    }
    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        if let Some(combined) = self.materialized_combined() {
            return fast_av(combined, vector);
        }
        // y = C_b · (V_b · v) − Σ_a A_a · (R_{a,b} · v)
        let tv = fast_av(&self.transform, vector);
        let mut out = self.inner.apply(&tv);
        for (anchor, r_block) in &self.anchors {
            let rv = fast_av(r_block.as_ref(), vector);
            let contrib = anchor.apply(&rv);
            out -= &contrib;
        }
        out
    }
    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        if let Some(combined) = self.materialized_combined() {
            return fast_atv(combined, vector);
        }
        let xtv = self.inner.apply_transpose(vector);
        let mut out = fast_atv(&self.transform, &xtv);
        for (anchor, r_block) in &self.anchors {
            let atv = anchor.apply_transpose(vector);
            let contrib = fast_atv(r_block.as_ref(), &atv);
            out -= &contrib;
        }
        out
    }
    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        certify_signed_weights("ResidualisedDesignOperator::diag_xtw_x", weights, self.n)?;
        if let Some(combined) = self.materialized_combined() {
            let mut xtwx = Array2::<f64>::zeros((self.p_out, self.p_out));
            stream_weighted_crossprod_into(
                combined,
                weights,
                &mut xtwx,
                CrossprodStructure::Full,
                CrossprodAccum::Replace,
                effective_global_parallelism(),
            );
            return Ok(xtwx);
        }
        // Fall back to the default DenseDesignOperator chunked path via
        // explicit materialisation: emit chunks through row_chunk_into and
        // accumulate XᵀWX without ever holding the full n × p_out block.
        let n = self.n;
        if weights.len() != n {
            return Err(format!(
                "ResidualisedDesignOperator::diag_xtw_x weights len {} != nrows {n}",
                weights.len()
            ));
        }
        let p = self.p_out;
        let chunk_rows = (8 * 1024 * 1024 / (p.max(1) * 8 * 2)).max(16).min(n.max(1));
        let mut xtwx = Array2::<f64>::zeros((p, p));
        let mut start = 0;
        while start < n {
            let end = (start + chunk_rows).min(n);
            let chunk = <Self as DenseDesignOperator>::try_row_chunk(self, start..end)
                .map_err(|e| e.to_string())?;
            let w_slice = weights.slice(s![start..end]).to_owned();
            let mut local = Array2::<f64>::zeros((p, p));
            stream_weighted_crossprod_into(
                &chunk,
                &w_slice,
                &mut local,
                CrossprodStructure::Full,
                CrossprodAccum::Replace,
                effective_global_parallelism(),
            );
            xtwx += &local;
            start = end;
        }
        Ok(xtwx)
    }
}

impl DenseDesignOperator for ResidualisedDesignOperator {
    fn as_dense_ref(&self) -> Option<&Array2<f64>> {
        self.materialized_combined()
    }

    fn materialization_policy(&self) -> Option<MaterializationPolicy> {
        self.anchors.iter().fold(
            self.inner.materialization_policy(),
            |policy, (anchor, _)| {
                merge_operator_materialization_policies(policy, anchor.materialization_policy())
            },
        )
    }

    fn to_dense(&self) -> Array2<f64> {
        if let Some(combined) = self.materialized_combined() {
            return combined.clone();
        }
        // Chunked fallback when the cache refuses (oversize block).
        dense_operator_to_dense_by_chunks(self).unwrap_or_else(|err| {
            std::panic::panic_any(format!(
                "ResidualisedDesignOperator::to_dense: failed to materialize {}x{} \
                 via row chunks: {err}",
                self.n, self.p_out,
            ))
        })
    }

    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if out.nrows() != rows.end - rows.start || out.ncols() != self.p_out {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "ResidualisedDesignOperator::row_chunk_into shape mismatch",
            });
        }
        if let Some(combined) = self.materialized_combined() {
            out.assign(&combined.slice(s![rows, ..]));
            return Ok(());
        }
        // C_b chunk in raw width, then project: out = (inner_chunk) · V_b
        let inner_chunk = self.inner.try_row_chunk(rows.clone())?;
        let mut combined = fast_ab(&inner_chunk, &self.transform);
        // Subtract Σ_a (anchor_chunk · r_block)
        for (anchor, r_block) in &self.anchors {
            let anchor_chunk = anchor.try_row_chunk(rows.clone())?;
            let contribution = fast_ab(&anchor_chunk, r_block.as_ref());
            combined -= &contribution;
        }
        out.assign(&combined);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ConditionedDesign — lazy per-column affine transform
// ---------------------------------------------------------------------------

/// A design matrix wrapper that lazily applies per-column centering and scaling
/// without materializing a new dense matrix.
///
/// For each conditioned column `j`, the effective column is
/// `(X[:,j] - mean_j) / scale_j`.  All other columns pass through unchanged.
/// Algebraically this is `X·diag(a) - 1·d'` where `a[j] = 1/scale` for
/// conditioned columns (1 otherwise) and `d[j] = mean/scale` for conditioned
/// columns (0 otherwise).
pub struct ConditionedDesign {
    inner: DesignMatrix,
    /// Per-conditioned-column: (global_col_idx, mean, scale).
    columns: Vec<(usize, f64, f64)>,
}

impl ConditionedDesign {
    pub fn new(inner: DesignMatrix, columns: Vec<(usize, f64, f64)>) -> Self {
        Self { inner, columns }
    }
}

impl LinearOperator for ConditionedDesign {
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    /// X_c v = X(a⊙v) - (d·v)·1
    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        let mut scaled = vector.clone();
        let mut shift = 0.0;
        for &(j, mean, scale) in &self.columns {
            scaled[j] /= scale;
            shift += mean * scaled[j];
        }
        let mut result = self.inner.apply(&scaled);
        if shift != 0.0 {
            result.mapv_inplace(|v| v - shift);
        }
        result
    }

    /// X_c'u = a⊙(X'u) - d·Σu
    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        let mut result = self.inner.apply_transpose(vector);
        let sum_u: f64 = vector.iter().sum();
        for &(j, mean, scale) in &self.columns {
            result[j] = (result[j] - mean * sum_u) / scale;
        }
        result
    }

    /// X_c'WX_c = D_a(X'WX)D_a - D_a(X'w)d' - d(X'w)'D_a + Σw·dd'
    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        certify_signed_weights("ConditionedDesign::diag_xtw_x", weights, self.nrows())?;
        let mut base = self.inner.diag_xtw_x(weights)?;
        if self.columns.is_empty() {
            return Ok(base);
        }
        let p = base.ncols();
        let sum_w: f64 = weights.sum();
        let cw = self.inner.apply_transpose(weights);

        // Precompute a[j] and d[j] for all columns.
        let mut a = vec![1.0_f64; p];
        let mut d = vec![0.0_f64; p];
        for &(j, mean, scale) in &self.columns {
            a[j] = 1.0 / scale;
            d[j] = mean / scale;
        }

        // Apply the full transformation in one pass (symmetric).
        for i in 0..p {
            for j in i..p {
                let val = a[i] * base[[i, j]] * a[j] - a[i] * cw[i] * d[j] - d[i] * cw[j] * a[j]
                    + sum_w * d[i] * d[j];
                base[[i, j]] = val;
                base[[j, i]] = val;
            }
        }
        Ok(base)
    }

    /// Diagonal of X_c'WX_c — only conditioned columns change.
    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        certify_signed_weights("ConditionedDesign::diag_gram", weights, self.nrows())?;
        let mut result = self.inner.diag_gram(weights)?;
        if self.columns.is_empty() {
            return Ok(result);
        }
        let sum_w: f64 = weights.sum();
        let cw = self.inner.apply_transpose(weights);
        for &(j, mean, scale) in &self.columns {
            let a_j = 1.0 / scale;
            let d_j = mean / scale;
            result[j] = a_j * a_j * result[j] - 2.0 * a_j * cw[j] * d_j + sum_w * d_j * d_j;
        }
        Ok(result)
    }

    fn uses_matrix_free_pcg(&self) -> bool {
        match &self.inner {
            DesignMatrix::Dense(_) => true,
            DesignMatrix::Sparse(_) => false,
        }
    }
}

impl DenseDesignOperator for ConditionedDesign {
    fn materialization_policy(&self) -> Option<MaterializationPolicy> {
        self.inner.materialization_policy()
    }

    /// X_c'(w⊙y) = a⊙(X'(w⊙y)) - d·Σ(w⊙y)
    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
        if y.len() != self.nrows() {
            return Err(format!(
                "ConditionedDesign::compute_xtwy response length mismatch: y={}, nrows={}",
                y.len(),
                self.nrows()
            ));
        }
        certify_signed_weights("ConditionedDesign::compute_xtwy", weights, self.nrows())?;
        let mut result = self.inner.compute_xtwy(weights, y)?;
        if self.columns.is_empty() {
            return Ok(result);
        }
        let sum_wy: f64 = weights.iter().zip(y.iter()).map(|(&w, &yi)| w * yi).sum();
        for &(j, mean, scale) in &self.columns {
            result[j] = (result[j] - mean * sum_wy) / scale;
        }
        Ok(result)
    }

    /// diag(X_c M X_c') = diag(X(D_a M D_a)X') - 2·X(D_a M d) + d'Md
    fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        if self.columns.is_empty() {
            return self.inner.quadratic_form_diag(middle);
        }
        let p = self.ncols();
        let mut d = Array1::zeros(p);
        for &(j, mean, scale) in &self.columns {
            d[j] = mean / scale;
        }

        // D_a M D_a: scale rows and columns for conditioned indices.
        let mut ama = middle.clone();
        for &(j, _, scale) in &self.columns {
            for k in 0..p {
                ama[[j, k]] /= scale;
                ama[[k, j]] /= scale;
            }
        }

        // D_a M d
        let md = middle.dot(&d);
        let mut amd = md;
        for &(j, _, scale) in &self.columns {
            amd[j] /= scale;
        }

        let dtmd: f64 = d.dot(&middle.dot(&d));

        let mut result = self.inner.quadratic_form_diag(&ama)?;
        let x_amd = self.inner.apply(&amd);
        for i in 0..result.len() {
            result[i] = (result[i] - 2.0 * x_amd[i] + dtmd).max(0.0);
        }
        Ok(result)
    }

    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if out.nrows() != rows.end - rows.start || out.ncols() != self.ncols() {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "ConditionedDesign::row_chunk_into shape mismatch",
            });
        }
        let mut chunk = self.inner.try_row_chunk(rows)?;
        for &(j, mean, scale) in &self.columns {
            chunk.column_mut(j).mapv_inplace(|v| (v - mean) / scale);
        }
        out.assign(&chunk);
        Ok(())
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut dense = self.inner.to_dense();
        for &(j, mean, scale) in &self.columns {
            dense.column_mut(j).mapv_inplace(|v| (v - mean) / scale);
        }
        dense
    }
}

/// Unified design matrix representation for dense and sparse workflows.
///
/// Dense matrices are wrapped in Arc for O(1) cloning — at large scale
/// design matrices are 100-500MB and get cloned repeatedly during GAMLSS
/// family construction, warm-start caching, and prediction.
///
/// The `Dense` variant wraps both materialized dense matrices and lazy
/// dense-backed operators (`DenseDesignMatrix::Lazy`) that implement
/// `DenseDesignOperator` without reopening a third top-level storage state.
#[derive(Clone)]
pub enum DesignMatrix {
    Dense(DenseDesignMatrix),
    Sparse(SparseDesignMatrix),
}

impl std::fmt::Debug for DesignMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense(m) => write!(f, "DesignMatrix::Dense({}x{})", m.nrows(), m.ncols()),
            Self::Sparse(s) => write!(f, "DesignMatrix::Sparse({}x{})", s.nrows(), s.ncols()),
        }
    }
}

// Symmetric-matrix container + Gram assembly (#1145): see `symmetric.rs`.
mod symmetric;
pub use symmetric::*;
/// A generic abstraction over a factorized symmetric positive-definite (or regularized) system.
pub trait FactorizedSystem: Send + Sync {
    /// Solve $H x = b$ for a single right-hand side.
    fn solve(&self, rhs: &Array1<f64>) -> Result<Array1<f64>, String>;

    /// Solve $H X = B$ for multiple right-hand sides.
    fn solvemulti(&self, rhs: &Array2<f64>) -> Result<Array2<f64>, String>;

    /// Return the log-determinant of the factorized matrix.
    fn logdet(&self) -> f64;
}

pub trait LinearOperator {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn apply(&self, vector: &Array1<f64>) -> Array1<f64>;
    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64>;
    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String>;

    /// Observed-Hessian / non-canonical-link Gram: `XᵀWX` with sign-honest
    /// weights. Returns a dense `Array2<f64>` because the result is symmetric
    /// but not guaranteed PSD (so consumers cannot assume the `SymmetricMatrix`
    /// PSD contract). Default impl delegates to `diag_xtw_x` for legacy
    /// operators; overriding impls may take a sign-aware fast path.
    fn xt_diag_x_signed_op(
        &self,
        weights: FiniteSignedWeightsView<'_>,
    ) -> Result<Array2<f64>, String> {
        self.diag_xtw_x(&weights.view().to_owned())
    }

    /// PSD-precondition Gram: `XᵀWX` with `w ≥ 0` discharged at the
    /// `PsdWeightsView` constructor. Returns a typed `SymmetricMatrix` so
    /// downstream consumers can route through PSD-only solvers (Cholesky).
    /// Default impl wraps the signed path's `Array2` in `SymmetricMatrix::Dense`.
    fn xt_diag_x_psd_op(&self, weights: PsdWeightsView<'_>) -> Result<SymmetricMatrix, String> {
        FiniteSignedWeightsView::try_new(weights.view())
            .map_err(|reason| format!("LinearOperator::xt_diag_x_psd_op: {reason}"))?;
        let xtwx = self.diag_xtw_x(&weights.view().to_owned())?;
        Ok(SymmetricMatrix::Dense(xtwx))
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        let xtwx = self.diag_xtw_x(weights)?;
        Ok(Array1::from_iter((0..self.ncols()).map(|j| xtwx[[j, j]])))
    }
    fn apply_weighted_normal(
        &self,
        weights: FiniteSignedWeightsView<'_>,
        vector: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge: f64,
    ) -> Array1<f64> {
        assert_eq!(
            weights.len(),
            self.nrows(),
            "apply_weighted_normal weight length mismatch"
        );
        assert_eq!(
            vector.len(),
            self.ncols(),
            "apply_weighted_normal vector length mismatch"
        );
        let weights = weights.view();
        let xv = self.apply(vector);
        let mut weighted_xv = xv;
        for i in 0..weighted_xv.len() {
            weighted_xv[i] *= weights[i];
        }
        let mut out = self.apply_transpose(&weighted_xv);
        if let Some(pen) = penalty {
            out += &fast_av(pen, vector);
        }
        if ridge > 0.0 {
            // BLAS axpy: out += ridge * vector, no temporary allocation.
            out.scaled_add(ridge, vector);
        }
        out
    }
    fn uses_matrix_free_pcg(&self) -> bool {
        false
    }
    fn solve_system_matrix_free_pcg_try(
        &self,
        weights: &Array1<f64>,
        rhs: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        baseridge: f64,
    ) -> Result<Array1<f64>, String> {
        self.solve_system_matrix_free_pcg_with_info_try(weights, rhs, penalty, baseridge)
            .map(|(solution, _)| solution)
    }
    fn solve_system_matrix_free_pcg_with_info_try(
        &self,
        weights: &Array1<f64>,
        rhs: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        baseridge: f64,
    ) -> Result<(Array1<f64>, PcgSolveInfo), String> {
        if rhs.len() != self.ncols() {
            return Err(format!(
                "solve_system_matrix_free_pcg rhs dimension mismatch: rhs length {} != ncols {}",
                rhs.len(),
                self.ncols()
            ));
        }
        if !self.uses_matrix_free_pcg() {
            return Err("matrix-free PCG is only enabled for eligible operator types".to_string());
        }
        if let Some(pen) = penalty
            && (pen.nrows() != self.ncols() || pen.ncols() != self.ncols())
        {
            return Err(format!(
                "solve_system_matrix_free_pcg penalty shape mismatch: got {}x{}, expected {}x{}",
                pen.nrows(),
                pen.ncols(),
                self.ncols(),
                self.ncols()
            ));
        }
        let p = self.ncols();
        let finite_weights = certify_signed_weights(
            "solve_system_matrix_free_pcg_with_info_try",
            weights,
            self.nrows(),
        )?;
        if !(baseridge.is_finite() && baseridge >= 0.0) {
            return Err(format!(
                "matrix-free PCG ridge must be finite and non-negative, got {baseridge:?}"
            ));
        }
        let normal_op = PenalizedWeightedNormalOperator {
            operator: self,
            weights,
            finite_weights,
            penalty,
            ridge: baseridge,
        };
        let preconditioner = normal_op.jacobi_preconditioner()?;
        let attempt_started = std::time::Instant::now();
        let (solution, info) = crate::utils::solve_spd_pcg_with_info(
            |v| normal_op.apply(v),
            rhs,
            &preconditioner,
            MATRIX_FREE_PCG_REL_TOL,
            MATRIX_FREE_PCG_MAX_ITER.max(4 * p),
        )
        .ok_or_else(|| {
            format!("matrix-free PCG broke down for explicitly requested ridge {baseridge:.3e}")
        })?;
        if !solution.iter().all(|value| value.is_finite()) {
            return Err("matrix-free PCG produced a non-finite solution".to_string());
        }
        log::debug!(
            "[matrix-free PCG] solved: p={p} ridge={baseridge:.3e} iters={} converged={} rel_resid={:.3e} elapsed={:.3}s",
            info.iterations,
            info.converged,
            info.relative_residual_norm,
            attempt_started.elapsed().as_secs_f64(),
        );
        Ok((solution, info))
    }
    fn factorize_system(
        &self,
        weights: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
    ) -> Result<Box<dyn FactorizedSystem>, String> {
        let mut system = self.diag_xtw_x(weights)?;
        if let Some(pen) = penalty {
            if pen.nrows() != system.nrows() || pen.ncols() != system.ncols() {
                return Err(format!(
                    "factorize_system penalty shape mismatch: got {}x{}, expected {}x{}",
                    pen.nrows(),
                    pen.ncols(),
                    system.nrows(),
                    system.ncols()
                ));
            }
            system += pen;
        }
        let factor = crate::utils::StableSolver::new()
            .factorize(&system)
            .map_err(|e| format!("factorize_system failed: {e:?}"))?;
        Ok(Box::new(factor))
    }
    fn solve_system(
        &self,
        weights: &Array1<f64>,
        rhs: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
    ) -> Result<Array1<f64>, String> {
        self.solve_systemwith_policy(weights, rhs, penalty, 0.0, RidgePolicy::solver_only())
    }
    fn solve_systemwith_policy(
        &self,
        weights: &Array1<f64>,
        rhs: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge_floor: f64,
        ridge_policy: RidgePolicy,
    ) -> Result<Array1<f64>, String> {
        if rhs.len() != self.ncols() {
            return Err(format!(
                "solve_systemwith_policy rhs dimension mismatch: rhs length {} != ncols {}",
                rhs.len(),
                self.ncols()
            ));
        }
        if !(ridge_floor.is_finite() && ridge_floor >= 0.0) {
            return Err(format!(
                "solve_systemwith_policy ridge floor must be finite and non-negative, got {ridge_floor:?}"
            ));
        }
        let ridge = ridge_floor;
        // The size policy selects exactly one algorithm. A failed matrix-free
        // solve is surfaced; silently switching algorithms or escalating ridge
        // would change both performance and the solved system.
        if self.uses_matrix_free_pcg() && self.ncols() >= MATRIX_FREE_PCG_MIN_P {
            return self.solve_system_matrix_free_pcg_try(weights, rhs, penalty, ridge);
        }
        let mut system = self.diag_xtw_x(weights)?;
        if let Some(pen) = penalty {
            if pen.nrows() != system.nrows() || pen.ncols() != system.ncols() {
                return Err(format!(
                    "solve_systemwith_policy penalty shape mismatch: got {}x{}, expected {}x{}",
                    pen.nrows(),
                    pen.ncols(),
                    system.nrows(),
                    system.ncols()
                ));
            }
            system += pen;
        }
        if ridge > 0.0 {
            for diagonal in 0..system.nrows() {
                system[[diagonal, diagonal]] += ridge;
            }
        }
        let factor = crate::utils::StableSolver::new()
            .factorize(&system)
            .map_err(|error| {
                format!(
                    "solve_systemwith_policy ({ridge_policy:?}) exact factorization failed at ridge {ridge:.3e}: {error:?}"
                )
            })?;
        let mut solution = rhs.clone();
        let mut solution_matrix = crate::faer_ndarray::array1_to_col_matmut(&mut solution);
        factor.solve_in_place(solution_matrix.as_mut());
        if solution.iter().all(|value| value.is_finite()) {
            Ok(solution)
        } else {
            Err("solve_systemwith_policy produced a non-finite solution".to_string())
        }
    }
}

impl LinearOperator for DesignMatrix {
    fn uses_matrix_free_pcg(&self) -> bool {
        match self {
            Self::Dense(matrix) => matrix.uses_matrix_free_pcg(),
            Self::Sparse(_) => false,
        }
    }

    fn nrows(&self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.nrows(),
            Self::Sparse(matrix) => matrix.nrows(),
        }
    }

    fn ncols(&self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.ncols(),
            Self::Sparse(matrix) => matrix.ncols(),
        }
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(matrix) => matrix.apply(vector),
            Self::Sparse(matrix) => {
                let mut output = Array1::<f64>::zeros(matrix.nrows());
                let (symbolic, values) = matrix.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                for col in 0..matrix.ncols() {
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    let x = vector[col];
                    for idx in start..end {
                        let row = row_idx[idx];
                        output[row] += values[idx] * x;
                    }
                }
                output
            }
        }
    }

    fn apply_weighted_normal(
        &self,
        weights: FiniteSignedWeightsView<'_>,
        vector: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge: f64,
    ) -> Array1<f64> {
        assert_eq!(
            weights.len(),
            self.nrows(),
            "DesignMatrix::apply_weighted_normal weight length mismatch"
        );
        assert_eq!(
            vector.len(),
            self.ncols(),
            "DesignMatrix::apply_weighted_normal vector length mismatch"
        );
        let weights_view = weights.view();
        match self {
            Self::Dense(matrix) => matrix.apply_weighted_normal(weights, vector, penalty, ridge),
            Self::Sparse(_) => {
                let sparse = self
                    .as_sparse()
                    .expect("DesignMatrix::Sparse must expose sparse view");
                let mut out = if let Some(csr) = sparse.to_csr_arc() {
                    let sym = csr.symbolic();
                    let row_ptr = sym.row_ptr();
                    let col_idx = sym.col_idx();
                    let vals = csr.val();
                    let mut fused = Array1::<f64>::zeros(self.ncols());
                    for i in 0..self.nrows() {
                        let wi = weights_view[i];
                        if wi == 0.0 {
                            continue;
                        }
                        let start = row_ptr[i];
                        let end = row_ptr[i + 1];
                        let mut row_dot = 0.0_f64;
                        for ptr in start..end {
                            row_dot += vals[ptr] * vector[col_idx[ptr]];
                        }
                        if row_dot == 0.0 {
                            continue;
                        }
                        let scaled = wi * row_dot;
                        for ptr in start..end {
                            fused[col_idx[ptr]] += vals[ptr] * scaled;
                        }
                    }
                    fused
                } else {
                    let xv = self.apply(vector);
                    let mut weighted_xv = xv;
                    for i in 0..weighted_xv.len() {
                        weighted_xv[i] *= weights_view[i];
                    }
                    self.apply_transpose(&weighted_xv)
                };
                if let Some(pen) = penalty {
                    out += &fast_av(pen, vector);
                }
                if ridge > 0.0 {
                    for j in 0..out.len() {
                        out[j] += ridge * vector[j];
                    }
                }
                out
            }
        }
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(matrix) => matrix.apply_transpose(vector),
            Self::Sparse(matrix) => {
                let mut output = Array1::<f64>::zeros(matrix.ncols());
                let (symbolic, values) = matrix.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                for col in 0..matrix.ncols() {
                    let mut acc = 0.0;
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    for idx in start..end {
                        let row = row_idx[idx];
                        acc += values[idx] * vector[row];
                    }
                    output[col] = acc;
                }
                output
            }
        }
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        certify_signed_weights("DesignMatrix::diag_xtw_x", weights, self.nrows())?;
        let p = self.ncols();
        match self {
            Self::Dense(x) => x.diag_xtw_x(weights),
            Self::Sparse(xs) => {
                // Two regimes for sparse-stored designs:
                //
                //   (A) Numerically dense — Matern / Duchon radial bases place
                //       a nonzero in every column for every row, so XᵀWX has
                //       O(p²) fills and the scalar row-loop is dominated by
                //       memory traffic over O(n·nnz_row²) ≈ O(n·p²) ops.  Faer
                //       hand-tuned BLAS3 runs parallel + SIMD over either the
                //       cached dense design or bounded CSC-materialized row
                //       chunks, depending on the dense materialization policy.
                //
                //   (B) Genuinely sparse — B-spline / banded bases keep
                //       nnz_row at a small constant (4–6), so the per-row
                //       O(nnz_row²) work is ~25× fewer FLOPs than the dense
                //       matmul and densification is a regression.  Run a
                //       row-parallel scalar accumulation in that regime.
                //
                // Heuristic: average nnz_per_row >= p/4 picks (A).  In practice
                // the upstream `should_use_sparse_native_pirls` already routes
                // banded sparse XᵀWX designs to a separate sparse-native PIRLS
                // path that does NOT call this function, so the (A) branch
                // covers every actual call site we have today; the (B) branch
                // is a correctness-preserving safety net for future callers.
                let n = self.nrows();
                let nnz_x = xs.as_ref().val().len();
                let avg_nnz_row = if n > 0 { nnz_x / n } else { p };
                let dense_regime = 4 * avg_nnz_row >= p;
                if dense_regime {
                    let mut xtwx = Array2::<f64>::zeros((p, p));
                    // Reserve-or-stream: the dense BLAS route runs only while
                    // its full dense footprint is admitted by the process-wide
                    // governor; a refusal picks the bounded streaming CSC path.
                    if let Ok(xd) =
                        xs.try_to_dense_governed("DesignMatrix::diag_xtw_x dense sparse route")
                    {
                        stream_weighted_crossprod_into(
                            &**xd,
                            weights,
                            &mut xtwx,
                            CrossprodStructure::Full,
                            CrossprodAccum::Replace,
                            effective_global_parallelism(),
                        );
                    } else {
                        let (symbolic, values) = xs.parts();
                        streaming_sparse_csc_xt_diag_x(
                            symbolic.col_ptr(),
                            symbolic.row_idx(),
                            values,
                            n,
                            p,
                            weights.view(),
                            &mut xtwx,
                        );
                    }
                    return Ok(xtwx);
                }
                let csr = xs
                    .to_csr_arc()
                    .ok_or_else(|| "failed to obtain CSR view in xt_diag_x".to_string())?;
                let sym = csr.symbolic();
                Ok(sparse_csr_weighted_xtwx(
                    sym.row_ptr(),
                    sym.col_idx(),
                    csr.val(),
                    n,
                    p,
                    weights.view(),
                ))
            }
        }
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        certify_signed_weights("DesignMatrix::diag_gram", weights, self.nrows())?;
        let p = self.ncols();
        match self {
            Self::Dense(x) => x.diag_gram(weights),
            Self::Sparse(xs) => {
                let csr = xs
                    .to_csr_arc()
                    .ok_or_else(|| "failed to obtain CSR view in diag_gram".to_string())?;
                let sym = csr.symbolic();
                Ok(sparse_csr_diag_gram(
                    sym.row_ptr(),
                    sym.col_idx(),
                    csr.val(),
                    self.nrows(),
                    p,
                    weights.view(),
                ))
            }
        }
    }

    fn factorize_system(
        &self,
        weights: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
    ) -> Result<Box<dyn FactorizedSystem>, String> {
        if weights.len() != self.nrows() {
            return Err(format!(
                "factorize_system dimension mismatch: weights length {} != nrows {}",
                weights.len(),
                self.nrows()
            ));
        }
        match self {
            Self::Dense(_) => self.factorize_system_dense(weights, penalty),
            Self::Sparse(matrix) => {
                let system = assemble_sparseweighted_gram_system(matrix, weights, penalty)?;
                let factor = crate::sparse_exact::factorize_sparse_spd(&system)
                    .map_err(|e| format!("factorize_system failed: {e:?}"))?;
                Ok(Box::new(factor))
            }
        }
    }
}

impl DenseDesignOperator for DesignMatrix {
    fn materialization_policy(&self) -> Option<MaterializationPolicy> {
        match self {
            Self::Dense(design) => design.materialization_policy(),
            Self::Sparse(_) => None,
        }
    }

    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
        if weights.len() != self.nrows() || y.len() != self.nrows() {
            return Err(format!(
                "compute_xtwy dimension mismatch: weights={}, y={}, nrows={}",
                weights.len(),
                y.len(),
                self.nrows()
            ));
        }
        certify_signed_weights("DesignMatrix::compute_xtwy", weights, self.nrows())?;
        match self {
            Self::Dense(x) => x.compute_xtwy(weights, y),
            Self::Sparse(xs) => {
                let csr = xs
                    .as_ref()
                    .to_row_major()
                    .map_err(|_| "failed to obtain CSR view in compute_xtwy".to_string())?;
                let sym = csr.symbolic();
                let row_ptr = sym.row_ptr();
                let col_idx = sym.col_idx();
                let vals = csr.val();
                let mut out = Array1::<f64>::zeros(xs.ncols());
                for i in 0..xs.nrows() {
                    let scaled = weights[i] * y[i];
                    if scaled == 0.0 {
                        continue;
                    }
                    for idx in row_ptr[i]..row_ptr[i + 1] {
                        out[col_idx[idx]] += vals[idx] * scaled;
                    }
                }
                Ok(out)
            }
        }
    }

    fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        if middle.nrows() != self.ncols() || middle.ncols() != self.ncols() {
            return Err(format!(
                "quadratic_form_diag dimension mismatch: matrix is {}x{}, expected {}x{}",
                middle.nrows(),
                middle.ncols(),
                self.ncols(),
                self.ncols()
            ));
        }

        match self {
            Self::Dense(xd) => xd.quadratic_form_diag(middle),
            Self::Sparse(xs) => {
                let csr = xs
                    .to_csr_arc()
                    .ok_or_else(|| "quadratic_form_diag: failed to obtain CSR view".to_string())?;
                let sym = csr.symbolic();
                let row_ptr = sym.row_ptr();
                let col_idx = sym.col_idx();
                let vals = csr.val();
                let mut out = Array1::<f64>::zeros(self.nrows());
                for i in 0..xs.nrows() {
                    let start = row_ptr[i];
                    let end = row_ptr[i + 1];
                    let mut acc = 0.0_f64;
                    for a in start..end {
                        let j = col_idx[a];
                        let xij = vals[a];
                        for b in start..end {
                            let k = col_idx[b];
                            let xik = vals[b];
                            acc += xij * middle[[j, k]] * xik;
                        }
                    }
                    out[i] = acc.max(0.0);
                }
                Ok(out)
            }
        }
    }

    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if out.nrows() != rows.end - rows.start || out.ncols() != self.ncols() {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "DesignMatrix::row_chunk_into shape mismatch",
            });
        }
        match self {
            Self::Dense(matrix) => matrix.row_chunk_into(rows, out),
            Self::Sparse(matrix) => matrix.row_chunk_into(rows, out),
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        DesignMatrix::to_dense(self)
    }
}

impl LinearOperator for DenseRightProductView<'_> {
    fn nrows(&self) -> usize {
        self.base.nrows()
    }

    fn ncols(&self) -> usize {
        self.transformed_ncols()
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        let rhs;
        let v = match (self.second, self.first) {
            (None, None) => vector,
            (Some(s), None) => {
                rhs = fast_av(s, vector);
                &rhs
            }
            (None, Some(f)) => {
                rhs = fast_av(f, vector);
                &rhs
            }
            (Some(s), Some(f)) => {
                let tmp = fast_av(s, vector);
                rhs = fast_av(f, &tmp);
                &rhs
            }
        };
        fast_av(self.base, v)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        let mut out = fast_atv(self.base, vector);
        if let Some(factor) = self.first {
            out = fast_atv(factor, &out);
        }
        if let Some(factor) = self.second {
            out = fast_atv(factor, &out);
        }
        out
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        if weights.len() != self.nrows() {
            return Err(format!(
                "xt_diag_x dimension mismatch: weights length {} != nrows {}",
                weights.len(),
                self.nrows()
            ));
        }
        certify_signed_weights("DenseRightProductView::diag_xtw_x", weights, self.nrows())?;
        let mut gram = fast_xt_diag_x(self.base, weights);
        if let Some(factor) = self.first {
            gram = fast_ab(&fast_atb(factor, &gram), factor);
        }
        if let Some(factor) = self.second {
            gram = fast_ab(&fast_atb(factor, &gram), factor);
        }
        Ok(gram)
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        Ok(self.diag_xtw_x(weights)?.diag().to_owned())
    }
}

impl DenseRightProductView<'_> {
    pub fn compute_xtwy(
        &self,
        weights: &Array1<f64>,
        y: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        if weights.len() != self.nrows() || y.len() != self.nrows() {
            return Err(format!(
                "compute_xtwy dimension mismatch: weights={}, y={}, nrows={}",
                weights.len(),
                y.len(),
                self.nrows()
            ));
        }
        certify_signed_weights("DenseRightProductView::compute_xtwy", weights, self.nrows())?;
        let weighted_xty = dense_transpose_weighted_response(self.base, weights, y, None);
        let mut out = weighted_xty;
        if let Some(factor) = self.first {
            out = fast_atv(factor, &out);
        }
        if let Some(factor) = self.second {
            out = fast_atv(factor, &out);
        }
        Ok(out)
    }

    pub fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        let dense = self.materialize();
        DesignMatrix::Dense(DenseDesignMatrix::from(dense)).quadratic_form_diag(middle)
    }
}

impl LinearOperator for EmbeddedColumnBlock<'_> {
    fn nrows(&self) -> usize {
        self.local.nrows()
    }

    fn ncols(&self) -> usize {
        self.total_cols
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        fast_av(
            self.local,
            &vector.slice(ndarray::s![self.global_range.clone()]),
        )
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.total_cols);
        out.slice_mut(ndarray::s![self.global_range.clone()])
            .assign(&fast_atv(self.local, vector));
        out
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        if weights.len() != self.nrows() {
            return Err(format!(
                "xt_diag_x dimension mismatch: weights length {} != nrows {}",
                weights.len(),
                self.nrows()
            ));
        }
        certify_signed_weights("EmbeddedColumnBlock::diag_xtw_x", weights, self.nrows())?;
        let mut out = Array2::<f64>::zeros((self.total_cols, self.total_cols));
        let local = fast_xt_diag_x(self.local, weights);
        out.slice_mut(ndarray::s![
            self.global_range.clone(),
            self.global_range.clone()
        ])
        .assign(&local);
        Ok(out)
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        let mut out = Array1::<f64>::zeros(self.total_cols);
        let local =
            DesignMatrix::Dense(DenseDesignMatrix::from(self.local.clone())).diag_gram(weights)?;
        out.slice_mut(ndarray::s![self.global_range.clone()])
            .assign(&local);
        Ok(out)
    }
}

impl EmbeddedColumnBlock<'_> {
    pub fn compute_xtwy(
        &self,
        weights: &Array1<f64>,
        y: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        if weights.len() != self.nrows() || y.len() != self.nrows() {
            return Err(format!(
                "compute_xtwy dimension mismatch: weights={}, y={}, nrows={}",
                weights.len(),
                y.len(),
                self.nrows()
            ));
        }
        certify_signed_weights("EmbeddedColumnBlock::compute_xtwy", weights, self.nrows())?;
        let local = dense_transpose_weighted_response(self.local, weights, y, None);
        let mut out = Array1::<f64>::zeros(self.total_cols);
        out.slice_mut(ndarray::s![self.global_range.clone()])
            .assign(&local);
        Ok(out)
    }

    pub fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        let middle_local = middle
            .slice(ndarray::s![
                self.global_range.clone(),
                self.global_range.clone()
            ])
            .to_owned();
        DesignMatrix::Dense(DenseDesignMatrix::from(self.local.clone()))
            .quadratic_form_diag(&middle_local)
    }
}

impl DesignMatrix {
    fn factorize_system_dense(
        &self,
        weights: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
    ) -> Result<Box<dyn FactorizedSystem>, String> {
        let mut system = self.diag_xtw_x(weights)?;
        if let Some(pen) = penalty {
            if pen.nrows() != system.nrows() || pen.ncols() != system.ncols() {
                return Err(format!(
                    "factorize_system penalty shape mismatch: got {}x{}, expected {}x{}",
                    pen.nrows(),
                    pen.ncols(),
                    system.nrows(),
                    system.ncols()
                ));
            }
            system += pen;
        }
        let factor = crate::utils::StableSolver::new()
            .factorize(&system)
            .map_err(|e| format!("factorize_system failed: {e:?}"))?;
        Ok(Box::new(factor))
    }
}

fn assemble_sparseweighted_gram_system(
    matrix: &SparseDesignMatrix,
    weights: &Array1<f64>,
    penalty: Option<&Array2<f64>>,
) -> Result<SparseColMat<usize, f64>, String> {
    certify_signed_weights(
        "assemble_sparseweighted_gram_system",
        weights,
        matrix.nrows(),
    )?;
    let csr = matrix
        .to_csr_arc()
        .ok_or_else(|| "failed to obtain CSR view in factorize_system".to_string())?;
    let sym = csr.symbolic();
    let row_ptr = sym.row_ptr();
    let col_idx = sym.col_idx();
    let vals = csr.val();
    let p = matrix.ncols();
    let mut upper = BTreeMap::<(usize, usize), f64>::new();

    for i in 0..csr.nrows() {
        let wi = weights[i];
        if wi == 0.0 {
            continue;
        }
        let start = row_ptr[i];
        let end = row_ptr[i + 1];
        for a_ptr in start..end {
            let a = col_idx[a_ptr];
            let xa = vals[a_ptr];
            for b_ptr in a_ptr..end {
                let b = col_idx[b_ptr];
                let xb = vals[b_ptr];
                let key = if a <= b { (a, b) } else { (b, a) };
                *upper.entry(key).or_insert(0.0) += wi * xa * xb;
            }
        }
    }

    if let Some(pen) = penalty {
        if pen.nrows() != p || pen.ncols() != p {
            return Err(format!(
                "factorize_system penalty shape mismatch: got {}x{}, expected {}x{}",
                pen.nrows(),
                pen.ncols(),
                p,
                p
            ));
        }
        for i in 0..p {
            for j in i..p {
                let value = pen[[i, j]];
                if value != 0.0 {
                    *upper.entry((i, j)).or_insert(0.0) += value;
                }
            }
        }
    }

    let mut triplets = Vec::with_capacity(upper.len());
    for ((row, col), value) in upper {
        if value != 0.0 {
            triplets.push(Triplet::new(row, col, value));
        }
    }
    SparseColMat::try_new_from_triplets(p, p, &triplets)
        .map_err(|_| "failed to build sparse penalized system".to_string())
}

impl DesignMatrix {
    /// Horizontally concatenate design blocks without forcing eager densification.
    ///
    /// The returned matrix is a lazy `BlockDesignOperator` when more than one
    /// block is provided, so operator-backed inputs stay chunkable on the
    /// prediction path.
    pub fn hstack(blocks: Vec<DesignMatrix>) -> Result<Self, String> {
        if blocks.is_empty() {
            return Err("DesignMatrix::hstack requires at least one block".to_string());
        }
        if blocks.len() == 1 {
            return Ok(blocks.into_iter().next().expect("non-empty block list"));
        }
        let operator =
            BlockDesignOperator::new(blocks.into_iter().map(DesignBlock::from).collect())?;
        Ok(Self::Dense(DenseDesignMatrix::from(Arc::new(operator))))
    }

    pub fn nrows(&self) -> usize {
        <Self as LinearOperator>::nrows(self)
    }

    pub fn ncols(&self) -> usize {
        <Self as LinearOperator>::ncols(self)
    }

    /// Extract a dense row chunk without materializing the full matrix.
    ///
    /// Returns a `(rows.len(), ncols())` dense `Array2` for the requested row
    /// range. For lazy dense designs this delegates to the operator-backed
    /// implementation, which should remain O(chunk).
    pub fn try_row_chunk(
        &self,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, MatrixMaterializationError> {
        match self {
            Self::Dense(matrix) => matrix.try_row_chunk(rows),
            Self::Sparse(matrix) => {
                let csr =
                    matrix
                        .to_csr_arc()
                        .ok_or(MatrixMaterializationError::MissingRowChunk {
                            context: "DesignMatrix::try_row_chunk: failed to obtain CSR view",
                        })?;
                let sym = csr.symbolic();
                let row_ptr = sym.row_ptr();
                let col_idx = sym.col_idx();
                let vals = csr.val();
                let chunk_rows = rows.end - rows.start;
                let ncols = self.ncols();
                let mut out = Array2::<f64>::zeros((chunk_rows, ncols));
                for (local_row, row) in rows.enumerate() {
                    for ptr in row_ptr[row]..row_ptr[row + 1] {
                        out[[local_row, col_idx[ptr]]] = vals[ptr];
                    }
                }
                Ok(out)
            }
        }
    }

    /// Borrow-only row-chunk accessor: writes the requested rows into an
    /// existing `(rows.len(), ncols())` buffer instead of allocating a fresh
    /// `Array2<f64>` like [`Self::try_row_chunk`]. Used by hot per-row loops
    /// (e.g. latent-survival evaluate) that want to reuse a single 1-row
    /// scratch buffer across iterations.
    pub fn row_chunk_into(
        &self,
        rows: Range<usize>,
        out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        <Self as DenseDesignOperator>::row_chunk_into(self, rows, out)
    }

    /// Fully materialize this design under the process-wide byte governor.
    ///
    /// The returned owner retains the RAII reservation for precisely the
    /// matrix lifetime. A refusal is typed, happens before allocation, and is
    /// the caller's signal to remain row-chunked or matrix-free.
    pub fn try_to_dense_governed(
        &self,
        context: &'static str,
    ) -> Result<Governed<Array2<f64>>, MatrixMaterializationError> {
        self.try_to_dense_governed_with_policy(
            &ResourcePolicy::default_library().material_policy(),
            context,
        )
    }

    /// Policy-aware form of [`Self::try_to_dense_governed`]. Structural
    /// operator-only policies refuse before consulting or charging the ledger.
    pub fn try_to_dense_governed_with_policy(
        &self,
        policy: &MaterializationPolicy,
        context: &'static str,
    ) -> Result<Governed<Array2<f64>>, MatrixMaterializationError> {
        governed_dense_operator_to_dense_by_chunks(self, policy, context)
    }

    pub fn try_to_dense_by_chunks(&self, context: &str) -> Result<Array2<f64>, String> {
        let n = self.nrows();
        let p = self.ncols();
        let chunk_rows = dense_materialization_chunk_rows(n, p);
        let mut out = Array2::<f64>::zeros((n, p));
        for start in (0..n).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(n);
            let slice = out.slice_mut(s![start..end, ..]);
            self.row_chunk_into(start..end, slice)
                .map_err(|err| format!("{context}: failed to materialize row chunk: {err}"))?;
        }
        Ok(out)
    }

    /// Like [`Self::try_to_dense_by_chunks`] but refuses to allocate when the
    /// dense footprint would exceed `max_bytes`. Returned `Err` is the same
    /// shape as a densification-refused error from the resource policy, so
    /// observability-only callers can convert it into a `warn!` and skip
    /// without ever touching the allocator at huge `n`.
    pub fn try_to_dense_by_chunks_budgeted(
        &self,
        context: &str,
        max_bytes: usize,
    ) -> Result<Array2<f64>, String> {
        let n = self.nrows();
        let p = self.ncols();
        let dense_bytes = checked_dense_nbytes(n, p, context)?;
        if dense_bytes > max_bytes {
            let gib = dense_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let cap_gib = max_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            return Err(MatrixError::DensificationRefused {
                reason: format!(
                    "{context}: refusing to densify {n}x{p} (~{gib:.2} GiB, cap ~{cap_gib:.2} GiB)"
                ),
            }
            .into());
        }
        self.try_to_dense_by_chunks(context)
    }

    /// Dot a single design row against a coefficient vector without allocating
    /// a standalone row buffer when the underlying storage permits.
    pub fn dot_row(&self, row: usize, beta: &Array1<f64>) -> f64 {
        self.dot_row_view(row, beta.view())
    }

    pub fn dot_row_view(&self, row: usize, beta: ArrayView1<'_, f64>) -> f64 {
        assert_eq!(
            beta.len(),
            self.ncols(),
            "DesignMatrix::dot_row_view length mismatch: beta={}, ncols={}",
            beta.len(),
            self.ncols()
        );
        match self {
            Self::Dense(matrix) => {
                if let Some(dense) = matrix.as_dense_ref() {
                    dense.row(row).dot(&beta)
                } else {
                    matrix
                        .try_row_chunk(row..row + 1)
                        .expect("DesignMatrix::dot_row_view: try_row_chunk must succeed")
                        .row(0)
                        .dot(&beta)
                }
            }
            Self::Sparse(matrix) => {
                // SAFETY: `to_csr_arc` only returns `None` if the underlying
                // SparseColMat fails `to_row_major`, which is infallible for
                // any well-formed sparse matrix the surrounding type system
                // permits (csc → csr conversion requires only valid column
                // pointers). Reaching `None` would mean the SparseDesignMatrix
                // invariant was violated upstream.
                // SAFETY: SparseDesignMatrix invariants guarantee csc→csr conversion succeeds.
                let csr = matrix
                    .to_csr_arc()
                    .expect("DesignMatrix::dot_row: failed to obtain CSR view");
                let sym = csr.symbolic();
                let row_ptr = sym.row_ptr();
                let col_idx = sym.col_idx();
                let vals = csr.val();
                let mut out = 0.0;
                for ptr in row_ptr[row]..row_ptr[row + 1] {
                    out += vals[ptr] * beta[col_idx[ptr]];
                }
                out
            }
        }
    }

    /// Add `alpha * X[row, :]` into `out` without allocating a row buffer.
    pub fn axpy_row_into(
        &self,
        row: usize,
        alpha: f64,
        out: &mut ArrayViewMut1<'_, f64>,
    ) -> Result<(), String> {
        self.axpy_row_into_impl(row, alpha, out, false, "axpy_row_into")
    }

    /// Add `alpha * X[row, :]^2` elementwise into `out` without allocating a
    /// standalone row buffer.
    pub fn squared_axpy_row_into(
        &self,
        row: usize,
        alpha: f64,
        out: &mut ArrayViewMut1<'_, f64>,
    ) -> Result<(), String> {
        self.axpy_row_into_impl(row, alpha, out, true, "squared_axpy_row_into")
    }

    /// Shared kernel for [`axpy_row_into`](Self::axpy_row_into) and
    /// [`squared_axpy_row_into`](Self::squared_axpy_row_into): adds
    /// `alpha * X[row, :]` (when `square` is `false`) or
    /// `alpha * X[row, :]^2` elementwise (when `square` is `true`) into `out`
    /// without allocating a row buffer. `method` names the public entry point
    /// in error messages.
    #[inline]
    fn axpy_row_into_impl(
        &self,
        row: usize,
        alpha: f64,
        out: &mut ArrayViewMut1<'_, f64>,
        square: bool,
        method: &str,
    ) -> Result<(), String> {
        if out.len() != self.ncols() {
            return Err(format!(
                "DesignMatrix::{method} length mismatch: out={}, ncols={}",
                out.len(),
                self.ncols()
            ));
        }
        if alpha == 0.0 {
            return Ok(());
        }
        // Per-element scaling: `alpha * v` (axpy) or `alpha * v^2` (squared).
        let scale = |value: f64| {
            if square {
                alpha * value * value
            } else {
                alpha * value
            }
        };
        match self {
            Self::Dense(matrix) => {
                if let Some(dense) = matrix.as_dense_ref() {
                    for (dst, &value) in out.iter_mut().zip(dense.row(row).iter()) {
                        *dst += scale(value);
                    }
                } else {
                    let chunk = matrix
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("DesignMatrix::{method}: {e}"))?;
                    for (dst, &value) in out.iter_mut().zip(chunk.row(0).iter()) {
                        *dst += scale(value);
                    }
                }
            }
            Self::Sparse(matrix) => {
                // SAFETY: `to_csr_arc` returns `None` only if csc→csr conversion
                // fails, which is infallible for the well-formed sparse matrices
                // that `SparseDesignMatrix` is contractually allowed to hold.
                // SAFETY: SparseDesignMatrix invariants guarantee csc→csr conversion succeeds.
                let csr = matrix
                    .to_csr_arc()
                    .ok_or_else(|| format!("DesignMatrix::{method}: failed to obtain CSR view"))?;
                let sym = csr.symbolic();
                let row_ptr = sym.row_ptr();
                let col_idx = sym.col_idx();
                let vals = csr.val();
                for ptr in row_ptr[row]..row_ptr[row + 1] {
                    out[col_idx[ptr]] += scale(vals[ptr]);
                }
            }
        }
        Ok(())
    }

    /// Add `alpha * self[row, :] * other[row, :]` elementwise into `out`.
    ///
    /// Both matrices must have the same number of columns (== `out.len()`).
    /// For Sparse×Sparse this runs in O(nnz_lhs + nnz_rhs) via sorted
    /// merge-intersection on the CSR column indices — no dense expansion.
    pub fn crossdiag_axpy_row_into(
        &self,
        row: usize,
        other: &DesignMatrix,
        alpha: f64,
        out: &mut ArrayViewMut1<'_, f64>,
    ) -> Result<(), String> {
        assert_eq!(self.ncols(), other.ncols());
        assert_eq!(out.len(), self.ncols());
        if alpha == 0.0 {
            return Ok(());
        }
        match (self, other) {
            (Self::Dense(lhs), Self::Dense(rhs)) => {
                let lhs_chunk;
                let rhs_chunk;
                let x = if let Some(lhs_dense) = lhs.as_dense_ref() {
                    lhs_dense.row(row)
                } else {
                    lhs_chunk = lhs
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("crossdiag_axpy_row_into lhs: {e}"))?;
                    lhs_chunk.row(0)
                };
                let y = if let Some(rhs_dense) = rhs.as_dense_ref() {
                    rhs_dense.row(row)
                } else {
                    rhs_chunk = rhs
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("crossdiag_axpy_row_into rhs: {e}"))?;
                    rhs_chunk.row(0)
                };
                for (dst, (&xi, &yi)) in out.iter_mut().zip(x.iter().zip(y.iter())) {
                    *dst += alpha * xi * yi;
                }
            }
            (Self::Sparse(lhs), Self::Sparse(rhs)) => {
                // `to_csr_arc` returns `None` only if csc→csr conversion fails;
                // `SparseDesignMatrix`'s validation invariants make that
                // structurally impossible, but the function returns `Result`
                // so propagate rather than panic if a future invariant break
                // surfaces here.
                let lhs_csr = lhs.to_csr_arc().ok_or_else(|| {
                    "crossdiag_axpy_row_into: failed to obtain lhs CSR view".to_string()
                })?;
                let rhs_csr = rhs.to_csr_arc().ok_or_else(|| {
                    "crossdiag_axpy_row_into: failed to obtain rhs CSR view".to_string()
                })?;
                let lhs_sym = lhs_csr.symbolic();
                let rhs_sym = rhs_csr.symbolic();
                let lhs_rp = lhs_sym.row_ptr();
                let rhs_rp = rhs_sym.row_ptr();
                let lhs_ci = lhs_sym.col_idx();
                let rhs_ci = rhs_sym.col_idx();
                let lhs_v = lhs_csr.val();
                let rhs_v = rhs_csr.val();
                // Merge-intersection: both col_idx slices are sorted.
                let mut li = lhs_rp[row];
                let mut ri = rhs_rp[row];
                let l_end = lhs_rp[row + 1];
                let r_end = rhs_rp[row + 1];
                while li < l_end && ri < r_end {
                    let lc = lhs_ci[li];
                    let rc = rhs_ci[ri];
                    if lc == rc {
                        out[lc] += alpha * lhs_v[li] * rhs_v[ri];
                        li += 1;
                        ri += 1;
                    } else if lc < rc {
                        li += 1;
                    } else {
                        ri += 1;
                    }
                }
            }
            _ => {
                // Mixed dense/sparse: iterate the sparse side, index into dense.
                let (sparse_mat, dense_mat) = match (self, other) {
                    (Self::Sparse(s), Self::Dense(d)) => (s, d),
                    (Self::Dense(d), Self::Sparse(s)) => (s, d),
                    // Outer match's first two arms already handled (Dense,Dense)
                    // and (Sparse,Sparse); only mixed pairs reach this fallback.
                    _ => {
                        return Err(
                            "crossdiag_axpy_row_into: mixed-arm dispatch reached non-mixed pair"
                                .to_string(),
                        );
                    }
                };
                // Same CSR conversion contract as the (Sparse, Sparse) arm
                // above — propagate the (structurally impossible) failure
                // through this fn's `Result` rather than panicking.
                let csr = sparse_mat.to_csr_arc().ok_or_else(|| {
                    "crossdiag_axpy_row_into: failed to obtain CSR view".to_string()
                })?;
                let sym = csr.symbolic();
                let row_ptr = sym.row_ptr();
                let col_idx = sym.col_idx();
                let vals = csr.val();
                let dense_chunk;
                let dense_row = if let Some(dense_ref) = dense_mat.as_dense_ref() {
                    dense_ref.row(row)
                } else {
                    dense_chunk = dense_mat
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("crossdiag_axpy_row_into dense chunk: {e}"))?;
                    dense_chunk.row(0)
                };
                for ptr in row_ptr[row]..row_ptr[row + 1] {
                    let c = col_idx[ptr];
                    out[c] += alpha * vals[ptr] * dense_row[c];
                }
            }
        }
        Ok(())
    }

    /// Symmetric rank-1 update `target += alpha * x_row x_row^T` for one row.
    pub fn syr_row_into(
        &self,
        row: usize,
        alpha: f64,
        target: &mut Array2<f64>,
    ) -> Result<(), String> {
        self.syr_row_into_view(row, alpha, target.view_mut())
    }

    /// Like `syr_row_into` but accepts a mutable view, so callers can pass
    /// a slice of a larger matrix without allocating a temporary.
    pub fn syr_row_into_view(
        &self,
        row: usize,
        alpha: f64,
        mut target: ArrayViewMut2<'_, f64>,
    ) -> Result<(), String> {
        if target.nrows() != self.ncols() || target.ncols() != self.ncols() {
            return Err(format!(
                "DesignMatrix::syr_row_into shape mismatch: target={}x{}, ncols={}",
                target.nrows(),
                target.ncols(),
                self.ncols()
            ));
        }
        if alpha == 0.0 {
            return Ok(());
        }
        match self {
            Self::Dense(matrix) => {
                if let Some(dense) = matrix.as_dense_ref() {
                    let x = dense.row(row);
                    for i in 0..x.len() {
                        let xi = x[i];
                        if xi == 0.0 {
                            continue;
                        }
                        for j in 0..x.len() {
                            target[[i, j]] += alpha * xi * x[j];
                        }
                    }
                } else {
                    let chunk = matrix
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("DesignMatrix::syr_row_into: {e}"))?;
                    let x = chunk.row(0);
                    for i in 0..x.len() {
                        let xi = x[i];
                        if xi == 0.0 {
                            continue;
                        }
                        for j in 0..x.len() {
                            target[[i, j]] += alpha * xi * x[j];
                        }
                    }
                }
            }
            Self::Sparse(matrix) => {
                // SAFETY: `to_csr_arc` returns `None` only on csc→csr conversion
                // failure for a malformed sparse matrix; `SparseDesignMatrix`
                // invariants forbid that case.
                // SAFETY: SparseDesignMatrix invariants guarantee csc→csr conversion succeeds.
                let csr = matrix.to_csr_arc().ok_or_else(|| {
                    "DesignMatrix::syr_row_into: failed to obtain CSR view".to_string()
                })?;
                let sym = csr.symbolic();
                let row_ptr = sym.row_ptr();
                let col_idx = sym.col_idx();
                let vals = csr.val();
                for ptr_i in row_ptr[row]..row_ptr[row + 1] {
                    let i = col_idx[ptr_i];
                    let xi = vals[ptr_i];
                    for ptr_j in row_ptr[row]..row_ptr[row + 1] {
                        let j = col_idx[ptr_j];
                        target[[i, j]] += alpha * xi * vals[ptr_j];
                    }
                }
            }
        }
        Ok(())
    }

    /// Asymmetric rank-1 update: `target += alpha * lhs_row * rhs_row^T`.
    ///
    /// `self` provides `lhs_row`, `other` provides `rhs_row`.
    /// `target` must be `self.ncols() x other.ncols()`.
    pub fn row_outer_into(
        &self,
        row: usize,
        other: &DesignMatrix,
        alpha: f64,
        target: &mut Array2<f64>,
    ) -> Result<(), String> {
        self.row_outer_into_view(row, other, alpha, target.view_mut())
    }

    /// Like `row_outer_into` but accepts a mutable view, so callers can pass
    /// a slice of a larger matrix without allocating a temporary.
    pub fn row_outer_into_view(
        &self,
        row: usize,
        other: &DesignMatrix,
        alpha: f64,
        mut target: ArrayViewMut2<'_, f64>,
    ) -> Result<(), String> {
        if target.nrows() != self.ncols() || target.ncols() != other.ncols() {
            return Err(format!(
                "DesignMatrix::row_outer_into shape mismatch: target={}x{}, lhs={}, rhs={}",
                target.nrows(),
                target.ncols(),
                self.ncols(),
                other.ncols()
            ));
        }
        if alpha == 0.0 {
            return Ok(());
        }
        match (self, other) {
            (Self::Dense(lhs), Self::Dense(rhs)) => {
                let lhs_chunk;
                let rhs_chunk;
                let x = if let Some(lhs_dense) = lhs.as_dense_ref() {
                    lhs_dense.row(row)
                } else {
                    lhs_chunk = lhs
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("row_outer_into_view lhs: {e}"))?;
                    lhs_chunk.row(0)
                };
                let y = if let Some(rhs_dense) = rhs.as_dense_ref() {
                    rhs_dense.row(row)
                } else {
                    rhs_chunk = rhs
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("row_outer_into_view rhs: {e}"))?;
                    rhs_chunk.row(0)
                };
                for i in 0..x.len() {
                    let xi = x[i];
                    if xi == 0.0 {
                        continue;
                    }
                    for j in 0..y.len() {
                        target[[i, j]] += alpha * xi * y[j];
                    }
                }
            }
            (Self::Sparse(lhs), Self::Sparse(rhs)) => {
                // SAFETY: both `to_csr_arc` calls only fail on csc→csr conversion
                // of a malformed sparse matrix; `SparseDesignMatrix` invariants
                // upstream guarantee both inputs round-trip to CSR.
                // SAFETY: SparseDesignMatrix invariants guarantee csc→csr conversion succeeds.
                let lhs_csr = lhs
                    .to_csr_arc()
                    .ok_or_else(|| "row_outer_into: failed to obtain lhs CSR view".to_string())?;
                // SAFETY: SparseDesignMatrix invariants guarantee csc→csr conversion succeeds.
                let rhs_csr = rhs
                    .to_csr_arc()
                    .ok_or_else(|| "row_outer_into: failed to obtain rhs CSR view".to_string())?;
                let lhs_sym = lhs_csr.symbolic();
                let rhs_sym = rhs_csr.symbolic();
                let lhs_rp = lhs_sym.row_ptr();
                let rhs_rp = rhs_sym.row_ptr();
                let lhs_ci = lhs_sym.col_idx();
                let rhs_ci = rhs_sym.col_idx();
                let lhs_v = lhs_csr.val();
                let rhs_v = rhs_csr.val();
                for pi in lhs_rp[row]..lhs_rp[row + 1] {
                    let i = lhs_ci[pi];
                    let xi = lhs_v[pi];
                    for pj in rhs_rp[row]..rhs_rp[row + 1] {
                        let j = rhs_ci[pj];
                        target[[i, j]] += alpha * xi * rhs_v[pj];
                    }
                }
            }
            _ => {
                // Mixed dense/sparse: materialize both rows.
                let x = self
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("row_outer_into_view lhs: {e}"))?;
                let x_row = x.row(0);
                let y = other
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("row_outer_into_view rhs: {e}"))?;
                let y_row = y.row(0);
                for i in 0..x_row.len() {
                    let xi = x_row[i];
                    if xi == 0.0 {
                        continue;
                    }
                    for j in 0..y_row.len() {
                        target[[i, j]] += alpha * xi * y_row[j];
                    }
                }
            }
        }
        Ok(())
    }

    /// Apply the design to a borrowed vector into caller-owned storage.
    ///
    /// Unlike [`Self::matrixvectormultiply`], this accepts an `ArrayView1` and
    /// does not require either operand to be copied for materialized dense or
    /// sparse designs.
    pub fn apply_view_into(&self, vector: ArrayView1<'_, f64>, mut output: ArrayViewMut1<'_, f64>) {
        assert_eq!(self.ncols(), vector.len());
        assert_eq!(self.nrows(), output.len());
        match self {
            Self::Dense(DenseDesignMatrix::Materialized(matrix)) => {
                crate::dense::matvec_into(matrix.as_ref(), vector, output);
            }
            Self::Dense(DenseDesignMatrix::Lazy(operator)) => {
                output.assign(&operator.apply(&vector.to_owned()));
            }
            Self::Sparse(matrix) => {
                output.fill(0.0);
                let (symbolic, values) = matrix.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                for col in 0..matrix.ncols() {
                    let x = vector[col];
                    if x == 0.0 {
                        continue;
                    }
                    for idx in col_ptr[col]..col_ptr[col + 1] {
                        output[row_idx[idx]] += values[idx] * x;
                    }
                }
            }
        }
    }

    /// Apply the design to a borrowed vector and return the owned result.
    pub fn apply_view(&self, vector: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut output = Array1::<f64>::zeros(self.nrows());
        self.apply_view_into(vector, output.view_mut());
        output
    }

    /// Apply the transposed design to a borrowed vector into caller storage.
    pub fn transpose_apply_view_into(
        &self,
        vector: ArrayView1<'_, f64>,
        mut output: ArrayViewMut1<'_, f64>,
    ) {
        assert_eq!(self.nrows(), vector.len());
        assert_eq!(self.ncols(), output.len());
        match self {
            Self::Dense(DenseDesignMatrix::Materialized(matrix)) => {
                crate::dense::transpose_matvec_into(matrix.as_ref(), vector, output);
            }
            Self::Dense(DenseDesignMatrix::Lazy(operator)) => {
                output.assign(&operator.apply_transpose(&vector.to_owned()));
            }
            Self::Sparse(matrix) => {
                let (symbolic, values) = matrix.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                for col in 0..matrix.ncols() {
                    let mut value = 0.0;
                    for idx in col_ptr[col]..col_ptr[col + 1] {
                        value += values[idx] * vector[row_idx[idx]];
                    }
                    output[col] = value;
                }
            }
        }
    }

    /// Extract one column into caller-owned storage without densification.
    pub fn column_into(&self, col: usize, mut output: ArrayViewMut1<'_, f64>) {
        assert!(col < self.ncols());
        assert_eq!(self.nrows(), output.len());
        match self {
            Self::Dense(DenseDesignMatrix::Materialized(matrix)) => {
                output.assign(&matrix.column(col));
            }
            Self::Dense(DenseDesignMatrix::Lazy(operator)) => {
                let mut basis = Array1::<f64>::zeros(operator.ncols());
                basis[col] = 1.0;
                output.assign(&operator.apply(&basis));
            }
            Self::Sparse(matrix) => {
                output.fill(0.0);
                let (symbolic, values) = matrix.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                for idx in col_ptr[col]..col_ptr[col + 1] {
                    output[row_idx[idx]] += values[idx];
                }
            }
        }
    }

    /// Element access: returns the value at row `i`, column `j`.
    ///
    /// For materialized dense matrices this is O(1). For sparse matrices,
    /// the dense form is cached on the `SparseDesignMatrix` itself, so
    /// repeated calls amortize to O(1) after the first call populates the
    /// cache. For operator-backed (Lazy) dense matrices this call performs
    /// an O(n) single-column materialization via `extract_column`; callers
    /// sweeping many cells should call `as_dense_cow()` or `to_dense()`
    /// once and index the returned array directly — calling `get` in a
    /// per-cell loop on a Lazy operator is O(nrows · ncols) per call
    /// because the operator has no dense cache.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        match self {
            Self::Dense(matrix) => match matrix.as_dense_ref() {
                Some(dense) => dense[[i, j]],
                // Lazy operator: pull a single column via apply(e_j) so that
                // each call is O(n) instead of O(nrows · ncols); the default
                // `try_to_dense_arc` would re-materialize the full operator
                // on every call because Lazy operators have no dense cache.
                None => {
                    let mut e_j = Array1::<f64>::zeros(matrix.ncols());
                    e_j[j] = 1.0;
                    matrix.apply(&e_j)[i]
                }
            },
            Self::Sparse(sp) => {
                // SAFETY: `DesignMatrix::get` is documented as an
                // infallible scalar accessor; callers that take this path
                // have already accepted dense materialization. A
                // densification failure here means the sparse matrix exceeds
                // the conservative byte budget, which `DesignMatrix::get`
                // contractually forbids.
                // SAFETY: `get` is an infallible scalar accessor; caller has accepted dense materialization budget.
                let dense = sp
                    .try_to_dense_arc("DesignMatrix::get")
                    .unwrap_or_else(|msg| std::panic::panic_any(msg));
                dense[[i, j]]
            }
        }
    }

    /// Extract a single column as a dense vector without full densification.
    ///
    /// - `Dense`: O(n) column copy.
    /// - `Sparse` (CSC): O(nnz_j) using the column pointer structure.
    /// - lazy `Dense`: O(matvec) via unit-vector application.
    pub fn extract_column(&self, j: usize) -> Array1<f64> {
        let mut column = Array1::zeros(self.nrows());
        self.column_into(j, column.view_mut());
        column
    }

    /// Batched column extraction: returns an `nrows × cols.len()` dense block
    /// whose k-th column equals `extract_column(cols[k])`.
    ///
    /// For lazy operator-backed designs this routes through the operator's
    /// `apply_columns`, which `ReparamOperator` implements as a single GEMM
    /// (`X · Qs[:, cols]`) instead of one matvec dispatch per column.
    pub fn extract_columns(&self, cols: &[usize]) -> Array2<f64> {
        match self {
            Self::Dense(m) => match m {
                DenseDesignMatrix::Materialized(mat) => mat.select(Axis(1), cols),
                DenseDesignMatrix::Lazy(op) => op.apply_columns(cols),
            },
            Self::Sparse(sp) => {
                let n = sp.nrows();
                let mut out = Array2::<f64>::zeros((n, cols.len()));
                let (symbolic, values) = sp.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                for (k, &j) in cols.iter().enumerate() {
                    let start = col_ptr[j];
                    let end = col_ptr[j + 1];
                    let mut out_col = out.column_mut(k);
                    for idx in start..end {
                        out_col[row_idx[idx]] += values[idx];
                    }
                }
                out
            }
        }
    }

    /// Returns a reference to the inner dense array if this is a `Dense` variant.
    pub fn as_dense_ref(&self) -> Option<&Array2<f64>> {
        match self {
            Self::Dense(matrix) => matrix.as_dense_ref(),
            Self::Sparse(_) => None,
        }
    }

    pub const fn is_materialized_dense(&self) -> bool {
        matches!(self, Self::Dense(DenseDesignMatrix::Materialized(_)))
    }

    pub const fn is_operator_backed(&self) -> bool {
        match self {
            Self::Dense(matrix) => matrix.is_operator_backed(),
            Self::Sparse(_) => false,
        }
    }

    /// Whether this design is backed by a sparse (CSR/COO) representation
    /// rather than a dense or dense-operator backing. Used to gate the
    /// row-chunked `Xᵀ diag(w) X` BLAS-3 Gram path, which is structurally
    /// applicable only to dense / dense-operator designs (a sparse block must
    /// keep the generic sparse-aware per-row pullback).
    pub const fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse(_))
    }

    /// Zero-copy borrow when `Dense`, materialized conversion when `Sparse`.
    ///
    /// This avoids the unconditional clone that `to_dense()` performs on dense
    /// matrices.  Callers that only need a `&Array2<f64>` should use this and
    /// then call `Cow::as_ref()` or `&*cow`.
    pub fn as_dense_cow(&self) -> Cow<'_, Array2<f64>> {
        match self {
            Self::Dense(DenseDesignMatrix::Materialized(matrix)) => Cow::Borrowed(matrix.as_ref()),
            Self::Dense(DenseDesignMatrix::Lazy(op)) => match op.as_dense_ref() {
                Some(dense) => Cow::Borrowed(dense),
                // SAFETY: `as_dense_cow` is the zero-copy view accessor; its
                // contract forbids operator-backed designs that cannot expose
                // a pre-materialized dense view. A caller that reached this
                // arm used the borrow API on an operator representation it
                // should have streamed through row chunks instead.
                // SAFETY: as_dense_cow's zero-copy contract forbids operator-backed designs without a materialized view.
                None => std::panic::panic_any(format!(
                    "DesignMatrix::as_dense_cow called on operator-backed design ({}x{}); use row chunks or matrix-vector products",
                    op.nrows(),
                    op.ncols()
                )),
            },
            Self::Sparse(matrix) => Cow::Owned(
                matrix
                    .try_to_dense_arc("DesignMatrix::as_dense_cow")
                    // SAFETY: callers of `as_dense_cow` have accepted dense
                    // materialization; densification failure here means the
                    // sparse matrix exceeds the byte-cap that this accessor
                    // contractually forbids.
                    // SAFETY: caller of as_dense_cow has accepted dense materialization budget.
                    .unwrap_or_else(|msg| std::panic::panic_any(msg))
                    .as_ref()
                    .clone(),
            ),
        }
    }

    /// Borrow when already-materialized dense, otherwise materialize via
    /// chunks (or via the sparse conversion path) and return an owned `Cow`.
    ///
    /// Use this when a code path genuinely needs a contiguous `Array2<f64>`
    /// view of an operator-backed design (e.g. legacy dense linear-algebra
    /// helpers that the operator-aware code paths have not yet replaced).
    /// Prefer `try_row_chunk` / `matrixvectormultiply` when chunked or
    /// matrix-free access suffices.
    pub fn to_dense_cow(&self) -> Cow<'_, Array2<f64>> {
        match self {
            Self::Dense(DenseDesignMatrix::Materialized(matrix)) => Cow::Borrowed(matrix.as_ref()),
            Self::Dense(DenseDesignMatrix::Lazy(lazy)) => {
                if let Some(dense) = lazy.as_dense_ref() {
                    Cow::Borrowed(dense)
                } else {
                    let policy = ResourcePolicy::default_library();
                    panic_or_error_if_large_scale_mode_and_to_dense_called_with_policy(
                        "DesignMatrix::to_dense_cow",
                        lazy.nrows(),
                        lazy.ncols(),
                        &policy,
                    )
                    .unwrap_or_else(|reason| std::panic::panic_any(reason));
                    // Materialize (or reuse) the design's governed dense memo
                    // and borrow from it: zero-copy for repeat callers, and
                    // the bytes stay charged on the joint ledger for the
                    // memo's lifetime instead of escaping as an unaccounted
                    // owned buffer per call.
                    lazy.try_governed_dense_arc("DesignMatrix::to_dense_cow")
                        // SAFETY: dense-by-contract accessor; refusal means the joint ledger cannot fit this design's dense form.
                        .unwrap_or_else(|msg| std::panic::panic_any(msg));
                    Cow::Borrowed(
                        lazy.dense_memo
                            .get()
                            .expect("memo initialized by try_governed_dense_arc just above")
                            .as_ref()
                            .as_ref(),
                    )
                }
            }
            Self::Sparse(matrix) => Cow::Owned(
                matrix
                    .try_to_dense_arc("DesignMatrix::to_dense_cow")
                    // SAFETY: callers of `to_dense_cow` have committed to a
                    // dense `Array2<f64>` consumer; densification failure
                    // would mean the sparse matrix exceeds the conservative
                    // byte cap which this accessor's contract forbids.
                    // SAFETY: caller of to_dense_cow has accepted dense materialization budget.
                    .unwrap_or_else(|msg| std::panic::panic_any(msg))
                    .as_ref()
                    .clone(),
            ),
        }
    }

    /// Returns the design as a contiguous `Array2<f64>`.
    ///
    /// Operator-backed designs consult the available-memory-derived policy
    /// before allocating. Production code that can handle refusal must prefer
    /// [`Self::try_to_dense_governed`], which additionally holds the
    /// process-wide reservation for the returned matrix's whole lifetime.
    ///
    /// Sparse designs refuse to densify past the process memory budget
    /// (an n×p dense materialization that can never fit is a caller bug —
    /// the design should have stayed sparse).
    pub fn to_dense(&self) -> Array2<f64> {
        match self {
            Self::Dense(matrix) => matrix.to_dense(),
            Self::Sparse(matrix) => matrix
                .try_to_dense_arc("DesignMatrix::to_dense")
                // SAFETY: dense-by-contract accessor; failure means the dense footprint exceeds the whole process budget.
                .unwrap_or_else(|msg| std::panic::panic_any(msg))
                .as_ref()
                .clone(),
        }
    }

    /// Arc-shared variant of [`Self::to_dense`], with the same policy guard.
    pub fn to_dense_arc(&self) -> Arc<Array2<f64>> {
        match self {
            Self::Dense(matrix) => matrix.to_dense_arc(),
            Self::Sparse(matrix) => matrix
                .try_to_dense_arc("DesignMatrix::to_dense_arc")
                // SAFETY: dense-by-contract accessor; failure means the dense footprint exceeds the whole process budget.
                .unwrap_or_else(|msg| std::panic::panic_any(msg)),
        }
    }

    pub fn try_to_dense_arc(&self, context: &str) -> Result<Arc<Array2<f64>>, String> {
        match self {
            Self::Dense(matrix) => matrix.try_to_dense_arc(context),
            Self::Sparse(matrix) => matrix.try_to_dense_arc(context),
        }
    }

    /// Policy-aware densify: callers that own the consumer's dense budget can
    /// override the conservative default cap used by [`Self::try_to_dense_arc`].
    pub fn try_to_dense_arc_with_policy(
        &self,
        context: &str,
        policy: &ResourcePolicy,
    ) -> Result<Arc<Array2<f64>>, String> {
        match self {
            Self::Dense(matrix) => matrix.try_to_dense_arc_with_policy(context, policy),
            Self::Sparse(matrix) => matrix.try_to_dense_arc(context),
        }
    }

    pub fn to_csr_cache(&self) -> Option<SparseRowMat<usize, f64>> {
        match self {
            Self::Dense(_) => None,
            Self::Sparse(matrix) => matrix.to_csr_arc().map(|arc| (*arc).clone()),
        }
    }

    pub fn as_sparse(&self) -> Option<&SparseDesignMatrix> {
        match self {
            Self::Sparse(matrix) => Some(matrix),
            Self::Dense(_) => None,
        }
    }

    pub fn as_dense(&self) -> Option<&Array2<f64>> {
        match self {
            Self::Dense(matrix) => matrix.as_dense_ref(),
            Self::Sparse(_) => None,
        }
    }

    fn apply_transpose_view(&self, vector: ArrayView1<'_, f64>) -> Array1<f64> {
        match self {
            Self::Dense(DenseDesignMatrix::Materialized(matrix)) => fast_atv(matrix, &vector),
            Self::Dense(DenseDesignMatrix::Lazy(op)) => op.apply_transpose(&vector.to_owned()),
            Self::Sparse(matrix) => {
                let mut output = Array1::<f64>::zeros(matrix.ncols());
                let (symbolic, values) = matrix.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                for col in 0..matrix.ncols() {
                    let mut acc = 0.0;
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    for idx in start..end {
                        acc += values[idx] * vector[row_idx[idx]];
                    }
                    output[col] = acc;
                }
                output
            }
        }
    }

    fn diag_gram_view(&self, weights: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        if weights.len() != self.nrows() {
            return Err(format!(
                "diag_gram dimension mismatch: weights length {} != nrows {}",
                weights.len(),
                self.nrows()
            ));
        }
        FiniteSignedWeightsView::try_new(weights)
            .map_err(|reason| format!("DesignMatrix::diag_gram_view: {reason}"))?;
        match self {
            Self::Dense(DenseDesignMatrix::Materialized(matrix)) => {
                Ok(dense_diag_gram_view(matrix, weights))
            }
            Self::Dense(DenseDesignMatrix::Lazy(op)) => op.diag_gram(&weights.to_owned()),
            Self::Sparse(xs) => {
                let p = xs.ncols();
                let csr = xs
                    .to_csr_arc()
                    .ok_or_else(|| "failed to obtain CSR view in diag_gram".to_string())?;
                let sym = csr.symbolic();
                Ok(sparse_csr_diag_gram(
                    sym.row_ptr(),
                    sym.col_idx(),
                    csr.val(),
                    xs.nrows(),
                    p,
                    weights,
                ))
            }
        }
    }

    fn compute_xtwy_view(
        &self,
        weights: ArrayView1<'_, f64>,
        y: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        if weights.len() != self.nrows() || y.len() != self.nrows() {
            return Err(format!(
                "compute_xtwy dimension mismatch: weights={}, y={}, nrows={}",
                weights.len(),
                y.len(),
                self.nrows()
            ));
        }
        FiniteSignedWeightsView::try_new(weights)
            .map_err(|reason| format!("DesignMatrix::compute_xtwy_view: {reason}"))?;
        match self {
            Self::Dense(DenseDesignMatrix::Materialized(matrix)) => {
                Ok(dense_transpose_weighted_response_view(matrix, weights, y))
            }
            Self::Dense(DenseDesignMatrix::Lazy(op)) => {
                op.compute_xtwy(&weights.to_owned(), &y.to_owned())
            }
            Self::Sparse(xs) => {
                let csr = xs
                    .as_ref()
                    .to_row_major()
                    .map_err(|_| "failed to obtain CSR view in compute_xtwy".to_string())?;
                let sym = csr.symbolic();
                let row_ptr = sym.row_ptr();
                let col_idx = sym.col_idx();
                let vals = csr.val();
                let mut out = Array1::<f64>::zeros(xs.ncols());
                for i in 0..xs.nrows() {
                    let scaled = weights[i] * y[i];
                    if scaled == 0.0 {
                        continue;
                    }
                    for idx in row_ptr[i]..row_ptr[i + 1] {
                        out[col_idx[idx]] += vals[idx] * scaled;
                    }
                }
                Ok(out)
            }
        }
    }

    pub fn dot(&self, vector: &Array1<f64>) -> Array1<f64> {
        <Self as LinearOperator>::apply(self, vector)
    }

    pub fn matrixvectormultiply(&self, vector: &Array1<f64>) -> Array1<f64> {
        <Self as LinearOperator>::apply(self, vector)
    }

    pub fn transpose_vector_multiply(&self, vector: &Array1<f64>) -> Array1<f64> {
        <Self as LinearOperator>::apply_transpose(self, vector)
    }

    pub fn compute_xtwy(
        &self,
        weights: &Array1<f64>,
        y: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        <Self as DenseDesignOperator>::compute_xtwy(self, weights, y)
    }

    pub fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        <Self as LinearOperator>::diag_gram(self, weights)
    }

    pub fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        <Self as DenseDesignOperator>::quadratic_form_diag(self, middle)
    }

    pub fn apply_weighted_normal(
        &self,
        weights: &Array1<f64>,
        vector: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge: f64,
    ) -> Result<Array1<f64>, String> {
        let finite =
            certify_signed_weights("DesignMatrix::apply_weighted_normal", weights, self.nrows())?;
        if vector.len() != self.ncols() {
            return Err(format!(
                "DesignMatrix::apply_weighted_normal vector length mismatch: vector={}, ncols={}",
                vector.len(),
                self.ncols()
            ));
        }
        Ok(<Self as LinearOperator>::apply_weighted_normal(
            self, finite, vector, penalty, ridge,
        ))
    }

    pub fn solve_system(
        &self,
        weights: &Array1<f64>,
        rhs: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
    ) -> Result<Array1<f64>, String> {
        <Self as LinearOperator>::solve_system(self, weights, rhs, penalty)
    }

    pub fn solve_systemwith_policy(
        &self,
        weights: &Array1<f64>,
        rhs: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge_floor: f64,
        ridge_policy: RidgePolicy,
    ) -> Result<Array1<f64>, String> {
        <Self as LinearOperator>::solve_systemwith_policy(
            self,
            weights,
            rhs,
            penalty,
            ridge_floor,
            ridge_policy,
        )
    }

    pub fn solve_system_matrix_free_pcg(
        &self,
        weights: &Array1<f64>,
        rhs: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge_floor: f64,
    ) -> Result<Array1<f64>, String> {
        <Self as LinearOperator>::solve_system_matrix_free_pcg_try(
            self,
            weights,
            rhs,
            penalty,
            ridge_floor,
        )
    }

    pub fn solve_system_matrix_free_pcg_with_info(
        &self,
        weights: &Array1<f64>,
        rhs: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge_floor: f64,
    ) -> Result<(Array1<f64>, PcgSolveInfo), String> {
        <Self as LinearOperator>::solve_system_matrix_free_pcg_with_info_try(
            self,
            weights,
            rhs,
            penalty,
            ridge_floor,
        )
    }

    pub fn should_use_matrix_free_pcg(&self) -> bool {
        <Self as LinearOperator>::uses_matrix_free_pcg(self)
            && self.ncols() >= MATRIX_FREE_PCG_MIN_P
    }

    pub fn factorize_system(
        &self,
        weights: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
    ) -> Result<Box<dyn FactorizedSystem>, String> {
        <Self as LinearOperator>::factorize_system(self, weights, penalty)
    }
}

impl<'a> From<ArrayView2<'a, f64>> for DesignMatrix {
    fn from(value: ArrayView2<'a, f64>) -> Self {
        Self::Dense(DenseDesignMatrix::from(value.to_owned()))
    }
}

impl From<Array2<f64>> for DesignMatrix {
    fn from(value: Array2<f64>) -> Self {
        Self::Dense(DenseDesignMatrix::from(value))
    }
}

impl From<Arc<Array2<f64>>> for DesignMatrix {
    fn from(value: Arc<Array2<f64>>) -> Self {
        Self::Dense(DenseDesignMatrix::from(value))
    }
}

impl From<&Array2<f64>> for DesignMatrix {
    fn from(value: &Array2<f64>) -> Self {
        Self::Dense(DenseDesignMatrix::from(value.clone()))
    }
}

impl From<DenseDesignMatrix> for DesignMatrix {
    fn from(value: DenseDesignMatrix) -> Self {
        Self::Dense(value)
    }
}

impl From<SparseColMat<usize, f64>> for DesignMatrix {
    fn from(value: SparseColMat<usize, f64>) -> Self {
        Self::Sparse(SparseDesignMatrix::new(value))
    }
}

impl From<&SparseColMat<usize, f64>> for DesignMatrix {
    fn from(value: &SparseColMat<usize, f64>) -> Self {
        Self::Sparse(SparseDesignMatrix::new(value.clone()))
    }
}

impl From<&DesignMatrix> for DesignMatrix {
    fn from(value: &DesignMatrix) -> Self {
        value.clone()
    }
}

impl From<DesignMatrix> for DesignBlock {
    fn from(value: DesignMatrix) -> Self {
        match value {
            DesignMatrix::Dense(matrix) => Self::Dense(matrix),
            DesignMatrix::Sparse(matrix) => Self::Sparse(matrix),
        }
    }
}

impl From<&DesignMatrix> for DesignBlock {
    fn from(value: &DesignMatrix) -> Self {
        match value {
            DesignMatrix::Dense(matrix) => Self::Dense(matrix.clone()),
            DesignMatrix::Sparse(matrix) => Self::Sparse(matrix.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BlockDesignOperator, CoefficientTransformOperator, ConditionedDesign, DenseDesignMatrix,
        DenseDesignOperator, DesignBlock, DesignMatrix, EmbeddedColumnBlock,
        FiniteSignedWeightsView, MultiChannelOperator, PsdWeightsView, RandomEffectOperator,
        ReparamOperator, ResidualisedDesignOperator, RowwiseKroneckerOperator, SparseDesignMatrix,
        dense_operator_to_dense_by_chunks, dense_transpose_weighted_response, fast_atv, fast_av,
        streaming_sparse_csc_xt_diag_x, weighted_crossprod_dense_view, xt_diag_x_symmetric,
    };
    use crate::matrix::LinearOperator;
    use crate::test_support::no_densify_design;
    use crate::types::RidgePolicy;
    use crate::utils::{PcgSolveInfo, StableSolver};
    use faer::sparse::{SparseColMat, SymbolicSparseColMat, Triplet};
    use gam_runtime::resource::{
        MaterializationPolicy, MatrixMaterializationError, MemoryGovernor, ResourcePolicy,
    };
    use ndarray::{Array1, Array2, ArrayViewMut2, Axis, array, s};
    use std::ops::Range;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct ChunkOnlyOperator {
        n: usize,
        p: usize,
        row_chunk_calls: AtomicUsize,
        materialization_policy: Option<MaterializationPolicy>,
    }

    impl ChunkOnlyOperator {
        fn value(&self, i: usize, j: usize) -> f64 {
            ((i % 251) as f64) * 0.25 - ((j % 127) as f64) * 0.5 + ((i + j) % 7) as f64
        }
    }

    impl LinearOperator for ChunkOnlyOperator {
        fn nrows(&self) -> usize {
            self.n
        }

        fn ncols(&self) -> usize {
            self.p
        }

        fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
            let mut out = Array1::<f64>::zeros(self.n);
            for i in 0..self.n {
                let mut acc = 0.0;
                for j in 0..self.p {
                    acc += self.value(i, j) * vector[j];
                }
                out[i] = acc;
            }
            out
        }

        fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
            let mut out = Array1::<f64>::zeros(self.p);
            for i in 0..self.n {
                for j in 0..self.p {
                    out[j] += self.value(i, j) * vector[i];
                }
            }
            out
        }

        fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
            let dense = dense_operator_to_dense_by_chunks(self).map_err(|err| err.to_string())?;
            let psd = PsdWeightsView::try_new(weights.view())?;
            Ok(weighted_crossprod_dense_view(&dense, psd.view(), &dense))
        }
    }

    impl DenseDesignOperator for ChunkOnlyOperator {
        fn materialization_policy(&self) -> Option<MaterializationPolicy> {
            self.materialization_policy.clone()
        }

        fn row_chunk_into(
            &self,
            rows: Range<usize>,
            mut out: ArrayViewMut2<'_, f64>,
        ) -> Result<(), MatrixMaterializationError> {
            self.row_chunk_calls.fetch_add(1, Ordering::SeqCst);
            if out.nrows() != rows.end - rows.start || out.ncols() != self.p {
                return Err(MatrixMaterializationError::MissingRowChunk {
                    context: "ChunkOnlyOperator::row_chunk_into shape mismatch",
                });
            }
            for (local, row) in rows.enumerate() {
                for col in 0..self.p {
                    out[[local, col]] = self.value(row, col);
                }
            }
            Ok(())
        }

        fn to_dense(&self) -> Array2<f64> {
            // SAFETY: test-only mock asserting row_chunk_into is exercised; reaching to_dense indicates a routing regression.
            panic!("ChunkOnlyOperator::to_dense fallback must not be used")
        }
    }

    struct DirectFillOnlyOperator {
        values: Array2<f64>,
        row_chunk_calls: AtomicUsize,
    }

    impl LinearOperator for DirectFillOnlyOperator {
        fn nrows(&self) -> usize {
            self.values.nrows()
        }

        fn ncols(&self) -> usize {
            self.values.ncols()
        }

        fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
            self.values.dot(vector)
        }

        fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
            self.values.t().dot(vector)
        }

        fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
            let mut out = Array2::<f64>::zeros((self.ncols(), self.ncols()));
            for row in 0..self.nrows() {
                for left in 0..self.ncols() {
                    for right in 0..self.ncols() {
                        out[[left, right]] +=
                            weights[row] * self.values[[row, left]] * self.values[[row, right]];
                    }
                }
            }
            Ok(out)
        }
    }

    impl DenseDesignOperator for DirectFillOnlyOperator {
        fn row_chunk_into(
            &self,
            rows: Range<usize>,
            mut out: ArrayViewMut2<'_, f64>,
        ) -> Result<(), MatrixMaterializationError> {
            self.row_chunk_calls.fetch_add(1, Ordering::SeqCst);
            if rows.end > self.nrows()
                || out.nrows() != rows.end - rows.start
                || out.ncols() != self.ncols()
            {
                return Err(MatrixMaterializationError::MissingRowChunk {
                    context: "DirectFillOnlyOperator::row_chunk_into shape mismatch",
                });
            }
            out.assign(&self.values.slice(s![rows, ..]));
            Ok(())
        }

        fn try_row_chunk(
            &self,
            rows: Range<usize>,
        ) -> Result<Array2<f64>, MatrixMaterializationError> {
            panic!(
                "DirectFillOnlyOperator owned row chunk {}..{} is forbidden",
                rows.start, rows.end
            )
        }

        fn to_dense(&self) -> Array2<f64> {
            panic!("DirectFillOnlyOperator dense materialization is forbidden")
        }
    }

    fn exact_weighted_penalized_solve(
        design: &Array2<f64>,
        weights: &Array1<f64>,
        rhs: &Array1<f64>,
        penalty: &Array2<f64>,
        ridge: f64,
    ) -> Array1<f64> {
        let mut h = design
            .t()
            .dot(&(design * &weights.view().insert_axis(Axis(1))));
        h += penalty;
        if ridge > 0.0 {
            for i in 0..h.nrows() {
                h[[i, i]] += ridge;
            }
        }
        let factor = StableSolver::new()
            .factorize(&h)
            .expect("exact reference factorization");
        let mut solution = rhs.clone();
        let mut solution_matrix = crate::faer_ndarray::array1_to_col_matmut(&mut solution);
        factor.solve_in_place(solution_matrix.as_mut());
        assert!(solution.iter().all(|value| value.is_finite()));
        solution
    }

    #[test]
    fn fast_av_matches_ndarray_dot() {
        let x = array![[1.0, 2.0, -1.0], [0.5, -3.0, 4.0], [2.0, 0.0, 1.5]];
        let v = array![0.25, -1.0, 2.0];
        let expected = x.dot(&v);
        let got = fast_av(&x, &v);
        for i in 0..expected.len() {
            assert!((expected[i] - got[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn fast_atv_matches_ndarray_dot() {
        let x = array![[1.0, 2.0, -1.0], [0.5, -3.0, 4.0], [2.0, 0.0, 1.5]];
        let v = array![0.25, -1.0, 2.0];
        let expected = x.t().dot(&v);
        let got = fast_atv(&x, &v);
        for i in 0..expected.len() {
            assert!((expected[i] - got[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn sparse_to_dense_accumulates_duplicate_entries() {
        // Build a non-canonical CSC with duplicate row index in the same column.
        // This can happen if a caller bypasses canonical constructors.
        let symbolic = SymbolicSparseColMat::new_unsorted_checked(
            3,
            2,
            vec![0_usize, 2, 3],
            None,
            vec![1_usize, 1, 0],
        );
        let sparse = SparseColMat::new(symbolic, vec![2.0_f64, 3.5, -1.0]);
        let design = DesignMatrix::from(sparse);
        let dense = design.to_dense_arc();

        assert!((dense[[1, 0]] - 5.5).abs() < 1e-12);
        assert!((dense[[0, 1]] + 1.0).abs() < 1e-12);

        let v = array![4.0, -2.0];
        let y_sparse = design.matrixvectormultiply(&v);
        let y_dense = dense.dot(&v);
        for i in 0..y_sparse.len() {
            assert!((y_sparse[i] - y_dense[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn sparse_column_extractors_accumulate_duplicate_entries() {
        // Same non-canonical CSC as the to_dense fixture: column 0 carries two
        // entries at row 1 (2.0 + 3.5 = 5.5); column 1 a single -1.0 at row 0.
        // column_into and extract_columns must accumulate duplicates exactly
        // like to_dense/apply, not last-write-wins.
        let symbolic = SymbolicSparseColMat::new_unsorted_checked(
            3,
            2,
            vec![0_usize, 2, 3],
            None,
            vec![1_usize, 1, 0],
        );
        let sparse = SparseColMat::new(symbolic, vec![2.0_f64, 3.5, -1.0]);
        let design = DesignMatrix::from(sparse);
        let dense = design.to_dense();

        let mut col0 = Array1::<f64>::zeros(3);
        design.column_into(0, col0.view_mut());
        assert!((col0[1] - 5.5).abs() < 1e-12);
        for i in 0..3 {
            assert!((col0[i] - dense[[i, 0]]).abs() < 1e-12);
        }

        let block = design.extract_columns(&[0, 1]);
        assert!((block[[1, 0]] - 5.5).abs() < 1e-12);
        assert!((block[[0, 1]] + 1.0).abs() < 1e-12);
        for i in 0..3 {
            for (k, &j) in [0usize, 1].iter().enumerate() {
                assert!((block[[i, k]] - dense[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn huge_sparse_densification_is_rejected_before_allocation() {
        // 2^44 × 4 cells → 2^49 dense bytes (512 TiB): over any physical
        // process budget, so the refusal is machine-independent. The column
        // count stays small so the sparse symbolic metadata is tiny.
        let sparse = SparseColMat::try_new_from_triplets(1usize << 44, 4, &[])
            .expect("empty sparse matrix should build");
        let design = SparseDesignMatrix::new(sparse);
        let err = design
            .try_to_dense_arc("matrix test")
            .expect_err("huge sparse densification should be rejected");
        assert!(err.contains("refusing to densify sparse design"));
    }

    /// The governed sparse densification charges the process ledger for
    /// exactly the dense footprint while the owner is alive, and a full
    /// ledger routes the weighted-Gram strategy to the streaming CSC path
    /// instead of failing or allocating.
    #[test]
    fn sparse_densification_reserves_ledger_and_full_ledger_streams() {
        let triplets = [
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, -2.0),
            Triplet::new(1, 0, 0.5),
            Triplet::new(1, 1, 3.0),
            Triplet::new(2, 0, -1.5),
            Triplet::new(2, 1, 0.25),
        ];
        let sparse = SparseColMat::try_new_from_triplets(3, 2, &triplets).expect("sparse");
        let governor = MemoryGovernor::global();

        let design = SparseDesignMatrix::new(sparse.clone());
        let before = governor.reserved_bytes();
        let governed = design
            .try_to_dense_governed("governed sparse test")
            .expect("small governed densification succeeds");
        assert_eq!(
            governor.reserved_bytes(),
            before + 3 * 2 * std::mem::size_of::<f64>(),
            "governed densification must charge its dense footprint"
        );
        assert_eq!(governed.dim(), (3, 2));
        drop(governed);
        assert_eq!(
            governor.reserved_bytes(),
            before,
            "dropping the governed owner must release its charge"
        );

        // Exhaust the remaining budget: a fresh (uncached) design must refuse
        // the governed dense route, while the strategy consumer falls back to
        // the streaming CSC path and still produces the exact weighted Gram.
        let filler = governor
            .try_reserve(governor.remaining_bytes(), "test ledger filler")
            .expect("filling the remaining budget succeeds");
        let pressured = SparseDesignMatrix::new(sparse.clone());
        assert!(
            pressured
                .try_to_dense_governed("governed sparse test under pressure")
                .is_err(),
            "a full ledger must refuse governed densification"
        );
        let weights = array![1.0, -2.0, 0.5];
        let gram = xt_diag_x_symmetric(&DesignMatrix::from(sparse.clone()), &weights)
            .expect("streaming fallback under a full ledger");
        let dense = pressured.to_dense_arc();
        let mut expected = Array2::<f64>::zeros((2, 2));
        for row in 0..3 {
            for a in 0..2 {
                for b in 0..2 {
                    expected[[a, b]] += weights[row] * dense[[row, a]] * dense[[row, b]];
                }
            }
        }
        let got = gram.as_dense().expect("dense symmetric result");
        for a in 0..2 {
            for b in 0..2 {
                assert!(
                    (got[[a, b]] - expected[[a, b]]).abs() < 1e-12,
                    "streaming fallback Gram mismatch at ({a}, {b})"
                );
            }
        }
        drop(filler);
    }

    #[test]
    fn streaming_sparse_csc_xt_diag_x_matches_dense_signed_weights() {
        let sparse = SparseColMat::try_new_from_triplets(
            4,
            3,
            &[
                Triplet::new(0, 0, 1.0),
                Triplet::new(1, 0, 2.0),
                Triplet::new(2, 0, -1.0),
                Triplet::new(0, 1, 0.5),
                Triplet::new(1, 1, -3.0),
                Triplet::new(3, 1, 4.0),
                Triplet::new(0, 2, 2.0),
                Triplet::new(2, 2, 1.5),
                Triplet::new(3, 2, -0.25),
            ],
        )
        .expect("sparse matrix");
        let design = SparseDesignMatrix::new(sparse.clone());
        let dense = design.to_dense_arc();
        let weights = array![1.0, -2.0, 0.5, -1.5];
        let (symbolic, values) = sparse.parts();
        let mut got = Array2::<f64>::zeros((3, 3));
        streaming_sparse_csc_xt_diag_x(
            symbolic.col_ptr(),
            symbolic.row_idx(),
            values,
            4,
            3,
            weights.view(),
            &mut got,
        );

        let mut expected = Array2::<f64>::zeros((3, 3));
        for row in 0..4 {
            for a in 0..3 {
                for b in 0..3 {
                    expected[[a, b]] += weights[row] * dense[[row, a]] * dense[[row, b]];
                }
            }
        }
        let max_diff = (&got - &expected)
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-12,
            "streamed sparse weighted Gram mismatch: max_diff={max_diff}"
        );
    }

    #[test]
    fn block_design_row_chunk_into_fills_mixed_blocks_without_owned_child_chunks() {
        let eager = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let lazy = Arc::new(DirectFillOnlyOperator {
            values: array![[10.0, 11.0], [12.0, 13.0], [14.0, 15.0], [16.0, 17.0]],
            row_chunk_calls: AtomicUsize::new(0),
        });
        let sparse = SparseColMat::try_new_from_triplets(
            4,
            3,
            &[
                Triplet::new(0, 0, 20.0),
                Triplet::new(1, 1, 21.0),
                Triplet::new(2, 2, 22.0),
                Triplet::new(3, 0, 23.0),
            ],
        )
        .expect("sparse block");
        let random_effect = Arc::new(RandomEffectOperator::new(
            vec![Some(0), None, Some(1), Some(0)],
            2,
        ));
        let op = BlockDesignOperator::new(vec![
            DesignBlock::Dense(DenseDesignMatrix::from(eager)),
            DesignBlock::Dense(DenseDesignMatrix::from(Arc::clone(&lazy))),
            DesignBlock::Sparse(SparseDesignMatrix::new(sparse)),
            DesignBlock::RandomEffect(random_effect),
            DesignBlock::Intercept(4),
        ])
        .expect("mixed block design");

        let mut got = Array2::<f64>::from_elem((3, 10), f64::NAN);
        op.row_chunk_into(1..4, got.view_mut())
            .expect("mixed block direct row fill");

        assert_eq!(
            got,
            array![
                [3.0, 4.0, 12.0, 13.0, 0.0, 21.0, 0.0, 0.0, 0.0, 1.0],
                [5.0, 6.0, 14.0, 15.0, 0.0, 0.0, 22.0, 0.0, 1.0, 1.0],
                [7.0, 8.0, 16.0, 17.0, 23.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            ]
        );
        assert_eq!(lazy.row_chunk_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn multi_channel_row_chunk_into_crosses_boundary_without_owned_channel_chunks() {
        let first = Arc::new(DirectFillOnlyOperator {
            values: array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            row_chunk_calls: AtomicUsize::new(0),
        });
        let second = Arc::new(DirectFillOnlyOperator {
            values: array![[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            row_chunk_calls: AtomicUsize::new(0),
        });
        let op = MultiChannelOperator::new(vec![
            DesignMatrix::Dense(DenseDesignMatrix::from(Arc::clone(&first))),
            DesignMatrix::Dense(DenseDesignMatrix::from(Arc::clone(&second))),
        ])
        .expect("direct-fill multi-channel operator");

        let mut got = Array2::<f64>::from_elem((3, 2), f64::NAN);
        op.row_chunk_into(2..5, got.view_mut())
            .expect("cross-channel direct row fill");

        assert_eq!(got, array![[5.0, 6.0], [10.0, 20.0], [30.0, 40.0]]);
        assert_eq!(first.row_chunk_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second.row_chunk_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn multi_channel_operator_view_paths_match_stacked_dense_reference() {
        let dense_channel = array![[1.0, 2.0], [0.5, -1.0], [3.0, 0.25]];
        let sparse_dense = array![[0.0, 1.5], [2.0, 0.0], [-1.0, 0.75]];
        let sparse = SparseColMat::try_new_from_triplets(
            3,
            2,
            &[
                Triplet::new(1, 0, 2.0),
                Triplet::new(2, 0, -1.0),
                Triplet::new(0, 1, 1.5),
                Triplet::new(2, 1, 0.75),
            ],
        )
        .expect("sparse channel");
        let op = MultiChannelOperator::new(vec![
            DesignMatrix::Dense(DenseDesignMatrix::from(dense_channel.clone())),
            DesignMatrix::from(sparse),
        ])
        .expect("multi-channel operator");
        let mut stacked = Array2::<f64>::zeros((6, 2));
        stacked.slice_mut(s![0..3, ..]).assign(&dense_channel);
        stacked.slice_mut(s![3..6, ..]).assign(&sparse_dense);

        let beta = array![0.25, -0.4];
        let expected_apply = stacked.dot(&beta);
        let got_apply = op.apply(&beta);
        for i in 0..expected_apply.len() {
            assert!((expected_apply[i] - got_apply[i]).abs() < 1e-12);
        }

        let probe = array![0.5, -1.0, 0.25, 1.5, -0.75, 0.2];
        let expected_transpose = stacked.t().dot(&probe);
        let got_transpose = op.apply_transpose(&probe);
        for i in 0..expected_transpose.len() {
            assert!((expected_transpose[i] - got_transpose[i]).abs() < 1e-12);
        }

        let weights = array![1.0, -0.5, 0.75, 2.0, -0.25, 1.5];
        let weighted = stacked.clone() * weights.view().insert_axis(Axis(1));
        let expected_xtwx = stacked.t().dot(&weighted);
        let got_xtwx = op.diag_xtw_x(&weights).expect("multi-channel xtwx");
        for i in 0..expected_xtwx.nrows() {
            for j in 0..expected_xtwx.ncols() {
                assert!((expected_xtwx[[i, j]] - got_xtwx[[i, j]]).abs() < 1e-12);
            }
        }

        let expected_diag = Array1::from_iter((0..2).map(|j| expected_xtwx[[j, j]]));
        let got_diag = op.diag_gram(&weights).expect("multi-channel diag gram");
        for i in 0..expected_diag.len() {
            assert!((expected_diag[i] - got_diag[i]).abs() < 1e-12);
        }

        let y = array![1.0, 0.5, -0.25, 2.0, -1.0, 0.75];
        let expected_xtwy = stacked.t().dot(&(&weights * &y));
        let got_xtwy = op.compute_xtwy(&weights, &y).expect("multi-channel xtwy");
        for i in 0..expected_xtwy.len() {
            assert!((expected_xtwy[i] - got_xtwy[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn random_effect_weighted_operators_preserve_signed_curvature() {
        let op = RandomEffectOperator::new(vec![Some(0), Some(1), Some(0), None, Some(1)], 2);
        let weights = array![2.0, -3.0, 0.5, -7.0, 1.25];
        let expected_diag = array![2.5, -1.75];

        let gram = op.diag_xtw_x(&weights).expect("signed random-effect Gram");
        assert_eq!(gram, Array2::from_diag(&expected_diag));
        assert_eq!(op.diag_gram(&weights).unwrap(), expected_diag);

        let dense = array![[1.0, 2.0], [3.0, -1.0], [4.0, 0.5], [9.0, 9.0], [-2.0, 3.0]];
        let cross = op
            .weighted_cross_with_dense(&dense, &weights)
            .expect("signed dense/random-effect cross product");
        let re_dense = op.to_dense();
        let expected_cross = dense
            .t()
            .dot(&(&re_dense * &weights.view().insert_axis(Axis(1))));
        assert_eq!(cross, expected_cross);

        let beta = array![4.0, -2.0];
        let finite = FiniteSignedWeightsView::try_from_array(&weights).unwrap();
        let normal = op.apply_weighted_normal(finite, &beta, None, 0.0);
        assert_eq!(normal, &expected_diag * &beta);

        let y = array![1.0, 2.0, -4.0, 100.0, 0.5];
        let got_xtwy = op.compute_xtwy(&weights, &y).unwrap();
        let expected_xtwy = re_dense.t().dot(&(&weights * &y));
        assert_eq!(got_xtwy, expected_xtwy);
    }

    #[test]
    fn conditioned_design_signed_gram_and_response_match_materialized_reference() {
        let raw = array![[1.0, 5.0], [2.0, -1.0], [-3.0, 2.0], [4.0, 7.0]];
        let conditioned = ConditionedDesign::new(
            DesignMatrix::Dense(DenseDesignMatrix::from(raw)),
            vec![(1, 2.0, 3.0)],
        );
        let dense = conditioned.to_dense();
        let weights = array![2.0, -4.0, 0.5, -1.5];
        let weighted = &dense * &weights.view().insert_axis(Axis(1));
        let expected_gram = dense.t().dot(&weighted);
        let got_gram = conditioned.diag_xtw_x(&weights).unwrap();
        assert!(
            (&got_gram - &expected_gram)
                .iter()
                .all(|value| value.abs() < 1e-12)
        );
        let got_diag = conditioned.diag_gram(&weights).unwrap();
        assert!(
            (&got_diag - &expected_gram.diag())
                .iter()
                .all(|value| value.abs() < 1e-12)
        );

        let y = array![0.5, -2.0, 3.0, 1.25];
        let expected_xtwy = dense.t().dot(&(&weights * &y));
        let got_xtwy = conditioned.compute_xtwy(&weights, &y).unwrap();
        assert!(
            (&got_xtwy - &expected_xtwy)
                .iter()
                .all(|value| value.abs() < 1e-12)
        );
    }

    #[test]
    fn weighted_operator_certification_reports_smallest_nonfinite_row() {
        let channel = DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0], [2.0], [3.0]]));
        let op = MultiChannelOperator::new(vec![channel]).unwrap();
        let bad = array![1.0, f64::NAN, f64::INFINITY];

        for err in [
            op.diag_xtw_x(&bad).unwrap_err(),
            op.diag_gram(&bad).unwrap_err(),
            op.compute_xtwy(&bad, &array![1.0, 1.0, 1.0]).unwrap_err(),
        ] {
            assert!(err.contains("row 1"), "unexpected diagnostic: {err}");
        }
    }

    /// Perf (#1017): the fused scale-once + `fast_atb` Dense×Dense cross-block
    /// assembly in `BlockDesignOperator::diag_xtw_x` must equal the full stacked
    /// reference Gram `Xᵀ diag(w) X` for a multi-block dense layout with SIGNED
    /// weights (the observed-Hessian regime that exercises the sign-correct
    /// asymmetric cross kernel). Several dense blocks of differing widths so the
    /// off-diagonal slicing and the symmetric transpose fill are both covered.
    #[test]
    fn block_design_fused_dense_cross_matches_stacked_reference_xtwx() {
        let b0 = array![
            [1.0, 2.0],
            [0.5, -1.0],
            [3.0, 0.25],
            [-2.0, 1.5],
            [0.75, -0.5],
        ];
        let b1 = array![
            [-1.0, 0.5, 2.0],
            [1.5, -0.25, 0.0],
            [0.0, 1.0, -1.5],
            [2.0, 0.5, 1.0],
            [-0.5, -1.0, 0.25],
        ];
        let b2 = array![[0.5], [-1.0], [2.0], [0.25], [-0.75]];

        let mut stacked = Array2::<f64>::zeros((5, 6));
        stacked.slice_mut(s![.., 0..2]).assign(&b0);
        stacked.slice_mut(s![.., 2..5]).assign(&b1);
        stacked.slice_mut(s![.., 5..6]).assign(&b2);

        let blocks = vec![
            DesignBlock::Dense(DenseDesignMatrix::from(b0)),
            DesignBlock::Dense(DenseDesignMatrix::from(b1)),
            DesignBlock::Dense(DenseDesignMatrix::from(b2)),
        ];
        let op = BlockDesignOperator::new(blocks).expect("block design");

        // Signed weights: the cross kernel must NOT clamp to PSD here.
        let weights = array![1.5, -0.5, 2.0, -1.0, 0.75];
        let weighted = stacked.clone() * weights.view().insert_axis(Axis(1));
        let expected = stacked.t().dot(&weighted);

        let got = op.diag_xtw_x(&weights).expect("block fused xtwx");
        assert_eq!(got.dim(), (6, 6));
        let max_diff = (&got - &expected)
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-10,
            "fused block Dense×Dense Gram mismatch: max_diff={max_diff}"
        );
    }

    #[test]
    fn block_design_intercept_cross_and_diag_are_sign_honest() {
        // Regression for the Intercept arms of `DesignBlock::diag_xtw_x` /
        // `diag_gram` and `BlockDesignOperator::cross_block` silently
        // clamping signed working weights to `w.max(0.0)`, corrupting the
        // intercept row/column of the observed-Hessian Gram whenever any
        // weight is negative (the normal case for non-canonical-link
        // observed-Hessian IRLS, e.g. binomial+cloglog, Gamma+identity).
        let x = array![[2.0], [5.0], [-1.0], [3.0]];
        let mut stacked = Array2::<f64>::zeros((4, 2));
        stacked.column_mut(0).fill(1.0);
        stacked.slice_mut(s![.., 1..2]).assign(&x);

        let blocks = vec![
            DesignBlock::Intercept(4),
            DesignBlock::Dense(DenseDesignMatrix::from(x)),
        ];
        let op = BlockDesignOperator::new(blocks).expect("block design");

        let weights = array![3.0, -1.0, 2.0, -0.5];
        let weighted = stacked.clone() * weights.view().insert_axis(Axis(1));
        let expected = stacked.t().dot(&weighted);

        let got = op.diag_xtw_x(&weights).expect("block fused xtwx");
        assert_eq!(got.dim(), (2, 2));
        let max_diff = (&got - &expected)
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-10,
            "intercept-block Gram mismatch: got={got:?} expected={expected:?} max_diff={max_diff}"
        );

        // The intercept's own diagonal entry (Σw, signed) must match too.
        let intercept_block = &op.blocks[0];
        let diag = intercept_block
            .diag_xtw_x(&weights)
            .expect("intercept diag_xtw_x");
        assert!((diag[[0, 0]] - weights.sum()).abs() < 1e-12);
        let gram = intercept_block
            .diag_gram(&weights)
            .expect("intercept diag_gram");
        assert!((gram[0] - weights.sum()).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "ReparamOperator: X cols (2) must match Qs rows (3)")]
    fn reparam_operator_rejects_incompatible_transform_shape() {
        let x = array![[1.0, 2.0], [0.5, -1.0]];
        let qs = Arc::new(Array2::<f64>::zeros((3, 1)));
        ReparamOperator::new(DesignMatrix::Dense(DenseDesignMatrix::from(x)), qs);
    }

    /// Locks in the dispatch path for the BLAS-3 cross-block fast path:
    /// when a `CoefficientTransformOperator` is wrapped as
    /// `DenseDesignMatrix::Lazy`, `DenseDesignMatrix::as_dense_ref` must reach
    /// the operator's cached materialization. The dispatch goes
    /// `DenseDesignMatrix::as_dense_ref` → `DenseDesignOperator::as_dense_ref`,
    /// so the override has to live on `DenseDesignOperator`, not
    /// `LinearOperator`. A misplaced override on `LinearOperator` is a hard
    /// build break today (E0407, fixed in b516891), but if `LinearOperator`
    /// ever grew an `as_dense_ref` slot the silent failure would be
    /// `BlockDesignOperator::cross_block` falling back to the chunked scalar
    /// path with no test signal — this assertion is the missing signal.
    #[test]
    fn coefficient_transform_operator_exposes_cached_dense_to_block_dispatch() {
        let inner = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let transform = array![[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]];
        let expected = inner.dot(&transform);

        let op =
            CoefficientTransformOperator::new(DenseDesignMatrix::from(inner), transform.clone())
                .expect("coefficient transform operator");
        let dense_design = DenseDesignMatrix::from(Arc::new(op));

        // Touch the cache through any LinearOperator path (`apply_transpose`
        // short-circuits through `materialized_combined`). The OnceLock is empty
        // until something exercises it, so `as_dense_ref` would otherwise
        // return None before the first hot call.
        let probe = Array1::from_elem(3, 1.0);
        let warmed = dense_design.apply_transpose(&probe);
        assert_eq!(warmed.len(), expected.ncols());

        let dense_ref = dense_design
            .as_dense_ref()
            .expect("DenseDesignMatrix::as_dense_ref must reach the cached X·T");
        assert_eq!(dense_ref.dim(), expected.dim());
        for ((r, c), v) in expected.indexed_iter() {
            assert!((dense_ref[[r, c]] - v).abs() < 1e-12);
        }
    }

    #[test]
    fn coefficient_transform_operator_preserves_lazy_inner_storage() {
        let inner_values = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let transform = array![[0.5, -1.0], [1.0, 0.25]];
        let expected = inner_values.dot(&transform);
        let DesignMatrix::Dense(inner) = no_densify_design(inner_values) else {
            panic!("no-densify fixture must be dense-operator-backed");
        };
        let op = CoefficientTransformOperator::new(inner, transform)
            .expect("coefficient transform operator");
        let dense_design = DenseDesignMatrix::from(Arc::new(op));

        let probe = Array1::from_elem(2, 1.0);
        let got = dense_design.apply(&probe);
        let want = expected.dot(&probe);
        for (got_i, want_i) in got.iter().zip(want.iter()) {
            assert!((got_i - want_i).abs() < 1e-12);
        }
        assert!(
            dense_design.as_dense_ref().is_none(),
            "coefficient transform must not materialize an operator-backed inner design"
        );
    }

    #[test]
    fn design_matrix_hstack_preserves_lazy_blocks() {
        let left_dense = array![[1.0, 2.0], [3.0, 4.0]];
        let right_dense = array![[5.0], [6.0]];
        let left = no_densify_design(left_dense.clone());
        let right = no_densify_design(right_dense.clone());
        let stacked = DesignMatrix::hstack(vec![left, right]).expect("stacked design");

        assert!(stacked.as_dense_ref().is_none());
        assert!(!stacked.is_materialized_dense());
        assert!(stacked.is_operator_backed());
        assert_eq!(stacked.nrows(), 2);
        assert_eq!(stacked.ncols(), 3);

        let beta = array![0.25, -0.5, 2.0];
        let expected = array![9.25, 10.75];
        let got = stacked.dot(&beta);
        for i in 0..expected.len() {
            assert!((got[i] - expected[i]).abs() < 1e-12);
        }

        let chunk = stacked
            .try_row_chunk(0..2)
            .expect("stacked.try_row_chunk must succeed");
        assert_eq!(chunk, array![[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]);
    }

    #[test]
    #[should_panic(expected = "DesignMatrix::as_dense_cow called on operator-backed design")]
    fn design_matrix_as_dense_cow_rejects_operator_backed_designs() {
        let design = no_densify_design(array![[1.0, 2.0], [3.0, 4.0]]);
        design.as_dense_cow();
    }

    #[test]
    fn sparse_factorized_solve_matches_dense_operator_solve() {
        let triplets = vec![
            Triplet::new(0usize, 0usize, 1.0),
            Triplet::new(1, 0, 2.0),
            Triplet::new(1, 1, -1.0),
            Triplet::new(2, 1, 3.0),
            Triplet::new(2, 2, 0.5),
        ];
        let sparse = SparseColMat::try_new_from_triplets(3, 3, &triplets)
            .expect("sparse design should build");
        let sparse_design = DesignMatrix::from(sparse);
        let dense_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            sparse_design.to_dense(),
        ));
        let weights = array![1.5, 0.75, 2.0];
        let rhs = array![1.0, -0.5, 2.0];
        let penalty = Array2::from_diag(&array![0.25, 0.5, 0.75]);

        let sparse_sol = sparse_design
            .solve_system(&weights, &rhs, Some(&penalty))
            .expect("sparse solve should factorize natively");
        let dense_sol = dense_design
            .solve_system(&weights, &rhs, Some(&penalty))
            .expect("dense solve should factorize");

        for i in 0..rhs.len() {
            assert!(
                (sparse_sol[i] - dense_sol[i]).abs() < 1e-10,
                "solution mismatch at {i}: sparse={} dense={}",
                sparse_sol[i],
                dense_sol[i]
            );
        }
    }

    #[test]
    fn solve_system_stabilizes_indefinite_penalty_and_returns_finite_solution() {
        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
            [1.0, 0.0],
            [0.0, 0.0]
        ]));
        let weights = array![1.0, 1.0];
        let rhs = array![2.0, 0.0];
        let penalty = array![[0.0, 0.0], [0.0, -1e-12]];

        let beta = design
            .solve_system(&weights, &rhs, Some(&penalty))
            .expect("solve_system should stabilize indefinite systems");

        assert!(beta.iter().all(|v| v.is_finite()));
        assert!((beta[0] - 2.0).abs() < 1e-10);
        assert!(beta[1].abs() < 1e-8);
    }

    #[test]
    fn explicit_matrix_free_pcg_matches_exact_large_dense_weighted_penalized_solve() {
        let n = 48usize;
        let p = 520usize;
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                x[[i, j]] = (((i + 3) * (j + 5)) % 17) as f64 / 17.0
                    + 0.02 * (i as f64)
                    + 0.001 * (j as f64);
            }
        }
        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x.clone()));
        let weights = Array1::from_iter((0..n).map(|i| 0.5 + (i as f64) / (2.0 * n as f64)));
        let rhs = Array1::from_iter((0..p).map(|j| ((j % 13) as f64 - 6.0) / 13.0));
        let penalty = Array2::from_diag(&Array1::from_iter(
            (0..p).map(|j| 0.1 + 0.005 * ((j % 7) as f64)),
        ));
        let ridge = 1e-8;

        let pcg = design
            .solve_system_matrix_free_pcg(&weights, &rhs, Some(&penalty), ridge)
            .expect("matrix-free pcg solve");
        let exact = exact_weighted_penalized_solve(&x, &weights, &rhs, &penalty, ridge);
        for i in 0..p {
            assert!(
                (pcg[i] - exact[i]).abs() < 1e-5,
                "solution mismatch at {i}: pcg={} exact={}",
                pcg[i],
                exact[i]
            );
        }
        let mut h = x
            .t()
            .dot(&(x.clone() * weights.view().insert_axis(Axis(1))));
        h += &penalty;
        for i in 0..p {
            h[[i, i]] += ridge;
        }
        let residual = h.dot(&pcg) - &rhs;
        let residual_norm = residual.dot(&residual).sqrt();
        assert!(residual_norm < 1e-4, "residual_norm={residual_norm}");
    }

    #[test]
    fn policy_solve_matches_explicit_matrix_free_pcg_on_large_dense_system() {
        let n = 40usize;
        let p = 520usize;
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                x[[i, j]] = (((2 * i + j + 11) % 23) as f64 / 23.0) + 0.0005 * (j as f64);
            }
        }
        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x));
        let weights = Array1::from_iter((0..n).map(|i| 1.0 + 0.01 * i as f64));
        let rhs = Array1::from_iter((0..p).map(|j| ((j % 5) as f64) - 2.0));
        let penalty = Array2::from_diag(&Array1::from_iter(
            (0..p).map(|j| 0.2 + 0.01 * ((j % 3) as f64)),
        ));
        let ridge_floor = 1e-8;

        let explicit = design
            .solve_system_matrix_free_pcg(&weights, &rhs, Some(&penalty), ridge_floor)
            .expect("explicit pcg");
        let policy = design
            .solve_systemwith_policy(
                &weights,
                &rhs,
                Some(&penalty),
                ridge_floor,
                RidgePolicy::solver_only(),
            )
            .expect("policy solve");
        for i in 0..p {
            // This system is heavily rank-deficient (rank ≤ n = 40, p = 520,
            // p ≫ n) with only a weak ~0.2 diagonal penalty + 1e-8 ridge_floor,
            // so the normal matrix is severely ill-conditioned. Both arms are
            // matrix-free PCG (explicit vs solver-only stabilization
            // policy); they terminate at slightly different points on the
            // near-null manifold. A fixed 1e-6 absolute gate is below what PCG
            // can guarantee at this conditioning; assert a relative tolerance
            // scaled by the coefficient magnitude instead (gam#846).
            let tol = 1e-5 * (1.0 + explicit[i].abs());
            assert!(
                (explicit[i] - policy[i]).abs() < tol,
                "policy mismatch at {i}: explicit={} policy={} (tol={tol})",
                explicit[i],
                policy[i]
            );
        }
    }

    #[test]
    fn explicit_matrix_free_pcg_reports_convergence_diagnostics() {
        let n = 36usize;
        let p = 2160usize;
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                x[[i, j]] = (((3 * i + 5 * j + 7) % 29) as f64 / 29.0)
                    + 0.015 * (i as f64)
                    + 1e-4 * j as f64;
            }
        }
        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x.clone()));
        assert!(design.should_use_matrix_free_pcg());
        let weights = Array1::from_iter((0..n).map(|i| 0.75 + 0.01 * i as f64));
        let rhs = Array1::from_iter((0..p).map(|j| ((j % 9) as f64 - 4.0) / 9.0));
        let penalty = Array2::from_diag(&Array1::from_iter(
            (0..p).map(|j| 0.05 + 0.002 * ((j % 11) as f64)),
        ));
        let ridge = 1e-8;

        let (pcg, info): (Array1<f64>, PcgSolveInfo) = design
            .solve_system_matrix_free_pcg_with_info(&weights, &rhs, Some(&penalty), ridge)
            .expect("pcg with info");
        assert!(info.converged);
        assert!(info.iterations > 0);
        assert!(info.relative_residual_norm.is_finite());
        assert!(info.relative_residual_norm < 1e-6);

        let exact = exact_weighted_penalized_solve(&x, &weights, &rhs, &penalty, ridge);
        for i in 0..p {
            assert!(
                (pcg[i] - exact[i]).abs() < 1e-5,
                "solution mismatch at {i}: pcg={} exact={}",
                pcg[i],
                exact[i]
            );
        }
    }

    #[test]
    fn compute_xtwy_dense_allocationfree_matches_matvec() {
        let n = 2_000usize;
        let p = 64usize;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut y = Array1::<f64>::zeros(n);
        let mut w = Array1::<f64>::zeros(n);
        for i in 0..n {
            y[i] = ((i % 17) as f64 - 8.0) * 0.1;
            w[i] = 0.25 + ((i % 11) as f64) * 0.05;
            for j in 0..p {
                x[[i, j]] = (((i * 13 + j * 7) % 97) as f64) / 97.0;
            }
        }

        let reference = {
            let wy = Array1::from_shape_fn(n, |i| y[i] * w[i]);
            fast_atv(&x, &wy)
        };
        let fused = dense_transpose_weighted_response(&x, &w, &y, None);
        for j in 0..p {
            assert!(
                (reference[j] - fused[j]).abs() < 1e-10,
                "mismatch at column {j}: ref={} fused={}",
                reference[j],
                fused[j]
            );
        }
    }

    #[test]
    fn large_lazy_dense_materialization_streams_chunks_without_to_dense_fallback() {
        let n = 11_000usize;
        let p = 128usize;
        let op = Arc::new(ChunkOnlyOperator {
            n,
            p,
            row_chunk_calls: AtomicUsize::new(0),
            materialization_policy: None,
        });
        let design = DenseDesignMatrix::from(Arc::clone(&op));

        let dense = design.to_dense_arc();

        assert_eq!(dense.dim(), (n, p));
        assert!(
            op.row_chunk_calls.load(Ordering::SeqCst) > 1,
            "expected dense materialization to stream more than one row chunk"
        );
        for &(i, j) in &[(0, 0), (8_191, 127), (8_192, 0), (10_999, 64)] {
            assert_eq!(dense[[i, j]], op.value(i, j));
        }
        assert!(
            design.as_dense_ref().is_some(),
            "as_dense_ref must expose a populated LazyDense memo so storage certificates can observe it"
        );
    }

    #[test]
    fn construction_policy_survives_nested_coefficient_and_block_operators() {
        let strict = ResourcePolicy::analytic_operator_required().material_policy();
        let op = Arc::new(ChunkOnlyOperator {
            n: 32,
            p: 2,
            row_chunk_calls: AtomicUsize::new(0),
            materialization_policy: Some(strict),
        });
        let inner = DenseDesignMatrix::from(Arc::clone(&op));
        let transformed = CoefficientTransformOperator::new(inner, Array2::<f64>::eye(2))
            .expect("coefficient transform");
        let block = BlockDesignOperator::new(vec![DesignBlock::Dense(DenseDesignMatrix::from(
            Arc::new(transformed),
        ))])
        .expect("block design");
        let design = DenseDesignMatrix::from(Arc::new(block));

        let error = design
            .try_to_dense_arc("nested construction-policy regression")
            .expect_err("strict construction policy must survive every wrapper");
        assert!(error.contains("construction policy requires streamed storage"));
        assert_eq!(op.row_chunk_calls.load(Ordering::SeqCst), 0);
        assert!(design.as_dense_ref().is_none());
    }

    /// The governed path couples the full allocation to its byte reservation,
    /// while strict and undersized policies refuse before row work begins.
    #[test]
    fn governed_to_dense_reserves_and_policy_refusals_are_typed() {
        let op = Arc::new(ChunkOnlyOperator {
            n: 128,
            p: 4,
            row_chunk_calls: AtomicUsize::new(0),
            materialization_policy: None,
        });
        let design = DesignMatrix::Dense(DenseDesignMatrix::from(Arc::clone(&op)));

        let dense = design
            .try_to_dense_governed("governed dense regression")
            .expect("small governed materialization");
        assert_eq!(dense.dim(), (128, 4));
        assert_eq!(dense.reserved_bytes(), 128 * 4 * std::mem::size_of::<f64>());

        let strict = ResourcePolicy::analytic_operator_required().material_policy();
        let err = design
            .try_to_dense_governed_with_policy(&strict, "regression strict refuses")
            .expect_err("strict policy must refuse lazy materialization");
        assert!(matches!(err, MatrixMaterializationError::Forbidden { .. }));

        let mut tight = ResourcePolicy::default_library().material_policy();
        tight.max_single_dense_bytes = 1;
        let size_err = design
            .try_to_dense_governed_with_policy(&tight, "regression tight refuses")
            .expect_err("undersized cap must refuse lazy materialization");
        assert!(matches!(
            size_err,
            MatrixMaterializationError::TooLarge { .. }
        ));
    }

    #[test]
    fn try_to_dense_by_chunks_writes_directly_into_output_slices() {
        let n = 11_000usize;
        let p = 128usize;
        let op = Arc::new(ChunkOnlyOperator {
            n,
            p,
            row_chunk_calls: AtomicUsize::new(0),
            materialization_policy: None,
        });
        let design = DesignMatrix::Dense(DenseDesignMatrix::from(Arc::clone(&op)));

        let dense = design
            .try_to_dense_by_chunks("large chunked regression")
            .expect("chunked materialization");

        assert_eq!(dense.dim(), (n, p));
        assert!(
            op.row_chunk_calls.load(Ordering::SeqCst) > 1,
            "expected direct chunked conversion to use bounded row chunks"
        );
        for &(i, j) in &[(1, 7), (4_096, 12), (8_193, 63), (10_998, 127)] {
            assert_eq!(dense[[i, j]], op.value(i, j));
        }
    }

    #[test]
    fn tensor_product_design_operator_matches_dense_2d() {
        use super::{DenseDesignOperator, TensorProductDesignOperator};

        // Two marginal B-spline-like bases: 10 rows, 4 and 3 columns.
        let n = 10;
        let q1 = 4;
        let q2 = 3;
        let mut b1 = Array2::<f64>::zeros((n, q1));
        let mut b2 = Array2::<f64>::zeros((n, q2));
        // Fill with simple hat-function-like patterns (sparse per row).
        for i in 0..n {
            let t1 = i as f64 / (n - 1) as f64 * (q1 - 1) as f64;
            let j1 = (t1.floor() as usize).min(q1 - 2);
            let frac1 = t1 - j1 as f64;
            b1[[i, j1]] = 1.0 - frac1;
            b1[[i, j1 + 1]] = frac1;

            let t2 = i as f64 / (n - 1) as f64 * (q2 - 1) as f64;
            let j2 = (t2.floor() as usize).min(q2 - 2);
            let frac2 = t2 - j2 as f64;
            b2[[i, j2]] = 1.0 - frac2;
            b2[[i, j2 + 1]] = frac2;
        }

        let op = TensorProductDesignOperator::new(vec![Arc::new(b1.clone()), Arc::new(b2.clone())])
            .unwrap();

        // Build dense reference via explicit Kronecker row products.
        let p = q1 * q2;
        let mut dense = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j1 in 0..q1 {
                for j2 in 0..q2 {
                    dense[[i, j1 * q2 + j2]] = b1[[i, j1]] * b2[[i, j2]];
                }
            }
        }

        // Test to_dense.
        let op_dense = op.to_dense();
        let max_diff = (&op_dense - &dense)
            .iter()
            .map(|v: &f64| v.abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-14, "to_dense mismatch: max_diff={max_diff}");

        // Test apply.
        let beta = Array1::from_vec((0..p).map(|j| (j as f64 + 1.0) * 0.1).collect());
        let ref_result = dense.dot(&beta);
        let op_result = op.apply(&beta);
        let max_diff = (&op_result - &ref_result)
            .iter()
            .map(|v: &f64| v.abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-12, "apply mismatch: max_diff={max_diff}");

        // Test apply_transpose.
        let v = Array1::from_vec((0..n).map(|i| (i as f64 + 1.0) * 0.3).collect());
        let ref_xt_v = dense.t().dot(&v);
        let op_xt_v = op.apply_transpose(&v);
        let max_diff = (&op_xt_v - &ref_xt_v)
            .iter()
            .map(|v: &f64| v.abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < 1e-12,
            "apply_transpose mismatch: max_diff={max_diff}"
        );

        // Test diag_xtw_x.
        let w = Array1::from_vec((0..n).map(|i| 1.0 + i as f64 * 0.1).collect());
        let ref_xtwx = {
            let mut out = Array2::<f64>::zeros((p, p));
            for i in 0..n {
                for a in 0..p {
                    for b in 0..p {
                        out[[a, b]] += w[i] * dense[[i, a]] * dense[[i, b]];
                    }
                }
            }
            out
        };
        let op_xtwx = op.diag_xtw_x(&w).unwrap();
        let max_diff = (&op_xtwx - &ref_xtwx)
            .iter()
            .map(|v: &f64| v.abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-10, "diag_xtw_x mismatch: max_diff={max_diff}");
    }

    #[test]
    fn tensor_product_design_operator_3d() {
        use super::{DenseDesignOperator, TensorProductDesignOperator};

        let n = 8;
        let dims = [3, 2, 2];
        let mut marginals: Vec<Array2<f64>> = Vec::new();
        for &q in &dims {
            let mut b = Array2::<f64>::zeros((n, q));
            for i in 0..n {
                let t = i as f64 / (n - 1) as f64 * (q - 1) as f64;
                let j = (t.floor() as usize).min(q - 2);
                let frac = t - j as f64;
                b[[i, j]] = 1.0 - frac;
                b[[i, j + 1]] = frac;
            }
            marginals.push(b);
        }

        let op = TensorProductDesignOperator::new(
            marginals.iter().map(|m| Arc::new(m.clone())).collect(),
        )
        .unwrap();

        // Dense reference.
        let p: usize = dims.iter().copied().product();
        let mut dense = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j0 in 0..dims[0] {
                for j1 in 0..dims[1] {
                    for j2 in 0..dims[2] {
                        let col = j0 * dims[1] * dims[2] + j1 * dims[2] + j2;
                        dense[[i, col]] =
                            marginals[0][[i, j0]] * marginals[1][[i, j1]] * marginals[2][[i, j2]];
                    }
                }
            }
        }

        let op_dense = op.to_dense();
        let max_diff = (&op_dense - &dense)
            .iter()
            .map(|v: &f64| v.abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < 1e-14,
            "3D to_dense mismatch: max_diff={max_diff}"
        );

        // Test round-trip: apply then apply_transpose.
        let beta = Array1::from_vec((0..p).map(|j| (j as f64).sin()).collect());
        let xb = op.apply(&beta);
        let xtxb = op.apply_transpose(&xb);
        let ref_xtxb = dense.t().dot(&dense.dot(&beta));
        let max_diff = (&xtxb - &ref_xtxb)
            .iter()
            .map(|v: &f64| v.abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-10, "3D X'Xβ mismatch: max_diff={max_diff}");
    }

    #[test]
    fn sparse_weighted_crossprod_parallel_path_matches_dense_reference() {
        use faer::sparse::Triplet;

        let n = 4096;
        let p = 192;
        let mut triplets = Vec::with_capacity(n * 4);
        let mut dense = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let base = (i * 37) % p;
            for k in 0..4 {
                let col = (base + k * 11) % p;
                let val = ((i + 3 * k + 1) as f64).sin() * 0.25 + 0.5;
                triplets.push(Triplet::new(i, col, val));
                dense[[i, col]] = val;
            }
        }
        let sparse = faer::sparse::SparseColMat::try_new_from_triplets(n, p, &triplets).unwrap();
        let design = DesignMatrix::Sparse(SparseDesignMatrix::new(sparse));
        let weights = Array1::from_iter((0..n).map(|i| match i % 7 {
            0 => 0.0,
            r => 0.5 + r as f64 * 0.125,
        }));

        let got = <DesignMatrix as LinearOperator>::xt_diag_x_signed_op(
            &design,
            FiniteSignedWeightsView::try_from_array(&weights).unwrap(),
        )
        .unwrap();
        let mut reference = Array2::<f64>::zeros((p, p));
        for i in 0..n {
            let wi = weights[i];
            if wi == 0.0 {
                continue;
            }
            for a in 0..p {
                let xa = dense[[i, a]];
                if xa == 0.0 {
                    continue;
                }
                for b in 0..p {
                    reference[[a, b]] += wi * xa * dense[[i, b]];
                }
            }
        }
        let max_diff = (&got - &reference)
            .iter()
            .map(|v: &f64| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-10,
            "sparse xtwx mismatch: max_diff={max_diff}"
        );

        let got_diag = design.diag_gram(&weights).unwrap();
        let ref_diag = reference.diag().to_owned();
        let max_diag_diff = (&got_diag - &ref_diag)
            .iter()
            .map(|v: &f64| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diag_diff < 1e-10,
            "sparse diag gram mismatch: max_diff={max_diag_diff}"
        );
    }

    #[test]
    fn rowwise_kronecker_sparse_structured_xtwx_matches_dense_reference() {
        use faer::sparse::Triplet;

        let n = 2048;
        let p_cov = 64;
        let p_time = 6;
        let mut triplets = Vec::with_capacity(n * 3);
        let mut cov_dense = Array2::<f64>::zeros((n, p_cov));
        for i in 0..n {
            let base = (i * 17) % p_cov;
            for k in 0..3 {
                let col = (base + k * 7) % p_cov;
                let val = 0.2 + (((i + k) % 13) as f64) / 17.0;
                triplets.push(Triplet::new(i, col, val));
                cov_dense[[i, col]] = val;
            }
        }
        let cov_sparse =
            faer::sparse::SparseColMat::try_new_from_triplets(n, p_cov, &triplets).unwrap();
        let cov = DesignMatrix::Sparse(SparseDesignMatrix::new(cov_sparse));
        let mut time = Array2::<f64>::zeros((n, p_time));
        for i in 0..n {
            for t in 0..p_time {
                time[[i, t]] = (((i + 1) * (t + 3)) as f64).cos() * 0.1 + 0.4;
            }
        }
        let op = RowwiseKroneckerOperator::new(cov, Arc::new(time.clone())).unwrap();
        let weights = Array1::from_iter((0..n).map(|i| 0.25 + ((i % 11) as f64) * 0.05));
        let got = op.diag_xtw_x(&weights).unwrap();

        let p_total = p_cov * p_time;
        let mut reference = Array2::<f64>::zeros((p_total, p_total));
        for i in 0..n {
            for c1 in 0..p_cov {
                let x1 = cov_dense[[i, c1]];
                if x1 == 0.0 {
                    continue;
                }
                for t1 in 0..p_time {
                    let a = c1 * p_time + t1;
                    let xa = x1 * time[[i, t1]];
                    for c2 in 0..p_cov {
                        let x2 = cov_dense[[i, c2]];
                        if x2 == 0.0 {
                            continue;
                        }
                        for t2 in 0..p_time {
                            let b = c2 * p_time + t2;
                            reference[[a, b]] += weights[i] * xa * x2 * time[[i, t2]];
                        }
                    }
                }
            }
        }
        let max_diff = (&got - &reference)
            .iter()
            .map(|v: &f64| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-9,
            "rowwise kronecker sparse xtwx mismatch: max_diff={max_diff}"
        );
    }

    #[test]
    fn embedded_column_block_zero_row_local_materializes_empty_global_width() {
        let local = Array2::<f64>::zeros((0, 0));
        let out = EmbeddedColumnBlock::new(&local, 2..5, 7).materialize();
        assert_eq!(out.dim(), (0, 7));
    }

    /// Identity case: with V_b = I and a zero r_block, the residualised
    /// operator must emit the raw inner block unchanged. Anchored by an
    /// arbitrary 3×2 anchor whose contribution is zeroed out by the all-zero
    /// r_block — verifies the subtraction path is wired but contributes
    /// nothing when the reparam happens to be identity-with-no-residual.
    #[test]
    fn residualised_design_operator_identity_passthrough() {
        let inner = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let transform = Array2::<f64>::eye(2);
        let anchor_raw = array![[7.0, -1.0], [0.5, 2.0], [-3.0, 1.5]];
        let r_block = Arc::new(Array2::<f64>::zeros((
            anchor_raw.ncols(),
            transform.ncols(),
        )));
        let anchor_design = DesignMatrix::from(anchor_raw);

        let op = ResidualisedDesignOperator::new(
            DenseDesignMatrix::from(inner.clone()),
            transform,
            vec![(anchor_design, r_block)],
        )
        .expect("residualised operator constructs");

        // Row-chunk path (cold — exercises the streaming branch before the
        // materialisation cache is warmed).
        let mut chunk = Array2::<f64>::zeros((3, 2));
        op.row_chunk_into(0..3, chunk.view_mut())
            .expect("row chunk");
        for ((r, c), v) in inner.indexed_iter() {
            assert!(
                (chunk[[r, c]] - v).abs() < 1e-12,
                "identity row_chunk mismatch at ({r},{c}): got {} expected {v}",
                chunk[[r, c]]
            );
        }

        // Through the DenseDesignMatrix wrapper — confirms the
        // generic `From<Arc<T: DenseDesignOperator>>` integration carries
        // shape and row-access semantics into the rest of the design stack.
        let dense_design = DenseDesignMatrix::from(Arc::new(op));
        assert_eq!(dense_design.nrows(), 3);
        assert_eq!(dense_design.ncols(), 2);
        let probe = ndarray::Array1::from_vec(vec![1.0, -2.0]);
        let got = dense_design.apply(&probe);
        let expected = inner.dot(&probe);
        for i in 0..3 {
            assert!((got[i] - expected[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn residualised_design_operator_preserves_lazy_inner_storage() {
        let inner_values = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let DesignMatrix::Dense(inner) = no_densify_design(inner_values.clone()) else {
            panic!("no-densify fixture must be dense-operator-backed");
        };
        let transform = Array2::<f64>::eye(2);
        let anchor = DesignMatrix::from(Array2::<f64>::zeros((3, 1)));
        let r_block = Arc::new(Array2::<f64>::zeros((1, 2)));
        let op = ResidualisedDesignOperator::new(inner, transform, vec![(anchor, r_block)])
            .expect("residualised operator constructs");
        let dense_design = DenseDesignMatrix::from(Arc::new(op));

        let probe = array![0.25, -0.5];
        let got = dense_design.apply(&probe);
        let want = inner_values.dot(&probe);
        for (got_i, want_i) in got.iter().zip(want.iter()) {
            assert!((got_i - want_i).abs() < 1e-12);
        }
        assert!(
            dense_design.as_dense_ref().is_none(),
            "residualised transform must not materialize an operator-backed inner design"
        );
    }

    /// Two-block case with a shared column: build raw A and B that overlap
    /// on one direction, hand-construct V_b and R_b so that
    /// `out_full = A·γ_A + (C_b·V_b − A·R_b)·θ_b` recovers the raw row
    /// prediction `A·γ_A + B·β_b` exactly. Anchors the contract that
    /// `R_b = M_b · V_b` projects out the anchor-overlapping direction so
    /// the emitted compiled column is orthogonal-to-the-anchor at the
    /// design-matrix level.
    #[test]
    fn residualised_design_operator_two_block_reconstruction() {
        // Anchor A (n × 2) and raw block B (n × 2); they share their first
        // column up to scale, so the "kept" direction of B is its second
        // column residualised against A's column space.
        let anchor = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
        let b_raw = array![[1.0, 2.0], [1.0, 1.5], [1.0, 0.5], [1.0, -1.0]];

        // Choose V_b that picks the second raw direction of B (kept dim = 1).
        // V_b is (p_b_raw=2) × (p_b_kept=1) selecting column 1 of B.
        let v_b = array![[0.0], [1.0]];

        // Solve M_b = (A'A)^{-1} A'B · V_b → here A'A is diag-ish so do
        // it via least squares directly. The contract is:
        //   R_b = M_b · V_b  where M_b ≈ (A'A)^{-1} A'B (size p_a × p_b_raw)
        // We can just compute the projection coefficients of B·V_b onto A.
        let bv = b_raw.dot(&v_b); // n × 1
        let ata = anchor.t().dot(&anchor); // 2x2
        let atbv = anchor.t().dot(&bv); // 2x1
        let ata_inv = {
            let det = ata[[0, 0]] * ata[[1, 1]] - ata[[0, 1]] * ata[[1, 0]];
            array![
                [ata[[1, 1]] / det, -ata[[0, 1]] / det],
                [-ata[[1, 0]] / det, ata[[0, 0]] / det],
            ]
        };
        let r_b: Array2<f64> = ata_inv.dot(&atbv); // 2 × 1  — already R_b

        let op = ResidualisedDesignOperator::new(
            DenseDesignMatrix::from(b_raw.clone()),
            v_b.clone(),
            vec![(DesignMatrix::from(anchor.clone()), Arc::new(r_b.clone()))],
        )
        .expect("residualised operator constructs");

        // Choose anchor coefficients γ_A and kept block coefficient θ_b.
        let gamma_a = ndarray::Array1::from_vec(vec![0.5, -1.25]);
        let theta_b = ndarray::Array1::from_vec(vec![2.5]);

        // Expected via the explicit emitted row: A·γ_A + (C_b·V_b − A·R_b)·θ_b.
        let cv = b_raw.dot(&v_b); // C_b · V_b
        let ar = anchor.dot(&r_b); // A · R_b
        let emitted_b_chunk = &cv - &ar;
        let expected = anchor.dot(&gamma_a) + emitted_b_chunk.dot(&theta_b);

        // Pull a streaming chunk through the operator and verify the
        // contribution matches the hand-computed (C_b·V_b − A·R_b)·θ_b row
        // by row.
        let mut got_chunk = Array2::<f64>::zeros((4, 1));
        op.row_chunk_into(0..4, got_chunk.view_mut())
            .expect("row chunk");
        let got = anchor.dot(&gamma_a) + got_chunk.dot(&theta_b);
        for i in 0..4 {
            assert!(
                (got[i] - expected[i]).abs() < 1e-10,
                "two-block reconstruction mismatch at row {i}: got {} expected {}",
                got[i],
                expected[i]
            );
        }

        // Cross-check via the LinearOperator::apply path (vector-valued v_b
        // matmul against the compiled width). This goes through the
        // streaming inner.apply / fast_av routes, distinct from the
        // row_chunk_into path covered above.
        let applied = op.apply(&theta_b);
        for i in 0..4 {
            assert!((applied[i] - emitted_b_chunk[[i, 0]] * theta_b[0]).abs() < 1e-10);
        }
    }
}

use crate::faer_ndarray::{
    FaerArrayView, array2_to_matmut, fast_ab, fast_atb, fast_atv, fast_atv_into, fast_av,
    fast_av_into, fast_xt_diag_x,
};
use crate::types::RidgePolicy;
use faer::Accum;
use faer::linalg::matmul::matmul;
use faer::sparse::{SparseColMat, SparseRowMat, Triplet};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ShapeBuilder, s};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::ops::Deref;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

const MATRIX_FREE_PCG_MIN_P: usize = 2048;
const MATRIX_FREE_PCG_REL_TOL: f64 = 1e-8;
const MATRIX_FREE_PCG_MAX_ITER: usize = 2000;
const MAX_PERSISTENT_SPARSE_DENSE_CACHE_BYTES: usize = 256 * 1024 * 1024;
const MAX_SPARSE_TO_DENSE_BYTES: usize = 4 * 1024 * 1024 * 1024;
const OPERATOR_ROW_CHUNK_SIZE: usize = 256;
/// Maximum bytes for the (n, tail_total) intermediate in GEMM-batched tensor
/// product matvecs.  Beyond this threshold, fall back to per-column GEMV.
const TENSOR_GEMM_MAX_INTERMEDIATE_BYTES: usize = 128 * 1024 * 1024; // 128 MB

pub use crate::linalg::utils::PcgSolveInfo;

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
            panic!("DenseRightProductView supports at most two right factors");
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

pub struct DenseRowScaledView<'a> {
    matrix: &'a Array2<f64>,
    scale: &'a Array1<f64>,
}

impl<'a> DenseRowScaledView<'a> {
    pub fn new(matrix: &'a Array2<f64>, scale: &'a Array1<f64>) -> Self {
        Self { matrix, scale }
    }

    pub fn materialize(&self) -> Array2<f64> {
        let mut out = self.matrix.clone();
        for (mut row, &weight) in out.outer_iter_mut().zip(self.scale.iter()) {
            row *= weight;
        }
        out
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
    penalty: Option<&'a Array2<f64>>,
    ridge: f64,
}

impl<'a, O: LinearOperator + ?Sized> PenalizedWeightedNormalOperator<'a, O> {
    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.operator
            .apply_weighted_normal(self.weights, vector, self.penalty, self.ridge)
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
fn dense_matvec(matrix: &Array2<f64>, vector: &Array1<f64>) -> Array1<f64> {
    fast_av(matrix, vector)
}

#[inline]
fn dense_transpose_matvec(matrix: &Array2<f64>, vector: &Array1<f64>) -> Array1<f64> {
    fast_atv(matrix, vector)
}

#[inline]
fn dense_transpose_weighted_response(
    matrix: &Array2<f64>,
    weights: &Array1<f64>,
    y: &Array1<f64>,
    row_scale: Option<&Array1<f64>>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(matrix.ncols());
    for i in 0..matrix.nrows() {
        let mut scaled = y[i] * weights[i].max(0.0);
        if let Some(scale) = row_scale {
            scaled *= scale[i];
        }
        if scaled == 0.0 {
            continue;
        }
        for j in 0..matrix.ncols() {
            out[j] += matrix[[i, j]] * scaled;
        }
    }
    out
}

#[derive(Clone)]
pub struct SparseDesignMatrix {
    matrix: SparseColMat<usize, f64>,
    dense_cache: Arc<OnceLock<Arc<Array2<f64>>>>,
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
        let dense_bytes = self.dense_nbytes()?;
        if dense_bytes > MAX_SPARSE_TO_DENSE_BYTES {
            let gib = dense_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            return Err(format!(
                "{context}: refusing to densify sparse design {}x{} (~{gib:.2} GiB); use sparse or matrix-free code",
                self.matrix.nrows(),
                self.matrix.ncols(),
            ));
        }
        if dense_bytes <= MAX_PERSISTENT_SPARSE_DENSE_CACHE_BYTES {
            Ok(self
                .dense_cache
                .get_or_init(|| self.materialize_dense_arc())
                .clone())
        } else {
            Ok(self.materialize_dense_arc())
        }
    }

    pub fn to_dense_arc(&self) -> Arc<Array2<f64>> {
        self.try_to_dense_arc("SparseDesignMatrix::to_dense_arc")
            .unwrap_or_else(|msg| panic!("{msg}"))
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
        let mut wy = Array1::<f64>::zeros(n);
        for i in 0..n {
            wy[i] = weights[i].max(0.0) * y[i];
        }
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
            let x_chunk = self.row_chunk(start..end);
            let xm_chunk = fast_ab(&x_chunk, middle);
            for i in 0..(end - start) {
                out[start + i] = x_chunk.row(i).dot(&xm_chunk.row(i)).max(0.0);
            }
            start = end;
        }
        Ok(out)
    }

    /// Extract a dense row chunk without materializing the full matrix.
    ///
    /// Returns a `(rows.len(), ncols())` dense matrix containing only the
    /// requested rows.  Concrete operators should override this with O(chunk)
    /// implementations; the default falls back through `to_dense()`.
    fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        // Fallback: materializes the full dense matrix.  Every concrete operator
        // should override this with an O(chunk) implementation.
        let dense_elems = self.nrows().saturating_mul(self.ncols());
        const DENSE_FALLBACK_WARN: usize = 10_000_000; // ~76 MiB at f64
        if dense_elems > DENSE_FALLBACK_WARN {
            log::warn!(
                "DenseDesignOperator::row_chunk default fallback materializing {}x{} ({:.1} MiB) — \
                 operator should implement a chunked override",
                self.nrows(),
                self.ncols(),
                (dense_elems * 8) as f64 / (1024.0 * 1024.0),
            );
        } else {
            log::debug!(
                "DenseDesignOperator::row_chunk default fallback (full materialization) for {}x{} operator",
                self.nrows(),
                self.ncols()
            );
        }
        self.to_dense().slice(s![rows, ..]).to_owned()
    }

    /// Borrow dense storage when this operator already owns it.
    fn as_dense_ref(&self) -> Option<&Array2<f64>> {
        None
    }

    /// Materialize the full dense matrix. Operators that exist precisely to
    /// avoid materialization should still support this for fallback paths,
    /// diagnostics, and prediction.
    fn to_dense(&self) -> Array2<f64>;

    /// Shared dense materialization. Implementations that already own an
    /// `Arc<Array2<_>>` should override this to return it directly.
    fn to_dense_arc(&self) -> Arc<Array2<f64>> {
        Arc::new(self.to_dense())
    }
}

#[derive(Clone)]
pub enum DenseDesignMatrix {
    Materialized(Arc<Array2<f64>>),
    Lazy(Arc<dyn DenseDesignOperator>),
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
        Self::Lazy(value)
    }
}

impl DenseDesignMatrix {
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
            Self::Lazy(op) => op.as_dense_ref(),
        }
    }

    pub fn to_dense(&self) -> Array2<f64> {
        match self {
            Self::Materialized(matrix) => matrix.as_ref().clone(),
            Self::Lazy(op) => op.to_dense(),
        }
    }

    pub fn to_dense_arc(&self) -> Arc<Array2<f64>> {
        match self {
            Self::Materialized(matrix) => Arc::clone(matrix),
            Self::Lazy(op) => op.to_dense_arc(),
        }
    }

    pub fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        match self {
            Self::Materialized(matrix) => matrix.slice(s![rows, ..]).to_owned(),
            Self::Lazy(op) => op.row_chunk(rows),
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
            Self::Materialized(matrix) => dense_matvec(matrix, vector),
            Self::Lazy(op) => op.apply(vector),
        }
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Materialized(matrix) => dense_transpose_matvec(matrix, vector),
            Self::Lazy(op) => op.apply_transpose(vector),
        }
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        match self {
            Self::Materialized(matrix) => {
                let mut xtwx = Array2::<f64>::zeros((matrix.ncols(), matrix.ncols()));
                streaming_blas_xt_diag_x(matrix, weights, &mut xtwx);
                Ok(xtwx)
            }
            Self::Lazy(op) => op.diag_xtw_x(weights),
        }
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        match self {
            Self::Materialized(matrix) => {
                let p = matrix.ncols();
                let mut diag = Array1::<f64>::zeros(p);
                for i in 0..matrix.nrows() {
                    let wi = weights[i].max(0.0);
                    if wi == 0.0 {
                        continue;
                    }
                    for j in 0..p {
                        let xij = matrix[[i, j]];
                        diag[j] += wi * xij * xij;
                    }
                }
                Ok(diag)
            }
            Self::Lazy(op) => op.diag_gram(weights),
        }
    }

    fn apply_weighted_normal(
        &self,
        weights: &Array1<f64>,
        vector: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge: f64,
    ) -> Array1<f64> {
        match self {
            Self::Materialized(matrix) => {
                let p = matrix.ncols();
                let mut out = Array1::<f64>::zeros(p);
                for i in 0..matrix.nrows() {
                    let wi = weights[i].max(0.0);
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
                if let Some(pen) = penalty {
                    out += &pen.dot(vector);
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
                let mut out = Array1::<f64>::zeros(matrix.nrows());
                for i in 0..matrix.nrows() {
                    out[i] = matrix.row(i).dot(&xc.row(i)).max(0.0);
                }
                Ok(out)
            }
            Self::Lazy(op) => op.quadratic_form_diag(middle),
        }
    }

    fn as_dense_ref(&self) -> Option<&Array2<f64>> {
        DenseDesignMatrix::as_dense_ref(self)
    }

    fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        DenseDesignMatrix::row_chunk(self, rows)
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
    /// Cached dense materialization of X·Qs.  Populated on first `to_dense()`
    /// call so that repeated outer-loop access (REML hyper derivatives, ALO,
    /// HMC) pays the O(n·p²) cost only once per PIRLS result.
    dense_cache: OnceLock<Array2<f64>>,
}

impl ReparamOperator {
    pub fn new(x_original: DesignMatrix, qs: Arc<Array2<f64>>) -> Self {
        let n = x_original.nrows();
        let p = qs.ncols();
        debug_assert_eq!(
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
            dense_cache: OnceLock::new(),
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
        weights: &Array1<f64>,
        vector: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge: f64,
    ) -> Array1<f64> {
        // Qs^T X^T W X Qs v + S v + ridge v
        let qv = self.qs.dot(vector);
        let xqv = self.x_original.apply(&qv);
        let mut wxqv = xqv;
        for i in 0..wxqv.len() {
            wxqv[i] *= weights[i].max(0.0);
        }
        let xtw = self.x_original.apply_transpose(&wxqv);
        let mut out = fast_atv(&self.qs, &xtw);
        if let Some(pen) = penalty {
            out += &pen.dot(vector);
        }
        if ridge > 0.0 {
            out += &vector.mapv(|x| ridge * x);
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
        // Cached materialization: pay the O(n·p²) cost at most once.
        self.dense_cache
            .get_or_init(|| match &self.x_original {
                DesignMatrix::Dense(x) => fast_ab(x.to_dense_arc().as_ref(), &self.qs),
                _ => {
                    let x_dense = self.x_original.to_dense();
                    fast_ab(&x_dense, &self.qs)
                }
            })
            .clone()
    }

    fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        // If the full dense product is already cached, just slice it.
        if let Some(cached) = self.dense_cache.get() {
            return cached.slice(s![rows, ..]).to_owned();
        }
        // Otherwise materialize only the requested rows: X[rows, :] · Qs
        match &self.x_original {
            DesignMatrix::Dense(x) => {
                let chunk = x.row_chunk(rows);
                fast_ab(&chunk, &self.qs)
            }
            DesignMatrix::Sparse(sdm) => {
                // Extract rows directly from CSR without densifying the full matrix.
                let csr = sdm
                    .to_csr_arc()
                    .expect("ReparamOperator::row_chunk: CSR conversion");
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
                fast_ab(&chunk.to_owned(), &self.qs)
            }
        }
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
    ) -> Array2<f64> {
        let p_dense = dense.ncols();
        let mut cross = Array2::<f64>::zeros((p_dense, self.num_groups));
        for i in 0..self.n {
            if let Some(g) = self.group_ids[i] {
                let wi = weights[i].max(0.0);
                if wi == 0.0 {
                    continue;
                }
                for j in 0..p_dense {
                    cross[[j, g]] += wi * dense[[i, j]];
                }
            }
        }
        cross
    }

    /// For two RE operators, compute X_re_a' diag(w) X_re_b → (qa × qb).
    /// Entry (a, b) = Σ_{i: group_a[i]=a AND group_b[i]=b} w[i].
    /// Cost: O(n).
    pub fn weighted_cross_with_re(
        &self,
        other: &RandomEffectOperator,
        weights: &Array1<f64>,
    ) -> Array2<f64> {
        let mut cross = Array2::<f64>::zeros((self.num_groups, other.num_groups));
        for i in 0..self.n {
            if let (Some(a), Some(b)) = (self.group_ids[i], other.group_ids[i]) {
                let wi = weights[i].max(0.0);
                if wi != 0.0 {
                    cross[[a, b]] += wi;
                }
            }
        }
        cross
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
        let mut out = Array1::<f64>::zeros(self.n);
        for i in 0..self.n {
            if let Some(g) = self.group_ids[i] {
                out[i] = vector[g];
            }
        }
        out
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
        let q = self.num_groups;
        let mut xtwx = Array2::<f64>::zeros((q, q));
        for i in 0..self.n {
            if let Some(g) = self.group_ids[i] {
                xtwx[[g, g]] += weights[i].max(0.0);
            }
        }
        Ok(xtwx)
    }

    /// Diagonal of X'WX: per-group weight sums.
    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        let mut diag = Array1::<f64>::zeros(self.num_groups);
        for i in 0..self.n {
            if let Some(g) = self.group_ids[i] {
                diag[g] += weights[i].max(0.0);
            }
        }
        Ok(diag)
    }

    /// Fused X'WXβ + Sβ + ridge·β.  O(n + q).
    fn apply_weighted_normal(
        &self,
        weights: &Array1<f64>,
        vector: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge: f64,
    ) -> Array1<f64> {
        // Step 1: accumulate per-group weighted β[g] contributions.
        //   group_acc[g] = Σ_{i in group g} w[i]
        //   result[g] = group_acc[g] * vector[g]
        let mut group_wacc = Array1::<f64>::zeros(self.num_groups);
        for i in 0..self.n {
            if let Some(g) = self.group_ids[i] {
                group_wacc[g] += weights[i].max(0.0);
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
        let mut out = Array1::<f64>::zeros(self.num_groups);
        for i in 0..self.n {
            if let Some(g) = self.group_ids[i] {
                let wi = weights[i].max(0.0);
                out[g] += wi * y[i];
            }
        }
        Ok(out)
    }

    /// diag(X M X') for one-hot X: out[i] = M[group[i], group[i]].
    fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        let mut out = Array1::<f64>::zeros(self.n);
        for i in 0..self.n {
            if let Some(g) = self.group_ids[i] {
                out[i] = middle[[g, g]].max(0.0);
            }
        }
        Ok(out)
    }

    fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        let chunk_rows = rows.end - rows.start;
        let mut out = Array2::<f64>::zeros((chunk_rows, self.num_groups));
        for (local, global) in rows.enumerate() {
            if let Some(g) = self.group_ids[global] {
                out[[local, g]] = 1.0;
            }
        }
        out
    }

    /// Materialize the full n × q one-hot matrix (fallback for diagnostics).
    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.n, self.num_groups));
        for i in 0..self.n {
            if let Some(g) = self.group_ids[i] {
                out[[i, g]] = 1.0;
            }
        }
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
    fn nrows(&self) -> usize {
        match self {
            Self::Dense(d) => d.nrows(),
            Self::Sparse(s) => s.nrows(),
            Self::RandomEffect(op) => op.nrows(),
            Self::Intercept(n) => *n,
        }
    }

    fn ncols(&self) -> usize {
        match self {
            Self::Dense(d) => d.ncols(),
            Self::Sparse(s) => s.ncols(),
            Self::RandomEffect(op) => op.ncols(),
            Self::Intercept(_) => 1,
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

    fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        match self {
            Self::Dense(d) => d.row_chunk(rows),
            Self::Sparse(s) => DesignMatrix::Sparse(s.clone()).row_chunk(rows),
            Self::RandomEffect(op) => op.row_chunk(rows),
            Self::Intercept(_) => Array2::ones((rows.end - rows.start, 1)),
        }
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        match self {
            Self::Dense(d) => d.diag_xtw_x(weights),
            Self::Sparse(s) => DesignMatrix::Sparse(s.clone()).diag_xtw_x(weights),
            Self::RandomEffect(op) => op.diag_xtw_x(weights),
            Self::Intercept(_) => {
                let sum: f64 = weights.iter().map(|w| w.max(0.0)).sum();
                Ok(Array2::from_elem((1, 1), sum))
            }
        }
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        match self {
            Self::Dense(d) => d.diag_gram(weights),
            Self::Sparse(s) => DesignMatrix::Sparse(s.clone()).diag_gram(weights),
            Self::RandomEffect(op) => op.diag_gram(weights),
            Self::Intercept(_) => {
                let sum: f64 = weights.iter().map(|w| w.max(0.0)).sum();
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
            let left_chunk = left.row_chunk(start..end);
            let right_chunk = right.row_chunk(start..end);
            for local in 0..(end - start) {
                let wi = weights[start + local].max(0.0);
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
            let a_chunk = block_a.row_chunk(start..end);
            let b_chunk = block_b.row_chunk(start..end);
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
                let d_i = d_i.to_dense_arc();
                let d_j = d_j.to_dense_arc();
                let pi = d_i.ncols();
                let pj = d_j.ncols();
                let mut cross = Array2::<f64>::zeros((pi, pj));
                for obs in 0..self.n {
                    let wi = weights[obs].max(0.0);
                    if wi == 0.0 {
                        continue;
                    }
                    for a in 0..pi {
                        let scaled = wi * d_i[[obs, a]];
                        if scaled == 0.0 {
                            continue;
                        }
                        for b in 0..pj {
                            cross[[a, b]] += scaled * d_j[[obs, b]];
                        }
                    }
                }
                Ok(cross)
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
                let dense = d.to_dense_arc();
                Ok(re.weighted_cross_with_dense(dense.as_ref(), weights))
            }
            (DesignBlock::RandomEffect(re), DesignBlock::Dense(d)) => {
                let dense = d.to_dense_arc();
                let cross_t = re.weighted_cross_with_dense(dense.as_ref(), weights);
                Ok(cross_t.t().to_owned())
            }

            // ── RandomEffect × RandomEffect ─────────────────────────────
            (DesignBlock::RandomEffect(re_a), DesignBlock::RandomEffect(re_b)) => {
                Ok(re_a.weighted_cross_with_re(re_b, weights))
            }

            // ── Intercept × anything ────────────────────────────────────
            // 1'·diag(w)·B_j  →  (1 × p_j) where entry [0,c] = Σ_i w[i] * B_j[i,c]
            (DesignBlock::Intercept(_), other) => {
                let pj = other.ncols();
                let mut cross = Array2::<f64>::zeros((1, pj));
                let weighted = Array1::from_shape_fn(self.n, |idx| weights[idx].max(0.0));
                let row = other.apply_transpose(&weighted);
                cross.row_mut(0).assign(&row);
                Ok(cross)
            }
            (other, DesignBlock::Intercept(_)) => {
                let pi = other.ncols();
                let mut cross = Array2::<f64>::zeros((pi, 1));
                let weighted = Array1::from_shape_fn(self.n, |idx| weights[idx].max(0.0));
                let col = other.apply_transpose(&weighted);
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
                let dense = d.to_dense_arc();
                let xm = fast_ab(dense.as_ref(), m_kk);
                let mut out = Array1::<f64>::zeros(self.n);
                for i in 0..self.n {
                    out[i] = dense.row(i).dot(&xm.row(i));
                }
                Ok(out)
            }
            DesignBlock::Sparse(s) => {
                let sparse = DesignMatrix::Sparse(s.clone());
                sparse.quadratic_form_diag(m_kk)
            }
            DesignBlock::RandomEffect(re) => {
                let mut out = Array1::<f64>::zeros(self.n);
                for i in 0..self.n {
                    if let Some(g) = re.group_ids[i] {
                        out[i] = m_kk[[g, g]];
                    }
                }
                Ok(out)
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
                let da = da.to_dense_arc();
                let db = db.to_dense_arc();
                let da_m = fast_ab(da.as_ref(), m_ab);
                let mut out = Array1::<f64>::zeros(self.n);
                for i in 0..self.n {
                    out[i] = da_m.row(i).dot(&db.row(i));
                }
                Ok(out)
            }
            (DesignBlock::Dense(_), DesignBlock::Sparse(_))
            | (DesignBlock::Sparse(_), DesignBlock::Dense(_))
            | (DesignBlock::Sparse(_), DesignBlock::Sparse(_))
            | (DesignBlock::Sparse(_), DesignBlock::RandomEffect(_))
            | (DesignBlock::RandomEffect(_), DesignBlock::Sparse(_)) => {
                self.quadratic_form_diag_cross_chunked(block_a, block_b, m_ab)
            }
            (DesignBlock::Dense(d), DesignBlock::RandomEffect(re)) => {
                let d = d.to_dense_arc();
                let mut out = Array1::<f64>::zeros(self.n);
                for i in 0..self.n {
                    if let Some(g) = re.group_ids[i] {
                        let mut val = 0.0;
                        for j in 0..d.ncols() {
                            val += d[[i, j]] * m_ab[[j, g]];
                        }
                        out[i] = val;
                    }
                }
                Ok(out)
            }
            (DesignBlock::RandomEffect(re), DesignBlock::Dense(d)) => {
                let d = d.to_dense_arc();
                let mut out = Array1::<f64>::zeros(self.n);
                for i in 0..self.n {
                    if let Some(g) = re.group_ids[i] {
                        let mut val = 0.0;
                        for j in 0..d.ncols() {
                            val += m_ab[[g, j]] * d[[i, j]];
                        }
                        out[i] = val;
                    }
                }
                Ok(out)
            }
            (DesignBlock::RandomEffect(re_a), DesignBlock::RandomEffect(re_b)) => {
                let mut out = Array1::<f64>::zeros(self.n);
                for i in 0..self.n {
                    if let (Some(ga), Some(gb)) = (re_a.group_ids[i], re_b.group_ids[i]) {
                        out[i] = m_ab[[ga, gb]];
                    }
                }
                Ok(out)
            }

            // Intercept × anything: contribution at row i = m_ab[0, :] · row_i(B_b)
            (DesignBlock::Intercept(_), other) => {
                let m_row = m_ab.row(0);
                let mut out = Array1::<f64>::zeros(self.n);
                for start in (0..self.n).step_by(OPERATOR_ROW_CHUNK_SIZE) {
                    let end = (start + OPERATOR_ROW_CHUNK_SIZE).min(self.n);
                    let chunk = other.row_chunk(start..end);
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
                    let chunk = other.row_chunk(start..end);
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
        for i in 0..self.blocks.len() {
            for j in (i + 1)..self.blocks.len() {
                let cross = self.cross_block(i, j, weights)?;
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
        weights: &Array1<f64>,
        vector: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge: f64,
    ) -> Array1<f64> {
        // Fused: X'W(Xβ) + Sβ + ridge·β
        let xv = self.apply(vector);
        let mut weighted = xv;
        for i in 0..weighted.len() {
            weighted[i] *= weights[i].max(0.0);
        }
        let mut out = self.apply_transpose(&weighted);
        if let Some(pen) = penalty {
            out += &pen.dot(vector);
        }
        if ridge > 0.0 {
            out += &vector.mapv(|x| ridge * x);
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
    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
        let mut wy = Array1::<f64>::zeros(self.n);
        for i in 0..self.n {
            wy[i] = weights[i].max(0.0) * y[i];
        }
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

    fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        let chunk_rows = rows.end - rows.start;
        let mut out = Array2::<f64>::zeros((chunk_rows, self.total_cols));
        for (idx, block) in self.blocks.iter().enumerate() {
            let cs = self.col_offsets[idx];
            let ce = self.col_offsets[idx + 1];
            let block_chunk = block.row_chunk(rows.clone());
            out.slice_mut(s![.., cs..ce]).assign(&block_chunk);
        }
        out
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
// ReparamDesignOperator — right-multiply by a coefficient transform
// ---------------------------------------------------------------------------

/// Reparameterized design operator: X_new = X_inner · Q.
///
/// Wraps any `DenseDesignOperator` and applies a (p_inner, p_new) right transform
/// without materializing the transformed design.  Common uses:
///
///  * Identifiability constraints (sum-to-zero projections)
///  * Shape reparameterization (cumulative-sum transforms)
///
///   apply(β)           = inner.apply(Q · β)
///   apply_transpose(v) = Q' · inner.apply_transpose(v)
///   X'WX               = Q' · inner.diag_xtw_x(w) · Q
pub struct ReparamDesignOperator {
    pub inner: Arc<dyn DenseDesignOperator>,
    pub q: Arc<Array2<f64>>,
    n: usize,
    p_new: usize,
}

impl ReparamDesignOperator {
    pub fn new(inner: Arc<dyn DenseDesignOperator>, q: Arc<Array2<f64>>) -> Result<Self, String> {
        let p_inner = inner.ncols();
        if q.nrows() != p_inner {
            return Err(format!(
                "ReparamDesignOperator: inner has {} cols but Q has {} rows",
                p_inner,
                q.nrows()
            ));
        }
        Ok(Self {
            n: inner.nrows(),
            p_new: q.ncols(),
            inner,
            q,
        })
    }
}

impl LinearOperator for ReparamDesignOperator {
    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.p_new
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        let transformed = self.q.dot(vector);
        self.inner.apply(&transformed)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        let inner_result = self.inner.apply_transpose(vector);
        self.q.t().dot(&inner_result)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        let inner_gram = self.inner.diag_xtw_x(weights)?;
        let qt_gram = fast_atb(&self.q, &inner_gram);
        Ok(fast_ab(&qt_gram, &self.q))
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        let xtwx = self.diag_xtw_x(weights)?;
        Ok(Array1::from_iter((0..self.p_new).map(|j| xtwx[[j, j]])))
    }

    fn apply_weighted_normal(
        &self,
        weights: &Array1<f64>,
        vector: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge: f64,
    ) -> Array1<f64> {
        let q_beta = self.q.dot(vector);
        let inner_result = self
            .inner
            .apply_weighted_normal(weights, &q_beta, None, 0.0);
        let mut out = self.q.t().dot(&inner_result);
        if let Some(pen) = penalty {
            out += &pen.dot(vector);
        }
        if ridge > 0.0 {
            out += &vector.mapv(|x| ridge * x);
        }
        out
    }

    fn uses_matrix_free_pcg(&self) -> bool {
        self.inner.uses_matrix_free_pcg()
    }
}

// ReparamDesignOperator contains Arc<dyn DenseDesignOperator> which is Send + Sync,
// so the type is safe to send/share across threads.
unsafe impl Send for ReparamDesignOperator {}
unsafe impl Sync for ReparamDesignOperator {}

impl DenseDesignOperator for ReparamDesignOperator {
    fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        let inner_chunk = self.inner.row_chunk(rows);
        fast_ab(&inner_chunk, &self.q)
    }

    fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        // diag(X_new M X_new') = diag(X_inner (Q M Q') X_inner')
        let qm = fast_ab(&self.q, middle); // (p_inner, p_new)
        let qmqt = fast_ab(&qm, &self.q.t()); // (p_inner, p_inner)
        self.inner.quadratic_form_diag(&qmqt)
    }

    fn to_dense(&self) -> Array2<f64> {
        let inner_dense = self.inner.to_dense();
        fast_ab(&inner_dense, &self.q)
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
            let slice = vector.slice(s![i * n..(i + 1) * n]).to_owned();
            out += &ch.transpose_vector_multiply(&slice);
        }
        out
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        let n = self.n_per_channel;
        if weights.len() != self.nrows() {
            return Err(format!(
                "MultiChannelOperator::diag_xtw_x: weights length {} != nrows {}",
                weights.len(),
                self.nrows()
            ));
        }
        let mut xtwx = Array2::<f64>::zeros((self.p, self.p));
        for (i, ch) in self.channels.iter().enumerate() {
            let w_slice = weights.slice(s![i * n..(i + 1) * n]).to_owned();
            let ch_xtwx = ch.compute_xtwx(&w_slice)?;
            xtwx += &ch_xtwx;
        }
        Ok(xtwx)
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        let n = self.n_per_channel;
        if weights.len() != self.nrows() {
            return Err(format!(
                "MultiChannelOperator::diag_gram: weights length {} != nrows {}",
                weights.len(),
                self.nrows()
            ));
        }
        let mut diag = Array1::<f64>::zeros(self.p);
        for (i, ch) in self.channels.iter().enumerate() {
            let w_slice = weights.slice(s![i * n..(i + 1) * n]).to_owned();
            diag += &<DesignMatrix as LinearOperator>::diag_gram(ch, &w_slice)?;
        }
        Ok(diag)
    }

    fn uses_matrix_free_pcg(&self) -> bool {
        true
    }
}

impl DenseDesignOperator for MultiChannelOperator {
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
        let mut out = Array1::<f64>::zeros(self.p);
        for (i, ch) in self.channels.iter().enumerate() {
            let w_slice = weights.slice(s![i * n..(i + 1) * n]).to_owned();
            let y_slice = y.slice(s![i * n..(i + 1) * n]).to_owned();
            out += &ch.compute_xtwy(&w_slice, &y_slice)?;
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

    fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        let n = self.n_per_channel;
        let chunk_rows = rows.end - rows.start;
        let mut out = Array2::<f64>::zeros((chunk_rows, self.p));
        let mut local = 0usize;
        let mut global = rows.start;
        while global < rows.end {
            let ch_idx = global / n;
            let ch_local_start = global % n;
            let ch_local_end = ((ch_idx + 1) * n).min(rows.end) - ch_idx * n;
            let segment_len = ch_local_end - ch_local_start;
            let ch_chunk = self.channels[ch_idx].row_chunk(ch_local_start..ch_local_end);
            out.slice_mut(s![local..local + segment_len, ..])
                .assign(&ch_chunk);
            local += segment_len;
            global += segment_len;
        }
        out
    }
}

/// Rowwise-Kronecker design operator: represents the (n, p_cov × p_time) matrix
/// whose row i is the Kronecker product cov[i,:] ⊗ time[i,:].
///
/// This avoids materializing the full tensor product design, which at biobank
/// scale can be tens of GB.
///
///   X[i, j*p_time + t] = cov[i, j] * time[i, t]
///
/// All matvec and Gram operations are performed in factored form.
#[derive(Clone)]
pub struct RowwiseKroneckerOperator {
    /// Covariate factor: (n, p_cov).
    pub cov: DesignMatrix,
    /// Time basis factor: (n, p_time).  Dense because B-spline bases are
    /// always dense (though banded — only ~4 nonzeros per row for degree 3).
    pub time_basis: Arc<Array2<f64>>,
    /// Cached dimensions.
    pub n: usize,
    pub p_cov: usize,
    pub p_time: usize,
}

/// Generic rowwise Kronecker operator for dense marginal designs.
///
/// Decode a flat index into per-dimension indices for a row-major tensor
/// with the given dimension sizes.  Writes into the provided `out` slice
/// to avoid allocation.
///
///   decode_multi_index(flat, &[3, 4], &mut out) → out = [flat / 4, flat % 4]
fn decode_multi_index(mut flat: usize, dims: &[usize], out: &mut [usize]) {
    for d in (0..dims.len()).rev() {
        out[d] = flat % dims[d];
        flat /= dims[d];
    }
}

/// Each row is the Kronecker product of the corresponding marginal rows:
/// `X[i, :] = B_1[i, :] ⊗ ... ⊗ B_d[i, :]`.
///
/// This keeps tensor-product terms operator-backed in the main model path so
/// fitting no longer requires an eager `n x prod(q_j)` realization.
pub struct TensorProductDesignOperator {
    marginals: Vec<Arc<Array2<f64>>>,
    n: usize,
    total_cols: usize,
}

impl TensorProductDesignOperator {
    pub fn new(marginals: Vec<Arc<Array2<f64>>>) -> Result<Self, String> {
        if marginals.is_empty() {
            return Err("TensorProductDesignOperator requires at least one marginal".to_string());
        }
        let n = marginals[0].nrows();
        let total_cols = marginals.iter().try_fold(1usize, |acc, marginal| {
            if marginal.nrows() != n {
                return Err(format!(
                    "TensorProductDesignOperator row mismatch: expected {n}, got {}",
                    marginal.nrows()
                ));
            }
            acc.checked_mul(marginal.ncols()).ok_or_else(|| {
                "TensorProductDesignOperator total column count overflow".to_string()
            })
        })?;
        Ok(Self {
            marginals,
            n,
            total_cols,
        })
    }

    /// Materialize the full Kronecker row for observation `row`.
    /// Only used by fallback paths (quadratic_form_diag, row_chunk);
    /// the hot-path apply/apply_transpose use sequential contraction instead.
    fn row_values(&self, row: usize) -> Vec<f64> {
        let mut values = vec![1.0_f64];
        for marginal in &self.marginals {
            let q = marginal.ncols();
            let mut next = vec![0.0_f64; values.len() * q];
            for (prefix_idx, &prefix) in values.iter().enumerate() {
                for col in 0..q {
                    next[prefix_idx * q + col] = prefix * marginal[[row, col]];
                }
            }
            values = next;
        }
        values
    }

    /// Compute Xβ via column-wise BLAS matvecs across all n observations.
    ///
    /// β is conceptually a (q₁, q₂, …, qₖ) tensor.  We iterate over all
    /// "tail columns" (indices into dimensions 2..k), and for each:
    ///
    ///   1. Extract β_slice = β[:, t₂, …, tₖ]          — q₁-vector
    ///   2. contrib = B₁ · β_slice                       — ONE BLAS matvec, O(n·q₁)
    ///   3. contrib ⊙= B₂[:,t₂] ⊙ … ⊙ Bₖ[:,tₖ]        — k-1 elementwise O(n) passes
    ///   4. result += contrib
    ///
    /// Total: ∏_{j>1}qⱼ BLAS matvecs.  Same asymptotic cost as per-row scalar
    /// contraction, but each operation is a vectorized n-length pass with BLAS
    /// cache optimization.  Zero per-row allocation.
    fn apply_vectorized(&self, vector: &Array1<f64>) -> Array1<f64> {
        let d = self.marginals.len();
        let n = self.n;
        if d == 0 {
            return Array1::zeros(n);
        }
        let b0 = &self.marginals[0];
        let q0 = b0.ncols();
        if d == 1 {
            return fast_av(b0.as_ref(), vector);
        }

        let tail_dims: Vec<usize> = self.marginals[1..].iter().map(|m| m.ncols()).collect();
        let tail_total: usize = tail_dims.iter().product();
        let intermediate_bytes = n * tail_total * std::mem::size_of::<f64>();

        if intermediate_bytes <= TENSOR_GEMM_MAX_INTERMEDIATE_BYTES {
            // ── GEMM path: one BLAS3 call for the B₁ contraction ────────
            //
            // Reshape β to (q₁, tail_total), compute B₁ · β_mat → (n, tail_total)
            // via a single GEMM.  Then elementwise-multiply each column by the
            // corresponding tail marginal products and row-sum.
            //
            // Zero-copy reshape: β is contiguous and q₁·tail_total = total_cols.
            let beta_view =
                ndarray::ArrayView2::from_shape((q0, tail_total), vector.as_slice().unwrap())
                    .expect("β reshape for GEMM");
            let temp = fast_ab(b0.as_ref(), &beta_view); // (n, tail_total)

            let mut out = Array1::<f64>::zeros(n);
            let mut tail_indices = vec![0usize; tail_dims.len()];
            for t_flat in 0..tail_total {
                decode_multi_index(t_flat, &tail_dims, &mut tail_indices);
                for i in 0..n {
                    let mut val = temp[[i, t_flat]];
                    for (dim_idx, &ti) in tail_indices.iter().enumerate() {
                        val *= self.marginals[dim_idx + 1][[i, ti]];
                    }
                    out[i] += val;
                }
            }
            out
        } else {
            // ── GEMV fallback: one BLAS2 call per tail column ───────────
            let mut tail_indices = vec![0usize; tail_dims.len()];
            let mut out = Array1::<f64>::zeros(n);
            let mut beta_slice = Array1::<f64>::zeros(q0);
            let mut contrib = Array1::<f64>::zeros(n);

            for t_flat in 0..tail_total {
                decode_multi_index(t_flat, &tail_dims, &mut tail_indices);
                for j1 in 0..q0 {
                    beta_slice[j1] = vector[j1 * tail_total + t_flat];
                }
                fast_av_into(b0.as_ref(), &beta_slice, &mut contrib);
                for (dim_idx, &ti) in tail_indices.iter().enumerate() {
                    let m = &self.marginals[dim_idx + 1];
                    for i in 0..n {
                        contrib[i] *= m[[i, ti]];
                    }
                }
                out += &contrib;
            }
            out
        }
    }

    /// Compute X'v via column-wise BLAS transpose matvecs across all n observations.
    ///
    /// For each tail column t = (t₂, …, tₖ):
    ///   1. scaled_v = v ⊙ B₂[:,t₂] ⊙ … ⊙ Bₖ[:,tₖ]   — elementwise O(n)
    ///   2. out[:, t] = B₁' · scaled_v                   — ONE BLAS transpose matvec
    ///
    /// Total: ∏_{j>1}qⱼ BLAS transpose matvecs.
    fn apply_transpose_vectorized(&self, vector: &Array1<f64>) -> Array1<f64> {
        let d = self.marginals.len();
        let n = self.n;
        if d == 0 {
            return Array1::zeros(self.total_cols);
        }
        let b0 = &self.marginals[0];
        let q0 = b0.ncols();
        if d == 1 {
            return fast_atv(b0.as_ref(), vector);
        }

        let tail_dims: Vec<usize> = self.marginals[1..].iter().map(|m| m.ncols()).collect();
        let tail_total: usize = tail_dims.iter().product();
        let intermediate_bytes = n * tail_total * std::mem::size_of::<f64>();

        if intermediate_bytes <= TENSOR_GEMM_MAX_INTERMEDIATE_BYTES {
            // ── GEMM path: build W matrix, one BLAS3 call ───────────────
            //
            // W[i, t_flat] = v[i] · ∏_{d>1} Bᵈ[i, tᵈ]
            // Then B₁' · W → (q₁, tail_total) via one GEMM.
            let mut w_mat = Array2::<f64>::zeros((n, tail_total));
            let mut tail_indices = vec![0usize; tail_dims.len()];
            for t_flat in 0..tail_total {
                decode_multi_index(t_flat, &tail_dims, &mut tail_indices);
                for i in 0..n {
                    let mut val = vector[i];
                    for (dim_idx, &ti) in tail_indices.iter().enumerate() {
                        val *= self.marginals[dim_idx + 1][[i, ti]];
                    }
                    w_mat[[i, t_flat]] = val;
                }
            }
            let result_mat = fast_atb(b0.as_ref(), &w_mat); // (q₁, tail_total)

            // Scatter from (q₁, tail_total) matrix into flat output.
            let mut out = Array1::<f64>::zeros(self.total_cols);
            for j1 in 0..q0 {
                for t_flat in 0..tail_total {
                    out[j1 * tail_total + t_flat] = result_mat[[j1, t_flat]];
                }
            }
            out
        } else {
            // ── GEMV fallback ───────────────────────────────────────────
            let mut tail_indices = vec![0usize; tail_dims.len()];
            let mut out = Array1::<f64>::zeros(self.total_cols);
            let mut scaled_v = Array1::<f64>::zeros(n);
            let mut col_result = Array1::<f64>::zeros(q0);

            for t_flat in 0..tail_total {
                decode_multi_index(t_flat, &tail_dims, &mut tail_indices);
                scaled_v.assign(vector);
                for (dim_idx, &ti) in tail_indices.iter().enumerate() {
                    let m = &self.marginals[dim_idx + 1];
                    for i in 0..n {
                        scaled_v[i] *= m[[i, ti]];
                    }
                }
                fast_atv_into(b0.as_ref(), &scaled_v, &mut col_result);
                for j1 in 0..q0 {
                    out[j1 * tail_total + t_flat] = col_result[j1];
                }
            }
            out
        }
    }
}

impl LinearOperator for TensorProductDesignOperator {
    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.total_cols
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.apply_vectorized(vector)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.apply_transpose_vectorized(vector)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        if weights.len() != self.n {
            return Err(format!(
                "TensorProductDesignOperator::diag_xtw_x: weights length {} != n {}",
                weights.len(),
                self.n
            ));
        }
        let d = self.marginals.len();
        if d == 0 {
            return Ok(Array2::zeros((0, 0)));
        }
        let n = self.n;
        let q0 = self.marginals[0].ncols();

        // ── Factored Gram computation ──────────────────────────────────
        //
        // Generalizes RowwiseKroneckerOperator's gamma approach to k factors.
        //
        // X'WX[multi_a, multi_b] =
        //   Σ_i w[i] · B₁[i,a₁]·B₂[i,a₂]·…·Bₖ[i,aₖ] · B₁[i,b₁]·B₂[i,b₂]·…·Bₖ[i,bₖ]
        //
        // Factor out B₁:
        //   = Σ_i (w[i] · B₂[i,a₂]·B₂[i,b₂] · … · Bₖ[i,aₖ]·Bₖ[i,bₖ]) · B₁[i,a₁]·B₁[i,b₁]
        //
        // For each tuple (a₂,b₂,…,aₖ,bₖ), form γ[i] = w[i]·∏_{d>1} Bd[i,ad]·Bd[i,bd],
        // then the (a₁,b₁) block = B₁'·diag(γ)·B₁  which is a q₁×q₁ gram.
        //
        // This avoids per-row allocation and computes many small BLAS grams
        // instead of one huge (∏qⱼ)×(∏qⱼ) outer product.

        let mut xtwx = Array2::<f64>::zeros((self.total_cols, self.total_cols));
        let b0 = &self.marginals[0];

        // Collect tail marginal dimensions.
        let tail_dims: Vec<usize> = self.marginals[1..].iter().map(|m| m.ncols()).collect();
        let tail_total: usize = tail_dims.iter().product();

        // Iterate over all (a_tail, b_tail) index pairs in the tail dimensions.
        // For symmetry, only iterate a_flat <= b_flat and mirror.
        let mut gamma = Array1::<f64>::zeros(n);
        let mut block = Array2::<f64>::zeros((q0, q0));
        let tail_d = tail_dims.len();
        let mut a_indices = vec![0usize; tail_d];
        let mut b_indices = vec![0usize; tail_d];

        for a_flat in 0..tail_total {
            decode_multi_index(a_flat, &tail_dims, &mut a_indices);

            for b_flat in a_flat..tail_total {
                decode_multi_index(b_flat, &tail_dims, &mut b_indices);

                // Form γ[i] = w[i] · ∏_{d=1..k-1} B_{d+1}[i, a_d] · B_{d+1}[i, b_d]
                for i in 0..n {
                    let mut prod = weights[i].max(0.0);
                    if prod == 0.0 {
                        gamma[i] = 0.0;
                        continue;
                    }
                    for dim_idx in 0..tail_d {
                        let m = &self.marginals[dim_idx + 1];
                        prod *= m[[i, a_indices[dim_idx]]] * m[[i, b_indices[dim_idx]]];
                        if prod == 0.0 {
                            break;
                        }
                    }
                    gamma[i] = prod;
                }

                // Compute B₁' · diag(γ) · B₁ → (q₀ × q₀) block via BLAS.
                block.fill(0.0);
                streaming_blas_xt_diag_x(b0.as_ref(), &gamma, &mut block);

                // Scatter block into the full xtwx.
                // Global column for (j₁, tail_flat) = j₁ * tail_total + tail_flat.
                for a1 in 0..q0 {
                    let ga = a1 * tail_total + a_flat;
                    for b1 in 0..q0 {
                        let gb = b1 * tail_total + b_flat;
                        xtwx[[ga, gb]] += block[[a1, b1]];
                        if a_flat != b_flat {
                            let ga_mirror = a1 * tail_total + b_flat;
                            let gb_mirror = b1 * tail_total + a_flat;
                            xtwx[[ga_mirror, gb_mirror]] += block[[a1, b1]];
                        }
                    }
                }
            }
        }
        Ok(xtwx)
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        if weights.len() != self.n {
            return Err(format!(
                "TensorProductDesignOperator::diag_gram: weights length {} != n {}",
                weights.len(),
                self.n
            ));
        }
        // diag(X'WX)[j] = Σ_i w[i] · x_{ij}²
        // For tensor product: x_{i, j₁·tail+j_tail} = B₁[i,j₁] · ∏_{d>1} Bᵈ[i,jᵈ]
        // So: diag[j] = Σ_i w[i] · B₁[i,j₁]² · ∏_{d>1} Bᵈ[i,jᵈ]²
        //
        // O(n · ∏qⱼ) instead of O(n · (∏qⱼ)²) from the full gram.
        let d = self.marginals.len();
        if d == 0 {
            return Ok(Array1::zeros(0));
        }
        let mut diag = vec![0.0_f64; self.total_cols];
        let tail_dims: Vec<usize> = self.marginals[1..].iter().map(|m| m.ncols()).collect();
        let tail_total: usize = tail_dims.iter().product();
        let q0 = self.marginals[0].ncols();
        let mut tail_indices = vec![0usize; tail_dims.len()];

        for t_flat in 0..tail_total {
            decode_multi_index(t_flat, &tail_dims, &mut tail_indices);
            for i in 0..self.n {
                let wi = weights[i].max(0.0);
                if wi == 0.0 {
                    continue;
                }
                let mut tail_prod_sq = wi;
                for (dim_idx, &ti) in tail_indices.iter().enumerate() {
                    let val = self.marginals[dim_idx + 1][[i, ti]];
                    tail_prod_sq *= val * val;
                    if tail_prod_sq == 0.0 {
                        break;
                    }
                }
                if tail_prod_sq == 0.0 {
                    continue;
                }
                for j1 in 0..q0 {
                    let b1 = self.marginals[0][[i, j1]];
                    diag[j1 * tail_total + t_flat] += tail_prod_sq * b1 * b1;
                }
            }
        }
        Ok(Array1::from_vec(diag))
    }

    fn uses_matrix_free_pcg(&self) -> bool {
        true
    }
}

impl DenseDesignOperator for TensorProductDesignOperator {
    fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        if middle.nrows() != self.total_cols || middle.ncols() != self.total_cols {
            return Err(format!(
                "TensorProductDesignOperator::quadratic_form_diag dimension mismatch: {}x{} vs expected {}x{}",
                middle.nrows(),
                middle.ncols(),
                self.total_cols,
                self.total_cols
            ));
        }
        let mut out = Array1::<f64>::zeros(self.n);
        for start in (0..self.n).step_by(OPERATOR_ROW_CHUNK_SIZE) {
            let end = (start + OPERATOR_ROW_CHUNK_SIZE).min(self.n);
            let chunk = self.row_chunk(start..end);
            let chunk_m = fast_ab(&chunk, middle);
            for local in 0..(end - start) {
                out[start + local] = chunk.row(local).dot(&chunk_m.row(local)).max(0.0);
            }
        }
        Ok(out)
    }

    fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((rows.end - rows.start, self.total_cols));
        for (local_row, global_row) in rows.enumerate() {
            let row_values = self.row_values(global_row);
            for (j, &value) in row_values.iter().enumerate() {
                out[[local_row, j]] = value;
            }
        }
        out
    }

    fn to_dense(&self) -> Array2<f64> {
        self.row_chunk(0..self.n)
    }
}

/// Chunked kernel design operator for spatial smooths (TPS, Matérn, Duchon).
///
/// Instead of storing a dense n × k matrix, evaluates K(data[i], center[j])
/// on-the-fly in row chunks. Memory usage is O(chunk_size × k) instead of O(n × k).
///
/// The optional `poly_basis` appends polynomial columns after the kernel columns
/// (e.g., linear polynomial for TPS identifiability).
///
/// The optional `constraint_transform` applies a column-space projection Z
/// such that the effective design is [K * Z | poly] instead of [K | poly].
pub struct ChunkedKernelDesignOperator {
    /// Observation data points (n × d).
    data: Arc<Array2<f64>>,
    /// Radial basis centers (k × d).
    centers: Arc<Array2<f64>>,
    /// Kernel evaluator: (data_row, center_row) → f64
    kernel_fn: Arc<dyn Fn(&[f64], &[f64]) -> f64 + Send + Sync>,
    /// Optional constraint projection (k × k_eff) applied to kernel columns.
    constraint_transform: Option<Arc<Array2<f64>>>,
    /// Optional polynomial basis columns (n × m) appended after kernel columns.
    poly_basis: Option<Arc<Array2<f64>>>,
    n: usize,
    total_cols: usize,
}

impl ChunkedKernelDesignOperator {
    pub fn new(
        data: Arc<Array2<f64>>,
        centers: Arc<Array2<f64>>,
        kernel_fn: Arc<dyn Fn(&[f64], &[f64]) -> f64 + Send + Sync>,
        constraint_transform: Option<Arc<Array2<f64>>>,
        poly_basis: Option<Arc<Array2<f64>>>,
    ) -> Result<Self, String> {
        let n = data.nrows();
        let k = centers.ncols();
        if data.ncols() != centers.ncols() {
            return Err(format!(
                "ChunkedKernelDesignOperator: data dim {} != centers dim {}",
                data.ncols(),
                centers.ncols(),
            ));
        }
        let k_eff = constraint_transform.as_ref().map_or(k, |z| z.ncols());
        let poly_cols = poly_basis.as_ref().map_or(0, |p| p.ncols());
        Ok(Self {
            data,
            centers,
            kernel_fn,
            constraint_transform,
            poly_basis,
            n,
            total_cols: k_eff + poly_cols,
        })
    }

    /// Evaluate kernel block for a range of rows: K[rows, :] or K[rows, :] * Z.
    fn kernel_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        let chunk_n = rows.end - rows.start;
        let k_raw = self.centers.nrows();
        let mut kernel_block = Array2::<f64>::zeros((chunk_n, k_raw));
        for (local, global) in rows.clone().enumerate() {
            let data_row = self.data.row(global);
            for j in 0..k_raw {
                let center_row = self.centers.row(j);
                kernel_block[[local, j]] =
                    (self.kernel_fn)(data_row.as_slice().unwrap(), center_row.as_slice().unwrap());
            }
        }
        if let Some(z) = self.constraint_transform.as_ref() {
            fast_ab(&kernel_block, z)
        } else {
            kernel_block
        }
    }
}

impl LinearOperator for ChunkedKernelDesignOperator {
    fn nrows(&self) -> usize {
        self.n
    }
    fn ncols(&self) -> usize {
        self.total_cols
    }
    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        let k_eff = self
            .constraint_transform
            .as_ref()
            .map_or(self.centers.nrows(), |z| z.ncols());
        let v_kernel = vector.slice(s![..k_eff]);
        let mut result = Array1::<f64>::zeros(self.n);
        // Process in chunks to limit memory.
        for start in (0..self.n).step_by(OPERATOR_ROW_CHUNK_SIZE) {
            let end = (start + OPERATOR_ROW_CHUNK_SIZE).min(self.n);
            let chunk = self.kernel_chunk(start..end);
            let partial = chunk.dot(&v_kernel);
            result.slice_mut(s![start..end]).assign(&partial);
        }
        if let Some(poly) = self.poly_basis.as_ref() {
            let v_poly = vector.slice(s![k_eff..]);
            let poly_part = poly.dot(&v_poly);
            result += &poly_part;
        }
        result
    }
    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        let k_eff = self
            .constraint_transform
            .as_ref()
            .map_or(self.centers.nrows(), |z| z.ncols());
        let mut result = Array1::<f64>::zeros(self.total_cols);
        // Kernel part: chunked accumulation of K^T v.
        for start in (0..self.n).step_by(OPERATOR_ROW_CHUNK_SIZE) {
            let end = (start + OPERATOR_ROW_CHUNK_SIZE).min(self.n);
            let chunk = self.kernel_chunk(start..end);
            let v_slice = vector.slice(s![start..end]);
            let partial = chunk.t().dot(&v_slice);
            result.slice_mut(s![..k_eff]).scaled_add(1.0, &partial);
        }
        // Poly part.
        if let Some(poly) = self.poly_basis.as_ref() {
            let poly_part = poly.t().dot(vector);
            result.slice_mut(s![k_eff..]).assign(&poly_part);
        }
        result
    }
    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        let p = self.total_cols;
        let mut xtwx = Array2::<f64>::zeros((p, p));
        for start in (0..self.n).step_by(OPERATOR_ROW_CHUNK_SIZE) {
            let end = (start + OPERATOR_ROW_CHUNK_SIZE).min(self.n);
            let mut chunk = self.row_chunk_combined(start..end);
            // Apply sqrt(w) to each row.
            for local in 0..(end - start) {
                let w = weights[start + local].max(0.0).sqrt();
                chunk.row_mut(local).mapv_inplace(|v| v * w);
            }
            // Accumulate chunk^T chunk.
            let ata = fast_ab(&chunk.t().to_owned(), &chunk);
            xtwx += &ata;
        }
        Ok(xtwx)
    }
}

impl ChunkedKernelDesignOperator {
    /// Combined row chunk: [kernel_chunk | poly_chunk].
    fn row_chunk_combined(&self, rows: Range<usize>) -> Array2<f64> {
        let chunk_n = rows.end - rows.start;
        let k_eff = self
            .constraint_transform
            .as_ref()
            .map_or(self.centers.nrows(), |z| z.ncols());
        let kernel = self.kernel_chunk(rows.clone());
        let poly_cols = self.poly_basis.as_ref().map_or(0, |p| p.ncols());
        let mut combined = Array2::<f64>::zeros((chunk_n, k_eff + poly_cols));
        combined.slice_mut(s![.., ..k_eff]).assign(&kernel);
        if let Some(poly) = self.poly_basis.as_ref() {
            combined
                .slice_mut(s![.., k_eff..])
                .assign(&poly.slice(s![rows, ..]));
        }
        combined
    }
}

impl DenseDesignOperator for ChunkedKernelDesignOperator {
    fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        self.row_chunk_combined(rows)
    }
    fn to_dense(&self) -> Array2<f64> {
        self.row_chunk_combined(0..self.n)
    }
}

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
}

impl CoefficientTransformOperator {
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
        })
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
        let tv = self.transform.dot(vector);
        self.inner.apply(&tv)
    }
    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        let xtv = self.inner.apply_transpose(vector);
        self.transform.t().dot(&xtv)
    }
    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        let inner_xtwx = self.inner.diag_xtw_x(weights)?;
        // T^T * (X^T W X) * T
        let tmp = fast_ab(&self.transform.t().to_owned(), &inner_xtwx);
        Ok(fast_ab(&tmp, &self.transform))
    }
}

impl DenseDesignOperator for CoefficientTransformOperator {
    fn to_dense(&self) -> Array2<f64> {
        let x = self.inner.to_dense();
        fast_ab(&x, &self.transform)
    }
    fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        let chunk = self.inner.row_chunk(rows);
        fast_ab(&chunk, &self.transform)
    }
}

impl RowwiseKroneckerOperator {
    pub fn new(cov: DesignMatrix, time_basis: Arc<Array2<f64>>) -> Result<Self, String> {
        let n = cov.nrows();
        if time_basis.nrows() != n {
            return Err(format!(
                "RowwiseKroneckerOperator: cov has {} rows but time_basis has {}",
                n,
                time_basis.nrows()
            ));
        }
        let p_cov = cov.ncols();
        let p_time = time_basis.ncols();
        Ok(Self {
            cov,
            time_basis,
            n,
            p_cov,
            p_time,
        })
    }
}

impl LinearOperator for RowwiseKroneckerOperator {
    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.p_cov * self.p_time
    }

    /// X β where β is reshaped as (p_cov, p_time):
    ///   result[i] = Σⱼ cov[i,j] * Σₜ time[i,t] * β[j*p_time + t]
    ///
    /// Computed via p_time calls to cov.apply() to stay sparse-native:
    ///   For each t: result += time[:,t] ⊙ cov · β[:,t]
    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        let p_cov = self.p_cov;
        let p_time = self.p_time;
        let n = self.n;
        let time = self.time_basis.as_ref();
        let mut out = Array1::<f64>::zeros(n);
        // For each time column t, extract β[:,t] = [β[0*pt+t], β[1*pt+t], ...],
        // compute cov · β[:,t], then weight by time[:,t].
        let mut beta_slice = Array1::<f64>::zeros(p_cov);
        for t in 0..p_time {
            for j in 0..p_cov {
                beta_slice[j] = vector[j * p_time + t];
            }
            let cov_beta_t = self.cov.matrixvectormultiply(&beta_slice);
            for i in 0..n {
                out[i] += cov_beta_t[i] * time[[i, t]];
            }
        }
        out
    }

    /// X' v where the result is (p_cov * p_time):
    ///   result[j*p_time + t] = Σᵢ v[i] * cov[i,j] * time[i,t]
    ///
    /// Computed via p_time calls to cov.apply_transpose() to stay sparse-native:
    ///   For each t: result[:,t] = cov' · (v ⊙ time[:,t])
    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        let p_cov = self.p_cov;
        let p_time = self.p_time;
        let n = self.n;
        let time = self.time_basis.as_ref();
        let mut out = Array1::<f64>::zeros(p_cov * p_time);
        // For each time column t, form w_t = v ⊙ time[:,t], compute cov' · w_t.
        let mut w_t = Array1::<f64>::zeros(n);
        for t in 0..p_time {
            for i in 0..n {
                w_t[i] = vector[i] * time[[i, t]];
            }
            let col_t = self.cov.transpose_vector_multiply(&w_t);
            for j in 0..p_cov {
                out[j * p_time + t] = col_t[j];
            }
        }
        out
    }

    /// X'WX via factored Gram computation.
    ///
    /// (X'WX)[j1*pt+t1, j2*pt+t2]
    ///   = Σᵢ w[i] * cov[i,j1] * time[i,t1] * cov[i,j2] * time[i,t2]
    ///   = Σᵢ (w[i] * cov[i,j1] * cov[i,j2]) * (time[i,t1] * time[i,t2])
    ///
    /// For each (t1, t2) pair, we form the n-vector
    ///   γ_{t1,t2}[i] = w[i] * time[i,t1] * time[i,t2]
    /// and then the (p_cov, p_cov) block is cov' diag(γ_{t1,t2}) cov.
    ///
    /// Cost: O(n * p_time² * p_cov²) vs O(n * (p_cov*p_time)²) for the naive path.
    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        let n = self.n;
        let p_cov = self.p_cov;
        let p_time = self.p_time;
        let p_total = p_cov * p_time;
        if weights.len() != n {
            return Err(format!(
                "RowwiseKroneckerOperator::diag_xtw_x: weights length {} != n {}",
                weights.len(),
                n
            ));
        }
        let mut xtwx = Array2::<f64>::zeros((p_total, p_total));
        let time = self.time_basis.as_ref();

        // For each time-basis pair (t1, t2), the (p_cov, p_cov) block is
        //   cov' diag(γ_{t1,t2}) cov
        // where γ[i] = w[i] * time[i,t1] * time[i,t2].
        // We delegate to cov.compute_xtwx(γ) which stays sparse-native.
        let mut gamma = Array1::<f64>::zeros(n);
        for t1 in 0..p_time {
            for t2 in 0..=t1 {
                for i in 0..n {
                    gamma[i] = weights[i].max(0.0) * time[[i, t1]] * time[[i, t2]];
                }
                let block = self.cov.compute_xtwx(&gamma)?;
                // Scatter block into xtwx for both (t1, t2) and (t2, t1).
                for j1 in 0..p_cov {
                    for j2 in 0..p_cov {
                        xtwx[[j1 * p_time + t1, j2 * p_time + t2]] = block[[j1, j2]];
                        if t1 != t2 {
                            xtwx[[j1 * p_time + t2, j2 * p_time + t1]] = block[[j1, j2]];
                        }
                    }
                }
            }
        }
        Ok(xtwx)
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        let n = self.n;
        let p_cov = self.p_cov;
        let p_time = self.p_time;
        if weights.len() != n {
            return Err(format!(
                "RowwiseKroneckerOperator::diag_gram: weights {} != n {}",
                weights.len(),
                n
            ));
        }
        let time = self.time_basis.as_ref();
        // diag(X'WX)[j*pt+t] = Σᵢ w[i] * cov[i,j]² * time[i,t]²
        // Use cov.diag_gram(w ⊙ time[:,t]²) which stays sparse-native.
        let mut out = Array1::<f64>::zeros(p_cov * p_time);
        let mut gamma = Array1::<f64>::zeros(n);
        for t in 0..p_time {
            for i in 0..n {
                gamma[i] = weights[i].max(0.0) * time[[i, t]] * time[[i, t]];
            }
            let cov_diag = <DesignMatrix as LinearOperator>::diag_gram(&self.cov, &gamma)?;
            for j in 0..p_cov {
                out[j * p_time + t] = cov_diag[j];
            }
        }
        Ok(out)
    }

    fn uses_matrix_free_pcg(&self) -> bool {
        true
    }
}

impl DenseDesignOperator for RowwiseKroneckerOperator {
    fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        let p_total = self.p_cov * self.p_time;
        if middle.nrows() != p_total || middle.ncols() != p_total {
            return Err(format!(
                "RowwiseKroneckerOperator::quadratic_form_diag dimension mismatch: {}x{} vs expected {}x{}",
                middle.nrows(),
                middle.ncols(),
                p_total,
                p_total
            ));
        }
        let mut out = Array1::<f64>::zeros(self.n);
        for start in (0..self.n).step_by(OPERATOR_ROW_CHUNK_SIZE) {
            let end = (start + OPERATOR_ROW_CHUNK_SIZE).min(self.n);
            let chunk = self.row_chunk(start..end);
            let chunk_m = fast_ab(&chunk, middle);
            for local in 0..(end - start) {
                out[start + local] = chunk.row(local).dot(&chunk_m.row(local)).max(0.0);
            }
        }
        Ok(out)
    }

    fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        let p_cov = self.p_cov;
        let p_time = self.p_time;
        let chunk_rows = rows.end - rows.start;
        let cov_chunk = self.cov.row_chunk(rows.clone());
        let time = self.time_basis.as_ref();
        let mut out = Array2::<f64>::zeros((chunk_rows, p_cov * p_time));
        for local in 0..chunk_rows {
            let global = rows.start + local;
            for j in 0..p_cov {
                let cij = cov_chunk[[local, j]];
                if cij == 0.0 {
                    continue;
                }
                for t in 0..p_time {
                    out[[local, j * p_time + t]] = cij * time[[global, t]];
                }
            }
        }
        out
    }

    fn to_dense(&self) -> Array2<f64> {
        let n = self.n;
        let p_cov = self.p_cov;
        let p_time = self.p_time;
        let cov_dense = self.cov.as_dense_cow();
        let time = self.time_basis.as_ref();
        let mut out = Array2::<f64>::zeros((n, p_cov * p_time));
        for i in 0..n {
            for j in 0..p_cov {
                let cij = cov_dense[[i, j]];
                if cij == 0.0 {
                    continue;
                }
                for t in 0..p_time {
                    out[[i, j * p_time + t]] = cij * time[[i, t]];
                }
            }
        }
        out
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
        let mut base = self.inner.diag_xtw_x(weights)?;
        if self.columns.is_empty() {
            return Ok(base);
        }
        let p = base.ncols();
        let w_pos: Array1<f64> = weights.mapv(|w| w.max(0.0));
        let sum_w: f64 = w_pos.sum();
        let cw = self.inner.apply_transpose(&w_pos);

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
        let mut result = self.inner.diag_gram(weights)?;
        if self.columns.is_empty() {
            return Ok(result);
        }
        let w_pos: Array1<f64> = weights.mapv(|w| w.max(0.0));
        let sum_w: f64 = w_pos.sum();
        let cw = self.inner.apply_transpose(&w_pos);
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
    /// X_c'(w⊙y) = a⊙(X'(w⊙y)) - d·Σ(w⊙y)
    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
        let mut result = self.inner.compute_xtwy(weights, y)?;
        if self.columns.is_empty() {
            return Ok(result);
        }
        let sum_wy: f64 = weights
            .iter()
            .zip(y.iter())
            .map(|(&w, &yi)| w.max(0.0) * yi)
            .sum();
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

    fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        let mut chunk = self.inner.row_chunk(rows);
        for &(j, mean, scale) in &self.columns {
            chunk.column_mut(j).mapv_inplace(|v| (v - mean) / scale);
        }
        chunk
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
/// Dense matrices are wrapped in Arc for O(1) cloning — at biobank scale
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

/// A unified representation of a symmetric matrix, typically an assembled Hessian.
#[derive(Clone, Debug)]
pub enum SymmetricMatrix {
    Dense(Array2<f64>),
    Sparse(faer::sparse::SparseColMat<usize, f64>),
}

impl SymmetricMatrix {
    pub fn as_dense(&self) -> Option<&Array2<f64>> {
        match self {
            Self::Dense(mat) => Some(mat),
            Self::Sparse(_) => None,
        }
    }

    pub fn as_sparse(&self) -> Option<&faer::sparse::SparseColMat<usize, f64>> {
        match self {
            Self::Sparse(mat) => Some(mat),
            Self::Dense(_) => None,
        }
    }

    pub fn to_dense(&self) -> Array2<f64> {
        match self {
            Self::Dense(mat) => mat.clone(),
            Self::Sparse(mat) => {
                let mut out = Array2::<f64>::zeros((mat.nrows(), mat.ncols()));
                let (symbolic, values) = mat.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                for col in 0..mat.ncols() {
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    for idx in start..end {
                        let row = row_idx[idx];
                        let value = values[idx];
                        out[[row, col]] += value;
                        if row != col {
                            out[[col, row]] += value;
                        }
                    }
                }
                out
            }
        }
    }

    pub fn factorize(&self) -> Result<Box<dyn FactorizedSystem>, String> {
        match self {
            Self::Dense(mat) => {
                let factor = crate::linalg::utils::StableSolver::new("unnamed")
                    .factorize(mat)
                    .map_err(|e| format!("Dense SymmetricMatrix factorization failed: {e:?}"))?;
                Ok(Box::new(factor))
            }
            Self::Sparse(mat) => {
                let factor = crate::linalg::sparse_exact::factorize_sparse_spd(mat)
                    .map_err(|e| format!("Sparse SymmetricMatrix factorization failed: {e:?}"))?;
                Ok(Box::new(factor))
            }
        }
    }

    pub fn add(&self, other: &SymmetricMatrix) -> Result<Self, String> {
        if self.nrows() != other.nrows() || self.ncols() != other.ncols() {
            return Err(format!(
                "SymmetricMatrix::add shape mismatch: lhs {}x{}, rhs {}x{}",
                self.nrows(),
                self.ncols(),
                other.nrows(),
                other.ncols()
            ));
        }
        match (self, other) {
            (Self::Dense(a), Self::Dense(b)) => Ok(Self::Dense(a + b)),
            (Self::Dense(a), Self::Sparse(_)) => {
                let b_dense = other.to_dense();
                Ok(Self::Dense(a + &b_dense))
            }
            (Self::Sparse(_), Self::Dense(b)) => {
                let a_dense = self.to_dense();
                Ok(Self::Dense(&a_dense + b))
            }
            (Self::Sparse(a), Self::Sparse(b)) => {
                Ok(Self::Sparse(add_sparse_symmetric_upper(a, b)?))
            }
        }
    }

    pub(crate) fn add_dense(&self, other: &Array2<f64>) -> Result<Self, String> {
        if self.nrows() != other.nrows() || self.ncols() != other.ncols() {
            return Err(format!(
                "SymmetricMatrix::add_dense shape mismatch: lhs {}x{}, rhs {}x{}",
                self.nrows(),
                self.ncols(),
                other.nrows(),
                other.ncols()
            ));
        }
        match self {
            Self::Dense(mat) => {
                let mut out = mat.clone();
                out += other;
                Ok(Self::Dense(out))
            }
            Self::Sparse(mat) => {
                let other_sparse =
                    crate::linalg::sparse_exact::dense_to_sparse_symmetric_upper(other, 0.0)
                        .map_err(|e| format!("SymmetricMatrix::add_dense failed: {e}"))?;
                Ok(Self::Sparse(add_sparse_symmetric_upper(
                    mat,
                    &other_sparse,
                )?))
            }
        }
    }

    pub fn addridge(&self, ridge: f64) -> Result<Self, String> {
        if ridge == 0.0 {
            return Ok(self.clone());
        }
        match self {
            Self::Dense(mat) => {
                let mut out = mat.clone();
                for i in 0..out.nrows() {
                    out[[i, i]] += ridge;
                }
                Ok(Self::Dense(out))
            }
            Self::Sparse(mat) => {
                let n = mat.nrows();
                let mut trip = Vec::with_capacity(n);
                for i in 0..n {
                    trip.push(Triplet::new(i, i, ridge));
                }
                let diagonal = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &trip)
                    .map_err(|_| {
                        "SymmetricMatrix::addridge failed to assemble sparse diagonal".to_string()
                    })?;
                Ok(Self::Sparse(add_sparse_symmetric_upper(mat, &diagonal)?))
            }
        }
    }

    pub fn nrows(&self) -> usize {
        match self {
            Self::Dense(m) => m.nrows(),
            Self::Sparse(m) => m.nrows(),
        }
    }

    pub fn ncols(&self) -> usize {
        match self {
            Self::Dense(m) => m.ncols(),
            Self::Sparse(m) => m.ncols(),
        }
    }

    pub fn dot(&self, rhs: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(mat) => mat.dot(rhs),
            Self::Sparse(mat) => {
                let mut out = Array1::<f64>::zeros(mat.nrows());
                let (symbolic, values) = mat.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                for col in 0..mat.ncols() {
                    let rhs_j = rhs[col];
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    for idx in start..end {
                        let row = row_idx[idx];
                        let value = values[idx];
                        out[row] += value * rhs_j;
                        if row != col {
                            out[col] += value * rhs[row];
                        }
                    }
                }
                out
            }
        }
    }

    /// Maximum absolute value on the diagonal.
    pub fn max_abs_diag(&self) -> f64 {
        match self {
            Self::Dense(mat) => {
                let n = mat.nrows().min(mat.ncols());
                (0..n).map(|i| mat[[i, i]].abs()).fold(0.0_f64, f64::max)
            }
            Self::Sparse(mat) => {
                let (symbolic, values) = mat.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                let mut max_val = 0.0_f64;
                for col in 0..mat.ncols() {
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    for idx in start..end {
                        if row_idx[idx] == col {
                            max_val = max_val.max(values[idx].abs());
                        }
                    }
                }
                max_val
            }
        }
    }

    /// Multiply on the right by a dense matrix: self * rhs.
    /// Returns a dense Array2.
    pub fn dot_matrix(&self, rhs: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::Dense(mat) => mat.dot(rhs),
            Self::Sparse(mat) => {
                let n = mat.nrows();
                let k = rhs.ncols();
                let mut out = Array2::<f64>::zeros((n, k));
                let (symbolic, values) = mat.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                for col in 0..mat.ncols() {
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    for idx in start..end {
                        let row = row_idx[idx];
                        let value = values[idx];
                        for c in 0..k {
                            out[[row, c]] += value * rhs[[col, c]];
                            if row != col {
                                out[[col, c]] += value * rhs[[row, c]];
                            }
                        }
                    }
                }
                out
            }
        }
    }

    /// Left-multiply by a dense matrix: lhs * self.
    /// Returns a dense Array2.
    pub fn left_dot_matrix(&self, lhs: &Array2<f64>) -> Array2<f64> {
        // (lhs * S)^T = S^T * lhs^T = S * lhs^T  (S is symmetric)
        // So lhs * S = (S * lhs^T)^T
        let lhs_t = lhs.t().to_owned();
        let result_t = self.dot_matrix(&lhs_t);
        result_t.t().to_owned()
    }
}

pub fn xt_diag_x_symmetric(
    design: &DesignMatrix,
    diag: &Array1<f64>,
) -> Result<SymmetricMatrix, String> {
    if design.nrows() != diag.len() {
        return Err(format!(
            "xt_diag_x_symmetric row mismatch: design has {} rows but diag has {} entries",
            design.nrows(),
            diag.len()
        ));
    }
    match design {
        DesignMatrix::Dense(x) => Ok(SymmetricMatrix::Dense(x.diag_xtw_x(diag)?)),
        DesignMatrix::Sparse(xs) => {
            let csr = xs
                .to_csr_arc()
                .ok_or_else(|| "xt_diag_x_symmetric: failed to obtain CSR view".to_string())?;
            let sym = csr.symbolic();
            let row_ptr = sym.row_ptr();
            let col_idx = sym.col_idx();
            let vals = csr.val();
            let mut upper = Vec::new();
            for i in 0..xs.nrows() {
                let wi = diag[i];
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
                        let value = wi * xa * xb;
                        let (row, col) = if a <= b { (a, b) } else { (b, a) };
                        upper.push(Triplet::new(row, col, value));
                    }
                }
            }
            let sparse = SparseColMat::try_new_from_triplets(xs.ncols(), xs.ncols(), &upper)
                .map_err(|_| {
                    "xt_diag_x_symmetric: failed to assemble sparse symmetric matrix".to_string()
                })?;
            Ok(SymmetricMatrix::Sparse(sparse))
        }
    }
}

fn add_sparse_symmetric_upper(
    lhs: &SparseColMat<usize, f64>,
    rhs: &SparseColMat<usize, f64>,
) -> Result<SparseColMat<usize, f64>, String> {
    if lhs.nrows() != rhs.nrows() || lhs.ncols() != rhs.ncols() {
        return Err(format!(
            "add_sparse_symmetric_upper shape mismatch: lhs {}x{}, rhs {}x{}",
            lhs.nrows(),
            lhs.ncols(),
            rhs.nrows(),
            rhs.ncols()
        ));
    }
    let mut upper = BTreeMap::<(usize, usize), f64>::new();
    for matrix in [lhs, rhs] {
        let (symbolic, values) = matrix.parts();
        let col_ptr = symbolic.col_ptr();
        let row_idx = symbolic.row_idx();
        for col in 0..matrix.ncols() {
            for idx in col_ptr[col]..col_ptr[col + 1] {
                let row = row_idx[idx];
                let key = if row <= col { (row, col) } else { (col, row) };
                *upper.entry(key).or_insert(0.0) += values[idx];
            }
        }
    }
    let triplets: Vec<_> = upper
        .into_iter()
        .filter_map(|((row, col), value)| (value != 0.0).then_some(Triplet::new(row, col, value)))
        .collect();
    SparseColMat::try_new_from_triplets(lhs.nrows(), lhs.ncols(), &triplets)
        .map_err(|_| "add_sparse_symmetric_upper failed to assemble CSC".to_string())
}

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
    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        let xtwx = self.diag_xtw_x(weights)?;
        Ok(Array1::from_iter((0..self.ncols()).map(|j| xtwx[[j, j]])))
    }
    fn apply_weighted_normal(
        &self,
        weights: &Array1<f64>,
        vector: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge: f64,
    ) -> Array1<f64> {
        let xv = self.apply(vector);
        let mut weighted_xv = xv;
        for i in 0..weighted_xv.len() {
            weighted_xv[i] *= weights[i].max(0.0);
        }
        let mut out = self.apply_transpose(&weighted_xv);
        if let Some(pen) = penalty {
            out += &pen.dot(vector);
        }
        if ridge > 0.0 {
            out += &vector.mapv(|x| ridge * x);
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
        for retry in 0..8 {
            let ridge = if baseridge > 0.0 {
                baseridge * 10f64.powi(retry as i32)
            } else {
                0.0
            };
            let normal_op = PenalizedWeightedNormalOperator {
                operator: self,
                weights,
                penalty,
                ridge,
            };
            let preconditioner = normal_op.jacobi_preconditioner()?;
            let solved = crate::linalg::utils::solve_spd_pcg_with_info(
                |v| normal_op.apply(v),
                rhs,
                &preconditioner,
                MATRIX_FREE_PCG_REL_TOL,
                MATRIX_FREE_PCG_MAX_ITER.max(4 * self.ncols()),
            );
            if let Some((solution, info)) = solved
                && solution.iter().all(|v| v.is_finite())
            {
                return Ok((solution, info));
            }
        }
        Err("matrix-free PCG failed after ridge retries".to_string())
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
        let factor = crate::linalg::utils::StableSolver::new("linear operator system")
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
        self.solve_systemwith_policy(
            weights,
            rhs,
            penalty,
            1e-15,
            RidgePolicy::explicit_stabilization_pospart(),
        )
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
        let baseridge = if ridge_policy.include_laplacehessian {
            ridge_floor.max(1e-15)
        } else {
            0.0
        };
        // Try matrix-free PCG first to avoid assembling the dense p×p normal matrix.
        if self.uses_matrix_free_pcg() && self.ncols() >= MATRIX_FREE_PCG_MIN_P {
            if let Ok(solution) =
                self.solve_system_matrix_free_pcg_try(weights, rhs, penalty, baseridge)
            {
                return Ok(solution);
            }
        }
        // Fallback: assemble dense system and solve via Cholesky with ridge retries.
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
        crate::linalg::utils::StableSolver::new("linear operator system")
            .solvevectorwithridge_retries(&system, rhs, baseridge)
            .ok_or_else(|| "solve_systemwith_policy failed after ridge retries".to_string())
    }

    // Backward-compatible aliases.
    fn matvec(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.apply(vector)
    }
    fn matvec_trans(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.apply_transpose(vector)
    }
    fn compute_xtwx(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.diag_xtw_x(weights)
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
        weights: &Array1<f64>,
        vector: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge: f64,
    ) -> Array1<f64> {
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
                        let wi = weights[i].max(0.0);
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
                        weighted_xv[i] *= weights[i].max(0.0);
                    }
                    self.apply_transpose(&weighted_xv)
                };
                if let Some(pen) = penalty {
                    out += &pen.dot(vector);
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
        if weights.len() != self.nrows() {
            return Err(format!(
                "compute_xtwx dimension mismatch: weights length {} != nrows {}",
                weights.len(),
                self.nrows()
            ));
        }
        let p = self.ncols();
        let mut xtwx = Array2::<f64>::zeros((p, p));
        match self {
            Self::Dense(x) => x.diag_xtw_x(weights),
            Self::Sparse(xs) => {
                let csr = xs
                    .as_ref()
                    .to_row_major()
                    .map_err(|_| "failed to obtain CSR view in compute_xtwx".to_string())?;
                let sym = csr.symbolic();
                let row_ptr = sym.row_ptr();
                let col_idx = sym.col_idx();
                let vals = csr.val();
                for i in 0..self.nrows() {
                    let wi = weights[i].max(0.0);
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
                            let v = wi * xa * xb;
                            xtwx[[a, b]] += v;
                            if a != b {
                                xtwx[[b, a]] += v;
                            }
                        }
                    }
                }
                Ok(xtwx)
            }
        }
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        if weights.len() != self.nrows() {
            return Err(format!(
                "diag_gram dimension mismatch: weights length {} != nrows {}",
                weights.len(),
                self.nrows()
            ));
        }
        let p = self.ncols();
        let mut diag = Array1::<f64>::zeros(p);
        match self {
            Self::Dense(x) => x.diag_gram(weights),
            Self::Sparse(xs) => {
                let csr = xs
                    .as_ref()
                    .to_row_major()
                    .map_err(|_| "failed to obtain CSR view in diag_gram".to_string())?;
                let sym = csr.symbolic();
                let row_ptr = sym.row_ptr();
                let col_idx = sym.col_idx();
                let vals = csr.val();
                for i in 0..self.nrows() {
                    let wi = weights[i].max(0.0);
                    if wi == 0.0 {
                        continue;
                    }
                    for idx in row_ptr[i]..row_ptr[i + 1] {
                        let j = col_idx[idx];
                        let xij = vals[idx];
                        diag[j] += wi * xij * xij;
                    }
                }
                Ok(diag)
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
                let factor = crate::linalg::sparse_exact::factorize_sparse_spd(&system)
                    .map_err(|e| format!("factorize_system failed: {e:?}"))?;
                Ok(Box::new(factor))
            }
        }
    }
}

impl DenseDesignOperator for DesignMatrix {
    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
        if weights.len() != self.nrows() || y.len() != self.nrows() {
            return Err(format!(
                "compute_xtwy dimension mismatch: weights={}, y={}, nrows={}",
                weights.len(),
                y.len(),
                self.nrows()
            ));
        }
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
                    let scaled = weights[i].max(0.0) * y[i];
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

    fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        DesignMatrix::row_chunk(self, rows)
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
        dense_matvec(self.base, v)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        let mut out = dense_transpose_matvec(self.base, vector);
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
                "compute_xtwx dimension mismatch: weights length {} != nrows {}",
                weights.len(),
                self.nrows()
            ));
        }
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

impl LinearOperator for DenseRowScaledView<'_> {
    fn nrows(&self) -> usize {
        self.matrix.nrows()
    }

    fn ncols(&self) -> usize {
        self.matrix.ncols()
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        let mut out = dense_matvec(self.matrix, vector);
        for i in 0..out.len() {
            out[i] *= self.scale[i];
        }
        out
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        let scaled = Array1::from_shape_fn(vector.len(), |i| vector[i] * self.scale[i]);
        dense_transpose_matvec(self.matrix, &scaled)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        if weights.len() != self.nrows() {
            return Err(format!(
                "compute_xtwx dimension mismatch: weights length {} != nrows {}",
                weights.len(),
                self.nrows()
            ));
        }
        let combined = Array1::from_shape_fn(weights.len(), |i| {
            weights[i] * self.scale[i] * self.scale[i]
        });
        Ok(fast_xt_diag_x(self.matrix, &combined))
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        Ok(self.diag_xtw_x(weights)?.diag().to_owned())
    }
}

impl DenseRowScaledView<'_> {
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
        Ok(dense_transpose_weighted_response(
            self.matrix,
            weights,
            y,
            Some(self.scale),
        ))
    }

    pub fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        if middle.nrows() != self.ncols() || middle.ncols() != self.ncols() {
            return Err(format!(
                "quadratic_form_diag dimension mismatch: matrix is {}x{}, expected {}x{}",
                middle.nrows(),
                middle.ncols(),
                self.ncols(),
                self.ncols()
            ));
        }
        let xm = fast_ab(self.matrix, middle);
        let mut out = Array1::<f64>::zeros(self.nrows());
        for i in 0..self.matrix.nrows() {
            let s2 = self.scale[i] * self.scale[i];
            out[i] = (self.matrix.row(i).dot(&xm.row(i)) * s2).max(0.0);
        }
        Ok(out)
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
        dense_matvec(
            self.local,
            &vector
                .slice(ndarray::s![self.global_range.clone()])
                .to_owned(),
        )
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.total_cols);
        out.slice_mut(ndarray::s![self.global_range.clone()])
            .assign(&dense_transpose_matvec(self.local, vector));
        out
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        if weights.len() != self.nrows() {
            return Err(format!(
                "compute_xtwx dimension mismatch: weights length {} != nrows {}",
                weights.len(),
                self.nrows()
            ));
        }
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

/// Streaming chunked BLAS computation of X^T * diag(W) * X.
///
/// Processes rows in cache-friendly chunks, scaling each chunk by sqrt(w)
/// and accumulating via BLAS matmul.  Peak intermediate allocation is
/// chunk_size × p × 8 bytes (~8 MB) instead of n × p × 8 bytes.
fn streaming_blas_xt_diag_x(x: &Array2<f64>, weights: &Array1<f64>, out: &mut Array2<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p == 0 {
        return;
    }

    // Target ~8MB working set per chunk (matches faer_ndarray streaming path).
    // Previous 2MB / MAX_ROWS=2048 was 64x smaller than needed and caused
    // excessive loop overhead on large n.
    const TARGET_BYTES: usize = 8 * 1024 * 1024;
    const MIN_ROWS: usize = 512;
    const MAX_ROWS: usize = 131_072;
    let chunk_rows = (TARGET_BYTES / (p * 8)).max(MIN_ROWS).min(MAX_ROWS).min(n);

    let par = faer::get_global_parallelism();
    let mut weighted_chunk = Array2::<f64>::zeros((chunk_rows, p).f());
    let mut out_view = array2_to_matmut(out);

    for start in (0..n).step_by(chunk_rows) {
        let rows = (n - start).min(chunk_rows);
        {
            let mut chunk = weighted_chunk.slice_mut(s![0..rows, ..]);
            for local in 0..rows {
                let src = start + local;
                let sqrtw = weights[src].max(0.0).sqrt();
                for col in 0..p {
                    chunk[[local, col]] = x[[src, col]] * sqrtw;
                }
            }
        }
        let chunk_slice = weighted_chunk.slice(s![0..rows, ..]);
        let chunk_view = FaerArrayView::new(&chunk_slice);
        matmul(
            out_view.as_mut(),
            Accum::Add,
            chunk_view.as_ref().transpose(),
            chunk_view.as_ref(),
            1.0,
            par,
        );
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
        let factor = crate::linalg::utils::StableSolver::new("linear operator system")
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
        let wi = weights[i].max(0.0);
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
    Ok(SparseColMat::try_new_from_triplets(p, p, &triplets)
        .map_err(|_| "failed to build sparse penalized system".to_string())?)
}

impl DesignMatrix {
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
    pub fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        match self {
            Self::Dense(matrix) => matrix.row_chunk(rows),
            Self::Sparse(matrix) => {
                let csr = matrix.to_csr_arc().unwrap_or_else(|| {
                    panic!("DesignMatrix::row_chunk: failed to obtain CSR view")
                });
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
                out
            }
        }
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
                    matrix.row_chunk(row..row + 1).row(0).dot(&beta)
                }
            }
            Self::Sparse(matrix) => {
                let csr = matrix
                    .to_csr_arc()
                    .unwrap_or_else(|| panic!("DesignMatrix::dot_row: failed to obtain CSR view"));
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
        if out.len() != self.ncols() {
            return Err(format!(
                "DesignMatrix::axpy_row_into length mismatch: out={}, ncols={}",
                out.len(),
                self.ncols()
            ));
        }
        if alpha == 0.0 {
            return Ok(());
        }
        match self {
            Self::Dense(matrix) => {
                if let Some(dense) = matrix.as_dense_ref() {
                    for (dst, &value) in out.iter_mut().zip(dense.row(row).iter()) {
                        *dst += alpha * value;
                    }
                } else {
                    let chunk = matrix.row_chunk(row..row + 1);
                    for (dst, &value) in out.iter_mut().zip(chunk.row(0).iter()) {
                        *dst += alpha * value;
                    }
                }
            }
            Self::Sparse(matrix) => {
                let csr = matrix.to_csr_arc().unwrap_or_else(|| {
                    panic!("DesignMatrix::axpy_row_into: failed to obtain CSR view")
                });
                let sym = csr.symbolic();
                let row_ptr = sym.row_ptr();
                let col_idx = sym.col_idx();
                let vals = csr.val();
                for ptr in row_ptr[row]..row_ptr[row + 1] {
                    out[col_idx[ptr]] += alpha * vals[ptr];
                }
            }
        }
        Ok(())
    }

    /// Add `alpha * X[row, :]^2` elementwise into `out` without allocating a
    /// standalone row buffer.
    pub fn squared_axpy_row_into(
        &self,
        row: usize,
        alpha: f64,
        out: &mut ArrayViewMut1<'_, f64>,
    ) -> Result<(), String> {
        if out.len() != self.ncols() {
            return Err(format!(
                "DesignMatrix::squared_axpy_row_into length mismatch: out={}, ncols={}",
                out.len(),
                self.ncols()
            ));
        }
        if alpha == 0.0 {
            return Ok(());
        }
        match self {
            Self::Dense(matrix) => {
                if let Some(dense) = matrix.as_dense_ref() {
                    for (dst, &value) in out.iter_mut().zip(dense.row(row).iter()) {
                        *dst += alpha * value * value;
                    }
                } else {
                    let chunk = matrix.row_chunk(row..row + 1);
                    for (dst, &value) in out.iter_mut().zip(chunk.row(0).iter()) {
                        *dst += alpha * value * value;
                    }
                }
            }
            Self::Sparse(matrix) => {
                let csr = matrix.to_csr_arc().unwrap_or_else(|| {
                    panic!("DesignMatrix::squared_axpy_row_into: failed to obtain CSR view")
                });
                let sym = csr.symbolic();
                let row_ptr = sym.row_ptr();
                let col_idx = sym.col_idx();
                let vals = csr.val();
                for ptr in row_ptr[row]..row_ptr[row + 1] {
                    let value = vals[ptr];
                    out[col_idx[ptr]] += alpha * value * value;
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
                    let chunk = matrix.row_chunk(row..row + 1);
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
                let csr = matrix.to_csr_arc().unwrap_or_else(|| {
                    panic!("DesignMatrix::syr_row_into: failed to obtain CSR view")
                });
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
                // Zero-copy borrow for materialized matrices; only Arc-clone
                // (not data-clone) for lazy operators.
                let lhs_owned;
                let lhs_ref: &Array2<f64> = match lhs.as_dense_ref() {
                    Some(r) => r,
                    None => {
                        lhs_owned = lhs.to_dense_arc();
                        lhs_owned.as_ref()
                    }
                };
                let rhs_owned;
                let rhs_ref: &Array2<f64> = match rhs.as_dense_ref() {
                    Some(r) => r,
                    None => {
                        rhs_owned = rhs.to_dense_arc();
                        rhs_owned.as_ref()
                    }
                };
                let x = lhs_ref.row(row);
                let y = rhs_ref.row(row);
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
                let lhs_csr = lhs
                    .to_csr_arc()
                    .unwrap_or_else(|| panic!("row_outer_into: failed to obtain lhs CSR view"));
                let rhs_csr = rhs
                    .to_csr_arc()
                    .unwrap_or_else(|| panic!("row_outer_into: failed to obtain rhs CSR view"));
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
                let x = self.row_chunk(row..row + 1);
                let x_row = x.row(0);
                let y = other.row_chunk(row..row + 1);
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

    /// Element access: returns the value at row `i`, column `j`.
    ///
    /// For dense matrices this is O(1). For sparse matrices, this converts to
    /// a dense cache first and then indexes, so callers doing per-row sweeps
    /// should prefer `to_dense()` for bulk access.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        match self {
            Self::Dense(matrix) => match matrix.as_dense_ref() {
                Some(dense) => dense[[i, j]],
                None => matrix.to_dense()[[i, j]],
            },
            Self::Sparse(sp) => {
                let dense = sp
                    .try_to_dense_arc("DesignMatrix::get")
                    .unwrap_or_else(|msg| panic!("{msg}"));
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
        match self {
            Self::Dense(m) => {
                if let Some(dense) = m.as_dense_ref() {
                    dense.column(j).to_owned()
                } else {
                    let mut e_j = Array1::zeros(m.ncols());
                    e_j[j] = 1.0;
                    m.apply(&e_j)
                }
            }
            Self::Sparse(sp) => {
                let n = sp.nrows();
                let mut col = Array1::zeros(n);
                let (symbolic, values) = sp.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                let start = col_ptr[j];
                let end = col_ptr[j + 1];
                for idx in start..end {
                    col[row_idx[idx]] = values[idx];
                }
                col
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

    /// Zero-copy borrow when `Dense`, materialized conversion when `Sparse`.
    ///
    /// This avoids the unconditional clone that `to_dense()` performs on dense
    /// matrices.  Callers that only need a `&Array2<f64>` should use this and
    /// then call `Cow::as_ref()` or `&*cow`.
    pub fn as_dense_cow(&self) -> Cow<'_, Array2<f64>> {
        match self {
            Self::Dense(matrix) => match matrix.as_dense_ref() {
                Some(dense) => Cow::Borrowed(dense),
                None => Cow::Owned(matrix.to_dense()),
            },
            Self::Sparse(matrix) => Cow::Owned(
                matrix
                    .try_to_dense_arc("DesignMatrix::as_dense_cow")
                    .unwrap_or_else(|msg| panic!("{msg}"))
                    .as_ref()
                    .clone(),
            ),
        }
    }

    pub fn to_dense(&self) -> Array2<f64> {
        match self {
            Self::Dense(matrix) => matrix.to_dense(),
            Self::Sparse(matrix) => matrix
                .try_to_dense_arc("DesignMatrix::to_dense")
                .unwrap_or_else(|msg| panic!("{msg}"))
                .as_ref()
                .clone(),
        }
    }

    pub fn to_dense_arc(&self) -> Arc<Array2<f64>> {
        match self {
            Self::Dense(matrix) => matrix.to_dense_arc(),
            Self::Sparse(matrix) => matrix
                .try_to_dense_arc("DesignMatrix::to_dense_arc")
                .unwrap_or_else(|msg| panic!("{msg}")),
        }
    }

    pub fn try_to_dense_arc(&self, context: &str) -> Result<Arc<Array2<f64>>, String> {
        match self {
            Self::Dense(matrix) => Ok(matrix.to_dense_arc()),
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

    pub fn dot(&self, vector: &Array1<f64>) -> Array1<f64> {
        <Self as LinearOperator>::apply(self, vector)
    }

    pub fn matrixvectormultiply(&self, vector: &Array1<f64>) -> Array1<f64> {
        <Self as LinearOperator>::apply(self, vector)
    }

    pub fn transpose_vector_multiply(&self, vector: &Array1<f64>) -> Array1<f64> {
        <Self as LinearOperator>::apply_transpose(self, vector)
    }

    pub fn compute_xtwx(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        <Self as LinearOperator>::diag_xtw_x(self, weights)
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
    ) -> Array1<f64> {
        <Self as LinearOperator>::apply_weighted_normal(self, weights, vector, penalty, ridge)
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
            ridge_floor.max(1e-15),
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
            ridge_floor.max(1e-15),
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

impl From<&Array2<f64>> for DesignMatrix {
    fn from(value: &Array2<f64>) -> Self {
        Self::Dense(DenseDesignMatrix::from(value.clone()))
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

#[cfg(test)]
mod tests {
    use super::{
        DesignMatrix, SparseDesignMatrix, dense_matvec, dense_transpose_matvec,
        dense_transpose_weighted_response,
    };
    use crate::linalg::matrix::LinearOperator;
    use crate::linalg::utils::{PcgSolveInfo, StableSolver};
    use crate::types::RidgePolicy;
    use faer::sparse::{SparseColMat, SymbolicSparseColMat, Triplet};
    use ndarray::{Array1, Array2, Axis, array};
    use std::sync::Arc;

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
        StableSolver::new("matrix-free pcg exact reference")
            .solvevectorwithridge_retries(&h, rhs, 0.0)
            .expect("exact reference solve")
    }

    #[test]
    fn dense_matvec_matches_ndarray_dot() {
        let x = array![[1.0, 2.0, -1.0], [0.5, -3.0, 4.0], [2.0, 0.0, 1.5]];
        let v = array![0.25, -1.0, 2.0];
        let expected = x.dot(&v);
        let got = dense_matvec(&x, &v);
        for i in 0..expected.len() {
            assert!((expected[i] - got[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn dense_transpose_matvec_matches_ndarray_dot() {
        let x = array![[1.0, 2.0, -1.0], [0.5, -3.0, 4.0], [2.0, 0.0, 1.5]];
        let v = array![0.25, -1.0, 2.0];
        let expected = x.t().dot(&v);
        let got = dense_transpose_matvec(&x, &v);
        for i in 0..expected.len() {
            assert!((expected[i] - got[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn sparse_to_dense_accumulates_duplicate_entries() {
        // Build a non-canonical CSC with duplicate row index in the same column.
        // This can happen if a caller bypasses canonical constructors.
        let symbolic = unsafe {
            SymbolicSparseColMat::new_unchecked(
                3,
                2,
                vec![0_usize, 2, 3],
                None,
                vec![1_usize, 1, 0],
            )
        };
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
    fn huge_sparse_densification_is_rejected_before_allocation() {
        let sparse = SparseColMat::try_new_from_triplets(500_000, 10_000, &[])
            .expect("empty sparse matrix should build");
        let design = SparseDesignMatrix::new(sparse);
        let err = design
            .try_to_dense_arc("matrix test")
            .expect_err("huge sparse densification should be rejected");
        assert!(err.contains("refusing to densify sparse design"));
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
            .dot(&(x.clone() * &weights.view().insert_axis(Axis(1))));
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
                RidgePolicy::explicit_stabilization_pospart(),
            )
            .expect("policy solve");
        for i in 0..p {
            assert!(
                (explicit[i] - policy[i]).abs() < 1e-6,
                "policy mismatch at {i}: explicit={} policy={}",
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
            let wy = Array1::from_shape_fn(n, |i| y[i] * w[i].max(0.0));
            dense_transpose_matvec(&x, &wy)
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
}

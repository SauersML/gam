use crate::faer_ndarray::{
    FaerArrayView, array2_to_matmut, fast_ab, fast_atb, fast_atv, fast_av, fast_xt_diag_x,
};
use crate::types::RidgePolicy;
use faer::Accum;
use faer::linalg::matmul::matmul;
use faer::sparse::{SparseColMat, SparseRowMat, Triplet};
use ndarray::{Array1, Array2, ArrayView2, ShapeBuilder, s};
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

/// Trait for implicit design matrix operators that avoid materialization.
///
/// Implement this trait for structured designs (multi-channel, rowwise-Kronecker,
/// etc.) that can perform matvecs and Gram-matrix assembly without forming the
/// full dense matrix.  Wrap implementations in `DesignMatrix::Operator(Arc<..>)`
/// to integrate them with the rest of the codebase.
pub trait DesignOperator: Send + Sync {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn apply(&self, vector: &Array1<f64>) -> Array1<f64>;
    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64>;
    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String>;

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        // Default: extract diagonal of X'WX.
        let xtwx = self.diag_xtw_x(weights)?;
        Ok(Array1::from_iter((0..self.ncols()).map(|j| xtwx[[j, j]])))
    }

    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
        // Default: X'(w ⊙ y) via apply_transpose.
        let n = self.nrows();
        if weights.len() != n || y.len() != n {
            return Err(format!(
                "DesignOperator::compute_xtwy dimension mismatch: weights={}, y={}, nrows={}",
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
        // Default: diag(X M X') computed row-by-row via matvecs.
        if middle.nrows() != self.ncols() || middle.ncols() != self.ncols() {
            return Err(format!(
                "DesignOperator::quadratic_form_diag dimension mismatch: {}x{} vs expected {}x{}",
                middle.nrows(),
                middle.ncols(),
                self.ncols(),
                self.ncols()
            ));
        }
        // Fallback via materialization.  Operator impls should override for efficiency.
        let x = self.to_dense();
        let xm = fast_ab(&x, middle);
        let mut out = Array1::<f64>::zeros(self.nrows());
        for i in 0..self.nrows() {
            out[i] = x.row(i).dot(&xm.row(i)).max(0.0);
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
        false
    }

    /// Extract a dense row chunk without materializing the full matrix.
    ///
    /// Returns a `(rows.len(), ncols())` dense matrix containing only the
    /// requested rows.  Concrete operators should override this with O(chunk)
    /// implementations; the default falls back through `to_dense()`.
    fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        self.to_dense().slice(s![rows, ..]).to_owned()
    }

    /// Materialize the full dense matrix.  Operators that exist precisely to
    /// avoid materialization should still support this for fallback paths,
    /// diagnostics, and prediction.
    fn to_dense(&self) -> Array2<f64>;
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

impl DesignOperator for ReparamOperator {
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

    fn to_dense(&self) -> Array2<f64> {
        // Fallback materialization for consumers that truly need dense access.
        let x_dense = self.x_original.to_dense();
        fast_ab(&x_dense, &self.qs)
    }

    fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        // Materialize only the requested rows: X[rows, :] · Qs
        match &self.x_original {
            DesignMatrix::Dense(x) => {
                let chunk = x.slice(s![rows, ..]);
                fast_ab(&chunk.to_owned(), &self.qs)
            }
            DesignMatrix::Sparse(sdm) => {
                let x_dense = sdm.to_dense_arc();
                let chunk = x_dense.slice(s![rows, ..]);
                fast_ab(&chunk.to_owned(), &self.qs)
            }
            DesignMatrix::Operator(op) => {
                let chunk = op.row_chunk(rows);
                fast_ab(&chunk, &self.qs)
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

impl DesignOperator for RandomEffectOperator {
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

    fn uses_matrix_free_pcg(&self) -> bool {
        true
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
    Dense(Arc<Array2<f64>>),
    RandomEffect(Arc<RandomEffectOperator>),
    /// Implicit all-ones intercept column: n rows, 1 column, zero storage.
    Intercept(usize),
    /// Generic operator block (tensor products, multi-channel, etc.).
    Operator(Arc<dyn DesignOperator>),
}

impl DesignBlock {
    fn nrows(&self) -> usize {
        match self {
            Self::Dense(d) => d.nrows(),
            Self::RandomEffect(op) => op.nrows(),
            Self::Intercept(n) => *n,
            Self::Operator(op) => op.nrows(),
        }
    }

    fn ncols(&self) -> usize {
        match self {
            Self::Dense(d) => d.ncols(),
            Self::RandomEffect(op) => op.ncols(),
            Self::Intercept(_) => 1,
            Self::Operator(op) => op.ncols(),
        }
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(d) => dense_matvec(d, vector),
            Self::RandomEffect(op) => op.apply(vector),
            Self::Intercept(n) => Array1::from_elem(*n, vector[0]),
            Self::Operator(op) => op.apply(vector),
        }
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(d) => dense_transpose_matvec(d, vector),
            Self::RandomEffect(op) => op.apply_transpose(vector),
            Self::Intercept(_) => {
                let sum: f64 = vector.iter().sum();
                Array1::from_vec(vec![sum])
            }
            Self::Operator(op) => op.apply_transpose(vector),
        }
    }

    fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        match self {
            Self::Dense(d) => d.slice(s![rows, ..]).to_owned(),
            Self::RandomEffect(op) => op.row_chunk(rows),
            Self::Intercept(_) => Array2::ones((rows.end - rows.start, 1)),
            Self::Operator(op) => op.row_chunk(rows),
        }
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        match self {
            Self::Dense(d) => {
                let mut xtwx = Array2::<f64>::zeros((d.ncols(), d.ncols()));
                streaming_blas_xt_diag_x(d, weights, &mut xtwx);
                Ok(xtwx)
            }
            Self::RandomEffect(op) => op.diag_xtw_x(weights),
            Self::Intercept(_) => {
                let sum: f64 = weights.iter().map(|w| w.max(0.0)).sum();
                Ok(Array2::from_elem((1, 1), sum))
            }
            Self::Operator(op) => op.diag_xtw_x(weights),
        }
    }

    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
        match self {
            Self::Dense(d) => {
                let p = d.ncols();
                let mut diag = Array1::<f64>::zeros(p);
                for i in 0..d.nrows() {
                    let wi = weights[i].max(0.0);
                    if wi == 0.0 {
                        continue;
                    }
                    for j in 0..p {
                        let xij = d[[i, j]];
                        diag[j] += wi * xij * xij;
                    }
                }
                Ok(diag)
            }
            Self::RandomEffect(op) => op.diag_gram(weights),
            Self::Intercept(_) => {
                let sum: f64 = weights.iter().map(|w| w.max(0.0)).sum();
                Ok(Array1::from_vec(vec![sum]))
            }
            Self::Operator(op) => op.diag_gram(weights),
        }
    }

    /// Materialize this block as a dense (n, p_k) matrix.
    fn to_dense(&self) -> Array2<f64> {
        match self {
            Self::Dense(d) => d.as_ref().clone(),
            Self::RandomEffect(op) => op.to_dense(),
            Self::Intercept(n) => Array2::ones((*n, 1)),
            Self::Operator(op) => op.to_dense(),
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

            // ── Dense × RandomEffect ────────────────────────────────────
            (DesignBlock::Dense(d), DesignBlock::RandomEffect(re)) => {
                Ok(re.weighted_cross_with_dense(d, weights))
            }
            (DesignBlock::RandomEffect(re), DesignBlock::Dense(d)) => {
                let cross_t = re.weighted_cross_with_dense(d, weights);
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
                let ones = Array1::from_elem(self.n, 1.0);
                let weighted = Array1::from_shape_fn(self.n, |idx| {
                    ones[idx] * weights[idx].max(0.0)
                });
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

            // ── Generic fallback for Operator × anything ────────────────
            // Compute column-by-column: cross[:, c] = block_i' · (w ⊙ block_j · e_c)
            _ => {
                let bi = &self.blocks[i];
                let bj = &self.blocks[j];
                let pi = bi.ncols();
                let pj = bj.ncols();
                let mut cross = Array2::<f64>::zeros((pi, pj));
                let mut e_c = Array1::<f64>::zeros(pj);
                for c in 0..pj {
                    e_c[c] = 1.0;
                    let col_j = bj.apply(&e_c);
                    let weighted = Array1::from_shape_fn(self.n, |idx| {
                        col_j[idx] * weights[idx].max(0.0)
                    });
                    let cross_col = bi.apply_transpose(&weighted);
                    cross.column_mut(c).assign(&cross_col);
                    e_c[c] = 0.0;
                }
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
                let xm = fast_ab(d, m_kk);
                let mut out = Array1::<f64>::zeros(self.n);
                for i in 0..self.n {
                    out[i] = d.row(i).dot(&xm.row(i));
                }
                Ok(out)
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
            DesignBlock::Operator(op) => op.quadratic_form_diag(m_kk),
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
                let da_m = fast_ab(da, m_ab);
                let mut out = Array1::<f64>::zeros(self.n);
                for i in 0..self.n {
                    out[i] = da_m.row(i).dot(&db.row(i));
                }
                Ok(out)
            }
            (DesignBlock::Dense(d), DesignBlock::RandomEffect(re)) => {
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
                // m_ab is (1, p_b).  For each i: m_ab[0,:] · row_i(other).
                let dense_b = other.to_dense();
                let m_row = m_ab.row(0);
                let mut out = Array1::<f64>::zeros(self.n);
                for i in 0..self.n {
                    out[i] = dense_b.row(i).dot(&m_row);
                }
                Ok(out)
            }
            (other, DesignBlock::Intercept(_)) => {
                // m_ab is (p_a, 1).  For each i: row_i(other) · m_ab[:,0].
                let dense_a = other.to_dense();
                let m_col = m_ab.column(0);
                let mut out = Array1::<f64>::zeros(self.n);
                for i in 0..self.n {
                    out[i] = dense_a.row(i).dot(&m_col);
                }
                Ok(out)
            }

            // Generic fallback: materialize both blocks.
            _ => {
                let da = block_a.to_dense();
                let db = block_b.to_dense();
                let da_m = fast_ab(&da, m_ab);
                let mut out = Array1::<f64>::zeros(self.n);
                for i in 0..self.n {
                    out[i] = da_m.row(i).dot(&db.row(i));
                }
                Ok(out)
            }
        }
    }
}

impl DesignOperator for BlockDesignOperator {
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

    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
        let mut wy = Array1::<f64>::zeros(self.n);
        for i in 0..self.n {
            wy[i] = weights[i].max(0.0) * y[i];
        }
        Ok(self.apply_transpose(&wy))
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

    fn uses_matrix_free_pcg(&self) -> bool {
        // Enable PCG when any block is non-dense (RE, Operator, or Intercept).
        self.blocks.iter().any(|b| {
            matches!(
                b,
                DesignBlock::RandomEffect(_)
                    | DesignBlock::Operator(_)
                    | DesignBlock::Intercept(_)
            )
        })
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

impl DesignOperator for MultiChannelOperator {
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

    fn uses_matrix_free_pcg(&self) -> bool {
        true
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

impl DesignOperator for RowwiseKroneckerOperator {
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

impl DesignOperator for ConditionedDesign {
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
                let val = a[i] * base[[i, j]] * a[j]
                    - a[i] * cw[i] * d[j]
                    - d[i] * cw[j] * a[j]
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

    fn uses_matrix_free_pcg(&self) -> bool {
        match &self.inner {
            DesignMatrix::Dense(_) => true,
            DesignMatrix::Sparse(_) => false,
            DesignMatrix::Operator(op) => op.uses_matrix_free_pcg(),
        }
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
/// The `Operator` variant wraps implicit/structured designs (multi-channel,
/// rowwise-Kronecker, etc.) that implement `DesignOperator` without
/// materialization.  New operator types are added by implementing
/// `DesignOperator` — no match-arm changes needed.
#[derive(Clone)]
pub enum DesignMatrix {
    Dense(Arc<Array2<f64>>),
    Sparse(SparseDesignMatrix),
    Operator(Arc<dyn DesignOperator>),
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

    /// Consuming conversion: moves the inner `Array2` when `Dense`, converts
    /// when `Sparse`.  Avoids the clone that `to_dense()` performs on a Dense
    /// variant that is about to be dropped.
    pub fn into_dense(self) -> Array2<f64> {
        match self {
            Self::Dense(mat) => mat,
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
        DesignMatrix::Dense(x) => Ok(SymmetricMatrix::Dense(fast_xt_diag_x(x, diag))),
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
        DesignMatrix::Operator(op) => Ok(SymmetricMatrix::Dense(op.diag_xtw_x(diag)?)),
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
    fn apply(&self, vector: &Array1<f64>) -> Array1<f64>;
    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64>;
    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String>;
    fn diag_gram(&self, weights: &Array1<f64>) -> Result<Array1<f64>, String>;
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
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn matvec(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.apply(vector)
    }
    fn matvec_trans(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.apply_transpose(vector)
    }
    fn compute_xtwx(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.diag_xtw_x(weights)
    }
    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String>;
    fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String>;
}

impl LinearOperator for DesignMatrix {
    fn uses_matrix_free_pcg(&self) -> bool {
        match self {
            Self::Dense(_) => true,
            Self::Sparse(_) => false,
            Self::Operator(op) => op.uses_matrix_free_pcg(),
        }
    }

    fn nrows(&self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.nrows(),
            Self::Sparse(matrix) => matrix.nrows(),
            Self::Operator(op) => op.nrows(),
        }
    }

    fn ncols(&self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.ncols(),
            Self::Sparse(matrix) => matrix.ncols(),
            Self::Operator(op) => op.ncols(),
        }
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(matrix) => dense_matvec(matrix, vector),
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
            Self::Operator(op) => op.apply(vector),
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
            Self::Dense(x) => {
                let p = x.ncols();
                let mut out = Array1::<f64>::zeros(p);
                for i in 0..x.nrows() {
                    let wi = weights[i].max(0.0);
                    if wi == 0.0 {
                        continue;
                    }
                    let mut row_dot = 0.0_f64;
                    for j in 0..p {
                        row_dot += x[[i, j]] * vector[j];
                    }
                    if row_dot == 0.0 {
                        continue;
                    }
                    let scaled = wi * row_dot;
                    for j in 0..p {
                        out[j] += scaled * x[[i, j]];
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
            Self::Operator(op) => op.apply_weighted_normal(weights, vector, penalty, ridge),
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
            Self::Dense(matrix) => dense_transpose_matvec(matrix, vector),
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
            Self::Operator(op) => op.apply_transpose(vector),
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
            Self::Dense(x) => {
                streaming_blas_xt_diag_x(x, weights, &mut xtwx);
                Ok(xtwx)
            }
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
            Self::Operator(op) => op.diag_xtw_x(weights),
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
            Self::Dense(x) => {
                for i in 0..x.nrows() {
                    let wi = weights[i].max(0.0);
                    if wi == 0.0 {
                        continue;
                    }
                    for j in 0..p {
                        let xij = x[[i, j]];
                        diag[j] += wi * xij * xij;
                    }
                }
                Ok(diag)
            }
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
            Self::Operator(op) => op.diag_gram(weights),
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
            Self::Operator(_) => {
                // Delegate to the default trait method which assembles dense X'WX.
                let mut system = self.diag_xtw_x(weights)?;
                if let Some(pen) = penalty {
                    system += pen;
                }
                let factor = crate::linalg::utils::StableSolver::new("operator system")
                    .factorize(&system)
                    .map_err(|e| format!("factorize_system (operator) failed: {e:?}"))?;
                Ok(Box::new(factor))
            }
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
        match self {
            Self::Dense(x) => Ok(dense_transpose_weighted_response(x, weights, y, None)),
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
            Self::Operator(op) => op.compute_xtwy(weights, y),
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
            Self::Dense(xd) => {
                let xc = fast_ab(xd, middle);
                let mut out = Array1::<f64>::zeros(self.nrows());
                for i in 0..xd.nrows() {
                    out[i] = xd.row(i).dot(&xc.row(i)).max(0.0);
                }
                Ok(out)
            }
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
            Self::Operator(op) => op.quadratic_form_diag(middle),
        }
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

    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
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

    fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        let dense = self.materialize();
        DesignMatrix::Dense(Arc::new(dense)).quadratic_form_diag(middle)
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

    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
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
        let local = DesignMatrix::Dense(Arc::new(self.local.clone())).diag_gram(weights)?;
        out.slice_mut(ndarray::s![self.global_range.clone()])
            .assign(&local);
        Ok(out)
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
        let local = dense_transpose_weighted_response(self.local, weights, y, None);
        let mut out = Array1::<f64>::zeros(self.total_cols);
        out.slice_mut(ndarray::s![self.global_range.clone()])
            .assign(&local);
        Ok(out)
    }

    fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        let middle_local = middle
            .slice(ndarray::s![
                self.global_range.clone(),
                self.global_range.clone()
            ])
            .to_owned();
        DesignMatrix::Dense(Arc::new(self.local.clone())).quadratic_form_diag(&middle_local)
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
    /// range.  For `Operator` variants this delegates to the operator's
    /// `row_chunk`, which is O(chunk) for all concrete operators.
    pub fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
        match self {
            Self::Dense(matrix) => matrix.slice(s![rows, ..]).to_owned(),
            Self::Sparse(matrix) => {
                let csr = matrix
                    .to_csr_arc()
                    .unwrap_or_else(|| panic!("DesignMatrix::row_chunk: failed to obtain CSR view"));
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
            Self::Operator(op) => op.row_chunk(rows),
        }
    }

    /// Element access: returns the value at row `i`, column `j`.
    ///
    /// For dense matrices this is O(1). For sparse matrices, this converts to
    /// a dense cache first and then indexes, so callers doing per-row sweeps
    /// should prefer `to_dense()` for bulk access.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        match self {
            Self::Dense(matrix) => matrix[[i, j]],
            Self::Sparse(sp) => {
                let dense = sp
                    .try_to_dense_arc("DesignMatrix::get")
                    .unwrap_or_else(|msg| panic!("{msg}"));
                dense[[i, j]]
            }
            Self::Operator(op) => {
                // Fallback: materialize and index.  Callers doing bulk access
                // should call to_dense() once instead.
                op.to_dense()[[i, j]]
            }
        }
    }

    /// Extract a single column as a dense vector without full densification.
    ///
    /// - `Dense`: O(n) column copy.
    /// - `Sparse` (CSC): O(nnz_j) using the column pointer structure.
    /// - `Operator`: O(matvec) via unit-vector application.
    pub fn extract_column(&self, j: usize) -> Array1<f64> {
        match self {
            Self::Dense(m) => m.column(j).to_owned(),
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
            Self::Operator(op) => {
                let mut e_j = Array1::zeros(op.ncols());
                e_j[j] = 1.0;
                op.apply(&e_j)
            }
        }
    }

    /// Returns a reference to the inner dense array if this is a `Dense` variant.
    pub fn as_dense_ref(&self) -> Option<&Array2<f64>> {
        match self {
            Self::Dense(matrix) => Some(matrix.as_ref()),
            Self::Sparse(_) | Self::Operator(_) => None,
        }
    }

    /// Zero-copy borrow when `Dense`, materialized conversion when `Sparse`.
    ///
    /// This avoids the unconditional clone that `to_dense()` performs on dense
    /// matrices.  Callers that only need a `&Array2<f64>` should use this and
    /// then call `Cow::as_ref()` or `&*cow`.
    pub fn as_dense_cow(&self) -> Cow<'_, Array2<f64>> {
        match self {
            Self::Dense(matrix) => Cow::Borrowed(matrix.as_ref()),
            Self::Sparse(matrix) => Cow::Owned(
                matrix
                    .try_to_dense_arc("DesignMatrix::as_dense_cow")
                    .unwrap_or_else(|msg| panic!("{msg}"))
                    .as_ref()
                    .clone(),
            ),
            Self::Operator(op) => Cow::Owned(op.to_dense()),
        }
    }

    pub fn to_dense(&self) -> Array2<f64> {
        match self {
            Self::Dense(matrix) => (**matrix).clone(),
            Self::Sparse(matrix) => matrix
                .try_to_dense_arc("DesignMatrix::to_dense")
                .unwrap_or_else(|msg| panic!("{msg}"))
                .as_ref()
                .clone(),
            Self::Operator(op) => op.to_dense(),
        }
    }

    pub fn to_dense_arc(&self) -> Arc<Array2<f64>> {
        match self {
            Self::Dense(matrix) => matrix.clone(),
            Self::Sparse(matrix) => matrix
                .try_to_dense_arc("DesignMatrix::to_dense_arc")
                .unwrap_or_else(|msg| panic!("{msg}")),
            Self::Operator(op) => Arc::new(op.to_dense()),
        }
    }

    pub fn try_to_dense_arc(&self, context: &str) -> Result<Arc<Array2<f64>>, String> {
        match self {
            Self::Dense(matrix) => Ok(matrix.clone()),
            Self::Sparse(matrix) => matrix.try_to_dense_arc(context),
            Self::Operator(op) => Ok(Arc::new(op.to_dense())),
        }
    }

    pub fn to_csr_cache(&self) -> Option<SparseRowMat<usize, f64>> {
        match self {
            Self::Dense(_) | Self::Operator(_) => None,
            Self::Sparse(matrix) => matrix.to_csr_arc().map(|arc| (*arc).clone()),
        }
    }

    pub fn as_sparse(&self) -> Option<&SparseDesignMatrix> {
        match self {
            Self::Sparse(matrix) => Some(matrix),
            Self::Dense(_) | Self::Operator(_) => None,
        }
    }

    pub fn as_dense(&self) -> Option<&Array2<f64>> {
        match self {
            Self::Dense(matrix) => Some(matrix.as_ref()),
            Self::Sparse(_) | Self::Operator(_) => None,
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
        <Self as LinearOperator>::compute_xtwy(self, weights, y)
    }

    pub fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String> {
        <Self as LinearOperator>::quadratic_form_diag(self, middle)
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
        Self::Dense(Arc::new(value.to_owned()))
    }
}

impl From<Array2<f64>> for DesignMatrix {
    fn from(value: Array2<f64>) -> Self {
        Self::Dense(Arc::new(value))
    }
}

impl From<&Array2<f64>> for DesignMatrix {
    fn from(value: &Array2<f64>) -> Self {
        Self::Dense(Arc::new(value.clone()))
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
        let dense_design = DesignMatrix::Dense(Arc::new(sparse_design.to_dense()));
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
        let design = DesignMatrix::Dense(Arc::new(array![[1.0, 0.0], [0.0, 0.0]]));
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
        let design = DesignMatrix::Dense(Arc::new(x.clone()));
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
        let design = DesignMatrix::Dense(Arc::new(x));
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
        let design = DesignMatrix::Dense(Arc::new(x.clone()));
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
}

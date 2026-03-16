use crate::faer_ndarray::{
    FaerArrayView, array2_to_matmut, fast_ab, fast_atb, fast_atv, fast_av, fast_xt_diag_x,
};
use crate::types::RidgePolicy;
use faer::Accum;
use faer::linalg::matmul::matmul;
use faer::sparse::{SparseColMat, SparseRowMat, Triplet};
use ndarray::{Array1, Array2, ArrayView2, ShapeBuilder, s};
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
        let _ = self.csr_cache.set(arc.clone());
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

/// Unified design matrix representation for dense and sparse workflows.
///
/// Dense matrices are wrapped in Arc for O(1) cloning — at biobank scale
/// design matrices are 100-500MB and get cloned repeatedly during GAMLSS
/// family construction, warm-start caching, and prediction.
#[derive(Clone)]
pub enum DesignMatrix {
    Dense(Arc<Array2<f64>>),
    Sparse(SparseDesignMatrix),
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
            Self::Dense(mat) => (**mat).clone(),
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
        let baseridge = if ridge_policy.include_laplacehessian {
            ridge_floor.max(1e-15)
        } else {
            0.0
        };
        if self.uses_matrix_free_pcg() && self.ncols() >= MATRIX_FREE_PCG_MIN_P {
            if let Ok(solution) =
                self.solve_system_matrix_free_pcg_try(weights, rhs, penalty, baseridge)
            {
                return Ok(solution);
            }
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
        matches!(self, Self::Dense(_))
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
            }
        }
        Ok(xtwx)
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
            }
        }
        Ok(diag)
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

        let mut out = Array1::<f64>::zeros(self.nrows());
        match self {
            Self::Dense(xd) => {
                let xc = fast_ab(xd, middle);
                for i in 0..xd.nrows() {
                    out[i] = xd.row(i).dot(&xc.row(i)).max(0.0);
                }
            }
            Self::Sparse(xs) => {
                let csr = xs
                    .to_csr_arc()
                    .ok_or_else(|| "quadratic_form_diag: failed to obtain CSR view".to_string())?;
                let sym = csr.symbolic();
                let row_ptr = sym.row_ptr();
                let col_idx = sym.col_idx();
                let vals = csr.val();
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
            }
        }
        Ok(out)
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
        }
    }

    /// Returns a reference to the inner dense array if this is a `Dense` variant.
    pub fn as_dense_ref(&self) -> Option<&Array2<f64>> {
        match self {
            Self::Dense(matrix) => Some(matrix),
            Self::Sparse(_) => None,
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
        }
    }

    pub fn to_dense_arc(&self) -> Arc<Array2<f64>> {
        match self {
            Self::Dense(matrix) => Arc::new(matrix.clone()),
            Self::Sparse(matrix) => matrix
                .try_to_dense_arc("DesignMatrix::to_dense_arc")
                .unwrap_or_else(|msg| panic!("{msg}")),
        }
    }

    pub fn try_to_dense_arc(&self, context: &str) -> Result<Arc<Array2<f64>>, String> {
        match self {
            Self::Dense(matrix) => Ok(Arc::new(matrix.clone())),
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
            Self::Dense(matrix) => Some(matrix),
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

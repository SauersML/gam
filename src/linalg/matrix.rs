use faer::sparse::{SparseColMat, SparseRowMat};
use ndarray::{Array1, Array2, ArrayView2};
use std::ops::Deref;
use std::sync::{Arc, OnceLock};

#[inline]
fn dense_matvec(matrix: &Array2<f64>, vector: &Array1<f64>) -> Array1<f64> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    let mut out = Array1::<f64>::zeros(nrows);

    if ncols == 0 || nrows == 0 {
        return out;
    }

    if matrix.is_standard_layout()
        && let (Some(ms), Some(vs), Some(os)) = (
            matrix.as_slice_memory_order(),
            vector.as_slice(),
            out.as_slice_mut(),
        )
    {
        for (i, row) in ms.chunks_exact(ncols).enumerate() {
            let mut acc = 0.0_f64;
            for j in 0..ncols {
                acc += row[j] * vs[j];
            }
            os[i] = acc;
        }
        return out;
    }

    for i in 0..nrows {
        let mut acc = 0.0_f64;
        for j in 0..ncols {
            acc += matrix[[i, j]] * vector[j];
        }
        out[i] = acc;
    }
    out
}

#[inline]
fn dense_transpose_matvec(matrix: &Array2<f64>, vector: &Array1<f64>) -> Array1<f64> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    let mut out = Array1::<f64>::zeros(ncols);

    if ncols == 0 || nrows == 0 {
        return out;
    }

    if matrix.is_standard_layout()
        && let (Some(ms), Some(vs), Some(os)) = (
            matrix.as_slice_memory_order(),
            vector.as_slice(),
            out.as_slice_mut(),
        )
    {
        for (i, row) in ms.chunks_exact(ncols).enumerate() {
            let vi = vs[i];
            for j in 0..ncols {
                os[j] += row[j] * vi;
            }
        }
        return out;
    }

    for i in 0..nrows {
        let vi = vector[i];
        for j in 0..ncols {
            out[j] += matrix[[i, j]] * vi;
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

    pub fn to_dense_arc(&self) -> Arc<Array2<f64>> {
        self.dense_cache
            .get_or_init(|| {
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
            })
            .clone()
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
#[derive(Clone)]
pub enum DesignMatrix {
    Dense(Array2<f64>),
    Sparse(SparseDesignMatrix),
}

pub trait LinearOperator {
    fn apply(&self, vector: &Array1<f64>) -> Array1<f64>;
    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64>;
    fn diag_xt_w_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String>;
    fn solve_system(
        &self,
        weights: &Array1<f64>,
        rhs: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
    ) -> Result<Array1<f64>, String> {
        if rhs.len() != self.ncols() {
            return Err(format!(
                "solve_system rhs dimension mismatch: rhs length {} != ncols {}",
                rhs.len(),
                self.ncols()
            ));
        }
        let mut system = self.diag_xt_w_x(weights)?;
        if let Some(pen) = penalty {
            if pen.nrows() != system.nrows() || pen.ncols() != system.ncols() {
                return Err(format!(
                    "solve_system penalty shape mismatch: got {}x{}, expected {}x{}",
                    pen.nrows(),
                    pen.ncols(),
                    system.nrows(),
                    system.ncols()
                ));
            }
            system += pen;
        }
        use crate::faer_ndarray::FaerCholesky;
        use faer::Side;
        system
            .cholesky(Side::Lower)
            .map(|chol| chol.solve_vec(rhs))
            .map_err(|_| "solve_system failed to factorize system".to_string())
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
        self.diag_xt_w_x(weights)
    }
    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String>;
    fn quadratic_form_diag(&self, middle: &Array2<f64>) -> Result<Array1<f64>, String>;
}

impl LinearOperator for DesignMatrix {
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

    fn diag_xt_w_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
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
                for i in 0..x.nrows() {
                    let wi = weights[i].max(0.0);
                    if wi == 0.0 {
                        continue;
                    }
                    for a in 0..p {
                        let xa = x[[i, a]];
                        for b in a..p {
                            let v = wi * xa * x[[i, b]];
                            xtwx[[a, b]] += v;
                            if a != b {
                                xtwx[[b, a]] += v;
                            }
                        }
                    }
                }
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

    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
        if weights.len() != self.nrows() || y.len() != self.nrows() {
            return Err(format!(
                "compute_xtwy dimension mismatch: weights={}, y={}, nrows={}",
                weights.len(),
                y.len(),
                self.nrows()
            ));
        }
        let mut wy = y.clone();
        for i in 0..wy.len() {
            wy[i] *= weights[i].max(0.0);
        }
        Ok(self.matvec_trans(&wy))
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
                let xc = xd.dot(middle);
                for i in 0..xd.nrows() {
                    out[i] = xd.row(i).dot(&xc.row(i)).max(0.0);
                }
            }
            Self::Sparse(xs) => {
                if let Ok(csr) = xs.as_ref().to_row_major() {
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
                } else {
                    let dense_arc = self.to_dense_arc();
                    let dense = dense_arc.as_ref();
                    let xc = dense.dot(middle);
                    for i in 0..dense.nrows() {
                        out[i] = dense.row(i).dot(&xc.row(i)).max(0.0);
                    }
                }
            }
        }
        Ok(out)
    }
}

impl DesignMatrix {
    pub fn nrows(&self) -> usize {
        <Self as LinearOperator>::nrows(self)
    }

    pub fn ncols(&self) -> usize {
        <Self as LinearOperator>::ncols(self)
    }

    pub fn to_dense(&self) -> Array2<f64> {
        match self {
            Self::Dense(matrix) => matrix.clone(),
            Self::Sparse(matrix) => matrix.to_dense_arc().as_ref().clone(),
        }
    }

    pub fn to_dense_arc(&self) -> Arc<Array2<f64>> {
        match self {
            Self::Dense(matrix) => Arc::new(matrix.clone()),
            Self::Sparse(matrix) => matrix.to_dense_arc(),
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

    pub fn matrix_vector_multiply(&self, vector: &Array1<f64>) -> Array1<f64> {
        <Self as LinearOperator>::apply(self, vector)
    }

    pub fn transpose_vector_multiply(&self, vector: &Array1<f64>) -> Array1<f64> {
        <Self as LinearOperator>::apply_transpose(self, vector)
    }

    pub fn compute_xtwx(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        <Self as LinearOperator>::diag_xt_w_x(self, weights)
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
}

impl<'a> From<ArrayView2<'a, f64>> for DesignMatrix {
    fn from(value: ArrayView2<'a, f64>) -> Self {
        Self::Dense(value.to_owned())
    }
}

impl From<Array2<f64>> for DesignMatrix {
    fn from(value: Array2<f64>) -> Self {
        Self::Dense(value)
    }
}

impl From<&Array2<f64>> for DesignMatrix {
    fn from(value: &Array2<f64>) -> Self {
        Self::Dense(value.clone())
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
    use super::{DesignMatrix, dense_matvec, dense_transpose_matvec};
    use faer::sparse::{SparseColMat, SymbolicSparseColMat};
    use ndarray::array;

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
        let y_sparse = design.matrix_vector_multiply(&v);
        let y_dense = dense.dot(&v);
        for i in 0..y_sparse.len() {
            assert!((y_sparse[i] - y_dense[i]).abs() < 1e-12);
        }
    }
}

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
                        out[[row_idx[idx], col]] = values[idx];
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

impl DesignMatrix {
    pub fn nrows(&self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.nrows(),
            Self::Sparse(matrix) => matrix.nrows(),
        }
    }

    pub fn ncols(&self) -> usize {
        match self {
            Self::Dense(matrix) => matrix.ncols(),
            Self::Sparse(matrix) => matrix.ncols(),
        }
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

    pub fn matrix_vector_multiply(&self, vector: &Array1<f64>) -> Array1<f64> {
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

    pub fn transpose_vector_multiply(&self, vector: &Array1<f64>) -> Array1<f64> {
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
    use super::{dense_matvec, dense_transpose_matvec};
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
}

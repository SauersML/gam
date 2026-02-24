use faer::sparse::{SparseColMat, SparseRowMat};
use ndarray::{Array1, Array2, ArrayView2};

/// Unified design matrix representation for dense and sparse workflows.
#[derive(Clone)]
pub enum DesignMatrix {
    Dense(Array2<f64>),
    Sparse(SparseColMat<usize, f64>),
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
            Self::Sparse(matrix) => {
                let dense = matrix.as_ref().to_dense();
                Array2::from_shape_fn((dense.nrows(), dense.ncols()), |(i, j)| dense[(i, j)])
            }
        }
    }

    pub fn to_csr_cache(&self) -> Option<SparseRowMat<usize, f64>> {
        match self {
            Self::Dense(_) => None,
            Self::Sparse(matrix) => matrix.as_ref().to_row_major().ok(),
        }
    }

    pub fn matrix_vector_multiply(&self, vector: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(matrix) => matrix.dot(vector),
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
            Self::Dense(matrix) => matrix.t().dot(vector),
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
        Self::Sparse(value)
    }
}

impl From<&SparseColMat<usize, f64>> for DesignMatrix {
    fn from(value: &SparseColMat<usize, f64>) -> Self {
        Self::Sparse(value.clone())
    }
}

impl From<&DesignMatrix> for DesignMatrix {
    fn from(value: &DesignMatrix) -> Self {
        value.clone()
    }
}

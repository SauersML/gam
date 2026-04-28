use crate::matrix::{DesignMatrix, FactorizedSystem, SymmetricMatrix};
use ndarray::{Array1, Array2, ArrayView2, s};
use std::ops::Range;

const PREDICTION_TARGET_WORK_BYTES: usize = 8 * 1024 * 1024;
const PREDICTION_MIN_CHUNK_ROWS: usize = 16;
const PREDICTION_MAX_CHUNK_ROWS: usize = 4096;

pub enum PredictionCovarianceBackend<'a> {
    Dense(ArrayView2<'a, f64>),
    Factorized {
        factor: Box<dyn FactorizedSystem>,
        dim: usize,
    },
}

impl<'a> PredictionCovarianceBackend<'a> {
    pub fn from_dense(covariance: ArrayView2<'a, f64>) -> Self {
        Self::Dense(covariance)
    }

    pub fn from_factorized_hessian(hessian: SymmetricMatrix) -> Result<Self, String> {
        if hessian.nrows() != hessian.ncols() {
            return Err(format!(
                "prediction precision backend requires a square Hessian, got {}x{}",
                hessian.nrows(),
                hessian.ncols()
            ));
        }
        let has_nonzero = match &hessian {
            SymmetricMatrix::Dense(m) => m.iter().any(|v| v.abs() > 0.0),
            SymmetricMatrix::Sparse(m) => {
                let (_, vals) = m.parts();
                vals.iter().any(|v| v.abs() > 0.0)
            }
        };
        if !has_nonzero {
            return Err("prediction precision backend requires a non-zero Hessian".to_string());
        }
        let dim = hessian.nrows();
        let factor = hessian.factorize()?;
        Ok(Self::Factorized { factor, dim })
    }

    pub fn parameter_dim(&self) -> usize {
        match self {
            Self::Dense(covariance) => covariance.nrows(),
            Self::Factorized { dim, .. } => *dim,
        }
    }

    pub fn nrows(&self) -> usize {
        match self {
            Self::Dense(covariance) => covariance.nrows(),
            Self::Factorized { .. } => self.parameter_dim(),
        }
    }

    pub fn apply_columns(&self, rhs: &Array2<f64>) -> Result<Array2<f64>, String> {
        if rhs.nrows() != self.nrows() {
            return Err(format!(
                "prediction covariance backend column mismatch: rhs has {} rows, expected {}",
                rhs.nrows(),
                self.nrows()
            ));
        }
        match self {
            Self::Dense(covariance) => Ok(covariance.dot(rhs)),
            Self::Factorized { factor, .. } => factor.solvemulti(rhs),
        }
    }
}

pub(crate) fn design_row_chunk(
    design: &DesignMatrix,
    rows: Range<usize>,
) -> Result<Array2<f64>, String> {
    if rows.end > design.nrows() || rows.start > rows.end {
        return Err(format!(
            "design_row_chunk row range {}..{} is out of bounds for {} rows",
            rows.start,
            rows.end,
            design.nrows()
        ));
    }
    design.try_row_chunk(rows).map_err(|e| e.to_string())
}

pub(crate) fn prediction_chunk_rows(
    parameter_dim: usize,
    local_dim: usize,
    n_rows: usize,
) -> usize {
    if n_rows == 0 {
        return 1;
    }
    let bytes_per_row = parameter_dim
        .max(1)
        .saturating_mul(local_dim.max(1))
        .saturating_mul(std::mem::size_of::<f64>())
        .saturating_mul(4);
    let target_rows = if bytes_per_row == 0 {
        n_rows
    } else {
        PREDICTION_TARGET_WORK_BYTES / bytes_per_row
    };
    target_rows
        .max(PREDICTION_MIN_CHUNK_ROWS)
        .min(PREDICTION_MAX_CHUNK_ROWS)
        .min(n_rows.max(1))
}

pub fn rowwise_local_covariances<F>(
    backend: &PredictionCovarianceBackend<'_>,
    n_rows: usize,
    local_dim: usize,
    mut build_chunk: F,
) -> Result<Vec<Vec<Array1<f64>>>, String>
where
    F: FnMut(Range<usize>) -> Result<Vec<Array2<f64>>, String>,
{
    if local_dim == 0 {
        return Err("rowwise_local_covariances requires local_dim > 0".to_string());
    }
    let parameter_dim = backend.nrows();
    let chunk_rows = prediction_chunk_rows(parameter_dim, local_dim, n_rows);
    let mut out: Vec<Vec<Array1<f64>>> = (0..local_dim)
        .map(|_| {
            (0..local_dim)
                .map(|_| Array1::<f64>::zeros(n_rows))
                .collect::<Vec<_>>()
        })
        .collect();

    let mut start = 0usize;
    while start < n_rows {
        let end = (start + chunk_rows).min(n_rows);
        let rows = start..end;
        let gradients = build_chunk(rows.clone())?;
        if gradients.len() != local_dim {
            return Err(format!(
                "rowwise_local_covariances chunk builder returned {} local components, expected {}",
                gradients.len(),
                local_dim
            ));
        }
        let rows_in_chunk = end - start;
        // Pack RHS by transposing each gradient block in one shot:
        //   rhs[:, component*R .. (component+1)*R] = gradients[component].t()
        // This replaces the per-row `rhs.slice_mut(...).assign(grad.row(r).t())`
        // loop with O(local_dim) ndarray block assignments — same memory motion
        // but with contiguous column-block destinations and a single bounds
        // check per component.
        let mut rhs = Array2::<f64>::zeros((parameter_dim, rows_in_chunk * local_dim));
        for (component, grad) in gradients.iter().enumerate() {
            if grad.nrows() != rows_in_chunk || grad.ncols() != parameter_dim {
                return Err(format!(
                    "rowwise_local_covariances component {component} has shape {}x{}, expected {}x{}",
                    grad.nrows(),
                    grad.ncols(),
                    rows_in_chunk,
                    parameter_dim
                ));
            }
            let col_start = component * rows_in_chunk;
            let col_end = col_start + rows_in_chunk;
            rhs.slice_mut(s![.., col_start..col_end]).assign(&grad.t());
        }

        let solved = backend.apply_columns(&rhs)?;
        if solved.nrows() != parameter_dim || solved.ncols() != rows_in_chunk * local_dim {
            return Err(format!(
                "rowwise_local_covariances backend returned {}x{}, expected {}x{}",
                solved.nrows(),
                solved.ncols(),
                parameter_dim,
                rows_in_chunk * local_dim
            ));
        }

        // Build the per-row local covariance entries.
        //
        // For each (a, b) the value at row r is
        //   v_ab[r] = gradients[a][r, :]  ·  solved[:, b*R + r]
        // which is the diagonal of  G_a · S_b  where
        //   G_a = gradients[a]                     (R × P)
        //   S_b = solved[:, b*R .. (b+1)*R]        (P × R)
        // We hoist the per-component slice of `solved` once per outer index
        // rather than re-deriving the column for each `local_row`, and the
        // off-diagonal symmetrization (0.5 * (v_ab + v_ba)) is preserved
        // exactly: `solved` is generally non-symmetric for factorized backends
        // due to floating-point round-off, so we keep the explicit average to
        // match the original numerics.
        for a in 0..local_dim {
            let g_a = gradients[a].view();
            for b in a..local_dim {
                let s_b = solved.slice(s![.., b * rows_in_chunk..(b + 1) * rows_in_chunk]);
                if a == b {
                    let mut col = out[a][b].slice_mut(s![start..end]);
                    for local_row in 0..rows_in_chunk {
                        col[local_row] = g_a.row(local_row).dot(&s_b.column(local_row));
                    }
                } else {
                    let g_b = gradients[b].view();
                    let s_a = solved.slice(s![.., a * rows_in_chunk..(a + 1) * rows_in_chunk]);
                    {
                        let mut col_ab = out[a][b].slice_mut(s![start..end]);
                        for local_row in 0..rows_in_chunk {
                            let v_ab = g_a.row(local_row).dot(&s_b.column(local_row));
                            let v_ba = g_b.row(local_row).dot(&s_a.column(local_row));
                            col_ab[local_row] = 0.5 * (v_ab + v_ba);
                        }
                    }
                    // Mirror to the (b, a) entry; disjoint from out[a][b]
                    // because a != b here.
                    let copy = out[a][b].slice(s![start..end]).to_owned();
                    let mut col_ba = out[b][a].slice_mut(s![start..end]);
                    col_ba.assign(&copy);
                }
            }
        }

        start = end;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::{SparseColMat, Triplet};
    use ndarray::array;

    fn sparse_design_from_dense(dense: &Array2<f64>) -> DesignMatrix {
        let mut triplets = Vec::<Triplet<usize, usize, f64>>::new();
        for i in 0..dense.nrows() {
            for j in 0..dense.ncols() {
                let value = dense[[i, j]];
                if value != 0.0 {
                    triplets.push(Triplet::new(i, j, value));
                }
            }
        }
        let sparse = SparseColMat::try_new_from_triplets(dense.nrows(), dense.ncols(), &triplets)
            .expect("assemble sparse design");
        DesignMatrix::Sparse(crate::matrix::SparseDesignMatrix::new(sparse))
    }

    #[test]
    fn rowwise_local_covariances_match_dense_direct_formula() {
        let covariance = array![[2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 1.1]];
        let backend = PredictionCovarianceBackend::from_dense(covariance.view());
        let grads0 = array![[1.0, 0.0, 2.0], [0.5, -1.0, 0.0], [0.0, 1.0, 1.0]];
        let grads1 = array![[0.0, 1.0, 1.0], [1.0, 0.5, -0.5], [2.0, 0.0, 0.5]];
        let out = rowwise_local_covariances(&backend, 3, 2, |rows| {
            Ok(vec![
                grads0.slice(s![rows.clone(), ..]).to_owned(),
                grads1.slice(s![rows, ..]).to_owned(),
            ])
        })
        .expect("chunked local covariances");

        for i in 0..3 {
            let g0 = grads0.row(i).to_owned();
            let g1 = grads1.row(i).to_owned();
            let expected00 = g0.dot(&covariance.dot(&g0));
            let expected01 = g0.dot(&covariance.dot(&g1));
            let expected11 = g1.dot(&covariance.dot(&g1));
            assert!((out[0][0][i] - expected00).abs() <= 1e-12);
            assert!((out[0][1][i] - expected01).abs() <= 1e-12);
            assert!((out[1][1][i] - expected11).abs() <= 1e-12);
        }
    }

    #[test]
    fn rowwise_local_covariances_match_factorized_precision() {
        let precision = array![[4.0, 1.0, 0.0], [1.0, 3.5, 0.2], [0.0, 0.2, 2.5]];
        // Compute the reference covariance via the same Cholesky factorization
        // path that the backend uses, so both sides agree without ridge bias.
        let p = precision.nrows();
        let factor = SymmetricMatrix::Dense(precision.clone())
            .factorize()
            .expect("factorize SPD precision for reference covariance");
        let covariance = factor
            .solvemulti(&Array2::eye(p))
            .expect("invert SPD precision via Cholesky");
        let backend = PredictionCovarianceBackend::from_factorized_hessian(SymmetricMatrix::Dense(
            precision.clone(),
        ))
        .expect("factorize");
        let grads = array![[1.0, 0.0, 2.0], [0.5, -1.0, 0.0], [0.0, 1.0, 1.0]];
        let out = rowwise_local_covariances(&backend, 3, 1, |rows| {
            Ok(vec![grads.slice(s![rows, ..]).to_owned()])
        })
        .expect("chunked variance");

        for i in 0..3 {
            let g = grads.row(i).to_owned();
            let expected = g.dot(&covariance.dot(&g));
            assert!((out[0][0][i] - expected).abs() <= 1e-10);
        }
    }

    #[test]
    fn design_row_chunk_preserves_sparse_rows() {
        let dense = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
        let sparse = sparse_design_from_dense(&dense);
        let chunk = design_row_chunk(&sparse, 1..3).expect("sparse row chunk");
        assert_eq!(chunk, dense.slice(s![1..3, ..]).to_owned());
    }
}

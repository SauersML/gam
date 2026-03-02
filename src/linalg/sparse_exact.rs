use crate::estimate::EstimationError;
use crate::faer_ndarray::{FaerArrayView, FaerCholesky, FaerEigh};
use crate::matrix::DesignMatrix;
use crate::solver::pirls::{PirlsWorkspace, sparse_reml_penalized_hessian};
use faer::Side;
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::Llt as SparseLlt;
use faer::sparse::{SparseColMat, SparseRowMat, Triplet};
use ndarray::{Array1, Array2};
use std::collections::BTreeMap;
use std::sync::Arc;

const ZERO_TOL: f64 = 1e-12;

#[derive(Clone)]
pub struct SparseExactFactor {
    factor: SparseLlt<usize, f64>,
    n: usize,
    logdet: f64,
}

#[derive(Clone, Default)]
pub struct SparseTraceWorkspace {
    rhs: Vec<f64>,
}

#[derive(Clone)]
pub struct SparsePenaltyBlock {
    pub term_index: usize,
    pub p_start: usize,
    pub p_end: usize,
    pub positive_eigenvalues: Arc<Vec<f64>>,
    pub s_k_sparse: SparseColMat<usize, f64>,
    pub r_k_sparse: SparseColMat<usize, f64>,
    r_k_rows: Arc<SparseRowMat<usize, f64>>,
}

#[derive(Clone)]
pub struct SparsePenalizedSystem {
    pub h_sparse: SparseColMat<usize, f64>,
    pub factor: SparseExactFactor,
    pub logdet_h: f64,
}

impl SparseTraceWorkspace {
    fn rhs_slice(&mut self, len: usize) -> &mut [f64] {
        if self.rhs.len() != len {
            self.rhs.resize(len, 0.0);
        }
        self.rhs.fill(0.0);
        &mut self.rhs
    }
}

pub fn dense_to_sparse(
    matrix: &Array2<f64>,
    tol: f64,
) -> Result<SparseColMat<usize, f64>, EstimationError> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    let mut triplets = Vec::new();
    for row in 0..nrows {
        for col in 0..ncols {
            let value = matrix[[row, col]];
            if value.abs() > tol {
                triplets.push(Triplet::new(row, col, value));
            }
        }
    }
    SparseColMat::try_new_from_triplets(nrows, ncols, &triplets).map_err(|_| {
        EstimationError::InvalidInput("failed to convert dense matrix to sparse CSC".to_string())
    })
}

/// Convert a dense symmetric matrix to sparse CSC storing only the upper triangle.
///
/// This encoding is required by sparse SPD routines in this module that interpret
/// entries as symmetric-upper storage and mirror off-diagonals when reconstructing
/// dense diagnostics.
pub fn dense_to_sparse_symmetric_upper(
    matrix: &Array2<f64>,
    tol: f64,
) -> Result<SparseColMat<usize, f64>, EstimationError> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    let mut triplets = Vec::new();
    for row in 0..nrows {
        for col in row..ncols {
            let value = matrix[[row, col]];
            if value.abs() > tol {
                triplets.push(Triplet::new(row, col, value));
            }
        }
    }
    SparseColMat::try_new_from_triplets(nrows, ncols, &triplets).map_err(|_| {
        EstimationError::InvalidInput(
            "failed to convert dense symmetric matrix to sparse upper-triangle CSC".to_string(),
        )
    })
}

pub fn sparse_to_dense_symmetric_upper_public(matrix: &SparseColMat<usize, f64>) -> Array2<f64> {
    let mut dense = Array2::<f64>::zeros((matrix.nrows(), matrix.ncols()));
    let (symbolic, values) = matrix.parts();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();
    for col in 0..matrix.ncols() {
        let start = col_ptr[col];
        let end = col_ptr[col + 1];
        for idx in start..end {
            let row = row_idx[idx];
            let value = values[idx];
            dense[[row, col]] += value;
            if row != col {
                dense[[col, row]] += value;
            }
        }
    }
    dense
}

fn sparse_matvec(matrix: &SparseColMat<usize, f64>, vector: &Array1<f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(matrix.nrows());
    let (symbolic, values) = matrix.parts();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();
    for col in 0..matrix.ncols() {
        let x = vector[col];
        if x == 0.0 {
            continue;
        }
        for idx in col_ptr[col]..col_ptr[col + 1] {
            out[row_idx[idx]] += values[idx] * x;
        }
    }
    out
}

pub fn sparse_symmetric_upper_matvec_public(
    matrix: &SparseColMat<usize, f64>,
    vector: &Array1<f64>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(matrix.nrows());
    let (symbolic, values) = matrix.parts();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();
    for col in 0..matrix.ncols() {
        let x_col = vector[col];
        for idx in col_ptr[col]..col_ptr[col + 1] {
            let row = row_idx[idx];
            let value = values[idx];
            out[row] += value * x_col;
            if row != col {
                out[col] += value * vector[row];
            }
        }
    }
    out
}

fn sparse_row_quadratic_form(
    factor: &SparseExactFactor,
    workspace: &mut SparseTraceWorkspace,
    row_idx: &[usize],
    row_val: &[f64],
) -> Result<f64, EstimationError> {
    let rhs = workspace.rhs_slice(factor.n);
    for (idx, &value) in row_idx.iter().zip(row_val.iter()) {
        rhs[*idx] = value;
    }
    let rhs_arr =
        Array2::from_shape_vec((factor.n, 1), rhs.to_vec()).expect("rhs vector should reshape");
    let rhs_view = FaerArrayView::new(&rhs_arr);
    let solved = factor.factor.solve(rhs_view.as_ref());
    let mut quad = 0.0_f64;
    for (&idx, &value) in row_idx.iter().zip(row_val.iter()) {
        quad += value * solved[(idx, 0)];
    }
    Ok(quad)
}

pub fn factorize_sparse_spd(
    h: &SparseColMat<usize, f64>,
) -> Result<SparseExactFactor, EstimationError> {
    // Canonicalize to symmetric-upper storage before factorization.
    //
    // Math contract:
    // - If callers pass upper-only storage, values are preserved.
    // - If callers pass full symmetric storage, paired (i,j)/(j,i) entries are averaged.
    // - If callers pass lower-only storage, it is mirrored into upper.
    //
    // This prevents off-diagonal double counting in paths that interpret input as
    // symmetric-upper and makes the sparse factor path robust to caller encoding.
    let h_upper = canonicalize_sparse_symmetric_upper(h, ZERO_TOL)?;
    let factor = h_upper.as_ref().sp_cholesky(Side::Upper).map_err(|_| {
        EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        }
    })?;
    let dense = sparse_to_dense_symmetric_upper_public(&h_upper);
    let chol =
        dense
            .cholesky(Side::Lower)
            .map_err(|_| EstimationError::HessianNotPositiveDefinite {
                min_eigenvalue: f64::NAN,
            })?;
    let logdet = 2.0 * chol.diag().mapv(f64::ln).sum();
    Ok(SparseExactFactor {
        factor,
        n: h_upper.ncols(),
        logdet,
    })
}

fn canonicalize_sparse_symmetric_upper(
    matrix: &SparseColMat<usize, f64>,
    tol: f64,
) -> Result<SparseColMat<usize, f64>, EstimationError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(EstimationError::InvalidInput(format!(
            "sparse SPD factorization requires square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        )));
    }

    #[derive(Default, Clone, Copy)]
    struct PairAccum {
        upper_sum: f64,
        upper_count: usize,
        lower_sum: f64,
        lower_count: usize,
    }

    let mut accum: BTreeMap<(usize, usize), PairAccum> = BTreeMap::new();
    let (symbolic, values) = matrix.parts();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();

    for col in 0..matrix.ncols() {
        let start = col_ptr[col];
        let end = col_ptr[col + 1];
        for idx in start..end {
            let row = row_idx[idx];
            let value = values[idx];
            let (r, c, is_upper) = if row <= col {
                (row, col, true)
            } else {
                (col, row, false)
            };
            let slot = accum.entry((r, c)).or_default();
            if is_upper {
                slot.upper_sum += value;
                slot.upper_count += 1;
            } else {
                slot.lower_sum += value;
                slot.lower_count += 1;
            }
        }
    }

    let mut triplets = Vec::<Triplet<usize, usize, f64>>::new();
    for ((row, col), slot) in accum {
        let value = if row == col {
            let count = slot.upper_count + slot.lower_count;
            if count == 0 {
                0.0
            } else {
                (slot.upper_sum + slot.lower_sum) / (count as f64)
            }
        } else {
            let upper_avg = if slot.upper_count > 0 {
                Some(slot.upper_sum / (slot.upper_count as f64))
            } else {
                None
            };
            let lower_avg = if slot.lower_count > 0 {
                Some(slot.lower_sum / (slot.lower_count as f64))
            } else {
                None
            };
            match (upper_avg, lower_avg) {
                (Some(u), Some(l)) => 0.5 * (u + l),
                (Some(u), None) => u,
                (None, Some(l)) => l,
                (None, None) => 0.0,
            }
        };

        if value.abs() > tol {
            triplets.push(Triplet::new(row, col, value));
        }
    }

    SparseColMat::try_new_from_triplets(matrix.nrows(), matrix.ncols(), &triplets).map_err(|_| {
        EstimationError::InvalidInput(
            "failed to canonicalize sparse matrix to symmetric-upper CSC".to_string(),
        )
    })
}

pub fn solve_sparse_spd(
    factor: &SparseExactFactor,
    rhs: &Array1<f64>,
) -> Result<Array1<f64>, EstimationError> {
    let rhs_arr =
        Array2::from_shape_vec((rhs.len(), 1), rhs.to_vec()).expect("rhs vector should reshape");
    let rhs_view = FaerArrayView::new(&rhs_arr);
    let out = factor.factor.solve(rhs_view.as_ref());
    let mut result = Array1::<f64>::zeros(rhs.len());
    for i in 0..rhs.len() {
        result[i] = out[(i, 0)];
    }
    Ok(result)
}

pub fn solve_sparse_spd_multi(
    factor: &SparseExactFactor,
    rhs: &Array2<f64>,
) -> Result<Array2<f64>, EstimationError> {
    let rhs_view = FaerArrayView::new(rhs);
    let out = factor.factor.solve(rhs_view.as_ref());
    let mut result = Array2::<f64>::zeros(rhs.raw_dim());
    for j in 0..rhs.ncols() {
        for i in 0..rhs.nrows() {
            result[[i, j]] = out[(i, j)];
        }
    }
    Ok(result)
}

pub fn trace_hinv_sk(
    factor: &SparseExactFactor,
    workspace: &mut SparseTraceWorkspace,
    penalty: &SparsePenaltyBlock,
) -> Result<f64, EstimationError> {
    let symbolic = penalty.r_k_rows.symbolic();
    let row_ptr = symbolic.row_ptr();
    let col_idx = symbolic.col_idx();
    let values = penalty.r_k_rows.val();
    let mut total = 0.0_f64;
    for row in 0..penalty.r_k_rows.nrows() {
        let start = row_ptr[row];
        let end = row_ptr[row + 1];
        total += sparse_row_quadratic_form(
            factor,
            workspace,
            &col_idx[start..end],
            &values[start..end],
        )?;
    }
    Ok(total)
}

pub fn leverages_from_factor(
    factor: &SparseExactFactor,
    x: &DesignMatrix,
) -> Result<Array1<f64>, EstimationError> {
    let mut workspace = SparseTraceWorkspace::default();
    match x {
        DesignMatrix::Dense(matrix) => {
            let mut out = Array1::<f64>::zeros(matrix.nrows());
            for row in 0..matrix.nrows() {
                let mut idx = Vec::new();
                let mut val = Vec::new();
                for col in 0..matrix.ncols() {
                    let x_ij = matrix[[row, col]];
                    if x_ij.abs() > ZERO_TOL {
                        idx.push(col);
                        val.push(x_ij);
                    }
                }
                out[row] = sparse_row_quadratic_form(factor, &mut workspace, &idx, &val)?;
            }
            Ok(out)
        }
        DesignMatrix::Sparse(matrix) => {
            let csr = matrix.to_csr_arc().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "failed to build CSR cache for sparse design".to_string(),
                )
            })?;
            let symbolic = csr.symbolic();
            let row_ptr = symbolic.row_ptr();
            let col_idx = symbolic.col_idx();
            let values = csr.val();
            let mut out = Array1::<f64>::zeros(csr.nrows());
            for row in 0..csr.nrows() {
                let start = row_ptr[row];
                let end = row_ptr[row + 1];
                out[row] = sparse_row_quadratic_form(
                    factor,
                    &mut workspace,
                    &col_idx[start..end],
                    &values[start..end],
                )?;
            }
            Ok(out)
        }
    }
}

pub fn logdet_from_factor(factor: &SparseExactFactor) -> Result<f64, EstimationError> {
    Ok(factor.logdet)
}

pub fn assemble_and_factor_sparse_penalized_system(
    workspace: &mut PirlsWorkspace,
    x: &SparseColMat<usize, f64>,
    weights: &Array1<f64>,
    s_lambda: &Array2<f64>,
    ridge: f64,
) -> Result<SparsePenalizedSystem, EstimationError> {
    let h_sparse = sparse_reml_penalized_hessian(workspace, x, weights, s_lambda, ridge)?;
    let factor = factorize_sparse_spd(&h_sparse)?;
    let logdet_h = logdet_from_factor(&factor)?;
    Ok(SparsePenalizedSystem {
        h_sparse,
        factor,
        logdet_h,
    })
}

pub fn build_sparse_penalty_blocks(
    s_list: &[Array2<f64>],
    r_list: &[Array2<f64>],
) -> Result<Option<Vec<SparsePenaltyBlock>>, EstimationError> {
    if s_list.len() != r_list.len() {
        return Err(EstimationError::LayoutError(format!(
            "Penalty/root count mismatch: S={}, R={}",
            s_list.len(),
            r_list.len()
        )));
    }
    let mut ranges = Vec::with_capacity(s_list.len());
    for (term_index, s_k) in s_list.iter().enumerate() {
        let mut min_idx = usize::MAX;
        let mut max_idx = 0usize;
        for row in 0..s_k.nrows() {
            for col in 0..s_k.ncols() {
                if s_k[[row, col]].abs() > ZERO_TOL {
                    min_idx = min_idx.min(row.min(col));
                    max_idx = max_idx.max(row.max(col));
                }
            }
        }
        if min_idx == usize::MAX {
            ranges.push((term_index, 0usize, 0usize));
        } else {
            ranges.push((term_index, min_idx, max_idx + 1));
        }
    }
    let mut sorted = ranges.clone();
    sorted.sort_by_key(|(_, start, _)| *start);
    for pair in sorted.windows(2) {
        let (_, _, end_left) = pair[0];
        let (_, start_right, _) = pair[1];
        if end_left > start_right {
            return Ok(None);
        }
    }

    let mut blocks = Vec::with_capacity(s_list.len());
    for (term_index, p_start, p_end) in ranges {
        let s_k = &s_list[term_index];
        let r_k = &r_list[term_index];
        let s_k_sparse = dense_to_sparse(s_k, ZERO_TOL)?;
        let r_k_sparse = dense_to_sparse(r_k, ZERO_TOL)?;
        let r_k_rows = Arc::new(r_k_sparse.as_ref().to_row_major().map_err(|_| {
            EstimationError::InvalidInput("failed to convert penalty root to CSR".to_string())
        })?);
        let block_dense = if p_end > p_start {
            s_k.slice(ndarray::s![p_start..p_end, p_start..p_end])
                .to_owned()
        } else {
            Array2::<f64>::zeros((0, 0))
        };
        let positive_eigenvalues = if block_dense.nrows() == 0 {
            Vec::new()
        } else {
            let (evals, _) = block_dense
                .eigh(Side::Lower)
                .map_err(EstimationError::EigendecompositionFailed)?;
            evals.iter().copied().filter(|v| *v > ZERO_TOL).collect()
        };
        blocks.push(SparsePenaltyBlock {
            term_index,
            p_start,
            p_end,
            positive_eigenvalues: Arc::new(positive_eigenvalues),
            s_k_sparse,
            r_k_sparse,
            r_k_rows,
        });
    }
    Ok(Some(blocks))
}

pub fn sparse_matvec_public(
    matrix: &SparseColMat<usize, f64>,
    vector: &Array1<f64>,
) -> Array1<f64> {
    sparse_matvec(matrix, vector)
}

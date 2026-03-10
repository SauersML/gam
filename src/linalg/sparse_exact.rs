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

impl crate::matrix::FactorizedSystem for SparseExactFactor {
    fn solve(&self, rhs: &Array1<f64>) -> Result<Array1<f64>, String> {
        solve_sparse_spd(self, rhs).map_err(|e| e.to_string())
    }

    fn solvemulti(&self, rhs: &Array2<f64>) -> Result<Array2<f64>, String> {
        solve_sparse_spdmulti(self, rhs).map_err(|e| e.to_string())
    }

    fn logdet(&self) -> f64 {
        self.logdet
    }
}

#[derive(Clone, Default)]
pub struct SparseTraceWorkspace {
    selected_block_inv_cache: BTreeMap<(usize, usize), Array2<f64>>,
    selected_support_inv_cache: BTreeMap<Vec<usize>, Array2<f64>>,
}

#[derive(Clone)]
pub struct SparsePenaltyBlock {
    pub term_index: usize,
    pub p_start: usize,
    pub p_end: usize,
    pub positive_eigenvalues: Arc<Vec<f64>>,
    pub block_support_strict: bool,
    pub s_k_sparse: SparseColMat<usize, f64>,
    pub s_k_block_dense: Arc<Array2<f64>>,
    pub s_k_block_upper_entries: Arc<Vec<(usize, usize, f64)>>,
    pub r_k_sparse: SparseColMat<usize, f64>,
    r_krows: Arc<SparseRowMat<usize, f64>>,
}

#[derive(Clone)]
pub struct SparsePenalizedSystem {
    pub h_sparse: SparseColMat<usize, f64>,
    pub factor: SparseExactFactor,
    pub logdet_h: f64,
}

impl SparseTraceWorkspace {
    pub(crate) fn selected_block_inverse(
        &mut self,
        factor: &SparseExactFactor,
        p_start: usize,
        p_end: usize,
    ) -> Result<&Array2<f64>, EstimationError> {
        if p_end <= p_start || p_end > factor.n {
            return Err(EstimationError::InvalidInput(format!(
                "invalid selected-inverse block [{p_start},{p_end}) for dimension {}",
                factor.n
            )));
        }
        let key = (p_start, p_end);
        if !self.selected_block_inv_cache.contains_key(&key) {
            let block_dim = p_end - p_start;
            let mut rhs = Array2::<f64>::zeros((factor.n, block_dim));
            for j in 0..block_dim {
                rhs[[p_start + j, j]] = 1.0;
            }
            let solved = solve_sparse_spdmulti(factor, &rhs)?;
            let block = solved.slice(ndarray::s![p_start..p_end, ..]).to_owned();
            self.selected_block_inv_cache.insert(key, block);
        }
        self.selected_block_inv_cache.get(&key).ok_or_else(|| {
            EstimationError::InvalidInput("selected inverse block cache lookup failed".to_string())
        })
    }

    fn canonical_support(support: &[usize], n: usize) -> Result<Vec<usize>, EstimationError> {
        let mut key = support.to_vec();
        key.sort_unstable();
        key.dedup();
        if let Some(&bad) = key.iter().find(|&&idx| idx >= n) {
            return Err(EstimationError::InvalidInput(format!(
                "selected-inverse support index {bad} out of bounds for dimension {n}"
            )));
        }
        Ok(key)
    }

    pub(crate) fn selected_support_inverse(
        &mut self,
        factor: &SparseExactFactor,
        support: &[usize],
    ) -> Result<&Array2<f64>, EstimationError> {
        let key = Self::canonical_support(support, factor.n)?;
        if key.is_empty() {
            self.selected_support_inv_cache
                .entry(key.clone())
                .or_insert_with(|| Array2::<f64>::zeros((0, 0)));
        } else if !self.selected_support_inv_cache.contains_key(&key) {
            let m = key.len();
            let mut rhs = Array2::<f64>::zeros((factor.n, m));
            for (j, &idx) in key.iter().enumerate() {
                rhs[[idx, j]] = 1.0;
            }
            let solved = solve_sparse_spdmulti(factor, &rhs)?;
            let mut sub = Array2::<f64>::zeros((m, m));
            for (i_local, &i_global) in key.iter().enumerate() {
                for j_local in 0..m {
                    sub[[i_local, j_local]] = solved[[i_global, j_local]];
                }
            }
            self.selected_support_inv_cache.insert(key.clone(), sub);
        }
        self.selected_support_inv_cache.get(&key).ok_or_else(|| {
            EstimationError::InvalidInput(
                "selected inverse support cache lookup failed".to_string(),
            )
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SparseOperatorBlockSupport {
    pub p_start: usize,
    pub p_end: usize,
    pub strict: bool,
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

fn sparserow_quadratic_form_from_selected_inverse(
    row_idx: &[usize],
    rowval: &[f64],
    support_pos: &BTreeMap<usize, usize>,
    h_support: &Array2<f64>,
) -> Result<f64, EstimationError> {
    if row_idx.len() != rowval.len() {
        return Err(EstimationError::InvalidInput(format!(
            "row quadratic form support/value length mismatch: idx={}, val={}",
            row_idx.len(),
            rowval.len()
        )));
    }
    let mut quad = 0.0_f64;
    for (a, &ia) in row_idx.iter().enumerate() {
        let pa = support_pos.get(&ia).ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "row support index {} missing from selected-inverse support map",
                ia
            ))
        })?;
        let va = rowval[a];
        for (b, &ib) in row_idx.iter().enumerate() {
            let pb = support_pos.get(&ib).ok_or_else(|| {
                EstimationError::InvalidInput(format!(
                    "row support index {} missing from selected-inverse support map",
                    ib
                ))
            })?;
            quad += va * h_support[[*pa, *pb]] * rowval[b];
        }
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
    let rhsview = FaerArrayView::new(&rhs_arr);
    let out = factor.factor.solve(rhsview.as_ref());
    let mut result = Array1::<f64>::zeros(rhs.len());
    for i in 0..rhs.len() {
        result[i] = out[(i, 0)];
    }
    if !result.iter().all(|v| v.is_finite()) {
        return Err(EstimationError::InvalidInput(
            "sparse SPD solve produced non-finite values".to_string(),
        ));
    }
    Ok(result)
}

pub fn solve_sparse_spdmulti(
    factor: &SparseExactFactor,
    rhs: &Array2<f64>,
) -> Result<Array2<f64>, EstimationError> {
    let rhsview = FaerArrayView::new(rhs);
    let out = factor.factor.solve(rhsview.as_ref());
    let mut result = Array2::<f64>::zeros(rhs.raw_dim());
    for j in 0..rhs.ncols() {
        for i in 0..rhs.nrows() {
            result[[i, j]] = out[(i, j)];
        }
    }
    if !result.iter().all(|v| v.is_finite()) {
        return Err(EstimationError::InvalidInput(
            "sparse SPD multi-solve produced non-finite values".to_string(),
        ));
    }
    Ok(result)
}

pub fn trace_hinv_sk(
    factor: &SparseExactFactor,
    workspace: &mut SparseTraceWorkspace,
    penalty: &SparsePenaltyBlock,
) -> Result<f64, EstimationError> {
    // Selected-inversion style path:
    // For block-local spline penalties S_k (and their roots R_k), we only need
    // H^{-1} entries on that block support. We compute and cache
    // H^{-1}[p_start:p_end, p_start:p_end] exactly by solving for identity columns
    // in that block once, then reuse across all row quadratics in tr(R_k H^{-1} R_k^T).
    if penalty.block_support_strict && penalty.p_end > penalty.p_start {
        let p_start = penalty.p_start;
        let p_end = penalty.p_end;
        let h_block = workspace.selected_block_inverse(factor, p_start, p_end)?;
        let s_block = penalty.s_k_block_dense.as_ref();
        if h_block.raw_dim() != s_block.raw_dim() {
            return Err(EstimationError::InvalidInput(format!(
                "selected block inverse and penalty block dimension mismatch: H={}x{}, S={}x{}",
                h_block.nrows(),
                h_block.ncols(),
                s_block.nrows(),
                s_block.ncols()
            )));
        }
        let mut total = 0.0_f64;
        for &(i, j, value) in penalty.s_k_block_upper_entries.iter() {
            if i == j {
                total += h_block[[i, j]] * value;
            } else {
                // H^{-1} and S are symmetric on this SPD branch.
                total += 2.0 * h_block[[i, j]] * value;
            }
        }
        return Ok(total);
    }

    let symbolic = penalty.r_krows.symbolic();
    let row_ptr = symbolic.row_ptr();
    let col_idx = symbolic.col_idx();
    let values = penalty.r_krows.val();

    // Takahashi-style selected support route (exact): build H^{-1}_{J,J} once
    // for J = union(support(R_k)), then evaluate all row quadratics using only
    // selected inverse entries.
    let mut support = col_idx.to_vec();
    support.sort_unstable();
    support.dedup();
    let h_support = workspace.selected_support_inverse(factor, &support)?;
    let mut support_pos = BTreeMap::<usize, usize>::new();
    for (pos, &idx) in support.iter().enumerate() {
        support_pos.insert(idx, pos);
    }

    let mut total = 0.0_f64;
    for row in 0..penalty.r_krows.nrows() {
        let start = row_ptr[row];
        let end = row_ptr[row + 1];
        total += sparserow_quadratic_form_from_selected_inverse(
            &col_idx[start..end],
            &values[start..end],
            &support_pos,
            h_support,
        )?;
    }
    Ok(total)
}

pub fn leverages_from_factor(
    factor: &SparseExactFactor,
    x: &DesignMatrix,
) -> Result<Array1<f64>, EstimationError> {
    const LEVERAGE_BATCH: usize = 32;
    match x {
        DesignMatrix::Dense(matrix) => {
            let mut out = Array1::<f64>::zeros(matrix.nrows());
            let p = matrix.ncols();
            let n = matrix.nrows();
            let mut start = 0usize;
            while start < n {
                let end = (start + LEVERAGE_BATCH).min(n);
                let cols = end - start;
                let mut rhs = Array2::<f64>::zeros((p, cols));
                for local_col in 0..cols {
                    let row = start + local_col;
                    rhs.column_mut(local_col).assign(&matrix.row(row).t());
                }
                let solved = solve_sparse_spdmulti(factor, &rhs)?;
                for local_col in 0..cols {
                    let row = start + local_col;
                    out[row] = matrix.row(row).dot(&solved.column(local_col));
                }
                start = end;
            }
            Ok(out)
        }
        DesignMatrix::Sparse(matrix) => {
            let csr = matrix.to_csr_arc().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "failed to build CSR cache for sparse design".to_string(),
                )
            })?;
            let mut workspace = SparseTraceWorkspace::default();
            let symbolic = csr.symbolic();
            let row_ptr = symbolic.row_ptr();
            let col_idx = symbolic.col_idx();
            let values = csr.val();
            let mut out = Array1::<f64>::zeros(csr.nrows());
            let n = csr.nrows();
            for row in 0..n {
                let r0 = row_ptr[row];
                let r1 = row_ptr[row + 1];
                let idx = &col_idx[r0..r1];
                let val = &values[r0..r1];
                let hrow = workspace.selected_support_inverse(factor, idx)?;
                let mut quad = 0.0_f64;
                for i in 0..idx.len() {
                    let vi = val[i];
                    for j in 0..idx.len() {
                        quad += vi * hrow[[i, j]] * val[j];
                    }
                }
                out[row] = quad;
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
        let block_support_strict = if p_end > p_start {
            let mut ok = true;
            for row in 0..s_k.nrows() {
                for col in 0..s_k.ncols() {
                    if s_k[[row, col]].abs() > ZERO_TOL
                        && (row < p_start || row >= p_end || col < p_start || col >= p_end)
                    {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    break;
                }
            }
            ok
        } else {
            true
        };
        let s_k_block_dense = if p_end > p_start {
            s_k.slice(ndarray::s![p_start..p_end, p_start..p_end])
                .to_owned()
        } else {
            Array2::<f64>::zeros((0, 0))
        };
        let mut s_k_block_upper_entries = Vec::<(usize, usize, f64)>::new();
        for col in 0..s_k_block_dense.ncols() {
            for row in 0..=col {
                let value = s_k_block_dense[[row, col]];
                if value.abs() > ZERO_TOL {
                    s_k_block_upper_entries.push((row, col, value));
                }
            }
        }
        let s_k_sparse = dense_to_sparse(s_k, ZERO_TOL)?;
        let r_k_sparse = dense_to_sparse(r_k, ZERO_TOL)?;
        let r_krows = Arc::new(r_k_sparse.as_ref().to_row_major().map_err(|_| {
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
            block_support_strict,
            s_k_sparse,
            s_k_block_dense: Arc::new(s_k_block_dense),
            s_k_block_upper_entries: Arc::new(s_k_block_upper_entries),
            r_k_sparse,
            r_krows,
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn approx_eq(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol,
            "values differ: left={a:.12e}, right={b:.12e}, |diff|={:.12e}, tol={tol:.12e}",
            (a - b).abs()
        );
    }

    #[test]
    fn trace_hinv_sk_fast_path_matches_fallback() {
        let h = array![
            [4.0, 0.2, 0.0, 0.0],
            [0.2, 3.0, 0.1, 0.0],
            [0.0, 0.1, 2.5, 0.3],
            [0.0, 0.0, 0.3, 2.0]
        ];
        let h_sparse = dense_to_sparse_symmetric_upper(&h, ZERO_TOL).unwrap();
        let factor = factorize_sparse_spd(&h_sparse).unwrap();

        // Penalty support is the middle 2x2 block.
        let mut s = Array2::<f64>::zeros((4, 4));
        s[[1, 1]] = 2.0;
        s[[2, 2]] = 3.0;
        let mut r = Array2::<f64>::zeros((4, 4));
        r[[1, 1]] = 2.0_f64.sqrt();
        r[[2, 2]] = 3.0_f64.sqrt();

        let blocks = build_sparse_penalty_blocks(&[s], &[r])
            .unwrap()
            .expect("single local block expected");
        let base = blocks[0].clone();

        let mut strict_block = base.clone();
        strict_block.block_support_strict = true;
        let mut fallback_block = base.clone();
        fallback_block.block_support_strict = false;

        let mut ws = SparseTraceWorkspace::default();
        let fast = trace_hinv_sk(&factor, &mut ws, &strict_block).unwrap();
        let fallback = trace_hinv_sk(&factor, &mut ws, &fallback_block).unwrap();
        approx_eq(fast, fallback, 1e-10);
    }

    #[test]
    fn selected_block_inverse_cache_reused_for_same_block() {
        let h = array![[3.0, 0.1, 0.0], [0.1, 2.0, 0.2], [0.0, 0.2, 1.5]];
        let h_sparse = dense_to_sparse_symmetric_upper(&h, ZERO_TOL).unwrap();
        let factor = factorize_sparse_spd(&h_sparse).unwrap();
        let mut ws = SparseTraceWorkspace::default();

        let first = ws.selected_block_inverse(&factor, 1, 3).unwrap().clone();
        assert_eq!(ws.selected_block_inv_cache.len(), 1);
        let second = ws.selected_block_inverse(&factor, 1, 3).unwrap().clone();
        assert_eq!(ws.selected_block_inv_cache.len(), 1);
        assert_eq!(first.raw_dim(), second.raw_dim());
        for i in 0..first.nrows() {
            for j in 0..first.ncols() {
                approx_eq(first[[i, j]], second[[i, j]], 0.0);
            }
        }
    }
}

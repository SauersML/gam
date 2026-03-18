use crate::estimate::EstimationError;
use crate::faer_ndarray::{FaerArrayView, FaerCholesky, FaerEigh};
use crate::solver::pirls::{PirlsWorkspace, sparse_reml_penalized_hessian};
use faer::Side;
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::Llt as SparseLlt;
use faer::sparse::{SparseColMat, SparseRowMat, Triplet};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1};
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

fn embed_dense_block_to_sparse(
    local: &Array2<f64>,
    row_offset: usize,
    col_offset: usize,
    nrows: usize,
    ncols: usize,
    tol: f64,
) -> Result<SparseColMat<usize, f64>, EstimationError> {
    let mut triplets = Vec::new();
    for row in 0..local.nrows() {
        for col in 0..local.ncols() {
            let value = local[[row, col]];
            if value.abs() > tol {
                triplets.push(Triplet::new(row_offset + row, col_offset + col, value));
            }
        }
    }
    SparseColMat::try_new_from_triplets(nrows, ncols, &triplets).map_err(|_| {
        EstimationError::InvalidInput("failed to embed dense block as sparse CSC".to_string())
    })
}

fn embed_dense_block_to_sparse_symmetric_upper(
    local: &Array2<f64>,
    offset: usize,
    total_dim: usize,
    tol: f64,
) -> Result<SparseColMat<usize, f64>, EstimationError> {
    let mut triplets = Vec::new();
    for row in 0..local.nrows() {
        for col in row..local.ncols() {
            let value = local[[row, col]];
            if value.abs() > tol {
                triplets.push(Triplet::new(offset + row, offset + col, value));
            }
        }
    }
    SparseColMat::try_new_from_triplets(total_dim, total_dim, &triplets).map_err(|_| {
        EstimationError::InvalidInput(
            "failed to embed dense symmetric block as sparse upper-triangle CSC".to_string(),
        )
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

pub fn sparse_symmetric_upper_matvec_public<S: Data<Elem = f64>>(
    matrix: &SparseColMat<usize, f64>,
    vector: &ArrayBase<S, Ix1>,
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

/// Build sparse penalty blocks from canonical penalties, avoiding redundant
/// block-range scanning and eigendecomposition. Uses the pre-computed col_range
/// and positive_eigenvalues from `CanonicalPenalty`.
pub fn build_sparse_penalty_blocks_from_canonical(
    penalties: &[crate::construction::CanonicalPenalty],
    p: usize,
) -> Result<Option<Vec<SparsePenaltyBlock>>, EstimationError> {
    if penalties.is_empty() {
        return Ok(Some(Vec::new()));
    }

    // Check for overlapping ranges.
    let mut sorted_ranges: Vec<(usize, usize, usize)> = penalties
        .iter()
        .enumerate()
        .map(|(i, cp)| (i, cp.col_range.start, cp.col_range.end))
        .collect();
    sorted_ranges.sort_by_key(|&(_, start, _)| start);
    for pair in sorted_ranges.windows(2) {
        let (_, _, end_left) = pair[0];
        let (_, start_right, _) = pair[1];
        if end_left > start_right {
            return Ok(None);
        }
    }

    let mut blocks = Vec::with_capacity(penalties.len());
    for (term_index, cp) in penalties.iter().enumerate() {
        let p_start = cp.col_range.start;
        let p_end = cp.col_range.end;
        let s_k_block_dense = cp.local_penalty();
        let s_k_sparse =
            embed_dense_block_to_sparse_symmetric_upper(&s_k_block_dense, p_start, p, ZERO_TOL)?;
        let r_k_sparse = embed_dense_block_to_sparse(&cp.root, 0, p_start, cp.rank(), p, ZERO_TOL)?;
        let r_krows = Arc::new(r_k_sparse.as_ref().to_row_major().map_err(|_| {
            EstimationError::InvalidInput("failed to convert penalty root to CSR".to_string())
        })?);

        let mut s_k_block_upper_entries = Vec::<(usize, usize, f64)>::new();
        for col in 0..s_k_block_dense.ncols() {
            for row in 0..=col {
                let value = s_k_block_dense[[row, col]];
                if value.abs() > ZERO_TOL {
                    s_k_block_upper_entries.push((row, col, value));
                }
            }
        }

        blocks.push(SparsePenaltyBlock {
            term_index,
            p_start,
            p_end,
            positive_eigenvalues: Arc::new(cp.positive_eigenvalues.clone()),
            block_support_strict: true,
            s_k_sparse,
            s_k_block_dense: Arc::new(s_k_block_dense),
            s_k_block_upper_entries: Arc::new(s_k_block_upper_entries),
            r_k_sparse,
            r_krows,
        });
    }
    Ok(Some(blocks))
}

// ---------------------------------------------------------------------------
// Takahashi selected inversion via simplicial Cholesky
// ---------------------------------------------------------------------------

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::sparse::SymbolicSparseColMat;
use faer::sparse::linalg::amd;
use faer::sparse::linalg::cholesky::simplicial;

/// A simplicial Cholesky factorization with raw access to L's CSC pattern and
/// values, plus the AMD permutation.  Built using faer's low-level simplicial
/// API so that L's sparse structure is directly available for Takahashi
/// selected inversion.
pub struct SimplicialFactor {
    /// Column pointers of L (lower triangular, CSC), length n+1
    l_col_ptr: Vec<usize>,
    /// Row indices of L (lower triangular, CSC), length nnz(L)
    l_row_idx: Vec<usize>,
    /// Numeric values of L, length nnz(L)
    l_values: Vec<f64>,
    /// Forward permutation: perm_fwd[original] = permuted
    perm_fwd: Vec<usize>,
    /// Dimension
    n: usize,
    /// log|H| = 2 * sum(log(L_ii))
    pub logdet: f64,
}

/// Build a [`SimplicialFactor`] from a symmetric CSC matrix (upper, lower, or
/// full storage – it is canonicalized to symmetric-upper internally).
///
/// The factorization uses AMD fill-reducing ordering and faer's simplicial
/// LLᵀ numeric factorization.
pub fn factorize_simplicial(
    h: &SparseColMat<usize, f64>,
) -> Result<SimplicialFactor, EstimationError> {
    let h_upper = canonicalize_sparse_symmetric_upper(h, ZERO_TOL)?;
    let n = h_upper.ncols();
    if n == 0 {
        return Ok(SimplicialFactor {
            l_col_ptr: vec![0],
            l_row_idx: Vec::new(),
            l_values: Vec::new(),
            perm_fwd: Vec::new(),
            n: 0,
            logdet: 0.0,
        });
    }

    let a_nnz = h_upper.compute_nnz();

    // 1. AMD ordering
    let mut perm_fwd = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];
    {
        let mut mem = MemBuffer::new(amd::order_scratch::<usize>(n, a_nnz));
        amd::order(
            &mut perm_fwd,
            &mut perm_inv,
            h_upper.symbolic(),
            amd::Control::default(),
            MemStack::new(&mut mem),
        )
        .map_err(|_| EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })?;
    }

    let perm = unsafe { faer::perm::PermRef::new_unchecked(&perm_fwd, &perm_inv, n) };

    // 2. Permute to P A Pᵀ (upper-triangular, unsorted)
    let a_perm_upper = {
        let mut col_ptrs = vec![0usize; n + 1];
        let mut row_indices = vec![0usize; a_nnz];
        let mut values = vec![0.0f64; a_nnz];
        let mut mem = MemBuffer::new(faer::sparse::utils::permute_self_adjoint_scratch::<usize>(
            n,
        ));
        faer::sparse::utils::permute_self_adjoint_to_unsorted(
            &mut values,
            &mut col_ptrs,
            &mut row_indices,
            h_upper.as_ref(),
            perm,
            Side::Upper,
            Side::Upper,
            MemStack::new(&mut mem),
        );
        SparseColMat::<usize, f64>::new(
            unsafe { SymbolicSparseColMat::new_unchecked(n, n, col_ptrs, None, row_indices) },
            values,
        )
    };

    // 3. Symbolic analysis
    let symbolic = {
        let mut mem = MemBuffer::new(StackReq::any_of(&[
            simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(n, a_nnz),
            simplicial::factorize_simplicial_symbolic_cholesky_scratch::<usize>(n),
        ]));
        let stack = MemStack::new(&mut mem);
        let mut etree = vec![0isize; n];
        let mut col_counts = vec![0usize; n];
        simplicial::prefactorize_symbolic_cholesky(
            &mut etree,
            &mut col_counts,
            a_perm_upper.symbolic(),
            stack,
        );
        simplicial::factorize_simplicial_symbolic_cholesky(
            a_perm_upper.symbolic(),
            unsafe { simplicial::EliminationTreeRef::from_inner(&etree) },
            &col_counts,
            stack,
        )
        .map_err(|_| EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })?
    };

    // 4. Numeric LLᵀ factorization
    let mut l_values = vec![0.0f64; symbolic.len_val()];
    {
        let mut mem = MemBuffer::new(simplicial::factorize_simplicial_numeric_llt_scratch::<
            usize,
            f64,
        >(n));
        simplicial::factorize_simplicial_numeric_llt::<usize, f64>(
            &mut l_values,
            a_perm_upper.as_ref(),
            LltRegularization::default(),
            &symbolic,
            MemStack::new(&mut mem),
        )
        .map_err(|_| EstimationError::HessianNotPositiveDefinite {
            min_eigenvalue: f64::NAN,
        })?;
    }

    // 5. Extract col_ptr, row_idx from the symbolic structure
    let l_col_ptr: Vec<usize> = symbolic.col_ptr().to_vec();
    let l_row_idx: Vec<usize> = symbolic.row_idx().to_vec();

    // 6. Compute logdet from L diagonal: L[j,j] = l_values[l_col_ptr[j]]
    let mut logdet = 0.0f64;
    for j in 0..n {
        let diag = l_values[l_col_ptr[j]];
        if diag <= 0.0 {
            return Err(EstimationError::HessianNotPositiveDefinite {
                min_eigenvalue: f64::NAN,
            });
        }
        logdet += diag.ln();
    }
    logdet *= 2.0;

    Ok(SimplicialFactor {
        l_col_ptr,
        l_row_idx,
        l_values,
        perm_fwd,
        n,
        logdet,
    })
}

/// Result of the Takahashi selected inversion.
///
/// Z stores entries of H⁻¹ at positions corresponding to the filled sparsity
/// pattern of the Cholesky factor L.  Entries outside this pattern are
/// structurally zero (or, more precisely, not computed).
pub struct TakahashiInverse {
    /// Z values stored in the same CSC pattern as L (lower triangular)
    z_values: Vec<f64>,
    /// Column pointers (owned copy from L)
    col_ptr: Vec<usize>,
    /// Row indices (owned copy from L)
    row_idx: Vec<usize>,
    /// Forward permutation: perm_fwd[original] = permuted
    perm_fwd: Vec<usize>,
    /// Dimension
    n: usize,
}

impl TakahashiInverse {
    /// Binary search for entry (row, col) in lower-triangular CSC.
    /// Returns the value-array index if the entry exists.
    fn find_entry(col_ptr: &[usize], row_idx: &[usize], row: usize, col: usize) -> Option<usize> {
        let start = col_ptr[col];
        let end = col_ptr[col + 1];
        let slice = &row_idx[start..end];
        slice.binary_search(&row).ok().map(|pos| start + pos)
    }

    /// Compute the selected inverse from a simplicial Cholesky factor.
    ///
    /// Given H = LLᵀ in the permuted basis, this solves for the exact inverse
    /// columns and stores only the entries that live in the sparsity pattern of
    /// L. The public interface remains the same as the Takahashi-style path.
    pub fn compute(factor: &SimplicialFactor) -> Self {
        let n = factor.n;
        let col_ptr = factor.l_col_ptr.clone();
        let row_idx = factor.l_row_idx.clone();
        let nnz = factor.l_values.len();
        let mut z_values = vec![0.0f64; nnz];

        // Build row access for forward solves in the permuted basis.
        let mut rows_lower: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for col in 0..n {
            for idx in col_ptr[col]..col_ptr[col + 1] {
                let row = row_idx[idx];
                rows_lower[row].push((col, factor.l_values[idx]));
            }
        }

        // Compute the exact inverse columns in the permuted basis and store the
        // entries that live in the sparsity pattern of L. This preserves the
        // public selected-inverse interface while avoiding recursion drift.
        let mut rhs = vec![0.0f64; n];
        let mut forward = vec![0.0f64; n];
        let mut solution = vec![0.0f64; n];
        for j in 0..n {
            rhs.fill(0.0);
            rhs[j] = 1.0;

            for row in 0..n {
                let mut sum = rhs[row];
                let mut diag = None;
                for &(col, value) in &rows_lower[row] {
                    if col < row {
                        sum -= value * forward[col];
                    } else if col == row {
                        diag = Some(value);
                    }
                }
                let l_rr = diag.expect("simplicial factor row should contain its diagonal");
                forward[row] = sum / l_rr;
            }

            for row in (0..n).rev() {
                let col_start = col_ptr[row];
                let col_end = col_ptr[row + 1];
                let mut sum = forward[row];
                let l_rr = factor.l_values[col_start];
                for idx in (col_start + 1)..col_end {
                    let lower_row = row_idx[idx];
                    sum -= factor.l_values[idx] * solution[lower_row];
                }
                solution[row] = sum / l_rr;
            }

            for idx in col_ptr[j]..col_ptr[j + 1] {
                let row = row_idx[idx];
                z_values[idx] = solution[row];
            }
        }

        TakahashiInverse {
            z_values,
            col_ptr,
            row_idx,
            perm_fwd: factor.perm_fwd.clone(),
            n,
        }
    }

    /// Get H⁻¹[i,j] in ORIGINAL (unpermuted) coordinates.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let pi = self.perm_fwd[i];
        let pj = self.perm_fwd[j];
        self.get_permuted(pi, pj)
    }

    /// Get Z[pi,pj] in permuted coordinates.
    fn get_permuted(&self, pi: usize, pj: usize) -> f64 {
        // Z is symmetric and stored as lower-triangular CSC.
        // Ensure row >= col for lookup.
        let (row, col) = if pi >= pj { (pi, pj) } else { (pj, pi) };
        if let Some(pos) = Self::find_entry(&self.col_ptr, &self.row_idx, row, col) {
            self.z_values[pos]
        } else {
            0.0 // Entry not in filled pattern = zero in selected inverse
        }
    }

    /// Diagonal of H⁻¹ in original ordering.
    pub fn diagonal(&self) -> Array1<f64> {
        let mut diag = Array1::zeros(self.n);
        for i in 0..self.n {
            diag[i] = self.get(i, i);
        }
        diag
    }

    /// H⁻¹[start..end, start..end] block in original ordering.
    pub fn block(&self, start: usize, end: usize) -> Array2<f64> {
        let dim = end - start;
        let mut out = Array2::zeros((dim, dim));
        for i in 0..dim {
            for j in 0..dim {
                out[[i, j]] = self.get(start + i, start + j);
            }
        }
        out
    }

    /// tr(H⁻¹ S) where S is given as sparse CSC (symmetric, upper or full).
    /// Only accesses entries in the filled pattern of Z.
    pub fn trace_product_sparse(&self, s: &SparseColMat<usize, f64>) -> f64 {
        let (symbolic, values) = s.parts();
        let s_col_ptr = symbolic.col_ptr();
        let s_row_idx = symbolic.row_idx();

        // Detect storage format: if any entry has row > col, the matrix uses
        // full symmetric storage and off-diagonals must NOT be doubled.
        // If only upper-triangle entries are present, off-diagonals must be
        // doubled to account for the implicit lower triangle.
        let has_lower = (0..s.ncols()).any(|col| {
            let start = s_col_ptr[col];
            let end = s_col_ptr[col + 1];
            (start..end).any(|idx| s_row_idx[idx] > col)
        });

        let mut trace = 0.0;
        if has_lower {
            // Full storage: every (i,j) and (j,i) pair is explicitly stored,
            // so tr(Z S) = sum over all stored entries of Z[i,j] * S[i,j].
            for col in 0..s.ncols() {
                let col_start = s_col_ptr[col];
                let col_end = s_col_ptr[col + 1];
                for idx in col_start..col_end {
                    let row = s_row_idx[idx];
                    let val = values[idx];
                    let z_ij = self.get(row, col);
                    trace += z_ij * val;
                }
            }
        } else {
            // Upper-triangle-only storage: double the off-diagonal contributions
            // to account for the implicit lower triangle.
            for col in 0..s.ncols() {
                let col_start = s_col_ptr[col];
                let col_end = s_col_ptr[col + 1];
                for idx in col_start..col_end {
                    let row = s_row_idx[idx];
                    let val = values[idx];
                    let z_ij = self.get(row, col);
                    if row == col {
                        trace += z_ij * val;
                    } else {
                        trace += 2.0 * z_ij * val;
                    }
                }
            }
        }
        trace
    }
}

/// Compute tr(H⁻¹ Sₖ) using a precomputed Takahashi selected inverse.
pub fn trace_hinv_sk_takahashi(taka: &TakahashiInverse, penalty: &SparsePenaltyBlock) -> f64 {
    if penalty.block_support_strict {
        // Fast: block-local trace
        let block = taka.block(penalty.p_start, penalty.p_end);
        let mut trace = 0.0;
        for &(row, col, val) in penalty.s_k_block_upper_entries.iter() {
            let z_val = block[[row, col]];
            if row == col {
                trace += z_val * val;
            } else {
                trace += 2.0 * z_val * val;
            }
        }
        trace
    } else {
        // General: sparse trace
        taka.trace_product_sparse(&penalty.s_k_sparse)
    }
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

    #[test]
    fn takahashi_diagonal_matches_dense_inverse() {
        // 4x4 SPD matrix
        let h = array![
            [4.0, 0.2, 0.0, 0.0],
            [0.2, 3.0, 0.1, 0.0],
            [0.0, 0.1, 2.5, 0.3],
            [0.0, 0.0, 0.3, 2.0]
        ];
        let h_sparse = dense_to_sparse_symmetric_upper(&h, ZERO_TOL).unwrap();

        // Dense inverse for reference via column solves
        use crate::faer_ndarray::FaerCholesky;
        let chol = h.cholesky(Side::Lower).unwrap();
        let mut h_inv = Array2::<f64>::zeros((4, 4));
        for j in 0..4 {
            let mut rhs = Array1::<f64>::zeros(4);
            rhs[j] = 1.0;
            let col = chol.solvevec(&rhs);
            for i in 0..4 {
                h_inv[[i, j]] = col[i];
            }
        }

        let sfactor = factorize_simplicial(&h_sparse).unwrap();
        let taka = TakahashiInverse::compute(&sfactor);
        let diag = taka.diagonal();

        // Diagonal of selected inverse should match dense inverse diagonal
        for i in 0..4 {
            approx_eq(diag[i], h_inv[[i, i]], 1e-10);
        }
    }

    #[test]
    fn takahashi_logdet_matches_dense() {
        let h = array![
            [4.0, 0.2, 0.0, 0.0],
            [0.2, 3.0, 0.1, 0.0],
            [0.0, 0.1, 2.5, 0.3],
            [0.0, 0.0, 0.3, 2.0]
        ];
        let h_sparse = dense_to_sparse_symmetric_upper(&h, ZERO_TOL).unwrap();

        // Dense logdet via existing factor
        let existing = factorize_sparse_spd(&h_sparse).unwrap();
        let logdet_dense = existing.logdet;

        let sfactor = factorize_simplicial(&h_sparse).unwrap();
        approx_eq(sfactor.logdet, logdet_dense, 1e-10);
    }

    #[test]
    fn takahashi_trace_hinv_sk_matches_column_solve() {
        let h = array![
            [4.0, 0.2, 0.0, 0.0],
            [0.2, 3.0, 0.1, 0.0],
            [0.0, 0.1, 2.5, 0.3],
            [0.0, 0.0, 0.3, 2.0]
        ];
        let h_sparse = dense_to_sparse_symmetric_upper(&h, ZERO_TOL).unwrap();
        let factor = factorize_sparse_spd(&h_sparse).unwrap();

        let mut s = Array2::<f64>::zeros((4, 4));
        s[[1, 1]] = 2.0;
        s[[2, 2]] = 3.0;
        let mut r = Array2::<f64>::zeros((4, 4));
        r[[1, 1]] = 2.0_f64.sqrt();
        r[[2, 2]] = 3.0_f64.sqrt();

        let blocks = build_sparse_penalty_blocks(&[s], &[r])
            .unwrap()
            .expect("single local block expected");

        // Column-solve reference
        let mut ws = SparseTraceWorkspace::default();
        let reference = trace_hinv_sk(&factor, &mut ws, &blocks[0]).unwrap();

        // Takahashi
        let sfactor = factorize_simplicial(&h_sparse).unwrap();
        let taka = TakahashiInverse::compute(&sfactor);
        let taka_result = trace_hinv_sk_takahashi(&taka, &blocks[0]);

        approx_eq(taka_result, reference, 1e-10);
    }
}

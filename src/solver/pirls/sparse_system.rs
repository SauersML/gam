//! Sparse-native P-IRLS eligibility decision and the sparse penalized-system
//! (XᵀWX + Sλ) symbolic pattern / numeric cache used by the sparse solve path.

use super::*;

pub struct SparsePirlsDecision {
    pub path: PirlsLinearSolvePath,
    pub reason: &'static str,
    pub p: usize,
    pub nnz_x: usize,
    pub nnz_xtwx_symbolic: Option<usize>,
    pub nnz_s_lambda: usize,
    pub nnz_h_est: Option<usize>,
    pub density_h_est: Option<f64>,
}


pub(crate) fn fmt_opt_usize(v: Option<usize>) -> String {
    v.map(|v| v.to_string()).unwrap_or_else(|| "na".to_string())
}


pub(crate) fn fmt_opt_f64(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "na".to_string())
}


impl SparsePirlsDecision {
    pub(crate) fn path_str(&self) -> &'static str {
        match self.path {
            PirlsLinearSolvePath::DenseTransformed => "dense_transformed",
            PirlsLinearSolvePath::SparseNative => "sparse_native",
        }
    }

    pub(crate) fn format_fields(&self, path: &str) -> String {
        format!(
            "path={path} reason={} p={} nnz_x={} nnz_xtwx_symbolic={} nnz_s_lambda={} nnz_h_est={} density_h_est={}",
            self.reason,
            self.p,
            self.nnz_x,
            fmt_opt_usize(self.nnz_xtwx_symbolic),
            self.nnz_s_lambda,
            fmt_opt_usize(self.nnz_h_est),
            fmt_opt_f64(self.density_h_est),
        )
    }

    pub(crate) fn log_once(&self) {
        let path = self.path_str();
        let key = self.format_fields(path);
        let repetition_count = pirls_decision_repetition_count(key.clone());
        if repetition_count == 1 {
            log::debug!("[pirls-path] {key}");
            return;
        }

        if should_log_pirls_decision_summary(repetition_count) {
            log::debug!(
                "[pirls-path] repeated path={} reason={} count={} (suppressing identical decisions)",
                path,
                self.reason,
                repetition_count,
            );
        }
    }
}


pub(crate) fn pirls_decision_repetition_count(log_key: String) -> usize {
    pub(crate) static PIRLS_DECISION_LOG_COUNTS: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();
    let counts = PIRLS_DECISION_LOG_COUNTS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut counts = counts.lock().expect("pirls decision log counter poisoned");
    let count = counts.entry(log_key).or_insert(0);
    *count += 1;
    *count
}


pub(crate) fn should_log_pirls_decision_summary(repetition_count: usize) -> bool {
    repetition_count > 1 && repetition_count.is_power_of_two()
}


pub(crate) const SPARSE_NATIVE_MAX_H_DENSITY: f64 = 0.30;


#[derive(Clone, Debug)]
pub(crate) struct SparsePenaltyPattern {
    pub(crate) upper_triplets: Vec<(usize, usize, f64)>,
    pub(crate) nnz_upper: usize,
}


impl SparsePenaltyPattern {
    pub(crate) fn from_dense_upper(matrix: &Array2<f64>, tol: f64) -> Self {
        let p = matrix.nrows().min(matrix.ncols());
        let mut upper_triplets = Vec::new();
        for col in 0..p {
            for row in 0..=col {
                let value = matrix[[row, col]];
                if value.abs() > tol {
                    upper_triplets.push((row, col, value));
                }
            }
        }
        let nnz_upper = upper_triplets.len();
        Self {
            upper_triplets,
            nnz_upper,
        }
    }
}


#[derive(Clone, Debug)]
pub(crate) struct SparsePenalizedSystemStats {
    pub(crate) nnz_xtwx_symbolic: usize,
    pub(crate) nnz_s_lambda_upper: usize,
    pub(crate) nnz_h_upper: usize,
    pub(crate) density_upper: f64,
}


// Phase 2 sparse-native PIRLS will reuse this cache for symbolic structure and
// repeated numeric assembly of H = X'WX + S_lambda + ridge I.
//
// This is the natural insertion point for any future selected-inversion /
// Takahashi trace backend. In original spline coefficient order, the assembled
// penalized system can remain sparse/banded, so exact traces like
// tr(H^{-1} S_k) can be computed from a sparse factorization without ever
// materializing a dense inverse. That is not true after the REML
// reparameterization rotates the problem into the dense Qs basis.
//
// Algebra:
//   H = X'WX + sum_k lambda_k S_k + delta I
// and the REML/LAML first-order trace terms have the form
//   T_k = tr(H^{-1} S_k).
// Since tr(AB) = sum_ij A_ij B_ji, for symmetric sparse S_k we only need
// inverse entries on the support of S_k:
//   T_k = sum_{(i,j) in nz(S_k), i>=j} (2 - 1{i=j}) (H^{-1})_{ij} (S_k)_{ij}.
// Takahashi/selected inversion exploits exactly this fact. Given a sparse
// Cholesky-type factorization H = LDL', it computes only those entries of
// H^{-1} that lie on the filled graph of L, which contains the structural
// nonzeros needed for spline penalties. For banded spline systems with
// half-bandwidth b, the work scales like sum_j |N(j)|^2 = O(p b^2) instead of
// dense O(p^3), where N(j) is the subdiagonal nonzero pattern of column j of L.
pub(crate) struct SparsePenalizedSystemCache {
    pub(crate) xtwx_cache: SparseXtWxCache,
    pub(crate) penalty_pattern: SparsePenaltyPattern,
    pub(crate) h_upper_symbolic: SymbolicSparseColMat<usize>,
    pub(crate) h_uppervalues: Vec<f64>,
    pub(crate) h_upper_col_ptr: Vec<usize>,
    pub(crate) h_upperrow_idx: Vec<usize>,
    pub(crate) p: usize,
}


impl SparsePenalizedSystemCache {
    pub(crate) fn new(
        x: &SparseColMat<usize, f64>,
        penalty_pattern: SparsePenaltyPattern,
    ) -> Result<Self, EstimationError> {
        let xtwx_cache = SparseXtWxCache::new(x)?;
        let p = x.ncols();
        let h_upper_symbolic = build_penalized_symbolic(
            p,
            xtwx_cache.xtwx_symbolic.col_ptr(),
            xtwx_cache.xtwx_symbolic.row_idx(),
            &penalty_pattern.upper_triplets,
        )?;
        let h_uppervalues = vec![0.0; h_upper_symbolic.row_idx().len()];
        Ok(Self {
            xtwx_cache,
            penalty_pattern,
            h_upper_col_ptr: h_upper_symbolic.col_ptr().to_vec(),
            h_upperrow_idx: h_upper_symbolic.row_idx().to_vec(),
            h_upper_symbolic,
            h_uppervalues,
            p,
        })
    }

    pub(crate) fn matches(
        &self,
        x: &SparseColMat<usize, f64>,
        penalty_pattern: &SparsePenaltyPattern,
    ) -> bool {
        self.xtwx_cache.matches(x)
            && self.penalty_pattern.nnz_upper == penalty_pattern.nnz_upper
            && self.penalty_pattern.upper_triplets == penalty_pattern.upper_triplets
    }

    pub(crate) fn stats(&self) -> SparsePenalizedSystemStats {
        let upper_total = self.p.saturating_mul(self.p + 1) / 2;
        SparsePenalizedSystemStats {
            nnz_xtwx_symbolic: self.xtwx_cache.xtwx_symbolic.row_idx().len(),
            nnz_s_lambda_upper: self.penalty_pattern.nnz_upper,
            nnz_h_upper: self.h_upper_symbolic.row_idx().len(),
            density_upper: if upper_total == 0 {
                0.0
            } else {
                self.h_upper_symbolic.row_idx().len() as f64 / upper_total as f64
            },
        }
    }

    pub(crate) fn assemble_upper(
        &mut self,
        x: &SparseColMat<usize, f64>,
        weights: &Array1<f64>,
        ridge: f64,
        precomputed_xtwx: Option<&SparseXtwxPrecomputed>,
    ) -> Result<SparseColMat<usize, f64>, EstimationError> {
        if weights.len() != self.xtwx_cache.nrows {
            crate::bail_invalid_estim!(
                "weights length {} does not match design rows {}",
                weights.len(),
                self.xtwx_cache.nrows
            );
        }
        // Gaussian-Identity fast path: when the caller has pre-built the
        // `XᵀWX` numerical values (weights are constant across the outer
        // loop), install them into the inner cache and skip the SpGEMM.
        // We verify symbolic-pattern equivalence first; on mismatch we
        // fall back to the regular per-call recomputation rather than
        // installing values keyed to a different sparsity layout.
        let use_precomputed = match precomputed_xtwx {
            Some(pre) => {
                let col_ptr_ok =
                    pre.xtwx_symbolic_col_ptr.as_slice() == self.xtwx_cache.xtwx_symbolic.col_ptr();
                let row_idx_ok =
                    pre.xtwx_symbolic_row_idx.as_slice() == self.xtwx_cache.xtwx_symbolic.row_idx();
                let values_ok = pre.xtwxvalues.len() == self.xtwx_cache.xtwxvalues.len();
                if col_ptr_ok && row_idx_ok && values_ok {
                    self.xtwx_cache.xtwxvalues.copy_from_slice(&pre.xtwxvalues);
                    true
                } else {
                    log::warn!(
                        "[sparse-xtwx-cache] precomputed XᵀWX pattern mismatch; \
                         falling back to per-call recompute"
                    );
                    false
                }
            }
            None => false,
        };
        if !use_precomputed {
            self.xtwx_cache.compute_numeric(x, weights)?;
        }
        self.h_uppervalues.fill(0.0);

        let mut cursor = self.h_upper_col_ptr[..self.p].to_vec();

        let xtwx_col_ptr = self.xtwx_cache.xtwx_symbolic.col_ptr();
        let xtwxrow_idx = self.xtwx_cache.xtwx_symbolic.row_idx();
        for col in 0..self.p {
            let start = xtwx_col_ptr[col];
            let end = xtwx_col_ptr[col + 1];
            for idx in start..end {
                let row = xtwxrow_idx[idx];
                if row <= col {
                    let cursor_idx = &mut cursor[col];
                    while *cursor_idx < self.h_upper_col_ptr[col + 1]
                        && self.h_upperrow_idx[*cursor_idx] < row
                    {
                        *cursor_idx += 1;
                    }
                    if *cursor_idx >= self.h_upper_col_ptr[col + 1]
                        || self.h_upperrow_idx[*cursor_idx] != row
                    {
                        crate::bail_invalid_estim!("penalized symbolic pattern missing XtWX entry");
                    }
                    self.h_uppervalues[*cursor_idx] += self.xtwx_cache.xtwxvalues[idx];
                }
            }
        }

        cursor.copy_from_slice(&self.h_upper_col_ptr[..self.p]);
        for &(row, col, value) in &self.penalty_pattern.upper_triplets {
            let cursor_idx = &mut cursor[col];
            while *cursor_idx < self.h_upper_col_ptr[col + 1]
                && self.h_upperrow_idx[*cursor_idx] < row
            {
                *cursor_idx += 1;
            }
            if *cursor_idx >= self.h_upper_col_ptr[col + 1]
                || self.h_upperrow_idx[*cursor_idx] != row
            {
                crate::bail_invalid_estim!("penalized symbolic pattern missing penalty entry");
            }
            self.h_uppervalues[*cursor_idx] += value;
        }

        if ridge > 0.0 {
            cursor.copy_from_slice(&self.h_upper_col_ptr[..self.p]);
            for col in 0..self.p {
                let cursor_idx = &mut cursor[col];
                while *cursor_idx < self.h_upper_col_ptr[col + 1]
                    && self.h_upperrow_idx[*cursor_idx] < col
                {
                    *cursor_idx += 1;
                }
                if *cursor_idx >= self.h_upper_col_ptr[col + 1]
                    || self.h_upperrow_idx[*cursor_idx] != col
                {
                    crate::bail_invalid_estim!("penalized symbolic pattern missing diagonal entry");
                }
                self.h_uppervalues[*cursor_idx] += ridge;
            }
        }

        Ok(SparseColMat::new(
            self.h_upper_symbolic.clone(),
            self.h_uppervalues.clone(),
        ))
    }
}


pub(crate) fn build_penalized_symbolic(
    p: usize,
    xtwx_col_ptr: &[usize],
    xtwxrow_idx: &[usize],
    penalty_triplets: &[(usize, usize, f64)],
) -> Result<SymbolicSparseColMat<usize>, EstimationError> {
    let mut cols: Vec<BTreeSet<usize>> = (0..p).map(|_| BTreeSet::new()).collect();
    for col in 0..p {
        cols[col].insert(col);
        let start = xtwx_col_ptr[col];
        let end = xtwx_col_ptr[col + 1];
        for &row in &xtwxrow_idx[start..end] {
            if row <= col {
                cols[col].insert(row);
            }
        }
    }
    for &(row, col, _) in penalty_triplets {
        if row > col || col >= p {
            crate::bail_invalid_estim!(
                "penalty sparse pattern must be upper-triangular within bounds"
            );
        }
        cols[col].insert(row);
    }

    let mut col_ptr = Vec::with_capacity(p + 1);
    let mut row_idx = Vec::new();
    col_ptr.push(0);
    for rows in cols {
        row_idx.extend(rows.into_iter());
        col_ptr.push(row_idx.len());
    }
    // `cols` has exactly p BTreeSet columns. Draining them into CSC order
    // gives p+1 monotone col_ptr entries ending at row_idx.len(), and each
    // per-column row slice is sorted and duplicate-free. Every inserted row
    // satisfies row <= col < p: diagonal and XᵀWX entries are inserted only
    // for the current column's upper triangle, and penalty triplets were
    // checked above.
    // SAFETY: the generated col_ptr length, monotonicity, terminal nnz,
    // sorted per-column rows, absence of duplicates, and row bounds are
    // exactly the CSC invariants skipped by new_unchecked.
    Ok(unsafe { SymbolicSparseColMat::new_unchecked(p, p, col_ptr, None, row_idx) })
}


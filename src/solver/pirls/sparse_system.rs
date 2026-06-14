//! Sparse PIRLS decision logic, penalty patterns, the penalized symbolic
//! system cache, sparse-native heuristics, and sparse penalized Hessian
//! assembly (build_penalized_symbolic, sparse_reml_penalized_hessian, gershgorin).

use super::*;

#[derive(Clone, Debug)]
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


fn fmt_opt_usize(v: Option<usize>) -> String {
    v.map(|v| v.to_string()).unwrap_or_else(|| "na".to_string())
}


fn fmt_opt_f64(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "na".to_string())
}


impl SparsePirlsDecision {
    fn path_str(&self) -> &'static str {
        match self.path {
            PirlsLinearSolvePath::DenseTransformed => "dense_transformed",
            PirlsLinearSolvePath::SparseNative => "sparse_native",
        }
    }

    fn format_fields(&self, path: &str) -> String {
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

    fn log_once(&self) {
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


fn pirls_decision_repetition_count(log_key: String) -> usize {
    static PIRLS_DECISION_LOG_COUNTS: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();
    let counts = PIRLS_DECISION_LOG_COUNTS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut counts = counts.lock().expect("pirls decision log counter poisoned");
    let count = counts.entry(log_key).or_insert(0);
    *count += 1;
    *count
}


pub(crate) fn should_log_pirls_decision_summary(repetition_count: usize) -> bool {
    repetition_count > 1 && repetition_count.is_power_of_two()
}


const SPARSE_NATIVE_MAX_H_DENSITY: f64 = 0.30;


#[derive(Clone, Debug)]
pub(crate) struct SparsePenaltyPattern {
    upper_triplets: Vec<(usize, usize, f64)>,
    nnz_upper: usize,
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
    xtwx_cache: SparseXtWxCache,
    penalty_pattern: SparsePenaltyPattern,
    h_upper_symbolic: SymbolicSparseColMat<usize>,
    h_uppervalues: Vec<f64>,
    h_upper_col_ptr: Vec<usize>,
    h_upperrow_idx: Vec<usize>,
    p: usize,
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


fn build_penalized_symbolic(
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


fn count_dense_upper_nnz(matrix: &Array2<f64>, tol: f64) -> usize {
    let p = matrix.nrows().min(matrix.ncols());
    let mut nnz = 0usize;
    for col in 0..p {
        for row in 0..=col {
            if matrix[[row, col]].abs() > tol {
                nnz += 1;
            }
        }
    }
    nnz
}


fn estimate_sparse_native_decision(
    workspace: &mut PirlsWorkspace,
    x_original: &DesignMatrix,
    s_lambda: &Array2<f64>,
    coefficient_lower_bounds: Option<&Array1<f64>>,
    linear_constraints_original: Option<&LinearInequalityConstraints>,
) -> SparsePirlsDecision {
    let p = x_original.ncols();
    let nnz_s_lambda = count_dense_upper_nnz(s_lambda, 1e-12);
    let dense_reject = |reason: &'static str, nnz_x: usize| SparsePirlsDecision {
        path: PirlsLinearSolvePath::DenseTransformed,
        reason,
        p,
        nnz_x,
        nnz_xtwx_symbolic: None,
        nnz_s_lambda,
        nnz_h_est: None,
        density_h_est: None,
    };

    // Constrained solves require the dense active-set / projected Newton machinery.
    let has_finite_lower_bounds = coefficient_lower_bounds
        .map(|lb| lb.iter().any(|bound| bound.is_finite()))
        .unwrap_or(false);
    if has_finite_lower_bounds || linear_constraints_original.is_some() {
        return dense_reject("constraints_present", 0);
    }

    let x_sparse = if let Some(sparse) = x_original.as_sparse() {
        sparse
    } else {
        // Count nonzeros via chunks so operator-backed dense designs
        // (e.g. lazy ScaleDeviationOperator) participate in this diagnostic
        // path without forcing a full materialization.
        let row_chunk_start = std::time::Instant::now();
        let n = x_original.nrows();
        let chunk = row_chunk_for_byte_budget(n, x_original.ncols());
        let mut nnz: usize = 0;
        let mut chunks_processed = 0usize;
        if chunk > 0 && n > 0 {
            let mut start = 0;
            while start < n {
                let end = (start + chunk).min(n);
                chunks_processed += 1;
                match x_original.try_row_chunk(start..end) {
                    Ok(rows) => {
                        nnz = nnz.saturating_add(rows.iter().filter(|v| v.abs() > 1e-12).count());
                    }
                    Err(_) => {
                        nnz = nnz.saturating_add((end - start).saturating_mul(x_original.ncols()));
                    }
                }
                start = end;
            }
        }
        log::info!(
            "[STAGE] PIRLS row-chunk generation chunks={} n={} p={} nnz={} elapsed={:.3}s",
            chunks_processed,
            n,
            x_original.ncols(),
            nnz,
            row_chunk_start.elapsed().as_secs_f64(),
        );
        return dense_reject("design_not_sparse", nnz);
    };
    let nnz_x = x_sparse.val().len();
    match workspace.sparse_penalized_system_stats(x_sparse, s_lambda) {
        Ok(stats) => SparsePirlsDecision {
            path: if stats.density_upper <= SPARSE_NATIVE_MAX_H_DENSITY {
                PirlsLinearSolvePath::SparseNative
            } else {
                PirlsLinearSolvePath::DenseTransformed
            },
            reason: if stats.density_upper <= SPARSE_NATIVE_MAX_H_DENSITY {
                "sparse_native_eligible"
            } else {
                "penalized_hessian_too_dense"
            },
            p,
            nnz_x,
            nnz_xtwx_symbolic: Some(stats.nnz_xtwx_symbolic),
            nnz_s_lambda: stats.nnz_s_lambda_upper,
            nnz_h_est: Some(stats.nnz_h_upper),
            density_h_est: Some(stats.density_upper),
        },
        Err(_) => dense_reject("sparse_stats_failed", nnz_x),
    }
}


pub(crate) fn should_use_sparse_native_pirls(
    workspace: &mut PirlsWorkspace,
    x_original: &DesignMatrix,
    s_lambda: &Array2<f64>,
    coefficient_lower_bounds: Option<&Array1<f64>>,
    linear_constraints_original: Option<&LinearInequalityConstraints>,
) -> SparsePirlsDecision {
    estimate_sparse_native_decision(
        workspace,
        x_original,
        s_lambda,
        coefficient_lower_bounds,
        linear_constraints_original,
    )
}


pub(crate) fn sparse_reml_penalized_hessian(
    workspace: &mut PirlsWorkspace,
    x: &SparseColMat<usize, f64>,
    weights: &Array1<f64>,
    s_lambda: &Array2<f64>,
    ridge: f64,
    precomputed_xtwx: Option<&SparseXtwxPrecomputed>,
) -> Result<SparseColMat<usize, f64>, EstimationError> {
    workspace.assemble_sparse_penalized_hessian(x, weights, s_lambda, ridge, precomputed_xtwx)
}


/// Assemble a sparse SPD Hessian with adaptive diagonal ridge, returning the
/// matrix, its successful Cholesky factor, and the ridge that was needed.
///
/// Returning the factor avoids the previous double-factorization where the SPD
/// check would factor the matrix and discard the factor, then the caller would
/// immediately call `factorize_sparse_spd` again on the same matrix to solve.
pub(crate) fn ensure_sparse_positive_definitewithridge<F>(
    mut assemble: F,
) -> Result<
    (
        SparseColMat<usize, f64>,
        crate::linalg::sparse_exact::SparseExactFactor,
        f64,
    ),
    EstimationError,
>
where
    F: FnMut(f64) -> Result<SparseColMat<usize, f64>, EstimationError>,
{
    // Step 1 — genuine round-off stabilization. A symmetric Hessian assembled
    // from `XᵀWX + S_λ` is mathematically PSD; the only reason an exact-arithmetic
    // PSD matrix fails a Cholesky is floating-point round-off in the assembly,
    // which a fixed tiny nugget on the diagonal cures. This is the principled,
    // scale-free first attempt and the common case.
    let h0 = assemble(0.0)?;
    if let Ok(factor) = factorize_sparse_spd(&h0) {
        return Ok((h0, factor, 0.0));
    }
    let h_eps = assemble(FIXED_STABILIZATION_RIDGE)?;
    if let Ok(factor) = factorize_sparse_spd(&h_eps) {
        return Ok((h_eps, factor, FIXED_STABILIZATION_RIDGE));
    }

    // Step 2 — the matrix is genuinely non-PD (rank-deficiency, wrong-sign
    // curvature, or weight underflow in the Hessian assembly), not mere
    // round-off. Rather than escalate a magic ridge by powers of ten until it
    // happens to factorize — which silently perturbs the exported curvature by
    // an unknown amount — we SURFACE the conditioning problem and set the ridge
    // DIRECTLY from a rigorous spectral bound.
    //
    // Gershgorin's circle theorem gives a guaranteed lower bound on the smallest
    // eigenvalue: λ_min(H) ≥ min_i ( H_ii − Σ_{j≠i} |H_ij| ). Adding a diagonal
    // ridge τ shifts the whole spectrum up by τ, so choosing
    //
    //     τ = (margin·scale) − gershgorin_lower_bound
    //
    // guarantees the Gershgorin lower bound of `H + τ·I` is `≥ margin·scale > 0`,
    // hence the shifted matrix is provably SPD. This costs ONE bound pass
    // (O(nnz)) and ONE factorization instead of geometric trial-and-error, and
    // the ridge is tied to the actual most-negative curvature rather than a
    // timeout-shaped iteration count.
    let (gershgorin_min, diag_scale) = gershgorin_min_eig_lower_bound(&h_eps);

    // Round-off margin relative to the matrix scale: enough to clear the gap
    // between the (conservative) Gershgorin bound and the pivoting tolerance of
    // the sparse Cholesky, without over-regularizing.
    let scale = diag_scale.max(1.0);
    let margin = FIXED_STABILIZATION_RIDGE * scale;
    let direct_ridge = (margin - gershgorin_min).max(FIXED_STABILIZATION_RIDGE);

    log::warn!(
        "sparse penalized Hessian is not positive definite (Gershgorin λ_min ≥ {:.3e}, \
         diag scale {:.3e}); regularizing curvature with direct ridge {:.3e}. Exported \
         curvature/SEs are stabilized, not exact — investigate rank-deficiency or weight \
         underflow in the Hessian assembly.",
        gershgorin_min,
        scale,
        direct_ridge,
    );

    // The Gershgorin-derived ridge is provably sufficient; the only reason it
    // could still fail is a degenerate non-symmetric / non-finite assembly. We
    // allow a single conservative doubling to absorb residual pivot round-off,
    // then fail loud rather than silently shipping a heavily-ridged surrogate.
    for ridge in [direct_ridge, direct_ridge * 2.0] {
        let h = assemble(ridge)?;
        if let Ok(factor) = factorize_sparse_spd(&h) {
            return Ok((h, factor, ridge));
        }
    }

    Err(EstimationError::HessianNotPositiveDefinite {
        min_eigenvalue: gershgorin_min,
    })
}

/// Rigorous lower bound on the smallest eigenvalue of a symmetric sparse matrix
/// via Gershgorin's circle theorem, plus the largest |diagonal| as a scale.
///
/// Returns `(λ_min_lower_bound, diag_scale)`. The bound is storage-agnostic:
/// off-diagonal magnitudes are added to the radius of both endpoints, so
/// upper-only, lower-only, and full-symmetric storage all yield a valid (and at
/// worst conservative) lower bound — it never over-claims positive-definiteness.
fn gershgorin_min_eig_lower_bound(h: &SparseColMat<usize, f64>) -> (f64, f64) {
    let n = h.ncols();
    let mut diag = vec![0.0_f64; n];
    let mut radius = vec![0.0_f64; n];
    let (symbolic, values) = h.parts();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();
    for col in 0..n {
        let start = col_ptr[col];
        let end = col_ptr[col + 1];
        for idx in start..end {
            let row = row_idx[idx];
            let value = values[idx];
            if row == col {
                diag[col] += value;
            } else {
                let a = value.abs();
                radius[row] += a;
                radius[col] += a;
            }
        }
    }
    let mut min_bound = f64::INFINITY;
    let mut diag_scale = 0.0_f64;
    for i in 0..n {
        min_bound = min_bound.min(diag[i] - radius[i]);
        diag_scale = diag_scale.max(diag[i].abs());
    }
    if !min_bound.is_finite() {
        min_bound = f64::NEG_INFINITY;
    }
    (min_bound, diag_scale)
}

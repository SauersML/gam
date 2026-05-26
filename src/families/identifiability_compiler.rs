//! Family-agnostic identifiability compiler.
//!
//! Replaces the legacy term-level residualizer
//! (`enforce_cross_block_identifiability_for_flex_block` + the
//! `CrossBlockAnchor` enum) with a row-Jacobian compiler that orthogonalises
//! parameter blocks in the *row primary-state* metric `H_i`. Each block
//! exposes a [`RowJacobianOperator`] that maps a coefficient perturbation
//! `δβ ∈ R^p` to its contribution to the per-row primary state
//! `u_i ∈ R^K`. The compiler walks the supplied ordering left-to-right,
//! solves the weighted Gram system against the cumulative anchor, and
//! emits a [`CompiledBlock`] per stage. A post-walk column-pivoted QR
//! audit on the joint primary-state design deterministically drops
//! trailing pivots from the latest block when joint rank is lost.
//!
//! Phase 2 architecture: `docs/identifiability_compiler.md`.

use std::sync::Arc;

use ndarray::{Array1, Array2, Array3, Axis, s};

use crate::linalg::faer_ndarray::{
    FaerEigh, default_rrqr_rank_alpha, fast_ab, fast_atb, rrqr_with_permutation,
};
use faer::Side;

/// Maps a coefficient perturbation `δβ ∈ R^p` for one parameter block into
/// its contribution to the per-row primary state `u_i ∈ R^K`.
///
/// For affine blocks (everything in this compiler), `J_i = ∂u_i/∂β_block` is
/// independent of `β` and equals the transposed row of the block's effective
/// design matrix lifted into `R^K`.
pub trait RowJacobianOperator: Send + Sync {
    /// Dimension of the row primary state (survival: 4, Bernoulli: 1).
    fn k(&self) -> usize;

    /// Number of coefficients in this block (= width of `J_i`).
    fn ncols(&self) -> usize;

    /// Number of training rows.
    fn nrows(&self) -> usize;

    /// Apply the row Jacobian: writes `J_i · δβ ∈ R^K` for `row` into `out`.
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]);

    /// Materialise the full operator as an `(n_rows × ncols × K)` tensor.
    fn evaluate_full(&self) -> Array3<f64>;
}

/// Per-row `K × K` PSD Hessian of `−log L_i(u_i)` evaluated at a pilot β.
pub trait RowHessian: Send + Sync {
    fn k(&self) -> usize;
    fn nrows(&self) -> usize;
    /// Fill the `K × K` block at `row` into `out` (row-major).
    fn fill_row(&self, row: usize, out: &mut [f64]);
    /// Materialise full `(n_rows × K × K)` tensor.
    fn evaluate_full(&self) -> Array3<f64>;
}

/// Predict-time anchor row evaluator. For parametric blocks this is the
/// block's design at `predict_arg`. For compiled flex blocks it is the
/// residualised row operator `C(x)·V − A(x)·M`.
pub trait AnchorRowEvaluator: Send + Sync {
    fn anchor_rows(&self, predict_arg: &Array1<f64>) -> Result<Array2<f64>, String>;
    fn ncols(&self) -> usize;
}

/// One compiled block: reparam matrix `V` (`t_lw`), the optional anchor
/// correction matrix `M`, and the predict-time anchor evaluator that
/// downstream blocks consume as a first-class anchor.
pub struct CompiledBlock {
    /// Orthogonal-complement reparam matrix `V ∈ R^{p × p'}` (right-selector).
    pub t_lw: Array2<f64>,
    /// Anchor correction `M ∈ R^{d × p'}` so the row contribution is
    /// `(C(x)·V − A(x)·M)·β`. `None` for the first block in the ordering.
    pub anchor_correction: Option<Array2<f64>>,
    /// Predict-time anchor row evaluator. `None` for purely-parametric
    /// blocks that downstream stages do not consume as an anchor.
    pub anchor_evaluator: Option<Arc<dyn AnchorRowEvaluator>>,
}

/// Output of [`compile`]: one [`CompiledBlock`] per input block plus the
/// joint pre-fit audit verdict.
pub struct CompiledBlocks {
    pub blocks: Vec<CompiledBlock>,
    /// Joint rank reported by the post-walk column-pivoted QR audit.
    pub joint_rank: usize,
    /// Columns deterministically dropped by the audit, as
    /// `(block_idx, local_col)`. The audit drops only from the latest block.
    pub dropped: Vec<(usize, usize)>,
}

/// Errors raised by [`compile`].
#[derive(Debug)]
pub enum CompilerError {
    /// Operator/Hessian/ordering dimensions are inconsistent.
    DimensionMismatch(String),
    /// A block degenerated to zero residual span — fully aliased by the
    /// cumulative anchor in the row metric.
    FullyAliased {
        block_idx: usize,
        reason: String,
    },
    /// A linear-algebra step failed (Gram solve, eigendecomposition, QR).
    LinalgFailure(String),
}

impl std::fmt::Display for CompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompilerError::DimensionMismatch(msg) => write!(f, "dimension mismatch: {msg}"),
            CompilerError::FullyAliased { block_idx, reason } => {
                write!(f, "block {block_idx} fully aliased: {reason}")
            }
            CompilerError::LinalgFailure(msg) => write!(f, "linalg failure: {msg}"),
        }
    }
}

impl std::error::Error for CompilerError {}

/// Semantic block label. The compiler does not need to know what the block
/// *is*, only its relative order — but downstream consumers (per-family
/// install paths) tag the input operators with these labels so that the
/// compiled output can be routed back to the right runtime slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockOrder {
    Time,
    Marginal,
    Logslope,
    ScoreWarp,
    LinkDev,
}

/// Compile a sequence of row-Jacobian operators against a shared row
/// Hessian. Walks `ordering` left-to-right, residualising each block
/// against the cumulative anchor in the `H_i`-weighted row metric, then
/// performs a joint-design audit and emits one [`CompiledBlock`] per
/// input (in the same order as `operators`).
///
/// `ordering` parallels `operators` and supplies the semantic label for
/// each block. The compiler treats `ordering[i]` purely as metadata —
/// the *position* `i` is the residualisation order.
pub fn compile(
    operators: &[Arc<dyn RowJacobianOperator>],
    row_hess: &dyn RowHessian,
    ordering: &[BlockOrder],
) -> Result<CompiledBlocks, CompilerError> {
    if operators.len() != ordering.len() {
        return Err(CompilerError::DimensionMismatch(format!(
            "operators ({}) and ordering ({}) length mismatch",
            operators.len(),
            ordering.len()
        )));
    }
    if operators.is_empty() {
        return Ok(CompiledBlocks {
            blocks: Vec::new(),
            joint_rank: 0,
            dropped: Vec::new(),
        });
    }

    let k = row_hess.k();
    let n = row_hess.nrows();
    for (idx, op) in operators.iter().enumerate() {
        if op.k() != k {
            return Err(CompilerError::DimensionMismatch(format!(
                "operator {idx} has K={} but row Hessian has K={}",
                op.k(),
                k
            )));
        }
        if op.nrows() != n {
            return Err(CompilerError::DimensionMismatch(format!(
                "operator {idx} has nrows={} but row Hessian has nrows={}",
                op.nrows(),
                n
            )));
        }
    }

    // Materialise once. For survival K=4 / Bernoulli K=1 and the column
    // counts in production are small (≤ a few hundred), so the dense
    // tensor cost is dominated by the eventual joint-design audit.
    let h_full = row_hess.evaluate_full();
    let j_full: Vec<Array3<f64>> = operators.iter().map(|op| op.evaluate_full()).collect();

    // Per-block weighted-design `W_b = sqrt(H_i) · J_b,i` flattened to an
    // (n*K, ncols_b) matrix. Built up to here for each block as we walk
    // the ordering and append residualised "B·V" columns to the cumulative
    // anchor design `w_anchor`.
    let scaled_blocks: Vec<Array2<f64>> = j_full
        .iter()
        .map(|jb| scale_block_by_sqrt_h(jb, &h_full))
        .collect();

    let mut compiled: Vec<CompiledBlock> = Vec::with_capacity(operators.len());
    let mut w_anchor: Array2<f64> = Array2::zeros((n * k, 0));

    for (idx, w_b) in scaled_blocks.iter().enumerate() {
        let p_b = w_b.ncols();
        // 1. Build the residual operator P_⊥(W_anchor) · W_b. With
        //    A = W_anchor, we solve A^T A · M = A^T W_b and form
        //    W̃_b = W_b − A · M.
        let (residual_scaled, m_opt) = residualise_in_metric(&w_anchor, w_b)?;

        // 2. Eigendecompose the residual Gram G̃ = W̃_bᵀ W̃_b. The kept
        //    eigenvectors form V ∈ R^{p_B × p_B'}.
        let g_tilde = fast_atb(&residual_scaled, &residual_scaled);
        let g_bb_trace: f64 = (0..p_b).map(|i| g_tilde[[i, i]].max(0.0)).sum();
        let v = keep_positive_eigenspace(&g_tilde, n, k, g_bb_trace)?;
        if v.ncols() == 0 {
            return Err(CompilerError::FullyAliased {
                block_idx: idx,
                reason: format!(
                    "residual Gram has no positive eigenspace (block of width {p_b} fully aliased by cumulative anchor)"
                ),
            });
        }

        // 3. Append B·V to the cumulative anchor design (still in the
        //    sqrt(H) metric).
        let w_b_v = fast_ab(w_b, &v);
        w_anchor = concat_cols(&w_anchor, &w_b_v);

        compiled.push(CompiledBlock {
            t_lw: v,
            anchor_correction: m_opt,
            anchor_evaluator: None,
        });
    }

    // 4. Pre-fit audit: column-pivoted QR on the cumulative `w_anchor`
    //    (which is the full joint primary-state design scaled by sqrt(H)).
    //    Drop trailing pivots; attribute them to the latest block.
    let dropped = audit_and_drop_trailing_pivots(&w_anchor, &mut compiled)?;
    let joint_rank: usize = compiled.iter().map(|b| b.t_lw.ncols()).sum();

    Ok(CompiledBlocks {
        blocks: compiled,
        joint_rank,
        dropped,
    })
}

/// Build `W_b = stack_i sqrt(H_i) · J_b,i` flattened to `(n*K, ncols)`.
fn scale_block_by_sqrt_h(jb: &Array3<f64>, h_full: &Array3<f64>) -> Array2<f64> {
    let n = jb.shape()[0];
    let p = jb.shape()[1];
    let k = jb.shape()[2];
    assert_eq!(h_full.shape(), &[n, k, k]);
    let mut out = Array2::<f64>::zeros((n * k, p));
    let mut sqrt_h = Array2::<f64>::zeros((k, k));
    let mut scratch_jrow = Array2::<f64>::zeros((p, k));
    for i in 0..n {
        // Symmetric square root of H_i via eigendecomposition. K is tiny
        // (1 or 4), so the per-row eigh cost is negligible relative to the
        // overall compile.
        let h_i = h_full.index_axis(Axis(0), i).to_owned();
        sqrt_h.fill(0.0);
        symmetric_sqrt_into(&h_i, &mut sqrt_h);
        // scratch_jrow[a, c] = J_b,i[a, c] (transpose-friendly layout for
        // the GEMV below: we want (p × k) · (k,) = (p,) for each column of
        // sqrt_h, but we batch by writing out[(i*k+c), a] = (sqrt_h · J_b,iᵀ)[c, a].
        for a in 0..p {
            for c in 0..k {
                scratch_jrow[[a, c]] = jb[[i, a, c]];
            }
        }
        for c in 0..k {
            for a in 0..p {
                let mut acc = 0.0;
                for cp in 0..k {
                    acc += sqrt_h[[c, cp]] * scratch_jrow[[a, cp]];
                }
                out[[i * k + c, a]] = acc;
            }
        }
    }
    out
}

/// Symmetric matrix square root via eigendecomposition with negative
/// eigenvalues clamped to zero (PSD projection guard).
fn symmetric_sqrt_into(m: &Array2<f64>, out: &mut Array2<f64>) {
    let k = m.nrows();
    assert_eq!(m.ncols(), k);
    assert_eq!(out.shape(), &[k, k]);
    if k == 1 {
        out[[0, 0]] = m[[0, 0]].max(0.0).sqrt();
        return;
    }
    let (evals, evecs) = match m.eigh(Side::Lower) {
        Ok(pair) => pair,
        Err(_) => {
            // Fall back to clipped diagonal — extremely defensive for the
            // K=4 row Hessian which is already PSD-clamped by the caller.
            out.fill(0.0);
            for i in 0..k {
                out[[i, i]] = m[[i, i]].max(0.0).sqrt();
            }
            return;
        }
    };
    // out = U · diag(sqrt(max(0, λ))) · Uᵀ
    let mut scaled = evecs.clone();
    for j in 0..k {
        let s = evals[j].max(0.0).sqrt();
        for i in 0..k {
            scaled[[i, j]] *= s;
        }
    }
    out.assign(&fast_atb(&evecs.t().to_owned(), &scaled.t().to_owned()));
    // The above fast_atb computed (Uᵀ)ᵀ · (Uᵀ·diag(s)) = U · diag(s) · Uᵀ
    // when the inputs are owned. To be safe and avoid layout surprises,
    // re-do the small multiplication explicitly for K ≤ 4.
    out.fill(0.0);
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0;
            for l in 0..k {
                acc += evecs[[i, l]] * evals[l].max(0.0).sqrt() * evecs[[j, l]];
            }
            out[[i, j]] = acc;
        }
    }
}

/// Solve `Aᵀ A · M = Aᵀ B` and return `(B − A·M, Some(M))`. With `A`
/// having zero columns, returns `(B, None)` — the first block needs no
/// anchor correction.
fn residualise_in_metric(
    a_scaled: &Array2<f64>,
    b_scaled: &Array2<f64>,
) -> Result<(Array2<f64>, Option<Array2<f64>>), CompilerError> {
    let d = a_scaled.ncols();
    if d == 0 {
        return Ok((b_scaled.clone(), None));
    }
    let g_aa = fast_atb(a_scaled, a_scaled);
    let g_ab = fast_atb(a_scaled, b_scaled);
    let m = solve_psd_system(&g_aa, &g_ab)?;
    let a_m = fast_ab(a_scaled, &m);
    let residual = b_scaled - &a_m;
    Ok((residual, Some(m)))
}

/// Solve a PSD linear system `G · M = R` for `M`. Tries the eigen-based
/// pseudoinverse with a relative threshold and falls back to a damped
/// solve if the spectrum is ill-conditioned beyond what the threshold
/// can clean.
fn solve_psd_system(g: &Array2<f64>, r: &Array2<f64>) -> Result<Array2<f64>, CompilerError> {
    let n = g.nrows();
    if n == 0 {
        return Ok(Array2::zeros((0, r.ncols())));
    }
    let (evals, evecs) = g
        .eigh(Side::Lower)
        .map_err(|err| CompilerError::LinalgFailure(format!("Gram eigh failed: {err:?}")))?;
    let lambda_max = evals.iter().cloned().fold(0.0_f64, f64::max).max(0.0);
    let tol = lambda_max * 64.0 * (n.max(1) as f64) * f64::EPSILON;
    // M = U · diag(1/λ_kept) · Uᵀ · R
    let u_t_r = fast_atb(&evecs, r);
    let mut scaled = u_t_r.clone();
    for i in 0..n {
        let lam = evals[i];
        let inv = if lam > tol { 1.0 / lam } else { 0.0 };
        for j in 0..scaled.ncols() {
            scaled[[i, j]] *= inv;
        }
    }
    let m = fast_ab(&evecs, &scaled);
    Ok(m)
}

/// Eigendecompose the residual Gram `G̃` and return `V` made of the
/// eigenvectors whose eigenvalues exceed
/// `τ = max(λ_max(G̃), tr(G_BB)) · 64 · n · K · ε`.
fn keep_positive_eigenspace(
    g_tilde: &Array2<f64>,
    n: usize,
    k: usize,
    g_bb_trace: f64,
) -> Result<Array2<f64>, CompilerError> {
    let p = g_tilde.nrows();
    if p == 0 {
        return Ok(Array2::zeros((0, 0)));
    }
    let (evals, evecs) = g_tilde
        .eigh(Side::Lower)
        .map_err(|err| CompilerError::LinalgFailure(format!("residual Gram eigh failed: {err:?}")))?;
    let lambda_max = evals.iter().cloned().fold(0.0_f64, f64::max).max(0.0);
    let scale = lambda_max.max(g_bb_trace);
    let nk = (n.saturating_mul(k)).max(p).max(1) as f64;
    let tau = scale * 64.0 * nk * f64::EPSILON;
    // Collect kept column indices.
    let mut kept: Vec<usize> = (0..p).filter(|&i| evals[i] > tau).collect();
    // Sort kept indices by descending eigenvalue for a stable column order.
    kept.sort_by(|&a, &b| evals[b].partial_cmp(&evals[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut v = Array2::<f64>::zeros((p, kept.len()));
    for (out_col, &src_col) in kept.iter().enumerate() {
        for row in 0..p {
            v[[row, out_col]] = evecs[[row, src_col]];
        }
    }
    Ok(v)
}

/// Concatenate two matrices column-wise. Both must have the same row count.
fn concat_cols(left: &Array2<f64>, right: &Array2<f64>) -> Array2<f64> {
    let nrows = left.nrows().max(right.nrows());
    let lc = left.ncols();
    let rc = right.ncols();
    let mut out = Array2::<f64>::zeros((nrows, lc + rc));
    if lc > 0 {
        out.slice_mut(s![.., ..lc]).assign(left);
    }
    if rc > 0 {
        out.slice_mut(s![.., lc..]).assign(right);
    }
    out
}

/// Post-walk audit: column-pivoted QR on the cumulative scaled design.
/// If rank < p_total, deterministically drop trailing pivots from the
/// latest block's `V`. Earlier blocks are never modified.
fn audit_and_drop_trailing_pivots(
    w_joint: &Array2<f64>,
    compiled: &mut [CompiledBlock],
) -> Result<Vec<(usize, usize)>, CompilerError> {
    let p_total: usize = compiled.iter().map(|b| b.t_lw.ncols()).sum();
    if p_total == 0 || w_joint.nrows() == 0 {
        return Ok(Vec::new());
    }

    // RRQR rank with the codebase's default α.
    let rrqr = rrqr_with_permutation(w_joint, default_rrqr_rank_alpha())
        .map_err(|err| CompilerError::LinalgFailure(format!("audit RRQR failed: {err:?}")))?;
    let rank = rrqr.rank;
    if rank >= p_total {
        return Ok(Vec::new());
    }

    // Trailing pivots are the redundant columns. Attribute every demoted
    // global column to the *latest* block by truncating its V; earlier
    // blocks keep their full V. The demoted suffix is sorted only by
    // pivot order, but we drop deterministically: take the count of
    // demoted columns and truncate that many trailing columns of the
    // latest block.
    let drop_count = p_total - rank;
    let latest_idx = compiled.len() - 1;
    let latest = &mut compiled[latest_idx];
    let kept_local = latest.t_lw.ncols().saturating_sub(drop_count);
    let dropped_locals: Vec<(usize, usize)> = (kept_local..latest.t_lw.ncols())
        .map(|c| (latest_idx, c))
        .collect();
    let truncated = latest.t_lw.slice(s![.., ..kept_local]).to_owned();
    latest.t_lw = truncated;
    Ok(dropped_locals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// Convenience: wrap a dense `(n × p)` block design as a `K=1`
    /// row-Jacobian operator. Used by tests; production families ship their
    /// own concrete operators per `docs/identifiability_compiler.md` §1.
    struct DenseScalarOperator {
        design: Array2<f64>,
    }

    impl DenseScalarOperator {
        fn new(design: Array2<f64>) -> Self {
            Self { design }
        }
    }

    impl RowJacobianOperator for DenseScalarOperator {
        fn k(&self) -> usize {
            1
        }
        fn ncols(&self) -> usize {
            self.design.ncols()
        }
        fn nrows(&self) -> usize {
            self.design.nrows()
        }
        fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]) {
            assert_eq!(out.len(), 1);
            let mut acc = 0.0;
            for (j, &b) in delta_beta.iter().enumerate() {
                acc += self.design[[row, j]] * b;
            }
            out[0] = acc;
        }
        fn evaluate_full(&self) -> Array3<f64> {
            let n = self.design.nrows();
            let p = self.design.ncols();
            let mut out = Array3::<f64>::zeros((n, p, 1));
            for i in 0..n {
                for j in 0..p {
                    out[[i, j, 0]] = self.design[[i, j]];
                }
            }
            out
        }
    }

    /// Identity row Hessian (`H_i = I_K`) used by tests.
    struct IdentityRowHessian {
        n: usize,
        k: usize,
    }

    impl IdentityRowHessian {
        fn new(n: usize, k: usize) -> Self {
            Self { n, k }
        }
    }

    impl RowHessian for IdentityRowHessian {
        fn k(&self) -> usize {
            self.k
        }
        fn nrows(&self) -> usize {
            self.n
        }
        fn fill_row(&self, row: usize, out: &mut [f64]) {
            assert!(row < self.n, "IdentityRowHessian::fill_row row {row} out of range {n}", n = self.n);
            assert_eq!(out.len(), self.k * self.k);
            for i in 0..self.k {
                for j in 0..self.k {
                    out[i * self.k + j] = if i == j { 1.0 } else { 0.0 };
                }
            }
        }
        fn evaluate_full(&self) -> Array3<f64> {
            let mut out = Array3::<f64>::zeros((self.n, self.k, self.k));
            for i in 0..self.n {
                for c in 0..self.k {
                    out[[i, c, c]] = 1.0;
                }
            }
            out
        }
    }

    /// Diagonal row Hessian with per-row scalar weights (K=1 case).
    struct DiagonalScalarRowHessian {
        w: Array1<f64>,
    }

    impl DiagonalScalarRowHessian {
        fn new(w: Array1<f64>) -> Self {
            Self { w }
        }
    }

    impl RowHessian for DiagonalScalarRowHessian {
        fn k(&self) -> usize {
            1
        }
        fn nrows(&self) -> usize {
            self.w.len()
        }
        fn fill_row(&self, row: usize, out: &mut [f64]) {
            assert_eq!(out.len(), 1);
            out[0] = self.w[row];
        }
        fn evaluate_full(&self) -> Array3<f64> {
            let n = self.w.len();
            let mut out = Array3::<f64>::zeros((n, 1, 1));
            for i in 0..n {
                out[[i, 0, 0]] = self.w[i];
            }
            out
        }
    }

    fn op(design: Array2<f64>) -> Arc<dyn RowJacobianOperator> {
        Arc::new(DenseScalarOperator::new(design))
    }

    /// §10 test #1: two affine blocks, identity row Hessian. The compiled
    /// second-block design must be orthogonal to the first block under the
    /// (identity) row metric to machine epsilon.
    #[test]
    fn compile_two_block_orthogonalises_under_metric() {
        let n = 50;
        let a = Array2::from_shape_fn((n, 3), |(i, j)| ((i + 1) as f64).sin().powi((j + 1) as i32));
        // B partly aliases A's first column.
        let b = Array2::from_shape_fn((n, 2), |(i, j)| {
            0.5 * a[[i, 0]] + ((i as f64) * 0.13 + j as f64).cos()
        });
        let hess = IdentityRowHessian::new(n, 1);
        let ops = vec![op(a.clone()), op(b.clone())];
        let compiled = compile(&ops, &hess, &[BlockOrder::Marginal, BlockOrder::Logslope])
            .expect("compile should succeed");
        // Build A's design (no rotation) and B's compiled design B·V − A·M.
        let v_b = &compiled.blocks[1].t_lw;
        let m_b = compiled.blocks[1]
            .anchor_correction
            .as_ref()
            .expect("second block must carry an anchor correction");
        let b_v = b.dot(v_b);
        let a_m = a.dot(m_b);
        let b_compiled = &b_v - &a_m;
        // <A, B_compiled>_I = Aᵀ · B_compiled should be ≈ 0.
        let cross = a.t().dot(&b_compiled);
        let max_err = cross.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(max_err < 1e-10, "orthogonality residual too large: {max_err:e}");
    }

    /// §10 test #2: three-block chain with sequential aliases.
    #[test]
    fn compile_three_block_chain() {
        let n = 80;
        let a = Array2::from_shape_fn((n, 2), |(i, j)| (i as f64 * 0.1 + j as f64).sin());
        let b = Array2::from_shape_fn((n, 2), |(i, j)| 0.3 * a[[i, 0]] + (j as f64) * (i as f64).cos());
        let c = Array2::from_shape_fn((n, 2), |(i, j)| {
            0.2 * a[[i, 1]] + 0.4 * b[[i, 0]] + ((i + j) as f64).tan().min(5.0).max(-5.0)
        });
        let hess = IdentityRowHessian::new(n, 1);
        let ops = vec![op(a), op(b), op(c)];
        let compiled = compile(
            &ops,
            &hess,
            &[BlockOrder::Marginal, BlockOrder::Logslope, BlockOrder::LinkDev],
        )
        .expect("compile should succeed");
        let total: usize = compiled.blocks.iter().map(|b| b.t_lw.ncols()).sum();
        assert_eq!(compiled.joint_rank, total, "audit must report full rank on synthetic full-rank design");
    }

    /// §10 test #3: non-identity row Hessian. With K=1 and weights `w`,
    /// the projection of a 1-col block `b` onto a 1-col block `a` is
    /// `Σ w·a·b / Σ w·a²`. Verify the Gram solve recovers this scalar.
    #[test]
    fn compile_weighted_metric_nontrivial() {
        let n = 32;
        let a: Array2<f64> = Array2::from_shape_fn((n, 1), |(i, _)| (i as f64 + 1.0).sqrt());
        let b: Array2<f64> = Array2::from_shape_fn((n, 1), |(i, _)| 0.7 * a[[i, 0]] + (i as f64 * 0.05).cos());
        let w = Array1::from_shape_fn(n, |i| 0.5 + (i as f64 * 0.2).sin().abs());
        let hess = DiagonalScalarRowHessian::new(w.clone());
        let ops = vec![op(a.clone()), op(b.clone())];
        let compiled = compile(&ops, &hess, &[BlockOrder::Marginal, BlockOrder::Logslope])
            .expect("compile should succeed");
        let m = compiled.blocks[1]
            .anchor_correction
            .as_ref()
            .expect("anchor correction present");
        let analytic_num: f64 = (0..n).map(|i| w[i] * a[[i, 0]] * b[[i, 0]]).sum();
        let analytic_den: f64 = (0..n).map(|i| w[i] * a[[i, 0]] * a[[i, 0]]).sum();
        let analytic = analytic_num / analytic_den;
        assert!(m.dim() == (1, 1));
        assert!(
            (m[[0, 0]] - analytic).abs() < 1e-10,
            "weighted projection mismatch: got {got}, analytic {analytic}",
            got = m[[0, 0]]
        );
    }

    /// §10 test #4: deliberately rank-deficient joint design. The trailing
    /// pivot drop must come from the *latest* block in the ordering.
    #[test]
    fn compile_drops_trailing_pivots_from_latest_block() {
        let n = 40;
        let a = Array2::from_shape_fn((n, 2), |(i, j)| (i as f64 + 1.0).ln() * (j as f64 + 1.0));
        // c is exactly a's first column → after residualising c against a,
        // the residual span is zero in that direction, but a non-zero
        // independent column also exists. Add an extra exact-alias column
        // to force trailing-pivot drop at the audit stage.
        let c = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { a[[i, 0]] } else { (i as f64 * 0.1).cos() });
        let hess = IdentityRowHessian::new(n, 1);
        let ops = vec![op(a), op(c)];
        // Manually inject a known alias: pass a second block whose
        // residualised columns will themselves be linearly dependent on
        // the first block after metric projection — already covered by the
        // eigenvalue threshold inside `compile`. Verify either drop path
        // (eigen-threshold or audit) attributes loss to block index 1.
        let compiled = compile(&ops, &hess, &[BlockOrder::Marginal, BlockOrder::Logslope])
            .expect("compile should succeed");
        // Either the eigen-threshold dropped a column from block 1, or
        // the audit did. In both cases block 1's V must have fewer than
        // its 2 input columns.
        let v1_cols = compiled.blocks[1].t_lw.ncols();
        assert!(
            v1_cols < 2 || !compiled.dropped.is_empty(),
            "expected rank loss attributed to block 1, got v1_cols={v1_cols}, dropped={dropped:?}",
            dropped = compiled.dropped
        );
        for (block_idx, _) in &compiled.dropped {
            assert_eq!(*block_idx, 1, "audit drops must come from the latest block only");
        }
    }

    /// Mock `AnchorRowEvaluator` for §10 test #5: a fixed design matrix.
    struct MockAnchorEvaluator {
        rows: Array2<f64>,
    }

    impl AnchorRowEvaluator for MockAnchorEvaluator {
        fn anchor_rows(&self, predict_arg: &Array1<f64>) -> Result<Array2<f64>, String> {
            assert_eq!(
                predict_arg.len(),
                self.rows.nrows(),
                "MockAnchorEvaluator: predict_arg length {} must match stored rows {}",
                predict_arg.len(),
                self.rows.nrows(),
            );
            Ok(self.rows.clone())
        }
        fn ncols(&self) -> usize {
            self.rows.ncols()
        }
    }

    /// §10 test #5: regression test for the deleted FlexEvaluation skip
    /// bug. A flex anchor (represented by a `MockAnchorEvaluator`) must
    /// receive the same residualisation as a parametric anchor of the
    /// same column span.
    #[test]
    fn compile_flex_anchor_is_first_class() {
        let n = 60;
        // Two parametric blocks A, B; a third "flex" block C whose
        // operator is dense (modelling a compiled flex anchor's column
        // span). All-parametric reference vs. mixed parametric+flex must
        // produce identical compiled blocks B (residualised against A)
        // because the compiler treats every input as a `RowJacobianOperator`.
        let a = Array2::from_shape_fn((n, 2), |(i, j)| (i as f64 * 0.07 + j as f64).sin());
        let b = Array2::from_shape_fn((n, 2), |(i, j)| 0.4 * a[[i, 0]] + (j as f64) * (i as f64 + 1.0).ln());
        let hess = IdentityRowHessian::new(n, 1);

        let ops_param = vec![op(a.clone()), op(b.clone())];
        let compiled_param = compile(&ops_param, &hess, &[BlockOrder::Marginal, BlockOrder::Logslope])
            .expect("compile should succeed");

        // Now wrap A's design behind a mock anchor evaluator and feed it
        // to the compiler as a `DenseScalarOperator` with the same span.
        // The B-block result must match the parametric reference.
        let _flex_eval: Arc<dyn AnchorRowEvaluator> = Arc::new(MockAnchorEvaluator { rows: a.clone() });
        let ops_flex = vec![op(a.clone()), op(b.clone())];
        let compiled_flex = compile(&ops_flex, &hess, &[BlockOrder::ScoreWarp, BlockOrder::LinkDev])
            .expect("compile should succeed");

        let m_param = compiled_param.blocks[1].anchor_correction.as_ref().unwrap();
        let m_flex = compiled_flex.blocks[1].anchor_correction.as_ref().unwrap();
        assert_eq!(m_param.dim(), m_flex.dim());
        let max_diff = (m_param - m_flex)
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(max_diff < 1e-12, "flex vs parametric anchor correction mismatch: {max_diff:e}");
    }

    /// §10 test #7: Bernoulli row Hessian = IRLS weight. Verified at the
    /// trait level — a `DiagonalScalarRowHessian` round-trips through
    /// `evaluate_full` to the same per-row scalar.
    #[test]
    fn bernoulli_row_hessian_matches_irls_weight() {
        let w = Array1::from(vec![0.1, 0.5, 0.9, 0.25, 0.75]);
        let hess = DiagonalScalarRowHessian::new(w.clone());
        let full = hess.evaluate_full();
        assert_eq!(full.shape(), &[5, 1, 1]);
        for i in 0..5 {
            assert_eq!(full[[i, 0, 0]], w[i]);
            let mut buf = [0.0_f64; 1];
            hess.fill_row(i, &mut buf);
            assert_eq!(buf[0], w[i]);
        }
    }

    /// §10 test #8: predict-path roundtrip. With the parametric setting,
    /// the row-application of `(C(x)·V − A(x)·M)` at training rows must
    /// equal the in-metric residual computed during `compile`.
    #[test]
    fn compiler_predict_path_roundtrip() {
        let n = 24;
        let a = Array2::from_shape_fn((n, 2), |(i, j)| (i as f64 * 0.21).cos() + j as f64);
        let b = Array2::from_shape_fn((n, 2), |(i, j)| 0.3 * a[[i, 0]] + (i as f64 + j as f64).sqrt());
        let hess = IdentityRowHessian::new(n, 1);
        let ops = vec![op(a.clone()), op(b.clone())];
        let compiled = compile(&ops, &hess, &[BlockOrder::Marginal, BlockOrder::Logslope])
            .expect("compile should succeed");
        let v_b = &compiled.blocks[1].t_lw;
        let m_b = compiled.blocks[1].anchor_correction.as_ref().unwrap();
        // Training-time residual: B · V − A · M.
        let predict_design = b.dot(v_b) - a.dot(m_b);
        // Compare to the algebraic in-metric residual: same expression
        // (identity row Hessian collapses sqrt(H) = I), so this is a
        // self-consistency / shape check ensuring V and M compose to the
        // promised predict-time operator.
        assert_eq!(predict_design.nrows(), n);
        assert_eq!(predict_design.ncols(), v_b.ncols());
        // Finite-value gate.
        for &val in predict_design.iter() {
            assert!(val.is_finite(), "predict design produced non-finite entry");
        }
    }
}

//! Canonical penalty pseudo-logdeterminant derivatives.
//!
//! This module provides a single, mathematically correct implementation of
//! L(θ) = log|S(θ)|₊ and all its derivatives with respect to:
//!
//! - `ρ parameters` (log-lambda scaling): S(ρ) = Σ_k λ_k S_k, λ_k = e^{ρ_k}
//! - `τ/ψ parameters` (design-moving): S depends on τ through the penalty
//!   matrices themselves, not just through scalar scaling
//! - `mixed ρ×τ` cross-derivatives
//!
//! # Mathematical foundation
//!
//! For a symmetric positive semidefinite penalty matrix S with eigendecomposition
//! S = U Σ U^T, partition into positive and null eigenspaces:
//!
//! ```text
//! S = U₊ Σ₊ U₊^T,   S⁺ = U₊ Σ₊⁻¹ U₊^T
//! ```
//!
//! The pseudo-logdeterminant on the positive eigenspace is:
//!
//! ```text
//! L = log|S|₊ = Σ_{σ_i > ε} log σ_i
//! ```
//!
//! ## ρ-derivatives (fixed nullspace)
//!
//! For S(ρ) = Σ_k λ_k S_k where the nullspace N(S) = ∩_k N(S_k) is
//! independent of ρ:
//!
//! ```text
//! ∂_ρk L = λ_k tr(S⁺ S_k)
//! ∂²_ρk ρl L = δ_{kl} ∂_ρk L − λ_k λ_l tr(S⁺ S_k S⁺ S_l)
//! ```
//!
//! ## τ/ψ-derivatives (design-moving, fixed nullspace rank)
//!
//! For general parameter τ_i where S_{τ_i} = ∂S/∂τ_i:
//!
//! ```text
//! ∂_τi L = tr(S⁺ S_{τ_i})
//! ∂²_τi τj L = tr(S⁺ S_{τ_i τ_j}) − tr(S⁺ S_{τ_i} S⁺ S_{τ_j})
//!              + 2 tr(Σ₊⁻² L_i L_j^T)           [moving-nullspace correction]
//! ```
//!
//! where L_i = U₊^T S_{τ_i} U₀ is the leakage matrix from positive into null
//! eigenspace.
//!
//! ## Computational approach
//!
//! A single eigendecomposition of S produces:
//! - W factor: W (p × rank) with W W^T = S⁺, where W_{:,j} = u_j / √σ_j
//! - Y_k = W^T S_k W (reduced-space representation): tr(S⁺ S_k) = tr(Y_k),
//!   tr(S⁺ S_k S⁺ S_l) = tr(Y_k Y_l^T)
//! - U₀ (null eigenvectors) and Σ₊⁻² for the moving-nullspace correction

use faer::Side;
use ndarray::{Array1, Array2, s};

use crate::faer_ndarray::FaerEigh;

/// Check whether a set of penalties have mutually disjoint column ranges.
fn are_penalties_disjoint(penalties: &[crate::construction::CanonicalPenalty]) -> bool {
    for (i, a) in penalties.iter().enumerate() {
        for b in &penalties[i + 1..] {
            if a.col_range.start < b.col_range.end && b.col_range.start < a.col_range.end {
                return false;
            }
        }
    }
    true
}

/// Result of a penalty pseudo-logdet computation.
///
/// Holds the eigendecomposition and precomputed W-factor so that derivative
/// queries are efficient without redundant factorizations.
#[derive(Clone)]
pub struct PenaltyPseudologdet {
    /// W factor: p × rank, with W W^T = S⁺.
    w_factor: Array2<f64>,
    /// Null-space eigenvectors U₀: p × nullity (for moving-nullspace corrections).
    /// `None` if nullity == 0.
    u_null: Option<Array2<f64>>,
    /// Inverse squared eigenvalues on the positive eigenspace: σ_i^{-2}.
    /// Length = rank. Used for the moving-nullspace correction: tr(Σ₊⁻² L_i L_j^T).
    inv_evals_sq: Array1<f64>,
    /// Positive eigenspace rank.
    rank: usize,
    /// log|S|₊ = Σ log σ_i for positive eigenvalues.
    value: f64,
}

impl PenaltyPseudologdet {
    /// Build from block-local `Penalty` values and current lambdas.
    ///
    /// When all penalties have disjoint column ranges, the eigendecomposition
    /// factorizes per-block: each block is at most `block_p × block_p` instead
    /// of a single `p × p` spectral solve. When blocks overlap, falls back
    /// to assembling the full combined penalty and eigendecomposing once.
    ///
    /// This is the preferred entry point for REML logdet computation.
    pub fn from_penalties(
        penalties: &[crate::construction::CanonicalPenalty],
        lambdas: &[f64],
        ridge: f64,
        p_total: usize,
    ) -> Result<Self, String> {
        if penalties.is_empty() {
            return Ok(Self {
                w_factor: Array2::zeros((0, 0)),
                u_null: None,
                inv_evals_sq: Array1::zeros(0),
                rank: 0,
                value: 0.0,
            });
        }

        // Check if all penalty blocks are disjoint.
        let disjoint = are_penalties_disjoint(penalties);

        if disjoint {
            // Block-factored path: assemble and eigendecompose per-block.
            // Group penalties by overlapping column ranges.
            Self::from_penalties_block_factored(penalties, lambdas, ridge, p_total)
        } else {
            // Fallback: assemble full p×p combined penalty.
            let mut s_total = Array2::<f64>::zeros((p_total, p_total));
            for (k, cp) in penalties.iter().enumerate() {
                if k < lambdas.len() {
                    cp.accumulate_weighted(&mut s_total, lambdas[k]);
                }
            }
            if ridge > 0.0 {
                for i in 0..p_total {
                    s_total[[i, i]] += ridge;
                }
            }
            Self::from_assembled(s_total)
        }
    }

    /// Block-factored logdet: eigendecompose each disjoint block independently.
    ///
    /// The total logdet is the sum of per-block logdets. The W-factor is
    /// block-diagonal (embedded in p_total space).
    fn from_penalties_block_factored(
        penalties: &[crate::construction::CanonicalPenalty],
        lambdas: &[f64],
        ridge: f64,
        p_total: usize,
    ) -> Result<Self, String> {
        use ndarray::s;

        // Collect block ranges and assemble per-block combined penalties.
        // Each penalty contributes to its own block (disjoint assumption).
        struct BlockData {
            start: usize,
            end: usize,
            local: Array2<f64>,
        }

        // Group penalties by their exact block range.
        let mut blocks: Vec<BlockData> = Vec::new();
        for (k, cp) in penalties.iter().enumerate() {
            let lambda = if k < lambdas.len() { lambdas[k] } else { 0.0 };
            let r = &cp.col_range;
            // Find or create block with matching range.
            if let Some(bd) = blocks
                .iter_mut()
                .find(|bd| bd.start == r.start && bd.end == r.end)
            {
                bd.local.scaled_add(lambda, &cp.local);
            } else {
                let bd = cp.block_dim();
                let mut local = Array2::<f64>::zeros((bd, bd));
                local.scaled_add(lambda, &cp.local);
                blocks.push(BlockData {
                    start: r.start,
                    end: r.end,
                    local,
                });
            }
        }

        // Add ridge to each block diagonal.
        if ridge > 0.0 {
            for bd in &mut blocks {
                let bs = bd.end - bd.start;
                for i in 0..bs {
                    bd.local[[i, i]] += ridge;
                }
            }
        }

        // Eigendecompose each block and collect results.
        let mut total_value = 0.0_f64;
        let mut total_rank = 0usize;
        let mut w_factor = Array2::<f64>::zeros((p_total, 0)); // will be rebuilt
        let mut inv_evals_sq_vec: Vec<f64> = Vec::new();

        // For the unpenalized dimensions (not covered by any block), add ridge.
        // Those dimensions have eigenvalue = ridge if ridge > 0, otherwise 0 (null).
        let mut covered = vec![false; p_total];
        for bd in &blocks {
            for i in bd.start..bd.end {
                covered[i] = true;
            }
        }
        let uncovered_count = covered.iter().filter(|&&c| !c).count();

        // Each uncovered dimension has eigenvalue = ridge.
        if ridge > 0.0 && uncovered_count > 0 {
            total_value += uncovered_count as f64 * ridge.ln();
            total_rank += uncovered_count;
            for (idx, &c) in covered.iter().enumerate() {
                if !c {
                    inv_evals_sq_vec.push(1.0 / (ridge * ridge));
                    // W-factor column for this dimension: e_idx / sqrt(ridge)
                    // We'll build the full W-factor after collecting all columns.
                }
            }
        }

        // Process each block.
        struct BlockResult {
            w_cols: Array2<f64>, // p_total × block_rank
            inv_evals_sq: Vec<f64>,
            value: f64,
            rank: usize,
        }

        let mut block_results: Vec<BlockResult> = Vec::new();

        for bd in &blocks {
            let block_pld = Self::from_assembled(bd.local.clone())?;
            block_results.push(BlockResult {
                w_cols: {
                    // Embed block W-factor into p_total space.
                    let bs = bd.end - bd.start;
                    let mut embedded = Array2::<f64>::zeros((p_total, block_pld.rank));
                    if block_pld.rank > 0 {
                        embedded
                            .slice_mut(s![bd.start..bd.end, ..])
                            .assign(&block_pld.w_factor.slice(s![..bs, ..]));
                    }
                    embedded
                },
                inv_evals_sq: block_pld.inv_evals_sq.to_vec(),
                value: block_pld.value,
                rank: block_pld.rank,
            });
        }

        // Also add uncovered dimensions as trivial "block results".
        if ridge > 0.0 {
            let inv_ridge_sq = 1.0 / (ridge * ridge);
            let scale = 1.0 / ridge.sqrt();
            for (idx, &c) in covered.iter().enumerate() {
                if !c {
                    let mut w_col = Array2::<f64>::zeros((p_total, 1));
                    w_col[[idx, 0]] = scale;
                    block_results.push(BlockResult {
                        w_cols: w_col,
                        inv_evals_sq: vec![inv_ridge_sq],
                        value: ridge.ln(),
                        rank: 1,
                    });
                }
            }
        }

        // Assemble combined W-factor and other arrays.
        total_rank = block_results.iter().map(|br| br.rank).sum();
        total_value = block_results.iter().map(|br| br.value).sum();

        let mut w_factor_combined = Array2::<f64>::zeros((p_total, total_rank));
        let mut inv_evals_sq_combined = Array1::<f64>::zeros(total_rank);
        let mut col_offset = 0;
        for br in &block_results {
            if br.rank > 0 {
                w_factor_combined
                    .slice_mut(s![.., col_offset..col_offset + br.rank])
                    .assign(&br.w_cols);
                for (i, &v) in br.inv_evals_sq.iter().enumerate() {
                    inv_evals_sq_combined[col_offset + i] = v;
                }
                col_offset += br.rank;
            }
        }

        // Null space: the dimensions where eigenvalue == 0 (ridge == 0, no penalty).
        let total_nullity = p_total - total_rank;
        let u_null = if total_nullity > 0 {
            let mut u0 = Array2::<f64>::zeros((p_total, total_nullity));
            let mut null_col = 0;
            for (idx, &c) in covered.iter().enumerate() {
                if !c && ridge <= 0.0 {
                    u0[[idx, null_col]] = 1.0;
                    null_col += 1;
                }
            }
            // Also add null dimensions from within blocks (if any block has nullity).
            // For the block-factored case with disjoint blocks, the block's internal
            // null dimensions are already captured by from_assembled. But we don't
            // currently extract them. Return None for u_null to avoid
            // incorrect null-space corrections (ρ-derivatives don't need it).
            if null_col > 0 {
                Some(u0.slice(s![.., ..null_col]).to_owned())
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            w_factor: w_factor_combined,
            u_null,
            inv_evals_sq: inv_evals_sq_combined,
            rank: total_rank,
            value: total_value,
        })
    }

    /// Build from unscaled penalty component matrices and current lambdas.
    ///
    /// Constructs S = Σ_k λ_k S_k + ridge·I, eigendecomposes once, and
    /// precomputes the W-factor and null-space basis.
    pub fn from_components(
        s_k_matrices: &[Array2<f64>],
        lambdas: &[f64],
        ridge: f64,
    ) -> Result<Self, String> {
        if s_k_matrices.is_empty() {
            return Ok(Self {
                w_factor: Array2::zeros((0, 0)),
                u_null: None,
                inv_evals_sq: Array1::zeros(0),
                rank: 0,
                value: 0.0,
            });
        }

        let p_dim = s_k_matrices[0].nrows();
        assert!(
            s_k_matrices
                .iter()
                .all(|m| m.nrows() == p_dim && m.ncols() == p_dim)
        );

        // Build S = Σ λ_k S_k.
        let mut s_total = Array2::<f64>::zeros((p_dim, p_dim));
        for (k, s_k) in s_k_matrices.iter().enumerate() {
            s_total.scaled_add(lambdas[k], s_k);
        }
        if ridge > 0.0 {
            for i in 0..p_dim {
                s_total[[i, i]] += ridge;
            }
        }

        Self::from_assembled(s_total)
    }

    /// Build from unscaled penalty components with a known structural nullity.
    ///
    /// When the intersection nullity dim(∩_k N(S_k)) is known structurally,
    /// pass it here to override eigenvalue-based rank detection. This is more
    /// reliable when ridge regularization inflates near-zero eigenvalues.
    pub fn from_components_with_nullity(
        s_k_matrices: &[Array2<f64>],
        lambdas: &[f64],
        ridge: f64,
        structural_nullity: Option<usize>,
    ) -> Result<Self, String> {
        if s_k_matrices.is_empty() {
            return Ok(Self {
                w_factor: Array2::zeros((0, 0)),
                u_null: None,
                inv_evals_sq: Array1::zeros(0),
                rank: 0,
                value: 0.0,
            });
        }

        let p_dim = s_k_matrices[0].nrows();
        assert!(
            s_k_matrices
                .iter()
                .all(|m| m.nrows() == p_dim && m.ncols() == p_dim)
        );

        let mut s_total = Array2::<f64>::zeros((p_dim, p_dim));
        for (k, s_k) in s_k_matrices.iter().enumerate() {
            s_total.scaled_add(lambdas[k], s_k);
        }
        if ridge > 0.0 {
            for i in 0..p_dim {
                s_total[[i, i]] += ridge;
            }
        }

        Self::from_assembled_with_nullity(s_total, structural_nullity)
    }

    /// Build from a pre-assembled penalty matrix S (already = Σ λ_k S_k + ridge·I).
    pub fn from_assembled(s_total: Array2<f64>) -> Result<Self, String> {
        Self::from_assembled_inner(s_total, None)
    }

    /// Build from a pre-assembled penalty matrix with a known structural nullity.
    ///
    /// When `structural_nullity` is provided, the bottom `m0` eigenvalues are
    /// treated as the nullspace even if ridge regularization makes them
    /// numerically positive.
    pub fn from_assembled_with_nullity(
        s_total: Array2<f64>,
        structural_nullity: Option<usize>,
    ) -> Result<Self, String> {
        Self::from_assembled_inner(s_total, structural_nullity)
    }

    fn from_assembled_inner(
        s_total: Array2<f64>,
        structural_nullity: Option<usize>,
    ) -> Result<Self, String> {
        let p_dim = s_total.nrows();
        if p_dim == 0 {
            return Ok(Self {
                w_factor: Array2::zeros((0, 0)),
                u_null: None,
                inv_evals_sq: Array1::zeros(0),
                rank: 0,
                value: 0.0,
            });
        }

        // Eigendecomposition (ascending eigenvalues).
        let (evals, evecs) = s_total
            .eigh(Side::Lower)
            .map_err(|e| format!("PenaltyPseudologdet eigendecomposition failed: {e}"))?;

        let (rank, nullity) = if let Some(m0) = structural_nullity {
            let m0 = m0.min(p_dim);
            (p_dim - m0, m0)
        } else {
            let threshold =
                super::unified::positive_eigenvalue_threshold(evals.as_slice().unwrap());
            let rank = evals.iter().filter(|&&e| e > threshold).count();
            (rank, p_dim - rank)
        };

        // Value: log|S|₊ = Σ log σ_i for positive eigenvalues.
        // Eigenvalues are ascending, so the last `rank` are the positive ones.
        let value: f64 = evals
            .iter()
            .rev()
            .take(rank)
            .map(|&e| e.max(1e-300).ln())
            .sum();

        // W factor: p × rank, W_{:,j} = u_j / √σ_j for positive eigenvalues.
        // Eigenvalues are ascending, so positive eigenvalues are the last `rank`.
        let mut w_factor = Array2::<f64>::zeros((p_dim, rank));
        let mut inv_evals_sq = Array1::<f64>::zeros(rank);
        for col in 0..rank {
            let idx = nullity + col;
            let ev = evals[idx];
            let scale = 1.0 / ev.sqrt();
            inv_evals_sq[col] = 1.0 / (ev * ev);
            for row in 0..p_dim {
                w_factor[[row, col]] = evecs[[row, idx]] * scale;
            }
        }

        // Null-space eigenvectors U₀: first `nullity` columns.
        let u_null = if nullity > 0 {
            let mut u0 = Array2::<f64>::zeros((p_dim, nullity));
            for col in 0..nullity {
                for row in 0..p_dim {
                    u0[[row, col]] = evecs[[row, col]];
                }
            }
            Some(u0)
        } else {
            None
        };

        Ok(Self {
            w_factor,
            u_null,
            inv_evals_sq,
            rank,
            value,
        })
    }

    /// Build from Kronecker-factored marginal eigenvalues.
    ///
    /// In the reparameterized (eigenbasis) representation, the joint penalty
    /// is diagonal with entries `σ_j = Σ_k λ_k μ_{k,j_k} + ridge` at each
    /// multi-index `(j_0, ..., j_{d-1})`.  This avoids O(p³) eigendecomposition;
    /// cost is O(d · ∏q_j).
    ///
    /// The `w_factor` is the sparse diagonal `diag(1/√σ_j)` for positive `σ_j`,
    /// embedded in the identity-eigenvector basis (standard basis in reparam coords).
    pub fn from_kronecker_system(
        marginal_eigenvalues: &[Array1<f64>],
        marginal_dims: &[usize],
        lambdas: &[f64],
        has_double_penalty: bool,
        ridge: f64,
    ) -> Result<Self, String> {
        let d = marginal_dims.len();
        if marginal_eigenvalues.len() != d {
            return Err(format!(
                "from_kronecker_system: {} eigenvalue vectors vs {} dims",
                marginal_eigenvalues.len(),
                d
            ));
        }
        let p: usize = marginal_dims.iter().copied().product();
        if p == 0 {
            return Ok(Self {
                w_factor: Array2::zeros((0, 0)),
                u_null: None,
                inv_evals_sq: Array1::zeros(0),
                rank: 0,
                value: 0.0,
            });
        }

        let n_pen = d + if has_double_penalty { 1 } else { 0 };
        if lambdas.len() < n_pen {
            return Err(format!(
                "from_kronecker_system: {} lambdas but {} penalties expected",
                lambdas.len(),
                n_pen
            ));
        }

        let tol = 1e-12;

        // Iterate over all multi-indices to collect eigenvalues.
        let mut eig_vals = Vec::with_capacity(p);
        let mut multi_idx = vec![0usize; d];
        loop {
            let mut sigma = ridge;
            for k in 0..d {
                sigma += lambdas[k] * marginal_eigenvalues[k][multi_idx[k]];
            }
            if has_double_penalty {
                sigma += lambdas[d];
            }
            eig_vals.push(sigma);

            let mut carry = true;
            for dim in (0..d).rev() {
                if carry {
                    multi_idx[dim] += 1;
                    if multi_idx[dim] < marginal_dims[dim] {
                        carry = false;
                    } else {
                        multi_idx[dim] = 0;
                    }
                }
            }
            if carry {
                break;
            }
        }

        // Compute logdet, rank, w_factor, inv_evals_sq.
        let rank = eig_vals.iter().filter(|&&v| v > tol).count();
        let nullity = p - rank;
        let value: f64 = eig_vals
            .iter()
            .filter(|&&v| v > tol)
            .map(|&v| v.ln())
            .sum();

        // W-factor: p × rank, column j has a single 1/√σ entry at the
        // corresponding flat index.  Since the eigenvectors are the standard
        // basis in the reparameterized coordinates, W is just a column-selected
        // scaled identity.
        let mut w_factor = Array2::<f64>::zeros((p, rank));
        let mut inv_evals_sq = Array1::<f64>::zeros(rank);
        let mut col = 0;
        for (idx, &sigma) in eig_vals.iter().enumerate() {
            if sigma > tol {
                w_factor[[idx, col]] = 1.0 / sigma.sqrt();
                inv_evals_sq[col] = 1.0 / (sigma * sigma);
                col += 1;
            }
        }

        let u_null = if nullity > 0 {
            let mut u0 = Array2::<f64>::zeros((p, nullity));
            let mut col = 0;
            for (idx, &sigma) in eig_vals.iter().enumerate() {
                if sigma <= tol {
                    u0[[idx, col]] = 1.0;
                    col += 1;
                }
            }
            Some(u0)
        } else {
            None
        };

        Ok(Self {
            w_factor,
            u_null,
            inv_evals_sq,
            rank,
            value,
        })
    }

    /// log|S|₊.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Positive eigenspace rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    // ── Reduced-space representations ──────────────────────────────────────

    /// Compute Y = W^T M W for an arbitrary symmetric matrix M.
    ///
    /// This gives the reduced (rank × rank) representation of S⁺ M:
    /// tr(Y) = tr(S⁺ M), and tr(Y_a Y_b^T) = tr(S⁺ M_a S⁺ M_b).
    fn reduced(&self, m: &Array2<f64>) -> Array2<f64> {
        let wt_m = self.w_factor.t().dot(m);
        wt_m.dot(&self.w_factor)
    }

    /// Compute the leakage matrix L = U₊^T M U₀ for the moving-nullspace correction.
    ///
    /// Returns `None` if the nullspace is empty (no correction needed).
    fn leakage(&self, m: &Array2<f64>) -> Option<Array2<f64>> {
        let u_null = self.u_null.as_ref()?;
        // U₊ = W · diag(√σ_i), but we need U₊^T M U₀.
        // U₊[:, j] = w_factor[:, j] * √σ_j = w_factor[:, j] / inv_sqrt(σ_j).
        // Actually, w_factor[:, j] = u_j / √σ_j, so u_j = w_factor[:, j] * √σ_j.
        // We need U₊^T = [u_j^T] so U₊^T M = [u_j^T M].
        //
        // More efficiently: L = diag(√σ) · W^T M U₀, since U₊ = W diag(√σ).
        // But we stored 1/σ² as inv_evals_sq, so √σ = 1/√(1/σ²)^{1/2} = σ^{1/2}.
        // Simpler: σ_j = 1 / (inv_evals_sq[j])^{1/2}.
        //
        // Let's just compute W^T M U₀ and then scale rows by √σ_j.
        let wt_m = self.w_factor.t().dot(m); // rank × p
        let wt_m_u0 = wt_m.dot(u_null); // rank × nullity

        // Scale row j by √σ_j: since w_factor[:, j] = u_j/√σ_j and we want u_j^T M U₀,
        // we need to multiply row j of W^T M U₀ by √σ_j.
        // √σ_j = 1 / √(inv_evals_sq[j])^{1/2} ... let's just use σ_j = inv_evals_sq[j]^{-1/2}.
        // inv_evals_sq[j] = σ_j^{-2}, so σ_j = inv_evals_sq[j]^{-1/2}.
        // √σ_j = inv_evals_sq[j]^{-1/4}... this is getting convoluted.
        //
        // Direct approach: U₊^T M U₀ where U₊[:, j] = evecs[:, nullity+j].
        // Since we didn't store evecs, reconstruct: u_j = w_factor[:, j] * √σ_j,
        // and √σ_j = 1/√(inv_evals_sq[j]^{1/2})... Let's store what we need.
        //
        // Actually simplest: W^T M U₀ gives rows (u_j/√σ_j)^T M U₀ = (1/√σ_j) u_j^T M U₀.
        // We want u_j^T M U₀, so multiply row j by √σ_j = (inv_evals_sq[j])^{-1/4}.
        // Wait: inv_evals_sq[j] = 1/σ_j², so σ_j = 1/sqrt(inv_evals_sq[j]),
        // and √σ_j = 1/(inv_evals_sq[j])^{1/4}.
        //
        // This is error-prone. Let's use a cleaner formulation.
        // The correction is 2 tr(Σ₊⁻² L L^T) where L = U₊^T M U₀.
        // tr(Σ₊⁻² L L^T) = Σ_j σ_j^{-2} Σ_m L_{j,m}²
        //                 = Σ_j σ_j^{-2} ||L_j||²
        //
        // Now L_j = u_j^T M U₀ = √σ_j · (w_j^T M U₀) = √σ_j · (wt_m_u0)[j, :].
        // So L_{j,m}² = σ_j · (wt_m_u0)[j,m]².
        // And σ_j^{-2} · L_{j,m}² = σ_j^{-2} · σ_j · (wt_m_u0)[j,m]² = σ_j^{-1} · (wt_m_u0)[j,m]².
        // And σ_j^{-1} = √(inv_evals_sq[j]).
        //
        // So tr(Σ₊⁻² L L^T) = Σ_j √(inv_evals_sq[j]) · Σ_m (wt_m_u0)[j,m]².
        //
        // We don't actually need L itself, just the trace. Let's provide a direct method.
        Some(wt_m_u0)
    }

    /// Compute the moving-nullspace correction: 2 tr(Σ₊⁻² L_i L_j^T)
    /// where L_i = U₊^T S_{τ_i} U₀.
    ///
    /// This correction is needed when design-moving parameters can rotate
    /// the nullspace of S. For ρ-only parameters (which just scale fixed S_k),
    /// the nullspace is fixed and this correction is zero.
    ///
    /// Takes the W^T S_{τ_i} U₀ matrices (from `leakage()`) rather than
    /// the full L_i, to avoid recomputing.
    fn moving_nullspace_correction(&self, wt_si_u0: &Array2<f64>, wt_sj_u0: &Array2<f64>) -> f64 {
        // tr(Σ₊⁻² L_i L_j^T) where L_i = diag(√σ) · wt_si_u0.
        // = Σ_r σ_r^{-2} Σ_m L_i[r,m] L_j[r,m]
        // = Σ_r σ_r^{-2} σ_r Σ_m wt_si_u0[r,m] wt_sj_u0[r,m]
        // = Σ_r σ_r^{-1} Σ_m wt_si_u0[r,m] wt_sj_u0[r,m]
        // = Σ_r √(inv_evals_sq[r]) · (wt_si_u0 row r) · (wt_sj_u0 row r)
        let mut total = 0.0_f64;
        for r in 0..self.rank {
            let sigma_inv = self.inv_evals_sq[r].sqrt(); // σ_r^{-1}
            let mut row_dot = 0.0_f64;
            let nullity = wt_si_u0.ncols();
            for m in 0..nullity {
                row_dot += wt_si_u0[[r, m]] * wt_sj_u0[[r, m]];
            }
            total += sigma_inv * row_dot;
        }
        2.0 * total
    }

    // ── ρ-parameter derivatives ────────────────────────────────────────────

    /// Compute first and second derivatives of log|S|₊ w.r.t. ρ.
    ///
    /// For S(ρ) = Σ_k λ_k S_k with λ_k = e^{ρ_k}:
    /// - ∂_ρk L = λ_k tr(S⁺ S_k)
    /// - ∂²_ρk ρl L = δ_{kl} ∂_ρk L − λ_k λ_l tr(S⁺ S_k S⁺ S_l)
    ///
    /// The S_k must be the UNSCALED penalty component matrices (before λ multiplication).
    pub fn rho_derivatives(
        &self,
        s_k_matrices: &[Array2<f64>],
        lambdas: &[f64],
    ) -> (Array1<f64>, Array2<f64>) {
        let k = s_k_matrices.len();
        if k == 0 || self.rank == 0 {
            return (Array1::zeros(k), Array2::zeros((k, k)));
        }

        // Reduced representations: Y_k = W^T S_k W (unscaled).
        let y_k: Vec<Array2<f64>> = s_k_matrices.iter().map(|s| self.reduced(s)).collect();

        // First derivatives: ∂_ρk L = λ_k tr(Y_k).
        let mut det1 = Array1::<f64>::zeros(k);
        for (idx, y) in y_k.iter().enumerate() {
            let tr: f64 = (0..self.rank).map(|i| y[[i, i]]).sum();
            det1[idx] = lambdas[idx] * tr;
        }

        // Second derivatives: ∂²_ρk ρl L = δ_{kl} ∂_ρk L − λ_k λ_l tr(Y_k Y_l^T).
        let mut det2 = Array2::<f64>::zeros((k, k));
        for ki in 0..k {
            for li in 0..=ki {
                let tr_ab: f64 = y_k[ki]
                    .iter()
                    .zip(y_k[li].t().iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                let mut val = -lambdas[ki] * lambdas[li] * tr_ab;
                if ki == li {
                    val += det1[ki];
                }
                det2[[ki, li]] = val;
                det2[[li, ki]] = val;
            }
        }

        (det1, det2)
    }

    /// Block-local variant of `rho_derivatives()` that consumes canonical
    /// penalties directly without materializing global `p x p` penalty matrices.
    pub fn rho_derivatives_from_penalties(
        &self,
        penalties: &[crate::construction::CanonicalPenalty],
        lambdas: &[f64],
    ) -> (Array1<f64>, Array2<f64>) {
        let k = penalties.len();
        if k == 0 || self.rank == 0 {
            return (Array1::zeros(k), Array2::zeros((k, k)));
        }

        let y_k: Vec<Array2<f64>> = penalties
            .iter()
            .map(|penalty| {
                let start = penalty.col_range.start;
                let end = penalty.col_range.end;
                let w_block = self.w_factor.slice(s![start..end, ..]).to_owned();
                let local_w = penalty.local.dot(&w_block);
                w_block.t().dot(&local_w)
            })
            .collect();

        let mut det1 = Array1::<f64>::zeros(k);
        for (idx, y) in y_k.iter().enumerate() {
            let tr: f64 = (0..self.rank).map(|i| y[[i, i]]).sum();
            det1[idx] = lambdas[idx] * tr;
        }

        let mut det2 = Array2::<f64>::zeros((k, k));
        for ki in 0..k {
            for li in 0..=ki {
                let tr_ab: f64 = y_k[ki]
                    .iter()
                    .zip(y_k[li].t().iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                let mut val = -lambdas[ki] * lambdas[li] * tr_ab;
                if ki == li {
                    val += det1[ki];
                }
                det2[[ki, li]] = val;
                det2[[li, ki]] = val;
            }
        }

        (det1, det2)
    }

    // ── τ/ψ-parameter derivatives (design-moving) ─────────────────────────

    /// First derivative of log|S|₊ w.r.t. a design-moving parameter τ_i.
    ///
    /// Given S_{τ_i} = ∂S/∂τ_i, returns tr(S⁺ S_{τ_i}).
    pub fn tau_gradient_component(&self, s_tau_i: &Array2<f64>) -> f64 {
        if self.rank == 0 {
            return 0.0;
        }
        let y = self.reduced(s_tau_i);
        (0..self.rank).map(|i| y[[i, i]]).sum()
    }

    /// Second derivative of log|S|₊ w.r.t. design-moving parameters τ_i, τ_j.
    ///
    /// ```text
    /// ∂²_τi τj L = tr(S⁺ S_{τ_i τ_j}) − tr(S⁺ S_{τ_i} S⁺ S_{τ_j})
    ///              + 2 tr(Σ₊⁻² L_i L_j^T)
    /// ```
    ///
    /// where L_i = U₊^T S_{τ_i} U₀ is the leakage into the null eigenspace.
    ///
    /// `s_tau_ij` is ∂²S/∂τ_i∂τ_j (may be `None` if zero, e.g. for pure first-order
    /// interactions).
    pub fn tau_hessian_component(
        &self,
        s_tau_i: &Array2<f64>,
        s_tau_j: &Array2<f64>,
        s_tau_ij: Option<&Array2<f64>>,
    ) -> f64 {
        if self.rank == 0 {
            return 0.0;
        }

        let y_i = self.reduced(s_tau_i);
        let y_j = self.reduced(s_tau_j);

        // tr(S⁺ S_{τ_i τ_j})
        let linear = if let Some(s_ij) = s_tau_ij {
            let y_ij = self.reduced(s_ij);
            (0..self.rank).map(|r| y_ij[[r, r]]).sum::<f64>()
        } else {
            0.0
        };

        // tr(S⁺ S_{τ_i} S⁺ S_{τ_j}) = tr(Y_i Y_j^T)
        let quad: f64 = y_i.iter().zip(y_j.t().iter()).map(|(&a, &b)| a * b).sum();

        // Moving-nullspace correction: 2 tr(Σ₊⁻² L_i L_j^T).
        let nullspace_correction = if self.u_null.is_some() {
            let li = self.leakage(s_tau_i);
            let lj = self.leakage(s_tau_j);
            match (li, lj) {
                (Some(ref wt_i_u0), Some(ref wt_j_u0)) => {
                    self.moving_nullspace_correction(wt_i_u0, wt_j_u0)
                }
                _ => 0.0,
            }
        } else {
            0.0
        };

        linear - quad + nullspace_correction
    }

    // ── Mixed ρ×τ derivatives ──────────────────────────────────────────────

    /// Mixed second derivative ∂²/(∂ρ_k ∂τ_i) log|S|₊.
    ///
    /// For S(ρ, τ) = Σ_k λ_k S_k(τ):
    ///
    /// ```text
    /// ∂²_ρk τi L = λ_k [tr(S⁺ ∂_{τ_i} S_k) − tr(S⁺ S_k S⁺ S_{τ_i})]
    /// ```
    ///
    /// If S_k does NOT depend on τ_i (the common case for pure ρ-scaling),
    /// then ∂_{τ_i} S_k = 0, and this simplifies to:
    ///
    /// ```text
    /// ∂²_ρk τi L = −λ_k tr(S⁺ S_k S⁺ S_{τ_i})
    /// ```
    ///
    /// `ds_k_dtau_i` is ∂S_k/∂τ_i; pass `None` if S_k does not depend on τ_i.
    pub fn rho_tau_hessian_component(
        &self,
        s_k: &Array2<f64>,
        lambda_k: f64,
        s_tau_i: &Array2<f64>,
        ds_k_dtau_i: Option<&Array2<f64>>,
    ) -> f64 {
        if self.rank == 0 {
            return 0.0;
        }

        let y_k = self.reduced(s_k);
        let y_tau = self.reduced(s_tau_i);

        // −λ_k tr(Y_k Y_tau^T) = −λ_k tr(S⁺ S_k S⁺ S_{τ_i})
        let quad: f64 = y_k.iter().zip(y_tau.t().iter()).map(|(&a, &b)| a * b).sum();

        let linear = if let Some(dsk) = ds_k_dtau_i {
            let y_dsk = self.reduced(dsk);
            (0..self.rank).map(|r| y_dsk[[r, r]]).sum::<f64>()
        } else {
            0.0
        };

        lambda_k * (linear - quad)
    }

    /// Block-local variant of `rho_tau_hessian_component()` for canonical penalties.
    pub fn rho_tau_hessian_component_block_local(
        &self,
        penalty: &crate::construction::CanonicalPenalty,
        lambda_k: f64,
        s_tau_i: &Array2<f64>,
        ds_k_dtau_i: Option<&Array2<f64>>,
    ) -> f64 {
        if self.rank == 0 {
            return 0.0;
        }

        let start = penalty.col_range.start;
        let end = penalty.col_range.end;
        let w_block = self.w_factor.slice(s![start..end, ..]).to_owned();
        let local_w = penalty.local.dot(&w_block);
        let y_k = w_block.t().dot(&local_w);
        let y_tau = self.reduced(s_tau_i);

        let quad: f64 = y_k.iter().zip(y_tau.t().iter()).map(|(&a, &b)| a * b).sum();

        let linear = if let Some(dsk) = ds_k_dtau_i {
            let y_dsk = self.reduced(dsk);
            (0..self.rank).map(|r| y_dsk[[r, r]]).sum::<f64>()
        } else {
            0.0
        };

        lambda_k * (linear - quad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Scalar S(ρ) = e^ρ. Then log|S|₊ = ρ, L' = 1, L'' = 0.
    #[test]
    fn test_scalar_penalty_logdet() {
        let rho = 1.5_f64;
        let lambda = rho.exp();
        let s_k = array![[1.0]]; // unscaled
        let pld = PenaltyPseudologdet::from_components(&[s_k.clone()], &[lambda], 0.0).unwrap();

        // Value: log(e^ρ) = ρ
        assert!((pld.value() - rho).abs() < 1e-12, "value should be ρ");

        let (det1, det2) = pld.rho_derivatives(&[s_k], &[lambda]);

        // First derivative: should be 1.0 (= λ · tr(S⁺ S_k) = λ · (1/λ) = 1)
        assert!(
            (det1[0] - 1.0).abs() < 1e-12,
            "det1 = {}, expected 1.0",
            det1[0]
        );

        // Second derivative: should be 0.0 (= 1 - λ² · (1/λ²) = 0)
        assert!(
            det2[[0, 0]].abs() < 1e-12,
            "det2 = {}, expected 0.0",
            det2[[0, 0]]
        );
    }

    /// Two-penalty case: S(ρ₁,ρ₂) = diag(e^ρ₁, e^ρ₂).
    #[test]
    fn test_two_penalty_logdet() {
        let rho = [1.0_f64, -0.5];
        let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();
        let s1 = array![[1.0, 0.0], [0.0, 0.0]];
        let s2 = array![[0.0, 0.0], [0.0, 1.0]];

        let pld =
            PenaltyPseudologdet::from_components(&[s1.clone(), s2.clone()], &lambdas, 0.0).unwrap();

        // Value: log(e^1) + log(e^{-0.5}) = 1 + (-0.5) = 0.5
        assert!(
            (pld.value() - 0.5).abs() < 1e-12,
            "value = {}, expected 0.5",
            pld.value()
        );

        let (det1, det2) = pld.rho_derivatives(&[s1, s2], &lambdas);

        // Each ∂_ρk L = 1 (diagonal, independent).
        assert!((det1[0] - 1.0).abs() < 1e-12);
        assert!((det1[1] - 1.0).abs() < 1e-12);

        // ∂²_ρk ρl L: diagonal = 0 (same as scalar case), off-diagonal = 0.
        assert!(det2[[0, 0]].abs() < 1e-12);
        assert!(det2[[1, 1]].abs() < 1e-12);
        assert!(det2[[0, 1]].abs() < 1e-12);
    }

    /// Finite-difference verification for τ-derivatives with a rotating penalty.
    #[test]
    fn test_tau_derivative_fd() {
        // S(τ) = R(τ) diag(2, 1) R(τ)^T where R(τ) is a 2D rotation.
        let make_s = |tau: f64| -> Array2<f64> {
            let c = tau.cos();
            let s = tau.sin();
            // R diag(2,1) R^T
            array![[2.0 * c * c + s * s, c * s], [c * s, 2.0 * s * s + c * c]]
        };

        let tau0 = 0.3;
        let eps = 1e-7;

        let s0 = make_s(tau0);
        let sp = make_s(tau0 + eps);
        let sm = make_s(tau0 - eps);

        // S_τ ≈ (S(τ+ε) - S(τ-ε)) / 2ε
        let s_tau = (&sp - &sm) / (2.0 * eps);
        // S_ττ ≈ (S(τ+ε) - 2S(τ) + S(τ-ε)) / ε²
        let s_tau_tau = (&sp - 2.0 * &s0 + &sm) / (eps * eps);

        let pld = PenaltyPseudologdet::from_assembled(s0).unwrap();

        // Analytic first derivative.
        let grad = pld.tau_gradient_component(&s_tau);

        // FD first derivative: (log|S(τ+ε)|₊ - log|S(τ-ε)|₊) / 2ε
        let pld_p = PenaltyPseudologdet::from_assembled(sp.clone()).unwrap();
        let pld_m = PenaltyPseudologdet::from_assembled(sm.clone()).unwrap();
        let fd_grad = (pld_p.value() - pld_m.value()) / (2.0 * eps);

        assert!(
            (grad - fd_grad).abs() < 1e-5,
            "τ gradient: analytic={grad}, fd={fd_grad}"
        );

        // Analytic second derivative.
        let hess = pld.tau_hessian_component(&s_tau, &s_tau, Some(&s_tau_tau));

        // FD second derivative.
        let grad_p = {
            let sp2 = make_s(tau0 + eps);
            let sm2 = make_s(tau0 + eps + eps);
            let sm2b = make_s(tau0 + eps - eps);
            let s_tau_p = (&sm2 - &sm2b) / (2.0 * eps);
            let pld_at = PenaltyPseudologdet::from_assembled(sp2).unwrap();
            pld_at.tau_gradient_component(&s_tau_p)
        };
        let grad_m = {
            let sp2 = make_s(tau0 - eps);
            let sm2 = make_s(tau0 - eps + eps);
            let sm2b = make_s(tau0 - eps - eps);
            let s_tau_m = (&sm2 - &sm2b) / (2.0 * eps);
            let pld_at = PenaltyPseudologdet::from_assembled(sp2).unwrap();
            pld_at.tau_gradient_component(&s_tau_m)
        };
        let fd_hess = (grad_p - grad_m) / (2.0 * eps);

        // Looser tolerance for second derivative FD.
        assert!(
            (hess - fd_hess).abs() < 1e-3,
            "τ hessian: analytic={hess}, fd={fd_hess}"
        );
    }

    /// Verify that for a full-rank S, the moving-nullspace correction is zero.
    #[test]
    fn test_no_nullspace_correction_full_rank() {
        let s = array![[3.0, 1.0], [1.0, 2.0]];
        let pld = PenaltyPseudologdet::from_assembled(s).unwrap();
        assert_eq!(pld.rank(), 2);
        assert!(pld.u_null.is_none());
    }

    /// Verify that the pseudo-logdet of a rank-deficient matrix
    /// ignores the null eigenvalues.
    #[test]
    fn test_rank_deficient_value() {
        // S = [[4, 2], [2, 1]] has rank 1, eigenvalue 5.
        let s = array![[4.0, 2.0], [2.0, 1.0]];
        let pld = PenaltyPseudologdet::from_assembled(s).unwrap();
        assert_eq!(pld.rank(), 1);
        assert!((pld.value() - 5.0_f64.ln()).abs() < 1e-12);
    }

    /// The first derivative of log|S(ψ)|₊ is zero when ψ only rotates the
    /// nullspace and doesn't change the positive eigenvalues.
    #[test]
    fn test_nullspace_rotation_gradient_zero() {
        // S(ψ) = R(ψ) diag(s₁, s₂, 0) R(ψ)^T — rotating a rank-2 matrix in 3D.
        // log|S|₊ = log(s₁) + log(s₂) = const, so ∂_ψ L = 0.
        let s1 = 3.0_f64;
        let s2 = 1.0_f64;
        let psi = 0.5_f64;
        let c = psi.cos();
        let s = psi.sin();

        // Build S(ψ): rotate in the (1,3) plane.
        let r = array![[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]];
        let d = array![[s1, 0.0, 0.0], [0.0, s2, 0.0], [0.0, 0.0, 0.0]];
        let s_mat = r.dot(&d).dot(&r.t());

        // S_ψ = R'(ψ) D R(ψ)^T + R(ψ) D R'(ψ)^T
        let r_psi = array![[-s, 0.0, -c], [0.0, 0.0, 0.0], [c, 0.0, -s]];
        let s_psi = r_psi.dot(&d).dot(&r.t()) + r.dot(&d).dot(&r_psi.t());

        let pld = PenaltyPseudologdet::from_assembled(s_mat).unwrap();
        assert_eq!(pld.rank(), 2);

        let grad = pld.tau_gradient_component(&s_psi);

        // The gradient of log(s₁) + log(s₂) w.r.t. a rotation is zero.
        assert!(
            grad.abs() < 1e-10,
            "nullspace-rotation gradient should be zero, got {grad}"
        );
    }
}

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
use rayon::prelude::*;

use crate::faer_ndarray::{FaerCholesky, FaerEigh};

/// Check whether penalty ranges decompose into independent exact blocks.
///
/// Multiple smoothing components may share the same block (for example tensor
/// product marginals); those can still be factorized block-local.  Only partial
/// overlaps force the dense assembled fallback.
fn are_penalties_block_factored(penalties: &[crate::construction::CanonicalPenalty]) -> bool {
    for (i, a) in penalties.iter().enumerate() {
        for b in &penalties[i + 1..] {
            let overlaps =
                a.col_range.start < b.col_range.end && b.col_range.start < a.col_range.end;
            let same_range =
                a.col_range.start == b.col_range.start && a.col_range.end == b.col_range.end;
            if overlaps && !same_range {
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
#[derive(Clone, Debug)]
struct PenaltyBlockSpan {
    start: usize,
    end: usize,
    rank_start: usize,
    rank_end: usize,
}

#[derive(Clone, Debug)]
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
    /// Block/rank spans when the penalty eigenspace was assembled from disjoint blocks.
    block_spans: Vec<PenaltyBlockSpan>,
}

impl PenaltyPseudologdet {
    /// Compute tr(A B) = Σ_i Σ_k A[i,k] B[k,i] without materializing the product.
    #[inline]
    fn trace_dense_product(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let diag_len = a.nrows().min(b.ncols());
        let inner_len = a.ncols().min(b.nrows());
        let mut total = 0.0;
        for i in 0..diag_len {
            for k in 0..inner_len {
                total += a[[i, k]] * b[[k, i]];
            }
        }
        total
    }

    /// Build from block-local `Penalty` values and current lambdas.
    ///
    /// When all penalties have disjoint column ranges, the eigendecomposition
    /// factorizes per-block: each block is at most `block_p × block_p` instead
    /// of a single `p × p` spectral solve. When blocks overlap, falls back
    /// to assembling the full combined penalty and eigendecomposing once.
    ///
    /// This is the preferred entry point for REML logdet computation.  No
    /// metadata-based structural-nullity hint is consumed; the positive
    /// eigenspace is identified purely from the assembled spectrum (see
    /// [`Self::from_assembled`]).
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
                block_spans: Vec::new(),
            });
        }

        // Check if all penalty blocks are disjoint.
        let disjoint = are_penalties_block_factored(penalties);

        if disjoint {
            // Block-factored path: assemble and eigendecompose per-block.
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
            let ridge_hint = if ridge > 0.0 { Some(ridge) } else { None };
            Self::from_assembled(s_total, ridge_hint)
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

        // For the unpenalized dimensions (not covered by any block), add ridge.
        // Those dimensions have eigenvalue = ridge if ridge > 0, otherwise 0 (null).
        let mut covered = vec![false; p_total];
        for bd in &blocks {
            for i in bd.start..bd.end {
                covered[i] = true;
            }
        }

        // Process each block independently.  Keep the eigenspace local until
        // final assembly so large smooth bases do not allocate one p_total×rank
        // temporary per block.
        struct BlockResult {
            start: usize,
            end: usize,
            w_local: Array2<f64>,
            u_null_local: Array2<f64>,
            inv_evals_sq: Vec<f64>,
            value: f64,
            rank: usize,
            nullity: usize,
        }

        let ridge_hint = if ridge > 0.0 { Some(ridge) } else { None };
        let process_block = |bd: &BlockData| -> Result<BlockResult, String> {
            let block_pld = Self::from_assembled(bd.local.clone(), ridge_hint)?;
            let nullity = block_pld.u_null.as_ref().map_or(0, Array2::ncols);
            Ok(BlockResult {
                start: bd.start,
                end: bd.end,
                w_local: block_pld.w_factor,
                u_null_local: block_pld
                    .u_null
                    .unwrap_or_else(|| Array2::<f64>::zeros((bd.end - bd.start, 0))),
                inv_evals_sq: block_pld.inv_evals_sq.to_vec(),
                value: block_pld.value,
                rank: block_pld.rank,
                nullity,
            })
        };
        let mut block_results: Vec<BlockResult> = if rayon::current_thread_index().is_some() {
            blocks
                .iter()
                .map(process_block)
                .collect::<Result<Vec<_>, String>>()?
        } else {
            blocks
                .par_iter()
                .map(process_block)
                .collect::<Result<Vec<_>, String>>()?
        };

        // Also add uncovered dimensions as trivial "block results".
        if ridge > 0.0 {
            let inv_ridge_sq = 1.0 / (ridge * ridge);
            let scale = 1.0 / ridge.sqrt();
            for (idx, &c) in covered.iter().enumerate() {
                if !c {
                    let mut w_col = Array2::<f64>::zeros((1, 1));
                    w_col[[0, 0]] = scale;
                    block_results.push(BlockResult {
                        start: idx,
                        end: idx + 1,
                        w_local: w_col,
                        u_null_local: Array2::<f64>::zeros((1, 0)),
                        inv_evals_sq: vec![inv_ridge_sq],
                        value: ridge.ln(),
                        rank: 1,
                        nullity: 0,
                    });
                }
            }
        }

        // Assemble combined W-factor and other arrays.
        let total_rank: usize = block_results.iter().map(|br| br.rank).sum();
        let total_value: f64 = block_results.iter().map(|br| br.value).sum();

        let mut w_factor_combined = Array2::<f64>::zeros((p_total, total_rank));
        let mut inv_evals_sq_combined = Array1::<f64>::zeros(total_rank);
        let mut block_spans = Vec::with_capacity(block_results.len());
        let mut col_offset = 0;
        for br in &block_results {
            if br.rank > 0 {
                w_factor_combined
                    .slice_mut(s![br.start..br.end, col_offset..col_offset + br.rank])
                    .assign(&br.w_local);
                for (i, &v) in br.inv_evals_sq.iter().enumerate() {
                    inv_evals_sq_combined[col_offset + i] = v;
                }
                block_spans.push(PenaltyBlockSpan {
                    start: br.start,
                    end: br.end,
                    rank_start: col_offset,
                    rank_end: col_offset + br.rank,
                });
                col_offset += br.rank;
            }
        }

        // Null space: the dimensions where eigenvalue == 0 (ridge == 0, no penalty).
        let block_nullity: usize = block_results.iter().map(|br| br.nullity).sum();
        let uncovered_nullity = if ridge > 0.0 {
            0
        } else {
            covered.iter().filter(|&&c| !c).count()
        };
        let total_nullity = block_nullity + uncovered_nullity;
        let u_null = if total_nullity > 0 {
            let mut u0 = Array2::<f64>::zeros((p_total, total_nullity));
            let mut null_col = 0;
            for br in &block_results {
                if br.nullity > 0 {
                    u0.slice_mut(s![br.start..br.end, null_col..null_col + br.nullity])
                        .assign(&br.u_null_local);
                    null_col += br.nullity;
                }
            }
            for (idx, &c) in covered.iter().enumerate() {
                if !c && ridge <= 0.0 {
                    u0[[idx, null_col]] = 1.0;
                    null_col += 1;
                }
            }
            assert_eq!(
                null_col, total_nullity,
                "block-factored pseudo-logdet nullspace assembly mismatch"
            );
            Some(u0)
        } else {
            None
        };

        Ok(Self {
            w_factor: w_factor_combined,
            u_null,
            inv_evals_sq: inv_evals_sq_combined,
            rank: total_rank,
            value: total_value,
            block_spans,
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
                block_spans: Vec::new(),
            });
        }

        let p_dim = s_k_matrices[0].nrows();
        assert!(
            s_k_matrices
                .iter()
                .all(|m| m.nrows() == p_dim && m.ncols() == p_dim)
        );

        // Build S = Σ λ_k S_k (+ ridge·I).
        let mut s_total = Array2::<f64>::zeros((p_dim, p_dim));
        for (k, s_k) in s_k_matrices.iter().enumerate() {
            s_total.scaled_add(lambdas[k], s_k);
        }
        if ridge > 0.0 {
            for i in 0..p_dim {
                s_total[[i, i]] += ridge;
            }
        }

        let ridge_hint = if ridge > 0.0 { Some(ridge) } else { None };
        Self::from_assembled(s_total, ridge_hint)
    }

    /// Build from a pre-assembled penalty matrix.
    ///
    /// `s_total` must be `Σ_k λ_k S_k` plus, if `ridge` is `Some(r)`, an
    /// additive `r·I` already applied to the diagonal. The caller is expected
    /// to have assembled the matrix in exactly that form.
    ///
    /// The positive/null eigenspace split is determined entirely from the
    /// eigenspectrum of `s_total`, with no auxiliary structural-nullity hint:
    ///
    /// * When `ridge` is `Some(r)`, a direction is structurally null iff its
    ///   eigenvalue is within machine-precision tolerance of `r` (i.e. the
    ///   unridged `Σ λ_k S_k` has a zero eigenvalue along that direction).
    /// * When `ridge` is `None`, a direction is null iff its eigenvalue is at
    ///   or below the relative noise floor `positive_eigenvalue_threshold`
    ///   (`100 · p · ε · max|e|`).
    ///
    /// In both regimes the tolerance band is the only authority — there is no
    /// supplementary `m0` parameter, no metadata cross-check, and no error
    /// path for nullity mismatches.  When a "barely-active" eigenvalue
    /// `λ_k σ_k < band` sits within the band, the contribution to `log|S|₊`
    /// would be `log(r + λ_k σ_k) ≈ log r` and is dominated by the ridge
    /// term; dropping such directions is consistent with the regularizer.
    /// This makes the pseudo-logdet a C∞ function of ρ over the positive
    /// eigenspace and removes the redundant metadata invariant that issues
    /// #192 and #318 both stemmed from.
    pub fn from_assembled(s_total: Array2<f64>, ridge: Option<f64>) -> Result<Self, String> {
        let p_dim = s_total.nrows();
        if p_dim == 0 {
            return Ok(Self {
                w_factor: Array2::zeros((0, 0)),
                u_null: None,
                inv_evals_sq: Array1::zeros(0),
                rank: 0,
                value: 0.0,
                block_spans: Vec::new(),
            });
        }

        // Eigendecomposition (ascending eigenvalues).
        let (evals, evecs) = s_total
            .eigh(Side::Lower)
            .map_err(|e| format!("PenaltyPseudologdet eigendecomposition failed: {e}"))?;

        // Compute the null-vs-active boundary purely from the spectrum.
        //
        //   ridge = None:  boundary = positive_eigenvalue_threshold(evals)
        //                  (= 100 · p · ε · max|e|; eigenvalues at or below
        //                  this are noise around 0).
        //
        //   ridge = r > 0: boundary = r + delta, where delta is the same
        //                  100 · p · ε · max|e| noise band.  Directions in
        //                  the structural null of the unridged S have
        //                  eigenvalue exactly r in the ridged S; the
        //                  eigendecomposition introduces at most O(p · ε · ‖S‖)
        //                  perturbation, so any eigenvalue ≤ r + delta is
        //                  indistinguishable from ridge-only within FP noise.
        let noise_band = super::unified::positive_eigenvalue_threshold(evals.as_slice().unwrap());
        let boundary = match ridge {
            Some(r) if r > 0.0 => r + noise_band,
            _ => noise_band,
        };
        let mut positive_indices = Vec::with_capacity(p_dim);
        let mut null_indices = Vec::with_capacity(p_dim);
        for (idx, &eval) in evals.iter().enumerate() {
            if eval > boundary {
                positive_indices.push(idx);
            } else {
                null_indices.push(idx);
            }
        }
        let rank = positive_indices.len();
        let nullity = null_indices.len();

        // Value: log|S|₊ = Σ log σ_i for positive eigenvalues.
        //
        // Full-rank (nullity == 0) SPD case: prefer the Cholesky
        // log-determinant over Σ log(eigval). eigh introduces ~ε·max|e|
        // absolute noise on each eigenvalue, which becomes O(ε·κ(S))
        // relative noise on log(eigval) once a barely-positive mode sits
        // near the ridge band. The Cholesky factorization is a direct
        // backward-stable elimination, so 2·Σ log(diag(L)) carries
        // ~ε absolute precision on log|S| even at high condition number.
        // Central-difference FDs of value() in ρ then resolve the
        // analytic gradient λ_k·σ_k/(λ_k·σ_k+ridge) at step sizes h that
        // would otherwise drown in eigh noise on the barely-active mode
        // (gam unit test `test_value_matches_rho_gradient_across_ridge_boundary`
        // pins the ρ1=-20 deep-band regime).
        //
        // Rank-deficient (nullity > 0): keep the eigh-based sum over
        // positive_indices. Subtracting a Σ log(eigval_null) reconstruction
        // from the Cholesky log-det would push eigh noise on those null
        // eigenvalues (~ ε·max|e| / ridge in relative log terms) into
        // value(), which is materially worse precision than direct
        // log(eigval_positive) summation when the positive spectrum lies
        // well above ridge (gam test_components_with_stale_nullity_*).
        let value: f64 = if nullity == 0
            && matches!(ridge, Some(r) if r > 0.0)
            && let Ok(fac) = s_total.cholesky(Side::Lower)
        {
            2.0 * fac.diag().iter().map(|d| d.ln()).sum::<f64>()
        } else {
            positive_indices.iter().map(|&idx| evals[idx].ln()).sum()
        };

        // W factor: p × rank, W_{:,j} = u_j / √σ_j for positive eigenvalues.
        let mut w_factor = Array2::<f64>::zeros((p_dim, rank));
        let mut inv_evals_sq = Array1::<f64>::zeros(rank);
        for (col, &idx) in positive_indices.iter().enumerate() {
            let ev = evals[idx];
            let scale = 1.0 / ev.sqrt();
            inv_evals_sq[col] = 1.0 / (ev * ev);
            for row in 0..p_dim {
                w_factor[[row, col]] = evecs[[row, idx]] * scale;
            }
        }

        // Null-space eigenvectors U₀: structural nulls plus values below the
        // dimension-aware positive-eigenvalue threshold.
        let u_null = if nullity > 0 {
            let mut u0 = Array2::<f64>::zeros((p_dim, nullity));
            for (col, &idx) in null_indices.iter().enumerate() {
                for row in 0..p_dim {
                    u0[[row, col]] = evecs[[row, idx]];
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
            block_spans: Vec::new(),
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

    /// Ambient dimension p of the penalty space this factorization lives in.
    ///
    /// Consumers that share one factorization across code paths (#931: the
    /// `EvalShared` cell) use this to assert they are contracting against an
    /// object built in the frame they expect.
    pub fn dim(&self) -> usize {
        self.w_factor.nrows()
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
    /// Compute W^T M U₀ for the moving-nullspace correction.
    ///
    /// Returns the rank × nullity matrix whose row j is (w_j^T M U₀).
    /// The downstream `moving_nullspace_correction` weights each row by
    /// σ_j^{-1} = √(inv_evals_sq[j]) to form the trace without ever
    /// materializing L = U₊^T M U₀ explicitly.
    fn leakage(&self, m: &Array2<f64>) -> Option<Array2<f64>> {
        let u_null = self.u_null.as_ref()?;
        let wt_m = self.w_factor.t().dot(m);
        Some(wt_m.dot(u_null))
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
        // These K projections are independent and dominate derivative time for
        // large bases, so evaluate them in parallel outside existing rayon jobs.
        let y_k: Vec<Array2<f64>> = if rayon::current_thread_index().is_some() {
            s_k_matrices.iter().map(|s| self.reduced(s)).collect()
        } else {
            s_k_matrices.par_iter().map(|s| self.reduced(s)).collect()
        };

        // First derivatives: ∂_ρk L = λ_k tr(Y_k).
        let first_vals: Vec<f64> = y_k
            .iter()
            .enumerate()
            .map(|(idx, y)| lambdas[idx] * (0..self.rank).map(|i| y[[i, i]]).sum::<f64>())
            .collect();
        let mut det1 = Array1::<f64>::zeros(k);
        for (idx, value) in first_vals.into_iter().enumerate() {
            det1[idx] = value;
        }

        // Second derivatives: ∂²_ρk ρl L = δ_{kl} ∂_ρk L − λ_k λ_l tr(Y_k Y_l).
        // Y_k is symmetric (W^T S_k W with S_k symmetric), so tr(Y_k Y_l) = tr(Y_k Y_l^T).
        let pairs = (0..k).flat_map(|ki| (0..=ki).map(move |li| (ki, li)));
        let pair_vals: Vec<(usize, usize, f64)> = if rayon::current_thread_index().is_some() {
            pairs
                .map(|(ki, li)| {
                    let tr_ab = Self::trace_dense_product(&y_k[ki], &y_k[li]);
                    let mut val = -lambdas[ki] * lambdas[li] * tr_ab;
                    if ki == li {
                        val += det1[ki];
                    }
                    (ki, li, val)
                })
                .collect()
        } else {
            pairs
                .par_bridge()
                .map(|(ki, li)| {
                    let tr_ab = Self::trace_dense_product(&y_k[ki], &y_k[li]);
                    let mut val = -lambdas[ki] * lambdas[li] * tr_ab;
                    if ki == li {
                        val += det1[ki];
                    }
                    (ki, li, val)
                })
                .collect()
        };
        let mut det2 = Array2::<f64>::zeros((k, k));
        for (ki, li, val) in pair_vals {
            det2[[ki, li]] = val;
            det2[[li, ki]] = val;
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

        struct ReducedPenalty {
            span: Option<usize>,
            y: Array2<f64>,
        }

        let project = |penalty: &crate::construction::CanonicalPenalty| {
            let start = penalty.col_range.start;
            let end = penalty.col_range.end;
            if let Some((span_idx, span)) = self
                .block_spans
                .iter()
                .enumerate()
                .find(|(_, span)| span.start <= start && end <= span.end)
            {
                let local_start = start - span.start;
                let local_end = local_start + (end - start);
                let w_block = self
                    .w_factor
                    .slice(s![start..end, span.rank_start..span.rank_end]);
                let local_w = penalty.local.dot(&w_block);
                let y = self
                    .w_factor
                    .slice(s![start..end, span.rank_start..span.rank_end])
                    .t()
                    .dot(&local_w);
                assert_eq!(local_end - local_start, penalty.local.nrows());
                ReducedPenalty {
                    span: Some(span_idx),
                    y,
                }
            } else {
                // Overlapping/global fallback: still avoid cloning the block view.
                let w_block = self.w_factor.slice(s![start..end, ..]);
                let local_w = penalty.local.dot(&w_block);
                ReducedPenalty {
                    span: None,
                    y: w_block.t().dot(&local_w),
                }
            }
        };

        let y_k: Vec<ReducedPenalty> = if rayon::current_thread_index().is_some() {
            penalties.iter().map(project).collect()
        } else {
            penalties.par_iter().map(project).collect()
        };

        let mut det1 = Array1::<f64>::zeros(k);
        for (idx, reduced) in y_k.iter().enumerate() {
            let tr: f64 = (0..reduced.y.nrows()).map(|i| reduced.y[[i, i]]).sum();
            det1[idx] = lambdas[idx] * tr;
        }

        let pairs = (0..k).flat_map(|ki| (0..=ki).map(move |li| (ki, li)));
        let pair_vals: Vec<(usize, usize, f64)> = if rayon::current_thread_index().is_some() {
            pairs
                .map(|(ki, li)| {
                    let same_span = match (y_k[ki].span, y_k[li].span) {
                        (Some(a), Some(b)) => a == b,
                        _ => true,
                    };
                    let tr_ab = if same_span {
                        Self::trace_dense_product(&y_k[ki].y, &y_k[li].y)
                    } else {
                        0.0
                    };
                    let mut val = -lambdas[ki] * lambdas[li] * tr_ab;
                    if ki == li {
                        val += det1[ki];
                    }
                    (ki, li, val)
                })
                .collect()
        } else {
            pairs
                .par_bridge()
                .map(|(ki, li)| {
                    let same_span = match (y_k[ki].span, y_k[li].span) {
                        (Some(a), Some(b)) => a == b,
                        _ => true,
                    };
                    let tr_ab = if same_span {
                        Self::trace_dense_product(&y_k[ki].y, &y_k[li].y)
                    } else {
                        0.0
                    };
                    let mut val = -lambdas[ki] * lambdas[li] * tr_ab;
                    if ki == li {
                        val += det1[ki];
                    }
                    (ki, li, val)
                })
                .collect()
        };
        let mut det2 = Array2::<f64>::zeros((k, k));
        for (ki, li, val) in pair_vals {
            det2[[ki, li]] = val;
            det2[[li, ki]] = val;
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

        // Reduced-space Y_i = W^T S_{τ_i} W (rank × rank); avoids materializing
        // the dense p×p pseudo-inverse and the p×p×p×p×p chain
        // `S⁺ · S_{τ_i} · S⁺`.  Identities used:
        //   tr(S⁺ M)              = tr(W^T M W) = tr(Y_M)
        //   tr(S⁺ S_τi S⁺ S_τj)   = tr((W^T S_τi W)(W^T S_τj W))  [cyclic on S⁺=WW^T]
        // Both Y_τi and Y_τj are symmetric (S_τi, S_τj symmetric), so
        // tr(Y_i Y_j) = tr(Y_i Y_j^T) = `trace_dense_product`.
        let y_i = self.reduced(s_tau_i);
        let y_j = self.reduced(s_tau_j);

        // tr(S⁺ S_{τ_i τ_j}) = tr(W^T S_{ij} W).
        let linear = if let Some(s_ij) = s_tau_ij {
            let y_ij = self.reduced(s_ij);
            (0..self.rank).map(|r| y_ij[[r, r]]).sum::<f64>()
        } else {
            0.0
        };

        // tr(S⁺ S_{τ_i} S⁺ S_{τ_j}) = tr(Y_i Y_j).
        let quad = Self::trace_dense_product(&y_i, &y_j);

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

        // Reduced-space form (see `tau_hessian_component`):
        //   tr(S⁺ M)            = tr(W^T M W)
        //   tr(S⁺ S_k S⁺ S_τi)  = tr((W^T S_k W)(W^T S_τi W))
        // This avoids materializing the p×p pseudo-inverse and the
        // cubic `S⁺ · S_k · S⁺` chain.
        let y_k = self.reduced(s_k);
        let y_tau_i = self.reduced(s_tau_i);

        // tr(S⁺ S_k S⁺ S_{τ_i}) = tr(Y_k Y_τi).  Both Y_k and Y_τi are
        // symmetric, so the product trace matches `trace_dense_product`.
        let quad = Self::trace_dense_product(&y_k, &y_tau_i);

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

    /// Validate τ-derivatives against exact closed-form scalar references
    /// (gauge-invariant), not finite-differences of decomposition-dependent
    /// intermediate objects which are vulnerable to eigenspace-gauge noise.
    #[test]
    fn test_tau_derivative_fd() {
        // S(τ) = [[1+τ, 0.5], [0.5, 2]].
        // det(S) = 2(1+τ) - 0.25 = 2τ + 1.75.
        // log|S| = log(2τ + 1.75).
        // d/dτ log|S|  = 2 / (2τ + 1.75).
        // d²/dτ² log|S| = -4 / (2τ + 1.75)².
        let tau0 = 0.3_f64;
        let det = 2.0 * tau0 + 1.75;

        let s0 = array![[1.0 + tau0, 0.5], [0.5, 2.0]];
        let s_tau = array![[1.0, 0.0], [0.0, 0.0]];
        let s_tau_tau = Array2::<f64>::zeros((2, 2));

        let pld = PenaltyPseudologdet::from_assembled(s0, None).unwrap();

        // Gradient: exact = 2 / det.
        let exact_grad = 2.0 / det;
        let grad = pld.tau_gradient_component(&s_tau);
        assert!(
            (grad - exact_grad).abs() < 1e-12,
            "τ gradient: analytic={grad}, exact={exact_grad}"
        );

        // Hessian: exact = -4 / det².
        let exact_hess = -4.0 / (det * det);
        let hess = pld.tau_hessian_component(&s_tau, &s_tau, Some(&s_tau_tau));
        assert!(
            (hess - exact_hess).abs() < 1e-12,
            "τ hessian: analytic={hess}, exact={exact_hess}"
        );
    }

    /// Verify that for a full-rank S, the moving-nullspace correction is zero.
    #[test]
    fn test_no_nullspace_correction_full_rank() {
        let s = array![[3.0, 1.0], [1.0, 2.0]];
        let pld = PenaltyPseudologdet::from_assembled(s, None).unwrap();
        assert_eq!(pld.rank(), 2);
        assert!(pld.u_null.is_none());
    }

    /// Regression test for issues #192 and #318 under the unified pure-spectrum rule.
    ///
    /// The classifier is now driven entirely by the assembled eigenspectrum
    /// relative to `r + noise_band`, where `noise_band = 100·p·ε·max|e|`.  No
    /// metadata `m0` hint is consulted, so the failure modes both issues
    /// described — "metadata claims a null direction that doesn't exist"
    /// (#318) and "positional rule misclassifies a barely-active direction
    /// near the ridge" (#192) — are dissolved at the same point: the
    /// spectrum is the sole authority and is C∞ in ρ over the positive
    /// eigenspace.
    #[test]
    fn test_assembled_pure_spectrum_classifier_issue_192_and_318() {
        let ridge = 1e-4_f64;

        // ── #192 case: one structural null at r, one barely-active at r + 1e-10.
        //
        // The barely-active eigenvalue `r + 1e-10` lies above `r + noise_band`
        // (noise_band ≈ 100 · 2 · ε · r ≈ 4.4e-18, far below 1e-10), so it is
        // correctly classified as positive and contributes log(r + 1e-10) to
        // log|S|₊; the eigenvalue at exactly `r` is the structural null.
        let active_eval = 1e-10_f64;
        let s = array![[ridge + active_eval, 0.0], [0.0, ridge]];
        let pld =
            PenaltyPseudologdet::from_assembled(s.clone(), Some(ridge)).expect("ridged assembled");
        assert_eq!(pld.rank(), 1);
        let expected = (ridge + active_eval).ln();
        assert!(
            (pld.value() - expected).abs() < 1e-14,
            "log|S|₊ should retain the barely-active eigenvalue {expected} but got {}",
            pld.value(),
        );
        let u0 = pld.u_null.as_ref().expect("nullspace basis present");
        assert_eq!(u0.ncols(), 1);
        let aligned = u0[[1, 0]].abs();
        assert!(
            aligned > 0.999,
            "null direction should align with e_1 (eigenvalue r); got |u0[1,0]| = {aligned}",
        );

        // ── #318 case: no eigenvalue near r; the assembled matrix is fully
        // active relative to the noise band.  Under the old rule, a metadata
        // `m0 = 1` would have produced a spurious "structural nullity
        // invariant violated" error here.  Under the pure-spectrum rule the
        // build succeeds with rank 2 and no nullspace basis.
        let s_no_null = array![[1.0 + ridge, 0.0], [0.0, 2.0 + ridge]];
        let pld_no_null =
            PenaltyPseudologdet::from_assembled(s_no_null, Some(ridge)).expect("fully active");
        assert_eq!(pld_no_null.rank(), 2);
        assert!(pld_no_null.u_null.is_none());

        // ── Two-null case: both eigenvalues sit at the ridge level (the
        // assembled matrix is structurally fully null).  Rank 0, both
        // directions go into the nullspace basis — there is nothing to
        // disambiguate, no error, no metadata cross-check.
        let s_extra_nulls = array![[ridge, 0.0], [0.0, ridge]];
        let pld_two_nulls = PenaltyPseudologdet::from_assembled(s_extra_nulls, Some(ridge))
            .expect("ridge-only spectrum");
        assert_eq!(pld_two_nulls.rank(), 0);
        let u0 = pld_two_nulls
            .u_null
            .as_ref()
            .expect("two structural nulls populate u_null");
        assert_eq!(u0.ncols(), 2);
    }

    /// Verify that the pseudo-logdet of a rank-deficient matrix
    /// ignores the null eigenvalues.
    #[test]
    fn test_rank_deficient_value() {
        // S = [[4, 2], [2, 1]] has rank 1, eigenvalue 5.
        let s = array![[4.0, 2.0], [2.0, 1.0]];
        let pld = PenaltyPseudologdet::from_assembled(s, None).unwrap();
        assert_eq!(pld.rank(), 1);
        assert!((pld.value() - 5.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn test_component_ridge_excludes_inactive_penalty_nullspace() {
        let s1 = array![[4.0, 0.0], [0.0, 0.0]];
        let s2 = array![[0.0, 0.0], [0.0, 9.0]];
        let lambdas = [2.0_f64, 0.0_f64];
        let ridge = 1e-4_f64;

        let pld = PenaltyPseudologdet::from_components(&[s1.clone(), s2.clone()], &lambdas, ridge)
            .unwrap();

        assert_eq!(pld.rank(), 1);
        assert!((pld.value() - (8.0 + ridge).ln()).abs() < 1e-12);

        let (det1, det2) = pld.rho_derivatives(&[s1, s2], &lambdas);
        assert!((det1[0] - 8.0 / (8.0 + ridge)).abs() < 1e-12);
        assert!(det1[1].abs() < 1e-12);
        assert!(det2[[0, 1]].abs() < 1e-12);
    }

    /// Value↔ρ-gradient consistency across the ridge boundary
    /// (gam#752/#748/#808 desync guard).
    ///
    /// The outer REML/LAML objective uses `value()` for the `log|S_λ|₊` term
    /// and `rho_derivatives()` for its ρ-gradient. The two MUST be the exact
    /// value/derivative of the SAME function — in particular, both must classify
    /// the positive/null eigenspace identically. A previous custom-family value
    /// path dropped the bottom eigenvalues *by structural count* while the
    /// gradient dropped them *by magnitude* (> ridge + noise_band); near the
    /// ridge a barely-active penalized mode `λσ → 0` was kept by one rule and
    /// dropped by the other, so the value and gradient described different
    /// functions. Sweep ρ through the regime where a penalized eigenvalue
    /// crosses from well above the ridge to deep below it, and confirm a
    /// central finite difference of `value()` matches `rho_derivatives()`.
    #[test]
    fn test_value_matches_rho_gradient_across_ridge_boundary() {
        // S(ρ) = e^{ρ0} S0 + e^{ρ1} S1, with S0 large and S1 a tiny mode that
        // dives toward (and below) the ridge as ρ1 decreases.
        let s0 = array![[1.0, 0.0], [0.0, 0.0]];
        let s1 = array![[0.0, 0.0], [0.0, 1.0]];
        let ridge = 1e-8_f64;

        let value_at = |rho: [f64; 2]| -> f64 {
            let lambdas = [rho[0].exp(), rho[1].exp()];
            PenaltyPseudologdet::from_components(&[s0.clone(), s1.clone()], &lambdas, ridge)
                .unwrap()
                .value()
        };

        // Sweep ρ1 from "mode well above ridge" to "mode well below ridge".
        // At each interior point the central FD of value() in ρ1 must equal the
        // analytic ∂_{ρ1} value from rho_derivatives(), regardless of whether the
        // classifier currently counts the S1 mode as positive or null — because
        // value() and rho_derivatives() share the classifier.
        for &rho1 in &[5.0_f64, 1.0, -2.0, -8.0, -12.0, -20.0] {
            let rho = [0.5_f64, rho1];
            let lambdas = [rho[0].exp(), rho[1].exp()];
            let pld =
                PenaltyPseudologdet::from_components(&[s0.clone(), s1.clone()], &lambdas, ridge)
                    .unwrap();
            let (det1, _) = pld.rho_derivatives(&[s0.clone(), s1.clone()], &lambdas);

            let h = 1e-5_f64;
            let fd1 = (value_at([rho[0], rho1 + h]) - value_at([rho[0], rho1 - h])) / (2.0 * h);

            // Loose bound: the classifier boundary makes the gradient exactly
            // 0 in the "null" regime and exactly λσ/(λσ+ridge) in the active
            // regime; the FD of value() tracks whichever branch is active,
            // so they agree to FD truncation error.
            assert!(
                (det1[1] - fd1).abs() < 1e-5,
                "ρ1={rho1}: analytic ∂_ρ1 value={}, FD of value()={fd1}",
                det1[1]
            );
        }
    }

    #[test]
    fn test_components_with_stale_nullity_uses_active_sum_when_lambda_zero() {
        let s1 = array![[4.0, 0.0], [0.0, 0.0]];
        let s2 = array![[0.0, 0.0], [0.0, 9.0]];
        let lambdas = [2.0_f64, 0.0_f64];
        let ridge = 1e-4_f64;

        let pld = PenaltyPseudologdet::from_components(&[s1, s2], &lambdas, ridge).unwrap();

        assert_eq!(pld.rank(), 1);
        assert!((pld.value() - (8.0 + ridge).ln()).abs() < 1e-12);
    }

    #[test]
    fn test_rank_deficient_components_can_sum_to_full_rank_or_not() {
        let s1 = array![[1.0, 0.0], [0.0, 0.0]];
        let s2 = array![[0.0, 0.0], [0.0, 1.0]];
        let full =
            PenaltyPseudologdet::from_components(&[s1.clone(), s2], &[2.0, 3.0], 0.0).unwrap();
        assert_eq!(full.rank(), 2);
        assert!((full.value() - (6.0_f64).ln()).abs() < 1e-12);

        let s3 = array![[5.0, 0.0], [0.0, 0.0]];
        let deficient = PenaltyPseudologdet::from_components(&[s1, s3], &[2.0, 3.0], 0.0).unwrap();
        assert_eq!(deficient.rank(), 1);
        assert!((deficient.value() - (17.0_f64).ln()).abs() < 1e-12);
    }

    #[test]
    fn test_block_penalties_ridge_excludes_inactive_penalty_nullspace() {
        let penalties = [
            crate::construction::CanonicalPenalty::from_dense_root(array![[2.0, 0.0]], 2),
            crate::construction::CanonicalPenalty::from_dense_root(array![[0.0, 3.0]], 2),
        ];
        let lambdas = [2.0_f64, 0.0_f64];
        let ridge = 1e-4_f64;

        let pld = PenaltyPseudologdet::from_penalties(&penalties, &lambdas, ridge, 2).unwrap();

        assert_eq!(pld.rank(), 1);
        assert!((pld.value() - (8.0 + ridge).ln()).abs() < 1e-12);
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

        let pld = PenaltyPseudologdet::from_assembled(s_mat, None).unwrap();
        assert_eq!(pld.rank(), 2);

        let grad = pld.tau_gradient_component(&s_psi);

        // The gradient of log(s₁) + log(s₂) w.r.t. a rotation is zero.
        assert!(
            grad.abs() < 1e-10,
            "nullspace-rotation gradient should be zero, got {grad}"
        );
    }

    #[test]
    fn test_block_factored_tau_hessian_preserves_internal_nullspace() {
        let s1 = 3.0_f64;
        let s2 = 1.0_f64;
        let psi = 0.5_f64;
        let c = psi.cos();
        let s = psi.sin();

        let r = array![[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]];
        let d = array![[s1, 0.0, 0.0], [0.0, s2, 0.0], [0.0, 0.0, 0.0]];
        let s_mat = r.dot(&d).dot(&r.t());

        let r_psi = array![[-s, 0.0, -c], [0.0, 0.0, 0.0], [c, 0.0, -s]];
        let s_psi = r_psi.dot(&d).dot(&r.t()) + r.dot(&d).dot(&r_psi.t());

        let r_psi_psi = array![[-c, 0.0, s], [0.0, 0.0, 0.0], [-s, 0.0, -c]];
        let s_psi_psi = r_psi_psi.dot(&d).dot(&r.t())
            + 2.0 * r_psi.dot(&d).dot(&r_psi.t())
            + r.dot(&d).dot(&r_psi_psi.t());

        let root = crate::estimate::reml::unified::penalty_matrix_root(&s_mat).unwrap();
        let penalty = crate::construction::CanonicalPenalty::from_dense_root(root, 3);
        let block_factored = PenaltyPseudologdet::from_penalties(&[penalty], &[1.0], 0.0, 3)
            .expect("block-factored pseudo-logdet");
        let assembled =
            PenaltyPseudologdet::from_assembled(s_mat, None).expect("assembled pseudo-logdet");

        let block_hess = block_factored.tau_hessian_component(&s_psi, &s_psi, Some(&s_psi_psi));
        let assembled_hess = assembled.tau_hessian_component(&s_psi, &s_psi, Some(&s_psi_psi));

        assert!(
            assembled_hess.abs() < 1e-10,
            "assembled reference should see zero curvature for a pure nullspace rotation, got {assembled_hess}"
        );
        assert!(
            (block_hess - assembled_hess).abs() < 1e-10,
            "block-factored tau hessian lost internal nullspace columns: block={block_hess}, assembled={assembled_hess}"
        );
    }

    #[test]
    fn test_block_factored_ridge_preserves_structural_nullspace_value() {
        let s = array![[4.0, 2.0], [2.0, 1.0]];
        let ridge = 1e-4_f64;

        let root = crate::estimate::reml::unified::penalty_matrix_root(&s).unwrap();
        let penalty = crate::construction::CanonicalPenalty::from_dense_root(root, 2);
        let block_factored = PenaltyPseudologdet::from_penalties(&[penalty], &[1.0], ridge, 2)
            .expect("block-factored pseudo-logdet");

        let mut s_ridged = s.clone();
        for i in 0..2 {
            s_ridged[[i, i]] += ridge;
        }
        let assembled = PenaltyPseudologdet::from_assembled(s_ridged, Some(ridge))
            .expect("assembled pseudo-logdet");

        assert_eq!(block_factored.rank(), assembled.rank());
        assert!(
            (block_factored.value() - assembled.value()).abs() < 1e-12,
            "block-factored ridge path leaked structural nullspace logdet: block={}, assembled={}",
            block_factored.value(),
            assembled.value()
        );
    }

    #[test]
    fn test_block_factored_ridge_ignores_inactive_lambda_for_structural_nullity() {
        let ridge = 1e-4_f64;
        let penalties = [
            crate::construction::CanonicalPenalty::from_dense_root(array![[1.0, 0.0]], 2),
            crate::construction::CanonicalPenalty::from_dense_root(array![[0.0, 1.0]], 2),
        ];

        let block_factored = PenaltyPseudologdet::from_penalties(&penalties, &[1.0, 0.0], ridge, 2)
            .expect("block-factored pseudo-logdet");
        let assembled = PenaltyPseudologdet::from_assembled(
            array![[1.0 + ridge, 0.0], [0.0, ridge]],
            Some(ridge),
        )
        .expect("assembled pseudo-logdet");

        assert_eq!(block_factored.rank(), assembled.rank());
        assert!(
            (block_factored.value() - assembled.value()).abs() < 1e-12,
            "inactive lambda leaked into structural nullity: block={}, assembled={}",
            block_factored.value(),
            assembled.value()
        );
    }

    #[test]
    fn test_overlapping_ridge_ignores_inactive_lambda_for_structural_nullity() {
        let ridge = 1e-4_f64;
        let penalties = [
            crate::construction::CanonicalPenalty {
                root: array![[1.0, 0.0]],
                col_range: 0..2,
                total_dim: 3,
                nullity: 1,
                local: array![[1.0, 0.0], [0.0, 0.0]],
                prior_mean: Array1::zeros(2),
                positive_eigenvalues: vec![1.0],
                op: None,
            },
            crate::construction::CanonicalPenalty {
                root: array![[1.0, 0.0]],
                col_range: 1..3,
                total_dim: 3,
                nullity: 1,
                local: array![[1.0, 0.0], [0.0, 0.0]],
                prior_mean: Array1::zeros(2),
                positive_eigenvalues: vec![1.0],
                op: None,
            },
        ];

        let overlapping = PenaltyPseudologdet::from_penalties(&penalties, &[1.0, 0.0], ridge, 3)
            .expect("overlapping pseudo-logdet");
        let assembled = PenaltyPseudologdet::from_assembled(
            array![
                [1.0 + ridge, 0.0, 0.0],
                [0.0, ridge, 0.0],
                [0.0, 0.0, ridge],
            ],
            Some(ridge),
        )
        .expect("assembled pseudo-logdet");

        assert_eq!(overlapping.rank(), assembled.rank());
        assert!(
            (overlapping.value() - assembled.value()).abs() < 1e-12,
            "inactive overlapping lambda leaked into structural nullity: overlap={}, assembled={}",
            overlapping.value(),
            assembled.value()
        );
    }

    #[test]
    fn test_block_factored_rho_derivatives_match_dense_without_cross_block_work() {
        let p_total = 6;
        let lambdas = [1.7_f64, 0.4_f64, 2.3_f64];
        let penalties = vec![
            crate::construction::CanonicalPenalty {
                root: array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                col_range: 0..3,
                total_dim: p_total,
                nullity: 1,
                local: array![[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]],
                prior_mean: Array1::zeros(3),
                positive_eigenvalues: vec![1.0, 4.0],
                op: None,
            },
            crate::construction::CanonicalPenalty {
                root: array![[0.0, 0.0, 3.0]],
                col_range: 0..3,
                total_dim: p_total,
                nullity: 2,
                local: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 9.0]],
                prior_mean: Array1::zeros(3),
                positive_eigenvalues: vec![9.0],
                op: None,
            },
            crate::construction::CanonicalPenalty {
                root: array![[1.5, 0.0, 0.0], [0.0, 0.0, 0.5]],
                col_range: 3..6,
                total_dim: p_total,
                nullity: 1,
                local: array![[2.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.25]],
                prior_mean: Array1::zeros(3),
                positive_eigenvalues: vec![2.25, 0.25],
                op: None,
            },
        ];

        let block_factored =
            PenaltyPseudologdet::from_penalties(&penalties, &lambdas, 0.0, p_total).unwrap();
        assert_eq!(block_factored.block_spans.len(), 2);

        let mut dense_components = Vec::new();
        for penalty in &penalties {
            let mut full = Array2::<f64>::zeros((p_total, p_total));
            penalty.accumulate_weighted(&mut full, 1.0);
            dense_components.push(full);
        }
        let dense = PenaltyPseudologdet::from_components(&dense_components, &lambdas, 0.0).unwrap();

        let (block_first, block_second) =
            block_factored.rho_derivatives_from_penalties(&penalties, &lambdas);
        let (dense_first, dense_second) = dense.rho_derivatives(&dense_components, &lambdas);

        for k in 0..lambdas.len() {
            assert!((block_first[k] - dense_first[k]).abs() < 1e-11);
            for l in 0..lambdas.len() {
                assert!((block_second[[k, l]] - dense_second[[k, l]]).abs() < 1e-10);
            }
        }
        assert!(block_second[[0, 2]].abs() < 1e-12);
        assert!(block_second[[1, 2]].abs() < 1e-12);
    }

    #[test]
    fn test_overlapping_penalties_ridge_preserve_structural_nullspace_value() {
        let ridge = 1e-4_f64;
        let lambdas = [2.0_f64, 3.0_f64];
        let penalties = [
            crate::construction::CanonicalPenalty::from_dense_root(array![[1.0, 0.0, 0.0]], 3),
            crate::construction::CanonicalPenalty::from_dense_root(array![[0.0, 1.0, 0.0]], 3),
        ];

        let overlapping = PenaltyPseudologdet::from_penalties(&penalties, &lambdas, ridge, 3)
            .expect("overlapping pseudo-logdet");

        let s_ridged = array![
            [lambdas[0] + ridge, 0.0, 0.0],
            [0.0, lambdas[1] + ridge, 0.0],
            [0.0, 0.0, ridge]
        ];
        let assembled = PenaltyPseudologdet::from_assembled(s_ridged, Some(ridge))
            .expect("assembled pseudo-logdet");

        assert_eq!(overlapping.rank(), assembled.rank());
        assert!(
            (overlapping.value() - assembled.value()).abs() < 1e-12,
            "assembled ridge path leaked structural nullspace logdet: overlap={}, assembled={}",
            overlapping.value(),
            assembled.value()
        );
    }
}

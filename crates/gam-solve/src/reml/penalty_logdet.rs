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

use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh, FaerSvd};

/// Which object a spectrum handed to [`PenaltyPseudologdet::from_eigensystem`]
/// was computed from — and therefore what accuracy `log|S_λ|₊` can have.
///
/// This is not a preference. `S_λ = Σ_k λ_k S_k` is a SUM OF SQUARES, so
/// forming it squares the conditioning of the objects it is built from: a mode
/// whose penalty ROOT sits a factor `t` below the dominant scale becomes an
/// eigenvalue a factor `t²` down. Every backward-stable factorization of the
/// assembled matrix therefore prices `log|S_λ|₊` to `O(ε·κ(S_λ))`, while the
/// same quantity taken from the stacked scaled ROOTS costs `O(ε·√κ(S_λ))`.
/// The outer smoothing search routinely drives `κ(S_λ)` past `1e14` (one λ at
/// its ceiling beside a null-space shrinkage λ near zero is enough), where the
/// two differ by twelve orders and the first is `±1e-2` — far coarser than the
/// criterion's own cost floor.
///
/// Carrying the provenance in the type is what stops a future caller from
/// silently handing an assembled spectrum to a value rule derived for a
/// root-scale one, or vice versa (#2644).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpectrumScale {
    /// `evals[i] = σ_i(A)²` for the stacked scaled roots
    /// `A = [√λ₁C₁; …; √λ_K C_K; √r I]`, `S_λ + rI = AᵀA`
    /// (see [`PenaltyPseudologdet::eigensystem_from_scaled_roots`]).
    /// Absolute error on `Σ log σ²` is `O(ε·√κ(S_λ))`.
    Root,
    /// Eigenvalues of the ASSEMBLED `Σ_k λ_k S_k + rI`. Absolute error on
    /// `Σ log σ` is `O(ε·κ(S_λ))` whichever factorization is used; reachable
    /// only by callers that never held the per-component roots.
    Assembled,
}

/// A root `R` of a symmetric PSD `S`, i.e. `RᵀR = S`, taken from `S`'s OWN
/// eigensystem and truncated at `S`'s own relative noise floor.
///
/// `S` here is always a λ-FREE unit penalty component, so its spectrum is
/// well-scaled and this eigendecomposition is a benign `O(ε)` operation — the
/// dynamic range that makes the weighted sum hard lives in the λ's, not here.
///
/// Rows are `√σ_i · u_iᵀ` for the modes above the floor, so `R` is
/// `rank × dim`. Reduced penalty projections built as `Rᵀ(RW)` instead of
/// `WᵀSW` are Grams: they cannot go negative, and a direction `w` orthogonal to
/// `range(S)` contributes `‖Rw‖² = O(ε²‖S‖)` rather than `wᵀSw = O(ε‖S‖)`.
/// Squaring the residual instead of carrying it linearly is the whole point —
/// see [`PenaltyPseudologdet::rho_derivatives`] (#2644).
fn psd_component_root(s: ndarray::ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let dim = s.nrows();
    if dim == 0 {
        return Ok(Array2::zeros((0, 0)));
    }
    let (evals, evecs) = s
        .eigh(Side::Lower)
        .map_err(|e| format!("penalty component root eigendecomposition failed: {e}"))?;
    let threshold = super::reml_outer_engine::positive_eigenvalue_threshold(
        evals
            .as_slice()
            .expect("eigh returns a freshly allocated contiguous eigenvalue array"),
    );
    let kept: Vec<usize> = (0..dim).filter(|&i| evals[i] > threshold).collect();
    let mut root = Array2::<f64>::zeros((kept.len(), dim));
    for (r, &i) in kept.iter().enumerate() {
        let scale = evals[i].sqrt();
        for c in 0..dim {
            root[[r, c]] = scale * evecs[[c, i]];
        }
    }
    Ok(root)
}

/// Check whether penalty ranges decompose into independent exact blocks.
///
/// Multiple smoothing components may share the same block (for example tensor
/// product marginals); those can still be factorized block-local.  Only partial
/// overlaps force the dense assembled fallback.
pub(crate) fn are_penalties_block_factored(
    penalties: &[gam_terms::construction::CanonicalPenalty],
) -> bool {
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

/// Partition dense penalty components into disjoint diagonal coordinate blocks.
///
/// Each `S_k` occupies a contiguous coordinate support `[min, max)` (the rows /
/// columns where it has any nonzero entry). Components whose supports overlap
/// are merged into one block; components with disjoint supports become separate
/// blocks. Returns the sorted, non-overlapping block ranges `[(start, end), …]`
/// when the union support partitions into **more than one** disjoint block, so
/// the per-block λ-coercivity threshold (#1237) can be applied; returns `None`
/// when there is a single connected block (the global threshold is already
/// block-local in that case) or when there are no penalized coordinates.
///
/// This mirrors `are_penalties_block_factored` / `from_penalties_block_factored`
/// but recovers the block structure from the dense component supports, which is
/// the only structure available on the `from_components` path (custom-family /
/// multinomial), where penalties arrive as dense `Array2` without
/// `CanonicalPenalty` column-range metadata.
fn disjoint_diagonal_blocks(s_k_matrices: &[Array2<f64>]) -> Option<Vec<(usize, usize)>> {
    if s_k_matrices.is_empty() {
        return None;
    }
    let p_dim = s_k_matrices[0].nrows();
    if p_dim == 0 {
        return None;
    }

    // Per-component support [min, max] over coordinates with any nonzero entry.
    // A symmetric S_k is supported on coordinate i iff its i-th row (equivalently
    // column) has a nonzero, so it suffices to scan row magnitudes.
    let mut spans: Vec<(usize, usize)> = Vec::with_capacity(s_k_matrices.len());
    for s_k in s_k_matrices {
        let mut lo: Option<usize> = None;
        let mut hi: usize = 0;
        for i in 0..p_dim {
            let nz = (0..p_dim).any(|j| s_k[[i, j]] != 0.0);
            if nz {
                if lo.is_none() {
                    lo = Some(i);
                }
                hi = i;
            }
        }
        if let Some(lo) = lo {
            spans.push((lo, hi + 1));
        }
    }
    if spans.is_empty() {
        return None;
    }

    // Merge overlapping/adjacent-by-overlap spans into maximal disjoint blocks.
    spans.sort_unstable();
    let mut blocks: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
    for (start, end) in spans {
        match blocks.last_mut() {
            // Overlap (strict: a coordinate shared by both spans) merges the two
            // into one block. Touching-but-disjoint spans (`prev_end == start`)
            // stay separate — they share no coordinate, so their penalties never
            // couple and each block carries its own λ.
            Some(last) if start < last.1 => last.1 = last.1.max(end),
            _ => blocks.push((start, end)),
        }
    }

    if blocks.len() > 1 { Some(blocks) } else { None }
}

/// Structural rank of a set of penalty components — ONE rule, shared with the
/// reparameterization's penalized/null split
/// (`gam_terms::construction::balanced_penalty_structural_rank`): the rank of
/// the Frobenius-balanced sum `Σ_k S_k/‖S_k‖_F` at its relative cut. Components
/// whose `λ` is zero are excluded, exactly as before; what changed is that the
/// rank is no longer read from the UNWEIGHTED sum at `100·p·ε·max`, which
/// disagreed with the split whenever one component's norm was small against
/// another's (gam#2454: `logdet_rank = 10` against `penalized_rank = 9` on the
/// Matérn double-penalty ladder, a criterion with no interior optimum in ρ).
fn structural_rank_from_components<'a, I>(components: I, p_dim: usize) -> Result<usize, String>
where
    I: IntoIterator<Item = ndarray::ArrayView2<'a, f64>>,
{
    gam_terms::construction::balanced_penalty_structural_rank(
        components.into_iter().map(|view| (view, 0..p_dim)),
        p_dim,
    )
    .map_err(|error| format!("PenaltyPseudologdet structural rank: {error}"))
}
fn structural_rank_from_canonical_penalties(
    penalties: &[gam_terms::construction::CanonicalPenalty],
    lambdas: &[f64],
    p_total: usize,
) -> Result<usize, String> {
    gam_terms::construction::balanced_penalty_structural_rank(
        penalties
            .iter()
            .enumerate()
            .filter(|(k, _)| *k < lambdas.len() && lambdas[*k] > 0.0)
            .map(|(_, penalty)| (penalty.local_ref().view(), penalty.col_range.clone())),
        p_total,
    )
    .map_err(|error| format!("PenaltyPseudologdet structural rank: {error}"))
}
/// Result of a penalty pseudo-logdet computation.
///
/// Holds the eigendecomposition and precomputed W-factor so that derivative
/// queries are efficient without redundant factorizations.
#[derive(Clone, Debug)]
pub(crate) struct PenaltyBlockSpan {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) rank_start: usize,
    pub(crate) rank_end: usize,
}

#[derive(Clone, Debug)]
pub struct PenaltyPseudologdet {
    /// W factor: p × rank, with W W^T = S⁺.
    pub(crate) w_factor: Array2<f64>,
    /// Null-space eigenvectors U₀: p × nullity (for moving-nullspace corrections).
    /// `None` if nullity == 0.
    pub(crate) u_null: Option<Array2<f64>>,
    /// Inverse squared eigenvalues on the positive eigenspace: σ_i^{-2}.
    /// Length = rank. Used for the moving-nullspace correction: tr(Σ₊⁻² L_i L_j^T).
    pub(crate) inv_evals_sq: Array1<f64>,
    /// Positive eigenspace rank.
    pub(crate) rank: usize,
    /// log|S|₊ = Σ log σ_i for positive eigenvalues.
    pub(crate) value: f64,
    /// Block/rank spans when the penalty eigenspace was assembled from disjoint blocks.
    pub(crate) block_spans: Vec<PenaltyBlockSpan>,
}

impl PenaltyPseudologdet {
    /// Compute tr(A B) = Σ_i Σ_k A[i,k] B[k,i] without materializing the product.
    #[inline]
    pub(crate) fn trace_dense_product(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let diag_len = a.nrows().min(b.ncols());
        let inner_len = a.ncols().min(b.nrows());
        let a = a.as_standard_layout();
        let b = b.as_standard_layout();
        let a_values = a
            .as_slice()
            .expect("standard-layout dense trace left operand is contiguous");
        let b_values = b
            .as_slice()
            .expect("standard-layout dense trace right operand is contiguous");
        let a_cols = a.ncols();
        let b_cols = b.ncols();
        let mut total = 0.0;
        for i in 0..diag_len {
            for k in 0..inner_len {
                total += a_values[i * a_cols + k] * b_values[k * b_cols + i];
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
    /// This is the preferred entry point for REML logdet computation.  For
    /// canonical penalties, the structural positive rank is computed from the
    /// unweighted active penalty span and then applied to the current weighted
    /// spectrum. That keeps real range-space modes active even when one lambda
    /// is tiny relative to another same-block penalty.
    pub fn from_penalties(
        penalties: &[gam_terms::construction::CanonicalPenalty],
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
            let structural_rank =
                structural_rank_from_canonical_penalties(penalties, lambdas, p_total)?;
            let ridge_hint = if ridge > 0.0 { Some(ridge) } else { None };
            // #2299/#2316-adjacent: the hinted split must see the spectrum at
            // ROOT scale (stacked scaled square roots), not the assembled
            // matrix's squared conditioning — see
            // `eigensystem_from_scaled_roots`.
            let components: Vec<(f64, ndarray::ArrayView2<'_, f64>, std::ops::Range<usize>)> =
                penalties
                    .iter()
                    .enumerate()
                    .map(|(k, cp)| {
                        let lambda = if k < lambdas.len() { lambdas[k] } else { 0.0 };
                        (lambda, cp.local.view(), cp.col_range.clone())
                    })
                    .collect();
            Self::from_scaled_components_with_rank_hint(
                &s_total,
                &components,
                ridge_hint,
                Some(structural_rank),
            )
        }
    }

    /// Block-factored logdet: eigendecompose each disjoint block independently.
    ///
    /// The total logdet is the sum of per-block logdets. The W-factor is
    /// block-diagonal (embedded in p_total space).
    pub(crate) fn from_penalties_block_factored(
        penalties: &[gam_terms::construction::CanonicalPenalty],
        lambdas: &[f64],
        ridge: f64,
        p_total: usize,
    ) -> Result<Self, String> {
        use ndarray::s;

        // Collect block ranges and assemble per-block combined penalties.
        // Each penalty contributes to its own block (disjoint assumption).
        struct BlockData {
            pub(crate) start: usize,
            pub(crate) end: usize,
            pub(crate) local: Array2<f64>,
            /// `(penalty index, λ)` of every component summed into `local`,
            /// so the hinted split can be computed at root scale.
            pub(crate) parts: Vec<(usize, f64)>,
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
                bd.parts.push((k, lambda));
            } else {
                let bd = cp.block_dim();
                let mut local = Array2::<f64>::zeros((bd, bd));
                local.scaled_add(lambda, &cp.local);
                blocks.push(BlockData {
                    start: r.start,
                    end: r.end,
                    local,
                    parts: vec![(k, lambda)],
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

        // Coordinates no penalty block covers are in the structural null space
        // of `Σ λ_k S_k`, ridge or no ridge. The ridge stabilises a factorization;
        // it is not a penalty, and the hinted split INSIDE a block already
        // treats a ridge-only direction as null (gam#2454: counting the
        // uncovered intercept as a rank-1 "ridge block" reported `rank = 10`
        // beside an `E` of 9 rows and a `log|S|₊` growing at exactly 9 per
        // unit ρ, and put `1/ridge²` into the inverse spectrum).
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
            pub(crate) start: usize,
            pub(crate) end: usize,
            pub(crate) w_local: Array2<f64>,
            pub(crate) u_null_local: Array2<f64>,
            pub(crate) inv_evals_sq: Vec<f64>,
            pub(crate) value: f64,
            pub(crate) rank: usize,
            pub(crate) nullity: usize,
        }

        let ridge_hint = if ridge > 0.0 { Some(ridge) } else { None };
        let process_block = |bd: &BlockData| -> Result<BlockResult, String> {
            let structural_rank = structural_rank_from_components(
                bd.parts
                    .iter()
                    .filter(|&&(_, lambda)| lambda > 0.0)
                    .map(|&(k, _)| penalties[k].local.view()),
                bd.end - bd.start,
            )?;
            // Root-scale spectrum for the hinted split (see
            // `eigensystem_from_scaled_roots`): the assembled block's eigh
            // cannot resolve a structurally-positive mode once the block's λ
            // ratio exceeds ~1/ε, and mid-optimization probes do reach that.
            let block_width = bd.end - bd.start;
            let components: Vec<(f64, ndarray::ArrayView2<'_, f64>, std::ops::Range<usize>)> = bd
                .parts
                .iter()
                .map(|&(k, lambda)| (lambda, penalties[k].local.view(), 0..block_width))
                .collect();
            let block_pld = Self::from_scaled_components_with_rank_hint(
                &bd.local,
                &components,
                ridge_hint,
                Some(structural_rank),
            )?;
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
        let block_results: Vec<BlockResult> = if rayon::current_thread_index().is_some() {
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
        let uncovered_nullity = covered.iter().filter(|&&c| !c).count();
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
                if !c {
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

        // λ-COERCIVITY ON SEPARATING DIRECTIONS (#1237). The penalty sum
        // S(λ) = Σ_k λ_k S_k decomposes into disjoint diagonal blocks whenever
        // the penalized terms occupy disjoint coordinate ranges (the common
        // multi-smooth case: each `s(xⱼ)` penalizes its own column block; the
        // multinomial K-1 class blocks are themselves separate `from_components`
        // calls, each carrying that class's per-term penalties). When one term's
        // `λ_t → 0` (the near-separable signature: a wigglier spline separates
        // the classes ever better, so the outer REML search drives that λ down),
        // ALL of S_t's eigenvalues scale by λ_t. If the positive/null split is
        // taken against the GLOBAL spectrum's `max|e|` — set by the other,
        // moderately-penalized terms — then S_t's genuine range-space modes
        // `λ_t σ_t` slide below the relative noise floor `100·p·ε·max|e|` and are
        // misclassified as structural null. Dropping them deletes their
        // `−½ log(λ_t σ_t) = −½ ρ_t − const` contribution to `−½ log|S|₊`, which
        // is exactly the coercivity term that would otherwise make V(ρ) blow up
        // as ρ_t → −∞. Without it the outer criterion is monotone-decreasing in
        // ρ_t, λ slams to its lower box bound and bounces, and the search never
        // certifies a stationary point (the #1082 penguin `_nnet` timeout).
        //
        // Thresholding PER disjoint diagonal block fixes this: within a single
        // block governed by one λ, every eigenvalue scales by that λ, so the
        // relative floor `100·b·ε·max_block|e|` scales identically and λ CANCELS
        // — the dropped set is the genuine λ-invariant structural null of the
        // unit penalty, and a barely-penalized real mode keeps its coercivity.
        // This mirrors `from_penalties_block_factored` (which already thresholds
        // block-local from `CanonicalPenalty` col-ranges); here we recover the
        // same block structure from the dense components' union support, so the
        // custom-family/multinomial value (joint_newton) and gradient
        // (`rho_derivatives` via `compute_block_penalty_logdet_derivs`) paths —
        // both routed through `from_components` — agree by construction.
        //
        // The per-block factorization keeps the COMPONENTS, not just the
        // block's assembled slice: `log|S_λ|₊` is summed over blocks, so a
        // block priced off its assembled slice contributes that slice's
        // `O(ε·κ)` error to the total (#2644, and see [`SpectrumScale`]). A
        // single smooth's own {wiggliness, null-space shrinkage} pair inside
        // ONE block is exactly where the λ ratio blows up, so this is not a
        // cross-block concern that block-splitting would fix by itself.
        if let Some(blocks) = disjoint_diagonal_blocks(s_k_matrices) {
            if blocks.len() > 1 {
                return Self::from_component_blocks(
                    &s_total,
                    s_k_matrices,
                    lambdas,
                    ridge_hint,
                    &blocks,
                );
            }
        }

        let structural_rank = structural_rank_from_components(
            s_k_matrices
                .iter()
                .enumerate()
                .filter(|(k, _)| *k < lambdas.len() && lambdas[*k] > 0.0)
                .map(|(_, s_k)| s_k.view()),
            p_dim,
        )?;
        // Root-scale spectrum for the hinted split (see
        // `eigensystem_from_scaled_roots`).
        let components: Vec<(f64, ndarray::ArrayView2<'_, f64>, std::ops::Range<usize>)> =
            s_k_matrices
                .iter()
                .enumerate()
                .map(|(k, s_k)| {
                    let lambda = if k < lambdas.len() { lambdas[k] } else { 0.0 };
                    (lambda, s_k.view(), 0..p_dim)
                })
                .collect();
        Self::from_scaled_components_with_rank_hint(
            &s_total,
            &components,
            ridge_hint,
            Some(structural_rank),
        )
    }

    /// Build from a pre-assembled penalty matrix.
    ///
    /// `s_total` must be `Σ_k λ_k S_k` plus, if `ridge` is `Some(r)`, an
    /// additive `r·I` already applied to the diagonal. The caller is expected
    /// to have assembled the matrix in exactly that form.
    ///
    /// By default, the positive/null eigenspace split is determined entirely
    /// from the eigenspectrum of `s_total`:
    ///
    /// * When `ridge` is `Some(r)`, a direction is structurally null iff its
    ///   eigenvalue is within machine-precision tolerance of `r` (i.e. the
    ///   unridged `Σ λ_k S_k` has a zero eigenvalue along that direction).
    /// * When `ridge` is `None`, a direction is null iff its eigenvalue is at
    ///   or below the relative noise floor `positive_eigenvalue_threshold`
    ///   (`100 · p · ε · max|e|`).
    ///
    /// This no-hint path is kept for callers that only have an assembled
    /// matrix. Canonical callers should prefer
    /// `Self::from_assembled_with_rank_hint` when multiple active penalty
    /// components share a block, because the current weighted spectrum can hide
    /// small but structurally real modes below a relative threshold.
    pub fn from_assembled(s_total: Array2<f64>, ridge: Option<f64>) -> Result<Self, String> {
        Self::from_assembled_with_rank_hint(s_total, ridge, None)
    }

    /// Build from a pre-assembled penalty matrix, optionally pinning the
    /// structural positive rank.
    ///
    /// A rank hint is needed for canonical overlapping penalties such as a
    /// B-spline bend penalty plus its Marra-Wood null-space shrinkage ridge:
    /// the two components live on the same coefficient block but can have very
    /// different lambdas. The current weighted spectrum can then contain
    /// small-but-real positive eigenvalues far below the block's largest
    /// eigenvalue. Those directions are still part of the structural penalty
    /// range and contribute `log(lambda)` coercivity to REML; dropping them by a
    /// relative threshold removes the cost term whose derivative keeps the
    /// bend lambda moving upward (#1266).
    pub(crate) fn from_assembled_with_rank_hint(
        s_total: Array2<f64>,
        ridge: Option<f64>,
        rank_hint: Option<usize>,
    ) -> Result<Self, String> {
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
        Self::from_eigensystem(
            &s_total,
            evals,
            evecs,
            ridge,
            rank_hint,
            SpectrumScale::Assembled,
        )
    }

    /// Eigensystem of `Σ_k λ_k S_k (+ ridge·I)` computed from the STACKED
    /// SCALED SQUARE ROOTS, never from the assembled weighted matrix.
    ///
    /// Forming `Σ λ_k S_k` and eigendecomposing it squares the conditioning:
    /// a structurally-positive mode whose singular value is a factor `t`
    /// below the dominant scale becomes an eigenvalue a factor `t²` below it,
    /// and once `t² < ε` the eigh result for that mode is pure roundoff — it
    /// can come out NEGATIVE, which the structural rank hint then rejects and
    /// a mid-optimization outer evaluation dies (observed as
    /// "structural rank hint selected non-positive eigenvalue -8e-5" on a
    /// plain `y ~ s(x)` fit whose bend/shrinkage λ ratio crossed ~1e16 during
    /// an ARC probe). Stacking the per-component scaled roots
    /// `A = [√λ_1·C_1; …; √λ_K·C_K; √r·I]` with `S_k = C_kᵀC_k` and taking
    /// singular values keeps every mode at its ROOT scale: the same mode is
    /// visible until `t < ε`, i.e. down to weighted-eigenvalue ratios of
    /// ~1e-32 — beyond any λ ratio the optimizer can express. Eigenvalues are
    /// `σ_i²` and eigenvectors are the right singular vectors, so the result
    /// plugs into the same [`Self::from_eigensystem`] tail.
    ///
    /// Each component root is taken from the eigh of its UNIT matrix (a
    /// λ-free, well-scaled problem thresholded on its own spectrum), placed
    /// into the stacked matrix at `col_range`.
    fn eigensystem_from_scaled_roots(
        components: &[(f64, ndarray::ArrayView2<'_, f64>, std::ops::Range<usize>)],
        ridge: Option<f64>,
        p_dim: usize,
    ) -> Result<(Array1<f64>, Array2<f64>), String> {
        // Collect scaled root rows.
        let mut rows: Vec<Array1<f64>> = Vec::new();
        for (lambda, local, col_range) in components {
            if !(lambda.is_finite() && *lambda >= 0.0) {
                return Err(format!(
                    "PenaltyPseudologdet scaled-root eigensystem requires finite λ ≥ 0, got {lambda}"
                ));
            }
            if *lambda == 0.0 {
                continue;
            }
            let bd = local.nrows();
            if local.ncols() != bd || col_range.len() != bd || col_range.end > p_dim {
                return Err(format!(
                    "PenaltyPseudologdet component block {bd}x{} does not fit col_range {col_range:?} in dimension {p_dim}",
                    local.ncols()
                ));
            }
            let (unit_evals, unit_evecs) = local.eigh(Side::Lower).map_err(|e| {
                format!("PenaltyPseudologdet component eigendecomposition failed: {e}")
            })?;
            let unit_threshold = super::reml_outer_engine::positive_eigenvalue_threshold(
                unit_evals
                    .as_slice()
                    .expect("eigh returns a freshly allocated contiguous eigenvalue array"),
            );
            for (idx, &ev) in unit_evals.iter().enumerate() {
                if ev > unit_threshold {
                    let scale = (lambda * ev).sqrt();
                    let mut row = Array1::<f64>::zeros(p_dim);
                    for (li, gi) in col_range.clone().enumerate() {
                        row[gi] = scale * unit_evecs[[li, idx]];
                    }
                    rows.push(row);
                }
            }
        }
        if let Some(r) = ridge
            && r > 0.0
        {
            let scale = r.sqrt();
            for i in 0..p_dim {
                let mut row = Array1::<f64>::zeros(p_dim);
                row[i] = scale;
                rows.push(row);
            }
        }
        // Pad with zero rows so the thin SVD's right factor spans all of ℝᵖ
        // (zero rows change neither the singular values nor the right
        // singular vectors).
        while rows.len() < p_dim {
            rows.push(Array1::<f64>::zeros(p_dim));
        }
        let mut stacked = Array2::<f64>::zeros((rows.len(), p_dim));
        for (i, row) in rows.iter().enumerate() {
            stacked.row_mut(i).assign(row);
        }
        let (_, singular, vt) = stacked
            .svd(false, true)
            .map_err(|_| "PenaltyPseudologdet stacked-root SVD did not converge".to_string())?;
        let vt = vt.ok_or_else(|| {
            "PenaltyPseudologdet stacked-root SVD returned no right factor".to_string()
        })?;
        if vt.nrows() < p_dim || singular.len() < p_dim {
            return Err(format!(
                "PenaltyPseudologdet stacked-root SVD returned a thin right factor ({}x{}, {} singular values) for dimension {p_dim}",
                vt.nrows(),
                vt.ncols(),
                singular.len()
            ));
        }
        // Singular values are descending; from_eigensystem expects ascending
        // eigenvalues with matching eigenvector columns.
        let mut evals = Array1::<f64>::zeros(p_dim);
        let mut evecs = Array2::<f64>::zeros((p_dim, p_dim));
        for j in 0..p_dim {
            let src = p_dim - 1 - j;
            let sv = singular[src];
            evals[j] = sv * sv;
            for row in 0..p_dim {
                evecs[[row, j]] = vt[[src, row]];
            }
        }
        Ok((evals, evecs))
    }

    /// Build with a structural rank hint from the weighted components
    /// themselves, computing the spectrum via the stacked scaled square
    /// roots (see [`Self::eigensystem_from_scaled_roots`] for why the
    /// assembled-matrix eigh is not accurate enough for the hinted split).
    ///
    /// `s_total` must be the assembled `Σ λ_k S_k (+ ridge·I)` over the same
    /// components; it feeds only the full-rank Cholesky value fast path.
    pub(crate) fn from_scaled_components_with_rank_hint(
        s_total: &Array2<f64>,
        components: &[(f64, ndarray::ArrayView2<'_, f64>, std::ops::Range<usize>)],
        ridge: Option<f64>,
        rank_hint: Option<usize>,
    ) -> Result<Self, String> {
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
        let (evals, evecs) = Self::eigensystem_from_scaled_roots(components, ridge, p_dim)?;
        Self::from_eigensystem(s_total, evals, evecs, ridge, rank_hint, SpectrumScale::Root)
    }

    /// Assemble the pseudo-logdet from a precomputed eigensystem of
    /// `s_total` (ascending eigenvalues, matching eigenvector columns).
    ///
    /// `s_total` itself is retained only for the assembled-spectrum
    /// log-determinant fallback; the positive/null split and the W-factor come
    /// entirely from `(evals, evecs)`, so a caller with a MORE accurate
    /// eigensystem than `eigh(s_total)` (see
    /// [`Self::from_scaled_components_with_rank_hint`]) plugs it in here.
    ///
    /// `scale` says which object the spectrum came from, and that decides how
    /// `log|S|₊` may be priced — see [`SpectrumScale`].
    fn from_eigensystem(
        s_total: &Array2<f64>,
        evals: Array1<f64>,
        evecs: Array2<f64>,
        ridge: Option<f64>,
        rank_hint: Option<usize>,
        scale: SpectrumScale,
    ) -> Result<Self, String> {
        let p_dim = s_total.nrows();
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
        let noise_band = super::reml_outer_engine::positive_eigenvalue_threshold(
            evals
                .as_slice()
                .expect("eigh returns a freshly allocated contiguous eigenvalue array"),
        );
        let boundary = match ridge {
            Some(r) if r > 0.0 => r + noise_band,
            _ => noise_band,
        };
        let mut positive_indices = Vec::with_capacity(p_dim);
        let mut null_indices = Vec::with_capacity(p_dim);
        if let Some(rank_hint) = rank_hint {
            let rank_hint = rank_hint.min(p_dim);
            let first_positive = p_dim.saturating_sub(rank_hint);
            for idx in 0..p_dim {
                if idx >= first_positive {
                    let eval = evals[idx];
                    if !(eval.is_finite() && eval > 0.0) {
                        return Err(format!(
                            "PenaltyPseudologdet structural rank hint {rank_hint} selected \
                             non-positive eigenvalue {eval} at sorted index {idx}"
                        ));
                    }
                    positive_indices.push(idx);
                } else {
                    null_indices.push(idx);
                }
            }
        } else {
            for (idx, &eval) in evals.iter().enumerate() {
                if eval > boundary {
                    positive_indices.push(idx);
                } else {
                    null_indices.push(idx);
                }
            }
        }
        let rank = positive_indices.len();
        let nullity = null_indices.len();

        // Value: log|S|₊ = Σ log σ_i over the positive eigenspace.
        //
        // WHICH FACTORIZATION IS ALLOWED TO PRICE IT is decided by `scale`,
        // because the two available objects do not carry the same accuracy and
        // the difference is the whole of #2644.
        //
        // `SpectrumScale::Root` — `evals` are `σ_i(A)²` for the stacked scaled
        // roots `A = [√λ₁C₁; …; √λ_K C_K; √r I]` with `S_λ + rI = AᵀA`. The
        // SVD's backward error is `O(ε·σ_max(A))` ABSOLUTE, so the RELATIVE
        // error on the smallest retained `σ²` — and hence the absolute error
        // on `Σ log σ²` — is `O(ε·√κ(S_λ))`. Sum the eigenvalues.
        //
        // `SpectrumScale::Assembled` — the caller only ever held
        // `Σ λ_k S_k + rI`. Both routes off that matrix carry `O(ε·κ(S_λ))`:
        // `eigh` perturbs each eigenvalue by `O(ε·‖S_λ‖)`, and the Cholesky
        // log-determinant inherits `tr(S⁻¹ΔS) = O(ε·κ)` from its own backward
        // error. Measured on the exact spectrum this issue was filed against
        // (`κ = 1.4e14`, 150 draws of an `ε·|S|` symmetric perturbation):
        //   Cholesky  std 1.40e-2  |bias| 7.5e-3
        //   eigh sum  std 2.27e-2  |bias| 7.5e-3
        //   roots     std 4.8e-15  |bias| 1.4e-14
        // so Cholesky stays preferred HERE — it is ~1.6x tighter and nothing
        // better is reachable from an assembled matrix — but it is twelve
        // orders worse than the root-scale route and must never be used in its
        // place. That substitution is what put ±1.2e-2 of noise on the outer
        // REML criterion (≈340x the relative cost floor the line search is
        // asked to resolve) and stalled the outer optimizer at |Pg| ≈ 1e-3.
        //
        // Rank-deficient assembled case (nullity > 0): keep the eigen-sum over
        // `positive_indices`. Subtracting a `Σ log(eigval_null)` reconstruction
        // from the Cholesky log-det would push eigh noise on those null
        // eigenvalues (~ ε·max|e| / ridge in relative log terms) into
        // `value()`, which is materially worse than direct
        // `log(eigval_positive)` summation when the positive spectrum lies well
        // above the ridge (gam `test_components_with_stale_nullity_*`).
        let eigen_sum = || -> f64 { positive_indices.iter().map(|&idx| evals[idx].ln()).sum() };
        let value: f64 = match scale {
            SpectrumScale::Root => eigen_sum(),
            SpectrumScale::Assembled => {
                if nullity == 0
                    && matches!(ridge, Some(r) if r > 0.0)
                    && let Ok(fac) = s_total.cholesky(Side::Lower)
                {
                    2.0 * fac.diag().iter().map(|d| d.ln()).sum::<f64>()
                } else {
                    eigen_sum()
                }
            }
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

    /// Assemble the pseudo-logdet block-locally over disjoint diagonal blocks,
    /// factorizing each block from its own PENALTY COMPONENTS.
    ///
    /// `s_total` is the fully assembled `Σ_k λ_k S_k (+ ridge·I)` and is used
    /// only to slice each block's assembled local; `blocks` are the sorted,
    /// non-overlapping coordinate ranges from [`disjoint_diagonal_blocks`], and
    /// `s_k_matrices`/`lambdas` are the same components `s_total` was summed
    /// from. Each block is factorized in isolation via
    /// [`Self::from_scaled_components_with_rank_hint`], so:
    ///
    /// * the positive/null split uses that block's OWN relative floor
    ///   `100·b·ε·max_block|e|`. Within a block every eigenvalue scales by that
    ///   block's single λ, so the floor scales with it and λ cancels: a
    ///   near-separable term keeps its genuine range-space modes (and their
    ///   `−½ log(λ σ)` coercivity) instead of having them slide below a global
    ///   floor set by the other, moderately-penalized blocks (#1237);
    /// * the block's `log|·|₊` is priced at ROOT scale, `O(ε·√κ)` rather than
    ///   the `O(ε·κ)` an assembled block slice can reach — see
    ///   [`SpectrumScale`] (#2644). The per-block values are SUMMED, so one
    ///   block factorized off its assembled slice contributes that slice's
    ///   error to the total; block-splitting alone does not bound `κ`, because
    ///   the {wiggliness, null-space shrinkage} λ pair of a single smooth lives
    ///   inside ONE block.
    ///
    /// The result is identical in shape to [`Self::from_assembled`] on `s_total`
    /// (a block-diagonal W-factor embedded in the full p×p space, the summed
    /// value, the stacked null basis) and carries `block_spans` for downstream
    /// per-block queries, exactly like [`Self::from_penalties_block_factored`].
    fn from_component_blocks(
        s_total: &Array2<f64>,
        s_k_matrices: &[Array2<f64>],
        lambdas: &[f64],
        ridge: Option<f64>,
        blocks: &[(usize, usize)],
    ) -> Result<Self, String> {
        let p_total = s_total.nrows();

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

        // Eigendecompose each disjoint block in its own frame.
        let mut covered = vec![false; p_total];
        for &(start, end) in blocks {
            for i in start..end {
                covered[i] = true;
            }
        }

        let mut block_results: Vec<BlockResult> = Vec::with_capacity(blocks.len());
        for &(start, end) in blocks {
            let width = end - start;
            let local = s_total.slice(s![start..end, start..end]).to_owned();
            // The components restricted to this block. `disjoint_diagonal_blocks`
            // guarantees each `S_k`'s support lies inside exactly one block, so
            // the restriction is lossless: a component supported elsewhere is
            // identically zero here and is dropped.
            let component_views: Vec<(f64, ndarray::ArrayView2<'_, f64>)> = s_k_matrices
                .iter()
                .enumerate()
                .filter_map(|(k, s_k)| {
                    let lambda = if k < lambdas.len() { lambdas[k] } else { 0.0 };
                    if lambda <= 0.0 {
                        return None;
                    }
                    let view = s_k.slice(s![start..end, start..end]);
                    view.iter().any(|&v| v != 0.0).then_some((lambda, view))
                })
                .collect();
            // Structural (λ-free) rank of this block's union support, the same
            // hint `from_penalties_block_factored` forms from its own parts.
            let structural_rank = structural_rank_from_components(
                component_views.iter().map(|(_, view)| *view),
                width,
            )?;
            let components: Vec<(f64, ndarray::ArrayView2<'_, f64>, std::ops::Range<usize>)> =
                component_views
                    .into_iter()
                    .map(|(lambda, view)| (lambda, view, 0..width))
                    .collect();
            let block_pld = Self::from_scaled_components_with_rank_hint(
                &local,
                &components,
                ridge,
                Some(structural_rank),
            )?;
            let nullity = block_pld.u_null.as_ref().map_or(0, Array2::ncols);
            block_results.push(BlockResult {
                start,
                end,
                w_local: block_pld.w_factor,
                u_null_local: block_pld
                    .u_null
                    .unwrap_or_else(|| Array2::<f64>::zeros((end - start, 0))),
                inv_evals_sq: block_pld.inv_evals_sq.to_vec(),
                value: block_pld.value,
                rank: block_pld.rank,
                nullity,
            });
        }

        // Coordinates no penalty block covers are structurally null, ridge or
        // no ridge — the same rule as `from_penalties_block_factored` and as the
        // hinted split inside a block (gam#2454).
        let total_rank: usize = block_results.iter().map(|br| br.rank).sum();
        let total_value: f64 = block_results.iter().map(|br| br.value).sum();

        let mut w_factor = Array2::<f64>::zeros((p_total, total_rank));
        let mut inv_evals_sq = Array1::<f64>::zeros(total_rank);
        let mut block_spans = Vec::with_capacity(block_results.len());
        let mut col_offset = 0;
        for br in &block_results {
            if br.rank > 0 {
                w_factor
                    .slice_mut(s![br.start..br.end, col_offset..col_offset + br.rank])
                    .assign(&br.w_local);
                for (i, &v) in br.inv_evals_sq.iter().enumerate() {
                    inv_evals_sq[col_offset + i] = v;
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

        let block_nullity: usize = block_results.iter().map(|br| br.nullity).sum();
        let uncovered_nullity = covered.iter().filter(|&&c| !c).count();
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
                if !c {
                    u0[[idx, null_col]] = 1.0;
                    null_col += 1;
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
            rank: total_rank,
            value: total_value,
            block_spans,
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
    pub(crate) fn reduced(&self, m: &Array2<f64>) -> Array2<f64> {
        let wt_m = self.w_factor.t().dot(m);
        wt_m.dot(&self.w_factor)
    }

    /// The reduced penalty projection `Y = WᵀSW` for a PSD `S = RᵀR`, built as
    /// the GRAM of `M = R·W` instead of by contracting `S`.
    ///
    /// Algebraically identical; numerically not. `WᵀSW` carries `S`'s roundoff
    /// LINEARLY: a `W` column `w` on a direction where `S` vanishes still
    /// returns `wᵀSw = O(ε‖S‖‖w‖²)`, and `W`'s columns are scaled by
    /// `σ_i(S_λ)^{-1/2}`, so that residual is divided by the SMALLEST penalty
    /// eigenvalue and then multiplied by `λ_k` in `∂_ρk L = λ_k tr(Y_k)` — an
    /// `O(ε·κ(S_λ))` error on a quantity bounded by `rank(S_k)`.
    ///
    /// Measured on the dense two-component fixture at `λ` ratio `1.4e14`
    /// (`rotated_extreme_lambda_ratio_rho_gradient_matches_central_difference`):
    /// the contracted form gives `3.00758` where the exact value and the central
    /// difference of `value()` both give `3.0000000000`. At the `κ ≈ 3.5e19` the
    /// `te(a,b)` witness of #2644 reaches, the same error term is `O(1e3)`.
    ///
    /// The Gram form squares that residual instead: `‖Rw‖² = O(ε²‖S‖‖w‖²)`. It
    /// is also PSD by construction, so `tr(Y)` can never come out negative.
    fn reduced_from_root(m: &Array2<f64>) -> Array2<f64> {
        m.t().dot(m)
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
    pub(crate) fn leakage(&self, m: &Array2<f64>) -> Option<Array2<f64>> {
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
    pub(crate) fn moving_nullspace_correction(
        &self,
        wt_si_u0: &Array2<f64>,
        wt_sj_u0: &Array2<f64>,
    ) -> f64 {
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

        // Reduced representations: Y_k = W^T S_k W (unscaled), formed as the
        // GRAM of `M_k = R_k W` with `R_kᵀR_k = S_k` rather than by contracting
        // `S_k` itself. See `reduced_from_root`.
        // These K projections are independent and dominate derivative time for
        // large bases, so evaluate them in parallel outside existing rayon jobs.
        let project = |s: &Array2<f64>| -> Result<Array2<f64>, String> {
            let root = psd_component_root(s.view())?;
            Ok(Self::reduced_from_root(&root.dot(&self.w_factor)))
        };
        let y_k: Result<Vec<Array2<f64>>, String> = if rayon::current_thread_index().is_some() {
            s_k_matrices.iter().map(project).collect()
        } else {
            s_k_matrices.par_iter().map(project).collect()
        };
        // A unit penalty component whose own eigendecomposition fails is a
        // malformed input, not a numerical regime: fall back to the direct
        // contraction so this routine keeps its infallible signature and the
        // failure surfaces where the component is built.
        let y_k: Vec<Array2<f64>> = match y_k {
            Ok(y) => y,
            Err(reason) => {
                log::warn!(
                    "penalty ρ-derivative root factorization failed ({reason}); falling back to \
                     the direct WᵀS_kW contraction, whose error grows like ε·κ(S_λ)"
                );
                s_k_matrices.iter().map(|s| self.reduced(s)).collect()
            }
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
        penalties: &[gam_terms::construction::CanonicalPenalty],
        lambdas: &[f64],
    ) -> (Array1<f64>, Array2<f64>) {
        let k = penalties.len();
        if k == 0 || self.rank == 0 {
            return (Array1::zeros(k), Array2::zeros((k, k)));
        }

        struct ReducedPenalty {
            pub(crate) span: Option<usize>,
            pub(crate) y: Array2<f64>,
        }

        // `CanonicalPenalty::root` IS the `rank_k × block_dim` factor with
        // `S_k = rootᵀroot`, so the numerically stable Gram form of the reduced
        // projection (`reduced_from_root`, #2644) costs nothing extra here — it
        // replaces one `block × block` multiply with one `rank_k × block` one.
        // A penalty whose cached root does not match its cached `local` shape is
        // a malformed component; fall back to the direct contraction rather than
        // panicking inside an infallible derivative routine.
        let project = |penalty: &gam_terms::construction::CanonicalPenalty| {
            let start = penalty.col_range.start;
            let end = penalty.col_range.end;
            let root_matches = penalty.root.ncols() == end - start;
            if let Some((span_idx, span)) = self
                .block_spans
                .iter()
                .enumerate()
                .find(|(_, span)| span.start <= start && end <= span.end)
            {
                let local_start = start - span.start;
                let local_end = local_start + (end - start);
                assert_eq!(local_end - local_start, penalty.local.nrows());
                let w_block = self
                    .w_factor
                    .slice(s![start..end, span.rank_start..span.rank_end]);
                let y = if root_matches {
                    Self::reduced_from_root(&penalty.root.dot(&w_block))
                } else {
                    w_block.t().dot(&penalty.local.dot(&w_block))
                };
                ReducedPenalty {
                    span: Some(span_idx),
                    y,
                }
            } else {
                // Overlapping/global fallback: still avoid cloning the block view.
                let w_block = self.w_factor.slice(s![start..end, ..]);
                let y = if root_matches {
                    Self::reduced_from_root(&penalty.root.dot(&w_block))
                } else {
                    w_block.t().dot(&penalty.local.dot(&w_block))
                };
                ReducedPenalty {
                    span: None,
                    y,
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
    /// gam#2454: the criterion's structural rank is the reparameterization's
    /// rank — the Frobenius-balanced rule — not the unweighted sum's. Three
    /// disjoint unit directions penalized at Frobenius norms `1e6`, `1e-9` and
    /// `1`: the unweighted sum `diag(1e6, 1e-9, 1)` at the `100·p·ε·max` cut
    /// loses the `1e-9` direction (rank 2), while every direction is
    /// structurally penalized (rank 3), and the rank must not move with λ.
    #[test]
    fn structural_rank_is_the_balanced_rank_not_the_unweighted_sums_2454() {
        use ndarray::array;
        let s1 = array![[1.0e6, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let s2 = array![[0.0, 0.0, 0.0], [0.0, 1.0e-9, 0.0], [0.0, 0.0, 0.0]];
        let s3 = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let components = [s1, s2, s3];
        let balanced = gam_terms::construction::balanced_penalty_structural_rank(
            components.iter().map(|s| (s.view(), 0..3)),
            3,
        )
        .expect("balanced rank");
        assert_eq!(balanced, 3, "every direction carries a penalty");
        for lambdas in [[1.0, 1.0, 1.0], [1.0e8, 1.0e-6, 1.0], [1.0e-6, 1.0e8, 1.0e-3]] {
            let pld = PenaltyPseudologdet::from_components(&components, &lambdas, 0.0)
                .expect("pseudo-logdet");
            assert_eq!(
                pld.rank(),
                balanced,
                "the criterion ranges over the balanced structural rank at lambdas {lambdas:?}"
            );
            let expected =
                (lambdas[0] * 1.0e6).ln() + (lambdas[1] * 1.0e-9).ln() + lambdas[2].ln();
            assert!(
                (pld.value() - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                "log|S|_+ over all three directions: {} vs {expected}",
                pld.value()
            );
        }
    }

    use super::*;
    use ndarray::array;

    #[test]
    fn dense_product_trace_accepts_nonstandard_owned_layouts() {
        let a = array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]].reversed_axes();
        let b = array![[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]].reversed_axes();

        assert_eq!(a.dim(), (2, 3));
        assert_eq!(b.dim(), (3, 2));
        assert_eq!(PenaltyPseudologdet::trace_dense_product(&a, &b), 212.0);
    }

    /// Scalar S(ρ) = e^ρ. Then log|S|₊ = ρ, L' = 1, L'' = 0.
    #[test]
    pub(crate) fn test_scalar_penalty_logdet() {
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
    pub(crate) fn test_two_penalty_logdet() {
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
    pub(crate) fn test_tau_derivative_fd() {
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
    pub(crate) fn test_no_nullspace_correction_full_rank() {
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
    pub(crate) fn test_assembled_pure_spectrum_classifier_issue_192_and_318() {
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
    pub(crate) fn test_rank_deficient_value() {
        // S = [[4, 2], [2, 1]] has rank 1, eigenvalue 5.
        let s = array![[4.0, 2.0], [2.0, 1.0]];
        let pld = PenaltyPseudologdet::from_assembled(s, None).unwrap();
        assert_eq!(pld.rank(), 1);
        assert!((pld.value() - 5.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    pub(crate) fn test_component_ridge_excludes_inactive_penalty_nullspace() {
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
    pub(crate) fn test_value_matches_rho_gradient_across_ridge_boundary() {
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
    pub(crate) fn test_components_with_stale_nullity_uses_active_sum_when_lambda_zero() {
        let s1 = array![[4.0, 0.0], [0.0, 0.0]];
        let s2 = array![[0.0, 0.0], [0.0, 9.0]];
        let lambdas = [2.0_f64, 0.0_f64];
        let ridge = 1e-4_f64;

        let pld = PenaltyPseudologdet::from_components(&[s1, s2], &lambdas, ridge).unwrap();

        assert_eq!(pld.rank(), 1);
        assert!((pld.value() - (8.0 + ridge).ln()).abs() < 1e-12);
    }

    #[test]
    pub(crate) fn test_rank_deficient_components_can_sum_to_full_rank_or_not() {
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
    pub(crate) fn test_block_penalties_ridge_excludes_inactive_penalty_nullspace() {
        let penalties = [
            gam_terms::construction::CanonicalPenalty::from_dense_root(array![[2.0, 0.0]], 2),
            gam_terms::construction::CanonicalPenalty::from_dense_root(array![[0.0, 3.0]], 2),
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
    pub(crate) fn test_nullspace_rotation_gradient_zero() {
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
    pub(crate) fn test_block_factored_tau_hessian_preserves_internal_nullspace() {
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

        let root = crate::estimate::reml::reml_outer_engine::penalty_matrix_root(&s_mat).unwrap();
        let penalty = gam_terms::construction::CanonicalPenalty::from_dense_root(root, 3);
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
    pub(crate) fn test_block_factored_ridge_preserves_structural_nullspace_value() {
        let s = array![[4.0, 2.0], [2.0, 1.0]];
        let ridge = 1e-4_f64;

        let root = crate::estimate::reml::reml_outer_engine::penalty_matrix_root(&s).unwrap();
        let penalty = gam_terms::construction::CanonicalPenalty::from_dense_root(root, 2);
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
    pub(crate) fn test_block_factored_ridge_ignores_inactive_lambda_for_structural_nullity() {
        let ridge = 1e-4_f64;
        let penalties = [
            gam_terms::construction::CanonicalPenalty::from_dense_root(array![[1.0, 0.0]], 2),
            gam_terms::construction::CanonicalPenalty::from_dense_root(array![[0.0, 1.0]], 2),
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
    pub(crate) fn test_overlapping_ridge_ignores_inactive_lambda_for_structural_nullity() {
        let ridge = 1e-4_f64;
        let penalties = [
            gam_terms::construction::CanonicalPenalty {
                root: array![[1.0, 0.0]],
                col_range: 0..2,
                total_dim: 3,
                nullity: 1,
                local: array![[1.0, 0.0], [0.0, 0.0]],
                prior_mean: Array1::zeros(2),
                positive_eigenvalues: vec![1.0],
                op: None,
            },
            gam_terms::construction::CanonicalPenalty {
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
    pub(crate) fn test_block_factored_rho_derivatives_match_dense_without_cross_block_work() {
        let p_total = 6;
        let lambdas = [1.7_f64, 0.4_f64, 2.3_f64];
        let penalties = vec![
            gam_terms::construction::CanonicalPenalty {
                root: array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                col_range: 0..3,
                total_dim: p_total,
                nullity: 1,
                local: array![[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]],
                prior_mean: Array1::zeros(3),
                positive_eigenvalues: vec![1.0, 4.0],
                op: None,
            },
            gam_terms::construction::CanonicalPenalty {
                root: array![[0.0, 0.0, 3.0]],
                col_range: 0..3,
                total_dim: p_total,
                nullity: 2,
                local: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 9.0]],
                prior_mean: Array1::zeros(3),
                positive_eigenvalues: vec![9.0],
                op: None,
            },
            gam_terms::construction::CanonicalPenalty {
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
    pub(crate) fn test_same_block_double_penalty_keeps_structural_rank_under_lambda_imbalance() {
        let p_total = 3;
        let lambdas = [1.0e12_f64, 1.0e-6_f64];
        let penalties = vec![
            gam_terms::construction::CanonicalPenalty {
                root: array![[10.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                col_range: 0..3,
                total_dim: p_total,
                nullity: 1,
                local: array![[100.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                prior_mean: Array1::zeros(3),
                positive_eigenvalues: vec![100.0, 1.0],
                op: None,
            },
            gam_terms::construction::CanonicalPenalty {
                root: array![[0.0, 0.0, 1.0]],
                col_range: 0..3,
                total_dim: p_total,
                nullity: 2,
                local: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                prior_mean: Array1::zeros(3),
                positive_eigenvalues: vec![1.0],
                op: None,
            },
        ];

        let pld = PenaltyPseudologdet::from_penalties(&penalties, &lambdas, 0.0, p_total)
            .expect("same-block double penalty pseudo-logdet");
        assert_eq!(
            pld.rank(),
            3,
            "same-block double penalty must keep the unweighted structural rank even when \
             the current null-ridge eigenvalue is tiny relative to the bend block"
        );

        let (det1, _) = pld.rho_derivatives_from_penalties(&penalties, &lambdas);
        assert!((det1[0] - 2.0).abs() < 1e-9);
        assert!((det1[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    pub(crate) fn test_dense_components_keep_structural_rank_under_lambda_imbalance() {
        let lambdas = [1.0e12_f64, 1.0e-6_f64];
        let penalties = vec![
            array![[100.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ];

        let pld = PenaltyPseudologdet::from_components(&penalties, &lambdas, 0.0)
            .expect("dense same-block double penalty pseudo-logdet");
        assert_eq!(
            pld.rank(),
            3,
            "dense component double penalty must keep positive-lambda structural rank"
        );

        let (det1, _) = pld.rho_derivatives(&penalties, &lambdas);
        assert!((det1[0] - 2.0).abs() < 1e-9);
        assert!((det1[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    pub(crate) fn test_overlapping_penalties_ridge_preserve_structural_nullspace_value() {
        let ridge = 1e-4_f64;
        let lambdas = [2.0_f64, 3.0_f64];
        let penalties = [
            gam_terms::construction::CanonicalPenalty::from_dense_root(array![[1.0, 0.0, 0.0]], 3),
            gam_terms::construction::CanonicalPenalty::from_dense_root(array![[0.0, 1.0, 0.0]], 3),
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

    /// Regression for the mid-optimization abort
    /// "structural rank hint selected non-positive eigenvalue" on plain
    /// `s(x)` fits: two penalties overlapping on one block (bend + full-block
    /// shrinkage, the double-penalty layout) with an extreme λ ratio. The
    /// assembled matrix's eigh sees the small component's genuine modes at
    /// SQUARED relative scale (λ ratio 1e26 → eigenvalue ratio below ε) and
    /// returns roundoff of either sign, so the hinted split either aborted
    /// the outer evaluation or silently mislabeled coercive modes. The
    /// stacked scaled-root spectrum keeps every mode at root scale, so the
    /// construction must succeed with the full structural rank and the EXACT
    /// closed-form value on this diagonal example.
    #[test]
    fn overlapping_extreme_lambda_ratio_keeps_structural_modes_exactly() {
        let p = 6usize;
        let mut s_bend = Array2::<f64>::zeros((p, p));
        for i in 0..4 {
            s_bend[[i, i]] = 1.0;
        }
        let mut s_shrink = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            s_shrink[[i, i]] = 1.0;
        }

        for &(l_bend, l_shrink) in &[(1e13_f64, 1e-13_f64), (1e-14, 1e12), (1e10, 1e-18)] {
            let pld = PenaltyPseudologdet::from_components(
                &[s_bend.clone(), s_shrink.clone()],
                &[l_bend, l_shrink],
                0.0,
            )
            .unwrap_or_else(|e| {
                panic!(
                    "overlapping penalties with λ ratio {:.1e} must stay evaluable \
                     (outer probes reach these ratios): {e}",
                    l_bend / l_shrink
                )
            });
            assert_eq!(
                pld.rank(),
                p,
                "every structural mode keeps its coercivity at λ ratio {:.1e}",
                l_bend / l_shrink
            );
            let expected: f64 = 4.0 * (l_bend + l_shrink).ln() + 2.0 * l_shrink.ln();
            assert!(
                (pld.value() - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                "exact diagonal pseudologdet at λ ratio {:.1e}: got {}, expected {expected}",
                l_bend / l_shrink,
                pld.value(),
            );
        }
    }
    /// A fixed orthogonal `p × p` matrix with no axis-aligned column, built
    /// deterministically from a product of Givens rotations so the test carries
    /// no RNG. Every penalty built as `Q diag(d) Qᵀ` below is therefore DENSE:
    /// that is the whole point (see
    /// `rotated_extreme_lambda_ratio_prices_logdet_at_root_scale`).
    fn dense_orthogonal(p: usize) -> Array2<f64> {
        let mut q = Array2::<f64>::eye(p);
        // Angles are irrational multiples of π so no entry lands on 0 or ±1.
        for i in 0..p {
            for j in (i + 1)..p {
                let theta = 0.3 + 0.17 * (i as f64) + 0.11 * (j as f64);
                let (sin, cos) = theta.sin_cos();
                for row in 0..p {
                    let a = q[[row, i]];
                    let b = q[[row, j]];
                    q[[row, i]] = cos * a - sin * b;
                    q[[row, j]] = sin * a + cos * b;
                }
            }
        }
        q
    }

    /// `Q diag(d) Qᵀ`, symmetrized so the input is exactly symmetric.
    fn rotated_penalty(q: &Array2<f64>, d: &[f64]) -> Array2<f64> {
        let p = q.nrows();
        let mut m = Array2::<f64>::zeros((p, p));
        for (i, &di) in d.iter().enumerate() {
            if di == 0.0 {
                continue;
            }
            for r in 0..p {
                for c in 0..p {
                    m[[r, c]] += di * q[[r, i]] * q[[c, i]];
                }
            }
        }
        let mt = m.t().to_owned();
        m += &mt;
        m *= 0.5;
        m
    }

    /// #2644 ROOT CAUSE. `log|S_λ|₊` must be priced at ROOT scale, not from any
    /// factorization of the assembled `Σ λ_k S_k`.
    ///
    /// `S_λ` is a sum of squares, so assembling it SQUARES the conditioning of
    /// the objects it is built from. Every backward-stable factorization of the
    /// assembled matrix — `eigh` and Cholesky alike — therefore carries
    /// `O(ε·κ(S_λ))` absolute error on `log|S_λ|₊`; the stacked scaled roots
    /// carry `O(ε·√κ(S_λ))`. At the `κ ≈ 1.4e14` this fixture reproduces (one
    /// smooth's wiggliness λ at its ceiling beside its own null-space shrinkage
    /// λ near zero — the shape the outer search reaches on
    /// `y ~ s(pc1,k=5) + s(pc2,k=5)`), that is `±1.2e-2` of noise on the outer
    /// REML criterion against a relative cost floor of `~5.6e-8·(1+|V|)`, so the
    /// line search cannot resolve a real decrease and the fit is refused for
    /// non-stationarity at `|Pg| ≈ 1.8e-3`.
    ///
    /// The construction is DENSE — `Q diag Qᵀ` for a fixed rotation `Q` with no
    /// axis-aligned column. `overlapping_extreme_lambda_ratio_keeps_structural_modes_exactly`
    /// above pins the same λ ratios on a DIAGONAL example, where the assembled
    /// matrix is exactly diagonal and its Cholesky is exact to one ulp — which
    /// is exactly why that test cannot see this defect and this one can. The two
    /// penalties commute here (same eigenbasis `Q`), so the expected value is
    /// closed-form and the assertion is against arithmetic, not another route.
    #[test]
    fn rotated_extreme_lambda_ratio_prices_logdet_at_root_scale() {
        let p = 6usize;
        let q = dense_orthogonal(p);
        // A rank-3 wiggliness penalty and the rank-3 null-space shrinkage on
        // its complement, in one shared eigenbasis.
        let d_wiggle = [1.0, 0.7, 0.44, 0.0, 0.0, 0.0];
        let d_shrink = [0.0, 0.0, 0.0, 1.0, 0.9, 0.55];
        let s_wiggle = rotated_penalty(&q, &d_wiggle);
        let s_shrink = rotated_penalty(&q, &d_shrink);

        for &(l_wiggle, l_shrink) in &[
            (3.388e12_f64, 2.480e-2_f64),
            (1.0e13, 1.0e-3),
            (1.0e-4, 5.0e11),
        ] {
            let pld = PenaltyPseudologdet::from_components(
                &[s_wiggle.clone(), s_shrink.clone()],
                &[l_wiggle, l_shrink],
                0.0,
            )
            .expect("dense rotated double penalty stays evaluable");
            assert_eq!(pld.rank(), p, "every structural mode is retained");
            let expected: f64 = d_wiggle
                .iter()
                .filter(|&&d| d > 0.0)
                .map(|&d| (l_wiggle * d).ln())
                .chain(
                    d_shrink
                        .iter()
                        .filter(|&&d| d > 0.0)
                        .map(|&d| (l_shrink * d).ln()),
                )
                .sum();
            let error = (pld.value() - expected).abs();
            // Root scale gives ~ε·√κ ≈ 1e-9 here; the assembled routes give
            // ~ε·κ ≈ 1e-2, four thousand times the outer criterion's own cost
            // floor. 1e-7 sits far below the defect and far above the achievable
            // floor, so it is a verdict on the ROUTE, not a tuned constant.
            assert!(
                error <= 1.0e-7,
                "log|S_λ|₊ at λ ratio {:.1e} must be priced at root scale: \
                 got {}, expected {expected}, error {error:.3e}",
                l_wiggle / l_shrink,
                pld.value(),
            );
        }
    }

    /// #2644, second angle: the ρ-gradient the outer optimizer consumes must be
    /// recoverable from CENTRAL DIFFERENCES of `value()` at the λ spreads the
    /// search actually visits.
    ///
    /// This is the property the criterion's user needs and the one the defect
    /// destroyed: with `±1.2e-2` of evaluation noise, a central difference at
    /// any usable step is pure noise, so "the analytic gradient disagrees with
    /// the function it claims to differentiate" — the shape #2644 reports as a
    /// stationarity refusal at an interior PSD minimum. It fails for a reason
    /// independent of the closed-form check above: that one compares a value to
    /// arithmetic, this one compares two DERIVATIVES of the same object.
    #[test]
    fn rotated_extreme_lambda_ratio_rho_gradient_matches_central_difference() {
        let p = 6usize;
        let q = dense_orthogonal(p);
        let d_wiggle = [1.0, 0.7, 0.44, 0.0, 0.0, 0.0];
        let d_shrink = [0.0, 0.0, 0.0, 1.0, 0.9, 0.55];
        let s_wiggle = rotated_penalty(&q, &d_wiggle);
        let s_shrink = rotated_penalty(&q, &d_shrink);
        let components = [s_wiggle, s_shrink];

        let rho = [28.851_f64, -3.697];
        let lambdas: Vec<f64> = rho.iter().map(|r| r.exp()).collect();
        let pld = PenaltyPseudologdet::from_components(&components, &lambdas, 0.0)
            .expect("dense rotated double penalty stays evaluable");
        let (det1, _) = pld.rho_derivatives(&components, &lambdas);

        let value_at = |rho: &[f64]| -> f64 {
            let lambdas: Vec<f64> = rho.iter().map(|r| r.exp()).collect();
            PenaltyPseudologdet::from_components(&components, &lambdas, 0.0)
                .expect("dense rotated double penalty stays evaluable")
                .value()
        };
        // `log|S_λ|₊` is a sum of `log λ_k` terms plus λ-free constants, so its
        // ρ-curvature is O(1) and a 1e-4 central difference truncates at ~1e-9.
        let h = 1.0e-4_f64;
        for k in 0..2 {
            let mut up = rho;
            let mut down = rho;
            up[k] += h;
            down[k] -= h;
            let fd = (value_at(&up) - value_at(&down)) / (2.0 * h);
            assert!(
                (det1[k] - fd).abs() <= 1.0e-6 * det1[k].abs().max(1.0),
                "∂log|S_λ|₊/∂ρ_{k}: analytic {} vs central difference {fd} \
                 (λ = {:.3e}, {:.3e})",
                det1[k],
                lambdas[0],
                lambdas[1],
            );
        }
    }
}

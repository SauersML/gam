//! Neutral penalty-coordinate contract (moved from solver::reml::reml_outer_engine
//! under #1521). The enum is pure data; its operators use only gam-problem's own
//! dense linalg helpers, so hosting it here lets the criterion/solver layers share
//! one definition without an upward edge into the engine.
use gam_linalg::dense;
use crate::reml_contract_panic;
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};

/// A rho-coordinate always contributes
///
///   A_k = λ_k S_k,
///   S_k = R_k^T R_k.
///
/// For single-block/small problems it is fine to store the full-root `R_k`
/// in the joint basis. For exact-joint multi-block paths that scaling is
/// wasteful: the root is naturally block-local. This enum lets the unified
/// evaluator consume both forms through one interface.
#[derive(Clone, Debug)]
pub enum PenaltyCoordinate {
    DenseRoot(Array2<f64>),
    DenseRootCentered {
        root: Array2<f64>,
        prior_mean: Array1<f64>,
    },
    BlockRoot {
        root: Array2<f64>,
        start: usize,
        end: usize,
        total_dim: usize,
    },
    BlockRootCentered {
        root: Array2<f64>,
        start: usize,
        end: usize,
        total_dim: usize,
        prior_mean: Array1<f64>,
    },
    /// Kronecker-factored penalty coordinate for tensor-product smooths.
    ///
    /// In the reparameterized (eigenbasis) representation, the penalty
    /// `I ⊗ ... ⊗ S_k ⊗ ... ⊗ I` becomes `I ⊗ ... ⊗ Λ_k ⊗ ... ⊗ I`
    /// where `Λ_k = diag(μ_{k,0}, ..., μ_{k,q_k-1})`.  This is diagonal
    /// in each mode, so apply/quadratic/trace operations avoid O(p²).
    KroneckerMarginal {
        /// Marginal eigenvalues for ALL dimensions: `eigenvalues[j]` has length `q_j`.
        eigenvalues: Vec<Array1<f64>>,
        /// Which marginal dimension this penalty coordinate corresponds to.
        dim_index: usize,
        /// Marginal basis dimensions: `[q_0, ..., q_{d-1}]`.
        marginal_dims: Vec<usize>,
        /// Total joint dimension: `∏ q_j`.
        total_dim: usize,
    },
}

impl PenaltyCoordinate {
    pub fn from_dense_root(root: Array2<f64>) -> Self {
        Self::DenseRoot(root)
    }

    pub fn from_dense_root_with_mean(root: Array2<f64>, prior_mean: Array1<f64>) -> Self {
        assert_eq!(root.ncols(), prior_mean.len());
        if prior_mean.iter().all(|&value| value == 0.0) {
            Self::DenseRoot(root)
        } else {
            Self::DenseRootCentered { root, prior_mean }
        }
    }

    pub fn from_block_root(root: Array2<f64>, start: usize, end: usize, total_dim: usize) -> Self {
        assert_eq!(
            root.ncols(),
            end.saturating_sub(start),
            "block prior root column count must match block width"
        );
        assert!(
            end <= total_dim,
            "block prior root end exceeds total dimension: start={start}, end={end}, total_dim={total_dim}, root_dim={:?}",
            root.dim()
        );
        Self::BlockRoot {
            root,
            start,
            end,
            total_dim,
        }
    }

    pub fn from_block_root_with_mean(
        root: Array2<f64>,
        start: usize,
        end: usize,
        total_dim: usize,
        prior_mean: Array1<f64>,
    ) -> Self {
        assert_eq!(
            root.ncols(),
            end.saturating_sub(start),
            "centered block prior root column count must match block width"
        );
        assert_eq!(
            prior_mean.len(),
            end.saturating_sub(start),
            "centered block prior mean length must match block width"
        );
        assert!(
            end <= total_dim,
            "centered block prior root end exceeds total dimension: start={start}, end={end}, total_dim={total_dim}, root_dim={:?}, prior_mean_len={}",
            root.dim(),
            prior_mean.len()
        );
        if prior_mean.iter().all(|&value| value == 0.0) {
            Self::from_block_root(root, start, end, total_dim)
        } else {
            Self::BlockRootCentered {
                root,
                start,
                end,
                total_dim,
                prior_mean,
            }
        }
    }

    pub fn rank(&self) -> usize {
        match self {
            Self::DenseRoot(root)
            | Self::DenseRootCentered { root, .. }
            | Self::BlockRoot { root, .. }
            | Self::BlockRootCentered { root, .. } => root.nrows(),
            Self::KroneckerMarginal {
                eigenvalues,
                dim_index,
                ..
            } => {
                // Rank = number of nonzero marginal eigenvalues for this dim,
                // times the product of all other dims.
                let nz = eigenvalues[*dim_index]
                    .iter()
                    .filter(|&&v| v.abs() > 1e-12)
                    .count();
                let other: usize = eigenvalues
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != *dim_index)
                    .map(|(_, e)| e.len())
                    .product::<usize>()
                    .max(1);
                nz * other
            }
        }
    }

    pub fn dim(&self) -> usize {
        match self {
            Self::DenseRoot(root) | Self::DenseRootCentered { root, .. } => root.ncols(),
            Self::BlockRoot { total_dim, .. }
            | Self::BlockRootCentered { total_dim, .. }
            | Self::KroneckerMarginal { total_dim, .. } => *total_dim,
        }
    }

    pub fn uses_operator_fast_path(&self) -> bool {
        matches!(
            self,
            Self::BlockRoot { .. }
                | Self::BlockRootCentered { .. }
                | Self::KroneckerMarginal { .. }
        )
    }

    /// Restrict this penalty coordinate onto the free subspace spanned by the
    /// orthonormal columns of `z` (shape `p × m`, `m ≤ p`, `zᵀz = I`).
    ///
    /// When a linear-inequality active set is non-empty, the inner solve and the
    /// penalized Hessian are reduced to the free subspace `β = z β_f` of
    /// dimension `m = p − active_set_size`. The penalty must move in lockstep:
    /// the quadratic `βᵀ S_k β = β_fᵀ (zᵀ S_k z) β_f`, and since `S_k = R_kᵀ R_k`
    /// the reduced root is `R_k z` (shape `rank_k × m`). For a block-local root
    /// `R_k` acting on `β[start..end]` the same identity gives reduced dense root
    /// `R_k · z[start..end, :]`, so the reduced coordinate is always a
    /// (dimension-`m`) `DenseRoot` / `DenseRootCentered` — the block structure
    /// does not survive an arbitrary subspace rotation. A centered mean `μ_k`
    /// maps to `zᵀ μ_k`, the representation of `μ_k` in the free subspace.
    ///
    /// This keeps `dim()` equal to the reduced `beta.len()`, which
    /// `InnerSolutionBuilder::build` asserts.
    pub fn project_into_subspace(&self, z: &Array2<f64>) -> Self {
        assert_eq!(
            z.nrows(),
            self.dim(),
            "PenaltyCoordinate::project_into_subspace: free-basis row count {} does not match coordinate dimension {}",
            z.nrows(),
            self.dim()
        );
        match self {
            Self::DenseRoot(root) => Self::DenseRoot(root.dot(z)),
            Self::DenseRootCentered { root, prior_mean } => {
                Self::from_dense_root_with_mean(root.dot(z), z.t().dot(prior_mean))
            }
            Self::BlockRoot {
                root, start, end, ..
            } => {
                let z_block = z.slice(ndarray::s![*start..*end, ..]);
                Self::DenseRoot(root.dot(&z_block))
            }
            Self::BlockRootCentered {
                root,
                start,
                end,
                prior_mean,
                ..
            } => {
                let z_block = z.slice(ndarray::s![*start..*end, ..]);
                // Reduced mean: the block-local prior `μ_k` sits at
                // `β[start..end]`; lift it into the full coordinate before
                // projecting so the free-space mean is `zᵀ (E_block μ_k)`.
                let z_block_owned = z_block.to_owned();
                Self::from_dense_root_with_mean(
                    root.dot(&z_block_owned),
                    z_block_owned.t().dot(prior_mean),
                )
            }
            Self::KroneckerMarginal { .. } => reml_contract_panic(
                "PenaltyCoordinate::project_into_subspace: Kronecker-factored \
                 coordinates do not co-occur with linear-inequality active sets \
                 (box/monotone constraints lower to dense/block roots)",
            ),
        }
    }

    pub(crate) fn apply_root(&self, beta: &Array1<f64>) -> Array1<f64> {
        assert_eq!(beta.len(), self.dim());
        match self {
            Self::DenseRoot(root) | Self::DenseRootCentered { root, .. } => root.dot(beta),
            Self::BlockRoot {
                root, start, end, ..
            }
            | Self::BlockRootCentered {
                root, start, end, ..
            } => root.dot(&beta.slice(ndarray::s![*start..*end])),
            Self::KroneckerMarginal { .. } => {
                // No single root for Kronecker — use apply_penalty instead.
                // SAFETY: `has_root()` returns `false` for the
                // KroneckerMarginal variant (see the `matches!` block
                // above); callers of `apply_root` are required to gate on
                // `has_root()`, so reaching this arm means a caller
                // invoked the rooted-only API on a rootless variant.
                // SAFETY: KroneckerMarginal has no root; callers must gate on has_root() before apply_root.
                reml_contract_panic(
                    "apply_root not supported for KroneckerMarginal; use apply_penalty directly",
                );
            }
        }
    }

    pub fn apply_penalty(&self, beta: &Array1<f64>, scale: f64) -> Array1<f64> {
        assert_eq!(beta.len(), self.dim());
        let mut out = Array1::<f64>::zeros(self.dim());
        self.apply_penalty_view_into(beta.view(), scale, out.view_mut());
        out
    }

    pub fn apply_penalty_view_into(
        &self,
        beta: ArrayView1<'_, f64>,
        scale: f64,
        mut out: ArrayViewMut1<'_, f64>,
    ) {
        assert_eq!(beta.len(), self.dim());
        assert_eq!(out.len(), self.dim());
        out.fill(0.0);
        self.scaled_add_penalty_view(beta, scale, out);
    }

    pub fn scaled_add_penalty_view(
        &self,
        beta: ArrayView1<'_, f64>,
        scale: f64,
        mut out: ArrayViewMut1<'_, f64>,
    ) {
        assert_eq!(beta.len(), self.dim());
        assert_eq!(out.len(), self.dim());
        if scale == 0.0 {
            return;
        }
        match self {
            Self::DenseRoot(_)
            | Self::DenseRootCentered { .. }
            | Self::BlockRoot { .. }
            | Self::BlockRootCentered { .. } => match self {
                Self::DenseRoot(root) | Self::DenseRootCentered { root, .. } => {
                    let mut root_beta = Array1::<f64>::zeros(root.nrows());
                    dense::matvec_into(root, beta, root_beta.view_mut());
                    dense::transpose_matvec_scaled_add_into(
                        root,
                        root_beta.view(),
                        scale,
                        out.view_mut(),
                    );
                }
                Self::BlockRoot {
                    root,
                    start,
                    end,
                    total_dim: _,
                }
                | Self::BlockRootCentered {
                    root,
                    start,
                    end,
                    total_dim: _,
                    ..
                } => {
                    let beta_block = beta.slice(ndarray::s![*start..*end]);
                    let mut root_beta = Array1::<f64>::zeros(root.nrows());
                    dense::matvec_into(root, beta_block, root_beta.view_mut());
                    let out_block = out.slice_mut(ndarray::s![*start..*end]);
                    dense::transpose_matvec_scaled_add_into(
                        root,
                        root_beta.view(),
                        scale,
                        out_block,
                    );
                }
                // Outer arm guarantees only the four root-bearing variants reach here.
                Self::KroneckerMarginal { .. } => {}
            },
            Self::KroneckerMarginal {
                eigenvalues,
                dim_index,
                marginal_dims,
                total_dim,
            } => {
                // Apply (I ⊗ ... ⊗ Λ_k ⊗ ... ⊗ I) β via mode-k scaling.
                // In the eigenbasis, Λ_k is diagonal, so this is element-wise.
                let k = *dim_index;
                let q_k = marginal_dims[k];
                let stride_k: usize = marginal_dims[k + 1..]
                    .iter()
                    .copied()
                    .product::<usize>()
                    .max(1);
                let outer_size: usize =
                    marginal_dims[..k].iter().copied().product::<usize>().max(1);
                let inner_size = stride_k;
                let eigs = &eigenvalues[k];
                assert_eq!(
                    outer_size * q_k * stride_k,
                    *total_dim,
                    "KroneckerMarginal dimension mismatch in apply"
                );

                for outer in 0..outer_size {
                    for j in 0..q_k {
                        let mu = eigs[j] * scale;
                        if mu == 0.0 {
                            continue;
                        }
                        let base = outer * q_k * stride_k + j * stride_k;
                        for inner in 0..inner_size {
                            let idx = base + inner;
                            out[idx] += mu * beta[idx];
                        }
                    }
                }
            }
        }
    }

    pub fn quadratic(&self, beta: &Array1<f64>, scale: f64) -> f64 {
        match self {
            Self::DenseRoot(_)
            | Self::DenseRootCentered { .. }
            | Self::BlockRoot { .. }
            | Self::BlockRootCentered { .. } => {
                let root_beta = self.apply_root(beta);
                scale * root_beta.dot(&root_beta)
            }
            Self::KroneckerMarginal {
                eigenvalues,
                dim_index,
                marginal_dims,
                ..
            } => {
                // β' (I ⊗ ... ⊗ Λ_k ⊗ ... ⊗ I) β = Σ μ_{k,j} β[...]²
                let k = *dim_index;
                let q_k = marginal_dims[k];
                let stride_k: usize = marginal_dims[k + 1..]
                    .iter()
                    .copied()
                    .product::<usize>()
                    .max(1);
                let outer_size: usize =
                    marginal_dims[..k].iter().copied().product::<usize>().max(1);
                let inner_size = stride_k;
                let eigs = &eigenvalues[k];

                let mut sum = 0.0;
                for outer in 0..outer_size {
                    for j in 0..q_k {
                        let mu = eigs[j];
                        if mu == 0.0 {
                            continue;
                        }
                        let base = outer * q_k * stride_k + j * stride_k;
                        for inner in 0..inner_size {
                            let v = beta[base + inner];
                            sum += mu * v * v;
                        }
                    }
                }
                sum * scale
            }
        }
    }

    pub fn apply_shifted_penalty(&self, beta: &Array1<f64>, scale: f64) -> Array1<f64> {
        match self {
            Self::DenseRootCentered { root, prior_mean } => {
                let centered = beta - prior_mean;
                let root_beta = root.dot(&centered);
                let mut out = root.t().dot(&root_beta);
                out *= scale;
                out
            }
            Self::BlockRootCentered {
                root,
                start,
                end,
                total_dim,
                prior_mean,
            } => {
                let mut out = Array1::<f64>::zeros(*total_dim);
                let beta_block = beta.slice(ndarray::s![*start..*end]);
                let centered = beta_block.to_owned() - prior_mean;
                let root_beta = root.dot(&centered);
                let mut block = root.t().dot(&root_beta);
                block *= scale;
                out.slice_mut(ndarray::s![*start..*end]).assign(&block);
                out
            }
            _ => self.apply_penalty(beta, scale),
        }
    }

    pub fn shifted_quadratic(&self, beta: &Array1<f64>, scale: f64) -> f64 {
        match self {
            Self::DenseRootCentered { root, prior_mean } => {
                let centered = beta - prior_mean;
                let root_beta = root.dot(&centered);
                scale * root_beta.dot(&root_beta)
            }
            Self::BlockRootCentered {
                root,
                start,
                end,
                prior_mean,
                ..
            } => {
                let beta_block = beta.slice(ndarray::s![*start..*end]);
                let centered = beta_block.to_owned() - prior_mean;
                let root_beta = root.dot(&centered);
                scale * root_beta.dot(&root_beta)
            }
            _ => self.quadratic(beta, scale),
        }
    }

    pub fn scaled_dense_matrix(&self, scale: f64) -> Array2<f64> {
        match self {
            Self::DenseRoot(root) | Self::DenseRootCentered { root, .. } => {
                let mut out = root.t().dot(root);
                out *= scale;
                out
            }
            Self::BlockRoot {
                root,
                start,
                end,
                total_dim,
            }
            | Self::BlockRootCentered {
                root,
                start,
                end,
                total_dim,
                ..
            } => {
                let mut out = Array2::<f64>::zeros((*total_dim, *total_dim));
                let mut block = root.t().dot(root);
                block *= scale;
                out.slice_mut(ndarray::s![*start..*end, *start..*end])
                    .assign(&block);
                out
            }
            Self::KroneckerMarginal {
                eigenvalues,
                dim_index,
                marginal_dims,
                total_dim,
            } => {
                // Materialize diagonal penalty in eigenbasis.
                let k = *dim_index;
                let q_k = marginal_dims[k];
                let stride_k: usize = marginal_dims[k + 1..]
                    .iter()
                    .copied()
                    .product::<usize>()
                    .max(1);
                let outer_size: usize =
                    marginal_dims[..k].iter().copied().product::<usize>().max(1);
                let eigs = &eigenvalues[k];
                assert_eq!(
                    outer_size * q_k * stride_k,
                    *total_dim,
                    "KroneckerMarginal dimension mismatch in to_dense"
                );

                let mut out = Array2::<f64>::zeros((*total_dim, *total_dim));
                for outer in 0..outer_size {
                    for j in 0..q_k {
                        let mu = eigs[j] * scale;
                        let base = outer * q_k * stride_k + j * stride_k;
                        for inner in 0..stride_k {
                            let idx = base + inner;
                            out[[idx, idx]] = mu;
                        }
                    }
                }
                out
            }
        }
    }

    /// Returns the block-local scaled penalty matrix (p_block × p_block) along
    /// with the embedding range, WITHOUT materializing into total_dim × total_dim.
    /// For DenseRoot (full-rank, no block structure), returns (matrix, 0, p).
    pub fn scaled_block_local(&self, scale: f64) -> (Array2<f64>, usize, usize) {
        match self {
            Self::DenseRoot(root) | Self::DenseRootCentered { root, .. } => {
                let mut out = root.t().dot(root);
                out *= scale;
                let p = out.nrows();
                (out, 0, p)
            }
            Self::BlockRoot {
                root, start, end, ..
            }
            | Self::BlockRootCentered {
                root, start, end, ..
            } => {
                let mut block = root.t().dot(root);
                block *= scale;
                (block, *start, *end)
            }
            Self::KroneckerMarginal { total_dim, .. } => {
                // Fallback: materialize full matrix.
                let mat = self.scaled_dense_matrix(scale);
                (mat, 0, *total_dim)
            }
        }
    }

    /// Whether this coordinate has block structure (not full-rank dense).
    pub fn is_block_local(&self) -> bool {
        matches!(
            self,
            Self::BlockRoot { .. }
                | Self::BlockRootCentered { .. }
                | Self::KroneckerMarginal { .. }
        )
    }

    /// Apply λ_k S_k to a vector v without materializing the full matrix.
    /// For BlockRoot: extracts v[start..end], multiplies by local S_k, embeds result.
    pub fn scaled_matvec(&self, v: &Array1<f64>, scale: f64) -> Array1<f64> {
        match self {
            Self::DenseRoot(root) | Self::DenseRootCentered { root, .. } => {
                let root_v = root.dot(v);
                let mut out = root.t().dot(&root_v);
                out *= scale;
                out
            }
            Self::BlockRoot {
                root, start, end, ..
            }
            | Self::BlockRootCentered {
                root, start, end, ..
            } => {
                let mut out = Array1::zeros(v.len());
                let v_block = v.slice(ndarray::s![*start..*end]);
                let root_v = root.dot(&v_block);
                let mut block_result = root.t().dot(&root_v);
                block_result *= scale;
                out.slice_mut(ndarray::s![*start..*end])
                    .assign(&block_result);
                out
            }
            Self::KroneckerMarginal { .. } => {
                // Reuse apply_penalty which handles mode-k contraction.
                self.apply_penalty(v, scale)
            }
        }
    }

    /// A stable, formula-order-independent signature of this penalty
    /// coordinate's STRUCTURAL CONTENT.
    ///
    /// Two penalty coordinates that represent the same smoothing structure —
    /// the same wiggliness root, the same null-space ridge, the same tensor
    /// margin — produce the same key regardless of which block of the joint
    /// coefficient vector they happen to occupy or which order the user typed
    /// the terms in. It is derived ENTIRELY from rotation/placement-invariant
    /// content (rank, block width, the spectrum of the block-local penalty
    /// `Sₖ = RₖᵀRₖ`, or the marginal eigenvalue spectrum for a Kronecker
    /// margin), and NEVER from a coordinate's position (`start`/`dim_index`)
    /// in the joint layout. Swapping `s(x)+s(z)` ↔ `s(z)+s(x)` or
    /// `te(x,z)` ↔ `te(z,x)` permutes the coordinates but leaves each
    /// coordinate's key fixed.
    ///
    /// This is the key the outer REML driver sorts on to present an identical
    /// canonical coordinate layout to the smoothing-parameter optimizer
    /// regardless of term/margin order, so the flat double-penalty REML valley
    /// is resolved order-invariantly (#1538/#1539). Values are quantized to a
    /// coarse relative grid so that floating-point round-off in the roots does
    /// not split an otherwise-identical key.
    pub fn canonical_structural_key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Quantize a magnitude to a coarse log-relative grid so tiny numeric
        // differences in equivalent roots collapse to the same bucket, while
        // genuinely different roughness scales stay distinct.
        let quant = |v: f64| -> i64 {
            if !v.is_finite() || v.abs() <= 1e-300 {
                return 0;
            }
            // ~1e-6 relative resolution: round log|v| to 6 decimals and keep sign.
            let q = (v.abs().ln() * 1.0e6).round() as i64;
            if v < 0.0 { -q } else { q }
        };

        match self {
            Self::DenseRoot(root)
            | Self::DenseRootCentered { root, .. }
            | Self::BlockRoot { root, .. }
            | Self::BlockRootCentered { root, .. } => {
                // Tag the rooted family uniformly: placement (start/end/total)
                // is deliberately excluded so a block that moves between term
                // orders keeps its key. The spectrum of Sₖ = RₖᵀRₖ is the
                // rotation-invariant fingerprint of the penalty.
                0u8.hash(&mut hasher);
                root.nrows().hash(&mut hasher); // rank
                root.ncols().hash(&mut hasher); // block width
                let sk = root.t().dot(root);
                // Orthogonal-invariants of the symmetric Sₖ = RₖᵀRₖ: the power
                // sums Σλ (trace), Σλ² (= ‖Sₖ‖²_F), Σλ³ (tr(Sₖ³)). Each is a
                // symmetric function of Sₖ's eigenvalues, so they are unchanged
                // by any orthonormal change of basis of the block coordinates
                // (hence by which joint block the penalty occupies) and by the
                // order of the terms. Together with rank and width they form a
                // strong placement-independent fingerprint without an
                // eigendecomposition.
                let n = sk.nrows().min(sk.ncols());
                let trace1 = (0..n).map(|i| sk[[i, i]]).sum::<f64>();
                let frob_sq = sk.iter().map(|&x| x * x).sum::<f64>(); // = Σλ²
                let sk2 = sk.dot(&sk);
                let trace3 = {
                    let sk3diag = sk2.dot(&sk);
                    (0..n).map(|i| sk3diag[[i, i]]).sum::<f64>()
                };
                let mut invariants = [quant(trace1), quant(frob_sq), quant(trace3)];
                // Power sums are already order-agnostic; sorting is a harmless
                // guard against any future addition of non-symmetric summaries.
                invariants.sort_unstable();
                invariants.hash(&mut hasher);
            }
            Self::KroneckerMarginal {
                eigenvalues,
                dim_index,
                marginal_dims,
                ..
            } => {
                // A tensor margin's identity is its OWN marginal penalty
                // spectrum plus the (sorted) set of marginal dimensions — both
                // independent of which slot `dim_index` the margin occupies, so
                // `te(x,z)` and `te(z,x)` give each margin the same key.
                1u8.hash(&mut hasher);
                let mut margin_spectrum: Vec<i64> =
                    eigenvalues[*dim_index].iter().map(|&e| quant(e)).collect();
                margin_spectrum.sort_unstable();
                margin_spectrum.hash(&mut hasher);
                let mut dims_sorted = marginal_dims.clone();
                dims_sorted.sort_unstable();
                dims_sorted.hash(&mut hasher);
            }
        }

        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    fn identity_root(n: usize) -> Array2<f64> {
        Array2::<f64>::eye(n)
    }

    // ── constructors ──────────────────────────────────────────────────────────

    #[test]
    fn from_dense_root_creates_dense_root_variant() {
        let root = identity_root(3);
        let pc = PenaltyCoordinate::from_dense_root(root);
        assert!(matches!(pc, PenaltyCoordinate::DenseRoot(_)));
    }

    #[test]
    fn from_dense_root_with_zero_mean_degrades_to_dense_root() {
        let root = identity_root(2);
        let mean = Array1::<f64>::zeros(2);
        let pc = PenaltyCoordinate::from_dense_root_with_mean(root, mean);
        assert!(matches!(pc, PenaltyCoordinate::DenseRoot(_)));
    }

    #[test]
    fn from_dense_root_with_nonzero_mean_creates_centered_variant() {
        let root = identity_root(2);
        let mean = array![1.0_f64, 0.0];
        let pc = PenaltyCoordinate::from_dense_root_with_mean(root, mean);
        assert!(matches!(pc, PenaltyCoordinate::DenseRootCentered { .. }));
    }

    #[test]
    fn from_block_root_creates_block_root_variant() {
        let root = Array2::<f64>::zeros((2, 2));
        let pc = PenaltyCoordinate::from_block_root(root, 0, 2, 5);
        assert!(matches!(pc, PenaltyCoordinate::BlockRoot { .. }));
    }

    // ── rank() and dim() ──────────────────────────────────────────────────────

    #[test]
    fn dense_root_rank_is_nrows_dim_is_ncols() {
        // root is 4 × 3
        let root = Array2::<f64>::zeros((4, 3));
        let pc = PenaltyCoordinate::from_dense_root(root);
        assert_eq!(pc.rank(), 4);
        assert_eq!(pc.dim(), 3);
    }

    #[test]
    fn block_root_dim_is_total_dim() {
        let root = Array2::<f64>::zeros((2, 2));
        let pc = PenaltyCoordinate::from_block_root(root, 1, 3, 7);
        assert_eq!(pc.dim(), 7);
    }

    // ── uses_operator_fast_path ───────────────────────────────────────────────

    #[test]
    fn dense_root_does_not_use_fast_path() {
        let pc = PenaltyCoordinate::from_dense_root(identity_root(2));
        assert!(!pc.uses_operator_fast_path());
    }

    #[test]
    fn block_root_uses_fast_path() {
        let root = Array2::<f64>::zeros((1, 2));
        let pc = PenaltyCoordinate::from_block_root(root, 0, 2, 4);
        assert!(pc.uses_operator_fast_path());
    }

    // ── apply_penalty ─────────────────────────────────────────────────────────

    #[test]
    fn dense_identity_root_penalty_is_beta() {
        // S = I^T I = I, so S β = β
        let pc = PenaltyCoordinate::from_dense_root(identity_root(3));
        let beta = array![1.0_f64, 2.0, 3.0];
        let out = pc.apply_penalty(&beta, 1.0);
        for i in 0..3 {
            assert!((out[i] - beta[i]).abs() < 1e-12, "index {i}: {}", out[i]);
        }
    }

    #[test]
    fn apply_penalty_zero_scale_returns_zeros() {
        let pc = PenaltyCoordinate::from_dense_root(identity_root(2));
        let beta = array![5.0_f64, 7.0];
        let out = pc.apply_penalty(&beta, 0.0);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn apply_penalty_scale_two_doubles_beta_for_identity_root() {
        let pc = PenaltyCoordinate::from_dense_root(identity_root(2));
        let beta = array![3.0_f64, 4.0];
        let out = pc.apply_penalty(&beta, 2.0);
        assert!((out[0] - 6.0).abs() < 1e-12);
        assert!((out[1] - 8.0).abs() < 1e-12);
    }
}

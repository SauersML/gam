//! Neutral penalty-coordinate contract (moved from solver::reml::reml_outer_engine
//! under #1521). The enum is pure data; its operators use only gam-problem's own
//! dense linalg helpers, so hosting it here lets the criterion/solver layers share
//! one definition without an upward edge into the engine.
use crate::reml_contract_panic;
use gam_linalg::dense;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};

/// `R ↦ R Π` for a block-local root, with `Π = I − N Nᵀ` the projector that
/// removes the declared-null directions `N` (orthonormal, `total_dim × m`).
///
/// This is the ONE implementation of the null-split projection on a penalty
/// root (#2454). Both penalty carriers that can hold a root — the solver's
/// [`PenaltyCoordinate`] and the term layer's `CanonicalPenalty` — project
/// through it, so the criterion's penalty, its ρ-derivatives, and its
/// `log|S|₊` cannot end up on three slightly different projections of the same
/// subspace.
///
/// Returns the FULL-WIDTH projected root (`rank × total_dim`) together with
/// whether its support stayed inside the original `[start, end)` block. Block
/// locality survives whenever the null basis is itself block-local — which is
/// the case for the non-overlapping reparameterization, whose balanced penalty
/// sum is block-diagonal and whose eigenvectors therefore are too. The caller
/// uses the flag to keep its block chart (and every block-local trace fast
/// path) instead of widening to a dense `p`-column root for nothing.
///
/// "Stayed inside the block" is decided against the projected root's OWN
/// magnitude: an out-of-block entry at `‖R Π‖_max · ε · total_dim` is the
/// projection's rounding, not support.
pub fn project_block_root_out_of_null_directions(
    root: ArrayView2<'_, f64>,
    start: usize,
    end: usize,
    total_dim: usize,
    null_basis: ArrayView2<'_, f64>,
) -> (Array2<f64>, bool) {
    let mut projected = Array2::<f64>::zeros((root.nrows(), total_dim));
    projected
        .slice_mut(ndarray::s![.., start..end])
        .assign(&root);
    if null_basis.ncols() > 0 {
        let coefficients = projected.dot(&null_basis);
        projected -= &coefficients.dot(&null_basis.t());
    }

    let root_scale = projected
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let column_support_tolerance = root_scale * f64::EPSILON * (total_dim as f64);
    let stays_block_local = (start > 0 || end < total_dim)
        && (0..total_dim)
            .filter(|column| *column < start || *column >= end)
            .all(|column| {
                projected
                    .column(column)
                    .iter()
                    .all(|value| value.abs() <= column_support_tolerance)
            });
    (projected, stays_block_local)
}

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

    /// Borrow the canonical penalty root in its native block chart.
    ///
    /// The root rows are the authoritative structural range coordinates: their
    /// count is `rank()` and must not be rediscovered by eigendecomposing the
    /// squared Gram `RᵀR`, which can promote roundoff in a structural zero.
    pub fn block_local_root(&self) -> Option<(&Array2<f64>, usize, usize)> {
        match self {
            Self::DenseRoot(root) | Self::DenseRootCentered { root, .. } => {
                Some((root, 0, root.ncols()))
            }
            Self::BlockRoot {
                root, start, end, ..
            }
            | Self::BlockRootCentered {
                root, start, end, ..
            } => Some((root, *start, *end)),
            Self::KroneckerMarginal { .. } => None,
        }
    }

    /// Remove this coordinate's support along the declared-null directions
    /// `N` (orthonormal, shape `p × m`), returning the coordinate for
    /// `Π S_k Π` with `Π = I − N Nᵀ`.
    ///
    /// # Why a penalty coordinate must know about the null split (#2454)
    ///
    /// The penalty reparameterization splits the coefficient space into a
    /// **λ-invariant penalized subspace** and its complement, and rebuilds the
    /// penalty the criterion actually applies as `S̃(λ) = E(λ)ᵀE(λ)` on the
    /// penalized subspace ALONE — so that `H`, `log|S|₊`, the inner solve and
    /// the criterion value all share one rank structure. A per-block `S_k`
    /// whose own root rank exceeds the split's penalized rank therefore
    /// describes a penalty that is NOT the one being optimized: it charges
    /// energy in directions `S̃` does not penalize. Because `β̂` is free in
    /// exactly those directions it accumulates `O(1)` coefficient energy there,
    /// and `∂/∂ρ_k` multiplies that phantom energy by `λ_k` — an additive
    /// `c·λ` contamination of the outer gradient that is invisible at `‖ρ‖ ≤ 1`
    /// and flips the gradient's sign a dozen e-folds up.
    ///
    /// Projecting restores the identity the outer derivatives are built on:
    /// `Σ_k λ_k (Π S_k Π) = Π (Σ_k λ_k S_k) Π = S̃(λ)` and, because `Π` is
    /// λ-invariant by construction, `∂S̃/∂ρ_k = λ_k · Π S_k Π` exactly. One
    /// penalty object then serves the value, the quadratic, the per-block
    /// scores, the `tr(H⁻¹ Ḣ_k)` drift and the outer Hessian.
    ///
    /// # Structure
    ///
    /// `Π` is applied on both sides, i.e. the root becomes `R_k Π`. Block
    /// locality is preserved whenever the null basis is block-local (the
    /// non-overlapping reparameterization path, where the balanced penalty sum
    /// is block-diagonal and its eigenvectors therefore are too); when a null
    /// direction straddles blocks the projected root genuinely is not
    /// block-local and a dense coordinate is returned. A centered coordinate
    /// keeps its prior mean: the quadratic stays `‖R_kΠ(β − μ_k)‖²`.
    ///
    /// Returns `self` unchanged when `N` has no columns.
    pub fn project_out_null_directions(&self, null_basis: ArrayView2<'_, f64>) -> Self {
        if null_basis.ncols() == 0 {
            return self.clone();
        }
        assert_eq!(
            null_basis.nrows(),
            self.dim(),
            "PenaltyCoordinate::project_out_null_directions: null-basis row count {} does not \
             match coordinate dimension {}",
            null_basis.nrows(),
            self.dim()
        );
        let total_dim = self.dim();
        let (root, start, end) = match self.block_local_root() {
            Some(parts) => parts,
            // A Kronecker-factored coordinate has no single root; the Kronecker
            // path builds its own marginal eigen-grid and never routes through
            // the dense reparameterization's null split, so there is nothing to
            // project against.
            None => return self.clone(),
        };
        // `R Π = R − (R N) Nᵀ`, through the shared primitive so this coordinate
        // and the term layer's `CanonicalPenalty` project identically.
        let (projected, stays_block_local) = project_block_root_out_of_null_directions(
            root.view(),
            start,
            end,
            total_dim,
            null_basis,
        );

        let prior_mean = self.prior_mean_block();
        if stays_block_local {
            let block = projected.slice(ndarray::s![.., start..end]).to_owned();
            match prior_mean {
                Some(mean) => {
                    Self::from_block_root_with_mean(block, start, end, total_dim, mean.to_owned())
                }
                None => Self::from_block_root(block, start, end, total_dim),
            }
        } else {
            match prior_mean {
                Some(mean) => {
                    let mut full_mean = Array1::<f64>::zeros(total_dim);
                    full_mean.slice_mut(ndarray::s![start..end]).assign(&mean);
                    Self::from_dense_root_with_mean(projected, full_mean)
                }
                None => Self::from_dense_root(projected),
            }
        }
    }

    /// The block-local prior mean, when this coordinate is centered.
    fn prior_mean_block(&self) -> Option<ArrayView1<'_, f64>> {
        match self {
            Self::DenseRootCentered { prior_mean, .. }
            | Self::BlockRootCentered { prior_mean, .. } => Some(prior_mean.view()),
            Self::DenseRoot(_) | Self::BlockRoot { .. } | Self::KroneckerMarginal { .. } => None,
        }
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

    /// The block-local scaled penalty ROOT `√scale · R_k` (rank × p_block) with
    /// its embedding range, so `scale · S_k = rootᵀroot` on that block.
    ///
    /// This is what [`Self::scaled_block_local`] squares before handing the
    /// result to a consumer, and the squaring is not free: `S_k` is a sum of
    /// squares, so contracting it against a metric scaled by `σ(H)^{-1}` (or
    /// `σ(S_λ)^{-1}`) carries `R_k`'s roundoff LINEARLY and divides it by the
    /// smallest eigenvalue, giving `O(ε·κ)` on traces the theory bounds by
    /// `rank(S_k)`. A consumer that keeps the root and forms a Gram instead
    /// squares that residual (#2644). Every root-bearing variant returns
    /// `Some`; `KroneckerMarginal` returns `None` because its penalty is stored
    /// as a marginal eigenvalue grid rather than a root.
    pub fn scaled_block_root(&self, scale: f64) -> Option<(Array2<f64>, usize, usize)> {
        if !(scale.is_finite() && scale >= 0.0) {
            return None;
        }
        let sqrt_scale = scale.sqrt();
        match self {
            Self::DenseRoot(root) | Self::DenseRootCentered { root, .. } => {
                Some((root * sqrt_scale, 0, root.ncols()))
            }
            Self::BlockRoot {
                root, start, end, ..
            }
            | Self::BlockRootCentered {
                root, start, end, ..
            } => Some((root * sqrt_scale, *start, *end)),
            Self::KroneckerMarginal { .. } => None,
        }
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
            // Only an exact zero has no logarithm to quantise; any nonzero
            // value, however small, is a distinct coordinate.
            if !v.is_finite() || v == 0.0 {
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
    // ─── #2454: null-split projection ───────────────────────────────────────

    /// `Π S Π` must be the penalty the criterion applies, and the quadratic at
    /// any β must equal the quadratic of the UNprojected penalty at `Πβ`.
    ///
    /// That identity is the whole reason the projection is legitimate: it lets
    /// the outer ρ-derivative multiply `q_k` by `λ_k` and still be the exact
    /// derivative of the criterion's own penalty energy.
    #[test]
    fn projection_equals_the_unprojected_quadratic_at_the_projected_beta() {
        // A rank-2 penalty on R^3 whose range deliberately overlaps the
        // direction that will be declared null.
        let root = array![[1.0_f64, 2.0, 0.5], [0.0, 1.0, -3.0]];
        let coord = PenaltyCoordinate::from_dense_root(root);
        // One orthonormal declared-null direction.
        let n = 1.0_f64 / 3.0_f64.sqrt();
        let null_basis = array![[n], [n], [n]];

        let projected = coord.project_out_null_directions(null_basis.view());
        let beta = array![0.7_f64, -1.3, 2.1];
        let coefficient = null_basis.column(0).dot(&beta);
        let beta_projected = &beta - &(&null_basis.column(0).to_owned() * coefficient);

        let via_projected_penalty = projected.quadratic(&beta, 1.0);
        let via_projected_beta = coord.quadratic(&beta_projected, 1.0);
        assert!(
            (via_projected_penalty - via_projected_beta).abs()
                <= 1e-12 * via_projected_beta.abs().max(1.0),
            "Pi S Pi at beta = {via_projected_penalty:.12e} must equal S at Pi beta = \
             {via_projected_beta:.12e}"
        );

        // And it must actually have removed something: the declared-null
        // direction now costs nothing.
        let null_direction = null_basis.column(0).to_owned();
        assert!(
            coord.quadratic(&null_direction, 1.0) > 1e-6,
            "fixture must have real support on the direction being projected out"
        );
        assert!(
            projected.quadratic(&null_direction, 1.0) <= 1e-24,
            "projected penalty must annihilate the declared-null direction, got {:.3e}",
            projected.quadratic(&null_direction, 1.0)
        );
    }

    /// The projected block sum must reproduce the projected TOTAL, which is the
    /// identity `Σ_k λ_k Π S_k Π = Π (Σ_k λ_k S_k) Π` the outer gradient relies
    /// on. Checked with unequal λ so a per-block scaling error cannot hide.
    #[test]
    fn projected_block_sum_reproduces_the_projected_total() {
        let coords = [
            PenaltyCoordinate::from_dense_root(array![[1.0_f64, 0.3, -0.2], [0.0, 1.0, 0.7]]),
            PenaltyCoordinate::from_dense_root(array![[0.4_f64, -1.1, 2.0]]),
        ];
        let lambdas = [7.5_f64, 0.125];
        let n = 1.0_f64 / 2.0_f64.sqrt();
        let null_basis = array![[n], [-n], [0.0]];
        let beta = array![1.9_f64, 0.4, -2.6];

        let coefficient = null_basis.column(0).dot(&beta);
        let beta_projected = &beta - &(&null_basis.column(0).to_owned() * coefficient);

        let block_sum: f64 = coords
            .iter()
            .zip(lambdas.iter())
            .map(|(coord, &lambda)| {
                coord
                    .project_out_null_directions(null_basis.view())
                    .quadratic(&beta, lambda)
            })
            .sum();
        let total_projected: f64 = coords
            .iter()
            .zip(lambdas.iter())
            .map(|(coord, &lambda)| coord.quadratic(&beta_projected, lambda))
            .sum();
        assert!(
            (block_sum - total_projected).abs() <= 1e-12 * total_projected.abs().max(1.0),
            "projected block sum {block_sum:.12e} must equal the projected total \
             {total_projected:.12e}"
        );
    }

    /// A block-local null basis must leave a block-local coordinate block-local.
    ///
    /// The non-overlapping reparameterization path builds `q_null` with strictly
    /// block-local columns, so this is the common case; forcing a dense p-wide
    /// root there would cost every block-local trace fast path for nothing.
    #[test]
    fn block_local_null_basis_preserves_the_block_chart() {
        let coord = PenaltyCoordinate::from_block_root(array![[1.0_f64, 1.0], [0.0, 2.0]], 1, 3, 5);
        let n = 1.0_f64 / 2.0_f64.sqrt();
        // Supported only on columns 1..3 — the coordinate's own block.
        let null_basis = array![[0.0_f64], [n], [-n], [0.0], [0.0]];

        let projected = coord.project_out_null_directions(null_basis.view());
        assert!(
            matches!(
                projected,
                PenaltyCoordinate::BlockRoot {
                    start: 1,
                    end: 3,
                    ..
                }
            ),
            "a block-local null basis must not densify the coordinate"
        );
        assert_eq!(projected.dim(), 5);

        // A null basis that straddles blocks genuinely cannot stay block-local.
        let straddling = array![[n], [n], [0.0], [0.0], [0.0]];
        let densified = coord.project_out_null_directions(straddling.view());
        assert!(
            matches!(densified, PenaltyCoordinate::DenseRoot(_)),
            "a straddling null basis must produce a dense coordinate rather than \
             silently dropping the out-of-block support"
        );
        assert_eq!(densified.dim(), 5);
        // Same identity as above must still hold on the densified route.
        let beta = array![0.5_f64, -1.5, 2.25, 3.0, -0.75];
        let coefficient = straddling.column(0).dot(&beta);
        let beta_projected = &beta - &(&straddling.column(0).to_owned() * coefficient);
        assert!(
            (densified.quadratic(&beta, 1.0) - coord.quadratic(&beta_projected, 1.0)).abs()
                <= 1e-12
        );
    }

    /// An empty null basis is the identity, bit-for-bit — every penalty whose
    /// numerical rank agrees with the split's must be untouched.
    #[test]
    fn empty_null_basis_is_the_identity() {
        let coord = PenaltyCoordinate::from_block_root(array![[1.0_f64, -0.25]], 0, 2, 4);
        let projected = coord.project_out_null_directions(Array2::zeros((4, 0)).view());
        let beta = array![1.0_f64, 2.0, 3.0, 4.0];
        assert_eq!(projected.quadratic(&beta, 3.0), coord.quadratic(&beta, 3.0));
        assert!(matches!(
            projected,
            PenaltyCoordinate::BlockRoot {
                start: 0,
                end: 2,
                ..
            }
        ));
    }

    /// A centered coordinate keeps its prior mean under projection: the
    /// quadratic stays `‖RΠ(β − μ)‖²`, which is what the shifted penalty
    /// channel and the IFT score both read.
    #[test]
    fn projection_carries_the_prior_mean() {
        let coord = PenaltyCoordinate::from_dense_root_with_mean(
            array![[1.0_f64, 0.5, -1.0], [0.0, 2.0, 0.25]],
            array![0.1_f64, -0.2, 0.3],
        );
        let n = 1.0_f64 / 3.0_f64.sqrt();
        let null_basis = array![[n], [n], [n]];
        let projected = coord.project_out_null_directions(null_basis.view());
        let beta = array![1.4_f64, -0.6, 0.9];

        // `Π S Π` applied to the CENTERED coefficient.
        let centered = array![beta[0] - 0.1, beta[1] + 0.2, beta[2] - 0.3];
        let coefficient = null_basis.column(0).dot(&centered);
        let centered_projected = &centered - &(&null_basis.column(0).to_owned() * coefficient);
        let expected = coord.quadratic(&centered_projected, 1.0);
        let got = projected.shifted_quadratic(&beta, 1.0);
        assert!(
            (got - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "shifted quadratic after projection = {got:.12e}, expected {expected:.12e}"
        );
    }
}

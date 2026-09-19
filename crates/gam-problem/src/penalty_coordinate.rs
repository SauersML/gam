//! Neutral penalty-coordinate contract (moved from solver::reml::reml_outer_engine
//! under #1521). The enum is pure data; its operators use only gam-problem's own
//! dense linalg helpers, so hosting it here lets the criterion/solver layers share
//! one definition without an upward edge into the engine.
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
/// The root is block-local, so `R Π = R − (R N_b) Nᵀ` with `N_b` the rows of
/// `N` inside `[start, end)`: only the `rank × m` coefficient matrix `R N_b`
/// is ever formed, and a zero-filled full-width copy of the root exists only
/// when the projection genuinely leaves the block. When `R N_b` is exactly
/// zero — every null direction supported outside the block, the ordinary case
/// for a random-effect factor beside unpenalized fixed columns — `R Π = R`
/// and nothing is allocated.
///
/// Block locality survives whenever the null basis is itself block-local —
/// which is the case for the non-overlapping reparameterization, whose
/// balanced penalty sum is block-diagonal and whose eigenvectors therefore
/// are too. The caller keeps its block chart (and every block-local trace
/// fast path) instead of widening to a dense `p`-column root for nothing.
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
) -> ProjectedBlockRoot {
    if null_basis.ncols() == 0 {
        return ProjectedBlockRoot::Unchanged;
    }
    let block_null = null_basis.slice(ndarray::s![start..end, ..]);
    let coefficients = root.dot(&block_null);
    if coefficients.iter().all(|value| *value == 0.0) {
        return ProjectedBlockRoot::Unchanged;
    }
    let block_correction = coefficients.dot(&block_null.t());
    let block = &root - &block_correction;
    let mut moved = max_abs(block_correction.iter());
    let mut root_scale = max_abs(block.iter());

    // `−(R N_b) Nᵀ` on the columns outside the block, in the order
    // `[0, start) ++ [end, total_dim)`.
    let outside_null = ndarray::concatenate(
        ndarray::Axis(0),
        &[
            null_basis.slice(ndarray::s![..start, ..]),
            null_basis.slice(ndarray::s![end.., ..]),
        ],
    )
    .expect("null-basis row slices share their column count");
    let outside = -coefficients.dot(&outside_null.t());
    let outside_max = max_abs(outside.iter());
    moved = moved.max(outside_max);
    root_scale = root_scale.max(outside_max);

    let column_support_tolerance = root_scale * f64::EPSILON * (total_dim as f64);
    let spans_every_column = start == 0 && end == total_dim;
    if !spans_every_column && outside_max <= column_support_tolerance {
        return ProjectedBlockRoot::BlockLocal { block, moved };
    }
    let mut projected = Array2::<f64>::zeros((root.nrows(), total_dim));
    projected.slice_mut(ndarray::s![.., start..end]).assign(&block);
    projected
        .slice_mut(ndarray::s![.., ..start])
        .assign(&outside.slice(ndarray::s![.., ..start]));
    projected
        .slice_mut(ndarray::s![.., end..])
        .assign(&outside.slice(ndarray::s![.., start..]));
    ProjectedBlockRoot::FullWidth {
        root: projected,
        moved,
    }
}

fn max_abs<'a>(values: impl Iterator<Item = &'a f64>) -> f64 {
    values.fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

/// `R Π` for a block-local root `R`, from
/// [`project_block_root_out_of_null_directions`].
#[derive(Clone, Debug)]
pub enum ProjectedBlockRoot {
    /// `R N = 0` exactly, so `R Π = R`.
    Unchanged,
    /// `R Π` is supported inside `[start, end)`; `block` is its
    /// `rank × (end − start)` restriction.
    BlockLocal { block: Array2<f64>, moved: f64 },
    /// `R Π` leaves the block (or the block spans every column); `root` is the
    /// full `rank × total_dim` projection.
    FullWidth { root: Array2<f64>, moved: f64 },
}

impl ProjectedBlockRoot {
    /// `‖R Π − R‖_max` over the full width.
    pub fn moved(&self) -> f64 {
        match self {
            Self::Unchanged => 0.0,
            Self::BlockLocal { moved, .. } | Self::FullWidth { moved, .. } => *moved,
        }
    }
}

/// The diagonal of `RᵀR` when that Gram is diagonal by structure: every row
/// of the penalty root `R` (rank × width) carries at most one nonzero.
///
/// Rows with disjoint supports make every off-diagonal product `R_rj R_rk`
/// (j ≠ k) exactly zero, so the Gram is `diag(Σ_r R_rj²)`. A random-effect
/// factor's ridge root has this form, and at thousands of levels the dense
/// product is an `O(width³)` step whose result is zero off its diagonal. The
/// structural test reads each entry once, the same scan the diagonal sum
/// needs; `None` means some row couples two columns.
pub fn penalty_root_gram_diagonal(root: ArrayView2<'_, f64>) -> Option<Array1<f64>> {
    let mut diagonal = Array1::<f64>::zeros(root.ncols());
    for row in root.outer_iter() {
        let mut support = None;
        for (col, &value) in row.iter().enumerate() {
            if value == 0.0 {
                continue;
            }
            if support.is_some() {
                return None;
            }
            support = Some((col, value));
        }
        if let Some((col, value)) = support {
            diagonal[col] += value * value;
        }
    }
    Some(diagonal)
}

/// `RᵀR` for a penalty root `R`, formed in `O(rank · width)` when
/// [`penalty_root_gram_diagonal`] finds it diagonal and densely otherwise.
pub fn penalty_root_gram(root: ArrayView2<'_, f64>) -> Array2<f64> {
    match penalty_root_gram_diagonal(root) {
        Some(diagonal) => Array2::from_diag(&diagonal),
        None => root.t().dot(&root),
    }
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
        }
    }

    pub fn dim(&self) -> usize {
        match self {
            Self::DenseRoot(root) | Self::DenseRootCentered { root, .. } => root.ncols(),
            Self::BlockRoot { total_dim, .. } | Self::BlockRootCentered { total_dim, .. } => {
                *total_dim
            }
        }
    }

    pub fn uses_operator_fast_path(&self) -> bool {
        matches!(
            self,
            Self::BlockRoot { .. }
                | Self::BlockRootCentered { .. }
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
            None => return self.clone(),
        };
        // `R Π = R − (R N) Nᵀ`, through the shared primitive so this coordinate
        // and the term layer's `CanonicalPenalty` project identically.
        let prior_mean = self.prior_mean_block();
        match project_block_root_out_of_null_directions(
            root.view(),
            start,
            end,
            total_dim,
            null_basis,
        ) {
            ProjectedBlockRoot::Unchanged => self.clone(),
            ProjectedBlockRoot::BlockLocal { block, .. } => match prior_mean {
                Some(mean) => {
                    Self::from_block_root_with_mean(block, start, end, total_dim, mean.to_owned())
                }
                None => Self::from_block_root(block, start, end, total_dim),
            },
            ProjectedBlockRoot::FullWidth {
                root: projected, ..
            } => match prior_mean {
                Some(mean) => {
                    let mut full_mean = Array1::<f64>::zeros(total_dim);
                    full_mean.slice_mut(ndarray::s![start..end]).assign(&mean);
                    Self::from_dense_root_with_mean(projected, full_mean)
                }
                None => Self::from_dense_root(projected),
            },
        }
    }

    /// The block-local prior mean, when this coordinate is centered.
    fn prior_mean_block(&self) -> Option<ArrayView1<'_, f64>> {
        match self {
            Self::DenseRootCentered { prior_mean, .. }
            | Self::BlockRootCentered { prior_mean, .. } => Some(prior_mean.view()),
            Self::DenseRoot(_) | Self::BlockRoot { .. } => None,
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
            },
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
                let mut out = penalty_root_gram(root.view());
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
                let mut block = penalty_root_gram(root.view());
                block *= scale;
                out.slice_mut(ndarray::s![*start..*end, *start..*end])
                    .assign(&block);
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
                let mut out = penalty_root_gram(root.view());
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
                let mut block = penalty_root_gram(root.view());
                block *= scale;
                (block, *start, *end)
            }
        }
    }

    /// Whether this coordinate has block structure (not full-rank dense).
    pub fn is_block_local(&self) -> bool {
        matches!(
            self,
            Self::BlockRoot { .. }
                | Self::BlockRootCentered { .. }
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
    /// squares that residual (#2644). Every variant returns `Some` for a finite,
    /// non-negative `scale`.
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
    /// `Sₖ = RₖᵀRₖ`), and NEVER from a coordinate's position (`start`) in the
    /// joint layout. Swapping `s(x)+s(z)` ↔ `s(z)+s(x)` or
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
                // Orthogonal-invariants of the symmetric Sₖ = RₖᵀRₖ: the power
                // sums Σλ (trace), Σλ² (= ‖Sₖ‖²_F), Σλ³ (tr(Sₖ³)). Each is a
                // symmetric function of Sₖ's eigenvalues, so they are unchanged
                // by any orthonormal change of basis of the block coordinates
                // (hence by which joint block the penalty occupies) and by the
                // order of the terms. Together with rank and width they form a
                // strong placement-independent fingerprint without an
                // eigendecomposition.
                // A structurally diagonal Sₖ has its diagonal as spectrum.
                let (trace1, frob_sq, trace3) = match penalty_root_gram_diagonal(root.view()) {
                    Some(d) => (
                        d.sum(),
                        d.iter().map(|&x| x * x).sum::<f64>(),
                        d.iter().map(|&x| x * x * x).sum::<f64>(),
                    ),
                    None => {
                        let sk = root.t().dot(root);
                        let n = sk.nrows().min(sk.ncols());
                        let trace1 = (0..n).map(|i| sk[[i, i]]).sum::<f64>();
                        let frob_sq = sk.iter().map(|&x| x * x).sum::<f64>(); // = Σλ²
                        let sk3diag = sk.dot(&sk).dot(&sk);
                        let trace3 = (0..n).map(|i| sk3diag[[i, i]]).sum::<f64>();
                        (trace1, frob_sq, trace3)
                    }
                };
                let mut invariants = [quant(trace1), quant(frob_sq), quant(trace3)];
                // Power sums are already order-agnostic; sorting is a harmless
                // guard against any future addition of non-symmetric summaries.
                invariants.sort_unstable();
                invariants.hash(&mut hasher);
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

    /// The block-local projection primitive reproduces the dense
    /// `R_full − (R_full N) Nᵀ` in each of its three outcomes, without ever
    /// being handed the full-width root.
    #[test]
    fn projected_block_root_matches_the_dense_projection() {
        let root = array![[1.0_f64, 0.5, -0.25], [0.0, 2.0, 0.75]];
        let (start, end, total) = (2, 5, 7);
        let mut full = Array2::<f64>::zeros((2, total));
        full.slice_mut(ndarray::s![.., start..end]).assign(&root);
        let dense = |null_basis: &Array2<f64>| &full - &full.dot(null_basis).dot(&null_basis.t());
        let n = 1.0_f64 / 2.0_f64.sqrt();

        // Null directions only on columns outside the block: `R N = 0`.
        let outside = array![[n], [n], [0.0], [0.0], [0.0], [0.0], [0.0]];
        let unchanged = project_block_root_out_of_null_directions(
            root.view(),
            start,
            end,
            total,
            outside.view(),
        );
        assert!(matches!(unchanged, ProjectedBlockRoot::Unchanged));
        assert_eq!(dense(&outside), full);

        // Null direction inside the block: the projection stays block-local.
        let inside = array![[0.0_f64], [0.0], [n], [0.0], [-n], [0.0], [0.0]];
        let local = project_block_root_out_of_null_directions(
            root.view(),
            start,
            end,
            total,
            inside.view(),
        );
        let expected = dense(&inside);
        let ProjectedBlockRoot::BlockLocal { block, moved } = &local else {
            panic!("a block-supported null basis must stay block-local, got {local:?}");
        };
        let gap = (&expected.slice(ndarray::s![.., start..end]) - block)
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(gap <= 1e-15, "block-local projection gap {gap:e}");
        assert!(
            expected
                .slice(ndarray::s![.., ..start])
                .iter()
                .chain(expected.slice(ndarray::s![.., end..]).iter())
                .all(|value| *value == 0.0)
        );
        let dense_moved = (&expected - &full)
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!((moved - dense_moved).abs() <= 1e-15);

        // Null direction straddling the block edge: full-width support.
        let straddling = array![[0.0_f64], [n], [n], [0.0], [0.0], [0.0], [0.0]];
        let wide = project_block_root_out_of_null_directions(
            root.view(),
            start,
            end,
            total,
            straddling.view(),
        );
        let expected = dense(&straddling);
        let ProjectedBlockRoot::FullWidth { root: projected, moved } = &wide else {
            panic!("a straddling null basis must widen the root, got {wide:?}");
        };
        let gap = (&expected - projected)
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(gap <= 1e-15, "full-width projection gap {gap:e}");
        let dense_moved = (&expected - &full)
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!((moved - dense_moved).abs() <= 1e-15);
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

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
///
/// Every variant's penalty energy is `‖R_k β_block − c_k‖²`, and the variants
/// differ only in how the anchor `c_k` is held: zero (`DenseRoot`,
/// `BlockRoot`) or a root-space offset with no coefficient-space preimage
/// (`DenseRootOffset`). The anchor enters only the shifted channels
/// ([`Self::shifted_quadratic`], [`Self::apply_shifted_penalty`]); every other
/// operator is the curvature `S_k = R_kᵀR_k`, which the anchor does not change.
#[derive(Clone, Debug)]
pub enum PenaltyCoordinate {
    DenseRoot(Array2<f64>),
    /// `‖R β − c‖²` with `c` (length `rank`) held in the root's range space.
    ///
    /// This is what a penalty becomes when it is restricted onto an affine
    /// constraint face ([`Self::project_into_subspace`]): the restricted
    /// energy `‖R z β_f + R β_⊥‖²` has a root `R z` whose range need not
    /// contain `R β_⊥`, so no coefficient-space anchor reproduces it — the
    /// best one, `−(R z)⁺ R β_⊥`, drops a `λ`-scaled constant whose
    /// `ρ`-derivative the outer gradient needs (gam#4170).
    DenseRootOffset {
        root: Array2<f64>,
        root_offset: Array1<f64>,
    },
    BlockRoot {
        root: Array2<f64>,
        start: usize,
        end: usize,
        total_dim: usize,
    },
}

impl PenaltyCoordinate {
    pub fn from_dense_root(root: Array2<f64>) -> Self {
        Self::DenseRoot(root)
    }

    /// The penalty `‖R β − c‖²` with root-space offset `c`; a zero offset is
    /// the plain `DenseRoot`.
    pub fn from_dense_root_with_offset(root: Array2<f64>, root_offset: Array1<f64>) -> Self {
        assert_eq!(
            root.nrows(),
            root_offset.len(),
            "penalty root-space offset length must match the root's row count"
        );
        if root_offset.iter().all(|&value| value == 0.0) {
            Self::DenseRoot(root)
        } else {
            Self::DenseRootOffset { root, root_offset }
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

    pub fn rank(&self) -> usize {
        match self {
            Self::DenseRoot(root)
            | Self::DenseRootOffset { root, .. }
            | Self::BlockRoot { root, .. } => root.nrows(),
        }
    }

    pub fn dim(&self) -> usize {
        match self {
            Self::DenseRoot(root) | Self::DenseRootOffset { root, .. } => root.ncols(),
            Self::BlockRoot { total_dim, .. } => *total_dim,
        }
    }

    pub fn uses_operator_fast_path(&self) -> bool {
        matches!(self, Self::BlockRoot { .. })
    }

    /// Borrow the canonical penalty root in its native block chart.
    ///
    /// The root rows are the authoritative structural range coordinates: their
    /// count is `rank()` and must not be rediscovered by eigendecomposing the
    /// squared Gram `RᵀR`, which can promote roundoff in a structural zero.
    pub fn block_local_root(&self) -> Option<(&Array2<f64>, usize, usize)> {
        match self {
            Self::DenseRoot(root) | Self::DenseRootOffset { root, .. } => {
                Some((root, 0, root.ncols()))
            }
            Self::BlockRoot {
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
    /// block-local and a dense coordinate is returned.
    ///
    /// A `DenseRootOffset` coordinate is refused: its anchor lives in root
    /// space, so there is no coefficient-space `Πβ` form of it, and it exists
    /// only as the output of the constraint-face restriction, which runs after
    /// this projection.
    ///
    /// Returns `self` unchanged when `N` has no columns.
    pub fn project_out_null_directions(&self, null_basis: ArrayView2<'_, f64>) -> Self {
        if null_basis.ncols() == 0 {
            return self.clone();
        }
        assert!(
            !matches!(self, Self::DenseRootOffset { .. }),
            "PenaltyCoordinate::project_out_null_directions: a root-space offset has no \
             coefficient-space preimage to project; apply the null split before the \
             constraint-face restriction"
        );
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
        match project_block_root_out_of_null_directions(
            root.view(),
            start,
            end,
            total_dim,
            null_basis,
        ) {
            ProjectedBlockRoot::Unchanged => self.clone(),
            ProjectedBlockRoot::BlockLocal { block, .. } => {
                Self::from_block_root(block, start, end, total_dim)
            }
            ProjectedBlockRoot::FullWidth {
                root: projected, ..
            } => Self::from_dense_root(projected),
        }
    }

    /// Restrict this penalty coordinate onto the active constraint face through
    /// `face_point`, parameterized by the orthonormal columns of `z` (shape
    /// `p × m`, `m ≤ p`, `zᵀz = I`) spanning the face's directions.
    ///
    /// When a linear-inequality active set is non-empty, the inner solve and the
    /// penalized Hessian are reduced to the face coordinate `β_f = zᵀβ` of
    /// dimension `m = p − active_set_size`. Every point of the face is
    /// `β = z β_f + β_⊥` with the same `β_⊥ = (I − z zᵀ) face_point`, which is
    /// non-zero whenever an active constraint has a non-zero right-hand side.
    /// The penalty must move in lockstep, and its energy on the face is
    ///
    /// ```text
    ///     ‖R_k β − c_k‖² = ‖R_k z β_f − c_k'‖²,
    ///     c_k' = c_k − R_k β_⊥
    /// ```
    ///
    /// (`c_k` zero where the coordinate carries none). The reduced root is
    /// `R_k z` (for a block-local root on `β[start..end]`, `R_k ·
    /// z[start..end, :]`: the block structure does not survive an arbitrary
    /// face rotation), and the anchor `c_k'` is a ROOT-SPACE offset: it lies
    /// in `range(R_k z)` only when `β_⊥` does not reach the constraint
    /// normals through `S_k`, so it cannot be rewritten as a coefficient-space
    /// anchor without dropping a `λ_k`-scaled constant from the value and a
    /// `λ_k zᵀS_k(I − z zᵀ)β̂` term from the gradient (gam#4170). The reduced
    /// coordinate evaluated at `zᵀβ` reproduces the full penalty at `β`
    /// exactly, its shifted score is `zᵀ S_k β`, and its curvature is
    /// `zᵀ S_k z`.
    ///
    /// This keeps `dim()` equal to the reduced `beta.len()`, which
    /// `InnerSolutionBuilder::build` asserts.
    pub fn project_into_subspace(&self, z: &Array2<f64>, face_point: ArrayView1<'_, f64>) -> Self {
        assert_eq!(
            z.nrows(),
            self.dim(),
            "PenaltyCoordinate::project_into_subspace: free-basis row count {} does not match coordinate dimension {}",
            z.nrows(),
            self.dim()
        );
        assert_eq!(
            face_point.len(),
            self.dim(),
            "PenaltyCoordinate::project_into_subspace: face point length {} does not match coordinate dimension {}",
            face_point.len(),
            self.dim()
        );
        let (root, start, end, root_offset) = match self {
            Self::DenseRoot(root) => (root, 0, root.ncols(), None),
            Self::DenseRootOffset { root, root_offset } => {
                (root, 0, root.ncols(), Some(root_offset))
            }
            Self::BlockRoot {
                root, start, end, ..
            } => (root, *start, *end, None),
        };
        // `β_⊥ = (I − z zᵀ) face_point`, the component every face point shares.
        let off_face = &face_point - &z.dot(&z.t().dot(&face_point));
        let anchor = -off_face.slice(ndarray::s![start..end]).to_owned();
        let mut offset = root.dot(&anchor);
        if let Some(existing) = root_offset {
            offset += existing;
        }
        let z_block = z.slice(ndarray::s![start..end, ..]);
        Self::from_dense_root_with_offset(root.dot(&z_block), offset)
    }

    pub(crate) fn apply_root(&self, beta: &Array1<f64>) -> Array1<f64> {
        assert_eq!(beta.len(), self.dim());
        match self {
            Self::DenseRoot(root) | Self::DenseRootOffset { root, .. } => root.dot(beta),
            Self::BlockRoot {
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
            Self::DenseRoot(root) | Self::DenseRootOffset { root, .. } => {
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
            } => {
                let beta_block = beta.slice(ndarray::s![*start..*end]);
                let mut root_beta = Array1::<f64>::zeros(root.nrows());
                dense::matvec_into(root, beta_block, root_beta.view_mut());
                let out_block = out.slice_mut(ndarray::s![*start..*end]);
                dense::transpose_matvec_scaled_add_into(root, root_beta.view(), scale, out_block);
            }
        }
    }

    pub fn quadratic(&self, beta: &Array1<f64>, scale: f64) -> f64 {
        let root_beta = self.apply_root(beta);
        scale * root_beta.dot(&root_beta)
    }

    pub fn apply_shifted_penalty(&self, beta: &Array1<f64>, scale: f64) -> Array1<f64> {
        match self {
            Self::DenseRootOffset { root, root_offset } => {
                let residual = root.dot(beta) - root_offset;
                let mut out = root.t().dot(&residual);
                out *= scale;
                out
            }
            // Anchored at the origin: the shifted channel IS the plain one.
            // Listed rather than caught by `_` so a new variant has to declare
            // which side of the anchor it falls on.
            Self::DenseRoot(_) | Self::BlockRoot { .. } => self.apply_penalty(beta, scale),
        }
    }

    pub fn shifted_quadratic(&self, beta: &Array1<f64>, scale: f64) -> f64 {
        match self {
            Self::DenseRootOffset { root, root_offset } => {
                let residual = root.dot(beta) - root_offset;
                scale * residual.dot(&residual)
            }
            Self::DenseRoot(_) | Self::BlockRoot { .. } => self.quadratic(beta, scale),
        }
    }

    pub fn scaled_dense_matrix(&self, scale: f64) -> Array2<f64> {
        match self {
            Self::DenseRoot(root) | Self::DenseRootOffset { root, .. } => {
                let mut out = penalty_root_gram(root.view());
                out *= scale;
                out
            }
            Self::BlockRoot {
                root,
                start,
                end,
                total_dim,
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
            Self::DenseRoot(root) | Self::DenseRootOffset { root, .. } => {
                let mut out = penalty_root_gram(root.view());
                out *= scale;
                let p = out.nrows();
                (out, 0, p)
            }
            Self::BlockRoot {
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
        matches!(self, Self::BlockRoot { .. })
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
            Self::DenseRoot(root) | Self::DenseRootOffset { root, .. } => {
                Some((root * sqrt_scale, 0, root.ncols()))
            }
            Self::BlockRoot {
                root, start, end, ..
            } => Some((root * sqrt_scale, *start, *end)),
        }
    }

    /// Apply λ_k S_k to a vector v without materializing the full matrix.
    /// For BlockRoot: extracts v[start..end], multiplies by local S_k, embeds result.
    pub fn scaled_matvec(&self, v: &Array1<f64>, scale: f64) -> Array1<f64> {
        match self {
            Self::DenseRoot(root) | Self::DenseRootOffset { root, .. } => {
                let root_v = root.dot(v);
                let mut out = root.t().dot(&root_v);
                out *= scale;
                out
            }
            Self::BlockRoot {
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
            | Self::DenseRootOffset { root, .. }
            | Self::BlockRoot { root, .. } => {
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
    fn from_dense_root_with_zero_offset_degrades_to_dense_root() {
        let root = identity_root(2);
        let offset = Array1::<f64>::zeros(2);
        let pc = PenaltyCoordinate::from_dense_root_with_offset(root, offset);
        assert!(matches!(pc, PenaltyCoordinate::DenseRoot(_)));
    }

    #[test]
    fn from_dense_root_with_nonzero_offset_creates_offset_variant() {
        let root = identity_root(2);
        let offset = array![1.0_f64, 0.0];
        let pc = PenaltyCoordinate::from_dense_root_with_offset(root, offset);
        assert!(matches!(pc, PenaltyCoordinate::DenseRootOffset { .. }));
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

    // ─── gam#4170: constraint-face restriction keeps the face offset ────────

    /// The face `aᵀβ = 0.9` (`a = (e0 + e1)/√2`), its orthonormal directions
    /// `z`, and a face point whose off-face component `0.9 a` is non-zero.
    fn affine_face_fixture() -> (Array2<f64>, Array1<f64>) {
        let h = 1.0_f64 / 2.0_f64.sqrt();
        let z = array![
            [h, 0.0, 0.0],
            [-h, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let normal = array![h, h, 0.0, 0.0];
        let face_point = z.dot(&array![0.5_f64, -1.2, 0.8]) + &(&normal * 0.9);
        (z, face_point)
    }

    fn max_abs_gap(left: &Array1<f64>, right: &Array1<f64>) -> f64 {
        (left - right)
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()))
    }

    /// Check that the face-reduced coordinate evaluated at `zᵀβ̂` reproduces
    /// the full coordinate's shifted value, shifted score (through `zᵀ`) and
    /// curvature `zᵀSz` exactly.
    fn assert_face_reduction_is_exact(
        full: &PenaltyCoordinate,
        z: &Array2<f64>,
        face_point: &Array1<f64>,
    ) {
        let reduced = full.project_into_subspace(z, face_point.view());
        assert_eq!(reduced.dim(), z.ncols());
        let beta_face = z.t().dot(face_point);
        let scale = 2.5_f64;

        let full_value = full.shifted_quadratic(face_point, scale);
        let reduced_value = reduced.shifted_quadratic(&beta_face, scale);
        assert!(
            (reduced_value - full_value).abs() <= 1e-12 * full_value.abs().max(1.0),
            "face-reduced shifted value {reduced_value:.15e} must equal the full \
             penalty at the face point {full_value:.15e}"
        );

        let full_score = z.t().dot(&full.apply_shifted_penalty(face_point, scale));
        let reduced_score = reduced.apply_shifted_penalty(&beta_face, scale);
        let score_gap = max_abs_gap(&reduced_score, &full_score);
        assert!(
            score_gap <= 1e-12 * full_score.iter().fold(1.0_f64, |a, v| a.max(v.abs())),
            "face-reduced shifted score {reduced_score:?} must equal zᵀ of the full \
             shifted score {full_score:?} (gap {score_gap:e})"
        );

        let full_curvature = z.t().dot(&full.scaled_dense_matrix(1.0)).dot(z);
        let curvature_gap = (&reduced.scaled_dense_matrix(1.0) - &full_curvature)
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(curvature_gap <= 1e-12, "curvature gap {curvature_gap:e}");
    }

    #[test]
    fn face_reduction_of_a_dense_penalty_is_exact() {
        let (z, face_point) = affine_face_fixture();
        let root = array![
            [1.0_f64, 2.0, 0.0, 0.5],
            [0.0, 1.0, -1.0, 0.0],
            [0.3, 0.0, 1.0, 2.0]
        ];
        let full = PenaltyCoordinate::from_dense_root(root);

        // The fixture must exercise the term a coefficient-space anchor drops:
        // `zᵀ S (I − z zᵀ) β̂` is the gradient piece gam#4170 lost.
        let s = full.scaled_dense_matrix(1.0);
        let off_face = &face_point - &z.dot(&z.t().dot(&face_point));
        let dropped = z.t().dot(&s.dot(&off_face));
        assert!(
            dropped.iter().any(|value| value.abs() > 1e-2),
            "fixture must couple the off-face component into the face score"
        );

        assert!(matches!(
            full.project_into_subspace(&z, face_point.view()),
            PenaltyCoordinate::DenseRootOffset { .. }
        ));
        assert_face_reduction_is_exact(&full, &z, &face_point);
    }

    #[test]
    fn face_reduction_of_a_block_penalty_is_exact() {
        // The same face embedded in a five-coefficient model, the block on
        // `β[1..3)` straddling the constraint normal's support.
        let h = 1.0_f64 / 2.0_f64.sqrt();
        let z = array![
            [0.0_f64, 1.0, 0.0, 0.0],
            [h, 0.0, 0.0, 0.0],
            [-h, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ];
        let normal = array![0.0_f64, h, h, 0.0, 0.0];
        let face_point = z.dot(&array![0.5_f64, -0.3, 1.1, -0.4]) + &(&normal * -0.6);
        // A block penalty on a face through a non-zero right-hand side
        // acquires an offset: `β_⊥` alone moves the anchor.
        let full =
            PenaltyCoordinate::from_block_root(array![[1.0_f64, 0.25], [0.5, -2.0]], 1, 3, 5);
        assert!(matches!(
            full.project_into_subspace(&z, face_point.view()),
            PenaltyCoordinate::DenseRootOffset { .. }
        ));
        assert_face_reduction_is_exact(&full, &z, &face_point);
    }

    /// A face through the origin (every active right-hand side zero) leaves the
    /// penalty unanchored: the restriction is the plain `R z`.
    #[test]
    fn face_reduction_through_the_origin_is_the_plain_root() {
        let z = array![[1.0_f64, 0.0], [0.0, 0.0], [0.0, 1.0]];
        let face_point = array![0.7_f64, 0.0, -1.9];
        let root = array![[1.0_f64, 2.0, -0.5], [0.0, 1.5, 3.0]];
        let reduced = PenaltyCoordinate::from_dense_root(root.clone())
            .project_into_subspace(&z, face_point.view());
        let PenaltyCoordinate::DenseRoot(reduced_root) = &reduced else {
            panic!("a zero face offset must yield a plain dense root, got {reduced:?}");
        };
        assert_eq!(reduced_root, &root.dot(&z));
    }

    /// The null split must run before the face restriction; a root-space
    /// offset has no coefficient-space mean for `Π` to act on.
    #[test]
    #[should_panic(expected = "root-space offset")]
    fn null_split_refuses_a_face_offset_coordinate() {
        let coord =
            PenaltyCoordinate::from_dense_root_with_offset(array![[1.0_f64, 0.0]], array![0.5]);
        let n = 1.0_f64 / 2.0_f64.sqrt();
        let projected = coord.project_out_null_directions(array![[n], [n]].view());
        panic!("null split unexpectedly returned {projected:?}");
    }
}

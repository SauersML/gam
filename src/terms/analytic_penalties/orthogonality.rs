use super::*;

// ---------------------------------------------------------------------------
// Block-orthogonality penalty
// ---------------------------------------------------------------------------

/// Between-block-only orthogonality on a row-major matrix-valued latent
/// block.
///
/// Lives on the extension-coordinate tier. Penalizes the squared Frobenius
/// norm of the between-block Gram matrices, where `T` is the row-major
/// `n_eff × latent_dim` view of the target slice and `groups` partitions
/// the latent axes into disjoint subsets:
///
/// ```text
///   P(T) = ½ · w · Σ_{g < h} ‖ T[:, group_g]^T T[:, group_h] ‖²_F
/// ```
///
/// Within-block structure is unconstrained: this penalty only pushes different
/// groups into mutually orthogonal subspaces. In the SAE objective it is the
/// block-level separability / gauge term for latent decompositions where known
/// or supervised coordinates should not leak into free coordinates.
///
/// Typical use: gauge-fixing a latent decomposition where one block has been
/// supervised (e.g. anchored to known coordinates) and a free block needs to
/// inhabit the orthogonal complement of that supervision. Pair with per-block
/// ARD or sparsity when you also want within-block axis selection.
///
/// Gotchas:
///
/// * `groups` must be a true partition of all latent axes: every axis appears
///   exactly once, and at least two groups are required.
/// * The Hessian is dense across rows and axes even though an exact diagonal is
///   available for diagnostics/preconditioning. Use the HVP for the full
///   Newton curvature.
#[derive(Debug, Clone)]
pub struct BlockOrthogonalityPenalty {
    pub target: PsiSlice,
    pub groups: Vec<Vec<usize>>,
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Number of rows in the row-major matrix-valued latent block.
    pub n_eff: usize,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}

impl BlockOrthogonalityPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        groups: Vec<Vec<usize>>,
        weight: f64,
        n_eff: usize,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err("BlockOrthogonalityPenalty::new requires a non-empty target".to_string());
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "BlockOrthogonalityPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("BlockOrthogonalityPenalty::new requires n_eff > 0".to_string());
        }
        if !target.len().is_multiple_of(n_eff) {
            return Err(format!(
                "BlockOrthogonalityPenalty::new target length {} is not divisible by n_eff {}",
                target.len(),
                n_eff
            ));
        }
        let latent_dim = target.len() / n_eff;
        if let Some(expected_dim) = target.latent_dim {
            let expected = n_eff.checked_mul(expected_dim).ok_or_else(|| {
                "BlockOrthogonalityPenalty::new target shape overflows usize".to_string()
            })?;
            if expected != target.len() {
                return Err(format!(
                    "BlockOrthogonalityPenalty::new target length {} does not match n_eff {} × latent_dim {}",
                    target.len(),
                    n_eff,
                    expected_dim
                ));
            }
        }
        if groups.len() < 2 {
            return Err("BlockOrthogonalityPenalty::new requires at least two groups".to_string());
        }
        let mut seen = vec![false; latent_dim];
        for (group_idx, group) in groups.iter().enumerate() {
            if group.is_empty() {
                return Err(format!(
                    "BlockOrthogonalityPenalty::new groups[{group_idx}] must not be empty"
                ));
            }
            for &axis in group {
                if axis >= latent_dim {
                    return Err(format!(
                        "BlockOrthogonalityPenalty::new groups[{group_idx}] axis {axis} exceeds latent_dim {latent_dim}"
                    ));
                }
                if seen[axis] {
                    return Err(format!(
                        "BlockOrthogonalityPenalty::new axis {axis} appears in more than one group"
                    ));
                }
                seen[axis] = true;
            }
        }
        for (axis, present) in seen.iter().copied().enumerate() {
            if !present {
                return Err(format!(
                    "BlockOrthogonalityPenalty::new groups must partition latent axes; missing axis {axis}"
                ));
            }
        }
        Ok(Self {
            target,
            groups,
            weight,
            n_eff,
            learnable_weight,
            rho_index: 0,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            resolve_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    fn latent_dim(&self, target_len: usize) -> Option<usize> {
        if self.n_eff == 0 || !target_len.is_multiple_of(self.n_eff) {
            assert_eq!(
                target_len % self.n_eff.max(1),
                0,
                "target length must be divisible by n_eff"
            );
            return None;
        }
        Some(target_len / self.n_eff)
    }

    fn target_matrix<'a>(&self, target: ArrayView1<'a, f64>) -> Option<ArrayView2<'a, f64>> {
        let d = self.latent_dim(target.len())?;
        target.into_shape_with_order((self.n_eff, d)).ok()
    }

    fn flatten_matrix(m: &Array2<f64>) -> Array1<f64> {
        let n_obs = m.nrows();
        let d = m.ncols();
        let mut out = Array1::<f64>::zeros(n_obs * d);
        for n in 0..n_obs {
            for a in 0..d {
                out[n * d + a] = m[[n, a]];
            }
        }
        out
    }

    fn cross_gram(t: ArrayView2<'_, f64>, left: &[usize], right: &[usize]) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((left.len(), right.len()));
        for (li, &a) in left.iter().enumerate() {
            for (ri, &b) in right.iter().enumerate() {
                let mut s = 0.0;
                for n in 0..t.nrows() {
                    s += t[[n, a]] * t[[n, b]];
                }
                out[[li, ri]] = s;
            }
        }
        out
    }

    /// `out[li, ri] = Σ_n a[n, left[li]] · b[n, right[ri]]` — two-argument
    /// cross-gram used to assemble the directional derivative of `C_{gh}` in
    /// direction `v`:  `∂_v C_{gh}[gi, hi] = Σ_n {v[n, axes_g[gi]] · t[n, axes_h[hi]] + t[n, axes_g[gi]] · v[n, axes_h[hi]]}`.
    /// The `cross_gram` helper (single-input self-product) was previously
    /// (mis)used for both terms, but `cross_gram(v, h, g) + cross_gram(t, h, g)`
    /// is the unrelated quantity `(v⊗v) + (t⊗t)`, not `(v⊗t) + (t⊗v)`.
    fn mixed_cross_gram(
        a: ArrayView2<'_, f64>,
        b: ArrayView2<'_, f64>,
        left: &[usize],
        right: &[usize],
    ) -> Array2<f64> {
        assert_eq!(a.nrows(), b.nrows(), "mixed_cross_gram row mismatch");
        let mut out = Array2::<f64>::zeros((left.len(), right.len()));
        for (li, &al) in left.iter().enumerate() {
            for (ri, &br) in right.iter().enumerate() {
                let mut s = 0.0;
                for n in 0..a.nrows() {
                    s += a[[n, al]] * b[[n, br]];
                }
                out[[li, ri]] = s;
            }
        }
        out
    }

    fn add_right_times_cross(
        out: &mut Array2<f64>,
        right: ArrayView2<'_, f64>,
        left_axes: &[usize],
        right_axes: &[usize],
        cross_right_left: ArrayView2<'_, f64>,
        factor: f64,
    ) {
        assert_eq!(cross_right_left.dim(), (right_axes.len(), left_axes.len()));
        for n in 0..out.nrows() {
            for (li, &left_axis) in left_axes.iter().enumerate() {
                let mut s = 0.0;
                for (ri, &right_axis) in right_axes.iter().enumerate() {
                    s += right[[n, right_axis]] * cross_right_left[[ri, li]];
                }
                out[[n, left_axis]] += factor * s;
            }
        }
    }

    fn hvp_with_precomputed_cross(
        &self,
        t: ArrayView2<'_, f64>,
        cross: &[Vec<Option<Array2<f64>>>],
        v: ArrayView2<'_, f64>,
        weight: f64,
    ) -> Array2<f64> {
        assert_eq!(v.dim(), t.dim(), "hvp matrix dimension mismatch");
        if v.dim() != t.dim() {
            return Array2::<f64>::zeros(t.dim());
        }
        let mut out = Array2::<f64>::zeros(t.dim());
        for g in 0..self.groups.len() {
            let group_g = &self.groups[g];
            for h in 0..self.groups.len() {
                if g == h {
                    continue;
                }
                let group_h = &self.groups[h];
                let c_hg = cross[h][g]
                    .as_ref()
                    .expect("between-block cross Gram must be precomputed");
                // Linear contribution: w · Σ_b C_{g,h}[i,b] · v[n, axes_h[b]] —
                // the C-direct piece of d/dv (∂P/∂t).
                Self::add_right_times_cross(&mut out, v, group_g, group_h, c_hg.view(), weight);

                // Directional derivative of C_{hg} in direction v:
                //   ∂_v C_{hg}[hi, gi] = Σ_n {v[n, axes_h[hi]] · t[n, axes_g[gi]]
                //                            + t[n, axes_h[hi]] · v[n, axes_g[gi]]}
                // = MixedCross(v, t, h, g) + MixedCross(t, v, h, g).
                // The earlier formulation used `cross_gram(v, h, g) +
                // cross_gram(t, h, g)`, which is `(v⊗v) + (t⊗t)` — quadratic in v
                // (resp. independent of v) and unrelated to the JVP. The bug made
                // the Hessian non-symmetric (it added a fixed `(t⊗t)`-driven term
                // to every column), violated the gradient/Hessian consistency
                // check that REML's spectral solve relies on, and the sibling
                // `OrthogonalityPenalty::hvp_with_precomputed_m` already uses the
                // correct `v_c · t_b + t_c · v_b` mixed pattern.
                let dv_h_g = Self::mixed_cross_gram(v, t, group_h, group_g);
                let tv_h_g = Self::mixed_cross_gram(t, v, group_h, group_g);
                let mut d_c_hg = dv_h_g;
                d_c_hg += &tv_h_g;
                Self::add_right_times_cross(&mut out, t, group_g, group_h, d_c_hg.view(), weight);
            }
        }
        out
    }

    fn precompute_cross(&self, t: ArrayView2<'_, f64>) -> Vec<Vec<Option<Array2<f64>>>> {
        let mut cross = vec![vec![None; self.groups.len()]; self.groups.len()];
        for g in 0..self.groups.len() {
            for h in 0..self.groups.len() {
                if g != h {
                    cross[g][h] = Some(Self::cross_gram(t, &self.groups[g], &self.groups[h]));
                }
            }
        }
        cross
    }

    /// Materialize the between-block orthogonality Hessian for small-block
    /// spectral paths.
    pub fn as_dense(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array2<f64> {
        let n = target.len();
        let Some(t) = self.target_matrix(target) else {
            return Array2::<f64>::zeros((n, n));
        };
        let cross = self.precompute_cross(t.view());
        let weight = self.resolved_weight(rho);
        let mut dense = Array2::<f64>::zeros((n, n));
        let mut e = Array1::<f64>::zeros(n);
        for j in 0..n {
            e[j] = 1.0;
            let Some(e_mat) = self.target_matrix(e.view()) else {
                return Array2::<f64>::zeros((n, n));
            };
            let col = self.hvp_with_precomputed_cross(t.view(), &cross, e_mat, weight);
            for i in 0..n {
                dense[[i, j]] = col[[i / t.ncols(), i % t.ncols()]];
            }
            e[j] = 0.0;
        }
        dense
    }
}

impl AnalyticPenalty for BlockOrthogonalityPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(t) = self.target_matrix(target) else {
            return 0.0;
        };
        let mut acc = 0.0;
        for g in 0..self.groups.len() {
            for h in (g + 1)..self.groups.len() {
                let c = Self::cross_gram(t.view(), &self.groups[g], &self.groups[h]);
                for &v in c.iter() {
                    acc += v * v;
                }
            }
        }
        0.5 * self.resolved_weight(rho) * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let cross = self.precompute_cross(t.view());
        let weight = self.resolved_weight(rho);
        let mut grad = Array2::<f64>::zeros(t.dim());
        for g in 0..self.groups.len() {
            for h in 0..self.groups.len() {
                if g == h {
                    continue;
                }
                let c_hg = cross[h][g]
                    .as_ref()
                    .expect("between-block cross Gram must be precomputed");
                Self::add_right_times_cross(
                    &mut grad,
                    t.view(),
                    &self.groups[g],
                    &self.groups[h],
                    c_hg.view(),
                    weight,
                );
            }
        }
        Self::flatten_matrix(&grad)
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let Some(v_mat) = self.target_matrix(v) else {
            return Array1::<f64>::zeros(target.len());
        };
        let cross = self.precompute_cross(t.view());
        let hv = self.hvp_with_precomputed_cross(
            t.view(),
            &cross,
            v_mat.view(),
            self.resolved_weight(rho),
        );
        Self::flatten_matrix(&hv)
    }

    fn hessian_diag(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Option<Array1<f64>> {
        let t = self.target_matrix(target)?;
        let n_obs = t.nrows();
        let d = t.ncols();
        let weight = self.resolved_weight(rho);
        let mut group_of = vec![usize::MAX; d];
        for (gi, group) in self.groups.iter().enumerate() {
            for &axis in group {
                group_of[axis] = gi;
            }
        }
        let mut out = Array1::<f64>::zeros(n_obs * d);
        for n in 0..n_obs {
            let mut row_sq = 0.0_f64;
            let mut group_sq = vec![0.0_f64; self.groups.len()];
            for b in 0..d {
                let v = t[[n, b]];
                let v2 = v * v;
                row_sq += v2;
                group_sq[group_of[b]] += v2;
            }
            for a in 0..d {
                let g = group_of[a];
                out[n * d + a] = weight * (row_sq - group_sq[g]);
            }
        }
        Some(out)
    }

    impl_learnable_weight_grad_rho!();

    impl_learnable_weight_rho_count!();

    fn name(&self) -> &str {
        "block_orthogonality"
    }

    impl_scalar_apply_schedule!(weight);
}

// ---------------------------------------------------------------------------
// Decoder column-space incoherence penalty
// ---------------------------------------------------------------------------

/// Cross-atom decoder column-space incoherence, restricted to co-activating
/// atom pairs (issue #671).
///
/// Lives on the β tier and targets the flat SAE decoder coefficient block. The
/// β layout concatenates the per-atom decoder blocks in atom order: atom `k`
/// owns `M_k · p_out` coefficients, stored as
/// `β[off_k + a·p_out + o]` for basis row `a` and output feature `o`.
/// The stored block is `B_k ∈ ℝ^{M_k × p_out}` with rows `B_k[a, :]`
/// representing decoder directions in output space.
///
/// The penalty is the co-activation-masked cross-column-space overlap
///
/// ```text
///   P = ½ · w · Σ_{j<k} W[j,k] · ‖B_j B_k^T‖²_F,
///   W[j,k] = ½ · (coactivation[j,k] + coactivation[k,j]).
/// ```
///
/// `coactivation[j,k]` is the mean over observations of
/// `gate[n,j] · gate[n,k]`; pairs that never co-fire (`W[j,k] = 0`) contribute
/// nothing. In the SAE objective this is the separability lever: atoms that
/// are active on the same examples are discouraged from spanning the same
/// decoder output directions, while unrelated atoms are not pushed apart just
/// because they both exist in the dictionary.
///
/// The Hessian used here is the Gauss-Newton (positive-semidefinite) curvature
/// of the Frobenius objective in `C`, dropping the indefinite second-order term
/// in `C`. This keeps the β-tier Newton / PIRLS curvature block PSD, matching
/// the other quadratic-on-Gram penalties.
///
/// Gotchas:
///
/// * `block_sizes` are decoder basis-row counts `M_k`, not output widths;
///   every atom shares the same `p_out`. Stored decoder blocks are
///   `(M_k, p_out)`, so `B_j B_k^T` is the cross-Gram of decoder directions in
///   output space and remains well-defined for heterogeneous `M_k`.
/// * The descriptor path builds a placeholder penalty; live SAE wiring replaces
///   the co-activation matrix with the current mean gate products.
/// * Offsets are interpreted against the vector passed to this penalty. In the
///   SAE decoder-incoherence path the registered target slice is zero-based;
///   callers using an already sliced target view must keep that convention.
#[derive(Debug, Clone)]
pub struct DecoderIncoherencePenalty {
    pub target: PsiSlice,
    /// Per-atom decoder basis-function counts `M_k`. The atom blocks are laid
    /// out contiguously in β order; `Σ_k M_k·p_out == target.len()`.
    pub block_sizes: Vec<usize>,
    /// Output / feature dimension `p_out` (decoder column count, shared by all
    /// atoms).
    pub p_out: usize,
    /// Atom count `K`. The operator only stores the SPARSE list of penalized
    /// atom pairs (`pairs`), not the dense `K×K` co-activation matrix — at
    /// `K = 32768` that dense matrix is 8 GiB. Every consumer of this operator
    /// already skipped pairs whose symmetrized weight is `0`, so storing only
    /// the nonzero pairs is exactly equivalent to the dense matrix while being
    /// linear in the number of co-active / near-collinear pairs (#1026).
    pub k_atoms: usize,
    /// Sparse penalized atom pairs `(j, k, w)` with `j < k` and the symmetrized
    /// weight `w = ½·(W[j,k] + W[k,j]) > 0` (this is exactly the value the old
    /// `pair_weight(j, k)` returned). Pairs with `w == 0` are omitted; the dense
    /// operator skipped them, so results are byte-identical.
    pub pairs: Vec<(usize, usize, f64)>,
    /// Base strength. If `learnable_weight` is true the resolved strength is
    /// `weight·exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}

impl DecoderIncoherencePenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        block_sizes: Vec<usize>,
        p_out: usize,
        coactivation: Array2<f64>,
        weight: f64,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err("DecoderIncoherencePenalty::new requires a non-empty target".to_string());
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "DecoderIncoherencePenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if p_out == 0 {
            return Err("DecoderIncoherencePenalty::new requires p_out > 0".to_string());
        }
        if block_sizes.len() < 2 {
            return Err(
                "DecoderIncoherencePenalty::new requires at least two atom blocks".to_string(),
            );
        }
        let k = block_sizes.len();
        if coactivation.dim() != (k, k) {
            return Err(format!(
                "DecoderIncoherencePenalty::new requires (K, K)=({k}, {k}) coactivation; got {:?}",
                coactivation.dim()
            ));
        }
        if !coactivation
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        {
            return Err(
                "DecoderIncoherencePenalty::new requires finite non-negative coactivation entries"
                    .to_string(),
            );
        }
        let mut total = 0usize;
        for (atom_idx, &m) in block_sizes.iter().enumerate() {
            if m == 0 {
                return Err(format!(
                    "DecoderIncoherencePenalty::new block_sizes[{atom_idx}] must be > 0"
                ));
            }
            let span = m.checked_mul(p_out).ok_or_else(|| {
                "DecoderIncoherencePenalty::new block span overflows usize".to_string()
            })?;
            total = total.checked_add(span).ok_or_else(|| {
                "DecoderIncoherencePenalty::new total span overflows usize".to_string()
            })?;
        }
        if total != target.len() {
            return Err(format!(
                "DecoderIncoherencePenalty::new Σ_k M_k·p_out = {total} does not match target length {}",
                target.len()
            ));
        }
        // Sparsify: store only the upper-triangular pairs whose symmetrized
        // weight `½·(W[j,k]+W[k,j])` is nonzero. The dense operator skipped every
        // pair with a zero symmetrized weight, so the sparse list reproduces it
        // bit-for-bit while never materializing the dense `K×K` matrix downstream.
        let mut pairs = Vec::new();
        for j in 0..k {
            for kk in (j + 1)..k {
                let w = 0.5 * (coactivation[[j, kk]] + coactivation[[kk, j]]);
                if w != 0.0 {
                    pairs.push((j, kk, w));
                }
            }
        }
        Ok(Self {
            target,
            block_sizes,
            p_out,
            k_atoms: k,
            pairs,
            weight,
            learnable_weight,
            rho_index: 0,
            weight_schedule: None,
        })
    }

    /// Sparse-pair constructor used by the SAE live wiring (#1026): build the
    /// operator directly from a list of penalized atom pairs `(j, k, w)` with
    /// `j < k` and the symmetrized per-pair weight `w` (exactly the value the old
    /// dense `pair_weight(j, k)` returned), avoiding any dense `K×K` allocation.
    /// `w == 0` pairs and out-of-range indices are dropped / rejected. This is
    /// equivalent to [`Self::new`] fed the dense symmetric matrix with the same
    /// nonzero entries.
    #[must_use = "build error must be handled"]
    pub fn new_sparse(
        target: PsiSlice,
        block_sizes: Vec<usize>,
        p_out: usize,
        pairs: Vec<(usize, usize, f64)>,
        weight: f64,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if target.is_empty() {
            return Err(
                "DecoderIncoherencePenalty::new_sparse requires a non-empty target".to_string(),
            );
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "DecoderIncoherencePenalty::new_sparse requires finite weight > 0, got {weight}"
            ));
        }
        if p_out == 0 {
            return Err("DecoderIncoherencePenalty::new_sparse requires p_out > 0".to_string());
        }
        if block_sizes.len() < 2 {
            return Err(
                "DecoderIncoherencePenalty::new_sparse requires at least two atom blocks"
                    .to_string(),
            );
        }
        let k = block_sizes.len();
        let mut total = 0usize;
        for (atom_idx, &m) in block_sizes.iter().enumerate() {
            if m == 0 {
                return Err(format!(
                    "DecoderIncoherencePenalty::new_sparse block_sizes[{atom_idx}] must be > 0"
                ));
            }
            let span = m.checked_mul(p_out).ok_or_else(|| {
                "DecoderIncoherencePenalty::new_sparse block span overflows usize".to_string()
            })?;
            total = total.checked_add(span).ok_or_else(|| {
                "DecoderIncoherencePenalty::new_sparse total span overflows usize".to_string()
            })?;
        }
        if total != target.len() {
            return Err(format!(
                "DecoderIncoherencePenalty::new_sparse Σ_k M_k·p_out = {total} does not match target length {}",
                target.len()
            ));
        }
        let mut clean = Vec::with_capacity(pairs.len());
        for (j, kk, w) in pairs {
            if j >= k || kk >= k {
                return Err(format!(
                    "DecoderIncoherencePenalty::new_sparse pair ({j}, {kk}) out of range K={k}"
                ));
            }
            if j >= kk {
                return Err(format!(
                    "DecoderIncoherencePenalty::new_sparse requires j < k for each pair, got ({j}, {kk})"
                ));
            }
            if !(w.is_finite() && w >= 0.0) {
                return Err(format!(
                    "DecoderIncoherencePenalty::new_sparse requires finite non-negative pair weight, got {w}"
                ));
            }
            if w != 0.0 {
                clean.push((j, kk, w));
            }
        }
        Ok(Self {
            target,
            block_sizes,
            p_out,
            k_atoms: k,
            pairs: clean,
            weight,
            learnable_weight,
            rho_index: 0,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            resolve_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    /// Flat-β offset of atom `k`'s decoder block within the vector passed to
    /// this penalty. SAE decoder-incoherence wiring registers a zero-based
    /// target slice, so `target.range.start` is normally zero here.
    fn block_offsets(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.block_sizes.len());
        let mut cursor = self.target.range.start;
        for &m in &self.block_sizes {
            out.push(cursor);
            cursor += m * self.p_out;
        }
        out
    }

    /// Cross-Gram `C[a, b] = Σ_o B_j[a, o]·B_k[b, o]`, shape `(M_j, M_k)`.
    fn cross_gram(
        target: ArrayView1<'_, f64>,
        off_j: usize,
        m_j: usize,
        off_k: usize,
        m_k: usize,
        p_out: usize,
    ) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((m_j, m_k));
        for a in 0..m_j {
            for b in 0..m_k {
                let mut s = 0.0;
                for o in 0..p_out {
                    s += target[off_j + a * p_out + o] * target[off_k + b * p_out + o];
                }
                out[[a, b]] = s;
            }
        }
        out
    }

    /// Shared kernel for the two curvature operators. Accumulates, per penalized
    /// atom pair `(j, k)`, the Gauss-Newton term `W·Σ_b dC[a,b]·B_k[b,o]` (and
    /// its `_k` transpose) always, and the residual term `W·Σ_b C[a,b]·V_k[b,o]`
    /// (and `_k` transpose) only when `include_residual`. With the residual the
    /// result is the exact `∂²P·v` ([`AnalyticPenalty::hvp`]); without it the
    /// result is the PSD Gauss-Newton surrogate
    /// ([`AnalyticPenalty::psd_majorizer_hvp`]).
    fn hvp_impl(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
        include_residual: bool,
    ) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(target.len());
        if target.len() != self.target.len() {
            return out;
        }
        let offsets = self.block_offsets();
        let weight = self.resolved_weight(rho);
        let p_out = self.p_out;
        for &(j, k, w_sym) in &self.pairs {
            {
                let w_pair = w_sym * weight;
                if w_pair == 0.0 {
                    continue;
                }
                let off_j = offsets[j];
                let off_k = offsets[k];
                let m_j = self.block_sizes[j];
                let m_k = self.block_sizes[k];
                // Directional Gram derivative driving the Gauss-Newton term:
                //   dC[a, b] = Σ_o (Vj[a, o]·Bk[b, o] + Bj[a, o]·Vk[b, o]).
                let mut d_c = Array2::<f64>::zeros((m_j, m_k));
                for a in 0..m_j {
                    for b in 0..m_k {
                        let mut s = 0.0;
                        for o in 0..p_out {
                            s += v[off_j + a * p_out + o] * target[off_k + b * p_out + o]
                                + target[off_j + a * p_out + o] * v[off_k + b * p_out + o];
                        }
                        d_c[[a, b]] = s;
                    }
                }
                // Cross-Gram C[a, b] = Σ_o Bj[a, o]·Bk[b, o] feeds the residual
                // term; only materialized for the exact Hessian path.
                let c = if include_residual {
                    Some(Self::cross_gram(target, off_j, m_j, off_k, m_k, p_out))
                } else {
                    None
                };
                // out_j[a, o] += w · Σ_b ( dC[a, b]·Bk[b, o] + C[a, b]·Vk[b, o] )
                for a in 0..m_j {
                    for o in 0..p_out {
                        let mut s = 0.0;
                        for b in 0..m_k {
                            s += d_c[[a, b]] * target[off_k + b * p_out + o];
                            if let Some(c) = &c {
                                s += c[[a, b]] * v[off_k + b * p_out + o];
                            }
                        }
                        out[off_j + a * p_out + o] += w_pair * s;
                    }
                }
                // out_k[b, o] += w · Σ_a ( dC[a, b]·Bj[a, o] + C[a, b]·Vj[a, o] )
                for b in 0..m_k {
                    for o in 0..p_out {
                        let mut s = 0.0;
                        for a in 0..m_j {
                            s += d_c[[a, b]] * target[off_j + a * p_out + o];
                            if let Some(c) = &c {
                                s += c[[a, b]] * v[off_j + a * p_out + o];
                            }
                        }
                        out[off_k + b * p_out + o] += w_pair * s;
                    }
                }
            }
        }
        out
    }
}

impl AnalyticPenalty for DecoderIncoherencePenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Beta
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        if target.len() != self.target.len() {
            return 0.0;
        }
        let offsets = self.block_offsets();
        let mut acc = 0.0;
        for &(j, k, w_pair) in &self.pairs {
            {
                if w_pair == 0.0 {
                    continue;
                }
                let c = Self::cross_gram(
                    target,
                    offsets[j],
                    self.block_sizes[j],
                    offsets[k],
                    self.block_sizes[k],
                    self.p_out,
                );
                let mut frob_sq = 0.0;
                for &value in c.iter() {
                    frob_sq += value * value;
                }
                acc += w_pair * frob_sq;
            }
        }
        0.5 * self.resolved_weight(rho) * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut grad = Array1::<f64>::zeros(target.len());
        if target.len() != self.target.len() {
            return grad;
        }
        let offsets = self.block_offsets();
        let weight = self.resolved_weight(rho);
        for &(j, k, w_sym) in &self.pairs {
            {
                let w_pair = w_sym * weight;
                if w_pair == 0.0 {
                    continue;
                }
                let off_j = offsets[j];
                let off_k = offsets[k];
                let m_j = self.block_sizes[j];
                let m_k = self.block_sizes[k];
                let c = Self::cross_gram(target, off_j, m_j, off_k, m_k, self.p_out);
                // grad_j[a, o] += w · Σ_b C[a, b] · B_k[b, o]
                for a in 0..m_j {
                    for o in 0..self.p_out {
                        let mut s = 0.0;
                        for b in 0..m_k {
                            s += c[[a, b]] * target[off_k + b * self.p_out + o];
                        }
                        grad[off_j + a * self.p_out + o] += w_pair * s;
                    }
                }
                // grad_k[b, o] += w · Σ_a C[a, b] · B_j[a, o]
                for b in 0..m_k {
                    for o in 0..self.p_out {
                        let mut s = 0.0;
                        for a in 0..m_j {
                            s += c[[a, b]] * target[off_j + a * self.p_out + o];
                        }
                        grad[off_k + b * self.p_out + o] += w_pair * s;
                    }
                }
            }
        }
        grad
    }

    /// Exact Hessian-vector product `H v = (∂²P/∂target²) v`.
    ///
    /// `P = ½ w Σ_{j<k} w_{jk} ‖C_{jk}‖²_F` is biquadratic (quartic) in the
    /// decoder blocks, so the second derivative of the nonlinear-least-squares
    /// objective carries **two** pieces along a direction `V` (per pair, with
    /// `W = w·w_{jk}`):
    ///
    /// ```text
    ///   (H v)_j[a,o] = W [ Σ_b dC[a,b]·B_k[b,o]   +   Σ_b C[a,b]·V_k[b,o] ]
    /// ```
    ///
    /// the Gauss-Newton term `Σ dC·B` and the residual term `Σ C·V`, with
    /// `dC[a,b] = Σ_o (V_j[a,o]·B_k[b,o] + B_j[a,o]·V_k[b,o])` (and the symmetric
    /// `_k` block). The residual term is what makes the exact Hessian indefinite;
    /// the GN-only surrogate lives in [`Self::psd_majorizer_hvp`].
    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        self.hvp_impl(target, rho, v, /* include_residual = */ true)
    }

    /// PSD majorizer-vector product `B_GN(target; ρ) v` for the **nonconvex**
    /// decoder-incoherence penalty.
    ///
    /// Dropping the indefinite residual term `W·Σ C·V` from the exact
    /// [`Self::hvp`] leaves the Gauss-Newton block `W·Jᵀ(J v)` with
    /// `J = ∂vec(C)/∂vec(B)`. That block is PSD by construction — a sum of
    /// `W ≥ 0` (`weight > 0`, `coactivation ≥ 0`) times rank-structured Gram
    /// products `JᵀJ` — and coincides with the exact Hessian as the cross-Gram
    /// `C → 0`. The inner Newton / PIRLS curvature block must stay
    /// positive-definite, so the GN block is the correct operator here, mirroring
    /// the other nonconvex penalties (sparsity, JumpReLU, isometry) that override
    /// the majorizer rather than hand back the indefinite true Hessian.
    fn psd_majorizer_hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(
            target.len(),
            v.len(),
            "psd_majorizer_hvp dimension mismatch"
        );
        self.hvp_impl(target, rho, v, /* include_residual = */ false)
    }

    // `hessian_diag` is intentionally left at the trait default (returns `None`
    // for a non-empty target): the Hessian of the cross-Gram Frobenius objective
    // is dense, not diagonal, so curvature is supplied via the closed-form
    // `hvp` / `psd_majorizer_hvp` path above.

    impl_learnable_weight_grad_rho!();

    impl_learnable_weight_rho_count!();

    fn name(&self) -> &str {
        "decoder_incoherence"
    }

    impl_scalar_apply_schedule!(weight);
}

// ---------------------------------------------------------------------------
// Orthogonality penalty
// ---------------------------------------------------------------------------

/// Gauge-fixing penalty for latent-coordinate axes.
///
/// ARD alone is rotation-invariant — pair with Orthogonality to identify
/// intrinsic dim. This penalty locks a canonical orthonormal basis first;
/// ARD can then shrink axes after the rotation gauge has been identified.
#[derive(Debug, Clone)]
pub struct OrthogonalityPenalty {
    pub target: PsiSlice,
    pub latent_dim: usize,
    /// Base strength. If `learnable_weight` is true, the resolved strength is
    /// `weight * exp(rho[rho_index])`; otherwise it is fixed at `weight`.
    pub weight: f64,
    /// Effective observation count used to keep the Frobenius contribution on
    /// the same scale as per-axis latent priors.
    pub n_eff: usize,
    pub learnable_weight: bool,
    pub rho_index: usize,
    pub weight_schedule: Option<ScalarWeightSchedule>,
}

impl OrthogonalityPenalty {
    #[must_use = "build error must be handled"]
    pub fn new(
        target: PsiSlice,
        latent_dim: usize,
        weight: f64,
        n_eff: usize,
        learnable_weight: bool,
    ) -> Result<Self, String> {
        if latent_dim == 0 {
            return Err("OrthogonalityPenalty::new requires latent_dim > 0".to_string());
        }
        if !target.len().is_multiple_of(latent_dim) {
            return Err(format!(
                "OrthogonalityPenalty::new target length {} is not divisible by latent_dim {}",
                target.len(),
                latent_dim
            ));
        }
        let n_obs = target.len() / latent_dim;
        if n_obs < latent_dim {
            return Err(format!(
                "OrthogonalityPenalty::new requires n_obs >= latent_dim for a feasible \
                 Stiefel target, got n_obs {n_obs} and latent_dim {latent_dim}"
            ));
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "OrthogonalityPenalty::new requires finite weight > 0, got {weight}"
            ));
        }
        if n_eff == 0 {
            return Err("OrthogonalityPenalty::new requires n_eff > 0".to_string());
        }
        if n_eff != n_obs {
            return Err(format!(
                "OrthogonalityPenalty::new requires n_eff to match target rows, got \
                 n_eff {n_eff} and target rows {n_obs}"
            ));
        }
        Ok(Self {
            target,
            latent_dim,
            weight,
            n_eff,
            learnable_weight,
            rho_index: 0,
            weight_schedule: None,
        })
    }

    impl_with_weight_schedule!(weight);

    fn resolved_weight(&self, rho: ArrayView1<'_, f64>) -> f64 {
        if self.learnable_weight {
            resolve_learnable_weight(self.weight, rho[self.rho_index])
        } else {
            self.weight
        }
    }

    pub(crate) fn scale(&self, rho: ArrayView1<'_, f64>) -> f64 {
        self.resolved_weight(rho) / self.n_eff as f64
    }

    pub(crate) fn target_matrix<'a>(
        &self,
        target: ArrayView1<'a, f64>,
    ) -> Option<ArrayView2<'a, f64>> {
        let d = self.latent_dim;
        if !target.len().is_multiple_of(d) {
            assert_eq!(
                target.len() % d,
                0,
                "target length must be divisible by latent_dim"
            );
            return None;
        }
        let n_obs = target.len() / d;
        target.into_shape_with_order((n_obs, d)).ok()
    }

    pub(crate) fn gram_minus_identity(t: ArrayView2<'_, f64>) -> Array2<f64> {
        let n_obs = t.nrows();
        let d = t.ncols();
        let mut gram = Array2::<f64>::zeros((d, d));
        for a in 0..d {
            for b in 0..d {
                let mut s = 0.0;
                for n in 0..n_obs {
                    s += t[[n, a]] * t[[n, b]];
                }
                gram[[a, b]] = s;
            }
            gram[[a, a]] -= 1.0;
        }
        gram
    }

    fn flatten_matrix(m: &Array2<f64>) -> Array1<f64> {
        let n_obs = m.nrows();
        let d = m.ncols();
        let mut out = Array1::<f64>::zeros(n_obs * d);
        for n in 0..n_obs {
            for a in 0..d {
                out[n * d + a] = m[[n, a]];
            }
        }
        out
    }

    pub(crate) fn hvp_with_precomputed_m(
        &self,
        t: ArrayView2<'_, f64>,
        m: ArrayView2<'_, f64>,
        v: ArrayView2<'_, f64>,
        scale: f64,
    ) -> Array2<f64> {
        let n_obs = t.nrows();
        let d = t.ncols();
        assert_eq!(v.dim(), t.dim(), "hvp matrix dimension mismatch");
        assert_eq!(m.dim(), (d, d), "precomputed gram dimension mismatch");
        if v.dim() != t.dim() {
            return Array2::<f64>::zeros((n_obs, d));
        }

        let mut vt_t_plus_tt_v = Array2::<f64>::zeros((d, d));
        for c in 0..d {
            for b in 0..d {
                let mut s = 0.0;
                for n in 0..n_obs {
                    s += v[[n, c]] * t[[n, b]] + t[[n, c]] * v[[n, b]];
                }
                vt_t_plus_tt_v[[c, b]] = s;
            }
        }

        let mut out = Array2::<f64>::zeros((n_obs, d));
        for n in 0..n_obs {
            for b in 0..d {
                let mut va = 0.0;
                let mut tb = 0.0;
                for c in 0..d {
                    va += v[[n, c]] * m[[c, b]];
                    tb += t[[n, c]] * vt_t_plus_tt_v[[c, b]];
                }
                out[[n, b]] = 2.0 * scale * (va + tb);
            }
        }
        out
    }

    pub(crate) fn as_dense_with_precomputed_m(
        &self,
        t: ArrayView2<'_, f64>,
        m: ArrayView2<'_, f64>,
        scale: f64,
    ) -> Array2<f64> {
        let n_obs = t.nrows();
        let d = t.ncols();
        assert_eq!(m.dim(), (d, d), "precomputed gram dimension mismatch");
        if m.dim() != (d, d) {
            return Array2::<f64>::zeros((n_obs * d, n_obs * d));
        }

        let mut dense = Array2::<f64>::zeros((n_obs * d, n_obs * d));
        let factor = 2.0 * scale;
        for row1 in 0..n_obs {
            for row2 in 0..n_obs {
                let mut row_dot = 0.0;
                for axis in 0..d {
                    row_dot += t[[row1, axis]] * t[[row2, axis]];
                }
                for col1 in 0..d {
                    let i = row1 * d + col1;
                    for col2 in 0..d {
                        let j = row2 * d + col2;
                        let mut entry = t[[row1, col2]] * t[[row2, col1]];
                        if row1 == row2 {
                            entry += m[[col2, col1]];
                        }
                        if col1 == col2 {
                            entry += row_dot;
                        }
                        dense[[i, j]] = factor * entry;
                    }
                }
            }
        }
        dense
    }
}

impl AnalyticPenalty for OrthogonalityPenalty {
    fn tier(&self) -> PenaltyTier {
        PenaltyTier::Psi
    }

    fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
        let Some(t) = self.target_matrix(target) else {
            return 0.0;
        };
        let gram = Self::gram_minus_identity(t.view());
        let mut acc = 0.0;
        for &v in gram.iter() {
            acc += v * v;
        }
        0.5 * self.scale(rho) * acc
    }

    fn grad_target(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> Array1<f64> {
        // Matrix-calculus core:
        //   d/dT ½·scale·||TᵀT - I||²_F = 2·scale·T·(TᵀT - I),
        // because TᵀT - I is symmetric.
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let gram = Self::gram_minus_identity(t.view());
        let n_obs = t.nrows();
        let d = t.ncols();
        let factor = 2.0 * self.scale(rho);
        let mut grad = Array2::<f64>::zeros((n_obs, d));
        for n in 0..n_obs {
            for a in 0..d {
                let mut s = 0.0;
                for b in 0..d {
                    s += t[[n, b]] * gram[[b, a]];
                }
                grad[[n, a]] = factor * s;
            }
        }
        Self::flatten_matrix(&grad)
    }

    fn hvp(
        &self,
        target: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(target.len(), v.len(), "hvp dimension mismatch");
        if target.len() != v.len() {
            return Array1::<f64>::zeros(target.len());
        }
        let Some(t) = self.target_matrix(target) else {
            return Array1::<f64>::zeros(target.len());
        };
        let Some(v_mat) = self.target_matrix(v) else {
            return Array1::<f64>::zeros(target.len());
        };
        let m = Self::gram_minus_identity(t.view());
        let hv = self.hvp_with_precomputed_m(t.view(), m.view(), v_mat.view(), self.scale(rho));
        Self::flatten_matrix(&hv)
    }

    impl_learnable_weight_grad_rho!();

    impl_learnable_weight_rho_count!();

    fn name(&self) -> &str {
        "orthogonality"
    }

    impl_scalar_apply_schedule!(weight);
}

//! Matrix-free assembled outer-Hessian operator.
//!
//! The [`UnifiedHessianOperator`] precomputes per-coordinate mode
//! responses, drift tables, and pair-wise logdet-Hessian cross traces once,
//! then applies the `K × K` outer Hessian as an `O(K)`-build HVP without
//! materialising the dense `K × K` matrix — winning when dense `p × p` drift
//! storage and pairwise row assembly dominate (see `routing`). Includes the
//! supporting drift carriers (`StoredFirstDrift`, `WeightedHyperOperator`) and
//! the `build_outer_hessian_operator` constructor.

use super::*;
use crate::estimate::smooth_floor_dp;

pub(crate) struct StoredFirstDrift {
    pub(crate) dense: Option<Array2<f64>>,
    pub(crate) dense_rotated: Option<Array2<f64>>,
    pub(crate) operators: Vec<Arc<dyn HyperOperator>>,
}

impl StoredFirstDrift {
    pub(crate) fn from_parts(
        dense: Option<Array2<f64>>,
        dense_rotated: Option<Array2<f64>>,
        operators: Vec<Arc<dyn HyperOperator>>,
    ) -> Self {
        Self {
            dense,
            dense_rotated,
            operators,
        }
    }

    pub(crate) fn scaled_add_apply(
        &self,
        v: ArrayView1<'_, f64>,
        scale: f64,
        out: &mut Array1<f64>,
    ) {
        assert_eq!(v.len(), out.len());
        if scale == 0.0 {
            return;
        }
        if let Some(matrix) = self.dense.as_ref() {
            dense::matvec_scaled_add_into(matrix, v, scale, out.view_mut());
        }
        if !self.operators.is_empty() {
            for op in &self.operators {
                op.scaled_add_mul_vec(v, scale, out.view_mut());
            }
        }
    }

    pub(crate) fn apply_dot(&self, v: ArrayView1<'_, f64>, test: ArrayView1<'_, f64>) -> f64 {
        assert_eq!(v.len(), test.len());
        let mut total = 0.0;
        if let Some(matrix) = self.dense.as_ref() {
            total += dense::bilinear(matrix, v, test);
        }
        for op in &self.operators {
            total += op.bilinear_view(v, test);
        }
        total
    }
}

pub(crate) struct BorrowedStoredDriftOperator<'a> {
    pub(crate) drift: &'a StoredFirstDrift,
    pub(crate) dim_hint: usize,
}

impl HyperOperator for BorrowedStoredDriftOperator<'_> {
    fn dim(&self) -> usize {
        self.dim_hint
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v.view(), out.view_mut());
        out
    }

    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v, out.view_mut());
        out
    }

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        out.fill(0.0);
        if let Some(matrix) = self.drift.dense.as_ref() {
            dense::matvec_into(matrix, v, out.view_mut());
        }
        for op in &self.drift.operators {
            op.scaled_add_mul_vec(v, 1.0, out.view_mut());
        }
    }

    fn scaled_add_mul_vec(&self, v: ArrayView1<'_, f64>, scale: f64, out: ArrayViewMut1<'_, f64>) {
        if scale == 0.0 {
            return;
        }
        let mut out = out;
        if let Some(matrix) = self.drift.dense.as_ref() {
            dense::matvec_scaled_add_into(matrix, v, scale, out.view_mut());
        }
        for op in &self.drift.operators {
            op.scaled_add_mul_vec(v, scale, out.view_mut());
        }
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        self.drift.apply_dot(v.view(), u.view())
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        self.drift.apply_dot(v, u)
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = self
            .drift
            .dense
            .clone()
            .unwrap_or_else(|| Array2::<f64>::zeros((self.dim_hint, self.dim_hint)));
        for op in &self.drift.operators {
            out += &op.to_dense();
        }
        out
    }

    fn is_implicit(&self) -> bool {
        !self.drift.operators.is_empty()
    }
}

/// Linear combination of `HyperOperator` factors with explicit scalar
/// weights. Used to bundle a coord's per-mode drift operators (or any other
/// per-term linear combination) into a single matrix-free operator that
/// implements the same `HyperOperator` trait, so callers downstream do not
/// need to handle a vector of (weight, op) pairs themselves.
pub struct WeightedHyperOperator {
    pub(crate) terms: Vec<(f64, Arc<dyn HyperOperator>)>,
    pub(crate) dim_hint: usize,
}

impl HyperOperator for WeightedHyperOperator {
    fn dim(&self) -> usize {
        self.dim_hint
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v.view(), out.view_mut());
        out
    }

    fn as_any(&self) -> &(dyn std::any::Any + 'static) {
        self
    }

    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v, out.view_mut());
        out
    }

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        let mut nonzero_terms = self.terms.iter().filter(|(weight, _)| *weight != 0.0);
        if let Some((weight, op)) = nonzero_terms.next()
            && nonzero_terms.next().is_none()
        {
            op.mul_vec_into(v, out.view_mut());
            if *weight != 1.0 {
                out.mapv_inplace(|value| *weight * value);
            }
            return;
        }

        out.fill(0.0);
        for (weight, op) in &self.terms {
            if *weight != 0.0 {
                op.scaled_add_mul_vec(v, *weight, out.view_mut());
            }
        }
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        let mut nonzero_terms = self.terms.iter().filter(|(weight, _)| *weight != 0.0);
        if let Some((weight, op)) = nonzero_terms.next()
            && nonzero_terms.next().is_none()
        {
            op.mul_basis_columns_into(start, out.view_mut());
            if *weight != 1.0 {
                out.mapv_inplace(|value| *weight * value);
            }
            return;
        }

        out.fill(0.0);
        let mut work = Array2::<f64>::zeros((out.nrows(), out.ncols()));
        for (weight, op) in &self.terms {
            if *weight == 0.0 {
                continue;
            }
            op.mul_basis_columns_into(start, work.view_mut());
            out.scaled_add(*weight, &work);
        }
    }

    fn scaled_add_mul_vec(
        &self,
        v: ArrayView1<'_, f64>,
        scale: f64,
        mut out: ArrayViewMut1<'_, f64>,
    ) {
        if scale == 0.0 {
            return;
        }
        for (weight, op) in &self.terms {
            let combined = scale * *weight;
            if combined != 0.0 {
                op.scaled_add_mul_vec(v, combined, out.view_mut());
            }
        }
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        self.terms
            .iter()
            .filter(|(weight, _)| *weight != 0.0)
            .map(|(weight, op)| weight * op.bilinear(v, u))
            .sum()
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        self.terms
            .iter()
            .filter(|(weight, _)| *weight != 0.0)
            .map(|(weight, op)| weight * op.bilinear_view(v, u))
            .sum()
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        self.terms
            .iter()
            .filter(|(weight, _)| *weight != 0.0)
            .map(|(weight, op)| weight * op.trace_projected_factor(factor))
            .sum()
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        self.terms
            .iter()
            .filter(|(weight, _)| *weight != 0.0)
            .map(|(weight, op)| weight * op.trace_projected_factor_cached(factor, cache))
            .sum()
    }

    fn projected_matrix(&self, factor: &Array2<f64>) -> Array2<f64> {
        let rank = factor.ncols();
        let mut projected = Array2::<f64>::zeros((rank, rank));
        for (weight, op) in &self.terms {
            if *weight != 0.0 {
                projected.scaled_add(*weight, &op.projected_matrix(factor));
            }
        }
        projected
    }

    fn projected_matrix_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> Array2<f64> {
        let rank = factor.ncols();
        let mut projected = Array2::<f64>::zeros((rank, rank));
        for (weight, op) in &self.terms {
            if *weight != 0.0 {
                projected.scaled_add(*weight, &op.projected_matrix_cached(factor, cache));
            }
        }
        projected
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.dim_hint, self.dim_hint));
        for (weight, op) in &self.terms {
            if *weight != 0.0 {
                out.scaled_add(*weight, &op.to_dense());
            }
        }
        out
    }

    fn is_implicit(&self) -> bool {
        self.terms.iter().any(|(_, op)| op.is_implicit())
    }
}

/// Per-matvec contraction of the ψψ-block second-order hook (#740), with each
/// `D²_ψ H_L` drift already traced through the logdet kernel into `base_h2`.
/// Indexed by ψ output row `i = idx - k_rho`.
pub(crate) struct PsiContractedContrib {
    pub(crate) objective: Array1<f64>,
    pub(crate) score: Array2<f64>,
    pub(crate) ld_s: Array1<f64>,
    pub(crate) base_h2: Vec<f64>,
}

pub(crate) struct OuterHessianCoord {
    pub(crate) a: f64,
    pub(crate) g: Array1<f64>,
    pub(crate) v: Array1<f64>,
    pub(crate) total_drift: StoredFirstDrift,
    pub(crate) base_drift: StoredFirstDrift,
    pub(crate) ext_index: Option<usize>,
    pub(crate) b_depends_on_beta: bool,
}

pub(crate) struct UnifiedHessianOperator {
    pub(crate) hop: Arc<dyn HessianFactorization>,
    pub(crate) coords: Vec<OuterHessianCoord>,
    pub(crate) pair_a: Array2<f64>,
    pub(crate) pair_ld_s: Array2<f64>,
    pub(crate) g_dot_v: Array2<f64>,
    pub(crate) pair_g: Vec<Vec<Option<Array1<f64>>>>,
    pub(crate) base_h2: Array2<f64>,
    pub(crate) m_pair_trace: Array2<f64>,
    /// Precomputed pair-wise logdet-Hessian cross traces.
    /// `cross_trace[i, j] = tr(G_ε(H) Ḣ_i Ḣ_j)` decomposed across the
    /// dense and operator components of each coord's `total_drift`.
    /// Populated only when `incl_logdet_h`.  matvec recovers the alpha-combo
    /// trace as `cross_trace.row(idx).dot(alpha)`, replacing the per-HVP
    /// recomputation that previously rebuilt these traces every time the
    /// K×K outer Hessian was materialized via K matvecs.
    pub(crate) cross_trace: Option<Array2<f64>>,
    pub(crate) profiled_phi: f64,
    pub(crate) profiled_nu: f64,
    pub(crate) profiled_dp_cgrad: f64,
    pub(crate) profiled_dp_cgrad2: f64,
    pub(crate) is_profiled: bool,
    pub(crate) incl_logdet_h: bool,
    pub(crate) incl_logdet_s: bool,
    pub(crate) kernel: OuterHessianDerivativeKernel,
    pub(crate) subspace: Option<Arc<PenaltySubspaceTrace>>,
    pub(crate) adjoint_z_c: Option<Array1<f64>>,
    pub(crate) leverage: Option<Array1<f64>>,
    pub(crate) fourth_trace: Option<Array2<f64>>,
    pub(crate) callback_second_modes: Option<Vec<Array1<f64>>>,
    /// Number of ρ (penalty) coordinates; coords `k_rho..` are the ψ rows. Used
    /// only when `contracted_psi` is present, to map an output index to its ψ
    /// output row.
    pub(crate) k_rho: usize,
    /// Direction-contracted ψψ second-order hook (#740). When `Some`, the build
    /// SKIPPED the `K²` per-pair `ext_coord_pair_fn` ψψ assembly (the `pair_a` /
    /// `pair_ld_s` / `base_h2` ψψ-block entries and the ψψ `pair_g` are left
    /// zero / `None`), and `matvec`/`apply_into` apply this once per call to add
    /// the ψ-row ψψ contributions in a single family row pass. The ρρ and ρψ
    /// blocks remain in the precomputed tables (cheap), so this changes only the
    /// representation of the ψψ block, not the math.
    pub(crate) contracted_psi: Option<ContractedPsiSecondOrderFn>,
}

impl UnifiedHessianOperator {
    /// Exact implicit-function-theorem mode response of the inner coefficient
    /// solution along a θ-direction `α = (α_ρ, α_ψ)` (#740 primitive).
    ///
    /// At the inner optimum `g(β̂(θ), θ) = ∇_β F = 0`, differentiating gives
    /// `β̇_j = −H⁻¹ ∂g/∂θ_j` for each coordinate j; the response along a θ
    /// direction is the linear combination `β̇(α) = Σ_j α_j β̇_j`. Each
    /// per-coordinate `coord.v = H⁻¹ a_j` is precomputed exactly via the shared
    /// inner mode inverse `hop.solve_multi` (see `build_outer_hessian_operator`),
    /// so this combination is the EXACT directional `β̇(α)` with no
    /// finite-difference or low-rank approximation — the same object the profiled
    /// θ-HVP needs as `β̇ = −H⁻¹ ∂g/∂θ·v`.
    ///
    /// SIGN CONVENTION: every coordinate stores `coord.v = +K · rhs` (the
    /// positive-convention mode-response solve; see `respond_stack`), and the
    /// true first mode derivative is `β_j = −v_j` for BOTH ρ and ψ/ext
    /// coordinates — the dense materialization (`dense.rs`) uses
    /// `β = v.mapv(|x| −x)` uniformly with no `is_ext` branch. So the directional
    /// derivative is `β̇(α) = Σ_j α_j β_j = −Σ_j α_j v_j`, the SAME `−` for every
    /// coordinate type. (A previous revision applied `+` to ρ and `−` to ext: a
    /// self-consistent convention that cancelled on the pure-ρ and pure-ψ
    /// diagonals because the second-order callback `D²_β H[·,·]` is bilinear, but
    /// it flipped the sign of the CROSS ρ-ψ second-order term — the bug the
    /// `mixed-ρψ` arm of `profiled_theta_hvp_outer_hessian_matches_fd_of_gradient`
    /// catches and the pure arms miss.)
    ///
    /// This is the reusable matrix-free primitive an O(K)-build θ-HVP matvec is
    /// organized around (one IFT solve per applied direction instead of the K²
    /// coordinate-pair assembly). The current `matvec` still consumes the
    /// precomputed pair traces; this primitive is the exact directional β̇ those
    /// pair traces are a (K²-amortized) re-expression of, exposed so the
    /// pair-assembly precompute can be replaced incrementally without changing
    /// the IFT mathematics.
    pub(crate) fn theta_direction_mode_response(&self, alpha: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.hop.dim());
        for (j, coord) in self.coords.iter().enumerate() {
            if alpha[j] == 0.0 {
                continue;
            }
            // True first mode derivative `β_j = −v_j` for every coordinate type.
            out.scaled_add(-alpha[j], &coord.v);
        }
        out
    }

    pub(crate) fn pair_rhs_dot(&self, row: usize, col: usize, test: ArrayView1<'_, f64>) -> f64 {
        let row_coord = &self.coords[row];
        let col_coord = &self.coords[col];
        let pair_g_dot = self.pair_g[row][col]
            .as_ref()
            .map(|pair_g| pair_g.dot(&test))
            .unwrap_or(0.0);

        col_coord.total_drift.apply_dot(row_coord.v.view(), test)
            + row_coord.base_drift.apply_dot(col_coord.v.view(), test)
            - pair_g_dot
    }

    pub(crate) fn scaled_add_pair_rhs(
        &self,
        row: usize,
        col: usize,
        scale: f64,
        out: &mut Array1<f64>,
    ) {
        if scale == 0.0 {
            return;
        }
        let row_coord = &self.coords[row];
        let col_coord = &self.coords[col];
        col_coord
            .total_drift
            .scaled_add_apply(row_coord.v.view(), scale, out);
        row_coord
            .base_drift
            .scaled_add_apply(col_coord.v.view(), scale, out);
        if let Some(pair_g) = self.pair_g[row][col].as_ref() {
            out.scaled_add(-scale, pair_g);
        }
    }

    pub(crate) fn pair_rhs_combo(&self, idx: usize, alpha: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.hop.dim());
        for j in 0..alpha.len() {
            if alpha[j] != 0.0 {
                self.scaled_add_pair_rhs(idx, j, alpha[j], &mut out);
            }
        }
        out
    }

    pub(crate) fn scalar_correction_trace(
        &self,
        idx: usize,
        alpha: &Array1<f64>,
        v_i: &Array1<f64>,
        m_alpha: &Array1<f64>,
        psi_score_alpha: Option<&Array1<f64>>,
    ) -> Result<f64, String> {
        let OuterHessianDerivativeKernel::ScalarGlm {
            c_array,
            d_array,
            x,
        } = &self.kernel
        else {
            return Err(RemlError::InvalidKernelMode {
                reason: "scalar correction requested for non-scalar kernel".to_string(),
            }
            .into());
        };

        // Cheap adjoint shortcut: works for both full-Hessian and projected
        // (subspace) regimes because the build populates `leverage` with the
        // projected `h^{G,proj}` under subspace while `adjoint_z_c = H⁻¹ · v`
        // stays a full solve, and the identity
        // tr(Kernel · C[u]) = uᵀ Xᵀ(c ⊙ h^G) = rhsᵀ H⁻¹ Xᵀ(c ⊙ h^G) carries
        // through (the projection lives entirely in the leverage `h^G`).
        let z_c = self.adjoint_z_c.as_ref().ok_or_else(|| {
            "missing adjoint trace cache for scalar outer Hessian operator".to_string()
        })?;
        let ingredients = ScalarGlmIngredients {
            c_array,
            d_array: d_array.as_ref(),
            x,
        };
        let h_g = self.leverage.as_ref().ok_or_else(|| {
            "missing leverage cache for scalar outer Hessian operator".to_string()
        })?;
        let mut c_trace = 0.0;
        for (j, &alpha_j) in alpha.iter().enumerate() {
            if alpha_j == 0.0 {
                continue;
            }
            c_trace += alpha_j * self.pair_rhs_dot(idx, j, z_c.view());
        }
        // #740: `pair_rhs_dot` reads `pair_g[idx][j]` for the `−g_{ij}·z_c`
        // adjoint term, but the build SKIPPED the ψψ `pair_g` when the contracted
        // hook is installed. `pair_rhs_dot` therefore drops the `−Σ_j α_j
        // g_{ψ_i ψ_j}` ψψ contribution; the hook supplies it as `score.row(i)`,
        // so add the missing `−score·z_c` here (mirrors the `−score` rhs
        // injection on the Callback path in `outer_hessian_index_entry`).
        if let Some(score_alpha) = psi_score_alpha {
            c_trace -= score_alpha.dot(z_c);
        }
        let d_trace = if let Some(trace) = self.fourth_trace.as_ref() {
            let mut combo = 0.0;
            for (j, &alpha_j) in alpha.iter().enumerate() {
                if alpha_j != 0.0 {
                    combo += alpha_j * trace[[idx, j]];
                }
            }
            combo
        } else {
            compute_fourth_derivative_trace(&ingredients, v_i, m_alpha, h_g)?.unwrap_or(0.0)
        };
        Ok(c_trace + d_trace)
    }

    /// Callback second-order correction trace for one HVP output row.
    ///
    /// `term1 = D_β H[u]` where `u` is the SECOND mode response `β̈ = K · rhs`,
    /// and `term2 = D²_β H[β̇(α), β̇_idx]` — both arguments of the bilinear
    /// `second` are TRUE first mode derivatives `β̇ = −v`: `mode_response_alpha`
    /// is the directional `β̇(α)` and `second_v` is the per-row `β̇_idx`. This
    /// mirrors the dense path's `compute_d2h(−v_l, −v_k)` exactly.
    ///
    /// `u` is solved with the FULL inner inverse `hop.solve` even when the LAML
    /// logdet uses the projected penalty-subspace trace: `u = β̈` is a mode
    /// response (an IFT stationarity derivative living in the full β-space), so
    /// the IFT identity demands the full inverse. The penalty-subspace
    /// projection acts only on the TRACE contraction (`trace_operator` below) —
    /// the same `ThetaModeResponseKernel` principle that the FIRST mode response
    /// and `dense.rs` follow (see `penalty_coordinate.rs` `respond_one`,
    /// `objective.rs` IFT-solve note).
    pub(crate) fn callback_correction_trace(
        &self,
        rhs: &Array1<f64>,
        second_v: &Array1<f64>,
        mode_response_alpha: &Array1<f64>,
    ) -> Result<f64, String> {
        let OuterHessianDerivativeKernel::Callback { first, second } = &self.kernel else {
            return Err(RemlError::InvalidKernelMode {
                reason: "callback correction requested for non-callback kernel".to_string(),
            }
            .into());
        };
        let u = self.hop.solve(rhs);
        let Some(term1) = first(&u)? else {
            return Ok(0.0);
        };
        let Some(term2) = second(mode_response_alpha, second_v)? else {
            return Ok(0.0);
        };
        let combined = CompositeHyperOperator {
            dense: None,
            operators: vec![term1.into_operator(), term2.into_operator()],
            dim_hint: self.hop.dim(),
        };
        if let Some(subspace) = self.subspace.as_deref() {
            Ok(subspace.trace_operator(&combined))
        } else {
            Ok(self.hop.trace_logdet_operator(&combined))
        }
    }

    /// Per-call contraction of the ψψ-block second-order hook (#740).
    ///
    /// Calls `contracted_psi` once with the ψ slice of `alpha` and pre-traces
    /// each per-output-row `D²_ψ H_L[ψ_i, ψ(α)]` drift through the logdet kernel
    /// so `outer_hessian_index_entry` reads scalars. Returns `None` when no hook
    /// is installed (the ψψ block then lives entirely in the precomputed
    /// tables). The `score` rows are carried through unchanged for injection
    /// into the callback-correction rhs (they replace the ψψ `pair_g` the build
    /// skipped). Indexed by ψ output row `i = idx - k_rho`.
    pub(crate) fn psi_contracted_contrib(
        &self,
        alpha: &Array1<f64>,
    ) -> Result<Option<PsiContractedContrib>, String> {
        let Some(hook) = self.contracted_psi.as_ref() else {
            return Ok(None);
        };
        let alpha_psi: Vec<f64> = alpha.iter().skip(self.k_rho).copied().collect();
        if alpha_psi.iter().all(|weight| *weight == 0.0) {
            let psi_dim = self.coords.len().saturating_sub(self.k_rho);
            return Ok(Some(PsiContractedContrib {
                objective: Array1::<f64>::zeros(psi_dim),
                score: Array2::<f64>::zeros((psi_dim, self.hop.dim())),
                ld_s: Array1::<f64>::zeros(psi_dim),
                base_h2: vec![0.0; psi_dim],
            }));
        }
        let Some(contracted) = hook(&alpha_psi)? else {
            // The hook declined this direction (e.g. a σ-aux axis carried
            // weight): this operator must not have been built with a skipped
            // ψψ assembly, so a decline here is a contract violation.
            return Err(RemlError::InvalidKernelMode {
                reason: "contracted ψψ hook declined a direction after the outer-Hessian \
                         build skipped per-pair ψψ assembly; the build-time and apply-time \
                         hook availability disagree"
                    .to_string(),
            }
            .into());
        };
        let base_h2: Vec<f64> = contracted
            .hessian
            .iter()
            .map(|drift| match (self.subspace.as_deref(), drift) {
                (Some(kernel), DriftDerivResult::Dense(m)) => kernel.trace_projected_logdet(m),
                (Some(kernel), DriftDerivResult::Operator(op)) => {
                    kernel.trace_operator(op.as_ref())
                }
                (None, DriftDerivResult::Dense(m)) => self.hop.trace_logdet_gradient(m),
                (None, DriftDerivResult::Operator(op)) => {
                    self.hop.trace_logdet_operator(op.as_ref())
                }
            })
            .collect();
        Ok(Some(PsiContractedContrib {
            objective: contracted.objective,
            score: contracted.score,
            ld_s: contracted.ld_s,
            base_h2,
        }))
    }

    /// Per-coordinate outer-Hessian-row × `alpha` contraction shared by the
    /// `matvec` and zero-alloc `apply_into` paths. `a_alpha` and
    /// `correction_m_alpha` (the directional first mode derivative
    /// `β̇(α) = −Σ_j α_j v_j`) are the alpha-dependent quantities precomputed once
    /// per call by the caller. `psi_contrib` carries the per-call ψψ-block hook
    /// contraction (#740); `None` keeps the ψψ block in the precomputed tables.
    pub(crate) fn outer_hessian_index_entry(
        &self,
        idx: usize,
        alpha: &Array1<f64>,
        a_alpha: f64,
        correction_m_alpha: &Array1<f64>,
        psi_contrib: Option<&PsiContractedContrib>,
    ) -> Result<f64, String> {
        let coord = &self.coords[idx];
        // ψ output row index into the hook contraction (when this idx is a ψ
        // coordinate and the hook is active); `None` for ρ rows or no hook.
        let psi_row = psi_contrib
            .and_then(|contrib| (idx >= self.k_rho).then(|| (contrib, idx - self.k_rho)));
        let mut pair_a = self.pair_a.row(idx).dot(alpha);
        let mut pair_ld_s = self.pair_ld_s.row(idx).dot(alpha);
        let g_dot_v_alpha = self.g_dot_v.row(idx).dot(alpha);
        let mut base_h2 = self.base_h2.row(idx).dot(alpha);
        let m_terms = self.m_pair_trace.row(idx).dot(alpha);
        if let Some((contrib, i)) = psi_row {
            // The build skipped the ψψ-block entries of these tables; add the
            // hook's α-contraction of the ψψ block (likelihood + ψψ penalty).
            pair_a += contrib.objective[i];
            pair_ld_s += contrib.ld_s[i];
            base_h2 += contrib.base_h2[i];
        }

        let cross_trace = match self.cross_trace.as_ref() {
            Some(ct) => ct.row(idx).dot(alpha),
            None => 0.0,
        };

        let correction = if self.incl_logdet_h {
            match &self.kernel {
                OuterHessianDerivativeKernel::Gaussian => 0.0,
                OuterHessianDerivativeKernel::ScalarGlm { .. } => {
                    // For ψ rows with the contracted hook, supply the
                    // `−Σ_j α_j g_{ψ_i ψ_j}` (= score.row(i)) that the skipped ψψ
                    // `pair_g` no longer provides to the scalar adjoint trace.
                    let psi_score = psi_row.map(|(contrib, i)| contrib.score.row(i).to_owned());
                    self.scalar_correction_trace(
                        idx,
                        alpha,
                        &coord.v,
                        correction_m_alpha,
                        psi_score.as_ref(),
                    )?
                }
                OuterHessianDerivativeKernel::Callback { .. } => {
                    let second_v = &self
                        .callback_second_modes
                        .as_ref()
                        .expect("callback second modes")[idx];
                    let mut rhs = self.pair_rhs_combo(idx, alpha);
                    // The build skipped the ψψ `pair_g`; the callback-correction
                    // second mode-response rhs needs `−Σ_j α_j g_{ψ_i ψ_j}`,
                    // which the hook supplies as `score.row(i)`. Inject it so the
                    // rhs matches the dense path's `pair_rhs_combo` exactly.
                    if let Some((contrib, i)) = psi_row {
                        rhs.scaled_add(-1.0, &contrib.score.row(i));
                    }
                    self.callback_correction_trace(&rhs, second_v, correction_m_alpha)?
                }
            }
        } else {
            0.0
        };

        Ok(outer_hessian_entry(
            coord.a,
            a_alpha,
            g_dot_v_alpha,
            pair_a,
            cross_trace,
            base_h2 + m_terms + correction,
            pair_ld_s,
            self.profiled_phi,
            self.profiled_nu,
            self.profiled_dp_cgrad,
            self.profiled_dp_cgrad2,
            self.is_profiled,
            self.incl_logdet_h,
            self.incl_logdet_s,
        ))
    }
}

impl gam_problem::HessianOperator for UnifiedHessianOperator {
    fn dim(&self) -> usize {
        self.coords.len()
    }

    /// Zero-alloc override for the inner-CG hot path.
    ///
    /// Results are written directly into the caller-supplied `out` buffer via
    /// `par_iter_mut().enumerate()`.
    fn apply_into(
        &self,
        alpha: &ndarray::Array1<f64>,
        out: &mut ndarray::Array1<f64>,
    ) -> Result<(), opt::ObjectiveEvalError> {
        if alpha.len() != self.coords.len() {
            return Err(opt::ObjectiveEvalError::fatal(format!(
                "outer Hessian alpha length mismatch: got {}, expected {}",
                alpha.len(),
                self.coords.len()
            )));
        }
        if out.len() != self.coords.len() {
            return Err(opt::ObjectiveEvalError::fatal(format!(
                "outer Hessian apply_into output length mismatch: got {}, expected {}",
                out.len(),
                self.coords.len()
            )));
        }
        let mut a_alpha = 0.0;
        for (idx, coord) in self.coords.iter().enumerate() {
            if alpha[idx] != 0.0 {
                a_alpha += alpha[idx] * coord.a;
            }
        }
        let correction_m_alpha = self.theta_direction_mode_response(alpha);
        // #740: one ψψ-block hook contraction per HVP.
        let psi_contrib = self
            .psi_contracted_contrib(alpha)
            .map_err(opt::ObjectiveEvalError::fatal)?;
        let slice = out.as_slice_mut().ok_or_else(|| {
            opt::ObjectiveEvalError::fatal("outer Hessian apply_into: non-contiguous output buffer")
        })?;
        slice
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(idx, cell)| {
                *cell = self.outer_hessian_index_entry(
                    idx,
                    alpha,
                    a_alpha,
                    &correction_m_alpha,
                    psi_contrib.as_ref(),
                )?;
                Ok(())
            })
            .map_err(opt::ObjectiveEvalError::fatal)
    }
}

pub(crate) fn build_outer_hessian_operator(
    solution: &InnerSolution<'_>,
    lambdas: &[f64],
    effective_deriv: &dyn HessianDerivativeProvider,
    kernel: OuterHessianDerivativeKernel,
    precomputed_coord_vs: Option<&[Array1<f64>]>,
    precomputed_coord_corrections: Option<&[Option<DriftDerivResult>]>,
) -> Result<UnifiedHessianOperator, String> {
    let hop = Arc::clone(&solution.hessian_op);
    let k = lambdas.len();
    let ext_dim = solution.ext_coords.len();
    let total = k + ext_dim;
    let curvature_lambdas: Vec<f64> = lambdas
        .iter()
        .copied()
        .map(|lambda| rho_curvature_lambda(solution, lambda))
        .collect();

    let (incl_logdet_h, incl_logdet_s) = match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => (true, true),
        DispersionHandling::Fixed {
            include_logdet_h,
            include_logdet_s,
            ..
        } => (*include_logdet_h, *include_logdet_s),
    };

    let det2 = solution.penalty_logdet.second.as_ref().ok_or_else(|| {
        "Outer Hessian requested but penalty second derivatives not provided".to_string()
    })?;

    let (profiled_phi, profiled_nu, profiled_dp_cgrad, profiled_dp_cgrad2, is_profiled) =
        match &solution.dispersion {
            DispersionHandling::ProfiledGaussian => {
                let dp_raw = -2.0 * solution.log_likelihood + solution.penalty_quadratic;
                let (dp_c, dp_cgrad, dp_cgrad2) = smooth_floor_dp(dp_raw, solution.dp_floor_scale);
                let nu = (solution.n_observations as f64 - solution.nullspace_dim).max(DENOM_RIDGE);
                let phi_hat = dp_c / nu;
                (phi_hat, nu, dp_cgrad, dp_cgrad2, true)
            }
            _ => (1.0, 1.0, 1.0, 0.0, false),
        };

    let penalty_quad_atom =
        crate::estimate::reml::atoms::PenaltyQuadAtom::from_penalty_coords(
            lambdas,
            &solution.penalty_coords,
            &solution.beta,
        )?;
    let curvature_penalty_quad_atom =
        crate::estimate::reml::atoms::PenaltyQuadAtom::from_penalty_coords(
            &curvature_lambdas,
            &solution.penalty_coords,
            &solution.beta,
        )?;

    let rho_penalty_a_k_betas: Vec<Array1<f64>> = penalty_quad_atom.block_penalty_scores().to_vec();
    let rho_curvature_a_k_betas: Vec<Array1<f64>> =
        curvature_penalty_quad_atom.block_penalty_scores().to_vec();
    // Mode responses are fixed-β stationarity derivatives. The main
    // evaluator passes precomputed responses so gradient and Hessian share
    // the same solve kernel; when none are provided this standalone path
    // routes through the SAME `ThetaModeResponseKernel::select` decision as
    // the gradient site and the dense Hessian (#931 pass 2) — pre-port it
    // hand-copied the selection rule and a comment warned it to "mirror the
    // dense evaluator's selection exactly, otherwise the operator-form
    // Hessian and dense materialization disagree on every entry whose row
    // or column lives outside `range(U_S)`" (the
    // `projected_operator_hessian_matches_dense_subspace_trace` regression).
    // That mirroring obligation is now structural: one constructor, no copy
    // to drift.
    let subspace = solution.penalty_subspace_trace.as_deref();
    let coord_vs_storage;
    let coord_vs: &[Array1<f64>] = if let Some(precomputed) = precomputed_coord_vs {
        if precomputed.len() != total {
            return Err(RemlError::DimensionMismatch {
                reason: format!(
                    "outer Hessian precomputed mode-response count mismatch: got {}, expected {}",
                    precomputed.len(),
                    total
                ),
            }
            .into());
        }
        precomputed
    } else {
        let owned = if total == 0 {
            Vec::new()
        } else {
            let mode_kernel = ThetaModeResponseKernel::select(
                subspace,
                solution.active_constraints.as_deref(),
                &*hop,
            );
            let mut rhs_stack = Array2::<f64>::zeros((hop.dim(), total));
            for idx in 0..k {
                rhs_stack
                    .column_mut(idx)
                    .assign(&rho_curvature_a_k_betas[idx]);
            }
            for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
                rhs_stack.column_mut(k + ext_idx).assign(&coord.g);
            }
            let solved_stack = mode_kernel.respond_stack(&rhs_stack);
            (0..total)
                .map(|idx| solved_stack.column(idx).to_owned())
                .collect::<Vec<_>>()
        };
        coord_vs_storage = owned;
        &coord_vs_storage
    };
    for (coord_idx, response) in coord_vs.iter().enumerate() {
        if let Some((entry_idx, value)) = response
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(RemlError::NonFiniteValue {
                reason: format!(
                    "outer Hessian mode response contains non-finite entry: \
                     coord={coord_idx} entry={entry_idx} value={value}"
                ),
            }
            .into());
        }
    }

    let coord_corrections_storage;
    let coord_corrections: &[Option<DriftDerivResult>] = if let Some(precomputed) =
        precomputed_coord_corrections
    {
        if precomputed.len() != total {
            return Err(RemlError::DimensionMismatch {
                reason: format!(
                    "outer Hessian precomputed correction count mismatch: got {}, expected {}",
                    precomputed.len(),
                    total
                ),
            }
            .into());
        }
        precomputed
    } else if effective_deriv.has_corrections() {
        if effective_deriv.has_batched_hessian_derivative_corrections() {
            log::info!(
                "[STAGE] outer_hessian coord_corrections mode=batched k={} ext_dim={} n={} dim={}",
                k,
                ext_dim,
                solution.n_observations,
                hop.dim()
            );
            coord_corrections_storage =
                effective_deriv.hessian_derivative_corrections_result(coord_vs)?;
        } else {
            coord_corrections_storage = coord_vs
                .par_iter()
                .map(|v_i| effective_deriv.hessian_derivative_correction_result(v_i))
                .collect::<Result<Vec<_>, _>>()?;
        }
        &coord_corrections_storage
    } else {
        coord_corrections_storage = (0..total).map(|_| None).collect::<Vec<_>>();
        &coord_corrections_storage
    };

    let mut coords = Vec::with_capacity(total);
    for idx in 0..k {
        let coord = &solution.penalty_coords[idx];
        let curvature_a_k_beta = rho_curvature_a_k_betas[idx].clone();
        let v_k = coord_vs[idx].clone();
        let correction = coord_corrections[idx].as_ref();
        let mut total_dense = None;
        let mut total_operators = Vec::new();
        match penalty_total_drift_result(coord, curvature_lambdas[idx], correction) {
            DriftDerivResult::Dense(matrix) => total_dense = Some(matrix),
            DriftDerivResult::Operator(op) => total_operators.push(op),
        }
        let mut base_dense = None;
        let mut base_operators = Vec::new();
        match penalty_total_drift_result(coord, curvature_lambdas[idx], None) {
            DriftDerivResult::Dense(matrix) => base_dense = Some(matrix),
            DriftDerivResult::Operator(op) => base_operators.push(op),
        }
        let dense_rotated = match (hop.as_dense_spectral(), total_dense.as_ref()) {
            (Some(dense_hop), Some(matrix)) => Some(dense_hop.rotate_to_eigenbasis(matrix)),
            _ => None,
        };
        let a_i = penalty_quad_atom.rho_frozen_d1(idx);
        coords.push(OuterHessianCoord {
            a: a_i,
            g: curvature_a_k_beta,
            v: v_k,
            total_drift: StoredFirstDrift::from_parts(total_dense, dense_rotated, total_operators),
            base_drift: StoredFirstDrift::from_parts(base_dense, None, base_operators),
            ext_index: None,
            b_depends_on_beta: false,
        });
    }

    for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
        let coord_idx = k + ext_idx;
        let v_i = coord_vs[coord_idx].clone();
        let correction = coord_corrections[coord_idx].as_ref();
        let (total_dense, total_operators) =
            hyper_coord_total_drift_parts(&coord.drift, correction);
        let (base_dense, base_operators) = hyper_coord_total_drift_parts(&coord.drift, None);
        let dense_rotated = match (hop.as_dense_spectral(), total_dense.as_ref()) {
            (Some(dense_hop), Some(matrix)) => Some(dense_hop.rotate_to_eigenbasis(matrix)),
            _ => None,
        };
        coords.push(OuterHessianCoord {
            a: coord.a,
            g: coord.g.clone(),
            v: v_i,
            total_drift: StoredFirstDrift::from_parts(total_dense, dense_rotated, total_operators),
            base_drift: StoredFirstDrift::from_parts(base_dense, None, base_operators),
            ext_index: Some(ext_idx),
            b_depends_on_beta: coord.b_depends_on_beta,
        });
    }

    let mut pair_a = Array2::<f64>::zeros((total, total));
    let mut pair_ld_s = Array2::<f64>::zeros((total, total));
    let mut g_dot_v = Array2::<f64>::zeros((total, total));
    let mut pair_g = vec![vec![None; total]; total];
    let mut base_h2 = Array2::<f64>::zeros((total, total));
    let mut m_pair_trace = Array2::<f64>::zeros((total, total));

    for ii in 0..total {
        for jj in ii..total {
            let value = match (coords[ii].ext_index, coords[jj].ext_index) {
                (None, None) => {
                    let rho_j = jj;
                    rho_penalty_a_k_betas[rho_j].dot(&coords[ii].v)
                }
                (None, Some(_)) => {
                    let rho_i = ii;
                    rho_penalty_a_k_betas[rho_i].dot(&coords[jj].v)
                }
                (Some(_), None) => {
                    let rho_j = jj;
                    rho_penalty_a_k_betas[rho_j].dot(&coords[ii].v)
                }
                (Some(_), Some(_)) => coords[ii].g.dot(&coords[jj].v),
            };
            g_dot_v[[ii, jj]] = value;
            g_dot_v[[jj, ii]] = value;
        }
    }

    for ii in 0..k {
        for jj in ii..k {
            pair_ld_s[[ii, jj]] = det2[[ii, jj]];
            if ii != jj {
                pair_ld_s[[jj, ii]] = det2[[ii, jj]];
            }
        }
    }

    for idx in 0..k {
        pair_a[[idx, idx]] = coords[idx].a;
        pair_g[idx][idx] = Some(coords[idx].g.clone());
        let base = if let Some(kernel) = subspace {
            let a_k = solution.penalty_coords[idx].scaled_dense_matrix(curvature_lambdas[idx]);
            kernel.trace_projected_logdet(&a_k)
        } else if solution.penalty_coords[idx].is_block_local() {
            let (block, start, end) = solution.penalty_coords[idx].scaled_block_local(1.0);
            hop.trace_logdet_block_local(&block, curvature_lambdas[idx], start, end)
        } else {
            let a_k = solution.penalty_coords[idx].scaled_dense_matrix(curvature_lambdas[idx]);
            hop.trace_logdet_gradient(&a_k)
        };
        base_h2[[idx, idx]] = base;
    }

    if let Some(rho_ext_fn) = solution.rho_ext_pair_fn.as_ref() {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let pair_count = k * ext_dim;
        let entries: Vec<(usize, usize, HyperCoordPair)> = (0..pair_count)
            .into_par_iter()
            .map(|pair_idx| {
                let rho_idx = pair_idx / ext_dim;
                let ext_idx = pair_idx % ext_dim;
                let pair = rho_ext_fn(rho_idx, ext_idx);
                (rho_idx, ext_idx, pair)
            })
            .collect();
        // Batch all second-drift traces so `--scale-dimensions` pays one
        // shared Hutchinson solve stream for the whole rho-ext block instead
        // of one estimator per pair.  Projected subspace traces skip the
        // stochastic shortcut inside `compute_base_h2_traces`.
        let pair_refs: Vec<&HyperCoordPair> = entries.iter().map(|(_, _, pair)| pair).collect();
        let bases = compute_base_h2_traces(
            hop.as_ref(),
            &pair_refs,
            subspace,
            Some(Arc::clone(&solution.stochastic_trace_state)),
        );
        for ((rho_idx, ext_idx, pair), base) in entries.into_iter().zip(bases.into_iter()) {
            let row = rho_idx;
            let col = k + ext_idx;
            pair_a[[row, col]] = pair.a;
            pair_a[[col, row]] = pair.a;
            pair_ld_s[[row, col]] = pair.ld_s;
            pair_ld_s[[col, row]] = pair.ld_s;
            pair_g[row][col] = Some(pair.g.clone());
            pair_g[col][row] = Some(pair.g);
            base_h2[[row, col]] = base;
            base_h2[[col, row]] = base;
        }
    }

    // #740: when the direction-contracted ψψ hook is installed, the ψψ-block
    // entries of pair_a / pair_ld_s / base_h2 and the ψψ pair_g are supplied
    // per-matvec by the hook in a single family row pass — so SKIP this `K²`
    // per-pair assembly entirely (each `ext_pair_fn(ii,jj)` is an O(n) family
    // row fold and `compute_base_h2_traces` then traces each at O(n·r)). The ρρ
    // and ρψ blocks above stay in the tables (cheap, no family row pass). The
    // ψψ entries left zero here are exactly the ones the hook adds in
    // `outer_hessian_index_entry`.
    if let (Some(ext_pair_fn), None) = (
        solution.ext_coord_pair_fn.as_ref(),
        solution.contracted_psi_second_order.as_ref(),
    ) {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let pair_count = ext_dim * (ext_dim + 1) / 2;
        let entries: Vec<(usize, usize, HyperCoordPair)> = (0..pair_count)
            .into_par_iter()
            .map(|pair_idx| {
                let (ii, jj) = upper_triangle_pair_from_index(pair_idx, ext_dim);
                let pair = ext_pair_fn(ii, jj);
                (ii, jj, pair)
            })
            .collect();
        let pair_refs: Vec<&HyperCoordPair> = entries.iter().map(|(_, _, pair)| pair).collect();
        let bases = compute_base_h2_traces(
            hop.as_ref(),
            &pair_refs,
            subspace,
            Some(Arc::clone(&solution.stochastic_trace_state)),
        );
        for ((ii, jj, pair), base) in entries.into_iter().zip(bases.into_iter()) {
            let row = k + ii;
            let col = k + jj;
            pair_a[[row, col]] = pair.a;
            pair_a[[col, row]] = pair.a;
            pair_ld_s[[row, col]] = pair.ld_s;
            pair_ld_s[[col, row]] = pair.ld_s;
            let g_pair = pair.g.clone();
            pair_g[row][col] = Some(g_pair.clone());
            pair_g[col][row] = Some(g_pair);
            base_h2[[row, col]] = base;
            base_h2[[col, row]] = base;
        }
    }

    {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let pair_count = total * (total + 1) / 2;
        let pair_drifts: Vec<((usize, usize), Vec<DriftDerivResult>)> = (0..pair_count)
            .into_par_iter()
            .map(|pair_idx| {
                let (ii, jj) = upper_triangle_pair_from_index(pair_idx, total);
                let beta_i = coords[ii].v.mapv(|value| -value);
                let beta_j = coords[jj].v.mapv(|value| -value);
                let mut drifts = Vec::new();
                if let Some(drift_fn) = solution.fixed_drift_deriv.as_ref() {
                    if coords[ii].b_depends_on_beta
                        && let Some(ext_i) = coords[ii].ext_index
                        && let Some(result) = drift_fn(ext_i, &beta_j)
                    {
                        drifts.push(result);
                    }
                    if coords[jj].b_depends_on_beta
                        && let Some(ext_j) = coords[jj].ext_index
                        && let Some(result) = drift_fn(ext_j, &beta_i)
                    {
                        drifts.push(result);
                    }
                }
                ((ii, jj), drifts)
            })
            .collect();

        let mut term_pairs = Vec::new();
        let mut term_drifts = Vec::new();
        for ((ii, jj), drifts) in pair_drifts {
            for drift in drifts {
                term_pairs.push((ii, jj));
                term_drifts.push(drift);
            }
        }

        if !term_drifts.is_empty() {
            let term_traces = if let Some(kernel) = subspace {
                penalty_subspace_trace_drifts_batched(kernel, &term_drifts)
            } else if let Some(ds) = hop.as_exact_dense_spectral() {
                dense_spectral_trace_logdet_drifts_batched(ds, &term_drifts)
            } else {
                term_drifts
                    .iter()
                    .map(|drift| drift.trace_logdet(hop.as_ref()))
                    .collect()
            };
            for ((ii, jj), trace) in term_pairs.into_iter().zip(term_traces.into_iter()) {
                m_pair_trace[[ii, jj]] += trace;
                if ii != jj {
                    m_pair_trace[[jj, ii]] += trace;
                }
            }
        }
    }

    // Precompute pair-wise logdet-Hessian cross traces:
    //   cross_trace[i, j] = tr(G_ε(H) Ḣ_i Ḣ_j)
    // Each coord's total Hessian drift Ḣ decomposes into a dense block plus
    // operator terms; the bilinear form expands across all four
    // dense-dense / dense-op / op-dense / op-op cross combinations.  By
    // bilinearity of `tr(G_ε(H) · · )` in the second factor, the full
    // alpha-combo cross trace recovered in matvec via
    //   cross_trace.row(i).dot(alpha)
    // matches the previous on-the-fly recomputation that built `alpha_dense`,
    // `alpha_dense_rotated`, and `alpha_op` at every HVP.
    let cross_trace: Option<Array2<f64>> = if incl_logdet_h {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let dense_hop_opt = hop.as_dense_spectral();
        if let Some(kernel) = subspace {
            let drift_parts = coords
                .iter()
                .map(|coord| {
                    let dense = coord.total_drift.dense.clone();
                    let op = if coord.total_drift.operators.is_empty() {
                        None
                    } else {
                        Some(Arc::new(CompositeHyperOperator {
                            dim_hint: hop.dim(),
                            dense: None,
                            operators: coord.total_drift.operators.clone(),
                        }) as Arc<dyn HyperOperator>)
                    };
                    match (dense, op) {
                        (Some(matrix), Some(operator)) => {
                            DriftDerivResult::Operator(Arc::new(CompositeHyperOperator {
                                dim_hint: hop.dim(),
                                dense: Some(matrix),
                                operators: vec![operator],
                            }))
                        }
                        (Some(matrix), None) => DriftDerivResult::Dense(matrix),
                        (None, Some(operator)) => DriftDerivResult::Operator(operator),
                        (None, None) => {
                            DriftDerivResult::Dense(Array2::zeros((hop.dim(), hop.dim())))
                        }
                    }
                })
                .collect::<Vec<_>>();
            let reduced = penalty_subspace_reduce_drifts_batched(kernel, &drift_parts);
            let pair_count = total * (total + 1) / 2;
            let pair_values: Vec<((usize, usize), f64)> = (0..pair_count)
                .into_par_iter()
                .map(|pair_idx| {
                    let (ii, jj) = upper_triangle_pair_from_index(pair_idx, total);
                    let value =
                        -kernel.trace_projected_logdet_cross_reduced(&reduced[ii], &reduced[jj]);
                    ((ii, jj), value)
                })
                .collect();
            let mut ct = Array2::<f64>::zeros((total, total));
            for ((ii, jj), value) in pair_values {
                if !value.is_finite() {
                    return Err(RemlError::NonFiniteValue {
                        reason: format!(
                            "outer Hessian operator projected cross_trace[{ii}, {jj}] is non-finite ({value})"
                        ),
                    }
                    .into());
                }
                ct[[ii, jj]] = value;
                if ii != jj {
                    ct[[jj, ii]] = value;
                }
            }
            Some(ct)
        } else if hop.prefers_stochastic_trace_estimation() && hop.logdet_traces_match_hinv_kernel()
        {
            // Matrix-free backends expose the SPD logdet kernel
            //   ∂² log|H|[A_i,A_j] = -tr(H⁻¹ A_i H⁻¹ A_j).
            //
            // Estimate the whole coordinate matrix in one Hutchinson batch
            // rather than launching one two-coordinate estimator per upper
            // triangle entry.  For `--scale-dimensions` with 16 ψ axes this
            // replaces 136 independent solve batches with one 16-coordinate
            // batch sharing the same probes and Krylov solves.
            let bundled: Vec<BorrowedStoredDriftOperator<'_>> = coords
                .iter()
                .map(|coord| BorrowedStoredDriftOperator {
                    drift: &coord.total_drift,
                    dim_hint: hop.dim(),
                })
                .collect();
            let op_refs: Vec<&dyn HyperOperator> =
                bundled.iter().map(|op| op as &dyn HyperOperator).collect();
            let estimator = StochasticTraceEstimator::for_outer_hessian_with_trace_state(
                hop.dim(),
                total,
                Arc::clone(&solution.stochastic_trace_state),
            );
            let no_dense: [&Array2<f64>; 0] = [];
            let mut ct = estimator.estimate_second_order_traces_with_operators(
                hop.as_ref(),
                &no_dense,
                &op_refs,
            );
            ct.mapv_inplace(|value| -value);
            Some(ct)
        } else if let Some(dense_hop) = dense_hop_opt {
            // Exact smooth-logdet Hessian kernel for operator-backed drifts.
            //
            // The second derivative of
            //     log |r_epsilon(H(theta))|
            // is not, in general,
            //     -tr(H_epsilon^{-1} H_i H_epsilon^{-1} H_j).
            // That identity only holds for the unregularized SPD logdet.
            // DenseSpectralOperator uses the divided-difference kernel of
            // log r_epsilon(sigma), so every dense/operator component must be
            // rotated into the eigenbasis and contracted with that same
            // kernel.  The dense Hessian assembly path already does this;
            // the matrix-free outer-Hv path must match it exactly.
            let mut rotated: Vec<Array2<f64>> =
                coords
                    .iter()
                    .map(|coord| {
                        coord.total_drift.dense_rotated.clone().unwrap_or_else(|| {
                            Array2::<f64>::zeros((dense_hop.n_dim, dense_hop.n_dim))
                        })
                    })
                    .collect();
            let mut terms: Vec<(usize, f64, &dyn HyperOperator)> = Vec::new();
            for (idx, coord) in coords.iter().enumerate() {
                for op in &coord.total_drift.operators {
                    collect_projected_matrix_terms(
                        idx,
                        1.0,
                        op.as_ref(),
                        &dense_hop.eigenvectors,
                        &mut rotated,
                        &mut terms,
                    );
                }
            }
            let projected_ops = project_hyper_operators_batched(
                total,
                &terms,
                &dense_hop.eigenvectors,
                &dense_hop.projected_factor_cache,
            );
            for (dst, projected) in rotated.iter_mut().zip(projected_ops.iter()) {
                *dst += projected;
            }

            let mut ct = Array2::<f64>::zeros((total, total));
            for ii in 0..total {
                for jj in ii..total {
                    let value =
                        dense_hop.trace_logdet_hessian_cross_rotated(&rotated[ii], &rotated[jj]);
                    if !value.is_finite() {
                        return Err(RemlError::NonFiniteValue {
                            reason: format!(
                                "outer Hessian operator cross_trace[{ii}, {jj}] is non-finite ({value})"
                            ),
                        }
                        .into());
                    }
                    ct[[ii, jj]] = value;
                    if ii != jj {
                        ct[[jj, ii]] = value;
                    }
                }
            }
            Some(ct)
        } else {
            // Enumerate the upper triangle (`ii ≤ jj`) so each `(ii, jj)` is an
            // independent unit of work — every entry of `cross_trace` is computed
            // from `coords[ii]` / `coords[jj]` only, with no shared mutable
            // state, so we can dispatch the K(K+1)/2 pair traces in parallel.
            let pair_count = total * (total + 1) / 2;
            let pair_values: Vec<((usize, usize), f64)> = (0..pair_count)
                .into_par_iter()
                .map(|pair_idx| {
                    let (ii, jj) = upper_triangle_pair_from_index(pair_idx, total);
                    let left = &coords[ii].total_drift;
                    let right = &coords[jj].total_drift;
                    let mut value = 0.0;
                    if let (Some(left_dense), Some(right_dense)) =
                        (left.dense.as_ref(), right.dense.as_ref())
                    {
                        if let (Some(dense_hop), Some(left_rot), Some(right_rot)) = (
                            dense_hop_opt,
                            left.dense_rotated.as_ref(),
                            right.dense_rotated.as_ref(),
                        ) {
                            value +=
                                dense_hop.trace_logdet_hessian_cross_rotated(left_rot, right_rot);
                        } else {
                            value += hop.trace_logdet_hessian_cross(left_dense, right_dense);
                        }
                    }
                    if let Some(left_dense) = left.dense.as_ref() {
                        for op in &right.operators {
                            value -= hop.trace_hinv_matrix_operator_cross(left_dense, op.as_ref());
                        }
                    }
                    if let Some(right_dense) = right.dense.as_ref() {
                        for op in &left.operators {
                            value -= hop.trace_hinv_matrix_operator_cross(right_dense, op.as_ref());
                        }
                    }
                    if !left.operators.is_empty() && !right.operators.is_empty() {
                        // Bundle each side's per-mode operators into a single
                        // weight-1 linear combination so the cross trace expands
                        // as `tr(H⁻¹ Â B̂) = Σ_a Σ_b tr(H⁻¹ A_a B_b)` with one
                        // call into the cross-trace kernel instead of the full
                        // O(|left.ops|·|right.ops|) sweep. Mathematically
                        // equivalent (bilinearity of `tr(H⁻¹ · ·)`).
                        let left_bundle = WeightedHyperOperator {
                            terms: left
                                .operators
                                .iter()
                                .map(|op| (1.0, Arc::clone(op)))
                                .collect(),
                            dim_hint: hop.dim(),
                        };
                        let right_bundle = WeightedHyperOperator {
                            terms: right
                                .operators
                                .iter()
                                .map(|op| (1.0, Arc::clone(op)))
                                .collect(),
                            dim_hint: hop.dim(),
                        };
                        value -= hop.trace_hinv_operator_cross(&left_bundle, &right_bundle);
                    }
                    ((ii, jj), value)
                })
                .collect();
            let mut ct = Array2::<f64>::zeros((total, total));
            for ((ii, jj), value) in pair_values {
                if !value.is_finite() {
                    return Err(RemlError::NonFiniteValue {
                        reason: format!(
                            "outer Hessian operator cross_trace[{ii}, {jj}] is non-finite ({value})"
                        ),
                    }
                    .into());
                }
                ct[[ii, jj]] = value;
                if ii != jj {
                    ct[[jj, ii]] = value;
                }
            }
            Some(ct)
        }
    } else {
        None
    };

    // Leverage and the scalar-GLM adjoint-z_c cache support both the
    // full-Hessian and projected-subspace paths.  The trace is
    //   tr(Kernel · C[u]) = Σ_r c_r (Xu)_r (X Kernel Xᵀ)_rr
    //                     = uᵀ · Xᵀ(c ⊙ h^G),   h^G = diag(X Kernel Xᵀ),
    // with the SECOND mode response `u = β̈ = H⁻¹ rhs` solved with the FULL
    // inner inverse in EVERY regime (it is an IFT stationarity derivative in
    // full β-space — see `compute_ift_correction_trace`'s slow path and
    // `compute_adjoint_z_c`).  Substituting,
    //   tr(Kernel · C[u]) = rhsᵀ H⁻¹ Xᵀ(c ⊙ h^G) = rhsᵀ · z_c,
    //   z_c = H⁻¹ · Xᵀ(c ⊙ h^G),
    // so `scalar_correction_trace` can take the cheap branch `rhs · z_c`
    // instead of materialising the second-derivative correction.  The
    // PROJECTION enters ONLY through the leverage: under subspace,
    //   h^{G,proj}_i = Xᵢᵀ · K · Xᵢ                  (K = U_S H_proj⁻¹ U_Sᵀ)
    // while the solve in `compute_adjoint_z_c` stays `H⁻¹`.  A previous version
    // ALSO projected the solve (`z_c = K · Xᵀ(c⊙h^{G,proj})`), making this path
    // compute `rhsᵀ K Xᵀ(c⊙h^{G,proj})` while the dense materialisation
    // correctly traces `tr(K · C[H⁻¹ rhs]) = rhsᵀ H⁻¹ Xᵀ(c⊙h^{G,proj})` — the
    // `projected_operator_hessian_matches_dense_subspace_trace` /
    // `outer_hessian_operator_matvec_matches_dense_subspace_with_null_alpha`
    // regressions.  Only the leverage is projected; the solve is not.
    let leverage = if incl_logdet_h {
        match &kernel {
            OuterHessianDerivativeKernel::Gaussian => None,
            OuterHessianDerivativeKernel::ScalarGlm { x, .. } => match subspace {
                Some(s) => Some(s.xt_projected_kernel_x_diagonal(x)),
                None => Some(hop.xt_logdet_kernel_x_diagonal(x)),
            },
            OuterHessianDerivativeKernel::Callback { .. } => None,
        }
    } else {
        None
    };
    let adjoint_z_c = if incl_logdet_h {
        match (&kernel, leverage.as_ref()) {
            (
                OuterHessianDerivativeKernel::ScalarGlm {
                    c_array,
                    d_array,
                    x,
                },
                Some(h_g),
            ) => Some(compute_adjoint_z_c(
                &ScalarGlmIngredients {
                    c_array,
                    d_array: d_array.as_ref(),
                    x,
                },
                hop.as_ref(),
                h_g,
            )?),
            _ => None,
        }
    } else {
        None
    };

    // Per-coordinate FIRST mode derivative `β_idx = −v_idx`, the second argument
    // of the bilinear `compute_d2h(β(α), β_idx)` callback correction. The true
    // mode derivative is `−v` for BOTH ρ and ψ/ext (matching `dense.rs`, which
    // uses `v.mapv(|x| −x)` with no `is_ext` branch); see the sign-convention
    // note on `theta_direction_mode_response`. Using `−v` uniformly here keeps
    // `compute_d2h`'s two arguments in the SAME true `β̇ = −v` convention as the
    // directional `theta_direction_mode_response`, so the cross ρ-ψ second-order
    // term carries the correct sign (the previous `is_ext` flip cancelled on the
    // pure-ρ/pure-ψ diagonals but flipped the cross block).
    let callback_second_modes = matches!(kernel, OuterHessianDerivativeKernel::Callback { .. })
        .then(|| coords.iter().map(|coord| -&coord.v).collect::<Vec<_>>());
    let fourth_trace = if incl_logdet_h && adjoint_z_c.is_some() {
        match (&kernel, leverage.as_ref()) {
            (
                OuterHessianDerivativeKernel::ScalarGlm {
                    c_array,
                    d_array: Some(d_array),
                    x,
                },
                Some(h_g),
            ) => {
                let modes = coords.iter().map(|coord| &coord.v).collect::<Vec<_>>();
                compute_fourth_derivative_trace_matrix(
                    &ScalarGlmIngredients {
                        c_array,
                        d_array: Some(d_array),
                        x,
                    },
                    &modes,
                    h_g,
                )?
            }
            _ => None,
        }
    } else {
        None
    };

    Ok(UnifiedHessianOperator {
        hop,
        coords,
        pair_a,
        pair_ld_s,
        g_dot_v,
        pair_g,
        base_h2,
        m_pair_trace,
        cross_trace,
        profiled_phi,
        profiled_nu,
        profiled_dp_cgrad,
        profiled_dp_cgrad2,
        is_profiled,
        incl_logdet_h,
        incl_logdet_s,
        kernel,
        subspace: solution.penalty_subspace_trace.clone(),
        adjoint_z_c,
        leverage,
        fourth_trace,
        callback_second_modes,
        k_rho: k,
        contracted_psi: solution.contracted_psi_second_order.clone(),
    })
}

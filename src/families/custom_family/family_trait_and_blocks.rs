pub use coefficient_groups::realize_coefficient_groups_for_custom_family;

use coefficient_groups::validate_penalized_complexity_prior;

use persistent_cache::{
    load_persistent_custom_family_warm_start, store_persistent_custom_family_warm_start,
    update_custom_outer_inner_cap_from_warm_start,
};


/// A penalty matrix that may be stored in Kronecker-factored form.
///
/// For tensor-product terms (e.g. time-varying survival covariates), the penalty
/// has the structure `S = left ⊗ right` (Kronecker product). Keeping this
/// factored avoids materializing (p_left × p_right)² dense entries and enables
/// exact log-determinant computation via `log|A ⊗ B| = n_B log|A| + n_A log|B|`.
///
/// Dense penalties are stored as-is.  Callers that need a raw `Array2<f64>` can
/// call `as_dense()` (zero-cost for Dense, lazy-materialized for KroneckerFactored).
#[derive(Clone, Debug)]
pub enum PenaltyMatrix {
    Dense(Array2<f64>),
    KroneckerFactored {
        left: Array2<f64>,
        right: Array2<f64>,
    },
    /// Block-local penalty: `local` is `block_dim × block_dim`, embedded at
    /// `col_range` in the full parameter space of dimension `total_dim`.
    /// Avoids materializing the full `total_dim × total_dim` matrix.
    Blockwise {
        local: Array2<f64>,
        col_range: std::ops::Range<usize>,
        total_dim: usize,
    },
    /// Wrapper assigning this penalty component to a user-visible precision
    /// label. Components with the same label share one smoothing parameter.
    Labeled {
        label: String,
        inner: Box<PenaltyMatrix>,
    },
    /// Wrapper fixing this penalty component at a physical log-precision.
    /// Fixed components remain in the block-local physical penalty layout but
    /// are removed from the REML outer coordinate vector.
    Fixed {
        log_lambda: f64,
        inner: Box<PenaltyMatrix>,
    },
}


impl PenaltyMatrix {
    /// Number of rows (= number of columns, since penalties are square).
    pub fn dim(&self) -> usize {
        match self {
            Self::Dense(m) => m.nrows(),
            Self::KroneckerFactored { left, right } => left.nrows() * right.nrows(),
            Self::Blockwise { total_dim, .. } => *total_dim,
            Self::Labeled { inner, .. } | Self::Fixed { inner, .. } => inner.dim(),
        }
    }

    /// Returns (nrows, ncols) like Array2::dim().
    pub fn shape(&self) -> (usize, usize) {
        let d = self.dim();
        (d, d)
    }

    /// Materialize the full dense matrix.
    pub fn to_dense(&self) -> Array2<f64> {
        match self {
            Self::Dense(m) => m.clone(),
            Self::KroneckerFactored { left, right } => {
                crate::terms::construction::kronecker_product(left, right)
            }
            Self::Blockwise {
                local,
                col_range,
                total_dim,
            } => {
                let mut g = Array2::zeros((*total_dim, *total_dim));
                g.slice_mut(ndarray::s![
                    col_range.start..col_range.end,
                    col_range.start..col_range.end
                ])
                .assign(local);
                g
            }
            Self::Labeled { inner, .. } | Self::Fixed { inner, .. } => inner.to_dense(),
        }
    }

    /// Borrow the inner dense matrix if Dense, otherwise materialize.
    pub fn as_dense_cow(&self) -> std::borrow::Cow<'_, Array2<f64>> {
        match self {
            Self::Dense(m) => std::borrow::Cow::Borrowed(m),
            Self::KroneckerFactored { .. }
            | Self::Blockwise { .. }
            | Self::Labeled { .. }
            | Self::Fixed { .. } => std::borrow::Cow::Owned(self.to_dense()),
        }
    }

    /// Returns a reference to the inner matrix if this is a Dense variant.
    pub fn as_dense_ref(&self) -> Option<&Array2<f64>> {
        match self {
            Self::Dense(m) => Some(m),
            Self::Fixed { inner, .. } => inner.as_dense_ref(),
            Self::KroneckerFactored { .. } | Self::Blockwise { .. } | Self::Labeled { .. } => None,
        }
    }

    pub fn with_precision_label(self, label: impl Into<String>) -> Self {
        Self::Labeled {
            label: label.into(),
            inner: Box::new(self),
        }
    }

    pub fn precision_label(&self) -> Option<&str> {
        match self {
            Self::Labeled { label, .. } => Some(label.as_str()),
            Self::Fixed { .. } => None,
            _ => None,
        }
    }

    pub fn with_fixed_log_lambda(self, log_lambda: f64) -> Self {
        Self::Fixed {
            log_lambda,
            inner: Box::new(self),
        }
    }

    pub fn fixed_log_lambda(&self) -> Option<f64> {
        match self {
            Self::Fixed { log_lambda, .. } => Some(*log_lambda),
            Self::Labeled { inner, .. } => inner.fixed_log_lambda(),
            _ => None,
        }
    }

    /// Compute S * v using the row-major Kronecker vec trick when factored:
    ///   (A ⊗ B) vec_rm(V) = vec_rm(A V Bᵀ)
    /// where V = reshape(v, (p_left, p_right)).
    pub fn dot(&self, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(m) => m.dot(v),
            Self::KroneckerFactored { left, right } => {
                let p_left = left.nrows();
                let p_right = right.nrows();
                // v is ordered by i_left * p_right + i_right.
                let v_mat =
                    ndarray::ArrayView2::from_shape((p_left, p_right), v.as_slice().unwrap())
                        .unwrap();
                let avbt = left.dot(&v_mat).dot(&right.t());
                let standard = avbt.as_standard_layout();
                Array1::from_iter(standard.iter().copied())
            }
            Self::Blockwise {
                local,
                col_range,
                total_dim,
            } => {
                let mut out = Array1::zeros(*total_dim);
                let v_block = v.slice(ndarray::s![col_range.clone()]);
                let result_block = local.dot(&v_block);
                out.slice_mut(ndarray::s![col_range.clone()])
                    .assign(&result_block);
                out
            }
            Self::Labeled { inner, .. } | Self::Fixed { inner, .. } => inner.dot(v),
        }
    }

    /// Add λ * self to a mutable dense accumulator.
    pub fn add_scaled_to(&self, lambda: f64, target: &mut Array2<f64>) {
        match self {
            Self::Dense(m) => {
                target.scaled_add(lambda, m);
            }
            Self::KroneckerFactored { left, right } => {
                let p_left = left.nrows();
                let p_right = right.nrows();
                for i1 in 0..p_left {
                    for j1 in 0..p_left {
                        let a_ij = left[[i1, j1]];
                        if a_ij == 0.0 {
                            continue;
                        }
                        let scaled_a = lambda * a_ij;
                        for i2 in 0..p_right {
                            let row = i1 * p_right + i2;
                            for j2 in 0..p_right {
                                let col = j1 * p_right + j2;
                                target[[row, col]] += scaled_a * right[[i2, j2]];
                            }
                        }
                    }
                }
            }
            Self::Blockwise {
                local, col_range, ..
            } => {
                target
                    .slice_mut(ndarray::s![col_range.clone(), col_range.clone()])
                    .scaled_add(lambda, local);
            }
            Self::Labeled { inner, .. } | Self::Fixed { inner, .. } => {
                inner.add_scaled_to(lambda, target)
            }
        }
    }

    /// Add λ * diag(self) to a mutable diagonal accumulator.
    pub fn add_scaled_diag_to(&self, lambda: f64, target: &mut Array1<f64>) {
        match self {
            Self::Dense(m) => {
                let p = m.nrows().min(target.len());
                for j in 0..p {
                    target[j] += lambda * m[[j, j]];
                }
            }
            Self::KroneckerFactored { left, right } => {
                let p_left = left.nrows();
                let p_right = right.nrows();
                assert_eq!(target.len(), p_left * p_right);
                for i_left in 0..p_left {
                    let left_diag = left[[i_left, i_left]];
                    if left_diag == 0.0 {
                        continue;
                    }
                    let scaled_left = lambda * left_diag;
                    for i_right in 0..p_right {
                        target[i_left * p_right + i_right] +=
                            scaled_left * right[[i_right, i_right]];
                    }
                }
            }
            Self::Blockwise {
                local, col_range, ..
            } => {
                let width = local.nrows().min(col_range.len());
                for local_idx in 0..width {
                    target[col_range.start + local_idx] += lambda * local[[local_idx, local_idx]];
                }
            }
            Self::Labeled { inner, .. } | Self::Fixed { inner, .. } => {
                inner.add_scaled_diag_to(lambda, target)
            }
        }
    }

    /// Compute the quadratic form β' S β.
    pub fn quadratic_form(&self, beta: &Array1<f64>) -> f64 {
        match self {
            Self::Dense(m) => beta.dot(&m.dot(beta)),
            Self::KroneckerFactored { .. } => {
                let sv = self.dot(beta);
                beta.dot(&sv)
            }
            Self::Blockwise {
                local, col_range, ..
            } => {
                let beta_block = beta.slice(ndarray::s![col_range.clone()]);
                let sv = local.dot(&beta_block);
                beta_block.dot(&sv)
            }
            Self::Labeled { inner, .. } | Self::Fixed { inner, .. } => inner.quadratic_form(beta),
        }
    }

    /// Access dimensions like an Array2.
    pub fn nrows(&self) -> usize {
        self.dim()
    }

    pub fn ncols(&self) -> usize {
        self.dim()
    }
}


impl From<Array2<f64>> for PenaltyMatrix {
    fn from(m: Array2<f64>) -> Self {
        Self::Dense(m)
    }
}


/// Per-subject channel Hessian provider for multi-output families.
///
/// The Fisher information decomposition for multi-output families is
///
/// ```text
/// I(β) = Σ_i  J_iᵀ W_i J_i
/// ```
///
/// where `J_i` is the channel-stacked Jacobian (shape `n_outputs × p` for
/// subject `i`) and `W_i` is the `n_outputs × n_outputs` per-subject channel
/// Hessian of the row negative log-likelihood (the second-derivative block of
/// `−log L_i(u_i)` at a pilot β, PSD-clamped).
///
/// For single-output families this is the scalar IRLS weight; for multi-output
/// families (survival marginal-slope: `n_outputs = 4`; location-scale:
/// `n_outputs = 2`) it carries full cross-channel curvature.
///
/// The identifiability canonicalisation step uses the `n_outputs`-channel
/// weighted joint design `W_joint = Σ_i sqrt(W_i) ⊗ J_i` to detect
/// block-against-block aliasing.  When this trait is present on
/// `ParameterBlockSpec::channel_hessian`, `canonicalize_for_identifiability`
/// routes through `audit_identifiability_channel_aware`; when absent it falls
/// back to the scalar-weight flat audit.
///
/// # W-metric rank theorem
///
/// The canonicalisation computes `rank(J^T W J)` where `W_blkdiag =
/// block-diagonal of per-subject W_i`.  This rank equals
///
/// ```text
/// rank(J) − dim(range(J) ∩ ker(W_blkdiag))
/// ```
///
/// i.e. columns of `J` that lie in the kernel of `W_blkdiag` (flat directions
/// in the curvature landscape at the pilot β) are correctly identified as
/// curvature-redundant and may be dropped.
pub trait FamilyChannelHessian: Send + Sync {
    /// Number of output channels `n_outputs` (= K in the row Jacobian).
    fn n_outputs(&self) -> usize;

    /// Number of subjects (rows).
    fn n_subjects(&self) -> usize;

    /// Fill the `n_outputs × n_outputs` per-subject channel Hessian `W_i`
    /// into `out` (row-major, length `n_outputs * n_outputs`) for subject `i`.
    /// Negative eigenvalues must be clamped to zero (PSD projection) before
    /// or inside this call.
    fn fill_subject(&self, i: usize, out: &mut [f64]);

    /// Materialise the full `(n_subjects × n_outputs × n_outputs)` tensor.
    /// Default implementation calls `fill_subject` for each row.
    fn evaluate_full(&self) -> ndarray::Array3<f64> {
        let n = self.n_subjects();
        let k = self.n_outputs();
        let mut out = ndarray::Array3::<f64>::zeros((n, k, k));
        let mut buf = vec![0.0_f64; k * k];
        for i in 0..n {
            self.fill_subject(i, &mut buf);
            for a in 0..k {
                for b in 0..k {
                    out[[i, a, b]] = buf[a * k + b];
                }
            }
        }
        out
    }

    /// Return a refreshed W evaluated at `beta` using `family_scalars` when
    /// those scalars carry the per-row primary state at the current β.
    ///
    /// # Fisher information identity
    ///
    /// `I(β) = J(β)^T W(β) J(β)`. T8 originally froze W at β=0; T34 refreshes
    /// both J and W at the current β so the audit's rank verdict reflects the
    /// actual local identifiability.
    ///
    /// # Default implementation (β-independent W)
    ///
    /// Families whose W is β-independent (e.g. Gaussian-identity where
    /// `W = prior_w`) return a clone of their frozen W by delegating to
    /// `evaluate_full()`. No recomputation is performed. `beta` and
    /// `family_scalars` are ignored.
    ///
    /// # Override (β-dependent W)
    ///
    /// Families with β-dependent W (e.g. survival marginal-slope where
    /// `W_i(β)` depends on `(q0_i, q1_i, qd1_i, g_i)`) must override this
    /// method and recompute W from the current primary state.
    ///
    /// When `beta` is non-zero in a way that affects W (i.e. `g_i != 0`),
    /// `family_scalars` MUST be `Some(..)`. Return `Err` if scalars are
    /// missing in that case (same error-message style as T26's contract).
    fn channel_hessian_at(
        &self,
        beta: &[f64],
        family_scalars: Option<&std::sync::Arc<dyn std::any::Any + Send + Sync>>,
    ) -> Result<Arc<dyn FamilyChannelHessian>, String> {
        // Default: W is β-independent — return a snapshot of the frozen W
        // wrapped in a simple tensor-backed implementation. β and
        // family_scalars are validated (NaN-guard, presence flag) so callers
        // that pass garbage state still see an Err rather than a silently-stale
        // W. The default impl does not require family_scalars; family-specific
        // overrides may.
        if beta.iter().any(|v| v.is_nan()) {
            return Err("channel_hessian_at: beta contains NaN".to_string());
        }
        // Acknowledge family_scalars without binding it to a discarded name.
        if family_scalars.is_some() && beta.is_empty() {
            return Err(
                "channel_hessian_at: family_scalars supplied but beta is empty".to_string(),
            );
        }
        let tensor = self.evaluate_full();
        Ok(Arc::new(TensorChannelHessian { h: tensor }))
    }
}


/// A [`FamilyChannelHessian`] backed directly by a pre-computed
/// `(n × K × K)` tensor. Used by the default `channel_hessian_at`
/// implementation and by tests.
///
/// This is the β-independent path: `fill_subject` reads from the frozen
/// tensor without any recomputation.
pub struct TensorChannelHessian {
    pub h: ndarray::Array3<f64>,
}


impl FamilyChannelHessian for TensorChannelHessian {
    fn n_outputs(&self) -> usize {
        self.h.shape()[1]
    }

    fn n_subjects(&self) -> usize {
        self.h.shape()[0]
    }

    fn fill_subject(&self, i: usize, out: &mut [f64]) {
        let k = self.h.shape()[1];
        assert_eq!(out.len(), k * k);
        for a in 0..k {
            for b in 0..k {
                out[a * k + b] = self.h[[i, a, b]];
            }
        }
    }

    fn evaluate_full(&self) -> ndarray::Array3<f64> {
        self.h.clone()
    }
}


/// β-linearization state passed to [`BlockEffectiveJacobian::effective_jacobian_at`].
///
/// At pre-fit initialization, pass `beta = &[]` / zeros and `family_scalars = None`.
/// Families that need β-dependent scalars (e.g. survival marginal-slope's q0, q1,
/// g, c, z) store them in `family_scalars` as a concrete type behind
/// `Arc<dyn Any + Send + Sync>` and downcast inside their impl.
pub struct FamilyLinearizationState<'a> {
    pub beta: &'a [f64],
    /// Optional family-shared scalars at this β linearization.
    /// Downcast via `state.family_scalars.as_ref().and_then(|a| a.downcast_ref::<T>())`.
    pub family_scalars: Option<Arc<dyn std::any::Any + Send + Sync>>,
    /// Optional per-subject channel Hessian for multi-output families.
    /// When `Some`, the identifiability canonicalisation step and the Gram
    /// builder use the channel-stacked Fisher information instead of the
    /// scalar-weight approximation.  Single-output families leave this `None`.
    pub channel_hessian: Option<Arc<dyn FamilyChannelHessian>>,
    /// Probit frailty scale factor `s_f = 1/√(1+σ²)`.
    ///
    /// For survival marginal-slope families the logslope η contribution is
    /// `s_f · g · z`, so any Jacobian callback that depends on g or z must
    /// read `s_f` from here rather than from a captured-at-construction value.
    /// When σ = 0 (no frailty) or for non-frailty families, set this to 1.0.
    ///
    /// Since σ is always **fixed** (not jointly optimised with β) in the
    /// survival family, `s_f` is a static scalar for the entire inner fit;
    /// `∂s_f/∂σ` never appears in the β-Jacobian.  The field is nonetheless
    /// carried through state so that Jacobian callbacks are not required to
    /// capture `s_f` at spec-construction time — they can read it at
    /// evaluation time and thus stay correct across outer-loop σ updates.
    pub probit_frailty_scale: f64,
}


/// β-dependent Jacobian callback for a parameter block.
///
/// Principled long-term contract for expressing how a block contributes to
/// the stacked linear predictor at a given β:
///
/// ```text
/// J(β) ∈ ℝ^{n_rows · n_outputs × p_block}
/// ```
///
/// - Single-output linear block: returns `design.clone()`.
/// - Row-scaled block (`RowScaledJacobian`): returns `diag(eta_scaling) · design` (still linear in β).
/// - Multi-output block (e.g. survival marginal-slope with η0, η1, ad1):
///   stacks `∂eta_r/∂β_k` for `r ∈ 0..n_outputs`, row-major ordering.
///
/// The default impl on [`ParameterBlockSpec::effective_jacobian_at`] is:
/// - `jacobian_callback = None` → `design.clone()`.
/// - `jacobian_callback = Some(cb)` → delegates to `cb.effective_jacobian_at`.
pub trait BlockEffectiveJacobian: Send + Sync {
    /// Stacked multi-output Jacobian for a contiguous observation row range.
    ///
    /// Shape: `(rows.len() * n_outputs, p_block)`, with the same channel-major
    /// layout as [`Self::effective_jacobian_at`]: row
    /// `channel * rows.len() + local_row` is `rows.start + local_row` in that
    /// output channel. Implementations should keep this as the single source of
    /// row math so large construction-time audits can stream chunks instead of
    /// materialising all `n * p * K` entries at once.
    fn effective_jacobian_rows(
        &self,
        state: &FamilyLinearizationState<'_>,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, String>;

    /// Stacked multi-output Jacobian at the current β.
    ///
    /// Shape: `(n_rows * n_outputs, p_block)`, **channel-major**: rows
    /// `r * n_rows .. (r + 1) * n_rows` carry output channel `r`'s row
    /// Jacobian, so `stacked[r * n_rows + i, j]` is observation `i`'s row at
    /// output `r` and coefficient column `j`.  Every consumer that destacks
    /// this matrix (audit, canonicaliser, fit) relies on this layout — see
    /// `BlockJacobianAsRowOp::from_callback` for the destacking transpose.
    /// For `n_outputs = 1` this is identical to the `(n_rows, p_block)` effective
    /// design used by the flat identifiability audit.
    fn effective_jacobian_at(
        &self,
        state: &FamilyLinearizationState<'_>,
    ) -> Result<Array2<f64>, String> {
        let full = self.effective_jacobian_rows(state, 0..usize::MAX)?;
        Ok(full)
    }

    /// Number of stacked output channels. 1 for most blocks.
    fn n_outputs(&self) -> usize {
        1
    }

    /// Returns the per-row scaling vector when this callback is a simple
    /// diagonal-scaling block (`RowScaledJacobian`).  Used by the
    /// identifiability audit's skewness-aware bias correction (T25).
    ///
    /// Returns `None` for all blocks except `RowScaledJacobian`.
    fn eta_row_scaling_for_skewness(&self) -> Option<Arc<[f64]>> {
        None
    }
}


/// A [`BlockEffectiveJacobian`] for any block that contributes linearly to
/// exactly one output of a multi-output family.
///
/// `own_output` is the zero-based output index that this block drives.
/// `n_family_outputs` is the total number of outputs (e.g. 2 for location-scale).
/// `design` is the block's effective design matrix (n × p_block).
///
/// The returned Jacobian has shape `(n_family_outputs * n, p_block)`:
/// rows `own_output * n .. (own_output + 1) * n` contain `design`,
/// all other rows are zero.
pub struct AdditiveBlockJacobian {
    pub design: Array2<f64>,
    pub own_output: usize,
    pub n_family_outputs: usize,
}


impl BlockEffectiveJacobian for AdditiveBlockJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &FamilyLinearizationState<'_>,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let n = self.design.nrows();
        let p = self.design.ncols();
        let rows = clamp_jacobian_rows(rows, n);
        // Additive (linear) block: Jacobian is β-independent — design does
        // not depend on state.beta. Verify beta contains no NaN when provided.
        if !state.beta.is_empty() && state.beta.iter().any(|v| v.is_nan()) {
            return Err(
                "AdditiveBlockJacobian::effective_jacobian_at: beta contains NaN".to_string(),
            );
        }
        let chunk = rows.end - rows.start;
        let total_rows = self.n_family_outputs * chunk;
        let mut jac = Array2::<f64>::zeros((total_rows, p));
        let row_start = self.own_output * chunk;
        jac.slice_mut(ndarray::s![row_start..row_start + chunk, ..])
            .assign(&self.design.slice(ndarray::s![rows.start..rows.end, ..]));
        Ok(jac)
    }

    fn n_outputs(&self) -> usize {
        self.n_family_outputs
    }
}


/// A [`BlockEffectiveJacobian`] for a single-output block whose contribution
/// to the linear predictor is `diag(eta_scaling) · design` (row-wise scaling).
///
/// This is the canonical replacement for the former `eta_row_scaling` field on
/// [`ParameterBlockSpec`].  The identifiability audit's skewness-aware bias
/// correction can recover the scaling vector via
/// [`BlockEffectiveJacobian::eta_row_scaling_for_skewness`].
pub struct RowScaledJacobian {
    pub design: Arc<Array2<f64>>,
    pub eta_scaling: Arc<[f64]>,
}


impl BlockEffectiveJacobian for RowScaledJacobian {
    fn effective_jacobian_rows(
        &self,
        state: &FamilyLinearizationState<'_>,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, String> {
        let n = self.design.nrows();
        let rows = clamp_jacobian_rows(rows, n);
        if self.eta_scaling.len() != n {
            return Err(format!(
                "RowScaledJacobian: eta_scaling length {} != design nrows {}",
                self.eta_scaling.len(),
                n,
            ));
        }
        // Row-scaled blocks are β-linear; verify the linearization point
        // contains no NaN when β is provided (sanity check on caller state).
        if !state.beta.is_empty() && state.beta.iter().any(|v| v.is_nan()) {
            return Err(
                "RowScaledJacobian::effective_jacobian_at: state.beta contains NaN".to_string(),
            );
        }
        let mut scaled = self
            .design
            .slice(ndarray::s![rows.start..rows.end, ..])
            .to_owned();
        for local_i in 0..scaled.nrows() {
            let s = self.eta_scaling[rows.start + local_i];
            for j in 0..scaled.ncols() {
                scaled[[local_i, j]] *= s;
            }
        }
        Ok(scaled)
    }

    fn eta_row_scaling_for_skewness(&self) -> Option<Arc<[f64]>> {
        Some(Arc::clone(&self.eta_scaling))
    }
}


fn clamp_jacobian_rows(rows: Range<usize>, n: usize) -> Range<usize> {
    let start = rows.start.min(n);
    let end = rows.end.min(n);
    start..end.max(start)
}


/// Static specification for one parameter block in a custom family.
///
/// `design` and `stacked_design` are two structurally distinct operators:
///
/// * `design` is the **canonical, single-channel, n-observation operator**.
///   `design.nrows()` ALWAYS equals `n_obs` (one row per training
///   observation).  This is the matrix the identifiability audit, the
///   shape policy, and every "what shape is this block?" reader inspect.
///   For most blocks `design` is also the eta-producing operator used by
///   the solver — see [`Self::solver_design`].
/// * `stacked_design`, when `Some`, is the **multi-channel eta-producing
///   operator** used by the solver.  Survival time-varying blocks stack
///   `[exit; entry; deriv]` into a `(3·n × p)` operator here so the
///   solver can produce a `3·n`-long `eta` in one mat-vec; the audit
///   never sees this matrix.  When `None`, the solver uses `design` (the
///   single-channel default).
///
/// The single contract that downstream code can rely on:
/// `design.nrows() == n_obs`.  No more dual semantics on `design`.
///
/// Read access:
/// * Audit / canonicalize / "n_obs is the row count" code → `&spec.design`.
/// * Eta-producing solver code → [`Self::solver_design`].
#[derive(Clone)]
pub struct ParameterBlockSpec {
    pub name: String,
    pub design: DesignMatrix,
    pub offset: Array1<f64>,
    /// Block-local penalty matrices (all p_block x p_block).
    pub penalties: Vec<PenaltyMatrix>,
    /// Structural nullspace dimension of each penalty matrix (same length as `penalties`).
    /// Used by the penalty pseudo-logdet to determine rank without numerical thresholds.
    /// If empty, falls back to eigenvalue-based rank detection.
    pub nullspace_dims: Vec<usize>,
    /// Initial log-smoothing parameters for this block (same length as `penalties`).
    pub initial_log_lambdas: Array1<f64>,
    /// Optional initial coefficients (defaults to zeros if omitted).
    pub initial_beta: Option<Array1<f64>>,
    /// Gauge ownership priority. Higher = more likely to retain a
    /// redundant direction during canonical-gauge reparameterisation.
    /// Defaults to 100. Set higher for blocks that should "own" shared
    /// affine/null-space directions (e.g. baseline time in survival).
    pub gauge_priority: u8,
    /// Full β-dependent Jacobian callback.  When `Some`, this is the
    /// authoritative source for `effective_jacobian_at`.  For simple
    /// single-output row-scaled blocks use [`RowScaledJacobian`].
    pub jacobian_callback: Option<Arc<dyn BlockEffectiveJacobian>>,
    /// Optional multi-channel eta-producing operator used by the solver.
    ///
    /// When `Some`, the solver consumes this matrix (typically
    /// `(k·n × p)` for `k` stacked channels — e.g. survival
    /// `[exit; entry; deriv]` with `k = 3`) to evaluate `eta = stacked · β + stacked_offset`.
    /// The audit and shape policy NEVER read this field; they only ever
    /// inspect `design` (which always has `n_obs` rows).
    ///
    /// When `None`, the solver falls back to `design` — the correct
    /// behavior for every single-channel block (i.e. all non-survival
    /// time-varying blocks).
    ///
    /// Read this field via [`Self::solver_design`], never directly.
    ///
    /// Invariant: when `stacked_design = Some(_)`, `stacked_offset` MUST
    /// also be `Some(_)` and its length MUST equal `stacked_design.nrows()`.
    pub stacked_design: Option<DesignMatrix>,
    /// Optional offset paired with [`Self::stacked_design`]. Same Option
    /// state as `stacked_design` (both `Some` or both `None`).
    /// Read via [`Self::solver_offset`].
    pub stacked_offset: Option<Array1<f64>>,
}


impl std::fmt::Debug for ParameterBlockSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParameterBlockSpec")
            .field("name", &self.name)
            .field("design", &self.design)
            .field("offset", &self.offset)
            .field("penalties", &self.penalties)
            .field("nullspace_dims", &self.nullspace_dims)
            .field("initial_log_lambdas", &self.initial_log_lambdas)
            .field("initial_beta", &self.initial_beta)
            .field("gauge_priority", &self.gauge_priority)
            .field(
                "jacobian_callback",
                &self
                    .jacobian_callback
                    .as_ref()
                    .map(|_| "<BlockEffectiveJacobian>"),
            )
            .finish()
    }
}


impl ParameterBlockSpec {
    /// Returns a ParameterBlockSpec with sensible defaults for all optional
    /// fields. Callers using struct literal syntax can use
    /// `..ParameterBlockSpec::defaults()` to fill in any fields added after
    /// the literal was written.
    pub fn defaults() -> Self {
        Self {
            name: String::new(),
            design: DesignMatrix::Dense(crate::linalg::matrix::DenseDesignMatrix::from(
                ndarray::Array2::<f64>::zeros((0, 0)),
            )),
            offset: ndarray::Array1::<f64>::zeros(0),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: ndarray::Array1::<f64>::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        }
    }

    /// Returns the eta-producing operator used by the solver.
    ///
    /// Resolution order:
    ///   1. `stacked_design = Some(d)` → return `d` (multi-channel
    ///      operator, e.g. `(3n × p)` for survival time-varying blocks).
    ///   2. otherwise → return `&self.design` (the single-channel default).
    ///
    /// Solver code that needs `eta = D · β` MUST call this accessor;
    /// reading `&self.design` directly silently breaks multi-channel
    /// (survival LS time-varying) blocks because `self.design.nrows()`
    /// always equals `n_obs`, never `3·n_obs`.
    pub fn solver_design(&self) -> &DesignMatrix {
        self.stacked_design.as_ref().unwrap_or(&self.design)
    }

    /// Returns the offset paired with [`Self::solver_design`]. When
    /// `stacked_offset = Some(o)` this returns `&o`; otherwise it falls
    /// back to `&self.offset`.
    pub fn solver_offset(&self) -> &Array1<f64> {
        self.stacked_offset.as_ref().unwrap_or(&self.offset)
    }

    /// Returns the effective design `D_eff` for this block at β = 0 with no
    /// family scalars — a convenience wrapper around [`Self::effective_jacobian_at`]
    /// for the single-output (n_outputs = 1) case.
    ///
    /// Callers that need multi-output Jacobians or β-dependent scalars should
    /// call `effective_jacobian_at` directly with the appropriate state.
    ///
    /// Returns `Err` if the design cannot be densified.
    pub fn effective_design(&self, caller: &str) -> Result<ndarray::Array2<f64>, String> {
        let p = self.design.ncols();
        let zeros = vec![0.0f64; p];
        let state = FamilyLinearizationState {
            beta: &zeros,
            family_scalars: None,
            channel_hessian: None,
            probit_frailty_scale: 1.0,
        };
        self.effective_jacobian_at(caller, &state)
    }

    /// Returns the β-dependent stacked Jacobian `J(β)` for this block.
    ///
    /// Shape: `(n_rows * n_outputs, p_block)`.  For most blocks `n_outputs = 1`
    /// and the result is the familiar `(n_rows, p_block)` effective design.
    ///
    /// Dispatch order:
    ///   1. `jacobian_callback = Some(cb)` → `cb.effective_jacobian_at(state)`.
    ///   2. `jacobian_callback = None` → `design.clone()` (ignores `beta` and `family_scalars`).
    ///
    /// Returns `Err` if the design cannot be densified.
    pub fn effective_jacobian_at(
        &self,
        caller: &str,
        state: &FamilyLinearizationState<'_>,
    ) -> Result<ndarray::Array2<f64>, String> {
        if let Some(cb) = self.jacobian_callback.as_ref() {
            return cb.effective_jacobian_at(state);
        }
        self.design
            .try_to_dense_arc(&format!(
                "{caller}::effective_jacobian_at block '{}'",
                self.name
            ))
            .map(|arc| arc.as_ref().clone())
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoefficientBlockSelector {
    Name(String),
    Index(usize),
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoefficientLabel {
    pub block: CoefficientBlockSelector,
    pub column: usize,
}


impl CoefficientLabel {
    pub fn by_block_name(block: impl Into<String>, column: usize) -> Self {
        Self {
            block: CoefficientBlockSelector::Name(block.into()),
            column,
        }
    }
}


pub fn coefficient_label(block: impl Into<String>, column: usize) -> CoefficientLabel {
    CoefficientLabel::by_block_name(block, column)
}


#[derive(Debug, Clone, PartialEq)]
pub enum CoefficientGroupPrior {
    Flat,
    NormalLogPrecision {
        mean: f64,
        sd: f64,
    },
    GammaPrecision {
        shape: f64,
        rate: f64,
    },
    /// Penalized-complexity prior calibrated by `P(exp(-ρ/2) > upper) =
    /// tail_prob`; see [`crate::types::RhoPrior::PenalizedComplexity`].
    PenalizedComplexity {
        upper: f64,
        tail_prob: f64,
    },
}


impl CoefficientGroupPrior {
    pub fn to_rho_prior(&self) -> crate::types::RhoPrior {
        match *self {
            Self::Flat => crate::types::RhoPrior::Flat,
            Self::NormalLogPrecision { mean, sd } => crate::types::RhoPrior::Normal { mean, sd },
            Self::GammaPrecision { shape, rate } => {
                crate::types::RhoPrior::GammaPrecision { shape, rate }
            }
            Self::PenalizedComplexity { upper, tail_prob } => {
                crate::types::RhoPrior::PenalizedComplexity { upper, tail_prob }
            }
        }
    }

    fn validate(&self, context: &str) -> Result<(), String> {
        match *self {
            Self::Flat => Ok(()),
            Self::NormalLogPrecision { mean, sd } => {
                if !mean.is_finite() {
                    return Err(format!(
                        "{context} Normal log-precision prior requires finite mean, got {mean}"
                    ));
                }
                if !sd.is_finite() || sd <= 0.0 {
                    return Err(format!(
                        "{context} Normal log-precision prior requires sd > 0, got {sd}"
                    ));
                }
                Ok(())
            }
            Self::PenalizedComplexity { upper, tail_prob } => {
                validate_penalized_complexity_prior(context, upper, tail_prob)
            }
            Self::GammaPrecision { shape, rate } => {
                if !shape.is_finite() || shape <= 0.0 {
                    return Err(CustomFamilyError::DimensionMismatch {
                        reason: format!(
                            "{context} Gamma precision prior requires shape > 0, got {shape}"
                        ),
                    }
                    .into());
                }
                if !rate.is_finite() || rate < 0.0 {
                    return Err(format!(
                        "{context} Gamma precision prior requires rate >= 0, got {rate}"
                    ));
                }
                Ok(())
            }
        }
    }
}


#[derive(Debug, Clone, PartialEq)]
pub struct CoefficientGroupSpec {
    pub label: String,
    pub coefficients: Vec<CoefficientLabel>,
    pub parent: Option<String>,
    pub prior: Option<CoefficientGroupPrior>,
    pub initial_log_precision: Option<f64>,
}


impl CoefficientGroupSpec {
    pub fn new(label: impl Into<String>, coefficients: Vec<CoefficientLabel>) -> Self {
        Self {
            label: label.into(),
            coefficients,
            parent: None,
            prior: None,
            initial_log_precision: None,
        }
    }

    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parent = Some(parent.into());
        self
    }

    pub fn with_prior(mut self, prior: CoefficientGroupPrior) -> Self {
        self.prior = Some(prior);
        self
    }
}


#[derive(Debug, Clone, PartialEq)]
pub struct RealizedCoefficientGroup {
    pub label: String,
    pub parent: Option<String>,
    pub coefficients: Vec<(usize, usize)>,
    pub prior: Option<CoefficientGroupPrior>,
    pub initial_log_precision: f64,
}


#[derive(Debug, Clone)]
pub struct RealizedCoefficientGroupSpecs {
    pub specs: Vec<ParameterBlockSpec>,
    pub groups: Vec<RealizedCoefficientGroup>,
    /// One entry per realized penalty in flattened block order. Built-in
    /// penalties receive unique internal labels; user groups carry their
    /// declared labels. Consumers that optimize one coordinate per label can
    /// use this to tie cross-block penalty pieces to a shared precision.
    pub penalty_labels: Vec<String>,
    /// Per-coordinate priors in `outer_labels` order.
    pub rho_prior: crate::types::RhoPrior,
    pub outer_labels: Vec<String>,
}


fn custom_family_block_role(
    name: &str,
    index: usize,
    n_blocks: usize,
) -> crate::solver::estimate::BlockRole {
    use crate::solver::estimate::BlockRole;

    if n_blocks == 1 {
        return BlockRole::Mean;
    }

    match name.trim().to_ascii_lowercase().as_str() {
        "eta" | "mean" | "beta" => BlockRole::Mean,
        "mu" | "location" | "marginal_surface" => BlockRole::Location,
        "threshold" => BlockRole::Threshold,
        "log_sigma" | "scale" | "logslope_surface" => BlockRole::Scale,
        "time" | "time_transform" | "time_surface" => BlockRole::Time,
        name if name.starts_with("time_cause_") => BlockRole::Time,
        "wiggle" | "linkwiggle" => BlockRole::LinkWiggle,
        _ if index == 0 => BlockRole::Location,
        _ => BlockRole::Scale,
    }
}


/// Current state for a parameter block.
#[derive(Clone, Debug)]
pub struct ParameterBlockState {
    pub beta: Array1<f64>,
    pub eta: Array1<f64>,
}


#[derive(Clone)]
pub struct BlockGeometryDirectionalDerivative {
    /// Directional derivative of the block design matrix along a coefficient-space direction.
    pub d_design: Option<Array2<f64>>,
    /// Directional derivative of the block offset along the same direction.
    pub d_offset: Array1<f64>,
}


/// Working quantities supplied by a custom family for one block.
///
/// # Observed vs expected information (see response.md Section 3)
///
/// For the outer REML/LAML criterion, the Hessian used in log|H| and trace terms
/// must be the **observed** (actual) Hessian at the mode, not the expected Fisher.
///
/// - `ExactNewton`: provides -nabla^2 log L directly, which is the observed Hessian
///   by construction. This is always correct.
///
/// - `Diagonal`: provides IRLS working weights W such that the per-block Hessian
///   is X'WX. For canonical links (logit-Binomial, log-Poisson), W_obs = W_Fisher.
///   For supported non-canonical diagonal links, W must be the observed weight
///   W_obs = W_Fisher - (y-mu)*B so the outer REML uses the exact Laplace
///   Hessian. The matching [`CustomFamily::diagonalworking_weights_directional_derivative`]
///   callback must differentiate the same observed W surface; silently using Fisher
///   weights or zero `dW` would change the criterion into a PQL-type surrogate.
#[derive(Clone, Debug)]
pub enum BlockWorkingSet {
    /// Standard IRLS/GLM-style diagonal working set for eta-space updates.
    Diagonal {
        /// IRLS pseudo-response for this block's linear predictor.
        working_response: Array1<f64>,
        /// IRLS working weights for this block (non-negative, length n).
        ///
        /// For the inner solver, Fisher or observed weights both find the same mode.
        /// For the outer REML/LAML log|H| term, observed weights are the correct
        /// Laplace choice (see response.md Section 3). Canonical-link families need
        /// no correction since observed = Fisher.
        working_weights: Array1<f64>,
    },
    /// Exact Newton block update in coefficient space.
    ///
    /// `gradient` is nabla log L wrt block coefficients.
    /// `hessian` is -nabla^2 log L wrt block coefficients (positive semidefinite near optimum).
    ///
    /// This is the observed Hessian by construction (actual second derivative of the
    /// log-likelihood), which is the correct quantity for the outer REML Laplace
    /// approximation.
    ExactNewton {
        gradient: Array1<f64>,
        hessian: SymmetricMatrix,
    },
}


impl BlockWorkingSet {
    /// Construct a `Diagonal` working set with the length invariant
    /// (`working_response.len() == working_weights.len()`) enforced at the
    /// type boundary. Use this from any new code path that produces a
    /// diagonal IRLS block; the legacy struct-literal form is preserved for
    /// existing call sites pending a full migration.
    #[inline]
    pub fn diagonal_checked(
        working_response: Array1<f64>,
        working_weights: Array1<f64>,
    ) -> Result<Self, String> {
        if working_response.len() != working_weights.len() {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "BlockWorkingSet::Diagonal length mismatch: working_response={}, working_weights={}",
                working_response.len(),
                working_weights.len(),
            ) }.into());
        }
        Ok(Self::Diagonal {
            working_response,
            working_weights,
        })
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactNewtonOuterObjective {
    RidgedQuadraticReml,
    StrictPseudoLaplace,
}


/// Highest exact outer derivative order a family wants to expose at the
/// current realized problem scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExactOuterDerivativeOrder {
    Zeroth,
    First,
    Second,
}


impl ExactOuterDerivativeOrder {
    pub const fn has_gradient(self) -> bool {
        !matches!(self, Self::Zeroth)
    }

    pub const fn has_hessian(self) -> bool {
        matches!(self, Self::Second)
    }
}


/// Exact outer derivative order for families that expose second-order
/// coefficient geometry.
///
/// This used to be a cost gate that demoted large large-scale problems to
/// first-order BFGS. That was a policy leak into the math layer: if the family
/// supplies analytic dense Hessian blocks or an analytic profiled-Hessian HVP,
/// the outer optimizer should see the exact second-order objective. Runtime
/// representation choices (dense vs operator) belong below this declaration,
/// not in a first-order downgrade.
/// Precondition check for the family capability / operator hooks (e.g.
/// `batched_outer_hessian_terms`, `outer_hyper_hessian_operator`).
///
/// These hooks operate on whatever block geometry the caller has assembled and
/// must validate the *consistency* of the specs they are handed — never the
/// fit-level "at least one block" precondition, which belongs to the fit entry
/// points (`validate_blockspecs`). An empty, self-consistent argument set is a
/// valid no-op probe of the operator path (the operator may ignore the specs
/// entirely), so it must not panic here.
fn assert_valid_blockspecs(specs: &[ParameterBlockSpec], context: &str) {
    assert!(
        validate_blockspec_consistency(specs).is_ok(),
        "{context}: inconsistent parameter block specs"
    );
}


fn assert_valid_options(options: &BlockwiseFitOptions, context: &str) {
    assert!(
        options.inner_tol.is_finite() && options.inner_tol >= 0.0,
        "{context}: inner_tol must be finite and non-negative"
    );
    assert!(
        options.outer_tol.is_finite() && options.outer_tol >= 0.0,
        "{context}: outer_tol must be finite and non-negative"
    );
    assert!(
        options.minweight.is_finite() && options.minweight >= 0.0,
        "{context}: minweight must be finite and non-negative"
    );
    assert!(
        options.ridge_floor.is_finite() && options.ridge_floor >= 0.0,
        "{context}: ridge_floor must be finite and non-negative"
    );
    if let Some(threshold) = options.early_exit_threshold {
        assert!(
            threshold.is_finite(),
            "{context}: early_exit_threshold must be finite"
        );
    }
}


fn assert_states_match_specs(
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    context: &str,
) {
    assert_eq!(
        states.len(),
        specs.len(),
        "{context}: state/spec block count mismatch"
    );
    for (block, (state, spec)) in states.iter().zip(specs).enumerate() {
        assert_eq!(
            state.beta.len(),
            spec.design.ncols(),
            "{context}: beta length mismatch in block {block}"
        );
        // `state.eta` is produced from `solver_design()` (see
        // `refresh_all_block_etas`), which is `stacked_design` when set
        // (3·n_obs rows for survival LS time-varying blocks) and `design`
        // (n_obs rows) otherwise. Use the same accessor here.
        assert_eq!(
            state.eta.len(),
            spec.solver_design().nrows(),
            "{context}: eta length mismatch in block {block}"
        );
    }
}


fn assert_derivative_blocks_match_specs(
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    specs: &[ParameterBlockSpec],
    context: &str,
) {
    assert_eq!(
        derivative_blocks.len(),
        specs.len(),
        "{context}: derivative/spec block count mismatch"
    );
}


fn assert_rho_matches_specs(rho: &Array1<f64>, specs: &[ParameterBlockSpec], context: &str) {
    let expected = specs.iter().map(|spec| spec.penalties.len()).sum::<usize>();
    assert_eq!(
        rho.len(),
        expected,
        "{context}: rho length does not match penalty count"
    );
}


fn validate_hessian_workspace_ready(
    hessian_workspace: &Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    context: &str,
) -> Result<(), String> {
    if let Some(workspace) = hessian_workspace.as_ref() {
        workspace
            .warm_up_outer_caches()
            .map_err(|err| format!("{context}: failed to warm Hessian workspace caches: {err}"))?;
    }
    Ok(())
}


pub fn exact_outer_order_from_capability(
    specs: &[ParameterBlockSpec],
    coefficient_cost: u64,
) -> ExactOuterDerivativeOrder {
    assert_valid_blockspecs(specs, "exact outer derivative order");
    match coefficient_cost {
        0 => ExactOuterDerivativeOrder::Second,
        _ => ExactOuterDerivativeOrder::Second,
    }
}


/// Capability-aware variant of [`exact_outer_order_from_capability`].
///
/// Kept as the public declaration helper for existing family impls, but it no
/// longer gates by cost. Once a caller has established dense or HVP analytic
/// second-order support, the correct derivative order is `Second`.
pub fn exact_outer_order_with_outer_hvp(
    specs: &[ParameterBlockSpec],
    coefficient_cost: u64,
    outer_hyper_hessian_hvp_available: bool,
) -> ExactOuterDerivativeOrder {
    if outer_hyper_hessian_hvp_available {
        assert_valid_blockspecs(specs, "exact outer derivative order with HVP");
        match coefficient_cost {
            0 => ExactOuterDerivativeOrder::Second,
            _ => ExactOuterDerivativeOrder::Second,
        }
    } else {
        exact_outer_order_from_capability(specs, coefficient_cost)
    }
}


/// Realized outer-derivative policy at the current problem size.
///
/// Capability (the family can produce exact second-order calculus) controls
/// whether the Hessian is declared. Runtime cost controls only representation
/// and staging choices below this layer. Large problems must stay on the exact
/// analytic Hessian path and use an operator representation when dense assembly
/// is too expensive; they are not demoted to first-order BFGS here.
///
/// `OuterDerivativePolicy` records the family's *capability*, the *predicted
/// per-eval cost* for both gradient-only and Hessian paths, and exposes the
/// two policy queries the outer optimizer actually needs:
///
/// * [`order_for_evaluation`](Self::order_for_evaluation) — clamp a requested
///   evaluation order against the policy gate.
/// * [`declared_hessian_form`](Self::declared_hessian_form) — what shape the
///   outer-strategy planner should declare to its plan ladder.
/// * [`should_use_staged_kappa`](Self::should_use_staged_kappa) — auto-route
///   the κ optimizer through the pilot/polish schedule at large `n`.
///
/// All thresholds are *const* — no env vars, no CLI flags. The cost model is
/// the family's own `coefficient_gradient_cost` / `coefficient_hessian_cost`
/// scaled by the joint outer-coordinate dimension, with `saturating_mul` so
/// overflow rounds up to the budget ceiling rather than wrapping silently.
#[derive(Clone, Copy, Debug)]
pub struct OuterDerivativePolicy {
    /// What exact calculus the family advertises in principle.
    pub capability: ExactOuterDerivativeOrder,
    /// Predicted per-eval work for one `ValueGradientHessian` evaluation.
    /// Rounded conservatively *up* via `saturating_mul`. Informational for
    /// representation and diagnostics; it does not disable Hessian capability.
    pub predicted_hessian_work: u128,
    /// Predicted per-eval work for one `ValueAndGradient` evaluation.
    /// Rounded conservatively *up* via `saturating_mul`.
    pub predicted_gradient_work: u128,
    /// True when the family's outer-only paths consume
    /// [`BlockwiseFitOptions::outer_score_subsample`] and produce
    /// Horvitz-Thompson-weighted partial sums (i.e. the family overrides
    /// `log_likelihood_only_with_options`,
    /// `exact_newton_joint_psi_workspace_with_options`, and any other
    /// outer-only hooks reached by `evaluate_custom_family_joint_hyper`).
    ///
    /// Determines whether the κ optimizer's pilot/polish staging schedule
    /// engages: when this is `false`, [`Self::should_use_staged_kappa`]
    /// returns `false` regardless of `n`. Engaging the schedule on a
    /// family that ignores the subsample is strictly worse than not
    /// engaging it — the schedule builds a `RowSet::Subsample` and the
    /// boundary plumbing installs an `OuterScoreSubsample` on options,
    /// but the family's default outer-only paths fall back to full-data
    /// sums, so the pilot evaluation costs the same as the polish but
    /// adds a Vec allocation per eval.
    ///
    /// Families that do **not** consume the subsample (default for new
    /// implementations, including the GAMLSS location-scale families
    /// today) leave this `false`. Families that do consume (today:
    /// `BernoulliMarginalSlopeFamily`) override `outer_derivative_policy`
    /// to set this `true`.
    pub subsample_capable: bool,
}


impl OuterDerivativePolicy {
    /// Per-eval gradient work ceiling above which the κ schedule switches
    /// to the staged pilot/polish path. At large scale (n ≳ 100 k) even
    /// the gradient sweep takes minutes per outer iter; subsampling the
    /// pilot stage cuts that to seconds and leaves the final polish on
    /// full data to recover the MLE.
    pub const OUTER_GRADIENT_WORK_BUDGET: u128 = 50_000_000_000;

    /// Pilot subsample auto-engages when full-data `n` exceeds this. Below
    /// this the κ schedule collapses to a single full-data stage —
    /// behaviour identical to the pre-P7 path.
    pub const STAGED_KAPPA_TRIGGER_N: usize = 30_000;

    /// Clamp a requested evaluation order against the policy gate.
    ///
    /// Returns the highest order this policy permits for the requested order:
    /// * `ValueGradientHessian` requested → keep only if `declared_hessian_form`
    ///   is something other than `Unavailable`.
    /// * `ValueAndGradient` requested → always permitted (gradient-only is
    ///   universal).
    pub fn order_for_evaluation(
        &self,
        requested: crate::solver::outer_strategy::OuterEvalOrder,
    ) -> crate::solver::outer_strategy::OuterEvalOrder {
        use crate::solver::outer_strategy::OuterEvalOrder;
        match requested {
            // Value-only is universal: every policy can evaluate the bare
            // objective, so the request passes through unclamped.
            OuterEvalOrder::Value => OuterEvalOrder::Value,
            OuterEvalOrder::ValueAndGradient => OuterEvalOrder::ValueAndGradient,
            OuterEvalOrder::ValueGradientHessian => {
                if matches!(
                    self.declared_hessian_form(),
                    crate::solver::outer_strategy::DeclaredHessianForm::Unavailable
                ) {
                    OuterEvalOrder::ValueAndGradient
                } else {
                    OuterEvalOrder::ValueGradientHessian
                }
            }
        }
    }

    /// Outer Hessian declaration for the outer-strategy planner.
    ///
    /// `Either` ⇔ capability has Hessian. Work estimates select dense vs
    /// operator assembly later; they must not erase analytic second-order
    /// capability from the planner.
    pub fn declared_hessian_form(&self) -> crate::solver::outer_strategy::DeclaredHessianForm {
        use crate::solver::outer_strategy::DeclaredHessianForm;
        if !self.capability.has_hessian() {
            return DeclaredHessianForm::Unavailable;
        }
        DeclaredHessianForm::Either
    }

    /// True when the κ optimizer should auto-route through the staged
    /// pilot/polish schedule. Triggers when **either** the data is big
    /// (`n ≥ STAGED_KAPPA_TRIGGER_N`) **or** the per-eval gradient work
    /// exceeds `OUTER_GRADIENT_WORK_BUDGET`. The second clause catches
    /// problems with moderate `n` but very wide design (large `p_total`
    /// or `psi_dim`) where a single full-data gradient sweep still
    /// dominates the κ trajectory.
    pub fn should_use_staged_kappa(&self, n: usize) -> bool {
        if !self.subsample_capable {
            // Family does not consume `outer_score_subsample` on its
            // outer-only paths. Engaging the schedule would build a
            // pilot `RowSet::Subsample` whose only effect is per-eval
            // Vec/Arc bookkeeping — the underlying coefficient gradient
            // would still sum every row. Gate the schedule off until
            // the family override declares consumption.
            return false;
        }
        n >= Self::STAGED_KAPPA_TRIGGER_N
            || self.predicted_gradient_work > Self::OUTER_GRADIENT_WORK_BUDGET
    }
}


/// Total outer-coordinate dimensionality used by the default policy work
/// model: `rho_dim + psi_dim`. Each outer evaluation propagates one
/// directional derivative per outer coordinate through the inner solve.
#[inline]
fn outer_coord_dim_for_policy(specs: &[ParameterBlockSpec], psi_dim: usize) -> u128 {
    let rho_total: u128 = specs
        .iter()
        .map(|s| s.penalties.len() as u128)
        .fold(0u128, |acc, k| acc.saturating_add(k));
    rho_total.saturating_add(psi_dim as u128)
}


/// Default predicted-cost model for [`OuterDerivativePolicy`]:
///
/// * gradient work ≈ `coefficient_gradient_cost · (rho_dim + psi_dim)`
/// * Hessian work  ≈ `coefficient_hessian_cost  · (rho_dim + psi_dim)`
///
/// Each outer coordinate triggers one analytic directional derivative
/// through the inner solve; the dense Hessian assembly carries the extra
/// `O(p_total)` factor already captured by `coefficient_hessian_cost`.
///
/// All multiplications saturate so an overflow rounds *up* to the gate
/// ceiling: we'd rather drop one Hessian evaluation that we could have
/// afforded than crash on a 600 s eval.
pub fn default_outer_derivative_policy_costs(
    specs: &[ParameterBlockSpec],
    psi_dim: usize,
    grad_cost: u64,
    hess_cost: u64,
) -> (u128, u128) {
    let k = outer_coord_dim_for_policy(specs, psi_dim);
    let grad = (grad_cost as u128).saturating_mul(k.max(1));
    let hess = (hess_cost as u128).saturating_mul(k.max(1));
    (grad, hess)
}


/// Default coefficient-space Hessian cost: `Σ_b n_b · p_b²`, summed across
/// blocks. Represents the work to assemble or apply the dense block-diagonal
/// inner Hessian once.
pub fn default_coefficient_hessian_cost(specs: &[ParameterBlockSpec]) -> u64 {
    specs
        .iter()
        .map(|s| {
            let n = s.design.nrows() as u64;
            let p = s.design.ncols() as u64;
            n.saturating_mul(p.saturating_mul(p))
        })
        .fold(0u64, |acc, c| acc.saturating_add(c))
}


/// Joint-coupled coefficient-space Hessian cost: `n · (Σ_b p_b)²`. The honest
/// per-evaluation work for any family whose row likelihood couples every block
/// (every observation contributes a rank-`m` outer-product update to the full
/// joint Hessian over `Σ p_b` coefficients), as opposed to the block-diagonal
/// `default_coefficient_hessian_cost` which assumes each `X_b' W_b X_b` is
/// assembled independently.
///
/// Used by all GAMLSS, marginal-slope, and joint-latent families. CTN does
/// not delegate here — it uses its Khatri–Rao factor dimensions internally.
pub fn joint_coupled_coefficient_hessian_cost(n: u64, specs: &[ParameterBlockSpec]) -> u64 {
    let p_total: u64 = specs
        .iter()
        .map(|s| s.design.ncols() as u64)
        .fold(0u64, |acc, p| acc.saturating_add(p));
    n.saturating_mul(p_total.saturating_mul(p_total))
}


/// Default coefficient-space gradient cost: half the Hessian cost.
///
/// The first-order analytic gradient in the unified evaluator runs the same
/// inner Newton solve as the second-order path but skips the `K`-fold
/// pairwise Hessian assembly (`B_{j,k}` blocks) and the `K`-fold inner
/// derivative solves; what remains is the inner solve plus a single
/// gradient-only sweep through the data. Empirically this is roughly half
/// the per-evaluation arithmetic of forming the dense Hessian, hence the
/// `/2` default. Families whose gradient assembly differs structurally
/// (e.g. matrix-free Hv operators with no dense Hessian assembly to halve)
/// should override [`CustomFamily::coefficient_gradient_cost`] explicitly.
pub fn default_coefficient_gradient_cost(specs: &[ParameterBlockSpec]) -> u64 {
    default_coefficient_hessian_cost(specs) / 2
}


/// Compute β-block column ranges from a slice of `ParameterBlockSpec`s.
///
/// Returns one `Range<usize>` per spec, covering the spec's columns in the
/// concatenated β vector (i.e. `offset .. offset + p_block` where `p_block =
/// spec.design.ncols()`). The ranges are non-overlapping, sorted, and their
/// union covers `0..Σ p_block`.
///
/// This is the canonical source of `block_offsets` for every
/// [`crate::solver::arrow_schur::ArrowSchurSystem`] built for a custom family
/// (survival, GAMLSS, transformation-normal, latent-survival, marginal-slope,
/// …). Pass the result to
/// [`crate::solver::arrow_schur::ArrowSchurSystem::set_block_offsets`] before
/// calling `solve` or `solve_with_options` whenever the system will use
/// [`crate::solver::arrow_schur::ArrowSolverMode::InexactPCG`].
///
/// Specs with zero columns produce a zero-width range; callers that want to
/// skip trivial blocks may filter on `r.start < r.end` after calling this
/// function.
pub fn block_offsets_from_specs(specs: &[ParameterBlockSpec]) -> Arc<[Range<usize>]> {
    let mut ranges: Vec<Range<usize>> = Vec::with_capacity(specs.len());
    let mut cursor = 0usize;
    for spec in specs {
        let p = spec.design.ncols();
        ranges.push(cursor..cursor + p);
        cursor += p;
    }
    Arc::from(ranges.into_boxed_slice())
}


/// Bound first-order outer iterations when each analytic-gradient evaluation is
/// already large-scale work. This is only applied after the planner has
/// selected a gradient-only route; second-order/ARC plans keep their requested
/// iteration budget.
pub fn cost_gated_first_order_max_iter(
    requested: usize,
    coefficient_gradient_cost: u64,
    has_outer_hessian: bool,
) -> usize {
    const FIRST_ORDER_OUTER_WORK_BUDGET: u64 = 80_000_000_000;
    const MIN_FIRST_ORDER_ITERS: usize = 4;

    if has_outer_hessian || requested <= 1 || coefficient_gradient_cost == 0 {
        return requested;
    }

    let affordable = (FIRST_ORDER_OUTER_WORK_BUDGET / coefficient_gradient_cost) as usize;
    requested.min(affordable.max(MIN_FIRST_ORDER_ITERS))
}


/// Local trust budget for first-order outer BFGS on log-smoothing parameters.
///
/// One unit in `rho = log(lambda)` is an `e`-fold smoothing-parameter change.
/// Previously this cap was `1.0`, which throttled BFGS to ~1/5 of its
/// quasi-Newton step on flat REML surfaces (the natural BFGS direction has
/// `|d|_inf` of ~5 in log-λ for large-scale survival fits). Probes whose
/// `step_inf > cap` are rejected for free in `OuterFirstOrderBridge::eval_cost`
/// (returning `BFGS_LINE_SEARCH_REJECT_COST` without running an inner solve),
/// so a larger cap costs nothing on rejection — it only lets Strong-Wolfe
/// accept bigger steps that the inner-PIRLS divergence guard can already
/// validate. `5.0` allows up to `e^5 ≈ 148`-fold smoothing-parameter change
/// per accepted outer iter, which matches the typical quasi-Newton direction
/// magnitude while still bounding pathological probes.
pub const fn first_order_bfgs_loglambda_step_cap(has_outer_hessian: bool) -> Option<f64> {
    if has_outer_hessian { None } else { Some(5.0) }
}


pub(crate) fn exact_newton_outer_geometry_supports_second_order_solver<F: CustomFamily + ?Sized>(
    family: &F,
) -> bool {
    family.exact_newton_outerobjective() == ExactNewtonOuterObjective::StrictPseudoLaplace
}


/// Family evaluation over all parameter blocks.
#[derive(Clone, Debug)]
pub struct FamilyEvaluation {
    pub log_likelihood: f64,
    pub blockworking_sets: Vec<BlockWorkingSet>,
}


pub struct ExactNewtonJointGradientEvaluation {
    pub log_likelihood: f64,
    pub gradient: Array1<f64>,
}


/// Batched per-θ_j contributions to the analytic outer gradient.
///
/// Used by [`CustomFamily::batched_outer_gradient_terms`] to amortize the
/// joint-Hessian factorization across all K hyperparameters: instead of
/// computing each `tr(H⁻¹ · Ḣ_j)` independently (K independent solves), the
/// family factors `H` once, computes per-row leverages `L_i = Z_i H⁻¹ Z_iᵀ`,
/// and accumulates all K traces in a single streaming pass.
///
/// All three vectors have length equal to the total number of outer
/// hyperparameters (K = `rho.len() + Σ derivative_blocks[b].len()`), in the
/// same coordinate order as the unified evaluator's gradient: ρ-coords first,
/// ψ-coords appended.
///
/// # Assembly formula
///
/// The caller assembles the outer gradient as
///
/// ```text
///   grad[j] = objective_theta[j]
///           + 0.5 * trace_h_inv_hdot[j]
///           - 0.5 * trace_s_pinv_sdot[j]
/// ```
///
/// matching the three-term convention in [`outer_gradient_entry`] (penalty +
/// trace − det).
pub struct BatchedOuterHessianTerms {
    /// Exact profiled outer Hessian over θ = (ρ, ψ), assembled or exposed in
    /// operator form by the family in one amortized evaluation.
    pub outer_hessian: crate::solver::outer_strategy::HessianResult,
}


pub struct BatchedOuterGradientTerms {
    /// Explicit ∂J/∂θ_j contributions evaluated at the converged β̂ holding
    /// β fixed (i.e. the part that does NOT flow through H or S):
    ///
    /// * For ρ-coords: `½ β̂ᵀ A_k β̂` (penalty quadratic).
    /// * For ψ-coords: `V_i^explicit + g_i^explicit · β̂` style contributions.
    pub objective_theta: Array1<f64>,
    /// `tr(H⁻¹ · ∂H/∂θ_j)` for each j, with H = -∇²log L + S the full
    /// penalized Hessian at the mode.
    pub trace_h_inv_hdot: Array1<f64>,
    /// `tr(S⁺ · ∂S/∂θ_j)` for each j (penalty pseudo-logdet first derivative).
    pub trace_s_pinv_sdot: Array1<f64>,
}


/// User-defined family contract for multi-block generalized models.
pub trait CustomFamily {
    /// Family-owned fingerprint for persistent coefficient warm-starts.
    ///
    /// The generic block specs contain design matrices, offsets, penalties,
    /// and dimensions, but they deliberately do not know the family response
    /// vector or likelihood-side data stored on `Self`. Reusing β across
    /// different responses is mathematically unsafe, so persistent block-level
    /// warm-starts are enabled only for families that provide a fingerprint of
    /// the data that defines their likelihood. Outer ρ cache remains available
    /// independently through `BlockwiseFitOptions::cache_session`.
    fn persistent_warm_start_fingerprint(
        &self,
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Option<String> {
        assert_valid_blockspecs(specs, "persistent warm-start fingerprint");
        assert_valid_options(options, "persistent warm-start fingerprint");
        None
    }

    /// Evaluate log-likelihood and per-block working quantities at current block predictors.
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String>;

    /// Compute only the log-likelihood without building working sets.
    ///
    /// This is used in backtracking line searches where only the objective value
    /// is needed, avoiding the O(n × blocks) cost of assembling IRLS working
    /// weights and responses that will be immediately discarded.
    ///
    /// The default implementation falls back to `evaluate()` and discards the
    /// working sets.  Families with expensive working-set assembly should
    /// override this for a significant speedup.
    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        self.evaluate(block_states).map(|e| e.log_likelihood)
    }

    /// Options-aware log-likelihood evaluation for line search.
    ///
    /// Default forwards to [`log_likelihood_only`] and ignores `_options`.
    /// Families that consult `options.outer_score_subsample` (or other
    /// per-call options that affect the LL value) must override this so the
    /// joint-Newton line search and the post-accept gradient reload agree
    /// on which row subset is being evaluated. Large-scale outer-only
    /// callers (including the joint-Newton line-search screening path) can
    /// override this to evaluate a deterministic paired Horvitz-Thompson
    /// estimate without constructing a full exact-Newton workspace.
    fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        assert_valid_options(options, "log_likelihood_only_with_options");
        self.log_likelihood_only(block_states)
    }

    /// Whether `log_likelihood_only_with_options` can use
    /// `BlockwiseFitOptions::early_exit_threshold` to reject line-search trials
    /// without computing the full log-likelihood.
    fn supports_log_likelihood_early_exit(&self) -> bool {
        false
    }

    /// Selects the outer objective semantics for exact-Newton families.
    ///
    /// `RidgedQuadraticReml` is the explicit ridged surrogate REML surface:
    ///
    ///   -loglik + penalty + 0.5 (log|H| - log|S|_+)
    ///
    /// The determinant terms in this mode are evaluated on the stabilized
    /// curvature surface declared by `ridge_policy`, so this objective is an
    /// explicitly modified surrogate rather than an exact Laplace expansion
    /// at an indefinite Hessian.
    ///
    /// `StrictPseudoLaplace` is the exact-mode pseudo-Laplace surface used by the
    /// Charbonnier spatial family:
    ///
    ///   -loglik + penalty + 0.5 log|H|
    ///
    /// The latter deliberately omits the quadratic-only `-0.5 log|S|_+`
    /// normalization term because there is no tractable exact analogue for the
    /// nonquadratic prior without introducing the intractable prior normalizer.
    fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
        ExactNewtonOuterObjective::RidgedQuadraticReml
    }

    /// Whether the joint likelihood Hessian H_L depends on β.
    ///
    /// When `true`, the unified evaluator includes M_j[u] = D_β B_j[u]
    /// moving-design drift correction for ψ coordinates and marks
    /// `HyperCoord::b_depends_on_beta = true`.
    ///
    /// Default: `true` for StrictPseudoLaplace, `false` for RidgedQuadraticReml.
    /// Gaussian location-scale must override to `true` because their
    /// joint Hessian depends on β even though outer objective is RidgedQuadraticReml.
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        self.exact_newton_outerobjective() != ExactNewtonOuterObjective::RidgedQuadraticReml
    }

    /// Whether the outer REML/LAML logdet term `½ log|H + Sλ|` and its analytic
    /// trace gradient `½ tr((H+Sλ)⁺ ∂Sλ)` are evaluated over the FULL
    /// identifiable subspace `range(H + Sλ)` (mgcv's generalized determinant,
    /// gam#752) rather than the penalty-range subspace `range(Sλ)`.
    ///
    /// This is a value/gradient SUBSPACE-CONSISTENCY concern, orthogonal to
    /// whether the Hessian depends on β (`exact_newton_joint_hessian_beta_dependent`,
    /// which gates the *drift* corrections). The previous code conflated the two
    /// by gating the projected logdet on β-dependence, so `RidgedQuadraticReml`
    /// families (survival/bernoulli marginal-slope) silently used the
    /// `range(Sλ)`-only determinant: on a near-collinear penalty-null trend (the
    /// clustered-PC matern marginal-slope geometry) that DROPS the penalty-null
    /// likelihood determinant `log|U_kᵀ H U_k|` from the value while
    /// `½ log|Sλ|₊` is correctly over `range(Sλ)`, making the ρ-derivative of the
    /// REML criterion inconsistent. The outer optimizer then drives that block's
    /// λ → ∞ and the envelope gradient (valid only at a stationary β̂) freezes —
    /// the constant-‖g‖ outer stall in gam#808/#787.
    ///
    /// The generalized determinant is the correct objective in ALL cases: when
    /// `H + Sλ` is full rank it equals the ordinary logdet (the projection is a
    /// no-op, so the correction is ≈0), and when it is rank-deficient it drops
    /// only the truly unidentified `ker(H) ∩ ker(Sλ)` directions — exactly the
    /// directions `½ log|Sλ|₊` also omits, keeping value and gradient over one
    /// subspace. Always enabled by default.
    fn use_projected_penalty_logdet(&self) -> bool {
        true
    }

    /// Per-evaluation arithmetic cost of forming or applying the inner
    /// coefficient-space Hessian once, in flop-equivalent units. This is used
    /// for diagnostics, seed-budget policy, and first-order iteration caps
    /// when a family genuinely lacks analytic second-order support. It is not
    /// allowed to hide an analytic Hessian from the outer optimizer.
    ///
    /// The default returns `Σ_b n_b · p_b²` via [`default_coefficient_hessian_cost`],
    /// which is the honest assembly cost only when the joint Hessian is
    /// **block-diagonal** — i.e. the inner solver assembles each block's
    /// `X_b' W_b X_b` independently, with no cross-block coupling per row.
    /// Families whose row likelihood couples all blocks (every row contributes
    /// a rank-`m` outer-product update to the full joint Hessian over
    /// `Σ p_b` coefficients) **must** override and delegate to
    /// [`joint_coupled_coefficient_hessian_cost`] (or the equivalent factored
    /// form for tensor designs), otherwise the default undercounts the
    /// cross-block outer-product terms `2·Σ_{a<b} n·p_a·p_b`.
    ///
    /// Concretely:
    ///
    /// * **Block-diagonal** (default OK): `LatentBinaryFamily` collects
    ///   separate `hess_time` and `hess_mean` per row, never forming an
    ///   off-diagonal contribution.
    /// * **Joint-coupled** (override via [`joint_coupled_coefficient_hessian_cost`]):
    ///   GAMLSS location-scale, GAMLSS wiggle variants, marginal-slope families
    ///   (Bernoulli, Survival), `LatentSurvivalFamily`,
    ///   `SurvivalLocationScaleFamily` — every row contributes to the full
    ///   `(Σ p_b)²` joint Hessian via Jacobian pullback of a multi-dimensional
    ///   primary kernel.
    /// * **Single-block** (default OK): tensor designs whose `design.ncols()`
    ///   already equals `p_total` (e.g. CTN's Khatri–Rao `n × (p_resp·p_cov)`);
    ///   `n · p²` reduces correctly to `n · p_resp² · p_cov²`.
    /// * **Matrix-free Hessian operator**: families that expose
    ///   [`Self::exact_newton_joint_hessian_workspace`] with operator-form
    ///   directional derivatives (CTN at large scale) may instead return
    ///   the per-`Hv` matvec cost (e.g. `n·(p_resp + p_cov)` for Khatri–Rao)
    ///   so the gate reflects the operator path rather than the dense
    ///   build that the unified evaluator skips.
    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        default_coefficient_hessian_cost(specs)
    }

    /// Per-evaluation arithmetic cost of one analytic-gradient outer
    /// evaluation, in flop-equivalent units. Used only when the family
    /// genuinely has no analytic outer Hessian and the planner must use a
    /// first-order optimizer.
    ///
    /// The default returns `coefficient_hessian_cost / 2` (see
    /// [`default_coefficient_gradient_cost`]). Families whose gradient
    /// assembly differs structurally should override; in particular,
    /// joint-coupled families that override `coefficient_hessian_cost` to
    /// `joint_coupled_coefficient_hessian_cost(n, specs)` automatically
    /// inherit the corresponding gradient cost via this default — no
    /// per-family override is required for the GAMLSS / marginal-slope /
    /// joint-latent path.
    fn coefficient_gradient_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        self.coefficient_hessian_cost(specs) / 2
    }

    /// Declares how much exact outer calculus this family wants to expose for
    /// the current realized problem size.
    ///
    /// The default exposes exact second-order calculus whenever the family
    /// advertises either dense outer Hessian blocks or profiled outer-Hessian
    /// HVPs. Large problems must stay exact and select an operator
    /// representation; they are not demoted to first-order optimizers.
    ///
    /// **Capability vs representation.** This method reports the highest
    /// analytic order this family implements. The realized policy carries
    /// work estimates for dense/operator routing and staged κ schedules, but
    /// those estimates do not downgrade a second-order family to a first-order
    /// optimizer.
    fn exact_outer_derivative_order(
        &self,
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> ExactOuterDerivativeOrder {
        assert!(std::mem::size_of_val(options) > 0);
        let coefficient_work = self
            .coefficient_hessian_cost(specs)
            .max(self.coefficient_gradient_cost(specs));
        if !self.outer_hyper_hessian_dense_available(specs)
            && !self.outer_hyper_hessian_hvp_available(specs)
        {
            return ExactOuterDerivativeOrder::First;
        }
        exact_outer_order_with_outer_hvp(
            specs,
            coefficient_work,
            self.outer_hyper_hessian_hvp_available(specs),
        )
    }

    /// Realized outer-derivative policy at the current problem size.
    ///
    /// Combines the capability query [`Self::exact_outer_derivative_order`]
    /// with predicted per-eval costs from [`Self::coefficient_gradient_cost`] /
    /// [`Self::coefficient_hessian_cost`] and the joint outer-coordinate
    /// dimension `rho_dim + psi_dim`. Capability decides derivative order;
    /// predicted costs inform dense/operator routing and staged κ schedules.
    ///
    /// Families with non-generic cost models (Khatri–Rao CTN, matrix-free
    /// HVP families, marginal-slope row-third workloads) should override
    /// this directly and set the `predicted_*_work` fields from their own
    /// cost model. The default uses the generic
    /// `n × (rho_dim + psi_dim) × p_total` shape via
    /// [`default_outer_derivative_policy_costs`].
    fn outer_derivative_policy(
        &self,
        specs: &[ParameterBlockSpec],
        psi_dim: usize,
        options: &BlockwiseFitOptions,
    ) -> OuterDerivativePolicy {
        let capability = self.exact_outer_derivative_order(specs, options);
        let grad_cost = self.coefficient_gradient_cost(specs);
        let hess_cost = self.coefficient_hessian_cost(specs);
        let (predicted_gradient_work, predicted_hessian_work) =
            default_outer_derivative_policy_costs(specs, psi_dim, grad_cost, hess_cost);
        OuterDerivativePolicy {
            capability,
            predicted_gradient_work,
            predicted_hessian_work,
            subsample_capable: self.outer_derivative_subsample_capable(),
        }
    }

    /// Whether this family's outer-only paths honour HT-weighted partial sums
    /// over `options.outer_score_subsample`.
    ///
    /// Default `false`: the trait's default outer-only paths
    /// (`log_likelihood_only_with_options`,
    /// `exact_newton_joint_psi_workspace_with_options`, ...) forward to the
    /// no-options variants and ignore `outer_score_subsample`. Families that
    /// override those hooks to honour HT-weighted partial sums should override
    /// this hook to return `true`; the default [`Self::outer_derivative_policy`]
    /// then threads the flag into the emitted [`OuterDerivativePolicy`].
    fn outer_derivative_subsample_capable(&self) -> bool {
        false
    }

    /// Family-specific outer seeding policy.
    ///
    /// The default preserves the generic custom-family behavior. Families with
    /// a strong warm start can override this to keep seed screening from
    /// dominating the fit.
    fn outer_seed_config(&self, n_params: usize) -> crate::seeding::SeedConfig {
        if n_params == 0 {
            return crate::seeding::SeedConfig::default();
        }
        let mut config = crate::seeding::SeedConfig::default();
        config.max_seeds = if n_params <= 4 { 6 } else { 4 };
        config.seed_budget = 1;
        config.screen_max_inner_iterations = 2;
        config
    }

    /// Whether outer hyper-derivative evaluation must use a joint exact path.
    ///
    /// Default `false` allows the generic blockwise diagonal fallback when a
    /// family does not provide joint exact curvature.
    ///
    /// Families with coupled multi-block likelihoods can override this to
    /// prevent the outer code from silently evaluating a mathematically
    /// invalid block-local surrogate. The failure mode is:
    ///
    /// 1. the outer derivative still has block-local forcing
    ///      g_k = A_k beta
    ///    because `rho_k` enters only through the penalty;
    /// 2. but the fitted mode response is not block-local,
    ///      H u_k = -g_k,
    ///    because the likelihood Hessian has off-diagonal block coupling;
    /// 3. therefore a blockwise solve
    ///      H_b u_{k,b} = -(A_k beta)_b
    ///    is not the derivative of the profiled objective the code claims to
    ///    be optimizing.
    ///
    /// When this flag is `true`, the family is asserting that any outer
    /// hyper-derivative path must first obtain the full joint exact curvature
    /// before it can return a mathematically valid result.
    fn requires_joint_outer_hyper_path(&self) -> bool {
        false
    }

    /// Per-block output-channel assignment for the identifiability audit.
    ///
    /// Multi-parameter families (Dirichlet, beta, Gaussian/binomial
    /// location-scale, multinomial, …) drive several *independent* linear
    /// predictors `η_r = X_r β_r`, one per distributional parameter / class.
    /// Each [`ParameterBlockSpec`] feeds exactly one of those output channels.
    /// When two blocks share the same covariate basis (e.g. every Dirichlet
    /// component uses the same `[1 | B]`), their columns are *not* gauge
    /// aliases — they are block-diagonal entries of the true joint Jacobian
    /// `blkdiag(X_0, …, X_{m-1})`, full rank `Σ p_b`.
    ///
    /// The pre-fit identifiability audit can only see this block-diagonal
    /// structure through the **channel-aware** route, which requires each
    /// block to carry a multi-output `jacobian_callback` (n_outputs > 1).
    /// Families built via the canonical helpers (`build_location_scale_block`,
    /// `MultinomialFamily::build_block_specs`) wire that callback themselves;
    /// families fit through the low-level `fit_custom_family` API with
    /// hand-built specs do not, and the flat audit then mistakes the repeated
    /// shared basis for cross-block aliases and refuses a well-posed fit
    /// (issues #319 / #363 / #558).
    ///
    /// Returning `Some(channels)` — a vector of length `specs.len()` giving the
    /// zero-based output channel each block drives — lets `fit_custom_family`
    /// install the appropriate [`AdditiveBlockJacobian`] on any block that
    /// lacks an explicit callback, so the audit routes channel-aware
    /// automatically. The total channel count is `channels.iter().max() + 1`.
    ///
    /// Default: every block drives output channel 0. `wire_output_channels`
    /// recognizes this as the single-output flat route and leaves specs unchanged.
    ///
    /// When `Some`, the returned vector MUST have length equal to the number
    /// of blocks; `fit_custom_family` surfaces a structured error otherwise.
    fn output_channel_assignment(&self, specs: &[ParameterBlockSpec]) -> Option<Vec<usize>> {
        Some(vec![0; specs.len()])
    }

    /// Optional dynamic geometry hook for blocks whose design/offset depend on
    /// current values of other blocks.
    fn block_geometry(
        &self,
        block_states: &[ParameterBlockState],
        spec: &ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        assert!(block_states.len() <= isize::MAX as usize);
        Ok((spec.design.clone(), spec.offset.clone()))
    }

    /// Whether `block_geometry(...)` can change with the current block state.
    ///
    /// The default implementation is static: the effective geometry is just the
    /// stored `spec.design/spec.offset`, so the fit engine can use those
    /// references directly without repeatedly cloning dense matrices.
    ///
    /// Families that override `block_geometry(...)` with state-dependent
    /// behavior must override this to return `true`.
    fn block_geometry_is_dynamic(&self) -> bool {
        false
    }

    /// Optional directional derivative of the effective block geometry wrt the
    /// current block coefficients.
    ///
    /// For a block with effective predictor
    ///
    ///   eta(beta) = X(beta) beta + o(beta),
    ///
    /// the directional derivative along `d_beta` is
    ///
    ///   D eta[d_beta] = X d_beta + (D X[d_beta]) beta + D o[d_beta].
    ///
    /// For diagonal working-set REML derivatives this contributes to both:
    ///
    ///   D H[d_beta]
    ///   = (D X[d_beta])^T W X
    ///   + X^T W (D X[d_beta])
    ///   + X^T diag(D w[D eta[d_beta]]) X,
    ///
    /// and to the predictor drift fed into the weight directional derivative.
    ///
    /// Default `None` means the family is declaring that the current block's
    /// geometry has no coefficient-dependent drift beyond the base `X d_beta`
    /// term. Families with dynamic `block_geometry` must implement this hook
    /// when that declaration is false.
    fn block_geometry_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        idx: usize,
        block_spec: &ParameterBlockSpec,
        arr: &Array1<f64>,
    ) -> Result<Option<BlockGeometryDirectionalDerivative>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(!block_spec.name.is_empty());
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(None)
    }

    /// Optional per-block coefficient projection applied after each block update.
    fn post_update_block_beta(
        &self,
        block_states: &[ParameterBlockState],
        idx: usize,
        block_spec: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(!block_spec.name.is_empty());
        Ok(beta)
    }

    /// Optional barrier-aware maximum feasible step size for a block update.
    ///
    /// Given the current block state and a proposed step direction `delta`,
    /// returns `Some(alpha_max)` where `alpha_max` is the largest step size
    /// in `(0, 1]` such that `beta + alpha_max * delta` remains strictly
    /// feasible with respect to any implicit barrier in the likelihood.
    ///
    /// Families whose log-likelihood contains natural log-barrier terms
    /// (e.g. `log(h')` in transformation-normal) should implement this to
    /// prevent the line search from evaluating the likelihood at infeasible
    /// points.  A fraction-to-boundary safety factor (e.g. 0.995) should be
    /// applied internally.
    ///
    /// Returns `None` if no barrier constraint applies (the default).
    fn max_feasible_step_size(
        &self,
        block_states: &[ParameterBlockState],
        idx: usize,
        arr: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(None)
    }

    /// Optional linear inequality constraints for a block update:
    /// `A * beta_block >= b`.
    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        idx: usize,
        block_spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(!block_spec.name.is_empty());
        Ok(None)
    }

    /// Optional exact directional derivative of a block's ExactNewton Hessian.
    ///
    /// Returns `Some(dH)` where:
    /// - `dH` is the directional derivative of the block Hessian with respect to
    ///   the provided coefficient-space direction `d_beta` at current state.
    /// - shape is `(p_block, p_block)`.
    ///
    /// Default `None` means no exact directional Hessian drift is available.
    /// Exact REML/LAML derivative paths that require this term should treat
    /// `None` as unavailable rather than silently substituting zero.
    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        idx: usize,
        arr: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(None)
    }

    /// Optional exact second directional derivative of a block's ExactNewton Hessian.
    ///
    /// Returns `Some(d2H)` where:
    /// - `d2H` is `D²_beta H_L[u, v]` for the provided block-local
    ///   coefficient-space directions.
    /// - shape is `(p_block, p_block)`.
    ///
    /// Generic single-block REML/LAML Hessian evaluation requires this term for
    /// `BlockWorkingSet::ExactNewton` blocks; `None` means the exact second
    /// Hessian drift is unavailable.
    fn exact_newton_hessian_second_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        idx: usize,
        arr: &Array1<f64>,
        arr2: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(arr.iter().all(|v| !v.is_nan()));
        assert!(arr2.iter().all(|v| !v.is_nan()));
        Ok(None)
    }

    /// Optional exact joint coefficient-space Hessian across all blocks.
    ///
    /// Returns the unpenalized matrix `H_L = -nabla^2 log L` in the flattened block order.
    ///
    /// This is the **observed** (actual) Hessian of the log-likelihood at the mode,
    /// NOT the expected Fisher information. The outer REML/LAML evaluator requires
    /// the observed Hessian for the exact Laplace approximation (see response.md
    /// Section 3). Since this method returns the actual second derivative of log L,
    /// it is correct by construction.
    ///
    /// For families using `BlockWorkingSet::Diagonal` (IRLS-style updates), the
    /// per-block Hessian is X'WX where W is the working weight. For canonical links
    /// W_obs = W_Fisher, but for non-canonical links the working weight should include
    /// the observed-information correction W_obs = W_Fisher - (y-mu)*B.
    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        // Default block-diagonal assembly from per-block ExactNewton hessians.
        // This is the inner-fit-side default and is *intentionally* not gated
        // by `likelihood_blocks_uncoupled()`: the inner joint-Newton loop only
        // uses this Hessian as a Newton-direction surrogate that is
        // immediately validated by the line-search + objective decrease, so
        // even if the family is coupled, an under-resolved block-diagonal
        // direction will simply backtrack instead of corrupting the outer
        // REML score.  The strict coupling gate lives one layer up, on
        // `exact_newton_joint_hessian_with_specs`, where outer REML trace
        // algebra would silently produce wrong answers from a missing
        // cross-block term.
        exact_newton_joint_hessian_from_exact_blocks(self, block_states)
    }

    /// Optional exact joint log-likelihood / score evaluation in flattened
    /// coefficient space without building per-block Hessian working sets.
    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(block_specs.len() <= isize::MAX as usize);
        Ok(None)
    }

    /// Optional block-concatenated log-likelihood gradient `g = nabla l(theta)`
    /// assembled from the SAME single source of truth as
    /// [`Self::exact_newton_joint_hessian`] (e.g. a per-row jet-tower kernel), so
    /// the damped Newton `H delta = g` is solved on a consistent (objective,
    /// gradient, Hessian) triple. The default returns `None`, leaving the caller
    /// on its legacy hand-assembled gradient.
    fn exact_newton_joint_loglik_gradient(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array1<f64>>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        Ok(None)
    }

    /// Optional exact directional derivative of the joint coefficient-space Hessian.
    ///
    /// Returns `Some(dH)` where `dH` is the directional derivative of the
    /// unpenalized joint Hessian `H = -∇² log L` along the flattened
    /// coefficient-space direction `d_beta_flat`.
    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        exact_newton_joint_hessian_directional_derivative_from_blocks(
            self,
            block_states,
            d_beta_flat,
        )
    }

    /// Optional exact second directional derivative of the joint Hessian.
    ///
    /// Returns `Some(d2H)` where `d2H` is:
    ///   D²H[u, v] = d/dε d/dδ H(beta + εu + δv) |_{ε=δ=0}
    /// for flattened coefficient-space directions `u = d_beta_u_flat`,
    /// `v = d_betav_flat`.
    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        exact_newton_joint_hessiansecond_directional_derivative_from_blocks(
            self,
            block_states,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    /// Optional per-evaluation workspace for exact joint Hessian operators and
    /// directional derivatives.
    ///
    /// Families with expensive cache construction can override this to build
    /// shared state once and reuse it across the repeated `dH[v]` / `d²H[u,v]`
    /// calls made by the unified outer evaluator.
    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(block_specs.len() <= isize::MAX as usize);
        Ok(None)
    }

    /// Outer-aware variant of `exact_newton_joint_hessian_workspace`.
    ///
    /// Families that consume the optional outer-only stratified row subsample
    /// (`options.outer_score_subsample`) override this method so the joint
    /// Hessian workspace can be constructed with the subsample mask attached.
    /// Generic families can stick with the default implementation, which
    /// simply forwards to the legacy no-options method and ignores the
    /// options. This keeps full backward compatibility with existing
    /// implementors while letting the marginal-slope families thread the
    /// subsample down into the cached per-evaluation joint-Hessian directional
    /// derivative paths.
    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        assert_valid_options(options, "exact Newton joint Hessian workspace");
        self.exact_newton_joint_hessian_workspace(states, specs)
    }

    /// Optional batched analytic-gradient hook.
    ///
    /// Returns the K per-θ_j gradient contributions ([`BatchedOuterGradientTerms`])
    /// in one amortized pass when the family can factor its joint Hessian
    /// once and stream row-block leverages instead of computing each
    /// `tr(H⁻¹ · ∂H/∂θ_j)` independently.
    ///
    /// # Cost amortization
    ///
    /// Generic per-θ_j path: `O(K · n · p²)` (K independent dense traces).
    /// Batched path: `O(n · p²)` (single factor + leverage stream)
    ///                 + `O(K · n · m²)` (per-row block-diagonal accumulators
    ///                   with `m` = per-row predictor dimension; m = 2 for
    ///                   GAMLSS location-scale, 1 for scalar GLMs).
    ///
    /// At large scale with K ≈ 15, p ≈ 64, m = 2 the batched path is
    /// ≈ K·p²/(p² + K·m²) ≈ 15× cheaper.
    ///
    /// # Default
    ///
    /// Returns `Ok(None)`. The unified outer gradient evaluator falls back
    /// to its generic per-coordinate path. Families with row-coupled
    /// likelihoods (GAMLSS location-scale, marginal-slope) should override.
    ///
    /// Implementations may return `Ok(None)` for ψ-coordinates whose
    /// design-drift is too involved for a batched leverage form, letting
    /// the generic path handle those cases.
    fn batched_outer_gradient_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        rho: &Array1<f64>,
        options: &BlockwiseFitOptions,
        hessian_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    ) -> Result<Option<BatchedOuterGradientTerms>, String> {
        assert_valid_blockspecs(specs, "batched outer gradient terms");
        assert_states_match_specs(block_states, specs, "batched outer gradient terms");
        assert_derivative_blocks_match_specs(
            derivative_blocks,
            specs,
            "batched outer gradient terms",
        );
        assert_rho_matches_specs(rho, specs, "batched outer gradient terms");
        assert_valid_options(options, "batched outer gradient terms");
        validate_hessian_workspace_ready(&hessian_workspace, "batched outer gradient terms")?;
        Ok(None)
    }

    /// Optional batched analytic-Hessian / HVP hook.
    ///
    /// This is the Hessian-side analogue of
    /// [`Self::batched_outer_gradient_terms`]: families that can share a
    /// single factorization, row-leverage stream, or directional θθ kernel
    /// across all explicit outer-Hessian terms return the exact profiled
    /// Hessian here.  The evaluator uses this hook only for Hessian-capable
    /// families and only after the inner mode has been fitted; default
    /// `None` leaves unsupported families on their existing exact path.
    fn batched_outer_hessian_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        rho: &Array1<f64>,
        hessian_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    ) -> Result<Option<BatchedOuterHessianTerms>, String> {
        assert_valid_blockspecs(specs, "batched outer Hessian terms");
        assert_states_match_specs(block_states, specs, "batched outer Hessian terms");
        assert_derivative_blocks_match_specs(
            derivative_blocks,
            specs,
            "batched outer Hessian terms",
        );
        assert_rho_matches_specs(rho, specs, "batched outer Hessian terms");
        validate_hessian_workspace_ready(&hessian_workspace, "batched outer Hessian terms")?;
        Ok(self
            .outer_hyper_hessian_operator(specs)
            .map(|operator| BatchedOuterHessianTerms {
                outer_hessian: crate::solver::outer_strategy::HessianResult::Operator(operator),
            }))
    }

    /// Explicit name for the inner coefficient-space Hessian HVP capability.
    ///
    /// Kept separate from outer hyper-Hessian capabilities so CTN/GAMLSS row
    /// operators do not accidentally advertise pairwise θθ calculus as cheap.
    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        assert_valid_blockspecs(specs, "inner coefficient Hessian HVP availability");
        false
    }

    fn inner_joint_workspace_gradient_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        assert_valid_blockspecs(specs, "inner joint workspace gradient availability");
        false
    }

    /// Opt families in to the matrix-free inner-Newton/PCG path on top of the
    /// generic `use_joint_matrix_free_path` heuristic.
    ///
    /// `use_joint_matrix_free_path` is tuned for families with cheap per-row
    /// work where dense `O(n·p²)` assembly is itself the bottleneck and HVPs
    /// cost the same. Families with very expensive per-row work (e.g. BMS flex
    /// streaming cell partitions + flex-jet evaluations per row) can override
    /// this to force the operator path even at moderate `p`, because each HVP
    /// reuses the row stream once and PCG converges in a handful of iters.
    /// Default `false` keeps the heuristic untouched for everyone else.
    fn prefers_matrix_free_inner_joint(
        &self,
        specs: &[ParameterBlockSpec],
        states: &[ParameterBlockState],
    ) -> bool {
        assert_valid_blockspecs(specs, "matrix-free inner-joint preference");
        assert!(states.len() <= isize::MAX as usize);
        false
    }

    fn inner_joint_workspace_log_likelihood_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        assert_valid_blockspecs(specs, "inner joint workspace log-likelihood availability");
        false
    }

    /// True only when the family has a real profiled outer Hessian-vector
    /// product over θ = (ρ, ψ), without enumerating all θ_i θ_j pairs.
    fn outer_hyper_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        assert_valid_blockspecs(specs, "outer hyper-Hessian HVP availability");
        false
    }

    /// True when the family can expose the dense profiled outer Hessian.
    /// Generic custom-family pairwise derivative paths default to dense
    /// availability; families with only inner HVP support should override this
    /// if dense θθ assembly is not a valid capability for their path.
    fn outer_hyper_hessian_dense_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        assert_valid_blockspecs(specs, "outer hyper-Hessian dense availability");
        true
    }

    /// Family-supplied exact outer Hessian operator over θ = (ρ, ψ).
    ///
    /// When a family can produce the full profiled outer Hessian as a
    /// matrix-free Hv operator — using its own directional θθ kernels and
    /// trace algebra rather than the generic per-pair enumeration — it
    /// overrides this method and returns `Some(op)`.  The unified REML/LAML
    /// evaluator wires the operator into [`HessianResult::Operator`] via
    /// the [`HessianDerivativeProvider::family_outer_hessian_operator`] hook
    /// the family installs on its provider; consumers see a generic
    /// `Arc<dyn OuterHessianOperator>` (matvec / dim / mul_mat /
    /// is_cheap_to_materialize).
    ///
    /// Default returns `None`, leaving the family on the existing pairwise
    /// assembly path.  This is the architectural contract for CTN, survival
    /// (Gompertz-Makeham + timewiggle), GAMLSS location-scale, and
    /// Bernoulli marginal-slope families to plug their directional
    /// outer-HVP operators into the same surface.
    fn outer_hyper_hessian_operator(
        &self,
        specs: &[ParameterBlockSpec],
    ) -> Option<Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>> {
        assert_valid_blockspecs(specs, "outer hyper-Hessian operator");
        None
    }

    /// Optional spec-aware exact joint Hessian.
    ///
    /// This hook exists because the outer hyper-derivative code works from the
    /// realized block specs, while some family instances may or may not cache
    /// those realized designs internally.
    ///
    /// The profiled/Laplace outer objective used here is
    ///
    ///   J(theta)
    ///   = V(beta(theta), theta)
    ///     + 0.5 log|H(beta(theta), theta)|
    ///     - 0.5 log|S(theta)|_+,
    ///
    /// evaluated at the fitted inner mode defined by
    ///
    ///   F(beta, theta) := D_beta V(beta, theta) = 0,
    ///   H(beta, theta) := F_beta(beta, theta) = H_L(beta, theta) + S(theta).
    ///
    /// For pure rho directions on families whose likelihood has no explicit
    /// rho-dependence, the fixed-beta forcing is
    ///
    ///   g_k := F_{rho_k} = A_k beta,
    ///   A_k := dS/drho_k.
    ///
    /// Differentiating stationarity gives the exact joint mode response
    ///
    ///   H u_k = -g_k,
    ///   u_k = d beta / d rho_k.
    ///
    /// Even if `A_k` is supported in only one penalty block, the solve for
    /// `u_k` must use the full joint Hessian `H`, because the likelihood can
    /// couple blocks through off-diagonal curvature. The first outer
    /// derivative is then
    ///
    ///   dJ/dtheta_i
    ///   = 0.5 beta^T A_k beta
    ///     + 0.5 tr(H^{-1}(A_k + D_beta H_L[u_k]))
    ///     - 0.5 tr(S^+ A_k),
    ///
    /// and when psi moves realized penalties the same spec-aware hook must be
    /// able to reconstruct H(beta, theta), D_beta H[u], and D_beta^2 H[u, v]
    /// from the current realized specs so the generic joint assembler can form
    ///
    ///   dot H_i  = H_i + D_beta H[beta_i],
    ///   ddot H_ij
    ///   = H_ij + T_i[beta_j] + T_j[beta_i]
    ///     + D_beta H[beta_ij] + D_beta^2 H[beta_i, beta_j].
    ///
    /// Families such as binomial location-scale with
    ///
    ///   q = -eta_t exp(-eta_ls)
    ///
    /// have exactly that coupled structure: the penalty forcing is block-local
    /// but the fitted mode response and the resulting `D_beta H_L[u_k]` drift
    /// are joint objects. If the realized `specs` already contain the designs
    /// needed to build those objects, the outer code should use them directly
    /// rather than falling back to a weaker blockwise surrogate just because
    /// the family instance itself did not cache the same designs.
    ///
    /// The default implementation delegates to `exact_newton_joint_hessian`.
    ///
    /// For multi-block families, the working-set fallback only fires when the
    /// family has explicitly declared its blocks are uncoupled in the
    /// likelihood Hessian via `likelihood_blocks_uncoupled() = true`.  This
    /// is critical: `exact_newton_joint_hessian_from_working_sets` produces a
    /// strictly block-diagonal joint Hessian, which silently drops cross-block
    /// `∂²L/∂β_a∂β_b` terms for coupled likelihoods (GAMLSS μ-σ, marginal
    /// slope, survival location-scale, etc.).  Default `false` ⇒ multi-block
    /// custom families must override `exact_newton_joint_hessian` (or
    /// `exact_newton_outer_curvature`) and the higher layer surfaces a loud
    /// "joint outer path required" error rather than silently using
    /// block-diagonal curvature.
    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        // Multi-axis dispatch over the joint Hessian source:
        //
        // * Single-block, or family declared `likelihood_blocks_uncoupled` —
        //   the working-sets block-diagonal IS exact (no cross-block coupling
        //   exists), so it's a valid fallback when the family override
        //   returns None.
        //
        // * Multi-block coupled with `has_explicit_joint_hessian = true` —
        //   the family override IS the only trusted joint Hessian.  If it
        //   returns None (e.g. dense form too large for memory at large-scale
        //   scale), propagate None.  Substituting the working-sets
        //   block-diagonal would silently drop the cross-block
        //   ∂²L/∂β_a∂β_b curvature the family is the only source of —
        //   exactly the corruption this gate exists to prevent.
        //
        // * Multi-block coupled, no explicit override — refuse entirely so
        //   the multi-block error surfaces upstream.
        if specs.len() <= 1 || self.likelihood_blocks_uncoupled() {
            match self.exact_newton_joint_hessian(block_states)? {
                Some(hessian) => Ok(Some(hessian)),
                None => exact_newton_joint_hessian_from_working_sets(self, block_states, specs),
            }
        } else if self.has_explicit_joint_hessian() {
            self.exact_newton_joint_hessian(block_states)
        } else {
            // Multi-block coupled family that did NOT set the explicit marker.
            // The marker exists because the trait cannot reflect on whether
            // `exact_newton_joint_hessian` was overridden — its *default* impl
            // assembles a strictly block-diagonal matrix from per-block exact
            // blocks, which would silently drop cross-block ∂²L/∂β_a∂β_b
            // curvature for a coupled likelihood. But the marker is not the
            // only available signal: a family that genuinely overrides the
            // joint Hessian with true coupled curvature produces a matrix with
            // *nonzero off-diagonal blocks*, which the block-diagonal default
            // can never produce. Detect that structurally and trust it. A
            // returned matrix that is block-diagonal is indistinguishable from
            // the default for a coupled family, so it stays gated to None.
            match self.exact_newton_joint_hessian(block_states)? {
                Some(hessian) if joint_hessian_has_cross_block_coupling(&hessian, block_states) => {
                    Ok(Some(hessian))
                }
                _ => Ok(None),
            }
        }
    }

    /// Structural-coupling probe shared by the `_with_specs` joint dispatch
    /// gates: is the family's `exact_newton_joint_hessian` a genuinely coupled
    /// matrix (nonzero off-diagonal blocks), as opposed to the trait's
    /// block-diagonal default? This is the marker-free signal that lets the
    /// engine trust a coupled multi-block family that overrode the joint
    /// Hessian without hand-setting `has_explicit_joint_hessian()`. Returns
    /// `false` when no joint Hessian is available or it is block-diagonal.
    fn joint_hessian_is_structurally_coupled(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<bool, String> {
        Ok(match self.exact_newton_joint_hessian(block_states)? {
            Some(hessian) => joint_hessian_has_cross_block_coupling(&hessian, block_states),
            None => false,
        })
    }

    /// Whether the family's log-likelihood Hessian is block-diagonal in the
    /// joint coefficient vector — i.e. `∂²L/∂β_a∂β_b = 0` for every pair of
    /// distinct blocks `a ≠ b`.  Default `false` (assume coupling, the safe
    /// answer); families whose blocks share no η/W coupling override to
    /// `true` to opt into the default working-set joint-Hessian assembly for
    /// multi-block specs.
    fn likelihood_blocks_uncoupled(&self) -> bool {
        false
    }

    /// Whether the family has an explicit override of `exact_newton_joint_hessian`
    /// (or its `_with_specs` variant) that returns the *true* coupled joint
    /// Hessian rather than the trait's block-diagonal default.
    ///
    /// Default `false`.  Production families that override
    /// `exact_newton_joint_hessian` with their analytic coupled curvature must
    /// set this to `true` so the outer-REML path can trust the override
    /// downstream of `exact_newton_joint_hessian_with_specs`.  The trait can't
    /// detect override status by reflection, so this marker is the contract
    /// signal.
    fn has_explicit_joint_hessian(&self) -> bool {
        false
    }

    /// Whether the family's inner/outer solves need the full-span Jeffreys
    /// curvature `H_Φ` and score `∇Φ`.
    ///
    /// Default `true` to preserve the existing separation/near-singular
    /// robustness on every family the term was historically armed for
    /// (probit/binomial, GAMLSS location-scale, BMS, survival marginal-slope).
    ///
    /// A family overrides this to `false` when it has no
    /// separation/under-identification regime by construction — the
    /// canonical case is a continuous-response monotone-transformation
    /// family like `TransformationNormalFamily`, where the Fisher information
    /// is `O(n)` on every identified direction at every working point and
    /// the Jeffreys gate would always smooth-step to zero anyway. There the
    /// term is pure overhead: each evaluation runs `p` directional
    /// derivatives of the joint Hessian (`O(n·p²)` per call for the SCOP
    /// directional derivative), called multiple times per inner cycle and
    /// once per outer evaluation. At large scale (`p=144`, `n=20000`) the
    /// overhead is the dominant per-cycle cost and exhausts the CI budget
    /// long before the inner Newton converges, while contributing
    /// essentially zero to the converged gradient and curvature.
    fn joint_jeffreys_term_required(&self) -> bool {
        true
    }

    /// Optional Tier-B Jeffreys information matrix.
    ///
    /// Defaults to the exact joint Newton Hessian for existing families.
    /// Non-canonical Bernoulli/binomial families should override this with the
    /// expected Fisher information: Jeffreys' prior is defined from expected
    /// information, while observed information can grow in saturated
    /// misclassified tails and create an artificial prior-reward valley.
    fn joint_jeffreys_information_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_with_specs(block_states, specs)
    }

    /// First beta-directional derivative of
    /// [`Self::joint_jeffreys_information_with_specs`].
    fn joint_jeffreys_information_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_with_specs(
            block_states,
            specs,
            d_beta_flat,
        )
    }

    /// Second beta-directional derivative of
    /// [`Self::joint_jeffreys_information_with_specs`].
    fn joint_jeffreys_information_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_with_specs(
            block_states,
            specs,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    /// Optional contracted second beta-derivative of the observed joint
    /// Newton information:
    ///
    ///   ∇²_β tr(W H(β))
    ///
    /// for a fixed full-joint trace weight `W`.
    ///
    /// This is the wide-p route for Jeffreys' omitted second-directional
    /// completion. The default returns `None`, so callers fall back to the
    /// existing p(p+1)/2 pairwise `H''[e_a,e_b]` path.
    fn exact_newton_joint_contracted_trace_hessian(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        weight: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != specs.len() {
            return Err(format!(
                "exact_newton_joint_contracted_trace_hessian default: block state count {} != spec count {}",
                block_states.len(),
                specs.len()
            ));
        }
        let total = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
        if weight.dim() != (total, total) {
            return Err(format!(
                "exact_newton_joint_contracted_trace_hessian default: weight shape {:?} != ({total}, {total})",
                weight.dim()
            ));
        }
        for (block_idx, (state, spec)) in block_states.iter().zip(specs.iter()).enumerate() {
            let p_block = spec.design.ncols();
            if state.beta.len() != p_block {
                return Err(format!(
                    "exact_newton_joint_contracted_trace_hessian default: block {block_idx} beta length {} != design cols {p_block}",
                    state.beta.len()
                ));
            }
        }
        Ok(None)
    }

    /// Contracted second beta-derivative matching
    /// [`Self::joint_jeffreys_information_with_specs`]:
    ///
    ///   ∇²_β tr(W I_J(β)).
    ///
    /// Defaults to the observed-information contract above. Families that
    /// override the Jeffreys information with expected/Fisher information
    /// should override this too when they can compute the contraction in one
    /// pass; otherwise the default `None` preserves the pairwise `H''`
    /// fallback through
    /// [`Self::joint_jeffreys_information_second_directional_derivative_with_specs`].
    fn joint_jeffreys_information_contracted_trace_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        weight: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_contracted_trace_hessian(block_states, specs, weight)
    }

    /// Whether
    /// [`Self::joint_jeffreys_information_contracted_trace_hessian_with_specs`]
    /// can supply the wide-p Jeffreys completion without the pairwise `H''`
    /// fallback. Default `false` preserves the historical width cap exactly.
    fn joint_jeffreys_information_contracted_trace_hessian_available(&self) -> bool {
        false
    }

    /// Whether [`Self::joint_jeffreys_information_with_specs`] is the SAME
    /// object as the observed joint Newton Hessian
    /// (`exact_newton_joint_hessian_with_specs`).
    ///
    /// Default `true`: the trait defaults delegate the Jeffreys information
    /// to the observed quantities, so conditioning certificates obtained from
    /// observed-Hessian matvecs transfer to the Jeffreys gate exactly.
    ///
    /// Families that override the Jeffreys information with the EXPECTED
    /// Fisher information must override this to `false`. Every matrix-free
    /// "Jeffreys provably skippable" pre-check
    /// (`jeffreys_term_skippable_via_matvec`) certifies conditioning from
    /// OBSERVED Hessian matvecs; that certificate does NOT transfer when the
    /// two informations diverge. For probit-class likelihoods the observed
    /// information GROWS (~η²) on saturated misclassified rows while the
    /// expected information DECAYS, so an observed-conditioning skip would
    /// zero the Jeffreys term exactly in the saturation regime it must police
    /// (gam#1020). When this returns `false` the pre-checks are bypassed and
    /// the exact expected-information gate always runs.
    fn joint_jeffreys_information_matches_observed_hessian(&self) -> bool {
        true
    }

    /// Whether the coupled-joint inner Newton should engage its self-vanishing
    /// Levenberg–Marquardt damping `μ` on a FULL-RANK-but-ILL-CONDITIONED
    /// penalized Hessian (cond > `COND_NEWTON_SAFETY`), not only on a
    /// rank-deficient one (`nullity > 0`). Default `false` (binary / AFT /
    /// others byte-identical). Survival marginal-slope overrides to `true`
    /// (#808: full-rank but cond ≈ 5.8e6; the self-vanishing μ shapes only the
    /// trajectory, so the converged β is unbiased and the log-slope target is
    /// preserved). Survival-local by trait override so the shared spectral-range
    /// solver stays byte-identical for every other family — in particular AFT
    /// (`survival_location_scale`), whose intercept-only-scale fits can be
    /// high-cond and which a shared (unconditional) gate would regress (#735/#736).
    fn levenberg_on_ill_conditioning(&self) -> bool {
        false
    }

    /// Internal helper: do the outer-REML `_with_specs` defaults trust the
    /// inner-fit's block-diagonal-from-blocks output for this family?
    ///
    /// Trustworthy iff:
    /// - single-block (no cross-block coupling possible), or
    /// - the family has declared its blocks uncoupled in the likelihood
    ///   Hessian (`likelihood_blocks_uncoupled` ⇒ block-diagonal IS exact),
    ///   or
    /// - the family has an explicit joint-Hessian override
    ///   (`has_explicit_joint_hessian` ⇒ what we receive from
    ///   `exact_newton_joint_hessian` is the true coupled Hessian, not the
    ///   block-diagonal default).
    fn outer_default_trustworthy_for_joint_hessian(&self, specs: &[ParameterBlockSpec]) -> bool {
        specs.len() <= 1 || self.likelihood_blocks_uncoupled() || self.has_explicit_joint_hessian()
    }

    /// Optional scale-aware exact joint curvature for the outer REML calculus.
    ///
    /// Families whose exact derivatives can overflow may return a uniformly
    /// rescaled Hessian together with the metadata needed to keep every outer
    /// path consistent:
    ///
    /// - `hessian`: the scale-stabilized unpenalized joint Hessian
    /// - `rho_curvature_scale`: the uniform factor applied to every ρ-driven
    ///   penalty Hessian derivative in H-dependent trace / solve terms
    /// - `hessian_logdet_correction`: the additive correction needed to recover
    ///   `log|H_exact|` from `log|H_scaled|`
    ///
    /// The scale is evaluation-local metadata: callers must use the same
    /// factor for `H`, `dH`, `d²H`, and penalized trace operators within that
    /// evaluation, but they do not differentiate the scale itself.
    ///
    /// Families overriding this must also make
    /// `exact_newton_outer_curvature_directional_derivative[_with_specs]` and
    /// `exact_newton_outer_curvature_second_directional_derivative[_with_specs]`
    /// return derivatives in that same scaled curvature space.
    fn exact_newton_outer_curvature(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<ExactNewtonOuterCurvature>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        Ok(None)
    }

    /// Optional first directional derivative matching
    /// `exact_newton_outer_curvature`.
    fn exact_newton_outer_curvature_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative(block_states, d_beta_flat)
    }

    /// Spec-aware variant of `exact_newton_outer_curvature_directional_derivative`.
    fn exact_newton_outer_curvature_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
        self.exact_newton_outer_curvature_directional_derivative(block_states, d_beta_flat)
    }

    /// Optional second directional derivative matching
    /// `exact_newton_outer_curvature`.
    fn exact_newton_outer_curvature_second_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessiansecond_directional_derivative(
            block_states,
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }

    /// Spec-aware variant of `exact_newton_outer_curvature_second_directional_derivative`.
    fn exact_newton_outer_curvature_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
        self.exact_newton_outer_curvature_second_directional_derivative(
            block_states,
            d_beta_u_flat,
            d_beta_v_flat,
        )
    }

    /// Optional spec-aware exact first directional derivative of the joint Hessian.
    ///
    /// This is the spec-aware analogue of
    /// `exact_newton_joint_hessian_directional_derivative`. It returns the
    /// exact joint likelihood-curvature drift
    ///
    ///   D_beta H_L[u],
    ///
    /// for a flattened coefficient-space direction `u`. In the profiled
    /// Laplace gradient this appears after solving the exact joint mode
    /// response
    ///
    ///   H u_k = -A_k beta,
    ///   dot H_k = A_k + D_beta H_L[u_k].
    ///
    /// Families that can reconstruct the exact joint geometry from `specs`
    /// should override this alongside
    /// `exact_newton_joint_hessian_with_specs`.
    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Same trust dispatch as `exact_newton_joint_hessian_with_specs` —
        // the default `_directional_derivative` and `_from_working_sets`
        // both build a block-diagonal `D_β H[u]`, which silently drops the
        // cross-block `∂²L_ab/∂β_a∂β_b · u_b` rows that drive the outer
        // mode-response correction for coupled families.
        if specs.len() <= 1 || self.likelihood_blocks_uncoupled() {
            match self
                .exact_newton_joint_hessian_directional_derivative(block_states, d_beta_flat)?
            {
                Some(dh) => Ok(Some(dh)),
                None => exact_newton_joint_hessian_directional_derivative_from_working_sets(
                    self,
                    block_states,
                    specs,
                    d_beta_flat,
                ),
            }
        } else if self.has_explicit_joint_hessian()
            || self.joint_hessian_is_structurally_coupled(block_states)?
        {
            // Marked, or structurally detected coupled (see
            // `exact_newton_joint_hessian_with_specs`): the family's own
            // directional derivative is the trusted cross-block `D_β H[u]`.
            self.exact_newton_joint_hessian_directional_derivative(block_states, d_beta_flat)
        } else {
            Ok(None)
        }
    }

    /// Optional spec-aware exact second directional derivative of the joint Hessian.
    ///
    /// This is the spec-aware analogue of
    /// `exact_newton_joint_hessiansecond_directional_derivative`. For
    /// rho/rho outer Hessian entries it supplies the exact joint second-order
    /// likelihood-curvature drift
    ///
    ///   D_beta^2 H_L[u_l, u_k],
    ///
    /// which combines with
    ///
    ///   dot H_k = A_k + D_beta H_L[u_k]
    ///
    /// and the second mode response
    ///
    ///   H u_{k,l}
    ///   = -(A_k u_l + A_l u_k + B_{k,l} beta + D_beta H_L[u_l] u_k)
    ///
    /// to form
    ///
    ///   ddot H_{k,l}
    ///   = B_{k,l} + D_beta H_L[u_{k,l}] + D_beta^2 H_L[u_l, u_k].
    fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Same trust dispatch as the Hessian / first-derivative paths.  The
        // delegated `exact_newton_joint_hessiansecond_directional_derivative`
        // default is block-diagonal-from-blocks, which is silently wrong for
        // outer trace assembly on coupled families.  Unlike the lower-order
        // paths, there is no working-sets fallback — both trusted branches
        // call the same delegate, so a single helper predicate suffices.
        // The marker predicate is supplemented by the marker-free structural
        // probe so an auto-routed coupled family (one that returns a genuinely
        // off-diagonal joint Hessian without setting the explicit marker) is
        // trusted consistently across all three derivative orders.
        if !self.outer_default_trustworthy_for_joint_hessian(specs)
            && !self.joint_hessian_is_structurally_coupled(block_states)?
        {
            return Ok(None);
        }
        self.exact_newton_joint_hessiansecond_directional_derivative(
            block_states,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    /// Optional joint multi-block outer-hyper surrogate Hessian over the
    /// flattened coefficient vector.
    ///
    /// This hook exists for families whose inner working representation is
    /// block-diagonal/diagonal in `evaluate(...)`, but whose outer profiled
    /// smoothing derivatives are still joint because the fitted mode response
    /// couples blocks. The generic blockwise outer-hyper surrogate only sees
    /// per-block working sets, so it cannot recover missing cross-block
    /// curvature on its own.
    ///
    /// Families that can construct a mathematically valid joint surrogate
    /// `H_L(beta)` for the current realized `specs` may override this and the
    /// two directional derivative hooks below. Generic code then reuses the
    /// same joint rho-calculus as the exact path, but on the family-supplied
    /// surrogate curvature instead of the exact Newton Hessian.
    ///
    /// Default behavior is to reuse the spec-aware exact joint curvature when
    /// the family already provides it. That is the mathematically correct
    /// repair for the old broken multi-block blockwise surrogate path: if the
    /// family knows the full coupled Hessian and its beta-drifts, generic code
    /// should use that joint information instead of pretending per-block
    /// working sets are enough.
    fn joint_outer_hyper_surrogate_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_with_specs(block_states, specs)
    }

    /// Optional first beta-directional derivative of the joint surrogate
    /// outer-hyper Hessian.
    fn joint_outer_hyper_surrogate_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_with_specs(
            block_states,
            specs,
            d_beta_flat,
        )
    }

    /// Optional second beta-directional derivative of the joint surrogate
    /// outer-hyper Hessian.
    fn joint_outer_hyper_surrogate_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_with_specs(
            block_states,
            specs,
            d_beta_u_flat,
            d_betav_flat,
        )
    }

    /// Optional exact directional derivative of diagonal working weights along
    /// a predictor-space direction `d_eta` for `BlockWorkingSet::Diagonal`.
    ///
    /// This callback supplies the `dw` term in
    ///
    ///   D_beta J[u] = X^T diag(dw) X
    ///
    /// for diagonal working-set blocks with
    ///
    ///   J = X^T W X + S.
    ///
    /// Default `None` means no exact working-weight directional derivative is
    /// available. Exact REML/LAML derivative paths should not silently replace
    /// this with zero unless the family truly has constant working weights.
    fn diagonalworking_weights_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        idx: usize,
        arr: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(None)
    }

    /// Optional exact second directional derivative of diagonal working weights.
    ///
    /// This callback supplies the `d²w` term for static-design single-block
    /// generic fallback Hessian drift:
    ///
    ///   D²_beta H_L[u, v] = X^T diag(D²w[D eta_u, D eta_v]) X.
    ///
    /// Families with coefficient-dependent block geometry must use an exact
    /// Newton Hessian path or a joint outer path until second-order geometry
    /// hooks are available; the generic diagonal fallback will reject nonzero
    /// first-order geometry while building `d²H`.
    fn diagonalworking_weights_second_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        idx: usize,
        arr: &Array1<f64>,
        arr2: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(arr.iter().all(|v| !v.is_nan()));
        assert!(arr2.iter().all(|v| !v.is_nan()));
        Ok(None)
    }

    /// Optional exact first-order joint psi terms over the flattened
    /// coefficient vector.
    ///
    /// Families with coupled exact-joint curvature must provide psi objects in
    /// the same flattened coefficient space used by the existing joint Hessian
    /// hooks:
    ///
    ///   objective_psi = V_psi^explicit,
    ///   score_psi     = g_psi^explicit,
    ///   hessian_psi   = H_psi^explicit.
    ///
    /// Generic code then adds the realized penalty surface, solves
    ///
    ///   beta_i = -H^{-1} g_i,
    ///
    /// forms
    ///
    ///   dot H_i = H_i + D_beta H[beta_i],
    ///
    /// and plugs those objects into the unified profiled/Laplace gradient
    ///
    ///   J_i = V_i + 0.5 tr(H^{-1} dot H_i) - 0.5 partial_i log|S(theta)|_+.
    ///
    /// The current block-local exact-Newton psi hooks are not sufficient for a
    /// full joint hyper Hessian on coupled families; joint exact-joint hyper
    /// evaluation must use this flattened-coefficient hook instead.
    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        idx: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(block_specs.len() <= isize::MAX as usize);
        assert!(derivative_blocks.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        Ok(None)
    }

    /// Optional exact second-order joint psi terms over the flattened
    /// coefficient vector.
    ///
    /// For two outer coordinates theta_i, theta_j the exact profiled/Laplace
    /// Hessian uses fixed-beta second partials
    ///
    ///   V_{ij}^explicit, g_{ij}^explicit, H_{ij}^explicit.
    ///
    /// For psi/psi blocks this callback returns those explicit family terms in
    /// flattened coefficient coordinates. Generic code adds penalty
    /// contributions and profile/Laplace corrections.
    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        idx: usize,
        idx2: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(block_specs.len() <= isize::MAX as usize);
        assert!(derivative_blocks.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(idx2 < usize::MAX);
        Ok(None)
    }

    /// Optional per-evaluation workspace for exact joint ψ derivatives.
    ///
    /// Families with expensive exact ψ calculus can override this hook to
    /// precompute shared state once per outer evaluation and serve:
    ///
    /// - exact fixed-β ψψ second-order terms, and
    /// - exact mixed β/ψ Hessian drifts `D_β H_ψ[u]`
    ///
    /// from one cached workspace. Generic code falls back to the direct hooks
    /// above when no workspace is provided.
    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(block_specs.len() <= isize::MAX as usize);
        assert!(derivative_blocks.len() <= isize::MAX as usize);
        Ok(None)
    }

    /// Outer-aware variant of `exact_newton_joint_psi_workspace`.
    ///
    /// Families that consume the optional outer-only stratified row subsample
    /// (`options.outer_score_subsample`) override this method so the workspace
    /// can be constructed with the subsample mask attached. Generic families
    /// can stick with the default implementation, which simply forwards to
    /// the legacy no-options method and ignores the options. This keeps full
    /// backward compatibility with existing implementors while letting the
    /// marginal-slope families thread the subsample down into the cached
    /// per-evaluation ψ calculus.
    fn exact_newton_joint_psi_workspace_with_options(
        &self,
        states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivs: &[Vec<CustomFamilyBlockPsiDerivative>],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        assert_valid_options(options, "exact Newton joint psi workspace");
        self.exact_newton_joint_psi_workspace(states, specs, derivs)
    }

    /// Whether the family's exact joint ψ workspace should also be built for
    /// first-order ψ terms during outer gradient evaluation.
    ///
    /// Default `false` avoids forcing every family to pay workspace setup cost
    /// on gradient-only outer evaluations. Families with expensive shared state
    /// that is reused by both first- and second-order ψ calculus can opt in.
    fn exact_newton_joint_psi_workspace_for_first_order_terms(&self) -> bool {
        false
    }

    /// Optional mixed beta/psi Hessian drift D_beta H_psi[u].
    ///
    /// This is the missing T_i[u] object in the full exact joint profiled
    /// Hessian:
    ///
    ///   ddot H_{ij}
    ///   = H_{ij}
    ///     + D_beta H_i[beta_j]
    ///     + D_beta H_j[beta_i]
    ///     + D_beta H[beta_{ij}]
    ///     + D_beta^2 H[beta_i, beta_j].
    ///
    /// For i = psi_a this hook supplies D_beta H_{psi_a}[u].
    ///
    /// This direct hook is dense-only. Families that can keep the drift in an
    /// operator-backed or block-local form should expose it through
    /// `exact_newton_joint_psi_workspace()` instead.
    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        idx: usize,
        arr: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(block_specs.len() <= isize::MAX as usize);
        assert!(derivative_blocks.len() <= isize::MAX as usize);
        assert!(idx < usize::MAX);
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(None)
    }

    /// How the penalized Hessian's log-determinant and its derivatives
    /// should handle eigenvalues below the numerical-stability floor.
    ///
    /// See [`PseudoLogdetMode`].  Default: `Smooth`, the stable choice for
    /// full-rank Hessians.  Families whose model structure carries a
    /// numerical null-space direction — e.g. multi-block GAMLSS wiggle
    /// models where `q = q_0 + B(q_0)^⊤ β_w` is not identified from a
    /// threshold shift — should override to `HardPseudo` so the null
    /// direction drops out of both the REML cost and its gradient
    /// consistently, rather than leaking a spurious first-order
    /// contribution through the eigensolver's arbitrary choice of basis
    /// inside the null space.
    fn pseudo_logdet_mode(&self) -> PseudoLogdetMode {
        PseudoLogdetMode::Smooth
    }
}


/// Scope of an outer-evaluation context — distinguishes a real outer
/// derivative evaluation (where auto-subsample is allowed to install a
/// fresh stratified mask and emit phase prints) from an inner
/// coefficient line-search trial (where the family must reuse the outer
/// row measure, so auto-subsample must stay disabled).
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum EvalScope {
    /// Real outer derivative evaluation: ρ has advanced; auto-subsample
    /// install paths may build/refresh a mask keyed on this ρ.
    OuterDerivative,
    /// Inner coefficient trial (joint-Newton / line-search) at fixed
    /// outer ρ: row measure must remain identical to the surrounding
    /// outer eval, so auto-subsample must not install a fresh mask.
    InnerCoefficient,
}


/// Context published by the outer smoothing optimizer for every
/// downstream family evaluation. Carries the current outer ρ and a
/// monotonic per-outer-eval id alongside the [`EvalScope`] tag used to
/// gate auto-subsample installation. See the
/// [`BlockwiseFitOptions::outer_eval_context`] field doc for the bug
/// this prevents.
#[derive(Clone, Debug)]
pub struct OuterEvalContext {
    pub rho: Arc<Array1<f64>>,
    pub eval_id: usize,
    pub scope: EvalScope,
}


/// Stable public API for installing outer-score subsampling.
#[derive(Clone)]
pub struct BlockwiseFitOptions {
    pub inner_max_cycles: usize,
    pub inner_tol: f64,
    pub outer_max_iter: usize,
    pub outer_tol: f64,
    pub minweight: f64,
    pub ridge_floor: f64,
    /// Shared ridge semantics used by solve/quadratic/logdet terms.
    pub ridge_policy: RidgePolicy,
    /// If true, outer smoothing optimization uses a Laplace/REML-style objective:
    ///   -loglik + penalty + 0.5(log|H| - log|S|_+)
    /// where H is blockwise working curvature and S is blockwise penalty.
    pub use_remlobjective: bool,
    /// If false, the outer smoothing optimizer uses exact gradients but does
    /// not request an analytic outer Hessian from the family.
    pub use_outer_hessian: bool,
    /// If false, skip post-fit joint covariance assembly.
    pub compute_covariance: bool,
    /// Shared cap engaged during seed screening so cost-only evaluations can
    /// stop inner iterations early without affecting the full solve.
    pub screening_max_inner_iterations: Option<Arc<AtomicUsize>>,
    /// Shared cap engaged during regular outer iterations. Unlike screening,
    /// this is only a budget: capped solves still have to earn the ordinary
    /// KKT certificate before derivatives may be exposed.
    pub outer_inner_max_iterations: Option<Arc<AtomicUsize>>,
    /// Optional line-search objective ceiling for lazy log-likelihood-only
    /// evaluations. Families whose per-row log-likelihood contributions are
    /// non-positive may stop once the partial negative log-likelihood is already
    /// above this ceiling, because the unvisited rows cannot improve the trial
    /// objective enough to be accepted. Default `None` preserves exact full-sum
    /// behavior and is the only mode used outside backtracking rejection tests.
    pub early_exit_threshold: Option<f64>,
    /// Stable public API for installing outer-score subsampling.
    ///
    /// Optional stratified row subsample used by outer-only score/gradient
    /// passes. When `Some(s)`, outer score/gradient hot loops should iterate
    /// only over `s.rows` and multiply each contribution by that row's
    /// Horvitz-Thompson inverse-inclusion weight. Inner-PIRLS and final
    /// covariance passes always run on the full data, so this field is
    /// consulted only by outer-only call sites. Default `None` preserves the
    /// full-data behavior. Wrapping in `Arc` keeps `Clone` cheap across the
    /// many places `BlockwiseFitOptions` is duplicated per-eval.
    pub outer_score_subsample:
        Option<Arc<crate::families::marginal_slope_shared::OuterScoreSubsample>>,
    /// Gate for marginal-slope families to auto-derive a stratified
    /// outer-score subsample at large scale (see
    /// [`crate::families::marginal_slope_shared::auto_outer_score_subsample`]).
    ///
    /// **Default `true`.** Auto-subsampling makes the early rho-gradient
    /// evaluations unbiased stochastic estimators with bounded relative
    /// variance (≈ 1 % at the conservative defaults), then the family switches
    /// back to full-data gradients for the remaining outer iterations. That
    /// keeps large marginal-slope fits fast during the high-motion part of the
    /// trajectory while preserving the default tight `outer_tol` polish on
    /// exact gradients. For small datasets the auto path declines to install a
    /// mask and the fit remains full-data throughout.
    ///
    /// When `outer_score_subsample` is already `Some(...)` the auto
    /// path is bypassed entirely (caller-provided masks always win).
    pub auto_outer_subsample: bool,
    /// Outer-evaluation context populated by the smoothing optimizer at
    /// the top of each real outer derivative evaluation. Used by
    /// auto-subsample install paths to key the stratified mask on the
    /// outer ρ rather than the inner β proxy: during the inner trust-
    /// region / coefficient line search β changes on every trial step,
    /// so keying on β re-fires phase prints (and re-shuffles the mask)
    /// inside a single outer eval. Keying on (rho, eval_id) instead
    /// keeps the mask stable across the inner Newton at one ρ, and
    /// suppresses auto-subsample entirely on inner trial evaluations via
    /// the [`EvalScope::InnerCoefficient`] tag set by
    /// [`coefficient_line_search_options`].
    ///
    /// `None` preserves legacy behavior (no context — install paths fall
    /// back to "no auto-subsample"). Default `None`.
    pub outer_eval_context: Option<OuterEvalContext>,
    /// Optional persistent warm-start cache session. When `Some`, the
    /// outer smoothing optimizer consults the on-disk cache before
    /// starting (to seed θ from the last accepted iterate) and writes
    /// checkpoints + a final entry on completion. When `None`, the fit
    /// runs cold and writes nothing — the default for unit tests and
    /// any caller that pinned a deterministic optimum.
    ///
    /// The session is opened at the workflow-level `fit_model`
    /// dispatcher so every family flows through one chokepoint; family
    /// code never has to remember to wire it. This mirrors the standard
    /// REML cache wiring in `solver/estimate.rs:2701`.
    pub cache_session: Option<Arc<crate::cache::Session>>,
    /// Optional mirror sessions that receive a copy of the final-result
    /// finalize() write. Used by the workflow dispatcher to broadcast a
    /// converged ρ to additional keyspace(s) — notably the data-
    /// independent seed prefix — so future fits with related structure
    /// can warm-start from this run. Writes still pass through the session
    /// rate limiter, so mirroring checkpoints does not add unbounded I/O.
    pub cache_mirror_sessions: Vec<Arc<crate::cache::Session>>,
    /// Optional bundle of cross-block (full-width) penalties, paired with
    /// their current `log λ` values from the outer ρ vector. When `Some`,
    /// the inner joint-Newton primitives add the contributions
    ///
    /// * objective: `½ Σ_j exp(ρ_j) βᵀ S_j β`
    /// * gradient:  `Σ_j exp(ρ_j) S_j β`
    /// * Hessian:   `Σ_j exp(ρ_j) S_j`
    ///
    /// in addition to the per-block penalty stack assembled from
    /// `ParameterBlockSpec.penalties`. The per-block path is unchanged.
    /// `None` preserves legacy behaviour for every existing caller.
    pub joint_penalties: Option<Arc<crate::families::joint_penalty::JointPenaltyBundle>>,
    /// Whether the outer smoothing optimizer screens the explicit
    /// `initial_rho` seed through the seed-screening cascade before the
    /// solver starts.
    ///
    /// **Default `true`** — the general path benefits from ranking the
    /// initial seed against the generated exploration seeds via cheap
    /// capped proxy fits.
    ///
    /// A caller sets this `false` when `initial_rho` is already the correct,
    /// identified optimum for its regime so that re-screening it adds only
    /// cost. The survival location-scale constant-scale (parametric-AFT)
    /// path uses this: its time-warp ρ seed is pinned AT the inner ρ box
    /// bound (the affine-baseline limit), where the REML/LAML profile is a
    /// dead-flat unidentified ridge. Running the screening cascade there
    /// drives each proxy fit (and, when every capped stage collapses to
    /// non-finite cost, the uncapped final stage) into a full inner solve on
    /// the near-singular flat Hessian — the source of the multi-minute
    /// no-iteration-log stall (#736, #735, #721). Skipping screening lets the
    /// already-correct seed flow straight to the outer solver, which certifies
    /// box-constraint stationarity at iteration 0. Genuinely flexible regimes
    /// (smooth scale / spatial) leave this `true` and keep full screening.
    pub screen_initial_rho: bool,
    /// Set ONLY while the inner solve is invoked from the seed-screening proxy
    /// (`custom_family_seed_screening_proxy_labeled`), which RANKS candidate
    /// seeds by their penalized objective and never produces the final fit.
    ///
    /// When `true`, the inner joint-Newton skips the full per-axis
    /// Jeffreys/Firth curvature (`custom_family_joint_jeffreys_term`'s
    /// `for k in 0..p` directional-derivative loop, O(p · per-axis-Hdot) per
    /// cycle), keeping ONLY the cheap value-only Jeffreys term
    /// (`custom_family_joint_jeffreys_value`, one reduced-info eigendecomposition)
    /// in the screening score. The per-axis gradient/curvature is what the inner
    /// Newton step needs to *converge* a near-separating fit; the screening proxy
    /// is capped and only ranks, so it does not need step convergence — it needs
    /// a finite, separation-aware score cheaply. For a K-block coupled family
    /// (Dirichlet/multinomial) each per-axis directional derivative is itself
    /// O(K²·n·p), so running the full term for every cascade candidate over the
    /// joint width `p` is the wrong cost class and made the coupled fit
    /// non-completing during screening alone (gam#729/#808). The actual fit
    /// (after a seed is selected) runs with this `false`, so the load-bearing
    /// Firth curvature is fully present where it matters.
    ///
    /// **Default `false`** — only the screening proxy sets it `true`.
    pub seed_screening: bool,
}


pub const DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES: usize = 1200;


impl Default for BlockwiseFitOptions {
    fn default() -> Self {
        Self {
            // Large-scale custom-family marginal-slope fits can have a
            // long, monotone joint-Newton tail: objective and step size keep
            // shrinking, but the exact KKT residual may need several hundred
            // additional cycles after the old 300-cycle cap. The outer
            // REML/LAML derivative path is correct only at a stationary inner
            // mode, so a merely descended iterate must not be accepted as
            // converged. Use a production-sized cap by default and rely on the
            // KKT/objective certificates to exit early for well-conditioned
            // Gaussian, logistic, and small-n fits.
            inner_max_cycles: DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES,
            inner_tol: 1e-6,
            outer_max_iter: 60,
            outer_tol: 1e-5,
            minweight: CUSTOM_FAMILY_WEIGHT_FLOOR,
            // `ridge_floor` is an ExplicitPrior in the canonical
            // stabilization ledger taxonomy (`StabilizationKind::ExplicitPrior`):
            // its δ enters the quadratic term, the Laplace Hessian, and the
            // penalty log-determinant — `ridge_policy` below is the live
            // policy that confirms which terms it lands in. The default
            // pos-part policy enables every inclusion flag, so callers
            // wanting solver-only damping should construct a custom policy
            // (or, preferably, a `StabilizationLedger::numerical_perturbation`)
            // rather than reusing this field.
            ridge_floor: CUSTOM_FAMILY_RIDGE_FLOOR,
            ridge_policy: RidgePolicy::explicit_stabilization_pospart(),
            use_remlobjective: true,
            // Default ON: families expose exact outer Hessians whenever their
            // analytic dense or operator representation is implemented.
            use_outer_hessian: true,
            compute_covariance: false,
            screening_max_inner_iterations: None,
            outer_inner_max_iterations: None,
            seed_screening: false,
            early_exit_threshold: None,
            outer_score_subsample: None,
            auto_outer_subsample: true,
            outer_eval_context: None,
            cache_session: None,
            cache_mirror_sessions: Vec::new(),
            joint_penalties: None,
            screen_initial_rho: true,
        }
    }
}


#[derive(Clone)]
pub struct BlockwiseInnerResult {
    pub block_states: Vec<ParameterBlockState>,
    pub active_sets: Vec<Option<Vec<usize>>>,
    pub log_likelihood: f64,
    pub penalty_value: f64,
    pub cycles: usize,
    pub converged: bool,
    pub block_logdet_h: f64,
    pub block_logdet_s: f64,
    /// Cached assembled penalty matrices S(ρ) = Σ_k exp(ρ_k) S_k per block.
    /// Avoids redundant re-assembly in the outer objective evaluation.
    pub s_lambdas: Vec<Array2<f64>>,
    pub joint_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    /// Projected KKT residual at the converged inner iterate, propagated to
    /// the unified evaluator's `InnerAssembly::kkt_residual` for the
    /// outer REML/LAML scoring path. `None` when the solver path doesn't
    /// produce a typed KKT diagnostic (blockwise NR fallback, eager-stop).
    pub kkt_residual: Option<crate::estimate::reml::unified::ProjectedKktResidual>,
    /// Active linear-inequality constraint rows at the converged inner
    /// iterate. When `Some`, the unified evaluator builds the
    /// constraint-aware kernel `K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S`
    /// for per-coordinate mode responses `v_k = ∂β/∂ρ_k`.
    pub active_constraints:
        Option<Arc<crate::estimate::reml::unified::ActiveLinearConstraintBlock>>,
}


impl std::fmt::Debug for BlockwiseInnerResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockwiseInnerResult")
            .field("block_states", &self.block_states)
            .field("active_sets", &self.active_sets)
            .field("log_likelihood", &self.log_likelihood)
            .field("penalty_value", &self.penalty_value)
            .field("cycles", &self.cycles)
            .field("converged", &self.converged)
            .field("block_logdet_h", &self.block_logdet_h)
            .field("block_logdet_s", &self.block_logdet_s)
            .field("s_lambdas", &self.s_lambdas)
            .field(
                "joint_workspace",
                &self.joint_workspace.as_ref().map(|_| "<workspace>"),
            )
            .finish()
    }
}


#[derive(Clone)]
struct ConstrainedWarmStart {
    rho: Array1<f64>,
    block_beta: Vec<Array1<f64>>,
    active_sets: Vec<Option<Vec<usize>>>,
    cached_inner: Option<CachedInnerMode>,
}


#[derive(Clone)]
struct CachedInnerMode {
    log_likelihood: f64,
    penalty_value: f64,
    cycles: usize,
    converged: bool,
    block_logdet_h: f64,
    block_logdet_s: f64,
    joint_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    kkt_residual: Option<crate::estimate::reml::unified::ProjectedKktResidual>,
    active_constraints: Option<Arc<crate::estimate::reml::unified::ActiveLinearConstraintBlock>>,
}


fn screened_outer_warm_start<'a>(
    warm_start: Option<&'a ConstrainedWarmStart>,
    rho: &Array1<f64>,
) -> Option<&'a ConstrainedWarmStart> {
    warm_start.filter(|seed| seed.rho.len() == rho.len())
}


fn warm_start_matches_block_log_lambdas(
    seed: &ConstrainedWarmStart,
    block_log_lambdas: &[Array1<f64>],
) -> bool {
    let expected = block_log_lambdas
        .iter()
        .map(|values| values.len())
        .sum::<usize>();
    if seed.rho.len() != expected {
        return false;
    }
    let mut offset = 0usize;
    for block in block_log_lambdas {
        let end = offset + block.len();
        if seed.rho.slice(s![offset..end]) != block.view() {
            return false;
        }
        offset = end;
    }
    true
}


fn cached_inner_mode_from_result(result: &BlockwiseInnerResult) -> CachedInnerMode {
    CachedInnerMode {
        log_likelihood: result.log_likelihood,
        penalty_value: result.penalty_value,
        cycles: result.cycles,
        converged: result.converged,
        block_logdet_h: result.block_logdet_h,
        block_logdet_s: result.block_logdet_s,
        joint_workspace: result.joint_workspace.clone(),
        kkt_residual: result.kkt_residual.clone(),
        active_constraints: result.active_constraints.clone(),
    }
}


fn constrained_warm_start_from_inner(
    rho: &Array1<f64>,
    inner: &BlockwiseInnerResult,
) -> ConstrainedWarmStart {
    ConstrainedWarmStart {
        rho: rho.clone(),
        block_beta: inner
            .block_states
            .iter()
            .map(|state| state.beta.clone())
            .collect(),
        active_sets: inner.active_sets.clone(),
        cached_inner: Some(cached_inner_mode_from_result(inner)),
    }
}


fn constrained_warm_start_from_cached_beta(
    rho_dim: usize,
    specs: &[ParameterBlockSpec],
    beta: &Array1<f64>,
) -> Result<ConstrainedWarmStart, EstimationError> {
    let expected = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    if beta.len() != expected {
        crate::bail_invalid_estim!(
            "cached inner beta has length {}, but custom-family blocks require length {}",
            beta.len(),
            expected
        );
    }
    crate::families::marginal_slope_shared::bail_if_cached_beta_non_finite(beta)?;

    let mut offset = 0usize;
    let mut block_beta = Vec::with_capacity(specs.len());
    for spec in specs {
        let end = offset + spec.design.ncols();
        block_beta.push(beta.slice(s![offset..end]).to_owned());
        offset = end;
    }

    Ok(ConstrainedWarmStart {
        rho: Array1::zeros(rho_dim),
        block_beta,
        active_sets: vec![None; specs.len()],
        cached_inner: None,
    })
}


fn inner_penalized_objective(
    inner: &BlockwiseInnerResult,
    include_logdet_h: bool,
    include_logdet_s: bool,
    context: &str,
) -> Result<f64, String> {
    let reml_term = if include_logdet_h {
        0.5 * inner.block_logdet_h
    } else {
        0.0
    } - if include_logdet_s {
        0.5 * inner.block_logdet_s
    } else {
        0.0
    };
    checked_penalizedobjective(
        inner.log_likelihood,
        inner.penalty_value,
        reml_term,
        context,
    )
}


fn nonconverged_outer_efs_result(
    inner: &BlockwiseInnerResult,
    rho: &Array1<f64>,
    theta_dim: usize,
    include_logdet_h: bool,
    include_logdet_s: bool,
    context: &str,
) -> Result<
    (
        crate::solver::outer_strategy::EfsEval,
        ConstrainedWarmStart,
        bool,
    ),
    String,
> {
    Ok((
        crate::solver::outer_strategy::EfsEval {
            cost: inner_penalized_objective(inner, include_logdet_h, include_logdet_s, context)?,
            steps: vec![0.0; theta_dim],
            beta: None,
            psi_gradient: None,
            psi_indices: None,
            inner_hessian_scale: None,
            logdet_enclosure_gap: None,
        },
        constrained_warm_start_from_inner(rho, inner),
        false,
    ))
}


fn warm_start_without_cached_inner_for_psi_derivatives(
    warm_start: Option<&ConstrainedWarmStart>,
    has_psi_derivatives: bool,
) -> Option<ConstrainedWarmStart> {
    if !has_psi_derivatives {
        return None;
    }
    warm_start.cloned().map(|mut warm| {
        warm.cached_inner = None;
        warm
    })
}


/// Helper struct mirroring the old `BlockwiseFitResultParts`.
pub struct BlockwiseFitResultParts {
    pub block_states: Vec<ParameterBlockState>,
    pub log_likelihood: f64,
    pub log_lambdas: Array1<f64>,
    pub lambdas: Array1<f64>,
    pub covariance_conditional: Option<Array2<f64>>,
    pub stable_penalty_term: f64,
    pub penalized_objective: f64,
    pub outer_iterations: usize,
    /// `None` = no gradient measured at termination (cache-hit, gradient-free,
    /// or trivial early-exit); `Some(g)` = measured norm. `outer_converged`
    /// is the authoritative convergence signal.
    pub outer_gradient_norm: Option<f64>,
    /// First-order optimality certificate from the outer smoothing solve
    /// (#934); `None` when no outer ran (fixed-λ, one-cycle probe) or the
    /// audit could not evaluate.
    pub criterion_certificate: Option<crate::solver::outer_strategy::CriterionCertificate>,
    pub inner_cycles: usize,
    pub outer_converged: bool,
    pub geometry: Option<FitGeometry>,
    /// Effective degrees of freedom computed by the caller in the *reduced*
    /// (canonical) coefficient space, where the penalized Hessian is full rank,
    /// as `(edf_total, edf_by_penalty, block_edf)`. The trace edf is invariant
    /// under the canonical reparameterization, so computing it in the reduced
    /// space and reporting it on the raw fit is exact — and it avoids the
    /// `tr((H_raw + εI)⁻¹ S_raw)` blow-up that a rank-deficient raw-lifted
    /// Hessian (zero rows/cols on canonicalization-dropped directions) would
    /// otherwise inject. `None` when the caller has no reduced geometry (e.g.
    /// the one-cycle inner probe), in which case `blockwise_fit_from_parts`
    /// falls back to computing edf from whatever geometry it was handed.
    pub precomputed_edf: Option<(f64, Vec<f64>, Vec<f64>)>,
}


fn validate_parameter_block_state_finiteness(
    label: &str,
    state: &ParameterBlockState,
) -> Result<(), String> {
    validate_all_finite_estimation(&format!("{label}.beta"), state.beta.iter().copied())
        .map_err(|e| e.to_string())?;
    validate_all_finite_estimation(&format!("{label}.eta"), state.eta.iter().copied())
        .map_err(|e| e.to_string())?;
    Ok(())
}


fn validate_lambda_pair_consistency(
    log_lambdas: &Array1<f64>,
    lambdas: &Array1<f64>,
    label: &str,
) -> Result<(), String> {
    if log_lambdas.len() != lambdas.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "{label} length mismatch: log_lambdas={}, lambdas={}",
                log_lambdas.len(),
                lambdas.len()
            ),
        }
        .into());
    }
    for (idx, (&log_lambda, &lambda)) in log_lambdas.iter().zip(lambdas.iter()).enumerate() {
        let expected = log_lambda.exp();
        let tolerance = 1e-10 * expected.abs().max(1.0);
        if (lambda - expected).abs() > tolerance {
            return Err(format!(
                "{label}[{idx}] inconsistent with exp(log_lambda): got {lambda}, expected {expected}",
            ));
        }
    }
    Ok(())
}


/// Effective degrees of freedom for a converged blockwise custom-family fit,
/// computed from the joint penalized Hessian `H = X'W_HX + S(λ)` and the
/// per-penalty matrices `S_k` exactly as the standard GAM path and mgcv do:
///
/// ```text
/// edf_total   = p − Σ_k λ_k · tr(H⁻¹ S_k)
/// edf_penalty = (rank_k − λ_k · tr(H⁻¹ S_k))   clamped to [0, rank_k]
/// ```
///
/// `S_k` here is the *unscaled* penalty (its `λ_k` factor is applied here), and
/// each `S_k.to_dense()` is already embedded in the joint `p × p` coefficient
/// layout (the Blockwise / Kronecker variants place their local block at the
/// correct column range), so the trace solve runs in the full joint space and
/// no per-block offset bookkeeping is required.
///
/// The custom-family path (CTN transformation-normal, Dirichlet, …) builds its
/// fit through `blockwise_fit_from_parts` and previously left `inference` at
/// `None`, so `edf_total` was unavailable for every custom family even though
/// the converged geometry already carries the penalized Hessian. This mirrors
/// the survival-path repair (`survival_transformation_edf`, #565) for the
/// blockwise engine: the same trace formula, factorized with the same
/// ridge-retry stabilization so a marginally indefinite Hessian at a boundary
/// optimum still yields a usable trace instead of dropping inference.
///
/// `edf_penalty` is returned aligned 1:1 with the flattened `lambdas`
/// (one entry per penalty across all blocks), matching the
/// `FitInference::edf_by_block` ↔ `lambdas` length invariant. The per-block
/// aggregate edf (for `FittedBlock::edf`) is the sum of that block's penalty
/// edfs, with an unpenalized block contributing its full column count.
fn custom_family_blockwise_edf(
    penalized_hessian: &Array2<f64>,
    specs: &[ParameterBlockSpec],
    lambdas: &ndarray::ArrayView1<'_, f64>,
) -> Result<(f64, Vec<f64>, Vec<f64>), String> {
    let p = penalized_hessian.nrows();
    let total_cols: usize = specs.iter().map(|s| s.design.ncols()).sum();
    if penalized_hessian.ncols() != p || total_cols != p {
        return Err(format!(
            "custom-family edf: penalized Hessian {}x{} inconsistent with total block width {}",
            penalized_hessian.nrows(),
            penalized_hessian.ncols(),
            total_cols
        ));
    }
    let expected_rho: usize = specs.iter().map(|s| s.penalties.len()).sum();
    if lambdas.len() != expected_rho {
        return Err(format!(
            "custom-family edf: lambdas length {} does not match total penalty count {}",
            lambdas.len(),
            expected_rho
        ));
    }

    let h_sym = SymmetricMatrix::Dense(penalized_hessian.clone());
    // Sparse-aware factorization with ridge retry (mirrors estimate.rs and
    // survival_transformation_edf): a boundary-constrained optimum can leave
    // the penalized Hessian marginally indefinite, in which case we add the
    // smallest diagonal shift that restores definiteness so the trace solve
    // succeeds rather than dropping inference for the whole fit.
    let factor = {
        let scale = h_sym.max_abs_diag();
        let min_step = scale * 1e-10;
        let mut ridge = 0.0_f64;
        let mut attempts = 0_usize;
        loop {
            let candidate = if ridge > 0.0 {
                h_sym.addridge(ridge).unwrap_or_else(|_| h_sym.clone())
            } else {
                h_sym.clone()
            };
            if let Ok(f) = candidate.factorize() {
                break f;
            }
            attempts += 1;
            if attempts >= 8 {
                return Err(
                    "custom-family edf: penalized Hessian could not be factorized".to_string(),
                );
            }
            ridge = if ridge <= 0.0 { min_step } else { ridge * 10.0 };
        }
    };

    let mut edf_by_penalty = vec![0.0_f64; expected_rho];
    let mut block_edf = Vec::with_capacity(specs.len());
    let mut total_trace = 0.0_f64;
    let mut penalty_offset = 0usize;
    let mut block_col_start = 0usize;
    for spec in specs.iter() {
        let block_cols = spec.design.ncols();
        let mut block_edf_acc = block_cols as f64;
        for (local_k, penalty) in spec.penalties.iter().enumerate() {
            let global_k = penalty_offset + local_k;
            let lambda = lambdas[global_k];
            // Embed S_k into the full p×p joint layout. `PenaltyMatrix::to_dense`
            // returns the *local* block matrix for the `Dense` variant but the
            // already-embedded full-width matrix for `Blockwise`/`Kronecker`, so
            // dispatch on the materialized dimension: a local (block_cols-wide)
            // penalty is placed at this block's column range, a full-width
            // penalty is used as-is (mirrors `survival_transformation_edf`'s
            // explicit block placement).
            let s_local = penalty.to_dense();
            let mut s_full = Array2::<f64>::zeros((p, p));
            if s_local.nrows() == p && s_local.ncols() == p {
                s_full.assign(&s_local);
            } else if s_local.nrows() == block_cols && s_local.ncols() == block_cols {
                let r = block_col_start..block_col_start + block_cols;
                s_full.slice_mut(ndarray::s![r.clone(), r]).assign(&s_local);
            } else {
                return Err(format!(
                    "custom-family edf: penalty {global_k} materialized to {}x{}, expected {p}x{p} or {block_cols}x{block_cols}",
                    s_local.nrows(),
                    s_local.ncols()
                ));
            }
            // tr(H⁻¹ S_k) via H Z = S_k, summing the diagonal of Z.
            let z = factor.solvemulti(&s_full).map_err(|e| {
                format!("custom-family edf trace solve failed for penalty {global_k}: {e}")
            })?;
            let mut trace = 0.0_f64;
            for d in 0..p {
                trace += z[[d, d]];
            }
            let lam_trace = if lambda > 0.0 { lambda * trace } else { 0.0 };
            total_trace += lam_trace;
            // Per-penalty edf is bounded by the columns this penalty acts on,
            // i.e. its block's column count (a `Blockwise` penalty reports the
            // full joint width from `dim()`, so cap at `block_cols`, not `dim()`).
            let penalty_cols = block_cols as f64;
            let edf_k = (penalty_cols - lam_trace).clamp(0.0, penalty_cols);
            edf_by_penalty[global_k] = edf_k;
            // The block's edf is the column count minus the total trace this
            // block's penalties spend (so multiple penalties on one block
            // compose), clamped to the block's column count.
            block_edf_acc -= lam_trace;
        }
        block_edf.push(block_edf_acc.clamp(0.0, block_cols as f64));
        penalty_offset += spec.penalties.len();
        block_col_start += block_cols;
    }

    let edf_total = (p as f64 - total_trace).clamp(0.0, p as f64);
    if !edf_total.is_finite()
        || edf_by_penalty.iter().any(|v| !v.is_finite())
        || block_edf.iter().any(|v| !v.is_finite())
    {
        return Err("custom-family edf: non-finite effective degrees of freedom".to_string());
    }
    Ok((edf_total, edf_by_penalty, block_edf))
}


/// Compute reduced-space effective degrees of freedom for a converged fit,
/// to be carried through `BlockwiseFitResultParts::precomputed_edf`.
///
/// The reduced (canonical) geometry's penalized Hessian is full rank and its
/// `reduced_specs` carry the pulled-back penalties `T_iᵀ S_k T_i`, so the trace
/// edf is computed exactly here (no rank-deficiency ridge bias). Because the
/// trace edf is invariant under the canonical reparameterization, the resulting
/// `edf_total` / per-penalty / per-block values are the same as they would be
/// in the raw basis and are reported directly on the lifted raw fit. Returns
/// `None` when no reduced geometry is available, so the caller can leave
/// `precomputed_edf` unset (and the raw-geometry fallback applies).
fn reduced_blockwise_edf(
    reduced_geometry: Option<&FitGeometry>,
    canonical: &crate::solver::identifiability_canonical::CanonicalSpecs,
    lambdas: &Array1<f64>,
) -> Option<(f64, Vec<f64>, Vec<f64>)> {
    let geom = reduced_geometry?;
    match custom_family_blockwise_edf(
        geom.penalized_hessian.as_array(),
        &canonical.reduced_specs,
        &lambdas.view(),
    ) {
        Ok(triple) => Some(triple),
        Err(err) => {
            log::warn!(
                "[custom-family inference] reduced-space effective degrees of freedom unavailable: {err}"
            );
            None
        }
    }
}


/// Build a `UnifiedFitResult` from blockwise-specific fields.
pub fn blockwise_fit_from_parts(
    parts: BlockwiseFitResultParts,
    specs: &[ParameterBlockSpec],
) -> Result<crate::solver::estimate::UnifiedFitResult, String> {
    let BlockwiseFitResultParts {
        block_states,
        log_likelihood,
        log_lambdas,
        lambdas,
        covariance_conditional,
        stable_penalty_term,
        penalized_objective,
        outer_iterations,
        outer_gradient_norm,
        criterion_certificate,
        inner_cycles,
        outer_converged,
        geometry,
        precomputed_edf,
    } = parts;

    if block_states.is_empty() {
        return Err(CustomFamilyError::UnsupportedConfiguration {
            reason: "blockwise fit requires at least one block state".to_string(),
        }
        .into());
    }
    ensure_finite_scalar_estimation("blockwise_fit.log_likelihood", log_likelihood)
        .map_err(|e| e.to_string())?;
    validate_all_finite_estimation("blockwise_fit.log_lambdas", log_lambdas.iter().copied())
        .map_err(|e| e.to_string())?;
    validate_all_finite_estimation("blockwise_fit.lambdas", lambdas.iter().copied())
        .map_err(|e| e.to_string())?;
    validate_lambda_pair_consistency(&log_lambdas, &lambdas, "blockwise_fit.lambdas")?;
    ensure_finite_scalar_estimation("blockwise_fit.penalized_objective", penalized_objective)
        .map_err(|e| e.to_string())?;
    ensure_finite_scalar_estimation("blockwise_fit.stable_penalty_term", stable_penalty_term)
        .map_err(|e| e.to_string())?;
    if let Some(g) = outer_gradient_norm {
        ensure_finite_scalar_estimation("blockwise_fit.outer_gradient_norm", g)
            .map_err(|e| e.to_string())?;
    }

    if block_states.len() != specs.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "blockwise_fit.block_states length ({}) does not match specs length ({})",
                block_states.len(),
                specs.len()
            ),
        }
        .into());
    }
    let n = specs[0].design.nrows();
    let total_p = block_states
        .iter()
        .map(|state| state.beta.len())
        .sum::<usize>();
    for (idx, state) in block_states.iter().enumerate() {
        validate_parameter_block_state_finiteness(
            &format!("blockwise_fit.block_states[{idx}]"),
            state,
        )?;
        let expected_rows = specs[idx].solver_design().nrows();
        if state.eta.len() != expected_rows {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "blockwise_fit.block_states[{idx}] eta length mismatch: got {}, expected {} (solver design rows)",
                state.eta.len(),
                expected_rows
            ) }.into());
        }
    }

    if let Some(cov) = covariance_conditional.as_ref() {
        validate_all_finite_estimation("blockwise_fit.covariance_conditional", cov.iter().copied())
            .map_err(|e| e.to_string())?;
        let (rows, cols) = cov.dim();
        if rows != total_p || cols != total_p {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "blockwise_fit.covariance_conditional must be {}x{}, got {}x{}",
                    total_p, total_p, rows, cols
                ),
            }
            .into());
        }
    }

    if let Some(geom) = geometry.as_ref() {
        geom.validate_numeric_finiteness()
            .map_err(|e| e.to_string())?;
        let (rows, cols) = geom.penalized_hessian.dim();
        if rows != total_p || cols != total_p {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "blockwise_fit.geometry.penalized_hessian must be {}x{}, got {}x{}",
                    total_p, total_p, rows, cols
                ),
            }
            .into());
        }
        let geom_len = geom.working_weights.len();
        if geom_len != geom.working_response.len() {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "blockwise_fit.geometry working vector length mismatch: weights={}, response={}",
                geom.working_weights.len(),
                geom.working_response.len(),
            ) }.into());
        }
        if geom_len != n && (n == 0 || geom_len % n != 0) {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "blockwise_fit.geometry.working_weights length mismatch: got {geom_len}, expected {n} or a stacked multiple of {n}",
            ) }.into());
        }
        if geom.working_response.len() != n && (n == 0 || geom.working_response.len() % n != 0) {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "blockwise_fit.geometry.working_response length mismatch: got {}, expected {n} or a stacked multiple of {n}",
                geom.working_response.len(),
            ) }.into());
        }
    }

    // Build unified blocks from the blockwise states.
    use crate::solver::estimate::{FittedBlock, FittedLinkState, UnifiedFitResultParts};
    let expected_rho: usize = specs.iter().map(|s| s.penalties.len()).sum();
    if lambdas.len() != expected_rho {
        return Err(CustomFamilyError::DimensionMismatch { reason: format!(
            "blockwise_fit.lambdas length ({}) does not match sum of per-block penalty counts ({})",
            lambdas.len(),
            expected_rho
        ) }.into());
    }
    // Effective degrees of freedom and the inference block. When the
    // converged geometry carries the joint penalized Hessian we compute the
    // mgcv trace edf `p − Σ_k λ_k·tr(H⁻¹ S_k)` here so every custom-family fit
    // (CTN transformation-normal, Dirichlet, …) reports `edf_total` /
    // per-block `edf` like the standard GAM path, instead of leaving inference
    // unpopulated. A factorization failure is non-fatal: the fit still returns
    // with `edf=0`/`inference=None` rather than aborting, but in practice the
    // ridge-retry inside `custom_family_blockwise_edf` recovers any boundary
    // indefiniteness.
    let (edf_total_opt, edf_by_penalty, block_edf): (Option<f64>, Vec<f64>, Vec<f64>) =
        match precomputed_edf {
            // Reduced-space edf supplied by the caller (the principled path:
            // the trace is computed where the Hessian is full rank, then
            // reported on the raw fit — exact because the trace edf is
            // reparameterization-invariant).
            Some((edf_total, edf_by_penalty, block_edf)) => {
                (Some(edf_total), edf_by_penalty, block_edf)
            }
            // Fallback: compute from whatever geometry we were handed. Used
            // only when the caller did not precompute (no reduced geometry);
            // the ridge-retry factorization makes this robust to a marginally
            // indefinite Hessian.
            None => match geometry.as_ref() {
                Some(geom) => {
                    match custom_family_blockwise_edf(
                        geom.penalized_hessian.as_array(),
                        specs,
                        &lambdas.view(),
                    ) {
                        Ok((edf_total, edf_by_penalty, block_edf)) => {
                            (Some(edf_total), edf_by_penalty, block_edf)
                        }
                        Err(err) => {
                            log::warn!(
                                "[custom-family inference] effective degrees of freedom unavailable: {err}"
                            );
                            (None, Vec::new(), vec![0.0; block_states.len()])
                        }
                    }
                }
                None => (None, Vec::new(), vec![0.0; block_states.len()]),
            },
        };

    let mut lambda_offset = 0usize;
    let blocks: Vec<FittedBlock> = block_states
        .iter()
        .enumerate()
        .map(|(i, bs)| {
            let role = custom_family_block_role(&specs[i].name, i, block_states.len());
            let k = specs[i].penalties.len();
            let block_lambdas = lambdas
                .slice(s![lambda_offset..lambda_offset + k])
                .to_owned();
            lambda_offset += k;
            FittedBlock {
                beta: bs.beta.clone(),
                role,
                edf: block_edf.get(i).copied().unwrap_or(0.0),
                lambdas: block_lambdas,
            }
        })
        .collect();
    let deviance = -2.0 * log_likelihood;

    // Assemble the inference block from the converged geometry. CTN and other
    // custom families estimate their own likelihood scale, so the penalized
    // Hessian is reported unscaled (dispersion = 1) — the EDF trace is
    // dispersion-free, and downstream covariance scaling pairs `H` with the
    // family's own dispersion where needed.
    let inference = match (edf_total_opt, geometry.as_ref()) {
        (Some(edf_total), Some(geom)) => Some(crate::solver::estimate::FitInference {
            edf_by_block: edf_by_penalty,
            edf_total,
            smoothing_correction: None,
            penalized_hessian: geom.penalized_hessian.clone(),
            working_weights: geom.working_weights.clone(),
            working_response: geom.working_response.clone(),
            reparam_qs: None,
            dispersion: crate::solver::estimate::Dispersion::Known(1.0),
            beta_covariance: None,
            beta_standard_errors: None,
            beta_covariance_corrected: None,
            beta_standard_errors_corrected: None,
            beta_covariance_frequentist: None,
            coefficient_influence: None,
            weighted_gram: None,
            bias_correction_beta: None,
        }),
        _ => None,
    };

    crate::solver::estimate::UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
        blocks,
        log_lambdas: log_lambdas.clone(),
        lambdas: lambdas.clone(),
        likelihood_family: None,
        likelihood_scale: crate::types::LikelihoodScaleMetadata::Unspecified,
        log_likelihood_normalization: crate::types::LogLikelihoodNormalization::UserProvided,
        log_likelihood,
        deviance,
        reml_score: penalized_objective,
        stable_penalty_term,
        penalized_objective,
        outer_iterations,
        outer_converged,
        outer_gradient_norm,
        standard_deviation: 1.0,
        covariance_conditional,
        covariance_corrected: None,
        inference,
        fitted_link: FittedLinkState::Standard(None),
        geometry,
        block_states,
        // Report the inner status honestly from the threaded `outer_converged`
        // flag rather than hardcoding `Converged`. When the outer optimization
        // did not converge (e.g. it escalated to posterior sampling), surface
        // `StalledAtValidMinimum` — the same non-converged-but-usable bucket the
        // smooth-term path maps to — so downstream consumers
        // (`pirls_status.is_converged()`, `outer_converged` derivation) do not
        // report a non-converged fit as converged.
        pirls_status: if outer_converged {
            crate::pirls::PirlsStatus::Converged
        } else {
            crate::pirls::PirlsStatus::StalledAtValidMinimum
        },
        max_abs_eta: 0.0,
        constraint_kkt: None,
        artifacts: crate::solver::estimate::FitArtifacts {
            pirls: None,
            criterion_certificate,
            ..Default::default()
        },
        inner_cycles,
    })
    .map_err(|e| e.to_string())
}


fn checked_penalizedobjective(
    log_likelihood: f64,
    penalty_value: f64,
    reml_term: f64,
    context: &str,
) -> Result<f64, String> {
    let objective = -log_likelihood + penalty_value + reml_term;
    if objective.is_finite() {
        Ok(objective)
    } else {
        Err(CustomFamilyError::NumericalFailure {
            reason: format!(
                "{context}: non-finite penalized objective \
             (log_likelihood={log_likelihood}, penalty_value={penalty_value}, \
             reml_term={reml_term}, objective={objective})"
            ),
        }
        .into())
    }
}


#[derive(Clone)]
pub struct CustomFamilyBlockPsiDerivative {
    pub penalty_index: Option<usize>,
    pub x_psi: Array2<f64>,
    pub s_psi: Array2<f64>,
    pub s_psi_components: Option<Vec<(usize, Array2<f64>)>>,
    pub s_psi_penalty_components: Option<Vec<(usize, PenaltyMatrix)>>,
    pub x_psi_psi: Option<Vec<Array2<f64>>>,
    pub s_psi_psi: Option<Vec<Array2<f64>>>,
    pub s_psi_psi_components: Option<Vec<Vec<(usize, Array2<f64>)>>>,
    pub s_psi_psi_penalty_components: Option<Vec<Vec<(usize, PenaltyMatrix)>>>,
    pub(crate) implicit_operator: Option<Arc<dyn CustomFamilyPsiDerivativeOperator>>,
    pub implicit_axis: usize,
    pub implicit_group_id: Option<usize>,
}


pub(crate) type SharedDerivativeBlocks = Arc<Vec<Vec<CustomFamilyBlockPsiDerivative>>>;


impl CustomFamilyBlockPsiDerivative {
    /// Public constructor for use in tests and external consumers.
    /// Sets `implicit_operator` to `None`.
    pub fn new(
        penalty_index: Option<usize>,
        x_psi: Array2<f64>,
        s_psi: Array2<f64>,
        s_psi_components: Option<Vec<(usize, Array2<f64>)>>,
        x_psi_psi: Option<Vec<Array2<f64>>>,
        s_psi_psi: Option<Vec<Array2<f64>>>,
        s_psi_psi_components: Option<Vec<Vec<(usize, Array2<f64>)>>>,
    ) -> Self {
        Self {
            penalty_index,
            x_psi,
            s_psi,
            s_psi_components,
            s_psi_penalty_components: None,
            x_psi_psi,
            s_psi_psi,
            s_psi_psi_components,
            s_psi_psi_penalty_components: None,
            implicit_operator: None,
            implicit_axis: 0,
            implicit_group_id: None,
        }
    }
}


pub(crate) trait CustomFamilyPsiDerivativeOperator: Send + Sync + Any {
    fn as_any(&self) -> &dyn Any;
    fn n_data(&self) -> usize;
    fn p_out(&self) -> usize;
    fn transpose_mul(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError>;
    fn forward_mul(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError>;
    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError>;
    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError>;
    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError>;
    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError>;
    fn row_chunk_first(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError>;
    /// Single-row specialization of `row_chunk_first`. Default implementation
    /// delegates to `row_chunk_first(axis, row..row+1)` and copies the
    /// resulting row into the output buffer; implementations that can avoid
    /// the temporary matrix allocation should override this method.
    fn row_vector_first_into(
        &self,
        axis: usize,
        row: usize,
        mut out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), crate::terms::basis::BasisError> {
        let chunk = self.row_chunk_first(axis, row..row + 1)?;
        out.assign(&chunk.row(0));
        Ok(())
    }
    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError>;
    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError>;

    /// Optional upcast to the dense materialization surface. Production exact
    /// paths should prefer the analytic matvec / row-chunk methods above and
    /// avoid forming the full derivative matrix; implementations that *do*
    /// support dense materialization (used by diagnostics, tests, and
    /// small-data fallbacks) should override this to return `Some(self)`.
    fn as_materializable(&self) -> Option<&dyn MaterializablePsiDerivativeOperator> {
        None
    }
}


/// Diagnostic / small-data extension that exposes dense materialization of
/// `\partial X / \partial \psi`. Production exact-Hessian code MUST NOT depend
/// on dense second-derivative materialization; second-order paths use the
/// row-chunk and matvec methods on [`CustomFamilyPsiDerivativeOperator`].
pub(crate) trait MaterializablePsiDerivativeOperator:
    CustomFamilyPsiDerivativeOperator
{
    fn materialize_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError>;
}


impl CustomFamilyPsiDerivativeOperator for crate::terms::basis::ImplicitDesignPsiDerivative {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_data(&self) -> usize {
        crate::terms::basis::ImplicitDesignPsiDerivative::n_data(self)
    }

    fn p_out(&self) -> usize {
        crate::terms::basis::ImplicitDesignPsiDerivative::p_out(self)
    }

    fn transpose_mul(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::transpose_mul(self, axis, v)
    }

    fn forward_mul(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::forward_mul(self, axis, u)
    }

    fn row_chunk_first(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let f: fn(
            &crate::terms::basis::ImplicitDesignPsiDerivative,
            usize,
            Range<usize>,
        ) -> Result<Array2<f64>, crate::terms::basis::BasisError> =
            crate::terms::basis::ImplicitDesignPsiDerivative::row_chunk_first;
        f(self, axis, rows)
    }

    fn row_vector_first_into(
        &self,
        axis: usize,
        row: usize,
        out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::row_vector_first_into(
            self, axis, row, out,
        )
    }

    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let f: fn(
            &crate::terms::basis::ImplicitDesignPsiDerivative,
            usize,
            Range<usize>,
        ) -> Result<Array2<f64>, crate::terms::basis::BasisError> =
            crate::terms::basis::ImplicitDesignPsiDerivative::row_chunk_second_diag;
        f(self, axis, rows)
    }

    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let f: fn(
            &crate::terms::basis::ImplicitDesignPsiDerivative,
            usize,
            usize,
            Range<usize>,
        ) -> Result<Array2<f64>, crate::terms::basis::BasisError> =
            crate::terms::basis::ImplicitDesignPsiDerivative::row_chunk_second_cross;
        f(self, axis_d, axis_e, rows)
    }

    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::transpose_mul_second_diag(self, axis, v)
    }

    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::transpose_mul_second_cross(
            self, axis_d, axis_e, v,
        )
    }

    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::forward_mul_second_diag(self, axis, u)
    }

    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::forward_mul_second_cross(
            self, axis_d, axis_e, u,
        )
    }

    fn as_materializable(&self) -> Option<&dyn MaterializablePsiDerivativeOperator> {
        Some(self)
    }
}


impl MaterializablePsiDerivativeOperator for crate::terms::basis::ImplicitDesignPsiDerivative {
    fn materialize_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        crate::terms::basis::ImplicitDesignPsiDerivative::materialize_first(self, axis)
    }
}


pub(crate) struct EmbeddedImplicitPsiDerivativeOperator {
    base: Arc<crate::terms::basis::ImplicitDesignPsiDerivative>,
    total_p: usize,
    global_range: Range<usize>,
}


impl EmbeddedImplicitPsiDerivativeOperator {
    pub(crate) fn new(
        base: Arc<crate::terms::basis::ImplicitDesignPsiDerivative>,
        global_range: Range<usize>,
        total_p: usize,
    ) -> Result<Self, String> {
        if base.p_out() != global_range.len() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "embedded implicit psi operator width mismatch: got {}, expected {}",
                    base.p_out(),
                    global_range.len()
                ),
            }
            .into());
        }
        if global_range.end > total_p {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "embedded implicit psi operator range {}..{} exceeds total width {total_p}",
                    global_range.start, global_range.end
                ),
            }
            .into());
        }
        Ok(Self {
            base,
            total_p,
            global_range,
        })
    }

    fn embed_vector(&self, local: Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.total_p);
        out.slice_mut(ndarray::s![self.global_range.clone()])
            .assign(&local);
        out
    }

    fn local_coeffs(
        &self,
        u: &ArrayView1<'_, f64>,
        context: &str,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        if u.len() != self.total_p {
            return Err(crate::terms::basis::BasisError::Other(format!(
                "{context} expected coefficient length {}, got {}",
                self.total_p,
                u.len()
            )));
        }
        Ok(u.slice(ndarray::s![self.global_range.clone()]).to_owned())
    }
}


impl CustomFamilyPsiDerivativeOperator for EmbeddedImplicitPsiDerivativeOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_data(&self) -> usize {
        self.base.n_data()
    }

    fn p_out(&self) -> usize {
        self.total_p
    }

    fn transpose_mul(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        Ok(self.embed_vector(self.base.transpose_mul(axis, v)?))
    }

    fn forward_mul(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let local = self.local_coeffs(u, "embedded implicit psi forward_mul")?;
        self.base.forward_mul(axis, &local.view())
    }

    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        Ok(self.embed_vector(self.base.transpose_mul_second_diag(axis, v)?))
    }

    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        Ok(self.embed_vector(self.base.transpose_mul_second_cross(axis_d, axis_e, v)?))
    }

    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let local = self.local_coeffs(u, "embedded implicit psi forward_mul_second_diag")?;
        self.base.forward_mul_second_diag(axis, &local.view())
    }

    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let local = self.local_coeffs(u, "embedded implicit psi forward_mul_second_cross")?;
        self.base
            .forward_mul_second_cross(axis_d, axis_e, &local.view())
    }

    fn row_chunk_first(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let local = self.base.row_chunk_first(axis, rows)?;
        Ok(EmbeddedColumnBlock::new(&local, self.global_range.clone(), self.total_p).materialize())
    }

    fn row_vector_first_into(
        &self,
        axis: usize,
        row: usize,
        mut out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), crate::terms::basis::BasisError> {
        out.fill(0.0);
        let local_slice = out.slice_mut(ndarray::s![self.global_range.clone()]);
        self.base.row_vector_first_into(axis, row, local_slice)
    }

    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let local = self.base.row_chunk_second_diag(axis, rows)?;
        Ok(EmbeddedColumnBlock::new(&local, self.global_range.clone(), self.total_p).materialize())
    }

    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let local = self.base.row_chunk_second_cross(axis_d, axis_e, rows)?;
        Ok(EmbeddedColumnBlock::new(&local, self.global_range.clone(), self.total_p).materialize())
    }

    fn as_materializable(&self) -> Option<&dyn MaterializablePsiDerivativeOperator> {
        Some(self)
    }
}


impl MaterializablePsiDerivativeOperator for EmbeddedImplicitPsiDerivativeOperator {
    fn materialize_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok(EmbeddedColumnBlock::new(
            &self.base.materialize_first(axis)?,
            self.global_range.clone(),
            self.total_p,
        )
        .materialize())
    }
}


/// Non-allocating zero operator for `\partial X / \partial \psi` derivative
/// blocks whose ψ coordinate does not move the design matrix at all (e.g.
/// the spatial-adaptive overlay's mass / tension / stiffness / ε
/// hyperparameters, which act through the penalty stack alone).
///
/// All matvec/transpose_mul methods return zero vectors of the correct
/// length, all row-chunk methods return chunk-sized zero matrices. The
/// operator never allocates an `(n, p)` dense buffer, which saves ~1.45 GiB
/// at the large-scale spatial-adaptive overlay (n ≈ 320 000, p ≈ 101,
/// six hyperparameters).
pub(crate) struct ZeroPsiDerivativeOperator {
    n: usize,
    p: usize,
}


impl ZeroPsiDerivativeOperator {
    pub(crate) fn new(n: usize, p: usize) -> Self {
        Self { n, p }
    }
}


impl CustomFamilyPsiDerivativeOperator for ZeroPsiDerivativeOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_data(&self) -> usize {
        self.n
    }

    fn p_out(&self) -> usize {
        self.p
    }

    fn transpose_mul(
        &self,
        idx: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        assert!(idx < usize::MAX);
        assert_eq!(v.len(), self.n, "zero psi transpose_mul length mismatch");
        Ok(Array1::<f64>::zeros(self.p))
    }

    fn forward_mul(
        &self,
        idx: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        assert!(idx < usize::MAX);
        assert_eq!(u.len(), self.p, "zero psi forward_mul length mismatch");
        Ok(Array1::<f64>::zeros(self.n))
    }

    fn transpose_mul_second_diag(
        &self,
        idx: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        assert!(idx < usize::MAX);
        assert_eq!(
            v.len(),
            self.n,
            "zero psi transpose_mul_second_diag length mismatch"
        );
        Ok(Array1::<f64>::zeros(self.p))
    }

    fn transpose_mul_second_cross(
        &self,
        idx: usize,
        idx2: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        assert!(idx < usize::MAX);
        assert!(idx2 < usize::MAX);
        assert_eq!(
            v.len(),
            self.n,
            "zero psi transpose_mul_second_cross length mismatch"
        );
        Ok(Array1::<f64>::zeros(self.p))
    }

    fn forward_mul_second_diag(
        &self,
        idx: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        assert!(idx < usize::MAX);
        assert_eq!(
            u.len(),
            self.p,
            "zero psi forward_mul_second_diag length mismatch"
        );
        Ok(Array1::<f64>::zeros(self.n))
    }

    fn forward_mul_second_cross(
        &self,
        idx: usize,
        idx2: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        assert!(idx < usize::MAX);
        assert!(idx2 < usize::MAX);
        assert_eq!(
            u.len(),
            self.p,
            "zero psi forward_mul_second_cross length mismatch"
        );
        Ok(Array1::<f64>::zeros(self.n))
    }

    fn row_chunk_first(
        &self,
        idx: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        assert!(idx < usize::MAX);
        assert!(
            rows.start <= rows.end && rows.end <= self.n,
            "zero psi row_chunk_first row range out of bounds"
        );
        Ok(Array2::<f64>::zeros((rows.end - rows.start, self.p)))
    }

    fn row_vector_first_into(
        &self,
        idx: usize,
        row: usize,
        mut out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), crate::terms::basis::BasisError> {
        assert!(idx < usize::MAX);
        assert!(
            row < self.n,
            "zero psi row_vector_first_into row out of bounds"
        );
        assert_eq!(
            out.len(),
            self.p,
            "zero psi row_vector_first_into output length mismatch"
        );
        out.fill(0.0);
        Ok(())
    }

    fn row_chunk_second_diag(
        &self,
        idx: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        assert!(idx < usize::MAX);
        assert!(
            rows.start <= rows.end && rows.end <= self.n,
            "zero psi row_chunk_second_diag row range out of bounds"
        );
        Ok(Array2::<f64>::zeros((rows.end - rows.start, self.p)))
    }

    fn row_chunk_second_cross(
        &self,
        idx: usize,
        idx2: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        assert!(idx < usize::MAX);
        assert!(idx2 < usize::MAX);
        assert!(
            rows.start <= rows.end && rows.end <= self.n,
            "zero psi row_chunk_second_cross row range out of bounds"
        );
        Ok(Array2::<f64>::zeros((rows.end - rows.start, self.p)))
    }
}


fn stack_dense_row_blocks(blocks: &[Array2<f64>]) -> Array2<f64> {
    let total_rows = blocks.iter().map(Array2::nrows).sum();
    let p = blocks.first().map(Array2::ncols).unwrap_or(0);
    let mut stacked = Array2::<f64>::zeros((total_rows, p));
    let mut row_start = 0usize;
    for block in blocks {
        assert_eq!(block.ncols(), p);
        let row_end = row_start + block.nrows();
        stacked
            .slice_mut(ndarray::s![row_start..row_end, ..])
            .assign(block);
        row_start = row_end;
    }
    stacked
}


struct EmbeddedDensePsiDerivativeOperator {
    axis: usize,
    total_p: usize,
    global_range: Range<usize>,
    first_local: Array2<f64>,
    second_diag_local: Array2<f64>,
    second_cross_local: HashMap<usize, Array2<f64>>,
}


impl EmbeddedDensePsiDerivativeOperator {
    fn new(
        axis: usize,
        total_p: usize,
        global_range: Range<usize>,
        first_local: Array2<f64>,
        second_diag_local: Array2<f64>,
        second_cross_local: HashMap<usize, Array2<f64>>,
    ) -> Result<Self, String> {
        let local_p = global_range.len();
        if first_local.ncols() != local_p {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "embedded dense psi operator first-derivative width mismatch: got {}, expected {local_p}",
                first_local.ncols()
            ) }.into());
        }
        if second_diag_local.ncols() != local_p {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "embedded dense psi operator second-diag width mismatch: got {}, expected {local_p}",
                second_diag_local.ncols()
            ) }.into());
        }
        for (cross_axis, local) in &second_cross_local {
            if local.ncols() != local_p {
                return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                    "embedded dense psi operator cross axis {cross_axis} width mismatch: got {}, expected {local_p}",
                    local.ncols()
                ) }.into());
            }
        }
        Ok(Self {
            axis,
            total_p,
            global_range,
            first_local,
            second_diag_local,
            second_cross_local,
        })
    }

    fn validate_axis(
        &self,
        axis: usize,
        context: &str,
    ) -> Result<(), crate::terms::basis::BasisError> {
        if axis == self.axis {
            Ok(())
        } else {
            Err(crate::terms::basis::BasisError::Other(format!(
                "{context} expected axis {}, got {axis}",
                self.axis
            )))
        }
    }

    fn embed_vector(&self, local: Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.total_p);
        out.slice_mut(ndarray::s![self.global_range.clone()])
            .assign(&local);
        out
    }

    fn local_coeffs(
        &self,
        u: &ArrayView1<'_, f64>,
        context: &str,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        if u.len() != self.total_p {
            return Err(crate::terms::basis::BasisError::Other(format!(
                "{context} expected coefficient length {}, got {}",
                self.total_p,
                u.len()
            )));
        }
        Ok(u.slice(ndarray::s![self.global_range.clone()]).to_owned())
    }

    fn cross_local(
        &self,
        axis_e: usize,
        context: &str,
    ) -> Result<&Array2<f64>, crate::terms::basis::BasisError> {
        self.second_cross_local.get(&axis_e).ok_or_else(|| {
            crate::terms::basis::BasisError::Other(format!(
                "{context} is missing cross-derivative data for axis {}",
                axis_e
            ))
        })
    }
}


impl CustomFamilyPsiDerivativeOperator for EmbeddedDensePsiDerivativeOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_data(&self) -> usize {
        self.first_local.nrows()
    }

    fn p_out(&self) -> usize {
        self.total_p
    }

    fn transpose_mul(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi transpose_mul")?;
        if v.len() != self.n_data() {
            return Err(crate::terms::basis::BasisError::Other(format!(
                "embedded dense psi transpose_mul expected {} rows, got {}",
                self.n_data(),
                v.len()
            )));
        }
        Ok(self.embed_vector(crate::faer_ndarray::fast_atv(&self.first_local, v)))
    }

    fn forward_mul(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi forward_mul")?;
        Ok(self
            .first_local
            .dot(&self.local_coeffs(u, "embedded dense psi forward_mul")?))
    }

    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi transpose_mul_second_diag")?;
        if v.len() != self.second_diag_local.nrows() {
            return Err(crate::terms::basis::BasisError::Other(format!(
                "embedded dense psi transpose_mul_second_diag expected {} rows, got {}",
                self.second_diag_local.nrows(),
                v.len()
            )));
        }
        Ok(self.embed_vector(crate::faer_ndarray::fast_atv(&self.second_diag_local, v)))
    }

    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis_d, "embedded dense psi transpose_mul_second_cross")?;
        let local = self.cross_local(axis_e, "embedded dense psi transpose_mul_second_cross")?;
        if v.len() != local.nrows() {
            return Err(crate::terms::basis::BasisError::Other(format!(
                "embedded dense psi transpose_mul_second_cross expected {} rows, got {}",
                local.nrows(),
                v.len()
            )));
        }
        Ok(self.embed_vector(crate::faer_ndarray::fast_atv(local, v)))
    }

    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi forward_mul_second_diag")?;
        Ok(self
            .second_diag_local
            .dot(&self.local_coeffs(u, "embedded dense psi forward_mul_second_diag")?))
    }

    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis_d, "embedded dense psi forward_mul_second_cross")?;
        Ok(self
            .cross_local(axis_e, "embedded dense psi forward_mul_second_cross")?
            .dot(&self.local_coeffs(u, "embedded dense psi forward_mul_second_cross")?))
    }

    fn row_chunk_first(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi row_chunk_first")?;
        let local = self.first_local.slice(ndarray::s![rows, ..]).to_owned();
        Ok(EmbeddedColumnBlock::new(&local, self.global_range.clone(), self.total_p).materialize())
    }

    fn row_vector_first_into(
        &self,
        axis: usize,
        row: usize,
        mut out: ArrayViewMut1<'_, f64>,
    ) -> Result<(), crate::terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi row_vector_first_into")?;
        if row >= self.first_local.nrows() {
            return Err(crate::terms::basis::BasisError::Other(format!(
                "embedded dense psi row_vector_first_into row {row} out of bounds for {}",
                self.first_local.nrows()
            )));
        }
        if out.len() != self.total_p {
            return Err(crate::terms::basis::BasisError::Other(format!(
                "embedded dense psi row_vector_first_into expected length {}, got {}",
                self.total_p,
                out.len()
            )));
        }
        out.fill(0.0);
        out.slice_mut(ndarray::s![self.global_range.clone()])
            .assign(&self.first_local.row(row));
        Ok(())
    }

    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi row_chunk_second_diag")?;
        let local = self
            .second_diag_local
            .slice(ndarray::s![rows, ..])
            .to_owned();
        Ok(EmbeddedColumnBlock::new(&local, self.global_range.clone(), self.total_p).materialize())
    }

    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis_d, "embedded dense psi row_chunk_second_cross")?;
        let local = self
            .cross_local(axis_e, "embedded dense psi row_chunk_second_cross")?
            .slice(ndarray::s![rows, ..])
            .to_owned();
        Ok(EmbeddedColumnBlock::new(&local, self.global_range.clone(), self.total_p).materialize())
    }

    fn as_materializable(&self) -> Option<&dyn MaterializablePsiDerivativeOperator> {
        Some(self)
    }
}


impl MaterializablePsiDerivativeOperator for EmbeddedDensePsiDerivativeOperator {
    fn materialize_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        self.validate_axis(axis, "embedded dense psi materialize_first")?;
        Ok(
            EmbeddedColumnBlock::new(&self.first_local, self.global_range.clone(), self.total_p)
                .materialize(),
        )
    }
}


pub(crate) fn build_embedded_dense_psi_operator(
    first_local: &Array2<f64>,
    second_diag_local: &Array2<f64>,
    second_cross_local: Option<&Vec<(usize, Array2<f64>)>>,
    global_range: Range<usize>,
    total_p: usize,
    axis: usize,
) -> Result<Arc<dyn CustomFamilyPsiDerivativeOperator>, String> {
    let second_cross_local = second_cross_local
        .map(|rows| {
            rows.iter()
                .map(|(axis, local)| (*axis, local.clone()))
                .collect()
        })
        .unwrap_or_default();
    Ok(Arc::new(EmbeddedDensePsiDerivativeOperator::new(
        axis,
        total_p,
        global_range,
        first_local.clone(),
        second_diag_local.clone(),
        second_cross_local,
    )?))
}


struct RowwiseKroneckerPsiDerivativeOperator {
    base: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    time_bases: Vec<Arc<Array2<f64>>>,
    n_per_block: usize,
    p_time: usize,
    p_out: usize,
}


impl RowwiseKroneckerPsiDerivativeOperator {
    fn new(
        base: Arc<dyn CustomFamilyPsiDerivativeOperator>,
        time_bases: Vec<Arc<Array2<f64>>>,
    ) -> Result<Self, String> {
        let first = time_bases.first().ok_or_else(|| {
            "rowwise kronecker psi operator needs at least one time basis".to_string()
        })?;
        let n_per_block = first.nrows();
        let p_time = first.ncols();
        for (idx, basis) in time_bases.iter().enumerate() {
            if basis.nrows() != n_per_block || basis.ncols() != p_time {
                return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                    "rowwise kronecker psi operator time basis {idx} shape mismatch: got {}x{}, expected {}x{}",
                    basis.nrows(),
                    basis.ncols(),
                    n_per_block,
                    p_time
                ) }.into());
            }
        }
        if base.n_data() != n_per_block {
            return Err(CustomFamilyError::DimensionMismatch { reason: format!(
                "rowwise kronecker psi operator base row mismatch: got {}, expected {n_per_block}",
                base.n_data()
            ) }.into());
        }
        Ok(Self {
            p_out: base.p_out() * p_time,
            base,
            time_bases,
            n_per_block,
            p_time,
        })
    }

    fn split_time_columns(&self, u: &ArrayView1<'_, f64>) -> Vec<Array1<f64>> {
        let p_base = self.base.p_out();
        assert_eq!(u.len(), self.p_out);
        let mut cols = vec![Array1::<f64>::zeros(p_base); self.p_time];
        for j in 0..p_base {
            for t in 0..self.p_time {
                cols[t][j] = u[j * self.p_time + t];
            }
        }
        cols
    }

    fn lifted_row_chunk_with_base<F>(
        &self,
        rows: Range<usize>,
        mut base_chunk: F,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError>
    where
        F: FnMut(Range<usize>) -> Result<Array2<f64>, crate::terms::basis::BasisError>,
    {
        if rows.start > rows.end || rows.end > self.n_data() {
            return Err(crate::terms::basis::BasisError::Other(format!(
                "rowwise kronecker psi row chunk {}..{} out of bounds for {} rows",
                rows.start,
                rows.end,
                self.n_data()
            )));
        }
        if rows.is_empty() {
            return Ok(Array2::<f64>::zeros((0, self.p_out)));
        }

        let first_block = rows.start / self.n_per_block;
        let last_block = (rows.end - 1) / self.n_per_block;
        let mut blocks = Vec::with_capacity(last_block + 1 - first_block);
        for block_idx in first_block..=last_block {
            let block_global_start = block_idx * self.n_per_block;
            let local_start = rows.start.saturating_sub(block_global_start);
            let local_end = (rows.end - block_global_start).min(self.n_per_block);
            let local_rows = local_start..local_end;
            let base = base_chunk(local_rows.clone())?;
            let time = self.time_bases[block_idx]
                .slice(ndarray::s![local_rows, ..])
                .to_owned();
            blocks.push(dense_rowwise_kronecker(base.view(), time.view()));
        }
        Ok(stack_dense_row_blocks(&blocks))
    }

    /// Canonical transpose-direction lifted matvec: for each time column `t`,
    /// weight `v` by the time basis column, delegate to the base operator via
    /// `base_op`, and scatter the per-base accumulator into the lifted layout.
    fn lifted_transpose_mul_with_base<F>(
        &self,
        v: &ArrayView1<'_, f64>,
        mut base_op: F,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError>
    where
        F: FnMut(&ArrayView1<'_, f64>) -> Result<Array1<f64>, crate::terms::basis::BasisError>,
    {
        assert_eq!(v.len(), self.n_data());
        let p_base = self.base.p_out();
        let mut out = Array1::<f64>::zeros(self.p_out);
        for t in 0..self.p_time {
            let mut accum = Array1::<f64>::zeros(p_base);
            for (block_idx, time_basis) in self.time_bases.iter().enumerate() {
                let row_start = block_idx * self.n_per_block;
                let row_end = row_start + self.n_per_block;
                let weighted = &v.slice(ndarray::s![row_start..row_end]).to_owned()
                    * &time_basis.column(t).to_owned();
                accum += &base_op(&weighted.view())?;
            }
            for j in 0..p_base {
                out[j * self.p_time + t] = accum[j];
            }
        }
        Ok(out)
    }

    /// Canonical forward-direction lifted matvec: split `u` into per-time-column
    /// coefficient vectors, delegate each to the base operator via `base_op`, and
    /// accumulate the time-basis-weighted contributions into the block rows.
    fn lifted_forward_mul_with_base<F>(
        &self,
        u: &ArrayView1<'_, f64>,
        mut base_op: F,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError>
    where
        F: FnMut(&ArrayView1<'_, f64>) -> Result<Array1<f64>, crate::terms::basis::BasisError>,
    {
        let time_cols = self.split_time_columns(u);
        let mut out = Array1::<f64>::zeros(self.n_data());
        for (t, coeffs) in time_cols.iter().enumerate() {
            let base_eval = base_op(&coeffs.view())?;
            for (block_idx, time_basis) in self.time_bases.iter().enumerate() {
                let row_start = block_idx * self.n_per_block;
                let row_end = row_start + self.n_per_block;
                let contrib = &base_eval * &time_basis.column(t).to_owned();
                let mut out_block = out.slice_mut(ndarray::s![row_start..row_end]);
                out_block += &contrib;
            }
        }
        Ok(out)
    }
}


impl CustomFamilyPsiDerivativeOperator for RowwiseKroneckerPsiDerivativeOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_data(&self) -> usize {
        self.n_per_block * self.time_bases.len()
    }

    fn p_out(&self) -> usize {
        self.p_out
    }

    fn transpose_mul(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.lifted_transpose_mul_with_base(v, |weighted| self.base.transpose_mul(axis, weighted))
    }

    fn forward_mul(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.lifted_forward_mul_with_base(u, |coeffs| self.base.forward_mul(axis, coeffs))
    }

    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.lifted_transpose_mul_with_base(v, |weighted| {
            self.base.transpose_mul_second_diag(axis, weighted)
        })
    }

    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.lifted_transpose_mul_with_base(v, |weighted| {
            self.base
                .transpose_mul_second_cross(axis_d, axis_e, weighted)
        })
    }

    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.lifted_forward_mul_with_base(u, |coeffs| {
            self.base.forward_mul_second_diag(axis, coeffs)
        })
    }

    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        self.lifted_forward_mul_with_base(u, |coeffs| {
            self.base.forward_mul_second_cross(axis_d, axis_e, coeffs)
        })
    }

    fn row_chunk_first(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        self.lifted_row_chunk_with_base(rows, |local_rows| {
            self.base.row_chunk_first(axis, local_rows)
        })
    }

    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        self.lifted_row_chunk_with_base(rows, |local_rows| {
            self.base.row_chunk_second_diag(axis, local_rows)
        })
    }

    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        self.lifted_row_chunk_with_base(rows, |local_rows| {
            self.base.row_chunk_second_cross(axis_d, axis_e, local_rows)
        })
    }

    fn as_materializable(&self) -> Option<&dyn MaterializablePsiDerivativeOperator> {
        Some(self)
    }
}


impl MaterializablePsiDerivativeOperator for RowwiseKroneckerPsiDerivativeOperator {
    fn materialize_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let base_mat = self.base.as_materializable().ok_or_else(|| {
            crate::terms::basis::BasisError::Other(
                "rowwise kronecker psi operator: base operator does not support materialization"
                    .to_string(),
            )
        })?;
        let base = base_mat.materialize_first(axis)?;
        let blocks: Vec<Array2<f64>> = self
            .time_bases
            .iter()
            .map(|basis| dense_rowwise_kronecker(base.view(), basis.view()))
            .collect();
        Ok(stack_dense_row_blocks(&blocks))
    }
}


pub(crate) fn build_rowwise_kronecker_psi_operator(
    base: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    time_bases: Vec<Arc<Array2<f64>>>,
) -> Result<Arc<dyn CustomFamilyPsiDerivativeOperator>, String> {
    Ok(Arc::new(RowwiseKroneckerPsiDerivativeOperator::new(
        base, time_bases,
    )?))
}


#[derive(Clone)]
pub(crate) struct CustomFamilyPsiDesignAction {
    operator: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    axis: usize,
    row_range: Range<usize>,
    p: usize,
}


impl CustomFamilyPsiDesignAction {
    pub(crate) fn from_first_derivative(
        deriv: &CustomFamilyBlockPsiDerivative,
        total_rows: usize,
        p: usize,
        row_range: Range<usize>,
        label: &str,
    ) -> Result<Self, String> {
        if row_range.end > total_rows {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "{label} row range {}..{} exceeds total rows {total_rows}",
                    row_range.start, row_range.end
                ),
            }
            .into());
        }
        if let Some(op) = deriv.implicit_operator.as_ref()
            && op.n_data() == total_rows
            && op.p_out() == p
        {
            return Ok(Self {
                operator: Arc::clone(op),
                axis: deriv.implicit_axis,
                row_range,
                p,
            });
        }
        Err(CustomFamilyError::UnsupportedConfiguration { reason: format!(
            "{label} is missing an implicit x_psi operator with shape {}x{}; got dense payload {}x{} instead",
            total_rows,
            p,
            deriv.x_psi.nrows(),
            deriv.x_psi.ncols(),
        ) }.into())
    }

    pub(crate) fn is_implicit(&self) -> bool {
        true
    }

    pub(crate) fn nrows(&self) -> usize {
        self.row_range.end - self.row_range.start
    }

    pub(crate) fn slice_rows(&self, row_range: Range<usize>) -> Result<Self, String> {
        if row_range.end > self.nrows() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "psi design row range {}..{} exceeds available rows {}",
                    row_range.start,
                    row_range.end,
                    self.nrows()
                ),
            }
            .into());
        }
        Ok(Self {
            operator: Arc::clone(&self.operator),
            axis: self.axis,
            row_range: (self.row_range.start + row_range.start)
                ..(self.row_range.start + row_range.end),
            p: self.p,
        })
    }

    pub(crate) fn forward_mul(&self, u: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(u.len(), self.p);
        self.operator
            .forward_mul(self.axis, &u)
            .expect("radial scalar evaluation failed during implicit psi forward_mul")
            .slice(ndarray::s![self.row_range.clone()])
            .to_owned()
    }

    pub(crate) fn transpose_mul(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.row_range.end - self.row_range.start);
        if self.row_range.start == 0 && self.row_range.end == self.operator.n_data() {
            self.operator
                .transpose_mul(self.axis, &v)
                .expect("radial scalar evaluation failed during implicit psi transpose_mul")
        } else {
            let mut expanded = Array1::<f64>::zeros(self.operator.n_data());
            expanded
                .slice_mut(ndarray::s![self.row_range.clone()])
                .assign(&v);
            self.operator
                .transpose_mul(self.axis, &expanded.view())
                .expect("radial scalar evaluation failed during implicit psi transpose_mul")
        }
    }

    fn absolute_rows(&self, rows: Range<usize>) -> Range<usize> {
        (self.row_range.start + rows.start)..(self.row_range.start + rows.end)
    }

    pub(crate) fn row_chunk(&self, rows: Range<usize>) -> Result<Array2<f64>, String> {
        if rows.end > self.nrows() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "psi design row range {}..{} exceeds available rows {}",
                    rows.start,
                    rows.end,
                    self.nrows()
                ),
            }
            .into());
        }
        self.operator
            .row_chunk_first(self.axis, self.absolute_rows(rows))
            .map_err(|e| e.to_string())
    }

    pub(crate) fn row_vector(&self, row: usize) -> Result<Array1<f64>, String> {
        if row >= self.nrows() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "psi design row {row} exceeds available rows {}",
                    self.nrows()
                ),
            }
            .into());
        }
        let absolute_row = self.row_range.start + row;
        let mut out = Array1::<f64>::zeros(self.p);
        self.operator
            .row_vector_first_into(self.axis, absolute_row, out.view_mut())
            .map_err(|e| e.to_string())?;
        Ok(out)
    }
}


#[derive(Clone, Copy)]
enum CustomFamilyPsiSecondDesignLevel {
    Diag(usize),
    Cross(usize, usize),
}


#[derive(Clone)]
pub(crate) struct CustomFamilyPsiSecondDesignAction {
    operator: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    level: CustomFamilyPsiSecondDesignLevel,
    row_range: Range<usize>,
    p: usize,
}


impl CustomFamilyPsiSecondDesignAction {
    pub(crate) fn from_second_derivative(
        deriv_i: &CustomFamilyBlockPsiDerivative,
        deriv_j: &CustomFamilyBlockPsiDerivative,
        total_rows: usize,
        p: usize,
        row_range: Range<usize>,
        label: &str,
    ) -> Result<Option<Self>, String> {
        if row_range.end > total_rows {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "{label} row range {}..{} exceeds total rows {total_rows}",
                    row_range.start, row_range.end
                ),
            }
            .into());
        }
        let Some(op) = deriv_i.implicit_operator.as_ref() else {
            return Ok(None);
        };
        if op.n_data() != total_rows || op.p_out() != p {
            return Err(CustomFamilyError::UnsupportedConfiguration {
                reason: format!(
                    "{label} is missing an implicit x_psi_psi operator with shape {}x{}",
                    total_rows, p
                ),
            }
            .into());
        }
        let same_group = deriv_i.implicit_group_id.is_some()
            && deriv_i.implicit_group_id == deriv_j.implicit_group_id;
        if !same_group {
            return Ok(None);
        }
        let level = if deriv_i.implicit_axis == deriv_j.implicit_axis {
            CustomFamilyPsiSecondDesignLevel::Diag(deriv_i.implicit_axis)
        } else {
            CustomFamilyPsiSecondDesignLevel::Cross(deriv_i.implicit_axis, deriv_j.implicit_axis)
        };
        Ok(Some(Self {
            operator: Arc::clone(op),
            level,
            row_range,
            p,
        }))
    }

    pub(crate) fn nrows(&self) -> usize {
        self.row_range.end - self.row_range.start
    }

    pub(crate) fn forward_mul(&self, u: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(u.len(), self.p);
        let out = match self.level {
            CustomFamilyPsiSecondDesignLevel::Diag(axis) => self
                .operator
                .forward_mul_second_diag(axis, &u)
                .expect("radial scalar evaluation failed during implicit psi second forward_mul"),
            CustomFamilyPsiSecondDesignLevel::Cross(axis_d, axis_e) => self
                .operator
                .forward_mul_second_cross(axis_d, axis_e, &u)
                .expect("radial scalar evaluation failed during implicit psi second forward_mul"),
        };
        out.slice(ndarray::s![self.row_range.clone()]).to_owned()
    }

    pub(crate) fn transpose_mul(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.nrows());
        let expanded = if self.row_range.start == 0 && self.row_range.end == self.operator.n_data()
        {
            None
        } else {
            let mut expanded = Array1::<f64>::zeros(self.operator.n_data());
            expanded
                .slice_mut(ndarray::s![self.row_range.clone()])
                .assign(&v);
            Some(expanded)
        };
        let full_v = expanded.as_ref().map_or(v, |arr| arr.view());
        match self.level {
            CustomFamilyPsiSecondDesignLevel::Diag(axis) => self
                .operator
                .transpose_mul_second_diag(axis, &full_v)
                .expect("radial scalar evaluation failed during implicit psi second transpose_mul"),
            CustomFamilyPsiSecondDesignLevel::Cross(axis_d, axis_e) => self
                .operator
                .transpose_mul_second_cross(axis_d, axis_e, &full_v)
                .expect("radial scalar evaluation failed during implicit psi second transpose_mul"),
        }
    }

    fn absolute_rows(&self, rows: Range<usize>) -> Range<usize> {
        (self.row_range.start + rows.start)..(self.row_range.start + rows.end)
    }

    pub(crate) fn row_chunk(&self, rows: Range<usize>) -> Result<Array2<f64>, String> {
        if rows.end > self.nrows() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "psi second-design row range {}..{} exceeds available rows {}",
                    rows.start,
                    rows.end,
                    self.nrows()
                ),
            }
            .into());
        }
        match self.level {
            CustomFamilyPsiSecondDesignLevel::Diag(axis) => self
                .operator
                .row_chunk_second_diag(axis, self.absolute_rows(rows))
                .map_err(|e| e.to_string()),
            CustomFamilyPsiSecondDesignLevel::Cross(axis_d, axis_e) => self
                .operator
                .row_chunk_second_cross(axis_d, axis_e, self.absolute_rows(rows))
                .map_err(|e| e.to_string()),
        }
    }

    pub(crate) fn row_vector(&self, row: usize) -> Result<Array1<f64>, String> {
        self.row_chunk(row..row + 1).map(|m| m.row(0).to_owned())
    }
}


#[derive(Clone, Copy)]
pub(crate) enum CustomFamilyPsiLinearMapRef<'a> {
    Dense(&'a Array2<f64>),
    First(&'a CustomFamilyPsiDesignAction),
    Second(&'a CustomFamilyPsiSecondDesignAction),
    Zero { nrows: usize, ncols: usize },
}


impl CustomFamilyPsiLinearMapRef<'_> {
    pub(crate) fn nrows(&self) -> usize {
        match self {
            Self::Dense(mat) => mat.nrows(),
            Self::First(action) => action.nrows(),
            Self::Second(action) => action.nrows(),
            Self::Zero { nrows, .. } => *nrows,
        }
    }

    pub(crate) fn ncols(&self) -> usize {
        match self {
            Self::Dense(mat) => mat.ncols(),
            Self::First(action) => action.p,
            Self::Second(action) => action.p,
            Self::Zero { ncols, .. } => *ncols,
        }
    }

    pub(crate) fn forward_mul(&self, u: ArrayView1<'_, f64>) -> Array1<f64> {
        match self {
            Self::Dense(mat) => mat.dot(&u),
            Self::First(action) => action.forward_mul(u),
            Self::Second(action) => action.forward_mul(u),
            Self::Zero { nrows, .. } => Array1::<f64>::zeros(*nrows),
        }
    }

    pub(crate) fn transpose_mul(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        match self {
            Self::Dense(mat) => crate::faer_ndarray::fast_atv(mat, &v),
            Self::First(action) => action.transpose_mul(v),
            Self::Second(action) => action.transpose_mul(v),
            Self::Zero { ncols, .. } => Array1::<f64>::zeros(*ncols),
        }
    }

    pub(crate) fn row_vector(&self, row: usize) -> Result<Array1<f64>, String> {
        if row >= self.nrows() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "psi linear-map row {row} out of bounds for {} rows",
                    self.nrows()
                ),
            }
            .into());
        }
        Ok(match self {
            Self::Dense(mat) => mat.row(row).to_owned(),
            Self::First(action) => action.row_vector(row)?,
            Self::Second(action) => action.row_vector(row)?,
            Self::Zero { ncols, .. } => Array1::<f64>::zeros(*ncols),
        })
    }

    pub(crate) fn row_chunk(&self, rows: Range<usize>) -> Result<Array2<f64>, String> {
        if rows.end > self.nrows() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "psi linear-map row range {}..{} out of bounds for {} rows",
                    rows.start,
                    rows.end,
                    self.nrows()
                ),
            }
            .into());
        }
        Ok(match self {
            Self::Dense(mat) => mat.slice(ndarray::s![rows, ..]).to_owned(),
            Self::First(action) => action.row_chunk(rows)?,
            Self::Second(action) => action.row_chunk(rows)?,
            Self::Zero { ncols, .. } => Array2::<f64>::zeros((rows.end - rows.start, *ncols)),
        })
    }
}


#[derive(Clone)]
pub(crate) enum PsiDesignMap {
    Zero {
        nrows: usize,
        ncols: usize,
    },
    Dense {
        matrix: Arc<Array2<f64>>,
    },
    First {
        action: CustomFamilyPsiDesignAction,
    },
    Second {
        action: CustomFamilyPsiSecondDesignAction,
    },
}


impl PsiDesignMap {
    pub(crate) fn ncols(&self) -> usize {
        match self {
            Self::Zero { ncols, .. } => *ncols,
            Self::Dense { matrix } => matrix.ncols(),
            Self::First { action } => action.p,
            Self::Second { action } => action.p,
        }
    }

    pub(crate) fn forward_mul(&self, u: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        match self {
            Self::Zero { nrows, .. } => Ok(Array1::<f64>::zeros(*nrows)),
            Self::Dense { matrix } => Ok(matrix.dot(&u)),
            Self::First { action } => Ok(action.forward_mul(u)),
            Self::Second { action } => Ok(action.forward_mul(u)),
        }
    }

    pub(crate) fn row_chunk(&self, rows: Range<usize>) -> Result<Array2<f64>, String> {
        let ncols = self.ncols();
        match self {
            Self::Zero { .. } => Ok(Array2::<f64>::zeros((rows.end - rows.start, ncols))),
            Self::Dense { matrix } => Ok(matrix.slice(ndarray::s![rows, ..]).to_owned()),
            Self::First { action } => action.row_chunk(rows),
            Self::Second { action } => action.row_chunk(rows),
        }
    }

    pub(crate) fn row_vector(&self, row: usize) -> Result<Array1<f64>, String> {
        match self {
            Self::Zero { ncols, .. } => Ok(Array1::<f64>::zeros(*ncols)),
            Self::Dense { matrix } => Ok(matrix.row(row).to_owned()),
            Self::First { action } => action.row_vector(row),
            Self::Second { action } => action.row_vector(row),
        }
    }

    /// Borrow this map as a `CustomFamilyPsiLinearMapRef`, handling every
    /// variant. This is the zero-allocation replacement for the pattern
    /// `first_psi_linear_map(action.as_ref(), dense.as_ref(), n, p)`.
    pub(crate) fn as_linear_map_ref(&self) -> CustomFamilyPsiLinearMapRef<'_> {
        match self {
            Self::Zero { nrows, ncols } => CustomFamilyPsiLinearMapRef::Zero {
                nrows: *nrows,
                ncols: *ncols,
            },
            Self::Dense { matrix } => CustomFamilyPsiLinearMapRef::Dense(matrix.as_ref()),
            Self::First { action } => CustomFamilyPsiLinearMapRef::First(action),
            Self::Second { action } => CustomFamilyPsiLinearMapRef::Second(action),
        }
    }

    /// Return a reference to the first-derivative operator action if this map
    /// holds one. Useful for callers that need to pass ownership of the action
    /// into downstream operator builders.
    pub(crate) fn as_first_action(&self) -> Option<&CustomFamilyPsiDesignAction> {
        match self {
            Self::First { action } => Some(action),
            _ => None,
        }
    }

    /// Clone the first-derivative operator action if this map holds one.
    pub(crate) fn cloned_first_action(&self) -> Option<CustomFamilyPsiDesignAction> {
        self.as_first_action().cloned()
    }
}


fn is_zero_array(a: &Array2<f64>) -> bool {
    a.iter().all(|x| *x == 0.0)
}


pub(crate) fn weighted_crossprod_psi_maps(
    left: CustomFamilyPsiLinearMapRef<'_>,
    weights: ArrayView1<'_, f64>,
    right: CustomFamilyPsiLinearMapRef<'_>,
) -> Result<Array2<f64>, String> {
    if left.nrows() != weights.len() || right.nrows() != weights.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "psi weighted crossprod row mismatch: left={}, weights={}, right={}",
                left.nrows(),
                weights.len(),
                right.nrows()
            ),
        }
        .into());
    }
    let p_left = left.ncols();
    let p_right = right.ncols();
    if p_left == 0 || p_right == 0 {
        return Ok(Array2::<f64>::zeros((p_left, p_right)));
    }
    // Zero fast path: either operand being the Zero variant makes the full product zero.
    if matches!(left, CustomFamilyPsiLinearMapRef::Zero { .. })
        || matches!(right, CustomFamilyPsiLinearMapRef::Zero { .. })
    {
        return Ok(Array2::<f64>::zeros((p_left, p_right)));
    }

    let mut out = Array2::<f64>::zeros((p_left, p_right));
    // Stream row chunks of both operands so the weighted intermediate is never
    // materialized at full n x p_right size. Chunk size is governed by the
    // resource policy's row_chunk_target_bytes.
    let policy = ResourcePolicy::default_library();
    let rows_per_chunk = crate::resource::rows_for_target_bytes(
        policy.row_chunk_target_bytes,
        p_left.saturating_add(p_right).max(1),
    );

    let n = weights.len();
    for start in (0..n).step_by(rows_per_chunk) {
        let end = (start + rows_per_chunk).min(n);
        let rows = start..end;
        let xl = left.row_chunk(rows.clone())?;
        let mut xr = right.row_chunk(rows.clone())?;
        for local in 0..xr.nrows() {
            let w = weights[start + local];
            if w != 1.0 {
                for j in 0..p_right {
                    xr[[local, j]] *= w;
                }
            }
        }
        out += &fast_atb(&xl, &xr);
    }
    Ok(out)
}


pub(crate) fn first_psi_linear_map<'a>(
    action: Option<&'a CustomFamilyPsiDesignAction>,
    dense: Option<&'a Array2<f64>>,
    nrows: usize,
    ncols: usize,
) -> CustomFamilyPsiLinearMapRef<'a> {
    if let Some(action) = action {
        CustomFamilyPsiLinearMapRef::First(action)
    } else if let Some(dense) = dense
        && dense.nrows() == nrows
        && dense.ncols() == ncols
    {
        CustomFamilyPsiLinearMapRef::Dense(dense)
    } else {
        CustomFamilyPsiLinearMapRef::Zero { nrows, ncols }
    }
}


pub(crate) fn second_psi_linear_map<'a>(
    action: Option<&'a CustomFamilyPsiSecondDesignAction>,
    dense: Option<&'a Array2<f64>>,
    nrows: usize,
    ncols: usize,
) -> CustomFamilyPsiLinearMapRef<'a> {
    if let Some(action) = action {
        CustomFamilyPsiLinearMapRef::Second(action)
    } else if let Some(dense) = dense
        && dense.nrows() == nrows
        && dense.ncols() == ncols
    {
        CustomFamilyPsiLinearMapRef::Dense(dense)
    } else {
        CustomFamilyPsiLinearMapRef::Zero { nrows, ncols }
    }
}


pub(crate) struct CustomFamilyJointDesignChannel {
    range: Range<usize>,
    design: DesignMatrix,
    psi_derivative: Option<CustomFamilyPsiDesignAction>,
}


impl CustomFamilyJointDesignChannel {
    pub(crate) fn new<D>(
        range: Range<usize>,
        design: D,
        psi_derivative: Option<CustomFamilyPsiDesignAction>,
    ) -> Self
    where
        D: Into<DesignMatrix>,
    {
        Self {
            range,
            design: design.into(),
            psi_derivative,
        }
    }

    fn coefficients(&self, full: &Array1<f64>) -> Array1<f64> {
        full.slice(ndarray::s![self.range.clone()]).to_owned()
    }

    fn apply(&self, full: &Array1<f64>) -> Array1<f64> {
        let coeffs = self.coefficients(full);
        self.design.matrixvectormultiply(&coeffs)
    }

    fn apply_transpose(&self, values: &Array1<f64>) -> Array1<f64> {
        self.design.transpose_vector_multiply(values)
    }
}


pub(crate) struct CustomFamilyJointDesignPairContribution {
    left_channel: usize,
    right_channel: usize,
    weights: Array1<f64>,
    drift_weights: Array1<f64>,
}


impl CustomFamilyJointDesignPairContribution {
    pub(crate) fn new(
        left_channel: usize,
        right_channel: usize,
        weights: Array1<f64>,
        drift_weights: Array1<f64>,
    ) -> Self {
        Self {
            left_channel,
            right_channel,
            weights,
            drift_weights,
        }
    }
}


pub(crate) struct CustomFamilyJointPsiOperator {
    total_dim: usize,
    channels: Vec<CustomFamilyJointDesignChannel>,
    pair_contributions: Vec<CustomFamilyJointDesignPairContribution>,
    /// Optional dense correction for small cross-blocks (e.g. h/w parameters)
    /// that don't warrant their own weighted-Gram channel.
    dense_correction: Option<Array2<f64>>,
}


impl CustomFamilyJointPsiOperator {
    pub(crate) fn new(
        total_dim: usize,
        channels: Vec<CustomFamilyJointDesignChannel>,
        pair_contributions: Vec<CustomFamilyJointDesignPairContribution>,
    ) -> Self {
        Self {
            total_dim,
            channels,
            pair_contributions,
            dense_correction: None,
        }
    }
}


impl HyperOperator for CustomFamilyJointPsiOperator {
    fn dim(&self) -> usize {
        self.total_dim
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.total_dim);
        let base_vals: Vec<Array1<f64>> = self
            .channels
            .iter()
            .map(|channel| channel.apply(v))
            .collect();
        let deriv_vals: Vec<Option<Array1<f64>>> = self
            .channels
            .iter()
            .map(|channel| {
                channel
                    .psi_derivative
                    .as_ref()
                    .map(|deriv| deriv.forward_mul(v.slice(ndarray::s![channel.range.clone()])))
            })
            .collect();

        let mut out = if let Some(ref corr) = self.dense_correction {
            corr.dot(v)
        } else {
            Array1::<f64>::zeros(self.total_dim)
        };
        for pair in &self.pair_contributions {
            let left = &self.channels[pair.left_channel];
            let right_base = &base_vals[pair.right_channel];
            let weighted_drift = &pair.drift_weights * right_base;
            let mut contrib = left.apply_transpose(&weighted_drift);

            if let Some(left_deriv) = left.psi_derivative.as_ref() {
                let weighted_right = &pair.weights * right_base;
                contrib += &left_deriv.transpose_mul(weighted_right.view());
            }

            if let Some(right_deriv) = deriv_vals[pair.right_channel].as_ref() {
                let weighted_right = &pair.weights * right_deriv;
                contrib += &left.apply_transpose(&weighted_right);
            }

            let mut out_slice = out.slice_mut(ndarray::s![left.range.clone()]);
            out_slice += &contrib;
        }

        out
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        assert_eq!(v.len(), self.total_dim);
        assert_eq!(u.len(), self.total_dim);
        let base_v: Vec<Array1<f64>> = self
            .channels
            .iter()
            .map(|channel| channel.apply(v))
            .collect();
        let base_u: Vec<Array1<f64>> = self
            .channels
            .iter()
            .map(|channel| channel.apply(u))
            .collect();
        let deriv_v: Vec<Option<Array1<f64>>> = self
            .channels
            .iter()
            .map(|channel| {
                channel
                    .psi_derivative
                    .as_ref()
                    .map(|deriv| deriv.forward_mul(v.slice(ndarray::s![channel.range.clone()])))
            })
            .collect();
        let deriv_u: Vec<Option<Array1<f64>>> = self
            .channels
            .iter()
            .map(|channel| {
                channel
                    .psi_derivative
                    .as_ref()
                    .map(|deriv| deriv.forward_mul(u.slice(ndarray::s![channel.range.clone()])))
            })
            .collect();

        let mut total = if let Some(ref corr) = self.dense_correction {
            v.dot(&corr.dot(u))
        } else {
            0.0
        };
        for pair in &self.pair_contributions {
            let left_base_u = &base_u[pair.left_channel];
            let right_base_v = &base_v[pair.right_channel];
            total += left_base_u.dot(&(&pair.drift_weights * right_base_v));

            if let Some(left_deriv_u) = deriv_u[pair.left_channel].as_ref() {
                total += left_deriv_u.dot(&(&pair.weights * right_base_v));
            }
            if let Some(right_deriv_v) = deriv_v[pair.right_channel].as_ref() {
                total += left_base_u.dot(&(&pair.weights * right_deriv_v));
            }
        }

        total
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = self
            .dense_correction
            .clone()
            .unwrap_or_else(|| Array2::<f64>::zeros((self.total_dim, self.total_dim)));
        let mut basis = Array1::<f64>::zeros(self.total_dim);
        for j in 0..self.total_dim {
            basis[j] = 1.0;
            // Use mul_vec without the dense_correction part (already in `out`).
            let base_vals: Vec<Array1<f64>> = self
                .channels
                .iter()
                .map(|channel| channel.apply(&basis))
                .collect();
            let deriv_vals: Vec<Option<Array1<f64>>> = self
                .channels
                .iter()
                .map(|channel| {
                    channel.psi_derivative.as_ref().map(|deriv| {
                        deriv.forward_mul(basis.slice(ndarray::s![channel.range.clone()]))
                    })
                })
                .collect();
            let mut col = Array1::<f64>::zeros(self.total_dim);
            for pair in &self.pair_contributions {
                let left = &self.channels[pair.left_channel];
                let right_base = &base_vals[pair.right_channel];
                let weighted_drift = &pair.drift_weights * right_base;
                let mut contrib = left.apply_transpose(&weighted_drift);
                if let Some(left_deriv) = left.psi_derivative.as_ref() {
                    let weighted_right = &pair.weights * right_base;
                    contrib += &left_deriv.transpose_mul(weighted_right.view());
                }
                if let Some(right_deriv) = deriv_vals[pair.right_channel].as_ref() {
                    let weighted_right = &pair.weights * right_deriv;
                    contrib += &left.apply_transpose(&weighted_right);
                }
                col.slice_mut(ndarray::s![left.range.clone()])
                    .scaled_add(1.0, &contrib);
            }
            out.column_mut(j).scaled_add(1.0, &col);
            basis[j] = 0.0;
        }
        out
    }

    fn is_implicit(&self) -> bool {
        self.dense_correction.is_none()
            && self.channels.iter().any(|channel| {
                channel
                    .psi_derivative
                    .as_ref()
                    .is_some_and(|d| d.is_implicit())
            })
    }
}


fn shared_dense_design_cache() -> &'static Mutex<HashMap<(usize, usize, usize), Weak<Array2<f64>>>>
{
    static CACHE: OnceLock<Mutex<HashMap<(usize, usize, usize), Weak<Array2<f64>>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}


pub(crate) fn shared_dense_arc(x: &Array2<f64>) -> Arc<Array2<f64>> {
    let key = (x.as_ptr() as usize, x.nrows(), x.ncols());
    let cache = shared_dense_design_cache();
    if let Ok(mut guard) = cache.lock() {
        if let Some(shared) = guard.get(&key).and_then(Weak::upgrade) {
            return shared;
        }
        guard.retain(|_, shared| shared.strong_count() > 0);
        let shared = Arc::new(x.clone());
        guard.insert(key, Arc::downgrade(&shared));
        shared
    } else {
        Arc::new(x.clone())
    }
}


pub(crate) fn resolve_custom_family_x_psi_map(
    deriv: &CustomFamilyBlockPsiDerivative,
    n: usize,
    p: usize,
    row_range: Range<usize>,
    label: &str,
    policy: &ResourcePolicy,
) -> Result<PsiDesignMap, String> {
    if row_range.end > n {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "{label}: row range {}..{} exceeds total rows {n}",
                row_range.start, row_range.end
            ),
        }
        .into());
    }

    // Prefer operator action when dimensions match.
    if let Some(op) = deriv.implicit_operator.as_ref()
        && op.n_data() == n
        && op.p_out() == p
    {
        return Ok(PsiDesignMap::First {
            action: CustomFamilyPsiDesignAction::from_first_derivative(
                deriv, n, p, row_range, label,
            )?,
        });
    }

    // Dense fallback guarded by policy.
    if deriv.x_psi.nrows() == n && deriv.x_psi.ncols() == p {
        match policy.derivative_storage_mode {
            DerivativeStorageMode::AnalyticOperatorRequired => {
                if is_zero_array(&deriv.x_psi) {
                    return Ok(PsiDesignMap::Zero {
                        nrows: row_range.end - row_range.start,
                        ncols: p,
                    });
                }
                return Err(CustomFamilyError::UnsupportedConfiguration {
                    reason: format!(
                        "{label}: dense x_psi fallback disabled by AnalyticOperatorRequired"
                    ),
                }
                .into());
            }
            DerivativeStorageMode::MaterializeIfSmall | DerivativeStorageMode::DiagnosticsOnly => {
                let matrix = if row_range.start == 0 && row_range.end == n {
                    Arc::new(deriv.x_psi.clone())
                } else {
                    Arc::new(
                        deriv
                            .x_psi
                            .slice(ndarray::s![row_range.clone(), ..])
                            .to_owned(),
                    )
                };
                return Ok(PsiDesignMap::Dense { matrix });
            }
        }
    }

    // Empty / zero sentinel.
    if deriv.x_psi.nrows() == 0 || deriv.x_psi.ncols() == 0 {
        return Ok(PsiDesignMap::Zero {
            nrows: row_range.end - row_range.start,
            ncols: p,
        });
    }

    Err(CustomFamilyError::DimensionMismatch {
        reason: format!(
            "{label}: x_psi shape {:?} does not match ({n}, {p})",
            deriv.x_psi.dim()
        ),
    }
    .into())
}


pub(crate) fn resolve_custom_family_x_psi_psi_map(
    deriv_i: &CustomFamilyBlockPsiDerivative,
    deriv_j: &CustomFamilyBlockPsiDerivative,
    local_j: usize,
    n: usize,
    p: usize,
    row_range: Range<usize>,
    label: &str,
    policy: &ResourcePolicy,
) -> Result<PsiDesignMap, String> {
    if row_range.end > n {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "{label}: row range {}..{} exceeds total rows {n}",
                row_range.start, row_range.end
            ),
        }
        .into());
    }

    // Prefer operator action when dimensions match.
    if let Some(op) = deriv_i.implicit_operator.as_ref()
        && op.n_data() == n
        && op.p_out() == p
    {
        let same_group = deriv_i.implicit_group_id.is_some()
            && deriv_i.implicit_group_id == deriv_j.implicit_group_id;
        if !same_group {
            return Ok(PsiDesignMap::Zero {
                nrows: row_range.end - row_range.start,
                ncols: p,
            });
        }
        match CustomFamilyPsiSecondDesignAction::from_second_derivative(
            deriv_i,
            deriv_j,
            n,
            p,
            row_range.clone(),
            label,
        )? {
            Some(action) => {
                return Ok(PsiDesignMap::Second { action });
            }
            None => {
                return Ok(PsiDesignMap::Zero {
                    nrows: row_range.end - row_range.start,
                    ncols: p,
                });
            }
        }
    }

    // Dense fallback guarded by policy, reading from the per-second-derivative
    // slot `x_psi_psi[local_j]` if provided.
    if let Some(x_psi_psi) = deriv_i.x_psi_psi.as_ref()
        && let Some(x_ab) = x_psi_psi.get(local_j)
    {
        if x_ab.nrows() == n && x_ab.ncols() == p {
            match policy.derivative_storage_mode {
                DerivativeStorageMode::AnalyticOperatorRequired => {
                    if is_zero_array(x_ab) {
                        return Ok(PsiDesignMap::Zero {
                            nrows: row_range.end - row_range.start,
                            ncols: p,
                        });
                    }
                    return Err(CustomFamilyError::UnsupportedConfiguration {
                        reason: format!(
                            "{label}: dense x_psi_psi fallback disabled by AnalyticOperatorRequired"
                        ),
                    }
                    .into());
                }
                DerivativeStorageMode::MaterializeIfSmall
                | DerivativeStorageMode::DiagnosticsOnly => {
                    let matrix = if row_range.start == 0 && row_range.end == n {
                        Arc::new(x_ab.clone())
                    } else {
                        Arc::new(x_ab.slice(ndarray::s![row_range.clone(), ..]).to_owned())
                    };
                    return Ok(PsiDesignMap::Dense { matrix });
                }
            }
        }
        if x_ab.is_empty() {
            return Ok(PsiDesignMap::Zero {
                nrows: row_range.end - row_range.start,
                ncols: p,
            });
        }
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "{label}: x_psi_psi shape {:?} does not match ({n}, {p})",
                x_ab.dim()
            ),
        }
        .into());
    }

    // No operator, no dense slot: treat as zero.
    Ok(PsiDesignMap::Zero {
        nrows: row_range.end - row_range.start,
        ncols: p,
    })
}


#[derive(Clone)]
pub struct ExactNewtonJointPsiTerms {
    pub objective_psi: f64,
    pub score_psi: Array1<f64>,
    pub hessian_psi: Array2<f64>,
    pub hessian_psi_operator: Option<Arc<dyn HyperOperator>>,
}


impl std::fmt::Debug for ExactNewtonJointPsiTerms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExactNewtonJointPsiTerms")
            .field("objective_psi", &self.objective_psi)
            .field("score_psi", &self.score_psi)
            .field("hessian_psi", &self.hessian_psi)
            .field(
                "hessian_psi_operator",
                &self.hessian_psi_operator.as_ref().map(|_| "<operator>"),
            )
            .finish()
    }
}


impl ExactNewtonJointPsiTerms {
    fn zeros(total: usize) -> Self {
        Self {
            objective_psi: 0.0,
            score_psi: Array1::zeros(total),
            hessian_psi: Array2::zeros((total, total)),
            hessian_psi_operator: None,
        }
    }
}


pub struct ExactNewtonJointPsiSecondOrderTerms {
    pub objective_psi_psi: f64,
    pub score_psi_psi: Array1<f64>,
    pub hessian_psi_psi: Array2<f64>,
    pub hessian_psi_psi_operator: Option<Box<dyn HyperOperator>>,
}


/// Direction-contracted second-order ψ terms for the profiled θ-HVP (#740).
///
/// The per-pair [`ExactNewtonJointPsiSecondOrderTerms`] are the `(ψ_i, ψ_j)`
/// entries of the joint hyper-Hessian; assembling the full outer Hessian from
/// them costs one O(n) family row pass per pair, i.e. `K²·n`. A matrix-free
/// profiled θ-HVP never needs the individual pairs — it needs, for one applied
/// outer direction with ψ-weights `α_ψ`, the `α`-contraction of those pairs
/// against the combined ψ-direction `ψ(α) = Σ_j α_j ψ_j`:
///
/// ```text
///   objective[i] = Σ_j α_j V_{ψ_i ψ_j}
///   score[i]     = Σ_j α_j g_{ψ_i ψ_j}          (a p-vector per output row i)
///   hessian[i]   = Σ_j α_j D²_β H_L[ψ_i, ψ_j]
///                = D²_β H_L[ψ_i, ψ(α)]            (bilinearity)
/// ```
///
/// All `psi_dim` output rows share the SAME contracted second leg `ψ(α)`, so a
/// family that streams its rows once over `ψ(α)` (carrying every fixed first
/// leg `ψ_i` as a batched factor column) produces every row in a SINGLE n-pass.
/// That is the cost the profiled θ-HVP turns into `K·n`-to-densify /
/// `m·n`-in-CG instead of the dense path's `K²·n`.
///
/// Indexing is over the flattened ψ coordinates in the same order as
/// [`ExactNewtonJointPsiWorkspace::second_order_terms`]; `hessian[i]` carries
/// the `D²_β H_L[ψ_i, ψ(α)]` drift as a [`DriftDerivResult`] (dense or
/// operator-backed) plus any block-local `S_{ψ_i ψ_j}` penalty motion folded by
/// the family, exactly mirroring the per-pair `hessian_psi_psi(_operator)`.
pub struct ExactNewtonJointPsiSecondOrderContracted {
    /// `objective[i] = Σ_j α_j V_{ψ_i ψ_j}`, one scalar per ψ output row.
    pub objective: Array1<f64>,
    /// `score[i] = Σ_j α_j g_{ψ_i ψ_j}`, the `psi_dim × total` matrix whose
    /// row `i` is the contracted fixed-β score derivative for output row `i`.
    pub score: Array2<f64>,
    /// `hessian[i] = D²_β H_L[ψ_i, ψ(α)]` for each ψ output row `i`.
    pub hessian: Vec<DriftDerivResult>,
}


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointHessianSourcePreference {
    Dense,
    Operator,
}


/// What the consumer is going to *do* with the joint Hessian. This is the
/// intent half of #738's capability-vs-representation split: the call site
/// states what it needs, and the workspace picks the cheapest representation
/// that serves that need (rather than a single per-workspace preference being
/// applied uniformly regardless of how the result is consumed).
///
/// The distinction matters because the same workspace serves several
/// consumers with opposite ideal representations:
/// - the inner Newton/PCG solve only ever applies `H · v`, so a matrix-free
///   HVP (`Operator`) is ideal and a dense build is pure waste;
/// - the REML logdet term factorizes `H + S_λ` (Cholesky / eigendecomposition),
///   so it must hold a dense matrix anyway — handing it an `Operator` only
///   forces an immediate column-basis (or `dense_forced`) re-materialization,
///   so a workspace with a structural direct-dense build should answer `Dense`
///   here and skip the operator wrapper entirely.
///
/// Workspaces refine their representation choice per intent via
/// [`ExactNewtonJointHessianWorkspace::hessian_source_preference_for_intent`];
/// the default keeps the legacy single-preference behaviour so existing
/// workspaces are unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterializationIntent {
    /// Inner Newton / PCG solve — only applies `H · v`. Matrix-free is ideal.
    InnerSolve,
    /// REML/LAML logdet term — factorizes `H + S_λ`, needs a dense matrix.
    LogdetFactorization,
    /// Outer-Hessian / EFS evaluation — builds the joint hyper terms; today
    /// these route through the same source as the gradient path.
    OuterEvaluation,
    /// Outer-gradient / IFT term assembly.
    OuterGradient,
}


pub trait ExactNewtonJointHessianWorkspace: Send + Sync {
    /// Pre-build any per-row jet caches the workspace will hand to the
    /// outer-eval directional-derivative path. Called once when the
    /// `compute_dh` / `compute_d2h` closures are wired up at top-level
    /// rayon, *before* the outer ext-coordinate `par_iter` enters. The
    /// alternative — letting the cache materialise lazily on first call
    /// from inside the outer `par_iter` — collapses the build's own
    /// `par_iter` to a single worker (the seven other workers are parked
    /// on the cache's `OnceLock`). Default impl is a no-op for workspaces
    /// with no per-row jet cache.
    ///
    /// Deliberately not called from PIRLS-side workspaces (which never
    /// invoke `directional_derivative_operator` and would pay the prime
    /// cost without ever consuming the cache).
    fn warm_up_outer_caches(&self) -> Result<(), String> {
        Ok(())
    }

    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    /// Preferred representation for callers that can consume either the dense
    /// coefficient Hessian or the matrix-free HVP source.
    fn hessian_source_preference(&self) -> JointHessianSourcePreference {
        JointHessianSourcePreference::Dense
    }

    /// Intent-aware representation choice (#738). Given what the consumer is
    /// about to do with the Hessian ([`MaterializationIntent`]), return the
    /// representation the workspace prefers to hand back. The default keeps the
    /// legacy intent-blind behaviour by delegating to
    /// [`Self::hessian_source_preference`], so existing workspaces are
    /// unchanged. Workspaces with a structural direct-dense build that also
    /// expose a matrix-free HVP override this to answer `Operator` for
    /// [`MaterializationIntent::InnerSolve`] (stream the HVP) and `Dense` for
    /// [`MaterializationIntent::LogdetFactorization`] (the consumer factorizes,
    /// so building the operator wrapper only to re-densify it is pure waste).
    fn hessian_source_preference_for_intent(
        &self,
        intent: MaterializationIntent,
    ) -> JointHessianSourcePreference {
        // Intent-agnostic default: every intent maps to the single legacy
        // preference. Implementors that benefit from per-intent representation
        // (e.g. CTN: dense for logdet, operator for inner solve) override this.
        match intent {
            MaterializationIntent::InnerSolve
            | MaterializationIntent::LogdetFactorization
            | MaterializationIntent::OuterEvaluation
            | MaterializationIntent::OuterGradient => self.hessian_source_preference(),
        }
    }

    /// Forced dense materialization that bypasses any amortization gate the
    /// workspace applies to `hessian_dense`. Callers that genuinely need a
    /// dense matrix (logdet, factorize-based QP solves) use this so they pay
    /// the workspace's structural direct-dense build cost rather than the
    /// caller-side column-basis HVP fallback. Returning `None` means the
    /// workspace has no preferred direct-dense path and the caller should
    /// fall back to column-basis HVP via `hessian_matvec` / `apply`.
    fn hessian_dense_forced(&self) -> Result<Option<Array2<f64>>, String> {
        self.hessian_dense()
    }

    fn joint_log_likelihood_evaluation(&self) -> Result<Option<f64>, String> {
        Ok(None)
    }

    fn joint_gradient_evaluation(
        &self,
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        Ok(None)
    }

    /// Whether `hessian_matvec` / `hessian_matvec_into` will return `Some`.
    /// A cheap synchronisation-free flag consulted by
    /// `exact_newton_joint_hessian_source_from_workspace` to decide whether
    /// to construct a matrix-free `JointHessianSource::Operator` variant.
    /// Returning `false` is equivalent to returning `Ok(None)` from
    /// `hessian_matvec` but avoids allocating and running a full HVP sweep
    /// against a zero vector just to discover unavailability.
    /// Default is `false` matching the base-trait `hessian_matvec` returning
    /// `Ok(None)`. Concrete impls that override `hessian_matvec` must also
    /// override this to return `true`.
    fn hessian_matvec_available(&self) -> bool {
        false
    }

    fn hessian_matvec(&self, arr: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(None)
    }

    /// Write-into variant of `hessian_matvec`. The default implementation
    /// delegates to the legacy owned-return form and copies the result into
    /// `out`, providing back-compat without per-impl work. Concrete impls in
    /// the inner-Newton large-scale hot path (Bernoulli marginal-slope and
    /// survival marginal-slope) override this to write directly into the
    /// caller-owned buffer, eliminating per-PCG-iter `Array1` allocations.
    fn hessian_matvec_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<bool, String> {
        match self.hessian_matvec(v)? {
            Some(result) => {
                if result.len() != out.len() {
                    return Err(CustomFamilyError::DimensionMismatch {
                        reason: format!(
                            "hessian_matvec_into: result length {} != out length {}",
                            result.len(),
                            out.len()
                        ),
                    }
                    .into());
                }
                out.assign(&result);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Batched multi-RHS Hessian apply: writes `H · V` into `out`, where `V`
    /// and `out` are `(total, n_rhs)` with each column an independent
    /// direction. Returns `Ok(true)` when the apply was performed and
    /// `Ok(false)` when the workspace exposes no matrix-free apply (mirroring
    /// `hessian_matvec_into`).
    ///
    /// The default implementation applies `hessian_matvec_into` column by
    /// column, so every existing workspace gets a correct batched apply for
    /// free and the batched result is, column for column, **numerically
    /// identical** to looping the single-vector HVP. Workspaces whose Hessian
    /// is `Σ_i Jᵢᵀ Hᵢ Jᵢ` over a streamed/tiled per-row primary Hessian `Hᵢ`
    /// (Bernoulli marginal-slope) override this to sweep each row tile **once**
    /// and apply its `Hᵢ` to all `n_rhs` columns in that single pass — the
    /// per-tile `Hᵢ` read and the design-row projection are then amortised
    /// across every RHS instead of paid once per column. This is the
    /// representation that makes dense reconstruction of a matrix-free operator
    /// (`H = H · [e_0 | … | e_{p-1}]`) one tile sweep wide instead of `p`.
    fn hessian_apply_mat(
        &self,
        v_cols: &Array2<f64>,
        out: &mut Array2<f64>,
    ) -> Result<bool, String> {
        if v_cols.nrows() != out.nrows() || v_cols.ncols() != out.ncols() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "hessian_apply_mat: v_cols {}x{} != out {}x{}",
                    v_cols.nrows(),
                    v_cols.ncols(),
                    out.nrows(),
                    out.ncols()
                ),
            }
            .into());
        }
        let total = v_cols.nrows();
        let mut col_in = Array1::<f64>::zeros(total);
        let mut col_out = Array1::<f64>::zeros(total);
        for col in 0..v_cols.ncols() {
            col_in.assign(&v_cols.column(col));
            if !self.hessian_matvec_into(&col_in, &mut col_out)? {
                return Ok(false);
            }
            out.column_mut(col).assign(&col_out);
        }
        Ok(true)
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(None)
    }

    /// Exact row-local contractions for
    /// `trace(F^T · D_beta H[d_j] · F)` over many coefficient directions.
    ///
    /// Workspaces that own the current row cache can implement this to avoid
    /// rebuilding row contexts or materializing each `D_beta H[d_j]` as a
    /// coefficient-space operator when the caller only needs its projected
    /// trace against the fixed logdet factor `F`.
    fn projected_directional_derivative_traces(
        &self,
        factor: &Array2<f64>,
        directions: &Array2<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        assert_eq!(
            factor.nrows(),
            directions.nrows(),
            "projected directional derivative traces require shared coefficient dimension"
        );
        Ok(None)
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String>;

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        Ok(self.directional_derivative(d_beta_flat)?.map(|matrix| {
            Arc::new(crate::solver::estimate::reml::unified::DenseMatrixHyperOperator { matrix })
                as Arc<dyn HyperOperator>
        }))
    }

    fn directional_derivative_operators(
        &self,
        d_beta_flats: &[Array1<f64>],
    ) -> Result<Vec<Option<Arc<dyn HyperOperator>>>, String> {
        d_beta_flats
            .iter()
            .map(|d_beta_flat| self.directional_derivative_operator(d_beta_flat))
            .collect()
    }

    fn second_directional_derivative(
        &self,
        arr: &Array1<f64>,
        arr2: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        assert!(arr2.iter().all(|v| !v.is_nan()));
        Ok(None)
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        Ok(self
            .second_directional_derivative(d_beta_u, d_beta_v)?
            .map(|matrix| {
                Arc::new(
                    crate::solver::estimate::reml::unified::DenseMatrixHyperOperator { matrix },
                ) as Arc<dyn HyperOperator>
            }))
    }

    fn second_directional_derivative_operators(
        &self,
        d_beta_pairs: &[(Array1<f64>, Array1<f64>)],
    ) -> Result<Vec<Option<Arc<dyn HyperOperator>>>, String> {
        d_beta_pairs
            .iter()
            .map(|(u, v)| self.second_directional_derivative_operator(u, v))
            .collect()
    }
}


pub trait ExactNewtonJointPsiWorkspace: Send + Sync {
    fn first_order_terms(&self, idx: usize) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        assert!(idx < usize::MAX);
        Ok(None)
    }

    fn first_order_terms_all(&self) -> Result<Option<Vec<ExactNewtonJointPsiTerms>>, String> {
        Ok(None)
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String>;

    /// Direction-contracted second-order ψ terms for the profiled θ-HVP (#740).
    ///
    /// Given the ψ-block weights `alpha_psi` (length `psi_dim`, the ψ slice of
    /// one applied outer direction α), return the `α`-contraction of every
    /// `(ψ_i, ψ_j)` second-order term against the combined ψ-direction
    /// `ψ(α) = Σ_j alpha_psi[j] · ψ_j`, as
    /// [`ExactNewtonJointPsiSecondOrderContracted`]. A family that can stream
    /// its rows once over `ψ(α)` overrides this so the profiled outer-Hessian
    /// operator applies one combined-direction n-pass per matvec instead of the
    /// dense path's `K²` per-pair [`Self::second_order_terms`] passes.
    ///
    /// Default returns `None`: the profiled θ-HVP operator is then not built and
    /// the evaluator keeps the exact per-pair assembly (dense
    /// `compute_outer_hessian` / `build_outer_hessian_operator`). Overriding
    /// this method is purely a representation/cost choice — it must produce the
    /// exact same contraction the per-pair terms would, which the
    /// `profiled_theta_hvp_outer_hessian_fd` finite-difference cross-check
    /// guards.
    fn second_order_terms_contracted(
        &self,
        alpha_psi: &[f64],
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderContracted>, String> {
        assert!(alpha_psi.len() < usize::MAX);
        Ok(None)
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String>;
}


pub(crate) struct ExactNewtonJointPsiDirectCache<T> {
    entries: Vec<Mutex<Option<Option<Arc<T>>>>>,
    lru: Mutex<std::collections::VecDeque<usize>>,
    limit: usize,
}


impl<T> ExactNewtonJointPsiDirectCache<T> {
    pub(crate) fn new(len: usize) -> Self {
        Self {
            entries: (0..len).map(|_| Mutex::new(None)).collect(),
            lru: Mutex::new(std::collections::VecDeque::new()),
            limit: len,
        }
    }

    fn touch_lru(&self, index: usize) -> Result<(), String> {
        let mut lru = self
            .lru
            .lock()
            .map_err(|_| "joint psi direct cache lru poisoned".to_string())?;
        lru.retain(|&existing| existing != index);
        lru.push_back(index);
        while lru.len() > self.limit {
            let Some(evict_index) = lru.pop_front() else {
                break;
            };
            if evict_index == index {
                continue;
            }
            if let Some(entry) = self.entries.get(evict_index) {
                let mut guard = entry
                    .lock()
                    .map_err(|_| "joint psi direct cache poisoned".to_string())?;
                *guard = None;
            }
        }
        Ok(())
    }

    pub(crate) fn get_or_try_init<F>(&self, index: usize, init: F) -> Result<Option<Arc<T>>, String>
    where
        F: FnOnce() -> Result<Option<T>, String>,
    {
        let Some(entry) = self.entries.get(index) else {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "psi cache index {index} out of bounds for size {}",
                    self.entries.len()
                ),
            }
            .into());
        };
        {
            let guard = entry
                .lock()
                .map_err(|_| "joint psi direct cache poisoned".to_string())?;
            if let Some(cached) = guard.as_ref() {
                let cached = cached.clone();
                // release-early-on-purpose: update LRU after releasing the entry mutex.
                drop(guard);
                self.touch_lru(index)?;
                return Ok(cached);
            }
        }

        let computed = init()?.map(Arc::new);
        let mut guard = entry
            .lock()
            .map_err(|_| "joint psi direct cache poisoned".to_string())?;
        let cached = guard.get_or_insert_with(|| computed.clone());
        let out = cached.clone();
        // release-early-on-purpose: update LRU after releasing the entry mutex.
        drop(guard);
        self.touch_lru(index)?;
        Ok(out)
    }
}


#[derive(Clone)]
pub struct CustomFamilyWarmStart {
    inner: ConstrainedWarmStart,
}


impl CustomFamilyWarmStart {
    pub(crate) fn compatible_with_rho(&self, rho: &Array1<f64>) -> bool {
        screened_outer_warm_start(Some(&self.inner), rho).is_some()
    }

    pub(crate) fn block_beta_len(&self, block_idx: usize) -> Option<usize> {
        self.inner.block_beta.get(block_idx).map(|beta| beta.len())
    }

    pub(crate) fn block_beta_abs_argmax_in_range(
        &self,
        block_idx: usize,
        range: std::ops::Range<usize>,
    ) -> Option<(usize, f64)> {
        let beta = self.inner.block_beta.get(block_idx)?;
        let end = range.end.min(beta.len());
        if range.start >= end {
            return None;
        }
        beta.slice(s![range.start..end])
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, value)| (range.start + idx, value.abs()))
            .filter(|(_, abs)| abs.is_finite())
            .max_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Build a warm-start payload from a flat cached β and the per-block
    /// coefficient widths. The returned warm-start carries a zero `rho`
    /// (the outer cache will overwrite it on the next eval) and empty
    /// active sets; only the per-block β slices feed the next inner
    /// PIRLS / Newton solve. Used by the spatial-joint outer cache to
    /// seed the family-owned warm-start slot on cache hits so the inner
    /// solve opens at the prior converged iterate instead of cold β.
    pub fn from_cached_beta(
        block_col_counts: &[usize],
        beta: &Array1<f64>,
    ) -> Result<Self, EstimationError> {
        let expected: usize = block_col_counts.iter().copied().sum();
        if beta.len() != expected {
            crate::bail_invalid_estim!(
                "cached inner beta has length {}, but spatial-joint blocks require length {}",
                beta.len(),
                expected
            );
        }
        crate::families::marginal_slope_shared::bail_if_cached_beta_non_finite(beta)?;
        let mut offset = 0usize;
        let mut block_beta = Vec::with_capacity(block_col_counts.len());
        for &width in block_col_counts {
            let end = offset + width;
            block_beta.push(beta.slice(s![offset..end]).to_owned());
            offset = end;
        }
        Ok(CustomFamilyWarmStart {
            inner: ConstrainedWarmStart {
                rho: Array1::zeros(0),
                block_beta,
                active_sets: vec![None; block_col_counts.len()],
                cached_inner: None,
            },
        })
    }
}


struct CustomOuterState {
    warm_cache: Option<ConstrainedWarmStart>,
    reset_warm_cache: Option<ConstrainedWarmStart>,
    last_error: Option<String>,
    initial_gradient_norm: Option<f64>,
}


impl CustomOuterState {
    fn new(warm_start: Option<ConstrainedWarmStart>) -> Self {
        Self {
            warm_cache: warm_start.clone(),
            reset_warm_cache: warm_start,
            last_error: None,
            initial_gradient_norm: None,
        }
    }

    fn reset(&mut self) {
        self.warm_cache = self.reset_warm_cache.clone();
    }

    fn seed_cached_beta(
        &mut self,
        rho_dim: usize,
        specs: &[ParameterBlockSpec],
        beta: &Array1<f64>,
    ) -> Result<(), EstimationError> {
        let warm_start = constrained_warm_start_from_cached_beta(rho_dim, specs, beta)?;
        self.reset_warm_cache = Some(warm_start.clone());
        self.warm_cache = Some(warm_start);
        self.last_error = None;
        Ok(())
    }
}


pub struct CustomFamilyJointHyperResult {
    pub objective: f64,
    pub gradient: Array1<f64>,
    pub outer_hessian: crate::solver::outer_strategy::HessianResult,
    pub warm_start: CustomFamilyWarmStart,
    /// `false` when the inner blockwise/Newton solve hit its divergence
    /// early-exit or its max-cycle cap. Envelope-theorem outer gradients
    /// and analytic outer Hessians are valid only at a stationary β̂ —
    /// callers that consume `gradient`/`outer_hessian` MUST gate on this
    /// flag and treat non-converged evaluations as inexact (e.g. let ARC
    /// back off the trust region) rather than feeding pathological
    /// derivatives into the outer optimizer.
    pub inner_converged: bool,
}


pub struct CustomFamilyJointHyperEfsResult {
    pub efs_eval: crate::solver::outer_strategy::EfsEval,
    pub warm_start: CustomFamilyWarmStart,
    /// See [`CustomFamilyJointHyperResult::inner_converged`]. EFS gradients
    /// also assume a stationary inner solve.
    pub inner_converged: bool,
}


struct OuterObjectiveEvalResult {
    objective: f64,
    gradient: Array1<f64>,
    outer_hessian: crate::solver::outer_strategy::HessianResult,
    warm_start: ConstrainedWarmStart,
    inner_converged: bool,
}


fn outer_eval_result_to_joint_hyper_result(
    result: OuterObjectiveEvalResult,
) -> CustomFamilyJointHyperResult {
    CustomFamilyJointHyperResult {
        objective: result.objective,
        gradient: result.gradient,
        outer_hessian: result.outer_hessian,
        warm_start: CustomFamilyWarmStart {
            inner: result.warm_start,
        },
        inner_converged: result.inner_converged,
    }
}


struct OwnedDenseOuterHessianOperator {
    matrix: Array2<f64>,
}


impl crate::solver::outer_strategy::OuterHessianOperator for OwnedDenseOuterHessianOperator {
    fn dim(&self) -> usize {
        self.matrix.nrows()
    }

    fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
        if v.len() != self.matrix.ncols() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "batched dense outer Hessian matvec length mismatch: got {}, expected {}",
                    v.len(),
                    self.matrix.ncols()
                ),
            }
            .into());
        }
        Ok(self.matrix.dot(v))
    }

    /// Zero-alloc override: write `matrix · v` directly into `out` using a
    /// row-dot loop, avoiding the `matrix.dot(v)` allocation.
    fn apply_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<(), String> {
        if v.len() != self.matrix.ncols() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "batched dense outer Hessian apply_into input length mismatch: got {}, expected {}",
                    v.len(),
                    self.matrix.ncols()
                ),
            }
            .into());
        }
        if out.len() != self.matrix.nrows() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "batched dense outer Hessian apply_into output length mismatch: got {}, expected {}",
                    out.len(),
                    self.matrix.nrows()
                ),
            }
            .into());
        }
        for (row, cell) in self.matrix.rows().into_iter().zip(out.iter_mut()) {
            *cell = row.dot(v);
        }
        Ok(())
    }

    fn is_cheap_to_materialize(&self) -> bool {
        true
    }
}


struct LabeledOuterHessianOperator {
    base: Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>,
    physical_to_outer: Vec<Option<usize>>,
    outer_dim: usize,
    /// Scratch buffers reused across `apply_into` calls to avoid
    /// per-call allocation of the permuted input and output vectors.
    /// `(physical_in, physical_out)`, each of length `physical_to_outer.len()`.
    scratch: std::sync::Mutex<(ndarray::Array1<f64>, ndarray::Array1<f64>)>,
}


impl LabeledOuterHessianOperator {
    fn new(
        base: Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>,
        layout: &PenaltyLabelLayout,
    ) -> Self {
        let n_physical = layout.physical_to_outer.len();
        Self {
            base,
            physical_to_outer: layout.physical_to_outer.clone(),
            outer_dim: layout.initial_rho.len(),
            scratch: std::sync::Mutex::new((
                ndarray::Array1::zeros(n_physical),
                ndarray::Array1::zeros(n_physical),
            )),
        }
    }
}


impl crate::solver::outer_strategy::OuterHessianOperator for LabeledOuterHessianOperator {
    fn dim(&self) -> usize {
        self.outer_dim
    }

    fn matvec(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
        if v.len() != self.outer_dim {
            return Err(format!(
                "labeled outer Hessian input length mismatch: got {}, expected {}",
                v.len(),
                self.outer_dim
            ));
        }
        let mut physical = Array1::<f64>::zeros(self.physical_to_outer.len());
        for (physical_idx, outer_idx) in self.physical_to_outer.iter().enumerate() {
            physical[physical_idx] = outer_idx.map(|idx| v[idx]).unwrap_or(0.0);
        }
        let physical_out = self.base.matvec(&physical)?;
        if physical_out.len() != self.physical_to_outer.len() {
            return Err(format!(
                "labeled outer Hessian physical matvec length mismatch: got {}, expected {}",
                physical_out.len(),
                self.physical_to_outer.len()
            ));
        }
        let mut out = Array1::<f64>::zeros(self.outer_dim);
        for (physical_idx, outer_idx) in self.physical_to_outer.iter().enumerate() {
            if let Some(outer_idx) = *outer_idx {
                out[outer_idx] += physical_out[physical_idx];
            }
        }
        Ok(out)
    }

    /// Zero-alloc override: reuses hoisted scratch buffers to avoid the
    /// per-call `physical` and `out` allocations in `matvec`.
    fn apply_into(
        &self,
        v: &ndarray::Array1<f64>,
        out: &mut ndarray::Array1<f64>,
    ) -> Result<(), String> {
        if v.len() != self.outer_dim {
            return Err(format!(
                "labeled outer Hessian apply_into input length mismatch: got {}, expected {}",
                v.len(),
                self.outer_dim
            ));
        }
        if out.len() != self.outer_dim {
            return Err(format!(
                "labeled outer Hessian apply_into output length mismatch: got {}, expected {}",
                out.len(),
                self.outer_dim
            ));
        }
        let mut guard = self
            .scratch
            .lock()
            .map_err(|_| "labeled outer Hessian scratch lock poisoned".to_string())?;
        let (physical_in, physical_out) = &mut *guard;
        for (physical_idx, outer_idx) in self.physical_to_outer.iter().enumerate() {
            physical_in[physical_idx] = outer_idx.map(|idx| v[idx]).unwrap_or(0.0);
        }
        self.base.apply_into(physical_in, physical_out)?;
        if physical_out.len() != self.physical_to_outer.len() {
            return Err(format!(
                "labeled outer Hessian physical apply_into length mismatch: got {}, expected {}",
                physical_out.len(),
                self.physical_to_outer.len()
            ));
        }
        out.fill(0.0);
        for (physical_idx, outer_idx) in self.physical_to_outer.iter().enumerate() {
            if let Some(outer_idx) = *outer_idx {
                out[outer_idx] += physical_out[physical_idx];
            }
        }
        Ok(())
    }

    fn mul_mat(&self, factor: ndarray::ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
        if factor.nrows() != self.outer_dim {
            return Err(format!(
                "labeled outer Hessian factor row mismatch: got {}, expected {}",
                factor.nrows(),
                self.outer_dim
            ));
        }
        let mut physical_factor =
            Array2::<f64>::zeros((self.physical_to_outer.len(), factor.ncols()));
        for (physical_idx, outer_idx) in self.physical_to_outer.iter().enumerate() {
            if let Some(outer_idx) = *outer_idx {
                physical_factor
                    .row_mut(physical_idx)
                    .assign(&factor.row(outer_idx));
            }
        }
        let physical_out = self.base.mul_mat(physical_factor.view())?;
        if physical_out.nrows() != self.physical_to_outer.len() {
            return Err(format!(
                "labeled outer Hessian physical output row mismatch: got {}, expected {}",
                physical_out.nrows(),
                self.physical_to_outer.len()
            ));
        }
        let mut out = Array2::<f64>::zeros((self.outer_dim, factor.ncols()));
        for (physical_idx, outer_idx) in self.physical_to_outer.iter().enumerate() {
            if let Some(outer_idx) = *outer_idx {
                let physical_row = physical_out.row(physical_idx);
                out.row_mut(outer_idx).scaled_add(1.0, &physical_row);
            }
        }
        Ok(out)
    }

    fn is_cheap_to_materialize(&self) -> bool {
        self.base.is_cheap_to_materialize()
    }

    fn materialization_capability(
        &self,
    ) -> crate::solver::outer_strategy::OuterHessianMaterialization {
        self.base.materialization_capability()
    }
}


fn custom_family_batched_outer_hessian_operator<F: CustomFamily>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    rho: &Array1<f64>,
    workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    eval_mode: EvalMode,
) -> Result<Option<Arc<dyn crate::solver::outer_strategy::OuterHessianOperator>>, String> {
    if eval_mode != EvalMode::ValueGradientHessian {
        return Ok(None);
    }
    let Some(terms) =
        family.batched_outer_hessian_terms(states, specs, derivative_blocks, rho, workspace)?
    else {
        return Ok(None);
    };
    match terms.outer_hessian {
        crate::solver::outer_strategy::HessianResult::Operator(operator) => Ok(Some(operator)),
        crate::solver::outer_strategy::HessianResult::Analytic(matrix) => {
            Ok(Some(Arc::new(OwnedDenseOuterHessianOperator { matrix })))
        }
        crate::solver::outer_strategy::HessianResult::Unavailable => Ok(None),
    }
}


fn outer_efs_result_to_joint_hyper_efs_result(
    efs_eval: crate::solver::outer_strategy::EfsEval,
    warm_start: ConstrainedWarmStart,
    inner_converged: bool,
) -> CustomFamilyJointHyperEfsResult {
    CustomFamilyJointHyperEfsResult {
        efs_eval,
        warm_start: CustomFamilyWarmStart { inner: warm_start },
        inner_converged,
    }
}


// Unified exact joint hyper-calculus over theta = [rho, psi].
//
// The correct outer problem is not “a rho objective plus a separate psi
// objective”. It is one profiled/Laplace surface over one flattened hypervector
//
//   theta = [rho, psi],
//
// one flattened joint coefficient vector
//
//   beta = [beta_1; ...; beta_B],
//
// and one joint exact mode system
//
//   F(beta, theta) := V_beta(beta, theta) = 0,
//   H(beta, theta) := V_beta_beta(beta, theta).
//
// For every hypercoordinate theta_i we need the fixed-beta objects
//
//   V_i = partial_{theta_i} V,
//   g_i = partial_{theta_i} F,
//   H_i = partial_{theta_i} H,
//
// and for every pair (i, j)
//
//   V_ij, g_ij, H_ij,
//
// together with the beta-curvature contractions
//
//   D_beta H[u],
//   D_beta^2 H[u, v],
//   T_i[u] := D_beta H_i[u].
//
// The exact profiled mode response and total Hessian drifts are then
//
//   beta_i  = -H^{-1} g_i,
//   beta_ij = -H^{-1}(g_ij + H_i beta_j + H_j beta_i + D_beta H[beta_i] beta_j),
//
//   dot H_i
//   = H_i + D_beta H[beta_i],
//
//   ddot H_ij
//   = H_ij
//     + T_i[beta_j]
//     + T_j[beta_i]
//     + D_beta H[beta_ij]
//     + D_beta^2 H[beta_i, beta_j].
//
// Hence the exact joint profiled/Laplace derivatives are
//
//   J_i
//   = V_i + 0.5 tr(H^{-1} dot H_i) - 0.5 partial_i log|S(theta)|_+,
//
//   J_ij
//   = (V_ij - g_i^T H^{-1} g_j)
//     + 0.5 [ tr(H^{-1} ddot H_ij)
//             - tr(H^{-1} dot H_j H^{-1} dot H_i) ]
//     - 0.5 partial^2_{ij} log|S(theta)|_+.
//
// In this unified view rho and psi are the same outer calculus. They differ
// only in where their fixed-beta derivative objects come from:
//
// - rho coordinates often contribute only through the penalty surface,
//     but the generic assembler intentionally treats the penalty as S(theta),
//     not S(rho), so mixed rho/psi penalty terms are allowed whenever realized
//     component penalties move with psi:
//       V_i  = D_i  + 0.5 beta^T S_i beta
//       g_i  = D_beta_i  + S_i beta
//       H_i  = D_beta_beta_i + S_i
//       V_ij = D_ij + 0.5 beta^T S_ij beta
//       g_ij = D_beta_ij + S_ij beta
//       H_ij = D_beta_beta_ij + S_ij.
//
// - psi coordinates come from the family-specific joint exact psi hooks, while
//   the generic assembler still owns any realized-penalty motion through
//   S_i / S_ij:
//     objective_psi            <-> V_i
//     score_psi                <-> g_i
//     hessian_psi              <-> H_i
//     objective_psi_psi        <-> V_ij
//     score_psi_psi            <-> g_ij
//     hessian_psi_psi          <-> H_ij
//     D_beta H_psi[u]          <-> T_i[u].
//
// For coupled families this means any block-local psi path is wrong. Even when
// g_i is sparse or penalty-local, beta_i is defined by the full joint solve
//
//   beta_i = -H^{-1} g_i,
//
// so every exact outer derivative must be assembled in this joint flattened
// space.

#[derive(Debug, Clone, Error)]
pub enum CustomFamilyError {
    #[error("custom-family invalid input in {context}: {reason}")]
    InvalidInput {
        context: &'static str,
        reason: String,
    },
    #[error("custom-family optimization error in {context}: {reason}")]
    Optimization {
        context: &'static str,
        reason: String,
    },
    #[error("{reason}")]
    DimensionMismatch { reason: String },
    #[error("{reason}")]
    NumericalFailure { reason: String },
    #[error("{reason}")]
    ConstraintViolation { reason: String },
    #[error("{reason}")]
    UnsupportedConfiguration { reason: String },
    #[error("{reason}")]
    BasisDecompositionFailed { reason: String },
    /// Pre-fit cross-block identifiability audit refused the fit. The
    /// joint design across `ParameterBlockSpec`s carries a rank
    /// deficiency that the post-`joint_null_rotation` absorption did
    /// not resolve: two or more blocks contribute the same direction,
    /// or a structural >2-way alias was detected without per-pair
    /// attribution. The full `IdentifiabilityAudit` is held so
    /// consumers (logs, structured-error sinks, the seed driver's
    /// classifier) can extract the alias pairs and the summary string
    /// without reparsing.
    #[error("identifiability audit refused the fit: {}", audit.summary)]
    IdentifiabilityFailure {
        audit: crate::solver::identifiability_audit::IdentifiabilityAudit,
    },
    /// MAP estimate uniqueness condition `ker(J^T W J) ∩ ker(S) = {0}` is
    /// violated.  A null direction of `J^T W J` carries zero penalty
    /// curvature, so the posterior is flat along that direction and the
    /// MAP is non-unique.  The structured [`MapUniquenessError`] names the
    /// dominant block so the caller can add the missing penalty or remove
    /// the unpenalised direction.
    #[error("MAP estimate non-unique: {}", error)]
    MapUniquenessFailure {
        error: crate::solver::identifiability_audit::MapUniquenessError,
    },
}


impl From<String> for CustomFamilyError {
    fn from(value: String) -> Self {
        Self::InvalidInput {
            context: "custom-family string boundary",
            reason: value,
        }
    }
}


impl From<CustomFamilyError> for String {
    fn from(value: CustomFamilyError) -> Self {
        value.to_string()
    }
}


pub(crate) fn validate_blockspecs(specs: &[ParameterBlockSpec]) -> Result<Vec<usize>, String> {
    // `fit_custom_family` is a fit entry point and genuinely requires at least
    // one parameter block — an empty model has nothing to estimate. This is a
    // *fit-level precondition*, distinct from the *consistency* of the block
    // specs themselves, which is checked by `validate_blockspec_consistency`.
    if specs.is_empty() {
        return Err(CustomFamilyError::UnsupportedConfiguration {
            reason: "fit_custom_family requires at least one parameter block".to_string(),
        }
        .into());
    }
    validate_blockspec_consistency(specs)
}


/// Validate the *internal consistency* of a slice of parameter block specs
/// (unique names; design/offset/initial_beta/penalty dimensions agree) without
/// imposing the fit-level "at least one block" precondition.
///
/// An empty slice is vacuously consistent and returns an empty penalty-count
/// vector. The non-empty fit precondition lives in [`validate_blockspecs`];
/// pure operator-materialization hooks (e.g. `batched_outer_hessian_terms`)
/// must use this consistency check instead, so they can be probed with an
/// empty, self-consistent argument set without tripping a fit precondition
/// that does not apply to them.
pub(crate) fn validate_blockspec_consistency(
    specs: &[ParameterBlockSpec],
) -> Result<Vec<usize>, String> {
    let mut seen_names = BTreeMap::<String, usize>::new();
    for (b, spec) in specs.iter().enumerate() {
        if let Some(prev) = seen_names.insert(spec.name.clone(), b) {
            return Err(CustomFamilyError::ConstraintViolation {
                reason: format!(
                    "duplicate parameter block name '{}' at indices {prev} and {b}: block names must be unique so coefficient labels resolved by name are unambiguous",
                    spec.name
                ),
            }
            .into());
        }
    }
    let mut penalty_counts = Vec::with_capacity(specs.len());
    for (b, spec) in specs.iter().enumerate() {
        let n = spec.design.nrows();
        if spec.offset.len() != n {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {b} offset length mismatch: got {}, expected {}",
                    spec.offset.len(),
                    n
                ),
            }
            .into());
        }
        // `stacked_design` and `stacked_offset` must be `Some` together
        // and their row/length must agree.  This enforces the contract
        // that `solver_design()` and `solver_offset()` always return a
        // matched pair.
        match (&spec.stacked_design, &spec.stacked_offset) {
            (Some(sd), Some(so)) => {
                if sd.nrows() != so.len() {
                    return Err(CustomFamilyError::DimensionMismatch {
                        reason: format!(
                            "block {b} stacked_design/stacked_offset row mismatch: \
                             stacked_design.nrows()={}, stacked_offset.len()={}",
                            sd.nrows(),
                            so.len(),
                        ),
                    }
                    .into());
                }
                if sd.ncols() != spec.design.ncols() {
                    return Err(CustomFamilyError::DimensionMismatch {
                        reason: format!(
                            "block {b} stacked_design column count {} disagrees with \
                             design column count {}",
                            sd.ncols(),
                            spec.design.ncols(),
                        ),
                    }
                    .into());
                }
            }
            (None, None) => {}
            (Some(_), None) | (None, Some(_)) => {
                return Err(CustomFamilyError::ConstraintViolation {
                    reason: format!(
                        "block {b} stacked_design and stacked_offset must be Some together \
                         or both None"
                    ),
                }
                .into());
            }
        }
        let p = spec.design.ncols();
        if let Some(beta0) = &spec.initial_beta
            && beta0.len() != p
        {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {b} initial_beta length mismatch: got {}, expected {p}",
                    beta0.len()
                ),
            }
            .into());
        }
        if spec.initial_log_lambdas.len() != spec.penalties.len() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {b} initial_log_lambdas length {} does not match penalties {}",
                    spec.initial_log_lambdas.len(),
                    spec.penalties.len()
                ),
            }
            .into());
        }
        for (k, s) in spec.penalties.iter().enumerate() {
            let (r, c) = s.shape();
            if r != p || c != p {
                return Err(CustomFamilyError::DimensionMismatch {
                    reason: format!("block {b} penalty {k} must be {p}x{p}, got {r}x{c}"),
                }
                .into());
            }
        }
        penalty_counts.push(spec.penalties.len());
    }
    Ok(penalty_counts)
}


fn with_block_geometry<F: CustomFamily + ?Sized, T>(
    family: &F,
    block_states: &[ParameterBlockState],
    spec: &ParameterBlockSpec,
    block_idx: usize,
    f: impl FnOnce(&DesignMatrix, &Array1<f64>) -> Result<T, String>,
) -> Result<T, String> {
    if family.block_geometry_is_dynamic() {
        let (x_dyn, off_dyn) = family.block_geometry(block_states, spec)?;
        let expected_rows = spec.solver_design().nrows();
        if x_dyn.nrows() != expected_rows {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {block_idx} dynamic design row mismatch: got {}, expected {}",
                    x_dyn.nrows(),
                    expected_rows
                ),
            }
            .into());
        }
        if x_dyn.ncols() != spec.design.ncols() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {block_idx} dynamic design col mismatch: got {}, expected {}",
                    x_dyn.ncols(),
                    spec.design.ncols()
                ),
            }
            .into());
        }
        if off_dyn.len() != expected_rows {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {block_idx} dynamic offset length mismatch: got {}, expected {}",
                    off_dyn.len(),
                    expected_rows
                ),
            }
            .into());
        }
        f(&x_dyn, &off_dyn)
    } else {
        f(spec.solver_design(), spec.solver_offset())
    }
}


fn flatten_log_lambdas(specs: &[ParameterBlockSpec]) -> Array1<f64> {
    let total = specs
        .iter()
        .map(|s| s.initial_log_lambdas.len())
        .sum::<usize>();
    let mut out = Array1::<f64>::zeros(total);
    let mut at = 0usize;
    for spec in specs {
        let len = spec.initial_log_lambdas.len();
        if len > 0 {
            out.slice_mut(ndarray::s![at..at + len])
                .assign(&spec.initial_log_lambdas);
        }
        at += len;
    }
    out
}


#[derive(Clone, Debug)]
struct PenaltyLabelLayout {
    penalty_counts: Vec<usize>,
    physical_to_outer: Vec<Option<usize>>,
    fixed_log_lambdas: Vec<Option<f64>>,
    initial_rho: Array1<f64>,
}


impl PenaltyLabelLayout {
    fn physical_count(&self) -> usize {
        self.physical_to_outer.len()
    }

    fn has_tied_coordinates(&self) -> bool {
        self.initial_rho.len() != self.physical_to_outer.len()
    }
}


fn penalty_label_layout(
    specs: &[ParameterBlockSpec],
    penalty_counts: Vec<usize>,
) -> Result<PenaltyLabelLayout, String> {
    let mut label_to_outer = BTreeMap::<String, usize>::new();
    let mut physical_to_outer = Vec::<Option<usize>>::new();
    let mut fixed_log_lambdas = Vec::<Option<f64>>::new();
    let mut initial = Vec::<f64>::new();

    for (block_idx, spec) in specs.iter().enumerate() {
        for penalty_idx in 0..spec.penalties.len() {
            if let Some(fixed) = spec.penalties[penalty_idx].fixed_log_lambda() {
                if !fixed.is_finite() {
                    return Err(CustomFamilyError::ConstraintViolation {
                        reason: format!(
                            "block {block_idx} penalty {penalty_idx} fixed log-precision is non-finite: {fixed}"
                        ),
                    }
                    .into());
                }
                physical_to_outer.push(None);
                fixed_log_lambdas.push(Some(fixed));
                continue;
            }
            let label = spec.penalties[penalty_idx]
                .precision_label()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("__block_{block_idx}_penalty_{penalty_idx}"));
            let rho0 = spec.initial_log_lambdas[penalty_idx];
            let outer = if let Some(&outer) = label_to_outer.get(&label) {
                let first = initial[outer];
                if first.is_finite() && rho0.is_finite() && (first - rho0).abs() > 1e-10 {
                    return Err(CustomFamilyError::ConstraintViolation { reason: format!(
                        "precision label '{label}' has inconsistent initial log-precisions: {first} and {rho0}"
                    ) }.into());
                }
                outer
            } else {
                let outer = initial.len();
                label_to_outer.insert(label, outer);
                initial.push(rho0);
                outer
            };
            physical_to_outer.push(Some(outer));
            fixed_log_lambdas.push(None);
        }
    }

    Ok(PenaltyLabelLayout {
        penalty_counts,
        physical_to_outer,
        fixed_log_lambdas,
        initial_rho: Array1::from_vec(initial),
    })
}


fn expand_labeled_log_lambdas(
    rho: &Array1<f64>,
    layout: &PenaltyLabelLayout,
) -> Result<Array1<f64>, String> {
    if rho.len() != layout.initial_rho.len() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "log-lambda label coordinate mismatch: got {}, expected {}",
                rho.len(),
                layout.initial_rho.len()
            ),
        }
        .into());
    }
    let mut expanded = Array1::<f64>::zeros(layout.physical_count());
    for (physical, outer) in layout.physical_to_outer.iter().enumerate() {
        expanded[physical] = match *outer {
            Some(outer) => rho[outer],
            None => layout.fixed_log_lambdas[physical].ok_or_else(|| {
                CustomFamilyError::ConstraintViolation {
                    reason: format!(
                        "fixed penalty layout missing value at physical slot {physical}"
                    ),
                }
                .to_string()
            })?,
        };
    }
    Ok(expanded)
}


fn split_labeled_log_lambdas(
    rho: &Array1<f64>,
    layout: &PenaltyLabelLayout,
) -> Result<Vec<Array1<f64>>, String> {
    let expanded = expand_labeled_log_lambdas(rho, layout)?;
    split_log_lambdas(&expanded, &layout.penalty_counts)
}


fn aggregate_labeled_gradient(
    gradient: &Array1<f64>,
    layout: &PenaltyLabelLayout,
) -> Result<Array1<f64>, String> {
    if gradient.len() != layout.physical_count() {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: format!(
                "physical gradient length mismatch: got {}, expected {}",
                gradient.len(),
                layout.physical_count()
            ),
        }
        .into());
    }
    let mut out = Array1::<f64>::zeros(layout.initial_rho.len());
    for (physical, outer) in layout.physical_to_outer.iter().enumerate() {
        if let Some(outer) = *outer {
            out[outer] += gradient[physical];
        }
    }
    Ok(out)
}

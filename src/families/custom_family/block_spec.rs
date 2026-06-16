//! Data model for the blockwise carrier: parameter-block specs, coefficient
//! groups/labels/priors, per-block working sets and states, the effective-Jacobian
//! and channel-Hessian abstractions, and blockspec validation.

use super::*;

use crate::types::CoefficientGroupPrior;

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

pub(crate) fn clamp_jacobian_rows(rows: Range<usize>, n: usize) -> Range<usize> {
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

pub(crate) fn custom_family_block_role(
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

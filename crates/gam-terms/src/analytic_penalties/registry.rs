use super::*;

// ---------------------------------------------------------------------------
// Operator-form wrapper for the REML/PIRLS canonical pipeline
// ---------------------------------------------------------------------------

/// Wraps any [`AnalyticPenalty`] so the existing PIRLS / REML consumers
/// (which expect a `value + gradient + (hvp | hessian-diag)` quintuple) can
/// query it uniformly. The wrapper is `Send + Sync` and `Arc`-shared so the
/// outer loop can hand it to multiple workers.
pub struct AnalyticPenaltyOp {
    pub penalty: Arc<dyn AnalyticPenalty>,
}

impl AnalyticPenaltyOp {
    #[must_use]
    pub fn new(penalty: Arc<dyn AnalyticPenalty>) -> Self {
        Self { penalty }
    }
}

// ---------------------------------------------------------------------------
// Registration helper — collects penalty kinds for the outer REML driver
// ---------------------------------------------------------------------------

/// Tagged sum of the analytic penalty kinds, with enough metadata for the outer
/// REML driver to:
///
///   1. Concatenate each penalty's owned ρ-axes onto the global ρ vector.
///   2. Route the inner gradient `∂L/∂target` contribution back into the
///      correct β or ext-coordinate slice.
///   3. Build a Hessian-block descriptor for `RemlState` cache-key invalidation.
macro_rules! define_analytic_penalty_kind {
    ($(register!($variant:ident, $ty:ty);)*) => {
        #[derive(Clone, Debug)]
        pub enum AnalyticPenaltyKind {
            $($variant(Arc<$ty>),)*
        }

        impl AnalyticPenaltyKind {
            pub fn apply_schedule(&mut self, iter: usize) {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => Arc::make_mut(p).apply_schedule(iter),)*
                }
            }

            pub fn tier(&self) -> PenaltyTier {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => p.dispatch_tier(),)*
                }
            }

            pub fn rho_count(&self) -> usize {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => p.rho_count(),)*
                }
            }

            pub fn name(&self) -> &str {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => p.name(),)*
                }
            }

            pub fn kind_tag(&self) -> &'static str {
                match self {
                    $(AnalyticPenaltyKind::$variant(_) => <$ty as PenaltyManifest>::KIND_TAG,)*
                }
            }

            pub fn python_wrapper_name(&self) -> &'static str {
                match self {
                    $(AnalyticPenaltyKind::$variant(_) => <$ty as PenaltyManifest>::PYTHON_WRAPPER,)*
                }
            }

            pub fn is_row_block_diagonal(&self) -> bool {
                match self {
                    $(AnalyticPenaltyKind::$variant(_) => <$ty as PenaltyManifest>::ROW_BLOCK_DIAGONAL,)*
                }
            }

            pub fn value(&self, target: ArrayView1<'_, f64>, rho: ArrayView1<'_, f64>) -> f64 {
                // UFCS forces dispatch through the AnalyticPenalty trait so a
                // wrapper type (e.g. SheafConsistencyPenalty) carrying both an
                // inherent `value(&self, s)` Python-API helper and the trait's
                // `value(&self, target, rho)` cannot silently bind the
                // inherent method here.
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => <$ty as AnalyticPenalty>::value(p, target, rho),)*
                }
            }

            pub fn grad_target(
                &self,
                target: ArrayView1<'_, f64>,
                rho: ArrayView1<'_, f64>,
            ) -> Array1<f64> {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => <$ty as AnalyticPenalty>::grad_target(p, target, rho),)*
                }
            }

            pub fn grad_rho(
                &self,
                target: ArrayView1<'_, f64>,
                rho: ArrayView1<'_, f64>,
            ) -> Array1<f64> {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => <$ty as AnalyticPenalty>::grad_rho(p, target, rho),)*
                }
            }

            pub fn hessian_diag(
                &self,
                target: ArrayView1<'_, f64>,
                rho: ArrayView1<'_, f64>,
            ) -> Option<Array1<f64>> {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => <$ty as AnalyticPenalty>::hessian_diag(p, target, rho),)*
                }
            }

            pub fn hvp(
                &self,
                target: ArrayView1<'_, f64>,
                rho: ArrayView1<'_, f64>,
                v: ArrayView1<'_, f64>,
            ) -> Array1<f64> {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => <$ty as AnalyticPenalty>::hvp(p, target, rho, v),)*
                }
            }

            pub fn psd_majorizer_diag(
                &self,
                target: ArrayView1<'_, f64>,
                rho: ArrayView1<'_, f64>,
            ) -> Option<Array1<f64>> {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => <$ty as AnalyticPenalty>::psd_majorizer_diag(p, target, rho),)*
                }
            }

            pub fn psd_majorizer_hvp(
                &self,
                target: ArrayView1<'_, f64>,
                rho: ArrayView1<'_, f64>,
                v: ArrayView1<'_, f64>,
            ) -> Array1<f64> {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => <$ty as AnalyticPenalty>::psd_majorizer_hvp(p, target, rho, v),)*
                }
            }
        }
    };
}

analytic_penalty_registry!(define_analytic_penalty_kind);

impl AnalyticPenaltyKind {
    pub(crate) fn isometry_scalar_weight(&self) -> Option<f64> {
        match self {
            AnalyticPenaltyKind::Isometry(p) => Some(p.scalar_weight),
            _ => None,
        }
    }

    pub(crate) fn set_isometry_scalar_weight(&mut self, weight: f64) {
        if let AnalyticPenaltyKind::Isometry(p) = self {
            Arc::make_mut(p).scalar_weight = weight;
        }
    }
}

/// Registry of analytic penalties active in a single fit. The owning
/// `RemlState` builder concatenates the per-penalty ρ-axes onto its global
/// ρ vector in the order they appear here, so the rho-index bookkeeping
/// inside each penalty is interpreted relative to its local slice.
#[derive(Clone, Default)]
pub struct AnalyticPenaltyRegistry {
    pub penalties: Vec<AnalyticPenaltyKind>,
}

impl std::fmt::Debug for AnalyticPenaltyRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnalyticPenaltyRegistry")
            .field("penalty_count", &self.penalties.len())
            .finish()
    }
}

impl AnalyticPenaltyRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, p: AnalyticPenaltyKind) {
        self.penalties.push(p);
    }

    pub fn total_rho_count(&self) -> usize {
        self.penalties.iter().map(|p| p.rho_count()).sum()
    }

    pub fn apply_weight_schedules(&mut self, iter: usize) {
        for penalty in &mut self.penalties {
            penalty.apply_schedule(iter);
        }
    }

    pub fn isometry_scalar_weights(&self) -> Vec<f64> {
        self.penalties
            .iter()
            .filter_map(AnalyticPenaltyKind::isometry_scalar_weight)
            .collect()
    }

    pub fn set_isometry_scalar_weights(&mut self, weights: &[f64]) {
        let mut idx = 0usize;
        for penalty in &mut self.penalties {
            if penalty.isometry_scalar_weight().is_some() {
                assert!(
                    idx < weights.len(),
                    "set_isometry_scalar_weights received fewer weights than registered isometry penalties"
                );
                penalty.set_isometry_scalar_weight(weights[idx]);
                idx += 1;
            }
        }
        assert_eq!(
            idx,
            weights.len(),
            "set_isometry_scalar_weights received extra weights"
        );
    }

    /// Returns `(local_rho_slice, target_tier, name)` for each registered
    /// penalty so the outer driver can wire its ρ-views.
    pub fn rho_layout(&self) -> Vec<(std::ops::Range<usize>, PenaltyTier, &str)> {
        let mut out = Vec::with_capacity(self.penalties.len());
        let mut offset = 0usize;
        for p in &self.penalties {
            let n = p.rho_count();
            out.push((offset..offset + n, p.tier(), p.name()));
            offset += n;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// PenaltyOp integration
// ---------------------------------------------------------------------------
//
// The canonical PIRLS / REML pipeline consumes square symmetric operators
// through the `PenaltyOp` trait (see `terms::analytic_penalties`). The non-quadratic
// analytic penalties here are *not* linear in their target, but the inner
// Newton step only sees their **Hessian at the current iterate**. We therefore
// expose each penalty as a `PenaltyOp` by
// freezing `(target, rho)` and routing `matvec` to `hvp`. The solver re-builds
// the frozen op once per outer iteration (after PIRLS converges on `β`), in
// exactly the same place the existing closed-form operator is rebuilt when
// the extension-coordinate block advances.

/// `PenaltyOp` view of an [`AnalyticPenalty`] frozen at `(target, rho)`.
///
/// `as_dense()` materializes the frozen local Hessian via `n` matvecs against
/// the standard basis — `O(n²)` and intended only for spectral diagnostics;
/// the hot path uses `matvec` and `diag` directly.
pub struct FrozenAnalyticPenaltyOp {
    penalty: AnalyticPenaltyKind,
    target: Array1<f64>,
    rho: Array1<f64>,
}

const ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD: usize = 1024;

const HUTCHINSON_DIAG_SAMPLES: usize = 32;

const ORTHOGONALITY_LOGDET_SLQ_PROBES: usize = 16;

const ORTHOGONALITY_LOGDET_LANCZOS_STEPS: usize = 32;

impl FrozenAnalyticPenaltyOp {
    #[must_use]
    pub fn new(penalty: AnalyticPenaltyKind, target: Array1<f64>, rho: Array1<f64>) -> Self {
        Self {
            penalty,
            target,
            rho,
        }
    }

    /// Underlying penalty (read-only). Useful for the outer driver that needs
    /// to query `grad_rho` while still holding the frozen op.
    pub fn penalty(&self) -> &AnalyticPenaltyKind {
        &self.penalty
    }
}

impl PenaltyOp for FrozenAnalyticPenaltyOp {
    fn dim(&self) -> usize {
        self.target.len()
    }

    fn matvec(&self, w: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        // `FrozenAnalyticPenaltyOp` is the PSD curvature operator routed into
        // the canonical PIRLS / preconditioner / log-det pipeline, so it must
        // expose the PSD majorizer, not the (possibly indefinite) exact
        // Hessian. For convex penalties the majorizer is the exact HVP.
        let h = self
            .penalty
            .psd_majorizer_hvp(self.target.view(), self.rho.view(), w);
        for i in 0..h.len() {
            out[i] = h[i];
        }
    }

    fn diag(&self) -> Array1<f64> {
        // Each diagonal penalty exposes `hessian_diag` directly (ARD,
        // smoothed-L¹, Log; Hoyer currently exposes its preconditioner
        // diagonal). Penalties whose exact diagonal is cheap or contractually
        // required use the analytic path even when the dense Hessian is large.
        match &self.penalty {
            AnalyticPenaltyKind::Ard(p) => p
                .psd_majorizer_diag(self.target.view(), self.rho.view())
                .expect("ARD diag"),
            AnalyticPenaltyKind::TopKActivation(p) => p
                .psd_majorizer_diag(self.target.view(), self.rho.view())
                .expect("TopK activation diag"),
            AnalyticPenaltyKind::JumpReLU(p) => p
                .psd_majorizer_diag(self.target.view(), self.rho.view())
                .expect("JumpReLU majorizer diag"),
            AnalyticPenaltyKind::TotalVariation(p) => {
                p.diag_target(self.target.view(), self.rho.view())
            }
            AnalyticPenaltyKind::BlockOrthogonality(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_diag_via_matvec()
            }
            AnalyticPenaltyKind::BlockOrthogonality(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::DecoderIncoherence(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_diag_via_matvec()
            }
            AnalyticPenaltyKind::DecoderIncoherence(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::Orthogonality(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::NuclearNorm(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::BlockSparsity(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_diag_via_matvec()
            }
            AnalyticPenaltyKind::BlockSparsity(p) => {
                p.diag_target(self.target.view(), self.rho.view())
            }
            AnalyticPenaltyKind::MechanismSparsity(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_diag_via_matvec()
            }
            AnalyticPenaltyKind::MechanismSparsity(p) => {
                p.diag_target(self.target.view(), self.rho.view())
            }
            AnalyticPenaltyKind::RowPrecisionPrior(p) => {
                p.diag_target(self.target.view(), self.rho.view())
            }
            AnalyticPenaltyKind::IvaeRidgeMeanGauge(p) => {
                p.diag_target(self.target.view(), self.rho.view())
            }
            AnalyticPenaltyKind::ParametricRowPrecisionPrior(p) => {
                p.diag_target(self.target.view(), self.rho.view())
            }
            AnalyticPenaltyKind::ScadMcp(p) => p.diag_target(self.target.view(), self.rho.view()),
            AnalyticPenaltyKind::IBPAssignment(p) => p
                .psd_majorizer_diag(self.target.view(), self.rho.view())
                .expect("IBP assignment diag"),
            AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::Sparsity(p) => {
                if let Some(d) = p.psd_majorizer_diag(self.target.view(), self.rho.view()) {
                    d
                } else {
                    self.diag_via_matvec()
                }
            }
            AnalyticPenaltyKind::Isometry(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_diag_via_matvec()
            }
            AnalyticPenaltyKind::Isometry(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::NestedPrefix(p) => p
                .psd_majorizer_diag(self.target.view(), self.rho.view())
                .expect("NestedPrefix diag"),
            AnalyticPenaltyKind::SheafConsistency(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_diag_via_matvec()
            }
            AnalyticPenaltyKind::SheafConsistency(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::Monotonicity(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_diag_via_matvec()
            }
            AnalyticPenaltyKind::Monotonicity(_) => self.diag_via_matvec(),
        }
    }

    fn log_det_plus_lambda_i(&self, lambda: f64) -> Result<f64, String> {
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(format!(
                "FrozenAnalyticPenaltyOp::log_det_plus_lambda_i requires finite λ > 0; got {lambda}"
            ));
        }
        // For the diagonal-Hessian penalties (ARD, smoothed-L¹ and Log) the
        // closed form is `Σ_i log(d_i + λ)`. Forward-difference TV uses the
        // tridiagonal path-graph structure. Graph TV, NuclearNorm,
        // BlockSparsity, BlockOrthogonality, and IvaeRidgeMeanGauge keep the
        // exact dense eigensolve only below the small-block threshold; large
        // blocks use SLQ against the analytic HVP.
        // Orthogonality is excluded because its exact Hessian is indefinite.
        match &self.penalty {
            AnalyticPenaltyKind::Ard(_)
            | AnalyticPenaltyKind::TopKActivation(_)
            | AnalyticPenaltyKind::JumpReLU(_)
            | AnalyticPenaltyKind::Sparsity(_)
            | AnalyticPenaltyKind::IBPAssignment(_)
            | AnalyticPenaltyKind::NestedPrefix(_) => {
                let d = self.diag();
                let mut s = 0.0;
                for &v in d.iter() {
                    let r = v + lambda;
                    if !r.is_finite() || r <= 0.0 {
                        return Err(format!(
                            "FrozenAnalyticPenaltyOp::log_det_plus_lambda_i: \
                             non-positive entry {r:.3e} after λ shift"
                        ));
                    }
                    s += r.ln();
                }
                Ok(s)
            }
            AnalyticPenaltyKind::TotalVariation(p) => match &p.difference_op {
                DifferenceOpKind::ForwardDiff1D => {
                    p.log_det_plus_lambda_i_forward_1d(self.target.view(), self.rho.view(), lambda)
                }
                DifferenceOpKind::GraphEdges(_)
                    if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
                {
                    self.stochastic_log_det_plus_lambda_i(lambda)
                }
                DifferenceOpKind::GraphEdges(_) => {
                    let dense = p.as_dense(self.target.view(), self.rho.view());
                    <Array2<f64> as PenaltyOp>::log_det_plus_lambda_i(&dense, lambda)
                }
            },
            AnalyticPenaltyKind::Orthogonality(_) => Err(
                "FrozenAnalyticPenaltyOp::log_det_plus_lambda_i cannot treat \
                 OrthogonalityPenalty as PSD; its exact Hessian is indefinite"
                    .to_string(),
            ),
            AnalyticPenaltyKind::NuclearNorm(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_log_det_plus_lambda_i(lambda)
            }
            AnalyticPenaltyKind::IvaeRidgeMeanGauge(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_log_det_plus_lambda_i(lambda)
            }
            AnalyticPenaltyKind::RowPrecisionPrior(p) => {
                p.log_det_plus_lambda_i(self.rho.view(), lambda)
            }
            AnalyticPenaltyKind::ParametricRowPrecisionPrior(p) => {
                p.log_det_plus_lambda_i(self.rho.view(), lambda)
            }
            AnalyticPenaltyKind::ScadMcp(p) => {
                p.log_det_plus_lambda_i(self.target.view(), self.rho.view(), lambda)
            }
            AnalyticPenaltyKind::BlockSparsity(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_log_det_plus_lambda_i(lambda)
            }
            AnalyticPenaltyKind::MechanismSparsity(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_log_det_plus_lambda_i(lambda)
            }
            AnalyticPenaltyKind::BlockOrthogonality(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_log_det_plus_lambda_i(lambda)
            }
            AnalyticPenaltyKind::DecoderIncoherence(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_log_det_plus_lambda_i(lambda)
            }
            AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_log_det_plus_lambda_i(lambda)
            }
            AnalyticPenaltyKind::Isometry(_) => {
                let dense = self.as_dense();
                <Array2<f64> as PenaltyOp>::log_det_plus_lambda_i(&dense, lambda)
            }
            AnalyticPenaltyKind::SheafConsistency(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_log_det_plus_lambda_i(lambda)
            }
            AnalyticPenaltyKind::Monotonicity(_)
                if self.dim() > ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD =>
            {
                self.stochastic_log_det_plus_lambda_i(lambda)
            }
            AnalyticPenaltyKind::NuclearNorm(_)
            | AnalyticPenaltyKind::BlockSparsity(_)
            | AnalyticPenaltyKind::MechanismSparsity(_)
            | AnalyticPenaltyKind::IvaeRidgeMeanGauge(_)
            | AnalyticPenaltyKind::BlockOrthogonality(_)
            | AnalyticPenaltyKind::DecoderIncoherence(_)
            | AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_)
            | AnalyticPenaltyKind::SheafConsistency(_)
            | AnalyticPenaltyKind::Monotonicity(_) => {
                let dense = self.as_dense();
                <Array2<f64> as PenaltyOp>::log_det_plus_lambda_i(&dense, lambda)
            }
        }
    }

    fn as_dense(&self) -> Array2<f64> {
        match &self.penalty {
            AnalyticPenaltyKind::TotalVariation(p) => {
                return p.as_dense(self.target.view(), self.rho.view());
            }
            AnalyticPenaltyKind::BlockSparsity(p) => {
                return p.as_dense(self.target.view(), self.rho.view());
            }
            AnalyticPenaltyKind::MechanismSparsity(p) => {
                return p.as_dense(self.target.view(), self.rho.view());
            }
            AnalyticPenaltyKind::BlockOrthogonality(p) => {
                return p.as_dense(self.target.view(), self.rho.view());
            }
            AnalyticPenaltyKind::RowPrecisionPrior(p) => {
                return p.as_dense(self.target.view(), self.rho.view());
            }
            AnalyticPenaltyKind::IvaeRidgeMeanGauge(p) => {
                return p.as_dense(self.target.view(), self.rho.view());
            }
            AnalyticPenaltyKind::ParametricRowPrecisionPrior(p) => {
                return p.as_dense(self.target.view(), self.rho.view());
            }
            AnalyticPenaltyKind::Orthogonality(p) => {
                let n = self.target.len();
                let Some(t) = p.target_matrix(self.target.view()) else {
                    return Array2::<f64>::zeros((n, n));
                };
                let gram = OrthogonalityPenalty::gram_minus_identity(t.view());
                return p.as_dense_with_precomputed_m(
                    t.view(),
                    gram.view(),
                    p.scale(self.rho.view()),
                );
            }
            AnalyticPenaltyKind::Isometry(p) => {
                let n = self.target.len();
                let Some(state) = p.hvp_state(self.target.view()) else {
                    return Array2::<f64>::zeros((n, n));
                };
                let mut dense = Array2::<f64>::zeros((n, n));
                let mut e = Array1::<f64>::zeros(n);
                for j in 0..n {
                    e[j] = 1.0;
                    let col = p.hvp_with_precomputed_state(&state, self.rho.view(), e.view());
                    for i in 0..n {
                        dense[[i, j]] = col[i];
                    }
                    e[j] = 0.0;
                }
                return dense;
            }
            _ => {}
        }
        let n = self.target.len();
        let mut m = Array2::<f64>::zeros((n, n));
        let mut e = Array1::<f64>::zeros(n);
        for j in 0..n {
            e[j] = 1.0;
            // `FrozenAnalyticPenaltyOp` is the PSD Newton / PIRLS / preconditioner
            // curvature operator (its `matvec` uses `psd_majorizer_hvp`), so the
            // dense-materialization fallback probes the PSD majorizer too — never
            // the (possibly indefinite) exact Hessian. For convex penalties the
            // majorizer equals the exact HVP, so this is exact for them.
            let col = self
                .penalty
                .psd_majorizer_hvp(self.target.view(), self.rho.view(), e.view());
            for i in 0..n {
                m[[i, j]] = col[i];
            }
            e[j] = 0.0;
        }
        m
    }
}

impl FrozenAnalyticPenaltyOp {
    fn diag_via_matvec(&self) -> Array1<f64> {
        match &self.penalty {
            AnalyticPenaltyKind::Orthogonality(p) => {
                let n = self.target.len();
                let Some(t) = p.target_matrix(self.target.view()) else {
                    return Array1::<f64>::zeros(n);
                };
                let latent_dim = t.ncols();
                let gram = OrthogonalityPenalty::gram_minus_identity(t.view());
                let scale = p.scale(self.rho.view());
                let factor = 2.0 * scale;
                let mut diag = Array1::<f64>::zeros(n);
                for row in 0..t.nrows() {
                    let mut row_norm_sq = 0.0;
                    for col in 0..latent_dim {
                        row_norm_sq += t[[row, col]] * t[[row, col]];
                    }
                    for col in 0..latent_dim {
                        let i = row * latent_dim + col;
                        diag[i] = factor
                            * (gram[[col, col]] + t[[row, col]] * t[[row, col]] + row_norm_sq);
                    }
                }
                return diag;
            }
            AnalyticPenaltyKind::Isometry(p) => {
                let n = self.target.len();
                let Some(state) = p.hvp_state(self.target.view()) else {
                    return Array1::<f64>::zeros(n);
                };
                let mut d = Array1::<f64>::zeros(n);
                let mut e = Array1::<f64>::zeros(n);
                for i in 0..n {
                    e[i] = 1.0;
                    let h = p.hvp_with_precomputed_state(&state, self.rho.view(), e.view());
                    d[i] = h[i];
                    e[i] = 0.0;
                }
                return d;
            }
            _ => {}
        }
        let n = self.target.len();
        let mut d = Array1::<f64>::zeros(n);
        let mut e = Array1::<f64>::zeros(n);
        for i in 0..n {
            e[i] = 1.0;
            // PSD curvature operator: probe the PSD majorizer (exact for convex
            // penalties), mirroring `matvec` and the dense fallback above.
            let h = self
                .penalty
                .psd_majorizer_hvp(self.target.view(), self.rho.view(), e.view());
            d[i] = h[i];
            e[i] = 0.0;
        }
        d
    }

    fn stochastic_diag_via_matvec(&self) -> Array1<f64> {
        match &self.penalty {
            AnalyticPenaltyKind::Orthogonality(p) => {
                let n = self.target.len();
                let Some(t) = p.target_matrix(self.target.view()) else {
                    return Array1::<f64>::zeros(n);
                };
                let gram = OrthogonalityPenalty::gram_minus_identity(t.view());
                let scale = p.scale(self.rho.view());
                let samples = HUTCHINSON_DIAG_SAMPLES.max(1);
                let mut diag = Array1::<f64>::zeros(n);
                let mut z = Array1::<f64>::zeros(n);
                for probe in 0..samples {
                    rademacher_unit_probe_into(z.view_mut(), probe as u64, 1.0);
                    let Some(z_mat) = p.target_matrix(z.view()) else {
                        return diag;
                    };
                    let hz = p.hvp_with_precomputed_m(t.view(), gram.view(), z_mat, scale);
                    for i in 0..n {
                        diag[i] += z[i] * hz[[i / t.ncols(), i % t.ncols()]];
                    }
                }
                let inv_samples = 1.0 / samples as f64;
                for i in 0..n {
                    diag[i] *= inv_samples;
                }
                return diag;
            }
            AnalyticPenaltyKind::Isometry(p) => {
                let n = self.target.len();
                let Some(state) = p.hvp_state(self.target.view()) else {
                    return Array1::<f64>::zeros(n);
                };
                let samples = HUTCHINSON_DIAG_SAMPLES.max(1);
                let mut diag = Array1::<f64>::zeros(n);
                let mut z = Array1::<f64>::zeros(n);
                for probe in 0..samples {
                    rademacher_unit_probe_into(z.view_mut(), probe as u64, 1.0);
                    let hz = p.hvp_with_precomputed_state(&state, self.rho.view(), z.view());
                    for i in 0..n {
                        diag[i] += z[i] * hz[i];
                    }
                }
                let inv_samples = 1.0 / samples as f64;
                for i in 0..n {
                    diag[i] *= inv_samples;
                }
                return diag;
            }
            _ => {}
        }
        let n = self.target.len();
        let samples = HUTCHINSON_DIAG_SAMPLES.max(1);
        let mut diag = Array1::<f64>::zeros(n);
        let mut z = Array1::<f64>::zeros(n);
        let mut hz = Array1::<f64>::zeros(n);
        // Hutchinson-Hadamard diagonal estimator (Bekas et al., 2007):
        // Var[(z ⊙ Hz)_i] = Σ_{j≠i} H_ij², so averaging m probes leaves
        // variance equal to the off-diagonal row mass divided by m.
        // With m=32, diagonally dominant Frobenius/TV Hessians have ~16% relative SD.
        for probe in 0..samples {
            rademacher_unit_probe_into(z.view_mut(), probe as u64, 1.0);
            self.matvec(z.view(), hz.view_mut());
            for i in 0..n {
                diag[i] += z[i] * hz[i];
            }
        }
        let inv_samples = 1.0 / samples as f64;
        for i in 0..n {
            diag[i] *= inv_samples;
        }
        diag
    }

    fn stochastic_log_det_plus_lambda_i(&self, lambda: f64) -> Result<f64, String> {
        let n = self.dim();
        if n == 0 {
            return Ok(0.0);
        }
        let probes = ORTHOGONALITY_LOGDET_SLQ_PROBES.max(1);
        let steps = ORTHOGONALITY_LOGDET_LANCZOS_STEPS.min(n).max(1);
        let inv_norm = 1.0 / (n as f64).sqrt();
        let mut estimate = 0.0;
        for probe in 0..probes {
            let mut q0 = Array1::<f64>::zeros(n);
            rademacher_unit_probe_into(q0.view_mut(), probe as u64, inv_norm);
            let quad = self.lanczos_log_quadrature(lambda, q0, steps)?;
            estimate += n as f64 * quad;
        }
        Ok(estimate / probes as f64)
    }

    fn lanczos_log_quadrature(
        &self,
        lambda: f64,
        q: Array1<f64>,
        max_steps: usize,
    ) -> Result<f64, String> {
        let n = self.dim();
        let eigen = symmetric_lanczos_eigenpairs(
            n,
            q.as_slice().ok_or_else(|| {
                "FrozenAnalyticPenaltyOp::log_det_plus_lambda_i SLQ start vector is not contiguous"
                    .to_string()
            })?,
            SymmetricLanczosOptions {
                max_steps,
                residual_tol: 1e-12,
                local_reorthogonalize: false,
                full_reorthogonalize: false,
            },
            |q, out| {
                self.matvec(ArrayView1::from(q), ArrayViewMut1::from(&mut *out));
                for i in 0..n {
                    out[i] += lambda * q[i];
                }
                Ok(())
            },
        )
        .map_err(|e| {
            format!("FrozenAnalyticPenaltyOp::log_det_plus_lambda_i SLQ Lanczos failed: {e}")
        })?;
        symmetric_lanczos_log_quadrature(
            &eigen,
            "FrozenAnalyticPenaltyOp::log_det_plus_lambda_i expected SPD S+λI",
        )
    }
}

fn rademacher_unit_probe_into(mut z: ArrayViewMut1<'_, f64>, probe: u64, scale: f64) {
    let mut state = 0x6A09E667F3BCC909_u64 ^ probe.wrapping_mul(0xD1B54A32D192ED03);
    let mut bits = 0_u64;
    let mut remaining_bits = 0_u32;
    for i in 0..z.len() {
        if remaining_bits == 0 {
            bits = splitmix64(&mut state);
            remaining_bits = 64;
        }
        z[i] = if bits & 1 == 0 { scale } else { -scale };
        bits >>= 1;
        remaining_bits -= 1;
    }
}

#[inline]
const fn splitmix64(state: &mut u64) -> u64 {
    gam_linalg::utils::splitmix64(state)
}

impl AnalyticPenaltyKind {
    /// Freeze this kind at `(target, rho)` and return an `Arc<dyn PenaltyOp>`
    /// ready to slot into `BlockwisePenalty::with_op` or `PenaltyForm::Operator`.
    #[must_use]
    pub fn freeze(&self, target: Array1<f64>, rho: Array1<f64>) -> Arc<dyn PenaltyOp> {
        Arc::new(FrozenAnalyticPenaltyOp::new(self.clone(), target, rho))
    }
}

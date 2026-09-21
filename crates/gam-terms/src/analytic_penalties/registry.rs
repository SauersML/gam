use super::*;
use ndarray::s;

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

            pub fn validate_rho(&self, rho: ArrayView1<'_, f64>) -> Result<(), String> {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => <$ty as AnalyticPenalty>::validate_rho(p, rho),)*
                }
            }

            pub fn rho_coordinate_domains(&self) -> Result<Vec<(f64, f64)>, String> {
                match self {
                    $(AnalyticPenaltyKind::$variant(p) => <$ty as AnalyticPenalty>::rho_coordinate_domains(p),)*
                }
            }

            pub fn kind_tag(&self) -> &'static str {
                match self {
                    $(AnalyticPenaltyKind::$variant(_) => <$ty as PenaltyManifest>::KIND_TAG,)*
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

// The single source of truth for the penalty registry list is the
// `analytic_penalty_registry!` macro re-used (via the `manifest` module's
// `#[path]` include) from `src/terms/analytic_penalties/manifest.rs`, which the
// root `gam` crate also consumes. Defining a second `#[macro_export]` copy here
// both collided in the crate-root macro namespace (E0428) and silently dropped
// four penalty kinds (Monotonicity, NestedPrefix, DecoderIncoherence,
// SheafConsistency) that the `AnalyticPenaltyKind` consumers below still match
// on — expand the canonical list so every registered variant is generated.
crate::analytic_penalty_registry!(define_analytic_penalty_kind);

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

    /// Check every isometry penalty holds the decoder jets an evaluation of
    /// `order` reads for a `target_len`-coordinate target and has a defined
    /// scale-invariant gauge, returning the first
    /// refusal ([`IsometryPenalty::evaluation_state_precondition`]). No other
    /// registered penalty reads state its owner installs, so the others pass.
    pub fn isometry_evaluation_precondition(
        &self,
        order: IsometryEvaluationOrder,
        target_len: usize,
    ) -> Result<(), String> {
        for penalty in &self.penalties {
            if let AnalyticPenaltyKind::Isometry(isometry) = penalty {
                isometry.evaluation_state_precondition(order, target_len)?;
            }
        }
        Ok(())
    }

    /// Gradient `Σ_p ∂P_p/∂target_t` of every ψ-tier penalty, with `rho` the global
    /// vector: the derivative of the ψ-tier registry energy a latent score adds.
    ///
    /// #2933 F02 — a β-tier penalty is priced on coefficients, so its `target_t`
    /// derivative runs through the fitted decoder, which this sum does not carry; it is
    /// refused. An isometry penalty must hold the jets its gradient reads
    /// ([`Self::isometry_evaluation_precondition`]).
    pub fn target_grad(
        &self,
        target_t: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        self.validate_rho(rho)?;
        self.isometry_evaluation_precondition(IsometryEvaluationOrder::Gradient, target_t.len())?;
        let mut out = Array1::<f64>::zeros(target_t.len());
        for (penalty, (rho_slice, tier, name)) in self.penalties.iter().zip(self.rho_layout()) {
            match tier {
                PenaltyTier::Rho => continue,
                PenaltyTier::Beta => {
                    return Err(format!(
                        "analytic penalty `{name}` is β-tier: its target gradient runs through the \
                         fitted coefficients, which a ψ-tier gradient does not carry"
                    ));
                }
                PenaltyTier::Psi => {}
            }
            out += &penalty.grad_target(target_t, rho.slice(s![rho_slice]));
        }
        Ok(out)
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

    pub fn validate_rho(&self, rho: ArrayView1<'_, f64>) -> Result<(), String> {
        if rho.len() != self.total_rho_count() {
            return Err(format!(
                "analytic-penalty rho length {} != registry dimension {}",
                rho.len(),
                self.total_rho_count()
            ));
        }
        // Hot evaluation seam: walk the concatenated vector directly rather
        // than allocating the diagnostic `rho_layout()` Vec on every value /
        // gradient / Hessian call.
        let mut offset = 0usize;
        for penalty in &self.penalties {
            let end = offset + penalty.rho_count();
            penalty
                .validate_rho(rho.slice(s![offset..end]))
                .map_err(|error| format!("analytic penalty `{}`: {error}", penalty.name()))?;
            offset = end;
        }
        Ok(())
    }

    pub fn rho_domain_bounds(&self) -> Result<(Array1<f64>, Array1<f64>), String> {
        let mut lower = Array1::<f64>::zeros(self.total_rho_count());
        let mut upper = Array1::<f64>::zeros(self.total_rho_count());
        let mut offset = 0usize;
        for penalty in &self.penalties {
            let domains = penalty.rho_coordinate_domains()?;
            if domains.len() != penalty.rho_count() {
                return Err(format!(
                    "analytic penalty `{}` returned {} rho domains for {} coordinates",
                    penalty.name(),
                    domains.len(),
                    penalty.rho_count()
                ));
            }
            for (local, &(lo, hi)) in domains.iter().enumerate() {
                // Infinite faces deliberately represent ordinary unbounded
                // real coordinates (for example parametric raw-beta and mu).
                // The optimizer intersects these with its finite configured
                // box; `validate_rho` separately refuses non-finite values.
                if lo.is_nan() || hi.is_nan() || lo >= hi {
                    return Err(format!(
                        "analytic penalty `{}` has invalid rho domain[{local}] [{lo}, {hi}]",
                        penalty.name()
                    ));
                }
                lower[offset + local] = lo;
                upper[offset + local] = hi;
            }
            offset += domains.len();
        }
        Ok((lower, upper))
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
pub(crate) struct FrozenAnalyticPenaltyOp {
    penalty: AnalyticPenaltyKind,
    target: Array1<f64>,
    rho: Array1<f64>,
}

impl FrozenAnalyticPenaltyOp {
    #[must_use = "invalid analytic-penalty rho must be handled"]
    pub fn new(
        penalty: AnalyticPenaltyKind,
        target: Array1<f64>,
        rho: Array1<f64>,
    ) -> Result<Self, String> {
        penalty.validate_rho(rho.view())?;
        // Every read of the frozen curvature (matvec, diag, dense) probes the PSD
        // majorizer, which for an isometry penalty reads the decoder Jacobian and
        // its motion and needs a defined scale-invariant gauge.
        if let AnalyticPenaltyKind::Isometry(isometry) = &penalty {
            isometry
                .evaluation_state_precondition(IsometryEvaluationOrder::Gradient, target.len())?;
        }
        Ok(Self {
            penalty,
            target,
            rho,
        })
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
        // Each diagonal penalty exposes its PSD-majorizer diagonal directly
        // (ARD, smoothed-L¹, Log); Hoyer's rank-2 majorizer is dense, so its
        // diagonal comes from unit probes. Every other penalty returns its
        // exact diagonal at every dimension: the closed form where the penalty
        // has one, otherwise unit probes of the PSD majorizer. A Hutchinson
        // estimate used to stand in above dimension 1024, so the same operator
        // answered with an estimate on one side of a size window and the exact
        // value on the other (#2900).
        match &self.penalty {
            AnalyticPenaltyKind::Ard(p) => p
                .psd_majorizer_diag(self.target.view(), self.rho.view())
                .expect("ARD diag"),
            AnalyticPenaltyKind::TopKActivation(p) => p
                .psd_majorizer_diag(self.target.view(), self.rho.view())
                .expect("TopK activation diag"),
            AnalyticPenaltyKind::SmoothThreshold(p) => p
                .psd_majorizer_diag(self.target.view(), self.rho.view())
                .expect("SmoothThreshold majorizer diag"),
            AnalyticPenaltyKind::TotalVariation(p) => {
                p.diag_target(self.target.view(), self.rho.view())
            }
            AnalyticPenaltyKind::HarmonicRoughness(p) => p
                .psd_majorizer_diag(self.target.view(), self.rho.view())
                .expect("HarmonicRoughness diag"),
            AnalyticPenaltyKind::BlockOrthogonality(p) => {
                // `B = I_n ⊗ M`: the diagonal is `diag(M)` tiled over the rows.
                let m = p
                    .psd_majorizer_row_block(self.target.view(), self.rho.view())
                    .expect("BlockOrthogonality PSD row-block majorizer");
                let d = m.nrows();
                Array1::from_shape_fn(self.target.len(), |i| m[[i % d, i % d]])
            }
            AnalyticPenaltyKind::DecoderIncoherence(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::Orthogonality(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::NuclearNorm(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::BlockSparsity(p) => {
                p.diag_target(self.target.view(), self.rho.view())
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
            AnalyticPenaltyKind::ScadMcp(p) => {
                p.psd_majorizer_target(self.target.view(), self.rho.view())
            }
            AnalyticPenaltyKind::OrderedBetaBernoulli(p) => p
                .psd_majorizer_diag(self.target.view(), self.rho.view())
                .expect("ordered Beta--Bernoulli assignment diag"),
            AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::Sparsity(p) => {
                if let Some(d) = p.psd_majorizer_diag(self.target.view(), self.rho.view()) {
                    d
                } else {
                    self.diag_via_matvec()
                }
            }
            AnalyticPenaltyKind::Isometry(_) => self.diag_via_matvec(),
            AnalyticPenaltyKind::NestedPrefix(p) => p
                .psd_majorizer_diag(self.target.view(), self.rho.view())
                .expect("NestedPrefix diag"),
            AnalyticPenaltyKind::SheafConsistency(_) => self.diag_via_matvec(),
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
        // closed form is `Σ_i log(d_i + λ)`. Hoyer's PSD majorizer is dense
        // (rank 2), so it takes the dense eigensolve instead of the product of
        // its diagonal, which Hadamard's inequality makes an upper bound
        // rather than the log-determinant. Forward-difference TV uses the
        // tridiagonal path-graph structure. Every other PSD penalty takes the
        // exact dense eigensolve at every dimension, admitted on the memory
        // governor's ledger. A 16-probe SLQ estimate used to replace it above
        // dimension 1024 (#2900). Every arm reads the same PSD majorizer that
        // `matvec`, `diag` and `as_dense` expose, never the exact Hessian.
        match &self.penalty {
            AnalyticPenaltyKind::Sparsity(p) if matches!(p.kind, SparsityKind::Hoyer) => {
                self.governed_dense_log_det_plus_lambda_i(lambda, || self.as_dense())
            }
            AnalyticPenaltyKind::Ard(_)
            | AnalyticPenaltyKind::TopKActivation(_)
            | AnalyticPenaltyKind::SmoothThreshold(_)
            | AnalyticPenaltyKind::Sparsity(_)
            | AnalyticPenaltyKind::OrderedBetaBernoulli(_)
            | AnalyticPenaltyKind::HarmonicRoughness(_)
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
                DifferenceOpKind::GraphEdges(_) => self.governed_dense_log_det_plus_lambda_i(
                    lambda,
                    || p.as_dense(self.target.view(), self.rho.view()),
                ),
            },
            AnalyticPenaltyKind::BlockOrthogonality(p) => {
                // `B = I_n ⊗ M`, so `log det(B + λI) = n · log det(M + λI)`: a
                // `d × d` eigensolve instead of an `nd × nd` one.
                let m = p.psd_majorizer_row_block(self.target.view(), self.rho.view())?;
                let rows = self.target.len() / m.nrows();
                let block = <Array2<f64> as PenaltyOp>::log_det_plus_lambda_i(&m, lambda)?;
                Ok(rows as f64 * block)
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
            AnalyticPenaltyKind::NuclearNorm(_)
            | AnalyticPenaltyKind::BlockSparsity(_)
            | AnalyticPenaltyKind::MechanismSparsity(_)
            | AnalyticPenaltyKind::IvaeRidgeMeanGauge(_)
            | AnalyticPenaltyKind::Orthogonality(_)
            | AnalyticPenaltyKind::DecoderIncoherence(_)
            | AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_)
            | AnalyticPenaltyKind::Isometry(_)
            | AnalyticPenaltyKind::SheafConsistency(_)
            | AnalyticPenaltyKind::Monotonicity(_) => {
                self.governed_dense_log_det_plus_lambda_i(lambda, || self.as_dense())
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
                // `B = I_n ⊗ M`, the row-block PSD majorizer.
                let m = p
                    .psd_majorizer_row_block(self.target.view(), self.rho.view())
                    .expect("BlockOrthogonality PSD row-block majorizer");
                let d = m.nrows();
                let n = self.target.len();
                let mut dense = Array2::<f64>::zeros((n, n));
                for row in 0..n / d {
                    dense
                        .slice_mut(s![row * d..(row + 1) * d, row * d..(row + 1) * d])
                        .assign(&m);
                }
                return dense;
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
                let envelope = OrthogonalityPenalty::psd_majorizer_gram(t.view())
                    .expect("Orthogonality PSD Gram envelope");
                return p.as_dense_with_precomputed_m(
                    t.view(),
                    envelope.view(),
                    p.scale(self.rho.view()),
                );
            }
            // No closed-form dense materialization: fall through to the
            // column-by-column PSD-majorizer probe below. Enumerated rather
            // than wildcarded so a newly registered penalty has to state
            // which side of this split it is on. Isometry is here because its
            // cached HVP state builds the exact, indefinite Hessian, while this
            // operator is its Gauss-Newton majorizer.
            AnalyticPenaltyKind::Isometry(_)
            | AnalyticPenaltyKind::Sparsity(_)
            | AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_)
            | AnalyticPenaltyKind::OrderedBetaBernoulli(_)
            | AnalyticPenaltyKind::Ard(_)
            | AnalyticPenaltyKind::TopKActivation(_)
            | AnalyticPenaltyKind::SmoothThreshold(_)
            | AnalyticPenaltyKind::HarmonicRoughness(_)
            | AnalyticPenaltyKind::NuclearNorm(_)
            | AnalyticPenaltyKind::Monotonicity(_)
            | AnalyticPenaltyKind::NestedPrefix(_)
            | AnalyticPenaltyKind::ScadMcp(_)
            | AnalyticPenaltyKind::DecoderIncoherence(_)
            | AnalyticPenaltyKind::Isometry(_)
            | AnalyticPenaltyKind::SheafConsistency(_) => {}
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
                // Diagonal of the PSD majorizer: `G = TᵀT − I` replaced by its
                // certified PSD envelope, as in `psd_majorizer_hvp`.
                let gram = OrthogonalityPenalty::psd_majorizer_gram(t.view())
                    .expect("Orthogonality PSD Gram envelope");
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
            // No closed-form majorizer diagonal: fall through to the generic
            // unit-probe loop below. Enumerated rather than wildcarded so a
            // newly registered penalty has to state which side it is on.
            // Isometry's cached HVP state is the exact, indefinite Hessian, not
            // the Gauss-Newton majorizer that `matvec` applies.
            AnalyticPenaltyKind::Isometry(_)
            | AnalyticPenaltyKind::Sparsity(_)
            | AnalyticPenaltyKind::SoftmaxAssignmentSparsity(_)
            | AnalyticPenaltyKind::OrderedBetaBernoulli(_)
            | AnalyticPenaltyKind::Ard(_)
            | AnalyticPenaltyKind::TopKActivation(_)
            | AnalyticPenaltyKind::SmoothThreshold(_)
            | AnalyticPenaltyKind::TotalVariation(_)
            | AnalyticPenaltyKind::HarmonicRoughness(_)
            | AnalyticPenaltyKind::NuclearNorm(_)
            | AnalyticPenaltyKind::BlockSparsity(_)
            | AnalyticPenaltyKind::MechanismSparsity(_)
            | AnalyticPenaltyKind::Monotonicity(_)
            | AnalyticPenaltyKind::NestedPrefix(_)
            | AnalyticPenaltyKind::RowPrecisionPrior(_)
            | AnalyticPenaltyKind::IvaeRidgeMeanGauge(_)
            | AnalyticPenaltyKind::ParametricRowPrecisionPrior(_)
            | AnalyticPenaltyKind::ScadMcp(_)
            | AnalyticPenaltyKind::BlockOrthogonality(_)
            | AnalyticPenaltyKind::DecoderIncoherence(_)
            | AnalyticPenaltyKind::Isometry(_)
            | AnalyticPenaltyKind::SheafConsistency(_) => {}
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

    /// `log det(S + λI)` of the dense form `dense` builds, admitted on the memory
    /// governor's ledger before it is built. The ledger is charged three `n × n`
    /// blocks: the dense form, the regularized copy and the eigenvectors of the
    /// eigensolve (faer's tridiagonalization workspace is not counted). A refusal is
    /// an error, never a stochastic estimate.
    fn governed_dense_log_det_plus_lambda_i(
        &self,
        lambda: f64,
        dense: impl FnOnce() -> Array2<f64>,
    ) -> Result<f64, String> {
        let n = self.dim();
        let reservation = gam_runtime::resource::MemoryGovernor::global()
            .try_reserve_dense_f64_copies(
                n,
                n,
                3,
                "FrozenAnalyticPenaltyOp::log_det_plus_lambda_i dense penalty form",
            )
            .map_err(|error| {
                format!(
                    "FrozenAnalyticPenaltyOp::log_det_plus_lambda_i: refusing a {n}x{n} dense \
                     penalty form: {error}"
                )
            })?;
        let log_det = <Array2<f64> as PenaltyOp>::log_det_plus_lambda_i(&dense(), lambda);
        drop(reservation);
        log_det
    }
}

impl AnalyticPenaltyKind {
    /// Freeze this kind at `(target, rho)` and return an `Arc<dyn PenaltyOp>`
    /// ready to slot into `BlockwisePenalty::with_op` or `PenaltyForm::Operator`.
    #[must_use = "invalid analytic-penalty rho must be handled"]
    pub fn freeze(
        &self,
        target: Array1<f64>,
        rho: Array1<f64>,
    ) -> Result<Arc<dyn PenaltyOp>, String> {
        Ok(Arc::new(FrozenAnalyticPenaltyOp::new(
            self.clone(),
            target,
            rho,
        )?))
    }
}

use super::inner_strategy::GeometryBackendKind;
use super::penalty_logdet::PenaltyPseudologdet;
use super::*;
use crate::model_types::SmoothingCorrectionMethod;
use std::sync::atomic::Ordering;

/// Structured outcome of [`RemlState::compute_smoothing_correction_outcome`].
///
/// `FirstOrder` carries the analytic correction `J V_ρ Jᵀ` as its square-root
/// factor `B`, `J V_ρ Jᵀ = B Bᵀ` (absent only when there is nothing to
/// correct: no smoothing parameters). A route that publishes a dense
/// covariance forms the matrix from it; the factorized route keeps `B`
/// (#3283). `Unavailable` is the one branch where the exact first-order
/// geometry could not be formed; its typed reason is preserved instead of
/// presenting a missing matrix as a routine skip.
#[derive(Clone, Debug)]
pub enum SmoothingCorrectionOutcome {
    /// The analytic first-order correction.
    FirstOrder {
        factor: Option<Array2<f64>>,
        rho_covariance: Option<Array2<f64>>,
        method: Option<SmoothingCorrectionMethod>,
    },
    /// Exact first-order geometry was unavailable.
    Unavailable {
        reason: SmoothingCorrectionUnavailable,
        rho_covariance: Option<Array2<f64>>,
    },
}

impl SmoothingCorrectionOutcome {
    /// Consume the outcome without discarding how the matrix was made:
    /// `(factor, method)`, the correction's square-root factor `B`.
    pub(crate) fn into_correction_with_method(
        self,
    ) -> (Option<Array2<f64>>, Option<SmoothingCorrectionMethod>) {
        match self {
            SmoothingCorrectionOutcome::FirstOrder { factor, method, .. } => (factor, method),
            SmoothingCorrectionOutcome::Unavailable { .. } => (None, None),
        }
    }

    /// Read the regularized inverse outer Hessian `Cov(rho_hat)`, when the
    /// selected path produced one. This is consumed by higher-order LR
    /// inference and does not affect the covariance correction matrix.
    pub fn rho_covariance(&self) -> Option<&Array2<f64>> {
        match self {
            SmoothingCorrectionOutcome::FirstOrder { rho_covariance, .. }
            | SmoothingCorrectionOutcome::Unavailable { rho_covariance, .. } => {
                rho_covariance.as_ref()
            }
        }
    }
}

/// Process-wide count of numerical failures inside
/// [`RemlState::compute_smoothing_correction_outcome`]: the exact first-order
/// geometry could not be formed on a fit that has smoothing parameters.
pub(crate) static SMOOTHING_CORRECTION_NUMERICAL_FAILURE_COUNT: AtomicU64 = AtomicU64::new(0);

impl<'a> RemlState<'a> {
    /// Compute the pseudo-logdet `log|Σ λ_k S_k|₊`, its rank, and its first and
    /// second derivatives with respect to ρ — all from one eigendecomposition.
    ///
    /// On the positive eigenspace of `Σ λ_k S_k`:
    ///
    ///   ∂_k L = tr(S⁺ Aₖ)
    ///   ∂²_kl L = δ_{kl} ∂_k L − λₖ λₗ tr(S⁺ Sₖ S⁺ Sₗ)
    ///
    /// where Aₖ = λₖ Sₖ and S⁺ is the pseudoinverse on that eigenspace.
    ///
    /// The value `log|Σ λ_k S_k|₊` and its ρ-derivatives must range over the
    /// SAME positive eigenspace, or the analytic gradient differentiates a
    /// different function than the cost reports (the objective↔gradient desync
    /// class). Sourcing both from one [`PenaltyPseudologdet`] is the structural
    /// cure — the rank convention (eigenvalue-threshold over `Σ λ_k S_k`) is
    /// identical on both sides by construction (#901: a separate
    /// structural-rank value path desynced the GLM ρ-gradient against FD).
    pub(super) fn structural_penalty_logdet_value_and_derivatives(
        &self,
        rs_transformed: &[Array2<f64>],
        lambdas: &Array1<f64>,
    ) -> Result<(f64, usize, Array1<f64>, Array2<f64>), EstimationError> {
        let k_count = lambdas.len();
        if rs_transformed.len() != k_count {
            return Err(EstimationError::LayoutError(format!(
                "Penalty root/lambda count mismatch in structural logdet derivatives: roots={}, lambdas={}",
                rs_transformed.len(),
                k_count
            )));
        }
        if k_count == 0 {
            return Ok((
                0.0,
                0,
                Array1::zeros(k_count),
                Array2::zeros((k_count, k_count)),
            ));
        }

        // Build S_k = R_k^T R_k for each penalty component.
        let s_k_matrices: Vec<Array2<f64>> = rs_transformed
            .iter()
            .map(|r_k| gam_linalg::faer_ndarray::fast_atb(r_k, r_k))
            .collect();

        let lambdas_slice = lambdas
            .as_slice()
            .expect("owned Array1 is contiguous, so as_slice always succeeds");

        let pld = PenaltyPseudologdet::from_components(&s_k_matrices, lambdas_slice, 0.0)
            .map_err(EstimationError::LayoutError)?;

        let value = pld.value();
        let rank = pld.rank();
        let (det1, det2) = pld.rho_derivatives(&s_k_matrices, lambdas_slice);
        Ok((value, rank, det1, det2))
    }

    /// Block-local penalty logdet derivatives using `CanonicalPenalty`.
    ///
    /// When all penalties are block-disjoint, the eigendecomposition factorizes
    /// per-block at O(block_p³) instead of O(p³). Falls back to the dense path
    /// when blocks overlap.
    pub(super) fn structural_penalty_logdet_derivatives_block_local(
        &self,
        lambdas: &Array1<f64>,
        bundle: &EvalShared,
    ) -> Result<(Array1<f64>, Array2<f64>), EstimationError> {
        let (_, _, det1, det2) =
            self.structural_penalty_logdet_value_and_derivatives_block_local(lambdas, bundle)?;
        Ok((det1, det2))
    }

    /// Same as [`structural_penalty_logdet_derivatives_block_local`] but also
    /// returns the pseudo-logdet VALUE and rank from the SAME object the
    /// derivatives are taken on — see
    /// [`structural_penalty_logdet_value_and_derivatives`] for why value and
    /// derivative must share one positive eigenspace (#901).
    pub(super) fn structural_penalty_logdet_value_and_derivatives_block_local(
        &self,
        lambdas: &Array1<f64>,
        bundle: &EvalShared,
    ) -> Result<(f64, usize, Array1<f64>, Array2<f64>), EstimationError> {
        let k_count = self.canonical_penalties.len();
        if k_count == 0 || lambdas.len() != k_count {
            return Ok((
                0.0,
                0,
                Array1::zeros(k_count),
                Array2::zeros((k_count, k_count)),
            ));
        }

        let lambdas_slice = lambdas
            .as_slice()
            .expect("owned Array1 is contiguous, so as_slice always succeeds");

        // ONE factorization per evaluation point (#931): the same object also
        // serves the τ/ψ hyper-coordinate components in hyper.rs, so the
        // ridge and positive-eigenspace threshold of `log|Sλ|₊` are decided
        // exactly once for value, ρ-derivatives, and τ components alike.
        let pld = bundle.penalty_pseudologdet_original(
            &self.canonical_penalties,
            &self.penalty_unit_spectra(),
            lambdas_slice,
            self.p,
        )?;

        // The derivative contraction must read the SAME penalty components the
        // factorization was built from (#2454): `∂log|S̃|₊/∂ρ_k = λ_k tr(S̃⁺S̃_k)`
        // is only the derivative of `pld.value()` when `S̃_k` is the block
        // whose weighted sum `pld` factorized.
        let applied = bundle.applied_canonical_penalties(&self.canonical_penalties)?;
        let value = pld.value();
        let rank = pld.rank();
        let (det1, det2) = pld.rho_derivatives_from_penalties(&applied, lambdas_slice);
        Ok((value, rank, det1, det2))
    }

    pub(super) fn compute_lamlhessian_exact_from_bundle(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
    ) -> Result<Array2<f64>, EstimationError> {
        let mode = super::reml_outer_engine::EvalMode::ValueGradientHessian;
        let result = if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            self.evaluate_unified_sparse(rho, bundle, mode)?
        } else {
            self.evaluate_unified(rho, bundle, mode)?
        };
        result
            .hessian
            .materialize_dense()
            .map_err(|error| EstimationError::RemlOptimizationFailed(error.to_string()))?
            .ok_or_else(|| {
                EstimationError::RemlOptimizationFailed(
                    "Unified Hessian returned no analytic representation for VGH mode".into(),
                )
            })
    }

    pub(crate) fn compute_lamlhessian_consistent(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        if self.block_correction_latched() {
            crate::bail_invalid_estim!(
                "{}",
                crate::estimate::smoothing_correction::BLOCK_CORRECTION_OUTER_HESSIAN_NOT_ANALYTIC
            );
        }
        let bundle = self.obtain_eval_bundle(rho)?;
        let decision = self.selecthessian_strategy_policy(&bundle);
        match decision.strategy {
            super::inner_strategy::HessianEvalStrategyKind::SpectralExact => {
                self.compute_lamlhessian_exact_from_bundle(rho, &bundle)
            }
        }
    }

    /// Tier-0 of the marginal-smoothing inference stack (#938): the PSIS
    /// `ρ`-uncertainty adequacy diagnostic, evaluated against THIS live objective.
    ///
    /// This is the objective-lifecycle seam. The marginal posterior factorizes
    /// as `π(β, ρ | y) = π(β | ρ, y) · π(ρ | y)` with
    /// `π(ρ|y) ∝ exp(−criterion(ρ))`, and the diagnostic needs to evaluate the
    /// outer criterion at a handful of `ρ` near `ρ̂`. The criterion IS
    /// `Self::compute_cost` and the proposal Hessian IS
    /// [`Self::compute_lamlhessian_consistent`] — both `&self` — so a converged
    /// fit can produce the diagnostic WITHOUT retaining or rebuilding a
    /// separate objective: it runs against the same `RemlState` the fit
    /// converged on, while it is still in scope. The criterion the diagnostic
    /// samples is therefore the fit's own criterion bit-for-bit
    /// (`criterion(ρ̂) == reml_score`), so no fingerprint reconciliation is
    /// needed — there is exactly one objective.
    ///
    /// Returns `(None, None)` when there are no smoothing parameters
    /// (`K == 0`), the outer Hessian at `final_rho` is unavailable, or the
    /// criterion is infeasible at `ρ̂` — the diagnostic is simply absent, never
    /// an error.
    ///
    /// The Tier-0 diagnostic costs `M` outer-criterion evaluations (each an
    /// inner solve) near `ρ̂` plus a fresh ρ-Hessian, `M` the 2155 draws at
    /// which PSIS is reliable for a tail shape at the escalation cutoff; the
    /// returned fit does not need it, so the caller runs it only when
    /// ρ-posterior inference was requested. When the diagnostic grades the
    /// plug-in [`Escalate`], the tiers (#938) run HERE, against the same live objective — Tier 1
    /// quadrature or Tier 2 NUTS with the exact LAML `ρ`-gradient
    /// (`Self::compute_gradient`), whichever needs fewer criterion evaluations,
    /// with an honest `Unavailable` when the chosen tier fails.
    /// Post-hoc escalation after the `RemlState` is gone would need an owned
    /// rebuild recipe; running at the live seam avoids that entirely.
    ///
    /// [`Escalate`]: gam_problem::rho_posterior::RhoProposalAdequacy::Escalate
    ///
    /// `continuation` carries the box the outer arm searched and certified
    /// against (the #2812 resolvability domain). That box is a numerical device,
    /// not the support of `π(ρ|y)`: a draw outside it is still a model, the one
    /// its saturated terms' limit fits give, and carries its mass. Such a draw
    /// is valued by the criterion's affine continuation from the face
    /// ([`CriterionContinuation`]), so the inner solve is only ever asked for
    /// `ρ` inside the box, where P-IRLS has a resolvable minimum to report. A
    /// draw past a literal face, or one the criterion cannot value, refuses the
    /// diagnostic (or fails the escalation tier) with its reason; no draw is
    /// dropped.
    ///
    /// [`CriterionContinuation`]: crate::estimate::rho_domain::CriterionContinuation
    pub(crate) fn rho_posterior_inference(
        &self,
        final_rho: &Array1<f64>,
        continuation: &crate::estimate::rho_domain::CriterionContinuation,
    ) -> (
        gam_problem::rho_posterior::RhoPosteriorOutcome,
        Option<gam_problem::rho_posterior::RhoPosteriorEscalation>,
    ) {
        // DATA types contract-downed to gam-problem (#1521); the adequacy /
        // escalation COMPUTATION (`rho_posterior_adequacy`,
        // `escalate_rho_posterior`) lives UP in the monolith
        // `inference::rho_posterior` (its Tier-2 NUTS pulls the gam-inference
        // `hmc_io` sampler), so it is called DOWN here through the contract-down
        // `gam_problem::rho_posterior` escalator registry (#1521 trait-inversion
        // — the upward-compute back-edge is gone). Every early exit names why the
        // diagnostic was not formed (#2627), and none runs an escalation, so the
        // intervals stay plug-in + first-order corrected.
        use gam_problem::rho_posterior::{
            RhoPosteriorNotComputed, RhoPosteriorOutcome, RhoProposalAdequacy,
        };
        if final_rho.is_empty() {
            return (RhoPosteriorOutcome::NotApplicable, None);
        }
        let Some(escalator) = gam_problem::rho_posterior::rho_posterior_escalator() else {
            return (
                RhoPosteriorOutcome::NotComputed(RhoPosteriorNotComputed::EscalatorUnregistered),
                None,
            );
        };
        let outer_hessian = match self.compute_lamlhessian_consistent(final_rho) {
            Ok(outer_hessian) => outer_hessian,
            Err(error) => {
                return (
                    RhoPosteriorOutcome::NotComputed(
                        RhoPosteriorNotComputed::OuterHessianUnavailable {
                            reason: error.to_string(),
                        },
                    ),
                    None,
                );
            }
        };
        let cost = |rho: &Array1<f64>| {
            self.without_persistent_warm_start_store(|| self.compute_cost(rho))
                .map_err(|error| error.to_string())
        };
        // NUTS leapfrog gradients need the criterion value and gradient at the
        // same rho; compute them through one value+gradient outer evaluation so
        // the inner PIRLS solve and IFT state are shared by construction.
        let cost_and_gradient = |rho: &Array1<f64>| {
            self.without_persistent_warm_start_store(|| self.compute_cost_and_gradient(rho))
                .map_err(|error| error.to_string())
        };
        let outcome = match escalator.rho_posterior_adequacy(
            final_rho,
            &outer_hessian,
            &|rho| continuation.value(rho, cost, cost_and_gradient),
        ) {
            Ok(Some(adequacy)) => RhoPosteriorOutcome::Assessed(adequacy),
            Ok(None) => RhoPosteriorOutcome::NotApplicable,
            // The grade is a post-fit diagnostic of a fit the outer
            // optimizer already certified, so a refusal publishes the fit and
            // carries its typed reason with it.
            Err(refusal) => {
                log::debug!("rho-posterior adequacy diagnostic refused at the converged rho: {refusal}");
                RhoPosteriorOutcome::Refused(refusal)
            }
        };
        let escalation = match &outcome {
            // The diagnostic grades the plug-in `Escalate`: run the escalation
            // tier (Tier-1 quadrature / Tier-2 NUTS over ρ).
            RhoPosteriorOutcome::Assessed(adequacy)
                if adequacy.adequacy == RhoProposalAdequacy::Escalate =>
            {
                // #2450 — THE SAMPLER TARGETS A DISTRIBUTION; THE CRITERION DOES NOT.
                //
                // The tiers below sample `π(ρ|y) ∝ exp(−criterion(ρ))`, so the
                // criterion they are handed has to BE a log-density. The one the
                // optimizer minimizes is not: `evaluate_configured_rho_prior`
                // evaluates every unset coordinate directly as `Flat`, hence
                // exact zero for every finite ρ. That is the declared pure
                // REML/LAML criterion, but handing it to a sampler leaves no
                // proper prior over ρ: measured on the
                // n=600 anisotropic-Duchon fit in
                // `margslope_duchon_slowdown`, the NUTS tier doubles to maximum
                // depth and the fit does not return in 2136 s, against 1.28 s
                // once ρ carries a proper prior.
                //
                // `rho_prior_distribution_correction` provides the proper PC
                // contribution missing from the flat criterion. Adding it HERE, at the
                // sampler's own call site, is what keeps the two apart: no
                // criterion site is touched, so certification, the rail
                // certificates and every fit's λ̂ are byte-unchanged by
                // construction rather than by review.
                //
                // The Tier-0 diagnostic above is left on the criterion as the
                // optimizer sees it: it asks whether the PLUG-IN Gaussian is
                // adequate, which is a question about the object the fit
                // reports, and moving it is a separate decision recorded on
                // #2450.
                Some(escalator.escalate_rho_posterior(
                    final_rho,
                    &outer_hessian,
                    // The prior is analytic in ρ, so it is read at the draw
                    // itself; only the LAML part is continued from the face.
                    &mut |rho| {
                        let laml = continuation.value(rho, cost, cost_and_gradient)?;
                        let (prior_cost, _) = self
                            .rho_prior_distribution_correction(rho)
                            .map_err(|error| error.to_string())?;
                        Ok(laml + prior_cost)
                    },
                    &mut |rho| {
                        let (laml, gradient) =
                            continuation.value_and_gradient(rho, cost_and_gradient)?;
                        let (prior_cost, prior_gradient) = self
                            .rho_prior_distribution_correction(rho)
                            .map_err(|error| error.to_string())?;
                        Ok((laml + prior_cost, gradient + prior_gradient))
                    },
                ))
            }
            _ => None,
        };
        (outcome, escalation)
    }

    /// The smoothing-parameter correction of the coefficient covariance,
    /// `V_c = V_β + J V_ρ Jᵀ` (Wood, Pya & Säfken 2016, §3.1, mgcv's `Vc1`).
    ///
    /// `J = ∂β̂/∂ρ` comes from the implicit function theorem at the converged
    /// mode and `V_ρ` from the certified inverse of the LAML ρ-Hessian on its
    /// identified subspace (`invert_identified_rho_hessian`), judged off the
    /// `railed` coordinates exactly as the outer certificate judged it: a
    /// coordinate certified at its rail is held fixed there. Both are exact
    /// analytic objects the outer optimizer already needed, so the correction
    /// is one p×k sensitivity solve and a k×k inverse, for ANY number of
    /// smoothing parameters: there is no dimension gate, no sampling budget
    /// and no alternative estimand to fall back to. When the exact geometry
    /// cannot be formed the outcome says why (`Unavailable`), never silently
    /// substituting a different matrix.
    pub(crate) fn compute_smoothing_correction_outcome(
        &self,
        final_rho: &Array1<f64>,
        final_lambdas: &Array1<f64>,
        final_fit: &PirlsResult,
        outer_gradient: &Array1<f64>,
        outer_hessian: Option<&Array2<f64>>,
        caller_measured_hessian_error: &[gam_linalg::curvature_resolution::MeasuredHessianError],
        railed: &[usize],
    ) -> SmoothingCorrectionOutcome {
        let first_order = super::compute_smoothing_correction(
            self,
            final_rho,
            final_lambdas,
            final_fit,
            outer_gradient,
            railed,
            outer_hessian,
            caller_measured_hessian_error,
        );
        let outcome = match first_order.status {
            SmoothingCorrectionStatus::Unavailable(reason) => SmoothingCorrectionOutcome::Unavailable {
                reason,
                rho_covariance: first_order.rho_covariance,
            },
            _ => {
                let method = first_order.factor.as_ref().map(|_| {
                    SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
                        active_rank: first_order.active_rank.unwrap_or(0),
                        rho_dimension: final_rho.len(),
                    }
                });
                SmoothingCorrectionOutcome::FirstOrder {
                    factor: first_order.factor,
                    rho_covariance: first_order.rho_covariance,
                    method,
                }
            }
        };
        match &outcome {
            SmoothingCorrectionOutcome::FirstOrder { method, .. } => {
                log::info!("[smoothing-correction] branch=first-order method={method:?}");
            }
            SmoothingCorrectionOutcome::Unavailable {
                reason: SmoothingCorrectionUnavailable::OuterHessianNotAnalytic { error },
                ..
            } => {
                // Structural, not numerical: no analytic outer Hessian exists
                // for this fit, so the counter of numerical failures does not
                // move.
                log::debug!(
                    "[smoothing-correction] branch=unavailable reason=outer-hessian-not-analytic \
                     ({error})"
                );
            }
            SmoothingCorrectionOutcome::Unavailable { reason, .. } => {
                SMOOTHING_CORRECTION_NUMERICAL_FAILURE_COUNT.fetch_add(1, Ordering::Relaxed);
                log::debug!(
                    "[smoothing-correction] branch=unavailable reason={reason:?} failure_count={}",
                    SMOOTHING_CORRECTION_NUMERICAL_FAILURE_COUNT.load(Ordering::Relaxed),
                );
            }
        }
        outcome
    }
}

#[cfg(test)]
mod smoothing_correction_outcome_tests {
    //! Unit tests for the structured [`SmoothingCorrectionOutcome`] type and
    //! the analytic first-order correction it carries.
    use super::*;
    use ndarray::array;
    use std::sync::atomic::Ordering;

    fn make_first_order(with_matrix: bool) -> SmoothingCorrectionOutcome {
        SmoothingCorrectionOutcome::FirstOrder {
            factor: with_matrix.then(|| array![[1.0, 0.0], [0.0, 1.0]]),
            rho_covariance: None,
            method: with_matrix.then_some(
                SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
                    active_rank: 1,
                    rho_dimension: 1,
                },
            ),
        }
    }

    #[test]
    pub(crate) fn first_order_extraction_carries_matrix_and_method() {
        let (mat, method) = make_first_order(true).into_correction_with_method();
        assert_eq!(mat.expect("first-order matrix").dim(), (2, 2));
        assert!(matches!(
            method,
            Some(SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace { .. })
        ));
    }

    #[test]
    pub(crate) fn first_order_without_matrix_returns_none() {
        let (mat, method) = make_first_order(false).into_correction_with_method();
        assert!(mat.is_none());
        assert!(method.is_none());
    }

    #[test]
    pub(crate) fn severity_counter_is_monotonic() {
        let before = SMOOTHING_CORRECTION_NUMERICAL_FAILURE_COUNT.load(Ordering::Relaxed);
        SMOOTHING_CORRECTION_NUMERICAL_FAILURE_COUNT.fetch_add(1, Ordering::Relaxed);
        let after = SMOOTHING_CORRECTION_NUMERICAL_FAILURE_COUNT.load(Ordering::Relaxed);
        assert!(
            after > before,
            "numerical-failure counter must be monotonic ({} -> {})",
            before,
            after
        );
    }

    /// #582 — the smoothing-parameter correction must be response-scale
    /// equivariant: under `y → c·y` the returned correction (and hence
    /// `Vp = Vb + correction`) must scale by exactly `c²`, never `c⁴`.
    ///
    /// `J = −H⁻¹ λ S β̂` scales by `c` with β̂, while the profiled Gaussian
    /// REML ρ-Hessian (and so `V_ρ`) is scale-invariant, so `J V_ρ Jᵀ` scales
    /// by `c²`. The fixture calls
    /// [`RemlState::compute_smoothing_correction_outcome`] directly at a
    /// certified stationary ρ̂ found by bisection of the scalar outer gradient.
    #[test]
    pub(crate) fn first_order_smoothing_correction_is_response_scale_equivariant() {
        use crate::estimate::PenaltySpec;
        use gam_problem::{
            GlmLikelihoodSpec, InverseLink, LikelihoodSpec, ResponseFamily, StandardLink,
        };

        // Deterministic small Gaussian identity design (n=24, p=4: intercept +
        // 3 penalized columns).
        fn design(scale: f64) -> (Array2<f64>, Array1<f64>) {
            let n = 24usize;
            let p = 4usize;
            let mut x = Array2::<f64>::zeros((n, p));
            let mut y = Array1::<f64>::zeros(n);
            for i in 0..n {
                let t = (i as f64) / ((n - 1) as f64);
                let tau = std::f64::consts::TAU;
                x[[i, 0]] = 1.0;
                x[[i, 1]] = t;
                x[[i, 2]] = (tau * t).sin();
                x[[i, 3]] = (tau * t).cos();
                let base =
                    0.7 + 0.9 * t + 0.5 * (tau * t).sin() + 0.05 * ((i as f64) * 2.399_963).sin();
                y[i] = scale * base;
            }
            (x, y)
        }

        // Ridge on the 3 non-intercept columns; nullspace dim 1 (the intercept).
        let p = 4usize;
        let mut s = Array2::<f64>::zeros((p, p));
        for j in 1..p {
            s[[j, j]] = 1.0;
        }

        let run = |scale: f64| -> Array2<f64> {
            let (x, y) = design(scale);
            let n = x.nrows();
            let w = Array1::<f64>::ones(n);
            let offset = Array1::<f64>::zeros(n);

            let spec = PenaltySpec::Dense(s.clone());
            let canonical =
                gam_terms::construction::canonicalize_penalty_specs(&[spec], &[1], p, "test")
                    .map(|(canonical, _)| canonical)
                    .expect("canonicalize penalty");
            let cfg = RemlConfig::external(
                GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                    ResponseFamily::Gaussian,
                    InverseLink::Standard(StandardLink::Identity),
                )),
                1e-12,
                false,
            );
            let state = RemlState::newwith_offset(
                y.view(),
                x.clone(),
                w.view(),
                offset.view(),
                canonical,
                p,
                &cfg,
                Some(vec![1]),
                None,
                None,
            )
            .expect("build RemlState");

            // ρ̂ by root-finding the outer stationarity condition (#2614): the
            // identification floor `Σ_k |g_k|·v_k²` is calibrated for the
            // RESIDUAL gradient of a converged fit, so a non-stationary ρ
            // would mask the only direction. The root is scale-invariant for
            // profiled Gaussian REML, so both runs bisect to the same ρ̂.
            let outer_gradient_at = |candidate: f64| -> f64 {
                let probe = Array1::from_vec(vec![candidate]);
                state
                    .compute_gradient(&probe)
                    .unwrap_or_else(|err| panic!("outer gradient at rho={candidate}: {err}"))[0]
            };
            const FIXTURE_RHO_BOX: f64 = 30.0;
            let mut lo_rho = 1.0 - FIXTURE_RHO_BOX;
            let mut hi_rho = FIXTURE_RHO_BOX - 1.0;
            let g_lo = outer_gradient_at(lo_rho);
            let g_hi = outer_gradient_at(hi_rho);
            assert!(
                g_lo < 0.0 && g_hi > 0.0,
                "the REML profile has no interior stationary ρ on this design at \
                 scale {scale}: g({lo_rho}) = {g_lo:.6e}, g({hi_rho}) = {g_hi:.6e}"
            );
            let mut bisections = 0usize;
            while hi_rho - lo_rho > 1e-13 * (1.0 + hi_rho.abs()) && bisections < 200 {
                let mid = 0.5 * (lo_rho + hi_rho);
                if outer_gradient_at(mid) > 0.0 {
                    hi_rho = mid;
                } else {
                    lo_rho = mid;
                }
                bisections += 1;
            }
            let final_rho = Array1::from_vec(vec![0.5 * (lo_rho + hi_rho)]);

            let final_fit = state
                .execute_pirls_stateless_for_test(&final_rho)
                .expect("inner PIRLS at the converged rho");
            let finalgrad = state
                .compute_gradient(&final_rho)
                .unwrap_or_else(|err| panic!("outer gradient at rho={final_rho:?}: {err}"));
            let finalgrad_norm = finalgrad.dot(&finalgrad).sqrt();
            let stationarity_tol = 1e-6 * (1.0 + final_fit.deviance.abs());
            assert!(
                finalgrad_norm <= stationarity_tol,
                "the ρ bracket did not reach stationarity in {bisections} bisections: \
                 ρ̂ = {final_rho:?}, |g| = {finalgrad_norm:.6e} exceeds {stationarity_tol:.6e}"
            );

            let final_lambdas = Array1::from_vec(
                gam_problem::checked_exp_log_strengths(final_rho.iter().copied())
                    .expect("test rho lies in exact strength domain"),
            );
            // This harness has no outer solver behind it, so there is no second
            // assembly of the rho-Hessian to compare against: an absent
            // measurement, not a zero (#2748).
            let outcome = state.compute_smoothing_correction_outcome(
                &final_rho,
                &final_lambdas,
                final_fit.as_ref(),
                &finalgrad,
                None,
                &[],
                &[],
            );
            let outcome_description = format!("{outcome:?}");
            let (factor, method) = outcome.into_correction_with_method();
            assert!(
                matches!(
                    method,
                    Some(SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
                        active_rank: 1,
                        rho_dimension: 1,
                    })
                ),
                "an identified stationary ρ̂ must yield the analytic first-order \
                 correction; got {outcome_description}"
            );
            crate::estimate::smoothing_correction::smoothing_correction_gram(&factor.unwrap_or_else(
                || panic!("first-order outcome carries no matrix: {outcome_description}"),
            ))
        };

        let c = 1000.0_f64;
        let c2 = c * c;

        let corr1 = run(1.0);
        let corrc = run(c);

        // The correction must be materially non-zero (so the equivariance check
        // is not vacuous) and finite.
        let frob1 = corr1.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            frob1.is_finite() && frob1 > 0.0,
            "scale-1 correction must be finite and non-zero (‖corr‖={frob1:.3e})"
        );
        assert_eq!(corr1.dim(), corrc.dim(), "correction shape mismatch across scales");

        // Property under test: every entry scales by exactly c² (never c⁴).
        let mut worst_rel = 0.0_f64;
        let (mut wi, mut wj) = (0usize, 0usize);
        for i in 0..p {
            for j in 0..p {
                let expected = c2 * corr1[[i, j]];
                let got = corrc[[i, j]];
                let denom = expected.abs().max(c2 * frob1 * 1e-12).max(1e-300);
                let rel = (got - expected).abs() / denom;
                if rel > worst_rel {
                    worst_rel = rel;
                    wi = i;
                    wj = j;
                }
            }
        }
        assert!(
            worst_rel < 1e-6,
            "smoothing correction is not response-scale equivariant: \
             corr[{wi},{wj}] scales by {factor:.3e}·c² instead of c² \
             (corr@1={a:.6e}, corr@{c}={b:.6e}, expected {e:.6e}, rel {worst_rel:.3e}) (#582).",
            factor = corrc[[wi, wj]] / (c2 * corr1[[wi, wj]]).abs().max(1e-300),
            a = corr1[[wi, wj]],
            b = corrc[[wi, wj]],
            e = c2 * corr1[[wi, wj]],
        );
    }
}

//! Inner-Newton joint-Hessian and psi workspaces: the operator/gradient/
//! log-likelihood caching workspace implementing `ExactNewtonJointHessian
//! Workspace`, and the family-owned typed psi workspace.

use super::*;

pub(crate) struct SurvivalMarginalSlopePsiWorkspace {
    pub(crate) family: SurvivalMarginalSlopeFamily,
    pub(crate) block_states: Vec<ParameterBlockState>,
    pub(crate) specs: Vec<ParameterBlockSpec>,
    pub(crate) hyper_layout: crate::custom_family::CustomFamilyHyperLayout,
    pub(crate) cache: Option<EvalCache>,
    /// Outer-only ψ-calculus options. The `outer_score_subsample` field is
    /// the row mask threaded through `sigma_exact_joint_psi_terms_with_options`
    /// and the second-order / Hessian-drift counterparts to make the cached
    /// ψ calculus subsample-aware.
    pub(crate) options: BlockwiseFitOptions,
}

pub(crate) struct SurvivalMarginalSlopeExactNewtonJointHessianWorkspace {
    pub(crate) family: SurvivalMarginalSlopeFamily,
    pub(crate) block_states: Vec<ParameterBlockState>,
    pub(crate) joint_hessian_operator: Arc<dyn HyperOperator>,
    pub(crate) joint_hessian_diagonal: Array1<f64>,
    /// Cached joint log-likelihood and joint gradient from the same row pass
    /// that built the joint Hessian operator. Publishing these via
    /// `joint_log_likelihood_evaluation` / `joint_gradient_evaluation` lets
    /// the inner Newton driver in `custom_family.rs` skip its fallback
    /// separate-pass implementations and run on a single fused n-row sweep
    /// per workspace build.
    pub(crate) joint_log_likelihood: f64,
    pub(crate) joint_gradient: Array1<f64>,
    /// Cached per-row primary gradient + Hessian for timewiggle directional
    /// derivative reuse.  Built once during workspace construction so that
    /// repeated directional-derivative calls do not recompute them.
    pub(crate) eval_cache: Option<EvalCache>,
    /// Outer-only joint-Hessian directional-derivative options. The
    /// `outer_score_subsample` field is the row mask threaded through the
    /// `_with_options` directional-derivative helpers so the cached joint
    /// Hessian Hv-action paths can downscale to the stratified subsample at
    /// large scale. When `None`, the row iteration is identical to the
    /// legacy full-data path.
    pub(crate) options: BlockwiseFitOptions,
}

impl SurvivalMarginalSlopeExactNewtonJointHessianWorkspace {
    pub(crate) fn new(
        family: SurvivalMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
        options: BlockwiseFitOptions,
    ) -> Result<Self, String> {
        let (joint_hessian_operator, joint_hessian_diagonal, joint_log_likelihood, joint_gradient) =
            family.exact_newton_joint_hessian_operator(&block_states, &options)?;
        let eval_cache = if family.flex_timewiggle_active() && !family.flex_active() {
            Some(family.build_eval_cache(&block_states)?)
        } else {
            None
        };
        Ok(Self {
            family,
            block_states,
            joint_hessian_operator,
            joint_hessian_diagonal,
            joint_log_likelihood,
            joint_gradient,
            eval_cache,
            options,
        })
    }
}

impl ExactNewtonJointHessianWorkspace for SurvivalMarginalSlopeExactNewtonJointHessianWorkspace {
    fn warm_up_outer_caches_for_mode(
        &self,
        eval_mode: gam_problem::EvalMode,
    ) -> Result<(), String> {
        match eval_mode {
            gam_problem::EvalMode::ValueOnly
            | gam_problem::EvalMode::ValueAndGradient
            | gam_problem::EvalMode::ValueGradientHessian => Ok(()),
        }
    }

    fn joint_log_likelihood_evaluation(&self) -> Result<Option<f64>, String> {
        // Phase 2d fused-pass result: the same n-row sweep that produced
        // `joint_hessian_operator` also produced the joint log-likelihood,
        // so the inner Newton driver reads it from the workspace instead
        // of running a second full-data evaluation.
        Ok(Some(self.joint_log_likelihood))
    }

    fn joint_gradient_evaluation(
        &self,
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        Ok(Some(ExactNewtonJointGradientEvaluation {
            log_likelihood: self.joint_log_likelihood,
            gradient: self.joint_gradient.clone(),
        }))
    }

    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        // The operator we already built in `Self::new` carries every block
        // (h_tt, h_mm, h_gg, h_tm, …) of the joint Hessian. Asking the family
        // to re-materialize a dense p×p Hessian via
        // `evaluate_exact_newton_joint_dynamic_q_dense` would re-walk all n
        // rows just to repeat the J^T H J + Σ f K pullback we just finished;
        // at large scale that is the same n-row sweep twice per inner
        // joint-Newton cycle. Reuse the operator's `to_dense()` instead — an
        // O(p²) block copy. Numerically identical to the dense path modulo
        // FMA summation order.
        Ok(Some(self.joint_hessian_operator.to_dense()))
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, beta_flat: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(self.joint_hessian_operator.mul_vec(beta_flat)))
    }

    fn hessian_matvec_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<bool, String> {
        // Forward to HyperOperator's existing `mul_vec_into`, which writes the
        // matvec result directly into the caller-owned buffer with no
        // intermediate allocation. Used by inner-Newton PCG so each CG iter
        // avoids a fresh Array1<f64> on the survival large-scale hot path.
        if v.len() != self.joint_hessian_operator.dim()
            || out.len() != self.joint_hessian_operator.dim()
        {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "hessian_matvec_into: dim mismatch v={} out={} op={}",
                    v.len(),
                    out.len(),
                    self.joint_hessian_operator.dim()
                ),
            }
            .into());
        }
        self.joint_hessian_operator
            .mul_vec_into(v.view(), out.view_mut());
        Ok(true)
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(self.joint_hessian_diagonal.clone()))
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        if self.family.effective_flex_active(&self.block_states)?
            && !self.family.flex_timewiggle_active()
        {
            return self
                .family
                .exact_newton_joint_hessian_directional_derivative_operator_flex_no_wiggle_with_options(
                    &self.block_states,
                    d_beta_flat,
                    &self.options,
                )
                .map(Some);
        }
        if let Some(cache) = self.eval_cache.as_ref() {
            return self
                .family
                .exact_newton_joint_hessian_directional_derivative_timewiggle_cached(
                    &self.block_states,
                    d_beta_flat,
                    cache,
                )
                .map(|matrix| {
                    Some(Arc::new(gam_problem::DenseMatrixHyperOperator { matrix })
                        as Arc<dyn HyperOperator>)
                });
        }
        self.family
            .exact_newton_joint_hessian_directional_derivative(&self.block_states, d_beta_flat)
            .map(|result| {
                result.map(|matrix| {
                    Arc::new(gam_problem::DenseMatrixHyperOperator { matrix })
                        as Arc<dyn HyperOperator>
                })
            })
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if let Some(cache) = self.eval_cache.as_ref() {
            return self
                .family
                .exact_newton_joint_hessian_directional_derivative_timewiggle_cached(
                    &self.block_states,
                    d_beta_flat,
                    cache,
                )
                .map(Some);
        }
        self.family
            .exact_newton_joint_hessian_directional_derivative(&self.block_states, d_beta_flat)
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        if self.family.effective_flex_active(&self.block_states)?
            && !self.family.flex_timewiggle_active()
        {
            return self
                .family
                .exact_newton_joint_hessiansecond_directional_derivative_operator_flex_no_wiggle_with_options(
                    &self.block_states,
                    d_beta_u_flat,
                    d_beta_v_flat,
                    &self.options,
                )
                .map(Some);
        }
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &self.block_states,
                d_beta_u_flat,
                d_beta_v_flat,
            )
            .map(|result| {
                result.map(|matrix| {
                    Arc::new(gam_problem::DenseMatrixHyperOperator { matrix })
                        as Arc<dyn HyperOperator>
                })
            })
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &self.block_states,
                d_beta_u_flat,
                d_beta_v_flat,
            )
    }
}

impl SurvivalMarginalSlopePsiWorkspace {
    pub(crate) fn new(
        family: SurvivalMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
        specs: Vec<ParameterBlockSpec>,
        hyper_layout: crate::custom_family::CustomFamilyHyperLayout,
        options: BlockwiseFitOptions,
    ) -> Result<Self, String> {
        family.family_hyper.validate_layout(&hyper_layout)?;
        let cache = if family.flex_active() {
            None
        } else {
            Some(family.build_eval_cache(&block_states)?)
        };
        Ok(Self {
            family,
            block_states,
            specs,
            hyper_layout,
            cache,
            options,
        })
    }
}

impl ExactNewtonJointPsiWorkspace for SurvivalMarginalSlopePsiWorkspace {
    fn first_order_terms(
        &self,
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        match self
            .family
            .family_hyper_role(&self.hyper_layout, psi_index)?
        {
            None => self.family.psi_terms_inner_with_options(
                &self.block_states,
                self.hyper_layout.design_derivative_blocks(),
                psi_index,
                self.cache.as_ref(),
                &self.options,
            ),
            Some(SurvivalMarginalSlopeFamilyHyperAxis::LogSigma) => {
                self.family.sigma_exact_joint_psi_terms_with_options(
                    &self.block_states,
                    &self.specs,
                    &self.options,
                )
            }
            Some(SurvivalMarginalSlopeFamilyHyperAxis::Baseline(axis)) => {
                self.family.baseline_exact_joint_psi_terms_with_options(
                    &self.block_states,
                    axis,
                    &self.options,
                )
            }
        }
    }

    fn first_order_terms_all(&self) -> Result<Option<Vec<ExactNewtonJointPsiTerms>>, String> {
        let total = self.hyper_layout.len();
        if total == 0 {
            return Ok(Some(Vec::new()));
        }
        if self.hyper_layout.family_axis_count() != 0 {
            return Ok(None);
        }
        let psi_indices: Vec<usize> = (0..total).collect();
        if let Some(terms) = self.family.psi_terms_inner_batched_with_options(
            &self.block_states,
            self.hyper_layout.design_derivative_blocks(),
            &psi_indices,
            self.cache.as_ref(),
            &self.options,
        )? {
            return Ok(Some(terms));
        }

        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let per_axis: Result<Vec<Option<ExactNewtonJointPsiTerms>>, String> = psi_indices
            .into_par_iter()
            .map(|psi_index| {
                gam_problem::with_nested_parallel(|| {
                    self.family.psi_terms_inner_with_options(
                        &self.block_states,
                        self.hyper_layout.design_derivative_blocks(),
                        psi_index,
                        self.cache.as_ref(),
                        &self.options,
                    )
                })
            })
            .collect();
        let mut terms = Vec::with_capacity(total);
        for maybe_term in per_axis? {
            let Some(term) = maybe_term else {
                return Ok(None);
            };
            terms.push(term);
        }
        Ok(Some(terms))
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let axis_i = self.family.family_hyper_role(&self.hyper_layout, psi_i)?;
        let axis_j = self.family.family_hyper_role(&self.hyper_layout, psi_j)?;
        match (axis_i, axis_j) {
            (None, None) => self.family.psi_second_order_terms_inner_with_options(
                &self.block_states,
                self.hyper_layout.design_derivative_blocks(),
                psi_i,
                psi_j,
                self.cache.as_ref(),
                &self.options,
            ),
            (
                Some(SurvivalMarginalSlopeFamilyHyperAxis::LogSigma),
                Some(SurvivalMarginalSlopeFamilyHyperAxis::LogSigma),
            ) => self
                .family
                .sigma_exact_joint_psisecond_order_terms_with_options(
                    &self.block_states,
                    &self.options,
                ),
            (
                Some(SurvivalMarginalSlopeFamilyHyperAxis::Baseline(axis)),
                Some(SurvivalMarginalSlopeFamilyHyperAxis::Baseline(other_axis)),
            ) => self
                .family
                .baseline_exact_joint_psisecond_order_terms_with_options(
                    &self.block_states,
                    axis,
                    other_axis,
                    &self.options,
                ),
            (Some(SurvivalMarginalSlopeFamilyHyperAxis::Baseline(axis)), None) => self
                .family
                .baseline_design_exact_joint_psisecond_order_terms_with_options(
                    &self.block_states,
                    self.hyper_layout.design_derivative_blocks(),
                    axis,
                    psi_j,
                    &self.options,
                ),
            (None, Some(SurvivalMarginalSlopeFamilyHyperAxis::Baseline(axis))) => self
                .family
                .baseline_design_exact_joint_psisecond_order_terms_with_options(
                    &self.block_states,
                    self.hyper_layout.design_derivative_blocks(),
                    axis,
                    psi_i,
                    &self.options,
                ),
            _ => Err(format!(
                "survival marginal-slope family-touching pair ({psi_i}, {psi_j}) has no installed exact second-order row calculus"
            )),
        }
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<gam_problem::DriftDerivResult>, String> {
        match self
            .family
            .family_hyper_role(&self.hyper_layout, psi_index)?
        {
            None => self
                .family
                .psi_hessian_directional_derivative_operator_with_options(
                    &self.block_states,
                    self.hyper_layout.design_derivative_blocks(),
                    psi_index,
                    d_beta_flat,
                    &self.options,
                )
                .map(|result| result.map(gam_problem::DriftDerivResult::Operator)),
            Some(SurvivalMarginalSlopeFamilyHyperAxis::LogSigma) => self
                .family
                .sigma_exact_joint_psihessian_directional_derivative_with_options(
                    &self.block_states,
                    d_beta_flat,
                    &self.options,
                )
                .map(|result| result.map(gam_problem::DriftDerivResult::Dense)),
            Some(SurvivalMarginalSlopeFamilyHyperAxis::Baseline(axis)) => self
                .family
                .baseline_exact_joint_psihessian_directional_derivative_with_options(
                    &self.block_states,
                    axis,
                    d_beta_flat,
                    &self.options,
                )
                .map(|result| result.map(gam_problem::DriftDerivResult::Dense)),
        }
    }
}

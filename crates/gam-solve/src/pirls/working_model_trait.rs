//! The `WorkingModel` / `WorkingLikelihood` trait surface plus the shared
//! working-buffer machinery: candidate-screen results, the accepted-state cache
//! key, and the contiguous mu/weights/z and Newton-derivative buffer slices that
//! every per-family working-state writer routes through.

use super::*;

pub trait WorkingModel {
    fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError>;

    fn update_with_curvature(
        &mut self,
        beta: &Coefficients,
        _: HessianCurvatureKind,
    ) -> Result<WorkingState, EstimationError> {
        self.update(beta)
    }

    fn update_candidate(
        &mut self,
        beta: &Coefficients,
        curvature: HessianCurvatureKind,
    ) -> Result<WorkingState, EstimationError> {
        self.update_with_curvature(beta, curvature)
    }

    fn screen_candidate(
        &mut self,
        beta: &Coefficients,
        arr: &Array1<f64>,
        _: &LinearPredictor,
        curvature: HessianCurvatureKind,
    ) -> Result<CandidateEvaluation, EstimationError> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        self.update_candidate(beta, curvature)
            .map(CandidateEvaluation::Full)
    }

    fn supports_observed_information_curvature(&self) -> bool {
        false
    }

    /// Dispersion factor `k` the inner working weight carries but the reported
    /// deviance (`state.deviance` / `CandidateScreen::deviance`) does not, so the
    /// LM gain-ratio / stall-detection objective must be `k·deviance + penalty`
    /// to stay consistent with the `k`-scaled gradient and Hessian the step is
    /// built from. `1.0` for families whose weight carries no such factor (the
    /// solver objective is already self-consistent there). See
    /// `curvature::penalized_objective_deviance_scale` and issue #2128.
    fn penalized_deviance_scale(&self) -> Result<f64, EstimationError> {
        Ok(1.0)
    }
}

/// Result of a cheap LM-candidate screen: penalized objective + arithmetic
/// finiteness, without the gradient/Hessian needed for an accepted step.
#[derive(Debug, Clone)]
pub struct CandidateScreen {
    pub deviance: f64,
    pub penalty_term: f64,
    pub arithmetic_finite: bool,
}

/// Outcome of `WorkingModel::screen_candidate`: either a cheap screen result
/// (LM loop must upgrade with `update_with_curvature` on acceptance) or the
/// full state when screening was not applicable.
pub enum CandidateEvaluation {
    Screen(CandidateScreen),
    Full(WorkingState),
}

impl CandidateEvaluation {
    /// The penalized objective `dev_scale·deviance + penalty` (minus the Firth
    /// Jeffreys term when active). `dev_scale` is the family dispersion factor
    /// `k` (see `WorkingModel::penalized_deviance_scale`): the trial's deviance
    /// must be scaled by the SAME `k` the accepted state's is, so the LM
    /// gain-ratio compares like with like (issue #2128).
    #[inline]
    pub(crate) fn penalized_objective(&self, firth_bias_reduction: bool, dev_scale: f64) -> f64 {
        match self {
            Self::Screen(s) => dev_scale * s.deviance + s.penalty_term,
            Self::Full(state) => {
                let mut value = dev_scale * state.deviance + state.penalty_term;
                if firth_bias_reduction && let Some(j) = state.jeffreys_logdet() {
                    value -= 2.0 * j;
                }
                value
            }
        }
    }

    #[inline]
    pub(crate) fn arithmetic_finite(&self) -> bool {
        match self {
            Self::Screen(s) => s.arithmetic_finite,
            Self::Full(state) => state.gradient.iter().all(|g| g.is_finite()),
        }
    }

    #[inline]
    pub(crate) fn into_full(self) -> Option<WorkingState> {
        match self {
            Self::Full(state) => Some(state),
            Self::Screen(_) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PirlsAcceptedStateCacheKey {
    curvature: HessianCurvatureKind,
    firth_active: bool,
    beta_bits: Vec<u64>,
    arrow_latent_bits: Option<Vec<u64>>,
}

impl PirlsAcceptedStateCacheKey {
    pub(crate) fn requested(
        beta: &Coefficients,
        curvature: HessianCurvatureKind,
        options: &WorkingModelPirlsOptions,
    ) -> Self {
        Self::new(beta, curvature, options.firth_bias_reduction, options)
    }

    pub(crate) fn accepted(
        beta: &Coefficients,
        state: &WorkingState,
        options: &WorkingModelPirlsOptions,
    ) -> Self {
        Self::new(
            beta,
            state.hessian_curvature,
            matches!(state.firth, FirthDiagnostics::Active { .. }),
            options,
        )
    }

    pub(crate) fn new(
        beta: &Coefficients,
        curvature: HessianCurvatureKind,
        firth_active: bool,
        options: &WorkingModelPirlsOptions,
    ) -> Self {
        let arrow_latent_bits = options.arrow_schur.as_ref().map(|arrow_cfg| {
            arrow_cfg.snapshot_t.as_ref()()
                .iter()
                .map(|value| value.to_bits())
                .collect()
        });
        Self {
            curvature,
            firth_active,
            beta_bits: beta.as_ref().iter().map(|value| value.to_bits()).collect(),
            arrow_latent_bits,
        }
    }
}

/// Uncertainty inputs for integrated (GHQ) IRLS updates.
#[derive(Clone, Copy)]
pub(crate) struct IntegratedWorkingInput<'a> {
    pub quadctx: &'a crate::quadrature::QuadratureContext,
    pub se: ArrayView1<'a, f64>,
    pub mixture_link_state: Option<&'a MixtureLinkState>,
    pub sas_link_state: Option<&'a SasLinkState>,
}

pub struct WorkingDerivativeBuffersMut<'a> {
    pub(crate) c: &'a mut Array1<f64>,
    pub(crate) d: &'a mut Array1<f64>,
    pub(crate) dmu_deta: &'a mut Array1<f64>,
    pub(crate) d2mu_deta2: &'a mut Array1<f64>,
    pub(crate) d3mu_deta3: &'a mut Array1<f64>,
}

/// Contiguous mutable views of the three core working buffers (`mu`, `weights`,
/// `z`) shared by every PIRLS working-state writer.
pub(super) struct WorkingSlices<'a> {
    pub mu: &'a mut [f64],
    pub weights: &'a mut [f64],
    pub z: &'a mut [f64],
}

/// Contiguous mutable views of the Newton derivative/curvature buffers
/// (`c`, `d`, `dmu/deta` jet) shared by the full-derivative PIRLS writers.
pub(super) struct WorkingDerivSlices<'a> {
    pub c: &'a mut [f64],
    pub d: &'a mut [f64],
    pub dmu: &'a mut [f64],
    pub d2: &'a mut [f64],
    pub d3: &'a mut [f64],
}

/// Canonical "contiguous-or-panic" unpacking of the three core working buffers.
///
/// Single source of truth for the contiguity contract and panic messages that
/// every working-state writer relies on; every writer routes through this.
#[inline]
pub(super) fn working_slices<'a>(
    mu: &'a mut Array1<f64>,
    weights: &'a mut Array1<f64>,
    z: &'a mut Array1<f64>,
) -> WorkingSlices<'a> {
    WorkingSlices {
        mu: mu.as_slice_mut().expect("mu must be contiguous"),
        weights: weights.as_slice_mut().expect("weights must be contiguous"),
        z: z.as_slice_mut().expect("z must be contiguous"),
    }
}

/// Canonical "contiguous-or-panic" unpacking of the Newton derivative buffers.
///
/// Single source of truth for the contiguity contract and panic messages of the
/// `c`/`d`/`dmu`/`d2`/`d3` curvature buffers; every full-derivative writer routes
/// through this.
#[inline]
pub(super) fn working_deriv_slices<'a>(
    derivs: &'a mut WorkingDerivativeBuffersMut<'_>,
) -> WorkingDerivSlices<'a> {
    WorkingDerivSlices {
        c: derivs.c.as_slice_mut().expect("c must be contiguous"),
        d: derivs.d.as_slice_mut().expect("d must be contiguous"),
        dmu: derivs
            .dmu_deta
            .as_slice_mut()
            .expect("dmu_deta must be contiguous"),
        d2: derivs
            .d2mu_deta2
            .as_slice_mut()
            .expect("d2mu_deta2 must be contiguous"),
        d3: derivs
            .d3mu_deta3
            .as_slice_mut()
            .expect("d3mu_deta3 must be contiguous"),
    }
}

#[derive(Clone, Copy)]
pub(crate) struct WorkingBernoulliGeometry {
    pub(crate) mu: f64,
    pub(crate) weight: f64,
    pub(crate) z: f64,
    pub(crate) c: f64,
    pub(crate) d: f64,
}

/// Shared likelihood interface used by PIRLS working updates.
///
/// This keeps the update/deviance math in one place so engine-level likelihoods
/// and higher-level wrappers (custom family, GAMLSS warm starts) can share a
/// consistent implementation.
pub(crate) trait WorkingLikelihood {
    fn irls_update(
        &self,
        y: ArrayView1<f64>,
        eta: &Array1<f64>,
        priorweights: ArrayView1<f64>,
        mu: &mut Array1<f64>,
        weights: &mut Array1<f64>,
        z: &mut Array1<f64>,
        integrated: Option<IntegratedWorkingInput<'_>>,
        derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
    ) -> Result<(), EstimationError>;

    fn loglik_deviance(
        &self,
        y: ArrayView1<f64>,
        eta: &Array1<f64>,
        inverse_link: &InverseLink,
        priorweights: ArrayView1<f64>,
    ) -> Result<f64, EstimationError>;
}

impl WorkingLikelihood for GlmLikelihoodSpec {
    fn irls_update(
        &self,
        y: ArrayView1<f64>,
        eta: &Array1<f64>,
        priorweights: ArrayView1<f64>,
        mu: &mut Array1<f64>,
        weights: &mut Array1<f64>,
        z: &mut Array1<f64>,
        integrated: Option<IntegratedWorkingInput<'_>>,
        derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
    ) -> Result<(), EstimationError> {
        match (&self.spec.response, &self.spec.link, integrated.is_some()) {
            (ResponseFamily::Binomial, _, true) => {
                let integ = integrated.unwrap();
                update_glmvectors_integrated_by_family(
                    integ.quadctx,
                    y,
                    eta,
                    integ.se,
                    &self.spec,
                    priorweights,
                    mu,
                    weights,
                    z,
                    derivatives,
                    integ.mixture_link_state,
                    integ.sas_link_state,
                )?;
                Ok(())
            }
            (ResponseFamily::Binomial, link, false) => {
                if matches!(link, InverseLink::Mixture(_)) {
                    crate::bail_invalid_estim!(
                        "BinomialMixture IRLS update requires explicit mixture link state"
                            .to_string(),
                    );
                }
                update_glmvectors(
                    y,
                    eta,
                    &self.spec.link,
                    priorweights,
                    mu,
                    weights,
                    z,
                    derivatives,
                )?;
                Ok(())
            }
            (ResponseFamily::Gaussian, _, _) => {
                let resolved_scale = self
                    .resolved_scale()
                    .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
                update_glmvectors(
                    y,
                    eta,
                    &InverseLink::Standard(StandardLink::Identity),
                    priorweights,
                    mu,
                    weights,
                    z,
                    None,
                )?;
                // For Gaussian identity, the canonical IRLS working weight is
                //     w_i = prior_i * (dmu/deta)^2 / Var(Y_i | mu_i) = prior_i / phi.
                // When the scale metadata explicitly fixes phi (rather than
                // profiling sigma out), the working weights must include 1/phi
                // so that PIRLS minimises the scaled deviance / scaled negative
                // log-likelihood that the calibrator and downstream variance
                // calculations expect. `ProfiledGaussian` returns `None` here,
                // preserving the historical "weights == prior" behaviour for
                // the default profiled case.
                if let gam_problem::ResolvedLikelihoodScale::FixedGaussian { phi } = resolved_scale
                {
                    let phi = phi.value();
                    if phi != 1.0 {
                        let inv_phi = 1.0 / phi;
                        if !(inv_phi.is_finite() && inv_phi > 0.0) {
                            crate::bail_invalid_estim!(
                                "Gaussian reciprocal dispersion is not representable for phi={phi}: {inv_phi:?}"
                            );
                        }
                        weights.mapv_inplace(|w| w * inv_phi);
                    }
                }
                Ok(())
            }
            (ResponseFamily::Poisson, _, _) => {
                write_poisson_log_working_state(y, eta, priorweights, mu, weights, z, derivatives)
            }
            (ResponseFamily::Tweedie { p }, _, _) => {
                let p = *p;
                write_tweedie_log_working_state(
                    y,
                    eta,
                    priorweights,
                    p,
                    self.resolved_tweedie_phi()
                        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?,
                    mu,
                    weights,
                    z,
                    derivatives,
                )?;
                Ok(())
            }
            (ResponseFamily::NegativeBinomial { .. }, _, _) => {
                let theta = self
                    .resolved_negbin_theta()
                    .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
                write_negative_binomial_log_working_state(
                    y,
                    eta,
                    priorweights,
                    theta,
                    mu,
                    weights,
                    z,
                    derivatives,
                )?;
                Ok(())
            }
            (ResponseFamily::Beta { .. }, _, _) => {
                let phi = self
                    .resolved_beta_precision()
                    .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
                write_beta_logit_working_state(
                    y,
                    eta,
                    priorweights,
                    phi,
                    mu,
                    weights,
                    z,
                    derivatives,
                )?;
                Ok(())
            }
            (ResponseFamily::Gamma, _, _) => {
                let shape = self
                    .resolved_gamma_shape()
                    .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
                write_gamma_log_working_state(
                    y,
                    eta,
                    priorweights,
                    shape,
                    mu,
                    weights,
                    z,
                    derivatives,
                )
            }
            (ResponseFamily::RoystonParmar, _, _) => Err(EstimationError::InvalidInput(
                "RoystonParmar is survival-specific and not a GLM IRLS family".to_string(),
            )),
        }
    }

    fn loglik_deviance(
        &self,
        y: ArrayView1<f64>,
        eta: &Array1<f64>,
        inverse_link: &InverseLink,
        priorweights: ArrayView1<f64>,
    ) -> Result<f64, EstimationError> {
        if matches!(self.spec.response, ResponseFamily::Tweedie { .. }) {
            validate_tweedie_responses(&y, &priorweights)?;
        }
        calculate_deviance_from_eta(y, eta, self, inverse_link, priorweights)
    }
}

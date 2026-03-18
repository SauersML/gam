use crate::custom_family::BlockwiseFitOptions;
use crate::estimate::{EstimationError, FitOptions, FittedLinkState, UnifiedFitResult};
use crate::families::bernoulli_marginal_slope::{
    BernoulliMarginalSlopeFitResult, BernoulliMarginalSlopeTermSpec,
    fit_bernoulli_marginal_slope_terms,
};
use crate::families::gamlss::{
    BinomialLocationScaleFitResult, BinomialLocationScaleTermSpec, BlockwiseTermFitResult,
    BlockwiseTermFitResultParts, GaussianLocationScaleFitResult, GaussianLocationScaleTermSpec,
    WiggleBlockConfig, fit_binomial_location_scale_terms,
    fit_binomial_location_scale_terms_with_selected_wiggle,
    fit_binomial_mean_wiggle_terms_with_selected_basis, fit_gaussian_location_scale_terms,
    fit_gaussian_location_scale_terms_with_selected_wiggle,
    select_binomial_location_scale_link_wiggle_basis_from_pilot,
    select_binomial_mean_link_wiggle_basis_from_pilot,
    select_gaussian_location_scale_link_wiggle_basis_from_pilot,
};
use crate::families::survival_location_scale::{
    DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD, SurvivalLocationScaleTermFitResult,
    SurvivalLocationScaleTermSpec, fit_survival_location_scale_terms,
    fit_survival_location_scale_terms_with_selected_wiggle,
    select_survival_link_wiggle_basis_from_pilot,
};
use crate::families::survival_marginal_slope::{
    DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD, SurvivalMarginalSlopeFitResult,
    SurvivalMarginalSlopeTermSpec, fit_survival_marginal_slope_terms,
};
use crate::families::transformation_normal::{
    TransformationNormalConfig, TransformationNormalFitResult, TransformationWarmStart,
    fit_transformation_normal,
};
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
use crate::smooth::{
    AdaptiveRegularizationDiagnostics, SpatialLengthScaleOptimizationOptions, TermCollectionDesign,
    TermCollectionSpec, fit_term_collectionwith_spatial_length_scale_optimization,
};
use crate::solver::optimize::{CostOnlyOptimizationRequest, optimize_cost_only};
use crate::types::{InverseLink, LikelihoodFamily, LinkFunction, MixtureLinkSpec, SasLinkSpec};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct LinkWiggleConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_orders: Vec<usize>,
    pub double_penalty: bool,
}

#[derive(Clone, Debug)]
pub struct StandardBinomialWiggleConfig {
    pub link_kind: InverseLink,
    pub wiggle: LinkWiggleConfig,
}

pub struct StandardFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub offset: Array1<f64>,
    pub spec: TermCollectionSpec,
    pub family: LikelihoodFamily,
    pub options: FitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
    pub wiggle: Option<StandardBinomialWiggleConfig>,
    pub wiggle_options: Option<BlockwiseFitOptions>,
}

pub struct GaussianLocationScaleFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: GaussianLocationScaleTermSpec,
    pub wiggle: Option<LinkWiggleConfig>,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}

pub struct BinomialLocationScaleFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: BinomialLocationScaleTermSpec,
    pub wiggle: Option<LinkWiggleConfig>,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}

pub struct SurvivalLocationScaleFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: SurvivalLocationScaleTermSpec,
    pub wiggle: Option<LinkWiggleConfig>,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
    pub optimize_inverse_link: bool,
}

pub(crate) fn survival_inverse_link_has_free_parameters(link: &InverseLink) -> bool {
    match link {
        InverseLink::Sas(_) | InverseLink::BetaLogistic(_) => true,
        InverseLink::Mixture(state) => !state.rho.is_empty(),
        InverseLink::Standard(_) => false,
    }
}

pub struct BernoulliMarginalSlopeFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: BernoulliMarginalSlopeTermSpec,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}

pub struct SurvivalMarginalSlopeFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: SurvivalMarginalSlopeTermSpec,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}

pub struct TransformationNormalFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub response: Array1<f64>,
    pub covariate_spec: TermCollectionSpec,
    pub config: TransformationNormalConfig,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
    pub warm_start: Option<TransformationWarmStart>,
}

pub enum FitRequest<'a> {
    Standard(StandardFitRequest<'a>),
    GaussianLocationScale(GaussianLocationScaleFitRequest<'a>),
    BinomialLocationScale(BinomialLocationScaleFitRequest<'a>),
    SurvivalLocationScale(SurvivalLocationScaleFitRequest<'a>),
    BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest<'a>),
    SurvivalMarginalSlope(SurvivalMarginalSlopeFitRequest<'a>),
    TransformationNormal(TransformationNormalFitRequest<'a>),
}

pub struct StandardFitResult {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub resolvedspec: TermCollectionSpec,
    pub adaptive_diagnostics: Option<AdaptiveRegularizationDiagnostics>,
    pub saved_link_state: FittedLinkState,
    pub wiggle_knots: Option<Array1<f64>>,
    pub wiggle_degree: Option<usize>,
}

pub struct SurvivalLocationScaleFitResult {
    pub fit: SurvivalLocationScaleTermFitResult,
    pub inverse_link: InverseLink,
    pub wiggle_knots: Option<Array1<f64>>,
    pub wiggle_degree: Option<usize>,
}

struct SurvivalLocationScaleProfile {
    fit: SurvivalLocationScaleTermFitResult,
    inverse_link: InverseLink,
    wiggle_knots: Option<Array1<f64>>,
    wiggle_degree: Option<usize>,
}

impl SurvivalLocationScaleProfile {
    fn objective(&self) -> f64 {
        self.fit.fit.reml_score
    }

    fn into_result(self) -> SurvivalLocationScaleFitResult {
        SurvivalLocationScaleFitResult {
            fit: self.fit,
            inverse_link: self.inverse_link,
            wiggle_knots: self.wiggle_knots,
            wiggle_degree: self.wiggle_degree,
        }
    }
}

pub enum FitResult {
    Standard(StandardFitResult),
    GaussianLocationScale(GaussianLocationScaleFitResult),
    BinomialLocationScale(BinomialLocationScaleFitResult),
    SurvivalLocationScale(SurvivalLocationScaleFitResult),
    BernoulliMarginalSlope(BernoulliMarginalSlopeFitResult),
    SurvivalMarginalSlope(SurvivalMarginalSlopeFitResult),
    TransformationNormal(TransformationNormalFitResult),
}

fn resolved_wiggle_inverse_link(
    family: LikelihoodFamily,
    fit: &UnifiedFitResult,
    fallback: &InverseLink,
) -> Result<InverseLink, String> {
    let resolved = match fit.fitted_link_state(family).map_err(|e| e.to_string())? {
        FittedLinkState::Standard(Some(link)) => InverseLink::Standard(link),
        FittedLinkState::Standard(None) => fallback.clone(),
        FittedLinkState::Sas { state, .. } => InverseLink::Sas(state),
        FittedLinkState::BetaLogistic { state, .. } => InverseLink::BetaLogistic(state),
        FittedLinkState::Mixture { state, .. } => InverseLink::Mixture(state),
    };
    ensure_joint_wiggle_supported(&resolved, "standard link wiggle")?;
    Ok(resolved)
}

fn ensure_joint_wiggle_supported(link: &InverseLink, context: &str) -> Result<(), String> {
    match link {
        InverseLink::Standard(crate::types::LinkFunction::Sas)
        | InverseLink::Standard(crate::types::LinkFunction::BetaLogistic)
        | InverseLink::Sas(_)
        | InverseLink::BetaLogistic(_)
        | InverseLink::Mixture(_) => Err(format!(
            "{context} does not support SAS/BetaLogistic/Mixture links; wiggle is only available for jointly fitted standard links"
        )),
        InverseLink::Standard(_) => Ok(()),
    }
}

fn fit_standard_model(request: StandardFitRequest<'_>) -> Result<StandardFitResult, String> {
    let fitted = fit_term_collectionwith_spatial_length_scale_optimization(
        request.data,
        request.y.clone(),
        request.weights.clone(),
        request.offset.clone(),
        &request.spec,
        request.family,
        &request.options,
        &request.kappa_options,
    )
    .map_err(|e| e.to_string())?;

    let result = StandardFitResult {
        saved_link_state: fitted.fit.fitted_link.clone(),
        fit: fitted.fit,
        design: fitted.design,
        resolvedspec: fitted.resolvedspec,
        adaptive_diagnostics: fitted.adaptive_diagnostics,
        wiggle_knots: None,
        wiggle_degree: None,
    };

    let Some(wiggle) = request.wiggle else {
        return Ok(result);
    };
    let wiggle_options = request
        .wiggle_options
        .ok_or_else(|| "standard wiggle workflow requires blockwise wiggle options".to_string())?;
    let wiggle_link_kind =
        resolved_wiggle_inverse_link(request.family, &result.fit, &wiggle.link_kind)?;
    let selected_wiggle_basis = select_binomial_mean_link_wiggle_basis_from_pilot(
        &result.design,
        &result.fit,
        &WiggleBlockConfig {
            degree: wiggle.wiggle.degree,
            num_internal_knots: wiggle.wiggle.num_internal_knots,
            penalty_order: 2,
            double_penalty: wiggle.wiggle.double_penalty,
        },
        &wiggle.wiggle.penalty_orders,
    )?;

    let solved = fit_binomial_mean_wiggle_terms_with_selected_basis(
        request.data,
        &result.resolvedspec,
        &result.design,
        &result.fit,
        &request.y,
        &request.weights,
        wiggle_link_kind,
        selected_wiggle_basis,
        &wiggle_options,
        &request.kappa_options,
    )?;

    Ok(StandardFitResult {
        saved_link_state: result.saved_link_state,
        fit: solved.fit,
        design: solved.design,
        resolvedspec: solved.resolvedspec,
        adaptive_diagnostics: result.adaptive_diagnostics,
        wiggle_knots: Some(solved.wiggle_knots),
        wiggle_degree: Some(solved.wiggle_degree),
    })
}

fn fit_gaussian_location_scale_model(
    request: GaussianLocationScaleFitRequest<'_>,
) -> Result<GaussianLocationScaleFitResult, String> {
    if let Some(wiggle_cfg) = request.wiggle {
        let pilot = fit_gaussian_location_scale_terms(
            request.data,
            GaussianLocationScaleTermSpec {
                y: request.spec.y.clone(),
                weights: request.spec.weights.clone(),
                meanspec: request.spec.meanspec.clone(),
                log_sigmaspec: request.spec.log_sigmaspec.clone(),
            },
            &request.options,
            &request.kappa_options,
        )?;
        let selected_wiggle_basis = select_gaussian_location_scale_link_wiggle_basis_from_pilot(
            &pilot,
            &WiggleBlockConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_order: 2,
                double_penalty: wiggle_cfg.double_penalty,
            },
            &wiggle_cfg.penalty_orders,
        )?;
        let solved = fit_gaussian_location_scale_terms_with_selected_wiggle(
            request.data,
            request.spec,
            selected_wiggle_basis,
            &request.options,
            &request.kappa_options,
        )?;
        let fit = solved.fit.fit;
        let beta_link_wiggle = fit.block_states.get(2).map(|b| b.beta.to_vec());
        Ok(GaussianLocationScaleFitResult {
            fit: BlockwiseTermFitResult::try_from_parts(BlockwiseTermFitResultParts {
                fit,
                meanspec_resolved: solved.fit.meanspec_resolved,
                noisespec_resolved: solved.fit.noisespec_resolved,
                mean_design: solved.fit.mean_design,
                noise_design: solved.fit.noise_design,
            })?,
            wiggle_knots: Some(solved.wiggle_knots),
            wiggle_degree: Some(solved.wiggle_degree),
            beta_link_wiggle,
        })
    } else {
        let fit = fit_gaussian_location_scale_terms(
            request.data,
            request.spec,
            &request.options,
            &request.kappa_options,
        )?;
        Ok(GaussianLocationScaleFitResult {
            fit,
            wiggle_knots: None,
            wiggle_degree: None,
            beta_link_wiggle: None,
        })
    }
}

fn fit_binomial_location_scale_model(
    request: BinomialLocationScaleFitRequest<'_>,
) -> Result<BinomialLocationScaleFitResult, String> {
    if let Some(wiggle_cfg) = request.wiggle {
        ensure_joint_wiggle_supported(
            &request.spec.link_kind,
            "binomial location-scale link wiggle",
        )?;
        let pilot = fit_binomial_location_scale_terms(
            request.data,
            BinomialLocationScaleTermSpec {
                y: request.spec.y.clone(),
                weights: request.spec.weights.clone(),
                link_kind: request.spec.link_kind.clone(),
                thresholdspec: request.spec.thresholdspec.clone(),
                log_sigmaspec: request.spec.log_sigmaspec.clone(),
            },
            &request.options,
            &request.kappa_options,
        )?;
        let selected_wiggle_basis = select_binomial_location_scale_link_wiggle_basis_from_pilot(
            &pilot,
            &WiggleBlockConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_order: 2,
                double_penalty: wiggle_cfg.double_penalty,
            },
            &wiggle_cfg.penalty_orders,
        )?;
        let solved = fit_binomial_location_scale_terms_with_selected_wiggle(
            request.data,
            request.spec,
            selected_wiggle_basis,
            &request.options,
            &request.kappa_options,
        )?;
        let fit = solved.fit.fit;
        let beta_link_wiggle = fit.block_states.get(2).map(|b| b.beta.to_vec());
        Ok(BinomialLocationScaleFitResult {
            fit: BlockwiseTermFitResult::try_from_parts(BlockwiseTermFitResultParts {
                fit,
                meanspec_resolved: solved.fit.meanspec_resolved,
                noisespec_resolved: solved.fit.noisespec_resolved,
                mean_design: solved.fit.mean_design,
                noise_design: solved.fit.noise_design,
            })?,
            wiggle_knots: Some(solved.wiggle_knots),
            wiggle_degree: Some(solved.wiggle_degree),
            beta_link_wiggle,
        })
    } else {
        let solved = fit_binomial_location_scale_terms(
            request.data,
            request.spec,
            &request.options,
            &request.kappa_options,
        )?;
        Ok(BinomialLocationScaleFitResult {
            fit: solved,
            wiggle_knots: None,
            wiggle_degree: None,
            beta_link_wiggle: None,
        })
    }
}

fn fit_survival_location_scale_model(
    request: SurvivalLocationScaleFitRequest<'_>,
) -> Result<SurvivalLocationScaleFitResult, String> {
    // Profile one coherent survival subproblem at a fixed inverse-link state:
    // select/apply the link-wiggle basis for that state, then solve the full
    // penalized location-scale fit on the resulting model.
    fn profile_survival_location_scale(
        data: ArrayView2<'_, f64>,
        spec: SurvivalLocationScaleTermSpec,
        wiggle: Option<LinkWiggleConfig>,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<SurvivalLocationScaleProfile, String> {
        let mut wiggle_knots = None;
        let mut wiggle_degree = None;
        let inverse_link = spec.inverse_link.clone();

        let fit = if let Some(wiggle) = wiggle {
            let mut pilot_spec = spec.clone();
            pilot_spec.linkwiggle_block = None;
            let pilot = fit_survival_location_scale_terms(data, pilot_spec, kappa_options)?;
            let selected_wiggle_basis = select_survival_link_wiggle_basis_from_pilot(
                &pilot,
                &WiggleBlockConfig {
                    degree: wiggle.degree,
                    num_internal_knots: wiggle.num_internal_knots,
                    penalty_order: 2,
                    double_penalty: wiggle.double_penalty,
                },
                &wiggle.penalty_orders,
            )?;
            wiggle_knots = Some(selected_wiggle_basis.knots.clone());
            wiggle_degree = Some(selected_wiggle_basis.degree);
            fit_survival_location_scale_terms_with_selected_wiggle(
                data,
                spec,
                selected_wiggle_basis,
                kappa_options,
            )?
        } else {
            fit_survival_location_scale_terms(data, spec, kappa_options)?
        };

        Ok(SurvivalLocationScaleProfile {
            fit,
            inverse_link,
            wiggle_knots,
            wiggle_degree,
        })
    }

    fn profile_survival_location_scale_with_inverse_link(
        data: ArrayView2<'_, f64>,
        spec: &SurvivalLocationScaleTermSpec,
        inverse_link: InverseLink,
        wiggle: Option<LinkWiggleConfig>,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<SurvivalLocationScaleProfile, String> {
        let mut spec_at_link = spec.clone();
        spec_at_link.inverse_link = inverse_link;
        profile_survival_location_scale(data, spec_at_link, wiggle, kappa_options)
    }

    fn optimize_survival_inverse_link_profile(
        data: ArrayView2<'_, f64>,
        spec: &SurvivalLocationScaleTermSpec,
        wiggle: Option<LinkWiggleConfig>,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<SurvivalLocationScaleProfile, String> {
        let optimize_link_parameters =
            |init: Array1<f64>,
             name: &str,
             final_wiggle: Option<LinkWiggleConfig>,
             mut objective: Box<dyn FnMut(&Array1<f64>) -> Result<f64, EstimationError>>,
             recover: Box<dyn Fn(&Array1<f64>) -> Option<InverseLink>>|
             -> Result<Option<SurvivalLocationScaleProfile>, String> {
                let dim = init.len();
                let mut seed_config = crate::seeding::SeedConfig::default();
                seed_config.max_seeds = 8;
                seed_config.screening_budget = 3;
                seed_config.risk_profile = crate::seeding::SeedRiskProfile::Survival;
                let mut outer_request = CostOnlyOptimizationRequest::new(init.clone());
                outer_request.tolerance = 1e-4;
                outer_request.max_iter = 30;
                outer_request.fd_step = 1e-3;
                outer_request.seed_config = seed_config;
                let context = format!("survival inverse-link optimization ({name}, dim={dim})");
                match optimize_cost_only(outer_request, &context, |rho| objective(rho)) {
                    Ok(result) => {
                        if let Some(link) = recover(&result.rho) {
                            eprintln!(
                                "[survival link opt] optimized {name} params: dim={} iters={} converged={} finalobj={:.6e}",
                                result.rho.len(),
                                result.iterations,
                                result.converged,
                                result.final_value
                            );
                            profile_survival_location_scale_with_inverse_link(
                                data,
                                spec,
                                link,
                                final_wiggle.clone(),
                                kappa_options,
                            )
                            .map(Some)
                        } else {
                            Ok(None)
                        }
                    }
                    Err(err) => {
                        eprintln!(
                            "[survival link opt] {name} optimization failed; using initial params: {err}"
                        );
                        Ok(None)
                    }
                }
            };

        match spec.inverse_link.clone() {
            InverseLink::Sas(state0) => {
                let init = Array1::from_vec(vec![state0.epsilon, state0.log_delta]);
                let wiggle_cfg = wiggle.clone();
                let optimized = optimize_link_parameters(
                    init,
                    "SAS",
                    wiggle.clone(),
                    Box::new(|theta: &Array1<f64>| {
                        let state = state_from_sasspec(SasLinkSpec {
                            initial_epsilon: theta[0],
                            initial_log_delta: theta[1],
                        })
                        .map_err(EstimationError::InvalidInput)?;
                        Ok(profile_survival_location_scale_with_inverse_link(
                            data,
                            spec,
                            InverseLink::Sas(state),
                            wiggle_cfg.clone(),
                            kappa_options,
                        )
                        .map_err(EstimationError::InvalidInput)?
                        .objective())
                    }),
                    Box::new(|rho| {
                        state_from_sasspec(SasLinkSpec {
                            initial_epsilon: rho[0],
                            initial_log_delta: rho[1],
                        })
                        .ok()
                        .map(InverseLink::Sas)
                    }),
                )?;
                if let Some(profile) = optimized {
                    Ok(profile)
                } else {
                    profile_survival_location_scale(data, spec.clone(), wiggle, kappa_options)
                }
            }
            InverseLink::BetaLogistic(state0) => {
                let init = Array1::from_vec(vec![state0.epsilon, state0.log_delta]);
                let wiggle_cfg = wiggle.clone();
                let optimized = optimize_link_parameters(
                    init,
                    "BetaLogistic",
                    wiggle.clone(),
                    Box::new(|theta: &Array1<f64>| {
                        let state = state_from_beta_logisticspec(SasLinkSpec {
                            initial_epsilon: theta[0],
                            initial_log_delta: theta[1],
                        })
                        .map_err(EstimationError::InvalidInput)?;
                        Ok(profile_survival_location_scale_with_inverse_link(
                            data,
                            spec,
                            InverseLink::BetaLogistic(state),
                            wiggle_cfg.clone(),
                            kappa_options,
                        )
                        .map_err(EstimationError::InvalidInput)?
                        .objective())
                    }),
                    Box::new(|rho| {
                        state_from_beta_logisticspec(SasLinkSpec {
                            initial_epsilon: rho[0],
                            initial_log_delta: rho[1],
                        })
                        .ok()
                        .map(InverseLink::BetaLogistic)
                    }),
                )?;
                if let Some(profile) = optimized {
                    Ok(profile)
                } else {
                    profile_survival_location_scale(data, spec.clone(), wiggle, kappa_options)
                }
            }
            InverseLink::Mixture(state0) if !state0.rho.is_empty() => {
                let components = state0.components.clone();
                let components_recover = components.clone();
                let wiggle_cfg = wiggle.clone();
                let optimized = optimize_link_parameters(
                    state0.rho.clone(),
                    "mixture",
                    wiggle.clone(),
                    Box::new(move |rho: &Array1<f64>| {
                        let state = state_fromspec(&MixtureLinkSpec {
                            components: components.clone(),
                            initial_rho: rho.clone(),
                        })
                        .map_err(EstimationError::InvalidInput)?;
                        Ok(profile_survival_location_scale_with_inverse_link(
                            data,
                            spec,
                            InverseLink::Mixture(state),
                            wiggle_cfg.clone(),
                            kappa_options,
                        )
                        .map_err(EstimationError::InvalidInput)?
                        .objective())
                    }),
                    Box::new(move |rho| {
                        state_fromspec(&MixtureLinkSpec {
                            components: components_recover.clone(),
                            initial_rho: rho.to_owned(),
                        })
                        .ok()
                        .map(InverseLink::Mixture)
                    }),
                )?;
                if let Some(profile) = optimized {
                    Ok(profile)
                } else {
                    profile_survival_location_scale(data, spec.clone(), wiggle, kappa_options)
                }
            }
            _ => profile_survival_location_scale(data, spec.clone(), wiggle, kappa_options),
        }
    }

    let profile = if request.optimize_inverse_link {
        optimize_survival_inverse_link_profile(
            request.data,
            &request.spec,
            request.wiggle.clone(),
            &request.kappa_options,
        )?
    } else {
        profile_survival_location_scale(
            request.data,
            request.spec.clone(),
            request.wiggle.clone(),
            &request.kappa_options,
        )?
    };

    Ok(profile.into_result())
}

fn fit_bernoulli_marginal_slope_model(
    request: BernoulliMarginalSlopeFitRequest<'_>,
) -> Result<BernoulliMarginalSlopeFitResult, String> {
    fit_bernoulli_marginal_slope_terms(
        request.data,
        request.spec,
        &request.options,
        &request.kappa_options,
    )
}

fn fit_survival_marginal_slope_model(
    request: SurvivalMarginalSlopeFitRequest<'_>,
) -> Result<SurvivalMarginalSlopeFitResult, String> {
    fit_survival_marginal_slope_terms(
        request.data,
        request.spec,
        &request.options,
        &request.kappa_options,
    )
}

fn fit_transformation_normal_model(
    request: TransformationNormalFitRequest<'_>,
) -> Result<TransformationNormalFitResult, String> {
    fit_transformation_normal(
        &request.response,
        request.data,
        &request.covariate_spec,
        &request.config,
        &request.options,
        &request.kappa_options,
        request.warm_start.as_ref(),
    )
}

pub fn fit_model(request: FitRequest<'_>) -> Result<FitResult, String> {
    match request {
        FitRequest::Standard(request) => fit_standard_model(request).map(FitResult::Standard),
        FitRequest::GaussianLocationScale(request) => {
            fit_gaussian_location_scale_model(request).map(FitResult::GaussianLocationScale)
        }
        FitRequest::BinomialLocationScale(request) => {
            fit_binomial_location_scale_model(request).map(FitResult::BinomialLocationScale)
        }
        FitRequest::SurvivalLocationScale(request) => {
            fit_survival_location_scale_model(request).map(FitResult::SurvivalLocationScale)
        }
        FitRequest::BernoulliMarginalSlope(request) => {
            fit_bernoulli_marginal_slope_model(request).map(FitResult::BernoulliMarginalSlope)
        }
        FitRequest::SurvivalMarginalSlope(request) => {
            fit_survival_marginal_slope_model(request).map(FitResult::SurvivalMarginalSlope)
        }
        FitRequest::TransformationNormal(request) => {
            fit_transformation_normal_model(request).map(FitResult::TransformationNormal)
        }
    }
}

// ---------------------------------------------------------------------------
// High-level formula-to-fit API
// ---------------------------------------------------------------------------

use crate::families::family_meta::{family_to_string, is_binomial_family};
use crate::families::survival_construction::{
    SurvivalBaselineTarget, SurvivalLikelihoodMode, SurvivalTimeWiggleBuild,
    append_zero_tail_columns, build_survival_baseline_offsets, build_survival_time_basis,
    build_survival_timewiggle_from_baseline, build_time_varying_survival_covariate_template,
    normalize_survival_time_pair, parse_survival_baseline_config, parse_survival_distribution,
    parse_survival_likelihood_mode, parse_survival_time_basis_config,
};
use crate::families::survival_location_scale::{
    SurvivalCovariateTermBlockTemplate, TimeBlockInput, TimeWiggleBlockInput,
    residual_distribution_inverse_link,
};
use crate::inference::data::EncodedDataset as Dataset;
use crate::inference::formula_dsl::{
    LinkChoice, ParsedFormula, effectivelinkwiggle_formulaspec, parse_formula, parse_link_choice,
    parse_surv_response,
};
use crate::term_builder::{build_termspec, enable_scale_dimensions};

/// Non-formula configuration for model fitting. All fields have sensible defaults.
#[derive(Clone, Debug)]
pub struct FitConfig {
    /// Family: "gaussian", "binomial", "poisson", "gamma", or None for auto-detect.
    pub family: Option<String>,
    /// Link: "identity", "logit", "probit", "cloglog", "sas", "beta-logistic", or None.
    pub link: Option<String>,
    /// Whether to use flexible (wiggle-augmented) link.
    pub flexible_link: bool,

    // Survival-specific
    /// Baseline target: "linear", "weibull", "gompertz", "gompertz-makeham".
    pub baseline_target: String,
    pub baseline_scale: Option<f64>,
    pub baseline_shape: Option<f64>,
    pub baseline_rate: Option<f64>,
    pub baseline_makeham: Option<f64>,
    /// Time basis: "ispline" or "none".
    pub time_basis: String,
    pub time_degree: usize,
    pub time_num_internal_knots: usize,
    pub time_smooth_lambda: f64,
    /// Survival likelihood mode: "location-scale", "transformation", "weibull", "marginal-slope".
    pub survival_likelihood: String,
    /// Residual distribution: "gaussian", "logistic", "gumbel".
    pub survival_distribution: String,
    pub threshold_time_k: Option<usize>,
    pub threshold_time_degree: usize,
    pub sigma_time_k: Option<usize>,
    pub sigma_time_degree: usize,

    // Location-scale (GAMLSS)
    /// If set, fit a location-scale model with this formula for the noise parameter.
    pub noise_formula: Option<String>,

    // Marginal-slope
    /// Formula for the log-slope model (survival marginal-slope or Bernoulli marginal-slope).
    pub logslope_formula: Option<String>,
    /// Column name for the z (exposure/dose) variable in marginal-slope models.
    pub z_column: Option<String>,

    // Fitting options
    pub scale_dimensions: bool,
    pub ridge_lambda: f64,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            family: None,
            link: None,
            flexible_link: false,
            baseline_target: "linear".into(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "ispline".into(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            survival_likelihood: "location-scale".into(),
            survival_distribution: "gaussian".into(),
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            noise_formula: None,
            logslope_formula: None,
            z_column: None,
            scale_dimensions: false,
            ridge_lambda: 1e-6,
        }
    }
}

/// The result of materializing a formula + config against a dataset.
pub struct MaterializedModel<'a> {
    pub request: FitRequest<'a>,
    pub inference_notes: Vec<String>,
}

/// Parse, materialize, and fit a model in one call.
pub fn fit_from_formula(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
) -> Result<FitResult, String> {
    let mat = materialize(formula, data, config)?;
    fit_model(mat.request)
}

/// Parse a formula, resolve it against a dataset, and produce a ready-to-fit `FitRequest`.
pub fn materialize<'a>(
    formula: &str,
    data: &'a Dataset,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, String> {
    let parsed = parse_formula(formula)?;
    let col_map = build_col_map(data);

    if let Some((entry_col, exit_col, event_col)) = parse_surv_response(&parsed.response)? {
        materialize_survival(
            &parsed, data, &col_map, config, &entry_col, &exit_col, &event_col,
        )
    } else if config.noise_formula.is_some() {
        materialize_location_scale(&parsed, data, &col_map, config)
    } else {
        materialize_standard(&parsed, data, &col_map, config)
    }
}

/// Detect whether a response column is binary (0/1 only).
pub fn is_binary_response(y: ArrayView1<'_, f64>) -> bool {
    if y.is_empty() {
        return false;
    }
    y.iter()
        .all(|v| (*v - 0.0).abs() < 1e-12 || (*v - 1.0).abs() < 1e-12)
}

/// Resolve a family from an optional name, optional link choice, and response data.
pub fn resolve_family(
    family: Option<&str>,
    link_choice: Option<&LinkChoice>,
    y: ArrayView1<'_, f64>,
) -> Result<LikelihoodFamily, String> {
    let explicit = family.and_then(|name| match name.to_ascii_lowercase().as_str() {
        "gaussian" => Some(LikelihoodFamily::GaussianIdentity),
        "binomial" | "binomial-logit" => Some(LikelihoodFamily::BinomialLogit),
        "binomial-probit" => Some(LikelihoodFamily::BinomialProbit),
        "binomial-cloglog" => Some(LikelihoodFamily::BinomialCLogLog),
        "poisson" => Some(LikelihoodFamily::PoissonLog),
        "gamma" => Some(LikelihoodFamily::GammaLog),
        _ => None,
    });

    if let Some(choice) = link_choice {
        let from_link = if choice.mixture_components.is_some() {
            LikelihoodFamily::BinomialMixture
        } else {
            match choice.link {
                LinkFunction::Identity => LikelihoodFamily::GaussianIdentity,
                LinkFunction::Log => {
                    if y.iter()
                        .all(|&yi| yi.is_finite() && yi >= 0.0 && (yi - yi.round()).abs() <= 1e-9)
                    {
                        LikelihoodFamily::PoissonLog
                    } else {
                        LikelihoodFamily::GammaLog
                    }
                }
                LinkFunction::Logit => LikelihoodFamily::BinomialLogit,
                LinkFunction::Probit => LikelihoodFamily::BinomialProbit,
                LinkFunction::CLogLog => LikelihoodFamily::BinomialCLogLog,
                LinkFunction::Sas => LikelihoodFamily::BinomialSas,
                LinkFunction::BetaLogistic => LikelihoodFamily::BinomialBetaLogistic,
            }
        };
        if let Some(explicit_family) = explicit {
            if explicit_family != from_link {
                return Err(format!(
                    "family '{}' conflicts with link",
                    family_to_string(explicit_family)
                ));
            }
        }
        return Ok(from_link);
    }

    if let Some(f) = explicit {
        return Ok(f);
    }

    // Auto-detect
    if is_binary_response(y) {
        Ok(LikelihoodFamily::BinomialLogit)
    } else {
        Ok(LikelihoodFamily::GaussianIdentity)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn build_col_map(data: &Dataset) -> HashMap<String, usize> {
    data.headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect()
}

fn materialize_standard<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, String> {
    let y_col = *col_map
        .get(&parsed.response)
        .ok_or_else(|| format!("response column '{}' not found", parsed.response))?;
    let y = data.values.column(y_col).to_owned();
    let mut inference_notes = Vec::new();

    let link_choice = parse_link_choice(config.link.as_deref(), config.flexible_link)?;
    let family = resolve_family(config.family.as_deref(), link_choice.as_ref(), y.view())?;

    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(parsed.linkwiggle.as_ref(), link_choice.as_ref());

    let mut spec = build_termspec(&parsed.terms, data, col_map, &mut inference_notes)?;
    if config.scale_dimensions {
        enable_scale_dimensions(&mut spec);
    }

    let n = data.values.nrows();
    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let options = FitOptions {
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        max_iter: 200,
        tol: 1e-7,
        nullspace_dims: vec![],
        linear_constraints: None,
        adaptive_regularization: None,
        penalty_shrinkage_floor: Some(1e-6),
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();

    let wiggle = effective_linkwiggle.as_ref().and_then(|cfg| {
        if !is_binomial_family(family) {
            return None;
        }
        let link_kind = link_choice
            .as_ref()
            .map(|c| InverseLink::Standard(c.link))
            .unwrap_or(InverseLink::Standard(LinkFunction::Logit));
        Some(StandardBinomialWiggleConfig {
            link_kind,
            wiggle: LinkWiggleConfig {
                degree: cfg.degree,
                num_internal_knots: cfg.num_internal_knots,
                penalty_orders: cfg.penalty_orders.clone(),
                double_penalty: cfg.double_penalty,
            },
        })
    });

    Ok(MaterializedModel {
        request: FitRequest::Standard(StandardFitRequest {
            data: data.values.view(),
            y,
            weights,
            offset,
            spec,
            family,
            options,
            kappa_options,
            wiggle,
            wiggle_options: None,
        }),
        inference_notes,
    })
}

fn materialize_survival<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
    entry_col: &str,
    exit_col: &str,
    event_col: &str,
) -> Result<MaterializedModel<'a>, String> {
    let mut inference_notes = Vec::new();

    // Extract columns
    let entry_idx = *col_map
        .get(entry_col)
        .ok_or_else(|| format!("entry column '{entry_col}' not found"))?;
    let exit_idx = *col_map
        .get(exit_col)
        .ok_or_else(|| format!("exit column '{exit_col}' not found"))?;
    let event_idx = *col_map
        .get(event_col)
        .ok_or_else(|| format!("event column '{event_col}' not found"))?;
    let n = data.values.nrows();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    let event = data.values.column(event_idx).to_owned();
    for i in 0..n {
        let (e, x) = normalize_survival_time_pair(
            data.values[[i, entry_idx]],
            data.values[[i, exit_idx]],
            i,
        )?;
        age_entry[i] = e;
        age_exit[i] = x;
    }

    // Parse survival config
    let survival_mode = parse_survival_likelihood_mode(&config.survival_likelihood)?;
    if matches!(
        survival_mode,
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
    ) {
        return Err(format!(
            "survival likelihood '{}' is not yet supported through the unified API; \
             use 'location-scale' or 'marginal-slope'. For transformation/weibull, use FitRequest directly.",
            config.survival_likelihood
        ));
    }
    let baseline_cfg = parse_survival_baseline_config(
        &config.baseline_target,
        config.baseline_scale,
        config.baseline_shape,
        config.baseline_rate,
        config.baseline_makeham,
    )?;
    let time_cfg = parse_survival_time_basis_config(
        &config.time_basis,
        config.time_degree,
        config.time_num_internal_knots,
        config.time_smooth_lambda,
    )?;

    // Build time basis
    let time_build = build_survival_time_basis(
        &age_entry,
        &age_exit,
        time_cfg,
        Some((config.time_num_internal_knots, config.time_smooth_lambda)),
    )?;

    // Build baseline offsets
    let (eta_offset_entry, eta_offset_exit, derivative_offset_exit) =
        build_survival_baseline_offsets(&age_entry, &age_exit, &baseline_cfg)?;

    // Time wiggles
    let effective_timewiggle = parsed.timewiggle.clone();
    if effective_timewiggle.is_some() && baseline_cfg.target == SurvivalBaselineTarget::Linear {
        return Err(
            "timewiggle requires a non-linear scalar survival baseline target; \
             use baseline_target weibull, gompertz, or gompertz-makeham"
                .to_string(),
        );
    }
    let mut time_design_entry = time_build.x_entry_time.clone();
    let mut time_design_exit = time_build.x_exit_time.clone();
    let mut time_design_derivative = time_build.x_derivative_time.clone();
    let mut time_penalties = time_build.penalties.clone();
    let mut time_nullspace_dims = time_build.nullspace_dims.clone();
    let mut timewiggle_build: Option<SurvivalTimeWiggleBuild> = None;
    let mut timewiggle_block: Option<TimeWiggleBlockInput> = None;

    if let Some(tw_cfg) = effective_timewiggle.as_ref() {
        let tw = build_survival_timewiggle_from_baseline(
            &eta_offset_entry,
            &eta_offset_exit,
            &derivative_offset_exit,
            tw_cfg,
        )?;
        let p_base = time_design_exit.ncols();
        append_zero_tail_columns(
            &mut time_design_entry,
            &mut time_design_exit,
            &mut time_design_derivative,
            tw.ncols,
        );
        for (idx, p) in tw.penalties.iter().enumerate() {
            let mut embedded = Array2::<f64>::zeros((p_base + tw.ncols, p_base + tw.ncols));
            embedded
                .slice_mut(s![p_base..p_base + tw.ncols, p_base..p_base + tw.ncols])
                .assign(p);
            time_penalties.push(embedded);
            time_nullspace_dims.push(tw.nullspace_dims.get(idx).copied().unwrap_or(0));
        }
        timewiggle_block = Some(TimeWiggleBlockInput {
            knots: tw.knots.clone(),
            degree: tw.degree,
            ncols: tw.ncols,
        });
        timewiggle_build = Some(tw);
    }

    // Build covariate spec
    let mut termspec = build_termspec(&parsed.terms, data, col_map, &mut inference_notes)?;
    if config.scale_dimensions {
        enable_scale_dimensions(&mut termspec);
    }

    // Resolve survival link
    let residual_dist = parse_survival_distribution(&config.survival_distribution)?;
    let survival_inverse_link = residual_distribution_inverse_link(residual_dist);

    // Link wiggle config
    let link_choice = parse_link_choice(config.link.as_deref(), config.flexible_link)?;
    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(parsed.linkwiggle.as_ref(), link_choice.as_ref());

    // Time-varying covariate templates
    let threshold_template = if let Some(k) = config.threshold_time_k {
        build_time_varying_survival_covariate_template(
            &age_entry,
            &age_exit,
            k,
            config.threshold_time_degree,
            "threshold",
        )?
    } else {
        SurvivalCovariateTermBlockTemplate::Static
    };
    let log_sigma_template = if let Some(k) = config.sigma_time_k {
        build_time_varying_survival_covariate_template(
            &age_entry,
            &age_exit,
            k,
            config.sigma_time_degree,
            "sigma",
        )?
    } else {
        SurvivalCovariateTermBlockTemplate::Static
    };

    // Build the noise spec for location-scale
    let log_sigmaspec = if let Some(noise) = config.noise_formula.as_deref() {
        let noise_parsed = parse_formula(&format!("{} ~ {noise}", parsed.response))?;
        build_termspec(&noise_parsed.terms, data, col_map, &mut inference_notes)?
    } else {
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        }
    };

    // Initial time lambdas
    let time_initial_log_lambdas = if !time_penalties.is_empty() {
        Some(Array1::from_elem(
            time_penalties.len(),
            config.time_smooth_lambda.ln(),
        ))
    } else {
        None
    };

    // Assemble the time block (shared between LocationScale and MarginalSlope)
    let time_p =
        time_build.x_exit_time.ncols() + timewiggle_build.as_ref().map_or(0, |tw| tw.ncols);
    let time_block = TimeBlockInput {
        design_entry: time_design_entry,
        design_exit: time_design_exit,
        design_derivative_exit: time_design_derivative.clone(),
        offset_entry: eta_offset_entry,
        offset_exit: eta_offset_exit,
        derivative_offset_exit: derivative_offset_exit.clone(),
        structural_monotonicity: true,
        penalties: time_penalties,
        nullspace_dims: time_nullspace_dims,
        initial_log_lambdas: time_initial_log_lambdas,
        initial_beta: Some(Array1::from_elem(time_p, 1e-4)),
    };

    let kappa_options = SpatialLengthScaleOptimizationOptions::default();

    match survival_mode {
        SurvivalLikelihoodMode::LocationScale => {
            let spec = SurvivalLocationScaleTermSpec {
                age_entry: age_entry.clone(),
                age_exit: age_exit.clone(),
                event_target: event,
                weights: Array1::ones(n),
                inverse_link: survival_inverse_link,
                derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
                time_anchor: None,
                max_iter: 200,
                tol: 1e-7,
                time_block,
                thresholdspec: termspec,
                log_sigmaspec,
                threshold_template,
                log_sigma_template,
                timewiggle_block,
                linkwiggle_block: None,
            };

            let optimize_inverse_link =
                survival_inverse_link_has_free_parameters(&spec.inverse_link);

            Ok(MaterializedModel {
                request: FitRequest::SurvivalLocationScale(SurvivalLocationScaleFitRequest {
                    data: data.values.view(),
                    spec,
                    wiggle: effective_linkwiggle.map(|cfg| LinkWiggleConfig {
                        degree: cfg.degree,
                        num_internal_knots: cfg.num_internal_knots,
                        penalty_orders: cfg.penalty_orders,
                        double_penalty: cfg.double_penalty,
                    }),
                    kappa_options,
                    optimize_inverse_link,
                }),
                inference_notes,
            })
        }
        SurvivalLikelihoodMode::MarginalSlope => {
            if timewiggle_build.is_some() {
                return Err(
                    "timewiggle is only implemented for survival-likelihood=location-scale in the exact dynamic path"
                        .to_string(),
                );
            }
            let z_col_name = config.z_column.as_deref().ok_or_else(|| {
                "marginal-slope survival requires z_column in FitConfig".to_string()
            })?;
            let z_idx = *col_map
                .get(z_col_name)
                .ok_or_else(|| format!("z column '{z_col_name}' not found"))?;
            let z = data.values.column(z_idx).to_owned();
            let marginalspec = termspec.clone();
            let logslopespec = if let Some(ls_formula) = config.logslope_formula.as_deref() {
                let ls_parsed = parse_formula(&format!("{} ~ {ls_formula}", parsed.response))?;
                build_termspec(&ls_parsed.terms, data, col_map, &mut inference_notes)?
            } else {
                termspec
            };
            let spec = SurvivalMarginalSlopeTermSpec {
                age_entry: age_entry.clone(),
                age_exit: age_exit.clone(),
                event_target: event,
                weights: Array1::ones(n),
                z,
                marginalspec,
                derivative_guard: DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD,
                time_block,
                logslopespec,
            };

            Ok(MaterializedModel {
                request: FitRequest::SurvivalMarginalSlope(SurvivalMarginalSlopeFitRequest {
                    data: data.values.view(),
                    spec,
                    options: BlockwiseFitOptions::default(),
                    kappa_options,
                }),
                inference_notes,
            })
        }
        // Transformation and Weibull are rejected earlier in this function.
        _ => unreachable!(),
    }
}

fn materialize_location_scale<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, String> {
    let y_col = *col_map
        .get(&parsed.response)
        .ok_or_else(|| format!("response column '{}' not found", parsed.response))?;
    let y = data.values.column(y_col).to_owned();
    let mut inference_notes = Vec::new();

    let noise_formula = config
        .noise_formula
        .as_deref()
        .ok_or_else(|| "noise_formula is required for location-scale models".to_string())?;
    let noise_parsed = parse_formula(&format!("{} ~ {noise_formula}", parsed.response))?;

    let link_choice = parse_link_choice(config.link.as_deref(), config.flexible_link)?;
    let family = resolve_family(config.family.as_deref(), link_choice.as_ref(), y.view())?;

    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(parsed.linkwiggle.as_ref(), link_choice.as_ref());

    let mut meanspec = build_termspec(&parsed.terms, data, col_map, &mut inference_notes)?;
    let mut log_sigmaspec =
        build_termspec(&noise_parsed.terms, data, col_map, &mut inference_notes)?;
    if config.scale_dimensions {
        enable_scale_dimensions(&mut meanspec);
        enable_scale_dimensions(&mut log_sigmaspec);
    }

    let weights = Array1::ones(data.values.nrows());
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();
    let options = BlockwiseFitOptions::default();

    let wiggle_cfg = effective_linkwiggle.map(|cfg| LinkWiggleConfig {
        degree: cfg.degree,
        num_internal_knots: cfg.num_internal_knots,
        penalty_orders: cfg.penalty_orders,
        double_penalty: cfg.double_penalty,
    });

    if is_binomial_family(family) {
        let link_kind = link_choice
            .as_ref()
            .map(|c| InverseLink::Standard(c.link))
            .unwrap_or(InverseLink::Standard(LinkFunction::Logit));
        Ok(MaterializedModel {
            request: FitRequest::BinomialLocationScale(BinomialLocationScaleFitRequest {
                data: data.values.view(),
                spec: BinomialLocationScaleTermSpec {
                    y,
                    weights,
                    link_kind,
                    thresholdspec: meanspec,
                    log_sigmaspec,
                },
                wiggle: wiggle_cfg,
                options,
                kappa_options,
            }),
            inference_notes,
        })
    } else {
        Ok(MaterializedModel {
            request: FitRequest::GaussianLocationScale(GaussianLocationScaleFitRequest {
                data: data.values.view(),
                spec: GaussianLocationScaleTermSpec {
                    y,
                    weights,
                    meanspec,
                    log_sigmaspec,
                },
                wiggle: wiggle_cfg,
                options,
                kappa_options,
            }),
            inference_notes,
        })
    }
}

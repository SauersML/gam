use crate::custom_family::BlockwiseFitOptions;
use crate::estimate::{EstimationError, FitOptions, FittedLinkState, UnifiedFitResult};
use crate::families::bernoulli_marginal_slope::{
    fit_bernoulli_marginal_slope_terms, BernoulliMarginalSlopeFitResult,
    BernoulliMarginalSlopeTermSpec,
};
use crate::families::gamlss::{
    fit_binomial_location_scale_terms, fit_binomial_location_scale_terms_with_selected_wiggle,
    fit_binomial_mean_wiggle_terms_with_selected_basis, fit_gaussian_location_scale_terms,
    fit_gaussian_location_scale_terms_with_selected_wiggle,
    select_binomial_location_scale_link_wiggle_basis_from_pilot,
    select_binomial_mean_link_wiggle_basis_from_pilot,
    select_gaussian_location_scale_link_wiggle_basis_from_pilot, BinomialLocationScaleFitResult,
    BinomialLocationScaleTermSpec, BlockwiseTermFitResult, BlockwiseTermFitResultParts,
    GaussianLocationScaleFitResult, GaussianLocationScaleTermSpec, WiggleBlockConfig,
};
use crate::families::survival_location_scale::{
    fit_survival_location_scale_terms, fit_survival_location_scale_terms_with_selected_wiggle,
    select_survival_link_wiggle_basis_from_pilot, SurvivalLocationScaleTermFitResult,
    SurvivalLocationScaleTermSpec,
};
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
use crate::smooth::{
    fit_term_collectionwith_spatial_length_scale_optimization, AdaptiveRegularizationDiagnostics,
    SpatialLengthScaleOptimizationOptions, TermCollectionDesign, TermCollectionSpec,
};
use crate::solver::outer_strategy::{
    ClosureObjective, Derivative, OuterCapability, OuterConfig, OuterEval,
};
use crate::types::{InverseLink, LikelihoodFamily, MixtureLinkSpec, SasLinkSpec};
use ndarray::{Array1, ArrayView2};

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

pub struct BernoulliMarginalSlopeFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: BernoulliMarginalSlopeTermSpec,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}

pub enum FitRequest<'a> {
    Standard(StandardFitRequest<'a>),
    GaussianLocationScale(GaussianLocationScaleFitRequest<'a>),
    BinomialLocationScale(BinomialLocationScaleFitRequest<'a>),
    SurvivalLocationScale(SurvivalLocationScaleFitRequest<'a>),
    BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest<'a>),
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

pub enum FitResult {
    Standard(StandardFitResult),
    GaussianLocationScale(GaussianLocationScaleFitResult),
    BinomialLocationScale(BinomialLocationScaleFitResult),
    SurvivalLocationScale(SurvivalLocationScaleFitResult),
    BernoulliMarginalSlope(BernoulliMarginalSlopeFitResult),
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
    fn fit_survival_with_link(
        data: ArrayView2<'_, f64>,
        spec: SurvivalLocationScaleTermSpec,
        wiggle: Option<LinkWiggleConfig>,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<
        (
            SurvivalLocationScaleTermFitResult,
            Option<Array1<f64>>,
            Option<usize>,
        ),
        String,
    > {
        let mut wiggle_knots = None;
        let mut wiggle_degree = None;

        let fit = if let Some(wiggle) = wiggle {
            ensure_joint_wiggle_supported(&spec.inverse_link, "survival link wiggle")?;
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

        Ok((fit, wiggle_knots, wiggle_degree))
    }

    fn optimize_survival_inverse_link(
        data: ArrayView2<'_, f64>,
        spec: &SurvivalLocationScaleTermSpec,
        kappa_options: &SpatialLengthScaleOptimizationOptions,
    ) -> Result<InverseLink, String> {
        let optimize_link_parameters = |init: Array1<f64>,
                                        name: &str,
                                        mut objective: Box<
            dyn FnMut(&Array1<f64>) -> Result<f64, EstimationError>,
        >,
                                        recover: Box<
            dyn Fn(&Array1<f64>) -> Option<InverseLink>,
        >|
         -> Option<InverseLink> {
            let dim = init.len();
            let mut seed_config = crate::seeding::SeedConfig::default();
            seed_config.max_seeds = 8;
            seed_config.screening_budget = 3;
            seed_config.risk_profile = crate::seeding::SeedRiskProfile::Survival;
            let outer_config = OuterConfig {
                tolerance: 1e-4,
                max_iter: 30,
                fd_step: 1e-3,
                bounds: None,
                seed_config,
                rho_bound: 30.0,
                heuristic_lambdas: Some(init.to_vec()),
                initial_rho: None,
                fallback_sequence: Vec::new(),
            };
            let mut obj = ClosureObjective {
                state: (),
                cap: OuterCapability {
                    gradient: Derivative::FiniteDifference,
                    hessian: Derivative::Unavailable,
                    n_params: dim,
                    all_penalty_like: false,
                    has_psi_coords: false,
                    barrier_config: None,
                },
                cost_fn: |state: &mut (), rho: &Array1<f64>| {
                    let _ = state;
                    objective(rho)
                },
                eval_fn: |state: &mut (),
                          rho: &Array1<f64>|
                 -> Result<OuterEval, EstimationError> {
                    let _ = state;
                    let _ = rho;
                    Err(EstimationError::InvalidInput(
                        "strategy should use finite-difference gradients for survival link optimization"
                            .to_string(),
                    ))
                },
                reset_fn: None::<fn(&mut ())>,
                efs_fn: None::<
                    fn(
                        &mut (),
                        &Array1<f64>,
                    )
                        -> Result<crate::solver::outer_strategy::EfsEval, EstimationError>,
                >,
            };
            match crate::solver::outer_strategy::run_outer(&mut obj, &outer_config, name) {
                Ok(result) => {
                    if let Some(link) = recover(&result.rho) {
                        eprintln!(
                            "[survival link opt] optimized {name} params: dim={} iters={} converged={} finalobj={:.6e}",
                            result.rho.len(),
                            result.iterations,
                            result.converged,
                            result.final_value
                        );
                        Some(link)
                    } else {
                        None
                    }
                }
                Err(err) => {
                    eprintln!(
                        "[survival link opt] {name} optimization failed; using initial params: {err}"
                    );
                    None
                }
            }
        };

        match spec.inverse_link.clone() {
            InverseLink::Sas(state0) => {
                let init = Array1::from_vec(vec![state0.epsilon, state0.log_delta]);
                Ok(optimize_link_parameters(
                    init,
                    "SAS",
                    Box::new(|theta: &Array1<f64>| {
                        let state = state_from_sasspec(SasLinkSpec {
                            initial_epsilon: theta[0],
                            initial_log_delta: theta[1],
                        })
                        .map_err(EstimationError::InvalidInput)?;
                        let mut spec_at_theta = spec.clone();
                        spec_at_theta.inverse_link = InverseLink::Sas(state);
                        Ok(
                            fit_survival_with_link(data, spec_at_theta, None, kappa_options)
                                .map_err(EstimationError::InvalidInput)?
                                .0
                                .fit
                                .reml_score,
                        )
                    }),
                    Box::new(|rho| {
                        state_from_sasspec(SasLinkSpec {
                            initial_epsilon: rho[0],
                            initial_log_delta: rho[1],
                        })
                        .ok()
                        .map(InverseLink::Sas)
                    }),
                )
                .unwrap_or(spec.inverse_link.clone()))
            }
            InverseLink::BetaLogistic(state0) => {
                let init = Array1::from_vec(vec![state0.epsilon, state0.log_delta]);
                Ok(optimize_link_parameters(
                    init,
                    "BetaLogistic",
                    Box::new(|theta: &Array1<f64>| {
                        let state = state_from_beta_logisticspec(SasLinkSpec {
                            initial_epsilon: theta[0],
                            initial_log_delta: theta[1],
                        })
                        .map_err(EstimationError::InvalidInput)?;
                        let mut spec_at_theta = spec.clone();
                        spec_at_theta.inverse_link = InverseLink::BetaLogistic(state);
                        Ok(
                            fit_survival_with_link(data, spec_at_theta, None, kappa_options)
                                .map_err(EstimationError::InvalidInput)?
                                .0
                                .fit
                                .reml_score,
                        )
                    }),
                    Box::new(|rho| {
                        state_from_beta_logisticspec(SasLinkSpec {
                            initial_epsilon: rho[0],
                            initial_log_delta: rho[1],
                        })
                        .ok()
                        .map(InverseLink::BetaLogistic)
                    }),
                )
                .unwrap_or(spec.inverse_link.clone()))
            }
            InverseLink::Mixture(state0) if !state0.rho.is_empty() => {
                let components = state0.components.clone();
                let components_recover = components.clone();
                Ok(optimize_link_parameters(
                    state0.rho.clone(),
                    "mixture",
                    Box::new(move |rho: &Array1<f64>| {
                        let state = state_fromspec(&MixtureLinkSpec {
                            components: components.clone(),
                            initial_rho: rho.clone(),
                        })
                        .map_err(EstimationError::InvalidInput)?;
                        let mut spec_at_theta = spec.clone();
                        spec_at_theta.inverse_link = InverseLink::Mixture(state);
                        Ok(
                            fit_survival_with_link(data, spec_at_theta, None, kappa_options)
                                .map_err(EstimationError::InvalidInput)?
                                .0
                                .fit
                                .reml_score,
                        )
                    }),
                    Box::new(move |rho| {
                        state_fromspec(&MixtureLinkSpec {
                            components: components_recover.clone(),
                            initial_rho: rho.to_owned(),
                        })
                        .ok()
                        .map(InverseLink::Mixture)
                    }),
                )
                .unwrap_or(spec.inverse_link.clone()))
            }
            _ => Ok(spec.inverse_link.clone()),
        }
    }

    let inverse_link = if request.optimize_inverse_link {
        optimize_survival_inverse_link(request.data, &request.spec, &request.kappa_options)?
    } else {
        request.spec.inverse_link.clone()
    };
    let mut spec = request.spec;
    spec.inverse_link = inverse_link.clone();
    let (fit, wiggle_knots, wiggle_degree) =
        fit_survival_with_link(request.data, spec, request.wiggle, &request.kappa_options)?;

    Ok(SurvivalLocationScaleFitResult {
        fit,
        inverse_link,
        wiggle_knots,
        wiggle_degree,
    })
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
    }
}

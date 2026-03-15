use crate::basis::create_difference_penalty_matrix;
use crate::custom_family::BlockwiseFitOptions;
use crate::estimate::{EstimationError, FitOptions, FittedLinkState, UnifiedFitResult};
use crate::families::gamlss::{
    BinomialLocationScaleTermSpec, BinomialLocationScaleWorkflowResult, BlockwiseTermFitResult,
    BlockwiseTermFitResultParts, GaussianLocationScaleTermSpec,
    GaussianLocationScaleWorkflowResult, ParameterBlockInput, WiggleBlockConfig,
    buildwiggle_block_input_from_seed, fit_binomial_location_scale_terms,
    fit_binomial_location_scalewiggle_terms_auto, fit_binomial_mean_wiggle_terms_auto_from_pilot,
    fit_gaussian_location_scale_terms, fit_gaussian_location_scalewiggle_terms_auto,
};
use crate::families::survival_location_scale::{
    LinkWiggleBlockInput, SurvivalLocationScaleTermFitResult, SurvivalLocationScaleTermSpec,
    fit_survival_location_scale_terms,
};
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
use crate::smooth::{
    AdaptiveRegularizationDiagnostics, SpatialLengthScaleOptimizationOptions, TermCollectionDesign,
    TermCollectionSpec, fit_term_collectionwith_spatial_length_scale_optimization,
};
use crate::solver::strategy::{
    ClosureObjective, Derivative, OuterCapability, OuterConfig, OuterEval,
};
use crate::types::{InverseLink, LikelihoodFamily, MixtureLinkSpec, SasLinkSpec};
use ndarray::{Array1, ArrayView2};

#[derive(Clone, Debug)]
pub struct LinkWiggleWorkflowConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_orders: Vec<usize>,
    pub double_penalty: bool,
}

#[derive(Clone, Debug)]
pub struct StandardBinomialWiggleWorkflowConfig {
    pub link_kind: InverseLink,
    pub wiggle: LinkWiggleWorkflowConfig,
}

pub struct StandardFitWorkflowRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub offset: Array1<f64>,
    pub spec: TermCollectionSpec,
    pub family: LikelihoodFamily,
    pub options: FitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
    pub wiggle: Option<StandardBinomialWiggleWorkflowConfig>,
    pub wiggle_options: Option<BlockwiseFitOptions>,
}

pub struct GaussianLocationScaleWorkflowRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: GaussianLocationScaleTermSpec,
    pub wiggle: Option<LinkWiggleWorkflowConfig>,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}

pub struct BinomialLocationScaleWorkflowRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: BinomialLocationScaleTermSpec,
    pub wiggle: Option<LinkWiggleWorkflowConfig>,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}

pub struct SurvivalLocationScaleWorkflowRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: SurvivalLocationScaleTermSpec,
    pub wiggle: Option<LinkWiggleWorkflowConfig>,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
    pub optimize_inverse_link: bool,
}

pub enum FitModelRequest<'a> {
    Standard(StandardFitWorkflowRequest<'a>),
    GaussianLocationScale(GaussianLocationScaleWorkflowRequest<'a>),
    BinomialLocationScale(BinomialLocationScaleWorkflowRequest<'a>),
    SurvivalLocationScale(SurvivalLocationScaleWorkflowRequest<'a>),
}

pub struct StandardFitWorkflowResult {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub resolvedspec: TermCollectionSpec,
    pub adaptive_diagnostics: Option<AdaptiveRegularizationDiagnostics>,
    pub saved_link_state: FittedLinkState,
    pub wiggle_knots: Option<Array1<f64>>,
    pub wiggle_degree: Option<usize>,
    pub betawiggle: Option<Vec<f64>>,
}

pub struct SurvivalLocationScaleWorkflowResult {
    pub fit: SurvivalLocationScaleTermFitResult,
    pub inverse_link: InverseLink,
    pub wiggle_knots: Option<Array1<f64>>,
    pub wiggle_degree: Option<usize>,
}

pub enum FitModelResult {
    Standard(StandardFitWorkflowResult),
    GaussianLocationScale(GaussianLocationScaleWorkflowResult),
    BinomialLocationScale(BinomialLocationScaleWorkflowResult),
    SurvivalLocationScale(SurvivalLocationScaleWorkflowResult),
}

fn augment_wiggle_penalties(
    block: &mut ParameterBlockInput,
    penalty_orders: &[usize],
) -> Result<(), String> {
    let p = block.design.ncols();
    if p == 0 {
        return Ok(());
    }
    for &order in penalty_orders {
        if order <= 1 || order >= p {
            continue;
        }
        let penalty =
            create_difference_penalty_matrix(p, order, None).map_err(|e| e.to_string())?;
        block.penalties.push(penalty);
    }
    Ok(())
}

fn resolved_wiggle_inverse_link(
    family: LikelihoodFamily,
    fit: &UnifiedFitResult,
    fallback: &InverseLink,
) -> Result<InverseLink, String> {
    match fit.fitted_link_state(family).map_err(|e| e.to_string())? {
        FittedLinkState::Standard(Some(link)) => Ok(InverseLink::Standard(link)),
        FittedLinkState::Standard(None) => Ok(fallback.clone()),
        FittedLinkState::Sas { state, .. } => Ok(InverseLink::Sas(state)),
        FittedLinkState::BetaLogistic { state, .. } => Ok(InverseLink::BetaLogistic(state)),
        FittedLinkState::Mixture { state, .. } => Ok(InverseLink::Mixture(state)),
    }
}

fn fit_standard_model(
    request: StandardFitWorkflowRequest<'_>,
) -> Result<StandardFitWorkflowResult, String> {
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

    let result = StandardFitWorkflowResult {
        saved_link_state: fitted.fit.fitted_link.clone(),
        fit: fitted.fit,
        design: fitted.design,
        resolvedspec: fitted.resolvedspec,
        adaptive_diagnostics: fitted.adaptive_diagnostics,
        wiggle_knots: None,
        wiggle_degree: None,
        betawiggle: None,
    };

    let Some(wiggle) = request.wiggle else {
        return Ok(result);
    };
    let wiggle_options = request
        .wiggle_options
        .ok_or_else(|| "standard wiggle workflow requires blockwise wiggle options".to_string())?;
    let wiggle_link_kind =
        resolved_wiggle_inverse_link(request.family, &result.fit, &wiggle.link_kind)?;

    let solved = fit_binomial_mean_wiggle_terms_auto_from_pilot(
        request.data,
        &result.resolvedspec,
        &result.design,
        &result.fit,
        &request.y,
        &request.weights,
        wiggle_link_kind,
        WiggleBlockConfig {
            degree: wiggle.wiggle.degree,
            num_internal_knots: wiggle.wiggle.num_internal_knots,
            penalty_order: 2,
            double_penalty: wiggle.wiggle.double_penalty,
        },
        &wiggle.wiggle.penalty_orders,
        &wiggle_options,
        &request.kappa_options,
    )?;

    let betawiggle = solved
        .fit
        .block_states
        .get(1)
        .ok_or_else(|| "standard wiggle fit is missing link-wiggle block".to_string())?
        .beta
        .to_vec();

    Ok(StandardFitWorkflowResult {
        saved_link_state: result.saved_link_state,
        fit: solved.fit,
        design: solved.design,
        resolvedspec: solved.resolvedspec,
        adaptive_diagnostics: None,
        wiggle_knots: Some(solved.wiggle_knots),
        wiggle_degree: Some(solved.wiggle_degree),
        betawiggle: Some(betawiggle),
    })
}

fn fit_gaussian_location_scale_model(
    request: GaussianLocationScaleWorkflowRequest<'_>,
) -> Result<GaussianLocationScaleWorkflowResult, String> {
    if let Some(wiggle_cfg) = request.wiggle {
        let solved = fit_gaussian_location_scalewiggle_terms_auto(
            request.data,
            request.spec,
            WiggleBlockConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_order: 2,
                double_penalty: wiggle_cfg.double_penalty,
            },
            &wiggle_cfg.penalty_orders,
            &request.options,
            &request.kappa_options,
        )?;
        let fit = solved.fit.fit;
        let betawiggle = fit.block_states.get(2).map(|b| b.beta.to_vec());
        Ok(GaussianLocationScaleWorkflowResult {
            fit: BlockwiseTermFitResult::try_from_parts(BlockwiseTermFitResultParts {
                fit,
                meanspec_resolved: solved.fit.meanspec_resolved,
                noisespec_resolved: solved.fit.noisespec_resolved,
                mean_design: solved.fit.mean_design,
                noise_design: solved.fit.noise_design,
            })?,
            wiggle_knots: Some(solved.wiggle_knots),
            wiggle_degree: Some(solved.wiggle_degree),
            betawiggle,
        })
    } else {
        let fit = fit_gaussian_location_scale_terms(
            request.data,
            request.spec,
            &request.options,
            &request.kappa_options,
        )?;
        Ok(GaussianLocationScaleWorkflowResult {
            fit,
            wiggle_knots: None,
            wiggle_degree: None,
            betawiggle: None,
        })
    }
}

fn fit_binomial_location_scale_model(
    request: BinomialLocationScaleWorkflowRequest<'_>,
) -> Result<BinomialLocationScaleWorkflowResult, String> {
    if let Some(wiggle_cfg) = request.wiggle {
        let solved = fit_binomial_location_scalewiggle_terms_auto(
            request.data,
            request.spec,
            WiggleBlockConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_order: 2,
                double_penalty: wiggle_cfg.double_penalty,
            },
            &wiggle_cfg.penalty_orders,
            &request.options,
            &request.kappa_options,
        )?;
        let fit = solved.fit.fit;
        let betawiggle = fit.block_states.get(2).map(|b| b.beta.to_vec());
        Ok(BinomialLocationScaleWorkflowResult {
            fit: BlockwiseTermFitResult::try_from_parts(BlockwiseTermFitResultParts {
                fit,
                meanspec_resolved: solved.fit.meanspec_resolved,
                noisespec_resolved: solved.fit.noisespec_resolved,
                mean_design: solved.fit.mean_design,
                noise_design: solved.fit.noise_design,
            })?,
            wiggle_knots: Some(solved.wiggle_knots),
            wiggle_degree: Some(solved.wiggle_degree),
            betawiggle,
        })
    } else {
        let solved = fit_binomial_location_scale_terms(
            request.data,
            request.spec,
            &request.options,
            &request.kappa_options,
        )?;
        Ok(BinomialLocationScaleWorkflowResult {
            fit: solved,
            wiggle_knots: None,
            wiggle_degree: None,
            betawiggle: None,
        })
    }
}

fn fit_survival_location_scale_model(
    request: SurvivalLocationScaleWorkflowRequest<'_>,
) -> Result<SurvivalLocationScaleWorkflowResult, String> {
    fn fit_survival_with_link(
        data: ArrayView2<'_, f64>,
        spec: SurvivalLocationScaleTermSpec,
        wiggle: Option<LinkWiggleWorkflowConfig>,
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
            let mut pilot_spec = spec.clone();
            pilot_spec.linkwiggle_block = None;
            let pilot = fit_survival_location_scale_terms(data, pilot_spec, kappa_options)?;
            let eta_threshold = pilot
                .threshold_design
                .design
                .dot(&pilot.fit.beta_threshold());
            let eta_log_sigma = pilot
                .log_sigma_design
                .design
                .dot(&pilot.fit.beta_log_sigma());
            let sigma = eta_log_sigma.mapv(f64::exp);
            let q_seed = Array1::from_iter(
                eta_threshold
                    .iter()
                    .zip(sigma.iter())
                    .map(|(&threshold, &scale)| -threshold / scale.max(1e-12)),
            );
            let (mut wiggle_block, knots) = buildwiggle_block_input_from_seed(
                q_seed.view(),
                &WiggleBlockConfig {
                    degree: wiggle.degree,
                    num_internal_knots: wiggle.num_internal_knots,
                    penalty_order: 2,
                    double_penalty: wiggle.double_penalty,
                },
            )?;
            augment_wiggle_penalties(&mut wiggle_block, &wiggle.penalty_orders)?;
            wiggle_knots = Some(knots);
            wiggle_degree = Some(wiggle.degree);
            let inverse_link = spec.inverse_link.clone();
            let mut wiggle_spec = spec;
            wiggle_spec.inverse_link = inverse_link;
            wiggle_spec.linkwiggle_block = Some(LinkWiggleBlockInput {
                design: wiggle_block.design,
                penalties: wiggle_block.penalties,
                initial_log_lambdas: wiggle_block.initial_log_lambdas,
                initial_beta: wiggle_block.initial_beta,
            });
            fit_survival_location_scale_terms(data, wiggle_spec, kappa_options)?
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
                        -> Result<crate::solver::strategy::EfsEval, EstimationError>,
                >,
            };
            match crate::solver::strategy::run_outer(&mut obj, &outer_config, name) {
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

    Ok(SurvivalLocationScaleWorkflowResult {
        fit,
        inverse_link,
        wiggle_knots,
        wiggle_degree,
    })
}

pub fn fit_model(request: FitModelRequest<'_>) -> Result<FitModelResult, String> {
    match request {
        FitModelRequest::Standard(request) => {
            fit_standard_model(request).map(FitModelResult::Standard)
        }
        FitModelRequest::GaussianLocationScale(request) => {
            fit_gaussian_location_scale_model(request).map(FitModelResult::GaussianLocationScale)
        }
        FitModelRequest::BinomialLocationScale(request) => {
            fit_binomial_location_scale_model(request).map(FitModelResult::BinomialLocationScale)
        }
        FitModelRequest::SurvivalLocationScale(request) => {
            fit_survival_location_scale_model(request).map(FitModelResult::SurvivalLocationScale)
        }
    }
}

use crate::custom_family::BlockwiseFitOptions;
use crate::estimate::{EstimationError, FitOptions, FittedLinkState, UnifiedFitResult};
use crate::families::bernoulli_marginal_slope::{
    BernoulliMarginalSlopeFitResult, BernoulliMarginalSlopeTermSpec, DeviationBlockConfig,
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
    select_gaussian_location_scale_link_wiggle_basis_from_pilot, split_wiggle_penalty_orders,
};
use crate::families::latent_survival::{
    LatentBinaryTermFitResult, LatentBinaryTermSpec, LatentSurvivalTermFitResult,
    LatentSurvivalTermSpec, fit_latent_binary_terms, fit_latent_survival_terms,
    latent_hazard_loading,
};
use crate::families::lognormal_kernel::FrailtySpec;
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
use crate::types::{
    InverseLink, LatentCLogLogState, LikelihoodFamily, LinkFunction, MixtureLinkSpec, SasLinkSpec,
};
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
        InverseLink::LatentCLogLog(_) | InverseLink::Standard(_) => false,
    }
}

fn recover_converged_survival_inverse_link<R>(
    result: crate::solver::outer_strategy::OuterResult,
    context: &str,
    recover: R,
) -> Result<InverseLink, String>
where
    R: FnOnce(&Array1<f64>) -> Option<InverseLink>,
{
    if !result.converged {
        return Err(format!(
            "{context} did not converge after {} iterations (final_objective={:.6e}, final_grad_norm={:.3e})",
            result.iterations, result.final_value, result.final_grad_norm
        ));
    }
    recover(&result.rho).ok_or_else(|| {
        format!(
            "{context} produced an invalid inverse-link state at rho={:?}",
            result.rho.to_vec()
        )
    })
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

pub struct LatentSurvivalFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: LatentSurvivalTermSpec,
    pub frailty: FrailtySpec,
    pub options: BlockwiseFitOptions,
}

pub struct LatentBinaryFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: LatentBinaryTermSpec,
    pub frailty: FrailtySpec,
    pub options: BlockwiseFitOptions,
}

pub struct TransformationNormalFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub response: Array1<f64>,
    pub weights: Array1<f64>,
    pub offset: Array1<f64>,
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
    LatentSurvival(LatentSurvivalFitRequest<'a>),
    LatentBinary(LatentBinaryFitRequest<'a>),
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
    LatentSurvival(LatentSurvivalTermFitResult),
    LatentBinary(LatentBinaryTermFitResult),
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
        FittedLinkState::LatentCLogLog { state } => InverseLink::LatentCLogLog(state),
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
        | InverseLink::LatentCLogLog(_)
        | InverseLink::Sas(_)
        | InverseLink::BetaLogistic(_)
        | InverseLink::Mixture(_) => Err(format!(
            "{context} does not support latent-cloglog, SAS, BetaLogistic, or Mixture links; wiggle is only available for jointly fitted standard links"
        )),
        InverseLink::Standard(_) => Ok(()),
    }
}

fn deviation_block_config_from_formula_linkwiggle(
    wiggle: &LinkWiggleFormulaSpec,
) -> DeviationBlockConfig {
    let (penalty_order, penalty_orders) = split_wiggle_penalty_orders(2, &wiggle.penalty_orders);
    DeviationBlockConfig {
        degree: wiggle.degree,
        num_internal_knots: wiggle.num_internal_knots,
        penalty_order,
        penalty_orders,
        double_penalty: wiggle.double_penalty,
        monotonicity_eps: 1e-4,
    }
}

struct MarginalSlopeDeviationRouting {
    score_warp: Option<DeviationBlockConfig>,
    link_dev: Option<DeviationBlockConfig>,
}

fn route_marginal_slope_deviation_blocks(
    main_linkwiggle: Option<&LinkWiggleFormulaSpec>,
    logslope_linkwiggle: Option<&LinkWiggleFormulaSpec>,
) -> MarginalSlopeDeviationRouting {
    MarginalSlopeDeviationRouting {
        score_warp: logslope_linkwiggle.map(deviation_block_config_from_formula_linkwiggle),
        link_dev: main_linkwiggle.map(deviation_block_config_from_formula_linkwiggle),
    }
}

fn fixed_gaussian_shift_frailty_from_spec(
    frailty: &FrailtySpec,
    context: &str,
) -> Result<FrailtySpec, String> {
    match frailty {
        FrailtySpec::None => Ok(FrailtySpec::None),
        FrailtySpec::GaussianShift {
            sigma_fixed: Some(sigma),
        } => Ok(FrailtySpec::GaussianShift {
            sigma_fixed: Some(*sigma),
        }),
        FrailtySpec::GaussianShift { sigma_fixed: None } => Err(format!(
            "{context} currently requires a fixed GaussianShift sigma"
        )),
        FrailtySpec::HazardMultiplier { .. } => Err(format!(
            "{context} requires FrailtySpec::GaussianShift or no frailty"
        )),
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
                mean_offset: request.spec.mean_offset.clone(),
                log_sigma_offset: request.spec.log_sigma_offset.clone(),
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
                threshold_offset: request.spec.threshold_offset.clone(),
                log_sigma_offset: request.spec.log_sigma_offset.clone(),
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
            ensure_joint_wiggle_supported(&inverse_link, "survival link wiggle")?;
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
        fn optimize_link_parameters<F, R>(
            data: ArrayView2<'_, f64>,
            spec: &SurvivalLocationScaleTermSpec,
            kappa_options: &SpatialLengthScaleOptimizationOptions,
            init: Array1<f64>,
            name: &str,
            final_wiggle: Option<LinkWiggleConfig>,
            objective: F,
            recover: R,
        ) -> Result<SurvivalLocationScaleProfile, String>
        where
            F: FnMut(&Array1<f64>) -> Result<f64, EstimationError>,
            R: Fn(&Array1<f64>) -> Option<InverseLink>,
        {
            use crate::solver::outer_strategy::{OuterProblem, SolverClass};
            let dim = init.len();
            // Inverse-link parameters (SAS epsilon/log_delta, BetaLogistic shape,
            // Mixture rho) have no analytic ∂LAML/∂θ_link; route through the
            // gated gradient-free CompassSearch variant rather than BFGS. Box
            // bounds keep line-search probes inside a physically admissible
            // region (|epsilon|, |log_delta| ≤ 6 gives the SAS link a finite
            // range on both tails).
            let lower = init.mapv(|v| v - 6.0);
            let upper = init.mapv(|v| v + 6.0);
            let problem = OuterProblem::new(dim)
                .with_solver_class(SolverClass::AuxiliaryGradientFree)
                .with_tolerance(1e-4)
                .with_max_iter(240)
                .with_bounds(lower, upper)
                .with_heuristic_lambdas(init.to_vec());
            let context = format!("survival inverse-link optimization ({name}, dim={dim})");
            let mut obj = problem.build_objective(
                objective,
                |f: &mut F, rho: &ndarray::Array1<f64>| f(rho),
                |_: &mut F, _: &ndarray::Array1<f64>| {
                    Err(EstimationError::InvalidInput(
                        "inverse-link aux optimizer: CompassSearch dispatch only \
                         calls eval_cost; eval(gradient) is unreachable by \
                         construction"
                            .to_string(),
                    ))
                },
                None::<fn(&mut F)>,
                None::<
                    fn(
                        &mut F,
                        &ndarray::Array1<f64>,
                    )
                        -> Result<crate::solver::outer_strategy::EfsEval, EstimationError>,
                >,
            );
            let result = problem
                .run(&mut obj, &context)
                .map_err(|err| format!("{context} failed: {err}"))?;
            let link = recover_converged_survival_inverse_link(result, &context, recover)?;
            profile_survival_location_scale_with_inverse_link(
                data,
                spec,
                link,
                final_wiggle,
                kappa_options,
            )
            .map_err(|err| format!("{context} final profiling failed: {err}"))
        }

        match spec.inverse_link.clone() {
            InverseLink::Sas(state0) => {
                let init = Array1::from_vec(vec![state0.epsilon, state0.log_delta]);
                let wiggle_cfg = wiggle.clone();
                optimize_link_parameters(
                    data,
                    spec,
                    kappa_options,
                    init,
                    "SAS",
                    wiggle.clone(),
                    |theta: &Array1<f64>| {
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
                    },
                    |rho| {
                        state_from_sasspec(SasLinkSpec {
                            initial_epsilon: rho[0],
                            initial_log_delta: rho[1],
                        })
                        .ok()
                        .map(InverseLink::Sas)
                    },
                )
            }
            InverseLink::BetaLogistic(state0) => {
                let init = Array1::from_vec(vec![state0.epsilon, state0.log_delta]);
                let wiggle_cfg = wiggle.clone();
                optimize_link_parameters(
                    data,
                    spec,
                    kappa_options,
                    init,
                    "BetaLogistic",
                    wiggle.clone(),
                    |theta: &Array1<f64>| {
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
                    },
                    |rho| {
                        state_from_beta_logisticspec(SasLinkSpec {
                            initial_epsilon: rho[0],
                            initial_log_delta: rho[1],
                        })
                        .ok()
                        .map(InverseLink::BetaLogistic)
                    },
                )
            }
            InverseLink::Mixture(state0) if !state0.rho.is_empty() => {
                let components = state0.components.clone();
                let components_recover = components.clone();
                let wiggle_cfg = wiggle.clone();
                optimize_link_parameters(
                    data,
                    spec,
                    kappa_options,
                    state0.rho.clone(),
                    "mixture",
                    wiggle.clone(),
                    move |rho: &Array1<f64>| {
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
                    },
                    move |rho| {
                        state_fromspec(&MixtureLinkSpec {
                            components: components_recover.clone(),
                            initial_rho: rho.to_owned(),
                        })
                        .ok()
                        .map(InverseLink::Mixture)
                    },
                )
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

fn fit_latent_survival_model(
    request: LatentSurvivalFitRequest<'_>,
) -> Result<LatentSurvivalTermFitResult, String> {
    fit_latent_survival_terms(
        request.data,
        request.spec,
        request.frailty,
        &request.options,
    )
}

fn fit_latent_binary_model(
    request: LatentBinaryFitRequest<'_>,
) -> Result<LatentBinaryTermFitResult, String> {
    fit_latent_binary_terms(
        request.data,
        request.spec,
        request.frailty,
        &request.options,
    )
}

fn fit_transformation_normal_model(
    request: TransformationNormalFitRequest<'_>,
) -> Result<TransformationNormalFitResult, String> {
    fit_transformation_normal(
        &request.response,
        &request.weights,
        &request.offset,
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
        FitRequest::LatentSurvival(request) => {
            fit_latent_survival_model(request).map(FitResult::LatentSurvival)
        }
        FitRequest::LatentBinary(request) => {
            fit_latent_binary_model(request).map(FitResult::LatentBinary)
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
    SurvivalBaselineTarget, SurvivalLikelihoodMode, SurvivalTimeBasisConfig,
    append_zero_tail_columns, build_latent_survival_baseline_offsets,
    build_survival_baseline_offsets, build_survival_time_basis,
    build_survival_timewiggle_from_baseline, build_time_varying_survival_covariate_template,
    center_survival_time_designs_at_anchor, evaluate_survival_time_basis_row,
    initial_survival_baseline_config_for_fit, normalize_survival_time_pair,
    optimize_survival_baseline_config, parse_survival_distribution, parse_survival_likelihood_mode,
    parse_survival_time_basis_config, require_structural_survival_time_basis,
    resolve_survival_time_anchor_value, resolved_survival_time_basis_config_from_build,
};
use crate::families::survival_location_scale::{
    SurvivalCovariateTermBlockTemplate, TimeBlockInput, TimeWiggleBlockInput,
    residual_distribution_inverse_link,
};
use crate::inference::data::EncodedDataset as Dataset;
use crate::inference::formula_dsl::{
    LinkChoice, LinkWiggleFormulaSpec, ParsedFormula, effectivelinkwiggle_formulaspec,
    parse_formula, parse_link_choice, parse_matching_auxiliary_formula, parse_surv_response,
    validate_marginal_slope_z_column_exclusion,
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
    /// Optional additive offset column for the primary linear predictor.
    pub offset_column: Option<String>,
    /// Optional additive offset column for the noise/log-scale predictor.
    pub noise_offset_column: Option<String>,
    /// Optional family-level frailty modifier.
    pub frailty: Option<FrailtySpec>,

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
    /// Survival likelihood mode: "location-scale", "transformation", "weibull",
    /// "marginal-slope", "latent", or "latent-binary".
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
    /// Optional non-negative per-row training weights column.
    pub weight_column: Option<String>,

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
            offset_column: None,
            noise_offset_column: None,
            frailty: None,
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
            weight_column: None,
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
        "latent-cloglog-binomial" => Some(LikelihoodFamily::BinomialLatentCLogLog),
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

struct PreparedWorkflowSurvivalTimeStack {
    eta_offset_entry: Array1<f64>,
    eta_offset_exit: Array1<f64>,
    derivative_offset_exit: Array1<f64>,
    unloaded_mass_entry: Array1<f64>,
    unloaded_mass_exit: Array1<f64>,
    unloaded_hazard_exit: Array1<f64>,
    time_design_entry: crate::matrix::DesignMatrix,
    time_design_exit: crate::matrix::DesignMatrix,
    time_design_derivative: crate::matrix::DesignMatrix,
    time_penalties: Vec<Array2<f64>>,
    time_nullspace_dims: Vec<usize>,
    timewiggle_block: Option<TimeWiggleBlockInput>,
}

fn add_workflow_survival_time_derivative_guard_offset(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    anchor_time: f64,
    derivative_guard: f64,
    eta_offset_entry: &mut Array1<f64>,
    eta_offset_exit: &mut Array1<f64>,
    derivative_offset_exit: &mut Array1<f64>,
) -> Result<(), String> {
    if derivative_guard <= 0.0 {
        return Ok(());
    }
    let n = age_entry.len();
    if age_exit.len() != n
        || eta_offset_entry.len() != n
        || eta_offset_exit.len() != n
        || derivative_offset_exit.len() != n
    {
        return Err("workflow survival derivative-guard offset lengths must match".to_string());
    }
    for i in 0..n {
        eta_offset_entry[i] += derivative_guard * (age_entry[i] - anchor_time);
        eta_offset_exit[i] += derivative_guard * (age_exit[i] - anchor_time);
        derivative_offset_exit[i] += derivative_guard;
    }
    Ok(())
}

fn prepare_workflow_survival_time_stack(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    baseline_cfg: &crate::families::survival_construction::SurvivalBaselineConfig,
    time_anchor: f64,
    derivative_guard: f64,
    time_build: &crate::families::survival_construction::SurvivalTimeBuildOutput,
    effective_timewiggle: Option<&LinkWiggleFormulaSpec>,
    latent_loading: Option<crate::families::lognormal_kernel::HazardLoading>,
) -> Result<PreparedWorkflowSurvivalTimeStack, String> {
    let (
        mut eta_offset_entry,
        mut eta_offset_exit,
        mut derivative_offset_exit,
        unloaded_mass_entry,
        unloaded_mass_exit,
        unloaded_hazard_exit,
    ) = if let Some(loading) = latent_loading {
        let offsets =
            build_latent_survival_baseline_offsets(age_entry, age_exit, baseline_cfg, loading)?;
        (
            offsets.loaded_eta_entry,
            offsets.loaded_eta_exit,
            offsets.loaded_derivative_exit,
            offsets.unloaded_mass_entry,
            offsets.unloaded_mass_exit,
            offsets.unloaded_hazard_exit,
        )
    } else {
        let (eta_offset_entry, eta_offset_exit, derivative_offset_exit) =
            build_survival_baseline_offsets(age_entry, age_exit, baseline_cfg)?;
        let n = age_entry.len();
        (
            eta_offset_entry,
            eta_offset_exit,
            derivative_offset_exit,
            Array1::zeros(n),
            Array1::zeros(n),
            Array1::zeros(n),
        )
    };
    add_workflow_survival_time_derivative_guard_offset(
        age_entry,
        age_exit,
        time_anchor,
        derivative_guard,
        &mut eta_offset_entry,
        &mut eta_offset_exit,
        &mut derivative_offset_exit,
    )?;
    let timewiggle_build = if let Some(cfg) = effective_timewiggle {
        Some(build_survival_timewiggle_from_baseline(
            &eta_offset_entry,
            &eta_offset_exit,
            &derivative_offset_exit,
            cfg,
        )?)
    } else {
        None
    };
    let mut time_design_entry = time_build.x_entry_time.clone();
    let mut time_design_exit = time_build.x_exit_time.clone();
    let mut time_design_derivative = time_build.x_derivative_time.clone();
    let mut time_penalties = time_build.penalties.clone();
    let mut time_nullspace_dims = time_build.nullspace_dims.clone();
    let mut timewiggle_block = None;
    if let Some(wiggle) = timewiggle_build.as_ref() {
        let p_base = time_design_exit.ncols();
        append_zero_tail_columns(
            &mut time_design_entry,
            &mut time_design_exit,
            &mut time_design_derivative,
            wiggle.ncols,
        );
        for (idx, penalty) in wiggle.penalties.iter().enumerate() {
            let mut embedded = Array2::<f64>::zeros((p_base + wiggle.ncols, p_base + wiggle.ncols));
            embedded
                .slice_mut(s![
                    p_base..p_base + wiggle.ncols,
                    p_base..p_base + wiggle.ncols
                ])
                .assign(penalty);
            time_penalties.push(embedded);
            time_nullspace_dims.push(wiggle.nullspace_dims.get(idx).copied().unwrap_or(0));
        }
        timewiggle_block = Some(TimeWiggleBlockInput {
            knots: wiggle.knots.clone(),
            degree: wiggle.degree,
            ncols: wiggle.ncols,
        });
    }
    Ok(PreparedWorkflowSurvivalTimeStack {
        eta_offset_entry,
        eta_offset_exit,
        derivative_offset_exit,
        unloaded_mass_entry,
        unloaded_mass_exit,
        unloaded_hazard_exit,
        time_design_entry,
        time_design_exit,
        time_design_derivative,
        time_penalties,
        time_nullspace_dims,
        timewiggle_block,
    })
}

fn resolve_continuous_column(
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    column_name: &str,
    role: &str,
) -> Result<Array1<f64>, String> {
    let col_idx = *col_map
        .get(column_name)
        .ok_or_else(|| format!("{role} column '{column_name}' not found"))?;
    let values = data.values.column(col_idx).to_owned();
    for (row_idx, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "{role} column '{column_name}' contains non-finite value at row {row_idx}: {value}"
            ));
        }
    }
    Ok(values)
}

pub fn resolve_offset_column(
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    column_name: Option<&str>,
) -> Result<Array1<f64>, String> {
    let Some(column_name) = column_name else {
        return Ok(Array1::zeros(data.values.nrows()));
    };
    resolve_continuous_column(data, col_map, column_name, "offset")
}

pub fn resolve_weight_column(
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    column_name: Option<&str>,
) -> Result<Array1<f64>, String> {
    let Some(column_name) = column_name else {
        return Ok(Array1::ones(data.values.nrows()));
    };
    let values = resolve_continuous_column(data, col_map, column_name, "weights")?;
    for (row_idx, value) in values.iter().enumerate() {
        if *value < 0.0 {
            return Err(format!(
                "weights column '{column_name}' must be non-negative; found {value} at row {row_idx}"
            ));
        }
    }
    Ok(values)
}

fn materialize_standard<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, String> {
    if config.noise_offset_column.is_some() {
        return Err(
            "noise_offset_column requires a location-scale model with noise_formula".to_string(),
        );
    }
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

    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;
    let latent_cloglog = if matches!(family, LikelihoodFamily::BinomialLatentCLogLog) {
        let sigma = match config.frailty.clone().unwrap_or(FrailtySpec::None) {
            FrailtySpec::HazardMultiplier {
                sigma_fixed: Some(sigma),
                loading: crate::families::lognormal_kernel::HazardLoading::Full,
            } => sigma,
            FrailtySpec::HazardMultiplier {
                sigma_fixed: Some(_),
                loading,
            } => {
                return Err(format!(
                    "latent-cloglog-binomial requires HazardLoading::Full, got {loading:?}"
                ));
            }
            FrailtySpec::HazardMultiplier {
                sigma_fixed: None, ..
            } => {
                return Err(
                    "latent-cloglog-binomial currently requires a fixed hazard-multiplier sigma"
                        .to_string(),
                );
            }
            FrailtySpec::GaussianShift { .. } => {
                return Err(
                    "latent-cloglog-binomial does not support GaussianShift frailty".to_string(),
                );
            }
            FrailtySpec::None => {
                return Err(
                    "latent-cloglog-binomial requires config.frailty=HazardMultiplier with a fixed sigma"
                        .to_string(),
                );
            }
        };
        Some(
            LatentCLogLogState::new(sigma)
                .map_err(|e| format!("invalid latent_cloglog state: {e}"))?,
        )
    } else {
        if config.frailty.is_some() {
            return Err(format!(
                "config.frailty is not supported for standard family {:?}; use a frailty-aware family instead",
                family
            ));
        }
        None
    };
    let options = FitOptions {
        latent_cloglog,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
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
            .unwrap_or_else(|| {
                if let Some(state) = latent_cloglog {
                    InverseLink::LatentCLogLog(state)
                } else {
                    InverseLink::Standard(LinkFunction::Logit)
                }
            });
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

    let survival_mode = parse_survival_likelihood_mode(&config.survival_likelihood)?;
    if matches!(
        survival_mode,
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
    ) {
        return Err(format!(
            "survival likelihood '{}' is not yet supported through the unified API; \
             use 'location-scale', 'marginal-slope', 'latent', or 'latent-binary'. For transformation/weibull, use FitRequest directly.",
            config.survival_likelihood
        ));
    }
    let baseline_cfg = initial_survival_baseline_config_for_fit(
        &config.baseline_target,
        config.baseline_scale,
        config.baseline_shape,
        config.baseline_rate,
        config.baseline_makeham,
        &age_exit,
    )?;
    if matches!(
        survival_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) && baseline_cfg.target == SurvivalBaselineTarget::Linear
    {
        return Err(
            "latent hazard-window families require a non-linear scalar baseline target; use baseline_target weibull, gompertz, or gompertz-makeham"
                .to_string(),
        );
    }
    let effective_timewiggle = parsed.timewiggle.clone();
    let time_cfg = if effective_timewiggle.is_some() {
        // Match the CLI path: the parametric baseline plus timewiggle supplies
        // the time structure, so the base time basis is disabled.
        SurvivalTimeBasisConfig::None
    } else {
        parse_survival_time_basis_config(
            &config.time_basis,
            config.time_degree,
            config.time_num_internal_knots,
            config.time_smooth_lambda,
        )?
    };
    let time_anchor = resolve_survival_time_anchor_value(&age_entry, None)?;
    let exact_derivative_guard = match survival_mode {
        SurvivalLikelihoodMode::LocationScale
        | SurvivalLikelihoodMode::Latent
        | SurvivalLikelihoodMode::LatentBinary => DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
        SurvivalLikelihoodMode::MarginalSlope => DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD,
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull => 0.0,
    };

    // Build time basis
    let mut time_build = build_survival_time_basis(
        &age_entry,
        &age_exit,
        time_cfg.clone(),
        Some((config.time_num_internal_knots, config.time_smooth_lambda)),
    )?;
    if survival_mode != SurvivalLikelihoodMode::Weibull && effective_timewiggle.is_none() {
        require_structural_survival_time_basis(&time_build.basisname, "workflow survival fitting")?;
    }
    let resolved_time_cfg = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    let time_anchor_row = evaluate_survival_time_basis_row(time_anchor, &resolved_time_cfg)?;
    center_survival_time_designs_at_anchor(
        &mut time_build.x_entry_time,
        &mut time_build.x_exit_time,
        &time_anchor_row,
    )?;
    if effective_timewiggle.is_some() && baseline_cfg.target == SurvivalBaselineTarget::Linear {
        return Err(
            "timewiggle requires a non-linear scalar survival baseline target; \
             use baseline_target weibull, gompertz, or gompertz-makeham"
                .to_string(),
        );
    }

    let mut termspec = build_termspec(&parsed.terms, data, col_map, &mut inference_notes)?;
    if config.scale_dimensions {
        enable_scale_dimensions(&mut termspec);
    }

    let residual_dist = parse_survival_distribution(&config.survival_distribution)?;
    let survival_inverse_link = residual_distribution_inverse_link(residual_dist);
    let link_choice = parse_link_choice(config.link.as_deref(), config.flexible_link)?;
    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(parsed.linkwiggle.as_ref(), link_choice.as_ref());
    let effective_linkwiggle_cfg = effective_linkwiggle.clone().map(|cfg| LinkWiggleConfig {
        degree: cfg.degree,
        num_internal_knots: cfg.num_internal_knots,
        penalty_orders: cfg.penalty_orders,
        double_penalty: cfg.double_penalty,
    });

    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let threshold_offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;
    let log_sigma_offset =
        resolve_offset_column(data, col_map, config.noise_offset_column.as_deref())?;
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
    let marginal_z_column_name =
        if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
            Some(config.z_column.as_deref().ok_or_else(|| {
                "marginal-slope survival requires z_column in FitConfig".to_string()
            })?)
        } else {
            None
        };
    let marginal_z = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        if parsed.linkspec.is_some() {
            return Err(
                "link(...) is not implemented for the survival marginal-slope family".to_string(),
            );
        }
        let z_col_name = marginal_z_column_name
            .expect("marginal-slope z column should be validated before materialization");
        let z_idx = *col_map
            .get(z_col_name)
            .ok_or_else(|| format!("z column '{z_col_name}' not found"))?;
        Some(data.values.column(z_idx).to_owned())
    } else {
        None
    };
    let (marginal_logslopespec, marginal_slope_deviation_routing) = if survival_mode
        == SurvivalLikelihoodMode::MarginalSlope
    {
        if let Some(ls_formula) = config.logslope_formula.as_deref() {
            let (_, ls_parsed) =
                parse_matching_auxiliary_formula(ls_formula, &parsed.response, "logslope_formula")?;
            if ls_parsed.linkspec.is_some() {
                return Err(
                        "link(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string(),
                    );
            }
            if ls_parsed.timewiggle.is_some() {
                return Err(
                        "timewiggle(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string(),
                    );
            }
            if ls_parsed.survivalspec.is_some() {
                return Err(
                        "survmodel(...) is not supported in logslope_formula for the survival marginal-slope family"
                            .to_string(),
                    );
            }
            validate_marginal_slope_z_column_exclusion(
                parsed,
                &ls_parsed,
                marginal_z_column_name.expect("marginal-slope z column should be available"),
                "survival marginal-slope",
                "logslope_formula",
            )?;
            (
                Some(build_termspec(
                    &ls_parsed.terms,
                    data,
                    col_map,
                    &mut inference_notes,
                )?),
                route_marginal_slope_deviation_blocks(
                    parsed.linkwiggle.as_ref(),
                    ls_parsed.linkwiggle.as_ref(),
                ),
            )
        } else {
            validate_marginal_slope_z_column_exclusion(
                parsed,
                parsed,
                marginal_z_column_name.expect("marginal-slope z column should be available"),
                "survival marginal-slope",
                "logslope_formula",
            )?;
            (
                Some(termspec.clone()),
                route_marginal_slope_deviation_blocks(parsed.linkwiggle.as_ref(), None),
            )
        }
    } else {
        (
            None,
            MarginalSlopeDeviationRouting {
                score_warp: None,
                link_dev: None,
            },
        )
    };
    let marginal_slope_score_warp = marginal_slope_deviation_routing.score_warp;
    let marginal_slope_link_dev = marginal_slope_deviation_routing.link_dev;
    if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        if parsed.linkwiggle.is_some() {
            inference_notes.push(
                "survival marginal-slope routes formula-level linkwiggle(...) into its anchored internal link-deviation block while keeping the probit survival base link".to_string(),
            );
        }
        if marginal_slope_score_warp.is_some() {
            inference_notes.push(
                "survival marginal-slope routes logslope_formula linkwiggle(...) into its anchored internal score-warp block while keeping the probit survival base link".to_string(),
            );
        }
        if marginal_slope_link_dev.is_none() && marginal_slope_score_warp.is_none() {
            inference_notes.push(
                "survival marginal-slope rigid mode is algebraic closed-form exact".to_string(),
            );
        } else {
            inference_notes.push(
                "survival marginal-slope flexible score/link mode uses calibrated de-nested cubic transport cells with analytic value evaluation and calibrated survival normalization"
                    .to_string(),
            );
        }
    }
    let marginal_slope_frailty = if survival_mode == SurvivalLikelihoodMode::MarginalSlope {
        Some(fixed_gaussian_shift_frailty_from_spec(
            config.frailty.as_ref().unwrap_or(&FrailtySpec::None),
            "survival marginal-slope",
        )?)
    } else {
        None
    };
    match survival_mode {
        SurvivalLikelihoodMode::LocationScale if config.frailty.is_some() => {
            return Err(
                "config.frailty is not implemented for survival-likelihood=location-scale"
                    .to_string(),
            );
        }
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
            if effective_timewiggle.is_some() =>
        {
            return Err(
                "timewiggle is not implemented for latent survival/binary likelihoods".to_string(),
            );
        }
        _ => {}
    }
    let latent_loading = if matches!(
        survival_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) {
        let frailty = config.frailty.as_ref().unwrap_or(&FrailtySpec::None);
        Some(latent_hazard_loading(
            frailty,
            "workflow latent survival/binary",
        )?)
    } else {
        None
    };

    let build_time_block =
        |candidate: &crate::families::survival_construction::SurvivalBaselineConfig| {
            let prepared = prepare_workflow_survival_time_stack(
                &age_entry,
                &age_exit,
                candidate,
                time_anchor,
                exact_derivative_guard,
                &time_build,
                effective_timewiggle.as_ref(),
                None,
            )?;
            let time_p = prepared.time_design_exit.ncols();
            let time_initial_log_lambdas = if prepared.time_penalties.is_empty() {
                None
            } else {
                Some(Array1::from_elem(
                    prepared.time_penalties.len(),
                    config.time_smooth_lambda.ln(),
                ))
            };
            let time_block = TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                structural_monotonicity: true,
                penalties: prepared.time_penalties.clone(),
                nullspace_dims: prepared.time_nullspace_dims.clone(),
                initial_log_lambdas: time_initial_log_lambdas,
                initial_beta: Some(Array1::from_elem(time_p, 1e-4)),
            };
            Ok::<_, String>((prepared, time_block))
        };

    let build_location_scale_request =
        |candidate: &crate::families::survival_construction::SurvivalBaselineConfig| {
            let (prepared, time_block) = build_time_block(candidate)?;
            let spec = SurvivalLocationScaleTermSpec {
                age_entry: age_entry.clone(),
                age_exit: age_exit.clone(),
                event_target: event.clone(),
                weights: weights.clone(),
                inverse_link: survival_inverse_link.clone(),
                derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
                max_iter: 200,
                tol: 1e-7,
                time_block,
                thresholdspec: termspec.clone(),
                log_sigmaspec: log_sigmaspec.clone(),
                threshold_offset: threshold_offset.clone(),
                log_sigma_offset: log_sigma_offset.clone(),
                threshold_template: threshold_template.clone(),
                log_sigma_template: log_sigma_template.clone(),
                timewiggle_block: prepared.timewiggle_block,
                linkwiggle_block: None,
            };
            let optimize_inverse_link =
                survival_inverse_link_has_free_parameters(&spec.inverse_link);
            Ok::<_, String>(SurvivalLocationScaleFitRequest {
                data: data.values.view(),
                spec,
                wiggle: effective_linkwiggle_cfg.clone(),
                kappa_options: SpatialLengthScaleOptimizationOptions::default(),
                optimize_inverse_link,
            })
        };

    let build_marginal_slope_request =
        |candidate: &crate::families::survival_construction::SurvivalBaselineConfig| {
            let (prepared, time_block) = build_time_block(candidate)?;
            Ok::<_, String>(SurvivalMarginalSlopeFitRequest {
                data: data.values.view(),
                spec: SurvivalMarginalSlopeTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event.clone(),
                    weights: weights.clone(),
                    z: marginal_z.clone().ok_or_else(|| {
                        "marginal-slope survival requires z_column in FitConfig".to_string()
                    })?,
                    marginalspec: termspec.clone(),
                    marginal_offset: threshold_offset.clone(),
                    frailty: marginal_slope_frailty.clone().ok_or_else(|| {
                        "internal error: marginal-slope frailty validation missing".to_string()
                    })?,
                    derivative_guard: DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD,
                    time_block,
                    timewiggle_block: prepared.timewiggle_block,
                    logslopespec: marginal_logslopespec.clone().ok_or_else(|| {
                        "marginal-slope survival is missing logslope spec".to_string()
                    })?,
                    logslope_offset: log_sigma_offset.clone(),
                    score_warp: marginal_slope_score_warp.clone(),
                    link_dev: marginal_slope_link_dev.clone(),
                },
                options: BlockwiseFitOptions {
                    compute_covariance: true,
                    ..Default::default()
                },
                kappa_options: SpatialLengthScaleOptimizationOptions::default(),
            })
        };

    let build_latent_survival_request =
        |candidate: &crate::families::survival_construction::SurvivalBaselineConfig| {
            let loading = latent_loading.ok_or_else(|| {
                "internal error: latent survival loading missing after frailty validation"
                    .to_string()
            })?;
            let prepared = prepare_workflow_survival_time_stack(
                &age_entry,
                &age_exit,
                candidate,
                time_anchor,
                exact_derivative_guard,
                &time_build,
                None,
                Some(loading),
            )?;
            let time_p = prepared.time_design_exit.ncols();
            let time_initial_log_lambdas = if prepared.time_penalties.is_empty() {
                None
            } else {
                Some(Array1::from_elem(
                    prepared.time_penalties.len(),
                    config.time_smooth_lambda.ln(),
                ))
            };
            let time_block = TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                structural_monotonicity: true,
                penalties: prepared.time_penalties.clone(),
                nullspace_dims: prepared.time_nullspace_dims.clone(),
                initial_log_lambdas: time_initial_log_lambdas,
                initial_beta: Some(Array1::from_elem(time_p, 1e-4)),
            };
            Ok::<_, String>(LatentSurvivalFitRequest {
                data: data.values.view(),
                spec: LatentSurvivalTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event.mapv(|v| if v >= 0.5 { 1 } else { 0 }),
                    weights: weights.clone(),
                    derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
                    time_block,
                    unloaded_mass_entry: prepared.unloaded_mass_entry,
                    unloaded_mass_exit: prepared.unloaded_mass_exit,
                    unloaded_hazard_exit: prepared.unloaded_hazard_exit,
                    meanspec: termspec.clone(),
                    mean_offset: threshold_offset.clone(),
                },
                frailty: config.frailty.clone().unwrap_or(FrailtySpec::None),
                options: BlockwiseFitOptions::default(),
            })
        };

    let build_latent_binary_request =
        |candidate: &crate::families::survival_construction::SurvivalBaselineConfig| {
            let loading = latent_loading.ok_or_else(|| {
                "internal error: latent binary loading missing after frailty validation".to_string()
            })?;
            let prepared = prepare_workflow_survival_time_stack(
                &age_entry,
                &age_exit,
                candidate,
                time_anchor,
                exact_derivative_guard,
                &time_build,
                None,
                Some(loading),
            )?;
            let time_p = prepared.time_design_exit.ncols();
            let time_initial_log_lambdas = if prepared.time_penalties.is_empty() {
                None
            } else {
                Some(Array1::from_elem(
                    prepared.time_penalties.len(),
                    config.time_smooth_lambda.ln(),
                ))
            };
            let time_block = TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                structural_monotonicity: true,
                penalties: prepared.time_penalties.clone(),
                nullspace_dims: prepared.time_nullspace_dims.clone(),
                initial_log_lambdas: time_initial_log_lambdas,
                initial_beta: Some(Array1::from_elem(time_p, 1e-4)),
            };
            Ok::<_, String>(LatentBinaryFitRequest {
                data: data.values.view(),
                spec: LatentBinaryTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event.mapv(|v| if v >= 0.5 { 1 } else { 0 }),
                    weights: weights.clone(),
                    derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
                    time_block,
                    unloaded_mass_entry: prepared.unloaded_mass_entry,
                    unloaded_mass_exit: prepared.unloaded_mass_exit,
                    meanspec: termspec.clone(),
                    mean_offset: threshold_offset.clone(),
                },
                frailty: config.frailty.clone().unwrap_or(FrailtySpec::None),
                options: BlockwiseFitOptions::default(),
            })
        };

    let baseline_cfg = if baseline_cfg.target != SurvivalBaselineTarget::Linear {
        optimize_survival_baseline_config(
            &baseline_cfg,
            "workflow survival baseline",
            |candidate| match survival_mode {
                SurvivalLikelihoodMode::LocationScale => Ok(fit_survival_location_scale_model(
                    build_location_scale_request(candidate)?,
                )
                .map_err(|e| format!("survival location-scale fit failed: {e}"))?
                .fit
                .fit
                .reml_score),
                SurvivalLikelihoodMode::MarginalSlope => Ok(fit_survival_marginal_slope_model(
                    build_marginal_slope_request(candidate)?,
                )
                .map_err(|e| format!("survival marginal-slope fit failed: {e}"))?
                .fit
                .reml_score),
                SurvivalLikelihoodMode::Latent => Ok(fit_latent_survival_model(
                    build_latent_survival_request(candidate)?,
                )
                .map_err(|e| format!("latent survival fit failed: {e}"))?
                .fit
                .reml_score),
                SurvivalLikelihoodMode::LatentBinary => Ok(fit_latent_binary_model(
                    build_latent_binary_request(candidate)?,
                )
                .map_err(|e| format!("latent binary fit failed: {e}"))?
                .fit
                .reml_score),
                SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull => {
                    unreachable!()
                }
            },
        )?
    } else {
        baseline_cfg
    };

    let request = match survival_mode {
        SurvivalLikelihoodMode::LocationScale => {
            FitRequest::SurvivalLocationScale(build_location_scale_request(&baseline_cfg)?)
        }
        SurvivalLikelihoodMode::MarginalSlope => {
            FitRequest::SurvivalMarginalSlope(build_marginal_slope_request(&baseline_cfg)?)
        }
        SurvivalLikelihoodMode::Latent => {
            FitRequest::LatentSurvival(build_latent_survival_request(&baseline_cfg)?)
        }
        SurvivalLikelihoodMode::LatentBinary => {
            FitRequest::LatentBinary(build_latent_binary_request(&baseline_cfg)?)
        }
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull => {
            unreachable!()
        }
    };

    Ok(MaterializedModel {
        request,
        inference_notes,
    })
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

    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let mean_offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;
    let noise_offset = resolve_offset_column(data, col_map, config.noise_offset_column.as_deref())?;
    let kappa_options = SpatialLengthScaleOptimizationOptions::default();
    let options = BlockwiseFitOptions::default();

    let wiggle_cfg = effective_linkwiggle.map(|cfg| LinkWiggleConfig {
        degree: cfg.degree,
        num_internal_knots: cfg.num_internal_knots,
        penalty_orders: cfg.penalty_orders,
        double_penalty: cfg.double_penalty,
    });

    if matches!(family, LikelihoodFamily::BinomialLatentCLogLog) {
        return Err(
            "latent-cloglog-binomial is not implemented for location-scale fitting".to_string(),
        );
    }

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
                    threshold_offset: mean_offset,
                    log_sigma_offset: noise_offset,
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
                    mean_offset,
                    log_sigma_offset: noise_offset,
                },
                wiggle: wiggle_cfg,
                options,
                kappa_options,
            }),
            inference_notes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::data::load_dataset_projected;
    use crate::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
    use crate::solver::outer_strategy::{HessianSource, OuterPlan, OuterResult, Solver};
    use ndarray::Array2;
    use std::fs;
    use tempfile::tempdir;

    fn load_survival_dataset() -> crate::inference::data::EncodedDataset {
        let td = tempdir().expect("tempdir");
        let data_path = td.path().join("survival.csv");
        fs::write(
            &data_path,
            "entry,exit,event,x,z\n0.0,1.0,1,0.2,-0.4\n0.3,1.6,0,-0.1,0.6\n",
        )
        .expect("write survival csv");
        load_dataset_projected(
            &data_path,
            &[
                "entry".to_string(),
                "exit".to_string(),
                "event".to_string(),
                "x".to_string(),
                "z".to_string(),
            ],
        )
        .expect("load survival dataset")
    }

    #[test]
    fn survival_marginal_slope_materialize_rejects_z_column_in_main_formula() {
        let data = load_survival_dataset();
        let mut config = FitConfig::default();
        config.survival_likelihood = "marginal-slope".to_string();
        config.logslope_formula = Some("1".to_string());
        config.z_column = Some("z".to_string());

        let err = materialize("Surv(entry, exit, event) ~ x + z", &data, &config)
            .err()
            .expect("main formula should reject z-column reuse");

        assert!(err.contains("survival marginal-slope reserves z column 'z'"));
        assert!(err.contains("main formula"));
    }

    #[test]
    fn survival_marginal_slope_materialize_rejects_z_column_in_logslope_formula() {
        let data = load_survival_dataset();
        let mut config = FitConfig::default();
        config.survival_likelihood = "marginal-slope".to_string();
        config.logslope_formula = Some("1 + z".to_string());
        config.z_column = Some("z".to_string());

        let err = materialize("Surv(entry, exit, event) ~ x", &data, &config)
            .err()
            .expect("logslope formula should reject z-column reuse");

        assert!(err.contains("survival marginal-slope reserves z column 'z'"));
        assert!(err.contains("logslope_formula"));
    }

    #[test]
    fn survival_marginal_slope_materialize_rejects_z_column_when_logslope_defaults_to_main_spec() {
        let data = load_survival_dataset();
        let mut config = FitConfig::default();
        config.survival_likelihood = "marginal-slope".to_string();
        config.z_column = Some("z".to_string());

        let err = materialize("Surv(entry, exit, event) ~ x + z", &data, &config)
            .err()
            .expect("defaulted logslope spec should still reject z-column reuse");

        assert!(err.contains("survival marginal-slope reserves z column 'z'"));
        assert!(err.contains("main formula"));
    }

    fn workflow_test_dataset() -> Dataset {
        Dataset {
            headers: vec![
                "age_entry".to_string(),
                "age_exit".to_string(),
                "event".to_string(),
                "bmi".to_string(),
                "z".to_string(),
            ],
            values: Array2::from_shape_vec(
                (4, 5),
                vec![
                    40.0, 43.0, 1.0, 22.0, -1.0, 41.0, 46.0, 0.0, 24.0, -0.2, 42.0, 47.0, 1.0,
                    27.0, 0.3, 44.0, 49.0, 0.0, 29.0, 1.2,
                ],
            )
            .expect("workflow test data shape"),
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "age_entry".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "age_exit".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "event".to_string(),
                        kind: ColumnKindTag::Binary,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "bmi".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "z".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Binary,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
            ],
        }
    }

    fn workflow_test_outer_result(converged: bool, rho: Array1<f64>) -> OuterResult {
        OuterResult {
            rho,
            final_value: 1.25,
            iterations: 7,
            final_grad_norm: 0.5,
            final_gradient: None,
            final_hessian: None,
            converged,
            plan_used: OuterPlan {
                solver: Solver::Bfgs,
                hessian_source: HessianSource::BfgsApprox,
            },
        }
    }

    #[test]
    fn workflow_survival_marginal_slope_routes_logslope_linkwiggle_into_score_warp_only() {
        let data = workflow_test_dataset();
        let config = FitConfig {
            survival_likelihood: "marginal-slope".to_string(),
            logslope_formula: Some(
                "1 + linkwiggle(degree=5, internal_knots=7, penalty_order=\"2,3\")".to_string(),
            ),
            z_column: Some("z".to_string()),
            ..FitConfig::default()
        };
        let materialized = materialize(
            "Surv(age_entry, age_exit, event) ~ s(bmi) + linkwiggle(degree=4, internal_knots=9, penalty_order=\"1\")",
            &data,
            &config,
        )
        .expect("workflow materialization should succeed");

        let MaterializedModel {
            request,
            inference_notes,
        } = materialized;
        let FitRequest::SurvivalMarginalSlope(request) = request else {
            panic!("expected survival marginal-slope request");
        };

        let link_dev = request.spec.link_dev.expect("main-formula link-dev");
        let score_warp = request.spec.score_warp.expect("logslope score-warp");
        assert_eq!(link_dev.degree, 4);
        assert_eq!(link_dev.num_internal_knots, 9);
        assert_eq!(link_dev.penalty_order, 1);
        assert!(link_dev.penalty_orders.is_empty());
        assert_eq!(score_warp.degree, 5);
        assert_eq!(score_warp.num_internal_knots, 7);
        assert_eq!(score_warp.penalty_order, 2);
        assert_eq!(score_warp.penalty_orders, vec![3]);
        assert!(
            inference_notes
                .iter()
                .any(|note| note.contains("link-deviation block")),
            "workflow notes should mention main-formula linkwiggle routing"
        );
        assert!(
            inference_notes
                .iter()
                .any(|note| note.contains("score-warp block")),
            "workflow notes should mention logslope_formula linkwiggle routing"
        );
    }

    #[test]
    fn survival_location_scale_wiggle_rejects_unsupported_inverse_link() {
        let data = workflow_test_dataset();
        let materialized = materialize(
            "Surv(age_entry, age_exit, event) ~ bmi + linkwiggle(degree=4, internal_knots=3, penalty_order=\"1\")",
            &data,
            &FitConfig::default(),
        )
        .expect("workflow materialization should succeed");

        let MaterializedModel { request, .. } = materialized;
        let FitRequest::SurvivalLocationScale(mut request) = request else {
            panic!("expected survival location-scale request");
        };
        request.spec.inverse_link = InverseLink::Sas(
            state_from_sasspec(SasLinkSpec {
                initial_epsilon: 0.1,
                initial_log_delta: 0.0,
            })
            .expect("valid SAS state"),
        );
        request.optimize_inverse_link = false;

        let err = match fit_survival_location_scale_model(request) {
            Ok(_) => panic!("survival link wiggle should reject unsupported inverse links"),
            Err(e) => e,
        };

        assert!(err.contains("survival link wiggle"));
        assert!(err.contains("does not support"));
    }

    #[test]
    fn survival_inverse_link_result_requires_convergence() {
        let err = recover_converged_survival_inverse_link(
            workflow_test_outer_result(false, Array1::from_vec(vec![0.1, -0.2])),
            "survival inverse-link optimization (SAS, dim=2)",
            |_| Some(InverseLink::Standard(LinkFunction::Logit)),
        )
        .expect_err("non-converged inverse-link search should fail");

        assert!(err.contains("did not converge"));
        assert!(err.contains("final_objective"));
    }

    #[test]
    fn survival_inverse_link_result_requires_recoverable_state() {
        let err = recover_converged_survival_inverse_link(
            workflow_test_outer_result(true, Array1::from_vec(vec![9.0, 8.0])),
            "survival inverse-link optimization (mixture, dim=2)",
            |_| None,
        )
        .expect_err("unrecoverable inverse-link state should fail");

        assert!(err.contains("produced an invalid inverse-link state"));
        assert!(err.contains("9.0"));
    }
}

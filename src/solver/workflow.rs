use crate::basis::create_difference_penalty_matrix;
use crate::custom_family::BlockwiseFitOptions;
use crate::estimate::{FitOptions, FittedLinkState, UnifiedFitResult};
use crate::families::gamlss::{
    BinomialLocationScaleTermSpec, BinomialLocationScaleWiggleWorkflowConfig,
    BinomialLocationScaleWorkflowResult, BlockwiseTermFitResult, GaussianLocationScaleTermSpec,
    ParameterBlockInput, WiggleBlockConfig, buildwiggle_block_input_from_seed,
    fit_binomial_location_scale_termsworkflow, fit_binomial_mean_wiggle_terms_auto_from_pilot,
    fit_gaussian_location_scale_terms,
};
use crate::families::survival_location_scale::{
    LinkWiggleBlockInput, SurvivalLocationScaleTermFitResult, SurvivalLocationScaleTermSpec,
    fit_survival_location_scale_terms,
};
use crate::joint::{JointLinkGeometry, JointModelConfig, JointModelResult, fit_joint_model_engine};
use crate::smooth::{
    AdaptiveRegularizationDiagnostics, SpatialLengthScaleOptimizationOptions, TermCollectionDesign,
    TermCollectionSpec, build_term_collection_design,
    fit_term_collectionwith_spatial_length_scale_optimization,
};
use crate::types::{InverseLink, LikelihoodFamily, LinkFunction};
use ndarray::{Array1, ArrayView1, ArrayView2};

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

pub struct FlexibleLinkWorkflowRequest<'a> {
    pub y: ArrayView1<'a, f64>,
    pub weights: ArrayView1<'a, f64>,
    pub data: ArrayView2<'a, f64>,
    pub spec: TermCollectionSpec,
    pub link: LinkFunction,
    pub geometry: JointLinkGeometry,
    pub config: JointModelConfig,
    pub include_base_covariance: bool,
}

pub struct GaussianLocationScaleWorkflowRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: GaussianLocationScaleTermSpec,
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
}

pub enum FitModelRequest<'a> {
    Standard(StandardFitWorkflowRequest<'a>),
    FlexibleLink(FlexibleLinkWorkflowRequest<'a>),
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

pub struct FlexibleLinkWorkflowResult {
    pub fit: JointModelResult,
    pub design: TermCollectionDesign,
    pub spec: TermCollectionSpec,
}

pub struct SurvivalLocationScaleWorkflowResult {
    pub fit: SurvivalLocationScaleTermFitResult,
    pub wiggle_knots: Option<Array1<f64>>,
    pub wiggle_degree: Option<usize>,
}

pub enum FitModelResult {
    Standard(StandardFitWorkflowResult),
    FlexibleLink(FlexibleLinkWorkflowResult),
    GaussianLocationScale(BlockwiseTermFitResult),
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

    let solved = fit_binomial_mean_wiggle_terms_auto_from_pilot(
        request.data,
        &result.resolvedspec,
        &result.design,
        &result.fit,
        &request.y,
        &request.weights,
        wiggle.link_kind,
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

fn fit_flexible_link_model(
    request: FlexibleLinkWorkflowRequest<'_>,
) -> Result<FlexibleLinkWorkflowResult, String> {
    let design =
        build_term_collection_design(request.data, &request.spec).map_err(|e| e.to_string())?;
    let fit = fit_joint_model_engine(
        request.y,
        request.weights,
        design.design.view(),
        design.penalties.clone(),
        request.link,
        request.geometry,
        request.config,
        request.include_base_covariance,
    )
    .map_err(|e| e.to_string())?;
    Ok(FlexibleLinkWorkflowResult {
        fit,
        design,
        spec: request.spec,
    })
}

fn fit_gaussian_location_scale_model(
    request: GaussianLocationScaleWorkflowRequest<'_>,
) -> Result<BlockwiseTermFitResult, String> {
    fit_gaussian_location_scale_terms(
        request.data,
        request.spec,
        &request.options,
        &request.kappa_options,
    )
}

fn fit_binomial_location_scale_model(
    request: BinomialLocationScaleWorkflowRequest<'_>,
) -> Result<BinomialLocationScaleWorkflowResult, String> {
    fit_binomial_location_scale_termsworkflow(
        request.data,
        request.spec,
        request
            .wiggle
            .map(|wiggle| BinomialLocationScaleWiggleWorkflowConfig {
                degree: wiggle.degree,
                num_internal_knots: wiggle.num_internal_knots,
                penalty_orders: wiggle.penalty_orders,
                double_penalty: wiggle.double_penalty,
            }),
        &request.options,
        &request.kappa_options,
    )
}

fn fit_survival_location_scale_model(
    request: SurvivalLocationScaleWorkflowRequest<'_>,
) -> Result<SurvivalLocationScaleWorkflowResult, String> {
    let mut wiggle_knots = None;
    let mut wiggle_degree = None;

    let fit = if let Some(wiggle) = request.wiggle {
        let mut pilot_spec = request.spec.clone();
        pilot_spec.linkwiggle_block = None;
        let pilot =
            fit_survival_location_scale_terms(request.data, pilot_spec, &request.kappa_options)?;
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
        let mut wiggle_spec = request.spec;
        wiggle_spec.linkwiggle_block = Some(LinkWiggleBlockInput {
            design: wiggle_block.design,
            penalties: wiggle_block.penalties,
            initial_log_lambdas: wiggle_block.initial_log_lambdas,
            initial_beta: wiggle_block.initial_beta,
        });
        fit_survival_location_scale_terms(request.data, wiggle_spec, &request.kappa_options)?
    } else {
        fit_survival_location_scale_terms(request.data, request.spec, &request.kappa_options)?
    };

    Ok(SurvivalLocationScaleWorkflowResult {
        fit,
        wiggle_knots,
        wiggle_degree,
    })
}

pub fn fit_model(request: FitModelRequest<'_>) -> Result<FitModelResult, String> {
    match request {
        FitModelRequest::Standard(request) => {
            fit_standard_model(request).map(FitModelResult::Standard)
        }
        FitModelRequest::FlexibleLink(request) => {
            fit_flexible_link_model(request).map(FitModelResult::FlexibleLink)
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

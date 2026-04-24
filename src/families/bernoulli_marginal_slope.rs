use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyWarmStart,
    ExactNewtonJointGradientEvaluation, ExactNewtonJointHessianWorkspace,
    ExactNewtonJointPsiSecondOrderTerms, ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace,
    ExactOuterDerivativeOrder, FamilyEvaluation, ParameterBlockSpec, ParameterBlockState,
    build_block_spatial_psi_derivatives, cost_gated_outer_order, custom_family_outer_derivatives,
    evaluate_custom_family_joint_hyper_efs_shared, evaluate_custom_family_joint_hyper_shared,
    fit_custom_family, slice_joint_into_block_working_sets,
};
use crate::estimate::UnifiedFitResult;
use crate::estimate::reml::unified::HyperOperator;
use crate::families::gamlss::{ParameterBlockInput, initialize_monotone_wiggle_knots_from_seed};
use crate::families::lognormal_kernel::FrailtySpec;
use crate::families::row_kernel::{
    RowKernel, RowKernelHessianWorkspace, build_row_kernel_cache, row_kernel_gradient,
    row_kernel_hessian_dense, row_kernel_log_likelihood,
};
use crate::matrix::{DesignMatrix, SymmetricMatrix};
use crate::pirls::LinearInequalityConstraints;
use crate::probability::{normal_cdf, normal_pdf, standard_normal_quantile};
use crate::smooth::{
    ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionDesign, TermCollectionSpec, build_term_collection_designs_and_freeze_joint,
    optimize_spatial_length_scale_exact_joint, spatial_length_scale_term_indices,
};
use crate::types::{InverseLink, LinkFunction, WigglePenaltyConfig};
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use statrs::function::erf::erfc;
use std::cell::RefCell;
use std::sync::Arc;

mod deviation_runtime;
pub(crate) mod exact_kernel;
pub use deviation_runtime::DeviationRuntime;
#[derive(Clone, Debug)]
pub struct DeviationBlockConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_order: usize,
    pub penalty_orders: Vec<usize>,
    pub double_penalty: bool,
    pub monotonicity_eps: f64,
}

impl Default for DeviationBlockConfig {
    fn default() -> Self {
        WigglePenaltyConfig::cubic_triple_operator_default().into()
    }
}

impl DeviationBlockConfig {
    pub fn triple_penalty_default() -> Self {
        Self::default()
    }
}

impl From<WigglePenaltyConfig> for DeviationBlockConfig {
    fn from(cfg: WigglePenaltyConfig) -> Self {
        let penalty_order = *cfg.penalty_orders.iter().max().unwrap_or(&2);
        Self {
            degree: cfg.degree,
            num_internal_knots: cfg.num_internal_knots,
            penalty_order,
            penalty_orders: cfg.penalty_orders,
            double_penalty: cfg.double_penalty,
            monotonicity_eps: cfg.monotonicity_eps,
        }
    }
}

#[derive(Clone)]
pub(crate) struct DeviationPrepared {
    pub(crate) block: ParameterBlockInput,
    pub(crate) runtime: DeviationRuntime,
}

impl std::fmt::Debug for DeviationPrepared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviationPrepared").finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub struct BernoulliMarginalSlopeTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub z: Array1<f64>,
    pub base_link: InverseLink,
    pub marginalspec: TermCollectionSpec,
    pub logslopespec: TermCollectionSpec,
    pub marginal_offset: Array1<f64>,
    pub logslope_offset: Array1<f64>,
    /// GaussianShift frailty on the final probit index: U ~ N(0, σ²) added
    /// to the scalar argument of Φ.  This is exact because the sextic
    /// microcell kernel is preserved — the Gaussian-decoupling identity
    /// E[Φ(η + U)] = Φ(η / √(1+σ²)) rescales the index by 1/τ where
    /// τ = √(1+σ²), and every derivative chain rule factor is polynomial
    /// in τ, so all six kernel derivatives remain closed-form.
    ///
    /// **HazardMultiplier frailty is NOT supported in this family.**
    /// HazardMultiplier frailty + score_warp/linkwiggle cubic marginal-slope
    /// is not finite-state exact.  For hazard-multiplier frailty, use the
    /// standalone LatentCloglogBinomial / LatentSurvival families instead.
    pub frailty: FrailtySpec,
    pub score_warp: Option<DeviationBlockConfig>,
    pub link_dev: Option<DeviationBlockConfig>,
    pub latent_z_policy: LatentZPolicy,
}

impl BernoulliMarginalSlopeTermSpec {
    pub fn calibrated_probit(
        y: Array1<f64>,
        weights: Array1<f64>,
        z: Array1<f64>,
        marginalspec: TermCollectionSpec,
        logslopespec: TermCollectionSpec,
        marginal_offset: Array1<f64>,
        logslope_offset: Array1<f64>,
        frailty: FrailtySpec,
        protocol: crate::solver::protocol::MarginalSlopeCalibrationProtocol,
    ) -> Self {
        Self {
            y,
            weights,
            z,
            base_link: protocol.base_link,
            marginalspec,
            logslopespec,
            marginal_offset,
            logslope_offset,
            frailty,
            score_warp: Some(protocol.score_warp),
            link_dev: Some(protocol.link_deviation),
            latent_z_policy: protocol.latent_score.into_policy(),
        }
    }
}

pub struct BernoulliMarginalSlopeFitResult {
    pub fit: UnifiedFitResult,
    pub marginalspec_resolved: TermCollectionSpec,
    pub logslopespec_resolved: TermCollectionSpec,
    pub marginal_design: TermCollectionDesign,
    pub logslope_design: TermCollectionDesign,
    pub baseline_marginal: f64,
    pub baseline_logslope: f64,
    pub z_normalization: LatentZNormalization,
    pub score_warp_runtime: Option<DeviationRuntime>,
    pub link_dev_runtime: Option<DeviationRuntime>,
    /// Learned or fixed Gaussian-shift frailty SD.  `None` = no frailty.
    pub gaussian_frailty_sd: Option<f64>,
}

#[derive(Clone, Debug)]
pub enum LatentZCheckMode {
    Strict,
    WarnOnly,
    Off,
}

#[derive(Clone, Debug)]
pub enum LatentZNormalizationMode {
    None,
    FitWeighted,
    Frozen { mean: f64, sd: f64 },
}

#[derive(Clone, Debug)]
pub struct LatentZPolicy {
    pub check_mode: LatentZCheckMode,
    pub normalization: LatentZNormalizationMode,
    pub mean_tol_multiplier: f64,
    pub sd_tol_multiplier: f64,
    pub max_abs_skew: f64,
    pub max_abs_excess_kurtosis: f64,
}

impl LatentZPolicy {
    pub fn frozen_transformation_normal() -> Self {
        Self {
            check_mode: LatentZCheckMode::Strict,
            normalization: LatentZNormalizationMode::Frozen { mean: 0.0, sd: 1.0 },
            mean_tol_multiplier: 4.0,
            sd_tol_multiplier: 4.0,
            max_abs_skew: 2.0,
            max_abs_excess_kurtosis: 7.0,
        }
    }

    pub fn exploratory_fit_weighted() -> Self {
        Self {
            check_mode: LatentZCheckMode::WarnOnly,
            normalization: LatentZNormalizationMode::FitWeighted,
            mean_tol_multiplier: 8.0,
            sd_tol_multiplier: 8.0,
            max_abs_skew: 4.0,
            max_abs_excess_kurtosis: 20.0,
        }
    }
}

impl Default for LatentZPolicy {
    fn default() -> Self {
        Self::frozen_transformation_normal()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatentZNormalization {
    pub mean: f64,
    pub sd: f64,
}

impl LatentZNormalization {
    pub fn apply(&self, z: &Array1<f64>, context: &str) -> Result<Array1<f64>, String> {
        if !(self.mean.is_finite() && self.sd.is_finite() && self.sd > 1e-12) {
            return Err(format!(
                "{context} requires finite latent z normalization with sd > 1e-12; got mean={} sd={}",
                self.mean, self.sd
            ));
        }
        if z.iter().any(|value| !value.is_finite()) {
            return Err(format!("{context} requires finite z values"));
        }
        Ok(z.mapv(|zi| (zi - self.mean) / self.sd))
    }
}

#[derive(Clone)]
struct BernoulliMarginalSlopeFamily {
    y: Arc<Array1<f64>>,
    weights: Arc<Array1<f64>>,
    z: Arc<Array1<f64>>,
    gaussian_frailty_sd: Option<f64>,
    base_link: InverseLink,
    marginal_design: DesignMatrix,
    logslope_design: DesignMatrix,
    score_warp: Option<DeviationRuntime>,
    link_dev: Option<DeviationRuntime>,
}

#[derive(Clone, Default)]
struct ThetaHints {
    marginal_beta: Option<Array1<f64>>,
    logslope_beta: Option<Array1<f64>>,
    score_warp_beta: Option<Array1<f64>>,
    link_dev_beta: Option<Array1<f64>>,
}

pub(crate) fn build_score_warp_deviation_block_from_seed(
    seed: &Array1<f64>,
    cfg: &DeviationBlockConfig,
) -> Result<DeviationPrepared, String> {
    build_deviation_block_from_knots_and_design_seed_with_anchor(
        seed,
        seed,
        cfg,
        DeviationAnchorKind::StandardNormal,
    )
}

const BERNOULLI_LINK_PROBABILITY_EPS: f64 = 1e-12;

#[derive(Clone, Copy, Debug)]
pub(crate) struct BernoulliMarginalLinkMap {
    pub mu: f64,
    pub mu1: f64,
    pub mu2: f64,
    pub mu3: f64,
    pub mu4: f64,
    pub q: f64,
    pub q1: f64,
    pub q2: f64,
    pub q3: f64,
    pub q4: f64,
}

#[inline]
fn clamp_bernoulli_link_probability(probability: f64) -> f64 {
    probability.clamp(
        BERNOULLI_LINK_PROBABILITY_EPS,
        1.0 - BERNOULLI_LINK_PROBABILITY_EPS,
    )
}

pub(crate) fn bernoulli_marginal_slope_eta_from_probability(
    base_link: &InverseLink,
    probability: f64,
    context: &str,
) -> Result<f64, String> {
    require_probit_marginal_slope_link(base_link, context)?;
    let target = clamp_bernoulli_link_probability(probability);
    standard_normal_quantile(target)
        .map_err(|e| format!("{context} failed to invert probit probability {target}: {e}"))
}

pub(crate) fn bernoulli_marginal_link_map(
    base_link: &InverseLink,
    eta: f64,
) -> Result<BernoulliMarginalLinkMap, String> {
    require_probit_marginal_slope_link(base_link, "bernoulli marginal-slope")?;
    let phi_eta = normal_pdf(eta);
    let mu = clamp_bernoulli_link_probability(normal_cdf(eta));
    let q = standard_normal_quantile(mu).map_err(|e| {
        format!("bernoulli marginal-slope probit target inversion failed at mu={mu}: {e}")
    })?;
    let phi_q = normal_pdf(q);
    if !phi_q.is_finite() || phi_q <= 0.0 {
        return Err(format!(
            "bernoulli marginal-slope internal probit density must be positive, got phi(q)={phi_q} at eta={eta}, q={q}"
        ));
    }
    let mu1 = phi_eta;
    let mu2 = -eta * phi_eta;
    let mu3 = (eta * eta - 1.0) * phi_eta;
    let mu4 = -(eta.powi(3) - 3.0 * eta) * phi_eta;
    let q1 = mu1 / phi_q;
    let q2 = mu2 / phi_q + q * q1 * q1;
    let q3 = mu3 / phi_q + 3.0 * q * q1 * q2 - (q * q - 1.0) * q1.powi(3);
    let q4 =
        mu4 / phi_q + (q.powi(3) - 3.0 * q) * q1.powi(4) + 4.0 * q * q1 * q3 + 3.0 * q * q2 * q2
            - 6.0 * (q * q - 1.0) * q1 * q1 * q2;
    Ok(BernoulliMarginalLinkMap {
        mu,
        mu1,
        mu2,
        mu3,
        mu4,
        q,
        q1,
        q2,
        q3,
        q4,
    })
}

fn require_probit_marginal_slope_link(
    base_link: &InverseLink,
    context: &str,
) -> Result<(), String> {
    if matches!(base_link, InverseLink::Standard(LinkFunction::Probit)) {
        Ok(())
    } else {
        Err(format!(
            "{context} requires link(type=probit); non-probit marginal-slope base links are not supported by the calibrated de-nested probit kernel"
        ))
    }
}

pub(crate) fn build_link_deviation_block_from_knots_design_seed_and_weights(
    knot_seed: &Array1<f64>,
    design_seed: &Array1<f64>,
    anchor_weights: &Array1<f64>,
    cfg: &DeviationBlockConfig,
) -> Result<DeviationPrepared, String> {
    build_deviation_block_from_knots_and_design_seed_with_anchor(
        knot_seed,
        design_seed,
        cfg,
        DeviationAnchorKind::EmpiricalDesign { anchor_weights },
    )
}

enum DeviationAnchorKind<'a> {
    StandardNormal,
    EmpiricalDesign { anchor_weights: &'a Array1<f64> },
}

fn build_deviation_block_from_knots_and_design_seed_with_anchor(
    knot_seed: &Array1<f64>,
    design_seed: &Array1<f64>,
    cfg: &DeviationBlockConfig,
    anchor: DeviationAnchorKind<'_>,
) -> Result<DeviationPrepared, String> {
    if cfg.degree != 3 {
        return Err(format!(
            "structural deviation runtime is cubic; degree must be 3, got {}",
            cfg.degree
        ));
    }
    let penalty_orders = resolve_deviation_operator_orders(cfg)?;
    let knots = initialize_monotone_wiggle_knots_from_seed(
        knot_seed.view(),
        cfg.degree,
        cfg.num_internal_knots,
    )?;
    let runtime = match anchor {
        DeviationAnchorKind::StandardNormal => {
            DeviationRuntime::try_new_standard_normal_anchor(knots, cfg.monotonicity_eps)?
        }
        DeviationAnchorKind::EmpiricalDesign { anchor_weights } => {
            DeviationRuntime::try_new_weighted_empirical_anchor(
                knots,
                cfg.monotonicity_eps,
                design_seed,
                anchor_weights,
            )?
        }
    };
    let design = runtime.design(design_seed)?;
    let p = design.ncols();
    if p == 0 {
        return Err("structural deviation basis has no free derivative controls".to_string());
    }
    let mut block = ParameterBlockInput {
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(design)),
        offset: Array1::zeros(design_seed.len()),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: None,
        initial_beta: Some(Array1::zeros(p)),
    };
    for order in penalty_orders {
        append_deviation_function_penalty(&mut block, &runtime, order)?;
    }
    if cfg.double_penalty {
        append_deviation_function_penalty(&mut block, &runtime, 0)?;
    }
    Ok(DeviationPrepared { block, runtime })
}

fn resolve_deviation_operator_orders(cfg: &DeviationBlockConfig) -> Result<Vec<usize>, String> {
    let mut orders = Vec::new();
    let requested = if cfg.penalty_orders.is_empty() {
        std::slice::from_ref(&cfg.penalty_order)
    } else {
        cfg.penalty_orders.as_slice()
    };
    for &order in requested {
        if order == 0 {
            continue;
        }
        if order > cfg.degree {
            return Err(format!(
                "deviation function penalty derivative order {order} exceeds basis degree {}",
                cfg.degree
            ));
        }
        if !orders.contains(&order) {
            orders.push(order);
        }
    }
    if orders.is_empty() {
        return Err(
            "deviation block requires at least one positive function-penalty derivative order"
                .to_string(),
        );
    }
    Ok(orders)
}

fn append_deviation_function_penalty(
    block: &mut ParameterBlockInput,
    runtime: &DeviationRuntime,
    derivative_order: usize,
) -> Result<(), String> {
    let (penalty, nullity) =
        runtime.integrated_derivative_penalty_with_nullity(derivative_order)?;
    block
        .penalties
        .push(crate::solver::estimate::PenaltySpec::Dense(penalty));
    block.nullspace_dims.push(nullity);
    Ok(())
}

pub(crate) fn project_monotone_feasible_beta(
    runtime: &DeviationRuntime,
    current: &Array1<f64>,
    proposed: &Array1<f64>,
    label: &str,
) -> Result<Array1<f64>, String> {
    if current.len() != runtime.basis_dim() {
        return Err(format!(
            "{label} monotone projection current length mismatch: current={}, expected={}",
            current.len(),
            runtime.basis_dim()
        ));
    }
    if proposed.len() != runtime.basis_dim() {
        return Err(format!(
            "{label} monotone projection length mismatch: proposed={}, expected={}",
            proposed.len(),
            runtime.basis_dim()
        ));
    }
    for (idx, value) in current.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{label} current coefficient {idx} is non-finite"));
        }
    }
    for (idx, value) in proposed.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{label} coefficient {idx} is non-finite"));
        }
    }
    runtime.monotonicity_feasible(current, &format!("{label} current beta"))?;
    if runtime
        .monotonicity_feasible(proposed, &format!("{label} proposed beta"))
        .is_ok()
    {
        return Ok(proposed.clone());
    }

    let direction = proposed - current;
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        let candidate = current + &direction.mapv(|value| value * mid);
        if runtime
            .monotonicity_feasible(&candidate, &format!("{label} projected beta"))
            .is_ok()
        {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(current + &direction.mapv(|value| value * lo))
}

fn validate_spec(
    data: ArrayView2<'_, f64>,
    spec: &BernoulliMarginalSlopeTermSpec,
) -> Result<(), String> {
    let n = data.nrows();
    if spec.y.len() != n
        || spec.weights.len() != n
        || spec.z.len() != n
        || spec.marginal_offset.len() != n
        || spec.logslope_offset.len() != n
    {
        return Err(format!(
            "bernoulli-marginal-slope row mismatch: data={}, y={}, weights={}, z={}, marginal_offset={}, logslope_offset={}",
            n,
            spec.y.len(),
            spec.weights.len(),
            spec.z.len(),
            spec.marginal_offset.len(),
            spec.logslope_offset.len()
        ));
    }
    if spec
        .y
        .iter()
        .any(|&yi| !yi.is_finite() || ((yi - 0.0).abs() > 1e-9 && (yi - 1.0).abs() > 1e-9))
    {
        return Err("bernoulli-marginal-slope requires binary y in {0,1}".to_string());
    }
    if spec.weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
        return Err("bernoulli-marginal-slope requires finite non-negative weights".to_string());
    }
    if spec.z.iter().any(|&zi| !zi.is_finite()) {
        return Err("bernoulli-marginal-slope requires finite z values".to_string());
    }
    if spec.marginal_offset.iter().any(|&value| !value.is_finite()) {
        return Err("bernoulli-marginal-slope requires finite marginal offsets".to_string());
    }
    if spec.logslope_offset.iter().any(|&value| !value.is_finite()) {
        return Err("bernoulli-marginal-slope requires finite logslope offsets".to_string());
    }
    require_probit_marginal_slope_link(&spec.base_link, "bernoulli-marginal-slope")?;
    spec.frailty.validate_for_marginal_slope()?;
    match &spec.frailty {
        FrailtySpec::None => {}
        FrailtySpec::GaussianShift { sigma_fixed } => {
            if let Some(sigma) = sigma_fixed
                && (!sigma.is_finite() || *sigma < 0.0)
            {
                return Err(format!(
                    "bernoulli-marginal-slope requires GaussianShift sigma >= 0, got {sigma}"
                ));
            }
        }
        FrailtySpec::HazardMultiplier { .. } => unreachable!(),
    }
    Ok(())
}

pub(crate) fn standardize_latent_z_with_policy(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    context: &str,
    policy: &LatentZPolicy,
) -> Result<(Array1<f64>, LatentZNormalization), String> {
    if z.len() != weights.len() {
        return Err(format!(
            "{context} latent-score normalization length mismatch: z={}, weights={}",
            z.len(),
            weights.len()
        ));
    }
    let weight_sum = weights.iter().copied().sum::<f64>();
    let weight_sq_sum = weights.iter().map(|&w| w * w).sum::<f64>();
    if !(weight_sum.is_finite()
        && weight_sum > 0.0
        && weight_sq_sum.is_finite()
        && weight_sq_sum > 0.0)
    {
        return Err(format!("{context} requires positive finite total weight"));
    }
    let effective_n = weight_sum * weight_sum / weight_sq_sum;
    if !(effective_n.is_finite() && effective_n > 1.0) {
        return Err(format!(
            "{context} requires at least two effective observations for latent-score normalization"
        ));
    }
    let mean = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi)
        .sum::<f64>()
        / weight_sum;
    let var = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * (zi - mean) * (zi - mean))
        .sum::<f64>()
        / weight_sum;
    let sd = var.sqrt();
    if !(sd.is_finite() && sd > 1e-12) {
        return Err(format!(
            "{context} requires z with positive finite weighted standard deviation"
        ));
    }
    let target_norm = match policy.normalization {
        LatentZNormalizationMode::None => LatentZNormalization { mean: 0.0, sd: 1.0 },
        LatentZNormalizationMode::FitWeighted => LatentZNormalization { mean, sd },
        LatentZNormalizationMode::Frozen {
            mean: frozen_mean,
            sd: frozen_sd,
        } => LatentZNormalization {
            mean: frozen_mean,
            sd: frozen_sd,
        },
    };
    let mean_tol = policy.mean_tol_multiplier / effective_n.sqrt();
    let sd_tol = policy.sd_tol_multiplier / (2.0 * (effective_n - 1.0).max(1.0)).sqrt();
    let check_msg = || {
        format!(
            "{context} requires z to already be approximately latent N(0,1) before identification normalization; got mean={mean:.6e}, sd={sd:.6e}, effective_n={effective_n:.1}, allowed_mean={mean_tol:.3e}, allowed_sd={sd_tol:.3e}"
        )
    };
    if mean.abs() > mean_tol || (sd - 1.0).abs() > sd_tol {
        match policy.check_mode {
            LatentZCheckMode::Strict => return Err(check_msg()),
            LatentZCheckMode::WarnOnly => log::warn!("{}", check_msg()),
            LatentZCheckMode::Off => {}
        }
    }

    let normalization = target_norm;
    let z_std = normalization.apply(z, context)?;
    let skew = z_std
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi.powi(3))
        .sum::<f64>()
        / weight_sum;
    let kurt = z_std
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi.powi(4))
        .sum::<f64>()
        / weight_sum
        - 3.0;
    if skew.abs() > policy.max_abs_skew || kurt.abs() > policy.max_abs_excess_kurtosis {
        let msg = format!(
            "{context} requires z to be approximately Gaussian after identification normalization; got skewness={skew:.3}, excess_kurtosis={kurt:.3}"
        );
        match policy.check_mode {
            LatentZCheckMode::Strict => return Err(msg),
            LatentZCheckMode::WarnOnly => log::warn!("{}", msg),
            LatentZCheckMode::Off => {}
        }
    }
    if skew.abs() > 0.75 || kurt.abs() > 2.0 {
        log::warn!(
            "{context}: z has skewness={skew:.3} and excess kurtosis={kurt:.3}; the calibrated marginal-slope model assumes latent Gaussian scores"
        );
    }
    Ok((z_std, normalization))
}

pub fn padded_deviation_seed(seed: &Array1<f64>, min_iqr: f64, pad_fraction: f64) -> Array1<f64> {
    let mut sorted = seed.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if sorted.len() < 4 {
        return seed.clone();
    }

    let n = sorted.len();
    let q1 = sorted[n / 4];
    let q3 = sorted[3 * n / 4];
    let iqr = (q3 - q1).max(min_iqr);
    let pad = pad_fraction * iqr;

    let mut out = seed.to_vec();
    out.push(sorted[0] - pad);
    out.push(sorted[n - 1] + pad);
    Array1::from_vec(out)
}

fn pooled_probit_baseline(
    y: &Array1<f64>,
    z: &Array1<f64>,
    weights: &Array1<f64>,
) -> Result<(f64, f64), String> {
    if y.len() != z.len() || y.len() != weights.len() {
        return Err(format!(
            "pooled bernoulli-marginal-slope pilot length mismatch: y={}, z={}, weights={}",
            y.len(),
            z.len(),
            weights.len()
        ));
    }
    let weight_sum = weights.iter().copied().sum::<f64>();
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(
            "pooled bernoulli-marginal-slope pilot requires positive finite total weight"
                .to_string(),
        );
    }
    let prevalence = y
        .iter()
        .zip(weights.iter())
        .map(|(&yi, &wi)| yi * wi)
        .sum::<f64>()
        / weight_sum;
    let prevalence = prevalence.clamp(1e-6, 1.0 - 1e-6);
    let z_mean = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| zi * wi)
        .sum::<f64>()
        / weight_sum;
    let z_var = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * (zi - z_mean) * (zi - z_mean))
        .sum::<f64>()
        / weight_sum;
    let yz_cov = y
        .iter()
        .zip(z.iter())
        .zip(weights.iter())
        .map(|((&yi, &zi), &wi)| wi * (yi - prevalence) * (zi - z_mean))
        .sum::<f64>()
        / weight_sum;
    let mut beta0 = standard_normal_quantile(prevalence).map_err(|e| {
        format!("failed to initialize pooled bernoulli-marginal-slope pilot intercept: {e}")
    })?;
    let mut beta1 = if z_var > 1e-12 { yz_cov / z_var } else { 0.0 };

    let objective_grad_hess =
        |intercept: f64, slope: f64| -> Result<(f64, f64, f64, f64, f64, f64), String> {
            let mut obj = 0.0;
            let mut g0 = 0.0;
            let mut g1 = 0.0;
            let mut h00 = 0.0;
            let mut h01 = 0.0;
            let mut h11 = 0.0;
            for ((&yi, &zi), &wi) in y.iter().zip(z.iter()).zip(weights.iter()) {
                if wi == 0.0 {
                    continue;
                }
                let eta = intercept + slope * zi;
                let s = 2.0 * yi - 1.0;
                let margin = s * eta;
                let (logcdf, lambda) = signed_probit_logcdf_and_mills_ratio(margin);
                let g_eta = -wi * s * lambda;
                let h_eta = wi * lambda * (margin + lambda);
                obj -= wi * logcdf;
                g0 += g_eta;
                g1 += g_eta * zi;
                h00 += h_eta;
                h01 += h_eta * zi;
                h11 += h_eta * zi * zi;
            }
            Ok((obj, g0, g1, h00, h01, h11))
        };

    let mut obj_prev = f64::INFINITY;
    for _ in 0..50 {
        let (obj, g0, g1, h00, h01, h11) = objective_grad_hess(beta0, beta1)?;
        if !obj.is_finite() || !g0.is_finite() || !g1.is_finite() {
            return Err(
                "pooled bernoulli-marginal-slope pilot produced non-finite objective or gradient"
                    .to_string(),
            );
        }
        let grad_max = g0.abs().max(g1.abs());
        if grad_max < 1e-8 {
            break;
        }
        let mut ridge = 1e-8;
        let (step0, step1) = loop {
            let h00_r = h00 + ridge;
            let h11_r = h11 + ridge;
            let det = h00_r * h11_r - h01 * h01;
            if det.is_finite() && det.abs() > 1e-18 {
                let s0 = (h11_r * g0 - h01 * g1) / det;
                let s1 = (-h01 * g0 + h00_r * g1) / det;
                if s0.is_finite() && s1.is_finite() {
                    break (s0, s1);
                }
            }
            ridge *= 10.0;
            if ridge > 1e6 {
                return Err(
                    "pooled bernoulli-marginal-slope pilot Hessian solve failed".to_string()
                );
            }
        };
        let mut accepted = false;
        let mut step_scale = 1.0;
        for _ in 0..25 {
            let cand0 = beta0 - step_scale * step0;
            let cand1 = beta1 - step_scale * step1;
            let (cand_obj, _, _, _, _, _) = objective_grad_hess(cand0, cand1)?;
            if cand_obj.is_finite() && cand_obj <= obj {
                beta0 = cand0;
                beta1 = cand1;
                obj_prev = cand_obj;
                accepted = true;
                break;
            }
            step_scale *= 0.5;
        }
        if !accepted {
            if (obj_prev - obj).abs() < 1e-10 {
                break;
            }
            return Err("pooled bernoulli-marginal-slope pilot line search failed".to_string());
        }
    }
    let a = beta0;
    // Signed slope: preserve direction from pilot probit.
    let b = if beta1.abs() < 1e-6 {
        if beta1.is_sign_negative() {
            -1e-6
        } else {
            1e-6
        }
    } else {
        beta1
    };
    Ok((a / (1.0 + b * b).sqrt(), b))
}

fn joint_setup(
    data: ArrayView2<'_, f64>,
    marginalspec: &TermCollectionSpec,
    logslopespec: &TermCollectionSpec,
    marginal_penalties: usize,
    logslope_penalties: usize,
    extra_rho0: &[f64],
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> ExactJointHyperSetup {
    let marginal_terms = spatial_length_scale_term_indices(marginalspec);
    let logslope_terms = spatial_length_scale_term_indices(logslopespec);
    let rho_dim = marginal_penalties + logslope_penalties + extra_rho0.len();
    let mut rho0vec = Array1::<f64>::zeros(rho_dim);
    for (idx, &value) in extra_rho0.iter().enumerate() {
        rho0vec[marginal_penalties + logslope_penalties + idx] = value;
    }
    let rho_lower = Array1::<f64>::from_elem(rho_dim, -12.0);
    let rho_upper = Array1::<f64>::from_elem(rho_dim, 12.0);
    let marginal_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        marginalspec,
        &marginal_terms,
        kappa_options,
    )
    .reseed_from_data(data, marginalspec, &marginal_terms, kappa_options);
    let logslope_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        logslopespec,
        &logslope_terms,
        kappa_options,
    )
    .reseed_from_data(data, logslopespec, &logslope_terms, kappa_options);
    let mut values = marginal_kappa.as_array().to_vec();
    values.extend(logslope_kappa.as_array().iter());
    let marginal_dims = marginal_kappa.dims_per_term().to_vec();
    let logslope_dims = logslope_kappa.dims_per_term().to_vec();
    let mut dims = marginal_dims.clone();
    dims.extend(logslope_dims.iter().copied());
    let log_kappa0 = SpatialLogKappaCoords::new_with_dims(Array1::from_vec(values), dims.clone());
    // Bounds: concatenate per-block data-aware bounds in the same order.
    let marginal_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        data,
        marginalspec,
        &marginal_terms,
        &marginal_dims,
        kappa_options,
    );
    let logslope_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        data,
        logslopespec,
        &logslope_terms,
        &logslope_dims,
        kappa_options,
    );
    let mut lower_vals = marginal_lower.as_array().to_vec();
    lower_vals.extend(logslope_lower.as_array().iter());
    let log_kappa_lower =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(lower_vals), dims.clone());
    let marginal_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        data,
        marginalspec,
        &marginal_terms,
        &marginal_dims,
        kappa_options,
    );
    let logslope_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        data,
        logslopespec,
        &logslope_terms,
        &logslope_dims,
        kappa_options,
    );
    let mut upper_vals = marginal_upper.as_array().to_vec();
    upper_vals.extend(logslope_upper.as_array().iter());
    let log_kappa_upper = SpatialLogKappaCoords::new_with_dims(Array1::from_vec(upper_vals), dims);
    // Project seed onto bounds in case a user-provided spec.length_scale falls
    // outside the data-derived ψ window; seed was a hint, not a hard constraint.
    let log_kappa0 = log_kappa0.clamp_to_bounds(&log_kappa_lower, &log_kappa_upper);
    ExactJointHyperSetup::new(
        rho0vec,
        rho_lower,
        rho_upper,
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
    )
}

#[inline]
pub(crate) fn erfcx_nonnegative(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_sign_positive() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x < 26.0 {
        ((x * x).min(700.0)).exp() * erfc(x)
    } else {
        let inv = 1.0 / x;
        let inv2 = inv * inv;
        let poly = 1.0
            + 0.5 * inv2
            + 0.75 * inv2 * inv2
            + 1.875 * inv2 * inv2 * inv2
            + 6.5625 * inv2 * inv2 * inv2 * inv2;
        inv * poly / std::f64::consts::PI.sqrt()
    }
}

#[inline]
pub(crate) fn signed_probit_logcdf_and_mills_ratio(x: f64) -> (f64, f64) {
    if x == f64::INFINITY {
        return (0.0, 0.0);
    }
    if x == f64::NEG_INFINITY {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if x < 0.0 {
        let u = -x / std::f64::consts::SQRT_2;
        let ex = erfcx_nonnegative(u).max(1e-300);
        let log_cdf = -u * u + (0.5 * ex).ln();
        let lambda = (2.0 / std::f64::consts::PI).sqrt() / ex;
        (log_cdf, lambda)
    } else {
        let cdf = normal_cdf(x).clamp(1e-300, 1.0);
        let lambda = normal_pdf(x) / cdf;
        (cdf.ln(), lambda)
    }
}

#[inline]
fn signed_probit_neglog_derivatives_up_to_fourth_numeric(
    signed_margin: f64,
    weight: f64,
) -> (f64, f64, f64, f64) {
    if weight == 0.0 || signed_margin == f64::INFINITY {
        return (0.0, 0.0, 0.0, 0.0);
    }
    if signed_margin == f64::NEG_INFINITY {
        return (f64::NEG_INFINITY, weight, 0.0, 0.0);
    }
    if signed_margin.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let (_, lambda) = signed_probit_logcdf_and_mills_ratio(signed_margin);
    let k1 = -lambda;
    let k2 = lambda * (signed_margin + lambda);
    let k3 = lambda
        * (1.0
            - signed_margin * signed_margin
            - 3.0 * signed_margin * lambda
            - 2.0 * lambda * lambda);
    let k4 = lambda
        * ((signed_margin.powi(3) - 3.0 * signed_margin)
            + (7.0 * signed_margin * signed_margin - 4.0) * lambda
            + 12.0 * signed_margin * lambda * lambda
            + 6.0 * lambda.powi(3));
    (weight * k1, weight * k2, weight * k3, weight * k4)
}

/// Exact probit derivative helper used by analytic jet code paths.
///
/// `+inf` is the saturated zero tail and is allowed. `-inf` and `NaN` are
/// rejected instead of being silently collapsed, so exact callers fail fast
/// rather than erasing curvature or domain errors. Numeric boundary behavior
/// that needs to preserve `-inf` / `NaN` values lives in
/// `signed_probit_neglog_derivatives_up_to_fourth_numeric`.
pub(crate) fn signed_probit_neglog_derivatives_up_to_fourth(
    signed_margin: f64,
    weight: f64,
) -> Result<(f64, f64, f64, f64), String> {
    if weight == 0.0 || signed_margin == f64::INFINITY {
        return Ok((0.0, 0.0, 0.0, 0.0));
    }
    if !signed_margin.is_finite() {
        return Err(format!(
            "non-finite signed margin in exact probit derivative helper: {signed_margin}"
        ));
    }
    Ok(signed_probit_neglog_derivatives_up_to_fourth_numeric(
        signed_margin,
        weight,
    ))
}

#[inline]
fn rigid_observed_logslope(logslope: f64, probit_scale: f64) -> f64 {
    probit_scale * logslope
}

#[inline]
fn rigid_observed_scale(logslope: f64, probit_scale: f64) -> f64 {
    let observed_logslope = rigid_observed_logslope(logslope, probit_scale);
    (1.0 + observed_logslope * observed_logslope).sqrt()
}

#[inline]
fn rigid_intercept_from_marginal(marginal_eta: f64, logslope: f64, probit_scale: f64) -> f64 {
    marginal_eta * rigid_observed_scale(logslope, probit_scale)
}

#[inline]
fn rigid_observed_eta(marginal_eta: f64, logslope: f64, z: f64, probit_scale: f64) -> f64 {
    rigid_intercept_from_marginal(marginal_eta, logslope, probit_scale)
        + rigid_observed_logslope(logslope, probit_scale) * z
}

/// Rigid probit scalar kernel: closed-form derivatives up to 4th order.
///
/// η = q·c(g) + s_f·g·z,  c(g) = √(1+(s_f g)²),  s = 2y−1,  m = s·η.
/// u_k absorb weight and sign: u1=w·s·κ₁, u2=w·κ₂, u3=w·s·κ₃, u4=w·κ₄.
struct RigidProbitKernel {
    logcdf: f64,
    u1: f64,
    u2: f64,
    u3: f64,
    u4: f64,
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    eta_q: f64,
    eta_g: f64,
}

impl RigidProbitKernel {
    #[inline]
    fn new(q: f64, g: f64, z: f64, y: f64, w: f64, probit_scale: f64) -> Result<Self, String> {
        let s = 2.0 * y - 1.0;
        let observed_logslope = rigid_observed_logslope(g, probit_scale);
        let g2 = observed_logslope * observed_logslope;
        let c = (1.0 + g2).sqrt();
        let c1 = probit_scale * observed_logslope / c;
        let c_inv3 = 1.0 / (c * c * c);
        let c_inv5 = c_inv3 / (c * c);
        let c_inv7 = c_inv5 / (c * c);
        let eta = q * c + observed_logslope * z;
        let m = s * eta;
        let (logcdf, _) = signed_probit_logcdf_and_mills_ratio(m);
        let (k1, k2, k3, k4) = signed_probit_neglog_derivatives_up_to_fourth(m, w)?;
        Ok(Self {
            logcdf,
            u1: s * k1,
            u2: k2,
            u3: s * k3,
            u4: k4,
            c1,
            c2: probit_scale * probit_scale * c_inv3,
            c3: -3.0 * probit_scale.powi(3) * observed_logslope * c_inv5,
            c4: probit_scale.powi(4) * (12.0 * g2 - 3.0) * c_inv7,
            eta_q: c,
            eta_g: q * c1 + probit_scale * z,
        })
    }

    #[inline]
    fn primary_hessian(&self, q: f64) -> [[f64; 2]; 2] {
        let h00 = self.u2 * self.eta_q * self.eta_q;
        let h01 = self.u2 * self.eta_q * self.eta_g + self.u1 * self.c1;
        let h11 = self.u2 * self.eta_g * self.eta_g + self.u1 * q * self.c2;
        [[h00, h01], [h01, h11]]
    }

    #[inline]
    fn third_contracted(&self, q: f64, dq: f64, dg: f64) -> [[f64; 2]; 2] {
        let dd = self.eta_q * dq + self.eta_g * dg;
        let dd_q = self.c1 * dg;
        let dd_g = self.c1 * dq + q * self.c2 * dg;
        let dd_qg = self.c2 * dg;
        let dd_gg = self.c2 * dq + q * self.c3 * dg;
        let t00 = self.u3 * self.eta_q * self.eta_q * dd + self.u2 * 2.0 * self.eta_q * dd_q;
        let t01 = self.u3 * self.eta_q * self.eta_g * dd
            + self.u2 * (self.c1 * dd + self.eta_q * dd_g + self.eta_g * dd_q)
            + self.u1 * dd_qg;
        let t11 = self.u3 * self.eta_g * self.eta_g * dd
            + self.u2 * (q * self.c2 * dd + 2.0 * self.eta_g * dd_g)
            + self.u1 * dd_gg;
        [[t00, t01], [t01, t11]]
    }

    #[inline]
    fn fourth_contracted(&self, q: f64, uq: f64, ug: f64, vq: f64, vg: f64) -> [[f64; 2]; 2] {
        let du = self.eta_q * uq + self.eta_g * ug;
        let dv = self.eta_q * vq + self.eta_g * vg;
        let du_a = [self.c1 * ug, self.c1 * uq + q * self.c2 * ug];
        let dv_a = [self.c1 * vg, self.c1 * vq + q * self.c2 * vg];
        let du_ab = [
            [0.0, self.c2 * ug],
            [self.c2 * ug, self.c2 * uq + q * self.c3 * ug],
        ];
        let dv_ab = [
            [0.0, self.c2 * vg],
            [self.c2 * vg, self.c2 * vq + q * self.c3 * vg],
        ];
        let dduv = self.c1 * (uq * vg + ug * vq) + q * self.c2 * ug * vg;
        let dduv_a = [
            self.c2 * ug * vg,
            self.c2 * (uq * vg + ug * vq) + q * self.c3 * ug * vg,
        ];
        let dduv_ab = [
            [0.0, self.c3 * ug * vg],
            [
                self.c3 * ug * vg,
                self.c3 * (uq * vg + ug * vq) + q * self.c4 * ug * vg,
            ],
        ];
        let eta_a = [self.eta_q, self.eta_g];
        let eta_ab = [[0.0, self.c1], [self.c1, q * self.c2]];
        let mut f = [[0.0f64; 2]; 2];
        for a in 0..2 {
            for b in a..2 {
                let val = self.u4 * eta_a[a] * eta_a[b] * du * dv
                    + self.u3
                        * (eta_ab[a][b] * du * dv
                            + du_a[a] * eta_a[b] * dv
                            + dv_a[a] * eta_a[b] * du
                            + du_a[b] * eta_a[a] * dv
                            + dv_a[b] * eta_a[a] * du
                            + dduv * eta_a[a] * eta_a[b])
                    + self.u2
                        * (eta_ab[a][b] * dduv
                            + du_a[a] * dv_a[b]
                            + dv_a[a] * du_a[b]
                            + du_ab[a][b] * dv
                            + dv_ab[a][b] * du
                            + eta_a[b] * dduv_a[a]
                            + eta_a[a] * dduv_a[b])
                    + self.u1 * dduv_ab[a][b];
                f[a][b] = val;
                f[b][a] = val;
            }
        }
        f
    }
}

#[inline]
fn rigid_transformed_gradient(
    marginal: BernoulliMarginalLinkMap,
    kernel: &RigidProbitKernel,
) -> [f64; 2] {
    [
        -kernel.u1 * kernel.eta_q * marginal.q1,
        -kernel.u1 * kernel.eta_g,
    ]
}

#[inline]
fn rigid_transformed_hessian(
    marginal: BernoulliMarginalLinkMap,
    kernel: &RigidProbitKernel,
) -> [[f64; 2]; 2] {
    let h_q = kernel.primary_hessian(marginal.q);
    [
        [
            h_q[0][0] * marginal.q1 * marginal.q1 + (-kernel.u1 * kernel.eta_q) * marginal.q2,
            h_q[0][1] * marginal.q1,
        ],
        [h_q[1][0] * marginal.q1, h_q[1][1]],
    ]
}

#[inline]
fn rigid_internal_third_components(
    marginal: BernoulliMarginalLinkMap,
    kernel: &RigidProbitKernel,
) -> (f64, f64, f64, f64) {
    let q_dir = kernel.third_contracted(marginal.q, 1.0, 0.0);
    let g_dir = kernel.third_contracted(marginal.q, 0.0, 1.0);
    (q_dir[0][0], q_dir[0][1], q_dir[1][1], g_dir[1][1])
}

#[inline]
fn rigid_transformed_third_contracted(
    marginal: BernoulliMarginalLinkMap,
    kernel: &RigidProbitKernel,
    d_eta: f64,
    d_g: f64,
) -> [[f64; 2]; 2] {
    let h_q = kernel.primary_hessian(marginal.q);
    let grad_q = -kernel.u1 * kernel.eta_q;
    let (f_qqq, f_qqg, f_qgg, f_ggg) = rigid_internal_third_components(marginal, kernel);
    let f_etaetaeta = f_qqq * marginal.q1.powi(3)
        + 3.0 * h_q[0][0] * marginal.q1 * marginal.q2
        + grad_q * marginal.q3;
    let f_etaetag = f_qqg * marginal.q1 * marginal.q1 + h_q[0][1] * marginal.q2;
    let f_etagg = f_qgg * marginal.q1;
    [
        [
            f_etaetaeta * d_eta + f_etaetag * d_g,
            f_etaetag * d_eta + f_etagg * d_g,
        ],
        [
            f_etaetag * d_eta + f_etagg * d_g,
            f_etagg * d_eta + f_ggg * d_g,
        ],
    ]
}

#[inline]
fn rigid_transformed_fourth_contracted(
    marginal: BernoulliMarginalLinkMap,
    kernel: &RigidProbitKernel,
    u_eta: f64,
    u_g: f64,
    v_eta: f64,
    v_g: f64,
) -> [[f64; 2]; 2] {
    let h_q = kernel.primary_hessian(marginal.q);
    let grad_q = -kernel.u1 * kernel.eta_q;
    let (f_qqq, f_qqg, f_qgg, _) = rigid_internal_third_components(marginal, kernel);
    let qq = kernel.fourth_contracted(marginal.q, 1.0, 0.0, 1.0, 0.0);
    let qg = kernel.fourth_contracted(marginal.q, 1.0, 0.0, 0.0, 1.0);
    let gg = kernel.fourth_contracted(marginal.q, 0.0, 1.0, 0.0, 1.0);
    let f_qqqq = qq[0][0];
    let f_qqqg = qq[0][1];
    let f_qqgg = qq[1][1];
    let f_qggg = qg[1][1];
    let f_gggg = gg[1][1];
    let f_eta4 = f_qqqq * marginal.q1.powi(4)
        + 6.0 * f_qqq * marginal.q1 * marginal.q1 * marginal.q2
        + 3.0 * h_q[0][0] * marginal.q2 * marginal.q2
        + 4.0 * h_q[0][0] * marginal.q1 * marginal.q3
        + grad_q * marginal.q4;
    let f_eta3g = f_qqqg * marginal.q1.powi(3)
        + 3.0 * f_qqg * marginal.q1 * marginal.q2
        + h_q[0][1] * marginal.q3;
    let f_eta2g2 = f_qqgg * marginal.q1 * marginal.q1 + f_qgg * marginal.q2;
    let f_etag3 = f_qggg * marginal.q1;
    [
        [
            f_eta4 * u_eta * v_eta + f_eta3g * (u_eta * v_g + u_g * v_eta) + f_eta2g2 * u_g * v_g,
            f_eta3g * u_eta * v_eta + f_eta2g2 * (u_eta * v_g + u_g * v_eta) + f_etag3 * u_g * v_g,
        ],
        [
            f_eta3g * u_eta * v_eta + f_eta2g2 * (u_eta * v_g + u_g * v_eta) + f_etag3 * u_g * v_g,
            f_eta2g2 * u_eta * v_eta + f_etag3 * (u_eta * v_g + u_g * v_eta) + f_gggg * u_g * v_g,
        ],
    ]
}

pub(crate) fn unary_derivatives_sqrt(x: f64) -> [f64; 5] {
    let s = x.max(1e-300).sqrt();
    let x1 = x.max(1e-300);
    let x2 = x1 * x1;
    let x3 = x2 * x1;
    [
        s,
        0.5 / s,
        -0.25 / (x1 * s),
        3.0 / (8.0 * x2 * s),
        -15.0 / (16.0 * x3 * s),
    ]
}
pub(crate) fn unary_derivatives_neglog_phi(x: f64, weight: f64) -> [f64; 5] {
    if weight == 0.0 || x == f64::INFINITY {
        return [0.0, 0.0, 0.0, 0.0, 0.0];
    }
    if x == f64::NEG_INFINITY {
        return [f64::INFINITY, f64::NEG_INFINITY, weight, 0.0, 0.0];
    }
    if x.is_nan() {
        return [f64::NAN; 5];
    }
    let (d1, d2, d3, d4) = signed_probit_neglog_derivatives_up_to_fourth_numeric(x, weight);
    let (log_cdf, _) = signed_probit_logcdf_and_mills_ratio(x);
    [-weight * log_cdf, d1, d2, d3, d4]
}

/// Derivatives of log(x) through 4th order.
pub(crate) fn unary_derivatives_log(x: f64) -> [f64; 5] {
    let x1 = x.max(1e-300);
    let x2 = x1 * x1;
    let x3 = x2 * x1;
    let x4 = x3 * x1;
    [x1.ln(), 1.0 / x1, -1.0 / x2, 2.0 / x3, -6.0 / x4]
}

/// Derivatives of log φ(x) = -½x² - ½ln(2π) through 4th order.
pub(crate) fn unary_derivatives_log_normal_pdf(x: f64) -> [f64; 5] {
    let c = 0.5 * (2.0 * std::f64::consts::PI).ln();
    [-0.5 * x * x - c, -x, -1.0, 0.0, 0.0]
}
/// Block-local psi derivative row: avoids allocating a full p-vector
/// when the psi derivative lives in a single channel (marginal or logslope).
struct BlockPsiRow {
    /// Which parameter block (0 = marginal, 1 = logslope).
    block_idx: usize,
    /// Coefficient range in global (flat) space for this block.
    range: std::ops::Range<usize>,
    /// The p_block-length psi design derivative row.
    local_vec: Array1<f64>,
}

#[derive(Clone)]
struct BlockSlices {
    marginal: std::ops::Range<usize>,
    logslope: std::ops::Range<usize>,
    h: Option<std::ops::Range<usize>>,
    w: Option<std::ops::Range<usize>>,
    total: usize,
}

fn block_slices(family: &BernoulliMarginalSlopeFamily) -> BlockSlices {
    let mut cursor = 0usize;
    let marginal = cursor..cursor + family.marginal_design.ncols();
    cursor = marginal.end;
    let logslope = cursor..cursor + family.logslope_design.ncols();
    cursor = logslope.end;
    let h = family.score_warp.as_ref().map(|runtime| {
        let range = cursor..cursor + runtime.basis_dim();
        cursor = range.end;
        range
    });
    let w = family.link_dev.as_ref().map(|runtime| {
        let range = cursor..cursor + runtime.basis_dim();
        cursor = range.end;
        range
    });
    BlockSlices {
        marginal,
        logslope,
        h,
        w,
        total: cursor,
    }
}

#[derive(Clone)]
struct PrimarySlices {
    q: usize,
    logslope: usize,
    h: Option<std::ops::Range<usize>>,
    w: Option<std::ops::Range<usize>>,
    total: usize,
}

fn primary_slices(slices: &BlockSlices) -> PrimarySlices {
    let q = 0usize;
    let logslope = 1usize;
    let mut cursor = 2usize;
    let h = slices.h.as_ref().map(|range| {
        let out = cursor..cursor + range.len();
        cursor = out.end;
        out
    });
    let w = slices.w.as_ref().map(|range| {
        let out = cursor..cursor + range.len();
        cursor = out.end;
        out
    });
    PrimarySlices {
        q,
        logslope,
        h,
        w,
        total: cursor,
    }
}
// ── Block-local Hessian accumulator for Bernoulli marginal-slope ─────
//
// The two large blocks are marginal (p_m) and logslope (p_g).
// Optional h/w blocks are tiny (1-5 params each), so their contributions
// go into a dense p_total x p_total correction matrix.  The main savings
// is avoiding O(n * (p_m^2 + p_g^2)) dense accumulation into a full p*p target.

struct BernoulliBlockHessianAccumulator {
    h_mm: Array2<f64>,
    h_gg: Array2<f64>,
    h_mg: Array2<f64>,
    dense_correction: Option<Array2<f64>>,
}

impl BernoulliBlockHessianAccumulator {
    fn new(slices: &BlockSlices) -> Self {
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let has_hw = slices.h.is_some() || slices.w.is_some();
        Self {
            h_mm: Array2::zeros((p_m, p_m)),
            h_gg: Array2::zeros((p_g, p_g)),
            h_mg: Array2::zeros((p_m, p_g)),
            dense_correction: if has_hw {
                Some(Array2::zeros((slices.total, slices.total)))
            } else {
                None
            },
        }
    }

    /// Accumulate a primary-space Hessian into block-local matrices.
    /// The marginal block uses H[0,0], logslope uses H[1,1],
    /// cross uses H[0,1].  All h/w cross-blocks go to dense_correction.
    fn add_pullback(
        &mut self,
        family: &BernoulliMarginalSlopeFamily,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        primary_hessian: &Array2<f64>,
    ) {
        let h = primary_hessian;

        // marginal x marginal: H[0,0] * x_row outer x_row
        family
            .marginal_design
            .syr_row_into(row, h[[0, 0]], &mut self.h_mm)
            .expect("marginal syr_row_into dimension mismatch");

        // logslope x logslope: H[1,1] * g_row outer g_row
        family
            .logslope_design
            .syr_row_into(row, h[[1, 1]], &mut self.h_gg)
            .expect("logslope syr_row_into dimension mismatch");

        // marginal x logslope: H[0,1] * x_row outer g_row
        if h[[0, 1]] != 0.0 {
            family
                .marginal_design
                .row_outer_into_view(
                    row,
                    &family.logslope_design,
                    h[[0, 1]],
                    self.h_mg.view_mut(),
                )
                .expect("marginal-logslope row_outer_into dimension mismatch");
        }

        // h/w cross-blocks -> dense_correction
        if let Some(ref mut dc) = self.dense_correction {
            family.add_pullback_primary_hessian_hw_only(dc, row, slices, primary, h);
        }
    }

    /// Add a rank-1 update from psi_row (in the psi block) crossed with the
    /// pullback of a primary-space vector.  Adds both left outer right and right outer left.
    ///
    /// psi_row lives in block `psi_block_idx` (0=marginal, 1=logslope).
    /// right_primary is a primary-space vector; its [0] component maps to marginal,
    /// [1] to logslope, and the rest to h/w (dense correction).
    fn add_rank1_psi_cross(
        &mut self,
        family: &BernoulliMarginalSlopeFamily,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        psi_block_idx: usize,
        psi_row: &Array1<f64>,
        right_primary: &Array1<f64>,
    ) {
        // Marginal component of right_primary
        if right_primary[0] != 0.0 {
            match psi_block_idx {
                0 => {
                    // psi=marginal, right=marginal -> h_mm, symmetric rank-2
                    for (idx, &value) in psi_row.iter().enumerate() {
                        if value == 0.0 {
                            continue;
                        }
                        let scale = right_primary[0] * value;
                        {
                            let mut col = self.h_mm.column_mut(idx);
                            family
                                .marginal_design
                                .axpy_row_into(row, scale, &mut col)
                                .expect("marginal axpy column mismatch");
                        }
                        {
                            let mut row_view = self.h_mm.row_mut(idx);
                            family
                                .marginal_design
                                .axpy_row_into(row, scale, &mut row_view)
                                .expect("marginal axpy row mismatch");
                        }
                    }
                }
                1 => {
                    // psi=logslope, right=marginal -> h_mg (marginal x logslope)
                    for (idx, &value) in psi_row.iter().enumerate() {
                        if value == 0.0 {
                            continue;
                        }
                        let mut col = self.h_mg.column_mut(idx);
                        family
                            .marginal_design
                            .axpy_row_into(row, right_primary[0] * value, &mut col)
                            .expect("marginal axpy column mismatch");
                    }
                }
                _ => {}
            }
        }

        // Logslope component of right_primary
        if right_primary[1] != 0.0 {
            match psi_block_idx {
                0 => {
                    // psi=marginal, right=logslope -> h_mg
                    for (idx, &value) in psi_row.iter().enumerate() {
                        if value == 0.0 {
                            continue;
                        }
                        let mut row_view = self.h_mg.row_mut(idx);
                        family
                            .logslope_design
                            .axpy_row_into(row, right_primary[1] * value, &mut row_view)
                            .expect("logslope axpy row mismatch");
                    }
                }
                1 => {
                    // psi=logslope, right=logslope -> h_gg, symmetric rank-2
                    for (idx, &value) in psi_row.iter().enumerate() {
                        if value == 0.0 {
                            continue;
                        }
                        let scale = right_primary[1] * value;
                        {
                            let mut col = self.h_gg.column_mut(idx);
                            family
                                .logslope_design
                                .axpy_row_into(row, scale, &mut col)
                                .expect("logslope axpy column mismatch");
                        }
                        {
                            let mut row_view = self.h_gg.row_mut(idx);
                            family
                                .logslope_design
                                .axpy_row_into(row, scale, &mut row_view)
                                .expect("logslope axpy row mismatch");
                        }
                    }
                }
                _ => {}
            }
        }

        // h/w components -> dense_correction
        if let Some(ref mut dc) = self.dense_correction {
            let psi_range = if psi_block_idx == 0 {
                slices.marginal.clone()
            } else {
                slices.logslope.clone()
            };
            if let (Some(ph), Some(bh)) = (primary.h.as_ref(), slices.h.as_ref()) {
                let h_part = right_primary.slice(ndarray::s![ph.start..ph.end]);
                for (li, gi) in psi_range.clone().enumerate() {
                    for (lj, gj) in bh.clone().enumerate() {
                        let val = psi_row[li] * h_part[lj];
                        dc[[gi, gj]] += val;
                        dc[[gj, gi]] += val;
                    }
                }
            }
            if let (Some(pw), Some(bw)) = (primary.w.as_ref(), slices.w.as_ref()) {
                let w_part = right_primary.slice(ndarray::s![pw.start..pw.end]);
                for (li, gi) in psi_range.enumerate() {
                    for (lj, gj) in bw.clone().enumerate() {
                        let val = psi_row[li] * w_part[lj];
                        dc[[gi, gj]] += val;
                        dc[[gj, gi]] += val;
                    }
                }
            }
        }
    }

    /// Add outer product of two psi block-local rows (possibly in different blocks).
    /// Adds both alpha * (a outer b) and alpha * (b outer a) to maintain symmetry.
    fn add_psi_psi_outer(
        &mut self,
        block_i: usize,
        psi_row_i: &Array1<f64>,
        block_j: usize,
        psi_row_j: &Array1<f64>,
        alpha: f64,
    ) {
        if alpha == 0.0 {
            return;
        }
        let col_i = psi_row_i.view().insert_axis(Axis(1));
        let row_j = psi_row_j.view().insert_axis(Axis(0));

        if block_i == block_j {
            // Same block: symmetric rank-2 update to diagonal block
            let col_j = psi_row_j.view().insert_axis(Axis(1));
            let row_i = psi_row_i.view().insert_axis(Axis(0));
            let target = match block_i {
                0 => &mut self.h_mm,
                1 => &mut self.h_gg,
                _ => return,
            };
            ndarray::linalg::general_mat_mul(alpha, &col_i, &row_j, 1.0, target);
            ndarray::linalg::general_mat_mul(alpha, &col_j, &row_i, 1.0, target);
        } else {
            // Different blocks: one rank-1 update to h_mg.
            // h_mg = marginal x logslope; the transpose is assembled automatically.
            let (marginal_row, logslope_row) = if block_i == 0 {
                (psi_row_i, psi_row_j)
            } else {
                (psi_row_j, psi_row_i)
            };
            let m_col = marginal_row.view().insert_axis(Axis(1));
            let g_row = logslope_row.view().insert_axis(Axis(0));
            ndarray::linalg::general_mat_mul(alpha, &m_col, &g_row, 1.0, &mut self.h_mg);
        }
    }

    /// Merge another accumulator into this one (for parallel reduce).
    fn add(&mut self, other: &BernoulliBlockHessianAccumulator) {
        self.h_mm += &other.h_mm;
        self.h_gg += &other.h_gg;
        self.h_mg += &other.h_mg;
        if let (Some(ref mut dc), Some(ref odc)) = (
            self.dense_correction.as_mut(),
            other.dense_correction.as_ref(),
        ) {
            dc.scaled_add(1.0, odc);
        }
    }

    fn to_dense(&self, slices: &BlockSlices) -> Array2<f64> {
        let mut out = Array2::zeros((slices.total, slices.total));
        out.slice_mut(s![slices.marginal.clone(), slices.marginal.clone()])
            .assign(&self.h_mm);
        out.slice_mut(s![slices.logslope.clone(), slices.logslope.clone()])
            .assign(&self.h_gg);
        out.slice_mut(s![slices.marginal.clone(), slices.logslope.clone()])
            .assign(&self.h_mg);
        out.slice_mut(s![slices.logslope.clone(), slices.marginal.clone()])
            .assign(&self.h_mg.t());
        if let Some(ref dc) = self.dense_correction {
            out += dc;
        }
        out
    }

    fn into_operator(self, slices: &BlockSlices) -> BernoulliBlockHessianOperator {
        BernoulliBlockHessianOperator {
            h_mm: self.h_mm,
            h_gg: self.h_gg,
            h_mg: self.h_mg,
            dense_correction: self.dense_correction,
            marginal: slices.marginal.clone(),
            logslope: slices.logslope.clone(),
            total: slices.total,
        }
    }
}

/// Block-structured HyperOperator for Bernoulli marginal-slope psi Hessians.
/// Stores 3 block matrices (h_mm, h_gg, h_mg) plus an optional dense correction
/// for h/w cross-blocks.  Matvec is O(p_m^2 + p_g^2 + p_m*p_g) for the block part,
/// plus O(p_total^2) only if h/w blocks exist (which is rare and tiny).
struct BernoulliBlockHessianOperator {
    h_mm: Array2<f64>,
    h_gg: Array2<f64>,
    h_mg: Array2<f64>,
    dense_correction: Option<Array2<f64>>,
    marginal: std::ops::Range<usize>,
    logslope: std::ops::Range<usize>,
    total: usize,
}

impl HyperOperator for BernoulliBlockHessianOperator {
    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let v_m = v.slice(s![self.marginal.clone()]);
        let v_g = v.slice(s![self.logslope.clone()]);
        let mut out = Array1::zeros(self.total);
        {
            let mut o_m = out.slice_mut(s![self.marginal.clone()]);
            o_m += &self.h_mm.dot(&v_m);
            o_m += &self.h_mg.dot(&v_g);
        }
        {
            let mut o_g = out.slice_mut(s![self.logslope.clone()]);
            o_g += &self.h_mg.t().dot(&v_m);
            o_g += &self.h_gg.dot(&v_g);
        }
        if let Some(ref dc) = self.dense_correction {
            out += &dc.dot(v);
        }
        out
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        let v_m = v.slice(s![self.marginal.clone()]);
        let v_g = v.slice(s![self.logslope.clone()]);
        let u_m = u.slice(s![self.marginal.clone()]);
        let u_g = u.slice(s![self.logslope.clone()]);
        // Diagonal blocks
        let mut total = v_m.dot(&self.h_mm.dot(&u_m));
        total += v_g.dot(&self.h_gg.dot(&u_g));
        // Off-diagonal blocks (symmetric)
        total += v_m.dot(&self.h_mg.dot(&u_g));
        total += v_g.dot(&self.h_mg.t().dot(&u_m));
        // Dense correction
        if let Some(ref dc) = self.dense_correction {
            total += v.dot(&dc.dot(u));
        }
        total
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::zeros((self.total, self.total));
        out.slice_mut(s![self.marginal.clone(), self.marginal.clone()])
            .assign(&self.h_mm);
        out.slice_mut(s![self.logslope.clone(), self.logslope.clone()])
            .assign(&self.h_gg);
        out.slice_mut(s![self.marginal.clone(), self.logslope.clone()])
            .assign(&self.h_mg);
        out.slice_mut(s![self.logslope.clone(), self.marginal.clone()])
            .assign(&self.h_mg.t());
        if let Some(ref dc) = self.dense_correction {
            out += dc;
        }
        out
    }

    fn is_implicit(&self) -> bool {
        false
    }
}

#[derive(Clone)]
struct BernoulliMarginalSlopeRowExactContext {
    intercept: f64,
    m_a: f64,
}

struct BernoulliMarginalSlopeFlexRowScratch {
    m_u: Array1<f64>,
    m_au: Array1<f64>,
    m_uv: Array2<f64>,
    a_u: Array1<f64>,
    a_uv: Array2<f64>,
    rho: Array1<f64>,
    tau: Array1<f64>,
    du: Array1<f64>,
    grad: Array1<f64>,
    hess: Array2<f64>,
}

impl BernoulliMarginalSlopeFlexRowScratch {
    fn new(primary_dim: usize) -> Self {
        Self {
            m_u: Array1::zeros(primary_dim),
            m_au: Array1::zeros(primary_dim),
            m_uv: Array2::zeros((primary_dim, primary_dim)),
            a_u: Array1::zeros(primary_dim),
            a_uv: Array2::zeros((primary_dim, primary_dim)),
            rho: Array1::zeros(primary_dim),
            tau: Array1::zeros(primary_dim),
            du: Array1::zeros(primary_dim),
            grad: Array1::zeros(primary_dim),
            hess: Array2::zeros((primary_dim, primary_dim)),
        }
    }

    fn reset(&mut self, need_hessian: bool) {
        self.m_u.fill(0.0);
        self.a_u.fill(0.0);
        self.rho.fill(0.0);
        self.tau.fill(0.0);
        self.du.fill(0.0);
        self.grad.fill(0.0);
        if need_hessian {
            self.m_au.fill(0.0);
            self.m_uv.fill(0.0);
            self.a_uv.fill(0.0);
            self.hess.fill(0.0);
        }
    }
}

#[inline]
fn add_scaled_coeff4(target: &mut [f64; 4], source: &[f64; 4], scale: f64) {
    for j in 0..4 {
        target[j] += scale * source[j];
    }
}

#[inline]
fn scale_coeff4(source: [f64; 4], scale: f64) -> [f64; 4] {
    [
        source[0] * scale,
        source[1] * scale,
        source[2] * scale,
        source[3] * scale,
    ]
}

fn probit_frailty_scale(gaussian_frailty_sd: Option<f64>) -> f64 {
    let sigma = gaussian_frailty_sd.unwrap_or(0.0);
    if sigma <= 0.0 {
        1.0
    } else {
        crate::families::lognormal_kernel::ProbitFrailtyScaleJet::from_log_sigma(sigma.ln()).s
    }
}

fn jet_subset_partitions(mask: usize) -> Vec<Vec<usize>> {
    if mask == 0 {
        return vec![Vec::new()];
    }
    let first = mask & mask.wrapping_neg();
    let rest = mask ^ first;
    let mut out = Vec::new();
    let mut subset = rest;
    loop {
        let block = first | subset;
        for mut remainder in jet_subset_partitions(rest ^ subset) {
            remainder.push(block);
            out.push(remainder);
        }
        if subset == 0 {
            break;
        }
        subset = (subset - 1) & rest;
    }
    out
}

#[derive(Clone)]
struct MultiDirJet {
    coeffs: Vec<f64>,
}

impl MultiDirJet {
    fn zero(n_dirs: usize) -> Self {
        Self {
            coeffs: vec![0.0; 1usize << n_dirs],
        }
    }

    fn constant(n_dirs: usize, value: f64) -> Self {
        let mut out = Self::zero(n_dirs);
        out.coeffs[0] = value;
        out
    }

    fn linear(n_dirs: usize, base: f64, first: &[f64]) -> Self {
        let mut out = Self::constant(n_dirs, base);
        for (idx, &value) in first.iter().take(n_dirs).enumerate() {
            out.coeffs[1usize << idx] = value;
        }
        out
    }

    fn with_coeffs(n_dirs: usize, coeffs: &[(usize, f64)]) -> Self {
        let mut out = Self::zero(n_dirs);
        for &(mask, value) in coeffs {
            if mask < out.coeffs.len() {
                out.coeffs[mask] = value;
            }
        }
        out
    }

    fn coeff(&self, mask: usize) -> f64 {
        self.coeffs[mask]
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            coeffs: self
                .coeffs
                .iter()
                .zip(other.coeffs.iter())
                .map(|(lhs, rhs)| lhs + rhs)
                .collect(),
        }
    }

    fn scale(&self, scalar: f64) -> Self {
        Self {
            coeffs: self.coeffs.iter().map(|value| scalar * value).collect(),
        }
    }

    fn mul(&self, other: &Self) -> Self {
        let count = self.coeffs.len();
        let mut out = vec![0.0; count];
        for mask in 0..count {
            let mut total = 0.0;
            let mut submask = mask;
            loop {
                total += self.coeffs[submask] * other.coeffs[mask ^ submask];
                if submask == 0 {
                    break;
                }
                submask = (submask - 1) & mask;
            }
            out[mask] = total;
        }
        Self { coeffs: out }
    }

    fn compose_unary(&self, derivs: [f64; 5]) -> Self {
        let count = self.coeffs.len();
        let mut out = vec![0.0; count];
        out[0] = derivs[0];
        for (mask, value) in out.iter_mut().enumerate().skip(1) {
            let mut total = 0.0;
            for partition in jet_subset_partitions(mask) {
                let order = partition.len();
                if order == 0 || order >= derivs.len() {
                    continue;
                }
                let mut prod = 1.0;
                for block in partition {
                    prod *= self.coeffs[block];
                }
                total += derivs[order] * prod;
            }
            *value = total;
        }
        Self { coeffs: out }
    }
}

#[inline]
fn eval_coeff4_at(coefficients: &[f64; 4], z: f64) -> f64 {
    ((coefficients[3] * z + coefficients[2]) * z + coefficients[1]) * z + coefficients[0]
}

struct ObservedDenestedCellPartials {
    coeff: [f64; 4],
    dc_da: [f64; 4],
    dc_db: [f64; 4],
    dc_daa: [f64; 4],
    dc_dab: [f64; 4],
    dc_dbb: [f64; 4],
    dc_daaa: [f64; 4],
    dc_daab: [f64; 4],
    dc_dabb: [f64; 4],
    dc_dbbb: [f64; 4],
}

#[derive(Clone, Copy)]
struct CoeffSupport {
    include_b: bool,
    include_h: bool,
    include_w: bool,
}

impl CoeffSupport {
    #[inline]
    fn without_b(self) -> Self {
        Self {
            include_b: false,
            ..self
        }
    }
}

const COEFF_SUPPORT_BHW: CoeffSupport = CoeffSupport {
    include_b: true,
    include_h: true,
    include_w: true,
};
const COEFF_SUPPORT_BW: CoeffSupport = CoeffSupport {
    include_b: true,
    include_h: false,
    include_w: true,
};
const COEFF_SUPPORT_W: CoeffSupport = CoeffSupport {
    include_b: false,
    include_h: false,
    include_w: true,
};

// Sparse coefficient jet for the flexible primary coordinates `(q, b, h, w)`.
// Only the `b` axis couples to other primary blocks:
//   - score-warp coefficients are affine in `b`
//   - link-deviation coefficients are cubic in `(a, b)`
// so there are no `hh`, `hw`, or `ww` families to maintain by hand.
struct SparsePrimaryCoeffJetView<'a> {
    b_index: usize,
    h_range: Option<std::ops::Range<usize>>,
    w_range: Option<std::ops::Range<usize>>,
    first: &'a [[f64; 4]],
    a_first: &'a [[f64; 4]],
    b_first: &'a [[f64; 4]],
    aa_first: &'a [[f64; 4]],
    ab_first: &'a [[f64; 4]],
    bb_first: &'a [[f64; 4]],
    aaa_first: &'a [[f64; 4]],
    aab_first: &'a [[f64; 4]],
    abb_first: &'a [[f64; 4]],
    bbb_first: &'a [[f64; 4]],
}

impl<'a> SparsePrimaryCoeffJetView<'a> {
    fn new(
        h_range: Option<&std::ops::Range<usize>>,
        w_range: Option<&std::ops::Range<usize>>,
        first: &'a [[f64; 4]],
        a_first: &'a [[f64; 4]],
        b_first: &'a [[f64; 4]],
        aa_first: &'a [[f64; 4]],
        ab_first: &'a [[f64; 4]],
        bb_first: &'a [[f64; 4]],
        aaa_first: &'a [[f64; 4]],
        aab_first: &'a [[f64; 4]],
        abb_first: &'a [[f64; 4]],
        bbb_first: &'a [[f64; 4]],
    ) -> Self {
        Self {
            b_index: 1,
            h_range: h_range.cloned(),
            w_range: w_range.cloned(),
            first,
            a_first,
            b_first,
            aa_first,
            ab_first,
            bb_first,
            aaa_first,
            aab_first,
            abb_first,
            bbb_first,
        }
    }

    #[inline]
    fn in_h_range(&self, idx: usize) -> bool {
        self.h_range
            .as_ref()
            .map(|range| range.contains(&idx))
            .unwrap_or(false)
    }

    #[inline]
    fn in_w_range(&self, idx: usize) -> bool {
        self.w_range
            .as_ref()
            .map(|range| range.contains(&idx))
            .unwrap_or(false)
    }

    #[inline]
    fn param_supported(&self, idx: usize, support: CoeffSupport) -> bool {
        (support.include_b && idx == self.b_index)
            || (support.include_h && self.in_h_range(idx))
            || (support.include_w && self.in_w_range(idx))
    }

    fn directional_family(
        &self,
        family: &[[f64; 4]],
        dir: &Array1<f64>,
        support: CoeffSupport,
    ) -> [f64; 4] {
        let mut out = [0.0; 4];
        if support.include_b {
            add_scaled_coeff4(&mut out, &family[self.b_index], dir[self.b_index]);
        }
        if support.include_h {
            if let Some(h_range) = self.h_range.as_ref() {
                for idx in h_range.clone() {
                    add_scaled_coeff4(&mut out, &family[idx], dir[idx]);
                }
            }
        }
        if support.include_w {
            if let Some(w_range) = self.w_range.as_ref() {
                for idx in w_range.clone() {
                    add_scaled_coeff4(&mut out, &family[idx], dir[idx]);
                }
            }
        }
        out
    }

    fn mixed_directional_from_b_family(
        &self,
        family: &[[f64; 4]],
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
        support: CoeffSupport,
    ) -> [f64; 4] {
        let mut out = [0.0; 4];
        let dir_u_b = dir_u[self.b_index];
        let dir_v_b = dir_v[self.b_index];
        if support.include_b {
            add_scaled_coeff4(&mut out, &family[self.b_index], dir_u_b * dir_v_b);
        }
        if support.include_h {
            if let Some(h_range) = self.h_range.as_ref() {
                for idx in h_range.clone() {
                    add_scaled_coeff4(
                        &mut out,
                        &family[idx],
                        dir_u_b * dir_v[idx] + dir_v_b * dir_u[idx],
                    );
                }
            }
        }
        if support.include_w {
            if let Some(w_range) = self.w_range.as_ref() {
                for idx in w_range.clone() {
                    add_scaled_coeff4(
                        &mut out,
                        &family[idx],
                        dir_u_b * dir_v[idx] + dir_v_b * dir_u[idx],
                    );
                }
            }
        }
        out
    }

    fn param_directional_from_b_family(
        &self,
        family: &[[f64; 4]],
        param: usize,
        dir: &Array1<f64>,
        support: CoeffSupport,
    ) -> [f64; 4] {
        if param == self.b_index {
            return self.directional_family(family, dir, support);
        }
        if self.param_supported(param, support.without_b()) {
            let mut out = [0.0; 4];
            add_scaled_coeff4(&mut out, &family[param], dir[self.b_index]);
            return out;
        }
        [0.0; 4]
    }

    fn param_mixed_from_bb_family(
        &self,
        family: &[[f64; 4]],
        param: usize,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
        support: CoeffSupport,
    ) -> [f64; 4] {
        if param == self.b_index {
            return self.mixed_directional_from_b_family(family, dir_u, dir_v, support);
        }
        if self.param_supported(param, support.without_b()) {
            let mut out = [0.0; 4];
            add_scaled_coeff4(
                &mut out,
                &family[param],
                dir_u[self.b_index] * dir_v[self.b_index],
            );
            return out;
        }
        [0.0; 4]
    }

    fn pair_from_b_family(
        &self,
        family: &[[f64; 4]],
        u: usize,
        v: usize,
        support: CoeffSupport,
    ) -> [f64; 4] {
        if u == self.b_index && v == self.b_index {
            if support.include_b {
                return family[self.b_index];
            }
            return [0.0; 4];
        }
        if u == self.b_index && self.param_supported(v, support.without_b()) {
            return family[v];
        }
        if v == self.b_index && self.param_supported(u, support.without_b()) {
            return family[u];
        }
        [0.0; 4]
    }

    fn pair_directional_from_bb_family(
        &self,
        family: &[[f64; 4]],
        u: usize,
        v: usize,
        dir: &Array1<f64>,
        support: CoeffSupport,
    ) -> [f64; 4] {
        if u == self.b_index && v == self.b_index {
            return self.directional_family(family, dir, support);
        }
        if u == self.b_index && self.param_supported(v, support.without_b()) {
            let mut out = [0.0; 4];
            add_scaled_coeff4(&mut out, &family[v], dir[self.b_index]);
            return out;
        }
        if v == self.b_index && self.param_supported(u, support.without_b()) {
            let mut out = [0.0; 4];
            add_scaled_coeff4(&mut out, &family[u], dir[self.b_index]);
            return out;
        }
        [0.0; 4]
    }

    fn pair_mixed_from_bbb_family(
        &self,
        family: &[[f64; 4]],
        u: usize,
        v: usize,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
        support: CoeffSupport,
    ) -> [f64; 4] {
        if u == self.b_index && v == self.b_index {
            return self.mixed_directional_from_b_family(family, dir_u, dir_v, support);
        }
        if u == self.b_index && self.param_supported(v, support.without_b()) {
            let mut out = [0.0; 4];
            add_scaled_coeff4(
                &mut out,
                &family[v],
                dir_u[self.b_index] * dir_v[self.b_index],
            );
            return out;
        }
        if v == self.b_index && self.param_supported(u, support.without_b()) {
            let mut out = [0.0; 4];
            add_scaled_coeff4(
                &mut out,
                &family[u],
                dir_u[self.b_index] * dir_v[self.b_index],
            );
            return out;
        }
        [0.0; 4]
    }
}

struct BernoulliExactNewtonAccumulator {
    ll: f64,
    grad_marginal: Array1<f64>,
    grad_logslope: Array1<f64>,
    hess_marginal: Array2<f64>,
    hess_logslope: Array2<f64>,
    grad_h: Option<Array1<f64>>,
    grad_w: Option<Array1<f64>>,
    hess_h: Option<Array2<f64>>,
    hess_w: Option<Array2<f64>>,
}

impl BernoulliExactNewtonAccumulator {
    fn new(slices: &BlockSlices) -> Self {
        Self {
            ll: 0.0,
            grad_marginal: Array1::zeros(slices.marginal.len()),
            grad_logslope: Array1::zeros(slices.logslope.len()),
            hess_marginal: Array2::zeros((slices.marginal.len(), slices.marginal.len())),
            hess_logslope: Array2::zeros((slices.logslope.len(), slices.logslope.len())),
            grad_h: slices.h.as_ref().map(|range| Array1::zeros(range.len())),
            grad_w: slices.w.as_ref().map(|range| Array1::zeros(range.len())),
            hess_h: slices
                .h
                .as_ref()
                .map(|range| Array2::zeros((range.len(), range.len()))),
            hess_w: slices
                .w
                .as_ref()
                .map(|range| Array2::zeros((range.len(), range.len()))),
        }
    }

    fn add(&mut self, other: &Self) {
        self.ll += other.ll;
        self.grad_marginal += &other.grad_marginal;
        self.grad_logslope += &other.grad_logslope;
        self.hess_marginal += &other.hess_marginal;
        self.hess_logslope += &other.hess_logslope;
        if let (Some(left), Some(right)) = (self.grad_h.as_mut(), other.grad_h.as_ref()) {
            *left += right;
        }
        if let (Some(left), Some(right)) = (self.grad_w.as_mut(), other.grad_w.as_ref()) {
            *left += right;
        }
        if let (Some(left), Some(right)) = (self.hess_h.as_mut(), other.hess_h.as_ref()) {
            *left += right;
        }
        if let (Some(left), Some(right)) = (self.hess_w.as_mut(), other.hess_w.as_ref()) {
            *left += right;
        }
    }
}

fn add_weighted_chunk_gradient(
    chunk: &Array2<f64>,
    weights: &Array1<f64>,
    target: &mut Array1<f64>,
) {
    *target += &chunk.t().dot(weights);
}

fn add_weighted_chunk_gram(chunk: &Array2<f64>, weights: &Array1<f64>, target: &mut Array2<f64>) {
    let mut weighted_chunk = chunk.clone();
    for (mut row, &weight) in weighted_chunk.outer_iter_mut().zip(weights.iter()) {
        row *= weight;
    }
    *target += &chunk.t().dot(&weighted_chunk);
}

/// Chunk size for parallel row accumulation. Rows within a chunk are
/// processed sequentially. Flexible exact-Newton caches keep only the
/// pre-solved row context; primary jets are recomputed in chunk-local work
/// to avoid retaining O(n * p_primary^2) Hessian storage.
const ROW_CHUNK_SIZE: usize = 1024;

/// Shared precomputed state plus pre-solved per-row contexts. All row
/// intercepts are solved once during cache construction so that workspace
/// calls (matvec, diagonal, psi, directional derivatives) never redundantly
/// re-solve the Newton intercept equation.
#[derive(Clone)]
struct BernoulliMarginalSlopeExactEvalCache {
    slices: BlockSlices,
    primary: PrimarySlices,
    /// Pre-solved row contexts (intercept, M_a, observed score-warp value).
    row_contexts: Vec<BernoulliMarginalSlopeRowExactContext>,
}

// ── RowKernel<2> implementation (rigid path only) ────────────────────

struct BernoulliRigidRowKernel {
    family: BernoulliMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    slices: BlockSlices,
}

impl BernoulliRigidRowKernel {
    fn new(family: BernoulliMarginalSlopeFamily, block_states: Vec<ParameterBlockState>) -> Self {
        let slices = block_slices(&family);
        Self {
            family,
            block_states,
            slices,
        }
    }
}

impl RowKernel<2> for BernoulliRigidRowKernel {
    fn n_rows(&self) -> usize {
        self.family.y.len()
    }
    fn n_coefficients(&self) -> usize {
        self.slices.total
    }

    fn row_kernel(&self, row: usize) -> Result<(f64, [f64; 2], [[f64; 2]; 2]), String> {
        let marginal = self
            .family
            .marginal_link_map(self.block_states[0].eta[row])?;
        let g = self.block_states[1].eta[row];
        let probit_scale = self.family.probit_frailty_scale();
        let k = RigidProbitKernel::new(
            marginal.q,
            g,
            self.family.z[row],
            self.family.y[row],
            self.family.weights[row],
            probit_scale,
        )?;
        let nll = -self.family.weights[row] * k.logcdf;
        let grad = rigid_transformed_gradient(marginal, &k);
        let h = rigid_transformed_hessian(marginal, &k);
        Ok((nll, grad, h))
    }

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; 2] {
        let d_beta = ndarray::ArrayView1::from(d_beta);
        [
            self.family
                .marginal_design
                .dot_row_view(row, d_beta.slice(s![self.slices.marginal.clone()])),
            self.family
                .logslope_design
                .dot_row_view(row, d_beta.slice(s![self.slices.logslope.clone()])),
        ]
    }

    fn jacobian_transpose_action(&self, row: usize, v: &[f64; 2], out: &mut [f64]) {
        {
            let mut m = ndarray::ArrayViewMut1::from(&mut out[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .axpy_row_into(row, v[0], &mut m)
                .expect("marginal axpy dim mismatch");
        }
        {
            let mut g = ndarray::ArrayViewMut1::from(&mut out[self.slices.logslope.clone()]);
            self.family
                .logslope_design
                .axpy_row_into(row, v[1], &mut g)
                .expect("logslope axpy dim mismatch");
        }
    }

    fn add_pullback_hessian(&self, row: usize, h: &[[f64; 2]; 2], target: &mut Array2<f64>) {
        self.family
            .marginal_design
            .syr_row_into_view(
                row,
                h[0][0],
                target.slice_mut(s![
                    self.slices.marginal.clone(),
                    self.slices.marginal.clone()
                ]),
            )
            .expect("marginal syr dim mismatch");
        if h[0][1] != 0.0 {
            self.family
                .marginal_design
                .row_outer_into_view(
                    row,
                    &self.family.logslope_design,
                    h[0][1],
                    target.slice_mut(s![
                        self.slices.marginal.clone(),
                        self.slices.logslope.clone()
                    ]),
                )
                .expect("marginal-logslope outer dim mismatch");
            self.family
                .logslope_design
                .row_outer_into_view(
                    row,
                    &self.family.marginal_design,
                    h[0][1],
                    target.slice_mut(s![
                        self.slices.logslope.clone(),
                        self.slices.marginal.clone()
                    ]),
                )
                .expect("logslope-marginal outer dim mismatch");
        }
        self.family
            .logslope_design
            .syr_row_into_view(
                row,
                h[1][1],
                target.slice_mut(s![
                    self.slices.logslope.clone(),
                    self.slices.logslope.clone()
                ]),
            )
            .expect("logslope syr dim mismatch");
    }

    fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; 2]; 2], diag: &mut [f64]) {
        {
            let mut md = ndarray::ArrayViewMut1::from(&mut diag[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .squared_axpy_row_into(row, h[0][0], &mut md)
                .expect("marginal squared_axpy dim mismatch");
        }
        {
            let mut gd = ndarray::ArrayViewMut1::from(&mut diag[self.slices.logslope.clone()]);
            self.family
                .logslope_design
                .squared_axpy_row_into(row, h[1][1], &mut gd)
                .expect("logslope squared_axpy dim mismatch");
        }
    }

    fn row_third_contracted(&self, row: usize, dir: &[f64; 2]) -> Result<[[f64; 2]; 2], String> {
        let marginal = self
            .family
            .marginal_link_map(self.block_states[0].eta[row])?;
        let g = self.block_states[1].eta[row];
        let probit_scale = self.family.probit_frailty_scale();
        let k = RigidProbitKernel::new(
            marginal.q,
            g,
            self.family.z[row],
            self.family.y[row],
            self.family.weights[row],
            probit_scale,
        )?;
        Ok(rigid_transformed_third_contracted(
            marginal, &k, dir[0], dir[1],
        ))
    }

    fn row_fourth_contracted(
        &self,
        row: usize,
        dir_u: &[f64; 2],
        dir_v: &[f64; 2],
    ) -> Result<[[f64; 2]; 2], String> {
        let marginal = self
            .family
            .marginal_link_map(self.block_states[0].eta[row])?;
        let g = self.block_states[1].eta[row];
        let probit_scale = self.family.probit_frailty_scale();
        let k = RigidProbitKernel::new(
            marginal.q,
            g,
            self.family.z[row],
            self.family.y[row],
            self.family.weights[row],
            probit_scale,
        )?;
        Ok(rigid_transformed_fourth_contracted(
            marginal, &k, dir_u[0], dir_u[1], dir_v[0], dir_v[1],
        ))
    }
}

struct BernoulliMarginalSlopeExactNewtonJointHessianWorkspace {
    family: BernoulliMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    cache: BernoulliMarginalSlopeExactEvalCache,
}

struct BernoulliMarginalSlopeExactNewtonJointPsiWorkspace {
    family: BernoulliMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    specs: Vec<ParameterBlockSpec>,
    derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
    cache: BernoulliMarginalSlopeExactEvalCache,
}

impl BernoulliMarginalSlopeFamily {
    #[inline]
    fn probit_frailty_scale(&self) -> f64 {
        probit_frailty_scale(self.gaussian_frailty_sd)
    }

    fn is_sigma_aux_index(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> bool {
        let total = derivative_blocks.iter().map(Vec::len).sum::<usize>();
        if self.gaussian_frailty_sd.is_none() || total == 0 || psi_index != total - 1 {
            return false;
        }
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return false;
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        deriv.penalty_index.is_none()
            && deriv.x_psi.is_empty()
            && deriv.s_psi.is_empty()
            && deriv.s_psi_components.is_none()
            && deriv.x_psi_psi.is_none()
            && deriv.s_psi_psi.is_none()
    }

    fn sigma_scale_jet(
        &self,
        n_dirs: usize,
        first_masks: &[usize],
        second_masks: &[usize],
    ) -> Result<MultiDirJet, String> {
        let sigma = self.gaussian_frailty_sd.ok_or_else(|| {
            "bernoulli marginal-slope log-sigma auxiliary requested without GaussianShift sigma"
                .to_string()
        })?;
        let jet =
            crate::families::lognormal_kernel::ProbitFrailtyScaleJet::from_log_sigma(sigma.ln());
        let mut coeffs = Vec::with_capacity(1 + first_masks.len() + second_masks.len());
        coeffs.push((0usize, jet.s));
        coeffs.extend(first_masks.iter().copied().map(|mask| (mask, jet.ds)));
        coeffs.extend(second_masks.iter().copied().map(|mask| (mask, jet.d2s)));
        Ok(MultiDirJet::with_coeffs(n_dirs, &coeffs))
    }

    fn row_neglog_directional_with_scale_jet(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dirs: &[Array1<f64>],
        scale_jet: &MultiDirJet,
    ) -> Result<f64, String> {
        let k = dirs.len();
        if k > 4 {
            return Err(format!(
                "bernoulli marginal-slope sigma row directional expects 0..=4 directions, got {k}"
            ));
        }
        if scale_jet.coeffs.len() != (1usize << k) {
            return Err(format!(
                "bernoulli marginal-slope sigma scale jet dimension mismatch: coeffs={}, dirs={k}",
                scale_jet.coeffs.len()
            ));
        }

        let first = |idx: usize| -> Vec<f64> { dirs.iter().map(|dir| dir[idx]).collect() };
        let marginal = self.marginal_link_map(block_states[0].eta[row])?;
        let eta_jet = MultiDirJet::linear(k, block_states[0].eta[row], &first(0));
        let q_jet =
            eta_jet.compose_unary([marginal.q, marginal.q1, marginal.q2, marginal.q3, marginal.q4]);
        let g_jet = MultiDirJet::linear(k, block_states[1].eta[row], &first(1));
        let observed_g_jet = g_jet.mul(scale_jet);
        let one_plus_b2 = MultiDirJet::constant(k, 1.0).add(&observed_g_jet.mul(&observed_g_jet));
        let c_jet = one_plus_b2.compose_unary(unary_derivatives_sqrt(one_plus_b2.coeff(0)));
        let z_jet = MultiDirJet::constant(k, self.z[row]);
        let eta_observed_jet = q_jet.mul(&c_jet).add(&observed_g_jet.mul(&z_jet));
        let signed_jet = eta_observed_jet.scale(2.0 * self.y[row] - 1.0);
        Ok(signed_jet
            .compose_unary(unary_derivatives_neglog_phi(
                signed_jet.coeff(0),
                self.weights[row],
            ))
            .coeff((1usize << k) - 1))
    }

    fn row_sigma_primary_terms(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        second_sigma: bool,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let primary_dim = 2usize;
        let zero = Array1::<f64>::zeros(primary_dim);
        let objective = if second_sigma {
            let scale = self.sigma_scale_jet(2, &[1, 2], &[3])?;
            self.row_neglog_directional_with_scale_jet(
                row,
                block_states,
                &[zero.clone(), zero.clone()],
                &scale,
            )?
        } else {
            let scale = self.sigma_scale_jet(1, &[1], &[])?;
            self.row_neglog_directional_with_scale_jet(row, block_states, &[zero.clone()], &scale)?
        };

        let mut grad = Array1::<f64>::zeros(primary_dim);
        for a in 0..primary_dim {
            let mut da = Array1::<f64>::zeros(primary_dim);
            da[a] = 1.0;
            grad[a] = if second_sigma {
                let scale = self.sigma_scale_jet(3, &[1, 2], &[3])?;
                self.row_neglog_directional_with_scale_jet(
                    row,
                    block_states,
                    &[zero.clone(), zero.clone(), da],
                    &scale,
                )?
            } else {
                let scale = self.sigma_scale_jet(2, &[1], &[])?;
                self.row_neglog_directional_with_scale_jet(
                    row,
                    block_states,
                    &[zero.clone(), da],
                    &scale,
                )?
            };
        }

        let mut hess = Array2::<f64>::zeros((primary_dim, primary_dim));
        for a in 0..primary_dim {
            let mut da = Array1::<f64>::zeros(primary_dim);
            da[a] = 1.0;
            for b in a..primary_dim {
                let mut db = Array1::<f64>::zeros(primary_dim);
                db[b] = 1.0;
                let value = if second_sigma {
                    let scale = self.sigma_scale_jet(4, &[1, 2], &[3])?;
                    self.row_neglog_directional_with_scale_jet(
                        row,
                        block_states,
                        &[zero.clone(), zero.clone(), da.clone(), db],
                        &scale,
                    )?
                } else {
                    let scale = self.sigma_scale_jet(3, &[1], &[])?;
                    self.row_neglog_directional_with_scale_jet(
                        row,
                        block_states,
                        &[zero.clone(), da.clone(), db],
                        &scale,
                    )?
                };
                hess[[a, b]] = value;
                hess[[b, a]] = value;
            }
        }

        Ok((objective, grad, hess))
    }

    fn accumulate_rigid_sigma_pullback(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary_grad: &Array1<f64>,
        primary_hessian: &Array2<f64>,
        score: &mut Array1<f64>,
        hessian: &mut BernoulliBlockHessianAccumulator,
    ) -> Result<(), String> {
        {
            let mut marginal = score.slice_mut(s![slices.marginal.clone()]);
            self.marginal_design
                .axpy_row_into(row, primary_grad[0], &mut marginal)?;
        }
        {
            let mut logslope = score.slice_mut(s![slices.logslope.clone()]);
            self.logslope_design
                .axpy_row_into(row, primary_grad[1], &mut logslope)?;
        }
        hessian.add_pullback(self, row, slices, &primary_slices(slices), primary_hessian);
        Ok(())
    }

    fn sigma_exact_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        _specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if self.effective_flex_active(block_states)? {
            return Err(
                "bernoulli marginal-slope log-sigma hyperderivatives are implemented for the rigid probit marginal-slope kernel; flexible score/link kernels require the analytic denested cell-tensor sigma path"
                    .to_string(),
            );
        }
        if self.gaussian_frailty_sd.is_none() {
            return Ok(None);
        }
        let slices = block_slices(self);
        let n = self.y.len();
        let (objective_psi, score_psi, acc) = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || {
                    (
                        0.0,
                        Array1::<f64>::zeros(slices.total),
                        BernoulliBlockHessianAccumulator::new(&slices),
                    )
                },
                |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    for row in start..end {
                        let (obj, grad, hess) =
                            self.row_sigma_primary_terms(row, block_states, false)?;
                        acc.0 += obj;
                        self.accumulate_rigid_sigma_pullback(
                            row, &slices, &grad, &hess, &mut acc.1, &mut acc.2,
                        )?;
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || {
                    (
                        0.0,
                        Array1::<f64>::zeros(slices.total),
                        BernoulliBlockHessianAccumulator::new(&slices),
                    )
                },
                |mut left, right| -> Result<_, String> {
                    left.0 += right.0;
                    left.1 += &right.1;
                    left.2.add(&right.2);
                    Ok(left)
                },
            )?;
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: Array2::zeros((0, 0)),
            hessian_psi_operator: Some(Arc::new(acc.into_operator(&slices))),
        }))
    }

    fn sigma_exact_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        if self.effective_flex_active(block_states)? {
            return Err(
                "bernoulli marginal-slope second log-sigma hyperderivatives are implemented for the rigid probit marginal-slope kernel; flexible score/link kernels require the analytic denested cell-tensor sigma path"
                    .to_string(),
            );
        }
        if self.gaussian_frailty_sd.is_none() {
            return Ok(None);
        }
        let slices = block_slices(self);
        let n = self.y.len();
        let (objective_psi_psi, score_psi_psi, acc) = (0..((n + ROW_CHUNK_SIZE - 1)
            / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || {
                    (
                        0.0,
                        Array1::<f64>::zeros(slices.total),
                        BernoulliBlockHessianAccumulator::new(&slices),
                    )
                },
                |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    for row in start..end {
                        let (obj, grad, hess) =
                            self.row_sigma_primary_terms(row, block_states, true)?;
                        acc.0 += obj;
                        self.accumulate_rigid_sigma_pullback(
                            row, &slices, &grad, &hess, &mut acc.1, &mut acc.2,
                        )?;
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || {
                    (
                        0.0,
                        Array1::<f64>::zeros(slices.total),
                        BernoulliBlockHessianAccumulator::new(&slices),
                    )
                },
                |mut left, right| -> Result<_, String> {
                    left.0 += right.0;
                    left.1 += &right.1;
                    left.2.add(&right.2);
                    Ok(left)
                },
            )?;
        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(Box::new(acc.into_operator(&slices))),
        }))
    }

    fn sigma_exact_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if self.effective_flex_active(block_states)? {
            return Err(
                "bernoulli marginal-slope log-sigma Hessian directional derivatives are implemented for the rigid probit marginal-slope kernel; flexible score/link kernels require the analytic denested cell-tensor sigma path"
                    .to_string(),
            );
        }
        if self.gaussian_frailty_sd.is_none() {
            return Ok(None);
        }
        let slices = block_slices(self);
        if d_beta_flat.len() != slices.total {
            return Err(format!(
                "bernoulli marginal-slope d_beta length mismatch for sigma Hessian derivative: got {}, expected {}",
                d_beta_flat.len(),
                slices.total
            ));
        }
        let n = self.y.len();
        let primary = primary_slices(&slices);
        let acc = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || BernoulliBlockHessianAccumulator::new(&slices),
                |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    for row in start..end {
                        let row_dir =
                            self.row_primary_direction_from_flat(row, &slices, &primary, d_beta_flat)?;
                        let zero = Array1::<f64>::zeros(primary.total);
                        let mut grad = Array1::<f64>::zeros(primary.total);
                        for a in 0..primary.total {
                            let mut da = Array1::<f64>::zeros(primary.total);
                            da[a] = 1.0;
                            let scale = self.sigma_scale_jet(3, &[1], &[])?;
                            grad[a] = self.row_neglog_directional_with_scale_jet(
                                row,
                                block_states,
                                &[zero.clone(), row_dir.clone(), da],
                                &scale,
                            )?;
                        }
                        let mut hess = Array2::<f64>::zeros((primary.total, primary.total));
                        for a in 0..primary.total {
                            let mut da = Array1::<f64>::zeros(primary.total);
                            da[a] = 1.0;
                            for b in a..primary.total {
                                let mut db = Array1::<f64>::zeros(primary.total);
                                db[b] = 1.0;
                                let scale = self.sigma_scale_jet(4, &[1], &[])?;
                                let value = self.row_neglog_directional_with_scale_jet(
                                    row,
                                    block_states,
                                    &[zero.clone(), row_dir.clone(), da.clone(), db],
                                    &scale,
                                )?;
                                hess[[a, b]] = value;
                                hess[[b, a]] = value;
                            }
                        }
                        acc.add_pullback(self, row, &slices, &primary, &hess);
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || BernoulliBlockHessianAccumulator::new(&slices),
                |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                },
            )?;
        Ok(Some(acc.into_operator(&slices).to_dense()))
    }

    #[inline]
    fn marginal_link_map(&self, eta: f64) -> Result<BernoulliMarginalLinkMap, String> {
        bernoulli_marginal_link_map(&self.base_link, eta)
    }

    #[inline]
    fn exact_newton_score_component_from_objective_gradient(
        objective_gradient_component: f64,
    ) -> f64 {
        -objective_gradient_component
    }

    #[inline]
    fn exact_newton_score_from_objective_gradient(objective_gradient: Array1<f64>) -> Array1<f64> {
        -objective_gradient
    }

    #[inline]
    fn exact_newton_observed_information_from_objective_hessian(
        objective_hessian: Array2<f64>,
    ) -> Array2<f64> {
        objective_hessian
    }

    #[inline]
    fn score_block_index(&self) -> Option<usize> {
        self.score_warp.as_ref().map(|_| 2)
    }

    #[inline]
    fn link_block_index(&self) -> Option<usize> {
        self.link_dev
            .as_ref()
            .map(|_| 2 + usize::from(self.score_warp.is_some()))
    }

    fn optional_exact_block_state<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
        block_idx: Option<usize>,
        label: &str,
    ) -> Result<Option<&'a ParameterBlockState>, String> {
        match block_idx {
            Some(idx) => block_states
                .get(idx)
                .map(Some)
                .ok_or_else(|| format!("missing {label} block state")),
            None => Ok(None),
        }
    }

    fn score_block_state<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a ParameterBlockState>, String> {
        self.optional_exact_block_state(block_states, self.score_block_index(), "score-warp")
    }

    fn link_block_state<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a ParameterBlockState>, String> {
        self.optional_exact_block_state(block_states, self.link_block_index(), "link deviation")
    }

    fn score_beta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a Array1<f64>>, String> {
        Ok(self
            .score_block_state(block_states)?
            .map(|state| &state.beta))
    }

    fn link_beta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a Array1<f64>>, String> {
        Ok(self
            .link_block_state(block_states)?
            .map(|state| &state.beta))
    }

    fn validate_exact_block_state_shapes(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(), String> {
        let expected_blocks =
            2usize + usize::from(self.score_warp.is_some()) + usize::from(self.link_dev.is_some());
        if block_states.len() != expected_blocks {
            return Err(format!(
                "bernoulli marginal-slope block count mismatch: got {}, expected {}",
                block_states.len(),
                expected_blocks
            ));
        }

        let n_rows = self.y.len();
        let marginal = &block_states[0];
        let marginal_ncols = self.marginal_design.ncols();
        if marginal_ncols > 0 && marginal.beta.len() != marginal_ncols {
            return Err(format!(
                "bernoulli marginal-slope marginal beta length mismatch: got {}, expected {}",
                marginal.beta.len(),
                marginal_ncols
            ));
        }
        if marginal.eta.len() != n_rows {
            return Err(format!(
                "bernoulli marginal-slope marginal eta length mismatch: got {}, expected {}",
                marginal.eta.len(),
                n_rows
            ));
        }

        let logslope = &block_states[1];
        let logslope_ncols = self.logslope_design.ncols();
        if logslope_ncols > 0 && logslope.beta.len() != logslope_ncols {
            return Err(format!(
                "bernoulli marginal-slope logslope beta length mismatch: got {}, expected {}",
                logslope.beta.len(),
                logslope_ncols
            ));
        }
        if logslope.eta.len() != n_rows {
            return Err(format!(
                "bernoulli marginal-slope logslope eta length mismatch: got {}, expected {}",
                logslope.eta.len(),
                n_rows
            ));
        }

        if let Some(runtime) = &self.score_warp {
            let score = self
                .score_block_state(block_states)?
                .expect("score-warp block should exist when runtime is present");
            if score.beta.len() != runtime.basis_dim() {
                return Err(format!(
                    "bernoulli marginal-slope score-warp beta length mismatch: got {}, expected {}",
                    score.beta.len(),
                    runtime.basis_dim()
                ));
            }
            if score.eta.len() != n_rows {
                return Err(format!(
                    "bernoulli marginal-slope score-warp eta length mismatch: got {}, expected {}",
                    score.eta.len(),
                    n_rows
                ));
            }
        }

        if let Some(runtime) = &self.link_dev {
            let link = self
                .link_block_state(block_states)?
                .expect("link-deviation block should exist when runtime is present");
            if link.beta.len() != runtime.basis_dim() {
                return Err(format!(
                    "bernoulli marginal-slope link-deviation beta length mismatch: got {}, expected {}",
                    link.beta.len(),
                    runtime.basis_dim()
                ));
            }
            if link.eta.len() != n_rows {
                return Err(format!(
                    "bernoulli marginal-slope link-deviation eta length mismatch: got {}, expected {}",
                    link.eta.len(),
                    n_rows
                ));
            }
        }

        Ok(())
    }

    fn denested_partition_cells(
        &self,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<Vec<exact_kernel::DenestedPartitionCell>, String> {
        let score_breaks = self
            .score_warp
            .as_ref()
            .map(|runtime| runtime.breakpoints().to_vec())
            .unwrap_or_default();
        let link_breaks = self
            .link_dev
            .as_ref()
            .map(|runtime| runtime.breakpoints().to_vec())
            .unwrap_or_default();

        let mut cells = exact_kernel::build_denested_partition_cells_with_tails(
            a,
            b,
            &score_breaks,
            &link_breaks,
            |z| {
                if let (Some(runtime), Some(beta)) = (self.score_warp.as_ref(), beta_h) {
                    runtime.local_cubic_at(beta, z)
                } else {
                    Ok(exact_kernel::LocalSpanCubic {
                        left: 0.0,
                        right: 1.0,
                        c0: 0.0,
                        c1: 0.0,
                        c2: 0.0,
                        c3: 0.0,
                    })
                }
            },
            |u| {
                if let (Some(runtime), Some(beta)) = (self.link_dev.as_ref(), beta_w) {
                    runtime.local_cubic_at(beta, u)
                } else {
                    Ok(exact_kernel::LocalSpanCubic {
                        left: 0.0,
                        right: 1.0,
                        c0: 0.0,
                        c1: 0.0,
                        c2: 0.0,
                        c3: 0.0,
                    })
                }
            },
        )?;
        let scale = self.probit_frailty_scale();
        if scale != 1.0 {
            for partition_cell in &mut cells {
                partition_cell.cell.c0 *= scale;
                partition_cell.cell.c1 *= scale;
                partition_cell.cell.c2 *= scale;
                partition_cell.cell.c3 *= scale;
            }
        }
        Ok(cells)
    }

    fn evaluate_denested_calibration(
        &self,
        a: f64,
        marginal_eta: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(f64, f64, f64), String> {
        let marginal = self.marginal_link_map(marginal_eta)?;
        let cells = self.denested_partition_cells(a, slope, beta_h, beta_w)?;
        let scale = self.probit_frailty_scale();
        let mut f = -marginal.mu;
        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        for partition_cell in cells {
            let cell = partition_cell.cell;
            let state = exact_kernel::evaluate_cell_moments(cell, 9)?;
            f += state.value;
            let (dc_da_raw, _) = exact_kernel::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let (d2c_da2_raw, _, _) = exact_kernel::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let dc_da = scale_coeff4(dc_da_raw, scale);
            let d2c_da2 = scale_coeff4(d2c_da2_raw, scale);
            f_a += exact_kernel::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            f_aa += exact_kernel::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &d2c_da2,
                &state.moments,
            )?;
        }
        Ok((f, f_a, f_aa))
    }

    fn flex_active(&self) -> bool {
        self.score_warp.is_some() || self.link_dev.is_some()
    }

    /// The denested exact path is active whenever either deviation runtime is
    /// configured. Zero coefficient vectors still keep the flexible geometry
    /// live so derivatives with respect to those coefficients remain available.
    fn effective_flex_active(&self, block_states: &[ParameterBlockState]) -> Result<bool, String> {
        if self.score_warp.is_some() && self.score_beta(block_states)?.is_none() {
            return Err("missing bernoulli score-warp block state".to_string());
        }
        if self.link_dev.is_some() && self.link_beta(block_states)?.is_none() {
            return Err("missing bernoulli link-deviation block state".to_string());
        }
        Ok(self.flex_active())
    }

    fn validate_exact_monotonicity(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(), String> {
        self.validate_exact_block_state_shapes(block_states)?;
        if let (Some(runtime), Some(score)) =
            (&self.score_warp, self.score_block_state(block_states)?)
        {
            runtime.monotonicity_feasible(
                &score.beta,
                "bernoulli marginal-slope score-warp deviation",
            )?;
        }
        if let (Some(runtime), Some(beta_w)) = (&self.link_dev, self.link_beta(block_states)?) {
            runtime.monotonicity_feasible(beta_w, "bernoulli marginal-slope link deviation")?;
        }
        Ok(())
    }

    /// Fast path: value + first derivative only, skips second derivative design.
    fn link_terms_value_d1(
        &self,
        eta0: &Array1<f64>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
        if let (Some(runtime), Some(beta)) = (&self.link_dev, beta_w) {
            let basis = runtime.design(eta0)?;
            let d1 = runtime.first_derivative_design(eta0)?;
            Ok((eta0 + &basis.dot(beta), d1.dot(beta) + 1.0))
        } else {
            Ok((eta0.clone(), Array1::ones(eta0.len())))
        }
    }

    fn solve_row_intercept_base(
        &self,
        marginal_eta: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(f64, f64), String> {
        let marginal = self.marginal_link_map(marginal_eta)?;
        let eval = |a: f64| -> Result<(f64, f64, f64), String> {
            self.evaluate_denested_calibration(a, marginal_eta, slope, beta_h, beta_w)
        };

        let probit_scale = self.probit_frailty_scale();

        // Initial guess: closed-form for rigid probit in pre-scale denested
        // coordinates:
        //   a₀ = q·√(1 + (s_f b)²) / s_f,  s_f = 1/√(1+σ²).
        // When link deviation is active, upgrade to affine-link warm start:
        //   s_f·L(u) ≈ s_f·(ℓ₀ + ℓ₁·u)
        //   ⟹  a = (q·√(1 + (s_f ℓ₁ b)²) / s_f − ℓ₀) / ℓ₁
        let a_rigid_pre_scale =
            rigid_intercept_from_marginal(marginal.q, slope, probit_scale) / probit_scale;
        let a_init = if beta_w.is_some() {
            let v = Array1::from_vec(vec![a_rigid_pre_scale]);
            let (l_val, l_d1) = self.link_terms_value_d1(&v, beta_w)?;
            let ell1 = l_d1[0];
            if ell1 > 1e-8 {
                let ell0 = l_val[0] - ell1 * a_rigid_pre_scale;
                let observed_logslope = probit_scale * ell1 * slope;
                (marginal.q * (1.0 + observed_logslope * observed_logslope).sqrt() / probit_scale
                    - ell0)
                    / ell1
            } else {
                a_rigid_pre_scale
            }
        } else {
            a_rigid_pre_scale
        };

        let (a, abs_deriv, f_best) = super::monotone_root::solve_monotone_root(
            eval,
            a_init,
            "bernoulli intercept",
            1e-10,
            64,
            48,
        )?;

        // Adaptive tolerance: for extreme slopes the intercept equation
        // becomes numerically flat and tight absolute precision is not
        // achievable.  Accept the best bracketed solution when the
        // relative residual is small.
        let target = marginal.mu;
        let abs_tol = 1e-8_f64.max(1e-4 * target.abs());
        if f_best.abs() > abs_tol {
            return Err(format!(
                "bernoulli marginal-slope intercept solve failed: \
                 residual={f_best:.3e} at a={a:.6}, target mu={target:.6}"
            ));
        }

        Ok((a, abs_deriv))
    }

    fn build_row_exact_context(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<BernoulliMarginalSlopeRowExactContext, String> {
        let marginal_eta = block_states[0].eta[row];
        let marginal = self.marginal_link_map(marginal_eta)?;
        // The log-slope block now parameterizes the signed slope directly.
        let slope = block_states[1].eta[row];
        let beta_h = self.score_beta(block_states)?;
        let beta_w = self.link_beta(block_states)?;
        let (intercept, m_a) = if self.effective_flex_active(block_states)? {
            self.solve_row_intercept_base(marginal_eta, slope, beta_h, beta_w)?
        } else {
            (
                rigid_intercept_from_marginal(marginal.q, slope, self.probit_frailty_scale()),
                f64::NAN,
            )
        };
        Ok(BernoulliMarginalSlopeRowExactContext { intercept, m_a })
    }

    /// Look up the pre-solved row context from the cache.
    #[inline]
    fn row_ctx(
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row: usize,
    ) -> &BernoulliMarginalSlopeRowExactContext {
        &cache.row_contexts[row]
    }

    fn build_exact_eval_cache_with_order(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<BernoulliMarginalSlopeExactEvalCache, String> {
        self.validate_exact_block_state_shapes(block_states)?;
        let slices = block_slices(self);
        let primary = primary_slices(&slices);
        let n = self.y.len();
        let row_contexts: Result<Vec<_>, String> = (0..n)
            .into_par_iter()
            .map(|row| self.build_row_exact_context(row, block_states))
            .collect();
        let row_contexts = row_contexts?;
        Ok(BernoulliMarginalSlopeExactEvalCache {
            slices,
            primary,
            row_contexts,
        })
    }

    fn build_exact_eval_cache(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<BernoulliMarginalSlopeExactEvalCache, String> {
        self.build_exact_eval_cache_with_order(block_states)
    }
    fn row_primary_direction_from_flat(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        if d_beta_flat.len() != slices.total {
            return Err(format!(
                "bernoulli marginal-slope d_beta length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                slices.total
            ));
        }
        let mut out = Array1::<f64>::zeros(primary.total);
        out[primary.q] = self
            .marginal_design
            .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
        out[primary.logslope] = self
            .logslope_design
            .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));
        if let (Some(block_range), Some(primary_range)) = (slices.h.as_ref(), primary.h.as_ref()) {
            out.slice_mut(s![primary_range.start..primary_range.end])
                .assign(&d_beta_flat.slice(s![block_range.clone()]).to_owned());
        }
        if let (Some(block_range), Some(primary_range)) = (slices.w.as_ref(), primary.w.as_ref()) {
            out.slice_mut(s![primary_range.start..primary_range.end])
                .assign(&d_beta_flat.slice(s![block_range.clone()]).to_owned());
        }
        Ok(out)
    }

    fn resolve_psi_location(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Option<(usize, usize)> {
        let mut cursor = 0usize;
        for (block_idx, block) in derivative_blocks.iter().enumerate() {
            if psi_index < cursor + block.len() {
                return Some((block_idx, psi_index - cursor));
            }
            cursor += block.len();
        }
        None
    }

    fn row_primary_psi_direction_from_map(
        &self,
        row: usize,
        block_idx: usize,
        psi_map: &crate::families::custom_family::PsiDesignMap,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
    ) -> Result<Array1<f64>, String> {
        let mut out = Array1::<f64>::zeros(primary.total);
        match block_idx {
            0 => {
                let x_row = psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                out[primary.q] = x_row.dot(&block_states[0].beta);
            }
            1 => {
                let x_row = psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                out[primary.logslope] = x_row.dot(&block_states[1].beta);
            }
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi direction only supports spatial marginal/logslope blocks, got block {block_idx}"
                ));
            }
        }
        Ok(out)
    }

    fn row_primary_psi_action_on_direction_from_map(
        &self,
        row: usize,
        block_idx: usize,
        psi_map: &crate::families::custom_family::PsiDesignMap,
        slices: &BlockSlices,
        d_beta_flat: &Array1<f64>,
        primary: &PrimarySlices,
    ) -> Result<Array1<f64>, String> {
        let mut out = Array1::<f64>::zeros(primary.total);
        match block_idx {
            0 => {
                let x_row = psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                out[primary.q] =
                    x_row.dot(&d_beta_flat.slice(s![slices.marginal.clone()]).to_owned())
            }
            1 => {
                let x_row = psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                out[primary.logslope] =
                    x_row.dot(&d_beta_flat.slice(s![slices.logslope.clone()]).to_owned())
            }
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi action only supports marginal/logslope blocks, got block {block_idx}"
                ));
            }
        }
        Ok(out)
    }

    fn row_primary_psi_second_direction_from_map(
        &self,
        row: usize,
        block_i: usize,
        block_j: usize,
        psi_map_ij: Option<&crate::families::custom_family::PsiDesignMap>,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
    ) -> Result<Array1<f64>, String> {
        if block_i != block_j {
            return Ok(Array1::<f64>::zeros(primary.total));
        }
        let psi_map_ij = psi_map_ij
            .expect("psi_map_ij must be provided when block_i == block_j");
        let mut out = Array1::<f64>::zeros(primary.total);
        match block_i {
            0 => {
                let x_row = psi_map_ij
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                out[primary.q] = x_row.dot(&block_states[0].beta);
            }
            1 => {
                let x_row = psi_map_ij
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                out[primary.logslope] = x_row.dot(&block_states[1].beta);
            }
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi second direction only supports marginal/logslope blocks, got block {block_i}"
                ));
            }
        }
        Ok(out)
    }

    fn pullback_primary_vector(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        primary_vec: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let mut out = Array1::<f64>::zeros(slices.total);
        {
            let mut marginal = out.slice_mut(s![slices.marginal.clone()]);
            self.marginal_design
                .axpy_row_into(row, primary_vec[primary.q], &mut marginal)?;
        }
        {
            let mut logslope = out.slice_mut(s![slices.logslope.clone()]);
            self.logslope_design.axpy_row_into(
                row,
                primary_vec[primary.logslope],
                &mut logslope,
            )?;
        }
        if let Some(primary_h) = primary.h.as_ref() {
            if let Some(block_h) = slices.h.as_ref() {
                out.slice_mut(s![block_h.clone()]).assign(
                    &primary_vec
                        .slice(s![primary_h.start..primary_h.end])
                        .to_owned(),
                );
            }
        }
        if let Some(primary_w) = primary.w.as_ref() {
            if let Some(block_w) = slices.w.as_ref() {
                out.slice_mut(s![block_w.clone()]).assign(
                    &primary_vec
                        .slice(s![primary_w.start..primary_w.end])
                        .to_owned(),
                );
            }
        }
        Ok(out)
    }

    fn block_psi_row_from_map(
        &self,
        row: usize,
        block_idx: usize,
        psi_map: &crate::families::custom_family::PsiDesignMap,
        slices: &BlockSlices,
    ) -> Result<BlockPsiRow, String> {
        let (local_vec, range) = match block_idx {
            0 => (
                psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?,
                slices.marginal.clone(),
            ),
            1 => (
                psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?,
                slices.logslope.clone(),
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi embedding only supports marginal/logslope blocks, got block {block_idx}"
                ));
            }
        };
        Ok(BlockPsiRow {
            block_idx,
            range,
            local_vec,
        })
    }

    fn block_psi_second_row_from_map(
        &self,
        row: usize,
        block_i: usize,
        block_j: usize,
        psi_map_ij: Option<&crate::families::custom_family::PsiDesignMap>,
        slices: &BlockSlices,
    ) -> Result<Option<BlockPsiRow>, String> {
        if block_i != block_j {
            return Ok(None);
        }
        let psi_map_ij = psi_map_ij
            .expect("psi_map_ij must be provided when block_i == block_j");
        let (local_vec, range) = match block_i {
            0 => (
                psi_map_ij
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?,
                slices.marginal.clone(),
            ),
            1 => (
                psi_map_ij
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?,
                slices.logslope.clone(),
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi second embedding only supports marginal/logslope blocks, got block {block_i}"
                ));
            }
        };
        Ok(Some(BlockPsiRow {
            block_idx: block_i,
            range,
            local_vec,
        }))
    }

    /// Returns (neg_log_lik, gradient, Hessian) in primary coordinates.
    /// Fully analytic for both flex and non-flex paths — no AD jets.
    fn compute_row_primary_gradient_hessian(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        // Flex path: full IFT analytic kernel.
        if self.effective_flex_active(block_states)? {
            let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
            let neglog = self.compute_row_analytic_flex_into(
                row,
                block_states,
                primary,
                row_ctx,
                true,
                &mut scratch,
            )?;
            return Ok((neglog, scratch.grad, scratch.hess));
        }
        // Rigid path: closed-form observed eta with probit frailty scaling.
        // primary.total == 2 (q at 0, g at 1), no h/w blocks.
        let marginal = self.marginal_link_map(block_states[0].eta[row])?;
        let g = block_states[1].eta[row];
        let kern = RigidProbitKernel::new(
            marginal.q,
            g,
            self.z[row],
            self.y[row],
            self.weights[row],
            self.probit_frailty_scale(),
        )?;
        let neglog = -self.weights[row] * kern.logcdf;
        let grad_pair = rigid_transformed_gradient(marginal, &kern);
        let mut grad = Array1::<f64>::zeros(2);
        grad[0] = grad_pair[0];
        grad[1] = grad_pair[1];

        let h = rigid_transformed_hessian(marginal, &kern);
        let mut hess = Array2::<f64>::zeros((2, 2));
        hess[[0, 0]] = h[0][0];
        hess[[0, 1]] = h[0][1];
        hess[[1, 0]] = h[1][0];
        hess[[1, 1]] = h[1][1];

        Ok((neglog, grad, hess))
    }

    fn compute_row_analytic_flex_into(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        need_hessian: bool,
        scratch: &mut BernoulliMarginalSlopeFlexRowScratch,
    ) -> Result<f64, String> {
        let q = block_states[0].eta[row];
        let b = block_states[1].eta[row];
        let beta_h = self.score_beta(block_states)?;
        let beta_w = self.link_beta(block_states)?;
        self.compute_row_analytic_flex_from_parts_into(
            row,
            primary,
            q,
            b,
            beta_h,
            beta_w,
            row_ctx,
            need_hessian,
            scratch,
        )
    }

    fn compute_row_analytic_flex_from_parts_into(
        &self,
        row: usize,
        primary: &PrimarySlices,
        q: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        need_hessian: bool,
        scratch: &mut BernoulliMarginalSlopeFlexRowScratch,
    ) -> Result<f64, String> {
        use crate::families::bernoulli_marginal_slope::exact_kernel as exact;

        let r = primary.total;
        scratch.reset(need_hessian);
        let a = row_ctx.intercept;
        let f_a = row_ctx.m_a;
        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let marginal = self.marginal_link_map(q)?;
        let inv_ma = 1.0 / f_a;
        let h_range = primary.h.as_ref();
        let w_range = primary.w.as_ref();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();
        let scale = self.probit_frailty_scale();
        let zero_family = vec![[0.0; 4]; r];

        let f_u = &mut scratch.m_u;
        let f_au = &mut scratch.m_au;
        let f_uv = &mut scratch.m_uv;
        let mut f_aa = 0.0f64;
        let mut coeff_u = vec![[0.0; 4]; r];
        let mut coeff_au = vec![[0.0; 4]; r];
        let mut coeff_bu = vec![[0.0; 4]; r];

        let cells = self.denested_partition_cells(a, b, beta_h, beta_w)?;
        for partition_cell in cells {
            let cell = partition_cell.cell;
            let z_mid = exact::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = exact::evaluate_cell_moments(cell, if need_hessian { 9 } else { 3 })?;
            let (dc_da_raw, dc_db_raw) = exact::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let dc_da = scale_coeff4(dc_da_raw, scale);
            let dc_db = scale_coeff4(dc_db_raw, scale);

            coeff_u[1] = dc_db;
            coeff_au[1] = [0.0; 4];
            coeff_bu[1] = [0.0; 4];
            if need_hessian {
                let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) = exact::denested_cell_second_partials(
                    partition_cell.score_span,
                    partition_cell.link_span,
                    a,
                    b,
                );
                let dc_daa = scale_coeff4(dc_daa_raw, scale);
                let dc_dab = scale_coeff4(dc_dab_raw, scale);
                let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
                f_aa += exact::cell_second_derivative_from_moments(
                    cell,
                    &dc_da,
                    &dc_da,
                    &dc_daa,
                    &state.moments,
                )?;
                coeff_au[1] = dc_dab;
                coeff_bu[1] = dc_dbb;
            }

            if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
                for local_idx in 0..h_range.len() {
                    let basis_span = runtime.basis_cubic_at(local_idx, z_mid)?;
                    let idx = h_range.start + local_idx;
                    coeff_u[idx] =
                        scale_coeff4(exact::score_basis_cell_coefficients(basis_span, b), scale);
                    if need_hessian {
                        coeff_bu[idx] = scale_coeff4(
                            exact::score_basis_cell_coefficients(basis_span, 1.0),
                            scale,
                        );
                    }
                }
            }

            if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
                for local_idx in 0..w_range.len() {
                    let basis_span = runtime.basis_cubic_at(local_idx, u_mid)?;
                    let idx = w_range.start + local_idx;
                    coeff_u[idx] =
                        scale_coeff4(exact::link_basis_cell_coefficients(basis_span, a, b), scale);
                    if need_hessian {
                        let (dc_aw_raw, dc_bw_raw) =
                            exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                        coeff_au[idx] = scale_coeff4(dc_aw_raw, scale);
                        coeff_bu[idx] = scale_coeff4(dc_bw_raw, scale);
                    }
                }
            }

            for u in 1..r {
                f_u[u] += exact::cell_first_derivative_from_moments(&coeff_u[u], &state.moments)?;
                if need_hessian {
                    f_au[u] += exact::cell_second_derivative_from_moments(
                        cell,
                        &dc_da,
                        &coeff_u[u],
                        &coeff_au[u],
                        &state.moments,
                    )?;
                }
            }

            if need_hessian {
                let coeff_jet = SparsePrimaryCoeffJetView::new(
                    h_range,
                    w_range,
                    &coeff_u,
                    &coeff_au,
                    &coeff_bu,
                    &zero_family,
                    &zero_family,
                    &zero_family,
                    &zero_family,
                    &zero_family,
                    &zero_family,
                    &zero_family,
                );
                for u in 1..r {
                    for v in u..r {
                        let second_coeff = coeff_jet.pair_from_b_family(
                            coeff_jet.b_first,
                            u,
                            v,
                            COEFF_SUPPORT_BHW,
                        );
                        let val = exact::cell_second_derivative_from_moments(
                            cell,
                            &coeff_jet.first[u],
                            &coeff_jet.first[v],
                            &second_coeff,
                            &state.moments,
                        )?;
                        f_uv[[u, v]] += val;
                        if u != v {
                            f_uv[[v, u]] += val;
                        }
                    }
                }
            }
        }

        f_u[0] = -marginal.mu1;
        if need_hessian {
            f_uv[[0, 0]] = -marginal.mu2;
        }

        let a_u = &mut scratch.a_u;
        for u in 0..r {
            a_u[u] = -f_u[u] * inv_ma;
        }
        let a_uv = &mut scratch.a_uv;
        if need_hessian {
            for u in 0..r {
                for v in u..r {
                    let val = -(f_uv[[u, v]]
                        + f_au[u] * a_u[v]
                        + f_au[v] * a_u[u]
                        + f_aa * a_u[u] * a_u[v])
                        * inv_ma;
                    a_uv[[u, v]] = val;
                    a_uv[[v, u]] = val;
                }
            }
        }

        let z_obs = self.z[row];
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let chi_obs = eval_coeff4_at(&obs.dc_da, z_obs);
        let eta_aa_obs = eval_coeff4_at(&obs.dc_daa, z_obs);
        let eta_val = eval_coeff4_at(&obs.coeff, z_obs);

        let mut g_u_fixed = vec![[0.0; 4]; r];
        let mut g_au_fixed = vec![[0.0; 4]; r];
        let mut g_bu_fixed = vec![[0.0; 4]; r];
        g_u_fixed[1] = obs.dc_db;
        g_au_fixed[1] = obs.dc_dab;
        g_bu_fixed[1] = obs.dc_dbb;
        if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
            for local_idx in 0..h_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, z_obs)?;
                let idx = h_range.start + local_idx;
                g_u_fixed[idx] =
                    scale_coeff4(exact::score_basis_cell_coefficients(basis_span, b), scale);
                g_bu_fixed[idx] =
                    scale_coeff4(exact::score_basis_cell_coefficients(basis_span, 1.0), scale);
            }
        }
        if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let idx = w_range.start + local_idx;
                g_u_fixed[idx] =
                    scale_coeff4(exact::link_basis_cell_coefficients(basis_span, a, b), scale);
                let (dc_aw_raw, dc_bw_raw) =
                    exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                g_au_fixed[idx] = scale_coeff4(dc_aw_raw, scale);
                g_bu_fixed[idx] = scale_coeff4(dc_bw_raw, scale);
            }
        }
        let g_jet = SparsePrimaryCoeffJetView::new(
            h_range,
            w_range,
            &g_u_fixed,
            &g_au_fixed,
            &g_bu_fixed,
            &zero_family,
            &zero_family,
            &zero_family,
            &zero_family,
            &zero_family,
            &zero_family,
            &zero_family,
        );

        let rho = &mut scratch.rho;
        let tau = &mut scratch.tau;
        rho.fill(0.0);
        tau.fill(0.0);
        for u in 1..r {
            rho[u] = eval_coeff4_at(&g_jet.first[u], z_obs);
            tau[u] = eval_coeff4_at(&g_jet.a_first[u], z_obs);
        }

        let eta_u = &mut scratch.grad;
        for u in 0..r {
            eta_u[u] = chi_obs * a_u[u] + rho[u];
        }

        let signed_margin = s_y * eta_val;
        let (log_cdf, lambda) = signed_probit_logcdf_and_mills_ratio(signed_margin);
        let neglog_val = -w_i * log_cdf;
        let d1_m = -w_i * lambda;
        let d2_m = w_i * lambda * (signed_margin + lambda);

        if need_hessian {
            let hess = &mut scratch.hess;
            hess.fill(0.0);
            for u in 0..r {
                for v in u..r {
                    let r_uv = eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_BHW),
                        z_obs,
                    );
                    let eta_uv = chi_obs * a_uv[[u, v]]
                        + eta_aa_obs * a_u[u] * a_u[v]
                        + tau[u] * a_u[v]
                        + a_u[u] * tau[v]
                        + r_uv;
                    let val = d2_m * eta_u[u] * eta_u[v] + d1_m * s_y * eta_uv;
                    hess[[u, v]] = val;
                    hess[[v, u]] = val;
                }
            }
        }

        eta_u.mapv_inplace(|eu| d1_m * s_y * eu);
        Ok(neglog_val)
    }

    fn primary_point_from_block_states(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
    ) -> Result<Array1<f64>, String> {
        let mut point = Array1::<f64>::zeros(primary.total);
        point[primary.q] = block_states[0].eta[row];
        point[primary.logslope] = block_states[1].eta[row];
        if let Some(h_range) = primary.h.as_ref() {
            let score = self
                .score_block_state(block_states)?
                .ok_or_else(|| "missing score-warp beta".to_string())?;
            point
                .slice_mut(s![h_range.start..h_range.end])
                .assign(&score.beta);
        }
        if let Some(w_range) = primary.w.as_ref() {
            let beta_w = self
                .link_block_state(block_states)?
                .ok_or_else(|| "missing link deviation beta".to_string())?;
            point
                .slice_mut(s![w_range.start..w_range.end])
                .assign(&beta_w.beta);
        }
        Ok(point)
    }

    fn primary_point_components(
        &self,
        point: &Array1<f64>,
        primary: &PrimarySlices,
    ) -> (f64, f64, Option<Array1<f64>>, Option<Array1<f64>>) {
        let beta_h = primary
            .h
            .as_ref()
            .map(|range| point.slice(s![range.start..range.end]).to_owned());
        let beta_w = primary
            .w
            .as_ref()
            .map(|range| point.slice(s![range.start..range.end]).to_owned());
        (point[primary.q], point[primary.logslope], beta_h, beta_w)
    }

    fn observed_denested_cell_partials(
        &self,
        row: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<ObservedDenestedCellPartials, String> {
        use crate::families::bernoulli_marginal_slope::exact_kernel as exact;

        let scale = self.probit_frailty_scale();
        let zero_score_span = exact::LocalSpanCubic {
            left: 0.0,
            right: 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        };
        let zero_link_span = exact::LocalSpanCubic {
            left: 0.0,
            right: 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        };
        let z_obs = self.z[row];
        let u_obs = a + b * z_obs;
        let score_span_obs =
            if let (Some(runtime), Some(beta_h)) = (self.score_warp.as_ref(), beta_h) {
                runtime.local_cubic_at(beta_h, z_obs)?
            } else {
                zero_score_span
            };
        let link_span_obs = if let (Some(runtime), Some(beta_w)) = (self.link_dev.as_ref(), beta_w)
        {
            runtime.local_cubic_at(beta_w, u_obs)?
        } else {
            zero_link_span
        };
        let coeff = scale_coeff4(
            exact::denested_cell_coefficients(score_span_obs, link_span_obs, a, b),
            scale,
        );
        let (dc_da_raw, dc_db_raw) =
            exact::denested_cell_coefficient_partials(score_span_obs, link_span_obs, a, b);
        let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) =
            exact::denested_cell_second_partials(score_span_obs, link_span_obs, a, b);
        let denested_third = exact::denested_cell_third_partials(link_span_obs);
        let dc_da = scale_coeff4(dc_da_raw, scale);
        let dc_db = scale_coeff4(dc_db_raw, scale);
        let dc_daa = scale_coeff4(dc_daa_raw, scale);
        let dc_dab = scale_coeff4(dc_dab_raw, scale);
        let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
        Ok(ObservedDenestedCellPartials {
            coeff,
            dc_da,
            dc_db,
            dc_daa,
            dc_dab,
            dc_dbb,
            dc_daaa: scale_coeff4(denested_third.0, scale),
            dc_daab: scale_coeff4(denested_third.1, scale),
            dc_dabb: scale_coeff4(denested_third.2, scale),
            dc_dbbb: scale_coeff4(denested_third.3, scale),
        })
    }

    /// Third-derivative tensor contracted with direction `dir`:
    ///   out[k,l] = sum_m f_{klm} dir[m]
    /// Rigid path uses the closed-form kernel. The flexible de-nested
    /// transport path contracts the cell-moment kernel analytically.
    fn row_primary_third_contracted_recompute(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if !self.effective_flex_active(block_states)? {
            let marginal = self.marginal_link_map(block_states[0].eta[row])?;
            let g = block_states[1].eta[row];
            let kern = RigidProbitKernel::new(
                marginal.q,
                g,
                self.z[row],
                self.y[row],
                self.weights[row],
                self.probit_frailty_scale(),
            )?;
            let t = rigid_transformed_third_contracted(marginal, &kern, dir[0], dir[1]);
            let mut out = Array2::<f64>::zeros((2, 2));
            out[[0, 0]] = t[0][0];
            out[[0, 1]] = t[0][1];
            out[[1, 0]] = t[1][0];
            out[[1, 1]] = t[1][1];
            return Ok(out);
        }
        if dir.iter().all(|value| value.abs() <= 0.0) {
            return Ok(Array2::<f64>::zeros((
                cache.primary.total,
                cache.primary.total,
            )));
        }
        if !row_ctx.intercept.is_finite() || !row_ctx.m_a.is_finite() || row_ctx.m_a <= 0.0 {
            return Err(
                "non-finite flexible row context in third-order directional contraction"
                    .to_string(),
            );
        }
        use crate::families::bernoulli_marginal_slope::exact_kernel as exact;

        let primary = &cache.primary;
        let point = self.primary_point_from_block_states(row, block_states, primary)?;
        let (q, b, beta_h_owned, beta_w_owned) = self.primary_point_components(&point, primary);
        let beta_h = beta_h_owned.as_ref();
        let beta_w = beta_w_owned.as_ref();
        let a = row_ctx.intercept;
        let r = primary.total;
        let marginal = self.marginal_link_map(q)?;
        let h_range = primary.h.as_ref();
        let w_range = primary.w.as_ref();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();
        let scale = self.probit_frailty_scale();
        let zero_family = vec![[0.0; 4]; r];

        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        let mut f_a_dir = 0.0;
        let mut f_aa_dir = 0.0;
        let mut f_u = Array1::<f64>::zeros(r);
        let mut f_au = Array1::<f64>::zeros(r);
        let mut f_au_dir = Array1::<f64>::zeros(r);
        let mut f_uv = Array2::<f64>::zeros((r, r));
        let mut f_uv_dir = Array2::<f64>::zeros((r, r));

        let cells = self.denested_partition_cells(a, b, beta_h, beta_w)?;
        for partition_cell in cells {
            let cell = partition_cell.cell;
            let z_mid = exact::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = exact::evaluate_cell_moments(cell, 15)?;

            let (dc_da_raw, dc_db_raw) = exact::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) = exact::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let denested_third = exact::denested_cell_third_partials(partition_cell.link_span);
            let dc_da = scale_coeff4(dc_da_raw, scale);
            let dc_db = scale_coeff4(dc_db_raw, scale);
            let dc_daa = scale_coeff4(dc_daa_raw, scale);
            let dc_dab = scale_coeff4(dc_dab_raw, scale);
            let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
            let dc_daab = scale_coeff4(denested_third.1, scale);
            let dc_dabb = scale_coeff4(denested_third.2, scale);
            let dc_dbbb = scale_coeff4(denested_third.3, scale);

            let mut coeff_u = vec![[0.0; 4]; r];
            let mut coeff_au = vec![[0.0; 4]; r];
            let mut coeff_bu = vec![[0.0; 4]; r];
            let mut coeff_aau = vec![[0.0; 4]; r];
            let mut coeff_abu = vec![[0.0; 4]; r];
            let mut coeff_bbu = vec![[0.0; 4]; r];

            coeff_u[1] = dc_db;
            coeff_au[1] = dc_dab;
            coeff_bu[1] = dc_dbb;
            coeff_aau[1] = dc_daab;
            coeff_abu[1] = dc_dabb;
            coeff_bbu[1] = dc_dbbb;

            if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
                for local_idx in 0..h_range.len() {
                    let basis_span = runtime.basis_cubic_at(local_idx, z_mid)?;
                    let idx = h_range.start + local_idx;
                    coeff_u[idx] =
                        scale_coeff4(exact::score_basis_cell_coefficients(basis_span, b), scale);
                    coeff_bu[idx] =
                        scale_coeff4(exact::score_basis_cell_coefficients(basis_span, 1.0), scale);
                }
            }

            if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
                for local_idx in 0..w_range.len() {
                    let basis_span = runtime.basis_cubic_at(local_idx, u_mid)?;
                    let idx = w_range.start + local_idx;
                    coeff_u[idx] =
                        scale_coeff4(exact::link_basis_cell_coefficients(basis_span, a, b), scale);
                    let (dc_aw_raw, dc_bw_raw) =
                        exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                    let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
                        exact::link_basis_cell_second_partials(basis_span, a, b);
                    coeff_au[idx] = scale_coeff4(dc_aw_raw, scale);
                    coeff_bu[idx] = scale_coeff4(dc_bw_raw, scale);
                    coeff_aau[idx] = scale_coeff4(dc_aaw_raw, scale);
                    coeff_abu[idx] = scale_coeff4(dc_abw_raw, scale);
                    coeff_bbu[idx] = scale_coeff4(dc_bbw_raw, scale);
                }
            }

            let coeff_jet = SparsePrimaryCoeffJetView::new(
                h_range,
                w_range,
                &coeff_u,
                &coeff_au,
                &coeff_bu,
                &coeff_aau,
                &coeff_abu,
                &coeff_bbu,
                &zero_family,
                &zero_family,
                &zero_family,
                &zero_family,
            );

            f_a += exact::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            f_aa += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &dc_daa,
                &state.moments,
            )?;

            for u in 1..r {
                f_u[u] +=
                    exact::cell_first_derivative_from_moments(&coeff_jet.first[u], &state.moments)?;
                f_au[u] += exact::cell_second_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_jet.first[u],
                    &coeff_jet.a_first[u],
                    &state.moments,
                )?;
            }
            let coeff_dir = coeff_jet.directional_family(coeff_jet.first, dir, COEFF_SUPPORT_BHW);
            let coeff_a_dir =
                coeff_jet.directional_family(coeff_jet.a_first, dir, COEFF_SUPPORT_BW);
            let coeff_aa_dir =
                coeff_jet.directional_family(coeff_jet.aa_first, dir, COEFF_SUPPORT_BW);

            f_a_dir += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &coeff_dir,
                &coeff_a_dir,
                &state.moments,
            )?;
            f_aa_dir += exact::cell_third_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &coeff_dir,
                &dc_daa,
                &coeff_a_dir,
                &coeff_a_dir,
                &coeff_aa_dir,
                &state.moments,
            )?;

            let mut coeff_u_dir = vec![[0.0; 4]; r];
            let mut coeff_au_dir = vec![[0.0; 4]; r];
            for u in 1..r {
                coeff_u_dir[u] = coeff_jet.param_directional_from_b_family(
                    coeff_jet.b_first,
                    u,
                    dir,
                    COEFF_SUPPORT_BHW,
                );
                coeff_au_dir[u] = coeff_jet.param_directional_from_b_family(
                    coeff_jet.ab_first,
                    u,
                    dir,
                    COEFF_SUPPORT_BW,
                );
            }

            for u in 1..r {
                f_au_dir[u] += exact::cell_third_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_jet.first[u],
                    &coeff_dir,
                    &coeff_jet.a_first[u],
                    &coeff_a_dir,
                    &coeff_u_dir[u],
                    &coeff_au_dir[u],
                    &state.moments,
                )?;
            }

            for u in 1..r {
                for v in u..r {
                    let second_coeff =
                        coeff_jet.pair_from_b_family(coeff_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                    let val = exact::cell_second_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &second_coeff,
                        &state.moments,
                    )?;
                    f_uv[[u, v]] += val;
                    if u != v {
                        f_uv[[v, u]] += val;
                    }

                    let third_coeff = coeff_jet.pair_directional_from_bb_family(
                        coeff_jet.bb_first,
                        u,
                        v,
                        dir,
                        COEFF_SUPPORT_BW,
                    );
                    let dir_val = exact::cell_third_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &coeff_dir,
                        &second_coeff,
                        &coeff_u_dir[u],
                        &coeff_u_dir[v],
                        &third_coeff,
                        &state.moments,
                    )?;
                    f_uv_dir[[u, v]] += dir_val;
                    if u != v {
                        f_uv_dir[[v, u]] += dir_val;
                    }
                }
            }
        }

        f_u[0] = -marginal.mu1;
        f_uv[[0, 0]] = -marginal.mu2;
        f_uv_dir[[0, 0]] = -dir[0] * marginal.mu3;

        let inv_f_a = 1.0 / f_a;
        let mut a_u = Array1::<f64>::zeros(r);
        for u in 0..r {
            a_u[u] = -f_u[u] * inv_f_a;
        }
        let mut a_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * a_u[v] + f_au[v] * a_u[u] + f_aa * a_u[u] * a_u[v])
                        * inv_f_a;
                a_uv[[u, v]] = val;
                a_uv[[v, u]] = val;
            }
        }
        let a_dir = a_u.dot(dir);
        let a_u_dir = a_uv.dot(dir);
        let mut a_uv_dir = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let n_dir = f_uv_dir[[u, v]]
                    + f_au_dir[u] * a_u[v]
                    + f_au[u] * a_u_dir[v]
                    + f_au_dir[v] * a_u[u]
                    + f_au[v] * a_u_dir[u]
                    + f_aa_dir * a_u[u] * a_u[v]
                    + f_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v]);
                let val = -(n_dir + f_a_dir * a_uv[[u, v]]) * inv_f_a;
                a_uv_dir[[u, v]] = val;
                a_uv_dir[[v, u]] = val;
            }
        }

        let z_obs = self.z[row];
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let eta_val = eval_coeff4_at(&obs.coeff, z_obs);

        let mut g_u_fixed = vec![[0.0; 4]; r];
        let mut g_au_fixed = vec![[0.0; 4]; r];
        let mut g_bu_fixed = vec![[0.0; 4]; r];
        let mut g_aau_fixed = vec![[0.0; 4]; r];
        let mut g_abu_fixed = vec![[0.0; 4]; r];
        let mut g_bbu_fixed = vec![[0.0; 4]; r];

        g_u_fixed[1] = obs.dc_db;
        g_au_fixed[1] = obs.dc_dab;
        g_bu_fixed[1] = obs.dc_dbb;
        g_aau_fixed[1] = obs.dc_daab;
        g_abu_fixed[1] = obs.dc_dabb;
        g_bbu_fixed[1] = obs.dc_dbbb;
        let scale = self.probit_frailty_scale();

        if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
            for local_idx in 0..h_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, z_obs)?;
                let idx = h_range.start + local_idx;
                g_u_fixed[idx] =
                    scale_coeff4(exact::score_basis_cell_coefficients(basis_span, b), scale);
                g_bu_fixed[idx] =
                    scale_coeff4(exact::score_basis_cell_coefficients(basis_span, 1.0), scale);
            }
        }

        if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let idx = w_range.start + local_idx;
                g_u_fixed[idx] =
                    scale_coeff4(exact::link_basis_cell_coefficients(basis_span, a, b), scale);
                let (dc_aw_raw, dc_bw_raw) =
                    exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
                    exact::link_basis_cell_second_partials(basis_span, a, b);
                g_au_fixed[idx] = scale_coeff4(dc_aw_raw, scale);
                g_bu_fixed[idx] = scale_coeff4(dc_bw_raw, scale);
                g_aau_fixed[idx] = scale_coeff4(dc_aaw_raw, scale);
                g_abu_fixed[idx] = scale_coeff4(dc_abw_raw, scale);
                g_bbu_fixed[idx] = scale_coeff4(dc_bbw_raw, scale);
            }
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            h_range,
            w_range,
            &g_u_fixed,
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &zero_family,
            &zero_family,
            &zero_family,
            &zero_family,
        );

        let g_a = eval_coeff4_at(&obs.dc_da, z_obs);
        let g_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let g_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let mut g_u = Array1::<f64>::zeros(r);
        let mut g_au = Array1::<f64>::zeros(r);
        let mut g_aau = Array1::<f64>::zeros(r);
        let mut g_uv = Array2::<f64>::zeros((r, r));
        let mut g_auv = Array2::<f64>::zeros((r, r));
        for u in 1..r {
            g_u[u] = eval_coeff4_at(&g_jet.first[u], z_obs);
            g_au[u] = eval_coeff4_at(&g_jet.a_first[u], z_obs);
            g_aau[u] = eval_coeff4_at(&g_jet.aa_first[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let second_coeff = g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                let val = eval_coeff4_at(&second_coeff, z_obs);
                g_uv[[u, v]] = val;
                g_uv[[v, u]] = val;

                let third_coeff = g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_BW);
                let third_val = eval_coeff4_at(&third_coeff, z_obs);
                g_auv[[u, v]] = third_val;
                g_auv[[v, u]] = third_val;
            }
        }

        let mut g_u_dir_fixed = vec![[0.0; 4]; r];
        let mut g_au_dir_fixed = vec![[0.0; 4]; r];
        let g_dir_fixed = g_jet.directional_family(g_jet.first, dir, COEFF_SUPPORT_BHW);
        let g_a_dir_fixed = g_jet.directional_family(g_jet.a_first, dir, COEFF_SUPPORT_BW);
        let g_aa_dir_fixed = g_jet.directional_family(g_jet.aa_first, dir, COEFF_SUPPORT_BW);
        let g_dir = eval_coeff4_at(&g_dir_fixed, z_obs);
        let g_a_dir = eval_coeff4_at(&g_a_dir_fixed, z_obs);
        let g_aa_dir = eval_coeff4_at(&g_aa_dir_fixed, z_obs);

        for u in 1..r {
            g_u_dir_fixed[u] =
                g_jet.param_directional_from_b_family(g_jet.b_first, u, dir, COEFF_SUPPORT_BHW);
            g_au_dir_fixed[u] =
                g_jet.param_directional_from_b_family(g_jet.ab_first, u, dir, COEFF_SUPPORT_BW);
        }

        let mut g_u_dir = Array1::<f64>::zeros(r);
        let mut g_uv_dir = Array2::<f64>::zeros((r, r));
        for u in 1..r {
            g_u_dir[u] = eval_coeff4_at(&g_u_dir_fixed[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let third_coeff = g_jet.pair_directional_from_bb_family(
                    g_jet.bb_first,
                    u,
                    v,
                    dir,
                    COEFF_SUPPORT_BW,
                );
                let val = eval_coeff4_at(&third_coeff, z_obs);
                g_uv_dir[[u, v]] = val;
                g_uv_dir[[v, u]] = val;
            }
        }

        let eta_u = g_a * &a_u + &g_u;
        let mut eta_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = g_a * a_uv[[u, v]]
                    + g_aa * a_u[u] * a_u[v]
                    + g_au[u] * a_u[v]
                    + g_au[v] * a_u[u]
                    + g_uv[[u, v]];
                eta_uv[[u, v]] = val;
                eta_uv[[v, u]] = val;
            }
        }
        let eta_dir = g_a * a_dir + g_dir;
        let eta_u_dir = eta_uv.dot(dir);
        let dg_a_dir = g_aa * a_dir + g_a_dir;
        let dg_aa_dir = g_aaa * a_dir + g_aa_dir;
        let mut dg_au_dir = Array1::<f64>::zeros(r);
        let mut dg_uv_dir = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            dg_au_dir[u] = g_aau[u] * a_dir + eval_coeff4_at(&g_au_dir_fixed[u], z_obs);
        }
        for u in 0..r {
            for v in u..r {
                let val = g_auv[[u, v]] * a_dir + g_uv_dir[[u, v]];
                dg_uv_dir[[u, v]] = val;
                dg_uv_dir[[v, u]] = val;
            }
        }

        let mut eta_uv_dir = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = dg_a_dir * a_uv[[u, v]]
                    + g_a * a_uv_dir[[u, v]]
                    + dg_aa_dir * a_u[u] * a_u[v]
                    + g_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v])
                    + dg_au_dir[u] * a_u[v]
                    + g_au[u] * a_u_dir[v]
                    + dg_au_dir[v] * a_u[u]
                    + g_au[v] * a_u_dir[u]
                    + dg_uv_dir[[u, v]];
                eta_uv_dir[[u, v]] = val;
                eta_uv_dir[[v, u]] = val;
            }
        }

        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let m = s_y * eta_val;
        let (k1, k2, k3, _) = signed_probit_neglog_derivatives_up_to_fourth(m, w_i)?;
        let u1 = s_y * k1;
        let u3 = s_y * k3;

        let mut out = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = u3 * eta_u[u] * eta_u[v] * eta_dir
                    + k2 * (eta_uv[[u, v]] * eta_dir
                        + eta_u[u] * eta_u_dir[v]
                        + eta_u[v] * eta_u_dir[u])
                    + u1 * eta_uv_dir[[u, v]];
                out[[u, v]] = val;
                out[[v, u]] = val;
            }
        }
        Ok(out)
    }

    /// Fourth-derivative tensor contracted with two directions dir_u, dir_v:
    ///   out[k,l] = sum_{m,n} f_{klmn} dir_u[m] dir_v[n]
    /// Rigid path uses the closed-form kernel. The flexible de-nested
    /// transport path contracts the cell-moment kernel analytically.
    fn row_primary_fourth_contracted_recompute_ordered(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if !self.effective_flex_active(block_states)? {
            let marginal = self.marginal_link_map(block_states[0].eta[row])?;
            let g = block_states[1].eta[row];
            let kern = RigidProbitKernel::new(
                marginal.q,
                g,
                self.z[row],
                self.y[row],
                self.weights[row],
                self.probit_frailty_scale(),
            )?;
            let f = rigid_transformed_fourth_contracted(
                marginal, &kern, dir_u[0], dir_u[1], dir_v[0], dir_v[1],
            );
            let mut out = Array2::<f64>::zeros((2, 2));
            out[[0, 0]] = f[0][0];
            out[[0, 1]] = f[0][1];
            out[[1, 0]] = f[1][0];
            out[[1, 1]] = f[1][1];
            return Ok(out);
        }
        if dir_u.iter().all(|value| value.abs() <= 0.0)
            || dir_v.iter().all(|value| value.abs() <= 0.0)
        {
            return Ok(Array2::<f64>::zeros((
                cache.primary.total,
                cache.primary.total,
            )));
        }
        if !row_ctx.intercept.is_finite() || !row_ctx.m_a.is_finite() || row_ctx.m_a <= 0.0 {
            return Err(
                "non-finite flexible row context in fourth-order directional contraction"
                    .to_string(),
            );
        }
        use crate::families::bernoulli_marginal_slope::exact_kernel as exact;

        let primary = &cache.primary;
        let point = self.primary_point_from_block_states(row, block_states, primary)?;
        let (q, b, beta_h_owned, beta_w_owned) = self.primary_point_components(&point, primary);
        let beta_h = beta_h_owned.as_ref();
        let beta_w = beta_w_owned.as_ref();
        let a = row_ctx.intercept;
        let r = primary.total;
        let marginal = self.marginal_link_map(q)?;
        let h_range = primary.h.as_ref();
        let w_range = primary.w.as_ref();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();
        let scale = self.probit_frailty_scale();

        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        let mut f_u = Array1::<f64>::zeros(r);
        let mut f_au = Array1::<f64>::zeros(r);
        let mut f_uv = Array2::<f64>::zeros((r, r));

        let mut f_a_u = 0.0;
        let mut f_aa_u = 0.0;
        let mut f_au_u = Array1::<f64>::zeros(r);
        let mut f_uv_u = Array2::<f64>::zeros((r, r));

        let mut f_a_v = 0.0;
        let mut f_aa_v = 0.0;
        let mut f_au_v = Array1::<f64>::zeros(r);
        let mut f_uv_v = Array2::<f64>::zeros((r, r));

        let mut f_a_uv = 0.0;
        let mut f_aa_uv = 0.0;
        let mut f_au_uv = Array1::<f64>::zeros(r);
        let mut f_uv_uv = Array2::<f64>::zeros((r, r));

        let cells = self.denested_partition_cells(a, b, beta_h, beta_w)?;
        for partition_cell in cells {
            let cell = partition_cell.cell;
            let z_mid = exact::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = exact::evaluate_cell_moments(cell, 21)?;

            let (dc_da_raw, dc_db_raw) = exact::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) = exact::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let denested_third = exact::denested_cell_third_partials(partition_cell.link_span);
            let dc_da = scale_coeff4(dc_da_raw, scale);
            let dc_db = scale_coeff4(dc_db_raw, scale);
            let dc_daa = scale_coeff4(dc_daa_raw, scale);
            let dc_dab = scale_coeff4(dc_dab_raw, scale);
            let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
            let dc_daab = scale_coeff4(denested_third.1, scale);
            let dc_dabb = scale_coeff4(denested_third.2, scale);
            let dc_dbbb = scale_coeff4(denested_third.3, scale);

            let mut coeff_u = vec![[0.0; 4]; r];
            let mut coeff_au = vec![[0.0; 4]; r];
            let mut coeff_bu = vec![[0.0; 4]; r];
            let mut coeff_aau = vec![[0.0; 4]; r];
            let mut coeff_abu = vec![[0.0; 4]; r];
            let mut coeff_bbu = vec![[0.0; 4]; r];
            let mut coeff_aaau = vec![[0.0; 4]; r];
            let mut coeff_aabu = vec![[0.0; 4]; r];
            let mut coeff_abbu = vec![[0.0; 4]; r];
            let mut coeff_bbbu = vec![[0.0; 4]; r];

            coeff_u[1] = dc_db;
            coeff_au[1] = dc_dab;
            coeff_bu[1] = dc_dbb;
            coeff_aau[1] = dc_daab;
            coeff_abu[1] = dc_dabb;
            coeff_bbu[1] = dc_dbbb;

            if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
                for local_idx in 0..h_range.len() {
                    let basis_span = runtime.basis_cubic_at(local_idx, z_mid)?;
                    let idx = h_range.start + local_idx;
                    coeff_u[idx] =
                        scale_coeff4(exact::score_basis_cell_coefficients(basis_span, b), scale);
                    coeff_bu[idx] =
                        scale_coeff4(exact::score_basis_cell_coefficients(basis_span, 1.0), scale);
                }
            }

            if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
                for local_idx in 0..w_range.len() {
                    let basis_span = runtime.basis_cubic_at(local_idx, u_mid)?;
                    let idx = w_range.start + local_idx;
                    coeff_u[idx] =
                        scale_coeff4(exact::link_basis_cell_coefficients(basis_span, a, b), scale);
                    let (dc_aw_raw, dc_bw_raw) =
                        exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                    let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
                        exact::link_basis_cell_second_partials(basis_span, a, b);
                    let (dc_aaaw, dc_aabw, dc_abbw, dc_bbbw) =
                        exact::link_basis_cell_third_partials(basis_span);
                    coeff_au[idx] = scale_coeff4(dc_aw_raw, scale);
                    coeff_bu[idx] = scale_coeff4(dc_bw_raw, scale);
                    coeff_aau[idx] = scale_coeff4(dc_aaw_raw, scale);
                    coeff_abu[idx] = scale_coeff4(dc_abw_raw, scale);
                    coeff_bbu[idx] = scale_coeff4(dc_bbw_raw, scale);
                    coeff_aaau[idx] = scale_coeff4(dc_aaaw, scale);
                    coeff_aabu[idx] = scale_coeff4(dc_aabw, scale);
                    coeff_abbu[idx] = scale_coeff4(dc_abbw, scale);
                    coeff_bbbu[idx] = scale_coeff4(dc_bbbw, scale);
                }
            }

            let coeff_jet = SparsePrimaryCoeffJetView::new(
                h_range,
                w_range,
                &coeff_u,
                &coeff_au,
                &coeff_bu,
                &coeff_aau,
                &coeff_abu,
                &coeff_bbu,
                &coeff_aaau,
                &coeff_aabu,
                &coeff_abbu,
                &coeff_bbbu,
            );

            f_a += exact::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            f_aa += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &dc_daa,
                &state.moments,
            )?;

            for u in 1..r {
                f_u[u] +=
                    exact::cell_first_derivative_from_moments(&coeff_jet.first[u], &state.moments)?;
                f_au[u] += exact::cell_second_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_jet.first[u],
                    &coeff_jet.a_first[u],
                    &state.moments,
                )?;
            }
            let coeff_dir_u =
                coeff_jet.directional_family(coeff_jet.first, dir_u, COEFF_SUPPORT_BHW);
            let coeff_dir_v =
                coeff_jet.directional_family(coeff_jet.first, dir_v, COEFF_SUPPORT_BHW);
            let coeff_a_dir_u =
                coeff_jet.directional_family(coeff_jet.a_first, dir_u, COEFF_SUPPORT_BW);
            let coeff_a_dir_v =
                coeff_jet.directional_family(coeff_jet.a_first, dir_v, COEFF_SUPPORT_BW);
            let coeff_aa_dir_u =
                coeff_jet.directional_family(coeff_jet.aa_first, dir_u, COEFF_SUPPORT_BW);
            let coeff_aa_dir_v =
                coeff_jet.directional_family(coeff_jet.aa_first, dir_v, COEFF_SUPPORT_BW);

            f_a_u += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &coeff_dir_u,
                &coeff_a_dir_u,
                &state.moments,
            )?;
            f_a_v += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &coeff_dir_v,
                &coeff_a_dir_v,
                &state.moments,
            )?;
            f_aa_u += exact::cell_third_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &coeff_dir_u,
                &dc_daa,
                &coeff_a_dir_u,
                &coeff_a_dir_u,
                &coeff_aa_dir_u,
                &state.moments,
            )?;
            f_aa_v += exact::cell_third_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &coeff_dir_v,
                &dc_daa,
                &coeff_a_dir_v,
                &coeff_a_dir_v,
                &coeff_aa_dir_v,
                &state.moments,
            )?;

            let coeff_dir_uv = coeff_jet.mixed_directional_from_b_family(
                coeff_jet.b_first,
                dir_u,
                dir_v,
                COEFF_SUPPORT_BHW,
            );
            let coeff_a_dir_uv = coeff_jet.mixed_directional_from_b_family(
                coeff_jet.ab_first,
                dir_u,
                dir_v,
                COEFF_SUPPORT_BW,
            );
            let coeff_aa_dir_uv = coeff_jet.mixed_directional_from_b_family(
                coeff_jet.aab_first,
                dir_u,
                dir_v,
                COEFF_SUPPORT_W,
            );

            f_a_uv += exact::cell_third_derivative_from_moments(
                cell,
                &dc_da,
                &coeff_dir_u,
                &coeff_dir_v,
                &coeff_a_dir_u,
                &coeff_a_dir_v,
                &coeff_dir_uv,
                &coeff_a_dir_uv,
                &state.moments,
            )?;
            f_aa_uv += exact::cell_fourth_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &coeff_dir_u,
                &coeff_dir_v,
                &dc_daa,
                &coeff_a_dir_u,
                &coeff_a_dir_v,
                &coeff_a_dir_u,
                &coeff_a_dir_v,
                &coeff_dir_uv,
                &coeff_aa_dir_u,
                &coeff_aa_dir_v,
                &coeff_a_dir_uv,
                &coeff_a_dir_uv,
                &coeff_aa_dir_uv,
                &state.moments,
            )?;

            let mut coeff_u_dir_u = vec![[0.0; 4]; r];
            let mut coeff_u_dir_v = vec![[0.0; 4]; r];
            let mut coeff_u_dir_uv = vec![[0.0; 4]; r];
            let mut coeff_au_dir_u = vec![[0.0; 4]; r];
            let mut coeff_au_dir_v = vec![[0.0; 4]; r];
            let mut coeff_au_dir_uv = vec![[0.0; 4]; r];
            for u in 1..r {
                coeff_u_dir_u[u] = coeff_jet.param_directional_from_b_family(
                    coeff_jet.b_first,
                    u,
                    dir_u,
                    COEFF_SUPPORT_BHW,
                );
                coeff_u_dir_v[u] = coeff_jet.param_directional_from_b_family(
                    coeff_jet.b_first,
                    u,
                    dir_v,
                    COEFF_SUPPORT_BHW,
                );
                coeff_u_dir_uv[u] = coeff_jet.param_mixed_from_bb_family(
                    coeff_jet.bb_first,
                    u,
                    dir_u,
                    dir_v,
                    COEFF_SUPPORT_BW,
                );
                coeff_au_dir_u[u] = coeff_jet.param_directional_from_b_family(
                    coeff_jet.ab_first,
                    u,
                    dir_u,
                    COEFF_SUPPORT_BW,
                );
                coeff_au_dir_v[u] = coeff_jet.param_directional_from_b_family(
                    coeff_jet.ab_first,
                    u,
                    dir_v,
                    COEFF_SUPPORT_BW,
                );
                coeff_au_dir_uv[u] = coeff_jet.param_mixed_from_bb_family(
                    coeff_jet.abb_first,
                    u,
                    dir_u,
                    dir_v,
                    COEFF_SUPPORT_W,
                );
            }

            for u in 1..r {
                f_au_u[u] += exact::cell_third_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_u[u],
                    &coeff_dir_u,
                    &coeff_au[u],
                    &coeff_a_dir_u,
                    &coeff_u_dir_u[u],
                    &coeff_au_dir_u[u],
                    &state.moments,
                )?;
                f_au_v[u] += exact::cell_third_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_u[u],
                    &coeff_dir_v,
                    &coeff_au[u],
                    &coeff_a_dir_v,
                    &coeff_u_dir_v[u],
                    &coeff_au_dir_v[u],
                    &state.moments,
                )?;
                f_au_uv[u] += exact::cell_fourth_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_u[u],
                    &coeff_dir_u,
                    &coeff_dir_v,
                    &coeff_au[u],
                    &coeff_a_dir_u,
                    &coeff_a_dir_v,
                    &coeff_u_dir_u[u],
                    &coeff_u_dir_v[u],
                    &coeff_dir_uv,
                    &coeff_au_dir_u[u],
                    &coeff_au_dir_v[u],
                    &coeff_a_dir_uv,
                    &coeff_u_dir_uv[u],
                    &coeff_au_dir_uv[u],
                    &state.moments,
                )?;
            }

            for u in 1..r {
                for v in u..r {
                    let second_coeff =
                        coeff_jet.pair_from_b_family(coeff_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                    let base_val = exact::cell_second_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &second_coeff,
                        &state.moments,
                    )?;
                    f_uv[[u, v]] += base_val;
                    if u != v {
                        f_uv[[v, u]] += base_val;
                    }

                    let third_u = coeff_jet.pair_directional_from_bb_family(
                        coeff_jet.bb_first,
                        u,
                        v,
                        dir_u,
                        COEFF_SUPPORT_BW,
                    );
                    let third_v = coeff_jet.pair_directional_from_bb_family(
                        coeff_jet.bb_first,
                        u,
                        v,
                        dir_v,
                        COEFF_SUPPORT_BW,
                    );
                    let fourth_uv = coeff_jet.pair_mixed_from_bbb_family(
                        coeff_jet.bbb_first,
                        u,
                        v,
                        dir_u,
                        dir_v,
                        COEFF_SUPPORT_W,
                    );

                    let dir_u_val = exact::cell_third_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &coeff_dir_u,
                        &second_coeff,
                        &coeff_u_dir_u[u],
                        &coeff_u_dir_u[v],
                        &third_u,
                        &state.moments,
                    )?;
                    let dir_v_val = exact::cell_third_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &coeff_dir_v,
                        &second_coeff,
                        &coeff_u_dir_v[u],
                        &coeff_u_dir_v[v],
                        &third_v,
                        &state.moments,
                    )?;
                    let mix_val = exact::cell_fourth_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &coeff_dir_u,
                        &coeff_dir_v,
                        &second_coeff,
                        &coeff_u_dir_u[u],
                        &coeff_u_dir_v[u],
                        &coeff_u_dir_u[v],
                        &coeff_u_dir_v[v],
                        &coeff_dir_uv,
                        &third_u,
                        &third_v,
                        &coeff_u_dir_uv[u],
                        &coeff_u_dir_uv[v],
                        &fourth_uv,
                        &state.moments,
                    )?;
                    f_uv_u[[u, v]] += dir_u_val;
                    f_uv_v[[u, v]] += dir_v_val;
                    f_uv_uv[[u, v]] += mix_val;
                    if u != v {
                        f_uv_u[[v, u]] += dir_u_val;
                        f_uv_v[[v, u]] += dir_v_val;
                        f_uv_uv[[v, u]] += mix_val;
                    }
                }
            }
        }

        f_u[0] = -marginal.mu1;
        f_uv[[0, 0]] = -marginal.mu2;
        f_uv_u[[0, 0]] = -dir_u[0] * marginal.mu3;
        f_uv_v[[0, 0]] = -dir_v[0] * marginal.mu3;
        f_uv_uv[[0, 0]] = -dir_u[0] * dir_v[0] * marginal.mu4;

        let inv_f_a = 1.0 / f_a;
        let mut a_u = Array1::<f64>::zeros(r);
        for u in 0..r {
            a_u[u] = -f_u[u] * inv_f_a;
        }
        let mut a_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * a_u[v] + f_au[v] * a_u[u] + f_aa * a_u[u] * a_u[v])
                        * inv_f_a;
                a_uv[[u, v]] = val;
                a_uv[[v, u]] = val;
            }
        }
        let a_u_dir_u = a_uv.dot(dir_u);
        let a_u_dir_v = a_uv.dot(dir_v);
        let mut a_uv_u = Array2::<f64>::zeros((r, r));
        let mut a_uv_v = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let n_u = f_uv_u[[u, v]]
                    + f_au_u[u] * a_u[v]
                    + f_au[u] * a_u_dir_u[v]
                    + f_au_u[v] * a_u[u]
                    + f_au[v] * a_u_dir_u[u]
                    + f_aa_u * a_u[u] * a_u[v]
                    + f_aa * (a_u_dir_u[u] * a_u[v] + a_u[u] * a_u_dir_u[v]);
                let val_u = -(n_u + f_a_u * a_uv[[u, v]]) * inv_f_a;
                a_uv_u[[u, v]] = val_u;
                a_uv_u[[v, u]] = val_u;

                let n_v = f_uv_v[[u, v]]
                    + f_au_v[u] * a_u[v]
                    + f_au[u] * a_u_dir_v[v]
                    + f_au_v[v] * a_u[u]
                    + f_au[v] * a_u_dir_v[u]
                    + f_aa_v * a_u[u] * a_u[v]
                    + f_aa * (a_u_dir_v[u] * a_u[v] + a_u[u] * a_u_dir_v[v]);
                let val_v = -(n_v + f_a_v * a_uv[[u, v]]) * inv_f_a;
                a_uv_v[[u, v]] = val_v;
                a_uv_v[[v, u]] = val_v;
            }
        }
        let a_u_uv = a_uv_u.dot(dir_v);
        let mut a_uv_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let n_uv = f_uv_uv[[u, v]]
                    + f_au_uv[u] * a_u[v]
                    + f_au_u[u] * a_u_dir_v[v]
                    + f_au_v[u] * a_u_dir_u[v]
                    + f_au[u] * a_u_uv[v]
                    + f_au_uv[v] * a_u[u]
                    + f_au_u[v] * a_u_dir_v[u]
                    + f_au_v[v] * a_u_dir_u[u]
                    + f_au[v] * a_u_uv[u]
                    + f_aa_uv * a_u[u] * a_u[v]
                    + f_aa_u * (a_u_dir_v[u] * a_u[v] + a_u[u] * a_u_dir_v[v])
                    + f_aa_v * (a_u_dir_u[u] * a_u[v] + a_u[u] * a_u_dir_u[v])
                    + f_aa
                        * (a_u_uv[u] * a_u[v]
                            + a_u_dir_u[u] * a_u_dir_v[v]
                            + a_u_dir_v[u] * a_u_dir_u[v]
                            + a_u[u] * a_u_uv[v]);
                let val = -(n_uv
                    + f_a_v * a_uv_u[[u, v]]
                    + f_a_u * a_uv_v[[u, v]]
                    + f_a_uv * a_uv[[u, v]])
                    * inv_f_a;
                a_uv_uv[[u, v]] = val;
                a_uv_uv[[v, u]] = val;
            }
        }

        let z_obs = self.z[row];
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let eta_val = eval_coeff4_at(&obs.coeff, z_obs);

        let mut g_u_fixed = vec![[0.0; 4]; r];
        let mut g_au_fixed = vec![[0.0; 4]; r];
        let mut g_bu_fixed = vec![[0.0; 4]; r];
        let mut g_aau_fixed = vec![[0.0; 4]; r];
        let mut g_abu_fixed = vec![[0.0; 4]; r];
        let mut g_bbu_fixed = vec![[0.0; 4]; r];
        let mut g_aaau_fixed = vec![[0.0; 4]; r];
        let mut g_aabu_fixed = vec![[0.0; 4]; r];
        let mut g_abbu_fixed = vec![[0.0; 4]; r];
        let mut g_bbbu_fixed = vec![[0.0; 4]; r];

        g_u_fixed[1] = obs.dc_db;
        g_au_fixed[1] = obs.dc_dab;
        g_bu_fixed[1] = obs.dc_dbb;
        g_aau_fixed[1] = obs.dc_daab;
        g_abu_fixed[1] = obs.dc_dabb;
        g_bbu_fixed[1] = obs.dc_dbbb;

        if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
            for local_idx in 0..h_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, z_obs)?;
                let idx = h_range.start + local_idx;
                g_u_fixed[idx] =
                    scale_coeff4(exact::score_basis_cell_coefficients(basis_span, b), scale);
                g_bu_fixed[idx] =
                    scale_coeff4(exact::score_basis_cell_coefficients(basis_span, 1.0), scale);
            }
        }
        if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let idx = w_range.start + local_idx;
                g_u_fixed[idx] =
                    scale_coeff4(exact::link_basis_cell_coefficients(basis_span, a, b), scale);
                let (dc_aw_raw, dc_bw_raw) =
                    exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
                    exact::link_basis_cell_second_partials(basis_span, a, b);
                let (dc_aaaw, dc_aabw, dc_abbw, dc_bbbw) =
                    exact::link_basis_cell_third_partials(basis_span);
                g_au_fixed[idx] = scale_coeff4(dc_aw_raw, scale);
                g_bu_fixed[idx] = scale_coeff4(dc_bw_raw, scale);
                g_aau_fixed[idx] = scale_coeff4(dc_aaw_raw, scale);
                g_abu_fixed[idx] = scale_coeff4(dc_abw_raw, scale);
                g_bbu_fixed[idx] = scale_coeff4(dc_bbw_raw, scale);
                g_aaau_fixed[idx] = scale_coeff4(dc_aaaw, scale);
                g_aabu_fixed[idx] = scale_coeff4(dc_aabw, scale);
                g_abbu_fixed[idx] = scale_coeff4(dc_abbw, scale);
                g_bbbu_fixed[idx] = scale_coeff4(dc_bbbw, scale);
            }
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            h_range,
            w_range,
            &g_u_fixed,
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &g_aaau_fixed,
            &g_aabu_fixed,
            &g_abbu_fixed,
            &g_bbbu_fixed,
        );

        let g_a = eval_coeff4_at(&obs.dc_da, z_obs);
        let g_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let g_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let mut g_u = Array1::<f64>::zeros(r);
        let mut g_au = Array1::<f64>::zeros(r);
        let mut g_aau = Array1::<f64>::zeros(r);
        let mut g_aaau = Array1::<f64>::zeros(r);
        let mut g_uv = Array2::<f64>::zeros((r, r));
        let mut g_auv = Array2::<f64>::zeros((r, r));
        let mut g_aauv = Array2::<f64>::zeros((r, r));
        for u in 1..r {
            g_u[u] = eval_coeff4_at(&g_jet.first[u], z_obs);
            g_au[u] = eval_coeff4_at(&g_jet.a_first[u], z_obs);
            g_aau[u] = eval_coeff4_at(&g_jet.aa_first[u], z_obs);
            g_aaau[u] = eval_coeff4_at(&g_jet.aaa_first[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let second_coeff = g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                let val = eval_coeff4_at(&second_coeff, z_obs);
                g_uv[[u, v]] = val;
                g_uv[[v, u]] = val;

                let third_coeff = g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_BW);
                let fourth_coeff = g_jet.pair_from_b_family(g_jet.aab_first, u, v, COEFF_SUPPORT_W);
                let third_val = eval_coeff4_at(&third_coeff, z_obs);
                let fourth_val = eval_coeff4_at(&fourth_coeff, z_obs);
                g_auv[[u, v]] = third_val;
                g_auv[[v, u]] = third_val;
                g_aauv[[u, v]] = fourth_val;
                g_aauv[[v, u]] = fourth_val;
            }
        }

        let g_dir_u_fixed = g_jet.directional_family(g_jet.first, dir_u, COEFF_SUPPORT_BHW);
        let g_dir_v_fixed = g_jet.directional_family(g_jet.first, dir_v, COEFF_SUPPORT_BHW);
        let g_a_dir_u_fixed = g_jet.directional_family(g_jet.a_first, dir_u, COEFF_SUPPORT_BW);
        let g_a_dir_v_fixed = g_jet.directional_family(g_jet.a_first, dir_v, COEFF_SUPPORT_BW);
        let g_aa_dir_u_fixed = g_jet.directional_family(g_jet.aa_first, dir_u, COEFF_SUPPORT_BW);
        let g_aa_dir_v_fixed = g_jet.directional_family(g_jet.aa_first, dir_v, COEFF_SUPPORT_BW);
        let g_dir_uv_fixed =
            g_jet.mixed_directional_from_b_family(g_jet.b_first, dir_u, dir_v, COEFF_SUPPORT_BHW);
        let g_a_dir_uv_fixed =
            g_jet.mixed_directional_from_b_family(g_jet.ab_first, dir_u, dir_v, COEFF_SUPPORT_BW);
        let g_aa_dir_uv_fixed =
            g_jet.mixed_directional_from_b_family(g_jet.aab_first, dir_u, dir_v, COEFF_SUPPORT_W);

        let g_dir_u = eval_coeff4_at(&g_dir_u_fixed, z_obs);
        let g_dir_v = eval_coeff4_at(&g_dir_v_fixed, z_obs);
        let g_dir_uv = eval_coeff4_at(&g_dir_uv_fixed, z_obs);
        let g_a_u_fixed = eval_coeff4_at(&g_a_dir_u_fixed, z_obs);
        let g_a_v_fixed = eval_coeff4_at(&g_a_dir_v_fixed, z_obs);
        let g_aa_u_fixed = eval_coeff4_at(&g_aa_dir_u_fixed, z_obs);
        let g_aa_v_fixed = eval_coeff4_at(&g_aa_dir_v_fixed, z_obs);
        let g_a_uv_fixed = eval_coeff4_at(&g_a_dir_uv_fixed, z_obs);
        let g_aa_uv_fixed = eval_coeff4_at(&g_aa_dir_uv_fixed, z_obs);

        let mut g_u_u_fixed = Array1::<f64>::zeros(r);
        let mut g_u_v_fixed = Array1::<f64>::zeros(r);
        let mut g_u_uv_fixed = Array1::<f64>::zeros(r);
        let mut g_au_u_fixed = Array1::<f64>::zeros(r);
        let mut g_au_v_fixed = Array1::<f64>::zeros(r);
        let mut g_au_uv_fixed = Array1::<f64>::zeros(r);
        let mut g_uv_u_fixed = Array2::<f64>::zeros((r, r));
        let mut g_uv_v_fixed = Array2::<f64>::zeros((r, r));
        let mut g_uv_uv_fixed = Array2::<f64>::zeros((r, r));
        let mut g_auv_u_fixed = Array2::<f64>::zeros((r, r));
        let mut g_auv_v_fixed = Array2::<f64>::zeros((r, r));

        for u in 1..r {
            let tmp_u =
                g_jet.param_directional_from_b_family(g_jet.b_first, u, dir_u, COEFF_SUPPORT_BHW);
            let tmp_v =
                g_jet.param_directional_from_b_family(g_jet.b_first, u, dir_v, COEFF_SUPPORT_BHW);
            let tmp_uv =
                g_jet.param_mixed_from_bb_family(g_jet.bb_first, u, dir_u, dir_v, COEFF_SUPPORT_BW);
            let tmp_au_u =
                g_jet.param_directional_from_b_family(g_jet.ab_first, u, dir_u, COEFF_SUPPORT_BW);
            let tmp_au_v =
                g_jet.param_directional_from_b_family(g_jet.ab_first, u, dir_v, COEFF_SUPPORT_BW);
            let tmp_au_uv =
                g_jet.param_mixed_from_bb_family(g_jet.abb_first, u, dir_u, dir_v, COEFF_SUPPORT_W);
            g_u_u_fixed[u] = eval_coeff4_at(&tmp_u, z_obs);
            g_u_v_fixed[u] = eval_coeff4_at(&tmp_v, z_obs);
            g_u_uv_fixed[u] = eval_coeff4_at(&tmp_uv, z_obs);
            g_au_u_fixed[u] = eval_coeff4_at(&tmp_au_u, z_obs);
            g_au_v_fixed[u] = eval_coeff4_at(&tmp_au_v, z_obs);
            g_au_uv_fixed[u] = eval_coeff4_at(&tmp_au_uv, z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let third_u = g_jet.pair_directional_from_bb_family(
                    g_jet.bb_first,
                    u,
                    v,
                    dir_u,
                    COEFF_SUPPORT_BW,
                );
                let third_v = g_jet.pair_directional_from_bb_family(
                    g_jet.bb_first,
                    u,
                    v,
                    dir_v,
                    COEFF_SUPPORT_BW,
                );
                let fourth_uv = g_jet.pair_mixed_from_bbb_family(
                    g_jet.bbb_first,
                    u,
                    v,
                    dir_u,
                    dir_v,
                    COEFF_SUPPORT_W,
                );
                let a_third_u = g_jet.pair_directional_from_bb_family(
                    g_jet.abb_first,
                    u,
                    v,
                    dir_u,
                    COEFF_SUPPORT_W,
                );
                let a_third_v = g_jet.pair_directional_from_bb_family(
                    g_jet.abb_first,
                    u,
                    v,
                    dir_v,
                    COEFF_SUPPORT_W,
                );
                let vu = eval_coeff4_at(&third_u, z_obs);
                let vv = eval_coeff4_at(&third_v, z_obs);
                let vuv = eval_coeff4_at(&fourth_uv, z_obs);
                g_uv_u_fixed[[u, v]] = vu;
                g_uv_v_fixed[[u, v]] = vv;
                g_uv_uv_fixed[[u, v]] = vuv;
                g_uv_u_fixed[[v, u]] = vu;
                g_uv_v_fixed[[v, u]] = vv;
                g_uv_uv_fixed[[v, u]] = vuv;
                let atu = eval_coeff4_at(&a_third_u, z_obs);
                let atv = eval_coeff4_at(&a_third_v, z_obs);
                g_auv_u_fixed[[u, v]] = atu;
                g_auv_v_fixed[[u, v]] = atv;
                g_auv_u_fixed[[v, u]] = atu;
                g_auv_v_fixed[[v, u]] = atv;
            }
        }

        let eta_u = g_a * &a_u + &g_u;
        let mut eta_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = g_a * a_uv[[u, v]]
                    + g_aa * a_u[u] * a_u[v]
                    + g_au[u] * a_u[v]
                    + g_au[v] * a_u[u]
                    + g_uv[[u, v]];
                eta_uv[[u, v]] = val;
                eta_uv[[v, u]] = val;
            }
        }

        let a_dir_u = a_u.dot(dir_u);
        let a_dir_v = a_u.dot(dir_v);
        let g_a_u = g_aa * a_dir_u + g_a_u_fixed;
        let g_a_v = g_aa * a_dir_v + g_a_v_fixed;
        let g_aa_u = g_aaa * a_dir_u + g_aa_u_fixed;
        let g_aa_v = g_aaa * a_dir_v + g_aa_v_fixed;

        let mut g_u_u = Array1::<f64>::zeros(r);
        let mut g_u_v = Array1::<f64>::zeros(r);
        let mut g_au_u = Array1::<f64>::zeros(r);
        let mut g_au_v = Array1::<f64>::zeros(r);
        for u in 0..r {
            g_u_u[u] = g_au[u] * a_dir_u + g_u_u_fixed[u];
            g_u_v[u] = g_au[u] * a_dir_v + g_u_v_fixed[u];
            g_au_u[u] = g_aau[u] * a_dir_u + g_au_u_fixed[u];
            g_au_v[u] = g_aau[u] * a_dir_v + g_au_v_fixed[u];
        }

        let mut eta_uv_u = Array2::<f64>::zeros((r, r));
        let mut eta_uv_v = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let g_uv_u = g_auv[[u, v]] * a_dir_u + g_uv_u_fixed[[u, v]];
                let g_uv_v = g_auv[[u, v]] * a_dir_v + g_uv_v_fixed[[u, v]];
                let val_u = g_a_u * a_uv[[u, v]]
                    + g_a * a_uv_u[[u, v]]
                    + g_aa_u * a_u[u] * a_u[v]
                    + g_aa * (a_u_dir_u[u] * a_u[v] + a_u[u] * a_u_dir_u[v])
                    + g_au_u[u] * a_u[v]
                    + g_au[u] * a_u_dir_u[v]
                    + g_au_u[v] * a_u[u]
                    + g_au[v] * a_u_dir_u[u]
                    + g_uv_u;
                eta_uv_u[[u, v]] = val_u;
                eta_uv_u[[v, u]] = val_u;

                let val_v = g_a_v * a_uv[[u, v]]
                    + g_a * a_uv_v[[u, v]]
                    + g_aa_v * a_u[u] * a_u[v]
                    + g_aa * (a_u_dir_v[u] * a_u[v] + a_u[u] * a_u_dir_v[v])
                    + g_au_v[u] * a_u[v]
                    + g_au[u] * a_u_dir_v[v]
                    + g_au_v[v] * a_u[u]
                    + g_au[v] * a_u_dir_v[u]
                    + g_uv_v;
                eta_uv_v[[u, v]] = val_v;
                eta_uv_v[[v, u]] = val_v;
            }
        }

        let a_dir_uv = a_u_dir_u.dot(dir_v);
        let g_a_uv = g_aaa * a_dir_u * a_dir_v
            + g_aa * a_dir_uv
            + g_aa_u_fixed * a_dir_v
            + g_aa_v_fixed * a_dir_u
            + g_a_uv_fixed;
        let g_aa_uv = g_aaau.dot(dir_u) * a_dir_v
            + g_aaau.dot(dir_v) * a_dir_u
            + g_aaa * a_dir_uv
            + g_aa_uv_fixed;
        let mut g_u_uv = Array1::<f64>::zeros(r);
        let mut g_au_uv = Array1::<f64>::zeros(r);
        for u in 0..r {
            g_u_uv[u] = g_aau[u] * a_dir_u * a_dir_v
                + g_au[u] * a_dir_uv
                + g_au_u_fixed[u] * a_dir_v
                + g_au_v_fixed[u] * a_dir_u
                + g_u_uv_fixed[u];
            let row_u_u = g_aauv.row(u).dot(dir_u);
            let row_u_v = g_aauv.row(u).dot(dir_v);
            g_au_uv[u] = g_aaau[u] * a_dir_u * a_dir_v
                + g_aau[u] * a_dir_uv
                + row_u_u * a_dir_v
                + row_u_v * a_dir_u
                + g_au_uv_fixed[u];
        }

        let mut eta_uv_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let g_uv_uv = g_aauv[[u, v]] * a_dir_u * a_dir_v
                    + g_auv[[u, v]] * a_dir_uv
                    + g_auv_u_fixed[[u, v]] * a_dir_v
                    + g_auv_v_fixed[[u, v]] * a_dir_u
                    + g_uv_uv_fixed[[u, v]];
                let val = g_a_uv * a_uv[[u, v]]
                    + g_a_u * a_uv_v[[u, v]]
                    + g_a_v * a_uv_u[[u, v]]
                    + g_a * a_uv_uv[[u, v]]
                    + g_aa_uv * a_u[u] * a_u[v]
                    + g_aa_u * (a_u_dir_v[u] * a_u[v] + a_u[u] * a_u_dir_v[v])
                    + g_aa_v * (a_u_dir_u[u] * a_u[v] + a_u[u] * a_u_dir_u[v])
                    + g_aa
                        * (a_u_uv[u] * a_u[v]
                            + a_u_dir_u[u] * a_u_dir_v[v]
                            + a_u_dir_v[u] * a_u_dir_u[v]
                            + a_u[u] * a_u_uv[v])
                    + g_au_uv[u] * a_u[v]
                    + g_au_u[u] * a_u_dir_v[v]
                    + g_au_v[u] * a_u_dir_u[v]
                    + g_au[u] * a_u_uv[v]
                    + g_au_uv[v] * a_u[u]
                    + g_au_u[v] * a_u_dir_v[u]
                    + g_au_v[v] * a_u_dir_u[u]
                    + g_au[v] * a_u_uv[u]
                    + g_uv_uv;
                eta_uv_uv[[u, v]] = val;
                eta_uv_uv[[v, u]] = val;
            }
        }

        let eta_dir_u = g_a * a_dir_u + g_dir_u;
        let eta_dir_v = g_a * a_dir_v + g_dir_v;
        let eta_u_dir_u = eta_uv.dot(dir_u);
        let eta_u_dir_v = eta_uv.dot(dir_v);
        let eta_dir_uv = g_a_v * a_dir_u + g_a_u_fixed * a_dir_v + g_a * a_dir_uv + g_dir_uv;
        let eta_u_uv = eta_uv_u.dot(dir_v);

        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let m = s_y * eta_val;
        let (k1, k2, k3, k4) = signed_probit_neglog_derivatives_up_to_fourth(m, w_i)?;
        let u1 = s_y * k1;
        let u3 = s_y * k3;

        let mut out = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let a_term = eta_u[u] * eta_u[v] * eta_dir_u;
                let a_term_v = eta_u_dir_v[u] * eta_u[v] * eta_dir_u
                    + eta_u[u] * eta_u_dir_v[v] * eta_dir_u
                    + eta_u[u] * eta_u[v] * eta_dir_uv;
                let b_term = eta_uv_u[[u, v]];
                let b_term_v = eta_uv_uv[[u, v]];
                let c_term = eta_uv[[u, v]] * eta_dir_u
                    + eta_u[u] * eta_u_dir_u[v]
                    + eta_u[v] * eta_u_dir_u[u];
                let c_term_v = eta_uv_v[[u, v]] * eta_dir_u
                    + eta_uv[[u, v]] * eta_dir_uv
                    + eta_u_dir_v[u] * eta_u_dir_u[v]
                    + eta_u[u] * eta_u_uv[v]
                    + eta_u_dir_v[v] * eta_u_dir_u[u]
                    + eta_u[v] * eta_u_uv[u];
                let val = k4 * eta_dir_v * a_term
                    + u3 * a_term_v
                    + u3 * eta_dir_v * c_term
                    + k2 * c_term_v
                    + k2 * eta_dir_v * b_term
                    + u1 * b_term_v;
                out[[u, v]] = val;
                out[[v, u]] = val;
            }
        }
        Ok(out)
    }

    fn row_primary_fourth_contracted_recompute(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let ordered = self.row_primary_fourth_contracted_recompute_ordered(
            row,
            block_states,
            cache,
            row_ctx,
            dir_u,
            dir_v,
        )?;
        if !self.effective_flex_active(block_states)? {
            return Ok(ordered);
        }

        let swapped = self.row_primary_fourth_contracted_recompute_ordered(
            row,
            block_states,
            cache,
            row_ctx,
            dir_v,
            dir_u,
        )?;
        let mut sym = ordered;
        for i in 0..sym.nrows() {
            for j in 0..sym.ncols() {
                sym[[i, j]] = 0.5 * (sym[[i, j]] + swapped[[i, j]]);
            }
        }
        Ok(sym)
    }

    /// Like `add_pullback_primary_hessian` but only accumulates the h/w
    /// cross-block contributions. The marginal-marginal, marginal-logslope,
    /// and logslope-logslope blocks are handled by the weighted-Gram operator.
    fn add_pullback_primary_hessian_hw_only(
        &self,
        target: &mut Array2<f64>,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        primary_hessian: &Array2<f64>,
    ) {
        let h = primary_hessian;
        if let (Some(primary_h), Some(block_h)) = (primary.h.as_ref(), slices.h.as_ref()) {
            for (local_idx, global_idx) in block_h.clone().enumerate() {
                let h_q = h[[0, primary_h.start + local_idx]];
                if h_q != 0.0 {
                    {
                        let mut col = target.slice_mut(s![slices.marginal.clone(), global_idx]);
                        self.marginal_design
                            .axpy_row_into(row, h_q, &mut col)
                            .expect("marginal axpy column mismatch");
                    }
                    {
                        let mut row_view =
                            target.slice_mut(s![global_idx, slices.marginal.clone()]);
                        self.marginal_design
                            .axpy_row_into(row, h_q, &mut row_view)
                            .expect("marginal axpy row mismatch");
                    }
                }

                let h_g = h[[1, primary_h.start + local_idx]];
                if h_g != 0.0 {
                    {
                        let mut col = target.slice_mut(s![slices.logslope.clone(), global_idx]);
                        self.logslope_design
                            .axpy_row_into(row, h_g, &mut col)
                            .expect("logslope axpy column mismatch");
                    }
                    {
                        let mut row_view =
                            target.slice_mut(s![global_idx, slices.logslope.clone()]);
                        self.logslope_design
                            .axpy_row_into(row, h_g, &mut row_view)
                            .expect("logslope axpy row mismatch");
                    }
                }
            }

            target
                .slice_mut(s![block_h.clone(), block_h.clone()])
                .scaled_add(
                    1.0,
                    &h.slice(s![
                        primary_h.start..primary_h.end,
                        primary_h.start..primary_h.end
                    ]),
                );
        }

        if let (Some(primary_w), Some(block_w)) = (primary.w.as_ref(), slices.w.as_ref()) {
            for (local_idx, global_idx) in block_w.clone().enumerate() {
                let w_q = h[[0, primary_w.start + local_idx]];
                if w_q != 0.0 {
                    {
                        let mut col = target.slice_mut(s![slices.marginal.clone(), global_idx]);
                        self.marginal_design
                            .axpy_row_into(row, w_q, &mut col)
                            .expect("marginal axpy column mismatch");
                    }
                    {
                        let mut row_view =
                            target.slice_mut(s![global_idx, slices.marginal.clone()]);
                        self.marginal_design
                            .axpy_row_into(row, w_q, &mut row_view)
                            .expect("marginal axpy row mismatch");
                    }
                }

                let w_g = h[[1, primary_w.start + local_idx]];
                if w_g != 0.0 {
                    {
                        let mut col = target.slice_mut(s![slices.logslope.clone(), global_idx]);
                        self.logslope_design
                            .axpy_row_into(row, w_g, &mut col)
                            .expect("logslope axpy column mismatch");
                    }
                    {
                        let mut row_view =
                            target.slice_mut(s![global_idx, slices.logslope.clone()]);
                        self.logslope_design
                            .axpy_row_into(row, w_g, &mut row_view)
                            .expect("logslope axpy row mismatch");
                    }
                }
            }

            if let (Some(primary_h), Some(block_h)) = (primary.h.as_ref(), slices.h.as_ref()) {
                target
                    .slice_mut(s![block_h.clone(), block_w.clone()])
                    .scaled_add(
                        1.0,
                        &h.slice(s![
                            primary_h.start..primary_h.end,
                            primary_w.start..primary_w.end
                        ]),
                    );
                target
                    .slice_mut(s![block_w.clone(), block_h.clone()])
                    .scaled_add(
                        1.0,
                        &h.slice(s![
                            primary_w.start..primary_w.end,
                            primary_h.start..primary_h.end
                        ]),
                    );
            }

            target
                .slice_mut(s![block_w.clone(), block_w.clone()])
                .scaled_add(
                    1.0,
                    &h.slice(s![
                        primary_w.start..primary_w.end,
                        primary_w.start..primary_w.end
                    ]),
                );
        }
    }

    fn exact_newton_joint_hessian_matvec_from_cache(
        &self,
        direction: &Array1<f64>,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Array1<f64>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();

        // ── Rigid closed-form: scalar kernel + design row ops ────────
        if !self.effective_flex_active(block_states)? {
            let out = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(
                    || Array1::<f64>::zeros(slices.total),
                    |mut chunk_out, chunk_idx| -> Result<_, String> {
                        let probit_scale = self.probit_frailty_scale();
                        let start = chunk_idx * ROW_CHUNK_SIZE;
                        let end = (start + ROW_CHUNK_SIZE).min(n);
                        for row in start..end {
                            let marginal = self.marginal_link_map(block_states[0].eta[row])?;
                            let g = block_states[1].eta[row];
                            let k = RigidProbitKernel::new(
                                marginal.q,
                                g,
                                self.z[row],
                                self.y[row],
                                self.weights[row],
                                probit_scale,
                            )?;
                            let h = rigid_transformed_hessian(marginal, &k);
                            let v_q = self
                                .marginal_design
                                .dot_row_view(row, direction.slice(s![slices.marginal.clone()]));
                            let v_g = self
                                .logslope_design
                                .dot_row_view(row, direction.slice(s![slices.logslope.clone()]));
                            let a_q = h[0][0] * v_q + h[0][1] * v_g;
                            let a_g = h[1][0] * v_q + h[1][1] * v_g;
                            {
                                let mut m = chunk_out.slice_mut(s![slices.marginal.clone()]);
                                self.marginal_design.axpy_row_into(row, a_q, &mut m)?;
                            }
                            {
                                let mut l = chunk_out.slice_mut(s![slices.logslope.clone()]);
                                self.logslope_design.axpy_row_into(row, a_g, &mut l)?;
                            }
                        }
                        Ok(chunk_out)
                    },
                )
                .try_reduce(
                    || Array1::<f64>::zeros(slices.total),
                    |mut left, right| -> Result<_, String> {
                        left += &right;
                        Ok(left)
                    },
                )?;
            return Ok(out);
        }

        let out = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || Array1::<f64>::zeros(slices.total),
                |mut chunk_out, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                    for row in start..end {
                        let row_ctx = Self::row_ctx(cache, row);
                        let row_dir =
                            self.row_primary_direction_from_flat(row, slices, primary, direction)?;
                        self.compute_row_analytic_flex_into(
                            row,
                            block_states,
                            primary,
                            row_ctx,
                            true,
                            &mut scratch,
                        )?;
                        let row_action = scratch.hess.dot(&row_dir);
                        chunk_out +=
                            &self.pullback_primary_vector(row, slices, primary, &row_action)?;
                    }
                    Ok(chunk_out)
                },
            )
            .try_reduce(
                || Array1::<f64>::zeros(slices.total),
                |mut left, right| -> Result<_, String> {
                    left += &right;
                    Ok(left)
                },
            )?;
        Ok(out)
    }

    fn exact_newton_joint_hessian_diagonal_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Array1<f64>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();

        // ── Rigid closed-form: no jets, no row contexts ──────────────
        if !self.effective_flex_active(block_states)? {
            let diagonal = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(
                    || Array1::<f64>::zeros(slices.total),
                    |mut chunk_diag, chunk_idx| -> Result<_, String> {
                        let probit_scale = self.probit_frailty_scale();
                        let start = chunk_idx * ROW_CHUNK_SIZE;
                        let end = (start + ROW_CHUNK_SIZE).min(n);
                        for row in start..end {
                            let marginal = self.marginal_link_map(block_states[0].eta[row])?;
                            let g = block_states[1].eta[row];
                            let k = RigidProbitKernel::new(
                                marginal.q,
                                g,
                                self.z[row],
                                self.y[row],
                                self.weights[row],
                                probit_scale,
                            )?;
                            let h = rigid_transformed_hessian(marginal, &k);
                            {
                                let mut m = chunk_diag.slice_mut(s![slices.marginal.clone()]);
                                self.marginal_design
                                    .squared_axpy_row_into(row, h[0][0], &mut m)?;
                            }
                            {
                                let mut l = chunk_diag.slice_mut(s![slices.logslope.clone()]);
                                self.logslope_design
                                    .squared_axpy_row_into(row, h[1][1], &mut l)?;
                            }
                        }
                        Ok(chunk_diag)
                    },
                )
                .try_reduce(
                    || Array1::<f64>::zeros(slices.total),
                    |mut left, right| -> Result<_, String> {
                        left += &right;
                        Ok(left)
                    },
                )?;
            return Ok(diagonal);
        }

        let diagonal = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || Array1::<f64>::zeros(slices.total),
                |mut chunk_diag, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                    for row in start..end {
                        let row_ctx = Self::row_ctx(cache, row);
                        self.compute_row_analytic_flex_into(
                            row,
                            block_states,
                            primary,
                            row_ctx,
                            true,
                            &mut scratch,
                        )?;

                        {
                            let mut marginal_diag =
                                chunk_diag.slice_mut(s![slices.marginal.clone()]);
                            self.marginal_design.squared_axpy_row_into(
                                row,
                                scratch.hess[[0, 0]],
                                &mut marginal_diag,
                            )?;
                        }
                        {
                            let mut logslope_diag =
                                chunk_diag.slice_mut(s![slices.logslope.clone()]);
                            self.logslope_design.squared_axpy_row_into(
                                row,
                                scratch.hess[[1, 1]],
                                &mut logslope_diag,
                            )?;
                        }

                        if let (Some(primary_h), Some(block_h)) =
                            (primary.h.as_ref(), slices.h.as_ref())
                        {
                            for (local_idx, global_idx) in block_h.clone().enumerate() {
                                chunk_diag[global_idx] += scratch.hess
                                    [[primary_h.start + local_idx, primary_h.start + local_idx]];
                            }
                        }
                        if let (Some(primary_w), Some(block_w)) =
                            (primary.w.as_ref(), slices.w.as_ref())
                        {
                            for (local_idx, global_idx) in block_w.clone().enumerate() {
                                chunk_diag[global_idx] += scratch.hess
                                    [[primary_w.start + local_idx, primary_w.start + local_idx]];
                            }
                        }
                    }
                    Ok(chunk_diag)
                },
            )
            .try_reduce(
                || Array1::<f64>::zeros(slices.total),
                |mut left, right| -> Result<_, String> {
                    left += &right;
                    Ok(left)
                },
            )?;
        Ok(diagonal)
    }

    fn exact_newton_joint_psi_terms_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let idx_primary = if block_idx == 0 { 0 } else { 1 };
        let n = self.y.len();
        let deriv = &derivative_blocks[block_idx][local_idx];
        let (p_psi, psi_label) = match block_idx {
            0 => (
                self.marginal_design.ncols(),
                "BernoulliMarginalSlopeFamily marginal",
            ),
            1 => (
                self.logslope_design.ncols(),
                "BernoulliMarginalSlopeFamily log-slope",
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi terms only support marginal/logslope blocks, got block {block_idx}"
                ));
            }
        };

        // Build the psi design map once; per-row calls use direct row_vector(row).
        let policy = crate::resource::ResourcePolicy::default_library();
        let psi_map = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv, n, p_psi, 0..n, psi_label, &policy,
        )?;

        // Block-local accumulator path: avoids O(n p^2) dense Hessian
        let (objective_psi, score_psi, block_acc) = (0..((n + ROW_CHUNK_SIZE - 1)
            / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || {
                    (
                        0.0f64,
                        Array1::<f64>::zeros(slices.total),
                        BernoulliBlockHessianAccumulator::new(slices),
                    )
                },
                |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    for row in start..end {
                        let dir = self.row_primary_psi_direction_from_map(
                            row,
                            block_idx,
                            &psi_map,
                            block_states,
                            primary,
                        )?;
                        let row_ctx = Self::row_ctx(cache, row);
                        let (_, f_pi, f_pipi) = self.compute_row_primary_gradient_hessian(
                            row,
                            block_states,
                            primary,
                            row_ctx,
                        )?;
                        let third = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &dir,
                        )?;
                        let psi_row = self.block_psi_row_from_map(
                            row, block_idx, &psi_map, slices,
                        )?;
                        acc.0 += f_pi.dot(&dir);
                        acc.1
                            .slice_mut(s![psi_row.range.clone()])
                            .scaled_add(f_pi[idx_primary], &psi_row.local_vec);
                        acc.1 += &self.pullback_primary_vector(
                            row,
                            slices,
                            primary,
                            &f_pipi.dot(&dir),
                        )?;

                        // psi_row outer pullback(f_pipi[idx_primary,:]) + transpose
                        let right_primary = f_pipi.row(idx_primary).to_owned();
                        acc.2.add_rank1_psi_cross(
                            self,
                            row,
                            slices,
                            primary,
                            block_idx,
                            &psi_row.local_vec,
                            &right_primary,
                        );
                        // third tensor pullback
                        acc.2.add_pullback(self, row, slices, primary, &third);
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || {
                    (
                        0.0f64,
                        Array1::<f64>::zeros(slices.total),
                        BernoulliBlockHessianAccumulator::new(slices),
                    )
                },
                |mut left, right| -> Result<_, String> {
                    left.0 += right.0;
                    left.1 += &right.1;
                    left.2.add(&right.2);
                    Ok(left)
                },
            )?;
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: Array2::zeros((0, 0)),
            hessian_psi_operator: Some(std::sync::Arc::new(block_acc.into_operator(slices))),
        }))
    }

    fn exact_newton_joint_psisecond_order_terms_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let Some((block_i, local_i)) = self.resolve_psi_location(derivative_blocks, psi_i) else {
            return Ok(None);
        };
        let Some((block_j, local_j)) = self.resolve_psi_location(derivative_blocks, psi_j) else {
            return Ok(None);
        };
        let idx_i = if block_i == 0 { 0 } else { 1 };
        let idx_j = if block_j == 0 { 0 } else { 1 };
        let n = self.y.len();
        let deriv_i = &derivative_blocks[block_i][local_i];
        let deriv_j = &derivative_blocks[block_j][local_j];
        let (p_psi_i, label_i) = match block_i {
            0 => (
                self.marginal_design.ncols(),
                "BernoulliMarginalSlopeFamily marginal",
            ),
            1 => (
                self.logslope_design.ncols(),
                "BernoulliMarginalSlopeFamily log-slope",
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi second-order only supports marginal/logslope blocks, got block {block_i}"
                ));
            }
        };
        let (p_psi_j, label_j) = match block_j {
            0 => (
                self.marginal_design.ncols(),
                "BernoulliMarginalSlopeFamily marginal",
            ),
            1 => (
                self.logslope_design.ncols(),
                "BernoulliMarginalSlopeFamily log-slope",
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi second-order only supports marginal/logslope blocks, got block {block_j}"
                ));
            }
        };

        // Build psi design maps once outside the row loop.
        let policy = crate::resource::ResourcePolicy::default_library();
        let psi_map_i = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv_i, n, p_psi_i, 0..n, label_i, &policy,
        )?;
        let psi_map_j = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv_j, n, p_psi_j, 0..n, label_j, &policy,
        )?;
        let psi_map_ij = if block_i == block_j {
            Some(
                crate::families::custom_family::resolve_custom_family_x_psi_psi_map(
                    deriv_i, deriv_j, local_j, n, p_psi_i, 0..n, label_i, &policy,
                )?,
            )
        } else {
            None
        };

        // Block-local accumulator path for second-order psi terms
        let (objective_psi_psi, score_psi_psi, block_acc) = (0..((n + ROW_CHUNK_SIZE - 1)
            / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || {
                    (
                        0.0f64,
                        Array1::<f64>::zeros(slices.total),
                        BernoulliBlockHessianAccumulator::new(slices),
                    )
                },
                |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    for row in start..end {
                        let dir_i = self.row_primary_psi_direction_from_map(
                            row,
                            block_i,
                            &psi_map_i,
                            block_states,
                            primary,
                        )?;
                        let dir_j = self.row_primary_psi_direction_from_map(
                            row,
                            block_j,
                            &psi_map_j,
                            block_states,
                            primary,
                        )?;
                        let dir_ij = self.row_primary_psi_second_direction_from_map(
                            row,
                            block_i,
                            block_j,
                            psi_map_ij.as_ref(),
                            block_states,
                            primary,
                        )?;
                        let row_ctx = Self::row_ctx(cache, row);
                        let (_, f_pi, f_pipi) = self.compute_row_primary_gradient_hessian(
                            row,
                            block_states,
                            primary,
                            row_ctx,
                        )?;
                        let third_i = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &dir_i,
                        )?;
                        let third_j = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &dir_j,
                        )?;
                        let fourth = self.row_primary_fourth_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &dir_i,
                            &dir_j,
                        )?;
                        let br_i = self.block_psi_row_from_map(
                            row, block_i, &psi_map_i, slices,
                        )?;
                        let br_j = self.block_psi_row_from_map(
                            row, block_j, &psi_map_j, slices,
                        )?;
                        let br_ij = self.block_psi_second_row_from_map(
                            row,
                            block_i,
                            block_j,
                            psi_map_ij.as_ref(),
                            slices,
                        )?;

                        // --- scalar and score accumulation (unchanged) ---
                        acc.0 += dir_i.dot(&f_pipi.dot(&dir_j)) + f_pi.dot(&dir_ij);
                        if let Some(ref bij) = br_ij {
                            let idx_ij = if bij.block_idx == 0 { 0 } else { 1 };
                            acc.1
                                .slice_mut(s![bij.range.clone()])
                                .scaled_add(f_pi[idx_ij], &bij.local_vec);
                        }
                        acc.1
                            .slice_mut(s![br_i.range.clone()])
                            .scaled_add(f_pipi.row(idx_i).dot(&dir_j), &br_i.local_vec);
                        acc.1
                            .slice_mut(s![br_j.range.clone()])
                            .scaled_add(f_pipi.row(idx_j).dot(&dir_i), &br_j.local_vec);
                        acc.1 += &self.pullback_primary_vector(
                            row,
                            slices,
                            primary,
                            &f_pipi.dot(&dir_ij),
                        )?;
                        acc.1 += &self.pullback_primary_vector(
                            row,
                            slices,
                            primary,
                            &third_i.dot(&dir_j),
                        )?;

                        // --- Hessian: bij outer pullback(f_pipi[idx_ij,:]) + transpose ---
                        if let Some(ref bij) = br_ij {
                            let idx_ij = if bij.block_idx == 0 { 0 } else { 1 };
                            let right_primary_ij = f_pipi.row(idx_ij).to_owned();
                            acc.2.add_rank1_psi_cross(
                                self,
                                row,
                                slices,
                                primary,
                                bij.block_idx,
                                &bij.local_vec,
                                &right_primary_ij,
                            );
                        }

                        // --- br_i outer br_j * f_pipi[[idx_i, idx_j]] + transpose ---
                        let scalar_ij = f_pipi[[idx_i, idx_j]];
                        acc.2.add_psi_psi_outer(
                            block_i,
                            &br_i.local_vec,
                            block_j,
                            &br_j.local_vec,
                            scalar_ij,
                        );

                        // --- br_i outer pullback(third_j[idx_i,:]) + transpose ---
                        let right_primary_i = third_j.row(idx_i).to_owned();
                        acc.2.add_rank1_psi_cross(
                            self,
                            row,
                            slices,
                            primary,
                            block_i,
                            &br_i.local_vec,
                            &right_primary_i,
                        );

                        // --- br_j outer pullback(third_i[idx_j,:]) + transpose ---
                        let right_primary_j = third_i.row(idx_j).to_owned();
                        acc.2.add_rank1_psi_cross(
                            self,
                            row,
                            slices,
                            primary,
                            block_j,
                            &br_j.local_vec,
                            &right_primary_j,
                        );

                        // --- fourth tensor pullback ---
                        acc.2.add_pullback(self, row, slices, primary, &fourth);

                        // --- third_ij tensor pullback ---
                        let third_ij = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &dir_ij,
                        )?;
                        acc.2.add_pullback(self, row, slices, primary, &third_ij);
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || {
                    (
                        0.0f64,
                        Array1::<f64>::zeros(slices.total),
                        BernoulliBlockHessianAccumulator::new(slices),
                    )
                },
                |mut left, right| -> Result<_, String> {
                    left.0 += right.0;
                    left.1 += &right.1;
                    left.2.add(&right.2);
                    Ok(left)
                },
            )?;
        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(Box::new(block_acc.into_operator(slices))),
        }))
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let idx_primary = if block_idx == 0 { 0 } else { 1 };
        let n = self.y.len();
        let deriv = &derivative_blocks[block_idx][local_idx];
        let (p_psi, psi_label) = match block_idx {
            0 => (
                self.marginal_design.ncols(),
                "BernoulliMarginalSlopeFamily marginal",
            ),
            1 => (
                self.logslope_design.ncols(),
                "BernoulliMarginalSlopeFamily log-slope",
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi hessian only supports marginal/logslope blocks, got block {block_idx}"
                ));
            }
        };

        // Build the psi design map once; rowwise calls use direct row_vector(row).
        let policy = crate::resource::ResourcePolicy::default_library();
        let psi_map = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv, n, p_psi, 0..n, psi_label, &policy,
        )?;

        let block_acc = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    for row in start..end {
                        let row_dir = self.row_primary_direction_from_flat(
                            row,
                            slices,
                            primary,
                            d_beta_flat,
                        )?;
                        let psi_dir = self.row_primary_psi_direction_from_map(
                            row,
                            block_idx,
                            &psi_map,
                            block_states,
                            primary,
                        )?;
                        let psi_action = self.row_primary_psi_action_on_direction_from_map(
                            row,
                            block_idx,
                            &psi_map,
                            slices,
                            d_beta_flat,
                            primary,
                        )?;
                        let row_ctx = Self::row_ctx(cache, row);
                        let third_beta = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &row_dir,
                        )?;
                        let fourth = self.row_primary_fourth_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &row_dir,
                            &psi_dir,
                        )?;
                        let psi_row = self.block_psi_row_from_map(
                            row, block_idx, &psi_map, slices,
                        )?;
                        let right_primary = third_beta.row(idx_primary).to_owned();
                        acc.add_rank1_psi_cross(
                            self,
                            row,
                            slices,
                            primary,
                            psi_row.block_idx,
                            &psi_row.local_vec,
                            &right_primary,
                        );
                        acc.add_pullback(self, row, slices, primary, &fourth);
                        let third_action = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &psi_action,
                        )?;
                        acc.add_pullback(self, row, slices, primary, &third_action);
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                },
            )?;
        Ok(Some(block_acc.to_dense(slices)))
    }

    fn exact_newton_joint_psihessian_directional_derivative_operator_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let idx_primary = if block_idx == 0 { 0 } else { 1 };
        let n = self.y.len();
        let deriv = &derivative_blocks[block_idx][local_idx];
        let (p_psi, psi_label) = match block_idx {
            0 => (
                self.marginal_design.ncols(),
                "BernoulliMarginalSlopeFamily marginal",
            ),
            1 => (
                self.logslope_design.ncols(),
                "BernoulliMarginalSlopeFamily log-slope",
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi hessian operator only supports marginal/logslope blocks, got block {block_idx}"
                ));
            }
        };

        // Build the psi design map once; rowwise calls use direct row_vector(row).
        let policy = crate::resource::ResourcePolicy::default_library();
        let psi_map = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv, n, p_psi, 0..n, psi_label, &policy,
        )?;

        let block_acc = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    for row in start..end {
                        let row_dir = self.row_primary_direction_from_flat(
                            row,
                            slices,
                            primary,
                            d_beta_flat,
                        )?;
                        let psi_dir = self.row_primary_psi_direction_from_map(
                            row,
                            block_idx,
                            &psi_map,
                            block_states,
                            primary,
                        )?;
                        let psi_action = self.row_primary_psi_action_on_direction_from_map(
                            row,
                            block_idx,
                            &psi_map,
                            slices,
                            d_beta_flat,
                            primary,
                        )?;
                        let row_ctx = Self::row_ctx(cache, row);
                        let third_beta = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &row_dir,
                        )?;
                        let fourth = self.row_primary_fourth_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &row_dir,
                            &psi_dir,
                        )?;
                        let psi_row = self.block_psi_row_from_map(
                            row, block_idx, &psi_map, slices,
                        )?;
                        let right_primary = third_beta.row(idx_primary).to_owned();
                        acc.add_rank1_psi_cross(
                            self,
                            row,
                            slices,
                            primary,
                            psi_row.block_idx,
                            &psi_row.local_vec,
                            &right_primary,
                        );
                        acc.add_pullback(self, row, slices, primary, &fourth);
                        let third_action = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &psi_action,
                        )?;
                        acc.add_pullback(self, row, slices, primary, &third_action);
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                },
            )?;
        Ok(Some(
            Arc::new(block_acc.into_operator(slices)) as Arc<dyn HyperOperator>
        ))
    }

    fn exact_newton_joint_hessian_directional_derivative_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();

        // ── Rigid closed-form: 3rd-order scalar kernel ───────────────
        if !self.effective_flex_active(block_states)? {
            let block_acc = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(
                    || BernoulliBlockHessianAccumulator::new(slices),
                    |mut acc, chunk_idx| -> Result<_, String> {
                        let probit_scale = self.probit_frailty_scale();
                        let start = chunk_idx * ROW_CHUNK_SIZE;
                        let end = (start + ROW_CHUNK_SIZE).min(n);
                        for row in start..end {
                            let marginal = self.marginal_link_map(block_states[0].eta[row])?;
                            let g = block_states[1].eta[row];
                            let k = RigidProbitKernel::new(
                                marginal.q,
                                g,
                                self.z[row],
                                self.y[row],
                                self.weights[row],
                                probit_scale,
                            )?;
                            let dq = self
                                .marginal_design
                                .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
                            let dg = self
                                .logslope_design
                                .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));
                            let t = rigid_transformed_third_contracted(marginal, &k, dq, dg);
                            let t_arr = Array2::from_shape_fn((2, 2), |(a, b)| t[a][b]);
                            acc.add_pullback(self, row, slices, primary, &t_arr);
                        }
                        Ok(acc)
                    },
                )
                .try_reduce(
                    || BernoulliBlockHessianAccumulator::new(slices),
                    |mut left, right| {
                        left.add(&right);
                        Ok(left)
                    },
                )?;
            return Ok(Some(block_acc.to_dense(slices)));
        }

        let block_acc = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    for row in start..end {
                        let row_dir = self.row_primary_direction_from_flat(
                            row,
                            slices,
                            primary,
                            d_beta_flat,
                        )?;
                        let row_ctx = Self::row_ctx(cache, row);
                        let third = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &row_dir,
                        )?;
                        acc.add_pullback(self, row, slices, primary, &third);
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                },
            )?;
        Ok(Some(block_acc.to_dense(slices)))
    }

    fn exact_newton_joint_hessian_directional_derivative_operator_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();

        if !self.effective_flex_active(block_states)? {
            let block_acc = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(
                    || BernoulliBlockHessianAccumulator::new(slices),
                    |mut acc, chunk_idx| -> Result<_, String> {
                        let probit_scale = self.probit_frailty_scale();
                        let start = chunk_idx * ROW_CHUNK_SIZE;
                        let end = (start + ROW_CHUNK_SIZE).min(n);
                        for row in start..end {
                            let marginal = self.marginal_link_map(block_states[0].eta[row])?;
                            let g = block_states[1].eta[row];
                            let k = RigidProbitKernel::new(
                                marginal.q,
                                g,
                                self.z[row],
                                self.y[row],
                                self.weights[row],
                                probit_scale,
                            )?;
                            let dq = self
                                .marginal_design
                                .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
                            let dg = self
                                .logslope_design
                                .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));
                            let t = rigid_transformed_third_contracted(marginal, &k, dq, dg);
                            let t_arr = Array2::from_shape_fn((2, 2), |(a, b)| t[a][b]);
                            acc.add_pullback(self, row, slices, primary, &t_arr);
                        }
                        Ok(acc)
                    },
                )
                .try_reduce(
                    || BernoulliBlockHessianAccumulator::new(slices),
                    |mut left, right| -> Result<_, String> {
                        left.add(&right);
                        Ok(left)
                    },
                )?;
            return Ok(Some(
                Arc::new(block_acc.into_operator(slices)) as Arc<dyn HyperOperator>
            ));
        }

        let block_acc = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    for row in start..end {
                        let row_dir = self.row_primary_direction_from_flat(
                            row,
                            slices,
                            primary,
                            d_beta_flat,
                        )?;
                        let row_ctx = Self::row_ctx(cache, row);
                        let third = self.row_primary_third_contracted_recompute(
                            row,
                            block_states,
                            cache,
                            row_ctx,
                            &row_dir,
                        )?;
                        acc.add_pullback(self, row, slices, primary, &third);
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                },
            )?;
        Ok(Some(
            Arc::new(block_acc.into_operator(slices)) as Arc<dyn HyperOperator>
        ))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let make_acc = || BernoulliBlockHessianAccumulator::new(slices);

        // ── Rigid closed-form: 4th-order scalar kernel ───────────────
        if !self.effective_flex_active(block_states)? {
            let block_acc = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(make_acc, |mut acc, chunk_idx| -> Result<_, String> {
                    let probit_scale = self.probit_frailty_scale();
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    for row in start..end {
                        let marginal = self.marginal_link_map(block_states[0].eta[row])?;
                        let g = block_states[1].eta[row];
                        let k = RigidProbitKernel::new(
                            marginal.q,
                            g,
                            self.z[row],
                            self.y[row],
                            self.weights[row],
                            probit_scale,
                        )?;
                        let uq = self
                            .marginal_design
                            .dot_row_view(row, d_beta_u_flat.slice(s![slices.marginal.clone()]));
                        let ug = self
                            .logslope_design
                            .dot_row_view(row, d_beta_u_flat.slice(s![slices.logslope.clone()]));
                        let vq = self
                            .marginal_design
                            .dot_row_view(row, d_beta_v_flat.slice(s![slices.marginal.clone()]));
                        let vg = self
                            .logslope_design
                            .dot_row_view(row, d_beta_v_flat.slice(s![slices.logslope.clone()]));
                        let f = rigid_transformed_fourth_contracted(marginal, &k, uq, ug, vq, vg);
                        let f_arr = Array2::from_shape_fn((2, 2), |(a, b)| f[a][b]);
                        acc.add_pullback(self, row, slices, primary, &f_arr);
                    }
                    Ok(acc)
                })
                .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                })?;
            return Ok(Some(block_acc.to_dense(slices)));
        }

        let block_acc = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(make_acc, |mut acc, chunk_idx| -> Result<_, String> {
                let start = chunk_idx * ROW_CHUNK_SIZE;
                let end = (start + ROW_CHUNK_SIZE).min(n);
                for row in start..end {
                    let row_u =
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_u_flat)?;
                    let row_v =
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_v_flat)?;
                    let row_ctx = Self::row_ctx(cache, row);
                    let fourth = self.row_primary_fourth_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &row_u,
                        &row_v,
                    )?;
                    acc.add_pullback(self, row, slices, primary, &fourth);
                }
                Ok(acc)
            })
            .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                left.add(&right);
                Ok(left)
            })?;
        Ok(Some(block_acc.to_dense(slices)))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative_operator_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let make_acc = || BernoulliBlockHessianAccumulator::new(slices);

        if !self.effective_flex_active(block_states)? {
            let block_acc = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(make_acc, |mut acc, chunk_idx| -> Result<_, String> {
                    let probit_scale = self.probit_frailty_scale();
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    for row in start..end {
                        let marginal = self.marginal_link_map(block_states[0].eta[row])?;
                        let g = block_states[1].eta[row];
                        let k = RigidProbitKernel::new(
                            marginal.q,
                            g,
                            self.z[row],
                            self.y[row],
                            self.weights[row],
                            probit_scale,
                        )?;
                        let uq = self
                            .marginal_design
                            .dot_row_view(row, d_beta_u_flat.slice(s![slices.marginal.clone()]));
                        let ug = self
                            .logslope_design
                            .dot_row_view(row, d_beta_u_flat.slice(s![slices.logslope.clone()]));
                        let vq = self
                            .marginal_design
                            .dot_row_view(row, d_beta_v_flat.slice(s![slices.marginal.clone()]));
                        let vg = self
                            .logslope_design
                            .dot_row_view(row, d_beta_v_flat.slice(s![slices.logslope.clone()]));
                        let f = rigid_transformed_fourth_contracted(marginal, &k, uq, ug, vq, vg);
                        let f_arr = Array2::from_shape_fn((2, 2), |(a, b)| f[a][b]);
                        acc.add_pullback(self, row, slices, primary, &f_arr);
                    }
                    Ok(acc)
                })
                .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                })?;
            return Ok(Some(
                Arc::new(block_acc.into_operator(slices)) as Arc<dyn HyperOperator>
            ));
        }

        let block_acc = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(make_acc, |mut acc, chunk_idx| -> Result<_, String> {
                let start = chunk_idx * ROW_CHUNK_SIZE;
                let end = (start + ROW_CHUNK_SIZE).min(n);
                for row in start..end {
                    let row_u =
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_u_flat)?;
                    let row_v =
                        self.row_primary_direction_from_flat(row, slices, primary, d_beta_v_flat)?;
                    let row_ctx = Self::row_ctx(cache, row);
                    let fourth = self.row_primary_fourth_contracted_recompute(
                        row,
                        block_states,
                        cache,
                        row_ctx,
                        &row_u,
                        &row_v,
                    )?;
                    acc.add_pullback(self, row, slices, primary, &fourth);
                }
                Ok(acc)
            })
            .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                left.add(&right);
                Ok(left)
            })?;
        Ok(Some(
            Arc::new(block_acc.into_operator(slices)) as Arc<dyn HyperOperator>
        ))
    }

    fn evaluate_blockwise_exact_newton(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        let slices = block_slices(self);
        let flex_active = self.effective_flex_active(block_states)?;

        // ── Joint-then-slice path (rigid, p < 512) ──────────────────────
        //
        // Block Hessians are principal blocks of the joint Hessian—never
        // independently derived. The RowKernel<2> is the single source of
        // truth in objective space (negative log-likelihood), and the family
        // score/observed-information convention is applied once here.
        if !flex_active && slices.total < 512 {
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let cache = build_row_kernel_cache(&kern)?;
            let ll = row_kernel_log_likelihood(&cache);
            let joint_gradient = Self::exact_newton_score_from_objective_gradient(
                row_kernel_gradient(&kern, &cache),
            );
            let joint_hessian = Self::exact_newton_observed_information_from_objective_hessian(
                row_kernel_hessian_dense(&kern, &cache),
            );
            let mut block_gradients = vec![
                joint_gradient.slice(s![slices.marginal.clone()]).to_owned(),
                joint_gradient.slice(s![slices.logslope.clone()]).to_owned(),
            ];
            let mut block_ranges = vec![slices.marginal.clone(), slices.logslope.clone()];
            if let Some(range) = slices.h.clone() {
                block_gradients.push(joint_gradient.slice(s![range.clone()]).to_owned());
                block_ranges.push(range);
            }
            if let Some(range) = slices.w.clone() {
                block_gradients.push(joint_gradient.slice(s![range.clone()]).to_owned());
                block_ranges.push(range);
            }
            return Ok(FamilyEvaluation {
                log_likelihood: ll,
                blockworking_sets: slice_joint_into_block_working_sets(
                    block_gradients,
                    &joint_hessian,
                    &block_ranges,
                ),
            });
        }

        // ── Joint-then-slice path (flex, p < 512) ───────────────────────
        //
        // The flex path has variable-dimension primary space (q, g, h?, w?).
        // The joint Hessian is built column-by-column via matvec.
        // Gradients are extracted from the per-row analytic flex loop.
        if flex_active && slices.total < 512 {
            let cache = self.build_exact_eval_cache_with_order(block_states)?;
            let primary = cache.primary.clone();
            let n = self.y.len();

            // Gradient-only accumulation (no Hessian).
            let mut ll = 0.0;
            let mut grad_marginal = Array1::<f64>::zeros(slices.marginal.len());
            let mut grad_logslope = Array1::<f64>::zeros(slices.logslope.len());
            let mut grad_h: Option<Array1<f64>> = slices.h.as_ref().map(|r| Array1::zeros(r.len()));
            let mut grad_w: Option<Array1<f64>> = slices.w.as_ref().map(|r| Array1::zeros(r.len()));
            let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
            for row in 0..n {
                let row_ctx = Self::row_ctx(&cache, row);
                let row_neglog = self.compute_row_analytic_flex_into(
                    row,
                    block_states,
                    &primary,
                    row_ctx,
                    true,
                    &mut scratch,
                )?;
                ll -= row_neglog;
                {
                    let mut m = grad_marginal.view_mut();
                    self.marginal_design.axpy_row_into(
                        row,
                        Self::exact_newton_score_component_from_objective_gradient(scratch.grad[0]),
                        &mut m,
                    )?;
                }
                {
                    let mut g = grad_logslope.view_mut();
                    self.logslope_design.axpy_row_into(
                        row,
                        Self::exact_newton_score_component_from_objective_gradient(scratch.grad[1]),
                        &mut g,
                    )?;
                }
                if let (Some(ph), Some(gh)) = (primary.h.as_ref(), grad_h.as_mut()) {
                    for idx in 0..ph.len() {
                        gh[idx] += Self::exact_newton_score_component_from_objective_gradient(
                            scratch.grad[ph.start + idx],
                        );
                    }
                }
                if let (Some(pw), Some(gw)) = (primary.w.as_ref(), grad_w.as_mut()) {
                    for idx in 0..pw.len() {
                        gw[idx] += Self::exact_newton_score_component_from_objective_gradient(
                            scratch.grad[pw.start + idx],
                        );
                    }
                }
            }

            let joint = self
                .exact_newton_joint_hessian(block_states)?
                .ok_or("BernoulliMarginalSlopeFamily flex: joint hessian unavailable")?;
            let mut block_gradients = vec![grad_marginal, grad_logslope];
            let mut block_ranges = vec![slices.marginal.clone(), slices.logslope.clone()];
            if let (Some(range), Some(gradient)) = (slices.h.clone(), grad_h) {
                block_ranges.push(range);
                block_gradients.push(gradient);
            }
            if let (Some(range), Some(gradient)) = (slices.w.clone(), grad_w) {
                block_ranges.push(range);
                block_gradients.push(gradient);
            }
            return Ok(FamilyEvaluation {
                log_likelihood: ll,
                blockworking_sets: slice_joint_into_block_working_sets(
                    block_gradients,
                    &joint,
                    &block_ranges,
                ),
            });
        }

        // ── Blockwise fallback (p >= 512) ───────────────────────────────
        //
        // The joint dense Hessian is too large to materialise.  Block
        // Hessians are assembled independently via the same per-row
        // kernel, so the algebra is correct but not structurally guaranteed
        // identical to the joint object.  This path should only be reached
        // for very large models where memory is the binding constraint.
        let cache = self.build_exact_eval_cache_with_order(block_states)?;
        let primary = cache.primary.clone();

        if !flex_active {
            let mut ll = 0.0;
            let mut grad_marginal = Array1::<f64>::zeros(slices.marginal.len());
            let mut grad_logslope = Array1::<f64>::zeros(slices.logslope.len());
            let mut hess_marginal =
                Array2::<f64>::zeros((slices.marginal.len(), slices.marginal.len()));
            let mut hess_logslope =
                Array2::<f64>::zeros((slices.logslope.len(), slices.logslope.len()));
            let probit_scale = self.probit_frailty_scale();
            for row in 0..self.y.len() {
                let marginal = self.marginal_link_map(block_states[0].eta[row])?;
                let g = block_states[1].eta[row];
                let kern = RigidProbitKernel::new(
                    marginal.q,
                    g,
                    self.z[row],
                    self.y[row],
                    self.weights[row],
                    probit_scale,
                )?;
                ll += self.weights[row] * kern.logcdf;
                let grad = rigid_transformed_gradient(marginal, &kern);
                let h = rigid_transformed_hessian(marginal, &kern);

                {
                    let mut marginal = grad_marginal.view_mut();
                    self.marginal_design.axpy_row_into(
                        row,
                        Self::exact_newton_score_component_from_objective_gradient(grad[0]),
                        &mut marginal,
                    )?;
                }
                {
                    let mut logslope = grad_logslope.view_mut();
                    self.logslope_design.axpy_row_into(
                        row,
                        Self::exact_newton_score_component_from_objective_gradient(grad[1]),
                        &mut logslope,
                    )?;
                }
                self.marginal_design
                    .syr_row_into(row, h[0][0], &mut hess_marginal)?;
                self.logslope_design
                    .syr_row_into(row, h[1][1], &mut hess_logslope)?;
            }

            return Ok(FamilyEvaluation {
                log_likelihood: ll,
                blockworking_sets: {
                    let mut sets = vec![
                        BlockWorkingSet::ExactNewton {
                            gradient: grad_marginal,
                            hessian: SymmetricMatrix::Dense(hess_marginal),
                        },
                        BlockWorkingSet::ExactNewton {
                            gradient: grad_logslope,
                            hessian: SymmetricMatrix::Dense(hess_logslope),
                        },
                    ];
                    if let Some(range) = slices.h.as_ref() {
                        sets.push(BlockWorkingSet::ExactNewton {
                            gradient: Array1::zeros(range.len()),
                            hessian: SymmetricMatrix::Dense(Array2::zeros((
                                range.len(),
                                range.len(),
                            ))),
                        });
                    }
                    if let Some(range) = slices.w.as_ref() {
                        sets.push(BlockWorkingSet::ExactNewton {
                            gradient: Array1::zeros(range.len()),
                            hessian: SymmetricMatrix::Dense(Array2::zeros((
                                range.len(),
                                range.len(),
                            ))),
                        });
                    }
                    sets
                },
            });
        }
        let n = self.y.len();
        let make_acc = || BernoulliExactNewtonAccumulator::new(&slices);
        let reduced = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(make_acc, |mut acc, chunk_idx| -> Result<_, String> {
                let start = chunk_idx * ROW_CHUNK_SIZE;
                let end = (start + ROW_CHUNK_SIZE).min(n);
                let rows = end - start;
                let marginal_chunk = self.marginal_design.row_chunk(start..end);
                let logslope_chunk = self.logslope_design.row_chunk(start..end);
                let mut grad_marginal_weights = Array1::<f64>::zeros(rows);
                let mut grad_logslope_weights = Array1::<f64>::zeros(rows);
                let mut hess_marginal_weights = Array1::<f64>::zeros(rows);
                let mut hess_logslope_weights = Array1::<f64>::zeros(rows);
                let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);

                for local_row in 0..rows {
                    let row = start + local_row;
                    let row_ctx = Self::row_ctx(&cache, row);
                    let row_neglog = self.compute_row_analytic_flex_into(
                        row,
                        block_states,
                        &primary,
                        row_ctx,
                        true,
                        &mut scratch,
                    )?;
                    acc.ll -= row_neglog;
                    grad_marginal_weights[local_row] =
                        Self::exact_newton_score_component_from_objective_gradient(scratch.grad[0]);
                    grad_logslope_weights[local_row] =
                        Self::exact_newton_score_component_from_objective_gradient(scratch.grad[1]);
                    hess_marginal_weights[local_row] = scratch.hess[[0, 0]];
                    hess_logslope_weights[local_row] = scratch.hess[[1, 1]];

                    if let (Some(primary_h), Some(grad_h), Some(hess_h)) =
                        (primary.h.as_ref(), acc.grad_h.as_mut(), acc.hess_h.as_mut())
                    {
                        for idx in 0..primary_h.len() {
                            grad_h[idx] +=
                                Self::exact_newton_score_component_from_objective_gradient(
                                    scratch.grad[primary_h.start + idx],
                                );
                        }
                        for row_h in 0..primary_h.len() {
                            for col_h in 0..primary_h.len() {
                                hess_h[[row_h, col_h]] += scratch.hess
                                    [[primary_h.start + row_h, primary_h.start + col_h]];
                            }
                        }
                    }
                    if let (Some(primary_w), Some(grad_w), Some(hess_w)) =
                        (primary.w.as_ref(), acc.grad_w.as_mut(), acc.hess_w.as_mut())
                    {
                        for idx in 0..primary_w.len() {
                            grad_w[idx] +=
                                Self::exact_newton_score_component_from_objective_gradient(
                                    scratch.grad[primary_w.start + idx],
                                );
                        }
                        for row_w in 0..primary_w.len() {
                            for col_w in 0..primary_w.len() {
                                hess_w[[row_w, col_w]] += scratch.hess
                                    [[primary_w.start + row_w, primary_w.start + col_w]];
                            }
                        }
                    }
                }

                add_weighted_chunk_gradient(
                    &marginal_chunk,
                    &grad_marginal_weights,
                    &mut acc.grad_marginal,
                );
                add_weighted_chunk_gradient(
                    &logslope_chunk,
                    &grad_logslope_weights,
                    &mut acc.grad_logslope,
                );
                add_weighted_chunk_gram(
                    &marginal_chunk,
                    &hess_marginal_weights,
                    &mut acc.hess_marginal,
                );
                add_weighted_chunk_gram(
                    &logslope_chunk,
                    &hess_logslope_weights,
                    &mut acc.hess_logslope,
                );
                Ok(acc)
            })
            .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                left.add(&right);
                Ok(left)
            })?;

        let BernoulliExactNewtonAccumulator {
            ll,
            grad_marginal,
            grad_logslope,
            hess_marginal,
            hess_logslope,
            grad_h,
            grad_w,
            hess_h,
            hess_w,
        } = reduced;
        let mut blockworking_sets = vec![
            BlockWorkingSet::ExactNewton {
                gradient: grad_marginal,
                hessian: SymmetricMatrix::Dense(hess_marginal),
            },
            BlockWorkingSet::ExactNewton {
                gradient: grad_logslope,
                hessian: SymmetricMatrix::Dense(hess_logslope),
            },
        ];
        if let (Some(gradient), Some(hessian)) = (grad_h, hess_h) {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            });
        }
        if let (Some(gradient), Some(hessian)) = (grad_w, hess_w) {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            });
        }
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets,
        })
    }
}

impl CustomFamily for BernoulliMarginalSlopeFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn exact_outer_derivative_order(
        &self,
        specs: &[ParameterBlockSpec],
        _: &BlockwiseFitOptions,
    ) -> ExactOuterDerivativeOrder {
        cost_gated_outer_order(specs)
    }

    fn exact_newton_joint_psi_workspace_for_first_order_terms(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        self.validate_exact_monotonicity(block_states)?;
        self.evaluate_blockwise_exact_newton(block_states)
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        self.validate_exact_monotonicity(block_states)?;
        let flex_active = self.effective_flex_active(block_states)?;
        if !flex_active {
            // Rigid probit: vectorized closed-form.
            // η_i = q_i·c_i + s_f b_i·z_i  where c_i = √(1+(s_f b_i)²)
            // ll = Σ w_i · log Φ((2y_i−1)·η_i)
            let b = &block_states[1].eta;
            let probit_scale = self.probit_frailty_scale();
            let n = self.y.len();
            return (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(
                    || 0.0,
                    |mut ll, chunk_idx| -> Result<_, String> {
                        let start = chunk_idx * ROW_CHUNK_SIZE;
                        let end = (start + ROW_CHUNK_SIZE).min(n);
                        for i in start..end {
                            let q_internal = self.marginal_link_map(block_states[0].eta[i])?.q;
                            let eta_i =
                                rigid_observed_eta(q_internal, b[i], self.z[i], probit_scale);
                            let signed = (2.0 * self.y[i] - 1.0) * eta_i;
                            let (log_cdf, _) = signed_probit_logcdf_and_mills_ratio(signed);
                            ll += self.weights[i] * log_cdf;
                        }
                        Ok(ll)
                    },
                )
                .try_reduce(
                    || 0.0,
                    |left, right| -> Result<_, String> { Ok(left + right) },
                );
        }
        let beta_h = self.score_beta(block_states)?;
        let beta_w = self.link_beta(block_states)?;
        let n = self.y.len();
        (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(
                || 0.0,
                |mut ll, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    for row in start..end {
                        let intercept = self
                            .solve_row_intercept_base(
                                block_states[0].eta[row],
                                block_states[1].eta[row],
                                beta_h,
                                beta_w,
                            )?
                            .0;
                        let slope = block_states[1].eta[row];
                        let obs = self.observed_denested_cell_partials(
                            row, intercept, slope, beta_h, beta_w,
                        )?;
                        let s_i = eval_coeff4_at(&obs.coeff, self.z[row]);
                        let signed = (2.0 * self.y[row] - 1.0) * s_i;
                        let (log_cdf, _) = signed_probit_logcdf_and_mills_ratio(signed);
                        ll += self.weights[row] * log_cdf;
                    }
                    Ok(ll)
                },
            )
            .try_reduce(
                || 0.0,
                |left, right| -> Result<_, String> { Ok(left + right) },
            )
    }

    fn max_feasible_step_size(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        self.validate_exact_block_state_shapes(block_states)?;
        let beta = block_states.get(block_idx).ok_or_else(|| {
            format!(
                "bernoulli marginal-slope block index {block_idx} is out of bounds for {} states",
                block_states.len()
            )
        })?;
        if delta.len() != beta.beta.len() {
            return Err(format!(
                "bernoulli marginal-slope step length mismatch for block {block_idx}: delta={}, beta={}",
                delta.len(),
                beta.beta.len()
            ));
        }
        Ok(None)
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = block_slices(self);
        if slices.total >= 512 {
            return Ok(None);
        }
        if !self.effective_flex_active(block_states)? {
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let cache = build_row_kernel_cache(&kern)?;
            return Ok(Some(crate::families::row_kernel::row_kernel_hessian_dense(
                &kern, &cache,
            )));
        }

        let cache = self.build_exact_eval_cache_with_order(block_states)?;
        let mut dense = Array2::<f64>::zeros((slices.total, slices.total));
        let mut basis = Array1::<f64>::zeros(slices.total);
        for col in 0..slices.total {
            basis[col] = 1.0;
            let applied =
                self.exact_newton_joint_hessian_matvec_from_cache(&basis, block_states, &cache)?;
            basis[col] = 0.0;
            dense.column_mut(col).assign(&applied);
        }
        Ok(Some(dense))
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        self.validate_exact_monotonicity(block_states)?;
        let slices = block_slices(self);
        if !self.effective_flex_active(block_states)? {
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let cache = build_row_kernel_cache(&kern)?;
            return Ok(Some(ExactNewtonJointGradientEvaluation {
                log_likelihood: row_kernel_log_likelihood(&cache),
                gradient: Self::exact_newton_score_from_objective_gradient(row_kernel_gradient(
                    &kern, &cache,
                )),
            }));
        }

        let cache = self.build_exact_eval_cache_with_order(block_states)?;
        let primary = &cache.primary;
        let n = self.y.len();
        let make_acc = || {
            (
                0.0_f64,
                Array1::<f64>::zeros(slices.marginal.len()),
                Array1::<f64>::zeros(slices.logslope.len()),
                slices
                    .h
                    .as_ref()
                    .map(|range| Array1::<f64>::zeros(range.len())),
                slices
                    .w
                    .as_ref()
                    .map(|range| Array1::<f64>::zeros(range.len())),
            )
        };
        let (log_likelihood, grad_marginal, grad_logslope, grad_h, grad_w) = (0..((n
            + ROW_CHUNK_SIZE
            - 1)
            / ROW_CHUNK_SIZE))
            .into_par_iter()
            .try_fold(make_acc, |mut acc, chunk_idx| -> Result<_, String> {
                let start = chunk_idx * ROW_CHUNK_SIZE;
                let end = (start + ROW_CHUNK_SIZE).min(n);
                let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                for row in start..end {
                    let row_ctx = Self::row_ctx(&cache, row);
                    let neglog = self.compute_row_analytic_flex_into(
                        row,
                        block_states,
                        primary,
                        row_ctx,
                        false,
                        &mut scratch,
                    )?;
                    acc.0 -= neglog;
                    {
                        let mut marginal = acc.1.view_mut();
                        self.marginal_design.axpy_row_into(
                            row,
                            Self::exact_newton_score_component_from_objective_gradient(
                                scratch.grad[0],
                            ),
                            &mut marginal,
                        )?;
                    }
                    {
                        let mut logslope = acc.2.view_mut();
                        self.logslope_design.axpy_row_into(
                            row,
                            Self::exact_newton_score_component_from_objective_gradient(
                                scratch.grad[1],
                            ),
                            &mut logslope,
                        )?;
                    }
                    if let (Some(primary_h), Some(grad_h)) = (primary.h.as_ref(), acc.3.as_mut()) {
                        for idx in 0..primary_h.len() {
                            grad_h[idx] +=
                                Self::exact_newton_score_component_from_objective_gradient(
                                    scratch.grad[primary_h.start + idx],
                                );
                        }
                    }
                    if let (Some(primary_w), Some(grad_w)) = (primary.w.as_ref(), acc.4.as_mut()) {
                        for idx in 0..primary_w.len() {
                            grad_w[idx] +=
                                Self::exact_newton_score_component_from_objective_gradient(
                                    scratch.grad[primary_w.start + idx],
                                );
                        }
                    }
                }
                Ok(acc)
            })
            .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                left.0 += right.0;
                left.1 += &right.1;
                left.2 += &right.2;
                if let (Some(lhs), Some(rhs)) = (left.3.as_mut(), right.3.as_ref()) {
                    *lhs += rhs;
                }
                if let (Some(lhs), Some(rhs)) = (left.4.as_mut(), right.4.as_ref()) {
                    *lhs += rhs;
                }
                Ok(left)
            })?;

        let mut gradient = Array1::<f64>::zeros(slices.total);
        gradient
            .slice_mut(s![slices.marginal.clone()])
            .assign(&grad_marginal);
        gradient
            .slice_mut(s![slices.logslope.clone()])
            .assign(&grad_logslope);
        if let (Some(range), Some(grad_h)) = (slices.h.as_ref(), grad_h.as_ref()) {
            gradient.slice_mut(s![range.clone()]).assign(grad_h);
        }
        if let (Some(range), Some(grad_w)) = (slices.w.as_ref(), grad_w.as_ref()) {
            gradient.slice_mut(s![range.clone()]).assign(grad_w);
        }
        Ok(Some(ExactNewtonJointGradientEvaluation {
            log_likelihood,
            gradient,
        }))
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        if !self.effective_flex_active(block_states)? {
            // Rigid path: use generic RowKernel<2> operator
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            Ok(Some(Arc::new(RowKernelHessianWorkspace::new(kern)?)))
        } else {
            // Flex path: keep existing workspace for variable-K primary space
            Ok(Some(Arc::new(
                BernoulliMarginalSlopeExactNewtonJointHessianWorkspace::new(
                    self.clone(),
                    block_states.to_vec(),
                )?,
            )))
        }
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if !self.effective_flex_active(block_states)? {
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let sl = d_beta_flat.as_slice().ok_or("non-contiguous d_beta")?;
            crate::families::row_kernel::row_kernel_directional_derivative(&kern, sl).map(Some)
        } else {
            let cache = self.build_exact_eval_cache(block_states)?;
            self.exact_newton_joint_hessian_directional_derivative_from_cache(
                block_states,
                d_beta_flat,
                &cache,
            )
        }
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if !self.effective_flex_active(block_states)? {
            let kern = BernoulliRigidRowKernel::new(self.clone(), block_states.to_vec());
            let su = d_beta_u_flat.as_slice().ok_or("non-contiguous d_beta_u")?;
            let sv = d_beta_v_flat.as_slice().ok_or("non-contiguous d_beta_v")?;
            crate::families::row_kernel::row_kernel_second_directional_derivative(&kern, su, sv)
                .map(Some)
        } else {
            let cache = self.build_exact_eval_cache(block_states)?;
            self.exact_newton_joint_hessiansecond_directional_derivative_from_cache(
                block_states,
                d_beta_u_flat,
                d_beta_v_flat,
                &cache,
            )
        }
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if self.is_sigma_aux_index(derivative_blocks, psi_index) {
            return self.sigma_exact_joint_psi_terms(block_states, specs);
        }
        let cache = self.build_exact_eval_cache(block_states)?;
        self.exact_newton_joint_psi_terms_from_cache(
            block_states,
            derivative_blocks,
            psi_index,
            &cache,
        )
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        if self.is_sigma_aux_index(derivative_blocks, psi_i)
            || self.is_sigma_aux_index(derivative_blocks, psi_j)
        {
            if self.is_sigma_aux_index(derivative_blocks, psi_i)
                && self.is_sigma_aux_index(derivative_blocks, psi_j)
            {
                return self.sigma_exact_joint_psisecond_order_terms(block_states);
            }
            return Err(
                "bernoulli marginal-slope mixed log-sigma/spatial psi second derivatives require cross auxiliary terms; only pure log-sigma second derivatives are supported"
                    .to_string(),
            );
        }
        let cache = self.build_exact_eval_cache(block_states)?;
        self.exact_newton_joint_psisecond_order_terms_from_cache(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            &cache,
        )
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if self.is_sigma_aux_index(derivative_blocks, psi_index) {
            return self
                .sigma_exact_joint_psihessian_directional_derivative(block_states, d_beta_flat);
        }
        let cache = self.build_exact_eval_cache(block_states)?;
        self.exact_newton_joint_psihessian_directional_derivative_from_cache(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &cache,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        Ok(Some(Arc::new(
            BernoulliMarginalSlopeExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                specs.to_vec(),
                derivative_blocks.to_vec(),
            )?,
        )))
    }

    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_states.len() == usize::MAX
            || block_idx == usize::MAX
            || spec.design.ncols() == usize::MAX
        {
            return Err("unreachable bernoulli marginal-slope constraint state".to_string());
        }
        if self.score_block_index().is_some_and(|idx| block_idx == idx) {
            return Ok(self
                .score_warp
                .as_ref()
                .map(DeviationRuntime::structural_monotonicity_constraints));
        }
        if self.link_block_index().is_some_and(|idx| block_idx == idx) {
            return Ok(self
                .link_dev
                .as_ref()
                .map(DeviationRuntime::structural_monotonicity_constraints));
        }
        Ok(None)
    }

    fn post_update_block_beta(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        self.validate_exact_block_state_shapes(block_states)?;
        if block_idx >= block_states.len() {
            return Err(format!(
                "post-update block index {} out of range for {} blocks",
                block_idx,
                block_states.len()
            ));
        }
        if self.score_block_index().is_some_and(|idx| block_idx == idx) {
            if let (Some(runtime), Some(score)) =
                (&self.score_warp, self.score_block_state(block_states)?)
            {
                let current = &score.beta;
                if current.len() != beta.len() {
                    return Err(format!(
                        "score-warp post-update beta length mismatch: current={}, proposed={}",
                        current.len(),
                        beta.len()
                    ));
                }
                return project_monotone_feasible_beta(runtime, current, &beta, "score_warp_dev");
            }
        }
        if self.link_block_index().is_some_and(|idx| block_idx == idx) {
            if let (Some(runtime), Some(link)) =
                (&self.link_dev, self.link_block_state(block_states)?)
            {
                let current = &link.beta;
                if current.len() != beta.len() {
                    return Err(format!(
                        "link-deviation post-update beta length mismatch: current={}, proposed={}",
                        current.len(),
                        beta.len()
                    ));
                }
                return project_monotone_feasible_beta(runtime, current, &beta, "link_dev");
            }
        }
        Ok(beta)
    }
}

impl BernoulliMarginalSlopeExactNewtonJointHessianWorkspace {
    fn new(
        family: BernoulliMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
    ) -> Result<Self, String> {
        let cache = family.build_exact_eval_cache(&block_states)?;
        Ok(Self {
            family,
            block_states,
            cache,
        })
    }
}

impl ExactNewtonJointHessianWorkspace for BernoulliMarginalSlopeExactNewtonJointHessianWorkspace {
    fn hessian_matvec(&self, beta_flat: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_matvec_from_cache(
                beta_flat,
                &self.block_states,
                &self.cache,
            )
            .map(Some)
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_diagonal_from_cache(&self.block_states, &self.cache)
            .map(Some)
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_operator_from_cache(
                &self.block_states,
                d_beta_flat,
                &self.cache,
            )
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_from_cache(
                &self.block_states,
                d_beta_flat,
                &self.cache,
            )
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative_operator_from_cache(
                &self.block_states,
                d_beta_u_flat,
                d_beta_v_flat,
                &self.cache,
            )
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative_from_cache(
                &self.block_states,
                d_beta_u_flat,
                d_beta_v_flat,
                &self.cache,
            )
    }
}

impl BernoulliMarginalSlopeExactNewtonJointPsiWorkspace {
    fn new(
        family: BernoulliMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
        specs: Vec<ParameterBlockSpec>,
        derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
    ) -> Result<Self, String> {
        let cache = family.build_exact_eval_cache(&block_states)?;
        Ok(Self {
            family,
            block_states,
            specs,
            derivative_blocks,
            cache,
        })
    }
}

impl ExactNewtonJointPsiWorkspace for BernoulliMarginalSlopeExactNewtonJointPsiWorkspace {
    fn first_order_terms(
        &self,
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if self
            .family
            .is_sigma_aux_index(&self.derivative_blocks, psi_index)
        {
            return self
                .family
                .sigma_exact_joint_psi_terms(&self.block_states, &self.specs);
        }
        self.family.exact_newton_joint_psi_terms_from_cache(
            &self.block_states,
            &self.derivative_blocks,
            psi_index,
            &self.cache,
        )
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        if self
            .family
            .is_sigma_aux_index(&self.derivative_blocks, psi_i)
            || self
                .family
                .is_sigma_aux_index(&self.derivative_blocks, psi_j)
        {
            if self
                .family
                .is_sigma_aux_index(&self.derivative_blocks, psi_i)
                && self
                    .family
                    .is_sigma_aux_index(&self.derivative_blocks, psi_j)
            {
                return self
                    .family
                    .sigma_exact_joint_psisecond_order_terms(&self.block_states);
            }
            return Err(
                "bernoulli marginal-slope mixed log-sigma/spatial psi second derivatives require cross auxiliary terms; only pure log-sigma second derivatives are supported"
                    .to_string(),
            );
        }
        self.family
            .exact_newton_joint_psisecond_order_terms_from_cache(
                &self.block_states,
                &self.derivative_blocks,
                psi_i,
                psi_j,
                &self.cache,
            )
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<crate::solver::estimate::reml::unified::DriftDerivResult>, String> {
        if self
            .family
            .is_sigma_aux_index(&self.derivative_blocks, psi_index)
        {
            return self
                .family
                .sigma_exact_joint_psihessian_directional_derivative(
                    &self.block_states,
                    d_beta_flat,
                )
                .map(|result| {
                    result.map(crate::solver::estimate::reml::unified::DriftDerivResult::Dense)
                });
        }
        self.family
            .exact_newton_joint_psihessian_directional_derivative_operator_from_cache(
                &self.block_states,
                &self.derivative_blocks,
                psi_index,
                d_beta_flat,
                &self.cache,
            )
            .map(|result| {
                result.map(crate::solver::estimate::reml::unified::DriftDerivResult::Operator)
            })
    }
}

fn build_blockspec(
    name: &str,
    design: &TermCollectionDesign,
    baseline: f64,
    offset: &Array1<f64>,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: name.to_string(),
        design: design.design.clone(),
        offset: offset + baseline,
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
    }
}

fn build_aux_blockspec(
    name: &str,
    prepared: &DeviationPrepared,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> Result<ParameterBlockSpec, String> {
    let mut block = prepared.block.clone();
    block.initial_log_lambdas = Some(rho);
    let candidate_beta = beta_hint.or_else(|| Some(Array1::<f64>::zeros(block.design.ncols())));
    block.initial_beta = candidate_beta
        .map(|beta| {
            let zero = Array1::<f64>::zeros(beta.len());
            project_monotone_feasible_beta(&prepared.runtime, &zero, &beta, name)
        })
        .transpose()?;
    block.intospec(name)
}

fn inner_fit(
    family: &BernoulliMarginalSlopeFamily,
    blocks: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, String> {
    fit_custom_family(family, blocks, options).map_err(|e| e.to_string())
}

pub fn fit_bernoulli_marginal_slope_terms(
    data: ArrayView2<'_, f64>,
    spec: BernoulliMarginalSlopeTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<BernoulliMarginalSlopeFitResult, String> {
    let mut spec = spec;
    let data_view = data;
    validate_spec(data_view, &spec)?;
    let (z_standardized, z_normalization) = standardize_latent_z_with_policy(
        &spec.z,
        &spec.weights,
        "bernoulli-marginal-slope",
        &spec.latent_z_policy,
    )?;
    spec.z = z_standardized;
    let pilot_baseline = pooled_probit_baseline(&spec.y, &spec.z, &spec.weights)?;
    let sigma_learnable = matches!(
        &spec.frailty,
        FrailtySpec::GaussianShift { sigma_fixed: None }
    );
    let initial_sigma = match &spec.frailty {
        FrailtySpec::GaussianShift {
            sigma_fixed: Some(s),
        } => Some(*s),
        FrailtySpec::GaussianShift { sigma_fixed: None } => Some(0.5),
        FrailtySpec::None => None,
        FrailtySpec::HazardMultiplier { .. } => {
            unreachable!("validate_spec rejects unsupported marginal-slope frailty")
        }
    };
    let probit_scale = probit_frailty_scale(initial_sigma);
    let baseline = (
        bernoulli_marginal_slope_eta_from_probability(
            &spec.base_link,
            normal_cdf(pilot_baseline.0),
            "bernoulli marginal-slope baseline link inversion",
        )?,
        pilot_baseline.1 / probit_scale,
    );
    let (mut joint_designs, mut joint_specs) = build_term_collection_designs_and_freeze_joint(
        data_view,
        &[spec.marginalspec.clone(), spec.logslopespec.clone()],
    )
    .map_err(|e| e.to_string())?;
    let marginal_design = joint_designs.remove(0);
    let logslope_design = joint_designs.remove(0);
    let marginalspec_boot = joint_specs.remove(0);
    let logslopespec_boot = joint_specs.remove(0);

    let y = Arc::new(spec.y.clone());
    let weights = Arc::new(spec.weights.clone());
    let z = Arc::new(spec.z.clone());

    let score_warp_prepared = spec
        .score_warp
        .as_ref()
        .map(|cfg| build_score_warp_deviation_block_from_seed(&spec.z, cfg))
        .transpose()?;
    // Build the link-deviation block if requested.  The seed only determines
    // knot placement for the deviation basis, so we use the closed-form
    // pooled-intercept probit solution instead of a full rigid pilot solve
    // (which would double total work at biobank scale):
    //   q0 ≈ a0 · √(1 + (s_f b0)²) + s_f b0 · z[i]
    // where b0 is the frailty-adjusted rigid logslope seed.
    let link_dev_prepared = spec
        .link_dev
        .as_ref()
        .map(|cfg| {
            let q0_seed = Array1::from_iter((0..spec.z.len()).map(|row| {
                let a0 = bernoulli_marginal_link_map(
                    &spec.base_link,
                    baseline.0 + spec.marginal_offset[row],
                )
                .expect("validated bernoulli marginal base link should produce finite pilot q")
                .q;
                let b0 = baseline.1 + spec.logslope_offset[row];
                rigid_observed_eta(a0, b0, spec.z[row], probit_scale)
            }));
            let link_dev_seed = padded_deviation_seed(&q0_seed, 1.0, 0.5);
            build_link_deviation_block_from_knots_design_seed_and_weights(
                &link_dev_seed,
                &q0_seed,
                &spec.weights,
                cfg,
            )
        })
        .transpose()?;
    let extra_rho0 = {
        let mut out = Vec::new();
        if let Some(ref prepared) = score_warp_prepared {
            out.extend(std::iter::repeat(0.0).take(prepared.block.penalties.len()));
        }
        if let Some(ref prepared) = link_dev_prepared {
            out.extend(std::iter::repeat(0.0).take(prepared.block.penalties.len()));
        }
        out
    };
    let setup = joint_setup(
        data_view,
        &marginalspec_boot,
        &logslopespec_boot,
        marginal_design.penalties.len(),
        logslope_design.penalties.len(),
        &extra_rho0,
        kappa_options,
    );
    let setup = if sigma_learnable {
        setup.with_auxiliary(
            Array1::from_vec(vec![initial_sigma.expect("learnable sigma seed").ln()]),
            Array1::from_vec(vec![0.01_f64.ln()]),
            Array1::from_vec(vec![5.0_f64.ln()]),
        )
    } else {
        setup
    };
    let final_sigma_cell = std::cell::Cell::new(initial_sigma);
    let exact_warm_start = RefCell::new(None::<CustomFamilyWarmStart>);
    let hints = RefCell::new(ThetaHints::default());
    let score_warp_runtime = score_warp_prepared.as_ref().map(|p| p.runtime.clone());
    let link_dev_runtime = link_dev_prepared.as_ref().map(|p| p.runtime.clone());

    let build_blocks = |rho: &Array1<f64>,
                        marginal_design: &TermCollectionDesign,
                        logslope_design: &TermCollectionDesign|
     -> Result<Vec<ParameterBlockSpec>, String> {
        let hints = hints.borrow();
        let mut cursor = 0usize;
        let rho_marginal = rho
            .slice(s![cursor..cursor + marginal_design.penalties.len()])
            .to_owned();
        cursor += marginal_design.penalties.len();
        let rho_logslope = rho
            .slice(s![cursor..cursor + logslope_design.penalties.len()])
            .to_owned();
        cursor += logslope_design.penalties.len();
        let mut blocks = vec![
            build_blockspec(
                "marginal_surface",
                marginal_design,
                baseline.0,
                &spec.marginal_offset,
                rho_marginal,
                hints.marginal_beta.clone(),
            ),
            build_blockspec(
                "logslope_surface",
                logslope_design,
                baseline.1,
                &spec.logslope_offset,
                rho_logslope,
                hints.logslope_beta.clone(),
            ),
        ];
        if let Some(ref prepared) = score_warp_prepared {
            let rho_h = rho
                .slice(s![cursor..cursor + prepared.block.penalties.len()])
                .to_owned();
            cursor += prepared.block.penalties.len();
            blocks.push(build_aux_blockspec(
                "score_warp_dev",
                prepared,
                rho_h,
                hints.score_warp_beta.clone(),
            )?);
        }
        if let Some(ref prepared) = link_dev_prepared {
            let rho_w = rho
                .slice(s![cursor..cursor + prepared.block.penalties.len()])
                .to_owned();
            blocks.push(build_aux_blockspec(
                "link_dev",
                prepared,
                rho_w,
                hints.link_dev_beta.clone(),
            )?);
        }
        Ok(blocks)
    };

    let make_family = |marginal_design: &TermCollectionDesign,
                       logslope_design: &TermCollectionDesign,
                       sigma: Option<f64>|
     -> BernoulliMarginalSlopeFamily {
        BernoulliMarginalSlopeFamily {
            y: Arc::clone(&y),
            weights: Arc::clone(&weights),
            z: Arc::clone(&z),
            gaussian_frailty_sd: sigma,
            base_link: spec.base_link.clone(),
            marginal_design: marginal_design.design.clone(),
            logslope_design: logslope_design.design.clone(),
            score_warp: score_warp_runtime.clone(),
            link_dev: link_dev_runtime.clone(),
        }
    };

    let marginal_terms = spatial_length_scale_term_indices(&marginalspec_boot);
    let logslope_terms = spatial_length_scale_term_indices(&logslopespec_boot);
    let marginal_has_spatial = !marginal_terms.is_empty();
    let logslope_has_spatial = !logslope_terms.is_empty();
    let analytic_joint_derivatives_available =
        marginal_has_spatial || logslope_has_spatial || setup.log_kappa_dim() == 0;
    if setup.log_kappa_dim() > 0 && !analytic_joint_derivatives_available {
        return Err(
            "exact bernoulli marginal-slope spatial optimization requires analytic joint psi derivatives"
                .to_string(),
        );
    }
    let initial_rho = setup.theta0().slice(s![..setup.rho_dim()]).to_owned();
    let initial_blocks = build_blocks(&initial_rho, &marginal_design, &logslope_design)?;
    let initial_family = make_family(&marginal_design, &logslope_design, initial_sigma);
    let (joint_gradient, joint_hessian) =
        custom_family_outer_derivatives(&initial_family, &initial_blocks, options);
    let analytic_joint_gradient_available = analytic_joint_derivatives_available
        && matches!(
            joint_gradient,
            crate::solver::outer_strategy::Derivative::Analytic
        );
    let analytic_joint_hessian_available = analytic_joint_derivatives_available
        && matches!(
            joint_hessian,
            crate::solver::outer_strategy::Derivative::Analytic
        );
    let sigma_from_theta = |theta: &Array1<f64>| -> Option<f64> {
        if sigma_learnable {
            Some(theta[setup.rho_dim() + setup.log_kappa_dim()].exp())
        } else {
            initial_sigma
        }
    };
    let derivative_block_cache = RefCell::new(
        None::<(
            Array1<f64>,
            Arc<Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>>,
        )>,
    );
    let theta_matches = |left: &Array1<f64>, right: &Array1<f64>| -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right.iter())
                .all(|(lhs, rhs)| (*lhs - *rhs).abs() <= 1e-12 * (1.0 + lhs.abs().max(rhs.abs())))
    };
    let get_derivative_blocks = |theta: &Array1<f64>,
                                 specs: &[TermCollectionSpec],
                                 designs: &[TermCollectionDesign]|
     -> Result<
        Arc<Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>>,
        String,
    > {
        if let Some((cached_theta, cached_blocks)) = derivative_block_cache.borrow().as_ref()
            && theta_matches(cached_theta, theta)
        {
            return Ok(Arc::clone(cached_blocks));
        }

        let built = |specs: &[TermCollectionSpec],
                     designs: &[TermCollectionDesign]|
         -> Result<
            Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
            String,
        > {
            let marginal_psi_derivs = if marginal_has_spatial {
                build_block_spatial_psi_derivatives(data_view, &specs[0], &designs[0])?.ok_or_else(
                    || {
                        "bernoulli marginal-slope: marginal block has spatial terms \
                         but spatial psi derivatives are unavailable"
                            .to_string()
                    },
                )?
            } else {
                Vec::new()
            };
            let logslope_psi_derivs = if logslope_has_spatial {
                build_block_spatial_psi_derivatives(data_view, &specs[1], &designs[1])?.ok_or_else(
                    || {
                        "bernoulli marginal-slope: logslope block has spatial terms \
                         but spatial psi derivatives are unavailable"
                            .to_string()
                    },
                )?
            } else {
                Vec::new()
            };
            let mut derivative_blocks = vec![marginal_psi_derivs, logslope_psi_derivs];
            if score_warp_runtime.is_some() {
                derivative_blocks.push(Vec::new());
            }
            if link_dev_runtime.is_some() {
                derivative_blocks.push(Vec::new());
            }
            if sigma_learnable {
                derivative_blocks
                    .last_mut()
                    .expect("bernoulli derivative block list is non-empty")
                    .push(crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                        None,
                        Array2::zeros((0, 0)),
                        Array2::zeros((0, 0)),
                        None,
                        None,
                        None,
                        None,
                    ));
            }
            Ok(derivative_blocks)
        }(specs, designs)?;
        let built = Arc::new(built);
        derivative_block_cache.replace(Some((theta.clone(), Arc::clone(&built))));
        Ok(built)
    };

    // Bernoulli marginal-slope is a multi-block family with β-dependent
    // joint Hessian: EFS/HybridEFS fixed-point structural invariant fails,
    // so we disable fixed-point at plan time rather than burning cycles on
    // a stalled first attempt that silently falls back.
    let solved = optimize_spatial_length_scale_exact_joint(
        data_view,
        &[marginalspec_boot.clone(), logslopespec_boot.clone()],
        &[marginal_terms.clone(), logslope_terms.clone()],
        kappa_options,
        &setup,
        crate::seeding::SeedRiskProfile::GeneralizedLinear,
        analytic_joint_gradient_available,
        analytic_joint_hessian_available,
        true,
        |theta, _: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            let rho = theta.slice(s![..setup.rho_dim()]).to_owned();
            let blocks = build_blocks(&rho, &designs[0], &designs[1])?;
            let sigma = sigma_from_theta(theta);
            final_sigma_cell.set(sigma);
            let family = make_family(&designs[0], &designs[1], sigma);
            let fit = inner_fit(&family, &blocks, options)?;
            let mut hints_mut = hints.borrow_mut();
            let mut bidx = 0usize;
            if let Some(block) = fit.block_states.get(bidx) {
                hints_mut.marginal_beta = Some(block.beta.clone());
            }
            bidx += 1;
            if let Some(block) = fit.block_states.get(bidx) {
                hints_mut.logslope_beta = Some(block.beta.clone());
            }
            bidx += 1;
            if score_warp_prepared.is_some() {
                if let Some(block) = fit.block_states.get(bidx) {
                    hints_mut.score_warp_beta = Some(block.beta.clone());
                }
                bidx += 1;
            }
            if link_dev_prepared.is_some() {
                if let Some(block) = fit.block_states.get(bidx) {
                    hints_mut.link_dev_beta = Some(block.beta.clone());
                }
            }
            Ok(fit)
        },
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign], need_hessian| {
            let rho = theta.slice(s![..setup.rho_dim()]).to_owned();
            let blocks = build_blocks(&rho, &designs[0], &designs[1])?;
            let sigma = sigma_from_theta(theta);
            final_sigma_cell.set(sigma);
            let family = make_family(&designs[0], &designs[1], sigma);
            let derivative_blocks = get_derivative_blocks(theta, specs, designs)?;
            let eval = evaluate_custom_family_joint_hyper_shared(
                &family,
                &blocks,
                options,
                &rho,
                derivative_blocks,
                exact_warm_start.borrow().as_ref(),
                if need_hessian && analytic_joint_hessian_available {
                    crate::solver::estimate::reml::unified::EvalMode::ValueGradientHessian
                } else {
                    crate::solver::estimate::reml::unified::EvalMode::ValueAndGradient
                },
            )?;
            exact_warm_start.replace(Some(eval.warm_start));
            if need_hessian && analytic_joint_hessian_available && !eval.outer_hessian.is_analytic()
            {
                return Err(
                    "exact bernoulli marginal-slope joint [rho, psi] objective did not return an outer Hessian"
                        .to_string(),
                );
            }
            Ok((eval.objective, eval.gradient, eval.outer_hessian))
        },
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            let rho = theta.slice(s![..setup.rho_dim()]).to_owned();
            let blocks = build_blocks(&rho, &designs[0], &designs[1])?;
            let sigma = sigma_from_theta(theta);
            final_sigma_cell.set(sigma);
            let family = make_family(&designs[0], &designs[1], sigma);
            let derivative_blocks = get_derivative_blocks(theta, specs, designs)?;
            let eval = evaluate_custom_family_joint_hyper_efs_shared(
                &family,
                &blocks,
                options,
                &rho,
                derivative_blocks,
                exact_warm_start.borrow().as_ref(),
            )?;
            exact_warm_start.replace(Some(eval.warm_start));
            Ok(eval.efs_eval)
        },
    )?;

    let mut resolved_specs = solved.resolved_specs;
    let mut designs = solved.designs;
    Ok(BernoulliMarginalSlopeFitResult {
        fit: solved.fit,
        marginalspec_resolved: resolved_specs.remove(0),
        logslopespec_resolved: resolved_specs.remove(0),
        marginal_design: designs.remove(0),
        logslope_design: designs.remove(0),
        baseline_marginal: baseline.0,
        baseline_logslope: baseline.1,
        z_normalization,
        score_warp_runtime,
        link_dev_runtime,
        gaussian_frailty_sd: final_sigma_cell.get(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_family::CustomFamily;
    use crate::families::bernoulli_marginal_slope::exact_kernel::{
        DenestedCubicCell as ExactDenestedCubicCell, ExactCellBranch as ExactCellBranchShared,
        LocalSpanCubic, branch_cell as branch_exact_cell, build_denested_partition_cells,
        denested_cell_coefficient_partials as exact_denested_cell_coefficient_partials,
        global_cubic_from_local as exact_global_cubic_from_local,
        transformed_link_cubic as exact_transformed_link_cubic,
    };
    use ndarray::array;

    #[inline]
    fn bernoulli_marginal_slope_probit_link() -> InverseLink {
        InverseLink::Standard(LinkFunction::Probit)
    }

    fn empty_termspec() -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        }
    }

    fn dummy_blockspec(p: usize, n_rows: usize) -> ParameterBlockSpec {
        ParameterBlockSpec {
            name: "dummy".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::<f64>::zeros((n_rows, p)),
            )),
            offset: Array1::zeros(n_rows),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(Array1::zeros(p)),
        }
    }

    fn dummy_block_state(beta: Array1<f64>, n_rows: usize) -> ParameterBlockState {
        ParameterBlockState {
            beta,
            eta: Array1::zeros(n_rows),
        }
    }

    fn base_spec(
        y: Array1<f64>,
        weights: Array1<f64>,
        z: Array1<f64>,
    ) -> BernoulliMarginalSlopeTermSpec {
        let n = y.len();
        BernoulliMarginalSlopeTermSpec {
            y,
            weights,
            z,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginalspec: empty_termspec(),
            logslopespec: empty_termspec(),
            marginal_offset: Array1::zeros(n),
            logslope_offset: Array1::zeros(n),
            frailty: FrailtySpec::None,
            score_warp: None,
            link_dev: None,
            latent_z_policy: LatentZPolicy::default(),
        }
    }

    fn pair_distance(lhs: (f64, f64), rhs: (f64, f64)) -> f64 {
        (lhs.0 - rhs.0).abs() + (lhs.1 - rhs.1).abs()
    }

    fn build_test_link_deviation_block_from_seed(
        seed: &Array1<f64>,
        cfg: &DeviationBlockConfig,
    ) -> Result<DeviationPrepared, String> {
        build_link_deviation_block_from_knots_design_seed_and_weights(
            seed,
            seed,
            &Array1::ones(seed.len()),
            cfg,
        )
    }

    #[test]
    fn score_warp_basis_is_orthogonal_to_standard_normal_location_and_scale() {
        let seed = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 5,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build standard-normal anchored score-warp");
        let rule = crate::quadrature::compute_gauss_hermite_n(51);
        let z = Array1::from_iter(
            rule.nodes
                .iter()
                .map(|&node| std::f64::consts::SQRT_2 * node),
        );
        let design = prepared
            .runtime
            .design(&z)
            .expect("score-warp quadrature design");
        let inv_sqrt_pi = std::f64::consts::PI.sqrt().recip();
        for basis_idx in 0..design.ncols() {
            let mut mean_moment = 0.0;
            let mut scale_moment = 0.0;
            for row in 0..design.nrows() {
                let weight = rule.weights[row] * inv_sqrt_pi;
                mean_moment += weight * design[[row, basis_idx]];
                scale_moment += weight * z[row] * design[[row, basis_idx]];
            }
            assert!(
                mean_moment.abs() <= 1e-10,
                "score-warp basis column {basis_idx} has nonzero standard-normal mean moment {mean_moment}"
            );
            assert!(
                scale_moment.abs() <= 1e-10,
                "score-warp basis column {basis_idx} has nonzero standard-normal scale moment {scale_moment}"
            );
        }
    }

    #[test]
    fn link_deviation_basis_is_orthogonal_to_weighted_training_index_moments() {
        let q = array![-2.0, -0.8, -0.1, 0.4, 1.3, 2.1];
        let weights = array![0.2, 1.7, 0.5, 2.3, 0.8, 1.1];
        let prepared = build_link_deviation_block_from_knots_design_seed_and_weights(
            &q,
            &q,
            &weights,
            &DeviationBlockConfig {
                num_internal_knots: 5,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build weighted empirical anchored link deviation");
        let design = prepared
            .runtime
            .design(&q)
            .expect("link-deviation training design");
        let total_weight: f64 = weights.iter().copied().sum();
        for basis_idx in 0..design.ncols() {
            let mut mean_moment = 0.0;
            let mut scale_moment = 0.0;
            for row in 0..design.nrows() {
                let weight = weights[row] / total_weight;
                mean_moment += weight * design[[row, basis_idx]];
                scale_moment += weight * q[row] * design[[row, basis_idx]];
            }
            assert!(
                mean_moment.abs() <= 1e-10,
                "link-deviation basis column {basis_idx} has nonzero weighted mean moment {mean_moment}"
            );
            assert!(
                scale_moment.abs() <= 1e-10,
                "link-deviation basis column {basis_idx} has nonzero weighted scale moment {scale_moment}"
            );
        }
    }

    #[test]
    fn bernoulli_marginal_slope_rejects_nonprobit_base_link() {
        let y = array![0.0, 1.0];
        let weights = array![1.0, 1.0];
        let z = array![-0.4, 0.9];
        let design =
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0], [1.0]]));
        let spec = BernoulliMarginalSlopeTermSpec {
            y,
            weights,
            z,
            base_link: InverseLink::Standard(LinkFunction::Logit),
            marginalspec: empty_termspec(),
            logslopespec: empty_termspec(),
            marginal_offset: Array1::zeros(2),
            logslope_offset: Array1::zeros(2),
            frailty: FrailtySpec::None,
            score_warp: None,
            link_dev: None,
            latent_z_policy: LatentZPolicy::default(),
        };
        let err = validate_spec(design.to_dense().view(), &spec)
            .expect_err("non-probit marginal-slope link should be rejected");
        assert!(err.contains("requires link(type=probit)"));
        let err = bernoulli_marginal_slope_eta_from_probability(
            &InverseLink::Standard(LinkFunction::Logit),
            0.5,
            "test logit inverse",
        )
        .expect_err("non-probit marginal-slope inverse should be rejected");
        assert!(err.contains("requires link(type=probit)"));
    }

    fn expand_integer_weight_rows(
        y: &Array1<f64>,
        z: &Array1<f64>,
        weights: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        let mut y_expanded = Vec::new();
        let mut z_expanded = Vec::new();
        for i in 0..y.len() {
            let reps = weights[i] as usize;
            assert!(
                (weights[i] - reps as f64).abs() < 1e-12,
                "test helper expects integer weights, got {}",
                weights[i]
            );
            for _ in 0..reps {
                y_expanded.push(y[i]);
                z_expanded.push(z[i]);
            }
        }
        (Array1::from_vec(y_expanded), Array1::from_vec(z_expanded))
    }

    #[test]
    fn link_dev_without_score_warp_exposes_structural_derivative_lower_bounds() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_test_link_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build link deviation block");
        let link_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("link block initial beta")
            .len();
        let beta_link = Array1::from_iter((0..link_dim).map(|idx| 0.1 * (idx as f64 + 1.0)));
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(Array1::zeros(seed.len())),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: None,
            link_dev: Some(prepared.runtime.clone()),
        };
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(beta_link.clone(), seed.len()),
        ];

        let slices = block_slices(&family);
        assert!(slices.h.is_none(), "score-warp slice should be absent");
        let link_slice = slices.w.as_ref().expect("link slice");
        assert_eq!(
            slices.marginal.len(),
            0,
            "zero-column marginal design should not contribute coefficient coordinates"
        );
        assert_eq!(
            slices.logslope.len(),
            0,
            "zero-column logslope design should not contribute coefficient coordinates"
        );
        assert_eq!(
            link_slice.start, 0,
            "link-only coefficients should start at 0"
        );
        assert_eq!(link_slice.len(), link_dim);

        let primary = primary_slices(&slices);
        assert!(primary.h.is_none(), "primary h slice should be absent");
        let primary_w = primary.w.as_ref().expect("primary link slice");
        assert_eq!(primary_w.start, 2, "primary link slice should start at 2");
        assert_eq!(primary.total, 2 + link_dim);

        // Verify that the analytic IFT path produces finite gradient/Hessian
        // for a link-dev-only family.
        family
            .build_exact_eval_cache(&block_states)
            .expect("eval cache");
        let row_ctx = family
            .build_row_exact_context(0, &block_states)
            .expect("row context");
        let (nll, grad, hess) = family
            .compute_row_primary_gradient_hessian(0, &block_states, &primary, &row_ctx)
            .expect("analytic flex eval");
        assert!(nll.is_finite(), "neglog should be finite for link-dev-only");
        assert!(
            grad.iter().all(|v| v.is_finite()),
            "gradient should be finite"
        );
        assert!(
            hess.iter().all(|v| v.is_finite()),
            "Hessian should be finite"
        );

        let dummy_spec = dummy_blockspec(link_dim, seed.len());
        assert!(
            family
                .block_linear_constraints(&block_states, 1, &dummy_spec)
                .expect("non-link constraint lookup")
                .is_none(),
            "non-link block should not expose auxiliary monotonicity constraints"
        );
        let constraints = family
            .block_linear_constraints(&block_states, 2, &dummy_spec)
            .expect("link constraint lookup")
            .expect("link constraints");
        assert_eq!(constraints.a.ncols(), link_dim);
        assert_eq!(constraints.b.len(), constraints.a.nrows());
        assert!(
            constraints.a.nrows() >= link_dim,
            "anchored link constraints should be expressed in raw derivative-control rows"
        );
        assert_eq!(
            constraints.b,
            Array1::<f64>::from_elem(
                constraints.a.nrows(),
                prepared.runtime.monotonicity_eps() - 1.0
            )
        );
    }

    #[test]
    fn exact_layout_ignores_dummy_beta_widths_for_empty_design_blocks() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score-warp block");
        let link_prepared = build_test_link_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build link deviation block");
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(Array1::zeros(seed.len())),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: Some(score_prepared.runtime.clone()),
            link_dev: Some(link_prepared.runtime.clone()),
        };
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(
                Array1::zeros(score_prepared.runtime.basis_dim()),
                seed.len(),
            ),
            dummy_block_state(Array1::zeros(link_prepared.runtime.basis_dim()), seed.len()),
        ];

        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("exact eval cache");
        assert_eq!(cache.slices.marginal.len(), 0);
        assert_eq!(cache.slices.logslope.len(), 0);
        assert_eq!(cache.slices.h.as_ref().expect("h slice").start, 0);
        assert_eq!(
            cache.slices.w.as_ref().expect("w slice").start,
            score_prepared.runtime.basis_dim()
        );
        assert_eq!(
            cache.slices.total,
            score_prepared.runtime.basis_dim() + link_prepared.runtime.basis_dim()
        );
        assert_eq!(cache.primary.q, 0);
        assert_eq!(cache.primary.logslope, 1);
        assert_eq!(cache.primary.h.as_ref().expect("primary h").start, 2);
        assert_eq!(
            cache.primary.w.as_ref().expect("primary w").start,
            2 + score_prepared.runtime.basis_dim()
        );
    }

    #[test]
    fn score_warp_block_exposes_structural_derivative_lower_bounds() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score-warp block");
        let score_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("score-warp initial beta")
            .len();
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(Array1::zeros(seed.len())),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: Some(prepared.runtime.clone()),
            link_dev: None,
        };
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(Array1::zeros(score_dim), seed.len()),
        ];

        let dummy_spec = dummy_blockspec(score_dim, seed.len());
        let constraints = family
            .block_linear_constraints(&block_states, 2, &dummy_spec)
            .expect("constraint lookup")
            .expect("score-warp constraints");
        assert_eq!(constraints.a.ncols(), score_dim);
        assert_eq!(constraints.b.len(), constraints.a.nrows());
        assert!(
            constraints.a.nrows() >= score_dim,
            "anchored score-warp constraints should be expressed in raw derivative-control rows"
        );
        assert_eq!(
            constraints.b,
            Array1::<f64>::from_elem(
                constraints.a.nrows(),
                prepared.runtime.monotonicity_eps() - 1.0
            )
        );
    }

    #[test]
    fn post_update_block_beta_projects_score_warp_to_feasible_step() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score-warp block");
        let score_dim = prepared.block.design.ncols();
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(Array1::zeros(seed.len())),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: Some(prepared.runtime.clone()),
            link_dev: None,
        };
        let current = Array1::<f64>::zeros(score_dim);
        let mut proposed = current.clone();
        proposed[0] = -128.0;
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(current.clone(), seed.len()),
        ];
        let spec = dummy_blockspec(score_dim, seed.len());
        let updated = family
            .post_update_block_beta(&block_states, 2, &spec, proposed.clone())
            .expect("projected beta");
        prepared
            .runtime
            .monotonicity_feasible(&updated, "projected score-warp")
            .expect("post-update beta should remain feasible");
        assert_ne!(updated, proposed);
    }

    #[test]
    fn structural_deviation_runtime_is_piecewise_cubic() {
        let seed = array![-1.0, 0.0, 1.0];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                ..DeviationBlockConfig::default()
            },
        )
        .expect("structural deviation basis");
        assert_eq!(prepared.runtime.degree(), 3);
        assert_eq!(prepared.runtime.value_span_degree(), 3);
        let has_cubic_curvature = prepared
            .runtime
            .span_c3()
            .iter()
            .any(|value| value.abs() > 1e-12);
        assert!(
            has_cubic_curvature,
            "structural deviation basis must expose true cubic span coefficients"
        );
    }

    #[test]
    fn structural_deviation_runtime_is_c2_at_internal_breakpoints() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("structural deviation basis");
        let dim = prepared.block.design.ncols();
        let beta = Array1::from_iter((0..dim).map(|idx| 0.015 * (idx as f64 + 1.0)));
        let n_spans = prepared.runtime.breakpoints().len().saturating_sub(1);
        for span_idx in 1..n_spans {
            let left_cubic = prepared
                .runtime
                .local_cubic_on_span(&beta, span_idx - 1)
                .expect("left span cubic");
            let right_cubic = prepared
                .runtime
                .local_cubic_on_span(&beta, span_idx)
                .expect("right span cubic");
            let knot = prepared.runtime.breakpoints()[span_idx];
            assert!(
                (left_cubic.evaluate(knot) - right_cubic.evaluate(knot)).abs() <= 1e-10,
                "deviation value should be continuous at breakpoint {span_idx}"
            );
            assert!(
                (left_cubic.first_derivative(knot) - right_cubic.first_derivative(knot)).abs()
                    <= 1e-10,
                "deviation first derivative should be continuous at breakpoint {span_idx}"
            );
            assert!(
                (left_cubic.second_derivative(knot) - right_cubic.second_derivative(knot)).abs()
                    <= 1e-10,
                "deviation second derivative should be continuous at breakpoint {span_idx}"
            );
        }
    }

    #[test]
    fn structural_deviation_rejects_noncubic_degree() {
        let seed = array![-1.0, 0.0, 1.0];
        let err = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                degree: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect_err("structural deviation block should reject non-cubic degree");
        assert!(err.contains("degree must be 3"));
    }

    #[test]
    fn deviation_runtime_replays_exact_training_design() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build deviation block");
        let replayed = prepared.runtime.design(&seed).expect("replayed design");
        let trained = prepared.block.design.to_dense();
        assert_eq!(replayed.dim(), trained.dim());
        for i in 0..replayed.nrows() {
            for j in 0..replayed.ncols() {
                assert!(
                    (replayed[[i, j]] - trained[[i, j]]).abs() <= 1e-10,
                    "training-basis replay mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn structural_constraints_match_exact_monotonicity_guard() {
        let seed = array![-1.0, 0.0, 1.0, 2.0];
        let prepared =
            build_score_warp_deviation_block_from_seed(&seed, &DeviationBlockConfig::default())
                .expect("build deviation block");
        let constraints = prepared.runtime.structural_monotonicity_constraints();
        let dim = constraints.a.ncols();
        assert_eq!(dim, prepared.runtime.basis_dim());
        assert_eq!(
            constraints.a.nrows(),
            3 * prepared.runtime.breakpoints().len().saturating_sub(1)
        );
        assert_eq!(
            constraints.b,
            Array1::<f64>::from_elem(
                constraints.a.nrows(),
                prepared.runtime.monotonicity_eps() - 1.0
            )
        );
        let feasible = Array1::<f64>::zeros(dim);
        prepared
            .runtime
            .monotonicity_feasible(&feasible, "feasible structural beta")
            .expect("zero deviation should be feasible");
        let d1_design = prepared
            .runtime
            .first_derivative_design(&seed)
            .expect("derivative design");
        let row_idx = (0..d1_design.nrows())
            .find(|&idx| d1_design.row(idx).dot(&d1_design.row(idx)) > 0.0)
            .expect("derivative design should include a nonzero row");
        let derivative_row = d1_design.row(row_idx).to_owned();
        let row_norm_sq = derivative_row.dot(&derivative_row);
        let infeasible = derivative_row.mapv(|value| -2.0 * value / row_norm_sq);
        assert!(
            prepared
                .runtime
                .monotonicity_feasible(&infeasible, "infeasible structural beta")
                .is_err()
        );
    }

    #[test]
    fn structural_constraints_are_quadratic_derivative_bernstein_controls() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build deviation block");
        let constraints = prepared.runtime.structural_monotonicity_constraints();
        let beta = Array1::from_iter((0..prepared.runtime.basis_dim()).map(|idx| {
            let centered = idx as f64 - 0.5 * (prepared.runtime.basis_dim() as f64 - 1.0);
            0.025 * centered
        }));
        let controls = constraints.a.dot(&beta);
        let n_spans = prepared.runtime.breakpoints().len().saturating_sub(1);
        for span_idx in 0..n_spans {
            let cubic = prepared
                .runtime
                .local_cubic_on_span(&beta, span_idx)
                .expect("local cubic");
            let left = cubic.left;
            let right = cubic.right;
            let mid = 0.5 * (left + right);
            let b0 = controls[3 * span_idx];
            let b1 = controls[3 * span_idx + 1];
            let b2 = controls[3 * span_idx + 2];
            assert!(
                (b0 - cubic.first_derivative(left)).abs() <= 1e-10,
                "left Bernstein control should equal derivative at span start"
            );
            assert!(
                (b2 - cubic.first_derivative(right)).abs() <= 1e-10,
                "right Bernstein control should equal derivative at span end"
            );
            let midpoint_from_bernstein = 0.25 * b0 + 0.5 * b1 + 0.25 * b2;
            assert!(
                (midpoint_from_bernstein - cubic.first_derivative(mid)).abs() <= 1e-10,
                "quadratic Bernstein controls should reconstruct derivative at span midpoint"
            );
        }
    }

    #[test]
    fn deviation_penalties_are_integrated_function_penalties() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                penalty_order: 2,
                penalty_orders: vec![1, 2, 3],
                double_penalty: true,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build deviation block");

        let expected_orders = [1, 0, 2, 3];
        assert_eq!(prepared.block.penalties.len(), expected_orders.len());

        for ((penalty, &nullity), &order) in prepared
            .block
            .penalties
            .iter()
            .zip(prepared.block.nullspace_dims.iter())
            .zip(expected_orders.iter())
        {
            let crate::solver::estimate::PenaltySpec::Dense(actual) = penalty else {
                panic!("deviation penalties should be dense local Gram matrices");
            };
            let (expected, expected_nullity) = prepared
                .runtime
                .integrated_derivative_penalty_with_nullity(order)
                .expect("integrated function penalty");
            assert_eq!(nullity, expected_nullity);
            assert_eq!(actual.dim(), expected.dim());
            for i in 0..actual.nrows() {
                for j in 0..actual.ncols() {
                    assert!(
                        (actual[[i, j]] - expected[[i, j]]).abs() <= 1e-10,
                        "penalty order {order} mismatch at ({i},{j}): got {}, expected {}",
                        actual[[i, j]],
                        expected[[i, j]]
                    );
                }
            }
        }

        let crate::solver::estimate::PenaltySpec::Dense(l2_penalty) = &prepared.block.penalties[1]
        else {
            panic!("deviation double penalty should be dense");
        };
        let mut max_identity_diff = 0.0_f64;
        for i in 0..l2_penalty.nrows() {
            for j in 0..l2_penalty.ncols() {
                let identity = if i == j { 1.0 } else { 0.0 };
                max_identity_diff = max_identity_diff.max((l2_penalty[[i, j]] - identity).abs());
            }
        }
        assert!(
            max_identity_diff > 1e-6,
            "deviation double penalty must be integrated L2, not coefficient identity"
        );
    }

    #[test]
    fn local_cubic_span_reconstructs_deviation_exactly() {
        // Score-warp deviation runtime: C² piecewise-cubic basis.
        //
        // Continuity across interior breakpoints:
        //   value  (d0) — C⁰ continuous (matches on both sides)
        //   slope  (d1) — C¹ continuous (matches on both sides)
        //   curvature (d2) — C² continuous (matches on both sides)
        //
        // `evaluate_span_polynomial_design` resolves the two-sided ambiguity at
        // an interior breakpoint x == endpoint_points[k] (0 < k < last) by
        // biasing to the LEFT span (span_idx = k - 1). For a C² cubic this is
        // numerically the same value through d2; only d3 is span-local.
        //
        // This test reconstructs each span's polynomial from design rows.
        // For d0/d1/d2 the expected value matches the selected span at every
        // sample point.
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build deviation block");
        let dim = prepared.block.design.ncols();
        let beta = Array1::from_iter((0..dim).map(|idx| 0.025 * (idx as f64 + 1.0)));
        let n_spans = prepared.runtime.breakpoints().len().saturating_sub(1);
        let support_left = prepared.runtime.breakpoints()[0];
        let support_right =
            prepared.runtime.breakpoints()[prepared.runtime.breakpoints().len() - 1];

        for span_idx in 0..n_spans {
            let cubic = prepared
                .runtime
                .local_cubic_on_span(&beta, span_idx)
                .expect("local cubic coefficients");
            let left = cubic.left;
            let right = cubic.right;
            let x_eval = array![left, 0.5 * (left + right), right];
            let value_design = prepared.runtime.design(&x_eval).expect("value design");
            let d1_design = prepared
                .runtime
                .first_derivative_design(&x_eval)
                .expect("first derivative design");
            let d2_design = prepared
                .runtime
                .second_derivative_design(&x_eval)
                .expect("second derivative design");
            let expected = value_design.dot(&beta);
            let expected_d1 = d1_design.dot(&beta);
            let expected_d2 = d2_design.dot(&beta);
            for i in 0..x_eval.len() {
                let x = x_eval[i];
                assert!(
                    (cubic.evaluate(x) - expected[i]).abs() < 1e-10,
                    "span {span_idx}, x={x:.6}: cubic value mismatch"
                );
                assert!(
                    (cubic.first_derivative(x) - expected_d1[i]).abs() < 1e-10,
                    "span {span_idx}, x={x:.6}: cubic first-derivative mismatch"
                );
                assert!(
                    (cubic.second_derivative(x) - expected_d2[i]).abs() < 1e-10,
                    "span {span_idx}, x={x:.6}: cubic second-derivative mismatch"
                );
                let selected = prepared
                    .runtime
                    .local_cubic_at(&beta, x)
                    .expect("lookup cubic at x");
                if x < support_left || x > support_right {
                    // Strictly outside support: tail saturation — constant
                    // value, zero slope and curvature.
                    assert!(selected.c1.abs() < 1e-12);
                    assert!(selected.c2.abs() < 1e-12);
                    assert!(selected.c3.abs() < 1e-12);
                    assert!((selected.evaluate(x) - expected[i]).abs() < 1e-10);
                } else {
                    // Interior or exact boundary point: uses the same
                    // left-biased span convention as derivative designs.
                    let expected_span_idx = if i == 0 && span_idx > 0 {
                        span_idx - 1
                    } else {
                        span_idx
                    };
                    let expected_cubic = prepared
                        .runtime
                        .local_cubic_on_span(&beta, expected_span_idx)
                        .expect("expected lookup cubic on span");
                    assert_eq!(selected, expected_cubic);
                }
            }
        }
    }

    #[test]
    fn basis_span_cubic_reconstructs_basis_column_exactly() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build deviation block");
        let basis_idx = 0usize;
        let support_left = prepared.runtime.breakpoints()[0];
        let support_right =
            prepared.runtime.breakpoints()[prepared.runtime.breakpoints().len() - 1];
        let cubic = prepared
            .runtime
            .basis_span_cubic(0, basis_idx)
            .expect("basis span cubic");
        let x_eval = array![cubic.left, 0.5 * (cubic.left + cubic.right), cubic.right];
        let design = prepared.runtime.design(&x_eval).expect("basis design");
        let d1 = prepared
            .runtime
            .first_derivative_design(&x_eval)
            .expect("basis d1 design");
        for i in 0..x_eval.len() {
            let x = x_eval[i];
            assert!((cubic.evaluate(x) - design[[i, basis_idx]]).abs() < 1e-10);
            assert!((cubic.first_derivative(x) - d1[[i, basis_idx]]).abs() < 1e-10);
            let selected = prepared
                .runtime
                .basis_cubic_at(basis_idx, x)
                .expect("basis cubic at x");
            if x < support_left || x > support_right {
                // Strictly outside support: tail saturation.
                assert!(selected.c1.abs() < 1e-12);
                assert!(selected.c2.abs() < 1e-12);
                assert!(selected.c3.abs() < 1e-12);
                assert!((selected.evaluate(x) - design[[i, basis_idx]]).abs() < 1e-10);
            } else {
                let expected_span_idx = 0;
                let expected_cubic = prepared
                    .runtime
                    .basis_span_cubic(expected_span_idx, basis_idx)
                    .expect("expected basis span cubic");
                assert_eq!(selected, expected_cubic);
            }
        }
    }

    #[test]
    fn deviation_runtime_saturates_outside_support() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build deviation block");
        let dim = prepared.block.design.ncols();
        let beta = Array1::from_iter((0..dim).map(|idx| 0.02 * (idx as f64 + 1.0)));
        let left = prepared.runtime.breakpoints()[0];
        let right = prepared.runtime.breakpoints()[prepared.runtime.breakpoints().len() - 1];

        let left_tail_near = prepared
            .runtime
            .local_cubic_at(&beta, left - 0.25)
            .expect("left tail");
        let left_tail_far = prepared
            .runtime
            .local_cubic_at(&beta, left - 3.0)
            .expect("left far tail");
        let right_tail_near = prepared
            .runtime
            .local_cubic_at(&beta, right + 0.25)
            .expect("right tail");
        let right_tail_far = prepared
            .runtime
            .local_cubic_at(&beta, right + 3.0)
            .expect("right far tail");

        for cubic in [
            left_tail_near,
            left_tail_far,
            right_tail_near,
            right_tail_far,
        ] {
            assert!(cubic.c1.abs() < 1e-12);
            assert!(cubic.c2.abs() < 1e-12);
            assert!(cubic.c3.abs() < 1e-12);
        }
        assert!((left_tail_near.c0 - left_tail_far.c0).abs() < 1e-12);
        assert!((right_tail_near.c0 - right_tail_far.c0).abs() < 1e-12);
    }

    #[test]
    fn deviation_runtime_replays_the_exact_training_basis() {
        let seed = array![-2.0, -1.0, -0.25, 0.25, 1.0, 2.0];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build deviation block");

        let replayed = prepared
            .runtime
            .design(&seed)
            .expect("replay anchored deviation design");
        let trained = prepared.block.design.to_dense();
        assert_eq!(replayed.dim(), trained.dim());
        for i in 0..replayed.nrows() {
            for j in 0..replayed.ncols() {
                assert!(
                    (replayed[[i, j]] - trained[[i, j]]).abs() <= 1e-10,
                    "replayed anchored deviation design mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn denested_microcells_follow_score_and_link_breaks() {
        let score_seed = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let link_seed = array![-1.5, -0.5, 0.5, 1.5];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &score_seed,
            &DeviationBlockConfig {
                num_internal_knots: 3,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score warp block");
        let link_prepared = build_test_link_deviation_block_from_seed(
            &link_seed,
            &DeviationBlockConfig {
                num_internal_knots: 3,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build link deviation block");
        let beta_h = Array1::from_iter(
            (0..score_prepared.block.design.ncols()).map(|idx| 0.02 * (idx as f64 + 1.0)),
        );
        let beta_w = Array1::from_iter(
            (0..link_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
        );

        let exact_cells_a0 = build_denested_partition_cells(
            0.25,
            0.9,
            score_prepared
                .runtime
                .breakpoints()
                .as_slice()
                .expect("score breaks"),
            link_prepared
                .runtime
                .breakpoints()
                .as_slice()
                .expect("link breaks"),
            |z| score_prepared.runtime.local_cubic_at(&beta_h, z),
            |u| link_prepared.runtime.local_cubic_at(&beta_w, u),
        )
        .expect("exact module microcells for a=0.25");
        let exact_cells_a1 = build_denested_partition_cells(
            0.55,
            0.9,
            score_prepared
                .runtime
                .breakpoints()
                .as_slice()
                .expect("score breaks"),
            link_prepared
                .runtime
                .breakpoints()
                .as_slice()
                .expect("link breaks"),
            |z| score_prepared.runtime.local_cubic_at(&beta_h, z),
            |u| link_prepared.runtime.local_cubic_at(&beta_w, u),
        )
        .expect("exact module microcells for a=0.55");

        assert!(
            exact_cells_a0.len() >= score_prepared.runtime.breakpoints().len().saturating_sub(1),
            "microcell partition should refine the score spans"
        );
        assert!(
            exact_cells_a0
                .windows(2)
                .all(|w| (w[0].cell.right - w[1].cell.left).abs() <= 1e-12),
            "microcells should tile the partition contiguously"
        );
        assert!(exact_cells_a0.first().unwrap().cell.left.is_infinite());
        assert!(exact_cells_a0.last().unwrap().cell.right.is_infinite());
        assert!(
            exact_cells_a0
                .iter()
                .zip(exact_cells_a1.iter())
                .any(|(lhs, rhs)| (lhs.cell.left - rhs.cell.left).abs() > 1e-10),
            "changing the intercept should move at least one link-induced breakpoint"
        );
    }

    #[test]
    fn denested_microcell_eta_matches_direct_denested_formula() {
        let score_seed = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let link_seed = array![-1.5, -0.5, 0.5, 1.5];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &score_seed,
            &DeviationBlockConfig {
                num_internal_knots: 3,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score warp block");
        let link_prepared = build_test_link_deviation_block_from_seed(
            &link_seed,
            &DeviationBlockConfig {
                num_internal_knots: 3,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build link deviation block");
        let beta_h = Array1::from_iter(
            (0..score_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
        );
        let beta_w = Array1::from_iter(
            (0..link_prepared.block.design.ncols()).map(|idx| 0.02 * (idx as f64 + 1.0)),
        );

        let a = 0.35;
        let b = -0.7;
        let cells = build_denested_partition_cells(
            a,
            b,
            score_prepared
                .runtime
                .breakpoints()
                .as_slice()
                .expect("score breaks"),
            link_prepared
                .runtime
                .breakpoints()
                .as_slice()
                .expect("link breaks"),
            |z| score_prepared.runtime.local_cubic_at(&beta_h, z),
            |u| link_prepared.runtime.local_cubic_at(&beta_w, u),
        )
        .expect("microcells");

        for cell in &cells {
            let z = exact_kernel::interval_probe_point(cell.cell.left, cell.cell.right)
                .expect("finite microcell probe");
            let h = score_prepared
                .runtime
                .design(&array![z])
                .expect("score design")
                .row(0)
                .dot(&beta_h);
            let link = link_prepared
                .runtime
                .design(&array![a + b * z])
                .expect("link design")
                .row(0)
                .dot(&beta_w);
            let expected = a + b * z + b * h + link;
            assert!(
                (cell.cell.eta(z) - expected).abs() < 1e-10,
                "microcell eta should equal the direct de-nested predictor at z={z:.6}"
            );
        }
    }

    #[test]
    fn local_cubic_global_transform_reconstructs_same_function() {
        let cubic = exact_kernel::LocalSpanCubic {
            left: -1.3,
            right: 0.7,
            c0: 0.4,
            c1: -0.2,
            c2: 0.15,
            c3: -0.05,
        };
        let (g0, g1, g2, g3) = exact_global_cubic_from_local(LocalSpanCubic {
            left: cubic.left,
            right: cubic.right,
            c0: cubic.c0,
            c1: cubic.c1,
            c2: cubic.c2,
            c3: cubic.c3,
        });
        for &x in &[-1.3, -0.8, -0.1, 0.5, 0.7] {
            let direct = cubic.evaluate(x);
            let global = g0 + g1 * x + g2 * x * x + g3 * x * x * x;
            assert!(
                (direct - global).abs() < 1e-12,
                "global cubic transform should preserve the span polynomial at x={x}"
            );
        }
    }

    #[test]
    fn denested_branch_selection_uses_normalized_cell_coefficients() {
        let affine = ExactDenestedCubicCell {
            left: -1.0,
            right: 1.0,
            c0: 0.2,
            c1: -0.4,
            c2: 1e-13,
            c3: -1e-13,
        };
        let quartic = ExactDenestedCubicCell {
            c2: 2e-4,
            c3: 1e-13,
            ..affine
        };
        let sextic = ExactDenestedCubicCell {
            c2: 2e-4,
            c3: 5e-3,
            ..affine
        };
        assert_eq!(
            branch_exact_cell(affine).expect("affine branch"),
            ExactCellBranchShared::Affine
        );
        assert_eq!(
            branch_exact_cell(quartic).expect("quartic branch"),
            ExactCellBranchShared::Quartic
        );
        assert_eq!(
            branch_exact_cell(sextic).expect("sextic branch"),
            ExactCellBranchShared::Sextic
        );
    }

    #[test]
    fn denested_cell_coefficient_partials_match_finite_differences() {
        let score_span = exact_kernel::LocalSpanCubic {
            left: -0.75,
            right: 0.25,
            c0: 0.08,
            c1: -0.03,
            c2: 0.02,
            c3: -0.01,
        };
        let link_span = exact_kernel::LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: -0.05,
            c1: 0.04,
            c2: -0.02,
            c3: 0.015,
        };
        let a = 0.3;
        let b = -0.7;
        let eps = 1e-6;

        let coeffs = |aa: f64, bb: f64| {
            let (h0, h1, h2, h3) = exact_global_cubic_from_local(LocalSpanCubic {
                left: score_span.left,
                right: score_span.right,
                c0: score_span.c0,
                c1: score_span.c1,
                c2: score_span.c2,
                c3: score_span.c3,
            });
            let (d0, d1, d2, d3) = exact_transformed_link_cubic(
                LocalSpanCubic {
                    left: link_span.left,
                    right: link_span.right,
                    c0: link_span.c0,
                    c1: link_span.c1,
                    c2: link_span.c2,
                    c3: link_span.c3,
                },
                aa,
                bb,
            );
            [
                aa + bb * h0 + d0,
                bb + bb * h1 + d1,
                bb * h2 + d2,
                bb * h3 + d3,
            ]
        };
        let (dc_da, dc_db) = exact_denested_cell_coefficient_partials(
            LocalSpanCubic {
                left: score_span.left,
                right: score_span.right,
                c0: score_span.c0,
                c1: score_span.c1,
                c2: score_span.c2,
                c3: score_span.c3,
            },
            LocalSpanCubic {
                left: link_span.left,
                right: link_span.right,
                c0: link_span.c0,
                c1: link_span.c1,
                c2: link_span.c2,
                c3: link_span.c3,
            },
            a,
            b,
        );
        let plus_a = coeffs(a + eps, b);
        let minus_a = coeffs(a - eps, b);
        let plus_b = coeffs(a, b + eps);
        let minus_b = coeffs(a, b - eps);
        for j in 0..4 {
            let fd_a = (plus_a[j] - minus_a[j]) / (2.0 * eps);
            let fd_b = (plus_b[j] - minus_b[j]) / (2.0 * eps);
            assert!(
                (dc_da[j] - fd_a).abs() < 1e-6,
                "dc/da mismatch at coefficient {j}: analytic={}, fd={fd_a}",
                dc_da[j]
            );
            assert!(
                (dc_db[j] - fd_b).abs() < 1e-6,
                "dc/db mismatch at coefficient {j}: analytic={}, fd={fd_b}",
                dc_db[j]
            );
        }
    }

    #[test]
    fn observed_denested_partials_include_third_a_derivative_for_piecewise_cubic_link() {
        let z = array![-0.8, 0.2, 1.1];
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_test_link_deviation_block_from_seed(
            &link_seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("link block");
        let beta_w = Array1::from_iter(
            (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
        );
        let family =
            BernoulliMarginalSlopeFamily {
                y: Arc::new(array![0.0, 1.0, 1.0]),
                weights: Arc::new(array![1.0, 0.7, 1.3]),
                z: Arc::new(z.clone()),
                gaussian_frailty_sd: None,
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: None,
                link_dev: Some(link_prepared.runtime.clone()),
            };

        let a = 0.35;
        let b = 0.6;
        let row = 1usize;
        let obs = family
            .observed_denested_cell_partials(row, a, b, None, Some(&beta_w))
            .expect("observed denested partials");
        let u_obs = a + b * z[row];
        let link_span = link_prepared
            .runtime
            .local_cubic_at(&beta_w, u_obs)
            .expect("local cubic at observed point");
        let expected_daaa = exact_kernel::denested_cell_third_partials(link_span).0;

        assert_eq!(obs.dc_daaa, expected_daaa);
        assert!(
            eval_coeff4_at(&obs.dc_daaa, z[row]).abs() > 1e-12,
            "piecewise-cubic link spans should contribute a third a-derivative"
        );
    }

    #[test]
    fn pooled_probit_baseline_matches_expanded_integer_weight_fit() {
        let y = array![0.0, 1.0, 0.0, 1.0];
        let z = array![-1.5, -0.2, 0.4, 1.4];
        let weights = array![25.0, 2.0, 1.0, 20.0];
        let weighted = pooled_probit_baseline(&y, &z, &weights).expect("weighted baseline");
        let unweighted =
            pooled_probit_baseline(&y, &z, &Array1::ones(y.len())).expect("unweighted baseline");
        let (y_expanded, z_expanded) = expand_integer_weight_rows(&y, &z, &weights);
        let expanded =
            pooled_probit_baseline(&y_expanded, &z_expanded, &Array1::ones(y_expanded.len()))
                .expect("expanded baseline");

        assert!(
            pair_distance(expanded, unweighted) > 1e-2,
            "test data should distinguish weighted from unweighted seeding"
        );
        assert!(
            pair_distance(weighted, expanded) < 1e-8,
            "weighted pilot baseline should match the expanded integer-weight fit"
        );
    }

    #[test]
    fn validate_spec_rejects_nonfinite_or_negative_weights() {
        let data = Array2::<f64>::zeros((3, 0));
        let y = array![0.0, 1.0, 0.0];
        let z = array![-1.0, 0.0, 1.0];

        let err = validate_spec(
            data.view(),
            &base_spec(y.clone(), array![1.0, f64::NAN, 1.0], z.clone()),
        )
        .expect_err("non-finite weights should be rejected");
        assert!(err.contains("finite non-negative weights"));

        let err = validate_spec(data.view(), &base_spec(y, array![1.0, -0.5, 1.0], z))
            .expect_err("negative weights should be rejected");
        assert!(err.contains("finite non-negative weights"));
    }

    #[test]
    fn validate_spec_rejects_nonfinite_z_values() {
        let data = Array2::<f64>::zeros((3, 0));
        let err = validate_spec(
            data.view(),
            &base_spec(
                array![0.0, 1.0, 0.0],
                array![1.0, 1.0, 1.0],
                array![-1.0, f64::INFINITY, 1.0],
            ),
        )
        .expect_err("non-finite z should be rejected");
        assert!(err.contains("finite z values"));
    }

    #[test]
    fn validate_spec_accepts_learnable_gaussian_shift_sigma() {
        let data = Array2::<f64>::zeros((3, 0));
        let mut spec = base_spec(
            array![0.0, 1.0, 0.0],
            array![1.0, 1.0, 1.0],
            array![-1.0, 0.0, 1.0],
        );
        spec.frailty = FrailtySpec::GaussianShift { sigma_fixed: None };

        validate_spec(data.view(), &spec).expect("learnable GaussianShift sigma should validate");
    }

    #[test]
    fn signed_probit_helpers_handle_nonfinite_boundaries_explicitly() {
        let (logcdf_pos, lambda_pos) = signed_probit_logcdf_and_mills_ratio(f64::INFINITY);
        assert_eq!(logcdf_pos, 0.0);
        assert_eq!(lambda_pos, 0.0);

        let (logcdf_neg, lambda_neg) = signed_probit_logcdf_and_mills_ratio(f64::NEG_INFINITY);
        assert_eq!(logcdf_neg, f64::NEG_INFINITY);
        assert_eq!(lambda_neg, f64::INFINITY);

        let (logcdf_nan, lambda_nan) = signed_probit_logcdf_and_mills_ratio(f64::NAN);
        assert!(logcdf_nan.is_nan());
        assert!(lambda_nan.is_nan());
    }

    #[test]
    fn signed_probit_exact_derivative_helper_rejects_invalid_nonfinite_margins() {
        assert_eq!(
            signed_probit_neglog_derivatives_up_to_fourth(f64::INFINITY, 2.5)
                .expect("+inf should use the zero tail"),
            (0.0, 0.0, 0.0, 0.0)
        );

        let neg_inf_err = signed_probit_neglog_derivatives_up_to_fourth(f64::NEG_INFINITY, 2.5)
            .expect_err("-inf should be rejected in the exact derivative path");
        assert!(neg_inf_err.contains("non-finite signed margin"));

        let nan_err = signed_probit_neglog_derivatives_up_to_fourth(f64::NAN, 2.5)
            .expect_err("NaN should be rejected in the exact derivative path");
        assert!(nan_err.contains("non-finite signed margin"));
    }

    #[test]
    fn unary_neglog_phi_preserves_negative_infinity_and_nan_boundaries() {
        assert_eq!(
            unary_derivatives_neglog_phi(f64::INFINITY, 1.75),
            [0.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            unary_derivatives_neglog_phi(f64::NEG_INFINITY, 1.75),
            [f64::INFINITY, f64::NEG_INFINITY, 1.75, 0.0, 0.0]
        );
        let nan_terms = unary_derivatives_neglog_phi(f64::NAN, 1.75);
        assert!(nan_terms.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn flexible_family_exposes_exact_outer_derivative_path() {
        let seed = array![-1.0, 0.0, 1.0];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 3,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let family =
            BernoulliMarginalSlopeFamily {
                y: Arc::new(array![0.0, 1.0, 0.0]),
                weights: Arc::new(Array1::ones(3)),
                z: Arc::new(seed.clone()),
                gaussian_frailty_sd: None,
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: Some(score_prepared.runtime),
                link_dev: None,
            };
        let specs = vec![
            dummy_blockspec(1, 3),
            dummy_blockspec(1, 3),
            dummy_blockspec(2, 3),
        ];
        assert_eq!(
            family.exact_outer_derivative_order(&specs, &BlockwiseFitOptions::default()),
            ExactOuterDerivativeOrder::Second
        );
        assert!(family.exact_newton_joint_psi_workspace_for_first_order_terms());
    }

    #[test]
    fn cost_gated_outer_order_ignores_inner_coefficient_dimension() {
        use crate::custom_family::cost_gated_outer_order;
        use crate::matrix::DesignMatrix;
        use ndarray::Array2;

        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::zeros((
            10, 500,
        ))));
        let specs: Vec<ParameterBlockSpec> = (0..2)
            .map(|i| ParameterBlockSpec {
                name: format!("block_{i}"),
                design: design.clone(),
                offset: ndarray::Array1::zeros(10),
                penalties: (0..10)
                    .map(|_| crate::custom_family::PenaltyMatrix::Dense(Array2::zeros((500, 500))))
                    .collect(),
                nullspace_dims: vec![0; 10],
                initial_log_lambdas: ndarray::Array1::zeros(10),
                initial_beta: None,
            })
            .collect();
        assert_eq!(
            cost_gated_outer_order(&specs),
            ExactOuterDerivativeOrder::Second
        );
    }

    #[test]
    fn rigid_fast_path_matches_loglik_finite_differences() {
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![1.0]),
            weights: Arc::new(array![1.2]),
            z: Arc::new(array![0.3]),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0
            ]])),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0
            ]])),
            score_warp: None,
            link_dev: None,
        };
        let states_at = |q: f64, g: f64| {
            vec![
                ParameterBlockState {
                    beta: array![q],
                    eta: array![q],
                },
                ParameterBlockState {
                    beta: array![g],
                    eta: array![g],
                },
            ]
        };
        let q = 0.4;
        let g = 0.7;
        let block_states = states_at(q, g);
        let eval = family
            .evaluate(&block_states)
            .expect("rigid family evaluation");
        let grad_q = match &eval.blockworking_sets[0] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            BlockWorkingSet::Diagonal { .. } => {
                panic!("expected exact-newton marginal block")
            }
        };
        let grad_g = match &eval.blockworking_sets[1] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            BlockWorkingSet::Diagonal { .. } => {
                panic!("expected exact-newton log-slope block")
            }
        };
        let hess_qq = match &eval.blockworking_sets[0] {
            BlockWorkingSet::ExactNewton { hessian, .. } => match hessian {
                SymmetricMatrix::Dense(h) => h[[0, 0]],
                _ => panic!("expected dense marginal Hessian"),
            },
            BlockWorkingSet::Diagonal { .. } => {
                panic!("expected exact-newton marginal block")
            }
        };
        let hess_gg = match &eval.blockworking_sets[1] {
            BlockWorkingSet::ExactNewton { hessian, .. } => match hessian {
                SymmetricMatrix::Dense(h) => h[[0, 0]],
                _ => panic!("expected dense log-slope Hessian"),
            },
            BlockWorkingSet::Diagonal { .. } => {
                panic!("expected exact-newton log-slope block")
            }
        };

        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("rigid exact eval cache");
        let row_ctx = family
            .build_row_exact_context(0, &block_states)
            .expect("rigid row context");
        let (_, primary_grad, primary_hess) = family
            .compute_row_primary_gradient_hessian(0, &block_states, &cache.primary, &row_ctx)
            .expect("rigid exact row derivatives");
        let expected_score_q =
            BernoulliMarginalSlopeFamily::exact_newton_score_component_from_objective_gradient(
                primary_grad[0],
            );
        let expected_score_g =
            BernoulliMarginalSlopeFamily::exact_newton_score_component_from_objective_gradient(
                primary_grad[1],
            );

        assert!(
            (grad_q - expected_score_q).abs() < 1e-10,
            "marginal gradient mismatch: fast={grad_q:.12e}, exact={expected_score_q:.12e}"
        );
        assert!(
            (grad_g - expected_score_g).abs() < 1e-10,
            "logslope gradient mismatch: fast={grad_g:.12e}, exact={expected_score_g:.12e}"
        );
        assert!(
            (hess_qq - primary_hess[[0, 0]]).abs() < 1e-10,
            "marginal Hessian mismatch: fast={hess_qq:.12e}, exact={:.12e}",
            primary_hess[[0, 0]]
        );
        assert!(
            (hess_gg - primary_hess[[1, 1]]).abs() < 1e-10,
            "logslope Hessian mismatch: fast={hess_gg:.12e}, exact={:.12e}",
            primary_hess[[1, 1]]
        );
    }

    /// Exercises the w-only (link_dev without score_warp) layout through the
    /// full gradient + Hessian path, verifying that:
    ///   (a) no index-out-of-bounds panic occurs,
    ///   (b) all outputs are finite,
    ///   (c) the Hessian is symmetric.
    ///
    /// This guards against arity bookkeeping bugs where the directional-jet or
    /// block-slice code assumes both h and w blocks are present.
    #[test]
    fn w_only_gradient_hessian_finite_and_symmetric() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_test_link_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build link deviation block");
        let link_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("link initial beta")
            .len();
        // Non-trivial link coefficients to exercise all jet branches.
        let beta_link = Array1::from_iter((0..link_dim).map(|idx| 0.05 * (idx as f64 + 1.0)));

        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: None,
            link_dev: Some(prepared.runtime.clone()),
        };

        // Three blocks: marginal (dim 0), logslope (dim 0), link_dev.
        // eta is irrelevant for zero-column designs; use zeros.
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(beta_link.clone(), seed.len()),
        ];

        let slices = block_slices(&family);
        assert!(slices.h.is_none(), "score-warp absent → no h slice");
        let primary = primary_slices(&slices);
        assert!(primary.h.is_none(), "primary h absent");
        assert_eq!(primary.total, 2 + link_dim);

        // Exercise every row — different z values exercise different link
        // regimes (negative tail, near zero, positive tail).
        for row in 0..seed.len() {
            let row_ctx = family
                .build_row_exact_context(row, &block_states)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));

            let (_, grad, hess) = family
                .compute_row_primary_gradient_hessian(row, &block_states, &primary, &row_ctx)
                .unwrap_or_else(|e| {
                    panic!("row {row}: compute_row_primary_gradient_hessian failed: {e}")
                });

            assert_eq!(
                grad.len(),
                primary.total,
                "row {row}: gradient length mismatch"
            );
            assert_eq!(
                hess.dim(),
                (primary.total, primary.total),
                "row {row}: hessian shape mismatch"
            );
            assert!(
                grad.iter().all(|v| v.is_finite()),
                "row {row}: non-finite gradient entry: {grad:?}"
            );
            assert!(
                hess.iter().all(|v| v.is_finite()),
                "row {row}: non-finite hessian entry"
            );

            // Symmetry check.
            for a in 0..primary.total {
                for b in 0..a {
                    let diff = (hess[[a, b]] - hess[[b, a]]).abs();
                    assert!(
                        diff < 1e-10,
                        "row {row}: hessian asymmetry at ({a},{b}): \
                         H[{a},{b}]={:.6e} vs H[{b},{a}]={:.6e}, diff={diff:.3e}",
                        hess[[a, b]],
                        hess[[b, a]]
                    );
                }
            }
        }
    }

    #[test]
    fn h_only_gradient_hessian_finite_and_symmetric() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score-warp block");
        let score_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("score-warp initial beta")
            .len();
        let beta_score = Array1::from_iter((0..score_dim).map(|idx| 0.04 * (idx as f64 + 1.0)));

        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: Some(prepared.runtime.clone()),
            link_dev: None,
        };

        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(beta_score, seed.len()),
        ];

        let slices = block_slices(&family);
        assert!(slices.w.is_none(), "link-dev absent → no w slice");
        let primary = primary_slices(&slices);
        assert!(primary.w.is_none(), "primary w absent");
        assert_eq!(primary.total, 2 + score_dim);

        for row in 0..seed.len() {
            let row_ctx = family
                .build_row_exact_context(row, &block_states)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));

            let (_, grad, hess) = family
                .compute_row_primary_gradient_hessian(row, &block_states, &primary, &row_ctx)
                .unwrap_or_else(|e| {
                    panic!("row {row}: compute_row_primary_gradient_hessian failed: {e}")
                });

            assert_eq!(
                grad.len(),
                primary.total,
                "row {row}: gradient length mismatch"
            );
            assert_eq!(
                hess.dim(),
                (primary.total, primary.total),
                "row {row}: hessian shape mismatch"
            );
            assert!(
                grad.iter().all(|v| v.is_finite()),
                "row {row}: non-finite gradient entry: {grad:?}"
            );
            assert!(
                hess.iter().all(|v| v.is_finite()),
                "row {row}: non-finite hessian entry"
            );

            for a in 0..primary.total {
                for b in 0..a {
                    let diff = (hess[[a, b]] - hess[[b, a]]).abs();
                    assert!(
                        diff < 1e-10,
                        "row {row}: hessian asymmetry at ({a},{b}): \
                         H[{a},{b}]={:.6e} vs H[{b},{a}]={:.6e}, diff={diff:.3e}",
                        hess[[a, b]],
                        hess[[b, a]]
                    );
                }
            }
        }
    }

    #[test]
    fn w_only_exact_outer_directional_derivatives_are_present_and_finite() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_test_link_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build link deviation block");
        let link_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("link initial beta")
            .len();
        let beta_link = Array1::from_iter((0..link_dim).map(|idx| 0.05 * (idx as f64 + 1.0)));

        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: None,
            link_dev: Some(prepared.runtime.clone()),
        };
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(beta_link, seed.len()),
        ];

        let slices = block_slices(&family);
        let total = slices.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        let w_range = slices.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.15;
        if w_range.len() > 1 {
            dir_u[w_range.start + 1] = -0.07;
        }

        dir_v[w_range.start] = 0.09;
        if w_range.len() > 1 {
            dir_v[w_range.start + 1] = 0.03;
        }

        let third = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_u)
            .expect("w-only third directional derivative")
            .expect("w-only third directional derivative matrix");
        let fourth = family
            .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_u, &dir_v)
            .expect("w-only fourth directional derivative")
            .expect("w-only fourth directional derivative matrix");

        assert_eq!(third.dim(), (total, total));
        assert_eq!(fourth.dim(), (total, total));
        assert!(third.iter().all(|value| value.is_finite()));
        assert!(fourth.iter().all(|value| value.is_finite()));
        let max_abs_third = third
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let max_abs_fourth = fourth
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            max_abs_third > 1e-10,
            "expected nonzero w-only third directional derivative"
        );
        assert!(
            max_abs_fourth > 1e-10,
            "expected nonzero w-only fourth directional derivative"
        );

        for i in 0..total {
            for j in 0..i {
                assert!((third[[i, j]] - third[[j, i]]).abs() < 1e-8);
                assert!((fourth[[i, j]] - fourth[[j, i]]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn h_only_exact_outer_directional_derivatives_are_present_and_finite() {
        // ROOT CAUSE (pre-4250aa07 vs current).  The pre-refactor `block_slices`
        // sized its marginal/logslope slices from `states[block_idx].beta.len()`
        // (1 each from `dummy_block_state(array![0.0], ...)`), giving a 3-slot
        // block-space {marginal, logslope, h...} even though the design matrices
        // were zero-width.  After the refactor, `block_slices` sizes slices from
        // `design.ncols()`, so zero-width designs collapse marginal and logslope
        // to empty ranges.  Any write to `dir_u[slices.marginal.start]` then
        // aliases `dir_u[slices.logslope.start]` and `dir_u[h_range.start]`,
        // which is why the refactor dropped those writes — but that also
        // dropped the only path by which `exact_newton_joint_hessian_directional_derivative`
        // (which maps block-space direction → primary-space direction via
        // `marginal_design·β_marginal` → `row_dir[primary.q]` and
        // `logslope_design·β_logslope` → `row_dir[primary.logslope]`) could
        // inject nonzero q / logslope components.
        //
        // At the current state (b ≡ block_states[1].eta[row] = 0 and pure
        // h-only block-space direction), the third directional derivative is
        // structurally zero:
        //
        //   In the flex score-warp path, the h-block contribution to the
        //   observed cell coefficient is
        //       coeff_u[idx_h] = s · score_basis_cell_coefficients(basis_span, b)
        //                      = s · [b·h0, b·h1, b·h2, b·h3]   (cubic_cell_kernel.rs:815)
        //   At b = 0 this vanishes, and the only other h-index term —
        //   coeff_bu[idx_h] = s · [h0, h1, h2, h3] — is reached via
        //   `param_directional_from_b_family` only when `dir[primary.logslope] ≠ 0`
        //   (bernoulli_marginal_slope.rs:1799-1815).  With row_dir[q] and
        //   row_dir[logslope] both zero, every directional contraction
        //   (coeff_dir, coeff_a_dir, coeff_aa_dir, coeff_u_dir[u], coeff_au_dir[u],
        //   pair_directional_from_bb_family(...)) collapses to zero.
        //   Therefore `max_abs_third > 1e-10` is analytically impossible.
        //
        // PRINCIPLED FIX.  Restore the block-space geometry the test is
        // actually designed to exercise: give marginal_design and logslope_design
        // single columns of ones (the canonical "scalar" parameterisation used
        // by the sigma FD test at rs:13436), set block_states so row-wise
        // q_internal and b are at typical nondegenerate values, and populate
        // dir_u / dir_v on all three blocks.  The test name was misleading:
        // "h-only" originally referred to the score-warp (h) BLOCK being the
        // only flex block configured (link_dev: None) — not to the direction
        // being supported solely on h.  This fix preserves that original
        // semantic and exercises the block→primary direction map end-to-end.
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score-warp block");
        let score_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("score-warp initial beta")
            .len();
        let beta_score = Array1::from_iter((0..score_dim).map(|idx| 0.04 * (idx as f64 + 1.0)));
        let scalar_design = || {
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::from_elem(
                (seed.len(), 1),
                1.0,
            )))
        };

        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: scalar_design(),
            logslope_design: scalar_design(),
            score_warp: Some(prepared.runtime.clone()),
            link_dev: None,
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.25],
                eta: Array1::from_elem(seed.len(), 0.25),
            },
            ParameterBlockState {
                beta: array![0.15],
                eta: Array1::from_elem(seed.len(), 0.15),
            },
            ParameterBlockState {
                beta: beta_score,
                eta: Array1::zeros(seed.len()),
            },
        ];

        let slices = block_slices(&family);
        let total = slices.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[slices.marginal.start] = -0.35;
        dir_u[slices.logslope.start] = 0.28;
        let h_range = slices.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.12;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.06;
        }

        dir_v[slices.marginal.start] = 0.18;
        dir_v[slices.logslope.start] = -0.22;
        dir_v[h_range.start] = 0.07;
        if h_range.len() > 1 {
            dir_v[h_range.start + 1] = 0.05;
        }

        let third = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_u)
            .expect("h-only third directional derivative")
            .expect("h-only third directional derivative matrix");
        let fourth = family
            .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_u, &dir_v)
            .expect("h-only fourth directional derivative")
            .expect("h-only fourth directional derivative matrix");

        assert_eq!(third.dim(), (total, total));
        assert_eq!(fourth.dim(), (total, total));
        assert!(third.iter().all(|value| value.is_finite()));
        assert!(fourth.iter().all(|value| value.is_finite()));
        let max_abs_third = third
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let max_abs_fourth = fourth
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            max_abs_third > 1e-10,
            "expected nonzero h-only third directional derivative"
        );
        assert!(
            max_abs_fourth > 1e-10,
            "expected nonzero h-only fourth directional derivative"
        );

        for i in 0..total {
            for j in 0..i {
                assert!((third[[i, j]] - third[[j, i]]).abs() < 1e-8);
                assert!((fourth[[i, j]] - fourth[[j, i]]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn h_only_row_primary_higher_order_contractions_are_finite_and_symmetric() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score-warp block");
        let score_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("score-warp initial beta")
            .len();
        let beta_score = Array1::from_iter((0..score_dim).map(|idx| 0.04 * (idx as f64 + 1.0)));

        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: Some(prepared.runtime.clone()),
            link_dev: None,
        };

        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(beta_score, seed.len()),
        ];

        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("exact eval cache");
        let total = cache.primary.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[cache.primary.q] = -0.35;
        dir_u[cache.primary.logslope] = 0.28;
        let h_range = cache.primary.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.12;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.06;
        }

        dir_v[cache.primary.q] = 0.18;
        dir_v[cache.primary.logslope] = -0.22;
        dir_v[h_range.start] = 0.07;
        if h_range.len() > 1 {
            dir_v[h_range.start + 1] = 0.05;
        }

        let mut max_abs_third = 0.0_f64;
        let mut max_abs_fourth = 0.0_f64;
        for row in 0..seed.len() {
            let row_ctx = family
                .build_row_exact_context(row, &block_states)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
            let third = family
                .row_primary_third_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_u,
                )
                .unwrap_or_else(|e| {
                    panic!("row {row}: row_primary_third_contracted_recompute failed: {e}")
                });
            let fourth = family
                .row_primary_fourth_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_u,
                    &dir_v,
                )
                .unwrap_or_else(|e| {
                    panic!("row {row}: row_primary_fourth_contracted_recompute failed: {e}")
                });

            assert_eq!(third.dim(), (total, total));
            assert_eq!(fourth.dim(), (total, total));
            assert!(third.iter().all(|value| value.is_finite()));
            assert!(fourth.iter().all(|value| value.is_finite()));
            max_abs_third = max_abs_third.max(
                third
                    .iter()
                    .fold(0.0_f64, |acc, value| acc.max(value.abs())),
            );
            max_abs_fourth = max_abs_fourth.max(
                fourth
                    .iter()
                    .fold(0.0_f64, |acc, value| acc.max(value.abs())),
            );

            for i in 0..total {
                for j in 0..i {
                    assert!((third[[i, j]] - third[[j, i]]).abs() < 1e-8);
                    assert!((fourth[[i, j]] - fourth[[j, i]]).abs() < 1e-8);
                }
            }
        }
        assert!(
            max_abs_third > 1e-10,
            "expected nonzero h-only third contraction"
        );
        assert!(
            max_abs_fourth > 1e-10,
            "expected nonzero h-only fourth contraction"
        );
    }

    #[test]
    fn w_only_row_primary_higher_order_contractions_are_finite_and_symmetric() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_test_link_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build link deviation block");
        let link_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("link initial beta")
            .len();
        let beta_link = Array1::from_iter((0..link_dim).map(|idx| 0.05 * (idx as f64 + 1.0)));

        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: None,
            link_dev: Some(prepared.runtime.clone()),
        };

        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(beta_link, seed.len()),
        ];

        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("exact eval cache");
        let total = cache.primary.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[cache.primary.q] = 0.4;
        dir_u[cache.primary.logslope] = -0.3;
        let w_range = cache.primary.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.15;
        if w_range.len() > 1 {
            dir_u[w_range.start + 1] = -0.07;
        }

        dir_v[cache.primary.q] = -0.2;
        dir_v[cache.primary.logslope] = 0.25;
        dir_v[w_range.start] = 0.09;
        if w_range.len() > 1 {
            dir_v[w_range.start + 1] = 0.03;
        }

        let mut max_abs_third = 0.0_f64;
        let mut max_abs_fourth = 0.0_f64;
        for row in 0..seed.len() {
            let row_ctx = family
                .build_row_exact_context(row, &block_states)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
            let third = family
                .row_primary_third_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_u,
                )
                .unwrap_or_else(|e| {
                    panic!("row {row}: row_primary_third_contracted_recompute failed: {e}")
                });
            let fourth = family
                .row_primary_fourth_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_u,
                    &dir_v,
                )
                .unwrap_or_else(|e| {
                    panic!("row {row}: row_primary_fourth_contracted_recompute failed: {e}")
                });

            assert_eq!(third.dim(), (total, total));
            assert_eq!(fourth.dim(), (total, total));
            assert!(third.iter().all(|value| value.is_finite()));
            assert!(fourth.iter().all(|value| value.is_finite()));
            max_abs_third = max_abs_third.max(
                third
                    .iter()
                    .fold(0.0_f64, |acc, value| acc.max(value.abs())),
            );
            max_abs_fourth = max_abs_fourth.max(
                fourth
                    .iter()
                    .fold(0.0_f64, |acc, value| acc.max(value.abs())),
            );

            for i in 0..total {
                for j in 0..i {
                    assert!((third[[i, j]] - third[[j, i]]).abs() < 1e-8);
                    assert!((fourth[[i, j]] - fourth[[j, i]]).abs() < 1e-8);
                }
            }
        }
        assert!(
            max_abs_third > 1e-10,
            "expected nonzero w-only third contraction"
        );
        assert!(
            max_abs_fourth > 1e-10,
            "expected nonzero w-only fourth contraction"
        );
    }

    #[test]
    fn dual_flex_row_primary_higher_order_contractions_are_finite_and_symmetric() {
        let z = array![-0.8, 0.2, 1.1];
        let y = array![0.0, 1.0, 1.0];
        let weights = array![1.0, 0.7, 1.3];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_test_link_deviation_block_from_seed(
            &link_seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("link block");
        let family =
            BernoulliMarginalSlopeFamily {
                y: Arc::new(y.clone()),
                weights: Arc::new(weights.clone()),
                z: Arc::new(z.clone()),
                gaussian_frailty_sd: None,
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: Some(score_prepared.runtime.clone()),
                link_dev: Some(link_prepared.runtime.clone()),
            };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.25],
                eta: Array1::from_elem(z.len(), 0.25),
            },
            ParameterBlockState {
                beta: array![0.6],
                eta: Array1::from_elem(z.len(), 0.6),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
        ];

        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("exact eval cache");
        let total = cache.primary.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[cache.primary.q] = 0.7;
        dir_u[cache.primary.logslope] = -0.2;
        let h_range = cache.primary.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.1;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.05;
        }
        let w_range = cache.primary.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.08;

        dir_v[cache.primary.q] = -0.4;
        dir_v[cache.primary.logslope] = 0.3;
        dir_v[h_range.start] = -0.03;
        dir_v[w_range.start] = 0.06;
        if w_range.len() > 1 {
            dir_v[w_range.start + 1] = -0.02;
        }

        let mut max_abs_third = 0.0_f64;
        let mut max_abs_fourth = 0.0_f64;
        for row in 0..z.len() {
            let row_ctx = family
                .build_row_exact_context(row, &block_states)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
            let third = family
                .row_primary_third_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_u,
                )
                .unwrap_or_else(|e| {
                    panic!("row {row}: row_primary_third_contracted_recompute failed: {e}")
                });
            let fourth = family
                .row_primary_fourth_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_u,
                    &dir_v,
                )
                .unwrap_or_else(|e| {
                    panic!("row {row}: row_primary_fourth_contracted_recompute failed: {e}")
                });

            assert_eq!(third.dim(), (total, total));
            assert_eq!(fourth.dim(), (total, total));
            assert!(third.iter().all(|value| value.is_finite()));
            assert!(fourth.iter().all(|value| value.is_finite()));
            max_abs_third = max_abs_third.max(
                third
                    .iter()
                    .fold(0.0_f64, |acc, value| acc.max(value.abs())),
            );
            max_abs_fourth = max_abs_fourth.max(
                fourth
                    .iter()
                    .fold(0.0_f64, |acc, value| acc.max(value.abs())),
            );

            for i in 0..total {
                for j in 0..i {
                    assert!((third[[i, j]] - third[[j, i]]).abs() < 1e-8);
                    assert!((fourth[[i, j]] - fourth[[j, i]]).abs() < 1e-8);
                }
            }
        }
        assert!(
            max_abs_third > 1e-10,
            "expected nonzero dual-flex third contraction"
        );
        assert!(
            max_abs_fourth > 1e-10,
            "expected nonzero dual-flex fourth contraction"
        );
    }

    #[test]
    fn dual_flex_row_primary_higher_order_zero_direction_returns_zero() {
        let z = array![-0.8, 0.2, 1.1];
        let y = array![0.0, 1.0, 1.0];
        let weights = array![1.0, 0.7, 1.3];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_test_link_deviation_block_from_seed(
            &link_seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("link block");
        let family =
            BernoulliMarginalSlopeFamily {
                y: Arc::new(y.clone()),
                weights: Arc::new(weights.clone()),
                z: Arc::new(z.clone()),
                gaussian_frailty_sd: None,
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: Some(score_prepared.runtime.clone()),
                link_dev: Some(link_prepared.runtime.clone()),
            };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.25],
                eta: Array1::from_elem(z.len(), 0.25),
            },
            ParameterBlockState {
                beta: array![0.6],
                eta: Array1::from_elem(z.len(), 0.6),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
        ];

        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("exact eval cache");
        let zero = Array1::<f64>::zeros(cache.primary.total);
        for row in 0..z.len() {
            let row_ctx = family
                .build_row_exact_context(row, &block_states)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
            let third = family
                .row_primary_third_contracted_recompute(row, &block_states, &cache, &row_ctx, &zero)
                .unwrap_or_else(|e| {
                    panic!("row {row}: row_primary_third_contracted_recompute failed: {e}")
                });
            let fourth = family
                .row_primary_fourth_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &zero,
                    &zero,
                )
                .unwrap_or_else(|e| {
                    panic!("row {row}: row_primary_fourth_contracted_recompute failed: {e}")
                });

            assert!(
                third.iter().all(|value| value.abs() <= 0.0),
                "row {row}: expected zero third contraction for zero direction"
            );
            assert!(
                fourth.iter().all(|value| value.abs() <= 0.0),
                "row {row}: expected zero fourth contraction for zero directions"
            );
        }
    }

    #[test]
    fn h_only_row_primary_higher_order_zero_direction_returns_zero() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score-warp block");
        let score_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("score-warp initial beta")
            .len();
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(
                Array1::from_iter((0..score_dim).map(|idx| 0.04 * (idx as f64 + 1.0))),
                seed.len(),
            ),
        ];
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: Some(prepared.runtime.clone()),
            link_dev: None,
        };
        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("exact eval cache");
        let zero = Array1::<f64>::zeros(cache.primary.total);
        for row in 0..seed.len() {
            let row_ctx = family
                .build_row_exact_context(row, &block_states)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
            let third = family
                .row_primary_third_contracted_recompute(row, &block_states, &cache, &row_ctx, &zero)
                .unwrap_or_else(|e| {
                    panic!("row {row}: row_primary_third_contracted_recompute failed: {e}")
                });
            let fourth = family
                .row_primary_fourth_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &zero,
                    &zero,
                )
                .unwrap_or_else(|e| {
                    panic!("row {row}: row_primary_fourth_contracted_recompute failed: {e}")
                });

            assert!(
                third.iter().all(|value| value.abs() <= 0.0),
                "row {row}: expected zero h-only third contraction for zero direction"
            );
            assert!(
                fourth.iter().all(|value| value.abs() <= 0.0),
                "row {row}: expected zero h-only fourth contraction for zero directions"
            );
        }
    }

    #[test]
    fn w_only_row_primary_higher_order_zero_direction_returns_zero() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_test_link_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build link deviation block");
        let link_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("link initial beta")
            .len();
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(
                Array1::from_iter((0..link_dim).map(|idx| 0.05 * (idx as f64 + 1.0))),
                seed.len(),
            ),
        ];
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: None,
            link_dev: Some(prepared.runtime.clone()),
        };
        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("exact eval cache");
        let zero = Array1::<f64>::zeros(cache.primary.total);
        for row in 0..seed.len() {
            let row_ctx = family
                .build_row_exact_context(row, &block_states)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
            let third = family
                .row_primary_third_contracted_recompute(row, &block_states, &cache, &row_ctx, &zero)
                .unwrap_or_else(|e| {
                    panic!("row {row}: row_primary_third_contracted_recompute failed: {e}")
                });
            let fourth = family
                .row_primary_fourth_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &zero,
                    &zero,
                )
                .unwrap_or_else(|e| {
                    panic!("row {row}: row_primary_fourth_contracted_recompute failed: {e}")
                });

            assert!(
                third.iter().all(|value| value.abs() <= 0.0),
                "row {row}: expected zero w-only third contraction for zero direction"
            );
            assert!(
                fourth.iter().all(|value| value.abs() <= 0.0),
                "row {row}: expected zero w-only fourth contraction for zero directions"
            );
        }
    }

    #[test]
    fn dual_flex_exact_outer_zero_direction_returns_zero() {
        let z = array![-0.8, 0.2, 1.1];
        let y = array![0.0, 1.0, 1.0];
        let weights = array![1.0, 0.7, 1.3];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_test_link_deviation_block_from_seed(
            &link_seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("link block");
        let family =
            BernoulliMarginalSlopeFamily {
                y: Arc::new(y.clone()),
                weights: Arc::new(weights.clone()),
                z: Arc::new(z.clone()),
                gaussian_frailty_sd: None,
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: Some(score_prepared.runtime.clone()),
                link_dev: Some(link_prepared.runtime.clone()),
            };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.25],
                eta: Array1::from_elem(z.len(), 0.25),
            },
            ParameterBlockState {
                beta: array![0.6],
                eta: Array1::from_elem(z.len(), 0.6),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
        ];

        let slices = block_slices(&family);
        let zero = Array1::<f64>::zeros(slices.total);
        let third = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &zero)
            .expect("dual-flex third directional derivative")
            .expect("dual-flex third directional derivative matrix");
        let fourth = family
            .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &zero, &zero)
            .expect("dual-flex fourth directional derivative")
            .expect("dual-flex fourth directional derivative matrix");

        assert!(
            third.iter().all(|value| value.abs() <= 0.0),
            "expected zero dual-flex third directional derivative for zero direction"
        );
        assert!(
            fourth.iter().all(|value| value.abs() <= 0.0),
            "expected zero dual-flex fourth directional derivative for zero directions"
        );
    }

    #[test]
    fn dual_flex_exact_outer_fourth_direction_swap_is_symmetric() {
        let z = array![-0.8, 0.2, 1.1];
        let y = array![0.0, 1.0, 1.0];
        let weights = array![1.0, 0.7, 1.3];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_test_link_deviation_block_from_seed(
            &link_seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("link block");
        let family =
            BernoulliMarginalSlopeFamily {
                y: Arc::new(y.clone()),
                weights: Arc::new(weights.clone()),
                z: Arc::new(z.clone()),
                gaussian_frailty_sd: None,
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: Some(score_prepared.runtime.clone()),
                link_dev: Some(link_prepared.runtime.clone()),
            };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.25],
                eta: Array1::from_elem(z.len(), 0.25),
            },
            ParameterBlockState {
                beta: array![0.6],
                eta: Array1::from_elem(z.len(), 0.6),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
        ];

        let slices = block_slices(&family);
        let total = slices.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[slices.marginal.start] = 0.7;
        dir_u[slices.logslope.start] = -0.2;
        let h_range = slices.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.1;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.05;
        }
        let w_range = slices.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.08;

        dir_v[slices.marginal.start] = -0.4;
        dir_v[slices.logslope.start] = 0.3;
        dir_v[h_range.start] = -0.03;
        dir_v[w_range.start] = 0.06;
        if w_range.len() > 1 {
            dir_v[w_range.start + 1] = -0.02;
        }

        let forward = family
            .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_u, &dir_v)
            .expect("dual-flex fourth directional derivative")
            .expect("dual-flex fourth directional derivative matrix");
        let swapped = family
            .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_v, &dir_u)
            .expect("dual-flex swapped fourth directional derivative")
            .expect("dual-flex swapped fourth directional derivative matrix");

        assert_eq!(forward.dim(), (total, total));
        assert_eq!(swapped.dim(), (total, total));
        for i in 0..total {
            for j in 0..total {
                assert!(
                    (forward[[i, j]] - swapped[[i, j]]).abs() < 1e-8,
                    "fourth directional derivative should be symmetric in direction arguments at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn dual_flex_row_primary_fourth_direction_swap_is_symmetric() {
        let z = array![-0.8, 0.2, 1.1];
        let y = array![0.0, 1.0, 1.0];
        let weights = array![1.0, 0.7, 1.3];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_test_link_deviation_block_from_seed(
            &link_seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("link block");
        let family =
            BernoulliMarginalSlopeFamily {
                y: Arc::new(y.clone()),
                weights: Arc::new(weights.clone()),
                z: Arc::new(z.clone()),
                gaussian_frailty_sd: None,
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: Some(score_prepared.runtime.clone()),
                link_dev: Some(link_prepared.runtime.clone()),
            };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.25],
                eta: Array1::from_elem(z.len(), 0.25),
            },
            ParameterBlockState {
                beta: array![0.6],
                eta: Array1::from_elem(z.len(), 0.6),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
        ];

        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("exact eval cache");
        let total = cache.primary.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[cache.primary.q] = 0.7;
        dir_u[cache.primary.logslope] = -0.2;
        let h_range = cache.primary.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.1;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.05;
        }
        let w_range = cache.primary.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.08;

        dir_v[cache.primary.q] = -0.4;
        dir_v[cache.primary.logslope] = 0.3;
        dir_v[h_range.start] = -0.03;
        dir_v[w_range.start] = 0.06;
        if w_range.len() > 1 {
            dir_v[w_range.start + 1] = -0.02;
        }

        for row in 0..z.len() {
            let row_ctx = family
                .build_row_exact_context(row, &block_states)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
            let forward = family
                .row_primary_fourth_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_u,
                    &dir_v,
                )
                .unwrap_or_else(|e| panic!("row {row}: forward fourth contraction failed: {e}"));
            let swapped = family
                .row_primary_fourth_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_v,
                    &dir_u,
                )
                .unwrap_or_else(|e| panic!("row {row}: swapped fourth contraction failed: {e}"));

            assert_eq!(forward.dim(), (total, total));
            assert_eq!(swapped.dim(), (total, total));
            for i in 0..total {
                for j in 0..total {
                    assert!(
                        (forward[[i, j]] - swapped[[i, j]]).abs() < 1e-8,
                        "row {row}: fourth contraction should be symmetric in direction arguments at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn dual_flex_row_primary_higher_order_direction_sign_rules_hold() {
        let z = array![-0.8, 0.2, 1.1];
        let y = array![0.0, 1.0, 1.0];
        let weights = array![1.0, 0.7, 1.3];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_test_link_deviation_block_from_seed(
            &link_seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("link block");
        let family =
            BernoulliMarginalSlopeFamily {
                y: Arc::new(y.clone()),
                weights: Arc::new(weights.clone()),
                z: Arc::new(z.clone()),
                gaussian_frailty_sd: None,
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: Some(score_prepared.runtime.clone()),
                link_dev: Some(link_prepared.runtime.clone()),
            };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.25],
                eta: Array1::from_elem(z.len(), 0.25),
            },
            ParameterBlockState {
                beta: array![0.6],
                eta: Array1::from_elem(z.len(), 0.6),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
        ];

        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("exact eval cache");
        let total = cache.primary.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[cache.primary.q] = 0.7;
        dir_u[cache.primary.logslope] = -0.2;
        let h_range = cache.primary.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.1;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.05;
        }
        let w_range = cache.primary.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.08;

        dir_v[cache.primary.q] = -0.4;
        dir_v[cache.primary.logslope] = 0.3;
        dir_v[h_range.start] = -0.03;
        dir_v[w_range.start] = 0.06;
        if w_range.len() > 1 {
            dir_v[w_range.start + 1] = -0.02;
        }

        let neg_dir_u = dir_u.mapv(|value| -value);
        for row in 0..z.len() {
            let row_ctx = family
                .build_row_exact_context(row, &block_states)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
            let third = family
                .row_primary_third_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_u,
                )
                .unwrap_or_else(|e| panic!("row {row}: third contraction failed: {e}"));
            let third_neg = family
                .row_primary_third_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &neg_dir_u,
                )
                .unwrap_or_else(|e| panic!("row {row}: negated third contraction failed: {e}"));
            let fourth = family
                .row_primary_fourth_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_u,
                    &dir_v,
                )
                .unwrap_or_else(|e| panic!("row {row}: fourth contraction failed: {e}"));
            let fourth_neg_u = family
                .row_primary_fourth_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &neg_dir_u,
                    &dir_v,
                )
                .unwrap_or_else(|e| panic!("row {row}: negated-u fourth contraction failed: {e}"));

            for i in 0..total {
                for j in 0..total {
                    assert!(
                        (third_neg[[i, j]] + third[[i, j]]).abs() < 1e-8,
                        "row {row}: third contraction should be odd in its direction at ({i},{j})"
                    );
                    assert!(
                        (fourth_neg_u[[i, j]] + fourth[[i, j]]).abs() < 1e-8,
                        "row {row}: fourth contraction should be linear in dir_u sign at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn h_only_row_primary_fourth_direction_swap_is_symmetric() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score-warp block");
        let score_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("score-warp initial beta")
            .len();
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(
                Array1::from_iter((0..score_dim).map(|idx| 0.04 * (idx as f64 + 1.0))),
                seed.len(),
            ),
        ];
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: Some(prepared.runtime.clone()),
            link_dev: None,
        };
        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("exact eval cache");
        let total = cache.primary.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[cache.primary.q] = -0.35;
        dir_u[cache.primary.logslope] = 0.28;
        let h_range = cache.primary.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.12;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.06;
        }

        dir_v[cache.primary.q] = 0.18;
        dir_v[cache.primary.logslope] = -0.22;
        dir_v[h_range.start] = 0.07;
        if h_range.len() > 1 {
            dir_v[h_range.start + 1] = 0.05;
        }

        for row in 0..seed.len() {
            let row_ctx = family
                .build_row_exact_context(row, &block_states)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
            let forward = family
                .row_primary_fourth_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_u,
                    &dir_v,
                )
                .unwrap_or_else(|e| panic!("row {row}: forward fourth contraction failed: {e}"));
            let swapped = family
                .row_primary_fourth_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_v,
                    &dir_u,
                )
                .unwrap_or_else(|e| panic!("row {row}: swapped fourth contraction failed: {e}"));

            for i in 0..total {
                for j in 0..total {
                    assert!(
                        (forward[[i, j]] - swapped[[i, j]]).abs() < 1e-8,
                        "row {row}: h-only fourth contraction should be symmetric in direction arguments at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn w_only_row_primary_fourth_direction_swap_is_symmetric() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_test_link_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build link deviation block");
        let link_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("link initial beta")
            .len();
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(
                Array1::from_iter((0..link_dim).map(|idx| 0.05 * (idx as f64 + 1.0))),
                seed.len(),
            ),
        ];
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: None,
            link_dev: Some(prepared.runtime.clone()),
        };
        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("exact eval cache");
        let total = cache.primary.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[cache.primary.q] = 0.4;
        dir_u[cache.primary.logslope] = -0.3;
        let w_range = cache.primary.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.15;
        if w_range.len() > 1 {
            dir_u[w_range.start + 1] = -0.07;
        }

        dir_v[cache.primary.q] = -0.2;
        dir_v[cache.primary.logslope] = 0.25;
        dir_v[w_range.start] = 0.09;
        if w_range.len() > 1 {
            dir_v[w_range.start + 1] = 0.03;
        }

        for row in 0..seed.len() {
            let row_ctx = family
                .build_row_exact_context(row, &block_states)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
            let forward = family
                .row_primary_fourth_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_u,
                    &dir_v,
                )
                .unwrap_or_else(|e| panic!("row {row}: forward fourth contraction failed: {e}"));
            let swapped = family
                .row_primary_fourth_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_v,
                    &dir_u,
                )
                .unwrap_or_else(|e| panic!("row {row}: swapped fourth contraction failed: {e}"));

            for i in 0..total {
                for j in 0..total {
                    assert!(
                        (forward[[i, j]] - swapped[[i, j]]).abs() < 1e-8,
                        "row {row}: w-only fourth contraction should be symmetric in direction arguments at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn h_only_row_primary_higher_order_direction_sign_rules_hold() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score-warp block");
        let score_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("score-warp initial beta")
            .len();
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(
                Array1::from_iter((0..score_dim).map(|idx| 0.04 * (idx as f64 + 1.0))),
                seed.len(),
            ),
        ];
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: Some(prepared.runtime.clone()),
            link_dev: None,
        };
        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("exact eval cache");
        let total = cache.primary.total;
        let mut dir = Array1::<f64>::zeros(total);
        dir[cache.primary.q] = -0.35;
        dir[cache.primary.logslope] = 0.28;
        let h_range = cache.primary.h.as_ref().expect("h slice");
        dir[h_range.start] = 0.12;
        if h_range.len() > 1 {
            dir[h_range.start + 1] = -0.06;
        }
        let neg_dir = dir.mapv(|value| -value);

        for row in 0..seed.len() {
            let row_ctx = family
                .build_row_exact_context(row, &block_states)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
            let third = family
                .row_primary_third_contracted_recompute(row, &block_states, &cache, &row_ctx, &dir)
                .unwrap_or_else(|e| panic!("row {row}: third contraction failed: {e}"));
            let third_neg = family
                .row_primary_third_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &neg_dir,
                )
                .unwrap_or_else(|e| panic!("row {row}: negated third contraction failed: {e}"));
            let fourth = family
                .row_primary_fourth_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir,
                    &dir,
                )
                .unwrap_or_else(|e| panic!("row {row}: fourth contraction failed: {e}"));
            let fourth_neg = family
                .row_primary_fourth_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &neg_dir,
                    &neg_dir,
                )
                .unwrap_or_else(|e| {
                    panic!("row {row}: doubly-negated fourth contraction failed: {e}")
                });

            for i in 0..total {
                for j in 0..total {
                    assert!(
                        (third_neg[[i, j]] + third[[i, j]]).abs() < 1e-8,
                        "row {row}: h-only third contraction should be odd at ({i},{j})"
                    );
                    assert!(
                        (fourth_neg[[i, j]] - fourth[[i, j]]).abs() < 1e-8,
                        "row {row}: h-only fourth contraction should be invariant under flipping both directions at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn w_only_row_primary_higher_order_direction_sign_rules_hold() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_test_link_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build link deviation block");
        let link_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("link initial beta")
            .len();
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(
                Array1::from_iter((0..link_dim).map(|idx| 0.05 * (idx as f64 + 1.0))),
                seed.len(),
            ),
        ];
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: None,
            link_dev: Some(prepared.runtime.clone()),
        };
        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("exact eval cache");
        let total = cache.primary.total;
        let mut dir = Array1::<f64>::zeros(total);
        dir[cache.primary.q] = 0.4;
        dir[cache.primary.logslope] = -0.3;
        let w_range = cache.primary.w.as_ref().expect("w slice");
        dir[w_range.start] = 0.15;
        if w_range.len() > 1 {
            dir[w_range.start + 1] = -0.07;
        }
        let neg_dir = dir.mapv(|value| -value);

        for row in 0..seed.len() {
            let row_ctx = family
                .build_row_exact_context(row, &block_states)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
            let third = family
                .row_primary_third_contracted_recompute(row, &block_states, &cache, &row_ctx, &dir)
                .unwrap_or_else(|e| panic!("row {row}: third contraction failed: {e}"));
            let third_neg = family
                .row_primary_third_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &neg_dir,
                )
                .unwrap_or_else(|e| panic!("row {row}: negated third contraction failed: {e}"));
            let fourth = family
                .row_primary_fourth_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir,
                    &dir,
                )
                .unwrap_or_else(|e| panic!("row {row}: fourth contraction failed: {e}"));
            let fourth_neg = family
                .row_primary_fourth_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &neg_dir,
                    &neg_dir,
                )
                .unwrap_or_else(|e| {
                    panic!("row {row}: doubly-negated fourth contraction failed: {e}")
                });

            for i in 0..total {
                for j in 0..total {
                    assert!(
                        (third_neg[[i, j]] + third[[i, j]]).abs() < 1e-8,
                        "row {row}: w-only third contraction should be odd at ({i},{j})"
                    );
                    assert!(
                        (fourth_neg[[i, j]] - fourth[[i, j]]).abs() < 1e-8,
                        "row {row}: w-only fourth contraction should be invariant under flipping both directions at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn dual_flex_exact_outer_direction_sign_rules_hold() {
        let z = array![-0.8, 0.2, 1.1];
        let y = array![0.0, 1.0, 1.0];
        let weights = array![1.0, 0.7, 1.3];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_test_link_deviation_block_from_seed(
            &link_seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("link block");
        let family =
            BernoulliMarginalSlopeFamily {
                y: Arc::new(y.clone()),
                weights: Arc::new(weights.clone()),
                z: Arc::new(z.clone()),
                gaussian_frailty_sd: None,
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: Some(score_prepared.runtime.clone()),
                link_dev: Some(link_prepared.runtime.clone()),
            };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.25],
                eta: Array1::from_elem(z.len(), 0.25),
            },
            ParameterBlockState {
                beta: array![0.6],
                eta: Array1::from_elem(z.len(), 0.6),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
        ];

        let slices = block_slices(&family);
        let total = slices.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[slices.marginal.start] = 0.7;
        dir_u[slices.logslope.start] = -0.2;
        let h_range = slices.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.1;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.05;
        }
        let w_range = slices.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.08;

        dir_v[slices.marginal.start] = -0.4;
        dir_v[slices.logslope.start] = 0.3;
        dir_v[h_range.start] = -0.03;
        dir_v[w_range.start] = 0.06;
        if w_range.len() > 1 {
            dir_v[w_range.start + 1] = -0.02;
        }

        let neg_dir_u = dir_u.mapv(|value| -value);
        let third = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_u)
            .expect("dual-flex third directional derivative")
            .expect("dual-flex third directional derivative matrix");
        let third_neg = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &neg_dir_u)
            .expect("dual-flex negated third directional derivative")
            .expect("dual-flex negated third directional derivative matrix");
        let fourth = family
            .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_u, &dir_v)
            .expect("dual-flex fourth directional derivative")
            .expect("dual-flex fourth directional derivative matrix");
        let fourth_neg_u = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &block_states,
                &neg_dir_u,
                &dir_v,
            )
            .expect("dual-flex negated-u fourth directional derivative")
            .expect("dual-flex negated-u fourth directional derivative matrix");

        assert_eq!(third.dim(), (total, total));
        assert_eq!(third_neg.dim(), (total, total));
        assert_eq!(fourth.dim(), (total, total));
        assert_eq!(fourth_neg_u.dim(), (total, total));
        for i in 0..total {
            for j in 0..total {
                assert!(
                    (third_neg[[i, j]] + third[[i, j]]).abs() < 1e-8,
                    "third directional derivative should be odd in its direction at ({i},{j})"
                );
                assert!(
                    (fourth_neg_u[[i, j]] + fourth[[i, j]]).abs() < 1e-8,
                    "fourth directional derivative should be linear in dir_u sign at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn dual_flex_exact_outer_fourth_double_sign_flip_is_invariant() {
        let z = array![-0.8, 0.2, 1.1];
        let y = array![0.0, 1.0, 1.0];
        let weights = array![1.0, 0.7, 1.3];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_test_link_deviation_block_from_seed(
            &link_seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("link block");
        let family =
            BernoulliMarginalSlopeFamily {
                y: Arc::new(y.clone()),
                weights: Arc::new(weights.clone()),
                z: Arc::new(z.clone()),
                gaussian_frailty_sd: None,
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: Some(score_prepared.runtime.clone()),
                link_dev: Some(link_prepared.runtime.clone()),
            };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.25],
                eta: Array1::from_elem(z.len(), 0.25),
            },
            ParameterBlockState {
                beta: array![0.6],
                eta: Array1::from_elem(z.len(), 0.6),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
        ];

        let slices = block_slices(&family);
        let total = slices.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[slices.marginal.start] = 0.7;
        dir_u[slices.logslope.start] = -0.2;
        let h_range = slices.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.1;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.05;
        }
        let w_range = slices.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.08;

        dir_v[slices.marginal.start] = -0.4;
        dir_v[slices.logslope.start] = 0.3;
        dir_v[h_range.start] = -0.03;
        dir_v[w_range.start] = 0.06;
        if w_range.len() > 1 {
            dir_v[w_range.start + 1] = -0.02;
        }

        let neg_dir_u = dir_u.mapv(|value| -value);
        let neg_dir_v = dir_v.mapv(|value| -value);
        let forward = family
            .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_u, &dir_v)
            .expect("dual-flex fourth directional derivative")
            .expect("dual-flex fourth directional derivative matrix");
        let flipped = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &block_states,
                &neg_dir_u,
                &neg_dir_v,
            )
            .expect("dual-flex doubly-negated fourth directional derivative")
            .expect("dual-flex doubly-negated fourth directional derivative matrix");

        assert_eq!(forward.dim(), (total, total));
        assert_eq!(flipped.dim(), (total, total));
        for i in 0..total {
            for j in 0..total {
                assert!(
                    (forward[[i, j]] - flipped[[i, j]]).abs() < 1e-8,
                    "fourth directional derivative should be invariant under flipping both directions at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn dual_flex_exact_outer_third_direction_is_linear() {
        let z = array![-0.8, 0.2, 1.1];
        let y = array![0.0, 1.0, 1.0];
        let weights = array![1.0, 0.7, 1.3];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_test_link_deviation_block_from_seed(
            &link_seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("link block");
        let family =
            BernoulliMarginalSlopeFamily {
                y: Arc::new(y.clone()),
                weights: Arc::new(weights.clone()),
                z: Arc::new(z.clone()),
                gaussian_frailty_sd: None,
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: Some(score_prepared.runtime.clone()),
                link_dev: Some(link_prepared.runtime.clone()),
            };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.25],
                eta: Array1::from_elem(z.len(), 0.25),
            },
            ParameterBlockState {
                beta: array![0.6],
                eta: Array1::from_elem(z.len(), 0.6),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
        ];

        let slices = block_slices(&family);
        let total = slices.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[slices.marginal.start] = 0.7;
        dir_u[slices.logslope.start] = -0.2;
        let h_range = slices.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.1;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.05;
        }
        let w_range = slices.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.08;

        dir_v[slices.marginal.start] = -0.4;
        dir_v[slices.logslope.start] = 0.3;
        dir_v[h_range.start] = -0.03;
        dir_v[w_range.start] = 0.06;
        if w_range.len() > 1 {
            dir_v[w_range.start + 1] = -0.02;
        }

        let dir_sum = &dir_u + &dir_v;
        let third_u = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_u)
            .expect("dual-flex third directional derivative u")
            .expect("dual-flex third directional derivative u matrix");
        let third_v = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_v)
            .expect("dual-flex third directional derivative v")
            .expect("dual-flex third directional derivative v matrix");
        let third_sum = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_sum)
            .expect("dual-flex third directional derivative sum")
            .expect("dual-flex third directional derivative sum matrix");

        for i in 0..total {
            for j in 0..total {
                let expected = third_u[[i, j]] + third_v[[i, j]];
                assert!(
                    (third_sum[[i, j]] - expected).abs() < 1e-8,
                    "third directional derivative should be linear in its direction at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn dual_flex_row_primary_third_direction_is_linear() {
        let z = array![-0.8, 0.2, 1.1];
        let y = array![0.0, 1.0, 1.0];
        let weights = array![1.0, 0.7, 1.3];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_test_link_deviation_block_from_seed(
            &link_seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("link block");
        let family =
            BernoulliMarginalSlopeFamily {
                y: Arc::new(y.clone()),
                weights: Arc::new(weights.clone()),
                z: Arc::new(z.clone()),
                gaussian_frailty_sd: None,
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: Some(score_prepared.runtime.clone()),
                link_dev: Some(link_prepared.runtime.clone()),
            };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.25],
                eta: Array1::from_elem(z.len(), 0.25),
            },
            ParameterBlockState {
                beta: array![0.6],
                eta: Array1::from_elem(z.len(), 0.6),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
        ];

        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("exact eval cache");
        let total = cache.primary.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[cache.primary.q] = 0.7;
        dir_u[cache.primary.logslope] = -0.2;
        let h_range = cache.primary.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.1;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.05;
        }
        let w_range = cache.primary.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.08;

        dir_v[cache.primary.q] = -0.4;
        dir_v[cache.primary.logslope] = 0.3;
        dir_v[h_range.start] = -0.03;
        dir_v[w_range.start] = 0.06;
        if w_range.len() > 1 {
            dir_v[w_range.start + 1] = -0.02;
        }

        let dir_sum = &dir_u + &dir_v;
        for row in 0..z.len() {
            let row_ctx = family
                .build_row_exact_context(row, &block_states)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
            let third_u = family
                .row_primary_third_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_u,
                )
                .unwrap_or_else(|e| panic!("row {row}: third contraction u failed: {e}"));
            let third_v = family
                .row_primary_third_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_v,
                )
                .unwrap_or_else(|e| panic!("row {row}: third contraction v failed: {e}"));
            let third_sum = family
                .row_primary_third_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_sum,
                )
                .unwrap_or_else(|e| panic!("row {row}: third contraction sum failed: {e}"));

            for i in 0..total {
                for j in 0..total {
                    let expected = third_u[[i, j]] + third_v[[i, j]];
                    assert!(
                        (third_sum[[i, j]] - expected).abs() < 1e-8,
                        "row {row}: third contraction should be linear in its direction at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn h_only_row_primary_third_direction_is_linear() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score-warp block");
        let score_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("score-warp initial beta")
            .len();
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(
                Array1::from_iter((0..score_dim).map(|idx| 0.04 * (idx as f64 + 1.0))),
                seed.len(),
            ),
        ];
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: Some(prepared.runtime.clone()),
            link_dev: None,
        };
        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("exact eval cache");
        let total = cache.primary.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[cache.primary.q] = -0.35;
        dir_u[cache.primary.logslope] = 0.28;
        let h_range = cache.primary.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.12;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.06;
        }

        dir_v[cache.primary.q] = 0.18;
        dir_v[cache.primary.logslope] = -0.22;
        dir_v[h_range.start] = 0.07;
        if h_range.len() > 1 {
            dir_v[h_range.start + 1] = 0.05;
        }

        let dir_sum = &dir_u + &dir_v;
        for row in 0..seed.len() {
            let row_ctx = family
                .build_row_exact_context(row, &block_states)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
            let third_u = family
                .row_primary_third_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_u,
                )
                .unwrap_or_else(|e| panic!("row {row}: third contraction u failed: {e}"));
            let third_v = family
                .row_primary_third_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_v,
                )
                .unwrap_or_else(|e| panic!("row {row}: third contraction v failed: {e}"));
            let third_sum = family
                .row_primary_third_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_sum,
                )
                .unwrap_or_else(|e| panic!("row {row}: third contraction sum failed: {e}"));

            for i in 0..total {
                for j in 0..total {
                    let expected = third_u[[i, j]] + third_v[[i, j]];
                    assert!(
                        (third_sum[[i, j]] - expected).abs() < 1e-8,
                        "row {row}: h-only third contraction should be linear at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn w_only_row_primary_third_direction_is_linear() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_test_link_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build link deviation block");
        let link_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("link initial beta")
            .len();
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(
                Array1::from_iter((0..link_dim).map(|idx| 0.05 * (idx as f64 + 1.0))),
                seed.len(),
            ),
        ];
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: None,
            link_dev: Some(prepared.runtime.clone()),
        };
        let cache = family
            .build_exact_eval_cache(&block_states)
            .expect("exact eval cache");
        let total = cache.primary.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[cache.primary.q] = 0.4;
        dir_u[cache.primary.logslope] = -0.3;
        let w_range = cache.primary.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.15;
        if w_range.len() > 1 {
            dir_u[w_range.start + 1] = -0.07;
        }

        dir_v[cache.primary.q] = -0.2;
        dir_v[cache.primary.logslope] = 0.25;
        dir_v[w_range.start] = 0.09;
        if w_range.len() > 1 {
            dir_v[w_range.start + 1] = 0.03;
        }

        let dir_sum = &dir_u + &dir_v;
        for row in 0..seed.len() {
            let row_ctx = family
                .build_row_exact_context(row, &block_states)
                .unwrap_or_else(|e| panic!("row {row}: build_row_exact_context failed: {e}"));
            let third_u = family
                .row_primary_third_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_u,
                )
                .unwrap_or_else(|e| panic!("row {row}: third contraction u failed: {e}"));
            let third_v = family
                .row_primary_third_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_v,
                )
                .unwrap_or_else(|e| panic!("row {row}: third contraction v failed: {e}"));
            let third_sum = family
                .row_primary_third_contracted_recompute(
                    row,
                    &block_states,
                    &cache,
                    &row_ctx,
                    &dir_sum,
                )
                .unwrap_or_else(|e| panic!("row {row}: third contraction sum failed: {e}"));

            for i in 0..total {
                for j in 0..total {
                    let expected = third_u[[i, j]] + third_v[[i, j]];
                    assert!(
                        (third_sum[[i, j]] - expected).abs() < 1e-8,
                        "row {row}: w-only third contraction should be linear at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn dual_flex_exact_outer_fourth_first_direction_is_linear() {
        let z = array![-0.8, 0.2, 1.1];
        let y = array![0.0, 1.0, 1.0];
        let weights = array![1.0, 0.7, 1.3];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_test_link_deviation_block_from_seed(
            &link_seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("link block");
        let family =
            BernoulliMarginalSlopeFamily {
                y: Arc::new(y.clone()),
                weights: Arc::new(weights.clone()),
                z: Arc::new(z.clone()),
                gaussian_frailty_sd: None,
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: Some(score_prepared.runtime.clone()),
                link_dev: Some(link_prepared.runtime.clone()),
            };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.25],
                eta: Array1::from_elem(z.len(), 0.25),
            },
            ParameterBlockState {
                beta: array![0.6],
                eta: Array1::from_elem(z.len(), 0.6),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
        ];

        let slices = block_slices(&family);
        let total = slices.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        let mut dir_w = Array1::<f64>::zeros(total);
        dir_u[slices.marginal.start] = 0.7;
        dir_u[slices.logslope.start] = -0.2;
        let h_range = slices.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.1;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.05;
        }
        let w_range = slices.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.08;

        dir_v[slices.marginal.start] = -0.4;
        dir_v[slices.logslope.start] = 0.3;
        dir_v[h_range.start] = -0.03;
        dir_v[w_range.start] = 0.06;
        if w_range.len() > 1 {
            dir_v[w_range.start + 1] = -0.02;
        }

        dir_w[slices.marginal.start] = 0.11;
        dir_w[slices.logslope.start] = -0.09;
        dir_w[h_range.start] = 0.04;
        dir_w[w_range.start] = -0.05;

        let dir_sum = &dir_u + &dir_v;
        let fourth_u = family
            .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_u, &dir_w)
            .expect("dual-flex fourth directional derivative u,w")
            .expect("dual-flex fourth directional derivative u,w matrix");
        let fourth_v = family
            .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_v, &dir_w)
            .expect("dual-flex fourth directional derivative v,w")
            .expect("dual-flex fourth directional derivative v,w matrix");
        let fourth_sum = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &block_states,
                &dir_sum,
                &dir_w,
            )
            .expect("dual-flex fourth directional derivative (u+v),w")
            .expect("dual-flex fourth directional derivative (u+v),w matrix");

        for i in 0..total {
            for j in 0..total {
                let expected = fourth_u[[i, j]] + fourth_v[[i, j]];
                assert!(
                    (fourth_sum[[i, j]] - expected).abs() < 1e-8,
                    "fourth directional derivative should be linear in its first direction at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn h_only_exact_outer_third_direction_is_linear() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score-warp block");
        let score_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("score-warp initial beta")
            .len();
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(
                Array1::from_iter((0..score_dim).map(|idx| 0.04 * (idx as f64 + 1.0))),
                seed.len(),
            ),
        ];
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: Some(prepared.runtime.clone()),
            link_dev: None,
        };
        let slices = block_slices(&family);
        let total = slices.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        let h_range = slices.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.12;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.06;
        }

        dir_v[h_range.start] = 0.07;
        if h_range.len() > 1 {
            dir_v[h_range.start + 1] = 0.05;
        }

        let dir_sum = &dir_u + &dir_v;
        let third_u = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_u)
            .expect("h-only third directional derivative u")
            .expect("h-only third directional derivative u matrix");
        let third_v = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_v)
            .expect("h-only third directional derivative v")
            .expect("h-only third directional derivative v matrix");
        let third_sum = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_sum)
            .expect("h-only third directional derivative sum")
            .expect("h-only third directional derivative sum matrix");

        for i in 0..total {
            for j in 0..total {
                let expected = third_u[[i, j]] + third_v[[i, j]];
                assert!(
                    (third_sum[[i, j]] - expected).abs() < 1e-8,
                    "h-only third directional derivative should be linear at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn w_only_exact_outer_third_direction_is_linear() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_test_link_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build link deviation block");
        let link_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("link initial beta")
            .len();
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(
                Array1::from_iter((0..link_dim).map(|idx| 0.05 * (idx as f64 + 1.0))),
                seed.len(),
            ),
        ];
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: None,
            link_dev: Some(prepared.runtime.clone()),
        };
        let slices = block_slices(&family);
        let total = slices.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        let w_range = slices.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.15;
        if w_range.len() > 1 {
            dir_u[w_range.start + 1] = -0.07;
        }

        dir_v[w_range.start] = 0.09;
        if w_range.len() > 1 {
            dir_v[w_range.start + 1] = 0.03;
        }

        let dir_sum = &dir_u + &dir_v;
        let third_u = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_u)
            .expect("w-only third directional derivative u")
            .expect("w-only third directional derivative u matrix");
        let third_v = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_v)
            .expect("w-only third directional derivative v")
            .expect("w-only third directional derivative v matrix");
        let third_sum = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_sum)
            .expect("w-only third directional derivative sum")
            .expect("w-only third directional derivative sum matrix");

        for i in 0..total {
            for j in 0..total {
                let expected = third_u[[i, j]] + third_v[[i, j]];
                assert!(
                    (third_sum[[i, j]] - expected).abs() < 1e-8,
                    "w-only third directional derivative should be linear at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn h_only_exact_outer_direction_sign_rules_hold() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score-warp block");
        let score_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("score-warp initial beta")
            .len();
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(
                Array1::from_iter((0..score_dim).map(|idx| 0.04 * (idx as f64 + 1.0))),
                seed.len(),
            ),
        ];
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: Some(prepared.runtime.clone()),
            link_dev: None,
        };
        let slices = block_slices(&family);
        let total = slices.total;
        let mut dir = Array1::<f64>::zeros(total);
        let h_range = slices.h.as_ref().expect("h slice");
        dir[h_range.start] = 0.12;
        if h_range.len() > 1 {
            dir[h_range.start + 1] = -0.06;
        }
        let neg_dir = dir.mapv(|value| -value);

        let third = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &dir)
            .expect("h-only third directional derivative")
            .expect("h-only third directional derivative matrix");
        let third_neg = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &neg_dir)
            .expect("h-only negated third directional derivative")
            .expect("h-only negated third directional derivative matrix");
        let fourth = family
            .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir, &dir)
            .expect("h-only fourth directional derivative")
            .expect("h-only fourth directional derivative matrix");
        let fourth_neg = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &block_states,
                &neg_dir,
                &neg_dir,
            )
            .expect("h-only doubly-negated fourth directional derivative")
            .expect("h-only doubly-negated fourth directional derivative matrix");

        for i in 0..total {
            for j in 0..total {
                assert!(
                    (third_neg[[i, j]] + third[[i, j]]).abs() < 1e-8,
                    "h-only third directional derivative should be odd at ({i},{j})"
                );
                assert!(
                    (fourth_neg[[i, j]] - fourth[[i, j]]).abs() < 1e-8,
                    "h-only fourth directional derivative should be invariant under flipping both directions at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn w_only_exact_outer_direction_sign_rules_hold() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_test_link_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build link deviation block");
        let link_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("link initial beta")
            .len();
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(
                Array1::from_iter((0..link_dim).map(|idx| 0.05 * (idx as f64 + 1.0))),
                seed.len(),
            ),
        ];
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: None,
            link_dev: Some(prepared.runtime.clone()),
        };
        let slices = block_slices(&family);
        let total = slices.total;
        let mut dir = Array1::<f64>::zeros(total);
        let w_range = slices.w.as_ref().expect("w slice");
        dir[w_range.start] = 0.15;
        if w_range.len() > 1 {
            dir[w_range.start + 1] = -0.07;
        }
        let neg_dir = dir.mapv(|value| -value);

        let third = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &dir)
            .expect("w-only third directional derivative")
            .expect("w-only third directional derivative matrix");
        let third_neg = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &neg_dir)
            .expect("w-only negated third directional derivative")
            .expect("w-only negated third directional derivative matrix");
        let fourth = family
            .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir, &dir)
            .expect("w-only fourth directional derivative")
            .expect("w-only fourth directional derivative matrix");
        let fourth_neg = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &block_states,
                &neg_dir,
                &neg_dir,
            )
            .expect("w-only doubly-negated fourth directional derivative")
            .expect("w-only doubly-negated fourth directional derivative matrix");

        for i in 0..total {
            for j in 0..total {
                assert!(
                    (third_neg[[i, j]] + third[[i, j]]).abs() < 1e-8,
                    "w-only third directional derivative should be odd at ({i},{j})"
                );
                assert!(
                    (fourth_neg[[i, j]] - fourth[[i, j]]).abs() < 1e-8,
                    "w-only fourth directional derivative should be invariant under flipping both directions at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn h_only_exact_outer_fourth_direction_swap_is_symmetric() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score-warp block");
        let score_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("score-warp initial beta")
            .len();
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(
                Array1::from_iter((0..score_dim).map(|idx| 0.04 * (idx as f64 + 1.0))),
                seed.len(),
            ),
        ];
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: Some(prepared.runtime.clone()),
            link_dev: None,
        };
        let slices = block_slices(&family);
        let total = slices.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        let h_range = slices.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.12;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.06;
        }

        dir_v[h_range.start] = 0.07;
        if h_range.len() > 1 {
            dir_v[h_range.start + 1] = 0.05;
        }

        let forward = family
            .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_u, &dir_v)
            .expect("h-only fourth directional derivative")
            .expect("h-only fourth directional derivative matrix");
        let swapped = family
            .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_v, &dir_u)
            .expect("h-only swapped fourth directional derivative")
            .expect("h-only swapped fourth directional derivative matrix");

        for i in 0..total {
            for j in 0..total {
                assert!(
                    (forward[[i, j]] - swapped[[i, j]]).abs() < 1e-8,
                    "h-only fourth directional derivative should be symmetric in direction arguments at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn w_only_exact_outer_fourth_direction_swap_is_symmetric() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_test_link_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build link deviation block");
        let link_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("link initial beta")
            .len();
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(
                Array1::from_iter((0..link_dim).map(|idx| 0.05 * (idx as f64 + 1.0))),
                seed.len(),
            ),
        ];
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: None,
            link_dev: Some(prepared.runtime.clone()),
        };
        let slices = block_slices(&family);
        let total = slices.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        let w_range = slices.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.15;
        if w_range.len() > 1 {
            dir_u[w_range.start + 1] = -0.07;
        }

        dir_v[w_range.start] = 0.09;
        if w_range.len() > 1 {
            dir_v[w_range.start + 1] = 0.03;
        }

        let forward = family
            .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_u, &dir_v)
            .expect("w-only fourth directional derivative")
            .expect("w-only fourth directional derivative matrix");
        let swapped = family
            .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_v, &dir_u)
            .expect("w-only swapped fourth directional derivative")
            .expect("w-only swapped fourth directional derivative matrix");

        for i in 0..total {
            for j in 0..total {
                assert!(
                    (forward[[i, j]] - swapped[[i, j]]).abs() < 1e-8,
                    "w-only fourth directional derivative should be symmetric in direction arguments at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn h_only_exact_outer_zero_direction_returns_zero() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_score_warp_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score-warp block");
        let score_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("score-warp initial beta")
            .len();
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(
                Array1::from_iter((0..score_dim).map(|idx| 0.04 * (idx as f64 + 1.0))),
                seed.len(),
            ),
        ];
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: Some(prepared.runtime.clone()),
            link_dev: None,
        };
        let slices = block_slices(&family);
        let zero = Array1::<f64>::zeros(slices.total);
        let third = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &zero)
            .expect("h-only third directional derivative")
            .expect("h-only third directional derivative matrix");
        let fourth = family
            .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &zero, &zero)
            .expect("h-only fourth directional derivative")
            .expect("h-only fourth directional derivative matrix");

        assert!(
            third.iter().all(|value| value.abs() <= 0.0),
            "expected zero h-only third directional derivative for zero direction"
        );
        assert!(
            fourth.iter().all(|value| value.abs() <= 0.0),
            "expected zero h-only fourth directional derivative for zero directions"
        );
    }

    #[test]
    fn w_only_exact_outer_zero_direction_returns_zero() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_test_link_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build link deviation block");
        let link_dim = prepared
            .block
            .initial_beta
            .as_ref()
            .expect("link initial beta")
            .len();
        let block_states = vec![
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(array![0.0], seed.len()),
            dummy_block_state(
                Array1::from_iter((0..link_dim).map(|idx| 0.05 * (idx as f64 + 1.0))),
                seed.len(),
            ),
        ];
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![0.0, 1.0, 0.0, 1.0, 0.0]),
            weights: Arc::new(Array1::ones(seed.len())),
            z: Arc::new(seed.clone()),
            gaussian_frailty_sd: None,
            base_link: bernoulli_marginal_slope_probit_link(),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: None,
            link_dev: Some(prepared.runtime.clone()),
        };
        let slices = block_slices(&family);
        let zero = Array1::<f64>::zeros(slices.total);
        let third = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &zero)
            .expect("w-only third directional derivative")
            .expect("w-only third directional derivative matrix");
        let fourth = family
            .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &zero, &zero)
            .expect("w-only fourth directional derivative")
            .expect("w-only fourth directional derivative matrix");

        assert!(
            third.iter().all(|value| value.abs() <= 0.0),
            "expected zero w-only third directional derivative for zero direction"
        );
        assert!(
            fourth.iter().all(|value| value.abs() <= 0.0),
            "expected zero w-only fourth directional derivative for zero directions"
        );
    }

    #[test]
    fn w_only_gradient_matches_loglik_finite_differences() {
        let z = array![-0.8, 0.2, 1.1];
        let y = array![0.0, 1.0, 1.0];
        let weights = array![1.0, 0.7, 1.3];
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_test_link_deviation_block_from_seed(
            &link_seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("link block");
        let family =
            BernoulliMarginalSlopeFamily {
                y: Arc::new(y.clone()),
                weights: Arc::new(weights.clone()),
                z: Arc::new(z.clone()),
                gaussian_frailty_sd: None,
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: None,
                link_dev: Some(link_prepared.runtime.clone()),
            };
        let beta_w = Array1::from_iter(
            (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
        );
        let states_at = |q: f64, b: f64, bw: Array1<f64>| {
            vec![
                ParameterBlockState {
                    beta: array![q],
                    eta: Array1::from_elem(z.len(), q),
                },
                ParameterBlockState {
                    beta: array![b],
                    eta: Array1::from_elem(z.len(), b),
                },
                ParameterBlockState {
                    beta: bw,
                    eta: Array1::zeros(z.len()),
                },
            ]
        };

        let q0 = 0.25;
        let b0 = 0.6;
        let block_states = states_at(q0, b0, beta_w.clone());
        let eval = family.evaluate(&block_states).expect("family evaluation");
        let grad_q = match &eval.blockworking_sets[0] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            _ => panic!("expected exact-newton q block"),
        };
        let grad_b = match &eval.blockworking_sets[1] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            _ => panic!("expected exact-newton b block"),
        };
        let grad_w0 = match &eval.blockworking_sets[2] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            _ => panic!("expected exact-newton w block"),
        };

        let fd = |which: &str, eps: f64| match which {
            "q" => {
                let plus = family
                    .log_likelihood_only(&states_at(q0 + eps, b0, beta_w.clone()))
                    .expect("ll plus q");
                let minus = family
                    .log_likelihood_only(&states_at(q0 - eps, b0, beta_w.clone()))
                    .expect("ll minus q");
                (plus - minus) / (2.0 * eps)
            }
            "b" => {
                let plus = family
                    .log_likelihood_only(&states_at(q0, b0 + eps, beta_w.clone()))
                    .expect("ll plus b");
                let minus = family
                    .log_likelihood_only(&states_at(q0, b0 - eps, beta_w.clone()))
                    .expect("ll minus b");
                (plus - minus) / (2.0 * eps)
            }
            "w0" => {
                let mut plus_w = beta_w.clone();
                plus_w[0] += eps;
                let mut minus_w = beta_w.clone();
                minus_w[0] -= eps;
                let plus = family
                    .log_likelihood_only(&states_at(q0, b0, plus_w))
                    .expect("ll plus w");
                let minus = family
                    .log_likelihood_only(&states_at(q0, b0, minus_w))
                    .expect("ll minus w");
                (plus - minus) / (2.0 * eps)
            }
            _ => panic!("unknown derivative target"),
        };

        let eps = 1e-5;
        assert!((grad_q - fd("q", eps)).abs() < 2e-4);
        assert!((grad_b - fd("b", eps)).abs() < 2e-4);
        assert!((grad_w0 - fd("w0", eps)).abs() < 2e-4);
    }

    #[test]
    fn h_only_gradient_matches_loglik_finite_differences() {
        let z = array![-0.8, 0.2, 1.1];
        let y = array![0.0, 1.0, 1.0];
        let weights = array![1.0, 0.7, 1.3];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let family =
            BernoulliMarginalSlopeFamily {
                y: Arc::new(y.clone()),
                weights: Arc::new(weights.clone()),
                z: Arc::new(z.clone()),
                gaussian_frailty_sd: None,
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: Some(score_prepared.runtime.clone()),
                link_dev: None,
            };
        let beta_h = Array1::from_iter(
            (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
        );
        let states_at = |q: f64, b: f64, bh: Array1<f64>| {
            vec![
                ParameterBlockState {
                    beta: array![q],
                    eta: Array1::from_elem(z.len(), q),
                },
                ParameterBlockState {
                    beta: array![b],
                    eta: Array1::from_elem(z.len(), b),
                },
                ParameterBlockState {
                    beta: bh,
                    eta: Array1::zeros(z.len()),
                },
            ]
        };

        let q0 = 0.25;
        let b0 = 0.6;
        let block_states = states_at(q0, b0, beta_h.clone());
        let eval = family.evaluate(&block_states).expect("family evaluation");
        let grad_q = match &eval.blockworking_sets[0] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            _ => panic!("expected exact-newton q block"),
        };
        let grad_b = match &eval.blockworking_sets[1] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            _ => panic!("expected exact-newton b block"),
        };
        let grad_h0 = match &eval.blockworking_sets[2] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            _ => panic!("expected exact-newton h block"),
        };

        let fd = |which: &str, eps: f64| match which {
            "q" => {
                let plus = family
                    .log_likelihood_only(&states_at(q0 + eps, b0, beta_h.clone()))
                    .expect("ll plus q");
                let minus = family
                    .log_likelihood_only(&states_at(q0 - eps, b0, beta_h.clone()))
                    .expect("ll minus q");
                (plus - minus) / (2.0 * eps)
            }
            "b" => {
                let plus = family
                    .log_likelihood_only(&states_at(q0, b0 + eps, beta_h.clone()))
                    .expect("ll plus b");
                let minus = family
                    .log_likelihood_only(&states_at(q0, b0 - eps, beta_h.clone()))
                    .expect("ll minus b");
                (plus - minus) / (2.0 * eps)
            }
            "h0" => {
                let mut plus_h = beta_h.clone();
                plus_h[0] += eps;
                let mut minus_h = beta_h.clone();
                minus_h[0] -= eps;
                let plus = family
                    .log_likelihood_only(&states_at(q0, b0, plus_h))
                    .expect("ll plus h");
                let minus = family
                    .log_likelihood_only(&states_at(q0, b0, minus_h))
                    .expect("ll minus h");
                (plus - minus) / (2.0 * eps)
            }
            _ => panic!("unknown derivative target"),
        };

        let eps = 1e-5;
        assert!((grad_q - fd("q", eps)).abs() < 2e-4);
        assert!((grad_b - fd("b", eps)).abs() < 2e-4);
        assert!((grad_h0 - fd("h0", eps)).abs() < 2e-4);
    }

    #[test]
    fn flexible_denested_gradient_matches_loglik_finite_differences() {
        let z = array![-0.8, 0.2, 1.1];
        let y = array![0.0, 1.0, 1.0];
        let weights = array![1.0, 0.7, 1.3];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_test_link_deviation_block_from_seed(
            &link_seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("link block");
        let family =
            BernoulliMarginalSlopeFamily {
                y: Arc::new(y.clone()),
                weights: Arc::new(weights.clone()),
                z: Arc::new(z.clone()),
                gaussian_frailty_sd: None,
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: Some(score_prepared.runtime.clone()),
                link_dev: Some(link_prepared.runtime.clone()),
            };
        let beta_h = Array1::from_iter(
            (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
        );
        let beta_w = Array1::from_iter(
            (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
        );
        let states_at = |q: f64, b: f64, bh: Array1<f64>, bw: Array1<f64>| {
            vec![
                ParameterBlockState {
                    beta: array![q],
                    eta: Array1::from_elem(z.len(), q),
                },
                ParameterBlockState {
                    beta: array![b],
                    eta: Array1::from_elem(z.len(), b),
                },
                ParameterBlockState {
                    beta: bh,
                    eta: Array1::zeros(z.len()),
                },
                ParameterBlockState {
                    beta: bw,
                    eta: Array1::zeros(z.len()),
                },
            ]
        };

        let q0 = 0.25;
        let b0 = 0.6;
        let block_states = states_at(q0, b0, beta_h.clone(), beta_w.clone());
        let eval = family.evaluate(&block_states).expect("family evaluation");
        let grad_q = match &eval.blockworking_sets[0] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            _ => panic!("expected exact-newton q block"),
        };
        let grad_b = match &eval.blockworking_sets[1] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            _ => panic!("expected exact-newton b block"),
        };
        let grad_h0 = match &eval.blockworking_sets[2] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            _ => panic!("expected exact-newton h block"),
        };
        let grad_w0 = match &eval.blockworking_sets[3] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
            _ => panic!("expected exact-newton w block"),
        };

        let fd = |which: &str, eps: f64| match which {
            "q" => {
                let plus = family
                    .log_likelihood_only(&states_at(q0 + eps, b0, beta_h.clone(), beta_w.clone()))
                    .expect("ll plus q");
                let minus = family
                    .log_likelihood_only(&states_at(q0 - eps, b0, beta_h.clone(), beta_w.clone()))
                    .expect("ll minus q");
                (plus - minus) / (2.0 * eps)
            }
            "b" => {
                let plus = family
                    .log_likelihood_only(&states_at(q0, b0 + eps, beta_h.clone(), beta_w.clone()))
                    .expect("ll plus b");
                let minus = family
                    .log_likelihood_only(&states_at(q0, b0 - eps, beta_h.clone(), beta_w.clone()))
                    .expect("ll minus b");
                (plus - minus) / (2.0 * eps)
            }
            "h0" => {
                let mut plus_h = beta_h.clone();
                plus_h[0] += eps;
                let mut minus_h = beta_h.clone();
                minus_h[0] -= eps;
                let plus = family
                    .log_likelihood_only(&states_at(q0, b0, plus_h, beta_w.clone()))
                    .expect("ll plus h");
                let minus = family
                    .log_likelihood_only(&states_at(q0, b0, minus_h, beta_w.clone()))
                    .expect("ll minus h");
                (plus - minus) / (2.0 * eps)
            }
            "w0" => {
                let mut plus_w = beta_w.clone();
                plus_w[0] += eps;
                let mut minus_w = beta_w.clone();
                minus_w[0] -= eps;
                let plus = family
                    .log_likelihood_only(&states_at(q0, b0, beta_h.clone(), plus_w))
                    .expect("ll plus w");
                let minus = family
                    .log_likelihood_only(&states_at(q0, b0, beta_h.clone(), minus_w))
                    .expect("ll minus w");
                (plus - minus) / (2.0 * eps)
            }
            _ => panic!("unknown derivative target"),
        };

        let eps = 1e-5;
        assert!((grad_q - fd("q", eps)).abs() < 2e-4);
        assert!((grad_b - fd("b", eps)).abs() < 2e-4);
        assert!((grad_h0 - fd("h0", eps)).abs() < 2e-4);
        assert!((grad_w0 - fd("w0", eps)).abs() < 2e-4);
    }

    #[test]
    fn flexible_exact_outer_directional_derivatives_are_present_and_finite() {
        let z = array![-0.8, 0.2, 1.1];
        let y = array![0.0, 1.0, 1.0];
        let weights = array![1.0, 0.7, 1.3];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_test_link_deviation_block_from_seed(
            &link_seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("link block");
        let family =
            BernoulliMarginalSlopeFamily {
                y: Arc::new(y.clone()),
                weights: Arc::new(weights.clone()),
                z: Arc::new(z.clone()),
                gaussian_frailty_sd: None,
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: Some(score_prepared.runtime.clone()),
                link_dev: Some(link_prepared.runtime.clone()),
            };
        let beta_h = Array1::from_iter(
            (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
        );
        let beta_w = Array1::from_iter(
            (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
        );
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.25],
                eta: Array1::from_elem(z.len(), 0.25),
            },
            ParameterBlockState {
                beta: array![0.6],
                eta: Array1::from_elem(z.len(), 0.6),
            },
            ParameterBlockState {
                beta: beta_h.clone(),
                eta: Array1::zeros(z.len()),
            },
            ParameterBlockState {
                beta: beta_w.clone(),
                eta: Array1::zeros(z.len()),
            },
        ];

        let slices = block_slices(&family);
        let total = slices.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[slices.marginal.start] = 0.7;
        dir_u[slices.logslope.start] = -0.2;
        if let Some(h_range) = slices.h.as_ref() {
            dir_u[h_range.start] = 0.1;
            if h_range.len() > 1 {
                dir_u[h_range.start + 1] = -0.05;
            }
        }
        if let Some(w_range) = slices.w.as_ref() {
            dir_u[w_range.start] = 0.08;
        }

        dir_v[slices.marginal.start] = -0.4;
        dir_v[slices.logslope.start] = 0.3;
        if let Some(h_range) = slices.h.as_ref() {
            dir_v[h_range.start] = -0.03;
        }
        if let Some(w_range) = slices.w.as_ref() {
            dir_v[w_range.start] = 0.06;
            if w_range.len() > 1 {
                dir_v[w_range.start + 1] = -0.02;
            }
        }

        let third = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &dir_u)
            .expect("flex third directional derivative")
            .expect("flex third directional derivative matrix");
        let fourth = family
            .exact_newton_joint_hessiansecond_directional_derivative(&block_states, &dir_u, &dir_v)
            .expect("flex fourth directional derivative")
            .expect("flex fourth directional derivative matrix");

        assert_eq!(third.dim(), (total, total));
        assert_eq!(fourth.dim(), (total, total));
        assert!(third.iter().all(|value| value.is_finite()));
        assert!(fourth.iter().all(|value| value.is_finite()));
        let max_abs_third = third
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let max_abs_fourth = fourth
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            max_abs_third > 1e-10,
            "expected nonzero dual-flex third directional derivative"
        );
        assert!(
            max_abs_fourth > 1e-10,
            "expected nonzero dual-flex fourth directional derivative"
        );

        for i in 0..total {
            for j in 0..i {
                assert!((third[[i, j]] - third[[j, i]]).abs() < 1e-8);
                assert!((fourth[[i, j]] - fourth[[j, i]]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn latent_z_normalization_accepts_finite_sample_gaussian_scores() {
        let z = array![
            -0.85, -0.12, 0.31, 1.04, -1.21, 0.56, 0.77, -0.44, 1.33, -0.09, 0.28, -0.67
        ];
        let weights = Array1::from_elem(12, 1.0);
        let (standardized, normalization) = standardize_latent_z_with_policy(
            &z,
            &weights,
            "bernoulli-marginal-slope",
            &LatentZPolicy::default(),
        )
        .expect("normalize z");
        let replayed = normalization
            .apply(&z, "bernoulli-marginal-slope replay")
            .expect("replay normalized z");
        let mean = standardized.sum() / standardized.len() as f64;
        let var = standardized.iter().map(|v| v * v).sum::<f64>() / standardized.len() as f64;
        assert_eq!(replayed, standardized);
        assert!(mean.abs() < 1e-12);
        assert!((var.sqrt() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn latent_z_normalization_rejects_extreme_non_gaussian_scores() {
        let z = array![0.0, 0.0, 0.0, 0.0, 10.0, -10.0];
        let weights = Array1::from_elem(6, 1.0);
        let err = standardize_latent_z_with_policy(
            &z,
            &weights,
            "bernoulli-marginal-slope",
            &LatentZPolicy::default(),
        )
        .expect_err("expected non-gaussian rejection");
        assert!(err.contains("approximately latent N(0,1)"));
    }

    #[test]
    fn flexible_family_exposes_exact_newton_workspaces() {
        let z = array![-0.8, 0.2, 1.1];
        let y = array![0.0, 1.0, 1.0];
        let weights = array![1.0, 0.7, 1.3];
        let score_prepared = build_score_warp_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_test_link_deviation_block_from_seed(
            &link_seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("link block");
        let family =
            BernoulliMarginalSlopeFamily {
                y: Arc::new(y.clone()),
                weights: Arc::new(weights.clone()),
                z: Arc::new(z.clone()),
                gaussian_frailty_sd: None,
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: Some(score_prepared.runtime.clone()),
                link_dev: Some(link_prepared.runtime.clone()),
            };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.25],
                eta: Array1::from_elem(z.len(), 0.25),
            },
            ParameterBlockState {
                beta: array![0.6],
                eta: Array1::from_elem(z.len(), 0.6),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..score_prepared.block.design.ncols()).map(|idx| 0.015 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
            ParameterBlockState {
                beta: Array1::from_iter(
                    (0..link_prepared.block.design.ncols()).map(|idx| 0.01 * (idx as f64 + 1.0)),
                ),
                eta: Array1::zeros(z.len()),
            },
        ];
        let specs = vec![
            dummy_blockspec(1, z.len()),
            dummy_blockspec(1, z.len()),
            dummy_blockspec(score_prepared.block.design.ncols(), z.len()),
            dummy_blockspec(link_prepared.block.design.ncols(), z.len()),
        ];
        let derivative_blocks = vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        assert!(
            family
                .exact_newton_joint_hessian_workspace(&block_states, &specs)
                .expect("flex hessian workspace")
                .is_some()
        );
        assert!(
            family
                .exact_newton_joint_psi_workspace(&block_states, &specs, &derivative_blocks)
                .expect("flex psi workspace")
                .is_some()
        );
    }

    #[test]
    fn sigma_exact_joint_psi_terms_returns_analytic_terms() {
        let z = array![-0.8, 0.2, 1.1];
        let y = array![0.0, 1.0, 1.0];
        let weights = array![1.0, 0.7, 1.3];
        let sigma = 0.7;
        let make_family = |sigma: f64| {
            BernoulliMarginalSlopeFamily {
                y: Arc::new(y.clone()),
                weights: Arc::new(weights.clone()),
                z: Arc::new(z.clone()),
                gaussian_frailty_sd: Some(sigma),
                base_link: bernoulli_marginal_slope_probit_link(),
                marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                    array![[1.0], [1.0], [1.0]],
                )),
                score_warp: None,
                link_dev: None,
            }
        };
        let family = make_family(sigma);
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.25],
                eta: Array1::from_elem(z.len(), 0.25),
            },
            ParameterBlockState {
                beta: array![0.6],
                eta: Array1::from_elem(z.len(), 0.6),
            },
        ];
        let specs = vec![dummy_blockspec(1, z.len()), dummy_blockspec(1, z.len())];

        let terms = family
            .sigma_exact_joint_psi_terms(&block_states, &specs)
            .expect("analytic sigma psi terms")
            .expect("sigma terms present");
        assert!(terms.objective_psi.is_finite());
        assert_eq!(terms.score_psi.len(), 2);
        assert!(terms.score_psi.iter().all(|value| value.is_finite()));
        assert_eq!(
            terms
                .hessian_psi_operator
                .as_ref()
                .expect("sigma Hessian operator")
                .to_dense()
                .dim(),
            (2, 2)
        );

        let second = family
            .sigma_exact_joint_psisecond_order_terms(&block_states)
            .expect("analytic second sigma terms")
            .expect("second sigma terms present");
        assert!(second.objective_psi_psi.is_finite());
        assert_eq!(second.score_psi_psi.len(), 2);

        let drift = family
            .sigma_exact_joint_psihessian_directional_derivative(&block_states, &array![0.1, -0.2])
            .expect("analytic sigma Hessian directional derivative")
            .expect("sigma drift present");
        assert_eq!(drift.dim(), (2, 2));
        assert!(drift.iter().all(|value| value.is_finite()));

        let tau = sigma.ln();
        let eps = 1e-5;
        let ll_plus = make_family((tau + eps).exp())
            .log_likelihood_only(&block_states)
            .expect("ll plus sigma");
        let ll_minus = make_family((tau - eps).exp())
            .log_likelihood_only(&block_states)
            .expect("ll minus sigma");
        let objective_fd = -(ll_plus - ll_minus) / (2.0 * eps);
        assert!((terms.objective_psi - objective_fd).abs() < 1e-5);
    }
}

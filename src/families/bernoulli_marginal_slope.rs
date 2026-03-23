use crate::basis::BasisOptions;
use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyJointDesignChannel,
    CustomFamilyJointDesignPairContribution, CustomFamilyJointPsiOperator,
    CustomFamilyPsiDesignAction, CustomFamilyPsiSecondDesignAction, CustomFamilyWarmStart,
    ExactNewtonJointHessianWorkspace, ExactNewtonJointPsiSecondOrderTerms,
    ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace, ExactOuterDerivativeOrder,
    FamilyEvaluation, ParameterBlockSpec, ParameterBlockState, build_block_spatial_psi_derivatives,
    cost_gated_outer_order, custom_family_outer_derivatives, evaluate_custom_family_joint_hyper,
    first_psi_linear_map, fit_custom_family, second_psi_linear_map,
};
use crate::estimate::UnifiedFitResult;
use crate::estimate::reml::unified::HyperOperator;
use crate::families::gamlss::{
    ParameterBlockInput, WiggleBlockConfig, monotone_wiggle_basis_with_derivative_order,
    select_wiggle_basis_from_seed,
};
use crate::families::lognormal_kernel::FrailtySpec;
use crate::families::row_kernel::{RowKernel, RowKernelHessianWorkspace, build_row_kernel_cache};
use crate::matrix::{DesignMatrix, SymmetricMatrix};
use crate::pirls::LinearInequalityConstraints;
use crate::probability::{normal_cdf, normal_pdf, standard_normal_quantile};
use crate::smooth::{
    ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionDesign, TermCollectionSpec, build_term_collection_designs_joint,
    freeze_term_collection_from_design, optimize_spatial_length_scale_exact_joint,
    spatial_length_scale_term_indices,
};
use crate::span::span_index_for_breakpoints;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use statrs::function::erf::erfc;
use std::cell::RefCell;
use std::sync::Arc;

pub(crate) mod exact_kernel;

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
        Self {
            degree: 3,
            num_internal_knots: 8,
            penalty_order: 2,
            penalty_orders: vec![1, 2],
            double_penalty: true,
            monotonicity_eps: 1e-4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeviationRuntime {
    pub knots: Array1<f64>,
    pub degree: usize,
    pub basis_dim: usize,
    pub monotonicity_eps: f64,
    endpoint_points: Array1<f64>,
    span_left_value_design: Array2<f64>,
    span_left_points: Array1<f64>,
    span_right_points: Array1<f64>,
    span_left_d1_design: Array2<f64>,
    span_right_d1_design: Array2<f64>,
    span_left_d2_design: Array2<f64>,
    span_mid_d3_design: Array2<f64>,
}

#[derive(Clone)]
pub(crate) struct DeviationPrepared {
    pub(crate) block: ParameterBlockInput,
    pub(crate) runtime: DeviationRuntime,
}

#[derive(Clone)]
pub struct BernoulliMarginalSlopeTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub z: Array1<f64>,
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
}

pub struct BernoulliMarginalSlopeFitResult {
    pub fit: UnifiedFitResult,
    pub marginalspec_resolved: TermCollectionSpec,
    pub logslopespec_resolved: TermCollectionSpec,
    pub marginal_design: TermCollectionDesign,
    pub logslope_design: TermCollectionDesign,
    pub baseline_marginal: f64,
    pub baseline_logslope: f64,
    pub score_warp_runtime: Option<DeviationRuntime>,
    pub link_dev_runtime: Option<DeviationRuntime>,
}

#[derive(Clone)]
struct BernoulliMarginalSlopeFamily {
    y: Arc<Array1<f64>>,
    weights: Arc<Array1<f64>>,
    z: Arc<Array1<f64>>,
    gaussian_frailty_sd: Option<f64>,
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

impl DeviationRuntime {
    fn validate_beta_shape(&self, beta: &Array1<f64>, label: &str) -> Result<(), String> {
        if beta.len() != self.basis_dim {
            return Err(format!(
                "{label} length mismatch: got {}, expected {}",
                beta.len(),
                self.basis_dim
            ));
        }
        Ok(())
    }

    pub fn design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(
            values.view(),
            &self.knots,
            self.degree,
            BasisOptions::value().derivative_order,
        )
    }

    pub fn first_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(
            values.view(),
            &self.knots,
            self.degree,
            BasisOptions::first_derivative().derivative_order,
        )
    }

    pub fn second_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(
            values.view(),
            &self.knots,
            self.degree,
            BasisOptions::second_derivative().derivative_order,
        )
    }

    pub fn third_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(values.view(), &self.knots, self.degree, 3)
    }

    pub fn fourth_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        monotone_wiggle_basis_with_derivative_order(values.view(), &self.knots, self.degree, 4)
    }

    fn span_count(&self) -> usize {
        self.span_left_points.len()
    }

    pub(crate) fn breakpoints(&self) -> &Array1<f64> {
        &self.endpoint_points
    }

    fn span_interval(&self, span_idx: usize) -> Result<(f64, f64), String> {
        if span_idx >= self.span_count() {
            return Err(format!(
                "deviation span index {} out of range for {} spans",
                span_idx,
                self.span_count()
            ));
        }
        Ok((
            self.span_left_points[span_idx],
            self.span_right_points[span_idx],
        ))
    }

    fn span_index_for(&self, value: f64) -> Result<usize, String> {
        span_index_for_breakpoints(
            self.endpoint_points
                .as_slice()
                .ok_or_else(|| "deviation runtime breakpoints are not contiguous".to_string())?,
            value,
            "deviation span lookup",
        )
    }

    fn local_cubic_on_span(
        &self,
        beta: &Array1<f64>,
        span_idx: usize,
    ) -> Result<exact_kernel::LocalSpanCubic, String> {
        self.validate_beta_shape(beta, "deviation local cubic coefficients")?;
        let (left, right) = self.span_interval(span_idx)?;
        let value_design = self.span_left_value_design.row(span_idx);
        let d1_design = self.span_left_d1_design.row(span_idx);
        let d2_design = self.span_left_d2_design.row(span_idx);
        let d3 = self.span_mid_d3_design.row(span_idx).dot(beta);
        Ok(exact_kernel::LocalSpanCubic {
            left,
            right,
            c0: value_design.dot(beta),
            c1: d1_design.dot(beta),
            c2: 0.5 * d2_design.dot(beta),
            c3: d3 / 6.0,
        })
    }

    pub fn basis_span_cubic(
        &self,
        span_idx: usize,
        basis_idx: usize,
    ) -> Result<exact_kernel::LocalSpanCubic, String> {
        if basis_idx >= self.basis_dim {
            return Err(format!(
                "deviation basis index {} out of range for {} coefficients",
                basis_idx, self.basis_dim
            ));
        }
        let (left, right) = self.span_interval(span_idx)?;
        Ok(exact_kernel::LocalSpanCubic {
            left,
            right,
            c0: self.span_left_value_design[[span_idx, basis_idx]],
            c1: self.span_left_d1_design[[span_idx, basis_idx]],
            c2: 0.5 * self.span_left_d2_design[[span_idx, basis_idx]],
            c3: self.span_mid_d3_design[[span_idx, basis_idx]] / 6.0,
        })
    }

    pub fn basis_cubic_at(
        &self,
        basis_idx: usize,
        value: f64,
    ) -> Result<exact_kernel::LocalSpanCubic, String> {
        let span_idx = self.span_index_for(value)?;
        self.basis_span_cubic(span_idx, basis_idx)
    }

    pub(crate) fn local_cubic_at(
        &self,
        beta: &Array1<f64>,
        value: f64,
    ) -> Result<exact_kernel::LocalSpanCubic, String> {
        let span_idx = self.span_index_for(value)?;
        self.local_cubic_on_span(beta, span_idx)
    }

    fn support_interval(&self) -> Result<(f64, f64), String> {
        match (self.endpoint_points.first(), self.endpoint_points.last()) {
            (Some(&left), Some(&right)) => Ok((left, right)),
            _ => Err("deviation runtime is missing monotonicity support points".to_string()),
        }
    }

    pub(crate) fn exact_monotonicity_min_slack(&self, beta: &Array1<f64>) -> Result<f64, String> {
        if beta.len() != self.basis_dim {
            return Err(format!(
                "deviation monotonicity length mismatch: got {}, expected {}",
                beta.len(),
                self.basis_dim
            ));
        }
        if beta.iter().any(|value| !value.is_finite()) {
            let bad = beta
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
                .map(|(idx, value)| format!("deviation coefficient {idx} is non-finite ({value})"))
                .unwrap_or_else(|| "deviation coefficient is non-finite".to_string());
            return Err(bad);
        }

        let d1_left = self.span_left_d1_design.dot(beta);
        let d1_right = self.span_right_d1_design.dot(beta);
        let d2_left = self.span_left_d2_design.dot(beta);
        let d3_mid = self.span_mid_d3_design.dot(beta);

        let mut min_slack = f64::INFINITY;
        for span_idx in 0..self.span_left_points.len() {
            let left = self.span_left_points[span_idx];
            let right = self.span_right_points[span_idx];
            let width = right - left;
            if !width.is_finite() || width <= 0.0 {
                continue;
            }
            let left_slack = 1.0 + d1_left[span_idx] - self.monotonicity_eps;
            let right_slack = 1.0 + d1_right[span_idx] - self.monotonicity_eps;
            min_slack = min_slack.min(left_slack.min(right_slack));

            let curvature = d3_mid[span_idx];
            if curvature > 0.0 {
                let t_star = -d2_left[span_idx] / curvature;
                if t_star > 0.0 && t_star < width {
                    let interior = 1.0
                        + d1_left[span_idx]
                        + d2_left[span_idx] * t_star
                        + 0.5 * curvature * t_star * t_star
                        - self.monotonicity_eps;
                    min_slack = min_slack.min(interior);
                }
            }
        }
        if min_slack.is_finite() {
            Ok(min_slack)
        } else {
            Err("deviation monotonicity slack computation produced no active spans".to_string())
        }
    }

    pub(crate) fn monotonicity_feasible(
        &self,
        beta: &Array1<f64>,
        context: &str,
    ) -> Result<(), String> {
        let slack = self.exact_monotonicity_min_slack(beta)?;
        if slack >= -1e-10 {
            Ok(())
        } else {
            let (left, right) = self.support_interval()?;
            Err(format!(
                "{context} violates exact monotonicity on [{left:.6}, {right:.6}] (minimum derivative slack {slack:.3e}, eps={:.3e})",
                self.monotonicity_eps
            ))
        }
    }

    pub(crate) fn max_feasible_monotone_step(
        &self,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        if beta.len() != self.basis_dim || delta.len() != self.basis_dim {
            return Err(format!(
                "deviation monotone step length mismatch: beta={}, delta={}, expected {}",
                beta.len(),
                delta.len(),
                self.basis_dim
            ));
        }
        for (idx, step) in delta.iter().enumerate() {
            if !step.is_finite() {
                return Err(format!("deviation step direction {idx} is non-finite"));
            }
        }
        self.monotonicity_feasible(beta, "deviation monotone coefficients")?;

        let mut trial = beta.clone();
        for idx in 0..trial.len() {
            trial[idx] += delta[idx];
        }
        if self.exact_monotonicity_min_slack(&trial)? >= 0.0 {
            return Ok(Some(1.0));
        }

        let mut alpha_lo = 0.0f64;
        let mut alpha_hi = 1.0f64;
        for _ in 0..48 {
            let alpha_mid = 0.5 * (alpha_lo + alpha_hi);
            for idx in 0..trial.len() {
                trial[idx] = beta[idx] + alpha_mid * delta[idx];
            }
            if self.exact_monotonicity_min_slack(&trial)? >= 0.0 {
                alpha_lo = alpha_mid;
            } else {
                alpha_hi = alpha_mid;
            }
        }
        if alpha_lo >= 1.0 {
            return Ok(Some(1.0));
        }
        let conservative = if alpha_lo <= 0.0 {
            0.0
        } else {
            (alpha_lo * 0.999_999).clamp(0.0, 1.0)
        };
        Ok(Some(conservative))
    }
}

pub(crate) fn build_deviation_block_from_seed(
    seed: &Array1<f64>,
    cfg: &DeviationBlockConfig,
) -> Result<DeviationPrepared, String> {
    if cfg.degree != 3 {
        return Err(format!(
            "exact de-nested cubic bernoulli marginal-slope requires cubic deviation blocks (degree=3), got degree={}",
            cfg.degree
        ));
    }
    let selected = select_wiggle_basis_from_seed(
        seed.view(),
        &WiggleBlockConfig {
            degree: cfg.degree,
            num_internal_knots: cfg.num_internal_knots,
            penalty_order: cfg.penalty_order,
            double_penalty: cfg.double_penalty,
        },
        &cfg.penalty_orders,
    )?;
    let block = selected.block;
    let dim = block.design.ncols();
    let knots = selected.knots;
    let degree = selected.degree;
    let mut endpoint_points = Vec::new();
    for &knot in knots.iter() {
        if endpoint_points
            .last()
            .is_none_or(|prev: &f64| (knot - *prev).abs() > 1e-12)
        {
            endpoint_points.push(knot);
        }
    }
    if endpoint_points.len() < 2 {
        return Err(
            "deviation runtime requires at least two distinct knot breakpoints".to_string(),
        );
    }
    let mut span_left = Vec::new();
    let mut span_right = Vec::new();
    let mut span_mid = Vec::new();
    for window in endpoint_points.windows(2) {
        let left = window[0];
        let right = window[1];
        if right - left > 1e-12 {
            span_left.push(left);
            span_right.push(right);
            span_mid.push(0.5 * (left + right));
        }
    }
    if span_left.is_empty() {
        return Err("deviation runtime requires at least one active knot span".to_string());
    }
    let endpoint_points = Array1::from_vec(endpoint_points);
    let span_left_points = Array1::from_vec(span_left);
    let span_right_points = Array1::from_vec(span_right);
    let span_mid_points = Array1::from_vec(span_mid);
    let runtime = DeviationRuntime {
        knots: knots.clone(),
        degree,
        basis_dim: dim,
        monotonicity_eps: cfg.monotonicity_eps,
        span_left_value_design: monotone_wiggle_basis_with_derivative_order(
            span_left_points.view(),
            &knots,
            degree,
            0,
        )?,
        span_left_d1_design: monotone_wiggle_basis_with_derivative_order(
            span_left_points.view(),
            &knots,
            degree,
            1,
        )?,
        span_right_d1_design: monotone_wiggle_basis_with_derivative_order(
            span_right_points.view(),
            &knots,
            degree,
            1,
        )?,
        span_left_d2_design: monotone_wiggle_basis_with_derivative_order(
            span_left_points.view(),
            &knots,
            degree,
            2,
        )?,
        span_mid_d3_design: monotone_wiggle_basis_with_derivative_order(
            span_mid_points.view(),
            &knots,
            degree,
            3,
        )?,
        endpoint_points,
        span_left_points,
        span_right_points,
    };
    Ok(DeviationPrepared { block, runtime })
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
    spec.frailty.validate_for_marginal_slope()?;
    match &spec.frailty {
        FrailtySpec::None => {}
        FrailtySpec::GaussianShift { sigma_fixed } => {
            let sigma = sigma_fixed.ok_or_else(|| {
                "bernoulli-marginal-slope currently requires FrailtySpec::GaussianShift with a fixed sigma".to_string()
            })?;
            if !sigma.is_finite() || sigma < 0.0 {
                return Err(format!(
                    "bernoulli-marginal-slope requires GaussianShift sigma >= 0, got {sigma}"
                ));
            }
        }
        FrailtySpec::HazardMultiplier { .. } => unreachable!(),
    }
    // Enforce z is approximately standard normal.
    // The Gaussian decoupling identity E[Φ(a + β Z)] = Φ(a / √(1+β²))
    // holds only when Z ~ N(0, 1).  Checking mean ≈ 0, sd ≈ 1 is necessary
    // but NOT sufficient: a standardized non-normal distribution would pass
    // those checks but invalidate the closed form.  We additionally check
    // weighted skewness ≈ 0 and excess kurtosis ≈ 0 to catch the most
    // common violations (e.g. heavy tails, asymmetry, discrete scores).
    let weight_sum = spec.weights.iter().copied().sum::<f64>();
    if weight_sum.is_finite() && weight_sum > 0.0 {
        let mean = spec
            .z
            .iter()
            .zip(spec.weights.iter())
            .map(|(&zi, &wi)| wi * zi)
            .sum::<f64>()
            / weight_sum;
        let var = spec
            .z
            .iter()
            .zip(spec.weights.iter())
            .map(|(&zi, &wi)| wi * (zi - mean) * (zi - mean))
            .sum::<f64>()
            / weight_sum;
        let sd = var.sqrt();
        if mean.abs() > 0.1 || (sd - 1.0).abs() > 0.1 {
            return Err(format!(
                "bernoulli-marginal-slope requires z to already represent a latent N(0,1) score; \
                 weighted mean 0 and weighted sd 1 are necessary sanity checks. got mean={mean:.6e}, sd={sd:.6e}"
            ));
        }
        // Weighted skewness and excess kurtosis as normality gates.
        // For Z ~ N(0,1): skewness = 0, excess kurtosis = 0.
        if sd > 1e-12 {
            let skew = spec
                .z
                .iter()
                .zip(spec.weights.iter())
                .map(|(&zi, &wi)| wi * ((zi - mean) / sd).powi(3))
                .sum::<f64>()
                / weight_sum;
            let kurt = spec
                .z
                .iter()
                .zip(spec.weights.iter())
                .map(|(&zi, &wi)| wi * ((zi - mean) / sd).powi(4))
                .sum::<f64>()
                / weight_sum
                - 3.0;
            // Tolerances are generous for finite-sample noise but catch
            // obviously non-Gaussian shapes (e.g. uniform: kurt ≈ -1.2).
            if skew.abs() > 0.5 || kurt.abs() > 1.5 {
                log::warn!(
                    "bernoulli-marginal-slope: z has skewness={skew:.3} and \
                     excess kurtosis={kurt:.3}; the calibrated marginal-slope \
                     model assumes the latent score is distributed as Z~N(0,1). \
                     Results may be biased if z is not approximately normal."
                );
            }
        }
    }
    Ok(())
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
    );
    let logslope_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        logslopespec,
        &logslope_terms,
        kappa_options,
    );
    let mut values = marginal_kappa.as_array().to_vec();
    values.extend(logslope_kappa.as_array().iter());
    let mut dims = marginal_kappa.dims_per_term().to_vec();
    dims.extend(logslope_kappa.dims_per_term());
    let log_kappa0 = SpatialLogKappaCoords::new_with_dims(Array1::from_vec(values), dims.clone());
    let log_kappa_lower = SpatialLogKappaCoords::lower_bounds_aniso(&dims, kappa_options);
    let log_kappa_upper = SpatialLogKappaCoords::upper_bounds_aniso(&dims, kappa_options);
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

pub(crate) fn signed_probit_neglog_derivatives_up_to_fourth(
    signed_margin: f64,
    weight: f64,
) -> (f64, f64, f64, f64) {
    if weight == 0.0 || !signed_margin.is_finite() {
        return (0.0, 0.0, 0.0, 0.0);
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

/// Rigid probit scalar kernel: closed-form derivatives up to 4th order.
///
/// η = q·c(g) + g·z,  c(g) = √(1+g²),  s = 2y−1,  m = s·η.
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
    fn new(q: f64, g: f64, z: f64, y: f64, w: f64) -> Self {
        let s = 2.0 * y - 1.0;
        let g2 = g * g;
        let c = (1.0 + g2).sqrt();
        let c1 = g / c;
        let c_inv3 = 1.0 / (c * c * c);
        let c_inv5 = c_inv3 / (c * c);
        let c_inv7 = c_inv5 / (c * c);
        let eta = q * c + g * z;
        let m = s * eta;
        let (logcdf, _) = signed_probit_logcdf_and_mills_ratio(m);
        let (k1, k2, k3, k4) = signed_probit_neglog_derivatives_up_to_fourth(m, w);
        Self {
            logcdf,
            u1: s * k1,
            u2: k2,
            u3: s * k3,
            u4: k4,
            c1,
            c2: c_inv3,
            c3: -3.0 * g * c_inv5,
            c4: (12.0 * g2 - 3.0) * c_inv7,
            eta_q: c,
            eta_g: q * c1 + z,
        }
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
    let (d1, d2, d3, d4) = signed_probit_neglog_derivatives_up_to_fourth(x, weight);
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

fn block_slices(
    states: &[ParameterBlockState],
    has_score_warp: bool,
    has_link_dev: bool,
) -> BlockSlices {
    let mut cursor = 0usize;
    let mut block_idx = 0usize;
    let marginal = cursor..cursor + states[block_idx].beta.len();
    cursor = marginal.end;
    block_idx += 1;
    let logslope = cursor..cursor + states[block_idx].beta.len();
    cursor = logslope.end;
    block_idx += 1;
    let h = if has_score_warp {
        let range = cursor..cursor + states[block_idx].beta.len();
        cursor = range.end;
        block_idx += 1;
        Some(range)
    } else {
        None
    };
    let w = if has_link_dev {
        let range = cursor..cursor + states[block_idx].beta.len();
        cursor = range.end;
        Some(range)
    } else {
        None
    };
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
    h: Option<std::ops::Range<usize>>,
    w: Option<std::ops::Range<usize>>,
    total: usize,
}

fn primary_slices(slices: &BlockSlices) -> PrimarySlices {
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

#[inline]
fn fixed_gaussian_shift_sigma(frailty: &FrailtySpec) -> Option<f64> {
    match frailty {
        FrailtySpec::None => None,
        FrailtySpec::GaussianShift { sigma_fixed } => *sigma_fixed,
        FrailtySpec::HazardMultiplier { .. } => None,
    }
}

fn probit_frailty_scale(gaussian_frailty_sd: Option<f64>) -> f64 {
    crate::families::lognormal_kernel::ProbitFrailtyScale::new(gaussian_frailty_sd.unwrap_or(0.0)).s
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

/// Chunk size for parallel row accumulation.  Rows within a chunk are
/// processed sequentially; per-row gradient / Hessian vectors are computed
/// on the fly and discarded after accumulation.  The per-row *context*
/// (intercept, M_a, observed score-warp value) is pre-solved once in the eval
/// cache.
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

/// Maximum row-loop work proxy before downgrading to first-order:
///   n × K_pairs × primary²
const EXACT_OUTER_MAX_ROW_WORK: u64 = 2_000_000;

/// Row-loop gate for Bernoulli marginal-slope.  The memory gate is handled
/// by [`cost_gated_outer_order`] at the call site.
fn bernoulli_row_work_order(
    n_rows: usize,
    score_warp_dim: usize,
    link_dev_dim: usize,
    k_smoothing: usize,
) -> ExactOuterDerivativeOrder {
    let n = n_rows as u64;
    let k = k_smoothing as u64;
    let k_pairs = k.saturating_mul(k.saturating_add(1)) / 2;
    let primary_total = 2u64
        .saturating_add(score_warp_dim as u64)
        .saturating_add(link_dev_dim as u64);
    // Rigid models (primary=2) use RigidProbitKernel for 3rd/4th
    // derivatives: O(1) closed-form scalars per row, not O(primary^2 Q).
    // Use a much higher threshold for this case.
    let effective_primary_cost = if score_warp_dim == 0 && link_dev_dim == 0 {
        1u64 // rigid: negligible per-row cost
    } else {
        primary_total.saturating_mul(primary_total)
    };
    let row_work = n
        .saturating_mul(k_pairs)
        .saturating_mul(effective_primary_cost);
    if row_work > EXACT_OUTER_MAX_ROW_WORK {
        ExactOuterDerivativeOrder::First
    } else {
        ExactOuterDerivativeOrder::Second
    }
}

// ── RowKernel<2> implementation (rigid path only) ────────────────────

struct BernoulliRigidRowKernel {
    family: BernoulliMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    slices: BlockSlices,
}

impl BernoulliRigidRowKernel {
    fn new(family: BernoulliMarginalSlopeFamily, block_states: Vec<ParameterBlockState>) -> Self {
        let slices = block_slices(
            &block_states,
            family.score_warp.is_some(),
            family.link_dev.is_some(),
        );
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
        let q = self.block_states[0].eta[row];
        let g = self.block_states[1].eta[row];
        let k = RigidProbitKernel::new(
            q,
            g,
            self.family.z[row],
            self.family.y[row],
            self.family.weights[row],
        );
        let nll = -self.family.weights[row] * k.logcdf;
        let grad = [-k.u1 * k.eta_q, -k.u1 * k.eta_g];
        let h = k.primary_hessian(q);
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
        let q = self.block_states[0].eta[row];
        let g = self.block_states[1].eta[row];
        let k = RigidProbitKernel::new(
            q,
            g,
            self.family.z[row],
            self.family.y[row],
            self.family.weights[row],
        );
        Ok(k.third_contracted(q, dir[0], dir[1]))
    }

    fn row_fourth_contracted(
        &self,
        row: usize,
        dir_u: &[f64; 2],
        dir_v: &[f64; 2],
    ) -> Result<[[f64; 2]; 2], String> {
        let q = self.block_states[0].eta[row];
        let g = self.block_states[1].eta[row];
        let k = RigidProbitKernel::new(
            q,
            g,
            self.family.z[row],
            self.family.y[row],
            self.family.weights[row],
        );
        Ok(k.fourth_contracted(q, dir_u[0], dir_u[1], dir_v[0], dir_v[1]))
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
    derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
    cache: BernoulliMarginalSlopeExactEvalCache,
}

impl BernoulliMarginalSlopeFamily {
    #[inline]
    fn probit_frailty_scale(&self) -> f64 {
        probit_frailty_scale(self.gaussian_frailty_sd)
    }

    fn flex_score_beta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a Array1<f64>>, String> {
        if self.score_warp.is_none() {
            return Ok(None);
        }
        block_states
            .get(2)
            .map(|state| Some(&state.beta))
            .ok_or_else(|| "missing score-warp block state".to_string())
    }

    fn denested_partition_cells(
        &self,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<Vec<exact_kernel::DenestedPartitionCell>, String> {
        let score_tail = 8.0;
        let mut score_breaks = if let Some(runtime) = self.score_warp.as_ref() {
            runtime.breakpoints().to_vec()
        } else {
            vec![-8.0, 8.0]
        };
        if score_breaks.first().copied().unwrap_or(score_tail) > -score_tail {
            score_breaks.insert(0, -score_tail);
        }
        if score_breaks.last().copied().unwrap_or(-score_tail) < score_tail {
            score_breaks.push(score_tail);
        }
        let link_tail = 8.0 * (1.0 + b.abs());
        let mut link_breaks = if let Some(runtime) = self.link_dev.as_ref() {
            runtime.breakpoints().to_vec()
        } else {
            vec![a - link_tail, a + link_tail]
        };
        if link_breaks.first().copied().unwrap_or(a + link_tail) > a - link_tail {
            link_breaks.insert(0, a - link_tail);
        }
        if link_breaks.last().copied().unwrap_or(a - link_tail) < a + link_tail {
            link_breaks.push(a + link_tail);
        }
        let mut cells = exact_kernel::build_denested_partition_cells(
            a,
            b,
            &score_breaks,
            &link_breaks,
            |z| {
                if let (Some(runtime), Some(beta)) = (self.score_warp.as_ref(), beta_h) {
                    runtime.local_cubic_at(beta, z)
                } else {
                    Ok(exact_kernel::LocalSpanCubic {
                        left: -8.0,
                        right: 8.0,
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
                        left: a - 8.0 * (1.0 + b.abs()),
                        right: a + 8.0 * (1.0 + b.abs()),
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
        let cells = self.denested_partition_cells(a, slope, beta_h, beta_w)?;
        let scale = self.probit_frailty_scale();
        let mut f = -normal_cdf(marginal_eta);
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
        self.score_warp.is_some()
            || self.link_dev.is_some()
            || self.gaussian_frailty_sd.unwrap_or(0.0) > 0.0
    }

    fn validate_exact_monotonicity(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(), String> {
        if let Some(runtime) = &self.score_warp {
            runtime.monotonicity_feasible(
                &block_states[2].beta,
                "bernoulli marginal-slope score-warp deviation",
            )?;
        }
        if let Some(runtime) = &self.link_dev {
            let beta_w = block_states
                .last()
                .map(|state| &state.beta)
                .ok_or_else(|| "missing link deviation block state".to_string())?;
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
        // Evaluate the de-nested calibration equation through the exact
        // cell-partition kernel: affine cells use the closed-form anchor and
        // quartic/sextic cells use transported non-affine moments.
        //   F(a) = E[ Φ(a + b z + b Δ_h(z) + Δ_w(a + b z)) ] - Φ(q)
        // and its derivative with respect to a.
        let eval = |a: f64| -> Result<(f64, f64, f64), String> {
            self.evaluate_denested_calibration(a, marginal_eta, slope, beta_h, beta_w)
        };

        // Initial guess: closed-form for rigid probit: a₀ = q·√(1+b²).
        // When link deviation is active, upgrade to affine-link warm start:
        //   L(u) ≈ ℓ₀ + ℓ₁·u  ⟹  a = (q·√(1+ℓ₁²b²) − ℓ₀) / ℓ₁
        let a_rigid = marginal_eta * (1.0 + slope * slope).sqrt();
        let mut a = if beta_w.is_some() {
            let v = Array1::from_vec(vec![a_rigid]);
            let (l_val, l_d1) = self.link_terms_value_d1(&v, beta_w)?;
            let ell1 = l_d1[0];
            if ell1 > 1e-8 {
                let ell0 = l_val[0] - ell1 * a_rigid;
                (marginal_eta * (1.0 + ell1 * ell1 * slope * slope).sqrt() - ell0) / ell1
            } else {
                a_rigid
            }
        } else {
            a_rigid
        };

        // First bracket the unique monotone root. The previous implementation
        // took unconstrained Newton steps before a sign bracket existed, which
        // is unstable in the extreme probit tail because F(a) saturates while
        // F'(a) is tiny, producing huge oscillatory jumps.
        let (f_init, f_deriv_init, _) = eval(a)?;
        if f_init == 0.0 {
            return Ok((a, f_deriv_init));
        }

        let mut lo;
        let mut hi;
        let f_lo;
        let f_hi;
        let mut bracket_step = (0.25 * (1.0 + a.abs())).max(1.0);
        if f_init < 0.0 {
            lo = a;
            let mut found = None;
            for _ in 0..64 {
                let hi_try = a + bracket_step;
                let (f_try, _, _) = eval(hi_try)?;
                if f_try >= 0.0 {
                    found = Some((hi_try, f_try));
                    break;
                }
                lo = hi_try;
                bracket_step *= 2.0;
            }
            let Some((hi_found, f_hi_found)) = found else {
                return Err(format!(
                    "bernoulli marginal-slope intercept solve failed to bracket root from below: \
                     q={marginal_eta:.6}, slope={slope:.6}, last_a={lo:.6}"
                ));
            };
            hi = hi_found;
            let (f_lo_val, _, _) = eval(lo)?;
            f_lo = f_lo_val;
            f_hi = f_hi_found;
        } else {
            hi = a;
            let mut found = None;
            for _ in 0..64 {
                let lo_try = a - bracket_step;
                let (f_try, _, _) = eval(lo_try)?;
                if f_try <= 0.0 {
                    found = Some((lo_try, f_try));
                    break;
                }
                hi = lo_try;
                bracket_step *= 2.0;
            }
            let Some((lo_found, f_lo_found)) = found else {
                return Err(format!(
                    "bernoulli marginal-slope intercept solve failed to bracket root from above: \
                     q={marginal_eta:.6}, slope={slope:.6}, last_a={hi:.6}"
                ));
            };
            lo = lo_found;
            let (f_hi_val, _, _) = eval(hi)?;
            f_hi = f_hi_val;
            f_lo = f_lo_found;
        }

        let (mut best_a, mut best_f, mut best_deriv) = if f_init.abs() <= f_lo.abs().min(f_hi.abs())
        {
            (a, f_init, f_deriv_init)
        } else if f_lo.abs() <= f_hi.abs() {
            let (_, d_lo, _) = eval(lo)?;
            (lo, f_lo, d_lo)
        } else {
            let (_, d_hi, _) = eval(hi)?;
            (hi, f_hi, d_hi)
        };

        a = a.clamp(lo, hi);
        for _ in 0..48 {
            let (f_val, f_deriv, _) = eval(a)?;
            if f_val.abs() < best_f.abs() {
                best_a = a;
                best_f = f_val;
                best_deriv = f_deriv;
            }
            if f_val.abs() <= 1e-10 {
                best_a = a;
                best_f = f_val;
                best_deriv = f_deriv;
                break;
            }

            if f_val < 0.0 {
                lo = a;
            } else {
                hi = a;
            }

            if (hi - lo) <= 1e-12 * (1.0 + a.abs()) {
                let mid = 0.5 * (lo + hi);
                let (f_mid, f_mid_deriv, _) = eval(mid)?;
                if f_mid.abs() < best_f.abs() {
                    best_a = mid;
                    best_f = f_mid;
                    best_deriv = f_mid_deriv;
                }
                break;
            }

            let midpoint = 0.5 * (lo + hi);
            let a_newton = if f_deriv.is_finite() && f_deriv > 0.0 {
                let cand = a - f_val / f_deriv;
                if cand > lo && cand < hi {
                    cand
                } else {
                    midpoint
                }
            } else {
                midpoint
            };
            a = a_newton;
        }

        // Adaptive tolerance: for extreme slopes the intercept equation
        // becomes numerically flat and 1e-8 absolute precision is not
        // achievable.  Accept the best bracketed solution when the
        // relative residual is small.
        let target = normal_cdf(marginal_eta);
        let abs_tol = 1e-8_f64.max(1e-4 * target.abs());
        if best_f.abs() > abs_tol {
            return Err(format!(
                "bernoulli marginal-slope intercept solve failed: \
                 residual={best_f:.3e} at a={best_a:.6}, target Φ(q)={target:.6}"
            ));
        }
        if !best_deriv.is_finite() || best_deriv <= 0.0 {
            return Err(format!(
                "bernoulli marginal-slope intercept solve produced non-positive calibration derivative: \
                 M_a={best_deriv:.3e} at a={best_a:.6}"
            ));
        }
        Ok((best_a, best_deriv))
    }

    fn build_row_exact_context(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<BernoulliMarginalSlopeRowExactContext, String> {
        let marginal_eta = block_states[0].eta[row];
        // The log-slope block now parameterizes the signed slope directly.
        let slope = block_states[1].eta[row];
        let beta_w = if self.link_dev.is_some() {
            block_states.last().map(|state| &state.beta)
        } else {
            None
        };
        let (intercept, m_a) = if self.flex_active() {
            self.solve_row_intercept_base(
                marginal_eta,
                slope,
                self.flex_score_beta(block_states)?,
                beta_w,
            )?
        } else {
            (marginal_eta * (1.0 + slope * slope).sqrt(), f64::NAN)
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
        let slices = block_slices(
            block_states,
            self.score_warp.is_some(),
            self.link_dev.is_some(),
        );
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
        out[0] = self
            .marginal_design
            .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
        out[1] = self
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

    fn psi_design_row_vector(
        &self,
        row: usize,
        deriv: &crate::custom_family::CustomFamilyBlockPsiDerivative,
        total_rows: usize,
        p: usize,
        label: &str,
    ) -> Result<Array1<f64>, String> {
        let action = CustomFamilyPsiDesignAction::from_first_derivative(
            deriv,
            total_rows,
            p,
            0..total_rows,
            label,
        )
        .ok();
        first_psi_linear_map(action.as_ref(), &deriv.x_psi, total_rows, p).row_vector(row)
    }

    fn psi_second_design_row_vector(
        &self,
        row: usize,
        deriv_i: &crate::custom_family::CustomFamilyBlockPsiDerivative,
        deriv_j: &crate::custom_family::CustomFamilyBlockPsiDerivative,
        local_j: usize,
        total_rows: usize,
        p: usize,
        label: &str,
    ) -> Result<Array1<f64>, String> {
        let action = CustomFamilyPsiSecondDesignAction::from_second_derivative(
            deriv_i,
            deriv_j,
            total_rows,
            p,
            0..total_rows,
            label,
        )?;
        let dense = deriv_i
            .x_psi_psi
            .as_ref()
            .and_then(|rows| rows.get(local_j));
        second_psi_linear_map(action.as_ref(), dense, total_rows, p).row_vector(row)
    }

    fn row_primary_psi_direction(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        primary: &PrimarySlices,
    ) -> Result<Option<Array1<f64>>, String> {
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let mut out = Array1::<f64>::zeros(primary.total);
        match block_idx {
            0 => {
                let x_row = self.psi_design_row_vector(
                    row,
                    deriv,
                    self.y.len(),
                    self.marginal_design.ncols(),
                    "BernoulliMarginalSlopeFamily marginal",
                )?;
                out[0] = x_row.dot(&block_states[0].beta);
            }
            1 => {
                let x_row = self.psi_design_row_vector(
                    row,
                    deriv,
                    self.y.len(),
                    self.logslope_design.ncols(),
                    "BernoulliMarginalSlopeFamily log-slope",
                )?;
                out[1] = x_row.dot(&block_states[1].beta);
            }
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi direction only supports spatial marginal/logslope blocks, got block {block_idx}"
                ));
            }
        }
        Ok(Some(out))
    }

    fn row_primary_psi_action_on_direction(
        &self,
        row: usize,
        slices: &BlockSlices,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        primary: &PrimarySlices,
    ) -> Result<Option<Array1<f64>>, String> {
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let mut out = Array1::<f64>::zeros(primary.total);
        match block_idx {
            0 => {
                let x_row = self.psi_design_row_vector(
                    row,
                    deriv,
                    self.y.len(),
                    self.marginal_design.ncols(),
                    "BernoulliMarginalSlopeFamily marginal",
                )?;
                out[0] = x_row.dot(&d_beta_flat.slice(s![slices.marginal.clone()]).to_owned())
            }
            1 => {
                let x_row = self.psi_design_row_vector(
                    row,
                    deriv,
                    self.y.len(),
                    self.logslope_design.ncols(),
                    "BernoulliMarginalSlopeFamily log-slope",
                )?;
                out[1] = x_row.dot(&d_beta_flat.slice(s![slices.logslope.clone()]).to_owned())
            }
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi action only supports marginal/logslope blocks, got block {block_idx}"
                ));
            }
        }
        Ok(Some(out))
    }

    fn row_primary_psi_second_direction(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        primary: &PrimarySlices,
    ) -> Result<Option<Array1<f64>>, String> {
        let Some((block_i, local_i)) = self.resolve_psi_location(derivative_blocks, psi_i) else {
            return Ok(None);
        };
        let Some((block_j, local_j)) = self.resolve_psi_location(derivative_blocks, psi_j) else {
            return Ok(None);
        };
        if block_i != block_j {
            return Ok(Some(Array1::<f64>::zeros(primary.total)));
        }
        let deriv_i = &derivative_blocks[block_i][local_i];
        let mut out = Array1::<f64>::zeros(primary.total);
        match block_i {
            0 => {
                let x_row = self.psi_second_design_row_vector(
                    row,
                    deriv_i,
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    self.y.len(),
                    self.marginal_design.ncols(),
                    "BernoulliMarginalSlopeFamily marginal",
                )?;
                out[0] = x_row.dot(&block_states[0].beta);
            }
            1 => {
                let x_row = self.psi_second_design_row_vector(
                    row,
                    deriv_i,
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    self.y.len(),
                    self.logslope_design.ncols(),
                    "BernoulliMarginalSlopeFamily log-slope",
                )?;
                out[1] = x_row.dot(&block_states[1].beta);
            }
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi second direction only supports marginal/logslope blocks, got block {block_i}"
                ));
            }
        }
        Ok(Some(out))
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
                .axpy_row_into(row, primary_vec[0], &mut marginal)?;
        }
        {
            let mut logslope = out.slice_mut(s![slices.logslope.clone()]);
            self.logslope_design
                .axpy_row_into(row, primary_vec[1], &mut logslope)?;
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

    fn block_psi_row(
        &self,
        row: usize,
        slices: &BlockSlices,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<BlockPsiRow>, String> {
        let Some((block_idx, local_idx)) = self.resolve_psi_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let (local_vec, range) = match block_idx {
            0 => (
                self.psi_design_row_vector(
                    row,
                    deriv,
                    self.y.len(),
                    self.marginal_design.ncols(),
                    "BernoulliMarginalSlopeFamily marginal",
                )?,
                slices.marginal.clone(),
            ),
            1 => (
                self.psi_design_row_vector(
                    row,
                    deriv,
                    self.y.len(),
                    self.logslope_design.ncols(),
                    "BernoulliMarginalSlopeFamily log-slope",
                )?,
                slices.logslope.clone(),
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi embedding only supports marginal/logslope blocks, got block {block_idx}"
                ));
            }
        };
        Ok(Some(BlockPsiRow {
            block_idx,
            range,
            local_vec,
        }))
    }

    fn block_psi_second_row(
        &self,
        row: usize,
        slices: &BlockSlices,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<BlockPsiRow>, String> {
        let Some((block_i, local_i)) = self.resolve_psi_location(derivative_blocks, psi_i) else {
            return Ok(None);
        };
        let Some((block_j, local_j)) = self.resolve_psi_location(derivative_blocks, psi_j) else {
            return Ok(None);
        };
        if block_i != block_j {
            return Ok(None);
        }
        let deriv_i = &derivative_blocks[block_i][local_i];
        let (local_vec, range) = match block_i {
            0 => (
                self.psi_second_design_row_vector(
                    row,
                    deriv_i,
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    self.y.len(),
                    self.marginal_design.ncols(),
                    "BernoulliMarginalSlopeFamily marginal",
                )?,
                slices.marginal.clone(),
            ),
            1 => (
                self.psi_second_design_row_vector(
                    row,
                    deriv_i,
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    self.y.len(),
                    self.logslope_design.ncols(),
                    "BernoulliMarginalSlopeFamily log-slope",
                )?,
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
        if self.flex_active() {
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
        // Rigid path: closed-form eta = q*c(g) + g*z.
        // primary.total == 2 (q at 0, g at 1), no h/w blocks.
        let q = block_states[0].eta[row];
        let g = block_states[1].eta[row];
        let yi = self.y[row];
        let wi = self.weights[row];
        let zi = self.z[row];
        let s = 2.0 * yi - 1.0;

        let g2 = g * g;
        let c = (1.0 + g2).sqrt();
        let c1 = g / c;
        let c2 = 1.0 / (c * c * c);
        let eta = q * c + g * zi;
        let m = s * eta;

        let (logcdf, lambda) = signed_probit_logcdf_and_mills_ratio(m);
        let neglog = -wi * logcdf;

        let u1 = -wi * s * lambda;
        let u2 = wi * lambda * (m + lambda);

        let deta_dq = c;
        let deta_dg = q * c1 + zi;

        let mut grad = Array1::<f64>::zeros(2);
        grad[0] = u1 * deta_dq;
        grad[1] = u1 * deta_dg;

        let mut hess = Array2::<f64>::zeros((2, 2));
        hess[[0, 0]] = u2 * deta_dq * deta_dq;
        hess[[0, 1]] = u2 * deta_dq * deta_dg;
        hess[[1, 0]] = hess[[0, 1]];
        hess[[1, 1]] = u2 * deta_dg * deta_dg + u1 * q * c2;

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
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = if self.link_dev.is_some() {
            block_states.last().map(|st| &st.beta)
        } else {
            None
        };
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
        let phi_q = normal_pdf(q);
        let inv_ma = 1.0 / f_a;
        let h_range = primary.h.as_ref();
        let w_range = primary.w.as_ref();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();
        let scale = self.probit_frailty_scale();

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
            let z_mid = 0.5 * (cell.left + cell.right);
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
                for u in 1..r {
                    for v in u..r {
                        let second_coeff = if u == 1 {
                            coeff_bu[v]
                        } else if v == 1 {
                            coeff_bu[u]
                        } else {
                            [0.0; 4]
                        };
                        let val = exact::cell_second_derivative_from_moments(
                            cell,
                            &coeff_u[u],
                            &coeff_u[v],
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

        f_u[0] = -phi_q;
        if need_hessian {
            f_uv[[0, 0]] = q * phi_q;
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

        let rho = &mut scratch.rho;
        let tau = &mut scratch.tau;
        rho.fill(0.0);
        tau.fill(0.0);
        rho[1] = eval_coeff4_at(&obs.dc_db, z_obs);
        tau[1] = eval_coeff4_at(&obs.dc_dab, z_obs);
        if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
            for local_idx in 0..h_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, z_obs)?;
                rho[h_range.start + local_idx] = eval_coeff4_at(
                    &scale_coeff4(exact::score_basis_cell_coefficients(basis_span, b), scale),
                    z_obs,
                );
            }
        }
        if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                rho[w_range.start + local_idx] = eval_coeff4_at(
                    &scale_coeff4(exact::link_basis_cell_coefficients(basis_span, a, b), scale),
                    z_obs,
                );
                tau[w_range.start + local_idx] = eval_coeff4_at(
                    &scale_coeff4(
                        exact::link_basis_cell_coefficient_partials(basis_span, a, b).0,
                        scale,
                    ),
                    z_obs,
                );
            }
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
                    let mut r_uv = 0.0;
                    if u == 1 && v == 1 {
                        r_uv = eval_coeff4_at(&obs.dc_dbb, z_obs);
                    } else if u == 1 {
                        if let Some(h_range) = h_range {
                            if v >= h_range.start && v < h_range.end {
                                let local_idx = v - h_range.start;
                                let runtime = score_runtime
                                    .ok_or_else(|| "missing score-warp runtime".to_string())?;
                                let basis_span = runtime.basis_cubic_at(local_idx, z_obs)?;
                                r_uv = eval_coeff4_at(
                                    &scale_coeff4(
                                        exact::score_basis_cell_coefficients(basis_span, 1.0),
                                        scale,
                                    ),
                                    z_obs,
                                );
                            }
                        }
                        if let Some(w_range) = w_range {
                            if v >= w_range.start && v < w_range.end {
                                let local_idx = v - w_range.start;
                                let runtime = link_runtime
                                    .ok_or_else(|| "missing link runtime".to_string())?;
                                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                                r_uv = eval_coeff4_at(
                                    &scale_coeff4(
                                        exact::link_basis_cell_coefficient_partials(
                                            basis_span, a, b,
                                        )
                                        .1,
                                        scale,
                                    ),
                                    z_obs,
                                );
                            }
                        }
                    } else if v == 1 {
                        if let Some(h_range) = h_range {
                            if u >= h_range.start && u < h_range.end {
                                let local_idx = u - h_range.start;
                                let runtime = score_runtime
                                    .ok_or_else(|| "missing score-warp runtime".to_string())?;
                                let basis_span = runtime.basis_cubic_at(local_idx, z_obs)?;
                                r_uv = eval_coeff4_at(
                                    &scale_coeff4(
                                        exact::score_basis_cell_coefficients(basis_span, 1.0),
                                        scale,
                                    ),
                                    z_obs,
                                );
                            }
                        }
                        if let Some(w_range) = w_range {
                            if u >= w_range.start && u < w_range.end {
                                let local_idx = u - w_range.start;
                                let runtime = link_runtime
                                    .ok_or_else(|| "missing link runtime".to_string())?;
                                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                                r_uv = eval_coeff4_at(
                                    &scale_coeff4(
                                        exact::link_basis_cell_coefficient_partials(
                                            basis_span, a, b,
                                        )
                                        .1,
                                        scale,
                                    ),
                                    z_obs,
                                );
                            }
                        }
                    }
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
    ) -> Array1<f64> {
        let mut point = Array1::<f64>::zeros(primary.total);
        point[0] = block_states[0].eta[row];
        point[1] = block_states[1].eta[row];
        if let Some(h_range) = primary.h.as_ref() {
            point
                .slice_mut(s![h_range.start..h_range.end])
                .assign(&block_states[2].beta);
        }
        if let Some(w_range) = primary.w.as_ref() {
            let beta_w = block_states.last().expect("missing link deviation beta");
            point
                .slice_mut(s![w_range.start..w_range.end])
                .assign(&beta_w.beta);
        }
        point
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
        (point[0], point[1], beta_h, beta_w)
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
            left: -8.0,
            right: 8.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        };
        let zero_link_span = exact::LocalSpanCubic {
            left: a - 8.0 * (1.0 + b.abs()),
            right: a + 8.0 * (1.0 + b.abs()),
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
    /// Rigid path uses the closed-form kernel. The flexible exact de-nested
    /// path contracts the exact cell-moment kernel analytically.
    fn row_primary_third_contracted_recompute(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if !self.flex_active() {
            let q = block_states[0].eta[row];
            let g = block_states[1].eta[row];
            let kern = RigidProbitKernel::new(q, g, self.z[row], self.y[row], self.weights[row]);
            let t = kern.third_contracted(q, dir[0], dir[1]);
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
        let point = self.primary_point_from_block_states(row, block_states, primary);
        let (q, b, beta_h_owned, beta_w_owned) = self.primary_point_components(&point, primary);
        let beta_h = beta_h_owned.as_ref();
        let beta_w = beta_w_owned.as_ref();
        let a = row_ctx.intercept;
        let r = primary.total;
        let phi_q = normal_pdf(q);
        let h_range = primary.h.as_ref();
        let w_range = primary.w.as_ref();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();
        let scale = self.probit_frailty_scale();

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
            let z_mid = 0.5 * (cell.left + cell.right);
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

            f_a += exact::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            f_aa += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &dc_daa,
                &state.moments,
            )?;

            let mut coeff_dir = [0.0; 4];
            let mut coeff_a_dir = [0.0; 4];
            let mut coeff_b_dir = [0.0; 4];
            let mut coeff_aa_dir = [0.0; 4];
            for u in 1..r {
                add_scaled_coeff4(&mut coeff_dir, &coeff_u[u], dir[u]);
                add_scaled_coeff4(&mut coeff_a_dir, &coeff_au[u], dir[u]);
                add_scaled_coeff4(&mut coeff_b_dir, &coeff_bu[u], dir[u]);
                add_scaled_coeff4(&mut coeff_aa_dir, &coeff_aau[u], dir[u]);
                f_u[u] += exact::cell_first_derivative_from_moments(&coeff_u[u], &state.moments)?;
                f_au[u] += exact::cell_second_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_u[u],
                    &coeff_au[u],
                    &state.moments,
                )?;
            }

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

            let dir_b = dir[1];
            let mut coeff_u_dir = vec![[0.0; 4]; r];
            let mut coeff_au_dir = vec![[0.0; 4]; r];
            coeff_u_dir[1] = coeff_b_dir;
            coeff_au_dir[1] = {
                let mut out = [0.0; 4];
                add_scaled_coeff4(&mut out, &coeff_abu[1], dir_b);
                if let Some(w_range) = w_range {
                    for idx in w_range.clone() {
                        add_scaled_coeff4(&mut out, &coeff_abu[idx], dir[idx]);
                    }
                }
                out
            };

            if let Some(h_range) = h_range {
                for idx in h_range.clone() {
                    let mut out = [0.0; 4];
                    add_scaled_coeff4(&mut out, &coeff_bu[idx], dir_b);
                    coeff_u_dir[idx] = out;
                }
            }
            if let Some(w_range) = w_range {
                for idx in w_range.clone() {
                    let mut up = [0.0; 4];
                    add_scaled_coeff4(&mut up, &coeff_bu[idx], dir_b);
                    coeff_u_dir[idx] = up;

                    let mut aup = [0.0; 4];
                    add_scaled_coeff4(&mut aup, &coeff_abu[idx], dir_b);
                    coeff_au_dir[idx] = aup;
                }
            }

            for u in 1..r {
                f_au_dir[u] += exact::cell_third_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_u[u],
                    &coeff_dir,
                    &coeff_au[u],
                    &coeff_a_dir,
                    &coeff_u_dir[u],
                    &coeff_au_dir[u],
                    &state.moments,
                )?;
            }

            for u in 1..r {
                for v in u..r {
                    let second_coeff = if u == 1 {
                        coeff_bu[v]
                    } else if v == 1 {
                        coeff_bu[u]
                    } else {
                        [0.0; 4]
                    };
                    let val = exact::cell_second_derivative_from_moments(
                        cell,
                        &coeff_u[u],
                        &coeff_u[v],
                        &second_coeff,
                        &state.moments,
                    )?;
                    f_uv[[u, v]] += val;
                    if u != v {
                        f_uv[[v, u]] += val;
                    }

                    let third_coeff = if u == 1 && v == 1 {
                        let mut out = [0.0; 4];
                        add_scaled_coeff4(&mut out, &coeff_bbu[1], dir_b);
                        if let Some(w_range) = w_range {
                            for idx in w_range.clone() {
                                add_scaled_coeff4(&mut out, &coeff_bbu[idx], dir[idx]);
                            }
                        }
                        out
                    } else if u == 1 {
                        if let Some(w_range) = w_range {
                            if w_range.contains(&v) {
                                let mut out = [0.0; 4];
                                add_scaled_coeff4(&mut out, &coeff_bbu[v], dir_b);
                                out
                            } else {
                                [0.0; 4]
                            }
                        } else {
                            [0.0; 4]
                        }
                    } else if v == 1 {
                        if let Some(w_range) = w_range {
                            if w_range.contains(&u) {
                                let mut out = [0.0; 4];
                                add_scaled_coeff4(&mut out, &coeff_bbu[u], dir_b);
                                out
                            } else {
                                [0.0; 4]
                            }
                        } else {
                            [0.0; 4]
                        }
                    } else {
                        [0.0; 4]
                    };
                    let dir_val = exact::cell_third_derivative_from_moments(
                        cell,
                        &coeff_u[u],
                        &coeff_u[v],
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

        f_u[0] = -phi_q;
        f_uv[[0, 0]] = q * phi_q;
        f_uv_dir[[0, 0]] = dir[0] * (1.0 - q * q) * phi_q;

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

        let g_a = eval_coeff4_at(&obs.dc_da, z_obs);
        let g_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let g_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let mut g_u = Array1::<f64>::zeros(r);
        let mut g_au = Array1::<f64>::zeros(r);
        let mut g_aau = Array1::<f64>::zeros(r);
        let mut g_uv = Array2::<f64>::zeros((r, r));
        let mut g_auv = Array2::<f64>::zeros((r, r));
        for u in 1..r {
            g_u[u] = eval_coeff4_at(&g_u_fixed[u], z_obs);
            g_au[u] = eval_coeff4_at(&g_au_fixed[u], z_obs);
            g_aau[u] = eval_coeff4_at(&g_aau_fixed[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let second_coeff = if u == 1 {
                    g_bu_fixed[v]
                } else if v == 1 {
                    g_bu_fixed[u]
                } else {
                    [0.0; 4]
                };
                let val = eval_coeff4_at(&second_coeff, z_obs);
                g_uv[[u, v]] = val;
                g_uv[[v, u]] = val;

                let third_coeff = if u == 1 && v == 1 {
                    g_abu_fixed[1]
                } else if u == 1 {
                    if let Some(w_range) = w_range {
                        if w_range.contains(&v) {
                            g_abu_fixed[v]
                        } else {
                            [0.0; 4]
                        }
                    } else {
                        [0.0; 4]
                    }
                } else if v == 1 {
                    if let Some(w_range) = w_range {
                        if w_range.contains(&u) {
                            g_abu_fixed[u]
                        } else {
                            [0.0; 4]
                        }
                    } else {
                        [0.0; 4]
                    }
                } else {
                    [0.0; 4]
                };
                let third_val = eval_coeff4_at(&third_coeff, z_obs);
                g_auv[[u, v]] = third_val;
                g_auv[[v, u]] = third_val;
            }
        }

        let mut g_dir_fixed = [0.0; 4];
        let mut g_a_dir_fixed = [0.0; 4];
        let mut g_aa_dir_fixed = [0.0; 4];
        let mut g_u_dir_fixed = vec![[0.0; 4]; r];
        let mut g_au_dir_fixed = vec![[0.0; 4]; r];
        for u in 1..r {
            add_scaled_coeff4(&mut g_dir_fixed, &g_u_fixed[u], dir[u]);
            add_scaled_coeff4(&mut g_a_dir_fixed, &g_au_fixed[u], dir[u]);
            add_scaled_coeff4(&mut g_aa_dir_fixed, &g_aau_fixed[u], dir[u]);
        }
        let g_a_dir = eval_coeff4_at(&g_a_dir_fixed, z_obs);
        let g_aa_dir = eval_coeff4_at(&g_aa_dir_fixed, z_obs);

        g_u_dir_fixed[1] = {
            let mut out = [0.0; 4];
            add_scaled_coeff4(&mut out, &g_bu_fixed[1], dir[1]);
            if let Some(h_range) = h_range {
                for idx in h_range.clone() {
                    add_scaled_coeff4(&mut out, &g_bu_fixed[idx], dir[idx]);
                }
            }
            if let Some(w_range) = w_range {
                for idx in w_range.clone() {
                    add_scaled_coeff4(&mut out, &g_bu_fixed[idx], dir[idx]);
                }
            }
            out
        };
        if let Some(h_range) = h_range {
            for idx in h_range.clone() {
                let mut out = [0.0; 4];
                add_scaled_coeff4(&mut out, &g_bu_fixed[idx], dir[1]);
                g_u_dir_fixed[idx] = out;
            }
        }
        if let Some(w_range) = w_range {
            for idx in w_range.clone() {
                let mut out = [0.0; 4];
                add_scaled_coeff4(&mut out, &g_bu_fixed[idx], dir[1]);
                g_u_dir_fixed[idx] = out;

                let mut aout = [0.0; 4];
                add_scaled_coeff4(&mut aout, &g_abu_fixed[idx], dir[1]);
                g_au_dir_fixed[idx] = aout;
            }
        }
        g_au_dir_fixed[1] = {
            let mut out = [0.0; 4];
            add_scaled_coeff4(&mut out, &g_abu_fixed[1], dir[1]);
            if let Some(w_range) = w_range {
                for idx in w_range.clone() {
                    add_scaled_coeff4(&mut out, &g_abu_fixed[idx], dir[idx]);
                }
            }
            out
        };

        let mut g_u_dir = Array1::<f64>::zeros(r);
        let mut g_uv_dir = Array2::<f64>::zeros((r, r));
        for u in 1..r {
            g_u_dir[u] = eval_coeff4_at(&g_u_dir_fixed[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let third_coeff = if u == 1 && v == 1 {
                    let mut out = [0.0; 4];
                    add_scaled_coeff4(&mut out, &g_bbu_fixed[1], dir[1]);
                    if let Some(w_range) = w_range {
                        for idx in w_range.clone() {
                            add_scaled_coeff4(&mut out, &g_bbu_fixed[idx], dir[idx]);
                        }
                    }
                    out
                } else if u == 1 {
                    if let Some(w_range) = w_range {
                        if w_range.contains(&v) {
                            let mut out = [0.0; 4];
                            add_scaled_coeff4(&mut out, &g_bbu_fixed[v], dir[1]);
                            out
                        } else {
                            [0.0; 4]
                        }
                    } else {
                        [0.0; 4]
                    }
                } else if v == 1 {
                    if let Some(w_range) = w_range {
                        if w_range.contains(&u) {
                            let mut out = [0.0; 4];
                            add_scaled_coeff4(&mut out, &g_bbu_fixed[u], dir[1]);
                            out
                        } else {
                            [0.0; 4]
                        }
                    } else {
                        [0.0; 4]
                    }
                } else {
                    [0.0; 4]
                };
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
        let eta_dir = eta_u.dot(dir);
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
        let (k1, k2, k3, _) = signed_probit_neglog_derivatives_up_to_fourth(m, w_i);
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
    /// Rigid path uses the closed-form kernel. The flexible exact de-nested
    /// path contracts the exact cell-moment kernel analytically.
    fn row_primary_fourth_contracted_recompute(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if !self.flex_active() {
            let q = block_states[0].eta[row];
            let g = block_states[1].eta[row];
            let kern = RigidProbitKernel::new(q, g, self.z[row], self.y[row], self.weights[row]);
            let f = kern.fourth_contracted(q, dir_u[0], dir_u[1], dir_v[0], dir_v[1]);
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
        let point = self.primary_point_from_block_states(row, block_states, primary);
        let (q, b, beta_h_owned, beta_w_owned) = self.primary_point_components(&point, primary);
        let beta_h = beta_h_owned.as_ref();
        let beta_w = beta_w_owned.as_ref();
        let a = row_ctx.intercept;
        let r = primary.total;
        let phi_q = normal_pdf(q);
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
        let dir_u_b = dir_u[1];
        let dir_v_b = dir_v[1];

        let cells = self.denested_partition_cells(a, b, beta_h, beta_w)?;
        for partition_cell in cells {
            let cell = partition_cell.cell;
            let z_mid = 0.5 * (cell.left + cell.right);
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

            f_a += exact::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            f_aa += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &dc_daa,
                &state.moments,
            )?;

            let mut coeff_dir_u = [0.0; 4];
            let mut coeff_dir_v = [0.0; 4];
            let mut coeff_a_dir_u = [0.0; 4];
            let mut coeff_a_dir_v = [0.0; 4];
            let mut coeff_b_dir_u = [0.0; 4];
            let mut coeff_b_dir_v = [0.0; 4];
            let mut coeff_aa_dir_u = [0.0; 4];
            let mut coeff_aa_dir_v = [0.0; 4];
            for u in 1..r {
                add_scaled_coeff4(&mut coeff_dir_u, &coeff_u[u], dir_u[u]);
                add_scaled_coeff4(&mut coeff_dir_v, &coeff_u[u], dir_v[u]);
                add_scaled_coeff4(&mut coeff_a_dir_u, &coeff_au[u], dir_u[u]);
                add_scaled_coeff4(&mut coeff_a_dir_v, &coeff_au[u], dir_v[u]);
                add_scaled_coeff4(&mut coeff_b_dir_u, &coeff_bu[u], dir_u[u]);
                add_scaled_coeff4(&mut coeff_b_dir_v, &coeff_bu[u], dir_v[u]);
                add_scaled_coeff4(&mut coeff_aa_dir_u, &coeff_aau[u], dir_u[u]);
                add_scaled_coeff4(&mut coeff_aa_dir_v, &coeff_aau[u], dir_v[u]);
                f_u[u] += exact::cell_first_derivative_from_moments(&coeff_u[u], &state.moments)?;
                f_au[u] += exact::cell_second_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_u[u],
                    &coeff_au[u],
                    &state.moments,
                )?;
            }

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

            let mut coeff_dir_uv = [0.0; 4];
            if let Some(h_range) = h_range {
                for idx in h_range.clone() {
                    add_scaled_coeff4(
                        &mut coeff_dir_uv,
                        &coeff_bu[idx],
                        dir_u_b * dir_v[idx] + dir_v_b * dir_u[idx],
                    );
                }
            }
            if let Some(w_range) = w_range {
                for idx in w_range.clone() {
                    add_scaled_coeff4(
                        &mut coeff_dir_uv,
                        &coeff_bu[idx],
                        dir_u_b * dir_v[idx] + dir_v_b * dir_u[idx],
                    );
                }
            }
            add_scaled_coeff4(&mut coeff_dir_uv, &coeff_bu[1], dir_u_b * dir_v_b);

            let mut coeff_a_dir_uv = [0.0; 4];
            if let Some(w_range) = w_range {
                for idx in w_range.clone() {
                    add_scaled_coeff4(
                        &mut coeff_a_dir_uv,
                        &coeff_abu[idx],
                        dir_u_b * dir_v[idx] + dir_v_b * dir_u[idx],
                    );
                }
            }

            let mut coeff_aa_dir_uv = [0.0; 4];
            if let Some(w_range) = w_range {
                for idx in w_range.clone() {
                    add_scaled_coeff4(
                        &mut coeff_aa_dir_uv,
                        &coeff_aabu[idx],
                        dir_u_b * dir_v[idx] + dir_v_b * dir_u[idx],
                    );
                }
            }

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

            coeff_u_dir_u[1] = coeff_b_dir_u;
            coeff_u_dir_v[1] = coeff_b_dir_v;
            coeff_u_dir_uv[1] = coeff_dir_uv;
            coeff_au_dir_u[1] = {
                let mut out = [0.0; 4];
                add_scaled_coeff4(&mut out, &coeff_abu[1], dir_u_b);
                if let Some(w_range) = w_range {
                    for idx in w_range.clone() {
                        add_scaled_coeff4(&mut out, &coeff_abu[idx], dir_u[idx]);
                    }
                }
                out
            };
            coeff_au_dir_v[1] = {
                let mut out = [0.0; 4];
                add_scaled_coeff4(&mut out, &coeff_abu[1], dir_v_b);
                if let Some(w_range) = w_range {
                    for idx in w_range.clone() {
                        add_scaled_coeff4(&mut out, &coeff_abu[idx], dir_v[idx]);
                    }
                }
                out
            };
            coeff_au_dir_uv[1] = {
                let mut out = [0.0; 4];
                if let Some(w_range) = w_range {
                    for idx in w_range.clone() {
                        add_scaled_coeff4(
                            &mut out,
                            &coeff_abbu[idx],
                            dir_u_b * dir_v[idx] + dir_v_b * dir_u[idx],
                        );
                    }
                }
                out
            };

            if let Some(h_range) = h_range {
                for idx in h_range.clone() {
                    add_scaled_coeff4(&mut coeff_u_dir_u[idx], &coeff_bu[idx], dir_u_b);
                    add_scaled_coeff4(&mut coeff_u_dir_v[idx], &coeff_bu[idx], dir_v_b);
                }
            }
            if let Some(w_range) = w_range {
                for idx in w_range.clone() {
                    add_scaled_coeff4(&mut coeff_u_dir_u[idx], &coeff_bu[idx], dir_u_b);
                    add_scaled_coeff4(&mut coeff_u_dir_v[idx], &coeff_bu[idx], dir_v_b);
                    add_scaled_coeff4(&mut coeff_u_dir_uv[idx], &coeff_bbu[idx], dir_u_b * dir_v_b);

                    add_scaled_coeff4(&mut coeff_au_dir_u[idx], &coeff_abu[idx], dir_u_b);
                    add_scaled_coeff4(&mut coeff_au_dir_v[idx], &coeff_abu[idx], dir_v_b);
                    add_scaled_coeff4(
                        &mut coeff_au_dir_uv[idx],
                        &coeff_abbu[idx],
                        dir_u_b * dir_v_b,
                    );
                }
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
                    let second_coeff = if u == 1 {
                        coeff_bu[v]
                    } else if v == 1 {
                        coeff_bu[u]
                    } else {
                        [0.0; 4]
                    };
                    let base_val = exact::cell_second_derivative_from_moments(
                        cell,
                        &coeff_u[u],
                        &coeff_u[v],
                        &second_coeff,
                        &state.moments,
                    )?;
                    f_uv[[u, v]] += base_val;
                    if u != v {
                        f_uv[[v, u]] += base_val;
                    }

                    let third_u = if u == 1 && v == 1 {
                        let mut out = [0.0; 4];
                        add_scaled_coeff4(&mut out, &coeff_bbu[1], dir_u_b);
                        if let Some(w_range) = w_range {
                            for idx in w_range.clone() {
                                add_scaled_coeff4(&mut out, &coeff_bbu[idx], dir_u[idx]);
                            }
                        }
                        out
                    } else if u == 1 {
                        if let Some(w_range) = w_range {
                            if w_range.contains(&v) {
                                let mut out = [0.0; 4];
                                add_scaled_coeff4(&mut out, &coeff_bbu[v], dir_u_b);
                                out
                            } else {
                                [0.0; 4]
                            }
                        } else {
                            [0.0; 4]
                        }
                    } else if v == 1 {
                        if let Some(w_range) = w_range {
                            if w_range.contains(&u) {
                                let mut out = [0.0; 4];
                                add_scaled_coeff4(&mut out, &coeff_bbu[u], dir_u_b);
                                out
                            } else {
                                [0.0; 4]
                            }
                        } else {
                            [0.0; 4]
                        }
                    } else {
                        [0.0; 4]
                    };
                    let third_v = if u == 1 && v == 1 {
                        let mut out = [0.0; 4];
                        add_scaled_coeff4(&mut out, &coeff_bbu[1], dir_v_b);
                        if let Some(w_range) = w_range {
                            for idx in w_range.clone() {
                                add_scaled_coeff4(&mut out, &coeff_bbu[idx], dir_v[idx]);
                            }
                        }
                        out
                    } else if u == 1 {
                        if let Some(w_range) = w_range {
                            if w_range.contains(&v) {
                                let mut out = [0.0; 4];
                                add_scaled_coeff4(&mut out, &coeff_bbu[v], dir_v_b);
                                out
                            } else {
                                [0.0; 4]
                            }
                        } else {
                            [0.0; 4]
                        }
                    } else if v == 1 {
                        if let Some(w_range) = w_range {
                            if w_range.contains(&u) {
                                let mut out = [0.0; 4];
                                add_scaled_coeff4(&mut out, &coeff_bbu[u], dir_v_b);
                                out
                            } else {
                                [0.0; 4]
                            }
                        } else {
                            [0.0; 4]
                        }
                    } else {
                        [0.0; 4]
                    };
                    let fourth_uv = if u == 1 && v == 1 {
                        let mut out = [0.0; 4];
                        if let Some(w_range) = w_range {
                            for idx in w_range.clone() {
                                add_scaled_coeff4(
                                    &mut out,
                                    &coeff_bbbu[idx],
                                    dir_u_b * dir_v[idx] + dir_v_b * dir_u[idx],
                                );
                            }
                        }
                        out
                    } else if u == 1 {
                        if let Some(w_range) = w_range {
                            if w_range.contains(&v) {
                                let mut out = [0.0; 4];
                                add_scaled_coeff4(&mut out, &coeff_bbbu[v], dir_u_b * dir_v_b);
                                out
                            } else {
                                [0.0; 4]
                            }
                        } else {
                            [0.0; 4]
                        }
                    } else if v == 1 {
                        if let Some(w_range) = w_range {
                            if w_range.contains(&u) {
                                let mut out = [0.0; 4];
                                add_scaled_coeff4(&mut out, &coeff_bbbu[u], dir_u_b * dir_v_b);
                                out
                            } else {
                                [0.0; 4]
                            }
                        } else {
                            [0.0; 4]
                        }
                    } else {
                        [0.0; 4]
                    };

                    let dir_u_val = exact::cell_third_derivative_from_moments(
                        cell,
                        &coeff_u[u],
                        &coeff_u[v],
                        &coeff_dir_u,
                        &second_coeff,
                        &coeff_u_dir_u[u],
                        &coeff_u_dir_u[v],
                        &third_u,
                        &state.moments,
                    )?;
                    let dir_v_val = exact::cell_third_derivative_from_moments(
                        cell,
                        &coeff_u[u],
                        &coeff_u[v],
                        &coeff_dir_v,
                        &second_coeff,
                        &coeff_u_dir_v[u],
                        &coeff_u_dir_v[v],
                        &third_v,
                        &state.moments,
                    )?;
                    let mix_val = exact::cell_fourth_derivative_from_moments(
                        cell,
                        &coeff_u[u],
                        &coeff_u[v],
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

        f_u[0] = -phi_q;
        f_uv[[0, 0]] = q * phi_q;
        f_uv_u[[0, 0]] = dir_u[0] * (1.0 - q * q) * phi_q;
        f_uv_v[[0, 0]] = dir_v[0] * (1.0 - q * q) * phi_q;
        f_uv_uv[[0, 0]] = dir_u[0] * dir_v[0] * (q * q * q - 3.0 * q) * phi_q;

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
            g_u[u] = eval_coeff4_at(&g_u_fixed[u], z_obs);
            g_au[u] = eval_coeff4_at(&g_au_fixed[u], z_obs);
            g_aau[u] = eval_coeff4_at(&g_aau_fixed[u], z_obs);
            g_aaau[u] = eval_coeff4_at(&g_aaau_fixed[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let second_coeff = if u == 1 {
                    g_bu_fixed[v]
                } else if v == 1 {
                    g_bu_fixed[u]
                } else {
                    [0.0; 4]
                };
                let val = eval_coeff4_at(&second_coeff, z_obs);
                g_uv[[u, v]] = val;
                g_uv[[v, u]] = val;

                let third_coeff = if u == 1 && v == 1 {
                    g_abu_fixed[1]
                } else if u == 1 {
                    if let Some(w_range) = w_range {
                        if w_range.contains(&v) {
                            g_abu_fixed[v]
                        } else {
                            [0.0; 4]
                        }
                    } else {
                        [0.0; 4]
                    }
                } else if v == 1 {
                    if let Some(w_range) = w_range {
                        if w_range.contains(&u) {
                            g_abu_fixed[u]
                        } else {
                            [0.0; 4]
                        }
                    } else {
                        [0.0; 4]
                    }
                } else {
                    [0.0; 4]
                };
                let fourth_coeff = if u == 1 && v == 1 {
                    [0.0; 4]
                } else if u == 1 {
                    if let Some(w_range) = w_range {
                        if w_range.contains(&v) {
                            g_aabu_fixed[v]
                        } else {
                            [0.0; 4]
                        }
                    } else {
                        [0.0; 4]
                    }
                } else if v == 1 {
                    if let Some(w_range) = w_range {
                        if w_range.contains(&u) {
                            g_aabu_fixed[u]
                        } else {
                            [0.0; 4]
                        }
                    } else {
                        [0.0; 4]
                    }
                } else {
                    [0.0; 4]
                };
                let third_val = eval_coeff4_at(&third_coeff, z_obs);
                let fourth_val = eval_coeff4_at(&fourth_coeff, z_obs);
                g_auv[[u, v]] = third_val;
                g_auv[[v, u]] = third_val;
                g_aauv[[u, v]] = fourth_val;
                g_aauv[[v, u]] = fourth_val;
            }
        }

        let mut g_dir_u_fixed = [0.0; 4];
        let mut g_dir_v_fixed = [0.0; 4];
        let mut g_a_dir_u_fixed = [0.0; 4];
        let mut g_a_dir_v_fixed = [0.0; 4];
        let mut g_aa_dir_u_fixed = [0.0; 4];
        let mut g_aa_dir_v_fixed = [0.0; 4];
        let mut g_dir_uv_fixed = [0.0; 4];
        let mut g_a_dir_uv_fixed = [0.0; 4];
        let mut g_aa_dir_uv_fixed = [0.0; 4];
        for u in 1..r {
            add_scaled_coeff4(&mut g_dir_u_fixed, &g_u_fixed[u], dir_u[u]);
            add_scaled_coeff4(&mut g_dir_v_fixed, &g_u_fixed[u], dir_v[u]);
            add_scaled_coeff4(&mut g_a_dir_u_fixed, &g_au_fixed[u], dir_u[u]);
            add_scaled_coeff4(&mut g_a_dir_v_fixed, &g_au_fixed[u], dir_v[u]);
            add_scaled_coeff4(&mut g_aa_dir_u_fixed, &g_aau_fixed[u], dir_u[u]);
            add_scaled_coeff4(&mut g_aa_dir_v_fixed, &g_aau_fixed[u], dir_v[u]);
        }
        add_scaled_coeff4(&mut g_dir_uv_fixed, &g_bu_fixed[1], dir_u_b * dir_v_b);
        if let Some(h_range) = h_range {
            for idx in h_range.clone() {
                add_scaled_coeff4(
                    &mut g_dir_uv_fixed,
                    &g_bu_fixed[idx],
                    dir_u_b * dir_v[idx] + dir_v_b * dir_u[idx],
                );
            }
        }
        if let Some(w_range) = w_range {
            for idx in w_range.clone() {
                add_scaled_coeff4(
                    &mut g_dir_uv_fixed,
                    &g_bu_fixed[idx],
                    dir_u_b * dir_v[idx] + dir_v_b * dir_u[idx],
                );
                add_scaled_coeff4(
                    &mut g_a_dir_uv_fixed,
                    &g_abu_fixed[idx],
                    dir_u_b * dir_v[idx] + dir_v_b * dir_u[idx],
                );
                add_scaled_coeff4(
                    &mut g_aa_dir_uv_fixed,
                    &g_aabu_fixed[idx],
                    dir_u_b * dir_v[idx] + dir_v_b * dir_u[idx],
                );
            }
        }

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

        let mut tmp_u = vec![[0.0; 4]; r];
        let mut tmp_v = vec![[0.0; 4]; r];
        let mut tmp_uv = vec![[0.0; 4]; r];
        let mut tmp_au_u = vec![[0.0; 4]; r];
        let mut tmp_au_v = vec![[0.0; 4]; r];
        let mut tmp_au_uv = vec![[0.0; 4]; r];

        tmp_u[1] = {
            let mut out = [0.0; 4];
            add_scaled_coeff4(&mut out, &g_bu_fixed[1], dir_u_b);
            if let Some(h_range) = h_range {
                for idx in h_range.clone() {
                    add_scaled_coeff4(&mut out, &g_bu_fixed[idx], dir_u[idx]);
                }
            }
            if let Some(w_range) = w_range {
                for idx in w_range.clone() {
                    add_scaled_coeff4(&mut out, &g_bu_fixed[idx], dir_u[idx]);
                }
            }
            out
        };
        tmp_v[1] = {
            let mut out = [0.0; 4];
            add_scaled_coeff4(&mut out, &g_bu_fixed[1], dir_v_b);
            if let Some(h_range) = h_range {
                for idx in h_range.clone() {
                    add_scaled_coeff4(&mut out, &g_bu_fixed[idx], dir_v[idx]);
                }
            }
            if let Some(w_range) = w_range {
                for idx in w_range.clone() {
                    add_scaled_coeff4(&mut out, &g_bu_fixed[idx], dir_v[idx]);
                }
            }
            out
        };
        tmp_uv[1] = g_dir_uv_fixed;
        tmp_au_u[1] = {
            let mut out = [0.0; 4];
            add_scaled_coeff4(&mut out, &g_abu_fixed[1], dir_u_b);
            if let Some(w_range) = w_range {
                for idx in w_range.clone() {
                    add_scaled_coeff4(&mut out, &g_abu_fixed[idx], dir_u[idx]);
                }
            }
            out
        };
        tmp_au_v[1] = {
            let mut out = [0.0; 4];
            add_scaled_coeff4(&mut out, &g_abu_fixed[1], dir_v_b);
            if let Some(w_range) = w_range {
                for idx in w_range.clone() {
                    add_scaled_coeff4(&mut out, &g_abu_fixed[idx], dir_v[idx]);
                }
            }
            out
        };
        tmp_au_uv[1] = {
            let mut out = [0.0; 4];
            if let Some(w_range) = w_range {
                for idx in w_range.clone() {
                    add_scaled_coeff4(
                        &mut out,
                        &g_abbu_fixed[idx],
                        dir_u_b * dir_v[idx] + dir_v_b * dir_u[idx],
                    );
                }
            }
            out
        };

        if let Some(h_range) = h_range {
            for idx in h_range.clone() {
                add_scaled_coeff4(&mut tmp_u[idx], &g_bu_fixed[idx], dir_u_b);
                add_scaled_coeff4(&mut tmp_v[idx], &g_bu_fixed[idx], dir_v_b);
            }
        }
        if let Some(w_range) = w_range {
            for idx in w_range.clone() {
                add_scaled_coeff4(&mut tmp_u[idx], &g_bu_fixed[idx], dir_u_b);
                add_scaled_coeff4(&mut tmp_v[idx], &g_bu_fixed[idx], dir_v_b);
                add_scaled_coeff4(&mut tmp_uv[idx], &g_bbu_fixed[idx], dir_u_b * dir_v_b);
                add_scaled_coeff4(&mut tmp_au_u[idx], &g_abu_fixed[idx], dir_u_b);
                add_scaled_coeff4(&mut tmp_au_v[idx], &g_abu_fixed[idx], dir_v_b);
                add_scaled_coeff4(&mut tmp_au_uv[idx], &g_abbu_fixed[idx], dir_u_b * dir_v_b);
            }
        }

        for u in 1..r {
            g_u_u_fixed[u] = eval_coeff4_at(&tmp_u[u], z_obs);
            g_u_v_fixed[u] = eval_coeff4_at(&tmp_v[u], z_obs);
            g_u_uv_fixed[u] = eval_coeff4_at(&tmp_uv[u], z_obs);
            g_au_u_fixed[u] = eval_coeff4_at(&tmp_au_u[u], z_obs);
            g_au_v_fixed[u] = eval_coeff4_at(&tmp_au_v[u], z_obs);
            g_au_uv_fixed[u] = eval_coeff4_at(&tmp_au_uv[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let third_u = if u == 1 && v == 1 {
                    let mut out = [0.0; 4];
                    add_scaled_coeff4(&mut out, &g_bbu_fixed[1], dir_u_b);
                    if let Some(w_range) = w_range {
                        for idx in w_range.clone() {
                            add_scaled_coeff4(&mut out, &g_bbu_fixed[idx], dir_u[idx]);
                        }
                    }
                    out
                } else if u == 1 {
                    if let Some(w_range) = w_range {
                        if w_range.contains(&v) {
                            let mut out = [0.0; 4];
                            add_scaled_coeff4(&mut out, &g_bbu_fixed[v], dir_u_b);
                            out
                        } else {
                            [0.0; 4]
                        }
                    } else {
                        [0.0; 4]
                    }
                } else if v == 1 {
                    if let Some(w_range) = w_range {
                        if w_range.contains(&u) {
                            let mut out = [0.0; 4];
                            add_scaled_coeff4(&mut out, &g_bbu_fixed[u], dir_u_b);
                            out
                        } else {
                            [0.0; 4]
                        }
                    } else {
                        [0.0; 4]
                    }
                } else {
                    [0.0; 4]
                };
                let third_v = if u == 1 && v == 1 {
                    let mut out = [0.0; 4];
                    add_scaled_coeff4(&mut out, &g_bbu_fixed[1], dir_v_b);
                    if let Some(w_range) = w_range {
                        for idx in w_range.clone() {
                            add_scaled_coeff4(&mut out, &g_bbu_fixed[idx], dir_v[idx]);
                        }
                    }
                    out
                } else if u == 1 {
                    if let Some(w_range) = w_range {
                        if w_range.contains(&v) {
                            let mut out = [0.0; 4];
                            add_scaled_coeff4(&mut out, &g_bbu_fixed[v], dir_v_b);
                            out
                        } else {
                            [0.0; 4]
                        }
                    } else {
                        [0.0; 4]
                    }
                } else if v == 1 {
                    if let Some(w_range) = w_range {
                        if w_range.contains(&u) {
                            let mut out = [0.0; 4];
                            add_scaled_coeff4(&mut out, &g_bbu_fixed[u], dir_v_b);
                            out
                        } else {
                            [0.0; 4]
                        }
                    } else {
                        [0.0; 4]
                    }
                } else {
                    [0.0; 4]
                };
                let fourth_uv = if u == 1 && v == 1 {
                    let mut out = [0.0; 4];
                    if let Some(w_range) = w_range {
                        for idx in w_range.clone() {
                            add_scaled_coeff4(
                                &mut out,
                                &g_bbbu_fixed[idx],
                                dir_u_b * dir_v[idx] + dir_v_b * dir_u[idx],
                            );
                        }
                    }
                    out
                } else if u == 1 {
                    if let Some(w_range) = w_range {
                        if w_range.contains(&v) {
                            let mut out = [0.0; 4];
                            add_scaled_coeff4(&mut out, &g_bbbu_fixed[v], dir_u_b * dir_v_b);
                            out
                        } else {
                            [0.0; 4]
                        }
                    } else {
                        [0.0; 4]
                    }
                } else if v == 1 {
                    if let Some(w_range) = w_range {
                        if w_range.contains(&u) {
                            let mut out = [0.0; 4];
                            add_scaled_coeff4(&mut out, &g_bbbu_fixed[u], dir_u_b * dir_v_b);
                            out
                        } else {
                            [0.0; 4]
                        }
                    } else {
                        [0.0; 4]
                    }
                } else {
                    [0.0; 4]
                };
                let vu = eval_coeff4_at(&third_u, z_obs);
                let vv = eval_coeff4_at(&third_v, z_obs);
                let vuv = eval_coeff4_at(&fourth_uv, z_obs);
                g_uv_u_fixed[[u, v]] = vu;
                g_uv_v_fixed[[u, v]] = vv;
                g_uv_uv_fixed[[u, v]] = vuv;
                g_uv_u_fixed[[v, u]] = vu;
                g_uv_v_fixed[[v, u]] = vv;
                g_uv_uv_fixed[[v, u]] = vuv;
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
        let g_a_uv =
            g_aa * a_dir_uv + g_aa_u_fixed * a_dir_v + g_aa_v_fixed * a_dir_u + g_a_uv_fixed;
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
                    + g_uv_u_fixed[[u, v]] * a_dir_v
                    + g_uv_v_fixed[[u, v]] * a_dir_u
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

        let eta_dir_u = eta_u.dot(dir_u);
        let eta_dir_v = eta_u.dot(dir_v);
        let eta_u_dir_u = eta_uv.dot(dir_u);
        let eta_u_dir_v = eta_uv.dot(dir_v);
        let eta_dir_uv = eta_u_dir_u.dot(dir_v);
        let eta_u_uv = eta_uv_u.dot(dir_v);

        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let m = s_y * eta_val;
        let (k1, k2, k3, k4) = signed_probit_neglog_derivatives_up_to_fourth(m, w_i);
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
        if !self.flex_active() {
            let out = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(
                    || Array1::<f64>::zeros(slices.total),
                    |mut chunk_out, chunk_idx| -> Result<_, String> {
                        let start = chunk_idx * ROW_CHUNK_SIZE;
                        let end = (start + ROW_CHUNK_SIZE).min(n);
                        for row in start..end {
                            let q = block_states[0].eta[row];
                            let g = block_states[1].eta[row];
                            let k = RigidProbitKernel::new(
                                q,
                                g,
                                self.z[row],
                                self.y[row],
                                self.weights[row],
                            );
                            let h = k.primary_hessian(q);
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
                    for row in start..end {
                        let row_ctx = Self::row_ctx(cache, row);
                        let row_dir =
                            self.row_primary_direction_from_flat(row, slices, primary, direction)?;
                        let (_, _, row_hessian) = self.compute_row_primary_gradient_hessian(
                            row,
                            block_states,
                            primary,
                            row_ctx,
                        )?;
                        let row_action = row_hessian.dot(&row_dir);
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
        if !self.flex_active() {
            let diagonal = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(
                    || Array1::<f64>::zeros(slices.total),
                    |mut chunk_diag, chunk_idx| -> Result<_, String> {
                        let start = chunk_idx * ROW_CHUNK_SIZE;
                        let end = (start + ROW_CHUNK_SIZE).min(n);
                        for row in start..end {
                            let q = block_states[0].eta[row];
                            let g = block_states[1].eta[row];
                            let k = RigidProbitKernel::new(
                                q,
                                g,
                                self.z[row],
                                self.y[row],
                                self.weights[row],
                            );
                            let h = k.primary_hessian(q);
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
                    for row in start..end {
                        let row_ctx = Self::row_ctx(cache, row);
                        let (_, _, row_hessian) = self.compute_row_primary_gradient_hessian(
                            row,
                            block_states,
                            primary,
                            row_ctx,
                        )?;

                        {
                            let mut marginal_diag =
                                chunk_diag.slice_mut(s![slices.marginal.clone()]);
                            self.marginal_design.squared_axpy_row_into(
                                row,
                                row_hessian[[0, 0]],
                                &mut marginal_diag,
                            )?;
                        }
                        {
                            let mut logslope_diag =
                                chunk_diag.slice_mut(s![slices.logslope.clone()]);
                            self.logslope_design.squared_axpy_row_into(
                                row,
                                row_hessian[[1, 1]],
                                &mut logslope_diag,
                            )?;
                        }

                        if let (Some(primary_h), Some(block_h)) =
                            (primary.h.as_ref(), slices.h.as_ref())
                        {
                            for (local_idx, global_idx) in block_h.clone().enumerate() {
                                chunk_diag[global_idx] += row_hessian
                                    [[primary_h.start + local_idx, primary_h.start + local_idx]];
                            }
                        }
                        if let (Some(primary_w), Some(block_w)) =
                            (primary.w.as_ref(), slices.w.as_ref())
                        {
                            for (local_idx, global_idx) in block_w.clone().enumerate() {
                                chunk_diag[global_idx] += row_hessian
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
        let psi_label = if block_idx == 0 {
            "BernoulliMarginalSlopeFamily marginal"
        } else {
            "BernoulliMarginalSlopeFamily log-slope"
        };
        let psi_p = if block_idx == 0 {
            self.marginal_design.ncols()
        } else {
            self.logslope_design.ncols()
        };
        let psi_action = CustomFamilyPsiDesignAction::from_first_derivative(
            deriv,
            self.y.len(),
            psi_p,
            0..self.y.len(),
            psi_label,
        )
        .ok();
        let can_operatorize = psi_action.is_some();
        let has_hw = primary.h.is_some() || primary.w.is_some();

        if can_operatorize {
            let (objective_psi, score_psi, w_mm, w_mg, w_gm, w_gg, d_mm, d_mg, d_gg, dense_corr) =
                (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                    .into_par_iter()
                    .try_fold(
                        || {
                            (
                                0.0,
                                Array1::<f64>::zeros(slices.total),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array2::<f64>::zeros((slices.total, slices.total)),
                            )
                        },
                        |mut acc, chunk_idx| -> Result<_, String> {
                            let start = chunk_idx * ROW_CHUNK_SIZE;
                            let end = (start + ROW_CHUNK_SIZE).min(n);
                            for row in start..end {
                                let Some(dir) = self.row_primary_psi_direction(
                                    row,
                                    block_states,
                                    derivative_blocks,
                                    psi_index,
                                    primary,
                                )?
                                else {
                                    continue;
                                };
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
                                let psi_row = self.psi_design_row_vector(
                                    row,
                                    deriv,
                                    self.y.len(),
                                    psi_p,
                                    psi_label,
                                )?;

                                acc.0 += f_pi.dot(&dir);
                                match block_idx {
                                    0 => {
                                        let mut score_m =
                                            acc.1.slice_mut(s![slices.marginal.clone()]);
                                        score_m.scaled_add(f_pi[idx_primary], &psi_row);
                                    }
                                    1 => {
                                        let mut score_g =
                                            acc.1.slice_mut(s![slices.logslope.clone()]);
                                        score_g.scaled_add(f_pi[idx_primary], &psi_row);
                                    }
                                    _ => unreachable!(),
                                }
                                acc.1 += &self.pullback_primary_vector(
                                    row,
                                    slices,
                                    primary,
                                    &f_pipi.dot(&dir),
                                )?;

                                if block_idx == 0 {
                                    acc.2[row] = f_pipi[[0, 0]];
                                    acc.3[row] = f_pipi[[0, 1]];
                                    acc.4[row] = f_pipi[[1, 0]];
                                } else {
                                    acc.3[row] = f_pipi[[0, 1]];
                                    acc.4[row] = f_pipi[[1, 0]];
                                    acc.5[row] = f_pipi[[1, 1]];
                                }
                                acc.6[row] = third[[0, 0]];
                                acc.7[row] = third[[0, 1]];
                                acc.8[row] = third[[1, 1]];

                                // h/w cross-block contributions -> dense correction
                                if has_hw {
                                    let right_primary = f_pipi.row(idx_primary).to_owned();
                                    let psi_range = if block_idx == 0 {
                                        slices.marginal.clone()
                                    } else {
                                        slices.logslope.clone()
                                    };
                                    if let (Some(ph), Some(bh)) =
                                        (primary.h.as_ref(), slices.h.as_ref())
                                    {
                                        let h_part = right_primary.slice(s![ph.start..ph.end]);
                                        for (li, gi) in psi_range.clone().enumerate() {
                                            for (lj, gj) in bh.clone().enumerate() {
                                                let val = psi_row[li] * h_part[lj];
                                                acc.9[[gi, gj]] += val;
                                                acc.9[[gj, gi]] += val;
                                            }
                                        }
                                    }
                                    if let (Some(pw), Some(bw)) =
                                        (primary.w.as_ref(), slices.w.as_ref())
                                    {
                                        let w_part = right_primary.slice(s![pw.start..pw.end]);
                                        for (li, gi) in psi_range.enumerate() {
                                            for (lj, gj) in bw.clone().enumerate() {
                                                let val = psi_row[li] * w_part[lj];
                                                acc.9[[gi, gj]] += val;
                                                acc.9[[gj, gi]] += val;
                                            }
                                        }
                                    }
                                    self.add_pullback_primary_hessian_hw_only(
                                        &mut acc.9, row, slices, primary, &third,
                                    );
                                }
                            }
                            Ok(acc)
                        },
                    )
                    .try_reduce(
                        || {
                            (
                                0.0,
                                Array1::<f64>::zeros(slices.total),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array1::<f64>::zeros(n),
                                Array2::<f64>::zeros((slices.total, slices.total)),
                            )
                        },
                        |mut left, right| -> Result<_, String> {
                            left.0 += right.0;
                            left.1 += &right.1;
                            left.2 += &right.2;
                            left.3 += &right.3;
                            left.4 += &right.4;
                            left.5 += &right.5;
                            left.6 += &right.6;
                            left.7 += &right.7;
                            left.8 += &right.8;
                            left.9 += &right.9;
                            Ok(left)
                        },
                    )?;

            let channels = vec![
                CustomFamilyJointDesignChannel::new(
                    slices.marginal.clone(),
                    self.marginal_design.to_dense_arc(),
                    if block_idx == 0 {
                        psi_action.clone()
                    } else {
                        None
                    },
                ),
                CustomFamilyJointDesignChannel::new(
                    slices.logslope.clone(),
                    self.logslope_design.to_dense_arc(),
                    if block_idx == 1 { psi_action } else { None },
                ),
            ];
            let pair_contributions = vec![
                CustomFamilyJointDesignPairContribution::new(0, 0, w_mm, d_mm),
                CustomFamilyJointDesignPairContribution::new(0, 1, w_mg.clone(), d_mg.clone()),
                CustomFamilyJointDesignPairContribution::new(1, 0, w_gm, d_mg),
                CustomFamilyJointDesignPairContribution::new(1, 1, w_gg, d_gg),
            ];

            let dense_correction = if has_hw { Some(dense_corr) } else { None };

            return Ok(Some(ExactNewtonJointPsiTerms {
                objective_psi,
                score_psi,
                hessian_psi: Array2::zeros((0, 0)),
                hessian_psi_operator: Some(Box::new(
                    CustomFamilyJointPsiOperator::with_dense_correction(
                        slices.total,
                        channels,
                        pair_contributions,
                        dense_correction,
                    ),
                )),
            }));
        }

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
                        let Some(dir) = self.row_primary_psi_direction(
                            row,
                            block_states,
                            derivative_blocks,
                            psi_index,
                            primary,
                        )?
                        else {
                            continue;
                        };
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
                        let psi_row = self
                            .block_psi_row(row, slices, derivative_blocks, psi_index)?
                            .ok_or_else(|| {
                                "missing bernoulli marginal-slope psi vector".to_string()
                            })?;
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
            hessian_psi_operator: Some(Box::new(block_acc.into_operator(slices))),
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
        let Some((block_i, _)) = self.resolve_psi_location(derivative_blocks, psi_i) else {
            return Ok(None);
        };
        let Some((block_j, _)) = self.resolve_psi_location(derivative_blocks, psi_j) else {
            return Ok(None);
        };
        let idx_i = if block_i == 0 { 0 } else { 1 };
        let idx_j = if block_j == 0 { 0 } else { 1 };
        let n = self.y.len();
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
                        let Some(dir_i) = self.row_primary_psi_direction(
                            row,
                            block_states,
                            derivative_blocks,
                            psi_i,
                            primary,
                        )?
                        else {
                            continue;
                        };
                        let Some(dir_j) = self.row_primary_psi_direction(
                            row,
                            block_states,
                            derivative_blocks,
                            psi_j,
                            primary,
                        )?
                        else {
                            continue;
                        };
                        let dir_ij = self
                            .row_primary_psi_second_direction(
                                row,
                                block_states,
                                derivative_blocks,
                                psi_i,
                                psi_j,
                                primary,
                            )?
                            .unwrap_or_else(|| Array1::<f64>::zeros(primary.total));
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
                        let br_i = self
                            .block_psi_row(row, slices, derivative_blocks, psi_i)?
                            .ok_or_else(|| {
                                "missing bernoulli marginal-slope psi_i vector".to_string()
                            })?;
                        let br_j = self
                            .block_psi_row(row, slices, derivative_blocks, psi_j)?
                            .ok_or_else(|| {
                                "missing bernoulli marginal-slope psi_j vector".to_string()
                            })?;
                        let br_ij = self.block_psi_second_row(
                            row,
                            slices,
                            derivative_blocks,
                            psi_i,
                            psi_j,
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
        let Some((block_idx, _)) = self.resolve_psi_location(derivative_blocks, psi_index) else {
            return Ok(None);
        };
        let idx_primary = if block_idx == 0 { 0 } else { 1 };
        let n = self.y.len();
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
                        let Some(psi_dir) = self.row_primary_psi_direction(
                            row,
                            block_states,
                            derivative_blocks,
                            psi_index,
                            primary,
                        )?
                        else {
                            continue;
                        };
                        let psi_action = self
                            .row_primary_psi_action_on_direction(
                                row,
                                slices,
                                derivative_blocks,
                                psi_index,
                                d_beta_flat,
                                primary,
                            )?
                            .unwrap_or_else(|| Array1::<f64>::zeros(primary.total));
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
                        let psi_row = self
                            .block_psi_row(row, slices, derivative_blocks, psi_index)?
                            .ok_or_else(|| {
                                "missing bernoulli marginal-slope psi vector".to_string()
                            })?;
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
        if !self.flex_active() {
            let block_acc = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(
                    || BernoulliBlockHessianAccumulator::new(slices),
                    |mut acc, chunk_idx| -> Result<_, String> {
                        let start = chunk_idx * ROW_CHUNK_SIZE;
                        let end = (start + ROW_CHUNK_SIZE).min(n);
                        for row in start..end {
                            let q = block_states[0].eta[row];
                            let g = block_states[1].eta[row];
                            let k = RigidProbitKernel::new(
                                q,
                                g,
                                self.z[row],
                                self.y[row],
                                self.weights[row],
                            );
                            let dq = self
                                .marginal_design
                                .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
                            let dg = self
                                .logslope_design
                                .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));
                            let t = k.third_contracted(q, dq, dg);
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
        if !self.flex_active() {
            let block_acc = (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(make_acc, |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    for row in start..end {
                        let q = block_states[0].eta[row];
                        let g = block_states[1].eta[row];
                        let k = RigidProbitKernel::new(
                            q,
                            g,
                            self.z[row],
                            self.y[row],
                            self.weights[row],
                        );
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
                        let f = k.fourth_contracted(q, uq, ug, vq, vg);
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

    fn evaluate_blockwise_exact_newton(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        let cache = self.build_exact_eval_cache_with_order(block_states)?;
        let slices = cache.slices.clone();
        let primary = cache.primary.clone();

        // Fast path: rigid model (no score-warp, no link-deviation).
        // Use closed-form scalar kernel: η = q·c + g·z, ℓ = -w·logΦ(s·η).
        // No jets, no intercept solve, no quadrature.
        if !self.flex_active() {
            let mut ll = 0.0;
            let mut grad_marginal = Array1::<f64>::zeros(slices.marginal.len());
            let mut grad_logslope = Array1::<f64>::zeros(slices.logslope.len());
            let mut hess_marginal =
                Array2::<f64>::zeros((slices.marginal.len(), slices.marginal.len()));
            let mut hess_logslope =
                Array2::<f64>::zeros((slices.logslope.len(), slices.logslope.len()));
            for row in 0..self.y.len() {
                let q = block_states[0].eta[row];
                let g = block_states[1].eta[row];
                let yi = self.y[row];
                let wi = self.weights[row];
                let zi = self.z[row];
                let s = 2.0 * yi - 1.0;

                let g2 = g * g;
                let c = (1.0 + g2).sqrt();
                // c(g) = sqrt(1+g²), c'(g) = g/c, c''(g) = 1/c³
                let c1 = g / c;
                let c2 = 1.0 / (c * c * c);
                let eta = q * c + g * zi;
                let m = s * eta;

                let (logcdf, lambda) = signed_probit_logcdf_and_mills_ratio(m);
                ll += wi * logcdf;

                let u1 = -wi * s * lambda;
                let u2 = wi * lambda * (m + lambda);

                let deta_dq = c;
                let deta_dg = q * c1 + zi;

                {
                    let mut marginal = grad_marginal.view_mut();
                    self.marginal_design
                        .axpy_row_into(row, -(u1 * deta_dq), &mut marginal)?;
                }
                {
                    let mut logslope = grad_logslope.view_mut();
                    self.logslope_design
                        .axpy_row_into(row, -(u1 * deta_dg), &mut logslope)?;
                }
                self.marginal_design.syr_row_into(
                    row,
                    u2 * deta_dq * deta_dq,
                    &mut hess_marginal,
                )?;
                self.logslope_design.syr_row_into(
                    row,
                    u2 * deta_dg * deta_dg + u1 * q * c2,
                    &mut hess_logslope,
                )?;
            }

            return Ok(FamilyEvaluation {
                log_likelihood: ll,
                blockworking_sets: vec![
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_marginal,
                        hessian: SymmetricMatrix::Dense(hess_marginal),
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_logslope,
                        hessian: SymmetricMatrix::Dense(hess_logslope),
                    },
                ],
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
                    grad_marginal_weights[local_row] = -scratch.grad[0];
                    grad_logslope_weights[local_row] = -scratch.grad[1];
                    hess_marginal_weights[local_row] = scratch.hess[[0, 0]];
                    hess_logslope_weights[local_row] = scratch.hess[[1, 1]];

                    if let (Some(primary_h), Some(grad_h), Some(hess_h)) =
                        (primary.h.as_ref(), acc.grad_h.as_mut(), acc.hess_h.as_mut())
                    {
                        for idx in 0..primary_h.len() {
                            grad_h[idx] -= scratch.grad[primary_h.start + idx];
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
                            grad_w[idx] -= scratch.grad[primary_w.start + idx];
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
        // Shared memory gate: K(K+1)/2 × p² dense psi Hessians.
        if cost_gated_outer_order(specs) == ExactOuterDerivativeOrder::First {
            return ExactOuterDerivativeOrder::First;
        }
        // Family-specific row-loop gate.
        let k_smoothing: usize = specs.iter().map(|s| s.penalties.len()).sum();
        bernoulli_row_work_order(
            self.y.len(),
            self.score_warp
                .as_ref()
                .map(|runtime| runtime.basis_dim)
                .unwrap_or(0),
            self.link_dev
                .as_ref()
                .map(|runtime| runtime.basis_dim)
                .unwrap_or(0),
            k_smoothing,
        )
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
        if !self.flex_active() {
            // Rigid probit: vectorized closed-form.
            // η_i = q_i·c_i + b_i·z_i  where c_i = √(1+b_i²)
            // ll = Σ w_i · log Φ((2y_i−1)·η_i)
            let q = &block_states[0].eta;
            let b = &block_states[1].eta;
            let n = self.y.len();
            return (0..((n + ROW_CHUNK_SIZE - 1) / ROW_CHUNK_SIZE))
                .into_par_iter()
                .try_fold(
                    || 0.0,
                    |mut ll, chunk_idx| -> Result<_, String> {
                        let start = chunk_idx * ROW_CHUNK_SIZE;
                        let end = (start + ROW_CHUNK_SIZE).min(n);
                        for i in start..end {
                            let c_i = (1.0 + b[i] * b[i]).sqrt();
                            let eta_i = q[i] * c_i + b[i] * self.z[i];
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
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = if self.link_dev.is_some() {
            block_states.last().map(|state| &state.beta)
        } else {
            None
        };
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
        let slices = block_slices(
            block_states,
            self.score_warp.is_some(),
            self.link_dev.is_some(),
        );
        if slices.h.as_ref().is_some_and(|_| block_idx == 2) {
            if let Some(runtime) = &self.score_warp {
                return runtime.max_feasible_monotone_step(&block_states[2].beta, delta);
            }
        }
        let link_block_idx = if slices.h.is_some() { 3 } else { 2 };
        if slices
            .w
            .as_ref()
            .is_some_and(|_| block_idx == link_block_idx)
        {
            if let Some(runtime) = &self.link_dev {
                let beta_w = block_states
                    .last()
                    .map(|state| &state.beta)
                    .ok_or_else(|| "missing link deviation block state".to_string())?;
                return runtime.max_feasible_monotone_step(beta_w, delta);
            }
        }
        Ok(None)
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let slices = block_slices(
            block_states,
            self.score_warp.is_some(),
            self.link_dev.is_some(),
        );
        if slices.total >= 512 {
            return Ok(None);
        }
        if !self.flex_active() {
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

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        if !self.flex_active() {
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
        if !self.flex_active() {
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
        if !self.flex_active() {
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
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
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
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        Ok(Some(Arc::new(
            BernoulliMarginalSlopeExactNewtonJointPsiWorkspace::new(
                self.clone(),
                block_states.to_vec(),
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
        // The exact de-nested cubic path does not expose the old endpoint-only
        // linearized derivative constraints. Monotonicity is enforced by the
        // exact cubic feasibility runtime and step-size limiter instead.
        Ok(None)
    }

    fn post_update_block_beta(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        if block_idx >= block_states.len() {
            return Err(format!(
                "post-update block index {} out of range for {} blocks",
                block_idx,
                block_states.len()
            ));
        }
        let slices = block_slices(
            block_states,
            self.score_warp.is_some(),
            self.link_dev.is_some(),
        );
        if slices.h.as_ref().is_some_and(|_| block_idx == 2) {
            if let Some(runtime) = &self.score_warp {
                let current = &block_states[2].beta;
                if current.len() != beta.len() {
                    return Err(format!(
                        "score-warp post-update beta length mismatch: current={}, proposed={}",
                        current.len(),
                        beta.len()
                    ));
                }
                let delta = &beta - current;
                let Some(alpha) = runtime.max_feasible_monotone_step(current, &delta)? else {
                    return Ok(current.clone());
                };
                return Ok(current + &(delta * alpha));
            }
        }
        let link_block_idx = if slices.h.is_some() { 3 } else { 2 };
        if slices
            .w
            .as_ref()
            .is_some_and(|_| block_idx == link_block_idx)
        {
            if let Some(runtime) = &self.link_dev {
                let current = block_states
                    .get(link_block_idx)
                    .map(|state| &state.beta)
                    .ok_or_else(|| "missing current link-deviation block state".to_string())?;
                if current.len() != beta.len() {
                    return Err(format!(
                        "link-deviation post-update beta length mismatch: current={}, proposed={}",
                        current.len(),
                        beta.len()
                    ));
                }
                let delta = &beta - current;
                let Some(alpha) = runtime.max_feasible_monotone_step(current, &delta)? else {
                    return Ok(current.clone());
                };
                return Ok(current + &(delta * alpha));
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
        derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
    ) -> Result<Self, String> {
        let cache = family.build_exact_eval_cache(&block_states)?;
        Ok(Self {
            family,
            block_states,
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
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_psihessian_directional_derivative_from_cache(
                &self.block_states,
                &self.derivative_blocks,
                psi_index,
                d_beta_flat,
                &self.cache,
            )
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
    block.initial_beta = beta_hint.or_else(|| block.initial_beta.clone());
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
    validate_spec(data, &spec)?;
    let baseline = pooled_probit_baseline(&spec.y, &spec.z, &spec.weights)?;
    let mut joint_designs = build_term_collection_designs_joint(
        data,
        &[spec.marginalspec.clone(), spec.logslopespec.clone()],
    )
    .map_err(|e| e.to_string())?;
    let marginal_design = joint_designs.remove(0);
    let logslope_design = joint_designs.remove(0);
    let marginalspec_boot =
        freeze_term_collection_from_design(&spec.marginalspec, &marginal_design)
            .map_err(|e| e.to_string())?;
    let logslopespec_boot =
        freeze_term_collection_from_design(&spec.logslopespec, &logslope_design)
            .map_err(|e| e.to_string())?;

    let y = Arc::new(spec.y.clone());
    let weights = Arc::new(spec.weights.clone());
    let z = Arc::new(spec.z.clone());

    let score_warp_prepared = spec
        .score_warp
        .as_ref()
        .map(|cfg| build_deviation_block_from_seed(&spec.z, cfg))
        .transpose()?;
    // Build the link-deviation block if requested.  The seed only determines
    // knot placement for the deviation basis, so we use the closed-form
    // pooled-intercept probit solution instead of a full rigid pilot solve
    // (which would double total work at biobank scale):
    //   q0 ≈ a0 · √(1 + b0²) + b0 · z[i]
    // where (a0, b0) = pooled_probit_baseline.
    let link_dev_prepared = spec
        .link_dev
        .as_ref()
        .map(|cfg| {
            let q0_seed = Array1::from_iter((0..spec.z.len()).map(|row| {
                let a0 = baseline.0 + spec.marginal_offset[row];
                let b0 = baseline.1 + spec.logslope_offset[row];
                let scale = (1.0 + b0 * b0).sqrt();
                a0 * scale + b0 * spec.z[row]
            }));
            // Pad with a conservative envelope so the basis support covers
            // the domain reached during flexible joint optimization, not just
            // the baseline range.  Padding is 50 % of the IQR on each side.
            let link_dev_seed = {
                let mut sorted = q0_seed.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n_s = sorted.len();
                if n_s >= 4 {
                    let q1 = sorted[n_s / 4];
                    let q3 = sorted[3 * n_s / 4];
                    let iqr = (q3 - q1).max(1.0);
                    let pad = 0.5 * iqr;
                    let lo = sorted[0] - pad;
                    let hi = sorted[n_s - 1] + pad;
                    let mut padded = q0_seed.to_vec();
                    padded.push(lo);
                    padded.push(hi);
                    Array1::from_vec(padded)
                } else {
                    q0_seed
                }
            };
            build_deviation_block_from_seed(&link_dev_seed, cfg)
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
        &marginalspec_boot,
        &logslopespec_boot,
        marginal_design.penalties.len(),
        logslope_design.penalties.len(),
        &extra_rho0,
        kappa_options,
    );
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
                       logslope_design: &TermCollectionDesign|
     -> BernoulliMarginalSlopeFamily {
        BernoulliMarginalSlopeFamily {
            y: Arc::clone(&y),
            weights: Arc::clone(&weights),
            z: Arc::clone(&z),
            gaussian_frailty_sd: fixed_gaussian_shift_sigma(&spec.frailty),
            marginal_design: marginal_design.design.clone(),
            logslope_design: logslope_design.design.clone(),
            score_warp: score_warp_runtime.clone(),
            link_dev: link_dev_runtime.clone(),
        }
    };

    let marginal_psi_result =
        build_block_spatial_psi_derivatives(data, &marginalspec_boot, &marginal_design);
    let logslope_psi_result =
        build_block_spatial_psi_derivatives(data, &logslopespec_boot, &logslope_design);
    // Track which blocks actually have spatial psi-dependence.  A block that is
    // entirely non-spatial returns Ok(None), which is fine — its psi contribution
    // is identically zero.  We only require that at least one block has spatial
    // derivatives when the model has kappa dimensions to optimize.
    let marginal_has_spatial = marginal_psi_result
        .as_ref()
        .map(|r| r.is_some())
        .unwrap_or(false);
    let logslope_has_spatial = logslope_psi_result
        .as_ref()
        .map(|r| r.is_some())
        .unwrap_or(false);
    let analytic_joint_derivatives_available = marginal_psi_result.is_ok()
        && logslope_psi_result.is_ok()
        && (marginal_has_spatial || logslope_has_spatial);
    if setup.log_kappa_dim() > 0 && !analytic_joint_derivatives_available {
        return Err(
            "exact bernoulli marginal-slope spatial optimization requires analytic joint psi derivatives"
                .to_string(),
        );
    }
    let initial_rho = setup.theta0().slice(s![..setup.rho_dim()]).to_owned();
    let initial_blocks = build_blocks(&initial_rho, &marginal_design, &logslope_design)?;
    let initial_family = make_family(&marginal_design, &logslope_design);
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

    let marginal_terms = spatial_length_scale_term_indices(&marginalspec_boot);
    let logslope_terms = spatial_length_scale_term_indices(&logslopespec_boot);
    let solved = optimize_spatial_length_scale_exact_joint(
        data,
        &[marginalspec_boot.clone(), logslopespec_boot.clone()],
        &[marginal_terms, logslope_terms],
        kappa_options,
        &setup,
        crate::seeding::SeedRiskProfile::GeneralizedLinear,
        analytic_joint_gradient_available,
        analytic_joint_hessian_available,
        |rho, _: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            let blocks = build_blocks(rho, &designs[0], &designs[1])?;
            let family = make_family(&designs[0], &designs[1]);
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
        |rho, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign], need_hessian| {
            let blocks = build_blocks(rho, &designs[0], &designs[1])?;
            let family = make_family(&designs[0], &designs[1]);
            // For blocks with spatial terms, require derivatives (error if missing).
            // For non-spatial blocks, use an empty vec (zero psi contribution).
            let marginal_psi_derivs = if marginal_has_spatial {
                build_block_spatial_psi_derivatives(data, &specs[0], &designs[0])?.ok_or_else(
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
                build_block_spatial_psi_derivatives(data, &specs[1], &designs[1])?.ok_or_else(
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
            if family.score_warp.is_some() {
                derivative_blocks.push(Vec::new());
            }
            if family.link_dev.is_some() {
                derivative_blocks.push(Vec::new());
            }
            let eval = evaluate_custom_family_joint_hyper(
                &family,
                &blocks,
                options,
                rho,
                &derivative_blocks,
                exact_warm_start.borrow().as_ref(),
                need_hessian,
            )?;
            exact_warm_start.replace(Some(eval.warm_start));
            if need_hessian && eval.outer_hessian.is_none() {
                return Err(
                    "exact bernoulli marginal-slope joint [rho, psi] objective did not return an outer Hessian"
                        .to_string(),
                );
            }
            Ok((eval.objective, eval.gradient, eval.outer_hessian))
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
        score_warp_runtime,
        link_dev_runtime,
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
            marginalspec: empty_termspec(),
            logslopespec: empty_termspec(),
            marginal_offset: Array1::zeros(n),
            logslope_offset: Array1::zeros(n),
            frailty: FrailtySpec::None,
            score_warp: None,
            link_dev: None,
        }
    }

    fn pair_distance(lhs: (f64, f64), rhs: (f64, f64)) -> f64 {
        (lhs.0 - rhs.0).abs() + (lhs.1 - rhs.1).abs()
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
    fn link_dev_without_score_warp_uses_w_block_without_linear_constraint_shortcut() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_deviation_block_from_seed(
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

        let slices = block_slices(&block_states, false, true);
        assert!(slices.h.is_none(), "score-warp slice should be absent");
        let link_slice = slices.w.as_ref().expect("link slice");
        assert_eq!(link_slice.start, 2, "link-only block should occupy block 2");
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
            "non-link block should not expose linearized monotonicity constraints"
        );
        assert!(
            family
                .block_linear_constraints(&block_states, 2, &dummy_spec)
                .expect("link constraint lookup")
                .is_none(),
            "link block should use exact monotonicity runtime rather than endpoint-only linear constraints"
        );
    }

    #[test]
    fn score_warp_block_uses_exact_monotonicity_runtime_without_linear_constraints() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_deviation_block_from_seed(
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
        assert!(
            family
                .block_linear_constraints(&block_states, 2, &dummy_spec)
                .expect("constraint lookup")
                .is_none(),
            "score-warp block should rely on exact monotonicity checks, not endpoint-only linear constraints"
        );
    }

    #[test]
    fn exact_monotonicity_step_limit_keeps_deviation_feasible() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build deviation block");
        let dim = prepared.block.design.ncols();
        let beta0 = Array1::<f64>::zeros(dim);
        let mut found = None;
        'search: for idx in 0..dim {
            for &step in &[-64.0, -32.0, -16.0, -8.0, -4.0, 4.0, 8.0, 16.0, 32.0, 64.0] {
                let mut direction = Array1::<f64>::zeros(dim);
                direction[idx] = step;
                let mut trial = beta0.clone();
                trial += &direction;
                if prepared
                    .runtime
                    .monotonicity_feasible(&trial, "trial monotonicity")
                    .is_err()
                {
                    found = Some(direction);
                    break 'search;
                }
            }
        }
        let direction = found.expect("expected an infeasible full step");
        let alpha = prepared
            .runtime
            .max_feasible_monotone_step(&beta0, &direction)
            .expect("max feasible step")
            .expect("step ceiling");
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "expected strict step ceiling, got {alpha}"
        );

        let mut feasible = beta0.clone();
        feasible.scaled_add(alpha, &direction);
        prepared
            .runtime
            .monotonicity_feasible(&feasible, "feasible monotonicity")
            .expect("ceiling step should stay feasible");
        let mut infeasible = beta0.clone();
        infeasible += &direction;
        assert!(
            prepared
                .runtime
                .monotonicity_feasible(&infeasible, "infeasible monotonicity")
                .is_err(),
            "full step should violate exact cubic monotonicity"
        );
    }

    #[test]
    fn post_update_block_beta_projects_score_warp_to_feasible_step() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_deviation_block_from_seed(
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
    fn exact_monotonicity_requires_cubic_deviation_basis() {
        let seed = array![-1.0, 0.0, 1.0];
        let err = match build_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                degree: 2,
                ..DeviationBlockConfig::default()
            },
        ) {
            Ok(_) => panic!("non-cubic deviation basis should be rejected"),
            Err(err) => err,
        };
        assert!(err.contains("requires cubic deviation blocks"));
    }

    #[test]
    fn local_cubic_span_reconstructs_deviation_exactly() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build deviation block");
        let dim = prepared.block.design.ncols();
        let beta = Array1::from_iter((0..dim).map(|idx| 0.025 * (idx as f64 + 1.0)));
        let n_spans = prepared.runtime.span_count();

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
                let expected_span_idx = if i + 1 == x_eval.len() && span_idx + 1 < n_spans {
                    span_idx + 1
                } else {
                    span_idx
                };
                let expected_cubic = prepared
                    .runtime
                    .local_cubic_on_span(&beta, expected_span_idx)
                    .expect("expected lookup cubic on span");
                assert_eq!(selected.left, expected_cubic.left);
                assert_eq!(selected.right, expected_cubic.right);
            }
        }
    }

    #[test]
    fn basis_span_cubic_reconstructs_basis_column_exactly() {
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_deviation_block_from_seed(
            &seed,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build deviation block");
        let basis_idx = 0usize;
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
            assert_eq!(selected.left, cubic.left);
            assert_eq!(selected.right, cubic.right);
        }
    }

    #[test]
    fn denested_microcells_follow_score_and_link_breaks() {
        let score_seed = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let link_seed = array![-1.5, -0.5, 0.5, 1.5];
        let score_prepared = build_deviation_block_from_seed(
            &score_seed,
            &DeviationBlockConfig {
                num_internal_knots: 3,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score warp block");
        let link_prepared = build_deviation_block_from_seed(
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
            exact_cells_a0.len() >= score_prepared.runtime.span_count(),
            "microcell partition should refine the score spans"
        );
        assert!(
            exact_cells_a0
                .windows(2)
                .all(|w| (w[0].cell.right - w[1].cell.left).abs() <= 1e-12),
            "microcells should tile the partition contiguously"
        );
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
        let score_prepared = build_deviation_block_from_seed(
            &score_seed,
            &DeviationBlockConfig {
                num_internal_knots: 3,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("build score warp block");
        let link_prepared = build_deviation_block_from_seed(
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
            let z = 0.5 * (cell.cell.left + cell.cell.right);
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
    fn observed_denested_partials_include_nonzero_third_a_derivative() {
        let z = array![-0.8, 0.2, 1.1];
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_deviation_block_from_seed(
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
            "expected a nonzero observed d^3 eta / da^3 contribution for a nontrivial cubic link span"
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
    fn row_work_order_keeps_small_primary_models_second_order() {
        // Small n, no auxiliary primaries, K=2.
        // n=100_000 × k_pairs=3 × primary²=4 = 1_200_000 < 2M.
        assert_eq!(
            bernoulli_row_work_order(100_000, 0, 0, 2),
            ExactOuterDerivativeOrder::Second
        );
        // Moderate n with rich primaries, K=1.
        // n=4000 × k_pairs=1 × primary²=256 = 1_024_000 < 2M.
        assert_eq!(
            bernoulli_row_work_order(4_000, 8, 6, 1),
            ExactOuterDerivativeOrder::Second
        );
    }

    #[test]
    fn row_work_order_downgrades_large_rich_models_to_first_order() {
        // n=10_000 × k_pairs=6 × primary²=256 = 15_360_000 > 2M.
        assert_eq!(
            bernoulli_row_work_order(10_000, 8, 6, 3),
            ExactOuterDerivativeOrder::First
        );
    }

    #[test]
    fn flexible_family_exposes_exact_outer_derivative_path() {
        let seed = array![-1.0, 0.0, 1.0];
        let score_prepared = build_deviation_block_from_seed(
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
    fn cost_gated_outer_order_downgrades_large_p() {
        use crate::custom_family::cost_gated_outer_order;
        use crate::matrix::DesignMatrix;
        use ndarray::Array2;

        // 2 blocks × 500 cols = p=1000, each block has 10 penalties → K=20,
        // K_pairs=210, psi_elements = 210 × 1000² = 210M > 200M limit.
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
            ExactOuterDerivativeOrder::First
        );
    }

    #[test]
    fn rigid_fast_path_matches_loglik_finite_differences() {
        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(array![1.0]),
            weights: Arc::new(array![1.2]),
            z: Arc::new(array![0.3]),
            gaussian_frailty_sd: None,
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

        assert!(
            (grad_q + primary_grad[0]).abs() < 1e-10,
            "marginal gradient mismatch: fast={grad_q:.12e}, exact={:.12e}",
            -primary_grad[0]
        );
        assert!(
            (grad_g + primary_grad[1]).abs() < 1e-10,
            "logslope gradient mismatch: fast={grad_g:.12e}, exact={:.12e}",
            -primary_grad[1]
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
        let prepared = build_deviation_block_from_seed(
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

        let slices = block_slices(&block_states, false, true);
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
        let prepared = build_deviation_block_from_seed(
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

        let slices = block_slices(&block_states, true, false);
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
        let prepared = build_deviation_block_from_seed(
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

        let slices = block_slices(&block_states, false, true);
        let total = slices.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[slices.marginal.start] = 0.4;
        dir_u[slices.logslope.start] = -0.3;
        let w_range = slices.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.15;
        if w_range.len() > 1 {
            dir_u[w_range.start + 1] = -0.07;
        }

        dir_v[slices.marginal.start] = -0.2;
        dir_v[slices.logslope.start] = 0.25;
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
        let seed = array![-1.5, -0.5, 0.0, 0.5, 1.5];
        let prepared = build_deviation_block_from_seed(
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

        let slices = block_slices(&block_states, true, false);
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
        let prepared = build_deviation_block_from_seed(
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
        dir_u[cache.slices.marginal.start] = -0.35;
        dir_u[cache.slices.logslope.start] = 0.28;
        let h_range = cache.slices.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.12;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.06;
        }

        dir_v[cache.slices.marginal.start] = 0.18;
        dir_v[cache.slices.logslope.start] = -0.22;
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
        let prepared = build_deviation_block_from_seed(
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
        dir_u[cache.slices.marginal.start] = 0.4;
        dir_u[cache.slices.logslope.start] = -0.3;
        let w_range = cache.slices.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.15;
        if w_range.len() > 1 {
            dir_u[w_range.start + 1] = -0.07;
        }

        dir_v[cache.slices.marginal.start] = -0.2;
        dir_v[cache.slices.logslope.start] = 0.25;
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
        let score_prepared = build_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_deviation_block_from_seed(
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
        dir_u[cache.slices.marginal.start] = 0.7;
        dir_u[cache.slices.logslope.start] = -0.2;
        let h_range = cache.slices.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.1;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.05;
        }
        let w_range = cache.slices.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.08;

        dir_v[cache.slices.marginal.start] = -0.4;
        dir_v[cache.slices.logslope.start] = 0.3;
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
        let score_prepared = build_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_deviation_block_from_seed(
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
        let prepared = build_deviation_block_from_seed(
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
        let prepared = build_deviation_block_from_seed(
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
        let score_prepared = build_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_deviation_block_from_seed(
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

        let slices = block_slices(&block_states, true, true);
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
        let score_prepared = build_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_deviation_block_from_seed(
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

        let slices = block_slices(&block_states, true, true);
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
        let score_prepared = build_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_deviation_block_from_seed(
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
        dir_u[cache.slices.marginal.start] = 0.7;
        dir_u[cache.slices.logslope.start] = -0.2;
        let h_range = cache.slices.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.1;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.05;
        }
        let w_range = cache.slices.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.08;

        dir_v[cache.slices.marginal.start] = -0.4;
        dir_v[cache.slices.logslope.start] = 0.3;
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
        let score_prepared = build_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_deviation_block_from_seed(
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
        dir_u[cache.slices.marginal.start] = 0.7;
        dir_u[cache.slices.logslope.start] = -0.2;
        let h_range = cache.slices.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.1;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.05;
        }
        let w_range = cache.slices.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.08;

        dir_v[cache.slices.marginal.start] = -0.4;
        dir_v[cache.slices.logslope.start] = 0.3;
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
        let prepared = build_deviation_block_from_seed(
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
        dir_u[cache.slices.marginal.start] = -0.35;
        dir_u[cache.slices.logslope.start] = 0.28;
        let h_range = cache.slices.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.12;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.06;
        }

        dir_v[cache.slices.marginal.start] = 0.18;
        dir_v[cache.slices.logslope.start] = -0.22;
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
        let prepared = build_deviation_block_from_seed(
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
        dir_u[cache.slices.marginal.start] = 0.4;
        dir_u[cache.slices.logslope.start] = -0.3;
        let w_range = cache.slices.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.15;
        if w_range.len() > 1 {
            dir_u[w_range.start + 1] = -0.07;
        }

        dir_v[cache.slices.marginal.start] = -0.2;
        dir_v[cache.slices.logslope.start] = 0.25;
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
        let prepared = build_deviation_block_from_seed(
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
        dir[cache.slices.marginal.start] = -0.35;
        dir[cache.slices.logslope.start] = 0.28;
        let h_range = cache.slices.h.as_ref().expect("h slice");
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
        let prepared = build_deviation_block_from_seed(
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
        dir[cache.slices.marginal.start] = 0.4;
        dir[cache.slices.logslope.start] = -0.3;
        let w_range = cache.slices.w.as_ref().expect("w slice");
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
        let score_prepared = build_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_deviation_block_from_seed(
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

        let slices = block_slices(&block_states, true, true);
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
        let score_prepared = build_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_deviation_block_from_seed(
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

        let slices = block_slices(&block_states, true, true);
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
        let score_prepared = build_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_deviation_block_from_seed(
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

        let slices = block_slices(&block_states, true, true);
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
        let score_prepared = build_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_deviation_block_from_seed(
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
        dir_u[cache.slices.marginal.start] = 0.7;
        dir_u[cache.slices.logslope.start] = -0.2;
        let h_range = cache.slices.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.1;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.05;
        }
        let w_range = cache.slices.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.08;

        dir_v[cache.slices.marginal.start] = -0.4;
        dir_v[cache.slices.logslope.start] = 0.3;
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
        let prepared = build_deviation_block_from_seed(
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
        dir_u[cache.slices.marginal.start] = -0.35;
        dir_u[cache.slices.logslope.start] = 0.28;
        let h_range = cache.slices.h.as_ref().expect("h slice");
        dir_u[h_range.start] = 0.12;
        if h_range.len() > 1 {
            dir_u[h_range.start + 1] = -0.06;
        }

        dir_v[cache.slices.marginal.start] = 0.18;
        dir_v[cache.slices.logslope.start] = -0.22;
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
        let prepared = build_deviation_block_from_seed(
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
        dir_u[cache.slices.marginal.start] = 0.4;
        dir_u[cache.slices.logslope.start] = -0.3;
        let w_range = cache.slices.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.15;
        if w_range.len() > 1 {
            dir_u[w_range.start + 1] = -0.07;
        }

        dir_v[cache.slices.marginal.start] = -0.2;
        dir_v[cache.slices.logslope.start] = 0.25;
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
        let score_prepared = build_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_deviation_block_from_seed(
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

        let slices = block_slices(&block_states, true, true);
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
        let prepared = build_deviation_block_from_seed(
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
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: Some(prepared.runtime.clone()),
            link_dev: None,
        };
        let slices = block_slices(&block_states, true, false);
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
        let prepared = build_deviation_block_from_seed(
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
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: None,
            link_dev: Some(prepared.runtime.clone()),
        };
        let slices = block_slices(&block_states, false, true);
        let total = slices.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[slices.marginal.start] = 0.4;
        dir_u[slices.logslope.start] = -0.3;
        let w_range = slices.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.15;
        if w_range.len() > 1 {
            dir_u[w_range.start + 1] = -0.07;
        }

        dir_v[slices.marginal.start] = -0.2;
        dir_v[slices.logslope.start] = 0.25;
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
        let prepared = build_deviation_block_from_seed(
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
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: Some(prepared.runtime.clone()),
            link_dev: None,
        };
        let slices = block_slices(&block_states, true, false);
        let total = slices.total;
        let mut dir = Array1::<f64>::zeros(total);
        dir[slices.marginal.start] = -0.35;
        dir[slices.logslope.start] = 0.28;
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
        let prepared = build_deviation_block_from_seed(
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
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: None,
            link_dev: Some(prepared.runtime.clone()),
        };
        let slices = block_slices(&block_states, false, true);
        let total = slices.total;
        let mut dir = Array1::<f64>::zeros(total);
        dir[slices.marginal.start] = 0.4;
        dir[slices.logslope.start] = -0.3;
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
        let prepared = build_deviation_block_from_seed(
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
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: Some(prepared.runtime.clone()),
            link_dev: None,
        };
        let slices = block_slices(&block_states, true, false);
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
        let prepared = build_deviation_block_from_seed(
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
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: None,
            link_dev: Some(prepared.runtime.clone()),
        };
        let slices = block_slices(&block_states, false, true);
        let total = slices.total;
        let mut dir_u = Array1::<f64>::zeros(total);
        let mut dir_v = Array1::<f64>::zeros(total);
        dir_u[slices.marginal.start] = 0.4;
        dir_u[slices.logslope.start] = -0.3;
        let w_range = slices.w.as_ref().expect("w slice");
        dir_u[w_range.start] = 0.15;
        if w_range.len() > 1 {
            dir_u[w_range.start + 1] = -0.07;
        }

        dir_v[slices.marginal.start] = -0.2;
        dir_v[slices.logslope.start] = 0.25;
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
        let prepared = build_deviation_block_from_seed(
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
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: Some(prepared.runtime.clone()),
            link_dev: None,
        };
        let slices = block_slices(&block_states, true, false);
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
        let prepared = build_deviation_block_from_seed(
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
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((seed.len(), 0)),
            )),
            score_warp: None,
            link_dev: Some(prepared.runtime.clone()),
        };
        let slices = block_slices(&block_states, false, true);
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
        let link_prepared = build_deviation_block_from_seed(
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
        let score_prepared = build_deviation_block_from_seed(
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
        let score_prepared = build_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_deviation_block_from_seed(
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
        let score_prepared = build_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_deviation_block_from_seed(
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

        let slices = block_slices(&block_states, true, true);
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
    fn flexible_family_exposes_exact_newton_workspaces() {
        let z = array![-0.8, 0.2, 1.1];
        let y = array![0.0, 1.0, 1.0];
        let weights = array![1.0, 0.7, 1.3];
        let score_prepared = build_deviation_block_from_seed(
            &z,
            &DeviationBlockConfig {
                num_internal_knots: 4,
                ..DeviationBlockConfig::default()
            },
        )
        .expect("score warp block");
        let link_seed = array![-2.0, -0.5, 0.0, 0.5, 2.0];
        let link_prepared = build_deviation_block_from_seed(
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
}

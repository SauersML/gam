use crate::basis::BasisOptions;
use crate::basis::BasisFamily;
use crate::estimate::{BlockRole, FittedLinkState, UnifiedFitResult};
use crate::families::gamlss::{
    monotone_wiggle_basis_with_derivative_order, validate_monotone_wiggle_beta_nonnegative,
};
use crate::families::survival_construction::{
    SurvivalBaselineConfig, SurvivalTimeBasisConfig, parse_survival_baseline_config,
};
use crate::inference::predict::{
    BernoulliMarginalSlopePredictor, BinomialLocationScalePredictor,
    GaussianLocationScalePredictor, PredictableModel, StandardPredictor, SurvivalPredictor,
};
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec};
use crate::smooth::{AdaptiveRegularizationDiagnostics, TermCollectionSpec};
use crate::types::{
    InverseLink, LikelihoodFamily, LinkFunction, MixtureLinkState, SasLinkSpec, SasLinkState,
};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fs;
use std::ops::{Deref, DerefMut};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataSchema {
    pub columns: Vec<SchemaColumn>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchemaColumn {
    pub name: String,
    pub kind: ColumnKindTag,
    #[serde(default)]
    pub levels: Vec<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ColumnKindTag {
    Continuous,
    Binary,
    Categorical,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct FittedModelPayload {
    pub version: u32,
    pub formula: String,
    pub model_kind: ModelKind,
    pub family_state: FittedFamily,
    pub family: String,
    #[serde(default)]
    pub fit_result: Option<UnifiedFitResult>,
    /// Unified (family-agnostic) representation of the fit result.
    #[serde(default)]
    pub unified: Option<UnifiedFitResult>,
    #[serde(default)]
    pub data_schema: Option<DataSchema>,
    pub link: Option<String>,
    #[serde(default)]
    pub mixture_link_param_covariance: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub sas_param_covariance: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub formula_noise: Option<String>,
    #[serde(default)]
    pub formula_logslope: Option<String>,
    #[serde(default)]
    pub beta_noise: Option<Vec<f64>>,
    #[serde(default)]
    pub noise_projection: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub noise_center: Option<Vec<f64>>,
    #[serde(default)]
    pub noise_scale: Option<Vec<f64>>,
    #[serde(default)]
    pub noise_non_intercept_start: Option<usize>,
    #[serde(default)]
    pub gaussian_response_scale: Option<f64>,
    #[serde(default)]
    pub linkwiggle_knots: Option<Vec<f64>>,
    #[serde(default)]
    pub linkwiggle_degree: Option<usize>,
    #[serde(default)]
    pub beta_link_wiggle: Option<Vec<f64>>,
    #[serde(default)]
    pub baseline_timewiggle_knots: Option<Vec<f64>>,
    #[serde(default)]
    pub baseline_timewiggle_degree: Option<usize>,
    #[serde(default)]
    pub baseline_timewiggle_penalty_orders: Option<Vec<usize>>,
    #[serde(default)]
    pub baseline_timewiggle_double_penalty: Option<bool>,
    #[serde(default)]
    pub beta_baseline_timewiggle: Option<Vec<f64>>,
    #[serde(default)]
    pub z_column: Option<String>,
    #[serde(default)]
    pub marginal_baseline: Option<f64>,
    #[serde(default)]
    pub logslope_baseline: Option<f64>,
    #[serde(default)]
    pub score_warp_runtime: Option<SavedAnchoredDeviationRuntime>,
    #[serde(default)]
    pub link_deviation_runtime: Option<SavedAnchoredDeviationRuntime>,
    #[serde(default)]
    pub survival_entry: Option<String>,
    #[serde(default)]
    pub survival_exit: Option<String>,
    #[serde(default)]
    pub survival_event: Option<String>,
    #[serde(default)]
    pub survivalspec: Option<String>,
    #[serde(default)]
    pub survival_baseline_target: Option<String>,
    #[serde(default)]
    pub survival_baseline_scale: Option<f64>,
    #[serde(default)]
    pub survival_baseline_shape: Option<f64>,
    #[serde(default)]
    pub survival_baseline_rate: Option<f64>,
    #[serde(default)]
    pub survival_baseline_makeham: Option<f64>,
    #[serde(default)]
    pub survival_time_basis: Option<String>,
    #[serde(default)]
    pub survival_time_degree: Option<usize>,
    #[serde(default)]
    pub survival_time_knots: Option<Vec<f64>>,
    #[serde(default)]
    pub survival_time_keep_cols: Option<Vec<usize>>,
    #[serde(default)]
    pub survival_time_smooth_lambda: Option<f64>,
    #[serde(default)]
    pub survival_time_anchor: Option<f64>,
    #[serde(default)]
    pub survivalridge_lambda: Option<f64>,
    #[serde(default)]
    pub survival_likelihood: Option<String>,
    #[serde(default)]
    pub survival_beta_time: Option<Vec<f64>>,
    #[serde(default)]
    pub survival_beta_threshold: Option<Vec<f64>>,
    #[serde(default)]
    pub survival_beta_log_sigma: Option<Vec<f64>>,
    #[serde(default)]
    pub survival_noise_projection: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub survival_noise_center: Option<Vec<f64>>,
    #[serde(default)]
    pub survival_noise_scale: Option<Vec<f64>>,
    #[serde(default)]
    pub survival_noise_non_intercept_start: Option<usize>,
    #[serde(default)]
    pub survival_distribution: Option<String>,
    #[serde(default)]
    pub training_headers: Option<Vec<String>>,
    /// Transformation-normal: B-spline knots for the response-direction basis.
    #[serde(default)]
    pub transformation_response_knots: Option<Vec<f64>>,
    /// Transformation-normal: deviation nullspace transform matrix (row-major).
    #[serde(default)]
    pub transformation_response_transform: Option<Vec<Vec<f64>>>,
    /// Transformation-normal: B-spline degree for the response basis.
    #[serde(default)]
    pub transformation_response_degree: Option<usize>,
    /// Transformation-normal: median of the response used for anchoring.
    #[serde(default)]
    pub transformation_response_median: Option<f64>,
    #[serde(default)]
    pub resolved_termspec: Option<TermCollectionSpec>,
    #[serde(default)]
    pub resolved_termspec_noise: Option<TermCollectionSpec>,
    #[serde(default)]
    pub adaptive_regularization_diagnostics: Option<AdaptiveRegularizationDiagnostics>,
}

impl FittedModelPayload {
    pub fn new(
        version: u32,
        formula: String,
        model_kind: ModelKind,
        family_state: FittedFamily,
        family: String,
    ) -> Self {
        Self {
            version,
            formula,
            model_kind,
            family_state,
            family,
            fit_result: None,
            unified: None,
            data_schema: None,
            link: None,
            mixture_link_param_covariance: None,
            sas_param_covariance: None,
            formula_noise: None,
            formula_logslope: None,
            beta_noise: None,
            noise_projection: None,
            noise_center: None,
            noise_scale: None,
            noise_non_intercept_start: None,
            gaussian_response_scale: None,
            linkwiggle_knots: None,
            linkwiggle_degree: None,
            beta_link_wiggle: None,
            baseline_timewiggle_knots: None,
            baseline_timewiggle_degree: None,
            baseline_timewiggle_penalty_orders: None,
            baseline_timewiggle_double_penalty: None,
            beta_baseline_timewiggle: None,
            z_column: None,
            marginal_baseline: None,
            logslope_baseline: None,
            score_warp_runtime: None,
            link_deviation_runtime: None,
            survival_entry: None,
            survival_exit: None,
            survival_event: None,
            survivalspec: None,
            survival_baseline_target: None,
            survival_baseline_scale: None,
            survival_baseline_shape: None,
            survival_baseline_rate: None,
            survival_baseline_makeham: None,
            survival_time_basis: None,
            survival_time_degree: None,
            survival_time_knots: None,
            survival_time_keep_cols: None,
            survival_time_smooth_lambda: None,
            survival_time_anchor: None,
            survivalridge_lambda: None,
            survival_likelihood: None,
            survival_beta_time: None,
            survival_beta_threshold: None,
            survival_beta_log_sigma: None,
            survival_noise_projection: None,
            survival_noise_center: None,
            survival_noise_scale: None,
            survival_noise_non_intercept_start: None,
            survival_distribution: None,
            training_headers: None,
            transformation_response_knots: None,
            transformation_response_transform: None,
            transformation_response_degree: None,
            transformation_response_median: None,
            resolved_termspec: None,
            resolved_termspec_noise: None,
            adaptive_regularization_diagnostics: None,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "model_type", rename_all = "kebab-case")]
pub enum FittedModel {
    Standard { payload: FittedModelPayload },
    LocationScale { payload: FittedModelPayload },
    MarginalSlope { payload: FittedModelPayload },
    Survival { payload: FittedModelPayload },
    TransformationNormal { payload: FittedModelPayload },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ModelKind {
    Standard,
    LocationScale,
    MarginalSlope,
    Survival,
    TransformationNormal,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "family_kind", rename_all = "kebab-case")]
pub enum FittedFamily {
    Standard {
        likelihood: LikelihoodFamily,
        link: Option<LinkFunction>,
        #[serde(default)]
        mixture_state: Option<MixtureLinkState>,
        #[serde(default)]
        sas_state: Option<SasLinkState>,
    },
    LocationScale {
        likelihood: LikelihoodFamily,
        #[serde(default)]
        base_link: Option<InverseLink>,
    },
    MarginalSlope {
        likelihood: LikelihoodFamily,
    },
    Survival {
        likelihood: LikelihoodFamily,
        #[serde(default)]
        survival_likelihood: Option<String>,
        #[serde(default)]
        survival_distribution: Option<String>,
    },
    TransformationNormal {
        likelihood: LikelihoodFamily,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredictModelClass {
    Standard,
    GaussianLocationScale,
    BinomialLocationScale,
    BernoulliMarginalSlope,
    Survival,
    TransformationNormal,
}

#[derive(Clone, Debug)]
pub struct SavedLinkWiggleRuntime {
    pub knots: Vec<f64>,
    pub degree: usize,
    pub beta: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct SavedBaselineTimeWiggleRuntime {
    pub knots: Vec<f64>,
    pub degree: usize,
    pub penalty_orders: Vec<usize>,
    pub double_penalty: bool,
    pub beta: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedAnchoredDeviationRuntime {
    pub kernel: String,
    pub knots: Vec<f64>,
    pub degree: usize,
    pub basis_dim: usize,
}

#[derive(Clone, Debug)]
pub struct SavedPredictionRuntime {
    pub model_class: PredictModelClass,
    pub likelihood: LikelihoodFamily,
    pub inverse_link: Option<InverseLink>,
    pub link_wiggle: Option<SavedLinkWiggleRuntime>,
    pub baseline_time_wiggle: Option<SavedBaselineTimeWiggleRuntime>,
    pub score_warp: Option<SavedAnchoredDeviationRuntime>,
    pub link_deviation: Option<SavedAnchoredDeviationRuntime>,
}

impl SavedLinkWiggleRuntime {
    fn validate_global_monotonicity(&self) -> Result<(), String> {
        validate_monotone_wiggle_beta_nonnegative(&self.beta, "saved link-wiggle")
    }

    fn validate_monotone_derivative(&self, q0: &Array1<f64>) -> Result<Array1<f64>, String> {
        self.validate_global_monotonicity()?;
        let d_constrained = self.constrained_basis(q0, BasisOptions::first_derivative())?;
        let beta_link_wiggle = Array1::from_vec(self.beta.clone());
        let dq_dq0 = d_constrained.dot(&beta_link_wiggle) + 1.0;
        if let Some((idx, value)) = dq_dq0.iter().copied().enumerate().find(|(_, v)| *v <= 0.0) {
            return Err(format!(
                "saved link-wiggle is not monotone at row {idx}: dq/dq0={value:.3e} <= 0"
            ));
        }
        Ok(dq_dq0)
    }

    pub fn constrained_basis(
        &self,
        q0: &Array1<f64>,
        basis_options: BasisOptions,
    ) -> Result<Array2<f64>, String> {
        let knot_arr = Array1::from_vec(self.knots.clone());
        let constrained = monotone_wiggle_basis_with_derivative_order(
            q0.view(),
            &knot_arr,
            self.degree,
            basis_options.derivative_order,
        )?;
        if constrained.ncols() != self.beta.len() {
            return Err(format!(
                "saved link-wiggle dimension mismatch: coefficients have {} entries but basis has {} columns",
                self.beta.len(),
                constrained.ncols()
            ));
        }
        Ok(constrained)
    }

    pub fn design(&self, q0: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.validate_global_monotonicity()?;
        self.constrained_basis(q0, BasisOptions::value())
    }

    pub fn basis_row_scalar(&self, q0: f64) -> Result<Array1<f64>, String> {
        let q = Array1::from_vec(vec![q0]);
        let x = self.design(&q)?;
        if x.nrows() != 1 {
            return Err(format!(
                "saved link-wiggle scalar evaluation expected 1 row, got {}",
                x.nrows()
            ));
        }
        Ok(x.row(0).to_owned())
    }

    pub fn apply(&self, q0: &Array1<f64>) -> Result<Array1<f64>, String> {
        self.validate_monotone_derivative(q0)?;
        let xwiggle = self.constrained_basis(q0, BasisOptions::value())?;
        let beta_link_wiggle = Array1::from_vec(self.beta.clone());
        Ok(q0 + &xwiggle.dot(&beta_link_wiggle))
    }

    pub fn derivative_q0(&self, q0: &Array1<f64>) -> Result<Array1<f64>, String> {
        self.validate_monotone_derivative(q0)
    }
}

impl SavedBaselineTimeWiggleRuntime {
    pub fn validate_global_monotonicity(&self) -> Result<(), String> {
        validate_monotone_wiggle_beta_nonnegative(&self.beta, "saved baseline-timewiggle")
    }
}

impl SavedAnchoredDeviationRuntime {
    fn validate_kernel(&self) -> Result<(), String> {
        if self.kernel.is_empty()
            || self.kernel
                == crate::families::bernoulli_marginal_slope::exact_kernel::LEGACY_ANCHORED_DEVIATION_KERNEL
        {
            return Err(
                "saved anchored deviation runtime is missing the ExactDenestedCubicV1 marker or uses legacy flexible Bernoulli marginal-slope semantics; the model must be refit"
                    .to_string(),
            );
        }
        if self.kernel
            != crate::families::bernoulli_marginal_slope::exact_kernel::ANCHORED_DEVIATION_KERNEL
        {
            return Err(format!(
                "saved anchored deviation runtime uses unsupported kernel '{}'; expected {}",
                self.kernel,
                crate::families::bernoulli_marginal_slope::exact_kernel::ANCHORED_DEVIATION_KERNEL
            ));
        }
        if self.degree != 3 {
            return Err(format!(
                "saved anchored deviation runtime must be cubic (degree=3), got degree={}",
                self.degree
            ));
        }
        Ok(())
    }

    fn constrained_basis(
        &self,
        values: &Array1<f64>,
        basis_options: BasisOptions,
    ) -> Result<Array2<f64>, String> {
        self.validate_kernel()?;
        let knot_arr = Array1::from_vec(self.knots.clone());
        let constrained = monotone_wiggle_basis_with_derivative_order(
            values.view(),
            &knot_arr,
            self.degree,
            basis_options.derivative_order,
        )?;
        if constrained.ncols() != self.basis_dim {
            return Err(format!(
                "saved anchored deviation basis mismatch: runtime expects {} columns but basis has {}",
                self.basis_dim,
                constrained.ncols()
            ));
        }
        Ok(constrained)
    }

    pub fn breakpoints(&self) -> Result<Vec<f64>, String> {
        self.validate_kernel()?;
        if self.knots.is_empty() {
            return Err("saved anchored deviation runtime is missing knots".to_string());
        }
        let mut points = Vec::new();
        for &knot in &self.knots {
            if points
                .last()
                .is_none_or(|prev: &f64| (knot - *prev).abs() > 1e-12)
            {
                points.push(knot);
            }
        }
        if points.len() < 2 {
            return Err(
                "saved anchored deviation runtime requires at least two distinct breakpoints"
                    .to_string(),
            );
        }
        Ok(points)
    }

    pub fn span_count(&self) -> Result<usize, String> {
        Ok(self.breakpoints()?.windows(2).count())
    }

    pub fn span_index_for(&self, value: f64) -> Result<usize, String> {
        if !value.is_finite() {
            return Err(format!(
                "saved anchored deviation span lookup requires finite value, got {value}"
            ));
        }
        let points = self.breakpoints()?;
        let n = points.len() - 1;
        if value <= points[0] {
            return Ok(0);
        }
        if value >= points[n] {
            return Ok(n - 1);
        }
        for idx in 0..n {
            if value >= points[idx] && value <= points[idx + 1] {
                return Ok(idx);
            }
        }
        Err(format!(
            "saved anchored deviation span lookup failed for value {value}; support is [{:.6}, {:.6}]",
            points[0], points[n]
        ))
    }

    pub fn local_cubic_on_span(
        &self,
        beta: &Array1<f64>,
        span_idx: usize,
    ) -> Result<crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic, String> {
        self.validate_kernel()?;
        if beta.len() != self.basis_dim {
            return Err(format!(
                "saved anchored deviation coefficient length mismatch: got {}, expected {}",
                beta.len(),
                self.basis_dim
            ));
        }
        let points = self.breakpoints()?;
        if span_idx + 1 >= points.len() {
            return Err(format!(
                "saved anchored deviation span index {} out of range for {} spans",
                span_idx,
                points.len() - 1
            ));
        }
        let left = points[span_idx];
        let right = points[span_idx + 1];
        let left_point = Array1::from_vec(vec![left]);
        let mid_point = Array1::from_vec(vec![0.5 * (left + right)]);
        let value = self.design(&left_point)?.row(0).dot(beta);
        let d1 = self.first_derivative_design(&left_point)?.row(0).dot(beta);
        let d2 = self.second_derivative_design(&left_point)?.row(0).dot(beta);
        let d3 = self.third_derivative_design(&mid_point)?.row(0).dot(beta);
        Ok(crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic {
            left,
            right,
            c0: value,
            c1: d1,
            c2: 0.5 * d2,
            c3: d3 / 6.0,
        })
    }

    pub fn basis_span_cubic(
        &self,
        span_idx: usize,
        basis_idx: usize,
    ) -> Result<crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic, String> {
        self.validate_kernel()?;
        if basis_idx >= self.basis_dim {
            return Err(format!(
                "saved anchored deviation basis index {} out of range for {} coefficients",
                basis_idx, self.basis_dim
            ));
        }
        let mut beta = Array1::<f64>::zeros(self.basis_dim);
        beta[basis_idx] = 1.0;
        self.local_cubic_on_span(&beta, span_idx)
    }

    pub fn basis_cubic_at(
        &self,
        basis_idx: usize,
        value: f64,
    ) -> Result<crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic, String> {
        let span_idx = self.span_index_for(value)?;
        self.basis_span_cubic(span_idx, basis_idx)
    }

    pub fn local_cubic_at(
        &self,
        beta: &Array1<f64>,
        value: f64,
    ) -> Result<crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic, String> {
        let span_idx = self.span_index_for(value)?;
        self.local_cubic_on_span(beta, span_idx)
    }

    pub fn design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.constrained_basis(values, BasisOptions::value())
    }

    pub fn first_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.constrained_basis(values, BasisOptions::first_derivative())
    }

    pub fn second_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.constrained_basis(values, BasisOptions::second_derivative())
    }

    pub fn third_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.constrained_basis(
            values,
            BasisOptions {
                derivative_order: 3,
                basis_family: BasisFamily::BSpline,
            },
        )
    }

    pub fn fourth_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.constrained_basis(
            values,
            BasisOptions {
                derivative_order: 4,
                basis_family: BasisFamily::BSpline,
            },
        )
    }
}

fn saved_link_name_disallows_wiggle(link_name: &str) -> bool {
    let link_name = link_name.trim().to_ascii_lowercase();
    link_name == "sas"
        || link_name == "beta-logistic"
        || link_name.starts_with("blended(")
        || link_name.starts_with("mixture(")
}

fn inverse_link_disallows_wiggle(link: &InverseLink) -> bool {
    matches!(
        link,
        InverseLink::Standard(LinkFunction::Sas)
            | InverseLink::Standard(LinkFunction::BetaLogistic)
            | InverseLink::Sas(_)
            | InverseLink::BetaLogistic(_)
            | InverseLink::Mixture(_)
    )
}

impl FittedFamily {
    #[inline]
    pub fn likelihood(&self) -> LikelihoodFamily {
        match self {
            Self::Standard { likelihood, .. }
            | Self::LocationScale { likelihood, .. }
            | Self::MarginalSlope { likelihood, .. }
            | Self::Survival { likelihood, .. }
            | Self::TransformationNormal { likelihood, .. } => *likelihood,
        }
    }
}

impl FittedModel {
    pub fn from_payload(mut payload: FittedModelPayload) -> Self {
        let likelihood = payload.family_state.likelihood();
        let class = match payload.model_kind {
            ModelKind::Survival => PredictModelClass::Survival,
            ModelKind::MarginalSlope => PredictModelClass::BernoulliMarginalSlope,
            ModelKind::TransformationNormal => PredictModelClass::TransformationNormal,
            ModelKind::LocationScale => {
                if likelihood == LikelihoodFamily::GaussianIdentity {
                    PredictModelClass::GaussianLocationScale
                } else {
                    PredictModelClass::BinomialLocationScale
                }
            }
            _ => PredictModelClass::Standard,
        };
        match class {
            PredictModelClass::Survival => {
                payload.model_kind = ModelKind::Survival;
                Self::Survival { payload }
            }
            PredictModelClass::BernoulliMarginalSlope => {
                payload.model_kind = ModelKind::MarginalSlope;
                Self::MarginalSlope { payload }
            }
            PredictModelClass::TransformationNormal => {
                payload.model_kind = ModelKind::TransformationNormal;
                Self::TransformationNormal { payload }
            }
            PredictModelClass::GaussianLocationScale | PredictModelClass::BinomialLocationScale => {
                payload.model_kind = ModelKind::LocationScale;
                Self::LocationScale { payload }
            }
            PredictModelClass::Standard => {
                payload.model_kind = ModelKind::Standard;
                Self::Standard { payload }
            }
        }
        .with_synchronized_stateful_link_metadata()
    }

    #[inline]
    pub fn payload(&self) -> &FittedModelPayload {
        match self {
            Self::Standard { payload }
            | Self::LocationScale { payload }
            | Self::MarginalSlope { payload }
            | Self::Survival { payload }
            | Self::TransformationNormal { payload } => payload,
        }
    }

    #[inline]
    fn payload_mut(&mut self) -> &mut FittedModelPayload {
        match self {
            Self::Standard { payload }
            | Self::LocationScale { payload }
            | Self::MarginalSlope { payload }
            | Self::Survival { payload }
            | Self::TransformationNormal { payload } => payload,
        }
    }

    fn with_synchronized_stateful_link_metadata(mut self) -> Self {
        self.synchronize_stateful_link_metadata();
        self
    }

    fn synchronize_stateful_link_metadata(&mut self) {
        let payload = self.payload_mut();
        let Some(fit) = payload.fit_result.as_ref() else {
            return;
        };
        match (&mut payload.family_state, &fit.fitted_link) {
            (
                FittedFamily::Standard {
                    likelihood: LikelihoodFamily::BinomialSas,
                    sas_state,
                    ..
                },
                FittedLinkState::Sas { state, covariance },
            ) => {
                *sas_state = Some(*state);
                payload.sas_param_covariance = covariance.as_ref().map(array2_to_nestedvec);
            }
            (
                FittedFamily::Standard {
                    likelihood: LikelihoodFamily::BinomialBetaLogistic,
                    sas_state,
                    ..
                },
                FittedLinkState::BetaLogistic { state, covariance },
            ) => {
                *sas_state = Some(*state);
                payload.sas_param_covariance = covariance.as_ref().map(array2_to_nestedvec);
            }
            (
                FittedFamily::Standard {
                    likelihood: LikelihoodFamily::BinomialMixture,
                    mixture_state,
                    ..
                },
                FittedLinkState::Mixture { state, covariance },
            ) => {
                *mixture_state = Some(state.clone());
                payload.mixture_link_param_covariance =
                    covariance.as_ref().map(array2_to_nestedvec);
            }
            _ => {}
        }
    }

    #[inline]
    pub fn likelihood(&self) -> LikelihoodFamily {
        self.payload().family_state.likelihood()
    }

    #[inline]
    pub fn predict_model_class(&self) -> PredictModelClass {
        match self.payload().family_state {
            FittedFamily::Survival { .. } => PredictModelClass::Survival,
            FittedFamily::MarginalSlope { .. } => PredictModelClass::BernoulliMarginalSlope,
            FittedFamily::TransformationNormal { .. } => PredictModelClass::TransformationNormal,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::GaussianIdentity,
                ..
            } => PredictModelClass::GaussianLocationScale,
            FittedFamily::LocationScale { .. } => PredictModelClass::BinomialLocationScale,
            _ => PredictModelClass::Standard,
        }
    }

    #[inline]
    pub fn is_survival_model(&self) -> bool {
        matches!(self.predict_model_class(), PredictModelClass::Survival)
    }

    pub fn saved_link_wiggle(&self) -> Result<Option<SavedLinkWiggleRuntime>, String> {
        let payload = self.payload();
        let (knots, degree) = match (
            payload.linkwiggle_knots.as_ref(),
            payload.linkwiggle_degree,
        ) {
            (None, None) => return Ok(None),
            (Some(knots), Some(degree)) => (knots.clone(), degree),
            _ => {
                return Err(
                    "saved model has partial link-wiggle metadata; expected linkwiggle_knots and linkwiggle_degree together"
                        .to_string(),
                )
            }
        };
        let resolved_link = self.resolved_inverse_link()?;
        let saved_link_disallows_wiggle = resolved_link
            .as_ref()
            .is_some_and(inverse_link_disallows_wiggle)
            || payload
                .link
                .as_deref()
                .is_some_and(saved_link_name_disallows_wiggle);
        if saved_link_disallows_wiggle {
            return Err(
                "link wiggle does not support SAS/BetaLogistic/Mixture links; refit without wiggle or with a jointly fitted standard link"
                    .to_string(),
            );
        }
        let beta = match self.predict_model_class() {
            PredictModelClass::Standard => {
                if payload.beta_link_wiggle.is_some() {
                    return Err(
                        "standard link-wiggle coefficients must be stored in fit_result LinkWiggle block, not payload.beta_link_wiggle"
                            .to_string(),
                    );
                }
                let fit = payload.fit_result.as_ref().ok_or_else(|| {
                    "standard link-wiggle model is missing canonical fit_result payload".to_string()
                })?;
                if fit.blocks.len() != 2
                    || fit.blocks[0].role != BlockRole::Mean
                    || fit.blocks[1].role != BlockRole::LinkWiggle
                {
                    return Err(
                        "standard link-wiggle models must store blocks in [Mean, LinkWiggle] order"
                            .to_string(),
                    );
                }
                fit.block_by_role(BlockRole::LinkWiggle)
                    .ok_or_else(|| {
                        "standard link-wiggle model is missing LinkWiggle coefficient block"
                            .to_string()
                    })?
                    .beta
                    .to_vec()
            }
            _ => payload.beta_link_wiggle.clone().ok_or_else(|| {
                "saved model has link-wiggle metadata but is missing payload.beta_link_wiggle"
                    .to_string()
            })?,
        };
        Ok(Some(SavedLinkWiggleRuntime {
            knots,
            degree,
            beta,
        }))
    }

    pub fn saved_baseline_time_wiggle(
        &self,
    ) -> Result<Option<SavedBaselineTimeWiggleRuntime>, String> {
        let payload = self.payload();
        match (
            payload.baseline_timewiggle_knots.as_ref(),
            payload.baseline_timewiggle_degree,
            payload.baseline_timewiggle_penalty_orders.as_ref(),
            payload.baseline_timewiggle_double_penalty,
            payload.beta_baseline_timewiggle.as_ref(),
        ) {
            (None, None, None, None, None) => Ok(None),
            (Some(knots), Some(degree), Some(penalty_orders), Some(double_penalty), Some(beta)) => {
                Ok(Some(SavedBaselineTimeWiggleRuntime {
                    knots: knots.clone(),
                    degree,
                    penalty_orders: penalty_orders.clone(),
                    double_penalty,
                    beta: beta.clone(),
                }))
            }
            _ => Err(
                "saved model has partial baseline-timewiggle metadata; expected knots+degree+penalty_order+double_penalty+beta_baseline_timewiggle together"
                    .to_string(),
            ),
        }
    }

    /// Whether this model has a link wiggle component with complete metadata.
    #[inline]
    pub fn has_link_wiggle(&self) -> bool {
        self.saved_link_wiggle()
            .map(|runtime| runtime.is_some())
            .unwrap_or(false)
    }

    /// Whether this model has a baseline-time wiggle component with complete metadata.
    #[inline]
    pub fn has_baseline_time_wiggle(&self) -> bool {
        self.saved_baseline_time_wiggle()
            .map(|runtime| runtime.is_some())
            .unwrap_or(false)
    }

    pub fn saved_prediction_runtime(&self) -> Result<SavedPredictionRuntime, String> {
        if self.predict_model_class() == PredictModelClass::BernoulliMarginalSlope {
            if let Some(runtime) = self.payload().score_warp_runtime.as_ref() {
                runtime
                    .validate_kernel()
                    .map_err(|err| format!("saved bernoulli marginal-slope score-warp runtime is incompatible: {err}. Refit the model."))?;
            }
            if let Some(runtime) = self.payload().link_deviation_runtime.as_ref() {
                runtime
                    .validate_kernel()
                    .map_err(|err| format!("saved bernoulli marginal-slope link-deviation runtime is incompatible: {err}. Refit the model."))?;
            }
        }
        Ok(SavedPredictionRuntime {
            model_class: self.predict_model_class(),
            likelihood: self.likelihood(),
            inverse_link: self.resolved_inverse_link()?,
            link_wiggle: self.saved_link_wiggle()?,
            baseline_time_wiggle: self.saved_baseline_time_wiggle()?,
            score_warp: self.payload().score_warp_runtime.clone(),
            link_deviation: self.payload().link_deviation_runtime.clone(),
        })
    }

    pub fn saved_sas_state(&self) -> Result<Option<SasLinkState>, String> {
        let payload = self.payload();
        let raw = match &payload.family_state {
            FittedFamily::Standard {
                likelihood: LikelihoodFamily::BinomialSas,
                sas_state,
                ..
            } => (*sas_state).ok_or_else(|| {
                "binomial-sas model is missing state in family_state.sas_state".to_string()
            })?,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialSas,
                base_link,
            } => match base_link {
                Some(InverseLink::Sas(state)) => *state,
                _ => {
                    return Err(
                        "binomial-sas location-scale model is missing SAS base_link state"
                            .to_string(),
                    );
                }
            },
            _ => return Ok(None),
        };
        state_from_sasspec(SasLinkSpec {
            initial_epsilon: raw.epsilon,
            initial_log_delta: raw.log_delta,
        })
        .map(Some)
        .map_err(|e| format!("invalid saved SAS link state: {e}"))
    }

    pub fn saved_beta_logistic_state(&self) -> Result<Option<SasLinkState>, String> {
        let payload = self.payload();
        let raw = match &payload.family_state {
            FittedFamily::Standard {
                likelihood: LikelihoodFamily::BinomialBetaLogistic,
                sas_state,
                ..
            } => (*sas_state).ok_or_else(|| {
                "binomial-beta-logistic model is missing state in family_state.sas_state"
                    .to_string()
            })?,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialBetaLogistic,
                base_link,
            } => match base_link {
                Some(InverseLink::BetaLogistic(state)) => *state,
                _ => {
                    return Err(
                        "binomial-beta-logistic location-scale model is missing beta-logistic base_link state"
                            .to_string(),
                    );
                }
            },
            _ => return Ok(None),
        };
        state_from_beta_logisticspec(SasLinkSpec {
            initial_epsilon: raw.epsilon,
            initial_log_delta: raw.log_delta,
        })
        .map(Some)
        .map_err(|e| format!("invalid saved Beta-Logistic link state: {e}"))
    }

    pub fn saved_mixture_state(&self) -> Result<Option<MixtureLinkState>, String> {
        let payload = self.payload();
        match &payload.family_state {
            FittedFamily::Standard {
                likelihood: LikelihoodFamily::BinomialMixture,
                mixture_state,
                ..
            } => mixture_state
                .clone()
                .ok_or_else(|| {
                    "binomial-mixture model is missing state in family_state.mixture_state"
                        .to_string()
                })
                .map(Some),
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialMixture,
                base_link,
            } => match base_link {
                Some(InverseLink::Mixture(state)) => Ok(Some(state.clone())),
                _ => Err(
                    "binomial-mixture location-scale model is missing mixture base_link state"
                        .to_string(),
                ),
            },
            _ => Ok(None),
        }
    }

    pub fn resolved_inverse_link(&self) -> Result<Option<InverseLink>, String> {
        let stateful = if let Some(state) = self.saved_mixture_state()? {
            Some(InverseLink::Mixture(state))
        } else if let Some(state) = self.saved_beta_logistic_state()? {
            Some(InverseLink::BetaLogistic(state))
        } else {
            self.saved_sas_state()?.map(InverseLink::Sas)
        };
        match &self.payload().family_state {
            FittedFamily::LocationScale { base_link, .. } => Ok(base_link.clone().or(stateful)),
            FittedFamily::Standard { link, .. } => {
                Ok(stateful.or_else(|| link.map(InverseLink::Standard)))
            }
            FittedFamily::MarginalSlope { .. } => Ok(None),
            FittedFamily::Survival { .. } => Ok(None),
            FittedFamily::TransformationNormal { .. } => Ok(None),
        }
    }

    /// Build a validated predictor for the saved model shape and runtime.
    ///
    /// Survival models still go through specialised top-level prediction
    /// assembly because they need time-basis construction from saved metadata.
    pub fn predictor(&self) -> Option<Box<dyn PredictableModel>> {
        let runtime = self.saved_prediction_runtime().ok()?;
        match self.predict_model_class() {
            PredictModelClass::GaussianLocationScale => {
                let fit = self.fit_result.as_ref()?;
                let beta_mu = fit.beta.clone();
                let beta_noise = Array1::from_vec(self.payload().beta_noise.clone()?);
                let response_scale = self.gaussian_response_scale.unwrap_or(1.0);
                Some(Box::new(GaussianLocationScalePredictor {
                    beta_mu,
                    beta_noise,
                    response_scale,
                    covariance: fit.beta_covariance().cloned(),
                    link_wiggle: runtime.link_wiggle,
                }) as Box<dyn PredictableModel>)
            }
            PredictModelClass::Standard => {
                let family = self.family_state.likelihood();
                let link_kind = self.resolved_inverse_link().ok().flatten();
                let fit = self.fit_result.as_ref()?;
                let beta = if runtime.link_wiggle.is_some() {
                    fit.block_by_role(BlockRole::Mean)?.beta.clone()
                } else if let Some(unified) = self.unified() {
                    StandardPredictor::from_unified(unified, family, link_kind.clone(), None)
                        .ok()
                        .map(|p| p.beta)
                        .unwrap_or_else(|| fit.beta.clone())
                } else {
                    fit.beta.clone()
                };
                let covariance = fit.beta_covariance().cloned();
                Some(Box::new(StandardPredictor {
                    beta,
                    family,
                    link_kind,
                    covariance,
                    link_wiggle: runtime.link_wiggle,
                }))
            }
            PredictModelClass::Survival => {
                let unified = self.unified()?;
                // Default to probit inverse link for survival models.
                let inverse_link = self
                    .resolved_inverse_link()
                    .ok()
                    .flatten()
                    .unwrap_or(InverseLink::Standard(LinkFunction::Probit));
                SurvivalPredictor::from_unified(unified, inverse_link)
                    .ok()
                    .map(|p| Box::new(p) as Box<dyn PredictableModel>)
            }
            PredictModelClass::BinomialLocationScale => {
                let inverse_link = self
                    .resolved_inverse_link()
                    .ok()
                    .flatten()
                    .unwrap_or(InverseLink::Standard(LinkFunction::Probit));
                let fit = self.fit_result.as_ref()?;
                let beta_threshold = fit.beta.clone();
                let beta_noise = Array1::from_vec(self.payload().beta_noise.clone()?);
                Some(Box::new(BinomialLocationScalePredictor {
                    beta_threshold,
                    beta_noise,
                    covariance: fit.beta_covariance().cloned(),
                    inverse_link,
                    link_wiggle: runtime.link_wiggle,
                }) as Box<dyn PredictableModel>)
            }
            PredictModelClass::BernoulliMarginalSlope => {
                let unified = self.unified()?;
                let payload = self.payload();
                let z_column = payload.z_column.clone()?;
                let predictor = BernoulliMarginalSlopePredictor::from_unified(
                    unified,
                    z_column,
                    payload.marginal_baseline?,
                    payload.logslope_baseline?,
                    runtime.score_warp,
                    runtime.link_deviation,
                )
                .ok()?;
                Some(Box::new(predictor) as Box<dyn PredictableModel>)
            }
            PredictModelClass::TransformationNormal => {
                // The h values are computed in build_predict_input_for_model
                // and stored in the offset field. The predictor is a simple
                // identity: eta = offset, mean = offset (h is already the
                // PIT-transformed value on the standard normal scale).
                let fit = self.fit_result.as_ref()?;
                Some(Box::new(super::predict::TransformationNormalPredictor {
                    covariance: fit.beta_covariance().cloned(),
                }) as Box<dyn PredictableModel>)
            }
        }
    }

    /// Returns the block roles for this model via the `PredictableModel` trait.
    ///
    /// For standard models this is `[BlockRole::Mean]`.
    pub fn block_roles(&self) -> Option<Vec<BlockRole>> {
        self.predictor().map(|p| p.block_roles())
    }

    /// Access the unified fit result, if stored.
    pub fn unified(&self) -> Option<&UnifiedFitResult> {
        self.payload().unified.as_ref()
    }

    pub fn load_from_path(path: &Path) -> Result<Self, String> {
        let payload = fs::read_to_string(path)
            .map_err(|e| format!("failed to read model '{}': {e}", path.display()))?;
        let model: Self = serde_json::from_str(&payload)
            .map_err(|e| format!("failed to parse model json: {e}"))?;
        let model = model.with_synchronized_stateful_link_metadata();
        model.validate_for_persistence()?;
        model.validate_numeric_finiteness()?;
        Ok(model)
    }

    pub fn save_to_path(&self, path: &Path) -> Result<(), String> {
        let normalized = self.clone().with_synchronized_stateful_link_metadata();
        normalized.validate_for_persistence()?;
        normalized.validate_numeric_finiteness()?;
        let payload = serde_json::to_string_pretty(&normalized)
            .map_err(|e| format!("failed to serialize model: {e}"))?;
        fs::write(path, payload)
            .map_err(|e| format!("failed to write model '{}': {e}", path.display()))?;
        Ok(())
    }

    pub fn require_data_schema(&self) -> Result<&DataSchema, String> {
        self.data_schema
            .as_ref()
            .ok_or_else(|| "model is missing data_schema; refit with current CLI".to_string())
    }

    pub fn validate_for_persistence(&self) -> Result<(), String> {
        if self.fit_result.is_none() {
            return Err(
                "model is missing canonical fit_result payload; refit with current CLI".to_string(),
            );
        }
        if self.data_schema.is_none() {
            return Err("model is missing data_schema; refit with current CLI".to_string());
        }
        if self.training_headers.is_none() {
            return Err(
                "model is missing training_headers; refit with current CLI to guarantee stable feature mapping at prediction time"
                    .to_string(),
            );
        }
        let spec = self.resolved_termspec.as_ref().ok_or_else(|| {
            "model is missing resolved_termspec; refit with the current CLI to guarantee train/predict design consistency"
                .to_string()
        })?;
        validate_frozen_term_collectionspec(spec, "resolved_termspec")?;

        if self.formula_noise.is_some() && self.resolved_termspec_noise.is_none() {
            return Err(
                "model defines formula_noise but is missing resolved_termspec_noise; refit with the current CLI"
                    .to_string(),
            );
        }
        if let Some(spec_noise) = self.resolved_termspec_noise.as_ref() {
            validate_frozen_term_collectionspec(spec_noise, "resolved_termspec_noise")?;
        }
        if matches!(self.family_state, FittedFamily::MarginalSlope { .. }) {
            if self.formula_logslope.is_none() {
                return Err(
                    "marginal-slope model is missing formula_logslope; refit with current CLI"
                        .to_string(),
                );
            }
            if self.z_column.is_none() {
                return Err(
                    "marginal-slope model is missing z_column; refit with current CLI".to_string(),
                );
            }
            if self.marginal_baseline.is_none() || self.logslope_baseline.is_none() {
                return Err(
                    "marginal-slope model is missing baseline offsets; refit with current CLI"
                        .to_string(),
                );
            }
            if self.resolved_termspec_noise.is_none() {
                return Err(
                    "marginal-slope model is missing resolved_termspec_noise for the logslope surface"
                        .to_string(),
                );
            }
        }

        if matches!(
            self.family_state,
            FittedFamily::Standard {
                likelihood: LikelihoodFamily::BinomialSas,
                ..
            }
        ) || matches!(
            self.family_state,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialSas,
                ..
            }
        ) {
            self.saved_sas_state()?;
        }
        if matches!(
            self.family_state,
            FittedFamily::Standard {
                likelihood: LikelihoodFamily::BinomialBetaLogistic,
                ..
            }
        ) || matches!(
            self.family_state,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialBetaLogistic,
                ..
            }
        ) {
            self.saved_beta_logistic_state()?;
        }
        if matches!(
            self.family_state,
            FittedFamily::Standard {
                likelihood: LikelihoodFamily::BinomialMixture,
                ..
            }
        ) || matches!(
            self.family_state,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialMixture,
                ..
            }
        ) {
            self.saved_mixture_state()?;
        }
        if self.is_survival_model()
            && self
                .survival_likelihood
                .as_deref()
                .unwrap_or("transformation")
                .eq_ignore_ascii_case("location-scale")
            && (self.survival_beta_time.is_none()
                || self.survival_beta_threshold.is_none()
                || self.survival_beta_log_sigma.is_none())
        {
            return Err(
                "saved location-scale survival model is missing block coefficients; refit with current CLI"
                    .to_string(),
            );
        }

        let has_any_saved_link_wiggle = self.linkwiggle_knots.is_some()
            || self.linkwiggle_degree.is_some()
            || self.beta_link_wiggle.is_some()
            || self
                .fit_result
                .as_ref()
                .and_then(|fit| fit.block_by_role(BlockRole::LinkWiggle))
                .is_some();
        if has_any_saved_link_wiggle && self.saved_link_wiggle()?.is_none() {
            return Err(
                "saved model has incomplete link-wiggle state; expected metadata and coefficients"
                    .to_string(),
            );
        }
        let has_any_saved_baseline_time_wiggle = self.baseline_timewiggle_knots.is_some()
            || self.baseline_timewiggle_degree.is_some()
            || self.baseline_timewiggle_penalty_orders.is_some()
            || self.baseline_timewiggle_double_penalty.is_some()
            || self.beta_baseline_timewiggle.is_some();
        if has_any_saved_baseline_time_wiggle && self.saved_baseline_time_wiggle()?.is_none() {
            return Err(
                "saved model has incomplete baseline-timewiggle state; expected metadata and coefficients"
                    .to_string(),
            );
        }

        // Structural invariant: nonlinear models MUST have beta covariance
        // stored so that posterior-mean prediction (the default) works.
        // Gaussian identity is the only linear family exempt from this.
        let needs_covariance = !matches!(
            self.family_state.likelihood(),
            LikelihoodFamily::GaussianIdentity
        );
        if needs_covariance {
            if let Some(fit) = self.fit_result.as_ref() {
                if fit.beta_covariance().is_none() {
                    return Err("nonlinear model is missing beta_covariance in fit_result; \
                         posterior-mean prediction requires covariance at save time"
                        .to_string());
                }
            }
        }

        Ok(())
    }

    pub fn validate_numeric_finiteness(&self) -> Result<(), String> {
        if let Some(fit) = self.fit_result.as_ref() {
            fit.validate_numeric_finiteness()
                .map_err(|e| e.to_string())?;
        }

        for (name, opt) in [
            ("survival_baseline_scale", self.survival_baseline_scale),
            ("survival_baseline_shape", self.survival_baseline_shape),
            ("survival_baseline_rate", self.survival_baseline_rate),
            ("survival_baseline_makeham", self.survival_baseline_makeham),
            (
                "survival_time_smooth_lambda",
                self.survival_time_smooth_lambda,
            ),
            ("survival_time_anchor", self.survival_time_anchor),
            ("survivalridge_lambda", self.survivalridge_lambda),
        ] {
            if let Some(v) = opt {
                ensure_finite_scalar(name, v)?;
            }
        }

        if let Some(v) = self.beta_noise.as_ref() {
            validate_all_finite("beta_noise", v.iter().copied())?;
        }
        if let Some(v) = self.noise_projection.as_ref() {
            validate_all_finite("noise_projection", v.iter().flatten().copied())?;
        }
        if let Some(v) = self.noise_center.as_ref() {
            validate_all_finite("noise_center", v.iter().copied())?;
        }
        if let Some(v) = self.noise_scale.as_ref() {
            validate_all_finite("noise_scale", v.iter().copied())?;
        }
        if let Some(v) = self.gaussian_response_scale {
            ensure_finite_scalar("gaussian_response_scale", v)?;
        }
        if let Some(v) = self.beta_link_wiggle.as_ref() {
            validate_all_finite("beta_link_wiggle", v.iter().copied())?;
        }
        if let Some(v) = self.beta_baseline_timewiggle.as_ref() {
            validate_all_finite("beta_baseline_timewiggle", v.iter().copied())?;
        }
        if let Some(v) = self.survival_beta_time.as_ref() {
            validate_all_finite("survival_beta_time", v.iter().copied())?;
        }
        if let Some(v) = self.survival_beta_threshold.as_ref() {
            validate_all_finite("survival_beta_threshold", v.iter().copied())?;
        }
        if let Some(v) = self.survival_beta_log_sigma.as_ref() {
            validate_all_finite("survival_beta_log_sigma", v.iter().copied())?;
        }
        if let Some(v) = self.survival_noise_projection.as_ref() {
            validate_all_finite("survival_noise_projection", v.iter().flatten().copied())?;
        }
        if let Some(v) = self.survival_noise_center.as_ref() {
            validate_all_finite("survival_noise_center", v.iter().copied())?;
        }
        if let Some(v) = self.survival_noise_scale.as_ref() {
            validate_all_finite("survival_noise_scale", v.iter().copied())?;
        }
        if let Some(v) = self.mixture_link_param_covariance.as_ref() {
            validate_all_finite("mixture_link_param_covariance", v.iter().flatten().copied())?;
        }
        if let Some(v) = self.sas_param_covariance.as_ref() {
            validate_all_finite("sas_param_covariance", v.iter().flatten().copied())?;
        }
        Ok(())
    }
}

fn array2_to_nestedvec(a: &ndarray::Array2<f64>) -> Vec<Vec<f64>> {
    a.rows().into_iter().map(|row| row.to_vec()).collect()
}

use crate::solver::estimate::{ensure_finite_scalar, validate_all_finite};

fn validate_frozen_term_collectionspec(
    spec: &TermCollectionSpec,
    label: &str,
) -> Result<(), String> {
    spec.validate_frozen(label)
}

impl Deref for FittedModel {
    type Target = FittedModelPayload;

    fn deref(&self) -> &Self::Target {
        self.payload()
    }
}

impl DerefMut for FittedModel {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.payload_mut()
    }
}

// ---------------------------------------------------------------------------
// Reconstruct library types from saved models
// ---------------------------------------------------------------------------

pub fn survival_baseline_config_from_model(
    model: &FittedModel,
) -> Result<SurvivalBaselineConfig, String> {
    parse_survival_baseline_config(
        model
            .survival_baseline_target
            .as_deref()
            .unwrap_or("linear"),
        model.survival_baseline_scale,
        model.survival_baseline_shape,
        model.survival_baseline_rate,
        model.survival_baseline_makeham,
    )
}

pub fn load_survival_time_basis_config_from_model(
    model: &FittedModel,
) -> Result<SurvivalTimeBasisConfig, String> {
    match model
        .survival_time_basis
        .as_deref()
        .ok_or_else(|| "saved survival model missing survival_time_basis".to_string())?
        .to_ascii_lowercase()
        .as_str()
    {
        "none" => Ok(SurvivalTimeBasisConfig::None),
        "linear" => Ok(SurvivalTimeBasisConfig::Linear),
        "bspline" => {
            let degree = model.survival_time_degree.ok_or_else(|| {
                "saved survival bspline model missing survival_time_degree".to_string()
            })?;
            let knots = model.survival_time_knots.clone().ok_or_else(|| {
                "saved survival bspline model missing survival_time_knots".to_string()
            })?;
            let smooth_lambda = model.survival_time_smooth_lambda.unwrap_or(1e-2);
            if degree < 1 || knots.is_empty() {
                return Err("saved survival bspline time basis metadata is invalid".to_string());
            }
            Ok(SurvivalTimeBasisConfig::BSpline {
                degree,
                knots: Array1::from_vec(knots),
                smooth_lambda,
            })
        }
        "ispline" => {
            let degree = model.survival_time_degree.ok_or_else(|| {
                "saved survival ispline model missing survival_time_degree".to_string()
            })?;
            let knots = model.survival_time_knots.clone().ok_or_else(|| {
                "saved survival ispline model missing survival_time_knots".to_string()
            })?;
            let keep_cols = model.survival_time_keep_cols.clone().ok_or_else(|| {
                "saved survival ispline model missing survival_time_keep_cols".to_string()
            })?;
            let smooth_lambda = model.survival_time_smooth_lambda.unwrap_or(1e-2);
            if degree < 1 || knots.is_empty() || keep_cols.is_empty() {
                return Err("saved survival ispline time basis metadata is invalid".to_string());
            }
            Ok(SurvivalTimeBasisConfig::ISpline {
                degree,
                knots: Array1::from_vec(knots),
                keep_cols,
                smooth_lambda,
            })
        }
        other => Err(format!("unsupported saved survival_time_basis '{other}'")),
    }
}

use crate::basis::BasisOptions;
use crate::estimate::{BlockRole, FittedLinkState, UnifiedFitResult};
use crate::families::gamlss::{
    monotone_wiggle_basis_with_derivative_order, validate_monotone_wiggle_beta_nonnegative,
};
use crate::families::lognormal_kernel::FrailtySpec;
use crate::families::survival_construction::{
    SurvivalBaselineConfig, SurvivalTimeBasisConfig, parse_survival_baseline_config,
};
use crate::inference::predict::{
    BernoulliMarginalSlopePredictor, BinomialLocationScalePredictor,
    GaussianLocationScalePredictor, PredictableModel, StandardPredictor, SurvivalPredictor,
};
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec};
use crate::smooth::{AdaptiveRegularizationDiagnostics, TermCollectionSpec};
use crate::span::span_index_for_breakpoints;
use crate::types::{
    InverseLink, LatentCLogLogState, LikelihoodFamily, LinkFunction, MixtureLinkState, SasLinkSpec,
    SasLinkState,
};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fs;
use std::ops::{Deref, DerefMut};
use std::path::Path;

/// Canonical saved-model payload schema version.
///
/// Every `FittedModelPayload` written by any binary (CLI `gam`, gam-pyffi,
/// downstream library users) must set this as its `version` field, and every
/// load path asserts equality via `validate_for_persistence`. Bump this when:
///   - A required field is added to `FittedModelPayload` and the set of
///     Option<T> fields that must be `Some(...)` for a given `family_state`
///     changes (otherwise the `#[serde(default)]` decode would silently fill
///     the new field with `None` when loading an older model and the CLI
///     predict path would run with stale metadata).
///   - The on-wire shape of any `serde`-tagged enum variant changes such that
///     older payloads no longer round-trip losslessly.
///   - The semantics of an existing field change (e.g. sign convention,
///     coordinate frame) in a way that predict output would silently diverge
///     between old and new readers.
///
/// Do NOT bump for purely additive `Option<T>` fields that the save-time
/// invariant (`validate_for_persistence`) does not yet require. Those are
/// forward-compatible.
pub const MODEL_PAYLOAD_VERSION: u32 = 4;

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

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedLatentZNormalization {
    pub mean: f64,
    pub sd: f64,
}

impl SavedLatentZNormalization {
    pub fn validate(&self, context: &str) -> Result<(), String> {
        if !self.mean.is_finite() {
            return Err(format!("{context} latent z mean must be finite"));
        }
        if !(self.sd.is_finite() && self.sd > 1e-12) {
            return Err(format!(
                "{context} latent z sd must be finite and > 1e-12; got {}",
                self.sd
            ));
        }
        Ok(())
    }

    pub fn apply(&self, z: &Array1<f64>, context: &str) -> Result<Array1<f64>, String> {
        self.validate(context)?;
        if z.iter().any(|value| !value.is_finite()) {
            return Err(format!("{context} requires finite z values"));
        }
        Ok(z.mapv(|zi| (zi - self.mean) / self.sd))
    }
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
    pub offset_column: Option<String>,
    #[serde(default)]
    pub noise_offset_column: Option<String>,
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
    pub latent_z_normalization: Option<SavedLatentZNormalization>,
    #[serde(default)]
    pub latent_score_contract: Option<SavedLatentScoreContract>,
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
    pub resolved_termspec_logslope: Option<TermCollectionSpec>,
    #[serde(default)]
    pub adaptive_regularization_diagnostics: Option<AdaptiveRegularizationDiagnostics>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedLatentScoreContract {
    pub semantics: String,
    pub source_transform_id: Option<String>,
    pub normalization_mean: f64,
    pub normalization_sd: f64,
    pub clip_eps: Option<f64>,
    pub conditioning_columns: Vec<String>,
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
            offset_column: None,
            noise_offset_column: None,
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
            latent_z_normalization: None,
            latent_score_contract: None,
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
            resolved_termspec_logslope: None,
            adaptive_regularization_diagnostics: None,
        }
    }

    fn validate_payload_version(&self) -> Result<(), String> {
        if self.version != MODEL_PAYLOAD_VERSION {
            return Err(format!(
                "saved model payload schema mismatch: file has version={}, \
                 this binary expects MODEL_PAYLOAD_VERSION={}. \
                 Refit with the current CLI, or rebuild the reader at the same \
                 version the model was written with.",
                self.version, MODEL_PAYLOAD_VERSION
            ));
        }
        Ok(())
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
        latent_cloglog_state: Option<LatentCLogLogState>,
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
        base_link: Option<InverseLink>,
        frailty: FrailtySpec,
    },
    Survival {
        likelihood: LikelihoodFamily,
        #[serde(default)]
        survival_likelihood: Option<String>,
        #[serde(default)]
        survival_distribution: Option<String>,
        frailty: FrailtySpec,
    },
    LatentSurvival {
        frailty: FrailtySpec,
    },
    LatentBinary {
        frailty: FrailtySpec,
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
    pub breakpoints: Vec<f64>,
    pub basis_dim: usize,
    pub span_c0: Vec<Vec<f64>>,
    pub span_c1: Vec<Vec<f64>>,
    pub span_c2: Vec<Vec<f64>>,
    pub span_c3: Vec<Vec<f64>>,
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

fn gaussian_location_scale_mean_beta(fit: &UnifiedFitResult) -> Option<Array1<f64>> {
    fit.block_by_role(BlockRole::Location)
        .or_else(|| fit.block_by_role(BlockRole::Mean))
        .map(|block| block.beta.clone())
}

fn binomial_location_scale_threshold_beta(fit: &UnifiedFitResult) -> Option<Array1<f64>> {
    fit.block_by_role(BlockRole::Threshold)
        .or_else(|| fit.block_by_role(BlockRole::Location))
        .or_else(|| fit.block_by_role(BlockRole::Mean))
        .map(|block| block.beta.clone())
}

fn location_scale_noise_beta(fit: &UnifiedFitResult) -> Option<Array1<f64>> {
    fit.block_by_role(BlockRole::Scale)
        .map(|block| block.beta.clone())
}

fn validate_location_scale_saved_fit(
    fit: &UnifiedFitResult,
    model_class: PredictModelClass,
    link_wiggle: Option<&SavedLinkWiggleRuntime>,
) -> Result<(), String> {
    let primary = match model_class {
        PredictModelClass::GaussianLocationScale => gaussian_location_scale_mean_beta(fit),
        PredictModelClass::BinomialLocationScale => binomial_location_scale_threshold_beta(fit),
        _ => None,
    }
    .ok_or_else(|| match model_class {
        PredictModelClass::GaussianLocationScale => {
            "gaussian-location-scale saved fit is missing mean/location block".to_string()
        }
        PredictModelClass::BinomialLocationScale => {
            "binomial-location-scale saved fit is missing threshold/location block".to_string()
        }
        _ => "location-scale saved fit is missing primary block".to_string(),
    })?;

    let scale = location_scale_noise_beta(fit)
        .ok_or_else(|| "location-scale saved fit is missing scale block".to_string())?;
    let expected =
        primary.len() + scale.len() + link_wiggle.map_or(0, |runtime| runtime.beta.len());

    if let Some(cov) = fit.beta_covariance()
        && (cov.nrows() != expected || cov.ncols() != expected)
    {
        return Err(format!(
            "location-scale saved conditional covariance shape mismatch: got {}x{}, expected {}x{}",
            cov.nrows(),
            cov.ncols(),
            expected,
            expected
        ));
    }
    if let Some(cov) = fit.beta_covariance_corrected()
        && (cov.nrows() != expected || cov.ncols() != expected)
    {
        return Err(format!(
            "location-scale saved corrected covariance shape mismatch: got {}x{}, expected {}x{}",
            cov.nrows(),
            cov.ncols(),
            expected,
            expected
        ));
    }
    Ok(())
}

fn validate_survival_saved_block_matches_payload(
    fit: &UnifiedFitResult,
    role: BlockRole,
    payload_beta: Option<&Vec<f64>>,
    label: &str,
) -> Result<usize, String> {
    let block = fit
        .block_by_role(role)
        .ok_or_else(|| format!("location-scale survival saved fit is missing {label} block"))?;
    if let Some(saved) = payload_beta
        && block.beta.to_vec() != *saved
    {
        return Err(format!(
            "location-scale survival saved {label} coefficients disagree with fit_result"
        ));
    }
    Ok(block.beta.len())
}

fn validate_survival_location_scale_saved_fit(
    payload: &FittedModelPayload,
    link_wiggle: Option<&SavedLinkWiggleRuntime>,
) -> Result<(), String> {
    let fit = payload.fit_result.as_ref().ok_or_else(|| {
        "location-scale survival model is missing canonical fit_result payload".to_string()
    })?;
    let p_time = validate_survival_saved_block_matches_payload(
        fit,
        BlockRole::Time,
        payload.survival_beta_time.as_ref(),
        "time",
    )?;
    let p_threshold = validate_survival_saved_block_matches_payload(
        fit,
        BlockRole::Threshold,
        payload.survival_beta_threshold.as_ref(),
        "threshold",
    )?;
    let p_log_sigma = validate_survival_saved_block_matches_payload(
        fit,
        BlockRole::Scale,
        payload.survival_beta_log_sigma.as_ref(),
        "log-sigma",
    )?;
    let p_wiggle = match link_wiggle {
        Some(runtime) => {
            let block = fit.block_by_role(BlockRole::LinkWiggle).ok_or_else(|| {
                "location-scale survival saved fit is missing link-wiggle block".to_string()
            })?;
            if block.beta.to_vec() != runtime.beta {
                return Err(
                    "location-scale survival saved link-wiggle coefficients disagree with fit_result"
                        .to_string(),
                );
            }
            runtime.beta.len()
        }
        None => {
            if fit.block_by_role(BlockRole::LinkWiggle).is_some() {
                return Err(
                    "location-scale survival saved fit has a LinkWiggle block without payload metadata"
                        .to_string(),
                );
            }
            0
        }
    };
    let expected = p_time + p_threshold + p_log_sigma + p_wiggle;

    if let Some(cov) = fit.beta_covariance()
        && (cov.nrows() != expected || cov.ncols() != expected)
    {
        return Err(format!(
            "location-scale survival saved conditional covariance shape mismatch: got {}x{}, expected {}x{}",
            cov.nrows(),
            cov.ncols(),
            expected,
            expected
        ));
    }
    if let Some(cov) = fit.beta_covariance_corrected()
        && (cov.nrows() != expected || cov.ncols() != expected)
    {
        return Err(format!(
            "location-scale survival saved corrected covariance shape mismatch: got {}x{}, expected {}x{}",
            cov.nrows(),
            cov.ncols(),
            expected,
            expected
        ));
    }
    Ok(())
}

fn validate_marginal_slope_saved_fit(
    fit: &UnifiedFitResult,
    score_warp: Option<&SavedAnchoredDeviationRuntime>,
    link_deviation: Option<&SavedAnchoredDeviationRuntime>,
    fit_label: &str,
) -> Result<(), String> {
    let expected_blocks =
        2 + usize::from(score_warp.is_some()) + usize::from(link_deviation.is_some());
    if fit.blocks.len() != expected_blocks {
        return Err(format!(
            "bernoulli marginal-slope saved {fit_label} requires exactly {expected_blocks} coefficient blocks [marginal, logslope{}{}], got {}",
            if score_warp.is_some() {
                ", score-warp"
            } else {
                ""
            },
            if link_deviation.is_some() {
                ", link-deviation"
            } else {
                ""
            },
            fit.blocks.len(),
        ));
    }
    if let Some(runtime) = score_warp {
        let beta = &fit.blocks[2].beta;
        if beta.len() != runtime.basis_dim {
            return Err(format!(
                "bernoulli marginal-slope saved {fit_label} score-warp coefficient mismatch: beta has {} entries but runtime expects {}",
                beta.len(),
                runtime.basis_dim
            ));
        }
    }
    if let Some(runtime) = link_deviation {
        let idx = 2 + usize::from(score_warp.is_some());
        let beta = &fit.blocks[idx].beta;
        if beta.len() != runtime.basis_dim {
            return Err(format!(
                "bernoulli marginal-slope saved {fit_label} link-deviation coefficient mismatch: beta has {} entries but runtime expects {}",
                beta.len(),
                runtime.basis_dim
            ));
        }
    }
    Ok(())
}

fn validate_survival_marginal_slope_saved_fit(
    fit: &UnifiedFitResult,
    score_warp: Option<&SavedAnchoredDeviationRuntime>,
    link_deviation: Option<&SavedAnchoredDeviationRuntime>,
    fit_label: &str,
) -> Result<(), String> {
    let expected_blocks =
        3 + usize::from(score_warp.is_some()) + usize::from(link_deviation.is_some());
    if fit.blocks.len() != expected_blocks {
        return Err(format!(
            "survival marginal-slope saved {fit_label} requires {expected_blocks} blocks [time, marginal, slope{}{}], got {}",
            if score_warp.is_some() {
                ", score-warp"
            } else {
                ""
            },
            if link_deviation.is_some() {
                ", link-deviation"
            } else {
                ""
            },
            fit.blocks.len(),
        ));
    }
    if let Some(runtime) = score_warp {
        let beta = &fit.blocks[3].beta;
        if beta.len() != runtime.basis_dim {
            return Err(format!(
                "survival marginal-slope saved {fit_label} score-warp coefficient mismatch: beta has {} entries but runtime expects {}",
                beta.len(),
                runtime.basis_dim
            ));
        }
    }
    if let Some(runtime) = link_deviation {
        let idx = 3 + usize::from(score_warp.is_some());
        let beta = &fit.blocks[idx].beta;
        if beta.len() != runtime.basis_dim {
            return Err(format!(
                "survival marginal-slope saved {fit_label} link-deviation coefficient mismatch: beta has {} entries but runtime expects {}",
                beta.len(),
                runtime.basis_dim
            ));
        }
    }
    Ok(())
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
    pub(crate) fn validate_exact_replay_contract(&self) -> Result<(), String> {
        if self.kernel.is_empty() {
            return Err(
                "saved anchored deviation runtime is missing the exact kernel marker".to_string(),
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
        if self.basis_dim == 0 {
            return Err(format!(
                "saved anchored deviation runtime basis_dim must be positive, got {}",
                self.basis_dim
            ));
        }
        if self.breakpoints.len() < 2 {
            return Err(format!(
                "saved anchored deviation runtime requires at least two breakpoints, got {}",
                self.breakpoints.len()
            ));
        }
        for window in self.breakpoints.windows(2) {
            let left = window[0];
            let right = window[1];
            if !left.is_finite() || !right.is_finite() || right <= left {
                return Err(format!(
                    "saved anchored deviation runtime breakpoints must be finite and strictly increasing, got [{left}, {right}]"
                ));
            }
        }
        let span_count = self.breakpoints.len() - 1;
        self.validate_coefficient_matrix(&self.span_c0, "c0", span_count)?;
        self.validate_coefficient_matrix(&self.span_c1, "c1", span_count)?;
        self.validate_coefficient_matrix(&self.span_c2, "c2", span_count)?;
        self.validate_coefficient_matrix(&self.span_c3, "c3", span_count)?;
        self.validate_c2_span_continuity()?;
        Ok(())
    }

    fn validate_c2_span_continuity(&self) -> Result<(), String> {
        const TOL: f64 = 1e-8;
        for span_idx in 1..self.breakpoints.len() - 1 {
            let left_span = span_idx - 1;
            let right_span = span_idx;
            let width = self.breakpoints[span_idx] - self.breakpoints[left_span];
            for basis_idx in 0..self.basis_dim {
                let left_value = self.span_c0[left_span][basis_idx]
                    + self.span_c1[left_span][basis_idx] * width
                    + self.span_c2[left_span][basis_idx] * width * width
                    + self.span_c3[left_span][basis_idx] * width * width * width;
                let left_d1 = self.span_c1[left_span][basis_idx]
                    + 2.0 * self.span_c2[left_span][basis_idx] * width
                    + 3.0 * self.span_c3[left_span][basis_idx] * width * width;
                let left_d2 = 2.0 * self.span_c2[left_span][basis_idx]
                    + 6.0 * self.span_c3[left_span][basis_idx] * width;
                let right_value = self.span_c0[right_span][basis_idx];
                let right_d1 = self.span_c1[right_span][basis_idx];
                let right_d2 = 2.0 * self.span_c2[right_span][basis_idx];
                if (left_value - right_value).abs() > TOL
                    || (left_d1 - right_d1).abs() > TOL
                    || (left_d2 - right_d2).abs() > TOL
                {
                    return Err(format!(
                        "saved anchored deviation runtime must be C2 cubic at breakpoint {span_idx}, basis {basis_idx}: value jump={:.3e}, d1 jump={:.3e}, d2 jump={:.3e}",
                        left_value - right_value,
                        left_d1 - right_d1,
                        left_d2 - right_d2
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_coefficient_matrix(
        &self,
        matrix: &[Vec<f64>],
        label: &str,
        expected_rows: usize,
    ) -> Result<(), String> {
        if matrix.len() != expected_rows {
            return Err(format!(
                "saved anchored deviation runtime {label} row count mismatch: got {}, expected {}",
                matrix.len(),
                expected_rows
            ));
        }
        for (row_idx, row) in matrix.iter().enumerate() {
            if row.len() != self.basis_dim {
                return Err(format!(
                    "saved anchored deviation runtime {label} row {} has width {}, expected {}",
                    row_idx,
                    row.len(),
                    self.basis_dim
                ));
            }
            for (j, &value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(format!(
                        "saved anchored deviation runtime {label} entry ({row_idx},{j}) is non-finite"
                    ));
                }
            }
        }
        Ok(())
    }

    fn right_boundary_basis_value(&self, basis_idx: usize) -> f64 {
        let last_span = self.breakpoints.len() - 2;
        let width = self.breakpoints[last_span + 1] - self.breakpoints[last_span];
        self.span_c0[last_span][basis_idx]
            + self.span_c1[last_span][basis_idx] * width
            + self.span_c2[last_span][basis_idx] * width * width
            + self.span_c3[last_span][basis_idx] * width * width * width
    }

    fn evaluate_span_polynomial_design(
        &self,
        values: &Array1<f64>,
        derivative_order: usize,
    ) -> Result<Array2<f64>, String> {
        self.validate_exact_replay_contract()?;
        let (left_ep, right_ep) = self.support_interval()?;
        let mut out = Array2::<f64>::zeros((values.len(), self.basis_dim));
        for (row_idx, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "saved anchored deviation runtime design value at row {row_idx} is non-finite ({value})"
                ));
            }
            if value < left_ep {
                if derivative_order == 0 {
                    for basis_idx in 0..self.basis_dim {
                        out[[row_idx, basis_idx]] = self.span_c0[0][basis_idx];
                    }
                }
                continue;
            }
            if value > right_ep {
                if derivative_order == 0 {
                    for basis_idx in 0..self.basis_dim {
                        out[[row_idx, basis_idx]] = self.right_boundary_basis_value(basis_idx);
                    }
                }
                continue;
            }
            let span_idx = self.left_biased_span_index_for(value)?;
            let t = value - self.breakpoints[span_idx];
            for basis_idx in 0..self.basis_dim {
                let c0 = self.span_c0[span_idx][basis_idx];
                let c1 = self.span_c1[span_idx][basis_idx];
                let c2 = self.span_c2[span_idx][basis_idx];
                let c3 = self.span_c3[span_idx][basis_idx];
                out[[row_idx, basis_idx]] = match derivative_order {
                    0 => c0 + c1 * t + c2 * t * t + c3 * t * t * t,
                    1 => c1 + 2.0 * c2 * t + 3.0 * c3 * t * t,
                    2 => 2.0 * c2 + 6.0 * c3 * t,
                    3 => 6.0 * c3,
                    4 => 0.0,
                    other => {
                        return Err(format!(
                            "saved anchored deviation runtime only supports derivative orders up to 4, got {other}"
                        ));
                    }
                };
            }
        }
        Ok(out)
    }

    pub fn breakpoints(&self) -> Result<Vec<f64>, String> {
        self.validate_exact_replay_contract()?;
        Ok(self.breakpoints.clone())
    }

    pub fn span_count(&self) -> Result<usize, String> {
        Ok(self.breakpoints()?.windows(2).count())
    }

    pub fn span_index_for(&self, value: f64) -> Result<usize, String> {
        let points = self.breakpoints()?;
        span_index_for_breakpoints(&points, value, "saved anchored deviation span lookup")
    }

    fn left_biased_span_index_for(&self, value: f64) -> Result<usize, String> {
        let mut span_idx = span_index_for_breakpoints(
            &self.breakpoints,
            value,
            "saved anchored deviation span lookup",
        )?;
        // LEFT-bias at interior breakpoints mirrors DeviationRuntime. The
        // saved cubic basis is C2, but d3 remains span-local.
        if span_idx > 0 && value == self.breakpoints[span_idx] {
            span_idx -= 1;
        }
        Ok(span_idx)
    }

    pub fn local_cubic_on_span(
        &self,
        beta: &Array1<f64>,
        span_idx: usize,
    ) -> Result<crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic, String>
    {
        self.validate_exact_replay_contract()?;
        if beta.len() != self.basis_dim {
            return Err(format!(
                "saved anchored deviation coefficient length mismatch: got {}, expected {}",
                beta.len(),
                self.basis_dim
            ));
        }
        self.local_cubic_on_span_validated(beta, span_idx)
    }

    fn local_cubic_on_span_validated(
        &self,
        beta: &Array1<f64>,
        span_idx: usize,
    ) -> Result<crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic, String>
    {
        let points = &self.breakpoints;
        if span_idx + 1 >= points.len() {
            return Err(format!(
                "saved anchored deviation span index {} out of range for {} spans",
                span_idx,
                points.len() - 1
            ));
        }
        let left = points[span_idx];
        let right = points[span_idx + 1];
        Ok(
            crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic {
                left,
                right,
                c0: self.span_c0[span_idx]
                    .iter()
                    .zip(beta.iter())
                    .map(|(coeff, weight)| coeff * weight)
                    .sum(),
                c1: self.span_c1[span_idx]
                    .iter()
                    .zip(beta.iter())
                    .map(|(coeff, weight)| coeff * weight)
                    .sum(),
                c2: self.span_c2[span_idx]
                    .iter()
                    .zip(beta.iter())
                    .map(|(coeff, weight)| coeff * weight)
                    .sum(),
                c3: self.span_c3[span_idx]
                    .iter()
                    .zip(beta.iter())
                    .map(|(coeff, weight)| coeff * weight)
                    .sum(),
            },
        )
    }

    pub fn basis_span_cubic(
        &self,
        span_idx: usize,
        basis_idx: usize,
    ) -> Result<crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic, String>
    {
        self.validate_exact_replay_contract()?;
        if basis_idx >= self.basis_dim {
            return Err(format!(
                "saved anchored deviation basis index {} out of range for {} coefficients",
                basis_idx, self.basis_dim
            ));
        }
        self.basis_span_cubic_validated(span_idx, basis_idx)
    }

    fn basis_span_cubic_validated(
        &self,
        span_idx: usize,
        basis_idx: usize,
    ) -> Result<crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic, String>
    {
        let points = &self.breakpoints;
        if span_idx + 1 >= points.len() {
            return Err(format!(
                "saved anchored deviation span index {} out of range for {} spans",
                span_idx,
                points.len() - 1
            ));
        }
        Ok(
            crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic {
                left: points[span_idx],
                right: points[span_idx + 1],
                c0: self.span_c0[span_idx][basis_idx],
                c1: self.span_c1[span_idx][basis_idx],
                c2: self.span_c2[span_idx][basis_idx],
                c3: self.span_c3[span_idx][basis_idx],
            },
        )
    }

    pub fn basis_cubic_at(
        &self,
        basis_idx: usize,
        value: f64,
    ) -> Result<crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic, String>
    {
        self.validate_exact_replay_contract()?;
        if basis_idx >= self.basis_dim {
            return Err(format!(
                "saved anchored deviation basis index {} out of range for {} coefficients",
                basis_idx, self.basis_dim
            ));
        }
        let (left_ep, right_ep) = self.support_interval()?;
        if value < left_ep {
            return Ok(
                crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic {
                    left: left_ep,
                    right: left_ep + 1.0,
                    c0: self.span_c0[0][basis_idx],
                    c1: 0.0,
                    c2: 0.0,
                    c3: 0.0,
                },
            );
        }
        if value > right_ep {
            return Ok(
                crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic {
                    left: right_ep,
                    right: right_ep + 1.0,
                    c0: self.right_boundary_basis_value(basis_idx),
                    c1: 0.0,
                    c2: 0.0,
                    c3: 0.0,
                },
            );
        }
        let span_idx = self.left_biased_span_index_for(value)?;
        self.basis_span_cubic_validated(span_idx, basis_idx)
    }

    pub fn local_cubic_at(
        &self,
        beta: &Array1<f64>,
        value: f64,
    ) -> Result<crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic, String>
    {
        self.validate_exact_replay_contract()?;
        if beta.len() != self.basis_dim {
            return Err(format!(
                "saved anchored deviation coefficient length mismatch: got {}, expected {}",
                beta.len(),
                self.basis_dim
            ));
        }
        let (left_ep, right_ep) = self.support_interval()?;
        if value < left_ep {
            return Ok(
                crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic {
                    left: left_ep,
                    right: left_ep + 1.0,
                    c0: self.span_c0[0]
                        .iter()
                        .zip(beta.iter())
                        .map(|(coeff, weight)| coeff * weight)
                        .sum(),
                    c1: 0.0,
                    c2: 0.0,
                    c3: 0.0,
                },
            );
        }
        if value > right_ep {
            return Ok(
                crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic {
                    left: right_ep,
                    right: right_ep + 1.0,
                    c0: (0..self.basis_dim)
                        .map(|basis_idx| {
                            self.right_boundary_basis_value(basis_idx) * beta[basis_idx]
                        })
                        .sum(),
                    c1: 0.0,
                    c2: 0.0,
                    c3: 0.0,
                },
            );
        }
        let span_idx = self.left_biased_span_index_for(value)?;
        self.local_cubic_on_span_validated(beta, span_idx)
    }

    fn support_interval(&self) -> Result<(f64, f64), String> {
        let points = self.breakpoints()?;
        match (points.first(), points.last()) {
            (Some(&left), Some(&right)) => Ok((left, right)),
            _ => Err("saved anchored deviation runtime is missing support breakpoints".to_string()),
        }
    }

    pub fn design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.evaluate_span_polynomial_design(values, BasisOptions::value().derivative_order)
    }

    pub fn first_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.evaluate_span_polynomial_design(
            values,
            BasisOptions::first_derivative().derivative_order,
        )
    }

    pub fn second_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.evaluate_span_polynomial_design(
            values,
            BasisOptions::second_derivative().derivative_order,
        )
    }

    pub fn third_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.evaluate_span_polynomial_design(values, 3)
    }

    pub fn fourth_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.evaluate_span_polynomial_design(values, 4)
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
            Self::LatentSurvival { .. } | Self::LatentBinary { .. } => {
                LikelihoodFamily::RoystonParmar
            }
        }
    }

    #[inline]
    pub fn frailty(&self) -> Option<&FrailtySpec> {
        match self {
            Self::MarginalSlope { frailty, .. }
            | Self::Survival { frailty, .. }
            | Self::LatentSurvival { frailty }
            | Self::LatentBinary { frailty } => Some(frailty),
            _ => None,
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
                    likelihood: LikelihoodFamily::BinomialLatentCLogLog,
                    latent_cloglog_state,
                    ..
                },
                FittedLinkState::LatentCLogLog { state },
            ) => {
                *latent_cloglog_state = Some(*state);
            }
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
            FittedFamily::Survival { .. }
            | FittedFamily::LatentSurvival { .. }
            | FittedFamily::LatentBinary { .. } => PredictModelClass::Survival,
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
        self.payload().validate_payload_version()?;
        if matches!(
            self.predict_model_class(),
            PredictModelClass::BernoulliMarginalSlope | PredictModelClass::Survival
        ) {
            if let Some(runtime) = self.payload().score_warp_runtime.as_ref() {
                runtime.validate_exact_replay_contract().map_err(|err| {
                    format!("saved anchored score-warp runtime is invalid: {err}")
                })?;
            }
            if let Some(runtime) = self.payload().link_deviation_runtime.as_ref() {
                runtime.validate_exact_replay_contract().map_err(|err| {
                    format!("saved anchored link-deviation runtime is invalid: {err}")
                })?;
            }
        }
        let runtime = SavedPredictionRuntime {
            model_class: self.predict_model_class(),
            likelihood: self.likelihood(),
            inverse_link: self.resolved_inverse_link()?,
            link_wiggle: self.saved_link_wiggle()?,
            baseline_time_wiggle: self.saved_baseline_time_wiggle()?,
            score_warp: self.payload().score_warp_runtime.clone(),
            link_deviation: self.payload().link_deviation_runtime.clone(),
        };
        if matches!(
            runtime.model_class,
            PredictModelClass::GaussianLocationScale | PredictModelClass::BinomialLocationScale
        ) {
            let fit = self.payload().fit_result.as_ref().ok_or_else(|| {
                "location-scale model is missing canonical fit_result payload".to_string()
            })?;
            validate_location_scale_saved_fit(
                fit,
                runtime.model_class,
                runtime.link_wiggle.as_ref(),
            )?;
        } else if matches!(runtime.model_class, PredictModelClass::Survival)
            && self
                .payload()
                .survival_likelihood
                .as_deref()
                .is_some_and(|value| value.eq_ignore_ascii_case("location-scale"))
        {
            validate_survival_location_scale_saved_fit(
                self.payload(),
                runtime.link_wiggle.as_ref(),
            )?;
        } else if matches!(
            runtime.model_class,
            PredictModelClass::BernoulliMarginalSlope
        ) {
            let unified = self.payload().unified.as_ref().ok_or_else(|| {
                "marginal-slope model is missing unified fit payload; refit with current CLI"
                    .to_string()
            })?;
            validate_marginal_slope_saved_fit(
                unified,
                runtime.score_warp.as_ref(),
                runtime.link_deviation.as_ref(),
                "unified",
            )?;
        } else if matches!(runtime.model_class, PredictModelClass::Survival)
            && self
                .payload()
                .survival_likelihood
                .as_deref()
                .is_some_and(|value| value.eq_ignore_ascii_case("marginal-slope"))
        {
            let fit = self.payload().fit_result.as_ref().ok_or_else(|| {
                "survival marginal-slope model is missing canonical fit_result payload".to_string()
            })?;
            validate_survival_marginal_slope_saved_fit(
                fit,
                runtime.score_warp.as_ref(),
                runtime.link_deviation.as_ref(),
                "fit_result",
            )?;
        }
        Ok(runtime)
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

    pub fn saved_latent_cloglog_state(&self) -> Result<Option<LatentCLogLogState>, String> {
        let payload = self.payload();
        match &payload.family_state {
            FittedFamily::Standard {
                likelihood: LikelihoodFamily::BinomialLatentCLogLog,
                latent_cloglog_state,
                ..
            } => latent_cloglog_state
                .ok_or_else(|| {
                    "latent-cloglog-binomial model is missing state in family_state.latent_cloglog_state"
                        .to_string()
                })
                .map(Some),
            _ => Ok(None),
        }
    }

    pub fn resolved_inverse_link(&self) -> Result<Option<InverseLink>, String> {
        let stateful = if let Some(state) = self.saved_mixture_state()? {
            Some(InverseLink::Mixture(state))
        } else if let Some(state) = self.saved_latent_cloglog_state()? {
            Some(InverseLink::LatentCLogLog(state))
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
            FittedFamily::MarginalSlope { base_link, .. } => Ok(base_link.clone()),
            FittedFamily::Survival { .. }
            | FittedFamily::LatentSurvival { .. }
            | FittedFamily::LatentBinary { .. } => Ok(None),
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
                let beta_mu = gaussian_location_scale_mean_beta(fit)?;
                let beta_noise = location_scale_noise_beta(fit)
                    .or_else(|| self.payload().beta_noise.clone().map(Array1::from_vec))?;
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
                if matches!(
                    self.family_state,
                    FittedFamily::Survival {
                        survival_likelihood: Some(ref survival_likelihood),
                        ..
                    } if survival_likelihood == "marginal-slope"
                ) {
                    return None;
                }
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
                let beta_threshold = binomial_location_scale_threshold_beta(fit)?;
                let beta_noise = location_scale_noise_beta(fit)
                    .or_else(|| self.payload().beta_noise.clone().map(Array1::from_vec))?;
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
                    payload.latent_z_normalization?,
                    payload.marginal_baseline?,
                    payload.logslope_baseline?,
                    self.resolved_inverse_link()
                        .ok()?
                        .unwrap_or(InverseLink::Standard(LinkFunction::Probit)),
                    self.family_state.frailty()?.clone(),
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
        // Hard version gate. The struct's ~40 Option<T> fields carry
        // `#[serde(default)]`, which is by design forward-compatible: old
        // payloads missing a new optional field decode with `None`. BUT:
        // when a new CLI release adds a required field for some family_state
        // (enforced below), an older model loaded by the newer CLI would have
        // `None` in that slot and the family-specific branch below would
        // correctly reject it — unless the new field also happens to slot
        // under a branch that hasn't been touched. Conversely, a newer model
        // loaded by an older CLI silently drops fields the older struct
        // doesn't know about. Both directions are silent-drift hazards. We
        // close them with an exact-version check anchored to the canonical
        // MODEL_PAYLOAD_VERSION constant — every payload must round-trip
        // identically between writers and readers running the same schema.
        self.validate_payload_version()?;
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
            let z_normalization = self.latent_z_normalization.ok_or_else(|| {
                "marginal-slope model is missing latent_z_normalization; refit with current CLI"
                    .to_string()
            })?;
            z_normalization.validate("marginal-slope model")?;
            if self.marginal_baseline.is_none() || self.logslope_baseline.is_none() {
                return Err(
                    "marginal-slope model is missing baseline offsets; refit with current CLI"
                        .to_string(),
                );
            }
            if self
                .resolved_termspec_logslope
                .as_ref()
                .or(self.resolved_termspec_noise.as_ref())
                .is_none()
            {
                return Err(
                    "marginal-slope model is missing resolved_termspec_logslope for the logslope surface"
                        .to_string(),
                );
            }
            match self.family_state.frailty() {
                Some(FrailtySpec::None)
                | Some(FrailtySpec::GaussianShift {
                    sigma_fixed: Some(_),
                }) => {}
                Some(FrailtySpec::GaussianShift { sigma_fixed: None }) => {
                    return Err(
                        "marginal-slope model requires a fixed GaussianShift sigma in family_state.frailty"
                            .to_string(),
                    );
                }
                Some(FrailtySpec::HazardMultiplier { .. }) => {
                    return Err(
                        "marginal-slope model does not support HazardMultiplier frailty"
                            .to_string(),
                    );
                }
                None => {
                    return Err(
                        "marginal-slope model is missing family_state.frailty; refit with current CLI"
                            .to_string(),
                    );
                }
            }
        }

        if let FittedFamily::Survival {
            survival_likelihood,
            frailty,
            ..
        } = &self.family_state
        {
            if matches!(
                survival_likelihood.as_deref(),
                Some("latent") | Some("latent-binary")
            ) {
                return Err(
                    "latent hazard-window models must persist explicit family_state metadata, not generic survival metadata"
                        .to_string(),
                );
            }
            if survival_likelihood.as_deref() == Some("marginal-slope") {
                if self.formula_logslope.is_none() {
                    return Err(
                        "survival marginal-slope model is missing formula_logslope; refit with current CLI"
                            .to_string(),
                    );
                }
                if self.z_column.is_none() {
                    return Err(
                        "survival marginal-slope model is missing z_column; refit with current CLI"
                            .to_string(),
                    );
                }
                let z_normalization = self.latent_z_normalization.ok_or_else(|| {
                    "survival marginal-slope model is missing latent_z_normalization; refit with current CLI"
                        .to_string()
                })?;
                z_normalization.validate("survival marginal-slope model")?;
                if self.logslope_baseline.is_none() {
                    return Err(
                        "survival marginal-slope model is missing logslope_baseline; refit with current CLI"
                            .to_string(),
                    );
                }
                if self
                    .resolved_termspec_logslope
                    .as_ref()
                    .or(self.resolved_termspec_noise.as_ref())
                    .is_none()
                {
                    return Err(
                        "survival marginal-slope model is missing resolved_termspec_logslope for the logslope surface"
                            .to_string(),
                    );
                }
                match frailty {
                    FrailtySpec::None
                    | FrailtySpec::GaussianShift {
                        sigma_fixed: Some(_),
                    } => {}
                    FrailtySpec::GaussianShift { sigma_fixed: None } => {
                        return Err(
                            "survival marginal-slope model requires a fixed GaussianShift sigma in family_state.frailty"
                                .to_string(),
                        );
                    }
                    FrailtySpec::HazardMultiplier { .. } => {
                        return Err(
                            "survival marginal-slope model does not support HazardMultiplier frailty"
                            .to_string(),
                        );
                    }
                }
            } else if !matches!(frailty, FrailtySpec::None) {
                return Err(
                    "non-marginal survival models do not currently persist a frailty modifier"
                        .to_string(),
                );
            }
        }
        if let FittedFamily::LatentSurvival { frailty } = &self.family_state {
            match frailty {
                FrailtySpec::HazardMultiplier {
                    sigma_fixed: Some(_),
                    ..
                } => {}
                FrailtySpec::HazardMultiplier {
                    sigma_fixed: None, ..
                } => {
                    return Err(
                        "latent survival model requires a fixed HazardMultiplier sigma in family_state.frailty"
                            .to_string(),
                    );
                }
                FrailtySpec::GaussianShift { .. } | FrailtySpec::None => {
                    return Err(
                        "latent survival model requires a fixed HazardMultiplier frailty specification"
                            .to_string(),
                    );
                }
            }
            if self.survival_likelihood.as_deref() != Some("latent") {
                return Err(
                    "latent survival model must persist survival_likelihood=latent".to_string(),
                );
            }
        }
        if let FittedFamily::LatentBinary { frailty } = &self.family_state {
            match frailty {
                FrailtySpec::HazardMultiplier {
                    sigma_fixed: Some(_),
                    ..
                } => {}
                FrailtySpec::HazardMultiplier {
                    sigma_fixed: None, ..
                } => {
                    return Err(
                        "latent binary model requires a fixed HazardMultiplier sigma in family_state.frailty"
                            .to_string(),
                    );
                }
                FrailtySpec::GaussianShift { .. } | FrailtySpec::None => {
                    return Err(
                        "latent binary model requires a fixed HazardMultiplier frailty specification"
                            .to_string(),
                    );
                }
            }
            if self.survival_likelihood.as_deref() != Some("latent-binary") {
                return Err(
                    "latent binary model must persist survival_likelihood=latent-binary"
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
        if matches!(
            self.family_state,
            FittedFamily::Standard {
                likelihood: LikelihoodFamily::BinomialLatentCLogLog,
                ..
            }
        ) {
            self.saved_latent_cloglog_state()?;
        }
        if matches!(
            self.family_state,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialLatentCLogLog,
                ..
            }
        ) {
            return Err(
                "latent-cloglog-binomial is not supported for location-scale saved models"
                    .to_string(),
            );
        }
        if matches!(self.family_state, FittedFamily::Survival { .. })
            && self.survival_likelihood.is_none()
        {
            return Err(
                "saved survival model is missing survival_likelihood metadata; refit with current CLI"
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
        let saved_link_wiggle = self.saved_link_wiggle()?;
        if has_any_saved_link_wiggle && saved_link_wiggle.is_none() {
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
        if self
            .survival_likelihood
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case("location-scale"))
        {
            validate_survival_location_scale_saved_fit(self.payload(), saved_link_wiggle.as_ref())?;
        }

        // Validate anchored-deviation replay contracts at LOAD/SAVE time rather
        // than waiting for first predict call. Previously these contracts
        // (span table dimensions, coefficient matrices, etc.) were only
        // asserted inside `saved_prediction_runtime`, which runs on the first
        // predict invocation. A corrupted runtime would therefore pass
        // `load_from_path` silently and fail later under a different error
        // surface. Enforcing the same check here makes the model self-
        // diagnostic: `gam fit` catches its own bad output at save, and
        // `gam predict` catches bad input at load rather than mid-pipeline.
        if let Some(runtime) = self.score_warp_runtime.as_ref() {
            runtime
                .validate_exact_replay_contract()
                .map_err(|err| format!("saved anchored score-warp runtime is invalid: {err}"))?;
        }
        if let Some(runtime) = self.link_deviation_runtime.as_ref() {
            runtime.validate_exact_replay_contract().map_err(|err| {
                format!("saved anchored link-deviation runtime is invalid: {err}")
            })?;
        }
        if matches!(self.family_state, FittedFamily::MarginalSlope { .. }) {
            validate_marginal_slope_saved_fit(
                self.fit_result.as_ref().expect("checked above"),
                self.score_warp_runtime.as_ref(),
                self.link_deviation_runtime.as_ref(),
                "fit_result",
            )?;
            let unified = self.unified.as_ref().ok_or_else(|| {
                "marginal-slope model is missing unified fit payload; refit with current CLI"
                    .to_string()
            })?;
            validate_marginal_slope_saved_fit(
                unified,
                self.score_warp_runtime.as_ref(),
                self.link_deviation_runtime.as_ref(),
                "unified",
            )?;
        }
        if self
            .survival_likelihood
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case("marginal-slope"))
        {
            validate_survival_marginal_slope_saved_fit(
                self.fit_result.as_ref().expect("checked above"),
                self.score_warp_runtime.as_ref(),
                self.link_deviation_runtime.as_ref(),
                "fit_result",
            )?;
            if let Some(unified) = self.unified.as_ref() {
                validate_survival_marginal_slope_saved_fit(
                    unified,
                    self.score_warp_runtime.as_ref(),
                    self.link_deviation_runtime.as_ref(),
                    "unified",
                )?;
            }
        }

        // Structural invariant: nonlinear saved models must retain a usable
        // posterior-mean backend. We prefer persisted covariance, but a saved
        // penalized Hessian is also sufficient because prediction can
        // reconstruct covariance on demand.
        let needs_covariance = !matches!(
            self.family_state.likelihood(),
            LikelihoodFamily::GaussianIdentity
        );
        if needs_covariance {
            if let Some(fit) = self.fit_result.as_ref() {
                if fit.beta_covariance().is_none() && fit.penalized_hessian().is_none() {
                    return Err(
                        "nonlinear model is missing both beta_covariance and a saved penalized Hessian in fit_result; \
                         posterior-mean prediction requires one of them at save time"
                            .to_string(),
                    );
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
        if let Some(v) = self.latent_z_normalization {
            v.validate("latent_z_normalization")?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::bernoulli_marginal_slope::exact_kernel::ANCHORED_DEVIATION_KERNEL;
    use crate::families::lognormal_kernel::FrailtySpec;
    use crate::pirls::PirlsStatus;
    use crate::solver::estimate::{FitArtifacts, FittedBlock, FittedLinkState};
    use crate::types::{LikelihoodScaleMetadata, LogLikelihoodNormalization};
    use ndarray::{Array1, Array2, array};

    fn empty_termspec() -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        }
    }

    fn anchored_runtime(basis_dim: usize) -> SavedAnchoredDeviationRuntime {
        SavedAnchoredDeviationRuntime {
            kernel: ANCHORED_DEVIATION_KERNEL.to_string(),
            breakpoints: vec![-1.0, 1.0],
            basis_dim,
            span_c0: vec![vec![0.0; basis_dim]],
            span_c1: vec![vec![0.0; basis_dim]],
            span_c2: vec![vec![0.0; basis_dim]],
            span_c3: vec![vec![0.0; basis_dim]],
        }
    }

    fn saved_fit(blocks: Vec<FittedBlock>) -> UnifiedFitResult {
        let beta = Array1::from_vec(
            blocks
                .iter()
                .flat_map(|block| block.beta.iter().copied())
                .collect(),
        );
        let p = beta.len();
        UnifiedFitResult {
            blocks,
            log_lambdas: Array1::zeros(0),
            lambdas: Array1::zeros(0),
            likelihood_family: Some(LikelihoodFamily::BinomialProbit),
            likelihood_scale: LikelihoodScaleMetadata::Unspecified,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: 0.0,
            standard_deviation: 1.0,
            covariance_conditional: Some(Array2::zeros((p, p))),
            covariance_corrected: Some(Array2::zeros((p, p))),
            inference: None,
            fitted_link: FittedLinkState::Standard(None),
            geometry: None,
            block_states: vec![],
            beta,
            pirls_status: PirlsStatus::Converged,
            max_abs_eta: 0.0,
            constraint_kkt: None,
            artifacts: FitArtifacts {
                pirls: None,
                survival_link_wiggle_knots: None,
                survival_link_wiggle_degree: None,
            },
            inner_cycles: 0,
        }
    }

    fn marginal_slope_payload(version: u32, fit: UnifiedFitResult) -> FittedModelPayload {
        let mut payload = FittedModelPayload::new(
            version,
            "y ~ 1".to_string(),
            ModelKind::MarginalSlope,
            FittedFamily::MarginalSlope {
                likelihood: LikelihoodFamily::BinomialProbit,
                base_link: Some(InverseLink::Standard(LinkFunction::Probit)),
                frailty: FrailtySpec::None,
            },
            "bernoulli-marginal-slope".to_string(),
        );
        payload.fit_result = Some(fit.clone());
        payload.unified = Some(fit);
        payload.data_schema = Some(DataSchema {
            columns: vec![SchemaColumn {
                name: "z".to_string(),
                kind: ColumnKindTag::Continuous,
                levels: vec![],
            }],
        });
        payload.training_headers = Some(vec!["z".to_string()]);
        payload.resolved_termspec = Some(empty_termspec());
        payload.resolved_termspec_noise = Some(empty_termspec());
        payload.formula_logslope = Some("1".to_string());
        payload.z_column = Some("z".to_string());
        payload.latent_z_normalization = Some(SavedLatentZNormalization { mean: 0.0, sd: 1.0 });
        payload.marginal_baseline = Some(0.0);
        payload.logslope_baseline = Some(0.0);
        payload.link = Some("probit".to_string());
        payload
    }

    fn survival_marginal_slope_payload(version: u32, fit: UnifiedFitResult) -> FittedModelPayload {
        let mut payload = FittedModelPayload::new(
            version,
            "Surv(entry, exit, event) ~ 1".to_string(),
            ModelKind::Survival,
            FittedFamily::Survival {
                likelihood: LikelihoodFamily::RoystonParmar,
                survival_likelihood: Some("marginal-slope".to_string()),
                survival_distribution: Some("probit".to_string()),
                frailty: FrailtySpec::None,
            },
            "survival".to_string(),
        );
        payload.fit_result = Some(fit.clone());
        payload.unified = Some(fit);
        payload.survival_likelihood = Some("marginal-slope".to_string());
        payload.survival_distribution = Some("probit".to_string());
        payload
    }

    #[test]
    fn validate_for_persistence_rejects_marginal_slope_score_warp_basis_mismatch() {
        let fit = saved_fit(vec![
            FittedBlock {
                beta: array![0.1],
                role: BlockRole::Mean,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.2],
                role: BlockRole::Scale,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.3],
                role: BlockRole::Mean,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
        ]);
        let mut payload = marginal_slope_payload(MODEL_PAYLOAD_VERSION, fit);
        payload.score_warp_runtime = Some(anchored_runtime(2));

        let err = FittedModel::from_payload(payload)
            .validate_for_persistence()
            .expect_err("marginal-slope score-warp basis mismatch should fail validation");
        assert!(err.contains("score-warp coefficient mismatch"));
    }

    #[test]
    fn saved_prediction_runtime_rejects_survival_marginal_slope_link_basis_mismatch() {
        let fit = saved_fit(vec![
            FittedBlock {
                beta: array![0.1],
                role: BlockRole::Time,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.2],
                role: BlockRole::Mean,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.3],
                role: BlockRole::Scale,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.4],
                role: BlockRole::LinkWiggle,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
        ]);
        let mut payload = survival_marginal_slope_payload(MODEL_PAYLOAD_VERSION, fit);
        payload.link_deviation_runtime = Some(anchored_runtime(2));

        let err = FittedModel::from_payload(payload)
            .saved_prediction_runtime()
            .expect_err(
                "survival marginal-slope link basis mismatch should fail runtime validation",
            );
        assert!(err.contains("link-deviation coefficient mismatch"));
    }

    #[test]
    fn saved_prediction_runtime_rejects_stale_payload_version() {
        let fit = saved_fit(vec![
            FittedBlock {
                beta: array![0.1],
                role: BlockRole::Mean,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: array![0.2],
                role: BlockRole::Scale,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
        ]);
        let payload = marginal_slope_payload(MODEL_PAYLOAD_VERSION - 1, fit);

        let err = FittedModel::from_payload(payload)
            .saved_prediction_runtime()
            .expect_err("stale payload version should fail before runtime assembly");
        assert!(err.contains("payload schema mismatch"));
    }
}

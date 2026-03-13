use crate::estimate::{BlockRole, FitResult, FittedLinkParameters, UnifiedFitResult};
use crate::inference::predict::{PredictableModel, StandardPredictor};
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec};
use crate::smooth::{
    AdaptiveRegularizationDiagnostics, BoundedCoefficientPriorSpec, LinearCoefficientGeometry,
    SmoothBasisSpec, TensorBSplineIdentifiability, TermCollectionSpec,
};
use crate::terms::basis::{BSplineKnotSpec, CenterStrategy, SpatialIdentifiability};
use crate::types::{
    InverseLink, LikelihoodFamily, LinkFunction, MixtureLinkState, SasLinkSpec, SasLinkState,
};
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
    pub fit_result: Option<FitResult>,
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
    pub joint_beta_link: Option<Vec<f64>>,
    #[serde(default)]
    pub joint_knot_range: Option<(f64, f64)>,
    #[serde(default)]
    pub joint_knot_vector: Option<Vec<f64>>,
    #[serde(default)]
    pub joint_link_transform: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub joint_degree: Option<usize>,
    #[serde(default)]
    pub jointridge_used: Option<f64>,
    #[serde(default)]
    pub probitwiggle_knots: Option<Vec<f64>>,
    #[serde(default)]
    pub probitwiggle_degree: Option<usize>,
    #[serde(default)]
    pub betawiggle: Option<Vec<f64>>,
    #[serde(default)]
    pub timewiggle_knots: Option<Vec<f64>>,
    #[serde(default)]
    pub timewiggle_degree: Option<usize>,
    #[serde(default)]
    pub timewiggle_penalty_orders: Option<Vec<usize>>,
    #[serde(default)]
    pub timewiggle_double_penalty: Option<bool>,
    #[serde(default)]
    pub betatimewiggle: Option<Vec<f64>>,
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
            beta_noise: None,
            noise_projection: None,
            noise_center: None,
            noise_scale: None,
            noise_non_intercept_start: None,
            gaussian_response_scale: None,
            joint_beta_link: None,
            joint_knot_range: None,
            joint_knot_vector: None,
            joint_link_transform: None,
            joint_degree: None,
            jointridge_used: None,
            probitwiggle_knots: None,
            probitwiggle_degree: None,
            betawiggle: None,
            timewiggle_knots: None,
            timewiggle_degree: None,
            timewiggle_penalty_orders: None,
            timewiggle_double_penalty: None,
            betatimewiggle: None,
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
    Survival { payload: FittedModelPayload },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ModelKind {
    Standard,
    LocationScale,
    Survival,
    FlexibleLink,
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
    Survival {
        likelihood: LikelihoodFamily,
        #[serde(default)]
        survival_likelihood: Option<String>,
        #[serde(default)]
        survival_distribution: Option<String>,
    },
    FlexibleLink {
        likelihood: LikelihoodFamily,
        link: LinkFunction,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredictModelClass {
    Standard,
    GaussianLocationScale,
    BinomialLocationScale,
    Survival,
}

impl FittedFamily {
    #[inline]
    pub fn likelihood(&self) -> LikelihoodFamily {
        match self {
            Self::Standard { likelihood, .. }
            | Self::LocationScale { likelihood, .. }
            | Self::Survival { likelihood, .. }
            | Self::FlexibleLink { likelihood, .. } => *likelihood,
        }
    }
}

impl FittedModel {
    pub fn from_payload(mut payload: FittedModelPayload) -> Self {
        let likelihood = payload.family_state.likelihood();
        let class = match payload.model_kind {
            ModelKind::Survival => PredictModelClass::Survival,
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
    fn payload(&self) -> &FittedModelPayload {
        match self {
            Self::Standard { payload }
            | Self::LocationScale { payload }
            | Self::Survival { payload } => payload,
        }
    }

    #[inline]
    fn payload_mut(&mut self) -> &mut FittedModelPayload {
        match self {
            Self::Standard { payload }
            | Self::LocationScale { payload }
            | Self::Survival { payload } => payload,
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
        match (&mut payload.family_state, &fit.fitted_link_parameters) {
            (
                FittedFamily::Standard {
                    likelihood: LikelihoodFamily::BinomialSas,
                    sas_state,
                    ..
                },
                FittedLinkParameters::Sas { state, covariance },
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
                FittedLinkParameters::BetaLogistic { state, covariance },
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
                FittedLinkParameters::Mixture { state, covariance },
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
            FittedFamily::FlexibleLink { link, .. } => {
                Ok(stateful.or(Some(InverseLink::Standard(*link))))
            }
            FittedFamily::Survival { .. } => Ok(None),
        }
    }

    /// Build a `PredictableModel` for the standard (single-block) case.
    ///
    /// Returns `None` for location-scale or survival models which require
    /// specialised predictors not yet wired through this trait.
    pub fn predictor(&self) -> Option<Box<dyn PredictableModel>> {
        if !matches!(self, FittedModel::Standard { .. }) {
            return None;
        }
        let fit = self.fit_result.as_ref()?;
        let family = self.family_state.likelihood();
        let link_kind = self.resolved_inverse_link().ok().flatten();
        let covariance = fit.beta_covariance().cloned();
        Some(Box::new(StandardPredictor {
            beta: fit.beta.clone(),
            family,
            link_kind,
            covariance,
        }))
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
            ensure_finite_scalar("fit_result.standard_deviation", fit.standard_deviation)?;
            ensure_finite_scalar("fit_result.finalgrad_norm", fit.finalgrad_norm)?;
            ensure_finite_scalar("fit_result.deviance", fit.deviance)?;
            ensure_finite_scalar("fit_result.stable_penalty_term", fit.stable_penalty_term)?;
            ensure_finite_scalar("fit_result.max_abs_eta", fit.max_abs_eta)?;
            ensure_finite_scalar("fit_result.reml_score", fit.reml_score)?;
            validate_all_finite("fit_result.beta", fit.beta.iter().copied())?;
            validate_all_finite("fit_result.lambdas", fit.lambdas.iter().copied())?;
            if let Some(inf) = fit.inference() {
                ensure_finite_scalar("fit_result.edf_total", inf.edf_total)?;
                validate_all_finite("fit_result.edf_by_block", inf.edf_by_block.iter().copied())?;
                validate_all_finite(
                    "fit_result.working_weights",
                    inf.working_weights.iter().copied(),
                )?;
                validate_all_finite(
                    "fit_result.working_response",
                    inf.working_response.iter().copied(),
                )?;
                validate_all_finite(
                    "fit_result.penalized_hessian",
                    inf.penalized_hessian.iter().copied(),
                )?;
                if let Some(v) = inf.beta_covariance.as_ref() {
                    validate_all_finite("fit_result.beta_covariance", v.iter().copied())?;
                }
                if let Some(v) = inf.beta_covariance_corrected.as_ref() {
                    validate_all_finite("fit_result.beta_covariance_corrected", v.iter().copied())?;
                }
                if let Some(v) = inf.beta_standard_errors.as_ref() {
                    validate_all_finite("fit_result.beta_standard_errors", v.iter().copied())?;
                }
                if let Some(v) = inf.beta_standard_errors_corrected.as_ref() {
                    validate_all_finite(
                        "fit_result.beta_standard_errors_corrected",
                        v.iter().copied(),
                    )?;
                }
            }
            match &fit.fitted_link_parameters {
                FittedLinkParameters::Standard => {}
                FittedLinkParameters::Mixture { state, covariance } => {
                    validate_all_finite("fit_result.mixture_link_rho", state.rho.iter().copied())?;
                    validate_all_finite(
                        "fit_result.mixture_linkweights",
                        state.pi.iter().copied(),
                    )?;
                    if let Some(v) = covariance.as_ref() {
                        validate_all_finite(
                            "fit_result.mixture_link_param_covariance",
                            v.iter().copied(),
                        )?;
                    }
                }
                FittedLinkParameters::Sas { state, covariance }
                | FittedLinkParameters::BetaLogistic { state, covariance } => {
                    ensure_finite_scalar("fit_result.sas_epsilon", state.epsilon)?;
                    ensure_finite_scalar("fit_result.sas_log_delta", state.log_delta)?;
                    ensure_finite_scalar("fit_result.sas_delta", state.delta)?;
                    if let Some(v) = covariance.as_ref() {
                        validate_all_finite("fit_result.sas_param_covariance", v.iter().copied())?;
                    }
                }
            }
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
            ("survivalridge_lambda", self.survivalridge_lambda),
            ("jointridge_used", self.jointridge_used),
        ] {
            if let Some(v) = opt {
                ensure_finite_scalar(name, v)?;
            }
        }

        if let Some((a, b)) = self.joint_knot_range {
            ensure_finite_scalar("joint_knot_range.0", a)?;
            ensure_finite_scalar("joint_knot_range.1", b)?;
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
        if let Some(v) = self.joint_beta_link.as_ref() {
            validate_all_finite("joint_beta_link", v.iter().copied())?;
        }
        if let Some(v) = self.joint_knot_vector.as_ref() {
            validate_all_finite("joint_knot_vector", v.iter().copied())?;
        }
        if let Some(v) = self.betawiggle.as_ref() {
            validate_all_finite("betawiggle", v.iter().copied())?;
        }
        if let Some(v) = self.betatimewiggle.as_ref() {
            validate_all_finite("betatimewiggle", v.iter().copied())?;
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

fn ensure_finite_scalar(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{name} must be finite, got {value}"))
    }
}

fn validate_all_finite<I>(label: &str, values: I) -> Result<(), String>
where
    I: IntoIterator<Item = f64>,
{
    for (idx, v) in values.into_iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("{label}[{idx}] must be finite, got {v}"));
        }
    }
    Ok(())
}

fn validate_frozen_term_collectionspec(
    spec: &TermCollectionSpec,
    label: &str,
) -> Result<(), String> {
    for linear in &spec.linear_terms {
        if let (Some(min), Some(max)) = (linear.coefficient_min, linear.coefficient_max)
            && (!min.is_finite() || !max.is_finite() || min > max)
        {
            return Err(format!(
                "{label} linear term '{}' has invalid coefficient constraint [{min}, {max}]",
                linear.name
            ));
        }
        if let Some(min) = linear.coefficient_min
            && !min.is_finite()
        {
            return Err(format!(
                "{label} linear term '{}' has non-finite coefficient minimum {min}",
                linear.name
            ));
        }
        if let Some(max) = linear.coefficient_max
            && !max.is_finite()
        {
            return Err(format!(
                "{label} linear term '{}' has non-finite coefficient maximum {max}",
                linear.name
            ));
        }
        if let LinearCoefficientGeometry::Bounded { min, max, prior } = &linear.coefficient_geometry
        {
            if !min.is_finite() || !max.is_finite() || min >= max {
                return Err(format!(
                    "{label} bounded term '{}' has invalid bounds [{min}, {max}]",
                    linear.name
                ));
            }
            match prior {
                BoundedCoefficientPriorSpec::None | BoundedCoefficientPriorSpec::Uniform => {}
                BoundedCoefficientPriorSpec::Beta { a, b } => {
                    if !a.is_finite() || !b.is_finite() || *a < 1.0 || *b < 1.0 {
                        return Err(format!(
                            "{label} bounded term '{}' has invalid Beta prior ({a}, {b})",
                            linear.name
                        ));
                    }
                }
            }
        }
    }
    for st in &spec.smooth_terms {
        match &st.basis {
            SmoothBasisSpec::BSpline1D { spec, .. } => {
                if !matches!(spec.knotspec, BSplineKnotSpec::Provided(_)) {
                    return Err(format!(
                        "{label} term '{}' is not frozen: BSpline knotspec must be Provided",
                        st.name
                    ));
                }
            }
            SmoothBasisSpec::ThinPlate { spec, .. } => {
                if !matches!(spec.center_strategy, CenterStrategy::UserProvided(_)) {
                    return Err(format!(
                        "{label} term '{}' is not frozen: ThinPlate centers must be UserProvided",
                        st.name
                    ));
                }
                if matches!(
                    spec.identifiability,
                    SpatialIdentifiability::OrthogonalToParametric
                ) {
                    return Err(format!(
                        "{label} term '{}' is not frozen: ThinPlate identifiability must be FrozenTransform or None",
                        st.name
                    ));
                }
            }
            SmoothBasisSpec::Matern { spec, .. } => {
                if !matches!(spec.center_strategy, CenterStrategy::UserProvided(_)) {
                    return Err(format!(
                        "{label} term '{}' is not frozen: Matern centers must be UserProvided",
                        st.name
                    ));
                }
            }
            SmoothBasisSpec::Duchon { spec, .. } => {
                if !matches!(spec.center_strategy, CenterStrategy::UserProvided(_)) {
                    return Err(format!(
                        "{label} term '{}' is not frozen: Duchon centers must be UserProvided",
                        st.name
                    ));
                }
                if matches!(
                    spec.identifiability,
                    SpatialIdentifiability::OrthogonalToParametric
                ) {
                    return Err(format!(
                        "{label} term '{}' is not frozen: Duchon identifiability must be FrozenTransform or None",
                        st.name
                    ));
                }
            }
            SmoothBasisSpec::TensorBSpline { spec, .. } => {
                for (dim, marginal) in spec.marginalspecs.iter().enumerate() {
                    if !matches!(marginal.knotspec, BSplineKnotSpec::Provided(_)) {
                        return Err(format!(
                            "{label} term '{}' dim {} is not frozen: tensor marginal knotspec must be Provided",
                            st.name, dim
                        ));
                    }
                }
                if matches!(
                    spec.identifiability,
                    TensorBSplineIdentifiability::SumToZero
                ) {
                    return Err(format!(
                        "{label} term '{}' is not frozen: tensor identifiability must be FrozenTransform or None",
                        st.name
                    ));
                }
            }
        }
    }

    for rt in &spec.random_effect_terms {
        if rt.frozen_levels.is_none() {
            return Err(format!(
                "{label} random-effect term '{}' is not frozen: missing frozen_levels",
                rt.name
            ));
        }
    }
    Ok(())
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

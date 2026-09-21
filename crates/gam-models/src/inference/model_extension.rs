//! Deployment-time model surgery: extending a fitted model with a new
//! random-effect group level without a refit.
//!
//! This capability used to live entirely inside the PyO3 boundary crate, which
//! made it reachable from Python only. SPEC rule 9 (CLI / Python / Rust
//! parity) requires one source of truth, so the typed request and the whole
//! mutation live here next to [`FittedModel`]; the FFI layer is now a thin
//! JSON adapter over [`FittedModel::extend_with_group`].

use crate::inference::model::{
    ColumnKindTag, FittedModel, FittedModelPayload, PredictModelClass, SavedDeploymentExtension,
    SchemaColumn,
};
use gam_solve::estimate::{BlockRole, UnifiedFitResult};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A request to extend a fitted model with one or more new group levels.
///
/// The field set is the on-wire contract the Python `extend_with_group` API
/// already spoke; it is plain serde over core types, so the CLI and Rust
/// library callers can build it directly instead of round-tripping JSON.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExtendGroupRequest {
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub term: Option<String>,
    #[serde(default)]
    pub column: Option<String>,
    #[serde(default)]
    pub level: Option<serde_json::Value>,
    #[serde(default)]
    pub levels: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub prior: Option<serde_json::Value>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtensionPrior {
    #[serde(default)]
    mean: Option<f64>,
    #[serde(default)]
    mu: Option<f64>,
    #[serde(default)]
    variance: Option<f64>,
    #[serde(default)]
    precision: Option<f64>,
}

impl FittedModel {
    /// Extend this model in place with the requested random-effect levels.
    ///
    /// On success the model has passed both save-time gates
    /// (`validate_for_persistence` and `validate_numeric_finiteness`), so any
    /// caller may persist or predict with it directly. On failure the model is
    /// left partially mutated and must be discarded — callers that need the
    /// original should clone before calling.
    pub fn extend_with_group(&mut self, request: ExtendGroupRequest) -> Result<(), String> {
        if !matches!(self.predict_model_class(), PredictModelClass::Standard) {
            return Err(format!(
                "extend_with_group currently supports standard GAM models only; got '{}'",
                self.predict_model_class().name()
            ));
        }
        if self.has_link_wiggle() {
            return Err("extend_with_group does not support link-wiggle models".to_string());
        }
        let ExtendGroupRequest {
            kind,
            name,
            term,
            column,
            level,
            levels,
            metadata,
            prior,
        } = request;
        let kind = kind
            .as_deref()
            .unwrap_or("random-effect-level")
            .replace('_', "-");
        if kind != "random-effect-level" {
            return Err(format!(
                "extend_with_group supports kind='random-effect-level'; got '{kind}'"
            ));
        }
        let mut levels = levels.unwrap_or_default();
        if let Some(level) = level {
            levels.push(level);
        }
        if levels.is_empty() {
            return Err("extend_with_group requires level or levels".to_string());
        }
        let term = match term.or(column) {
            Some(term) => term,
            None => {
                let payload = self.payload();
                let spec = payload.resolved_termspec.as_ref().ok_or_else(|| {
                    "extend_with_group requires saved resolved_termspec; refit".to_string()
                })?;
                if spec.random_effect_terms.len() == 1 {
                    spec.random_effect_terms[0].name.clone()
                } else {
                    return Err(
                        "extend_with_group requires term when the model has zero or multiple group terms"
                            .to_string(),
                    );
                }
            }
        };

        for level in levels {
            extend_model_with_random_effect_level(
                self,
                term.as_str(),
                name.as_deref(),
                level,
                metadata.clone(),
                prior.clone(),
            )?;
        }
        self.validate_for_persistence()?;
        self.validate_numeric_finiteness()?;
        Ok(())
    }
}

fn extend_model_with_random_effect_level(
    model: &mut FittedModel,
    term_name: &str,
    requested_name: Option<&str>,
    level: serde_json::Value,
    metadata: Option<serde_json::Value>,
    prior: Option<serde_json::Value>,
) -> Result<(), String> {
    let payload: &mut FittedModelPayload = &mut *model;
    let (term_idx, feature_col, penalty_index) = {
        let spec = payload.resolved_termspec.as_ref().ok_or_else(|| {
            "extend_with_group requires saved resolved_termspec; refit".to_string()
        })?;
        let term_idx = spec
            .random_effect_terms
            .iter()
            .position(|term| term.name == term_name)
            .ok_or_else(|| format!("extend_with_group unknown random-effect term '{term_name}'"))?;
        (
            term_idx,
            spec.random_effect_terms[term_idx].feature_col,
            spec.random_effect_penalty_index(term_idx),
        )
    };
    let coefficient_index = payload
        .fit_result
        .as_ref()
        .ok_or_else(|| "extend_with_group requires saved fit_result; refit".to_string())?
        .beta
        .len();
    let (coefficient_mean, supplied_variance) = extension_prior_parameters(prior.as_ref())?;
    let coefficient_variance = match supplied_variance {
        Some(variance) => variance,
        None => default_unseen_level_prior_variance(
            payload
                .fit_result
                .as_ref()
                .ok_or_else(|| "extend_with_group requires saved fit_result; refit".to_string())?,
            penalty_index,
            term_name,
        )?,
    };
    if let Some(fit) = payload.fit_result.as_ref() {
        unscaled_prior_precision(fit, coefficient_variance)?;
    }
    let schema = payload
        .data_schema
        .as_mut()
        .ok_or_else(|| "extend_with_group requires saved data_schema; refit".to_string())?;
    let schema_col = schema.columns.get_mut(feature_col).ok_or_else(|| {
        format!(
            "extend_with_group term '{term_name}' feature column {feature_col} out of saved schema bounds"
        )
    })?;
    let (level_bits, encoded_value) = level_bits_for_extension(schema_col, &level)?;
    {
        let spec = payload.resolved_termspec.as_ref().ok_or_else(|| {
            "extend_with_group requires saved resolved_termspec; refit".to_string()
        })?;
        let levels = spec.random_effect_terms[term_idx]
            .frozen_levels
            .as_ref()
            .ok_or_else(|| {
                format!(
                    "extend_with_group term '{term_name}' is not frozen; refit with persisted metadata"
                )
            })?;
        if levels.contains(&level_bits) {
            return Err(format!(
                "extend_with_group level {} already exists for random-effect term '{term_name}'",
                compact_json(&level)
            ));
        }
    }
    if payload.deployment_extensions.iter().any(|extension| {
        extension.kind == "random-effect-level"
            && extension.term == term_name
            && extension.level_bits == level_bits
    }) {
        return Err(format!(
            "extend_with_group level {} is already deployed for random-effect term '{term_name}'",
            compact_json(&level)
        ));
    }
    extend_training_feature_range(
        payload.training_feature_ranges.as_mut(),
        feature_col,
        encoded_value,
    );
    insert_coefficient_into_saved_fit(
        payload.fit_result.as_mut(),
        coefficient_index,
        coefficient_mean,
        coefficient_variance,
    )?;

    let extension_name = requested_name
        .map(str::to_string)
        .unwrap_or_else(|| format!("{term_name}:{}", compact_json(&level)));
    if let Some(metadata_value) = metadata.clone() {
        let group_metadata = payload.group_metadata.get_or_insert_with(BTreeMap::new);
        group_metadata.insert(extension_name.clone(), metadata_value.clone());
    }
    payload
        .deployment_extensions
        .push(SavedDeploymentExtension {
            name: extension_name,
            kind: "random-effect-level".to_string(),
            term: term_name.to_string(),
            level,
            level_bits,
            coefficient_index,
            coefficient_mean,
            coefficient_variance,
            metadata,
            prior,
        });
    Ok(())
}

fn level_bits_for_extension(
    schema_col: &mut SchemaColumn,
    level: &serde_json::Value,
) -> Result<(u64, f64), String> {
    match schema_col.kind {
        ColumnKindTag::Categorical => {
            let label = match level {
                serde_json::Value::String(s) => s.clone(),
                other => compact_json(other),
            };
            if schema_col.levels.iter().any(|existing| existing == &label) {
                return Err(format!(
                    "extend_with_group categorical level '{label}' already exists in column '{}'",
                    schema_col.name
                ));
            }
            let encoded = schema_col.levels.len() as f64;
            schema_col.levels.push(label);
            Ok((encoded.to_bits(), encoded))
        }
        ColumnKindTag::Continuous | ColumnKindTag::Binary => {
            let value = json_level_to_f64(level)?;
            Ok((value.to_bits(), value))
        }
    }
}

fn json_level_to_f64(value: &serde_json::Value) -> Result<f64, String> {
    let out = match value {
        serde_json::Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| format!("extend_with_group level {n} is not representable as f64"))?,
        serde_json::Value::String(s) => s
            .parse::<f64>()
            .map_err(|_| format!("extend_with_group level '{s}' is not numeric"))?,
        other => {
            return Err(format!(
                "extend_with_group numeric random-effect levels must be numbers or numeric strings; got {}",
                compact_json(other)
            ));
        }
    };
    if !out.is_finite() {
        return Err(format!(
            "extend_with_group random-effect level must be finite; got {out}"
        ));
    }
    Ok(out)
}

fn compact_json(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|error| format!("<unserializable: {error}>"))
}

fn extension_prior_parameters(
    prior: Option<&serde_json::Value>,
) -> Result<(f64, Option<f64>), String> {
    let Some(value) = prior else {
        return Ok((0.0, None));
    };
    if value.is_null() {
        return Ok((0.0, None));
    }
    let parsed: ExtensionPrior = serde_json::from_value(value.clone())
        .map_err(|err| format!("failed to parse extend_with_group prior: {err}"))?;
    let mean = parsed.mean.or(parsed.mu).unwrap_or(0.0);
    if !mean.is_finite() {
        return Err(format!(
            "extend_with_group prior mean must be finite; got {mean}"
        ));
    }
    let variance = match (parsed.variance, parsed.precision) {
        (Some(variance), _) => {
            if !(variance.is_finite() && variance > 0.0) {
                return Err(format!(
                    "extend_with_group prior variance must be finite and positive; got {variance}"
                ));
            }
            Some(variance)
        }
        (None, Some(precision)) => {
            if !(precision.is_finite() && precision > 0.0) {
                return Err(format!(
                    "extend_with_group prior precision must be finite and positive; got {precision}"
                ));
            }
            Some(1.0 / precision)
        }
        (None, None) => None,
    };
    Ok((mean, variance))
}

/// The default prior variance of an unseen random-effect level: the fitted
/// variance component `σ_b²` of the term's ridge, in the same units as the
/// reported coefficient covariance `Vb = scale · H⁻¹`.
///
/// The penalty `λ·S` enters the stored `H = XᵀWX + S_λ` unscaled, so the
/// ridge's prior precision in `H` units is `λ` and its covariance in `Vb`
/// units is `scale / λ`, where `scale` is
/// [`UnifiedFitResult::coefficient_covariance_scale`]: the profiled `σ̂²` for
/// the scale-free Gaussian (mgcv's `λ = σ̂² / σ_b²`, #674) and `1` for every
/// family whose working weight already carries `1/φ` (Gamma, Tweedie, Beta,
/// NB, fixed-scale Gaussian, Poisson, Binomial). The response-level
/// `dispersion_phi()` is a different quantity (`1/shape` for Gamma) and would
/// scale these families' prior by `φ` a second time.
fn default_unseen_level_prior_variance(
    fit: &UnifiedFitResult,
    penalty_index: usize,
    term_name: &str,
) -> Result<f64, String> {
    let lambda = fit
        .lambdas
        .get(penalty_index)
        .copied()
        .filter(|lambda| lambda.is_finite() && *lambda > 0.0)
        .ok_or_else(|| {
            format!("extend_with_group term '{term_name}' has no finite positive prior lambda")
        })?;
    let variance = extension_covariance_scale(fit)? / lambda;
    if !(variance.is_finite() && variance > 0.0) {
        return Err(format!("extend_with_group term '{term_name}' prior variance is not finite and positive ({variance})"));
    }
    Ok(variance)
}

/// `scale` in `Vb = scale · H⁻¹` for a saved fit, required finite and
/// positive: a new coordinate's prior variance and its entry in the unscaled
/// penalized Hessian are related through it.
fn extension_covariance_scale(fit: &UnifiedFitResult) -> Result<f64, String> {
    let scale = fit.coefficient_covariance_scale().map_err(|err| {
        format!("cannot resolve the coefficient-covariance scale for the unseen-level prior: {err}")
    })?;
    if !(scale.is_finite() && scale > 0.0) {
        return Err(format!(
            "extend_with_group saved fit has a non-finite or non-positive coefficient-covariance \
             scale ({scale}); cannot place the new level's prior"
        ));
    }
    Ok(scale)
}

/// The new coordinate's diagonal in the UNSCALED penalized Hessian
/// (`UnscaledPrecision`, `Vb = scale · H⁻¹`) for a prior variance stated in
/// `Vb` units. For the default prior this is exactly the ridge's `λ`.
fn unscaled_prior_precision(fit: &UnifiedFitResult, variance: f64) -> Result<f64, String> {
    if !(variance.is_finite() && variance > 0.0) {
        return Err(format!("extend_with_group prior variance must be finite and positive; got {variance}"));
    }
    let precision = extension_covariance_scale(fit)? / variance;
    if !(precision.is_finite() && precision > 0.0) {
        return Err(format!("extend_with_group unscaled prior precision is not finite and positive ({precision})"));
    }
    Ok(precision)
}

fn extend_training_feature_range(
    ranges: Option<&mut Vec<(f64, f64)>>,
    feature_col: usize,
    value: f64,
) {
    if let Some(ranges) = ranges
        && let Some((lo, hi)) = ranges.get_mut(feature_col)
    {
        if value.is_finite() {
            *lo = (*lo).min(value);
            *hi = (*hi).max(value);
        }
    }
}

fn insert_coefficient_into_saved_fit(
    fit: Option<&mut UnifiedFitResult>,
    index: usize,
    value: f64,
    variance: f64,
) -> Result<(), String> {
    let Some(fit) = fit else {
        return Ok(());
    };
    if !(variance.is_finite() && variance > 0.0) {
        return Err(format!(
            "extend_with_group coefficient variance must be finite and positive; got {variance}"
        ));
    }
    // `variance` is in reported-covariance units; the Hessians below are the
    // unscaled `H` of `Vb = scale · H⁻¹`, so the new diagonal is
    // `scale / variance` (resolved before any mutation).
    let precision_diag = unscaled_prior_precision(fit, variance)?;
    if index > fit.beta.len() {
        return Err(format!(
            "extend_with_group coefficient index {index} exceeds fit coefficient length {}",
            fit.beta.len()
        ));
    }
    fit.beta = insert_array1(&fit.beta, index, value);
    let block_idx = fit
        .blocks
        .iter()
        .position(|block| block.role == BlockRole::Mean)
        .unwrap_or(0);
    if block_idx >= fit.blocks.len() {
        return Err("extend_with_group saved fit has no coefficient blocks".to_string());
    }
    if index > fit.blocks[block_idx].beta.len() {
        return Err(format!(
            "extend_with_group coefficient index {index} exceeds mean block length {}",
            fit.blocks[block_idx].beta.len()
        ));
    }
    fit.blocks[block_idx].beta = insert_array1(&fit.blocks[block_idx].beta, index, value);
    // The saved geometry carries the coefficient gauge, and `UnifiedFitResult`
    // validation requires the gauge's raw block widths to equal the saved
    // per-block β widths. Growing `blocks[block_idx].beta` above without
    // growing the gauge alongside it is exactly the +1 disagreement that
    // refused nine Python deployment tests with
    //   "geometry coefficient gauge raw block 0 has width W, expected saved
    //    beta width W+1"
    // (5→6, 42→43, 82→83 — always the one appended level). A new unseen
    // random-effect level is a FREE raw coordinate: it took part in no
    // identifiability constraint of the fit, so it enters the gauge as an
    // identity row carrying its own reduced coordinate.
    //
    // That reduced coordinate's index is also the only correct insertion point
    // for the two REDUCED-coordinate objects below. `penalized_hessian` on both
    // `geometry` and `inference` is validated against `gauge.reduced_total()`
    // whenever a geometry is present, so inserting at the RAW `index` is only
    // accidentally right on an identity gauge and is out of bounds as soon as
    // any block is genuinely reduced.
    let (grown_gauge, reduced_index) = match fit.geometry.as_ref() {
        Some(geometry) => {
            if geometry.constrained_posterior.is_some() {
                // `constrained_posterior` is the other active-frame object, and
                // its truncation identity is stated in the pre-extension
                // coordinates. Widening the frame underneath it would leave a
                // posterior whose truncation refers to a coordinate system that
                // no longer exists, so refuse instead.
                return Err(
                    "extend_with_group cannot extend a fit carrying an inequality-truncated \
                     posterior geometry: the truncation identity is stated in the pre-extension \
                     active coordinates. Refit with the new level present."
                        .to_string(),
                );
            }
            let gauge = &geometry.coefficient_gauge;
            let raw_end = gauge.block_starts_raw[block_idx + 1];
            if index != raw_end {
                return Err(format!(
                    "extend_with_group appends coefficient {index} but the saved gauge places \
                     block {block_idx}'s raw coordinates at ..{raw_end}; the appended level would \
                     not land in the block whose β was grown"
                ));
            }
            let (grown, reduced_index) = gauge.append_free_coordinate_to_block(block_idx)?;
            (Some(grown), reduced_index)
        }
        None => (None, index),
    };
    // No-refit posterior algebra for a deployment-only block:
    //
    // The fitted posterior precision for the original coefficients is H_old.
    // Extending with a new random-effect coefficient b and no likelihood
    // refit contributes only its Gaussian prior,
    //
    //   -log p(b) = 1/2 (b - mu)' (lambda_new S_new) (b - mu) + const.
    //
    // Since no old likelihood rows or old penalties are recomputed, the joint
    // unscaled precision is blockdiag(H_old, lambda_new S_new), and the
    // reported covariance is scale * blockdiag(H_old, lambda_new S_new)^{-1}
    // = blockdiag(V_old, scale S_new^{-1}/lambda_new).  The current API
    // extends one iid random-effect coordinate at a time, so S_new = [1] and
    // `variance` is exactly scale/lambda_new (see
    // `default_unseen_level_prior_variance`), or the caller's supplied scalar
    // prior covariance; either way the unscaled Hessian gains scale/variance.
    if let Some(cov) = fit.covariance_conditional.as_mut() {
        *cov = insert_symmetric_array2(cov, index, variance)?;
    }
    if let Some(cov) = fit.covariance_corrected.as_mut() {
        *cov = insert_symmetric_array2(cov, index, variance)?;
    }
    let variance_diag = variance;
    if let Some(inference) = fit.inference.as_mut() {
        // Boundary adapter: `penalized_hessian` is the `UnscaledPrecision`
        // newtype; unwrap for the `insert_symmetric_array2` helper and wrap
        // the result back on assignment.
        inference.penalized_hessian = insert_symmetric_array2(
            inference.penalized_hessian.as_array(),
            reduced_index,
            precision_diag,
        )?
        .into();
        if let Some(se) = inference.factorized_standard_errors.as_mut() {
            *se = insert_array1(se, index, variance_diag.sqrt());
        }
        // The new coordinate carries no smoothing-parameter uncertainty, so the
        // factorized correction `C = B·Bᵀ` gains a zero row in `B` and its
        // corrected variance is the prior variance (#3283).
        if let Some(factorized) = inference.smoothing_correction_factorized.as_mut() {
            factorized.factor = insert_zero_row(&factorized.factor, index)?;
            factorized.standard_errors =
                insert_array1(&factorized.standard_errors, index, variance_diag.sqrt());
        }
        if let Some(cov) = inference.beta_covariance_frequentist.as_mut() {
            *cov = insert_symmetric_array2(cov, index, 0.0)?;
        }
        if let Some(influence) = inference.coefficient_influence.as_mut() {
            *influence = insert_symmetric_array2(influence, index, 0.0)?;
        }
        if let Some(correction) = inference.smoothing_correction.as_mut() {
            *correction = insert_symmetric_array2(correction, index, 0.0)?;
        }
        if let Some(qs) = inference.reparam_qs.as_mut() {
            *qs = insert_symmetric_array2(qs, index, 1.0)?;
        }
    }
    if let Some(geometry) = fit.geometry.as_mut() {
        geometry.penalized_hessian = insert_symmetric_array2(
            geometry.penalized_hessian.as_array(),
            reduced_index,
            precision_diag,
        )?
        .into();
        if let Some(gauge) = grown_gauge {
            geometry.coefficient_gauge = gauge;
        }
    }
    Ok(())
}

fn insert_array1(values: &Array1<f64>, index: usize, value: f64) -> Array1<f64> {
    let mut out = Vec::<f64>::with_capacity(values.len() + 1);
    out.extend(values.iter().take(index).copied());
    out.push(value);
    out.extend(values.iter().skip(index).copied());
    Array1::from_vec(out)
}

fn insert_zero_row(matrix: &Array2<f64>, index: usize) -> Result<Array2<f64>, String> {
    if index > matrix.nrows() {
        return Err(format!(
            "extend_with_group factor insert index {index} exceeds its {} rows",
            matrix.nrows()
        ));
    }
    let mut out = Array2::<f64>::zeros((matrix.nrows() + 1, matrix.ncols()));
    for (old_i, row) in matrix.rows().into_iter().enumerate() {
        let new_i = if old_i < index { old_i } else { old_i + 1 };
        out.row_mut(new_i).assign(&row);
    }
    Ok(out)
}

fn insert_symmetric_array2(
    matrix: &Array2<f64>,
    index: usize,
    diagonal: f64,
) -> Result<Array2<f64>, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err(format!(
            "extend_with_group expected square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    if index > matrix.nrows() {
        return Err(format!(
            "extend_with_group matrix insert index {index} exceeds dimension {}",
            matrix.nrows()
        ));
    }
    let n = matrix.nrows();
    let mut out = Array2::<f64>::zeros((n + 1, n + 1));
    for old_i in 0..n {
        let new_i = if old_i < index { old_i } else { old_i + 1 };
        for old_j in 0..n {
            let new_j = if old_j < index { old_j } else { old_j + 1 };
            out[[new_i, new_j]] = matrix[[old_i, old_j]];
        }
    }
    out[[index, index]] = diagonal;
    Ok(out)
}

#[cfg(test)]
mod unseen_level_prior_scale_tests {
    use super::*;
    use gam_problem::types::{LikelihoodScaleMetadata, LikelihoodSpec, LogLikelihoodNormalization};
    use gam_solve::estimate::{FitArtifacts, FittedBlock, FittedLinkState};
    use gam_solve::pirls::PirlsStatus;
    use ndarray::array;

    fn saved_fit(
        likelihood_family: LikelihoodSpec,
        likelihood_scale: LikelihoodScaleMetadata,
        standard_deviation: f64,
        lambda: f64,
    ) -> UnifiedFitResult {
        let log_lambda = lambda.ln();
        let lambda = gam_problem::checked_exp_log_strength(log_lambda).expect("finite fixture strength");
        let blocks = vec![FittedBlock {
            beta: array![0.25, -0.5],
            role: BlockRole::Mean,
            edf: 1.5,
            lambdas: array![lambda],
        }];
        let lambdas = array![lambda];
        UnifiedFitResult::try_from_parts(gam_solve::estimate::UnifiedFitResultParts {
            blocks,
            training_sample_size: 16,
            log_lambdas: array![log_lambda],
            lambdas,
            likelihood_family: Some(likelihood_family),
            likelihood_scale,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: Some(0.0),
            stable_penalty_term: 0.0,
            penalized_objective: Some(0.0),
            used_device: false,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: None,
            standard_deviation,
            covariance_conditional: Some(Array2::zeros((2, 2))),
            covariance_corrected: Some(Array2::zeros((2, 2))),
            inference: None,
            fitted_link: FittedLinkState::Standard(None),
            geometry: None,
            block_states: vec![],
            pirls_status: PirlsStatus::Converged,
            max_abs_eta: 0.0,
            constraint_kkt: None,
            artifacts: FitArtifacts {
                pirls: None,
                null_space_logdet: None,
                null_space_dim: None,
                survival_link_wiggle_knots: None,
                survival_link_wiggle_degree: None,
                criterion_certificate: None,
                rho_posterior: Default::default(),
                rho_posterior_escalation: None,
                rho_covariance: None,
                joint_log_lambdas: None,
                firth_bias_reduction: false,
                covariance_declined: None,
                jeffreys_arming_evidence: None,
                improper_penalty_null_posterior: None,
                outer_warm_start: None,
                null_deviance: None,
                binomial_trial_counts: false,
                coefficient_mode_selection:
                    gam_solve::model_types::CoefficientModeSelection::NotRecorded,
                random_effect_tests: Vec::new(),
            },
            inner_cycles: 0,
        })
        .expect("test fixture fit must assemble")
    }

    /// The unseen-level prior is `scale / λ` in reported-covariance units and
    /// enters the unscaled Hessian as exactly `λ`, for a family whose working
    /// weight carries `1/φ` (Gamma: scale 1, although `dispersion_phi` is
    /// `1/shape`) and for the scale-free profiled Gaussian (scale `σ̂²`).
    #[test]
    fn unseen_level_prior_uses_the_coefficient_covariance_scale() {
        let lambda = 2.0;
        let gamma = saved_fit(
            LikelihoodSpec::gamma_log(),
            LikelihoodScaleMetadata::EstimatedGammaShape { shape: 4.0 },
            1.0,
            lambda,
        );
        assert_eq!(gamma.coefficient_covariance_scale().unwrap(), 1.0);
        assert_eq!(gamma.dispersion_phi().unwrap(), 0.25);
        let gaussian = saved_fit(
            LikelihoodSpec::gaussian_identity(),
            LikelihoodScaleMetadata::ProfiledGaussian,
            2.0,
            lambda,
        );
        assert_eq!(gaussian.coefficient_covariance_scale().unwrap(), 4.0);

        for (fit, expected_variance) in [(&gamma, 1.0 / lambda), (&gaussian, 4.0 / lambda)] {
            let variance = default_unseen_level_prior_variance(fit, 0, "g").unwrap();
            assert!(
                (variance - expected_variance).abs() <= 1e-15,
                "prior variance {variance} != scale/lambda {expected_variance}"
            );
            let precision = unscaled_prior_precision(fit, variance).unwrap();
            assert!(
                (precision - lambda).abs() <= 1e-15,
                "unscaled prior precision {precision} != lambda {lambda}"
            );
        }
    }
    #[test]
    fn unseen_level_prior_refuses_unrepresentable_variance_and_precision() {
        let gamma = saved_fit(
            LikelihoodSpec::gamma_log(),
            LikelihoodScaleMetadata::EstimatedGammaShape { shape: 4.0 },
            1.0,
            2.0,
        );
        let overflow = saved_fit(
            LikelihoodSpec::gaussian_identity(),
            LikelihoodScaleMetadata::ProfiledGaussian,
            1e100,
            1e-200,
        );
        assert!(default_unseen_level_prior_variance(&overflow, 0, "g").is_err());
        for variance in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::from_bits(1)] {
            assert!(unscaled_prior_precision(&gamma, variance).is_err(), "variance={variance}");
        }
        let gaussian = saved_fit(
            LikelihoodSpec::gaussian_identity(),
            LikelihoodScaleMetadata::ProfiledGaussian,
            1e-100,
            1e200,
        );
        assert!(default_unseen_level_prior_variance(&gaussian, 0, "g").is_err());
        assert!(unscaled_prior_precision(&gaussian, f64::MAX).is_err());
    }

    #[test]
    fn unseen_level_insertion_preserves_covariance_precision_scale() {
        for (family, scale, standard_deviation, variance) in [
            (LikelihoodSpec::gamma_log(), LikelihoodScaleMetadata::EstimatedGammaShape { shape: 4.0 }, 1.0, 0.5),
            (LikelihoodSpec::gaussian_identity(), LikelihoodScaleMetadata::ProfiledGaussian, 2.0, 2.0),
        ] {
            let mut fit = saved_fit(family, scale, standard_deviation, 2.0);
            let covariance_scale = fit.coefficient_covariance_scale().unwrap();
            fit.covariance_conditional = Some(Array2::eye(2) * covariance_scale);
            fit.covariance_corrected = Some(Array2::eye(2) * covariance_scale);
            fit.geometry = Some(gam_solve::model_types::FitGeometry {
                coefficient_gauge: gam_problem::gauge::Gauge::identity(&[2]),
                penalized_hessian: Array2::eye(2).into(),
                constrained_posterior: None,
                working: None,
            });
            insert_coefficient_into_saved_fit(Some(&mut fit), 2, 0.75, variance).unwrap();
            assert_eq!(fit.beta.to_vec(), vec![0.25, -0.5, 0.75]);
            for covariance in [&fit.covariance_conditional, &fit.covariance_corrected] {
                let covariance = covariance.as_ref().unwrap();
                assert_eq!(covariance.dim(), (3, 3));
                assert_eq!(covariance[[0, 0]], covariance_scale);
                assert_eq!(covariance[[2, 2]], variance);
                assert_eq!(covariance[[0, 2]], 0.0);
            }
            let geometry = fit.geometry.as_ref().unwrap();
            assert_eq!(geometry.coefficient_gauge.raw_total(), 3);
            assert_eq!(geometry.coefficient_gauge.reduced_total(), 3);
            assert_eq!(geometry.penalized_hessian.as_array()[[2, 2]], 2.0);
            assert_eq!(geometry.penalized_hessian.as_array()[[0, 0]], 1.0);
        }
    }

}

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
use gam_terms::smooth::TermCollectionSpec;
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
            random_effect_penalty_index(spec, term_idx),
        )
    };
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
    let coefficient_index = payload
        .fit_result
        .as_ref()
        .ok_or_else(|| "extend_with_group requires saved fit_result; refit".to_string())?
        .beta
        .len();
    let (coefficient_mean, supplied_variance) = extension_prior_parameters(prior.as_ref())?;
    let coefficient_variance = match supplied_variance {
        Some(variance) => variance,
        None => {
            let fit = payload
                .fit_result
                .as_ref()
                .ok_or_else(|| "extend_with_group requires saved fit_result; refit".to_string())?;
            let lambda = fit
                .lambdas
                .get(penalty_index)
                .copied()
                .filter(|lambda| lambda.is_finite() && *lambda > 0.0)
                .ok_or_else(|| {
                    format!(
                        "extend_with_group term '{term_name}' has no finite positive prior lambda"
                    )
                })?;
            // The unseen-level default prior is the fitted random-effect
            // variance component `σ_b² = φ̂ / λ` (mgcv's `λ = φ̂ / σ_b²`
            // convention), NOT the scale-free `1 / λ`. `φ̂` is the residual
            // dispersion that scales every predict-time covariance: `1` for
            // fixed-scale families (Poisson/Binomial — where `φ̂/λ` collapses
            // to the old `1/λ`), but `σ̂²` for Gaussian and the estimated
            // dispersion for Gamma/Tweedie/NB. Omitting `φ̂` made the prior
            // (and any deployment interval built from it) wrong by `1/φ̂` and,
            // for an estimated scale, not response-scale equivariant. See #674.
            let phi = fit
                .dispersion_phi()
                .map_err(|err| format!("cannot resolve unseen-level prior dispersion: {err}"))?;
            if !(phi.is_finite() && phi > 0.0) {
                return Err(format!(
                    "extend_with_group term '{term_name}' has a non-finite or non-positive \
                     dispersion (φ̂ = {phi}); cannot form the default prior variance"
                ));
            }
            phi / lambda
        }
    };
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
    insert_coefficient_into_saved_fit(
        payload.unified.as_mut(),
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

fn random_effect_penalty_index(spec: &TermCollectionSpec, term_idx: usize) -> usize {
    usize::from(spec.linear_terms.iter().any(|term| term.double_penalty)) + term_idx
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
    // precision is blockdiag(H_old, lambda_new S_new).  Therefore the
    // conditional covariance is blockdiag(V_old, S_new^{-1}/lambda_new).  The
    // current API extends one iid random-effect coordinate at a time, so
    // S_new = [1] and `variance` is exactly 1/lambda_new, or the caller's
    // supplied scalar prior covariance.
    if let Some(cov) = fit.covariance_conditional.as_mut() {
        *cov = insert_symmetric_array2(cov, index, variance)?;
    }
    if let Some(cov) = fit.covariance_corrected.as_mut() {
        *cov = insert_symmetric_array2(cov, index, variance)?;
    }
    let variance_diag = variance;
    let precision_diag = 1.0 / variance_diag;
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
        if let Some(cov) = inference.beta_covariance.as_mut() {
            // `beta_covariance` is the `PhiScaledCovariance` newtype.
            *cov = insert_symmetric_array2(cov.as_array(), index, variance_diag)?.into();
        }
        if let Some(se) = inference.beta_standard_errors.as_mut() {
            *se = insert_array1(se, index, variance_diag.sqrt());
        }
        if let Some(cov) = inference.beta_covariance_corrected.as_mut() {
            *cov = insert_symmetric_array2(cov, index, variance_diag)?;
        }
        if let Some(se) = inference.beta_standard_errors_corrected.as_mut() {
            *se = insert_array1(se, index, variance_diag.sqrt());
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
        if let Some(bias) = inference.bias_correction_beta.as_mut() {
            *bias = insert_array1(bias, index, 0.0);
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

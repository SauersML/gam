//! A prediction frame projected onto a fitted model's input contract, and the
//! model's affine predictor design at that frame.
//!
//! Every front door that predicts from an encoded table reads the model's columns
//! the same way: select what the model can consume, project it onto the training
//! schema, refuse unseen numeric factor levels, then build the design. The Python
//! bindings and the response-geometry model both read it here (#2899 P1).

use std::collections::BTreeSet;

use gam_data::{DataError, DataSchema, EncodedDataset, UnseenCategoryPolicy};
use gam_models::fit_orchestration::resolve_offset_column;
use gam_models::inference::model::FittedModel;
use ndarray::Array1;

use crate::input::build_predict_input_for_model;
use crate::{AffineDesign, affine_design_unavailable_reason};

/// Why a frame cannot be projected onto a model's input contract.
#[derive(Debug)]
pub enum ModelFrameError {
    /// The frame lacks a column the model needs, or a column's kind disagrees with
    /// the saved schema.
    SchemaMismatch(String),
    /// A cell cannot be predicted from: a non-finite covariate, a missing categorical
    /// label, or an unseen fixed-factor level.
    Input(DataError),
    Other(String),
}

impl std::fmt::Display for ModelFrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SchemaMismatch(message) | Self::Other(message) => f.write_str(message),
            Self::Input(error) => write!(f, "{error}"),
        }
    }
}

/// The columns a fitted model can legitimately consume from a prediction frame: every
/// column it requires (`FittedModel::prediction_required_columns`), plus the response
/// (the conformal calibration fold and label-bearing frames read it) and the
/// prior-weights column when the fit carried one (the replicate path reconstructs
/// `Var(y_i) = σ²/w_i` from it, #2025/#2033). Any other column (a row ID, a label kept
/// for bookkeeping) is irrelevant to the model and is never re-encoded against the
/// training schema, so a held-out fold carrying a new level in such a column still
/// predicts (#840).
pub fn prediction_consumable_columns(model: &FittedModel) -> Result<BTreeSet<String>, String> {
    let mut consumable = model.prediction_required_columns()?;
    if let Some(response) =
        gam_terms::inference::formula_dsl::formula_response_column(model.payload().formula.as_str())
    {
        consumable.insert(response);
    }
    if let Some(weight) = model.weight_column.as_deref() {
        consumable.insert(weight.to_string());
    }
    Ok(consumable)
}

/// `source` restricted to the columns the model consumes and projected onto its
/// training schema ([`gam_data::project_encoded_to_schema`]): labels map onto
/// training levels, a random-effect group's unseen or missing label takes the
/// unknown-level code, and a missing or non-finite numeric cell is refused naming its
/// column. A numeric-coded fixed `factor(g)` has no categorical schema, so its unseen
/// levels are refused by [`FittedModel::unseen_numeric_factor_levels`].
pub fn project_to_model_schema(
    model: &FittedModel,
    source: &EncodedDataset,
) -> Result<EncodedDataset, ModelFrameError> {
    let required = model
        .prediction_required_columns()
        .map_err(ModelFrameError::Other)?;
    let present = source.headers.iter().cloned().collect::<BTreeSet<_>>();
    let missing = required
        .difference(&present)
        .map(|name| format!("missing required column '{name}'"))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(ModelFrameError::SchemaMismatch(missing.join(" ")));
    }
    let consumable = prediction_consumable_columns(model).map_err(ModelFrameError::Other)?;
    let keep = source
        .headers
        .iter()
        .enumerate()
        .filter(|(_, name)| consumable.contains(name.as_str()))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let selected = EncodedDataset {
        headers: keep.iter().map(|&index| source.headers[index].clone()).collect(),
        values: source.values.select(ndarray::Axis(1), &keep),
        schema: DataSchema {
            columns: keep
                .iter()
                .map(|&index| {
                    source.schema.columns.get(index).cloned().ok_or_else(|| {
                        ModelFrameError::Other(format!(
                            "encoded table column '{}' has no source schema",
                            source.headers[index]
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        },
        column_kinds: keep.iter().map(|&index| source.column_kinds[index]).collect(),
    };
    let policy =
        UnseenCategoryPolicy::encode_unknown_for_columns(model.random_effect_group_columns());
    let schema = model
        .require_data_schema()
        .map_err(|error| ModelFrameError::Other(error.to_string()))?;
    let dataset = gam_data::project_encoded_to_schema(selected, schema, &policy).map_err(
        |error| match error {
            DataError::InvalidCell { .. } => ModelFrameError::Input(error),
            DataError::SchemaMismatch { reason } => ModelFrameError::SchemaMismatch(reason),
            other => ModelFrameError::Other(other.to_string()),
        },
    )?;
    if let Some(unseen) = model
        .unseen_numeric_factor_levels(&dataset.headers, dataset.values.view())
        .into_iter()
        .next()
    {
        return Err(ModelFrameError::Input(unseen));
    }
    Ok(dataset)
}

/// The model's affine predictor design at the rows of a frame already projected onto
/// its schema ([`project_to_model_schema`]). Which fitted models have no affine
/// coefficient frame, and the wording of that refusal, is decided before any row is
/// built ([`affine_design_unavailable_reason`]).
pub fn affine_design_for_model_frame(
    model: &FittedModel,
    dataset: &EncodedDataset,
) -> Result<AffineDesign, String> {
    if let Some(reason) = affine_design_unavailable_reason(model)? {
        return Err(reason);
    }
    let col_map = dataset.column_map();
    let offset = resolve_offset_column(dataset, &col_map, model.offset_column.as_deref())
        .map_err(|error| error.to_string())?;
    let offset_noise = Array1::zeros(dataset.values.nrows());
    let input = build_predict_input_for_model(
        model,
        dataset.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        &offset,
        &offset_noise,
        false,
    )?;
    crate::fitted_standard_affine_design(model, &input)
}

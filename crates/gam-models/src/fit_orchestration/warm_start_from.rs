//! Warm starts from a saved model (`warm_start_from`, gam#3002).
//!
//! A fit records its published search's certified outer point beside its result
//! (`FitArtifacts::outer_warm_start`): the outer coordinates, the inner mode
//! there, the criterion value certified there, and the fingerprint of the inputs.
//! A new fit resolves a saved model into a [`gam_model_api::WarmStart`] here, and
//! every outer search takes it through `OuterProblem::with_warm_start`: on the
//! parent's own inputs the search that certified the point accepts it where it
//! stands and every other search runs cold, and on other inputs the point can
//! only join an argmin-over-seeds multistart. No search reports another point
//! for its own criterion than it reports cold, except a multistart's argmin over
//! more seeds.

use crate::fit_orchestration::{FitConfig, WarmStartRefusal, WorkflowError};
use crate::inference::model::FittedModelPayload;
use gam_data::EncodedDataset;

/// The fingerprint of a fit's inputs: its formula, its data (headers, schema and
/// values) and every field of its request that defines the model, a frozen CTN's
/// fitted transform included. Two fits with one fingerprint optimize the same
/// criteria, which is what lets a certified point of the one resume the other.
/// Left out are the warm start itself, the persistence store, the container type
/// of the training table and the resource policy: none of them changes a
/// criterion. `None` when an input cannot be fingerprinted; such a fit's point
/// can only join a later search, never resume one.
pub fn fit_input_fingerprint(
    formula: &str,
    dataset: &EncodedDataset,
    config: &FitConfig,
) -> Option<String> {
    let mut request = config.clone();
    request.warm_start = None;
    request.persistent_warm_start_store = None;
    request.training_table_kind = String::new();
    request.resource_policy = None;
    let mut fingerprint = gam_runtime::warm_start::Fingerprinter::new();
    fingerprint.write_str(&formula.split_whitespace().collect::<String>());
    fingerprint.write_str(&format!("{:?}", dataset.headers));
    fingerprint.write_str(&format!("{:?}", dataset.schema));
    fingerprint.write_f64_array2(&dataset.values);
    fingerprint.write_str(&format!("{request:?}"));
    // A frozen CTN's `Debug` names only its formula; the transform it froze is an
    // input of every criterion the chain optimizes.
    if let Some(frozen) = config.frozen_ctn.as_ref() {
        fingerprint.write_str(&serde_json::to_string(&*frozen.0).ok()?);
    }
    Some(fingerprint.finish_hex())
}

/// The warm start a new fit of `formula` on `dataset` under `config` takes from
/// the saved `model`. Refused by name when the model was fitted with another
/// formula, was saved before points were recorded (refit it with this version), or
/// comes from a route that records none.
pub fn resolve_warm_start(
    model: &FittedModelPayload,
    formula: &str,
    dataset: &EncodedDataset,
    config: &FitConfig,
) -> Result<gam_model_api::WarmStart, WorkflowError> {
    // A model stores the formula it fitted, an automatic `.` term expanded, and
    // the fit records its inputs under that formula; so the new fit's formula is
    // expanded the same way before either is compared.
    let formula =
        crate::fit_orchestration::expand_automatic_fit_formula(formula, dataset, config)?.formula;
    let spelling = |text: &str| text.split_whitespace().collect::<String>();
    if spelling(&model.formula) != spelling(&formula) {
        return Err(WorkflowError::WarmStartRefused {
            refusal: WarmStartRefusal::FormulaDiffers {
                model: model.formula.clone(),
                fit: formula,
            },
        });
    }
    let record = model
        .fit_result
        .as_ref()
        .and_then(|fit| fit.artifacts.outer_warm_start.as_ref())
        .ok_or(WorkflowError::WarmStartRefused {
            refusal: if model.version
                <= crate::inference::model::OUTER_WARM_START_ABSENT_PAYLOAD_VERSION
            {
                WarmStartRefusal::RefitRequired {
                    payload_version: model.version,
                }
            } else {
                WarmStartRefusal::NoRecordedPoint
            },
        })?;
    // A v25 to v27 record carries neither its value nor its fingerprint, so it can only
    // join a search.
    let same_inputs = record.value.is_some()
        && record.input_fingerprint.is_some()
        && record.input_fingerprint == fit_input_fingerprint(&formula, dataset, config);
    Ok(gam_model_api::WarmStart {
        theta: ndarray::Array1::from_vec(record.theta.clone()),
        beta: ndarray::Array1::from_vec(record.beta.clone()),
        value: record.value.unwrap_or(f64::NAN),
        same_inputs,
        outcome: std::sync::Arc::new(std::sync::Mutex::new(None)),
    })
}

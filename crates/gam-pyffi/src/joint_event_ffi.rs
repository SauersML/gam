//! Python bindings for the joint event model: fit, save, load, condition and
//! forecast through the one Rust model path the CLI also calls. This layer only
//! moves arrays into the model's tables; every record is checked in Rust.

use crate::ffi::ffi_errors::{detach_py_result, py_value_error, saved_document_error_to_pyerr};
use gam::event_history::MarkKind;
use gam::event_history::joint::{self, EventTable, JointEventModel, JointTables, SubjectTable};
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use std::path::Path;
use std::sync::Arc;

/// A fitted or loaded joint event model.
#[pyclass(module = "gamfit._rust", name = "_JointEventModel", frozen)]
pub(crate) struct PyJointEventModel {
    model: Arc<JointEventModel>,
}

#[pymethods]
impl PyJointEventModel {
    fn mark_names(&self) -> Vec<String> {
        self.model.mark_names().to_vec()
    }

    fn mark_kinds(&self) -> Vec<String> {
        self.model
            .mark_kinds()
            .iter()
            .map(|k| k.name().to_string())
            .collect()
    }

    /// Save the model to `path`: the frozen encoding schema and the posterior,
    /// never the training records.
    fn save(&self, path: &str) -> PyResult<()> {
        self.model
            .save(Path::new(path))
            .map_err(saved_document_error_to_pyerr)
    }

    /// Condition on one history (entry, exit, and its events as parallel time
    /// and mark-label vectors) and forecast `horizons` after its exit.
    #[pyo3(signature = (entry, exit, event_time, event_marks, horizons))]
    fn forecast<'py>(
        &self,
        py: Python<'py>,
        entry: f64,
        exit: f64,
        event_time: Vec<f64>,
        event_marks: Vec<String>,
        horizons: Vec<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let history = "history".to_string();
        let tables = JointTables {
            subjects: SubjectTable {
                id: vec![history.clone()],
                entry: vec![entry],
                exit: vec![exit],
            },
            events: EventTable {
                id: vec![history; event_time.len()],
                time: event_time,
                mark: event_marks,
            },
            ..JointTables::default()
        };
        let model = Arc::clone(&self.model);
        let result = detach_py_result(py, "joint event forecast", move || {
            let conditioned = model.condition(&tables).map_err(|e| e.to_string())?;
            let history = conditioned.into_iter().next().ok_or_else(|| {
                "conditioning on one history returned no conditioned model".to_string()
            })?;
            history.forecast(&horizons).map_err(|e| e.to_string())
        })?;
        let out = PyDict::new(py);
        out.set_item("horizons", result.horizons)?;
        out.set_item("survival", result.survival)?;
        out.set_item("incidence", PyArray2::from_owned_array(py, result.incidence))?;
        out.set_item(
            "incidence_error",
            PyArray2::from_owned_array(py, result.incidence_error),
        )?;
        Ok(out)
    }
}

/// Fit the joint event model from the subjects and events tables as columns.
#[pyfunction]
#[pyo3(signature = (declared_marks, subject_ids, entry, exit, event_subject, event_time, event_marks))]
fn fit_joint_event_model(
    py: Python<'_>,
    declared_marks: Option<Vec<(String, String)>>,
    subject_ids: Vec<String>,
    entry: Vec<f64>,
    exit: Vec<f64>,
    event_subject: Vec<String>,
    event_time: Vec<f64>,
    event_marks: Vec<String>,
) -> PyResult<PyJointEventModel> {
    let marks = declared_marks
        .map(|pairs| {
            pairs
                .into_iter()
                .map(|(name, kind)| {
                    MarkKind::parse(&kind)
                        .map(|kind| (name, kind))
                        .map_err(|e| py_value_error(e.to_string()))
                })
                .collect::<PyResult<Vec<(String, MarkKind)>>>()
        })
        .transpose()?;
    let tables = JointTables {
        subjects: SubjectTable {
            id: subject_ids,
            entry,
            exit,
        },
        events: EventTable {
            id: event_subject,
            time: event_time,
            mark: event_marks,
        },
        ..JointTables::default()
    };
    let model = detach_py_result(py, "joint event fit", move || {
        joint::fit_joint_event_model(marks, &tables).map_err(|e| e.to_string())
    })?;
    Ok(PyJointEventModel {
        model: Arc::new(model),
    })
}

/// Load a saved joint event model.
#[pyfunction]
fn load_joint_event_model(path: &str) -> PyResult<PyJointEventModel> {
    let model =
        JointEventModel::load(Path::new(path)).map_err(saved_document_error_to_pyerr)?;
    Ok(PyJointEventModel {
        model: Arc::new(model),
    })
}

/// A joint event model from a saved document's bytes, as `gamfit.loads`
/// reads a document whose header names the kind `joint` (gam#3053).
#[pyfunction]
fn loads_joint_event_model(model_bytes: Vec<u8>) -> PyResult<PyJointEventModel> {
    let text = std::str::from_utf8(&model_bytes).map_err(|error| {
        saved_document_error_to_pyerr(gam_model_api::saved_model::SavedModelError::Malformed {
            reason: error.to_string(),
        })
    })?;
    let model = JointEventModel::from_saved_text(text).map_err(saved_document_error_to_pyerr)?;
    Ok(PyJointEventModel {
        model: Arc::new(model),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyJointEventModel>()?;
    module.add_function(wrap_pyfunction!(fit_joint_event_model, module)?)?;
    module.add_function(wrap_pyfunction!(load_joint_event_model, module)?)?;
    module.add_function(wrap_pyfunction!(loads_joint_event_model, module)?)?;
    Ok(())
}

//! Manifold parameter decomposition (#2951): the Python transport of the Rust
//! surface.
//!
//! `gam_sae::parameter_decomposition::surface` owns the request document, every
//! operation and the report. This module converts a `dict[str, ndarray]` to owned
//! `f64` arrays, runs the surface with the interpreter lock released, and hands the
//! report's JSON text and named arrays back. The CLI calls the same entry, so both
//! produce the same report bytes.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use gam::terms::sae::parameter_decomposition::surface::run_parameter_decomposition;
use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::py_value_error;

/// Run one `gam.mpd-request` document against its named input arrays and return
/// `(report_json, arrays)`, where `arrays` holds every array the report names.
///
/// The surface has no cancellation point, so an interrupt returns control to Python
/// and the abandoned worker finishes and drops on its own thread.
#[pyfunction]
fn parameter_decomposition_run<'py>(
    py: Python<'py>,
    request_json: String,
    tensors: &Bound<'py, PyDict>,
) -> PyResult<(String, Bound<'py, PyDict>)> {
    let mut inputs = BTreeMap::new();
    for (key, value) in tensors.iter() {
        let name: String = key.extract().map_err(|error| {
            py_value_error(format!(
                "parameter_decomposition_run: input array names must be str: {error}"
            ))
        })?;
        let array: PyReadonlyArrayDyn<'py, f64> = value.extract().map_err(|error| {
            py_value_error(format!(
                "parameter_decomposition_run: input array {name:?} must be a float64 ndarray: {error}"
            ))
        })?;
        inputs.insert(name, array.as_array().to_owned());
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let (report_json, arrays) = crate::run_sae_fit_interruptible(py, "gam-mpd", &cancel, move || {
        run_parameter_decomposition(&request_json, &inputs)
            .and_then(|output| output.report_json().map(|json| (json, output.arrays)))
    })?
    .map_err(|error| py_value_error(error.to_string()))?;
    let named = PyDict::new(py);
    for (name, array) in arrays {
        named.set_item(name, array.into_pyarray(py))?;
    }
    Ok((report_json, named))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(parameter_decomposition_run, module)?)?;
    Ok(())
}

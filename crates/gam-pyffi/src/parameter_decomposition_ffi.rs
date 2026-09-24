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

use gam::terms::sae::parameter_decomposition::minimal_support::{
    Fidelity, SupportEvaluation, SupportExecutor, minimal_support,
};
use gam::terms::sae::parameter_decomposition::support_fit::{Alternation, PieceExecutor, fit_supports_and_pieces};
use gam::terms::sae::parameter_decomposition::surface::run_parameter_decomposition;
use ndarray::{Array1, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
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

/// A Python executor `evaluate(keep) -> (divergence, removal_cost, restore_gain)`, with
/// `keep` a `P x C` bool array, `divergence` float64 `P`, and the two predictions float64
/// `P x C` (removing a kept piece; restoring a removed one).
struct PythonSupportExecutor<'py> {
    evaluate: Bound<'py, PyAny>,
    error: Option<PyErr>,
}

impl SupportExecutor for PythonSupportExecutor<'_> {
    fn evaluate(&mut self, keep: ArrayView2<'_, bool>) -> Result<SupportEvaluation, String> {
        let py = self.evaluate.py();
        let mut keep_error = |e: PyErr| {
            let message = e.to_string();
            self.error = Some(e);
            message
        };
        let result = self
            .evaluate
            .call1((keep.to_owned().into_pyarray(py),))
            .map_err(&mut keep_error)?;
        let (divergence, cost, gain): (PyReadonlyArray1<'_, f64>, PyReadonlyArray2<'_, f64>, PyReadonlyArray2<'_, f64>) =
            result.extract().map_err(&mut keep_error)?;
        Ok(SupportEvaluation {
            divergence: divergence.as_array().to_owned(),
            removal_cost: cost.as_array().to_owned(),
            restore_gain: gain.as_array().to_owned(),
        })
    }
}

/// The smallest per-position supports at fidelity `eps` that the round rule of
/// `parameter_decomposition::minimal_support` reaches, executed by the Python
/// `evaluate` at fidelity `eps` in the declared `form` (`"per_position"`: every
/// position; `"mean"`: the batch mean), with positions in causal sequences of length `sequence` (1 for
/// independent examples), from the full support or from `start` (a `P x C` bool array). Returns
/// `{"keep", "divergence", "rounds"}`; an exception raised by
/// `evaluate` propagates with its own type.
#[pyfunction]
#[pyo3(signature = (evaluate, positions, pieces, eps, form, sequence, start=None))]
fn parameter_decomposition_minimal_support<'py>(
    py: Python<'py>,
    evaluate: Bound<'py, PyAny>,
    positions: usize,
    pieces: usize,
    eps: f64,
    form: &str,
    sequence: usize,
    start: Option<numpy::PyReadonlyArray2<'py, bool>>,
) -> PyResult<Bound<'py, PyDict>> {
    let fidelity = match form {
        "per_position" => Fidelity::PerPosition(eps),
        "mean" => Fidelity::Mean(eps),
        other => return Err(py_value_error(format!("minimal_support: form must be \"per_position\" or \"mean\", got {other:?}"))),
    };
    let start = start.map(|s| s.as_array().to_owned());
    let mut executor = PythonSupportExecutor { evaluate, error: None };
    let result = minimal_support(&mut executor, positions, pieces, fidelity, sequence, start, None);
    let result = match result {
        Ok(result) => result,
        Err(error) => return Err(executor.error.take().unwrap_or_else(|| py_value_error(error.to_string()))),
    };
    let out = PyDict::new(py);
    out.set_item("keep", result.keep.into_pyarray(py))?;
    out.set_item("divergence", result.divergence.into_pyarray(py))?;
    let rounds = pyo3::types::PyList::empty(py);
    for round in &result.rounds {
        let entry = PyDict::new(py);
        entry.set_item("proposed", round.proposed)?;
        entry.set_item("removed", round.removed)?;
        entry.set_item("halvings", round.halvings)?;
        entry.set_item("kept", round.kept)?;
        entry.set_item("max_divergence", round.max_divergence)?;
        rounds.append(entry)?;
    }
    out.set_item("rounds", rounds)?;
    Ok(out)
}

/// A Python executor object for `support_fit`: methods `supports(theta, keep)`,
/// `divergence(theta, keep)`, `weighted_gradient(theta, keep, weights)`,
/// `directional(theta, keep, v)`, `weighted_hessian(theta, keep, weights, v)` (the
/// exact Hessian product `Σ_t w_t ∇²KL_t v`), `gradient_arithmetic()` (the unit
/// roundoff of its gradients and the length of their longest reduction) and
/// `observe(alternation)` (a dict per completed alternation, for progress),
/// with float64 vectors and a `P x C` bool `keep`.
struct PythonPieceExecutor<'py> {
    object: Bound<'py, PyAny>,
    error: Option<PyErr>,
}

impl<'py> PythonPieceExecutor<'py> {
    fn vector<A: pyo3::call::PyCallArgs<'py>>(&mut self, method: &str, args: A) -> Result<Array1<f64>, String> {
        let result = self.object.call_method1(method, args).and_then(|r| {
            let array: PyReadonlyArray1<'_, f64> = r.extract()?;
            Ok(array.as_array().to_owned())
        });
        result.map_err(|e| {
            let message = e.to_string();
            self.error = Some(e);
            message
        })
    }
}

fn alternation_dict<'py>(py: Python<'py>, a: &Alternation) -> PyResult<Bound<'py, PyDict>> {
    let entry = PyDict::new(py);
    entry.set_item("level", a.level)?;
    entry.set_item("kept", a.kept)?;
    entry.set_item("mean_divergence", a.mean_divergence)?;
    entry.set_item("barrier_before", a.barrier_before)?;
    entry.set_item("barrier_after", a.barrier_after)?;
    entry.set_item("iterations", a.iterations)?;
    entry.set_item("certified", a.certified)?;
    entry.set_item("searched", a.searched)?;
    entry.set_item("residual", a.residual)?;
    entry.set_item("tolerance", a.tolerance)?;
    entry.set_item("radius", a.radius)?;
    entry.set_item("step", a.step)?;
    Ok(entry)
}

impl PieceExecutor for PythonPieceExecutor<'_> {
    fn observe(&mut self, alternation: &Alternation) {
        let py = self.object.py();
        let result = alternation_dict(py, alternation).and_then(|d| self.object.call_method1("observe", (d,)).map(|r| r.unbind()));
        if let Err(e) = result
            && self.error.is_none()
        {
            // Raised when the fit returns, never swallowed.
            self.error = Some(e);
        }
    }

    fn gradient_arithmetic(&self) -> Result<(f64, usize), String> {
        // Declared by the executor; there is no default.
        self.object
            .call_method0("gradient_arithmetic")
            .and_then(|r| r.extract::<(f64, usize)>())
            .map_err(|e| e.to_string())
    }

    fn supports(&mut self, theta: ArrayView1<'_, f64>, keep: ArrayView2<'_, bool>) -> Result<SupportEvaluation, String> {
        let py = self.object.py();
        let result = self
            .object
            .call_method1("supports", (theta.to_owned().into_pyarray(py), keep.to_owned().into_pyarray(py)))
            .and_then(|r| {
                let (d, c, g): (PyReadonlyArray1<'_, f64>, PyReadonlyArray2<'_, f64>, PyReadonlyArray2<'_, f64>) = r.extract()?;
                Ok(SupportEvaluation {
                    divergence: d.as_array().to_owned(),
                    removal_cost: c.as_array().to_owned(),
                    restore_gain: g.as_array().to_owned(),
                })
            });
        result.map_err(|e| {
            let message = e.to_string();
            self.error = Some(e);
            message
        })
    }
    fn divergence(&mut self, theta: ArrayView1<'_, f64>, keep: ArrayView2<'_, bool>) -> Result<Array1<f64>, String> {
        let py = self.object.py();
        self.vector("divergence", (theta.to_owned().into_pyarray(py), keep.to_owned().into_pyarray(py)))
    }
    fn weighted_gradient(&mut self, theta: ArrayView1<'_, f64>, keep: ArrayView2<'_, bool>, weights: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        let py = self.object.py();
        self.vector(
            "weighted_gradient",
            (theta.to_owned().into_pyarray(py), keep.to_owned().into_pyarray(py), weights.to_owned().into_pyarray(py)),
        )
    }
    fn directional(&mut self, theta: ArrayView1<'_, f64>, keep: ArrayView2<'_, bool>, v: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        let py = self.object.py();
        self.vector(
            "directional",
            (theta.to_owned().into_pyarray(py), keep.to_owned().into_pyarray(py), v.to_owned().into_pyarray(py)),
        )
    }
    fn weighted_hessian(&mut self, theta: ArrayView1<'_, f64>, keep: ArrayView2<'_, bool>, weights: ArrayView1<'_, f64>, v: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        let py = self.object.py();
        self.vector(
            "weighted_hessian",
            (
                theta.to_owned().into_pyarray(py),
                keep.to_owned().into_pyarray(py),
                weights.to_owned().into_pyarray(py),
                v.to_owned().into_pyarray(py),
            ),
        )
    }
}

/// `support_fit::fit_supports_and_pieces` executed by the Python `executor` object from
/// pieces `theta`. Returns `{"theta", "keep", "divergence", "alternations"}`.
#[pyfunction]
fn parameter_decomposition_fit_supports<'py>(
    py: Python<'py>,
    executor: Bound<'py, PyAny>,
    theta: PyReadonlyArray1<'py, f64>,
    positions: usize,
    pieces: usize,
    eps: f64,
    form: &str,
    sequence: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let mean = match form {
        "per_position" => false,
        "mean" => true,
        other => return Err(py_value_error(format!("fit_supports: form must be \"per_position\" or \"mean\", got {other:?}"))),
    };
    let theta = theta.as_array().to_owned();
    let mut runner = PythonPieceExecutor { object: executor, error: None };
    let fit = match fit_supports_and_pieces(&mut runner, theta, positions, pieces, eps, mean, sequence) {
        Ok(fit) => fit,
        Err(error) => return Err(runner.error.take().unwrap_or_else(|| py_value_error(error.to_string()))),
    };
    if let Some(error) = runner.error.take() {
        return Err(error);
    }
    let out = PyDict::new(py);
    out.set_item("theta", fit.theta.into_pyarray(py))?;
    out.set_item("keep", fit.supports.keep.into_pyarray(py))?;
    out.set_item("divergence", fit.supports.divergence.into_pyarray(py))?;
    let alternations = pyo3::types::PyList::empty(py);
    for a in &fit.alternations {
        alternations.append(alternation_dict(py, a)?)?;
    }
    out.set_item("alternations", alternations)?;
    Ok(out)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(parameter_decomposition_fit_supports, module)?)?;
    module.add_function(wrap_pyfunction!(parameter_decomposition_run, module)?)?;
    module.add_function(wrap_pyfunction!(parameter_decomposition_minimal_support, module)?)?;
    Ok(())
}

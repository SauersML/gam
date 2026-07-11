/// Thin Python marshalling for the typed manifold-crosscoder entry (#2231 Inc D).
/// Validation, defaults, seeding, fitting, diagnostics, and the wire schema all
/// live in `gam-sae`.

#[pyclass(module = "gamfit._rust", name = "ManifoldCrosscoderCore")]
struct ManifoldCrosscoderCore {
    inner: gam::terms::sae::manifold::SaeCrosscoderFitReport,
    wire: gam::terms::sae::manifold::SaeCrosscoderWireReport,
}

impl ManifoldCrosscoderCore {
    fn wire_value(&self) -> PyResult<serde_json::Value> {
        serde_json::to_value(&self.wire).map_err(|error| py_value_error(error.to_string()))
    }
}

#[pymethods]
impl ManifoldCrosscoderCore {
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_value_to_py(py, self.wire_value()?)
    }

    fn to_json(&self) -> PyResult<String> {
        // Serialize the struct directly — building the intermediate
        // `serde_json::Value` tree first doubled the serialization cost.
        serde_json::to_string(&self.wire).map_err(|error| py_value_error(error.to_string()))
    }

    /// Honest-unit `(n, p_layer)` reconstruction of one fitted layer as a numpy
    /// array (single copy). Reconstructions are deliberately absent from
    /// `to_dict`/`to_json` — as nested JSON they cost ~32 bytes per number.
    fn layer_fitted<'py>(
        &self,
        py: Python<'py>,
        layer: &str,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fit = self
            .inner
            .layers
            .iter()
            .find(|candidate| candidate.label == layer)
            .ok_or_else(|| {
                py_value_error(format!("crosscoder layer {layer:?} is not fitted"))
            })?;
        Ok(fit.fitted.clone().into_pyarray(py))
    }

    fn steer_layer_delta<'py>(
        &self,
        py: Python<'py>,
        atom: usize,
        layer: &str,
        rows: Vec<usize>,
        delta: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let values = self
            .inner
            .steer_layer_delta(atom, layer, &rows, delta.as_array())
            .map_err(py_value_error)?;
        Ok(values.into_pyarray(py))
    }

    fn steer_layer_decode<'py>(
        &self,
        py: Python<'py>,
        atom: usize,
        layer: &str,
        rows: Vec<usize>,
        delta: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let values = self
            .inner
            .steer_layer_decode(atom, layer, &rows, delta.as_array())
            .map_err(py_value_error)?;
        Ok(values.into_pyarray(py))
    }

    /// The intrinsic collateral-damage curve (gam#2234 E2) for steering `atom`
    /// along `axis` over `doses`, measured against every other fitted atom — the
    /// on-manifold vs matched-norm-flat comparison rendered as the wire dict the
    /// library owns (per-dose effect/collateral/cross-feature plus each arm's
    /// efficiency and the dominance verdict). No model in the loop.
    fn collateral_curve(
        &self,
        py: Python<'_>,
        atom: usize,
        axis: usize,
        doses: Vec<f64>,
    ) -> PyResult<PyObject> {
        let curve = self
            .inner
            .collateral_curve(atom, axis, &doses)
            .map_err(py_value_error)?;
        let value = serde_json::to_value(&curve).map_err(|error| py_value_error(error.to_string()))?;
        json_value_to_py(py, value)
    }
}

#[pyfunction(signature = (
    anchor,
    anchor_label,
    target_labels,
    targets,
    n_atoms,
    n_harmonics,
    sparsity_strength = None,
    smoothness = None,
    max_iter = None,
    learning_rate = None,
    ridge_ext_coord = None,
    ridge_beta = None,
    random_state = None,
    run_outer_rho_search = None,
    transport_grid_resolution = None,
    law_gap_tolerance = None,
))]
fn sae_crosscoder_fit<'py>(
    py: Python<'py>,
    anchor: PyReadonlyArray2<'py, f64>,
    anchor_label: String,
    target_labels: Vec<String>,
    targets: Vec<PyReadonlyArray2<'py, f64>>,
    n_atoms: usize,
    n_harmonics: usize,
    sparsity_strength: Option<f64>,
    smoothness: Option<f64>,
    max_iter: Option<usize>,
    learning_rate: Option<f64>,
    ridge_ext_coord: Option<f64>,
    ridge_beta: Option<f64>,
    random_state: Option<u64>,
    run_outer_rho_search: Option<bool>,
    transport_grid_resolution: Option<usize>,
    law_gap_tolerance: Option<f64>,
) -> PyResult<Py<ManifoldCrosscoderCore>> {
    use gam::terms::sae::manifold::{
        SaeCrosscoderAutoFitOverrides, SaeCrosscoderAutoFitRequest, SaeCrosscoderEvaluationConfig,
        pair_crosscoder_targets, run_auto_sae_crosscoder_fit,
    };

    // Boundary-only ownership transfer. Parallel-vector consistency and every
    // semantic check are performed by the typed library request.
    let blocks = pair_crosscoder_targets(
        target_labels,
        targets
            .into_iter()
            .map(|target| target.as_array().to_owned())
            .collect(),
    )
    .map_err(py_value_error)?;
    let config = SaeCrosscoderAutoFitOverrides {
        sparsity_strength,
        smoothness,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
        random_state,
        run_outer_rho_search,
    }
    .resolve(n_atoms, n_harmonics);
    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let request = SaeCrosscoderAutoFitRequest {
        anchor_label,
        anchor: anchor.as_array().to_owned(),
        blocks,
        config,
        cancel: Some(std::sync::Arc::clone(&cancel)),
    };
    let inner = run_sae_fit_interruptible(py, "gam-sae-crosscoder-fit", &cancel, move || {
        run_auto_sae_crosscoder_fit(request)
    })?
    .map_err(|error| sae_fit_error_to_pyerr(py, error))?;
    let wire = inner
        .wire_report(SaeCrosscoderEvaluationConfig {
            transport_grid_resolution,
            law_gap_tolerance,
        })
        .map_err(py_value_error)?;
    Py::new(
        py,
        ManifoldCrosscoderCore {
            inner,
            wire,
        },
    )
}

/// Rust-owned behavior-anchored fit (#2015). The wire report is materialized by
/// `gam-sae`; this class only re-serializes it.
#[pyclass(module = "gamfit._rust", name = "ManifoldBehaviorCore")]
struct ManifoldBehaviorCore {
    wire: gam::terms::sae::manifold::SaeBehaviorWireReport,
}

impl ManifoldBehaviorCore {
    fn wire_value(&self) -> PyResult<serde_json::Value> {
        serde_json::to_value(&self.wire).map_err(|error| py_value_error(error.to_string()))
    }
}

#[pymethods]
impl ManifoldBehaviorCore {
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_value_to_py(py, self.wire_value()?)
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.wire_value()?)
            .map_err(|error| py_value_error(error.to_string()))
    }
}

#[pyfunction(signature = (
    activation,
    probabilities,
    n_atoms,
    n_harmonics,
    sparsity_strength = None,
    smoothness = None,
    max_iter = None,
    learning_rate = None,
    ridge_ext_coord = None,
    ridge_beta = None,
    random_state = None,
    run_outer_rho_search = None,
))]
fn sae_behavior_fit<'py>(
    py: Python<'py>,
    activation: PyReadonlyArray2<'py, f64>,
    probabilities: PyReadonlyArray2<'py, f64>,
    n_atoms: usize,
    n_harmonics: usize,
    sparsity_strength: Option<f64>,
    smoothness: Option<f64>,
    max_iter: Option<usize>,
    learning_rate: Option<f64>,
    ridge_ext_coord: Option<f64>,
    ridge_beta: Option<f64>,
    random_state: Option<u64>,
    run_outer_rho_search: Option<bool>,
) -> PyResult<Py<ManifoldBehaviorCore>> {
    use gam::terms::sae::manifold::{
        SaeBehaviorAutoFitRequest, SaeCrosscoderAutoFitOverrides, run_auto_sae_behavior_fit,
    };

    let config = SaeCrosscoderAutoFitOverrides {
        sparsity_strength,
        smoothness,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
        random_state,
        run_outer_rho_search,
    }
    .resolve(n_atoms, n_harmonics);
    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let request = SaeBehaviorAutoFitRequest {
        activation: activation.as_array().to_owned(),
        probabilities: probabilities.as_array().to_owned(),
        config,
        cancel: Some(std::sync::Arc::clone(&cancel)),
    };
    let inner = run_sae_fit_interruptible(py, "gam-sae-behavior-fit", &cancel, move || {
        run_auto_sae_behavior_fit(request)
    })?
    .map_err(|error| sae_fit_error_to_pyerr(py, error))?;
    let wire = inner.wire_report().map_err(py_value_error)?;
    Py::new(py, ManifoldBehaviorCore { wire })
}

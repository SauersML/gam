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
        serde_json::to_string(&self.wire_value()?)
            .map_err(|error| py_value_error(error.to_string()))
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

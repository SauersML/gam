//! Public fitted-model surface for overcomplete hard-TopK manifold SAEs.
//!
//! This is a distinct representation of the same public model family: routing
//! is canonical support state (`N×s` indices plus heterogeneous coordinates),
//! never dense logits or gates. The dense `K<=P` artifact remains its
//! full-support specialization; crossing `K>P` changes representation before
//! any seed allocation.

use gam::terms::sae::manifold::{
    SaeSupportFixedPointReport, SaeSupportRehydrateRequest, SaeSupportSparseFit,
    SaeSupportSparseFitRequest, SaeSupportSparseTerm, SaeSupportStationarity,
    fit_sae_support_sparse_with_census, rehydrate_sae_support_term,
};
use ndarray::{Array1, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::{PyObject, json_value_to_py, py_value_error};

pub(crate) struct SupportSparseFitRequest<'a> {
    pub target: ArrayView2<'a, f64>,
    pub atom_basis: Vec<String>,
    pub atom_dim: Vec<usize>,
    pub support_k: usize,
    pub initial_smoothness: f64,
    pub max_iter: usize,
    pub trust_radius: f64,
    pub random_state: u64,
}

/// The on-disk tag for this representation. Checked exactly, never sniffed:
/// the dense artifact's tag also begins `gamfit.ManifoldSAE`, so a substring
/// test routes dense payloads here and support payloads there (#2567).
pub(crate) const SUPPORT_SCHEMA_TAG: &str = "gamfit.ManifoldSAE/support-v2";

/// The criterion every support fit reports. The representation fixes it, so a
/// payload carries the tag for its readers and loading re-derives it.
const SUPPORT_CRITERION_KIND: gam::terms::sae::front_door::SaeCriterionKind =
    gam::terms::sae::front_door::SaeCriterionKind::SupportQuasiLaplace;

fn required_field<'py>(
    payload: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Bound<'py, PyAny>> {
    payload.get_item(key)?.ok_or_else(|| {
        py_value_error(format!(
            "ManifoldSAESupport.from_dict: payload is missing required field {key:?}"
        ))
    })
}

/// Extract a 2-D f64 payload field from either a numpy array (the pickle
/// round-trip) or nested lists (the JSON round-trip). One loader, two carriers
/// -- the values are the same numbers either way, so the type must not decide
/// which files can be reopened (#2567).
fn extract_f64_matrix(value: &Bound<'_, PyAny>, what: &str) -> PyResult<Array2<f64>> {
    if let Ok(array) = value.extract::<PyReadonlyArray2<'_, f64>>() {
        return Ok(array.as_array().to_owned());
    }
    let rows: Vec<Vec<f64>> = value.extract().map_err(|_| {
        py_value_error(format!(
            "ManifoldSAESupport.from_dict: {what} is neither a 2-D float array nor nested lists"
        ))
    })?;
    let width = rows.first().map(Vec::len).unwrap_or(0);
    if rows.iter().any(|row| row.len() != width) {
        return Err(py_value_error(format!(
            "ManifoldSAESupport.from_dict: {what} rows have unequal lengths"
        )));
    }
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    Array2::from_shape_vec((rows.len(), width), flat)
        .map_err(|error| py_value_error(format!("ManifoldSAESupport.from_dict: {what}: {error}")))
}

/// The u32 counterpart of [`extract_f64_matrix`] for the support index grid.
fn extract_u32_matrix(value: &Bound<'_, PyAny>, what: &str) -> PyResult<Array2<u32>> {
    if let Ok(array) = value.extract::<PyReadonlyArray2<'_, u32>>() {
        return Ok(array.as_array().to_owned());
    }
    let rows: Vec<Vec<u32>> = value.extract().map_err(|_| {
        py_value_error(format!(
            "ManifoldSAESupport.from_dict: {what} is neither a 2-D u32 array nor nested lists"
        ))
    })?;
    let width = rows.first().map(Vec::len).unwrap_or(0);
    if rows.iter().any(|row| row.len() != width) {
        return Err(py_value_error(format!(
            "ManifoldSAESupport.from_dict: {what} rows have unequal lengths"
        )));
    }
    let flat: Vec<u32> = rows.iter().flatten().copied().collect();
    Array2::from_shape_vec((rows.len(), width), flat)
        .map_err(|error| py_value_error(format!("ManifoldSAESupport.from_dict: {what}: {error}")))
}

/// 1-D f64 field: numpy array or plain list.
fn extract_f64_vector(value: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<f64>> {
    if let Ok(array) = value.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(array.as_array().to_vec());
    }
    value.extract::<Vec<f64>>().map_err(|_| {
        py_value_error(format!(
            "ManifoldSAESupport.from_dict: {what} is neither a 1-D float array nor a list"
        ))
    })
}

/// JSON refuses non-finite numbers, so the save path names the field instead
/// of silently emitting null; every scalar this guards is certified finite by
/// the fit that produced it, so a trip here is a real defect upstream.
fn require_finite_for_json(name: &str, value: f64) -> PyResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(py_value_error(format!(
            "ManifoldSAESupport.save: {name} is {value}, which JSON cannot carry; \
             refusing to write a payload that could not be reloaded"
        )))
    }
}

fn matrix_json(matrix: &Array2<f64>) -> serde_json::Value {
    serde_json::Value::Array(
        matrix
            .rows()
            .into_iter()
            .map(|row| serde_json::Value::Array(row.iter().map(|&v| serde_json::json!(v)).collect()))
            .collect(),
    )
}

#[pyclass(module = "gamfit._rust", name = "ManifoldSAESupport", frozen)]
pub(crate) struct SupportSparseManifoldSaeCore {
    term: SaeSupportSparseTerm,
    requested_k: usize,
    retained_atom_indices: Vec<usize>,
    atom_basis: Vec<String>,
    atom_dim: Vec<usize>,
    atom_topologies: Vec<String>,
    support_k: usize,
    training_mean: Vec<f64>,
    fitted: Array2<f64>,
    reconstruction_r2: f64,
    log_lambda_smooth: Vec<f64>,
    ard_precisions: Vec<Vec<f64>>,
    criterion: f64,
    certificates: serde_json::Value,
    termination: serde_json::Value,
    trust_radius: f64,
    random_state: u64,
}

fn add_mean(mut fitted: Array2<f64>, mean: &[f64]) -> Array2<f64> {
    for mut row in fitted.rows_mut() {
        for (value, location) in row.iter_mut().zip(mean) {
            *value += location;
        }
    }
    fitted
}

fn centered(target: ArrayView2<'_, f64>, mean: &[f64]) -> Array2<f64> {
    Array2::from_shape_fn(target.dim(), |(row, column)| {
        target[[row, column]] - mean[column]
    })
}

fn support_indices(term: &SaeSupportSparseTerm) -> Result<Array2<u32>, String> {
    let rows = term.n_obs();
    let support = term.assignment.support_indices(0).len();
    let values = (0..rows)
        .flat_map(|row| term.assignment.support_indices(row).iter().copied())
        .collect::<Vec<_>>();
    Array2::from_shape_vec((rows, support), values)
        .map_err(|error| format!("support index shape: {error}"))
}

fn support_values(term: &SaeSupportSparseTerm) -> Array2<f64> {
    Array2::ones((term.n_obs(), term.assignment.support_indices(0).len()))
}

fn coords_rows<'py>(py: Python<'py>, term: &SaeSupportSparseTerm) -> PyResult<Bound<'py, PyList>> {
    let rows = PyList::empty(py);
    for row in 0..term.n_obs() {
        rows.append(Array1::from(term.assignment.coords_row(row).to_vec()).into_pyarray(py))?;
    }
    Ok(rows)
}

/// The inner fixed point's certificate, with the tolerance it certified to: the
/// engine derives that tolerance from the term's own objective resolution, so the
/// numbers above it are readable only beside it.
fn fixed_point_json(report: &SaeSupportFixedPointReport, tolerance: f64) -> serde_json::Value {
    let SaeSupportStationarity {
        decoder_l2,
        decoder_max_abs,
        coordinate_l2,
        coordinate_max_abs,
        decoder_scaled_max_abs,
        coordinate_scaled_max_abs,
    } = report.stationarity;
    serde_json::json!({
        "iterations": report.iterations,
        "objective": report.objective,
        "decoder_l2": decoder_l2,
        "decoder_max_abs": decoder_max_abs,
        "coordinate_l2": coordinate_l2,
        "coordinate_max_abs": coordinate_max_abs,
        // #2517: the gradient-space numbers above scale with rows-per-atom, so
        // they cannot be read as a distance to the fixed point. The scaled pair
        // is the diagonal-preconditioned residual that schedules the certificate.
        "decoder_scaled_max_abs": decoder_scaled_max_abs,
        "coordinate_scaled_max_abs": coordinate_scaled_max_abs,
        // #2933 F08: the certificate itself, the exact Newton displacement `A⁻¹g`
        // per block, in parameter units, and its squared decrement `gᵀA⁻¹g`.
        "newton_displacement_decoder_max_abs": report.newton_displacement.decoder_max_abs,
        "newton_displacement_coordinate_max_abs": report.newton_displacement.coordinate_max_abs,
        "newton_decrement_sq": report.newton_displacement.decrement_sq,
        "max_recurrence_change": report.max_recurrence_change,
        "recurred": report.recurred,
        "tolerance": tolerance,
    })
}

impl SupportSparseManifoldSaeCore {
    fn infer(
        &self,
        target: ArrayView2<'_, f64>,
    ) -> Result<(SaeSupportSparseTerm, Array2<f64>, serde_json::Value), String> {
        if target.ncols() != self.training_mean.len() || target.nrows() == 0 {
            return Err(format!(
                "ManifoldSAE.converged_latents requires positive rows and P={}; got {:?}",
                self.training_mean.len(),
                target.dim()
            ));
        }
        if target.iter().any(|value| !value.is_finite()) {
            return Err("ManifoldSAE.converged_latents target contains a non-finite value".into());
        }
        let centered_target = centered(target, &self.training_mean);
        let mut term = self.term.reroute_fixed_decoder(
            centered_target.view(),
            self.support_k,
            self.random_state,
        )?;
        // The frozen-decoder objective sums these rows' cells, so the tolerance is
        // derived from this term, not copied from the training fit's.
        let tolerance = term.fixed_point_tolerance();
        // An out-of-sample solve has no outer search: it sweeps until its certificate
        // holds, and refuses once a sweep moves no coordinate short of it.
        let report = term.solve_coordinates_fixed_decoder(
            centered_target.view(),
            &self.ard_precisions,
            tolerance,
            self.trust_radius,
        )?;
        let fitted = add_mean(term.reconstruct()?, &self.training_mean);
        let certificate = serde_json::json!({
            "iterations": report.iterations,
            "objective": report.objective,
            "coordinate_l2": report.coordinate_l2,
            "coordinate_max_abs": report.coordinate_max_abs,
            "max_recurrence_change": report.max_recurrence_change,
            "recurred": report.recurred,
            "tolerance": tolerance,
        });
        Ok((term, fitted, certificate))
    }

    fn latents_dict<'py>(
        &self,
        py: Python<'py>,
        term: &SaeSupportSparseTerm,
        fitted: Array2<f64>,
        certificate: serde_json::Value,
    ) -> PyResult<Py<PyDict>> {
        let out = PyDict::new(py);
        out.set_item("fitted", fitted.into_pyarray(py))?;
        out.set_item(
            "support_indices",
            support_indices(term)
                .map_err(py_value_error)?
                .into_pyarray(py),
        )?;
        out.set_item("support_values", support_values(term).into_pyarray(py))?;
        out.set_item("coords", coords_rows(py, term)?)?;
        out.set_item("certificate", json_value_to_py(py, &certificate)?)?;
        Ok(out.unbind())
    }
}

#[pymethods]
impl SupportSparseManifoldSaeCore {
    #[pyo3(signature = (x_new=None))]
    fn converged_latents<'py>(
        &self,
        py: Python<'py>,
        x_new: Option<PyReadonlyArray2<'py, f64>>,
    ) -> PyResult<Py<PyDict>> {
        match x_new {
            Some(values) => {
                let (term, fitted, certificate) =
                    self.infer(values.as_array()).map_err(py_value_error)?;
                self.latents_dict(py, &term, fitted, certificate)
            }
            None => self.latents_dict(
                py,
                &self.term,
                self.fitted.clone(),
                self.certificates["inner_fixed_point"].clone(),
            ),
        }
    }

    fn reconstruct<'py>(
        &self,
        py: Python<'py>,
        x_new: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (_, fitted, _) = self.infer(x_new.as_array()).map_err(py_value_error)?;
        Ok(fitted.into_pyarray(py))
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x_new: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.reconstruct(py, x_new)
    }

    fn encode<'py>(
        &self,
        py: Python<'py>,
        x_new: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Py<PyDict>> {
        let (term, _, certificate) = self.infer(x_new.as_array()).map_err(py_value_error)?;
        let out = PyDict::new(py);
        out.set_item(
            "indices",
            support_indices(&term)
                .map_err(py_value_error)?
                .into_pyarray(py),
        )?;
        out.set_item("values", support_values(&term).into_pyarray(py))?;
        out.set_item("coords", coords_rows(py, &term)?)?;
        out.set_item("certificate", json_value_to_py(py, &certificate)?)?;
        Ok(out.unbind())
    }

    /// Sample one atom's decoded curve at coordinates `(n, d)` → `(n, P)`.
    fn atom_curve<'py>(
        &self,
        py: Python<'py>,
        atom_k: usize,
        coords: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let curve = self
            .term
            .decode_atom_at(atom_k, coords.as_array())
            .map_err(py_value_error)?;
        Ok(curve.into_pyarray(py))
    }

    /// On-manifold steering delta along one atom:
    /// `amplitude · (γ_k(t_to) − γ_k(t_from))`, decoded by the atom's own
    /// evaluator. Returns the ambient-chart delta plus both decoded images.
    fn steer<'py>(
        &self,
        py: Python<'py>,
        atom_k: usize,
        amplitude: f64,
        t_from: PyReadonlyArray1<'py, f64>,
        t_to: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<PyDict>> {
        if !amplitude.is_finite() {
            return Err(py_value_error(format!(
                "ManifoldSAE.steer: amplitude must be finite; got {amplitude}"
            )));
        }
        let from = t_from.as_array();
        let to = t_to.as_array();
        let width = from.len();
        if to.len() != width {
            return Err(py_value_error(format!(
                "ManifoldSAE.steer: t_from width {} != t_to width {}",
                width,
                to.len()
            )));
        }
        let stack = |values: ndarray::ArrayView1<'_, f64>| {
            ndarray::Array2::from_shape_vec((1, width), values.to_vec())
                .map_err(|error| py_value_error(format!("ManifoldSAE.steer: {error}")))
        };
        let g_from = self
            .term
            .decode_atom_at(atom_k, stack(from)?.view())
            .map_err(py_value_error)?;
        let g_to = self
            .term
            .decode_atom_at(atom_k, stack(to)?.view())
            .map_err(py_value_error)?;
        let delta = (&g_to - &g_from).row(0).mapv(|value| amplitude * value);
        let out = PyDict::new(py);
        out.set_item("delta", delta.into_pyarray(py))?;
        out.set_item("decoded_from", g_from.row(0).to_owned().into_pyarray(py))?;
        out.set_item("decoded_to", g_to.row(0).to_owned().into_pyarray(py))?;
        Ok(out.unbind())
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("schema", SUPPORT_SCHEMA_TAG)?;
        out.set_item("requested_k", self.requested_k)?;
        out.set_item("retained_atom_indices", self.retained_atom_indices.clone())?;
        out.set_item("atom_basis", self.atom_basis.clone())?;
        out.set_item("atom_dim", self.atom_dim.clone())?;
        out.set_item("top_k", self.support_k)?;
        out.set_item("training_mean", self.training_mean.clone())?;
        out.set_item("fitted", self.fitted.clone().into_pyarray(py))?;
        out.set_item(
            "support_indices",
            support_indices(&self.term)
                .map_err(py_value_error)?
                .into_pyarray(py),
        )?;
        out.set_item(
            "support_values",
            support_values(&self.term).into_pyarray(py),
        )?;
        out.set_item("coords", coords_rows(py, &self.term)?)?;
        let decoders = PyList::empty(py);
        for atom in &self.term.atoms {
            decoders.append(atom.decoder_coefficients().clone().into_pyarray(py))?;
        }
        out.set_item("decoder_blocks", decoders)?;
        out.set_item("log_lambda_smooth", self.log_lambda_smooth.clone())?;
        out.set_item("ard_precisions", self.ard_precisions.clone())?;
        out.set_item("criterion", self.criterion)?;
        out.set_item("criterion_kind", SUPPORT_CRITERION_KIND.tag())?;
        out.set_item(
            "certificates",
            json_value_to_py(py, &self.certificates)?,
        )?;
        out.set_item(
            "termination",
            json_value_to_py(py, &self.termination)?,
        )?;
        // Without these the payload cannot rebuild the model it came from:
        // `reconstruction_r2` is not recoverable from `fitted` alone (the
        // training target is not stored), and the two fit knobs are model
        // state that `from_dict` must restore rather than invent. The inner
        // tolerance and the iteration budget are not among them: the tolerance is
        // derived from each solve's term, and the budget is the engine's.
        out.set_item("reconstruction_r2", self.reconstruction_r2)?;
        out.set_item("trust_radius", self.trust_radius)?;
        out.set_item("random_state", self.random_state)?;
        Ok(out.unbind().into_any())
    }

    /// Rebuild a fitted overcomplete model from [`Self::to_dict`] (#2567).
    ///
    /// `atom_topologies` is re-derived from `atom_basis` rather than stored,
    /// because it is a function of the bases and a stored copy could disagree
    /// with them.
    #[staticmethod]
    fn from_dict(payload: &Bound<'_, PyDict>) -> PyResult<Self> {
        let schema: String = required_field(payload, "schema")?.extract()?;
        if schema != SUPPORT_SCHEMA_TAG {
            return Err(py_value_error(format!(
                "ManifoldSAESupport.from_dict: schema {schema:?} is not {SUPPORT_SCHEMA_TAG:?}; \
                 dense payloads tagged {} load with ManifoldSAE.from_dict",
                crate::manifold::manifold_sae_payload::SCHEMA_TAG
            )));
        }
        let requested_k: usize = required_field(payload, "requested_k")?.extract()?;
        let retained_atom_indices: Vec<usize> =
            required_field(payload, "retained_atom_indices")?.extract()?;
        let atom_basis: Vec<String> = required_field(payload, "atom_basis")?.extract()?;
        let atom_dim: Vec<usize> = required_field(payload, "atom_dim")?.extract()?;
        let support_k: usize = required_field(payload, "top_k")?.extract()?;
        let training_mean: Vec<f64> = required_field(payload, "training_mean")?.extract()?;
        let fitted = extract_f64_matrix(&required_field(payload, "fitted")?, "fitted")?;
        let support_indices = extract_u32_matrix(
            &required_field(payload, "support_indices")?,
            "support_indices",
        )?
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
        let support_values = extract_f64_matrix(
            &required_field(payload, "support_values")?,
            "support_values",
        )?
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
        let coord_items: Vec<Bound<'_, PyAny>> =
            required_field(payload, "coords")?.extract()?;
        let mut coords = Vec::with_capacity(coord_items.len());
        for item in &coord_items {
            coords.push(extract_f64_vector(item, "coords")?);
        }
        let decoder_items: Vec<Bound<'_, PyAny>> =
            required_field(payload, "decoder_blocks")?.extract()?;
        let mut decoder_blocks = Vec::with_capacity(decoder_items.len());
        for item in &decoder_items {
            decoder_blocks.push(extract_f64_matrix(item, "decoder_blocks")?);
        }
        let log_lambda_smooth: Vec<f64> =
            required_field(payload, "log_lambda_smooth")?.extract()?;
        let ard_precisions: Vec<Vec<f64>> =
            required_field(payload, "ard_precisions")?.extract()?;
        let criterion: f64 = required_field(payload, "criterion")?.extract()?;
        let reconstruction_r2: f64 = required_field(payload, "reconstruction_r2")?.extract()?;
        let trust_radius: f64 = required_field(payload, "trust_radius")?.extract()?;
        let random_state: u64 = required_field(payload, "random_state")?.extract()?;
        let certificates = crate::manifold::manifold_sae_coercion::py_any_to_json_value(
            &required_field(payload, "certificates")?,
        )?;
        let termination = crate::manifold::manifold_sae_coercion::py_any_to_json_value(
            &required_field(payload, "termination")?,
        )?;

        let output_dim = training_mean.len();
        let atom_topologies = gam::terms::sae::atom_schema::topologies_for_bases(&atom_basis)
            .map_err(py_value_error)?;
        let term = rehydrate_sae_support_term(SaeSupportRehydrateRequest {
            atom_basis: atom_basis.clone(),
            atom_dim: atom_dim.clone(),
            output_dim,
            support_k,
            random_state,
            support_indices,
            support_values,
            coords,
            decoder_blocks,
        })
        .map_err(py_value_error)?;
        Ok(Self {
            term,
            requested_k,
            retained_atom_indices,
            atom_basis,
            atom_dim,
            atom_topologies,
            support_k,
            training_mean,
            fitted,
            reconstruction_r2,
            log_lambda_smooth,
            ard_precisions,
            criterion,
            certificates,
            termination,
            trust_radius,
            random_state,
        })
    }

    /// Write the tagged JSON payload for `gamfit.load` (#2567 follow-up: the
    /// pickle round-trip worked, the JSON leg had no writer). Field-for-field
    /// the same payload as [`Self::to_dict`], with arrays carried as nested
    /// lists -- which [`Self::from_dict`] now accepts from either carrier.
    fn save(&self, path: &str) -> PyResult<()> {
        let support = support_indices(&self.term).map_err(py_value_error)?;
        let support_json = serde_json::Value::Array(
            support
                .rows()
                .into_iter()
                .map(|row| {
                    serde_json::Value::Array(
                        row.iter().map(|&v| serde_json::json!(v)).collect(),
                    )
                })
                .collect(),
        );
        let coords_json = serde_json::Value::Array(
            (0..self.term.n_obs())
                .map(|row| {
                    serde_json::Value::Array(
                        self.term
                            .assignment
                            .coords_row(row)
                            .iter()
                            .map(|&v| serde_json::json!(v))
                            .collect(),
                    )
                })
                .collect(),
        );
        let decoders_json = serde_json::Value::Array(
            self.term
                .atoms
                .iter()
                .map(|atom| matrix_json(atom.decoder_coefficients()))
                .collect(),
        );
        let payload = serde_json::json!({
            "schema": SUPPORT_SCHEMA_TAG,
            "requested_k": self.requested_k,
            "retained_atom_indices": self.retained_atom_indices,
            "atom_basis": self.atom_basis,
            "atom_dim": self.atom_dim,
            "top_k": self.support_k,
            "training_mean": self.training_mean,
            "fitted": matrix_json(&self.fitted),
            "support_indices": support_json,
            "support_values": matrix_json(&support_values(&self.term)),
            "coords": coords_json,
            "decoder_blocks": decoders_json,
            "log_lambda_smooth": self.log_lambda_smooth,
            "ard_precisions": self.ard_precisions,
            "criterion": require_finite_for_json("criterion", self.criterion)?,
            "criterion_kind": SUPPORT_CRITERION_KIND.tag(),
            "certificates": self.certificates,
            "termination": self.termination,
            "reconstruction_r2": require_finite_for_json(
                "reconstruction_r2",
                self.reconstruction_r2,
            )?,
            "trust_radius": require_finite_for_json("trust_radius", self.trust_radius)?,
            "random_state": self.random_state,
        });
        let text = serde_json::to_string(&payload).map_err(|error| {
            py_value_error(format!("ManifoldSAESupport.save: serialization failed: {error}"))
        })?;
        std::fs::write(path, text).map_err(|error| {
            py_value_error(format!("ManifoldSAESupport.save: writing {path:?} failed: {error}"))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ManifoldSAESupport(K_requested={}, K_retained={}, n={}, p={}, assignment=\"topk\", support={})",
            self.requested_k,
            self.term.k_atoms(),
            self.term.n_obs(),
            self.term.output_dim(),
            self.support_k,
        )
    }

    #[getter]
    fn chosen_k(&self) -> usize {
        self.term.k_atoms()
    }
    #[getter]
    fn requested_k(&self) -> usize {
        self.requested_k
    }
    #[getter]
    fn assignment(&self) -> &'static str {
        "topk"
    }
    #[getter]
    fn top_k(&self) -> usize {
        self.support_k
    }
    #[getter]
    fn atom_topologies(&self) -> Vec<String> {
        self.atom_topologies.clone()
    }
    #[getter]
    fn hybrid_split(&self, py: Python<'_>) -> PyObject {
        py.None()
    }
    #[getter]
    fn certificates(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_value_to_py(py, &self.certificates)
    }
    #[getter]
    fn termination(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_value_to_py(py, &self.termination)
    }
    #[getter]
    fn fitted<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.fitted.clone().into_pyarray(py)
    }
    #[getter]
    fn training_mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from(self.training_mean.clone()).into_pyarray(py)
    }
    #[getter]
    fn reconstruction_r2(&self) -> f64 {
        self.reconstruction_r2
    }
    /// The terminal support criterion value. It is the support route's
    /// quasi-Laplace score on the Gauss–Newton reduced Schur, not the dense fit's
    /// `penalized_quasi_laplace_criterion`, and the two do not compare (#2933 F27).
    #[getter]
    fn criterion(&self) -> f64 {
        self.criterion
    }
    #[getter]
    fn criterion_kind(&self) -> &'static str {
        SUPPORT_CRITERION_KIND.tag()
    }
    #[getter]
    fn support_indices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u32>>> {
        Ok(support_indices(&self.term)
            .map_err(py_value_error)?
            .into_pyarray(py))
    }
    #[getter]
    fn support_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        support_values(&self.term).into_pyarray(py)
    }
    #[getter]
    fn decoder_blocks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let blocks = PyList::empty(py);
        for atom in &self.term.atoms {
            blocks.append(atom.decoder_coefficients().clone().into_pyarray(py))?;
        }
        Ok(blocks)
    }
}

pub(crate) fn fit_support_sparse_manifold_sae(
    py: Python<'_>,
    request: SupportSparseFitRequest<'_>,
) -> PyResult<PyObject> {
    let censused = fit_sae_support_sparse_with_census(SaeSupportSparseFitRequest {
        target: request.target,
        atom_basis: request.atom_basis,
        atom_dim: request.atom_dim,
        support_k: request.support_k,
        initial_smoothness: request.initial_smoothness,
        // `max_iter` is the caller's OUTER smoothing-search budget. The inner fixed
        // point has no budget: it stops on its certificate or on a proven stall (#2576).
        max_outer_iter: request.max_iter,
        trust_radius: request.trust_radius,
        random_state: request.random_state,
    })
    .map_err(py_value_error)?;
    let linear_bulk_census = censused.census_json();
    let SaeSupportSparseFit {
        outer,
        requested_atoms,
        retained_atom_indices,
        atom_basis,
        atom_dim,
        training_mean,
        fitted,
        reconstruction_r2,
        migration,
    } = censused.fit;
    let fixed = fixed_point_json(&outer.fixed_point, outer.inner_tolerance);
    let outer_certificate = serde_json::to_value(&outer.outer_certificate)
        .map_err(|error| py_value_error(error.to_string()))?;
    let logdet = &outer.logdet_uncertainty;
    let certificates = serde_json::json!({
        "representation": "support_sparse",
        "inner_fixed_point": fixed,
        "outer_stationarity": outer_certificate,
        "stochastic_log_det": {
            "probes": logdet.probes,
            "criterion_std_err": logdet.criterion_std_err,
            "seen_gradient_std_err_norm": logdet.seen_gradient_std_err_norm,
            "unseen_projected_gradient_norm": logdet.unseen_projected_gradient_norm,
            "unseen_gradient_std_err_norm": logdet.unseen_gradient_std_err_norm,
            "plans": logdet.plans,
        },
        "migration": migration.to_json(),
        "linear_bulk_census": linear_bulk_census,
    });
    let termination = serde_json::json!({
        "verdict": "converged",
        "inner_iterations": outer.fixed_point.iterations,
        "outer_iterations": outer.outer_iterations,
        "recurred": outer.fixed_point.recurred,
    });
    let atom_topologies =
        gam::terms::sae::atom_schema::topologies_for_bases(&atom_basis).map_err(py_value_error)?;
    let log_lambda_smooth = outer.lambda_smooth.iter().map(|value| value.ln()).collect();
    let model = SupportSparseManifoldSaeCore {
        term: outer.term,
        requested_k: requested_atoms,
        retained_atom_indices,
        atom_basis,
        atom_dim,
        atom_topologies,
        support_k: request.support_k,
        training_mean,
        fitted,
        reconstruction_r2,
        log_lambda_smooth,
        ard_precisions: outer.ard_precisions,
        criterion: outer.criterion.value(),
        certificates,
        termination,
        trust_radius: request.trust_radius,
        random_state: request.random_state,
    };
    Ok(Py::new(py, model)?.into_any())
}

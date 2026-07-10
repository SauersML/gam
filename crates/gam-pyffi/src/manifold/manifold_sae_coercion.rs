//! Rust-owned coercion helpers for the `ManifoldSAE.from_payload` -> flat
//! `to_dict` schema derivation (#2091 phase-2, design (A)). These own the
//! derivations so the fit path can return
//! a Rust-owned `ManifoldSaeCore` built directly from the raw
//! `sae_manifold_fit_minimal` payload, with no Python dataclass in the middle.
//!
//! This module owns assignment tokens and topology aliases, topology naming, distilled
//! assignment activation dispatch, and the periodic shape-band reorder,
//! exposed to Python as `sae_atom_topologies` / `sae_periodic_shape_band_reorder`
//! — the same Rust-owner pattern as `sae_canonical_n_harmonics`. The full
//! `RawFitPayload -> ManifoldSaePayload` assembly below consumes those same
//! helpers; Python only marshals typed inputs to this owner.

use crate::manifold::manifold_sae_payload::{
    AtomPayload, CrosscoderPayload, ManifoldSaePayload, SCHEMA_TAG,
};
use ndarray::{Array2, Array3, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyTuple};
use serde_json::Value;

// The pure token-canonicalization schema (basis/topology aliases, assignment
// tables, n_harmonics repair, chart periods) moved to the library
// (`gam_sae::atom_schema`) so the CLI, Rust users, and this binding share one
// vocabulary — issue #2236. Re-exported here so every established
// `manifold_sae_coercion::X` path keeps resolving.
pub(crate) use gam::terms::sae::atom_schema::{
    basis_kind_for_topology, basis_to_topology, canon_name, canonical_assignment_kind,
    canonical_basis_kind, canonical_n_harmonics, canonical_topology, coordinate_periods_for_basis,
    flat_block_assignment, topologies_for_bases, topology_for_bases,
};

/// Column mean `x.mean(axis=0)` -> `(P,)` — the training-mean centering vector.
/// Owned in Rust so neither the fit builder nor the gamfit SAC lift
/// (`StagewiseSAE.to_manifold_sae`) computes a reduction in production Python
/// (SPEC thin-wrapper rule). `n == 0` yields zeros (matching the builder's prior
/// inline guard). Shared by [`build_manifold_sae_payload`] and the
/// `sae_manifold_training_mean` pyfunction.
pub(crate) fn column_mean(x: ArrayView2<'_, f64>) -> Vec<f64> {
    let (n, p) = x.dim();
    (0..p)
        .map(|j| {
            if n == 0 {
                0.0
            } else {
                x.column(j).sum() / n as f64
            }
        })
        .collect()
}

/// Extract the compact per-channel decoder-covariance factor `(p, M, M)` from the
/// dense `(M*p, M*p)` phi-scaled covariance (#2091), the shared core of the
/// `decoder_channel_cov_factors` pyfunction and the `from_fit_payload` builder.
/// `blocks[c, b1, b2] = cov[b1*p + c, b2*p + c]` — the same-channel Schur blocks
/// the shape band reads. `None` when the layout does not match an `(M, p)`
/// decoder (`m_basis <= 0`, empty, or `total % m != 0`), matching the prior
/// contract; the band's stored `shape_band_sd` still recovers.
pub(crate) fn channel_cov_factors(cov: ArrayView2<'_, f64>, m_basis: i64) -> Option<Array3<f64>> {
    let total = cov.nrows();
    if m_basis <= 0 {
        return None;
    }
    let m = m_basis as usize;
    if total == 0 || total % m != 0 {
        return None;
    }
    let p = total / m;
    let mut blocks = Array3::<f64>::zeros((p, m, m));
    for c in 0..p {
        for b1 in 0..m {
            for b2 in 0..m {
                blocks[[c, b1, b2]] = cov[[b1 * p + c, b2 * p + c]];
            }
        }
    }
    Some(blocks)
}

/// Convert an arbitrary JSON-shaped Python object into a `serde_json::Value`
/// (#2091). The inverse of the crate's `json_value_to_py`; the
/// `from_fit_payload` builder needs it to carry the raw fit payload's opaque
/// report blocks (`diagnostics`, `atom_inference`, `hybrid_split`, …) through as
/// `Value` without a Python `json.dumps` hop. This owns the coercion applied to
/// report blocks: numpy arrays/scalars are
/// `.tolist()`-flattened, `dict` keys are stringified, `list`/`tuple` become
/// arrays, and JSON scalars pass through. `bool` is checked before `int`
/// (Python `bool` subclasses `int`), and `int` before `float` so an integral
/// value serializes as a JSON integer, matching `json.dumps`.
///
/// NaN / +-Inf policy (explicit, #2091): a non-finite `f64` maps to
/// `Value::Null`. `serde_json::Value` provably cannot represent non-finite
/// numbers (`Number::from_f64` returns `None`), and the existing
/// `ManifoldSaeCore::new` path (Python `json.dumps` -> `ManifoldSaePayload::from_json`)
/// rejects the bare `NaN` / `Infinity` literals `json.dumps` emits (serde_json's
/// parser errors), so a non-finite report value has never round-tripped through
/// the Rust schema at all. Nulling it (rather than erroring) keeps a fit whose
/// diagnostics carry a meaningless non-finite value loadable instead of failing
/// the whole build; the `json_value_cannot_hold_nonfinite_*` test pins both the
/// schema limitation and the parser rejection so the policy is asserted, not
/// assumed.
pub(crate) fn py_any_to_json_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::Number(i.into()));
    }
    if let Ok(f) = obj.extract::<f64>() {
        // Non-finite (NaN / +-Inf) -> Null: `from_f64` returns `None`, matching
        // the schema's inability to hold non-finite (see the fn doc-comment).
        return Ok(serde_json::Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or(Value::Null));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = serde_json::Map::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            // Schema keys are always strings.
            let key: String = match k.extract::<String>() {
                Ok(key) => key,
                Err(_) => k.str()?.extract::<String>()?,
            };
            map.insert(key, py_any_to_json_value(&v)?);
        }
        return Ok(Value::Object(map));
    }
    // A numpy array (or any object exposing `.tolist()`) flattens to native
    // Python lists first, then recurse.
    if obj.hasattr("tolist")? {
        let listed = obj.call_method0("tolist")?;
        return py_any_to_json_value(&listed);
    }
    if let Ok(seq) = obj.cast::<PyList>() {
        let mut arr = Vec::with_capacity(seq.len());
        for item in seq.iter() {
            arr.push(py_any_to_json_value(&item)?);
        }
        return Ok(Value::Array(arr));
    }
    if let Ok(seq) = obj.cast::<PyTuple>() {
        let mut arr = Vec::with_capacity(seq.len());
        for item in seq.iter() {
            arr.push(py_any_to_json_value(&item)?);
        }
        return Ok(Value::Array(arr));
    }
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "py_any_to_json_value: cannot convert Python object of type {} to JSON",
        obj.get_type().name()?,
    )))
}

/// Python bridge for the complete dictionary's structural chart periods.
#[pyfunction]
pub(crate) fn sae_coordinate_periods(
    basis_kinds: Vec<String>,
    atom_dims: Vec<usize>,
) -> PyResult<Vec<Vec<Option<f64>>>> {
    if basis_kinds.len() != atom_dims.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "basis_kinds and atom_dims must have equal length; got {} and {}",
            basis_kinds.len(),
            atom_dims.len()
        )));
    }
    basis_kinds
        .iter()
        .zip(atom_dims)
        .map(|(basis, dim)| coordinate_periods_for_basis(basis, dim))
        .collect::<Result<Vec<_>, _>>()
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Python bridge for [`canonical_assignment_kind`].
#[pyfunction]
pub(crate) fn sae_canonical_assignment_kind(kind: &str) -> PyResult<String> {
    canonical_assignment_kind(kind)
        .map(str::to_string)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Python bridge for [`canonical_basis_kind`].
#[pyfunction]
pub(crate) fn sae_canonical_basis_kind(name: &str) -> String {
    canonical_basis_kind(name)
}

/// Python bridge for [`basis_kind_for_topology`].
#[pyfunction]
pub(crate) fn sae_basis_kind_for_topology(name: &str) -> String {
    basis_kind_for_topology(name)
}

/// Python bridge for [`basis_to_topology`].
#[pyfunction]
pub(crate) fn sae_topology_for_basis(name: &str) -> String {
    basis_to_topology(name)
}

/// Python bridge for [`canonical_topology`].
#[pyfunction]
pub(crate) fn sae_canonical_topology(name: &str) -> String {
    canonical_topology(name)
}

/// Python bridge for [`flat_block_assignment`].
#[pyfunction]
pub(crate) fn sae_flat_block_assignment(gating: &str) -> PyResult<String> {
    flat_block_assignment(gating)
        .map(str::to_string)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Convert a fitted encoder's `(N, K)` routing logits into assignment values
/// with the exact production kernel used by the Rust fit. This deletes the
/// former NumPy reimplementation (including its independently-maintained IBP
/// exponent) from `gamfit.distill`.
#[pyfunction(signature = (
    logits, assignment, temperature, alpha, threshold,
    learnable_alpha=false, log_lambda_sparse=None, top_k=None
))]
pub(crate) fn sae_activation_matrix_from_logits<'py>(
    py: Python<'py>,
    logits: PyReadonlyArray2<'py, f64>,
    assignment: &str,
    temperature: f64,
    alpha: f64,
    threshold: f64,
    learnable_alpha: bool,
    log_lambda_sparse: Option<f64>,
    top_k: Option<usize>,
) -> PyResult<Py<PyArray2<f64>>> {
    let kind =
        canonical_assignment_kind(assignment).map_err(pyo3::exceptions::PyValueError::new_err)?;
    if kind == "ibp_map" && !(alpha.is_finite() && alpha > 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "alpha must be finite and positive; got {alpha}"
        )));
    }
    let effective_alpha = if kind == "ibp_map" && learnable_alpha {
        let rho = log_lambda_sparse.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "learnable IBP activation requires terminal log_lambda_sparse",
            )
        })?;
        if !rho.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "log_lambda_sparse must be finite; got {rho}"
            )));
        }
        gam::terms::sae::manifold::resolve_learnable_weight(alpha, rho)
    } else {
        alpha
    };
    if kind == "topk" {
        let support = top_k.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "topk activation requires the fitted top_k support size",
            )
        })?;
        let logits = logits.as_array();
        let (n_rows, k_atoms) = logits.dim();
        if support == 0 || support > k_atoms {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "top_k must be in [1, K={k_atoms}]; got {support}"
            )));
        }
        if let Some(((row, atom), value)) =
            logits.indexed_iter().find(|(_, value)| !value.is_finite())
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "assignment logits contain non-finite value at ({row}, {atom}): {value}"
            )));
        }
        let mut values = Array2::<f64>::zeros((n_rows, k_atoms));
        for row in 0..n_rows {
            values
                .row_mut(row)
                .assign(&gam::terms::sae::assignment::topk_row(
                    logits.row(row),
                    support,
                ));
        }
        return Ok(values.into_pyarray(py).unbind());
    }
    if kind == "threshold_gate" && !threshold.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "threshold must be finite; got {threshold}"
        )));
    }
    gam::terms::sae::assignment::activation_matrix_from_logits(
        logits.as_array(),
        kind,
        temperature,
        effective_alpha,
        threshold,
    )
    .map(|values| values.into_pyarray(py).unbind())
    .map_err(pyo3::exceptions::PyValueError::new_err)
}

struct StagewisePayloadConfig {
    assignment: String,
    assignment_label: String,
    alpha: f64,
    learnable_alpha: bool,
    tau: f64,
    sparsity_strength: f64,
    smoothness: f64,
    learning_rate: f64,
    max_iter: i64,
    random_state: i64,
    jumprelu_threshold: f64,
}

fn require_matrix_shape(
    matrix: &Array2<f64>,
    rows: usize,
    cols: usize,
    label: &str,
) -> Result<(), String> {
    if matrix.dim() != (rows, cols) {
        return Err(format!(
            "sae_manifold_core_from_stagewise: {label} must have shape ({rows}, {cols}); got {:?}",
            matrix.dim()
        ));
    }
    if let Some(((row, col), value)) = matrix.indexed_iter().find(|(_, value)| !value.is_finite()) {
        return Err(format!(
            "sae_manifold_core_from_stagewise: {label} contains non-finite value \
             at ({row}, {col}): {value}"
        ));
    }
    Ok(())
}

/// Build the persisted model schema for a stagewise-composed frozen dictionary.
/// This is the only owner of the v1 field/default layout for that lift; Python
/// passes typed arrays and seed scalars and receives a [`ManifoldSaePayload`].
fn build_stagewise_manifold_sae_payload(
    basis_kinds: Vec<String>,
    decoder_blocks: Vec<Array2<f64>>,
    atom_dims: Vec<i64>,
    coords: Vec<Array2<f64>>,
    assignments: Array2<f64>,
    fitted: Array2<f64>,
    logits: Array2<f64>,
    training: Array2<f64>,
    reconstruction_r2: f64,
    cfg: &StagewisePayloadConfig,
) -> Result<ManifoldSaePayload, String> {
    let k = basis_kinds.len();
    if k == 0 {
        return Err("sae_manifold_core_from_stagewise requires at least one atom".to_string());
    }
    for (label, len) in [
        ("decoder_blocks", decoder_blocks.len()),
        ("atom_dims", atom_dims.len()),
        ("coords", coords.len()),
    ] {
        if len != k {
            return Err(format!(
                "sae_manifold_core_from_stagewise: {label} has length {len}, expected K={k}"
            ));
        }
    }
    let (n, p) = training.dim();
    if n == 0 || p == 0 {
        return Err(
            "sae_manifold_core_from_stagewise requires non-empty training data".to_string(),
        );
    }
    require_matrix_shape(&training, n, p, "training")?;
    require_matrix_shape(&fitted, n, p, "fitted")?;
    require_matrix_shape(&assignments, n, k, "assignments")?;
    require_matrix_shape(&logits, n, k, "logits")?;
    for atom in 0..k {
        let dim = usize::try_from(atom_dims[atom]).map_err(|_| {
            format!(
                "sae_manifold_core_from_stagewise: atom_dims[{atom}] must be positive; got {}",
                atom_dims[atom]
            )
        })?;
        if dim == 0 {
            return Err(format!(
                "sae_manifold_core_from_stagewise: atom_dims[{atom}] must be positive"
            ));
        }
        require_matrix_shape(&coords[atom], n, dim, &format!("coords[{atom}]"))?;
        if decoder_blocks[atom].nrows() == 0 {
            return Err(format!(
                "sae_manifold_core_from_stagewise: decoder_blocks[{atom}] has zero basis rows"
            ));
        }
        require_matrix_shape(
            &decoder_blocks[atom],
            decoder_blocks[atom].nrows(),
            p,
            &format!("decoder_blocks[{atom}]"),
        )?;
    }
    if !reconstruction_r2.is_finite() {
        return Err(format!(
            "sae_manifold_core_from_stagewise: reconstruction_r2 must be finite; got {reconstruction_r2}"
        ));
    }

    let basis_sizes: Vec<i64> = decoder_blocks
        .iter()
        .map(|block| block.nrows() as i64)
        .collect();
    let raw_n_harmonics: Vec<i64> = basis_kinds
        .iter()
        .zip(&basis_sizes)
        .map(|(kind, &size)| {
            if kind == "periodic" {
                (size - 1) / 2
            } else {
                0
            }
        })
        .collect();
    let n_harmonics = canonical_n_harmonics(&basis_kinds, &raw_n_harmonics, &basis_sizes);
    let decoder_nested: Vec<Vec<Vec<f64>>> = decoder_blocks.iter().map(array2_to_nested).collect();
    let coords_nested: Vec<Vec<Vec<f64>>> = coords.iter().map(array2_to_nested).collect();
    let assignments_nested = array2_to_nested(&assignments);

    let atoms: Vec<AtomPayload> = (0..k)
        .map(|atom| AtomPayload {
            basis: basis_kinds[atom].clone(),
            decoder_coefficients: decoder_nested[atom].clone(),
            assignments: assignments_nested.iter().map(|row| row[atom]).collect(),
            coords: coords_nested[atom].clone(),
            coords_u_arc: None,
            evidence: None,
            active_dim: atom_dims[atom],
            decoder_covariance_channel_factors: None,
            shape_band_coords: None,
            shape_band_mean: None,
            shape_band_sd: None,
            functional_evidence: None,
        })
        .collect();
    let atom_topologies = topologies_for_bases(&basis_kinds);
    let atom_topology = topology_for_bases(&basis_kinds)
        .ok_or("sae_manifold_core_from_stagewise requires at least one basis")?;

    Ok(ManifoldSaePayload {
        schema: SCHEMA_TAG.to_string(),
        atom_topology,
        atom_topologies,
        assignment: cfg.assignment.clone(),
        assignment_label: cfg.assignment_label.clone(),
        alpha: cfg.alpha,
        learnable_alpha: cfg.learnable_alpha,
        tau: cfg.tau,
        sparsity_strength: cfg.sparsity_strength,
        smoothness: cfg.smoothness,
        learning_rate: cfg.learning_rate,
        max_iter: cfg.max_iter,
        random_state: cfg.random_state,
        top_k: None,
        jumprelu_threshold: cfg.jumprelu_threshold,
        oos_projection_top1: false,
        dispersion: 1.0,
        penalized_loss_score: None,
        reml_score: None,
        reconstruction_r2,
        primitive_names: vec!["sae_manifold_fit_stagewise".to_string()],
        basis_specs: basis_kinds.clone(),
        basis_kinds,
        atom_dims,
        basis_sizes,
        n_harmonics,
        training_mean: column_mean(training.view()),
        training_data: None,
        training_data_retained: false,
        fitted: array2_to_nested(&fitted),
        assignments: assignments_nested,
        low_level_logits: array2_to_nested(&logits),
        coords: coords_nested,
        decoder_blocks: decoder_nested,
        duchon_centers: vec![None; k],
        crosscoder: None,
        atoms,
        diagnostics: serde_json::json!({"atom_trust": [], "atoms": []}),
        top_k_projection: None,
        pre_topk: None,
        solver_plan: None,
        atom_two_lens: None,
        residual_gauge: None,
        incoherence_report: None,
        curvature_report: None,
        coordinate_fidelity: None,
        topology_persistence: None,
        atom_inference: None,
        certificates: None,
        structure_certificate: None,
        cotrain: None,
        hybrid_split: None,
        fisher_factors: None,
        fisher_provenance: None,
        metric_provenance: "Euclidean".to_string(),
        fisher_mass_residual: None,
        selected_log_lambda_sparse: None,
        selected_log_lambda_smooth: None,
        selected_log_ard: None,
        structured_residual_diagnostics: Vec::new(),
        termination: None,
    })
}

/// Typed Python entry point for the stagewise-to-model lift.
#[pyfunction(signature = (
    atom_topologies, decoder_blocks, atom_dims, coords, assignments, fitted,
    logits, training, assignment, assignment_label, alpha, learnable_alpha, tau,
    sparsity_strength, smoothness, learning_rate, max_iter, random_state,
    jumprelu_threshold, reconstruction_r2
))]
pub(crate) fn sae_manifold_core_from_stagewise<'py>(
    py: Python<'py>,
    atom_topologies: Vec<String>,
    decoder_blocks: Vec<PyReadonlyArray2<'py, f64>>,
    atom_dims: Vec<i64>,
    coords: Vec<PyReadonlyArray2<'py, f64>>,
    assignments: PyReadonlyArray2<'py, f64>,
    fitted: PyReadonlyArray2<'py, f64>,
    logits: PyReadonlyArray2<'py, f64>,
    training: PyReadonlyArray2<'py, f64>,
    assignment: &str,
    assignment_label: String,
    alpha: f64,
    learnable_alpha: bool,
    tau: f64,
    sparsity_strength: f64,
    smoothness: f64,
    learning_rate: f64,
    max_iter: i64,
    random_state: i64,
    jumprelu_threshold: f64,
    reconstruction_r2: f64,
) -> PyResult<Py<crate::ManifoldSaeCore>> {
    let assignment = canonical_assignment_kind(assignment)
        .map(str::to_string)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let basis_kinds = atom_topologies
        .iter()
        .map(|topology| canonical_basis_kind(topology))
        .collect();
    let decoder_blocks = decoder_blocks
        .iter()
        .map(|block| block.as_array().to_owned())
        .collect();
    let coords = coords
        .iter()
        .map(|block| block.as_array().to_owned())
        .collect();
    let cfg = StagewisePayloadConfig {
        assignment,
        assignment_label,
        alpha,
        learnable_alpha,
        tau,
        sparsity_strength,
        smoothness,
        learning_rate,
        max_iter,
        random_state,
        jumprelu_threshold,
    };
    let payload = build_stagewise_manifold_sae_payload(
        basis_kinds,
        decoder_blocks,
        atom_dims,
        coords,
        assignments.as_array().to_owned(),
        fitted.as_array().to_owned(),
        logits.as_array().to_owned(),
        training.as_array().to_owned(),
        reconstruction_r2,
        &cfg,
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Py::new(py, crate::ManifoldSaeCore { inner: payload })
}

/// Restore the `linear_block` label a flat-block fit reports as the generic
/// `linear`: for each atom the caller
/// DECLARED as `linear_block`/`flat_block` AND that the fit left at topology
/// `linear` (K unchanged), relabel its `basis_kinds` entry and topology to
/// `linear_block`. `basis_specs` (the fitted kinds) is intentionally not patched;
/// only the public `basis_kinds` / `atom_topologies` labels change. No-op
/// when `bases` is empty, length-mismatched (an evidence-gated K change), or
/// declares no block; positions the fit RETYPED keep their discovered topology.
pub(crate) fn preserve_linear_block_labels(
    basis_kinds: &mut [String],
    atom_topologies: &mut [String],
    bases: &[String],
) {
    if bases.is_empty() || bases.len() != atom_topologies.len() {
        return;
    }
    let want: Vec<bool> = bases
        .iter()
        .map(|b| matches!(canon_name(b).as_str(), "linear_block" | "flat_block"))
        .collect();
    if !want.iter().any(|&w| w) {
        return;
    }
    for i in 0..atom_topologies.len() {
        if want[i] && canon_name(&atom_topologies[i]) == "linear" {
            atom_topologies[i] = "linear_block".to_string();
            basis_kinds[i] = "linear_block".to_string();
        }
    }
}

/// Stable ascending reorder of a periodic atom's shape band by its single
/// coordinate column:
///
///   * `coords` is `(G, 1)` (one coordinate column, `d_k = 1` for a periodic
///     atom) — an error otherwise.
///   * `mean` / `sd` are `(G, p)` (the ambient band value / per-channel sd on
///     the `G`-point coordinate grid — not 1-D.
///   * `order = argsort(coords[:, 0], kind="mergesort")` — a STABLE ascending
///     sort; `coords` and the ROWS of `mean` / `sd` are all reindexed by it.
///   * When `coords` is absent OR `mean` is absent the band is dropped entirely
///     (`(None, None, None)`) — a periodic band without its amplitude-correct
///     Rust mean draws at the wrong radius, so an absent mean means no band.
///
/// `sd` follows `mean` (reindexed when present, `None` when absent), and both
/// must share `coords`'s row count `G`. The sort key uses `f64::total_cmp` for a
/// total, deterministic order (NaN sorts last, matching numpy's mergesort
/// NaN-at-end); shape-band coordinates are finite in practice so this only fixes
/// the degenerate tie/NaN ordering deterministically.
pub(crate) fn periodic_shape_band_reorder(
    coords: Option<Array2<f64>>,
    mean: Option<Array2<f64>>,
    sd: Option<Array2<f64>>,
) -> Result<
    (
        Option<Array2<f64>>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
    ),
    String,
> {
    let Some(coords) = coords else {
        return Ok((None, None, None));
    };
    if coords.ncols() != 1 {
        return Err(format!(
            "periodic shape_band_coords must be a 2D array with one coordinate \
             column; got shape ({}, {})",
            coords.nrows(),
            coords.ncols()
        ));
    }
    let Some(mean) = mean else {
        return Ok((None, None, None));
    };
    let n = coords.nrows();
    if mean.nrows() != n {
        return Err(format!(
            "periodic shape_band_mean rows ({}) must match shape_band_coords rows ({n})",
            mean.nrows()
        ));
    }
    if let Some(sd) = sd.as_ref() {
        if sd.nrows() != n {
            return Err(format!(
                "periodic shape_band_sd rows ({}) must match shape_band_coords rows ({n})",
                sd.nrows()
            ));
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    // Stable sort (Rust's sort_by is stable) on the single coordinate column,
    // matching numpy argsort(kind="mergesort").
    order.sort_by(|&a, &b| coords[[a, 0]].total_cmp(&coords[[b, 0]]));
    let coords_sorted = Array2::from_shape_fn((n, 1), |(i, _)| coords[[order[i], 0]]);
    // Reindex the ROWS of the (G, p) band arrays by the coordinate order.
    let reindex_rows = |m: &Array2<f64>| {
        let p = m.ncols();
        Array2::from_shape_fn((n, p), |(i, j)| m[[order[i], j]])
    };
    let mean_sorted = reindex_rows(&mean);
    let sd_sorted = sd.map(|s| reindex_rows(&s));
    Ok((Some(coords_sorted), Some(mean_sorted), sd_sorted))
}

/// Fit-time scalars and labels absent from the raw solver payload. Threaded into
/// [`build_manifold_sae_payload`] so the Rust-owned persisted schema is complete.
/// `assignment` is already canonicalized by this module's shared parser.
pub(crate) struct FitConfig {
    pub(crate) topology_fallback: String,
    pub(crate) assignment: String,
    pub(crate) assignment_label: String,
    pub(crate) penalties: Vec<String>,
    pub(crate) alpha: f64,
    pub(crate) learnable_alpha: bool,
    pub(crate) tau: f64,
    pub(crate) sparsity_strength: f64,
    pub(crate) smoothness: f64,
    pub(crate) learning_rate: f64,
    pub(crate) max_iter: i64,
    pub(crate) random_state: i64,
    pub(crate) top_k: Option<i64>,
    pub(crate) jumprelu_threshold: f64,
    /// The retained WP-D output-Fisher shard `U` `(n, p, r)`; `None` for a
    /// Euclidean fit. When present, `metric_provenance` still comes from the raw
    /// solver payload (the Rust fit stamped `"OutputFisher"`).
    pub(crate) fisher_factors: Option<Vec<Vec<Vec<f64>>>>,
    /// The shard's provenance tag (`shard[2]`); serialized only when
    /// `fisher_factors` is present, mirroring `to_dict`.
    pub(crate) fisher_provenance: Option<String>,
    /// The caller's declared per-atom bases, used by
    /// [`preserve_linear_block_labels`]. `None`/empty leaves the fitted labels.
    pub(crate) declared_bases: Option<Vec<String>>,
}

// --- typed serde_json::Value accessors for the raw solver payload -------------
fn vget<'a>(v: &'a Value, key: &str) -> Result<&'a Value, String> {
    v.get(key)
        .ok_or_else(|| format!("sae fit payload missing '{key}'"))
}
fn vf64(v: &Value, key: &str) -> Result<f64, String> {
    vget(v, key)?
        .as_f64()
        .ok_or_else(|| format!("sae fit payload '{key}' is not a number"))
}
fn vi64(v: &Value, key: &str) -> Result<i64, String> {
    vget(v, key)?
        .as_i64()
        .ok_or_else(|| format!("sae fit payload '{key}' is not an integer"))
}
fn vbool(v: &Value, key: &str) -> Result<bool, String> {
    vget(v, key)?
        .as_bool()
        .ok_or_else(|| format!("sae fit payload '{key}' is not a bool"))
}
fn vstr(v: &Value, key: &str) -> Result<String, String> {
    vget(v, key)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| format!("sae fit payload '{key}' is not a string"))
}
/// A key whose value is absent OR JSON null -> `None`; else `Some(&Value)`.
fn vopt<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    match v.get(key) {
        None => None,
        Some(x) if x.is_null() => None,
        Some(x) => Some(x),
    }
}
fn v_arr2(v: &Value) -> Result<Vec<Vec<f64>>, String> {
    serde_json::from_value(v.clone()).map_err(|e| format!("expected a 2-D float array: {e}"))
}
fn v_arr1(v: &Value) -> Result<Vec<f64>, String> {
    serde_json::from_value(v.clone()).map_err(|e| format!("expected a 1-D float array: {e}"))
}
fn nested_to_array2(rows: &[Vec<f64>]) -> Result<Array2<f64>, String> {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    let mut flat = Vec::with_capacity(nrows * ncols);
    for r in rows {
        if r.len() != ncols {
            return Err("ragged 2-D array".to_string());
        }
        flat.extend_from_slice(r);
    }
    Array2::from_shape_vec((nrows, ncols), flat).map_err(|e| e.to_string())
}
fn array2_to_nested(a: &Array2<f64>) -> Vec<Vec<f64>> {
    a.rows().into_iter().map(|r| r.to_vec()).collect()
}
fn array3_to_nested(a: &Array3<f64>) -> Vec<Vec<Vec<f64>>> {
    a.outer_iter()
        .map(|m| m.rows().into_iter().map(|r| r.to_vec()).collect())
        .collect()
}

/// The honest penalized-loss score (#1231):
/// `penalized_loss_score` else `oos_penalized_loss`; a present-but-null value
/// stays `None`; neither key present -> error.
fn penalized_loss_score(raw: &Value) -> Result<Option<f64>, String> {
    for key in ["penalized_loss_score", "oos_penalized_loss"] {
        if let Some(v) = raw.get(key) {
            return Ok(if v.is_null() {
                None
            } else {
                Some(
                    v.as_f64()
                        .ok_or_else(|| format!("sae fit payload '{key}' is not a number"))?,
                )
            });
        }
    }
    Err("sae fit payload is missing a penalized-loss score".to_string())
}

/// Reorder (periodic) or pass through (else) an atom's shape band. A
/// `periodic` atom's `(G, .)` band is stably reordered by
/// its coordinate column via [`periodic_shape_band_reorder`]; every other kind
/// keeps the arrays verbatim.
fn shape_band_for_kind(
    kind: &str,
    coords: Option<Vec<Vec<f64>>>,
    mean: Option<Vec<Vec<f64>>>,
    sd: Option<Vec<Vec<f64>>>,
) -> Result<
    (
        Option<Vec<Vec<f64>>>,
        Option<Vec<Vec<f64>>>,
        Option<Vec<Vec<f64>>>,
    ),
    String,
> {
    if kind != "periodic" {
        return Ok((coords, mean, sd));
    }
    let coords = coords.map(|c| nested_to_array2(&c)).transpose()?;
    let mean = mean.map(|m| nested_to_array2(&m)).transpose()?;
    let sd = sd.map(|s| nested_to_array2(&s)).transpose()?;
    let (c, m, s) = periodic_shape_band_reorder(coords, mean, sd)?;
    Ok((
        c.map(|a| array2_to_nested(&a)),
        m.map(|a| array2_to_nested(&a)),
        s.map(|a| array2_to_nested(&a)),
    ))
}

/// Build a [`ManifoldSaePayload`] (the flat `to_dict` schema) directly from the
/// raw `sae_manifold_fit_minimal` payload (as a `serde_json::Value`), the
/// training mean, and [`FitConfig`]. Fisher retention and `linear_block`
/// relabeling happen in this same builder, with no parallel Python schema.
pub(crate) fn build_manifold_sae_payload(
    raw: &Value,
    training_mean: Vec<f64>,
    cfg: &FitConfig,
) -> Result<ManifoldSaePayload, String> {
    let plans = vget(raw, "atom_plans")?
        .as_array()
        .ok_or("atom_plans is not a list")?;
    let atoms = vget(raw, "atoms")?
        .as_array()
        .ok_or("atoms is not a list")?;
    let k = atoms.len();
    // #977 variable-K contract (mirror from_payload's ingest asserts).
    if plans.len() != k {
        return Err(format!(
            "variable-K boundary: {k} atoms but {} atom_plans",
            plans.len()
        ));
    }
    let chosen_k = vi64(raw, "chosen_k")?;
    if chosen_k != k as i64 {
        return Err(format!("chosen_k={chosen_k} does not match {k} atoms"));
    }
    let assignments = v_arr2(vget(raw, "assignments_z")?)?;
    let logits = v_arr2(vget(raw, "logits")?)?;
    for (name, arr) in [("assignments_z", &assignments), ("logits", &logits)] {
        let cols = arr.first().map_or(0, Vec::len);
        if cols != k {
            return Err(format!("'{name}' must be (N, K={k}); got {cols} columns"));
        }
    }

    let kinds: Vec<String> = plans
        .iter()
        .map(|p| vstr(p, "kind"))
        .collect::<Result<_, _>>()?;
    let dims: Vec<i64> = plans
        .iter()
        .map(|p| vi64(p, "latent_dim"))
        .collect::<Result<_, _>>()?;
    let sizes: Vec<i64> = plans
        .iter()
        .map(|p| vi64(p, "basis_size"))
        .collect::<Result<_, _>>()?;
    let raw_nharm: Vec<i64> = plans
        .iter()
        .map(|p| vi64(p, "n_harmonics"))
        .collect::<Result<_, _>>()?;
    let decoder_blocks: Vec<Vec<Vec<f64>>> = atoms
        .iter()
        .map(|a| v_arr2(vget(a, "decoder_B")?))
        .collect::<Result<_, _>>()?;
    let widths: Vec<i64> = decoder_blocks.iter().map(|b| b.len() as i64).collect();
    let n_harmonics = canonical_n_harmonics(&kinds, &raw_nharm, &widths);
    let coords: Vec<Vec<Vec<f64>>> = atoms
        .iter()
        .map(|a| v_arr2(vget(a, "on_atom_coords_t")?))
        .collect::<Result<_, _>>()?;
    let duchon_centers: Vec<Option<Vec<Vec<f64>>>> = plans
        .iter()
        .map(|p| match vopt(p, "duchon_centers") {
            None => Ok(None),
            Some(v) => v_arr2(v).map(Some),
        })
        .collect::<Result<_, _>>()?;
    let score = penalized_loss_score(raw)?;

    let mut atom_payloads = Vec::with_capacity(k);
    for (idx, a) in atoms.iter().enumerate() {
        let decoder_coefficients = decoder_blocks[idx].clone();
        // Per-atom gate column = column idx of the top-level assignments_z.
        let assignments_col: Vec<f64> = assignments
            .iter()
            .map(|row| row.get(idx).copied().unwrap_or(0.0))
            .collect();
        let coords_u_arc = match vopt(a, "on_atom_coords_u_arc") {
            None => None,
            Some(v) => Some(v_arr1(v)?),
        };
        let decoder_covariance_channel_factors = match vopt(a, "decoder_covariance") {
            None => None,
            Some(v) => {
                let cov = nested_to_array2(&v_arr2(v)?)?;
                // M_k = decoder rows = a.decoder_coefficients.shape[0].
                let m = decoder_coefficients.len() as i64;
                channel_cov_factors(cov.view(), m).map(|b| array3_to_nested(&b))
            }
        };
        let sb_coords = vopt(a, "shape_band_coords").map(v_arr2).transpose()?;
        let sb_mean = vopt(a, "shape_band_mean").map(v_arr2).transpose()?;
        let sb_sd = vopt(a, "shape_band_sd").map(v_arr2).transpose()?;
        let (sb_coords, sb_mean, sb_sd) =
            shape_band_for_kind(&kinds[idx], sb_coords, sb_mean, sb_sd)?;
        atom_payloads.push(AtomPayload {
            basis: vstr(a, "basis_kind")?,
            decoder_coefficients,
            assignments: assignments_col,
            coords: coords[idx].clone(),
            coords_u_arc,
            evidence: score,
            active_dim: vi64(a, "active_dim")?,
            decoder_covariance_channel_factors,
            shape_band_coords: sb_coords,
            shape_band_mean: sb_mean,
            shape_band_sd: sb_sd,
            functional_evidence: vopt(a, "functional_evidence").cloned(),
        });
    }

    let mut primitive_names = Vec::with_capacity(cfg.penalties.len() + 1);
    primitive_names.push("rust_module.sae_manifold_fit_minimal".to_string());
    primitive_names.extend(cfg.penalties.iter().cloned());

    // `basis_specs` keeps the fitted kinds; public labels may be restored to
    // `linear_block` for a flat-block fit.
    let mut basis_kinds = kinds.clone();
    let mut atom_topologies = topologies_for_bases(&kinds);
    if let Some(bases) = cfg.declared_bases.as_deref() {
        preserve_linear_block_labels(&mut basis_kinds, &mut atom_topologies, bases);
    }
    let atom_topology =
        topology_for_bases(&basis_kinds).unwrap_or_else(|| cfg.topology_fallback.clone());
    let metric_provenance = vopt(raw, "metric_provenance")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| "Euclidean".to_string());
    let fisher_mass_residual = vopt(raw, "fisher_mass_residual").map(v_arr1).transpose()?;
    let selected_log_lambda_sparse = match vopt(raw, "log_lambda_sparse") {
        None => None,
        Some(v) => Some(
            v.as_f64()
                .ok_or("sae fit payload 'log_lambda_sparse' is not a number")?,
        ),
    };
    let selected_log_lambda_smooth = vopt(raw, "log_lambda_smooth").map(v_arr1).transpose()?;
    let selected_log_ard = vopt(raw, "log_ard").map(v_arr2).transpose()?;
    let structure_certificate = vopt(raw, "structure_certificate")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let crosscoder = vopt(raw, "crosscoder")
        .map(|value| {
            serde_json::from_value::<CrosscoderPayload>(value.clone())
                .map_err(|error| format!("invalid crosscoder payload: {error}"))
        })
        .transpose()?;
    let report = |key: &str| vopt(raw, key).cloned();

    Ok(ManifoldSaePayload {
        schema: SCHEMA_TAG.to_string(),
        atom_topology,
        atom_topologies,
        assignment: cfg.assignment.clone(),
        assignment_label: cfg.assignment_label.clone(),
        alpha: cfg.alpha,
        learnable_alpha: cfg.learnable_alpha,
        tau: cfg.tau,
        sparsity_strength: cfg.sparsity_strength,
        smoothness: cfg.smoothness,
        learning_rate: cfg.learning_rate,
        max_iter: cfg.max_iter,
        random_state: cfg.random_state,
        top_k: cfg.top_k,
        jumprelu_threshold: cfg.jumprelu_threshold,
        oos_projection_top1: vbool(raw, "oos_projection_top1")?,
        dispersion: vf64(raw, "dispersion")?,
        penalized_loss_score: score,
        // Duplicate write-alias re-derived in `to_json`; the field value here is
        // irrelevant to the serialized output.
        reml_score: None,
        reconstruction_r2: vf64(raw, "reconstruction_r2")?,
        primitive_names,
        // basis_specs = fitted kinds (unpatched); basis_kinds = post-relabel.
        basis_specs: kinds,
        basis_kinds,
        atom_dims: dims,
        basis_sizes: sizes,
        n_harmonics,
        training_mean,
        training_data: None,
        training_data_retained: false,
        fitted: v_arr2(vget(raw, "fitted")?)?,
        assignments,
        low_level_logits: logits,
        coords,
        decoder_blocks,
        duchon_centers,
        crosscoder,
        atoms: atom_payloads,
        diagnostics: vget(raw, "diagnostics")?.clone(),
        top_k_projection: report("top_k_projection"),
        pre_topk: report("pre_topk"),
        solver_plan: report("solver_plan"),
        atom_two_lens: report("atom_two_lens"),
        residual_gauge: report("residual_gauge"),
        incoherence_report: report("incoherence_report"),
        curvature_report: report("curvature_report"),
        coordinate_fidelity: report("coordinate_fidelity"),
        topology_persistence: report("topology_persistence"),
        atom_inference: report("atom_inference"),
        certificates: report("certificates"),
        structure_certificate,
        cotrain: report("cotrain"),
        hybrid_split: report("hybrid_split"),
        fisher_factors: cfg.fisher_factors.clone(),
        // to_dict: `None if fisher_factors is None else str(fisher_provenance)`.
        fisher_provenance: if cfg.fisher_factors.is_some() {
            cfg.fisher_provenance.clone()
        } else {
            None
        },
        metric_provenance,
        fisher_mass_residual,
        selected_log_lambda_sparse,
        selected_log_lambda_smooth,
        selected_log_ard,
        structured_residual_diagnostics: Vec::new(),
        // #2235 — runtime-only termination ledger, carried straight from the raw
        // fit payload (the legacy dataclass `from_payload` did the same:
        // `termination = payload.get("termination")`). Not persisted by to_dict.
        termination: report("termination"),
    })
}

#[cfg(test)]
mod manifold_sae_coercion_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn canon_name_lowercases_trims_and_dashes_to_underscore() {
        assert_eq!(canon_name("  Periodic-Spline "), "periodic_spline");
        assert_eq!(canon_name("EUCLIDEAN"), "euclidean");
        assert_eq!(canon_name("linear-rank1"), "linear_rank1");
    }

    #[test]
    fn assignment_tokens_have_one_strict_parser() {
        for canonical in ["softmax", "ibp_map", "threshold_gate", "topk"] {
            assert_eq!(
                canonical_assignment_kind(canonical),
                Ok(canonical),
                "canonical token {canonical}"
            );
        }
        for rejected in [
            " SoftMax ",
            "ibp",
            "ibp-map",
            "gated",
            "jump-relu",
            "jumprelu",
            "top-k",
        ] {
            let err = canonical_assignment_kind(rejected).expect_err("alias must be rejected");
            assert!(err.contains("not a recognized assignment kind"));
            assert!(err.contains("ibp_map") && err.contains("threshold_gate"));
        }
    }

    #[test]
    fn topology_and_basis_aliases_share_one_directional_table() {
        for (alias, canonical) in [
            ("circle", "periodic"),
            ("Periodic-Spline", "periodic"),
            ("affine", "linear"),
            ("flat-block", "linear_block"),
            ("euclidean_quadratic_patch", "euclidean"),
            ("hyperbolic", "poincare"),
            ("mobius-band", "mobius"),
            (" AUTO ", "auto"),
        ] {
            assert_eq!(canonical_basis_kind(alias), canonical, "alias {alias}");
            assert_eq!(basis_kind_for_topology(alias), canonical, "alias {alias}");
        }
        assert_eq!(canonical_basis_kind(" Weird-Kind "), "weird_kind");
        assert_eq!(basis_kind_for_topology(" Weird-Kind "), " Weird-Kind ");
        assert_eq!(canonical_topology(" AUTO "), "auto");
        assert_eq!(canonical_topology("mobius-band"), "mobius");
    }

    #[test]
    fn coordinate_periods_cover_compact_and_twisted_charts() {
        assert_eq!(
            coordinate_periods_for_basis("torus", 2),
            Ok(vec![Some(1.0), Some(1.0)])
        );
        assert_eq!(
            coordinate_periods_for_basis("cylinder", 2),
            Ok(vec![Some(1.0), None])
        );
        assert_eq!(
            coordinate_periods_for_basis("sphere", 2),
            Ok(vec![None, Some(std::f64::consts::TAU)])
        );
        assert_eq!(
            coordinate_periods_for_basis("mobius-band", 2),
            Ok(vec![Some(2.0), None])
        );
        assert!(coordinate_periods_for_basis("mobius", 1).is_err());
    }

    #[test]
    fn flat_block_gating_uses_rust_owned_assignment_tokens() {
        assert_eq!(flat_block_assignment("norm_selection"), Ok("ibp_map"));
        assert_eq!(flat_block_assignment("separate_gate"), Ok("threshold_gate"));
        assert!(flat_block_assignment("norm-selection").is_err());
        assert!(flat_block_assignment(" Separate-Gate ").is_err());
        assert!(flat_block_assignment("unknown").is_err());
    }

    #[test]
    fn stagewise_payload_is_built_from_the_rust_schema() {
        let cfg = StagewisePayloadConfig {
            assignment: "softmax".to_string(),
            assignment_label: "softmax".to_string(),
            alpha: 1.0,
            learnable_alpha: false,
            tau: 0.5,
            sparsity_strength: 1.0,
            smoothness: 2.0,
            learning_rate: 1.0,
            max_iter: 20,
            random_state: 7,
            jumprelu_threshold: 0.0,
        };
        let payload = build_stagewise_manifold_sae_payload(
            vec!["periodic".to_string(), "euclidean".to_string()],
            vec![
                array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                array![[1.0, 1.0], [0.5, -0.5]],
            ],
            vec![1, 1],
            vec![array![[0.1], [0.2]], array![[0.3], [0.4]]],
            array![[0.75, 0.25], [0.4, 0.6]],
            array![[1.0, 2.0], [3.0, 4.0]],
            array![[1.0, 0.0], [0.0, 1.0]],
            array![[0.0, 2.0], [4.0, 6.0]],
            0.8,
            &cfg,
        )
        .expect("stagewise payload");

        assert_eq!(payload.schema, SCHEMA_TAG);
        assert_eq!(payload.atom_topology, "mixed");
        assert_eq!(payload.atom_topologies, vec!["circle", "euclidean"]);
        assert_eq!(payload.training_mean, vec![2.0, 4.0]);
        assert_eq!(payload.basis_sizes, vec![3, 2]);
        assert_eq!(payload.n_harmonics, vec![1, 0]);
        assert_eq!(payload.atoms[0].assignments, vec![0.75, 0.4]);
        assert_eq!(payload.primitive_names, vec!["sae_manifold_fit_stagewise"]);
        assert!(payload.penalized_loss_score.is_none());
        assert!(payload.fisher_factors.is_none());
    }

    #[test]
    fn basis_to_topology_matches_python_alias_map() {
        // Every documented alias -> canonical topology, plus alias/casing forms.
        for (basis, topo) in [
            ("periodic", "circle"),
            ("periodic_spline", "circle"),
            ("Circle", "circle"),
            ("sphere", "sphere"),
            ("torus", "torus"),
            ("linear", "linear"),
            ("linear_rank1", "linear"),
            ("affine", "linear"),
            ("linear_block", "linear_block"),
            ("flat-block", "linear_block"),
            ("duchon", "euclidean"),
            ("euclidean", "euclidean"),
            ("euclidean_patch", "euclidean"),
            ("euclidean_quadratic_patch", "euclidean"),
            ("poincare", "poincare"),
            ("hyperbolic", "poincare"),
            ("poincare_patch", "poincare"),
            ("cylinder", "cylinder"),
            ("mobius-band", "mobius"),
            (" AUTO ", "auto"),
        ] {
            assert_eq!(basis_to_topology(basis), topo, "basis {basis}");
        }
    }

    #[test]
    fn basis_to_topology_unknown_passes_through_original_string() {
        // The round-trip label conversion returns the raw argument on a miss.
        assert_eq!(basis_to_topology("Weird-Kind"), "Weird-Kind");
    }

    #[test]
    fn topology_for_bases_common_vs_mixed() {
        assert_eq!(
            topology_for_bases(&["periodic".into(), "circle".into()]),
            Some("circle".to_string()),
        );
        assert_eq!(
            topology_for_bases(&["periodic".into(), "euclidean".into()]),
            Some("mixed".to_string()),
        );
        assert_eq!(topology_for_bases(&[]), None);
        assert_eq!(
            topologies_for_bases(&["duchon".into(), "linear".into()]),
            vec!["euclidean".to_string(), "linear".to_string()],
        );
    }

    #[test]
    fn periodic_shape_band_reorder_stable_ascending_rows() {
        // coords (G,1); mean/sd (G,p) — the ROWS reindex by the coord order.
        let coords = array![[0.9], [0.1], [0.5], [0.1]];
        let mean = array![[9.0, 90.0], [1.0, 10.0], [5.0, 50.0], [2.0, 20.0]];
        let sd = array![[0.9, 0.09], [0.1, 0.01], [0.5, 0.05], [0.2, 0.02]];
        let (c, m, s) =
            periodic_shape_band_reorder(Some(coords), Some(mean), Some(sd)).expect("reorder");
        // Ascending by coord; the two 0.1 rows keep input order (stable): idx 1 then 3.
        assert_eq!(c.unwrap(), array![[0.1], [0.1], [0.5], [0.9]]);
        assert_eq!(
            m.unwrap(),
            array![[1.0, 10.0], [2.0, 20.0], [5.0, 50.0], [9.0, 90.0]]
        );
        assert_eq!(
            s.unwrap(),
            array![[0.1, 0.01], [0.2, 0.02], [0.5, 0.05], [0.9, 0.09]]
        );
    }

    #[test]
    fn periodic_shape_band_reorder_drops_band_without_mean() {
        let coords = array![[0.1], [0.2]];
        let (c, m, s) = periodic_shape_band_reorder(Some(coords), None, None).expect("ok");
        assert!(c.is_none() && m.is_none() && s.is_none());
        let (c2, m2, s2) =
            periodic_shape_band_reorder(None, Some(array![[1.0, 2.0]]), None).expect("ok");
        assert!(c2.is_none() && m2.is_none() && s2.is_none());
    }

    #[test]
    fn periodic_shape_band_reorder_rejects_multicolumn_coords() {
        let coords = array![[0.1, 0.2], [0.3, 0.4]];
        assert!(
            periodic_shape_band_reorder(Some(coords), Some(array![[1.0], [2.0]]), None).is_err()
        );
    }

    #[test]
    fn preserve_linear_block_relabels_declared_linear() {
        let mut basis_kinds = vec!["linear".to_string(), "periodic".to_string()];
        let mut topos = vec!["linear".to_string(), "circle".to_string()];
        let bases = vec!["linear_block".to_string(), "periodic".to_string()];
        preserve_linear_block_labels(&mut basis_kinds, &mut topos, &bases);
        // Declared-block + fitted-linear atom relabels; the periodic atom is left.
        assert_eq!(
            basis_kinds,
            vec!["linear_block".to_string(), "periodic".to_string()]
        );
        assert_eq!(
            topos,
            vec!["linear_block".to_string(), "circle".to_string()]
        );
    }

    #[test]
    fn preserve_linear_block_noops() {
        // No block declared -> unchanged.
        let mut bk = vec!["linear".to_string()];
        let mut tp = vec!["linear".to_string()];
        preserve_linear_block_labels(&mut bk, &mut tp, &["linear".to_string()]);
        assert_eq!(bk, vec!["linear".to_string()]);
        // Declared block but the fit RETYPED to a non-linear topology -> keep it.
        let mut bk2 = vec!["periodic".to_string()];
        let mut tp2 = vec!["circle".to_string()];
        preserve_linear_block_labels(&mut bk2, &mut tp2, &["linear_block".to_string()]);
        assert_eq!(bk2, vec!["periodic".to_string()]);
        // Length mismatch (evidence-gated K change) -> unchanged.
        let mut bk3 = vec!["linear".to_string()];
        let mut tp3 = vec!["linear".to_string()];
        preserve_linear_block_labels(
            &mut bk3,
            &mut tp3,
            &["linear_block".to_string(), "linear_block".to_string()],
        );
        assert_eq!(bk3, vec!["linear".to_string()]);
    }

    #[test]
    fn periodic_shape_band_reorder_rejects_row_mismatch() {
        let coords = array![[0.1], [0.2], [0.3]];
        // mean has 2 rows but coords has 3 -> error (Python would index-error).
        assert!(
            periodic_shape_band_reorder(Some(coords), Some(array![[1.0], [2.0]]), None).is_err()
        );
    }

    #[test]
    fn json_value_cannot_hold_nonfinite_and_parser_rejects_nan() {
        // Pins the existing-path behavior py_any_to_json_value's NaN/Inf policy
        // matches: serde_json::Value cannot represent non-finite numbers, and the
        // json.dumps -> ManifoldSaePayload::from_json path rejects the bare
        // `NaN` / `Infinity` literals json.dumps emits, so a non-finite report
        // value never round-trips through the Rust schema -> the helper Nulls it.
        assert!(serde_json::Number::from_f64(f64::NAN).is_none());
        assert!(serde_json::Number::from_f64(f64::INFINITY).is_none());
        assert!(serde_json::Number::from_f64(f64::NEG_INFINITY).is_none());
        assert!(serde_json::from_str::<Value>("NaN").is_err());
        assert!(serde_json::from_str::<Value>("Infinity").is_err());
        assert!(serde_json::from_str::<Value>("-Infinity").is_err());
    }
}

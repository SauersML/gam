//! Rust-owned conversion from the native fit report to the persisted
//! `ManifoldSaePayload` schema. The public fit path returns a Rust-owned
//! `ManifoldSaeCore` directly, with no Python model or schema in the middle.
//!
//! This module owns assignment tokens, topology naming, periodic shape-band
//! ordering, and `RawFitPayload -> ManifoldSaePayload` assembly. Python only
//! marshals typed inputs to this owner.

use crate::manifold::manifold_sae_payload::{
    AtomPayload, CrosscoderPayload, ManifoldSaePayload, SCHEMA_TAG,
};
use ndarray::{Array2, Array3, ArrayView2};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyTuple};
use serde_json::Value;

// The strict token schema (basis/topology vocabulary, assignment tables,
// harmonic validation, chart periods) lives in the library
// (`gam_sae::atom_schema`) so the CLI, Rust users, and this binding share one
// vocabulary.
pub(crate) use gam::terms::sae::atom_schema::canonical_assignment_kind;
use gam::terms::sae::atom_schema::{
    flat_block_assignment, topologies_for_bases, topology_for_bases, validated_n_harmonics,
};

/// Column mean `x.mean(axis=0)` -> `(P,)` — the training-mean centering vector
/// used by the native fitted-artifact builder. `n == 0` yields zeros.
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
/// Non-finite floats are rejected. A converged fit artifact cannot silently
/// replace a failed diagnostic with JSON `null`, and `serde_json::Value` cannot
/// represent NaN or infinity in any case.
pub(crate) fn py_any_to_json_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    // NumPy arrays with a single element can be extracted as Rust scalars.
    // Preserve their rank by applying numpy's own `.tolist()` conversion before
    // the scalar extractors below. NumPy scalar objects also expose `.tolist()`;
    // those correctly become native Python scalars and recurse once.
    if obj.hasattr("tolist")? {
        let listed = obj.call_method0("tolist")?;
        return py_any_to_json_value(&listed);
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::Number(i.into()));
    }
    if let Ok(f) = obj.extract::<f64>() {
        let number = serde_json::Number::from_f64(f).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "SAE fit report contains a non-finite floating-point value",
            )
        })?;
        return Ok(Value::Number(number));
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

/// Python bridge for [`flat_block_assignment`].
#[pyfunction]
pub(crate) fn sae_flat_block_assignment(gating: &str) -> PyResult<String> {
    flat_block_assignment(gating)
        .map(str::to_string)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Restore the `linear_block` label a flat-block fit reports as the generic
/// `linear`: for each atom the caller
/// DECLARED as `linear_block` AND that the fit left at topology
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
    let want: Vec<bool> = bases.iter().map(|b| b == "linear_block").collect();
    if !want.iter().any(|&w| w) {
        return;
    }
    for i in 0..atom_topologies.len() {
        if want[i] && atom_topologies[i] == "linear" {
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
    pub(crate) threshold_gate_threshold: f64,
    /// The retained WP-D output-Fisher shard `U` `(n, p, r)`; `None` for a
    /// Euclidean fit. When present, `metric_provenance` still comes from the raw
    /// solver payload (the Rust fit stamped `"OutputFisher"`).
    pub(crate) fisher_factors: Option<Vec<Vec<Vec<f64>>>>,
    /// The shard's provenance tag (`shard[2]`); serialized only when
    /// `fisher_factors` is present, mirroring `to_dict`.
    pub(crate) fisher_provenance: Option<String>,
    /// Required operator status for the retained factor stack (#2249).
    pub(crate) fisher_factor_kind: Option<String>,
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
/// native raw fit payload (as a `serde_json::Value`), the
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
        for (row_index, row) in arr.iter().enumerate() {
            if row.len() != k {
                return Err(format!(
                    "'{name}' must be rectangular with K={k} columns; row {row_index} has {}",
                    row.len()
                ));
            }
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
    let n_harmonics = validated_n_harmonics(&kinds, &raw_nharm, &widths)?;
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
        let assignments_col: Vec<f64> = assignments.iter().map(|row| row[idx]).collect();
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
    primitive_names.push("rust_module.sae_manifold_fit_model".to_string());
    primitive_names.extend(cfg.penalties.iter().cloned());

    // `basis_specs` keeps the fitted kinds; public labels may be restored to
    // `linear_block` for a flat-block fit.
    let mut basis_kinds = kinds.clone();
    let mut atom_topologies = topologies_for_bases(&kinds)?;
    if let Some(bases) = cfg.declared_bases.as_deref() {
        preserve_linear_block_labels(&mut basis_kinds, &mut atom_topologies, bases);
    }
    let atom_topology = topology_for_bases(&basis_kinds)?
        .ok_or_else(|| "converged SAE payload contains no atoms".to_string())?;
    let metric_provenance = vstr(raw, "metric_provenance")?;
    let tier0_scale = vopt(raw, "tier0_scale").map(v_arr1).transpose()?;
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
    let structured_residual_diagnostics: Vec<Value> =
        serde_json::from_value(vget(raw, "structured_residual_diagnostics")?.clone())
            .map_err(|error| format!("invalid structured_residual_diagnostics payload: {error}"))?;
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
        threshold_gate_threshold: cfg.threshold_gate_threshold,
        oos_projection_top1: vbool(raw, "oos_projection_top1")?,
        dispersion: vf64(raw, "dispersion")?,
        penalized_loss_score: score,
        penalized_quasi_laplace_criterion: vf64(raw, "penalized_quasi_laplace_criterion")?,
        reconstruction_r2: vf64(raw, "reconstruction_r2")?,
        primitive_names,
        // basis_specs = fitted kinds (unpatched); basis_kinds = post-relabel.
        basis_specs: kinds,
        basis_kinds,
        atom_dims: dims,
        basis_sizes: sizes,
        n_harmonics,
        training_mean,
        tier0_scale,
        fitted: v_arr2(vget(raw, "fitted")?)?,
        assignments,
        low_level_logits: logits,
        coords,
        decoder_blocks,
        duchon_centers,
        crosscoder,
        atoms: atom_payloads,
        diagnostics: vget(raw, "diagnostics")?.clone(),
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
        fisher_factor_kind: if cfg.fisher_factors.is_some() {
            cfg.fisher_factor_kind.clone()
        } else {
            None
        },
        metric_provenance,
        fisher_mass_residual,
        selected_log_lambda_sparse,
        selected_log_lambda_smooth,
        selected_log_ard,
        structured_residual_diagnostics,
        // #2235 — termination ledger carried straight from the raw fit payload
        // into the persisted v5 artifact.
        termination: report("termination"),
    })
}

#[cfg(test)]
mod manifold_sae_coercion_tests {
    use super::*;
    use gam::terms::sae::atom_schema::{
        basis_kind_for_topology, basis_to_topology, canonical_topology,
        coordinate_periods_for_basis,
    };
    use ndarray::array;

    #[test]
    fn assignment_tokens_have_one_strict_parser() {
        for canonical in [
            "softmax",
            "ordered_beta_bernoulli",
            "threshold_gate",
            "topk",
        ] {
            assert_eq!(
                canonical_assignment_kind(canonical),
                Ok(canonical),
                "canonical token {canonical}"
            );
        }
        for rejected in ["unknown"] {
            let err = canonical_assignment_kind(rejected).expect_err("alias must be rejected");
            assert!(err.contains("not a recognized assignment kind"));
            assert!(err.contains("ordered_beta_bernoulli") && err.contains("threshold_gate"));
        }
    }

    #[test]
    fn topology_and_basis_tokens_are_strict() {
        assert_eq!(
            basis_kind_for_topology("circle"),
            Ok("periodic".to_string())
        );
        assert_eq!(basis_kind_for_topology("auto"), Ok("auto".to_string()));
        for removed in [
            "Periodic-Spline",
            "affine",
            "flat-block",
            "euclidean_quadratic_patch",
            "hyperbolic",
            "mobius-band",
            " AUTO ",
            "Weird-Kind",
        ] {
            assert!(
                basis_kind_for_topology(removed).is_err(),
                "removed {removed}"
            );
            assert!(gam::terms::sae::atom_schema::validate_seed_basis_kind(removed).is_err());
        }
        assert_eq!(canonical_topology("circle"), Ok("circle".to_string()));
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
            coordinate_periods_for_basis("mobius", 2),
            Ok(vec![Some(2.0), None])
        );
        assert!(coordinate_periods_for_basis("mobius", 1).is_err());
    }

    #[test]
    fn flat_block_gating_uses_rust_owned_assignment_tokens() {
        assert_eq!(
            flat_block_assignment("norm_selection"),
            Ok("ordered_beta_bernoulli")
        );
        assert_eq!(flat_block_assignment("separate_gate"), Ok("threshold_gate"));
        assert!(flat_block_assignment("norm-selection").is_err());
        assert!(flat_block_assignment(" Separate-Gate ").is_err());
        assert!(flat_block_assignment("unknown").is_err());
    }

    #[test]
    fn basis_to_topology_maps_only_canonical_native_kinds() {
        for (basis, topo) in [
            ("periodic", "circle"),
            ("sphere", "sphere"),
            ("torus", "torus"),
            ("linear", "linear"),
            ("linear_block", "linear_block"),
            ("duchon", "euclidean"),
            ("euclidean", "euclidean"),
            ("poincare", "poincare"),
            ("cylinder", "cylinder"),
            ("mobius", "mobius"),
            ("finite_set", "finite_set"),
        ] {
            assert_eq!(
                basis_to_topology(basis),
                Ok(topo.to_string()),
                "basis {basis}"
            );
        }
        assert!(basis_to_topology("Weird-Kind").is_err());
    }

    #[test]
    fn topology_for_bases_common_vs_mixed() {
        assert_eq!(
            topology_for_bases(&["periodic".into(), "periodic".into()]),
            Ok(Some("circle".to_string())),
        );
        assert_eq!(
            topology_for_bases(&["periodic".into(), "euclidean".into()]),
            Ok(Some("mixed".to_string())),
        );
        assert_eq!(topology_for_bases(&[]), Ok(None));
        assert_eq!(
            topologies_for_bases(&["duchon".into(), "linear".into()]),
            Ok(vec!["euclidean".to_string(), "linear".to_string()]),
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
        // The binding rejects these values instead of manufacturing JSON nulls.
        // Pin the underlying serde limitation that requires that policy.
        assert!(serde_json::Number::from_f64(f64::NAN).is_none());
        assert!(serde_json::Number::from_f64(f64::INFINITY).is_none());
        assert!(serde_json::Number::from_f64(f64::NEG_INFINITY).is_none());
        assert!(serde_json::from_str::<Value>("NaN").is_err());
        assert!(serde_json::from_str::<Value>("Infinity").is_err());
        assert!(serde_json::from_str::<Value>("-Infinity").is_err());
    }
}

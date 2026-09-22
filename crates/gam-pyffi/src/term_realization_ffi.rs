//! One smooth term's penalties, realized by the term builder that fits it.
//!
//! SPEC: the CLI, the Python library and the Rust library have a single source
//! of truth. `gamfit/torch/fit.py` builds its own design for a smooth, because
//! the design is what a torch caller differentiates through and only torch can
//! carry that autograd. Its PENALTY is not a torch quantity at all: it is a
//! structural property of the realized basis, and the engine already decides
//! it. Building a second one in Python produced a different model under the
//! same spec — `I ⊗ S_a ⊗ I` under one λ where the tensor builder emits one
//! candidate per margin measured by its neighbours' function Grams, and the raw
//! kernel Gram `K_cc` where the Matérn builder emits ν-gated collocation
//! operator candidates through an identifiability chart (#4492).
//!
//! This entry returns what `gamfit.fit` would realize for the same term: the
//! coefficient chart the term's design is expressed in, and the penalty of each
//! smoothing parameter the fit would carry. It runs the fit's own lowering —
//! `parse_formula` → `build_termspec` → `apply_smooth_overrides` →
//! `build_term_collection_design` — so the two paths cannot disagree about a
//! penalty without disagreeing about a fit.

use crate::ffi::ffi_errors::detach_py_result;
use gam::data::{ColumnKindTag, DataSchema, EncodedDataset, SchemaColumn};
use gam::terms::inference::formula_dsl::parse_formula;
use gam::terms::smooth::build_term_collection_design;
use gam::terms::smooth_overrides::apply_smooth_overrides;
use gam::terms::term_builder::build_termspec;
use ndarray::{Array2, s};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::collections::HashMap;

/// The covariate columns this entry names, and the response the formula needs.
///
/// A caller passes points, not a data frame, so the columns are named here.
/// `x0 … x{d-1}` are the covariate axes in the order the points' columns carry
/// them; `__gamfit_points_response` exists because a formula has a left-hand
/// side, and it is never read — `build_termspec` resolves term columns only.
const RESPONSE_COLUMN: &str = "__gamfit_points_response";

fn axis_column(axis: usize) -> String {
    format!("x{axis}")
}

/// The realized design block and penalties of the single smooth term `term`
/// names, evaluated at `points`.
///
/// `points` is `N × d`, one column per covariate axis. `term` is the formula
/// term as the DSL writes it over those axes, e.g. `te(x0, x1)` or
/// `matern(x0)`. `descriptor_json` is the object
/// `Smooth.to_rust_descriptor()` produces, the same payload
/// `gamfit.fit(..., smooths={...})` sends.
///
/// Returns `(design, penalties)`:
///
/// * `design` is `N × p`, the term's OWN columns in the collection's layout.
///   It is the chart the penalties are expressed in, already carrying the
///   term's identifiability constraint against the collection's intercept, so a
///   caller that rebuilds the raw basis itself must map its coefficients
///   through this block rather than assume the raw columns.
/// * `penalties` has one `p × p` matrix per smoothing parameter the fit would
///   carry for this term, in the order the fit carries them. A tensor term
///   returns one per margin; a Matérn term returns one per active operator
///   dial. Each is the term's realized `BlockwisePenalty`, embedded at its own
///   column range inside the block, so `βᵀ S_k β` is the quantity the fit
///   prices and no caller re-derives a range.
///
/// The design is returned rather than assumed because the penalties are only
/// meaningful in its chart. A caller that wants an autograd path builds its own
/// design over the same basis and uses this one to check that the two charts
/// agree; a caller that does not simply uses this one.
#[pyfunction(signature = (points, term, descriptor_json))]
fn smooth_term_realized_penalties<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    term: &str,
    descriptor_json: &str,
) -> PyResult<(Py<PyArray2<f64>>, Vec<Py<PyArray2<f64>>>)> {
    let owned = points.as_array().to_owned();
    let term = term.to_string();
    let descriptor_json = descriptor_json.to_string();
    let (design, penalties) = detach_py_result(py, "smooth_term_realized_penalties", move || {
        realize_smooth_term(owned, &term, &descriptor_json)
    })?;
    Ok((
        design.into_pyarray(py).unbind(),
        penalties
            .into_iter()
            .map(|penalty| penalty.into_pyarray(py).unbind())
            .collect(),
    ))
}

fn realize_smooth_term(
    points: Array2<f64>,
    term: &str,
    descriptor_json: &str,
) -> Result<(Array2<f64>, Vec<Array2<f64>>), String> {
    let (rows, dim) = points.dim();
    if rows == 0 || dim == 0 {
        return Err(format!(
            "smooth_term_realized_penalties needs a non-empty N x d points array, got {rows}x{dim}"
        ));
    }
    if points.iter().any(|value| !value.is_finite()) {
        return Err(
            "smooth_term_realized_penalties: points contains a non-finite value".to_string(),
        );
    }

    // The dataset the fit's lowering reads: the response the formula's
    // left-hand side names, then the covariate axes. The response column is
    // zero because no term consumes it and `build_termspec` resolves term
    // columns only; it exists so the table is well formed rather than relying
    // on an unresolved name.
    let mut values = Array2::<f64>::zeros((rows, dim + 1));
    values.slice_mut(s![.., 1..]).assign(&points);
    let mut headers = Vec::with_capacity(dim + 1);
    headers.push(RESPONSE_COLUMN.to_string());
    for axis in 0..dim {
        headers.push(axis_column(axis));
    }
    let column_kinds = vec![ColumnKindTag::Continuous; dim + 1];
    let schema = DataSchema {
        columns: headers
            .iter()
            .map(|name| SchemaColumn {
                name: name.clone(),
                kind: ColumnKindTag::Continuous,
                levels: Vec::new(),
            })
            .collect(),
    };
    let data = EncodedDataset {
        headers: headers.clone(),
        values,
        schema,
        column_kinds,
    };
    let column_map: HashMap<String, usize> = headers
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();

    let formula = format!("{RESPONSE_COLUMN} ~ {term}");
    // `FormulaDslError` carries no `Display`, so the debug form is the error's
    // own text rather than a paraphrase of it.
    let parsed = parse_formula(&formula).map_err(|error| {
        format!("smooth_term_realized_penalties: formula {formula:?}: {error:?}")
    })?;
    let mut notes: Vec<String> = Vec::new();
    let mut spec = build_termspec(&parsed.terms, &data, &column_map, &mut notes)
        .map_err(|error| format!("smooth_term_realized_penalties: {error}"))?;
    if spec.smooth_terms.len() != 1 {
        return Err(format!(
            "smooth_term_realized_penalties takes one smooth term, but {term:?} lowered to {} \
             smooth terms",
            spec.smooth_terms.len()
        ));
    }

    // The descriptor registry is keyed by the comma-joined covariate names, the
    // form `gamfit._api._normalize_smooths` sends and `apply_smooth_overrides`
    // matches on. Every tunable the Python spec carries reaches the term
    // through this one path, so an override that would change the fit changes
    // the penalties here too.
    let descriptor: serde_json::Value = serde_json::from_str(descriptor_json).map_err(|error| {
        format!("smooth_term_realized_penalties: descriptor is not JSON: {error}")
    })?;
    if !descriptor.is_object() {
        return Err("smooth_term_realized_penalties: descriptor must be a JSON object".to_string());
    }
    let key = (0..dim).map(axis_column).collect::<Vec<String>>().join(",");
    let mut registry = serde_json::Map::new();
    registry.insert(key, descriptor);
    apply_smooth_overrides(
        &mut spec,
        &serde_json::Value::Object(registry),
        &data,
        &mut notes,
    )
    .map_err(|reason| format!("smooth_term_realized_penalties: {reason}"))?;

    let realized = build_term_collection_design(data.values.view(), &spec)
        .map_err(|error| format!("smooth_term_realized_penalties: {error}"))?;
    let layout = realized.column_layout();
    let (term_name, block) = layout.smooth_ranges.first().ok_or_else(|| {
        "smooth_term_realized_penalties: the realized collection carries no smooth block"
            .to_string()
    })?;
    let width = block.end - block.start;
    if width == 0 {
        return Err(format!(
            "smooth_term_realized_penalties: the realized smooth block for term \
             {term_name:?} has no columns"
        ));
    }

    let dense = realized.design.to_dense();
    let design = dense.slice(s![.., block.clone()]).to_owned();

    // One `p × p` penalty per smoothing parameter, each embedded at its own
    // column range inside the block. A penalty outside the block belongs to
    // another term and cannot appear: the collection holds one smooth term and
    // the formula adds no penalized parametric block.
    let mut penalties = Vec::with_capacity(realized.penalties.len());
    for penalty in &realized.penalties {
        let range = &penalty.col_range;
        if range.start < block.start || range.end > block.end {
            return Err(format!(
                "smooth_term_realized_penalties: a realized penalty covers columns \
                 {}..{} outside the smooth block {}..{}",
                range.start, range.end, block.start, block.end
            ));
        }
        let start = range.start - block.start;
        let end = range.end - block.start;
        let mut embedded = Array2::<f64>::zeros((width, width));
        embedded
            .slice_mut(s![start..end, start..end])
            .assign(&penalty.local);
        penalties.push(embedded);
    }
    if penalties.is_empty() {
        return Err(format!(
            "smooth_term_realized_penalties: {term:?} realized no penalty, so it carries no \
             smoothing parameter"
        ));
    }
    Ok((design, penalties))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(smooth_term_realized_penalties, module)?)?;
    Ok(())
}

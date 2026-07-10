//! Survival-surface interpolation and CSV/IO FFI helpers.
//!
//! This is a self-contained seam extracted from the pyffi monolith (issue
//! #780): a cohesive group of `#[pyfunction]`s that interpolate survival
//! surfaces along a time grid, reshape/validate survival parameter matrices,
//! convert between survival / cumulative-hazard / hazard representations, and
//! stream interpolated surfaces to CSV. Surface math and resource policy live
//! in `gam-models`; this module only converts numpy buffers and performs the
//! external CSV I/O.

use std::fs::File;
use std::io::{BufWriter, Write};

use gam::families::survival::{
    SurvivalSurface, SurvivalSurfaceChunkPolicy, SurvivalSurfaceKind,
    failure_probability_from_survival,
};
use ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use crate::py_value_error;

#[pyfunction]
pub(crate) fn interpolate_rows<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray1<'py, f64>,
    surface: PyReadonlyArray2<'py, f64>,
    query: PyReadonlyArray1<'py, f64>,
    kind: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let kind = SurvivalSurfaceKind::parse(kind).map_err(py_value_error)?;
    let surface =
        SurvivalSurface::new(kind, grid.as_array(), surface.as_array()).map_err(py_value_error)?;
    let out = surface
        .interpolate(query.as_array())
        .map_err(py_value_error)?;
    Ok(out.into_pyarray(py))
}

fn write_csv_field(writer: &mut BufWriter<File>, value: &str) -> std::io::Result<()> {
    let needs_quotes = value
        .bytes()
        .any(|byte| matches!(byte, b',' | b'"' | b'\n' | b'\r'));
    if !needs_quotes {
        writer.write_all(value.as_bytes())?;
        return Ok(());
    }
    writer.write_all(b"\"")?;
    for byte in value.bytes() {
        if byte == b'"' {
            writer.write_all(b"\"\"")?;
        } else {
            writer.write_all(&[byte])?;
        }
    }
    writer.write_all(b"\"")
}

/// Expose the resource-derived chunking policy to Python as
/// `(people_chunk, time_grid_chunk, dense_auto_chunk_cells)`.
#[pyfunction]
pub(crate) fn survival_chunk_defaults() -> (usize, usize, usize) {
    let policy = SurvivalSurfaceChunkPolicy::default();
    let (people, times) = policy.default_shape();
    (people, times, policy.target_cells())
}

#[pyfunction]
pub(crate) fn survival_should_chunk(n_rows: usize, n_times: usize) -> bool {
    SurvivalSurfaceChunkPolicy::default().should_chunk(n_rows, n_times)
}

#[pyfunction]
#[pyo3(signature = (n_rows, n_times, people_chunk = None, time_grid_chunk = None))]
pub(crate) fn survival_chunk_ranges(
    n_rows: usize,
    n_times: usize,
    people_chunk: Option<usize>,
    time_grid_chunk: Option<usize>,
) -> PyResult<Vec<(usize, usize, usize, usize)>> {
    SurvivalSurfaceChunkPolicy::default()
        .chunks(n_rows, n_times, people_chunk, time_grid_chunk)
        .map(|chunks| {
            chunks
                .into_iter()
                .map(|chunk| {
                    (
                        chunk.row_start,
                        chunk.row_end,
                        chunk.time_start,
                        chunk.time_end,
                    )
                })
                .collect()
        })
        .map_err(py_value_error)
}

#[pyfunction]
pub(crate) fn survival_chunk_iter_collect<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray1<'py, f64>,
    surface: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
    kind: &str,
    people_chunk: usize,
    time_grid_chunk: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let kind = SurvivalSurfaceKind::parse(kind).map_err(py_value_error)?;
    let grid = grid.as_array().to_owned();
    let values = surface.as_array().to_owned();
    let times = times.as_array().to_owned();
    let out = py
        .detach(move || {
            let surface = SurvivalSurface::new(kind, grid.view(), values.view())?;
            surface.interpolate_chunked(
                times.view(),
                SurvivalSurfaceChunkPolicy::default(),
                Some(people_chunk),
                Some(time_grid_chunk),
            )
        })
        .map_err(py_value_error)?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
pub(crate) fn write_survival_csv(
    py: Python<'_>,
    path: &str,
    grid: PyReadonlyArray1<'_, f64>,
    surface: PyReadonlyArray2<'_, f64>,
    times: PyReadonlyArray1<'_, f64>,
    id_column: Option<String>,
    row_ids: Option<Vec<String>>,
    people_chunk: usize,
    time_grid_chunk: usize,
) -> PyResult<String> {
    let grid = grid.as_array().to_owned();
    let values = surface.as_array().to_owned();
    let times = times.as_array().to_owned();
    let n_rows = values.nrows();
    if id_column.is_some() {
        match row_ids.as_ref() {
            Some(ids) if ids.len() >= n_rows => {}
            Some(ids) => {
                return Err(py_value_error(format!(
                    "row_ids length {} is smaller than survival row count {n_rows}",
                    ids.len()
                )));
            }
            None => {
                return Err(py_value_error(
                    "row_ids are required when id_column is set".to_string(),
                ));
            }
        }
    }
    let path_owned = path.to_string();
    py.detach(move || -> Result<String, String> {
        let surface =
            SurvivalSurface::new(SurvivalSurfaceKind::Survival, grid.view(), values.view())?;
        let chunks = SurvivalSurfaceChunkPolicy::default().chunks(
            surface.nrows(),
            times.len(),
            Some(people_chunk),
            Some(time_grid_chunk),
        )?;
        let file = File::create(&path_owned)
            .map_err(|error| format!("failed to create survival CSV '{path_owned}': {error}"))?;
        let mut writer = BufWriter::new(file);
        match id_column.as_ref() {
            Some(column) => {
                writer
                    .write_all(b"row,")
                    .map_err(|error| format!("failed to write survival CSV: {error}"))?;
                write_csv_field(&mut writer, column)
                    .map_err(|error| format!("failed to write survival CSV: {error}"))?;
                writer
                    .write_all(b",time,survival\n")
                    .map_err(|error| format!("failed to write survival CSV: {error}"))?;
            }
            None => writer
                .write_all(b"row,time,survival\n")
                .map_err(|error| format!("failed to write survival CSV: {error}"))?,
        }

        for chunk in chunks {
            for row_idx in chunk.row_start..chunk.row_end {
                for time_index in chunk.time_start..chunk.time_end {
                    let query_value = times[time_index];
                    let survival = surface.value_at(row_idx, query_value)?;
                    match (id_column.as_ref(), row_ids.as_ref()) {
                        (Some(_column), Some(ids)) => {
                            write!(writer, "{row_idx},").map_err(|error| {
                                format!("failed to write survival CSV: {error}")
                            })?;
                            write_csv_field(&mut writer, &ids[row_idx]).map_err(|error| {
                                format!("failed to write survival CSV: {error}")
                            })?;
                            writeln!(writer, ",{query_value},{survival}").map_err(|error| {
                                format!("failed to write survival CSV: {error}")
                            })?;
                        }
                        _ => writeln!(writer, "{row_idx},{query_value},{survival}")
                            .map_err(|error| format!("failed to write survival CSV: {error}"))?,
                    }
                }
            }
        }
        writer
            .flush()
            .map_err(|error| format!("failed to flush survival CSV: {error}"))?;
        Ok(path_owned)
    })
    .map_err(py_value_error)
}

#[pyfunction]
pub(crate) fn survival_coerce_times<'py>(
    py: Python<'py>,
    times: &Bound<'py, PyAny>,
) -> PyResult<Py<PyArray1<f64>>> {
    // ``times`` is documented as array-like (`Any`): a scalar, list, tuple, or
    // ndarray of any shape. Normalize through ``numpy.asarray`` so the Rust
    // core remains the single source of truth for coercion and validation,
    // matching the ``numeric_matrix_validate`` pattern used elsewhere in this
    // module. ``reshape(-1)`` flattens a 0-d scalar or N-d input to the 1-D
    // time grid the survival kernels expect.
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "float64")?;
    let array = np.call_method("asarray", (times,), Some(&kwargs))?;
    let flat = array.call_method1("reshape", (-1,))?;
    let typed = flat.cast::<PyArray1<f64>>().map_err(PyErr::from)?;
    let values: Vec<f64> = typed.readonly().as_array().iter().copied().collect();
    if values.is_empty() {
        return Err(py_value_error(
            "survival prediction requires at least one time".to_string(),
        ));
    }
    // This is the prediction-QUERY coercion (callers: `survival_at` /
    // `hazard_at` / `cumulative_hazard_at` / `competing_risks_cif`), NOT the
    // fit-time observed-time validator. The survival function `S(t) = P(T > t)`
    // is defined for every real `t`: below the support `S(t <= 0) = 1`, and in
    // the limit `S(+inf) = 0` (mirrored by `H(t <= 0) = 0`, `H(+inf) = +inf`).
    // So a negative or `+inf` query is a legitimate boundary evaluation, not an
    // error (issue #965): the earlier `finite && >= 0` reject conflated this
    // query path with the core fit-time grid validation
    // (`crates/gam-models/src/survival/predict.rs`), where rejecting negative
    // *observed* times is correct. The downstream interpolation/parametric
    // kernels apply the single boundary policy `t <= 0 => S = 1, H = 0` and the
    // explicit `+inf` asymptote, so neither the `exp(-hazard * t) > 1` value nor
    // a representation-dependent answer can arise. Only `NaN` is rejected: it
    // carries no order and no defined survival value.
    if let Some((idx, value)) = values.iter().enumerate().find(|(_, value)| value.is_nan()) {
        return Err(py_value_error(format!(
            "survival prediction times must not be NaN (index {idx} = {value})"
        )));
    }
    Ok(Array1::from_vec(values).into_pyarray(py).unbind())
}

#[pyfunction]
pub(crate) fn survival_parameters_matrix<'py>(
    py: Python<'py>,
    parameters: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let view = parameters.as_array();
    match view.ndim() {
        1 => {
            let n = view.shape()[0];
            let data: Vec<f64> = view.iter().copied().collect();
            let out = Array2::from_shape_vec((n, 1), data)
                .map_err(|err| py_value_error(format!("failed to reshape parameters: {err}")))?;
            Ok(out.into_pyarray(py).unbind())
        }
        2 => {
            let (rows, cols) = (view.shape()[0], view.shape()[1]);
            let data: Vec<f64> = view.iter().copied().collect();
            let out = Array2::from_shape_vec((rows, cols), data)
                .map_err(|err| py_value_error(format!("failed to reshape parameters: {err}")))?;
            Ok(out.into_pyarray(py).unbind())
        }
        other => Err(py_value_error(format!(
            "survival parameters must be 1D or 2D, got {other}D"
        ))),
    }
}

#[pyfunction]
pub(crate) fn hazard_from_cumulative_knots<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray1<'py, f64>,
    surface: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    hazard_from_cumulative_knots_impl(grid.as_array(), surface.as_array(), times.as_array())
        .map(|out| out.into_pyarray(py))
        .map_err(py_value_error)
}

/// Pointwise hazard of the STORED piecewise-linear cumulative-hazard surface:
/// the slope of the knot interval containing each query time.
///
/// The hazard at a time is a property of the saved `(grid, H)` interpolant
/// alone, so the answer for one query cannot depend on which other query
/// times happen to share the call. The retired approach interpolated `H` at
/// the queries and then took secants between consecutive *query* points; for
/// stored knots `(t, H) = (0, 0), (1, 1), (2, 4)` it returned hazard 2 at
/// `t = 1.5`, and inserting an unrelated query at `t = 1` changed that answer
/// to 3 (issue #966 follow-up). Here every query is answered from the knot
/// interval `(grid[k-1], grid[k]]` that contains it — left-continuous at
/// knots, matching a càdlàg cumulative hazard.
///
/// Outside the grid the interpolant is flat (`H = 0` below the first knot and
/// the #1595 flat clamp past the last), so the hazard there is exactly 0;
/// `t = +inf` falls in the same flat-clamp region.
fn hazard_from_cumulative_knots_impl(
    grid: ndarray::ArrayView1<'_, f64>,
    surface: ndarray::ArrayView2<'_, f64>,
    times: ndarray::ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    let (n_rows, n_knots) = surface.dim();
    if grid.len() != n_knots {
        return Err(format!(
            "hazard_from_cumulative_knots: surface has {n_knots} knot columns but the grid \
             has {} entries",
            grid.len()
        ));
    }
    if n_knots == 0 {
        return Err("hazard_from_cumulative_knots: empty knot grid".to_string());
    }
    for k in 1..n_knots {
        if !(grid[k] > grid[k - 1]) {
            return Err(format!(
                "hazard_from_cumulative_knots: knot grid must be strictly increasing; \
                 grid[{k}] = {} does not exceed grid[{}] = {}",
                grid[k],
                k - 1,
                grid[k - 1]
            ));
        }
    }
    let mut out = Array2::<f64>::zeros((n_rows, times.len()));
    for (j, &t) in times.iter().enumerate() {
        if t.is_nan() {
            return Err(format!(
                "hazard_from_cumulative_knots: query time at index {j} is NaN"
            ));
        }
        if !(t > grid[0] && t <= grid[n_knots - 1]) {
            // Flat extrapolation regions of the stored interpolant: slope 0.
            continue;
        }
        // Binary search for the knot interval (grid[lo], grid[hi]] with t
        // inside; the range check above guarantees 0 <= lo < hi < n_knots.
        let (mut lo, mut hi) = (0usize, n_knots - 1);
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if grid[mid] < t {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let width = grid[hi] - grid[lo];
        for i in 0..n_rows {
            let delta = surface[[i, hi]] - surface[[i, lo]];
            if delta < 0.0 {
                return Err(format!(
                    "cumulative hazard must be non-decreasing in time; row {i} drops by \
                     {} between knots {lo} and {hi}",
                    -delta
                ));
            }
            out[[i, j]] = delta / width;
        }
    }
    Ok(out)
}

/// Convert a survival probability to a cumulative hazard via `H = -ln S`.
///
/// This is exact, not floored: a valid but tiny survival probability (e.g.
/// `S = e^-100`) maps to the correspondingly large `H = 100`, instead of being
/// capped at `-ln(1e-12) = 27.63` as an earlier clamp did (issue #964). The
/// only special cases are the mathematically forced ones:
///   * `S = 0`   -> `H = +inf`     (`-ln 0 = +inf`, produced naturally),
///   * `S = NaN` -> `H = NaN`      (propagate undefined input),
///   * `S` outside `[0, 1]`        -> `Err` (not a survival probability).
/// Any flooring for display belongs in formatting, never in this S->H math.
fn cumulative_hazard_from_survival_value(s: f64) -> Result<f64, String> {
    if s.is_nan() {
        return Ok(f64::NAN);
    }
    if !(0.0..=1.0).contains(&s) {
        return Err(format!(
            "survival probability must lie in [0, 1] to convert to cumulative hazard, got {s}"
        ));
    }
    // `-ln S` is exact across the full representable range; `S = 0` yields the
    // mathematically correct `+inf`, `S = 1` yields `0`.
    Ok(-s.ln())
}

#[pyfunction]
pub(crate) fn survival_cumulative_from_survival<'py>(
    py: Python<'py>,
    survival: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let view = survival.as_array();
    let (n_rows, n_cols) = (view.shape()[0], view.shape()[1]);
    let mut out = Array2::<f64>::zeros((n_rows, n_cols));
    for i in 0..n_rows {
        for j in 0..n_cols {
            out[[i, j]] =
                cumulative_hazard_from_survival_value(view[[i, j]]).map_err(py_value_error)?;
        }
    }
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
pub(crate) fn survival_failure_from_survival<'py>(
    py: Python<'py>,
    survival: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let failure = failure_probability_from_survival(survival.as_array()).map_err(py_value_error)?;
    Ok(failure.into_pyarray(py).unbind())
}

/// Survival probability of the parametric exponential fallback at one time.
///
/// `S(t) = exp(-hazard * t)` for the constant-hazard exponential model, with
/// `S(0) = 1` enforced exactly. The explicit `t == 0` branch matters because
/// when `exp(log_hazard)` overflows to `+inf`, the naive `exp(-inf * 0)`
/// evaluates `(-inf * 0) = NaN`, then `exp(NaN) = NaN`, even though every
/// survival function satisfies `S(0) = 1` (issue #965).
fn exponential_survival_at(hazard: f64, t: f64) -> f64 {
    // `S(t) = P(T > t)` for the constant-hazard exponential fallback is defined
    // on all of R (issue #965): below the origin everyone is still at risk, so
    // `S(t <= 0) = 1` exactly (this also avoids the `exp(-hazard * t) > 1`
    // impossibility for `t < 0` and the `inf * 0 = NaN` case at `t = 0` when
    // `hazard` overflowed). At `t = +inf`, `S = 0` for any positive hazard.
    if t <= 0.0 {
        return 1.0;
    }
    if t == f64::INFINITY {
        // exp(-hazard * inf): 0 for hazard > 0, 1 for a degenerate zero hazard.
        return if hazard > 0.0 { 0.0 } else { 1.0 };
    }
    (-hazard * t).exp()
}

/// Cumulative hazard of the parametric exponential fallback at one time.
///
/// `H(t) = hazard * t` computed DIRECTLY, never reconstructed as
/// `-ln(exp(-hazard * t))`: that round trip returns 0 instead of ~1e-20 when
/// `S` rounds to 1 (hazard·t ≈ 1e-20) and `+inf` instead of 1000 when `S`
/// underflows to 0 (hazard·t ≳ 745), corrupting risk scores and concordance
/// ordering. Boundary cases mirror `exponential_survival_at`: `H(t <= 0) = 0`
/// exactly, and `H(+inf) = +inf` for a positive hazard (0 for a degenerate
/// zero hazard, avoiding `0 * inf = NaN`).
fn exponential_cumulative_hazard_at(hazard: f64, t: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    if t == f64::INFINITY {
        return if hazard > 0.0 { f64::INFINITY } else { 0.0 };
    }
    hazard * t
}

#[pyfunction]
pub(crate) fn survival_block_cumulative_hazard<'py>(
    py: Python<'py>,
    params: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let params = params.as_array();
    let times_view = times.as_array();
    let n_rows = params.shape()[0];
    let n_times = times_view.len();
    if params.shape()[1] == 0 {
        return Err(py_value_error(
            "survival parameter matrix must have at least one column".to_string(),
        ));
    }
    let mut out = Array2::<f64>::zeros((n_rows, n_times));
    for i in 0..n_rows {
        let hazard = params[[i, 0]].exp();
        for j in 0..n_times {
            out[[i, j]] = exponential_cumulative_hazard_at(hazard, times_view[j]);
        }
    }
    Ok(out.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn survival_block_failure<'py>(
    py: Python<'py>,
    params: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // Failure probability of the parametric exponential fallback,
    // `F(t) = -expm1(-H(t))` evaluated directly: `1 - exp(-H)` cancels to 0
    // for `H ≈ 1e-20` where the true failure probability is ~1e-20. The
    // `H = +inf` endpoint maps to `-expm1(-inf) = 1` exactly.
    let params = params.as_array();
    let times_view = times.as_array();
    let n_rows = params.shape()[0];
    let n_times = times_view.len();
    if params.shape()[1] == 0 {
        return Err(py_value_error(
            "survival parameter matrix must have at least one column".to_string(),
        ));
    }
    let mut out = Array2::<f64>::zeros((n_rows, n_times));
    for i in 0..n_rows {
        let hazard = params[[i, 0]].exp();
        for j in 0..n_times {
            out[[i, j]] = -(-exponential_cumulative_hazard_at(hazard, times_view[j])).exp_m1();
        }
    }
    Ok(out.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn survival_block<'py>(
    py: Python<'py>,
    params: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let params = params.as_array();
    let times_view = times.as_array();
    let n_rows = params.shape()[0];
    let n_times = times_view.len();
    if params.shape()[1] == 0 {
        return Err(py_value_error(
            "survival parameter matrix must have at least one column".to_string(),
        ));
    }
    let mut out = Array2::<f64>::zeros((n_rows, n_times));
    for i in 0..n_rows {
        let hazard = params[[i, 0]].exp();
        for j in 0..n_times {
            out[[i, j]] = exponential_survival_at(hazard, times_view[j]);
        }
    }
    Ok(out.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn survival_block_hazard<'py>(
    py: Python<'py>,
    params: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // Hazard of the parametric exponential fallback. For a constant-hazard
    // exponential model the hazard is `exp(log_hazard)` at every time, so we
    // return it directly rather than finite-differencing the sampled
    // cumulative hazard. Differencing query-ordered cumulatives invents zero
    // hazards at repeated times and negative hazards at unsorted times (issue
    // #966); the closed-form constant is exact and order-independent.
    let params = params.as_array();
    let times_view = times.as_array();
    let n_rows = params.shape()[0];
    let n_times = times_view.len();
    if params.shape()[1] == 0 {
        return Err(py_value_error(
            "survival parameter matrix must have at least one column".to_string(),
        ));
    }
    let mut out = Array2::<f64>::zeros((n_rows, n_times));
    for i in 0..n_rows {
        let hazard = params[[i, 0]].exp();
        for j in 0..n_times {
            // Hazard is constant in `t`; `times_view[j]` selects the column
            // only. (Times are already validated finite and non-negative.)
            out[[i, j]] = hazard;
        }
    }
    Ok(out.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn survival_ffi_surface<'py>(
    py: Python<'py>,
    times: PyReadonlyArrayDyn<'py, f64>,
    surface: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Option<(Py<PyArray1<f64>>, Py<PyArray2<f64>>)>> {
    let times_view = times.as_array();
    let grid: Vec<f64> = times_view.iter().copied().collect();
    if grid.is_empty() {
        return Ok(None);
    }
    let surf_view = surface.as_array();
    let (n_rows, n_cols) = match surf_view.ndim() {
        1 => (surf_view.shape()[0], 1usize),
        2 => (surf_view.shape()[0], surf_view.shape()[1]),
        _ => return Ok(None),
    };
    if n_cols != grid.len() {
        return Ok(None);
    }
    let surface_data: Vec<f64> = surf_view.iter().copied().collect();
    let surface_arr = Array2::from_shape_vec((n_rows, n_cols), surface_data)
        .map_err(|err| py_value_error(format!("failed to reshape surface: {err}")))?;
    let grid_arr = Array1::from_vec(grid);
    Ok(Some((
        grid_arr.into_pyarray(py).unbind(),
        surface_arr.into_pyarray(py).unbind(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ---- issue #964: S -> H conversion must not floor a valid survival prob ----

    #[test]
    fn cumulative_hazard_does_not_cap_small_survival() {
        // S = e^-100 is a valid, finite, representable survival probability for
        // constant hazard 1 at t = 100; the cumulative hazard must be 100, not
        // the clamped -ln(1e-12) = 27.63 the previous implementation returned.
        let s = (-100.0_f64).exp();
        let h = cumulative_hazard_from_survival_value(s).expect("valid survival prob");
        assert!(
            (h - 100.0).abs() < 1e-9,
            "expected H = 100 from S = e^-100, got {h}"
        );
        // Guard against regressing to the old clamp value.
        assert!(h > 50.0, "cumulative hazard was capped: {h}");
    }

    #[test]
    fn cumulative_hazard_maps_zero_survival_to_infinity() {
        let h = cumulative_hazard_from_survival_value(0.0).expect("S = 0 is valid");
        assert!(
            h.is_infinite() && h > 0.0,
            "S = 0 must map to +inf, got {h}"
        );
    }

    #[test]
    fn cumulative_hazard_maps_unit_survival_to_zero() {
        let h = cumulative_hazard_from_survival_value(1.0).expect("S = 1 is valid");
        assert_eq!(h, 0.0);
    }

    #[test]
    fn cumulative_hazard_passes_nan_through() {
        let h = cumulative_hazard_from_survival_value(f64::NAN).expect("NaN propagates");
        assert!(h.is_nan(), "NaN survival probability must propagate as NaN");
    }

    #[test]
    fn cumulative_hazard_rejects_out_of_range_survival() {
        assert!(cumulative_hazard_from_survival_value(1.5).is_err());
        assert!(cumulative_hazard_from_survival_value(-0.1).is_err());
        assert!(cumulative_hazard_from_survival_value(f64::INFINITY).is_err());
    }

    // ---- issue #965: parametric survival fallback is S(0) = 1 exactly ----

    #[test]
    fn exponential_survival_at_origin_is_one() {
        // exp(log_hazard) overflowing to +inf must still give S(0) = 1, not the
        // naive exp(-inf * 0) = exp(NaN) = NaN.
        let huge_hazard = f64::MAX;
        assert_eq!(exponential_survival_at(huge_hazard, 0.0), 1.0);
        // A finite hazard at t = 0 is likewise exactly 1.
        assert_eq!(exponential_survival_at(2.0, 0.0), 1.0);
    }

    #[test]
    fn exponential_survival_matches_closed_form_for_positive_time() {
        let s = exponential_survival_at(1.0, 100.0);
        assert!((s - (-100.0_f64).exp()).abs() < 1e-300);
    }

    // ---- issue #966 follow-up: pointwise hazard from the STORED knot lattice ----

    #[test]
    fn hazard_knots_uses_the_containing_knot_interval() {
        // Stored knots (t, H) = (0, 0), (1, 1), (2, 4): segment slopes 1 and 3.
        let grid = array![0.0, 1.0, 2.0];
        let surface = array![[0.0, 1.0, 4.0]];
        let queries = array![0.5, 1.5];
        let out = hazard_from_cumulative_knots_impl(grid.view(), surface.view(), queries.view())
            .expect("valid knots");
        assert!((out[[0, 0]] - 1.0).abs() < 1e-12, "got {}", out[[0, 0]]);
        assert!((out[[0, 1]] - 3.0).abs() < 1e-12, "got {}", out[[0, 1]]);
    }

    #[test]
    fn hazard_knots_is_independent_of_the_other_query_times() {
        // The core contract: adding an unrelated query must not change any
        // other query's hazard. The retired secant-between-queries differencing
        // returned 2 at t = 1.5 for [0.5, 1.5] and 3 once t = 1 was inserted.
        let grid = array![0.0, 1.0, 2.0];
        let surface = array![[0.0, 1.0, 4.0]];
        let sparse =
            hazard_from_cumulative_knots_impl(grid.view(), surface.view(), array![0.5, 1.5].view())
                .expect("valid knots");
        let dense = hazard_from_cumulative_knots_impl(
            grid.view(),
            surface.view(),
            array![0.5, 1.0, 1.5].view(),
        )
        .expect("valid knots");
        assert_eq!(sparse[[0, 1]].to_bits(), dense[[0, 2]].to_bits());
    }

    #[test]
    fn hazard_knots_accepts_any_query_order_and_repeats() {
        // Pointwise evaluation has no ordering precondition: unsorted and
        // repeated queries are answered identically per time.
        let grid = array![0.0, 1.0, 2.0];
        let surface = array![[0.0, 1.0, 4.0]];
        let out = hazard_from_cumulative_knots_impl(
            grid.view(),
            surface.view(),
            array![1.5, 0.5, 1.5].view(),
        )
        .expect("valid knots");
        assert!((out[[0, 0]] - 3.0).abs() < 1e-12);
        assert!((out[[0, 1]] - 1.0).abs() < 1e-12);
        assert!((out[[0, 2]] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn hazard_knots_is_zero_on_the_flat_extrapolation_regions() {
        // Below the grid H ≡ 0 and past the top knot H flat-clamps (#1595), so
        // the interpolant's slope — hence the hazard — is exactly 0 there,
        // including at t = +inf.
        let grid = array![1.0, 2.0];
        let surface = array![[1.0, 3.0]];
        let out = hazard_from_cumulative_knots_impl(
            grid.view(),
            surface.view(),
            array![0.5, 1.0, 5.0, f64::INFINITY].view(),
        )
        .expect("valid knots");
        assert_eq!(out[[0, 0]], 0.0);
        // t exactly at the FIRST knot sits at the edge of the flat left
        // region (left-continuous derivative): 0.
        assert_eq!(out[[0, 1]], 0.0);
        assert_eq!(out[[0, 2]], 0.0);
        assert_eq!(out[[0, 3]], 0.0);
    }

    #[test]
    fn hazard_knots_rejects_a_non_increasing_grid_and_decreasing_h() {
        let surface = array![[1.0, 3.0]];
        let err = hazard_from_cumulative_knots_impl(
            array![2.0, 1.0].view(),
            surface.view(),
            array![1.5].view(),
        )
        .expect_err("decreasing grid must error");
        assert!(
            err.contains("strictly increasing"),
            "unexpected error: {err}"
        );
        let err = hazard_from_cumulative_knots_impl(
            array![1.0, 2.0].view(),
            array![[3.0, 1.0]].view(),
            array![1.5].view(),
        )
        .expect_err("decreasing cumulative must error");
        assert!(err.contains("non-decreasing"), "unexpected error: {err}");
    }

    // ---- exponential fallback: exact H and F accessors (no S round trip) ----

    #[test]
    fn exponential_cumulative_hazard_is_exact_where_survival_rounds_away() {
        // hazard * t = 1e-20: S rounds to 1.0, so -ln S is 0; the direct H is
        // exact. hazard * t = 1000: S underflows to 0, so -ln S is +inf; the
        // direct H is 1000.
        assert_eq!(exponential_cumulative_hazard_at(1e-20, 1.0), 1e-20);
        assert_eq!(exponential_cumulative_hazard_at(1.0, 1000.0), 1000.0);
        assert_eq!(exponential_cumulative_hazard_at(1.0, 0.0), 0.0);
        assert_eq!(exponential_cumulative_hazard_at(1.0, -3.0), 0.0);
        assert_eq!(
            exponential_cumulative_hazard_at(1.0, f64::INFINITY),
            f64::INFINITY
        );
        assert_eq!(exponential_cumulative_hazard_at(0.0, f64::INFINITY), 0.0);
    }

    #[test]
    fn exponential_failure_preserves_tiny_probabilities() {
        // F(t) = -expm1(-h t): for h t = 1e-20 the failure probability is
        // ~1e-20, which 1 - exp(-h t) cancels to 0.
        let h = 1e-20_f64;
        let f = -(-exponential_cumulative_hazard_at(h, 1.0)).exp_m1();
        assert!(
            (f - 1e-20).abs() < 1e-32,
            "tiny failure probability lost: {f}"
        );
    }

    // ---- issue #965: the exponential parametric fallback is defined on all of
    //      R, with the genuine +inf asymptote distinguished from t <= 0. ----

    #[test]
    fn exponential_survival_negative_time_is_one() {
        // S(t) = P(T > t): everyone is still at risk before the origin.
        assert_eq!(exponential_survival_at(2.0, -5.0), 1.0);
        // Even an overflowing hazard must not produce exp(-inf * -5) = +inf.
        assert_eq!(exponential_survival_at(f64::MAX, -1.0), 1.0);
    }

    #[test]
    fn exponential_survival_plus_infinity_is_zero_for_positive_hazard() {
        assert_eq!(exponential_survival_at(2.0, f64::INFINITY), 0.0);
        // Degenerate zero hazard: nobody ever fails, so S stays 1 even at +inf.
        assert_eq!(exponential_survival_at(0.0, f64::INFINITY), 1.0);
    }
}

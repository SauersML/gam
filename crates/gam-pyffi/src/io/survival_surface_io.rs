//! Survival-surface interpolation and CSV/IO FFI helpers.
//!
//! This is a self-contained seam extracted from the pyffi monolith (issue
//! #780): a cohesive group of `#[pyfunction]`s that interpolate survival
//! surfaces along a time grid, reshape/validate survival parameter matrices,
//! convert between survival / cumulative-hazard / hazard representations, and
//! stream interpolated surfaces to CSV. They share the same nearest-endpoint
//! linear-interpolation kernel (`survival_csv_interpolate`) and asymptotic
//! extrapolation policy, and depend on nothing in the rest of the module
//! except the boundary error helper `py_value_error`.

use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufWriter, Write};

use ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use rayon::prelude::*;

use crate::py_value_error;

#[pyfunction]
pub(crate) fn interpolate_survival_surface<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray1<'py, f64>,
    surface: PyReadonlyArray2<'py, f64>,
    query: PyReadonlyArray1<'py, f64>,
    clip_lo: Option<f64>,
    clip_hi: Option<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let grid_view = grid.as_array();
    let surface_view = surface.as_array();
    let query_values: Vec<f64> = query.as_array().iter().copied().collect();
    let n_grid = grid_view.len();
    let (n_rows, n_cols) = surface_view.dim();
    if n_grid == 0 || n_cols != n_grid {
        return Err(py_value_error(
            "survival interpolation requires a non-empty grid".to_string(),
        ));
    }

    let mut order: Vec<(f64, usize)> = grid_view
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (value, index))
        .collect();
    order.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
    let sorted_grid: Vec<f64> = order.iter().map(|(value, _index)| *value).collect();
    let sorted_indices: Vec<usize> = order.iter().map(|(_value, index)| *index).collect();
    let n_query = query_values.len();
    let mut values = vec![0.0_f64; n_rows * n_query];

    values
        .par_chunks_mut(n_query.max(1))
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            if n_query == 0 {
                return;
            }
            let surface_row = surface_view.row(row_idx);
            for (query_idx, query_value) in query_values.iter().copied().enumerate() {
                let mut interpolated = if query_value.is_nan() {
                    f64::NAN
                } else if query_value <= sorted_grid[0] {
                    surface_row[sorted_indices[0]]
                } else if query_value >= sorted_grid[n_grid - 1] {
                    surface_row[sorted_indices[n_grid - 1]]
                } else {
                    let upper =
                        sorted_grid.partition_point(|grid_value| *grid_value <= query_value);
                    let lower = upper - 1;
                    let x0 = sorted_grid[lower];
                    let x1 = sorted_grid[upper];
                    let y0 = surface_row[sorted_indices[lower]];
                    let y1 = surface_row[sorted_indices[upper]];
                    if x1 == x0 {
                        y1
                    } else {
                        y0 + (query_value - x0) * (y1 - y0) / (x1 - x0)
                    }
                };
                if let Some(lo) = clip_lo {
                    if interpolated < lo {
                        interpolated = lo;
                    }
                }
                if let Some(hi) = clip_hi {
                    if interpolated > hi {
                        interpolated = hi;
                    }
                }
                out_row[query_idx] = interpolated;
            }
        });

    let out = Array2::from_shape_vec((n_rows, n_query), values).map_err(|err| {
        py_value_error(format!(
            "failed to assemble survival interpolation result: {err}"
        ))
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (grid, surface, query, clip_lo, clip_hi, left_value = None, right_value = None, inf_value = None))]
pub(crate) fn interpolate_rows<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray1<'py, f64>,
    surface: PyReadonlyArray2<'py, f64>,
    query: PyReadonlyArray1<'py, f64>,
    clip_lo: Option<f64>,
    clip_hi: Option<f64>,
    left_value: Option<f64>,
    right_value: Option<f64>,
    inf_value: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let grid_view = grid.as_array();
    let surface_view = surface.as_array();
    let query_values: Vec<f64> = query.as_array().iter().copied().collect();
    let n_grid = grid_view.len();
    let (n_rows, n_cols) = surface_view.dim();
    if n_grid == 0 || n_cols != n_grid {
        return Err(py_value_error(
            "survival interpolation requires a non-empty grid".to_string(),
        ));
    }

    let mut order: Vec<(f64, usize)> = grid_view
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (value, index))
        .collect();
    order.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
    let sorted_grid: Vec<f64> = order.iter().map(|(value, _index)| *value).collect();
    let sorted_indices: Vec<usize> = order.iter().map(|(_value, index)| *index).collect();
    let n_query = query_values.len();
    let mut values = vec![0.0_f64; n_rows * n_query];

    values
        .par_chunks_mut(n_query.max(1))
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            if n_query == 0 {
                return;
            }
            let surface_row = surface_view.row(row_idx);
            for (query_idx, query_value) in query_values.iter().copied().enumerate() {
                let mut interpolated = if query_value.is_nan() {
                    f64::NAN
                } else if query_value == f64::INFINITY {
                    // `t -> +inf` asymptote, distinct from the finite past-grid
                    // flat-clamp (#1595): survival surfaces continue to
                    // `S(+inf) = 0`, cumulative hazard to `H(+inf) = +inf`.
                    // Surfaces with no canonical asymptote (`hazard`,
                    // `survival_se`) pass `None` and fall back to the
                    // nearest-endpoint value (issue #965).
                    inf_value.unwrap_or(surface_row[sorted_indices[n_grid - 1]])
                } else if query_value <= sorted_grid[0] {
                    // Below-grid extrapolation: callers (e.g. survival surfaces)
                    // can supply an explicit asymptotic `left_value` so that
                    // extrapolating before the modeled support returns the
                    // theoretically correct boundary (S(t)=1 for t<=0). The
                    // exact-equality case still uses the grid value to remain
                    // continuous at the boundary; only strict left extrapolation
                    // honors the override.
                    if query_value < sorted_grid[0] {
                        left_value.unwrap_or(surface_row[sorted_indices[0]])
                    } else {
                        surface_row[sorted_indices[0]]
                    }
                } else if query_value >= sorted_grid[n_grid - 1] {
                    // Above-grid extrapolation: same idea, with `right_value`
                    // expressing the asymptote (S(inf)=0 for survival). Use
                    // strict inequality so the grid endpoint itself remains
                    // continuous.
                    if query_value > sorted_grid[n_grid - 1] {
                        right_value.unwrap_or(surface_row[sorted_indices[n_grid - 1]])
                    } else {
                        surface_row[sorted_indices[n_grid - 1]]
                    }
                } else {
                    let upper =
                        sorted_grid.partition_point(|grid_value| *grid_value <= query_value);
                    let lower = upper - 1;
                    let x0 = sorted_grid[lower];
                    let x1 = sorted_grid[upper];
                    let y0 = surface_row[sorted_indices[lower]];
                    let y1 = surface_row[sorted_indices[upper]];
                    if x1 == x0 {
                        y1
                    } else {
                        y0 + (query_value - x0) * (y1 - y0) / (x1 - x0)
                    }
                };
                if let Some(lo) = clip_lo {
                    if interpolated < lo {
                        interpolated = lo;
                    }
                }
                if let Some(hi) = clip_hi {
                    if interpolated > hi {
                        interpolated = hi;
                    }
                }
                out_row[query_idx] = interpolated;
            }
        });

    let out = Array2::from_shape_vec((n_rows, n_query), values)
        .map_err(|err| py_value_error(format!("failed to assemble interpolation result: {err}")))?;
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

fn survival_csv_interpolate(
    surface_values: &[f64],
    n_cols: usize,
    sorted_grid: &[f64],
    sorted_indices: &[usize],
    row_idx: usize,
    query_value: f64,
    left_value: Option<f64>,
    right_value: Option<f64>,
    inf_value: Option<f64>,
) -> f64 {
    if query_value.is_nan() {
        return f64::NAN;
    }
    let n_grid = sorted_grid.len();
    let row_offset = row_idx * n_cols;
    if query_value == f64::INFINITY {
        // `t -> +inf` asymptote, distinct from the finite past-grid flat-clamp
        // (#1595): `S(+inf) = 0`, `H(+inf) = +inf`; no-asymptote surfaces fall
        // back to the nearest endpoint (issue #965).
        return inf_value.unwrap_or(surface_values[row_offset + sorted_indices[n_grid - 1]]);
    }
    if query_value <= sorted_grid[0] {
        if query_value < sorted_grid[0] {
            left_value.unwrap_or(surface_values[row_offset + sorted_indices[0]])
        } else {
            surface_values[row_offset + sorted_indices[0]]
        }
    } else if query_value >= sorted_grid[n_grid - 1] {
        if query_value > sorted_grid[n_grid - 1] {
            right_value.unwrap_or(surface_values[row_offset + sorted_indices[n_grid - 1]])
        } else {
            surface_values[row_offset + sorted_indices[n_grid - 1]]
        }
    } else {
        let upper = sorted_grid.partition_point(|grid_value| *grid_value <= query_value);
        let lower = upper - 1;
        let x0 = sorted_grid[lower];
        let x1 = sorted_grid[upper];
        let y0 = surface_values[row_offset + sorted_indices[lower]];
        let y1 = surface_values[row_offset + sorted_indices[upper]];
        if x1 == x0 {
            y1
        } else {
            y0 + (query_value - x0) * (y1 - y0) / (x1 - x0)
        }
    }
}

fn clip_survival_surface_value(mut value: f64, clip_lo: Option<f64>, clip_hi: Option<f64>) -> f64 {
    if let Some(lo) = clip_lo {
        if value < lo {
            value = lo;
        }
    }
    if let Some(hi) = clip_hi {
        if value > hi {
            value = hi;
        }
    }
    value
}

#[pyfunction]
pub(crate) fn survival_chunk_iter_collect<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray1<'py, f64>,
    surface: PyReadonlyArray2<'py, f64>,
    times: PyReadonlyArray1<'py, f64>,
    kind: &str,
    clip_lo: Option<f64>,
    clip_hi: Option<f64>,
    people_chunk: usize,
    time_grid_chunk: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    // Mirror the asymptotic-extrapolation policy in
    // `gamfit._survival._SURVIVAL_EXTRAPOLATION` EXACTLY so the dense
    // auto-chunked path and the small non-chunked path agree past the grid
    // (issue #965 / completing #1595):
    //   * left edge (`t <= t_min`): `S = 1`, `H = 0` -- unambiguous boundary.
    //   * right edge (finite `t > t_max`): flat-clamp to the last grid value
    //     (`right_value = None`). Forcing `S -> 0` here (the old hardcoded
    //     `Some(0.0)`) contradicted the non-chunked path and broke the
    //     `S(t) = exp(-H(t))` identity past the grid (#1595): it made `S` jump
    //     to 0 while `H` stayed finite.
    //   * `t == +inf`: the genuine asymptote `S = 0`, `H = +inf` (carried by
    //     `kind_inf_value`, separate from the finite flat-clamp).
    // Hazards and standard errors have no canonical asymptote, so they keep
    // nearest-endpoint behavior on every edge (`None`).
    let (kind_left_value, kind_right_value, kind_inf_value): (
        Option<f64>,
        Option<f64>,
        Option<f64>,
    ) = match kind {
        "survival" => (Some(1.0), None, Some(0.0)),
        "cumulative_hazard" => (Some(0.0), None, Some(f64::INFINITY)),
        "hazard" | "survival_se" => (None, None, None),
        other => {
            return Err(py_value_error(format!(
                "unknown survival surface kind '{other}'"
            )));
        }
    };
    if people_chunk == 0 {
        return Err(py_value_error("people_chunk must be positive".to_string()));
    }
    if time_grid_chunk == 0 {
        return Err(py_value_error(
            "time_grid_chunk must be positive".to_string(),
        ));
    }
    let grid_view = grid.as_array();
    let surface_view = surface.as_array();
    let n_grid = grid_view.len();
    let (n_rows, n_cols) = surface_view.dim();
    if n_grid == 0 || n_cols != n_grid {
        return Err(py_value_error(
            "survival interpolation requires a non-empty grid".to_string(),
        ));
    }

    let mut order: Vec<(f64, usize)> = grid_view
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (value, index))
        .collect();
    order.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
    let sorted_grid: Vec<f64> = order.iter().map(|(value, _index)| *value).collect();
    let sorted_indices: Vec<usize> = order.iter().map(|(_value, index)| *index).collect();
    let mut surface_values = Vec::with_capacity(n_rows * n_cols);
    for row_idx in 0..n_rows {
        for col_idx in 0..n_cols {
            surface_values.push(surface_view[[row_idx, col_idx]]);
        }
    }
    let times_values: Vec<f64> = times.as_array().iter().copied().collect();
    let n_times = times_values.len();

    let values = py.detach(move || {
        let mut values = vec![0.0_f64; n_rows * n_times];
        for row_start in (0..n_rows).step_by(people_chunk) {
            let row_stop = (row_start + people_chunk).min(n_rows);
            for time_start in (0..n_times).step_by(time_grid_chunk) {
                let time_stop = (time_start + time_grid_chunk).min(n_times);
                for row_idx in row_start..row_stop {
                    let out_row_start = row_idx * n_times;
                    for time_idx in time_start..time_stop {
                        let interpolated = survival_csv_interpolate(
                            &surface_values,
                            n_cols,
                            &sorted_grid,
                            &sorted_indices,
                            row_idx,
                            times_values[time_idx],
                            kind_left_value,
                            kind_right_value,
                            kind_inf_value,
                        );
                        values[out_row_start + time_idx] =
                            clip_survival_surface_value(interpolated, clip_lo, clip_hi);
                    }
                }
            }
        }
        values
    });
    let out = Array2::from_shape_vec((n_rows, n_times), values).map_err(|err| {
        py_value_error(format!("failed to assemble survival chunk result: {err}"))
    })?;
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
    if people_chunk == 0 {
        return Err(py_value_error("people_chunk must be positive".to_string()));
    }
    if time_grid_chunk == 0 {
        return Err(py_value_error(
            "time_grid_chunk must be positive".to_string(),
        ));
    }
    let grid_view = grid.as_array();
    let surface_view = surface.as_array();
    let n_grid = grid_view.len();
    let (n_rows, n_cols) = surface_view.dim();
    if n_grid == 0 || n_cols != n_grid {
        return Err(py_value_error(
            "survival interpolation requires a non-empty grid".to_string(),
        ));
    }
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

    let mut order: Vec<(f64, usize)> = grid_view
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (value, index))
        .collect();
    order.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
    let sorted_grid: Vec<f64> = order.iter().map(|(value, _index)| *value).collect();
    let sorted_indices: Vec<usize> = order.iter().map(|(_value, index)| *index).collect();
    let mut surface_values = Vec::with_capacity(n_rows * n_cols);
    for row_idx in 0..n_rows {
        for col_idx in 0..n_cols {
            surface_values.push(surface_view[[row_idx, col_idx]]);
        }
    }
    let times_values: Vec<f64> = times.as_array().iter().copied().collect();
    let path_owned = path.to_string();

    Ok(py.detach(move || -> std::io::Result<String> {
        let file = File::create(&path_owned)?;
        let mut writer = BufWriter::new(file);
        match id_column.as_ref() {
            Some(column) => {
                writer.write_all(b"row,")?;
                write_csv_field(&mut writer, column)?;
                writer.write_all(b",time,survival\n")?;
            }
            None => writer.write_all(b"row,time,survival\n")?,
        }

        for row_start in (0..n_rows).step_by(people_chunk) {
            let row_stop = (row_start + people_chunk).min(n_rows);
            for time_start in (0..times_values.len()).step_by(time_grid_chunk) {
                let time_stop = (time_start + time_grid_chunk).min(times_values.len());
                for row_idx in row_start..row_stop {
                    for query_value in &times_values[time_start..time_stop] {
                        let survival = survival_csv_interpolate(
                            &surface_values,
                            n_cols,
                            &sorted_grid,
                            &sorted_indices,
                            row_idx,
                            *query_value,
                            // Same unified policy as the query path (#965):
                            // S(t<=0)=1, finite past-grid flat-clamp (None),
                            // S(+inf)=0.
                            Some(1.0),
                            None,
                            Some(0.0),
                        )
                        .clamp(0.0, 1.0);
                        match (id_column.as_ref(), row_ids.as_ref()) {
                            (Some(_column), Some(ids)) => {
                                write!(writer, "{row_idx},")?;
                                write_csv_field(&mut writer, &ids[row_idx])?;
                                writeln!(writer, ",{query_value},{survival}")?;
                            }
                            _ => writeln!(writer, "{row_idx},{query_value},{survival}")?,
                        }
                    }
                }
            }
        }
        writer.flush()?;
        Ok(path_owned)
    })?)
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
    if let Some((idx, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| value.is_nan())
    {
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
pub(crate) fn survival_collect_chunks<'py>(
    py: Python<'py>,
    n_rows: usize,
    n_times: usize,
    blocks: Vec<(usize, usize, usize, usize, PyReadonlyArray2<'py, f64>)>,
) -> PyResult<Py<PyArray2<f64>>> {
    let mut dense = Array2::<f64>::zeros((n_rows, n_times));
    for (row_start, row_stop, time_start, time_stop, block) in blocks {
        if row_start > row_stop
            || row_stop > n_rows
            || time_start > time_stop
            || time_stop > n_times
        {
            return Err(py_value_error(
                "survival chunk block exceeds dense matrix bounds".to_string(),
            ));
        }
        let block_view = block.as_array();
        let (br, bc) = (block_view.shape()[0], block_view.shape()[1]);
        if br != row_stop - row_start || bc != time_stop - time_start {
            return Err(py_value_error(
                "survival chunk block shape mismatch".to_string(),
            ));
        }
        for i in 0..br {
            for j in 0..bc {
                dense[[row_start + i, time_start + j]] = block_view[[i, j]];
            }
        }
    }
    Ok(dense.into_pyarray(py).unbind())
}

#[pyfunction]
pub(crate) fn hazard_from_cumulative<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'py, f64>,
    cumulative: PyReadonlyArray2<'py, f64>,
    previous_cumulative: Option<PyReadonlyArray2<'py, f64>>,
    previous_time: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cum_view = cumulative.as_array();
    let times_view = times.as_array();
    let (n_rows, n_times) = cum_view.dim();
    if times_view.len() != n_times {
        return Err(py_value_error(
            "hazard_from_cumulative requires matching time count".to_string(),
        ));
    }
    let previous = match previous_cumulative {
        Some(arr) => {
            let view = arr.as_array();
            if view.dim() != (n_rows, 1) {
                return Err(py_value_error(
                    "previous_cumulative must have one column per row".to_string(),
                ));
            }
            view.to_owned()
        }
        None => Array2::<f64>::zeros((n_rows, 1)),
    };
    hazard_from_cumulative_impl(times_view, cum_view, previous.view(), previous_time)
        .map(|out| out.into_pyarray(py))
        .map_err(py_value_error)
}

/// Forward-difference the sampled cumulative hazard into an instantaneous
/// hazard, with the discretization preconditions a finite difference requires
/// (issue #966).
///
/// For each row the hazard at query `j` is the slope of the cumulative-hazard
/// interpolant over the interval ending at `times[j]`:
/// `(H(times[j]) - H(prev_t)) / (times[j] - prev_t)`, where the predecessor of
/// the first query is the `(previous_time, previous_cumulative)` carry from the
/// prior chunk. This is only meaningful when the evaluation times are strictly
/// increasing (a forward difference needs a positive, well-defined width) and
/// the cumulative hazard is non-decreasing (it is monotone by definition, so a
/// decrease signals a corrupt input rather than a hazard the slope should
/// "invent"). The previous code instead forced `width = 1.0` whenever the gap
/// was non-positive, which silently changed the time unit, manufactured a zero
/// hazard at repeated times, and produced a negative hazard from unsorted
/// query times. We reject those inputs rather than fabricate a slope; callers
/// that need hazards from an arbitrary order must sort their query times first.
fn hazard_from_cumulative_impl(
    times: ndarray::ArrayView1<'_, f64>,
    cumulative: ndarray::ArrayView2<'_, f64>,
    previous_cumulative: ndarray::ArrayView2<'_, f64>,
    previous_time: f64,
) -> Result<Array2<f64>, String> {
    let (n_rows, n_times) = cumulative.dim();
    let mut out = Array2::<f64>::zeros((n_rows, n_times));
    for j in 0..n_times {
        let prev_t = if j == 0 { previous_time } else { times[j - 1] };
        let width = times[j] - prev_t;
        if !(width > 0.0) {
            return Err(format!(
                "hazard_from_cumulative requires strictly increasing query times; \
                 width {width} at index {j} (time {} after {prev_t}) is not positive. \
                 Sort the query times (and de-duplicate) before differencing.",
                times[j]
            ));
        }
        for i in 0..n_rows {
            let prev_h = if j == 0 {
                previous_cumulative[[i, 0]]
            } else {
                cumulative[[i, j - 1]]
            };
            let delta = cumulative[[i, j]] - prev_h;
            if delta < 0.0 {
                return Err(format!(
                    "cumulative hazard must be non-decreasing in time; row {i} drops by \
                     {} between index {} and {j}",
                    -delta,
                    j.wrapping_sub(1)
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

    // ---- issue #966: hazard differencing requires strictly increasing times ----

    #[test]
    fn hazard_from_cumulative_matches_constant_slope() {
        // H(t) = 2 t over strictly increasing times sampled from the origin
        // carry (previous_time = 0, previous_cumulative = 0) yields hazard 2.
        let times = array![1.0, 2.0, 3.0];
        let cumulative = array![[2.0, 4.0, 6.0]];
        let previous = array![[0.0]];
        let out =
            hazard_from_cumulative_impl(times.view(), cumulative.view(), previous.view(), 0.0)
                .expect("strictly increasing times");
        for &h in out.iter() {
            assert!(
                (h - 2.0).abs() < 1e-12,
                "expected constant hazard 2, got {h}"
            );
        }
    }

    #[test]
    fn hazard_from_cumulative_rejects_repeated_times() {
        // Repeated times (width 0) previously invented a zero hazard via the
        // forced width = 1; now they are rejected.
        let times = array![1.0, 1.0];
        let cumulative = array![[2.0, 2.0]];
        let previous = array![[0.0]];
        let err =
            hazard_from_cumulative_impl(times.view(), cumulative.view(), previous.view(), 0.0)
                .expect_err("repeated times must error");
        assert!(
            err.contains("strictly increasing"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn hazard_from_cumulative_rejects_unsorted_times() {
        // Unsorted times previously produced a negative hazard from a negative
        // width forced to 1; now they are rejected rather than fabricated.
        let times = array![2.0, 1.0];
        let cumulative = array![[4.0, 2.0]];
        let previous = array![[0.0]];
        let err =
            hazard_from_cumulative_impl(times.view(), cumulative.view(), previous.view(), 0.0)
                .expect_err("unsorted times must error");
        assert!(
            err.contains("strictly increasing"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn hazard_from_cumulative_rejects_origin_collision_with_carry() {
        // First query equal to the previous-time carry has zero width.
        let times = array![0.0, 1.0];
        let cumulative = array![[0.0, 1.0]];
        let previous = array![[0.0]];
        assert!(
            hazard_from_cumulative_impl(times.view(), cumulative.view(), previous.view(), 0.0)
                .is_err()
        );
    }

    #[test]
    fn hazard_from_cumulative_rejects_decreasing_cumulative() {
        // Cumulative hazard is monotone non-decreasing by definition; a drop
        // signals corrupt input rather than a (nonsensical) negative hazard.
        let times = array![1.0, 2.0];
        let cumulative = array![[4.0, 2.0]];
        let previous = array![[0.0]];
        let err =
            hazard_from_cumulative_impl(times.view(), cumulative.view(), previous.view(), 0.0)
                .expect_err("decreasing cumulative must error");
        assert!(err.contains("non-decreasing"), "unexpected error: {err}");
    }

    // ---- issue #965 / completing #1595: unified extrapolation policy on the
    //      pure CSV/chunk interpolation kernel. A single monotone-increasing
    //      grid 0..2 with the surface row equal to the grid values so the
    //      interior interpolant is the identity (easy to reason about). ----

    fn unit_grid() -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        // grid = [0, 1, 2], surface row = [0, 1, 2] (already sorted).
        let surface_values = vec![0.0, 1.0, 2.0];
        let sorted_grid = vec![0.0, 1.0, 2.0];
        let sorted_indices = vec![0usize, 1, 2];
        (surface_values, sorted_grid, sorted_indices)
    }

    fn interp(query: f64, left: Option<f64>, right: Option<f64>, inf: Option<f64>) -> f64 {
        let (surface_values, sorted_grid, sorted_indices) = unit_grid();
        survival_csv_interpolate(
            &surface_values,
            sorted_grid.len(),
            &sorted_grid,
            &sorted_indices,
            0,
            query,
            left,
            right,
            inf,
        )
    }

    #[test]
    fn csv_interpolate_below_grid_honors_left_asymptote() {
        // Survival surface: S(t) = 1 strictly before the modeled support, even
        // when the grid endpoint value is 0 (issue #965: negative predict-query
        // times must extrapolate to the boundary, not be rejected).
        assert_eq!(interp(-2.0, Some(1.0), None, Some(0.0)), 1.0);
        // The grid boundary itself stays continuous (uses the grid value, not
        // the override).
        assert_eq!(interp(0.0, Some(1.0), None, Some(0.0)), 0.0);
    }

    #[test]
    fn csv_interpolate_finite_past_grid_flat_clamps() {
        // #1595: a FINITE time past the last grid point flat-clamps to the last
        // grid value (right_value = None), it does NOT jump to the +inf
        // asymptote. Here the last grid value is 2.0.
        assert_eq!(interp(100.0, Some(1.0), None, Some(0.0)), 2.0);
        // Grid endpoint itself is continuous.
        assert_eq!(interp(2.0, Some(1.0), None, Some(0.0)), 2.0);
    }

    #[test]
    fn csv_interpolate_plus_infinity_uses_genuine_asymptote() {
        // #965: t == +inf is the genuine asymptote, distinct from the finite
        // flat-clamp. Survival -> 0, cumulative hazard -> +inf.
        assert_eq!(interp(f64::INFINITY, Some(1.0), None, Some(0.0)), 0.0);
        let h = interp(f64::INFINITY, Some(0.0), None, Some(f64::INFINITY));
        assert!(h.is_infinite() && h > 0.0, "H(+inf) must be +inf, got {h}");
    }

    #[test]
    fn csv_interpolate_no_asymptote_kinds_fall_back_to_endpoint() {
        // hazard / survival_se carry None on every edge: both the finite
        // past-grid extrapolation and +inf fall back to the nearest endpoint.
        assert_eq!(interp(-2.0, None, None, None), 0.0); // first grid value
        assert_eq!(interp(100.0, None, None, None), 2.0); // last grid value
        assert_eq!(interp(f64::INFINITY, None, None, None), 2.0); // last grid value
    }

    #[test]
    fn csv_interpolate_interior_is_linear() {
        // Sanity: interior query interpolates linearly (identity surface).
        assert!((interp(0.5, Some(1.0), None, Some(0.0)) - 0.5).abs() < 1e-12);
        assert!((interp(1.5, Some(1.0), None, Some(0.0)) - 1.5).abs() < 1e-12);
    }

    #[test]
    fn csv_interpolate_nan_query_propagates() {
        assert!(interp(f64::NAN, Some(1.0), None, Some(0.0)).is_nan());
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

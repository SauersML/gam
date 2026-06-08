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
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
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
#[pyo3(signature = (grid, surface, query, clip_lo, clip_hi, left_value = None, right_value = None))]
pub(crate) fn interpolate_rows<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray1<'py, f64>,
    surface: PyReadonlyArray2<'py, f64>,
    query: PyReadonlyArray1<'py, f64>,
    clip_lo: Option<f64>,
    clip_hi: Option<f64>,
    left_value: Option<f64>,
    right_value: Option<f64>,
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
) -> f64 {
    if query_value.is_nan() {
        return f64::NAN;
    }
    let n_grid = sorted_grid.len();
    let row_offset = row_idx * n_cols;
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
    // `gamfit._survival._SURVIVAL_EXTRAPOLATION`: survival surfaces are
    // continued to S(t<=0)=1 and S(t->inf)=0 outside the modeled grid, and
    // cumulative hazard mirrors that via H(t<=0)=0. Hazards and standard
    // errors have no canonical asymptote, so they keep nearest-endpoint
    // behavior (signaled by `None`/`None`).
    let (kind_left_value, kind_right_value): (Option<f64>, Option<f64>) = match kind {
        "survival" => (Some(1.0), Some(0.0)),
        "cumulative_hazard" => (Some(0.0), None),
        "hazard" | "survival_se" => (None, None),
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
                            Some(1.0),
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
    let typed = flat.cast_into::<PyArray1<f64>>().map_err(PyErr::from)?;
    let values: Vec<f64> = typed.as_array().iter().copied().collect();
    if values.is_empty() {
        return Err(py_value_error(
            "survival prediction requires at least one time".to_string(),
        ));
    }
    if !values.iter().all(|value| value.is_finite()) {
        return Err(py_value_error(
            "survival prediction times must be finite".to_string(),
        ));
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
    let mut out = Array2::<f64>::zeros((n_rows, n_times));
    for j in 0..n_times {
        let prev_t = if j == 0 {
            previous_time
        } else {
            times_view[j - 1]
        };
        let mut width = times_view[j] - prev_t;
        if width <= 0.0 {
            width = 1.0;
        }
        for i in 0..n_rows {
            let prev_h = if j == 0 {
                previous[[i, 0]]
            } else {
                cum_view[[i, j - 1]]
            };
            out[[i, j]] = (cum_view[[i, j]] - prev_h) / width;
        }
    }
    Ok(out.into_pyarray(py))
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
            let s = view[[i, j]].clamp(1e-12, 1.0);
            out[[i, j]] = -s.ln();
        }
    }
    Ok(out.into_pyarray(py).unbind())
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
            out[[i, j]] = (-hazard * times_view[j]).exp();
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

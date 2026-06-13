//! Competing-risks prediction JSON decoding helpers.
//!
//! Self-contained seam extracted from the pyffi monolith (issue #780): the pure
//! free functions that decode a competing-risks prediction payload from its JSON
//! representation into ndarray / `BTreeMap` form and stage the optional
//! `survival` / `cif` / hazard vectors and matrices onto a result `PyDict`. They
//! operate only on `serde_json::Value` inputs (plus the two `set_optional_*`
//! stagers that touch a `PyDict`), share the nested-array depth/shape validators
//! (`competing_risks_array_depth`, `competing_risks_matrix2`,
//! `competing_risks_stacked_matrix3`), and depend on nothing in the rest of the
//! module except the boundary error helper `py_value_error`. The
//! `#[pyfunction]` that consumes them (`competing_risks_prediction_payload_from_json`)
//! stays in the parent module via a focused re-import.

use std::collections::BTreeMap;

use ndarray::{Array1, Array2};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::py_value_error;

pub(crate) fn set_optional_competing_risks_vector<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyDict>,
    key: &str,
    raw: Option<&serde_json::Value>,
) -> PyResult<()> {
    match raw {
        Some(serde_json::Value::Null) | None => out.set_item(key, py.None()),
        Some(value) => out.set_item(
            key,
            Array1::from_vec(competing_risks_flattened_numbers(value, key)?).into_pyarray(py),
        ),
    }
}

pub(crate) fn set_optional_competing_risks_matrix<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyDict>,
    key: &str,
    raw: Option<&serde_json::Value>,
) -> PyResult<()> {
    match raw {
        Some(serde_json::Value::Null) | None => out.set_item(key, py.None()),
        Some(value) => out.set_item(
            key,
            competing_risks_numeric_matrix(value, key)?.into_pyarray(py),
        ),
    }
}

pub(crate) fn competing_risks_columns(
    raw: Option<&serde_json::Value>,
) -> PyResult<BTreeMap<String, Vec<f64>>> {
    let Some(value) = raw else {
        return Ok(BTreeMap::new());
    };
    if value.is_null() {
        return Ok(BTreeMap::new());
    }
    let object = value
        .as_object()
        .ok_or_else(|| py_value_error("columns must be a JSON object".to_string()))?;
    let mut columns = BTreeMap::new();
    for (name, values) in object {
        columns.insert(
            name.clone(),
            competing_risks_numeric_list(Some(values), &format!("columns.{name}"))?,
        );
    }
    Ok(columns)
}

pub(crate) fn competing_risks_string_list(
    raw: Option<&serde_json::Value>,
    key: &str,
) -> PyResult<Vec<String>> {
    let Some(value) = raw else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let items = value
        .as_array()
        .ok_or_else(|| py_value_error(format!("{key} must be a JSON array")))?;
    let mut out = Vec::with_capacity(items.len());
    for (idx, item) in items.iter().enumerate() {
        let text = item
            .as_str()
            .ok_or_else(|| py_value_error(format!("{key}[{idx}] must be a string")))?;
        out.push(text.to_string());
    }
    Ok(out)
}

pub(crate) fn competing_risks_numeric_list(
    raw: Option<&serde_json::Value>,
    key: &str,
) -> PyResult<Vec<f64>> {
    let Some(value) = raw else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let items = value
        .as_array()
        .ok_or_else(|| py_value_error(format!("{key} must be a JSON array")))?;
    let mut out = Vec::with_capacity(items.len());
    for (idx, item) in items.iter().enumerate() {
        out.push(competing_risks_number(item, &format!("{key}[{idx}]"))?);
    }
    Ok(out)
}

pub(crate) fn competing_risks_flattened_numbers(
    value: &serde_json::Value,
    key: &str,
) -> PyResult<Vec<f64>> {
    let mut out = Vec::new();
    competing_risks_flatten_numbers_into(value, key, &mut out)?;
    Ok(out)
}

pub(crate) fn competing_risks_flatten_numbers_into(
    value: &serde_json::Value,
    key: &str,
    out: &mut Vec<f64>,
) -> PyResult<()> {
    match value {
        serde_json::Value::Number(_) => {
            out.push(competing_risks_number(value, key)?);
            Ok(())
        }
        serde_json::Value::Array(items) => {
            for (idx, item) in items.iter().enumerate() {
                competing_risks_flatten_numbers_into(item, &format!("{key}[{idx}]"), out)?;
            }
            Ok(())
        }
        _ => Err(py_value_error(format!("{key} must contain only numbers"))),
    }
}

pub(crate) fn competing_risks_numeric_matrix(
    value: &serde_json::Value,
    key: &str,
) -> PyResult<Array2<f64>> {
    match competing_risks_array_depth(value, key)? {
        1 => competing_risks_vector_as_matrix(value, key),
        2 => competing_risks_matrix2(value, key),
        3 => competing_risks_stacked_matrix3(value, key),
        depth => Err(py_value_error(format!(
            "{key} must be a vector, matrix, or list of matrices; got array depth {depth}"
        ))),
    }
}

pub(crate) fn competing_risks_vector_as_matrix(
    value: &serde_json::Value,
    key: &str,
) -> PyResult<Array2<f64>> {
    let values = competing_risks_numeric_list(Some(value), key)?;
    let n_rows = values.len();
    Array2::from_shape_vec((n_rows, 1), values)
        .map_err(|err| py_value_error(format!("failed to reshape {key}: {err}")))
}

pub(crate) fn competing_risks_matrix2(
    value: &serde_json::Value,
    key: &str,
) -> PyResult<Array2<f64>> {
    let rows = value
        .as_array()
        .ok_or_else(|| py_value_error(format!("{key} must be a JSON array")))?;
    if rows.is_empty() {
        return Ok(Array2::<f64>::zeros((0, 1)));
    }
    let first_row = rows[0]
        .as_array()
        .ok_or_else(|| py_value_error(format!("{key}[0] must be a JSON array")))?;
    let n_cols = first_row.len();
    let mut flat = Vec::with_capacity(rows.len() * n_cols);
    for (row_idx, row) in rows.iter().enumerate() {
        let cells = row
            .as_array()
            .ok_or_else(|| py_value_error(format!("{key}[{row_idx}] must be a JSON array")))?;
        if cells.len() != n_cols {
            return Err(py_value_error(format!(
                "{key}[{row_idx}] has length {}, expected {n_cols}",
                cells.len()
            )));
        }
        for (col_idx, cell) in cells.iter().enumerate() {
            flat.push(competing_risks_number(
                cell,
                &format!("{key}[{row_idx}][{col_idx}]"),
            )?);
        }
    }
    Array2::from_shape_vec((rows.len(), n_cols), flat)
        .map_err(|err| py_value_error(format!("failed to reshape {key}: {err}")))
}

pub(crate) fn competing_risks_stacked_matrix3(
    value: &serde_json::Value,
    key: &str,
) -> PyResult<Array2<f64>> {
    let matrices = value
        .as_array()
        .ok_or_else(|| py_value_error(format!("{key} must be a JSON array")))?;
    if matrices.is_empty() {
        return Ok(Array2::<f64>::zeros((0, 1)));
    }
    let mut n_cols = None;
    let mut n_rows = 0usize;
    let mut flat = Vec::new();
    for (matrix_idx, matrix_value) in matrices.iter().enumerate() {
        let matrix_key = format!("{key}[{matrix_idx}]");
        let matrix = competing_risks_matrix2(matrix_value, &matrix_key)?;
        match n_cols {
            Some(expected) if matrix.ncols() != expected => {
                return Err(py_value_error(format!(
                    "{matrix_key} has {} columns, expected {expected}",
                    matrix.ncols()
                )));
            }
            Some(_) => {}
            None => {
                n_cols = Some(matrix.ncols());
            }
        }
        n_rows += matrix.nrows();
        flat.extend(matrix.iter().copied());
    }
    let cols = n_cols.unwrap_or(1);
    Array2::from_shape_vec((n_rows, cols), flat)
        .map_err(|err| py_value_error(format!("failed to reshape {key}: {err}")))
}

pub(crate) fn competing_risks_array_depth(value: &serde_json::Value, key: &str) -> PyResult<usize> {
    match value {
        serde_json::Value::Array(items) => {
            if items.is_empty() {
                return Ok(1);
            }
            let first_depth = competing_risks_array_depth(&items[0], &format!("{key}[0]"))?;
            for (idx, item) in items.iter().enumerate().skip(1) {
                let depth = competing_risks_array_depth(item, &format!("{key}[{idx}]"))?;
                if depth != first_depth {
                    return Err(py_value_error(format!(
                        "{key} has inconsistent array depth at index {idx}: got {depth}, expected {first_depth}"
                    )));
                }
            }
            Ok(first_depth + 1)
        }
        serde_json::Value::Number(_) => Ok(0),
        _ => Err(py_value_error(format!(
            "{key} must contain only numeric arrays"
        ))),
    }
}

pub(crate) fn competing_risks_number(value: &serde_json::Value, key: &str) -> PyResult<f64> {
    let number = value
        .as_f64()
        .ok_or_else(|| py_value_error(format!("{key} must be a finite number")))?;
    if !number.is_finite() {
        return Err(py_value_error(format!("{key} must be finite")));
    }
    Ok(number)
}

//! scikit-learn estimator metadata helpers.
//!
//! Self-contained seam extracted from the pyffi monolith (issue #780): the
//! pure formula-introspection helpers (`sklearn_response_column_name`,
//! `sklearn_resolved_formula`) and the `#[pyfunction]` that consumes them
//! (`sklearn_fit_metadata`) which resolves the fit formula, response/target
//! name, and feature-name list for the sklearn-compatible estimator wrapper.
//! They depend on nothing in the rest of the module except the boundary error
//! helper `py_value_error`. The parent re-imports all three so the
//! `#[pymodule]` registration and call sites are unchanged.

use pyo3::prelude::*;

use crate::py_value_error;

pub(crate) fn sklearn_response_column_name(formula: &str) -> Option<String> {
    if !formula.contains('~') {
        return None;
    }
    let candidate = formula.split('~').next()?.trim();
    if candidate.is_empty() || candidate.starts_with("Surv(") {
        return None;
    }
    let without_underscores = candidate.replace('_', "");
    if without_underscores.is_empty()
        || !without_underscores
            .chars()
            .all(|character| character.is_alphanumeric())
    {
        return None;
    }
    Some(candidate.to_string())
}

pub(crate) fn sklearn_resolved_formula(formula: &str, target_name: &str) -> String {
    match formula.split_once('~') {
        Some((_lhs, rhs)) => format!("{target_name} ~ {}", rhs.trim()),
        None => format!("{target_name} ~ {}", formula.trim()),
    }
}

#[pyfunction(signature = (columns, formula, target_column = None, has_external_target = false))]
pub(crate) fn sklearn_fit_metadata(
    columns: Vec<String>,
    formula: &str,
    target_column: Option<String>,
    has_external_target: bool,
) -> PyResult<(String, Vec<String>, String)> {
    let has_target_column = target_column.is_some();
    if has_target_column && has_external_target {
        return Err(py_value_error(
            "target_column and has_external_target are mutually exclusive".to_string(),
        ));
    }

    let target_name = if let Some(target_column) = target_column {
        if !columns.iter().any(|column| column == &target_column) {
            return Err(py_value_error(format!(
                "target column '{target_column}' is missing from the training table"
            )));
        }
        target_column
    } else if has_external_target {
        sklearn_response_column_name(formula).unwrap_or_else(|| "y".to_string())
    } else {
        let target_name = sklearn_response_column_name(formula).ok_or_else(|| {
            py_value_error("formula must include a response when y is not provided".to_string())
        })?;
        if !columns.iter().any(|column| column == &target_name) {
            return Err(py_value_error(format!(
                "response column '{target_name}' is missing from the training table"
            )));
        }
        target_name
    };

    if has_external_target && columns.iter().any(|column| column == &target_name) {
        return Err(py_value_error(format!(
            "target column '{target_name}' already exists in the feature table"
        )));
    }

    let fit_formula = if has_external_target || has_target_column {
        sklearn_resolved_formula(formula, &target_name)
    } else {
        formula.to_string()
    };
    let feature_names = columns
        .into_iter()
        .filter(|column| column != &target_name)
        .collect();

    Ok((fit_formula, feature_names, target_name))
}

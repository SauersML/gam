//! scikit-learn estimator metadata helpers.
//!
//! Self-contained seam extracted from the pyffi monolith (issue #780): the
//! formula rewrite `sklearn_resolved_formula` and the `#[pyfunction]` that
//! consumes it (`sklearn_fit_metadata`), which resolves the fit formula,
//! response/target name, and feature-name list for the sklearn-compatible
//! estimator wrapper. The response column comes from the formula DSL's own
//! `formula_response_column`, the same authority the CLI fit uses.

use gam::terms::inference::formula_dsl::formula_response_column;
use pyo3::prelude::*;

use crate::py_value_error;

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
        formula_response_column(formula).unwrap_or_else(|| "y".to_string())
    } else {
        let target_name = formula_response_column(formula).ok_or_else(|| {
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

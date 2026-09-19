//! scikit-learn estimator metadata helpers.
//!
//! Self-contained seam extracted from the pyffi monolith (issue #780): the
//! formula rewrite `sklearn_resolved_formula` and the `#[pyfunction]` that
//! consumes it (`sklearn_fit_metadata`), which resolves the fit formula,
//! response/target name, and feature-name list for the sklearn-compatible
//! estimator wrapper. The response column comes from the formula DSL's own
//! `formula_response_column`, the same authority the CLI fit uses.

use gam::solver::fit_orchestration::formula_columns;
use gam::terms::inference::formula_dsl::{
    AUTOMATIC_REST_TERM, formula_response_column, parse_formula,
};
use pyo3::prelude::*;

use crate::py_value_error;

pub(crate) fn sklearn_resolved_formula(formula: &str, target_name: &str) -> String {
    match formula.split_once('~') {
        Some((_lhs, rhs)) => format!("{target_name} ~ {}", rhs.trim()),
        None => format!("{target_name} ~ {}", formula.trim()),
    }
}

/// First name of the form `sample_weight`, `sample_weight_1`, ... that is
/// neither a table column nor the response, so sklearn `sample_weight`
/// can be attached to the training table as the prior-weight column.
fn sklearn_weight_column(columns: &[String], target_name: &str) -> String {
    let taken = |name: &str| name == target_name || columns.iter().any(|column| column == name);
    let base = "sample_weight";
    if !taken(base) {
        return base.to_string();
    }
    (1usize..)
        .map(|suffix| format!("{base}_{suffix}"))
        .find(|name| !taken(name))
        .expect("an unbounded suffix sequence always yields a free name")
}

#[pyfunction(signature = (
    columns,
    formula,
    target_column = None,
    has_external_target = false,
    has_sample_weight = false,
))]
pub(crate) fn sklearn_fit_metadata(
    columns: Vec<String>,
    formula: Option<&str>,
    target_column: Option<String>,
    has_external_target: bool,
    has_sample_weight: bool,
) -> PyResult<(String, Vec<String>, String, Option<String>)> {
    // No formula is the automatic formula `target ~ .`: the fit expands `.`
    // against the training table's schema in Rust
    // (`gam_terms::inference::automatic_formula`), the same expansion the CLI
    // and the library apply to `y ~ .`.
    let formula = formula.unwrap_or(AUTOMATIC_REST_TERM);
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
            py_value_error(
                "formula must include a response when y is not provided; without a formula, \
                 pass y (an array or the name of the target column)"
                    .to_string(),
            )
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
    let feature_names: Vec<String> = columns
        .iter()
        .filter(|column| *column != &target_name)
        .cloned()
        .collect();
    // A formula that parses reads a known set of columns; one of them absent
    // from X is a width/schema mismatch between the formula and the input,
    // reported against X's feature count. A formula that does not parse is
    // left to the fit, which reports it with its typed formula error; so is
    // an automatic `.` formula, which reads whatever columns X has.
    if let Ok(parsed) = parse_formula(&fit_formula) {
        let read = formula_columns(&parsed).map_err(|err| py_value_error(err.to_string()))?;
        if let Some(absent) = read
            .iter()
            .find(|name| *name != &target_name && !columns.iter().any(|column| column == *name))
        {
            return Err(py_value_error(format!(
                "formula '{fit_formula}' reads column '{absent}', but X has {} feature(s): {:?}",
                feature_names.len(),
                feature_names
            )));
        }
    }
    let weight_column = has_sample_weight.then(|| sklearn_weight_column(&columns, &target_name));
    Ok((fit_formula, feature_names, target_name, weight_column))
}

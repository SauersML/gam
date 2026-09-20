use super::*;
use std::collections::BTreeSet;

/// The columns a parsed formula reads: the response (or a survival response's
/// entry, exit and event columns), each term's variables and `by=` column, and
/// each slope surface's z column and terms.
pub fn formula_columns(parsed: &ParsedFormula) -> Result<BTreeSet<String>, WorkflowError> {
    let mut out = BTreeSet::<String>::new();
    if let Some((entry, exit, event)) =
        gam_terms::inference::formula_dsl::parse_surv_response(&parsed.response)?
    {
        out.extend(entry);
        out.insert(exit);
        out.insert(event);
    } else if let Some((left, right, event)) =
        gam_terms::inference::formula_dsl::parse_surv_interval_response(&parsed.response)?
    {
        out.insert(left);
        out.insert(right);
        out.insert(event);
    } else {
        out.insert(parsed.response.clone());
    }
    gam_terms::inference::formula_dsl::parsed_term_column_names(&parsed.terms, &mut out);
    for surface in &parsed.slope_surfaces {
        out.insert(surface.z_column.clone());
        gam_terms::inference::formula_dsl::parsed_term_column_names(&surface.terms, &mut out);
    }
    Ok(out)
}

/// Every column a formula fit reads: [`formula_columns`] of the main, noise and
/// slope formulas and of a CTN stage-1 recipe, the z, weight, offset and
/// noise-offset columns, the recipe's weight, offset, fold and group columns,
/// a frozen CTN's inputs and response, and the variables and `by=` columns of
/// smooth overrides.
///
/// This is the fit's input contract. `gam fit` loads exactly these columns and
/// the fit boundary validates exactly these, so a column the model never reads
/// cannot refuse, change or block a fit on any front door.
pub fn fit_required_columns(
    parsed: &ParsedFormula,
    config: &FitConfig,
) -> Result<BTreeSet<String>, WorkflowError> {
    use gam_terms::inference::formula_dsl::{parse_formula, parse_matching_auxiliary_formula};
    let mut required = formula_columns(parsed)?;
    if let Some(noise_formula) = config.noise_formula.as_deref() {
        let (_, parsed_noise) =
            parse_matching_auxiliary_formula(noise_formula, &parsed.response, "noise_formula")?;
        required.extend(formula_columns(&parsed_noise)?);
    }
    if let Some(slope_formula) = config.slope_formula.as_deref() {
        let (_, parsed_slope) =
            parse_matching_auxiliary_formula(slope_formula, &parsed.response, "slope_formula")?;
        required.extend(formula_columns(&parsed_slope)?);
    }
    required.extend(config.z_column.iter().cloned());
    required.extend(config.residual_columns.iter().cloned());
    required.extend(config.weight_column.iter().cloned());
    required.extend(config.offset_column.iter().cloned());
    required.extend(config.noise_offset_column.iter().cloned());
    if let Some(stage1) = config.ctn_stage1.as_ref() {
        let parsed_stage1 = parse_formula(&format!(
            "{} ~ {}",
            stage1.response_column, stage1.covariate_formula_rhs
        ))?;
        required.extend(formula_columns(&parsed_stage1)?);
        required.extend(stage1.weight_column.iter().cloned());
        required.extend(stage1.offset_column.iter().cloned());
        // The cross-fitting folds are read from these labels.
        required.extend(stage1.fold_column.iter().cloned());
        required.extend(stage1.group_column.iter().cloned());
    }
    // A frozen CTN's score is evaluated from its own inputs: its covariates,
    // offset and the stage-1 response it transforms.
    if let Some(frozen) = config.frozen_ctn.as_ref() {
        let transform = crate::inference::model::FittedModel::from_payload((*frozen.0).clone());
        required.extend(
            transform
                .prediction_required_columns()
                .map_err(|reason| WorkflowError::InvalidConfig { reason })?,
        );
        required.insert(parse_formula(&transform.formula)?.response);
    }
    if let Some(descriptors) = config
        .smooth_overrides
        .as_ref()
        .and_then(serde_json::Value::as_object)
    {
        for descriptor in descriptors.values().filter_map(serde_json::Value::as_object) {
            if let Some(vars) = descriptor.get("vars").and_then(serde_json::Value::as_array) {
                required.extend(
                    vars.iter()
                        .filter_map(serde_json::Value::as_str)
                        .map(str::to_string),
                );
            }
            if let Some(by) = descriptor.get("by").and_then(serde_json::Value::as_str) {
                required.insert(by.to_string());
            }
        }
    }
    Ok(required)
}

/// Expand the automatic `.` term of `formula` against `data`.
///
/// The columns `.` stands for are those no other part of the fit reads:
/// [`fit_required_columns`] of the formula without `.`, under `config`, are
/// reserved. The per-column rule is
/// [`gam_terms::inference::automatic_formula::expand_automatic_formula`], the
/// one implementation behind every front door. A formula without `.` comes
/// back unchanged with no notes.
pub fn expand_automatic_fit_formula(
    formula: &str,
    data: &Dataset,
    config: &FitConfig,
) -> Result<gam_terms::inference::automatic_formula::AutomaticFormula, WorkflowError> {
    use gam_terms::inference::automatic_formula::{
        AutomaticFormula, expand_automatic_formula, formula_has_automatic_term,
        formula_without_automatic_term,
    };
    let invalid = |reason: String| WorkflowError::InvalidConfig { reason };
    if !formula_has_automatic_term(formula)? {
        return Ok(AutomaticFormula {
            formula: formula.to_string(),
            notes: Vec::new(),
        });
    }
    let explicit = gam_terms::inference::formula_dsl::parse_formula(
        &formula_without_automatic_term(formula)?,
    )?;
    let reserved = fit_required_columns(&explicit, config)?;
    expand_automatic_formula(formula, data, &reserved).map_err(invalid)
}

pub(crate) fn resolve_continuous_column(
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    column_name: &str,
    role: &str,
) -> Result<Array1<f64>, WorkflowError> {
    let col_idx = resolve_role_col(col_map, column_name, role)?;
    let values = data.values.column(col_idx).to_owned();
    for (row_idx, value) in values.iter().enumerate() {
        if !value.is_finite() {
            // Row index is reported 1-based to match the rest of gam's data
            // validators (gam-data ingestion, gamfit `_tables.py`).
            let row = row_idx + 1;
            return Err(WorkflowError::InvalidData {
                column: column_name.to_string(),
                problem: format!("is the {role} column and has non-finite value {value} at row {row}"),
            });
        }
    }
    Ok(values)
}

#[cfg(test)]
mod weight_row_index_tests {
    use super::*;
    use gam_data::{ColumnKindTag, DataSchema, SchemaColumn};
    use ndarray::Array2;

    /// Build a single-column dataset named `w` whose only column carries the
    /// supplied weight values, so the weight validators can be exercised in
    /// isolation.
    fn weight_dataset(weights: &[f64]) -> Dataset {
        let nrows = weights.len();
        let values =
            Array2::from_shape_vec((nrows, 1), weights.to_vec()).expect("rectangular weight data");
        Dataset {
            headers: vec!["w".to_string()],
            values,
            schema: DataSchema {
                columns: vec![SchemaColumn {
                    name: "w".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                }],
            },
            column_kinds: vec![ColumnKindTag::Continuous],
        }
    }

    /// Parse the integer following the `at row ` token in a validator message.
    fn parsed_row(message: &str) -> usize {
        let tail = message
            .split("at row ")
            .nth(1)
            .unwrap_or_else(|| panic!("message has no `at row` token: {message}"));
        let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits
            .parse()
            .unwrap_or_else(|err| panic!("no row number after `at row` ({err}): {message}"))
    }

    /// Regression for #1597: a negative weight and a NaN weight at the SAME
    /// physical array row must report the SAME 1-based row number. Before the
    /// fix the non-negative (Rust) check reported a 0-based row while the
    /// non-finite path reported a 1-based row, so the same row 2 was named
    /// "row 2" and "row 3".
    #[test]
    fn negative_and_nonfinite_weight_report_same_one_based_row() {
        // Bad value sits at array index 2, i.e. the 3rd row (1-based).
        let neg = weight_dataset(&[1.0, 1.0, -1.0, 1.0, 1.0]);
        let nan = weight_dataset(&[1.0, 1.0, f64::NAN, 1.0, 1.0]);

        let neg_msg = match resolve_weight_column(&neg, &neg.column_map(), Some("w")) {
            Err(WorkflowError::InvalidData { column, problem }) if column == "w" => problem,
            other => panic!("expected InvalidData for negative weight, got {other:?}"),
        };
        let nan_msg = match resolve_weight_column(&nan, &nan.column_map(), Some("w")) {
            Err(WorkflowError::InvalidData { column, problem }) if column == "w" => problem,
            other => panic!("expected InvalidData for non-finite weight, got {other:?}"),
        };

        let neg_row = parsed_row(&neg_msg);
        let nan_row = parsed_row(&nan_msg);

        // Both messages must name the SAME row, and that row must be the
        // 1-based index (3) used by every other gam validator.
        assert_eq!(
            neg_row, 3,
            "negative-weight message must report 1-based row 3: {neg_msg}"
        );
        assert_eq!(
            nan_row, 3,
            "non-finite-weight message must report 1-based row 3: {nan_msg}"
        );
        assert_eq!(
            neg_row, nan_row,
            "negative and non-finite weight checks must agree on the row number: \
             {neg_msg} vs {nan_msg}"
        );
    }

    /// Every row carrying zero weight leaves nothing in the likelihood; that
    /// is a data error at the weight column, not a deep solver failure. A
    /// single positive weight is a valid (if tiny) fit and passes.
    #[test]
    fn all_zero_weights_are_rejected_as_invalid_data() {
        let zeros = weight_dataset(&[0.0, 0.0, 0.0, 0.0]);
        match resolve_fit_weight_column(&zeros, &zeros.column_map(), Some("w")) {
            Err(WorkflowError::InvalidData { column, problem }) => {
                assert_eq!(column, "w");
                assert!(problem.contains("no positive weight"), "{problem}");
            }
            other => panic!("expected InvalidData for all-zero weights, got {other:?}"),
        }

        let one_positive = weight_dataset(&[0.0, 0.0, 2.5, 0.0]);
        let weights = resolve_fit_weight_column(&one_positive, &one_positive.column_map(), Some("w"))
            .expect("zero weights alongside a positive weight are valid exclusions");
        assert_eq!(weights.to_vec(), vec![0.0, 0.0, 2.5, 0.0]);
    }
}

#[cfg(test)]
mod ctn_reserved_columns_tests {
    use super::*;
    use crate::fit_orchestration::CtnStage1Recipe;
    use crate::transformation_normal::TransformationNormalConfig;
    use gam_data::{ColumnKindTag, DataSchema, SchemaColumn};
    use ndarray::Array2;

    fn ctn_config() -> FitConfig {
        let mut recipe =
            CtnStage1Recipe::new("pgs", "x", TransformationNormalConfig::default(), None, None)
                .expect("valid stage-1 recipe");
        recipe.fold_column = Some("fold".to_string());
        recipe.group_column = Some("site".to_string());
        FitConfig {
            family: Some("bernoulli-marginal-slope".to_string()),
            ctn_stage1: Some(recipe),
            ..FitConfig::default()
        }
    }

    /// The cross-fitting fold and group labels are inputs of a CTN chain: the
    /// fit reads them to assign its folds.
    #[test]
    fn ctn_fold_and_group_columns_are_fit_inputs() {
        let parsed = gam_terms::inference::formula_dsl::parse_formula("event ~ age")
            .expect("formula parses");
        let required = fit_required_columns(&parsed, &ctn_config()).expect("required columns");
        for name in ["event", "age", "pgs", "x", "fold", "site"] {
            assert!(required.contains(name), "'{name}' missing from {required:?}");
        }
    }

    /// `.` stands for the columns no other part of the fit reads, so it must
    /// not turn the CTN's fold or group labels into outcome covariates.
    #[test]
    fn automatic_term_leaves_ctn_fold_and_group_columns_out() {
        let headers: Vec<String> = ["event", "age", "pgs", "x", "fold", "site"]
            .iter()
            .map(|name| name.to_string())
            .collect();
        let n = 40;
        let values = Array2::from_shape_fn((n, headers.len()), |(row, column)| match column {
            0 => (row % 2) as f64,
            1 => 20.0 + row as f64 * 1.37,
            2 => (row as f64 * 0.61).sin(),
            3 => row as f64 / n as f64,
            4 => (row % 5) as f64,
            _ => (row % 8) as f64 * 1.5,
        });
        let data = Dataset {
            headers: headers.clone(),
            values,
            schema: DataSchema {
                columns: headers
                    .iter()
                    .map(|name| SchemaColumn {
                        name: name.clone(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    })
                    .collect(),
            },
            column_kinds: vec![ColumnKindTag::Continuous; headers.len()],
        };
        let expanded = expand_automatic_fit_formula("event ~ .", &data, &ctn_config())
            .expect("automatic formula expands")
            .formula;
        assert!(expanded.contains("age"), "{expanded}");
        assert!(!expanded.contains("fold"), "fold labels became a covariate: {expanded}");
        assert!(!expanded.contains("site"), "group labels became a covariate: {expanded}");
    }
}

pub fn resolve_offset_column(
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    column_name: Option<&str>,
) -> Result<Array1<f64>, WorkflowError> {
    let Some(column_name) = column_name else {
        return Ok(Array1::zeros(data.values.nrows()));
    };
    resolve_continuous_column(data, col_map, column_name, "offset")
}

pub fn resolve_weight_column(
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    column_name: Option<&str>,
) -> Result<Array1<f64>, WorkflowError> {
    let Some(column_name) = column_name else {
        return Ok(Array1::ones(data.values.nrows()));
    };
    let values = resolve_continuous_column(data, col_map, column_name, "weights")?;
    // A prior weight scales its row's log-likelihood contribution, so it must
    // be non-negative; a zero weight excludes the row. Non-finite weights are
    // rejected by `resolve_continuous_column` / the gam-data fit boundary.
    // Row indices are 1-based to match the rest of gam's data validators.
    if let Some((row_idx, value)) = values.iter().enumerate().find(|(_, v)| **v < 0.0) {
        return Err(WorkflowError::InvalidData {
            column: column_name.to_string(),
            problem: format!(
                "is a prior-weight column and must be non-negative; found {value} at row {}",
                row_idx + 1
            ),
        });
    }
    Ok(values)
}

/// Fit-time weight column: [`resolve_weight_column`] plus the requirement that
/// the weights leave a data term to fit.
pub fn resolve_fit_weight_column(
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    column_name: Option<&str>,
) -> Result<Array1<f64>, WorkflowError> {
    let values = resolve_weight_column(data, col_map, column_name)?;
    let Some(column_name) = column_name else {
        return Ok(values);
    };
    // Every row's likelihood contribution is scaled by its prior weight, so an
    // all-zero weight vector leaves no data term at all: the fit would be the
    // prior alone. Reject it here rather than let the REML outer search fail
    // on an objective with no data in it.
    if !values.iter().any(|value| *value > 0.0) {
        return Err(no_positive_weight_error(column_name, values.len()));
    }
    Ok(values)
}

/// The data error for a prior-weight column that excludes every one of its
/// `nrows` rows; shared by the fit-time weight resolver and the zero-weight
/// row seam so both report the same typed error.
pub(crate) fn no_positive_weight_error(column_name: &str, nrows: usize) -> WorkflowError {
    WorkflowError::InvalidData {
        column: column_name.to_string(),
        problem: format!(
            "is a prior-weight column with no positive weight; a zero weight \
             excludes its row, so all {nrows} rows would be excluded from the likelihood"
        ),
    }
}

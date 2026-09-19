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
/// noise-offset columns, the recipe's weight and offset columns, and the
/// variables and `by=` columns of smooth overrides.
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
            return Err(WorkflowError::SchemaMismatch {
                reason: format!(
                    "{role} column '{column_name}' contains non-finite value at row {row}: {value}"
                ),
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
            Err(WorkflowError::SchemaMismatch { reason }) => reason,
            other => panic!("expected SchemaMismatch for negative weight, got {other:?}"),
        };
        let nan_msg = match resolve_weight_column(&nan, &nan.column_map(), Some("w")) {
            Err(WorkflowError::SchemaMismatch { reason }) => reason,
            other => panic!("expected SchemaMismatch for non-finite weight, got {other:?}"),
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
    for (row_idx, value) in values.iter().enumerate() {
        if *value < 0.0 {
            // Row index is reported 1-based to match the rest of gam's data
            // validators (gam-data ingestion, gamfit `_tables.py`).
            let row = row_idx + 1;
            return Err(WorkflowError::SchemaMismatch {
                reason: format!(
                    "weights column '{column_name}' must be non-negative; found {value} at row {row}"
                ),
            });
        }
    }
    Ok(values)
}

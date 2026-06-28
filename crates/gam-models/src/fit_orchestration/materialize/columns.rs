use super::*;

fn resolve_continuous_column(
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
            .unwrap_or_else(|_| panic!("no row number after `at row`: {message}"))
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

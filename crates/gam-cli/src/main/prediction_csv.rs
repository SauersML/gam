use super::*;

pub(crate) const STANDARD_PREDICTION_BASE_COLUMNS: [&str; 2] = ["eta", "mean"];
pub(crate) const GAUSSIAN_LOCATION_SCALE_BASE_COLUMNS: [&str; 3] = ["eta", "mean", "sigma"];
pub(crate) const SURVIVAL_PREDICTION_BASE_COLUMNS: [&str; 4] =
    ["eta", "survival_prob", "failure_prob", "risk_score"];
pub(crate) const SURVIVAL_BINARY_PREDICTION_BASE_COLUMNS: [&str; 6] = [
    "eta",
    "mean",
    "event_prob",
    "failure_prob",
    "survival_prob",
    "risk_score",
];
pub(crate) const PREDICTION_INTERVAL_COLUMNS: [&str; 2] = ["mean_lower", "mean_upper"];
pub(crate) const PREDICTION_STD_ERROR_COLUMN: &str = "std_error";

pub(crate) fn load_prediction_id_values(
    path: &Path,
    id_column: &str,
    expected_rows: usize,
) -> Result<Vec<String>, String> {
    if id_column.trim().is_empty() {
        return Err("--id-column must be a non-empty column name".to_string());
    }
    let projected = load_dataset_projected(path, &[id_column.to_string()])?;
    if projected.values.nrows() != expected_rows {
        return Err(format!(
            "id column '{id_column}' row count {} does not match prediction row count {expected_rows}",
            projected.values.nrows()
        ));
    }
    let col_idx = resolve_role_col(&projected.column_map(), id_column, "id")?;
    let schema_col = projected
        .schema
        .columns
        .iter()
        .find(|column| column.name == id_column)
        .ok_or_else(|| format!("id column '{id_column}' missing from inferred schema"))?;
    let mut out = Vec::<String>::with_capacity(projected.values.nrows());
    for row_idx in 0..projected.values.nrows() {
        let value = projected.values[[row_idx, col_idx]];
        if !value.is_finite() {
            return Err(format!(
                "id column '{id_column}' contains non-finite value at row {row_idx}"
            ));
        }
        let rendered = match schema_col.kind {
            ColumnKindTag::Categorical => {
                let level_idx = value.round() as usize;
                schema_col.levels.get(level_idx).cloned().ok_or_else(|| {
                    format!(
                        "id column '{id_column}' categorical code {level_idx} at row {row_idx} is out of bounds"
                    )
                })?
            }
            ColumnKindTag::Continuous | ColumnKindTag::Binary => format_id_number(value),
        };
        out.push(rendered);
    }
    Ok(out)
}

pub(crate) fn format_id_number(value: f64) -> String {
    if (value - value.round()).abs() <= 1e-9 {
        format!("{value:.0}")
    } else {
        format!("{value:.12}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

pub(crate) fn prepend_id_column_to_prediction_csv(
    path: &Path,
    id_column: &str,
    id_values: &[String],
) -> Result<(), String> {
    let mut rdr = csv::Reader::from_path(path)
        .map_err(|e| format!("failed to read prediction csv '{}': {e}", path.display()))?;
    let headers = rdr
        .headers()
        .map_err(|e| format!("failed to read prediction csv header: {e}"))?
        .clone();
    if headers.iter().any(|name| name == id_column) {
        return Err(format!(
            "prediction output already contains id column '{id_column}'"
        ));
    }

    let tmp_path = path.with_extension("tmp-id-column.csv");
    let mut wtr = WriterBuilder::new()
        .has_headers(true)
        .from_path(&tmp_path)
        .map_err(|e| {
            format!(
                "failed to create temporary prediction csv '{}': {e}",
                tmp_path.display()
            )
        })?;
    let mut out_headers = Vec::<String>::with_capacity(headers.len() + 1);
    out_headers.push(id_column.to_string());
    out_headers.extend(headers.iter().map(str::to_string));
    wtr.write_record(&out_headers)
        .map_err(|e| format!("failed writing prediction csv header with id column: {e}"))?;

    let mut row_count = 0usize;
    for record in rdr.records() {
        let record = record.map_err(|e| format!("failed reading prediction csv row: {e}"))?;
        let id = id_values.get(row_count).ok_or_else(|| {
            format!(
                "prediction csv has more rows than id column '{id_column}' (first extra row index {row_count})"
            )
        })?;
        let mut out_record = Vec::<String>::with_capacity(record.len() + 1);
        out_record.push(id.clone());
        out_record.extend(record.iter().map(str::to_string));
        wtr.write_record(&out_record)
            .map_err(|e| format!("failed writing prediction csv row {row_count}: {e}"))?;
        row_count += 1;
    }
    if row_count != id_values.len() {
        return Err(format!(
            "prediction csv row count {row_count} does not match id column '{id_column}' row count {}",
            id_values.len()
        ));
    }
    wtr.flush()
        .map_err(|e| format!("failed to flush prediction csv with id column: {e}"))?;
    std::fs::rename(&tmp_path, path).map_err(|e| {
        format!(
            "failed to replace prediction csv '{}' with id-column version '{}': {e}",
            path.display(),
            tmp_path.display()
        )
    })?;
    Ok(())
}

/// Unified CSV prediction writer.  Each column is a `(name, data)` pair;
/// the function writes a header row from the names and one data row per
/// element, formatting every value to 12 decimal places.
///
/// All columns must have the same length.  An empty column list is an error.
pub(crate) fn write_prediction_csv_unified(
    path: &Path,
    columns: &[(&str, &[f64])],
) -> CliResult<()> {
    if columns.is_empty() {
        return Err(CliError::Internal {
            reason: "internal error: write_prediction_csv_unified called with no columns"
                .to_string(),
        });
    }
    let n = columns[0].1.len();
    for (name, data) in columns.iter() {
        if data.len() != n {
            return Err(CliError::Internal {
                reason: format!(
                    "internal error: column '{}' has length {} but expected {}",
                    name,
                    data.len(),
                    n,
                ),
            });
        }
    }

    let mut wtr = WriterBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| CliError::FileWriteFailed {
            reason: format!("failed to create output csv '{}': {e}", path.display()),
        })?;

    let headers: Vec<&str> = columns.iter().map(|(name, _)| *name).collect();
    wtr.write_record(&headers)
        .map_err(|e| CliError::FileWriteFailed {
            reason: format!("failed writing csv header: {e}"),
        })?;

    // Validate all prediction values are finite before writing.
    // NaN or Inf in clinical output would be dangerous.
    for (col_name, data) in columns {
        for (i, val) in data.iter().enumerate() {
            if !val.is_finite() {
                return Err(CliError::Internal {
                    reason: format!(
                        "non-finite prediction value in column '{}' at row {}: {}",
                        col_name, i, val
                    ),
                });
            }
        }
    }

    for i in 0..n {
        let row: Vec<String> = columns
            .iter()
            .map(|(_, data)| format!("{:.12}", data[i]))
            .collect();
        wtr.write_record(&row)
            .map_err(|e| CliError::FileWriteFailed {
                reason: format!("failed writing csv row {i}: {e}"),
            })?;
    }

    wtr.flush().map_err(|e| CliError::FileWriteFailed {
        reason: format!("failed to flush csv writer: {e}"),
    })?;
    Ok(())
}

/// Convenience wrapper: builds a standard (non-survival, non-location-scale)
/// prediction column list and delegates to [`write_prediction_csv_unified`].
pub(crate) fn write_prediction_csv(
    path: &Path,
    eta: ArrayView1<'_, f64>,
    mean: ArrayView1<'_, f64>,
    eta_se: Option<ArrayView1<'_, f64>>,
    mean_lower: Option<ArrayView1<'_, f64>>,
    mean_upper: Option<ArrayView1<'_, f64>>,
) -> CliResult<()> {
    // Materialise views into contiguous vecs so we can pass &[f64] slices.
    let eta_v: Vec<f64> = eta.to_vec();
    let mean_v: Vec<f64> = mean.to_vec();

    let mut cols: Vec<(&str, &[f64])> = vec![
        (STANDARD_PREDICTION_BASE_COLUMNS[0], &eta_v),
        (STANDARD_PREDICTION_BASE_COLUMNS[1], &mean_v),
    ];

    let se_v: Vec<f64>;
    let lo_v: Vec<f64>;
    let hi_v: Vec<f64>;
    if let Some(se) = eta_se {
        se_v = se.to_vec();
        lo_v = mean_lower
            .ok_or_else(|| {
                "internal error: mean_lower missing while std_error is present".to_string()
            })?
            .to_vec();
        hi_v = mean_upper
            .ok_or_else(|| {
                "internal error: mean_upper missing while std_error is present".to_string()
            })?
            .to_vec();
        cols.push((PREDICTION_STD_ERROR_COLUMN, &se_v));
        cols.push((PREDICTION_INTERVAL_COLUMNS[0], &lo_v));
        cols.push((PREDICTION_INTERVAL_COLUMNS[1], &hi_v));
    } else if let (Some(lo), Some(hi)) = (mean_lower, mean_upper) {
        lo_v = lo.to_vec();
        hi_v = hi.to_vec();
        cols.push((PREDICTION_INTERVAL_COLUMNS[0], &lo_v));
        cols.push((PREDICTION_INTERVAL_COLUMNS[1], &hi_v));
    } else if mean_lower.is_some() {
        return Err(CliError::Internal {
            reason: "internal error: mean_upper missing while mean_lower is present".to_string(),
        });
    } else if mean_upper.is_some() {
        return Err(CliError::Internal {
            reason: "internal error: mean_lower missing while mean_upper is present".to_string(),
        });
    }

    write_prediction_csv_unified(path, &cols)
}

/// Convenience wrapper for Gaussian location-scale predictions (always
/// includes a `sigma` column).
pub(crate) fn write_gaussian_location_scale_prediction_csv(
    path: &Path,
    eta: ArrayView1<'_, f64>,
    mean: ArrayView1<'_, f64>,
    sigma: ArrayView1<'_, f64>,
    mean_standard_error: Option<ArrayView1<'_, f64>>,
    mean_lower: Option<ArrayView1<'_, f64>>,
    mean_upper: Option<ArrayView1<'_, f64>>,
) -> CliResult<()> {
    let eta_v: Vec<f64> = eta.to_vec();
    let mean_v: Vec<f64> = mean.to_vec();
    let sigma_v: Vec<f64> = sigma.to_vec();

    let mut cols: Vec<(&str, &[f64])> = vec![
        (GAUSSIAN_LOCATION_SCALE_BASE_COLUMNS[0], &eta_v),
        (GAUSSIAN_LOCATION_SCALE_BASE_COLUMNS[1], &mean_v),
        (GAUSSIAN_LOCATION_SCALE_BASE_COLUMNS[2], &sigma_v),
    ];

    let se_v: Vec<f64>;
    let lo_v: Vec<f64>;
    let hi_v: Vec<f64>;
    if let Some(se) = mean_standard_error {
        se_v = se.to_vec();
        lo_v = mean_lower
            .ok_or_else(|| CliError::Internal {
                reason: "internal error: mean_lower missing while std_error is present".to_string(),
            })?
            .to_vec();
        hi_v = mean_upper
            .ok_or_else(|| CliError::Internal {
                reason: "internal error: mean_upper missing while std_error is present".to_string(),
            })?
            .to_vec();
        cols.push((PREDICTION_STD_ERROR_COLUMN, &se_v));
        cols.push((PREDICTION_INTERVAL_COLUMNS[0], &lo_v));
        cols.push((PREDICTION_INTERVAL_COLUMNS[1], &hi_v));
    } else if let (Some(lo), Some(hi)) = (mean_lower, mean_upper) {
        lo_v = lo.to_vec();
        hi_v = hi.to_vec();
        cols.push((PREDICTION_INTERVAL_COLUMNS[0], &lo_v));
        cols.push((PREDICTION_INTERVAL_COLUMNS[1], &hi_v));
    } else if mean_lower.is_some() {
        return Err(CliError::Internal {
            reason: "internal error: mean_upper missing while mean_lower is present".to_string(),
        });
    } else if mean_upper.is_some() {
        return Err(CliError::Internal {
            reason: "internal error: mean_lower missing while mean_upper is present".to_string(),
        });
    }

    write_prediction_csv_unified(path, &cols)
}

/// Convenience wrapper for survival predictions. Survival output uses explicit
/// probability semantics because the event probability is `1 - survival_prob`.
pub(crate) fn write_survival_prediction_csv(
    path: &Path,
    eta: ArrayView1<'_, f64>,
    survival_prob: ArrayView1<'_, f64>,
    eta_se: Option<ArrayView1<'_, f64>>,
    survival_lower: Option<ArrayView1<'_, f64>>,
    survival_upper: Option<ArrayView1<'_, f64>>,
) -> CliResult<()> {
    let eta_v: Vec<f64> = eta.to_vec();
    let surv_v: Vec<f64> = survival_prob.iter().map(|&v| v.clamp(0.0, 1.0)).collect();
    let risk_v: Vec<f64> = eta_v.clone();
    let fail_v: Vec<f64> = surv_v.iter().map(|&s| (1.0 - s).clamp(0.0, 1.0)).collect();

    let mut cols: Vec<(&str, &[f64])> = vec![
        (SURVIVAL_PREDICTION_BASE_COLUMNS[0], &eta_v),
        (SURVIVAL_PREDICTION_BASE_COLUMNS[1], &surv_v),
        (SURVIVAL_PREDICTION_BASE_COLUMNS[2], &fail_v),
        (SURVIVAL_PREDICTION_BASE_COLUMNS[3], &risk_v),
    ];

    let se_v: Vec<f64>;
    let lo_v: Vec<f64>;
    let hi_v: Vec<f64>;
    if let Some(se) = eta_se {
        se_v = se.to_vec();
        lo_v = survival_lower
            .ok_or_else(|| {
                "internal error: survival_lower missing while std_error is present".to_string()
            })?
            .to_vec();
        hi_v = survival_upper
            .ok_or_else(|| {
                "internal error: survival_upper missing while std_error is present".to_string()
            })?
            .to_vec();
        cols.push((PREDICTION_STD_ERROR_COLUMN, &se_v));
        cols.push((PREDICTION_INTERVAL_COLUMNS[0], &lo_v));
        cols.push((PREDICTION_INTERVAL_COLUMNS[1], &hi_v));
    } else if let (Some(lo), Some(hi)) = (survival_lower, survival_upper) {
        lo_v = lo.to_vec();
        hi_v = hi.to_vec();
        cols.push((PREDICTION_INTERVAL_COLUMNS[0], &lo_v));
        cols.push((PREDICTION_INTERVAL_COLUMNS[1], &hi_v));
    } else if survival_lower.is_some() {
        return Err(CliError::Internal {
            reason: "internal error: survival_upper missing while survival_lower is present"
                .to_string(),
        });
    } else if survival_upper.is_some() {
        return Err(CliError::Internal {
            reason: "internal error: survival_lower missing while survival_upper is present"
                .to_string(),
        });
    }

    write_prediction_csv_unified(path, &cols)
}

/// Convenience wrapper for binary deployment predictions backed by a survival
/// hazard window (includes explicit `event_prob`, `failure_prob`, and
/// `survival_prob` columns).
pub(crate) fn write_survival_binary_prediction_csv(
    path: &Path,
    eta: ArrayView1<'_, f64>,
    event_prob: ArrayView1<'_, f64>,
    eta_se: Option<ArrayView1<'_, f64>>,
    event_lower: Option<ArrayView1<'_, f64>>,
    event_upper: Option<ArrayView1<'_, f64>>,
) -> CliResult<()> {
    let eta_v: Vec<f64> = eta.to_vec();
    let event_v: Vec<f64> = event_prob.iter().map(|&v| v.clamp(0.0, 1.0)).collect();
    let risk_v: Vec<f64> = eta_v.clone();
    let survival_v: Vec<f64> = event_v.iter().map(|&p| (1.0 - p).clamp(0.0, 1.0)).collect();

    let mut cols: Vec<(&str, &[f64])> = vec![
        (SURVIVAL_BINARY_PREDICTION_BASE_COLUMNS[0], &eta_v),
        (SURVIVAL_BINARY_PREDICTION_BASE_COLUMNS[1], &event_v),
        (SURVIVAL_BINARY_PREDICTION_BASE_COLUMNS[2], &event_v),
        (SURVIVAL_BINARY_PREDICTION_BASE_COLUMNS[3], &event_v),
        (SURVIVAL_BINARY_PREDICTION_BASE_COLUMNS[4], &survival_v),
        (SURVIVAL_BINARY_PREDICTION_BASE_COLUMNS[5], &risk_v),
    ];

    let se_v: Vec<f64>;
    let lo_v: Vec<f64>;
    let hi_v: Vec<f64>;
    if let Some(se) = eta_se {
        se_v = se.to_vec();
        lo_v = event_lower
            .ok_or_else(|| CliError::Internal {
                reason: "internal error: event_lower missing while std_error is present"
                    .to_string(),
            })?
            .to_vec();
        hi_v = event_upper
            .ok_or_else(|| CliError::Internal {
                reason: "internal error: event_upper missing while std_error is present"
                    .to_string(),
            })?
            .to_vec();
        cols.push((PREDICTION_STD_ERROR_COLUMN, &se_v));
        cols.push((PREDICTION_INTERVAL_COLUMNS[0], &lo_v));
        cols.push((PREDICTION_INTERVAL_COLUMNS[1], &hi_v));
    } else if let (Some(lo), Some(hi)) = (event_lower, event_upper) {
        lo_v = lo.to_vec();
        hi_v = hi.to_vec();
        cols.push((PREDICTION_INTERVAL_COLUMNS[0], &lo_v));
        cols.push((PREDICTION_INTERVAL_COLUMNS[1], &hi_v));
    } else if event_lower.is_some() {
        return Err(CliError::Internal {
            reason: "internal error: event_upper missing while event_lower is present".to_string(),
        });
    } else if event_upper.is_some() {
        return Err(CliError::Internal {
            reason: "internal error: event_lower missing while event_upper is present".to_string(),
        });
    }

    write_prediction_csv_unified(path, &cols)
}

use super::*;

pub(crate) const STANDARD_PREDICTION_BASE_COLUMNS: [&str; 3] = [
    "linear_predictor_plugin",
    "mean_plugin",
    "posterior_mean",
];
pub(crate) const STANDARD_PREDICTION_INTERVAL_COLUMNS: [&str; 2] =
    ["posterior_mean_lower", "posterior_mean_upper"];
pub(crate) const STANDARD_PREDICTION_STD_ERROR_COLUMN: &str =
    "posterior_mean_standard_error";
pub(crate) const STANDARD_PREDICTION_ETA_STD_ERROR_COLUMN: &str =
    "linear_predictor_standard_error";
pub(crate) const PREDICTION_NOISE_SCALE_COLUMN: &str = "noise_scale";
pub(crate) const SPECIALIZED_PREDICTION_BASE_COLUMNS: [&str; 2] = ["eta", "mean"];
/// Survival prediction columns. `survival_prob_plugin` is the plug-in
/// `S(η̂)` at the fitted coefficients; `survival_prob` is the posterior mean
/// `E[S(η)]` under the coefficient posterior — the point estimand every
/// prediction surface reports — and `failure_prob` is its complement. Both
/// estimands are published by name; there is no mode switch between them.
pub(crate) const SURVIVAL_PREDICTION_BASE_COLUMNS: [&str; 5] = [
    "eta",
    "survival_prob_plugin",
    "survival_prob",
    "failure_prob",
    "risk_score",
];
/// Latent-window event-probability columns; `mean_plugin` is the plug-in event
/// probability beside the posterior-mean `mean`, exactly as the standard
/// surface carries `mean_plugin` beside `posterior_mean`.
pub(crate) const SURVIVAL_BINARY_PREDICTION_BASE_COLUMNS: [&str; 7] = [
    "eta",
    "mean_plugin",
    "mean",
    "event_prob",
    "failure_prob",
    "survival_prob",
    "risk_score",
];
pub(crate) const PREDICTION_INTERVAL_COLUMNS: [&str; 2] = ["mean_lower", "mean_upper"];
pub(crate) const PREDICTION_STD_ERROR_COLUMN: &str = "std_error";
/// Posterior SD of the linear predictor `η` on the class-specific schemas that
/// publish it. `std_error` is the response-scale posterior SD of the published
/// mean, the quantity every prediction table's `std_error` /
/// `posterior_mean_standard_error` column carries; the link-scale η SD is a
/// different quantity and never rides under that name.
pub(crate) const PREDICTION_ETA_STD_ERROR_COLUMN: &str = "eta_std_error";

/// A posterior band on the published response-scale mean: its posterior
/// standard deviation and its credible bounds, which always travel together.
#[derive(Clone, Copy)]
pub(crate) struct ResponseBand<'a> {
    pub(crate) std_error: ArrayView1<'a, f64>,
    pub(crate) lower: ArrayView1<'a, f64>,
    pub(crate) upper: ArrayView1<'a, f64>,
}

impl<'a> ResponseBand<'a> {
    /// Pairs the three band columns an uncertainty evaluator returns. A partial
    /// band is an internal contract violation, never a smaller table.
    pub(crate) fn from_parts(
        std_error: Option<ArrayView1<'a, f64>>,
        lower: Option<ArrayView1<'a, f64>>,
        upper: Option<ArrayView1<'a, f64>>,
    ) -> CliResult<Option<Self>> {
        match (std_error, lower, upper) {
            (Some(std_error), Some(lower), Some(upper)) => Ok(Some(Self {
                std_error,
                lower,
                upper,
            })),
            (None, None, None) => Ok(None),
            (std_error, lower, upper) => Err(CliError::Internal {
                reason: format!(
                    "internal error: a prediction band needs its standard error and both bounds \
                     together (std_error present: {}, lower present: {}, upper present: {})",
                    std_error.is_some(),
                    lower.is_some(),
                    upper.is_some()
                ),
            }),
        }
    }

    fn append_to(self, columns: &mut Vec<(&'static str, Vec<f64>)>) {
        columns.push((PREDICTION_STD_ERROR_COLUMN, self.std_error.to_vec()));
        columns.push((PREDICTION_INTERVAL_COLUMNS[0], self.lower.to_vec()));
        columns.push((PREDICTION_INTERVAL_COLUMNS[1], self.upper.to_vec()));
    }
}

fn write_owned_prediction_columns(
    path: &Path,
    columns: &[(&'static str, Vec<f64>)],
) -> CliResult<()> {
    let borrowed: Vec<(&str, &[f64])> = columns
        .iter()
        .map(|(name, values)| (*name, values.as_slice()))
        .collect();
    write_prediction_csv_unified(path, &borrowed)
}

/// Read the `--id-column` values to echo into prediction output.
///
/// The ID column is carried through, not modelled, so it is read as text
/// ([`gam::data::load_column_text`]) and never passed through the numeric
/// encoder, whose re-rendering would turn `00123` into `123` and round int64
/// keys above 2^53 onto their neighbours.
pub(crate) fn load_prediction_id_values(
    path: &Path,
    id_column: &str,
    expected_rows: usize,
) -> Result<Vec<String>, String> {
    if id_column.trim().is_empty() {
        return Err("--id-column must be a non-empty column name".to_string());
    }
    let ids = gam::data::load_column_text(path, id_column).map_err(|e| e.to_string())?;
    if ids.len() != expected_rows {
        return Err(format!(
            "id column '{id_column}' row count {} does not match prediction row count {expected_rows}",
            ids.len()
        ));
    }
    Ok(ids)
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

/// Class-specific `eta,mean` writer (transformation-normal and every other
/// class without the estimand-explicit or survival schema), with the
/// response-scale band when one was requested.
pub(crate) fn write_prediction_csv(
    path: &Path,
    eta: ArrayView1<'_, f64>,
    mean: ArrayView1<'_, f64>,
    band: Option<ResponseBand<'_>>,
) -> CliResult<()> {
    let mut columns = vec![
        (SPECIALIZED_PREDICTION_BASE_COLUMNS[0], eta.to_vec()),
        (SPECIALIZED_PREDICTION_BASE_COLUMNS[1], mean.to_vec()),
    ];
    if let Some(band) = band {
        band.append_to(&mut columns);
    }
    write_owned_prediction_columns(path, &columns)
}

/// Prediction writer for every model class that publishes the
/// estimand-explicit schema (#2785/#2803).
///
/// The plug-in pair is always present and coherent; the posterior response
/// mean is present unless the caller explicitly forced a curved-link plug-in
/// prediction. Location-scale classes add their fitted response-side scale as
/// `noise_scale`. Posterior uncertainty columns are named for the posterior
/// estimand they accompany.
pub(crate) fn write_estimand_explicit_prediction_csv(
    path: &Path,
    linear_predictor_plugin: ArrayView1<'_, f64>,
    mean_plugin: ArrayView1<'_, f64>,
    posterior_mean: Option<ArrayView1<'_, f64>>,
    noise_scale: Option<ArrayView1<'_, f64>>,
    expectile_curves: &[(String, Array1<f64>)],
    linear_predictor_standard_error: Option<ArrayView1<'_, f64>>,
    posterior_mean_standard_error: Option<ArrayView1<'_, f64>>,
    posterior_mean_lower: Option<ArrayView1<'_, f64>>,
    posterior_mean_upper: Option<ArrayView1<'_, f64>>,
) -> CliResult<()> {
    let linear_predictor_plugin = linear_predictor_plugin.to_vec();
    let mean_plugin = mean_plugin.to_vec();
    let posterior_mean = posterior_mean.map(|values| values.to_vec());
    let mut columns: Vec<(&str, &[f64])> = vec![
        (
            STANDARD_PREDICTION_BASE_COLUMNS[0],
            &linear_predictor_plugin,
        ),
        (STANDARD_PREDICTION_BASE_COLUMNS[1], &mean_plugin),
    ];
    if let Some(values) = posterior_mean.as_ref() {
        columns.push((STANDARD_PREDICTION_BASE_COLUMNS[2], values));
    }
    let noise_scale = noise_scale.map(|values| values.to_vec());
    if let Some(values) = noise_scale.as_ref() {
        columns.push((PREDICTION_NOISE_SCALE_COLUMN, values));
    }
    // A joint expectile fit's level curves, in increasing level order.
    for (name, curve) in expectile_curves {
        let values = curve.as_slice().ok_or_else(|| CliError::Internal {
            reason: format!("expectile curve `{name}` is not contiguous"),
        })?;
        columns.push((name.as_str(), values));
    }

    let eta_standard_error = linear_predictor_standard_error.map(|values| values.to_vec());
    let standard_error = posterior_mean_standard_error.map(|values| values.to_vec());
    let lower = posterior_mean_lower.map(|values| values.to_vec());
    let upper = posterior_mean_upper.map(|values| values.to_vec());
    match (
        eta_standard_error.as_ref(),
        standard_error.as_ref(),
        lower.as_ref(),
        upper.as_ref(),
    ) {
        (Some(eta_standard_error), Some(standard_error), Some(lower), Some(upper)) => {
            if posterior_mean.is_none() {
                return Err(CliError::Internal {
                    reason: "posterior uncertainty cannot be emitted without posterior_mean"
                        .to_string(),
                });
            }
            columns.push((STANDARD_PREDICTION_ETA_STD_ERROR_COLUMN, eta_standard_error));
            columns.push((
                STANDARD_PREDICTION_STD_ERROR_COLUMN,
                standard_error,
            ));
            columns.push((STANDARD_PREDICTION_INTERVAL_COLUMNS[0], lower));
            columns.push((STANDARD_PREDICTION_INTERVAL_COLUMNS[1], upper));
        }
        (None, None, None, None) => {}
        _ => {
            return Err(CliError::Internal {
                reason: "standard prediction requires both posterior standard errors and both bounds together"
                    .to_string(),
            });
        }
    }
    write_prediction_csv_unified(path, &columns)
}

/// Survival prediction writer. Survival output uses explicit probability
/// semantics because the event probability is `1 - survival_prob`. The band's
/// `std_error` is the posterior SD of `survival_prob`; `eta_std_error` is the
/// posterior SD of `eta`.
pub(crate) fn write_survival_prediction_csv(
    path: &Path,
    eta: ArrayView1<'_, f64>,
    survival_prob_plugin: ArrayView1<'_, f64>,
    survival_prob: ArrayView1<'_, f64>,
    eta_std_error: Option<ArrayView1<'_, f64>>,
    band: Option<ResponseBand<'_>>,
) -> CliResult<()> {
    let survival: Vec<f64> = survival_prob.iter().map(|&v| v.clamp(0.0, 1.0)).collect();
    let failure: Vec<f64> = survival
        .iter()
        .map(|&s| (1.0 - s).clamp(0.0, 1.0))
        .collect();
    let mut columns = vec![
        (SURVIVAL_PREDICTION_BASE_COLUMNS[0], eta.to_vec()),
        (
            SURVIVAL_PREDICTION_BASE_COLUMNS[1],
            survival_prob_plugin
                .iter()
                .map(|&v| v.clamp(0.0, 1.0))
                .collect(),
        ),
        (SURVIVAL_PREDICTION_BASE_COLUMNS[2], survival),
        (SURVIVAL_PREDICTION_BASE_COLUMNS[3], failure),
        (SURVIVAL_PREDICTION_BASE_COLUMNS[4], eta.to_vec()),
    ];
    if let Some(values) = eta_std_error {
        columns.push((PREDICTION_ETA_STD_ERROR_COLUMN, values.to_vec()));
    }
    if let Some(band) = band {
        band.append_to(&mut columns);
    }
    write_owned_prediction_columns(path, &columns)
}

/// Writer for binary deployment predictions backed by a survival hazard window
/// (includes explicit `event_prob`, `failure_prob`, and `survival_prob`
/// columns). The band's `std_error` is the posterior SD of the event
/// probability `mean`.
pub(crate) fn write_survival_binary_prediction_csv(
    path: &Path,
    eta: ArrayView1<'_, f64>,
    event_prob_plugin: ArrayView1<'_, f64>,
    event_prob: ArrayView1<'_, f64>,
    band: Option<ResponseBand<'_>>,
) -> CliResult<()> {
    let event: Vec<f64> = event_prob.iter().map(|&v| v.clamp(0.0, 1.0)).collect();
    let survival: Vec<f64> = event.iter().map(|&p| (1.0 - p).clamp(0.0, 1.0)).collect();
    let mut columns = vec![
        (SURVIVAL_BINARY_PREDICTION_BASE_COLUMNS[0], eta.to_vec()),
        (
            SURVIVAL_BINARY_PREDICTION_BASE_COLUMNS[1],
            event_prob_plugin
                .iter()
                .map(|&v| v.clamp(0.0, 1.0))
                .collect(),
        ),
        (SURVIVAL_BINARY_PREDICTION_BASE_COLUMNS[2], event.clone()),
        (SURVIVAL_BINARY_PREDICTION_BASE_COLUMNS[3], event.clone()),
        (SURVIVAL_BINARY_PREDICTION_BASE_COLUMNS[4], event),
        (SURVIVAL_BINARY_PREDICTION_BASE_COLUMNS[5], survival),
        (SURVIVAL_BINARY_PREDICTION_BASE_COLUMNS[6], eta.to_vec()),
    ];
    if let Some(band) = band {
        band.append_to(&mut columns);
    }
    write_owned_prediction_columns(path, &columns)
}

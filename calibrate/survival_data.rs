use crate::calibrate::survival::{
    AgeTransform, CovariateViews, SurvivalError, SurvivalPredictionInputs, SurvivalTrainingData,
    validate_survival_inputs,
};
use ndarray::{Array1, Array2};
use polars::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;
use thiserror::Error;

/// Errors surfaced while reading or validating survival datasets.
#[derive(Debug, Error)]
pub enum SurvivalDataError {
    #[error("Error from the underlying Polars library: {0}")]
    Polars(#[from] PolarsError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("The required column '{0}' was not found in the input file.")]
    ColumnNotFound(String),
    #[error(
        "Column '{column_name}' could not be converted to the expected type '{expected_type}'. (Found type: {found_type})"
    )]
    ColumnWrongType {
        column_name: String,
        expected_type: &'static str,
        found_type: String,
    },
    #[error("Missing or null values were found in the column '{0}'.")]
    MissingValues(String),
    #[error(
        "Column '{column_name}' has {found} rows but {expected} were expected based on age_entry."
    )]
    LengthMismatch {
        column_name: String,
        expected: usize,
        found: usize,
    },
    #[error("Validation error: {0}")]
    Validation(#[from] SurvivalError),
}

/// Bundle containing frequency-weighted survival training data and the cached age transform.
#[derive(Debug)]
pub struct SurvivalTrainingBundle {
    pub data: SurvivalTrainingData,
    pub age_transform: AgeTransform,
}

/// Owned arrays backing `SurvivalPredictionInputs` alongside the raw covariates.
#[derive(Debug)]
pub struct SurvivalPredictionData {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,
    pub event_competing: Array1<u8>,
    pub sample_weight: Array1<f64>,
    pub pgs: Array1<f64>,
    pub sex: Array1<f64>,
    pub pcs: Array2<f64>,
    pub extra_static_covariates: Array2<f64>,
    pub extra_static_names: Vec<String>,
}

impl SurvivalPredictionData {
    /// Borrow the owned arrays as `SurvivalPredictionInputs` suitable for scoring.
    pub fn as_inputs(&self) -> SurvivalPredictionInputs<'_> {
        SurvivalPredictionInputs {
            age_entry: self.age_entry.view(),
            age_exit: self.age_exit.view(),
            event_target: self.event_target.view(),
            event_competing: self.event_competing.view(),
            sample_weight: self.sample_weight.view(),
            covariates: CovariateViews {
                pgs: self.pgs.view(),
                sex: self.sex.view(),
                pcs: self.pcs.view(),
                static_covariates: self.extra_static_covariates.view(),
            },
        }
    }
}

/// Load survival training data from a TSV or Parquet file, validating and caching the age transform.
pub fn load_survival_training_data(
    path: &str,
    num_pcs: usize,
    guard_delta: f64,
) -> Result<SurvivalTrainingBundle, SurvivalDataError> {
    let arrays = read_survival_arrays(path, num_pcs)?;

    validate_survival_inputs(
        arrays.age_entry.view(),
        arrays.age_exit.view(),
        arrays.event_target.view(),
        arrays.event_competing.view(),
        arrays.sample_weight.view(),
        arrays.pgs.view(),
        arrays.sex.view(),
        arrays.pcs.view(),
        arrays.extra_static_covariates.view(),
    )?;

    let age_transform = AgeTransform::from_training(&arrays.age_entry, guard_delta)?;

    Ok(SurvivalTrainingBundle {
        data: SurvivalTrainingData {
            age_entry: arrays.age_entry,
            age_exit: arrays.age_exit,
            event_target: arrays.event_target,
            event_competing: arrays.event_competing,
            sample_weight: arrays.sample_weight,
            pgs: arrays.pgs,
            sex: arrays.sex,
            pcs: arrays.pcs,
            extra_static_covariates: arrays.extra_static_covariates,
            extra_static_names: arrays.extra_static_names,
        },
        age_transform,
    })
}

/// Load survival prediction data from a TSV or Parquet file, validating the arrays and
/// preparing owned covariate storage.
pub fn load_survival_prediction_data(
    path: &str,
    num_pcs: usize,
) -> Result<SurvivalPredictionData, SurvivalDataError> {
    let arrays = read_survival_arrays(path, num_pcs)?;

    validate_survival_inputs(
        arrays.age_entry.view(),
        arrays.age_exit.view(),
        arrays.event_target.view(),
        arrays.event_competing.view(),
        arrays.sample_weight.view(),
        arrays.pgs.view(),
        arrays.sex.view(),
        arrays.pcs.view(),
        arrays.extra_static_covariates.view(),
    )?;

    Ok(SurvivalPredictionData {
        age_entry: arrays.age_entry,
        age_exit: arrays.age_exit,
        event_target: arrays.event_target,
        event_competing: arrays.event_competing,
        sample_weight: arrays.sample_weight,
        pgs: arrays.pgs,
        sex: arrays.sex,
        pcs: arrays.pcs,
        extra_static_covariates: arrays.extra_static_covariates,
        extra_static_names: arrays.extra_static_names,
    })
}

#[derive(Debug)]
struct SurvivalArrays {
    age_entry: Array1<f64>,
    age_exit: Array1<f64>,
    event_target: Array1<u8>,
    event_competing: Array1<u8>,
    sample_weight: Array1<f64>,
    pgs: Array1<f64>,
    sex: Array1<f64>,
    pcs: Array2<f64>,
    extra_static_covariates: Array2<f64>,
    extra_static_names: Vec<String>,
}

fn read_survival_arrays(path: &str, num_pcs: usize) -> Result<SurvivalArrays, SurvivalDataError> {
    let df = read_tabular(path)?;
    let name_map = build_case_insensitive_map(
        df.get_column_names()
            .into_iter()
            .map(|name| name.as_str().to_string()),
    );
    let mut used_columns = HashSet::new();

    let age_entry = extract_f64_column(&df, &name_map, "age_entry")?;
    if let Some(actual) = name_map.get("age_entry") {
        used_columns.insert(actual.clone());
    }
    let age_exit = extract_f64_column(&df, &name_map, "age_exit")?;
    if let Some(actual) = name_map.get("age_exit") {
        used_columns.insert(actual.clone());
    }
    let event_target = extract_u8_column(&df, &name_map, "event_target")?;
    if let Some(actual) = name_map.get("event_target") {
        used_columns.insert(actual.clone());
    }
    let event_competing = extract_u8_column(&df, &name_map, "event_competing")?;
    if let Some(actual) = name_map.get("event_competing") {
        used_columns.insert(actual.clone());
    }
    let pgs = extract_f64_column(&df, &name_map, "pgs")?;
    if let Some(actual) = name_map.get("pgs") {
        used_columns.insert(actual.clone());
    }
    let sex = extract_f64_column(&df, &name_map, "sex")?;
    if let Some(actual) = name_map.get("sex") {
        used_columns.insert(actual.clone());
    }

    let n = age_entry.len();
    let sample_weight = if let Some(name) = name_map.get("sample_weight") {
        let weights = extract_f64_column(&df, &name_map, "sample_weight")?;
        if weights.len() != n {
            return Err(SurvivalDataError::LengthMismatch {
                column_name: name.clone(),
                expected: n,
                found: weights.len(),
            });
        }
        used_columns.insert(name.clone());
        weights
    } else {
        Array1::from_elem(n, 1.0)
    };

    let mut pc_columns = Vec::with_capacity(num_pcs);
    for idx in 0..num_pcs {
        let key = format!("pc{}", idx + 1);
        let values = extract_f64_column(&df, &name_map, &key)?;
        if values.len() != n {
            let actual = name_map
                .get(&key.to_lowercase())
                .cloned()
                .unwrap_or(key.clone());
            return Err(SurvivalDataError::LengthMismatch {
                column_name: actual,
                expected: n,
                found: values.len(),
            });
        }
        if let Some(actual) = name_map.get(&key.to_ascii_lowercase()) {
            used_columns.insert(actual.clone());
        }
        pc_columns.push(values);
    }

    let mut pcs = Array2::<f64>::zeros((n, num_pcs));
    for (j, column) in pc_columns.into_iter().enumerate() {
        pcs.column_mut(j).assign(&column);
    }

    let mut extra_names = Vec::new();
    let mut extra_columns = Vec::new();
    for original in df.get_column_names() {
        let original_str = original.as_ref();
        if used_columns.contains(original_str) {
            continue;
        }
        let series = df
            .column(original_str)
            .map_err(|_| SurvivalDataError::ColumnNotFound(original_str.to_string()))?;
        let casted = match series.cast(&DataType::Float64) {
            Ok(values) => values,
            Err(_) => continue,
        };
        let values = casted.f64().expect("casted to f64");
        if values.null_count() > 0 {
            return Err(SurvivalDataError::MissingValues(original_str.to_string()));
        }
        if values.len() != n {
            return Err(SurvivalDataError::LengthMismatch {
                column_name: original_str.to_string(),
                expected: n,
                found: values.len(),
            });
        }
        let column = Array1::from_iter(values.into_no_null_iter());
        extra_names.push(original_str.to_string());
        extra_columns.push(column);
        used_columns.insert(original_str.to_string());
    }

    let extra_width = extra_columns.len();
    let mut extra_static = Array2::<f64>::zeros((n, extra_width));
    for (idx, column) in extra_columns.into_iter().enumerate() {
        extra_static.column_mut(idx).assign(&column);
    }

    Ok(SurvivalArrays {
        age_entry,
        age_exit,
        event_target,
        event_competing,
        sample_weight,
        pgs,
        sex,
        pcs,
        extra_static_covariates: extra_static,
        extra_static_names: extra_names,
    })
}

fn read_tabular(path: &str) -> Result<DataFrame, SurvivalDataError> {
    let path = Path::new(path);
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("parquet") | Some("pq") => {
            let file = File::open(path)?;
            ParquetReader::new(file)
                .finish()
                .map_err(SurvivalDataError::from)
        }
        _ => {
            let file = File::open(path)?;
            CsvReadOptions::default()
                .with_has_header(true)
                .map_parse_options(|options| options.with_separator(b'\t'))
                .into_reader_with_file_handle(file)
                .finish()
                .map_err(SurvivalDataError::from)
        }
    }
}

fn build_case_insensitive_map<I, S>(names: I) -> HashMap<String, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut map = HashMap::new();
    for name in names {
        let original = name.as_ref().to_string();
        map.insert(original.to_ascii_lowercase(), original);
    }
    map
}

fn extract_f64_column(
    df: &DataFrame,
    map: &HashMap<String, String>,
    key: &str,
) -> Result<Array1<f64>, SurvivalDataError> {
    let actual = map
        .get(&key.to_ascii_lowercase())
        .ok_or_else(|| SurvivalDataError::ColumnNotFound(key.to_string()))?;
    let series = df
        .column(actual)
        .map_err(|_| SurvivalDataError::ColumnNotFound(actual.clone()))?;
    let dtype = series.dtype().clone();
    let series = if dtype != DataType::Float64 {
        series
            .cast(&DataType::Float64)
            .map_err(|_| SurvivalDataError::ColumnWrongType {
                column_name: actual.clone(),
                expected_type: "float",
                found_type: dtype.to_string(),
            })?
    } else {
        series.clone()
    };
    let values = series.f64().expect("casted to f64");
    if values.null_count() > 0 {
        return Err(SurvivalDataError::MissingValues(actual.clone()));
    }
    Ok(Array1::from_iter(values.into_no_null_iter()))
}

fn extract_u8_column(
    df: &DataFrame,
    map: &HashMap<String, String>,
    key: &str,
) -> Result<Array1<u8>, SurvivalDataError> {
    let actual = map
        .get(&key.to_ascii_lowercase())
        .ok_or_else(|| SurvivalDataError::ColumnNotFound(key.to_string()))?;
    let series = df
        .column(actual)
        .map_err(|_| SurvivalDataError::ColumnNotFound(actual.clone()))?;
    let dtype = series.dtype().clone();
    let casted = series
        .cast(&DataType::Int64)
        .map_err(|_| SurvivalDataError::ColumnWrongType {
            column_name: actual.clone(),
            expected_type: "integer",
            found_type: dtype.to_string(),
        })?;
    let values = casted.i64().expect("casted to i64");
    if values.null_count() > 0 {
        return Err(SurvivalDataError::MissingValues(actual.clone()));
    }
    let mut result = Array1::<u8>::zeros(values.len());
    for (idx, value) in values.into_no_null_iter().enumerate() {
        if value < 0 || value > u8::MAX as i64 {
            return Err(SurvivalDataError::ColumnWrongType {
                column_name: actual.clone(),
                expected_type: "integer",
                found_type: dtype.to_string(),
            });
        }
        result[idx] = value as u8;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::{CsvWriter, DataFrame, ParquetWriter, SerWriter, Series};
    use tempfile::{Builder, NamedTempFile};

    fn sample_dataframe() -> DataFrame {
        DataFrame::new(vec![
            Series::new("age_entry".into(), vec![50.0, 60.0, 70.0]).into(),
            Series::new("age_exit".into(), vec![55.0, 65.0, 75.0]).into(),
            Series::new("event_target".into(), vec![1i32, 0, 0]).into(),
            Series::new("event_competing".into(), vec![0i32, 1, 0]).into(),
            Series::new("sample_weight".into(), vec![1.0, 2.0, 3.0]).into(),
            Series::new("pgs".into(), vec![0.1, 0.2, 0.3]).into(),
            Series::new("sex".into(), vec![0.0, 1.0, 0.0]).into(),
            Series::new("pc1".into(), vec![0.5, 0.6, 0.7]).into(),
            Series::new("pc2".into(), vec![1.5, 1.6, 1.7]).into(),
            Series::new("bmi".into(), vec![22.0, 23.5, 24.1]).into(),
        ])
        .expect("construct sample dataframe")
    }

    fn write_tsv(df: &DataFrame) -> NamedTempFile {
        let mut file = Builder::new().suffix(".tsv").tempfile().expect("tempfile");
        let mut writer = CsvWriter::new(file.as_file_mut()).with_separator(b'\t');
        let mut clone = df.clone();
        writer.finish(&mut clone).expect("write tsv");
        file
    }

    fn write_parquet(df: &DataFrame) -> NamedTempFile {
        let mut file = Builder::new()
            .suffix(".parquet")
            .tempfile()
            .expect("tempfile");
        let writer = ParquetWriter::new(file.as_file_mut());
        let mut clone = df.clone();
        writer.finish(&mut clone).expect("write parquet");
        file
    }

    #[test]
    fn training_loader_reads_tsv() {
        let df = sample_dataframe();
        let file = write_tsv(&df);
        let bundle = load_survival_training_data(file.path().to_str().unwrap(), 2, 0.1)
            .expect("load training data");
        assert_eq!(bundle.data.age_entry.len(), 3);
        assert_eq!(bundle.age_transform.minimum_age, 50.0);
        assert_eq!(bundle.data.sample_weight[1], 2.0);
        assert_eq!(bundle.data.extra_static_covariates.ncols(), 1);
        assert_eq!(bundle.data.extra_static_names, vec!["bmi".to_string()]);
    }

    #[test]
    fn training_loader_defaults_weights() {
        let mut df = sample_dataframe();
        df.drop_in_place("sample_weight").unwrap();
        let file = write_tsv(&df);
        let bundle = load_survival_training_data(file.path().to_str().unwrap(), 2, 0.1)
            .expect("load training data");
        assert!(
            bundle
                .data
                .sample_weight
                .iter()
                .all(|&w| (w - 1.0).abs() < 1e-12)
        );
    }

    #[test]
    fn prediction_loader_reads_parquet() {
        let df = sample_dataframe();
        let file = write_parquet(&df);
        let prediction = load_survival_prediction_data(file.path().to_str().unwrap(), 2)
            .expect("load prediction");
        let inputs = prediction.as_inputs();
        assert_eq!(inputs.covariates.pgs.len(), 3);
        assert_eq!(inputs.covariates.static_covariates.ncols(), 1);
        assert_eq!(prediction.extra_static_names, vec!["bmi".to_string()]);
        assert_eq!(inputs.age_exit.len(), 3);
    }

    #[test]
    fn loader_rejects_conflicting_events() {
        let mut df = sample_dataframe();
        df.with_column(Series::new("event_target".into(), vec![1i32, 1, 0]))
            .unwrap();
        let file = write_tsv(&df);
        let err = load_survival_training_data(file.path().to_str().unwrap(), 2, 0.1)
            .expect_err("conflicting events");
        match err {
            SurvivalDataError::Validation(SurvivalError::ConflictingEvents) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }
}

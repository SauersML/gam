//! # Data Loading and Validation Module
//!
//! This module serves as the exclusive entry point for user-provided data.
//! Its primary responsibility is to read tabular data files (TSV),
//! validate them against a strict, predefined schema, and transform them
//! into the clean `ndarray` structures required by the application's
//! statistical core.
//!
//! - Strict Schema: Column names are not configurable. The module enforces
//!   the use of `phenotype`, `score`, `PC1`, `PC2`, etc. This simplifies the
//!   user interface and eliminates a class of configuration errors.
//! - User-Centric Errors: Failures are assumed to be user-input errors.
//!   The `DataError` enum is designed to provide clear, actionable feedback.
//! - Performance: It leverages the `polars` Lazy API to minimize memory
//!   usage and I/O by only loading required columns from disk.

use ndarray::{Array1, Array2};
use polars::prelude::*;
use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

/// A container for validated data ready for model training.
#[derive(Debug)]
pub struct TrainingData {
    /// The phenotype vector (`Y`), from the 'phenotype' column.
    pub y: Array1<f64>,
    /// The Polygenic Score vector (`P`), from the 'score' column.
    pub p: Array1<f64>,
    /// Binary sex indicator (e.g., 0 for female, 1 for male).
    pub sex: Array1<f64>,
    /// The Principal Components matrix (`PC`), from 'PC1', 'PC2', ... columns.
    /// Shape: [n_samples, num_pcs].
    pub pcs: Array2<f64>,
    /// The prior weights vector, from the required 'weights' column.
    pub weights: Array1<f64>,
}

/// A container for validated data ready for prediction.
#[derive(Debug)]
pub struct PredictionData {
    /// The Polygenic Score vector (`P`), from the 'score' column.
    pub p: Array1<f64>,
    /// Binary sex indicator (e.g., 0 for female, 1 for male).
    pub sex: Array1<f64>,
    /// The Principal Components matrix (`PC`), from 'PC1', 'PC2', ... columns.
    /// Shape: [n_samples, num_pcs].
    pub pcs: Array2<f64>,
    /// Optional sample identifiers. If an input column `sample_id` exists, it is used;
    /// otherwise sequential IDs (1-based) are generated as strings.
    pub sample_ids: Vec<String>,
}

/// A comprehensive error type for all data loading and validation failures.
#[derive(Error, Debug)]
pub enum DataError {
    #[error("Error from the underlying Polars DataFrame library: {0}")]
    PolarsError(#[from] PolarsError),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error(
        "The required column '{0}' was not found in the input file. Please check spelling and case."
    )]
    ColumnNotFound(String),
    #[error(
        "The required column '{column_name}' could not be converted to the expected type '{expected_type}'. It contains non-numeric data. (Found type: {found_type})"
    )]
    ColumnWrongType {
        column_name: String,
        expected_type: &'static str,
        found_type: String,
    },
    #[error(
        "Missing or null values were found in the required column '{0}'. This tool requires complete data with no missing values."
    )]
    MissingValuesFound(String),
    #[error(
        "Input file contains only {found} data rows, but at least {required} are recommended for a stable model."
    )]
    InsufficientRows { found: usize, required: usize },

    #[error(
        "Non-finite values (NaN or Infinity) were found in the required column '{0}'. This tool requires all data to be finite."
    )]
    NonFiniteValuesFound(String),

    #[error(
        "Required 'weights' column was not found in the input file. Sample weights must be explicitly provided."
    )]
    WeightsColumnRequired,
}

/// Loads and validates data specifically for model training.
pub fn load_training_data(path: &str, num_pcs: usize) -> Result<TrainingData, DataError> {
    // Do not require an explicit 'weights' column. If missing, default to 1.0.
    let (p, sex, pcs, y_opt, weights) = internal::load_data(path, num_pcs, true, false)?;
    // This unwrap is safe because we passed `include_phenotype: true`.
    let y = y_opt.unwrap();
    Ok(TrainingData {
        y,
        p,
        sex,
        pcs,
        weights,
    })
}

/// Loads and validates data specifically for prediction.
pub fn load_prediction_data(path: &str, num_pcs: usize) -> Result<PredictionData, DataError> {
    let (p, sex, pcs, _, _, sample_ids) =
        internal::load_data_with_ids(path, num_pcs, false, false)?;
    Ok(PredictionData {
        p,
        sex,
        pcs,
        sample_ids,
    })
}

/// Internal module for shared data loading logic.
mod internal {
    use super::*;

    const MINIMUM_ROWS: usize = 20;

    /// The single, unified data loading function. It reads a file, validates it against
    /// a dynamically generated schema, and returns the core `ndarray` objects.
    pub(super) fn load_data(
        path: &str,
        num_pcs: usize,
        include_phenotype: bool,
        require_weights: bool,
    ) -> Result<
        (
            Array1<f64>,
            Array1<f64>,
            Array2<f64>,
            Option<Array1<f64>>,
            Array1<f64>,
        ),
        DataError,
    > {
        // Small helper: validate that an array contains only finite values
        fn validate_is_finite(arr: &Array1<f64>, column_name: &str) -> Result<(), DataError> {
            if arr.iter().any(|&v| !v.is_finite()) {
                return Err(DataError::NonFiniteValuesFound(column_name.to_string()));
            }
            Ok(())
        }

        // --- Generate the exact list of required column names ---
        let pc_names: Vec<String> = (1..=num_pcs).map(|i| format!("PC{i}")).collect();
        let mut required_cols: Vec<String> = Vec::with_capacity(3 + num_pcs);
        if include_phenotype {
            required_cols.push("phenotype".to_string());
        }
        required_cols.push("score".to_string());
        required_cols.push("sex".to_string());
        required_cols.extend_from_slice(&pc_names);

        // --- Read and validate the DataFrame using Polars ---
        println!("Loading data from '{path}'");

        // Use the Polars CsvReader for efficiency
        let df = CsvReader::new(File::open(Path::new(path))?)
            .with_options(
                CsvReadOptions::default()
                    .with_has_header(true)
                    .with_parse_options(
                        CsvParseOptions::default().with_separator(b'\t'), // Using tab as separator
                    ),
            )
            .finish()?;

        println!("Successfully loaded data file.");

        // Check if we have the minimum number of rows
        if df.height() < MINIMUM_ROWS {
            return Err(DataError::InsufficientRows {
                found: df.height(),
                required: MINIMUM_ROWS,
            });
        }

        // Verify all required columns exist
        let df_columns = df.get_column_names();
        let columns_set: HashSet<String> = df_columns.into_iter().map(|s| s.to_string()).collect();

        for col_name in &required_cols {
            if !columns_set.contains(col_name) {
                return Err(DataError::ColumnNotFound(col_name.clone()));
            }
        }

        // Check if required weights column exists
        let has_weights = columns_set.contains("weights");
        println!("All required columns found: {required_cols:?}");
        if has_weights {
            println!("Required 'weights' column found.");
        }

        // --- Convert columns efficiently to ndarray structures ---

        // Process phenotype column if needed
        let phenotype_opt = if include_phenotype {
            let phenotype_series = df.column("phenotype")?;
            if phenotype_series.null_count() > 0 {
                return Err(DataError::MissingValuesFound("phenotype".to_string()));
            }

            // Convert to f64
            let phenotype_casted = match phenotype_series.cast(&DataType::Float64) {
                Ok(casted) => casted,
                Err(_) => {
                    return Err(DataError::ColumnWrongType {
                        column_name: "phenotype".to_string(),
                        expected_type: "f64 (numeric)",
                        found_type: format!("{:?}", phenotype_series.dtype()),
                    });
                }
            };

            // Check for nulls AFTER casting to detect non-numeric values
            if phenotype_casted.null_count() > 0 {
                return Err(DataError::ColumnWrongType {
                    column_name: "phenotype".to_string(),
                    expected_type: "f64 (numeric)",
                    found_type: format!("{:?}", phenotype_series.dtype()),
                });
            }

            // Now convert to ndarray
            let arr = phenotype_casted.rechunk().f64()?.to_ndarray()?.to_owned();
            validate_is_finite(&arr, "phenotype")?;
            Some(arr)
        } else {
            None
        };

        // Process score column
        let score_series = df.column("score")?;
        if score_series.null_count() > 0 {
            return Err(DataError::MissingValuesFound("score".to_string()));
        }

        // Convert to f64
        let score_casted = match score_series.cast(&DataType::Float64) {
            Ok(casted) => casted,
            Err(_) => {
                return Err(DataError::ColumnWrongType {
                    column_name: "score".to_string(),
                    expected_type: "f64 (numeric)",
                    found_type: format!("{:?}", score_series.dtype()),
                });
            }
        };

        // Check for nulls AFTER casting to detect non-numeric values
        // Casting non-numeric strings to f64 produces nulls, which we need to detect here
        if score_casted.null_count() > 0 {
            return Err(DataError::ColumnWrongType {
                column_name: "score".to_string(),
                expected_type: "f64 (numeric)",
                found_type: format!("{:?}", score_series.dtype()),
            });
        }

        // Now convert to ndarray
        let pgs = score_casted.rechunk().f64()?.to_ndarray()?.to_owned();
        validate_is_finite(&pgs, "score")?;

        // Process sex column
        let sex_series = df.column("sex")?;
        if sex_series.null_count() > 0 {
            return Err(DataError::MissingValuesFound("sex".to_string()));
        }

        let sex_casted = match sex_series.cast(&DataType::Float64) {
            Ok(casted) => casted,
            Err(_) => {
                return Err(DataError::ColumnWrongType {
                    column_name: "sex".to_string(),
                    expected_type: "f64 (numeric)",
                    found_type: format!("{:?}", sex_series.dtype()),
                });
            }
        };

        if sex_casted.null_count() > 0 {
            return Err(DataError::ColumnWrongType {
                column_name: "sex".to_string(),
                expected_type: "f64 (numeric)",
                found_type: format!("{:?}", sex_series.dtype()),
            });
        }

        let sex = sex_casted.rechunk().f64()?.to_ndarray()?.to_owned();
        validate_is_finite(&sex, "sex")?;

        // Process PC columns efficiently
        let mut pc_arrays = Vec::with_capacity(num_pcs);
        for pc_name in &pc_names {
            let pc_series = df.column(pc_name)?;
            if pc_series.null_count() > 0 {
                return Err(DataError::MissingValuesFound(pc_name.clone()));
            }

            // Convert to f64
            let pc_casted = match pc_series.cast(&DataType::Float64) {
                Ok(casted) => casted,
                Err(_) => {
                    return Err(DataError::ColumnWrongType {
                        column_name: pc_name.clone(),
                        expected_type: "f64 (numeric)",
                        found_type: format!("{:?}", pc_series.dtype()),
                    });
                }
            };

            // Check for nulls AFTER casting to detect non-numeric values
            if pc_casted.null_count() > 0 {
                return Err(DataError::ColumnWrongType {
                    column_name: pc_name.clone(),
                    expected_type: "f64 (numeric)",
                    found_type: format!("{:?}", pc_series.dtype()),
                });
            }

            // Now convert to ndarray
            let arr = pc_casted.rechunk().f64()?.to_ndarray()?.to_owned();
            validate_is_finite(&arr, pc_name)?;
            pc_arrays.push(arr);
        }

        // Stack PC arrays into a matrix
        // Handle the case where num_pcs = 0 (no PC covariates)
        let (pcs_flat, n_rows, n_cols) = if pc_arrays.is_empty() {
            // No PCs requested - create empty matrix with correct number of rows
            let n_rows = pgs.len();
            (Vec::new(), n_rows, 0)
        } else {
            let n_rows = pc_arrays[0].len();
            let n_cols = pc_arrays.len();
            let mut pcs_flat = Vec::with_capacity(n_rows * n_cols);

            // Convert column vectors to row-major format
            for row_idx in 0..n_rows {
                for col_idx in 0..n_cols {
                    pcs_flat.push(pc_arrays[col_idx][row_idx]);
                }
            }
            (pcs_flat, n_rows, n_cols)
        };

        let pcs = Array2::from_shape_vec((n_rows, n_cols), pcs_flat)
            .expect("PC arrays should have consistent dimensions");

        // Process weights column
        let weights = if require_weights {
            if !has_weights {
                return Err(DataError::WeightsColumnRequired);
            }

            let weights_series = df.column("weights")?;
            if weights_series.null_count() > 0 {
                return Err(DataError::MissingValuesFound("weights".to_string()));
            }

            // Convert to f64
            let weights_casted = match weights_series.cast(&DataType::Float64) {
                Ok(casted) => casted,
                Err(_) => {
                    return Err(DataError::ColumnWrongType {
                        column_name: "weights".to_string(),
                        expected_type: "f64 (numeric)",
                        found_type: format!("{:?}", weights_series.dtype()),
                    });
                }
            };

            // Check for nulls AFTER casting to detect non-numeric values
            if weights_casted.null_count() > 0 {
                return Err(DataError::ColumnWrongType {
                    column_name: "weights".to_string(),
                    expected_type: "f64 (numeric)",
                    found_type: format!("{:?}", weights_series.dtype()),
                });
            }

            // Convert to ndarray
            let weights_array = weights_casted.rechunk().f64()?.to_ndarray()?.to_owned();
            validate_is_finite(&weights_array, "weights")?;

            // Validate that all weights are non-negative
            for (i, &weight) in weights_array.iter().enumerate() {
                if weight < 0.0 {
                    return Err(DataError::ColumnWrongType {
                        column_name: "weights".to_string(),
                        expected_type: "non-negative f64 values",
                        found_type: format!("negative value {} at row {}", weight, i + 1),
                    });
                }
            }

            weights_array
        } else {
            // If weights are not required and not present, default to 1.0 per row
            if has_weights {
                let weights_series = df.column("weights")?;
                if weights_series.null_count() > 0 {
                    return Err(DataError::MissingValuesFound("weights".to_string()));
                }

                let weights_casted = match weights_series.cast(&DataType::Float64) {
                    Ok(casted) => casted,
                    Err(_) => {
                        return Err(DataError::ColumnWrongType {
                            column_name: "weights".to_string(),
                            expected_type: "f64 (numeric)",
                            found_type: format!("{:?}", weights_series.dtype()),
                        });
                    }
                };

                if weights_casted.null_count() > 0 {
                    return Err(DataError::ColumnWrongType {
                        column_name: "weights".to_string(),
                        expected_type: "f64 (numeric)",
                        found_type: format!("{:?}", weights_series.dtype()),
                    });
                }

                let weights_array = weights_casted.rechunk().f64()?.to_ndarray()?.to_owned();
                validate_is_finite(&weights_array, "weights")?;
                weights_array
            } else {
                Array1::from_elem(pgs.len(), 1.0)
            }
        };

        println!(
            "Data validation successful: all required columns have numeric data with no missing values."
        );
        Ok((pgs, sex, pcs, phenotype_opt, weights))
    }

    /// Variant of `load_data` that also extracts or synthesizes sample IDs.
    pub(super) fn load_data_with_ids(
        path: &str,
        num_pcs: usize,
        include_phenotype: bool,
        require_weights: bool,
    ) -> Result<
        (
            Array1<f64>,
            Array1<f64>,
            Array2<f64>,
            Option<Array1<f64>>,
            Array1<f64>,
            Vec<String>,
        ),
        DataError,
    > {
        // Load using the existing path
        let (p, sex, pcs, y_opt, w) = load_data(path, num_pcs, include_phenotype, require_weights)?;

        // Reload a lightweight frame to try to get sample_id without duplicating conversions
        let df = CsvReader::new(File::open(Path::new(path))?)
            .with_options(
                CsvReadOptions::default()
                    .with_has_header(true)
                    .with_parse_options(CsvParseOptions::default().with_separator(b'\t')),
            )
            .finish()?;

        let n = p.len();
        let sample_ids: Vec<String> = if df.get_column_names().iter().any(|c| c == &"sample_id") {
            let s = df.column("sample_id")?;
            if s.null_count() > 0 {
                // If nulls exist, synthesize IDs
                (1..=n).map(|i| i.to_string()).collect()
            } else {
                // Collect via per-row AnyValue -> String for robustness across dtypes
                let len = s.len().min(n);
                let mut out = Vec::with_capacity(n);
                for i in 0..len {
                    let v = s.get(i).unwrap_or(polars::prelude::AnyValue::Null);
                    out.push(match v {
                        polars::prelude::AnyValue::Null => (i + 1).to_string(),
                        _ => v.to_string(),
                    });
                }
                // If fewer than n due to any mismatch, pad sequentially
                for i in len..n {
                    out.push((i + 1).to_string());
                }
                out
            }
        } else {
            // No sample_id column; synthesize sequential numeric IDs as strings
            (1..=n).map(|i| i.to_string()).collect()
        };

        Ok((p, sex, pcs, y_opt, w, sample_ids))
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::io::{self, Write};
    use tempfile::NamedTempFile;

    /// A robust helper to create a temporary CSV file for testing.
    fn create_test_csv(content: &str) -> io::Result<NamedTempFile> {
        let mut file = NamedTempFile::new()?;
        writeln!(file, "{}", content)?;
        file.flush()?;
        Ok(file)
    }

    #[test]
    fn test_non_finite_values_rejected_in_score() {
        let header = "phenotype\tscore\tsex\tPC1";
        let data_row = "1.0\tNaN\t0\t0.1"; // NaN should be parsed as f64::NAN
        let content = generate_csv_content(header, data_row, 30);
        let file = create_test_csv(&content).unwrap();
        let err = load_training_data(file.path().to_str().unwrap(), 1).unwrap_err();
        match err {
            DataError::NonFiniteValuesFound(col) => assert_eq!(col, "score"),
            other => panic!("Expected NonFiniteValuesFound(score), got {:?}", other),
        }
    }

    #[test]
    fn test_non_finite_values_rejected_in_pc() {
        let header = "phenotype\tscore\tsex\tPC1";
        let data_row = "1.0\t2.0\t0\tNaN"; // NaN in PC1
        let content = generate_csv_content(header, data_row, 30);
        let file = create_test_csv(&content).unwrap();
        let err = load_training_data(file.path().to_str().unwrap(), 1).unwrap_err();
        match err {
            DataError::NonFiniteValuesFound(col) => assert_eq!(col, "PC1"),
            other => panic!("Expected NonFiniteValuesFound(PC1), got {:?}", other),
        }
    }

    /// Generates CSV content with a specified number of data rows.
    fn generate_csv_content(header: &str, data_row: &str, num_rows: usize) -> String {
        let data_rows = std::iter::repeat(data_row)
            .take(num_rows)
            .collect::<Vec<_>>()
            .join("\n");
        format!("{}\n{}", header, data_rows)
    }

    const TEST_HEADER: &str = "phenotype\tscore\tsex\tPC1\tPC2\textra_col";
    const TEST_DATA_ROW: &str = "1.0\t1.5\t0\t0.1\t0.2\t1.0";

    #[test]
    fn test_load_training_data_success() {
        // Create test data that includes required columns including weights
        let header = "phenotype\tscore\tsex\tPC1\tPC2\tweights";

        // Generate varied test data with different values in each row
        let mut rows = Vec::with_capacity(31);
        rows.push(header.to_string());

        for i in 0..30 {
            let row = format!(
                "{:.2}\t{:.2}\t{}\t{:.3}\t{:.3}\t{:.1}",
                i as f64 / 10.0,
                (i as f64 + 5.0) / 10.0,
                i % 2,
                (i as f64 - 2.0) / 20.0,
                (i as f64 + 3.0) / 15.0,
                1.0 // Add weight of 1.0 for each row
            );
            rows.push(row);
        }

        let content = rows.join("\n");
        let file = create_test_csv(&content).unwrap();
        let data = load_training_data(file.path().to_str().unwrap(), 2).unwrap();

        // Test dimensions
        assert_eq!(data.y.len(), 30);
        assert_eq!(data.p.len(), 30);
        assert_eq!(data.sex.len(), 30);
        assert_eq!(data.pcs.shape(), &[30, 2]);

        // Test specific values in the first row
        assert_abs_diff_eq!(data.y[0], 0.00, epsilon = 1e-6);
        assert_abs_diff_eq!(data.p[0], 0.50, epsilon = 1e-6);
        assert_abs_diff_eq!(data.sex[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[0, 0]], -0.100, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[0, 1]], 0.200, epsilon = 1e-6);

        // Test specific values in a middle row
        assert_abs_diff_eq!(data.y[15], 1.50, epsilon = 1e-6);
        assert_abs_diff_eq!(data.p[15], 2.00, epsilon = 1e-6);
        assert_abs_diff_eq!(data.sex[15], (15 % 2) as f64, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[15, 0]], 0.650, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[15, 1]], 1.200, epsilon = 1e-6);

        // Test specific values in the last row
        assert_abs_diff_eq!(data.y[29], 2.90, epsilon = 1e-6);
        assert_abs_diff_eq!(data.p[29], 3.40, epsilon = 1e-6);
        assert_abs_diff_eq!(data.sex[29], (29 % 2) as f64, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[29, 0]], 1.350, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[29, 1]], 2.133, epsilon = 1e-6);
    }

    #[test]
    fn test_pcs_matrix_structure_with_custom_data() {
        // Create test data with different values for each row including weights
        let custom_header = "phenotype\tscore\tsex\tPC1\tPC2\tweights";

        // Generate 20 rows with a clear pattern to meet the minimum row requirement
        let mut custom_rows = Vec::with_capacity(21);
        custom_rows.push(custom_header.to_string());

        for i in 0..20 {
            let row = format!(
                "{:.1}\t{:.1}\t{}\t{:.1}\t{:.1}\t{:.1}",
                i as f64 / 10.0,
                (i as f64 + 1.0) / 10.0,
                (i % 2),
                (i as f64 + 2.0) / 10.0,
                (i as f64 + 3.0) / 10.0,
                1.0 // Add weight of 1.0 for each row
            );
            custom_rows.push(row); // Push the owned String, not a reference
        }

        let custom_content = custom_rows.join("\n");
        let file = create_test_csv(&custom_content).unwrap();
        let data = load_training_data(file.path().to_str().unwrap(), 2).unwrap();

        // Verify dimensions
        assert_eq!(data.sex.len(), 20);
        assert_eq!(data.pcs.shape(), &[20, 2]);

        // Verify the matrix contents for a few key elements
        assert_abs_diff_eq!(data.sex[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data.sex[1], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[0, 0]], 0.2, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[0, 1]], 0.3, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[10, 0]], 1.2, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[10, 1]], 1.3, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[19, 0]], 2.1, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[19, 1]], 2.2, epsilon = 1e-6);
    }

    #[test]
    fn test_load_prediction_data_success() {
        // Create test data that includes required columns including weights
        let header = "score\tsex\tPC1\tweights";
        let data_row = "1.5\t1\t0.1\t1.0"; // Added dummy weight
        let content = generate_csv_content(header, data_row, 30);
        let file = create_test_csv(&content).unwrap();
        let data = load_prediction_data(file.path().to_str().unwrap(), 1).unwrap();

        assert_eq!(data.p.len(), 30);
        assert_eq!(data.sex.len(), 30);
        assert_eq!(data.pcs.shape(), &[30, 1]);
        assert_abs_diff_eq!(data.p[0], 1.5, epsilon = 1e-6);
        assert_abs_diff_eq!(data.sex[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[0, 0]], 0.1, epsilon = 1e-6);
    }

    #[test]
    fn test_error_column_not_found() {
        let content = generate_csv_content("phenotype\tscore\tsex\tPC1", "1.0\t1.5\t0\t0.1", 30);
        let file = create_test_csv(&content).unwrap();
        let err = load_training_data(file.path().to_str().unwrap(), 2).unwrap_err();
        match err {
            DataError::ColumnNotFound(col) => assert_eq!(col, "PC2"),
            _ => panic!("Expected ColumnNotFound error"),
        }
    }

    #[test]
    fn test_error_missing_values() {
        // Create test data with missing values in the score column
        let content = generate_csv_content("phenotype\tscore\tsex\tPC1", "1.0\t\t0\t0.1", 30);
        let file = create_test_csv(&content).unwrap();

        // Call the function that should detect the error
        let result = load_training_data(file.path().to_str().unwrap(), 1);

        // Verify the function returns the correct error
        match result {
            Err(DataError::MissingValuesFound(col_name)) => {
                assert_eq!(col_name, "score", "Expected error for 'score' column");
            }
            _ => panic!(
                "Expected MissingValuesFound error for 'score' column, but got {:?}",
                result
            ),
        }
    }

    #[test]
    fn test_error_wrong_type() {
        // Create test data with a non-numeric value in the score column
        let content = generate_csv_content(
            "phenotype\tscore\tsex\tPC1",
            "1.0\tnot_a_number\t0\t0.1",
            30,
        );
        let file = create_test_csv(&content).unwrap();

        // Call the function that should detect the error
        let result = load_training_data(file.path().to_str().unwrap(), 1);

        // Verify the function returns the correct error
        match result {
            Err(DataError::ColumnWrongType {
                column_name,
                expected_type,
                found_type,
            }) => {
                assert_eq!(column_name, "score", "Expected error for 'score' column");
                assert_eq!(expected_type, "f64 (numeric)", "Expected f64 type");
                assert!(
                    found_type.contains("String") || found_type.contains("text"),
                    "Expected found_type to indicate String or text, got {}",
                    found_type
                );
            }
            _ => panic!(
                "Expected ColumnWrongType error for 'score' column, but got {:?}",
                result
            ),
        }
    }

    #[test]
    fn test_error_insufficient_rows() {
        let content = generate_csv_content(TEST_HEADER, TEST_DATA_ROW, 5);
        let file = create_test_csv(&content).unwrap();
        let err = load_training_data(file.path().to_str().unwrap(), 1).unwrap_err();
        match err {
            DataError::InsufficientRows { found, required } => {
                assert_eq!(found, 5);
                assert_eq!(required, 20);
            }
            _ => panic!("Expected InsufficientRows error"),
        }
    }
}

use crate::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use csv::{ReaderBuilder, StringRecord};
use ndarray::{Array2, Axis};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;

// ---------------------------------------------------------------------------
// Typed error
// ---------------------------------------------------------------------------

/// Typed error variants for the data-loading module.
///
/// Public entry points continue to return `Result<_, String>`; this enum is
/// materialized at leaf sites and converted at the boundary via
/// `From<DataError> for String` so error text remains byte-identical to the
/// previous ad-hoc `format!(...)` output.
#[derive(Debug, Clone)]
pub enum DataError {
    /// Schema/column shape disagrees with the file: row width mismatch,
    /// requested column missing from headers, schema-declared kind violated by
    /// a row, or an unseen categorical level encountered under
    /// `UnseenCategoryPolicy::Error`.
    SchemaMismatch { reason: String },
    /// Failed to open, decode, or read structural bytes of the source
    /// (CSV/TSV row read, parquet metadata, file extension detection, parquet
    /// arrow-cast for string columns).
    ParseError { reason: String },
    /// Internal encoding bookkeeping failed: a categorical map expected by the
    /// schema path was missing, or a level expected to be present in the
    /// per-column inference state was not found during fix-up.
    EncodingFailure { reason: String },
    /// The source has no headers, no rows, or contains an empty / missing
    /// field at a row that requires a value.
    EmptyInput { reason: String },
    /// A cell value cannot be used as a feature: non-finite float, null in a
    /// numeric parquet column, or an unsupported parquet data type for the
    /// column.
    InvalidValue { reason: String },
    /// A formula or call site references a column name that is not present in
    /// the input data. Structured so the FFI boundary can raise a typed
    /// Python exception (`gamfit.ColumnNotFoundError`) carrying the missing
    /// name, available columns, and similarity suggestions as attributes —
    /// not as a parsed-back-out substring of the human display text.
    ///
    ColumnNotFound {
        /// The missing column name, exactly as the user wrote it.
        name: String,
        /// Optional role label (`"response"`, `"entry"`, `"exit"`, etc.)
        /// supplied at the resolution site to disambiguate which slot in the
        /// formula referenced the bad name. `None` for bare term references.
        role: Option<String>,
        /// All headers present in the input table at resolution time, sorted.
        available: Vec<String>,
        /// Cheap similarity suggestions (case-insensitive substring or
        /// shared-prefix length ≥ 3), sorted; empty when no header is close.
        similar: Vec<String>,
        /// True iff the available set has exactly one entry and that entry
        /// contains a literal tab — i.e. the user almost certainly handed gam
        /// a TSV file under a `.csv` filename. Surfaced as a structured
        /// boolean rather than re-parsed from prose at the boundary.
        tsv_hint: bool,
    },
}

impl DataError {
    /// Build a typed `ColumnNotFound` from the column map of the resolved
    /// dataset. Centralises the similarity / TSV-hint heuristics that the
    /// legacy `missing_column_message` helper used to perform inline so all
    /// callers — leaf `resolve_col*` shims and the multi-column requested-
    /// columns aggregator — produce identical payloads.
    pub fn column_not_found(
        col_map: &HashMap<String, usize>,
        name: &str,
        role: Option<&str>,
    ) -> Self {
        let target_lower = name.to_lowercase();
        let mut similar: Vec<String> = col_map
            .keys()
            .filter(|k| {
                let k_lower = k.to_lowercase();
                k_lower.contains(&target_lower)
                    || target_lower.contains(&k_lower)
                    || shared_prefix(&k_lower, &target_lower) >= 3
            })
            .cloned()
            .collect();
        similar.sort_unstable();
        let mut available: Vec<String> = col_map.keys().cloned().collect();
        available.sort_unstable();
        let tsv_hint = available.len() == 1 && available[0].contains('\t');
        Self::ColumnNotFound {
            name: name.to_string(),
            role: role.map(str::to_string),
            available,
            similar,
            tsv_hint,
        }
    }
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::SchemaMismatch { reason }
            | DataError::ParseError { reason }
            | DataError::EncodingFailure { reason }
            | DataError::EmptyInput { reason }
            | DataError::InvalidValue { reason } => f.write_str(reason),
            DataError::ColumnNotFound {
                name,
                role,
                available,
                similar,
                tsv_hint,
            } => {
                let label = match role {
                    Some(r) => format!("{r} column '{name}'"),
                    None => format!("column '{name}'"),
                };
                let tsv_suffix = if *tsv_hint {
                    " — your file appears to be tab-separated; gam expects comma-separated CSV. \
         Replace tabs with commas, or pre-convert with `tr '\\t' ',' < file.tsv > file.csv`."
                } else {
                    ""
                };
                if similar.is_empty() {
                    write!(
                        f,
                        "{label} not found in data. Available columns: [{}]{tsv_suffix}",
                        available.join(", ")
                    )
                } else {
                    write!(
                        f,
                        "{label} not found in data. Did you mean one of [{}]? Full list: [{}]{tsv_suffix}",
                        similar.join(", "),
                        available.join(", ")
                    )
                }
            }
        }
    }
}

impl std::error::Error for DataError {}

impl From<DataError> for String {
    fn from(err: DataError) -> String {
        err.to_string()
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UnseenCategoryPolicy {
    Error,
    EncodeUnknownForColumns(HashSet<String>),
}

impl UnseenCategoryPolicy {
    pub fn encode_unknown_for_columns(columns: HashSet<String>) -> Self {
        if columns.is_empty() {
            Self::Error
        } else {
            Self::EncodeUnknownForColumns(columns)
        }
    }

    fn unseen_code_for(&self, column_name: &str, level_count: usize) -> Option<f64> {
        match self {
            Self::Error => None,
            Self::EncodeUnknownForColumns(columns) => {
                columns.contains(column_name).then_some(level_count as f64)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct EncodedDataset {
    pub headers: Vec<String>,
    pub values: Array2<f64>,
    pub schema: DataSchema,
    pub column_kinds: Vec<ColumnKindTag>,
}

impl EncodedDataset {
    pub fn column_map(&self) -> HashMap<String, usize> {
        self.headers
            .iter()
            .enumerate()
            .map(|(index, header)| (header.clone(), index))
            .collect()
    }

    /// Per-column finite (min, max) of the training values, parallel to
    /// `headers`. Columns with no finite values default to (0.0, 0.0) so that
    /// downstream clipping is a no-op for them. Used to populate
    /// `training_feature_ranges` so prediction can clip out-of-hull inputs
    /// to the training bounding box.
    pub fn feature_ranges(&self) -> Vec<(f64, f64)> {
        // Iterate column-by-column (contiguous in C-order Array2 along axis 0
        // only when the array is Fortran-order; here Array2 is row-major so
        // each column is strided. However, scanning one column at a time keeps
        // each column's working set hot, lets rayon parallelize across
        // columns, and avoids the previous outer-col/inner-row pattern that
        // re-streamed all rows per column with stride `p`.
        self.values
            .axis_iter(Axis(1))
            .into_par_iter()
            .map(|col| {
                let (lo, hi) =
                    col.iter()
                        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                            if v.is_finite() {
                                (lo.min(v), hi.max(v))
                            } else {
                                (lo, hi)
                            }
                        });
                if !lo.is_finite() || !hi.is_finite() {
                    (0.0, 0.0)
                } else {
                    (lo, hi)
                }
            })
            .collect()
    }
}

fn shared_prefix(a: &str, b: &str) -> usize {
    a.chars()
        .zip(b.chars())
        .take_while(|(ca, cb)| ca == cb)
        .count()
}

// ---------------------------------------------------------------------------
// Format detection
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DataFormat {
    Csv,
    Tsv,
    Parquet,
}

fn detect_format(path: &Path) -> Result<DataFormat, DataError> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match ext.as_str() {
        "csv" => Ok(DataFormat::Csv),
        "tsv" | "txt" | "tab" => Ok(DataFormat::Tsv),
        "parquet" | "pq" | "pqt" => Ok(DataFormat::Parquet),
        other => Err(DataError::ParseError {
            reason: format!(
                "unsupported data file extension '.{other}'; expected csv, tsv, txt, parquet, or pq: '{}'",
                path.display()
            ),
        }),
    }
}

// ---------------------------------------------------------------------------
// Unified public API  — format auto-detected, zero extra CLI args
// ---------------------------------------------------------------------------

pub fn load_dataset_projected(
    path: &Path,
    requested_columns: &[String],
) -> Result<EncodedDataset, DataError> {
    match detect_format(path)? {
        DataFormat::Csv => load_delimited_inferred(path, b',', requested_columns),
        DataFormat::Tsv => load_delimited_inferred(path, b'\t', requested_columns),
        DataFormat::Parquet => load_parquet_inferred(path, requested_columns),
    }
}

pub fn load_datasetwith_schema_projected(
    path: &Path,
    schema: &DataSchema,
    unseen_policy: UnseenCategoryPolicy,
    requested_columns: &[String],
) -> Result<EncodedDataset, DataError> {
    match detect_format(path)? {
        DataFormat::Csv => {
            load_delimited_with_schema(path, b',', schema, unseen_policy, requested_columns)
        }
        DataFormat::Tsv => {
            load_delimited_with_schema(path, b'\t', schema, unseen_policy, requested_columns)
        }
        DataFormat::Parquet => {
            load_parquet_with_schema(path, schema, unseen_policy, requested_columns)
        }
    }
}

// ---------------------------------------------------------------------------
// CSV convenience loader — infers the schema from the file header.
// ---------------------------------------------------------------------------

pub fn load_csvwith_inferred_schema(path: &Path) -> Result<EncodedDataset, DataError> {
    load_delimited_inferred(path, b',', &[])
}

// ---------------------------------------------------------------------------
// Delimited (CSV / TSV) — streaming, columnar, single-pass
// ---------------------------------------------------------------------------

/// Maximum number of rows used for schema inference when no schema is provided.
const SCHEMA_SAMPLE_ROWS: usize = 1024;

/// Prefix a typed Python frame stamps onto a cell that originates from a
/// genuinely-categorical source column (string / object / categorical dtype).
/// The column-major inference (`infer_and_encode_column_major`) and the
/// schema-guided predict ingest (`gam-pyffi::string_records_from_rows`) both
/// strip this prefix before recording or matching a level; its presence forces
/// the column to `Categorical` even when every label parses as a number, so a
/// string column labeled "0","1","2" is one centred factor level per label
/// rather than a numeric ramp (#1317 / #1318). A leading NUL never appears in a
/// numeric literal, so an untyped CSV/array frame (no prefix) is unaffected.
pub const CATEGORICAL_CELL_SENTINEL: char = '\u{0}';

/// Strip the leading [`CATEGORICAL_CELL_SENTINEL`] from a cell if present,
/// returning the clean text and whether the marker was found.
pub fn strip_categorical_sentinel(cell: &str) -> (&str, bool) {
    match cell.strip_prefix(CATEGORICAL_CELL_SENTINEL) {
        Some(rest) => (rest, true),
        None => (cell, false),
    }
}

fn resolve_requested_columns(
    all_headers: &[String],
    requested_columns: &[String],
) -> Result<Vec<usize>, DataError> {
    if requested_columns.is_empty() {
        return Ok((0..all_headers.len()).collect());
    }

    let requested_set: HashSet<&str> = requested_columns.iter().map(String::as_str).collect();
    let mut selected = Vec::with_capacity(requested_set.len());
    for (idx, name) in all_headers.iter().enumerate() {
        if requested_set.contains(name.as_str()) {
            selected.push(idx);
        }
    }

    if selected.len() != requested_set.len() {
        let available_map: HashMap<String, usize> = all_headers
            .iter()
            .enumerate()
            .map(|(index, header)| (header.clone(), index))
            .collect();
        let missing = requested_columns
            .iter()
            .filter(|name| !available_map.contains_key(name.as_str()))
            .map(|name| {
                DataError::column_not_found(&available_map, name, Some("requested")).to_string()
            })
            .collect::<Vec<_>>();
        return Err(DataError::SchemaMismatch {
            reason: missing.join("; "),
        });
    }

    Ok(selected)
}

fn projected_headers(all_headers: &[String], selected_indices: &[usize]) -> Vec<String> {
    selected_indices
        .iter()
        .map(|&idx| all_headers[idx].clone())
        .collect()
}

fn load_delimited_inferred(
    path: &Path,
    delimiter: u8,
    requested_columns: &[String],
) -> Result<EncodedDataset, DataError> {
    let t_open = std::time::Instant::now();
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(delimiter)
        .from_path(path)
        .map_err(|e| DataError::ParseError {
            reason: format!("failed to open '{}': {e}", path.display()),
        })?;

    let all_headers: Vec<String> = rdr
        .headers()
        .map_err(|e| DataError::ParseError {
            reason: format!("failed to read headers: {e}"),
        })?
        .iter()
        .map(|s| s.trim().to_string())
        .collect();
    if all_headers.is_empty() {
        return Err(DataError::EmptyInput {
            reason: "file has no headers".to_string(),
        });
    }
    let selected_indices = resolve_requested_columns(&all_headers, requested_columns)?;
    let headers = projected_headers(&all_headers, &selected_indices);
    let p = headers.len();
    let open_ms = t_open.elapsed().as_secs_f64() * 1000.0;
    if open_ms > 100.0 {
        log::info!(
            "[DATA-LOAD] delim_open+headers | n_headers={} | n_proj={} | {:.1}ms",
            all_headers.len(),
            p,
            open_ms
        );
    }

    // Phase 1: stream CSV structure exactly as before, but keep projected,
    // trimmed fields in row-major order. Field validation, type conversion,
    // inference, and row-to-column transposition happen after the streaming
    // read so those CPU-heavy passes can run in parallel across independent
    // columns. If a later row has malformed CSV width, defer returning that
    // error until previously streamed rows have been validated to preserve the
    // serial row-major error precedence.
    let mut raw_fields = Vec::<String>::new();
    let mut total_rows: usize = 0;
    let mut stream_error: Option<DataError> = None;

    let t_stream = std::time::Instant::now();
    let mut record = StringRecord::new();
    while rdr
        .read_record(&mut record)
        .map_err(|e| DataError::ParseError {
            reason: format!("failed reading row: {e}"),
        })?
    {
        if record.len() != all_headers.len() {
            stream_error = Some(DataError::SchemaMismatch {
                reason: format!(
                    "row width mismatch at row {}: got {} fields, expected {}",
                    total_rows + 1,
                    record.len(),
                    all_headers.len()
                ),
            });
            break;
        }
        total_rows += 1;

        for &selected_idx in &selected_indices {
            let raw = record.get(selected_idx).unwrap().trim();
            raw_fields.push(raw.to_string());
        }
    }

    let stream_ms = t_stream.elapsed().as_secs_f64() * 1000.0;
    if stream_ms > 100.0 {
        log::info!(
            "[DATA-LOAD] delim_stream | n_rows={} | n_cols={} | {:.1}ms",
            total_rows,
            p,
            stream_ms
        );
    }

    if total_rows == 0 {
        if let Some(err) = stream_error {
            return Err(err);
        }
        return Err(DataError::EmptyInput {
            reason: "file has no rows".to_string(),
        });
    }

    let t_schema = std::time::Instant::now();
    let sample_count = total_rows.min(SCHEMA_SAMPLE_ROWS);
    let inferred_columns = (0..p)
        .into_par_iter()
        .map(|j| infer_delimited_column(&raw_fields, total_rows, p, j, &headers[j], sample_count))
        .collect::<Vec<_>>();

    let first_error = inferred_columns
        .iter()
        .filter_map(|result| result.as_ref().err())
        .min_by_key(|err| (err.row, err.col));
    if let Some(err) = first_error {
        return Err(err.error.clone());
    }
    if let Some(err) = stream_error {
        return Err(err);
    }

    let inferred_columns = inferred_columns
        .into_iter()
        .map(Result::unwrap)
        .collect::<Vec<_>>();

    // Build schema from inference state.
    let mut schema_cols = Vec::<SchemaColumn>::with_capacity(p);
    let mut column_kinds = Vec::<ColumnKindTag>::with_capacity(p);
    for (j, inferred) in inferred_columns.iter().enumerate() {
        column_kinds.push(inferred.kind);
        schema_cols.push(SchemaColumn {
            name: headers[j].clone(),
            kind: inferred.kind,
            levels: if matches!(inferred.kind, ColumnKindTag::Categorical) {
                inferred.levels.clone()
            } else {
                Vec::new()
            },
        });
    }
    let schema_ms = t_schema.elapsed().as_secs_f64() * 1000.0;
    if schema_ms > 100.0 {
        let n_cat = column_kinds
            .iter()
            .filter(|k| matches!(k, ColumnKindTag::Categorical))
            .count();
        log::info!(
            "[DATA-LOAD] delim_convert+infer | n_cols={} | n_cat={} | {:.1}ms",
            p,
            n_cat,
            schema_ms
        );
    }

    let t_assemble = std::time::Instant::now();
    // Assemble into Array2 from independent column vectors in parallel.
    let mut values = Array2::<f64>::zeros((total_rows, p));
    values
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .zip(inferred_columns.par_iter())
        .for_each(|(mut out_col, inferred)| {
            for (dst, &src) in out_col.iter_mut().zip(inferred.values.iter()) {
                *dst = src;
            }
        });
    let assemble_ms = t_assemble.elapsed().as_secs_f64() * 1000.0;
    if assemble_ms > 100.0 {
        log::info!(
            "[DATA-LOAD] delim_assemble_array2 | n_rows={} | n_cols={} | {:.1}ms",
            total_rows,
            p,
            assemble_ms
        );
    }

    let schema = DataSchema {
        columns: schema_cols,
    };
    Ok(EncodedDataset {
        headers,
        values,
        schema,
        column_kinds,
    })
}

struct InferredDelimitedColumn {
    values: Vec<f64>,
    kind: ColumnKindTag,
    levels: Vec<String>,
}

#[derive(Debug)]
struct DelimitedInferenceError {
    row: usize,
    col: usize,
    error: DataError,
}

fn infer_delimited_column(
    raw_fields: &[String],
    total_rows: usize,
    n_cols: usize,
    col: usize,
    header: &str,
    sample_count: usize,
) -> Result<InferredDelimitedColumn, DelimitedInferenceError> {
    // Per-column inference state (mirrors infer_schema_column logic).
    let mut values = Vec::<f64>::with_capacity(total_rows);
    let mut all_numeric = true;
    let mut all_binary = true;
    let mut level_index = HashMap::<String, usize>::new();
    let mut levels = Vec::<String>::new();

    // Shared constructor for the "non-finite parsed value" rejection, which is
    // raised identically from the sample-window, post-window, and final recode
    // passes below. `col`/`header` are in scope for the whole function.
    let non_finite_err = |row_idx: usize| DelimitedInferenceError {
        row: row_idx + 1,
        col,
        error: DataError::InvalidValue {
            reason: format!(
                "non-finite value at row {}, column '{}'",
                row_idx + 1,
                header
            ),
        },
    };

    for row_idx in 0..total_rows {
        let raw = raw_fields[row_idx * n_cols + col].as_str();
        if raw.is_empty() {
            return Err(DelimitedInferenceError {
                row: row_idx + 1,
                col,
                error: DataError::EmptyInput {
                    reason: format!("empty field at row {}, column '{}'", row_idx + 1, header),
                },
            });
        }

        // Schema inference on sample window.
        if row_idx < sample_count {
            if let Ok(v) = raw.parse::<f64>() {
                if !v.is_finite() {
                    return Err(non_finite_err(row_idx));
                }
                if (v - 0.0).abs() >= 1e-12 && (v - 1.0).abs() >= 1e-12 {
                    all_binary = false;
                }
                values.push(v);
            } else {
                all_numeric = false;
                all_binary = false;
                level_index.entry(raw.to_string()).or_insert_with(|| {
                    let idx = levels.len();
                    levels.push(raw.to_string());
                    idx
                });
                // Store a placeholder for sample-window strings; once the
                // final column kind is known, categorical columns are fixed up
                // with the same level codes as the previous serial path.
                values.push(f64::NAN);
            }
        } else if let Ok(v) = raw.parse::<f64>() {
            // After sample window: we still accumulate inference state for
            // correctness (a column that looks binary in the first 1024 rows
            // may contain 2.5 on row 1025).
            if !v.is_finite() {
                return Err(non_finite_err(row_idx));
            }
            if (v - 0.0).abs() >= 1e-12 && (v - 1.0).abs() >= 1e-12 {
                all_binary = false;
            }
            values.push(v);
        } else {
            all_numeric = false;
            all_binary = false;
            let idx = *level_index.entry(raw.to_string()).or_insert_with(|| {
                let new_idx = levels.len();
                levels.push(raw.to_string());
                new_idx
            });
            values.push(idx as f64);
        }
    }

    let kind = if all_numeric {
        if all_binary {
            ColumnKindTag::Binary
        } else {
            ColumnKindTag::Continuous
        }
    } else {
        ColumnKindTag::Categorical
    };

    if matches!(kind, ColumnKindTag::Categorical) {
        // A column is categorical only if at least one row failed numeric
        // parsing. Two failure modes used to silently corrupt the encoded
        // values for such columns:
        //   1. Sample-window rows that parsed as numbers stored the raw f64
        //      (e.g. 0.0) in `values` without adding the raw string to
        //      `level_index`.
        //   2. Post-window rows that parsed as numbers stored the raw f64
        //      directly without consulting `level_index`.
        // After the column is declared categorical, those rows must be
        // recoded as level indices using their original raw strings, treating
        // every distinct raw string as a categorical level (including the
        // numeric ones). Without this pass, a column like
        // "0, 0, ..., 0, foo" mixes raw doubles with level codes, breaking
        // the categorical encoding invariant.
        for row_idx in 0..total_rows {
            let raw = raw_fields[row_idx * n_cols + col].as_str();
            let idx = *level_index.entry(raw.to_string()).or_insert_with(|| {
                let new_idx = levels.len();
                levels.push(raw.to_string());
                new_idx
            });
            values[row_idx] = idx as f64;
        }
    }

    for (row_idx, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(non_finite_err(row_idx));
        }
    }

    Ok(InferredDelimitedColumn {
        values,
        kind,
        levels,
    })
}

fn load_delimited_with_schema(
    path: &Path,
    delimiter: u8,
    schema: &DataSchema,
    unseen_policy: UnseenCategoryPolicy,
    requested_columns: &[String],
) -> Result<EncodedDataset, DataError> {
    let t_open = std::time::Instant::now();
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(delimiter)
        .from_path(path)
        .map_err(|e| DataError::ParseError {
            reason: format!("failed to open '{}': {e}", path.display()),
        })?;

    let all_headers: Vec<String> = rdr
        .headers()
        .map_err(|e| DataError::ParseError {
            reason: format!("failed to read headers: {e}"),
        })?
        .iter()
        .map(|s| s.trim().to_string())
        .collect();
    if all_headers.is_empty() {
        return Err(DataError::EmptyInput {
            reason: "file has no headers".to_string(),
        });
    }
    let selected_indices = resolve_requested_columns(&all_headers, requested_columns)?;
    let headers = projected_headers(&all_headers, &selected_indices);
    let p = headers.len();
    let open_ms = t_open.elapsed().as_secs_f64() * 1000.0;
    if open_ms > 100.0 {
        log::info!(
            "[DATA-LOAD] delim_schema_open+headers | n_headers={} | n_proj={} | {:.1}ms",
            all_headers.len(),
            p,
            open_ms
        );
    }

    // Build per-column metadata from schema.
    let schema_byname: HashMap<&str, &SchemaColumn> = schema
        .columns
        .iter()
        .map(|c| (c.name.as_str(), c))
        .collect();

    let mut col_meta = Vec::<ColMeta>::with_capacity(p);
    for name in &headers {
        if let Some(sc) = schema_byname.get(name.as_str()) {
            let level_map = if matches!(sc.kind, ColumnKindTag::Categorical) {
                Some(
                    sc.levels
                        .iter()
                        .enumerate()
                        .map(|(idx, v)| (v.clone(), idx as f64))
                        .collect::<HashMap<_, _>>(),
                )
            } else {
                None
            };
            col_meta.push(ColMeta {
                kind: sc.kind,
                level_map,
                schema_col: (*sc).clone(),
            });
        } else {
            // Column not in schema — will be inferred below (fallback).
            col_meta.push(ColMeta {
                kind: ColumnKindTag::Continuous, // tentative
                level_map: None,
                schema_col: SchemaColumn {
                    name: name.clone(),
                    kind: ColumnKindTag::Continuous,
                    levels: Vec::new(),
                },
            });
        }
    }

    // Track which columns need inference (not in provided schema).
    let needs_inference: Vec<bool> = headers
        .iter()
        .map(|h| !schema_byname.contains_key(h.as_str()))
        .collect();

    // Stream rows into column vecs.
    let mut col_vecs: Vec<Vec<f64>> = vec![Vec::new(); p];
    // For columns needing inference, track strings for categorical fixup.
    let mut infer_all_numeric: Vec<bool> = vec![true; p];
    let mut infer_all_binary: Vec<bool> = vec![true; p];
    let mut infer_level_index: Vec<HashMap<String, usize>> = vec![HashMap::new(); p];
    let mut infer_levels: Vec<Vec<String>> = vec![Vec::new(); p];
    let mut infer_strings: Vec<Vec<(usize, String)>> = vec![Vec::new(); p]; // (row_idx, raw)

    let mut total_rows: usize = 0;
    let t_stream = std::time::Instant::now();
    let mut record = StringRecord::new();
    while rdr
        .read_record(&mut record)
        .map_err(|e| DataError::ParseError {
            reason: format!("failed reading row: {e}"),
        })?
    {
        if record.len() != all_headers.len() {
            return Err(DataError::SchemaMismatch {
                reason: format!(
                    "row width mismatch at row {}: got {} fields, expected {}",
                    total_rows + 1,
                    record.len(),
                    all_headers.len()
                ),
            });
        }
        total_rows += 1;

        for j in 0..p {
            let raw = record.get(selected_indices[j]).unwrap().trim();
            if raw.is_empty() {
                return Err(DataError::EmptyInput {
                    reason: format!(
                        "empty field at row {}, column '{}'",
                        total_rows, &headers[j]
                    ),
                });
            }

            if needs_inference[j] {
                // Accumulate inference state.
                if let Ok(v) = raw.parse::<f64>() {
                    if !v.is_finite() {
                        return Err(DataError::InvalidValue {
                            reason: format!(
                                "non-finite value at row {}, column '{}'",
                                total_rows, &headers[j]
                            ),
                        });
                    }
                    if (v - 0.0).abs() >= 1e-12 && (v - 1.0).abs() >= 1e-12 {
                        infer_all_binary[j] = false;
                    }
                    col_vecs[j].push(v);
                    // Also remember the raw string in case this column ends up
                    // categorical (because a *later* row fails numeric parsing).
                    // Without this, numeric-parsing rows would keep their raw
                    // f64 values mixed with level codes — silently corrupting
                    // the encoding for columns like `0, 0, ..., 0, foo`. This
                    // mirrors the fix-up already performed in the schema-less
                    // `infer_delimited_column` path. If the column ends up
                    // continuous/binary, this Vec is simply dropped.
                    infer_strings[j].push((total_rows - 1, raw.to_string()));
                } else {
                    infer_all_numeric[j] = false;
                    infer_all_binary[j] = false;
                    let levels_ref = &mut infer_levels[j];
                    infer_level_index[j]
                        .entry(raw.to_string())
                        .or_insert_with(|| {
                            let idx = levels_ref.len();
                            levels_ref.push(raw.to_string());
                            idx
                        });
                    infer_strings[j].push((total_rows - 1, raw.to_string()));
                    col_vecs[j].push(f64::NAN); // placeholder
                }
            } else {
                // Schema-driven parse.
                let val = parse_cell_with_schema(
                    raw,
                    &col_meta[j],
                    total_rows,
                    &headers[j],
                    &unseen_policy,
                )?;
                col_vecs[j].push(val);
            }
        }
    }

    let stream_ms = t_stream.elapsed().as_secs_f64() * 1000.0;
    if stream_ms > 100.0 {
        let n_inf = needs_inference.iter().filter(|x| **x).count();
        log::info!(
            "[DATA-LOAD] delim_schema_stream | n_rows={} | n_cols={} | n_inf={} | {:.1}ms",
            total_rows,
            p,
            n_inf,
            stream_ms
        );
    }

    if total_rows == 0 {
        return Err(DataError::EmptyInput {
            reason: "file has no rows".to_string(),
        });
    }

    let t_finalize = std::time::Instant::now();
    // Finalize inferred columns.
    let mut column_kinds = Vec::<ColumnKindTag>::with_capacity(p);
    for j in 0..p {
        if needs_inference[j] {
            let kind = if infer_all_numeric[j] {
                if infer_all_binary[j] {
                    ColumnKindTag::Binary
                } else {
                    ColumnKindTag::Continuous
                }
            } else {
                ColumnKindTag::Categorical
            };
            col_meta[j].kind = kind;
            col_meta[j].schema_col.kind = kind;
            if matches!(kind, ColumnKindTag::Categorical) {
                // Re-encode the entire column as categorical level codes.
                // `infer_strings[j]` contains every (row_idx, raw) seen during
                // streaming (both numeric- and non-numeric-parsing rows), so
                // numeric-looking strings like "0" become their own levels
                // instead of leaking through as raw f64 values that would
                // collide with real level codes.
                for (row_idx, raw) in &infer_strings[j] {
                    let levels_ref = &mut infer_levels[j];
                    let code = *infer_level_index[j].entry(raw.clone()).or_insert_with(|| {
                        let new_idx = levels_ref.len();
                        levels_ref.push(raw.clone());
                        new_idx
                    });
                    col_vecs[j][*row_idx] = code as f64;
                }
                col_meta[j].schema_col.levels = infer_levels[j].clone();
            }
        }
        column_kinds.push(col_meta[j].kind);
    }
    let finalize_ms = t_finalize.elapsed().as_secs_f64() * 1000.0;
    if finalize_ms > 100.0 {
        log::info!(
            "[DATA-LOAD] delim_schema_finalize | n_cols={} | {:.1}ms",
            p,
            finalize_ms
        );
    }

    let t_assemble = std::time::Instant::now();
    // Assemble Array2 by column in parallel (mirrors the inferred path).
    // Each column carries its own finiteness check; errors are surfaced
    // through a parallel reduce so the first detected non-finite cell wins
    // by lexicographic (column, row) order — deterministic given the
    // collect.
    let mut values = Array2::<f64>::zeros((total_rows, p));
    let assemble_err: Option<DataError> = values
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .zip(col_vecs.par_iter())
        .zip(headers.par_iter())
        .map(|((mut out_col, col_vec), header)| {
            for (i, &v) in col_vec.iter().enumerate() {
                if !v.is_finite() {
                    return Some(DataError::InvalidValue {
                        reason: format!("non-finite value at row {}, column '{}'", i + 1, header),
                    });
                }
                out_col[i] = v;
            }
            None
        })
        .reduce(|| None, |a, b| a.or(b));
    if let Some(e) = assemble_err {
        return Err(e);
    }
    let assemble_ms = t_assemble.elapsed().as_secs_f64() * 1000.0;
    if assemble_ms > 100.0 {
        log::info!(
            "[DATA-LOAD] delim_schema_assemble | n_rows={} | n_cols={} | {:.1}ms",
            total_rows,
            p,
            assemble_ms
        );
    }

    let schema_out = DataSchema {
        columns: col_meta.into_iter().map(|m| m.schema_col).collect(),
    };
    Ok(EncodedDataset {
        headers,
        values,
        schema: schema_out,
        column_kinds,
    })
}

fn parse_cell_with_schema(
    raw: &str,
    meta: &ColMeta,
    row: usize,
    col_name: &str,
    unseen_policy: &UnseenCategoryPolicy,
) -> Result<f64, DataError> {
    let val = match meta.kind {
        ColumnKindTag::Continuous => raw.parse::<f64>().map_err(|err| {
            DataError::SchemaMismatch {
                reason: format!(
                    "column '{}' is continuous in schema but row {} has non-numeric value '{}': {}",
                    col_name, row, raw, err
                ),
            }
        })?,
        ColumnKindTag::Binary => {
            let v = raw
                .parse::<f64>()
                .map_err(|err| DataError::SchemaMismatch {
                    reason: format!(
                        "column '{}' is binary in schema but row {} has non-numeric value '{}': {}",
                        col_name, row, raw, err
                    ),
                })?;
            if (v - 0.0).abs() >= 1e-12 && (v - 1.0).abs() >= 1e-12 {
                return Err(DataError::SchemaMismatch {
                    reason: format!(
                        "column '{}' is binary in schema but row {} has value {}; expected 0 or 1",
                        col_name, row, v
                    ),
                });
            }
            v
        }
        ColumnKindTag::Categorical => {
            let map = meta
                .level_map
                .as_ref()
                .ok_or_else(|| DataError::EncodingFailure {
                    reason: "internal categorical schema map missing".to_string(),
                })?;
            match map.get(raw) {
                Some(v) => *v,
                None => unseen_policy
                    .unseen_code_for(col_name, meta.schema_col.levels.len())
                    .ok_or_else(|| DataError::SchemaMismatch {
                        reason: format!(
                            "unseen level '{}' in categorical column '{}' at row {}",
                            raw, col_name, row
                        ),
                    })?,
            }
        }
    };
    if !val.is_finite() {
        return Err(DataError::InvalidValue {
            reason: format!("non-finite value at row {}, column '{}'", row, col_name),
        });
    }
    Ok(val)
}

// Inner type used by load_delimited_with_schema; defined here to keep
// parse_cell_with_schema usable without forward-declaring inside the fn.
struct ColMeta {
    kind: ColumnKindTag,
    level_map: Option<HashMap<String, f64>>,
    schema_col: SchemaColumn,
}

// ---------------------------------------------------------------------------
// Parquet — columnar, zero StringRecord, schema from metadata
// ---------------------------------------------------------------------------

enum ParquetBatchColumn {
    Numeric(Vec<f64>),
    Strings(Vec<String>),
}

/// True iff an Arrow column should be treated as a string/categorical column.
///
/// Dictionary encoding is a *storage* detail, not a semantic type: pyarrow
/// dictionary-encodes low-cardinality columns by default, including numeric
/// ones (integer factor levels, small enums stored as ints). A
/// `Dictionary(K, V)` column is categorical iff its *value* type `V` is a
/// string type; `Dictionary(_, Int*/UInt*/Float*/Bool)` is numeric. We recurse
/// through the value type so nested dictionaries resolve to their leaf type.
fn parquet_field_is_string(dt: &arrow::datatypes::DataType) -> bool {
    use arrow::datatypes::DataType;
    match dt {
        DataType::Utf8 | DataType::LargeUtf8 => true,
        DataType::Dictionary(_, value_type) => parquet_field_is_string(value_type),
        _ => false,
    }
}

fn decode_parquet_batch_column(
    col: &dyn arrow::array::Array,
    n_rows: usize,
    base_row: usize,
    header: &str,
    is_string_col: bool,
) -> Result<ParquetBatchColumn, DataError> {
    use arrow::array::{
        Array as ArrowArray, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array,
        Int32Array, Int64Array, LargeStringArray, StringArray, UInt8Array, UInt16Array,
        UInt32Array, UInt64Array,
    };
    use arrow::datatypes::DataType;

    if col.null_count() > 0 {
        for i in 0..n_rows {
            if col.is_null(i) {
                return Err(DataError::InvalidValue {
                    reason: format!(
                        "null value at row {}, column '{}'",
                        base_row + i + 1,
                        header
                    ),
                });
            }
        }
    }

    if is_string_col {
        if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
            return Ok(ParquetBatchColumn::Strings(
                (0..n_rows).map(|i| arr.value(i).to_string()).collect(),
            ));
        }
        if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
            return Ok(ParquetBatchColumn::Strings(
                (0..n_rows).map(|i| arr.value(i).to_string()).collect(),
            ));
        }

        // Dictionary-encoded strings are not directly a StringArray. Cast only
        // those remaining string-like arrays rather than falling back for every
        // Utf8/LargeUtf8 column.
        let casted =
            arrow::compute::cast(col, &DataType::Utf8).map_err(|e| DataError::ParseError {
                reason: format!("failed to cast column '{}' to string: {e}", header),
            })?;
        let arr = casted
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| DataError::EncodingFailure {
                reason: format!("column '{}' could not be read as string after cast", header),
            })?;
        return Ok(ParquetBatchColumn::Strings(
            (0..n_rows).map(|i| arr.value(i).to_string()).collect(),
        ));
    }

    // Numeric-valued dictionary columns (pyarrow dictionary-encodes
    // low-cardinality numeric columns by default) are not directly a
    // primitive array. Decode them to their concrete value type so the normal
    // numeric arms below apply. `parquet_field_is_string` has already routed
    // string-valued dictionaries through the categorical branch above, so any
    // dictionary reaching here has a numeric value type.
    let decoded_col;
    let col: &dyn arrow::array::Array = if let DataType::Dictionary(_, value_type) = col.data_type()
    {
        decoded_col = arrow::compute::cast(col, value_type).map_err(|e| DataError::ParseError {
            reason: format!(
                "failed to decode dictionary-encoded numeric column '{}': {e}",
                header
            ),
        })?;
        decoded_col.as_ref()
    } else {
        col
    };

    let mut values = Vec::with_capacity(n_rows);
    match col.data_type() {
        DataType::Float64 => {
            let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
            values.extend(arr.values().iter().copied());
        }
        DataType::Float32 => {
            let arr = col.as_any().downcast_ref::<Float32Array>().unwrap();
            values.extend(arr.values().iter().map(|&v| v as f64));
        }
        DataType::Int64 => {
            let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
            values.extend(arr.values().iter().map(|&v| v as f64));
        }
        DataType::Int32 => {
            let arr = col.as_any().downcast_ref::<Int32Array>().unwrap();
            values.extend(arr.values().iter().map(|&v| v as f64));
        }
        DataType::Int16 => {
            let arr = col.as_any().downcast_ref::<Int16Array>().unwrap();
            values.extend(arr.values().iter().map(|&v| v as f64));
        }
        DataType::Int8 => {
            let arr = col.as_any().downcast_ref::<Int8Array>().unwrap();
            values.extend(arr.values().iter().map(|&v| v as f64));
        }
        DataType::UInt64 => {
            let arr = col.as_any().downcast_ref::<UInt64Array>().unwrap();
            values.extend(arr.values().iter().map(|&v| v as f64));
        }
        DataType::UInt32 => {
            let arr = col.as_any().downcast_ref::<UInt32Array>().unwrap();
            values.extend(arr.values().iter().map(|&v| v as f64));
        }
        DataType::UInt16 => {
            let arr = col.as_any().downcast_ref::<UInt16Array>().unwrap();
            values.extend(arr.values().iter().map(|&v| v as f64));
        }
        DataType::UInt8 => {
            let arr = col.as_any().downcast_ref::<UInt8Array>().unwrap();
            values.extend(arr.values().iter().map(|&v| v as f64));
        }
        DataType::Boolean => {
            let arr = col.as_any().downcast_ref::<BooleanArray>().unwrap();
            values.extend((0..n_rows).map(|i| if arr.value(i) { 1.0 } else { 0.0 }));
        }
        other => {
            return Err(DataError::InvalidValue {
                reason: format!(
                    "unsupported parquet column type {:?} for column '{}'",
                    other, header
                ),
            });
        }
    }

    if let Some(i) = values.iter().position(|v| !v.is_finite()) {
        return Err(DataError::InvalidValue {
            reason: format!(
                "non-finite value at row {}, column '{}'",
                base_row + i + 1,
                header
            ),
        });
    }

    Ok(ParquetBatchColumn::Numeric(values))
}

fn load_parquet_inferred(
    path: &Path,
    requested_columns: &[String],
) -> Result<EncodedDataset, DataError> {
    use parquet::arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder};
    use rayon::prelude::*;
    use std::fs::File;

    let t_open = std::time::Instant::now();
    let file = File::open(path).map_err(|e| DataError::ParseError {
        reason: format!("failed to open parquet '{}': {e}", path.display()),
    })?;
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| DataError::ParseError {
            reason: format!("failed to read parquet metadata '{}': {e}", path.display()),
        })?;

    let full_schema = builder.schema().clone();
    let all_headers: Vec<String> = full_schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    if all_headers.is_empty() {
        return Err(DataError::EmptyInput {
            reason: "parquet file has no columns".to_string(),
        });
    }
    let selected_indices = resolve_requested_columns(&all_headers, requested_columns)?;
    let headers = projected_headers(&all_headers, &selected_indices);
    let selected_fields = selected_indices
        .iter()
        .map(|&idx| full_schema.fields()[idx].clone())
        .collect::<Vec<_>>();
    let projection =
        ProjectionMask::roots(builder.parquet_schema(), selected_indices.iter().copied());
    let reader =
        builder
            .with_projection(projection)
            .build()
            .map_err(|e| DataError::ParseError {
                reason: format!("failed to build parquet reader: {e}"),
            })?;
    let p = headers.len();
    let open_ms = t_open.elapsed().as_secs_f64() * 1000.0;
    if open_ms > 100.0 {
        log::info!(
            "[DATA-LOAD] parquet_open+meta | n_headers={} | n_proj={} | {:.1}ms",
            all_headers.len(),
            p,
            open_ms
        );
    }

    let t_batches = std::time::Instant::now();
    // Collect all batches.
    let mut col_vecs: Vec<Vec<f64>> = vec![Vec::new(); p];
    // For string columns: accumulate raw strings to build level maps.
    let mut string_cols: Vec<Option<Vec<String>>> = (0..p).map(|_| None).collect();
    let mut is_string_col: Vec<bool> = vec![false; p];

    for (j, field) in selected_fields.iter().enumerate() {
        // A dictionary-encoded column is categorical only when its *value* type
        // is a string type; a numeric-valued dictionary (e.g. pyarrow's default
        // encoding of low-cardinality integer columns) must stay numeric.
        if parquet_field_is_string(field.data_type()) {
            is_string_col[j] = true;
            string_cols[j] = Some(Vec::new());
        }
    }

    let mut rows_seen = 0usize;
    for batch_result in reader {
        let batch = batch_result.map_err(|e| DataError::ParseError {
            reason: format!("failed to read parquet record batch: {e}"),
        })?;
        let n_rows = batch.num_rows();

        let decoded_columns: Vec<Result<ParquetBatchColumn, DataError>> = (0..p)
            .into_par_iter()
            .map(|j| {
                decode_parquet_batch_column(
                    batch.column(j).as_ref(),
                    n_rows,
                    rows_seen,
                    &headers[j],
                    is_string_col[j],
                )
            })
            .collect();

        for (j, decoded) in decoded_columns.into_iter().enumerate() {
            match decoded? {
                ParquetBatchColumn::Strings(mut strings) => {
                    assert!(is_string_col[j]);
                    string_cols[j].as_mut().unwrap().append(&mut strings);
                    let new_len = col_vecs[j].len() + n_rows;
                    col_vecs[j].resize(new_len, f64::NAN);
                }
                ParquetBatchColumn::Numeric(mut values) => {
                    assert!(!is_string_col[j]);
                    col_vecs[j].append(&mut values);
                }
            }
        }
        rows_seen += n_rows;
    }

    let total_rows = col_vecs[0].len();
    let batches_ms = t_batches.elapsed().as_secs_f64() * 1000.0;
    if batches_ms > 100.0 {
        log::info!(
            "[DATA-LOAD] parquet_batches_decode | n_rows={} | n_cols={} | {:.1}ms",
            total_rows,
            p,
            batches_ms
        );
    }
    if total_rows == 0 {
        return Err(DataError::EmptyInput {
            reason: "parquet file has no rows".to_string(),
        });
    }

    let t_schema = std::time::Instant::now();
    // Build schema: infer kind from data.
    let mut schema_cols = Vec::<SchemaColumn>::with_capacity(p);
    let mut column_kinds = Vec::<ColumnKindTag>::with_capacity(p);

    let finalized_columns: Vec<(Vec<f64>, ColumnKindTag, SchemaColumn)> = col_vecs
        .into_par_iter()
        .zip(string_cols.into_par_iter())
        .zip(is_string_col.into_par_iter())
        .zip(headers.par_iter())
        .map(|(((mut col_values, strings), is_string), header)| {
            if is_string {
                // Categorical. Preserve level order by scanning each column in
                // row order; columns are independent and can be finalized in
                // parallel without changing schema order after collection.
                let strings = strings.expect("string column storage missing");
                let mut level_index: HashMap<String, usize> = HashMap::new();
                let mut levels_vec: Vec<String> = Vec::new();
                for s in &strings {
                    level_index.entry(s.clone()).or_insert_with(|| {
                        let idx = levels_vec.len();
                        levels_vec.push(s.clone());
                        idx
                    });
                }
                for (i, s) in strings.iter().enumerate() {
                    col_values[i] = *level_index.get(s.as_str()).unwrap() as f64;
                }
                (
                    col_values,
                    ColumnKindTag::Categorical,
                    SchemaColumn {
                        name: header.clone(),
                        kind: ColumnKindTag::Categorical,
                        levels: levels_vec,
                    },
                )
            } else {
                // Numeric: check if binary.
                let all_binary = col_values
                    .iter()
                    .all(|&v| (v - 0.0).abs() < 1e-12 || (v - 1.0).abs() < 1e-12);
                let kind = if all_binary {
                    ColumnKindTag::Binary
                } else {
                    ColumnKindTag::Continuous
                };
                (
                    col_values,
                    kind,
                    SchemaColumn {
                        name: header.clone(),
                        kind,
                        levels: Vec::new(),
                    },
                )
            }
        })
        .collect();

    let mut col_vecs = Vec::with_capacity(p);
    for (col_values, kind, schema_col) in finalized_columns {
        col_vecs.push(col_values);
        column_kinds.push(kind);
        schema_cols.push(schema_col);
    }
    let schema_ms = t_schema.elapsed().as_secs_f64() * 1000.0;
    if schema_ms > 100.0 {
        let n_cat = column_kinds
            .iter()
            .filter(|k| matches!(k, ColumnKindTag::Categorical))
            .count();
        log::info!(
            "[DATA-LOAD] parquet_finalize_schema | n_cols={} | n_cat={} | {:.1}ms",
            p,
            n_cat,
            schema_ms
        );
    }

    let t_assemble = std::time::Instant::now();
    // Assemble Array2. Columns are independent; write by column in parallel
    // so each task touches a contiguous source vec (and the strided
    // destination column once) rather than scattering across all p columns
    // per row.
    let mut values = Array2::<f64>::zeros((total_rows, p));
    values
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .zip(col_vecs.par_iter())
        .for_each(|(mut out_col, src)| {
            for (dst, &v) in out_col.iter_mut().zip(src.iter()) {
                *dst = v;
            }
        });
    let assemble_ms = t_assemble.elapsed().as_secs_f64() * 1000.0;
    if assemble_ms > 100.0 {
        log::info!(
            "[DATA-LOAD] parquet_assemble_array2 | n_rows={} | n_cols={} | {:.1}ms",
            total_rows,
            p,
            assemble_ms
        );
    }

    Ok(EncodedDataset {
        headers,
        values,
        schema: DataSchema {
            columns: schema_cols,
        },
        column_kinds,
    })
}

fn load_parquet_with_schema(
    path: &Path,
    schema: &DataSchema,
    unseen_policy: UnseenCategoryPolicy,
    requested_columns: &[String],
) -> Result<EncodedDataset, DataError> {
    // Load with inference first, then validate/re-encode against provided schema.
    let inferred = load_parquet_inferred(path, requested_columns)?;
    let p = inferred.headers.len();
    let n = inferred.values.nrows();

    let schema_byname: HashMap<&str, &SchemaColumn> = schema
        .columns
        .iter()
        .map(|c| (c.name.as_str(), c))
        .collect();

    let mut column_kinds = Vec::<ColumnKindTag>::with_capacity(p);
    let mut schema_cols = Vec::<SchemaColumn>::with_capacity(p);
    let mut values = inferred.values;

    for j in 0..p {
        let name = &inferred.headers[j];
        if let Some(sc) = schema_byname.get(name.as_str()) {
            column_kinds.push(sc.kind);
            schema_cols.push((*sc).clone());

            match sc.kind {
                ColumnKindTag::Continuous => {
                    if matches!(inferred.column_kinds[j], ColumnKindTag::Categorical) {
                        return Err(DataError::SchemaMismatch {
                            reason: format!(
                                "column '{}' is continuous in schema but parquet column is string/categorical",
                                name
                            ),
                        });
                    }
                }
                ColumnKindTag::Binary => {
                    if matches!(inferred.column_kinds[j], ColumnKindTag::Categorical) {
                        return Err(DataError::SchemaMismatch {
                            reason: format!(
                                "column '{}' is binary in schema but parquet column is string/categorical",
                                name
                            ),
                        });
                    }
                    if let Some(row) = values.column(j).iter().position(|value| {
                        (*value - 0.0).abs() >= 1e-12 && (*value - 1.0).abs() >= 1e-12
                    }) {
                        return Err(DataError::SchemaMismatch {
                            reason: format!(
                                "column '{}' is binary in schema but row {} has value {}; expected 0 or 1",
                                name,
                                row + 1,
                                values[[row, j]]
                            ),
                        });
                    }
                }
                ColumnKindTag::Categorical => {
                    if !matches!(inferred.column_kinds[j], ColumnKindTag::Categorical) {
                        return Err(DataError::SchemaMismatch {
                            reason: format!(
                                "column '{}' is categorical in schema but parquet column is numeric",
                                name
                            ),
                        });
                    }
                    let inferred_col = &inferred.schema.columns[j];
                    // Build mapping: inferred_level_name -> schema_level_index.
                    let schema_level_map: HashMap<&str, f64> = sc
                        .levels
                        .iter()
                        .enumerate()
                        .map(|(idx, v)| (v.as_str(), idx as f64))
                        .collect();
                    let inferred_to_schema: Vec<f64> = inferred_col
                        .levels
                        .iter()
                        .map(|lv| {
                            schema_level_map
                                .get(lv.as_str())
                                .copied()
                                .or_else(|| unseen_policy.unseen_code_for(name, sc.levels.len()))
                                .ok_or_else(|| DataError::SchemaMismatch {
                                    reason: format!(
                                        "unseen level '{}' in categorical column '{}'",
                                        lv, name
                                    ),
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    for i in 0..n {
                        let old_code = values[[i, j]] as usize;
                        if old_code >= inferred_to_schema.len() {
                            let Some(unseen_code) =
                                unseen_policy.unseen_code_for(name, sc.levels.len())
                            else {
                                return Err(DataError::SchemaMismatch {
                                    reason: format!(
                                        "unseen categorical code at row {}, column '{}'",
                                        i + 1,
                                        name
                                    ),
                                });
                            };
                            values[[i, j]] = unseen_code;
                            continue;
                        }
                        values[[i, j]] = inferred_to_schema[old_code];
                    }
                }
            }
        } else {
            // Column not in schema — keep inferred.
            column_kinds.push(inferred.column_kinds[j]);
            schema_cols.push(inferred.schema.columns[j].clone());
        }
    }

    Ok(EncodedDataset {
        headers: inferred.headers,
        values,
        schema: DataSchema {
            columns: schema_cols,
        },
        column_kinds,
    })
}

pub fn encode_recordswith_inferred_schema(
    headers: Vec<String>,
    records: Vec<StringRecord>,
) -> Result<EncodedDataset, String> {
    if records.is_empty() {
        return Err(DataError::EmptyInput {
            reason: "table data cannot be empty".to_string(),
        }
        .into());
    }
    // Schema inference is column-independent: each column scans only its own
    // field across all rows. With wide frames (e.g. biobank: 22 cols × 194k
    // rows) the serial outer loop dominated ingest time, so fan the per-column
    // inference passes out over rayon. Order is preserved because `map` over an
    // indexed parallel iterator collects back in column order.
    let schema_cols = headers
        .par_iter()
        .enumerate()
        .map(|(j, name)| infer_schema_column(name, &records, j).map_err(String::from))
        .collect::<Result<Vec<SchemaColumn>, String>>()?;
    let schema = DataSchema {
        columns: schema_cols,
    };
    encode_recordswith_schema(headers, records, &schema, UnseenCategoryPolicy::Error)
}

pub fn encode_recordswith_schema(
    headers: Vec<String>,
    records: Vec<StringRecord>,
    schema: &DataSchema,
    unseen_policy: UnseenCategoryPolicy,
) -> Result<EncodedDataset, String> {
    let n = records.len();
    if n == 0 {
        return Err(DataError::EmptyInput {
            reason: "table data cannot be empty".to_string(),
        }
        .into());
    }
    let p = headers.len();
    if p == 0 {
        return Err(DataError::EmptyInput {
            reason: "table data must have at least one header column".to_string(),
        }
        .into());
    }
    // Validate the row-width invariant up front. Without this check, records
    // wider than `headers` would be silently truncated (only the first
    // `headers.len()` fields per record would be encoded) and records
    // narrower than `headers` would only fail late when a per-column
    // `rec.get(j)` lookup returned `None`. Reject both cases explicitly so
    // callers cannot accidentally drop data via header/record shape skew.
    for (i, rec) in records.iter().enumerate() {
        if rec.len() != p {
            return Err(DataError::SchemaMismatch {
                reason: format!(
                    "row width mismatch at row {}: got {} fields, expected {} (one per header)",
                    i + 1,
                    rec.len(),
                    p
                ),
            }
            .into());
        }
    }
    let schema_byname: HashMap<&str, &SchemaColumn> = schema
        .columns
        .iter()
        .map(|c| (c.name.as_str(), c))
        .collect();

    // Each column is encoded independently from the same row-major records, so
    // fan the per-column passes out over rayon (columns, not rows, so threads
    // never contend on a shared output cell). Each task returns its dense
    // `(kind, Vec<f64>)`; we then assemble the row-major `Array2` from the
    // collected columns. For wide frames this is the dominant ingest cost.
    let encoded_columns = headers
        .par_iter()
        .enumerate()
        .map(|(j, name)| {
            let inferred_for_extra;
            let col_schema = if let Some(s) = schema_byname.get(name.as_str()) {
                *s
            } else {
                inferred_for_extra =
                    infer_schema_column(name, &records, j).map_err(String::from)?;
                &inferred_for_extra
            };
            let column = encode_one_column(name, &records, j, col_schema, &unseen_policy)?;
            Ok::<(ColumnKindTag, Vec<f64>), String>((col_schema.kind, column))
        })
        .collect::<Result<Vec<(ColumnKindTag, Vec<f64>)>, String>>()?;

    let mut column_kinds = Vec::<ColumnKindTag>::with_capacity(p);
    let mut values = Array2::<f64>::zeros((n, p));
    for (j, (kind, column)) in encoded_columns.into_iter().enumerate() {
        column_kinds.push(kind);
        values
            .column_mut(j)
            .assign(&ndarray::ArrayView1::from(&column));
    }

    Ok(EncodedDataset {
        headers,
        values,
        schema: schema.clone(),
        column_kinds,
    })
}

/// Encode a single column `j` of `records` to its dense `f64` representation
/// under `col_schema`. Continuous/binary values are parsed; categorical values
/// are mapped to their level index (or the unseen code under `unseen_policy`).
/// This is the per-column work unit fanned out across columns in
/// [`encode_recordswith_schema`]; it scans only field `j` of each record so
/// distinct columns never touch shared state.
fn encode_one_column(
    name: &str,
    records: &[StringRecord],
    j: usize,
    col_schema: &SchemaColumn,
    unseen_policy: &UnseenCategoryPolicy,
) -> Result<Vec<f64>, String> {
    let level_map = if matches!(col_schema.kind, ColumnKindTag::Categorical) {
        Some(
            col_schema
                .levels
                .iter()
                .enumerate()
                .map(|(idx, v)| (v.as_str(), idx as f64))
                .collect::<HashMap<_, _>>(),
        )
    } else {
        None
    };

    let mut column = Vec::<f64>::with_capacity(records.len());
    for (i, rec) in records.iter().enumerate() {
        let raw = rec
            .get(j)
            .ok_or_else(|| {
                String::from(DataError::SchemaMismatch {
                    reason: format!("missing field at row {}, col {}", i + 1, j + 1),
                })
            })?
            .trim();
        if raw.is_empty() {
            return Err(DataError::EmptyInput {
                reason: format!("empty field at row {}, column '{}'", i + 1, name),
            }
            .into());
        }
        let val = match col_schema.kind {
            ColumnKindTag::Continuous => raw.parse::<f64>().map_err(|err| {
                String::from(DataError::SchemaMismatch {
                    reason: format!(
                        "column '{}' is continuous in schema but row {} has non-numeric value '{}': {}",
                        name,
                        i + 1,
                        raw,
                        err
                    ),
                })
            })?,
            ColumnKindTag::Binary => {
                let v = raw.parse::<f64>().map_err(|err| {
                    String::from(DataError::SchemaMismatch {
                        reason: format!(
                            "column '{}' is binary in schema but row {} has non-numeric value '{}': {}",
                            name,
                            i + 1,
                            raw,
                            err
                        ),
                    })
                })?;
                if (v - 0.0).abs() >= 1e-12 && (v - 1.0).abs() >= 1e-12 {
                    return Err(DataError::SchemaMismatch {
                        reason: format!(
                            "column '{}' is binary in schema but row {} has value {}; expected 0 or 1",
                            name,
                            i + 1,
                            v
                        ),
                    }
                    .into());
                }
                v
            }
            ColumnKindTag::Categorical => {
                let map = level_map.as_ref().ok_or_else(|| {
                    String::from(DataError::EncodingFailure {
                        reason: "internal categorical schema map missing".to_string(),
                    })
                })?;
                match map.get(raw) {
                    Some(v) => *v,
                    None => unseen_policy
                        .unseen_code_for(name, col_schema.levels.len())
                        .ok_or_else(|| {
                            String::from(DataError::SchemaMismatch {
                                reason: format!(
                                    "unseen level '{}' in categorical column '{}' at row {}; allowed levels: {}",
                                    raw,
                                    name,
                                    i + 1,
                                    col_schema.levels.join(",")
                                ),
                            })
                        })?,
                }
            }
        };
        if !val.is_finite() {
            return Err(DataError::InvalidValue {
                reason: format!("non-finite value at row {}, column '{}'", i + 1, name),
            }
            .into());
        }
        column.push(val);
    }
    Ok(column)
}

fn infer_schema_column(
    name: &str,
    records: &[StringRecord],
    col_idx: usize,
) -> Result<SchemaColumn, DataError> {
    let mut all_numeric = true;
    let mut all_binary = true;
    let mut levels = Vec::<String>::new();
    let mut level_index = HashMap::<String, usize>::new();
    for (i, rec) in records.iter().enumerate() {
        let raw = rec
            .get(col_idx)
            .ok_or_else(|| DataError::SchemaMismatch {
                reason: format!("missing field at row {}, col {}", i + 1, col_idx + 1),
            })?
            .trim();
        if raw.is_empty() {
            return Err(DataError::EmptyInput {
                reason: format!("empty field at row {}, column '{}'", i + 1, name),
            });
        }
        if let Ok(v) = raw.parse::<f64>() {
            if !v.is_finite() {
                return Err(DataError::InvalidValue {
                    reason: format!("non-finite value at row {}, column '{}'", i + 1, name),
                });
            }
            if (v - 0.0).abs() >= 1e-12 && (v - 1.0).abs() >= 1e-12 {
                all_binary = false;
            }
        } else {
            all_numeric = false;
            all_binary = false;
            level_index.entry(raw.to_string()).or_insert_with(|| {
                let idx = levels.len();
                levels.push(raw.to_string());
                idx
            });
        }
    }
    let kind = if all_numeric {
        if all_binary {
            ColumnKindTag::Binary
        } else {
            ColumnKindTag::Continuous
        }
    } else {
        ColumnKindTag::Categorical
    };
    Ok(SchemaColumn {
        name: name.to_string(),
        kind,
        levels: if matches!(kind, ColumnKindTag::Categorical) {
            levels
        } else {
            Vec::new()
        },
    })
}

/// Infer the schema of, and densely encode, a single column presented in
/// column-major form (`name` + its raw string field for every row).
///
/// This is the column-major sibling of the record-driven path: it produces the
/// byte-identical `(SchemaColumn, Vec<f64>)` that `encode_recordswith_inferred_schema`
/// would produce for the same column, but it reads from a `&[&str]` column
/// slice instead of indexing field `col_idx` of every `StringRecord`. It exists
/// so callers holding column-major data (e.g. the Python FFI, which can
/// fingerprint and cache invariant columns shared across many fits of the same
/// base cohort) can encode one column at a time without first materializing the
/// full row-major record table. `col_index` is 1-based only for error text and
/// matches the record-driven messages.
pub fn infer_and_encode_column_major(
    name: &str,
    column: &[&str],
    col_index: usize,
) -> Result<(SchemaColumn, Vec<f64>), String> {
    if column.is_empty() {
        return Err(DataError::EmptyInput {
            reason: "table data cannot be empty".to_string(),
        }
        .into());
    }
    // A typed Python frame prefixes every cell of a categorical-dtype column
    // with `CATEGORICAL_CELL_SENTINEL` so the column is encoded as a factor even
    // when its labels parse as numbers ("0","1","2"). Detect and strip the
    // marker before inference; its presence forces `Categorical` (#1317/#1318).
    let force_categorical = column.iter().any(|c| strip_categorical_sentinel(c).1);
    let mut all_numeric = !force_categorical;
    let mut all_binary = !force_categorical;
    let mut levels = Vec::<String>::new();
    let mut level_index = HashMap::<String, usize>::new();
    let mut trimmed = Vec::<&str>::with_capacity(column.len());
    // Capture the parsed numeric value alongside each trimmed field during the
    // single inference scan, so the encode pass below never re-parses a numeric
    // string. For wide biobank frames the f64 parse dominated ingest, and the
    // record-driven path used to parse every continuous/binary field twice
    // (once to infer the schema, once to encode). `parsed[i]` is `Some(v)` iff
    // field `i` parsed as a finite f64; categorical columns ignore it.
    let mut parsed = Vec::<Option<f64>>::with_capacity(column.len());
    for (i, raw_field) in column.iter().enumerate() {
        // Strip the categorical marker (if any) so the recorded level label and
        // any numeric parse see the user's clean text, not the sentinel.
        let (raw, _) = strip_categorical_sentinel(raw_field);
        let raw = raw.trim();
        if raw.is_empty() {
            return Err(DataError::EmptyInput {
                reason: format!("empty field at row {}, column '{}'", i + 1, name),
            }
            .into());
        }
        // When the source column is dtype-categorical, every cell is a level
        // regardless of whether its label parses as a number.
        if !force_categorical {
            if let Ok(v) = raw.parse::<f64>() {
                if !v.is_finite() {
                    return Err(DataError::InvalidValue {
                        reason: format!("non-finite value at row {}, column '{}'", i + 1, name),
                    }
                    .into());
                }
                if (v - 0.0).abs() >= 1e-12 && (v - 1.0).abs() >= 1e-12 {
                    all_binary = false;
                }
                parsed.push(Some(v));
                trimmed.push(raw);
                continue;
            }
            all_numeric = false;
            all_binary = false;
        }
        level_index.entry(raw.to_string()).or_insert_with(|| {
            let idx = levels.len();
            levels.push(raw.to_string());
            idx
        });
        parsed.push(None);
        trimmed.push(raw);
    }
    let kind = if all_numeric {
        if all_binary {
            ColumnKindTag::Binary
        } else {
            ColumnKindTag::Continuous
        }
    } else {
        ColumnKindTag::Categorical
    };
    let schema = SchemaColumn {
        name: name.to_string(),
        kind,
        levels: if matches!(kind, ColumnKindTag::Categorical) {
            levels
        } else {
            Vec::new()
        },
    };

    let level_map = if matches!(kind, ColumnKindTag::Categorical) {
        Some(
            schema
                .levels
                .iter()
                .enumerate()
                .map(|(idx, v)| (v.as_str(), idx as f64))
                .collect::<HashMap<_, _>>(),
        )
    } else {
        None
    };

    let mut values = Vec::<f64>::with_capacity(trimmed.len());
    for (i, raw) in trimmed.iter().enumerate() {
        let raw = *raw;
        let val = match kind {
            // Continuous/Binary kinds are only selected when every field parsed
            // as a finite f64 during inference, so `parsed[i]` is always `Some`
            // here — reuse it instead of re-parsing the string.
            ColumnKindTag::Continuous => parsed[i].ok_or_else(|| {
                String::from(DataError::EncodingFailure {
                    reason: format!(
                        "internal: continuous column '{}' lost its parsed value at row {} (col {})",
                        name,
                        i + 1,
                        col_index
                    ),
                })
            })?,
            ColumnKindTag::Binary => {
                let v = parsed[i].ok_or_else(|| {
                    String::from(DataError::EncodingFailure {
                        reason: format!(
                            "internal: binary column '{}' lost its parsed value at row {} (col {})",
                            name,
                            i + 1,
                            col_index
                        ),
                    })
                })?;
                if (v - 0.0).abs() >= 1e-12 && (v - 1.0).abs() >= 1e-12 {
                    return Err(DataError::SchemaMismatch {
                        reason: format!(
                            "column '{}' is binary in schema but row {} has value {}; expected 0 or 1",
                            name,
                            i + 1,
                            v
                        ),
                    }
                    .into());
                }
                v
            }
            ColumnKindTag::Categorical => {
                let map = level_map.as_ref().ok_or_else(|| {
                    String::from(DataError::EncodingFailure {
                        reason: "internal categorical schema map missing".to_string(),
                    })
                })?;
                *map.get(raw).ok_or_else(|| {
                    String::from(DataError::EncodingFailure {
                        reason: format!(
                            "internal: level '{}' missing from freshly built map for column '{}' (col {})",
                            raw, name, col_index
                        ),
                    })
                })?
            }
        };
        if !val.is_finite() {
            return Err(DataError::InvalidValue {
                reason: format!("non-finite value at row {}, column '{}'", i + 1, name),
            }
            .into());
        }
        values.push(val);
    }
    Ok((schema, values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_records_rejects_empty_input() {
        let headers = vec!["x".to_string()];
        let schema = DataSchema {
            columns: vec![SchemaColumn {
                name: "x".to_string(),
                kind: ColumnKindTag::Continuous,
                levels: Vec::new(),
            }],
        };

        let err = encode_recordswith_inferred_schema(headers.clone(), Vec::new())
            .expect_err("empty inferred records should error");
        assert_eq!(err, "table data cannot be empty");

        let err =
            encode_recordswith_schema(headers, Vec::new(), &schema, UnseenCategoryPolicy::Error)
                .expect_err("empty schema-guided records should error");
        assert_eq!(err, "table data cannot be empty");
    }

    #[test]
    fn column_major_matches_record_driven_inferred_encode() {
        // The FFI ingest path encodes column-by-column via
        // `infer_and_encode_column_major`; it must produce byte-identical
        // schema + values to the record-driven `encode_recordswith_inferred_schema`
        // for the same frame across all three inferred kinds.
        let headers = vec!["cont".to_string(), "bin".to_string(), "cat".to_string()];
        let raw_rows = vec![
            vec!["1.5", "0", "a"],
            vec!["2.0", "1", "b"],
            vec!["-3.25", "1", "a"],
            vec!["0.0", "0", "c"],
        ];
        let records: Vec<StringRecord> = raw_rows
            .iter()
            .map(|r| StringRecord::from(r.clone()))
            .collect();
        let record_ds = encode_recordswith_inferred_schema(headers.clone(), records)
            .expect("record-driven encode");

        for (j, name) in headers.iter().enumerate() {
            let column: Vec<&str> = raw_rows.iter().map(|r| r[j]).collect();
            let (schema_col, values) =
                infer_and_encode_column_major(name, &column, j + 1).expect("column-major encode");
            assert_eq!(schema_col.kind, record_ds.schema.columns[j].kind);
            assert_eq!(schema_col.levels, record_ds.schema.columns[j].levels);
            for (i, v) in values.iter().enumerate() {
                assert_eq!(*v, record_ds.values[[i, j]], "row {i} col {name}");
            }
        }
    }

    #[test]
    fn encode_records_can_encode_unseen_named_categorical_column() {
        let schema = DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "g".to_string(),
                    kind: ColumnKindTag::Categorical,
                    levels: vec!["a".to_string(), "b".to_string()],
                },
                SchemaColumn {
                    name: "x".to_string(),
                    kind: ColumnKindTag::Categorical,
                    levels: vec!["low".to_string(), "high".to_string()],
                },
            ],
        };
        let headers = vec!["g".to_string(), "x".to_string()];
        let records = vec![StringRecord::from(vec!["new-group", "low"])];
        let policy =
            UnseenCategoryPolicy::encode_unknown_for_columns(HashSet::from(["g".to_string()]));

        let ds =
            encode_recordswith_schema(headers, records, &schema, policy).expect("encoded dataset");

        assert_eq!(ds.values[[0, 0]], 2.0);
        assert_eq!(ds.values[[0, 1]], 0.0);
    }

    #[test]
    fn numeric_valued_dictionary_column_classifies_and_decodes_as_numeric() {
        // Regression for #1162: pyarrow dictionary-encodes low-cardinality
        // *numeric* columns by default (e.g. `Dictionary(Int8, Int64)`).
        // Dictionary encoding is a storage detail, not a semantic type, so such
        // a column must stay numeric — both at classification time
        // (`parquet_field_is_string`) and at decode time
        // (`decode_parquet_batch_column`). Previously the loader matched ALL
        // `Dictionary(_, _)` as string/categorical, silently flipping numeric
        // features to categorical and rejecting valid numeric prediction files
        // with SchemaMismatch.
        use arrow::array::{Array, ArrayRef, DictionaryArray, Int8Array, Int64Array};
        use arrow::datatypes::{DataType, Int8Type};
        use std::sync::Arc;

        // Logical column values: 5, 7, 5, 7, 5 (low-cardinality integers).
        let keys = Int8Array::from(vec![0i8, 1, 0, 1, 0]);
        let dict_values: ArrayRef = Arc::new(Int64Array::from(vec![5i64, 7]));
        let dict: DictionaryArray<Int8Type> = DictionaryArray::new(keys, dict_values);

        // The dictionary's *value* type is numeric, so the column must NOT be
        // classified as string/categorical.
        assert!(matches!(dict.data_type(), DataType::Dictionary(_, _)));
        assert!(
            !parquet_field_is_string(dict.data_type()),
            "Dictionary(Int8, Int64) must not be treated as a string column"
        );

        // A genuine string-valued dictionary still classifies as string.
        let str_dict: DictionaryArray<Int8Type> = vec!["a", "b", "a"].into_iter().collect();
        assert!(
            parquet_field_is_string(str_dict.data_type()),
            "Dictionary(Int8, Utf8) must remain a string column"
        );

        // Decoding the numeric dictionary with `is_string_col = false` must
        // resolve indices through the dictionary and yield the underlying
        // numeric values (not error, not strings).
        let decoded = decode_parquet_batch_column(&dict, dict.len(), 0, "x", false)
            .expect("numeric dictionary column should decode as numeric");
        match decoded {
            ParquetBatchColumn::Numeric(values) => {
                assert_eq!(values, vec![5.0, 7.0, 5.0, 7.0, 5.0]);
            }
            ParquetBatchColumn::Strings(_) => {
                panic!("numeric dictionary column was decoded as strings");
            }
        }

        // End-to-end: write a real parquet file whose only column is the
        // dictionary-encoded numeric one, then load it both ways. This is the
        // exact repro from #1162 (pyarrow's default dictionary encoding of a
        // low-cardinality numeric column).
        use arrow::datatypes::{Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;

        let arrow_schema = Arc::new(Schema::new(vec![Field::new(
            "x",
            dict.data_type().clone(),
            false,
        )]));
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(dict.clone())])
            .expect("record batch with a dictionary numeric column");

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("dict_numeric.parquet");
        {
            let file = std::fs::File::create(&path).expect("create parquet");
            let mut writer =
                ArrowWriter::try_new(file, arrow_schema, None).expect("arrow parquet writer");
            writer.write(&batch).expect("write batch");
            writer.close().expect("close writer");
        }

        // Inferred load: the column must be Continuous (5 and 7 are not 0/1),
        // never Categorical.
        let inferred = load_parquet_inferred(&path, &[]).expect("inferred parquet load");
        assert_eq!(inferred.column_kinds, vec![ColumnKindTag::Continuous]);
        assert_eq!(
            inferred.values.column(0).to_vec(),
            vec![5.0, 7.0, 5.0, 7.0, 5.0]
        );

        // Schema-driven load with the column declared Continuous (as it would be
        // after training on CSV / non-dictionary parquet) must NOT raise the
        // SchemaMismatch that #1162 reported on valid numeric data.
        let schema = DataSchema {
            columns: vec![SchemaColumn {
                name: "x".to_string(),
                kind: ColumnKindTag::Continuous,
                levels: Vec::new(),
            }],
        };
        let schema_loaded =
            load_parquet_with_schema(&path, &schema, UnseenCategoryPolicy::Error, &[])
                .expect("dictionary-encoded numeric parquet must load against a Continuous schema");
        assert_eq!(schema_loaded.column_kinds, vec![ColumnKindTag::Continuous]);
        assert_eq!(
            schema_loaded.values.column(0).to_vec(),
            vec![5.0, 7.0, 5.0, 7.0, 5.0]
        );
    }

    #[test]
    fn encode_records_keeps_unlisted_categorical_columns_strict() {
        let schema = DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "g".to_string(),
                    kind: ColumnKindTag::Categorical,
                    levels: vec!["a".to_string(), "b".to_string()],
                },
                SchemaColumn {
                    name: "x".to_string(),
                    kind: ColumnKindTag::Categorical,
                    levels: vec!["low".to_string(), "high".to_string()],
                },
            ],
        };
        let headers = vec!["g".to_string(), "x".to_string()];
        let records = vec![StringRecord::from(vec!["a", "new-level"])];
        let policy =
            UnseenCategoryPolicy::encode_unknown_for_columns(HashSet::from(["g".to_string()]));

        let err = encode_recordswith_schema(headers, records, &schema, policy)
            .expect_err("ordinary categorical column should stay strict");

        assert!(err.contains("unseen level 'new-level' in categorical column 'x'"));
    }
}

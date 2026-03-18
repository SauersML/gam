use crate::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use csv::{ReaderBuilder, StringRecord};
use ndarray::Array2;
use std::collections::{HashMap, HashSet};
use std::path::Path;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnseenCategoryPolicy {
    Error,
}

#[derive(Clone, Debug)]
pub struct EncodedDataset {
    pub headers: Vec<String>,
    pub values: Array2<f64>,
    pub schema: DataSchema,
    pub column_kinds: Vec<ColumnKindTag>,
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

fn detect_format(path: &Path) -> Result<DataFormat, String> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match ext.as_str() {
        "csv" => Ok(DataFormat::Csv),
        "tsv" | "txt" | "tab" => Ok(DataFormat::Tsv),
        "parquet" | "pq" | "pqt" => Ok(DataFormat::Parquet),
        other => Err(format!(
            "unsupported data file extension '.{other}'; expected csv, tsv, txt, parquet, or pq: '{}'",
            path.display()
        )),
    }
}

// ---------------------------------------------------------------------------
// Unified public API  — format auto-detected, zero extra CLI args
// ---------------------------------------------------------------------------

pub fn load_dataset(path: &Path) -> Result<EncodedDataset, String> {
    load_dataset_projected(path, &[])
}

pub fn load_dataset_projected(
    path: &Path,
    requested_columns: &[String],
) -> Result<EncodedDataset, String> {
    match detect_format(path)? {
        DataFormat::Csv => load_delimited_inferred(path, b',', requested_columns),
        DataFormat::Tsv => load_delimited_inferred(path, b'\t', requested_columns),
        DataFormat::Parquet => load_parquet_inferred(path, requested_columns),
    }
}

pub fn load_datasetwith_schema(
    path: &Path,
    schema: &DataSchema,
    unseen_policy: UnseenCategoryPolicy,
) -> Result<EncodedDataset, String> {
    load_datasetwith_schema_projected(path, schema, unseen_policy, &[])
}

pub fn load_datasetwith_schema_projected(
    path: &Path,
    schema: &DataSchema,
    unseen_policy: UnseenCategoryPolicy,
    requested_columns: &[String],
) -> Result<EncodedDataset, String> {
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
// Legacy wrappers — keep existing call-sites compiling
// ---------------------------------------------------------------------------

pub fn load_csvwith_inferred_schema(path: &Path) -> Result<EncodedDataset, String> {
    load_delimited_inferred(path, b',', &[])
}

pub fn load_csvwith_schema(
    path: &Path,
    schema: &DataSchema,
    unseen_policy: UnseenCategoryPolicy,
) -> Result<EncodedDataset, String> {
    load_delimited_with_schema(path, b',', schema, unseen_policy, &[])
}

// ---------------------------------------------------------------------------
// Delimited (CSV / TSV) — streaming, columnar, single-pass
// ---------------------------------------------------------------------------

/// Maximum number of rows used for schema inference when no schema is provided.
const SCHEMA_SAMPLE_ROWS: usize = 1024;

fn resolve_requested_columns(
    all_headers: &[String],
    requested_columns: &[String],
) -> Result<Vec<usize>, String> {
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
        let available: HashSet<&str> = all_headers.iter().map(String::as_str).collect();
        let missing = requested_columns
            .iter()
            .filter(|name| !available.contains(name.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        return Err(format!(
            "requested column(s) not found in data: {}",
            missing.join(", ")
        ));
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
) -> Result<EncodedDataset, String> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(delimiter)
        .from_path(path)
        .map_err(|e| format!("failed to open '{}': {e}", path.display()))?;

    let all_headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("failed to read headers: {e}"))?
        .iter()
        .map(|s| s.trim().to_string())
        .collect();
    if all_headers.is_empty() {
        return Err("file has no headers".to_string());
    }
    let selected_indices = resolve_requested_columns(&all_headers, requested_columns)?;
    let headers = projected_headers(&all_headers, &selected_indices);
    let p = headers.len();

    // Phase 1: sample rows for schema inference, accumulate into column vecs.
    let mut col_vecs: Vec<Vec<f64>> = vec![Vec::new(); p];
    let mut sample_strings: Vec<Vec<String>> = vec![Vec::new(); p];
    let mut sample_count: usize = 0;
    let mut total_rows: usize = 0;

    // Per-column inference state (mirrors infer_schema_column logic).
    let mut all_numeric: Vec<bool> = vec![true; p];
    let mut all_binary: Vec<bool> = vec![true; p];
    let mut level_index: Vec<HashMap<String, usize>> = vec![HashMap::new(); p];
    let mut levels: Vec<Vec<String>> = vec![Vec::new(); p];

    let mut record = StringRecord::new();
    while rdr
        .read_record(&mut record)
        .map_err(|e| format!("failed reading row: {e}"))?
    {
        if record.len() != all_headers.len() {
            return Err(format!(
                "row width mismatch at row {}: got {} fields, expected {}",
                total_rows + 1,
                record.len(),
                all_headers.len()
            ));
        }
        total_rows += 1;

        for j in 0..p {
            let raw = record.get(selected_indices[j]).unwrap().trim();
            if raw.is_empty() {
                return Err(format!(
                    "empty field at row {}, column '{}'",
                    total_rows, &headers[j]
                ));
            }

            // Schema inference on sample window.
            if sample_count < SCHEMA_SAMPLE_ROWS {
                if let Ok(v) = raw.parse::<f64>() {
                    if !v.is_finite() {
                        return Err(format!(
                            "non-finite value at row {}, column '{}'",
                            total_rows, &headers[j]
                        ));
                    }
                    if (v - 0.0).abs() >= 1e-12 && (v - 1.0).abs() >= 1e-12 {
                        all_binary[j] = false;
                    }
                    col_vecs[j].push(v);
                } else {
                    all_numeric[j] = false;
                    all_binary[j] = false;
                    if !level_index[j].contains_key(raw) {
                        level_index[j].insert(raw.to_string(), levels[j].len());
                        levels[j].push(raw.to_string());
                    }
                    // Store raw string; we'll encode after schema is finalized.
                    sample_strings[j].push(raw.to_string());
                    col_vecs[j].push(f64::NAN); // placeholder
                }
            } else {
                // After sample window: we still accumulate inference state for
                // correctness (a column that looks binary in the first 1024 rows
                // may contain 2.5 on row 1025).
                if let Ok(v) = raw.parse::<f64>() {
                    if !v.is_finite() {
                        return Err(format!(
                            "non-finite value at row {}, column '{}'",
                            total_rows, &headers[j]
                        ));
                    }
                    if (v - 0.0).abs() >= 1e-12 && (v - 1.0).abs() >= 1e-12 {
                        all_binary[j] = false;
                    }
                    col_vecs[j].push(v);
                } else {
                    all_numeric[j] = false;
                    all_binary[j] = false;
                    if !level_index[j].contains_key(raw) {
                        level_index[j].insert(raw.to_string(), levels[j].len());
                        levels[j].push(raw.to_string());
                    }
                    let idx = *level_index[j].get(raw).unwrap();
                    col_vecs[j].push(idx as f64);
                }
            }
        }
        if sample_count < SCHEMA_SAMPLE_ROWS {
            sample_count += 1;
        }
    }

    if total_rows == 0 {
        return Err("file has no rows".to_string());
    }

    // Build schema from inference state.
    let mut schema_cols = Vec::<SchemaColumn>::with_capacity(p);
    let mut column_kinds = Vec::<ColumnKindTag>::with_capacity(p);
    for j in 0..p {
        let kind = if all_numeric[j] {
            if all_binary[j] {
                ColumnKindTag::Binary
            } else {
                ColumnKindTag::Continuous
            }
        } else {
            ColumnKindTag::Categorical
        };
        column_kinds.push(kind);
        schema_cols.push(SchemaColumn {
            name: headers[j].clone(),
            kind,
            levels: if matches!(kind, ColumnKindTag::Categorical) {
                levels[j].clone()
            } else {
                Vec::new()
            },
        });
    }

    // Fix up the sample-window rows for categorical columns: the NaN
    // placeholders need to be replaced with integer level codes.
    for j in 0..p {
        if matches!(column_kinds[j], ColumnKindTag::Categorical) {
            let map = &level_index[j];
            let mut str_idx = 0;
            for i in 0..sample_count.min(total_rows) {
                if col_vecs[j][i].is_nan() {
                    let raw = &sample_strings[j][str_idx];
                    str_idx += 1;
                    let code = *map.get(raw.as_str()).ok_or_else(|| {
                        format!(
                            "internal error: sample string '{}' missing from level map for column '{}'",
                            raw, &headers[j]
                        )
                    })?;
                    col_vecs[j][i] = code as f64;
                }
            }
        }
    }

    // Assemble into Array2 (column-major fill is cache-friendly for column vecs).
    let mut values = Array2::<f64>::zeros((total_rows, p));
    for j in 0..p {
        for (i, &v) in col_vecs[j].iter().enumerate() {
            if !v.is_finite() {
                return Err(format!(
                    "non-finite value at row {}, column '{}'",
                    i + 1,
                    &headers[j]
                ));
            }
            values[[i, j]] = v;
        }
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

fn load_delimited_with_schema(
    path: &Path,
    delimiter: u8,
    schema: &DataSchema,
    unseen_policy: UnseenCategoryPolicy,
    requested_columns: &[String],
) -> Result<EncodedDataset, String> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(delimiter)
        .from_path(path)
        .map_err(|e| format!("failed to open '{}': {e}", path.display()))?;

    let all_headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("failed to read headers: {e}"))?
        .iter()
        .map(|s| s.trim().to_string())
        .collect();
    if all_headers.is_empty() {
        return Err("file has no headers".to_string());
    }
    let selected_indices = resolve_requested_columns(&all_headers, requested_columns)?;
    let headers = projected_headers(&all_headers, &selected_indices);
    let p = headers.len();

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
    let mut record = StringRecord::new();
    while rdr
        .read_record(&mut record)
        .map_err(|e| format!("failed reading row: {e}"))?
    {
        if record.len() != all_headers.len() {
            return Err(format!(
                "row width mismatch at row {}: got {} fields, expected {}",
                total_rows + 1,
                record.len(),
                all_headers.len()
            ));
        }
        total_rows += 1;

        for j in 0..p {
            let raw = record.get(selected_indices[j]).unwrap().trim();
            if raw.is_empty() {
                return Err(format!(
                    "empty field at row {}, column '{}'",
                    total_rows, &headers[j]
                ));
            }

            if needs_inference[j] {
                // Accumulate inference state.
                if let Ok(v) = raw.parse::<f64>() {
                    if !v.is_finite() {
                        return Err(format!(
                            "non-finite value at row {}, column '{}'",
                            total_rows, &headers[j]
                        ));
                    }
                    if (v - 0.0).abs() >= 1e-12 && (v - 1.0).abs() >= 1e-12 {
                        infer_all_binary[j] = false;
                    }
                    col_vecs[j].push(v);
                } else {
                    infer_all_numeric[j] = false;
                    infer_all_binary[j] = false;
                    if !infer_level_index[j].contains_key(raw) {
                        infer_level_index[j].insert(raw.to_string(), infer_levels[j].len());
                        infer_levels[j].push(raw.to_string());
                    }
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
                    unseen_policy,
                )?;
                col_vecs[j].push(val);
            }
        }
    }

    if total_rows == 0 {
        return Err("file has no rows".to_string());
    }

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
                col_meta[j].schema_col.levels = infer_levels[j].clone();
                // Fix up NaN placeholders.
                for (row_idx, raw) in &infer_strings[j] {
                    let code = *infer_level_index[j].get(raw.as_str()).unwrap();
                    col_vecs[j][*row_idx] = code as f64;
                }
            }
        }
        column_kinds.push(col_meta[j].kind);
    }

    // Assemble Array2.
    let mut values = Array2::<f64>::zeros((total_rows, p));
    for j in 0..p {
        for (i, &v) in col_vecs[j].iter().enumerate() {
            if !v.is_finite() {
                return Err(format!(
                    "non-finite value at row {}, column '{}'",
                    i + 1,
                    &headers[j]
                ));
            }
            values[[i, j]] = v;
        }
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
    unseen_policy: UnseenCategoryPolicy,
) -> Result<f64, String> {
    let val = match meta.kind {
        ColumnKindTag::Continuous => raw.parse::<f64>().map_err(|_| {
            format!(
                "column '{}' is continuous in schema but row {} has non-numeric value '{}'",
                col_name, row, raw
            )
        })?,
        ColumnKindTag::Binary => {
            let v = raw.parse::<f64>().map_err(|_| {
                format!(
                    "column '{}' is binary in schema but row {} has non-numeric value '{}'",
                    col_name, row, raw
                )
            })?;
            if (v - 0.0).abs() >= 1e-12 && (v - 1.0).abs() >= 1e-12 {
                return Err(format!(
                    "column '{}' is binary in schema but row {} has value {}; expected 0 or 1",
                    col_name, row, v
                ));
            }
            v
        }
        ColumnKindTag::Categorical => {
            let map = meta
                .level_map
                .as_ref()
                .ok_or_else(|| "internal categorical schema map missing".to_string())?;
            match map.get(raw) {
                Some(v) => *v,
                None => match unseen_policy {
                    UnseenCategoryPolicy::Error => {
                        return Err(format!(
                            "unseen level '{}' in categorical column '{}' at row {}",
                            raw, col_name, row
                        ));
                    }
                },
            }
        }
    };
    if !val.is_finite() {
        return Err(format!(
            "non-finite value at row {}, column '{}'",
            row, col_name
        ));
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

fn load_parquet_inferred(
    path: &Path,
    requested_columns: &[String],
) -> Result<EncodedDataset, String> {
    use arrow::array::{
        Array as ArrowArray, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array,
        Int32Array, Int64Array, StringArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    };
    use arrow::datatypes::DataType;
    use parquet::arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder};
    use std::fs::File;

    let file = File::open(path)
        .map_err(|e| format!("failed to open parquet '{}': {e}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("failed to read parquet metadata '{}': {e}", path.display()))?;

    let full_schema = builder.schema().clone();
    let all_headers: Vec<String> = full_schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    let selected_indices = resolve_requested_columns(&all_headers, requested_columns)?;
    let headers = projected_headers(&all_headers, &selected_indices);
    let selected_fields = selected_indices
        .iter()
        .map(|&idx| full_schema.fields()[idx].clone())
        .collect::<Vec<_>>();
    let projection =
        ProjectionMask::roots(builder.parquet_schema(), selected_indices.iter().copied());
    let reader = builder
        .with_projection(projection)
        .build()
        .map_err(|e| format!("failed to build parquet reader: {e}"))?;
    let p = headers.len();

    // Collect all batches.
    let mut col_vecs: Vec<Vec<f64>> = vec![Vec::new(); p];
    // For string columns: accumulate raw strings to build level maps.
    let mut string_cols: Vec<Option<Vec<String>>> = (0..p).map(|_| None).collect();
    let mut is_string_col: Vec<bool> = vec![false; p];

    for (j, field) in selected_fields.iter().enumerate() {
        match field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Dictionary(_, _) => {
                is_string_col[j] = true;
                string_cols[j] = Some(Vec::new());
            }
            _ => {}
        }
    }

    for batch_result in reader {
        let batch =
            batch_result.map_err(|e| format!("failed to read parquet record batch: {e}"))?;
        let n_rows = batch.num_rows();

        for j in 0..p {
            let col = batch.column(j);
            if is_string_col[j] {
                // String/dictionary column → accumulate raw strings.
                let strings = string_cols[j].as_mut().unwrap();
                // Try to get as StringArray (handles Utf8, LargeUtf8, Dictionary).
                let str_arr = col.as_any().downcast_ref::<StringArray>();
                if let Some(arr) = str_arr {
                    for i in 0..n_rows {
                        if ArrowArray::is_null(arr, i) {
                            return Err(format!(
                                "null value at row {}, column '{}'",
                                col_vecs[j].len() + i + 1,
                                &headers[j]
                            ));
                        }
                        strings.push(arr.value(i).to_string());
                    }
                } else {
                    // Dictionary-encoded or LargeUtf8: cast to StringArray via arrow cast.
                    let casted = arrow::compute::cast(col, &DataType::Utf8).map_err(|e| {
                        format!("failed to cast column '{}' to string: {e}", &headers[j])
                    })?;
                    let arr = casted
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| {
                            format!(
                                "column '{}' could not be read as string after cast",
                                &headers[j]
                            )
                        })?;
                    for i in 0..n_rows {
                        if ArrowArray::is_null(arr, i) {
                            return Err(format!(
                                "null value at row {}, column '{}'",
                                col_vecs[j].len() + i + 1,
                                &headers[j]
                            ));
                        }
                        strings.push(arr.value(i).to_string());
                    }
                }
                // Push NaN placeholders into col_vecs; will be replaced.
                for _ in 0..n_rows {
                    col_vecs[j].push(f64::NAN);
                }
            } else {
                // Numeric / boolean column → parse to f64.
                let base_row = col_vecs[j].len();
                // Check for nulls.
                if col.null_count() > 0 {
                    for i in 0..n_rows {
                        if col.is_null(i) {
                            return Err(format!(
                                "null value at row {}, column '{}'",
                                base_row + i + 1,
                                &headers[j]
                            ));
                        }
                    }
                }
                match col.data_type() {
                    DataType::Float64 => {
                        let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
                        col_vecs[j].extend(arr.values().iter().copied());
                    }
                    DataType::Float32 => {
                        let arr = col.as_any().downcast_ref::<Float32Array>().unwrap();
                        col_vecs[j].extend(arr.values().iter().map(|&v| v as f64));
                    }
                    DataType::Int64 => {
                        let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
                        col_vecs[j].extend(arr.values().iter().map(|&v| v as f64));
                    }
                    DataType::Int32 => {
                        let arr = col.as_any().downcast_ref::<Int32Array>().unwrap();
                        col_vecs[j].extend(arr.values().iter().map(|&v| v as f64));
                    }
                    DataType::Int16 => {
                        let arr = col.as_any().downcast_ref::<Int16Array>().unwrap();
                        col_vecs[j].extend(arr.values().iter().map(|&v| v as f64));
                    }
                    DataType::Int8 => {
                        let arr = col.as_any().downcast_ref::<Int8Array>().unwrap();
                        col_vecs[j].extend(arr.values().iter().map(|&v| v as f64));
                    }
                    DataType::UInt64 => {
                        let arr = col.as_any().downcast_ref::<UInt64Array>().unwrap();
                        col_vecs[j].extend(arr.values().iter().map(|&v| v as f64));
                    }
                    DataType::UInt32 => {
                        let arr = col.as_any().downcast_ref::<UInt32Array>().unwrap();
                        col_vecs[j].extend(arr.values().iter().map(|&v| v as f64));
                    }
                    DataType::UInt16 => {
                        let arr = col.as_any().downcast_ref::<UInt16Array>().unwrap();
                        col_vecs[j].extend(arr.values().iter().map(|&v| v as f64));
                    }
                    DataType::UInt8 => {
                        let arr = col.as_any().downcast_ref::<UInt8Array>().unwrap();
                        col_vecs[j].extend(arr.values().iter().map(|&v| v as f64));
                    }
                    DataType::Boolean => {
                        let arr = col.as_any().downcast_ref::<BooleanArray>().unwrap();
                        for i in 0..n_rows {
                            col_vecs[j].push(if arr.value(i) { 1.0 } else { 0.0 });
                        }
                    }
                    other => {
                        return Err(format!(
                            "unsupported parquet column type {:?} for column '{}'",
                            other, &headers[j]
                        ));
                    }
                }
                // Validate finiteness for this batch.
                for i in base_row..col_vecs[j].len() {
                    if !col_vecs[j][i].is_finite() {
                        return Err(format!(
                            "non-finite value at row {}, column '{}'",
                            i + 1,
                            &headers[j]
                        ));
                    }
                }
            }
        }
    }

    let total_rows = col_vecs[0].len();
    if total_rows == 0 {
        return Err("parquet file has no rows".to_string());
    }

    // Build schema: infer kind from data.
    let mut schema_cols = Vec::<SchemaColumn>::with_capacity(p);
    let mut column_kinds = Vec::<ColumnKindTag>::with_capacity(p);

    for j in 0..p {
        if is_string_col[j] {
            // Categorical.
            let strings = string_cols[j].as_ref().unwrap();
            let mut level_index: HashMap<String, usize> = HashMap::new();
            let mut levels_vec: Vec<String> = Vec::new();
            for s in strings {
                if !level_index.contains_key(s.as_str()) {
                    level_index.insert(s.clone(), levels_vec.len());
                    levels_vec.push(s.clone());
                }
            }
            // Encode into col_vecs.
            for (i, s) in strings.iter().enumerate() {
                col_vecs[j][i] = *level_index.get(s.as_str()).unwrap() as f64;
            }
            column_kinds.push(ColumnKindTag::Categorical);
            schema_cols.push(SchemaColumn {
                name: headers[j].clone(),
                kind: ColumnKindTag::Categorical,
                levels: levels_vec,
            });
        } else {
            // Numeric: check if binary.
            let all_binary = col_vecs[j]
                .iter()
                .all(|&v| (v - 0.0).abs() < 1e-12 || (v - 1.0).abs() < 1e-12);
            let kind = if all_binary {
                ColumnKindTag::Binary
            } else {
                ColumnKindTag::Continuous
            };
            column_kinds.push(kind);
            schema_cols.push(SchemaColumn {
                name: headers[j].clone(),
                kind,
                levels: Vec::new(),
            });
        }
    }

    // Assemble Array2.
    let mut values = Array2::<f64>::zeros((total_rows, p));
    for j in 0..p {
        for (i, &v) in col_vecs[j].iter().enumerate() {
            values[[i, j]] = v;
        }
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
) -> Result<EncodedDataset, String> {
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

            // If schema says categorical but inferred data was also categorical,
            // re-map levels to match the schema's level order.
            if matches!(sc.kind, ColumnKindTag::Categorical)
                && matches!(inferred.column_kinds[j], ColumnKindTag::Categorical)
            {
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
                        schema_level_map.get(lv.as_str()).copied().ok_or_else(|| {
                            format!("unseen level '{}' in categorical column '{}'", lv, name)
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                for i in 0..n {
                    let old_code = values[[i, j]] as usize;
                    if old_code >= inferred_to_schema.len() {
                        match unseen_policy {
                            UnseenCategoryPolicy::Error => {
                                return Err(format!(
                                    "unseen categorical code at row {}, column '{}'",
                                    i + 1,
                                    name
                                ));
                            }
                        }
                    }
                    values[[i, j]] = inferred_to_schema[old_code];
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

// ---------------------------------------------------------------------------
// Legacy: encode_recordswith_schema — kept for test backward compatibility
// ---------------------------------------------------------------------------

pub fn encode_recordswith_inferred_schema(
    headers: Vec<String>,
    records: Vec<StringRecord>,
) -> Result<EncodedDataset, String> {
    let mut schema_cols = Vec::<SchemaColumn>::with_capacity(headers.len());
    for (j, name) in headers.iter().enumerate() {
        schema_cols.push(infer_schema_column(name, &records, j)?);
    }
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
    let p = headers.len();
    let mut values = Array2::<f64>::zeros((n, p));
    let schema_byname: HashMap<&str, &SchemaColumn> = schema
        .columns
        .iter()
        .map(|c| (c.name.as_str(), c))
        .collect();
    let mut column_kinds = Vec::<ColumnKindTag>::with_capacity(p);

    for (j, name) in headers.iter().enumerate() {
        let inferred_for_extra;
        let col_schema = if let Some(s) = schema_byname.get(name.as_str()) {
            *s
        } else {
            inferred_for_extra = infer_schema_column(name, &records, j)?;
            &inferred_for_extra
        };
        column_kinds.push(col_schema.kind);

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

        for (i, rec) in records.iter().enumerate() {
            let raw = rec
                .get(j)
                .ok_or_else(|| format!("missing field at row {}, col {}", i + 1, j + 1))?
                .trim();
            if raw.is_empty() {
                return Err(format!("empty field at row {}, column '{}'", i + 1, name));
            }
            let val = match col_schema.kind {
                ColumnKindTag::Continuous => raw.parse::<f64>().map_err(|_| {
                    format!(
                        "column '{}' is continuous in schema but row {} has non-numeric value '{}'",
                        name,
                        i + 1,
                        raw
                    )
                })?,
                ColumnKindTag::Binary => {
                    let v = raw.parse::<f64>().map_err(|_| {
                        format!(
                            "column '{}' is binary in schema but row {} has non-numeric value '{}'",
                            name,
                            i + 1,
                            raw
                        )
                    })?;
                    if (v - 0.0).abs() >= 1e-12 && (v - 1.0).abs() >= 1e-12 {
                        return Err(format!(
                            "column '{}' is binary in schema but row {} has value {}; expected 0 or 1",
                            name,
                            i + 1,
                            v
                        ));
                    }
                    v
                }
                ColumnKindTag::Categorical => {
                    let map = level_map
                        .as_ref()
                        .ok_or_else(|| "internal categorical schema map missing".to_string())?;
                    match map.get(raw) {
                        Some(v) => *v,
                        None => match unseen_policy {
                            UnseenCategoryPolicy::Error => {
                                return Err(format!(
                                    "unseen level '{}' in categorical column '{}' at row {}; allowed levels: {}",
                                    raw,
                                    name,
                                    i + 1,
                                    col_schema.levels.join(",")
                                ));
                            }
                        },
                    }
                }
            };
            if !val.is_finite() {
                return Err(format!(
                    "non-finite value at row {}, column '{}'",
                    i + 1,
                    name
                ));
            }
            values[[i, j]] = val;
        }
    }

    Ok(EncodedDataset {
        headers,
        values,
        schema: schema.clone(),
        column_kinds,
    })
}

fn infer_schema_column(
    name: &str,
    records: &[StringRecord],
    col_idx: usize,
) -> Result<SchemaColumn, String> {
    let mut all_numeric = true;
    let mut all_binary = true;
    let mut levels = Vec::<String>::new();
    let mut level_index = HashMap::<String, usize>::new();
    for (i, rec) in records.iter().enumerate() {
        let raw = rec
            .get(col_idx)
            .ok_or_else(|| format!("missing field at row {}, col {}", i + 1, col_idx + 1))?
            .trim();
        if raw.is_empty() {
            return Err(format!("empty field at row {}, column '{}'", i + 1, name));
        }
        if let Ok(v) = raw.parse::<f64>() {
            if !v.is_finite() {
                return Err(format!(
                    "non-finite value at row {}, column '{}'",
                    i + 1,
                    name
                ));
            }
            if (v - 0.0).abs() >= 1e-12 && (v - 1.0).abs() >= 1e-12 {
                all_binary = false;
            }
        } else {
            all_numeric = false;
            all_binary = false;
            if !level_index.contains_key(raw) {
                level_index.insert(raw.to_string(), levels.len());
                levels.push(raw.to_string());
            }
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

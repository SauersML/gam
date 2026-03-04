use crate::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use csv::{ReaderBuilder, StringRecord};
use ndarray::Array2;
use std::collections::HashMap;
use std::path::Path;

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

pub fn load_csv_with_inferred_schema(path: &Path) -> Result<EncodedDataset, String> {
    let (headers, records) = read_csv_raw(path)?;
    encode_records_with_inferred_schema(headers, records)
}

pub fn load_csv_with_schema(
    path: &Path,
    schema: &DataSchema,
    unseen_policy: UnseenCategoryPolicy,
) -> Result<EncodedDataset, String> {
    let (headers, records) = read_csv_raw(path)?;
    encode_records_with_schema(headers, records, schema, unseen_policy)
}

pub fn read_csv_raw(path: &Path) -> Result<(Vec<String>, Vec<StringRecord>), String> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if ext != "csv" {
        return Err(format!(
            "only CSV is currently supported by this entrypoint; got '{}'",
            path.display()
        ));
    }

    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("failed to open csv '{}': {e}", path.display()))?;

    let headers = rdr
        .headers()
        .map_err(|e| format!("failed to read csv headers: {e}"))?
        .iter()
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();
    if headers.is_empty() {
        return Err("csv has no headers".to_string());
    }

    let mut records = Vec::<StringRecord>::new();
    for rec in rdr.records() {
        let record = rec.map_err(|e| format!("failed reading csv row: {e}"))?;
        if record.len() != headers.len() {
            return Err(format!(
                "csv row width mismatch: got {} fields, expected {}",
                record.len(),
                headers.len()
            ));
        }
        records.push(record);
    }
    if records.is_empty() {
        return Err("csv has no rows".to_string());
    }
    Ok((headers, records))
}

pub fn encode_records_with_inferred_schema(
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
    encode_records_with_schema(headers, records, &schema, UnseenCategoryPolicy::Error)
}

pub fn encode_records_with_schema(
    headers: Vec<String>,
    records: Vec<StringRecord>,
    schema: &DataSchema,
    unseen_policy: UnseenCategoryPolicy,
) -> Result<EncodedDataset, String> {
    let n = records.len();
    let p = headers.len();
    let mut values = Array2::<f64>::zeros((n, p));
    let schema_by_name: HashMap<&str, &SchemaColumn> = schema
        .columns
        .iter()
        .map(|c| (c.name.as_str(), c))
        .collect();
    let mut column_kinds = Vec::<ColumnKindTag>::with_capacity(p);

    for (j, name) in headers.iter().enumerate() {
        let inferred_for_extra;
        let col_schema = if let Some(s) = schema_by_name.get(name.as_str()) {
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

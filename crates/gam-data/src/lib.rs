use csv::{ReaderBuilder, StringRecord};
use ndarray::{Array2, ArrayViewMut1, Axis, s};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;

fn natural_level_cmp(a: &str, b: &str) -> Ordering {
    let mut ia = 0;
    let mut ib = 0;
    let ba = a.as_bytes();
    let bb = b.as_bytes();
    while ia < ba.len() && ib < bb.len() {
        if ba[ia].is_ascii_digit() && bb[ib].is_ascii_digit() {
            let sa = ia;
            let sb = ib;
            while ia < ba.len() && ba[ia].is_ascii_digit() {
                ia += 1;
            }
            while ib < bb.len() && bb[ib].is_ascii_digit() {
                ib += 1;
            }
            let da = &a[sa..ia];
            let db = &b[sb..ib];
            let ta = da.trim_start_matches('0');
            let tb = db.trim_start_matches('0');
            let ta = if ta.is_empty() { "0" } else { ta };
            let tb = if tb.is_empty() { "0" } else { tb };
            match ta.len().cmp(&tb.len()).then_with(|| ta.cmp(tb)) {
                Ordering::Equal if da.len() != db.len() => return da.len().cmp(&db.len()),
                Ordering::Equal => {}
                ord => return ord,
            }
        } else {
            match ba[ia].cmp(&bb[ib]) {
                Ordering::Equal => {
                    ia += 1;
                    ib += 1;
                }
                ord => return ord,
            }
        }
    }
    ba.len().cmp(&bb.len())
}

fn sort_levels_canonical(levels: &mut [String]) {
    levels.sort_by(|a, b| natural_level_cmp(a, b));
}

/// Canonical bit key for a floating-point categorical / grouping level.
///
/// Factor dummies, random-effect groups, `by=` gates and factor-smooth blocks
/// all identify a level by the raw bits of its numeric code — they intern the
/// observed codes with `f64::to_bits()` and, at fit/predict time, gate each row
/// by `data_bits == level_bits`. Raw `to_bits()` is a **bit** identity, not the
/// **numeric** equality IEEE-754 defines, and the two disagree in exactly two
/// places:
///
/// * **Signed zero.** `+0.0` is `0x0000_0000_0000_0000` and `-0.0` is
///   `0x8000_0000_0000_0000`, yet IEEE-754 guarantees `+0.0 == -0.0`. Keying on
///   raw bits splits one physical group into two: a row whose code is `-0.0`
///   matches no `+0.0` dummy, so its factor / random effect silently drops and
///   the prediction collapses onto the intercept / population mean. Signed zero
///   arises routinely from ordinary float arithmetic on a computed group column
///   (`-1.0 * 0.0`, a centred/differenced column landing on `-0.0`, `np.round`
///   emitting `-0.0`), and the miss is silent — no schema error, `check()` still
///   reports `ok=True`. See #2145 (random effect) and #2146 (factor dummy).
/// * **NaN.** Every quiet/signalling NaN payload and sign bit denotes "not a
///   number", so `2^53`-ish distinct bit patterns would otherwise intern as
///   distinct levels. (NaN group codes are rejected upstream on most paths, but
///   canonicalising here keeps the key numerically honest regardless.)
///
/// This maps both encodings to a single canonical key while leaving every
/// ordinary finite value bit-stable, so that
/// `canonical_level_bits(a) == canonical_level_bits(b)` iff `a` and `b` name the
/// same real-valued level. Interning and lookup must **both** route through this
/// function; because it is idempotent, applying it to an already-canonical
/// frozen level set is a no-op.
#[inline]
pub fn canonical_level_bits(v: f64) -> u64 {
    if v == 0.0 {
        // Matches both +0.0 and -0.0 (IEEE-754: -0.0 == 0.0); collapse to +0.0.
        0.0_f64.to_bits()
    } else if v.is_nan() {
        // Collapse every NaN payload/sign to one canonical quiet-NaN key.
        f64::NAN.to_bits()
    } else {
        v.to_bits()
    }
}

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
    /// (CSV/TSV row read, parquet metadata, file extension detection, Arrow
    /// record-batch reads, or dictionary decoding).
    ParseError { reason: String },
    /// Internal encoding bookkeeping failed: a categorical map expected by the
    /// schema path was missing, or a level expected to be present in the
    /// per-column inference state was not found during fix-up.
    EncodingFailure { reason: String },
    /// The source has no headers, no rows, or contains an empty / missing
    /// field at a row that requires a value.
    EmptyInput { reason: String },
    /// A cell value cannot be used as a feature: non-finite float, Arrow null,
    /// or an unsupported Arrow data type for the column.
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataSchema {
    pub columns: Vec<SchemaColumn>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchemaColumn {
    pub name: String,
    pub kind: ColumnKindTag,
    #[serde(default)]
    pub levels: Vec<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ColumnKindTag {
    Continuous,
    Binary,
    Categorical,
}

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
    load_dataset_projected_with_categorical_roles(path, requested_columns, &HashSet::new())
}

/// Schema-inferring projected loader that forces a set of columns to
/// [`ColumnKindTag::Categorical`] regardless of whether their labels parse as
/// numbers.
///
/// An untyped CSV/TSV/parquet-numeric frame cannot carry the dtype the typed
/// Python frame stamps via [`CATEGORICAL_CELL_SENTINEL`], so the value-based
/// inferer would otherwise demote an integer/numeric-coded factor (e.g. a
/// `group(region)` grouping coded `0,1,2,3`) to `Continuous` and fit a single
/// numeric ramp instead of one centred factor level per code. That makes the
/// CLI's design strictly lower-capacity than the Python `gamfit.fit` design for
/// the same data, which generalizes worse on every seed.
///
/// `categorical_roles` is keyed on the *formula role*, not on a value
/// heuristic: a column is forced categorical only when the formula uses it in a
/// role that is a factor by construction (`group(g)` / `factor(g)` / `re(g)`
/// random-effect terms, or a categorical/multinomial response). A bare `+ x`
/// linear term and a smooth argument `s(x)` are deliberately NOT included — they
/// stay value-inferred, so a genuinely continuous integer covariate like
/// `s(age)` or `+ age` remains `Continuous`. This mirrors the Python sentinel
/// outcome (`force_categorical`, the column-major inferer) while keying it on
/// the role the user actually declared.
pub fn load_dataset_projected_with_categorical_roles(
    path: &Path,
    requested_columns: &[String],
    categorical_roles: &HashSet<&str>,
) -> Result<EncodedDataset, DataError> {
    match detect_format(path)? {
        DataFormat::Csv => {
            load_delimited_inferred(path, b',', requested_columns, categorical_roles)
        }
        DataFormat::Tsv => {
            load_delimited_inferred(path, b'\t', requested_columns, categorical_roles)
        }
        DataFormat::Parquet => load_parquet_inferred(path, requested_columns, categorical_roles),
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
    load_delimited_inferred(path, b',', &[], &HashSet::new())
}

// ---------------------------------------------------------------------------
// Delimited (CSV / TSV) — streaming, columnar, single-pass
// ---------------------------------------------------------------------------

/// Maximum number of rows used for schema inference when no schema is provided.
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
    categorical_roles: &HashSet<&str>,
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

    // Pass 1 discovers only the schema and row count. Keeping one inference
    // state per column (instead of one owned String per cell) makes peak memory
    // independent of the file's textual representation.
    let mut inference = vec![DelimitedInferenceState::default(); p];
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
        for (j, &selected_idx) in selected_indices.iter().enumerate() {
            inference[j].observe(
                record.get(selected_idx).unwrap().trim(),
                total_rows,
                &headers[j],
            )?;
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
        return Err(DataError::EmptyInput {
            reason: "file has no rows".to_string(),
        });
    }

    let t_schema = std::time::Instant::now();
    let column_kinds = inference
        .iter()
        .enumerate()
        .map(|(j, state)| state.kind(categorical_roles.contains(headers[j].as_str())))
        .collect::<Vec<_>>();
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

    // Pass 2 parses directly into the one final allocation. Categorical cells
    // receive provisional encounter-order codes while only the UNIQUE level
    // strings remain resident; a final in-place remap establishes canonical
    // sorted codes without retaining a per-cell string table.
    let t_assemble = std::time::Instant::now();
    let mut values = Array2::<f64>::zeros((total_rows, p));
    let mut categorical_encoders = (0..p)
        .map(|j| {
            matches!(column_kinds[j], ColumnKindTag::Categorical).then(CategoricalEncoder::default)
        })
        .collect::<Vec<_>>();
    let mut encode_rdr = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(delimiter)
        .from_path(path)
        .map_err(|e| DataError::ParseError {
            reason: format!("failed to reopen '{}': {e}", path.display()),
        })?;
    encode_rdr.headers().map_err(|e| DataError::ParseError {
        reason: format!("failed to reread headers: {e}"),
    })?;
    let mut encoded_rows = 0usize;
    while encode_rdr
        .read_record(&mut record)
        .map_err(|e| DataError::ParseError {
            reason: format!("failed reading row: {e}"),
        })?
    {
        if record.len() != all_headers.len() {
            return Err(DataError::SchemaMismatch {
                reason: format!(
                    "row width mismatch at row {}: got {} fields, expected {}",
                    encoded_rows + 1,
                    record.len(),
                    all_headers.len()
                ),
            });
        }
        if encoded_rows >= total_rows {
            return Err(DataError::SchemaMismatch {
                reason: "data file changed while its schema was being discovered".to_string(),
            });
        }
        for (j, &selected_idx) in selected_indices.iter().enumerate() {
            let raw = record.get(selected_idx).unwrap().trim();
            values[[encoded_rows, j]] = match column_kinds[j] {
                ColumnKindTag::Continuous | ColumnKindTag::Binary => {
                    parse_inferred_numeric_cell(raw, encoded_rows + 1, &headers[j])?
                }
                ColumnKindTag::Categorical => {
                    if raw.is_empty() {
                        return Err(DataError::EmptyInput {
                            reason: format!(
                                "empty field at row {}, column '{}'",
                                encoded_rows + 1,
                                &headers[j]
                            ),
                        });
                    }
                    categorical_encoders[j]
                        .as_mut()
                        .expect("categorical encoder")
                        .encode(raw) as f64
                }
            };
        }
        encoded_rows += 1;
    }
    if encoded_rows != total_rows {
        return Err(DataError::SchemaMismatch {
            reason: "data file changed while its schema was being discovered".to_string(),
        });
    }

    let mut levels = vec![Vec::<String>::new(); p];
    for (j, encoder) in categorical_encoders.into_iter().enumerate() {
        if let Some(encoder) = encoder {
            levels[j] = encoder.finish(values.column_mut(j), LevelOrder::Canonical);
        }
    }
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
        columns: headers
            .iter()
            .enumerate()
            .map(|(j, name)| SchemaColumn {
                name: name.clone(),
                kind: column_kinds[j],
                levels: std::mem::take(&mut levels[j]),
            })
            .collect(),
    };
    Ok(EncodedDataset {
        headers,
        values,
        schema,
        column_kinds,
    })
}

#[derive(Clone, Copy)]
struct DelimitedInferenceState {
    all_numeric: bool,
    all_binary: bool,
}

impl Default for DelimitedInferenceState {
    fn default() -> Self {
        Self {
            all_numeric: true,
            all_binary: true,
        }
    }
}

impl DelimitedInferenceState {
    fn observe(&mut self, raw: &str, row: usize, header: &str) -> Result<(), DataError> {
        if raw.is_empty() {
            return Err(DataError::EmptyInput {
                reason: format!("empty field at row {row}, column '{header}'"),
            });
        }
        match raw.parse::<f64>() {
            Ok(value) => {
                if !value.is_finite() {
                    return Err(DataError::InvalidValue {
                        reason: format!("non-finite value at row {row}, column '{header}'"),
                    });
                }
                if (value - 0.0).abs() >= 1e-12 && (value - 1.0).abs() >= 1e-12 {
                    self.all_binary = false;
                }
            }
            Err(_) => {
                self.all_numeric = false;
                self.all_binary = false;
            }
        }
        Ok(())
    }

    fn kind(self, force_categorical: bool) -> ColumnKindTag {
        if force_categorical || !self.all_numeric {
            ColumnKindTag::Categorical
        } else if self.all_binary {
            ColumnKindTag::Binary
        } else {
            ColumnKindTag::Continuous
        }
    }
}

fn parse_inferred_numeric_cell(raw: &str, row: usize, header: &str) -> Result<f64, DataError> {
    if raw.is_empty() {
        return Err(DataError::EmptyInput {
            reason: format!("empty field at row {row}, column '{header}'"),
        });
    }
    let value = raw
        .parse::<f64>()
        .map_err(|error| DataError::EncodingFailure {
            reason: format!(
                "failed to parse numeric value '{raw}' at row {row}, column '{header}': {error}"
            ),
        })?;
    if !value.is_finite() {
        return Err(DataError::InvalidValue {
            reason: format!("non-finite value at row {row}, column '{header}'"),
        });
    }
    Ok(value)
}

#[derive(Clone, Copy)]
enum LevelOrder {
    Encounter,
    Canonical,
}

/// Single-owner categorical encoder used by the file loaders.
///
/// The map owns each distinct label exactly once and assigns an encounter-order
/// code while rows stream into the final numeric matrix. Finalization consumes
/// the map, moving those same `String` allocations into the schema and, when a
/// canonical order is required, remapping only the numeric codes in place.
/// This avoids the former map + level-vector + canonicalization-clone triple
/// ownership, whose text payload became row-sized for high-cardinality factors.
#[derive(Default)]
struct CategoricalEncoder {
    encounter_codes: HashMap<String, usize>,
}

impl CategoricalEncoder {
    fn encode(&mut self, label: &str) -> usize {
        if let Some(&code) = self.encounter_codes.get(label) {
            return code;
        }
        let code = self.encounter_codes.len();
        self.encounter_codes.insert(label.to_owned(), code);
        code
    }

    fn finish(self, mut encoded: ArrayViewMut1<'_, f64>, order: LevelOrder) -> Vec<String> {
        match order {
            LevelOrder::Encounter => {
                let mut levels = std::iter::repeat_with(|| None)
                    .take(self.encounter_codes.len())
                    .collect::<Vec<Option<String>>>();
                for (level, old_code) in self.encounter_codes {
                    levels[old_code] = Some(level);
                }
                levels
                    .into_iter()
                    .map(|level| level.expect("encounter code must name one level"))
                    .collect()
            }
            LevelOrder::Canonical => {
                let mut levels_with_old_codes =
                    self.encounter_codes.into_iter().collect::<Vec<_>>();
                levels_with_old_codes
                    .sort_by(|(a, _), (b, _)| natural_level_cmp(a.as_str(), b.as_str()));
                let mut remap = vec![0usize; levels_with_old_codes.len()];
                for (new_code, (_, old_code)) in levels_with_old_codes.iter().enumerate() {
                    remap[*old_code] = new_code;
                }
                for code in encoded.iter_mut() {
                    *code = remap[*code as usize] as f64;
                }
                levels_with_old_codes
                    .into_iter()
                    .map(|(level, _)| level)
                    .collect()
            }
        }
    }
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
                        .map(|(idx, v)| (v.as_str(), idx as f64))
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

    // A complete supplied schema needs no discovery pass. Parse its rows once
    // in historical row-major error order into a growable row-major buffer;
    // `Array2::from_shape_vec` adopts that same allocation as the final matrix.
    // The two-pass path below remains only for projected columns absent from the
    // schema, whose kind genuinely has to be discovered before encoding.
    if needs_inference.iter().all(|needs| !needs) {
        let t_stream = std::time::Instant::now();
        let mut flat_values = Vec::<f64>::new();
        let mut total_rows = 0usize;
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
                flat_values.push(parse_cell_with_schema(
                    raw,
                    &col_meta[j],
                    total_rows,
                    &headers[j],
                    &unseen_policy,
                )?);
            }
        }
        if total_rows == 0 {
            return Err(DataError::EmptyInput {
                reason: "file has no rows".to_string(),
            });
        }
        let values = Array2::from_shape_vec((total_rows, p), flat_values).map_err(|error| {
            DataError::EncodingFailure {
                reason: format!("failed to assemble schema-guided delimited matrix: {error}"),
            }
        })?;
        let stream_ms = t_stream.elapsed().as_secs_f64() * 1000.0;
        if stream_ms > 100.0 {
            log::info!(
                "[DATA-LOAD] delim_schema_direct | n_rows={} | n_cols={} | {:.1}ms",
                total_rows,
                p,
                stream_ms
            );
        }
        let column_kinds = col_meta.iter().map(|meta| meta.kind).collect();
        let schema_out = DataSchema {
            columns: col_meta.into_iter().map(|meta| meta.schema_col).collect(),
        };
        return Ok(EncodedDataset {
            headers,
            values,
            schema: schema_out,
            column_kinds,
        });
    }

    // Pass 1 validates schema-bound cells, discovers the kinds of schema-less
    // columns, and counts rows. No cell payload survives this pass.
    let mut inference = vec![DelimitedInferenceState::default(); p];
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
            if needs_inference[j] {
                inference[j].observe(raw, total_rows, &headers[j])?;
            } else {
                parse_cell_with_schema(raw, &col_meta[j], total_rows, &headers[j], &unseen_policy)?;
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
    for j in 0..p {
        if needs_inference[j] {
            let kind = inference[j].kind(false);
            col_meta[j].kind = kind;
            col_meta[j].schema_col.kind = kind;
        }
    }
    let finalize_ms = t_finalize.elapsed().as_secs_f64() * 1000.0;
    if finalize_ms > 100.0 {
        log::info!(
            "[DATA-LOAD] delim_schema_finalize | n_cols={} | {:.1}ms",
            p,
            finalize_ms
        );
    }

    // Pass 2 parses directly into final storage. Only unique strings for newly
    // inferred categorical columns are retained until their in-place canonical
    // code remap.
    let t_assemble = std::time::Instant::now();
    let mut values = Array2::<f64>::zeros((total_rows, p));
    let mut inferred_encoders = (0..p)
        .map(|j| {
            (needs_inference[j] && matches!(col_meta[j].kind, ColumnKindTag::Categorical))
                .then(CategoricalEncoder::default)
        })
        .collect::<Vec<_>>();
    let mut encode_rdr = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(delimiter)
        .from_path(path)
        .map_err(|e| DataError::ParseError {
            reason: format!("failed to reopen '{}': {e}", path.display()),
        })?;
    encode_rdr.headers().map_err(|e| DataError::ParseError {
        reason: format!("failed to reread headers: {e}"),
    })?;
    let mut encoded_rows = 0usize;
    while encode_rdr
        .read_record(&mut record)
        .map_err(|e| DataError::ParseError {
            reason: format!("failed reading row: {e}"),
        })?
    {
        if record.len() != all_headers.len() {
            return Err(DataError::SchemaMismatch {
                reason: format!(
                    "row width mismatch at row {}: got {} fields, expected {}",
                    encoded_rows + 1,
                    record.len(),
                    all_headers.len()
                ),
            });
        }
        if encoded_rows >= total_rows {
            return Err(DataError::SchemaMismatch {
                reason: "data file changed while its schema was being discovered".to_string(),
            });
        }
        for j in 0..p {
            let raw = record.get(selected_indices[j]).unwrap().trim();
            values[[encoded_rows, j]] = if !needs_inference[j] {
                parse_cell_with_schema(
                    raw,
                    &col_meta[j],
                    encoded_rows + 1,
                    &headers[j],
                    &unseen_policy,
                )?
            } else {
                match col_meta[j].kind {
                    ColumnKindTag::Continuous | ColumnKindTag::Binary => {
                        parse_inferred_numeric_cell(raw, encoded_rows + 1, &headers[j])?
                    }
                    ColumnKindTag::Categorical => {
                        if raw.is_empty() {
                            return Err(DataError::EmptyInput {
                                reason: format!(
                                    "empty field at row {}, column '{}'",
                                    encoded_rows + 1,
                                    &headers[j]
                                ),
                            });
                        }
                        let encoder = inferred_encoders[j]
                            .as_mut()
                            .expect("inferred categorical encoder");
                        encoder.encode(raw) as f64
                    }
                }
            };
        }
        encoded_rows += 1;
    }
    if encoded_rows != total_rows {
        return Err(DataError::SchemaMismatch {
            reason: "data file changed while its schema was being discovered".to_string(),
        });
    }
    for (j, encoder) in inferred_encoders.into_iter().enumerate() {
        if let Some(encoder) = encoder {
            col_meta[j].schema_col.levels =
                encoder.finish(values.column_mut(j), LevelOrder::Canonical);
        }
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

    let column_kinds = col_meta.iter().map(|meta| meta.kind).collect();
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
    meta: &ColMeta<'_>,
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
struct ColMeta<'a> {
    kind: ColumnKindTag,
    level_map: Option<HashMap<&'a str, f64>>,
    schema_col: SchemaColumn,
}

// ---------------------------------------------------------------------------
// Parquet — columnar, zero StringRecord, schema from metadata
// ---------------------------------------------------------------------------

/// True iff an Arrow column should be treated as a string/categorical column.
///
/// Dictionary encoding is a *storage* detail, not a semantic type: pyarrow
/// dictionary-encodes low-cardinality columns by default, including numeric
/// ones (integer factor levels, small enums stored as ints). A
/// `Dictionary(K, V)` column is categorical iff its *value* type `V` is a
/// string type; `Dictionary(_, Int*/UInt*/Float*/Bool)` is numeric. We recurse
/// through the value type so nested dictionaries resolve to their leaf type.
fn arrow_field_is_string(dt: &arrow::datatypes::DataType) -> bool {
    use arrow::datatypes::DataType;
    match dt {
        DataType::Utf8 | DataType::LargeUtf8 => true,
        DataType::Dictionary(_, value_type) => arrow_field_is_string(value_type),
        _ => false,
    }
}

fn write_arrow_numeric_values(
    values: impl IntoIterator<Item = f64>,
    base_row: usize,
    header: &str,
    mut output: ArrayViewMut1<'_, f64>,
    categorical_encoder: Option<&mut CategoricalEncoder>,
    all_binary: &mut bool,
) -> Result<(), DataError> {
    match categorical_encoder {
        Some(encoder) => {
            for (batch_row, value) in values.into_iter().enumerate() {
                if !value.is_finite() {
                    return Err(DataError::InvalidValue {
                        reason: format!(
                            "non-finite value at row {}, column '{}'",
                            base_row + batch_row + 1,
                            header
                        ),
                    });
                }
                output[batch_row] = encoder.encode(&value.to_string()) as f64;
            }
        }
        None => {
            for (batch_row, value) in values.into_iter().enumerate() {
                if !value.is_finite() {
                    return Err(DataError::InvalidValue {
                        reason: format!(
                            "non-finite value at row {}, column '{}'",
                            base_row + batch_row + 1,
                            header
                        ),
                    });
                }
                if (value - 0.0).abs() >= 1e-12 && (value - 1.0).abs() >= 1e-12 {
                    *all_binary = false;
                }
                output[batch_row] = value;
            }
        }
    }
    Ok(())
}

fn reject_arrow_null_values(
    col: &dyn arrow::array::Array,
    base_row: usize,
    header: &str,
) -> Result<(), DataError> {
    let Some(nulls) = col.logical_nulls() else {
        return Ok(());
    };
    if let Some(batch_row) = (0..col.len()).find(|&row| nulls.is_null(row)) {
        return Err(DataError::InvalidValue {
            reason: format!(
                "null value at row {}, column '{}'",
                base_row + batch_row + 1,
                header
            ),
        });
    }
    Ok(())
}

fn arrow_dictionary_string_value_at<'a, K>(
    col: &'a dyn arrow::array::Array,
    index: usize,
    logical_row: usize,
    header: &str,
) -> Result<&'a str, DataError>
where
    K: arrow::datatypes::ArrowDictionaryKeyType,
{
    use arrow::array::DictionaryArray;

    let dictionary = col
        .as_any()
        .downcast_ref::<DictionaryArray<K>>()
        .ok_or_else(|| DataError::EncodingFailure {
            reason: format!(
                "Arrow dictionary column '{}' did not match its declared key type",
                header
            ),
        })?;
    let value_index = dictionary
        .key(index)
        .ok_or_else(|| DataError::InvalidValue {
            reason: format!("null value at row {logical_row}, column '{header}'"),
        })?;
    if value_index >= dictionary.values().len() {
        return Err(DataError::EncodingFailure {
            reason: format!(
                "Arrow dictionary column '{}' has out-of-range key {} at row {}",
                header, value_index, logical_row
            ),
        });
    }
    arrow_string_value_at(
        dictionary.values().as_ref(),
        value_index,
        logical_row,
        header,
    )
}

/// Resolve a string or (possibly nested) dictionary-string value without
/// materializing a row-sized decoded string array.
fn arrow_string_value_at<'a>(
    col: &'a dyn arrow::array::Array,
    index: usize,
    logical_row: usize,
    header: &str,
) -> Result<&'a str, DataError> {
    use arrow::array::{LargeStringArray, StringArray};
    use arrow::datatypes::{
        DataType, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type,
        UInt64Type,
    };

    if index >= col.len() {
        return Err(DataError::EncodingFailure {
            reason: format!(
                "Arrow string column '{}' has out-of-range index {} at row {}",
                header, index, logical_row
            ),
        });
    }
    if col.is_null(index) {
        return Err(DataError::InvalidValue {
            reason: format!("null value at row {logical_row}, column '{header}'"),
        });
    }

    match col.data_type() {
        DataType::Utf8 => col
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|array| array.value(index))
            .ok_or_else(|| DataError::EncodingFailure {
                reason: format!("Arrow column '{}' could not be read as Utf8", header),
            }),
        DataType::LargeUtf8 => col
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .map(|array| array.value(index))
            .ok_or_else(|| DataError::EncodingFailure {
                reason: format!("Arrow column '{}' could not be read as LargeUtf8", header),
            }),
        DataType::Dictionary(key_type, _) => match key_type.as_ref() {
            DataType::Int8 => {
                arrow_dictionary_string_value_at::<Int8Type>(col, index, logical_row, header)
            }
            DataType::Int16 => {
                arrow_dictionary_string_value_at::<Int16Type>(col, index, logical_row, header)
            }
            DataType::Int32 => {
                arrow_dictionary_string_value_at::<Int32Type>(col, index, logical_row, header)
            }
            DataType::Int64 => {
                arrow_dictionary_string_value_at::<Int64Type>(col, index, logical_row, header)
            }
            DataType::UInt8 => {
                arrow_dictionary_string_value_at::<UInt8Type>(col, index, logical_row, header)
            }
            DataType::UInt16 => {
                arrow_dictionary_string_value_at::<UInt16Type>(col, index, logical_row, header)
            }
            DataType::UInt32 => {
                arrow_dictionary_string_value_at::<UInt32Type>(col, index, logical_row, header)
            }
            DataType::UInt64 => {
                arrow_dictionary_string_value_at::<UInt64Type>(col, index, logical_row, header)
            }
            other => Err(DataError::InvalidValue {
                reason: format!(
                    "unsupported Arrow dictionary key type {:?} for column '{}'",
                    other, header
                ),
            }),
        },
        other => Err(DataError::InvalidValue {
            reason: format!(
                "unsupported Arrow string column type {:?} for column '{}'",
                other, header
            ),
        }),
    }
}

/// Decode one Arrow column directly into its final matrix column.
///
/// The Arrow record batch remains the bounded input buffer. No Rust
/// `Vec<f64>` or per-cell `Vec<String>` is materialized beside it: primitive
/// values are converted into the strided ndarray view, and categorical labels
/// are interned immediately into their single-owner encoder.
fn decode_arrow_batch_column_into(
    col: &dyn arrow::array::Array,
    base_row: usize,
    header: &str,
    is_string_col: bool,
    mut output: ArrayViewMut1<'_, f64>,
    mut categorical_encoder: Option<&mut CategoricalEncoder>,
    all_binary: &mut bool,
) -> Result<(), DataError> {
    use arrow::array::{
        BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
        UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    };
    use arrow::datatypes::DataType;

    let n_rows = output.len();
    if col.len() != n_rows {
        return Err(DataError::SchemaMismatch {
            reason: format!(
                "Arrow column '{}' has {} rows, but its record batch has {}",
                header,
                col.len(),
                n_rows
            ),
        });
    }
    reject_arrow_null_values(col, base_row, header)?;

    if is_string_col {
        let encoder =
            categorical_encoder
                .as_deref_mut()
                .ok_or_else(|| DataError::EncodingFailure {
                    reason: format!("categorical Arrow encoder missing for column '{header}'"),
                })?;
        for batch_row in 0..n_rows {
            let label = arrow_string_value_at(col, batch_row, base_row + batch_row + 1, header)?;
            output[batch_row] = encoder.encode(label) as f64;
        }
        return Ok(());
    }

    // Numeric-valued dictionary columns (pyarrow dictionary-encodes
    // low-cardinality numeric columns by default) are not directly a
    // primitive array. Decode them to their concrete value type so the normal
    // numeric arms below apply. `arrow_field_is_string` has already routed
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
    // Defensively validate the decoded primitive too: the cast itself must not
    // introduce a null while resolving dictionary keys.
    reject_arrow_null_values(col, base_row, header)?;

    macro_rules! write_primitive {
        ($array_type:ty, $convert:expr) => {{
            let array = col.as_any().downcast_ref::<$array_type>().unwrap();
            write_arrow_numeric_values(
                array.values().iter().copied().map($convert),
                base_row,
                header,
                output,
                categorical_encoder,
                all_binary,
            )
        }};
    }

    match col.data_type() {
        DataType::Float64 => write_primitive!(Float64Array, |value: f64| value),
        DataType::Float32 => write_primitive!(Float32Array, |value: f32| value as f64),
        DataType::Int64 => write_primitive!(Int64Array, |value: i64| value as f64),
        DataType::Int32 => write_primitive!(Int32Array, |value: i32| value as f64),
        DataType::Int16 => write_primitive!(Int16Array, |value: i16| value as f64),
        DataType::Int8 => write_primitive!(Int8Array, |value: i8| value as f64),
        DataType::UInt64 => write_primitive!(UInt64Array, |value: u64| value as f64),
        DataType::UInt32 => write_primitive!(UInt32Array, |value: u32| value as f64),
        DataType::UInt16 => write_primitive!(UInt16Array, |value: u16| value as f64),
        DataType::UInt8 => write_primitive!(UInt8Array, |value: u8| value as f64),
        DataType::Boolean => {
            let arr = col.as_any().downcast_ref::<BooleanArray>().unwrap();
            write_arrow_numeric_values(
                (0..n_rows).map(|i| if arr.value(i) { 1.0 } else { 0.0 }),
                base_row,
                header,
                output,
                categorical_encoder,
                all_binary,
            )
        }
        other => Err(DataError::InvalidValue {
            reason: format!(
                "unsupported Arrow column type {:?} for column '{}'",
                other, header
            ),
        }),
    }
}

/// Infer and encode an Arrow record-batch stream into gam's native dataset.
///
/// `headers` supplies the already-normalized public column names in record-
/// batch column order. The Arrow schema supplies only physical types: integer,
/// unsigned integer, floating-point, and boolean columns remain numeric, while
/// `Utf8`, `LargeUtf8`, and string-valued dictionaries become categorical.
/// Categorical labels are interned as batches stream and canonicalized with the
/// same natural ordering as the other inferred ingestion paths.
///
/// The final row-major `N x P` `f64` allocation is grown in place and then
/// moved into [`EncodedDataset::values`] without a second dense allocation.
/// Apart from Arrow's current input batch, only one owned string per distinct
/// categorical level is retained; no row-major string table is materialized.
pub fn encode_arrow_record_batch_reader_with_inferred_schema(
    reader: &mut dyn arrow::record_batch::RecordBatchReader,
    headers: Vec<String>,
) -> Result<EncodedDataset, DataError> {
    if headers.is_empty() {
        return Err(DataError::EmptyInput {
            reason: "Arrow table must have at least one header column".to_string(),
        });
    }

    let mut seen_headers = HashSet::<&str>::with_capacity(headers.len());
    for (column, header) in headers.iter().enumerate() {
        if header.trim().is_empty() {
            return Err(DataError::EmptyInput {
                reason: format!("Arrow header at column {} cannot be empty", column + 1),
            });
        }
        if !seen_headers.insert(header.as_str()) {
            return Err(DataError::SchemaMismatch {
                reason: format!("duplicate Arrow header '{}'", header),
            });
        }
    }

    let arrow_schema = reader.schema();
    let p = headers.len();
    if arrow_schema.fields().len() != p {
        return Err(DataError::SchemaMismatch {
            reason: format!(
                "Arrow schema has {} columns, but {} normalized headers were supplied",
                arrow_schema.fields().len(),
                p
            ),
        });
    }

    let is_string_col = arrow_schema
        .fields()
        .iter()
        .map(|field| arrow_field_is_string(field.data_type()))
        .collect::<Vec<_>>();
    let mut all_binary = vec![true; p];
    let mut categorical_encoders = is_string_col
        .iter()
        .map(|&is_string| is_string.then(CategoricalEncoder::default))
        .collect::<Vec<_>>();
    let mut encoded_values = Vec::<f64>::new();
    let mut rows_seen = 0usize;

    for batch_result in reader {
        let batch = batch_result.map_err(|error| DataError::ParseError {
            reason: format!("failed to read Arrow record batch: {error}"),
        })?;
        if batch.num_columns() != p {
            return Err(DataError::SchemaMismatch {
                reason: format!(
                    "Arrow record batch has {} columns, but {} normalized headers were supplied",
                    batch.num_columns(),
                    p
                ),
            });
        }
        for j in 0..p {
            let expected = arrow_schema.field(j).data_type();
            let actual = batch.column(j).data_type();
            if actual != expected {
                return Err(DataError::SchemaMismatch {
                    reason: format!(
                        "Arrow column '{}' changed type between schema and batch: expected {:?}, got {:?}",
                        headers[j], expected, actual
                    ),
                });
            }
        }

        let n_rows = batch.num_rows();
        let batch_values = n_rows
            .checked_mul(p)
            .ok_or_else(|| DataError::EncodingFailure {
                reason: "Arrow batch dimensions do not fit in memory address space".to_string(),
            })?;
        let next_len = encoded_values
            .len()
            .checked_add(batch_values)
            .ok_or_else(|| DataError::EncodingFailure {
                reason: "Arrow dataset dimensions do not fit in memory address space".to_string(),
            })?;
        encoded_values
            .try_reserve(batch_values)
            .map_err(|error| DataError::EncodingFailure {
                reason: format!("failed to reserve Arrow dataset storage: {error}"),
            })?;
        let batch_offset = encoded_values.len();
        encoded_values.resize(next_len, 0.0);
        let mut batch_output = ndarray::ArrayViewMut2::from_shape(
            (n_rows, p),
            &mut encoded_values[batch_offset..next_len],
        )
        .map_err(|error| DataError::EncodingFailure {
            reason: format!("failed to shape Arrow batch output: {error}"),
        })?;

        let decoded_columns = batch_output
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(categorical_encoders.par_iter_mut())
            .zip(all_binary.par_iter_mut())
            .enumerate()
            .map(|(j, ((output, encoder), column_all_binary))| {
                decode_arrow_batch_column_into(
                    batch.column(j).as_ref(),
                    rows_seen,
                    &headers[j],
                    is_string_col[j],
                    output,
                    encoder.as_mut(),
                    column_all_binary,
                )
            })
            .collect::<Vec<_>>();
        for decoded in decoded_columns {
            decoded?;
        }
        rows_seen = rows_seen
            .checked_add(n_rows)
            .ok_or_else(|| DataError::EncodingFailure {
                reason: "Arrow row count does not fit in memory address space".to_string(),
            })?;
    }

    if rows_seen == 0 {
        return Err(DataError::EmptyInput {
            reason: "Arrow table data cannot be empty".to_string(),
        });
    }

    let mut values = Array2::from_shape_vec((rows_seen, p), encoded_values).map_err(|error| {
        DataError::EncodingFailure {
            reason: format!("failed to shape encoded Arrow dataset: {error}"),
        }
    })?;
    let mut levels = vec![Vec::<String>::new(); p];
    for (j, encoder) in categorical_encoders.into_iter().enumerate() {
        if let Some(encoder) = encoder {
            levels[j] = encoder.finish(values.column_mut(j), LevelOrder::Canonical);
        }
    }

    let mut schema_columns = Vec::<SchemaColumn>::with_capacity(p);
    let mut column_kinds = Vec::<ColumnKindTag>::with_capacity(p);
    for (j, name) in headers.iter().enumerate() {
        let kind = if is_string_col[j] {
            ColumnKindTag::Categorical
        } else if all_binary[j] {
            ColumnKindTag::Binary
        } else {
            ColumnKindTag::Continuous
        };
        column_kinds.push(kind);
        schema_columns.push(SchemaColumn {
            name: name.clone(),
            kind,
            levels: std::mem::take(&mut levels[j]),
        });
    }

    Ok(EncodedDataset {
        headers,
        values,
        schema: DataSchema {
            columns: schema_columns,
        },
        column_kinds,
    })
}

fn load_parquet_inferred(
    path: &Path,
    requested_columns: &[String],
    categorical_roles: &HashSet<&str>,
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
    let total_rows =
        usize::try_from(builder.metadata().file_metadata().num_rows()).map_err(|_| {
            DataError::ParseError {
                reason: "parquet row count does not fit in memory address space".to_string(),
            }
        })?;
    if total_rows == 0 {
        return Err(DataError::EmptyInput {
            reason: "parquet file has no rows".to_string(),
        });
    }
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
    let is_string_col = selected_fields
        .iter()
        .map(|field| arrow_field_is_string(field.data_type()))
        .collect::<Vec<_>>();
    let forced_numeric_categorical = headers
        .iter()
        .enumerate()
        .map(|(j, header)| !is_string_col[j] && categorical_roles.contains(header.as_str()))
        .collect::<Vec<_>>();
    let mut values = Array2::<f64>::zeros((total_rows, p));
    let mut all_binary = vec![true; p];
    let mut categorical_encoders = (0..p)
        .map(|j| {
            (is_string_col[j] || forced_numeric_categorical[j]).then(CategoricalEncoder::default)
        })
        .collect::<Vec<_>>();
    let mut rows_seen = 0usize;
    for batch_result in reader {
        let batch = batch_result.map_err(|e| DataError::ParseError {
            reason: format!("failed to read parquet record batch: {e}"),
        })?;
        let n_rows = batch.num_rows();
        if rows_seen.saturating_add(n_rows) > total_rows {
            return Err(DataError::SchemaMismatch {
                reason: "parquet row count changed while reading record batches".to_string(),
            });
        }

        let decoded_columns = values
            .slice_mut(s![rows_seen..rows_seen + n_rows, ..])
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(categorical_encoders.par_iter_mut())
            .zip(all_binary.par_iter_mut())
            .enumerate()
            .map(|(j, ((output, encoder), column_all_binary))| {
                decode_arrow_batch_column_into(
                    batch.column(j).as_ref(),
                    rows_seen,
                    &headers[j],
                    is_string_col[j],
                    output,
                    encoder.as_mut(),
                    column_all_binary,
                )
            })
            .collect::<Vec<_>>();

        // Rayon preserves indexed collection order, so checking the compact
        // per-column result vector from left to right retains the loader's
        // historical lowest-column error precedence without retaining decoded
        // cell payloads.
        for decoded in decoded_columns {
            decoded?;
        }
        rows_seen += n_rows;
    }

    if rows_seen != total_rows {
        return Err(DataError::SchemaMismatch {
            reason: format!(
                "parquet metadata reports {total_rows} rows but record batches yielded {rows_seen}"
            ),
        });
    }
    let batches_ms = t_batches.elapsed().as_secs_f64() * 1000.0;
    if batches_ms > 100.0 {
        log::info!(
            "[DATA-LOAD] parquet_batches_decode | n_rows={} | n_cols={} | {:.1}ms",
            total_rows,
            p,
            batches_ms
        );
    }
    let t_schema = std::time::Instant::now();
    // Numeric factor roles use canonical sorted label order, matching the CSV
    // factor-by-construction path. String parquet columns retain their existing
    // encounter-order contract. Finalization consumes each label map, moving its
    // sole String allocation into the schema and remapping codes in place only
    // for the canonical case.
    let mut levels = vec![Vec::<String>::new(); p];
    for (j, encoder) in categorical_encoders.into_iter().enumerate() {
        if let Some(encoder) = encoder {
            let order = if forced_numeric_categorical[j] {
                LevelOrder::Canonical
            } else {
                LevelOrder::Encounter
            };
            levels[j] = encoder.finish(values.column_mut(j), order);
        }
    }
    let mut schema_cols = Vec::<SchemaColumn>::with_capacity(p);
    let mut column_kinds = Vec::<ColumnKindTag>::with_capacity(p);
    for j in 0..p {
        let kind = if is_string_col[j] || forced_numeric_categorical[j] {
            ColumnKindTag::Categorical
        } else if all_binary[j] {
            ColumnKindTag::Binary
        } else {
            ColumnKindTag::Continuous
        };
        column_kinds.push(kind);
        schema_cols.push(SchemaColumn {
            name: headers[j].clone(),
            kind,
            levels: std::mem::take(&mut levels[j]),
        });
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
    // No formula roles are threaded here: the saved schema already records each
    // column's categorical kind, and the re-encode pass below pins kinds to it.
    let inferred = load_parquet_inferred(path, requested_columns, &HashSet::new())?;
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
    // Canonical natural-sorted level order — see `infer_and_encode_column_major`. The
    // record-driven and column-major inference paths must produce byte-identical
    // schemas, so both sort the level set with the same natural comparator
    // (#1319).
    if matches!(kind, ColumnKindTag::Categorical) {
        sort_levels_canonical(&mut levels);
    }
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
    // Canonical level ordering: sort factor levels lexicographically rather than
    // recording them in first-appearance order. Every reference tool a gam user
    // comes from — R `factor()` (C-locale sort), pandas `Categorical`, sklearn
    // `LabelEncoder` — orders categorical levels canonically, and downstream
    // consumers key off that order: the multinomial driver lays out one output
    // probability column per level and takes the *last* level as the softmax
    // reference, so first-appearance order made the `(n, K)` prediction columns
    // depend on which class happened to appear first in the training rows (a
    // row-shuffle would permute the output) instead of on the class labels
    // (#1319). Sorting makes the encoding a deterministic function of the label
    // *set*, independent of row order, and matches the factor convention so
    // column `k` of a multinomial prediction is class `levels[k]`. Use natural
    // ordering so generated labels like g2 stay before g10.
    if matches!(kind, ColumnKindTag::Categorical) {
        sort_levels_canonical(&mut levels);
    }
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
    use arrow::array::ArrayRef;
    use arrow::datatypes::{Field, Schema};
    use arrow::error::ArrowError;
    use arrow::record_batch::{RecordBatch, RecordBatchIterator};
    use std::sync::Arc;

    fn encode_single_arrow_array(array: ArrayRef) -> Result<EncodedDataset, DataError> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "source",
            array.data_type().clone(),
            true,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).expect("record batch");
        let batches: Vec<Result<RecordBatch, ArrowError>> = vec![Ok(batch)];
        let mut reader = RecordBatchIterator::new(batches, schema);
        encode_arrow_record_batch_reader_with_inferred_schema(
            &mut reader,
            vec!["normalized".to_string()],
        )
    }

    #[test]
    fn arrow_reader_streams_typed_columns_in_supplied_order() {
        use arrow::array::{
            Array, BooleanArray, DictionaryArray, Float32Array, Int8Array, Int32Array, Int64Array,
            LargeStringArray, StringArray,
        };
        use arrow::datatypes::Int8Type;

        let string_dictionary_1: DictionaryArray<Int8Type> =
            vec!["beta", "alpha"].into_iter().collect();
        let string_dictionary_2: DictionaryArray<Int8Type> = vec!["beta"].into_iter().collect();
        let numeric_dictionary_1 = DictionaryArray::<Int8Type>::new(
            Int8Array::from(vec![0, 1]),
            Arc::new(Int64Array::from(vec![5, 7])),
        );
        let numeric_dictionary_2 = DictionaryArray::<Int8Type>::new(
            Int8Array::from(vec![0]),
            Arc::new(Int64Array::from(vec![5])),
        );

        let schema = Arc::new(Schema::new(vec![
            Field::new("source_float", arrow::datatypes::DataType::Float32, false),
            Field::new("source_integer", arrow::datatypes::DataType::Int32, false),
            Field::new("source_flag", arrow::datatypes::DataType::Boolean, false),
            Field::new("source_utf8", arrow::datatypes::DataType::Utf8, false),
            Field::new(
                "source_large_utf8",
                arrow::datatypes::DataType::LargeUtf8,
                false,
            ),
            Field::new(
                "source_dictionary_string",
                string_dictionary_1.data_type().clone(),
                false,
            ),
            Field::new(
                "source_dictionary_number",
                numeric_dictionary_1.data_type().clone(),
                false,
            ),
        ]));
        let batch_1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float32Array::from(vec![1.5, 2.5])) as ArrayRef,
                Arc::new(Int32Array::from(vec![0, 1])),
                Arc::new(BooleanArray::from(vec![true, false])),
                Arc::new(StringArray::from(vec!["item10", "item2"])),
                Arc::new(LargeStringArray::from(vec!["z", "a"])),
                Arc::new(string_dictionary_1),
                Arc::new(numeric_dictionary_1),
            ],
        )
        .expect("first record batch");
        let batch_2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float32Array::from(vec![-4.0])) as ArrayRef,
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(BooleanArray::from(vec![true])),
                Arc::new(StringArray::from(vec!["item1"])),
                Arc::new(LargeStringArray::from(vec!["z"])),
                Arc::new(string_dictionary_2),
                Arc::new(numeric_dictionary_2),
            ],
        )
        .expect("second record batch");
        let batches: Vec<Result<RecordBatch, ArrowError>> = vec![Ok(batch_1), Ok(batch_2)];
        let mut reader = RecordBatchIterator::new(batches, schema);
        let headers = [
            "float",
            "integer",
            "flag",
            "utf8",
            "large_utf8",
            "dictionary_string",
            "dictionary_number",
        ]
        .map(str::to_string)
        .to_vec();

        let dataset =
            encode_arrow_record_batch_reader_with_inferred_schema(&mut reader, headers.clone())
                .expect("Arrow stream should encode");

        assert_eq!(dataset.headers, headers);
        assert_eq!(
            dataset.column_kinds,
            vec![
                ColumnKindTag::Continuous,
                ColumnKindTag::Binary,
                ColumnKindTag::Binary,
                ColumnKindTag::Categorical,
                ColumnKindTag::Categorical,
                ColumnKindTag::Categorical,
                ColumnKindTag::Continuous,
            ]
        );
        assert_eq!(
            dataset.values,
            ndarray::arr2(&[
                [1.5, 0.0, 1.0, 2.0, 1.0, 1.0, 5.0],
                [2.5, 1.0, 0.0, 1.0, 0.0, 0.0, 7.0],
                [-4.0, 1.0, 1.0, 0.0, 1.0, 1.0, 5.0],
            ])
        );
        assert_eq!(
            dataset.schema.columns[3].levels,
            vec!["item1", "item2", "item10"]
        );
        assert_eq!(dataset.schema.columns[4].levels, vec!["a", "z"]);
        assert_eq!(dataset.schema.columns[5].levels, vec!["alpha", "beta"]);
        assert!(
            dataset
                .schema
                .columns
                .iter()
                .zip(dataset.headers.iter())
                .all(|(column, header)| column.name == *header)
        );
    }

    #[test]
    fn arrow_reader_rejects_empty_duplicate_and_mismatched_headers() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "source",
            arrow::datatypes::DataType::Int32,
            false,
        )]));

        let mut empty_name_reader = RecordBatchIterator::new(
            Vec::<Result<RecordBatch, ArrowError>>::new(),
            schema.clone(),
        );
        let empty_name = encode_arrow_record_batch_reader_with_inferred_schema(
            &mut empty_name_reader,
            vec!["  ".to_string()],
        )
        .expect_err("blank header should fail");
        assert!(matches!(empty_name, DataError::EmptyInput { .. }));

        let mut duplicate_reader = RecordBatchIterator::new(
            Vec::<Result<RecordBatch, ArrowError>>::new(),
            Arc::new(Schema::new(vec![
                Field::new("a", arrow::datatypes::DataType::Int32, false),
                Field::new("b", arrow::datatypes::DataType::Int32, false),
            ])),
        );
        let duplicate = encode_arrow_record_batch_reader_with_inferred_schema(
            &mut duplicate_reader,
            vec!["x".to_string(), "x".to_string()],
        )
        .expect_err("duplicate header should fail");
        assert!(matches!(duplicate, DataError::SchemaMismatch { .. }));

        let mut mismatch_reader =
            RecordBatchIterator::new(Vec::<Result<RecordBatch, ArrowError>>::new(), schema);
        let mismatch = encode_arrow_record_batch_reader_with_inferred_schema(
            &mut mismatch_reader,
            vec!["x".to_string(), "y".to_string()],
        )
        .expect_err("header count mismatch should fail");
        assert!(matches!(mismatch, DataError::SchemaMismatch { .. }));
    }

    #[test]
    fn arrow_reader_reports_typed_null_nonfinite_and_unsupported_errors() {
        use arrow::array::{Date32Array, DictionaryArray, Float64Array, Int8Array, StringArray};
        use arrow::datatypes::Int8Type;

        let null_numeric =
            encode_single_arrow_array(Arc::new(Float64Array::from(vec![Some(1.0), None])))
                .expect_err("numeric null should fail");
        assert!(matches!(&null_numeric, DataError::InvalidValue { .. }));
        assert!(null_numeric.to_string().contains("null value at row 2"));

        let null_dictionary = DictionaryArray::<Int8Type>::new(
            Int8Array::from(vec![0, 1]),
            Arc::new(StringArray::from(vec![Some("present"), None])),
        );
        let logical_null = encode_single_arrow_array(Arc::new(null_dictionary))
            .expect_err("null dictionary value should fail");
        assert!(matches!(&logical_null, DataError::InvalidValue { .. }));
        assert!(logical_null.to_string().contains("null value at row 2"));

        let nonfinite = encode_single_arrow_array(Arc::new(Float64Array::from(vec![f64::NAN])))
            .expect_err("NaN should fail");
        assert!(matches!(&nonfinite, DataError::InvalidValue { .. }));
        assert!(nonfinite.to_string().contains("non-finite value"));

        let unsupported = encode_single_arrow_array(Arc::new(Date32Array::from(vec![1])))
            .expect_err("date column should fail");
        assert!(matches!(&unsupported, DataError::InvalidValue { .. }));
        assert!(
            unsupported
                .to_string()
                .contains("unsupported Arrow column type")
        );
    }

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
    fn categorical_encoder_consumes_labels_and_remaps_canonically() {
        use ndarray::Array1;

        let mut encoder = CategoricalEncoder::default();
        let mut encoded = Array1::from_vec(
            ["item10", "item2", "item1", "item2"]
                .into_iter()
                .map(|label| encoder.encode(label) as f64)
                .collect(),
        );

        let levels = encoder.finish(encoded.view_mut(), LevelOrder::Canonical);

        assert_eq!(levels, vec!["item1", "item2", "item10"]);
        assert_eq!(encoded.to_vec(), vec![2.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn complete_delimited_schema_encodes_projected_rows_directly() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("schema_direct.csv");
        std::fs::write(
            &path,
            "y,group,flag,unused\n1.5,b,0,first\n2.5,a,1,second\n",
        )
        .expect("write csv");
        let schema = DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "group".to_string(),
                    kind: ColumnKindTag::Categorical,
                    levels: vec!["a".to_string(), "b".to_string()],
                },
                SchemaColumn {
                    name: "flag".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: Vec::new(),
                },
                SchemaColumn {
                    name: "y".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: Vec::new(),
                },
            ],
        };

        let loaded = load_datasetwith_schema_projected(
            &path,
            &schema,
            UnseenCategoryPolicy::Error,
            &["y".to_string(), "group".to_string(), "flag".to_string()],
        )
        .expect("schema-guided projected load");

        assert_eq!(loaded.headers, vec!["y", "group", "flag"]);
        assert_eq!(loaded.values.row(0).to_vec(), vec![1.5, 1.0, 0.0]);
        assert_eq!(loaded.values.row(1).to_vec(), vec![2.5, 0.0, 1.0]);
        assert_eq!(
            loaded.column_kinds,
            vec![
                ColumnKindTag::Continuous,
                ColumnKindTag::Categorical,
                ColumnKindTag::Binary,
            ]
        );
    }

    #[test]
    fn direct_parquet_decoder_preserves_string_encounter_order() {
        use arrow::array::{Array, DictionaryArray};
        use arrow::datatypes::Int8Type;
        use ndarray::Array1;

        let dictionary: DictionaryArray<Int8Type> =
            vec!["beta", "alpha", "beta"].into_iter().collect();
        let mut encoded = Array1::<f64>::zeros(dictionary.len());
        let mut encoder = CategoricalEncoder::default();
        let mut all_binary = true;

        decode_arrow_batch_column_into(
            &dictionary,
            0,
            "group",
            true,
            encoded.view_mut(),
            Some(&mut encoder),
            &mut all_binary,
        )
        .expect("dictionary strings decode directly");
        let levels = encoder.finish(encoded.view_mut(), LevelOrder::Encounter);

        assert_eq!(levels, vec!["beta", "alpha"]);
        assert_eq!(encoded.to_vec(), vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn numeric_valued_dictionary_column_classifies_and_decodes_as_numeric() {
        // Regression for #1162: pyarrow dictionary-encodes low-cardinality
        // *numeric* columns by default (e.g. `Dictionary(Int8, Int64)`).
        // Dictionary encoding is a storage detail, not a semantic type, so such
        // a column must stay numeric — both at classification time
        // (`arrow_field_is_string`) and at decode time
        // (`decode_arrow_batch_column_into`). Previously the loader matched ALL
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
            !arrow_field_is_string(dict.data_type()),
            "Dictionary(Int8, Int64) must not be treated as a string column"
        );

        // A genuine string-valued dictionary still classifies as string.
        let str_dict: DictionaryArray<Int8Type> = vec!["a", "b", "a"].into_iter().collect();
        assert!(
            arrow_field_is_string(str_dict.data_type()),
            "Dictionary(Int8, Utf8) must remain a string column"
        );

        // Decoding the numeric dictionary with `is_string_col = false` must
        // resolve indices through the dictionary and write the underlying
        // numeric values directly into the destination view.
        let mut decoded = ndarray::Array1::<f64>::zeros(dict.len());
        let mut all_binary = true;
        decode_arrow_batch_column_into(
            &dict,
            0,
            "x",
            false,
            decoded.view_mut(),
            None,
            &mut all_binary,
        )
        .expect("numeric dictionary column should decode as numeric");
        assert_eq!(decoded.to_vec(), vec![5.0, 7.0, 5.0, 7.0, 5.0]);
        assert!(!all_binary);

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
        let inferred =
            load_parquet_inferred(&path, &[], &HashSet::new()).expect("inferred parquet load");
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

    // -----------------------------------------------------------------------
    // strip_categorical_sentinel
    // -----------------------------------------------------------------------

    #[test]
    fn sentinel_strip_present_returns_rest_and_true() {
        let marked = format!("{}{}", CATEGORICAL_CELL_SENTINEL, "hello");
        let (rest, found) = strip_categorical_sentinel(&marked);
        assert_eq!(rest, "hello");
        assert!(found);
    }

    #[test]
    fn sentinel_strip_absent_returns_original_and_false() {
        let (rest, found) = strip_categorical_sentinel("hello");
        assert_eq!(rest, "hello");
        assert!(!found);
    }

    #[test]
    fn sentinel_strip_empty_string_returns_empty_and_false() {
        let (rest, found) = strip_categorical_sentinel("");
        assert_eq!(rest, "");
        assert!(!found);
    }

    #[test]
    fn sentinel_strip_only_sentinel_returns_empty_and_true() {
        let marked = CATEGORICAL_CELL_SENTINEL.to_string();
        let (rest, found) = strip_categorical_sentinel(&marked);
        assert_eq!(rest, "");
        assert!(found);
    }

    // -----------------------------------------------------------------------
    // EncodedDataset::feature_ranges
    // -----------------------------------------------------------------------

    #[test]
    fn feature_ranges_two_columns() {
        let values = ndarray::arr2(&[[1.0_f64, 10.0], [3.0, 20.0], [2.0, 15.0]]);
        let ds = EncodedDataset {
            headers: vec!["a".to_string(), "b".to_string()],
            values,
            schema: DataSchema { columns: vec![] },
            column_kinds: vec![ColumnKindTag::Continuous, ColumnKindTag::Continuous],
        };
        let ranges = ds.feature_ranges();
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0], (1.0, 3.0));
        assert_eq!(ranges[1], (10.0, 20.0));
    }

    #[test]
    fn feature_ranges_single_row_min_equals_max() {
        let values = ndarray::arr2(&[[5.0_f64, -3.0]]);
        let ds = EncodedDataset {
            headers: vec!["x".to_string(), "y".to_string()],
            values,
            schema: DataSchema { columns: vec![] },
            column_kinds: vec![ColumnKindTag::Continuous, ColumnKindTag::Continuous],
        };
        let ranges = ds.feature_ranges();
        assert_eq!(ranges[0], (5.0, 5.0));
        assert_eq!(ranges[1], (-3.0, -3.0));
    }

    #[test]
    fn feature_ranges_all_nan_defaults_to_zero() {
        let values = ndarray::arr2(&[[f64::NAN], [f64::NAN]]);
        let ds = EncodedDataset {
            headers: vec!["x".to_string()],
            values,
            schema: DataSchema { columns: vec![] },
            column_kinds: vec![ColumnKindTag::Continuous],
        };
        let ranges = ds.feature_ranges();
        assert_eq!(ranges[0], (0.0, 0.0));
    }

    // -----------------------------------------------------------------------
    // EncodedDataset::column_map
    // -----------------------------------------------------------------------

    #[test]
    fn column_map_indexes_by_name() {
        let values = ndarray::arr2(&[[0.0_f64, 1.0], [2.0, 3.0]]);
        let ds = EncodedDataset {
            headers: vec!["alpha".to_string(), "beta".to_string()],
            values,
            schema: DataSchema { columns: vec![] },
            column_kinds: vec![ColumnKindTag::Continuous, ColumnKindTag::Continuous],
        };
        let map = ds.column_map();
        assert_eq!(map["alpha"], 0);
        assert_eq!(map["beta"], 1);
        assert_eq!(map.len(), 2);
    }

    // ── shared_prefix ─────────────────────────────────────────────────────────

    #[test]
    fn shared_prefix_identical_strings() {
        assert_eq!(shared_prefix("hello", "hello"), 5);
    }

    #[test]
    fn shared_prefix_no_common_prefix() {
        assert_eq!(shared_prefix("abc", "xyz"), 0);
    }

    #[test]
    fn shared_prefix_partial_match() {
        assert_eq!(shared_prefix("foobar", "foobaz"), 5);
    }

    #[test]
    fn shared_prefix_one_empty() {
        assert_eq!(shared_prefix("", "hello"), 0);
        assert_eq!(shared_prefix("hello", ""), 0);
    }

    #[test]
    fn shared_prefix_both_empty() {
        assert_eq!(shared_prefix("", ""), 0);
    }

    #[test]
    fn shared_prefix_shorter_string_is_prefix() {
        assert_eq!(shared_prefix("foo", "foobar"), 3);
    }

    // ── detect_format ─────────────────────────────────────────────────────────

    #[test]
    fn detect_format_csv() {
        let path = std::path::Path::new("data.csv");
        assert_eq!(detect_format(path).unwrap(), DataFormat::Csv);
    }

    #[test]
    fn detect_format_tsv() {
        assert_eq!(
            detect_format(std::path::Path::new("data.tsv")).unwrap(),
            DataFormat::Tsv
        );
        assert_eq!(
            detect_format(std::path::Path::new("data.txt")).unwrap(),
            DataFormat::Tsv
        );
        assert_eq!(
            detect_format(std::path::Path::new("data.tab")).unwrap(),
            DataFormat::Tsv
        );
    }

    #[test]
    fn detect_format_parquet() {
        assert_eq!(
            detect_format(std::path::Path::new("data.parquet")).unwrap(),
            DataFormat::Parquet
        );
        assert_eq!(
            detect_format(std::path::Path::new("data.pq")).unwrap(),
            DataFormat::Parquet
        );
        assert_eq!(
            detect_format(std::path::Path::new("data.pqt")).unwrap(),
            DataFormat::Parquet
        );
    }

    #[test]
    fn detect_format_uppercase_extension() {
        assert_eq!(
            detect_format(std::path::Path::new("data.CSV")).unwrap(),
            DataFormat::Csv
        );
    }

    #[test]
    fn detect_format_unknown_extension_is_error() {
        let err = detect_format(std::path::Path::new("data.json")).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("json") || msg.contains("unsupported"),
            "error should mention extension, got: {msg}"
        );
    }

    // ── strip_categorical_sentinel ────────────────────────────────────────────

    #[test]
    fn strip_categorical_sentinel_marked_cell() {
        // Sentinel is a single NUL character prefix
        let marked = "\u{0}hello";
        let (text, found) = strip_categorical_sentinel(marked);
        assert!(found);
        assert_eq!(text, "hello");
    }

    #[test]
    fn strip_categorical_sentinel_unmarked_cell() {
        let (text, found) = strip_categorical_sentinel("plain");
        assert!(!found);
        assert_eq!(text, "plain");
    }

    #[test]
    fn strip_categorical_sentinel_empty_string() {
        let (text, found) = strip_categorical_sentinel("");
        assert!(!found);
        assert_eq!(text, "");
    }

    #[test]
    fn strip_categorical_sentinel_only_sentinel() {
        let s = "\u{0}";
        let (text, found) = strip_categorical_sentinel(s);
        assert!(found);
        assert_eq!(text, "");
    }

    // ── projected_headers ─────────────────────────────────────────────────────

    #[test]
    fn projected_headers_selects_by_index() {
        let all = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let selected = projected_headers(&all, &[1, 3]);
        assert_eq!(selected, vec!["b".to_string(), "d".to_string()]);
    }

    #[test]
    fn projected_headers_empty_selection() {
        let all = vec!["x".to_string(), "y".to_string()];
        let selected = projected_headers(&all, &[]);
        assert!(selected.is_empty());
    }

    #[test]
    fn projected_headers_all_indices() {
        let all = vec!["p".to_string(), "q".to_string()];
        let selected = projected_headers(&all, &[0, 1]);
        assert_eq!(selected, all);
    }

    #[test]
    fn canonical_level_bits_collapses_signed_zero() {
        // The whole point of the helper: +0.0 and -0.0 name the same real
        // number (IEEE-754: 0.0 == -0.0) and MUST map to the same key, even
        // though their raw bit patterns differ (#2145 / #2146).
        let pos = 0.0_f64;
        let neg = -0.0_f64;
        assert_ne!(pos.to_bits(), neg.to_bits(), "precondition: raw bits differ");
        assert_eq!(pos, neg, "precondition: numerically equal");
        assert_eq!(canonical_level_bits(pos), canonical_level_bits(neg));
        assert_eq!(canonical_level_bits(neg), 0.0_f64.to_bits());
        // -0.0 reached via ordinary arithmetic is handled the same way.
        assert_eq!(canonical_level_bits(-1.0 * 0.0), 0.0_f64.to_bits());
        assert_eq!(canonical_level_bits(0.0 - 0.0), 0.0_f64.to_bits());
    }

    #[test]
    fn canonical_level_bits_is_bit_stable_on_ordinary_values() {
        // Every ordinary finite value keeps its raw key — the helper must not
        // perturb the identity of any genuine level.
        for &v in &[1.0_f64, -1.0, 2.5, -3.75, 1e300, -1e-300, f64::MIN, f64::MAX] {
            assert_eq!(canonical_level_bits(v), v.to_bits(), "value {v}");
        }
        // Distinct real values keep distinct keys.
        assert_ne!(canonical_level_bits(1.0), canonical_level_bits(2.0));
        assert_ne!(canonical_level_bits(0.0), canonical_level_bits(1.0));
        // Signed infinities are distinct (they are distinct real limits).
        assert_ne!(
            canonical_level_bits(f64::INFINITY),
            canonical_level_bits(f64::NEG_INFINITY)
        );
    }

    #[test]
    fn canonical_level_bits_collapses_nan_payloads() {
        // Every NaN encoding denotes "not a number"; they collapse to one key.
        let a = f64::NAN;
        let b = f64::from_bits(0x7ff8_0000_0000_0001); // a different NaN payload
        let c = -f64::NAN; // sign-bit-set NaN
        assert!(a.is_nan() && b.is_nan() && c.is_nan());
        assert_eq!(canonical_level_bits(a), canonical_level_bits(b));
        assert_eq!(canonical_level_bits(a), canonical_level_bits(c));
    }

    #[test]
    fn canonical_level_bits_is_idempotent() {
        // Re-canonicalizing an already-canonical key is a no-op — the property
        // the frozen-level resolution paths rely on.
        for &v in &[0.0_f64, -0.0, 1.0, -2.0, f64::NAN] {
            let once = canonical_level_bits(v);
            let twice = canonical_level_bits(f64::from_bits(once));
            assert_eq!(once, twice, "value {v}");
        }
    }
}

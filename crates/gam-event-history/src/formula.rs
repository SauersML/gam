//! The covariate/time formula of an event-history model: the right-hand side
//! of the ordinary gam formula language, resolved against the node data
//! matrix (the covariate table's columns followed by `time`).

use super::cohort::{EventHistoryCohort, EventHistoryError};
use gam_data::{ColumnKindTag, DataSchema, EncodedDataset, SchemaColumn};
use gam_terms::inference::formula_dsl::parse_formula;
use gam_terms::smooth::TermCollectionSpec;
use gam_terms::term_builder::build_termspec;
use ndarray::ArrayView2;

/// Name of the node-time column visible to the formula.
pub const TIME_COLUMN: &str = "time";

/// A row matrix over the cohort's covariate columns and `time` as an encoded
/// dataset: continuous columns stay continuous, a categorical covariate
/// carries its level labels so factor terms, `by=` gates and random effects
/// resolve against the labels the user supplied.
pub fn node_dataset(
    rows: ArrayView2<'_, f64>,
    cohort: &EventHistoryCohort,
) -> Result<EncodedDataset, EventHistoryError> {
    let covariate_names = &cohort.covariate_names;
    if covariate_names.len() + 1 != rows.ncols() {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "{} covariate names for a node data matrix with {} covariate columns",
                covariate_names.len(),
                rows.ncols() - 1
            ),
        });
    }
    if covariate_names.iter().any(|h| h == TIME_COLUMN) {
        return Err(EventHistoryError::InvalidInput {
            reason: format!("a covariate may not be named {TIME_COLUMN:?}"),
        });
    }
    let mut headers: Vec<String> = covariate_names.to_vec();
    headers.push(TIME_COLUMN.to_string());
    let mut column_kinds = Vec::with_capacity(headers.len());
    let mut columns = Vec::with_capacity(headers.len());
    for (j, name) in headers.iter().enumerate() {
        let levels = cohort.covariate_levels.get(j).cloned().unwrap_or_default();
        let kind = if levels.is_empty() {
            ColumnKindTag::Continuous
        } else {
            ColumnKindTag::Categorical
        };
        column_kinds.push(kind);
        columns.push(SchemaColumn {
            name: name.clone(),
            kind,
            levels,
        });
    }
    Ok(EncodedDataset {
        column_kinds,
        headers,
        values: rows.to_owned(),
        schema: DataSchema { columns },
    })
}

/// Resolve a formula right-hand side such as `x + s(time)` into the term
/// collection that every mark's log-intensity uses, against `rows`.
pub fn covariate_spec_from_formula(
    right_hand_side: &str,
    rows: ArrayView2<'_, f64>,
    cohort: &EventHistoryCohort,
) -> Result<TermCollectionSpec, EventHistoryError> {
    let rhs = right_hand_side.trim().trim_start_matches('~').trim();
    let formula = format!("events ~ {}", if rhs.is_empty() { "1" } else { rhs });
    let parsed = parse_formula(&formula).map_err(|error| EventHistoryError::InvalidInput {
        reason: format!("event-history formula {right_hand_side:?}: {error}"),
    })?;
    let dataset = node_dataset(rows, cohort)?;
    let col_map = dataset.column_map();
    let mut notes = Vec::new();
    build_termspec(
        &parsed.terms,
        &dataset,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .map_err(|error| EventHistoryError::InvalidInput {
        reason: format!("event-history formula {right_hand_side:?}: {}", String::from(error)),
    })
}

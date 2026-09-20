//! A fitted term's partial effect `f_t = X_t β_t` over a grid of its axes, with
//! pointwise intervals and a simultaneous band at one level.
//!
//! The grid comes from the saved term specification
//! (`gam_inference::partial_dependence`), the design from the model's own
//! mean-block builder at those rows, and the bands from
//! `gam_inference::effects::effect_bands` on the term's coefficient block and
//! the covariance the fit publishes. Every front end (Python, CLI) reads this
//! one function, so their numbers agree by construction.

use gam_data::EncodedDataset;
use gam_inference::effects::{EffectBands, effect_bands};
pub use gam_inference::partial_dependence::PartialDependenceGrid;
use gam_inference::partial_dependence::{
    HeldValue, PARTIAL_DEPENDENCE_SCALE, PartialDependenceInputs, PartialDependenceTable,
    partial_dependence_table,
};
use gam_models::inference::model::{
    FittedModel, PredictModelClass, append_deployment_extension_columns,
};
use gam_models::inference::saved_summary::{
    prediction_model_class_label, scan_introspection, scan_smooth_label,
};
use gam_models::survival::predict::{resolve_termspec_for_prediction, saved_fit_result};
use gam_solve::model_types::InferenceCovarianceMode;
use gam_terms::smooth::{
    TermCollectionPredictionDesign, TermCollectionSpec, build_term_collection_prediction_design,
    build_term_prediction_columns, term_collection_has_nonzero_anchor,
};
use ndarray::{Array2, s};
use serde::Serialize;

const NONZERO_ANCHOR_DESIGN_ERROR: &str = "design_matrix cannot represent a model with non-zero \
     smooth anchors as a single coefficient matrix; use Model.predict for the complete affine \
     predictor";

/// The frozen mean-block term specification a term-design diagnostic
/// evaluates, resolved against the dataset's columns.
fn standard_mean_termspec(
    model: &FittedModel,
    dataset: &EncodedDataset,
) -> Result<TermCollectionSpec, String> {
    // A scan-routed model never materializes a dense B-spline design — the
    // exact O(n) state-space smoother is the whole point — so there is no model
    // matrix to export (#1046).
    if let Some(scan) = scan_introspection(model)? {
        return Err(format!(
            "{} is fit by the exact O(n) state-space spline scan, which does not \
             build a finite coefficient-frame design; term-design diagnostics \
             are unavailable for this fitted model.",
            scan_smooth_label(&scan)
        ));
    }
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "design_matrix currently supports only standard GAM models; got '{}'. \
             For other classes use Model.predict / posterior.predict, which dispatch \
             through the saved-model predictor.",
            prediction_model_class_label(model)
        ));
    }
    if model.saved_link_wiggle()?.is_some() {
        return Err(
            "term-design diagnostics do not define an additive mean-block \
             decomposition for link-wiggle models; use design_matrix() for the \
             exact fitted affine predictor or Model.predict for response-scale output."
                .to_string(),
        );
    }
    resolve_termspec_for_prediction(
        &model.resolved_termspec,
        model.training_headers.as_ref(),
        &dataset.column_map(),
        "resolved_termspec",
    )
    .map_err(|err| err.to_string())
}

/// The undensified mean-block design behind [`standard_mean_design`], so a
/// caller that reads one term's columns never materializes the rest.
fn standard_mean_prediction_design(
    model: &FittedModel,
    dataset: &EncodedDataset,
) -> Result<TermCollectionPredictionDesign, String> {
    let spec = standard_mean_termspec(model, dataset)?;
    let design = build_term_collection_prediction_design(dataset.values.view(), &spec)
        .map_err(|err| format!("failed to build design matrix: {err}"))?;
    if design.affine_offset.iter().any(|value| *value != 0.0) {
        return Err(NONZERO_ANCHOR_DESIGN_ERROR.to_string());
    }
    Ok(design)
}

/// The full mean-block design of a standard GAM at `dataset`'s rows.
///
/// This is deliberately distinct from the public affine predictor design. A
/// link-wiggle's final fitted predictor uses the mean block as its row offset
/// and a LinkWiggle-frame matrix, so returning this internal matrix from the
/// public API was the architectural root cause of #2299.
pub fn standard_mean_design(
    model: &FittedModel,
    dataset: EncodedDataset,
) -> Result<Array2<f64>, String> {
    let design = standard_mean_prediction_design(model, &dataset)?;
    let dense = design
        .design
        .try_to_dense_by_chunks("design_matrix prediction design")?;
    append_deployment_extension_columns(
        model.payload(),
        dataset.values.view(),
        &dataset.column_map(),
        model.training_headers.as_ref(),
        dense,
    )
    .map_err(|err| err.to_string())
}

/// One term's partial effect on a grid, on the linear-predictor scale.
pub struct PartialEffect {
    pub term: String,
    /// The grid, its axes and levels, and what the curve is.
    pub table: PartialDependenceTable,
    /// The curve, its standard errors, and both bands, one entry per grid row.
    pub bands: EffectBands,
    /// The coefficient covariance the standard errors come from.
    pub covariance_source: InferenceCovarianceMode,
}

/// The partial effect of `term` on `grid`, with pointwise and simultaneous
/// bands at `level`.
///
/// The curve is `X_t β_t`, where `X_t` is the term's columns of the model's
/// design at the grid rows. That design already carries the identifiability
/// constraint the fit absorbed into the term's basis, so the curve is the
/// centred effect the fit estimated and its covariance is the term block
/// `V_tt` of the covariance the fit publishes: smoothing-parameter corrected
/// when the fit carries it, conditional otherwise, and named in the result.
pub fn partial_effect(
    model: &FittedModel,
    term: &str,
    grid: PartialDependenceGrid,
    level: f64,
) -> Result<PartialEffect, String> {
    let table = partial_effect_table(model, term, grid)?;
    let spec = standard_mean_termspec(model, &table.table)?;
    if term_collection_has_nonzero_anchor(&spec) {
        return Err(NONZERO_ANCHOR_DESIGN_ERROR.to_string());
    }
    let fit = saved_fit_result(model)?;
    let covariance_source = fit.published_covariance_mode();
    let covariance = match covariance_source {
        InferenceCovarianceMode::SmoothingCorrected => fit.beta_covariance_corrected(),
        InferenceCovarianceMode::Conditional => fit.beta_covariance(),
    }
    .ok_or_else(|| {
        "partial effects require a persisted coefficient covariance; refit before requesting \
         partial-effect intervals"
            .to_string()
    })?;
    // The term's coefficient range is a property of the layout, not of the
    // rows, so one grid row places it; the grid itself then realizes only the
    // term's own columns and the blocks they read.
    let rows = table.table.values.view();
    let layout =
        build_term_collection_prediction_design(rows.slice(s![..rows.nrows().min(1), ..]), &spec)
            .map_err(|err| format!("failed to build design matrix: {err}"))?
            .layout;
    let block = layout.term_range(term).ok_or_else(|| {
        format!(
            "partial effect: term {term:?} has no coefficient block; available: {:?}",
            layout.term_names()
        )
    })?;
    if block.is_empty() {
        return Err(format!(
            "partial effect: term {term:?} has no coefficients in the fitted model"
        ));
    }
    let p = fit.beta.len();
    if block.end > p || covariance.dim() != (p, p) {
        return Err(format!(
            "partial effect of {term:?}: columns {block:?} and the {:?} covariance do not fit \
             the {p} saved coefficients",
            covariance.dim()
        ));
    }
    let term_design = build_term_prediction_columns(rows, &spec, term)
        .map_err(|err| format!("failed to build design matrix: {err}"))?;
    if term_design.ncols() != block.len() {
        return Err(format!(
            "partial effect of {term:?}: the term realizes {} columns on the grid but spans \
             {block:?} in the model layout",
            term_design.ncols()
        ));
    }
    let bands = effect_bands(
        fit.beta.slice(s![block.clone()]),
        covariance.slice(s![block.clone(), block]),
        term_design.view(),
        level,
    )
    .map_err(|error| format!("partial effect of {term:?}: {error}"))?;
    Ok(PartialEffect {
        term: term.to_string(),
        table,
        bands,
        covariance_source,
    })
}

/// The grid table of `term`'s partial effect: its axes, their levels, the
/// evaluation points and the encoded rows the design is built at.
pub fn partial_effect_table(
    model: &FittedModel,
    term: &str,
    grid: PartialDependenceGrid,
) -> Result<PartialDependenceTable, String> {
    let payload = model.payload();
    let schema = payload
        .data_schema
        .as_ref()
        .ok_or_else(|| "partial effects require a saved model schema".to_string())?;
    let training_feature_ranges = payload
        .training_feature_ranges
        .as_deref()
        .ok_or_else(|| "partial effects require saved training feature ranges".to_string())?;
    let termspec = payload
        .resolved_termspec
        .as_ref()
        .ok_or_else(|| "partial effects require a saved resolved term specification".to_string())?;
    let training_headers = model
        .training_headers
        .as_deref()
        .ok_or_else(|| "partial effects require saved training headers".to_string())?;
    partial_dependence_table(
        PartialDependenceInputs {
            schema,
            training_headers,
            training_feature_ranges,
            termspec,
        },
        term,
        grid,
    )
}

/// A caller grid given as text, one field per axis: a number for a numeric axis
/// and a level label for a factor axis, as [`PartialEffect::to_csv`] writes them.
/// `header` names the axes in any order; the result follows the term's axis order.
pub fn encode_labelled_grid(
    table: &PartialDependenceTable,
    header: &[String],
    rows: &[Vec<String>],
) -> Result<Array2<f64>, String> {
    let mut sorted_header = header.to_vec();
    sorted_header.sort();
    let mut sorted_axes = table.axes.clone();
    sorted_axes.sort();
    if sorted_header != sorted_axes {
        return Err(format!(
            "partial effect grid columns {header:?} must be exactly the term's axes {:?}",
            table.axes
        ));
    }
    let source: Vec<usize> = table
        .axes
        .iter()
        .map(|axis| {
            header
                .iter()
                .position(|name| name == axis)
                .expect("checked above")
        })
        .collect();
    let mut grid = Array2::zeros((rows.len(), table.axes.len()));
    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != header.len() {
            return Err(format!(
                "partial effect grid row {} has {} fields; the header has {}",
                row_index + 1,
                row.len(),
                header.len()
            ));
        }
        for (axis, &column) in source.iter().enumerate() {
            let field = row[column].trim();
            grid[[row_index, axis]] = match &table.axis_levels[axis] {
                Some(levels) => levels
                    .labels
                    .iter()
                    .position(|label| label == field)
                    .map(|index| levels.values[index])
                    .ok_or_else(|| {
                        format!(
                            "partial effect grid row {}: {field:?} is not a level of {:?}; levels: {:?}",
                            row_index + 1,
                            table.axes[axis],
                            levels.labels
                        )
                    })?,
                None => field.parse::<f64>().map_err(|_| {
                    format!(
                        "partial effect grid row {}: {field:?} is not a number for axis {:?}",
                        row_index + 1,
                        table.axes[axis]
                    )
                })?,
            };
        }
    }
    Ok(grid)
}

/// The output columns of a partial effect, after its axes.
pub const PARTIAL_EFFECT_COLUMNS: [&str; 6] = [
    "fit",
    "se",
    "lower",
    "upper",
    "simultaneous_lower",
    "simultaneous_upper",
];

impl PartialEffect {
    /// The six per-row output series, in [`PARTIAL_EFFECT_COLUMNS`] order.
    pub fn series(&self) -> [&ndarray::Array1<f64>; 6] {
        let bands = &self.bands;
        [
            &bands.fit,
            &bands.se,
            &bands.lower,
            &bands.upper,
            &bands.simultaneous_lower,
            &bands.simultaneous_upper,
        ]
    }

    /// The grid value of row `row` on axis `axis`, as a factor axis's level label
    /// or a number.
    pub fn axis_cell(&self, row: usize, axis: usize) -> HeldValue {
        let value = self.table.grid[[row, axis]];
        match self.table.axis_levels[axis]
            .as_ref()
            .and_then(|levels| levels.label_of(value))
        {
            Some(label) => HeldValue::Level(label.to_string()),
            None => HeldValue::Number(value),
        }
    }

    /// The scale the curve is on.
    pub fn scale(&self) -> &'static str {
        PARTIAL_DEPENDENCE_SCALE
    }

    /// One CSV row per grid point: the axes (factor axes by level label), then
    /// [`PARTIAL_EFFECT_COLUMNS`].
    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        let header: Vec<String> = self
            .table
            .axes
            .iter()
            .map(|axis| csv_field(axis))
            .chain(PARTIAL_EFFECT_COLUMNS.iter().map(|name| name.to_string()))
            .collect();
        out.push_str(&header.join(","));
        out.push('\n');
        let series = self.series();
        for row in 0..self.table.grid.nrows() {
            let mut fields = Vec::with_capacity(self.table.axes.len() + series.len());
            for axis in 0..self.table.axes.len() {
                fields.push(match self.axis_cell(row, axis) {
                    HeldValue::Level(label) => csv_field(&label),
                    HeldValue::Number(number) => format_number(number),
                });
            }
            fields.extend(series.iter().map(|values| format_number(values[row])));
            out.push_str(&fields.join(","));
            out.push('\n');
        }
        out
    }
}

/// A grid cell or held value in a [`PartialEffectRecord`].
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum PartialEffectCell {
    Level(String),
    Number(f64),
}

impl From<&HeldValue> for PartialEffectCell {
    fn from(value: &HeldValue) -> Self {
        match value {
            HeldValue::Level(label) => Self::Level(label.clone()),
            HeldValue::Number(number) => Self::Number(*number),
        }
    }
}

/// A factor axis's levels in a [`PartialEffectRecord`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PartialEffectAxisLevels {
    pub values: Vec<f64>,
    pub labels: Vec<String>,
}

/// The serializable form of a [`PartialEffect`], which the CLI writes as JSON.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PartialEffectRecord {
    pub term: String,
    pub scale: String,
    pub quantity: String,
    pub contribution: String,
    pub covariance_source: String,
    pub level: f64,
    pub pointwise_critical: f64,
    pub simultaneous_critical: f64,
    pub simulations: usize,
    pub seed: u64,
    pub axes: Vec<String>,
    pub axis_levels: Vec<Option<PartialEffectAxisLevels>>,
    pub axis_values: Option<Vec<Vec<f64>>>,
    /// One row per evaluation point, factor axes by level code.
    pub grid: Vec<Vec<f64>>,
    /// The columns the term reads that every grid row holds fixed, as an object.
    #[serde(serialize_with = "serialize_pairs_as_map")]
    pub held: Vec<(String, PartialEffectCell)>,
    pub fit: Vec<f64>,
    pub se: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub simultaneous_lower: Vec<f64>,
    pub simultaneous_upper: Vec<f64>,
}

impl PartialEffect {
    pub fn record(&self) -> PartialEffectRecord {
        let table = &self.table;
        let bands = &self.bands;
        PartialEffectRecord {
            term: self.term.clone(),
            scale: self.scale().to_string(),
            quantity: table.quantity.as_str().to_string(),
            contribution: table.contribution(),
            covariance_source: self.covariance_source.as_str().to_string(),
            level: bands.level,
            pointwise_critical: bands.pointwise_critical,
            simultaneous_critical: bands.simultaneous_critical,
            simulations: bands.simulations,
            seed: bands.seed,
            axes: table.axes.clone(),
            axis_levels: table
                .axis_levels
                .iter()
                .map(|levels| {
                    levels.as_ref().map(|levels| PartialEffectAxisLevels {
                        values: levels.values.clone(),
                        labels: levels.labels.clone(),
                    })
                })
                .collect(),
            axis_values: table.axis_values.clone(),
            grid: table
                .grid
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect(),
            held: table
                .held
                .iter()
                .map(|(column, value)| (column.clone(), value.into()))
                .collect(),
            fit: bands.fit.to_vec(),
            se: bands.se.to_vec(),
            lower: bands.lower.to_vec(),
            upper: bands.upper.to_vec(),
            simultaneous_lower: bands.simultaneous_lower.to_vec(),
            simultaneous_upper: bands.simultaneous_upper.to_vec(),
        }
    }
}

fn serialize_pairs_as_map<S: serde::Serializer>(
    pairs: &[(String, PartialEffectCell)],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.collect_map(pairs.iter().map(|(key, value)| (key, value)))
}

/// Shortest round-tripping decimal form.
fn format_number(value: f64) -> String {
    format!("{value:?}")
}

fn csv_field(text: &str) -> String {
    if text.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

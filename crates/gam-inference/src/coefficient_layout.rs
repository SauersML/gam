//! Which model term owns each fitted coefficient of a saved fit (#3522).
//!
//! The column ranges are never re-derived from the term specification: they
//! are read from [`frozen_term_collection_layout`], which rebuilds the frozen
//! term collection with the same builder that produced the fitted design. The
//! global intercept column, when the model has one, is therefore wherever the
//! engine put it, and a model without one (`y ~ 0 + x + group(g)`, or an
//! anchored B-spline that absorbs the constant) starts its first term at
//! column 0. The CLI, the Python wrapper and `difference_smooth` all read the
//! same layout.

use gam_data::DataSchema;
use gam_models::inference::model::{FittedModelPayload, GroupMetadata, GroupMetadataValue};
use gam_terms::smooth::{
    SmoothBasisSpec, TermCollectionLayout, TermCollectionSpec, frozen_term_collection_layout,
};
use serde::Serialize;
use std::ops::Range;

/// A contiguous run of fitted coefficients owned by one model term.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TermBlock {
    pub name: String,
    /// `intercept`, `linear`, `random_effect`, or the smooth's basis family.
    pub kind: String,
    pub start: usize,
    pub end: usize,
}

/// What one fitted coefficient is.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CoefficientProvenance {
    pub index: usize,
    pub label: String,
    /// `intercept`, `linear`, `group`, `smooth`, or `global` for a coefficient
    /// that no term of the saved term collection owns (for example the
    /// coefficients of a second predictor block).
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub term: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<GroupMetadataValue>,
}

/// The term layout of a saved fit's coefficient vector.
#[derive(Clone, Debug, PartialEq)]
pub struct CoefficientLayout {
    /// The engine's column layout; `None` when the payload saves no resolved
    /// term collection, so no coefficient belongs to a named term.
    pub columns: Option<TermCollectionLayout>,
    /// One entry per fitted coefficient, in coefficient order.
    pub provenance: Vec<CoefficientProvenance>,
    /// The term blocks, sorted by first column.
    pub term_blocks: Vec<TermBlock>,
}

/// The coefficient layout of `payload`'s fit, whose coefficient vector has
/// `beta_len` entries.
pub fn coefficient_layout(
    payload: &FittedModelPayload,
    beta_len: usize,
) -> Result<CoefficientLayout, String> {
    let Some(spec) = payload.resolved_termspec.as_ref() else {
        return Ok(CoefficientLayout {
            columns: None,
            provenance: unowned_coefficients(beta_len),
            term_blocks: Vec::new(),
        });
    };
    let training_feature_ranges = payload.training_feature_ranges.as_deref().ok_or_else(|| {
        "the coefficient layout requires the saved training feature ranges".to_string()
    })?;
    term_collection_coefficient_layout(
        spec,
        training_feature_ranges,
        payload.data_schema.as_ref(),
        payload.group_metadata.as_ref(),
        beta_len,
    )
}

fn unowned_coefficients(beta_len: usize) -> Vec<CoefficientProvenance> {
    (0..beta_len)
        .map(|index| CoefficientProvenance {
            index,
            label: "__global__".to_string(),
            source: "global".to_string(),
            term: None,
            column: None,
            level: None,
            metadata: None,
        })
        .collect()
}

fn term_collection_coefficient_layout(
    spec: &TermCollectionSpec,
    training_feature_ranges: &[(f64, f64)],
    schema: Option<&DataSchema>,
    group_metadata: Option<&GroupMetadata>,
    beta_len: usize,
) -> Result<CoefficientLayout, String> {
    let columns = frozen_term_collection_layout(spec, training_feature_ranges)
        .map_err(|error| format!("coefficient layout: {error}"))?;
    if columns.ncols() > beta_len {
        return Err(format!(
            "the saved term collection spans {} coefficient columns but the fit has {beta_len}",
            columns.ncols()
        ));
    }
    let mut provenance = unowned_coefficients(beta_len);
    let term_blocks = label_coefficients(spec, &columns, schema, group_metadata, &mut provenance)?;
    Ok(CoefficientLayout {
        columns: Some(columns),
        provenance,
        term_blocks,
    })
}

fn label_coefficients(
    spec: &TermCollectionSpec,
    columns: &TermCollectionLayout,
    schema: Option<&DataSchema>,
    group_metadata: Option<&GroupMetadata>,
    provenance: &mut [CoefficientProvenance],
) -> Result<Vec<TermBlock>, String> {
    let mut blocks = Vec::new();
    let mut block = |name: &str, kind: &str, range: &Range<usize>| {
        blocks.push(TermBlock {
            name: name.to_string(),
            kind: kind.to_string(),
            start: range.start,
            end: range.end,
        })
    };

    if !columns.intercept_range.is_empty() {
        for entry in &mut provenance[columns.intercept_range.clone()] {
            entry.label = "intercept".to_string();
            entry.source = "intercept".to_string();
            entry.term = Some("intercept".to_string());
        }
        block("intercept", "intercept", &columns.intercept_range);
    }

    for (name, range) in &columns.linear_ranges {
        for (local, entry) in provenance[range.clone()].iter_mut().enumerate() {
            entry.label = if range.len() == 1 {
                name.clone()
            } else {
                format!("{name}[{local}]")
            };
            entry.source = "linear".to_string();
            entry.term = Some(name.clone());
            entry.column = Some(name.clone());
        }
        block(name, "linear", range);
    }

    for (name, range) in &columns.random_effect_ranges {
        let term = spec
            .random_effect_terms
            .iter()
            .find(|term| &term.name == name)
            .ok_or_else(|| {
                format!("the layout names random effect {name:?}, which the spec lacks")
            })?;
        let levels = term.frozen_levels.as_deref().unwrap_or(&[]);
        if levels.len() != range.len() {
            return Err(format!(
                "random effect {name:?} has {} frozen levels but owns {} coefficient columns",
                levels.len(),
                range.len()
            ));
        }
        let level_names = schema.and_then(|schema| schema.columns.get(term.feature_col));
        for (entry, &bits) in provenance[range.clone()].iter_mut().zip(levels) {
            let label = level_name(level_names.map(|column| column.levels.as_slice()), bits);
            entry.metadata = group_metadata
                .and_then(|metadata| metadata.get(&label))
                .cloned();
            entry.source = "group".to_string();
            entry.term = Some(name.clone());
            entry.column = Some(name.clone());
            entry.level = Some(label.clone());
            entry.label = label;
        }
        if !range.is_empty() {
            block(name, "random_effect", range);
        }
    }

    for (name, range) in &columns.smooth_ranges {
        let term = spec
            .smooth_terms
            .iter()
            .find(|term| &term.name == name)
            .ok_or_else(|| format!("the layout names smooth {name:?}, which the spec lacks"))?;
        for (local, entry) in provenance[range.clone()].iter_mut().enumerate() {
            entry.label = format!("{name}[{local}]");
            entry.source = "smooth".to_string();
            entry.term = Some(name.clone());
            entry.column = Some(name.clone());
        }
        block(name, smooth_basis_kind_label(&term.basis), range);
    }

    blocks.sort_by_key(|block| block.start);
    Ok(blocks)
}

/// The saved level name of a frozen level code, or the code itself when the
/// schema names no level for it.
fn level_name(levels: Option<&[String]>, bits: u64) -> String {
    let value = f64::from_bits(bits);
    // A level code is a stored integer, so it round-trips `usize` exactly; a
    // fractional, negative or out-of-range value does not (the cast truncates
    // or saturates) and names no level.
    let index = value as usize;
    levels
        .filter(|_| value.is_finite() && index as f64 == value)
        .and_then(|levels| levels.get(index))
        .cloned()
        .unwrap_or_else(|| value.to_string())
}

fn smooth_basis_kind_label(basis: &SmoothBasisSpec) -> &'static str {
    use gam_terms::smooth::SmoothBasisSpec as S;
    match basis {
        S::BSpline1D { .. } => "smooth_bspline1d",
        S::TensorBSpline { .. } => "tensor",
        S::ThinPlate { .. } => "thin_plate",
        S::Sphere { .. } => "sphere",
        S::ConstantCurvature { .. } => "constant_curvature",
        S::Matern { .. } => "matern",
        S::Duchon { .. } => "duchon",
        S::Pca { .. } => "pca",
        S::FactorSmooth { .. } => "factor_smooth",
        S::BySmooth { .. } => "by_smooth",
        S::ByVariable { .. } => "by_variable",
        S::FactorSumToZero { .. } => "factor_sum_to_zero",
        S::MeasureJet { .. } => "measurejet",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_data::{ColumnKindTag, SchemaColumn};
    use gam_terms::smooth::{LinearTermSpec, ModelLevel, RandomEffectTermSpec};

    fn spec(level: ModelLevel) -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: vec![LinearTermSpec {
                name: "x".to_string(),
                feature_col: 0,
                feature_cols: vec![0],
                categorical_levels: Vec::new(),
                double_penalty: false,
                coefficient_geometry: Default::default(),
                coefficient_min: None,
                coefficient_max: None,
                frozen_function_mass: None,
            }],
            smooth_terms: Vec::new(),
            random_effect_terms: vec![RandomEffectTermSpec {
                name: "g".to_string(),
                feature_col: 1,
                frozen_levels: Some(
                    (0..3)
                        .map(|code| gam_data::canonical_level_bits(code as f64))
                        .collect(),
                ),
                lenient_unseen: true,
            }],
            level,
        }
    }

    fn schema() -> DataSchema {
        DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "x".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: Vec::new(),
                },
                SchemaColumn {
                    name: "g".to_string(),
                    kind: ColumnKindTag::Categorical,
                    levels: vec!["a".to_string(), "b".to_string(), "c".to_string()],
                },
            ],
        }
    }

    fn blocks(layout: &CoefficientLayout) -> Vec<(&str, &str, usize, usize)> {
        layout
            .term_blocks
            .iter()
            .map(|block| {
                (
                    block.name.as_str(),
                    block.kind.as_str(),
                    block.start,
                    block.end,
                )
            })
            .collect()
    }

    /// #3522: `y ~ 0 + x + group(g)` has no intercept column, so `x` owns
    /// column 0 and `g` owns columns 1..4. The old hand-rolled layout put an
    /// intercept block at 0..1, `x` at 1..2 and `g` at 2..5, one column past
    /// the end of the four-coefficient fit.
    #[test]
    fn no_intercept_layout_starts_at_column_zero() {
        let ranges = [(0.0, 1.0), (0.0, 2.0)];
        let schema = schema();
        let layout = term_collection_coefficient_layout(
            &spec(ModelLevel::NoIntercept),
            &ranges,
            Some(&schema),
            None,
            4,
        )
        .expect("layout");
        assert_eq!(
            blocks(&layout),
            vec![("x", "linear", 0, 1), ("g", "random_effect", 1, 4)]
        );
        let labels: Vec<_> = layout
            .provenance
            .iter()
            .map(|entry| entry.label.as_str())
            .collect();
        assert_eq!(labels, vec!["x", "a", "b", "c"]);
        assert!(
            layout
                .columns
                .as_ref()
                .expect("columns")
                .intercept_range
                .is_empty()
        );

        let with = term_collection_coefficient_layout(
            &spec(ModelLevel::Intercept),
            &ranges,
            Some(&schema),
            None,
            5,
        )
        .expect("layout");
        assert_eq!(
            blocks(&with),
            vec![
                ("intercept", "intercept", 0, 1),
                ("x", "linear", 1, 2),
                ("g", "random_effect", 2, 5),
            ]
        );
    }

    /// Coefficients past the term collection (a second predictor block) stay
    /// unowned; a layout wider than the fit is a mismatch, not a truncation.
    #[test]
    fn layout_is_checked_against_the_fit_width() {
        let ranges = [(0.0, 1.0), (0.0, 2.0)];
        let spec = spec(ModelLevel::NoIntercept);
        let wider =
            term_collection_coefficient_layout(&spec, &ranges, None, None, 6).expect("layout");
        assert_eq!(wider.provenance[4].source, "global");
        assert_eq!(wider.provenance[5].source, "global");
        assert!(term_collection_coefficient_layout(&spec, &ranges, None, None, 3).is_err());
    }
}

//! Per-term partial-dependence grid, built from the saved term specification.
//!
//! A term's partial effect `f_t = X_t β_t` reads only the term's own axes and,
//! for a `by=` smooth, its by column. This module resolves a term block to its
//! saved spec entry, sweeps its axes over the training range or a caller grid,
//! pins a by column at the value the spec records, and fills every other schema
//! column with a valid default. The result is an encoded table over the saved
//! schema that front ends hand to the model's design builder unchanged, so no
//! term label is parsed and no cell is encoded a second time.

use gam_data::{ColumnKindTag, DataSchema, EncodedDataset};
use gam_terms::smooth::{
    ByVarKind, ByVariableSpec, SmoothBasisSpec, TermCollectionSpec, smooth_term_feature_cols,
};
use ndarray::Array2;
use std::collections::HashMap;

pub struct PartialDependenceInputs<'a> {
    pub schema: &'a DataSchema,
    /// The training column names, in the order the saved spec's column indices follow.
    pub training_headers: &'a [String],
    /// One `(min, max)` per saved schema column.
    pub training_feature_ranges: &'a [(f64, f64)],
    pub termspec: &'a TermCollectionSpec,
}

/// Where a term's axes are evaluated.
pub enum PartialDependenceGrid {
    /// `n_points` evenly spaced values over the one axis's training range.
    TrainingRange { n_points: usize },
    /// One row per evaluation point and one column per axis, in the term's axis order.
    Explicit(Array2<f64>),
}

pub struct PartialDependenceTable {
    /// The term's axis columns, in the order the grid's columns follow.
    pub axes: Vec<String>,
    /// The evaluation points, `(n, axes.len())`.
    pub grid: Array2<f64>,
    /// One encoded row per grid point over the saved schema.
    pub table: EncodedDataset,
    /// The columns the term reads that every grid row holds fixed.
    pub held: Vec<(String, HeldValue)>,
    /// What the curve evaluated on this table is.
    pub quantity: PartialDependenceQuantity,
}

impl PartialDependenceTable {
    /// How the curve enters the linear predictor, such as `f(x)` or `z * f(x)`.
    pub fn contribution(&self) -> String {
        let axes = self.axes.join(", ");
        match &self.quantity {
            PartialDependenceQuantity::TermContribution => format!("f({axes})"),
            PartialDependenceQuantity::CoefficientFunction { by } => format!("{by} * f({axes})"),
        }
    }
}

/// What a partial-dependence curve is. Either way it lives on the linear-predictor
/// scale, `X_t β_t` for the term's block, and is never a response-scale average.
#[derive(Clone, Debug, PartialEq)]
pub enum PartialDependenceQuantity {
    /// The term's contribution to the linear predictor.
    TermContribution,
    /// The coefficient function `f` of a numeric `by=` smooth, with the by column
    /// held at one. The term contributes `by · f` to the linear predictor.
    CoefficientFunction { by: String },
}

impl PartialDependenceQuantity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TermContribution => "term_contribution",
            Self::CoefficientFunction { .. } => "coefficient_function",
        }
    }
}

/// The scale every partial-dependence curve is reported on.
pub const PARTIAL_DEPENDENCE_SCALE: &str = "linear_predictor";

/// A value a grid holds a column at.
#[derive(Clone, Debug, PartialEq)]
pub enum HeldValue {
    /// A factor level, by its saved label.
    Level(String),
    Number(f64),
}

/// The columns a term's partial effect reads: the swept axes, and the columns
/// held at the value the spec records (a by level, a numeric by at one, or a
/// linear interaction's categorical gates). Indices are training columns.
struct ResolvedTerm {
    axis_cols: Vec<usize>,
    pins: Vec<(usize, f64)>,
    /// The training column of a numeric by, whose smooth is reported as its coefficient function.
    numeric_by: Option<usize>,
}

pub fn partial_dependence_table(
    inputs: PartialDependenceInputs<'_>,
    term: &str,
    grid: PartialDependenceGrid,
) -> Result<PartialDependenceTable, String> {
    let schema = inputs.schema;
    if schema.columns.len() != inputs.training_feature_ranges.len() {
        return Err(format!(
            "partial_dependence schema/range mismatch: {} columns but {} training ranges",
            schema.columns.len(),
            inputs.training_feature_ranges.len()
        ));
    }
    let schema_index: HashMap<&str, usize> = schema
        .columns
        .iter()
        .enumerate()
        .map(|(index, column)| (column.name.as_str(), index))
        .collect();
    let column_of = |training_col: usize| -> Result<usize, String> {
        let name = inputs.training_headers.get(training_col).ok_or_else(|| {
            format!(
                "partial_dependence: term {term:?} reads training column {training_col}, but the \
                 model saved {} training headers",
                inputs.training_headers.len()
            )
        })?;
        schema_index.get(name.as_str()).copied().ok_or_else(|| {
            format!(
                "partial_dependence: term {term:?} reads column {name:?}, which the saved schema \
                 does not carry"
            )
        })
    };

    let resolved = resolve_term(inputs.termspec, term)?;
    let axes = resolved
        .axis_cols
        .iter()
        .map(|&training_col| column_of(training_col))
        .collect::<Result<Vec<_>, _>>()?;
    let axis_names: Vec<String> = axes
        .iter()
        .map(|&column| schema.columns[column].name.clone())
        .collect();

    let mut template = Vec::with_capacity(schema.columns.len());
    for (index, column) in schema.columns.iter().enumerate() {
        template.push(match column.kind {
            // Level codes are indices into the saved levels, so code 0 is the first level.
            ColumnKindTag::Categorical => {
                if column.levels.is_empty() {
                    return Err(format!(
                        "partial_dependence: categorical column {:?} has no saved levels",
                        column.name
                    ));
                }
                0.0
            }
            ColumnKindTag::Binary => 0.0,
            ColumnKindTag::Continuous => {
                let (lo, hi) = inputs.training_feature_ranges[index];
                if !(lo.is_finite() && hi.is_finite()) {
                    return Err(format!(
                        "partial_dependence: training range for {:?} must be finite",
                        column.name
                    ));
                }
                0.5 * (lo + hi)
            }
        });
    }

    let grid = match grid {
        PartialDependenceGrid::Explicit(points) => {
            if points.ncols() != axes.len() {
                return Err(format!(
                    "partial_dependence: the grid has {} columns but term {term:?} has axes {axis_names:?}",
                    points.ncols()
                ));
            }
            if points.nrows() == 0 {
                return Err("partial_dependence: the grid must have at least one row".to_string());
            }
            if let Some(value) = points.iter().find(|value| !value.is_finite()) {
                return Err(format!(
                    "partial_dependence: grid values must be finite; got {value}"
                ));
            }
            points
        }
        PartialDependenceGrid::TrainingRange { n_points } => {
            let &[axis] = axes.as_slice() else {
                return Err(format!(
                    "partial_dependence: term {term:?} has axes {axis_names:?}, and a default grid \
                     sweeps exactly one axis; pass a grid with one column per axis"
                ));
            };
            if n_points < 2 {
                return Err(format!(
                    "partial_dependence: n_points must be at least 2; got {n_points}"
                ));
            }
            let column = &schema.columns[axis];
            if column.kind != ColumnKindTag::Continuous {
                return Err(format!(
                    "partial_dependence: axis {:?} is not continuous, so it has no training range \
                     to sweep; pass a grid",
                    column.name
                ));
            }
            let (lo, hi) = inputs.training_feature_ranges[axis];
            if !(lo.is_finite() && hi.is_finite() && lo < hi) {
                return Err(format!(
                    "partial_dependence: training range for {:?} must be finite and increasing; \
                     got ({lo:?}, {hi:?})",
                    column.name
                ));
            }
            let step = (hi - lo) / (n_points - 1) as f64;
            Array2::from_shape_fn((n_points, 1), |(row, _)| {
                if row + 1 == n_points {
                    hi
                } else {
                    lo + step * row as f64
                }
            })
        }
    };

    let mut values = Array2::from_shape_fn((grid.nrows(), template.len()), |(_, column)| {
        template[column]
    });
    let mut held = Vec::with_capacity(resolved.pins.len());
    for &(training_col, value) in &resolved.pins {
        let column = column_of(training_col)?;
        values.column_mut(column).fill(value);
        let schema_column = &schema.columns[column];
        let shown = match schema_column.kind {
            ColumnKindTag::Categorical => HeldValue::Level(
                schema_column
                    .levels
                    .get(value as usize)
                    .filter(|_| value >= 0.0 && value.fract() == 0.0)
                    .cloned()
                    .ok_or_else(|| {
                        format!(
                            "partial_dependence: term {term:?} holds factor {:?} at code {value}, \
                             which is not one of its {} saved levels",
                            schema_column.name,
                            schema_column.levels.len()
                        )
                    })?,
            ),
            ColumnKindTag::Binary | ColumnKindTag::Continuous => HeldValue::Number(value),
        };
        held.push((schema_column.name.clone(), shown));
    }
    let quantity = match resolved.numeric_by {
        Some(training_col) => PartialDependenceQuantity::CoefficientFunction {
            by: schema.columns[column_of(training_col)?].name.clone(),
        },
        None => PartialDependenceQuantity::TermContribution,
    };
    for (index, &column) in axes.iter().enumerate() {
        values.column_mut(column).assign(&grid.column(index));
    }
    let table = EncodedDataset {
        headers: schema.columns.iter().map(|column| column.name.clone()).collect(),
        values,
        schema: schema.clone(),
        column_kinds: schema.columns.iter().map(|column| column.kind).collect(),
    };
    Ok(PartialDependenceTable {
        axes: axis_names,
        grid,
        table,
        held,
        quantity,
    })
}

fn resolve_term(spec: &TermCollectionSpec, term: &str) -> Result<ResolvedTerm, String> {
    if let Some(linear) = spec.linear_terms.iter().find(|linear| linear.name == term) {
        let axis_cols = if linear.feature_cols.is_empty() {
            vec![linear.feature_col]
        } else {
            linear.feature_cols.clone()
        };
        return Ok(ResolvedTerm {
            axis_cols,
            pins: linear
                .categorical_levels
                .iter()
                .map(|&(column, level_bits)| (column, f64::from_bits(level_bits)))
                .collect(),
            numeric_by: None,
        });
    }
    if let Some(smooth) = spec.smooth_terms.iter().find(|smooth| smooth.name == term) {
        // A factor by-smooth is one block per level, and the block's spec records its
        // level, so the by column is held there. A numeric by multiplies the inner
        // smooth, and holding it at one reports the coefficient function itself.
        let pin = match &smooth.basis {
            SmoothBasisSpec::ByVariable { by_col, by, .. } => Some((
                *by_col,
                match by {
                    ByVariableSpec::Level { value_bits, .. } => f64::from_bits(*value_bits),
                    ByVariableSpec::Numeric => 1.0,
                },
            )),
            SmoothBasisSpec::BySmooth {
                by_kind: ByVarKind::Numeric { feature_col },
                ..
            } => Some((*feature_col, 1.0)),
            SmoothBasisSpec::BySmooth {
                by_kind: ByVarKind::Factor { .. },
                ..
            }
            | SmoothBasisSpec::FactorSumToZero { .. }
            | SmoothBasisSpec::FactorSmooth { .. } => {
                return Err(format!(
                    "partial_dependence: term {term:?} carries every level of its factor in one \
                     block, so its partial effect depends on a level the block does not record"
                ));
            }
            _ => None,
        };
        let numeric_by = match &smooth.basis {
            SmoothBasisSpec::ByVariable {
                by_col,
                by: ByVariableSpec::Numeric,
                ..
            } => Some(*by_col),
            SmoothBasisSpec::BySmooth {
                by_kind: ByVarKind::Numeric { feature_col },
                ..
            } => Some(*feature_col),
            _ => None,
        };
        let mut axis_cols = smooth_term_feature_cols(smooth);
        if let Some((by_col, _)) = pin {
            axis_cols.retain(|&column| column != by_col);
        }
        return Ok(ResolvedTerm {
            axis_cols,
            pins: pin.into_iter().collect(),
            numeric_by,
        });
    }
    let available: Vec<&str> = spec
        .linear_terms
        .iter()
        .map(|linear| linear.name.as_str())
        .chain(spec.smooth_terms.iter().map(|smooth| smooth.name.as_str()))
        .collect();
    Err(format!(
        "partial_dependence: {term:?} is not a linear or smooth term of this model; available: \
         {available:?}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_data::SchemaColumn;
    use gam_terms::basis::{
        BSplineBasisSpec, BSplineBoundaryConditions, BSplineIdentifiability, BSplineKnotSpec,
        OneDimensionalBoundary,
    };
    use gam_terms::smooth::{
        BySmoothKind, LinearCoefficientGeometry, LinearTermSpec, ShapeConstraint, SmoothTermSpec,
    };
    use ndarray::array;

    const RANGES: [(f64, f64); 3] = [(0.0, 2.0), (0.0, 1.0), (1.0, 3.0)];

    fn headers() -> Vec<String> {
        vec!["x".to_string(), "g".to_string(), "z".to_string()]
    }

    fn schema() -> DataSchema {
        let continuous = |name: &str| SchemaColumn {
            name: name.to_string(),
            kind: ColumnKindTag::Continuous,
            levels: Vec::new(),
        };
        DataSchema {
            columns: vec![
                continuous("x"),
                SchemaColumn {
                    name: "g".to_string(),
                    kind: ColumnKindTag::Categorical,
                    levels: vec!["a".to_string(), "b".to_string()],
                },
                continuous("z"),
            ],
        }
    }

    fn bspline(feature_col: usize) -> SmoothBasisSpec {
        SmoothBasisSpec::BSpline1D {
            feature_col,
            spec: BSplineBasisSpec {
                degree: 2,
                penalty_order: 1,
                knotspec: BSplineKnotSpec::Generate {
                    data_range: (0.0, 1.0),
                    num_internal_knots: 2,
                },
                double_penalty: false,
                identifiability: BSplineIdentifiability::None,
                boundary: OneDimensionalBoundary::Open,
                boundary_conditions: BSplineBoundaryConditions::default(),
            },
        }
    }

    fn smooth(name: &str, basis: SmoothBasisSpec) -> SmoothTermSpec {
        SmoothTermSpec {
            name: name.to_string(),
            basis,
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
            frozen_parametric_residualization: None,
        }
    }

    fn linear(name: &str, feature_cols: Vec<usize>, categorical_levels: Vec<(usize, u64)>) -> LinearTermSpec {
        LinearTermSpec {
            name: name.to_string(),
            feature_col: feature_cols[0],
            feature_cols,
            categorical_levels,
            double_penalty: false,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }
    }

    fn table(
        spec: &TermCollectionSpec,
        term: &str,
        grid: PartialDependenceGrid,
    ) -> Result<PartialDependenceTable, String> {
        let schema = schema();
        let headers = headers();
        partial_dependence_table(
            PartialDependenceInputs {
                schema: &schema,
                training_headers: &headers,
                training_feature_ranges: &RANGES,
                termspec: spec,
            },
            term,
            grid,
        )
    }

    #[test]
    fn a_by_level_block_holds_its_own_level_and_sweeps_its_axis() {
        let level_b = 1.0_f64.to_bits();
        let spec = TermCollectionSpec {
            linear_terms: Vec::new(),
            random_effect_terms: Vec::new(),
            smooth_terms: vec![smooth(
                "s(x):by=g[b]",
                SmoothBasisSpec::ByVariable {
                    inner: Box::new(bspline(0)),
                    by_col: 1,
                    kind: BySmoothKind::Level { level_bits: level_b },
                    by: ByVariableSpec::Level {
                        value_bits: level_b,
                        label: "b".to_string(),
                    },
                },
            )],
            level: Default::default(),
        };
        let pd = table(&spec, "s(x):by=g[b]", PartialDependenceGrid::TrainingRange { n_points: 5 })
            .expect("by-level table");
        assert_eq!(pd.axes, vec!["x".to_string()]);
        assert_eq!(pd.grid.column(0).to_vec(), vec![0.0, 0.5, 1.0, 1.5, 2.0]);
        assert_eq!(pd.table.headers, headers());
        assert_eq!(pd.held, vec![("g".to_string(), HeldValue::Level("b".to_string()))]);
        assert_eq!(pd.quantity, PartialDependenceQuantity::TermContribution);
        assert_eq!(pd.contribution(), "f(x)");
        assert_eq!(pd.table.values.column(0).to_vec(), pd.grid.column(0).to_vec());
        assert!(
            pd.table.values.column(1).iter().all(|code| *code == 1.0),
            "the block's own level, code 1 = \"b\""
        );
        assert!(
            pd.table.values.column(2).iter().all(|value| *value == 2.0),
            "an unread continuous column sits at its training midpoint"
        );
    }

    #[test]
    fn a_numeric_by_is_held_at_one() {
        let spec = TermCollectionSpec {
            linear_terms: Vec::new(),
            random_effect_terms: Vec::new(),
            smooth_terms: vec![smooth(
                "s(x, by=z)",
                SmoothBasisSpec::ByVariable {
                    inner: Box::new(bspline(0)),
                    by_col: 2,
                    kind: BySmoothKind::Numeric,
                    by: ByVariableSpec::Numeric,
                },
            )],
            level: Default::default(),
        };
        let pd = table(&spec, "s(x, by=z)", PartialDependenceGrid::TrainingRange { n_points: 3 })
            .expect("numeric by table");
        assert_eq!(pd.axes, vec!["x".to_string()]);
        assert!(pd.table.values.column(2).iter().all(|value| *value == 1.0));
        assert_eq!(
            pd.quantity,
            PartialDependenceQuantity::CoefficientFunction { by: "z".to_string() }
        );
        assert_eq!(pd.held, vec![("z".to_string(), HeldValue::Number(1.0))]);
        assert_eq!(pd.contribution(), "z * f(x)");
        assert!(
            pd.table.values.column(1).iter().all(|code| *code == 0.0),
            "an unread factor sits at its first saved level"
        );
    }

    #[test]
    fn a_linear_interaction_holds_its_gate_and_a_two_axis_term_needs_a_grid() {
        let spec = TermCollectionSpec {
            linear_terms: vec![
                linear("x:g[b]", vec![0], vec![(1, 1.0_f64.to_bits())]),
                linear("x:z", vec![0, 2], Vec::new()),
            ],
            random_effect_terms: Vec::new(),
            smooth_terms: Vec::new(),
            level: Default::default(),
        };
        let gated = table(&spec, "x:g[b]", PartialDependenceGrid::TrainingRange { n_points: 2 })
            .expect("gated linear table");
        assert!(gated.table.values.column(1).iter().all(|code| *code == 1.0));

        let refusal = table(&spec, "x:z", PartialDependenceGrid::TrainingRange { n_points: 4 })
            .err()
            .expect("a two-axis term refuses a default grid");
        assert!(refusal.contains("sweeps exactly one axis"), "got: {refusal}");

        let explicit = table(
            &spec,
            "x:z",
            PartialDependenceGrid::Explicit(array![[0.25, 1.5], [1.75, 2.5]]),
        )
        .expect("explicit two-axis grid");
        assert_eq!(explicit.axes, vec!["x".to_string(), "z".to_string()]);
        assert_eq!(explicit.table.values.column(0).to_vec(), vec![0.25, 1.75]);
        assert_eq!(explicit.table.values.column(2).to_vec(), vec![1.5, 2.5]);
    }

    #[test]
    fn a_block_spanning_every_level_and_an_unknown_term_are_refused_by_name() {
        let spec = TermCollectionSpec {
            linear_terms: vec![linear("z", vec![2], Vec::new())],
            random_effect_terms: Vec::new(),
            smooth_terms: vec![smooth(
                "s(x, g, bs=sz)",
                SmoothBasisSpec::FactorSumToZero {
                    inner: Box::new(bspline(0)),
                    by_col: 1,
                    levels: vec![0.0_f64.to_bits(), 1.0_f64.to_bits()],
                    frozen_global_orthogonality: None,
                },
            )],
            level: Default::default(),
        };
        let spanning = table(&spec, "s(x, g, bs=sz)", PartialDependenceGrid::TrainingRange { n_points: 3 })
            .err()
            .expect("a level-spanning block is refused");
        assert!(spanning.contains("carries every level"), "got: {spanning}");

        let unknown = table(&spec, "s(w)", PartialDependenceGrid::TrainingRange { n_points: 3 })
            .err()
            .expect("an unknown term is refused");
        assert!(
            unknown.contains("available") && unknown.contains("\"z\""),
            "got: {unknown}"
        );
    }
}

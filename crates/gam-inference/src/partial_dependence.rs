//! Per-term partial-dependence grid, built from the saved term specification.
//!
//! A term's partial effect `f_t = X_t β_t` reads only the term's own axes and,
//! for a `by=` smooth, its by column. This module resolves a term block to its
//! saved spec entry, sweeps its axes over the training range or a caller grid,
//! pins a by column at the value the spec records, and fills every other schema
//! column with a valid default. A factor the block reads for every level (a
//! bare factor term, or a factor smooth's grouping column) is an axis that
//! takes the block's saved levels. The result is an encoded table over the saved
//! schema that front ends hand to the model's design builder unchanged, so no
//! term label is parsed and no cell is encoded a second time.

use gam_data::{ColumnKindTag, DataSchema, EncodedDataset, SchemaColumn};
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
    /// The Cartesian product of `n_points` evenly spaced values over each
    /// numeric axis's training range and every level of each factor axis, with
    /// the last axis varying fastest.
    TrainingRange { n_points: usize },
    /// One row per evaluation point and one column per axis, in the term's axis
    /// order. A factor axis takes its levels' codes, as [`AxisLevels::values`] lists them.
    Explicit(Array2<f64>),
}

/// The values a factor axis can take, and their labels.
#[derive(Clone, Debug, PartialEq)]
pub struct AxisLevels {
    /// The encoded value of each level, as a grid column holds it.
    pub values: Vec<f64>,
    pub labels: Vec<String>,
}

impl AxisLevels {
    /// The label of the level encoded as `value`, if `value` is one of the levels.
    pub fn label_of(&self, value: f64) -> Option<&str> {
        self.values
            .iter()
            .position(|&level| level == value)
            .map(|index| self.labels[index].as_str())
    }
}

pub struct PartialDependenceTable {
    /// The term's axis columns, in the order the grid's columns follow.
    pub axes: Vec<String>,
    /// For each axis, its levels when it is a factor axis and `None` when it is numeric.
    pub axis_levels: Vec<Option<AxisLevels>>,
    /// For the default grid, the values each axis sweeps; the grid is their
    /// Cartesian product with the last axis varying fastest. `None` for a caller grid.
    pub axis_values: Option<Vec<Vec<f64>>>,
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
    /// Axes the block reads at a level set it records, as encoded values.
    axis_level_sets: Vec<(usize, Vec<f64>)>,
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

    let axis_levels = axes
        .iter()
        .zip(&resolved.axis_cols)
        .map(|(&column, training_col)| {
            let recorded = resolved
                .axis_level_sets
                .iter()
                .find(|(col, _)| col == training_col)
                .map(|(_, values)| values.clone());
            axis_levels_of(&schema.columns[column], recorded, term)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let (grid, axis_values) = match grid {
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
            for (index, levels) in axis_levels.iter().enumerate() {
                let Some(levels) = levels else { continue };
                if let Some(value) = points
                    .column(index)
                    .iter()
                    .find(|value| levels.label_of(**value).is_none())
                {
                    return Err(format!(
                        "partial_dependence: factor axis {:?} takes the level codes {:?} ({:?}); \
                         the grid holds {value}",
                        axis_names[index], levels.values, levels.labels
                    ));
                }
            }
            (points, None)
        }
        PartialDependenceGrid::TrainingRange { n_points } => {
            if n_points < 2 {
                return Err(format!(
                    "partial_dependence: n_points must be at least 2; got {n_points}"
                ));
            }
            let values = axes
                .iter()
                .zip(&axis_levels)
                .map(|(&axis, levels)| match levels {
                    Some(levels) => Ok(levels.values.clone()),
                    None => {
                        let (lo, hi) = inputs.training_feature_ranges[axis];
                        if !(lo.is_finite() && hi.is_finite() && lo < hi) {
                            return Err(format!(
                                "partial_dependence: training range for {:?} must be finite and \
                                 increasing; got ({lo:?}, {hi:?})",
                                schema.columns[axis].name
                            ));
                        }
                        let step = (hi - lo) / (n_points - 1) as f64;
                        Ok((0..n_points)
                            .map(|index| {
                                if index + 1 == n_points {
                                    hi
                                } else {
                                    lo + step * index as f64
                                }
                            })
                            .collect())
                    }
                })
                .collect::<Result<Vec<Vec<f64>>, String>>()?;
            (cartesian_product(&values), Some(values))
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
        axis_levels,
        axis_values,
        grid,
        table,
        held,
        quantity,
    })
}

/// The levels a factor axis takes: the level set the block records when it
/// records one, else every saved level of a categorical column, and `{0, 1}`
/// for a binary column. A continuous column the block does not level is numeric.
fn axis_levels_of(
    column: &SchemaColumn,
    recorded: Option<Vec<f64>>,
    term: &str,
) -> Result<Option<AxisLevels>, String> {
    let values = match (recorded, column.kind) {
        (Some(values), _) => values,
        (None, ColumnKindTag::Categorical) => (0..column.levels.len()).map(|code| code as f64).collect(),
        (None, ColumnKindTag::Binary) => vec![0.0, 1.0],
        (None, ColumnKindTag::Continuous) => return Ok(None),
    };
    if values.is_empty() {
        return Err(format!(
            "partial_dependence: term {term:?} reads factor {:?} at no levels",
            column.name
        ));
    }
    let labels = values
        .iter()
        .map(|&value| match column.kind {
            ColumnKindTag::Categorical => column
                .levels
                .get(value as usize)
                .filter(|_| value >= 0.0 && value.fract() == 0.0)
                .cloned()
                .ok_or_else(|| {
                    format!(
                        "partial_dependence: term {term:?} records level code {value} of factor \
                         {:?}, which is not one of its {} saved levels",
                        column.name,
                        column.levels.len()
                    )
                }),
            ColumnKindTag::Binary | ColumnKindTag::Continuous => Ok(format!("{value}")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(AxisLevels { values, labels }))
}

/// Rows of the Cartesian product of `values`, with the last axis varying fastest.
fn cartesian_product(values: &[Vec<f64>]) -> Array2<f64> {
    let rows: usize = values.iter().map(Vec::len).product();
    Array2::from_shape_fn((rows, values.len()), |(row, axis)| {
        let stride: usize = values[axis + 1..].iter().map(Vec::len).product();
        values[axis][(row / stride) % values[axis].len()]
    })
}

fn levels_from_bits(bits: &[u64]) -> Vec<f64> {
    bits.iter().map(|&bits| f64::from_bits(bits)).collect()
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
            axis_level_sets: Vec::new(),
            pins: linear
                .categorical_levels
                .iter()
                .map(|&(column, level_bits)| (column, f64::from_bits(level_bits)))
                .collect(),
            numeric_by: None,
        });
    }
    if let Some(effect) = spec
        .random_effect_terms
        .iter()
        .find(|effect| effect.name == term)
    {
        // One coefficient per level: the effect is a function of the level alone.
        let levels = effect.frozen_levels.as_deref().ok_or_else(|| {
            format!("partial_dependence: factor term {term:?} has no saved level set")
        })?;
        return Ok(ResolvedTerm {
            axis_cols: vec![effect.feature_col],
            axis_level_sets: vec![(effect.feature_col, levels_from_bits(levels))],
            pins: Vec::new(),
            numeric_by: None,
        });
    }
    if let Some(smooth) = spec.smooth_terms.iter().find(|smooth| smooth.name == term) {
        // A factor by-smooth is one block per level, and the block's spec records its
        // level, so the by column is held there. A numeric by multiplies the inner
        // smooth, and holding it at one reports the coefficient function itself.
        // A block that carries every level of its factor reads the factor as an
        // axis over the levels it records.
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
            _ => None,
        };
        let factor_axis = match &smooth.basis {
            SmoothBasisSpec::BySmooth {
                by_kind:
                    ByVarKind::Factor {
                        feature_col,
                        frozen_levels,
                        ..
                    },
                ..
            } => Some((*feature_col, frozen_levels.as_deref())),
            SmoothBasisSpec::FactorSumToZero { by_col, levels, .. } => {
                Some((*by_col, Some(levels.as_slice())))
            }
            SmoothBasisSpec::FactorSmooth { spec } => {
                Some((spec.group_col, spec.group_frozen_levels.as_deref()))
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
        let mut axis_level_sets = Vec::new();
        if let Some((factor_col, levels)) = factor_axis {
            if !axis_cols.contains(&factor_col) {
                axis_cols.push(factor_col);
            }
            if let Some(levels) = levels {
                axis_level_sets.push((factor_col, levels_from_bits(levels)));
            }
        }
        return Ok(ResolvedTerm {
            axis_cols,
            axis_level_sets,
            pins: pin.into_iter().collect(),
            numeric_by,
        });
    }
    let available: Vec<&str> = spec
        .linear_terms
        .iter()
        .map(|linear| linear.name.as_str())
        .chain(spec.random_effect_terms.iter().map(|effect| effect.name.as_str()))
        .chain(spec.smooth_terms.iter().map(|smooth| smooth.name.as_str()))
        .collect();
    Err(format!(
        "partial_dependence: {term:?} is not a term of this model; available: {available:?}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_terms::basis::{
        BSplineBasisSpec, BSplineBoundaryConditions, BSplineIdentifiability, BSplineKnotSpec,
        OneDimensionalBoundary,
    };
    use gam_terms::smooth::{
        BySmoothKind, LinearCoefficientGeometry, LinearTermSpec, RandomEffectTermSpec,
        ShapeConstraint, SmoothTermSpec,
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
    fn a_linear_interaction_holds_its_gate_and_a_two_axis_term_sweeps_a_product_grid() {
        let spec = TermCollectionSpec {
            linear_terms: vec![
                linear("x:g[b]", vec![0], vec![(1, 1.0_f64.to_bits())]),
                linear("x:z", vec![0, 2], Vec::new()),
            ],
            random_effect_terms: Vec::new(),
            smooth_terms: Vec::new(),
        };
        let gated = table(&spec, "x:g[b]", PartialDependenceGrid::TrainingRange { n_points: 2 })
            .expect("gated linear table");
        assert!(gated.table.values.column(1).iter().all(|code| *code == 1.0));

        let product = table(&spec, "x:z", PartialDependenceGrid::TrainingRange { n_points: 3 })
            .expect("a two-axis term sweeps the product of its training ranges");
        assert_eq!(product.axes, vec!["x".to_string(), "z".to_string()]);
        assert_eq!(product.axis_levels, vec![None, None]);
        assert_eq!(
            product.axis_values,
            Some(vec![vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0]])
        );
        assert_eq!(product.grid.nrows(), 9);
        assert_eq!(product.grid.row(0).to_vec(), vec![0.0, 1.0]);
        assert_eq!(product.grid.row(1).to_vec(), vec![0.0, 2.0], "the last axis varies fastest");
        assert_eq!(product.grid.row(3).to_vec(), vec![1.0, 1.0]);
        assert_eq!(product.grid.row(8).to_vec(), vec![2.0, 3.0]);
        assert_eq!(product.table.values.column(0).to_vec(), product.grid.column(0).to_vec());
        assert_eq!(product.table.values.column(2).to_vec(), product.grid.column(1).to_vec());

        let explicit = table(
            &spec,
            "x:z",
            PartialDependenceGrid::Explicit(array![[0.25, 1.5], [1.75, 2.5]]),
        )
        .expect("explicit two-axis grid");
        assert_eq!(explicit.axes, vec!["x".to_string(), "z".to_string()]);
        assert_eq!(explicit.axis_values, None);
        assert_eq!(explicit.table.values.column(0).to_vec(), vec![0.25, 1.75]);
        assert_eq!(explicit.table.values.column(2).to_vec(), vec![1.5, 2.5]);
    }

    fn factor_term(name: &str, feature_col: usize, levels: &[f64]) -> RandomEffectTermSpec {
        RandomEffectTermSpec {
            name: name.to_string(),
            feature_col,
            drop_first_level: false,
            penalized: true,
            frozen_levels: Some(levels.iter().map(|level| level.to_bits()).collect()),
            lenient_unseen: false,
        }
    }

    #[test]
    fn a_factor_term_sweeps_its_saved_levels() {
        let spec = TermCollectionSpec {
            linear_terms: Vec::new(),
            random_effect_terms: vec![factor_term("g", 1, &[0.0, 1.0])],
            smooth_terms: Vec::new(),
        };
        let pd = table(&spec, "g", PartialDependenceGrid::TrainingRange { n_points: 50 })
            .expect("a bare factor term");
        assert_eq!(pd.axes, vec!["g".to_string()]);
        assert_eq!(
            pd.axis_levels,
            vec![Some(AxisLevels {
                values: vec![0.0, 1.0],
                labels: vec!["a".to_string(), "b".to_string()],
            })]
        );
        assert_eq!(pd.grid.column(0).to_vec(), vec![0.0, 1.0], "one row per level, not n_points");
        assert_eq!(pd.table.values.column(1).to_vec(), vec![0.0, 1.0]);
        assert_eq!(pd.contribution(), "f(g)");

        let explicit = table(&spec, "g", PartialDependenceGrid::Explicit(array![[1.0]]))
            .expect("an explicit level code");
        assert_eq!(explicit.table.values.column(1).to_vec(), vec![1.0]);
        let refusal = table(&spec, "g", PartialDependenceGrid::Explicit(array![[0.5]]))
            .err()
            .expect("a non-level code is refused");
        assert!(refusal.contains("level codes"), "got: {refusal}");
    }

    #[test]
    fn a_numeric_coded_factor_term_labels_its_levels_by_value() {
        let spec = TermCollectionSpec {
            linear_terms: Vec::new(),
            random_effect_terms: vec![factor_term("factor(z)", 2, &[1.0, 2.0, 3.0])],
            smooth_terms: Vec::new(),
        };
        let pd = table(&spec, "factor(z)", PartialDependenceGrid::TrainingRange { n_points: 4 })
            .expect("a numeric-coded factor term");
        assert_eq!(pd.grid.column(0).to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(
            pd.axis_levels[0].as_ref().map(|levels| levels.labels.clone()),
            Some(vec!["1".to_string(), "2".to_string(), "3".to_string()])
        );
    }

    #[test]
    fn a_block_spanning_every_level_sweeps_the_factor_as_an_axis() {
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
        };
        let pd = table(&spec, "s(x, g, bs=sz)", PartialDependenceGrid::TrainingRange { n_points: 3 })
            .expect("a level-spanning block reads its factor as an axis");
        assert_eq!(pd.axes, vec!["x".to_string(), "g".to_string()]);
        assert_eq!(pd.axis_levels[0], None);
        assert_eq!(
            pd.axis_levels[1].as_ref().map(|levels| levels.labels.clone()),
            Some(vec!["a".to_string(), "b".to_string()])
        );
        assert_eq!(pd.grid.nrows(), 6);
        assert_eq!(pd.table.values.column(1).to_vec(), vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        assert!(pd.held.is_empty());

        let unknown = table(&spec, "s(w)", PartialDependenceGrid::TrainingRange { n_points: 3 })
            .err()
            .expect("an unknown term is refused");
        assert!(
            unknown.contains("available") && unknown.contains("\"z\""),
            "got: {unknown}"
        );
    }
}

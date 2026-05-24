//! Term construction: bridge from parsed formula terms to `TermCollectionSpec`.
//!
//! This module takes the AST produced by `inference::formula_dsl` and a loaded
//! dataset, resolves column references, infers knot counts and center strategies,
//! and produces a `TermCollectionSpec` ready for `build_term_collection_design`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;

use ndarray::{Array2, ArrayView1};

use crate::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineEndpointBoundaryCondition,
    BSplineIdentifiability, BSplineKnotSpec, CenterCountRequest, CenterStrategy, DuchonBasisSpec,
    DuchonNullspaceOrder, DuchonOperatorPenaltySpec, MaternBasisSpec, MaternIdentifiability,
    MaternNu, OneDimensionalBoundary, SpatialIdentifiability, SphereMethod,
    SphereWahbaKernel, SphericalSplineBasisSpec, ThinPlateBasisSpec, auto_spatial_center_strategy,
    default_num_centers, default_spatial_center_strategy, default_spherical_harmonic_degree,
    plan_spatial_basis, resolve_duchon_orders,
};
use crate::inference::data::{EncodedDataset as Dataset, missing_column_message};
use crate::inference::formula_dsl::{
    ParsedTerm, SmoothKind, option_bool, option_f64, option_f64_strict, option_usize,
    option_usize_any, option_usize_any_strict, option_usize_strict, strip_quotes,
};
use crate::inference::model::ColumnKindTag;
use crate::resource::ResourcePolicy;
use crate::smooth::{
    ByVarKind, BySmoothKind, ByVariableSpec, FactorSmoothFlavour, FactorSmoothSpec,
    LinearCoefficientGeometry, LinearTermSpec, RandomEffectTermSpec, ShapeConstraint,
    SmoothBasisSpec, SmoothTermSpec, TensorBSplineIdentifiability, TensorBSplineSpec,
    TermCollectionSpec,
};

// ---------------------------------------------------------------------------
// Typed errors
// ---------------------------------------------------------------------------

/// Typed errors emitted by term-builder helpers. `Display` reproduces the exact
/// pre-refactor `format!(...)` text byte-for-byte, so callers that string-match
/// on the message (tests, log assertions) keep working unchanged. Public-API
/// functions still return `Result<_, String>` and use `.to_string()` shims at
/// their boundary to stay compatible with callers in protected modules.
#[derive(Clone, Debug)]
pub enum TermBuilderError {
    /// Column-resolution / column-kind lookup failures, including "column does
    /// not exist" diagnostics produced via `missing_column_message`.
    MissingColumn { reason: String },
    /// User-specified configuration is internally inconsistent (e.g. too few
    /// variables for a smooth type, conflicting size options, requested basis
    /// dimension below the polynomial nullspace).
    IncompatibleConfig { reason: String },
    /// Option parsing failure: malformed numeric expression, unknown option
    /// key, out-of-range integer, list-length mismatch, etc.
    InvalidOption { reason: String },
    /// User requested a feature that is intentionally not supported (unknown
    /// smooth type / method / kernel / identifiability, non-zero anchor,
    /// internal-only token, etc.).
    UnsupportedFeature { reason: String },
    /// Input data is degenerate for the requested term (constant column,
    /// non-finite categorical entries, ...).
    DegenerateData { reason: String },
    /// Term-collection-stage formula error — a node that the caller was
    /// supposed to resolve upstream reached the builder.
    MalformedFormula { reason: String },
}

impl std::fmt::Display for TermBuilderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TermBuilderError::MissingColumn { reason }
            | TermBuilderError::IncompatibleConfig { reason }
            | TermBuilderError::InvalidOption { reason }
            | TermBuilderError::UnsupportedFeature { reason }
            | TermBuilderError::DegenerateData { reason }
            | TermBuilderError::MalformedFormula { reason } => f.write_str(reason),
        }
    }
}

impl From<TermBuilderError> for String {
    fn from(err: TermBuilderError) -> String {
        err.to_string()
    }
}

// Constructor helpers — keep error-site code compact and consistent.
impl TermBuilderError {
    #[inline]
    fn missing_column(reason: impl Into<String>) -> Self {
        TermBuilderError::MissingColumn {
            reason: reason.into(),
        }
    }
    #[inline]
    fn incompatible_config(reason: impl Into<String>) -> Self {
        TermBuilderError::IncompatibleConfig {
            reason: reason.into(),
        }
    }
    #[inline]
    fn invalid_option(reason: impl Into<String>) -> Self {
        TermBuilderError::InvalidOption {
            reason: reason.into(),
        }
    }
    #[inline]
    fn unsupported_feature(reason: impl Into<String>) -> Self {
        TermBuilderError::UnsupportedFeature {
            reason: reason.into(),
        }
    }
    #[inline]
    fn degenerate_data(reason: impl Into<String>) -> Self {
        TermBuilderError::DegenerateData {
            reason: reason.into(),
        }
    }
    #[inline]
    fn malformed_formula(reason: impl Into<String>) -> Self {
        TermBuilderError::MalformedFormula {
            reason: reason.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Column resolution
// ---------------------------------------------------------------------------

pub fn resolve_col(col_map: &HashMap<String, usize>, name: &str) -> Result<usize, String> {
    col_map
        .get(name)
        .copied()
        .ok_or_else(|| missing_column_message(col_map, name, None))
}

pub fn resolve_role_col(
    col_map: &HashMap<String, usize>,
    name: &str,
    role: &str,
) -> Result<usize, String> {
    col_map
        .get(name)
        .copied()
        .ok_or_else(|| missing_column_message(col_map, name, Some(role)))
}

fn encoded_levels_for_column(ds: &Dataset, col: usize) -> Vec<(u64, String)> {
    let mut seen = BTreeSet::<u64>::new();
    for value in ds.values.column(col) {
        if value.is_finite() {
            seen.insert(value.to_bits());
        }
    }
    let schema_levels = ds
        .schema
        .columns
        .get(col)
        .map(|column| column.levels.as_slice())
        .unwrap_or(&[]);
    seen.into_iter()
        .enumerate()
        .map(|(idx, bits)| {
            let fallback = format!("level{}", idx + 1);
            let label = schema_levels.get(idx).cloned().unwrap_or(fallback);
            (bits, label)
        })
        .collect()
}

pub fn column_map_with_alias(
    col_map: &HashMap<String, usize>,
    alias: &str,
    target_column: &str,
) -> HashMap<String, usize> {
    let mut aliased = col_map.clone();
    if let Some(idx) = col_map.get(target_column).copied() {
        aliased.entry(alias.to_string()).or_insert(idx);
    }
    aliased
}

// ---------------------------------------------------------------------------
// ParsedTerm[] + Dataset → TermCollectionSpec
// ---------------------------------------------------------------------------

pub fn build_termspec(
    terms: &[ParsedTerm],
    ds: &Dataset,
    col_map: &HashMap<String, usize>,
    inference_notes: &mut Vec<String>,
    policy: &ResourcePolicy,
) -> Result<TermCollectionSpec, String> {
    let mut linear_terms = Vec::<LinearTermSpec>::new();
    let mut random_terms = Vec::<RandomEffectTermSpec>::new();
    let mut smooth_terms = Vec::<SmoothTermSpec>::new();
    let smooth_coordinate_count = terms
        .iter()
        .map(|term| match term {
            ParsedTerm::Smooth { vars, .. } => vars.len(),
            _ => 0,
        })
        .sum::<usize>();

    for t in terms {
        match t {
            ParsedTerm::Linear {
                name,
                explicit,
                coefficient_min,
                coefficient_max,
            } => {
                let col = resolve_col(col_map, name)?;
                let auto_kind = ds.column_kinds.get(col).copied().ok_or_else(|| {
                    TermBuilderError::missing_column(format!(
                        "internal column-kind lookup failed for '{name}'"
                    ))
                    .to_string()
                })?;
                if *explicit {
                    linear_terms.push(LinearTermSpec {
                        name: name.clone(),
                        feature_col: col,
                        double_penalty: true,
                        coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                        coefficient_min: *coefficient_min,
                        coefficient_max: *coefficient_max,
                    });
                } else {
                    match auto_kind {
                        ColumnKindTag::Continuous | ColumnKindTag::Binary => {
                            linear_terms.push(LinearTermSpec {
                                name: name.clone(),
                                feature_col: col,
                                double_penalty: true,
                                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                                coefficient_min: *coefficient_min,
                                coefficient_max: *coefficient_max,
                            });
                        }
                        ColumnKindTag::Categorical => {
                            if coefficient_min.is_some() || coefficient_max.is_some() {
                                return Err(TermBuilderError::incompatible_config(format!(
                                    "coefficient constraints are not supported for categorical auto-random-effect term '{name}'; use group({name}) or an unconstrained numeric term"
                                ))
                                .to_string());
                            }
                            random_terms.push(RandomEffectTermSpec {
                                name: name.clone(),
                                feature_col: col,
                                drop_first_level: false,
                                penalized: true,
                                frozen_levels: None,
                            });
                        }
                    }
                }
            }
            ParsedTerm::BoundedLinear {
                name,
                min,
                max,
                prior,
            } => {
                let col = resolve_col(col_map, name)?;
                let auto_kind = ds.column_kinds.get(col).copied().ok_or_else(|| {
                    TermBuilderError::missing_column(format!(
                        "internal column-kind lookup failed for '{name}'"
                    ))
                    .to_string()
                })?;
                if !matches!(auto_kind, ColumnKindTag::Continuous | ColumnKindTag::Binary) {
                    return Err(TermBuilderError::incompatible_config(format!(
                        "bounded() currently supports only numeric columns, got categorical '{name}'"
                    ))
                    .to_string());
                }
                linear_terms.push(LinearTermSpec {
                    name: name.clone(),
                    feature_col: col,
                    double_penalty: false,
                    coefficient_geometry: LinearCoefficientGeometry::Bounded {
                        min: *min,
                        max: *max,
                        prior: prior.clone(),
                    },
                    coefficient_min: None,
                    coefficient_max: None,
                });
            }
            ParsedTerm::RandomEffect { name } => {
                let col = resolve_col(col_map, name)?;
                random_terms.push(RandomEffectTermSpec {
                    name: name.clone(),
                    feature_col: col,
                    drop_first_level: false,
                    penalized: true,
                    frozen_levels: None,
                });
            }
            ParsedTerm::Smooth {
                label,
                vars,
                kind,
                options,
            } => {
                let mut smooth_vars = vars.clone();
                let by_name = options.get("by").cloned();
                let bs_name = options
                    .get("bs")
                    .or_else(|| options.get("type"))
                    .map(|v| v.to_ascii_lowercase());
                let is_sz = matches!(
                    bs_name.as_deref(),
                    Some("sz") | Some("sum-to-zero") | Some("sum_to_zero")
                );
                if is_sz {
                    if vars.len() < 2 {
                        return Err(format!(
                            "bs=sz smooth '{}' expects a factor followed by one or more smooth variables",
                            label
                        ));
                    }
                    smooth_vars = vars[1..].to_vec();
                }
                let cols = smooth_vars
                    .iter()
                    .map(|v| resolve_col(col_map, v))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut inner_options = options.clone();
                inner_options.remove("by");
                if is_sz {
                    inner_options.remove("bs");
                    inner_options.remove("type");
                }
                let inner_basis = build_smooth_basis(
                    *kind,
                    &smooth_vars,
                    &cols,
                    &inner_options,
                    ds,
                    inference_notes,
                    policy,
                    smooth_coordinate_count,
                )?;
                if is_sz {
                    let by_col = resolve_col(col_map, &vars[0])?;
                    if !matches!(
                        ds.column_kinds.get(by_col),
                        Some(ColumnKindTag::Categorical)
                    ) {
                        return Err(format!(
                            "bs=sz smooth '{}' requires categorical factor '{}'; got numeric column",
                            label, vars[0]
                        ));
                    }
                    let mut levels: Vec<u64> = ds
                        .values
                        .column(by_col)
                        .iter()
                        .map(|v| v.to_bits())
                        .collect();
                    levels.sort_unstable();
                    levels.dedup();
                    smooth_terms.push(SmoothTermSpec {
                        name: label.clone(),
                        basis: SmoothBasisSpec::FactorSumToZero {
                            inner: Box::new(inner_basis),
                            by_col,
                            levels,
                        },
                        shape: ShapeConstraint::None,
                    });
                } else if let Some(by_name) = by_name {
                    let by_col = resolve_col(col_map, &by_name)?;
                    match ds.column_kinds.get(by_col).copied().ok_or_else(|| {
                        format!("internal column-kind lookup failed for by variable '{by_name}'")
                    })? {
                        ColumnKindTag::Categorical => {
                            let levels = encoded_levels_for_column(ds, by_col);
                            // Add an unpenalized treatment-coded fixed main effect for the factor, unless present already.
                            if !random_terms
                                .iter()
                                .any(|rt| rt.name == by_name && !rt.penalized)
                            {
                                random_terms.push(RandomEffectTermSpec {
                                    name: by_name.clone(),
                                    feature_col: by_col,
                                    drop_first_level: true,
                                    penalized: false,
                                    frozen_levels: None,
                                });
                            }
                            for (level_bits, level_label) in levels {
                                smooth_terms.push(SmoothTermSpec {
                                    name: format!("{}:{}", label, level_bits),
                                    basis: SmoothBasisSpec::ByVariable {
                                        inner: Box::new(inner_basis.clone()),
                                        by_col,
                                        kind: BySmoothKind::Level { level_bits },
                                        by: ByVariableSpec::Level {
                                            value_bits: level_bits,
                                            label: level_label,
                                        },
                                    },
                                    shape: ShapeConstraint::None,
                                });
                            }
                        }
                        ColumnKindTag::Binary | ColumnKindTag::Continuous => {
                            smooth_terms.push(SmoothTermSpec {
                                name: label.clone(),
                                basis: SmoothBasisSpec::ByVariable {
                                    inner: Box::new(inner_basis),
                                    by_col,
                                    kind: BySmoothKind::Numeric,
                                    by: ByVariableSpec::Numeric,
                                },
                                shape: ShapeConstraint::None,
                            });
                        }
                    }
                } else {
                    smooth_terms.push(SmoothTermSpec {
                        name: label.clone(),
                        basis: inner_basis,
                        shape: ShapeConstraint::None,
                    });
                }
            }
            ParsedTerm::LinkWiggle { .. }
            | ParsedTerm::TimeWiggle { .. }
            | ParsedTerm::LinkConfig { .. }
            | ParsedTerm::SurvivalConfig { .. } => {
                // Consumed at formula level, not design terms.
            }
            ParsedTerm::LogSlopeSurface { .. } => {
                return Err(TermBuilderError::malformed_formula(
                    "logslope(...) declarations must be resolved by the marginal-slope formula path before building a term spec",
                )
                .to_string());
            }
        }
    }

    Ok(TermCollectionSpec {
        linear_terms,
        random_effect_terms: random_terms,
        smooth_terms,
    })
}

fn split_list_option(raw: &str) -> Vec<String> {
    let t = raw.trim();
    let inner = t
        .strip_prefix('[')
        .and_then(|u| u.strip_suffix(']'))
        .unwrap_or(t);
    inner
        .split(',')
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .collect()
}

fn parse_numeric_expr(raw: &str) -> Result<f64, String> {
    let mut acc = 1.0f64;
    let normalized = raw.replace(' ', "");
    if normalized.eq_ignore_ascii_case("none") {
        return Err("None is not numeric".to_string());
    }
    for factor in normalized.split('*') {
        if factor.is_empty() {
            return Err(format!("invalid numeric expression '{raw}'"));
        }
        let value = if factor.eq_ignore_ascii_case("pi") || factor == "π" {
            std::f64::consts::PI
        } else if factor.eq_ignore_ascii_case("tau") || factor == "τ" {
            std::f64::consts::TAU
        } else if let Some(prefix) = factor
            .strip_suffix("pi")
            .or_else(|| factor.strip_suffix("π"))
        {
            let coefficient = if prefix.is_empty() {
                1.0
            } else {
                prefix
                    .parse::<f64>()
                    .map_err(|err| format!("invalid numeric expression '{raw}': {err}"))?
            };
            coefficient * std::f64::consts::PI
        } else if let Some(prefix) = factor
            .strip_suffix("tau")
            .or_else(|| factor.strip_suffix("τ"))
        {
            let coefficient = if prefix.is_empty() {
                1.0
            } else {
                prefix
                    .parse::<f64>()
                    .map_err(|err| format!("invalid numeric expression '{raw}': {err}"))?
            };
            coefficient * std::f64::consts::TAU
        } else {
            factor
                .parse::<f64>()
                .map_err(|err| format!("invalid numeric expression '{raw}': {err}"))?
        };
        acc *= value;
    }
    Ok(acc)
}

fn parse_periods_option(
    options: &BTreeMap<String, String>,
    dim: usize,
) -> Result<Option<Vec<Option<f64>>>, String> {
    let Some(raw) = options.get("period") else {
        return Ok(None);
    };
    let values = split_list_option(raw);
    let mut periods = vec![None; dim];
    if values.len() == 1 && dim == 1 {
        periods[0] = Some(parse_numeric_expr(&values[0])?);
    } else {
        if values.len() != dim {
            return Err(format!(
                "period list length {} must match smooth dimension {}",
                values.len(),
                dim
            ));
        }
        for (i, v) in values.iter().enumerate() {
            if v.eq_ignore_ascii_case("none") {
                continue;
            }
            periods[i] = Some(parse_numeric_expr(v)?);
        }
    }
    Ok(Some(periods))
}

fn parse_periodic_axes_option(
    options: &BTreeMap<String, String>,
    dim: usize,
) -> Result<Option<Vec<Option<f64>>>, String> {
    let Some(raw_axes) = options.get("periodic") else {
        return Ok(None);
    };
    let mut periods = parse_periods_option(options, dim)?.unwrap_or_else(|| vec![None; dim]);
    let axes = split_list_option(raw_axes);
    if axes.is_empty() {
        return Ok(Some(periods));
    }
    for a in axes {
        let axis = a
            .parse::<usize>()
            .map_err(|err| format!("invalid periodic axis '{a}': {err}"))?;
        if axis >= dim {
            return Err(format!(
                "periodic axis {axis} out of range for {dim}D smooth"
            ));
        }
        if periods[axis].is_none() {
            return Err(format!(
                "periodic axis {axis} requires period[{axis}] to be finite"
            ));
        }
    }
    // Axes not listed are non-periodic even if period list has a finite placeholder.
    let listed: std::collections::BTreeSet<usize> = split_list_option(raw_axes)
        .into_iter()
        .filter_map(|a| a.parse::<usize>().ok())
        .collect();
    for i in 0..dim {
        if !listed.contains(&i) {
            periods[i] = None;
        }
    }
    Ok(Some(periods))
}

// ---------------------------------------------------------------------------
// Smooth basis spec construction
// ---------------------------------------------------------------------------

fn parse_option_list(raw: &str) -> Vec<String> {
    let trimmed = raw.trim();
    let inner = trimmed
        .strip_prefix('[')
        .and_then(|v| v.strip_suffix(']'))
        .unwrap_or(trimmed);
    inner
        .split(',')
        .map(|v| {
            v.trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_ascii_lowercase()
        })
        .filter(|v| !v.is_empty())
        .collect()
}

fn parse_periodic_axes(
    options: &BTreeMap<String, String>,
    dim: usize,
) -> Result<Vec<bool>, String> {
    let mut axes = vec![false; dim];
    if let Some(raw) = options.get("periodic").or_else(|| options.get("cyclic")) {
        let lowered = raw.trim().to_ascii_lowercase();
        match lowered.as_str() {
            "true" | "yes" | "y" => {
                axes.fill(true);
                return Ok(axes);
            }
            "false" | "no" | "n" => return Ok(axes),
            _ => {}
        }
        for axis_raw in parse_option_list(raw) {
            let axis = axis_raw
                .parse::<usize>()
                .map_err(|err| format!("invalid periodic axis '{axis_raw}': {err}"))?;
            if axis >= dim {
                return Err(format!(
                    "periodic axis {axis} out of range for {dim}D smooth"
                ));
            }
            axes[axis] = true;
        }
    }
    if let Some(raw) = options.get("boundary").or_else(|| options.get("bc")) {
        let boundary = parse_option_list(raw);
        if boundary.len() == dim {
            for (axis, value) in boundary.iter().enumerate() {
                if matches!(value.as_str(), "periodic" | "cyclic" | "cc") {
                    axes[axis] = true;
                }
            }
        } else if dim == 1
            && matches!(
                boundary.first().map(String::as_str),
                Some("periodic" | "cyclic" | "cc")
            )
        {
            axes[0] = true;
        }
    }
    Ok(axes)
}

fn parse_optional_numeric_list(
    options: &BTreeMap<String, String>,
    keys: &[&str],
    dim: usize,
) -> Result<Vec<Option<f64>>, String> {
    let Some(raw) = keys.iter().find_map(|key| options.get(*key)) else {
        return Ok(vec![None; dim]);
    };
    let values = split_list_option(raw);
    let mut out = vec![None; dim];
    if values.len() == 1 && dim == 1 {
        if !values[0].eq_ignore_ascii_case("none") {
            out[0] = Some(parse_numeric_expr(&values[0])?);
        }
        return Ok(out);
    }
    if values.len() != dim {
        return Err(format!(
            "numeric option list length {} must match smooth dimension {}",
            values.len(),
            dim
        ));
    }
    for (i, value) in values.iter().enumerate() {
        if !value.eq_ignore_ascii_case("none") {
            out[i] = Some(parse_numeric_expr(value)?);
        }
    }
    Ok(out)
}

fn parse_periods(
    options: &BTreeMap<String, String>,
    periodic_axes: &[bool],
) -> Result<Vec<Option<f64>>, String> {
    let periods =
        parse_optional_numeric_list(options, &["period", "periods"], periodic_axes.len())?;
    for (axis, (periodic, period)) in periodic_axes.iter().zip(periods.iter()).enumerate() {
        if *periodic
            && let Some(value) = period
            && (!value.is_finite() || *value <= 0.0)
        {
            return Err(format!(
                "period for periodic axis {axis} must be finite and positive, got {value}"
            ));
        }
    }
    Ok(periods)
}

fn parse_period_origins(
    options: &BTreeMap<String, String>,
    periodic_axes: &[bool],
) -> Result<Vec<Option<f64>>, String> {
    parse_optional_numeric_list(
        options,
        &[
            "origin",
            "origins",
            "period_origin",
            "period-origin",
            "domain_origin",
        ],
        periodic_axes.len(),
    )
}

fn bspline_boundary_declares_periodic_axis(options: &BTreeMap<String, String>) -> bool {
    options
        .get("boundary")
        .or_else(|| options.get("bc"))
        .map(|raw| {
            parse_option_list(raw)
                .into_iter()
                .any(|value| matches!(value.as_str(), "periodic" | "cyclic" | "cc"))
        })
        .unwrap_or(false)
}

fn parse_f64_option_list(raw: &str) -> Result<Vec<Option<f64>>, String> {
    parse_option_list(raw)
        .into_iter()
        .map(|v| {
            if v.eq_ignore_ascii_case("none") {
                Ok(None)
            } else {
                parse_numeric_expr(&v)
                    .map(Some)
                    .map_err(|err| format!("invalid numeric option list value '{v}': {err}"))
            }
        })
        .collect()
}

fn tensor_margin_boundaries(
    options: &BTreeMap<String, String>,
    cols: &[usize],
    ds: &Dataset,
) -> Result<Vec<OneDimensionalBoundary>, String> {
    let mut out = vec![OneDimensionalBoundary::Open; cols.len()];
    let Some(raw_boundary) = options.get("boundary").or_else(|| options.get("bc")) else {
        return Ok(out);
    };
    let boundaries = parse_option_list(raw_boundary);
    if boundaries.len() != cols.len() {
        return Err(format!(
            "te()/tensor() boundary must have one entry per margin: got {} for {} variables",
            boundaries.len(),
            cols.len()
        ));
    }
    let periods = options
        .get("period")
        .or_else(|| options.get("periods"))
        .map(|raw| parse_f64_option_list(raw))
        .transpose()?
        .unwrap_or_default();
    let origins = options
        .get("origin")
        .or_else(|| options.get("origins"))
        .map(|raw| parse_f64_option_list(raw))
        .transpose()?
        .unwrap_or_default();
    for (i, boundary) in boundaries.iter().enumerate() {
        match boundary.as_str() {
            "periodic" | "cyclic" | "cc" => {
                let origin = origins.get(i).copied().flatten().map(Ok).unwrap_or_else(|| {
                    col_minmax(ds.values.column(cols[i])).map(|(minv, _)| minv)
                })?;
                let period = periods.get(i).copied().flatten().ok_or_else(|| {
                    "te()/tensor() periodic margins require period=[...] with one positive period per margin".to_string()
                })?;
                if !period.is_finite() || period <= 0.0 {
                    return Err(format!(
                        "te()/tensor() period for margin {} must be finite and positive, got {}",
                        i, period
                    ));
                }
                out[i] = OneDimensionalBoundary::Cyclic {
                    start: origin,
                    end: origin + period,
                };
            }
            "open" | "none" | "natural" => {}
            other => {
                return Err(format!(
                    "unsupported te()/tensor() boundary '{other}'; supported values are open and periodic"
                ));
            }
        }
    }
    Ok(out)
}

pub fn build_smooth_basis(
    kind: SmoothKind,
    vars: &[String],
    cols: &[usize],
    options: &BTreeMap<String, String>,
    ds: &Dataset,
    inference_notes: &mut Vec<String>,
    policy: &ResourcePolicy,
    smooth_coordinate_count: usize,
) -> Result<SmoothBasisSpec, String> {
    // Fail fast on degenerate input columns: a smooth over a column that takes
    // only one finite value can only ever fit the response mean — the design
    // matrix is rank-1, and the user almost certainly didn't mean to model a
    // constant predictor as a smooth. Without this guard, `smooth(x)` and
    // `matern(x)` silently fit the mean of `y` regardless of `x`, and the
    // user has no way to tell from looking at the predictions (they're all
    // the same number). Duchon already errors loudly via the basis layer
    // ("smooth basis collapses onto the parametric block"); this lift makes
    // the same diagnosis explicit and uniform across smooth families.
    for (var, &col) in vars.iter().zip(cols.iter()) {
        if matches!(ds.column_kinds.get(col), Some(ColumnKindTag::Categorical)) {
            continue;
        }
        if unique_count_column(ds.values.column(col)) <= 1 {
            return Err(TermBuilderError::degenerate_data(format!(
                "smooth term over '{var}' has only one unique value in the training data \
                 — a smooth on a constant column is degenerate and would only fit the response mean. \
                 Remove `{var}` from the smooth, drop the term, or check the data."
            ))
            .to_string());
        }
    }
    if let Some(by_name) = options.get("by").cloned() {
        let by_col = options
            .get("__by_col")
            .and_then(|raw| raw.parse::<usize>().ok())
            .or_else(|| vars.iter().position(|v| v == &by_name).map(|idx| cols[idx]))
            .ok_or_else(|| format!("unknown by= column '{by_name}'"))?;
        let mut inner_options = options.clone();
        inner_options.remove("by");
        inner_options.remove("__by_col");
        inner_options.remove("id");
        let inner = build_smooth_basis(
            kind,
            vars,
            cols,
            &inner_options,
            ds,
            inference_notes,
            policy,
            smooth_coordinate_count,
        )?;
        let by_kind = match ds.column_kinds.get(by_col).copied() {
            Some(ColumnKindTag::Categorical) => ByVarKind::Factor {
                feature_col: by_col,
                ordered: option_bool(options, "ordered").unwrap_or(false),
                frozen_levels: None,
            },
            Some(ColumnKindTag::Continuous | ColumnKindTag::Binary) => ByVarKind::Numeric {
                feature_col: by_col,
            },
            None => {
                return Err(format!(
                    "internal column-kind lookup failed for by='{by_name}'"
                ));
            }
        };
        return Ok(SmoothBasisSpec::BySmooth {
            smooth: Box::new(inner),
            by_kind,
        });
    }

    let smooth_double_penalty = option_bool(options, "double_penalty").unwrap_or(true);
    let has_periodic_option = options.contains_key("periodic")
        || options.contains_key("cyclic")
        || options
            .get("boundary")
            .or_else(|| options.get("bc"))
            .map(|boundary| {
                boundary.to_ascii_lowercase().contains("periodic")
                    || boundary.to_ascii_lowercase().contains("cyclic")
            })
            .unwrap_or(false);
    let type_opt = options
        .get("type")
        .or_else(|| options.get("bs"))
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| match kind {
            SmoothKind::Te => "tensor".to_string(),
            SmoothKind::S if cols.len() == 1 => "bspline".to_string(),
            // Mixed periodic Euclidean radial kernels are not separable on the
            // cylinder. Use a tensor product with a cyclic margin so s(theta,h)
            // honors seam continuity while preserving the formula-level s(...).
            SmoothKind::S if has_periodic_option => "tensor".to_string(),
            SmoothKind::S => "tps".to_string(),
        });

    if matches!(type_opt.as_str(), "fs" | "sz" | "re") {
        validate_known_options(
            type_opt.as_str(),
            options,
            &[
                "type",
                "bs",
                "k",
                "basis_dim",
                "basis-dim",
                "basisdim",
                "knots",
                "degree",
                "penalty_order",
                "m",
                "double_penalty",
                "ordered",
            ],
        )?;
        if cols.len() != 2 {
            return Err(format!(
                "{} factor-smooth currently expects exactly two variables (one numeric, one categorical)",
                type_opt
            ));
        }
        let kinds = cols
            .iter()
            .map(|&c| ds.column_kinds.get(c).copied())
            .collect::<Vec<_>>();
        let (cont_idx, group_idx) = if type_opt == "re" {
            // mgcv random-slope examples are often s(g, x, bs="re").
            match (kinds[0], kinds[1]) {
                (Some(ColumnKindTag::Categorical), _) => (1usize, 0usize),
                (_, Some(ColumnKindTag::Categorical)) => (0usize, 1usize),
                _ => (1usize, 0usize),
            }
        } else {
            match (kinds[0], kinds[1]) {
                (_, Some(ColumnKindTag::Categorical)) => (0usize, 1usize),
                (Some(ColumnKindTag::Categorical), _) => (1usize, 0usize),
                _ => {
                    return Err(format!(
                        "{} factor-smooth requires one categorical factor variable",
                        type_opt
                    ));
                }
            }
        };
        let c = cols[cont_idx];
        let (minv, maxv) = col_minmax(ds.values.column(c))?;
        let degree = if type_opt == "re" {
            1
        } else {
            option_usize(options, "degree").unwrap_or(3)
        };
        let default_internal = heuristic_knots_for_column(ds.values.column(c));
        let (n_knots, _) = parse_ps_internal_knots(options, degree, default_internal)?;
        let marginal = BSplineBasisSpec {
            degree,
            penalty_order: option_usize(options, "penalty_order").unwrap_or(if degree > 1 {
                2
            } else {
                1
            }),
            knotspec: BSplineKnotSpec::Generate {
                data_range: (minv, maxv),
                num_internal_knots: n_knots,
            },
            double_penalty: true,
            identifiability: BSplineIdentifiability::None,
            boundary_conditions: Default::default(),
            boundary: OneDimensionalBoundary::Open,
            streaming_chunk_size: None,
        };
        let flavour = match type_opt.as_str() {
            "fs" => FactorSmoothFlavour::Fs {
                m_null_penalty_orders: vec![option_usize(options, "m").unwrap_or(2)],
            },
            "sz" => FactorSmoothFlavour::Sz,
            "re" => FactorSmoothFlavour::Re,
            // Outer `matches!` already restricts to fs/sz/re.
            other => {
                return Err(format!(
                    "internal: factor-smooth flavour dispatch reached unexpected type `{}`",
                    other
                ));
            }
        };
        return Ok(SmoothBasisSpec::FactorSmooth {
            spec: FactorSmoothSpec {
                continuous_cols: vec![c],
                group_col: cols[group_idx],
                marginal,
                flavour,
                group_frozen_levels: None,
            },
        });
    }

    match type_opt.as_str() {
        "cyclic" | "cc" | "cp" | "cyclic-ps" => {
            if cols.len() != 1 {
                return Err(format!(
                    "periodic smooth expects one variable, got {}",
                    cols.len()
                ));
            }
            let c = cols[0];
            let (minv, maxv) = col_minmax(ds.values.column(c))?;
            let degree = option_usize(options, "degree").unwrap_or(3);
            let mut default_internal = heuristic_knots_for_column(ds.values.column(c));
            if ds.values.nrows() <= 32 && smooth_coordinate_count >= 5 {
                default_internal = default_internal.min(1);
            }
            let default_basis = default_internal + degree + 1;
            let num_basis = option_usize_any(options, &["k", "basis_dim", "basis-dim", "basisdim"])
                .unwrap_or(default_basis);
            if num_basis < degree + 1 {
                return Err(format!(
                    "periodic smooth: k={} too small for degree {}; expected k >= {}",
                    num_basis,
                    degree,
                    degree + 1
                ));
            }
            let (domain_start, period) = parse_periodic_domain_1d(options, minv, maxv)?;
            Ok(SmoothBasisSpec::BSpline1D {
                feature_col: c,
                spec: BSplineBasisSpec {
                    degree,
                    penalty_order: option_usize(options, "penalty_order").unwrap_or(2),
                    knotspec: BSplineKnotSpec::PeriodicUniform {
                        data_range: (domain_start, domain_start + period),
                        num_basis,
                    },
                    double_penalty: smooth_double_penalty,
                    identifiability: BSplineIdentifiability::default(),
                    boundary_conditions: Default::default(),
                    boundary: OneDimensionalBoundary::Cyclic {
                        start: domain_start,
                        end: domain_start + period,
                    },
                    streaming_chunk_size: None,
                },
            })
        }
        "bspline" | "ps" | "p-spline" => {
            validate_known_options(
                "bspline",
                options,
                &[
                    "type",
                    "bs",
                    "by",
                    "k",
                    "basis_dim",
                    "basis-dim",
                    "basisdim",
                    "knots",
                    "degree",
                    "penalty_order",
                    "boundary",
                    "bc",
                    "boundary_conditions",
                    "bc_left",
                    "bc_right",
                    "left_bc",
                    "right_bc",
                    "start_bc",
                    "end_bc",
                    "side",
                    "anchor",
                    "anchor_value",
                    "value",
                    "anchor_left",
                    "left_anchor",
                    "anchor_right",
                    "right_anchor",
                    "periodic",
                    "period",
                    "periods",
                    "period_start",
                    "period_end",
                    "origin",
                    "double_penalty",
                    "by",
                    "id",
                    "__by_col",
                    "identifiability",
                    "streaming_chunk_size",
                    "chunk_size",
                    "by",
                ],
            )?;
            if cols.len() != 1 {
                return Err(TermBuilderError::incompatible_config(format!(
                    "bspline smooth expects one variable, got {}",
                    cols.len()
                ))
                .to_string());
            }
            let c = cols[0];
            let (minv, maxv) = col_minmax(ds.values.column(c))?;
            let degree = option_usize(options, "degree").unwrap_or(3);
            let default_internal = heuristic_knots_for_column(ds.values.column(c));
            let (mut n_knots, inferred) =
                parse_ps_internal_knots(options, degree, default_internal)?;
            if inferred && ds.values.nrows() <= 32 && smooth_coordinate_count >= 5 {
                n_knots = n_knots.min(1);
            }
            if inferred {
                let unique = unique_count_column(ds.values.column(c));
                let ceiling = ((unique as f64).cbrt() as usize).max(20);
                inference_notes.push(format!(
                    "Automatically set {} internal knots for smooth '{}' from {} unique values (rule: clamp(unique/4, 4..max(20, cbrt(unique))) = clamp(unique/4, 4..{})). Override with knots=... or k=....",
                    n_knots,
                    vars.join(","),
                    unique,
                    ceiling,
                ));
            }
            let periodic_axes = parse_periodic_axes(options, 1).map_err(|e| e.to_string())?;
            let boundary_conditions =
                if periodic_axes[0] && bspline_boundary_declares_periodic_axis(options) {
                    BSplineBoundaryConditions::default()
                } else {
                    parse_bspline_boundary_conditions(options).map_err(|e| e.to_string())?
                };
            let periods = parse_periods(options, &periodic_axes).map_err(|e| e.to_string())?;
            let origins =
                parse_period_origins(options, &periodic_axes).map_err(|e| e.to_string())?;
            let (knotspec, boundary) = if periodic_axes[0] {
                if !boundary_conditions.is_free() {
                    return Err(TermBuilderError::incompatible_config(
                        "periodic B-splines cannot also declare endpoint boundary conditions",
                    )
                    .to_string());
                }
                {
                    let (domain_start, p_value) = if periods[0].is_some() {
                        (origins[0].unwrap_or(minv), periods[0].unwrap())
                    } else {
                        parse_periodic_domain_1d(options, minv, maxv).map_err(|e| e.to_string())?
                    };
                    let domain_end = domain_start + p_value;
                    (
                        BSplineKnotSpec::PeriodicUniform {
                            data_range: (domain_start, domain_end),
                            num_basis: n_knots + degree + 1,
                        },
                        OneDimensionalBoundary::Cyclic {
                            start: domain_start,
                            end: domain_end,
                        },
                    )
                }
            } else {
                (
                    BSplineKnotSpec::Generate {
                        data_range: (minv, maxv),
                        num_internal_knots: n_knots,
                    },
                    parse_cyclic_boundary(options, minv, maxv)?,
                )
            };
            Ok(SmoothBasisSpec::BSpline1D {
                feature_col: c,
                spec: BSplineBasisSpec {
                    degree,
                    penalty_order: option_usize(options, "penalty_order").unwrap_or(2),
                    knotspec,
                    double_penalty: smooth_double_penalty,
                    identifiability: BSplineIdentifiability::default(),
                    boundary,
                    boundary_conditions,
                    streaming_chunk_size: option_usize(options, "streaming_chunk_size")
                        .or_else(|| option_usize(options, "chunk_size")),
                },
            })
        }
        "tps" | "thinplate" | "thin-plate" => {
            validate_known_options(
                "thinplate",
                options,
                &[
                    "type",
                    "bs",
                    "by",
                    "length_scale",
                    "centers",
                    "k",
                    "basis_dim",
                    "basis-dim",
                    "basisdim",
                    "knots",
                    "include_intercept",
                    "double_penalty",
                    "by",
                    "id",
                    "__by_col",
                    "identifiability",
                    "by",
                    "scale_dims",
                ],
            )?;
            let plan = plan_spatial_basis(
                ds.values.nrows(),
                cols.len(),
                CenterCountRequest::Default,
                DuchonNullspaceOrder::Linear,
                option_bool(options, "scale_dims").unwrap_or(false),
                policy,
            )
            .map_err(|e| e.to_string())?;
            let centers = parse_countwith_basis_alias(options, "centers", plan.centers)?;
            let center_strategy = if has_explicit_countwith_basis_alias(options, "centers") {
                spatial_center_strategy_for_dimension(centers, cols.len())
            } else {
                auto_spatial_center_strategy(centers, cols.len())
            };
            Ok(SmoothBasisSpec::ThinPlate {
                feature_cols: cols.to_vec(),
                spec: ThinPlateBasisSpec {
                    center_strategy,
                    periodic: parse_periodic_axes_option(options, cols.len())?,
                    length_scale: option_f64(options, "length_scale").unwrap_or(1.0),
                    double_penalty: smooth_double_penalty,
                    identifiability: parse_spatial_identifiability(options)
                        .map_err(|e| e.to_string())?,
                    radial_reparam: None,
                },
                input_scales: None,
            })
        }
        "sphere" | "s2" | "sos" => {
            validate_known_options(
                "sphere",
                options,
                &[
                    "type",
                    "bs",
                    "by",
                    "centers",
                    "k",
                    "basis_dim",
                    "basis-dim",
                    "basisdim",
                    "knots",
                    "penalty_order",
                    "m",
                    "double_penalty",
                    "id",
                    "__by_col",
                    "kernel",
                    "method",
                    "radians",
                    "units",
                    "degree",
                    "l",
                    "max_degree",
                    "max-degree",
                    "streaming_chunk_size",
                    "chunk_size",
                ],
            )?;
            if cols.len() != 2 {
                return Err(format!(
                    "sphere smooth expects exactly two variables (lat, lon), got {}",
                    cols.len()
                ));
            }
            let radians = option_bool(options, "radians").unwrap_or_else(|| {
                options
                    .get("units")
                    .map(|u| u.eq_ignore_ascii_case("radian") || u.eq_ignore_ascii_case("radians"))
                    .unwrap_or(false)
            });
            let kernel = options
                .get("kernel")
                .or_else(|| options.get("method"))
                .map(|raw| strip_quotes(raw).trim().to_ascii_lowercase())
                .unwrap_or_else(|| "sobolev".to_string());
            let (method, wahba_kernel) = match kernel.as_str() {
                "sobolev" | "wahba" => (SphereMethod::Wahba, SphereWahbaKernel::Sobolev),
                "pseudo" | "mgcv" | "sos" => (SphereMethod::Wahba, SphereWahbaKernel::Pseudo),
                "harmonic" | "spherical_harmonic" | "spherical-harmonic" => {
                    (SphereMethod::Harmonic, SphereWahbaKernel::Sobolev)
                }
                other => {
                    return Err(format!(
                        "unsupported sphere kernel '{other}'; expected sobolev, pseudo, or harmonic"
                    ));
                }
            };
            let streaming_chunk_size = option_usize(options, "streaming_chunk_size")
                .or_else(|| option_usize(options, "chunk_size"));
            if matches!(method, SphereMethod::Harmonic) && streaming_chunk_size.is_some() {
                return Err(
                    "sphere streaming_chunk_size is only supported for Wahba kernels".to_string(),
                );
            }
            let max_degree = if matches!(method, SphereMethod::Harmonic) {
                let degree = option_usize_any(options, &["degree", "l", "max_degree", "max-degree"])
                    .or_else(|| option_usize(options, "centers"))
                    .or_else(|| {
                        option_usize_any(options, &["k", "basis_dim", "basis-dim", "basisdim"])
                            .and_then(|k| (1..=128).find(|&l| l * (l + 2) >= k))
                    })
                    .unwrap_or_else(|| default_spherical_harmonic_degree(ds.values.nrows()));
                if degree == 0 {
                    return Err("sphere smooth requires degree/max_degree >= 1".to_string());
                }
                if degree > 32 {
                    return Err(format!(
                        "sphere smooth max_degree={} is too large for the dense harmonic engine (limit 32)",
                        degree
                    ));
                }
                Some(degree)
            } else {
                None
            };
            let center_strategy = if matches!(method, SphereMethod::Wahba) {
                let centers = parse_countwith_basis_alias(
                    options,
                    "centers",
                    default_num_centers(ds.values.nrows(), cols.len()),
                )?;
                CenterStrategy::FarthestPoint {
                    num_centers: centers,
                }
            } else {
                CenterStrategy::FarthestPoint { num_centers: 0 }
            };
            Ok(SmoothBasisSpec::Sphere {
                feature_cols: cols.to_vec(),
                spec: SphericalSplineBasisSpec {
                    center_strategy,
                    penalty_order: option_usize(options, "penalty_order")
                        .or_else(|| option_usize(options, "m"))
                        .unwrap_or(2),
                    double_penalty: smooth_double_penalty,
                    radians,
                    method,
                    max_degree,
                    wahba_kernel,
                    streaming_chunk_size,
                },
            })
        }
        "matern" => {
            // Catch typos like `lengt_scale=` / `nyu=` / `centerz=` before
            // they get silently ignored and the user wonders why their
            // option had no effect. The matern() term accepts exactly
            // these options.
            validate_known_options(
                "matern",
                options,
                &[
                    "type",
                    "bs",
                    "by",
                    "nu",
                    "length_scale",
                    "centers",
                    "k",
                    "basis_dim",
                    "basis-dim",
                    "basisdim",
                    "knots",
                    "include_intercept",
                    "double_penalty",
                    "by",
                    "id",
                    "__by_col",
                    "identifiability",
                    "by",
                    "scale_dims",
                    "streaming_chunk_size",
                    "chunk_size",
                ],
            )?;
            let plan = plan_spatial_basis(
                ds.values.nrows(),
                cols.len(),
                CenterCountRequest::Default,
                DuchonNullspaceOrder::Zero,
                option_bool(options, "scale_dims").unwrap_or(false),
                policy,
            )
            .map_err(|e| e.to_string())?;
            let centers = parse_countwith_basis_alias(options, "centers", plan.centers)?;
            let center_strategy = if has_explicit_countwith_basis_alias(options, "centers") {
                spatial_center_strategy_for_dimension(centers, cols.len())
            } else {
                auto_spatial_center_strategy(centers, cols.len())
            };
            let nu = parse_matern_nu(options.get("nu").map(String::as_str).unwrap_or("5/2"))?;
            // The exponential (ν = 1/2) Matérn kernel has a singular Laplacian
            // at zero in d ≥ 2, so the operator-collocation penalty machinery
            // hits a non-invertible matrix during fit. Surface the cause
            // up-front instead of letting the user see the generic
            // "Matrix conditioning issue detected" wrapper from PIRLS.
            if matches!(nu, MaternNu::Half) && cols.len() >= 2 {
                return Err(TermBuilderError::unsupported_feature(format!(
                    "matern() with nu=1/2 is not supported for d>=2 (got {} covariates): \
                     the exponential kernel's Laplacian is singular at center collisions, \
                     which makes the operator-collocation penalty non-invertible. \
                     Choose nu>=3/2 (e.g. nu=3/2 or the default nu=5/2) for multi-dimensional smooths.",
                    cols.len()
                ))
                .to_string());
            }
            let aniso_log_scales = if option_bool(options, "scale_dims").unwrap_or(false) {
                Some(vec![0.0; cols.len()])
            } else {
                None
            };
            Ok(SmoothBasisSpec::Matern {
                feature_cols: cols.to_vec(),
                spec: MaternBasisSpec {
                    center_strategy,
                    periodic: parse_periodic_axes_option(options, cols.len())?,
                    length_scale: option_f64(options, "length_scale").unwrap_or(1.0),
                    nu,
                    include_intercept: option_bool(options, "include_intercept").unwrap_or(false),
                    double_penalty: smooth_double_penalty,
                    identifiability: parse_matern_identifiability(options)
                        .map_err(|e| e.to_string())?,
                    aniso_log_scales,
                    streaming_chunk_size: option_usize(options, "streaming_chunk_size")
                        .or_else(|| option_usize(options, "chunk_size")),
                },
                input_scales: None,
            })
        }
        "duchon" => {
            validate_known_options(
                "duchon",
                options,
                &[
                    "type",
                    "bs",
                    "by",
                    "length_scale",
                    "centers",
                    "k",
                    "basis_dim",
                    "basis-dim",
                    "basisdim",
                    "knots",
                    "power",
                    "p",
                    "nullspace_order",
                    "order",
                    "identifiability",
                    "by",
                    "periodic",
                    "cyclic",
                    "period",
                    "period_start",
                    "period_end",
                    "scale_dims",
                    "double_penalty",
                    "by",
                    "id",
                    "__by_col",
                ],
            )?;
            if options.contains_key("double_penalty") {
                return Err(TermBuilderError::incompatible_config(format!(
                    "Duchon smooth '{}' does not support double_penalty; Duchon uses mass, tension, and stiffness operator penalties.",
                    vars.join(", ")
                ))
                .to_string());
            }
            let requested_nullspace_order = parse_duchon_order(options)?;
            let length_scale = option_f64_strict(options, "length_scale")?;
            // Resolve `(nullspace_order, power)` against the joint constraints
            // (operator collocation + scale-free CPD). Explicit power keeps the
            // user's nullspace as-is (validator will reject inconsistent combos);
            // the policy path may auto-escalate the nullspace order in scale-free mode
            // when CPD requires a richer polynomial absorption space.
            let (nullspace_order, power) = match parse_duchon_power_policy(options)? {
                DuchonPowerPolicy::Explicit(req_power) => {
                    // Honor the user's nullspace_order, but auto-escalate
                    // explicit power when it sits below the minimum needed
                    // for the active operator-penalty triple
                    // (mass + tension + stiffness ⇒ D2 collocation requires
                    // 2(p+s) > d+2). Without this escalation the basis
                    // builder later rejects the user's combination with an
                    // opaque "Duchon D2 collocation requires …" diagnostic
                    // even though the requested kernel exists — the user's
                    // power was simply too low for the *operator* derivative
                    // order, not for the kernel itself. Only escalate when
                    // CPD does not also force a different nullspace order;
                    // when CPD bumps the nullspace, the explicit power is
                    // bound to the user-requested order, so leave it alone
                    // and let the kernel validator emit a precise
                    // diagnostic for that combination.
                    let (resolved_nullspace, min_admissible_power) = resolve_duchon_orders(
                        cols.len(),
                        requested_nullspace_order,
                        2,
                        length_scale,
                    );
                    let final_power = if resolved_nullspace == requested_nullspace_order {
                        req_power.max(min_admissible_power)
                    } else {
                        req_power
                    };
                    if final_power != req_power {
                        inference_notes.push(format!(
                            "Note: explicit Duchon power={} is below the minimum admissible \
                             power={} for D2 (stiffness) collocation at dimension={}, \
                             nullspace_order={:?} (requires 2(p+s) > d+2). Auto-escalated \
                             to power={} so all three Duchon operator penalties (mass, \
                             tension, stiffness) remain active.",
                            req_power,
                            min_admissible_power,
                            cols.len(),
                            requested_nullspace_order,
                            final_power,
                        ));
                    }
                    (requested_nullspace_order, final_power)
                }
                DuchonPowerPolicy::MinimumAdmissibleForTripleOperator => {
                    let resolved = resolve_duchon_orders(
                        cols.len(),
                        requested_nullspace_order,
                        2,
                        length_scale,
                    );
                    if resolved.0 != requested_nullspace_order {
                        inference_notes.push(format!(
                            "Note: scale-free Duchon CPD against polynomial nullspace requires order ≥ {:?} \
                             at dimension {} (Wendland 8.17, 2s < d); auto-escalated from {:?}. \
                             Specify length_scale=... to use the hybrid Duchon-Matern kernel.",
                            resolved.0,
                            cols.len(),
                            requested_nullspace_order,
                        ));
                    }
                    resolved
                }
            };
            let plan = plan_spatial_basis(
                ds.values.nrows(),
                cols.len(),
                CenterCountRequest::Default,
                nullspace_order,
                option_bool(options, "scale_dims").unwrap_or(false),
                policy,
            )
            .map_err(|e| e.to_string())?;
            let centers_explicit = has_explicit_countwith_basis_alias(options, "centers");
            let requested_centers = parse_countwith_basis_alias(options, "centers", plan.centers)?;
            let polynomial_cols = match nullspace_order {
                DuchonNullspaceOrder::Zero => 1,
                DuchonNullspaceOrder::Linear => cols.len() + 1,
                DuchonNullspaceOrder::Degree(degree) => {
                    crate::basis::duchon_nullspace_dimension(cols.len(), degree)
                }
            };
            if requested_centers <= polynomial_cols {
                return Err(TermBuilderError::incompatible_config(format!(
                    "Duchon smooth '{}' requested basis dimension {} but order={:?} in {}D needs {} polynomial null-space columns; choose centers/k > {}",
                    vars.join(", "),
                    requested_centers,
                    nullspace_order,
                    cols.len(),
                    polynomial_cols,
                    polynomial_cols,
                ))
                .to_string());
            }
            let mut centers = requested_centers;
            if !centers_explicit && ds.values.nrows() <= 32 && smooth_coordinate_count >= 5 {
                centers = centers.max(polynomial_cols + 4);
            }
            let center_strategy = if centers_explicit {
                spatial_center_strategy_for_dimension(centers, cols.len())
            } else {
                auto_spatial_center_strategy(centers, cols.len())
            };
            let aniso_log_scales = if option_bool(options, "scale_dims").unwrap_or(false) {
                Some(vec![0.0; cols.len()])
            } else {
                None
            };
            let operator_penalties = DuchonOperatorPenaltySpec::default();
            Ok(SmoothBasisSpec::Duchon {
                feature_cols: cols.to_vec(),
                spec: DuchonBasisSpec {
                    center_strategy,
                    periodic: parse_periodic_axes_option(options, cols.len())?,
                    length_scale,
                    power: power as f64,
                    nullspace_order,
                    identifiability: parse_spatial_identifiability(options)
                        .map_err(|e| e.to_string())?,
                    aniso_log_scales,
                    operator_penalties,
                    boundary: if cols.len() == 1 {
                        let c = cols[0];
                        let (minv, maxv) = col_minmax(ds.values.column(c))?;
                        parse_cyclic_boundary(options, minv, maxv)?
                    } else {
                        OneDimensionalBoundary::Open
                    },
                },
                input_scales: None,
            })
        }
        "pca" => {
            validate_known_options(
                "pca",
                options,
                &[
                    "type",
                    "bs",
                    "by",
                    "k",
                    "basis_dim",
                    "basis-dim",
                    "basisdim",
                    "lazy_path",
                    "path",
                    "pca_basis_path",
                    "chunk_size",
                    "smooth_penalty",
                    "centered",
                    "double_penalty",
                    "id",
                    "__by_col",
                ],
            )?;
            let path = options
                .get("lazy_path")
                .or_else(|| options.get("pca_basis_path"))
                .or_else(|| options.get("path"))
                .map(|raw| PathBuf::from(strip_quotes(raw)));
            let Some(path) = path else {
                return Err(TermBuilderError::incompatible_config(
                    "pca smooth requires lazy_path=... on the formula path",
                )
                .to_string());
            };
            let k = option_usize_any(options, &["k", "basis_dim", "basis-dim", "basisdim"])
                .unwrap_or(0);
            let chunk_size = option_usize(options, "chunk_size").unwrap_or(4096);
            Ok(SmoothBasisSpec::Pca {
                feature_cols: cols.to_vec(),
                basis_matrix: Array2::<f64>::zeros((cols.len(), k)),
                centered: option_bool(options, "centered").unwrap_or(true),
                smooth_penalty: option_f64(options, "smooth_penalty").unwrap_or(1.0),
                center_mean: None,
                pca_basis_path: Some(path),
                chunk_size,
            })
        }
        other => Err(TermBuilderError::unsupported_feature(format!(
            "unsupported smooth type '{other}'"
        ))
        .to_string()),
    }
}

/// Initialise per-axis anisotropic log-scales on eligible spatial smooth specs.
pub fn enable_scale_dimensions(spec: &mut TermCollectionSpec) {
    for smooth in spec.smooth_terms.iter_mut() {
        match &mut smooth.basis {
            SmoothBasisSpec::Matern {
                feature_cols,
                spec: matern,
                ..
            } => {
                if matern.aniso_log_scales.is_none() {
                    let d = feature_cols.len();
                    matern.aniso_log_scales = Some(vec![0.0; d]);
                }
            }
            SmoothBasisSpec::Duchon {
                feature_cols,
                spec: duchon,
                ..
            } => {
                if duchon.aniso_log_scales.is_none() {
                    let d = feature_cols.len();
                    duchon.aniso_log_scales = Some(vec![0.0; d]);
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Data-aware helpers
// ---------------------------------------------------------------------------

pub fn spatial_center_strategy_for_dimension(num_centers: usize, d: usize) -> CenterStrategy {
    default_spatial_center_strategy(num_centers, d)
}

pub fn col_minmax(col: ArrayView1<'_, f64>) -> Result<(f64, f64), String> {
    let min = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    if !min.is_finite() || !max.is_finite() {
        return Err(TermBuilderError::degenerate_data(
            "non-finite data encountered while inferring knot range",
        )
        .to_string());
    }
    if (max - min).abs() < 1e-12 {
        Ok((min, min + 1e-6))
    } else {
        Ok((min, max))
    }
}

/// Default knot count for an n-row design, growing the ceiling with n^(1/3).
///
/// Wood (2017, sec 5.5) shows the optimal smoothing-spline rank for an
/// asymptotically smooth target grows as n^(1/(2k+1)) where k is the order
/// of the penalty (k=2 cubic ⇒ n^(1/5)) and as n^(1/3) under a fixed-knot
/// regression-spline regime (k → ∞ in the rank-bias trade-off). Capping the
/// floor heuristic `floor(sqrt(n))` at 30 strangles biobank fits at n = 3e5
/// where the underlying signal can support ~60 knots without overfitting.
/// We retain the sqrt() floor as the starting point, raise the ceiling with
/// n^(1/3) so it crosses 30 around n = 27_000 and reaches ~100 at n = 1e6,
/// and keep the lower bound at 6 to guarantee enough degrees of freedom for
/// monotone constraints.
pub fn heuristic_knots(n: usize) -> usize {
    let n_f = n as f64;
    let base = n_f.sqrt() as usize;
    let n_cbrt = n_f.cbrt();
    // Ceiling grows with n^(1/3): 30 at n ≈ 27k, 60 at n ≈ 216k, 100 at n ≈ 1M.
    let ceiling = (n_cbrt as usize).max(30);
    base.clamp(6, ceiling)
}

pub fn unique_count_column(col: ArrayView1<'_, f64>) -> usize {
    use std::collections::HashSet;
    let mut set = HashSet::<u64>::with_capacity(col.len());
    for &v in col {
        let norm = if v == 0.0 { 0.0 } else { v };
        set.insert(norm.to_bits());
    }
    set.len().max(1)
}

/// Per-column knot count from the unique-value count, with the same n^(1/3)
/// ceiling growth as `heuristic_knots` so per-column smooths can support more
/// detail at biobank scale. The 4-knot floor stays put because we still need
/// enough basis functions to fit a non-trivial smooth at all.
pub fn heuristic_knots_for_column(col: ArrayView1<'_, f64>) -> usize {
    let unique = unique_count_column(col);
    let ceiling = ((unique as f64).cbrt() as usize).max(20);
    (unique / 4).clamp(4, ceiling)
}

pub fn heuristic_centers(n: usize, d: usize) -> usize {
    default_num_centers(n, d)
}

// ---------------------------------------------------------------------------
// Smooth option parsers
// ---------------------------------------------------------------------------

fn parse_endpoint_side(
    value: &str,
    context: &str,
) -> Result<BSplineEndpointBoundaryCondition, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "none" | "open" | "unconstrained" => Ok(BSplineEndpointBoundaryCondition::Free),
        "clamped" | "clamp" | "zero_derivative" | "zero-derivative" => {
            Ok(BSplineEndpointBoundaryCondition::Clamped)
        }
        "anchored" | "anchor" | "zero" | "zero_value" | "zero-value" => {
            Ok(BSplineEndpointBoundaryCondition::Anchored { value: 0.0 })
        }
        other => Err(format!(
            "unsupported {context} boundary condition '{other}'; expected none, clamped, or anchored"
        )),
    }
}

fn boundary_anchor_value(
    options: &BTreeMap<String, String>,
    side: &str,
    fallback: Option<f64>,
) -> Option<f64> {
    [
        format!("anchor_{side}"),
        format!("{side}_anchor"),
        format!("anchor-value-{side}"),
    ]
    .iter()
    .find_map(|key| option_f64(options, key))
    .or(fallback)
}

fn apply_anchor_value(
    cond: BSplineEndpointBoundaryCondition,
    value: Option<f64>,
) -> BSplineEndpointBoundaryCondition {
    match cond {
        BSplineEndpointBoundaryCondition::Anchored { .. } => {
            BSplineEndpointBoundaryCondition::Anchored {
                value: value.unwrap_or(0.0),
            }
        }
        other => other,
    }
}

fn parse_bspline_boundary_conditions(
    options: &BTreeMap<String, String>,
) -> Result<BSplineBoundaryConditions, String> {
    let fallback_anchor = option_f64(options, "anchor")
        .or_else(|| option_f64(options, "anchor_value"))
        .or_else(|| option_f64(options, "value"));
    let global_boundary_conditions = options
        .get("boundary_conditions")
        .or_else(|| options.get("bc"));
    let mut boundary_conditions = BSplineBoundaryConditions::default();

    if let Some(raw_boundary_conditions) = global_boundary_conditions {
        let cond = parse_endpoint_side(raw_boundary_conditions, "boundary_conditions")?;
        let side = options
            .get("side")
            .map(|s| s.trim().to_ascii_lowercase())
            .unwrap_or_else(|| "both".to_string());
        match side.as_str() {
            "both" | "all" | "endpoints" => {
                boundary_conditions.left = cond;
                boundary_conditions.right = cond;
            }
            "left" | "start" | "lower" => boundary_conditions.left = cond,
            "right" | "end" | "upper" => boundary_conditions.right = cond,
            other => {
                return Err(format!(
                    "unsupported B-spline boundary side '{other}'; expected left, right, or both"
                ));
            }
        }
    }

    if let Some(raw) = options
        .get("bc_left")
        .or_else(|| options.get("left_bc"))
        .or_else(|| options.get("bc_start"))
        .or_else(|| options.get("start_bc"))
    {
        boundary_conditions.left = parse_endpoint_side(raw, "left endpoint")?;
    }
    if let Some(raw) = options
        .get("bc_right")
        .or_else(|| options.get("right_bc"))
        .or_else(|| options.get("bc_end"))
        .or_else(|| options.get("end_bc"))
    {
        boundary_conditions.right = parse_endpoint_side(raw, "right endpoint")?;
    }

    boundary_conditions.left = apply_anchor_value(
        boundary_conditions.left,
        boundary_anchor_value(options, "left", fallback_anchor),
    );
    boundary_conditions.right = apply_anchor_value(
        boundary_conditions.right,
        boundary_anchor_value(options, "right", fallback_anchor),
    );

    Ok(boundary_conditions)
}

pub fn parse_ps_internal_knots(
    options: &BTreeMap<String, String>,
    degree: usize,
    default_internal_knots: usize,
) -> Result<(usize, bool), String> {
    const MIN_EXPRESSIVE_INTERNAL_KNOTS: usize = 2;
    // Strict variants: reject `k=-1`, `k=1.5`, `knots=-2` etc. with a
    // focused error instead of silently dropping the value and using the
    // default. Lenient `option_usize` / `option_usize_any` silently swallow
    // unparseable values, which leaves the user thinking they configured
    // something when they did not.
    let knots_internal = option_usize_strict(options, "knots")?;
    let basis_dim = option_usize_any_strict(options, &["k", "basis_dim", "basis-dim", "basisdim"])?;
    if knots_internal.is_some() && basis_dim.is_some() {
        return Err(TermBuilderError::incompatible_config(
            "ps/bspline smooth: specify either knots=<internal_knots> or k=<basis_dim> (not both)",
        )
        .to_string());
    }
    if let Some(k) = basis_dim {
        let min_k = degree + 1;
        if k < min_k {
            return Err(TermBuilderError::invalid_option(format!(
                "ps/bspline smooth: k={} too small for degree {}; expected k >= {}",
                k, degree, min_k
            ))
            .to_string());
        }
        Ok(((k - min_k).max(MIN_EXPRESSIVE_INTERNAL_KNOTS), false))
    } else {
        Ok((
            knots_internal.unwrap_or(default_internal_knots),
            knots_internal.is_none(),
        ))
    }
}

/// Reject unknown option keys with a focused error that names the term and
/// the offending key, plus suggests near-matches from the known-key list.
/// Without this, typos like `lengt_scale=0.1` or `nyu=5/2` are silently
/// dropped, the term uses the default, and the user has no idea why their
/// option had no effect.
pub fn validate_known_options(
    term_name: &str,
    options: &BTreeMap<String, String>,
    known: &[&str],
) -> Result<(), String> {
    let known_set: std::collections::BTreeSet<&&str> = known.iter().collect();
    for key in options.keys() {
        if !known_set.contains(&key.as_str()) {
            // Suggest near-matches (substring or shared prefix ≥ 3).
            let key_l = key.to_ascii_lowercase();
            let mut suggestions: Vec<&str> = known
                .iter()
                .filter(|k| {
                    let kl = k.to_ascii_lowercase();
                    kl.contains(&key_l) || key_l.contains(&kl) || {
                        let n = kl
                            .chars()
                            .zip(key_l.chars())
                            .take_while(|(a, b)| a == b)
                            .count();
                        n >= 3
                    }
                })
                .copied()
                .collect();
            suggestions.sort_unstable();
            suggestions.dedup();
            let hint = if suggestions.is_empty() {
                String::new()
            } else {
                format!(" — did you mean one of [{}]?", suggestions.join(", "))
            };
            return Err(TermBuilderError::invalid_option(format!(
                "{term_name}() does not accept option `{key}`{hint} Known options: [{}]",
                {
                    let mut sorted = known.to_vec();
                    sorted.sort_unstable();
                    sorted.join(", ")
                }
            ))
            .to_string());
        }
    }
    Ok(())
}

pub fn parse_countwith_basis_alias(
    options: &BTreeMap<String, String>,
    primarykey: &str,
    default_count: usize,
) -> Result<usize, String> {
    // Strict: reject unparseable values (e.g. `centers=many`, `centers=-1`,
    // `centers=1.5`) instead of silently dropping them and falling through
    // to the default. Without this the user gets the auto-inferred count
    // silently and never realizes their explicit option was ignored.
    let primary = option_usize_strict(options, primarykey)?;
    let basis_dim = option_usize_any_strict(
        options,
        &["k", "basis_dim", "basis-dim", "basisdim", "knots"],
    )?;
    if primary.is_some() && basis_dim.is_some() {
        return Err(TermBuilderError::incompatible_config(format!(
            "specify either {}=<count> or k=<basis_dim> (not both)",
            primarykey
        ))
        .to_string());
    }
    Ok(primary.or(basis_dim).unwrap_or(default_count))
}

fn has_explicit_countwith_basis_alias(
    options: &BTreeMap<String, String>,
    primarykey: &str,
) -> bool {
    options.contains_key(primarykey)
        || ["k", "basis_dim", "basis-dim", "basisdim", "knots"]
            .iter()
            .any(|alias| options.contains_key(*alias))
}

pub fn parse_cyclic_boundary(
    options: &BTreeMap<String, String>,
    minv: f64,
    maxv: f64,
) -> Result<OneDimensionalBoundary, String> {
    let cyclic = option_bool(options, "cyclic")
        .or_else(|| option_bool(options, "periodic"))
        .unwrap_or(false);
    if !cyclic {
        return Ok(OneDimensionalBoundary::Open);
    }
    let start = option_f64(options, "period_start")
        .or_else(|| option_f64(options, "start"))
        .unwrap_or(minv);
    let end = option_f64(options, "period_end")
        .or_else(|| option_f64(options, "end"))
        .unwrap_or(maxv);
    if end <= start {
        return Err(format!(
            "cyclic smooth requires period_end/end ({end}) > period_start/start ({start})"
        ));
    }
    Ok(OneDimensionalBoundary::Cyclic { start, end })
}

/// Parse the periodic-uniform domain for a one-dimensional cyclic smooth.
///
/// Returns the `(domain_start, period)` pair derived from
/// `period_start` / `start`, `period_end` / `end`, falling back to the
/// data range `[minv, maxv)` when neither bound is provided. The period
/// must be strictly positive.
pub fn parse_periodic_domain_1d(
    options: &BTreeMap<String, String>,
    minv: f64,
    maxv: f64,
) -> Result<(f64, f64), String> {
    let start = option_f64(options, "period_start")
        .or_else(|| option_f64(options, "start"))
        .unwrap_or(minv);
    let end = option_f64(options, "period_end")
        .or_else(|| option_f64(options, "end"))
        .unwrap_or(maxv);
    if !(start.is_finite() && end.is_finite()) {
        return Err(format!(
            "periodic smooth domain requires finite endpoints, got ({start}, {end})"
        ));
    }
    if end <= start {
        return Err(format!(
            "periodic smooth requires period_end/end ({end}) > period_start/start ({start})"
        ));
    }
    Ok((start, end - start))
}

fn parse_matern_nu(raw: &str) -> Result<MaternNu, String> {
    let trimmed = raw.trim();
    let lowered = trimmed.to_ascii_lowercase();
    match lowered.as_str() {
        "1/2" | "0.5" | "half" => return Ok(MaternNu::Half),
        "3/2" | "1.5" => return Ok(MaternNu::ThreeHalves),
        "5/2" | "2.5" => return Ok(MaternNu::FiveHalves),
        "7/2" | "3.5" => return Ok(MaternNu::SevenHalves),
        "9/2" | "4.5" => return Ok(MaternNu::NineHalves),
        _ => {}
    }

    let value = if let Some((num, den)) = trimmed.split_once('/') {
        let num = num
            .trim()
            .parse::<f64>()
            .map_err(|err| format!("{}: {err}", unsupported_matern_nu_message(raw)))?;
        let den = den
            .trim()
            .parse::<f64>()
            .map_err(|err| format!("{}: {err}", unsupported_matern_nu_message(raw)))?;
        if den == 0.0 || !num.is_finite() || !den.is_finite() {
            return Err(unsupported_matern_nu_message(raw));
        }
        num / den
    } else {
        trimmed
            .parse::<f64>()
            .map_err(|err| format!("{}: {err}", unsupported_matern_nu_message(raw)))?
    };

    const TOL: f64 = 1e-12;
    if (value - 0.5).abs() <= TOL {
        Ok(MaternNu::Half)
    } else if (value - 1.5).abs() <= TOL {
        Ok(MaternNu::ThreeHalves)
    } else if (value - 2.5).abs() <= TOL {
        Ok(MaternNu::FiveHalves)
    } else if (value - 3.5).abs() <= TOL {
        Ok(MaternNu::SevenHalves)
    } else if (value - 4.5).abs() <= TOL {
        Ok(MaternNu::NineHalves)
    } else {
        Err(unsupported_matern_nu_message(raw))
    }
}

fn unsupported_matern_nu_message(raw: &str) -> String {
    TermBuilderError::unsupported_feature(format!(
        "unsupported Matern nu '{raw}'; supported half-integer values are 1/2, 3/2, 5/2, 7/2, and 9/2"
    ))
    .to_string()
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum DuchonPowerPolicy {
    Explicit(usize),
    MinimumAdmissibleForTripleOperator,
}

pub fn parse_duchon_power_policy(
    options: &BTreeMap<String, String>,
) -> Result<DuchonPowerPolicy, String> {
    if let Some(raw_nu) = options.get("nu") {
        return Err(TermBuilderError::incompatible_config(format!(
            "Duchon smooths use power=<integer>, not nu='{}'. Use power=0, power=1, etc.",
            raw_nu
        ))
        .to_string());
    }
    match options.get("power") {
        Some(raw) => raw.parse::<usize>().map(DuchonPowerPolicy::Explicit).map_err(|err| {
            TermBuilderError::invalid_option(format!(
                "invalid Duchon power '{}'; expected a non-negative integer such as power=0 or power=1: {}",
                raw, err
            ))
            .to_string()
        }),
        None => Ok(DuchonPowerPolicy::MinimumAdmissibleForTripleOperator),
    }
}

pub fn parse_duchon_power(options: &BTreeMap<String, String>) -> Result<usize, String> {
    match parse_duchon_power_policy(options)? {
        DuchonPowerPolicy::Explicit(power) => Ok(power),
        DuchonPowerPolicy::MinimumAdmissibleForTripleOperator => Ok(2),
    }
}

pub fn parse_duchon_order(
    options: &BTreeMap<String, String>,
) -> Result<DuchonNullspaceOrder, String> {
    match options.get("order") {
        None => Ok(DuchonNullspaceOrder::Zero),
        Some(raw) => match raw.parse::<usize>() {
            Ok(0) => Ok(DuchonNullspaceOrder::Zero),
            Ok(1) => Ok(DuchonNullspaceOrder::Linear),
            Ok(other) => Ok(DuchonNullspaceOrder::Degree(other)),
            Err(_) => Err(TermBuilderError::invalid_option(format!(
                "invalid Duchon order '{}'; expected a non-negative integer such as order=0, order=1, or order=2",
                raw
            ))
            .to_string()),
        },
    }
}

fn parse_matern_identifiability(
    options: &BTreeMap<String, String>,
) -> Result<MaternIdentifiability, TermBuilderError> {
    let Some(raw) = options.get("identifiability").map(String::as_str) else {
        return Ok(MaternIdentifiability::default());
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(MaternIdentifiability::None),
        "sum_tozero" | "sum-to-zero" | "center_sum_tozero" | "center-sum-to-zero" | "centered" => {
            Ok(MaternIdentifiability::CenterSumToZero)
        }
        "linear" | "center_linear_orthogonal" | "center-linear-orthogonal" => {
            Ok(MaternIdentifiability::CenterLinearOrthogonal)
        }
        other => Err(TermBuilderError::unsupported_feature(format!(
            "invalid Matérn identifiability '{other}'; expected one of: none, sum_tozero, linear"
        ))),
    }
}

fn parse_spatial_identifiability(
    options: &BTreeMap<String, String>,
) -> Result<SpatialIdentifiability, TermBuilderError> {
    let Some(raw) = options.get("identifiability").map(String::as_str) else {
        return Ok(SpatialIdentifiability::default());
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(SpatialIdentifiability::None),
        "orthogonal"
        | "orthogonal_to_parametric"
        | "orthogonal-to-parametric"
        | "parametric_orthogonal" => Ok(SpatialIdentifiability::OrthogonalToParametric),
        "frozen" => Err(TermBuilderError::unsupported_feature(
            "spatial identifiability 'frozen' is internal-only; use none or orthogonal_to_parametric",
        )),
        other => Err(TermBuilderError::unsupported_feature(format!(
            "invalid spatial identifiability '{other}'; expected one of: none, orthogonal_to_parametric"
        ))),
    }
}

fn parse_tensor_identifiability(
    options: &BTreeMap<String, String>,
) -> Result<TensorBSplineIdentifiability, TermBuilderError> {
    let Some(raw) = options.get("identifiability").map(String::as_str) else {
        return Ok(TensorBSplineIdentifiability::default());
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(TensorBSplineIdentifiability::None),
        "sum_tozero" | "sum-to-zero" | "centered" => Ok(TensorBSplineIdentifiability::SumToZero),
        "frozen" | "frozen_transform" | "frozen-transform" => {
            Err(TermBuilderError::unsupported_feature(
                "tensor identifiability 'frozen' is internal-only; use none or sum_tozero",
            ))
        }
        other => Err(TermBuilderError::unsupported_feature(format!(
            "invalid tensor identifiability '{other}'; expected one of: none, sum_tozero"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::formula_dsl::parse_formula;
    use crate::inference::model::{DataSchema, SchemaColumn};
    use ndarray::Array2;
    use std::collections::BTreeMap;

    fn continuous_dataset(headers: &[&str], rows: Vec<Vec<f64>>) -> Dataset {
        let nrows = rows.len();
        let ncols = headers.len();
        let values = Array2::from_shape_vec(
            (nrows, ncols),
            rows.into_iter().flat_map(|row| row.into_iter()).collect(),
        )
        .expect("rectangular test data");
        Dataset {
            headers: headers.iter().map(|name| name.to_string()).collect(),
            values,
            schema: DataSchema {
                columns: headers
                    .iter()
                    .map(|name| SchemaColumn {
                        name: name.to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    })
                    .collect(),
            },
            column_kinds: vec![ColumnKindTag::Continuous; ncols],
        }
    }

    fn inferred_tensor_basis_product(ds: &Dataset) -> usize {
        let parsed = parse_formula("y ~ te(theta, h)").expect("parse tensor formula");
        let col_map = ds.column_map();
        let mut notes = Vec::new();
        let terms = build_termspec(
            &parsed.terms,
            ds,
            &col_map,
            &mut notes,
            &ResourcePolicy::default_library(),
        )
        .expect("build tensor termspec");
        let SmoothBasisSpec::TensorBSpline { spec, .. } = &terms.smooth_terms[0].basis else {
            panic!("expected tensor smooth");
        };
        spec.marginalspecs
            .iter()
            .map(|marginal| match marginal.knotspec {
                BSplineKnotSpec::Generate {
                    num_internal_knots, ..
                } => num_internal_knots + marginal.degree + 1,
                BSplineKnotSpec::PeriodicUniform { num_basis, .. } => num_basis,
                BSplineKnotSpec::Automatic {
                    num_internal_knots: Some(num_internal_knots),
                    ..
                } => num_internal_knots + marginal.degree + 1,
                BSplineKnotSpec::Automatic {
                    num_internal_knots: None,
                    ..
                } => panic!("test helper cannot infer automatic knot count"),
                BSplineKnotSpec::Provided(ref knots) => {
                    knots.len().saturating_sub(marginal.degree + 1)
                }
            })
            .product()
    }

    #[test]
    fn parse_cylinder_periodic_options_match_requested_forms() {
        let mut opts = BTreeMap::new();
        opts.insert("periodic".to_string(), "[0]".to_string());
        opts.insert("period".to_string(), "[2*pi, None]".to_string());
        let axes = parse_periodic_axes(&opts, 2).expect("axes");
        let periods = parse_periods(&opts, &axes).expect("periods");
        assert_eq!(axes, vec![true, false]);
        assert!((periods[0].unwrap() - 2.0 * std::f64::consts::PI).abs() < 1e-12);
        assert_eq!(periods[1], None);

        let mut boundary_opts = BTreeMap::new();
        boundary_opts.insert(
            "boundary".to_string(),
            "['periodic', 'natural']".to_string(),
        );
        boundary_opts.insert("period".to_string(), "[2*pi, None]".to_string());
        let boundary_axes = parse_periodic_axes(&boundary_opts, 2).expect("boundary axes");
        let boundary_periods =
            parse_periods(&boundary_opts, &boundary_axes).expect("boundary periods");
        assert_eq!(boundary_axes, vec![true, false]);
        assert!((boundary_periods[0].unwrap() - 2.0 * std::f64::consts::PI).abs() < 1e-12);
        assert_eq!(boundary_periods[1], None);

        let mut unicode_opts = BTreeMap::new();
        unicode_opts.insert("periodic".to_string(), "[0,1]".to_string());
        unicode_opts.insert("period".to_string(), "[2π, τ]".to_string());
        let unicode_axes = parse_periodic_axes(&unicode_opts, 2).expect("unicode axes");
        let unicode_periods = parse_periods(&unicode_opts, &unicode_axes).expect("unicode periods");
        assert_eq!(unicode_axes, vec![true, true]);
        assert!((unicode_periods[0].unwrap() - 2.0 * std::f64::consts::PI).abs() < 1e-12);
        assert!((unicode_periods[1].unwrap() - std::f64::consts::TAU).abs() < 1e-12);
    }

    #[test]
    fn parse_single_axis_periodic_zero_as_axis_not_false() {
        let mut opts = BTreeMap::new();
        opts.insert("periodic".to_string(), "[0]".to_string());
        opts.insert("period".to_string(), "2*pi".to_string());
        opts.insert("origin".to_string(), "0".to_string());
        let axes = parse_periodic_axes(&opts, 1).expect("axes");
        let periods = parse_periods(&opts, &axes).expect("periods");
        let origins = parse_period_origins(&opts, &axes).expect("origins");
        assert_eq!(axes, vec![true]);
        assert!((periods[0].unwrap() - 2.0 * std::f64::consts::PI).abs() < 1e-12);
        assert_eq!(origins[0], Some(0.0));
    }

    #[test]
    fn one_dimensional_bspline_accepts_boundary_periodic() {
        let ds = continuous_dataset(
            &["y", "theta"],
            (0..16)
                .map(|i| {
                    let theta = std::f64::consts::TAU * i as f64 / 16.0;
                    vec![theta.sin(), theta]
                })
                .collect(),
        );
        let parsed = parse_formula("y ~ s(theta, boundary=periodic, period=2*pi, origin=0, k=8)")
            .expect("parse");
        let col_map = ds.column_map();
        let mut notes = Vec::new();
        let terms = build_termspec(
            &parsed.terms,
            &ds,
            &col_map,
            &mut notes,
            &crate::resource::ResourcePolicy::default_library(),
        )
        .expect("periodic boundary should build");
        let SmoothBasisSpec::BSpline1D { spec, .. } = &terms.smooth_terms[0].basis else {
            panic!("expected 1D B-spline");
        };
        assert!(matches!(
            &spec.knotspec,
            BSplineKnotSpec::PeriodicUniform {
                data_range,
                num_basis: 8
            } if *data_range == (0.0, std::f64::consts::TAU)
        ));
    }

    #[test]
    fn default_sphere_smooth_uses_spherical_farthest_point_centers() {
        let ds = continuous_dataset(
            &["y", "lat", "lon"],
            (0..24)
                .map(|i| {
                    let t = i as f64 / 24.0;
                    let lat = -60.0 + 120.0 * t;
                    let lon = -180.0 + 360.0 * ((7 * i) % 24) as f64 / 24.0;
                    vec![lat.to_radians().sin(), lat, lon]
                })
                .collect(),
        );
        let parsed = parse_formula("y ~ sphere(lat, lon)").expect("parse");
        let col_map = ds.column_map();
        let mut notes = Vec::new();
        let terms = build_termspec(
            &parsed.terms,
            &ds,
            &col_map,
            &mut notes,
            &crate::resource::ResourcePolicy::default_library(),
        )
        .expect("build sphere termspec");
        let SmoothBasisSpec::Sphere { spec, .. } = &terms.smooth_terms[0].basis else {
            panic!("expected sphere term");
        };
        assert!(matches!(
            spec.center_strategy,
            CenterStrategy::FarthestPoint { .. }
        ));
    }

    #[test]
    fn one_dimensional_duchon_defaults_to_scale_free_length_scale() {
        let ds = continuous_dataset(
            &["y", "x"],
            (0..32)
                .map(|i| {
                    let x = i as f64 / 31.0;
                    vec![(std::f64::consts::TAU * x).sin(), x]
                })
                .collect(),
        );
        let parsed = parse_formula("y ~ duchon(x)").expect("parse");
        let col_map = ds.column_map();
        let mut notes = Vec::new();
        let terms = build_termspec(
            &parsed.terms,
            &ds,
            &col_map,
            &mut notes,
            &crate::resource::ResourcePolicy::default_library(),
        )
        .expect("build default duchon termspec");
        let SmoothBasisSpec::Duchon { spec, .. } = &terms.smooth_terms[0].basis else {
            panic!("expected Duchon term");
        };
        assert_eq!(spec.length_scale, None);
    }

    #[test]
    fn one_dimensional_duchon_length_scale_opts_into_hybrid_mode() {
        let ds = continuous_dataset(
            &["y", "x"],
            (0..32)
                .map(|i| {
                    let x = i as f64 / 31.0;
                    vec![(std::f64::consts::TAU * x).sin(), x]
                })
                .collect(),
        );
        let parsed = parse_formula("y ~ duchon(x, length_scale=0.25)").expect("parse");
        let col_map = ds.column_map();
        let mut notes = Vec::new();
        let terms = build_termspec(
            &parsed.terms,
            &ds,
            &col_map,
            &mut notes,
            &crate::resource::ResourcePolicy::default_library(),
        )
        .expect("build hybrid duchon termspec");
        let SmoothBasisSpec::Duchon { spec, .. } = &terms.smooth_terms[0].basis else {
            panic!("expected Duchon term");
        };
        assert_eq!(spec.length_scale, Some(0.25));
    }

    #[test]
    fn parse_matern_nu_accepts_equivalent_half_integer_forms() {
        let cases = [
            ("1/2", MaternNu::Half),
            (" 1 / 2 ", MaternNu::Half),
            (".5", MaternNu::Half),
            ("0.50", MaternNu::Half),
            ("half", MaternNu::Half),
            ("3 / 2", MaternNu::ThreeHalves),
            ("1.50", MaternNu::ThreeHalves),
            ("5 / 2", MaternNu::FiveHalves),
            ("2.500000000000", MaternNu::FiveHalves),
            ("7 / 2", MaternNu::SevenHalves),
            ("3.50", MaternNu::SevenHalves),
            ("9 / 2", MaternNu::NineHalves),
            ("4.50", MaternNu::NineHalves),
        ];
        for (raw, expected) in cases {
            let parsed = parse_matern_nu(raw).expect(raw);
            assert!(
                matches!(
                    (parsed, expected),
                    (MaternNu::Half, MaternNu::Half)
                        | (MaternNu::ThreeHalves, MaternNu::ThreeHalves)
                        | (MaternNu::FiveHalves, MaternNu::FiveHalves)
                        | (MaternNu::SevenHalves, MaternNu::SevenHalves)
                        | (MaternNu::NineHalves, MaternNu::NineHalves)
                ),
                "parsed {raw:?} as {parsed:?}, expected {expected:?}"
            );
        }
    }

    #[test]
    fn parse_matern_nu_rejects_unsupported_or_invalid_values() {
        for raw in ["1", "2", "11/2", "1/0", "nan", "fast"] {
            let err = parse_matern_nu(raw).expect_err(raw);
            assert!(
                err.contains("supported half-integer values"),
                "unexpected error for {raw:?}: {err}"
            );
        }
    }

    #[test]
    fn parse_ps_k_promotes_underexpressive_cubic_basis() {
        let mut opts = BTreeMap::new();
        opts.insert("k".to_string(), "4".to_string());
        let (internal, inferred) = parse_ps_internal_knots(&opts, 3, 20).expect("k=4");
        assert_eq!(internal, 2);
        assert!(!inferred);

        opts.insert("k".to_string(), "6".to_string());
        let (internal, inferred) = parse_ps_internal_knots(&opts, 3, 20).expect("k=6");
        assert_eq!(internal, 2);
        assert!(!inferred);

        opts.insert("k".to_string(), "10".to_string());
        let (internal, inferred) = parse_ps_internal_knots(&opts, 3, 20).expect("k=10");
        assert_eq!(internal, 6);
        assert!(!inferred);
    }

    #[test]
    fn parse_tensor_periods_and_origins_aliases() {
        let mut opts = BTreeMap::new();
        opts.insert("boundary".to_string(), "['periodic', 'periodic']".to_string());
        opts.insert("periods".to_string(), "[7, 24]".to_string());
        opts.insert("origins".to_string(), "[0, -12]".to_string());
        let axes = parse_periodic_axes(&opts, 2).expect("axes");
        let periods = parse_periods(&opts, &axes).expect("periods");
        let origins = parse_period_origins(&opts, &axes).expect("origins");
        assert_eq!(axes, vec![true, true]);
        assert_eq!(periods, vec![Some(7.0), Some(24.0)]);
        assert_eq!(origins, vec![Some(0.0), Some(-12.0)]);
    }

    #[test]
    fn tensor_smooth_honors_per_margin_k_list() {
        let ds = continuous_dataset(
            &["y", "theta", "h"],
            (0..20)
                .map(|i| {
                    let theta = std::f64::consts::TAU * i as f64 / 20.0;
                    let h = -1.0 + 2.0 * (i % 5) as f64 / 4.0;
                    vec![theta.cos() + h, theta, h]
                })
                .collect(),
        );
        let parsed = parse_formula(
            "y ~ te(theta, h, periodic=[0], period=[2*pi, None], origin=[0, None], k=[9,5])",
        )
        .expect("parse tensor formula");
        let col_map = ds.column_map();
        let mut notes = Vec::new();
        let terms = build_termspec(
            &parsed.terms,
            &ds,
            &col_map,
            &mut notes,
            &crate::resource::ResourcePolicy::default_library(),
        )
        .expect("build tensor terms");
        let SmoothBasisSpec::TensorBSpline { spec, .. } = &terms.smooth_terms[0].basis else {
            panic!("expected tensor B-spline");
        };
        let dims = spec
            .marginalspecs
            .iter()
            .map(|m| match m.knotspec {
                BSplineKnotSpec::PeriodicUniform { num_basis, .. } => num_basis,
                BSplineKnotSpec::Generate {
                    num_internal_knots, ..
                } => num_internal_knots + m.degree + 1,
                _ => panic!("unexpected tensor marginal knotspec"),
            })
            .collect::<Vec<_>>();
        assert_eq!(dims, vec![9, 5]);
    }

    #[test]
    fn explicit_basis_sizes_are_not_small_n_clamped() {
        let ds = continuous_dataset(
            &["y", "x1", "x2", "x3", "x4", "x5"],
            (0..12)
                .map(|i| {
                    let x = i as f64 / 11.0;
                    vec![x.sin(), x, x * x, x + 0.1, 1.0 - x, (2.0 * x).sin()]
                })
                .collect(),
        );
        let parsed = parse_formula("y ~ s(x1, k=10) + s(x2) + s(x3) + s(x4) + s(x5)")
            .expect("parse multi-smooth formula");
        let col_map = ds.column_map();
        let mut notes = Vec::new();
        let terms = build_termspec(
            &parsed.terms,
            &ds,
            &col_map,
            &mut notes,
            &crate::resource::ResourcePolicy::default_library(),
        )
        .expect("build multi-smooth terms");
        let SmoothBasisSpec::BSpline1D { spec, .. } = &terms.smooth_terms[0].basis else {
            panic!("expected first smooth to be B-spline");
        };
        assert!(matches!(
            &spec.knotspec,
            BSplineKnotSpec::Generate {
                num_internal_knots: 6,
                ..
            }
        ));
    }

    #[test]
    fn explicit_duchon_centers_are_not_small_n_bumped() {
        let ds = continuous_dataset(
            &["y", "x1", "x2", "x3", "x4", "x5"],
            (0..12)
                .map(|i| {
                    let x = i as f64 / 11.0;
                    vec![x.sin(), x, x * x, x + 0.1, 1.0 - x, (2.0 * x).sin()]
                })
                .collect(),
        );
        // Pure 1D Duchon at default options resolves the nullspace to Linear
        // (2s < d forces escalation), giving 2 polynomial nullspace columns;
        // the well-posedness gate requires num_centers > polynomial_cols, so
        // 3 is the smallest valid count. It is still well below the small-N
        // bump target of polynomial_cols + 4 = 6, so this exercises the
        // "explicit value is honored" path the test name advertises.
        let parsed = parse_formula("y ~ duchon(x1, centers=3) + s(x2) + s(x3) + s(x4) + s(x5)")
            .expect("parse multi-smooth formula");
        let col_map = ds.column_map();
        let mut notes = Vec::new();
        let terms = build_termspec(
            &parsed.terms,
            &ds,
            &col_map,
            &mut notes,
            &crate::resource::ResourcePolicy::default_library(),
        )
        .expect("build multi-smooth terms");
        let SmoothBasisSpec::Duchon { spec, .. } = &terms.smooth_terms[0].basis else {
            panic!("expected first smooth to be Duchon");
        };
        assert!(matches!(
            spec.center_strategy,
            CenterStrategy::EqualMass { num_centers: 3 }
        ));
    }

    #[test]
    fn inferred_tensor_basis_cap_uses_coordinate_support_not_duplicate_rows() {
        let mut unique_rows = Vec::new();
        for i in 0..50 {
            let theta = i as f64 / 50.0;
            for j in 0..16 {
                let h = -1.0 + 2.0 * (j as f64) / 15.0;
                let y = theta.cos() + h;
                unique_rows.push(vec![y, theta, h]);
            }
        }
        let mut repeated_rows = Vec::new();
        for _ in 0..12 {
            repeated_rows.extend(unique_rows.iter().cloned());
        }

        let unique = continuous_dataset(&["y", "theta", "h"], unique_rows);
        let repeated = continuous_dataset(&["y", "theta", "h"], repeated_rows);

        let unique_basis = inferred_tensor_basis_product(&unique);
        let repeated_basis = inferred_tensor_basis_product(&repeated);

        assert_eq!(
            unique_basis, repeated_basis,
            "duplicating existing tensor coordinates must not inflate inferred basis width"
        );
    }

    #[test]
    fn parse_bspline_boundary_conditions_and_side_selector() {
        // Non-zero anchors are rejected at parse time; the diagnostic must
        // name the side and value, which doubles as a check that the
        // `side=left` filter routes the global `anchor=` value to the
        // left endpoint (not the right).
        let mut opts = BTreeMap::new();
        opts.insert("boundary_conditions".to_string(), "anchored".to_string());
        opts.insert("side".to_string(), "left".to_string());
        opts.insert("anchor".to_string(), "2.5".to_string());
        let err = parse_bspline_boundary_conditions(&opts)
            .expect_err("non-zero left anchor must be rejected")
            .to_string();
        assert!(
            err.contains("left") && err.contains("2.5"),
            "rejection should name the affected side and value: {err}"
        );

        // Side-specific aliases (`start_bc`/`end_bc`) plus the side-specific
        // anchor key (`right_anchor`) must funnel the value onto the right
        // endpoint — verified through the rejection diagnostic.
        let mut opts = BTreeMap::new();
        opts.insert("start_bc".to_string(), "clamped".to_string());
        opts.insert("end_bc".to_string(), "zero".to_string());
        opts.insert("right_anchor".to_string(), "-1.0".to_string());
        let err = parse_bspline_boundary_conditions(&opts)
            .expect_err("non-zero right anchor must be rejected")
            .to_string();
        assert!(
            err.contains("right") && err.contains("-1"),
            "rejection should name the affected side and value: {err}"
        );

        // With anchors at zero the basis builder accepts the configuration,
        // so the same alias plumbing yields a clean `Anchored { value: 0.0 }`
        // on the right and `Clamped` on the left.
        let mut opts = BTreeMap::new();
        opts.insert("start_bc".to_string(), "clamped".to_string());
        opts.insert("end_bc".to_string(), "zero".to_string());
        let parsed = parse_bspline_boundary_conditions(&opts).expect("boundary conditions");
        assert!(matches!(
            parsed.left,
            BSplineEndpointBoundaryCondition::Clamped
        ));
        assert!(matches!(
            parsed.right,
            BSplineEndpointBoundaryCondition::Anchored { value } if value.abs() < 1e-12
        ));
    }
}

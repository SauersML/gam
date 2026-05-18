//! Term construction: bridge from parsed formula terms to `TermCollectionSpec`.
//!
//! This module takes the AST produced by `inference::formula_dsl` and a loaded
//! dataset, resolves column references, infers knot counts and center strategies,
//! and produces a `TermCollectionSpec` ready for `build_term_collection_design`.

use std::collections::{BTreeMap, HashMap};

use ndarray::ArrayView1;

use crate::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineEndpointBoundaryCondition,
    BSplineIdentifiability, BSplineKnotSpec, CenterCountRequest, CenterStrategy, DuchonBasisSpec,
    DuchonNullspaceOrder, DuchonOperatorPenaltySpec, MaternBasisSpec, MaternIdentifiability,
    MaternNu, SpatialIdentifiability, SphericalSplineBasisSpec, ThinPlateBasisSpec,
    auto_spatial_center_strategy, default_num_centers, default_spatial_center_strategy,
    plan_spatial_basis, resolve_duchon_orders,
};
use crate::inference::data::{EncodedDataset as Dataset, missing_column_message};
use crate::inference::formula_dsl::{
    ParsedTerm, SmoothKind, option_bool, option_f64, option_f64_strict, option_usize,
    option_usize_any, option_usize_any_strict, option_usize_strict,
};
use crate::inference::model::ColumnKindTag;
use crate::resource::ResourcePolicy;
use crate::smooth::{
    ByVarKind, FactorSmoothFlavour, FactorSmoothSpec, LinearCoefficientGeometry, LinearTermSpec,
    RandomEffectTermSpec, ShapeConstraint, SmoothBasisSpec, SmoothTermSpec,
    TensorBSplineIdentifiability, TensorBSplineSpec, TermCollectionSpec,
};

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
                let auto_kind =
                    ds.column_kinds.get(col).copied().ok_or_else(|| {
                        format!("internal column-kind lookup failed for '{name}'")
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
                                return Err(format!(
                                    "coefficient constraints are not supported for categorical auto-random-effect term '{name}'; use group({name}) or an unconstrained numeric term"
                                ));
                            }
                            random_terms.push(RandomEffectTermSpec {
                                name: name.clone(),
                                feature_col: col,
                                drop_first_level: false,
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
                let auto_kind =
                    ds.column_kinds.get(col).copied().ok_or_else(|| {
                        format!("internal column-kind lookup failed for '{name}'")
                    })?;
                if !matches!(auto_kind, ColumnKindTag::Continuous | ColumnKindTag::Binary) {
                    return Err(format!(
                        "bounded() currently supports only numeric columns, got categorical '{name}'"
                    ));
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
                    frozen_levels: None,
                });
            }
            ParsedTerm::Smooth {
                label,
                vars,
                kind,
                options,
            } => {
                let cols = vars
                    .iter()
                    .map(|v| resolve_col(col_map, v))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut inner_options = options.clone();
                let by_name = inner_options.remove("by");
                let basis_inner = build_smooth_basis(
                    *kind,
                    vars,
                    &cols,
                    &inner_options,
                    ds,
                    inference_notes,
                    policy,
                    smooth_coordinate_count,
                )?;
                let basis = if let Some(by_name) = by_name {
                    let by_col = resolve_col(col_map, &by_name)?;
                    let by_kind = match ds
                        .column_kinds
                        .get(by_col)
                        .copied()
                        .unwrap_or(ColumnKindTag::Continuous)
                    {
                        ColumnKindTag::Categorical => {
                            let has_main = terms.iter().any(|term| matches!(term,
                                ParsedTerm::Linear { name, .. } | ParsedTerm::RandomEffect { name } if name == &by_name));
                            if !has_main {
                                inference_notes.push(format!(
                                    "factor by-smooth '{}' uses by={} without an explicit main-effect term; add '{}' for mgcv-style level identifiability",
                                    label, by_name, by_name
                                ));
                            }
                            ByVarKind::Factor {
                                feature_col: by_col,
                                ordered: false,
                                frozen_levels: None,
                            }
                        }
                        _ => ByVarKind::Numeric {
                            feature_col: by_col,
                        },
                    };
                    SmoothBasisSpec::BySmooth {
                        smooth: Box::new(basis_inner),
                        by_kind,
                    }
                } else {
                    basis_inner
                };
                smooth_terms.push(SmoothTermSpec {
                    name: label.clone(),
                    basis,
                    shape: ShapeConstraint::None,
                });
            }
            ParsedTerm::LinkWiggle { .. }
            | ParsedTerm::TimeWiggle { .. }
            | ParsedTerm::LinkConfig { .. }
            | ParsedTerm::SurvivalConfig { .. } => {
                // Consumed at formula level, not design terms.
            }
        }
    }

    Ok(TermCollectionSpec {
        linear_terms,
        random_effect_terms: random_terms,
        smooth_terms,
    })
}

// ---------------------------------------------------------------------------
// Smooth basis spec construction
// ---------------------------------------------------------------------------

/// Look up a per-side boundary-condition key under any of several names that
/// users intuitively reach for, picking the first present. Aliases:
///   - left  → bc_left, left_bc, start_bc
///   - right → bc_right, right_bc, end_bc
fn endpoint_bc_raw<'a>(options: &'a BTreeMap<String, String>, side: &str) -> Option<&'a String> {
    let aliases: &[&str] = match side {
        "left" => &["bc_left", "left_bc", "start_bc"],
        "right" => &["bc_right", "right_bc", "end_bc"],
        _ => &[],
    };
    aliases.iter().find_map(|k| options.get(*k))
}

/// Anchor lookup: per-side `anchor_<side>`/`<side>_anchor`, else global
/// `anchor`/`anchor_value`/`value`.
fn endpoint_anchor_value(options: &BTreeMap<String, String>, side: &str) -> Option<f64> {
    let side_keys: &[&str] = match side {
        "left" => &["anchor_left", "left_anchor"],
        "right" => &["anchor_right", "right_anchor"],
        _ => &[],
    };
    for k in side_keys {
        if let Some(v) = option_f64(options, k) {
            return Some(v);
        }
    }
    for k in &["anchor", "anchor_value", "value"] {
        if let Some(v) = option_f64(options, k) {
            return Some(v);
        }
    }
    None
}

#[derive(Clone, Copy)]
enum SideFilter {
    Both,
    Left,
    Right,
}

fn parse_bspline_endpoint_condition(
    options: &BTreeMap<String, String>,
    side: &str,
    global_bc: Option<&str>,
    side_filter: SideFilter,
) -> Result<BSplineEndpointBoundaryCondition, String> {
    let applies_global = matches!(
        (side, side_filter),
        (_, SideFilter::Both) | ("left", SideFilter::Left) | ("right", SideFilter::Right)
    );
    let raw = endpoint_bc_raw(options, side)
        .map(String::as_str)
        .or_else(|| if applies_global { global_bc } else { None })
        .unwrap_or("free")
        .trim()
        .to_ascii_lowercase();
    match raw.as_str() {
        "free" | "none" | "open" => Ok(BSplineEndpointBoundaryCondition::Free),
        "clamped" | "clamp" | "zero_derivative" | "zero-derivative" => {
            Ok(BSplineEndpointBoundaryCondition::Clamped)
        }
        "anchored" | "anchor" | "zero" | "zero_value" | "zero-value" => {
            let value = endpoint_anchor_value(options, side).unwrap_or(0.0);
            // Nonzero anchor values aren't supported by the basis builder yet
            // (it would need an affine offset term). Reject upfront with a
            // clear actionable message instead of letting the user see a
            // generic "Matrix conditioning issue / basis function generation
            // failed" wrapped error during fit.
            if value != 0.0 {
                return Err(format!(
                    "anchored {side} endpoint with non-zero value {value} is not supported yet; \
                     pass anchor value 0 (or omit `anchor_{side}=`) and subtract the offset from \
                     `y` before fitting if you need to pin the boundary at a non-zero level."
                ));
            }
            Ok(BSplineEndpointBoundaryCondition::Anchored { value })
        }
        other => Err(format!(
            "unsupported B-spline boundary condition '{other}' for {side} endpoint; use free|clamped|anchored"
        )),
    }
}

fn parse_bspline_boundary_conditions(
    options: &BTreeMap<String, String>,
) -> Result<BSplineBoundaryConditions, String> {
    let global_bc = options.get("bc").map(String::as_str);
    let side_filter = match options
        .get("side")
        .map(|s| s.trim().to_ascii_lowercase())
        .as_deref()
    {
        None | Some("both") => SideFilter::Both,
        Some("left") | Some("start") => SideFilter::Left,
        Some("right") | Some("end") => SideFilter::Right,
        Some(other) => {
            return Err(format!(
                "unsupported B-spline boundary side '{other}'; use left|right|both"
            ));
        }
    };
    Ok(BSplineBoundaryConditions {
        left: parse_bspline_endpoint_condition(options, "left", global_bc, side_filter)?,
        right: parse_bspline_endpoint_condition(options, "right", global_bc, side_filter)?,
    })
}

fn bspline_bc_declares_periodic_axis(options: &BTreeMap<String, String>) -> bool {
    options
        .get("bc")
        .map(|raw| {
            let vals = split_list_option(raw);
            vals.len() == 1
                && matches!(
                    vals[0]
                        .trim_matches('"')
                        .trim_matches('\'')
                        .to_ascii_lowercase()
                        .as_str(),
                    "periodic" | "cyclic" | "cc"
                )
        })
        .unwrap_or(false)
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
        if unique_count_column(ds.values.column(col)) <= 1 {
            return Err(format!(
                "smooth term over '{var}' has only one unique value in the training data \
                 — a smooth on a constant column is degenerate and would only fit the response mean. \
                 Remove `{var}` from the smooth, drop the term, or check the data."
            ));
        }
    }
    let smooth_double_penalty = option_bool(options, "double_penalty").unwrap_or(true);
    let has_periodic_option = options.contains_key("periodic")
        || options.contains_key("cyclic")
        || options
            .get("bc")
            .map(|bc| {
                bc.to_ascii_lowercase().contains("periodic")
                    || bc.to_ascii_lowercase().contains("cyclic")
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

    match type_opt.as_str() {
        "fs" | "sz" | "re" => {
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
                    "id",
                ],
            )?;
            let mut group_idx = None;
            let mut cont = Vec::new();
            for (i, &c) in cols.iter().enumerate() {
                match ds
                    .column_kinds
                    .get(c)
                    .copied()
                    .unwrap_or(ColumnKindTag::Continuous)
                {
                    ColumnKindTag::Categorical if group_idx.is_none() => group_idx = Some(i),
                    _ => cont.push((i, c)),
                }
            }
            let group_i = group_idx.ok_or_else(|| {
                format!(
                    "bs=\"{}\" smooth requires one categorical grouping variable",
                    type_opt
                )
            })?;
            if cont.len() != 1 {
                return Err(format!(
                    "bs=\"{}\" currently supports exactly one continuous variable plus one factor",
                    type_opt
                ));
            }
            let c = cont[0].1;
            let (minv, maxv) = col_minmax(ds.values.column(c))?;
            let degree = if type_opt == "re" {
                1
            } else {
                option_usize(options, "degree").unwrap_or(3)
            };
            let default_internal = heuristic_knots_for_column(ds.values.column(c));
            let (n_knots, _) = parse_ps_internal_knots(options, degree, default_internal)?;
            let flavour = match type_opt.as_str() {
                "fs" => FactorSmoothFlavour::Fs {
                    m_null_penalty_orders: (0..option_usize(options, "m").unwrap_or(2)).collect(),
                },
                "sz" => FactorSmoothFlavour::Sz,
                _ => FactorSmoothFlavour::Re,
            };
            Ok(SmoothBasisSpec::FactorSmooth {
                spec: FactorSmoothSpec {
                    continuous_cols: vec![c],
                    group_col: cols[group_i],
                    marginal: BSplineBasisSpec {
                        degree,
                        penalty_order: option_usize(options, "penalty_order").unwrap_or(2),
                        knotspec: BSplineKnotSpec::Generate {
                            data_range: (minv, maxv),
                            num_internal_knots: n_knots,
                        },
                        double_penalty: false,
                        identifiability: BSplineIdentifiability::None,
                        boundary_conditions: Default::default(),
                    },
                    flavour,
                    group_frozen_levels: None,
                },
            })
        }
        "tensor" | "te" | "tensor-bspline" => {
            validate_known_options(
                "te",
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
                    "bc",
                    "periodic",
                    "period",
                    "periods",
                    "period_start",
                    "period_end",
                    "origin",
                    "origins",
                    "period_origin",
                    "period-origin",
                    "domain_origin",
                    "double_penalty",
                    "identifiability",
                ],
            )?;
            if cols.len() < 2 {
                return Err(format!(
                    "tensor smooth requires >=2 variables: {}",
                    vars.join(", ")
                ));
            }
            let tensor_double_penalty = option_bool(options, "double_penalty").unwrap_or(false);
            let degree = 3usize;
            let knots_internal = option_usize_list_any(options, &["knots"])?
                .map(|v| expand_margin_usize_option("knots", v, cols.len()))
                .transpose()?;
            let basis_dim =
                option_usize_list_any(options, &["k", "basis_dim", "basis-dim", "basisdim"])?
                    .map(|v| expand_margin_usize_option("k", v, cols.len()))
                    .transpose()?;
            if knots_internal.is_some() && basis_dim.is_some() {
                return Err(
                    "tensor smooth: specify either knots=<internal_knots> or k=<basis_dim> (not both)"
                        .to_string(),
                );
            }
            let inferred = knots_internal.is_none() && basis_dim.is_none();
            let mut internal_knots_by_dim = if let Some(k_values) = basis_dim {
                let min_k = degree + 1;
                for &k in &k_values {
                    if k < min_k {
                        return Err(format!(
                            "tensor smooth: k={} too small for degree {}; expected k >= {}",
                            k, degree, min_k
                        ));
                    }
                }
                k_values.into_iter().map(|k| k - min_k).collect()
            } else if let Some(knots) = knots_internal {
                knots
            } else {
                cols.iter()
                    .map(|&c| heuristic_knots_for_column(ds.values.column(c)))
                    .collect()
            };
            if inferred {
                let effective_n = tensor_effective_support_count(ds, cols);
                cap_inferred_tensor_internal_knots(&mut internal_knots_by_dim, degree, effective_n);
            }
            if inferred && ds.values.nrows() <= 32 && smooth_coordinate_count >= 5 {
                for n_knots in &mut internal_knots_by_dim {
                    *n_knots = (*n_knots).min(1);
                }
            }
            if inferred {
                inference_notes.push(format!(
                    "Automatically set tensor smooth '{}' internal knots per margin to [{}] from per-axis cardinality and rank-product limits. Override with knots=... or k=....",
                    vars.join(","),
                    internal_knots_by_dim
                        .iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            validate_tensor_bc_entries(options, cols.len())?;
            let periodic_axes = parse_periodic_axes(options, cols.len())?;
            let periods = parse_periods(options, &periodic_axes)?;
            let origins = parse_period_origins(options, &periodic_axes)?;
            let specs = cols
                .iter()
                .enumerate()
                .map(|(dim, &c)| {
                    let (minv, maxv) = col_minmax(ds.values.column(c))?;
                    let n_knots = internal_knots_by_dim[dim];
                    let knotspec = if periodic_axes[dim] {
                        let period = periods[dim].ok_or_else(|| {
                            format!(
                                "periodic tensor margin {} ('{}') requires period=[...] entry",
                                dim, vars[dim]
                            )
                        })?;
                        let domain_start = origins[dim].unwrap_or(minv);
                        BSplineKnotSpec::PeriodicUniform {
                            data_range: (domain_start, domain_start + period),
                            num_basis: n_knots + degree + 1,
                        }
                    } else {
                        BSplineKnotSpec::Generate {
                            data_range: (minv, maxv),
                            num_internal_knots: n_knots,
                        }
                    };
                    Ok(BSplineBasisSpec {
                        degree,
                        penalty_order: 2,
                        knotspec,
                        double_penalty: tensor_double_penalty,
                        identifiability: BSplineIdentifiability::None,
                        boundary_conditions: Default::default(),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(SmoothBasisSpec::TensorBSpline {
                feature_cols: cols.to_vec(),
                spec: TensorBSplineSpec {
                    marginalspecs: specs,
                    double_penalty: tensor_double_penalty,
                    identifiability: parse_tensor_identifiability(options)?,
                },
            })
        }
        "periodic" | "cyclic" | "periodic-bspline" | "cc" => {
            validate_known_options(
                "periodic",
                options,
                &[
                    "type",
                    "bs",
                    "k",
                    "basis_dim",
                    "basis-dim",
                    "basisdim",
                    "degree",
                    "penalty_order",
                    "period",
                    "periods",
                    "period_start",
                    "period_end",
                    "origin",
                    "period_origin",
                    "period-origin",
                    "domain_origin",
                    "double_penalty",
                ],
            )?;
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
                    "k",
                    "basis_dim",
                    "basis-dim",
                    "basisdim",
                    "knots",
                    "degree",
                    "penalty_order",
                    "bc",
                    "bc_left",
                    "bc_right",
                    "left_bc",
                    "right_bc",
                    "start_bc",
                    "end_bc",
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
                    "identifiability",
                ],
            )?;
            if cols.len() != 1 {
                return Err(format!(
                    "bspline smooth expects one variable, got {}",
                    cols.len()
                ));
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
            let periodic_axes = parse_periodic_axes(options, 1)?;
            let boundary_conditions =
                if periodic_axes[0] && bspline_bc_declares_periodic_axis(options) {
                    BSplineBoundaryConditions::default()
                } else {
                    parse_bspline_boundary_conditions(options)?
                };
            let periods = parse_periods(options, &periodic_axes)?;
            let origins = parse_period_origins(options, &periodic_axes)?;
            let knotspec = if periodic_axes[0] {
                if !boundary_conditions.is_free() {
                    return Err(
                        "periodic B-splines cannot also declare endpoint boundary conditions"
                            .to_string(),
                    );
                }
                {
                    let (domain_start, p_value) = if periods[0].is_some() {
                        (origins[0].unwrap_or(minv), periods[0].unwrap())
                    } else {
                        parse_periodic_domain_1d(options, minv, maxv)?
                    };
                    BSplineKnotSpec::PeriodicUniform {
                        data_range: (domain_start, domain_start + p_value),
                        num_basis: n_knots + degree + 1,
                    }
                }
            } else {
                BSplineKnotSpec::Generate {
                    data_range: (minv, maxv),
                    num_internal_knots: n_knots,
                }
            };
            Ok(SmoothBasisSpec::BSpline1D {
                feature_col: c,
                spec: BSplineBasisSpec {
                    degree,
                    penalty_order: option_usize(options, "penalty_order").unwrap_or(2),
                    knotspec,
                    double_penalty: smooth_double_penalty,
                    identifiability: BSplineIdentifiability::default(),
                    boundary_conditions,
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
                    "length_scale",
                    "centers",
                    "k",
                    "basis_dim",
                    "basis-dim",
                    "basisdim",
                    "knots",
                    "include_intercept",
                    "double_penalty",
                    "identifiability",
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
                    // 0.0 = "auto"; replaced with a data-driven init at planning
                    // time. The default 1.0 was a basin from which REML cannot
                    // escape for high-ν / high-frequency truths.
                    // Strict: reject unparseable values like `length_scale=abc`
                    // upfront instead of silently dropping them and using the auto
                    // init sentinel.
                    length_scale: option_f64_strict(options, "length_scale")?.unwrap_or(0.0),
                    double_penalty: smooth_double_penalty,
                    identifiability: parse_spatial_identifiability(options)?,
                    radial_reparam: None,
                },
                input_scales: None,
            })
        }
        "sphere" | "sos" | "spherical" => {
            validate_known_options(
                "sphere",
                options,
                &[
                    "type",
                    "bs",
                    "method",
                    "m",
                    "order",
                    "penalty_order",
                    "radians",
                    "units",
                    "max_degree",
                    "max-degree",
                    "max_l",
                    "max-l",
                    "harmonic_degree",
                    "l",
                    "centers",
                    "k",
                    "basis_dim",
                    "basis-dim",
                    "basisdim",
                    "knots",
                    "double_penalty",
                    // Wahba-kernel selector: pick Sobolev (default) or
                    // pseudo-spline (mgcv-compat). `kernel=sobolev|pseudo|mgcv|sos`.
                    "kernel",
                    "wahba_kernel",
                    "wahba-kernel",
                ],
            )?;
            if cols.len() != 2 {
                return Err(format!(
                    "sphere smooth expects exactly two variables (latitude, longitude), got {}",
                    cols.len()
                ));
            }
            let plan = plan_spatial_basis(
                ds.values.nrows(),
                2,
                CenterCountRequest::Default,
                DuchonNullspaceOrder::Zero,
                false,
                policy,
            )
            .map_err(|e| e.to_string())?;
            let centers = parse_countwith_basis_alias(options, "centers", plan.centers)?;
            let center_strategy = CenterStrategy::FarthestPoint {
                num_centers: centers,
            };
            let penalty_order = option_usize(options, "m")
                .or_else(|| option_usize(options, "order"))
                .or_else(|| option_usize(options, "penalty_order"))
                .unwrap_or(2);
            if !(1..=4).contains(&penalty_order) {
                return Err(format!(
                    "sphere smooth penalty order must be one of 1, 2, 3, 4; got {penalty_order}"
                ));
            }
            let radians = option_bool(options, "radians").unwrap_or_else(|| {
                options
                    .get("units")
                    .map(|u| {
                        u.trim().to_ascii_lowercase() == "radians"
                            || u.trim().to_ascii_lowercase() == "rad"
                    })
                    .unwrap_or(false)
            });
            // Parse `method=`. The Wahba-family kernel choice (Sobolev vs
            // pseudo-spline) is also accepted via `method=`: passing
            // `wahba_sobolev` / `wahba_pseudo` is equivalent to
            // `method=wahba, kernel=sobolev` / `method=wahba, kernel=pseudo`.
            // Plain `wahba` defaults to the Sobolev kernel (more correct).
            let (method, method_kernel) = match options
                .get("method")
                .map(|s| s.trim().to_ascii_lowercase())
                .as_deref()
            {
                None | Some("wahba") | Some("kernel") => (
                    crate::basis::SphereMethod::Wahba,
                    Option::<crate::basis::SphereWahbaKernel>::None,
                ),
                Some("wahba_sobolev")
                | Some("wahba-sobolev")
                | Some("sobolev")
                | Some("sobolev_wahba")
                | Some("sobolev-wahba") => (
                    crate::basis::SphereMethod::Wahba,
                    Some(crate::basis::SphereWahbaKernel::Sobolev),
                ),
                Some("wahba_pseudo") | Some("wahba-pseudo") | Some("pseudo")
                | Some("pseudo_wahba") | Some("pseudo-wahba") | Some("mgcv") | Some("sos") => (
                    crate::basis::SphereMethod::Wahba,
                    Some(crate::basis::SphereWahbaKernel::Pseudo),
                ),
                Some("harmonic")
                | Some("harmonics")
                | Some("spherical_harmonics")
                | Some("spherical-harmonics")
                | Some("sh") => (crate::basis::SphereMethod::Harmonic, None),
                Some(other) => {
                    return Err(format!(
                        "unsupported sphere method '{other}'; use one of: \
                         wahba | wahba_sobolev (default Wahba) | wahba_pseudo (mgcv `bs=\"sos\"` compatible) | harmonic"
                    ));
                }
            };
            // Also accept an explicit `kernel=` option for users who set
            // `method=wahba` and want to pick the underlying kernel
            // separately. `kernel=` overrides any implicit choice from
            // `method=wahba_<kind>`. Defaults to Sobolev.
            let explicit_kernel = match options
                .get("kernel")
                .or_else(|| options.get("wahba_kernel"))
                .or_else(|| options.get("wahba-kernel"))
                .map(|s| s.trim().to_ascii_lowercase())
                .as_deref()
            {
                None => None,
                Some("sobolev") | Some("true") | Some("h") | Some("hm") => {
                    Some(crate::basis::SphereWahbaKernel::Sobolev)
                }
                Some("pseudo")
                | Some("pseudo_spline")
                | Some("pseudo-spline")
                | Some("mgcv")
                | Some("sos") => Some(crate::basis::SphereWahbaKernel::Pseudo),
                Some(other) => {
                    return Err(format!(
                        "unsupported sphere kernel '{other}'; use one of: sobolev | pseudo"
                    ));
                }
            };
            let wahba_kernel = explicit_kernel
                .or(method_kernel)
                .unwrap_or(crate::basis::SphereWahbaKernel::Sobolev);
            let max_degree = option_usize_any(
                options,
                &[
                    "max_degree",
                    "max-degree",
                    "max_l",
                    "max-l",
                    "harmonic_degree",
                    "l",
                ],
            );
            Ok(SmoothBasisSpec::Sphere {
                feature_cols: cols.to_vec(),
                spec: SphericalSplineBasisSpec {
                    center_strategy,
                    penalty_order,
                    double_penalty: smooth_double_penalty,
                    radians,
                    method,
                    max_degree,
                    wahba_kernel,
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
                    "identifiability",
                    "scale_dims",
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
                return Err(format!(
                    "matern() with nu=1/2 is not supported for d>=2 (got {} covariates): \
                     the exponential kernel's Laplacian is singular at center collisions, \
                     which makes the operator-collocation penalty non-invertible. \
                     Choose nu>=3/2 (e.g. nu=3/2 or the default nu=5/2) for multi-dimensional smooths.",
                    cols.len()
                ));
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
                    // 0.0 = "auto"; replaced with a data-driven init at planning
                    // time. The default 1.0 was a basin from which REML cannot
                    // escape for ν ≥ 5/2 on high-frequency truths.
                    // Strict: reject unparseable values like `length_scale=abc`
                    // upfront instead of silently dropping them and using the auto
                    // init sentinel.
                    length_scale: option_f64_strict(options, "length_scale")?.unwrap_or(0.0),
                    nu,
                    include_intercept: option_bool(options, "include_intercept").unwrap_or(false),
                    double_penalty: smooth_double_penalty,
                    identifiability: parse_matern_identifiability(options)?,
                    aniso_log_scales,
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
                    "periodic",
                    "period",
                    "period_start",
                    "period_end",
                    "scale_dims",
                    "double_penalty",
                ],
            )?;
            if options.contains_key("double_penalty") {
                return Err(format!(
                    "Duchon smooth '{}' does not support double_penalty; Duchon uses mass, tension, and stiffness operator penalties.",
                    vars.join(", ")
                ));
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
                return Err(format!(
                    "Duchon smooth '{}' requested basis dimension {} but order={:?} in {}D needs {} polynomial null-space columns; choose centers/k > {}",
                    vars.join(", "),
                    requested_centers,
                    nullspace_order,
                    cols.len(),
                    polynomial_cols,
                    polynomial_cols,
                ));
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
            let periodic = option_bool(options, "cyclic").unwrap_or(false)
                || option_bool(options, "periodic").unwrap_or(false);
            // Periodic Duchon (`duchon(x, periodic=true)`) is implemented at
            // the basis level (`build_periodic_duchon_basis_1d`) and is used
            // internally by tests, but the REML hyperparameter path has no
            // working kappa-derivative for the wrapped-distance kernel:
            // `prepare_duchon_derivative_context` explicitly rejects
            // `spec.periodic`. Without that derivative the outer kappa
            // optimization fails partway with an opaque
            // "spatial kappa optimization failed" wrapper. Until the
            // periodic kappa derivative is wired up, redirect users from
            // the formula DSL to the alternatives that actually fit
            // end-to-end: `s(x, periodic=true, period=...)` for 1D
            // periodic smoothing, or `te(...)` with a periodic margin for
            // higher-dimensional periodic geometries.
            if periodic {
                return Err(format!(
                    "duchon(..., periodic=true) is not currently supported through the formula DSL because the \
                     wrapped-distance kernel has no REML kappa-derivative path. Use `s({}, periodic=true, period=2*pi)` for \
                     1D periodic smoothing, or `te(...)` with `bc=['periodic', ...]` for higher-dimensional \
                     periodic geometries.",
                    vars.join(", "),
                ));
            }
            Ok(SmoothBasisSpec::Duchon {
                feature_cols: cols.to_vec(),
                spec: DuchonBasisSpec {
                    center_strategy,
                    length_scale,
                    power,
                    nullspace_order,
                    identifiability: parse_spatial_identifiability(options)?,
                    aniso_log_scales,
                    operator_penalties,
                    periodic,
                },
                input_scales: None,
            })
        }
        other => Err(format!("unsupported smooth type '{other}'")),
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
        return Err("non-finite data encountered while inferring knot range".to_string());
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

fn cap_inferred_tensor_internal_knots(internal_knots: &mut [usize], degree: usize, n: usize) {
    if internal_knots.is_empty() {
        return;
    }
    let min_basis = degree + 1;
    let max_product = inferred_tensor_basis_product_cap(n, internal_knots.len());
    loop {
        let product = internal_knots
            .iter()
            .try_fold(1usize, |acc, &knots| acc.checked_mul(knots + min_basis));
        if matches!(product, Some(p) if p <= max_product) {
            break;
        }
        let Some((idx, _)) = internal_knots
            .iter()
            .enumerate()
            .filter(|(_, knots)| **knots > 1)
            .max_by_key(|(_, knots)| **knots + min_basis)
        else {
            break;
        };
        internal_knots[idx] -= 1;
    }
}

fn tensor_effective_support_count(ds: &Dataset, cols: &[usize]) -> usize {
    let product = cols.iter().try_fold(1usize, |acc, &col| {
        acc.checked_mul(unique_count_column(ds.values.column(col)))
    });
    product.unwrap_or(usize::MAX).min(ds.values.nrows()).max(1)
}

fn inferred_tensor_basis_product_cap(n: usize, ndim: usize) -> usize {
    let upper = 128usize;
    let n_cap = (4.0 * (n.max(1) as f64).sqrt()).round() as usize;
    let dim_floor = 4usize.saturating_pow(ndim as u32).max(64).min(upper);
    n_cap.clamp(dim_floor, upper)
}

pub fn heuristic_centers(n: usize, d: usize) -> usize {
    default_num_centers(n, d)
}

fn parse_math_f64(raw: &str) -> Result<f64, String> {
    let t = raw.trim().trim_matches('"').trim_matches('\'').trim();
    if t.eq_ignore_ascii_case("none") || t.eq_ignore_ascii_case("null") {
        return Err("None/null is not a number".to_string());
    }
    let lower = t
        .to_ascii_lowercase()
        .replace(' ', "")
        .replace('π', "pi")
        .replace('τ', "tau");
    match lower.as_str() {
        "pi" => return Ok(std::f64::consts::PI),
        "tau" | "2pi" | "2*pi" => return Ok(2.0 * std::f64::consts::PI),
        _ => {}
    }
    if let Some(rest) = lower.strip_suffix("*pi") {
        let coef = if rest.is_empty() {
            1.0
        } else {
            rest.parse::<f64>()
                .map_err(|_| format!("invalid numeric expression '{raw}'"))?
        };
        return Ok(coef * std::f64::consts::PI);
    }
    if let Some(rest) = lower.strip_prefix("pi*") {
        let coef = rest
            .parse::<f64>()
            .map_err(|_| format!("invalid numeric expression '{raw}'"))?;
        return Ok(coef * std::f64::consts::PI);
    }
    lower
        .parse::<f64>()
        .map_err(|_| format!("invalid numeric expression '{raw}'"))
}

fn split_list_option(raw: &str) -> Vec<String> {
    let t = raw.trim();
    let inner = t
        .strip_prefix('[')
        .and_then(|u| u.strip_suffix(']'))
        .unwrap_or(t);
    inner.split(',').map(|v| v.trim().to_string()).collect()
}

fn option_usize_list_any(
    options: &BTreeMap<String, String>,
    keys: &[&str],
) -> Result<Option<Vec<usize>>, String> {
    for key in keys {
        if let Some(raw) = options.get(*key) {
            let vals = split_list_option(raw);
            if vals.is_empty() {
                return Err(format!("{key} must contain at least one integer"));
            }
            let parsed = vals
                .iter()
                .map(|v| {
                    v.parse::<usize>().map_err(|_| {
                        format!("{key} entries must be non-negative integers, got '{v}'")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(Some(parsed));
        }
    }
    Ok(None)
}

fn expand_margin_usize_option(
    name: &str,
    values: Vec<usize>,
    ndim: usize,
) -> Result<Vec<usize>, String> {
    match values.len() {
        1 => Ok(vec![values[0]; ndim]),
        n if n == ndim => Ok(values),
        n => Err(format!(
            "{name} must be scalar or have one entry per margin ({ndim}), got {n}"
        )),
    }
}

fn parse_periodic_axes(
    options: &BTreeMap<String, String>,
    ndim: usize,
) -> Result<Vec<bool>, String> {
    let mut out = vec![false; ndim];
    let Some(raw) = options.get("periodic").or_else(|| options.get("cyclic")) else {
        if let Some(raw_bc) = options.get("bc") {
            let vals = split_list_option(raw_bc);
            if vals.len() != ndim {
                return Err(format!(
                    "bc must have one entry per margin ({ndim}), got {}",
                    vals.len()
                ));
            }
            for (i, v) in vals.iter().enumerate() {
                let l = v.trim_matches('"').trim_matches('\'').to_ascii_lowercase();
                out[i] = matches!(l.as_str(), "periodic" | "cyclic" | "cc");
            }
        }
        return Ok(out);
    };
    let vals = split_list_option(raw);
    if vals.len() == ndim
        && vals.iter().all(|v| {
            matches!(
                v.to_ascii_lowercase().as_str(),
                "true" | "false" | "yes" | "no"
            )
        })
    {
        for (i, v) in vals.iter().enumerate() {
            out[i] = matches!(v.to_ascii_lowercase().as_str(), "true" | "yes");
        }
    } else {
        for v in vals {
            if v.is_empty() {
                continue;
            }
            let axis = v
                .parse::<usize>()
                .map_err(|_| format!("periodic axes must be zero-based integers, got '{v}'"))?;
            if axis >= ndim {
                return Err(format!(
                    "periodic axis {axis} out of range for {ndim}D smooth"
                ));
            }
            out[axis] = true;
        }
    }
    Ok(out)
}

fn validate_tensor_bc_entries(
    options: &BTreeMap<String, String>,
    ndim: usize,
) -> Result<(), String> {
    let Some(raw_bc) = options.get("bc") else {
        return Ok(());
    };
    let vals = split_list_option(raw_bc);
    if vals.len() != ndim {
        return Err(format!(
            "bc must have one entry per margin ({ndim}), got {}",
            vals.len()
        ));
    }
    for (dim, value) in vals.iter().enumerate() {
        let l = value
            .trim_matches('"')
            .trim_matches('\'')
            .to_ascii_lowercase();
        if matches!(
            l.as_str(),
            "periodic" | "cyclic" | "cc" | "natural" | "free" | "none"
        ) {
            continue;
        }
        return Err(format!(
            "tensor smooth bc entry {dim}={value:?} is not supported. Tensor margins currently support only periodic/cyclic/cc or natural/free/none; use separate s(..., bc=...) terms for endpoint clamped/anchored 1D boundary conditions."
        ));
    }
    Ok(())
}

fn parse_periods(
    options: &BTreeMap<String, String>,
    periodic: &[bool],
) -> Result<Vec<Option<f64>>, String> {
    let ndim = periodic.len();
    let mut out = vec![None; ndim];
    let Some(raw) = options.get("period").or_else(|| options.get("periods")) else {
        return Ok(out);
    };
    let vals = split_list_option(raw);
    if vals.len() == 1 && periodic.iter().filter(|&&b| b).count() == 1 {
        let value = parse_math_f64(&vals[0])?;
        if value <= 0.0 || !value.is_finite() {
            return Err("period entries must be positive finite values".to_string());
        }
        if let Some(axis) = periodic.iter().position(|&b| b) {
            out[axis] = Some(value);
        }
        return Ok(out);
    }
    if vals.len() != ndim {
        return Err(format!(
            "period must have one entry per margin ({ndim}), got {}",
            vals.len()
        ));
    }
    for (i, v) in vals.iter().enumerate() {
        let l = v.to_ascii_lowercase();
        if matches!(l.as_str(), "none" | "null" | "na" | "") {
            continue;
        }
        let value = parse_math_f64(v)?;
        if value <= 0.0 || !value.is_finite() {
            return Err("period entries must be positive finite values".to_string());
        }
        out[i] = Some(value);
    }
    Ok(out)
}

fn parse_period_origins(
    options: &BTreeMap<String, String>,
    periodic: &[bool],
) -> Result<Vec<Option<f64>>, String> {
    let ndim = periodic.len();
    let mut out = vec![None; ndim];
    let Some(raw) = options.get("origin").or_else(|| options.get("origins")) else {
        return Ok(out);
    };
    let vals = split_list_option(raw);
    if vals.len() == 1 && periodic.iter().filter(|&&b| b).count() == 1 {
        if let Some(axis) = periodic.iter().position(|&b| b) {
            out[axis] = Some(parse_math_f64(&vals[0])?);
        }
        return Ok(out);
    }
    if vals.len() != ndim {
        return Err(format!(
            "origin must have one entry per margin ({ndim}), got {}",
            vals.len()
        ));
    }
    for (i, v) in vals.iter().enumerate() {
        let l = v.to_ascii_lowercase();
        if matches!(l.as_str(), "none" | "null" | "na" | "") {
            continue;
        }
        out[i] = Some(parse_math_f64(v)?);
    }
    Ok(out)
}

fn option_math_f64_any(
    options: &BTreeMap<String, String>,
    keys: &[&str],
) -> Result<Option<f64>, String> {
    for key in keys {
        if let Some(raw) = options.get(*key) {
            return parse_math_f64(raw).map(Some);
        }
    }
    Ok(None)
}

fn parse_periodic_domain_1d(
    options: &BTreeMap<String, String>,
    data_min: f64,
    _data_max: f64,
) -> Result<(f64, f64), String> {
    let start = option_math_f64_any(
        options,
        &[
            "period_start",
            "period-start",
            "domain_start",
            "domain-start",
            "start",
        ],
    )?;
    let end = option_math_f64_any(
        options,
        &[
            "period_end",
            "period-end",
            "domain_end",
            "domain-end",
            "end",
        ],
    )?;
    match (start, end) {
        (Some(domain_start), Some(domain_end)) => {
            let period = domain_end - domain_start;
            if !period.is_finite() || period <= 0.0 {
                return Err(format!(
                    "period_end must be greater than period_start for periodic smooths, got start={domain_start}, end={domain_end}"
                ));
            }
            Ok((domain_start, period))
        }
        (Some(_), None) | (None, Some(_)) => Err(
            "periodic smooths require both period_start and period_end when either is provided"
                .to_string(),
        ),
        (None, None) => {
            let explicit_period = parse_periods(options, &[true])?[0];
            let explicit_origin = option_math_f64_any(
                options,
                &["origin", "period_origin", "period-origin", "domain_origin"],
            )?;
            match (explicit_period, explicit_origin) {
                (Some(period), origin) => Ok((origin.unwrap_or(data_min), period)),
                (None, _) => {
                    // Silently inferring period from data range is a user-facing
                    // footgun: if data on [0, 2π] is sampled uniformly the empirical
                    // range is [ε, 2π−ε], so the inferred period is slightly less
                    // than 2π and predictions at t=0 vs t=2π reach different points
                    // on the inferred circle. Force the user to be explicit.
                    Err("periodic=true requires an explicit `period=<value>` (or \
                         `period_start=<lo>, period_end=<hi>`). Silent inference \
                         from data range would set period = data_max − data_min, \
                         which is sample-dependent and rarely what users mean \
                         (e.g. uniform draws on [0, 2π] give period ≈ 2π − 2ε, \
                         not 2π, leading to off-by-ε wrap discontinuities)."
                        .to_string())
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Smooth option parsers
// ---------------------------------------------------------------------------

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
        return Err(
            "ps/bspline smooth: specify either knots=<internal_knots> or k=<basis_dim> (not both)"
                .to_string(),
        );
    }
    if let Some(k) = basis_dim {
        let min_k = degree + 1;
        if k < min_k {
            return Err(format!(
                "ps/bspline smooth: k={} too small for degree {}; expected k >= {}",
                k, degree, min_k
            ));
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
            return Err(format!(
                "{term_name}() does not accept option `{key}`{hint} Known options: [{}]",
                {
                    let mut sorted = known.to_vec();
                    sorted.sort_unstable();
                    sorted.join(", ")
                }
            ));
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
        return Err(format!(
            "specify either {}=<count> or k=<basis_dim> (not both)",
            primarykey
        ));
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

pub fn parse_matern_nu(raw: &str) -> Result<MaternNu, String> {
    let trimmed = raw.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower == "half" {
        return Ok(MaternNu::Half);
    }

    let value = if let Some((num, den)) = trimmed.split_once('/') {
        let num = num
            .trim()
            .parse::<f64>()
            .map_err(|_| unsupported_matern_nu_message(raw))?;
        let den = den
            .trim()
            .parse::<f64>()
            .map_err(|_| unsupported_matern_nu_message(raw))?;
        if den == 0.0 || !num.is_finite() || !den.is_finite() {
            return Err(unsupported_matern_nu_message(raw));
        }
        num / den
    } else {
        trimmed
            .parse::<f64>()
            .map_err(|_| unsupported_matern_nu_message(raw))?
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
    format!(
        "unsupported Matern nu '{raw}'; supported half-integer values are 1/2, 3/2, 5/2, 7/2, and 9/2"
    )
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
        return Err(format!(
            "Duchon smooths use power=<integer>, not nu='{}'. Use power=0, power=1, etc.",
            raw_nu
        ));
    }
    match options.get("power") {
        Some(raw) => raw.parse::<usize>().map(DuchonPowerPolicy::Explicit).map_err(|_| {
            format!(
                "invalid Duchon power '{}'; expected a non-negative integer such as power=0 or power=1",
                raw
            )
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
            Err(_) => Err(format!(
                "invalid Duchon order '{}'; expected a non-negative integer such as order=0, order=1, or order=2",
                raw
            )),
        },
    }
}

fn parse_matern_identifiability(
    options: &BTreeMap<String, String>,
) -> Result<MaternIdentifiability, String> {
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
        other => Err(format!(
            "invalid Matérn identifiability '{other}'; expected one of: none, sum_tozero, linear"
        )),
    }
}

fn parse_spatial_identifiability(
    options: &BTreeMap<String, String>,
) -> Result<SpatialIdentifiability, String> {
    let Some(raw) = options.get("identifiability").map(String::as_str) else {
        return Ok(SpatialIdentifiability::default());
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(SpatialIdentifiability::None),
        "orthogonal"
        | "orthogonal_to_parametric"
        | "orthogonal-to-parametric"
        | "parametric_orthogonal" => Ok(SpatialIdentifiability::OrthogonalToParametric),
        "frozen" => Err(
            "spatial identifiability 'frozen' is internal-only; use none or orthogonal_to_parametric".to_string(),
        ),
        other => Err(format!(
            "invalid spatial identifiability '{other}'; expected one of: none, orthogonal_to_parametric"
        )),
    }
}

fn parse_tensor_identifiability(
    options: &BTreeMap<String, String>,
) -> Result<TensorBSplineIdentifiability, String> {
    let Some(raw) = options.get("identifiability").map(String::as_str) else {
        return Ok(TensorBSplineIdentifiability::default());
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(TensorBSplineIdentifiability::None),
        "sum_tozero" | "sum-to-zero" | "centered" => Ok(TensorBSplineIdentifiability::SumToZero),
        "frozen" | "frozen_transform" | "frozen-transform" => Err(
            "tensor identifiability 'frozen' is internal-only; use none or sum_tozero".to_string(),
        ),
        other => Err(format!(
            "invalid tensor identifiability '{other}'; expected one of: none, sum_tozero"
        )),
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

        let mut bc_opts = BTreeMap::new();
        bc_opts.insert("bc".to_string(), "['periodic', 'natural']".to_string());
        bc_opts.insert("period".to_string(), "[2*pi, None]".to_string());
        let bc_axes = parse_periodic_axes(&bc_opts, 2).expect("bc axes");
        let bc_periods = parse_periods(&bc_opts, &bc_axes).expect("bc periods");
        assert_eq!(bc_axes, vec![true, false]);
        assert!((bc_periods[0].unwrap() - 2.0 * std::f64::consts::PI).abs() < 1e-12);
        assert_eq!(bc_periods[1], None);

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
    fn one_dimensional_bspline_accepts_bc_periodic_alias() {
        let ds = continuous_dataset(
            &["y", "theta"],
            (0..16)
                .map(|i| {
                    let theta = std::f64::consts::TAU * i as f64 / 16.0;
                    vec![theta.sin(), theta]
                })
                .collect(),
        );
        let parsed =
            parse_formula("y ~ s(theta, bc=periodic, period=2*pi, origin=0, k=8)").expect("parse");
        let col_map = ds.column_map();
        let mut notes = Vec::new();
        let terms = build_termspec(
            &parsed.terms,
            &ds,
            &col_map,
            &mut notes,
            &crate::resource::ResourcePolicy::default_library(),
        )
        .expect("periodic bc alias should build");
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
        opts.insert("bc".to_string(), "['periodic', 'periodic']".to_string());
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
    fn parse_bspline_boundary_aliases_and_side_selector() {
        // Non-zero anchors are rejected at parse time; the diagnostic must
        // name the side and value, which doubles as a check that the
        // `side=left` filter routes the global `anchor=` value to the
        // left endpoint (not the right).
        let mut opts = BTreeMap::new();
        opts.insert("bc".to_string(), "anchored".to_string());
        opts.insert("side".to_string(), "left".to_string());
        opts.insert("anchor".to_string(), "2.5".to_string());
        let err = parse_bspline_boundary_conditions(&opts)
            .expect_err("non-zero left anchor must be rejected");
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
            .expect_err("non-zero right anchor must be rejected");
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
        let parsed = parse_bspline_boundary_conditions(&opts).expect("boundary aliases");
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

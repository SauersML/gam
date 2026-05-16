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
    ParsedTerm, SmoothKind, option_bool, option_f64, option_usize, option_usize_any,
};
use crate::inference::model::ColumnKindTag;
use crate::resource::ResourcePolicy;
use crate::smooth::{
    LinearCoefficientGeometry, LinearTermSpec, RandomEffectTermSpec, ShapeConstraint,
    SmoothBasisSpec, SmoothTermSpec, TensorBSplineIdentifiability, TensorBSplineSpec,
    TermCollectionSpec,
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
                let basis = build_smooth_basis(
                    *kind,
                    vars,
                    &cols,
                    options,
                    ds,
                    inference_notes,
                    policy,
                    smooth_coordinate_count,
                )?;
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
        "tensor" | "te" | "tensor-bspline" => {
            if cols.len() < 2 {
                return Err(format!(
                    "tensor smooth requires >=2 variables: {}",
                    vars.join(", ")
                ));
            }
            let tensor_double_penalty = option_bool(options, "double_penalty").unwrap_or(false);
            let degree = 3usize;
            let knots_internal = option_usize(options, "knots");
            let basis_dim = option_usize_any(options, &["k", "basis_dim", "basis-dim", "basisdim"]);
            if knots_internal.is_some() && basis_dim.is_some() {
                return Err(
                    "tensor smooth: specify either knots=<internal_knots> or k=<basis_dim> (not both)"
                        .to_string(),
                );
            }
            let inferred = knots_internal.is_none() && basis_dim.is_none();
            let mut internal_knots_by_dim = if let Some(k) = basis_dim {
                let min_k = degree + 1;
                if k < min_k {
                    return Err(format!(
                        "tensor smooth: k={} too small for degree {}; expected k >= {}",
                        k, degree, min_k
                    ));
                }
                vec![k - min_k; cols.len()]
            } else if let Some(knots) = knots_internal {
                vec![knots; cols.len()]
            } else {
                cols.iter()
                    .map(|&c| heuristic_knots_for_column(ds.values.column(c)))
                    .collect()
            };
            if inferred {
                let effective_n = tensor_effective_support_count(ds, cols);
                cap_inferred_tensor_internal_knots(&mut internal_knots_by_dim, degree, effective_n);
            }
            if ds.values.nrows() <= 32 && smooth_coordinate_count >= 5 {
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
            if ds.values.nrows() <= 32 && smooth_coordinate_count >= 5 {
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
                    length_scale: option_f64(options, "length_scale").unwrap_or(0.0),
                    double_penalty: smooth_double_penalty,
                    identifiability: parse_spatial_identifiability(options)?,
                    radial_reparam: None,
                },
                input_scales: None,
            })
        }
        "sphere" | "sos" | "spherical" => {
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
            let center_strategy = if has_explicit_countwith_basis_alias(options, "centers") {
                CenterStrategy::FarthestPoint {
                    num_centers: centers,
                }
            } else {
                auto_spatial_center_strategy(centers, 2)
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
            let method = match options
                .get("method")
                .map(|s| s.trim().to_ascii_lowercase())
                .as_deref()
            {
                None | Some("wahba") | Some("kernel") => crate::basis::SphereMethod::Wahba,
                Some("harmonic")
                | Some("harmonics")
                | Some("spherical_harmonics")
                | Some("spherical-harmonics")
                | Some("sh") => crate::basis::SphereMethod::Harmonic,
                Some(other) => {
                    return Err(format!(
                        "unsupported sphere method '{other}'; use wahba or harmonic"
                    ));
                }
            };
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
                },
            })
        }
        "matern" => {
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
                    length_scale: option_f64(options, "length_scale").unwrap_or(0.0),
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
            if options.contains_key("double_penalty") {
                return Err(format!(
                    "Duchon smooth '{}' does not support double_penalty; Duchon uses mass, tension, and stiffness operator penalties.",
                    vars.join(", ")
                ));
            }
            let requested_nullspace_order = parse_duchon_order(options)?;
            if options.contains_key("pure") {
                return Err(
                    "duchon() is pure scale-free Duchon by default; remove pure=... or specify length_scale=... for hybrid Duchon".to_string(),
                );
            }
            let length_scale = option_f64(options, "length_scale");
            // Resolve `(nullspace_order, power)` against the joint constraints
            // (operator collocation + pure-mode CPD). Explicit power keeps the
            // user's nullspace as-is (validator will reject inconsistent combos);
            // the policy path may auto-escalate the nullspace order in pure mode
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
                            "Note: pure Duchon CPD against polynomial nullspace requires order ≥ {:?} \
                             at dimension {} (Wendland 8.17, 2s < d); auto-escalated from {:?}. \
                             Specify length_scale=... to use hybrid Duchon and bypass this constraint.",
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
            if ds.values.nrows() <= 32 && smooth_coordinate_count >= 5 {
                centers = centers.max(polynomial_cols + 4);
            }
            let center_strategy = if has_explicit_countwith_basis_alias(options, "centers") {
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
                    length_scale,
                    power,
                    nullspace_order,
                    identifiability: parse_spatial_identifiability(options)?,
                    aniso_log_scales,
                    operator_penalties,
                    periodic: option_bool(options, "cyclic").unwrap_or(false)
                        || option_bool(options, "periodic").unwrap_or(false),
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
    let lower = t.to_ascii_lowercase().replace(' ', "");
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
    data_max: f64,
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
            let period = parse_periods(options, &[true])?[0].unwrap_or(data_max - data_min);
            let origin = option_math_f64_any(
                options,
                &["origin", "period_origin", "period-origin", "domain_origin"],
            )?
            .unwrap_or(data_min);
            Ok((origin, period))
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
    let knots_internal = option_usize(options, "knots");
    let basis_dim = option_usize_any(options, &["k", "basis_dim", "basis-dim", "basisdim"]);
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

pub fn parse_countwith_basis_alias(
    options: &BTreeMap<String, String>,
    primarykey: &str,
    default_count: usize,
) -> Result<usize, String> {
    let primary = option_usize(options, primarykey);
    let basis_dim = option_usize_any(
        options,
        &["k", "basis_dim", "basis-dim", "basisdim", "knots"],
    );
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
        let mut opts = BTreeMap::new();
        opts.insert("bc".to_string(), "anchored".to_string());
        opts.insert("side".to_string(), "left".to_string());
        opts.insert("anchor".to_string(), "2.5".to_string());
        let parsed = parse_bspline_boundary_conditions(&opts).expect("boundary options");
        assert!(matches!(
            parsed.left,
            BSplineEndpointBoundaryCondition::Anchored { value } if (value - 2.5).abs() < 1e-12
        ));
        assert!(matches!(
            parsed.right,
            BSplineEndpointBoundaryCondition::Free
        ));

        let mut opts = BTreeMap::new();
        opts.insert("start_bc".to_string(), "clamped".to_string());
        opts.insert("end_bc".to_string(), "zero".to_string());
        opts.insert("right_anchor".to_string(), "-1.0".to_string());
        let parsed = parse_bspline_boundary_conditions(&opts).expect("boundary aliases");
        assert!(matches!(
            parsed.left,
            BSplineEndpointBoundaryCondition::Clamped
        ));
        assert!(matches!(
            parsed.right,
            BSplineEndpointBoundaryCondition::Anchored { value } if (value + 1.0).abs() < 1e-12
        ));
    }
}

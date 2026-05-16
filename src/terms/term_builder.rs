//! Term construction: bridge from parsed formula terms to `TermCollectionSpec`.
//!
//! This module takes the AST produced by `inference::formula_dsl` and a loaded
//! dataset, resolves column references, infers knot counts and center strategies,
//! and produces a `TermCollectionSpec` ready for `build_term_collection_design`.

use std::collections::{BTreeMap, HashMap};

use ndarray::ArrayView1;

use crate::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, CenterCountRequest, CenterStrategy,
    DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec, MaternBasisSpec,
    MaternIdentifiability, MaternNu, SpatialIdentifiability, ThinPlateBasisSpec,
    auto_spatial_center_strategy, default_num_centers, default_spatial_center_strategy,
    plan_spatial_basis, resolve_duchon_orders,
};
use crate::inference::data::{EncodedDataset as Dataset, missing_column_message};
use crate::inference::formula_dsl::{
    ParsedTerm, SmoothKind, option_bool, option_f64, option_usize, option_usize_any, strip_quotes,
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
        let value = if factor.eq_ignore_ascii_case("pi") {
            std::f64::consts::PI
        } else {
            factor
                .parse::<f64>()
                .map_err(|_| format!("invalid numeric expression '{raw}'"))?
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
            .map_err(|_| format!("invalid periodic axis '{a}'"))?;
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

fn parse_bc_periods_option(
    options: &BTreeMap<String, String>,
    dim: usize,
) -> Result<Vec<Option<f64>>, String> {
    let mut periods = vec![None; dim];
    if let Some(raw_bc) = options.get("bc") {
        let bc = split_list_option(raw_bc);
        if bc.len() != dim {
            return Err(format!(
                "bc list length {} must match tensor dimension {}",
                bc.len(),
                dim
            ));
        }
        let period_values = parse_periods_option(options, dim)?.unwrap_or_else(|| vec![None; dim]);
        for (i, b) in bc.iter().enumerate() {
            let b = strip_quotes(b).trim().to_ascii_lowercase();
            match b.as_str() {
                "periodic" | "cyclic" | "cc" => {
                    periods[i] =
                        period_values[i].or_else(|| if dim == 1 { period_values[0] } else { None });
                    if periods[i].is_none() {
                        return Err(format!("periodic tensor margin {i} requires period[{i}]"));
                    }
                }
                "natural" | "cr" | "bspline" | "p-spline" | "ps" | "none" => {}
                other => return Err(format!("unsupported tensor bc '{other}' at margin {i}")),
            }
        }
    }
    Ok(periods)
}

// ---------------------------------------------------------------------------
// Smooth basis spec construction
// ---------------------------------------------------------------------------

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
    let type_opt = options
        .get("type")
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| match kind {
            SmoothKind::Te => "tensor".to_string(),
            SmoothKind::S if cols.len() == 1 => "bspline".to_string(),
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
            let degree = 3usize;
            let default_internal = cols
                .iter()
                .map(|&c| heuristic_knots_for_column(ds.values.column(c)))
                .max()
                .unwrap_or_else(|| heuristic_knots(ds.values.nrows()));
            let (mut n_knots, inferred) =
                parse_ps_internal_knots(options, degree, default_internal)?;
            if ds.values.nrows() <= 32 && smooth_coordinate_count >= 5 {
                n_knots = n_knots.min(1);
            }
            if inferred {
                inference_notes.push(format!(
                    "Automatically set {} internal knots per margin for tensor smooth '{}' (max unique/4 rule across margins, clamped to [4,20]). Override with knots=... or k=....",
                    n_knots,
                    vars.join(",")
                ));
            }
            let specs = cols
                .iter()
                .map(|&c| {
                    let (minv, maxv) = col_minmax(ds.values.column(c))?;
                    Ok(BSplineBasisSpec {
                        degree,
                        penalty_order: 2,
                        knotspec: BSplineKnotSpec::Generate {
                            data_range: (minv, maxv),
                            num_internal_knots: n_knots,
                        },
                        double_penalty: smooth_double_penalty,
                        identifiability: BSplineIdentifiability::None,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(SmoothBasisSpec::TensorBSpline {
                feature_cols: cols.to_vec(),
                spec: TensorBSplineSpec {
                    marginalspecs: specs,
                    periods: parse_bc_periods_option(options, cols.len())?,
                    double_penalty: smooth_double_penalty,
                    identifiability: parse_tensor_identifiability(options)?,
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
            Ok(SmoothBasisSpec::BSpline1D {
                feature_col: c,
                spec: BSplineBasisSpec {
                    degree,
                    penalty_order: option_usize(options, "penalty_order").unwrap_or(2),
                    knotspec: BSplineKnotSpec::Generate {
                        data_range: (minv, maxv),
                        num_internal_knots: n_knots,
                    },
                    double_penalty: smooth_double_penalty,
                    identifiability: BSplineIdentifiability::default(),
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
                    periodic: parse_periodic_axes_option(options, cols.len())?,
                    length_scale: option_f64(options, "length_scale").unwrap_or(1.0),
                    double_penalty: smooth_double_penalty,
                    identifiability: parse_spatial_identifiability(options)?,
                    radial_reparam: None,
                },
                input_scales: None,
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
                    periodic: parse_periodic_axes_option(options, cols.len())?,
                    length_scale,
                    power,
                    nullspace_order,
                    identifiability: parse_spatial_identifiability(options)?,
                    aniso_log_scales,
                    operator_penalties,
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

pub fn heuristic_centers(n: usize, d: usize) -> usize {
    default_num_centers(n, d)
}

// ---------------------------------------------------------------------------
// Smooth option parsers
// ---------------------------------------------------------------------------

pub fn parse_ps_internal_knots(
    options: &BTreeMap<String, String>,
    degree: usize,
    default_internal_knots: usize,
) -> Result<(usize, bool), String> {
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
        Ok((k - min_k, false))
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
    match raw.trim().to_ascii_lowercase().as_str() {
        "1/2" | "0.5" | "half" => Ok(MaternNu::Half),
        "3/2" | "1.5" => Ok(MaternNu::ThreeHalves),
        "5/2" | "2.5" => Ok(MaternNu::FiveHalves),
        "7/2" | "3.5" => Ok(MaternNu::SevenHalves),
        "9/2" | "4.5" => Ok(MaternNu::NineHalves),
        _ => Err(format!("unsupported Matern nu '{raw}'")),
    }
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

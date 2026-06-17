//! Apply the `gamfit.fit(..., smooths={symbol: BasisDescriptor})` Python
//! override registry onto a freshly-built `TermCollectionSpec`.
//!
//! This is the second half of the symmetric lowering described in
//! `solver::fit_orchestration::build_termspec_with_geometry_and_overrides`:
//! the formula DSL builds the initial `SmoothBasisSpec`, then this module
//! patches per-term tunables (explicit center matrices, knot vectors, kernel
//! hyperparameters) into the spec in place. The output is bit-identical to
//! what the formula DSL would have produced for an equivalent
//! `smooth(..., centers=..., m=..., length_scale=..., ...)` invocation —
//! the only difference is that user-provided `centers` arrays travel as
//! `CenterStrategy::UserProvided(...)` rather than a data-driven strategy.
//!
//! Matching strategy
//! -----------------
//!
//! Each registry key is a comma-joined column-name list (the Python wrapper
//! normalizes single names and tuples down to this form). We resolve the
//! names to column indices via the dataset headers and look for a smooth
//! term whose `feature_cols` are exactly that set (order-insensitive). When
//! a match is found we mutate the spec in place; an unmatched registry key
//! is reported as a configuration error so silent drops cannot happen.
//!
//! Magic by default — when a descriptor field defaults to `None` we leave
//! the formula-DSL-chosen value alone, so `Duchon(centers=K)` (with K an
//! integer) only swaps in `EqualMass { num_centers: K }` and leaves the
//! kernel order, identifiability, and nullspace order untouched.

use ndarray::{Array1, Array2};
use serde_json::Value as JsonValue;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::inference::data::EncodedDataset as Dataset;
use crate::inference::model::ColumnKindTag;
use crate::terms::basis::{
    BSplineBasisSpec, BSplineKnotSpec, CenterStrategy, ConstantCurvatureBasisSpec, DuchonBasisSpec,
    DuchonNullspaceOrder, MaternBasisSpec, MaternNu, MeasureJetBasisSpec, OneDimensionalBoundary,
    SphereMethod, SphericalSplineBasisSpec, ThinPlateBasisSpec,
};
use crate::terms::smooth::{
    BySmoothKind, ByVariableSpec, SmoothBasisSpec, SmoothTermSpec, TensorBSplineSpec,
    TermCollectionSpec, parse_shape_constraint,
};

/// Apply the Python-side `smooths={...}` registry to a built term collection.
///
/// Returns `Err` when a registry key fails to resolve (unknown column name,
/// no matching smooth term, malformed descriptor), so users see the typo
/// rather than a silent no-op.
pub fn apply_smooth_overrides(
    spec: &mut TermCollectionSpec,
    overrides: &JsonValue,
    data: &Dataset,
    inference_notes: &mut Vec<String>,
) -> Result<(), String> {
    let registry = overrides
        .as_object()
        .ok_or_else(|| "smooths kwarg must be a mapping (symbol -> descriptor)".to_string())?;
    if registry.is_empty() {
        return Ok(());
    }
    let column_index: HashMap<&str, usize> = data
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.as_str(), i))
        .collect();

    for (symbol, descriptor) in registry {
        let descriptor_obj = descriptor
            .as_object()
            .ok_or_else(|| format!("smooths[{symbol:?}] descriptor must be a JSON object"))?;
        let vars = resolve_symbol_columns(symbol, descriptor_obj, &column_index)?;
        let term = locate_smooth_term(spec, &vars).ok_or_else(|| {
            format!(
                "smooths[{symbol:?}] does not match any smooth term in the formula; \
                 expected a smooth on columns {vars:?}",
            )
        })?;
        let kind = descriptor_obj
            .get("kind")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                format!("smooths[{symbol:?}] descriptor missing required \"kind\" field")
            })?;
        apply_one_override(term, kind, descriptor_obj, symbol, inference_notes)?;
        apply_by_variable(
            term,
            descriptor_obj,
            symbol,
            data,
            &column_index,
            inference_notes,
        )?;
    }
    Ok(())
}

/// Wrap the term's basis in the `ByVariable` row-gating envelope when the
/// descriptor carries a `by` key (`Smooth.by` — the per-row multiplier
/// `by · s(x)`). On the descriptor path `by` is the *name* of a data-frame
/// column, resolved here to a `by_col` exactly as the formula `s(x, by=g)`
/// syntax does in `term_builder.rs`. Numeric / binary columns scale the inner
/// smooth (`ByVariableSpec::Numeric`); categorical `by` columns require the
/// per-level term replication the formula builder performs and are not
/// expressible as a single in-place override, so they are rejected with a
/// pointer to the formula syntax.
fn apply_by_variable(
    term: &mut SmoothTermSpec,
    descriptor: &serde_json::Map<String, JsonValue>,
    symbol: &str,
    data: &Dataset,
    column_index: &HashMap<&str, usize>,
    inference_notes: &mut Vec<String>,
) -> Result<(), String> {
    let by_name = match descriptor.get("by") {
        None => return Ok(()),
        Some(v) => v
            .as_str()
            .ok_or_else(|| format!("smooths[{symbol:?}].by must be a column name string"))?,
    };
    let by_col = *column_index.get(by_name).ok_or_else(|| {
        format!(
            "smooths[{symbol:?}].by references unknown column {by_name:?}; \
             known columns: {known:?}",
            known = {
                let mut k: Vec<&&str> = column_index.keys().collect();
                k.sort();
                k
            },
        )
    })?;
    match data.column_kinds.get(by_col).copied().ok_or_else(|| {
        format!("internal column-kind lookup failed for smooths[{symbol:?}].by = {by_name:?}")
    })? {
        ColumnKindTag::Binary | ColumnKindTag::Continuous => {
            // Wrap the (already-overridden) geometric core in place. Taking the
            // basis out via a cheap stand-in avoids cloning the inner spec.
            let inner = std::mem::replace(
                &mut term.basis,
                SmoothBasisSpec::BSpline1D {
                    feature_col: by_col,
                    spec: BSplineBasisSpec {
                        degree: 0,
                        penalty_order: 0,
                        knotspec: BSplineKnotSpec::Generate {
                            data_range: (0.0, 0.0),
                            num_internal_knots: 0,
                        },
                        double_penalty: false,
                        identifiability: Default::default(),
                        boundary: OneDimensionalBoundary::Open,
                        boundary_conditions: Default::default(),
                    },
                },
            );
            term.basis = SmoothBasisSpec::ByVariable {
                inner: Box::new(inner),
                by_col,
                kind: BySmoothKind::Numeric,
                by: ByVariableSpec::Numeric,
            };
            inference_notes.push(format!(
                "smooths[{symbol:?}] gated by numeric column {by_name:?} (by·s(x))",
            ));
            Ok(())
        }
        ColumnKindTag::Categorical => Err(format!(
            "smooths[{symbol:?}].by = {by_name:?} is a categorical column; a factor `by` \
             replicates the smooth per level and is not expressible on the smooths={{}} \
             descriptor path. Use the formula by-smooth syntax instead, e.g. \
             fit(df, \"y ~ s({symbol}, by={by_name})\")."
        )),
    }
}

fn resolve_symbol_columns(
    symbol: &str,
    descriptor: &serde_json::Map<String, JsonValue>,
    column_index: &HashMap<&str, usize>,
) -> Result<Vec<usize>, String> {
    let raw_vars: Vec<String> = if let Some(vars_val) = descriptor.get("vars") {
        let arr = vars_val
            .as_array()
            .ok_or_else(|| format!("smooths[{symbol:?}].vars must be an array of column names"))?;
        let mut out = Vec::with_capacity(arr.len());
        for v in arr {
            let s = v
                .as_str()
                .ok_or_else(|| format!("smooths[{symbol:?}].vars entries must be strings"))?;
            out.push(s.trim().to_string());
        }
        out
    } else {
        symbol
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };
    if raw_vars.is_empty() {
        return Err(format!(
            "smooths[{symbol:?}] resolved to empty variable list"
        ));
    }
    let mut cols = Vec::with_capacity(raw_vars.len());
    for name in &raw_vars {
        let idx = column_index.get(name.as_str()).ok_or_else(|| {
            format!(
                "smooths[{symbol:?}] references unknown column {name:?}; \
                 known columns: {known:?}",
                known = {
                    let mut k: Vec<&&str> = column_index.keys().collect();
                    k.sort();
                    k
                },
            )
        })?;
        cols.push(*idx);
    }
    Ok(cols)
}

fn locate_smooth_term<'a>(
    spec: &'a mut TermCollectionSpec,
    vars: &[usize],
) -> Option<&'a mut SmoothTermSpec> {
    let needle: HashSet<usize> = vars.iter().copied().collect();
    for term in spec.smooth_terms.iter_mut() {
        if smooth_basis_feature_cols(&term.basis)
            .map(|got| got.iter().copied().collect::<HashSet<_>>() == needle)
            .unwrap_or(false)
        {
            return Some(term);
        }
    }
    None
}

/// Recover the underlying `feature_cols` from a `SmoothBasisSpec`, unwrapping
/// `ByVariable` / `FactorSumToZero` / `BySmooth` envelopes the formula DSL
/// inserts around the geometric core. Returns `None` for term shapes that
/// don't carry an explicit column list (e.g. factor smooths whose marginal
/// references a single continuous column via `continuous_cols`).
fn smooth_basis_feature_cols(basis: &SmoothBasisSpec) -> Option<Vec<usize>> {
    match basis {
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => smooth_basis_feature_cols(inner),
        SmoothBasisSpec::BySmooth { smooth, .. } => smooth_basis_feature_cols(smooth),
        SmoothBasisSpec::BSpline1D { feature_col, .. } => Some(vec![*feature_col]),
        SmoothBasisSpec::FactorSmooth { spec } => Some(spec.continuous_cols.clone()),
        SmoothBasisSpec::ThinPlate { feature_cols, .. }
        | SmoothBasisSpec::Sphere { feature_cols, .. }
        | SmoothBasisSpec::ConstantCurvature { feature_cols, .. }
        | SmoothBasisSpec::Matern { feature_cols, .. }
        | SmoothBasisSpec::MeasureJet { feature_cols, .. }
        | SmoothBasisSpec::Duchon { feature_cols, .. }
        | SmoothBasisSpec::Pca { feature_cols, .. }
        | SmoothBasisSpec::TensorBSpline { feature_cols, .. } => Some(feature_cols.clone()),
    }
}

fn apply_one_override(
    term: &mut SmoothTermSpec,
    kind: &str,
    descriptor: &serde_json::Map<String, JsonValue>,
    symbol: &str,
    inference_notes: &mut Vec<String>,
) -> Result<(), String> {
    // Push the descriptor's optional `name` into the term name for downstream
    // diagnostics (purely cosmetic — the term identity is its feature_cols).
    if let Some(name) = descriptor.get("name").and_then(|v| v.as_str())
        && !name.is_empty()
    {
        term.name = name.to_string();
    }

    // Universal shape constraint (`Smooth.shape_constraint`). Stamped onto the
    // term, not the basis: the constraint solver (box-reparam / tangent-LAML)
    // keys off `SmoothTermSpec.shape`. A basis-incompatible request fails
    // loudly downstream via `shape_supports_basis`.
    if let Some(shape_val) = descriptor.get("shape_constraint") {
        let raw = shape_val
            .as_str()
            .ok_or_else(|| format!("smooths[{symbol:?}].shape_constraint must be a string"))?;
        term.shape = parse_shape_constraint(raw).map_err(|e| format!("smooths[{symbol:?}].{e}"))?;
    }

    apply_kind_specific(&mut term.basis, kind, descriptor, symbol)?;

    inference_notes.push(format!(
        "smooths[{symbol:?}] descriptor (kind={kind}) merged onto formula-built term",
    ));
    Ok(())
}

fn apply_kind_specific(
    basis: &mut SmoothBasisSpec,
    kind: &str,
    descriptor: &serde_json::Map<String, JsonValue>,
    symbol: &str,
) -> Result<(), String> {
    // Recurse through the row-gating envelopes so the override targets the
    // geometric core, not the gating wrapper.
    match basis {
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            return apply_kind_specific(inner, kind, descriptor, symbol);
        }
        SmoothBasisSpec::BySmooth { smooth, .. } => {
            return apply_kind_specific(smooth, kind, descriptor, symbol);
        }
        _ => {}
    }

    let normalized = kind.to_ascii_lowercase();
    let normalized = normalized.as_str();
    match (normalized, &mut *basis) {
        ("duchon", SmoothBasisSpec::Duchon { spec, .. })
        | ("tps", SmoothBasisSpec::Duchon { spec, .. }) => apply_duchon(spec, descriptor, symbol),
        ("duchon", SmoothBasisSpec::ThinPlate { spec, .. })
        | ("tps", SmoothBasisSpec::ThinPlate { spec, .. }) => {
            apply_thinplate(spec, descriptor, symbol)
        }
        ("matern", SmoothBasisSpec::Matern { spec, .. }) => apply_matern(spec, descriptor, symbol),
        ("sphere", SmoothBasisSpec::Sphere { spec, .. })
        | ("s2", SmoothBasisSpec::Sphere { spec, .. }) => apply_sphere(spec, descriptor, symbol),
        ("curvature", SmoothBasisSpec::ConstantCurvature { spec, .. })
        | ("curv", SmoothBasisSpec::ConstantCurvature { spec, .. })
        | ("constant_curvature", SmoothBasisSpec::ConstantCurvature { spec, .. })
        | ("mkappa", SmoothBasisSpec::ConstantCurvature { spec, .. }) => {
            apply_constant_curvature(spec, descriptor, symbol)
        }
        ("mjs", SmoothBasisSpec::MeasureJet { spec, .. })
        | ("measurejet", SmoothBasisSpec::MeasureJet { spec, .. })
        | ("measure_jet", SmoothBasisSpec::MeasureJet { spec, .. })
        | ("web", SmoothBasisSpec::MeasureJet { spec, .. }) => {
            apply_measure_jet(spec, descriptor, symbol)
        }
        ("bspline", SmoothBasisSpec::BSpline1D { spec, .. })
        | ("periodic", SmoothBasisSpec::BSpline1D { spec, .. })
        | ("bc", SmoothBasisSpec::BSpline1D { spec, .. }) => {
            apply_bspline_1d(spec, descriptor, symbol)
        }
        ("tensor_bspline", SmoothBasisSpec::TensorBSpline { spec, .. })
        | ("tensor", SmoothBasisSpec::TensorBSpline { spec, .. })
        | ("te", SmoothBasisSpec::TensorBSpline { spec, .. }) => {
            apply_tensor_bspline(spec, descriptor, symbol)
        }
        ("pca", basis @ SmoothBasisSpec::Pca { .. }) => apply_pca(basis, descriptor, symbol),
        ("periodic_spline_curve", _) => apply_periodic_spline_curve_reject(descriptor, symbol),
        ("categorical", _) => apply_categorical_reject(descriptor, symbol),
        (other, _) => Err(format!(
            "smooths[{symbol:?}] descriptor kind={other:?} is not compatible with the \
             formula-built smooth shape (term basis: {})",
            smooth_basis_kind_name(basis),
        )),
    }
}

fn smooth_basis_kind_name(basis: &SmoothBasisSpec) -> &'static str {
    match basis {
        SmoothBasisSpec::ByVariable { .. } => "by_variable",
        SmoothBasisSpec::FactorSumToZero { .. } => "factor_sum_to_zero",
        SmoothBasisSpec::BSpline1D { .. } => "bspline_1d",
        SmoothBasisSpec::BySmooth { .. } => "by_smooth",
        SmoothBasisSpec::FactorSmooth { .. } => "factor_smooth",
        SmoothBasisSpec::ThinPlate { .. } => "thin_plate",
        SmoothBasisSpec::Sphere { .. } => "sphere",
        SmoothBasisSpec::ConstantCurvature { .. } => "constant_curvature",
        SmoothBasisSpec::MeasureJet { .. } => "measurejet",
        SmoothBasisSpec::Matern { .. } => "matern",
        SmoothBasisSpec::Duchon { .. } => "duchon",
        SmoothBasisSpec::Pca { .. } => "pca",
        SmoothBasisSpec::TensorBSpline { .. } => "tensor_bspline",
    }
}

// --------------------------------------------------------------------------
// Kind-specific overrides
// --------------------------------------------------------------------------

fn apply_duchon(
    spec: &mut DuchonBasisSpec,
    descriptor: &serde_json::Map<String, JsonValue>,
    symbol: &str,
) -> Result<(), String> {
    apply_center_strategy(&mut spec.center_strategy, descriptor, symbol)?;
    // `m` is the spline ORDER: it selects the polynomial nullspace the smoother
    // leaves unpenalized (1 -> mean only, 2 -> mean + linear, k -> degree k-1),
    // mirroring `duchon_nullspace_from_m` in gam-pyffi. It is NOT the spectral
    // power. The Riesz spectral power `s` is a separate descriptor key.
    if let Some(m_val) = descriptor.get("m") {
        let m = m_val.as_u64().filter(|m| *m >= 1).ok_or_else(|| {
            format!("smooths[{symbol:?}].m must be a positive integer (spline order)")
        })?;
        spec.nullspace_order = match m {
            1 => DuchonNullspaceOrder::Zero,
            2 => DuchonNullspaceOrder::Linear,
            other => DuchonNullspaceOrder::Degree((other - 1) as usize),
        };
    }
    if let Some(s_val) = descriptor.get("s").or_else(|| descriptor.get("power")) {
        let s = s_val
            .as_f64()
            .ok_or_else(|| format!("smooths[{symbol:?}].s (spectral power) must be a number"))?;
        if !s.is_finite() || s < 0.0 {
            return Err(format!(
                "smooths[{symbol:?}].s (spectral power) must be a non-negative finite value, got {s}"
            ));
        }
        spec.power = s;
    }
    if let Some(ls) = descriptor.get("length_scale") {
        if ls.is_null() {
            spec.length_scale = None;
        } else {
            let v = ls.as_f64().ok_or_else(|| {
                format!("smooths[{symbol:?}].length_scale must be a number or null")
            })?;
            if !v.is_finite() || v <= 0.0 {
                return Err(format!(
                    "smooths[{symbol:?}].length_scale must be a positive finite value, got {v}"
                ));
            }
            spec.length_scale = Some(v);
        }
    }
    if let Some(anis) = descriptor.get("aniso_log_scales") {
        spec.aniso_log_scales = Some(parse_f64_vec(anis, "aniso_log_scales", symbol)?);
    }
    if let Some(per) = descriptor.get("periodic_per_axis") {
        let parsed = parse_periodic_per_axis(per, symbol)?;
        if parsed.iter().any(Option::is_some) {
            spec.periodic = Some(parsed);
        }
    }
    // The Duchon function-norm penalty already spans the polynomial null space,
    // so `DuchonBasisSpec` deliberately carries no `double_penalty` field
    // (`deny_unknown_fields`, locked by
    // `test_duchon_basis_spec_rejects_removed_double_penalty_field`). Python
    // only emits the key when `True`, so reject it loudly rather than drop it.
    if descriptor
        .get("double_penalty")
        .and_then(JsonValue::as_bool)
        == Some(true)
    {
        return Err(format!(
            "smooths[{symbol:?}]: double_penalty is not supported on Duchon smooths; the \
             Duchon function-norm penalty already spans the polynomial null space"
        ));
    }
    Ok(())
}

fn apply_thinplate(
    spec: &mut ThinPlateBasisSpec,
    descriptor: &serde_json::Map<String, JsonValue>,
    symbol: &str,
) -> Result<(), String> {
    apply_center_strategy(&mut spec.center_strategy, descriptor, symbol)?;
    if let Some(ls) = descriptor.get("length_scale").and_then(JsonValue::as_f64) {
        if !ls.is_finite() || ls <= 0.0 {
            return Err(format!(
                "smooths[{symbol:?}].length_scale must be a positive finite value, got {ls}"
            ));
        }
        spec.length_scale = ls;
    }
    if let Some(per) = descriptor.get("periodic_per_axis") {
        let parsed = parse_periodic_per_axis(per, symbol)?;
        if parsed.iter().any(Option::is_some) {
            spec.periodic = Some(parsed);
        }
    }
    if let Some(dp) = descriptor
        .get("double_penalty")
        .and_then(JsonValue::as_bool)
    {
        spec.double_penalty = dp;
    }
    Ok(())
}

fn apply_matern(
    spec: &mut MaternBasisSpec,
    descriptor: &serde_json::Map<String, JsonValue>,
    symbol: &str,
) -> Result<(), String> {
    apply_center_strategy(&mut spec.center_strategy, descriptor, symbol)?;
    if let Some(nu) = descriptor.get("nu").and_then(JsonValue::as_f64) {
        spec.nu = parse_matern_nu(nu, symbol)?;
    }
    if let Some(ls) = descriptor.get("length_scale").and_then(JsonValue::as_f64) {
        if !ls.is_finite() || ls <= 0.0 {
            return Err(format!(
                "smooths[{symbol:?}].length_scale must be a positive finite value, got {ls}"
            ));
        }
        spec.length_scale = ls;
    }
    if let Some(anis) = descriptor.get("aniso_log_scales") {
        spec.aniso_log_scales = Some(parse_f64_vec(anis, "aniso_log_scales", symbol)?);
    }
    if let Some(dp) = descriptor
        .get("double_penalty")
        .and_then(JsonValue::as_bool)
    {
        spec.double_penalty = dp;
    }
    Ok(())
}

fn apply_sphere(
    spec: &mut SphericalSplineBasisSpec,
    descriptor: &serde_json::Map<String, JsonValue>,
    symbol: &str,
) -> Result<(), String> {
    if let Some(centers_val) = descriptor.get("centers") {
        let centers = parse_2d_array(centers_val, "centers", symbol)?;
        if centers.ncols() != 2 {
            return Err(format!(
                "smooths[{symbol:?}].centers must have shape (K, 2) for Sphere; got ({}, {})",
                centers.nrows(),
                centers.ncols(),
            ));
        }
        spec.center_strategy = CenterStrategy::UserProvided(centers);
    } else if let Some(n) = descriptor.get("n_centers").and_then(JsonValue::as_u64) {
        let n = n as usize;
        if n < 2 {
            return Err(format!(
                "smooths[{symbol:?}].n_centers must be at least 2 for Sphere"
            ));
        }
        spec.center_strategy = CenterStrategy::FarthestPoint { num_centers: n };
    }
    if let Some(po) = descriptor.get("penalty_order").and_then(JsonValue::as_u64) {
        spec.penalty_order = po as usize;
    }
    if let Some(rad) = descriptor.get("radians").and_then(JsonValue::as_bool) {
        spec.radians = rad;
    }
    if let Some(kernel) = descriptor.get("kernel").and_then(JsonValue::as_str) {
        let k = kernel.to_ascii_lowercase();
        match k.as_str() {
            "harmonic" => spec.method = SphereMethod::Harmonic,
            "sobolev" | "pseudo" => spec.method = SphereMethod::Wahba,
            other => {
                return Err(format!(
                    "smooths[{symbol:?}].kernel must be one of \
                     \"sobolev\" / \"pseudo\" / \"harmonic\"; got {other:?}"
                ));
            }
        }
    }
    if let Some(double_penalty) = descriptor
        .get("double_penalty")
        .and_then(JsonValue::as_bool)
    {
        spec.double_penalty = double_penalty;
    }
    Ok(())
}

fn apply_constant_curvature(
    spec: &mut ConstantCurvatureBasisSpec,
    descriptor: &serde_json::Map<String, JsonValue>,
    symbol: &str,
) -> Result<(), String> {
    if let Some(centers_val) = descriptor.get("centers") {
        let centers = parse_2d_array(centers_val, "centers", symbol)?;
        if centers.nrows() < 2 {
            return Err(format!(
                "smooths[{symbol:?}].centers must contain at least 2 rows for ConstantCurvature"
            ));
        }
        spec.center_strategy = CenterStrategy::UserProvided(centers);
    } else if let Some(n) = descriptor.get("n_centers").and_then(JsonValue::as_u64) {
        let n = n as usize;
        if n < 2 {
            return Err(format!(
                "smooths[{symbol:?}].n_centers must be at least 2 for ConstantCurvature"
            ));
        }
        spec.center_strategy = CenterStrategy::FarthestPoint { num_centers: n };
    }
    if let Some(kappa) = descriptor.get("kappa").and_then(JsonValue::as_f64) {
        if !kappa.is_finite() {
            return Err(format!("smooths[{symbol:?}].kappa must be finite"));
        }
        spec.kappa = kappa;
    }
    if let Some(ls) = descriptor.get("length_scale").and_then(JsonValue::as_f64) {
        if !(ls.is_finite() && ls > 0.0) {
            return Err(format!(
                "smooths[{symbol:?}].length_scale must be a positive finite number"
            ));
        }
        spec.length_scale = ls;
    }
    if let Some(double_penalty) = descriptor
        .get("double_penalty")
        .and_then(JsonValue::as_bool)
    {
        spec.double_penalty = double_penalty;
    }
    Ok(())
}

fn apply_measure_jet(
    spec: &mut MeasureJetBasisSpec,
    descriptor: &serde_json::Map<String, JsonValue>,
    symbol: &str,
) -> Result<(), String> {
    if let Some(centers_val) = descriptor.get("centers") {
        let centers = parse_2d_array(centers_val, "centers", symbol)?;
        if centers.nrows() < 3 {
            return Err(format!(
                "smooths[{symbol:?}].centers must contain at least 3 rows for MeasureJet"
            ));
        }
        spec.center_strategy = CenterStrategy::UserProvided(centers);
    } else if let Some(n) = descriptor.get("n_centers").and_then(JsonValue::as_u64) {
        let n = n as usize;
        if n < 3 {
            return Err(format!(
                "smooths[{symbol:?}].n_centers must be at least 3 for MeasureJet"
            ));
        }
        spec.center_strategy = CenterStrategy::FarthestPoint { num_centers: n };
    }
    if let Some(s) = descriptor.get("s").and_then(JsonValue::as_f64) {
        if !(s.is_finite() && s > 0.0 && s < 2.0) {
            return Err(format!(
                "smooths[{symbol:?}].s must lie in (0, 2) for the measure-jet affine-jet energy"
            ));
        }
        spec.order_s = s;
    }
    if let Some(alpha) = descriptor.get("alpha").and_then(JsonValue::as_f64) {
        if !alpha.is_finite() {
            return Err(format!("smooths[{symbol:?}].alpha must be finite"));
        }
        spec.alpha = alpha;
    }
    if let Some(tau) = descriptor.get("tau").and_then(JsonValue::as_f64) {
        if !(tau.is_finite() && tau >= 0.0) {
            return Err(format!(
                "smooths[{symbol:?}].tau must be a finite nonnegative number"
            ));
        }
        spec.tau0 = tau;
    }
    if let Some(n) = descriptor.get("scales").and_then(JsonValue::as_u64) {
        spec.num_scales = n as usize;
    }
    if let Some(ls) = descriptor.get("length_scale").and_then(JsonValue::as_f64) {
        if !(ls.is_finite() && ls > 0.0) {
            return Err(format!(
                "smooths[{symbol:?}].length_scale must be a positive finite number"
            ));
        }
        spec.length_scale = ls;
    }
    if let Some(double_penalty) = descriptor
        .get("double_penalty")
        .and_then(JsonValue::as_bool)
    {
        spec.double_penalty = double_penalty;
    }
    // Multiscale (per-scale spectral split + ψ dials + ridge) is an explicit
    // opt-in (#1116); default single-scale at any center count.
    if let Some(multiscale) = descriptor.get("multiscale").and_then(JsonValue::as_bool) {
        spec.multiscale = multiscale;
    }
    // REML-learning the representer length-scale ℓ is explicit opt-in; the
    // default keeps the realized auto/user scale fixed.
    if let Some(learn) = descriptor
        .get("learn_length_scale")
        .and_then(JsonValue::as_bool)
    {
        spec.learn_length_scale = learn;
    }
    Ok(())
}

fn apply_bspline_1d(
    spec: &mut BSplineBasisSpec,
    descriptor: &serde_json::Map<String, JsonValue>,
    symbol: &str,
) -> Result<(), String> {
    if let Some(knots_val) = descriptor.get("knots") {
        let knots = parse_f64_vec(knots_val, "knots", symbol)?;
        spec.knotspec = BSplineKnotSpec::Provided(Array1::from(knots));
    } else if let Some(n) = descriptor.get("n_knots").and_then(JsonValue::as_u64) {
        let n_internal = (n as usize).saturating_sub(spec.degree + 1).max(1);
        spec.knotspec = match &spec.knotspec {
            BSplineKnotSpec::Generate { data_range, .. } => BSplineKnotSpec::Generate {
                data_range: *data_range,
                num_internal_knots: n_internal,
            },
            BSplineKnotSpec::Automatic { placement, .. } => BSplineKnotSpec::Automatic {
                num_internal_knots: Some(n_internal),
                placement: *placement,
            },
            BSplineKnotSpec::PeriodicUniform { data_range, .. } => {
                BSplineKnotSpec::PeriodicUniform {
                    data_range: *data_range,
                    num_basis: n as usize,
                }
            }
            BSplineKnotSpec::Provided(existing) => BSplineKnotSpec::Provided(existing.clone()),
        };
    }
    if let Some(d) = descriptor.get("degree").and_then(JsonValue::as_u64) {
        spec.degree = d as usize;
    }
    if let Some(po) = descriptor.get("penalty_order").and_then(JsonValue::as_u64) {
        spec.penalty_order = po as usize;
    }
    if let Some(dp) = descriptor
        .get("double_penalty")
        .and_then(JsonValue::as_bool)
    {
        spec.double_penalty = dp;
    }
    // `BSpline(periodic=True)` — promote the spec to a cyclic basis, producing
    // a knot/boundary pair bit-identical to the formula DSL `cyclic()`/`cc()`
    // build (see `term_builder.rs` periodic arm). Python emits the key only
    // when `True`.
    if descriptor.get("periodic").and_then(JsonValue::as_bool) == Some(true) {
        let (start, end, num_basis) = match &spec.knotspec {
            BSplineKnotSpec::Generate {
                data_range,
                num_internal_knots,
            } => (
                data_range.0,
                data_range.1,
                num_internal_knots + spec.degree + 1,
            ),
            BSplineKnotSpec::Automatic { .. } => {
                // The data range for an Automatic knot spec is not resolved at
                // override time, so the periodic loop cannot be placed here.
                return Err(format!(
                    "smooths[{symbol:?}]: periodic=True needs a known data range, but the \
                     term uses automatically inferred knots whose domain is not resolved at \
                     override time. Pass knots= with an explicit range, or build the smooth \
                     periodically via the formula DSL `cyclic()`/`cc()`."
                ));
            }
            BSplineKnotSpec::Provided(_) => {
                return Err(format!(
                    "smooths[{symbol:?}]: periodic=True is ambiguous against an explicit open \
                     knot vector. Build the periodic smooth via the formula DSL `cc()`/`cyclic()`."
                ));
            }
            BSplineKnotSpec::PeriodicUniform { .. } => {
                // Already periodic — nothing to do.
                return Ok(());
            }
        };
        spec.knotspec = BSplineKnotSpec::PeriodicUniform {
            data_range: (start, end),
            num_basis,
        };
        spec.boundary = OneDimensionalBoundary::Cyclic { start, end };
    }
    Ok(())
}

fn apply_tensor_bspline(
    spec: &mut TensorBSplineSpec,
    descriptor: &serde_json::Map<String, JsonValue>,
    symbol: &str,
) -> Result<(), String> {
    if let Some(marginals) = descriptor.get("marginals") {
        let arr = marginals.as_array().ok_or_else(|| {
            format!("smooths[{symbol:?}].marginals must be an array of BSpline descriptors")
        })?;
        if arr.len() != spec.marginalspecs.len() {
            return Err(format!(
                "smooths[{symbol:?}].marginals length {} does not match the tensor \
                 smooth dimension {}",
                arr.len(),
                spec.marginalspecs.len(),
            ));
        }
        for (axis, marginal_val) in arr.iter().enumerate() {
            let marginal_obj = marginal_val.as_object().ok_or_else(|| {
                format!("smooths[{symbol:?}].marginals[{axis}] must be a JSON object")
            })?;
            apply_bspline_1d(&mut spec.marginalspecs[axis], marginal_obj, symbol)?;
        }
    }
    if let Some(dp) = descriptor
        .get("double_penalty")
        .and_then(JsonValue::as_bool)
    {
        spec.double_penalty = dp;
    }
    Ok(())
}

fn apply_pca(
    basis: &mut SmoothBasisSpec,
    descriptor: &serde_json::Map<String, JsonValue>,
    symbol: &str,
) -> Result<(), String> {
    let SmoothBasisSpec::Pca {
        feature_cols,
        basis_matrix,
        centered,
        smooth_penalty,
        pca_basis_path,
        chunk_size,
        ..
    } = basis
    else {
        return Err(format!(
            "smooths[{symbol:?}]: internal error — apply_pca dispatched on a non-PCA basis"
        ));
    };

    if let Some(basis_val) = descriptor.get("basis") {
        let parsed = parse_2d_array(basis_val, "basis", symbol)?;
        if parsed.nrows() != feature_cols.len() {
            return Err(format!(
                "smooths[{symbol:?}].basis must have one row per feature column \
                 (expected {} rows for columns {:?}); got {} rows",
                feature_cols.len(),
                feature_cols,
                parsed.nrows(),
            ));
        }
        if let Some(k) = descriptor.get("K").and_then(JsonValue::as_u64) {
            if k as usize != parsed.ncols() {
                return Err(format!(
                    "smooths[{symbol:?}].K ({k}) must equal the number of basis columns ({})",
                    parsed.ncols(),
                ));
            }
        }
        *basis_matrix = parsed;
        // An explicit dense basis overrides any lazy memmap path.
        *pca_basis_path = None;
    } else if let Some(k) = descriptor.get("K").and_then(JsonValue::as_u64) {
        if k as usize != basis_matrix.ncols() {
            return Err(format!(
                "smooths[{symbol:?}].K ({k}) must equal the number of basis columns ({}) on \
                 the formula-built PCA term; pass basis= to change the column count",
                basis_matrix.ncols(),
            ));
        }
    }

    if let Some(lazy_val) = descriptor.get("lazy_path") {
        let path = lazy_val.as_str().ok_or_else(|| {
            format!("smooths[{symbol:?}].lazy_path must be a string filesystem path")
        })?;
        *pca_basis_path = Some(PathBuf::from(path));
    }

    if let Some(c) = descriptor.get("centered").and_then(JsonValue::as_bool) {
        *centered = c;
    }

    if let Some(sp) = descriptor.get("smooth_penalty").and_then(JsonValue::as_f64) {
        if !sp.is_finite() || sp < 0.0 {
            return Err(format!(
                "smooths[{symbol:?}].smooth_penalty must be a non-negative finite value, got {sp}"
            ));
        }
        *smooth_penalty = sp;
    }

    if let Some(cs) = descriptor.get("chunk_size").and_then(JsonValue::as_u64) {
        *chunk_size = (cs as usize).max(1);
    }

    Ok(())
}

/// `PeriodicSplineCurve` is a parametric closed-curve construction whose
/// knot/degree/output-dimension tunables are consumed only while building the
/// basis in the formula DSL; there is no post-build override surface. Honor a
/// name-only / shape-only descriptor (those universal fields are applied in
/// `apply_one_override` before dispatch), but reject any kind-specific tunable
/// loudly rather than silently dropping it.
fn apply_periodic_spline_curve_reject(
    descriptor: &serde_json::Map<String, JsonValue>,
    symbol: &str,
) -> Result<(), String> {
    // Python always emits these with their build defaults; only a value that
    // differs from the formula-DSL default could change the built term, and
    // that change cannot be re-applied post-build.
    let touched = descriptor
        .get("n_knots")
        .and_then(JsonValue::as_u64)
        .is_some_and(|v| v != 20)
        || descriptor
            .get("degree")
            .and_then(JsonValue::as_u64)
            .is_some_and(|v| v != 3)
        || descriptor
            .get("output_dim")
            .and_then(JsonValue::as_u64)
            .is_some_and(|v| v != 1)
        || descriptor
            .get("penalty_order")
            .and_then(JsonValue::as_u64)
            .is_some_and(|v| v != 2);
    if touched {
        return Err(format!(
            "smooths[{symbol:?}]: PeriodicSplineCurve tunables (n_knots / degree / output_dim / \
             penalty_order) are build-time-only and cannot be applied via smooths={{...}}; set \
             them on the formula DSL `pcurve(...)` construction instead. Only the universal \
             name / shape_constraint fields are honored through the override path."
        ));
    }
    Ok(())
}

/// `Categorical` materializes into sum-to-zero contrasts during formula
/// compilation; its `levels` / `n_levels` tunables have no post-build override
/// surface. Honor a name-only / shape-only descriptor; reject the tunables
/// loudly.
fn apply_categorical_reject(
    descriptor: &serde_json::Map<String, JsonValue>,
    symbol: &str,
) -> Result<(), String> {
    let touched = descriptor.contains_key("levels")
        || descriptor
            .get("n_levels")
            .and_then(JsonValue::as_u64)
            .is_some_and(|v| v != 0);
    if touched {
        return Err(format!(
            "smooths[{symbol:?}]: Categorical tunables (levels / n_levels) are consumed during \
             formula compilation and cannot be applied via smooths={{...}}; declare the factor in \
             the formula instead. Only the universal name / shape_constraint fields are honored \
             through the override path."
        ));
    }
    Ok(())
}

// --------------------------------------------------------------------------
// Common parsers
// --------------------------------------------------------------------------

fn apply_center_strategy(
    target: &mut CenterStrategy,
    descriptor: &serde_json::Map<String, JsonValue>,
    symbol: &str,
) -> Result<(), String> {
    if let Some(centers_val) = descriptor.get("centers") {
        let centers = parse_2d_array(centers_val, "centers", symbol)?;
        *target = CenterStrategy::UserProvided(centers);
        return Ok(());
    }
    if let Some(n) = descriptor.get("n_centers").and_then(JsonValue::as_u64) {
        if n == 0 {
            return Err(format!("smooths[{symbol:?}].n_centers must be positive"));
        }
        *target = CenterStrategy::EqualMass {
            num_centers: n as usize,
        };
    }
    Ok(())
}

fn parse_f64_vec(value: &JsonValue, field: &str, symbol: &str) -> Result<Vec<f64>, String> {
    let arr = value
        .as_array()
        .ok_or_else(|| format!("smooths[{symbol:?}].{field} must be a 1-D numeric array"))?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, v) in arr.iter().enumerate() {
        let f = v
            .as_f64()
            .ok_or_else(|| format!("smooths[{symbol:?}].{field}[{i}] must be a finite number"))?;
        if !f.is_finite() {
            return Err(format!(
                "smooths[{symbol:?}].{field}[{i}] must be finite, got {f}"
            ));
        }
        out.push(f);
    }
    Ok(out)
}

fn parse_2d_array(value: &JsonValue, field: &str, symbol: &str) -> Result<Array2<f64>, String> {
    let outer = value.as_array().ok_or_else(|| {
        format!("smooths[{symbol:?}].{field} must be a (K,) or (K, d) numeric array")
    })?;
    if outer.is_empty() {
        return Err(format!(
            "smooths[{symbol:?}].{field} must contain at least one row"
        ));
    }
    // 1-D case: promote (K,) to (K, 1).
    if outer.iter().all(|v| v.is_number()) {
        let mut data = Vec::with_capacity(outer.len());
        for (i, v) in outer.iter().enumerate() {
            let f = v
                .as_f64()
                .ok_or_else(|| format!("smooths[{symbol:?}].{field}[{i}] must be a number"))?;
            if !f.is_finite() {
                return Err(format!(
                    "smooths[{symbol:?}].{field}[{i}] must be finite, got {f}"
                ));
            }
            data.push(f);
        }
        let k = data.len();
        return Array2::from_shape_vec((k, 1), data)
            .map_err(|e| format!("smooths[{symbol:?}].{field} shape conversion failed: {e}"));
    }
    // 2-D case: each row is itself an array of numbers.
    let first_row = outer[0].as_array().ok_or_else(|| {
        format!(
            "smooths[{symbol:?}].{field} must be a uniform 2-D numeric array (got mixed shapes)"
        )
    })?;
    let d = first_row.len();
    if d == 0 {
        return Err(format!(
            "smooths[{symbol:?}].{field} row dimension must be at least 1"
        ));
    }
    let k = outer.len();
    let mut data = Vec::with_capacity(k * d);
    for (i, row_val) in outer.iter().enumerate() {
        let row = row_val
            .as_array()
            .ok_or_else(|| format!("smooths[{symbol:?}].{field}[{i}] must be an array"))?;
        if row.len() != d {
            return Err(format!(
                "smooths[{symbol:?}].{field} is not rectangular: row 0 has width {d}, \
                 row {i} has width {}",
                row.len()
            ));
        }
        for (j, v) in row.iter().enumerate() {
            let f = v
                .as_f64()
                .ok_or_else(|| format!("smooths[{symbol:?}].{field}[{i}][{j}] must be a number"))?;
            if !f.is_finite() {
                return Err(format!(
                    "smooths[{symbol:?}].{field}[{i}][{j}] must be finite, got {f}"
                ));
            }
            data.push(f);
        }
    }
    Array2::from_shape_vec((k, d), data)
        .map_err(|e| format!("smooths[{symbol:?}].{field} shape conversion failed: {e}"))
}

fn parse_periodic_per_axis(value: &JsonValue, symbol: &str) -> Result<Vec<Option<f64>>, String> {
    let arr = value
        .as_array()
        .ok_or_else(|| format!("smooths[{symbol:?}].periodic_per_axis must be an array"))?;
    // We accept either form: per-axis booleans (the Python descriptor's
    // default emission) or per-axis explicit numeric periods. Booleans map
    // to `None` (open axis) or `Some(0.0)` (sentinel meaning "infer from
    // data range" — downstream basis builders treat a non-positive period
    // as a request for data-range inference). Explicit numbers must be
    // strictly positive and finite.
    let mut out = Vec::with_capacity(arr.len());
    for (i, v) in arr.iter().enumerate() {
        if let Some(b) = v.as_bool() {
            if b {
                return Err(format!(
                    "smooths[{symbol:?}].periodic_per_axis[{i}] is `true` without a numeric \
                     period; the override path needs an explicit period (e.g. \
                     `[2.0 * math.pi, None]`). Mixing bool=False with an explicit period for \
                     periodic axes is supported."
                ));
            }
            out.push(None);
            continue;
        }
        if v.is_null() {
            out.push(None);
        } else {
            let f = v.as_f64().ok_or_else(|| {
                format!(
                    "smooths[{symbol:?}].periodic_per_axis[{i}] must be a positive number, bool, or null"
                )
            })?;
            if !f.is_finite() || f <= 0.0 {
                return Err(format!(
                    "smooths[{symbol:?}].periodic_per_axis[{i}] must be positive and finite"
                ));
            }
            out.push(Some(f));
        }
    }
    Ok(out)
}

/// Absolute tolerance for matching a user-supplied `nu` to one of the supported
/// half-integer Matérn smoothness values; loose enough to absorb the float
/// round-trip through JSON, tight enough that no two half-integers collide.
const MATERN_NU_HALF_INTEGER_TOL: f64 = 1e-9;

fn parse_matern_nu(nu: f64, symbol: &str) -> Result<MaternNu, String> {
    // Half-integer match with tolerance.
    let candidates = [
        (0.5, MaternNu::Half),
        (1.5, MaternNu::ThreeHalves),
        (2.5, MaternNu::FiveHalves),
        (3.5, MaternNu::SevenHalves),
        (4.5, MaternNu::NineHalves),
    ];
    for (target, variant) in candidates {
        if (nu - target).abs() < MATERN_NU_HALF_INTEGER_TOL {
            return Ok(variant);
        }
    }
    Err(format!(
        "smooths[{symbol:?}].nu must be one of 0.5, 1.5, 2.5, 3.5, 4.5; got {nu}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terms::basis::{
        BSplineIdentifiability, DuchonNullspaceOrder, MaternNu, SpatialIdentifiability,
    };
    use crate::terms::smooth::ShapeConstraint;
    use serde_json::json;

    fn obj(v: serde_json::Value) -> serde_json::Map<String, JsonValue> {
        v.as_object()
            .expect("test descriptor must be a JSON object")
            .clone()
    }

    fn open_bspline_spec() -> BSplineBasisSpec {
        BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: 5,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
            boundary: OneDimensionalBoundary::Open,
            boundary_conditions: Default::default(),
        }
    }

    fn bspline_term() -> SmoothTermSpec {
        SmoothTermSpec {
            name: "s(x)".to_string(),
            basis: SmoothBasisSpec::BSpline1D {
                feature_col: 0,
                spec: open_bspline_spec(),
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }
    }

    fn thinplate_spec() -> ThinPlateBasisSpec {
        ThinPlateBasisSpec {
            center_strategy: CenterStrategy::EqualMass { num_centers: 10 },
            periodic: None,
            length_scale: 1.0,
            double_penalty: false,
            identifiability: SpatialIdentifiability::default(),
            radial_reparam: None,
        }
    }

    fn matern_spec() -> MaternBasisSpec {
        MaternBasisSpec {
            center_strategy: CenterStrategy::EqualMass { num_centers: 10 },
            periodic: None,
            length_scale: 1.0,
            nu: MaternNu::ThreeHalves,
            include_intercept: false,
            double_penalty: false,
            identifiability: Default::default(),
            aniso_log_scales: None,
            nullspace_shrinkage_survived: None,
        }
    }

    fn duchon_spec() -> DuchonBasisSpec {
        DuchonBasisSpec {
            center_strategy: CenterStrategy::EqualMass { num_centers: 10 },
            periodic: None,
            length_scale: None,
            power: 1.0,
            nullspace_order: DuchonNullspaceOrder::Linear,
            identifiability: SpatialIdentifiability::default(),
            aniso_log_scales: None,
            operator_penalties: Default::default(),
            boundary: OneDimensionalBoundary::Open,
        }
    }

    use crate::inference::model::{DataSchema, SchemaColumn};

    /// Minimal dataset carrying only the `headers` + `column_kinds` that
    /// `apply_smooth_overrides` reads (it never touches `values`/`schema`).
    fn dataset_with(cols: &[(&str, ColumnKindTag)]) -> Dataset {
        Dataset {
            headers: cols.iter().map(|(n, _)| n.to_string()).collect(),
            values: Array2::zeros((0, cols.len())),
            schema: DataSchema {
                columns: cols
                    .iter()
                    .map(|(n, k)| SchemaColumn {
                        name: n.to_string(),
                        kind: *k,
                        levels: Vec::new(),
                    })
                    .collect(),
            },
            column_kinds: cols.iter().map(|(_, k)| *k).collect(),
        }
    }

    fn collection_with(term: SmoothTermSpec) -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: Vec::new(),
            random_effect_terms: Vec::new(),
            smooth_terms: vec![term],
        }
    }

    /// Regression for gam issue #1160: `Smooth(by=...)` on the formula
    /// `smooths={}` descriptor path used to be silently dropped. It must now
    /// reach the fit as a `ByVariable` row-gating envelope, resolving the
    /// `by` column name to the gating `by_col` just like `s(x, by=g)`.
    #[test]
    fn by_column_name_wraps_in_by_variable_envelope() {
        // Columns: x (the smooth), g (the numeric gating multiplier at col 1).
        let data = dataset_with(&[
            ("x", ColumnKindTag::Continuous),
            ("g", ColumnKindTag::Continuous),
        ]);
        let mut spec = collection_with(bspline_term()); // s(x) on feature_col 0
        let overrides = json!({"x": {"kind": "bspline", "by": "g"}});
        let mut notes = Vec::new();

        apply_smooth_overrides(&mut spec, &overrides, &data, &mut notes)
            .expect("by= column-name descriptor must wire end-to-end");

        match &spec.smooth_terms[0].basis {
            SmoothBasisSpec::ByVariable {
                inner,
                by_col,
                kind,
                by,
            } => {
                assert_eq!(*by_col, 1, "by= must resolve to column 'g' (index 1)");
                assert!(matches!(kind, BySmoothKind::Numeric));
                assert!(matches!(by, ByVariableSpec::Numeric));
                assert!(
                    matches!(**inner, SmoothBasisSpec::BSpline1D { feature_col: 0, .. }),
                    "inner geometric core (s(x) on col 0) must be preserved",
                );
            }
            other => panic!("expected ByVariable envelope, got {other:?}"),
        }
    }

    /// `by=` referencing a categorical column is rejected with a pointer to the
    /// formula by-smooth syntax (per-level replication isn't an in-place merge).
    #[test]
    fn by_categorical_column_is_rejected_with_pointer() {
        let data = dataset_with(&[
            ("x", ColumnKindTag::Continuous),
            ("g", ColumnKindTag::Categorical),
        ]);
        let mut spec = collection_with(bspline_term());
        let overrides = json!({"x": {"kind": "bspline", "by": "g"}});
        let mut notes = Vec::new();

        let err = apply_smooth_overrides(&mut spec, &overrides, &data, &mut notes)
            .expect_err("categorical by= must be rejected on the descriptor path");
        assert!(err.contains("categorical"), "got: {err}");
        assert!(err.contains("by-smooth syntax"), "got: {err}");
    }

    /// An unknown `by=` column name surfaces a clear error rather than a panic.
    #[test]
    fn by_unknown_column_errors() {
        let data = dataset_with(&[("x", ColumnKindTag::Continuous)]);
        let mut spec = collection_with(bspline_term());
        let overrides = json!({"x": {"kind": "bspline", "by": "nope"}});
        let mut notes = Vec::new();

        let err = apply_smooth_overrides(&mut spec, &overrides, &data, &mut notes)
            .expect_err("unknown by= column must error");
        assert!(err.contains("unknown column"), "got: {err}");
    }

    #[test]
    fn shape_constraint_string_sets_term_shape() {
        let mut term = bspline_term();
        let descriptor = obj(json!({"kind": "bspline", "shape_constraint": "monotone_increasing"}));
        let mut notes = Vec::new();
        apply_one_override(&mut term, "bspline", &descriptor, "x", &mut notes)
            .expect("valid shape constraint should apply");
        assert_eq!(term.shape, ShapeConstraint::MonotoneIncreasing);

        let mut term2 = bspline_term();
        let descriptor2 = obj(json!({"kind": "bspline", "shape_constraint": "convex"}));
        let mut notes2 = Vec::new();
        apply_one_override(&mut term2, "bspline", &descriptor2, "x", &mut notes2).unwrap();
        assert_eq!(term2.shape, ShapeConstraint::Convex);
    }

    #[test]
    fn shape_constraint_bad_string_errors() {
        let mut term = bspline_term();
        let descriptor = obj(json!({"kind": "bspline", "shape_constraint": "wiggly"}));
        let mut notes = Vec::new();
        let err = apply_one_override(&mut term, "bspline", &descriptor, "x", &mut notes)
            .expect_err("unknown shape constraint must error");
        assert!(err.contains("unknown shape constraint"), "got: {err}");

        let mut term2 = bspline_term();
        let descriptor2 = obj(json!({"kind": "bspline", "shape_constraint": 7}));
        let mut notes2 = Vec::new();
        let err2 = apply_one_override(&mut term2, "bspline", &descriptor2, "x", &mut notes2)
            .expect_err("non-string shape constraint must error");
        assert!(err2.contains("must be a string"), "got: {err2}");
    }

    #[test]
    fn double_penalty_wires_into_thinplate_and_matern() {
        let mut tps = thinplate_spec();
        apply_thinplate(&mut tps, &obj(json!({"double_penalty": true})), "x").unwrap();
        assert!(tps.double_penalty);

        let mut mat = matern_spec();
        apply_matern(&mut mat, &obj(json!({"double_penalty": true})), "x").unwrap();
        assert!(mat.double_penalty);
    }

    #[test]
    fn double_penalty_rejected_for_duchon() {
        let mut duchon = duchon_spec();
        let err = apply_duchon(&mut duchon, &obj(json!({"double_penalty": true})), "x")
            .expect_err("double_penalty on Duchon must be rejected");
        assert!(err.contains("not supported on Duchon"), "got: {err}");

        // A Duchon descriptor without double_penalty (or with it false) is fine.
        let mut duchon_ok = duchon_spec();
        apply_duchon(&mut duchon_ok, &obj(json!({"m": 2})), "x").unwrap();
    }

    #[test]
    fn periodic_true_promotes_generate_to_cyclic() {
        let mut spec = open_bspline_spec(); // Generate { (0,1), 5 internal }, degree 3
        apply_bspline_1d(&mut spec, &obj(json!({"periodic": true})), "x").unwrap();
        match spec.knotspec {
            BSplineKnotSpec::PeriodicUniform {
                data_range,
                num_basis,
            } => {
                assert_eq!(data_range, (0.0, 1.0));
                // num_internal_knots + degree + 1 = 5 + 3 + 1 = 9
                assert_eq!(num_basis, 9);
            }
            other => panic!("expected PeriodicUniform, got {other:?}"),
        }
        match spec.boundary {
            OneDimensionalBoundary::Cyclic { start, end } => {
                assert_eq!((start, end), (0.0, 1.0));
            }
            other => panic!("expected Cyclic boundary, got {other:?}"),
        }
    }

    #[test]
    fn periodic_true_rejects_provided_and_automatic_knots() {
        let mut provided = open_bspline_spec();
        provided.knotspec =
            BSplineKnotSpec::Provided(Array1::from(vec![0.0, 0.25, 0.5, 0.75, 1.0]));
        let err = apply_bspline_1d(&mut provided, &obj(json!({"periodic": true})), "x")
            .expect_err("periodic against explicit knots must error");
        assert!(err.contains("ambiguous"), "got: {err}");

        let mut automatic = open_bspline_spec();
        automatic.knotspec = BSplineKnotSpec::Automatic {
            num_internal_knots: Some(5),
            placement: crate::terms::basis::BSplineKnotPlacement::Quantile,
        };
        let err2 = apply_bspline_1d(&mut automatic, &obj(json!({"periodic": true})), "x")
            .expect_err("periodic against automatic knots must error");
        assert!(err2.contains("data range"), "got: {err2}");
    }

    #[test]
    fn pca_basis_sets_matrix_and_clears_lazy_path() {
        // PCA over two feature columns; basis must have 2 rows.
        let mut basis = SmoothBasisSpec::Pca {
            feature_cols: vec![0, 1],
            basis_matrix: Array2::<f64>::zeros((2, 1)),
            centered: true,
            smooth_penalty: 1.0,
            center_mean: None,
            pca_basis_path: Some(PathBuf::from("/tmp/scores.npy")),
            chunk_size: 4096,
        };
        let descriptor = obj(json!({
            "kind": "pca",
            "basis": [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]],
            "K": 3,
            "centered": false,
            "smooth_penalty": 2.5,
            "chunk_size": 0,
        }));
        apply_pca(&mut basis, &descriptor, "x").unwrap();
        match basis {
            SmoothBasisSpec::Pca {
                basis_matrix,
                centered,
                smooth_penalty,
                pca_basis_path,
                chunk_size,
                ..
            } => {
                assert_eq!(basis_matrix.shape(), &[2, 3]);
                assert!(!centered);
                assert_eq!(smooth_penalty, 2.5);
                assert!(
                    pca_basis_path.is_none(),
                    "explicit basis must clear lazy path"
                );
                assert_eq!(chunk_size, 1, "chunk_size 0 must clamp to 1");
            }
            other => panic!("expected Pca, got {other:?}"),
        }
    }

    #[test]
    fn pca_basis_rejects_row_count_mismatch() {
        let mut basis = SmoothBasisSpec::Pca {
            feature_cols: vec![0, 1, 2],
            basis_matrix: Array2::<f64>::zeros((3, 1)),
            centered: true,
            smooth_penalty: 1.0,
            center_mean: None,
            pca_basis_path: None,
            chunk_size: 4096,
        };
        // 2 rows but 3 feature columns.
        let descriptor = obj(json!({"kind": "pca", "basis": [[1.0], [2.0]]}));
        let err = apply_pca(&mut basis, &descriptor, "x")
            .expect_err("basis row count must match feature column count");
        assert!(err.contains("one row per feature column"), "got: {err}");
    }

    #[test]
    fn pca_k_mismatch_against_built_basis_errors() {
        let mut basis = SmoothBasisSpec::Pca {
            feature_cols: vec![0, 1],
            basis_matrix: Array2::<f64>::zeros((2, 4)),
            centered: true,
            smooth_penalty: 1.0,
            center_mean: None,
            pca_basis_path: None,
            chunk_size: 4096,
        };
        let err = apply_pca(&mut basis, &obj(json!({"kind": "pca", "K": 7})), "x")
            .expect_err("K must match built basis column count");
        assert!(
            err.contains("must equal the number of basis columns"),
            "got: {err}"
        );
    }

    #[test]
    fn periodic_spline_curve_tunables_error_but_name_only_passes() {
        // Defaults (n_knots=20, degree=3, output_dim=1, penalty_order=2) → accepted.
        apply_periodic_spline_curve_reject(
            &obj(json!({
                "kind": "periodic_spline_curve",
                "n_knots": 20,
                "degree": 3,
                "output_dim": 1,
                "penalty_order": 2,
            })),
            "t",
        )
        .expect("default tunables (name-only descriptor) must be accepted");

        let err = apply_periodic_spline_curve_reject(
            &obj(json!({"kind": "periodic_spline_curve", "n_knots": 40})),
            "t",
        )
        .expect_err("non-default tunable must be rejected");
        assert!(err.contains("build-time-only"), "got: {err}");
    }

    #[test]
    fn categorical_tunables_error_but_name_only_passes() {
        apply_categorical_reject(&obj(json!({"kind": "categorical", "n_levels": 0})), "g")
            .expect("default tunables (name-only descriptor) must be accepted");

        let err = apply_categorical_reject(
            &obj(json!({"kind": "categorical", "levels": [0, 1, 2], "n_levels": 3})),
            "g",
        )
        .expect_err("explicit levels must be rejected");
        assert!(err.contains("consumed during"), "got: {err}");
    }
}

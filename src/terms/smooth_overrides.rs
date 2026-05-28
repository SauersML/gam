//! Apply the `gamfit.fit(..., smooths={symbol: BasisDescriptor})` Python
//! override registry onto a freshly-built `TermCollectionSpec`.
//!
//! This is the second half of the symmetric lowering described in
//! `solver::workflow::build_termspec_with_geometry_and_overrides`:
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

use crate::inference::data::EncodedDataset as Dataset;
use crate::terms::basis::{
    BSplineBasisSpec, BSplineKnotSpec, CenterStrategy, DuchonBasisSpec, MaternBasisSpec, MaternNu,
    SphereMethod, SphericalSplineBasisSpec, ThinPlateBasisSpec,
};
use crate::terms::smooth::{
    SmoothBasisSpec, SmoothTermSpec, TensorBSplineSpec, TermCollectionSpec,
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
        let descriptor_obj = descriptor.as_object().ok_or_else(|| {
            format!("smooths[{symbol:?}] descriptor must be a JSON object")
        })?;
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
    }
    Ok(())
}

fn resolve_symbol_columns(
    symbol: &str,
    descriptor: &serde_json::Map<String, JsonValue>,
    column_index: &HashMap<&str, usize>,
) -> Result<Vec<usize>, String> {
    let raw_vars: Vec<String> = if let Some(vars_val) = descriptor.get("vars") {
        let arr = vars_val.as_array().ok_or_else(|| {
            format!("smooths[{symbol:?}].vars must be an array of column names")
        })?;
        let mut out = Vec::with_capacity(arr.len());
        for v in arr {
            let s = v.as_str().ok_or_else(|| {
                format!("smooths[{symbol:?}].vars entries must be strings")
            })?;
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
        return Err(format!("smooths[{symbol:?}] resolved to empty variable list"));
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
        | SmoothBasisSpec::Matern { feature_cols, .. }
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
        ("pca", SmoothBasisSpec::Pca { .. }) => {
            // PCA descriptors only support the formula path today; tunables
            // (basis matrix, lazy_path) are loaded in the formula builder
            // because they need a path resolved against the working
            // directory. Accept the descriptor as a no-op so users can use
            // the same `smooths={...}` dict to attach a name/by/double_penalty
            // to a PCA term without erroring.
            Ok(())
        }
        ("periodic_spline_curve", SmoothBasisSpec::TensorBSpline { .. })
        | ("periodic_spline_curve", SmoothBasisSpec::BSpline1D { .. }) => {
            // PeriodicSplineCurve is a parametric closed-curve construction
            // built directly in the formula DSL; the override surface for
            // it is currently empty (knots / degree / penalty order are
            // already exposed via the DSL itself).
            Ok(())
        }
        ("categorical", _) => Ok(()),
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
    if let Some(power) = descriptor.get("m").and_then(JsonValue::as_f64) {
        spec.power = power;
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
    if let Some(double_penalty) = descriptor.get("double_penalty").and_then(JsonValue::as_bool) {
        spec.double_penalty = double_penalty;
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
            BSplineKnotSpec::PeriodicUniform { data_range, .. } => BSplineKnotSpec::PeriodicUniform {
                data_range: *data_range,
                num_basis: n as usize,
            },
            BSplineKnotSpec::Provided(existing) => BSplineKnotSpec::Provided(existing.clone()),
        };
    }
    if let Some(d) = descriptor.get("degree").and_then(JsonValue::as_u64) {
        spec.degree = d as usize;
    }
    if let Some(po) = descriptor.get("penalty_order").and_then(JsonValue::as_u64) {
        spec.penalty_order = po as usize;
    }
    if let Some(dp) = descriptor.get("double_penalty").and_then(JsonValue::as_bool) {
        spec.double_penalty = dp;
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
    if let Some(dp) = descriptor.get("double_penalty").and_then(JsonValue::as_bool) {
        spec.double_penalty = dp;
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
    let arr = value.as_array().ok_or_else(|| {
        format!("smooths[{symbol:?}].{field} must be a 1-D numeric array")
    })?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, v) in arr.iter().enumerate() {
        let f = v.as_f64().ok_or_else(|| {
            format!("smooths[{symbol:?}].{field}[{i}] must be a finite number")
        })?;
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
            let f = v.as_f64().ok_or_else(|| {
                format!("smooths[{symbol:?}].{field}[{i}] must be a number")
            })?;
            if !f.is_finite() {
                return Err(format!(
                    "smooths[{symbol:?}].{field}[{i}] must be finite, got {f}"
                ));
            }
            data.push(f);
        }
        let k = data.len();
        return Array2::from_shape_vec((k, 1), data).map_err(|e| {
            format!("smooths[{symbol:?}].{field} shape conversion failed: {e}")
        });
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
        let row = row_val.as_array().ok_or_else(|| {
            format!("smooths[{symbol:?}].{field}[{i}] must be an array")
        })?;
        if row.len() != d {
            return Err(format!(
                "smooths[{symbol:?}].{field} is not rectangular: row 0 has width {d}, \
                 row {i} has width {}",
                row.len()
            ));
        }
        for (j, v) in row.iter().enumerate() {
            let f = v.as_f64().ok_or_else(|| {
                format!("smooths[{symbol:?}].{field}[{i}][{j}] must be a number")
            })?;
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

fn parse_periodic_per_axis(
    value: &JsonValue,
    symbol: &str,
) -> Result<Vec<Option<f64>>, String> {
    let arr = value.as_array().ok_or_else(|| {
        format!("smooths[{symbol:?}].periodic_per_axis must be an array")
    })?;
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
        if (nu - target).abs() < 1e-9 {
            return Ok(variant);
        }
    }
    Err(format!(
        "smooths[{symbol:?}].nu must be one of 0.5, 1.5, 2.5, 3.5, 4.5; got {nu}"
    ))
}


//! Shared analytic-penalty descriptor parser.
//!
//! A single source of truth for the analytic-penalty descriptor schema:
//! weight schedules, temperature schedules, dense flattened arrays, the
//! per-kind descriptor fields, and the final [`AnalyticPenaltyRegistry`]
//! construction. Both entry points — the in-process `solver::fit_orchestration`
//! pipeline and the Python FFI (`gam-pyffi`) — parse descriptors here, so
//! workflow and Python users see the same accepted JSON, the same defaults,
//! the same shape checks, and byte-identical error messages for the same
//! analytic penalty.
//!
//! PyFFI only deserializes Python objects/JSON into [`serde_json::Value`] and
//! hands them to [`build_analytic_penalty_registry_from_descriptors`]; it never
//! re-implements any of the parsing below.

use ndarray::{Array1, Array2, Array3};
use serde_json::Value as JsonValue;
use std::sync::Arc;

use gam_problem::schedule::{GumbelTemperatureSchedule, ScheduleKind};
use gam_terms::{
    ARDPenalty, AnalyticPenaltyKind, AnalyticPenaltyRegistry, BlockOrthogonalityPenalty,
    BlockSparsityPenalty, DecoderIncoherencePenalty, DifferenceOpKind, HarmonicRoughnessPenalty,
    IsometryPenalty, IvaeRidgeMeanGauge, SmoothThresholdPenalty,
    MechanismSparsityPenalty, NestedPrefixPenalty, NuclearNormPenalty,
    OrderedBetaBernoulliPenalty, OrthogonalityPenalty,
    ParametricRowPrecisionPriorPenalty, PenaltyConcavity, PenaltyTier, PsiSlice,
    RowPrecisionPriorPenalty, ScadMcpPenalty, ScalarWeightSchedule, ShapeMonotonicityPenalty,
    SoftmaxAssignmentSparsityPenalty, SparsityPenalty, TopKActivationPenalty,
    TotalVariationPenalty,
};

/// A latent block a penalty descriptor can target, identified either by name or
/// by declaration index in the `latents` object.
#[derive(Clone)]
pub(crate) struct LatentPenaltyTarget {
    pub(crate) name: String,
    pub(crate) n: usize,
    pub(crate) d: usize,
}

fn json_u64_to_usize(value: u64, context: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{context} exceeds usize::MAX"))
}

fn json_positive_u64_to_usize(value: u64, context: &str) -> Result<usize, String> {
    if value == 0 {
        return Err(format!("{context} must be > 0"));
    }
    json_u64_to_usize(value, context)
}

/// Parse the `latents` object into the list of penalty targets. Returns an
/// empty list when `latents` is absent or JSON `null`.
pub(crate) fn latent_penalty_targets(
    latents: Option<&JsonValue>,
) -> Result<Vec<LatentPenaltyTarget>, String> {
    let Some(raw) = latents.filter(|value| !value.is_null()) else {
        return Ok(Vec::new());
    };
    let map = raw
        .as_object()
        .ok_or_else(|| "latents must be a JSON object keyed by formula symbol".to_string())?;
    let mut out = Vec::with_capacity(map.len());
    for (key, raw_block) in map {
        let obj = raw_block
            .as_object()
            .ok_or_else(|| format!("latents['{key}'] must be an object"))?;
        let name = obj
            .get("name")
            .and_then(JsonValue::as_str)
            .unwrap_or(key)
            .to_string();
        let n = json_positive_u64_to_usize(
            obj.get("n")
                .and_then(JsonValue::as_u64)
                .ok_or_else(|| format!("latents['{key}'].n is required"))?,
            &format!("latents['{key}'].n"),
        )?;
        let d = json_positive_u64_to_usize(
            obj.get("d")
                .and_then(JsonValue::as_u64)
                .ok_or_else(|| format!("latents['{key}'].d is required"))?,
            &format!("latents['{key}'].d"),
        )?;
        out.push(LatentPenaltyTarget { name, n, d });
    }
    Ok(out)
}

fn penalty_target_for_descriptor<'a>(
    targets: &'a [LatentPenaltyTarget],
    descriptor: &serde_json::Map<String, JsonValue>,
    context: &str,
) -> Result<&'a LatentPenaltyTarget, String> {
    let raw = descriptor
        .get("target")
        .ok_or_else(|| format!("{context}.target is required"))?;
    if let Some(name) = raw.as_str() {
        return targets
            .iter()
            .find(|target| target.name == name)
            .ok_or_else(|| {
                format!(
                    "{context}.target references latent block {name:?}, but latents declares [{}]",
                    targets
                        .iter()
                        .map(|target| target.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            });
    }
    if let Some(index) = raw.as_u64() {
        let index = json_u64_to_usize(index, &format!("{context}.target"))?;
        return targets.get(index).ok_or_else(|| {
            format!(
                "{context}.target references latent index {index}, but latents declares {} block(s)",
                targets.len()
            )
        });
    }
    Err(format!(
        "{context}.target must be a latent block name or index"
    ))
}

fn descriptor_f64(
    descriptor: &serde_json::Map<String, JsonValue>,
    key: &str,
    default: f64,
) -> Result<f64, String> {
    let value = descriptor
        .get(key)
        .and_then(JsonValue::as_f64)
        .unwrap_or(default);
    if !(value.is_finite() && value > 0.0) {
        return Err(format!("analytic penalty {key} must be finite and > 0"));
    }
    Ok(value)
}

fn descriptor_usize(
    descriptor: &serde_json::Map<String, JsonValue>,
    key: &str,
    default: usize,
) -> Result<usize, String> {
    let Some(raw) = descriptor.get(key).and_then(JsonValue::as_u64) else {
        return Ok(default);
    };
    json_positive_u64_to_usize(raw, &format!("analytic penalty {key}"))
}

fn descriptor_no_unknown_keys(
    descriptor: &serde_json::Map<String, JsonValue>,
    context: &str,
    allowed: &[&str],
) -> Result<(), String> {
    for key in descriptor.keys() {
        if !allowed.iter().any(|allowed_key| allowed_key == key) {
            return Err(format!(
                "{context}.{key} is not a recognized field for this analytic penalty"
            ));
        }
    }
    Ok(())
}

fn descriptor_weight_scalar(
    descriptor: &serde_json::Map<String, JsonValue>,
    context: &str,
) -> Result<f64, String> {
    let Some(value) = descriptor.get("weight") else {
        return Ok(1.0);
    };
    if value.as_str() == Some("auto") {
        return Ok(1.0);
    }
    let Some(weight) = value.as_f64() else {
        return Err(format!(
            "{context}.weight must be 'auto' or a finite positive float"
        ));
    };
    if !(weight.is_finite() && weight > 0.0) {
        return Err(format!("{context}.weight must be finite and > 0"));
    }
    Ok(weight)
}

fn descriptor_weight_schedule(
    descriptor: &serde_json::Map<String, JsonValue>,
    context: &str,
) -> Result<Option<ScalarWeightSchedule>, String> {
    let Some(raw_schedule) = descriptor.get("weight_schedule") else {
        return Ok(None);
    };
    if raw_schedule.is_null() {
        return Ok(None);
    }
    let schedule = raw_schedule
        .as_object()
        .ok_or_else(|| format!("{context}.weight_schedule must be an object"))?;
    let w_start = schedule
        .get("w_start")
        .and_then(JsonValue::as_f64)
        .ok_or_else(|| format!("{context}.weight_schedule.w_start must be a finite number"))?;
    let w_end = schedule
        .get("w_end")
        .and_then(JsonValue::as_f64)
        .ok_or_else(|| format!("{context}.weight_schedule.w_end must be a finite number"))?;
    let kind_name = schedule
        .get("kind")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| format!("{context}.weight_schedule.kind is required"))?
        .to_ascii_lowercase()
        .replace('-', "_");
    let kind = match kind_name.as_str() {
        "geometric" => {
            let rate = schedule
                .get("rate")
                .and_then(JsonValue::as_f64)
                .ok_or_else(|| {
                    format!("{context}.weight_schedule.rate is required for geometric")
                })?;
            ScheduleKind::Geometric { rate }
        }
        "linear" => {
            let steps = schedule
                .get("steps")
                .and_then(JsonValue::as_u64)
                .ok_or_else(|| format!("{context}.weight_schedule.steps is required for linear"))?;
            ScheduleKind::Linear {
                steps: json_u64_to_usize(steps, &format!("{context}.weight_schedule.steps"))?,
            }
        }
        "reciprocal_iter" => ScheduleKind::ReciprocalIter,
        other => {
            return Err(format!(
                "{context}.weight_schedule.kind must be geometric, linear, or reciprocal_iter; got {other:?}"
            ));
        }
    };
    let mut parsed = ScalarWeightSchedule::new(w_start, w_end, kind)
        .map_err(|err| format!("{context}.weight_schedule: {err}"))?;
    if let Some(iter_count) = schedule.get("iter_count") {
        let raw_iter_count = iter_count.as_u64().ok_or_else(|| {
            format!("{context}.weight_schedule.iter_count must be a non-negative integer")
        })?;
        parsed.iter_count = json_u64_to_usize(
            raw_iter_count,
            &format!("{context}.weight_schedule.iter_count"),
        )?;
    }
    Ok(Some(parsed))
}

fn descriptor_temperature_schedule(
    descriptor: &serde_json::Map<String, JsonValue>,
    context: &str,
) -> Result<Option<GumbelTemperatureSchedule>, String> {
    let Some(raw_schedule) = descriptor.get("temperature_schedule") else {
        return Ok(None);
    };
    if raw_schedule.is_null() {
        return Ok(None);
    }
    let schedule = raw_schedule
        .as_object()
        .ok_or_else(|| format!("{context}.temperature_schedule must be an object"))?;
    let tau_start = schedule
        .get("tau_start")
        .and_then(JsonValue::as_f64)
        .ok_or_else(|| {
            format!("{context}.temperature_schedule.tau_start must be a finite number")
        })?;
    let tau_min = schedule
        .get("tau_min")
        .or_else(|| schedule.get("tau_end"))
        .and_then(JsonValue::as_f64)
        .ok_or_else(|| format!("{context}.temperature_schedule.tau_min must be a finite number"))?;
    let decay_name = schedule
        .get("decay")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| format!("{context}.temperature_schedule.decay is required"))?
        .to_ascii_lowercase()
        .replace('-', "_");
    let decay = match decay_name.as_str() {
        "geometric" | "exponential" => {
            let rate = schedule
                .get("rate")
                .and_then(JsonValue::as_f64)
                .unwrap_or(0.9);
            ScheduleKind::Geometric { rate }
        }
        "linear" => {
            let steps = schedule
                .get("steps")
                .and_then(JsonValue::as_u64)
                .ok_or_else(|| {
                    format!("{context}.temperature_schedule.steps is required for linear")
                })?;
            ScheduleKind::Linear {
                steps: json_u64_to_usize(steps, &format!("{context}.temperature_schedule.steps"))?,
            }
        }
        "reciprocal_iter" => ScheduleKind::ReciprocalIter,
        other => {
            return Err(format!(
                "{context}.temperature_schedule.decay must be geometric, exponential, linear, or reciprocal_iter; got {other:?}"
            ));
        }
    };
    let mut parsed = GumbelTemperatureSchedule::new(tau_start, tau_min, decay)
        .map_err(|err| format!("{context}.temperature_schedule: {err}"))?;
    if let Some(iter_count) = schedule.get("iter_count") {
        let raw_iter_count = iter_count.as_u64().ok_or_else(|| {
            format!("{context}.temperature_schedule.iter_count must be a non-negative integer")
        })?;
        parsed.iter_count = json_u64_to_usize(
            raw_iter_count,
            &format!("{context}.temperature_schedule.iter_count"),
        )?;
        parsed
            .validate()
            .map_err(|err| format!("{context}.temperature_schedule: {err}"))?;
    }
    Ok(Some(parsed))
}

fn descriptor_difference_op(
    descriptor: &serde_json::Map<String, JsonValue>,
    context: &str,
) -> Result<DifferenceOpKind, String> {
    let op = descriptor
        .get("difference_op")
        .and_then(JsonValue::as_str)
        .unwrap_or("forward_1d")
        .to_ascii_lowercase()
        .replace('-', "_");
    match op.as_str() {
        "forward_1d" => Ok(DifferenceOpKind::ForwardDiff1D),
        "graph_edges" => {
            let raw_edges = descriptor
                .get("edges")
                .and_then(JsonValue::as_array)
                .ok_or_else(|| format!("{context}.edges is required for graph_edges"))?;
            let mut edges = Vec::with_capacity(raw_edges.len());
            for (edge_idx, raw_edge) in raw_edges.iter().enumerate() {
                let pair = raw_edge.as_array().ok_or_else(|| {
                    format!("{context}.edges[{edge_idx}] must be a two-item list")
                })?;
                if pair.len() != 2 {
                    return Err(format!(
                        "{context}.edges[{edge_idx}] must contain exactly two row indices"
                    ));
                }
                let from = pair[0].as_u64().ok_or_else(|| {
                    format!("{context}.edges[{edge_idx}][0] must be a non-negative integer")
                })?;
                let to = pair[1].as_u64().ok_or_else(|| {
                    format!("{context}.edges[{edge_idx}][1] must be a non-negative integer")
                })?;
                edges.push((
                    json_u64_to_usize(from, &format!("{context}.edges[{edge_idx}][0]"))?,
                    json_u64_to_usize(to, &format!("{context}.edges[{edge_idx}][1]"))?,
                ));
            }
            Ok(DifferenceOpKind::GraphEdges(edges))
        }
        other => Err(format!(
            "{context}.difference_op must be forward_1d or graph_edges; got {other:?}"
        )),
    }
}

fn descriptor_array3_flat(
    descriptor: &serde_json::Map<String, JsonValue>,
    data_key: &str,
    shape_key: &str,
    context: &str,
) -> Result<Array3<f64>, String> {
    let shape_values = descriptor
        .get(shape_key)
        .and_then(JsonValue::as_array)
        .ok_or_else(|| format!("{context}.{shape_key} must be a three-item shape list"))?;
    if shape_values.len() != 3 {
        return Err(format!(
            "{context}.{shape_key} must contain exactly three dimensions"
        ));
    }
    let mut shape = [0usize; 3];
    for (idx, raw_dim) in shape_values.iter().enumerate() {
        let dim = raw_dim
            .as_u64()
            .ok_or_else(|| format!("{context}.{shape_key}[{idx}] must be a positive integer"))?;
        shape[idx] = json_positive_u64_to_usize(dim, &format!("{context}.{shape_key}[{idx}]"))?;
    }
    let expected_len = shape[0]
        .checked_mul(shape[1])
        .and_then(|value| value.checked_mul(shape[2]))
        .ok_or_else(|| format!("{context}.{shape_key} overflows usize"))?;
    let values = descriptor
        .get(data_key)
        .and_then(JsonValue::as_array)
        .ok_or_else(|| format!("{context}.{data_key} must be a flattened numeric array"))?;
    if values.len() != expected_len {
        return Err(format!(
            "{context}.{data_key} length {} does not match {shape_key} product {expected_len}",
            values.len()
        ));
    }
    let mut flat = Vec::with_capacity(expected_len);
    for (idx, cell) in values.iter().enumerate() {
        let value = cell
            .as_f64()
            .ok_or_else(|| format!("{context}.{data_key}[{idx}] must be a finite number"))?;
        if !value.is_finite() {
            return Err(format!("{context}.{data_key}[{idx}] must be finite"));
        }
        flat.push(value);
    }
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), flat)
        .map_err(|err| format!("{context}.{data_key} shape reconstruction failed: {err}"))
}

fn descriptor_array1_flat(
    descriptor: &serde_json::Map<String, JsonValue>,
    data_key: &str,
    context: &str,
) -> Result<Array1<f64>, String> {
    let values = descriptor
        .get(data_key)
        .and_then(JsonValue::as_array)
        .ok_or_else(|| format!("{context}.{data_key} must be a flattened numeric array"))?;
    if values.is_empty() {
        return Err(format!("{context}.{data_key} must be non-empty"));
    }
    let mut flat = Vec::with_capacity(values.len());
    for (idx, cell) in values.iter().enumerate() {
        let value = cell
            .as_f64()
            .ok_or_else(|| format!("{context}.{data_key}[{idx}] must be a finite number"))?;
        if !value.is_finite() {
            return Err(format!("{context}.{data_key}[{idx}] must be finite"));
        }
        flat.push(value);
    }
    Ok(Array1::from(flat))
}

fn descriptor_array2_flat(
    descriptor: &serde_json::Map<String, JsonValue>,
    data_key: &str,
    shape_key: &str,
    context: &str,
) -> Result<Array2<f64>, String> {
    let shape_values = descriptor
        .get(shape_key)
        .and_then(JsonValue::as_array)
        .ok_or_else(|| format!("{context}.{shape_key} must be a two-item shape list"))?;
    if shape_values.len() != 2 {
        return Err(format!(
            "{context}.{shape_key} must contain exactly two dimensions"
        ));
    }
    let mut shape = [0usize; 2];
    for (idx, raw_dim) in shape_values.iter().enumerate() {
        let dim = raw_dim
            .as_u64()
            .ok_or_else(|| format!("{context}.{shape_key}[{idx}] must be a positive integer"))?;
        shape[idx] = json_positive_u64_to_usize(dim, &format!("{context}.{shape_key}[{idx}]"))?;
    }
    let expected_len = shape[0]
        .checked_mul(shape[1])
        .ok_or_else(|| format!("{context}.{shape_key} overflows usize"))?;
    let values = descriptor
        .get(data_key)
        .and_then(JsonValue::as_array)
        .ok_or_else(|| format!("{context}.{data_key} must be a flattened numeric array"))?;
    if values.len() != expected_len {
        return Err(format!(
            "{context}.{data_key} length {} does not match {shape_key} product {expected_len}",
            values.len()
        ));
    }
    let mut flat = Vec::with_capacity(expected_len);
    for (idx, cell) in values.iter().enumerate() {
        let value = cell
            .as_f64()
            .ok_or_else(|| format!("{context}.{data_key}[{idx}] must be a finite number"))?;
        if !value.is_finite() {
            return Err(format!("{context}.{data_key}[{idx}] must be finite"));
        }
        flat.push(value);
    }
    Array2::from_shape_vec((shape[0], shape[1]), flat)
        .map_err(|err| format!("{context}.{data_key} shape reconstruction failed: {err}"))
}

/// Build the analytic-penalty registry from `(latents, penalties)` descriptor
/// JSON. This is the single shared parser invoked by both the in-process
/// `solver::fit_orchestration` pipeline and the Python FFI.
///
/// `latents` is the JSON object keyed by formula symbol that declares the
/// penalty targets (`n`, `d`, optional `name`); `penalties` is the JSON array of
/// per-kind descriptors. Either may be absent or `null`, in which case an empty
/// registry is returned (when there are no penalties) or an error is raised
/// (when penalties are requested without any latent block).
pub fn build_analytic_penalty_registry_from_descriptors(
    latents: Option<&JsonValue>,
    penalties: Option<&JsonValue>,
) -> Result<AnalyticPenaltyRegistry, String> {
    let mut registry = AnalyticPenaltyRegistry::new();
    let Some(raw) = penalties.filter(|value| !value.is_null()) else {
        return Ok(registry);
    };
    let items = raw
        .as_array()
        .ok_or_else(|| "penalties must be a list of analytic penalty descriptors".to_string())?;
    let targets = latent_penalty_targets(latents)?;
    if !items.is_empty() && targets.is_empty() {
        return Err("penalties requires latents with at least one latent block".to_string());
    }
    for (idx, raw_item) in items.iter().enumerate() {
        let context = format!("penalties[{idx}]");
        let descriptor = raw_item
            .as_object()
            .ok_or_else(|| format!("{context} must be an object"))?;
        let target = penalty_target_for_descriptor(&targets, descriptor, &context)?;
        let slice_len = target
            .n
            .checked_mul(target.d)
            .ok_or_else(|| format!("{context}.target latent shape overflows usize"))?;
        let slice = PsiSlice::full(slice_len, Some(target.d));
        let kind = descriptor
            .get("kind")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| format!("{context}.kind is required"))?
            .to_ascii_lowercase()
            .replace('-', "_");
        let weight_schedule = descriptor_weight_schedule(descriptor, &context)?;
        match kind.as_str() {
            "isometry" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &["kind", "target", "weight", "p_out", "weight_schedule"],
                )?;
                let weight = descriptor_weight_scalar(descriptor, &context)?;
                let p_out = descriptor_usize(descriptor, "p_out", target.d)?;
                let mut penalty = IsometryPenalty::new_euclidean(slice, p_out);
                penalty.scalar_weight = weight;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::Isometry(Arc::new(penalty)));
            }
            "ard" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &["kind", "target", "weight_schedule"],
                )?;
                let penalty = ARDPenalty::new(slice, target.d);
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::Ard(Arc::new(penalty)));
            }
            "topk" | "topk_activation" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &["kind", "target", "k", "weight", "weight_schedule"],
                )?;
                let k = descriptor_usize(descriptor, "k", 1)?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let penalty = TopKActivationPenalty::new(slice, k, weight)
                    .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::TopKActivation(Arc::new(penalty)));
            }
            "smooth_threshold" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "thresholds",
                        "weight",
                        "smoothing_eps",
                        "weight_schedule",
                    ],
                )?;
                let thresholds = descriptor_array1_flat(descriptor, "thresholds", &context)?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let smoothing_eps = descriptor_f64(descriptor, "smoothing_eps", 1.0e-3)?;
                let penalty = SmoothThresholdPenalty::new(slice, thresholds, weight, smoothing_eps)
                    .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::SmoothThreshold(Arc::new(penalty)));
            }
            "orthogonality" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "weight",
                        "n_eff",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let penalty = OrthogonalityPenalty::new(slice, target.d, weight, n_eff, learnable)
                    .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::Orthogonality(Arc::new(penalty)));
            }
            "sparsity" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "sparsity_kind",
                        "weight",
                        "eps",
                        "eps_weight",
                        "weight_schedule",
                    ],
                )?;
                let weight = descriptor_weight_scalar(descriptor, &context)?;
                let sparsity_kind = descriptor
                    .get("sparsity_kind")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("smooth_l1")
                    .to_ascii_lowercase()
                    .replace('-', "_");
                let eps = descriptor_f64(descriptor, "eps", 1.0e-3)?;
                let eps_weight = descriptor
                    .get("eps_weight")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("fixed")
                    .to_ascii_lowercase()
                    .replace('-', "_");
                let mut penalty = match sparsity_kind.as_str() {
                    "smooth_l1" | "smoothed_l1" => {
                        SparsityPenalty::smoothed_l1(PenaltyTier::Psi, eps)
                    }
                    "log" => SparsityPenalty::log(PenaltyTier::Psi, eps),
                    "hoyer" => Ok(SparsityPenalty::hoyer(PenaltyTier::Psi)),
                    other => Err(format!(
                        "{context}.sparsity_kind must be smooth_l1, hoyer, or log; got {other:?}"
                    )),
                }?;
                penalty.weight = weight;
                let penalty = match eps_weight.as_str() {
                    "fixed" => penalty,
                    "auto" if sparsity_kind == "hoyer" => {
                        return Err(format!(
                            "{context}.eps_weight='auto' is not meaningful for Hoyer sparsity"
                        ));
                    }
                    "auto" => penalty.with_eps_reml(1),
                    other => {
                        return Err(format!(
                            "{context}.eps_weight must be 'auto' or 'fixed'; got {other:?}"
                        ));
                    }
                };
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::Sparsity(Arc::new(penalty)));
            }
            "scad_mcp" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "weight",
                        "n_eff",
                        "gamma",
                        "variant",
                        "smoothing_eps",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let variant = descriptor
                    .get("variant")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("mcp")
                    .to_ascii_lowercase()
                    .replace('-', "_");
                let (variant, gamma_default) = match variant.as_str() {
                    "mcp" => (PenaltyConcavity::Mcp, 2.5),
                    "scad" => (PenaltyConcavity::Scad, 3.7),
                    other => {
                        return Err(format!(
                            "{context}.variant must be 'mcp' or 'scad'; got {other:?}"
                        ));
                    }
                };
                let gamma = descriptor_f64(descriptor, "gamma", gamma_default)?;
                let smoothing_eps = descriptor_f64(descriptor, "smoothing_eps", 1.0e-6)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let penalty = ScadMcpPenalty::new(
                    slice,
                    weight,
                    n_eff,
                    gamma,
                    smoothing_eps,
                    variant,
                    learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::ScadMcp(Arc::new(penalty)));
            }
            "block_orthogonality" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &["kind", "target", "groups", "weight", "n_eff", "learnable"],
                )?;
                let groups = descriptor_axis_groups(descriptor, "groups", &context)?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let penalty =
                    BlockOrthogonalityPenalty::new(slice, groups, weight, n_eff, learnable)
                        .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::BlockOrthogonality(Arc::new(penalty)));
            }
            "decoder_incoherence" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "block_sizes",
                        "p_out",
                        "weight",
                        "learnable",
                        "weight_schedule",
                        "coactivation",
                        "coactivation_shape",
                    ],
                )?;
                // Parse the per-atom decoder basis-function counts M_k. These are
                // a placeholder shape: the real M_k / p_out / target / coactivation
                // are injected at fit time from the live SAE in
                // `SaeManifoldTerm::add_sae_beta_penalty` (#671). The descriptor
                // still constructs a well-formed penalty so the registry / FFI
                // round-trips and the Python smoke test can confirm the descriptor
                // is recognized.
                let raw_block_sizes = descriptor
                    .get("block_sizes")
                    .and_then(JsonValue::as_array)
                    .ok_or_else(|| format!("{context}.block_sizes is required"))?;
                let mut block_sizes = Vec::with_capacity(raw_block_sizes.len());
                for (atom_idx, raw_m) in raw_block_sizes.iter().enumerate() {
                    let raw_m = raw_m.as_u64().ok_or_else(|| {
                        format!("{context}.block_sizes[{atom_idx}] must be a positive integer")
                    })?;
                    let m = json_positive_u64_to_usize(
                        raw_m,
                        &format!("{context}.block_sizes[{atom_idx}]"),
                    )?;
                    block_sizes.push(m);
                }
                let p_out = descriptor_usize(descriptor, "p_out", target.n)?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let total: usize = block_sizes.iter().map(|&m| m * p_out).sum();
                let decoder_slice = PsiSlice {
                    range: 0..total,
                    latent_dim: Some(block_sizes.len()),
                };
                let k = block_sizes.len();
                let coactivation = if descriptor.contains_key("coactivation") {
                    descriptor_array2_flat(
                        descriptor,
                        "coactivation",
                        "coactivation_shape",
                        &context,
                    )?
                } else {
                    // Uniform off-diagonal placeholder co-activation (1.0
                    // off-diag, 0.0 diag); overwritten with the live mean gate
                    // product at SAE fit time.
                    let mut placeholder = Array2::<f64>::from_elem((k, k), 1.0);
                    for d in 0..k {
                        placeholder[[d, d]] = 0.0;
                    }
                    placeholder
                };
                let penalty = DecoderIncoherencePenalty::new(
                    decoder_slice,
                    block_sizes,
                    p_out,
                    coactivation,
                    weight,
                    learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::DecoderIncoherence(Arc::new(penalty)));
            }
            "ordered_beta_bernoulli" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "k_max",
                        "alpha",
                        "tau",
                        "learnable",
                        "learnable_alpha",
                        "temperature_schedule",
                        "weight_schedule",
                    ],
                )?;
                let k_max = descriptor_usize(descriptor, "k_max", target.d)?;
                let alpha = descriptor_f64(descriptor, "alpha", 1.0)?;
                let tau = descriptor_f64(descriptor, "tau", 1.0)?;
                let temperature_schedule = descriptor_temperature_schedule(descriptor, &context)?;
                let learnable = descriptor
                    .get("learnable")
                    .or_else(|| descriptor.get("learnable_alpha"))
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let penalty = OrderedBetaBernoulliPenalty::new(k_max, alpha, tau, learnable);
                let penalty = match temperature_schedule {
                    Some(schedule) => penalty.with_temperature_schedule(schedule),
                    None => penalty,
                };
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::OrderedBetaBernoulli(Arc::new(penalty)));
            }
            "softmax_assignment_sparsity" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "k_atoms",
                        "temperature",
                        "weight_schedule",
                    ],
                )?;
                let k_atoms = descriptor_usize(descriptor, "k_atoms", target.d)?;
                let temperature = descriptor_f64(descriptor, "temperature", 1.0)?;
                let penalty = SoftmaxAssignmentSparsityPenalty::new(k_atoms, temperature);
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::SoftmaxAssignmentSparsity(Arc::new(
                    penalty,
                )));
            }
            "total_variation" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "weight",
                        "n_eff",
                        "difference_op",
                        "edges",
                        "smoothing_eps",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let difference_op = descriptor_difference_op(descriptor, &context)?;
                let smoothing_eps = descriptor_f64(descriptor, "smoothing_eps", 1.0e-6)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let penalty = TotalVariationPenalty::new(
                    weight,
                    n_eff,
                    difference_op,
                    smoothing_eps,
                    learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::TotalVariation(Arc::new(penalty)));
            }
            "harmonic_roughness" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "weight",
                        "n_eff",
                        "row_weights",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let row_weights = descriptor_array1_flat(descriptor, "row_weights", &context)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let penalty = HarmonicRoughnessPenalty::new(weight, n_eff, row_weights, learnable)
                    .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::HarmonicRoughness(Arc::new(penalty)));
            }
            "monotonicity" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "weight",
                        "n_eff",
                        "direction",
                        "smoothing_eps",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let direction = match descriptor.get("direction") {
                    None | Some(JsonValue::Null) => 1.0,
                    Some(value) => value.as_f64().ok_or_else(|| {
                        format!("{context}.direction must be a finite number (+1 or -1)")
                    })?,
                };
                let smoothing_eps = descriptor_f64(descriptor, "smoothing_eps", 1.0e-3)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let penalty = ShapeMonotonicityPenalty::new(
                    weight,
                    n_eff,
                    direction,
                    smoothing_eps,
                    learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::Monotonicity(Arc::new(penalty)));
            }
            "nuclear_norm" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "weight",
                        "n_eff",
                        "smoothing_eps",
                        "max_rank",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let smoothing_eps = descriptor_f64(descriptor, "smoothing_eps", 1.0e-6)?;
                let max_rank = match descriptor.get("max_rank") {
                    None | Some(JsonValue::Null) => None,
                    Some(value) => {
                        let raw = value.as_u64().ok_or_else(|| {
                            format!("{context}.max_rank must be null or a positive integer")
                        })?;
                        Some(json_positive_u64_to_usize(
                            raw,
                            &format!("{context}.max_rank"),
                        )?)
                    }
                };
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let penalty = NuclearNormPenalty::new(
                    slice,
                    weight,
                    n_eff,
                    smoothing_eps,
                    max_rank,
                    learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::NuclearNorm(Arc::new(penalty)));
            }
            "block_sparsity" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "groups",
                        "weight",
                        "n_eff",
                        "smoothing_eps",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let groups = descriptor_axis_groups(descriptor, "groups", &context)?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let smoothing_eps = descriptor_f64(descriptor, "smoothing_eps", 1.0e-6)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let penalty = BlockSparsityPenalty::new(
                    slice,
                    groups,
                    weight,
                    n_eff,
                    smoothing_eps,
                    learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::BlockSparsity(Arc::new(penalty)));
            }
            "mechanism_sparsity" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "feature_groups",
                        "weight",
                        "smoothing_eps",
                        "n_eff",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let raw_groups = descriptor
                    .get("feature_groups")
                    .and_then(JsonValue::as_array)
                    .ok_or_else(|| format!("{context}.feature_groups is required"))?;
                let mut feature_groups = Vec::with_capacity(raw_groups.len());
                let mut max_feature = None::<usize>;
                for (group_idx, raw_group) in raw_groups.iter().enumerate() {
                    let raw_features = raw_group.as_array().ok_or_else(|| {
                        format!("{context}.feature_groups[{group_idx}] must be a list of feature indices")
                    })?;
                    let mut group = Vec::with_capacity(raw_features.len());
                    for (feature_idx, raw_feature) in raw_features.iter().enumerate() {
                        let raw_feature = raw_feature.as_u64().ok_or_else(|| {
                            format!(
                                "{context}.feature_groups[{group_idx}][{feature_idx}] must be a non-negative integer"
                            )
                        })?;
                        let feature = json_u64_to_usize(
                            raw_feature,
                            &format!("{context}.feature_groups[{group_idx}][{feature_idx}]"),
                        )?;
                        max_feature =
                            Some(max_feature.map_or(feature, |current| current.max(feature)));
                        group.push(feature);
                    }
                    feature_groups.push(group);
                }
                let feature_count = max_feature
                    .and_then(|feature| feature.checked_add(1))
                    .ok_or_else(|| format!("{context}.feature_groups must not be empty"))?;
                let target_len = target.d.checked_mul(feature_count).ok_or_else(|| {
                    format!("{context}.target latent_dim × feature_count overflows usize")
                })?;
                let mechanism_slice = PsiSlice::full(target_len, Some(target.d));
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let smoothing_eps = descriptor_f64(descriptor, "smoothing_eps", 1.0e-6)?;
                let n_eff = descriptor_f64(descriptor, "n_eff", target.n as f64)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let penalty = MechanismSparsityPenalty::new(
                    mechanism_slice,
                    feature_groups,
                    weight,
                    smoothing_eps,
                    n_eff,
                    learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::MechanismSparsity(Arc::new(penalty)));
            }
            "row_precision_prior" | "aux_conditional_prior" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "lambda_per_row",
                        "lambda_per_row_shape",
                        "weight",
                        "n_eff",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let lambda_per_row = descriptor_array3_flat(
                    descriptor,
                    "lambda_per_row",
                    "lambda_per_row_shape",
                    &context,
                )?;
                let penalty =
                    RowPrecisionPriorPenalty::new(slice, lambda_per_row, weight, n_eff, learnable)
                        .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::RowPrecisionPrior(Arc::new(penalty)));
            }
            "ivae_ridge_mean_gauge" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "aux",
                        "aux_shape",
                        "ridge_eps",
                        "weight",
                        "n_eff",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let aux = descriptor_array2_flat(descriptor, "aux", "aux_shape", &context)?;
                let ridge_eps = descriptor_f64(descriptor, "ridge_eps", 1.0e-6)?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let penalty =
                    IvaeRidgeMeanGauge::new(slice, aux, ridge_eps, weight, n_eff, learnable)
                        .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::IvaeRidgeMeanGauge(Arc::new(penalty)));
            }
            "parametric_row_precision_prior" | "parametric_aux_conditional_prior" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "aux",
                        "aux_shape",
                        "log_alpha",
                        "raw_beta",
                        "mu",
                        "mu_shape",
                        "weight",
                        "n_eff",
                        "learnable",
                        "weight_schedule",
                    ],
                )?;
                let weight = descriptor_f64(descriptor, "weight", 1.0)?;
                let n_eff = descriptor_usize(descriptor, "n_eff", target.n)?;
                let learnable = descriptor
                    .get("learnable")
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false);
                let aux = descriptor_array2_flat(descriptor, "aux", "aux_shape", &context)?;
                let log_alpha = descriptor_array1_flat(descriptor, "log_alpha", &context)?;
                let raw_beta = descriptor_array1_flat(descriptor, "raw_beta", &context)?;
                let mu = descriptor_array2_flat(descriptor, "mu", "mu_shape", &context)?;
                let penalty = ParametricRowPrecisionPriorPenalty::new(
                    slice, aux, log_alpha, raw_beta, mu, weight, n_eff, learnable,
                )
                .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::ParametricRowPrecisionPrior(Arc::new(
                    penalty,
                )));
            }
            "nested_prefix" | "nestedprefix" => {
                descriptor_no_unknown_keys(
                    descriptor,
                    &context,
                    &[
                        "kind",
                        "target",
                        "prefix_sizes",
                        "shell_weights",
                        "eps",
                        "tier",
                        "weight_schedule",
                    ],
                )?;
                let prefix_values = descriptor
                    .get("prefix_sizes")
                    .and_then(JsonValue::as_array)
                    .ok_or_else(|| {
                        format!(
                            "{context}.prefix_sizes must be a non-empty array of positive integers"
                        )
                    })?;
                if prefix_values.is_empty() {
                    return Err(format!("{context}.prefix_sizes must be non-empty"));
                }
                let mut prefix_sizes: Vec<usize> = Vec::with_capacity(prefix_values.len());
                for (idx, raw) in prefix_values.iter().enumerate() {
                    let cell = raw.as_u64().ok_or_else(|| {
                        format!("{context}.prefix_sizes[{idx}] must be a positive integer")
                    })?;
                    prefix_sizes.push(json_positive_u64_to_usize(
                        cell,
                        &format!("{context}.prefix_sizes[{idx}]"),
                    )?);
                }
                let shell_array = descriptor_array1_flat(descriptor, "shell_weights", &context)?;
                if shell_array.len() != prefix_sizes.len() {
                    return Err(format!(
                        "{context}.shell_weights length {} must equal prefix_sizes length {}",
                        shell_array.len(),
                        prefix_sizes.len()
                    ));
                }
                let shell_weights: Vec<f64> = shell_array.to_vec();
                let eps = descriptor_f64(descriptor, "eps", 1.0e-6)?;
                let tier_str = descriptor
                    .get("tier")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("psi")
                    .to_ascii_lowercase();
                let tier = match tier_str.as_str() {
                    "psi" => PenaltyTier::Psi,
                    "beta" => PenaltyTier::Beta,
                    "rho" => PenaltyTier::Rho,
                    other => {
                        return Err(format!(
                            "{context}.tier must be one of 'psi', 'beta', 'rho'; got {other:?}"
                        ));
                    }
                };
                let penalty =
                    NestedPrefixPenalty::new(slice, tier, prefix_sizes, shell_weights, eps)
                        .map_err(|err| format!("{context}: {err}"))?;
                let penalty = match weight_schedule {
                    Some(schedule) => penalty.with_weight_schedule(schedule),
                    None => penalty,
                };
                registry.push(AnalyticPenaltyKind::NestedPrefix(Arc::new(penalty)));
            }
            other => {
                return Err(format!(
                    "{context}.kind has unsupported analytic penalty {other:?}"
                ));
            }
        }
    }
    Ok(registry)
}

/// Parse a `groups` field — a list of latent-axis index lists — shared by the
/// `block_orthogonality` and `block_sparsity` descriptors.
fn descriptor_axis_groups(
    descriptor: &serde_json::Map<String, JsonValue>,
    key: &str,
    context: &str,
) -> Result<Vec<Vec<usize>>, String> {
    let raw_groups = descriptor
        .get(key)
        .and_then(JsonValue::as_array)
        .ok_or_else(|| format!("{context}.{key} is required"))?;
    let mut groups = Vec::with_capacity(raw_groups.len());
    for (group_idx, raw_group) in raw_groups.iter().enumerate() {
        let raw_axes = raw_group
            .as_array()
            .ok_or_else(|| format!("{context}.{key}[{group_idx}] must be a list of latent axes"))?;
        let mut group = Vec::with_capacity(raw_axes.len());
        for (axis_idx, raw_axis) in raw_axes.iter().enumerate() {
            let raw_axis = raw_axis.as_u64().ok_or_else(|| {
                format!("{context}.{key}[{group_idx}][{axis_idx}] must be a non-negative integer")
            })?;
            let axis = json_u64_to_usize(
                raw_axis,
                &format!("{context}.{key}[{group_idx}][{axis_idx}]"),
            )?;
            group.push(axis);
        }
        groups.push(group);
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    //! Parity / schema-lock tests for the single shared descriptor parser.
    //!
    //! Both the in-process orchestration pipeline (`solver::fit_orchestration`) and the Python
    //! FFI (`gam-pyffi`) call
    //! [`build_analytic_penalty_registry_from_descriptors`] directly — no entry
    //! point owns its own copy of the schema. Exercising the shared parser here
    //! therefore pins one schema, one set of defaults, one set of shape checks,
    //! and one set of error messages for every caller — no entry point can drift
    //! away from these without breaking the suite.

    use super::build_analytic_penalty_registry_from_descriptors;
    use serde_json::json;

    /// A two-block `latents` fixture every accepted-penalty case targets.
    fn latents() -> serde_json::Value {
        json!({
            "z": { "name": "z", "n": 4, "d": 3 },
            "w": { "name": "w", "n": 5, "d": 2 },
        })
    }

    fn kind_tags(penalties: &serde_json::Value) -> Vec<String> {
        let latents = latents();
        let registry =
            build_analytic_penalty_registry_from_descriptors(Some(&latents), Some(penalties))
                .expect("accepted descriptor fixture must build a registry");
        registry
            .penalties
            .iter()
            .map(|penalty| penalty.kind_tag().to_string())
            .collect()
    }

    fn reject(penalties: &serde_json::Value) -> String {
        let latents = latents();
        build_analytic_penalty_registry_from_descriptors(Some(&latents), Some(penalties))
            .expect_err("rejected descriptor fixture must error")
    }

    #[test]
    fn accepts_the_full_kind_surface() {
        // One descriptor per dispatch arm, including the kinds that previously
        // existed only in the PyFFI parser (monotonicity, nested_prefix) and the
        // richer fields the workflow parser used to drop (isometry.p_out,
        // sparsity.eps_weight). Every arm must round-trip to its canonical tag.
        let penalties = json!([
            { "kind": "isometry", "target": "z", "weight": 2.0, "p_out": 3 },
            { "kind": "ard", "target": "z" },
            { "kind": "topk", "target": "z", "k": 2, "weight": 1.5 },
            {
                "kind": "sparsity",
                "target": "z",
                "sparsity_kind": "smooth_l1",
                "eps": 1.0e-3,
                "eps_weight": "auto"
            },
            { "kind": "scad_mcp", "target": "z", "variant": "scad" },
            { "kind": "orthogonality", "target": "z" },
            {
                "kind": "block_orthogonality",
                "target": "z",
                "groups": [[0, 1], [2]]
            },
            {
                "kind": "decoder_incoherence",
                "target": "z",
                "block_sizes": [2, 1],
                "p_out": 3,
                "coactivation": [0.0, 0.25, 0.75, 0.0],
                "coactivation_shape": [2, 2]
            },
            { "kind": "monotonicity", "target": "z", "direction": -1.0 },
            { "kind": "total_variation", "target": "z" },
            {
                "kind": "nested_prefix",
                "target": "z",
                "prefix_sizes": [1, 2],
                "shell_weights": [1.0, 0.5],
                "tier": "psi"
            },
        ]);
        assert_eq!(
            kind_tags(&penalties),
            vec![
                "isometry",
                "ard",
                "topk_activation",
                "sparsity",
                "scad_mcp",
                "orthogonality",
                "block_orthogonality",
                "decoder_incoherence",
                "monotonicity",
                "total_variation",
                "nested_prefix",
            ]
        );
    }

    #[test]
    fn empty_or_null_penalties_build_empty_registry() {
        let latents = latents();
        let empty =
            build_analytic_penalty_registry_from_descriptors(Some(&latents), Some(&json!([])))
                .expect("empty penalties is valid");
        assert!(empty.penalties.is_empty());
        let null =
            build_analytic_penalty_registry_from_descriptors(Some(&latents), Some(&json!(null)))
                .expect("null penalties is valid");
        assert!(null.penalties.is_empty());
        let absent = build_analytic_penalty_registry_from_descriptors(Some(&latents), None)
            .expect("absent penalties is valid");
        assert!(absent.penalties.is_empty());
    }

    #[test]
    fn rejects_unknown_field_with_canonical_message() {
        let penalties = json!([
            { "kind": "isometry", "target": "z", "nonsense": 1 }
        ]);
        assert_eq!(
            reject(&penalties),
            "penalties[0].nonsense is not a recognized field for this analytic penalty"
        );
    }

    #[test]
    fn rejects_unsupported_kind() {
        let penalties = json!([{ "kind": "not_a_penalty", "target": "z" }]);
        assert_eq!(
            reject(&penalties),
            "penalties[0].kind has unsupported analytic penalty \"not_a_penalty\""
        );
    }

    #[test]
    fn rejects_missing_target() {
        let penalties = json!([{ "kind": "ard" }]);
        assert_eq!(reject(&penalties), "penalties[0].target is required");
    }

    #[test]
    fn rejects_penalties_without_latent_blocks() {
        let penalties = json!([{ "kind": "ard", "target": "z" }]);
        let err = build_analytic_penalty_registry_from_descriptors(None, Some(&penalties))
            .expect_err("penalties without latents must error");
        assert_eq!(
            err,
            "penalties requires latents with at least one latent block"
        );
    }

    #[test]
    fn rejects_nonpositive_weight() {
        let penalties = json!([{ "kind": "topk", "target": "z", "weight": 0.0 }]);
        assert_eq!(
            reject(&penalties),
            "analytic penalty weight must be finite and > 0"
        );
    }

    #[test]
    fn rejects_hoyer_with_auto_eps_weight() {
        let penalties = json!([
            {
                "kind": "sparsity",
                "target": "z",
                "sparsity_kind": "hoyer",
                "eps_weight": "auto"
            }
        ]);
        assert_eq!(
            reject(&penalties),
            "penalties[0].eps_weight='auto' is not meaningful for Hoyer sparsity"
        );
    }

    #[test]
    fn rejects_shape_mismatch_in_flattened_array() {
        // nested_prefix shell_weights must match prefix_sizes length — the
        // shared shape check, identical for both entry points.
        let penalties = json!([
            {
                "kind": "nested_prefix",
                "target": "z",
                "prefix_sizes": [1, 2],
                "shell_weights": [1.0]
            }
        ]);
        assert_eq!(
            reject(&penalties),
            "penalties[0].shell_weights length 1 must equal prefix_sizes length 2"
        );
    }

    #[test]
    fn decoder_incoherence_rejects_negative_coactivation_descriptor() {
        let penalties = json!([
            {
                "kind": "decoder_incoherence",
                "target": "z",
                "block_sizes": [1, 1],
                "p_out": 3,
                "coactivation": [0.0, -0.1, 0.0, 0.0],
                "coactivation_shape": [2, 2]
            }
        ]);
        assert_eq!(
            reject(&penalties),
            "penalties[0]: DecoderIncoherencePenalty::new requires finite non-negative coactivation entries"
        );
    }

    #[test]
    fn unknown_target_lists_declared_blocks() {
        let penalties = json!([{ "kind": "ard", "target": "missing" }]);
        // serde_json's default `Map` is a `BTreeMap`, so the declared latent
        // blocks are listed in sorted-key order (`w` before `z`).
        assert_eq!(
            reject(&penalties),
            "penalties[0].target references latent block \"missing\", but latents declares [w, z]"
        );
    }
}

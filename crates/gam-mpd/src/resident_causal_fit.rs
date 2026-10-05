//! Training-only full-sequence KL fitting of a single ordinary graph.
//! Episode controls must be graph inputs; forward hooks are not differentiated.
#[path = "fixed_head_target.rs"]
pub mod fixed_head_target;
use fixed_head_target::{Head, ResidentHead, Target};

use crate::{
    artifact_device::mapped_inlined,
    device_program::DeviceProgram,
    operator_program::{
        FamilyInputs, Node, OperatorBody, OperatorProgram, Slot, SlotValues, exact_precision,
    },
};
use gam_gpu::tensor::{Arithmetic, Device, Indices, Tensor};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Instant,
};
#[derive(Clone)]
pub struct Episode {
    pub label: String,
    pub group: String,
    pub inputs: FamilyInputs,
    pub target_logits: Array2<f64>,
    pub scored: Option<Vec<bool>>,
}
/// Optional compact labels for the same causal fitter and objective.
#[derive(Clone)]
pub struct FixedHeadEpisode {
    pub label: String,
    pub group: String,
    pub inputs: FamilyInputs,
    pub target: Target,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub iterations: usize,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub numeric_bytes: usize,
}
#[derive(Clone, Debug, Serialize)]
pub struct EpisodeMeasurement {
    pub label: String,
    pub group: String,
    pub scored_rows: usize,
    pub mean_kl: f64,
    /// Tail diagnostic only; the optimized objective remains the declared group mean.
    pub maximum_scored_row_kl: f64,
    pub worst_scored_row: usize,
}
#[derive(Clone, Debug, Serialize)]
pub struct Measurement {
    pub objective: f64,
    pub active_group: String,
    pub groups: BTreeMap<String, f64>,
    pub episodes: Vec<EpisodeMeasurement>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Iteration {
    pub step: usize,
    pub measurement: Measurement,
}
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub settings: Settings,
    pub trainable: Vec<usize>,
    pub initial: Measurement,
    pub best: Measurement,
    pub final_measurement: Measurement,
    pub best_step: usize,
    pub iterations: Vec<Iteration>,
    pub planned_numeric_bytes: usize,
    pub complete_episode_forward_passes: usize,
    pub complete_episode_reverse_passes: usize,
    pub seconds: f64,
    pub scope: &'static str,
}
pub struct Fit {
    pub program: OperatorProgram,
    pub report: Report,
}
struct ResidentEpisode {
    family: FamilyInputs,
    raw: BTreeMap<usize, Tensor>,
    target: ResidentTarget,
    flags: Option<Indices>,
    label: String,
    group: String,
    scored_rows: usize,
    scored_mask: Option<Vec<bool>>,
}
enum ResidentTarget {
    Logits(Tensor),
    Fixed {
        target: Target,
        head: Arc<ResidentHead>,
    },
}
fn score(
    p: &DeviceProgram,
    e: &ResidentEpisode,
    trace: &crate::device_program::DeviceTrace,
    gradient: bool,
) -> Result<(Vec<f64>, Option<Tensor>), String> {
    match &e.target {
        ResidentTarget::Logits(target) => {
            let mut logits = p.device().copy(trace.value(p.hidden())?).map_err(error)?;
            let kl = if gradient {
                p.device().kl_rows(target, &mut logits, e.flags.as_ref())
            } else {
                p.device()
                    .kl_score_rows(target, &mut logits, e.flags.as_ref())
            }
            .map_err(error)?;
            Ok((kl, if gradient { Some(logits) } else { None }))
        }
        ResidentTarget::Fixed { target, head } => {
            head.score(p.device(), trace.value(p.hidden())?, target, gradient)
        }
    }
}

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn add(a: usize, b: usize) -> Result<usize, String> {
    a.checked_add(b)
        .ok_or_else(|| "numeric plan overflow".into())
}
fn mul(a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or_else(|| "numeric plan overflow".into())
}
fn validate(
    source: &OperatorProgram,
    episodes: &[Episode],
) -> Result<(OperatorProgram, usize), String> {
    if episodes.is_empty() {
        return Err("empty training episode inventory".into());
    }
    let (expanded, _) = mapped_inlined(source)?;
    let widths = expanded.interfaces().map_err(error)?;
    let classes = widths[expanded.output].width();
    if classes == 0 {
        return Err("zero logit width".into());
    }
    let mut labels = BTreeSet::new();
    for e in episodes {
        if e.label.is_empty() || e.group.is_empty() || !labels.insert(&e.label) {
            return Err("unique nonempty episode labels and nonempty groups required".into());
        }
        if e.inputs.rows == 0
            || e.target_logits.dim() != (e.inputs.rows, classes)
            || e.inputs.slots.len() != expanded.declarations.slots.len()
            || !e.target_logits.iter().all(|x| x.is_finite())
        {
            return Err("invalid episode input/target dimensions or nonfinite target".into());
        }
        if e.scored
            .as_ref()
            .is_some_and(|s| s.len() != e.inputs.rows || !s.iter().any(|x| *x))
        {
            return Err("nonempty scored row domain of exact episode length required".into());
        }
        for (decl, values) in expanded.declarations.slots.iter().zip(&e.inputs.slots) {
            match (decl, values) {
                (Slot::Raw { width }, SlotValues::Raw(x))
                    if x.dim() == (e.inputs.rows, *width) && x.iter().all(|v| v.is_finite()) => {}
                (Slot::Token { domain }, SlotValues::Tokens(tokens))
                    if tokens.len() == e.inputs.rows
                        && tokens.iter().all(|t| {
                            (*t as usize) < expanded.declarations.domains[*domain].size
                        }) => {}
                _ => return Err("episode slots do not match graph declarations".into()),
            }
        }
    }
    Ok((expanded, classes))
}
fn parameter_elements(source: &OperatorProgram, trainable: &[usize]) -> Result<usize, String> {
    let unique: BTreeSet<_> = trainable.iter().copied().collect();
    if unique.len() != trainable.len() {
        return Err("duplicate trainable operators".into());
    }
    trainable.iter().try_fold(0, |count, index| {
        let op = source
            .operators
            .get(*index)
            .ok_or("unknown trainable operator")?;
        match &op.body {
            OperatorBody::Dense {
                values, present, ..
            } if present.iter().all(|p| *p) && values.iter().all(|v| v.is_finite()) => {
                add(count, values.len())
            }
            _ => Err("trainable operators must be full-present finite Dense literals".into()),
        }
    })
}
fn prepare(
    d: &Device,
    source: &OperatorProgram,
    episodes: &[Episode],
    trainable: &[usize],
    limit: usize,
) -> Result<(DeviceProgram, Vec<ResidentEpisode>, usize), String> {
    let (expanded, classes) = validate(source, episodes)?;
    let parameters = mul(parameter_elements(source, trainable)?, 8)?;
    let mut p = DeviceProgram::compile_values_bounded(d, &expanded, limit)?;
    let mut panels = 0;
    let mut trace_peak = 0;
    let mut attention_peak = 0;
    let mut indices = 0;
    // Pointwise law-code arrays live with the compiled graph, independently of operators.
    for node in &expanded.nodes {
        if let Node::Pointwise { input, .. } = node {
            indices = add(indices, mul(widths_for(&expanded, *input)?, 4)?)?;
        }
    }
    let max_factor_rank = expanded
        .operators
        .iter()
        .filter_map(|op| match &op.body {
            OperatorBody::LowRank { left, .. } => Some(left.ncols()),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    for e in episodes {
        panels = add(panels, mul(e.target_logits.len(), 8)?)?;
        for x in &e.inputs.slots {
            match x {
                SlotValues::Raw(x) => panels = add(panels, mul(x.len(), 8)?)?,
                SlotValues::Tokens(x) => indices = add(indices, mul(x.len(), 4)?)?,
            }
        }
        if e.scored.is_some() {
            indices = add(indices, mul(e.inputs.rows, 4)?)?;
        }
        trace_peak = trace_peak.max(mul(p.bytes_per_row(), e.inputs.rows)?);
        // Bound dense attention matrices using rows squared (safe even for multiple sequences).
        // Add q/k rotation, cotangent and temporary vectors. Workspaces are reused sequentially.
        for node in &expanded.nodes {
            if let Node::Attend { query, rotary, .. } = node {
                let width = widths_for(&expanded, *query)?;
                let scratch = add(
                    mul(mul(e.inputs.rows, e.inputs.rows)?, 8 * 12)?,
                    mul(mul(e.inputs.rows, width)?, 8 * 16)?,
                )?;
                attention_peak = attention_peak.max(scratch);
                if rotary.is_some() {
                    indices = add(indices, mul(mul(e.inputs.rows, width)?, 8 * 2)?)?;
                }
            }
        }
    }
    // Includes peak forward values, reverse values/retained cotangents, KL arrays, parameter
    // snapshots, accumulated/episode gradients, moments, updates and promotion/column copies.
    let mut planned = add(p.operator_numeric_bytes()?, mul(parameters, 12)?)?;
    planned = add(planned, panels)?;
    planned = add(planned, indices)?;
    planned = add(planned, mul(trace_peak, 5)?)?;
    planned = add(planned, attention_peak)?;
    let max_rows = episodes
        .iter()
        .map(|e| e.inputs.rows)
        .max()
        .ok_or("empty episodes")?;
    planned = add(planned, mul(mul(max_rows, classes)?, 8 * 4)?)?;
    planned = add(planned, mul(mul(max_rows, max_factor_rank)?, 8 * 4)?)?;
    p.set_arithmetic(Arithmetic::F64);
    if planned > limit {
        return Err(format!(
            "causal fitter numeric plan {planned} exceeds {limit}"
        ));
    }
    p.prepare_dense_parameters(trainable)?;
    let resident = episodes
        .iter()
        .map(|e| {
            let mut family = e.inputs.clone();
            let mut raw = BTreeMap::new();
            for (slot, x) in family.slots.iter_mut().enumerate() {
                if let SlotValues::Raw(values) = x {
                    raw.insert(slot, d.upload(values.view()).map_err(error)?);
                    *values = Array2::zeros((0, values.ncols()));
                }
            }
            Ok(ResidentEpisode {
                family,
                raw,
                target: ResidentTarget::Logits(d.upload(e.target_logits.view()).map_err(error)?),
                flags: e
                    .scored
                    .as_ref()
                    .map(|s| {
                        d.upload_indices(&s.iter().map(|x| u32::from(*x)).collect::<Vec<_>>())
                            .map_err(error)
                    })
                    .transpose()?,
                label: e.label.clone(),
                group: e.group.clone(),
                scored_mask: e.scored.clone(),
                scored_rows: e
                    .scored
                    .as_ref()
                    .map_or(e.inputs.rows, |s| s.iter().filter(|x| **x).count()),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok((p, resident, planned))
}
fn prepare_compact(
    d: &Device,
    source: &OperatorProgram,
    episodes: &[FixedHeadEpisode],
    trainable: &[usize],
    limit: usize,
    tile_rows: usize,
) -> Result<(DeviceProgram, Vec<ResidentEpisode>, usize), String> {
    if episodes.is_empty() || tile_rows == 0 || limit == 0 {
        return Err("nonempty compact episodes, positive tile/budget required".into());
    }
    let (expanded, _) = mapped_inlined(source)?;
    let head = Head::of(&expanded)?;
    if trainable.contains(&head.operator) {
        return Err("compact head cannot be trainable (including tied embedding)".into());
    }
    let prefix = head.prefix(&expanded);
    let mut p = DeviceProgram::compile_values_bounded(d, &prefix, limit)?;
    let mut labels = BTreeSet::new();
    let mut panels = 0usize;
    let mut trace_peak = 0usize;
    let mut attention_peak = 0usize;
    let mut indices = 0usize;
    let mut max_rows = 0usize;
    for node in &prefix.nodes {
        if let Node::Pointwise { input, .. } = node {
            indices = add(indices, mul(widths_for(&prefix, *input)?, 4)?)?;
        }
    }
    for e in episodes {
        if e.label.is_empty() || e.group.is_empty() || !labels.insert(&e.label) {
            return Err("unique nonempty compact labels/groups required".into());
        }
        if e.inputs.rows == 0
            || e.target.rows() != e.inputs.rows
            || e.target.width() != head.embedding.ncols()
            || !head.same(&e.target.head)
            || e.target.entropy.len() != e.inputs.rows
            || e.target.entropy.iter().any(|v| !v.is_finite())
            || e.inputs.slots.len() != expanded.declarations.slots.len()
        {
            return Err("compact target dimensions or fixed-head identity mismatch".into());
        }
        if e.target
            .scored
            .as_ref()
            .is_some_and(|s| s.len() != e.inputs.rows || !s.iter().any(|v| *v))
        {
            return Err("invalid compact scored domain".into());
        }
        panels = add(panels, e.target.numeric_bytes())?;
        for (decl, values) in expanded.declarations.slots.iter().zip(&e.inputs.slots) {
            match (decl, values) {
                (Slot::Raw { width }, SlotValues::Raw(x))
                    if x.dim() == (e.inputs.rows, *width) && x.iter().all(|v| v.is_finite()) =>
                {
                    panels = add(panels, mul(x.len(), 8)?)?;
                }
                (Slot::Token { domain }, SlotValues::Tokens(tokens))
                    if tokens.len() == e.inputs.rows
                        && tokens.iter().all(|t| {
                            (*t as usize) < expanded.declarations.domains[*domain].size
                        }) =>
                {
                    indices = add(indices, mul(tokens.len(), 4)?)?;
                }
                _ => return Err("compact episode slots mismatch".into()),
            }
        }
        if e.target.scored.is_some() {
            indices = add(indices, mul(e.inputs.rows, 4)?)?;
        }
        max_rows = max_rows.max(e.inputs.rows);
        trace_peak = trace_peak.max(mul(p.bytes_per_row(), e.inputs.rows)?);
        for node in &prefix.nodes {
            if let Node::Attend { query, rotary, .. } = node {
                let width = widths_for(&prefix, *query)?;
                attention_peak = attention_peak.max(add(
                    mul(mul(e.inputs.rows, e.inputs.rows)?, 8 * 12)?,
                    mul(mul(e.inputs.rows, width)?, 8 * 16)?,
                )?);
                if rotary.is_some() {
                    indices = add(indices, mul(mul(e.inputs.rows, width)?, 8 * 2)?)?;
                }
            }
        }
    }
    let mut planned = add(
        p.operator_numeric_bytes()?,
        mul(mul(parameter_elements(source, trainable)?, 8)?, 12)?,
    )?;
    planned = add(planned, mul(head.embedding.len(), 8)?)?;
    planned = add(planned, panels)?;
    planned = add(planned, indices)?;
    planned = add(planned, mul(trace_peak, 5)?)?;
    planned = add(planned, attention_peak)?;
    planned = add(
        planned,
        mul(mul(tile_rows.min(max_rows), head.embedding.nrows())?, 8 * 2)?,
    )?;
    planned = add(planned, mul(mul(max_rows, head.embedding.ncols())?, 8 * 6)?)?;
    if planned > limit {
        return Err(format!(
            "compact fitter numeric plan {planned} exceeds {limit}"
        ));
    }
    p.set_arithmetic(Arithmetic::F64);
    p.prepare_dense_parameters(trainable)?;
    let resident_head = Arc::new(ResidentHead::new(d, &head, tile_rows)?);
    let resident = episodes
        .iter()
        .map(|e| {
            let mut family = e.inputs.clone();
            let mut raw = BTreeMap::new();
            for (slot, x) in family.slots.iter_mut().enumerate() {
                if let SlotValues::Raw(values) = x {
                    raw.insert(slot, d.upload(values.view()).map_err(error)?);
                    *values = Array2::zeros((0, values.ncols()));
                }
            }
            let scored = e.target.scored.clone();
            Ok(ResidentEpisode {
                family,
                raw,
                target: ResidentTarget::Fixed {
                    target: e.target.clone(),
                    head: resident_head.clone(),
                },
                flags: None,
                label: e.label.clone(),
                group: e.group.clone(),
                scored_rows: scored
                    .as_ref()
                    .map_or(e.inputs.rows, |s| s.iter().filter(|v| **v).count()),
                scored_mask: scored,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok((p, resident, planned))
}

fn widths_for(p: &OperatorProgram, node: usize) -> Result<usize, String> {
    Ok(p.node_interface(node).map_err(error)?.width())
}
fn forward(
    p: &DeviceProgram,
    e: &ResidentEpisode,
) -> Result<crate::device_program::DeviceTrace, String> {
    let d = p.device();
    // forward_given owns its arguments; these resident copies preserve the reusable panel.
    let given = e
        .raw
        .iter()
        .map(|(slot, x)| Ok((*slot, d.copy(x).map_err(error)?)))
        .collect::<Result<_, String>>()?;
    p.forward_given(&e.family, given)
}
fn scan(p: &DeviceProgram, episodes: &[ResidentEpisode]) -> Result<Measurement, String> {
    let mut groups: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    let mut scores = Vec::new();
    for e in episodes {
        let trace = forward(p, e)?;
        let (kl, _) = score(p, e, &trace, false)?;
        if kl.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite training KL".into());
        }
        let mean = kl.iter().sum::<f64>() / e.scored_rows as f64;
        if !mean.is_finite() {
            return Err("nonfinite episode mean KL".into());
        }
        let entry = groups.entry(e.group.clone()).or_insert((0., 0));
        entry.0 += mean;
        entry.1 += 1;
        let (worst_scored_row, maximum_scored_row_kl) = kl
            .iter()
            .enumerate()
            .filter(|(i, _)| e.scored_mask.as_ref().is_none_or(|mask| mask[*i]))
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, value)| (i, *value))
            .ok_or("empty scored domain")?;
        scores.push(EpisodeMeasurement {
            label: e.label.clone(),
            group: e.group.clone(),
            scored_rows: e.scored_rows,
            mean_kl: mean,
            maximum_scored_row_kl,
            worst_scored_row,
        });
    }
    let groups: BTreeMap<_, _> = groups
        .into_iter()
        .map(|(name, (sum, count))| (name, sum / count as f64))
        .collect();
    let mut active = None;
    for (group, value) in &groups {
        if !value.is_finite() {
            return Err("nonfinite group mean".into());
        }
        if active.as_ref().is_none_or(|(_, best)| value > best) {
            active = Some((group.clone(), *value));
        }
    }
    let (active_group, objective) = active.ok_or("empty groups")?;
    Ok(Measurement {
        objective,
        active_group,
        groups,
        episodes: scores,
    })
}
fn gradient(
    p: &DeviceProgram,
    episodes: &[ResidentEpisode],
    group: &str,
    trainable: &[usize],
) -> Result<BTreeMap<usize, Tensor>, String> {
    let d = p.device();
    let count = episodes.iter().filter(|e| e.group == group).count();
    if count == 0 {
        return Err("unknown active group".into());
    }
    let mut gradients = trainable
        .iter()
        .map(|index| {
            let a = p.dense_parameter(*index)?;
            Ok((*index, d.zeros(a.rows(), a.cols()).map_err(error)?))
        })
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    for e in episodes.iter().filter(|e| e.group == group) {
        let trace = forward(p, e)?;
        let (kl, seed) = score(p, e, &trace, true)?;
        let seed = seed.ok_or("missing KL gradient seed")?;
        if kl.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite gradient KL".into());
        }
        let weight = 1. / (count as f64 * e.scored_rows as f64);
        let mut scaled = d.zeros(seed.rows(), seed.cols()).map_err(error)?;
        d.axpy(&mut scaled, weight, &seed).map_err(error)?;
        let (_, per_episode) = p.vjp_values_dense(
            &trace,
            BTreeMap::from([(p.hidden(), scaled)]),
            &[],
            trainable,
            Arithmetic::F64,
        )?;
        for index in trainable {
            d.axpy(
                gradients.get_mut(index).ok_or("gradient buffer")?,
                1.,
                per_episode.get(index).ok_or("episode gradient")?,
            )
            .map_err(error)?;
        }
    }
    Ok(gradients)
}
/// Full-episode measurement only; never updates parameters or chooses a validation checkpoint.
pub fn measure(
    d: &Device,
    source: &OperatorProgram,
    episodes: &[Episode],
    numeric_bytes: usize,
) -> Result<Measurement, String> {
    let (p, resident, _) = prepare(d, source, episodes, &[], numeric_bytes)?;
    scan(&p, &resident)
}
fn validate_settings(trainable: &[usize], settings: &Settings) -> Result<(), String> {
    if trainable.is_empty()
        || settings.numeric_bytes == 0
        || !settings.learning_rate.is_finite()
        || settings.learning_rate <= 0.
        || !settings.epsilon.is_finite()
        || settings.epsilon <= 0.
        || !settings.beta1.is_finite()
        || !settings.beta2.is_finite()
        || !(0. ..1.).contains(&settings.beta1)
        || !(0. ..1.).contains(&settings.beta2)
        || u64::try_from(settings.iterations).is_err()
    {
        return Err("invalid causal fitter settings or empty trainable inventory".into());
    }
    Ok(())
}

/// Same optimizer/group objective, with row-tiled fixed-head sufficient statistics.
/// The head must remain bias-free, dense, numerically identical and untrainable.
pub fn fit_fixed_head(
    d: &Device,
    source: &OperatorProgram,
    episodes: &[FixedHeadEpisode],
    trainable: &[usize],
    settings: Settings,
    head_tile_rows: usize,
) -> Result<Fit, String> {
    let started = Instant::now();
    validate_settings(trainable, &settings)?;
    let (p, resident, planned) = prepare_compact(
        d,
        source,
        episodes,
        trainable,
        settings.numeric_bytes,
        head_tile_rows,
    )?;
    fit_prepared(
        d, source, p, resident, planned, trainable, settings, started,
    )
}
pub fn measure_fixed_head(
    d: &Device,
    source: &OperatorProgram,
    episodes: &[FixedHeadEpisode],
    numeric_bytes: usize,
    head_tile_rows: usize,
) -> Result<Measurement, String> {
    let (p, resident, _) =
        prepare_compact(d, source, episodes, &[], numeric_bytes, head_tile_rows)?;
    scan(&p, &resident)
}

pub fn fit(
    d: &Device,
    source: &OperatorProgram,
    episodes: &[Episode],
    trainable: &[usize],
    settings: Settings,
) -> Result<Fit, String> {
    let started = Instant::now();
    validate_settings(trainable, &settings)?;
    let (p, resident, planned) = prepare(d, source, episodes, trainable, settings.numeric_bytes)?;
    fit_prepared(
        d, source, p, resident, planned, trainable, settings, started,
    )
}
fn fit_prepared(
    d: &Device,
    source: &OperatorProgram,
    mut p: DeviceProgram,
    resident: Vec<ResidentEpisode>,
    planned: usize,
    trainable: &[usize],
    settings: Settings,
    started: Instant,
) -> Result<Fit, String> {
    let mut moments = BTreeMap::new();
    for index in trainable {
        let a = p.dense_parameter(*index)?;
        moments.insert(
            *index,
            (
                d.zeros(a.rows(), a.cols()).map_err(error)?,
                d.zeros(a.rows(), a.cols()).map_err(error)?,
            ),
        );
    }
    let initial = scan(&p, &resident)?;
    let mut best = initial.clone();
    let mut best_step = 0;
    let mut snapshot = trainable
        .iter()
        .map(|index| Ok((*index, d.copy(p.dense_parameter(*index)?).map_err(error)?)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let mut history = vec![Iteration {
        step: 0,
        measurement: initial.clone(),
    }];
    let mut reverse = 0;
    let mut forwards = resident.len();
    for step in 1..=settings.iterations {
        let active_group = &history
            .last()
            .ok_or("missing training score")?
            .measurement
            .active_group;
        let gradients = gradient(&p, &resident, active_group, trainable)?;
        let count = resident.iter().filter(|e| e.group == *active_group).count();
        reverse += count;
        forwards += count;
        for index in trainable {
            let mut next = d.copy(p.dense_parameter(*index)?).map_err(error)?;
            let (m, v) = moments.get_mut(index).ok_or("moment pair")?;
            d.adam(
                &mut next,
                (m, v),
                gradients.get(index).ok_or("parameter gradient")?,
                settings.learning_rate,
                (settings.beta1, settings.beta2, settings.epsilon),
                step as u64,
            )
            .map_err(error)?;
            p.replace_dense_parameter(*index, next)?;
        }
        let measurement = scan(&p, &resident)?;
        forwards += resident.len();
        if measurement.objective < best.objective {
            best = measurement.clone();
            best_step = step;
            for index in trainable {
                snapshot.insert(*index, d.copy(p.dense_parameter(*index)?).map_err(error)?);
            }
        }
        history.push(Iteration { step, measurement });
    }
    for (index, value) in snapshot {
        p.replace_dense_parameter(index, value)?;
    }
    let final_measurement = scan(&p, &resident)?;
    forwards += resident.len();
    let mut fitted = source.clone();
    for index in trainable {
        let values = d.download(p.dense_parameter(*index)?).map_err(error)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite fitted coefficient".into());
        }
        let precision = exact_precision(values.iter().copied()).map_err(error)?;
        let op = Arc::make_mut(&mut fitted.operators[*index]);
        let OperatorBody::Dense {
            values: stored,
            precision: stored_precision,
            ..
        } = &mut op.body
        else {
            return Err("source parameter kind changed".into());
        };
        *stored = values;
        *stored_precision = precision;
    }
    let compact = resident
        .iter()
        .any(|e| matches!(e.target, ResidentTarget::Fixed { .. }));
    Ok(Fit {
        program: fitted,
        report: Report {
            settings,
            trainable: trainable.to_vec(),
            initial,
            best,
            final_measurement,
            best_step,
            iterations: history,
            planned_numeric_bytes: planned,
            complete_episode_forward_passes: forwards,
            complete_episode_reverse_passes: reverse,
            seconds: started.elapsed().as_secs_f64(),
            scope: if compact {
                "Proposal fitting only: same maximum named-group mean objective and Adam/best-TRAIN loop. Immutable fixed bias-free full-vocabulary head targets retain E^T p and sum p log p; each row-tiled candidate KL is logZ(Eh)-mu.h+c and hidden seed E^T q-mu. Head input is after original final normalization, with full-prefix ordinary VJP; both target and candidate unscored seeds are zero. F64 vendor exp/log/GEMM are operational, not certified real-arithmetic intervals; final ordinary artifact acceptance is unchanged. Numeric plan counts resident compact labels, head, inputs, traces/gradients/parameter buffers and conservative attention/tiled vocabulary scratch. Host metadata/source, context/library/allocator scratch excluded."
            } else {
                "Proposal fitting only: maximum named-group mean of equal-weight episode means over declared scored rows. F64 KL q-p gradients, full-sequence ordinary reverse including controls supplied as Raw graph inputs. One parameter owner across every episode. Best TRAIN objective only; no validation input/selection. Vendor exp/log and neural arithmetic are not certified intervals; ordinary serialized acceptance is separate. Numeric plan includes fixed operators (all table/product roles), resident inputs/targets/flags, full trace/cotangents/KL scratch, conservative dense attention scratch and parameter/moment/snapshot/update buffers. Excludes host panels/source bytes, CUDA context/library/allocator/register/spill scratch; token/rotation preparation peak conservatively counted, not a measured memory claim."
            },
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_program::{Declarations, Interface, Operator, Scale, SequenceLayout};
    fn program(causal: bool, weight: f64) -> OperatorProgram {
        let interface = Interface::native(2).expect("interface");
        let values = ndarray::array![[weight, 0.2], [-0.3, 0.7]];
        let operator = Operator::dense(
            "reader",
            interface.clone(),
            interface,
            values.clone(),
            exact_precision(values.iter().copied()).expect("precision"),
            Default::default(),
        )
        .expect("reader");
        let mut nodes = vec![
            Node::Raw { slot: 0 },
            Node::Affine {
                terms: vec![(0, 0)],
                bias: None,
            },
        ];
        let output = if causal {
            nodes.push(Node::Attend {
                query: 0,
                key: 0,
                value: 1,
                scale: Scale::InverseSqrt(2),
                rotary: None,
                causal: true,
            });
            2
        } else {
            nodes.push(Node::Raw { slot: 1 });
            nodes.push(Node::Hadamard { left: 1, right: 2 });
            3
        };
        OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: if causal {
                    vec![Slot::Raw { width: 2 }]
                } else {
                    vec![Slot::Raw { width: 2 }; 2]
                },
                parameters: 0,
            },
            bases: vec![],
            operators: vec![Arc::new(operator)],
            rules: vec![],
            nodes,
            output,
        }
    }
    fn episode(
        source: &OperatorProgram,
        teacher: &OperatorProgram,
        label: &str,
        group: &str,
        x: Array2<f64>,
        mask: Option<Array2<f64>>,
        scored: Option<Vec<bool>>,
    ) -> Episode {
        let rows = x.nrows();
        let mut slots = vec![SlotValues::Raw(x)];
        if let Some(mask) = mask {
            slots.push(SlotValues::Raw(mask));
        }
        let inputs = FamilyInputs {
            rows,
            slots,
            layout: if source.declarations.slots.len() == 1 {
                Some(SequenceLayout {
                    sequence: vec![0; rows],
                    position: (0..rows as u32).collect(),
                })
            } else {
                None
            },
        };
        let trace = teacher.execute(&inputs, false).expect("ordinary teacher");
        Episode {
            label: label.into(),
            group: group.into(),
            inputs,
            target_logits: trace.values[teacher.output].clone(),
            scored,
        }
    }
    fn settings() -> Settings {
        Settings {
            iterations: 60,
            learning_rate: 0.03,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            numeric_bytes: 1 << 24,
        }
    }
    fn finite_difference(source: &OperatorProgram, episodes: &[Episode], group: &str) -> f64 {
        let d = Device::host();
        let h = 1e-5;
        let mut values = Vec::new();
        for sign in [-1., 1.] {
            let mut p = source.clone();
            let op = Arc::make_mut(&mut p.operators[0]);
            if let OperatorBody::Dense {
                values: matrix,
                precision,
                ..
            } = &mut op.body
            {
                matrix[[0, 0]] += sign * h;
                *precision = exact_precision(matrix.iter().copied()).expect("perturbed precision");
            }
            values.push(
                measure(&d, &p, episodes, 1 << 24)
                    .expect("perturbed measure")
                    .groups[group],
            );
        }
        (values[1] - values[0]) / (2. * h)
    }
    #[test]
    fn equal_episode_weights_scored_rows_and_raw_controls_match_group_gradient() {
        let source = program(false, 0.4);
        let teacher = program(false, 1.1);
        let episodes = vec![
            episode(
                &source,
                &teacher,
                "first",
                "active",
                ndarray::array![[1., 0.5], [-0.4, 1.], [0.2, -0.5]],
                Some(ndarray::array![[1., 1.], [0.5, 0.5], [0., 0.]]),
                Some(vec![true, true, false]),
            ),
            episode(
                &source,
                &teacher,
                "second",
                "active",
                ndarray::array![[2., -0.1]],
                Some(ndarray::array![[0.8, 0.8]]),
                None,
            ),
            episode(
                &source,
                &source,
                "inactive",
                "other",
                ndarray::array![[1., 1.]],
                Some(ndarray::array![[1., 1.]]),
                None,
            ),
        ];
        let d = Device::host();
        let (p, resident, _) = prepare(&d, &source, &episodes, &[0], 1 << 24).expect("prepare");
        let measurement = scan(&p, &resident).expect("measure");
        assert_eq!(measurement.active_group, "active");
        let mean = (measurement.episodes[0].mean_kl + measurement.episodes[1].mean_kl) / 2.;
        assert_eq!(measurement.groups["active"], mean);
        let gradients = gradient(&p, &resident, "active", &[0]).expect("group gradient");
        let actual = d.download(&gradients[&0]).expect("download")[[0, 0]];
        assert!((actual - finite_difference(&source, &episodes, "active")).abs() < 1e-8);
        let fit = fit(&d, &source, &episodes, &[0], settings()).expect("fit");
        assert!(fit.report.best.objective < fit.report.initial.objective * 0.1);
        assert_eq!(
            fit.report.final_measurement.objective,
            fit.report.best.objective
        );
        assert_eq!(
            source.operators[0].matrix()[[0, 0]],
            0.4,
            "source immutable"
        );
        assert_eq!(fit.program.nodes, source.nodes, "same graph and indices");
        let replay = measure(&d, &fit.program, &episodes, 1 << 24).expect("exported measure");
        assert_eq!(replay.objective, fit.report.best.objective);
    }
    #[test]
    fn causal_attention_full_sequence_gradient_includes_unscored_prefix() {
        let source = program(true, 0.4);
        let teacher = program(true, 1.1);
        // Only the last row is scored; previous rows must still participate in its attention.
        let episodes = vec![episode(
            &source,
            &teacher,
            "causal",
            "run",
            ndarray::array![[2., 0.2], [-1., 0.6], [0.1, 0.4]],
            None,
            Some(vec![false, false, true]),
        )];
        let d = Device::host();
        let (p, resident, _) =
            prepare(&d, &source, &episodes, &[0], 1 << 24).expect("prepare attention");
        let gradients = gradient(&p, &resident, "run", &[0]).expect("causal gradient");
        let actual = d.download(&gradients[&0]).expect("download")[[0, 0]];
        assert!((actual - finite_difference(&source, &episodes, "run")).abs() < 1e-8);
        assert!(actual.abs() > 1e-4, "nonzero full-sequence gradient");
        let fit = fit(&d, &source, &episodes, &[0], settings()).expect("causal fit");
        assert!(fit.report.best.objective < fit.report.initial.objective * 0.1);
        assert_eq!(fit.report.complete_episode_reverse_passes, 60);
    }
    #[test]
    fn invalid_domains_nonfinite_targets_and_small_budget_are_errors() {
        let p = program(false, 0.4);
        let mut e = episode(
            &p,
            &p,
            "clean",
            "clean",
            ndarray::array![[1., 0.]],
            Some(ndarray::array![[1., 1.]]),
            None,
        );
        let d = Device::host();
        assert!(measure(&d, &p, &[e.clone()], 1).is_err());
        e.scored = Some(vec![false]);
        assert!(measure(&d, &p, &[e.clone()], 1 << 24).is_err());
        e.scored = None;
        e.target_logits[[0, 0]] = f64::NAN;
        assert!(measure(&d, &p, &[e], 1 << 24).is_err());
    }
    fn with_fixed_head(mut p: OperatorProgram, scale: f64) -> OperatorProgram {
        let input = p.output;
        let hidden = p.nodes.len();
        p.nodes.push(Node::RmsNorm {
            input,
            epsilon: 1e-6,
        });
        let values = ndarray::array![
            [scale, 0.3 * scale],
            [-0.4 * scale, 0.7 * scale],
            [0.2 * scale, -0.8 * scale]
        ];
        let operator = p.operators.len();
        p.operators.push(Arc::new(
            Operator::dense(
                "fixed head",
                Interface::native(3).expect("classes"),
                Interface::native(2).expect("hidden"),
                values.clone(),
                exact_precision(values.iter().copied()).expect("head precision"),
                Default::default(),
            )
            .expect("head"),
        ));
        p.output = p.nodes.len();
        p.nodes.push(Node::Affine {
            terms: vec![(hidden, operator)],
            bias: None,
        });
        p
    }
    #[test]
    fn compact_labels_match_full_kl_and_attention_rms_parameter_gradients() {
        let source = with_fixed_head(program(true, 0.4), 1.);
        let teacher = with_fixed_head(program(true, 1.1), 1.);
        let full = vec![episode(
            &source,
            &teacher,
            "causal",
            "active",
            ndarray::array![[1., 0.5], [-0.4, 1.], [0.2, -0.5]],
            None,
            Some(vec![false, true, true]),
        )];
        let d = Device::host();
        let targeter =
            fixed_head_target::Teacher::new(&d, &teacher, 1, 1 << 24).expect("compact teacher");
        let compact = vec![FixedHeadEpisode {
            label: "causal".into(),
            group: "active".into(),
            inputs: full[0].inputs.clone(),
            target: targeter
                .target(&full[0].inputs, full[0].scored.as_deref())
                .expect("project teacher"),
        }];
        let (a, ae, _) = prepare(&d, &source, &full, &[0], 1 << 24).expect("full prepare");
        let (b, be, _) =
            prepare_compact(&d, &source, &compact, &[0], 1 << 24, 2).expect("compact prepare");
        let av = scan(&a, &ae).expect("full score");
        let bv = scan(&b, &be).expect("compact score");
        assert!((av.objective - bv.objective).abs() < 2e-14);
        let ag = gradient(&a, &ae, "active", &[0]).expect("full gradient");
        let bg = gradient(&b, &be, "active", &[0]).expect("compact gradient");
        let ag = d.download(&ag[&0]).expect("full gradient download");
        let bg = d.download(&bg[&0]).expect("compact gradient download");
        for (x, y) in ag.iter().zip(bg.iter()) {
            assert!((x - y).abs() < 2e-13, "{x} vs {y}");
        }
        let trace = forward(&b, &be[0]).expect("prefix");
        let (_, seed) = score(&b, &be[0], &trace, true).expect("hidden seed");
        let seed = d
            .download(&seed.expect("gradient seed"))
            .expect("seed download");
        assert!(seed.row(0).iter().all(|v| *v == 0.));
        // Unscored earlier parent still affects scored future rows through causal attention.
        assert!(bg.iter().any(|v| v.abs() > 1e-6));
        let fit = fit_fixed_head(
            &d,
            &source,
            &compact,
            &[0],
            Settings {
                iterations: 2,
                ..settings()
            },
            2,
        )
        .expect("same optimizer");
        assert!(fit.report.best.objective <= fit.report.initial.objective);
    }
    #[test]
    fn compact_confident_logits_underflow_and_head_identity_guards() {
        let d = Device::host();
        let source = with_fixed_head(program(true, 0.4), 1000.);
        let teacher = with_fixed_head(program(true, 1.1), 1000.);
        let full = vec![episode(
            &source,
            &teacher,
            "confident",
            "active",
            ndarray::array![[1., 0.5], [-0.4, 1.]],
            None,
            None,
        )];
        let targeter = fixed_head_target::Teacher::new(&d, &teacher, 1, 1 << 24).expect("teacher");
        let compact = vec![FixedHeadEpisode {
            label: "confident".into(),
            group: "active".into(),
            inputs: full[0].inputs.clone(),
            target: targeter
                .target(&full[0].inputs, None)
                .expect("underflow-safe target"),
        }];
        let a = measure(&d, &source, &full, 1 << 24).expect("direct KL");
        let b = measure_fixed_head(&d, &source, &compact, 1 << 24, 1).expect("compact KL");
        assert!((a.objective - b.objective).abs() < 1e-10);
        assert!(fit_fixed_head(&d, &source, &compact, &[1], settings(), 1).is_err());
        let other = with_fixed_head(program(true, 0.4), 999.);
        assert!(measure_fixed_head(&d, &other, &compact, 1 << 24, 1).is_err());
        let mut biased = source.clone();
        if let Node::Affine { bias, .. } = &mut biased.nodes[source.output] {
            *bias = Some(0);
        }
        assert!(fixed_head_target::Teacher::new(&d, &biased, 1, 1 << 24).is_err());
    }
}

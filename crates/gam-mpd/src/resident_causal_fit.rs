//! Training-only full-sequence KL fitting of a single ordinary graph.
//! Episode controls must be graph inputs; forward hooks are not differentiated.
//! Fitting only proposes: every step runs each group's episodes as one appended batch in the
//! proposal arithmetic, and exact F64 scans choose the checkpoint and make every reported
//! measurement.
#[path = "fixed_head_target.rs"]
pub mod fixed_head_target;
use fixed_head_target::{Head, ResidentHead, Target};

use crate::{
    artifact_device::mapped_inlined,
    device_program::{DeviceProgram, DeviceTrace},
    operator_program::{
        FamilyInputs, Node, OperatorBody, OperatorProgram, Slot, SlotValues, exact_precision,
    },
};
use gam_gpu::{
    tensor::{Arithmetic, ColumnBlocks, Device, Indices, Op, Storage, Tensor},
    trace,
};
use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, btree_map::Entry},
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

/// Fixed native-space observations of the candidate's autonomous source-node values.
/// `scale` and `weight` are declared constants, never recomputed from candidate values.
#[derive(Clone)]
pub struct NativeResponseTarget {
    pub label: String,
    pub source_node: usize,
    pub values: Array2<f64>,
    pub scored: Option<Vec<bool>>,
    pub scale: f64,
    pub weight: f64,
}
/// Episode labels bind response targets without changing the existing episode APIs.
pub type NativeResponses = BTreeMap<String, Vec<NativeResponseTarget>>;
#[derive(Clone, Debug, Serialize)]
pub struct ResponseMeasurement {
    pub label: String,
    pub source_node: usize,
    pub scored_rows: usize,
    pub scale: f64,
    pub weight: f64,
    pub mean_normalized_squared_error: f64,
    pub weighted_loss: f64,
}

/// Arithmetic of the proposal's forward and reverse products. Exact scans always use F64.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FitArithmetic {
    F64,
    F32,
    Tf32,
}
impl FitArithmetic {
    fn device(self) -> Arithmetic {
        match self {
            Self::F64 => Arithmetic::F64,
            Self::F32 => Arithmetic::F32,
            Self::Tf32 => Arithmetic::Tf32,
        }
    }
}
// On the L40, F32 SGEMM runs at the TF32 tensor-core rate and its logits keep every KL
// difference that TF32 rounding would blur.
fn fast_arithmetic() -> FitArithmetic {
    FitArithmetic::F32
}
fn exact_scan_interval() -> usize {
    32
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
    #[serde(default)]
    pub schedule: Option<BatchSchedule>,
    /// Arithmetic of every proposal forward and reverse product.
    #[serde(default = "fast_arithmetic")]
    pub arithmetic: FitArithmetic,
    /// Steps between exact F64 scans, plus one at the end. Each rescores the best proposal
    /// measured since the previous scan; only rescored parameters can become the checkpoint.
    #[serde(default = "exact_scan_interval")]
    pub exact_scan_every: usize,
}
/// Proposal optimization only. Group weights are softmax weights of the latest complete
/// proposal measurement, made every `scan_every` steps; complete causal sequences cycle
/// within each group (tokens are never sliced).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchSchedule {
    pub episodes_per_group: usize,
    pub scan_every: usize,
    pub temperature: f64,
}
#[derive(Clone, Debug, Serialize)]
pub struct EpisodeMeasurement {
    pub label: String,
    pub group: String,
    pub scored_rows: usize,
    pub mean_kl: f64,
    pub total_loss: f64,
    pub responses: Vec<ResponseMeasurement>,
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
/// One exact F64 scan.
#[derive(Clone, Debug, Serialize)]
pub struct Iteration {
    pub step: usize,
    pub measurement: Measurement,
    /// The same parameters' objective in the proposal arithmetic (a parity diagnostic).
    pub proposal_objective: Option<f64>,
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
    pub exact_scans: usize,
    /// The device and storage the proposals ran on.
    pub proposal_device: String,
    /// Wall seconds of proposal steps and of exact scans.
    pub proposal_seconds: f64,
    pub exact_seconds: f64,
    pub seconds: f64,
    pub scope: &'static str,
}
pub struct Fit {
    pub program: OperatorProgram,
    pub report: Report,
}
/// One episode. Its raw inputs and target are its rows of its group's batch.
struct ResidentEpisode {
    family: FamilyInputs,
    batch: usize,
    offset: usize,
    responses: Vec<ResidentResponse>,
    flags: Option<Indices>,
    label: String,
    group: String,
    scored_rows: usize,
    scored_mask: Option<Vec<bool>>,
}
/// A group's episodes appended in episode order: one proposal forward and one reverse.
struct Batch {
    family: FamilyInputs,
    raw: BTreeMap<usize, Tensor>,
    target: ResidentTarget,
    flags: Option<Indices>,
    members: Vec<usize>,
}
struct Resident {
    episodes: Vec<ResidentEpisode>,
    batches: Vec<Batch>,
    /// The graph the programs execute (the fixed-head prefix for compact labels).
    executable: OperatorProgram,
}
struct ResidentResponse {
    label: String,
    source_node: usize,
    node: usize,
    target: Tensor,
    coefficients: Tensor,
    blocks: ColumnBlocks,
    scored: Option<Vec<bool>>,
    scored_rows: usize,
    scale: f64,
    weight: f64,
}
enum ResidentTarget {
    Logits(Tensor),
    Fixed {
        target: Target,
        head: Arc<ResidentHead>,
    },
}
/// Per-row output KL against the trace's output; with `gradient`, also the hidden seed.
fn score(
    p: &DeviceProgram,
    target: &ResidentTarget,
    flags: Option<&Indices>,
    trace: &DeviceTrace,
    gradient: bool,
) -> Result<(Vec<f64>, Option<Tensor>), String> {
    match target {
        ResidentTarget::Logits(target) => {
            let mut logits = p.device().copy(trace.value(p.hidden())?).map_err(error)?;
            let kl = if gradient {
                p.device().kl_rows(target, &mut logits, flags)
            } else {
                p.device().kl_score_rows(target, &mut logits, flags)
            }
            .map_err(error)?;
            Ok((kl, if gradient { Some(logits) } else { None }))
        }
        ResidentTarget::Fixed { target, head } => head.score(
            p.device(),
            trace.value(p.hidden())?,
            target,
            gradient,
            p.arithmetic(),
        ),
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
/// Episode indices of each group, in group name order.
fn group_members<'a>(groups: impl Iterator<Item = &'a str>) -> Vec<Vec<usize>> {
    let mut members: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, group) in groups.enumerate() {
        members.entry(group).or_default().push(index);
    }
    members.into_values().collect()
}
/// The family with its raw slots emptied (their values live on the device).
fn hollow(inputs: &FamilyInputs) -> FamilyInputs {
    let mut family = inputs.clone();
    for x in &mut family.slots {
        if let SlotValues::Raw(values) = x {
            *values = Array2::zeros((0, values.ncols()));
        }
    }
    family
}
/// The members' families appended in order, raw slots uploaded and emptied.
fn batch_inputs<'a>(
    d: &Device,
    mut members: impl Iterator<Item = &'a FamilyInputs>,
) -> Result<(FamilyInputs, BTreeMap<usize, Tensor>), String> {
    let first = members.next().ok_or("empty episode batch")?.clone();
    let family = members.try_fold(first, |all, next| all.append(next).map_err(error))?;
    let mut raw = BTreeMap::new();
    for (slot, x) in family.slots.iter().enumerate() {
        if let SlotValues::Raw(values) = x {
            raw.insert(slot, d.upload(values.view()).map_err(error)?);
        }
    }
    Ok((hollow(&family), raw))
}
/// Concatenated row flags (an unmasked member scores every row), or none when no member masks.
fn batch_flags<'a>(
    d: &Device,
    masks: impl Iterator<Item = (Option<&'a Vec<bool>>, usize)> + Clone,
) -> Result<Option<Indices>, String> {
    if masks.clone().all(|(mask, _)| mask.is_none()) {
        return Ok(None);
    }
    let flags = masks
        .flat_map(|(mask, rows)| (0..rows).map(move |r| u32::from(mask.is_none_or(|m| m[r]))))
        .collect::<Vec<_>>();
    Ok(Some(d.upload_indices(&flags).map_err(error)?))
}
fn episode_flags(d: &Device, scored: Option<&Vec<bool>>) -> Result<Option<Indices>, String> {
    scored
        .map(|s| {
            d.upload_indices(&s.iter().map(|x| u32::from(*x)).collect::<Vec<_>>())
                .map_err(error)
        })
        .transpose()
}
/// Rows of every forward a fit runs: the largest episode, the largest batch, and all batches.
struct RowPlan {
    episode: usize,
    batch: usize,
    total: usize,
}
fn row_plan(members: &[Vec<usize>], rows: impl Fn(usize) -> usize) -> Result<RowPlan, String> {
    let mut plan = RowPlan {
        episode: 0,
        batch: 0,
        total: 0,
    };
    for group in members {
        let mut batch = 0usize;
        for index in group {
            plan.episode = plan.episode.max(rows(*index));
            batch = add(batch, rows(*index))?;
        }
        plan.batch = plan.batch.max(batch);
        plan.total = add(plan.total, batch)?;
    }
    Ok(plan)
}
/// Every group batch's forward values stay resident until the active group is chosen, then one
/// batch is reversed (values, retained cotangents); exact scans run one episode at a time.
fn trace_bytes(bytes_per_row: usize, rows: &RowPlan) -> Result<usize, String> {
    Ok(mul(bytes_per_row, add(rows.total, mul(rows.batch, 4)?)?)?
        .max(mul(mul(bytes_per_row, rows.episode)?, 5)?))
}
/// Dense attention matrices bounded by rows squared (safe for multiple sequences), with q/k
/// rotation, cotangent and temporary vectors, for the largest forward; rotation tables of the
/// largest batch and the largest episode stay cached.
fn attention_bytes(source: &OperatorProgram, rows: &RowPlan) -> Result<(usize, usize), String> {
    let mut peak = 0usize;
    let mut indices = 0usize;
    for node in &source.nodes {
        if let Node::Attend { query, rotary, .. } = node {
            let width = widths_for(source, *query)?;
            peak = peak.max(add(
                mul(mul(rows.batch, rows.batch)?, 8 * 12)?,
                mul(mul(rows.batch, width)?, 8 * 16)?,
            )?);
            if rotary.is_some() {
                indices = add(indices, mul(mul(add(rows.batch, rows.episode)?, width)?, 8 * 2)?)?;
            }
        }
    }
    Ok((peak, indices))
}
fn law_code_bytes(source: &OperatorProgram) -> Result<usize, String> {
    // Pointwise law-code arrays live with the compiled graph, independently of operators.
    source.nodes.iter().try_fold(0, |bytes, node| match node {
        Node::Pointwise { input, .. } => add(bytes, mul(widths_for(source, *input)?, 4)?),
        _ => Ok(bytes),
    })
}
fn prepare(
    d: &Device,
    source: &OperatorProgram,
    episodes: &[Episode],
    trainable: &[usize],
    limit: usize,
) -> Result<(DeviceProgram, Resident, usize), String> {
    let (expanded, classes) = validate(source, episodes)?;
    let parameters = mul(parameter_elements(source, trainable)?, 8)?;
    let mut p = DeviceProgram::compile_values_bounded(d, &expanded, limit)?;
    let members = group_members(episodes.iter().map(|e| e.group.as_str()));
    let rows = row_plan(&members, |i| episodes[i].inputs.rows)?;
    let mut panels = 0;
    let mut indices = law_code_bytes(&expanded)?;
    // An exact scan copies one episode's raw inputs and target rows out of its batch.
    let mut slices = 0usize;
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
        let mut slice = mul(e.target_logits.len(), 8)?;
        for x in &e.inputs.slots {
            match x {
                SlotValues::Raw(x) => slice = add(slice, mul(x.len(), 8)?)?,
                // Batch and episode forwards each upload their token ids.
                SlotValues::Tokens(x) => indices = add(indices, mul(x.len(), 8)?)?,
            }
        }
        panels = add(panels, slice)?;
        slices = slices.max(slice);
        if e.scored.is_some() {
            indices = add(indices, mul(e.inputs.rows, 8)?)?;
        }
    }
    let (attention_peak, rotations) = attention_bytes(&expanded, &rows)?;
    // Includes forward values of every batch, one reverse's values/retained cotangents, KL arrays,
    // parameter snapshots/candidates/swaps, accumulated/batch gradients, moments, updates and
    // promotion/column copies.
    let mut planned = add(p.operator_numeric_bytes()?, mul(parameters, 15)?)?;
    planned = add(planned, panels)?;
    planned = add(planned, slices)?;
    planned = add(planned, add(indices, rotations)?)?;
    planned = add(planned, trace_bytes(p.bytes_per_row(), &rows)?)?;
    planned = add(planned, attention_peak)?;
    planned = add(planned, mul(mul(rows.batch, classes)?, 8 * 4)?)?;
    planned = add(planned, mul(mul(rows.batch, max_factor_rank)?, 8 * 4)?)?;
    p.set_arithmetic(Arithmetic::F64);
    if planned > limit {
        return Err(format!(
            "causal fitter numeric plan {planned} exceeds {limit}"
        ));
    }
    p.prepare_dense_parameters(trainable)?;
    let mut place = vec![(0, 0); episodes.len()];
    let mut batches = Vec::with_capacity(members.len());
    for (b, group) in members.into_iter().enumerate() {
        let (family, raw) = batch_inputs(d, group.iter().map(|i| &episodes[*i].inputs))?;
        let views = group
            .iter()
            .map(|i| episodes[*i].target_logits.view())
            .collect::<Vec<_>>();
        let target = ndarray::concatenate(Axis(0), &views).map_err(error)?;
        let flags = batch_flags(
            d,
            group
                .iter()
                .map(|i| (episodes[*i].scored.as_ref(), episodes[*i].inputs.rows)),
        )?;
        let mut offset = 0;
        for i in &group {
            place[*i] = (b, offset);
            offset += episodes[*i].inputs.rows;
        }
        batches.push(Batch {
            family,
            raw,
            target: ResidentTarget::Logits(d.upload(target.view()).map_err(error)?),
            flags,
            members: group,
        });
    }
    let episodes = episodes
        .iter()
        .zip(place)
        .map(|(e, (batch, offset))| {
            Ok(ResidentEpisode {
                family: hollow(&e.inputs),
                batch,
                offset,
                responses: Vec::new(),
                flags: episode_flags(d, e.scored.as_ref())?,
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
    let resident = Resident {
        episodes,
        batches,
        executable: expanded,
    };
    Ok((p, resident, planned))
}
/// The members' compact labels as one target over their appended rows.
fn appended_target(d: &Device, parts: &[&Target]) -> Result<Target, String> {
    let first = parts.first().ok_or("empty compact batch")?;
    let rows = parts.iter().try_fold(0usize, |sum, t| add(sum, t.rows()))?;
    let mut mu = d.zeros(rows, first.width()).map_err(error)?;
    let mut offset = 0;
    for t in parts {
        d.set_rows(&mut mu, offset, &t.mu).map_err(error)?;
        offset += t.rows();
    }
    let scored = parts.iter().any(|t| t.scored.is_some()).then(|| {
        parts
            .iter()
            .flat_map(|t| t.scored.clone().unwrap_or_else(|| vec![true; t.rows()]))
            .collect()
    });
    Ok(Target {
        mu: Arc::new(mu),
        entropy: parts.iter().flat_map(|t| t.entropy.iter().copied()).collect(),
        head: first.head.clone(),
        scored,
    })
}
fn prepare_compact(
    d: &Device,
    source: &OperatorProgram,
    episodes: &[FixedHeadEpisode],
    trainable: &[usize],
    limit: usize,
    tile_rows: usize,
) -> Result<(DeviceProgram, Resident, usize), String> {
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
    let mut slices = 0usize;
    let mut indices = law_code_bytes(&prefix)?;
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
        // Episode labels and their batch's concatenated copy.
        panels = add(panels, mul(e.target.numeric_bytes(), 2)?)?;
        let mut slice = 0usize;
        for (decl, values) in expanded.declarations.slots.iter().zip(&e.inputs.slots) {
            match (decl, values) {
                (Slot::Raw { width }, SlotValues::Raw(x))
                    if x.dim() == (e.inputs.rows, *width) && x.iter().all(|v| v.is_finite()) =>
                {
                    panels = add(panels, mul(x.len(), 8)?)?;
                    slice = add(slice, mul(x.len(), 8)?)?;
                }
                (Slot::Token { domain }, SlotValues::Tokens(tokens))
                    if tokens.len() == e.inputs.rows
                        && tokens.iter().all(|t| {
                            (*t as usize) < expanded.declarations.domains[*domain].size
                        }) =>
                {
                    indices = add(indices, mul(tokens.len(), 8)?)?;
                }
                _ => return Err("compact episode slots mismatch".into()),
            }
        }
        slices = slices.max(slice);
        if e.target.scored.is_some() {
            indices = add(indices, mul(e.inputs.rows, 8)?)?;
        }
    }
    let members = group_members(episodes.iter().map(|e| e.group.as_str()));
    let rows = row_plan(&members, |i| episodes[i].inputs.rows)?;
    let (attention_peak, rotations) = attention_bytes(&prefix, &rows)?;
    let mut planned = add(
        p.operator_numeric_bytes()?,
        mul(mul(parameter_elements(source, trainable)?, 8)?, 15)?,
    )?;
    planned = add(planned, mul(head.embedding.len(), 8)?)?;
    planned = add(planned, panels)?;
    planned = add(planned, slices)?;
    planned = add(planned, add(indices, rotations)?)?;
    planned = add(planned, trace_bytes(p.bytes_per_row(), &rows)?)?;
    planned = add(planned, attention_peak)?;
    planned = add(
        planned,
        mul(mul(tile_rows.min(rows.batch), head.embedding.nrows())?, 8 * 2)?,
    )?;
    planned = add(planned, mul(mul(rows.batch, head.embedding.ncols())?, 8 * 6)?)?;
    if planned > limit {
        return Err(format!(
            "compact fitter numeric plan {planned} exceeds {limit}"
        ));
    }
    p.set_arithmetic(Arithmetic::F64);
    p.prepare_dense_parameters(trainable)?;
    let resident_head = Arc::new(ResidentHead::new(d, &head, tile_rows)?);
    let mut place = vec![(0, 0); episodes.len()];
    let mut batches = Vec::with_capacity(members.len());
    for (b, group) in members.into_iter().enumerate() {
        let (family, raw) = batch_inputs(d, group.iter().map(|i| &episodes[*i].inputs))?;
        let parts = group
            .iter()
            .map(|i| &episodes[*i].target)
            .collect::<Vec<_>>();
        let mut offset = 0;
        for i in &group {
            place[*i] = (b, offset);
            offset += episodes[*i].inputs.rows;
        }
        batches.push(Batch {
            family,
            raw,
            target: ResidentTarget::Fixed {
                target: appended_target(d, &parts)?,
                head: resident_head.clone(),
            },
            flags: None,
            members: group,
        });
    }
    let episodes = episodes
        .iter()
        .zip(place)
        .map(|(e, (batch, offset))| {
            let scored = e.target.scored.clone();
            ResidentEpisode {
                family: hollow(&e.inputs),
                batch,
                offset,
                responses: Vec::new(),
                flags: None,
                label: e.label.clone(),
                group: e.group.clone(),
                scored_rows: scored
                    .as_ref()
                    .map_or(e.inputs.rows, |s| s.iter().filter(|v| **v).count()),
                scored_mask: scored,
            }
        })
        .collect();
    let resident = Resident {
        episodes,
        batches,
        executable: prefix,
    };
    Ok((p, resident, planned))
}

// Validate the complete sidecar and its numeric bound before uploading any response panels.
fn attach_responses(
    p: &DeviceProgram,
    source: &OperatorProgram,
    resident: &mut Resident,
    responses: &NativeResponses,
    base: usize,
    limit: usize,
) -> Result<usize, String> {
    let (_, mapping) = mapped_inlined(source)?;
    if responses
        .keys()
        .any(|label| !resident.episodes.iter().any(|e| &e.label == label))
    {
        return Err("unknown native response episode label".into());
    }
    let mut panels = 0usize;
    let mut peak = 0usize;
    for e in &resident.episodes {
        let mut labels = BTreeSet::new();
        let mut seeds = BTreeMap::new();
        let mut workspace = 0usize;
        for t in responses.get(&e.label).into_iter().flatten() {
            let node = *mapping
                .get(t.source_node)
                .ok_or("native response source node out of range")?;
            let width = *p
                .widths()
                .get(node)
                .ok_or("native response node dropped from fixed-head prefix")?;
            let rows = e.family.rows;
            if t.label.is_empty()
                || !labels.insert(&t.label)
                || width == 0
                || t.values.dim() != (rows, width)
                || t.values.iter().any(|v| !v.is_finite())
                || !t.scale.is_finite()
                || t.scale <= 0.
                || !t.weight.is_finite()
                || t.weight <= 0.
                || !(t.weight / t.scale / t.scale).is_finite()
                || t.scored
                    .as_ref()
                    .is_some_and(|m| m.len() != rows || !m.iter().any(|v| *v))
            {
                return Err(
                    "invalid native response label/dimensions/values/scored rows/scale/weight"
                        .into(),
                );
            }
            let scored_rows = t
                .scored
                .as_ref()
                .map_or(rows, |m| m.iter().filter(|v| **v).count());
            let coefficient = 2. * (t.weight / t.scale / t.scale / scored_rows as f64);
            if !coefficient.is_finite() || coefficient <= 0. {
                return Err("native response seed coefficient is not positive finite f64".into());
            }
            let bytes = mul(t.values.len(), 8)?;
            // Fixed targets and explicit per-element masked coefficients; two u32 block offsets.
            panels = add(panels, add(mul(bytes, 2)?, 8)?)?;
            seeds.insert(node, bytes);
            // Residual, row reduction, masked seed, and temporary merged-node contribution.
            workspace = workspace.max(add(mul(bytes, 3)?, mul(rows, 8)?)?);
        }
        let seed_bytes = seeds.values().try_fold(0usize, |sum, &v| add(sum, v))?;
        peak = peak.max(add(seed_bytes, workspace)?);
    }
    // Seeds over a whole batch's rows: at most every episode's seed bytes at once.
    let planned = add(base, add(panels, mul(peak, 2)?)?)?;
    if planned > limit {
        return Err(format!(
            "joint causal fitter numeric plan {planned} exceeds {limit}"
        ));
    }
    for e in &mut resident.episodes {
        for t in responses.get(&e.label).into_iter().flatten() {
            let node = mapping[t.source_node];
            let scored_rows = t
                .scored
                .as_ref()
                .map_or(e.family.rows, |m| m.iter().filter(|v| **v).count());
            let coefficient = 2. * (t.weight / t.scale / t.scale / scored_rows as f64);
            let coefficients = Array2::from_shape_fn(t.values.dim(), |(row, _)| {
                if t.scored.as_ref().is_none_or(|m| m[row]) {
                    coefficient
                } else {
                    0.
                }
            });
            e.responses.push(ResidentResponse {
                label: t.label.clone(),
                source_node: t.source_node,
                node,
                target: p.device().upload(t.values.view()).map_err(error)?,
                coefficients: p.device().upload(coefficients.view()).map_err(error)?,
                blocks: p
                    .device()
                    .column_blocks(&[t.values.ncols()])
                    .map_err(error)?,
                scored: t.scored.clone(),
                scored_rows,
                scale: t.scale,
                weight: t.weight,
            });
        }
    }
    Ok(planned)
}
/// Response losses of episode `i`, whose rows start at `offset` in `trace`; with `seeds`, adds
/// `group_weight` times their seeds (over the episode's rows).
fn response_score(
    v: &View,
    i: usize,
    trace: &DeviceTrace,
    offset: usize,
    mut seeds: Option<&mut BTreeMap<usize, Tensor>>,
    group_weight: f64,
) -> Result<Vec<ResponseMeasurement>, String> {
    let d = v.p.device();
    let e = &v.r.episodes[i];
    let rows = e.family.rows;
    let whole = offset == 0 && trace.rows == rows;
    let mut measurements = Vec::new();
    for (k, t) in e.responses.iter().enumerate() {
        let (target, coefficients) = v.response(i, k);
        let value = trace.value(t.node)?;
        let mut residual = if whole {
            d.copy(value)
        } else {
            d.rows_of(value, offset, rows)
        }
        .map_err(error)?;
        d.axpy(&mut residual, -1., target).map_err(error)?;
        let squared = d
            .block_products(&residual, &residual, &t.blocks)
            .map_err(error)?;
        let squared = d.download(&squared).map_err(error)?;
        let mut sum = 0.;
        for row in 0..rows {
            if t.scored.as_ref().is_none_or(|m| m[row]) {
                let value = squared[[row, 0]] / t.scale / t.scale;
                if !value.is_finite() {
                    return Err("nonfinite native response loss".into());
                }
                sum += value;
            }
        }
        let mean = sum / t.scored_rows as f64;
        let weighted_loss = t.weight * mean;
        if !weighted_loss.is_finite() {
            return Err("nonfinite weighted native response loss".into());
        }
        measurements.push(ResponseMeasurement {
            label: t.label.clone(),
            source_node: t.source_node,
            scored_rows: t.scored_rows,
            scale: t.scale,
            weight: t.weight,
            mean_normalized_squared_error: mean,
            weighted_loss,
        });
        if let Some(seeds) = seeds.as_deref_mut() {
            let mut masked = d.zeros(residual.rows(), residual.cols()).map_err(error)?;
            d.hadamard(&mut masked, &residual, coefficients, false)
                .map_err(error)?;
            match seeds.get_mut(&t.node) {
                Some(existing) => d.axpy(existing, group_weight, &masked).map_err(error)?,
                None => {
                    let mut scaled = d.zeros(masked.rows(), masked.cols()).map_err(error)?;
                    d.axpy(&mut scaled, group_weight, &masked).map_err(error)?;
                    seeds.insert(t.node, scaled);
                }
            }
        }
    }
    Ok(measurements)
}

fn widths_for(p: &OperatorProgram, node: usize) -> Result<usize, String> {
    Ok(p.node_interface(node).map_err(error)?.width())
}
/// Episode `i`'s rows of its batch's target.
fn episode_target(v: &View, i: usize) -> Result<ResidentTarget, String> {
    let d = v.p.device();
    let e = &v.r.episodes[i];
    let rows = e.offset..e.offset + e.family.rows;
    Ok(match v.target(e.batch) {
        ResidentTarget::Logits(all) => {
            ResidentTarget::Logits(d.rows_of(all, e.offset, e.family.rows).map_err(error)?)
        }
        ResidentTarget::Fixed { target, head } => ResidentTarget::Fixed {
            target: Target {
                mu: Arc::new(
                    d.rows_of(&target.mu, e.offset, e.family.rows)
                        .map_err(error)?,
                ),
                entropy: target.entropy[rows.clone()].to_vec(),
                head: target.head.clone(),
                scored: target.scored.as_ref().map(|s| s[rows].to_vec()),
            },
            head: head.clone(),
        },
    })
}
/// One episode's forward on its own rows (copies of its batch rows).
fn forward(v: &View, i: usize) -> Result<DeviceTrace, String> {
    let d = v.p.device();
    let e = &v.r.episodes[i];
    let given = v
        .raw(e.batch)
        .iter()
        .map(|(slot, x)| Ok((*slot, d.rows_of(x, e.offset, e.family.rows).map_err(error)?)))
        .collect::<Result<_, String>>()?;
    trace::within("forward", d, || v.p.forward_given(&e.family, given))
}
fn episode_measurement(
    e: &ResidentEpisode,
    kl: &[f64],
    responses: Vec<ResponseMeasurement>,
) -> Result<EpisodeMeasurement, String> {
    if kl.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite training KL".into());
    }
    let mean = kl.iter().sum::<f64>() / e.scored_rows as f64;
    if !mean.is_finite() {
        return Err("nonfinite episode mean KL".into());
    }
    let total_loss = mean + responses.iter().map(|r| r.weighted_loss).sum::<f64>();
    if !total_loss.is_finite() {
        return Err("nonfinite joint episode loss".into());
    }
    let (worst_scored_row, maximum_scored_row_kl) = kl
        .iter()
        .enumerate()
        .filter(|(i, _)| e.scored_mask.as_ref().is_none_or(|mask| mask[*i]))
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, value)| (i, *value))
        .ok_or("empty scored domain")?;
    Ok(EpisodeMeasurement {
        label: e.label.clone(),
        group: e.group.clone(),
        scored_rows: e.scored_rows,
        mean_kl: mean,
        total_loss,
        responses,
        maximum_scored_row_kl,
        worst_scored_row,
    })
}
/// Group means of equal-weight episode losses (summed in episode order); the worst is active.
fn summarize(scores: Vec<EpisodeMeasurement>) -> Result<Measurement, String> {
    let mut groups: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for e in &scores {
        let entry = groups.entry(e.group.clone()).or_insert((0., 0));
        entry.0 += e.total_loss;
        entry.1 += 1;
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
/// Every episode on its own rows in the program's current arithmetic.
fn scan(p: &DeviceProgram, r: &Resident) -> Result<Measurement, String> {
    let v = View::exact(p, r);
    let mut scores = Vec::new();
    for (i, e) in r.episodes.iter().enumerate() {
        let trace = forward(&v, i)?;
        let target = episode_target(&v, i)?;
        let (kl, _) = trace::within("kl", p.device(), || {
            score(p, &target, e.flags.as_ref(), &trace, false)
        })?;
        let responses = response_score(&v, i, &trace, 0, None, 0.)?;
        scores.push(episode_measurement(e, &kl, responses)?);
    }
    summarize(scores)
}
/// [`scan`] in F64, whatever the program's proposal arithmetic.
fn exact_scan(p: &mut DeviceProgram, r: &Resident) -> Result<Measurement, String> {
    let proposal = p.arithmetic();
    p.set_arithmetic(Arithmetic::F64);
    let measured = trace::within("fit.exact", p.device(), || scan(p, r));
    p.set_arithmetic(proposal);
    measured
}
/// The proposal's program and its copies of the batches and response targets in the proposal
/// storage (f32 on CUDA), when that differs from the exact program's.
struct Narrow {
    program: DeviceProgram,
    batches: Vec<(BTreeMap<usize, Tensor>, ResidentTarget)>,
    /// Per episode, per response: its target and seed coefficients.
    responses: Vec<Vec<(Tensor, Tensor)>>,
}
/// A program with the inputs and targets held in its storage.
#[derive(Clone, Copy)]
struct View<'a> {
    p: &'a DeviceProgram,
    r: &'a Resident,
    narrow: Option<&'a Narrow>,
}
impl<'a> View<'a> {
    fn exact(p: &'a DeviceProgram, r: &'a Resident) -> Self {
        Self { p, r, narrow: None }
    }
    fn proposal(p: &'a DeviceProgram, r: &'a Resident, narrow: Option<&'a Narrow>) -> Self {
        Self {
            p: narrow.map_or(p, |n| &n.program),
            r,
            narrow,
        }
    }
    fn raw(&self, b: usize) -> &'a BTreeMap<usize, Tensor> {
        self.narrow
            .map_or(&self.r.batches[b].raw, |n| &n.batches[b].0)
    }
    fn target(&self, b: usize) -> &'a ResidentTarget {
        self.narrow
            .map_or(&self.r.batches[b].target, |n| &n.batches[b].1)
    }
    fn response(&self, i: usize, k: usize) -> (&'a Tensor, &'a Tensor) {
        match self.narrow {
            Some(n) => (&n.responses[i][k].0, &n.responses[i][k].1),
            None => {
                let t = &self.r.episodes[i].responses[k];
                (&t.target, &t.coefficients)
            }
        }
    }
}
/// The fixed head through which a full-logit fit's proposals can use compact statistics: an
/// untrainable bias-free dense head whose logits no response target reads.
fn proposal_head(r: &Resident, trainable: &[usize]) -> Option<Head> {
    if !r
        .batches
        .iter()
        .all(|b| matches!(b.target, ResidentTarget::Logits(_)))
    {
        return None;
    }
    let head = Head::of(&r.executable).ok()?;
    let read = r
        .episodes
        .iter()
        .flat_map(|e| &e.responses)
        .all(|t| t.node <= head.hidden);
    (read && !trainable.contains(&head.operator)).then_some(head)
}
/// Compact labels of teacher logits in F64, once per fit: per row `mu = p E` and
/// `c = sum p log p` (unscored rows zero), `mu` narrowed into `n`'s storage.
fn compact_labels(
    d: &Device,
    n: &Device,
    logits: &Tensor,
    flags: Option<&Indices>,
    scored: Option<Vec<bool>>,
    (head, embedding): (&Arc<Head>, &Tensor),
) -> Result<Target, String> {
    let mut probabilities = d.copy(logits).map_err(error)?;
    let stats = d
        .softmax_stats_rows(&mut probabilities, flags)
        .map_err(error)?;
    let mut mu = d.zeros(logits.rows(), embedding.cols()).map_err(error)?;
    d.gemm(
        &mut mu,
        1.,
        &probabilities,
        Op::N,
        embedding,
        Op::N,
        0.,
        Arithmetic::F64,
    )
    .map_err(error)?;
    Ok(Target {
        mu: Arc::new(n.convert(&mu).map_err(error)?),
        entropy: stats.iter().map(|s| s[1]).collect(),
        head: head.clone(),
        scored,
    })
}
/// F32-storage copies of the program, batches and response targets for F32/TF32 proposals, or
/// none when the device holds only F64 (the host then rounds just the products' operands).
/// A full-logit fit through a fixed head proposes on the head's input with compact labels, so
/// its proposals never form logits. The numeric plan is checked before anything is converted.
fn narrow_copies(
    d: &Device,
    p: &DeviceProgram,
    r: &Resident,
    trainable: &[usize],
    arithmetic: FitArithmetic,
    planned: usize,
    limit: usize,
) -> Result<(Option<Narrow>, usize), String> {
    if arithmetic == FitArithmetic::F64 {
        return Ok((None, planned));
    }
    let Ok(n) = d.with_storage(Storage::F32) else {
        return Ok((None, planned));
    };
    let compact = proposal_head(r, trainable);
    // Halved operator, input, target and response buffers; for compact proposals one f64
    // batch of probabilities, its labels and the f64 head while labels are made.
    let mut bytes = p.operator_numeric_bytes()? / 2;
    let mut transient = 0usize;
    for b in &r.batches {
        for x in b.raw.values() {
            bytes = add(bytes, mul(x.len(), 4)?)?;
        }
        bytes = add(
            bytes,
            match (&b.target, &compact) {
                (ResidentTarget::Logits(t), Some(head)) => {
                    let mu = mul(t.rows(), head.embedding.ncols())?;
                    transient = transient.max(add(mul(t.len(), 8)?, mul(mu, 8)?)?);
                    mul(mu, 4)?
                }
                (ResidentTarget::Logits(t), None) => mul(t.len(), 4)?,
                (ResidentTarget::Fixed { target, .. }, _) => mul(target.mu.len(), 4)?,
            },
        )?;
    }
    if let Some(head) = &compact {
        bytes = add(bytes, mul(head.embedding.len(), 4)?)?;
        transient = add(transient, mul(head.embedding.len(), 8)?)?;
    } else if let Some(ResidentTarget::Fixed { target, .. }) = r.batches.first().map(|b| &b.target) {
        bytes = add(bytes, mul(target.head.embedding.len(), 4)?)?;
    }
    for t in r.episodes.iter().flat_map(|e| &e.responses) {
        bytes = add(bytes, mul(t.target.len(), 8)?)?;
    }
    let planned = add(planned, add(bytes, transient)?)?;
    if planned > limit {
        return Err(format!(
            "f32 proposal numeric plan {planned} exceeds {limit}"
        ));
    }
    let executable = compact
        .as_ref()
        .map_or_else(|| r.executable.clone(), |head| head.prefix(&r.executable));
    let mut program = DeviceProgram::compile_values_bounded(&n, &executable, limit)?;
    program.prepare_dense_parameters(trainable)?;
    let rows = r.batches.iter().map(|b| b.family.rows).max().unwrap_or(1);
    let compact = compact
        .map(|head| {
            let embedding = d.upload(head.embedding.view()).map_err(error)?;
            let resident = Arc::new(ResidentHead::new(&n, &head, rows)?);
            Ok::<_, String>((Arc::new(head), embedding, resident))
        })
        .transpose()?;
    let mut shared: Option<Arc<ResidentHead>> = None;
    let mut batches = Vec::with_capacity(r.batches.len());
    for b in &r.batches {
        let raw = b
            .raw
            .iter()
            .map(|(slot, x)| Ok((*slot, n.convert(x).map_err(error)?)))
            .collect::<Result<_, String>>()?;
        let target = match (&b.target, &compact) {
            (ResidentTarget::Logits(t), Some((head, embedding, resident))) => {
                let masks = b
                    .members
                    .iter()
                    .map(|i| (r.episodes[*i].scored_mask.as_ref(), r.episodes[*i].family.rows));
                let scored = masks.clone().any(|(m, _)| m.is_some()).then(|| {
                    masks
                        .flat_map(|(m, rows)| m.cloned().unwrap_or_else(|| vec![true; rows]))
                        .collect()
                });
                ResidentTarget::Fixed {
                    target: compact_labels(d, &n, t, b.flags.as_ref(), scored, (head, embedding))?,
                    head: resident.clone(),
                }
            }
            (ResidentTarget::Logits(t), None) => ResidentTarget::Logits(n.convert(t).map_err(error)?),
            (ResidentTarget::Fixed { target, head: exact }, _) => {
                let head = match &shared {
                    Some(head) => head.clone(),
                    None => Arc::new(ResidentHead::new(&n, &target.head, exact.tile_rows)?),
                };
                shared = Some(head.clone());
                ResidentTarget::Fixed {
                    target: Target {
                        mu: Arc::new(n.convert(&target.mu).map_err(error)?),
                        ..target.clone()
                    },
                    head,
                }
            }
        };
        batches.push((raw, target));
    }
    let responses = r
        .episodes
        .iter()
        .map(|e| {
            e.responses
                .iter()
                .map(|t| {
                    Ok((
                        n.convert(&t.target).map_err(error)?,
                        n.convert(&t.coefficients).map_err(error)?,
                    ))
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .collect::<Result<_, String>>()?;
    Ok((
        Some(Narrow {
            program,
            batches,
            responses,
        }),
        planned,
    ))
}
/// The rows of one proposal forward: a group batch, or one episode of it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Unit {
    Batch(usize),
    Episode(usize),
}
/// A unit's forward values at the current parameters.
struct Pass {
    unit: Unit,
    trace: DeviceTrace,
}
/// The unit's episodes and the row where each starts.
fn members(r: &Resident, unit: Unit) -> Vec<(usize, usize)> {
    match unit {
        Unit::Batch(b) => r.batches[b]
            .members
            .iter()
            .map(|i| (*i, r.episodes[*i].offset))
            .collect(),
        Unit::Episode(i) => vec![(i, 0)],
    }
}
fn unit_forward(v: &View, unit: Unit) -> Result<DeviceTrace, String> {
    match unit {
        Unit::Batch(b) => {
            let d = v.p.device();
            // forward_given owns its arguments; these copies preserve the resident batch.
            let given = v
                .raw(b)
                .iter()
                .map(|(slot, x)| Ok((*slot, d.copy(x).map_err(error)?)))
                .collect::<Result<_, String>>()?;
            trace::within("fit.proposal.forward", d, || {
                v.p.forward_given(&v.r.batches[b].family, given)
            })
        }
        Unit::Episode(i) => forward(v, i),
    }
}
/// Per-row output KL of the unit; with `seed`, also its unscaled hidden seed, made in place of
/// the trace's logits when no executed node follows them.
fn unit_kl(
    v: &View,
    unit: Unit,
    trace: &mut DeviceTrace,
    seed: bool,
) -> Result<(Vec<f64>, Option<Tensor>), String> {
    let p = v.p;
    let d = p.device();
    let owned;
    let (target, flags) = match unit {
        Unit::Batch(b) => (v.target(b), v.r.batches[b].flags.as_ref()),
        Unit::Episode(i) => {
            owned = episode_target(v, i)?;
            (&owned, v.r.episodes[i].flags.as_ref())
        }
    };
    match target {
        ResidentTarget::Logits(target) if seed => {
            let mut logits = if p.hidden() + 1 == trace.len() {
                trace.take(p.hidden())?
            } else {
                d.copy(trace.value(p.hidden())?).map_err(error)?
            };
            let kl = d.kl_rows(target, &mut logits, flags).map_err(error)?;
            Ok((kl, Some(logits)))
        }
        // Score-only KL leaves its logits unchanged.
        ResidentTarget::Logits(target) => Ok((
            d.kl_score_rows(target, trace.value_mut(p.hidden())?, flags)
                .map_err(error)?,
            None,
        )),
        ResidentTarget::Fixed { target, head } => {
            head.score(d, trace.value(p.hidden())?, target, seed, p.arithmetic())
        }
    }
}
fn add_seed(
    d: &Device,
    seeds: &mut BTreeMap<usize, Tensor>,
    node: usize,
    term: Tensor,
) -> Result<(), String> {
    match seeds.entry(node) {
        Entry::Occupied(mut existing) => d.axpy(existing.get_mut(), 1., &term).map_err(error),
        Entry::Vacant(slot) => {
            slot.insert(term);
            Ok(())
        }
    }
}
/// Every group batch in the proposal arithmetic: the measurement and each batch's pass.
fn proposal_scan(v: &View) -> Result<(Measurement, Vec<Pass>), String> {
    let r = v.r;
    let d = v.p.device();
    let mut scores = (0..r.episodes.len()).map(|_| None).collect::<Vec<_>>();
    let mut passes = Vec::with_capacity(r.batches.len());
    for b in 0..r.batches.len() {
        let unit = Unit::Batch(b);
        let mut trace = unit_forward(v, unit)?;
        let (kl, _) = trace::within("fit.proposal.kl", d, || {
            unit_kl(v, unit, &mut trace, false)
        })?;
        for (i, offset) in members(r, unit) {
            let e = &r.episodes[i];
            let responses = response_score(v, i, &trace, offset, None, 0.)?;
            scores[i] = Some(episode_measurement(
                e,
                &kl[offset..offset + e.family.rows],
                responses,
            )?);
        }
        passes.push(Pass { unit, trace });
    }
    let scores = scores
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or("unmeasured training episode")?;
    Ok((summarize(scores)?, passes))
}
/// Adds the pass's gradient of the `weights`-weighted episode losses to `gradients`; returns
/// the episodes reversed. The reverse is linear in its seeds, so it runs on seeds relative to
/// the largest per-row KL weight and scales the parameter gradient once.
fn unit_gradient(
    v: &View,
    pass: Pass,
    weights: &[f64],
    trainable: &[usize],
    gradients: &mut BTreeMap<usize, Tensor>,
) -> Result<usize, String> {
    let (p, r) = (v.p, v.r);
    let d = p.device();
    let Pass { unit, mut trace } = pass;
    let spans = members(r, unit);
    let factors = spans
        .iter()
        .map(|(i, _)| weights[*i] / r.episodes[*i].scored_rows as f64)
        .collect::<Vec<_>>();
    if factors.iter().any(|f| !f.is_finite() || *f < 0.) {
        return Err("invalid training episode weight".into());
    }
    let reference = factors.iter().copied().fold(0., f64::max);
    if reference == 0. {
        return Ok(0);
    }
    // Response seeds read node values, so they precede the KL seed that may consume logits.
    let mut seeds = BTreeMap::new();
    for ((i, offset), factor) in spans.iter().zip(&factors) {
        if *factor == 0. || r.episodes[*i].responses.is_empty() {
            continue;
        }
        let mut own = BTreeMap::new();
        response_score(v, *i, &trace, *offset, Some(&mut own), weights[*i] / reference)?;
        for (node, term) in own {
            if trace.rows == term.rows() {
                add_seed(d, &mut seeds, node, term)?;
                continue;
            }
            let all = match seeds.entry(node) {
                Entry::Occupied(all) => all.into_mut(),
                Entry::Vacant(slot) => {
                    slot.insert(d.zeros(trace.rows, term.cols()).map_err(error)?)
                }
            };
            // Members' rows are disjoint and each member's terms are already summed.
            d.set_rows(all, *offset, &term).map_err(error)?;
        }
    }
    let (kl, seed) = trace::within("fit.proposal.kl", d, || {
        unit_kl(v, unit, &mut trace, true)
    })?;
    if kl.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite gradient KL".into());
    }
    let mut seed = seed.ok_or("missing KL gradient seed")?;
    for ((i, offset), factor) in spans.iter().zip(&factors) {
        if *factor == reference {
            continue;
        }
        let rows = r.episodes[*i].family.rows;
        let own = d.rows_of(&seed, *offset, rows).map_err(error)?;
        let mut scaled = d.zeros(rows, seed.cols()).map_err(error)?;
        d.axpy(&mut scaled, factor / reference, &own)
            .map_err(error)?;
        d.set_rows(&mut seed, *offset, &scaled).map_err(error)?;
    }
    add_seed(d, &mut seeds, p.hidden(), seed)?;
    let (_, own) = trace::within("fit.proposal.reverse", d, || {
        p.vjp_values_dense(&trace, seeds, &[], trainable, p.arithmetic())
    })?;
    for index in trainable {
        d.axpy(
            gradients.get_mut(index).ok_or("gradient buffer")?,
            reference,
            own.get(index).ok_or("batch gradient")?,
        )
        .map_err(error)?;
    }
    Ok(factors.iter().filter(|f| **f > 0.).count())
}
/// The proposal gradient of the `weights`-weighted episode losses (one weight per episode).
/// It reverses `passes` (forward values at the current parameters) and forwards any other
/// group holding a weighted episode: whole when every member is, else episode by episode.
/// Returns the gradients and the episodes forwarded and reversed.
fn proposal_gradient(
    v: &View,
    weights: &[f64],
    passes: Vec<Pass>,
    trainable: &[usize],
) -> Result<(BTreeMap<usize, Tensor>, usize, usize), String> {
    let (p, r) = (v.p, v.r);
    let d = p.device();
    if weights.len() != r.episodes.len() || weights.iter().any(|w| !w.is_finite() || *w < 0.) {
        return Err("invalid training episode weights".into());
    }
    let mut gradients = trainable
        .iter()
        .map(|index| {
            let a = p.dense_parameter(*index)?;
            Ok((*index, d.zeros(a.rows(), a.cols()).map_err(error)?))
        })
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let (mut forwarded, mut reversed) = (0, 0);
    let mut covered = vec![false; r.batches.len()];
    for pass in passes {
        if let Unit::Batch(b) = pass.unit {
            covered[b] = true;
        }
        reversed += unit_gradient(v, pass, weights, trainable, &mut gradients)?;
    }
    for (b, batch) in r.batches.iter().enumerate() {
        let selected = batch
            .members
            .iter()
            .copied()
            .filter(|i| weights[*i] > 0.)
            .collect::<Vec<_>>();
        if covered[b] || selected.is_empty() {
            continue;
        }
        let units = if selected.len() == batch.members.len() {
            vec![Unit::Batch(b)]
        } else {
            selected.into_iter().map(Unit::Episode).collect()
        };
        for unit in units {
            let trace = unit_forward(v, unit)?;
            forwarded += members(r, unit).len();
            reversed += unit_gradient(v, Pass { unit, trace }, weights, trainable, &mut gradients)?;
        }
    }
    Ok((gradients, forwarded, reversed))
}
fn scheduled_batch(
    groups: &BTreeMap<String, Vec<usize>>,
    measurement: &Measurement,
    schedule: &BatchSchedule,
    cursors: &mut BTreeMap<String, usize>,
) -> Result<Vec<(usize, f64)>, String> {
    let mut weights = groups
        .keys()
        .map(|group| {
            let value = *measurement
                .groups
                .get(group)
                .ok_or("missing measured group")?;
            Ok((
                group,
                ((value - measurement.objective) / schedule.temperature).exp(),
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let total = weights.iter().map(|(_, weight)| weight).sum::<f64>();
    if !total.is_finite() || total <= 0. {
        return Err("nonfinite proposal group weights".into());
    }
    let mut selected = Vec::new();
    for (group, weight) in &mut weights {
        let indices = &groups[*group];
        let count = schedule.episodes_per_group.min(indices.len());
        if count == 0 {
            return Err("empty proposal group".into());
        }
        let cursor = cursors.entry((*group).clone()).or_default();
        for _ in 0..count {
            selected.push((indices[*cursor], *weight / total / count as f64));
            *cursor = if *cursor + 1 == indices.len() {
                0
            } else {
                *cursor + 1
            };
        }
    }
    Ok(selected)
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
    if settings.schedule.as_ref().is_some_and(|s| {
        s.episodes_per_group == 0
            || s.scan_every == 0
            || !s.temperature.is_finite()
            || s.temperature <= 0.
    }) {
        return Err("invalid causal batch schedule".into());
    }
    if trainable.is_empty()
        || settings.numeric_bytes == 0
        || settings.exact_scan_every == 0
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
    let (p, resident, planned) = trace::within("fit.prepare", d, || {
        prepare(d, source, episodes, trainable, settings.numeric_bytes)
    })?;
    fit_prepared(
        d, source, p, resident, planned, trainable, settings, started,
    )
}
/// Same causal fitter with fixed observations at multiple native source nodes.
/// Each episode loss is mean output KL plus weighted mean normalized squared response errors;
/// the optimized objective remains the maximum named-group mean of equal-weight episodes.
pub fn fit_with_native(
    d: &Device,
    source: &OperatorProgram,
    episodes: &[Episode],
    responses: &NativeResponses,
    trainable: &[usize],
    settings: Settings,
) -> Result<Fit, String> {
    let started = Instant::now();
    validate_settings(trainable, &settings)?;
    let (p, mut resident, base) = prepare(d, source, episodes, trainable, settings.numeric_bytes)?;
    let planned = attach_responses(
        &p,
        source,
        &mut resident,
        responses,
        base,
        settings.numeric_bytes,
    )?;
    fit_prepared(
        d, source, p, resident, planned, trainable, settings, started,
    )
}
pub fn measure_with_native(
    d: &Device,
    source: &OperatorProgram,
    episodes: &[Episode],
    responses: &NativeResponses,
    numeric_bytes: usize,
) -> Result<Measurement, String> {
    let (p, mut resident, base) = prepare(d, source, episodes, &[], numeric_bytes)?;
    attach_responses(&p, source, &mut resident, responses, base, numeric_bytes)?;
    scan(&p, &resident)
}
pub fn fit_fixed_head_with_native(
    d: &Device,
    source: &OperatorProgram,
    episodes: &[FixedHeadEpisode],
    responses: &NativeResponses,
    trainable: &[usize],
    settings: Settings,
    head_tile_rows: usize,
) -> Result<Fit, String> {
    let started = Instant::now();
    validate_settings(trainable, &settings)?;
    let (p, mut resident, base) = prepare_compact(
        d,
        source,
        episodes,
        trainable,
        settings.numeric_bytes,
        head_tile_rows,
    )?;
    let planned = attach_responses(
        &p,
        source,
        &mut resident,
        responses,
        base,
        settings.numeric_bytes,
    )?;
    fit_prepared(
        d, source, p, resident, planned, trainable, settings, started,
    )
}
pub fn measure_fixed_head_with_native(
    d: &Device,
    source: &OperatorProgram,
    episodes: &[FixedHeadEpisode],
    responses: &NativeResponses,
    numeric_bytes: usize,
    head_tile_rows: usize,
) -> Result<Measurement, String> {
    let (p, mut resident, base) =
        prepare_compact(d, source, episodes, &[], numeric_bytes, head_tile_rows)?;
    attach_responses(&p, source, &mut resident, responses, base, numeric_bytes)?;
    scan(&p, &resident)
}

/// Device copies of the trainable parameters.
fn parameters(p: &DeviceProgram, trainable: &[usize]) -> Result<BTreeMap<usize, Tensor>, String> {
    trainable
        .iter()
        .map(|index| {
            Ok((
                *index,
                p.device()
                    .copy(p.dense_parameter(*index)?)
                    .map_err(error)?,
            ))
        })
        .collect()
}
fn install(p: &mut DeviceProgram, values: &BTreeMap<usize, Tensor>) -> Result<(), String> {
    for (index, value) in values {
        let value = p.device().copy(value).map_err(error)?;
        p.replace_dense_parameter(*index, value)?;
    }
    Ok(())
}
/// Proposal parameters awaiting an exact scan.
struct Candidate {
    step: usize,
    objective: f64,
    parameters: BTreeMap<usize, Tensor>,
}

fn fit_prepared(
    d: &Device,
    source: &OperatorProgram,
    mut p: DeviceProgram,
    resident: Resident,
    planned: usize,
    trainable: &[usize],
    settings: Settings,
    started: Instant,
) -> Result<Fit, String> {
    let (mut narrow, planned) = narrow_copies(
        d,
        &p,
        &resident,
        trainable,
        settings.arithmetic,
        planned,
        settings.numeric_bytes,
    )?;
    let proposal_device = match &mut narrow {
        Some(n) => {
            n.program.set_arithmetic(settings.arithmetic.device());
            n.program.device().name()
        }
        None => p.device().name(),
    };
    // F64 master parameters and moments live with the exact program.
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
    let clock = Instant::now();
    let initial = exact_scan(&mut p, &resident)?;
    let mut exact_seconds = clock.elapsed().as_secs_f64();
    let mut proposal_seconds = 0.;
    let mut exact_scans = 1;
    let mut best = initial.clone();
    let mut best_step = 0;
    let mut snapshot = parameters(&p, trainable)?;
    let mut history = Vec::new();
    let mut initial_proposal = None;
    let mut reverse = 0;
    let mut forwards = resident.episodes.len();
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, episode) in resident.episodes.iter().enumerate() {
        groups.entry(episode.group.clone()).or_default().push(index);
    }
    let mut cursors = BTreeMap::new();
    // The best proposal measured since the last exact scan, and the latest complete one.
    let mut window: Option<Candidate> = None;
    let mut latest: Option<Measurement> = None;
    let steps = settings.iterations;
    if narrow.is_none() {
        p.set_arithmetic(settings.arithmetic.device());
    }
    // Step t measures the parameters after t updates, then makes update t + 1.
    for t in 0..=steps {
        let clock = Instant::now();
        let complete = t == steps
            || settings
                .schedule
                .as_ref()
                .is_none_or(|s| t % s.scan_every == 0);
        let mut passes = Vec::new();
        if complete {
            let (measurement, scanned) =
                proposal_scan(&View::proposal(&p, &resident, narrow.as_ref()))?;
            forwards += resident.episodes.len();
            if t == 0 {
                initial_proposal = Some(measurement.objective);
            } else if window
                .as_ref()
                .is_none_or(|c| measurement.objective < c.objective)
            {
                window = Some(Candidate {
                    step: t,
                    objective: measurement.objective,
                    parameters: parameters(&p, trainable)?,
                });
            }
            latest = Some(measurement);
            passes = scanned;
        }
        if t < steps {
            let measured = latest.as_ref().ok_or("missing proposal measurement")?;
            let mut weights = vec![0.; resident.episodes.len()];
            if let Some(schedule) = &settings.schedule {
                for (index, weight) in scheduled_batch(&groups, measured, schedule, &mut cursors)? {
                    weights[index] += weight;
                }
            } else {
                let active = groups
                    .get(&measured.active_group)
                    .ok_or("unknown active group")?;
                for index in active {
                    weights[*index] = 1. / active.len() as f64;
                }
            }
            let (gradients, forwarded, reversed) = trace::within("fit.gradient", d, || {
                proposal_gradient(
                    &View::proposal(&p, &resident, narrow.as_ref()),
                    &weights,
                    passes,
                    trainable,
                )
            })?;
            forwards += forwarded;
            reverse += reversed;
            trace::within("fit.adam", d, || {
                for index in trainable {
                    let gradient = gradients.get(index).ok_or("parameter gradient")?;
                    let widened;
                    let gradient = if gradient.storage() == d.storage() {
                        gradient
                    } else {
                        widened = d.convert(gradient).map_err(error)?;
                        &widened
                    };
                    let mut next = d.copy(p.dense_parameter(*index)?).map_err(error)?;
                    let (m, v) = moments.get_mut(index).ok_or("moment pair")?;
                    d.adam(
                        &mut next,
                        (m, v),
                        gradient,
                        settings.learning_rate,
                        (settings.beta1, settings.beta2, settings.epsilon),
                        t as u64 + 1,
                    )
                    .map_err(error)?;
                    p.replace_dense_parameter(*index, next)?;
                    if let Some(n) = &mut narrow {
                        let value = n
                            .program
                            .device()
                            .convert(p.dense_parameter(*index)?)
                            .map_err(error)?;
                        n.program.replace_dense_parameter(*index, value)?;
                    }
                }
                Ok::<_, String>(())
            })?;
        }
        proposal_seconds += clock.elapsed().as_secs_f64();
        if t == 0 || (t % settings.exact_scan_every != 0 && t != steps) {
            continue;
        }
        let Some(candidate) = window.take() else {
            continue;
        };
        let clock = Instant::now();
        let current = parameters(&p, trainable)?;
        install(&mut p, &candidate.parameters)?;
        let measurement = exact_scan(&mut p, &resident)?;
        install(&mut p, &current)?;
        exact_scans += 1;
        forwards += resident.episodes.len();
        if measurement.objective < best.objective {
            best = measurement.clone();
            best_step = candidate.step;
            snapshot = candidate.parameters;
        }
        history.push(Iteration {
            step: candidate.step,
            measurement,
            proposal_objective: Some(candidate.objective),
        });
        exact_seconds += clock.elapsed().as_secs_f64();
    }
    history.insert(
        0,
        Iteration {
            step: 0,
            measurement: initial.clone(),
            proposal_objective: initial_proposal,
        },
    );
    // The best exact scan measured exactly these parameters.
    install(&mut p, &snapshot)?;
    let final_measurement = best.clone();
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
        .batches
        .iter()
        .any(|b| matches!(b.target, ResidentTarget::Fixed { .. }));
    let scheduled = settings.schedule.is_some();
    let joint = resident.episodes.iter().any(|e| !e.responses.is_empty());
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
            exact_scans,
            proposal_device,
            proposal_seconds,
            exact_seconds,
            seconds: started.elapsed().as_secs_f64(),
            scope: if scheduled {
                "Proposal fitting only, in the declared proposal arithmetic: deterministic complete-sequence batches cycle within every named group. Softmax group weights use the latest complete proposal group losses and remain fixed until the next complete proposal measurement; intermediate gradients are a proposal heuristic, not exact gradients of the current worst-group loss. Output KL and weighted fixed-scale native-response losses contribute through the same ordinary VJP. Every exact_scan_every steps and at the end, an F64 scan of each episode on its own rows rescores the best complete proposal since the previous scan; only rescored parameters can become the checkpoint, and every reported measurement is such an F64 scan, never a minibatch estimate or heldout data. Forward/reverse counts include every executed episode, proposal or exact. Numeric evidence is operational; ordinary artifact acceptance is unchanged."
            } else if joint {
                "Proposal fitting only: maximum named-group mean of equal-weight joint episode losses (mean output KL plus explicitly weighted fixed-scale native-response mean squared errors). Output KL and response terms reported separately. Fixed targets never replace autonomous candidate values. Each step runs every group's appended episodes in one forward in the declared proposal arithmetic, picks the worst group from them, and reverses only that group's batch with one accumulated multi-node ordinary VJP, shared parameter owner and Adam. Every exact_scan_every steps and at the end, an F64 scan of each episode on its own rows rescores the best proposal since the previous scan; checkpoints and every reported measurement come from those scans. Numeric plan adds resident response targets/masked coefficients/block offsets and accumulated seeds/sequential residual/reduction scratch to the ordinary or fixed-head baseline. Host metadata, CUDA/context/library/allocator scratch excluded; ordinary artifact acceptance remains separate."
            } else if compact {
                "Proposal fitting only: same maximum named-group mean objective and Adam loop. Immutable fixed bias-free full-vocabulary head targets retain E^T p and sum p log p; each row-tiled candidate KL is logZ(Eh)-mu.h+c and hidden seed E^T q-mu. Head input is after original final normalization, with full-prefix ordinary VJP; both target and candidate unscored seeds are zero. Each step runs every group's appended episodes in one forward in the declared proposal arithmetic and reverses only the worst group's batch; every exact_scan_every steps and at the end, an F64 scan of each episode on its own rows rescores the best proposal since the previous scan, and checkpoints and every reported measurement come from those scans. F64 vendor exp/log/GEMM are operational, not certified real-arithmetic intervals; final ordinary artifact acceptance is unchanged. Numeric plan counts resident compact labels and their batch copies, head, inputs, every batch's trace, one reverse, parameter/candidate buffers and conservative attention/tiled vocabulary scratch. Host metadata/source, context/library/allocator scratch excluded."
            } else {
                "Proposal fitting only: maximum named-group mean of equal-weight episode means over declared scored rows. KL q-p gradients, full-sequence ordinary reverse including controls supplied as Raw graph inputs. One parameter owner across every episode. Each step runs every group's appended episodes in one forward in the declared proposal arithmetic, picks the worst group from them, and reverses only that group's batch. Every exact_scan_every steps and at the end, an F64 scan of each episode on its own rows rescores the best proposal since the previous scan; the best TRAIN checkpoint and every reported measurement come from those scans, with no validation input/selection. Vendor exp/log and neural arithmetic are not certified intervals; ordinary serialized acceptance is separate. Numeric plan includes fixed operators (all table/product roles), resident batched inputs/targets/flags, every batch's trace, one reverse's cotangents, KL scratch, an exact scan's episode slices, conservative dense attention scratch and parameter/moment/snapshot/candidate/update buffers. Excludes host panels/source bytes, CUDA context/library/allocator/register/spill scratch; token/rotation preparation peak conservatively counted, not a measured memory claim."
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
            schedule: None,
            arithmetic: FitArithmetic::F64,
            exact_scan_every: 1,
        }
    }
    /// The group's proposal gradient at equal episode weights, every batch forwarded afresh.
    fn gradient(
        p: &DeviceProgram,
        r: &Resident,
        group: &str,
        trainable: &[usize],
    ) -> Result<BTreeMap<usize, Tensor>, String> {
        let count = r.episodes.iter().filter(|e| e.group == group).count();
        if count == 0 {
            return Err("unknown active group".into());
        }
        let weights = r
            .episodes
            .iter()
            .map(|e| if e.group == group { 1. / count as f64 } else { 0. })
            .collect::<Vec<_>>();
        Ok(proposal_gradient(&View::exact(p, r), &weights, Vec::new(), trainable)?.0)
    }
    fn weighted_gradient(
        p: &DeviceProgram,
        r: &Resident,
        selected: &[(usize, f64)],
        trainable: &[usize],
    ) -> Result<BTreeMap<usize, Tensor>, String> {
        let mut weights = vec![0.; r.episodes.len()];
        for (index, weight) in selected {
            weights[*index] += weight;
        }
        Ok(proposal_gradient(&View::exact(p, r), &weights, Vec::new(), trainable)?.0)
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
    fn native_target(
        label: &str,
        node: usize,
        values: Array2<f64>,
        scored: Option<Vec<bool>>,
    ) -> NativeResponseTarget {
        NativeResponseTarget {
            label: label.into(),
            source_node: node,
            values,
            scored,
            scale: 1.7,
            weight: 0.6,
        }
    }
    fn joint_fd(
        source: &OperatorProgram,
        episodes: &[Episode],
        targets: &NativeResponses,
        row: usize,
        col: usize,
    ) -> f64 {
        let mut losses = Vec::new();
        for sign in [-1., 1.] {
            let mut perturbed = source.clone();
            let op = Arc::make_mut(&mut perturbed.operators[0]);
            if let OperatorBody::Dense {
                values, precision, ..
            } = &mut op.body
            {
                values[[row, col]] += sign * 1e-5;
                *precision = exact_precision(values.iter().copied()).unwrap();
            }
            losses.push(
                measure_with_native(&Device::host(), &perturbed, episodes, targets, 1 << 24)
                    .unwrap()
                    .groups["active"],
            );
        }
        (losses[1] - losses[0]) / 2e-5
    }
    #[test]
    fn native_joint_loss_detects_compensating_exits_and_accumulates_shared_sites() {
        let mut source = program(false, 0.4);
        let interface = Interface::native(2).unwrap();
        source
            .operators
            .push(Arc::new(Operator::identity("positive", interface.clone())));
        source.operators.push(Arc::new(
            Operator::dense(
                "negative",
                interface.clone(),
                interface,
                ndarray::array![[-1., 0.], [0., -1.]],
                exact_precision([-1., 0.]).unwrap(),
                Default::default(),
            )
            .unwrap(),
        ));
        source.nodes = vec![
            Node::Raw { slot: 0 },
            Node::Affine {
                terms: vec![(0, 0)],
                bias: None,
            },
            Node::Raw { slot: 1 },
            Node::Affine {
                terms: vec![(0, 0)],
                bias: None,
            },
            Node::Affine {
                terms: vec![(1, 1), (3, 2)],
                bias: None,
            },
        ];
        source.output = 4;
        let mut teacher = source.clone();
        if let OperatorBody::Dense {
            values, precision, ..
        } = &mut Arc::make_mut(&mut teacher.operators[0]).body
        {
            values[[0, 0]] = 1.1;
            *precision = exact_precision(values.iter().copied()).unwrap();
        }
        let episodes = vec![episode(
            &source,
            &teacher,
            "first",
            "active",
            ndarray::array![[1., 0.5], [10., -3.], [-0.4, 1.]],
            Some(ndarray::Array2::ones((3, 2))),
            Some(vec![true, false, true]),
        )];
        let trace = teacher.execute(&episodes[0].inputs, false).unwrap();
        let mask = Some(vec![true, false, true]);
        let targets = BTreeMap::from([(
            "first".into(),
            vec![
                native_target("exit one", 1, trace.values[1].clone(), mask.clone()),
                native_target(
                    "overlapping exit one",
                    1,
                    trace.values[1].clone(),
                    mask.clone(),
                ),
                native_target("shared exit two", 3, trace.values[3].clone(), mask),
            ],
        )]);
        let d = Device::host();
        let (p, mut resident, base) = prepare(&d, &source, &episodes, &[0], 1 << 24).unwrap();
        let planned =
            attach_responses(&p, &source, &mut resident, &targets, base, 1 << 24).unwrap();
        assert!(planned > base);
        let measurement = scan(&p, &resident).unwrap();
        assert_eq!(measurement.episodes[0].mean_kl, 0.);
        assert!(measurement.objective > 0.);
        assert_eq!(measurement.episodes[0].responses.len(), 3);
        let gradient = gradient(&p, &resident, "active", &[0]).unwrap();
        let actual = d.download(&gradient[&0]).unwrap();
        for row in 0..2 {
            for col in 0..2 {
                assert!(
                    (actual[[row, col]] - joint_fd(&source, &episodes, &targets, row, col)).abs()
                        < 1e-8
                );
            }
        }
        assert!(actual[[0, 0]].abs() > 0.1);
        let fitted = fit_with_native(&d, &source, &episodes, &targets, &[0], settings()).unwrap();
        assert!(fitted.report.best.objective < measurement.objective * 0.1);
        assert_eq!(fitted.report.best.episodes[0].mean_kl, 0.);
        assert_eq!(fitted.program.nodes, source.nodes);
        let mut bad = targets.clone();
        bad.get_mut("first").unwrap()[0].scale = 0.;
        assert!(measure_with_native(&d, &source, &episodes, &bad, 1 << 24).is_err());
        bad = targets.clone();
        bad.get_mut("first").unwrap()[0].scale = 1e200;
        assert!(
            measure_with_native(&d, &source, &episodes, &bad, 1 << 24).is_err(),
            "an underflowed seed coefficient must not silently erase response gradients"
        );
        bad = targets.clone();
        bad.insert("unknown".into(), vec![]);
        assert!(measure_with_native(&d, &source, &episodes, &bad, 1 << 24).is_err());
        bad = targets.clone();
        bad.get_mut("first").unwrap()[0].source_node = usize::MAX;
        assert!(measure_with_native(&d, &source, &episodes, &bad, 1 << 24).is_err());
        assert!(attach_responses(&p, &source, &mut resident, &targets, base, planned - 1).is_err());
    }
    #[test]
    fn native_joint_causal_seeds_include_unscored_prefix_and_output_overlap() {
        let source = program(true, 0.4);
        let teacher = program(true, 1.1);
        let episodes = vec![episode(
            &source,
            &teacher,
            "first",
            "active",
            ndarray::array![[2., 0.5], [-0.4, 1.], [0.2, -0.5]],
            None,
            Some(vec![false, false, true]),
        )];
        let trace = teacher.execute(&episodes[0].inputs, false).unwrap();
        let targets = BTreeMap::from([(
            "first".into(),
            vec![
                native_target(
                    "before attention",
                    1,
                    trace.values[1].clone(),
                    Some(vec![false, false, true]),
                ),
                native_target(
                    "attention output",
                    2,
                    trace.values[2].clone(),
                    Some(vec![false, false, true]),
                ),
            ],
        )]);
        let d = Device::host();
        let (p, mut resident, base) = prepare(&d, &source, &episodes, &[0], 1 << 24).unwrap();
        attach_responses(&p, &source, &mut resident, &targets, base, 1 << 24).unwrap();
        let gradient = gradient(&p, &resident, "active", &[0]).unwrap();
        let actual = d.download(&gradient[&0]).unwrap();
        for row in 0..2 {
            for col in 0..2 {
                assert!(
                    (actual[[row, col]] - joint_fd(&source, &episodes, &targets, row, col)).abs()
                        < 1e-8
                );
            }
        }
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
    fn scheduled_batches_preserve_group_weights_causal_sequences_and_full_scan_selection() {
        let source = program(true, 0.4);
        let teacher = program(true, 1.1);
        let episodes = (0..8)
            .map(|i| {
                episode(
                    &source,
                    &teacher,
                    &format!("episode{i}"),
                    "run",
                    ndarray::array![[2. + i as f64 * 0.1, 0.2], [-1., 0.6], [0.1, 0.4]],
                    None,
                    Some(vec![false, false, true]),
                )
            })
            .collect::<Vec<_>>();
        let d = Device::host();
        let (p, resident, _) = prepare(&d, &source, &episodes, &[0], 1 << 24).unwrap();
        let measured = scan(&p, &resident).unwrap();
        let groups = BTreeMap::from([("run".into(), (0..8).collect())]);
        let schedule = BatchSchedule {
            episodes_per_group: 2,
            scan_every: 3,
            temperature: 0.2,
        };
        let mut cursors = BTreeMap::new();
        let mut seen = Vec::new();
        for _ in 0..4 {
            let selected = scheduled_batch(&groups, &measured, &schedule, &mut cursors).unwrap();
            assert_eq!(selected.len(), 2);
            assert_eq!(selected.iter().map(|(_, w)| w).sum::<f64>(), 1.);
            seen.extend(selected.into_iter().map(|(i, _)| i));
        }
        assert_eq!(seen, (0..8).collect::<Vec<_>>());
        let fitted = fit(
            &d,
            &source,
            &episodes,
            &[0],
            Settings {
                iterations: 7,
                schedule: Some(schedule),
                ..settings()
            },
        )
        .unwrap();
        assert_eq!(
            fitted
                .report
                .iterations
                .iter()
                .map(|i| i.step)
                .collect::<Vec<_>>(),
            vec![0, 3, 6, 7]
        );
        assert_eq!(fitted.report.complete_episode_reverse_passes, 14);
        // Initial exact scan, complete proposals at 0/3/6/7, partial batches at 1/2/4/5 and
        // exact rescoring of the three windows.
        assert_eq!(
            fitted.report.complete_episode_forward_passes,
            8 + 4 * 8 + 4 * 2 + 3 * 8
        );
        assert!(fitted.report.best.objective < fitted.report.initial.objective);
        assert_eq!(
            measure(&d, &fitted.program, &episodes, 1 << 24)
                .unwrap()
                .objective,
            fitted.report.best.objective
        );
        // Full-group batch and scan interval one reduce exactly to the original single-group loop.
        let ordinary = fit(
            &d,
            &source,
            &episodes,
            &[0],
            Settings {
                iterations: 3,
                ..settings()
            },
        )
        .unwrap();
        let scheduled = fit(
            &d,
            &source,
            &episodes,
            &[0],
            Settings {
                iterations: 3,
                schedule: Some(BatchSchedule {
                    episodes_per_group: 8,
                    scan_every: 1,
                    temperature: 0.2,
                }),
                ..settings()
            },
        )
        .unwrap();
        assert_eq!(
            ordinary.program.operators[0].matrix(),
            scheduled.program.operators[0].matrix()
        );
        assert_eq!(
            ordinary.report.best.objective,
            scheduled.report.best.objective
        );
    }

    #[test]
    fn complete_smooth_group_gradient_matches_finite_differences() {
        let source = program(true, 0.4);
        let teacher = program(true, 1.1);
        let episodes = (0..3)
            .map(|i| {
                episode(
                    &source,
                    &teacher,
                    &format!("episode{i}"),
                    if i == 0 { "first" } else { "second" },
                    ndarray::array![[2. + i as f64, 0.2], [-1., 0.6], [0.1, 0.4]],
                    None,
                    Some(vec![false, false, true]),
                )
            })
            .collect::<Vec<_>>();
        let d = Device::host();
        let (p, resident, _) = prepare(&d, &source, &episodes, &[0], 1 << 24).unwrap();
        let measured = scan(&p, &resident).unwrap();
        let groups = BTreeMap::from([("first".into(), vec![0]), ("second".into(), vec![1, 2])]);
        let schedule = BatchSchedule {
            episodes_per_group: 2,
            scan_every: 1,
            temperature: 0.3,
        };
        let selected =
            scheduled_batch(&groups, &measured, &schedule, &mut BTreeMap::new()).unwrap();
        let weights = selected.iter().map(|(_, w)| w).sum::<f64>();
        assert!((weights - 1.).abs() < 1e-15);
        assert_eq!(selected[1].1, selected[2].1);
        let actual = d
            .download(&weighted_gradient(&p, &resident, &selected, &[0]).unwrap()[&0])
            .unwrap();
        for row in 0..2 {
            for col in 0..2 {
                let mut losses = Vec::new();
                for sign in [-1., 1.] {
                    let mut changed = source.clone();
                    let op = Arc::make_mut(&mut changed.operators[0]);
                    if let OperatorBody::Dense {
                        values, precision, ..
                    } = &mut op.body
                    {
                        values[[row, col]] += sign * 1e-5;
                        *precision = exact_precision(values.iter().copied()).unwrap();
                    }
                    let m = measure(&d, &changed, &episodes, 1 << 24).unwrap();
                    let sum = m
                        .groups
                        .values()
                        .map(|v| ((v - m.objective) / schedule.temperature).exp())
                        .sum::<f64>();
                    losses.push(m.objective + schedule.temperature * sum.ln());
                }
                assert!((actual[[row, col]] - (losses[1] - losses[0]) / 2e-5).abs() < 1e-8);
            }
        }
        for invalid in [
            BatchSchedule {
                episodes_per_group: 0,
                ..schedule.clone()
            },
            BatchSchedule {
                scan_every: 0,
                ..schedule.clone()
            },
            BatchSchedule {
                temperature: f64::NAN,
                ..schedule
            },
        ] {
            assert!(
                validate_settings(
                    &[0],
                    &Settings {
                        schedule: Some(invalid),
                        ..settings()
                    }
                )
                .is_err()
            );
        }
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
        let view = View::exact(&b, &be);
        let trace = forward(&view, 0).expect("prefix");
        let labels = episode_target(&view, 0).expect("episode labels");
        let (_, seed) = score(&b, &labels, None, &trace, true).expect("hidden seed");
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
        let teacher_trace = teacher.execute(&full[0].inputs, false).unwrap();
        let responses = BTreeMap::from([(
            "causal".into(),
            vec![
                native_target(
                    "native exit",
                    1,
                    teacher_trace.values[1].clone(),
                    full[0].scored.clone(),
                ),
                native_target(
                    "final norm",
                    3,
                    teacher_trace.values[3].clone(),
                    full[0].scored.clone(),
                ),
            ],
        )]);
        let (a, mut ae, ap) = prepare(&d, &source, &full, &[0], 1 << 24).unwrap();
        let (b, mut be, bp) = prepare_compact(&d, &source, &compact, &[0], 1 << 24, 2).unwrap();
        attach_responses(&a, &source, &mut ae, &responses, ap, 1 << 24).unwrap();
        attach_responses(&b, &source, &mut be, &responses, bp, 1 << 24).unwrap();
        let full_joint = measure_with_native(&d, &source, &full, &responses, 1 << 24).unwrap();
        let compact_joint =
            measure_fixed_head_with_native(&d, &source, &compact, &responses, 1 << 24, 2).unwrap();
        assert!((full_joint.objective - compact_joint.objective).abs() < 2e-13);
        let ag = d
            .download(&gradient(&a, &ae, "active", &[0]).unwrap()[&0])
            .unwrap();
        let bg = d
            .download(&gradient(&b, &be, "active", &[0]).unwrap()[&0])
            .unwrap();
        for row in 0..2 {
            for col in 0..2 {
                assert!((ag[[row, col]] - bg[[row, col]]).abs() < 2e-13);
                assert!(
                    (ag[[row, col]] - joint_fd(&source, &full, &responses, row, col)).abs() < 1e-8
                );
            }
        }
        let joint_fit = fit_fixed_head_with_native(
            &d,
            &source,
            &compact,
            &responses,
            &[0],
            Settings {
                iterations: 2,
                ..settings()
            },
            2,
        )
        .unwrap();
        assert!(joint_fit.report.best.objective <= joint_fit.report.initial.objective);
        let mut dropped = responses;
        dropped.get_mut("causal").unwrap()[0].source_node = source.output;
        assert!(
            measure_fixed_head_with_native(&d, &source, &compact, &dropped, 1 << 24, 2).is_err()
        );
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
    #[test]
    fn proposal_arithmetic_and_exact_cadence_keep_exact_checkpoints() {
        // The host rounds only product operands; CUDA proposals run in f32 storage, through
        // compact head statistics when a fixed head ends the graph.
        let mut devices = vec![Device::host()];
        if let Some(device) = Device::accelerator(gam_gpu::GpuPolicy::Auto).expect("device probe")
            && device.float64()
        {
            devices.push(device);
        }
        let graphs = [
            (program(true, 0.4), program(true, 1.1)),
            (
                with_fixed_head(program(true, 0.4), 1.),
                with_fixed_head(program(true, 1.1), 1.),
            ),
        ];
        for (d, (source, teacher)) in devices
            .iter()
            .flat_map(|d| graphs.iter().map(move |g| (d.clone(), g)))
        {
            let episodes = (0..3)
                .map(|i| {
                    episode(
                        source,
                        teacher,
                        &format!("episode{i}"),
                        if i == 0 { "first" } else { "second" },
                        ndarray::array![[2. + i as f64 * 0.3, 0.2], [-1., 0.6], [0.1, 0.4]],
                        None,
                        Some(vec![false, true, true]),
                    )
                })
                .collect::<Vec<_>>();
            let exact = fit(&d, source, &episodes, &[0], settings()).unwrap();
            let fast = fit(
                &d,
                source,
                &episodes,
                &[0],
                Settings {
                    arithmetic: FitArithmetic::F32,
                    exact_scan_every: 7,
                    ..settings()
                },
            )
            .unwrap();
            assert_eq!(
                fast.report.proposal_device.ends_with("(f32)"),
                !d.is_host(),
                "{}",
                fast.report.proposal_device
            );
            // One initial scan, one per complete window of 7 and one for the final partial window.
            assert_eq!(fast.report.exact_scans, 1 + 60_usize.div_ceil(7));
            assert_eq!(fast.report.iterations.len(), fast.report.exact_scans);
            let windows = fast.report.iterations[1..]
                .iter()
                .map(|i| (i.step - 1) / 7)
                .collect::<Vec<_>>();
            assert_eq!(windows, (0..60_usize.div_ceil(7)).collect::<Vec<_>>());
            // Checkpoint and reported numbers are exact F64 scans of the exported parameters.
            assert_eq!(
                fast.report.final_measurement.objective,
                fast.report.best.objective
            );
            assert!(fast.report.iterations.iter().any(|i| i.step == fast.report.best_step
                && i.measurement.objective == fast.report.best.objective));
            let replay = measure(&d, &fast.program, &episodes, 1 << 24).unwrap();
            assert_eq!(replay.objective, fast.report.best.objective);
            // F32 proposals track their exact rescoring and reach the F64 fit's quality.
            for i in &fast.report.iterations {
                let proposal = i.proposal_objective.unwrap();
                assert!(
                    (proposal - i.measurement.objective).abs()
                        <= 1e-5 * (1. + i.measurement.objective)
                );
            }
            assert!(fast.report.best.objective < fast.report.initial.objective * 0.1);
            assert!(
                (fast.report.best.objective - exact.report.best.objective).abs()
                    <= 1e-3 * exact.report.initial.objective
            );
        }
    }
}

//! Bounded proposal fitting for explicit shared nonlinear programs, independent of acceptance.
//! Uses the existing resident Adam primitive and values-seeded operator VJP. The objective is
//! the maximum Euclidean row error divided by RMS native output norm; no Fisher/token loss.
use crate::{
    artifact_device::mapped_inlined,
    device_program::{DeviceProgram, DeviceTrace},
    operator_program::{
        exact_precision, FamilyInputs, OperatorBody, OperatorProgram, Slot, SlotValues,
    },
};
use gam_gpu::tensor::{Arithmetic, Device, Tensor};
use gam_math::categorical::log_softmax;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Instant,
};

/// Product arithmetic for proposing coefficients. Acceptance and standalone `measure`
/// always use the ordinary f64 evaluator. This does not change stored parameter precision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProposalArithmetic {
    #[default]
    F64,
    F32,
}
impl ProposalArithmetic {
    fn device(self) -> Arithmetic {
        match self {
            Self::F64 => Arithmetic::F64,
            Self::F32 => Arithmetic::F32,
        }
    }
}

/// Finite training-only trial schedule: 1, factor, ..., factor^(max_trials-1).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Backtracking {
    pub factor: f64,
    pub max_trials: usize,
}
/// Proposal-only aggregation. Neither mode changes full TRAIN maximum snapshot selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchObjective {
    #[default]
    GlobalSmoothMaximum,
    /// Equal average of each output group's smooth maximum over selected rows.
    /// Prevents a high-error group from suppressing all gradients of another group.
    MeanGroupSmoothMaximum,
}
/// Optional proposal schedule. Temperature is in normalized squared-error units.
/// Ordinary rows cycle deterministically through the complete training panel; hard rows
/// are the top distinct rows (maximum over groups) from the last complete training scan.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchSchedule {
    pub ordinary_rows: usize,
    pub hard_rows: usize,
    pub scan_every: usize,
    pub temperature: f64,
    #[serde(default)]
    pub objective: BatchObjective,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct TrainingRowWork {
    pub full_scan_rows: usize,
    pub proposal_forward_rows: usize,
    pub proposal_backward_rows: usize,
    pub full_training_scans: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    pub iterations: usize,
    pub forward_rows: usize,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    /// Bound on planned numeric buffers, not CUDA/library/allocator/attention workspaces.
    pub numeric_bytes: usize,
    /// Explicit optional fast proposal products; f64 preserves previous experiment settings.
    #[serde(default)]
    pub arithmetic: ProposalArithmetic,
    #[serde(default)]
    pub backtracking: Option<Backtracking>,
}
/// A complete, disjoint output partition. Labels are reporting metadata; ranges are explicit.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputGroup {
    pub label: String,
    pub start: usize,
    pub end: usize,
}
#[derive(Clone, Debug, Serialize)]
pub struct GroupScale {
    pub group: OutputGroup,
    pub native_rms: f64,
}
#[derive(Clone, Debug, Serialize)]
pub struct GroupIteration {
    pub step: usize,
    pub training_max: f64,
    pub worst_row: usize,
    pub worst_group: usize,
    pub active_recomputed_error: Option<f64>,
}
#[derive(Clone, Debug, Serialize)]
pub struct GroupMeasurement {
    pub maximum: f64,
    pub worst_row: usize,
    pub worst_group: usize,
    pub group_maxima: Vec<f64>,
    pub scales: Vec<GroupScale>,
}
#[derive(Clone, Debug, Serialize)]
pub struct BacktrackingTrial {
    pub proposed_step: usize,
    pub multiplier: f64,
    pub training_max: f64,
    pub accepted: bool,
}
#[derive(Clone, Debug, Serialize)]
pub struct GroupReport {
    pub settings: Settings,
    pub batch_schedule: Option<BatchSchedule>,
    pub training_row_work: TrainingRowWork,
    pub trainable: Vec<usize>,
    pub training_rows: usize,
    pub validation_rows: usize,
    pub training_scales: Vec<GroupScale>,
    pub validation_scales: Vec<GroupScale>,
    pub initial_training_max: f64,
    pub best_training_max: f64,
    pub best_training_groups: Vec<f64>,
    pub accepted_steps: usize,
    pub trial_training_evaluations: usize,
    pub backtracking_trials: Vec<BacktrackingTrial>,
    pub stop_reason: &'static str,
    pub backtracking_seconds: f64,
    pub initial_validation: GroupMeasurement,
    pub final_validation: GroupMeasurement,
    pub best_step: usize,
    pub iterations: Vec<GroupIteration>,
    pub planned_numeric_bytes: usize,
    pub seconds: f64,
    pub scope: &'static str,
}
pub struct GroupFit {
    pub program: OperatorProgram,
    pub report: GroupReport,
}
fn output_groups(groups: &[OutputGroup], width: usize) -> Result<(), String> {
    let mut at = 0;
    let mut labels = BTreeSet::new();
    for group in groups {
        if group.start != at
            || group.end <= group.start
            || group.end > width
            || group.label.is_empty()
            || !labels.insert(&group.label)
        {
            return Err("output groups must be a complete contiguous disjoint partition with unique nonempty labels".into());
        }
        at = group.end;
    }
    if at != width || groups.is_empty() {
        return Err("output groups do not cover the complete output".into());
    }
    Ok(())
}
fn source_inputs(source: &OperatorProgram) -> Result<Vec<usize>, String> {
    if source.declarations.parameters != 0
        || !source.declarations.domains.is_empty()
        || source.declarations.slots.is_empty()
    {
        return Err("fitter requires Raw slots and no external parameters/domains".into());
    }
    source
        .declarations
        .slots
        .iter()
        .map(|slot| match slot {
            Slot::Raw { width } if *width > 0 => Ok(*width),
            _ => Err("fitter requires positive-width Raw slots".into()),
        })
        .collect()
}
fn panel_groups(
    inputs: &[Array2<f64>],
    y: &Array2<f64>,
    widths: &[usize],
    groups: &[OutputGroup],
) -> Result<Vec<GroupScale>, String> {
    if inputs.len() != widths.len()
        || y.nrows() == 0
        || !y.iter().all(|v| v.is_finite())
        || inputs
            .iter()
            .zip(widths)
            .any(|(x, w)| x.dim() != (y.nrows(), *w) || !x.iter().all(|v| v.is_finite()))
    {
        return Err("invalid aligned grouped fitting panel shapes/finite values".into());
    }
    output_groups(groups, y.ncols())?;
    groups
        .iter()
        .map(|group| {
            let values = y.slice(ndarray::s![.., group.start..group.end]);
            let largest = values.iter().fold(0f64, |a, v| a.max(v.abs()));
            if largest == 0. {
                return Err(format!("group {} has zero native RMS", group.label));
            }
            let native_rms = largest
                * (values
                    .iter()
                    .map(|v| (v / largest) * (v / largest))
                    .sum::<f64>()
                    / y.nrows() as f64)
                    .sqrt();
            if !native_rms.is_finite() || native_rms <= 0. {
                return Err(format!(
                    "group {} native RMS overflow/underflow",
                    group.label
                ));
            }
            Ok(GroupScale {
                group: group.clone(),
                native_rms,
            })
        })
        .collect()
}

fn checked_bytes(elements: usize) -> Result<usize, String> {
    elements
        .checked_mul(8)
        .ok_or("numeric byte overflow".into())
}
fn sum_bytes(parts: &[usize]) -> Result<usize, String> {
    parts.iter().try_fold(0usize, |a, b| {
        a.checked_add(*b).ok_or("numeric byte overflow".into())
    })
}
fn validate_settings(s: &Settings) -> Result<(), String> {
    if s.forward_rows == 0
        || s.iterations == 0
        || s.numeric_bytes == 0
        || !s.learning_rate.is_finite()
        || s.learning_rate <= 0.
        || !s.epsilon.is_finite()
        || s.epsilon <= 0.
        || !s.beta1.is_finite()
        || !s.beta2.is_finite()
        || !(0. ..1.).contains(&s.beta1)
        || !(0. ..1.).contains(&s.beta2)
    {
        return Err("invalid explicit optimizer settings".into());
    }
    if let Some(b) = &s.backtracking {
        if !b.factor.is_finite()
            || !(0. ..1.).contains(&b.factor)
            || b.factor == 0.
            || b.max_trials == 0
        {
            return Err("invalid explicit backtracking schedule".into());
        }
    }
    Ok(())
}
struct Panel {
    family: FamilyInputs,
    inputs: Vec<Tensor>,
    target: Tensor,
    groups: Vec<GroupScale>,
}
impl Panel {
    fn new_grouped(
        d: &Device,
        inputs: &[Array2<f64>],
        y: &Array2<f64>,
        groups: Vec<GroupScale>,
    ) -> Result<Self, String> {
        Ok(Self {
            family: FamilyInputs {
                rows: y.nrows(),
                slots: inputs
                    .iter()
                    .map(|x| SlotValues::Raw(Array2::zeros((0, x.ncols()))))
                    .collect(),
                layout: None,
            },
            inputs: inputs
                .iter()
                .map(|x| d.upload(x.view()).map_err(|e| e.to_string()))
                .collect::<Result<_, _>>()?,
            target: d.upload(y.view()).map_err(|e| e.to_string())?,
            groups,
        })
    }
    fn evaluate_group_rows(
        &self,
        p: &DeviceProgram,
        start: usize,
        rows: usize,
    ) -> Result<(DeviceTrace, Tensor, Vec<Vec<f64>>), String> {
        let d = p.device();
        let given = self
            .inputs
            .iter()
            .enumerate()
            .map(|(slot, x)| Ok((slot, d.rows_of(x, start, rows).map_err(|e| e.to_string())?)))
            .collect::<Result<_, String>>()?;
        let trace = p.forward_given(
            &FamilyInputs {
                rows,
                slots: self.family.slots.clone(),
                layout: None,
            },
            given,
        )?;
        let mut residual = d
            .copy(trace.value(p.hidden())?)
            .map_err(|e| e.to_string())?;
        d.axpy(
            &mut residual,
            -1.,
            &d.rows_of(&self.target, start, rows)
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let norms = self
            .groups
            .iter()
            .map(|scale| {
                d.scaled_row_l2(
                    &residual,
                    scale.group.start..scale.group.end,
                    scale.native_rms,
                )
                .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        if norms.iter().flatten().any(|v| !v.is_finite()) {
            return Err("nonfinite grouped proposal objective".into());
        }
        Ok((trace, residual, norms))
    }
    fn scan(&self, p: &DeviceProgram, forward_rows: usize) -> Result<GroupMeasurement, String> {
        self.scan_hard(p, forward_rows, 0).map(|(score, _)| score)
    }
    fn scan_hard(
        &self,
        p: &DeviceProgram,
        forward_rows: usize,
        hard_rows: usize,
    ) -> Result<(GroupMeasurement, Vec<usize>), String> {
        let mut ranked = vec![0f64; if hard_rows > 0 { self.family.rows } else { 0 }];
        let mut result = GroupMeasurement {
            maximum: 0.,
            worst_row: 0,
            worst_group: 0,
            group_maxima: vec![0.; self.groups.len()],
            scales: self.groups.clone(),
        };
        for start in (0..self.family.rows).step_by(forward_rows) {
            let rows = forward_rows.min(self.family.rows - start);
            let (_, _, norms) = self.evaluate_group_rows(p, start, rows)?;
            for (group, values) in norms.iter().enumerate() {
                for (row, value) in values.iter().copied().enumerate() {
                    result.group_maxima[group] = result.group_maxima[group].max(value);
                    let row = start + row;
                    if hard_rows > 0 {
                        ranked[row] = ranked[row].max(value);
                    }
                    if value > result.maximum
                        || (value == result.maximum
                            && (row, group) < (result.worst_row, result.worst_group))
                    {
                        result.maximum = value;
                        result.worst_row = row;
                        result.worst_group = group;
                    }
                }
            }
        }
        let mut hard: Vec<_> = (0..ranked.len()).collect();
        hard.sort_by(|&a, &b| ranked[b].total_cmp(&ranked[a]).then(a.cmp(&b)));
        hard.truncate(hard_rows);
        Ok((result, hard))
    }
    fn selected(&self, d: &Device, rows: &[usize]) -> Result<Self, String> {
        let indices = rows
            .iter()
            .map(|&row| u32::try_from(row).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let indices = d.upload_indices(&indices).map_err(|e| e.to_string())?;
        Ok(Self {
            family: FamilyInputs {
                rows: rows.len(),
                slots: self.family.slots.clone(),
                layout: None,
            },
            inputs: self
                .inputs
                .iter()
                .map(|input| d.gather_rows(input, &indices).map_err(|e| e.to_string()))
                .collect::<Result<_, _>>()?,
            target: d
                .gather_rows(&self.target, &indices)
                .map_err(|e| e.to_string())?,
            groups: self.groups.clone(),
        })
    }
}

fn proposal_rows(step: usize, ordinary_rows: usize, rows: usize, hard: &[usize]) -> Vec<usize> {
    let count = ordinary_rows.min(rows);
    let start = ((step as u128 * count as u128) % rows as u128) as usize;
    let mut selected: Vec<_> = (0..count).map(|offset| (start + offset) % rows).collect();
    for &row in hard {
        if !selected.contains(&row) {
            selected.push(row);
        }
    }
    selected
}
fn smooth_weights(norms: &[Vec<f64>], temperature: f64) -> Result<Vec<Vec<f64>>, String> {
    // The softmax of `v²/T` over every row of every group.
    let logits: Vec<f64> = norms.iter().flatten().map(|v| v * v / temperature).collect();
    let log_weights = log_softmax(&logits).map_err(|e| format!("smooth proposal weights: {e}"))?;
    let mut flat = log_weights.into_iter().map(f64::exp);
    Ok(norms.iter().map(|group| group.iter().zip(flat.by_ref()).map(|(_, w)| w).collect()).collect())
}
fn proposal_weights(norms: &[Vec<f64>], batch: &BatchSchedule) -> Result<Vec<Vec<f64>>, String> {
    match batch.objective {
        BatchObjective::GlobalSmoothMaximum => smooth_weights(norms, batch.temperature),
        BatchObjective::MeanGroupSmoothMaximum => {
            if norms.is_empty() {
                return Err("group-balanced proposal needs output groups".into());
            }
            let count = norms.len() as f64;
            norms
                .iter()
                .map(|group| {
                    let weights = smooth_weights(std::slice::from_ref(group), batch.temperature)?;
                    Ok(weights
                        .into_iter()
                        .next()
                        .ok_or("missing group weights")?
                        .into_iter()
                        .map(|weight| weight / count)
                        .collect())
                })
                .collect()
        }
    }
}

fn snapshot(p: &DeviceProgram, trainable: &[usize]) -> Result<BTreeMap<usize, Tensor>, String> {
    trainable
        .iter()
        .map(|i| {
            Ok((
                *i,
                p.device()
                    .copy(p.dense_parameter(*i)?)
                    .map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}

fn fitting_program(source: &OperatorProgram, widths: &[usize]) -> Result<OperatorProgram, String> {
    let (expanded, _) = mapped_inlined(source)?;
    let mut raw_counts = vec![0usize; widths.len()];
    for node in &expanded.nodes {
        if let crate::operator_program::Node::Raw { slot } = node {
            let count = raw_counts
                .get_mut(*slot)
                .ok_or("Raw slot beyond declarations")?;
            *count += 1;
        }
    }
    if raw_counts.iter().any(|n| *n != 1) {
        return Err("fitter requires exactly one Raw node per declared input slot".into());
    }
    if expanded.nodes.iter().any(|node| {
        !matches!(
            node,
            crate::operator_program::Node::Raw { .. }
                | crate::operator_program::Node::Constant { .. }
                | crate::operator_program::Node::Affine { .. }
                | crate::operator_program::Node::Pointwise { .. }
                | crate::operator_program::Node::Hadamard { .. }
                | crate::operator_program::Node::Concat { .. }
                | crate::operator_program::Node::Transposed { .. }
                | crate::operator_program::Node::Gain { .. }
        )
    }) {
        return Err("fitter numeric plan supports only Raw/Constant/Affine/Pointwise/Hadamard/Concat/Transpose/fixed Gain after Call expansion; attention and other primitive scratch are not budgeted".into());
    }
    Ok(expanded)
}

/// Batched smooth normalized squared-error proposal optimization. Complete TRAIN maximum
/// scans alone select snapshots; validation never selects updates or snapshots. No claim that
/// a minibatch smooth objective bounds the full-panel maximum. Legacy fit_grouped is unchanged.
pub fn fit_grouped_batched(
    d: &Device,
    source: &OperatorProgram,
    train_inputs: &[Array2<f64>],
    train_y: &Array2<f64>,
    valid_inputs: &[Array2<f64>],
    valid_y: &Array2<f64>,
    groups: &[OutputGroup],
    trainable: &[usize],
    settings: Settings,
    schedule: BatchSchedule,
) -> Result<GroupFit, String> {
    fit_grouped_internal(
        d,
        source,
        train_inputs,
        train_y,
        valid_inputs,
        valid_y,
        groups,
        trainable,
        settings,
        Some(schedule),
    )
}
fn fit_grouped_internal(
    d: &Device,
    source: &OperatorProgram,
    train_inputs: &[Array2<f64>],
    train_y: &Array2<f64>,
    valid_inputs: &[Array2<f64>],
    valid_y: &Array2<f64>,
    groups: &[OutputGroup],
    trainable: &[usize],
    settings: Settings,
    batch_schedule: Option<BatchSchedule>,
) -> Result<GroupFit, String> {
    let started = Instant::now();
    validate_settings(&settings)?;
    if let Some(batch) = &batch_schedule {
        if batch.ordinary_rows == 0
            || batch.scan_every == 0
            || !batch.temperature.is_finite()
            || batch.temperature <= 0.
            || batch.ordinary_rows.checked_add(batch.hard_rows).is_none()
        {
            return Err("invalid declared batch schedule".into());
        }
        if settings.backtracking.is_some() {
            return Err(
                "batched schedule cannot use legacy single-active maximum backtracking".into(),
            );
        }
    }
    let widths = source_inputs(source)?;
    let expanded = fitting_program(source, &widths)?;
    let interfaces = expanded.interfaces().map_err(|e| e.to_string())?;
    let output = interfaces[expanded.output].width();
    output_groups(groups, output)?;
    let training_scales = panel_groups(train_inputs, train_y, &widths, groups)?;
    let validation_scales = panel_groups(valid_inputs, valid_y, &widths, groups)?;
    let unique: BTreeSet<_> = trainable.iter().copied().collect();
    if unique.len() != trainable.len() || trainable.is_empty() {
        return Err("empty/duplicate trainable indices".into());
    }
    let mut parameter_elements = 0usize;
    for &index in trainable {
        let op = source
            .operators
            .get(index)
            .ok_or("unknown trainable operator")?;
        let OperatorBody::Dense {
            values, present, ..
        } = &op.body
        else {
            return Err("trainable operator must be Dense literal".into());
        };
        if !present.iter().all(|p| *p) || !values.iter().all(|v| v.is_finite()) {
            return Err("trainable Dense needs all-present finite coefficients; absent-block gradients are not implicitly densified".into());
        }
        parameter_elements = parameter_elements
            .checked_add(values.len())
            .ok_or("parameter size overflow")?;
    }
    let mut program = DeviceProgram::compile_values_bounded(d, &expanded, settings.numeric_bytes)?;
    program.set_arithmetic(settings.arithmetic.device());
    let parameters = checked_bytes(parameter_elements)?;
    let input_elements = train_inputs
        .iter()
        .chain(valid_inputs)
        .try_fold(0usize, |n, x| {
            n.checked_add(x.len()).ok_or("input panel size overflow")
        })?;
    let panels = checked_bytes(
        input_elements
            .checked_add(train_y.len())
            .and_then(|n| n.checked_add(valid_y.len()))
            .ok_or("panel size overflow")?,
    )?;
    let max_rows = settings
        .forward_rows
        .min(train_y.nrows().max(valid_y.nrows()))
        .max(
            batch_schedule
                .as_ref()
                .map_or(0, |b| (b.ordinary_rows + b.hard_rows).min(train_y.nrows())),
        );
    let trace_bytes = program
        .bytes_per_row()
        .checked_mul(max_rows)
        .ok_or("trace size overflow")?;
    let row_output = checked_bytes(max_rows.checked_mul(output).ok_or("output size overflow")?)?;
    // Conservative numeric plan: promotion/replacement copies, two moments, best snapshots,
    // all cotangents/forward values, residual/seeds, and resident panels. Libraries/attention
    // scratch, allocator/context/register/spill memory are explicitly excluded.
    let planned = sum_bytes(&[
        program.operator_numeric_bytes()?,
        parameters
            .checked_mul(if settings.backtracking.is_some() {
                16
            } else {
                10
            })
            .ok_or("parameter plan overflow")?,
        panels,
        if batch_schedule.is_some() {
            checked_bytes(
                max_rows
                    .checked_mul(widths.iter().sum::<usize>() + output)
                    .ok_or("gather plan overflow")?,
            )?
        } else {
            0
        },
        trace_bytes.checked_mul(3).ok_or("trace plan overflow")?,
        row_output
            .checked_mul(if batch_schedule.is_some() { 6 } else { 4 })
            .ok_or("row plan overflow")?,
    ])?;
    if planned > settings.numeric_bytes {
        return Err(format!(
            "resident fitter numeric plan {planned} exceeds {}",
            settings.numeric_bytes
        ));
    }
    program.prepare_dense_parameters(trainable)?;
    let training = Panel::new_grouped(d, train_inputs, train_y, training_scales.clone())?;
    let validation = Panel::new_grouped(d, valid_inputs, valid_y, validation_scales.clone())?;
    let mut moments: BTreeMap<usize, (Tensor, Tensor)> = trainable
        .iter()
        .map(|i| {
            let p = program.dense_parameter(*i)?;
            Ok((
                *i,
                (
                    d.zeros(p.rows(), p.cols()).map_err(|e| e.to_string())?,
                    d.zeros(p.rows(), p.cols()).map_err(|e| e.to_string())?,
                ),
            ))
        })
        .collect::<Result<_, String>>()?;
    let initial_validation = validation.scan(&program, settings.forward_rows)?;
    let mut best = snapshot(&program, trainable)?;
    let mut best_value = f64::INFINITY;
    let mut best_step = 0;
    let mut best_training_groups = Vec::new();
    let mut history = Vec::new();
    let mut accepted_steps = 0;
    let mut backtracking_trials = Vec::new();
    let mut backtracking_seconds = 0.;
    let mut stop_reason = "iteration_budget";
    let mut row_work = TrainingRowWork::default();
    let mut hard_rows = Vec::new();
    let mut last_score: Option<GroupMeasurement> = None;
    for step in 0..=settings.iterations {
        let full_scan = batch_schedule
            .as_ref()
            .is_none_or(|batch| step % batch.scan_every == 0 || step == settings.iterations);
        if full_scan {
            let (score, hard) = training.scan_hard(
                &program,
                settings.forward_rows,
                batch_schedule.as_ref().map_or(0, |b| b.hard_rows),
            )?;
            row_work.full_scan_rows += training.family.rows;
            row_work.full_training_scans += 1;
            hard_rows = hard;
            history.push(GroupIteration {
                step,
                training_max: score.maximum,
                worst_row: score.worst_row,
                worst_group: score.worst_group,
                active_recomputed_error: None,
            });
            if score.maximum < best_value {
                best_value = score.maximum;
                best_step = step;
                best_training_groups = score.group_maxima.clone();
                best = snapshot(&program, trainable)?;
            }
            last_score = Some(score);
        }
        let score = last_score.as_ref().ok_or("initial training scan missing")?;
        let (row, group, value) = (score.worst_row, score.worst_group, score.maximum);
        if step == settings.iterations || value == 0. {
            if value == 0. {
                stop_reason = "zero_training_maximum";
            }
            break;
        }
        let gradients = if let Some(batch) = &batch_schedule {
            let selected =
                proposal_rows(step, batch.ordinary_rows, training.family.rows, &hard_rows);
            row_work.proposal_forward_rows += selected.len();
            row_work.proposal_backward_rows += selected.len();
            let selected = training.selected(d, &selected)?;
            let (trace, residual, norms) =
                selected.evaluate_group_rows(&program, 0, selected.family.rows)?;
            let weights = proposal_weights(&norms, batch)?;
            let mut coefficients = Array2::zeros((selected.family.rows, output));
            for (group, scale) in training.groups.iter().enumerate() {
                for row in 0..selected.family.rows {
                    let coefficient =
                        2. * weights[group][row] / scale.native_rms / scale.native_rms;
                    for column in scale.group.start..scale.group.end {
                        coefficients[[row, column]] = coefficient;
                    }
                }
            }
            if coefficients.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite smooth proposal seed coefficient".into());
            }
            let coefficients = d.upload(coefficients.view()).map_err(|e| e.to_string())?;
            let mut seed = d
                .zeros(selected.family.rows, output)
                .map_err(|e| e.to_string())?;
            d.hadamard(&mut seed, &residual, &coefficients, false)
                .map_err(|e| e.to_string())?;
            program
                .vjp_values_dense(
                    &trace,
                    BTreeMap::from([(program.hidden(), seed)]),
                    &[],
                    trainable,
                    settings.arithmetic.device(),
                )?
                .1
        } else {
            row_work.proposal_forward_rows += 1;
            row_work.proposal_backward_rows += 1;
            // An active row is a subgradient of max row norm; deterministic first-row tie.
            // Normalize in two stages to avoid squaring the native RMS.
            let (trace, residual, active_norms) = training.evaluate_group_rows(&program, row, 1)?;
            history
                .last_mut()
                .ok_or("missing current iteration")?
                .active_recomputed_error = Some(active_norms[group][0]);
            let training_rms = training.groups[group].native_rms;
            let norm = active_norms[group][0] * training_rms;
            if !norm.is_finite() || norm <= 0. {
                return Err("active residual norm cannot be represented".into());
            }
            let range = training.groups[group].group.start..training.groups[group].group.end;
            let group_width = range.end - range.start;
            let active = d
                .columns_of(&residual, range.clone())
                .map_err(|e| e.to_string())?;
            let mut normalized = d.zeros(1, group_width).map_err(|e| e.to_string())?;
            d.axpy(&mut normalized, 1. / norm, &active)
                .map_err(|e| e.to_string())?;
            let mut row_seed = d.zeros(1, group_width).map_err(|e| e.to_string())?;
            d.axpy(&mut row_seed, 1. / training_rms, &normalized)
                .map_err(|e| e.to_string())?;
            let mut seed = d.zeros(1, output).map_err(|e| e.to_string())?;
            d.set_columns(&mut seed, range.start, &row_seed)
                .map_err(|e| e.to_string())?;
            let (_, gradients) = program.vjp_values_dense(
                &trace,
                BTreeMap::from([(program.hidden(), seed)]),
                &[],
                trainable,
                settings.arithmetic.device(),
            )?;
            gradients
        };
        if let Some(schedule) = &settings.backtracking {
            let trial_started = Instant::now();
            let old = snapshot(&program, trainable)?;
            let mut proposed_moments = BTreeMap::new();
            let mut displacement = BTreeMap::new();
            for &index in trainable {
                let previous = old.get(&index).ok_or("missing old parameter")?;
                let mut next = d.copy(previous).map_err(|e| e.to_string())?;
                let (m, v) = moments.get(&index).ok_or("missing moment pair")?;
                let mut nm = d.copy(m).map_err(|e| e.to_string())?;
                let mut nv = d.copy(v).map_err(|e| e.to_string())?;
                d.adam(
                    &mut next,
                    (&mut nm, &mut nv),
                    gradients.get(&index).ok_or("missing gradient")?,
                    settings.learning_rate,
                    (settings.beta1, settings.beta2, settings.epsilon),
                    (accepted_steps + 1) as u64,
                )
                .map_err(|e| e.to_string())?;
                d.axpy(&mut next, -1., previous)
                    .map_err(|e| e.to_string())?;
                displacement.insert(index, next);
                proposed_moments.insert(index, (nm, nv));
            }
            let mut multiplier = 1.;
            let mut accepted = false;
            for _ in 0..schedule.max_trials {
                for &index in trainable {
                    let mut trial = d
                        .copy(old.get(&index).ok_or("missing old parameter")?)
                        .map_err(|e| e.to_string())?;
                    d.axpy(
                        &mut trial,
                        multiplier,
                        displacement.get(&index).ok_or("missing displacement")?,
                    )
                    .map_err(|e| e.to_string())?;
                    program.replace_dense_parameter(index, trial)?;
                }
                let trial = training.scan(&program, settings.forward_rows)?;
                row_work.full_scan_rows += training.family.rows;
                row_work.full_training_scans += 1;
                accepted = trial.maximum < value;
                backtracking_trials.push(BacktrackingTrial {
                    proposed_step: step + 1,
                    multiplier,
                    training_max: trial.maximum,
                    accepted,
                });
                if accepted {
                    break;
                }
                multiplier *= schedule.factor;
            }
            backtracking_seconds += trial_started.elapsed().as_secs_f64();
            if accepted {
                moments = proposed_moments;
                accepted_steps += 1;
            } else {
                for (index, parameter) in old {
                    program.replace_dense_parameter(index, parameter)?;
                }
                stop_reason = "no_strict_training_decrease_in_declared_trials";
                break;
            }
        } else {
            for &index in trainable {
                let mut value = d
                    .copy(program.dense_parameter(index)?)
                    .map_err(|e| e.to_string())?;
                let (m, v) = moments.get_mut(&index).ok_or("missing moment pair")?;
                d.adam(
                    &mut value,
                    (m, v),
                    gradients.get(&index).ok_or("missing gradient")?,
                    settings.learning_rate,
                    (settings.beta1, settings.beta2, settings.epsilon),
                    (step + 1) as u64,
                )
                .map_err(|e| e.to_string())?;
                program.replace_dense_parameter(index, value)?;
            }
            accepted_steps += 1;
        }
    }
    for (index, value) in best {
        program.replace_dense_parameter(index, value)?;
    }
    let final_validation = validation.scan(&program, settings.forward_rows)?;
    let mut fitted = source.clone();
    for &index in trainable {
        let values = d
            .download(program.dense_parameter(index)?)
            .map_err(|e| e.to_string())?;
        if !values.iter().all(|v| v.is_finite()) {
            return Err("nonfinite final parameter".into());
        }
        let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
        let op = Arc::make_mut(&mut fitted.operators[index]);
        let OperatorBody::Dense {
            values: stored,
            precision: p,
            ..
        } = &mut op.body
        else {
            return Err("trainable source changed kind".into());
        };
        *stored = values;
        *p = precision;
    }
    let initial_training_max = history[0].training_max;
    Ok(GroupFit {
        program: fitted,
        report: GroupReport {
            settings,
            batch_schedule: batch_schedule.clone(),
            training_row_work: row_work,
            trainable: trainable.to_vec(),
            training_rows: train_y.nrows(),
            validation_rows: valid_y.nrows(),
            training_scales,
            validation_scales,
            initial_training_max,
            best_training_max: best_value,
            best_training_groups,
            accepted_steps,
            trial_training_evaluations: backtracking_trials.len(),
            backtracking_trials,
            stop_reason,
            backtracking_seconds,
            initial_validation,
            final_validation,
            best_step,
            iterations: history,
            planned_numeric_bytes: planned,
            seconds: started.elapsed().as_secs_f64(),
            scope: if batch_schedule.is_some() {
                "Batched proposal optimization only. Deterministic cyclic ordinary rows plus retained top-hard rows across all groups. Explicit global smoothmax or mean of per-group smoothmax normalized squared-error gradients on selected rows; minibatch surrogate is NOT a full-panel maximum bound. Complete training maximum scanned initially, periodically and finally; ONLY these full scans select best training snapshot. Validation excluded from updates/selection. Row-separable ordinary graph only. Legacy active-row backtracking forbidden. Explicit arithmetic setting applies to proposal products; exact decoded Local acceptance unchanged. Numeric plan includes gathered input/target/seed tensors, excludes host weight arrays/product conversion/library/allocator/context/register/spill scratch."
            } else {
                "Proposal optimization only, fixed finite f64 input/output rows. Explicit settings.arithmetic governs forward and backward products, including training/validation proposal scores; parameter/moment storage and other primitives remain f64. The f32 option uses rounded-product surrogate gradients, not derivatives through rounding. Maximum over ALL output groups and rows, each group normalized by its own complete native-family RMS. Optional declared finite backtracking scans the SAME full training maximum and accepts only strict decrease; proposed moments and Adam time commit only with an accepted step, exhaustion restores old parameters and stops. Default uses original fixed-step Adam. Deterministic active-group/row max-norm Adam; full row scan in declared chunks and one-row reverse. Single-row GEMM rounding may differ from batch; recomputed error recorded; nonconvex, no optimum certificate. Best training snapshot, validation excluded from selection. Resident numerical parameters/gradients/moments/inputs/targets; O(rows) norms downloaded each step and final parameters downloaded. Numeric plan excludes product conversion/library/allocator/context/register/spill scratch. Final decoded f64 measurement and acceptance are separate."
            },
        },
    })
}

/// Ordinary f64 measurement of all output groups, independent of fit product arithmetic.
pub fn measure_grouped(
    d: &Device,
    source: &OperatorProgram,
    inputs: &[Array2<f64>],
    y: &Array2<f64>,
    groups: &[OutputGroup],
    numeric_bytes: usize,
    forward_rows: usize,
) -> Result<GroupMeasurement, String> {
    if forward_rows == 0 {
        return Err("zero measurement forward_rows".into());
    }
    let widths = source_inputs(source)?;
    let expanded = fitting_program(source, &widths)?;
    let interfaces = expanded.interfaces().map_err(|e| e.to_string())?;
    if y.ncols() != interfaces[expanded.output].width() {
        return Err("measurement output width mismatch".into());
    }
    let scales = panel_groups(inputs, y, &widths, groups)?;
    let p = DeviceProgram::compile_values_bounded(d, &expanded, numeric_bytes)?;
    let elements = inputs.iter().try_fold(y.len(), |n, x| {
        n.checked_add(x.len()).ok_or("panel size overflow")
    })?;
    let planned = sum_bytes(&[
        p.operator_numeric_bytes()?,
        checked_bytes(elements)?,
        p.bytes_per_row()
            .checked_mul(forward_rows.min(y.nrows()))
            .ok_or("trace overflow")?,
        checked_bytes(y.len())?
            .checked_mul(2)
            .ok_or("residual overflow")?,
    ])?;
    if planned > numeric_bytes {
        return Err("measurement numeric plan exceeds budget".into());
    }
    Panel::new_grouped(d, inputs, y, scales)?.scan(&p, forward_rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_program::{Declarations, Interface, Law, Node, Operator, Rule};
    fn panel(x: &Array2<f64>, y: &Array2<f64>, input: usize, output: usize) -> Result<f64, String> {
        if x.nrows() == 0
            || x.nrows() != y.nrows()
            || x.ncols() != input
            || y.ncols() != output
            || !x.iter().chain(y.iter()).all(|v| v.is_finite())
        {
            return Err("invalid fitting panel shapes/finite values".into());
        }
        // Scaled accumulation prevents avoidable overflow/underflow in the fixed native denominator.
        let largest = y.iter().fold(0f64, |a, v| a.max(v.abs()));
        if largest == 0. {
            return Err("zero native RMS: normalized proposal objective undefined".into());
        }
        let rms = largest
            * (y.iter().map(|v| (v / largest) * (v / largest)).sum::<f64>() / y.nrows() as f64)
                .sqrt();
        if !rms.is_finite() || rms <= 0. {
            return Err("nonfinite/underflow native RMS".into());
        }
        Ok(rms)
    }
    fn maximum(values: &[f64]) -> (usize, f64) {
        values
            .iter()
            .copied()
            .enumerate()
            .fold(
                (0, values[0]),
                |best, (i, v)| if v > best.1 { (i, v) } else { best },
            )
    }
    impl Panel {
        fn new(d: &Device, x: &Array2<f64>, y: &Array2<f64>, rms: f64) -> Result<Self, String> {
            Self::new_grouped(
                d,
                std::slice::from_ref(x),
                y,
                vec![GroupScale {
                    group: OutputGroup {
                        label: "output".into(),
                        start: 0,
                        end: y.ncols(),
                    },
                    native_rms: rms,
                }],
            )
        }
        fn evaluate_rows(
            &self,
            p: &DeviceProgram,
            start: usize,
            rows: usize,
        ) -> Result<(DeviceTrace, Tensor, Vec<f64>), String> {
            let (trace, residual, mut norms) = self.evaluate_group_rows(p, start, rows)?;
            Ok((trace, residual, norms.remove(0)))
        }
    }
    fn model(weight: f64, offset: f64) -> OperatorProgram {
        let i = Interface::native(1).expect("interface");
        let c = Interface::constant();
        let op = |name: &str, value: f64, cols: &crate::operator_program::Interface| {
            Arc::new(
                Operator::dense(
                    name,
                    i.clone(),
                    cols.clone(),
                    Array2::from_elem((1, 1), value),
                    exact_precision([value]).expect("precision"),
                    Default::default(),
                )
                .expect("operator"),
            )
        };
        OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 1 }],
                parameters: 0,
            },
            bases: vec![],
            operators: vec![
                op("shared weight", weight, &i),
                op("shared offset", offset, &c),
                op("second binding", 1.5, &i),
            ],
            rules: vec![Rule {
                name: "shared nonlinear function".into(),
                inputs: vec![i],
                nodes: vec![
                    Node::Param { index: 0 },
                    Node::Affine {
                        terms: vec![(0, 0)],
                        bias: Some(1),
                    },
                    Node::Pointwise {
                        input: 1,
                        laws: vec![Law::GeluTanh],
                    },
                ],
                output: 2,
            }],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 2)],
                    bias: None,
                },
                Node::Call {
                    rule: 0,
                    arguments: vec![0],
                },
                Node::Call {
                    rule: 0,
                    arguments: vec![1],
                },
                Node::Concat { parts: vec![2, 3] },
            ],
            output: 4,
        }
    }
    fn target(p: &OperatorProgram, x: &Array2<f64>) -> Array2<f64> {
        p.execute(
            &FamilyInputs {
                rows: x.nrows(),
                slots: vec![SlotValues::Raw(x.clone())],
                layout: None,
            },
            false,
        )
        .expect("teacher")
        .values[p.output]
            .clone()
    }
    fn settings() -> Settings {
        Settings {
            iterations: 160,
            forward_rows: 3,
            learning_rate: 0.02,
            beta1: 0.9,
            beta2: 0.99,
            epsilon: 1e-8,
            numeric_bytes: 1 << 20,
            arithmetic: ProposalArithmetic::F64,
            backtracking: None,
        }
    }
    #[test]
    fn batch_realizable_generated_interior_teacher_fits_full_maps_and_shared_body() {
        use crate::composed_rule_search::{compile, Expr, Unary, UseSpec};
        let expression = Expr::Unary(
            Unary::GeluTanh,
            Box::new(Expr::Affine(Box::new(Expr::Unary(
                Unary::GeluTanh,
                Box::new(Expr::Argument(0)),
            )))),
        );
        let uses = [UseSpec {
            input_width: 2,
            output_width: 2,
        }; 2];
        let teacher = compile(&expression, 2, &uses, 712).expect("generated teacher");
        let teacher = crate::artifact::Artifact::native(&teacher.program)
            .expect("teacher artifact")
            .f32_literals()
            .expect("teacher f32");
        let bytes = teacher.to_bytes().expect("teacher bytes");
        let teacher = crate::artifact::Artifact::from_bytes(&bytes, &teacher.program.declarations)
            .expect("ordinary teacher");
        let proposal = compile(&expression, 2, &uses, 981).expect("independent initialization");
        let x = vec![
            Array2::from_shape_fn((16, 2), |(r, c)| ((r * 7 + c * 3) % 19) as f64 / 9. - 1.),
            Array2::from_shape_fn((16, 2), |(r, c)| ((r * 5 + c * 11) % 23) as f64 / 11. - 1.),
        ];
        let y = grouped_target(&teacher.program, &x);
        let eval_x = vec![
            Array2::from_shape_fn((32, 2), |(r, c)| {
                ((r * 13 + c * 5) % 37) as f64 / 9. - 2. + 0.031
            }),
            Array2::from_shape_fn((32, 2), |(r, c)| {
                ((r * 17 + c * 7) % 41) as f64 / 10. - 2. + 0.047
            }),
        ];
        let eval_y = grouped_target(&teacher.program, &eval_x);
        let mut settings = settings();
        settings.iterations = 2500;
        settings.forward_rows = 8;
        settings.learning_rate = 0.01;
        let fit = fit_grouped_batched(
            &Device::host(),
            &proposal.program,
            &x,
            &y,
            &eval_x,
            &eval_y,
            &proposal.groups,
            &proposal.trainable,
            settings,
            BatchSchedule {
                ordinary_rows: 8,
                hard_rows: 4,
                scan_every: 10,
                temperature: 0.02,
                objective: BatchObjective::GlobalSmoothMaximum,
            },
        )
        .expect("generated composition fit");
        eprintln!(
            "generated interior composition: initial={}, best={}, replay validation={}",
            fit.report.initial_training_max,
            fit.report.best_training_max,
            fit.report.final_validation.maximum
        );
        assert!(
            fit.report.best_training_max < 1e-3,
            "{}",
            fit.report.best_training_max
        );
        assert!(fit.report.final_validation.maximum.is_finite());
        assert_eq!(fit.report.best_training_groups.len(), 2);
        assert_eq!(fit.program.rules.len(), 1);
        let saved = crate::artifact::Artifact::native(&fit.program)
            .expect("fit artifact")
            .f32_literals()
            .expect("f32 fit");
        let bytes = saved.to_bytes().expect("saved fit");
        let saved = crate::artifact::Artifact::from_bytes(&bytes, &saved.program.declarations)
            .expect("ordinary fit replay");
        let replay = measure_grouped(
            &Device::host(),
            &saved.program,
            &x,
            &y,
            &proposal.groups,
            1 << 20,
            8,
        )
        .expect("saved measurement");
        assert!(replay.maximum < 1e-3, "{}", replay.maximum);
        let eval_replay = measure_grouped(
            &Device::host(),
            &saved.program,
            &eval_x,
            &eval_y,
            &proposal.groups,
            1 << 20,
            8,
        )
        .expect("frozen heldout replay");
        eprintln!(
            "CALIBRATION_RESULT={}",
            serde_json::json!({"scope":"SYNTHETIC supplied correct topology coefficient calibration, NOT structure discovery or LLM evidence","expression":expression,"width":2,"uses":2,"teacher_seed":712,"proposal_seed":981,"teacher_ordinary_f32_replay":true,"train_rows":16,"train_site_rows_per_scan":32,"eval_rows":32,"eval_site_rows_per_scan":64,"train_inputs":"two deterministic modular 2D panels in approximately [-1,1]","eval_inputs":"two independent deterministic modular 2D panels in approximately [-2,2]","initial_training_max":fit.report.initial_training_max,"best_training_max":fit.report.best_training_max,"saved_f32_training_max":replay.maximum,"heldout_max":fit.report.final_validation.maximum,"saved_f32_heldout_max":eval_replay.maximum,"train_threshold":0.001,"heldout_threshold":null,"best_step":fit.report.best_step,"accepted_updates":fit.report.accepted_steps,"row_work":fit.report.training_row_work,"initial_final_validation_scans":2,"additional_saved_train_scans":1,"additional_saved_eval_scans":1,"settings":fit.report.settings,"batch_schedule":fit.report.batch_schedule,"fit_seconds":fit.report.seconds})
        );
    }
    #[test]
    fn retained_hard_rows_cover_different_groups_and_ordinary_cycle_covers_every_row() {
        let interface = Interface::native(2).expect("interface");
        let p = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                parameters: 0,
                slots: vec![Slot::Raw { width: 2 }],
            },
            bases: vec![],
            rules: vec![],
            operators: vec![Arc::new(Operator::identity("I", interface))],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
            ],
            output: 1,
        };
        let x = ndarray::array![[10., 0.], [0., 9.], [1., 1.], [8., 0.]];
        let y = Array2::ones((4, 2));
        let scales = panel_groups(std::slice::from_ref(&x), &y, &[2], &groups())
            .expect("native denominators");
        let device = Device::host();
        let resident =
            DeviceProgram::compile_values(&device, &p).expect("row-separable resident program");
        let panel =
            Panel::new_grouped(&device, std::slice::from_ref(&x), &y, scales).expect("panel");
        let (score, hard) = panel.scan_hard(&resident, 2, 2).expect("one full scan");
        assert_eq!(score.maximum, 9.);
        assert_eq!(hard, vec![0, 1]);
        let mut visited = BTreeSet::new();
        for step in 0..3 {
            let rows = proposal_rows(step, 2, 5, &hard);
            assert_eq!(
                rows.iter().copied().collect::<BTreeSet<_>>().len(),
                rows.len()
            );
            visited.extend(rows[..2].iter().copied());
            assert!(hard.iter().all(|row| rows.contains(row)));
        }
        assert_eq!(visited, (0..5).collect());
    }
    #[test]
    fn batch_schedule_is_explicit_and_zero_rms_and_cross_row_refusals_remain() {
        let p = model(0.6, 0.1);
        let x = ndarray::array![[1.], [2.]];
        let y = target(&p, &x);
        let invalid = BatchSchedule {
            ordinary_rows: 1,
            hard_rows: 1,
            scan_every: 0,
            temperature: 0.1,
            objective: BatchObjective::GlobalSmoothMaximum,
        };
        assert!(fit_grouped_batched(
            &Device::host(),
            &p,
            std::slice::from_ref(&x),
            &y,
            std::slice::from_ref(&x),
            &y,
            &[OutputGroup {
                label: "all".into(),
                start: 0,
                end: 1
            }],
            &[0, 1],
            settings(),
            invalid
        )
        .is_err());
        let good = BatchSchedule {
            ordinary_rows: 1,
            hard_rows: 1,
            scan_every: 2,
            temperature: 0.1,
            objective: BatchObjective::GlobalSmoothMaximum,
        };
        let zero = Array2::zeros(y.dim());
        assert!(fit_grouped_batched(
            &Device::host(),
            &p,
            std::slice::from_ref(&x),
            &zero,
            std::slice::from_ref(&x),
            &zero,
            &[OutputGroup {
                label: "all".into(),
                start: 0,
                end: 1
            }],
            &[0, 1],
            settings(),
            good.clone()
        )
        .is_err());
        let mut cross = p.clone();
        cross.nodes.push(Node::Attend {
            query: 0,
            key: 0,
            value: 0,
            scale: crate::operator_program::Scale::One,
            rotary: None,
            causal: true,
        });
        cross.output = cross.nodes.len() - 1;
        assert!(fit_grouped_batched(
            &Device::host(),
            &cross,
            std::slice::from_ref(&x),
            &y,
            std::slice::from_ref(&x),
            &y,
            &[OutputGroup {
                label: "all".into(),
                start: 0,
                end: 1
            }],
            &[0, 1],
            settings(),
            good.clone()
        )
        .is_err());
        let mut old = settings();
        old.backtracking = Some(Backtracking {
            factor: 0.5,
            max_trials: 2,
        });
        assert!(fit_grouped_batched(
            &Device::host(),
            &p,
            std::slice::from_ref(&x),
            &y,
            std::slice::from_ref(&x),
            &y,
            &[OutputGroup {
                label: "all".into(),
                start: 0,
                end: 1
            }],
            &[0, 1],
            old,
            good
        )
        .is_err());
        let weights = smooth_weights(&[vec![1., 1.], vec![1., 1.]], 0.1).expect("stable tie");
        assert!(weights.iter().flatten().all(|w| (w - 0.25).abs() <= 4.0 * f64::EPSILON), "{weights:?}");
        let huge = smooth_weights(&[vec![f64::MAX]], 0.1);
        assert!(huge.is_err());
    }
    #[test]
    fn group_balanced_proposal_preserves_clean_gradient_under_scalar_imbalance() {
        let scalar = Interface::native(1).expect("scalar");
        let vector = Interface::native(2).expect("vector");
        let values = ndarray::array![[3.], [17.]];
        let p = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                parameters: 0,
                slots: vec![Slot::Raw { width: 1 }],
            },
            bases: vec![],
            rules: vec![],
            operators: vec![Arc::new(
                Operator::dense(
                    "separate output rows",
                    vector,
                    scalar,
                    values.clone(),
                    exact_precision(values.iter().copied()).expect("precision"),
                    Default::default(),
                )
                .expect("dense"),
            )],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
            ],
            output: 1,
        };
        let x = ndarray::array![[1.]];
        let y = ndarray::array![[1., 1.]];
        let mut optimizer = settings();
        optimizer.iterations = 1;
        optimizer.learning_rate = 0.01;
        let schedule = BatchSchedule {
            ordinary_rows: 1,
            hard_rows: 0,
            scan_every: 1,
            temperature: 0.1,
            objective: BatchObjective::GlobalSmoothMaximum,
        };
        let global = proposal_weights(&[vec![2.], vec![16.]], &schedule).expect("global weights");
        assert_eq!(global[0][0], 0.);
        let mut balanced = schedule.clone();
        balanced.objective = BatchObjective::MeanGroupSmoothMaximum;
        assert_eq!(
            proposal_weights(&[vec![2.], vec![16.]], &balanced).expect("balanced"),
            vec![vec![0.5], vec![0.5]]
        );
        let fit = |batch| {
            fit_grouped_batched(
                &Device::host(),
                &p,
                std::slice::from_ref(&x),
                &y,
                std::slice::from_ref(&x),
                &y,
                &groups(),
                &[0],
                optimizer.clone(),
                batch,
            )
            .expect("fit")
        };
        let old = fit(schedule);
        let new = fit(balanced);
        let clean = |fit: &GroupFit| match &fit.program.operators[0].body {
            OperatorBody::Dense { values, .. } => values[(0, 0)],
            _ => panic!("dense parameter"),
        };
        assert_eq!(clean(&old), 3.);
        assert!(clean(&new) < 2.999, "clean gradient must update the writer");
        for result in [&old, &new] {
            assert_eq!(result.report.best_step, 1);
            assert!(result.report.best_training_max < result.report.initial_training_max);
        }
    }
    #[test]
    fn group_balanced_weights_match_objective_derivatives_and_legacy_json() {
        let old =
            serde_json::json!({"ordinary_rows":3,"hard_rows":0,"scan_every":1,"temperature":0.7});
        let mut schedule: BatchSchedule = serde_json::from_value(old).expect("legacy schedule");
        assert_eq!(schedule.objective, BatchObjective::GlobalSmoothMaximum);
        let norms = vec![vec![0.2, 1.3, 0.8], vec![3.1, 2.9, 3.2]];
        assert_eq!(
            proposal_weights(&norms, &schedule).expect("default"),
            smooth_weights(&norms, 0.7).expect("legacy")
        );
        schedule.objective = BatchObjective::MeanGroupSmoothMaximum;
        let weights = proposal_weights(&norms, &schedule).expect("weights");
        let objective = |values: &[Vec<f64>]| {
            values
                .iter()
                .map(|group| {
                    let logits: Vec<f64> = group.iter().map(|x| x * x / schedule.temperature).collect();
                    schedule.temperature * gam_math::categorical::log_sum_exp(&logits).expect("finite logits")
                })
                .sum::<f64>()
                / values.len() as f64
        };
        for group in 0..norms.len() {
            assert!((weights[group].iter().sum::<f64>() - 0.5).abs() < 1e-14);
            for row in 0..norms[group].len() {
                let mut plus = norms.clone();
                let mut minus = norms.clone();
                plus[group][row] += 1e-5;
                minus[group][row] -= 1e-5;
                let finite_difference = (objective(&plus) - objective(&minus)) / 2e-5;
                let derivative = 2. * norms[group][row] * weights[group][row];
                assert!((finite_difference - derivative).abs() < 1e-8);
            }
        }
        assert!(proposal_weights(&[], &schedule).is_err());
        assert!(proposal_weights(&[vec![]], &schedule).is_err());
        assert!(proposal_weights(&[vec![f64::MAX]], &schedule).is_err());
    }
    #[test]
    fn absent_backtracking_preserves_legacy_settings() {
        let mut value = serde_json::to_value(settings()).expect("settings JSON");
        value
            .as_object_mut()
            .expect("object")
            .remove("backtracking");
        let decoded: Settings = serde_json::from_value(value).expect("legacy settings");
        assert!(decoded.backtracking.is_none());
        let mut invalid = settings();
        invalid.backtracking = Some(Backtracking {
            factor: 1.,
            max_trials: 2,
        });
        assert!(validate_settings(&invalid).is_err());
    }

    #[test]
    fn active_single_row_reverse_matches_full_joint_shared_call_gradient() {
        let p = model(0.6, 0.1);
        let x = ndarray::array![[-1.2], [0.4], [1.8]];
        let y = target(&model(1.2, -0.2), &x);
        let d = Device::host();
        let (expanded, _) = mapped_inlined(&p).expect("expand");
        let mut resident = DeviceProgram::compile_values(&d, &expanded).expect("compile");
        resident
            .prepare_dense_parameters(&[0, 1])
            .expect("parameters");
        let rms = panel(&x, &y, 1, 2).expect("scale");
        let data = Panel::new(&d, &x, &y, rms).expect("resident panel");
        let (full, full_residual, norms) = data.evaluate_rows(&resident, 0, 3).expect("full trace");
        let (row, value) = maximum(&norms);
        let (single, single_residual, single_norms) = data
            .evaluate_rows(&resident, row, 1)
            .expect("active row trace");
        assert_eq!(value, single_norms[0]);
        assert_eq!(
            d.download(&d.rows_of(&full_residual, row, 1).expect("row"))
                .expect("download"),
            d.download(&single_residual).expect("single")
        );
        let mut active = d.zeros(1, 2).expect("seed");
        d.axpy(&mut active, 1. / (rms * rms * value), &single_residual)
            .expect("normalize seed");
        let mut full_seed = d.zeros(3, 2).expect("full seed");
        d.set_rows(&mut full_seed, row, &active)
            .expect("active seed row");
        let (_, full_grad) = resident
            .vjp_values_dense(
                &full,
                BTreeMap::from([(resident.hidden(), full_seed)]),
                &[],
                &[0, 1],
                Arithmetic::F64,
            )
            .expect("full reverse");
        let (_, single_grad) = resident
            .vjp_values_dense(
                &single,
                BTreeMap::from([(resident.hidden(), active)]),
                &[],
                &[0, 1],
                Arithmetic::F64,
            )
            .expect("one-row reverse");
        for index in [0, 1] {
            assert_eq!(
                d.download(&full_grad[&index]).expect("full gradient"),
                d.download(&single_grad[&index]).expect("single gradient")
            );
        }
    }
    fn grouped_target(p: &OperatorProgram, x: &[Array2<f64>]) -> Array2<f64> {
        p.execute(
            &FamilyInputs {
                rows: x[0].nrows(),
                slots: x.iter().cloned().map(SlotValues::Raw).collect(),
                layout: None,
            },
            false,
        )
        .expect("teacher")
        .values[p.output]
            .clone()
    }
    fn groups() -> Vec<OutputGroup> {
        vec![
            OutputGroup {
                label: "large native use".into(),
                start: 0,
                end: 1,
            },
            OutputGroup {
                label: "tiny native use".into(),
                start: 1,
                end: 2,
            },
        ]
    }
}

//! Bounded proposal fitting for explicit shared nonlinear programs, independent of acceptance.
//! Uses the existing resident Adam primitive and values-seeded operator VJP. The objective is
//! the maximum Euclidean row error divided by RMS native output norm; no Fisher/token loss.
use crate::{
    artifact_device::mapped_inlined,
    device_program::{DeviceProgram, DeviceTrace},
    operator_program::{
        FamilyInputs, OperatorBody, OperatorProgram, Slot, SlotValues, exact_precision,
    },
};
use gam_gpu::tensor::{Arithmetic, Device, Tensor};
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
#[derive(Clone, Debug, Serialize)]
pub struct Iteration {
    pub step: usize,
    pub training_max: f64,
    pub worst_row: usize,
    pub active_recomputed_error: Option<f64>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub settings: Settings,
    pub trainable: Vec<usize>,
    pub training_rows: usize,
    pub validation_rows: usize,
    pub training_rms: f64,
    pub validation_rms: f64,
    pub initial_training_max: f64,
    pub best_training_max: f64,
    pub initial_validation_max: f64,
    pub final_validation_max: f64,
    pub best_step: usize,
    pub iterations: Vec<Iteration>,
    pub planned_numeric_bytes: usize,
    pub seconds: f64,
    pub scope: &'static str,
}
pub struct Fit {
    pub program: OperatorProgram,
    pub report: Report,
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
    let largest = norms.iter().flatten().try_fold(0f64, |largest, &value| {
        let squared = value * value;
        if !squared.is_finite() {
            Err("smooth squared error overflow")
        } else {
            Ok(largest.max(squared))
        }
    })?;
    let mut weights: Vec<Vec<f64>> = norms
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|v| ((v * v - largest) / temperature).exp())
                .collect()
        })
        .collect();
    let total = weights.iter().flatten().sum::<f64>();
    if !total.is_finite() || total <= 0. {
        return Err("invalid smooth proposal normalizer".into());
    }
    for value in weights.iter_mut().flatten() {
        *value /= total;
    }
    Ok(weights)
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

/// Fit every declared dense parameter jointly; the same operator appearing in many Calls has
/// one parameter/moment pair and receives the sum of all actual call cotangents. No arity/rank cap.
/// Requires one Raw input slot, a values-only output, and all-present trainable Dense literals.
/// Sparse present masks, derived-operator constraints, teacher episodes and structural search
/// are outside this primitive. The example preserves the complete artifact and performs f32 replay.
pub fn fit(
    d: &Device,
    source: &OperatorProgram,
    train_x: &Array2<f64>,
    train_y: &Array2<f64>,
    valid_x: &Array2<f64>,
    valid_y: &Array2<f64>,
    trainable: &[usize],
    settings: Settings,
) -> Result<Fit, String> {
    if source.declarations.slots.len() != 1 {
        return Err(
            "legacy fit requires one Raw input; use fit_grouped for multiple inputs".into(),
        );
    }
    let groups = [OutputGroup {
        label: "output".into(),
        start: 0,
        end: train_y.ncols(),
    }];
    let result = fit_grouped(
        d,
        source,
        std::slice::from_ref(train_x),
        train_y,
        std::slice::from_ref(valid_x),
        valid_y,
        &groups,
        trainable,
        settings,
    )?;
    let r = result.report;
    Ok(Fit {
        program: result.program,
        report: Report {
            settings: r.settings,
            trainable: r.trainable,
            training_rows: r.training_rows,
            validation_rows: r.validation_rows,
            training_rms: r.training_scales[0].native_rms,
            validation_rms: r.validation_scales[0].native_rms,
            initial_training_max: r.initial_training_max,
            best_training_max: r.best_training_max,
            initial_validation_max: r.initial_validation.maximum,
            final_validation_max: r.final_validation.maximum,
            best_step: r.best_step,
            iterations: r
                .iterations
                .into_iter()
                .map(|i| Iteration {
                    step: i.step,
                    training_max: i.training_max,
                    worst_row: i.worst_row,
                    active_recomputed_error: i.active_recomputed_error,
                })
                .collect(),
            planned_numeric_bytes: r.planned_numeric_bytes,
            seconds: r.seconds,
            scope: r.scope,
        },
    })
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

/// Joint fitting of aligned Raw arguments with a maximum over separately normalized output uses.
pub fn fit_grouped(
    d: &Device,
    source: &OperatorProgram,
    train_inputs: &[Array2<f64>],
    train_y: &Array2<f64>,
    valid_inputs: &[Array2<f64>],
    valid_y: &Array2<f64>,
    groups: &[OutputGroup],
    trainable: &[usize],
    settings: Settings,
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
        None,
    )
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
            let weights = smooth_weights(&norms, batch.temperature)?;
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
                "Batched proposal optimization only. Deterministic cyclic ordinary rows plus retained top-hard rows across all groups. Stable smoothmax normalized squared-error gradients on selected rows/groups; minibatch surrogate is NOT a full-panel maximum bound. Complete training maximum scanned initially, periodically and finally; ONLY these full scans select best training snapshot. Validation excluded from updates/selection. Row-separable ordinary graph only. Legacy active-row backtracking forbidden. Explicit arithmetic setting applies to proposal products; exact decoded Local acceptance unchanged. Numeric plan includes gathered input/target/seed tensors, excludes host weight arrays/product conversion/library/allocator/context/register/spill scratch."
            } else {
                "Proposal optimization only, fixed finite f64 input/output rows. Explicit settings.arithmetic governs forward and backward products, including training/validation proposal scores; parameter/moment storage and other primitives remain f64. The f32 option uses rounded-product surrogate gradients, not derivatives through rounding. Maximum over ALL output groups and rows, each group normalized by its own complete native-family RMS. Optional declared finite backtracking scans the SAME full training maximum and accepts only strict decrease; proposed moments and Adam time commit only with an accepted step, exhaustion restores old parameters and stops. Default uses original fixed-step Adam. Deterministic active-group/row max-norm Adam; full row scan in declared chunks and one-row reverse. Single-row GEMM rounding may differ from batch; recomputed error recorded; nonconvex, no optimum certificate. Best training snapshot, validation excluded from selection. Resident numerical parameters/gradients/moments/inputs/targets; O(rows) norms downloaded each step and final parameters downloaded. Numeric plan excludes product conversion/library/allocator/context/register/spill scratch. Final decoded f64 measurement and acceptance are separate."
            },
        },
    })
}

/// A separately labelled proposal measurement, including after standalone f32 decoding.
pub fn measure(
    d: &Device,
    source: &OperatorProgram,
    x: &Array2<f64>,
    y: &Array2<f64>,
    numeric_bytes: usize,
    forward_rows: usize,
) -> Result<f64, String> {
    if source.declarations.slots.len() != 1 {
        return Err("legacy measure requires one Raw slot; use measure_grouped".into());
    }
    Ok(measure_grouped(
        d,
        source,
        std::slice::from_ref(x),
        y,
        &[OutputGroup {
            label: "output".into(),
            start: 0,
            end: y.ncols(),
        }],
        numeric_bytes,
        forward_rows,
    )?
    .maximum)
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
    fn batch_combined_gradient_escapes_tied_convex_single_active_stall() {
        use crate::operator_program::Provenance;
        let vector = Interface::native(2).expect("vector");
        let scalar = Interface::constant();
        let dense = |name, rows, cols, values: Array2<f64>| {
            Arc::new(
                Operator::dense(
                    name,
                    rows,
                    cols,
                    values.clone(),
                    exact_precision(values.iter().copied()).expect("precision"),
                    Provenance::default(),
                )
                .expect("dense"),
            )
        };
        let program = OperatorProgram {
            declarations: Declarations {
                parameters: 0,
                domains: vec![],
                slots: vec![Slot::Raw { width: 1 }],
            },
            bases: vec![],
            rules: vec![],
            operators: vec![
                dense(
                    "a,b",
                    vector.clone(),
                    scalar.clone(),
                    ndarray::array![[0.], [0.]],
                ),
                dense(
                    "signed map",
                    vector.clone(),
                    vector.clone(),
                    ndarray::array![[1., 0.], [-1., 1.]],
                ),
                dense("fixed offset", vector, scalar, ndarray::array![[2.], [2.]]),
            ],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Constant { operator: 0 },
                Node::Affine {
                    terms: vec![(1, 1)],
                    bias: Some(2),
                },
            ],
            output: 2,
        };
        let x = ndarray::array![[1.]];
        let y = ndarray::array![[1., 1.]];
        let mut settings = settings();
        settings.iterations = 1000;
        settings.learning_rate = 0.03;
        settings.backtracking = Some(Backtracking {
            factor: 0.5,
            max_trials: 16,
        });
        let legacy = fit_grouped(
            &Device::host(),
            &program,
            std::slice::from_ref(&x),
            &y,
            std::slice::from_ref(&x),
            &y,
            &groups(),
            &[0],
            settings.clone(),
        )
        .expect("legacy stall");
        assert_eq!(legacy.report.accepted_steps, 0);
        assert_eq!(legacy.report.best_training_max, 1.);
        settings.backtracking = None;
        let batch = fit_grouped_batched(
            &Device::host(),
            &program,
            std::slice::from_ref(&x),
            &y,
            std::slice::from_ref(&x),
            &y,
            &groups(),
            &[0],
            settings,
            BatchSchedule {
                ordinary_rows: 1,
                hard_rows: 1,
                scan_every: 10,
                temperature: 0.1,
            },
        )
        .expect("combined fit");
        assert!(
            batch.report.best_training_max < 1e-3,
            "{}",
            batch.report.best_training_max
        );
        assert_eq!(batch.report.stop_reason, "zero_training_maximum");
        assert_eq!(
            batch.report.training_row_work.proposal_backward_rows,
            batch.report.accepted_steps
        );
        let final_step = batch
            .report
            .iterations
            .last()
            .expect("final complete scan")
            .step;
        assert!(final_step <= 1000);
        assert_eq!(
            batch.report.training_row_work.full_training_scans,
            final_step / 10 + 1
        );
    }
    #[test]
    fn batch_realizable_generated_interior_teacher_fits_full_maps_and_shared_body() {
        use crate::composed_rule_search::{Expr, Unary, UseSpec, compile};
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
        };
        assert!(
            fit_grouped_batched(
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
            .is_err()
        );
        let good = BatchSchedule {
            ordinary_rows: 1,
            hard_rows: 1,
            scan_every: 2,
            temperature: 0.1,
        };
        let zero = Array2::zeros(y.dim());
        assert!(
            fit_grouped_batched(
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
            .is_err()
        );
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
        assert!(
            fit_grouped_batched(
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
            .is_err()
        );
        let mut old = settings();
        old.backtracking = Some(Backtracking {
            factor: 0.5,
            max_trials: 2,
        });
        assert!(
            fit_grouped_batched(
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
            .is_err()
        );
        let weights = smooth_weights(&[vec![1., 1.], vec![1., 1.]], 0.1).expect("stable tie");
        assert_eq!(weights, vec![vec![0.25, 0.25]; 2]);
        let huge = smooth_weights(&[vec![f64::MAX]], 0.1);
        assert!(huge.is_err());
    }
    #[test]
    fn backtracking_corrects_overshoot_and_stops_without_committing_rejected_step() {
        let p = model(0.6, 0.1);
        let teacher = model(1.2, -0.2);
        let x = ndarray::array![[-1.2], [-0.3], [0.4], [1.1], [1.8]];
        let y = target(&teacher, &x);
        let d = Device::host();
        let groups = vec![OutputGroup {
            label: "all".into(),
            start: 0,
            end: y.ncols(),
        }];
        let run = |trials| {
            let mut s = settings();
            s.iterations = 1;
            s.learning_rate = 100.;
            s.backtracking = Some(Backtracking {
                factor: 0.5,
                max_trials: trials,
            });
            fit_grouped(
                &d,
                &p,
                std::slice::from_ref(&x),
                &y,
                std::slice::from_ref(&x),
                &y,
                &groups,
                &[0, 1],
                s,
            )
            .expect("bounded trial fit")
        };
        let rejected = run(1);
        assert_eq!(rejected.report.accepted_steps, 0);
        assert_eq!(rejected.report.trial_training_evaluations, 1);
        assert_eq!(
            rejected.report.stop_reason,
            "no_strict_training_decrease_in_declared_trials"
        );
        assert_eq!(target(&rejected.program, &x), target(&p, &x));
        let corrected = run(16);
        assert_eq!(corrected.report.accepted_steps, 1);
        assert!(corrected.report.trial_training_evaluations > 1);
        assert!(corrected.report.best_training_max < corrected.report.initial_training_max);
        assert!(
            corrected
                .report
                .backtracking_trials
                .last()
                .expect("accepted trial")
                .accepted
        );
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
    fn proposal_product_arithmetic_is_explicit_and_measurement_remains_f64() {
        let p = model(0.6, 0.1);
        let teacher = model(1.2, -0.2);
        let x = ndarray::array![[-1.2], [-0.3], [0.4], [1.1], [1.8]];
        let v = ndarray::array![[-0.7], [0.8], [1.5]];
        let y = target(&teacher, &x);
        let vy = target(&teacher, &v);
        let d = Device::host();
        let mut s = settings();
        s.arithmetic = ProposalArithmetic::F32;
        let result = fit(&d, &p, &x, &y, &v, &vy, &[0, 1], s.clone()).unwrap();
        assert_eq!(result.report.settings.arithmetic, ProposalArithmetic::F32);
        assert!(result.report.best_training_max < result.report.initial_training_max * 0.1);
        // Final f64 scoring must use decoded numerical parameters, never cached f32 fit scores.
        let fitted_y = target(&result.program, &x);
        let scale = panel(&x, &y, 1, 2).unwrap();
        let expected = fitted_y
            .outer_iter()
            .zip(y.outer_iter())
            .map(|(a, b)| {
                a.iter()
                    .zip(b)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    / scale
            })
            .fold(0.0_f64, f64::max);
        let actual = measure(&d, &result.program, &x, &y, s.numeric_bytes, s.forward_rows).unwrap();
        assert!((actual - expected).abs() < 2e-12);
        let mut encoded = serde_json::to_value(s).unwrap();
        assert_eq!(encoded["arithmetic"], "f32");
        encoded.as_object_mut().unwrap().remove("arithmetic");
        assert_eq!(
            serde_json::from_value::<Settings>(encoded)
                .unwrap()
                .arithmetic,
            ProposalArithmetic::F64
        );
    }
    #[test]
    fn reordered_interleaved_raw_slots_fit_and_measure_the_same_program() {
        let mut ordered = model(0.6, 0.1);
        ordered.rules.clear();
        ordered.declarations.slots = vec![Slot::Raw { width: 1 }, Slot::Raw { width: 1 }];
        ordered.nodes = vec![
            Node::Raw { slot: 0 },
            Node::Raw { slot: 1 },
            Node::Affine {
                terms: vec![(1, 2)],
                bias: None,
            },
            Node::Affine {
                terms: vec![(0, 0), (2, 2)],
                bias: Some(1),
            },
        ];
        ordered.output = 3;
        let mut reordered = ordered.clone();
        reordered.nodes = vec![
            Node::Raw { slot: 1 },
            Node::Affine {
                terms: vec![(0, 2)],
                bias: None,
            },
            Node::Raw { slot: 0 },
            Node::Affine {
                terms: vec![(2, 0), (1, 2)],
                bias: Some(1),
            },
        ];
        let inputs = vec![
            ndarray::array![[1.], [-2.], [3.]],
            ndarray::array![[4.], [1.], [-1.]],
        ];
        let target = ndarray::array![[10.], [-1.], [2.]];
        let groups = [OutputGroup {
            label: "output".into(),
            start: 0,
            end: 1,
        }];
        let d = Device::host();
        let measured = measure_grouped(&d, &ordered, &inputs, &target, &groups, 1 << 20, 2)
            .expect("ordered measure");
        let other = measure_grouped(&d, &reordered, &inputs, &target, &groups, 1 << 20, 2)
            .expect("reordered measure");
        assert_eq!(measured.maximum, other.maximum);
        assert_eq!(measured.worst_row, other.worst_row);
        let mut config = settings();
        config.iterations = 4;
        config.forward_rows = 2;
        let a = fit_grouped(
            &d,
            &ordered,
            &inputs,
            &target,
            &inputs,
            &target,
            &groups,
            &[0, 1],
            config.clone(),
        )
        .expect("ordered fit");
        let b = fit_grouped(
            &d,
            &reordered,
            &inputs,
            &target,
            &inputs,
            &target,
            &groups,
            &[0, 1],
            config,
        )
        .expect("reordered fit");
        assert_eq!(a.report.initial_training_max, b.report.initial_training_max);
        assert_eq!(a.report.best_training_max, b.report.best_training_max);
        for index in [0, 1] {
            assert_eq!(
                a.program.operators[index].matrix(),
                b.program.operators[index].matrix()
            );
        }
    }

    #[test]
    fn shared_nonlinear_parameters_fit_joint_output_max_error_without_validation_selection() {
        let p = model(0.6, 0.1);
        let teacher = model(1.2, -0.2);
        let x = ndarray::array![[-1.2], [-0.3], [0.4], [1.1], [1.8]];
        let v = ndarray::array![[-0.7], [0.8], [1.5]];
        let y = target(&teacher, &x);
        let vy = target(&teacher, &v);
        let d = Device::host();
        let result = fit(&d, &p, &x, &y, &v, &vy, &[0, 1], settings()).expect("resident fit");
        assert!(result.report.best_training_max < result.report.initial_training_max * 0.1);
        assert!(result.report.final_validation_max < result.report.initial_validation_max * 0.2);
        assert_eq!(
            result.report.best_training_max,
            result
                .report
                .iterations
                .iter()
                .map(|r| r.training_max)
                .fold(f64::INFINITY, f64::min)
        );
        assert_eq!(
            p.operators[0].matrix(),
            Array2::from_elem((1, 1), 0.6),
            "input program immutable"
        );
        // Validation values never choose updates or the retained training snapshot.
        let altered = &vy + 7.;
        let other =
            fit(&d, &p, &x, &y, &v, &altered, &[0, 1], settings()).expect("different validation");
        assert_eq!(
            result.program.operators[0].matrix(),
            other.program.operators[0].matrix()
        );
        assert_eq!(
            result.program.operators[1].matrix(),
            other.program.operators[1].matrix()
        );
        let artifact = crate::artifact::Artifact::native(&result.program)
            .expect("artifact")
            .f32_literals()
            .expect("f32");
        let decoded = crate::artifact::Artifact::from_bytes(
            &artifact.to_bytes().expect("saved bytes"),
            &artifact.program.declarations,
        )
        .expect("independent saved replay");
        assert!(
            measure(&d, &decoded.program, &v, &vy, 1 << 20, 3).expect("decoded measurement") < 0.05
        );
    }
    #[test]
    fn invalid_budget_zero_native_scale_and_sparse_parameters_are_explicit_errors() {
        let mut p = model(0.6, 0.1);
        let x = ndarray::array![[1.], [2.]];
        let y = target(&model(1.2, -0.2), &x);
        let d = Device::host();
        let mut s = settings();
        s.numeric_bytes = 1;
        assert!(fit(&d, &p, &x, &y, &x, &y, &[0, 1], s).is_err());
        assert!(
            fit(
                &d,
                &p,
                &x,
                &Array2::zeros((2, 2)),
                &x,
                &y,
                &[0, 1],
                settings()
            )
            .is_err()
        );
        assert!(fit(&d, &p, &x, &y, &x, &y, &[0, 0], settings()).is_err());
        let OperatorBody::Dense { present, .. } = &mut Arc::make_mut(&mut p.operators[0]).body
        else {
            panic!("dense")
        };
        present.fill(false);
        assert!(fit(&d, &p, &x, &y, &x, &y, &[0], settings()).is_err());
    }
    #[test]
    fn resident_input_is_bound_by_slot_when_raw_is_not_first_node() {
        let mut p = model(0.6, 0.1);
        p.nodes = vec![
            Node::Constant { operator: 1 },
            Node::Raw { slot: 0 },
            Node::Affine {
                terms: vec![(1, 2)],
                bias: None,
            },
            Node::Call {
                rule: 0,
                arguments: vec![1],
            },
            Node::Call {
                rule: 0,
                arguments: vec![2],
            },
            Node::Concat { parts: vec![3, 4] },
        ];
        p.output = 5;
        let x = ndarray::array![[-0.3], [0.7], [1.3]];
        let y = target(&model(1.2, -0.2), &x);
        let result = fit(&Device::host(), &p, &x, &y, &x, &y, &[0, 1], settings())
            .expect("nonzero raw root index");
        assert!(result.report.best_training_max < result.report.initial_training_max * 0.2);
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
    fn grouped_model(weight: f64, offset: f64, tiny_scale: f64) -> OperatorProgram {
        let mut p = model(weight, offset);
        p.declarations.slots.push(Slot::Raw { width: 1 });
        let i = Interface::native(1).expect("interface");
        for (name, value) in [("large writer", 1000.), ("tiny writer", tiny_scale)] {
            p.operators.push(Arc::new(
                Operator::dense(
                    name,
                    i.clone(),
                    i.clone(),
                    Array2::from_elem((1, 1), value),
                    exact_precision([value]).expect("precision"),
                    Default::default(),
                )
                .expect("writer"),
            ));
        }
        p.nodes = vec![
            Node::Raw { slot: 0 },
            Node::Raw { slot: 1 },
            Node::Call {
                rule: 0,
                arguments: vec![0],
            },
            Node::Call {
                rule: 0,
                arguments: vec![1],
            },
            Node::Affine {
                terms: vec![(2, 3)],
                bias: None,
            },
            Node::Affine {
                terms: vec![(3, 4)],
                bias: None,
            },
            Node::Concat { parts: vec![4, 5] },
        ];
        p.output = 6;
        p
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
    #[test]
    fn grouped_normalization_cannot_hide_small_use_and_frozen_body_is_preserved() {
        let p = grouped_model(0.6, 0.1, 0.0005);
        let teacher = grouped_model(0.6, 0.1, 0.001);
        let x = vec![ndarray::array![[1.], [2.]], ndarray::array![[1.], [2.]]];
        let y = grouped_target(&teacher, &x);
        let d = Device::host();
        let score =
            measure_grouped(&d, &p, &x, &y, &groups(), 1 << 20, 1).expect("group measurement");
        assert_eq!(score.group_maxima[0], 0.);
        assert!(score.maximum > 0.5);
        assert_eq!(score.worst_group, 1);
        let predicted = grouped_target(&p, &x);
        let native_pooled = (y.iter().map(|v| v * v).sum::<f64>() / 2.).sqrt();
        let pooled = predicted
            .outer_iter()
            .zip(y.outer_iter())
            .map(|(a, b)| {
                a.iter()
                    .zip(b)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt()
                    / native_pooled
            })
            .fold(0f64, f64::max);
        assert!(pooled < 1e-5, "pooled normalization would hide this error");
        let mut s = settings();
        s.learning_rate = 0.00002;
        let result = fit_grouped(&d, &p, &x, &y, &x, &y, &groups(), &[4], s)
            .expect("fit only paid tiny writer");
        assert!(result.report.best_training_max < score.maximum * 0.05);
        assert!(Arc::ptr_eq(&result.program.operators[0], &p.operators[0]));
        assert!(
            Arc::ptr_eq(&result.program.operators[1], &p.operators[1]),
            "shared body coefficients frozen"
        );
        assert!(
            result.report.training_scales[0].native_rms
                / result.report.training_scales[1].native_rms
                > 999999.
        );
        let saved = crate::artifact::Artifact::native(&result.program)
            .expect("artifact")
            .f32_literals()
            .expect("f32");
        let replay = crate::artifact::Artifact::from_bytes(
            &saved.to_bytes().expect("saved bytes"),
            &saved.program.declarations,
        )
        .expect("ordinary decode");
        let actual = measure_grouped(&d, &replay.program, &x, &y, &groups(), 1 << 20, 1)
            .expect("saved grouped replay");
        assert!(actual.maximum < 0.03);
        let gap = [OutputGroup {
            label: "incomplete".into(),
            start: 0,
            end: 1,
        }];
        assert!(measure_grouped(&d, &p, &x, &y, &gap, 1 << 20, 1).is_err());
        let mut misaligned = x.clone();
        misaligned[1] = Array2::zeros((1, 1));
        assert!(measure_grouped(&d, &p, &misaligned, &y, &groups(), 1 << 20, 1).is_err());
    }
    #[test]
    fn grouped_shared_nonlinear_body_fits_distinct_raw_bindings_jointly() {
        let p = grouped_model(0.6, 0.1, 0.001);
        let teacher = grouped_model(1.2, -0.2, 0.001);
        let x = vec![
            ndarray::array![[-1.2], [0.4], [1.8]],
            ndarray::array![[-0.7], [0.8], [1.5]],
        ];
        let y = grouped_target(&teacher, &x);
        let result = fit_grouped(
            &Device::host(),
            &p,
            &x,
            &y,
            &x,
            &y,
            &groups(),
            &[0, 1],
            settings(),
        )
        .expect("joint shared parameters");
        assert!(result.report.best_training_max < result.report.initial_training_max * 0.1);
        assert!(result.report.best_training_groups.iter().all(|v| *v < 0.05));
        assert!(result.report.iterations.iter().any(|i| i.worst_group == 1));
        assert_ne!(
            result.program.operators[0].matrix(),
            p.operators[0].matrix()
        );
    }
}

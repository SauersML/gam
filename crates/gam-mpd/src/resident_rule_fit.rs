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
        Ok(result)
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
    let started = Instant::now();
    validate_settings(&settings)?;
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
        .min(train_y.nrows().max(valid_y.nrows()));
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
        trace_bytes.checked_mul(3).ok_or("trace plan overflow")?,
        row_output.checked_mul(4).ok_or("row plan overflow")?,
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
    for step in 0..=settings.iterations {
        let score = training.scan(&program, settings.forward_rows)?;
        let (row, group, value) = (score.worst_row, score.worst_group, score.maximum);
        history.push(GroupIteration {
            step,
            training_max: value,
            worst_row: row,
            worst_group: group,
            active_recomputed_error: None,
        });
        if value < best_value {
            best_value = value;
            best_step = step;
            best_training_groups = score.group_maxima;
            best = snapshot(&program, trainable)?;
        }
        if step == settings.iterations || value == 0. {
            if value == 0. {
                stop_reason = "zero_training_maximum";
            }
            break;
        }
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
            scope: "Proposal optimization only, fixed finite f64 input/output rows. Explicit settings.arithmetic governs forward and backward products, including training/validation proposal scores; parameter/moment storage and other primitives remain f64. The f32 option uses rounded-product surrogate gradients, not derivatives through rounding. Maximum over ALL output groups and rows, each group normalized by its own complete native-family RMS. Optional declared finite backtracking scans the SAME full training maximum and accepts only strict decrease; proposed moments and Adam time commit only with an accepted step, exhaustion restores old parameters and stops. Default uses original fixed-step Adam. Deterministic active-group/row max-norm Adam; full row scan in declared chunks and one-row reverse. Single-row GEMM rounding may differ from batch; recomputed error recorded; nonconvex, no optimum certificate. Best training snapshot, validation excluded from selection. Resident numerical parameters/gradients/moments/inputs/targets; O(rows) norms downloaded each step and final parameters downloaded. Numeric plan excludes product conversion/library/allocator/context/register/spill scratch. Final decoded f64 measurement and acceptance are separate.",
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

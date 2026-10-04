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
        match self { Self::F64 => Arithmetic::F64, Self::F32 => Arithmetic::F32 }
    }
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
    Ok(())
}
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
        * (y.iter().map(|v| (v / largest) * (v / largest)).sum::<f64>() / y.nrows() as f64).sqrt();
    if !rms.is_finite() || rms <= 0. {
        return Err("nonfinite/underflow native RMS".into());
    }
    Ok(rms)
}
struct Panel {
    family: FamilyInputs,
    input: Tensor,
    target: Tensor,
    rms: f64,
}
impl Panel {
    fn new(d: &Device, x: &Array2<f64>, y: &Array2<f64>, rms: f64) -> Result<Self, String> {
        Ok(Self {
            family: FamilyInputs {
                rows: x.nrows(),
                slots: vec![SlotValues::Raw(Array2::zeros((0, x.ncols())))],
                layout: None,
            },
            input: d.upload(x.view()).map_err(|e| e.to_string())?,
            target: d.upload(y.view()).map_err(|e| e.to_string())?,
            rms,
        })
    }
    fn evaluate_rows(
        &self,
        p: &DeviceProgram,
        start: usize,
        rows: usize,
    ) -> Result<(DeviceTrace, Tensor, Vec<f64>), String> {
        let d = p.device();
        let trace = p.forward_given(
            &FamilyInputs {
                rows,
                slots: self.family.slots.clone(),
                layout: None,
            },
            BTreeMap::from([(
                0,
                d.rows_of(&self.input, start, rows)
                    .map_err(|e| e.to_string())?,
            )]),
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
        let norms = d
            .scaled_row_l2(&residual, 0..residual.cols(), self.rms)
            .map_err(|e| e.to_string())?;
        if norms.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite proposal objective".into());
        }
        Ok((trace, residual, norms))
    }
    fn maximum(&self, p: &DeviceProgram, forward_rows: usize) -> Result<(usize, f64), String> {
        let mut worst = (0, 0.0);
        for start in (0..self.family.rows).step_by(forward_rows) {
            let rows = forward_rows.min(self.family.rows - start);
            let (_, _, norms) = self.evaluate_rows(p, start, rows)?;
            let (row, value) = maximum(&norms);
            if value > worst.1 {
                worst = (start + row, value);
            }
        }
        Ok(worst)
    }
}
fn maximum(values: &[f64]) -> (usize, f64) {
    values.iter().copied().enumerate().fold(
        (0, values[0]),
        |best, (i, v)| if v > best.1 { (i, v) } else { best },
    )
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
    let started = Instant::now();
    validate_settings(&settings)?;
    let input = match source.declarations.slots.as_slice() {
        [Slot::Raw { width }]
            if source.declarations.parameters == 0 && source.declarations.domains.is_empty() =>
        {
            *width
        }
        _ => {
            return Err(
                "fitter requires one Raw input slot and no external parameters/domains".into(),
            );
        }
    };
    let (expanded, _) = mapped_inlined(source)?;
    if expanded
        .nodes
        .iter()
        .filter(|n| matches!(n, crate::operator_program::Node::Raw { .. }))
        .count()
        != 1
    {
        return Err("fitter requires exactly one raw input node".into());
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
    let interfaces = expanded.interfaces().map_err(|e| e.to_string())?;
    let output = interfaces[expanded.output].width();
    let training_rms = panel(train_x, train_y, input, output)?;
    let validation_rms = panel(valid_x, valid_y, input, output)?;
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
    let panels = checked_bytes(
        train_x
            .len()
            .checked_add(train_y.len())
            .and_then(|a| a.checked_add(valid_x.len()))
            .and_then(|a| a.checked_add(valid_y.len()))
            .ok_or("panel size overflow")?,
    )?;
    let max_rows = settings
        .forward_rows
        .min(train_x.nrows().max(valid_x.nrows()));
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
            .checked_mul(10)
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
    let training = Panel::new(d, train_x, train_y, training_rms)?;
    let validation = Panel::new(d, valid_x, valid_y, validation_rms)?;
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
    let initial_validation_max = validation.maximum(&program, settings.forward_rows)?.1;
    let mut best = snapshot(&program, trainable)?;
    let mut best_value = f64::INFINITY;
    let mut best_step = 0;
    let mut history = Vec::new();
    for step in 0..=settings.iterations {
        let (row, value) = training.maximum(&program, settings.forward_rows)?;
        history.push(Iteration {
            step,
            training_max: value,
            worst_row: row,
            active_recomputed_error: None,
        });
        if value < best_value {
            best_value = value;
            best_step = step;
            best = snapshot(&program, trainable)?;
        }
        if step == settings.iterations || value == 0. {
            break;
        }
        // An active row is a subgradient of max row norm; deterministic first-row tie.
        // Normalize in two stages to avoid squaring the native RMS.
        let (trace, residual, active_norms) = training.evaluate_rows(&program, row, 1)?;
        history
            .last_mut()
            .ok_or("missing current iteration")?
            .active_recomputed_error = Some(active_norms[0]);
        let norm = active_norms[0] * training_rms;
        if !norm.is_finite() || norm <= 0. {
            return Err("active residual norm cannot be represented".into());
        }
        let active = residual;
        let mut normalized = d.zeros(1, output).map_err(|e| e.to_string())?;
        d.axpy(&mut normalized, 1. / norm, &active)
            .map_err(|e| e.to_string())?;
        let mut row_seed = d.zeros(1, output).map_err(|e| e.to_string())?;
        d.axpy(&mut row_seed, 1. / training_rms, &normalized)
            .map_err(|e| e.to_string())?;
        let seed = row_seed;
        let (_, gradients) = program.vjp_values_dense(
            &trace,
            BTreeMap::from([(program.hidden(), seed)]),
            &[],
            trainable,
            settings.arithmetic.device(),
        )?;
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
    }
    for (index, value) in best {
        program.replace_dense_parameter(index, value)?;
    }
    let final_validation_max = validation.maximum(&program, settings.forward_rows)?.1;
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
    Ok(Fit {
        program: fitted,
        report: Report {
            settings,
            trainable: trainable.to_vec(),
            training_rows: train_x.nrows(),
            validation_rows: valid_x.nrows(),
            training_rms,
            validation_rms,
            initial_training_max,
            best_training_max: best_value,
            initial_validation_max,
            final_validation_max,
            best_step,
            iterations: history,
            planned_numeric_bytes: planned,
            seconds: started.elapsed().as_secs_f64(),
            scope: "Proposal optimization only, fixed finite f64 input/output rows. Explicit settings.arithmetic governs forward and backward products, including training/validation proposal scores; parameter/moment storage and other primitives remain f64. The f32 option uses rounded-product surrogate gradients, not derivatives through rounding. Deterministic active-row max-norm Adam; full row scan in declared chunks and one-row reverse. Single-row GEMM rounding may differ from batch; recomputed error recorded; nonconvex, no optimum certificate. Best training snapshot, validation excluded from selection. Resident numerical parameters/gradients/moments/inputs/targets; O(rows) norms downloaded each step and final parameters downloaded. Numeric plan excludes product conversion/library/allocator/context/register/spill scratch. Final decoded f64 measurement and acceptance are separate.",
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
    if forward_rows == 0 {
        return Err("zero measurement forward_rows".into());
    }
    let (expanded, _) = mapped_inlined(source)?;
    let interfaces = expanded.interfaces().map_err(|e| e.to_string())?;
    let input = match expanded.declarations.slots.as_slice() {
        [Slot::Raw { width }] => *width,
        _ => return Err("measure requires one raw input".into()),
    };
    let rms = panel(x, y, input, interfaces[expanded.output].width())?;
    let p = DeviceProgram::compile_values_bounded(d, &expanded, numeric_bytes)?;
    let planned = sum_bytes(&[
        p.operator_numeric_bytes()?,
        checked_bytes(x.len().checked_add(y.len()).ok_or("panel overflow")?)?,
        p.bytes_per_row()
            .checked_mul(forward_rows.min(x.nrows()))
            .ok_or("trace overflow")?,
        checked_bytes(y.len())?
            .checked_mul(2)
            .ok_or("residual overflow")?,
    ])?;
    if planned > numeric_bytes {
        return Err("measurement numeric plan exceeds budget".into());
    }
    Ok(Panel::new(d, x, y, rms)?.maximum(&p, forward_rows)?.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_program::{Declarations, Interface, Law, Node, Operator, Rule};
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
        }
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
        let mut s = settings(); s.arithmetic = ProposalArithmetic::F32;
        let result = fit(&d, &p, &x, &y, &v, &vy, &[0, 1], s.clone()).unwrap();
        assert_eq!(result.report.settings.arithmetic, ProposalArithmetic::F32);
        assert!(result.report.best_training_max < result.report.initial_training_max * 0.1);
        // Final f64 scoring must use decoded numerical parameters, never cached f32 fit scores.
        let fitted_y = target(&result.program, &x);
        let scale = panel(&x, &y, 1, 2).unwrap();
        let expected = fitted_y.outer_iter().zip(y.outer_iter()).map(|(a,b)|
            a.iter().zip(b).map(|(a,b)| (a-b).powi(2)).sum::<f64>().sqrt()/scale
        ).fold(0.0_f64, f64::max);
        let actual = measure(&d, &result.program, &x, &y, s.numeric_bytes, s.forward_rows).unwrap();
        assert!((actual - expected).abs() < 2e-12);
        let mut encoded = serde_json::to_value(s).unwrap();
        assert_eq!(encoded["arithmetic"], "f32");
        encoded.as_object_mut().unwrap().remove("arithmetic");
        assert_eq!(serde_json::from_value::<Settings>(encoded).unwrap().arithmetic, ProposalArithmetic::F64);
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
}

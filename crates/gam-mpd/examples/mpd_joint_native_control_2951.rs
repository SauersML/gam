//! Supplied native nonlinear capacity control, not discovered computation.
//! EXPORT SETTINGS.json FRESH_OUT host|cuda. All four MLPs, fixed native parents.
use gam_gpu::{tensor::Device, GpuPolicy};
use gam_mpd::{
    acceptance::{structural_cost, CostCache},
    artifact::Artifact,
    coder_capture::sha256,
    device_program::DeviceProgram,
    down_edit_family::{self, Direction},
    import::import_language_model,
    operator_program::{
        exact_precision, Declarations, FamilyInputs, Interface, Node, Operator, OperatorBody,
        OperatorProgram, Slot, SlotValues,
    },
    parameter_response_program,
    resident_rule_fit::{self, BatchSchedule, OutputGroup},
    run_check::{layer_nodes, split_sites},
};
use ndarray::{Array1, Array2, Axis};
use rand::{rngs::StdRng, RngExt, SeedableRng};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{path::Path, sync::Arc, time::Instant};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    export_sha256: String,
    sequences: usize,
    context: usize,
    max_rows: usize,
    numeric_bytes: usize,
    absolute_tolerance: f64,
    directions: Vec<DeclaredDirection>,
    amplitudes: Vec<Vec<f64>>,
    #[serde(default)]
    fitting: Option<Fitting>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fitting {
    training_sequences: usize,
    seed: u64,
    relative_perturbation: f64,
    optimizer: resident_rule_fit::Settings,
    batch: BatchSchedule,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeclaredDirection {
    output: Vec<f64>,
    hidden: Vec<f64>,
}
fn save(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
fn dense(
    name: &str,
    rows: Interface,
    cols: Interface,
    values: Array2<f64>,
) -> Result<Arc<Operator>, String> {
    let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
    Ok(Arc::new(
        Operator::dense(name, rows, cols, values, precision, Default::default())
            .map_err(|e| e.to_string())?,
    ))
}
fn native_body(
    source: &OperatorProgram,
    read: usize,
    pre: usize,
    active: usize,
    write: usize,
    directions: &[Direction],
) -> Result<(OperatorProgram, Vec<usize>), String> {
    let types = source.interfaces().map_err(|e| e.to_string())?;
    let (reader, reader_bias) = match &source.nodes[pre] {
        Node::Affine { terms, bias } if terms.len() == 1 && terms[0].0 == read => {
            (terms[0].1, *bias)
        }
        _ => return Err("capacity control needs direct native affine reader".into()),
    };
    let laws = match &source.nodes[active] {
        Node::Pointwise { input, laws } if *input == pre => laws.clone(),
        _ => return Err("capacity control needs actual native pointwise activation".into()),
    };
    let (writer, writer_bias) = match &source.nodes[write] {
        Node::Affine { terms, bias } if terms.len() == 1 && terms[0].0 == active => {
            (terms[0].1, *bias)
        }
        _ => return Err("capacity control needs direct native affine writer".into()),
    };
    let mut reader_op = (*source.operators[reader]).clone();
    if !matches!(&reader_op.body,OperatorBody::Dense{present,..} if present.iter().all(|p|*p)) {
        return Err("native reader must be complete Dense".into());
    }
    // Explicit Raw boundary has flat grouping. Same coefficients, no adapter.
    reader_op.cols = Interface::native(types[read].width()).map_err(|e| e.to_string())?;
    if let OperatorBody::Dense { present, .. } = &mut reader_op.body {
        *present = Array2::from_elem(
            (reader_op.rows.group_count(), reader_op.cols.group_count()),
            true,
        );
    }
    let mut operators = vec![Arc::new(reader_op), source.operators[writer].clone()];
    let mut held_bias = |bias: Option<usize>| -> Option<usize> {
        bias.map(|old| {
            let id = operators.len();
            operators.push(source.operators[old].clone());
            id
        })
    };
    let rb = held_bias(reader_bias);
    let wb = held_bias(writer_bias);
    let mut nodes = vec![
        Node::Raw { slot: 0 },
        Node::Affine {
            terms: vec![(0, 0)],
            bias: rb,
        },
        Node::Pointwise { input: 1, laws },
        Node::Affine {
            terms: vec![(2, 1)],
            bias: wb,
        },
    ];
    let mut response_nodes = Vec::new();
    for (i, direction) in directions.iter().enumerate() {
        if direction.hidden.len() != types[active].width()
            || direction.output.len() != types[write].width()
        {
            return Err("declared direction width differs from actual native interface".into());
        }
        let scalar = Interface::native(1).map_err(|e| e.to_string())?;
        let op = operators.len();
        operators.push(dense(
            &format!("supplied scalar direction {i}"),
            scalar,
            types[active].clone(),
            Array2::from_shape_vec((1, direction.hidden.len()), direction.hidden.to_vec())
                .map_err(|e| e.to_string())?,
        )?);
        response_nodes.push(nodes.len());
        nodes.push(Node::Affine {
            terms: vec![(2, op)],
            bias: None,
        });
    }
    let program = OperatorProgram {
        declarations: Declarations {
            domains: vec![],
            slots: vec![Slot::Raw {
                width: types[read].width(),
            }],
            parameters: 0,
        },
        bases: vec![],
        rules: vec![],
        operators,
        nodes,
        output: 3,
    };
    program.interfaces().map_err(|e| e.to_string())?;
    Ok((program, response_nodes))
}
fn maximum_error(a: &Array2<f64>, b: &Array2<f64>) -> Result<f64, String> {
    if a.dim() != b.dim() || a.iter().chain(b.iter()).any(|x| !x.is_finite()) {
        return Err("nonfinite or mismatched capacity output".into());
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max))
}
fn sequence_split(family: &FamilyInputs, count: usize) -> Result<(Vec<usize>, Vec<usize>), String> {
    let layout = family
        .layout
        .as_ref()
        .ok_or("fitting requires sequence layout")?;
    if layout.sequence.len() != family.rows || layout.position.len() != family.rows {
        return Err("sequence layout does not cover native rows".into());
    }
    let sequences = layout
        .sequence
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if count == 0 || count >= sequences.len() {
        return Err(
            "training_sequences must leave complete nonempty training and validation sequences"
                .into(),
        );
    }
    let train = sequences
        .into_iter()
        .take(count)
        .collect::<std::collections::BTreeSet<_>>();
    Ok((0..family.rows).partition(|row| train.contains(&layout.sequence[*row])))
}
fn initialized(
    native: &OperatorProgram,
    trainable: &[usize],
    seed: u64,
    perturbation: Option<f64>,
) -> Result<OperatorProgram, String> {
    let mut candidate = native.clone();
    let mut rng = StdRng::seed_from_u64(seed);
    for &id in trainable {
        let mut op = (*candidate.operators[id]).clone();
        let OperatorBody::Dense {
            values, precision, ..
        } = &mut op.body
        else {
            return Err("initialization requires dense native parameters".into());
        };
        let rms = (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt();
        let scale = perturbation.map_or((values.ncols() as f64).sqrt().recip(), |p| {
            p * if rms > 0. {
                rms
            } else {
                (values.ncols() as f64).sqrt().recip()
            }
        });
        for value in values.iter_mut() {
            let noise = rng.random_range(-3f64.sqrt()..3f64.sqrt());
            *value = if perturbation.is_some() {
                *value + scale * noise
            } else if id >= 2 {
                0.
            } else {
                scale * noise
            };
        }
        *precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
        candidate.operators[id] = Arc::new(op);
    }
    Ok(candidate)
}
// Frozen TRAIN RMS denominators; validation cannot redefine the metric or choose a snapshot.
fn frozen_errors(
    device: &Device,
    program: &OperatorProgram,
    x: &Array2<f64>,
    y: &Array2<f64>,
    scales: &[resident_rule_fit::GroupScale],
    settings: &resident_rule_fit::Settings,
) -> Result<Vec<f64>, String> {
    let resident = DeviceProgram::compile_values_bounded(device, program, settings.numeric_bytes)?;
    let mut maxima = vec![0f64; scales.len()];
    for start in (0..x.nrows()).step_by(settings.forward_rows) {
        let end = (start + settings.forward_rows).min(x.nrows());
        let inputs = FamilyInputs {
            rows: end - start,
            layout: None,
            slots: vec![SlotValues::Raw(
                x.slice(ndarray::s![start..end, ..]).to_owned(),
            )],
        };
        let trace = resident.forward(&inputs)?;
        let prediction = device
            .download(trace.value(program.output)?)
            .map_err(|e| e.to_string())?;
        for (i, scale) in scales.iter().enumerate() {
            for row in 0..prediction.nrows() {
                let norm = (scale.group.start..scale.group.end)
                    .map(|col| (prediction[(row, col)] - y[(start + row, col)]).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    / scale.native_rms;
                if !norm.is_finite() {
                    return Err("nonfinite frozen-scale diagnostic".into());
                }
                maxima[i] = maxima[i].max(norm);
            }
        }
    }
    Ok(maxima)
}
fn fit_diagnostic(
    device: &Device,
    function: &OperatorProgram,
    responses: &[usize],
    parent: &Array2<f64>,
    train: &[usize],
    valid: &[usize],
    settings: &Fitting,
    out: &Path,
) -> Result<Value, String> {
    let mut teacher = function.clone();
    let width = teacher
        .node_interface(3)
        .map_err(|e| e.to_string())?
        .width();
    teacher.output = teacher.nodes.len();
    teacher.nodes.push(Node::Concat {
        parts: std::iter::once(3)
            .chain(responses.iter().copied())
            .collect(),
    });
    let resident =
        DeviceProgram::compile_values_bounded(device, &teacher, settings.optimizer.numeric_bytes)?;
    let inputs = FamilyInputs {
        rows: parent.nrows(),
        layout: None,
        slots: vec![SlotValues::Raw(parent.clone())],
    };
    let trace = resident.forward(&inputs)?;
    let target = device
        .download(trace.value(teacher.output)?)
        .map_err(|e| e.to_string())?;
    drop(trace);
    drop(resident);
    let train_x = parent.select(Axis(0), train);
    let valid_x = parent.select(Axis(0), valid);
    let train_y = target.select(Axis(0), train);
    let valid_y = target.select(Axis(0), valid);
    let mut groups = vec![OutputGroup {
        label: "clean".into(),
        start: 0,
        end: width,
    }];
    groups.extend((0..responses.len()).map(|i| OutputGroup {
        label: format!("response{i}"),
        start: width + i,
        end: width + i + 1,
    }));
    // Native reader/writer/biases only. Supplied hidden response readers stay fixed.
    let trainable = (0..teacher.operators.len() - responses.len()).collect::<Vec<_>>();
    let mut reports = Vec::new();
    for (label, perturbation) in [
        ("perturbed_native", Some(settings.relative_perturbation)),
        ("independent", None),
    ] {
        let arm_started = Instant::now();
        let candidate = initialized(&teacher, &trainable, settings.seed, perturbation)?;
        let initialization_seconds = arm_started.elapsed().as_secs_f64();
        let fit = resident_rule_fit::fit_grouped_batched(
            device,
            &candidate,
            std::slice::from_ref(&train_x),
            &train_y,
            std::slice::from_ref(&valid_x),
            &valid_y,
            &groups,
            &trainable,
            settings.optimizer.clone(),
            settings.batch.clone(),
        )?;
        let scales = &fit.report.training_scales;
        let evidence_started = Instant::now();
        let mut measurements = Vec::new();
        for (stage, program) in [("initial", &candidate), ("best", &fit.program)] {
            let bytes = Artifact::native(program)?.to_bytes()?;
            let decoded = Artifact::from_bytes(&bytes, &program.declarations)?;
            let path = out.join(format!("{label}-{stage}.artifact"));
            std::fs::write(&path, &bytes).map_err(|e| e.to_string())?;
            measurements.push(json!({"stage":stage,"artifact_sha256":sha256(&path)?,
                "training_group_maxima":frozen_errors(device,&decoded.program,&train_x,&train_y,scales,&settings.optimizer)?,
                "validation_group_maxima":frozen_errors(device,&decoded.program,&valid_x,&valid_y,scales,&settings.optimizer)?}));
        }
        reports.push(json!({"initialization":label,"initialization_seconds":initialization_seconds,
            "artifact_and_decoded_evaluation_seconds":evidence_started.elapsed().as_secs_f64(),
            "arm_seconds":arm_started.elapsed().as_secs_f64(),
            "measurements_frozen_training_scales":measurements,"optimizer_report":fit.report}));
        save(
            &out.join("FITTING.json"),
            &json!({"arms":reports,"groups":groups,
            "training_rows":train,"validation_rows":valid,"seed":settings.seed,
            "scope":"Supplied full-width native topology; supervised clean and scalar response fitting. Primary decoded errors use TRAIN RMS for both splits. Optimizer validation diagnostics use panel-specific RMS and never choose snapshots. No output KL, structure discovery, or autonomous fidelity claim."}),
        )?;
    }
    Ok(json!({"arms":reports,"training_rows":train.len(),"validation_rows":valid.len()}))
}
fn check_layer(
    source: &OperatorProgram,
    read: usize,
    pre: usize,
    active: usize,
    write: usize,
    parent: &Array2<f64>,
    native_clean: &Array2<f64>,
    directions: &[Direction],
    amplitudes: &[Vec<f64>],
    tolerance: f64,
    out: &Path,
) -> Result<Value, String> {
    let (function, responses) = native_body(source, read, pre, active, write, directions)?;
    let target = source.node_interface(write).map_err(|e| e.to_string())?;
    let output_directions = directions
        .iter()
        .enumerate()
        .map(|(i, d)| {
            dense(
                &format!("supplied output direction {i}"),
                target.clone(),
                Interface::native(1).map_err(|e| e.to_string())?,
                Array2::from_shape_vec((d.output.len(), 1), d.output.to_vec())
                    .map_err(|e| e.to_string())?,
            )
        })
        .collect::<Result<Vec<_>, String>>()?;
    let composed = parameter_response_program::compose_joint_with_outputs(
        &function,
        3,
        &responses,
        &output_directions,
        &target,
    )?;
    let artifact = Artifact::native(&composed.program)?.f32_literals()?;
    let bytes = artifact.to_bytes()?;
    let decoded = Artifact::from_bytes(&bytes, &artifact.program.declarations)?;
    if decoded.to_bytes()? != bytes {
        return Err("ordinary canonical replay mismatch".into());
    }
    std::fs::write(out.join("program.artifact"), &bytes).map_err(|e| e.to_string())?;
    let c32 = structural_cost(&decoded, &mut CostCache::default())?.total();
    let family = down_edit_family::build(source, read, write, directions)?;
    let mut reports = Vec::new();
    let mut worst = 0f64;
    for alpha in amplitudes {
        if alpha.len() != directions.len() || alpha.iter().any(|x| !x.is_finite()) {
            return Err("explicit finite amplitude vector per declared direction required".into());
        }
        let mut slots = vec![SlotValues::Raw(parent.clone())];
        slots.extend(
            alpha
                .iter()
                .map(|a| SlotValues::Raw(Array2::from_elem((parent.nrows(), 1), *a))),
        );
        let inputs = FamilyInputs {
            rows: parent.nrows(),
            slots,
            layout: None,
        };
        let trace = decoded.execute(&inputs)?;
        // Independently mutate native down weights; execute on SAME fixed parent.
        let literal = family.literal_native(alpha)?;
        let (literal_function, _) = native_body(&literal, read, pre, active, write, directions)?;
        let raw = FamilyInputs {
            rows: parent.nrows(),
            slots: vec![SlotValues::Raw(parent.clone())],
            layout: None,
        };
        let reference = literal_function
            .execute(&raw, false)
            .map_err(|e| e.to_string())?;
        let error = maximum_error(
            &trace.values[decoded.program.output],
            &reference.values[literal_function.output],
        )?;
        let clean_error = maximum_error(&trace.values[composed.clean_output.node], native_clean)?;
        let original = function.execute(&raw, false).map_err(|e| e.to_string())?;
        let response_errors = composed
            .response_nodes
            .iter()
            .zip(&responses)
            .map(|(actual, expected)| {
                maximum_error(&trace.values[*actual], &original.values[*expected])
            })
            .collect::<Result<Vec<_>, _>>()?;
        worst = worst
            .max(error)
            .max(clean_error)
            .max(response_errors.iter().copied().fold(0., f64::max));
        reports.push(json!({"amplitudes":alpha,"literal_weight_output_max_abs_error":error,
            "native_clean_output_max_abs_error":clean_error,"scalar_response_max_abs_errors":response_errors}));
    }
    let report = json!({"native_read":read,"native_pre":pre,"native_active":active,"native_write":write,
        "rows":parent.nrows(),"input_width":parent.ncols(),"hidden_width":source.node_interface(active).map_err(|e|e.to_string())?.width(),
        "output_width":target.width(),"shared_hidden_node":2,"clean_node":composed.clean_output.node,
        "response_nodes":composed.response_nodes,"artifact_sha256":sha256(&out.join("program.artifact"))?,
        "c32":c32,"saved_bytes":bytes.len(),"maximum_abs_error":worst,"declared_absolute_tolerance":tolerance,
        "passed":worst<=tolerance,"conditions":reports,
        "scope":"Supplied native weights and activation architecture; fixed native parents; exact shared nonlinear capacity control. No fitted or discovered equation, autonomous fidelity, or compression claim."});
    save(&out.join("REPORT.json"), &report)?;
    if worst > tolerance {
        return Err(format!(
            "capacity discrepancy {worst} exceeds declared tolerance {tolerance}"
        ));
    }
    Ok(report)
}
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("EXPORT SETTINGS.json FRESH_OUT host|cuda".into());
    }
    let export = Path::new(&args[0]);
    let settings_path = Path::new(&args[1]);
    let out = Path::new(&args[2]);
    let settings: Settings =
        serde_json::from_slice(&std::fs::read(settings_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if settings.sequences == 0
        || settings.context == 0
        || settings.max_rows == 0
        || settings.numeric_bytes == 0
        || !settings.absolute_tolerance.is_finite()
        || settings.absolute_tolerance < 0.
        || settings.directions.len() != 2
        || settings.amplitudes.is_empty()
    {
        return Err("positive resource limits, exactly two directions and explicit tolerance/cases required".into());
    }
    if let Some(fit) = &settings.fitting {
        if !fit.relative_perturbation.is_finite()
            || fit.relative_perturbation <= 0.
            || fit.optimizer.forward_rows == 0
            || fit.optimizer.numeric_bytes > settings.numeric_bytes
        {
            return Err("fitting needs positive finite perturbation/forward rows and budget within capture budget".into());
        }
    }
    if sha256(&export.join("export.json"))? != settings.export_sha256 {
        return Err("export hash mismatch".into());
    }
    if out.exists() {
        return Err("fresh output directory required".into());
    }
    let device = match args[3].as_str() {
        "host" => Device::host(),
        "cuda" => Device::accelerator(GpuPolicy::Required)
            .map_err(|e| e.to_string())?
            .ok_or("CUDA required")?,
        _ => return Err("host|cuda required".into()),
    };
    let start = Instant::now();
    let imported = import_language_model(export, settings.sequences, settings.context)?;
    let split = settings
        .fitting
        .as_ref()
        .map(|fit| sequence_split(&imported.contract.family, fit.training_sequences))
        .transpose()?;
    let native = split_sites(&imported.program)?;
    let layers = layer_nodes(&native, 4)?;
    if layers.len() != 4 || imported.contract.family.rows > settings.max_rows {
        return Err("four native layers and declared row budget required".into());
    }
    let directions = settings
        .directions
        .iter()
        .map(|d| Direction {
            output: Array1::from(d.output.clone()),
            hidden: Array1::from(d.hidden.clone()),
        })
        .collect::<Vec<_>>();
    let mut prefix = native.clone();
    prefix.output = layers[3].mlp;
    prefix.nodes.truncate(prefix.output + 1);
    let resident = DeviceProgram::compile_values_bounded(&device, &prefix, settings.numeric_bytes)?;
    let rows = imported.contract.family.rows;
    let plan = resident
        .operator_numeric_bytes()?
        .checked_add(
            resident
                .edited_bytes_per_row()
                .checked_mul(rows)
                .and_then(|n| n.checked_mul(4))
                .ok_or("trace overflow")?,
        )
        .and_then(|n| n.checked_add(rows.checked_mul(rows)?.checked_mul(96)?))
        .ok_or("capture plan overflow")?;
    if plan > settings.numeric_bytes {
        return Err(format!("capture plan {plan} exceeds budget"));
    }
    let trace = resident.forward(&imported.contract.family)?;
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    save(
        &out.join("PROVENANCE.json"),
        &json!({"export_sha256":settings.export_sha256,"settings_sha256":sha256(settings_path)?,
        "native_export":imported.record,"backend":device.name(),"source_revision":option_env!("GAM_BUILD_GIT_SHA"),
        "scope":"All four native MLPs; fixed clean native parent activations, not autonomous counterfactual model execution; supplied native capacity control."}),
    )?;
    let mut reports = Vec::new();
    for (i, l) in layers.iter().enumerate() {
        let layer_dir = out.join(format!("layer{i}"));
        std::fs::create_dir(&layer_dir).map_err(|e| e.to_string())?;
        let x = device
            .download(trace.value(l.normed)?)
            .map_err(|e| e.to_string())?;
        let y = device
            .download(trace.value(l.mlp)?)
            .map_err(|e| e.to_string())?;
        let mut report = check_layer(
            &native,
            l.normed,
            l.pre,
            l.active,
            l.mlp,
            &x,
            &y,
            &directions,
            &settings.amplitudes,
            settings.absolute_tolerance,
            &layer_dir,
        )?;
        if let (Some(fit), Some((train, valid))) = (&settings.fitting, &split) {
            let (function, responses) =
                native_body(&native, l.normed, l.pre, l.active, l.mlp, &directions)?;
            report["fitting"] = fit_diagnostic(
                &device, &function, &responses, &x, train, valid, fit, &layer_dir,
            )?;
        }
        reports.push(report);
    }
    save(
        &out.join("REPORT.json"),
        &json!({"layers":reports,"capture_planned_numeric_bytes":plan,
        "elapsed_seconds":start.elapsed().as_secs_f64(),"fitting_updates_per_arm":settings.fitting.as_ref().map_or(0,|f|f.optimizer.iterations),
        "capacity_passed":true,"scope":"Native-supplied capacity gate and optional optimization diagnostic; fitting does not constitute discovered computation."}),
    )
}
fn main() -> Result<(), String> {
    run()
}
#[cfg(test)]
mod tests {
    use super::*;
    use gam_mpd::operator_program::Law;
    use ndarray::array;
    #[test]
    fn shared_native_hidden_bias_and_independent_signed_weights_survive_codec() {
        let scalar = Interface::constant();
        let hidden = Interface::native(3).expect("hidden");
        let output = Interface::native(2).expect("out");
        let source = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }],
                parameters: 0,
            },
            bases: vec![],
            rules: vec![],
            operators: vec![
                dense(
                    "reader",
                    hidden.clone(),
                    output.clone(),
                    array![[1., 0.5], [-0.25, 1.], [0.5, -0.5]],
                )
                .expect("reader"),
                dense(
                    "writer",
                    output.clone(),
                    hidden.clone(),
                    array![[1., -0.5, 0.25], [-0.25, 0.5, 1.]],
                )
                .expect("writer"),
                dense(
                    "reader bias",
                    hidden.clone(),
                    scalar.clone(),
                    array![[0.25], [-0.5], [0.125]],
                )
                .expect("bias"),
                dense(
                    "writer bias",
                    output.clone(),
                    scalar,
                    array![[0.25], [-0.125]],
                )
                .expect("bias"),
            ],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: Some(2),
                },
                Node::Pointwise {
                    input: 1,
                    laws: vec![Law::GeluTanh],
                },
                Node::Affine {
                    terms: vec![(2, 1)],
                    bias: Some(3),
                },
            ],
            output: 3,
        };
        let parent = array![[0.5, -0.25], [-0.5, 0.75], [1., 0.25]];
        let inputs = FamilyInputs {
            rows: 3,
            slots: vec![SlotValues::Raw(parent.clone())],
            layout: None,
        };
        let clean = source.execute(&inputs, false).expect("native").values[3].clone();
        let directions = vec![
            Direction {
                output: array![1., 0.],
                hidden: array![1., 0., 0.],
            },
            Direction {
                output: array![0., 1.],
                hidden: array![0., 1., 0.],
            },
        ];
        let out =
            std::env::temp_dir().join(format!("mpd-joint-native-control-{}", std::process::id()));
        std::fs::create_dir_all(&out).expect("temp");
        let report = check_layer(
            &source,
            0,
            1,
            2,
            3,
            &parent,
            &clean,
            &directions,
            &[vec![0., 0.], vec![0.5, 0.], vec![0., -0.5], vec![0.5, -0.5]],
            1e-12,
            &out,
        )
        .expect("capacity");
        assert_eq!(report["shared_hidden_node"], 2);
        assert_eq!(report["passed"], true);
        assert!(report["c32"].as_u64().expect("paid cost") > 0);
        let (function, responses) = native_body(&source, 0, 1, 2, 3, &directions).expect("body");
        let fit = Fitting {
            training_sequences: 1,
            seed: 2951,
            relative_perturbation: 0.01,
            optimizer: resident_rule_fit::Settings {
                iterations: 2,
                forward_rows: 2,
                learning_rate: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                numeric_bytes: 1 << 24,
                arithmetic: resident_rule_fit::ProposalArithmetic::F64,
                backtracking: None,
            },
            batch: BatchSchedule {
                objective: Default::default(),
                ordinary_rows: 2,
                hard_rows: 0,
                scan_every: 1,
                temperature: 0.1,
            },
        };
        let result = fit_diagnostic(
            &Device::host(),
            &function,
            &responses,
            &parent,
            &[0, 1],
            &[2],
            &fit,
            &out,
        )
        .expect("direct fitting");
        assert_eq!(result["arms"].as_array().expect("arms").len(), 2);
        for arm in result["arms"].as_array().expect("arms") {
            assert_eq!(arm["optimizer_report"]["trainable"], json!([0, 1, 2, 3]));
            assert_eq!(
                arm["measurements_frozen_training_scales"]
                    .as_array()
                    .expect("snapshots")
                    .len(),
                2
            );
        }
        std::fs::remove_dir_all(out).expect("cleanup");
    }
    #[test]
    fn partition_keeps_every_sequence_intact() {
        let family = FamilyInputs {
            rows: 5,
            slots: vec![],
            layout: Some(gam_mpd::operator_program::SequenceLayout {
                sequence: vec![4, 4, 9, 9, 9],
                position: vec![0, 1, 0, 1, 2],
            }),
        };
        assert_eq!(
            sequence_split(&family, 1).expect("split"),
            (vec![0, 1], vec![2, 3, 4])
        );
        assert!(sequence_split(&family, 2).is_err());
    }
}

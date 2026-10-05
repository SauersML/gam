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
    run_check::{layer_nodes, split_sites},
};
use ndarray::{Array1, Array2};
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
        reports.push(check_layer(
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
        )?);
    }
    save(
        &out.join("REPORT.json"),
        &json!({"layers":reports,"capture_planned_numeric_bytes":plan,
        "elapsed_seconds":start.elapsed().as_secs_f64(),"fitting_updates":0,"passed":true,"scope":"Native-supplied capacity gate only."}),
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
        std::fs::remove_dir_all(out).expect("cleanup");
    }
}

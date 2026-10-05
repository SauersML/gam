//! host|cuda [NATIVE_EXPORT]. Optional compact-label proposal parity, not acceptance.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    artifact::Artifact,
    coder_capture::sha256,
    device_program::DeviceProgram,
    import::import_language_model,
    operator_program::{
        Declarations, FamilyInputs, Interface, Node, Operator, OperatorBody, OperatorProgram,
        Scale, SequenceLayout, Slot, SlotValues, exact_precision,
    },
    resident_causal_fit::{self, Episode, FixedHeadEpisode, Settings, fixed_head_target::Teacher},
};
use ndarray::Array2;
use serde_json::{Value, json};
use std::{path::Path, sync::Arc, time::Instant};
fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn dense(name: &str, values: Array2<f64>) -> Result<Arc<Operator>, String> {
    Ok(Arc::new(
        Operator::dense(
            name,
            Interface::native(values.nrows()).map_err(error)?,
            Interface::native(values.ncols()).map_err(error)?,
            values.clone(),
            exact_precision(values.iter().copied()).map_err(error)?,
            Default::default(),
        )
        .map_err(error)?,
    ))
}
fn tiny(scale: f64, weight: f64) -> Result<OperatorProgram, String> {
    Ok(OperatorProgram {
        declarations: Declarations {
            domains: vec![],
            slots: vec![Slot::Raw { width: 2 }],
            parameters: 0,
        },
        bases: vec![],
        rules: vec![],
        operators: vec![
            dense("reader", ndarray::array![[weight, 0.25], [-0.5, 0.75]])?,
            dense(
                "fixed head",
                ndarray::array![
                    [scale, 0.25 * scale],
                    [-0.5 * scale, 0.75 * scale],
                    [0.25 * scale, -0.75 * scale]
                ],
            )?,
        ],
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Affine {
                terms: vec![(0, 0)],
                bias: None,
            },
            Node::Attend {
                query: 0,
                key: 0,
                value: 1,
                scale: Scale::InverseSqrt(2),
                rotary: None,
                causal: true,
            },
            Node::RmsNorm {
                input: 2,
                epsilon: 1e-6,
            },
            Node::Affine {
                terms: vec![(3, 1)],
                bias: None,
            },
        ],
        output: 4,
    })
}
fn max_parameter_difference(
    a: &OperatorProgram,
    b: &OperatorProgram,
    indices: &[usize],
) -> Result<f64, String> {
    let mut maximum = 0f64;
    for &i in indices {
        let a = a.operators.get(i).ok_or("parameter index")?.matrix_cow();
        let b = b.operators.get(i).ok_or("parameter index")?.matrix_cow();
        if a.dim() != b.dim() {
            return Err("parameter shape mismatch".into());
        }
        for (a, b) in a.iter().zip(b.iter()) {
            if !a.is_finite() || !b.is_finite() {
                return Err("nonfinite fitted parameter".into());
            }
            maximum = maximum.max((a - b).abs());
        }
    }
    Ok(maximum)
}
fn saved(source: &OperatorProgram) -> Result<OperatorProgram, String> {
    let artifact = Artifact::native(source)?.f32_literals()?;
    let bytes = artifact.to_bytes()?;
    let decoded = Artifact::from_bytes(&bytes, &artifact.program.declarations)?;
    if decoded.to_bytes()? != bytes {
        return Err("ordinary saved replay is not canonical".into());
    }
    Ok(decoded.program)
}
fn run_case(
    d: &Device,
    label: &str,
    source: &OperatorProgram,
    teacher: &OperatorProgram,
    inputs: &FamilyInputs,
    scored: Option<Vec<bool>>,
    trainable: &[usize],
    numeric_bytes: usize,
    loss_tolerance: f64,
    parameter_tolerance: f64,
) -> Result<Value, String> {
    let started = Instant::now();
    let t = Instant::now();
    let targeter = Teacher::new(d, teacher, 2, numeric_bytes)?;
    let target = targeter.target(inputs, scored.as_deref())?;
    let compact_teacher_seconds = t.elapsed().as_secs_f64();
    let compact_target_numeric_bytes = target.numeric_bytes();
    let compact = vec![FixedHeadEpisode {
        label: label.into(),
        group: "active".into(),
        inputs: inputs.clone(),
        target,
    }];
    let t = Instant::now();
    let teacher_program = DeviceProgram::compile(d, teacher)?;
    let trace = teacher_program.forward(inputs)?;
    let logits = teacher_program.logits_on_device(&trace)?;
    let full_target_numeric_bytes = logits.rows() * logits.cols() * 8;
    let target_logits = d.download(&logits).map_err(error)?;
    drop(logits);
    drop(trace);
    drop(teacher_program);
    let full_teacher_seconds = t.elapsed().as_secs_f64();
    let full = vec![Episode {
        label: label.into(),
        group: "active".into(),
        inputs: inputs.clone(),
        target_logits,
        scored,
    }];
    let full_measure = resident_causal_fit::measure(d, source, &full, numeric_bytes)?;
    let compact_measure =
        resident_causal_fit::measure_fixed_head(d, source, &compact, numeric_bytes, 2)?;
    let initial_loss_difference = (full_measure.objective - compact_measure.objective).abs();
    let settings = Settings {
        iterations: 2,
        learning_rate: 1e-4,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-6,
        numeric_bytes,
        schedule: None,
    };
    let t = Instant::now();
    let full_fit = resident_causal_fit::fit(d, source, &full, trainable, settings.clone())?;
    let full_fit_seconds = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let compact_fit =
        resident_causal_fit::fit_fixed_head(d, source, &compact, trainable, settings, 2)?;
    let compact_fit_seconds = t.elapsed().as_secs_f64();
    let parameter_difference =
        max_parameter_difference(&full_fit.program, &compact_fit.program, trainable)?;
    let curve_difference = full_fit
        .report
        .iterations
        .iter()
        .zip(&compact_fit.report.iterations)
        .map(|(a, b)| (a.measurement.objective - b.measurement.objective).abs())
        .fold(0f64, f64::max);
    let t = Instant::now();
    let full_saved = saved(&full_fit.program)?;
    let compact_saved = saved(&compact_fit.program)?;
    let saved_parameter_difference =
        max_parameter_difference(&full_saved, &compact_saved, trainable)?;
    let direct_replay = resident_causal_fit::measure(d, &full_saved, &full, numeric_bytes)?;
    let compact_replay =
        resident_causal_fit::measure_fixed_head(d, &compact_saved, &compact, numeric_bytes, 2)?;
    let saved_loss_difference = (direct_replay.objective - compact_replay.objective).abs();
    let saved_replay_seconds = t.elapsed().as_secs_f64();
    let finite = [
        initial_loss_difference,
        parameter_difference,
        curve_difference,
        saved_parameter_difference,
        saved_loss_difference,
    ]
    .iter()
    .all(|v| v.is_finite());
    let passed = finite
        && initial_loss_difference <= loss_tolerance
        && curve_difference <= loss_tolerance
        && saved_loss_difference <= loss_tolerance
        && parameter_difference <= parameter_tolerance
        && saved_parameter_difference <= parameter_tolerance;
    Ok(
        json!({"case":label,"rows":inputs.rows,"trainable":trainable,"tolerances":{"kl_absolute":loss_tolerance,"parameter_max_absolute":parameter_tolerance},"finite":finite,"passed":passed,"initial_full_kl":full_measure.objective,"initial_compact_kl":compact_measure.objective,"maximum_initial_kl_difference":initial_loss_difference,"maximum_training_curve_kl_difference":curve_difference,"maximum_parameter_difference":parameter_difference,"saved_f32_parameter_difference":saved_parameter_difference,"saved_f32_kl_difference":saved_loss_difference,"full_target_numeric_bytes":full_target_numeric_bytes,"compact_target_numeric_bytes":compact_target_numeric_bytes,"budgets":{"numeric_bytes":numeric_bytes,"head_tile_rows":2},"numeric_plans":{"full":full_fit.report.planned_numeric_bytes,"compact":compact_fit.report.planned_numeric_bytes},"seconds":{"compact_teacher":compact_teacher_seconds,"full_teacher":full_teacher_seconds,"full_fit":full_fit_seconds,"compact_fit":compact_fit_seconds,"ordinary_saved_replay":saved_replay_seconds,"total":started.elapsed().as_secs_f64()},"iterations":2,"full_fit":full_fit.report,"compact_fit":compact_fit.report}),
    )
}
fn main() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 1 && args.len() != 2 {
        return Err("host|cuda [NATIVE_EXPORT]".into());
    }
    let d = match args[0].as_str() {
        "host" => Device::host(),
        "cuda" => Device::accelerator(GpuPolicy::Required)
            .map_err(error)?
            .ok_or("CUDA required")?,
        _ => return Err("host|cuda required".into()),
    };
    if args[0] == "cuda" && (d.is_host() || !d.float64()) {
        return Err("actual f64 CUDA backend required".into());
    }
    let inputs = FamilyInputs {
        rows: 3,
        slots: vec![SlotValues::Raw(ndarray::array![
            [1., 0.5],
            [-0.5, 1.],
            [0.25, -0.5]
        ])],
        layout: Some(SequenceLayout {
            sequence: vec![0; 3],
            position: vec![0, 1, 2],
        }),
    };
    let mut cases = vec![
        run_case(
            &d,
            "attention-rms-masked",
            &tiny(1., 0.5)?,
            &tiny(1., 1.)?,
            &inputs,
            Some(vec![false, true, true]),
            &[0],
            1 << 27,
            1e-10,
            1e-9,
        )?,
        run_case(
            &d,
            "confident-underflow",
            &tiny(1000., 0.5)?,
            &tiny(1000., 1.)?,
            &inputs,
            None,
            &[0],
            1 << 27,
            1e-8,
            1e-8,
        )?,
    ];
    let mut native = Value::Null;
    if let Some(export) = args.get(1) {
        let export = Path::new(export);
        let imported = import_language_model(export, 1, 4)?;
        let mut teacher = imported.program.clone();
        let matches = teacher
            .operators
            .iter()
            .enumerate()
            .filter(|(_, op)| op.name == "blocks.0.c_fc")
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err("exact original native blocks.0.c_fc operator required".into());
        }
        let index = matches[0];
        let op = Arc::make_mut(&mut teacher.operators[index]);
        let OperatorBody::Dense {
            values,
            precision,
            present,
        } = &mut op.body
        else {
            return Err("native reader must be dense".into());
        };
        if present.iter().any(|v| !*v) {
            return Err("native reader must be full-present".into());
        }
        values[(0, 0)] += 0.02;
        *precision = exact_precision(values.iter().copied()).map_err(error)?;
        cases.push(run_case(
            &d,
            "actual-native4l-four-tokens",
            &imported.program,
            &teacher,
            &imported.contract.family,
            Some(vec![false, true, true, true]),
            &[index],
            16usize << 30,
            1e-8,
            1e-8,
        )?);
        native = json!({"export_sha256":sha256(&export.join("export.json"))?,"source":imported.record,"teacher_perturbation":{"operator":"blocks.0.c_fc","row":0,"column":0,"add":0.02},"scope":"known perturbation for backend parity only, not a fitted scientific candidate"});
    }
    let passed = cases.iter().all(|c| c["passed"].as_bool() == Some(true));
    println!("{}",serde_json::to_string_pretty(&json!({"backend":args[0],"device":d.name(),"passed":passed,"cases":cases,"native":native,"scope":"Optional proposal-training numerical parity. Disclosed absolute tolerances, F64 full-prefix VJP and full-vocabulary normalization; no real-arithmetic certificate or acceptance claim. Saved candidate literals replayed as ordinary f32 artifacts."})).map_err(error)?);
    if !passed {
        return Err("fixed-head parity exceeded declared tolerances".into());
    }
    Ok(())
}

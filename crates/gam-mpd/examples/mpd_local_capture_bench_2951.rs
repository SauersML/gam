//! Same-input native supervision capture on a real engine export.
//!
//! `mpd_local_capture_bench_2951 EXPORT SEQUENCES CONTEXT REPS OUT [host|cuda]`
//!
//! Compares the CPU reference with prepared F64 execution, including final panel downloads.
//! Captures the final MLP's normalized input -> activation/write boundary on clean states and
//! states reached after multiplying the first attention output matrices by 1.5 and -0.5.
//! The native teacher evaluates its reader(s), including both branches of gated MLPs;
//! incomplete boundaries fail rather than use hidden teacher inputs. Every capture retains
//! all supplied context tokens and their sequence layout.
//! Optional numeric budgets: MPD_CAPTURE_OPERATOR_BYTES (8 GiB by default),
//! MPD_CAPTURE_TRACE_BYTES (16 GiB). These are the APIs' numeric plans, not process RSS limits.
//! No vocabulary head, learning, acceptance, or mechanism-discovery measurement is performed.

use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    artifact::Artifact,
    import::import_language_model,
    native_local_supervision::{self, Panel, PreparedDeviceCapture},
    operator_program::{Node, OperatorBody, exact_precision},
    run_check::{layer_nodes, split_sites},
};
use ndarray::Array2;
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::PathBuf, sync::Arc, time::Instant};

const ABS_TOL: f64 = 1e-9;
const REL_TOL: f64 = 1e-8;

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.
    } else {
        sorted[middle]
    }
}

fn budget(name: &str, default: usize) -> Result<usize, String> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|v| *v > 0)
            .ok_or_else(|| format!("{name} must be a positive byte count")),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("{name}: {error}")),
    }
}

fn compare(actual: &Array2<f64>, reference: &Array2<f64>) -> Result<(Value, bool), String> {
    if actual.dim() != reference.dim() || actual.is_empty() {
        return Err("capture panel shapes disagree or are empty".into());
    }
    let (mut maximum, mut scale, mut normalized) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&a, &r) in actual.iter().zip(reference) {
        if !a.is_finite() || !r.is_finite() {
            return Err("nonfinite parity panel".into());
        }
        let error = (a - r).abs();
        maximum = maximum.max(error);
        scale = scale.max(r.abs());
        normalized = normalized.max(error / (ABS_TOL + REL_TOL * r.abs()));
    }
    Ok((
        json!({
            "shape": [actual.nrows(), actual.ncols()],
            "max_absolute_error": maximum,
            "max_error_relative_to_reference_max": maximum / scale.max(f64::MIN_POSITIVE),
            "max_error_over_elementwise_tolerance": normalized,
            "passed": normalized <= 1.,
        }),
        normalized <= 1.,
    ))
}

fn parity(actual: &Panel, reference: &Panel) -> Result<(Value, bool), String> {
    if actual.inputs.len() != reference.inputs.len() {
        return Err("capture boundary count differs".into());
    }
    let (mut inputs, mut passed) = (Vec::new(), true);
    for (a, r) in actual.inputs.iter().zip(&reference.inputs) {
        let (record, ok) = compare(a, r)?;
        inputs.push(record);
        passed &= ok;
    }
    let (targets, ok) = compare(&actual.targets, &reference.targets)?;
    passed &= ok;
    if actual.report.arguments != reference.report.arguments
        || actual.report.native_outputs != reference.report.native_outputs
        || actual.report.native_evaluated_nodes != reference.report.native_evaluated_nodes
        || actual.report.rows != reference.report.rows
    {
        return Err("capture native correspondence reports disagree".into());
    }
    Ok((
        json!({"inputs": inputs, "targets": targets, "passed": passed}),
        passed,
    ))
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let usage = "mpd_local_capture_bench_2951 EXPORT SEQUENCES CONTEXT REPS OUT [host|cuda]";
    if !(5..=6).contains(&args.len()) {
        return Err(usage.into());
    }
    let positive = |s: &str| {
        s.parse::<usize>()
            .ok()
            .filter(|v| *v > 0)
            .ok_or_else(|| format!("positive count required: {s}"))
    };
    let (sequences, context, reps) = (
        positive(&args[1])?,
        positive(&args[2])?,
        positive(&args[3])?,
    );
    let export = PathBuf::from(&args[0]);
    let out = PathBuf::from(&args[4]);
    let mode = args.get(5).map(String::as_str).unwrap_or("host");
    let operator_budget = budget("MPD_CAPTURE_OPERATOR_BYTES", 8 << 30)?;
    let trace_budget = budget("MPD_CAPTURE_TRACE_BYTES", 16 << 30)?;
    let started = Instant::now();
    let device = match mode {
        "host" => Device::host(),
        "cuda" => {
            if !cfg!(target_os = "linux") {
                return Err("cuda mode requires Linux CUDA; no Metal or CPU fallback".into());
            }
            let device = Device::accelerator(GpuPolicy::Required)
                .map_err(|e| e.to_string())?
                .ok_or("required CUDA accelerator absent")?;
            if device.is_host() || !device.float64() {
                return Err("required F64 CUDA device absent".into());
            }
            device
        }
        _ => return Err(usage.into()),
    };
    device.synchronize().map_err(|e| e.to_string())?;
    let device_setup_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let imported = import_language_model(&export, sequences, context)?;
    let layers = usize::try_from(
        imported.record["config"]["n_layers"]
            .as_u64()
            .ok_or("export n_layers absent")?,
    )
    .map_err(|e| e.to_string())?;
    let native = split_sites(&imported.program)?;
    let sites = layer_nodes(&native, layers)?;
    let selected = sites.last().ok_or("no native MLP")?;
    let arguments = [selected.normed];
    let outputs = [selected.active, selected.mlp];
    let Node::Affine { terms, bias } = &native.nodes[sites[0].attention] else {
        return Err("first attention output is not affine".into());
    };
    let mut edits = terms.iter().map(|(_, op)| *op).collect::<BTreeSet<_>>();
    edits.extend(*bias);
    if edits.is_empty() {
        return Err("no attention output weight edits".into());
    }
    let family = &imported.family;
    let rows = sequences.checked_mul(context).ok_or("row count overflow")?;
    if family.rows != rows {
        return Err("import changed requested context count".into());
    }
    let layout = family.layout.as_ref().ok_or("missing sequence layout")?;
    for row in 0..rows {
        if layout.sequence.get(row).copied() != u32::try_from(row / context).ok()
            || layout.position.get(row).copied() != u32::try_from(row % context).ok()
        {
            return Err("imported rows are not the requested complete contexts".into());
        }
    }
    let import_and_boundary_setup_seconds = started.elapsed().as_secs_f64();
    let mut cases = Vec::new();
    let mut clean_inputs: Option<Vec<Array2<f64>>> = None;
    let mut all_passed = true;
    for (label, gain) in [
        ("native", 1.),
        ("attention_writer_gain_1.5", 1.5),
        ("attention_writer_gain_-0.5", -0.5),
    ] {
        let started = Instant::now();
        let mut candidate = Artifact::native(&native)?;
        if gain != 1. {
            for &op in &edits {
                let OperatorBody::Dense {
                    values, precision, ..
                } = &mut Arc::make_mut(&mut candidate.program.operators[op]).body
                else {
                    return Err("upstream writer edit requires dense native matrices".into());
                };
                values.mapv_inplace(|v| v * gain);
                *precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
            }
        }
        let candidate_setup_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let prepared = PreparedDeviceCapture::new(
            &device,
            &native,
            &candidate,
            &arguments,
            &outputs,
            operator_budget,
        )?;
        device.synchronize().map_err(|e| e.to_string())?;
        let compile_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let reference = native_local_supervision::capture(
            &native,
            &candidate,
            &arguments,
            &outputs,
            family,
            trace_budget,
        )?;
        let cpu_warmup_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let warmup = prepared.capture(family, trace_budget)?;
        device.synchronize().map_err(|e| e.to_string())?;
        let device_warmup_seconds = started.elapsed().as_secs_f64();
        let (warmup_parity, ok) = parity(&warmup, &reference)?;
        all_passed &= ok;
        let device_capture_report = warmup.report.clone();
        drop(warmup);
        let changed_input_max_absolute = if let Some(clean) = &clean_inputs {
            reference
                .inputs
                .iter()
                .zip(clean)
                .flat_map(|(a, b)| a.iter().zip(b))
                .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()))
        } else {
            0.
        };
        if gain == 1. {
            clean_inputs = Some(reference.inputs.clone());
        } else if changed_input_max_absolute == 0. {
            return Err("upstream edit did not change the captured input".into());
        }
        let (mut cpu_seconds, mut device_seconds, mut checks) =
            (Vec::new(), Vec::new(), Vec::new());
        for rep in 0..reps {
            // Alternate timed order. Comparisons run after timing; panel allocation/downloads
            // remain timed because they are part of capture's actual API contract.
            for on_device in if rep % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                device.synchronize().map_err(|e| e.to_string())?;
                let started = Instant::now();
                let panel = if on_device {
                    prepared.capture(family, trace_budget)?
                } else {
                    native_local_supervision::capture(
                        &native,
                        &candidate,
                        &arguments,
                        &outputs,
                        family,
                        trace_budget,
                    )?
                };
                device.synchronize().map_err(|e| e.to_string())?;
                let seconds = started.elapsed().as_secs_f64();
                if on_device {
                    device_seconds.push(seconds);
                } else {
                    cpu_seconds.push(seconds);
                }
                let (check, ok) = parity(&panel, &reference)?;
                all_passed &= ok;
                checks.push(
                    json!({"repetition": rep, "prepared_device": on_device, "parity": check}),
                );
            }
        }
        cases.push(json!({
            "label": label, "upstream_weight_gain": gain,
            "changed_input_max_absolute": changed_input_max_absolute,
            "candidate_setup_seconds": candidate_setup_seconds, "compile_seconds": compile_seconds,
            "cpu_warmup_seconds": cpu_warmup_seconds, "device_warmup_seconds": device_warmup_seconds,
            "cpu_capture_seconds": cpu_seconds, "prepared_capture_seconds": device_seconds,
            "cpu_capture_median_seconds": median(&cpu_seconds), "prepared_capture_median_seconds": median(&device_seconds),
            "operator_numeric_bytes": prepared.operator_numeric_bytes(),
            "cpu_capture_report": reference.report, "device_capture_report": device_capture_report,
            "warmup_parity": warmup_parity, "repeated_parity": checks,
        }));
    }
    let report = json!({
        "schema": "native-local-capture-bench-v1", "parity_passed": all_passed,
        "scope": "Native same-input local supervision capture only; complete contexts, final input/target downloads included. No vocabulary head, fitting, acceptance, or mechanism-discovery result.",
        "device": device.name(), "mode": mode, "storage": "F64", "arithmetic": "F64",
        "export": export,
        "export_manifest_sha256": gam_mpd::engine::sha256(&export.join("export.json"))?,
        "checkpoint_sha256": imported.record["source"]["checkpoint_sha256"],
        "sequences": sequences, "context": context, "rows": rows, "repetitions_after_one_warmup": reps,
        "selected_layer": layers - 1, "arguments": arguments, "outputs": outputs,
        "upstream_edit_operators": edits.iter().map(|i| native.operators[*i].name.clone()).collect::<Vec<_>>(),
        "operator_numeric_budget_bytes": operator_budget, "trace_panel_numeric_budget_bytes": trace_budget,
        "budget_scope": "Numeric API budgets, not RSS/VRAM caps. Exclude imported host coefficients, caller-retained reference/clean panels, allocator and backend scratch, input storage and metadata.",
        "absolute_tolerance": ABS_TOL, "relative_tolerance": REL_TOL,
        "parity_criterion": "Every element: abs(actual-reference) <= atol + rtol*abs(reference)",
        "device_setup_seconds": device_setup_seconds,
        "import_and_boundary_setup_seconds": import_and_boundary_setup_seconds,
        "cases": cases,
    });
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    std::fs::write(out.join("local_capture_bench.json"), &text).map_err(|e| e.to_string())?;
    println!("{text}");
    if !all_passed {
        return Err(
            "capture parity failed; timings are not evidence of an equivalent speedup".into(),
        );
    }
    Ok(())
}

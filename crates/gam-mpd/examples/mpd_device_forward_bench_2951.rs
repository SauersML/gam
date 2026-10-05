//! Forward and reverse wall time of a language model's resident program on this host's device
//! (#2951, `gam_mpd::device_program`), with its parity against the CPU.
//!
//! `mpd_device_forward_bench_2951 TRAIN OUT [KEY=VALUE ...]`
//!
//! `TRAIN` is an engine export (`~/mpd-data/engine/vpd4l_e2e_train`). On its first `sequences`
//! sequences (8) of `context` positions (512), the program up to the head's hidden node is
//! compiled in resident-value mode (the fitter's prefix) and timed over `reps` forwards (5), each
//! forward also followed by a reverse pass from the hidden node to the embedding, after one
//! warm-up. `arithmetic` (`f64`, `tf32`) picks the products' arithmetic; `cpu=0` skips the CPU
//! reference. One JSON object goes to `OUT/bench.json` and stdout.

use gam_gpu::{GpuPolicy, tensor::{Arithmetic, Device}};
use gam_mpd::device_program::DeviceProgram;
use gam_mpd::import::import_language_model;
use gam_mpd::operator_program::Node;
use ndarray::Array2;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err("mpd_device_forward_bench_2951 TRAIN OUT [KEY=VALUE ...]".into());
    }
    let (train, out) = (PathBuf::from(&args[1]), PathBuf::from(&args[2]));
    let (mut sequences, mut context, mut reps, mut cpu, mut arithmetic) = (8usize, 512usize, 5usize, true, Arithmetic::F64);
    for pair in &args[3..] {
        let (key, value) = pair.split_once('=').ok_or_else(|| format!("{pair}: not KEY=VALUE"))?;
        let count = || value.parse::<usize>().map_err(|e| format!("{key}: {e}"));
        match key {
            "sequences" => sequences = count()?,
            "context" => context = count()?,
            "reps" => reps = count()?.max(1),
            "cpu" => cpu = count()? != 0,
            "arithmetic" => arithmetic = match value { "f64" => Arithmetic::F64, "tf32" => Arithmetic::Tf32, other => return Err(format!("arithmetic {other}")) },
            other => return Err(format!("unknown key {other}")),
        }
    }
    let imported = import_language_model(&train, sequences, context)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    // The hidden node the head reads: the input of the transposed (tied) unembedding.
    let logits = match &model.nodes[model.output] { Node::Readout { input, .. } => *input, _ => model.output };
    let hidden = match &model.nodes[logits] {
        Node::Transposed { input, .. } => *input,
        Node::Affine { terms, .. } => terms[0].0,
        _ => return Err("the output is not a linear head".into()),
    };
    let mut prefix = model.clone();
    prefix.nodes.truncate(hidden + 1);
    prefix.output = hidden;
    // The reverse pass ends at the embedding (the first node an affine term reads a feature into).
    let first = prefix.nodes.iter().position(|n| matches!(n, Node::Affine { .. })).ok_or("no affine node")?;
    let device = Device::accelerator(GpuPolicy::Required).map_err(|e| e.to_string())?.ok_or("no accelerator")?;
    let started = Instant::now();
    let mut program = DeviceProgram::compile_values(&device, &prefix)?;
    program.set_arithmetic(arithmetic);
    let compile_seconds = started.elapsed().as_secs_f64();
    let width = program.widths()[hidden];
    let seed = Array2::from_shape_fn((family.rows, width), |(r, c)| (((r * 7 + c * 13) % 17) as f64 - 8.0) / 64.0);
    let seed = device.upload(seed.view()).map_err(|e| e.to_string())?;
    let (mut forward, mut reverse) = (Vec::new(), Vec::new());
    let mut values = None;
    for rep in 0..=reps {
        device.synchronize().map_err(|e| e.to_string())?;
        let t = Instant::now();
        let trace = program.forward(family)?;
        device.synchronize().map_err(|e| e.to_string())?;
        let f = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let seeds = BTreeMap::from([(hidden, device.copy(&seed).map_err(|e| e.to_string())?)]);
        let kept = program.vjp_values_seeded(&trace, seeds, &[first], arithmetic)?;
        device.synchronize().map_err(|e| e.to_string())?;
        let r = t.elapsed().as_secs_f64();
        if rep > 0 {
            forward.push(f);
            reverse.push(r);
        } else {
            values = Some((device.download(trace.value(hidden)?).map_err(|e| e.to_string())?, device.download(&kept[&first]).map_err(|e| e.to_string())?));
        }
    }
    let (values, cotangent) = values.ok_or("no warm-up pass")?;
    let parity = if cpu {
        let t = Instant::now();
        let trace = prefix.execute(family, false).map_err(|e| e.to_string())?;
        let host = Array2::from_shape_fn((family.rows, width), |(r, c)| (((r * 7 + c * 13) % 17) as f64 - 8.0) / 64.0);
        let reference = gam_mpd::derivatives::vjp(&prefix, family, &trace, host).map_err(|e| e.to_string())?;
        let cpu_seconds = t.elapsed().as_secs_f64();
        let relative = |a: &Array2<f64>, b: &Array2<f64>| {
            let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
            (a - b).iter().fold(0.0_f64, |m, v| m.max(v.abs())) / scale
        };
        let expected = reference[first].as_ref().ok_or("no reference cotangent")?;
        json!({ "cpu_seconds": cpu_seconds, "hidden_max_relative": relative(&values, &trace.values[hidden]), "cotangent_max_relative": relative(&cotangent, expected) })
    } else {
        json!(null)
    };
    let median = |v: &mut Vec<f64>| { v.sort_by(f64::total_cmp); v[v.len() / 2] };
    let report = json!({
        "device": device.name(),
        "rows": family.rows,
        "nodes": prefix.nodes.len(),
        "arithmetic": format!("{arithmetic:?}"),
        "compile_seconds": compile_seconds,
        "forward_seconds": forward.clone(),
        "reverse_seconds": reverse.clone(),
        "forward_median": median(&mut forward),
        "reverse_median": median(&mut reverse),
        "parity": parity,
    });
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    std::fs::write(out.join("bench.json"), &text).map_err(|e| e.to_string())?;
    println!("{text}");
    Ok(())
}

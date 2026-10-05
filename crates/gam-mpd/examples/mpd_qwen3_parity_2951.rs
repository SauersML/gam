//! A Hugging Face checkpoint's operator program (`import::hugging_face_language_model`) run as a
//! device program, against the Hugging Face forward of the same model (#2951).
//!
//! `mpd_qwen3_parity_2951 MODEL_DIR WINDOWS OUT [KEY=VALUE ...]`
//!
//! `MODEL_DIR` holds `config.json` and `model.safetensors`; `WINDOWS` is a file of rows of `context`
//! little-endian u32 tokens (`bench/mpd_qwen3_fineweb_2951.py`). The first `sequences` rows (4) of
//! `context` tokens (128), each its own sequence from position 0, run through the whole program on
//! `device` (`host`, the CPU lowering in f64; `cuda`, with `storage` `f64` or `f32` and `arithmetic`
//! `f64`, `f32` or `tf32`). Each `reference=PATH` (repeatable) is the logits of
//! `bench/mpd_qwen3_reference_2951.py` on the same rows; per reference the report gives the largest
//! absolute logit difference and `KL(p_ref ‖ p_ours)` per token in nats (mean and largest) with the
//! top-1 agreement. The device's resident bytes after compiling and after the forward are reported
//! with the program's trace bytes per row. One JSON object goes to `OUT/parity.json` and stdout.

use gam_gpu::{
    GpuPolicy,
    tensor::{Arithmetic, Device, Op, Storage},
};
use gam_mpd::{
    device_program::DeviceProgram,
    import::hugging_face_language_model,
    operator_program::{FamilyInputs, Node, SequenceLayout, SlotValues},
};
use ndarray::{Array2, ArrayView1};
use serde_json::json;
use std::{path::PathBuf, time::Instant};

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// `ln Σ exp(row)`.
fn log_partition(row: ArrayView1<'_, f64>) -> f64 {
    let peak = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    peak + row.iter().map(|x| (x - peak).exp()).sum::<f64>().ln()
}

fn argmax(row: ArrayView1<'_, f64>) -> usize {
    row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i)
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("mpd_qwen3_parity_2951 MODEL_DIR WINDOWS OUT [KEY=VALUE ...]".into());
    }
    let (model_dir, windows, out) = (PathBuf::from(&args[1]), PathBuf::from(&args[2]), PathBuf::from(&args[3]));
    let (mut sequences, mut context, mut on_device) = (4usize, 128usize, false);
    let (mut storage, mut arithmetic, mut references) = (Storage::F64, Arithmetic::F64, Vec::new());
    for pair in &args[4..] {
        let (key, value) = pair.split_once('=').ok_or_else(|| format!("{pair}: not KEY=VALUE"))?;
        let count = || value.parse::<usize>().map_err(|e| format!("{key}: {e}"));
        match key {
            "sequences" => sequences = count()?,
            "context" => context = count()?,
            "device" => {
                on_device = match value {
                    "host" => false,
                    "cuda" => true,
                    other => return Err(format!("device {other}")),
                }
            }
            "storage" => {
                storage = match value {
                    "f64" => Storage::F64,
                    "f32" => Storage::F32,
                    other => return Err(format!("storage {other}")),
                }
            }
            "arithmetic" => {
                arithmetic = match value {
                    "f64" => Arithmetic::F64,
                    "f32" => Arithmetic::F32,
                    "tf32" => Arithmetic::Tf32,
                    other => return Err(format!("arithmetic {other}")),
                }
            }
            "reference" => references.push(PathBuf::from(value)),
            other => return Err(format!("unknown key {other}")),
        }
    }
    if sequences == 0 || context == 0 {
        return Err("sequences and context must be positive".into());
    }
    let bytes = std::fs::read(&windows).map_err(|e| format!("{}: {e}", windows.display()))?;
    if bytes.len() < sequences * context * 4 {
        return Err(format!("{}: fewer than {sequences} rows of {context} tokens", windows.display()));
    }
    let tokens: Vec<u32> = bytes[..sequences * context * 4].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().expect("four bytes"))).collect();
    let rows = tokens.len();
    let family = FamilyInputs {
        rows,
        slots: vec![SlotValues::Tokens(tokens)],
        layout: Some(SequenceLayout {
            sequence: (0..rows).map(|r| (r / context) as u32).collect(),
            position: (0..rows).map(|r| (r % context) as u32).collect(),
        }),
    };

    let started = Instant::now();
    let text = std::fs::read_to_string(model_dir.join("config.json")).map_err(error)?;
    let layers = serde_json::from_str::<serde_json::Value>(&text).map_err(error)?["num_hidden_layers"].as_u64().ok_or("num_hidden_layers")? as usize;
    let (program, record) = hugging_face_language_model(&model_dir, 0..layers)?;
    let import_seconds = started.elapsed().as_secs_f64();
    let device = if on_device {
        match storage {
            Storage::F64 => Device::accelerator(GpuPolicy::Required),
            Storage::F32 => Device::single_precision(GpuPolicy::Required),
            Storage::Bf16 => return Err("bfloat16 is not a device storage".into()),
        }
        .map_err(error)?
        .ok_or("no accelerator")?
    } else {
        Device::host()
    };
    // The trunk runs up to the hidden node `h` the head reads; the logits are `h Eᵀ`, `E` the head
    // matrix as classes × hidden (the transposed token embedding when tied), formed on the device
    // in the same arithmetic as the fit's teacher.
    let logits_node = match &program.nodes[program.output] {
        Node::Readout { input, .. } => *input,
        _ => program.output,
    };
    let (hidden, embedding) = match &program.nodes[logits_node] {
        Node::Transposed { input, operator } => (*input, program.operators[*operator].matrix().reversed_axes()),
        Node::Affine { terms, bias: None } if terms.len() == 1 => (terms[0].0, program.operators[terms[0].1].matrix()),
        _ => return Err("the output is not a linear head".into()),
    };
    let mut prefix = program.clone();
    prefix.nodes.truncate(hidden + 1);
    prefix.output = hidden;
    let free = |device: &Device| device.memory().map_err(error).map(|m| m.map(|(free, _)| free));
    let free_before = free(&device)?;
    let started = Instant::now();
    let mut compiled = DeviceProgram::compile_values(&device, &prefix)?;
    compiled.set_arithmetic(arithmetic);
    let head = device.upload(embedding.view()).map_err(error)?;
    let compile_seconds = started.elapsed().as_secs_f64();
    let free_compiled = free(&device)?;
    let started = Instant::now();
    let trace = compiled.forward(&family)?;
    device.synchronize().map_err(error)?;
    let forward_seconds = started.elapsed().as_secs_f64();
    let free_forward = free(&device)?;
    let vocab = embedding.nrows();
    let states = trace.value(hidden)?;
    let mut ours = Array2::<f64>::zeros((rows, vocab));
    for start in (0..rows).step_by(context) {
        let n = context.min(rows - start);
        let h = device.rows_of(states, start, n).map_err(error)?;
        let mut tile = device.zeros(n, vocab).map_err(error)?;
        device.gemm(&mut tile, 1.0, &h, Op::N, &head, Op::T, 0.0, arithmetic).map_err(error)?;
        ours.slice_mut(ndarray::s![start..start + n, ..]).assign(&device.download(&tile).map_err(error)?);
    }
    if ours.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite logits".into());
    }
    let mut compared = Vec::new();
    for path in &references {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if bytes.len() != rows * vocab * 8 {
            return Err(format!("{}: {} bytes, not {rows} x {vocab} float64", path.display(), bytes.len()));
        }
        let reference = Array2::from_shape_vec((rows, vocab), bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("eight bytes"))).collect())
            .map_err(error)?;
        let (mut max_abs, mut kl_sum, mut kl_max, mut agree) = (0.0_f64, 0.0_f64, 0.0_f64, 0usize);
        let mut per_sequence = vec![0.0_f64; sequences];
        for r in 0..rows {
            let (p, q) = (reference.row(r), ours.row(r));
            let difference = p.iter().zip(q.iter()).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
            max_abs = max_abs.max(difference);
            per_sequence[r / context] = per_sequence[r / context].max(difference);
            let (zp, zq) = (log_partition(p), log_partition(q));
            let kl: f64 = p.iter().zip(q.iter()).map(|(a, b)| (a - zp).exp() * ((a - zp) - (b - zq))).sum();
            kl_sum += kl;
            kl_max = kl_max.max(kl);
            agree += usize::from(argmax(p) == argmax(q));
        }
        compared.push(json!({
            "reference": path.display().to_string(),
            "max_abs_logit_difference": max_abs,
            "max_abs_logit_difference_per_sequence": per_sequence,
            "mean_kl_nats": kl_sum / rows as f64,
            "max_kl_nats": kl_max,
            "top1_agreement": agree as f64 / rows as f64,
        }));
    }
    let report = json!({
        "model": model_dir.display().to_string(),
        "config": record["config"],
        "windows": windows.display().to_string(),
        "sequences": sequences,
        "context": context,
        "device": device.name(),
        "storage": format!("{storage:?}"),
        "arithmetic": format!("{arithmetic:?}"),
        "operators": program.operators.len(),
        "nodes": program.nodes.len(),
        "reals": program.real_count(),
        "trace_bytes_per_row": compiled.bytes_per_row(),
        "device_free_bytes": {"before": free_before, "compiled": free_compiled, "after_forward": free_forward},
        "import_seconds": import_seconds,
        "compile_seconds": compile_seconds,
        "forward_seconds": forward_seconds,
        "comparisons": compared,
    });
    std::fs::create_dir_all(&out).map_err(error)?;
    let text = serde_json::to_string_pretty(&report).map_err(error)?;
    std::fs::write(out.join("parity.json"), &text).map_err(error)?;
    println!("{text}");
    Ok(())
}

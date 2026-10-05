//! Device memory and time of one fitting step's passes on a Hugging Face checkpoint's program
//! (#2951): the native model's prefix up to the head's hidden node, every block's dense operator
//! trainable, a forward and a reverse pass into those operators, as the library fit runs them.
//!
//! `mpd_qwen3_step_bench_2951 MODEL_DIR WINDOWS OUT [KEY=VALUE ...]`
//!
//! `WINDOWS` holds rows of `context` little-endian u32 tokens (128). For each count in `batches`
//! (`8,16,32`, sequences per pass), the first rows run `reps` times (3, after one warm-up) in f32
//! device storage with products in `arithmetic` (`f32` or `tf32`): a forward pass, then the
//! reverse pass from a fixed cotangent at the hidden node into every trainable operator. A thread
//! samples the device's free memory every millisecond; the report gives, per batch, the median
//! seconds of each pass, the device bytes in use at its peak and after the forward, and the
//! resident bytes of the compiled program before any pass, so the bytes a token adds are the
//! slope over the batches. One JSON object goes to `OUT/step.json` and stdout.

use gam_gpu::{
    GpuPolicy,
    tensor::{Arithmetic, Device},
};
use gam_mpd::{
    device_program::DeviceProgram,
    import::hugging_face_language_model,
    operator_program::{FamilyInputs, Node, OperatorProgram, SequenceLayout, SlotValues},
};
use ndarray::Array2;
use serde_json::json;
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// The hidden node the tied or untied head reads.
fn hidden_node(program: &OperatorProgram) -> Result<usize, String> {
    let logits = match &program.nodes[program.output] {
        Node::Readout { input, .. } => *input,
        _ => program.output,
    };
    match &program.nodes[logits] {
        Node::Transposed { input, .. } => Ok(*input),
        Node::Affine { terms, .. } if terms.len() == 1 => Ok(terms[0].0),
        _ => Err("the output is not a linear head".into()),
    }
}

/// The family of the first `count` rows of `context` tokens.
fn family(tokens: &[u32], count: usize, context: usize) -> FamilyInputs {
    let rows = count * context;
    FamilyInputs {
        rows,
        slots: vec![SlotValues::Tokens(tokens[..rows].to_vec())],
        layout: Some(SequenceLayout {
            sequence: (0..rows).map(|r| (r / context) as u32).collect(),
            position: (0..rows).map(|r| (r % context) as u32).collect(),
        }),
    }
}

/// Samples the device's free memory until stopped; the least free bytes seen.
struct Sampler {
    stop: Arc<AtomicBool>,
    least: Arc<AtomicUsize>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Sampler {
    fn start(device: &Device) -> Self {
        let (stop, least) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicUsize::new(usize::MAX)));
        let (device, flag, low) = (device.clone(), Arc::clone(&stop), Arc::clone(&least));
        let thread = std::thread::spawn(move || {
            while !flag.load(Ordering::Relaxed) {
                if let Ok(Some((free, _))) = device.memory() {
                    low.fetch_min(free, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        });
        Self { stop, least, thread: Some(thread) }
    }

    fn finish(mut self) -> usize {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().expect("the memory sampler ends");
        }
        self.least.load(Ordering::Relaxed)
    }
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("mpd_qwen3_step_bench_2951 MODEL_DIR WINDOWS OUT [KEY=VALUE ...]".into());
    }
    let (model_dir, windows, out) = (PathBuf::from(&args[1]), PathBuf::from(&args[2]), PathBuf::from(&args[3]));
    let (mut context, mut batches, mut reps, mut arithmetic) = (128usize, vec![8usize, 16, 32], 3usize, Arithmetic::Tf32);
    for pair in &args[4..] {
        let (key, value) = pair.split_once('=').ok_or_else(|| format!("{pair}: not KEY=VALUE"))?;
        match key {
            "context" => context = value.parse().map_err(|e| format!("{key}: {e}"))?,
            "reps" => reps = value.parse().map_err(|e| format!("{key}: {e}"))?,
            "batches" => batches = value.split(',').map(|v| v.parse::<usize>().map_err(|e| format!("{key}: {e}"))).collect::<Result<_, _>>()?,
            "arithmetic" => {
                arithmetic = match value {
                    "f32" => Arithmetic::F32,
                    "tf32" => Arithmetic::Tf32,
                    other => return Err(format!("arithmetic {other}")),
                }
            }
            other => return Err(format!("unknown key {other}")),
        }
    }
    let largest = batches.iter().copied().max().ok_or("no batches")?;
    if context == 0 || reps == 0 || batches.contains(&0) {
        return Err("context, reps and batches must be positive".into());
    }
    let bytes = std::fs::read(&windows).map_err(|e| format!("{}: {e}", windows.display()))?;
    if bytes.len() < largest * context * 4 {
        return Err(format!("{}: fewer than {largest} rows of {context} tokens", windows.display()));
    }
    let tokens: Vec<u32> = bytes[..largest * context * 4].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().expect("four bytes"))).collect();
    let text = std::fs::read_to_string(model_dir.join("config.json")).map_err(error)?;
    let layers = serde_json::from_str::<serde_json::Value>(&text).map_err(error)?["num_hidden_layers"].as_u64().ok_or("num_hidden_layers")? as usize;
    let (program, record) = hugging_face_language_model(&model_dir, 0..layers)?;
    let hidden = hidden_node(&program)?;
    let mut prefix = program.clone();
    prefix.nodes.truncate(hidden + 1);
    prefix.output = hidden;
    let block = |name: &str| {
        name.strip_prefix("blocks.").and_then(|rest| rest.split_once('.')).is_some_and(|(_, part)| {
            ["c_fc", "gate_proj", "down_proj"].contains(&part) || ["q", "k", "v", "o"].iter().any(|p| part.strip_prefix(p).is_some_and(|h| h.parse::<usize>().is_ok()))
        })
    };
    let trainable: Vec<usize> = prefix.operators.iter().enumerate().filter(|(_, op)| block(&op.name)).map(|(i, _)| i).collect();
    let parameters: usize = trainable.iter().map(|op| prefix.operators[*op].rows.width() * prefix.operators[*op].cols.width()).sum();
    let device = Device::single_precision(GpuPolicy::Required).map_err(error)?.ok_or("no accelerator")?;
    let total = device.memory().map_err(error)?.map_or(0, |(_, total)| total);
    let used = |free: usize| total.saturating_sub(free);
    let free_start = device.memory().map_err(error)?.map_or(0, |(free, _)| free);
    let started = Instant::now();
    let mut compiled = DeviceProgram::compile_values(&device, &prefix)?;
    compiled.set_arithmetic(arithmetic);
    compiled.prepare_dense_parameters(&trainable)?;
    device.synchronize().map_err(error)?;
    let compile_seconds = started.elapsed().as_secs_f64();
    let free_compiled = device.memory().map_err(error)?.map_or(0, |(free, _)| free);
    let width = compiled.widths()[hidden];
    let mut results = Vec::new();
    for &count in &batches {
        let inputs = family(&tokens, count, context);
        let seed = Array2::from_shape_fn((inputs.rows, width), |(r, c)| (((r * 7 + c * 13) % 17) as f64 - 8.0) / 64.0);
        let (mut forward, mut reverse, mut after_forward) = (Vec::new(), Vec::new(), 0usize);
        let sampler = Sampler::start(&device);
        for rep in 0..=reps {
            device.synchronize().map_err(error)?;
            let t = Instant::now();
            let trace = compiled.forward(&inputs)?;
            device.synchronize().map_err(error)?;
            let f = t.elapsed().as_secs_f64();
            after_forward = after_forward.max(used(device.memory().map_err(error)?.map_or(0, |(free, _)| free)));
            let t = Instant::now();
            let seeds = BTreeMap::from([(hidden, device.upload(seed.view()).map_err(error)?)]);
            let (_, gradients) = compiled.vjp_values_dense(&trace, seeds, &[], &trainable, arithmetic)?;
            device.synchronize().map_err(error)?;
            let r = t.elapsed().as_secs_f64();
            if gradients.len() != trainable.len() {
                return Err(format!("{} gradients for {} trainable operators", gradients.len(), trainable.len()));
            }
            if rep > 0 {
                forward.push(f);
                reverse.push(r);
            }
        }
        let peak = used(sampler.finish());
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let row = json!({
            "sequences": count,
            "tokens": inputs.rows,
            "forward_seconds": median(&mut forward),
            "reverse_seconds": median(&mut reverse),
            "used_bytes_after_forward": after_forward,
            "used_bytes_peak": peak,
        });
        log::info!("step bench: {row}");
        results.push(row);
    }
    let report = json!({
        "model": model_dir.display().to_string(),
        "config": record["config"],
        "device": device.name(),
        "arithmetic": format!("{arithmetic:?}"),
        "context": context,
        "trainable_operators": trainable.len(),
        "trainable_parameters": parameters,
        "trace_bytes_per_row": compiled.bytes_per_row(),
        "device_total_bytes": total,
        "used_bytes_before_compile": used(free_start),
        "used_bytes_compiled": used(free_compiled),
        "compile_seconds": compile_seconds,
        "batches": results,
    });
    std::fs::create_dir_all(&out).map_err(error)?;
    let text = serde_json::to_string_pretty(&report).map_err(error)?;
    std::fs::write(out.join("step.json"), &text).map_err(error)?;
    println!("{text}");
    Ok(())
}

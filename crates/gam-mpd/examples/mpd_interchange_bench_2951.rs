//! Time and device memory of one fitting step on interchange experiments (`gam_mpd::interchange`,
//! #2951) on a language model, with the explanation at its start.
//!
//! `mpd_interchange_bench_2951 MODEL SEQUENCES CONTEXT REPS OUT host|device [WINDOWS]`
//!
//! `MODEL` is an engine export (`export.json`; its token rows) or a Hugging Face checkpoint
//! directory (`config.json`; token rows from `WINDOWS`, rows of `CONTEXT` little-endian u32). On an
//! export whose library explanation builds (`library_mdl::explanation`), `P` is that library with
//! every library operator trainable, patched at `M`'s functions (`interchange::reads`). The
//! library does not build on gated MLPs or normed queries and keys (Qwen3); there `P` is the split
//! native program with every head's query, key and value map and both MLP input maps trainable,
//! and its read variables are each head's maps and each MLP unit's gate and input rows: the same
//! shapes as the library's, so the same cost.
//!
//! The first `SEQUENCES` token rows are the bases and the next `SEQUENCES` the sources. Per base the
//! experiments are one unpatched and one patched (`Interchange::sample`, seed 1). Measured, each
//! `REPS` times after one warm-up: `M`'s targets
//! (`interchange::targets`), the evaluation with and without its gradient (`evaluate`), the gradient's download
//! to the host, the whole step
//! (`Interchange::evaluate` with its gradient, downloaded), and for comparison the clean step (`P`
//! alone on the bases, every block its own, with its gradient). On `device` the programs
//! run in f32 storage with f32 products (CUDA, else the Apple GPU), as the fit does. A thread
//! samples the device's free memory every millisecond. One JSON object goes to
//! `OUT/interchange_bench.json` and stdout: per part the median seconds and the least free device
//! bytes seen while it ran.

use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    artifact::Artifact,
    engine::log_to_stderr,
    import::{hugging_face_language_model, import_language_model},
    interchange::{self, Batch, Experiment, Interchange, Patch, ReadVariable},
    library_mdl,
    operator_program::{OperatorProgram, SlotValues},
    run_check::{layer_nodes, split_sites},
};
use rand::SeedableRng;
use serde_json::{Value, json};
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

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

    fn finish(mut self) -> Result<Option<usize>, String> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().map_err(|_| "the memory sampler panicked".to_string())?;
        }
        let least = self.least.load(Ordering::Relaxed);
        Ok((least != usize::MAX).then_some(least))
    }
}

/// `part` run `reps` times after one warm-up: the median seconds and the least free device bytes.
fn measure<T>(device: &Device, reps: usize, mut part: impl FnMut() -> Result<T, String>) -> Result<(Value, T), String> {
    let mut last = part()?;
    device.synchronize().map_err(|e| e.to_string())?;
    let sampler = Sampler::start(device);
    let mut seconds = Vec::with_capacity(reps);
    for _ in 0..reps {
        let started = Instant::now();
        last = part()?;
        device.synchronize().map_err(|e| e.to_string())?;
        seconds.push(started.elapsed().as_secs_f64());
    }
    let least_free = sampler.finish()?;
    seconds.sort_by(f64::total_cmp);
    log::info!("measured: median {} s", seconds[seconds.len() / 2]);
    Ok((json!({"median_seconds": seconds[seconds.len() / 2], "seconds": seconds, "least_free_bytes": least_free}), last))
}

/// The split native program's read variables: each head's query, key and value maps, and per MLP
/// unit its rows of the MLP's input maps (`gate_proj` and `c_fc` when gated); every operator they
/// read through is trainable.
fn native_reads(program: &OperatorProgram, layers: usize) -> Result<(Vec<ReadVariable>, Vec<usize>), String> {
    let named = |name: &str| program.operators.iter().position(|op| op.name == name);
    let (mut variables, mut trainable) = (Vec::new(), Vec::new());
    for l in 0..layers {
        for map in ["q", "k", "v"] {
            for h in 0.. {
                let Some(op) = named(&format!("blocks.{l}.{map}{h}")) else { break };
                variables.push(ReadVariable { block: 2 * l, parts: vec![(op, 0..program.operators[op].rows.width())] });
                trainable.push(op);
            }
        }
        let inputs: Vec<usize> = ["gate_proj", "c_fc"].iter().filter_map(|m| named(&format!("blocks.{l}.{m}"))).collect();
        let units = inputs.first().map(|op| program.operators[*op].rows.width()).ok_or(format!("layer {l}: no MLP input map"))?;
        variables.extend((0..units).map(|i| ReadVariable { block: 2 * l + 1, parts: inputs.iter().map(|op| (*op, i..i + 1)).collect() }));
        trainable.extend(&inputs);
    }
    trainable.sort_unstable();
    Ok((variables, trainable))
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "MODEL SEQUENCES CONTEXT REPS OUT host|device [WINDOWS]";
    let (model, rest) = args.split_first().ok_or(usage)?;
    let [sequences, context, reps, out, mode, windows @ ..] = rest else {
        return Err(usage.into());
    };
    let parse = |s: &str| s.parse::<usize>().map_err(|e| format!("{s}: {e}"));
    let (sequences, context, reps) = (parse(sequences)?, parse(context)?, parse(reps)?);
    if sequences == 0 || context == 0 || reps == 0 {
        return Err("positive sequences, context and reps required".into());
    }
    let device = match mode.as_str() {
        "host" => Device::host(),
        "device" => Device::single_precision(GpuPolicy::Required).map_err(|e| e.to_string())?.ok_or("no accelerator")?,
        _ => return Err(usage.into()),
    };
    let model = Path::new(model);
    let (program, rows, layer_count) = if model.join("config.json").exists() {
        let [windows] = windows else { return Err(format!("a Hugging Face checkpoint needs WINDOWS: {usage}")) };
        let text = std::fs::read_to_string(model.join("config.json")).map_err(|e| e.to_string())?;
        let layers = serde_json::from_str::<Value>(&text).map_err(|e| e.to_string())?["num_hidden_layers"].as_u64().ok_or("num_hidden_layers")? as usize;
        let bytes = std::fs::read(windows).map_err(|e| format!("{windows}: {e}"))?;
        if bytes.len() < 2 * sequences * context * 4 {
            return Err(format!("{windows}: fewer than {} rows of {context} tokens", 2 * sequences));
        }
        let tokens: Vec<u32> = bytes[..2 * sequences * context * 4].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        (hugging_face_language_model(model, 0..layers)?.0, tokens.chunks(context).map(<[u32]>::to_vec).collect::<Vec<_>>(), layers)
    } else {
        let imported = import_language_model(model, 2 * sequences, context)?;
        let layers = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else {
            return Err("a token slot".into());
        };
        (imported.program, tokens.chunks(context).map(<[u32]>::to_vec).collect(), layers)
    };
    let native = split_sites(&program)?;
    let layers = layer_nodes(&native, layer_count)?;
    let batch = Batch::new(rows[..sequences].to_vec(), rows[sequences..2 * sequences].to_vec())?;
    let (artifact, trainable, variables, explanation_kind) = match library_mdl::explanation(&native, &layers) {
        Ok(explanation) => {
            let variables = interchange::reads(&native, &layers)?;
            (explanation.artifact, explanation.trainable, variables, "library")
        }
        Err(reason) => {
            log::info!("no library explanation ({reason}); P is the split native program");
            let (variables, trainable) = native_reads(&native, layer_count)?;
            (Artifact::native(&native)?, trainable, variables, "native")
        }
    };
    let started = Instant::now();
    let x = Interchange::new(&device, &native, &layers, &artifact, &trainable, variables, usize::MAX, 4096)?;
    device.synchronize().map_err(|e| e.to_string())?;
    let compile_seconds = started.elapsed().as_secs_f64();
    let free_compiled = device.memory().map_err(|e| e.to_string())?.map(|(free, _)| free);
    let (m, p) = x.models();
    let head = x.head();
    let variables = x.variables();
    let experiments = interchange::sample(&mut rand::rngs::StdRng::seed_from_u64(1), sequences, variables, 2 * layer_count, context)?;
    let clean: Vec<Experiment> = (0..sequences).map(|n| Experiment { base: n, source: n, explained: vec![true; 2 * layer_count], patch: None, position: 0 }).collect();
    let (teacher_time, targets) = measure(&device, reps, || interchange::targets(&m, head, &batch, &experiments))?;
    let (gradient_time, evaluation) = measure(&device, reps, || interchange::evaluate(&m, &p, head, &batch, &targets, &experiments, true))?;
    let (forward_time, values) = measure(&device, reps, || interchange::evaluate(&m, &p, head, &batch, &targets, &experiments, false))?;
    let (download_time, _) = measure(&device, reps, || evaluation.gradient.values().map(|g| device.download(g).map_err(|e| e.to_string())).collect::<Result<Vec<_>, String>>())?;
    drop(evaluation);
    let (step_time, _) = measure(&device, reps, || x.evaluate(&batch, &experiments, true))?;
    let (clean_time, clean_values) = measure(&device, reps, || x.evaluate(&batch, &clean, true))?;
    let mean = |bits: &[Vec<f64>]| bits.iter().flatten().sum::<f64>() / bits.iter().map(Vec::len).sum::<usize>().max(1) as f64;
    let patched_bits: Vec<Vec<f64>> = experiments.iter().zip(&values.bits).filter(|(e, _)| e.patch.is_some()).map(|(_, b)| b.clone()).collect();
    let report = json!({
        "model": model.display().to_string(),
        "explanation": explanation_kind,
        "device": device.name(),
        "arithmetic": format!("{:?}", p.program.arithmetic()),
        "sequences": sequences,
        "context": context,
        "layers": layer_count,
        "read_variables": variables.len(),
        "trainable_operators": trainable.len(),
        "experiments": experiments.len(),
        "patched_experiments": patched_bits.len(),
        "hybrid_sizes": experiments.iter().map(|e| e.explained.iter().filter(|x| **x).count()).collect::<Vec<_>>(),
        "patched_blocks": experiments.iter().filter_map(|e| match &e.patch {
            Some(Patch::Read { variable }) => Some(variables[*variable].block),
            Some(Patch::Reads { variables: chosen }) => chosen.first().map(|v| variables[*v].block),
            Some(Patch::Part { .. } | Patch::Head { .. } | Patch::Cut { .. } | Patch::Parts { .. } | Patch::Swap { .. } | Patch::PartFrom { .. } | Patch::HeadFrom { .. }) | None => None,
        }).collect::<Vec<_>>(),
        "compile_seconds": compile_seconds,
        "free_bytes_compiled": free_compiled,
        "families": interchange::census(&experiments, variables),
        "targets": teacher_time,
        "evaluate": forward_time,
        "evaluate_with_gradient": gradient_time,
        "download_gradient": download_time,
        "step_with_gradient": step_time,
        "clean_step_with_gradient": clean_time,
        "mean_bits_per_token": mean(&values.bits),
        "mean_patched_bits_per_token": mean(&patched_bits),
        "mean_clean_bits_per_token": mean(&clean_values.bits),
    });
    println!("{report}");
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    std::fs::write(Path::new(out).join("interchange_bench.json"), serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

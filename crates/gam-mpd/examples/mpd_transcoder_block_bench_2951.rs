//! Time and device memory of one library-fit step through a transcoder block (#2951): the block
//! `x ↦ relu(x Gᵀ + c) Wᵀ` of a transcoder's first `FEATURES` features (`library_transcoder`) on
//! `M`'s own MLP input at layer `LAYER`, its output replaced by its input at each sequence's first
//! token (the attention sink, where the library runs `M`'s MLP), every feature's gate row with its
//! bias one prior group and its output column another.
//!
//! `mpd_transcoder_block_bench_2951 MODEL WINDOWS TRANSCODERS LAYER FEATURES SEQUENCES CONTEXT REPS OUT`
//!
//! `MODEL` is a Hugging Face checkpoint directory, `WINDOWS` rows of `CONTEXT` little-endian u32
//! tokens (the first `SEQUENCES` are the batch), `TRANSCODERS` holds `layer_{LAYER}.safetensors`.
//! A step, as the fit takes it, on the single-precision device: the weight sample written into the
//! block (`DevicePosterior::sample_into`), the forward pass in f32, two reverse passes into `G`, `c`
//! and `W` in bfloat16 on CUDA, f32 on the Apple GPU (the data term's and the Gauss–Newton
//! factor's), and the IVON step. Each
//! part is timed to a device synchronization, `REPS` times after one warm-up, first with the
//! products reading only the features nonzero on the batch (the default) and then reading every
//! feature (`DeviceProgram::read_densely`). A thread samples the device's free memory every
//! millisecond. One JSON object goes to `OUT/transcoder_block_bench.json` and stdout: per arm the
//! median seconds per part, the device bytes at the peak and before the first step, and the
//! features read per step.

use gam_gpu::{
    GpuPolicy,
    tensor::{Arithmetic, Device},
};
use gam_mpd::{
    device_posterior::{DevicePosterior, Ivon, Parts},
    device_program::DeviceProgram,
    engine::log_to_stderr,
    import::hugging_face_language_model,
    library_mdl::sequence_family,
    library_transcoder::Transcoder,
    operator_program::{Declarations, FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorProgram, Provenance, SequenceLayout, Slot, SlotValues},
    run_check::{layer_nodes, split_sites},
    safetensors::SafetensorsFile,
};
use ndarray::Array2;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Seconds of `part` to a device synchronization.
fn timed<T>(device: &Device, seconds: &mut BTreeMap<&'static str, Vec<f64>>, name: &'static str, part: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    let started = Instant::now();
    let out = part()?;
    device.synchronize().map_err(error)?;
    seconds.entry(name).or_default().push(started.elapsed().as_secs_f64());
    Ok(out)
}

fn medians(seconds: &BTreeMap<&'static str, Vec<f64>>) -> Value {
    let mut out = serde_json::Map::new();
    let mut total = 0.0;
    for (name, values) in seconds {
        let mut v = values[1..].to_vec();
        v.sort_by(f64::total_cmp);
        let median = v[v.len() / 2];
        total += median;
        out.insert((*name).to_string(), json!(median));
    }
    out.insert("step".into(), json!(total));
    Value::Object(out)
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "MODEL WINDOWS TRANSCODERS LAYER FEATURES SEQUENCES CONTEXT REPS OUT";
    let [model, windows, transcoders, layer, features, sequences, context, reps, out] = &args[..] else {
        return Err(usage.into());
    };
    let parse = |s: &str| s.parse::<usize>().map_err(|e| format!("{s}: {e}"));
    let (layer, features, sequences, context, reps) = (parse(layer)?, parse(features)?, parse(sequences)?, parse(context)?, parse(reps)?);
    if sequences == 0 || context == 0 || reps == 0 || features == 0 {
        return Err("positive features, sequences, context and reps required".into());
    }
    std::fs::create_dir_all(out).map_err(error)?;
    let device = Device::single_precision(GpuPolicy::Required).map_err(error)?.ok_or("no accelerator")?;
    let bytes = std::fs::read(windows).map_err(|e| format!("{windows}: {e}"))?;
    if bytes.len() < sequences * context * 4 {
        return Err(format!("{windows}: fewer than {sequences} rows of {context} tokens"));
    }
    let tokens: Vec<u32> = bytes[..sequences * context * 4].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let rows: Vec<&[u32]> = tokens.chunks(context).collect();
    // M's MLP input at the layer.
    let model = Path::new(model);
    let text = std::fs::read_to_string(model.join("config.json")).map_err(error)?;
    let all = serde_json::from_str::<Value>(&text).map_err(error)?["num_hidden_layers"].as_u64().ok_or("num_hidden_layers")? as usize;
    let native = split_sites(&hugging_face_language_model(model, 0..all)?.0)?;
    let layers = layer_nodes(&native, all)?;
    let normed = layers.get(layer).ok_or(format!("layer {layer} of {all}"))?.normed;
    let x = {
        let mut m = DeviceProgram::compile(&device, &native)?;
        m.set_arithmetic(Arithmetic::F32);
        let family = sequence_family(&rows)?;
        let trace = m.forward_span(&family, None, normed, |_, _| Ok(None))?;
        device.download(trace.value(normed)?).map_err(error)?
    };
    let (n, d) = x.dim();
    // The block of the first FEATURES features.
    let transcoder = Transcoder::open(&Path::new(transcoders).join(format!("layer_{layer}.safetensors")))?;
    let kept: Vec<usize> = (0..features.min(transcoder.features)).collect();
    let k = kept.len();
    let file = Path::new(out).join(format!("kept_{layer}_{k}.safetensors"));
    transcoder.write_kept(&kept, &file)?;
    let stored = SafetensorsFile::open(&file).map_err(error)?;
    let units = Interface::uniform(k, 1, LabelKind::Unit, 0).map_err(error)?;
    let input = Interface::native(d).map_err(error)?;
    let operator = |name: &str, rows: Interface, cols: Interface, (r, c): (usize, usize)| -> Result<Arc<Operator>, String> {
        Ok(Arc::new(Operator::stored(name, rows, cols, stored.stored(name, r, c).map_err(error)?, Provenance::native("transcoder")).map_err(error)?))
    };
    let block = OperatorProgram {
        declarations: Declarations { domains: vec![], slots: vec![Slot::Raw { width: d }], parameters: 0 },
        operators: vec![
            operator("gate", units.clone(), input.clone(), (k, d))?,
            operator("gate_bias", units.clone(), Interface::constant(), (k, 1))?,
            operator("out", input, units, (d, k))?,
        ],
        bases: vec![],
        rules: vec![],
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: Some(1) },
            Node::Pointwise { input: 1, laws: vec![Law::Relu; k] },
            Node::Affine { terms: vec![(2, 2)], bias: None },
            Node::Select { inside: 0, outside: 3, positions: vec![0] },
        ],
        output: 4,
    };
    let family = FamilyInputs {
        rows: n,
        slots: vec![SlotValues::Raw(x)],
        layout: Some(SequenceLayout { sequence: (0..n as u32).map(|r| r / context as u32).collect(), position: (0..n as u32).map(|r| r % context as u32).collect() }),
    };
    // The posterior's start: the transcoder's weights, ln σ at each group's mean square over the
    // batch's tokens (`library_mdl::Posterior::new`'s start), one group per gate row with its bias
    // and one per output column.
    let trainable = [0usize, 1, 2];
    let mean: Vec<Array2<f64>> = block.operators.iter().map(|op| op.matrix().into_owned()).collect();
    let mut groups: Vec<Vec<u32>> = Vec::new();
    groups.push((0..k * d).map(|i| (i / d) as u32).collect());
    groups.push((0..k).map(|i| i as u32).collect());
    groups.push((0..d * k).map(|i| (k + i % k) as u32).collect());
    let mut square = vec![0.0f64; 2 * k];
    let mut count = vec![0usize; 2 * k];
    for (m, g) in mean.iter().zip(&groups) {
        for (v, &group) in m.iter().zip(g) {
            square[group as usize] += v * v;
            count[group as usize] += 1;
        }
    }
    let log_sd: Vec<Array2<f64>> = mean
        .iter()
        .zip(&groups)
        .map(|(m, g)| Array2::from_shape_vec(m.dim(), g.iter().map(|&group| 0.5 * ((square[group as usize] / count[group as usize] as f64).max(f64::MIN_POSITIVE) / n as f64).ln()).collect()).map_err(error))
        .collect::<Result<_, String>>()?;
    let parameters: usize = mean.iter().map(Array2::len).sum();
    let seed = |s: usize| device.upload(Array2::from_shape_fn((n, d), |(r, c)| (((r * 7 + c * 13 + s) % 17) as f64 - 8.0) / 64.0).view()).map_err(error);
    // The reverse passes' products in bfloat16 where the device has them (CUDA), else in f32.
    let reverse = if cfg!(target_os = "linux") { Arithmetic::Bf16 } else { Arithmetic::F32 };
    let total = device.memory().map_err(error)?.map_or(0, |(_, total)| total);
    let mut arms = Vec::new();
    for dense in [false, true] {
        let (least, stop) = (Arc::new(AtomicUsize::new(usize::MAX)), Arc::new(AtomicBool::new(false)));
        let sampler = {
            let (device, least, stop) = (device.clone(), Arc::clone(&least), Arc::clone(&stop));
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    if let Ok(Some((free, _))) = device.memory() {
                        least.fetch_min(free, Ordering::Relaxed);
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
            })
        };
        let mut program = DeviceProgram::compile_values(&device, &block)?;
        program.set_arithmetic(Arithmetic::F32);
        program.prepare_dense_parameters(&trainable)?;
        if dense {
            program.read_densely();
        }
        let parts = Parts { operators: &trainable, mean: &mean, log_sd: &log_sd, groups: &groups, count: 2 * k, reference: None };
        let mut posterior = DevicePosterior::from_parts(&device, &parts, n as f64, None, 0)?;
        posterior.hold_means(true);
        device.synchronize().map_err(error)?;
        let before = total.saturating_sub(device.memory().map_err(error)?.map_or(0, |(free, _)| free));
        let ivon = Ivon { beta1: 0.99, beta2: 0.999 };
        let mut seconds: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
        let mut read = Vec::new();
        for step in 0..=reps {
            let s = &mut seconds;
            timed(&device, s, "sample", || posterior.sample_into(&mut program, step as u64))?;
            let trace = timed(&device, s, "forward", || program.forward(&family))?;
            read.push(program.columns_read(&trace, 3, 2).unwrap_or(k));
            let (_, gradient) = timed(&device, s, "reverse_gradient", || program.vjp_values_dense(&trace, BTreeMap::from([(4, seed(0)?)]), &[], &trainable, reverse))?;
            let (_, factor) = timed(&device, s, "reverse_factor", || program.vjp_values_dense(&trace, BTreeMap::from([(4, seed(5)?)]), &[], &trainable, reverse))?;
            drop(trace);
            timed(&device, s, "posterior_step", || posterior.step(&gradient, 1.0 / n as f64, (&factor, 1.0 / n as f64), &BTreeMap::new(), &ivon))?;
        }
        stop.store(true, Ordering::Relaxed);
        sampler.join().map_err(|_| "the memory sampler panicked")?;
        let peak = total.saturating_sub(least.load(Ordering::Relaxed));
        let arm = json!({
            "reads": if dense { "dense" } else { "sparse" },
            "seconds": medians(&seconds),
            "device_bytes_peak": peak,
            "device_bytes_before_steps": before,
            "features_read_per_step": read,
        });
        log::info!("{arm}");
        arms.push(arm);
    }
    let report = json!({
        "model": model.display().to_string(),
        "device": device.name(),
        "layer": layer,
        "features": k,
        "sequences": sequences,
        "context": context,
        "rows": n,
        "parameters": parameters,
        "reps": reps,
        "arms": arms,
    });
    println!("{report}");
    std::fs::write(Path::new(out).join("transcoder_block_bench.json"), serde_json::to_vec_pretty(&report).map_err(error)?).map_err(error)
}

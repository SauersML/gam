//! Time of one library-fit training step (#2951), its posterior resident on the device
//! (`device_posterior`).
//!
//! `mpd_library_step_bench_2951 MODEL SEQUENCES CONTEXT REPS OUT [WINDOWS [LAYERS]]`
//!
//! `MODEL` is an engine export (its token rows) or a Hugging Face checkpoint directory (token rows
//! from `WINDOWS`, rows of `CONTEXT` little-endian u32; its first `LAYERS` layers, all when
//! omitted, so a device or host too small for the whole model still measures per-layer costs). `P` is the library explanation when it
//! builds, its prior groups the library's; otherwise (gated MLPs, normed queries and keys: Qwen3)
//! the split native program with every head's query, key and value map and the MLP's input maps
//! trainable, each operator row one prior group (the library's plane, value and gate groups are
//! rows of the same shapes). The first `SEQUENCES` rows are the bases, the next the sources; one
//! clean and one patched experiment per base (`interchange::sample`, seed 1). The programs run on
//! the single-precision device (CUDA in f32 storage, else the Apple GPU), with f32 products.
//!
//! A step, as the fit takes it: the patch directions at the posterior mean (the means written into
//! `P`, `interchange::design`), the weight sample written into `P`, `M`'s clean runs
//! (`interchange::targets`), the experiments with the gradient (`interchange::evaluate`), the
//! description `Σ_G KL_G`, and the IVON step (`Device::posterior_ivon`); none of the parameters
//! leave the device. The step runs `REPS` times after one warm-up, each part timed to a device
//! synchronization. One JSON object goes to `OUT/library_step_bench.json` and stdout: per part the
//! median seconds.

use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    artifact::Artifact,
    decoder::Decoder,
    device_posterior::{DevicePosterior, Ivon, Parts},
    device_program::DeviceProgram,
    engine::log_to_stderr,
    import::{hugging_face_language_model, import_language_model},
    interchange::{self, Batch, BlockEngine, FixedHead, Model, ReadVariable},
    library_mdl,
    operator_program::{OperatorProgram, SlotValues},
    run_check::{layer_nodes, split_sites},
};
use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, time::Instant};

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// The split native program's read variables and trainable operators (each head's query, key and
/// value maps, the MLP's input maps), as `mpd_interchange_bench_2951` takes them.
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

/// The posterior's host arrays at the start, per trainable operator: `μ` (the operator), `ln σ`,
/// each entry's group, and the group count.
struct Start {
    mean: Vec<Array2<f64>>,
    log_sd: Vec<Array2<f64>>,
    groups: Vec<Vec<u32>>,
    count: usize,
}

/// Each operator row one group; `ln σ` at the group's mean square over `tokens`
/// (`library_mdl::Posterior::new`'s start).
fn row_groups(program: &OperatorProgram, trainable: &[usize], tokens: usize) -> Start {
    let mut start = Start { mean: Vec::new(), log_sd: Vec::new(), groups: Vec::new(), count: 0 };
    for &op in trainable {
        let mean = program.operators[op].matrix();
        let (rows, cols) = mean.dim();
        let log_sd = Array2::from_shape_fn((rows, cols), |(r, _)| {
            let square = mean.row(r).iter().map(|v| v * v).sum::<f64>() / cols as f64;
            0.5 * (square.max(f64::MIN_POSITIVE) / tokens as f64).ln()
        });
        start.groups.push((0..rows * cols).map(|i| (start.count + i / cols) as u32).collect());
        start.count += rows;
        start.mean.push(mean);
        start.log_sd.push(log_sd);
    }
    start
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
    // `products=f32|tf32|bf16` (f32 by default) sets the arithmetic of both programs' products;
    // `engine=decoder` runs the experiments' blocks on the fixed decoder computations
    // (`gam_mpd::decoder`, bfloat16 products) instead of the programs.
    let (settings, args): (Vec<String>, Vec<String>) = std::env::args().skip(1).partition(|a| a.contains('='));
    let (mut products, mut fixed) = (gam_gpu::tensor::Arithmetic::F32, false);
    for setting in &settings {
        match setting.as_str() {
            "products=f32" => products = gam_gpu::tensor::Arithmetic::F32,
            "products=tf32" => products = gam_gpu::tensor::Arithmetic::Tf32,
            "products=bf16" => products = gam_gpu::tensor::Arithmetic::Bf16,
            "engine=program" => fixed = false,
            "engine=decoder" => fixed = true,
            other => return Err(format!("unknown setting {other}")),
        }
    }
    let usage = "MODEL SEQUENCES CONTEXT REPS OUT [WINDOWS [LAYERS]] [products=f32|tf32|bf16] [engine=program|decoder]";
    let (model, rest) = args.split_first().ok_or(usage)?;
    let [sequences, context, reps, out, windows @ ..] = rest else {
        return Err(usage.into());
    };
    let parse = |s: &str| s.parse::<usize>().map_err(|e| format!("{s}: {e}"));
    let (sequences, context, reps) = (parse(sequences)?, parse(context)?, parse(reps)?);
    if sequences == 0 || context == 0 || reps == 0 {
        return Err("positive sequences, context and reps required".into());
    }
    let device = Device::single_precision(GpuPolicy::Required).map_err(error)?.ok_or("no accelerator")?;
    let model = Path::new(model);
    let (program, rows, layer_count) = if model.join("config.json").exists() {
        let (windows, kept) = match windows {
            [windows] => (windows, None),
            [windows, kept] => (windows, Some(parse(kept)?)),
            _ => return Err(format!("a Hugging Face checkpoint needs WINDOWS: {usage}")),
        };
        let text = std::fs::read_to_string(model.join("config.json")).map_err(error)?;
        let all = serde_json::from_str::<Value>(&text).map_err(error)?["num_hidden_layers"].as_u64().ok_or("num_hidden_layers")? as usize;
        let layers = kept.unwrap_or(all).min(all);
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
    // The training tokens a fit of this batch size weighs a batch by: the batch's own here.
    let tokens = 2 * sequences * context;
    let (artifact, trainable, variables, start, kind) = match library_mdl::explanation(&native, &layers) {
        Ok(explanation) => {
            let variables = interchange::library_reads(&explanation.artifact.program, layer_count)?;
            let posterior = library_mdl::Posterior::new(&explanation, tokens)?;
            let shapes: Vec<usize> = posterior.mean.iter().map(Array2::len).collect();
            let mut groups: Vec<Vec<u32>> = shapes.iter().map(|n| vec![0; *n]).collect();
            let position: BTreeMap<usize, usize> = explanation.trainable.iter().enumerate().map(|(i, op)| (*op, i)).collect();
            for (g, group) in explanation.groups.iter().enumerate() {
                for cell in &group.cells {
                    let i = position[&cell.operator];
                    let cols = posterior.mean[i].ncols();
                    for &r in &cell.rows {
                        for c in cell.cols.clone() {
                            groups[i][r * cols + c] = g as u32;
                        }
                    }
                }
            }
            let start = Start { mean: posterior.mean.clone(), log_sd: posterior.log_sd.clone(), groups, count: explanation.groups.len() };
            (explanation.artifact, explanation.trainable, variables, start, "library")
        }
        Err(reason) => {
            log::info!("no library explanation ({reason}); P is the split native program");
            let (variables, trainable) = native_reads(&native, layer_count)?;
            let start = row_groups(&native, &trainable, tokens);
            (Artifact::native(&native)?, trainable, variables, start, "native")
        }
    };
    let parameters: usize = start.mean.iter().map(Array2::len).sum();
    log::info!("{kind} P: {} trainable operators, {parameters} parameters, {} groups", trainable.len(), start.count);
    let (m_flat, m_streams, m_reads) = interchange::sites(&Artifact::native(&native)?, &layers)?;
    let (p_flat, p_streams, p_reads) = interchange::sites(&artifact, &layers)?;
    let mut m_program = DeviceProgram::compile_values_bounded(&device, &interchange::prefix(&m_flat)?, usize::MAX)?;
    m_program.set_arithmetic(products);
    let mut p_program = DeviceProgram::compile_values_bounded(&device, &interchange::prefix(&p_flat)?, usize::MAX)?;
    p_program.set_arithmetic(products);
    p_program.prepare_dense_parameters(&trainable)?;
    // Bfloat16 products read bfloat16 copies of the weights (`DeviceProgram::hold_bf16`).
    if products == gam_gpu::tensor::Arithmetic::Bf16 {
        m_program.hold_bf16()?;
        p_program.hold_bf16()?;
    }
    // Where the experiments cannot be scored (a head the compact targets do not take), the posterior
    // parts are timed alone, from a zero gradient on the device.
    let scoring = match FixedHead::new(&device, &m_flat, &p_flat, 4096) {
        Ok(head) => Some((head, Model::new(&m_program, &m_flat, m_streams.clone(), m_reads.clone(), &[])?)),
        Err(reason) => {
            log::info!("the experiments are not scored ({reason}); the posterior parts are timed from a zero gradient");
            None
        }
    };
    let zeros: BTreeMap<usize, gam_gpu::tensor::Tensor> =
        trainable.iter().zip(&start.mean).map(|(op, m)| Ok((*op, device.zeros(m.nrows(), m.ncols()).map_err(error)?))).collect::<Result<_, String>>()?;
    let experiments = interchange::sample(&mut StdRng::seed_from_u64(1), sequences, &variables, 2 * layer_count, context)?;
    let ivon = Ivon { rate: 0.1, beta1: 0.9, beta2: 0.999 };
    // The gradient of the data term per scored token, in nats.
    let scale = std::f64::consts::LN_2 / (experiments.len() * context) as f64;
    fn p_model<'a>(program: &'a DeviceProgram, (flat, streams, reads, trainable): (&OperatorProgram, &[usize], &[usize], &[usize])) -> Result<Model<'a>, String> {
        Model::new(program, flat, streams.to_vec(), reads.to_vec(), trainable)
    }
    let sites = (&p_flat, &p_streams[..], &p_reads[..], &trainable[..]);
    let decoders = if fixed {
        let (m_prefix, p_prefix) = (interchange::prefix(&m_flat)?, interchange::prefix(&p_flat)?);
        let m_ends = (&m_streams[..], &m_reads[..], m_program.hidden());
        let outside = if products == gam_gpu::tensor::Arithmetic::F32 { gam_gpu::tensor::Arithmetic::Bf16 } else { products };
        let m = Decoder::new(&device, &m_prefix, m_ends, &[])?.with_arithmetic(outside);
        Some((m, Decoder::new(&device, &p_prefix, (&p_streams[..], &p_reads[..], p_program.hidden()), &trainable)?.with_arithmetic(outside)))
    } else {
        None
    };
    let mut decoders = decoders;

    // The device path.
    let mut device_seconds: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    // The last step's mean KL(M_e ‖ P_e) per scored token, in bits (a check that the products'
    // arithmetic did not change what is computed).
    let mut mean_bits = None;
    {
        let parts = Parts { operators: &trainable, mean: &start.mean, log_sd: &start.log_sd, groups: &start.groups, count: start.count };
        let mut posterior = DevicePosterior::from_parts(&device, &parts, tokens as f64, None, 0)?;
        for step in 0..=reps {
            let s = &mut device_seconds;
            timed(&device, s, "means_loaded", || posterior.mean_into(&mut p_program))?;
            let design = match &scoring {
                Some(_) => Some(timed(&device, s, "design", || interchange::design(&p_model(&p_program, sites)?, &variables, &experiments))?),
                None => None,
            };
            timed(&device, s, "sample_loaded", || {
                posterior.sample_into(&mut p_program, step as u64)?;
                decoders.as_mut().map_or(Ok(()), |(_, p)| p.refresh(&p_program))
            })?;
            let gradient = match (&scoring, &design) {
                (Some((head, m)), Some(design)) => {
                    let evaluation = match &decoders {
                        Some((m, p)) => {
                            let targets = timed(&device, s, "teacher", || interchange::targets(m, head, &batch, &experiments, design))?;
                            timed(&device, s, "evaluate", || interchange::evaluate(m, p, head, &batch, &targets, &experiments, design, true))?
                        }
                        None => {
                            let targets = timed(&device, s, "teacher", || interchange::targets(m, head, &batch, &experiments, design))?;
                            timed(&device, s, "evaluate", || interchange::evaluate(m, &p_model(&p_program, sites)?, head, &batch, &targets, &experiments, design, true))?
                        }
                    };
                    let tokens = evaluation.bits.iter().map(Vec::len).sum::<usize>();
                    mean_bits = Some(evaluation.bits.iter().flatten().sum::<f64>() / tokens as f64);
                    evaluation.gradient
                }
                _ => zeros.iter().map(|(op, z)| Ok((*op, device.copy(z).map_err(error)?))).collect::<Result<BTreeMap<_, _>, String>>()?,
            };
            timed(&device, s, "description", || Ok(posterior.divergences()?.iter().sum::<f64>()))?;
            timed(&device, s, "posterior_step", || posterior.step(&gradient, scale, &ivon, step as u64))?;
        }
    }
    // The decoder's blocks alone, forward with tapes then reverse, on the bases (no head, no
    // patches): the engine's own time and product rate.
    let blocks_seconds = match &decoders {
        Some((_, p)) => {
            let mut seconds: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
            let length = context;
            let ranges: Vec<std::ops::Range<usize>> = (0..sequences).map(|i| i * length..(i + 1) * length).collect();
            let tokens: Vec<&[u32]> = rows[..sequences].iter().map(Vec::as_slice).collect();
            let mut gradient = BTreeMap::new();
            for _ in 0..=reps {
                let s = &mut seconds;
                let mut stream = device.zeros(sequences * length, p.width()).map_err(error)?;
                let tapes = timed(&device, s, "blocks_forward", || {
                    (0..p.blocks()).map(|b| p.forward(b, &mut stream, &ranges, &tokens, None, true)?.ok_or_else(|| "no tape".to_string())).collect::<Result<Vec<_>, String>>()
                })?;
                let mut cotangent = device.copy(&stream).map_err(error)?;
                timed(&device, s, "blocks_reverse", || {
                    for (b, tape) in tapes.iter().enumerate().rev() {
                        p.reverse(b, tape, &mut cotangent, &ranges, None, &mut gradient)?;
                    }
                    Ok(())
                })?;
            }
            let (forward, reverse) = p.product_flops(sequences * length, length);
            let medians = medians(&seconds);
            let rate = |part: &str, flops: f64| medians[part].as_f64().map(|t| flops / t / 1e12);
            Some(json!({"seconds": medians, "forward_tflops": rate("blocks_forward", forward), "reverse_tflops": rate("blocks_reverse", reverse), "forward_flops": forward, "reverse_flops": reverse}))
        }
        None => None,
    };
    let report = json!({
        "decoder_blocks": blocks_seconds,
        "model": model.display().to_string(),
        "explanation": kind,
        "products": format!("{products:?}"),
        "engine": if fixed { "decoder" } else { "program" },
        "fused_head_groups": [m_program.fused_groups(), p_program.fused_groups()],
        "scored": scoring.is_some(),
        "mean_bits_per_token": mean_bits,
        "device": device.name(),
        "sequences": sequences,
        "context": context,
        "layers": layer_count,
        "experiments": experiments.len(),
        "trainable_operators": trainable.len(),
        "parameters": parameters,
        "groups": start.count,
        "reps": reps,
        "device_posterior_seconds": medians(&device_seconds),
    });
    println!("{report}");
    std::fs::create_dir_all(out).map_err(error)?;
    std::fs::write(Path::new(out).join("library_step_bench.json"), serde_json::to_vec_pretty(&report).map_err(error)?).map_err(error)
}

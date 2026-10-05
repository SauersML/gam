//! Time of one library-fit training step (#2951), its posterior on the host as the fit kept it
//! before (`library_mdl`'s `Posterior`) against resident on the device (`device_posterior`).
//!
//! `mpd_library_step_bench_2951 MODEL SEQUENCES CONTEXT REPS OUT [WINDOWS]`
//!
//! `MODEL` is an engine export (its token rows) or a Hugging Face checkpoint directory (token rows
//! from `WINDOWS`, rows of `CONTEXT` little-endian u32). `P` is the library explanation when it
//! builds, its prior groups the library's; otherwise (gated MLPs, normed queries and keys: Qwen3)
//! the split native program with every head's query, key and value map and the MLP's input maps
//! trainable, each operator row one prior group (the library's plane, value and gate groups are
//! rows of the same shapes). The first `SEQUENCES` rows are the bases, the next the sources; one
//! clean and one patched experiment per base (`interchange::sample`, seed 1). The programs run on
//! the single-precision device (CUDA in f32 storage, else the Apple GPU), with f32 products.
//!
//! A step, as the fit takes it: the patch directions at the posterior mean (the means written into
//! `P`, `interchange::design`), the weight sample written into `P`, `M`'s clean runs
//! (`interchange::Teacher`), the experiments with the gradient (`interchange::evaluate`), the
//! description `Σ_G KL_G`, and Adam's step in `(μ, ln σ)`. On the host path the means and the
//! sample are drawn and uploaded from float64 host arrays, the gradient downloaded, and the
//! derivatives, Adam's step and the description computed on the host (rayon), as `library_mdl`
//! does; on the device path none of the parameters leave the device. Each path's step runs `REPS`
//! times after one warm-up, each part timed to a device synchronization. One JSON object goes to
//! `OUT/library_step_bench.json` and stdout: per path and part the median seconds.

use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    artifact::Artifact,
    device_posterior::{Adam, DevicePosterior, Parts},
    device_program::DeviceProgram,
    engine::log_to_stderr,
    import::{hugging_face_language_model, import_language_model},
    interchange::{self, Batch, FixedHead, Model, ReadVariable, Teacher},
    library_mdl,
    operator_program::{OperatorProgram, SlotValues},
    run_check::{layer_nodes, split_sites},
};
use ndarray::{Array2, Zip};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
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

/// Load host arrays into `P`'s trainable operators (`Interchange::load`).
fn load(program: &mut DeviceProgram, trainable: &[usize], values: &[Array2<f64>]) -> Result<(), String> {
    for (&op, value) in trainable.iter().zip(values) {
        let tensor = program.device().upload(value.view()).map_err(error)?;
        program.replace_dense_parameter(op, tensor)?;
    }
    Ok(())
}

/// The host posterior's weight sample and noise from `seed` (`library_mdl::Posterior::sample`).
fn host_sample(mean: &[Array2<f64>], log_sd: &[Array2<f64>], seed: u64) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
    (0..mean.len())
        .into_par_iter()
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let n = mean[i].len();
            let mut values = Vec::with_capacity(n + 1);
            while values.len() < n {
                let u = 1.0 - rng.random::<f64>();
                let angle = std::f64::consts::TAU * rng.random::<f64>();
                let radius = (-2.0 * u.ln()).sqrt();
                values.push(radius * angle.cos());
                values.push(radius * angle.sin());
            }
            values.truncate(n);
            let noise = Array2::from_shape_vec(mean[i].dim(), values).expect("shape of the drawn values");
            let mut theta = mean[i].clone();
            Zip::from(&mut theta).and(&noise).and(&log_sd[i]).for_each(|t, e, s| *t += s.exp() * e);
            (theta, noise)
        })
        .unzip()
}

/// The host posterior's group moments `(n, Σ μ² + σ², Σ 2s)` (`library_mdl::Posterior::moments`).
fn host_moments(mean: &[Array2<f64>], log_sd: &[Array2<f64>], groups: &[Vec<u32>], count: usize) -> Vec<[f64; 3]> {
    let partial: Vec<Vec<[f64; 3]>> = (0..mean.len())
        .into_par_iter()
        .map(|i| {
            let mut local = vec![[0.0; 3]; count];
            for ((mu, s), g) in mean[i].iter().zip(log_sd[i].iter()).zip(&groups[i]) {
                let m = &mut local[*g as usize];
                m[0] += 1.0;
                m[1] += mu * mu + (2.0 * s).exp();
                m[2] += 2.0 * s;
            }
            local
        })
        .collect();
    let mut out = vec![[0.0; 3]; count];
    for local in partial {
        for (o, l) in out.iter_mut().zip(local) {
            o[0] += l[0];
            o[1] += l[1];
            o[2] += l[2];
        }
    }
    out
}

/// The host posterior as `library_mdl` keeps it: per operator `μ`, `ln σ`, Adam's moments and each
/// entry's group.
struct HostPosterior {
    mean: Vec<Array2<f64>>,
    log_sd: Vec<Array2<f64>>,
    moments: Vec<[Array2<f64>; 4]>,
    groups: Vec<Array2<u32>>,
    count: usize,
}

/// One Adam step of `value` (`library_mdl`'s `Moment::step`).
fn adam_step(value: &mut Array2<f64>, gradient: &Array2<f64>, (first, second): (&mut Array2<f64>, &mut Array2<f64>), rate: f64, adam: &Adam, step: i32) {
    let (c1, c2) = (1.0 - adam.beta1.powi(step), 1.0 - adam.beta2.powi(step));
    Zip::from(value).and(gradient).and(first).and(second).for_each(|w, g, m, v| {
        *m = adam.beta1 * *m + (1.0 - adam.beta1) * g;
        *v = adam.beta2 * *v + (1.0 - adam.beta2) * g * g;
        *w -= rate * (*m / c1) / ((*v / c2).sqrt() + adam.epsilon);
    });
}

impl HostPosterior {
    /// The description `Σ_G KL_G` in nats, then the derivatives and Adam's step
    /// (`library_mdl::Posterior::description`, `derivatives`, `Moment::step`).
    fn step(&mut self, gradients: &[Array2<f64>], noise: &[Array2<f64>], adam: &Adam, step: i32) -> f64 {
        let flat: Vec<Vec<u32>> = self.groups.iter().map(|g| g.iter().copied().collect()).collect();
        let sums = host_moments(&self.mean, &self.log_sd, &flat, self.count);
        let description: f64 = sums.iter().filter(|m| m[0] > 0.0).map(|m| 0.5 * (m[0] * (m[1] / m[0]).ln() - m[2])).sum();
        let variance: Vec<f64> = host_moments(&self.mean, &self.log_sd, &flat, self.count).iter().map(|m| if m[0] > 0.0 { m[1] / m[0] } else { 0.0 }).collect();
        let derivatives: Vec<(Array2<f64>, Array2<f64>)> = (0..gradients.len())
            .into_par_iter()
            .map(|i| {
                let (mut gm, mut gs) = (gradients[i].clone(), Array2::zeros(gradients[i].dim()));
                Zip::from(&mut gm).and(&mut gs).and(&noise[i]).and(&self.mean[i]).and(&self.log_sd[i]).and(&self.groups[i]).for_each(|gm, gs, e, mu, s, g| {
                    let v = variance[*g as usize];
                    let sd = s.exp();
                    *gs = *gm * e * sd + sd * sd / v - 1.0;
                    *gm += mu / v;
                });
                (gm, gs)
            })
            .collect();
        self.mean.par_iter_mut().zip(self.log_sd.par_iter_mut()).zip(self.moments.par_iter_mut()).zip(derivatives.par_iter()).for_each(|(((mean, log_sd), moments), (gm, gs))| {
            let [m0, m1, m2, m3] = moments;
            adam_step(mean, gm, (m0, m1), adam.mean_rate, adam, step);
            adam_step(log_sd, gs, (m2, m3), adam.log_sd_rate, adam, step);
        });
        description
    }
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "MODEL SEQUENCES CONTEXT REPS OUT [WINDOWS]";
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
        let [windows] = windows else { return Err(format!("a Hugging Face checkpoint needs WINDOWS: {usage}")) };
        let text = std::fs::read_to_string(model.join("config.json")).map_err(error)?;
        let layers = serde_json::from_str::<Value>(&text).map_err(error)?["num_hidden_layers"].as_u64().ok_or("num_hidden_layers")? as usize;
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
    m_program.set_arithmetic(gam_gpu::tensor::Arithmetic::F32);
    let mut p_program = DeviceProgram::compile_values_bounded(&device, &interchange::prefix(&p_flat)?, usize::MAX)?;
    p_program.set_arithmetic(gam_gpu::tensor::Arithmetic::F32);
    p_program.prepare_dense_parameters(&trainable)?;
    let head = FixedHead::new(&device, &m_flat, &p_flat, 4096)?;
    let m = Model::new(&m_program, &m_flat, m_streams, m_reads, &[])?;
    let experiments = interchange::sample(&mut StdRng::seed_from_u64(1), sequences, layer_count, variables.len());
    let adam = Adam { mean_rate: 5e-5, log_sd_rate: 1e-2, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 };
    let scale = tokens as f64 / (experiments.len() * context) as f64 * std::f64::consts::LN_2;
    fn p_model<'a>(program: &'a DeviceProgram, (flat, streams, reads, trainable): (&OperatorProgram, &[usize], &[usize], &[usize])) -> Result<Model<'a>, String> {
        Model::new(program, flat, streams.to_vec(), reads.to_vec(), trainable)
    }
    let sites = (&p_flat, &p_streams[..], &p_reads[..], &trainable[..]);

    // The host path.
    let mut host_seconds: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    {
        let mut host = HostPosterior {
            mean: start.mean.clone(),
            log_sd: start.log_sd.clone(),
            moments: start.mean.iter().map(|m| std::array::from_fn(|_| Array2::zeros(m.dim()))).collect(),
            groups: start.groups.iter().zip(&start.mean).map(|(g, m)| Array2::from_shape_vec(m.dim(), g.clone()).map_err(error)).collect::<Result<_, _>>()?,
            count: start.count,
        };
        for step in 0..=reps {
            let s = &mut host_seconds;
            timed(&device, s, "means_loaded", || load(&mut p_program, &trainable, &host.mean))?;
            let design = timed(&device, s, "design", || interchange::design(&p_model(&p_program, sites)?, &variables, &experiments))?;
            let (theta, noise) = timed(&device, s, "sample", || Ok(host_sample(&host.mean, &host.log_sd, step as u64)))?;
            timed(&device, s, "sample_loaded", || load(&mut p_program, &trainable, &theta))?;
            let teacher = timed(&device, s, "teacher", || Teacher::new(&m, &head, &batch, &variables, &experiments))?;
            let evaluation = timed(&device, s, "evaluate", || interchange::evaluate(&m, &p_model(&p_program, sites)?, &head, &batch, &teacher, &experiments, &design, true))?;
            let gradients = timed(&device, s, "gradient_downloaded", || {
                trainable
                    .iter()
                    .zip(&host.mean)
                    .map(|(op, values)| match evaluation.gradient.get(op) {
                        Some(g) => Ok(device.download(g).map_err(error)? * scale),
                        None => Ok(Array2::zeros(values.dim())),
                    })
                    .collect::<Result<Vec<_>, String>>()
            })?;
            timed(&device, s, "posterior_step", || Ok(host.step(&gradients, &noise, &adam, step as i32 + 1)))?;
        }
    }
    // The device path.
    let mut device_seconds: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    {
        let parts = Parts { operators: &trainable, mean: &start.mean, log_sd: &start.log_sd, groups: &start.groups, count: start.count };
        let mut posterior = DevicePosterior::from_parts(&device, &parts, None, 0)?;
        for step in 0..=reps {
            let s = &mut device_seconds;
            timed(&device, s, "means_loaded", || posterior.mean_into(&mut p_program))?;
            let design = timed(&device, s, "design", || interchange::design(&p_model(&p_program, sites)?, &variables, &experiments))?;
            timed(&device, s, "sample_loaded", || posterior.sample_into(&mut p_program, step as u64))?;
            let teacher = timed(&device, s, "teacher", || Teacher::new(&m, &head, &batch, &variables, &experiments))?;
            let evaluation = timed(&device, s, "evaluate", || interchange::evaluate(&m, &p_model(&p_program, sites)?, &head, &batch, &teacher, &experiments, &design, true))?;
            timed(&device, s, "description", || Ok(posterior.divergences()?.iter().sum::<f64>()))?;
            timed(&device, s, "posterior_step", || posterior.step(&evaluation.gradient, scale, &adam, step as u64))?;
        }
    }
    let report = json!({
        "model": model.display().to_string(),
        "explanation": kind,
        "device": device.name(),
        "sequences": sequences,
        "context": context,
        "layers": layer_count,
        "experiments": experiments.len(),
        "trainable_operators": trainable.len(),
        "parameters": parameters,
        "groups": start.count,
        "reps": reps,
        "host_posterior_seconds": medians(&host_seconds),
        "device_posterior_seconds": medians(&device_seconds),
    });
    println!("{report}");
    std::fs::create_dir_all(out).map_err(error)?;
    std::fs::write(Path::new(out).join("library_step_bench.json"), serde_json::to_vec_pretty(&report).map_err(error)?).map_err(error)
}

//! The common evaluation battery (#2951) of a library explanation `P` of a language model `M`, on
//! held-out token rows of an export; `bench/vpd_2951/vpd_battery.py` applies the same battery to
//! VPD's decomposition.
//!
//! EXPORT SETTINGS.json OUT.json host|gpu [ARTIFACT]
//!
//! `ARTIFACT` is a `library_mdl` posterior-mean `artifact.bin`, or a fit's `checkpoint.bin`, whose
//! posterior mean is scored (with its `KL(q ‖ p)` and description reported). Without it `P` is the
//! library's starting point, which computes `M` exactly: every divergence must vanish to rounding.
//! An empty source range skips the interchange experiments.
//!
//! Every divergence is `KL(M ‖ E)` of the next-token distributions per token, in bits, `E` the
//! explanation in the given protocol; per-token quantiles are over every token of the held-out
//! rows. Cross-entropy is in nats per predicted token (the 511 predictions of a 512-token row).
//!
//! * Behaviour and VPD's protocols, from the logits: error-propagating (every layer of `P`, each
//!   reading `P`'s own stream; `P` alone), clean-input (every layer of `P` reading `M`'s stream
//!   entering it, the final stream `M`'s embedding plus each layer's increment), single-layer (one
//!   layer of `P`, `M` elsewhere), the cuts (`P`'s layers before `ℓ`, `M`'s after), and per size
//!   `k` a uniformly drawn subset of `k` layers per batch of bases.
//! * Interchange (`interchange`) with `P` alone (cut `L`): per base one read patch of a read
//!   variable drawn uniformly, the complement patch of every block, and the joint complement
//!   patch at a uniformly drawn set of at least two blocks (its size uniform in `2..=2L`, where
//!   cancellation between blocks shows), each with the source a sequence shared across the whole
//!   batch of bases. Each base's patches replace one position drawn uniformly and are scored from
//!   that position on (earlier tokens are the unpatched run's). Every source row is scored; the
//!   worst source of the first `K` (by the mean over all bases) is reported for each `K`.
use gam_gpu::{
    GpuPolicy,
    tensor::{Arithmetic, Device, Op, Tensor},
};
use gam_math::categorical::log_softmax;
use gam_mpd::{
    artifact::Artifact,
    device_program::DeviceProgram,
    engine::{log_to_stderr, sha256},
    import::import_language_model,
    interchange::{self, Batch, Experiment, Interchange, Patch, ReadVariable},
    library_mdl,
    operator_program::{FamilyInputs, Node, OperatorBody, OperatorProgram, SlotValues},
    run_check::{LayerNodes, layer_nodes, split_sites},
};
use ndarray::Array2;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, f64::consts::LN_2, path::Path, time::Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    export_sha256: String,
    context: usize,
    /// The held-out base rows `[start, end)`, and the rows `[start, end)` that serve as shared
    /// patch sources.
    held_out: [usize; 2],
    sources: [usize; 2],
    /// Bases per forward batch, the operator bytes each program may hold on the device, rows of
    /// vocabulary logits per tile, and the seed of the drawn patches and subsets.
    batch_sequences: usize,
    numeric_bytes: usize,
    head_tile_rows: usize,
    seed: u64,
    /// The numbers of shared sources of which the worst is reported.
    worst_of: Vec<usize>,
}

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// One model run layer by layer: its program through the final normed stream, the stream entering
/// each layer, the final residual (the final norm's input), the final normed stream, and the head
/// (the unembedding, classes × width) on the device.
struct Side {
    program: DeviceProgram,
    streams: Vec<usize>,
    residual: usize,
    hidden: usize,
    head: Tensor,
}

impl Side {
    fn new(device: &Device, artifact: &Artifact, layers: &[LayerNodes], numeric_bytes: usize) -> Result<Self, String> {
        let (flat, entries, _) = interchange::sites(artifact, layers)?;
        // The stream entering each layer: the entering stream of its attention block.
        let streams = entries.into_iter().step_by(2).collect();
        let mut program = DeviceProgram::compile_values_bounded(device, &interchange::prefix(&flat)?, numeric_bytes)?;
        program.set_arithmetic(if device.float64() { Arithmetic::F64 } else { Arithmetic::F32 });
        let hidden = program.hidden();
        let head = device.upload(unembedding(&flat, hidden)?.view()).map_err(error)?;
        Ok(Self { residual: final_residual(&flat, hidden)?, program, streams, hidden, head })
    }

    /// Layer `l` on `family` entering at the stream `entry` (none at layer 0): the stream it
    /// leaves (after the last layer, the final residual), and at layer 0 the embedding.
    fn layer(&self, family: &FamilyInputs, l: usize, entry: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>), String> {
        let d = self.program.device();
        let end = if l + 1 < self.streams.len() { self.streams[l + 1] } else { self.residual };
        let entry = entry.map(|x| d.copy(x).map(|x| (self.streams[l], x))).transpose().map_err(error)?;
        let trace = self.program.forward_span(family, entry, end, |_, _| Ok(None))?;
        let embedding = if l == 0 { Some(d.copy(trace.value(self.streams[0])?).map_err(error)?) } else { None };
        Ok((d.copy(trace.value(end)?).map_err(error)?, embedding))
    }

    /// Per sequence of `family` (each `length` rows), the logits of the final residual `x`.
    fn logits(&self, family: &FamilyInputs, x: &Tensor, length: usize) -> Result<Vec<Array2<f64>>, String> {
        let d = self.program.device();
        let trace = self.program.forward_span(family, Some((self.residual, d.copy(x).map_err(error)?)), self.hidden, |_, _| Ok(None))?;
        let hidden = trace.value(self.hidden)?;
        (0..family.rows / length)
            .map(|s| {
                let rows = d.rows_of(hidden, s * length, length).map_err(error)?;
                let mut logits = d.zeros(length, self.head.rows()).map_err(error)?;
                d.gemm(&mut logits, 1.0, &rows, Op::N, &self.head, Op::T, 0.0, self.program.arithmetic()).map_err(error)?;
                d.download(&logits).map_err(error)
            })
            .collect()
    }
}

/// The unembedding (classes × width) the final normed stream `hidden` of `flat` is read by: the
/// single bias-free dense map of the logits node, before the output's readout.
fn unembedding(flat: &OperatorProgram, hidden: usize) -> Result<Array2<f64>, String> {
    let logits = match &flat.nodes[flat.output] {
        Node::Readout { input, .. } => *input,
        _ => flat.output,
    };
    match &flat.nodes[logits] {
        Node::Transposed { input, operator } if *input == hidden => Ok(flat.operators[*operator].matrix().t().to_owned()),
        Node::Affine { terms, bias: None } if terms.len() == 1 && terms[0].0 == hidden => Ok(flat.operators[terms[0].1].matrix()),
        other => Err(format!("the logits are not one dense map of the final normed stream: {other:?}")),
    }
}

/// The input of the final norm before the node `hidden` (a gain over an RMS norm).
fn final_residual(flat: &OperatorProgram, hidden: usize) -> Result<usize, String> {
    let mut n = hidden;
    loop {
        match &flat.nodes[n] {
            Node::RmsNorm { input, .. } => return Ok(*input),
            Node::Affine { terms, bias: None } if terms.len() == 1 => n = terms[0].0,
            Node::Gain { input, .. } => n = *input,
            other => return Err(format!("the final normed stream is not a norm of the residual: {other:?}")),
        }
    }
}

/// Per-token values and their summary.
#[derive(Default)]
struct Tokens(Vec<f64>);

impl Tokens {
    fn summary(&self) -> Value {
        let mut v = self.0.clone();
        v.sort_by(f64::total_cmp);
        let q = |p: f64| v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        json!({"tokens": v.len(), "mean": mean, "q50": q(0.5), "q90": q(0.9), "q99": q(0.99), "max": q(1.0)})
    }
}

/// One protocol's per-token KL in bits, its cross-entropy and its top-1 agreement with `M`.
#[derive(Default)]
struct Behaviour {
    kl: Tokens,
    ce: Vec<f64>,
    agree: Vec<bool>,
}

impl Behaviour {
    /// Score the explanation's logits against `M`'s log-probabilities per sequence.
    fn add(&mut self, m: &[(Array2<f64>, Vec<usize>)], e: &[Array2<f64>], sequences: &[&[u32]]) -> Result<(), String> {
        for (((m, m_top), e), tokens) in m.iter().zip(e).zip(sequences) {
            let rows: Vec<(f64, f64, bool)> = (0..e.nrows())
                .into_par_iter()
                .map(|t| -> Result<(f64, f64, bool), String> {
                    let lq = log_softmax(e.row(t).as_slice().ok_or("contiguous logits")?).map_err(error)?;
                    let kl: f64 = m.row(t).iter().zip(&lq).map(|(lp, lq)| lp.exp() * (lp - lq)).sum();
                    let ce = if t + 1 < tokens.len() { -lq[tokens[t + 1] as usize] } else { f64::NAN };
                    let top = lq.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
                    Ok((kl / LN_2, ce, top == Some(m_top[t])))
                })
                .collect::<Result<_, _>>()?;
            for (kl, ce, agree) in rows {
                self.kl.0.push(kl);
                if ce.is_finite() {
                    self.ce.push(ce);
                }
                self.agree.push(agree);
            }
        }
        Ok(())
    }

    fn summary(&self) -> Value {
        json!({
            "kl_bits": self.kl.summary(),
            "ce": self.ce.iter().sum::<f64>() / self.ce.len() as f64,
            "top1_agreement": self.agree.iter().filter(|a| **a).count() as f64 / self.agree.len() as f64,
        })
    }
}

/// `M`'s log-probabilities and argmax per row of each sequence's logits.
fn log_probabilities(logits: Vec<Array2<f64>>) -> Result<Vec<(Array2<f64>, Vec<usize>)>, String> {
    logits
        .into_iter()
        .map(|mut l| {
            let top = (0..l.nrows()).map(|t| l.row(t).iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i)).collect();
            l.axis_iter_mut(ndarray::Axis(0)).into_par_iter().try_for_each(|mut row| -> Result<(), String> {
                let lp = log_softmax(row.as_slice().ok_or("contiguous logits")?).map_err(error)?;
                row.iter_mut().zip(lp).for_each(|(v, p)| *v = p);
                Ok(())
            })?;
            Ok((l, top))
        })
        .collect()
}

/// The posterior mean of a `library_mdl` checkpoint (a little-endian header length, the progress
/// header, then per trainable operator `μ`, `ln σ` and the four Adam moments in float64) as the
/// explanation's artifact, with the posterior's `KL(q ‖ p)` and description in bits and the fit's
/// last held-out evaluation.
fn checkpoint_mean(path: &Path, explanation: &library_mdl::Explanation) -> Result<(Artifact, Value), String> {
    let bytes = std::fs::read(path).map_err(error)?;
    let length = u64::from_le_bytes(bytes.get(..8).ok_or("a truncated checkpoint")?.try_into().map_err(error)?) as usize;
    let header: Value = serde_json::from_slice(bytes.get(8..8 + length).ok_or("a truncated checkpoint")?).map_err(error)?;
    let tokens = header["tokens"].as_u64().ok_or("checkpoint tokens")? as usize;
    let mut posterior = library_mdl::Posterior::new(explanation, tokens)?;
    let mut at = 8 + length;
    let mut take = |rows: usize, cols: usize| -> Result<Array2<f64>, String> {
        let raw = bytes.get(at..at + 8 * rows * cols).ok_or("a truncated checkpoint")?;
        at += 8 * rows * cols;
        let values = raw.chunks_exact(8).map(|c| c.try_into().map(f64::from_le_bytes).map_err(error)).collect::<Result<Vec<_>, _>>()?;
        Array2::from_shape_vec((rows, cols), values).map_err(error)
    };
    for i in 0..posterior.mean.len() {
        let (rows, cols) = posterior.mean[i].dim();
        posterior.mean[i] = take(rows, cols)?;
        posterior.log_sd[i] = take(rows, cols)?;
        for _ in 0..4 {
            take(rows, cols)?;
        }
    }
    if at != bytes.len() {
        return Err("the checkpoint's arrays do not match the explanation".into());
    }
    let active = header["active"].as_array().ok_or("checkpoint active")?;
    if active.len() != posterior.active.len() {
        return Err("the checkpoint's groups do not match the explanation".into());
    }
    for (a, v) in posterior.active.iter_mut().zip(active) {
        *a = v.as_bool().ok_or("checkpoint active")?;
    }
    let divergence: f64 = posterior.divergences().iter().sum();
    let size = json!({
        "epoch": header["epoch"],
        "training_tokens": tokens,
        "active_groups": posterior.active.iter().filter(|a| **a).count(),
        "divergence_bits": divergence / LN_2,
        "description_bits": posterior.description() / LN_2,
        "held_out": header["epochs"].as_array().and_then(|e| e.last()).map(|e| e["held_out"].clone()),
    });
    Ok((library_mdl::posterior_mean(explanation, &posterior)?.f32_literals()?, size))
}

/// The read variables of `artifact` whose rows hold a nonzero value (a removed function reads
/// nothing).
fn live_variables(artifact: &Artifact, layers: usize) -> Result<Vec<ReadVariable>, String> {
    let program = &artifact.program;
    let mut out = Vec::new();
    for v in interchange::library_reads(program, layers)? {
        let live = v.parts.iter().any(|(op, rows)| match &program.operators[*op].body {
            OperatorBody::Dense { .. } => {
                let m = program.operators[*op].matrix();
                rows.clone().any(|r| m.row(r).iter().any(|x| *x != 0.0))
            }
            _ => true,
        });
        if live {
            out.push(v);
        }
    }
    Ok(out)
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (export, settings_path, out, mode, artifact_path) = match &args[..] {
        [e, s, o, m] => (e, s, o, m, None),
        [e, s, o, m, a] => (e, s, o, m, Some(Path::new(a))),
        _ => return Err("EXPORT SETTINGS.json OUT.json host|gpu [ARTIFACT]".into()),
    };
    let (export, settings_path) = (Path::new(export), Path::new(settings_path));
    let settings: Settings = serde_json::from_slice(&std::fs::read(settings_path).map_err(error)?).map_err(error)?;
    if sha256(&export.join("export.json"))? != settings.export_sha256 {
        return Err("export hash mismatch".into());
    }
    let ([first, end], [s_first, s_end]) = (settings.held_out, settings.sources);
    if first >= end || s_first > s_end || settings.batch_sequences == 0 || settings.worst_of.iter().any(|k| *k == 0 || *k > s_end - s_first) {
        return Err("nonempty base and source ranges, positive batches, and at most as many worst-of sources as source rows".into());
    }
    let device = match mode.as_str() {
        "host" => Device::host(),
        "gpu" => Device::single_precision(GpuPolicy::Required).map_err(error)?.ok_or("a GPU required")?,
        _ => return Err("host|gpu required".into()),
    };
    let started = Instant::now();
    let imported = import_language_model(export, end.max(s_end), settings.context)?;
    let layer_count = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
    let native = split_sites(&imported.program)?;
    let layers = layer_nodes(&native, layer_count)?;
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else {
        return Err("a token slot".into());
    };
    let rows: Vec<Vec<u32>> = tokens.chunks(settings.context).map(<[u32]>::to_vec).collect();
    let bases = &rows[first..end];
    let sources = &rows[s_first..s_end];
    let explanation = library_mdl::explanation(&native, &layers)?;
    let (artifact, size) = match artifact_path {
        Some(path) if path.extension().is_some_and(|e| e == "bin") && path.file_name().is_some_and(|n| n == "checkpoint.bin") => checkpoint_mean(path, &explanation)?,
        Some(path) => (Artifact::from_bytes(&std::fs::read(path).map_err(error)?, &native.declarations)?, Value::Null),
        None => (explanation.artifact.clone(), Value::Null),
    };
    artifact.validate_coverage(&native)?;
    let length = settings.context;
    let mut rng = StdRng::seed_from_u64(settings.seed);

    // Behaviour and the protocols.
    let (m, p) = (Side::new(&device, &Artifact::native(&native)?, &layers, settings.numeric_bytes)?, Side::new(&device, &artifact, &layers, settings.numeric_bytes)?);
    let d = device.clone();
    let mut protocols: BTreeMap<String, Behaviour> = BTreeMap::new();
    let mut m_ce = Vec::new();
    for chunk in bases.chunks(settings.batch_sequences) {
        let views: Vec<&[u32]> = chunk.iter().map(Vec::as_slice).collect();
        let family = library_mdl::sequence_family(&views)?;
        // M's streams entering each layer, its embedding and final residual.
        let mut streams: Vec<Tensor> = Vec::with_capacity(layer_count + 1);
        let mut embedding = None;
        for l in 0..layer_count {
            let (x, e) = m.layer(&family, l, streams.last())?;
            if l == 0 {
                embedding = e;
            }
            streams.push(x);
        }
        let embedding = embedding.ok_or("no embedding")?;
        let m_logits = log_probabilities(m.logits(&family, &streams[layer_count - 1], length)?)?;
        for ((lp, _), tokens) in m_logits.iter().zip(&views) {
            m_ce.extend((0..length - 1).map(|t| -lp[[t, tokens[t + 1] as usize]]));
        }
        // The layer sets: per cut, per single layer, and per size one uniformly drawn subset.
        let mut sets: Vec<(String, Vec<bool>)> = Vec::new();
        for cut in 1..=layer_count {
            sets.push((if cut == layer_count { "error_propagating".into() } else { format!("cut_{cut}") }, (0..layer_count).map(|l| l < cut).collect()));
        }
        for l in 0..layer_count {
            sets.push((format!("single_layer_{l}"), (0..layer_count).map(|j| j == l).collect()));
        }
        for k in 2..layer_count {
            let mut order: Vec<usize> = (0..layer_count).collect();
            for i in 0..k {
                let j = rng.random_range(i..layer_count);
                order.swap(i, j);
            }
            sets.push((format!("subset_{k}"), (0..layer_count).map(|l| order[..k].contains(&l)).collect()));
        }
        for (name, set) in &sets {
            let mut x: Option<Tensor> = None;
            for (l, explained) in set.iter().enumerate() {
                let side = if *explained { &p } else { &m };
                x = Some(side.layer(&family, l, x.as_ref())?.0);
            }
            let e = m.logits(&family, x.as_ref().ok_or("no layers")?, length)?;
            protocols.entry(name.clone()).or_default().add(&m_logits, &e, &views)?;
        }
        // Clean-input: M's embedding plus each layer of P's increment on M's stream entering it.
        let mut x = d.copy(&embedding).map_err(error)?;
        for l in 0..layer_count {
            let entering = if l == 0 { &embedding } else { &streams[l - 1] };
            let (out, _) = p.layer(&family, l, (l > 0).then_some(entering))?;
            d.axpy(&mut x, 1.0, &out).map_err(error)?;
            d.axpy(&mut x, -1.0, entering).map_err(error)?;
        }
        let e = m.logits(&family, &x, length)?;
        protocols.entry("clean_input".into()).or_default().add(&m_logits, &e, &views)?;
        log::info!("battery: protocols on {} bases ({:.0} s)", chunk.len(), started.elapsed().as_secs_f64());
    }
    drop((m, p));
    let mut single = Vec::new();
    for l in 0..layer_count {
        single.extend_from_slice(&protocols[&format!("single_layer_{l}")].kl.0);
    }
    let mut report = json!({
        "export": export.display().to_string(),
        "export_sha256": settings.export_sha256,
        "settings_sha256": sha256(settings_path)?,
        "artifact": artifact_path.map(|p| p.display().to_string()),
        "artifact_sha256": artifact_path.map(sha256).transpose()?,
        "source_revision": option_env!("GAM_BUILD_GIT_SHA"),
        "device": device.name(),
        "held_out_rows": [first, end],
        "source_rows": [s_first, s_end],
        "ce_target": m_ce.iter().sum::<f64>() / m_ce.len() as f64,
        "protocols": protocols.iter().map(|(k, v)| (k.clone(), v.summary())).collect::<serde_json::Map<_, _>>(),
        "single_layer_mean": Tokens(single).summary(),
        "size": size,
    });
    std::fs::write(out, serde_json::to_vec_pretty(&report).map_err(error)?).map_err(error)?;
    if s_first == s_end {
        return Ok(());
    }

    // Interchange with P alone, sources shared across the batch.
    let variables = live_variables(&artifact, layer_count)?;
    let blocks = 2 * layer_count;
    let interchange = Interchange::new(&device, &native, &layers, &artifact, &explanation.trainable, variables.clone(), settings.numeric_bytes, settings.head_tile_rows)?;
    let read_of: Vec<usize> = (0..bases.len()).map(|_| rng.random_range(0..variables.len())).collect();
    let position_of: Vec<usize> = (0..bases.len()).map(|_| rng.random_range(0..bases[0].len())).collect();
    let joint_of: Vec<Vec<usize>> = (0..bases.len())
        .map(|_| {
            let k = rng.random_range(2..=blocks);
            interchange::hybrid_of(&mut rng, blocks, k).iter().enumerate().filter(|(_, x)| **x).map(|(b, _)| b).collect()
        })
        .collect();
    // Per family (the read patch, each block's complement, then the joint complement) per source:
    // bits and tokens over every base, and every token's bits.
    let families = 2 + blocks;
    let mut per_source = vec![vec![(0.0f64, 0usize); sources.len()]; families];
    let mut all: Vec<Tokens> = (0..families).map(|_| Tokens::default()).collect();
    let mut clean = Tokens::default();
    for (s, source) in sources.iter().enumerate() {
        for (c, chunk) in bases.chunks(settings.batch_sequences).enumerate() {
            let batch = Batch::new(chunk.to_vec(), vec![source.clone()])?;
            let mut experiments = Vec::with_capacity(chunk.len() * (families + 1));
            for b in 0..chunk.len() {
                let position = position_of[c * settings.batch_sequences + b];
                let at = |patch: Option<Patch>| {
                    let position = if patch.is_some() { position } else { 0 };
                    Experiment { base: b, source: 0, explained: vec![true; blocks], patch, position }
                };
                experiments.push(at(Some(Patch::Read { variable: read_of[c * settings.batch_sequences + b] })));
                experiments.extend((0..blocks).map(|block| at(Some(Patch::Complement { blocks: vec![block] }))));
                experiments.push(at(Some(Patch::Complement { blocks: joint_of[c * settings.batch_sequences + b].clone() })));
                if s == 0 {
                    experiments.push(at(None));
                }
            }
            // The directions of P's own reads at the scored explanation.
            let design = interchange::design(&interchange.models().1, &variables, &experiments)?;
            let scored = interchange.evaluate(&batch, &experiments, &design, false)?;
            for (e, bits) in experiments.iter().zip(&scored.bits) {
                let family = match &e.patch {
                    None => {
                        clean.0.extend_from_slice(bits);
                        continue;
                    }
                    Some(Patch::Read { .. }) => 0,
                    Some(Patch::Complement { blocks: one }) if one.len() == 1 => 1 + one[0],
                    Some(Patch::Complement { .. }) => 1 + blocks,
                };
                per_source[family][s].0 += bits.iter().sum::<f64>();
                per_source[family][s].1 += bits.len();
                all[family].0.extend_from_slice(bits);
            }
        }
        log::info!("battery: interchange source {}/{} ({:.0} s)", s + 1, sources.len(), started.elapsed().as_secs_f64());
    }
    let name = |f: usize| match f {
        0 => "read".to_string(),
        f if f <= blocks => format!("complement_block_{}", f - 1),
        _ => "complement_joint".to_string(),
    };
    let mut patches = serde_json::Map::new();
    for f in 0..families {
        let means: Vec<f64> = per_source[f].iter().map(|(bits, n)| bits / *n as f64).collect();
        let worst: serde_json::Map<String, Value> =
            settings.worst_of.iter().map(|&k| (format!("worst_of_{k}"), json!(means[..k].iter().copied().fold(f64::NEG_INFINITY, f64::max)))).collect();
        patches.insert(name(f), json!({"all_sources": all[f].summary(), "shared_source": worst}));
    }
    let mut complement = Tokens::default();
    (1..=blocks).for_each(|f| complement.0.extend_from_slice(&all[f].0));
    report["interchange"] = json!({
        "read_variables": variables.len(),
        "clean": clean.summary(),
        "patches": patches,
        "complement_all_blocks": complement.summary(),
    });
    report["seconds"] = json!(started.elapsed().as_secs_f64());
    std::fs::write(out, serde_json::to_vec_pretty(&report).map_err(error)?).map_err(error)?;
    log::info!("battery done in {:.0} s: {out}", started.elapsed().as_secs_f64());
    Ok(())
}

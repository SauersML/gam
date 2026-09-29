//! Causal-state extraction driver (#2951): the model's own state machine for a behaviour,
//! selected by its two-part code (`parameter_decomposition::causal_states`).
//!
//! It reads a harvest of `bench/mpd_causal_states_2951.py` and runs counterexample-guided
//! refinement per machine language on the model's readout (finite machines only, finite
//! plus counters, every kind), with every one-token continuation of the branched prefixes
//! as the pool. Each language's machine is refitted and compared on the sampled rows plus
//! every branch, with its quotient-consistency defect there. It reports each language's
//! machine (kind, states, transitions, counters, registers, bits), the refinement rounds,
//! the finite machine the winner unrolls to, the winner's margin over every other
//! language, its fidelity to the model and to the planted process, the winner's states
//! read off the native residual, and `do(state := s')` interventions compiled into edits
//! of the last MLP's down projection and executed through the exact final readout.
//!
//! Usage: `mpd_causal_states_2951 --harvest DIR [--out FILE]`.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use gam_math::gaussian_activation::GaussianActivation;
use gam_runtime::resource::MemoryGovernor;
use gam_mpd::attention::{
    AffineProjection, AttentionGeometry, NativeAttention, RotaryEmbedding, RotaryPairing,
};
use gam_mpd::causal_states::{
    Branch, Fitted, Harvest, Language, Machine, SearchReport, StateIntervention, consistency, distinct_start, fit, refine,
    unroll,
};
use gam_mpd::compile::ControlRealization;
use gam_mpd::lift::{TensorId, TensorRegistry, TieOrientation, UseMap, UseSiteId};
use gam_mpd::joint_operators::attention_letters;
use gam_mpd::module_split::MlpNormalForm;
use gam_mpd::state::{ObservabilityLetter, ObservabilityStep, WeightedObservability};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use serde_json::{Value, json};

/// A little-endian C-order `.npy` of `<f4` or `<i8`, as `f64` values and its shape.
fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f64>), String> {
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|mut file| file.read_to_end(&mut bytes))
        .map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(format!("{}: not an .npy file", path.display()));
    }
    let (length, offset) = if bytes[6] >= 2 {
        (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
    } else {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    };
    let header = std::str::from_utf8(&bytes[offset..offset + length]).map_err(|error| error.to_string())?;
    if !header.contains("'fortran_order': False") {
        return Err(format!("{}: expected C order", path.display()));
    }
    let open = header.find("'shape': (").ok_or("no shape")? + "'shape': (".len();
    let close = open + header[open..].find(')').ok_or("no shape end")?;
    let shape: Vec<usize> =
        header[open..close].split(',').filter_map(|part| part.trim().parse().ok()).collect();
    let data = &bytes[offset + length..];
    let values: Vec<f64> = if header.contains("'<f4'") {
        data.chunks_exact(4).map(|chunk| f64::from(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))).collect()
    } else if header.contains("'<i8'") {
        data.chunks_exact(8)
            .map(|chunk| {
                i64::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7]]) as f64
            })
            .collect()
    } else {
        return Err(format!("{}: expected <f4 or <i8: {header}", path.display()));
    };
    if values.len() != shape.iter().product::<usize>() {
        return Err(format!("{}: {} values for shape {shape:?}", path.display(), values.len()));
    }
    Ok((shape, values))
}

/// A harvest from `tokens` (`N × T`) and readout rows (`N × T × V`).
fn harvest_of(tokens: &(Vec<usize>, Vec<f64>), rows: &(Vec<usize>, Vec<f64>)) -> Result<Harvest, String> {
    let (sequences, length) = (tokens.0[0], tokens.0[1]);
    let vocabulary = rows.0[2];
    let starts: Vec<usize> = (0..=sequences).map(|index| index * length).collect();
    let ids: Vec<u32> = tokens.1.iter().map(|&value| value as u32).collect();
    let probabilities =
        Array2::from_shape_vec((sequences * length, vocabulary), rows.1.clone()).map_err(|error| error.to_string())?;
    Harvest::new(vocabulary, 1, ids, starts, probabilities).map_err(|error| error.to_string())
}

fn describe(fitted: &Fitted, harvest: &Harvest) -> Result<Value, String> {
    let machine: &Machine = &fitted.machine;
    let structure = &machine.structure;
    let (score, divergences) = machine.score(harvest).map_err(|error| error.to_string())?;
    let mean = divergences.iter().sum::<f64>() / divergences.len() as f64;
    let worst = divergences.iter().copied().fold(0.0_f64, f64::max);
    Ok(json!({
        "kind": machine.kind(),
        "total_bits": score.total(),
        "machine_bits": score.machine_bits,
        "data_bits": score.data_bits,
        "mean_kl_nats": mean,
        "max_kl_nats": worst,
        "symbols": structure.symbols.symbols(),
        "symbol_members": structure.symbols.members(),
        "classes": structure.classes,
        "initial": structure.initial,
        "table": structure.table,
        "counters": structure.counters.iter().map(|counter| json!({
            "increments": counter.increments, "period": counter.period})).collect::<Vec<_>>(),
        "registers": structure.registers.iter().map(|register| json!({
            "counter": register.counter, "writes": register.writes})).collect::<Vec<_>>(),
        "clamps": machine.readout.clamps.iter().map(|clamp| clamp.map(|clamp| [clamp.lo, clamp.hi])).collect::<Vec<_>>(),
        "cells": machine.readout.present.len(),
        "fraction_bits": machine.readout.logits.precision().fraction_bits(),
    }))
}

fn steps(report: &SearchReport) -> Vec<Value> {
    report
        .steps
        .iter()
        .map(|step| json!({"move": step.description, "total_bits": step.score.total()}))
        .collect()
}


/// A weight of the harvest as a matrix (`w_{name}.npy`).
fn matrix(dir: &Path, name: &str) -> Result<Array2<f64>, String> {
    let (shape, values) = read_npy(&dir.join(format!("w_{name}.npy")))?;
    if shape.len() != 2 {
        return Err(format!("{name}: expected a matrix, got {shape:?}"));
    }
    Array2::from_shape_vec((shape[0], shape[1]), values).map_err(|error| error.to_string())
}

fn vector(dir: &Path, name: &str) -> Result<Array1<f64>, String> {
    let (shape, values) = read_npy(&dir.join(format!("w_{name}.npy")))?;
    if shape.len() != 1 {
        return Err(format!("{name}: expected a vector, got {shape:?}"));
    }
    Ok(Array1::from(values))
}

/// The planted model's layout, from `harvest.json`.
struct Layout {
    layers: usize,
    heads: usize,
    width: usize,
}

/// The weighted observability factor `F` (`k × d`) at the residual entering block `layer`
/// (`layer = L` is the final residual): the behaviour readout (the unembedding rows times the
/// final norm gain, centred over the readout classes, since a softmax ignores a common
/// shift) pulled back through every later block. Each block is two steps: attention, whose
/// letters are the identity (the residual edge) and the routing laws' value/output transports
/// with the first norm's gain folded in (`joint_operators::attention_letters`), then the MLP
/// as its merged normal form with the identity skip and the second norm's gain folded into
/// its reads (`module_split::MlpNormalForm`). The norms' per-input `1/rms` scalings are not
/// letters, so `F` weighs directions, not their per-input scale.
fn observability_chart(dir: &Path, layout: &Layout, layer: usize) -> Result<Array2<f64>, String> {
    let governor = MemoryGovernor::global();
    let width = layout.width;
    let head_dim = width / layout.heads;
    let identity = Array2::<f64>::eye(width);
    let final_gain = vector(dir, "ln_f")?;
    let unembedding = matrix(dir, "lm_head")?;
    let classes = unembedding.nrows();
    let mean = unembedding.mean_axis(Axis(0)).ok_or("empty unembedding")?;
    let mut readout = &unembedding - &mean.insert_axis(Axis(0));
    for mut row in readout.rows_mut() {
        row *= &final_gain;
    }
    if classes < 2 {
        return Err("a readout needs two classes".to_string());
    }
    let planes = head_dim / 2;
    let rotary = RotaryEmbedding {
        pairing: RotaryPairing::HalfSplit,
        inverse_frequencies: (0..planes).map(|plane| 10000_f64.powf(-(plane as f64) / planes as f64)).collect(),
        attention_scaling: 1.0,
    };
    let mut routing = Vec::new();
    let mut mlps = Vec::new();
    for block in layer..layout.layers {
        let projection = |name: &str| -> Result<AffineProjection, String> {
            let weight = matrix(dir, &format!("{name}.{block}"))?;
            let rows = weight.nrows();
            Ok(AffineProjection { weight, bias: Array1::zeros(rows) })
        };
        let native = NativeAttention::new(
            AttentionGeometry { model_dim: width, n_heads: layout.heads, n_kv_heads: layout.heads, head_dim },
            rotary.clone(),
            1.0 / (head_dim as f64).sqrt(),
            projection("q")?,
            projection("k")?,
            projection("v")?,
            projection("o")?,
        )
        .map_err(|error| error.to_string())?;
        let gain = vector(dir, &format!("rms_1.{block}"))?;
        routing.push(attention_letters(governor, &native, Some(gain.view())).map_err(|error| format!("{error:?}"))?);
        let mut reads = matrix(dir, &format!("fc.{block}"))?;
        let gain = vector(dir, &format!("rms_2.{block}"))?;
        for mut row in reads.rows_mut() {
            row *= &gain;
        }
        let writes = matrix(dir, &format!("down.{block}"))?;
        mlps.push(
            MlpNormalForm::new(
                GaussianActivation::ExactGelu,
                reads.view(),
                Array1::zeros(reads.nrows()).view(),
                writes.view(),
                Array1::zeros(width).view(),
                Some(identity.view()),
            )
            .map_err(|error| error.to_string())?,
        );
    }
    let mut steps = Vec::new();
    for (letters, mlp) in routing.iter().zip(&mlps) {
        steps.push(ObservabilityStep {
            letters: vec![ObservabilityLetter::Linear(identity.view()), ObservabilityLetter::RoutingLaws(letters)],
            readouts: Vec::new(),
        });
        steps.push(ObservabilityStep { letters: mlp.observability_letters(), readouts: Vec::new() });
    }
    if steps.is_empty() {
        steps.push(ObservabilityStep { letters: vec![ObservabilityLetter::Linear(identity.view())], readouts: Vec::new() });
    }
    if let Some(last) = steps.last_mut() {
        last.readouts.push(readout.view());
    }
    let observability = WeightedObservability::pull_back(governor, &steps).map_err(|error| error.to_string())?;
    Ok(observability.factor)
}

/// Per row, the machine's full state tuple `(class, counters.., registers..)`.
fn machine_states(machine: &Machine, harvest: &Harvest) -> Result<Vec<Vec<i64>>, String> {
    let structure = &machine.structure;
    let fields = 1 + structure.counters.len() + structure.registers.len();
    let trace = structure.trace(harvest).map_err(|error| error.to_string())?;
    Ok(trace.chunks_exact(fields).map(<[i64]>::to_vec).collect())
}

/// The native chart coordinates `F h` of `rows` (`n × d`).
fn chart_of(chart: &Array2<f64>, rows: ArrayView2<'_, f64>) -> Array2<f64> {
    rows.dot(&chart.t())
}

/// Squared Euclidean distance.
fn distance(a: ndarray::ArrayView1<'_, f64>, b: ndarray::ArrayView1<'_, f64>) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// The machine's abstract states read off the model's native state at one layer: the
/// centroid of each state tuple in the observability chart, the nearest-centroid decoder's
/// agreement with the machine on every row, and the chart's separation (least squared
/// distance between two centroids against the mean squared spread within a state).
fn native_states(
    dir: &Path,
    layout: &Layout,
    layer: usize,
    machine: &Machine,
    harvest: &Harvest,
) -> Result<Value, String> {
    let (shape, values) = read_npy(&dir.join("resid.npy"))?;
    let (rows, width) = (shape[1] * shape[2], shape[3]);
    let per_layer = rows * width;
    let residual = ArrayView2::from_shape((rows, width), &values[layer * per_layer..(layer + 1) * per_layer])
        .map_err(|error| error.to_string())?;
    let chart = observability_chart(dir, layout, layer)?;
    let coordinates = chart_of(&chart, residual);
    let states = machine_states(machine, harvest)?;
    let mut index: std::collections::BTreeMap<Vec<i64>, usize> = std::collections::BTreeMap::new();
    for state in &states {
        let next = index.len();
        index.entry(state.clone()).or_insert(next);
    }
    let labels: Vec<usize> = states.iter().map(|state| index[state]).collect();
    let mut centroids = Array2::<f64>::zeros((index.len(), coordinates.ncols()));
    let mut counts = vec![0.0_f64; index.len()];
    for (row, &label) in labels.iter().enumerate() {
        let mut target = centroids.row_mut(label);
        target += &coordinates.row(row);
        counts[label] += 1.0;
    }
    for (mut centroid, count) in centroids.rows_mut().into_iter().zip(&counts) {
        centroid /= *count;
    }
    let decode = |point: ndarray::ArrayView1<'_, f64>| -> usize {
        (0..centroids.nrows())
            .min_by(|&a, &b| distance(point, centroids.row(a)).total_cmp(&distance(point, centroids.row(b))))
            .unwrap_or(0)
    };
    let decoded: Vec<usize> = coordinates.rows().into_iter().map(decode).collect();
    let agree = decoded.iter().zip(&labels).filter(|(a, b)| a == b).count();
    let spread = labels
        .iter()
        .enumerate()
        .map(|(row, &label)| distance(coordinates.row(row), centroids.row(label)))
        .sum::<f64>()
        / rows as f64;
    let mut separation = f64::INFINITY;
    for a in 0..centroids.nrows() {
        for b in a + 1..centroids.nrows() {
            separation = separation.min(distance(centroids.row(a), centroids.row(b)));
        }
    }
    let per_state: Vec<Value> = index
        .iter()
        .map(|(state, &label)| {
            let members: Vec<usize> = (0..rows).filter(|&row| labels[row] == label).collect();
            let hits = members.iter().filter(|&&row| decoded[row] == label).count();
            json!({"state": state, "rows": members.len(), "decoded_correctly": hits})
        })
        .collect();
    Ok(json!({
        "layer": layer,
        "chart_rank": chart.nrows(),
        "states": index.len(),
        "decoder_agreement": agree as f64 / rows as f64,
        "mean_within_state_sq_spread": spread,
        "least_between_centroid_sq_distance": separation,
        "per_state": per_state,
    }))
}

/// A row-major `.npy` of the harvest as `(shape, values)` restricted to its first `rows`
/// leading-axis rows (sequences × positions flattened by the caller).
fn read_rows(dir: &Path, name: &str) -> Result<(Vec<usize>, Vec<f64>), String> {
    read_npy(&dir.join(format!("{name}.npy")))
}

/// Which writers into the residual stream carry the machine's state distinctions at the
/// interface entering block `layer`. The residual is exactly the sum of its writes (the
/// token embedding, every head's attention write, every MLP's write), and the chart is
/// linear, so each state's centroid is the sum of the writers' centroids. A writer's share
/// is `Σ_s n_s ⟨c_s^w − c̄^w, c_s − c̄⟩ / Σ_s n_s ‖c_s − c̄‖²`; the shares sum to one exactly.
/// An MLP's share is split over its units the same way, since its write is `Σ_j a_j u_j`.
fn attribution(dir: &Path, layout: &Layout, layer: usize, machine: &Machine, harvest: &Harvest) -> Result<Value, String> {
    let chart = observability_chart(dir, layout, layer)?;
    let states = machine_states(machine, harvest)?;
    let (shape, _) = read_rows(dir, "mlp_act.0")?;
    let rows = shape[0] * shape[1];
    let mut index: std::collections::BTreeMap<Vec<i64>, usize> = std::collections::BTreeMap::new();
    for state in &states[..rows] {
        let next = index.len();
        index.entry(state.clone()).or_insert(next);
    }
    let labels: Vec<usize> = states[..rows].iter().map(|state| index[state]).collect();
    let classes = index.len();
    let mut counts = vec![0.0_f64; classes];
    for &label in &labels {
        counts[label] += 1.0;
    }
    // Per writer, the chart coordinates of its write on every row, reduced to state centroids.
    let centroid_of = |coordinates: &Array2<f64>| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((classes, coordinates.ncols()));
        for (row, &label) in labels.iter().enumerate() {
            let mut target = out.row_mut(label);
            target += &coordinates.row(row);
        }
        for (mut centroid, count) in out.rows_mut().into_iter().zip(&counts) {
            centroid /= *count;
        }
        out
    };
    let mut writers: Vec<(String, Array2<f64>)> = Vec::new();
    let embedding = matrix(dir, "wte")?;
    let tokens = harvest.tokens();
    let mut embedded = Array2::<f64>::zeros((rows, embedding.ncols()));
    for (row, mut target) in embedded.rows_mut().into_iter().enumerate() {
        target.assign(&embedding.row(tokens[row] as usize));
    }
    writers.push(("embedding".to_string(), centroid_of(&chart_of(&chart, embedded.view()))));
    let mut unit_shares: Vec<Value> = Vec::new();
    let mut unit_centroids: Vec<(usize, Array2<f64>, Array2<f64>)> = Vec::new();
    for block in 0..layer {
        let (head_shape, heads) = read_rows(dir, &format!("head_writes.{block}"))?;
        let width = head_shape[3];
        for head in 0..head_shape[2] {
            let mut write = Array2::<f64>::zeros((rows, width));
            for (row, mut target) in write.rows_mut().into_iter().enumerate() {
                let offset = (row * head_shape[2] + head) * width;
                target.assign(&ndarray::ArrayView1::from(&heads[offset..offset + width]));
            }
            writers.push((format!("L{block}.head{head}"), centroid_of(&chart_of(&chart, write.view()))));
        }
        let (act_shape, acts) = read_rows(dir, &format!("mlp_act.{block}"))?;
        let activations = Array2::from_shape_vec((rows, act_shape[2]), acts).map_err(|error| error.to_string())?;
        let down = matrix(dir, &format!("down.{block}"))?;
        // Unit j's chart direction F u_j, and its mean activation per state.
        let directions = chart.dot(&down);
        let means = centroid_of(&activations);
        let write = activations.dot(&down.t());
        writers.push((format!("L{block}.mlp"), centroid_of(&chart_of(&chart, write.view()))));
        unit_centroids.push((block, means, directions));
    }
    let total: Array2<f64> = writers.iter().fold(Array2::zeros(writers[0].1.dim()), |sum, (_, c)| sum + c);
    let weights = Array1::from(counts.clone());
    let grand = total.t().dot(&weights) / weights.sum();
    let deviation = &total - &grand.clone().insert_axis(Axis(0));
    let scatter: f64 = deviation.rows().into_iter().zip(&counts).map(|(row, n)| n * row.dot(&row)).sum();
    let share = |centroids: &Array2<f64>| -> f64 {
        let mean = centroids.t().dot(&weights) / weights.sum();
        centroids
            .rows()
            .into_iter()
            .zip(deviation.rows())
            .zip(&counts)
            .map(|((row, dev), n)| n * (&row - &mean).dot(&dev))
            .sum::<f64>()
            / scatter
    };
    let writer_shares: Vec<Value> =
        writers.iter().map(|(name, centroids)| json!({"writer": name, "share": share(centroids)})).collect();
    for (block, means, directions) in &unit_centroids {
        let grand_mean = means.t().dot(&weights) / weights.sum();
        let mut shares: Vec<(usize, f64)> = (0..means.ncols())
            .map(|unit| {
                let direction = directions.column(unit);
                let value: f64 = (0..classes)
                    .map(|class| counts[class] * (means[[class, unit]] - grand_mean[unit]) * direction.dot(&deviation.row(class)))
                    .sum();
                (unit, value / scatter)
            })
            .collect();
        let magnitude: f64 = shares.iter().map(|(_, value)| value.abs()).sum();
        let quartic: f64 = shares.iter().map(|(_, value)| value * value).sum();
        shares.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        unit_shares.push(json!({
            "block": block,
            "participation_ratio": if quartic > 0.0 { magnitude * magnitude / quartic } else { 0.0 },
            "top_units": shares.iter().take(12).map(|(unit, value)| json!([unit, value])).collect::<Vec<_>>(),
        }));
    }
    Ok(json!({"layer": layer, "rows": rows, "states": classes, "writer_shares": writer_shares, "mlp_units": unit_shares}))
}

/// The branch pool: every harvested prefix of the first `B` sequences followed by every
/// token the harvest reads after its first position (`branch_probs.npy`, `B × T × V × V`).
fn branch_pool(dir: &Path, harvest: &Harvest, length: usize) -> Result<Vec<Branch>, String> {
    let (shape, values) = read_npy(&dir.join("branch_probs.npy"))?;
    let (sequences, positions, vocabulary) = (shape[0], shape[1], shape[2]);
    if positions != length || vocabulary != harvest.vocabulary() {
        return Err(format!("branch shape {shape:?} against length {length}"));
    }
    let mut alphabet = vec![false; vocabulary];
    for sequence in 0..harvest.sequences() {
        for row in harvest.sequence(sequence).skip(1) {
            alphabet[harvest.tokens()[row] as usize] = true;
        }
    }
    let mut pool = Vec::new();
    for sequence in 0..sequences {
        for position in 0..positions {
            for token in (0..vocabulary).filter(|&token| alphabet[token]) {
                let offset = ((sequence * positions + position) * vocabulary + token) * vocabulary;
                pool.push(Branch {
                    row: sequence * length + position,
                    token: token as u32,
                    probabilities: values[offset..offset + vocabulary].to_vec(),
                });
            }
        }
    }
    Ok(pool)
}

/// `softmax(lm_head · rms(h) ⊙ ln_f)` of every row of `residual`, the planted model's exact
/// final readout (RMSNorm with ε = 1e-6).
fn final_readout(dir: &Path, residual: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let gain = vector(dir, "ln_f")?;
    let unembedding = matrix(dir, "lm_head")?;
    let mut normed = residual.to_owned();
    for mut row in normed.rows_mut() {
        let scale = 1.0 / (row.iter().map(|value| value * value).sum::<f64>() / row.len() as f64 + 1e-6).sqrt();
        row *= scale;
        row *= &gain;
    }
    let mut logits = normed.dot(&unembedding.t());
    for mut row in logits.rows_mut() {
        let top = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        row.mapv_inplace(|logit| (logit - top).exp());
        let total = row.sum();
        row /= total;
    }
    Ok(logits)
}

fn kl(p: ndarray::ArrayView1<'_, f64>, q: ndarray::ArrayView1<'_, f64>) -> f64 {
    p.iter().zip(q).filter(|(p, _)| **p > 0.0).map(|(p, q)| p * (p.ln() - q.max(f64::MIN_POSITIVE).ln())).sum::<f64>().max(0.0)
}

/// Every occupied state of the branched sequences set to the state the machine reads most
/// differently, compiled into one edit of the last MLP's down projection through the edit
/// compiler, and executed through the exact final readout. Per intervention: the compiler's
/// status, the model's KL from the machine's target readout before and after the edit on
/// the intervened rows, and the edit's damage (KL from the native readout) on every other row.
fn interventions(dir: &Path, layout: &Layout, machine: &Machine, harvest: &Harvest) -> Result<Value, String> {
    let block = layout.layers - 1;
    let (act_shape, acts) = read_rows(dir, &format!("mlp_act.{block}"))?;
    let rows = act_shape[0] * act_shape[1];
    let activations = Array2::from_shape_vec((rows, act_shape[2]), acts).map_err(|error| error.to_string())?;
    let (shape, values) = read_npy(&dir.join("resid.npy"))?;
    let (all_rows, width) = (shape[1] * shape[2], shape[3]);
    let offset = layout.layers * all_rows * width;
    let residual = ArrayView2::from_shape((rows, width), &values[offset..offset + rows * width])
        .map_err(|error| error.to_string())?
        .to_owned();
    drop(values);
    let down = matrix(dir, &format!("down.{block}"))?;
    let states = machine_states(machine, harvest)?;
    let cells = machine.cells(harvest).map_err(|error| error.to_string())?;
    let mut index: std::collections::BTreeMap<Vec<i64>, usize> = std::collections::BTreeMap::new();
    for state in &states[..rows] {
        let next = index.len();
        index.entry(state.clone()).or_insert(next);
    }
    let labels: Vec<usize> = states[..rows].iter().map(|state| index[state]).collect();
    let mut cell_of_state = vec![0; index.len()];
    for (row, &label) in labels.iter().enumerate() {
        cell_of_state[label] = cells[row];
    }
    let readout = harvest.readout();
    let targets: Vec<Array1<f64>> =
        cell_of_state.iter().map(|&cell| Array1::from(machine.distribution(cell, readout))).collect();
    let mut registry = TensorRegistry::default();
    let storage = TensorId(format!("down.{block}"));
    let site = UseSiteId(format!("down.{block}#0"));
    registry.register_storage(storage.clone(), down.view().into_dyn()).map_err(|error| format!("{error:?}"))?;
    registry
        .register_use_site(site.clone(), storage.clone(), UseMap::Linear(TieOrientation::Identity))
        .map_err(|error| format!("{error:?}"))?;
    let intervention = StateIntervention {
        registry: &registry,
        storage,
        site,
        native: down.view(),
        inputs: activations.view(),
        residual: residual.view(),
        labels: &labels,
    };
    let native = final_readout(dir, residual.view())?;
    let mut out = Vec::new();
    for (from, _) in cell_of_state.iter().enumerate() {
        let to = (0..targets.len())
            .max_by(|&a, &b| kl(targets[a].view(), targets[from].view()).total_cmp(&kl(targets[b].view(), targets[from].view())))
            .unwrap_or(from);
        if to == from {
            continue;
        }
        let report = intervention.compile(from, to, MemoryGovernor::global()).map_err(|error| error.to_string())?;
        let status = match &report.compiled.realization {
            ControlRealization::ExactlyRealized { .. } => "exactly realized",
            ControlRealization::EmpiricallyValidated { .. } => "empirically validated",
            ControlRealization::Descriptive { .. } => "descriptive",
        };
        let Some(plan) = report.compiled.plan.as_ref() else {
            out.push(json!({"from": from, "to": to, "status": status}));
            continue;
        };
        let mut edited = residual.clone();
        for edit in plan.edits() {
            edited += &activations.dot(&edit.delta.right()).dot(&edit.delta.left().t());
        }
        let after = final_readout(dir, edited.view())?;
        let (mut before_kl, mut after_kl, mut hits) = (0.0, 0.0, 0.0);
        let (mut damage_mean, mut damage_max, mut others) = (0.0, 0.0_f64, 0.0);
        for (row, &label) in labels.iter().enumerate() {
            if label == from {
                before_kl += kl(targets[to].view(), native.row(row));
                after_kl += kl(targets[to].view(), after.row(row));
                hits += 1.0;
            } else {
                let damage = kl(native.row(row), after.row(row));
                damage_mean += damage;
                damage_max = damage_max.max(damage);
                others += 1.0;
            }
        }
        out.push(json!({
            "from": index.iter().find(|(_, label)| **label == from).map(|(state, _)| state.clone()),
            "to": index.iter().find(|(_, label)| **label == to).map(|(state, _)| state.clone()),
            "rows": hits,
            "status": status,
            "edit_frobenius": report.metric_norm.map(|(norm, _)| norm),
            "target_kl_before_nats": before_kl / hits,
            "target_kl_after_nats": after_kl / hits,
            "off_target_kl_mean_nats": if others > 0.0 { damage_mean / others } else { 0.0 },
            "off_target_kl_max_nats": damage_max,
        }));
    }
    Ok(json!(out))
}

fn run(dir: &Path) -> Result<Value, String> {
    let tokens = read_npy(&dir.join("tokens.npy"))?;
    let probabilities = read_npy(&dir.join("probs.npy"))?;
    let truth = read_npy(&dir.join("truth.npy"))?;
    let harvest = harvest_of(&tokens, &probabilities)?;
    let process = harvest_of(&tokens, &truth)?;
    let pool = branch_pool(dir, &harvest, tokens.0[1])?;
    let everything = harvest.with_branches(&pool).map_err(|error| error.to_string())?;
    let start = distinct_start(&harvest).map_err(|error| error.to_string())?;
    let mut languages = Vec::new();
    let mut winner: Option<(f64, String, Fitted)> = None;
    for (name, language) in [("finite", Language::FINITE), ("counters", Language::COUNTERS), ("full", Language::FULL)] {
        let refined = refine(&start, &harvest, &pool, language).map_err(|error| error.to_string())?;
        // Every language is compared on one harvest: the sampled rows and every branch.
        let common = fit(&refined.report.fitted.machine.structure, &everything).map_err(|error| error.to_string())?;
        let quotient = consistency(&common.machine, &everything).map_err(|error| error.to_string())?;
        let total = common.score.total();
        languages.push(json!({
            "language": name,
            "best": describe(&common, &everything)?,
            "steps": steps(&refined.report),
            "rounds": refined.rounds.iter().map(|round| json!({
                "counterexamples": round.counterexamples, "total_bits": round.fitted.score.total(),
                "kind": round.fitted.machine.kind()})).collect::<Vec<_>>(),
            "admitted": refined.admitted.len(),
            "consistency": {"defect_bits": quotient.defect_bits, "cells": quotient.cells, "worst": quotient.worst},
        }));
        if winner.as_ref().is_none_or(|(incumbent, ..)| total < *incumbent) {
            winner = Some((total, name.to_string(), common));
        }
    }
    let (total, name, fitted) = winner.ok_or("no language ran")?;
    let unrolled = match unroll(&fitted.machine, &everything) {
        Ok(structure) => Some(describe(&fit(&structure, &everything).map_err(|error| error.to_string())?, &everything)?),
        Err(_) => None,
    };
    // The winner against the planted process: the same machine scored on the process law.
    let (_, process_divergences) = fitted.machine.score(&process).map_err(|error| error.to_string())?;
    let model_to_process: Vec<f64> = harvest
        .probabilities()
        .rows()
        .into_iter()
        .zip(process.probabilities().rows())
        .map(|(model, law)| law.iter().zip(model).filter(|(p, _)| **p > 0.0).map(|(p, q)| p * (p.ln() - q.ln())).sum())
        .collect();
    let meta: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("harvest.json")).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let layout = Layout {
        layers: meta["layers"].as_u64().ok_or("layers")? as usize,
        heads: meta["heads"].as_u64().ok_or("heads")? as usize,
        width: meta["d"].as_u64().ok_or("d")? as usize,
    };
    let mut native = Vec::new();
    for layer in 0..=layout.layers {
        native.push(native_states(dir, &layout, layer, &fitted.machine, &harvest)?);
    }
    let interventions = interventions(dir, &layout, &fitted.machine, &harvest)?;
    let mut attributions = Vec::new();
    for layer in 1..=layout.layers {
        attributions.push(attribution(dir, &layout, layer, &fitted.machine, &harvest)?);
    }
    let margins: Vec<Value> = languages
        .iter()
        .map(|entry| json!({"language": entry["language"], "bits_above_winner": entry["best"]["total_bits"].as_f64().unwrap_or(f64::NAN) - total}))
        .collect();
    Ok(json!({
        "harvest": dir.display().to_string(),
        "rows": harvest.rows(),
        "sequences": harvest.sequences(),
        "winner": name,
        "winner_total_bits": total,
        "margins": margins,
        "languages": languages,
        "winner_unrolled_to_finite": unrolled,
        "branches": pool.len(),
        "native": native,
        "interventions": interventions,
        "attribution": attributions,
        "winner_mean_kl_process_nats": process_divergences.iter().sum::<f64>() / process_divergences.len() as f64,
        "model_mean_kl_process_nats": model_to_process.iter().sum::<f64>() / model_to_process.len() as f64,
    }))
}

fn main() -> Result<(), String> {
    let mut arguments = std::env::args().skip(1);
    let mut harvest: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--harvest" => harvest = arguments.next().map(PathBuf::from),
            "--out" => out = arguments.next().map(PathBuf::from),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let harvest = harvest.ok_or("--harvest DIR is required")?;
    let report = run(&harvest)?;
    let text = serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
    if let Some(out) = out {
        std::fs::write(&out, &text).map_err(|error| error.to_string())?;
    }
    println!("{text}");
    Ok(())
}

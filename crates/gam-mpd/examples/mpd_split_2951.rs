//! Does the library's code length `F` prefer explanations whose functions are active on fewer
//! tokens (#2951)? The function-preserving split: each listed MLP function `f_i(x) = φ(g_i·x + c_i)
//! u_i` of the native model is duplicated into two copies writing `u_i / 2` each, which computes
//! the same function. Both arms start from one fit checkpoint of the unsplit library: the base arm
//! from its posterior as it is, the split arm from the same posterior with each listed function's
//! gate rows copied and its output column halved in mean and standard deviation, the copies' groups
//! active as the function's are. Both continue under the same settings, weight noise (the
//! checkpoint's next epoch), convergence test and removal (`library_mdl::fit_from`), each with
//! IVON's curvature started afresh from its posterior's standard deviations, so the arms differ only
//! in the split. `ΔF = F_split − F_base`, its description and data parts, and each function's and
//! copy's groups are reported; the posterior noise is what can break the copies' symmetry. The
//! copies' activation contexts come from each arm's `artifact.bin` (the posterior mean) through the
//! readout.
//!
//! EXPORT SETTINGS.json CHECKPOINT OUT host|gpu LAYER:UNIT[,LAYER:UNIT…]|top:K
//!
//! SETTINGS.json is `mpd_library_mdl_2951`'s for an engine export (with its `blocks` for a
//! one-block explanation), CHECKPOINT that fit's `checkpoint.bin`. `top:K` splits the `K` surviving
//! MLP functions of the checkpoint's posterior whose groups carry the most information
//! (`Σ KL(q_G ‖ p_G)` over the function's groups). `OUT/base` and `OUT/split` hold the two arms' checkpoints; rerunning the same
//! command resumes them.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    engine::{log_to_stderr, sha256},
    import::import_language_model,
    library_mdl::{self, Explanation, Start},
    operator_program::{Interface, Node, Operator, OperatorProgram, SlotValues, exact_precision},
    run_check::{LayerNodes, layer_nodes, split_sites},
};
use ndarray::{Array2, Axis, concatenate};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, f64::consts::LN_2, path::Path, sync::Arc};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    export_sha256: String,
    training_sequences: usize,
    context: usize,
    held_out: [usize; 2],
    /// A Hugging Face checkpoint's token rows, which this example does not take.
    #[serde(default)]
    windows: Option<Value>,
    /// The blocks the checkpoint's explanation explains, when not all (`library_mdl::scoped`).
    #[serde(default)]
    blocks: Option<Vec<usize>>,
    fit: library_mdl::Settings,
}

fn save(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(path, serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

/// A dense operator like `like` with new interfaces and values, exactly representable.
fn operator(like: &Operator, rows: Interface, cols: Interface, values: Array2<f64>) -> Result<Arc<Operator>, String> {
    let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
    Operator::dense(like.name.clone(), rows, cols, values, precision, like.provenance.clone()).map(Arc::new).map_err(|e| e.to_string())
}

/// `native` with each listed MLP unit `(layer, unit)` duplicated: the copy is appended after the
/// layer's units, and the unit and its copy each write half the unit's output column, so the
/// program computes the same function. Returns the program and each copy's unit index.
fn duplicate(native: &OperatorProgram, layers: &[LayerNodes], units: &[(usize, usize)]) -> Result<(OperatorProgram, Vec<usize>), String> {
    let mut out = native.clone();
    let mut copies = vec![0; units.len()];
    for (l, layer) in layers.iter().enumerate() {
        let mine: Vec<(usize, usize)> = units.iter().enumerate().filter(|(_, (m, _))| *m == l).map(|(k, (_, i))| (k, *i)).collect();
        if mine.is_empty() {
            continue;
        }
        // The pointwise law is per group of the units' interface, which stays one group.
        let Node::Pointwise { input: pre, .. } = out.nodes[layer.active] else {
            return Err(format!("layer {l}: the MLP activation is not one pointwise law"));
        };
        let (up, bias) = match &out.nodes[pre] {
            Node::Affine { terms, bias } if terms.len() == 1 => (terms[0].1, *bias),
            other => return Err(format!("layer {l}: the MLP's input map is {other:?}")),
        };
        let down = match &out.nodes[layer.mlp] {
            Node::Affine { terms, bias: None } if terms.len() == 1 && terms[0].0 == layer.active => terms[0].1,
            other => return Err(format!("layer {l}: the MLP's output map is {other:?}")),
        };
        let (up_op, down_op) = (Arc::clone(&out.operators[up]), Arc::clone(&out.operators[down]));
        let bias_op = bias.map(|b| Arc::clone(&out.operators[b]));
        let width = up_op.rows.width();
        if up_op.rows.group_count() != 1 || down_op.cols.width() != width {
            return Err(format!("layer {l}: the MLP's units are not one native interface"));
        }
        let units = Interface::native(width + mine.len()).map_err(|e| e.to_string())?;
        let (mut gate, mut offset, mut output) = (up_op.matrix(), bias_op.as_ref().map(|b| b.matrix()), down_op.matrix());
        for (j, (k, i)) in mine.iter().enumerate() {
            if *i >= width {
                return Err(format!("layer {l} has no unit {i}"));
            }
            gate.push_row(gate.row(*i).to_owned().view()).map_err(|e| e.to_string())?;
            if let Some(offset) = offset.as_mut() {
                offset.push_row(offset.row(*i).to_owned().view()).map_err(|e| e.to_string())?;
            }
            let half = output.column(*i).mapv(|v| v / 2.0);
            output.column_mut(*i).assign(&half);
            output.push_column(half.view()).map_err(|e| e.to_string())?;
            copies[*k] = width + j;
        }
        out.operators[up] = operator(&up_op, units.clone(), up_op.cols.clone(), gate)?;
        if let (Some(b), Some(op), Some(values)) = (bias, bias_op, offset) {
            out.operators[b] = operator(&op, units.clone(), op.cols.clone(), values)?;
        }
        out.operators[down] = operator(&down_op, down_op.rows.clone(), units, output)?;
    }
    Ok((out, copies))
}

/// The split arm's start from the unsplit library's `start`: per trainable operator (the two
/// explanations list them in one order), the duplicated units' gate rows (with their biases) copied
/// to the copies and their output columns halved, `μ / 2` and `ln σ − ln 2`, in the unit and its
/// copy; every group active as the base group of its name is, a copy's as its unit's.
fn split_start(base: &Explanation, split: &Explanation, start: &Start, units: &[(usize, usize)], copies: &[usize]) -> Result<Start, String> {
    let name = |e: &Explanation, i: usize| e.artifact.program.operators[e.trainable[i]].name.clone();
    if base.trainable.len() != split.trainable.len() || base.trainable.len() != start.mean.len() {
        return Err("the split changed the trainable operators".into());
    }
    let (mut mean, mut log_sd) = (Vec::with_capacity(start.mean.len()), Vec::with_capacity(start.mean.len()));
    for i in 0..start.mean.len() {
        let (operator, (m, s)) = (name(split, i), (&start.mean[i], &start.log_sd[i]));
        if operator != name(base, i) {
            return Err(format!("trainable operator {i}: {operator} against {}", name(base, i)));
        }
        let layer: Option<usize> = operator.strip_prefix("library.l").and_then(|rest| rest.split('.').next()).and_then(|l| l.parse().ok());
        let duplicated: Vec<usize> = units.iter().filter(|(l, _)| Some(*l) == layer).map(|(_, u)| *u).collect();
        if duplicated.is_empty() || !operator.contains(".mlp.") {
            mean.push(m.clone());
            log_sd.push(s.clone());
        } else if operator.ends_with(".out") {
            let (mut m, mut s) = (m.clone(), s.clone());
            for u in &duplicated {
                m.column_mut(*u).mapv_inplace(|v| v / 2.0);
                s.column_mut(*u).mapv_inplace(|v| v - LN_2);
            }
            mean.push(concatenate(Axis(1), &[m.view(), m.select(Axis(1), &duplicated).view()]).map_err(|e| e.to_string())?);
            log_sd.push(concatenate(Axis(1), &[s.view(), s.select(Axis(1), &duplicated).view()]).map_err(|e| e.to_string())?);
        } else {
            mean.push(concatenate(Axis(0), &[m.view(), m.select(Axis(0), &duplicated).view()]).map_err(|e| e.to_string())?);
            log_sd.push(concatenate(Axis(0), &[s.view(), s.select(Axis(0), &duplicated).view()]).map_err(|e| e.to_string())?);
        }
    }
    // Groups by name; a copy's (`library.l{l}.mlp.f{c}.…`) as its unit's.
    let active: BTreeMap<&str, bool> = base.groups.iter().zip(&start.active).map(|(g, a)| (g.name.as_str(), *a)).collect();
    let renamed: BTreeMap<String, String> = units
        .iter()
        .zip(copies)
        .flat_map(|((l, u), c)| ["gate", "up", "out"].map(|part| (format!("library.l{l}.mlp.f{c}.{part}"), format!("library.l{l}.mlp.f{u}.{part}"))))
        .collect();
    let active = split
        .groups
        .iter()
        .map(|g| {
            let own = renamed.get(&g.name).map_or(g.name.as_str(), String::as_str);
            active.get(own).copied().ok_or_else(|| format!("{}: no group of that name before the split", g.name))
        })
        .collect::<Result<Vec<bool>, String>>()?;
    for (i, (m, s)) in mean.iter().zip(&log_sd).enumerate() {
        let dim = split.artifact.program.operators[split.trainable[i]].matrix().dim();
        if m.dim() != dim || s.dim() != dim {
            return Err(format!("{}: {:?} against {dim:?}", name(split, i), m.dim()));
        }
    }
    Ok(Start { mean, log_sd, active, state: None, iterate: None, steps: 0, epoch: start.epoch })
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [export, settings_path, checkpoint, out, mode, list] = &args[..] else {
        return Err("EXPORT SETTINGS.json CHECKPOINT OUT host|gpu LAYER:UNIT[,LAYER:UNIT…]|top:K".into());
    };
    let (export, settings_path, checkpoint, out) = (Path::new(export), Path::new(settings_path), Path::new(checkpoint), Path::new(out));
    let settings: Settings = serde_json::from_slice(&std::fs::read(settings_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if settings.windows.is_some() {
        return Err("an engine export's settings required (no windows)".into());
    }
    if sha256(&export.join("export.json"))? != settings.export_sha256 {
        return Err("export hash mismatch".into());
    }
    let [first, end] = settings.held_out;
    if first >= end || settings.training_sequences == 0 {
        return Err("held-out sequences must be a nonempty range, and training sequences nonempty".into());
    }
    let rows = end.max(settings.training_sequences + if settings.training_sequences > first { end - first } else { 0 });
    let device = match mode.as_str() {
        "host" => Device::host(),
        "gpu" => Device::single_precision(GpuPolicy::Required).map_err(|e| e.to_string())?.ok_or("a GPU required")?,
        _ => return Err("host|gpu required".into()),
    };
    let imported = import_language_model(export, rows, settings.context)?;
    let layer_count = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
    let native = split_sites(&imported.program)?;
    let layers = layer_nodes(&native, layer_count)?;
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else {
        return Err("a token slot".into());
    };
    let sequences: Vec<Vec<u32>> = tokens.chunks(settings.context).map(<[u32]>::to_vec).collect();
    let held_out = &sequences[first..end];
    let train: Vec<Vec<u32>> = sequences[..first].iter().chain(&sequences[end..]).take(settings.training_sequences).cloned().collect();
    if train.len() != settings.training_sequences {
        return Err("the export holds fewer training sequences than asked for".into());
    }
    let scope = |e: Explanation| -> Result<Explanation, String> {
        match &settings.blocks {
            Some(blocks) => library_mdl::scoped(&e, blocks),
            None => Ok(e),
        }
    };
    let units: Vec<(usize, usize)> = match list.strip_prefix("top:") {
        Some(k) => {
            let k: usize = k.parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
            let base = scope(library_mdl::explanation(&native, &layers)?)?;
            let posterior = library_mdl::checkpoint_posterior(&base, checkpoint)?;
            let divergences = posterior.divergences();
            let mut ranked: Vec<(f64, (usize, usize))> = base
                .layers
                .iter()
                .enumerate()
                .flat_map(|(l, layer)| layer.functions.iter().enumerate().map(move |(i, groups)| (l, i, groups)))
                .filter(|(_, _, groups)| groups.iter().all(|g| posterior.active[*g]))
                .map(|(l, i, groups)| (groups.iter().map(|g| divergences[*g]).sum(), (l, i)))
                .collect();
            ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
            ranked.into_iter().take(k).map(|(_, unit)| unit).collect()
        }
        None => list
            .split(',')
            .map(|s| {
                let (l, i) = s.split_once(':').ok_or_else(|| format!("{s}: LAYER:UNIT"))?;
                Ok((l.parse().map_err(|e: std::num::ParseIntError| e.to_string())?, i.parse().map_err(|e: std::num::ParseIntError| e.to_string())?))
            })
            .collect::<Result<_, String>>()?,
    };
    if units.is_empty() {
        return Err("no units to split".into());
    }
    let (split, copies) = duplicate(&native, &layers, &units)?;
    let split_layers = layer_nodes(&split, layer_count)?;
    let base_explanation = scope(library_mdl::explanation(&native, &layers)?)?;
    let split_explanation = scope(library_mdl::explanation(&split, &split_layers)?)?;
    split_explanation.artifact.validate_coverage(&split)?;
    let start = library_mdl::checkpoint_start(&base_explanation, checkpoint)?;
    let split_from = split_start(&base_explanation, &split_explanation, &start, &units, &copies)?;
    let base_from = Start { state: None, steps: 0, ..start };
    let mut results = Vec::new();
    for (name, program, explanation, from) in [("base", &native, &base_explanation, base_from), ("split", &split, &split_explanation, split_from)] {
        let dir = out.join(name);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let fit = library_mdl::fit_from(&device, program, explanation, &train, held_out, &settings.fit, &settings.export_sha256, Some(&dir.join("checkpoint.bin")), None, Some(from))?;
        save(&dir.join("REPORT.json"), &serde_json::to_value(&fit.report).map_err(|e| e.to_string())?)?;
        let artifact = library_mdl::posterior_mean(&explanation, &fit.posterior)?.f32_literals()?;
        std::fs::write(dir.join("artifact.bin"), artifact.to_bytes()?).map_err(|e| e.to_string())?;
        // Per listed function, its groups' divergences in bits and whether they survive; for the
        // split, the unit and its copy.
        let divergences = fit.posterior.divergences();
        let groups = |l: usize, i: usize| -> Value {
            let ids = &explanation.layers[l].functions[i];
            json!(ids.iter().map(|g| json!({
                "group": explanation.groups[*g].name,
                "divergence_bits": divergences[*g] / LN_2,
                "active": fit.posterior.active[*g],
            })).collect::<Vec<_>>())
        };
        let functions: Vec<Value> = units
            .iter()
            .zip(&copies)
            .map(|((l, i), c)| if name == "split" { json!({"layer": l, "unit": i, "unit_groups": groups(*l, *i), "copy": c, "copy_groups": groups(*l, *c)}) } else { json!({"layer": l, "unit": i, "unit_groups": groups(*l, *i)}) })
            .collect();
        let last = fit.report.epochs.last().ok_or("a fit without epochs")?;
        let description = (fit.posterior.description() + explanation.fixed_nats) / LN_2;
        results.push(json!({
            "fit": name,
            "objective_bits": fit.report.objective_bits,
            "description_bits": description,
            "data_bits": fit.report.objective_bits - description,
            "active_groups": fit.report.active_groups,
            "epochs": fit.report.epochs.len(),
            "held_out": last.held_out,
            "functions": functions,
        }));
    }
    let delta = |key: &str| results[1][key].as_f64().zip(results[0][key].as_f64()).map(|(a, b)| a - b);
    let summary = json!({
        "units": units,
        "delta_objective_bits": delta("objective_bits"),
        "delta_description_bits": delta("description_bits"),
        "delta_data_bits": delta("data_bits"),
        "fits": results,
    });
    log::info!("split summary: {summary}");
    save(&out.join("SPLIT.json"), &summary)
}

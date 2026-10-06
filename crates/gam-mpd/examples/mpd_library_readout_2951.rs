//! The read-out of a library explanation (`gam_mpd::library_readout`, #2951) on an export's
//! held-out token rows: per surviving function its contexts, written and read tokens,
//! attention summary and inputs, with every context decoded to text.
//!
//! EXPORT TOKENIZER SETTINGS.json OUT.json [ARTIFACT]
//! costs EXPORT CHECKPOINT OUT.json [READOUT.json]
//! removals EXPORT REMOVALS.json OUT.json [ARTIFACT]
//!
//! The `removals` mode measures every function's removal (`Library::removal_effects`) on held-out
//! rows: per function the mean change in KL(M || P) per token in bits and its quantiles over the
//! tokens, and per token the effective number of functions its prediction rests on,
//! `(Σ_i |ΔKL_i|)² / Σ_i ΔKL_i²`, with its quantiles. `REMOVALS.json` is `{export_sha256, context,
//! held_out: [start, end), batch, numeric_bytes, tile_rows}`.
//!
//! Without `ARTIFACT` (a `library_mdl` posterior-mean `artifact.bin`), the read-out is of the
//! library's starting point, where every function is a native head or neuron. The model runs on
//! CUDA in float64 when present, else on the Apple GPU in f32 when present, else on the host;
//! vocabulary-wide searches and wiring products run on the single-precision device when present
//! (CUDA f32 or the Apple GPU).
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    artifact::Artifact,
    engine::{log_to_stderr, sha256},
    import::import_language_model,
    library_mdl,
    library_readout::{self, Library, Vocabulary},
    operator_program::{OperatorProgram, SlotValues},
    resident_causal_fit::fixed_head_target::Teacher,
    run_check::{LayerNodes, layer_nodes, split_sites},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, time::Instant};

/// The model's device (CUDA in float64, else the Apple GPU, else the host) and the device of
/// vocabulary-wide products (the single-precision device, else the model's).
fn devices() -> Result<(Device, Device), String> {
    let wide = Device::single_precision(GpuPolicy::Auto).map_err(|e| e.to_string())?;
    let model = match Device::accelerator(GpuPolicy::Auto).map_err(|e| e.to_string())? {
        Some(cuda) => cuda,
        None => wide.clone().unwrap_or_else(Device::host),
    };
    let wide = wide.unwrap_or_else(|| model.clone());
    Ok((model, wide))
}

/// The split native program, its layers, the export's first `sequences` token rows of `context`,
/// and the artifact (the library's starting point without one).
fn load(export: &Path, sequences: usize, context: usize, artifact: Option<&Path>) -> Result<(OperatorProgram, Vec<LayerNodes>, Vec<u32>, Artifact), String> {
    let imported = import_language_model(export, sequences, context)?;
    let layer_count = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
    let native = split_sites(&imported.program)?;
    let layers = layer_nodes(&native, layer_count)?;
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else {
        return Err("a token slot".into());
    };
    let tokens = tokens.clone();
    drop(imported);
    let artifact = match artifact {
        Some(path) => Artifact::from_bytes(&std::fs::read(path).map_err(|e| e.to_string())?, &native.declarations)?,
        None => library_mdl::explanation(&native, &layers)?.artifact,
    };
    artifact.validate_coverage(&native)?;
    Ok((native, layers, tokens, artifact))
}

/// Where a fit checkpoint's code length goes: per function its bits, summed per layer and kind,
/// and the most expensive functions with, from a read-out of the same explanation, their
/// importance, important fractions and top contexts.
fn costs(args: &[String]) -> Result<(), String> {
    let (export, checkpoint, out, readout) = match args {
        [e, c, o] => (e, c, o, None),
        [e, c, o, r] => (e, c, o, Some(Path::new(r))),
        _ => return Err("costs EXPORT CHECKPOINT OUT.json [READOUT.json]".into()),
    };
    let imported = import_language_model(Path::new(export), 1, 1)?;
    let layer_count = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
    let native = split_sites(&imported.program)?;
    drop(imported);
    let layers = layer_nodes(&native, layer_count)?;
    let explanation = library_mdl::explanation(&native, &layers)?;
    let posterior = library_mdl::checkpoint_posterior(&explanation, Path::new(checkpoint))?;
    let costs = library_readout::function_costs(&explanation, &posterior);
    // Totals count each prior group once (a key and value group may serve several query heads).
    let by_layer = library_readout::layer_costs(&explanation, &posterior);
    let total: f64 = by_layer.values().map(|[_, bits]| bits).sum();
    let divergence: f64 = by_layer.values().map(|[kl, _]| kl).sum();
    let readout: Option<Value> = readout.map(|p| serde_json::from_slice(&std::fs::read(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())).transpose()?;
    let by_name: BTreeMap<String, &Value> =
        readout.as_ref().and_then(|r| r["functions"].as_array()).into_iter().flatten().filter_map(|f| Some((f["name"].as_str()?.to_string(), f))).collect();
    let mut order: Vec<usize> = (0..costs.len()).collect();
    order.sort_by(|a, b| costs[*b].bits.total_cmp(&costs[*a].bits));
    let expensive: Vec<Value> = order
        .iter()
        .take(20)
        .map(|i| {
            let c = &costs[*i];
            let f = by_name.get(&c.name);
            json!({
                "name": c.name, "bits": c.bits, "parts": c.parts,
                "important_fraction": f.map(|f| f["important_fraction"].clone()), "importance": f.map(|f| f["importance"].clone()),
                "contexts": f.map(|f| f["contexts"].as_array().map(|a| a.iter().take(3).map(|c| json!({"before": c["before"], "token": c["token"]})).collect::<Vec<_>>())),
            })
        })
        .collect();
    let report = json!({
        "export": export,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(Path::new(checkpoint))?,
        "readout": readout.as_ref().map(|_| args[3].clone()),
        "source_revision": option_env!("GAM_BUILD_GIT_SHA"),
        "active_groups": posterior.active.iter().filter(|a| **a).count(),
        "groups": posterior.active.len(),
        "bits": total,
        "divergence_bits": divergence,
        "by_layer_and_kind": by_layer,
        "most_expensive": expensive,
        "functions": costs,
    });
    std::fs::write(out, serde_json::to_vec(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Removals {
    export_sha256: String,
    context: usize,
    held_out: [usize; 2],
    batch: usize,
    numeric_bytes: usize,
    tile_rows: usize,
}

/// The `q`-quantile of `values` (sorted ascending), by the nearest rank.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    sorted[((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1]
}

fn removals(args: &[String]) -> Result<(), String> {
    let (export, settings_path, out, artifact_path) = match args {
        [e, s, o] => (e, s, o, None),
        [e, s, o, a] => (e, s, o, Some(Path::new(a))),
        _ => return Err("removals EXPORT REMOVALS.json OUT.json [ARTIFACT]".into()),
    };
    let export = Path::new(export);
    let settings: Removals = serde_json::from_slice(&std::fs::read(settings_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if sha256(&export.join("export.json"))? != settings.export_sha256 {
        return Err("export hash mismatch".into());
    }
    let [first, end] = settings.held_out;
    if first >= end {
        return Err("an empty held-out range".into());
    }
    let started = Instant::now();
    let (model, wide) = devices()?;
    let (native, layers, tokens, artifact) = load(export, end, settings.context, artifact_path)?;
    let sequences: Vec<Vec<u32>> = tokens.chunks(settings.context).skip(first).map(<[u32]>::to_vec).collect();
    let library = Library::new(&model, &wide, &native, &layers, &artifact, settings.numeric_bytes, settings.tile_rows)?;
    let teacher = Teacher::new(&model, &native, settings.tile_rows, settings.numeric_bytes)?;
    let effects = library.removal_effects(&teacher, &sequences, None, settings.batch)?;
    let functions: Vec<Value> = library
        .functions()
        .iter()
        .zip(effects.outer_iter())
        .map(|(f, row)| {
            let mut sorted = row.to_vec();
            sorted.sort_by(f64::total_cmp);
            json!({"name": f.name, "layer": f.layer, "kind": f.kind, "mean_bits": row.mean(),
                   "quantiles": ([0.5, 0.9, 0.99, 1.0].map(|q| quantile(&sorted, q)))})
        })
        .collect();
    let mut participation: Vec<f64> = effects
        .columns()
        .into_iter()
        .map(|column| {
            let (absolute, square) = column.iter().fold((0.0, 0.0), |(a, q), v| (a + v.abs(), q + v * v));
            if square > 0.0 { absolute * absolute / square } else { 0.0 }
        })
        .collect();
    participation.sort_by(f64::total_cmp);
    let report = json!({
        "export": export.display().to_string(),
        "settings_sha256": sha256(Path::new(settings_path))?,
        "artifact": artifact_path.map(|p| p.display().to_string()),
        "artifact_sha256": artifact_path.map(sha256).transpose()?,
        "source_revision": option_env!("GAM_BUILD_GIT_SHA"),
        "model_device": model.name(),
        "held_out_sequences": [first, end],
        "tokens": effects.ncols(),
        "participation_quantiles": ([0.1, 0.5, 0.9].map(|q| quantile(&participation, q))),
        "participation_mean": participation.iter().sum::<f64>() / participation.len().max(1) as f64,
        "functions": functions,
        "seconds": started.elapsed().as_secs_f64(),
    });
    log::info!("removal effects of {} functions on {} tokens in {:.1} s", effects.nrows(), effects.ncols(), started.elapsed().as_secs_f64());
    std::fs::write(out, serde_json::to_vec(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    export_sha256: String,
    context: usize,
    /// The held-out sequences `[start, end)` of the export's token rows.
    held_out: [usize; 2],
    /// Tokens of text shown before each context's token.
    window: usize,
    readout: library_readout::Settings,
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "costs") {
        return costs(&args[1..]);
    }
    if args.first().is_some_and(|a| a == "removals") {
        return removals(&args[1..]);
    }
    let (export, tokenizer, settings_path, out, artifact_path) = match &args[..] {
        [e, t, s, o] => (e, t, s, o, None),
        [e, t, s, o, a] => (e, t, s, o, Some(Path::new(a))),
        _ => return Err("EXPORT TOKENIZER SETTINGS.json OUT.json [ARTIFACT]".into()),
    };
    let (export, settings_path) = (Path::new(export), Path::new(settings_path));
    let settings: Settings = serde_json::from_slice(&std::fs::read(settings_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if sha256(&export.join("export.json"))? != settings.export_sha256 {
        return Err("export hash mismatch".into());
    }
    let [first, end] = settings.held_out;
    if first >= end {
        return Err("an empty held-out range".into());
    }
    let started = Instant::now();
    let (model, wide) = devices()?;
    let (native, layers, tokens, artifact) = load(export, end, settings.context, artifact_path)?;
    let sequences: Vec<Vec<u32>> = tokens.chunks(settings.context).skip(first).map(<[u32]>::to_vec).collect();
    let library = Library::new(&model, &wide, &native, &layers, &artifact, settings.readout.numeric_bytes, settings.readout.tile_rows)?;
    let teacher = Teacher::new(&model, &native, settings.readout.tile_rows, settings.readout.numeric_bytes)?;
    // The library and the teacher hold what the read-out reads; the programs they came from go.
    drop((native, layers, artifact));
    let readout = library_readout::read_out(&library, &teacher, &sequences, &settings.readout)?;
    let vocabulary = Vocabulary::from_tokenizer(Path::new(tokenizer))?;
    let text = |t: u32| vocabulary.text(&[t]);
    let mut functions = serde_json::to_value(&readout.functions).map_err(|e| e.to_string())?;
    // Decoded text beside every token id.
    for (function, value) in readout.functions.iter().zip(functions.as_array_mut().ok_or("functions")?) {
        for (list, contexts) in [("contexts", &function.contexts), ("supports", &function.supports), ("opposes", &function.opposes)] {
            for (c, context) in contexts.iter().enumerate() {
                let sequence = &sequences[context.sequence];
                let start = context.position.saturating_sub(settings.window);
                let entry = &mut value[list][c];
                entry["before"] = json!(vocabulary.text(&sequence[start..context.position]));
                entry["token"] = json!(text(sequence[context.position]));
                entry["predicted"] = json!(text(readout.predicted[context.sequence * settings.context + context.position]));
                entry["sequence"] = json!(first + context.sequence);
                if let Some(source) = context.source {
                    entry["source_token"] = json!(text(sequence[source]));
                }
            }
        }
        for field in ["supports_token", "opposes_token"] {
            if let Some(token) = value[field].as_u64() {
                value[format!("{field}_text")] = json!(text(token as u32));
            }
        }
        for list in ["promoted", "suppressed", "reads"] {
            for entry in value[list].as_array_mut().into_iter().flatten() {
                entry["text"] = json!(text(entry["token"].as_u64().ok_or("token")? as u32));
            }
        }
        if let Some(attention) = value.get_mut("attention").filter(|a| !a.is_null()) {
            for entry in attention["sources"].as_array_mut().into_iter().flatten() {
                entry["text"] = json!(text(entry["token"].as_u64().ok_or("token")? as u32));
            }
            for entry in attention["ov"].as_array_mut().into_iter().flatten() {
                entry["source_text"] = json!(text(entry["source"].as_u64().ok_or("source")? as u32));
                entry["output_text"] = json!(text(entry["output"].as_u64().ok_or("output")? as u32));
            }
        }
    }
    let report: Value = json!({
        "export": export.display().to_string(),
        "export_sha256": settings.export_sha256,
        "settings_sha256": sha256(settings_path)?,
        "artifact": artifact_path.map(|p| p.display().to_string()),
        "artifact_sha256": artifact_path.map(sha256).transpose()?,
        "source_revision": option_env!("GAM_BUILD_GIT_SHA"),
        "model_device": model.name(),
        "wide_device": wide.name(),
        "held_out_sequences": [first, end],
        "held_out_tokens": readout.held_out_tokens,
        "measured_tokens": readout.measured_tokens,
        "participation": readout.participation,
        "surviving": readout.functions.len(),
        "removed": readout.removed,
        "core": readout.core,
        "wiring": readout.wiring,
        "functions": functions,
        "seconds": started.elapsed().as_secs_f64(),
    });
    std::fs::write(out, serde_json::to_vec(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    log::info!("read-out of {} functions in {:.1} s: {out}", readout.functions.len(), started.elapsed().as_secs_f64());
    Ok(())
}

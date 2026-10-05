//! The read-out of a library explanation (`gam_mpd::library_readout`, #2951) on an export's
//! held-out token rows: per surviving function its frequency, contexts, written and read tokens,
//! attention summary and inputs, with every context decoded to text.
//!
//! EXPORT TOKENIZER SETTINGS.json OUT.json [ARTIFACT]
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
    library_readout::{self, Vocabulary},
    operator_program::SlotValues,
    run_check::{layer_nodes, split_sites},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{path::Path, time::Instant};

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
    let wide = Device::single_precision(GpuPolicy::Auto).map_err(|e| e.to_string())?;
    let model = match Device::accelerator(GpuPolicy::Auto).map_err(|e| e.to_string())? {
        Some(cuda) => cuda,
        None => wide.clone().unwrap_or_else(Device::host),
    };
    let wide = wide.unwrap_or_else(|| model.clone());
    let imported = import_language_model(export, end, settings.context)?;
    let layer_count = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
    let native = split_sites(&imported.program)?;
    let layers = layer_nodes(&native, layer_count)?;
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else {
        return Err("a token slot".into());
    };
    let tokens = tokens.clone();
    drop(imported);
    let sequences: Vec<Vec<u32>> = tokens.chunks(settings.context).skip(first).map(<[u32]>::to_vec).collect();
    let artifact = match artifact_path {
        Some(path) => Artifact::from_bytes(&std::fs::read(path).map_err(|e| e.to_string())?, &native.declarations)?,
        None => library_mdl::explanation(&native, &layers)?.artifact,
    };
    artifact.validate_coverage(&native)?;
    let readout = library_readout::read_out(&model, &wide, &native, &layers, &artifact, &sequences, &settings.readout)?;
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
        "mean_logit": readout.mean_logit,
        "mean_attributed": readout.mean_attributed,
        "participation": readout.participation,
        "background": readout.functions.iter().enumerate().filter(|(_, f)| f.background).map(|(i, _)| i).collect::<Vec<_>>(),
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

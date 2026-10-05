//! The library explanation of a language model, fitted end to end by variational minimum
//! description length (`gam_mpd::library_mdl`, #2951), on an export's token rows, and its
//! posterior-mean artifact scored on held-out rows.
//!
//! EXPORT SETTINGS.json FRESH_OUT host|cuda
//!
//! The last `held_out` sequences are never fitted. On CUDA the fit runs in f32 storage; every
//! reported divergence and cost is float64.
use gam_gpu::{
    GpuPolicy,
    tensor::{Device, Storage},
};
use gam_mpd::{
    acceptance::{CostCache, structural_cost},
    artifact::Artifact,
    engine::{log_to_stderr, sha256},
    import::import_language_model,
    library_mdl,
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
    sequences: usize,
    context: usize,
    held_out: usize,
    /// Sequences per evaluation batch.
    evaluation_batch: usize,
    fit: library_mdl::Settings,
}

fn save(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(path, serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [export, settings_path, out, mode] = &args[..] else {
        return Err("EXPORT SETTINGS.json FRESH_OUT host|cuda".into());
    };
    let (export, settings_path, out) = (Path::new(export), Path::new(settings_path), Path::new(out));
    let settings: Settings = serde_json::from_slice(&std::fs::read(settings_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if sha256(&export.join("export.json"))? != settings.export_sha256 {
        return Err("export hash mismatch".into());
    }
    if settings.held_out == 0 || settings.held_out >= settings.sequences || settings.evaluation_batch == 0 {
        return Err("held-out sequences must leave training sequences, and batches must be nonempty".into());
    }
    if out.exists() {
        return Err("fresh output directory required".into());
    }
    let (exact, fitting) = match mode.as_str() {
        "host" => (Device::host(), Device::host()),
        "cuda" => {
            let device = Device::accelerator(GpuPolicy::Required).map_err(|e| e.to_string())?.ok_or("CUDA required")?;
            let fitting = device.with_storage(Storage::F32).map_err(|e| e.to_string())?;
            (device, fitting)
        }
        _ => return Err("host|cuda required".into()),
    };
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let imported = import_language_model(export, settings.sequences, settings.context)?;
    let layer_count = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
    let native = split_sites(&imported.program)?;
    let layers = layer_nodes(&native, layer_count)?;
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else {
        return Err("a token slot".into());
    };
    let sequences: Vec<Vec<u32>> = tokens.chunks(settings.context).map(<[u32]>::to_vec).collect();
    let (train, held_out) = sequences.split_at(settings.sequences - settings.held_out);
    let explanation = library_mdl::explanation(&native, &layers)?;
    explanation.artifact.validate_coverage(&native)?;
    let (bytes, tiles) = (settings.fit.numeric_bytes, settings.fit.head_tile_rows);
    let start_held_out = library_mdl::mean_divergence(&exact, &native, &explanation.artifact, held_out, settings.evaluation_batch, bytes, tiles)?;
    let mut cache = CostCache::default();
    let native_cost = structural_cost(&Artifact::native(&native)?.f32_literals()?, &mut cache)?;
    let start_cost = structural_cost(&explanation.artifact.f32_literals()?, &mut cache)?;
    let provenance = json!({
        "export": export.display().to_string(),
        "export_sha256": settings.export_sha256,
        "settings_sha256": sha256(settings_path)?,
        "source_revision": option_env!("GAM_BUILD_GIT_SHA"),
        "fitting_device": fitting.name(),
        "exact_device": exact.name(),
        "training_sequences": train.len(),
        "held_out_sequences": held_out.len(),
        "context": settings.context,
        "groups": explanation.groups.len(),
        "start_held_out_mean_kl": start_held_out,
        "native_c32_bits": native_cost.total(),
        "start_c32_bits": start_cost.total(),
    });
    log::info!("library start: {provenance}");
    save(&out.join("START.json"), &provenance)?;
    let fit = library_mdl::fit(&fitting, &native, &explanation, train, &settings.fit)?;
    save(&out.join("REPORT.json"), &serde_json::to_value(&fit.report).map_err(|e| e.to_string())?)?;
    let artifact = library_mdl::posterior_mean(&explanation, &fit.posterior)?.f32_literals()?;
    artifact.validate_coverage(&native)?;
    std::fs::write(out.join("artifact.bin"), artifact.to_bytes()?).map_err(|e| e.to_string())?;
    let cost = structural_cost(&artifact, &mut cache)?;
    let held_out_kl = library_mdl::mean_divergence(&exact, &native, &artifact, held_out, settings.evaluation_batch, bytes, tiles)?;
    let fitted_kl = library_mdl::mean_divergence(&exact, &native, &artifact, &train[..held_out.len().min(train.len())], settings.evaluation_batch, bytes, tiles)?;
    let summary = json!({
        "provenance": provenance,
        "objective_bits": fit.report.objective_bits,
        "active_groups": fit.report.active_groups,
        "epochs": fit.report.epochs.len(),
        "removals": fit.report.removals.iter().map(|r| r.removed).collect::<Vec<_>>(),
        "c32_bits": cost.total(),
        "literals": cost.literals,
        "structure_bits": cost.structure_bits,
        "binding_bits": cost.binding_bits,
        "held_out_mean_kl": held_out_kl,
        "fitted_rows_mean_kl": fitted_kl,
        "seconds": started.elapsed().as_secs_f64(),
        "scope": "Clean next-token divergence only; the intervention panel is scored separately from artifact.bin.",
    });
    log::info!("library summary: {summary}");
    save(&out.join("SUMMARY.json"), &summary)
}

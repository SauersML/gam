//! The library explanation of a language model, fitted end to end by variational minimum
//! description length (`gam_mpd::library_mdl`, #2951), on an export's token rows, and its
//! posterior-mean artifact scored on held-out rows.
//!
//! EXPORT SETTINGS.json OUT host|cuda
//!
//! The sequences `held_out = [start, end)` are never fitted (for VPD-4L's `vpd4l_clean4096`, rows
//! 1024..1056 are `vpd4l_frontier32`, which holds the intervention panel's passages). On CUDA the
//! fit runs in f32 storage; every reported divergence and cost is float64. The fit is checkpointed in
//! `OUT/checkpoint.bin` after every epoch; rerunning the same command resumes it.
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
    held_out: [usize; 2],
    /// Sequences per evaluation batch.
    evaluation_batch: usize,
    /// The declared interventions: every held native quantity (each head's read, each MLP's
    /// activations) scaled by each of these; empty for clean data alone.
    intervention_scales: Vec<f64>,
    fit: library_mdl::Settings,
}

fn save(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(path, serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [export, settings_path, out, mode] = &args[..] else {
        return Err("EXPORT SETTINGS.json OUT host|cuda".into());
    };
    let (export, settings_path, out) = (Path::new(export), Path::new(settings_path), Path::new(out));
    let settings: Settings = serde_json::from_slice(&std::fs::read(settings_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if sha256(&export.join("export.json"))? != settings.export_sha256 {
        return Err("export hash mismatch".into());
    }
    let [first, end] = settings.held_out;
    if first >= end || end > settings.sequences || end - first == settings.sequences || settings.evaluation_batch == 0 {
        return Err("held-out sequences must be a nonempty range leaving training sequences, and batches must be nonempty".into());
    }
    let checkpoint = out.join("checkpoint.bin");
    if out.exists() && !checkpoint.exists() {
        return Err("a fresh output directory, or one holding this fit's checkpoint, required".into());
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
    let held_out = &sequences[first..end];
    let train: Vec<Vec<u32>> = sequences[..first].iter().chain(&sequences[end..]).cloned().collect();
    let train = &train[..];
    let explanation = library_mdl::explanation(&native, &layers)?;
    explanation.artifact.validate_coverage(&native)?;
    let interventions: Vec<library_mdl::Intervention> = (0..explanation.holdings.len())
        .flat_map(|holding| settings.intervention_scales.iter().map(move |&scale| library_mdl::Intervention { holding, scale }))
        .collect();
    let (bytes, tiles) = (settings.fit.numeric_bytes, settings.fit.head_tile_rows);
    let measure = |artifact: &Artifact, sequences: &[Vec<u32>], declared: &[library_mdl::Intervention]| {
        library_mdl::Measure { device: &exact, native: &native, holdings: &explanation.holdings, batch_sequences: settings.evaluation_batch, numeric_bytes: bytes, tile_rows: tiles }
            .mean_divergence(artifact, sequences, declared)
    };
    let start_held_out = measure(&explanation.artifact, held_out, &[])?;
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
        "held_out_sequences": [first, end],
        "context": settings.context,
        "groups": explanation.groups.len(),
        "start_held_out_mean_kl": start_held_out,
        "native_c32_bits": native_cost.total(),
        "start_c32_bits": start_cost.total(),
    });
    log::info!("library start: {provenance}");
    save(&out.join("START.json"), &provenance)?;
    let fit = library_mdl::fit(&fitting, &native, &explanation, train, &interventions, &settings.fit, Some(&checkpoint))?;
    save(&out.join("REPORT.json"), &serde_json::to_value(&fit.report).map_err(|e| e.to_string())?)?;
    let artifact = library_mdl::posterior_mean(&explanation, &fit.posterior)?.f32_literals()?;
    artifact.validate_coverage(&native)?;
    std::fs::write(out.join("artifact.bin"), artifact.to_bytes()?).map_err(|e| e.to_string())?;
    let cost = structural_cost(&artifact, &mut cache)?;
    let held_out_kl = measure(&artifact, held_out, &[])?;
    let held_out_intervened_kl = if interventions.is_empty() { None } else { Some(measure(&artifact, held_out, &interventions)?) };
    let fitted_kl = measure(&artifact, &train[..held_out.len().min(train.len())], &[])?;
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
        "held_out_intervened_mean_kl": held_out_intervened_kl,
        "interventions": interventions.len(),
        "fitted_rows_mean_kl": fitted_kl,
        "seconds": started.elapsed().as_secs_f64(),
        "scope": "Next-token divergence on held-out rows, clean and under every declared holding scale; the 80-episode panel (including mixes) is scored separately from artifact.bin.",
    });
    log::info!("library summary: {summary}");
    save(&out.join("SUMMARY.json"), &summary)
}

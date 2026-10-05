//! The library explanation of a language model, fitted end to end by variational minimum
//! description length on interchange experiments (`gam_mpd::library_mdl`, #2951), on an export's
//! token rows, and scored on held-out rows after every epoch.
//!
//! EXPORT SETTINGS.json OUT host|gpu
//!
//! The sequences `held_out = [start, end)` are never fitted (for VPD-4L's `vpd4l_clean4096`, rows
//! 1024..1056 are `vpd4l_frontier32`); the training sequences are the first `training_sequences`
//! of the others, in order. `gpu` is the single-precision device (CUDA in f32 storage, or the Apple GPU). The
//! fit is checkpointed in `OUT/checkpoint.bin` after every epoch, with its trajectory readable in
//! `OUT/checkpoint.json`; rerunning the same command resumes it.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
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
    training_sequences: usize,
    context: usize,
    held_out: [usize; 2],
    fit: library_mdl::Settings,
}

fn save(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(path, serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [export, settings_path, out, mode] = &args[..] else {
        return Err("EXPORT SETTINGS.json OUT host|gpu".into());
    };
    let (export, settings_path, out) = (Path::new(export), Path::new(settings_path), Path::new(out));
    let settings: Settings = serde_json::from_slice(&std::fs::read(settings_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if sha256(&export.join("export.json"))? != settings.export_sha256 {
        return Err("export hash mismatch".into());
    }
    let [first, end] = settings.held_out;
    if first >= end || settings.training_sequences == 0 {
        return Err("held-out sequences must be a nonempty range, and training sequences nonempty".into());
    }
    // The rows to import: the held-out range and the training sequences around it.
    let rows = end.max(settings.training_sequences + if settings.training_sequences > first { end - first } else { 0 });
    let checkpoint = out.join("checkpoint.bin");
    if out.exists() && !checkpoint.exists() {
        return Err("a fresh output directory, or one holding this fit's checkpoint, required".into());
    }
    let device = match mode.as_str() {
        "host" => Device::host(),
        "gpu" => Device::single_precision(GpuPolicy::Required).map_err(|e| e.to_string())?.ok_or("a GPU required")?,
        _ => return Err("host|gpu required".into()),
    };
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let started = Instant::now();
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
    let explanation = library_mdl::explanation(&native, &layers)?;
    explanation.artifact.validate_coverage(&native)?;
    // A checkpoint there must be this fit's (export, sequences, program, groups, shared
    // parameters) before anything is written.
    let identity = library_mdl::identity(&settings.export_sha256, &native, &explanation, &train, held_out);
    library_mdl::check_checkpoint(&checkpoint, &identity)?;
    let provenance = json!({
        "export": export.display().to_string(),
        "export_sha256": settings.export_sha256,
        "settings_sha256": sha256(settings_path)?,
        "source_revision": option_env!("GAM_BUILD_GIT_SHA"),
        "device": device.name(),
        "training_sequences": train.len(),
        "training_tokens": train.len() * settings.context,
        "held_out_sequences": [first, end],
        "context": settings.context,
        "groups": explanation.groups.len(),
        "identity": identity,
    });
    log::info!("library run: {provenance}");
    save(&out.join("START.json"), &provenance)?;
    let fit = library_mdl::fit(&device, &native, &explanation, &train, held_out, &settings.fit, &settings.export_sha256, Some(&checkpoint))?;
    save(&out.join("REPORT.json"), &serde_json::to_value(&fit.report).map_err(|e| e.to_string())?)?;
    let artifact = library_mdl::posterior_mean(&explanation, &fit.posterior)?.f32_literals()?;
    artifact.validate_coverage(&native)?;
    std::fs::write(out.join("artifact.bin"), artifact.to_bytes()?).map_err(|e| e.to_string())?;
    let summary = json!({
        "provenance": provenance,
        "objective_bits": fit.report.objective_bits,
        "active_groups": fit.report.active_groups,
        "epochs": fit.report.epochs.len(),
        "removals": fit.report.removals.iter().map(|r| r.removed).collect::<Vec<_>>(),
        "start": fit.report.start,
        "held_out": fit.report.end,
        "seconds": started.elapsed().as_secs_f64(),
    });
    log::info!("library summary: {summary}");
    save(&out.join("SUMMARY.json"), &summary)
}

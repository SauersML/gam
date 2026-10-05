//! A shared query-key function proposed on a fitted library explanation and decided by the code
//! length (`gam_mpd::library_sharing`, #2951).
//!
//! EXPORT SETTINGS.json FITTED.artifact.bin|native OUT host|gpu
//!
//! SETTINGS.json is the library fit's (`mpd_library_mdl_2951`), FITTED its posterior-mean artifact
//! (`checkpoint.artifact.bin` or `artifact.bin`), or `native` for the library's start at `M`. From
//! those values the heads' pairs across layers
//! are ranked by the cosine of their score maps, and the first pair is made one shared function.
//! The library as it was (OUT/base) and with the shared function (OUT/shared) are each fitted to
//! convergence from there; the proposal is accepted when the shared library's code length `F` is
//! the smaller.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    artifact::Artifact,
    engine::{log_to_stderr, sha256},
    import::import_language_model,
    library_mdl, library_sharing,
    operator_program::SlotValues,
    run_check::{layer_nodes, split_sites},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::Path;

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
    let [export, settings_path, fitted, out, mode] = &args[..] else {
        return Err("EXPORT SETTINGS.json FITTED.artifact.bin|native OUT host|gpu".into());
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
    let device = match mode.as_str() {
        "host" => Device::host(),
        "gpu" => Device::single_precision(GpuPolicy::Required).map_err(|e| e.to_string())?.ok_or("a GPU required")?,
        _ => return Err("host|gpu required".into()),
    };
    let rows = end.max(settings.training_sequences + if settings.training_sequences > first { end - first } else { 0 });
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
    let start = library_mdl::explanation(&native, &layers)?;
    let base = match fitted.as_str() {
        "native" => start,
        path => library_sharing::warm(&start, &Artifact::from_bytes(&std::fs::read(path).map_err(|e| e.to_string())?, &native.declarations)?)?,
    };
    let pairs = library_sharing::query_key_pairs(&base)?;
    let best = pairs.first().ok_or("no pair of heads in different layers")?;
    let shared = library_sharing::share_query_key(&base, &[best.first, best.second])?;
    shared.artifact.validate_coverage(&native)?;
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let ranked: Vec<Value> = pairs.iter().take(24).map(|p| json!({"first": p.first, "second": p.second, "cosine": p.cosine})).collect();
    save(&out.join("PAIRS.json"), &json!(ranked))?;
    log::info!("sharing heads {:?} and {:?} (score-map cosine {:.4})", best.first, best.second, best.cosine);
    let mut fits = Vec::new();
    for (name, explanation) in [("base", &base), ("shared", &shared)] {
        let dir = out.join(name);
        let checkpoint = dir.join("checkpoint.bin");
        library_mdl::check_checkpoint(&checkpoint, &library_mdl::identity(&settings.export_sha256, &native, explanation, &train, held_out))?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let fit = library_mdl::fit(&device, &native, explanation, &train, held_out, &settings.fit, &settings.export_sha256, Some(&checkpoint))?;
        save(&dir.join("REPORT.json"), &serde_json::to_value(&fit.report).map_err(|e| e.to_string())?)?;
        fits.push(fit.report);
    }
    let summary = json!({
        "export_sha256": settings.export_sha256,
        "settings_sha256": sha256(settings_path)?,
        "pair": {"first": best.first, "second": best.second, "cosine": best.cosine},
        "base_objective_bits": fits[0].objective_bits,
        "shared_objective_bits": fits[1].objective_bits,
        "accepted": fits[1].objective_bits < fits[0].objective_bits,
        "base_held_out": &fits[0].end,
        "shared_held_out": &fits[1].end,
    });
    log::info!("sharing summary: {summary}");
    save(&out.join("SUMMARY.json"), &summary)
}

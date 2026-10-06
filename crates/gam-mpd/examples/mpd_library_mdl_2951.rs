//! The library explanation of a language model, fitted end to end by variational minimum
//! description length on interchange experiments (`gam_mpd::library_mdl`, #2951), on an export's
//! token rows, and scored on held-out rows after every epoch.
//!
//! MODEL SETTINGS.json OUT host|gpu [artifact]
//!
//! `MODEL` is an engine export (`export.json` and its token rows), or a Hugging Face checkpoint
//! directory (`config.json` and its safetensors, one file or sharded) whose token rows come from
//! the settings' `windows` (files of rows of `context` little-endian u32, as
//! `bench/mpd_qwen3_fineweb_2951.py` writes them): its `held_out` file's rows and its `training`
//! file's rows are the two sets. `export_sha256` is `export.json`'s SHA-256, or for a checkpoint
//! the fingerprint of `config.json`'s and every safetensors file's SHA-256 (a mismatch names it).
//!
//! On an export, the sequences `held_out = [start, end)` are never fitted (for VPD-4L's
//! `vpd4l_clean4096`, rows 1024..1056 are `vpd4l_frontier32`); the training sequences are the
//! first `training_sequences` of the others, in order. On a checkpoint, `held_out` is a range of
//! the held-out file's rows and the training sequences are the training file's first rows. `gpu` is the single-precision device (CUDA in f32 storage, or the Apple GPU). The
//! fit is checkpointed in `OUT/checkpoint.bin` after every epoch, with its trajectory readable in
//! `OUT/checkpoint.json`; rerunning the same command resumes it. With `artifact`, nothing is
//! fitted: the explanation the checkpoint holds, at its posterior mean with the device's literals,
//! is written to `OUT/checkpoint.artifact.bin` (a running fit's explanation, read where it is wanted).
//!
//! With `blocks` (block `2l` layer `l`'s attention, `2l + 1` its MLP), the explanation is of those
//! blocks alone and `M` everywhere else (`library_mdl::scoped`): the fast loop for comparing method
//! changes, F against N for one block before a whole-model run.
//!
//! With `transcoders` (`{"dir": D, "layers": [l, ...]}`, `D/layer_{l}.safetensors` circuit-tracer
//! transcoder files), those layers' MLPs are the transcoders' features (`library_transcoder`): the
//! features that fire on the training sequences at `M`'s MLP inputs are written to
//! `OUT/transcoder_l{l}.safetensors` (kept from an earlier run of the same command), with their
//! counts in `OUT/TRANSCODERS.json`; every other layer keeps `M`'s own MLP functions.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    engine::{log_to_stderr, sha256},
    import::{hugging_face_language_model, hugging_face_language_model_prefix, import_language_model},
    library_mdl, library_transcoder,
    operator_program::{OperatorProgram, SlotValues},
    run_check::{LayerNodes, layer_nodes, split_sites},
};
use gam_runtime::warm_start::Fingerprinter;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    export_sha256: String,
    training_sequences: usize,
    context: usize,
    held_out: [usize; 2],
    /// The blocks the explanation explains, when not all of them.
    #[serde(default)]
    blocks: Option<Vec<usize>>,
    /// A Hugging Face checkpoint's token rows (absent for an engine export, which holds its own).
    #[serde(default)]
    windows: Option<Windows>,
    /// A Hugging Face checkpoint cut to its first `layers` blocks, then its final norm and head
    /// (`import::hugging_face_language_model_prefix`): the same fit on a smaller model of the same
    /// tokens, for measuring per-layer costs. All blocks when absent.
    #[serde(default)]
    layers: Option<usize>,
    /// Layers whose MLPs are transcoder features, and the directory of the transcoder files.
    #[serde(default)]
    transcoders: Option<Transcoders>,
    fit: library_mdl::Settings,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Transcoders {
    dir: PathBuf,
    layers: Vec<usize>,
}

/// Per transcoder layer its kept file in `out`: the features that fire on `train` at `M`'s MLP
/// inputs, counted on `device` once and kept for a resumed run, with the counts in
/// `out/TRANSCODERS.json`.
fn transcoder_files(device: &Device, native: &OperatorProgram, layers: &[LayerNodes], settings: &Transcoders, train: &[Vec<u32>], batch: usize, out: &Path) -> Result<BTreeMap<usize, PathBuf>, String> {
    let files: BTreeMap<usize, PathBuf> = settings.layers.iter().map(|&l| (l, out.join(format!("transcoder_l{l}.safetensors")))).collect();
    if files.values().all(|f| f.exists()) {
        return Ok(files);
    }
    let started = Instant::now();
    let transcoders = settings
        .layers
        .iter()
        .map(|&l| Ok((l, library_transcoder::Transcoder::open(&settings.dir.join(format!("layer_{l}.safetensors")))?)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let counts = library_transcoder::firing(device, native, layers, &transcoders, train, batch)?;
    let tokens: usize = train.iter().map(Vec::len).sum();
    let mut record = Vec::new();
    for (l, transcoder) in &transcoders {
        let kept: Vec<usize> = (0..transcoder.features).filter(|&f| counts[l][f] > 0).collect();
        transcoder.write_kept(&kept, &files[l])?;
        let fired: u64 = counts[l].iter().sum();
        record.push(json!({
            "layer": l,
            "features": transcoder.features,
            "kept": kept.len(),
            "active_per_token": fired as f64 / tokens as f64,
            "tokens": tokens,
        }));
        log::info!("transcoder layer {l}: {} of {} features fire on {tokens} training tokens, {:.2} per token", kept.len(), transcoder.features, fired as f64 / tokens as f64);
    }
    save(&out.join("TRANSCODERS.json"), &json!({"layers": record, "seconds": started.elapsed().as_secs_f64()}))?;
    Ok(files)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Windows {
    training: PathBuf,
    held_out: PathBuf,
}

/// The first `count` rows of `context` tokens of a windows file.
fn rows(path: &Path, count: usize, context: usize) -> Result<Vec<Vec<u32>>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < count * context * 4 {
        return Err(format!("{}: fewer than {count} rows of {context} tokens", path.display()));
    }
    Ok(bytes[..count * context * 4]
        .chunks_exact(context * 4)
        .map(|row| row.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        .collect())
}

/// A Hugging Face checkpoint's fingerprint: `config.json`'s and every safetensors file's SHA-256,
/// in name order.
fn checkpoint_digest(dir: &Path) -> Result<String, String> {
    let mut names: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .map(|entry| entry.map(|e| e.path()).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|p| p.file_name().is_some_and(|n| n == "config.json") || p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    names.sort();
    let mut fingerprint = Fingerprinter::new();
    for path in &names {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        fingerprint.absorb_str(name.as_bytes(), &sha256(path)?);
    }
    Ok(fingerprint.finalize().to_hex())
}

fn save(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(path, serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (export, settings_path, out, mode, read_artifact) = match &args[..] {
        [export, settings, out, mode] => (export, settings, out, mode, false),
        [export, settings, out, mode, artifact] if artifact == "artifact" => (export, settings, out, mode, true),
        _ => return Err("EXPORT SETTINGS.json OUT host|gpu [artifact]".into()),
    };
    let (export, settings_path, out) = (Path::new(export), Path::new(settings_path), Path::new(out));
    let settings: Settings = serde_json::from_slice(&std::fs::read(settings_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let checkpoint_model = export.join("config.json").exists() && !export.join("export.json").exists();
    let digest = if checkpoint_model { checkpoint_digest(export)? } else { sha256(&export.join("export.json"))? };
    if digest != settings.export_sha256 {
        return Err(format!("model digest mismatch: {} is {digest}", export.display()));
    }
    let [first, end] = settings.held_out;
    if first >= end || settings.training_sequences == 0 {
        return Err("held-out sequences must be a nonempty range, and training sequences nonempty".into());
    }
    let checkpoint = out.join("checkpoint.bin");
    if (out.exists() || read_artifact) && !checkpoint.exists() {
        return Err("a fresh output directory, or one holding this fit's checkpoint, required".into());
    }
    let device = match mode.as_str() {
        "host" => Device::host(),
        "gpu" => Device::single_precision(GpuPolicy::Required).map_err(|e| e.to_string())?.ok_or("a GPU required")?,
        _ => return Err("host|gpu required".into()),
    };
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let (program, layer_count, train, held_out): (OperatorProgram, usize, Vec<Vec<u32>>, Vec<Vec<u32>>) = if checkpoint_model {
        let windows = settings.windows.as_ref().ok_or("a Hugging Face checkpoint needs the settings' windows")?;
        let text = std::fs::read_to_string(export.join("config.json")).map_err(|e| e.to_string())?;
        let all = serde_json::from_str::<Value>(&text).map_err(|e| e.to_string())?["num_hidden_layers"].as_u64().ok_or("num_hidden_layers")? as usize;
        let (program, layers) = match settings.layers {
            Some(k) => (hugging_face_language_model_prefix(export, k)?.0, k),
            None => (hugging_face_language_model(export, 0..all)?.0, all),
        };
        let held_out = rows(&windows.held_out, end, settings.context)?[first..].to_vec();
        (program, layers, rows(&windows.training, settings.training_sequences, settings.context)?, held_out)
    } else {
        if settings.windows.is_some() || settings.layers.is_some() {
            return Err("an engine export holds its own token rows and blocks; windows and layers are for a Hugging Face checkpoint".into());
        }
        // The rows to import: the held-out range and the training sequences around it.
        let count = end.max(settings.training_sequences + if settings.training_sequences > first { end - first } else { 0 });
        let imported = import_language_model(export, count, settings.context)?;
        let layers = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else {
            return Err("a token slot".into());
        };
        let sequences: Vec<Vec<u32>> = tokens.chunks(settings.context).map(<[u32]>::to_vec).collect();
        let train: Vec<Vec<u32>> = sequences[..first].iter().chain(&sequences[end..]).take(settings.training_sequences).cloned().collect();
        (imported.program, layers, train, sequences[first..end].to_vec())
    };
    if train.len() != settings.training_sequences {
        return Err("the model's token rows hold fewer training sequences than asked for".into());
    }
    let held_out = &held_out[..];
    let native = split_sites(&program)?;
    // The imported program's operators the split replaced are not read again.
    drop(program);
    let layers = layer_nodes(&native, layer_count)?;
    let explanation = match &settings.transcoders {
        Some(transcoders) => {
            let files = transcoder_files(&device, &native, &layers, transcoders, &train, settings.fit.batch_sequences, out)?;
            library_mdl::explanation_with(&native, &layers, &files)?
        }
        None => library_mdl::explanation(&native, &layers)?,
    };
    let explanation = match &settings.blocks {
        Some(blocks) => library_mdl::scoped(&explanation, blocks)?,
        None => explanation,
    };
    explanation.artifact.validate_coverage(&native)?;
    // A checkpoint there must be this fit's (export, sequences, program, groups, shared
    // parameters) before anything is written.
    let identity = library_mdl::identity(&settings.export_sha256, &native, &explanation, &train, held_out);
    library_mdl::check_checkpoint(&checkpoint, &identity)?;
    if read_artifact {
        let artifact = library_mdl::checkpoint_artifact(&explanation, &checkpoint, library_mdl::Literals::of(&device))?;
        artifact.validate_coverage(&native)?;
        return std::fs::write(out.join("checkpoint.artifact.bin"), artifact.to_bytes()?).map_err(|e| e.to_string());
    }
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
        "blocks": settings.blocks,
        "layers": settings.layers,
        "identity": identity,
    });
    log::info!("library run: {provenance}");
    save(&out.join("START.json"), &provenance)?;
    let fit = library_mdl::fit(&device, &native, &explanation, &train, held_out, &settings.fit, &settings.export_sha256, Some(&checkpoint), None)?;
    save(&out.join("REPORT.json"), &serde_json::to_value(&fit.report).map_err(|e| e.to_string())?)?;
    // The reported artifact: the posterior mean or its rounding, whichever scores better held out,
    // with the literals its evaluation ran with.
    let artifact = fit.artifact(&explanation)?;
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
        "representative": fit.report.representative,
        // The artifact's held-out data term against the samples' (the posterior's expected one).
        "mean_minus_sample_bits_per_token": fit.report.end.mean_bits_per_token - fit.report.end.data_bits_per_token,
        "seconds": started.elapsed().as_secs_f64(),
    });
    log::info!("library summary: {summary}");
    save(&out.join("SUMMARY.json"), &summary)
}

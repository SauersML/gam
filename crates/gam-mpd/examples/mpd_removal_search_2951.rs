//! The removal step of a library fit (`gam_mpd::library_mdl`, #2951) run on one checkpoint with
//! each search of `library_removal`: the previous prefix search (groups in increasing `KL_G`, the
//! longest prefix bisection finds that does not increase `F`) and the ranked search (groups without
//! effect first, then units ranked by their predicted change of `F`, in galloping segments). Both
//! score the fit's own training collection at the fit's weight noise, with least-squares
//! compensation, and are evaluated on the held-out sequences before and after.
//!
//! EXPORT SETTINGS.json CHECKPOINT OUT host|gpu
//!
//! `EXPORT` and `SETTINGS.json` are the fit's (`mpd_library_mdl_2951`, an engine export);
//! `CHECKPOINT` is its `checkpoint.bin` (or a copy). `OUT` receives `REPORT.json`, each search's
//! log (`prefix.removals.jsonl`, `ranked.removals.jsonl`) and `M`'s targets (`targets/`).
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    engine::{log_to_stderr, sha256},
    import::import_language_model,
    library_mdl::{self, Step},
    library_removal::Search,
    operator_program::SlotValues,
    run_check::{layer_nodes, split_sites},
};
use serde::Deserialize;
use serde_json::json;
use std::{path::Path, time::Instant};

const USAGE: &str = "EXPORT SETTINGS.json CHECKPOINT OUT host|gpu";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    export_sha256: String,
    training_sequences: usize,
    context: usize,
    held_out: [usize; 2],
    fit: library_mdl::Settings,
}

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [export, settings_path, checkpoint, out, mode] = &args[..] else {
        return Err(USAGE.into());
    };
    let (export, settings_path, checkpoint, out) = (Path::new(export), Path::new(settings_path), Path::new(checkpoint), Path::new(out));
    let settings: Settings = serde_json::from_slice(&std::fs::read(settings_path).map_err(error)?).map_err(error)?;
    if sha256(&export.join("export.json"))? != settings.export_sha256 {
        return Err("export hash mismatch".into());
    }
    let [first, end] = settings.held_out;
    if first >= end || settings.training_sequences < 2 {
        return Err("held-out sequences must be a nonempty range, and a source needs another training sequence".into());
    }
    let device = match mode.as_str() {
        "host" => Device::host(),
        "gpu" => Device::single_precision(GpuPolicy::Required).map_err(error)?.ok_or("a GPU required")?,
        _ => return Err(USAGE.into()),
    };
    // The fit's rows: the held-out range and the training sequences around it.
    let count = end.max(settings.training_sequences + if settings.training_sequences > first { end - first } else { 0 });
    let imported = import_language_model(export, count, settings.context)?;
    let layer_count = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else {
        return Err("a token slot".into());
    };
    let sequences: Vec<Vec<u32>> = tokens.chunks(settings.context).map(<[u32]>::to_vec).collect();
    let train: Vec<Vec<u32>> = sequences[..first].iter().chain(&sequences[end..]).take(settings.training_sequences).cloned().collect();
    if train.len() != settings.training_sequences {
        return Err("the export holds fewer training sequences than asked for".into());
    }
    let held = &sequences[first..end];
    let native = split_sites(&imported.program)?;
    let layers = layer_nodes(&native, layer_count)?;
    let explanation = library_mdl::explanation(&native, &layers)?;
    library_mdl::check_checkpoint(checkpoint, &library_mdl::identity(&settings.export_sha256, &native, &explanation, &train, held))?;
    let start = library_mdl::checkpoint_posterior(&explanation, checkpoint)?;
    std::fs::create_dir_all(out).map_err(error)?;
    let mut runs = serde_json::Map::new();
    for (name, search) in [("prefix", Search::Prefix), ("ranked", Search::Ranked)] {
        let started = Instant::now();
        let mut posterior = start.clone();
        let log = out.join(format!("{name}.removals.jsonl"));
        let step = Step {
            sequences: &train,
            held,
            settings: &settings.fit,
            search,
            log: Some(&log),
        };
        let (removal, before, after) = library_mdl::removal_step(&device, &native, &explanation, &mut posterior, step)?;
        let run = json!({
            "removal": removal,
            "active_groups": posterior.active.iter().filter(|a| **a).count(),
            "held_out_before": before,
            "held_out_after": after,
            "seconds": started.elapsed().as_secs_f64(),
        });
        log::info!("removal search {name}: {run}");
        runs.insert(name.into(), run);
    }
    let report = json!({
        "export_sha256": settings.export_sha256,
        "settings_sha256": sha256(settings_path)?,
        "checkpoint": checkpoint.display().to_string(),
        "checkpoint_sha256": sha256(checkpoint)?,
        "device": device.name(),
        "source_revision": option_env!("GAM_BUILD_GIT_SHA"),
        "runs": runs,
    });
    std::fs::write(out.join("REPORT.json"), serde_json::to_vec_pretty(&report).map_err(error)?).map_err(error)?;
    println!("{}", serde_json::to_string_pretty(&report).map_err(error)?);
    Ok(())
}

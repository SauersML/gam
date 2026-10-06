//! The removal step of a library fit (`gam_mpd::library_mdl`, #2951) run on one checkpoint: the
//! search of `library_removal` (groups without effect first, then units ranked by their predicted
//! change of `F`, in segments) scores the fit's own training collection at the fit's weight noise,
//! with compensation, and is evaluated on the held-out sequences before and after.
//!
//! EXPORT SETTINGS.json CHECKPOINT OUT host|gpu
//!
//! `EXPORT` and `SETTINGS.json` are the fit's (`mpd_library_mdl_2951`, an engine export);
//! `CHECKPOINT` is its `checkpoint.bin` (or a copy), or `start` for the posterior a fresh fit starts
//! from (`library_mdl::start_posterior`). `OUT` receives `REPORT.json`, the search's
//! log (`ranked.removals.jsonl`) and `M`'s targets (`targets/`). With a sixth argument, group sets
//! (`;` between sets, `,` between groups), each set is removed alone without and with compensation
//! and the changes of the data term and the description go to `OUT/CHANGES.json` instead.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    engine::{log_to_stderr, sha256},
    import::import_language_model,
    library_mdl::{self, Step},
    operator_program::SlotValues,
    run_check::{layer_nodes, split_sites},
};
use serde::Deserialize;
use serde_json::json;
use std::{path::Path, time::Instant};

const USAGE: &str = "EXPORT SETTINGS.json CHECKPOINT OUT host|gpu [G,G;G,…]";

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
    let (export, settings_path, checkpoint, out, mode, sets) = match &args[..] {
        [export, settings, checkpoint, out, mode] => (export, settings, checkpoint, out, mode, None),
        [export, settings, checkpoint, out, mode, sets] => (export, settings, checkpoint, out, mode, Some(sets)),
        _ => return Err(USAGE.into()),
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
    // `start`: the posterior a fresh fit starts from, built here.
    let start = if checkpoint == Path::new("start") {
        library_mdl::start_posterior(&device, &native, &explanation, &train, &settings.fit)?
    } else {
        library_mdl::check_checkpoint(checkpoint, &library_mdl::identity(&settings.export_sha256, &native, &explanation, &train, held))?;
        library_mdl::checkpoint_posterior(&explanation, checkpoint)?
    };
    std::fs::create_dir_all(out).map_err(error)?;
    if let Some(sets) = sets {
        // Each listed group set removed alone, without and with compensation.
        let sets: Vec<Vec<usize>> = sets.split(';').map(|set| set.split(',').map(|g| g.trim().parse::<usize>().map_err(error)).collect()).collect::<Result<_, _>>()?;
        let step = Step { sequences: &train, held, settings: &settings.fit, log: None };
        let changes = library_mdl::removal_changes(&device, &native, &explanation, &start, step, &sets)?;
        let bits = |(data, description): (f64, f64)| json!({"data_bits": data / std::f64::consts::LN_2, "description_bits": description / std::f64::consts::LN_2, "f_bits": (data + description) / std::f64::consts::LN_2});
        let rows: Vec<_> = sets
            .iter()
            .zip(&changes)
            .map(|(set, [plain, compensated])| {
                json!({"groups": set, "names": set.iter().map(|g| explanation.groups[*g].name.clone()).collect::<Vec<_>>(), "plain": bits(*plain), "compensated": bits(*compensated)})
            })
            .collect();
        let report = json!({"checkpoint": checkpoint.display().to_string(), "device": device.name(), "changes": rows});
        std::fs::write(out.join("CHANGES.json"), serde_json::to_vec_pretty(&report).map_err(error)?).map_err(error)?;
        println!("{}", serde_json::to_string_pretty(&report).map_err(error)?);
        return Ok(());
    }
    let mut runs = serde_json::Map::new();
    let name = "ranked";
    {
        let started = Instant::now();
        let mut posterior = start.clone();
        let log = out.join(format!("{name}.removals.jsonl"));
        let step = Step {
            sequences: &train,
            held,
            settings: &settings.fit,
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
        "checkpoint_sha256": if checkpoint.exists() { sha256(checkpoint)? } else { "start".into() },
        "device": device.name(),
        "source_revision": option_env!("GAM_BUILD_GIT_SHA"),
        "runs": runs,
    });
    std::fs::write(out.join("REPORT.json"), serde_json::to_vec_pretty(&report).map_err(error)?).map_err(error)?;
    println!("{}", serde_json::to_string_pretty(&report).map_err(error)?);
    Ok(())
}

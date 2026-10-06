//! Sharing found by a learned mixture prior and decided by the code length
//! (`gam_mpd::library_mixture`, #2951).
//!
//! EXPORT SETTINGS.json FROM OUT host|gpu K
//!
//! SETTINGS.json is the library fit's (`mpd_library_mdl_2951`). FROM is `native` (the library's
//! start at `M`), `artifact:PATH` (a fit's posterior-mean artifact), `checkpoint:PATH` (a fit's
//! checkpoint: its means, standard deviations and removed groups) or `start:PATH`.
//!
//! With `start:PATH` the driver runs one arm of a paired comparison: the fit continues the
//! checkpoint at PATH exactly (its posterior, optimizer state, rotations, removals and epoch,
//! `library_mdl::checkpoint_start`), or with `start:native` makes its own Laplace start at `M`,
//! with the mixture prior of `K` candidates per target for the settings' budget of epochs
//! (`fit.epochs`), on the same batches and weight noise as every other arm of that start. `K = 0` is the control: no target keeps a candidate, so the prior is the
//! groups' Gaussian alone, through the same code. OUT/REPORT.json holds the arm's fit,
//! OUT/PROPOSAL.json what the mixture keeps and the dominant components predicted to lower `F`
//! when made exact, counted by kind, and OUT/SUMMARY.json `F` per epoch and held out.
//!
//! Otherwise: The library is fitted with the
//! mixture prior over its parameter blocks (each MLP function's gate, up and output vectors, each
//! key-value group's query–key maps and value map; at most `K` candidates per target; OUT/soft);
//! every candidate holding more than half of its target's weight whose equality is predicted to
//! lower `F` is made exact (a tie of a row or a column, a shared query–key function, a shared value
//! map; OUT/PROPOSAL.json lists them). The library without and with the exact
//! sharing (OUT/base, OUT/moved) is then fitted to convergence on the same experiments from the
//! soft fit's posterior means, and the sharing is accepted when the moved library's code length `F`
//! is the smaller.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    artifact::Artifact,
    engine::{log_to_stderr, sha256},
    import::import_language_model,
    library_mdl, library_mixture, library_sharing,
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

/// The kind of sharing a component of `target` would make exact: a tie of a read to an earlier
/// write, read or token row, of a write to an earlier write, or a shared query–key function or
/// value map.
fn sharing_kind(target: library_mixture::Kind, write: library_mixture::Write) -> &'static str {
    use library_mixture::{Kind, Write};
    match (target, write) {
        (Kind::Gate { .. }, Write::Output { .. }) => "tie: read of an earlier write",
        (Kind::Gate { .. } | Kind::Up { .. }, Write::Gate { .. } | Write::Up { .. }) => "tie: read of an earlier read",
        (Kind::Gate { .. } | Kind::Up { .. }, Write::Token(_)) => "tie: read of a token's embedding",
        (Kind::Output { .. }, _) => "tie: write of an earlier write",
        (Kind::QueryKey { .. }, _) => "shared query-key function",
        (Kind::Value { .. }, _) => "shared value map",
        (Kind::Gate { .. } | Kind::Up { .. }, _) => "tie: read of an earlier write",
    }
}

/// One arm of a paired comparison from a checkpoint (module note).
fn arm(device: &Device, (native, start): (&gam_mpd::operator_program::OperatorProgram, &library_mdl::Explanation), (train, held_out): (&[Vec<u32>], &[Vec<u32>]), settings: &Settings, (checkpoint, out, width): (&Path, &Path, usize)) -> Result<(), String> {
    let fit = &settings.fit;
    if fit.epochs.is_none() {
        return Err("a paired arm needs the settings' budget of epochs (fit.epochs)".into());
    }
    let begin = if checkpoint == Path::new("native") { None } else { Some(library_mdl::checkpoint_start(start, checkpoint)?) };
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let steps = library_mixture::Steps { rate: 0.05, beta1: fit.beta1, beta2: 0.999, epsilon: 1e-8 };
    let mut mixture = library_mixture::Mixture::new(start, width, steps)?;
    let fitted = library_mdl::fit_from(device, native, start, train, held_out, fit, &settings.export_sha256, Some(&out.join("checkpoint.bin")), Some(&mut mixture), begin)?;
    save(&out.join("REPORT.json"), &serde_json::to_value(&fitted.report).map_err(|e| e.to_string())?)?;
    let mut kept: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for target in &mixture.targets {
        for component in &target.components {
            *kept.entry(sharing_kind(target.kind, component.write)).or_default() += 1;
        }
    }
    let proposals = mixture.proposals(&fitted.posterior)?;
    let mut exact: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    let listed: Vec<Value> = proposals
        .iter()
        .map(|(t, j, saving)| {
            let target = &mixture.targets[*t];
            *exact.entry(sharing_kind(target.kind, target.components[*j].write)).or_default() += 1;
            json!({"target": target.kind, "candidate": target.components[*j].write, "scale": target.components[*j].scale, "weights": target.weights().unwrap_or_default(), "choices": target.choices, "predicted_saving_bits": saving / std::f64::consts::LN_2})
        })
        .collect();
    save(&out.join("PROPOSAL.json"), &json!({"targets": mixture.targets.len(), "kept_by_kind": kept, "exact_by_kind": exact, "exact": listed}))?;
    let epochs: Vec<Value> = fitted.report.epochs.iter().map(|e| json!({"epoch": e.epoch, "objective_bits": e.objective_bits, "data_bits": e.data_bits, "description_bits": e.description_bits, "held_out_objective_bits_per_token": e.held_out.objective_bits_per_token})).collect();
    let summary = json!({
        "export_sha256": settings.export_sha256,
        "from": format!("start:{}", checkpoint.display()),
        "candidates": width,
        "objective_bits": fitted.report.objective_bits,
        "start": &fitted.report.start,
        "end": &fitted.report.end,
        "epochs": epochs,
        "kept_by_kind": kept,
        "exact_by_kind": exact,
    });
    log::info!("sharing arm summary: {summary}");
    save(&out.join("SUMMARY.json"), &summary)
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [export, settings_path, from, out, mode, width] = &args[..] else {
        return Err("EXPORT SETTINGS.json native|artifact:PATH|checkpoint:PATH OUT host|gpu K".into());
    };
    let width: usize = width.parse().map_err(|e| format!("K: {e}"))?;
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
    if let Some(path) = from.strip_prefix("start:") {
        return arm(&device, (&native, &start), (&train, held_out), &settings, (Path::new(path), out, width));
    }
    let base = match from.split_once(':') {
        None if from == "native" => start,
        Some(("artifact", path)) => library_sharing::warm(&start, &Artifact::from_bytes(&std::fs::read(path).map_err(|e| e.to_string())?, &native.declarations)?)?,
        Some(("checkpoint", path)) => {
            let posterior = library_mdl::checkpoint_posterior(&start, Path::new(path))?;
            let mut base = library_sharing::warm(&start, &library_mdl::posterior_mean(&start, &posterior)?)?;
            base.removed = (0..posterior.active.len()).filter(|g| !posterior.active[*g]).collect();
            base
        }
        _ => return Err("FROM is native, artifact:PATH or checkpoint:PATH".into()),
    };
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let fit = &settings.fit;
    let steps = library_mixture::Steps { rate: 0.05, beta1: fit.beta1, beta2: 0.999, epsilon: 1e-8 };
    let mut mixture = library_mixture::Mixture::new(&base, width, steps)?;
    let dir = out.join("soft");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let soft = library_mdl::fit(&device, &native, &base, &train, held_out, fit, &settings.export_sha256, Some(&dir.join("checkpoint.bin")), Some(&mut mixture))?;
    save(&dir.join("REPORT.json"), &serde_json::to_value(&soft.report).map_err(|e| e.to_string())?)?;
    // Both libraries start from the soft fit's posterior means and removals.
    let mut base = library_sharing::warm(&base, &library_mdl::posterior_mean(&base, &soft.posterior)?)?;
    base.removed = (0..soft.posterior.active.len()).filter(|g| !soft.posterior.active[*g]).collect();
    let proposals = mixture.proposals(&soft.posterior)?;
    let kept = mixture.targets.iter().filter(|t| !t.components.is_empty()).count();
    log::info!("{kept} targets keep a mixture; {} dominant candidates are predicted to lower F when made exact", proposals.len());
    let listed: Vec<Value> = proposals
        .iter()
        .map(|(t, j, saving)| {
            let target = &mixture.targets[*t];
            json!({"target": target.kind, "candidate": target.components[*j].write, "scale": target.components[*j].scale, "weights": target.weights().unwrap_or_default(), "choices": target.choices, "predicted_saving_bits": saving / std::f64::consts::LN_2})
        })
        .collect();
    let moved = mixture.harden(&base, &soft.posterior)?;
    let proposal = json!({"targets": mixture.targets.len(), "kept": kept, "exact": listed, "choice_bits": moved.fixed_nats / std::f64::consts::LN_2});
    moved.artifact.validate_coverage(&native)?;
    save(&out.join("PROPOSAL.json"), &proposal)?;
    let mut fits = Vec::new();
    for (name, explanation) in [("base", &base), ("moved", &moved)] {
        let dir = out.join(name);
        let checkpoint = dir.join("checkpoint.bin");
        library_mdl::check_checkpoint(&checkpoint, &library_mdl::identity(&settings.export_sha256, &native, explanation, &train, held_out))?;
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let fit = library_mdl::fit(&device, &native, explanation, &train, held_out, &settings.fit, &settings.export_sha256, Some(&checkpoint), None)?;
        save(&dir.join("REPORT.json"), &serde_json::to_value(&fit.report).map_err(|e| e.to_string())?)?;
        fits.push(fit.report);
    }
    let summary = json!({
        "export_sha256": settings.export_sha256,
        "settings_sha256": sha256(settings_path)?,
        "from": from,
        "candidates": width,
        "base_objective_bits": fits[0].objective_bits,
        "moved_objective_bits": fits[1].objective_bits,
        "accepted": fits[1].objective_bits < fits[0].objective_bits,
        "base_held_out": &fits[0].end,
        "moved_held_out": &fits[1].end,
    });
    log::info!("sharing move summary: {summary}");
    save(&out.join("SUMMARY.json"), &summary)
}

//! Whether the least-squares compensation of a removal (`library_compensation`) lowers the code
//! length the removal step reaches (#2951): the removal step run twice on one posterior, removing
//! prefixes plainly and with compensation.
//!
//! EXPORT SETTINGS.json checkpoint:PATH OUT host|gpu
//!
//! SETTINGS.json is the library fit's (`mpd_library_mdl_2951`). Both runs order the checkpoint's
//! active groups by increasing divergence and search prefix lengths by the fit's bisection,
//! accepting the longest evaluated prefix that does not increase `F`. Every evaluation scores the
//! same experiments (per training batch of the fit's size, one draw of `interchange::sample` over
//! `M`'s functions) at the same weight noise per batch (`ε` of `gam_gpu`'s Philox normals,
//! zero where a group is removed): `F` is the summed divergence over every training experiment's
//! scored tokens plus the posterior's description and the explanation's fixed choices, in bits.
use gam_gpu::{
    GpuPolicy,
    tensor::{Device, posterior_normal},
};
use gam_mpd::{
    engine::{log_to_stderr, sha256},
    import::import_language_model,
    interchange::{self, Batch, Experiment, Interchange, Targets},
    library_compensation::Compensation,
    library_mdl::{self, Posterior},
    operator_program::SlotValues,
    run_check::{layer_nodes, split_sites},
};
use ndarray::Array2;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{f64::consts::LN_2, path::Path};

const USAGE: &str = "EXPORT SETTINGS.json checkpoint:PATH OUT host|gpu";

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

/// One training batch's experiments and `M`'s targets for them.
struct Evidence {
    batch: Batch,
    experiments: Vec<Experiment>,
    targets: Targets,
}

/// `F` in bits of `posterior` on `evidence`, batch `b` at the weight sample of noise key `b`.
fn objective(ic: &mut Interchange, evidence: &[Evidence], posterior: &Posterior, fixed: f64) -> Result<f64, String> {
    let mut bits = 0.0;
    for (b, e) in evidence.iter().enumerate() {
        let theta: Vec<Array2<f64>> = posterior
            .mean
            .iter()
            .zip(&posterior.log_sd)
            .enumerate()
            .map(|(i, (mean, log_sd))| {
                let cols = mean.ncols();
                Array2::from_shape_fn(mean.dim(), |(r, c)| {
                    let s = log_sd[[r, c]];
                    if s == f64::NEG_INFINITY { mean[[r, c]] } else { mean[[r, c]] + s.exp() * f64::from(posterior_normal(b as u64, i as u64, (r * cols + c) as u64)) }
                })
            })
            .collect();
        ic.load(&theta)?;
        let evaluation = ic.evaluate_resident(&e.batch, &e.experiments, &e.targets, false)?;
        bits += evaluation.bits.iter().flatten().sum::<f64>();
    }
    Ok(bits + (posterior.description() + fixed) / LN_2)
}

/// The removal step on `posterior` with `remove` making each prefix's posterior: the prefix lengths
/// bisection evaluates, each with its change of `F` in bits, and the length accepted.
fn search(
    ic: &mut Interchange,
    evidence: &[Evidence],
    posterior: &Posterior,
    order: &[usize],
    fixed: f64,
    remove: &dyn Fn(&[usize]) -> Result<Posterior, String>,
) -> Result<(Vec<(usize, f64)>, usize), String> {
    let base = objective(ic, evidence, posterior, fixed)?;
    let mut evaluations = Vec::new();
    let mut accepted = |k: usize, evaluations: &mut Vec<(usize, f64)>| -> Result<bool, String> {
        let change = objective(ic, evidence, &remove(&order[..k])?, fixed)? - base;
        log::info!("removal of {k} of {} groups: F changes by {change:.6e} bits", order.len());
        evaluations.push((k, change));
        Ok(change <= 0.0)
    };
    if accepted(order.len(), &mut evaluations)? {
        return Ok((evaluations, order.len()));
    }
    let (mut low, mut high) = (0, order.len());
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if accepted(middle, &mut evaluations)? {
            low = middle;
        } else {
            high = middle;
        }
    }
    Ok((evaluations, low))
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [export, settings_path, from, out, mode] = &args[..] else {
        return Err(USAGE.into());
    };
    let (export, settings_path, out) = (Path::new(export), Path::new(settings_path), Path::new(out));
    let settings: Settings = serde_json::from_slice(&std::fs::read(settings_path).map_err(error)?).map_err(error)?;
    if sha256(&export.join("export.json"))? != settings.export_sha256 {
        return Err("export hash mismatch".into());
    }
    let [first, end] = settings.held_out;
    if first >= end || settings.training_sequences < 2 {
        return Err("held-out sequences must be a nonempty range, and a source needs another training sequence".into());
    }
    let Some(("checkpoint", checkpoint)) = from.split_once(':') else {
        return Err(USAGE.into());
    };
    let device = match mode.as_str() {
        "host" => Device::host(),
        "gpu" => Device::single_precision(GpuPolicy::Required).map_err(error)?.ok_or("a GPU required")?,
        _ => return Err(USAGE.into()),
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
    let train: Vec<Vec<u32>> = sequences[..first].iter().chain(&sequences[end..]).take(settings.training_sequences).cloned().collect();
    if train.len() != settings.training_sequences {
        return Err("the export holds fewer training sequences than asked for".into());
    }
    let explanation = library_mdl::explanation(&native, &layers)?;
    // Entry by entry along the operators' own axes, as the objective below samples it.
    let posterior = library_mdl::checkpoint_posterior(&explanation, Path::new(checkpoint))?;
    let sites: Vec<_> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
    let reads = interchange::reads(&native, &sites)?;
    let mut ic = Interchange::new(&device, &native, &sites, &explanation.artifact, &explanation.trainable, reads, settings.fit.numeric_bytes, settings.fit.head_tile_rows)?;
    let variables = ic.variables().to_vec();
    // The fixed collection: the training sequences in order, in batches, each base's source drawn
    // among the other training sequences.
    let mut rng = StdRng::seed_from_u64(settings.fit.seed);
    let indices: Vec<usize> = (0..train.len()).collect();
    let mut evidence = Vec::new();
    for bases in indices.chunks(settings.fit.batch_sequences) {
        let sources: Vec<usize> = bases
            .iter()
            .map(|&i| {
                let j = rng.random_range(0..train.len() - 1);
                if j >= i { j + 1 } else { j }
            })
            .collect();
        let pick = |chosen: &[usize]| chosen.iter().map(|i| train[*i].clone()).collect::<Vec<_>>();
        let batch = Batch::new(pick(bases), pick(&sources))?;
        let experiments = interchange::sample(&mut rng, bases.len(), &variables, 2 * layer_count, settings.context)?;
        let targets = ic.targets(&batch, &experiments)?;
        evidence.push(Evidence { batch, experiments, targets });
    }
    let divergences = posterior.divergences();
    let mut order: Vec<usize> = (0..divergences.len()).filter(|g| posterior.active[*g]).collect();
    order.sort_by(|a, b| divergences[*a].total_cmp(&divergences[*b]));
    let fixed = explanation.fixed_nats;
    let compensation = Compensation::new(&mut ic, &explanation, &posterior, &train, settings.fit.batch_sequences)?;
    let plain = |prefix: &[usize]| -> Result<Posterior, String> {
        let mut trial = posterior.clone();
        trial.remove(prefix);
        Ok(trial)
    };
    let compensated = |prefix: &[usize]| compensation.proposal(&posterior, prefix);
    let base = objective(&mut ic, &evidence, &posterior, fixed)?;
    let mut runs = serde_json::Map::new();
    for (name, remove) in [("plain", &plain as &dyn Fn(&[usize]) -> Result<Posterior, String>), ("compensated", &compensated as &dyn Fn(&[usize]) -> Result<Posterior, String>)] {
        let (evaluations, accepted) = search(&mut ic, &evidence, &posterior, &order, fixed, remove)?;
        let after = evaluations.iter().find(|(k, _)| *k == accepted).map_or(0.0, |(_, change)| *change);
        runs.insert(
            name.into(),
            json!({
                "removed": accepted,
                "objective_change_bits": after,
                "evaluations": evaluations.iter().map(|(k, change)| json!({"prefix": k, "change_bits": change})).collect::<Vec<Value>>(),
            }),
        );
    }
    let report = json!({
        "export_sha256": settings.export_sha256,
        "settings_sha256": sha256(settings_path)?,
        "checkpoint": checkpoint,
        "device": device.name(),
        "active_groups": order.len(),
        "objective_bits": base,
        "runs": runs,
    });
    std::fs::create_dir_all(out).map_err(error)?;
    std::fs::write(out.join("REPORT.json"), serde_json::to_vec_pretty(&report).map_err(error)?).map_err(error)?;
    println!("{}", serde_json::to_string_pretty(&report).map_err(error)?);
    Ok(())
}

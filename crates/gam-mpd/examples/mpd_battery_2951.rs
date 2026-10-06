//! The common evaluation battery (#2951, `gam_mpd::explanation_battery`) on held-out token rows of
//! an export: a library explanation `P` of a language model `M`, or VPD's decomposition of it.
//!
//! EXPORT SETTINGS.json OUT.json host|gpu library [ARTIFACT]
//! EXPORT SETTINGS.json OUT.json host|gpu vpd DECOMPOSITION
//! EXPORT SETTINGS.json OUT.json host|gpu price DECOMPOSITION [START]
//! EXPORT SETTINGS.json OUT.json host|gpu price_charged DECOMPOSITION [START]
//! EXPORT SETTINGS.json OUT.json host|gpu masks DECOMPOSITION
//!
//! `price_charged` prices VPD's causal-importance network beside its subcomponents
//! (`vpd_pricing` with `charge`).
//!
//! `masks` measures where VPD's masks come from (`explanation_battery::vpd_mask_sources`): held-out
//! KL with masks from `M`'s activations, from them through a causal network, with every mask 1,
//! and from VPD's own activations by three fixed-point rounds.
//!
//! `price` prices VPD's decomposition in the library's code length
//! (`explanation_battery::vpd_pricing`) on the settings' `training_sequences` and writes the
//! converged posterior to OUT with the extension `posterior.f32`; `START`, a posterior written so
//! (at any number of training tokens), starts the fit.
//!
//! `ARTIFACT` is a `library_mdl` posterior-mean `artifact.bin`, or a fit's `checkpoint.bin`, whose
//! posterior mean is scored (with its `KL(q ‖ p)` and description reported). Without it `P` is the
//! library's starting point, which computes `M` exactly: every divergence must vanish to rounding.
//! `DECOMPOSITION` is VPD's exported decomposition (`bench/vpd_2951/vpd_export.py`), scored by
//! `explanation_battery::{vpd_protocols, vpd_cancellation, vpd_interchange_atomic, vpd_interchange}`
//! (the interchange families at one position as asked of a library explanation, then at every
//! position). An empty source range
//! skips the interchange experiments.
//!
//! * Behaviour and VPD's protocols (`explanation_battery::protocols`), from the logits: every
//!   nonempty layer subset run as the explanation's (all layers is error-propagating, one layer
//!   single-layer, a prefix a cut), and clean-input.
//! * Interchange (`interchange`) with `P` alone: per base one read patch of one of `M`'s functions
//!   drawn uniformly (`read`), and a joint read patch of a random subset of the functions at that
//!   function's block, its size uniform (`read_joint`): each model's values of the patched functions
//!   replaced by its own on the source (`P`'s at the call sites that replaced them), a function `P`
//!   no longer computes patched in `M` alone. Each with the source a sequence shared across the
//!   whole batch of bases. Each base's patches replace one position drawn uniformly and are scored
//!   from that position on (earlier tokens are the unpatched run's). Every source row is scored;
//!   the worst source of the first `K` (by the mean over all bases) is reported for each `K`.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    artifact::Artifact,
    engine::{log_to_stderr, sha256},
    explanation_battery::{self as battery, Behaviour, Decomposition, Reference, Side, Tokens, Vpd},
    import::import_language_model,
    interchange::{self, Batch, Experiment, Interchange, Patch},
    library_mdl,
    operator_program::SlotValues,
    run_check::{layer_nodes, split_sites},
};
use ndarray::Array2;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, f64::consts::LN_2, path::Path, time::Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    export_sha256: String,
    context: usize,
    /// The held-out base rows `[start, end)`, and the rows `[start, end)` that serve as shared
    /// patch sources.
    held_out: [usize; 2],
    sources: [usize; 2],
    /// Bases per forward batch, the operator bytes each program may hold on the device, rows of
    /// vocabulary logits per tile, and the seed of the drawn patches and subsets.
    batch_sequences: usize,
    numeric_bytes: usize,
    head_tile_rows: usize,
    seed: u64,
    /// The numbers of shared sources of which the worst is reported.
    worst_of: Vec<usize>,
    /// For VPD, the check against its reported evaluation: on the rows `[start, end)`, its masks'
    /// cross-entropies and active subcomponents, and its adversary's KL after `pgd_steps` steps of
    /// `pgd_step_size` (its evaluation's 20 and 0.1).
    #[serde(default)]
    vpd_check: Option<VpdCheck>,
    /// For pricing VPD in the library's code length: the training sequences, the first of the
    /// export's rows outside the held-out ones (as `mpd_library_mdl_2951` takes them).
    #[serde(default)]
    training_sequences: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VpdCheck {
    rows: [usize; 2],
    pgd_steps: usize,
    pgd_step_size: f64,
}

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// The posterior mean of a `library_mdl` checkpoint (a little-endian header length, the progress
/// header, then per trainable operator `μ`, `ln σ` and the four Adam moments in float64) as the
/// explanation's artifact, with the posterior's `KL(q ‖ p)` and description in bits and the fit's
/// last held-out evaluation.
fn checkpoint_mean(path: &Path, explanation: &library_mdl::Explanation) -> Result<(Artifact, Value), String> {
    let posterior = library_mdl::checkpoint_posterior(explanation, path)?;
    // The fit's progress, which the fit writes beside the checkpoint at every save.
    let header: Value = serde_json::from_slice(&std::fs::read(path.with_extension("json")).map_err(error)?).map_err(error)?;
    let tokens = header["tokens"].as_u64().ok_or("checkpoint tokens")? as usize;
    let divergence: f64 = posterior.divergences().iter().sum();
    let size = json!({
        "epoch": header["epoch"],
        "training_tokens": tokens,
        "active_groups": posterior.active.iter().filter(|a| **a).count(),
        "divergence_bits": divergence / LN_2,
        "description_bits": posterior.description() / LN_2,
        "held_out": header["epochs"].as_array().and_then(|e| e.last()).map(|e| e["held_out"].clone()),
    });
    Ok((library_mdl::posterior_mean(explanation, &posterior)?.f32_literals()?, size))
}

/// The circuit curves (`explanation_battery::circuit_curve`) on each task of `pairs`
/// (`bench/vpd_2951/sva_export.py`) of `M`'s MLP neurons and, with `decomposition`, of VPD's
/// subcomponents (all sites, and the MLP sites alone), and their means over the tasks.
fn circuits(device: &Device, export: &Path, layers: &[gam_mpd::run_check::LayerNodes], pairs: &Path, decomposition: Option<&Path>, settings: &Settings) -> Result<Value, String> {
    #[derive(Deserialize)]
    struct Task {
        train: Vec<battery::Pair>,
        test: Vec<battery::Pair>,
    }
    let record: Value = serde_json::from_slice(&std::fs::read(pairs).map_err(error)?).map_err(error)?;
    let tasks: BTreeMap<String, Task> = serde_json::from_value(record["tasks"].clone()).map_err(error)?;
    let (model, unembedding) = battery::model(export, None)?;
    let m = Side::of_model(device, &model, &unembedding, settings.numeric_bytes)?;
    let none = |_: usize| -> Result<BTreeMap<usize, gam_gpu::tensor::Tensor>, String> { Ok(BTreeMap::new()) };
    let count = layers.len();
    let down = |l: usize| model.layout.inputs[battery::KINDS.len() * l + 5];
    let neurons = battery::NodeBasis { side: &m, groups: (0..count).map(down).collect(), given: &none, unembedding: &unembedding };
    // VPD's subcomponents, as nodes of its program with every mask and remainder at one (`M`):
    // at all 24 sites, and at the MLP sites alone (the coverage of `M`'s neurons).
    let vpd = decomposition
        .map(|dir| -> Result<_, String> {
            let factors = battery::load_factors(dir)?;
            let (model, _) = battery::model(export, Some(&factors))?;
            let side = Side::of_model(device, &model, &unembedding, settings.numeric_bytes)?;
            let widths: Vec<(usize, usize)> = model
                .layout
                .masks
                .iter()
                .zip(&model.layout.deltas)
                .map(|(m, r)| match (&model.program.declarations.slots[*m], &model.program.declarations.slots[*r]) {
                    (gam_mpd::operator_program::Slot::Raw { width: c }, gam_mpd::operator_program::Slot::Raw { width }) => Ok((*c, *width)),
                    _ => Err("a mask slot is not raw".to_string()),
                })
                .collect::<Result<_, _>>()?;
            Ok((factors, model.layout, side, widths))
        })
        .transpose()?;
    let ones = |rows: usize| -> Result<BTreeMap<usize, gam_gpu::tensor::Tensor>, String> {
        let mut out = BTreeMap::new();
        if let Some((_, layout, _, widths)) = &vpd {
            for (s, (c, width)) in widths.iter().enumerate() {
                out.insert(layout.masks[s], device.upload(Array2::<f64>::ones((rows, *c)).view()).map_err(error)?);
                out.insert(layout.deltas[s], device.upload(Array2::<f64>::ones((rows, *width)).view()).map_err(error)?);
            }
        }
        Ok(out)
    };
    let mlp_sites: Vec<usize> = (0..count * battery::KINDS.len()).filter(|s| battery::KINDS[s % battery::KINDS.len()].block() == 1).collect();
    let mut out = serde_json::Map::new();
    let mut curves: BTreeMap<&str, Vec<Value>> = BTreeMap::new();
    for (name, task) in &tasks {
        let mut entry = json!({"train": task.train.len(), "test": task.test.len()});
        // Nodes ranked by their measured patching effects on the training pairs, as many prompt
        // copies per pass as fit in the head's tile rows.
        let attribution = neurons.patch_effects(&task.train, settings.head_tile_rows)?;
        entry["neurons"] = battery::circuit_curve(&neurons, &attribution, &task.train, &task.test)?;
        if let Some((_, layout, side, _)) = &vpd {
            let all = battery::NodeBasis { side, groups: layout.activations.clone(), given: &ones, unembedding: &unembedding };
            let attribution = all.patch_effects(&task.train, settings.head_tile_rows)?;
            entry["vpd"] = battery::circuit_curve(&all, &attribution, &task.train, &task.test)?;
            let mlp = battery::NodeBasis { side, groups: mlp_sites.iter().map(|s| layout.activations[*s]).collect(), given: &ones, unembedding: &unembedding };
            let mlp_attribution: Vec<_> = mlp_sites.iter().map(|s| attribution[*s].clone()).collect();
            entry["vpd_mlp"] = battery::circuit_curve(&mlp, &mlp_attribution, &task.train, &task.test)?;
        }
        for basis in ["neurons", "vpd", "vpd_mlp"] {
            if !entry[basis].is_null() {
                curves.entry(basis).or_default().push(entry[basis].clone());
            }
        }
        log::info!("circuits: {name} done");
        out.insert(name.clone(), entry);
    }
    let mut mean = serde_json::Map::new();
    for (basis, list) in &curves {
        let average = |key: &str| -> Vec<f64> {
            let n = list[0][key].as_array().map_or(0, Vec::len);
            (0..n).map(|i| list.iter().filter_map(|c| c[key][i].as_f64()).sum::<f64>() / list.len() as f64).collect()
        };
        mean.insert(basis.to_string(), json!({"k": list[0]["k"], "faithfulness": average("faithfulness"), "completeness": average("completeness")}));
    }
    out.insert("mean".into(), Value::Object(mean));
    Ok(Value::Object(out))
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "EXPORT SETTINGS.json OUT.json host|gpu library [ARTIFACT] | vpd DECOMPOSITION | circuits PAIRS.json [DECOMPOSITION] | price|price_charged DECOMPOSITION [START] | masks DECOMPOSITION";
    let (export, settings_path, out, mode, kind, extra, more) = match &args[..] {
        [e, s, o, m, k] => (e, s, o, m, k.as_str(), None, None),
        [e, s, o, m, k, a] => (e, s, o, m, k.as_str(), Some(Path::new(a)), None),
        [e, s, o, m, k, a, b] => (e, s, o, m, k.as_str(), Some(Path::new(a)), Some(Path::new(b))),
        _ => return Err(usage.into()),
    };
    let (export, settings_path) = (Path::new(export), Path::new(settings_path));
    let settings: Settings = serde_json::from_slice(&std::fs::read(settings_path).map_err(error)?).map_err(error)?;
    if sha256(&export.join("export.json"))? != settings.export_sha256 {
        return Err("export hash mismatch".into());
    }
    let ([first, end], [s_first, s_end]) = (settings.held_out, settings.sources);
    if first >= end || s_first > s_end || settings.batch_sequences == 0 || settings.worst_of.iter().any(|k| *k == 0 || *k > s_end - s_first) {
        return Err("nonempty base and source ranges, positive batches, and at most as many worst-of sources as source rows".into());
    }
    let device = match mode.as_str() {
        "host" => Device::host(),
        "gpu" => Device::single_precision(GpuPolicy::Required).map_err(error)?.ok_or("a GPU required")?,
        _ => return Err("host|gpu required".into()),
    };
    let started = Instant::now();
    let check_end = settings.vpd_check.as_ref().map_or(0, |c| c.rows[1]);
    let training = settings.training_sequences.unwrap_or(0);
    let training_end = if training > first { training + (end - first) } else { training };
    let imported = import_language_model(export, end.max(s_end).max(check_end).max(training_end), settings.context)?;
    let layer_count = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
    let native = split_sites(&imported.program)?;
    let layers = layer_nodes(&native, layer_count)?;
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else {
        return Err("a token slot".into());
    };
    let all_rows: Vec<Vec<u32>> = tokens.chunks(settings.context).map(<[u32]>::to_vec).collect();
    let bases = &all_rows[first..end];
    let sources = &all_rows[s_first..s_end];
    let length = settings.context;
    let mut report = json!({
        "export": export.display().to_string(),
        "export_sha256": settings.export_sha256,
        "settings_sha256": sha256(settings_path)?,
        "explanation": kind,
        "artifact": extra.map(|p| p.display().to_string()),
        "source_revision": option_env!("GAM_BUILD_GIT_SHA"),
        "device": device.name(),
        "held_out_rows": [first, end],
        "source_rows": [s_first, s_end],
    });
    let save = |report: &Value| -> Result<(), String> { std::fs::write(out, serde_json::to_vec_pretty(report).map_err(error)?).map_err(error) };
    if kind == "vpd" {
        let vpd = Vpd::new(&device, export, Decomposition::load(extra.ok_or(usage)?)?, settings.numeric_bytes)?;
        report["protocols"] = battery::vpd_protocols(&vpd, bases, settings.batch_sequences, settings.seed, true)?;
        save(&report)?;
        if let Some(check) = &settings.vpd_check {
            let rows = &all_rows[check.rows[0]..check.rows[1]];
            report["vpd_check"] = json!({"rows": check.rows, "masks": battery::vpd_protocols(&vpd, rows, settings.batch_sequences, settings.seed, false)?});
            save(&report)?;
            report["vpd_check"]["adversary"] = battery::vpd_pgd(&vpd, rows, settings.batch_sequences, check.pgd_steps, check.pgd_step_size, settings.seed)?;
            save(&report)?;
        }
        report["cancellation"] = battery::vpd_cancellation(&vpd, bases, settings.batch_sequences, settings.seed)?;
        save(&report)?;
        if s_first < s_end {
            report["interchange_one_position"] = battery::vpd_interchange_atomic(&vpd, bases, sources, settings.batch_sequences, settings.seed, &settings.worst_of)?;
            save(&report)?;
            report["interchange"] = battery::vpd_interchange(&vpd, bases, sources, settings.batch_sequences, settings.seed, &settings.worst_of)?;
        }
        report["seconds"] = json!(started.elapsed().as_secs_f64());
        save(&report)?;
        log::info!("battery done in {:.0} s: {out}", started.elapsed().as_secs_f64());
        return Ok(());
    }
    if kind == "masks" {
        let decomposition = extra.ok_or(usage)?;
        let vpd = Vpd::new(&device, export, Decomposition::load(decomposition)?, settings.numeric_bytes)?;
        report["mask_sources"] = battery::vpd_mask_sources(&vpd, export, decomposition, bases, settings.batch_sequences, 3, settings.numeric_bytes)?;
        report["seconds"] = json!(started.elapsed().as_secs_f64());
        save(&report)?;
        log::info!("battery done in {:.0} s: {out}", started.elapsed().as_secs_f64());
        return Ok(());
    }
    if kind == "price" || kind == "price_charged" {
        // `price_charged` charges VPD's causal-importance network too.
        let charge = (kind == "price_charged").then(|| extra.ok_or(usage)).transpose()?;
        let vpd = Vpd::new(&device, export, Decomposition::load(extra.ok_or(usage)?)?, settings.numeric_bytes)?;
        let train: Vec<Vec<u32>> = all_rows[..first].iter().chain(&all_rows[end..]).take(training).cloned().collect();
        if training == 0 || train.len() != training {
            return Err("pricing needs training_sequences rows outside the held-out ones".into());
        }
        let mut progress = report.clone();
        // The converged posterior beside OUT; START, a posterior written so, starts the fit.
        let posterior = Path::new(out).with_extension("posterior.f32");
        report["pricing"] = battery::vpd_pricing(&vpd, export, &train, bases, settings.batch_sequences, settings.seed, (more, charge), &posterior, |state| {
            progress["pricing"] = state.clone();
            save(&progress)
        })?;
        report["seconds"] = json!(started.elapsed().as_secs_f64());
        save(&report)?;
        log::info!("battery done in {:.0} s: {out}", started.elapsed().as_secs_f64());
        return Ok(());
    }
    if kind == "circuits" {
        report["circuits"] = circuits(&device, export, &layers, extra.ok_or(usage)?, more, &settings)?;
        report["seconds"] = json!(started.elapsed().as_secs_f64());
        save(&report)?;
        log::info!("battery done in {:.0} s: {out}", started.elapsed().as_secs_f64());
        return Ok(());
    }
    if kind != "library" || more.is_some() {
        return Err(usage.into());
    }
    let artifact_path = extra;
    let explanation = library_mdl::explanation(&native, &layers)?;
    let (artifact, size) = match artifact_path {
        Some(path) if path.extension().is_some_and(|e| e == "bin") && path.file_name().is_some_and(|n| n == "checkpoint.bin") => checkpoint_mean(path, &explanation)?,
        Some(path) => (Artifact::from_bytes(&std::fs::read(path).map_err(error)?, &native.declarations)?, Value::Null),
        None => (explanation.artifact.clone(), Value::Null),
    };
    artifact.validate_coverage(&native)?;
    let mut rng = StdRng::seed_from_u64(settings.seed);

    // Behaviour and the protocols.
    let (m, p) = (Side::of_artifact(&device, &Artifact::native(&native)?, &layers, settings.numeric_bytes)?, Side::of_artifact(&device, &artifact, &layers, settings.numeric_bytes)?);
    let d = device.clone();
    let mut protocols: BTreeMap<String, Behaviour> = BTreeMap::new();
    let mut m_ce = Vec::new();
    for chunk in bases.chunks(settings.batch_sequences) {
        let views: Vec<&[u32]> = chunk.iter().map(Vec::as_slice).collect();
        let family = library_mdl::sequence_family(&views)?;
        let (embedding, m_streams) = battery::streams(&m, &family, |_| BTreeMap::new())?;
        let reference = Reference::of(m.logits(&family, &m_streams[layer_count], length)?)?;
        m_ce.extend(reference.cross_entropy(&views));
        let finals = battery::protocols(&d, &m_streams, &embedding, |l, explained, entry| {
            if explained { p.layer(&family, l, entry, BTreeMap::new()) } else { m.layer(&family, l, entry, BTreeMap::new()) }
        })?;
        for (name, x) in finals {
            let e = m.logits(&family, &x, length)?;
            protocols.entry(name).or_default().add(&reference, &e, &views)?;
        }
        log::info!("battery: protocols on {} bases ({:.0} s)", chunk.len(), started.elapsed().as_secs_f64());
    }
    drop((m, p));
    report["artifact_sha256"] = json!(artifact_path.map(sha256).transpose()?);
    report["ce_target"] = json!(m_ce.iter().sum::<f64>() / m_ce.len() as f64);
    report["protocols"] = battery::protocol_summary(&protocols, layer_count);
    report["size"] = size;
    save(&report)?;
    if s_first == s_end {
        return Ok(());
    }

    // Interchange with P alone, sources shared across the batch, over M's functions.
    let variables = interchange::reads(&native, &layers)?;
    let blocks = 2 * layer_count;
    let interchange = Interchange::new(&device, &native, &layers, &artifact, &explanation.trainable, variables.clone(), settings.numeric_bytes, settings.head_tile_rows)?;
    let read_of: Vec<usize> = (0..bases.len()).map(|_| rng.random_range(0..variables.len())).collect();
    let position_of: Vec<usize> = (0..bases.len()).map(|_| rng.random_range(0..bases[0].len())).collect();
    let subset_of: Vec<Vec<usize>> = read_of
        .iter()
        .map(|&v| {
            let at: Vec<usize> = (0..variables.len()).filter(|i| variables[*i].block == variables[v].block).collect();
            interchange::subset(&mut rng, &at)
        })
        .collect();
    // Per family (the read patch, then the joint read patch) per source: bits and tokens over
    // every base, and every token's bits.
    let families = 2;
    let mut per_source = vec![vec![(0.0f64, 0usize); sources.len()]; families];
    let mut all: Vec<Tokens> = (0..families).map(|_| Tokens::default()).collect();
    let mut clean = Tokens::default();
    for (s, source) in sources.iter().enumerate() {
        for (c, chunk) in bases.chunks(settings.batch_sequences).enumerate() {
            let batch = Batch::new(chunk.to_vec(), vec![source.clone()])?;
            let mut experiments = Vec::new();
            for b in 0..chunk.len() {
                let position = position_of[c * settings.batch_sequences + b];
                let at = |patch: Option<Patch>| {
                    let position = if patch.is_some() { position } else { 0 };
                    Experiment { base: b, source: 0, explained: vec![true; blocks], patch, position }
                };
                experiments.push(at(Some(Patch::Read { variable: read_of[c * settings.batch_sequences + b] })));
                experiments.push(at(Some(Patch::Reads { variables: subset_of[c * settings.batch_sequences + b].clone() })));
                if s == 0 {
                    experiments.push(at(None));
                }
            }
            let scored = interchange.evaluate(&batch, &experiments, false)?;
            for (e, bits) in experiments.iter().zip(&scored.bits) {
                let family = match &e.patch {
                    None => {
                        clean.0.extend_from_slice(bits);
                        continue;
                    }
                    Some(Patch::Read { .. }) => 0,
                    Some(Patch::Reads { .. }) => 1,
                    // The battery draws no edits of parts.
                    Some(Patch::Part { .. } | Patch::Head { .. } | Patch::Cut { .. } | Patch::Parts { .. } | Patch::Swap { .. } | Patch::PartFrom { .. } | Patch::HeadFrom { .. } | Patch::FixedPart { .. } | Patch::Ops { .. }) => continue,
                };
                per_source[family][s].0 += bits.iter().sum::<f64>();
                per_source[family][s].1 += bits.len();
                all[family].0.extend_from_slice(bits);
            }
        }
        log::info!("battery: interchange source {}/{} ({:.0} s)", s + 1, sources.len(), started.elapsed().as_secs_f64());
    }
    let name = |f: usize| if f == 0 { "read" } else { "read_joint" };
    let mut patches = serde_json::Map::new();
    for f in 0..families {
        let means: Vec<f64> = per_source[f].iter().map(|(bits, n)| bits / *n as f64).collect();
        let worst: serde_json::Map<String, Value> =
            settings.worst_of.iter().map(|&k| (format!("worst_of_{k}"), json!(means[..k].iter().copied().fold(f64::NEG_INFINITY, f64::max)))).collect();
        patches.insert(name(f).to_string(), json!({"all_sources": all[f].summary(), "shared_source": worst}));
    }
    report["interchange"] = json!({
        "read_variables": variables.len(),
        "clean": clean.summary(),
        "patches": patches,
    });
    report["seconds"] = json!(started.elapsed().as_secs_f64());
    std::fs::write(out, serde_json::to_vec_pretty(&report).map_err(error)?).map_err(error)?;
    log::info!("battery done in {:.0} s: {out}", started.elapsed().as_secs_f64());
    Ok(())
}

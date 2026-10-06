//! Counterfactual simulatability of a library explanation on held-out text, after CHIVE
//! (Karvonen et al., arXiv 2608.16747) (`gam_mpd::library_readout`, #2951).
//!
//! EXPORT TOKENIZER SETTINGS.json OUT.json [ARTIFACT]
//!
//! A passage is the first `length` tokens of a held-out row. A behaviour is the model's predicted
//! next token `y` at a target position `t` of a passage, and its rate the probability `p(y)` that
//! `M` assigns it there. An edit replaces the token at one position `s ≤ t` with another token of
//! the held-out text, drawn uniformly; its change is `p′(y) − p(y)`, `p′` from `M` run on the
//! edited passage. CHIVE's claim about an edit is that it changes the rate by at least 30
//! percentage points; the claim is true when `|p′(y) − p(y)| ≥ 0.5` and false when it is at most
//! 0.15 (between the two it is not scored). Targets are drawn among the positions whose `p(y)` is
//! at least 0.5, where a true claim is possible.
//!
//! A predictor may read the passage, `M`'s next-token distribution at the target and an
//! explanation of the unedited forward pass, and never runs an edited passage. The explanation is
//! the artifact's account of each target ([`Library::accounts`]): the functions whose measured
//! removal raises `KL(M ‖ P)` at the target most, where each head attends from the target, each
//! MLP function's activation, and the tokens each function's write raises and lowers. Without
//! ARTIFACT the explanation is the library's start, whose functions are `M`'s own heads and MLP
//! neurons.
//!
//! The output holds, per target, `M`'s top next tokens, every edit with its change, and the
//! account; the claims and the predictors are formed from it (bench/chive_2951).
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    artifact::Artifact,
    engine::{log_to_stderr, sha256},
    import::import_language_model,
    library_mdl,
    library_readout::{Library, Vocabulary},
    operator_program::SlotValues,
    resident_causal_fit::fixed_head_target::Teacher,
    run_check::{layer_nodes, split_sites},
};
use rand::{RngExt, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

/// CHIVE's threshold of a true claim: a change of the rate of at least 50 percentage points.
const TRUE_CHANGE: f64 = 0.5;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    export_sha256: String,
    context: usize,
    /// The held-out rows `[start, end)`; one passage per row.
    held_out: [usize; 2],
    /// Tokens per passage, targets per passage, and replacement tokens per edited position.
    length: usize,
    targets: usize,
    replacements: usize,
    /// Functions per account and tokens per list in it.
    top: usize,
    tokens: usize,
    /// Functions removed at a time, and passages run at a time.
    removal_batch: usize,
    batch_sequences: usize,
    numeric_bytes: usize,
    tile_rows: usize,
    seed: u64,
}

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (export, tokenizer, settings_path, out, artifact_path) = match &args[..] {
        [e, t, s, o] => (e, t, s, o, None),
        [e, t, s, o, a] => (e, t, s, o, Some(Path::new(a))),
        _ => return Err("EXPORT TOKENIZER SETTINGS.json OUT.json [ARTIFACT]".into()),
    };
    let export = Path::new(export);
    let settings: Settings = serde_json::from_slice(&std::fs::read(settings_path).map_err(error)?).map_err(error)?;
    if sha256(&export.join("export.json"))? != settings.export_sha256 {
        return Err("export hash mismatch".into());
    }
    let [first, end] = settings.held_out;
    let length = settings.length;
    if end <= first || length < 2 || length > settings.context || settings.batch_sequences == 0 || settings.replacements == 0 || settings.tokens == 0 {
        return Err("held-out rows, passages of at least two tokens within the context, positive batches and token lists required".into());
    }
    let started = std::time::Instant::now();
    let wide = Device::single_precision(GpuPolicy::Auto).map_err(error)?;
    let model = match Device::accelerator(GpuPolicy::Auto).map_err(error)? {
        Some(cuda) => cuda,
        None => wide.clone().unwrap_or_else(Device::host),
    };
    let wide = wide.unwrap_or_else(|| model.clone());
    let imported = import_language_model(export, end, settings.context)?;
    let layer_count = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
    let native = split_sites(&imported.program)?;
    let layers = layer_nodes(&native, layer_count)?;
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { return Err("a token slot".into()) };
    let passages: Vec<Vec<u32>> = tokens.chunks(settings.context).skip(first).map(|row| row[..length].to_vec()).collect();
    let mut present: Vec<u32> = tokens.chunks(settings.context).skip(first).flatten().copied().collect();
    present.sort_unstable();
    present.dedup();
    let start = library_mdl::explanation(&native, &layers)?.artifact;
    // `M` (the library's start computes `M` exactly) gives the behaviours and the edits' changes.
    let truth = Library::new(&model, &wide, &native, &layers, &start, settings.numeric_bytes, settings.tile_rows)?;
    let artifact = match artifact_path {
        Some(path) => Some(Artifact::from_bytes(&std::fs::read(path).map_err(error)?, &native.declarations)?),
        None => None,
    };
    let explained = match &artifact {
        Some(artifact) => Some(Library::new(&model, &wide, &native, &layers, artifact, settings.numeric_bytes, settings.tile_rows)?),
        None => None,
    };
    let teacher = Teacher::new(&model, &native, settings.tile_rows, settings.numeric_bytes)?;
    drop((native, layers, start, artifact, imported));
    let explanation = explained.as_ref().unwrap_or(&truth);

    // Behaviours: per passage and position, the predicted token and its probability (and the
    // runner-up tokens at the targets).
    let mut rng = StdRng::seed_from_u64(settings.seed);
    let mut targets: Vec<(usize, usize, u32, f64, Vec<(u32, f64)>)> = Vec::new();
    for (chunk, batch) in passages.chunks(settings.batch_sequences).enumerate() {
        let run = truth.run(batch, &BTreeMap::new())?;
        let log_p = truth.log_probabilities(&run.last)?;
        for (b, _) in batch.iter().enumerate() {
            let passage = chunk * settings.batch_sequences + b;
            let mut candidates: Vec<(usize, Vec<(u32, f64)>)> = (1..length)
                .map(|t| {
                    let row = log_p.row(b * length + t);
                    let mut order: Vec<usize> = (0..row.len()).collect();
                    order.select_nth_unstable_by(settings.tokens, |x, y| row[*y].total_cmp(&row[*x]));
                    order.truncate(settings.tokens);
                    order.sort_by(|x, y| row[*y].total_cmp(&row[*x]));
                    (t, order.into_iter().map(|v| (v as u32, row[v].exp())).collect::<Vec<_>>())
                })
                .filter(|(_, top)| top[0].1 >= TRUE_CHANGE)
                .collect();
            candidates.shuffle(&mut rng);
            candidates.truncate(settings.targets);
            candidates.sort_by_key(|(t, _)| *t);
            for (t, top) in candidates {
                targets.push((passage, t, top[0].0, top[0].1, top));
            }
        }
    }
    log::info!("{} targets in {} passages", targets.len(), passages.len());

    // Edits: per passage with targets, every position up to its last target, `replacements`
    // tokens each.
    let mut by_passage: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, (passage, ..)) in targets.iter().enumerate() {
        by_passage.entry(*passage).or_default().push(i);
    }
    let mut edits: Vec<(usize, usize, u32)> = Vec::new();
    for (&passage, members) in &by_passage {
        let last = members.iter().map(|i| targets[*i].1).max().ok_or("a passage without targets")?;
        for s in 0..=last {
            for _ in 0..settings.replacements {
                let original = passages[passage][s];
                let token = loop {
                    let token = present[rng.random_range(0..present.len())];
                    if token != original {
                        break token;
                    }
                };
                edits.push((passage, s, token));
            }
        }
    }
    let mut changes: Vec<Vec<(usize, u32, f64)>> = vec![Vec::new(); targets.len()];
    for batch in edits.chunks(settings.batch_sequences) {
        let sequences: Vec<Vec<u32>> = batch
            .iter()
            .map(|(passage, s, token)| {
                let mut edited = passages[*passage].clone();
                edited[*s] = *token;
                edited
            })
            .collect();
        let run = truth.run(&sequences, &BTreeMap::new())?;
        // Each edit's rows at its passage's targets at or after the edit.
        let mut rows: Vec<(usize, usize, usize, u32)> = Vec::new();
        for (e, (passage, s, token)) in batch.iter().enumerate() {
            for &i in &by_passage[passage] {
                if targets[i].1 >= *s {
                    rows.push((i, e * length + targets[i].1, *s, *token));
                }
            }
        }
        let last = ndarray::Array2::from_shape_fn((rows.len(), run.last.ncols()), |(r, c)| run.last[[rows[r].1, c]]);
        let log_p = truth.log_probabilities(&last)?;
        for (r, (i, _, s, token)) in rows.iter().enumerate() {
            let (y, p) = (targets[*i].2, targets[*i].3);
            changes[*i].push((*s, *token, log_p[[r, y as usize]].exp() - p));
        }
    }
    log::info!("{} edits scored in {:.0} s", edits.len(), started.elapsed().as_secs_f64());

    // The explanation's accounts of the targets, on the passages that hold them.
    let held: Vec<usize> = by_passage.keys().copied().collect();
    let sequences: Vec<Vec<u32>> = held.iter().map(|p| passages[*p].clone()).collect();
    let rows: Vec<usize> = targets.iter().map(|(passage, t, ..)| held.iter().position(|p| p == passage).map(|k| k * length + t)).collect::<Option<_>>().ok_or("a target outside the held passages")?;
    let accounts = explanation.accounts(&teacher, &sequences, &rows, settings.top, settings.tokens, settings.removal_batch)?;
    log::info!("accounts in {:.0} s", started.elapsed().as_secs_f64());

    let vocabulary = Vocabulary::from_tokenizer(Path::new(tokenizer))?;
    let text = |t: u32| vocabulary.text(&[t]);
    let words = |list: &[gam_mpd::library_readout::TokenScore]| -> Vec<Value> { list.iter().map(|s| json!({"token": s.token, "text": text(s.token), "logit": s.score})).collect() };
    let report: Vec<Value> = targets
        .iter()
        .zip(&changes)
        .zip(&accounts)
        .map(|(((passage, t, y, p, top), changes), account)| {
            let functions: Vec<Value> = account
                .functions
                .iter()
                .map(|f| {
                    json!({"name": f.name, "removal_bits": f.removal_bits, "activation": f.activation,
                           "attention": f.attention.iter().map(|(u, w)| json!({"position": u, "text": text(passages[*passage][*u]), "weight": w})).collect::<Vec<_>>(),
                           "promoted": words(&f.promoted), "suppressed": words(&f.suppressed)})
                })
                .collect();
            json!({"passage": passage, "position": t, "token": y, "text": text(*y), "probability": p,
                   "top": top.iter().map(|(v, q)| json!({"token": v, "text": text(*v), "probability": q})).collect::<Vec<_>>(),
                   "edits": changes.iter().map(|(s, w, change)| json!({"position": s, "token": w, "text": text(*w), "change": change})).collect::<Vec<_>>(),
                   "account": functions})
        })
        .collect();
    let artifact_sha = match artifact_path {
        Some(path) => Some(sha256(path)?),
        None => None,
    };
    let output = json!({
        "export": export.display().to_string(),
        "artifact": artifact_path.map(|p| p.display().to_string()),
        "artifact_sha256": artifact_sha,
        "source_revision": option_env!("GAM_BUILD_GIT_SHA"),
        "model_device": model.name(),
        "held_out_sequences": [first, end],
        "length": length,
        "passages": passages.iter().map(|p| json!({"tokens": p, "text": p.iter().map(|t| text(*t)).collect::<Vec<_>>()})).collect::<Vec<_>>(),
        "targets": report,
        "seconds": started.elapsed().as_secs_f64(),
    });
    std::fs::write(out, serde_json::to_vec(&output).map_err(error)?).map_err(error)?;
    Ok(())
}

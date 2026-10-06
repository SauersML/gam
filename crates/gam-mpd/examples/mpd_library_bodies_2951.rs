//! Reusable rule bodies found from the library's own parameters and decided by the code length
//! (`gam_mpd::library_bodies`, `gam_mpd::library_crossing`, #2951).
//!
//! gate OUT [SCOREBOARD.tsv]
//! model EXPORT SETTINGS.json FROM OUT host|gpu [SCOREBOARD.tsv]
//!
//! `gate` is the fast regression gate: a toy decoder (two layers, width 16, 32 GELU units per MLP,
//! 48 tokens; 4096 training and 32 held-out sequences of 16 random tokens) whose MLPs both
//! implement one subroutine of six units on a two-dimensional input and output, in different bases
//! at the two layers and interacting with different heads (layer 0's copy is read by a head of
//! layer 1, layer 1's copy reads a head of layer 0), the other units random and weaker.
//! It is built in OUT/toy and fitted on the host. `model` runs the method on an engine export with
//! the library fit's settings (`mpd_library_mdl_2951`'s SETTINGS.json); FROM is `native` (the
//! library's start at `M`) or `checkpoint:PATH` (a library fit's checkpoint of this export).
//!
//! The method, each fit to convergence on the same fixed native experiments:
//! 1. the library is fitted (OUT/base);
//! 2. among each MLP's functions in the explanation where it starts (`M`'s, or a checkpoint's
//!    survivors), the regions are the groups a count of the parameters a rewrite saves at the
//!    posterior's resolution expects to save (`library_bodies::regions`, which lists every
//!    candidate; the others are recorded and not proposed), and, of the functions left, those
//!    reading only one head's writes (`library_crossing::regions_through`: a head and the
//!    functions it feeds, across the attention/MLP boundary);
//! 3. every region is rewritten as a call of its own body from the same start
//!    (`library_bodies::rewrite`), a region through a head reading through it
//!    (`library_crossing::read_through`), and fitted (OUT/rewritten);
//! 4. reuse by gradient: the library is fitted with the mixture prior over bodies
//!    (`library_bodies::BodyMixture`; OUT/soft{n}), each dominant component is compiled by a merge
//!    (`library_bodies::merge`), and the merged library is fitted (OUT/merged{n}) and kept when its
//!    `F` is below the extraction's so far; repeated while a merge is kept;
//! 5. the extraction with its reuse is accepted when its final `F` is below the base's. An
//!    extraction that pays only through the reuse it enables is one proposal with that reuse, so a
//!    costlier intermediate step does not end the search. A rejection means this search found no
//!    better description, not that none exists.
//!
//! OUT/SUMMARY.json: per stage `F` and the held-out evaluation (KL per token by experiment family),
//! the regions, the bodies with their calls and the native functions each replaced, the alignments
//! and every decision; per call the heads whose writes its reads take in and its reads and writes in
//! token terms; for the gate, where the planted units are. With SCOREBOARD.tsv each fitted stage
//! appends a row to it (agent "bodies").
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    engine::{log_to_stderr, sha256},
    import::import_language_model,
    library_bodies::{self, Call},
    library_crossing,
    library_mdl::{self, Explanation, Fit},
    library_mixture, library_sharing,
    operator_program::{OperatorProgram, SlotValues},
    run_check::{LayerNodes, layer_nodes, split_sites},
};
use ndarray::Array2;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn save(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(path, serde_json::to_vec_pretty(value).map_err(error)?).map_err(error)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    export_sha256: String,
    training_sequences: usize,
    context: usize,
    held_out: [usize; 2],
    fit: library_mdl::Settings,
}

/// The toy's sizes: width, heads and head width, MLP units, vocabulary, token rows, context, and the
/// planted subroutine's units.
const D: usize = 16;
const HEADS: usize = 2;
const HEAD: usize = 8;
const UNITS: usize = 32;
const VOCABULARY: usize = 48;
const ROWS: usize = 4128;
const CONTEXT: usize = 16;
const PLANTED: usize = 6;

/// The toy export in `dir` (module note) and, per layer, the native functions computing the
/// planted subroutine's units in the subroutine's order.
fn toy(dir: &Path) -> Result<[Vec<usize>; 2], String> {
    std::fs::create_dir_all(dir).map_err(error)?;
    let rng = &mut StdRng::seed_from_u64(2951);
    let uniform = |rng: &mut StdRng, rows: usize, cols: usize, scale: f64| Array2::from_shape_fn((rows, cols), |_| (rng.random::<f64>() * 2.0 - 1.0) * scale);
    let mut files = serde_json::Map::new();
    let mut write = |name: &str, values: &Array2<f64>| -> Result<(), String> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("{name}.f64")), bytes).map_err(error)?;
        files.insert(name.to_string(), json!({"shape": [values.nrows(), values.ncols()]}));
        Ok(())
    };
    let embedding = uniform(rng, VOCABULARY, D, 1.0);
    write("wte", &embedding)?;
    write("final_norm.gain", &Array2::ones((1, D)))?;
    // The subroutine: gate `G` (six units on two inputs) and output `U` (two outputs).
    let (g, u) = (uniform(rng, PLANTED, 2, 1.5), uniform(rng, 2, PLANTED, 1.0));
    let normalized = |m: Array2<f64>, norm: f64| {
        let mut m = m;
        for mut column in m.columns_mut() {
            let n = column.dot(&column).sqrt();
            column.mapv_inplace(|v| v * norm / n);
        }
        m
    };
    // Each copy reads two directions and writes two: layer 0's copy reads the normed embedding and
    // writes `W0`, which layer 1's head 0 reads through its value map and writes towards the
    // embeddings of tokens 2 and 3; layer 0's head 1 writes into `R1`, which layer 1's copy reads,
    // writing towards the embeddings of tokens 0 and 1. The copies are in different bases and
    // interact with different functions.
    let read0 = normalized(uniform(rng, D, 2, 1.0), 1.5).reversed_axes();
    let write0 = normalized(uniform(rng, D, 2, 1.0), 4.0);
    let read1 = normalized(uniform(rng, D, 2, 1.0), 1.5).reversed_axes();
    let write1 = normalized(embedding.slice(ndarray::s![0..2, ..]).t().to_owned(), 4.0);
    let tokens23 = normalized(embedding.slice(ndarray::s![2..4, ..]).t().to_owned(), 1.0);
    let mut order: Vec<usize> = (0..UNITS).collect();
    let mut sites = [Vec::new(), Vec::new()];
    for (l, (read, written)) in [(read0, write0.clone()), (read1.clone(), write1)].into_iter().enumerate() {
        for i in 0..UNITS {
            let j = rng.random_range(i..UNITS);
            order.swap(i, j);
        }
        sites[l] = order[..PLANTED].to_vec();
        let mut gate = uniform(rng, UNITS, D, 0.3);
        let mut down = uniform(rng, D, UNITS, 0.3);
        let (gr, wu) = (g.dot(&read), written.dot(&u));
        for (j, &f) in sites[l].iter().enumerate() {
            gate.row_mut(f).assign(&gr.row(j));
            down.column_mut(f).assign(&wu.column(j));
        }
        write(&format!("blocks.{l}.mlp.c_fc"), &gate)?;
        write(&format!("blocks.{l}.mlp.down_proj"), &down)?;
        for name in ["attn.q_proj", "attn.k_proj"] {
            write(&format!("blocks.{l}.{name}"), &uniform(rng, HEADS * HEAD, D, 0.5))?;
        }
        let mut value = uniform(rng, HEADS * HEAD, D, 0.5);
        let mut output = uniform(rng, D, HEADS * HEAD, 0.25);
        if l == 0 {
            // Head 1 writes into the directions layer 1's copy reads.
            let into = normalized(read1.t().dot(&uniform(rng, 2, HEAD, 1.0)), 3.0);
            output.slice_mut(ndarray::s![.., HEAD..2 * HEAD]).assign(&into);
        } else {
            // Head 0 reads what layer 0's copy wrote and writes towards tokens 2 and 3.
            let from = uniform(rng, HEAD, 2, 1.0).dot(&normalized(write0.clone(), 1.0).t());
            value.slice_mut(ndarray::s![0..HEAD, ..]).assign(&from);
            output.slice_mut(ndarray::s![.., 0..HEAD]).assign(&normalized(tokens23.dot(&uniform(rng, 2, HEAD, 1.0)), 3.0));
        }
        write(&format!("blocks.{l}.attn.v_proj"), &value)?;
        write(&format!("blocks.{l}.attn.o_proj"), &output)?;
        for gain in ["rms1", "rms2"] {
            write(&format!("blocks.{l}.{gain}.gain"), &Array2::ones((1, D)))?;
        }
    }
    let tokens = Array2::from_shape_fn((ROWS, CONTEXT), |_| rng.random_range(0..VOCABULARY) as f64);
    write("tokens", &tokens)?;
    let record = json!({
        "config": {"d_model": D, "n_layers": 2, "n_heads": HEADS, "n_kv_heads": HEADS, "head_dim": HEAD, "d_mlp": UNITS, "vocab": VOCABULARY,
                   "rope_theta": 10000.0, "rope_pairing": "rotate_half", "norm_eps": 1e-6, "mlp_act": "gelu_tanh", "tied_embeddings": true},
        "files": files,
    });
    std::fs::write(dir.join("export.json"), record.to_string()).map_err(error)?;
    Ok(sites)
}

/// One run of the method: the native model, its sequences, the fit's settings and the output.
struct Run {
    /// The model's name in the scoreboard, and the scoreboard a fitted stage appends its row to.
    model: String,
    scoreboard: Option<PathBuf>,
    device: Device,
    native: OperatorProgram,
    layers: Vec<LayerNodes>,
    train: Vec<Vec<u32>>,
    held: Vec<Vec<u32>>,
    fit: library_mdl::Settings,
    digest: String,
    out: PathBuf,
}

/// A stage's record in the summary.
fn stage(name: &str, fit: &Fit) -> Value {
    json!({
        "stage": name,
        "objective_bits": fit.report.objective_bits,
        "active_groups": fit.report.active_groups,
        "groups": fit.report.groups,
        "epochs": fit.report.epochs.len(),
        "held_out": fit.report.end,
    })
}

/// A row of the scoreboard (`SCOREBOARD.tsv`, e.g. ~/mpd-data/scoreboard/accuracy.tsv) for a fitted stage (its columns: time, commit,
/// agent, model, training tokens, optimizer, epoch, wall hours, held-out F per token, P alone's KL
/// per token, description bits, surviving groups, decoded KL per token, note).
fn scoreboard(run: &Run, model: &str, name: &str, fit: &Fit) -> Result<(), String> {
    let Some(path) = &run.scoreboard else { return Ok(()) };
    let end = serde_json::to_value(&fit.report.end).map_err(error)?;
    let number = |key: &str| end[key].as_f64().map_or("".to_string(), |v| format!("{v}"));
    let alone = end["clean"].as_array().and_then(|c| c.last()).and_then(Value::as_f64).map_or("".to_string(), |v| format!("{v}"));
    let description = ["divergence_bits", "variance_bits", "choice_bits", "prior_bits"].iter().filter_map(|k| end[*k].as_f64()).sum::<f64>();
    let time = std::process::Command::new("date").arg("-u").arg("+%Y-%m-%dT%H:%M:%SZ").output().map_err(error)?;
    let row = [
        String::from_utf8_lossy(&time.stdout).trim().to_string(),
        option_env!("GAM_BUILD_GIT_SHA").unwrap_or("").to_string(),
        "bodies".to_string(),
        model.to_string(),
        fit.report.scored_tokens.to_string(),
        "ivon".to_string(),
        fit.report.epochs.len().to_string(),
        format!("{:.3}", fit.report.seconds / 3600.0),
        number("objective_bits_per_token"),
        alone,
        format!("{description}"),
        fit.report.active_groups.to_string(),
        number("rounded_bits_per_token"),
        format!("bodies {name} {}", run.out.display()),
    ];
    let mut file = std::fs::OpenOptions::new().append(true).open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    std::io::Write::write_all(&mut file, format!("{}\n", row.join("\t")).as_bytes()).map_err(error)
}

impl Run {
    /// `explanation` fitted to convergence in OUT/`name` (resumed from its checkpoint there).
    fn fit(&self, name: &str, explanation: &Explanation) -> Result<Fit, String> {
        self.fit_with(name, explanation, None)
    }

    /// `explanation` fitted to convergence in OUT/`name` with the prior term `prior`.
    fn fit_with(&self, name: &str, explanation: &Explanation, prior: Option<&mut (dyn library_mdl::PriorTerm + 'static)>) -> Result<Fit, String> {
        let dir = self.out.join(name);
        std::fs::create_dir_all(&dir).map_err(error)?;
        let checkpoint = dir.join("checkpoint.bin");
        library_mdl::check_checkpoint(&checkpoint, &library_mdl::identity(&self.digest, &self.native, explanation, &self.train, &self.held))?;
        let fit = library_mdl::fit(&self.device, &self.native, explanation, &self.train, &self.held, &self.fit, &self.digest, Some(&checkpoint), prior)?;
        save(&dir.join("REPORT.json"), &serde_json::to_value(&fit.report).map_err(error)?)?;
        scoreboard(self, &self.model, name, &fit)?;
        Ok(fit)
    }
}

/// `explanation` at `fit`'s posterior means, with its removed groups.
fn warm(explanation: &Explanation, fit: &Fit) -> Result<Explanation, String> {
    let mut out = library_sharing::warm(explanation, &library_mdl::posterior_mean(explanation, &fit.posterior)?)?;
    out.removed = (0..fit.posterior.active.len()).filter(|g| !fit.posterior.active[*g]).collect();
    Ok(out)
}

/// The method (module note) from `base` with its posterior `posterior` (a checkpoint's; none for
/// `M`, whose posterior is the library's start); with `planted`, the gate's planted units per layer.
fn method(run: &Run, base: Explanation, posterior: Option<library_mdl::Posterior>, planted: Option<&[Vec<usize>; 2]>) -> Result<(), String> {
    let mut summary = json!({"stages": [], "decisions": []});
    let record = |summary: &mut Value, key: &str, value: Value| -> Result<(), String> {
        summary[key].as_array_mut().ok_or("a summary list")?.push(value);
        save(&run.out.join("SUMMARY.json"), summary)
    };
    let fitted = run.fit("base", &base)?;
    record(&mut summary, "stages", stage("base", &fitted))?;
    // Among each MLP's functions in the explanation where it starts, the regions at the start's
    // posterior; of the functions left, those reading through one head.
    let posterior = match posterior {
        Some(posterior) => posterior,
        None => library_mdl::Posterior::new(&base, fitted.report.scored_tokens)?,
    };
    let (mut regions, mut candidates, mut through) = (Vec::new(), Vec::new(), Vec::new());
    for l in 0..run.layers.len() {
        let pool: Vec<usize> = (0..base.layers[l].functions.len()).filter(|i| base.layers[l].functions[*i].iter().all(|g| posterior.active[*g])).collect();
        let mut taken = Vec::new();
        for (region, saving) in library_bodies::regions(&base, &posterior, l, &pool)? {
            candidates.push(json!({"layer": l, "functions": region, "saving": saving}));
            if saving > 0.0 {
                taken.extend(region.iter().copied());
                regions.push((l, region));
            }
        }
        let rest: Vec<usize> = pool.into_iter().filter(|i| !taken.contains(i)).collect();
        let writers = library_crossing::writers(&run.native, &run.layers, l)?;
        for (w, region) in library_crossing::regions_through(&base, &posterior, &writers, l, &rest)? {
            through.push((l, writers[w].clone(), writers.len(), region));
        }
    }
    summary["candidates"] = json!(candidates);
    summary["through"] = json!(through.iter().map(|(l, w, _, r)| json!({"layer": l, "head": [w.layer, w.head], "functions": r})).collect::<Vec<_>>());
    log::info!("bodies: {} regions and {} regions through a head", regions.len(), through.len());
    summary["regions"] = json!(regions);
    if let Some(planted) = planted {
        summary["planted"] = json!(planted);
    }
    save(&run.out.join("SUMMARY.json"), &summary)?;
    if regions.is_empty() && through.is_empty() {
        return Ok(());
    }
    // Every region rewritten as a call of its own body, from the same start; a region through a
    // head reads through it.
    let mut rewritten = base;
    let mut calls: Vec<Call> = Vec::new();
    for (layer, region) in &regions {
        let (next, call) = library_bodies::rewrite(&rewritten, *layer, region)?;
        rewritten = next;
        calls.push(call);
    }
    for (layer, writer, choices, region) in &through {
        let (next, call) = library_bodies::rewrite(&rewritten, *layer, region)?;
        rewritten = library_crossing::read_through(&next, &call, writer, *choices)?;
        calls.push(call);
    }
    rewritten.artifact.validate_coverage(&run.native)?;
    let mut current = run.fit("rewritten", &rewritten)?;
    record(&mut summary, "stages", stage("rewritten", &current))?;
    // The extraction is judged with its reuse (module note, step 5).
    let mut explanation = rewritten;
    // Reuse by gradient: the fit with the mixture prior over bodies (each body's components the
    // earlier bodies it aligns to, OUT/soft{n}); every dominant component made exact by a merge, the
    // merged library fitted (OUT/merged{n}) from the soft fit's means and accepted when its `F` is
    // below the library's; repeated while a merge is accepted.
    let steps = library_mixture::Steps { rate: 0.05, beta1: run.fit.beta1, beta2: 0.999, epsilon: 1e-8 };
    for round in 0.. {
        let mut mixture = library_bodies::BodyMixture::new(&explanation, steps)?;
        let soft = run.fit_with(&format!("soft{round}"), &warm(&explanation, &current)?, Some(&mut mixture))?;
        let weights: Vec<Value> = mixture.targets.iter().map(|t| json!({"body": t.body, "components": t.components.iter().map(|c| &c.body).collect::<Vec<_>>(), "weights": t.weights().unwrap_or_default()})).collect();
        let soft_start = warm(&explanation, &soft)?;
        let (merged, merged_calls, pairs) = mixture.harden(&soft_start, &calls, &soft.posterior)?;
        record(&mut summary, "stages", json!({"stage": format!("soft{round}"), "objective_bits": soft.report.objective_bits, "prior_bits": soft.report.end.prior_bits, "mixture": weights}))?;
        if pairs.is_empty() {
            break;
        }
        merged.artifact.validate_coverage(&run.native)?;
        let name = format!("merged{round}");
        let fit = run.fit(&name, &merged)?;
        record(&mut summary, "stages", stage(&name, &fit))?;
        let kept = fit.report.objective_bits < current.report.objective_bits;
        record(
            &mut summary,
            "decisions",
            json!({"move": "merge", "merged": pairs, "calls": merged_calls, "before_bits": current.report.objective_bits, "after_bits": fit.report.objective_bits, "kept": kept}),
        )?;
        if !kept {
            break;
        }
        (explanation, current, calls) = (merged, fit, merged_calls);
    }
    let accepted = current.report.objective_bits < fitted.report.objective_bits;
    record(
        &mut summary,
        "decisions",
        json!({"move": "extraction with reuse", "calls": calls, "before_bits": fitted.report.objective_bits, "after_bits": current.report.objective_bits, "accepted": accepted}),
    )?;
    if !accepted {
        return Ok(());
    }
    summary["calls"] = json!(calls);
    // The evidence for each call: the share of its reads that each head's writes take in (the
    // read binding's rows projected on the head's columns `H H⁺`; a call through a head reads it
    // alone), and its reads and writes in token terms at the accepted explanation.
    let mut evidence = Vec::new();
    for call in &calls {
        let program = &explanation.artifact.program;
        let named = |name: String| program.operators.iter().find(|op| op.name == name).map(|op| op.matrix()).ok_or(format!("no operator {name}"));
        let read = named(format!("{}.read", call.name))?;
        let mut shares: Vec<(String, f64)> = Vec::new();
        match through.iter().find(|(l, _, _, region)| *l == call.layer && call.replaced.iter().map(|(f, _)| *f).eq(region.iter().copied())) {
            Some((_, w, _, _)) => shares.push((format!("L{}.H{}", w.layer, w.head), 1.0)),
            None => {
                let total = read.iter().map(|v| v * v).sum::<f64>();
                for w in library_crossing::writers(&run.native, &run.layers, call.layer)? {
                    let projected = read.dot(&w.writes).dot(&gam_linalg::decompose::pseudo_inverse(w.writes.view()).map_err(error)?);
                    shares.push((format!("L{}.H{}", w.layer, w.head), projected.iter().map(|v| v * v).sum::<f64>() / total));
                }
                shares.sort_by(|a, b| b.1.total_cmp(&a.1));
                shares.truncate(4);
            }
        }
        evidence.push(json!({"call": call.name, "body": call.body, "layer": call.layer, "heads": shares}));
    }
    summary["evidence"] = json!(evidence);
    summary["readings"] = json!(library_bodies::describe(&run.native, &run.layers, &explanation, &calls, 8)?);
    save(&run.out.join("SUMMARY.json"), &summary)
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("gate") => {
            let (out, scoreboard) = match &args[..] {
                [_, out] => (out, None),
                [_, out, board] => (out, Some(PathBuf::from(board))),
                _ => return Err("gate OUT [SCOREBOARD.tsv]".into()),
            };
            let out = Path::new(out);
            let planted = toy(&out.join("toy"))?;
            let imported = import_language_model(&out.join("toy"), ROWS, CONTEXT)?;
            let native = split_sites(&imported.program)?;
            let layers = layer_nodes(&native, 2)?;
            let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { return Err("a token slot".into()) };
            let sequences: Vec<Vec<u32>> = tokens.chunks(CONTEXT).map(<[u32]>::to_vec).collect();
            let fit = library_mdl::Settings {
                batch_sequences: 32,
                rate: 0.1,
                beta1: 0.9,
                seed: 1,
                numeric_bytes: 1 << 28,
                head_tile_rows: 64,
            };
            let base = library_mdl::explanation(&native, &layers)?;
            let run = Run {
                model: "bodies toy".to_string(),
                scoreboard,
                device: Device::host(),
                digest: sha256(&out.join("toy").join("export.json"))?,
                train: sequences[..ROWS - 32].to_vec(),
                held: sequences[ROWS - 32..].to_vec(),
                native,
                layers,
                fit,
                out: out.to_path_buf(),
            };
            method(&run, base, None, Some(&planted))
        }
        Some("model") => {
            let (export, settings_path, from, out, mode, scoreboard) = match &args[..] {
                [_, export, settings, from, out, mode] => (export, settings, from, out, mode, None),
                [_, export, settings, from, out, mode, board] => (export, settings, from, out, mode, Some(PathBuf::from(board))),
                _ => return Err("model EXPORT SETTINGS.json native|checkpoint:PATH OUT host|gpu [SCOREBOARD.tsv]".into()),
            };
            let (export, out) = (Path::new(export), Path::new(out));
            let settings: Settings = serde_json::from_slice(&std::fs::read(settings_path).map_err(error)?).map_err(error)?;
            if sha256(&export.join("export.json"))? != settings.export_sha256 {
                return Err("export hash mismatch".into());
            }
            let [first, end] = settings.held_out;
            if first >= end || settings.training_sequences == 0 {
                return Err("held-out sequences must be a nonempty range, and training sequences nonempty".into());
            }
            let device = match mode.as_str() {
                "host" => Device::host(),
                "gpu" => Device::single_precision(GpuPolicy::Required).map_err(error)?.ok_or("a GPU required")?,
                _ => return Err("host|gpu required".into()),
            };
            let rows = end.max(settings.training_sequences + if settings.training_sequences > first { end - first } else { 0 });
            let imported = import_language_model(export, rows, settings.context)?;
            let layer_count = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
            let native = split_sites(&imported.program)?;
            let layers = layer_nodes(&native, layer_count)?;
            let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { return Err("a token slot".into()) };
            let sequences: Vec<Vec<u32>> = tokens.chunks(settings.context).map(<[u32]>::to_vec).collect();
            let train: Vec<Vec<u32>> = sequences[..first].iter().chain(&sequences[end..]).take(settings.training_sequences).cloned().collect();
            if train.len() != settings.training_sequences {
                return Err("the export holds fewer training sequences than asked for".into());
            }
            let start = library_mdl::explanation(&native, &layers)?;
            let (base, posterior) = match from.split_once(':') {
                None if from == "native" => (start, None),
                Some(("checkpoint", path)) => {
                    // Entry by entry along the operators' own axes, as the bodies read it.
                    let posterior = library_mdl::checkpoint_posterior(&start, Path::new(path))?;
                    let mut base = library_sharing::warm(&start, &library_mdl::posterior_mean(&start, &posterior)?)?;
                    base.removed = (0..posterior.active.len()).filter(|g| !posterior.active[*g]).collect();
                    (base, Some(posterior))
                }
                _ => return Err("FROM is native or checkpoint:PATH".into()),
            };
            std::fs::create_dir_all(out).map_err(error)?;
            let model = export.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let run = Run { model, scoreboard, device, native, layers, held: sequences[first..end].to_vec(), train, fit: settings.fit, digest: settings.export_sha256, out: out.to_path_buf() };
            method(&run, base, posterior, None)
        }
        _ => Err("gate OUT [SCOREBOARD.tsv] | model EXPORT SETTINGS.json FROM OUT host|gpu [SCOREBOARD.tsv]".into()),
    }
}

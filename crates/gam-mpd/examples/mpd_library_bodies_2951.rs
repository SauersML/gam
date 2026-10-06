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
//! library's start at `M`), `checkpoint:PATH` (a library fit's checkpoint of this export, its means
//! and removed groups, from which the base fit makes its own Laplace start) or `start:PATH` (the
//! base fit continues that checkpoint exactly: its posterior, optimizer state, removals and epoch,
//! so the regions are read at the checkpoint's resolution; with the settings' budget of epochs,
//! `fit.epochs`, every fit of the method stops there).
//!
//! The method, each fit to convergence on the same fixed native experiments:
//! 1. the library is fitted (OUT/base);
//! 2. among each MLP's functions in the fitted library, at its posterior, the regions are those reading only one head's writes
//!    (`library_crossing::regions_through`: heads and the functions they feed, across the
//!    attention/MLP boundary; linear in the MLP's width) and, of the functions left, in the gate
//!    only, the groups a count of the parameters a rewrite saves at the posterior's resolution
//!    expects to save (`library_bodies::regions`, which lists every candidate; the others are
//!    recorded and not proposed). That search evaluates on the order of `n⁴` unions of an MLP's `n`
//!    functions, so at a model's width only the regions through a head are proposed;
//! 3. extract-and-reuse, one transaction per proposal (OUT/t{n}): the child is the accepted
//!    explanation with the proposal made, fitted, and accepted when its `F` is below the accepted
//!    one's; a rejection does not end the search. Proposals in order: two regions whose bodies
//!    align with evidence (each extracted alone, `library_bodies::align`, by their fit statistic),
//!    extracted together as one body called at both sites (`library_bodies::rewrite`, a region
//!    through a head reading through it, `library_crossing::read_through`, then
//!    `library_bodies::merge`), and where each reads through one head, with the two heads made one
//!    head function as well (`library_crossing::share_writers`: a head and the functions it feeds,
//!    one unit at both sites); then each region left as a call of an accepted body it aligns to;
//!    then each region left as its own body. An extraction that pays only through the reuse it
//!    enables is proposed with that reuse, so a costlier intermediate never ends the search;
//! 4. reuse by gradient among the accepted bodies: the fit with the mixture prior over bodies
//!    (`library_bodies::BodyMixture`; OUT/soft{n}), its dominant components merged
//!    (`library_bodies::BodyMixture::harden`), each hardening one transaction.
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
    /// Whether the grown regions are searched (module note: the gate's width only).
    grown: bool,
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
    /// `explanation` fitted to convergence in OUT/`name` with the prior term `prior`, from `start`
    /// (a move's start carrying its parent's fit) when given.
    fn fit_with(&self, name: &str, explanation: &Explanation, prior: Option<&mut (dyn library_mdl::PriorTerm + 'static)>, start: Option<library_mdl::Start>) -> Result<Fit, String> {
        let dir = self.out.join(name);
        std::fs::create_dir_all(&dir).map_err(error)?;
        let checkpoint = dir.join("checkpoint.bin");
        library_mdl::check_checkpoint(&checkpoint, &library_mdl::identity(&self.digest, &self.native, explanation, &self.train, &self.held))?;
        let fit = library_mdl::fit_from(&self.device, &self.native, explanation, &self.train, &self.held, &self.fit, &self.digest, Some(&checkpoint), prior, start)?;
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

/// A region the method may extract: its layer, its functions, and the heads it reads through (with
/// the number of heads its MLP may read; none for a region reading the stream).
#[derive(Clone, serde::Serialize)]
struct Region {
    layer: usize,
    functions: Vec<usize>,
    #[serde(serialize_with = "heads")]
    through: Option<(Vec<library_crossing::Writer>, usize)>,
}

fn heads<S: serde::Serializer>(through: &Option<(Vec<library_crossing::Writer>, usize)>, serializer: S) -> Result<S::Ok, S::Error> {
    serde::Serialize::serialize(&through.as_ref().map(|(ws, _)| ws.iter().map(|w| [w.layer, w.head]).collect::<Vec<_>>()), serializer)
}

/// `explanation` with `region` rewritten as a call of its own body, read through its heads.
fn extract(explanation: &Explanation, region: &Region) -> Result<(Explanation, Call), String> {
    let (next, call) = library_bodies::rewrite(explanation, region.layer, &region.functions)?;
    Ok(match &region.through {
        Some((writers, choices)) => (library_crossing::read_through(&next, &call, &writers.iter().collect::<Vec<_>>(), *choices)?, call),
        None => (next, call),
    })
}

/// The method (module note) from `base`; with `planted`, the gate's planted units per layer.
fn method(run: &Run, base: Explanation, planted: Option<&[Vec<usize>; 2]>, begin: Option<library_mdl::Start>) -> Result<(), String> {
    let mut summary = json!({"stages": [], "decisions": []});
    let record = |summary: &mut Value, key: &str, value: Value| -> Result<(), String> {
        summary[key].as_array_mut().ok_or("a summary list")?.push(value);
        save(&run.out.join("SUMMARY.json"), summary)
    };
    let fitted = run.fit_with("base", &base, None, begin)?;
    record(&mut summary, "stages", stage("base", &fitted))?;
    // Among each MLP's functions in the fitted explanation, the regions at its posterior (the one
    // that resolves what the data determine; the extraction starts from its means).
    let posterior = &fitted.posterior.clone();
    let (mut regions, mut candidates, mut through) = (Vec::new(), Vec::new(), Vec::new());
    for l in 0..run.layers.len() {
        let pool: Vec<usize> = (0..base.layers[l].functions.len()).filter(|i| base.layers[l].functions[*i].iter().all(|g| posterior.active[*g])).collect();
        let writers = library_crossing::writers(&run.native, &run.layers, l)?;
        let mut taken = Vec::new();
        for (set, region) in library_crossing::regions_through(&base, posterior, &writers, l, &pool)? {
            taken.extend(region.iter().copied());
            through.push((l, set.iter().map(|w| writers[*w].clone()).collect::<Vec<_>>(), writers.len(), region));
        }
        if run.grown {
            let rest: Vec<usize> = pool.into_iter().filter(|i| !taken.contains(i)).collect();
            for (region, saving) in library_bodies::regions(&base, posterior, l, &rest)? {
                candidates.push(json!({"layer": l, "functions": region, "saving": saving}));
                if saving > 0.0 {
                    regions.push((l, region));
                }
            }
        }
    }
    summary["candidates"] = json!(candidates);
    summary["through"] = json!(through.iter().map(|(l, ws, _, r)| json!({"layer": l, "heads": ws.iter().map(|w| [w.layer, w.head]).collect::<Vec<_>>(), "functions": r})).collect::<Vec<_>>());
    log::info!("bodies: {} regions and {} regions through heads", regions.len(), through.len());
    summary["regions"] = json!(regions);
    if let Some(planted) = planted {
        summary["planted"] = json!(planted);
    }
    save(&run.out.join("SUMMARY.json"), &summary)?;
    let mut pending: Vec<Region> = through
        .iter()
        .map(|(layer, writers, choices, functions)| Region { layer: *layer, functions: functions.clone(), through: Some((writers.clone(), *choices)) })
        .chain(regions.iter().map(|(layer, functions)| Region { layer: *layer, functions: functions.clone(), through: None }))
        .collect();
    // Extract-and-reuse, one transaction per proposal (module note, step 3): every child is the
    // accepted explanation with the proposal made, fitted, and accepted when its `F` is below the
    // accepted one's; a rejected proposal does not end the search.
    let (mut explanation, mut current) = (base, fitted);
    let mut calls: Vec<Call> = Vec::new();
    let mut transaction = 0;
    let mut decide = |summary: &mut Value, kind: &str, detail: Value, child: Explanation, child_calls: Vec<Call>, explanation: &mut Explanation, current: &mut Fit, calls: &mut Vec<Call>| -> Result<bool, String> {
        child.artifact.validate_coverage(&run.native)?;
        let name = format!("t{transaction}");
        transaction += 1;
        // The child starts from its parent's fit: what the move left alone keeps its posterior.
        let start = library_bodies::carried(explanation, &current.posterior, &child, current.report.scored_tokens)?;
        let fit = run.fit_with(&name, &child, None, Some(start))?;
        record(summary, "stages", stage(&name, &fit))?;
        let accepted = fit.report.objective_bits < current.report.objective_bits;
        record(summary, "decisions", json!({"move": kind, "stage": name, "detail": detail, "before_bits": current.report.objective_bits, "after_bits": fit.report.objective_bits, "accepted": accepted}))?;
        if accepted {
            (*explanation, *current, *calls) = (child, fit, child_calls);
        }
        Ok(accepted)
    };
    // Reuse first: the pairs of proposals whose bodies align with evidence (each extracted alone
    // from the accepted explanation, aligned at the start's resolution), in order of their fit
    // statistic; then the proposals left onto the bodies accepted so far; then each proposal
    // alone.
    loop {
        let start = warm(&explanation, &current)?;
        let tokens = current.report.scored_tokens;
        let shadows = pending.iter().map(|r| extract(&start, r)).collect::<Result<Vec<_>, _>>()?;
        let values = shadows
            .iter()
            .map(|(e, call)| library_bodies::body_values(e, &library_mdl::Posterior::new(e, tokens)?, &call.body))
            .collect::<Result<Vec<_>, String>>()?;
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..pending.len() {
            for j in i + 1..pending.len() {
                if let Ok(alignment) = library_bodies::align(&values[j], &values[i])
                    && alignment.constrains()
                {
                    pairs.push((alignment.misfit / (alignment.entries - alignment.gauge) as f64, i, j));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut accepted = None;
        for &(statistic, i, j) in &pairs {
            let (one, first) = extract(&start, &pending[i])?;
            let (two, second) = extract(&one, &pending[j])?;
            let posterior = library_mdl::Posterior::new(&two, tokens)?;
            let alignment = library_bodies::align(&library_bodies::body_values(&two, &posterior, &second.body)?, &library_bodies::body_values(&two, &posterior, &first.body)?)?;
            let mut child_calls = calls.clone();
            child_calls.extend([first.clone(), second.clone()]);
            let (child, child_calls) = library_bodies::merge(&two, &child_calls, &second.body, &first.body, &alignment)?;
            // When each region reads through one head of its own layer, the heads can be one head
            // function too: the head and the functions it feeds one unit at both sites.
            let heads = match (&pending[i].through, &pending[j].through) {
                (Some((a, _)), Some((b, _))) if a.len() == 1 && b.len() == 1 && a[0].layer != b[0].layer => Some((a[0].clone(), b[0].clone())),
                _ => None,
            };
            let detail = json!({"regions": [&pending[i], &pending[j]], "statistic": statistic, "alignment": alignment});
            let with_heads = |e: &Explanation| heads.as_ref().map(|(a, b)| library_crossing::share_writers(e, a, b)).transpose();
            let composite = with_heads(&child)?;
            if decide(&mut summary, "extract two regions as one body", detail.clone(), child, child_calls.clone(), &mut explanation, &mut current, &mut calls)? {
                if let Some(shared) = with_heads(&warm(&explanation, &current)?)? {
                    decide(&mut summary, "make the heads the body reads one head function", detail, shared, calls.clone(), &mut explanation, &mut current, &mut calls)?;
                }
                accepted = Some((i, j));
                break;
            }
            if let Some(composite) = composite
                && decide(&mut summary, "extract two regions and the heads they read as one unit", detail, composite, child_calls, &mut explanation, &mut current, &mut calls)?
            {
                accepted = Some((i, j));
                break;
            }
        }
        match accepted {
            Some((i, j)) => {
                pending.remove(j);
                pending.remove(i);
            }
            None => break,
        }
    }
    // The proposals left onto an accepted body, then alone.
    let mut index = 0;
    while index < pending.len() {
        let start = warm(&explanation, &current)?;
        let tokens = current.report.scored_tokens;
        let (child, call) = extract(&start, &pending[index])?;
        let posterior = library_mdl::Posterior::new(&child, tokens)?;
        let fresh = library_bodies::body_values(&child, &posterior, &call.body)?;
        let mut bodies: Vec<&String> = calls.iter().map(|c| &c.body).collect();
        bodies.sort();
        bodies.dedup();
        let mut onto: Vec<(f64, String, library_bodies::Alignment)> = Vec::new();
        for body in bodies {
            if let Ok(alignment) = library_bodies::align(&fresh, &library_bodies::body_values(&child, &posterior, body)?)
                && alignment.constrains()
            {
                onto.push((alignment.misfit / (alignment.entries - alignment.gauge) as f64, body.clone(), alignment));
            }
        }
        onto.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut done = false;
        for (statistic, body, alignment) in onto {
            let mut child_calls = calls.clone();
            child_calls.push(call.clone());
            let (merged, merged_calls) = library_bodies::merge(&child, &child_calls, &call.body, &body, &alignment)?;
            let detail = json!({"region": &pending[index], "onto": body, "statistic": statistic});
            if decide(&mut summary, "extract a region as a call of an accepted body", detail, merged, merged_calls, &mut explanation, &mut current, &mut calls)? {
                done = true;
                break;
            }
        }
        if !done {
            let mut child_calls = calls.clone();
            child_calls.push(call);
            done = decide(&mut summary, "extract a region as its own body", json!({"region": &pending[index]}), child, child_calls, &mut explanation, &mut current, &mut calls)?;
        }
        if done {
            pending.remove(index);
        } else {
            index += 1;
        }
    }
    if calls.is_empty() {
        return Ok(());
    }
    // Reuse by gradient among the accepted bodies: the fit with the mixture prior over bodies
    // (OUT/soft{n}); its dominant components made exact by merges, each hardening one transaction.
    let steps = library_mixture::Steps { rate: 0.05, beta1: run.fit.beta1, beta2: 0.999, epsilon: 1e-8 };
    for round in 0.. {
        let mut mixture = library_bodies::BodyMixture::new(&explanation, steps)?;
        let warmed = warm(&explanation, &current)?;
        let start = library_bodies::carried(&explanation, &current.posterior, &warmed, current.report.scored_tokens)?;
        let soft = run.fit_with(&format!("soft{round}"), &warmed, Some(&mut mixture), Some(start))?;
        let weights: Vec<Value> = mixture.targets.iter().map(|t| json!({"body": t.body, "components": t.components.iter().map(|c| &c.body).collect::<Vec<_>>(), "weights": t.weights().unwrap_or_default()})).collect();
        record(&mut summary, "stages", json!({"stage": format!("soft{round}"), "objective_bits": soft.report.objective_bits, "prior_bits": soft.report.end.prior_bits, "mixture": weights}))?;
        let (merged, merged_calls, pairs) = mixture.harden(&warm(&explanation, &soft)?, &calls, &soft.posterior)?;
        if pairs.is_empty() || !decide(&mut summary, "merge bodies the mixture prior found", json!({"merged": pairs}), merged, merged_calls, &mut explanation, &mut current, &mut calls)? {
            break;
        }
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
            Some((_, ws, _, _)) => shares.extend(ws.iter().map(|w| (format!("L{}.H{}", w.layer, w.head), 1.0 / ws.len() as f64))),
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
                beta1: 0.9,
                seed: 1,
                numeric_bytes: 1 << 28,
                head_tile_rows: 64,
                epochs: None,
            };
            let base = library_mdl::explanation(&native, &layers)?;
            let run = Run {
                model: "bodies toy".to_string(),
                grown: true,
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
            method(&run, base, Some(&planted), None)
        }
        Some("model") => {
            let (export, settings_path, from, out, mode, scoreboard) = match &args[..] {
                [_, export, settings, from, out, mode] => (export, settings, from, out, mode, None),
                [_, export, settings, from, out, mode, board] => (export, settings, from, out, mode, Some(PathBuf::from(board))),
                _ => return Err("model EXPORT SETTINGS.json native|checkpoint:PATH|start:PATH OUT host|gpu [SCOREBOARD.tsv]".into()),
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
            let mut begin = None;
            let base = match from.split_once(':') {
                None if from == "native" => start,
                Some(("start", path)) => {
                    begin = Some(library_mdl::checkpoint_start(&start, Path::new(path))?);
                    start
                }
                Some(("checkpoint", path)) => {
                    // Entry by entry along the operators' own axes, as the bodies read it.
                    let posterior = library_mdl::checkpoint_posterior(&start, Path::new(path))?;
                    let mut base = library_sharing::warm(&start, &library_mdl::posterior_mean(&start, &posterior)?)?;
                    base.removed = (0..posterior.active.len()).filter(|g| !posterior.active[*g]).collect();
                    base
                }
                _ => return Err("FROM is native, checkpoint:PATH or start:PATH".into()),
            };
            std::fs::create_dir_all(out).map_err(error)?;
            let model = export.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let run = Run { model, grown: false, scoreboard, device, native, layers, held: sequences[first..end].to_vec(), train, fit: settings.fit, digest: settings.export_sha256, out: out.to_path_buf() };
            method(&run, base, None, begin)
        }
        _ => Err("gate OUT [SCOREBOARD.tsv] | model EXPORT SETTINGS.json FROM OUT host|gpu [SCOREBOARD.tsv]".into()),
    }
}

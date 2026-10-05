//! Reusable rule bodies found from flows and decided by the code length
//! (`gam_mpd::library_bodies`, #2951).
//!
//! gate OUT
//! model EXPORT SETTINGS.json FROM OUT host|gpu
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
//! 2. among each MLP's functions carrying RelP flow at the fitted posterior mean on the held-out
//!    sequences (`library_readout`), the regions are the groups whose rewrite saves parameters at
//!    the posterior's resolution (`library_bodies::regions`);
//! 3. every region is rewritten as a call of its own body from the fitted library
//!    (`library_bodies::rewrite`), fitted (OUT/rewritten), and accepted when its code length `F` is
//!    below the base's;
//! 4. reuse by gradient: the library is fitted with the mixture prior over bodies
//!    (`library_bodies::BodyMixture`; OUT/soft{n}), each dominant component is made exact by a merge
//!    (`library_bodies::merge`), and the merged library is fitted (OUT/merged{n}) and accepted when
//!    `F` falls; repeated while a merge is accepted.
//!
//! OUT/SUMMARY.json: per stage `F` and the held-out evaluation (KL per token by experiment family),
//! the regions, the bodies with their calls and the native functions each replaced, the alignments
//! and every decision; for the gate, where the planted units are.
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    engine::{log_to_stderr, sha256},
    import::import_language_model,
    library_bodies::{self, Call},
    library_mdl::{self, Explanation, Fit},
    library_mixture, library_readout,
    library_sharing,
    operator_program::{OperatorProgram, SlotValues},
    run_check::{LayerNodes, layer_nodes, split_sites},
};
use ndarray::Array2;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

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

/// A function of the explanation in the flows: a head or an MLP function of a layer.
#[derive(Clone, Copy, Debug, serde::Serialize)]
enum Function {
    Head { layer: usize, head: usize },
    Mlp { layer: usize, function: usize },
}

/// One run of the method: the native model, its sequences, the fit's settings and the output.
struct Run {
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
        Ok(fit)
    }

    /// The RelP flows among the functions of `explanation` as it stands (its artifact's values) on
    /// the held-out sequences: `flows[t][s]` from `s` to `t` among the functions with any flow.
    fn flows(&self, explanation: &Explanation) -> Result<(Array2<f64>, Vec<Function>), String> {
        let artifact = &explanation.artifact;
        let all: usize = explanation.layers.iter().map(|l| l.heads.len() + l.functions.len()).sum();
        let settings = library_readout::Settings {
            batch_sequences: self.fit.batch_sequences,
            numeric_bytes: self.fit.numeric_bytes,
            tile_rows: self.fit.head_tile_rows,
            contexts: 1,
            tokens: 1,
            edges: all,
            candidates: all,
            core: 0,
            thresholds: vec![1.0],
            targets: self.held[0].len(),
        };
        let library = library_readout::Library::new(&self.device, &self.device, &self.native, &self.layers, artifact, settings.numeric_bytes, settings.tile_rows)?;
        let readout = library_readout::read_out(&library, &self.held, &settings)?;
        let parse = |name: &str, layer: usize| -> Result<Function, String> {
            let (_, rest) = name.split_once('.').ok_or_else(|| format!("function name {name}"))?;
            let index = |s: &str| s.parse::<usize>().map_err(|e| format!("function name {name}: {e}"));
            match (rest.strip_prefix('H'), rest.strip_prefix('M')) {
                (Some(h), _) => Ok(Function::Head { layer, head: index(h)? }),
                (_, Some(m)) => Ok(Function::Mlp { layer, function: index(m)? }),
                _ => Err(format!("function name {name}")),
            }
        };
        let mut index: BTreeMap<usize, usize> = BTreeMap::new();
        for (t, f) in readout.functions.iter().enumerate() {
            for edge in &f.inputs {
                for v in [t, edge.from] {
                    let next = index.len();
                    index.entry(v).or_insert(next);
                }
            }
        }
        let mut flows = Array2::zeros((index.len(), index.len()));
        let mut functions = vec![Function::Head { layer: 0, head: 0 }; index.len()];
        for (&v, &at) in &index {
            functions[at] = parse(&readout.functions[v].name, readout.functions[v].layer)?;
        }
        for (t, f) in readout.functions.iter().enumerate() {
            for edge in &f.inputs {
                flows[[index[&t], index[&edge.from]]] += edge.flow;
            }
        }
        Ok((flows, functions))
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
    // The functions carrying flow where the library starts, and among each MLP's the regions at the
    // start's posterior.
    let posterior = match posterior {
        Some(posterior) => posterior,
        None => library_mdl::Posterior::new(&base, fitted.report.scored_tokens)?,
    };
    let (flows, functions) = run.flows(&base)?;
    save(&run.out.join("FLOWS.json"), &json!({"functions": functions, "flows": flows.outer_iter().map(|r| r.to_vec()).collect::<Vec<_>>()}))?;
    let mut regions = Vec::new();
    for l in 0..run.layers.len() {
        let pool: Vec<usize> = functions
            .iter()
            .filter_map(|f| match f {
                Function::Mlp { layer, function } if *layer == l => Some(*function),
                Function::Mlp { .. } | Function::Head { .. } => None,
            })
            .collect();
        regions.extend(library_bodies::regions(&base, &posterior, l, &pool)?.into_iter().map(|r| (l, r)));
    }
    log::info!("bodies: {} regions among {} functions carrying flow: {regions:?}", regions.len(), functions.len());
    summary["regions"] = json!(regions);
    if let Some(planted) = planted {
        summary["planted"] = json!(planted);
    }
    save(&run.out.join("SUMMARY.json"), &summary)?;
    if regions.is_empty() {
        return Ok(());
    }
    // Every region rewritten as a call of its own body, from the same start.
    let mut rewritten = base;
    let mut calls: Vec<Call> = Vec::new();
    for (layer, region) in &regions {
        let (next, call) = library_bodies::rewrite(&rewritten, *layer, region)?;
        rewritten = next;
        calls.push(call);
    }
    rewritten.artifact.validate_coverage(&run.native)?;
    let mut current = run.fit("rewritten", &rewritten)?;
    record(&mut summary, "stages", stage("rewritten", &current))?;
    let accepted = current.report.objective_bits < fitted.report.objective_bits;
    record(&mut summary, "decisions", json!({"move": "rewrite", "calls": calls, "before_bits": fitted.report.objective_bits, "after_bits": current.report.objective_bits, "accepted": accepted}))?;
    if !accepted {
        return Ok(());
    }
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
        let accepted = fit.report.objective_bits < current.report.objective_bits;
        record(
            &mut summary,
            "decisions",
            json!({"move": "merge", "merged": pairs, "calls": merged_calls, "before_bits": current.report.objective_bits, "after_bits": fit.report.objective_bits, "accepted": accepted}),
        )?;
        if !accepted {
            break;
        }
        (explanation, current, calls) = (merged, fit, merged_calls);
    }
    summary["calls"] = json!(calls);
    // The evidence for each call: the functions whose flows enter and leave its region (RelP flows
    // at the start), and its reads and writes in token terms at the accepted explanation.
    let name = |f: &Function| match f {
        Function::Head { layer, head } => format!("L{layer}.H{head}"),
        Function::Mlp { layer, function } => format!("L{layer}.M{function}"),
    };
    let mut evidence = Vec::new();
    for call in &calls {
        let inside: Vec<usize> = (0..functions.len())
            .filter(|&v| matches!(functions[v], Function::Mlp { layer, function } if layer == call.layer && call.replaced.iter().any(|(f, _)| *f == function)))
            .collect();
        let mut into: BTreeMap<String, f64> = BTreeMap::new();
        let mut out_of: BTreeMap<String, f64> = BTreeMap::new();
        for &v in &inside {
            for u in (0..functions.len()).filter(|u| !inside.contains(u)) {
                *into.entry(name(&functions[u])).or_default() += flows[[v, u]].abs();
                *out_of.entry(name(&functions[u])).or_default() += flows[[u, v]].abs();
            }
        }
        let strongest = |m: BTreeMap<String, f64>| {
            let mut v: Vec<(String, f64)> = m.into_iter().filter(|(_, f)| *f > 0.0).collect();
            v.sort_by(|a, b| b.1.total_cmp(&a.1));
            v.truncate(8);
            v
        };
        evidence.push(json!({"call": call.name, "body": call.body, "layer": call.layer, "flow_in": strongest(into), "flow_out": strongest(out_of)}));
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
            let [_, out] = &args[..] else { return Err("gate OUT".into()) };
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
            let [_, export, settings_path, from, out, mode] = &args[..] else {
                return Err("model EXPORT SETTINGS.json native|checkpoint:PATH OUT host|gpu".into());
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
                    let posterior = library_mdl::checkpoint_posterior(&start, Path::new(path))?;
                    let mut base = library_sharing::warm(&start, &library_mdl::posterior_mean(&start, &posterior)?)?;
                    base.removed = (0..posterior.active.len()).filter(|g| !posterior.active[*g]).collect();
                    (base, Some(posterior))
                }
                _ => return Err("FROM is native or checkpoint:PATH".into()),
            };
            std::fs::create_dir_all(out).map_err(error)?;
            let run = Run { device, native, layers, held: sequences[first..end].to_vec(), train, fit: settings.fit, digest: settings.export_sha256, out: out.to_path_buf() };
            method(&run, base, posterior, None)
        }
        _ => Err("gate OUT | model EXPORT SETTINGS.json FROM OUT host|gpu".into()),
    }
}

//! The graph oracle's checker as a JSON-lines server (#2951): one request per stdin line, one JSON
//! answer per stdout line.
//!
//! * `{"op": "load", "export": DIR, "vpd": DIR?, "transcoders": DIR?, "library": START_JSON?,
//!   "library_arm": ARM?}`: the native model of an export (`import::import_language_model`), with
//!   VPD's, the transcoders' or the library's view attached when named;
//!   its weights read from the native program (`graph::Weights::from_native`).
//! * `{"op": "behavior", "path": FILE}` or `{"op": "behavior", "behavior": {...}}`: a behavior file
//!   (design.txt section 5); measures its stand-in averages on `M`. With `"manifest": FILE` (an
//!   immutable experiment manifest, `mpd_library_mdl_2951`'s or `draw_manifest`'s; by default
//!   `~/mpd-data/compare/manifest/MANIFEST_vpd4l_s1.json` for the vpd4l export and
//!   `~/mpd-data/graph_oracle/experiments/MANIFEST_{export directory}_s1.json` for others when it
//!   exists, `null` for none), a third of the behavior's half of the experiments are its site
//!   operations (`graph::SiteUnits::manifest`, positions drawn on the manifest's or the export's
//!   context). A failed request keeps the loaded model.
//! * `{"op": "score", "program": IR}` or `{"op": "score", "programs": [IR, ...]}` (the answer
//!   `{"ok", "scores": [...], "seconds"}`), with `"experiments": 32, "seed": 0, "routing": "edges" |
//!   "nodes", "N": null, "reader_top": 0, "uniform_seeds": null`: every score term (`graph::Score`)
//!   per program, all programs under one seed (`Checker::score_batch`: the behavior's half of the
//!   experiments shared, `M` once per experiment, runs on parallel threads); with `reader_top` k > 0,
//!   per experiment its words and per target token `M_e`'s and the program's probabilities of `M`'s
//!   k most probable clean tokens and of everything else (the reader's items); with
//!   `uniform_seeds` m, experiments from seed mod m (`M`'s cache serves recurring seeds).
//!   `GRAPH_CACHE_GIB` bounds `M`'s cached outcomes (2), `GRAPH_DISK_CACHE` shares them on disk.
//! * `{"op": "draw_manifest", "out": FILE, "seed": 1, "count": 1024, "length": 512, "sequences": 8,
//!   "families": [...]}`: an immutable manifest of site operations for a model without one
//!   (`graph::SiteUnits::write_manifest` on the export's first token rows), made the pool.
//! * `{"op": "quit"}`.
//!
//! Command line: `--cache-gib G` sets each checker's budget of `M`'s cached outcomes
//! (`Checker::cache_bytes`, 4 GiB by default) and `--disk-cache DIR` its disk cache, shared by
//! every checker process given the same directory (`Checker::disk_cache`, none by default).
use gam_gpu::tensor::Device;
use gam_mpd::{
    engine::log_to_stderr,
    graph::{Behavior, Checker, Experiment, Measured, Program, SiteUnits, WeightEdit, Weights},
    import::import_language_model,
    run_check::{layer_nodes, split_sites},
};
use serde_json::{Value, json};
use std::io::{BufRead, Write};
use std::path::Path;

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// The checkers' caches from the command line (module note).
#[derive(Default)]
struct Caches {
    bytes: Option<usize>,
    disk: Option<std::path::PathBuf>,
}

impl Caches {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut caches = Self::default();
        while let Some(flag) = args.next() {
            let value = args.next().ok_or_else(|| format!("{flag} without its value"))?;
            match flag.as_str() {
                "--cache-gib" => caches.bytes = Some((value.parse::<f64>().map_err(error)? * f64::from(1u32 << 30)) as usize),
                "--disk-cache" => caches.disk = Some(value.into()),
                _ => return Err(format!("unknown flag {flag} (--cache-gib G, --disk-cache DIR)")),
            }
        }
        Ok(caches)
    }
}

fn load(export: &Path) -> Result<Weights, String> {
    let imported = import_language_model(export, 1, 1)?;
    let layer_count = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
    let native = split_sites(&imported.program)?;
    drop(imported);
    let layers = layer_nodes(&native, layer_count)?;
    // Read straight from the native program: a start library on the host held several more copies
    // of the model (Qwen3-0.6B's load went past 24 GiB).
    Weights::from_native(&native, &layers)
}

/// The experiment in reader_score.py's form (its `words` renders it) for one prompt.
fn reader_experiment(e: &Experiment, behavior: &Behavior, prompt: usize, donor: Option<usize>) -> Value {
    let native = |layer: usize, kind: &str, index: Value| json!([{"view": "native", "layer": layer, "kind": kind, "index": index}]);
    match e {
        Experiment::Clean => json!({"kind": "clean"}),
        Experiment::Counterfactual => json!({"kind": "prompt_edit", "clean_text": behavior.prompts[prompt].text}),
        Experiment::Edit { edit, .. } => match edit {
            WeightEdit::Head { layer, head, factor } => json!({"kind": "scale", "pieces": native(*layer, "head", json!(head)), "factor": factor}),
            WeightEdit::Neurons { layer, neurons, factor } => json!({"kind": "scale", "pieces": native(*layer, "mlp", json!(neurons)), "factor": factor}),
            WeightEdit::Subcomponents { layer, down, indices, factor } => json!({"kind": "scale", "pieces": [{"view": "vpd", "layer": layer, "kind": if *down { "down_proj" } else { "c_fc" }, "index": indices}], "factor": factor}),
            WeightEdit::AttnSubcomponents { layer, map, indices, factor } => {
                let kind = ["q_proj", "k_proj", "v_proj", "o_proj"][(*map).min(3)];
                json!({"kind": "scale", "pieces": [{"view": "vpd", "layer": layer, "kind": kind, "index": indices}], "factor": factor})
            }
            WeightEdit::RankOne { layer, head, matrix, .. } => {
                let name = match head {
                    Some(h) => format!("head {h} {matrix:?}").to_lowercase(),
                    None => format!("MLP {matrix:?}").to_lowercase(),
                };
                json!({"kind": "low_rank", "rank": 1, "matrix": name, "layer": layer, "relative_norm": 0.5})
            }
        },
        Experiment::Sites { .. } => {
            let source = behavior.prompts[prompt].counterfactual.as_ref().map_or_else(|| donor.map_or(String::new(), |d| behavior.prompts[d].text.clone()), |c| c.text.clone());
            json!({"words": format!("{} (the counterfactual text: <<<{source}>>>)", e.describe())})
        }
    }
}

/// The reader's items (reader_score.py's format, without the texts, which score.py decodes): per
/// experiment and target token, `M`'s clean top `k` tokens with their clean probabilities, `M_e`'s
/// probabilities of them and of everything else, and the program's.
fn items(checker: &Checker, outcomes: &[Measured]) -> Value {
    let behavior = &checker.behavior;
    let donors = checker.donors();
    let mut list = Vec::new();
    for (x, Measured(e, _, candidates)) in outcomes.iter().enumerate() {
        let Some(c) = candidates else { continue };
        for (r, &(prompt, position)) in checker.rows_of(e).iter().enumerate().take(c.tokens.len()) {
            let donor = donors.iter().find(|(i, _)| *i == prompt).map(|(_, j)| *j);
            let (ids, cut) = match e {
                Experiment::Counterfactual => {
                    let cf = behavior.prompts[prompt].counterfactual.as_ref().map_or(&behavior.prompts[prompt].token_ids, |c| &c.token_ids);
                    let shift = cf.len() as isize - behavior.prompts[prompt].token_ids.len() as isize;
                    (cf, (position as isize + shift) as usize)
                }
                _ => (&behavior.prompts[prompt].token_ids, position),
            };
            let candidates: Vec<Value> = c.tokens[r].iter().enumerate().map(|(j, &t)| json!({"token_id": t, "clean": c.clean[r][j], "p": c.model[r][j], "q_program": c.program[r][j]})).collect();
            list.push(json!({
                "id": format!("{x}:{prompt}:{position}"),
                "family": e.family(),
                "experiment": reader_experiment(e, behavior, prompt, donor),
                "words_checker": e.describe(),
                "prompt": prompt,
                "position": cut,
                "token_ids": ids[..=cut.min(ids.len() - 1)],
                "candidates": candidates,
                "clean_other": c.clean_other[r],
                "other": c.model_other[r],
                "other_program": c.program_other[r],
            }));
        }
    }
    Value::Array(list)
}

/// The export's context in tokens (`export.json`'s `source.context`), what a manifest's positions
/// were drawn on (a manifest stating its own `context` overrides it).
fn context(export: &Path) -> Result<usize, String> {
    let record: Value = serde_json::from_slice(&std::fs::read(export.join("export.json")).map_err(error)?).map_err(error)?;
    record["source"]["context"].as_u64().or_else(|| record["context"].as_u64()).map(|c| c as usize).ok_or_else(|| "export.json states no context".into())
}

/// The first `count` sequences of the export's token table (`tokens.f64`, rows × columns of
/// float64 token ids).
fn export_sequences(export: &Path, count: usize) -> Result<Vec<Vec<u32>>, String> {
    let record: Value = serde_json::from_slice(&std::fs::read(export.join("export.json")).map_err(error)?).map_err(error)?;
    let shape = &record["files"]["tokens"]["shape"];
    let (rows, cols) = (shape[0].as_u64().ok_or("tokens shape")? as usize, shape[1].as_u64().ok_or("tokens shape")? as usize);
    let bytes = std::fs::read(export.join("tokens.f64")).map_err(error)?;
    if bytes.len() != rows * cols * 8 {
        return Err("tokens.f64 disagrees with its shape".into());
    }
    let id = |i: usize| f64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap_or_default()) as u32;
    Ok((0..count.min(rows)).map(|r| (0..cols).map(|c| id(r * cols + c)).collect()).collect())
}

/// The manifest a behavior request names, or by default the export's shared one: VPD-4L's
/// `~/mpd-data/compare/manifest/MANIFEST_vpd4l_s1.json`, any other export's
/// `~/mpd-data/graph_oracle/experiments/MANIFEST_{export directory}_s1.json` (draw_manifest
/// writes it) when it exists.
fn manifest(request: &Value, export: &Path) -> Option<std::path::PathBuf> {
    match request.get("manifest") {
        Some(Value::Null) => None,
        Some(Value::String(path)) => Some(path.into()),
        _ => {
            let home = std::env::home_dir()?;
            let name = export.file_name()?.to_string_lossy().into_owned();
            let default = if name == "vpd4l" { home.join("mpd-data/compare/manifest/MANIFEST_vpd4l_s1.json") } else { home.join(format!("mpd-data/graph_oracle/experiments/MANIFEST_{name}_s1.json")) };
            default.exists().then_some(default)
        }
    }
}

fn handle(request: &Value, weights: &mut Option<Weights>, checker: &mut Option<Checker>, export: &mut Option<std::path::PathBuf>, caches: &Caches) -> Result<Value, String> {
    match request["op"].as_str().ok_or("an op")? {
        "load" => {
            let path = request["export"].as_str().ok_or("export")?;
            *checker = None;
            *export = Some(path.into());
            let mut w = load(Path::new(path))?;
            // Decomposition views, attached when named: "vpd": VPD's decomposition export (its MLP and
            // attention subcomponents, `Weights::attach_vpd`), "transcoders": a directory of
            // layer_{l}.safetensors (`Weights::attach_transcoders`). Pieces of a view that is not
            // attached do not resolve.
            let mut views = serde_json::Map::new();
            if let Some(dir) = request["vpd"].as_str() {
                views.insert("vpd".into(), json!(w.attach_vpd(Path::new(dir))?));
            }
            if let Some(dir) = request["transcoders"].as_str() {
                views.insert("transcoders".into(), json!(w.attach_transcoders(Path::new(dir))?));
            }
            // "library": a start file of components over VPD's slices (decomp's
            // start.components.json), arm "library_arm" (grouped_own when absent), `Weights::attach_library`.
            if let Some(start) = request["library"].as_str() {
                let arm = request["library_arm"].as_str().unwrap_or("grouped_own");
                views.insert("library".into(), json!(w.attach_library(Path::new(start), arm)?));
            }
            // "device": "gpu" runs the executor's large products on the single-precision device
            // (Metal on the Mac, CUDA elsewhere) for the rest of the process.
            let device = match request["device"].as_str() {
                Some("gpu") => {
                    let d = Device::single_precision(gam_gpu::GpuPolicy::Required).map_err(error)?.ok_or("no single-precision device")?;
                    gam_mpd::graph::use_device(d)
                }
                _ => false,
            };
            let answer = json!({"ok": true, "device": device, "views": views, "layers": w.layers.len(), "heads": w.layers.first().map_or(0, |l| l.heads.len()), "neurons": w.layers.first().and_then(|l| l.mlp.as_ref()).map_or(0, |m| m.gate.nrows()), "vocabulary": w.embedding.nrows(), "width": w.embedding.ncols()});
            *weights = Some(w);
            Ok(answer)
        }
        "behavior" => {
            let behavior: Behavior = match request.get("path").and_then(Value::as_str) {
                Some(path) => serde_json::from_slice(&std::fs::read(path).map_err(error)?).map_err(error)?,
                None => serde_json::from_value(request["behavior"].clone()).map_err(error)?,
            };
            // Everything that can fail runs before the model's weights move into the new checker,
            // so a failed request keeps the loaded model.
            let width = weights.as_ref().map(|w| w.embedding.ncols()).or_else(|| checker.as_ref().map(|c| c.weights.embedding.ncols())).ok_or("load a model first")?;
            let dir = export.as_deref().ok_or("load a model first")?;
            let named = manifest(request, dir);
            // A manifest stating its own context (draw_manifest's) needs none from the export (0).
            let units = named.as_ref().map(|path| SiteUnits::manifest(path, context(dir).unwrap_or(0), width)).transpose()?;
            let w = match (weights.take(), checker.take()) {
                (Some(w), _) => w,
                (None, Some(c)) => c.weights,
                (None, None) => return Err("load a model first".into()),
            };
            let (id, prompts) = (behavior.id.clone(), behavior.prompts.len());
            let mut c = Checker::new(w, behavior)?;
            if let Some(bytes) = caches.bytes {
                c.cache_bytes = bytes;
            }
            c.disk_cache = caches.disk.clone();
            if let Some(u) = units {
                c.sites = u;
            }
            let pool = c.sites.pool.len();
            *checker = Some(c);
            Ok(json!({"ok": true, "id": id, "prompts": prompts, "manifest": named.map(|p| p.display().to_string()), "site_experiments": pool}))
        }
        // One program ("program") or many ("programs", the batch answer {"ok", "scores": [...]}),
        // every program under the same seed: the behavior's half of the experiments is shared and
        // M runs once per experiment; runs go in parallel threads (RAYON_NUM_THREADS).
        "score" | "score_batch" => {
            let c = checker.as_mut().ok_or("load a behavior first")?;
            let batch = request.get("programs").is_some();
            let programs: Vec<Program> = if batch { serde_json::from_value(request["programs"].clone()).map_err(error)? } else { vec![serde_json::from_value(request["program"].clone()).map_err(error)?] };
            let count = request["experiments"].as_u64().unwrap_or(32) as usize;
            let seed = request["seed"].as_u64().unwrap_or(0);
            let edges = request["routing"].as_str().unwrap_or("edges") == "edges";
            let n = request["N"].as_f64();
            // "uniform_seeds": m draws from seed mod m (m collections of experiments recur across
            // seeds, so M's cached outcomes serve them).
            c.uniform_seeds = request["uniform_seeds"].as_u64();
            let k = request["reader_top"].as_u64().unwrap_or(0) as usize;
            let started = std::time::Instant::now();
            let scored = c.score_batch(&programs, count, seed, edges, n, k)?;
            let seconds = started.elapsed().as_secs_f64();
            let mut answers = Vec::with_capacity(scored.len());
            for (score, outcomes) in &scored {
                let mut answer = serde_json::to_value(score).map_err(error)?;
                if k > 0 {
                    answer["items"] = items(c, outcomes);
                }
                answers.push(answer);
            }
            if batch {
                Ok(json!({"ok": true, "scores": answers, "seconds": seconds}))
            } else {
                let mut answer = answers.pop().ok_or("no score")?;
                answer["seconds"] = json!(seconds);
                Ok(answer)
            }
        }
        // An immutable manifest of site operations for a model without one, drawn by interchange's
        // code (graph::SiteUnits::write_manifest) on the export's first `sequences` token rows, and
        // made the behavior's pool: {"out": FILE, "seed": 1, "count": 1024, "length": 512,
        // "sequences": 8, "families": ["swap", "zero", "scale", "push", "cut"]}.
        "draw_manifest" => {
            let c = checker.as_mut().ok_or("load a behavior first (its stand-in averages run the model)")?;
            let dir = export.as_deref().ok_or("load a model first")?;
            let out = Path::new(request["out"].as_str().ok_or("out")?);
            let seed = request["seed"].as_u64().unwrap_or(1);
            let count = request["count"].as_u64().unwrap_or(1024) as usize;
            let length = request["length"].as_u64().unwrap_or(512) as usize;
            let sequences = export_sequences(dir, request["sequences"].as_u64().unwrap_or(8) as usize)?;
            let families: Vec<gam_mpd::interchange::Family> = match request.get("families") {
                Some(f) if !f.is_null() => serde_json::from_value(f.clone()).map_err(error)?,
                _ => serde_json::from_value(json!(["swap", "zero", "scale", "push", "cut"])).map_err(error)?,
            };
            let identity = gam_mpd::engine::sha256(&dir.join("export.json"))?;
            c.sites = SiteUnits::write_manifest(out, &identity, &c.weights, &sequences, &families, count, seed, length)?;
            Ok(json!({"ok": true, "manifest": out.display().to_string(), "site_experiments": c.sites.pool.len(), "typical_sites": c.sites.typical.len()}))
        }
        other => Err(format!("unknown op {other}")),
    }
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let caches = Caches::parse(std::env::args().skip(1))?;
    let (mut weights, mut checker, mut export) = (None, None, None);
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line.map_err(error)?;
        if line.trim().is_empty() {
            continue;
        }
        let answer = match serde_json::from_str::<Value>(&line) {
            Ok(request) if request["op"] == "quit" => break,
            Ok(request) => handle(&request, &mut weights, &mut checker, &mut export, &caches).unwrap_or_else(|e| json!({"ok": false, "error": e})),
            Err(e) => json!({"ok": false, "error": e.to_string()}),
        };
        writeln!(stdout, "{answer}").map_err(error)?;
        stdout.flush().map_err(error)?;
    }
    Ok(())
}

//! The graph oracle's checker as a JSON-lines server (#2951): one request per stdin line, one JSON
//! answer per stdout line.
//!
//! * `{"op": "load", "export": DIR}`: the native model of an export (`import::import_language_model`),
//!   its weights taken from the start library (`library_mdl::explanation`, equal to `M`).
//! * `{"op": "behavior", "path": FILE}` or `{"op": "behavior", "behavior": {...}}`: a behavior file
//!   (design.txt section 5); measures its stand-in averages on `M`.
//! * `{"op": "score", "program": IR, "experiments": 32, "seed": 0, "routing": "edges" | "nodes",
//!   "N": null, "reader_top": 0}`: every score term (`graph::Score`); with `reader_top` k > 0, per
//!   experiment its words and per target token `M_e`'s and the program's probabilities of `M`'s k
//!   most probable clean tokens and of everything else (the reader's items).
//! * `{"op": "quit"}`.
use gam_gpu::tensor::Device;
use gam_mpd::{
    engine::log_to_stderr,
    graph::{Behavior, Checker, Experiment, Graph, Program, WeightEdit, Weights, Writer},
    import::import_language_model,
    library_mdl,
    library_readout::Library,
    run_check::{layer_nodes, split_sites},
};
use ndarray::Array2;
use serde_json::{Value, json};
use std::io::{BufRead, Write};
use std::path::Path;

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn load(export: &Path) -> Result<Weights, String> {
    let imported = import_language_model(export, 1, 1)?;
    let layer_count = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
    let native = split_sites(&imported.program)?;
    drop(imported);
    let layers = layer_nodes(&native, layer_count)?;
    let artifact = library_mdl::explanation(&native, &layers)?.artifact;
    let device = Device::host();
    let library = Library::new(&device, &device, &native, &layers, &artifact, 1 << 30, 256)?;
    Ok(Weights::of(&library))
}

/// A node's or writer's pieces in IR form (reader_score.py renders them), or "embed" / "logits".
fn pieces(program: &Program, graph: &Graph, unit: Option<usize>, end: &str) -> Value {
    match unit {
        None => json!(end),
        Some(u) => program.nodes.iter().find(|n| graph.ids.get(u) == Some(&n.id)).map_or(Value::Null, |n| json!(n.pieces)),
    }
}

/// The experiment in reader_score.py's form (its `words` renders it) for one prompt.
fn reader_experiment(e: &Experiment, program: &Program, graph: &Graph, behavior: &Behavior, prompt: usize, donor: Option<usize>) -> Value {
    let native = |layer: usize, kind: &str, index: Value| json!([{"view": "native", "layer": layer, "kind": kind, "index": index}]);
    match e {
        Experiment::Clean => json!({"kind": "clean"}),
        Experiment::Counterfactual => json!({"kind": "prompt_edit", "clean_text": behavior.prompts[prompt].text}),
        Experiment::Edit { edit, .. } => match edit {
            WeightEdit::Head { layer, head, factor } => json!({"kind": "scale", "pieces": native(*layer, "head", json!(head)), "factor": factor}),
            WeightEdit::Neurons { layer, neurons, factor } => json!({"kind": "scale", "pieces": native(*layer, "mlp", json!(neurons)), "factor": factor}),
            WeightEdit::RankOne { layer, head, matrix, .. } => {
                let name = match head {
                    Some(h) => format!("head {h} {matrix:?}").to_lowercase(),
                    None => format!("MLP {matrix:?}").to_lowercase(),
                };
                json!({"kind": "low_rank", "rank": 1, "matrix": name, "layer": layer, "relative_norm": 0.5})
            }
        },
        Experiment::Swap { node } => json!({"kind": "swap", "pieces": pieces(program, graph, Some(*node), ""), "source_text": donor.map_or(String::new(), |d| behavior.prompts[d].text.clone())}),
        Experiment::Cut { from, to, route, .. } => {
            let from = match from {
                Writer::Embed => json!("embed"),
                Writer::Unit(u) => pieces(program, graph, Some(*u), ""),
            };
            json!({"kind": "cut", "from": from, "to": pieces(program, graph, *to, "logits"), "route": route})
        }
    }
}

/// The reader's items (reader_score.py's format, without the texts, which score.py decodes): per
/// experiment and target token, `M`'s clean top `k` tokens with their clean probabilities, `M_e`'s
/// probabilities of them and of everything else, and the program's.
fn items(checker: &Checker, outcomes: &[(Experiment, Array2<f64>, Array2<f64>)], program: &Program, k: usize) -> Value {
    let graph = checker.graph(program);
    let Some((_, clean, _)) = outcomes.iter().find(|o| o.0 == Experiment::Clean) else { return Value::Null };
    let top: Vec<Vec<usize>> = clean
        .outer_iter()
        .map(|row| {
            let mut order: Vec<usize> = (0..row.len()).collect();
            order.sort_by(|a, b| row[*b].total_cmp(&row[*a]));
            order.truncate(k);
            order
        })
        .collect();
    let behavior = &checker.behavior;
    // Rows are the behavior's target tokens in prompt order (a swap's: its prompts' only).
    let targets: Vec<(usize, usize)> = behavior.prompts.iter().enumerate().flat_map(|(i, p)| p.target_positions.iter().map(move |&t| (i, t))).collect();
    let swapped = checker.swap_targets();
    let donors = checker.donors();
    let mut list = Vec::new();
    for (x, (e, m, p)) in outcomes.iter().enumerate() {
        let rows: &[(usize, usize)] = if matches!(e, Experiment::Swap { .. }) { &swapped } else { &targets };
        for (r, &(prompt, position)) in rows.iter().enumerate().filter(|(r, _)| *r < m.nrows()) {
            let clean_row = targets.iter().position(|t| *t == (prompt, position)).unwrap_or(r);
            let tokens = &top[clean_row];
            let donor = donors.iter().find(|(i, _)| *i == prompt).map(|(_, j)| *j);
            let (ids, cut) = match e {
                Experiment::Counterfactual => {
                    let c = behavior.prompts[prompt].counterfactual.as_ref().map_or(&behavior.prompts[prompt].token_ids, |c| &c.token_ids);
                    let shift = c.len() as isize - behavior.prompts[prompt].token_ids.len() as isize;
                    (c, (position as isize + shift) as usize)
                }
                _ => (&behavior.prompts[prompt].token_ids, position),
            };
            let candidates: Vec<Value> = tokens.iter().map(|&t| json!({"token_id": t, "clean": clean[[clean_row, t]].exp(), "p": m[[r, t]].exp(), "q_program": p[[r, t]].exp()})).collect();
            let rest = |row: ndarray::ArrayView1<f64>| (1.0 - tokens.iter().map(|&t| row[t].exp()).sum::<f64>()).max(0.0);
            list.push(json!({
                "id": format!("{x}:{prompt}:{position}"),
                "family": e.family(),
                "experiment": reader_experiment(e, program, &graph, behavior, prompt, donor),
                "words_checker": e.describe(&graph),
                "prompt": prompt,
                "position": cut,
                "token_ids": ids[..=cut.min(ids.len() - 1)],
                "candidates": candidates,
                "clean_other": rest(clean.row(clean_row)),
                "other": rest(m.row(r)),
                "other_program": rest(p.row(r)),
            }));
        }
    }
    Value::Array(list)
}

fn handle(request: &Value, weights: &mut Option<Weights>, checker: &mut Option<Checker>) -> Result<Value, String> {
    match request["op"].as_str().ok_or("an op")? {
        "load" => {
            let export = request["export"].as_str().ok_or("export")?;
            *checker = None;
            let w = load(Path::new(export))?;
            let answer = json!({"ok": true, "layers": w.layers.len(), "heads": w.layers.first().map_or(0, |l| l.heads.len()), "neurons": w.layers.first().and_then(|l| l.mlp.as_ref()).map_or(0, |m| m.gate.nrows()), "vocabulary": w.embedding.nrows(), "width": w.embedding.ncols()});
            *weights = Some(w);
            Ok(answer)
        }
        "behavior" => {
            let behavior: Behavior = match request.get("path").and_then(Value::as_str) {
                Some(path) => serde_json::from_slice(&std::fs::read(path).map_err(error)?).map_err(error)?,
                None => serde_json::from_value(request["behavior"].clone()).map_err(error)?,
            };
            let w = match (weights.take(), checker.take()) {
                (Some(w), _) => w,
                (None, Some(c)) => c.weights,
                (None, None) => return Err("load a model first".into()),
            };
            let (id, prompts) = (behavior.id.clone(), behavior.prompts.len());
            *checker = Some(Checker::new(w, behavior)?);
            Ok(json!({"ok": true, "id": id, "prompts": prompts}))
        }
        "score" => {
            let c = checker.as_mut().ok_or("load a behavior first")?;
            let program: Program = serde_json::from_value(request["program"].clone()).map_err(error)?;
            let count = request["experiments"].as_u64().unwrap_or(32) as usize;
            let seed = request["seed"].as_u64().unwrap_or(0);
            let edges = request["routing"].as_str().unwrap_or("edges") == "edges";
            let n = request["N"].as_f64();
            let (score, outcomes) = c.score(&program, count, seed, edges, n)?;
            let mut answer = serde_json::to_value(&score).map_err(error)?;
            let k = request["reader_top"].as_u64().unwrap_or(0) as usize;
            if k > 0 {
                answer["items"] = items(c, &outcomes, &program, k);
            }
            Ok(answer)
        }
        other => Err(format!("unknown op {other}")),
    }
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let (mut weights, mut checker) = (None, None);
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line.map_err(error)?;
        if line.trim().is_empty() {
            continue;
        }
        let answer = match serde_json::from_str::<Value>(&line) {
            Ok(request) if request["op"] == "quit" => break,
            Ok(request) => handle(&request, &mut weights, &mut checker).unwrap_or_else(|e| json!({"ok": false, "error": e})),
            Err(e) => json!({"ok": false, "error": e.to_string()}),
        };
        writeln!(stdout, "{answer}").map_err(error)?;
        stdout.flush().map_err(error)?;
    }
    Ok(())
}

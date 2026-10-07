//! The graph checker against the library's own runs of `M` on the tiny export (#2951).
use crate::{
    graph::{Batch, Behavior, Checker, EdgeIr, Experiment, Graph, Index, NodeIr, PieceIr, Program, Prompt, Stats, WeightEdit, Weights, execute, kl_bits, reference},
    import::import_language_model,
    library_mdl,
    library_readout::{Activity, Edit, Library},
    operator_program::SlotValues,
    run_check::{layer_nodes, split_sites},
    test_support::tiny_export,
};
use gam_gpu::tensor::Device;
use std::collections::BTreeMap;

const LAYERS: usize = 2;

struct Fixture {
    device: Device,
    native: crate::operator_program::OperatorProgram,
    layers: Vec<crate::run_check::LayerNodes>,
    artifact: crate::artifact::Artifact,
    sequences: Vec<Vec<u32>>,
}

fn fixture(tag: &str) -> Fixture {
    let dir = tiny_export(tag, LAYERS);
    let imported = import_language_model(&dir, 6, 12).expect("import");
    let native = split_sites(&imported.program).expect("split");
    let layers = layer_nodes(&native, LAYERS).expect("layers");
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
    let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
    let artifact = library_mdl::explanation(&native, &layers).expect("start library").artifact;
    Fixture { device: Device::host(), native, layers, artifact, sequences }
}

fn piece(layer: usize, kind: &str) -> PieceIr {
    PieceIr { view: "native".into(), layer, kind: kind.into(), index: None }
}

/// Every piece declared (per layer its heads and its MLP) and every edge kept.
fn full_program() -> Program {
    let mut nodes = Vec::new();
    for l in 0..LAYERS {
        nodes.push(NodeIr { id: format!("a{l}"), pieces: vec![piece(l, "head")], rule: None });
        nodes.push(NodeIr { id: format!("m{l}"), pieces: vec![piece(l, "mlp")], rule: None });
    }
    let order: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
    let mut edges = Vec::new();
    let reader_routes = |id: &str| -> Vec<&'static str> { if id.starts_with('a') { vec!["query", "key", "value"] } else { vec!["input"] } };
    for (r, reader) in order.iter().chain([&"logits".to_string()]).enumerate() {
        let writers: Vec<String> = ["embed".to_string()].into_iter().chain(order.iter().take(r).cloned()).collect();
        for w in writers {
            for route in reader_routes(reader) {
                edges.push(EdgeIr { from: w.clone(), to: reader.clone(), route: route.into() });
            }
        }
    }
    Program { model: "tiny".into(), nodes, edges, python_tokens: 0, token_types: 0, source: String::new(), valid: true, error: None, standin: None }
}

fn max(values: &[f64]) -> f64 {
    values.iter().fold(0.0f64, |a, b| a.max(b.abs()))
}

#[test]
fn full_graph_with_every_edge_is_the_model() {
    let f = fixture("graph_full");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let weights = Weights::of(&library);
    let reference = library.log_probabilities(&library.run(&f.sequences, &BTreeMap::new()).expect("run").last).expect("log p");
    let batch = Batch::new(&f.sequences).expect("batch");
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    let stats = Stats::measure(&weights, &f.sequences).expect("stats");
    let graph = Graph::parse(&full_program(), &weights).expect("parse");
    for (name, circuit) in [("edges", graph.program(&weights, true)), ("nodes", graph.program(&weights, false)), ("model", graph.model(&weights))] {
        let run = execute(&weights, &stats, &circuit, &batch, &rows, &BTreeMap::new(), false).expect("execute");
        let kl = max(&kl_bits(&reference, &run.log_probabilities));
        assert!(kl < 1e-9, "{name}: KL(M ‖ graph) = {kl:e} bits");
    }
}

#[test]
fn head_removal_matches_the_library_edit() {
    let f = fixture("graph_removal");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let mut weights = Weights::of(&library);
    // Library functions: per layer its heads, then its MLP's functions; head (1, 0).
    let (heads, neurons) = (weights.layers[0].heads.len(), weights.layers[0].mlp.as_ref().expect("an MLP").gate.nrows());
    let function = heads + neurons;
    let length = f.sequences[0].len();
    let edits: Vec<Edit> = (0..f.sequences.len()).map(|s| Edit { sequence: s, scale: vec![(function, 0.0)], add: Vec::new(), rows: (0..length).collect() }).collect();
    let edited = library.edited(&f.sequences, &edits, &Activity::None, false, false).expect("edited");
    let reference = library.log_probabilities(&edited.last).expect("log p");
    let batch = Batch::new(&f.sequences).expect("batch");
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    let stats = Stats::measure(&weights, &f.sequences).expect("stats");
    let restore = WeightEdit::Head { layer: 1, head: 0, factor: 0.0 }.apply(&mut weights).expect("edit");
    let circuit = Graph::empty().model(&weights);
    let run = execute(&weights, &stats, &circuit, &batch, &rows, &BTreeMap::new(), false).expect("execute");
    restore.restore(&mut weights).expect("restore");
    let kl = max(&kl_bits(&reference, &run.log_probabilities));
    assert!(kl < 1e-9, "KL(library edit ‖ graph edit) = {kl:e} bits");
}

#[test]
fn empty_graph_is_every_stand_in() {
    let f = fixture("graph_empty");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let weights = Weights::of(&library);
    let batch = Batch::new(&f.sequences).expect("batch");
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    let stats = Stats::measure(&weights, &f.sequences).expect("stats");
    let run = execute(&weights, &stats, &Graph::empty().program(&weights, true), &batch, &rows, &BTreeMap::new(), false).expect("execute");
    let first = run.log_probabilities.row(0).to_owned();
    let spread = run.log_probabilities.outer_iter().map(|r| (&r - &first).iter().fold(0.0f64, |a, b| a.max(b.abs()))).fold(0.0f64, f64::max);
    assert!(spread < 1e-12, "the empty graph's distribution varies by {spread:e} across tokens");
}

#[test]
fn parse_rejects_bad_programs() {
    let f = fixture("graph_parse");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let weights = Weights::of(&library);
    let node = |id: &str, layer: usize, kind: &str, index: Option<Index>| NodeIr { id: id.into(), pieces: vec![PieceIr { view: "native".into(), layer, kind: kind.into(), index }], rule: None };
    let mut program = Program { model: "tiny".into(), valid: true, ..Program::default() };
    program.nodes = vec![node("a", 0, "head", Some(Index::One(1))), node("b", 0, "head", Some(Index::Many(vec![0, 1])))];
    assert!(Graph::parse(&program, &weights).is_err(), "a head in two nodes");
    program.nodes = vec![node("a", 1, "head", Some(Index::One(0))), node("b", 0, "mlp", Some(Index::One(3)))];
    program.edges = vec![EdgeIr { from: "a".into(), to: "b".into(), route: "input".into() }];
    assert!(Graph::parse(&program, &weights).is_err(), "an edge backwards in depth");
    program.edges = vec![EdgeIr { from: "b".into(), to: "a".into(), route: "input".into() }];
    assert_eq!(Graph::parse(&program, &weights).expect("all of a head's inputs").edges.len(), 3);
    program.edges = vec![EdgeIr { from: "b".into(), to: "a".into(), route: "gate".into() }];
    assert!(Graph::parse(&program, &weights).is_err(), "no such route");
    program.edges = vec![EdgeIr { from: "b".into(), to: "a".into(), route: "key".into() }];
    assert!(Graph::parse(&program, &weights).is_ok());
}

#[test]
fn checker_scores_the_full_program_at_zero_error() {
    let f = fixture("graph_score");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let weights = Weights::of(&library);
    let prompts = f.sequences.iter().map(|s| Prompt { text: String::new(), token_ids: s.clone(), target_positions: vec![s.len() - 2, s.len() - 1], counterfactual: None }).collect();
    let behavior = Behavior { id: "tiny".into(), model: "tiny".into(), family: String::new(), description: String::new(), frequency: None, prompts, split: "train".into(), model_accuracy: None };
    let mut checker = Checker::new(weights, behavior).expect("checker");
    // These prompts have no counterfactuals: average stand-ins.
    let global = |mut p: Program| {
        p.standin = Some("global".into());
        p
    };
    let (full, outcomes) = checker.score(&global(full_program()), 24, 7, true, None).expect("score");
    assert!(full.valid);
    assert!(full.exec_error_bits / full.n < 1e-9, "full program error {:e} bits per token over {:?}", full.exec_error_bits / full.n, outcomes.iter().map(|o| o.0.family()).collect::<Vec<_>>());
    assert!(outcomes.iter().any(|o| matches!(o.0, Experiment::Edit { .. })));
    let (empty, _) = checker.score(&global(Program { model: "tiny".into(), valid: true, ..Program::default() }), 24, 7, true, None).expect("score");
    assert!(empty.exec_error_bits > full.exec_error_bits && empty.opaque_numbers < full.opaque_numbers);
}

/// Each sequence with one token changed (its counterfactual).
fn counterfactuals(sequences: &[Vec<u32>]) -> Vec<Vec<u32>> {
    sequences
        .iter()
        .map(|s| {
            let mut c = s.clone();
            c[5] = (c[5] + 1) % 11;
            c
        })
        .collect()
}

#[test]
fn counterfactual_empty_graph_is_the_model_on_the_counterfactual() {
    let f = fixture("graph_cf_empty");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let weights = Weights::of(&library);
    let cf = counterfactuals(&f.sequences);
    let target = library.log_probabilities(&library.run(&cf, &BTreeMap::new()).expect("run").last).expect("log p");
    let stats = Stats::measure(&weights, &f.sequences).expect("stats");
    let mut batch = Batch::new(&f.sequences).expect("batch");
    batch.reference = Some(std::sync::Arc::new(reference(&weights, &stats, &Batch::new(&cf).expect("cf batch")).expect("reference")));
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    for (name, circuit) in [("empty", Graph::empty().program(&weights, true)), ("full", Graph::parse(&full_program(), &weights).expect("parse").program(&weights, true))] {
        let run = execute(&weights, &stats, &circuit, &batch, &rows, &BTreeMap::new(), false).expect("execute");
        let expected = if name == "empty" { target.clone() } else { library.log_probabilities(&library.run(&f.sequences, &BTreeMap::new()).expect("run").last).expect("log p") };
        let kl = max(&kl_bits(&expected, &run.log_probabilities));
        assert!(kl < 1e-9, "{name}: KL = {kl:e} bits");
    }
}

#[test]
fn counterfactual_undeclared_head_is_patched_from_the_counterfactual() {
    let f = fixture("graph_cf_patch");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let weights = Weights::of(&library);
    let cf = counterfactuals(&f.sequences);
    // Head (1, 0) (library head 2) reads its counterfactual read; everything else runs on the prompt.
    let z = library.run(&cf, &BTreeMap::new()).expect("cf run").reads[2].clone();
    let expected = library.log_probabilities(&library.run(&f.sequences, &[(2, z)].into()).expect("patched").last).expect("log p");
    let node = |id: &str, layer: usize, kind: &str, index: Option<Index>| NodeIr { id: id.into(), pieces: vec![PieceIr { view: "native".into(), layer, kind: kind.into(), index }], rule: None };
    let program = Program {
        model: "tiny".into(),
        valid: true,
        nodes: vec![node("a0", 0, "head", None), node("m0", 0, "mlp", None), node("a1", 1, "head", Some(Index::One(1))), node("m1", 1, "mlp", Some(Index::Many((0..16).collect())))],
        ..Program::default()
    };
    let stats = Stats::measure(&weights, &f.sequences).expect("stats");
    let mut batch = Batch::new(&f.sequences).expect("batch");
    batch.reference = Some(std::sync::Arc::new(reference(&weights, &stats, &Batch::new(&cf).expect("cf batch")).expect("reference")));
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    let circuit = Graph::parse(&program, &weights).expect("parse").program(&weights, false);
    let run = execute(&weights, &stats, &circuit, &batch, &rows, &BTreeMap::new(), false).expect("execute");
    let kl = max(&kl_bits(&expected, &run.log_probabilities));
    assert!(kl < 1e-9, "KL(activation patch ‖ graph) = {kl:e} bits");
    // A few neurons left out of layer 1's MLP: the complement is assembled by subtraction.
    let mut partial = program.clone();
    partial.nodes[3] = node("m1", 1, "mlp", Some(Index::Many((0..13).collect())));
    let circuit = Graph::parse(&partial, &weights).expect("parse").program(&weights, false);
    let left = execute(&weights, &stats, &circuit, &batch, &rows, &BTreeMap::new(), false).expect("execute");
    assert!(max(&kl_bits(&expected, &left.log_probabilities)) > 1e-12, "leaving neurons out changes nothing");
}

#[test]
fn checker_counterfactual_default_scores_the_empty_program_at_the_behavior_signal() {
    let f = fixture("graph_cf_score");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let weights = Weights::of(&library);
    let cf = counterfactuals(&f.sequences);
    let clean = library.log_probabilities(&library.run(&f.sequences, &BTreeMap::new()).expect("run").last).expect("log p");
    let target = library.log_probabilities(&library.run(&cf, &BTreeMap::new()).expect("run").last).expect("log p");
    let prompts: Vec<Prompt> = f
        .sequences
        .iter()
        .zip(&cf)
        .map(|(s, c)| Prompt { text: String::new(), token_ids: s.clone(), target_positions: vec![s.len() - 1], counterfactual: Some(crate::graph::Counterfactual { text: String::new(), token_ids: c.clone() }) })
        .collect();
    let behavior = Behavior { id: "tiny".into(), model: "tiny".into(), family: String::new(), description: String::new(), frequency: None, prompts, split: "train".into(), model_accuracy: None };
    let mut checker = Checker::new(weights, behavior).expect("checker");
    let (empty, outcomes) = checker.score(&Program { model: "tiny".into(), valid: true, ..Program::default() }, 0, 3, true, None).expect("score");
    let clean_rows: Vec<usize> = (0..f.sequences.len()).map(|s| (s + 1) * 12 - 1).collect();
    let signal = kl_bits(&clean.select(ndarray::Axis(0), &clean_rows), &target.select(ndarray::Axis(0), &clean_rows));
    let measured = outcomes.iter().find(|o| o.0 == Experiment::Clean).map(|o| o.1.clone()).expect("clean");
    assert!(max(&signal.iter().zip(&measured).map(|(a, b)| a - b).collect::<Vec<_>>()) < 1e-9, "empty program clean error {measured:?} vs KL(M(x) ‖ M(x')) {signal:?}");
    assert!(empty.valid);
    let (full, _) = checker.score(&full_program(), 16, 3, true, None).expect("score");
    assert!(full.exec_error_bits / full.n < 1e-9, "full program error {:e} bits per token", full.exec_error_bits / full.n);
}

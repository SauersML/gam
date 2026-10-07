//! The graph checker against the library's own runs of `M` on the tiny export (#2951).
use crate::{
    graph::{Batch, Behavior, Checker, EdgeIr, Experiment, Graph, Index, NodeIr, PieceIr, Program, Prompt, Stats, WeightEdit, Weights, execute, kl_bits},
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
    Program { model: "tiny".into(), nodes, edges, python_tokens: 0, token_types: 0, source: String::new(), valid: true, error: None }
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
    let (full, outcomes) = checker.score(&full_program(), 24, 7, true, None).expect("score");
    assert!(full.valid);
    assert!(full.exec_error_bits / full.n < 1e-9, "full program error {:e} bits per token over {:?}", full.exec_error_bits / full.n, outcomes.iter().map(|o| o.0.family()).collect::<Vec<_>>());
    assert!(outcomes.iter().any(|o| matches!(o.0, Experiment::Edit { .. })));
    let (empty, _) = checker.score(&Program { model: "tiny".into(), valid: true, ..Program::default() }, 24, 7, true, None).expect("score");
    assert!(empty.exec_error_bits > full.exec_error_bits && empty.opaque_numbers < full.opaque_numbers);
}

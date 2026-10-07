#![cfg(test)]
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
    fixture_of(tiny_export(tag, LAYERS))
}

fn fixture_of(dir: std::path::PathBuf) -> Fixture {
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

/// The Checker keeps `M`'s log-probabilities in float32 (relative rounding `2^-24`): on
/// log-probabilities above −64 a token's KL moves by at most `2^-24 · 64 / ln 2` bits.
const F32_KL: f64 = 64.0 / 16_777_216.0 / std::f64::consts::LN_2;

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
    let prompts = f.sequences.iter().map(|s| Prompt { text: String::new(), token_ids: s.clone(), target_positions: vec![s.len() - 2, s.len() - 1], counterfactual: None, attention_block: Vec::new() }).collect();
    let behavior = Behavior { id: "tiny".into(), model: "tiny".into(), family: String::new(), description: String::new(), frequency: None, prompts, split: "train".into(), model_accuracy: None };
    let mut checker = Checker::new(weights, behavior).expect("checker");
    // These prompts have no counterfactuals: average stand-ins.
    let global = |mut p: Program| {
        p.standin = Some("global".into());
        p
    };
    let (full, outcomes) = checker.score(&global(full_program()), 24, 7, true, None).expect("score");
    assert!(full.valid);
    assert!(full.exec_error_bits / full.n < F32_KL, "full program error {:e} bits per token over {:?}", full.exec_error_bits / full.n, outcomes.iter().map(|o| o.0.family()).collect::<Vec<_>>());
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
        .map(|(s, c)| Prompt { text: String::new(), token_ids: s.clone(), target_positions: vec![s.len() - 1], counterfactual: Some(crate::graph::Counterfactual { text: String::new(), token_ids: c.clone() }), attention_block: Vec::new() })
        .collect();
    let behavior = Behavior { id: "tiny".into(), model: "tiny".into(), family: String::new(), description: String::new(), frequency: None, prompts, split: "train".into(), model_accuracy: None };
    let mut checker = Checker::new(weights, behavior).expect("checker");
    let (empty, outcomes) = checker.score(&Program { model: "tiny".into(), valid: true, ..Program::default() }, 0, 3, true, None).expect("score");
    let clean_rows: Vec<usize> = (0..f.sequences.len()).map(|s| (s + 1) * 12 - 1).collect();
    let signal = kl_bits(&clean.select(ndarray::Axis(0), &clean_rows), &target.select(ndarray::Axis(0), &clean_rows));
    let measured = outcomes.iter().find(|o| o.0 == Experiment::Clean).map(|o| o.1.clone()).expect("clean");
    assert!(max(&signal.iter().zip(&measured).map(|(a, b)| a - b).collect::<Vec<_>>()) < F32_KL, "empty program clean error {measured:?} vs KL(M(x) ‖ M(x')) {signal:?}");
    assert!(empty.valid);
    let (full, _) = checker.score(&full_program(), 16, 3, true, None).expect("score");
    assert!(full.exec_error_bits / full.n < F32_KL, "full program error {:e} bits per token", full.exec_error_bits / full.n);
}

#[test]
fn transcoder_features_write_the_transcoder_and_the_rest_is_exact() {
    let f = fixture("graph_transcoder");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let mut weights = Weights::of(&library);
    let dir = std::env::temp_dir().join(format!("gam_mpd_graph_tc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    crate::test_support::transcoder_file(&dir.join("layer_1.safetensors"), 6, 8, 5);
    assert_eq!(weights.attach_transcoders(&dir).expect("attach"), 1);
    let feature = |id: &str, index: Index| NodeIr { id: id.into(), pieces: vec![PieceIr { view: "transcoder".into(), layer: 1, kind: "feature".into(), index: Some(index) }], rule: None };
    let program = Program { model: "tiny".into(), valid: true, nodes: vec![feature("f", Index::Many((0..6).collect()))], ..Program::default() };
    let graph = Graph::parse(&program, &weights).expect("parse");
    weights.load_features(&graph).expect("features");
    let stats = Stats::measure(&weights, &f.sequences).expect("stats");
    let batch = Batch::new(&f.sequences).expect("batch");
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    // M with the features as a node: the features plus the rest (other features and the error) is M.
    let run = execute(&weights, &stats, &graph.model(&weights), &batch, &rows, &BTreeMap::new(), false).expect("execute");
    let reference_p = library.log_probabilities(&library.run(&f.sequences, &BTreeMap::new()).expect("run").last).expect("log p");
    assert!(max(&kl_bits(&reference_p, &run.log_probabilities)) < 1e-9);
    // The node's write is the transcoder's output on layer 1's normed MLP input, less its decoder bias.
    let x = library.run(&f.sequences, &BTreeMap::new()).expect("run").streams[3].clone();
    let norm = &weights.layers[1].mlp_norm;
    let mut x_hat = x.clone();
    for mut row in x_hat.outer_iter_mut() {
        let r = crate::operator_program::rms_scale(row.view(), norm.epsilon);
        row.zip_mut_with(&norm.gain, |v, g| *v *= r * g);
    }
    let t = crate::library_transcoder::Transcoder::open(&dir.join("layer_1.safetensors")).expect("transcoder");
    let file = crate::safetensors::SafetensorsFile::open(&dir.join("layer_1.safetensors")).expect("file");
    let b = file.vector("b_dec", 8).expect("b_dec");
    let expected = t.reconstruction(&x_hat).expect("reconstruction") - &b.view().insert_axis(ndarray::Axis(0));
    let got = run.writes[0].clone().expect("the node computes");
    let gap = (&got - &expected).iter().fold(0.0f64, |a, v| a.max(v.abs()));
    assert!(gap < 1e-9, "feature writes differ from the transcoder by {gap:e}");
    // One view per site; features in range.
    let mut both = program.clone();
    both.nodes.push(NodeIr { id: "n".into(), pieces: vec![PieceIr { view: "native".into(), layer: 1, kind: "mlp".into(), index: Some(Index::One(0)) }], rule: None });
    assert!(Graph::parse(&both, &weights).is_err(), "neurons and features of one MLP");
    let far = Program { nodes: vec![feature("f", Index::One(6))], ..program.clone() };
    assert!(Graph::parse(&far, &weights).is_err(), "feature out of range");
}

#[test]
fn attention_block_hides_the_prefix_like_running_the_suffix_alone() {
    let f = fixture("graph_mask");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let weights = Weights::of(&library);
    let stats = Stats::measure(&weights, &f.sequences).expect("stats");
    // Queries from position 4 on do not see keys before 4: with rotary (relative) positions and no
    // absolute ones, positions 4.. then compute exactly what the suffix computes alone.
    let (t, length) = (4, f.sequences[0].len());
    let mut batch = Batch::new(&f.sequences).expect("batch");
    batch.blocks = vec![vec![[t, length, 0, t]]; f.sequences.len()];
    let rows: Vec<usize> = (0..f.sequences.len()).flat_map(|s| (t..length).map(move |p| s * length + p)).collect();
    let run = execute(&weights, &stats, &Graph::empty().model(&weights), &batch, &rows, &BTreeMap::new(), false).expect("execute");
    let suffixes: Vec<Vec<u32>> = f.sequences.iter().map(|s| s[t..].to_vec()).collect();
    let expected = library.log_probabilities(&library.run(&suffixes, &BTreeMap::new()).expect("run").last).expect("log p");
    let kl = max(&kl_bits(&expected, &run.log_probabilities));
    assert!(kl < 1e-9, "KL(suffix alone ‖ masked prefix) = {kl:e} bits");
    let open = execute(&weights, &stats, &Graph::empty().model(&weights), &Batch::new(&f.sequences).expect("batch"), &rows, &BTreeMap::new(), false).expect("execute");
    assert!(max(&kl_bits(&expected, &open.log_probabilities)) > 1e-6, "the prefix changes nothing unmasked");
}

/// VPD's view of layer 1's MLP, one subcomponent per hidden unit in each matrix (exact: the
/// remainders are zero).
fn exact_vpd(weights: &Weights) -> crate::graph::VpdMlp {
    let mlp = weights.layers[1].mlp.as_ref().expect("an MLP");
    let n = mlp.gate.nrows();
    crate::graph::VpdMlp { fc_u: ndarray::Array2::eye(n), fc_v: mlp.gate.t().to_owned(), down_u: mlp.out.t().to_owned(), down_v: ndarray::Array2::eye(n) }
}

#[test]
fn vpd_mlp_view_reads_the_hidden_stream_through_its_edges() {
    let f = fixture("graph_vpd");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let mut weights = Weights::of(&library);
    let vpd = exact_vpd(&weights);
    weights.vpd.insert(1, vpd);
    let piece = |layer: usize, view: &str, kind: &str, index: Option<Index>| PieceIr { view: view.into(), layer, kind: kind.into(), index };
    let node = |id: &str, p: PieceIr| NodeIr { id: id.into(), pieces: vec![p], rule: None };
    let all: Vec<usize> = (0..16).collect();
    let nodes = vec![
        node("a0", piece(0, "native", "head", None)),
        node("m0", piece(0, "native", "mlp", None)),
        node("a1", piece(1, "native", "head", None)),
        node("F", piece(1, "vpd", "c_fc", Some(Index::Many(all.clone())))),
        node("D", piece(1, "vpd", "down_proj", Some(Index::Many(all)))),
    ];
    let edge = |from: &str, to: &str, route: &str| EdgeIr { from: from.into(), to: to.into(), route: route.into() };
    let mut edges = Vec::new();
    for (to, writers) in [("a0", vec!["embed"]), ("m0", vec!["embed", "a0"]), ("a1", vec!["embed", "a0", "m0"]), ("F", vec!["embed", "a0", "m0", "a1"]), ("logits", vec!["embed", "a0", "m0", "a1", "D"])] {
        for w in writers {
            edges.push(edge(w, to, "input"));
        }
    }
    let cut = Program { model: "tiny".into(), valid: true, nodes, edges: edges.clone(), ..Program::default() };
    let mut joined = cut.clone();
    joined.edges.push(edge("F", "D", "input"));
    let cf = counterfactuals(&f.sequences);
    let stats = Stats::measure(&weights, &f.sequences).expect("stats");
    let mut batch = Batch::new(&f.sequences).expect("batch");
    batch.reference = Some(std::sync::Arc::new(reference(&weights, &stats, &Batch::new(&cf).expect("cf batch")).expect("reference")));
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    let clean = library.log_probabilities(&library.run(&f.sequences, &BTreeMap::new()).expect("run").last).expect("log p");
    let run_of = |program: &Program, edges: bool| {
        let graph = Graph::parse(program, &weights).expect("parse");
        execute(&weights, &stats, &graph.program(&weights, edges), &batch, &rows, &BTreeMap::new(), false).expect("execute").log_probabilities
    };
    // Every edge, the hidden one included: M. M itself with VPD nodes: M.
    assert!(max(&kl_bits(&clean, &run_of(&joined, true))) < 1e-9);
    assert!(max(&kl_bits(&clean, &run_of(&cut, false))) < 1e-9);
    let model = Graph::parse(&cut, &weights).expect("parse").model(&weights);
    assert!(max(&kl_bits(&clean, &execute(&weights, &stats, &model, &batch, &rows, &BTreeMap::new(), false).expect("execute").log_probabilities)) < 1e-9);
    // Without F >> D, D reads the counterfactual hidden activations: layer 1's MLP patched from x'.
    let cf_stream = library.run(&cf, &BTreeMap::new()).expect("cf run").streams[3].clone();
    let lw = &weights.layers[1];
    let mlp = lw.mlp.as_ref().expect("an MLP");
    let mut x_hat = cf_stream.clone();
    for mut row in x_hat.outer_iter_mut() {
        let r = crate::operator_program::rms_scale(row.view(), lw.mlp_norm.epsilon);
        row.zip_mut_with(&lw.mlp_norm.gain, |v, g| *v *= r * g);
    }
    let mut active = x_hat.dot(&mlp.gate.t()) + &mlp.bias.view().insert_axis(ndarray::Axis(0));
    active.mapv_inplace(|t| mlp.law.apply(t));
    let patched = library.log_probabilities(&library.run_with(&f.sequences, &BTreeMap::new(), &[(1, active)].into()).expect("patched").last).expect("log p");
    let kl = max(&kl_bits(&patched, &run_of(&cut, true)));
    assert!(kl < 1e-9, "KL(MLP patched from x' ‖ D without its hidden edge) = {kl:e} bits");
    // A c_fc node writes no residual; down_proj subcomponents read no residual.
    let mut bad = cut.clone();
    bad.edges.push(edge("F", "logits", "input"));
    assert!(Graph::parse(&bad, &weights).is_err());
    let mut bad = cut.clone();
    bad.edges.push(edge("a1", "D", "input"));
    assert!(Graph::parse(&bad, &weights).is_err());
}

#[test]
fn vpd_attention_view_reads_queries_keys_values_through_its_edges() {
    let f = fixture("graph_vpd_attention");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let mut weights = Weights::of(&library);
    // One subcomponent per output coordinate of q, k, v and per input coordinate of o (exact).
    let lw = &weights.layers[1];
    let stack = |m: &dyn Fn(&crate::graph::HeadWeights) -> ndarray::Array2<f64>| ndarray::concatenate(ndarray::Axis(0), &lw.heads.iter().map(m).collect::<Vec<_>>().iter().map(|a| a.view()).collect::<Vec<_>>()).expect("stack");
    let (wq, wk, wv) = (stack(&|h| h.query.clone()), stack(&|h| h.key.clone()), stack(&|h| h.value.clone()));
    let wo = ndarray::concatenate(ndarray::Axis(1), &lw.heads.iter().map(|h| h.output.view()).collect::<Vec<_>>()).expect("hstack");
    let exact_in = |w: &ndarray::Array2<f64>| (ndarray::Array2::eye(w.nrows()), w.t().to_owned());
    weights.vpd_attention.insert(1, crate::graph::VpdAttention { q: exact_in(&wq), k: exact_in(&wk), v: exact_in(&wv), o: (wo.t().to_owned(), ndarray::Array2::eye(wo.ncols())) });
    let piece = |layer: usize, view: &str, kind: &str, index: Option<Index>| PieceIr { view: view.into(), layer, kind: kind.into(), index };
    let all = |n: usize| Some(Index::Many((0..n).collect()));
    let nodes = vec![
        NodeIr { id: "a0".into(), pieces: vec![piece(0, "native", "head", None)], rule: None },
        NodeIr { id: "m0".into(), pieces: vec![piece(0, "native", "mlp", None)], rule: None },
        NodeIr { id: "QKV".into(), pieces: vec![piece(1, "vpd", "q_proj", all(wq.nrows())), piece(1, "vpd", "k_proj", all(wk.nrows())), piece(1, "vpd", "v_proj", all(wv.nrows()))], rule: None },
        NodeIr { id: "O".into(), pieces: vec![piece(1, "vpd", "o_proj", all(wo.ncols()))], rule: None },
        NodeIr { id: "m1".into(), pieces: vec![piece(1, "native", "mlp", None)], rule: None },
    ];
    let edge = |from: &str, to: &str| EdgeIr { from: from.into(), to: to.into(), route: "input".into() };
    let mut edges = Vec::new();
    for (to, writers) in [("a0", vec!["embed"]), ("m0", vec!["embed", "a0"]), ("QKV", vec!["embed", "a0", "m0"]), ("m1", vec!["embed", "a0", "m0", "O"]), ("logits", vec!["embed", "a0", "m0", "O", "m1"])] {
        edges.extend(writers.into_iter().map(|w| edge(w, to)));
    }
    let cut = Program { model: "tiny".into(), valid: true, nodes, edges, ..Program::default() };
    let mut joined = cut.clone();
    joined.edges.push(edge("QKV", "O"));
    let cf = counterfactuals(&f.sequences);
    let stats = Stats::measure(&weights, &f.sequences).expect("stats");
    let mut batch = Batch::new(&f.sequences).expect("batch");
    batch.reference = Some(std::sync::Arc::new(reference(&weights, &stats, &Batch::new(&cf).expect("cf batch")).expect("reference")));
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    let clean = library.log_probabilities(&library.run(&f.sequences, &BTreeMap::new()).expect("run").last).expect("log p");
    let run_of = |program: &Program, edges: bool| {
        let graph = Graph::parse(program, &weights).expect("parse");
        execute(&weights, &stats, &graph.program(&weights, edges), &batch, &rows, &BTreeMap::new(), false).expect("execute").log_probabilities
    };
    assert!(max(&kl_bits(&clean, &run_of(&joined, true))) < 1e-9, "every edge");
    assert!(max(&kl_bits(&clean, &run_of(&cut, false))) < 1e-9, "node level");
    let model = Graph::parse(&cut, &weights).expect("parse").model(&weights);
    let plain = Batch::new(&f.sequences).expect("batch");
    assert!(max(&kl_bits(&clean, &execute(&weights, &stats, &model, &plain, &rows, &BTreeMap::new(), false).expect("execute").log_probabilities)) < 1e-9, "M");
    // Without QKV >> O, O reads the counterfactual queries, keys and values: layer 1's heads patched from x'.
    let reads = library.run(&cf, &BTreeMap::new()).expect("cf run").reads;
    let patched = library.log_probabilities(&library.run(&f.sequences, &[(2, reads[2].clone()), (3, reads[3].clone())].into()).expect("patched").last).expect("log p");
    let kl = max(&kl_bits(&patched, &run_of(&cut, true)));
    assert!(kl < 1e-9, "KL(heads patched from x' ‖ O without its q/k/v edge) = {kl:e} bits");
    let mut bad = cut.clone();
    bad.edges.push(edge("QKV", "logits"));
    assert!(Graph::parse(&bad, &weights).is_err(), "q/k/v subcomponents write no residual");
    let mut mixed = cut.clone();
    mixed.nodes.push(NodeIr { id: "h".into(), pieces: vec![piece(1, "native", "head", Some(Index::One(0)))], rule: None });
    assert!(Graph::parse(&mixed, &weights).is_err(), "heads and VPD subcomponents of one attention");
}

#[test]
fn library_parts_resolve_to_their_vpd_subcomponents() {
    let f = fixture("graph_library");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let mut weights = Weights::of(&library);
    let vpd = exact_vpd(&weights);
    weights.vpd.insert(1, vpd);
    let lw = &weights.layers[1];
    let width: usize = lw.heads.iter().map(|h| h.query.nrows()).sum();
    let wo = ndarray::concatenate(ndarray::Axis(1), &lw.heads.iter().map(|h| h.output.view()).collect::<Vec<_>>()).expect("hstack");
    let eye_in = |rows: usize, w: ndarray::Array2<f64>| (ndarray::Array2::eye(rows), w);
    let stack = |m: &dyn Fn(&crate::graph::HeadWeights) -> ndarray::Array2<f64>| ndarray::concatenate(ndarray::Axis(0), &lw.heads.iter().map(m).collect::<Vec<_>>().iter().map(|a| a.view()).collect::<Vec<_>>()).expect("stack").t().to_owned();
    let attention = crate::graph::VpdAttention { q: eye_in(width, stack(&|h| h.query.clone())), k: eye_in(width, stack(&|h| h.key.clone())), v: eye_in(width, stack(&|h| h.value.clone())), o: (wo.t().to_owned(), ndarray::Array2::eye(width)) };
    weights.vpd_attention.insert(1, attention);
    // Layer 1: attention part 0 = every q, k, v subcomponent, part 1 = every o one; MLP part 0 = c_fc, part 1 = down_proj.
    let slices = |site: usize, n: usize| (0..n).map(|i| serde_json::json!([site, i])).collect::<Vec<_>>();
    let qkv: Vec<_> = [6, 7, 8].iter().flat_map(|&s| slices(s, width)).collect();
    let start = serde_json::json!([{"arm": "test", "components": [{"slices": qkv}, {"slices": slices(9, width)}, {"slices": slices(10, 16)}, {"slices": slices(11, 16)}]}]);
    let path = std::env::temp_dir().join(format!("gam_mpd_graph_library_{}.json", std::process::id()));
    std::fs::write(&path, start.to_string()).expect("write");
    assert_eq!(weights.attach_library(&path, "test").expect("attach"), 4);
    let piece = |view: &str, kind: &str, index: Index| PieceIr { view: view.into(), layer: 1, kind: kind.into(), index: Some(index) };
    let node = |id: &str, pieces: Vec<PieceIr>| NodeIr { id: id.into(), pieces, rule: None };
    let all = |n: usize| Index::Many((0..n).collect());
    let lib = Program { model: "tiny".into(), valid: true, nodes: vec![node("QKV", vec![piece("library", "attn", Index::One(0))]), node("O", vec![piece("library", "attn", Index::One(1))]), node("F", vec![piece("library", "mlp", Index::One(0))]), node("D", vec![piece("library", "mlp", Index::One(1))])], ..Program::default() };
    let vpd = Program {
        nodes: vec![
            node("QKV", vec![piece("vpd", "q_proj", all(width)), piece("vpd", "k_proj", all(width)), piece("vpd", "v_proj", all(width))]),
            node("O", vec![piece("vpd", "o_proj", all(width))]),
            node("F", vec![piece("vpd", "c_fc", all(16))]),
            node("D", vec![piece("vpd", "down_proj", all(16))]),
        ],
        ..lib.clone()
    };
    let (a, b) = (Graph::parse(&lib, &weights).expect("library"), Graph::parse(&vpd, &weights).expect("vpd"));
    assert_eq!(a.blocks, b.blocks);
    let mut mixed = lib.clone();
    mixed.nodes[1] = node("O", vec![piece("vpd", "o_proj", Index::One(0))]);
    assert!(Graph::parse(&mixed, &weights).is_err(), "library and VPD views of one attention");
}

#[test]
fn qwen3_like_model_full_graph_and_counterfactual_empty_graph() {
    // Grouped-query attention, head norms on queries and keys, SiLU-gated MLPs.
    let f = fixture_of(crate::test_support::tiny_qwen3_export("graph_qwen3", LAYERS));
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let weights = Weights::of(&library);
    assert!(weights.layers[0].heads[0].query_norm.is_some() && weights.layers[0].mlp.as_ref().is_some_and(|m| m.up.is_some()));
    let clean = library.log_probabilities(&library.run(&f.sequences, &BTreeMap::new()).expect("run").last).expect("log p");
    let cf = counterfactuals(&f.sequences);
    let target = library.log_probabilities(&library.run(&cf, &BTreeMap::new()).expect("run").last).expect("log p");
    let stats = Stats::measure(&weights, &f.sequences).expect("stats");
    let mut batch = Batch::new(&f.sequences).expect("batch");
    batch.reference = Some(std::sync::Arc::new(reference(&weights, &stats, &Batch::new(&cf).expect("cf batch")).expect("reference")));
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    let full = Graph::parse(&full_program(), &weights).expect("parse");
    for (name, circuit, expected) in [("full", full.program(&weights, true), &clean), ("empty", Graph::empty().program(&weights, true), &target)] {
        let run = execute(&weights, &stats, &circuit, &batch, &rows, &BTreeMap::new(), false).expect("execute");
        let kl = max(&kl_bits(expected, &run.log_probabilities));
        assert!(kl < 1e-9, "{name}: KL = {kl:e} bits");
    }
    // Opaque numbers count a key-value pair shared by two query heads once.
    let one = Program { model: "tiny".into(), valid: true, nodes: vec![NodeIr { id: "h".into(), pieces: vec![PieceIr { view: "native".into(), layer: 0, kind: "head".into(), index: None }], rule: None }], ..Program::default() };
    let g = Graph::parse(&one, &weights).expect("parse");
    let h = &weights.layers[0].heads[0];
    let per_query = h.query.len() + h.output.len() + h.query_norm.as_ref().map_or(0, |(g, _)| g.len());
    let shared = h.key.len() + h.value.len() + h.key_norm.as_ref().map_or(0, |(g, _)| g.len());
    assert_eq!(g.opaque_numbers(&weights), 2 * per_query + shared);
}

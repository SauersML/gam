#![cfg(test)]
//! The graph checker against the library's own runs of `M` on the tiny export (#2951).
use crate::{
    graph::{Batch, Behavior, Checker, Counterfactual, EdgeIr, Experiment, Graph, Index, NodeIr, PieceIr, Program, Prompt, WeightEdit, Weights, execute, kl_bits, reference},
    import::import_language_model,
    library_mdl,
    library_readout::{Activity, Edit, Library},
    operator_program::SlotValues,
    run_check::{layer_nodes, split_sites},
    test_support::tiny_export,
};
use gam_gpu::tensor::Device;
use std::collections::{BTreeMap, BTreeSet};

/// `graph_device::run`'s execution on a given device state: its logits normalized.
fn run_on(s: &mut crate::graph_device::DeviceState, weights: &Weights, circuit: &crate::graph::Circuit, job: &crate::graph_device::Run) -> Result<crate::graph::Execution, String> {
    crate::graph_device::logits_on_device(s, weights, circuit, job).and_then(crate::graph_device::normalized)
}

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
        nodes.push(NodeIr { id: format!("a{l}"), pieces: vec![piece(l, "head")], claim: None });
        nodes.push(NodeIr { id: format!("m{l}"), pieces: vec![piece(l, "mlp")], claim: None });
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
    Program { model: "tiny".into(), nodes, edges, python_tokens: 0, token_types: 0, source: String::new(), valid: true, error: None, standin: None, base: Vec::new(), explanation_tokens: 0, explanation_token_types: 0, alignments: Vec::new() }
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
    let graph = Graph::parse(&full_program(), &weights).expect("parse");
    for (name, circuit) in [("edges", graph.program(&weights, true)), ("nodes", graph.program(&weights, false)), ("model", graph.model(&weights))] {
        let run = execute(&weights, &circuit, &batch, &rows, &BTreeMap::new()).expect("execute");
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
    let restore = WeightEdit::Head { layer: 1, head: 0, factor: 0.0 }.apply(&mut weights).expect("edit");
    let circuit = Graph::empty().model(&weights);
    let run = execute(&weights, &circuit, &batch, &rows, &BTreeMap::new()).expect("execute");
    restore.restore(&mut weights).expect("restore");
    let kl = max(&kl_bits(&reference, &run.log_probabilities));
    assert!(kl < 1e-9, "KL(library edit ‖ graph edit) = {kl:e} bits");
}

#[test]
fn parse_rejects_bad_programs() {
    let f = fixture("graph_parse");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let weights = Weights::of(&library);
    let node = |id: &str, layer: usize, kind: &str, index: Option<Index>| NodeIr { id: id.into(), pieces: vec![PieceIr { view: "native".into(), layer, kind: kind.into(), index }], claim: None };
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
    // Each prompt's counterfactual is the next sequence (stand-ins come from it).
    let n = f.sequences.len();
    let prompts = f.sequences.iter().enumerate().map(|(i, s)| Prompt { text: String::new(), token_ids: s.clone(), target_positions: vec![s.len() - 2, s.len() - 1], counterfactual: Some(Counterfactual { text: String::new(), token_ids: f.sequences[(i + 1) % n].clone() }), attention_block: Vec::new() }).collect();
    let behavior = Behavior { id: "tiny".into(), model: "tiny".into(), family: String::new(), description: String::new(), frequency: None, prompts, split: "train".into(), model_accuracy: None };
    let mut checker = Checker::new(weights, behavior).expect("checker");
    let (full, outcomes) = checker.score(&full_program(), 24, 7, true, None).expect("score");
    assert!(full.valid);
    assert!(full.exec_error_bits / full.n < F32_KL, "full program error {:e} bits per token over {:?}", full.exec_error_bits / full.n, outcomes.iter().map(|o| o.0.family()).collect::<Vec<_>>());
    assert!(outcomes.iter().any(|o| matches!(o.0, Experiment::Edit { .. })));
    let (empty, _) = checker.score(&Program { model: "tiny".into(), valid: true, ..Program::default() }, 24, 7, true, None).expect("score");
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
    let mut batch = Batch::new(&f.sequences).expect("batch");
    batch.reference = Some(std::sync::Arc::new(reference(&weights, &Batch::new(&cf).expect("cf batch")).expect("reference")));
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    for (name, circuit) in [("empty", Graph::empty().program(&weights, true)), ("full", Graph::parse(&full_program(), &weights).expect("parse").program(&weights, true))] {
        let run = execute(&weights, &circuit, &batch, &rows, &BTreeMap::new()).expect("execute");
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
    let node = |id: &str, layer: usize, kind: &str, index: Option<Index>| NodeIr { id: id.into(), pieces: vec![PieceIr { view: "native".into(), layer, kind: kind.into(), index }], claim: None };
    let program = Program {
        model: "tiny".into(),
        valid: true,
        nodes: vec![node("a0", 0, "head", None), node("m0", 0, "mlp", None), node("a1", 1, "head", Some(Index::One(1))), node("m1", 1, "mlp", Some(Index::Many((0..16).collect())))],
        ..Program::default()
    };
    let mut batch = Batch::new(&f.sequences).expect("batch");
    batch.reference = Some(std::sync::Arc::new(reference(&weights, &Batch::new(&cf).expect("cf batch")).expect("reference")));
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    let circuit = Graph::parse(&program, &weights).expect("parse").program(&weights, false);
    let run = execute(&weights, &circuit, &batch, &rows, &BTreeMap::new()).expect("execute");
    let kl = max(&kl_bits(&expected, &run.log_probabilities));
    assert!(kl < 1e-9, "KL(activation patch ‖ graph) = {kl:e} bits");
    // A few neurons left out of layer 1's MLP: the complement is assembled by subtraction.
    let mut partial = program.clone();
    partial.nodes[3] = node("m1", 1, "mlp", Some(Index::Many((0..13).collect())));
    let circuit = Graph::parse(&partial, &weights).expect("parse").program(&weights, false);
    let left = execute(&weights, &circuit, &batch, &rows, &BTreeMap::new()).expect("execute");
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
    let feature = |id: &str, index: Index| NodeIr { id: id.into(), pieces: vec![PieceIr { view: "transcoder".into(), layer: 1, kind: "feature".into(), index: Some(index) }], claim: None };
    let program = Program { model: "tiny".into(), valid: true, nodes: vec![feature("f", Index::Many((0..6).collect()))], ..Program::default() };
    let graph = Graph::parse(&program, &weights).expect("parse");
    weights.load_features(&graph).expect("features");
    let batch = Batch::new(&f.sequences).expect("batch");
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    // M with the features as a node: the features plus the rest (other features and the error) is M.
    let run = execute(&weights, &graph.model(&weights), &batch, &rows, &BTreeMap::new()).expect("execute");
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
    // Writes come back from a run that scores no rows (on the device path as well).
    let writes = execute(&weights, &graph.model(&weights), &batch, &[], &BTreeMap::new()).expect("execute").writes;
    let got = writes[0].clone().expect("the node computes");
    let gap = (&got - &expected).iter().fold(0.0f64, |a, v| a.max(v.abs()));
    // The checker stores weights in float32; the start library's come from float64 arithmetic.
    assert!(gap < 1e-6, "feature writes differ from the transcoder by {gap:e}");
    // One view per site; features in range.
    let mut both = program.clone();
    both.nodes.push(NodeIr { id: "n".into(), pieces: vec![PieceIr { view: "native".into(), layer: 1, kind: "mlp".into(), index: Some(Index::One(0)) }], claim: None });
    assert!(Graph::parse(&both, &weights).is_err(), "neurons and features of one MLP");
    let far = Program { nodes: vec![feature("f", Index::One(6))], ..program.clone() };
    assert!(Graph::parse(&far, &weights).is_err(), "feature out of range");
}

#[test]
fn attention_block_hides_the_prefix_like_running_the_suffix_alone() {
    let f = fixture("graph_mask");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let weights = Weights::of(&library);
    // Queries from position 4 on do not see keys before 4: with rotary (relative) positions and no
    // absolute ones, positions 4.. then compute exactly what the suffix computes alone.
    let (t, length) = (4, f.sequences[0].len());
    let mut batch = Batch::new(&f.sequences).expect("batch");
    batch.blocks = vec![vec![[t, length, 0, t]]; f.sequences.len()];
    let rows: Vec<usize> = (0..f.sequences.len()).flat_map(|s| (t..length).map(move |p| s * length + p)).collect();
    let run = execute(&weights, &Graph::empty().model(&weights), &batch, &rows, &BTreeMap::new()).expect("execute");
    let suffixes: Vec<Vec<u32>> = f.sequences.iter().map(|s| s[t..].to_vec()).collect();
    let expected = library.log_probabilities(&library.run(&suffixes, &BTreeMap::new()).expect("run").last).expect("log p");
    let kl = max(&kl_bits(&expected, &run.log_probabilities));
    assert!(kl < 1e-9, "KL(suffix alone ‖ masked prefix) = {kl:e} bits");
    let open = execute(&weights, &Graph::empty().model(&weights), &Batch::new(&f.sequences).expect("batch"), &rows, &BTreeMap::new()).expect("execute");
    assert!(max(&kl_bits(&expected, &open.log_probabilities)) > 1e-6, "the prefix changes nothing unmasked");
}

/// VPD's view of layer 1's MLP, one subcomponent per hidden unit in each matrix (exact: the
/// remainders are zero).
fn exact_vpd(weights: &Weights) -> crate::graph::VpdMlp {
    let mlp = weights.layers[1].mlp.as_ref().expect("an MLP");
    let n = mlp.gate.nrows();
    crate::graph::VpdMlp { fc_u: ndarray::Array2::eye(n), fc_v: crate::graph::wide(mlp.gate.t()), down_u: crate::graph::wide(mlp.out.t()), down_v: ndarray::Array2::eye(n) }
}

#[test]
fn vpd_mlp_view_reads_the_hidden_stream_through_its_edges() {
    let f = fixture("graph_vpd");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let mut weights = Weights::of(&library);
    let vpd = exact_vpd(&weights);
    weights.vpd.insert(1, vpd);
    let piece = |layer: usize, view: &str, kind: &str, index: Option<Index>| PieceIr { view: view.into(), layer, kind: kind.into(), index };
    let node = |id: &str, p: PieceIr| NodeIr { id: id.into(), pieces: vec![p], claim: None };
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
    let mut batch = Batch::new(&f.sequences).expect("batch");
    batch.reference = Some(std::sync::Arc::new(reference(&weights, &Batch::new(&cf).expect("cf batch")).expect("reference")));
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    let clean = library.log_probabilities(&library.run(&f.sequences, &BTreeMap::new()).expect("run").last).expect("log p");
    let run_of = |program: &Program, edges: bool| {
        let graph = Graph::parse(program, &weights).expect("parse");
        execute(&weights, &graph.program(&weights, edges), &batch, &rows, &BTreeMap::new()).expect("execute").log_probabilities
    };
    // Every edge, the hidden one included: M. M itself with VPD nodes: M.
    assert!(max(&kl_bits(&clean, &run_of(&joined, true))) < 1e-9);
    assert!(max(&kl_bits(&clean, &run_of(&cut, false))) < 1e-9);
    let model = Graph::parse(&cut, &weights).expect("parse").model(&weights);
    assert!(max(&kl_bits(&clean, &execute(&weights, &model, &batch, &rows, &BTreeMap::new()).expect("execute").log_probabilities)) < 1e-9);
    // Without F >> D, D reads the counterfactual hidden activations: layer 1's MLP patched from x'.
    let cf_stream = library.run(&cf, &BTreeMap::new()).expect("cf run").streams[3].clone();
    let lw = &weights.layers[1];
    let mlp = lw.mlp.as_ref().expect("an MLP");
    let mut x_hat = cf_stream.clone();
    for mut row in x_hat.outer_iter_mut() {
        let r = crate::operator_program::rms_scale(row.view(), lw.mlp_norm.epsilon);
        row.zip_mut_with(&lw.mlp_norm.gain, |v, g| *v *= r * g);
    }
    let mut active = x_hat.dot(&crate::graph::wide(mlp.gate.t())) + &mlp.bias.view().insert_axis(ndarray::Axis(0));
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
    let (wq, wk, wv) = (stack(&|h| crate::graph::wide(h.query.view())), stack(&|h| crate::graph::wide(h.key.view())), stack(&|h| crate::graph::wide(h.value.view())));
    let wo = crate::graph::wide(ndarray::concatenate(ndarray::Axis(1), &lw.heads.iter().map(|h| h.output.view()).collect::<Vec<_>>()).expect("hstack").view());
    let exact_in = |w: &ndarray::Array2<f64>| (ndarray::Array2::eye(w.nrows()), w.t().to_owned());
    weights.vpd_attention.insert(1, crate::graph::VpdAttention { q: exact_in(&wq), k: exact_in(&wk), v: exact_in(&wv), o: (wo.t().to_owned(), ndarray::Array2::eye(wo.ncols())) });
    let piece = |layer: usize, view: &str, kind: &str, index: Option<Index>| PieceIr { view: view.into(), layer, kind: kind.into(), index };
    let all = |n: usize| Some(Index::Many((0..n).collect()));
    let nodes = vec![
        NodeIr { id: "a0".into(), pieces: vec![piece(0, "native", "head", None)], claim: None },
        NodeIr { id: "m0".into(), pieces: vec![piece(0, "native", "mlp", None)], claim: None },
        NodeIr { id: "QKV".into(), pieces: vec![piece(1, "vpd", "q_proj", all(wq.nrows())), piece(1, "vpd", "k_proj", all(wk.nrows())), piece(1, "vpd", "v_proj", all(wv.nrows()))], claim: None },
        NodeIr { id: "O".into(), pieces: vec![piece(1, "vpd", "o_proj", all(wo.ncols()))], claim: None },
        NodeIr { id: "m1".into(), pieces: vec![piece(1, "native", "mlp", None)], claim: None },
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
    let mut batch = Batch::new(&f.sequences).expect("batch");
    batch.reference = Some(std::sync::Arc::new(reference(&weights, &Batch::new(&cf).expect("cf batch")).expect("reference")));
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    let clean = library.log_probabilities(&library.run(&f.sequences, &BTreeMap::new()).expect("run").last).expect("log p");
    let run_of = |program: &Program, edges: bool| {
        let graph = Graph::parse(program, &weights).expect("parse");
        execute(&weights, &graph.program(&weights, edges), &batch, &rows, &BTreeMap::new()).expect("execute").log_probabilities
    };
    assert!(max(&kl_bits(&clean, &run_of(&joined, true))) < 1e-9, "every edge");
    assert!(max(&kl_bits(&clean, &run_of(&cut, false))) < 1e-9, "node level");
    let model = Graph::parse(&cut, &weights).expect("parse").model(&weights);
    let plain = Batch::new(&f.sequences).expect("batch");
    assert!(max(&kl_bits(&clean, &execute(&weights, &model, &plain, &rows, &BTreeMap::new()).expect("execute").log_probabilities)) < 1e-9, "M");
    // Without QKV >> O, O reads the counterfactual queries, keys and values: layer 1's heads patched from x'.
    let reads = library.run(&cf, &BTreeMap::new()).expect("cf run").reads;
    let patched = library.log_probabilities(&library.run(&f.sequences, &[(2, reads[2].clone()), (3, reads[3].clone())].into()).expect("patched").last).expect("log p");
    let kl = max(&kl_bits(&patched, &run_of(&cut, true)));
    assert!(kl < 1e-9, "KL(heads patched from x' ‖ O without its q/k/v edge) = {kl:e} bits");
    let mut bad = cut.clone();
    bad.edges.push(edge("QKV", "logits"));
    assert!(Graph::parse(&bad, &weights).is_err(), "q/k/v subcomponents write no residual");
    let mut mixed = cut.clone();
    mixed.nodes.push(NodeIr { id: "h".into(), pieces: vec![piece(1, "native", "head", Some(Index::One(0)))], claim: None });
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
    let wo = crate::graph::wide(ndarray::concatenate(ndarray::Axis(1), &lw.heads.iter().map(|h| h.output.view()).collect::<Vec<_>>()).expect("hstack").view());
    let eye_in = |rows: usize, w: ndarray::Array2<f64>| (ndarray::Array2::eye(rows), w);
    let stack = |m: &dyn Fn(&crate::graph::HeadWeights) -> ndarray::Array2<f64>| ndarray::concatenate(ndarray::Axis(0), &lw.heads.iter().map(m).collect::<Vec<_>>().iter().map(|a| a.view()).collect::<Vec<_>>()).expect("stack").t().to_owned();
    let attention = crate::graph::VpdAttention { q: eye_in(width, stack(&|h| crate::graph::wide(h.query.view()))), k: eye_in(width, stack(&|h| crate::graph::wide(h.key.view()))), v: eye_in(width, stack(&|h| crate::graph::wide(h.value.view()))), o: (wo.t().to_owned(), ndarray::Array2::eye(width)) };
    weights.vpd_attention.insert(1, attention);
    // Layer 1: attention part 0 = every q, k, v subcomponent, part 1 = every o one; MLP part 0 = c_fc, part 1 = down_proj.
    let slices = |site: usize, n: usize| (0..n).map(|i| serde_json::json!([site, i])).collect::<Vec<_>>();
    let qkv: Vec<_> = [6, 7, 8].iter().flat_map(|&s| slices(s, width)).collect();
    let start = serde_json::json!([{"arm": "test", "components": [{"slices": qkv}, {"slices": slices(9, width)}, {"slices": slices(10, 16)}, {"slices": slices(11, 16)}]}]);
    let path = std::env::temp_dir().join(format!("gam_mpd_graph_library_{}.json", std::process::id()));
    std::fs::write(&path, start.to_string()).expect("write");
    assert_eq!(weights.attach_library(&path, "test").expect("attach"), 4);
    let piece = |view: &str, kind: &str, index: Index| PieceIr { view: view.into(), layer: 1, kind: kind.into(), index: Some(index) };
    let node = |id: &str, pieces: Vec<PieceIr>| NodeIr { id: id.into(), pieces, claim: None };
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
    let mut batch = Batch::new(&f.sequences).expect("batch");
    batch.reference = Some(std::sync::Arc::new(reference(&weights, &Batch::new(&cf).expect("cf batch")).expect("reference")));
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    let full = Graph::parse(&full_program(), &weights).expect("parse");
    for (name, circuit, expected) in [("full", full.program(&weights, true), &clean), ("empty", Graph::empty().program(&weights, true), &target)] {
        let run = execute(&weights, &circuit, &batch, &rows, &BTreeMap::new()).expect("execute");
        let kl = max(&kl_bits(expected, &run.log_probabilities));
        assert!(kl < 1e-9, "{name}: KL = {kl:e} bits");
    }
    // Opaque numbers count a key-value pair shared by two query heads once.
    let one = Program { model: "tiny".into(), valid: true, nodes: vec![NodeIr { id: "h".into(), pieces: vec![PieceIr { view: "native".into(), layer: 0, kind: "head".into(), index: None }], claim: None }], ..Program::default() };
    let g = Graph::parse(&one, &weights).expect("parse");
    let h = &weights.layers[0].heads[0];
    let per_query = h.query.len() + h.output.len() + h.query_norm.as_ref().map_or(0, |(g, _)| g.len());
    let shared = h.key.len() + h.value.len() + h.key_norm.as_ref().map_or(0, |(g, _)| g.len());
    assert_eq!(g.opaque_numbers(&weights), 2 * per_query + shared);
}

#[test]
fn device_path_on_the_host_backend_is_the_host_run() {
    let f = fixture("graph_device");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let weights = Weights::of(&library);
    let cf = counterfactuals(&f.sequences);
    let counterfactual = std::sync::Arc::new(reference(&weights, &Batch::new(&cf).expect("cf batch")).expect("reference"));
    let mut state = crate::graph_device::DeviceState::new(Device::host());
    let plain = Batch::new(&f.sequences).expect("batch");
    let mut with_reference = plain.clone();
    with_reference.reference = Some(counterfactual.clone());
    let rows: Vec<usize> = (0..plain.tokens.len()).collect();
    let node = |id: &str, layer: usize, kind: &str, index: Option<Index>| NodeIr { id: id.into(), pieces: vec![PieceIr { view: "native".into(), layer, kind: kind.into(), index }], claim: None };
    let partial = Program { model: "tiny".into(), valid: true, nodes: vec![node("h", 1, "head", Some(Index::One(0))), node("n", 0, "mlp", Some(Index::Many(vec![1, 4, 9]))), node("big", 1, "mlp", Some(Index::Many((0..14).collect())))], ..Program::default() };
    let full = Graph::parse(&full_program(), &weights).expect("full");
    let some = Graph::parse(&partial, &weights).expect("partial");
    let cases = [
        ("model", full.model(&weights), &plain),
        ("full", full.program(&weights, true), &with_reference),
        ("empty", Graph::empty().program(&weights, true), &with_reference),
        ("partial nodes", some.program(&weights, false), &with_reference),
        ("partial edges", some.program(&weights, true), &with_reference),
        ("partial model", some.model(&weights), &plain),
    ];
    for (name, circuit, batch) in cases {
        let host = execute(&weights, &circuit, batch, &rows, &BTreeMap::new()).expect("host");
        let job = crate::graph_device::Run { tokens: &batch.tokens, spans: &batch.spans, scored: &rows, swaps: &BTreeMap::new(), capture: false, reference: batch.reference.as_deref(), ops: &crate::graph::Interventions::default() };
        let device = run_on(&mut state, &weights, &circuit, &job).expect("device");
        let kl = max(&kl_bits(&host.log_probabilities, &device.log_probabilities));
        assert!(kl < 1e-9, "{name}: KL(host ‖ device) = {kl:e} bits");
    }
    // A capture is the host's Reference.
    let circuit = Graph::empty().model(&weights);
    let cf_batch = Batch::new(&cf).expect("cf batch");
    let job = crate::graph_device::Run { tokens: &cf_batch.tokens, spans: &cf_batch.spans, scored: &[], swaps: &BTreeMap::new(), capture: true, reference: None, ops: &crate::graph::Interventions::default() };
    let captured = run_on(&mut state, &weights, &circuit, &job).expect("capture").captured().expect("captured");
    let gap = |a: &ndarray::Array2<f64>, b: &ndarray::Array2<f64>| (a - b).iter().fold(0.0f64, |m, v| m.max(v.abs()));
    // (Normed inputs are captured only when a view reads them; none is attached here.)
    assert!(gap(&captured.active[1], &counterfactual.active[1]) < 1e-9 && gap(&captured.reads[1][0], &counterfactual.reads[1][0]) < 1e-9 && gap(&captured.mlp[0], &counterfactual.mlp[0]) < 1e-9);
}

#[test]
fn device_site_operations_on_the_host_backend_are_the_host_run() {
    use crate::graph::{After, Donor, Interventions, OnInput, Writer, execute_with};
    let f = fixture("graph_device_sites");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let weights = Weights::of(&library);
    let cf = counterfactuals(&f.sequences);
    let cf_batch = Batch::new(&cf).expect("cf batch");
    let counterfactual = std::sync::Arc::new(reference(&weights, &cf_batch).expect("reference"));
    let mut batch = Batch::new(&f.sequences).expect("batch");
    batch.reference = Some(counterfactual);
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    let node = |id: &str, layer: usize, kind: &str, index: Option<Index>| NodeIr { id: id.into(), pieces: vec![PieceIr { view: "native".into(), layer, kind: kind.into(), index }], claim: None };
    let program = Program { model: "tiny".into(), valid: true, nodes: vec![node("h", 1, "head", Some(Index::One(0))), node("n", 0, "mlp", Some(Index::Many(vec![1, 4, 9])))], ..Program::default() };
    let graph = Graph::parse(&program, &weights).expect("parse");
    let mut state = crate::graph_device::DeviceState::new(Device::host());
    for (name, circuit) in [("model", graph.model(&weights)), ("program", graph.program(&weights, true))] {
        // The donor run: the same circuit on the counterfactuals, normed inputs kept at every site.
        let mut donor_batch = cf_batch.clone();
        donor_batch.reference = batch.reference.clone().filter(|_| !circuit.is_model());
        let recording = Interventions { record: (0..4).collect(), ..Interventions::default() };
        let donor_run = execute_with(&weights, &circuit, &donor_batch, &[], &BTreeMap::new(), &recording).expect("donor run");
        let embed = crate::graph::wide(weights.embedding.select(ndarray::Axis(0), &cf_batch.tokens.iter().map(|t| *t as usize).collect::<Vec<_>>()).view());
        let donor = Donor { embed, writes: donor_run.writes.clone(), normed: donor_run.normed.clone(), reads: BTreeMap::new() };
        let push = ndarray::Array1::from_shape_fn(weights.width(), |c| 0.3 * (c as f64 + 1.0).sin());
        let ops = Interventions {
            after: vec![
                (None, After::Scale(vec![Writer::Embed], vec![0, 13, 30], 0.5)),
                (Some(0), After::Swap(vec![Writer::Unit(0)], vec![5, 6, 40])),
                (Some(1), After::Push(vec![2, 17], push.clone())),
                (Some(2), After::Scale(vec![Writer::Unit(1)], vec![3, 25], 2.0)),
            ],
            inputs: vec![(1, OnInput::Scale(vec![1, 4], 2.0)), (2, OnInput::Push(vec![8], push.clone())), (3, OnInput::Swap(vec![9, 33]))],
            cuts: vec![(3, vec![Writer::Embed, Writer::Unit(0)], vec![7, 50])],
            donor: Some(donor),
            record: [2].into_iter().collect(),
            head_reads: Vec::new(),
            record_reads: BTreeSet::new(),
        };
        let host = execute_with(&weights, &circuit, &batch, &rows, &BTreeMap::new(), &ops).expect("host");
        let reference = if circuit.is_model() { None } else { batch.reference.as_deref() };
        let job = crate::graph_device::Run { tokens: &batch.tokens, spans: &batch.spans, scored: &rows, swaps: &BTreeMap::new(), capture: false, reference, ops: &ops };
        let device = run_on(&mut state, &weights, &circuit, &job).expect("device");
        let kl = max(&kl_bits(&host.log_probabilities, &device.log_probabilities));
        assert!(kl < 1e-9, "{name}: KL(host ‖ device) under site operations = {kl:e} bits");
        assert_eq!(host.normed.keys().collect::<Vec<_>>(), device.normed.keys().collect::<Vec<_>>(), "{name}: recorded inputs");
    }
}

#[test]
fn weights_read_from_the_native_program_are_the_librarys() {
    for (tag, qwen) in [("graph_native_weights", false), ("graph_native_weights_qwen3", true)] {
        let dir = if qwen { crate::test_support::tiny_qwen3_export(tag, LAYERS) } else { tiny_export(tag, LAYERS) };
        let f = fixture_of(dir);
        let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
        let (a, b) = (Weights::of(&library), Weights::from_native(&f.native, &f.layers).expect("native weights"));
        let same = |x: &crate::graph::Stored, y: &crate::graph::Stored| x.dim() == y.dim() && x.iter().zip(y.iter()).all(|(p, q)| p == q);
        assert!(same(&a.embedding, &b.embedding) && same(&a.unembedding, &b.unembedding) && a.final_norm.gain == b.final_norm.gain, "{tag}: embeddings");
        for (la, lb) in a.layers.iter().zip(&b.layers) {
            assert!(la.attention.gain == lb.attention.gain && la.mlp_norm.gain == lb.mlp_norm.gain, "{tag}: norms");
            for (ha, hb) in la.heads.iter().zip(&lb.heads) {
                assert!(same(&ha.query, &hb.query) && same(&*ha.key, &*hb.key) && same(&*ha.value, &*hb.value) && same(&ha.output, &hb.output) && ha.query_norm == hb.query_norm && ha.key_norm == hb.key_norm && ha.scale == hb.scale, "{tag}: heads");
            }
            let (ma, mb) = (la.mlp.as_ref().expect("mlp"), lb.mlp.as_ref().expect("mlp"));
            assert!(same(&ma.gate, &mb.gate) && same(&ma.out, &mb.out) && ma.bias == mb.bias && ma.up_bias == mb.up_bias && ma.law == mb.law && ma.up.as_ref().map(|u| u.dim()) == mb.up.as_ref().map(|u| u.dim()), "{tag}: MLPs");
        }
    }
}

#[test]
fn device_path_pads_sequences_of_different_lengths() {
    let f = fixture("graph_device_lengths");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let weights = Weights::of(&library);
    // Six sequences of four lengths.
    let sequences: Vec<Vec<u32>> = f.sequences.iter().zip([12, 7, 9, 12, 5, 9]).map(|(s, n)| s[..n].to_vec()).collect();
    let batch = Batch::new(&sequences).expect("batch");
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    let circuit = Graph::parse(&full_program(), &weights).expect("full").model(&weights);
    let host = execute(&weights, &circuit, &batch, &rows, &BTreeMap::new()).expect("host");
    let mut state = crate::graph_device::DeviceState::new(Device::host());
    let job = crate::graph_device::Run { tokens: &batch.tokens, spans: &batch.spans, scored: &rows, swaps: &BTreeMap::new(), capture: false, reference: None, ops: &crate::graph::Interventions::default() };
    let device = run_on(&mut state, &weights, &circuit, &job).expect("device");
    let kl = max(&kl_bits(&host.log_probabilities, &device.log_probabilities));
    assert!(kl < 1e-9, "KL(host ‖ device) over sequences of different lengths = {kl:e} bits");
}

#[test]
fn device_path_on_a_qwen3_like_model_is_the_host_run() {
    // Head norms, grouped keys and values, gated MLPs, prompts of different lengths: the batched
    // heads path and the per-head path against the host.
    let f = fixture_of(crate::test_support::tiny_qwen3_export("graph_device_qwen3", LAYERS));
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let weights = Weights::of(&library);
    let sequences: Vec<Vec<u32>> = f.sequences.iter().zip([12, 7, 9, 12, 5, 9]).map(|(s, n)| s[..n].to_vec()).collect();
    let cf = counterfactuals(&f.sequences).into_iter().zip([12, 7, 9, 12, 5, 9]).map(|(s, n)| s[..n].to_vec()).collect::<Vec<_>>();
    let mut batch = Batch::new(&sequences).expect("batch");
    batch.reference = Some(std::sync::Arc::new(reference(&weights, &Batch::new(&cf).expect("cf batch")).expect("reference")));
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    let node = |id: &str, layer: usize, kind: &str, index: Option<Index>| NodeIr { id: id.into(), pieces: vec![PieceIr { view: "native".into(), layer, kind: kind.into(), index }], claim: None };
    let partial = Program { model: "tiny".into(), valid: true, nodes: vec![node("h", 1, "head", Some(Index::One(0))), node("n", 0, "mlp", Some(Index::Many(vec![1, 4, 9])))], ..Program::default() };
    let graph = Graph::parse(&partial, &weights).expect("partial");
    let mut state = crate::graph_device::DeviceState::new(Device::host());
    for (name, circuit) in [("model", graph.model(&weights)), ("program", graph.program(&weights, true))] {
        let host = execute(&weights, &circuit, &batch, &rows, &BTreeMap::new()).expect("host");
        let job = crate::graph_device::Run { tokens: &batch.tokens, spans: &batch.spans, scored: &rows, swaps: &BTreeMap::new(), capture: false, reference: batch.reference.as_deref(), ops: &crate::graph::Interventions::default() };
        let device = run_on(&mut state, &weights, &circuit, &job).expect("device");
        let kl = max(&kl_bits(&host.log_probabilities, &device.log_probabilities));
        assert!(kl < 1e-9, "{name}: KL(host ‖ device) = {kl:e} bits");
    }
}

#[test]
fn attend_rules_pick_their_positions() {
    use crate::graph::{HeadRule, TokenExpr};
    let tokens = [7u32, 3, 5, 7, 3, 9];
    let offset = HeadRule::Offset(1).pattern(&tokens);
    assert!(offset[[0, 0]] == 1.0 && (1..6).all(|t| offset[[t, t - 1]] == 1.0));
    let first = HeadRule::First.pattern(&tokens);
    assert!((0..6).all(|t| first[[t, 0]] == 1.0));
    // Induction: attend to the position after an earlier copy of the current token.
    let induction = HeadRule::Match { query: TokenExpr::Tokens, key: TokenExpr::Shift(Box::new(TokenExpr::Tokens), 1) }.pattern(&tokens);
    assert_eq!(induction[[3, 1]], 1.0, "7 at 3 attends to the token after the 7 at 0");
    assert_eq!(induction[[4, 2]], 1.0, "3 at 4 attends to the token after the 3 at 1");
    assert_eq!(induction[[5, 0]], 1.0, "9 has no earlier copy: position 0");
    let parsed = HeadRule::parse(&serde_json::json!({"op": "attend", "query": {"op": "tokens"}, "key": {"op": "shift", "arg": {"op": "tokens"}, "by": 1}})).expect("parse");
    assert_eq!(parsed.pattern(&tokens), induction);
}

/// Deleting VPD programs that differ only in the subcomponents they name run stacked as one batch
/// of copies (`graph::execute_stacked`), each copy giving its own program's run: q/k/v/o and
/// c_fc/down_proj subcomponents, remainders named by some copies only, a copy without an attention
/// unit and one with o_proj and v_proj alone.
#[test]
fn stacked_vpd_programs_are_their_own_runs() {
    use ndarray::{Array2, Axis, s};
    let f = fixture("graph_device_stacked");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let mut weights = Weights::of(&library);
    // Inexact views of layer 1, as in device_path_runs_vpd_views_as_the_host: every remainder nonzero.
    let partial = |w: &Array2<f64>, count: usize, scale: f64| (Array2::<f64>::eye(w.nrows()).slice(s![..count, ..]).to_owned() * scale, w.t().slice(s![.., ..count]).to_owned());
    let lw = &weights.layers[1];
    let mlp = lw.mlp.as_ref().expect("an MLP");
    let hidden = mlp.gate.nrows();
    let (fc_u, fc_v) = partial(&crate::graph::wide(mlp.gate.view()), hidden - 5, 0.7);
    let (down_u, down_v) = (crate::graph::wide(mlp.out.t().slice(s![..hidden - 4, ..])) * 0.6, Array2::<f64>::eye(hidden).slice(s![.., ..hidden - 4]).to_owned());
    let stack = |m: &dyn Fn(&crate::graph::HeadWeights) -> &crate::graph::Stored| crate::graph::wide(ndarray::concatenate(Axis(0), &lw.heads.iter().map(|h| m(h).view()).collect::<Vec<_>>()).expect("stack").view());
    let (wq, wk, wv) = (stack(&|h| &h.query), stack(&|h| &*h.key), stack(&|h| &*h.value));
    let wo = crate::graph::wide(ndarray::concatenate(Axis(1), &lw.heads.iter().map(|h| h.output.view()).collect::<Vec<_>>()).expect("hstack").view());
    let o = (wo.t().slice(s![..wo.ncols() - 3, ..]).to_owned() * 0.5, Array2::<f64>::eye(wo.ncols()).slice(s![.., ..wo.ncols() - 3]).to_owned());
    let attention = crate::graph::VpdAttention { q: partial(&wq, wq.nrows() - 2, 0.8), k: partial(&wk, wk.nrows() - 3, 0.9), v: partial(&wv, wv.nrows() - 1, 0.6), o };
    weights.vpd.insert(1, crate::graph::VpdMlp { fc_u, fc_v, down_u, down_v });
    weights.vpd_attention.insert(1, attention);
    let piece = |kind: &str, index: Index| PieceIr { view: "vpd".into(), layer: 1, kind: kind.into(), index: Some(index) };
    let rest = || Index::Name("rest".into());
    let many = |v: &[usize]| Index::Many(v.to_vec());
    // Per program its attention node's and MLP node's pieces (none: no such node).
    let programs: Vec<(Vec<PieceIr>, Vec<PieceIr>)> = vec![
        (vec![piece("q_proj", many(&[0, 3])), piece("k_proj", many(&[1])), piece("k_proj", rest()), piece("v_proj", many(&[2, 5])), piece("o_proj", many(&[0, 1, 4]))], vec![piece("c_fc", many(&[0, 2, 7])), piece("c_fc", rest()), piece("down_proj", many(&[1, 3, 5]))]),
        (vec![piece("q_proj", many(&[0])), piece("k_proj", many(&[1, 2])), piece("v_proj", many(&[2])), piece("o_proj", many(&[1])), piece("o_proj", rest())], vec![piece("c_fc", many(&[2, 9])), piece("down_proj", many(&[3, 5, 6])), piece("down_proj", rest())]),
        (Vec::new(), vec![piece("c_fc", many(&[0, 2])), piece("down_proj", many(&[1]))]),
        (vec![piece("v_proj", many(&[5])), piece("o_proj", many(&[4]))], Vec::new()),
    ];
    let edge = |from: &str, to: &str, route: &str| EdgeIr { from: from.into(), to: to.into(), route: route.into() };
    let graphs: Vec<Graph> = programs
        .into_iter()
        .map(|(a, m)| {
            let mut nodes = Vec::new();
            let mut edges = Vec::new();
            let mut writers = vec!["embed"];
            if !a.is_empty() {
                // An edge on each route the node's parts read.
                let read = |kind: &str| a.iter().any(|p| p.kind == kind);
                edges.extend([("q_proj", "query"), ("k_proj", "key"), ("v_proj", "value")].iter().filter(|(k, _)| read(k)).map(|(_, r)| edge("embed", "A", r)));
                nodes.push(NodeIr { id: "A".into(), pieces: a, claim: None });
                writers.push("A");
            }
            if !m.is_empty() {
                nodes.push(NodeIr { id: "M".into(), pieces: m, claim: None });
                edges.extend(writers.iter().map(|w| edge(w, "M", "input")));
                writers.push("M");
            }
            edges.extend(writers.iter().map(|w| edge(w, "logits", "input")));
            let program = Program { model: "tiny".into(), valid: true, nodes, edges, standin: Some("delete".into()), ..Program::default() };
            Graph::parse(&program, &weights).expect("parse")
        })
        .collect();
    let circuits: Vec<crate::graph::Circuit> = graphs.iter().map(|g| g.program(&weights, true)).collect();
    assert!(circuits.iter().all(crate::graph::stacks), "every deleting VPD program with every edge stacks");
    let plain = Batch::new(&f.sequences).expect("batch");
    let rows: Vec<usize> = (0..plain.tokens.len()).collect();
    let mut deleting = plain.clone();
    deleting.reference = Some(std::sync::Arc::new(crate::graph::Reference::zeros(&weights, plain.tokens.len())));
    let refs: Vec<&crate::graph::Circuit> = circuits.iter().collect();
    let stacked = crate::graph::stack(&weights, &refs, &plain, &rows).expect("the programs stack");
    let mut state = crate::graph_device::DeviceState::new(Device::host());
    let job = stacked.job();
    assert!(crate::graph_device::stack_covered(&weights, &stacked.merged, &job, &stacked.copies), "the device stacks VPD units");
    let copies = crate::graph_device::per_copy(crate::graph_device::copies_on(&mut state, &weights, &stacked.merged, &job, Some(&stacked.copies)).expect("stacked run"), stacked.copies.count).expect("copies");
    assert_eq!(copies.len(), circuits.len());
    for (j, (circuit, copy)) in circuits.iter().zip(&copies).enumerate() {
        let own = execute(&weights, circuit, &deleting, &rows, &BTreeMap::new()).expect("own run").log_probabilities;
        let kl = max(&kl_bits(&own, copy));
        assert!(kl < 1e-9, "program {j}: KL(own run ‖ stacked copy) = {kl:e} bits");
    }
    // The copies differ: each names its own subcomponents.
    assert!(max(&kl_bits(&copies[0], &copies[2])) > 1e-6, "programs 0 and 2 run alike");
    // Counterfactual stand-ins do not stack.
    let mut counterfactual = graphs[0].clone();
    counterfactual.delete = false;
    assert!(!crate::graph::stacks(&counterfactual.program(&weights, true)), "a counterfactual program stacked");
}

#[test]
fn device_path_runs_vpd_views_as_the_host() {
    use ndarray::{Array2, Axis, s};
    let f = fixture("graph_device_vpd");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let mut weights = Weights::of(&library);
    // Inexact views of layer 1 (scaled, some of each matrix's directions only), so every remainder
    // is nonzero: `U` scaled rows of the identity, `V` the matching columns of `Wᵀ`.
    let partial = |w: &Array2<f64>, count: usize, scale: f64| (Array2::<f64>::eye(w.nrows()).slice(s![..count, ..]).to_owned() * scale, w.t().slice(s![.., ..count]).to_owned());
    let lw = &weights.layers[1];
    let mlp = lw.mlp.as_ref().expect("an MLP");
    let hidden = mlp.gate.nrows();
    let (fc_u, fc_v) = partial(&crate::graph::wide(mlp.gate.view()), hidden - 5, 0.7);
    let (down_u, down_v) = (crate::graph::wide(mlp.out.t().slice(s![..hidden - 4, ..])) * 0.6, Array2::<f64>::eye(hidden).slice(s![.., ..hidden - 4]).to_owned());
    let stack = |m: &dyn Fn(&crate::graph::HeadWeights) -> &crate::graph::Stored| crate::graph::wide(ndarray::concatenate(Axis(0), &lw.heads.iter().map(|h| m(h).view()).collect::<Vec<_>>()).expect("stack").view());
    let (wq, wk, wv) = (stack(&|h| &h.query), stack(&|h| &*h.key), stack(&|h| &*h.value));
    let wo = crate::graph::wide(ndarray::concatenate(Axis(1), &lw.heads.iter().map(|h| h.output.view()).collect::<Vec<_>>()).expect("hstack").view());
    let o = (wo.t().slice(s![..wo.ncols() - 3, ..]).to_owned() * 0.5, Array2::<f64>::eye(wo.ncols()).slice(s![.., ..wo.ncols() - 3]).to_owned());
    let attention = crate::graph::VpdAttention { q: partial(&wq, wq.nrows() - 2, 0.8), k: partial(&wk, wk.nrows() - 3, 0.9), v: partial(&wv, wv.nrows() - 1, 0.6), o };
    weights.vpd.insert(1, crate::graph::VpdMlp { fc_u, fc_v, down_u, down_v });
    weights.vpd_attention.insert(1, attention);
    let piece = |kind: &str, index: Index| PieceIr { view: "vpd".into(), layer: 1, kind: kind.into(), index: Some(index) };
    let rest = || Index::Name("rest".into());
    let nodes = vec![
        NodeIr { id: "a0".into(), pieces: vec![PieceIr { view: "native".into(), layer: 0, kind: "head".into(), index: None }], claim: None },
        NodeIr { id: "m0".into(), pieces: vec![PieceIr { view: "native".into(), layer: 0, kind: "mlp".into(), index: None }], claim: None },
        NodeIr { id: "QK".into(), pieces: vec![piece("q_proj", Index::Many(vec![0, 3])), piece("k_proj", Index::One(1)), piece("k_proj", rest())], claim: None },
        NodeIr { id: "V".into(), pieces: vec![piece("v_proj", Index::Many(vec![2, 5]))], claim: None },
        NodeIr { id: "O".into(), pieces: vec![piece("o_proj", Index::Many(vec![0, 1, 4]))], claim: None },
        NodeIr { id: "F".into(), pieces: vec![piece("c_fc", Index::Many(vec![0, 2, 7])), piece("c_fc", rest())], claim: None },
        NodeIr { id: "D".into(), pieces: vec![piece("down_proj", Index::Many(vec![1, 3, 5]))], claim: None },
    ];
    let edge = |from: &str, to: &str| EdgeIr { from: from.into(), to: to.into(), route: "input".into() };
    let mut edges = vec![edge("QK", "O"), edge("F", "D")];
    for (to, writers) in [("a0", vec!["embed"]), ("m0", vec!["embed", "a0"]), ("QK", vec!["embed", "a0", "m0"]), ("V", vec!["embed", "m0"]), ("F", vec!["embed", "a0", "O"]), ("logits", vec!["embed", "a0", "m0", "O", "D"])] {
        edges.extend(writers.into_iter().map(|w| edge(w, to)));
    }
    let program = Program { model: "tiny".into(), valid: true, nodes, edges, ..Program::default() };
    let graph = Graph::parse(&program, &weights).expect("parse");
    let cf = counterfactuals(&f.sequences);
    let plain = Batch::new(&f.sequences).expect("batch");
    let mut with_reference = plain.clone();
    with_reference.reference = Some(std::sync::Arc::new(reference(&weights, &Batch::new(&cf).expect("cf batch")).expect("reference")));
    let rows: Vec<usize> = (0..plain.tokens.len()).collect();
    let mut state = crate::graph_device::DeviceState::new(Device::host());
    let cases = [
        ("model", graph.model(&weights), &plain),
        ("edges", graph.program(&weights, true), &with_reference),
        ("nodes", graph.program(&weights, false), &with_reference),
        ("empty", Graph::empty().program(&weights, true), &with_reference),
    ];
    for (name, circuit, batch) in cases {
        assert!(circuit.units.iter().any(|u| matches!(u.block, crate::graph::Block::Slices { .. })) || name == "empty", "{name}: VPD units");
        let host = execute(&weights, &circuit, batch, &rows, &BTreeMap::new()).expect("host");
        let job = crate::graph_device::Run { tokens: &batch.tokens, spans: &batch.spans, scored: &rows, swaps: &BTreeMap::new(), capture: false, reference: batch.reference.as_deref(), ops: &crate::graph::Interventions::default() };
        let device = run_on(&mut state, &weights, &circuit, &job).expect("device");
        let kl = max(&kl_bits(&host.log_probabilities, &device.log_probabilities));
        assert!(kl < 1e-9, "{name}: KL(host ‖ device) with VPD views = {kl:e} bits");
    }
    // The program is not M: its undeclared subcomponents and edges carry the counterfactual.
    let clean = execute(&weights, &graph.model(&weights), &plain, &rows, &BTreeMap::new()).expect("M").log_probabilities;
    let open = execute(&weights, &graph.program(&weights, true), &with_reference, &rows, &BTreeMap::new()).expect("program").log_probabilities;
    assert!(max(&kl_bits(&clean, &open)) > 1e-6, "the program equals M");
}

#[test]
fn device_path_runs_transcoder_features_as_the_host() {
    let f = fixture("graph_device_transcoder");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let mut weights = Weights::of(&library);
    let dir = std::env::temp_dir().join(format!("gam_mpd_graph_device_tc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    crate::test_support::transcoder_file(&dir.join("layer_1.safetensors"), 6, 8, 5);
    assert_eq!(weights.attach_transcoders(&dir).expect("attach"), 1);
    let mut program = full_program();
    let m1 = program.nodes.iter().position(|n| n.id == "m1").expect("m1");
    program.nodes[m1].pieces = vec![PieceIr { view: "transcoder".into(), layer: 1, kind: "feature".into(), index: Some(Index::Many(vec![0, 2, 3])) }];
    program.edges.retain(|e| !(e.from == "a0" && e.to == "m1"));
    let graph = Graph::parse(&program, &weights).expect("parse");
    weights.load_features(&graph).expect("features");
    let cf = counterfactuals(&f.sequences);
    let plain = Batch::new(&f.sequences).expect("batch");
    let mut with_reference = plain.clone();
    with_reference.reference = Some(std::sync::Arc::new(reference(&weights, &Batch::new(&cf).expect("cf batch")).expect("reference")));
    let rows: Vec<usize> = (0..plain.tokens.len()).collect();
    let mut state = crate::graph_device::DeviceState::new(Device::host());
    for (name, circuit, batch) in [("model", graph.model(&weights), &plain), ("edges", graph.program(&weights, true), &with_reference), ("nodes", graph.program(&weights, false), &with_reference)] {
        let host = execute(&weights, &circuit, batch, &rows, &BTreeMap::new()).expect("host");
        let job = crate::graph_device::Run { tokens: &batch.tokens, spans: &batch.spans, scored: &rows, swaps: &BTreeMap::new(), capture: false, reference: batch.reference.as_deref(), ops: &crate::graph::Interventions::default() };
        let device = run_on(&mut state, &weights, &circuit, &job).expect("device");
        let kl = max(&kl_bits(&host.log_probabilities, &device.log_probabilities));
        assert!(kl < 1e-9, "{name}: KL(host ‖ device) with transcoder features = {kl:e} bits");
    }
}

/// `M`'s normed attention input at `layer` on `sequences` (recorded as the checker records it) and
/// the batch.
fn attention_input(weights: &Weights, sequences: &[Vec<u32>], layer: usize) -> (ndarray::Array2<f64>, Batch) {
    let batch = Batch::new(sequences).expect("batch");
    let circuit = Graph::empty().model(weights);
    let unit = circuit.units.iter().position(|u| matches!(u.block, crate::graph::Block::Heads { layer: l, .. } if l == layer)).expect("heads");
    let run = crate::graph::execute_with(weights, &circuit, &batch, &[], &BTreeMap::new(), &crate::graph::Interventions::recording([2 * layer].into())).expect("run");
    (run.normed[&(unit, 0)].clone(), batch)
}

/// The pattern `block`'s first weighed head produces on each of `sequences`, as a pattern claim's
/// rows (row t: positions 0..=t).
fn produced_tables(weights: &Weights, block: &crate::graph::Block, sequences: &[Vec<u32>], layer: usize) -> Vec<serde_json::Value> {
    let (x_hat, batch) = attention_input(weights, sequences, layer);
    let produced = crate::graph::produced_patterns(weights, block, &x_hat, &batch.spans, &[]).expect("patterns");
    let (_, patterns) = produced.iter().find(|(w, _)| *w > 0.0).expect("a head the parts reach");
    patterns.iter().map(|a| serde_json::json!((0..a.nrows()).map(|t| (0..=t).map(|j| a[[t, j]]).collect::<Vec<f64>>()).collect::<Vec<_>>())).collect()
}

/// A behavior over `sequences` with their counterfactuals.
fn claims_behavior(sequences: &[Vec<u32>], cf: &[Vec<u32>]) -> Behavior {
    let prompts = sequences.iter().zip(cf).map(|(s, c)| Prompt { text: String::new(), token_ids: s.clone(), target_positions: vec![s.len() - 1], counterfactual: Some(Counterfactual { text: String::new(), token_ids: c.clone() }), attention_block: Vec::new() }).collect();
    Behavior { id: "tiny".into(), model: "tiny".into(), family: String::new(), description: String::new(), frequency: None, prompts, split: "train".into(), model_accuracy: None }
}

#[test]
fn attention_claims_are_checked_not_executed() {
    let f = fixture("graph_claims");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let weights = Weights::of(&library);
    let cf = counterfactuals(&f.sequences);
    // A node of layer 1's head 0 among the full program's nodes.
    let mut program = full_program();
    let a1 = program.nodes.iter().position(|n| n.id == "a1").expect("a1");
    program.nodes[a1].pieces = vec![PieceIr { view: "native".into(), layer: 1, kind: "head".into(), index: Some(Index::One(0)) }];
    let plain = Graph::parse(&program, &weights).expect("parse");
    let tables = |seqs: &[Vec<u32>]| produced_tables(&weights, &plain.blocks[a1], seqs, 1);
    let with = |claim: serde_json::Value| {
        let mut p = program.clone();
        p.nodes[a1].claim = Some(claim);
        Graph::parse(&p, &weights).expect("parse")
    };
    let truth = with(serde_json::json!({"op": "pattern", "prompts": tables(&f.sequences), "counterfactuals": tables(&cf)}));
    let offset = with(serde_json::json!({"op": "attend", "offset": 1}));
    let mut checker = Checker::new(weights.clone(), claims_behavior(&f.sequences, &cf)).expect("checker");
    let true_error = checker.claim_error(&truth).expect("claim");
    let false_error = checker.claim_error(&offset).expect("claim");
    assert!(true_error < 1e-9, "a true claim costs {true_error:e} bits per row");
    assert!(false_error > 1e-2, "a false claim costs {false_error:e} bits per row");
    assert_eq!(checker.claim_error(&plain).expect("no claim"), 0.0);
    // The claim changes nothing in execution (host and device), and the score charges it.
    let mut batch = Batch::new(&f.sequences).expect("batch");
    batch.reference = Some(std::sync::Arc::new(reference(&weights, &Batch::new(&cf).expect("cf batch")).expect("reference")));
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    let base = execute(&weights, &plain.program(&weights, true), &batch, &rows, &BTreeMap::new()).expect("plain").log_probabilities;
    let claimed = execute(&weights, &offset.program(&weights, true), &batch, &rows, &BTreeMap::new()).expect("claimed").log_probabilities;
    assert_eq!(base, claimed, "a claim changes execution");
    let mut state = crate::graph_device::DeviceState::new(Device::host());
    let job = crate::graph_device::Run { tokens: &batch.tokens, spans: &batch.spans, scored: &rows, swaps: &BTreeMap::new(), capture: false, reference: batch.reference.as_deref(), ops: &crate::graph::Interventions::default() };
    let device = run_on(&mut state, &weights, &offset.program(&weights, true), &job).expect("device").log_probabilities;
    assert!(max(&kl_bits(&base, &device)) < 1e-9, "device");
    assert_eq!(offset.opaque_numbers(&weights), plain.opaque_numbers(&weights), "a claim removes no weights from the price");
    let mut false_program = program.clone();
    false_program.nodes[a1].claim = Some(serde_json::json!({"op": "attend", "offset": 1}));
    let (score, _) = checker.score(&false_program, 4, 1, true, None).expect("score");
    assert!(score.claim_error_bits > 0.0 && (score.claim_error_bits - score.n * false_error).abs() < 1e-6 * score.claim_error_bits);
    // A claim needs queries and keys.
    let mut bad = program.clone();
    let m1 = bad.nodes.iter().position(|n| n.id == "m1").expect("m1");
    bad.nodes[m1].claim = Some(serde_json::json!({"op": "attend", "offset": 1}));
    assert!(Graph::parse(&bad, &weights).is_err(), "a claim on an MLP");
}

#[test]
fn attention_claims_on_vpd_parts_weigh_the_heads_they_reach() {
    use ndarray::{Array2, Axis};
    let f = fixture("graph_claims_vpd");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let mut weights = Weights::of(&library);
    // Layer 1's attention as one subcomponent per query, key, value and output coordinate (U the
    // identity's rows, V the matching columns of Wᵀ): subcomponents below the head width write
    // head 0 alone.
    let lw = &weights.layers[1];
    let stack = |m: &dyn Fn(&crate::graph::HeadWeights) -> &crate::graph::Stored| crate::graph::wide(ndarray::concatenate(Axis(0), &lw.heads.iter().map(|h| m(h).view()).collect::<Vec<_>>()).expect("stack").view());
    let (wq, wk, wv) = (stack(&|h| &h.query), stack(&|h| &*h.key), stack(&|h| &*h.value));
    let wo = crate::graph::wide(ndarray::concatenate(Axis(1), &lw.heads.iter().map(|h| h.output.view()).collect::<Vec<_>>()).expect("hstack").view());
    let exact = |w: &Array2<f64>| (Array2::<f64>::eye(w.nrows()), w.t().to_owned());
    let width = lw.heads[0].query.nrows();
    weights.vpd_attention.insert(1, crate::graph::VpdAttention { q: exact(&wq), k: exact(&wk), v: exact(&wv), o: (wo.t().to_owned(), Array2::<f64>::eye(wo.ncols())) });
    let cf = counterfactuals(&f.sequences);
    let vpd = |kind: &str, index: Vec<usize>| PieceIr { view: "vpd".into(), layer: 1, kind: kind.into(), index: Some(Index::Many(index)) };
    let program = |claim: Option<serde_json::Value>| Program { model: "tiny".into(), valid: true, nodes: vec![NodeIr { id: "QK".into(), pieces: vec![vpd("q_proj", (0..width).collect()), vpd("k_proj", (0..width / 2).collect())], claim }], ..Program::default() };
    let plain = Graph::parse(&program(None), &weights).expect("parse");
    // Only head 0 is reached (the other heads' query and key subcomponents are not named).
    let (x_hat, batch) = attention_input(&weights, &f.sequences, 1);
    let produced = crate::graph::produced_patterns(&weights, &plain.blocks[0], &x_hat, &batch.spans, &[]).expect("patterns");
    assert!(produced[0].0 > 0.0 && produced[1..].iter().all(|(w, _)| *w == 0.0), "head weights {:?}", produced.iter().map(|p| p.0).collect::<Vec<_>>());
    let tables = |seqs: &[Vec<u32>]| produced_tables(&weights, &plain.blocks[0], seqs, 1);
    let truth = Graph::parse(&program(Some(serde_json::json!({"op": "pattern", "prompts": tables(&f.sequences), "counterfactuals": tables(&cf)}))), &weights).expect("parse");
    let offset = Graph::parse(&program(Some(serde_json::json!({"op": "attend", "offset": 1}))), &weights).expect("parse");
    let mut checker = Checker::new(weights.clone(), claims_behavior(&f.sequences, &cf)).expect("checker");
    let (true_error, false_error) = (checker.claim_error(&truth).expect("claim"), checker.claim_error(&offset).expect("claim"));
    assert!(true_error < 1e-9 && false_error > 1e-2, "true {true_error:e}, false {false_error:e} bits per row");
    // Execution is the program without the claim, on the host and the device.
    let mut with_reference = Batch::new(&f.sequences).expect("batch");
    with_reference.reference = Some(std::sync::Arc::new(reference(&weights, &Batch::new(&cf).expect("cf batch")).expect("reference")));
    let rows: Vec<usize> = (0..with_reference.tokens.len()).collect();
    let base = execute(&weights, &plain.program(&weights, true), &with_reference, &rows, &BTreeMap::new()).expect("plain").log_probabilities;
    assert_eq!(base, execute(&weights, &offset.program(&weights, true), &with_reference, &rows, &BTreeMap::new()).expect("claimed").log_probabilities);
    let mut state = crate::graph_device::DeviceState::new(Device::host());
    let job = crate::graph_device::Run { tokens: &with_reference.tokens, spans: &with_reference.spans, scored: &rows, swaps: &BTreeMap::new(), capture: false, reference: with_reference.reference.as_deref(), ops: &crate::graph::Interventions::default() };
    let device = run_on(&mut state, &weights, &offset.program(&weights, true), &job).expect("device").log_probabilities;
    assert!(max(&kl_bits(&base, &device)) < 1e-9, "device");
    // A claim on value and output subcomponents alone has no pattern to check.
    let vo = Program { nodes: vec![NodeIr { id: "VO".into(), pieces: vec![vpd("v_proj", vec![0]), vpd("o_proj", vec![0])], claim: Some(serde_json::json!({"op": "attend", "offset": 1})) }], ..program(None) };
    assert!(Graph::parse(&vo, &weights).is_err());
}

#[test]
fn alignments_are_checked_by_interchange() {
    use crate::graph::{AlignmentIr, PairIr};
    let f = fixture("graph_alignments");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let weights = Weights::of(&library);
    let cf = counterfactuals(&f.sequences);
    let program = full_program();
    let graph = Graph::parse(&program, &weights).expect("parse");
    let m0 = program.nodes.iter().position(|n| n.id == "m0").expect("m0");
    // M's own answers at each base's last token with layer 0's MLP write taken from the next prompt.
    let n = f.sequences.len();
    let pairs: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    let circuit = graph.model(&weights);
    let bases = Batch::new(&pairs.iter().map(|&(i, _)| f.sequences[i].clone()).collect::<Vec<_>>()).expect("bases");
    let sources = Batch::new(&pairs.iter().map(|&(_, j)| f.sequences[j].clone()).collect::<Vec<_>>()).expect("sources");
    let written = execute(&weights, &circuit, &sources, &[], &BTreeMap::new()).expect("sources").writes;
    let rows: Vec<usize> = bases.spans.iter().map(|&(start, length)| start + length - 1).collect();
    let swapped = execute(&weights, &circuit, &bases, &rows, &[(m0, written[m0].clone().expect("m0 writes"))].into()).expect("swapped").log_probabilities;
    let top: Vec<u32> = swapped.outer_iter().map(|r| r.iter().enumerate().fold((0, f64::NEG_INFINITY), |best, (k, &x)| if x > best.1 { (k, x) } else { best }).0 as u32).collect();
    let vocabulary = weights.embedding.nrows() as u32;
    let bound = |answers: &[u32]| {
        let mut p = program.clone();
        p.alignments = vec![AlignmentIr { variable: "v".into(), nodes: vec!["m0".into()], pairs: pairs.iter().zip(answers).map(|(&(base, source), &a)| PairIr { base, source, answer: vec![a], ..PairIr::default() }).collect() }];
        p
    };
    let wrong: Vec<u32> = top.iter().map(|t| (t + 1) % vocabulary).collect();
    let mut checker = Checker::new(weights.clone(), claims_behavior(&f.sequences, &cf)).expect("checker");
    let true_error = checker.alignment_error(&Graph::parse(&bound(&top), &weights).expect("parse")).expect("alignment");
    let false_error = checker.alignment_error(&Graph::parse(&bound(&wrong), &weights).expect("parse")).expect("alignment");
    assert!(true_error < 1e-6, "M's own interchanged answers cost {true_error:e} bits per target");
    assert!(false_error > 1e-3, "wrong answers cost {false_error:e} bits per target");
    assert_eq!(checker.alignment_error(&graph).expect("no alignment"), 0.0);
    // A set of answers that holds M's top token costs nothing; a set without it pays.
    let with_sets = |sets: Vec<Vec<u32>>| {
        let mut p = bound(&top);
        for (pair, set) in p.alignments[0].pairs.iter_mut().zip(sets) {
            pair.answers = vec![set];
        }
        Graph::parse(&p, &weights).expect("parse")
    };
    let holding = checker.alignment_error(&with_sets(top.iter().zip(&wrong).map(|(t, w)| vec![*w, *t]).collect())).expect("sets");
    let missing = checker.alignment_error(&with_sets(wrong.iter().map(|w| vec![*w]).collect())).expect("sets");
    assert!(holding == 0.0 && (missing - false_error).abs() < 1e-9, "sets: holding {holding:e}, missing {missing:e} vs {false_error:e}");
    let (score, _) = checker.score(&bound(&wrong), 4, 1, true, None).expect("score");
    assert!((score.alignment_error_bits - score.n * false_error).abs() <= 1e-6 * score.alignment_error_bits, "the score charges N times the error");
    // A alignment names nodes of the program.
    let mut unknown = bound(&top);
    unknown.alignments[0].nodes = vec!["nowhere".into()];
    assert!(Graph::parse(&unknown, &weights).is_err());
}

#[test]
fn device_path_runs_head_operations_on_a_vpd_attention_as_the_host() {
    use crate::graph::{Donor, HeadRead, Interventions, execute_with};
    let f = fixture("graph_device_head_reads");
    let library = Library::new(&f.device, &f.device, &f.native, &f.layers, &f.artifact, 1 << 28, 64).expect("library");
    let mut weights = Weights::of(&library);
    let lw = &weights.layers[1];
    let stack = |m: &dyn Fn(&crate::graph::HeadWeights) -> &crate::graph::Stored| crate::graph::wide(ndarray::concatenate(ndarray::Axis(0), &lw.heads.iter().map(|h| m(h).view()).collect::<Vec<_>>()).expect("stack").view());
    let (wq, wk, wv) = (stack(&|h| &h.query), stack(&|h| &*h.key), stack(&|h| &*h.value));
    let wo = crate::graph::wide(ndarray::concatenate(ndarray::Axis(1), &lw.heads.iter().map(|h| h.output.view()).collect::<Vec<_>>()).expect("hstack").view());
    let exact_in = |w: &ndarray::Array2<f64>| (ndarray::Array2::eye(w.nrows()), w.t().to_owned());
    weights.vpd_attention.insert(1, crate::graph::VpdAttention { q: exact_in(&wq), k: exact_in(&wk), v: exact_in(&wv), o: (wo.t().to_owned(), ndarray::Array2::eye(wo.ncols())) });
    let piece = |layer: usize, view: &str, kind: &str, index: Option<Index>| PieceIr { view: view.into(), layer, kind: kind.into(), index };
    let all = |n: usize| Some(Index::Many((0..n).collect()));
    let nodes = vec![
        NodeIr { id: "a0".into(), pieces: vec![piece(0, "native", "head", None)], claim: None },
        NodeIr { id: "QKV".into(), pieces: vec![piece(1, "vpd", "q_proj", all(wq.nrows())), piece(1, "vpd", "k_proj", all(wk.nrows())), piece(1, "vpd", "v_proj", all(wv.nrows()))], claim: None },
        NodeIr { id: "O".into(), pieces: vec![piece(1, "vpd", "o_proj", all(wo.ncols()))], claim: None },
    ];
    let edge = |from: &str, to: &str| EdgeIr { from: from.into(), to: to.into(), route: "input".into() };
    let edges = vec![edge("embed", "a0"), edge("embed", "QKV"), edge("a0", "QKV"), edge("QKV", "O"), edge("embed", "logits"), edge("O", "logits")];
    let graph = Graph::parse(&Program { model: "tiny".into(), valid: true, nodes, edges, ..Program::default() }, &weights).expect("parse");
    let circuit = graph.program(&weights, true);
    let cf = counterfactuals(&f.sequences);
    let mut batch = Batch::new(&f.sequences).expect("batch");
    batch.reference = Some(std::sync::Arc::new(reference(&weights, &Batch::new(&cf).expect("cf batch")).expect("reference")));
    let mut donor_batch = Batch::new(&cf).expect("donor");
    donor_batch.reference = Some(std::sync::Arc::new(reference(&weights, &Batch::new(&f.sequences).expect("batch")).expect("reference")));
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    let donor_run = execute_with(&weights, &circuit, &donor_batch, &[], &BTreeMap::new(), &Interventions { record_reads: [2].into(), ..Interventions::default() }).expect("donor");
    let embed = crate::graph::wide(weights.embedding.select(ndarray::Axis(0), &donor_batch.tokens.iter().map(|t| *t as usize).collect::<Vec<_>>()).view());
    let ops = Interventions {
        head_reads: vec![(2, 1, HeadRead::Scale(0.5), vec![3, 7, 20]), (2, 0, HeadRead::Swap, vec![5, 6, 40])],
        record_reads: [2].into(),
        donor: Some(Donor { embed, writes: donor_run.writes.clone(), normed: donor_run.normed.clone(), reads: donor_run.reads.clone() }),
        ..Interventions::default()
    };
    let host = execute_with(&weights, &circuit, &batch, &rows, &BTreeMap::new(), &ops).expect("host");
    let mut state = crate::graph_device::DeviceState::new(Device::host());
    let job = crate::graph_device::Run { tokens: &batch.tokens, spans: &batch.spans, scored: &rows, swaps: &BTreeMap::new(), capture: false, reference: batch.reference.as_deref(), ops: &ops };
    let device = run_on(&mut state, &weights, &circuit, &job).expect("device");
    let kl = max(&kl_bits(&host.log_probabilities, &device.log_probabilities));
    assert!(kl < 1e-9, "KL(host ‖ device) under head operations = {kl:e} bits");
    let o = 2;
    let gap = (&host.reads[&o] - &device.reads[&o]).iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(gap < 1e-9, "recorded reads differ by {gap:e}");
    let plain = execute(&weights, &circuit, &batch, &rows, &BTreeMap::new()).expect("plain").log_probabilities;
    assert!(max(&kl_bits(&plain, &host.log_probabilities)) > 1e-9, "the operations change the run");
}

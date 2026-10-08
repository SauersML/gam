#![cfg(test)]
//! The graph checker's site operations (interchange's `SiteOp`, `graph::Interventions`) on the tiny
//! export (#2951): identical for `M` and a program that declares every piece, and equal to `M`'s
//! own weight edits and donor runs where those define the same experiment.
use crate::{
    graph::{Batch, Behavior, Checker, Circuit, Counterfactual, EdgeIr, Graph, NodeIr, PieceIr, Program, Prompt, SiteDraw, SiteUnits, WeightEdit, Weights, complements, execute, kl_bits, op_rows, reference, reference_under, run_sites, sample},
    import::import_language_model,
    interchange::{self, Family, Operation, SharedSite, SiteOp},
    library_mdl,
    library_readout::Library,
    operator_program::SlotValues,
    run_check::{layer_nodes, split_sites},
    test_support::tiny_export,
};
use gam_gpu::tensor::Device;
use ndarray::Array2;
use std::collections::BTreeMap;

const LAYERS: usize = 2;

fn model(tag: &str) -> (Weights, Vec<Vec<u32>>) {
    let dir = tiny_export(tag, LAYERS);
    let imported = import_language_model(&dir, 6, 12).expect("import");
    let native = split_sites(&imported.program).expect("split");
    let layers = layer_nodes(&native, LAYERS).expect("layers");
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
    let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
    let artifact = library_mdl::explanation(&native, &layers).expect("start library").artifact;
    let device = Device::host();
    let library = Library::new(&device, &device, &native, &layers, &artifact, 1 << 28, 64).expect("library");
    (Weights::of(&library), sequences)
}

/// Every shared site at typical norm 1 and four seeded directions.
fn units(weights: &Weights) -> SiteUnits {
    let (layers, heads) = (weights.layers.len(), weights.layers.iter().map(|l| l.heads.len()).sum::<usize>());
    let mut sites: Vec<SharedSite> = (0..2 * layers).map(SharedSite::Stream).chain((0..heads).map(SharedSite::Head)).collect();
    sites.extend((0..layers).map(SharedSite::Attention).chain((0..layers).map(SharedSite::Mlp)).chain((0..2 * layers).map(SharedSite::Input)));
    sites.push(SharedSite::Embedding);
    SiteUnits { typical: sites.into_iter().map(|s| (s, 1.0)).collect(), directions: interchange::seeded_directions(4, weights.embedding.ncols(), 3), pool: Vec::new() }
}

fn draw(family: Family, ops: &[(SharedSite, Operation)], position: usize, onward: bool) -> SiteDraw {
    SiteDraw { family, ops: ops.iter().map(|&(site, operation)| SiteOp { site, operation, onward }).collect(), position, length: 12 }
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
    for (r, reader) in order.iter().chain([&"logits".to_string()]).enumerate() {
        for w in ["embed".to_string()].into_iter().chain(order.iter().take(r).cloned()) {
            let routes: &[&str] = if reader.starts_with('a') { &["query", "key", "value"] } else { &["input"] };
            for route in routes {
                edges.push(EdgeIr { from: w.clone(), to: reader.clone(), route: (*route).into() });
            }
        }
    }
    Program { model: "tiny".into(), nodes, edges, python_tokens: 0, token_types: 0, source: String::new(), valid: true, error: None, standin: None, base: Vec::new(), explanation_tokens: 0, explanation_token_types: 0, bindings: Vec::new() }
}

fn max(values: &[f64]) -> f64 {
    values.iter().fold(0.0f64, |a, b| a.max(b.abs()))
}

struct Setup {
    weights: Weights,
    base: Batch,
    donor: Batch,
    rows: Vec<usize>,
    units: SiteUnits,
}

fn setup(tag: &str) -> Setup {
    let (weights, sequences) = model(tag);
    // Each sequence's donor is the next one (all of one length), as a counterfactual would be.
    let donors: Vec<Vec<u32>> = (0..sequences.len()).map(|i| sequences[(i + 1) % sequences.len()].clone()).collect();
    let base = Batch::new(&sequences).expect("batch");
    let rows: Vec<usize> = (0..base.tokens.len()).collect();
    let units = units(&weights);
    Setup { weights, base, donor: Batch::new(&donors).expect("donors"), rows, units }
}

impl Setup {
    fn run(&self, circuit: &Circuit, d: &SiteDraw) -> Array2<f64> {
        run_sites(&self.weights, circuit, (&self.base, &self.rows), Some(&self.donor), d, &self.units).expect("site run")
    }
}

/// Every family at every kind of site: the program that declares every piece and keeps every edge
/// responds as `M` does, routed by its edges or node by node.
#[test]
fn full_program_responds_as_the_model_to_site_operations() {
    let s = setup("graph_sites_full");
    let graph = Graph::parse(&full_program(), &s.weights).expect("parse");
    let model = graph.model(&s.weights);
    let push = |direction: usize| Operation::Push { direction, size: 1 };
    let draws = [
        draw(Family::Zero, &[(SharedSite::Head(1), Operation::Scale(0)), (SharedSite::Mlp(0), Operation::Scale(0))], 0, true),
        draw(Family::Scale, &[(SharedSite::Attention(0), Operation::Scale(2)), (SharedSite::Input(1), Operation::Scale(3)), (SharedSite::Stream(2), Operation::Scale(1))], 5, false),
        draw(Family::Push, &[(SharedSite::Stream(1), push(0)), (SharedSite::Input(2), push(1)), (SharedSite::Embedding, push(2)), (SharedSite::Mlp(1), push(3))], 3, true),
        draw(Family::Swap, &[(SharedSite::Head(2), Operation::Swap), (SharedSite::Stream(1), Operation::Swap), (SharedSite::Input(3), Operation::Swap), (SharedSite::Embedding, Operation::Swap)], 4, true),
        draw(Family::Cut, &[(SharedSite::Attention(0), Operation::Cut { to: 3 }), (SharedSite::Mlp(0), Operation::Cut { to: 2 })], 2, true),
    ];
    for d in &draws {
        let m = s.run(&model, d);
        for (name, circuit) in [("edges", graph.program(&s.weights, true)), ("nodes", graph.program(&s.weights, false))] {
            let p = s.run(&circuit, d);
            let kl = max(&kl_bits(&m, &p));
            assert!(kl < 1e-9, "{:?} ({name}): KL(M_e ‖ P_e) = {kl:e} bits", d.family);
        }
        let clean = execute(&s.weights, &model, &s.base, &s.rows, &BTreeMap::new()).expect("clean").log_probabilities;
        assert!(max(&kl_bits(&clean, &m)) > 1e-6, "{:?} changes nothing", d.family);
    }
}

/// A head's output zeroed at every token is the head removed from the weights (its output map
/// times 0), and an MLP's output scaled at every token is its down map scaled.
#[test]
fn site_scales_at_every_token_are_weight_edits() {
    let mut s = setup("graph_sites_edits");
    let model = Graph::empty().model(&s.weights);
    let heads = s.weights.layers[0].heads.len();
    let cases = [
        (draw(Family::Zero, &[(SharedSite::Head(heads), Operation::Scale(0))], 0, true), WeightEdit::Head { layer: 1, head: 0, factor: 0.0 }),
        (draw(Family::Scale, &[(SharedSite::Mlp(0), Operation::Scale(1))], 0, true), WeightEdit::Neurons { layer: 0, neurons: (0..s.weights.layers[0].mlp.as_ref().expect("an MLP").gate.nrows()).collect(), factor: interchange::SCALES[1] }),
    ];
    for (d, edit) in cases {
        let site = s.run(&model, &d);
        let restore = edit.apply(&mut s.weights).expect("edit");
        let edited = execute(&s.weights, &model, &s.base, &s.rows, &BTreeMap::new()).expect("edited").log_probabilities;
        restore.restore(&mut s.weights).expect("restore");
        let kl = max(&kl_bits(&edited, &site));
        assert!(kl < 1e-9, "{edit:?}: KL(weight edit ‖ site operation) = {kl:e} bits");
    }
}

/// The embeddings, or the stream after the last block, swapped at every token from the donor is
/// `M` run on the donor.
#[test]
fn swaps_at_every_token_are_the_donor_run() {
    let s = setup("graph_sites_swaps");
    let model = Graph::empty().model(&s.weights);
    let donor = execute(&s.weights, &model, &s.donor, &s.rows, &BTreeMap::new()).expect("donor").log_probabilities;
    for site in [SharedSite::Embedding, SharedSite::Stream(2 * LAYERS - 1)] {
        let swapped = s.run(&model, &draw(Family::Swap, &[(site, Operation::Swap)], 0, true));
        let kl = max(&kl_bits(&donor, &swapped));
        assert!(kl < 1e-9, "{site:?}: KL(donor run ‖ swap) = {kl:e} bits");
    }
}

#[test]
fn positions_map_onto_each_prompt() {
    let batch = Batch::new(&[vec![1; 12], vec![2; 5]]).expect("batch");
    assert_eq!(op_rows(&batch, 0, 512, true), (0..17).collect::<Vec<_>>());
    // 1 + 255 · 11 / 511 = 6 in the first sequence, 1 + 255 · 4 / 511 = 2 in the second.
    assert_eq!(op_rows(&batch, 256, 512, false), vec![6, 12 + 2]);
    assert_eq!(op_rows(&batch, 511, 512, true), vec![11, 16]);
}

#[test]
fn manifest_experiments_load() {
    let dir = std::env::temp_dir().join(format!("graph_sites_manifest_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let path = dir.join("MANIFEST.json");
    let ops = interchange::Patch::Ops { family: Family::Zero, ops: vec![SiteOp { site: SharedSite::Head(3), operation: Operation::Scale(0), onward: true }] };
    let experiments = vec![vec![
        interchange::Experiment { base: 0, source: 1, explained: vec![true; 4], patch: Some(ops), position: 7 },
        interchange::Experiment { base: 0, source: 0, explained: vec![true; 4], patch: None, position: 0 },
    ]];
    let manifest = serde_json::json!({"export": "x", "seed": 1, "typical": [[{"stream": 0}, 2.5], ["embedding", 0.5]], "experiments": experiments});
    std::fs::write(&path, manifest.to_string()).expect("write");
    let units = SiteUnits::manifest(&path, 512, 8).expect("manifest");
    assert_eq!(units.pool, vec![SiteDraw { family: Family::Zero, ops: vec![SiteOp { site: SharedSite::Head(3), operation: Operation::Scale(0), onward: true }], position: 7, length: 512 }]);
    assert_eq!(units.typical.get(&SharedSite::Stream(0)), Some(&2.5));
    assert_eq!(units.directions, interchange::seeded_directions(interchange::DIRECTIONS, 8, 1));
    std::fs::remove_dir_all(&dir).expect("the test directory is removed");
}

/// The draws depend on the seed and the measured targets alone (no program), and targeted edits
/// take the strongest pieces whole.
#[test]
fn the_experiments_come_from_the_seed_and_the_targets() {
    let (weights, _) = model("graph_sites_uniform");
    let pool = vec![draw(Family::Zero, &[(SharedSite::Head(0), Operation::Scale(0))], 0, true), draw(Family::Push, &[(SharedSite::Stream(1), Operation::Push { direction: 0, size: 0 })], 3, false)];
    let units = SiteUnits { pool, ..SiteUnits::default() };
    let neurons = weights.layers[1].mlp.as_ref().expect("an MLP").gate.nrows();
    let targets = crate::graph::Targets {
        pieces: vec![(crate::graph::Block::Heads { layer: 1, heads: vec![0] }, 0.5), (crate::graph::Block::Neurons { layer: 1, neurons: (0..neurons).collect() }, 0.2)],
        cuts: vec![(SiteOp { site: SharedSite::Attention(0), operation: Operation::Cut { to: 3 }, onward: true }, 0.3)],
    };
    let a = sample(&weights, true, 40, 11, &targets, &units);
    assert_eq!(a, sample(&weights, true, 40, 11, &targets, &units));
    assert_eq!(&a[..2], &[crate::graph::Experiment::Clean, crate::graph::Experiment::Counterfactual]);
    assert!(a.iter().any(|e| e.family().starts_with("site_")));
    let targeted: Vec<&crate::graph::Experiment> = a.iter().skip(3).step_by(2).collect();
    assert!(targeted.iter().all(|e| match e {
        crate::graph::Experiment::Edit { edit: WeightEdit::Head { layer: 1, head: 0, .. }, targeted: true } => true,
        crate::graph::Experiment::Edit { edit: WeightEdit::Neurons { layer: 1, neurons: n, .. }, targeted: true } => n.len() == neurons,
        crate::graph::Experiment::Sites { draw } => draw.ops.len() == 1 && (draw.ops[0].site == SharedSite::Head(weights.layers[0].heads.len()) || draw.ops[0].site == SharedSite::Mlp(1) || draw.ops[0].operation == Operation::Cut { to: 3 }),
        _ => false,
    }), "{targeted:?}");
}

/// The empty program under counterfactual stand-ins and a site operation is `M` on the
/// counterfactual under the same operation: its stand-ins come from that run (`reference_under`).
#[test]
fn counterfactual_stand_ins_take_the_same_site_operations() {
    let s = setup("graph_sites_reference");
    let model = Graph::empty().model(&s.weights);
    let empty = Graph::empty().program(&s.weights, true);
    let heads = s.weights.layers[0].heads.len();
    let push = Operation::Push { direction: 1, size: 2 };
    for d in [
        draw(Family::Zero, &[(SharedSite::Head(heads + 1), Operation::Scale(0)), (SharedSite::Input(1), Operation::Scale(0))], 0, true),
        draw(Family::Push, &[(SharedSite::Stream(0), push), (SharedSite::Mlp(0), push), (SharedSite::Embedding, push)], 4, true),
        draw(Family::Scale, &[(SharedSite::Attention(1), Operation::Scale(3)), (SharedSite::Stream(2), Operation::Scale(1))], 6, false),
    ] {
        let on_partner = run_sites(&s.weights, &model, (&s.donor, &s.rows), None, &d, &s.units).expect("M on x'");
        let mut base = Batch::new(&s.base.sequences()).expect("batch");
        base.reference = Some(std::sync::Arc::new(reference_under(&s.weights, &s.donor, &d, &s.units).expect("reference")));
        let program = run_sites(&s.weights, &empty, (&base, &s.rows), None, &d, &s.units).expect("empty program");
        let kl = max(&kl_bits(&on_partner, &program));
        assert!(kl < 1e-9, "{:?}: KL(M_e(x') ‖ empty P_e(x)) = {kl:e} bits", d.family);
    }
}

/// A head zeroed at every token in the counterfactual run records what the run with the head
/// removed from the weights records (its read, every later read and MLP).
#[test]
fn a_zeroed_head_in_the_reference_is_the_removed_head() {
    let mut s = setup("graph_sites_reference_edit");
    let heads = s.weights.layers[0].heads.len();
    let zeroed = reference_under(&s.weights, &s.donor, &draw(Family::Zero, &[(SharedSite::Head(heads), Operation::Scale(0))], 0, true), &s.units).expect("reference");
    let restore = WeightEdit::Head { layer: 1, head: 0, factor: 0.0 }.apply(&mut s.weights).expect("edit");
    let removed = reference(&s.weights, &s.donor).expect("reference");
    restore.restore(&mut s.weights).expect("restore");
    let gap = |a: &Array2<f64>, b: &Array2<f64>| (a - b).iter().fold(0.0f64, |m, v| m.max(v.abs()));
    for l in 0..LAYERS {
        for h in 0..s.weights.layers[l].heads.len() {
            assert!(gap(&zeroed.reads[l][h], &removed.reads[l][h]) < 1e-9, "head {l}.{h}'s read");
        }
        assert!(gap(&zeroed.active[l], &removed.active[l]) < 1e-9 && gap(&zeroed.mlp[l], &removed.mlp[l]) < 1e-9, "layer {l}'s MLP");
    }
}

#[test]
fn typical_norms_cover_every_site_and_manifests_round_trip() {
    let s = setup("graph_sites_typical");
    let typical = SiteUnits::measure_typical(&s.weights, &s.base.sequences()).expect("typical");
    let (sites, _) = SiteUnits::shared(&s.weights);
    assert_eq!(typical.keys().copied().collect::<Vec<_>>(), sites);
    assert!(typical.values().all(|v| v.is_finite() && *v > 0.0), "{typical:?}");
    let dir = std::env::temp_dir().join(format!("graph_sites_write_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let path = dir.join("MANIFEST_tiny.json");
    let families = [Family::Swap, Family::Zero, Family::Scale, Family::Push, Family::Cut];
    let written = SiteUnits::write_manifest(&path, "tiny", &s.weights, &s.base.sequences(), &families, 40, 5, 64).expect("write");
    assert!(SiteUnits::write_manifest(&path, "tiny", &s.weights, &s.base.sequences(), &families, 40, 5, 64).is_err(), "a manifest is immutable");
    let read = SiteUnits::manifest(&path, 512, s.weights.embedding.ncols()).expect("read");
    assert_eq!(read.pool, written.pool);
    assert_eq!(read.typical, written.typical);
    assert_eq!(read.directions, written.directions);
    assert!(read.pool.iter().all(|d| d.length == 64 && !d.ops.is_empty()));
    std::fs::remove_dir_all(&dir).expect("the test directory is removed");
}


/// A behavior of the tiny export's sequences, each with the next one as its counterfactual.
fn behavior(sequences: &[Vec<u32>]) -> Behavior {
    let prompts = sequences
        .iter()
        .enumerate()
        .map(|(i, s)| Prompt { text: String::new(), token_ids: s.clone(), target_positions: vec![s.len() - 2, s.len() - 1], counterfactual: Some(Counterfactual { text: String::new(), token_ids: sequences[(i + 1) % sequences.len()].clone() }), attention_block: Vec::new() })
        .collect();
    Behavior { id: "tiny".into(), model: "tiny".into(), family: String::new(), description: String::new(), frequency: None, prompts, split: "train".into(), model_accuracy: None }
}

/// Scoring programs together (parallel runs, M once per experiment, edits grouped) gives each the
/// score it gets alone, site operations included; an invalid program is flagged and scored as the
/// empty program.
#[test]
fn batch_scores_equal_single_scores() {
    let (weights, sequences) = model("graph_sites_batch");
    let units = units(&weights);
    let mut checker = Checker::new(weights, behavior(&sequences)).expect("checker");
    let pool = vec![
        draw(Family::Zero, &[(SharedSite::Head(1), Operation::Scale(0))], 0, true),
        draw(Family::Swap, &[(SharedSite::Mlp(0), Operation::Swap), (SharedSite::Input(2), Operation::Swap)], 3, true),
        draw(Family::Push, &[(SharedSite::Stream(1), Operation::Push { direction: 0, size: 1 })], 5, false),
    ];
    checker.sites = SiteUnits { pool, ..units };
    let node = |id: &str, layer: usize, kind: &str, index: usize| NodeIr { id: id.into(), pieces: vec![PieceIr { view: "native".into(), layer, kind: kind.into(), index: Some(crate::graph::Index::One(index)) }], claim: None };
    let mut partial = Program { model: "tiny".into(), valid: true, ..Program::default() };
    partial.nodes = vec![node("h", 1, "head", 0), node("m", 0, "mlp", 2)];
    partial.edges = vec![EdgeIr { from: "embed".into(), to: "m".into(), route: "input".into() }, EdgeIr { from: "m".into(), to: "h".into(), route: "value".into() }, EdgeIr { from: "h".into(), to: "logits".into(), route: "input".into() }];
    let mut invalid = partial.clone();
    invalid.edges.push(EdgeIr { from: "h".into(), to: "m".into(), route: "input".into() });
    let programs = vec![full_program(), partial.clone(), Program { model: "tiny".into(), valid: true, ..Program::default() }, invalid];
    let batch = checker.score_batch(&programs, 30, 3, true, None, 4).expect("batch");
    assert!(batch.iter().any(|(_, m)| m.iter().any(|x| x.0.family().starts_with("site_"))), "no site operation drawn");
    for (program, (together, measured)) in programs.iter().zip(&batch) {
        let (alone, _) = checker.score(program, 30, 3, true, None).expect("alone");
        assert_eq!(together.valid, alone.valid);
        assert!((together.exec_error_bits - alone.exec_error_bits).abs() <= 1e-6 * alone.exec_error_bits.max(1.0), "batch {} vs alone {}", together.exec_error_bits, alone.exec_error_bits);
        assert!(measured.iter().all(|x| x.2.as_ref().is_some_and(|c| c.tokens.iter().all(|t| t.len() == 4))), "every experiment has its reader candidates");
    }
    assert!(!batch[3].0.valid && batch[3].0.error.is_some());
    assert!(batch[0].0.exec_error_bits / batch[0].0.n < 1e-9, "the full program is M");
}

/// Given a disk cache (`Checker::disk_cache`), a second checker of the same behavior reads `M`'s
/// outcomes from the first one's files and scores the same.
#[test]
fn the_disk_cache_serves_a_second_checker() {
    let (weights, sequences) = model("graph_sites_disk");
    let dir = std::env::temp_dir().join(format!("graph_sites_disk_{}", std::process::id()));
    let mut first = Checker::new(weights.clone(), behavior(&sequences)).expect("checker");
    first.disk_cache = Some(dir.clone());
    let (a, _) = first.score(&Program { model: "tiny".into(), valid: true, ..Program::default() }, 12, 4, true, None).expect("score");
    let files = std::fs::read_dir(&dir).map(|d| d.flatten().flat_map(|e| std::fs::read_dir(e.path()).into_iter().flatten()).count()).unwrap_or(0);
    let mut second = Checker::new(weights, behavior(&sequences)).expect("checker");
    second.disk_cache = Some(dir.clone());
    let (b, _) = second.score(&Program { model: "tiny".into(), valid: true, ..Program::default() }, 12, 4, true, None).expect("score");
    assert!(files > 0, "no outcome written");
    std::fs::remove_dir_all(&dir).expect("the cache directory is removed");
    assert_eq!(a.exec_error_bits, b.exec_error_bits);
}

/// A VPD subcomponent edit changes the matrix by `(factor − 1) U_i ⊗ V_i` (M runs as under that
/// rank-one change) and scales the subcomponent's factor; restoring returns both.
#[test]
fn subcomponent_edits_are_their_rank_one_changes() {
    let mut s = setup("graph_sites_vpd_edit");
    let (hidden, width) = s.weights.layers[0].mlp.as_ref().expect("an MLP").gate.dim();
    let wave = |rows: usize, cols: usize, phase: f64| Array2::from_shape_fn((rows, cols), |(i, j)| 0.1 * ((i * 7 + j * 3) as f64 + phase).sin());
    s.weights.vpd.insert(0, crate::graph::VpdMlp { fc_u: wave(5, hidden, 0.3), fc_v: wave(width, 5, 1.1), down_u: wave(4, width, 2.0), down_v: wave(hidden, 4, 0.7) });
    let model = Graph::empty().model(&s.weights);
    for (down, matrix) in [(false, crate::graph::Matrix::Gate), (true, crate::graph::Matrix::Down)] {
        let before = s.weights.clone();
        let vpd = &s.weights.vpd[&0];
        let (u, v) = if down { (vpd.down_u.row(2).to_vec(), vpd.down_v.column(2).to_vec()) } else { (vpd.fc_u.row(2).to_vec(), vpd.fc_v.column(2).to_vec()) };
        let rank_one = WeightEdit::RankOne { layer: 0, head: None, matrix, u: u.iter().map(|x| -0.5 * x).collect(), v };
        let restore = rank_one.apply(&mut s.weights).expect("rank one");
        let expected = execute(&s.weights, &model, &s.base, &s.rows, &BTreeMap::new()).expect("run").log_probabilities;
        restore.restore(&mut s.weights).expect("restore");
        let restore = WeightEdit::Subcomponents { layer: 0, down, indices: vec![2], factor: 0.5 }.apply(&mut s.weights).expect("subcomponents");
        let got = execute(&s.weights, &model, &s.base, &s.rows, &BTreeMap::new()).expect("run").log_probabilities;
        let factor = if down { s.weights.vpd[&0].down_u.row(2).to_vec() } else { s.weights.vpd[&0].fc_u.row(2).to_vec() };
        restore.restore(&mut s.weights).expect("restore");
        assert!(max(&kl_bits(&expected, &got)) < 1e-12, "down {down}: the subcomponent edit is not its rank-one change");
        assert!(factor.iter().zip(&u).all(|(a, b)| (a - 0.5 * b).abs() < 1e-15), "the factor is scaled");
        assert_eq!(s.weights.layers[0].mlp.as_ref().map(|m| (m.gate.clone(), m.out.clone())), before.layers[0].mlp.as_ref().map(|m| (m.gate.clone(), m.out.clone())));
        assert_eq!((s.weights.vpd[&0].fc_u.clone(), s.weights.vpd[&0].down_u.clone()), (before.vpd[&0].fc_u.clone(), before.vpd[&0].down_u.clone()));
    }
}

/// A rank-one edit of a key a group of query heads shares changes it for every head of the group,
/// and restoring returns them all.
#[test]
fn shared_keys_are_edited_together() {
    let (mut weights, _) = model("graph_sites_gqa");
    let key = weights.layers[0].heads[0].key.clone();
    weights.layers[0].heads[1].key = key.clone();
    let (rows, cols) = key.dim();
    let edit = WeightEdit::RankOne { layer: 0, head: Some(0), matrix: crate::graph::Matrix::Key, u: vec![0.1; rows], v: vec![0.2; cols] };
    let restore = edit.apply(&mut weights).expect("edit");
    assert_ne!(weights.layers[0].heads[0].key, key);
    assert_eq!(weights.layers[0].heads[1].key, weights.layers[0].heads[0].key, "the sharing head is edited alike");
    if weights.layers[0].heads.len() > 2 {
        assert_ne!(weights.layers[0].heads[2].key, weights.layers[0].heads[0].key);
    }
    restore.restore(&mut weights).expect("restore");
    assert_eq!((weights.layers[0].heads[0].key.clone(), weights.layers[0].heads[1].key.clone()), (key.clone(), key));
}

/// A VPD attention subcomponent edit adds `(factor − 1) U_i ⊗ V_i` to the stacked query map (and to
/// the side-by-side output columns), split over the heads, and restores exactly.
#[test]
fn attention_subcomponent_edits_split_over_heads() {
    let (mut weights, _) = model("graph_sites_vpd_attention");
    let heads = weights.layers[0].heads.len();
    let (dh, width) = weights.layers[0].heads[0].query.dim();
    let wave = |rows: usize, cols: usize, phase: f64| Array2::from_shape_fn((rows, cols), |(i, j)| 0.1 * ((i * 5 + j * 3) as f64 + phase).cos());
    let (q, o) = ((wave(3, heads * dh, 0.2), wave(width, 3, 0.9)), (wave(3, width, 1.3), wave(heads * dh, 3, 0.4)));
    weights.vpd_attention.insert(0, crate::graph::VpdAttention { q: q.clone(), k: q.clone(), v: q.clone(), o: o.clone() });
    let before = weights.clone();
    for (map, (u, v)) in [(0usize, &q), (3, &o)] {
        let restore = WeightEdit::AttnSubcomponents { layer: 0, map, indices: vec![1], factor: 3.0 }.apply(&mut weights).expect("edit");
        let delta = |r: usize, c: usize| 2.0 * u[[1, r]] * v[[c, 1]];
        for h in 0..heads {
            let (now, was) = if map == 0 { (&weights.layers[0].heads[h].query, &before.layers[0].heads[h].query) } else { (&weights.layers[0].heads[h].output, &before.layers[0].heads[h].output) };
            let gap = now.indexed_iter().map(|((r, c), x)| {
                let (row, col) = if map == 0 { (h * dh + r, c) } else { (r, h * dh + c) };
                (f64::from(*x) - f64::from(was[[r, c]]) - delta(row, col)).abs()
            }).fold(0.0f64, f64::max);
            // The edited weights are stored in float32.
            assert!(gap < 1e-6, "map {map} head {h}: off by {gap:e}");
        }
        restore.restore(&mut weights).expect("restore");
        for h in 0..heads {
            assert_eq!(weights.layers[0].heads[h].query, before.layers[0].heads[h].query);
            assert_eq!(weights.layers[0].heads[h].output, before.layers[0].heads[h].output);
        }
        assert_eq!(weights.vpd_attention[&0].q.0, q.0);
        assert_eq!(weights.vpd_attention[&0].o.0, o.0);
    }
}

/// Every program of a behavior faces the same experiments, its strongest pieces measured once, and
/// with `uniform_seeds` m a seed and the seed plus m score alike.
#[test]
fn every_program_faces_the_same_experiments() {
    let (weights, sequences) = model("graph_sites_fixed_set");
    let units = units(&weights);
    let mut checker = Checker::new(weights, behavior(&sequences)).expect("checker");
    checker.sites = units;
    checker.uniform_seeds = Some(4);
    let program = full_program();
    let (a, _) = checker.score(&program, 10, 3, true, None).expect("score");
    let (b, _) = checker.score(&program, 10, 7, true, None).expect("score");
    assert_eq!(a.exec_error_bits, b.exec_error_bits);
    assert_eq!(a.per_family.keys().collect::<Vec<_>>(), b.per_family.keys().collect::<Vec<_>>());
    let empty = Program { model: "tiny".into(), valid: true, ..Program::default() };
    let both = checker.score_batch(&[program, empty], 10, 3, true, None, 0).expect("batch");
    let experiments = |m: &[crate::graph::Measured]| m.iter().map(|x| x.0.clone()).collect::<Vec<_>>();
    assert_eq!(experiments(&both[0].1), experiments(&both[1].1));
    let targets = checker.targets().expect("targets");
    assert!(targets.pieces.windows(2).all(|w| w[0].1 >= w[1].1) && targets.cuts.windows(2).all(|w| w[0].1 >= w[1].1));
    assert!(targets.pieces.iter().any(|(b, _)| matches!(b, crate::graph::Block::Heads { .. })) && !targets.cuts.is_empty());
}

/// A score holds its own M outcomes: a cache budget that keeps none of them still scores, and
/// scores as a large one does.
#[test]
fn a_tiny_cache_still_scores() {
    let (weights, sequences) = model("graph_sites_tiny_cache");
    let mut small = Checker::new(weights.clone(), behavior(&sequences)).expect("checker");
    small.cache_bytes = 1;
    let mut large = Checker::new(weights, behavior(&sequences)).expect("checker");
    let empty = Program { model: "tiny".into(), valid: true, ..Program::default() };
    let (a, _) = small.score(&empty, 12, 2, true, None).expect("small cache");
    let (b, _) = large.score(&empty, 12, 2, true, None).expect("large cache");
    assert_eq!(a.exec_error_bits, b.exec_error_bits);
}

/// With room for only some counterfactual runs, the runs kept after a score are the ones the next
/// score of the behavior reads first: the second score reuses every kept run instead of remaking it.
#[test]
fn kept_counterfactual_runs_serve_the_next_score() {
    let (weights, sequences) = model("graph_sites_kept_runs");
    let empty = Program { model: "tiny".into(), valid: true, ..Program::default() };
    let mut large = Checker::new(weights.clone(), behavior(&sequences)).expect("checker");
    let (whole, _) = large.score(&empty, 12, 2, true, None).expect("large budget");
    let bytes = |c: &Checker| -> Vec<usize> { c.references.lock().expect("lock").iter().map(|(_, cell)| cell.get().and_then(|r| r.as_ref().ok()).map_or(0, |r| r.bytes())).collect() };
    let all = bytes(&large);
    assert!(all.len() >= 2, "the empty program reads {} counterfactual runs", all.len());
    let mut small = Checker::new(weights, behavior(&sequences)).expect("checker");
    small.reference_bytes = all.iter().sum::<usize>() / 2;
    let kept = |c: &Checker| -> Vec<(String, usize)> { c.references.lock().expect("lock").iter().map(|(k, cell)| (k.clone(), std::sync::Arc::as_ptr(cell) as usize)).collect() };
    let (first, _) = small.score(&empty, 12, 2, true, None).expect("first score");
    let after_first = kept(&small);
    assert!(!after_first.is_empty() && after_first.len() < all.len(), "{} of {} runs kept", after_first.len(), all.len());
    assert!(bytes(&small).iter().sum::<usize>() <= small.reference_bytes);
    let (second, _) = small.score(&empty, 12, 2, true, None).expect("second score");
    let mut after_second = kept(&small);
    let mut expected = after_first.clone();
    after_second.sort();
    expected.sort();
    assert_eq!(after_second, expected, "the kept runs were remade or replaced");
    assert_eq!(first.exec_error_bits, whole.exec_error_bits);
    assert_eq!(second.exec_error_bits, whole.exec_error_bits);
}

/// A head's site operation on a program whose attention is in VPD's view acts on the head's read
/// before o_proj: it runs (there is no unit of that head alone), and zeroing the head changes the
/// program's outcome.
#[test]
fn head_operations_reach_a_vpd_view_attention() {
    let mut s = setup("graph_sites_vpd_heads");
    let heads = s.weights.layers[0].heads.len();
    let (dh, width) = s.weights.layers[0].heads[0].query.dim();
    let wave = |rows: usize, cols: usize, phase: f64| Array2::from_shape_fn((rows, cols), |(i, j)| 0.1 * ((i * 5 + j * 3) as f64 + phase).cos());
    let qkv = (wave(3, heads * dh, 0.2), wave(width, 3, 0.9));
    s.weights.vpd_attention.insert(0, crate::graph::VpdAttention { q: qkv.clone(), k: qkv.clone(), v: qkv, o: (wave(3, width, 1.3), wave(heads * dh, 3, 0.4)) });
    let mut program = Program { model: "tiny".into(), valid: true, ..Program::default() };
    program.nodes = vec![NodeIr { id: "o".into(), pieces: vec![PieceIr { view: "vpd".into(), layer: 0, kind: "o_proj".into(), index: Some(crate::graph::Index::One(0)) }], claim: None }];
    program.edges = vec![EdgeIr { from: "o".into(), to: "logits".into(), route: "input".into() }];
    let graph = Graph::parse(&program, &s.weights).expect("parse");
    let circuit = graph.program(&s.weights, true);
    // The donor (the counterfactual) takes its stand-ins from the prompts.
    let mut donor = s.donor.clone();
    donor.reference = Some(std::sync::Arc::new(reference(&s.weights, &s.base).expect("reference")));
    let run = |d: &SiteDraw| {
        let mut base = Batch::new(&s.base.sequences()).expect("batch");
        base.reference = Some(std::sync::Arc::new(reference_under(&s.weights, &s.donor, d, &s.units).expect("reference")));
        run_sites(&s.weights, &circuit, (&base, &s.rows), Some(&donor), d, &s.units).expect("site run")
    };
    let zero = run(&draw(Family::Zero, &[(SharedSite::Head(1), Operation::Scale(0))], 0, true));
    let push = run(&draw(Family::Push, &[(SharedSite::Stream(0), Operation::Push { direction: 0, size: 0 })], 0, true));
    let swap = run(&draw(Family::Swap, &[(SharedSite::Head(0), Operation::Swap)], 3, true));
    assert!(max(&kl_bits(&zero, &push)) > 1e-9 && zero.iter().chain(swap.iter()).all(|v| v.is_finite() || *v == f64::NEG_INFINITY));
}

/// A VPD MLP declared whole, its subcomponents and both remainders ("rest"), computes the MLP
/// itself: the full program with it equals `M`; leaving out a remainder does not.
#[test]
fn vpd_remainders_are_declarable_pieces() {
    let (mut weights, sequences) = model("graph_sites_vpd_rest");
    let (hidden, width) = weights.layers[0].mlp.as_ref().expect("an MLP").gate.dim();
    let wave = |rows: usize, cols: usize, phase: f64| Array2::from_shape_fn((rows, cols), |(i, j)| 0.1 * ((i * 7 + j * 3) as f64 + phase).sin());
    weights.vpd.insert(0, crate::graph::VpdMlp { fc_u: wave(5, hidden, 0.3), fc_v: wave(width, 5, 1.1), down_u: wave(4, width, 2.0), down_v: wave(hidden, 4, 0.7) });
    let vpd = |kind: &str, index: crate::graph::Index| PieceIr { view: "vpd".into(), layer: 0, kind: kind.into(), index: Some(index) };
    let with = |rest: bool| {
        let mut p = full_program();
        let m0 = p.nodes.iter_mut().find(|n| n.id == "m0").expect("m0");
        m0.pieces = vec![vpd("c_fc", crate::graph::Index::Many((0..5).collect())), vpd("down_proj", crate::graph::Index::Many((0..4).collect()))];
        if rest {
            m0.pieces.push(vpd("c_fc", crate::graph::Index::Name("rest".into())));
            m0.pieces.push(vpd("down_proj", crate::graph::Index::Name("rest".into())));
        }
        p
    };
    let batch = Batch::new(&sequences).expect("batch");
    let rows: Vec<usize> = (0..batch.tokens.len()).collect();
    let mut cf = batch.clone();
    cf.reference = None;
    let m = execute(&weights, &Graph::empty().model(&weights), &batch, &rows, &BTreeMap::new()).expect("M").log_probabilities;
    let mut referenced = batch.clone();
    referenced.reference = Some(std::sync::Arc::new(reference(&weights, &Batch::new(&sequences.iter().rev().cloned().collect::<Vec<_>>()).expect("cf")).expect("reference")));
    let run = |p: &Program| {
        let graph = Graph::parse(p, &weights).expect("parse");
        execute(&weights, &graph.program(&weights, true), &referenced, &rows, &BTreeMap::new()).expect("program").log_probabilities
    };
    assert!(max(&kl_bits(&m, &run(&with(true)))) < 1e-9, "subcomponents and remainders are the MLP");
    assert!(max(&kl_bits(&m, &run(&with(false)))) > 1e-9, "without the remainders the MLP is incomplete");
    let numbers = Graph::parse(&with(true), &weights).expect("parse").opaque_numbers(&weights) - Graph::parse(&with(false), &weights).expect("parse").opaque_numbers(&weights);
    assert_eq!(numbers, 2 * hidden * width, "a remainder costs its matrix");
}

/// A second checker given the same memo directory reads the first one's targets instead of
/// measuring them again, and scores a program exactly as the first did.
#[test]
fn memos_serve_a_later_checker() {
    let (weights, sequences) = model("graph_sites_memo");
    let dir = std::env::temp_dir().join(format!("graph_memo_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut program = Program { model: "tiny".into(), valid: true, ..Program::default() };
    program.nodes = vec![NodeIr { id: "h".into(), pieces: vec![PieceIr { view: "native".into(), layer: 0, kind: "head".into(), index: Some(crate::graph::Index::One(0)) }], claim: None }];
    program.edges = vec![EdgeIr { from: "embed".into(), to: "h".into(), route: "input".into() }, EdgeIr { from: "h".into(), to: "logits".into(), route: "input".into() }];
    let mut first = Checker::new(weights.clone(), behavior(&sequences)).expect("checker");
    first.memo_dir = Some(dir.clone());
    let targets = first.targets().expect("targets");
    let (a, _) = first.score(&program, 12, 2, true, None).expect("first score");
    let mut second = Checker::new(weights, behavior(&sequences)).expect("checker");
    second.memo_dir = Some(dir.clone());
    let memo = second.targets().expect("memoized targets");
    assert_eq!(serde_json::to_string(&memo.pieces).expect("json"), serde_json::to_string(&targets.pieces).expect("json"));
    assert_eq!(serde_json::to_string(&memo.cuts).expect("json"), serde_json::to_string(&targets.cuts).expect("json"));
    let (b, _) = second.score(&program, 12, 2, true, None).expect("second score");
    assert_eq!(a.total_bits, b.total_bits);
    assert_eq!(a.exec_error_bits, b.exec_error_bits);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Necessity: the full program predicts `M` with its nodes at their counterfactual values (no
/// error); a program naming layer 0's heads alone, whose information reaches the logits through the
/// later pieces it leaves out, pays; the empty program claims nothing and pays none (measured, its
/// complement is `M` on the prompt and so is its swapped run) while it pays the whole sufficiency
/// error.
#[test]
fn necessity_pays_for_left_out_mediators() {
    // Float32 weights' rounding, as in graph_tests.
    let f32_kl = 64.0 / 16_777_216.0 / std::f64::consts::LN_2;
    let (weights, sequences) = model("graph_sites_necessity");
    let units = units(&weights);
    // Counterfactuals apart from every prompt (one token changed), so each sequence's partner is
    // its own prompt or counterfactual.
    let mut distinct = behavior(&sequences);
    for p in &mut distinct.prompts {
        if let Some(c) = p.counterfactual.as_mut() {
            c.token_ids = p.token_ids.clone();
            c.token_ids[5] = (c.token_ids[5] + 1) % 11;
        }
    }
    let mut checker = Checker::new(weights, distinct).expect("checker");
    checker.sites = SiteUnits { pool: vec![draw(Family::Zero, &[(SharedSite::Mlp(0), Operation::Scale(0))], 0, true), draw(Family::Push, &[(SharedSite::Stream(1), Operation::Push { direction: 0, size: 1 })], 5, false)], ..units };
    let empty = Program { model: "tiny".into(), valid: true, ..Program::default() };
    let mut partial = Program { model: "tiny".into(), valid: true, ..Program::default() };
    partial.nodes = vec![NodeIr { id: "a0".into(), pieces: vec![piece(0, "head")], claim: None }];
    partial.edges = ["query", "key", "value"].iter().map(|r| EdgeIr { from: "embed".into(), to: "a0".into(), route: (*r).into() }).chain(["embed", "a0"].iter().map(|w| EdgeIr { from: (*w).into(), to: "logits".into(), route: "input".into() })).collect();
    let scores = checker.score_batch(&[full_program(), partial, empty], 30, 3, true, None, 0).expect("scores");
    let (full, partial, none) = (&scores[0].0, &scores[1].0, &scores[2].0);
    let families: Vec<&String> = full.per_family.keys().filter(|k| k.starts_with("necessity_")).collect();
    assert!(families.iter().any(|k| *k == "necessity_clean") && families.iter().any(|k| k.starts_with("necessity_edit") || k.starts_with("necessity_rank")) && families.iter().any(|k| k.starts_with("necessity_site")), "{families:?}");
    assert!(full.necessity_error_bits / full.n < f32_kl, "full program necessity {:e} bits per token", full.necessity_error_bits / full.n);
    assert!(partial.necessity_error_bits / partial.n > 1e-3 && partial.necessity_error_bits > 100.0 * full.necessity_error_bits, "partial {:e}, full {:e} bits per token", partial.necessity_error_bits / partial.n, full.necessity_error_bits / full.n);
    assert!((partial.total_bits - partial.exec_error_bits - partial.necessity_error_bits - partial.claim_error_bits - partial.binding_error_bits - partial.complexity_bits).abs() < 1e-6 * partial.total_bits);
    assert_eq!(none.necessity_error_bits, 0.0);
    assert!(none.exec_error_bits > full.exec_error_bits && none.exec_error_bits / none.n > 1e-3);
    let empty_graph = Graph::empty();
    let circuit = empty_graph.program(&checker.weights, true);
    let targets = checker.targets().expect("targets");
    let experiments = sample(&checker.weights, true, 30, 3, &targets, &checker.sites);
    let measured = checker.necessity_runs(&[(&empty_graph, &circuit)], &complements(&experiments)).expect("empty necessity");
    assert!(measured[0].len() >= 3 && measured[0].iter().all(|kl| kl.as_ref().is_some_and(|kl| max(kl) < f32_kl)), "{:?}", measured[0].iter().map(|kl| kl.as_ref().map(|kl| max(kl))).collect::<Vec<_>>());
}

/// Deletion (the default with a VPD view attached): a program naming every VPD subcomponent and
/// remainder of layer 0's MLP and every other piece is `M` (no sufficiency or necessity error);
/// the empty program's error on the clean prompts is `M`'s KL to the all-deleted model (a zero
/// stream); a program naming layer 0's heads alone pays necessity, as its heads' information
/// reaches the logits through the later pieces it deletes; and the counterfactual semantics stay
/// available per program.
#[test]
fn deleting_programs_score_named_parts_alone() {
    let f32_kl = 64.0 / 16_777_216.0 / std::f64::consts::LN_2;
    let (mut weights, sequences) = model("graph_sites_delete");
    let (hidden, width) = weights.layers[0].mlp.as_ref().expect("an MLP").gate.dim();
    let wave = |rows: usize, cols: usize, phase: f64| Array2::from_shape_fn((rows, cols), |(i, j)| 0.1 * ((i * 7 + j * 3) as f64 + phase).sin());
    weights.vpd.insert(0, crate::graph::VpdMlp { fc_u: wave(5, hidden, 0.3), fc_v: wave(width, 5, 1.1), down_u: wave(4, width, 2.0), down_v: wave(hidden, 4, 0.7) });
    let vpd = |kind: &str, index: crate::graph::Index| PieceIr { view: "vpd".into(), layer: 0, kind: kind.into(), index: Some(index) };
    let mut full = full_program();
    full.nodes.iter_mut().find(|n| n.id == "m0").expect("m0").pieces = ["c_fc", "down_proj"].iter().flat_map(|k| [vpd(k, crate::graph::Index::Many((0..if *k == "c_fc" { 5 } else { 4 }).collect())), vpd(k, crate::graph::Index::Name("rest".into()))]).collect();
    let mut partial = Program { model: "tiny".into(), valid: true, ..Program::default() };
    partial.nodes = vec![NodeIr { id: "a0".into(), pieces: vec![piece(0, "head")], claim: None }];
    partial.edges = ["query", "key", "value"].iter().map(|r| EdgeIr { from: "embed".into(), to: "a0".into(), route: (*r).into() }).chain(["embed", "a0"].iter().map(|w| EdgeIr { from: (*w).into(), to: "logits".into(), route: "input".into() })).collect();
    let counterfactual = Program { standin: Some("counterfactual".into()), ..partial.clone() };
    let empty = Program { model: "tiny".into(), valid: true, ..Program::default() };
    let d = weights.width();
    let mut checker = Checker::new(weights, behavior(&sequences)).expect("checker");
    // N large enough that every block stays exact: the semantics alone, not the precision search.
    let scores = checker.score_batch(&[full, partial, counterfactual, empty], 24, 5, true, Some(1e15), 0).expect("scores");
    let (full, partial, counterfactual, empty) = (&scores[0].0, &scores[1].0, &scores[2].0, &scores[3].0);
    assert_eq!((full.standin.as_str(), counterfactual.standin.as_str(), empty.standin.as_str()), ("delete", "counterfactual", "delete"));
    assert!(full.exec_error_bits / full.n < f32_kl && full.necessity_error_bits / full.n < f32_kl, "full program: sufficiency {:e}, necessity {:e} bits per token", full.exec_error_bits / full.n, full.necessity_error_bits / full.n);
    assert!(partial.necessity_error_bits / partial.n > 1e-3, "partial necessity {:e} bits per token", partial.necessity_error_bits / partial.n);
    assert!(partial.per_family.keys().all(|k| !k.starts_with("necessity_site")), "deletion measures necessity on clean prompts and weight edits");
    assert!((partial.exec_error_bits - counterfactual.exec_error_bits).abs() > 1e-6 * partial.exec_error_bits, "the stand-ins differ");
    let m = checker.model_outcome(&Graph::empty(), &crate::graph::Experiment::Clean).expect("M");
    let deleted = crate::graph::log_probabilities(&checker.weights, &Array2::zeros((m.nrows(), d))).expect("all deleted");
    let expected = kl_bits(&m, &deleted).iter().sum::<f64>() / m.nrows() as f64;
    let clean = empty.per_family["clean"].mean_kl_bits;
    assert!((clean - expected).abs() < 1e-9 * expected.max(1.0), "empty program on the clean prompts {clean} vs KL(M ‖ all deleted) {expected}");
}

/// The score's complexity is what a reader takes in (design_v2 section 2): each listed part costs
/// `log2 V` bits, a VPD remainder its matrix's rank in names, each edge `log2(3 (n + 1)²)`; a shared
/// base's nodes are priced apart, outside the total; weights are reported but not scored; and `N` is
/// the number of tokens the experiment set scores.
#[test]
fn complexity_is_what_a_reader_takes_in() {
    let (mut weights, sequences) = model("graph_sites_complexity");
    let (hidden, width) = weights.layers[0].mlp.as_ref().expect("an MLP").gate.dim();
    let wave = |rows: usize, cols: usize, phase: f64| Array2::from_shape_fn((rows, cols), |(i, j)| 0.1 * ((i * 7 + j * 3) as f64 + phase).sin());
    weights.vpd.insert(0, crate::graph::VpdMlp { fc_u: wave(5, hidden, 0.3), fc_v: wave(width, 5, 1.1), down_u: wave(4, width, 2.0), down_v: wave(hidden, 4, 0.7) });
    let vocabulary = weights.vocabulary();
    let vpd = |kind: &str, index: crate::graph::Index| PieceIr { view: "vpd".into(), layer: 0, kind: kind.into(), index: Some(index) };
    let mut program = Program { model: "tiny".into(), valid: true, ..Program::default() };
    program.nodes = vec![
        NodeIr { id: "f".into(), pieces: vec![vpd("c_fc", crate::graph::Index::Many(vec![0, 3])), vpd("down_proj", crate::graph::Index::Name("rest".into()))], claim: None },
        NodeIr { id: "h".into(), pieces: vec![PieceIr { view: "native".into(), layer: 1, kind: "head".into(), index: Some(crate::graph::Index::One(1)) }], claim: None },
    ];
    program.edges = vec![EdgeIr { from: "embed".into(), to: "f".into(), route: "input".into() }, EdgeIr { from: "f".into(), to: "logits".into(), route: "input".into() }, EdgeIr { from: "h".into(), to: "logits".into(), route: "input".into() }];
    let with_base = Program { base: vec!["f".into()], ..program.clone() };
    let mut checker = Checker::new(weights, behavior(&sequences)).expect("checker");
    let scores = checker.score_batch(&[program, with_base], 6, 2, true, None, 0).expect("scores");
    let (plain, based) = (&scores[0].0, &scores[1].0);
    let name = (vocabulary as f64).log2();
    let edge = (3.0 * 9.0f64).log2();
    // Two c_fc subcomponents, the down_proj remainder (rank min(width, hidden)) and one head.
    assert_eq!(plain.parts, 2 + width.min(hidden) + 1);
    assert!((plain.structure_bits - (plain.parts as f64 * name + 3.0 * edge)).abs() < 1e-9, "structure {} bits", plain.structure_bits);
    assert_eq!(plain.base_bits, 0.0);
    // The base node f, its edges and the edge into it are charged apart.
    assert!((based.base_bits - ((2 + width.min(hidden)) as f64 * name + 2.0 * edge)).abs() < 1e-9 && (based.structure_bits - (name + edge)).abs() < 1e-9);
    assert!((plain.total_bits - plain.exec_error_bits - plain.necessity_error_bits - plain.claim_error_bits - plain.binding_error_bits - plain.complexity_bits).abs() < 1e-6 * plain.total_bits);
    assert!(plain.opaque_numbers > 0);
    let scored: usize = scores[0].1.iter().map(|m| m.1.len()).sum();
    assert_eq!(plain.n, scored as f64, "N is the tokens the experiments score");
}

/// A shared base (`Checker::set_base`): its nodes join every program, connected to every node by
/// implied edges that cost nothing, so a base of every piece with no declared edge is `M`; its
/// parts are priced apart; necessity keeps it (the empty program pays none); a program naming some
/// of its parts takes them over and still runs `M`; an id the base holds makes a program invalid,
/// and an invalid program is scored as the base alone; a base that does not parse is refused and
/// the previous one kept.
#[test]
fn a_shared_base_joins_every_program() {
    let f32_kl = 64.0 / 16_777_216.0 / std::f64::consts::LN_2;
    let (mut weights, sequences) = model("graph_sites_base");
    let (hidden, width) = weights.layers[0].mlp.as_ref().expect("an MLP").gate.dim();
    let wave = |rows: usize, cols: usize, phase: f64| Array2::from_shape_fn((rows, cols), |(i, j)| 0.1 * ((i * 7 + j * 3) as f64 + phase).sin());
    weights.vpd.insert(0, crate::graph::VpdMlp { fc_u: wave(5, hidden, 0.3), fc_v: wave(width, 5, 1.1), down_u: wave(4, width, 2.0), down_v: wave(hidden, 4, 0.7) });
    let name = (weights.vocabulary() as f64).log2();
    let vpd = |kind: &str, index: crate::graph::Index| PieceIr { view: "vpd".into(), layer: 0, kind: kind.into(), index: Some(index) };
    let mut base = full_program();
    base.edges.clear();
    base.nodes.iter_mut().find(|n| n.id == "m0").expect("m0").pieces = ["c_fc", "down_proj"].iter().flat_map(|k| [vpd(k, crate::graph::Index::Many((0..if *k == "c_fc" { 5 } else { 4 }).collect())), vpd(k, crate::graph::Index::Name("rest".into()))]).collect();
    let empty = Program { model: "tiny".into(), valid: true, ..Program::default() };
    // Two c_fc subcomponents the base also holds, fed by embed (the base's earlier nodes feed it by implication).
    let mut taking = empty.clone();
    taking.nodes = vec![NodeIr { id: "p".into(), pieces: vec![vpd("c_fc", crate::graph::Index::Many(vec![0, 1]))], claim: None }];
    taking.edges = vec![EdgeIr { from: "embed".into(), to: "p".into(), route: "input".into() }];
    let mut clash = empty.clone();
    clash.nodes = vec![NodeIr { id: "a1".into(), pieces: vec![piece(1, "head")], claim: None }];
    let invalid = Program { valid: false, error: Some("untraceable".into()), ..empty.clone() };
    let mut checker = Checker::new(weights, behavior(&sequences)).expect("checker");
    let (bits, parts) = checker.set_base(Some(base.clone())).expect("base");
    assert!(parts > 0 && (bits - parts as f64 * name).abs() < 1e-9 * bits, "a base of no declared edges costs its part names: {bits} bits, {parts} parts");
    let scores = checker.score_batch(&[empty, taking, clash, invalid], 12, 4, true, None, 0).expect("scores");
    let (none, taking, clash, invalid) = (&scores[0].0, &scores[1].0, &scores[2].0, &scores[3].0);
    assert!(none.valid && none.exec_error_bits / none.n < f32_kl, "the base of every piece is M: {:e} bits per token", none.exec_error_bits / none.n);
    assert_eq!((none.necessity_error_bits, none.structure_bits), (0.0, 0.0));
    assert!((none.base_bits - bits).abs() < 1e-9 * bits);
    assert!(taking.valid && taking.exec_error_bits / taking.n < f32_kl, "the program and the rest of the base are M: {:e}", taking.exec_error_bits / taking.n);
    assert!(taking.necessity_error_bits > 0.0, "deleting the program's c_fc subcomponents moves M");
    assert!((taking.base_bits - (bits - 2.0 * name)).abs() < 1e-9 * bits, "the program's two parts leave the base's price");
    assert!(!clash.valid && clash.error.as_deref().is_some_and(|e| e.contains("a1")));
    assert!(!invalid.valid && (invalid.exec_error_bits - none.exec_error_bits).abs() <= 1e-12 * none.exec_error_bits.max(1.0));
    // A base that does not parse is refused; the previous one stays.
    let mut broken = base;
    broken.nodes[0].pieces[0].layer = 9;
    assert!(checker.set_base(Some(broken)).is_err());
    assert!(checker.base.as_ref().is_some_and(|b| b.nodes[0].pieces[0].layer == 0));
}

//! The graph checker's site operations (interchange's `SiteOp`, `graph::Interventions`) on the tiny
//! export (#2951): identical for `M` and a program that declares every piece, and equal to `M`'s
//! own weight edits and donor runs where those define the same experiment.
use crate::{
    graph::{Batch, Behavior, Checker, Circuit, Counterfactual, EdgeIr, Graph, NodeIr, PieceIr, Program, Prompt, SiteDraw, SiteUnits, Stats, WeightEdit, Weights, execute, kl_bits, op_rows, reference, reference_under, run_sites, sample},
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
        nodes.push(NodeIr { id: format!("a{l}"), pieces: vec![piece(l, "head")], rule: None });
        nodes.push(NodeIr { id: format!("m{l}"), pieces: vec![piece(l, "mlp")], rule: None });
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
    Program { model: "tiny".into(), nodes, edges, python_tokens: 0, token_types: 0, source: String::new(), valid: true, error: None, standin: None }
}

fn max(values: &[f64]) -> f64 {
    values.iter().fold(0.0f64, |a, b| a.max(b.abs()))
}

struct Setup {
    weights: Weights,
    stats: Stats,
    base: Batch,
    donor: Batch,
    rows: Vec<usize>,
    units: SiteUnits,
}

fn setup(tag: &str) -> Setup {
    let (weights, sequences) = model(tag);
    // Each sequence's donor is the next one (all of one length), as a counterfactual would be.
    let donors: Vec<Vec<u32>> = (0..sequences.len()).map(|i| sequences[(i + 1) % sequences.len()].clone()).collect();
    let stats = Stats::measure(&weights, &sequences).expect("stats");
    let base = Batch::new(&sequences).expect("batch");
    let rows: Vec<usize> = (0..base.tokens.len()).collect();
    let units = units(&weights);
    Setup { weights, stats, base, donor: Batch::new(&donors).expect("donors"), rows, units }
}

impl Setup {
    fn run(&self, circuit: &Circuit, d: &SiteDraw) -> Array2<f64> {
        run_sites(&self.weights, &self.stats, circuit, (&self.base, &self.rows), Some(&self.donor), d, &self.units).expect("site run")
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
        let clean = execute(&s.weights, &s.stats, &model, &s.base, &s.rows, &BTreeMap::new(), false).expect("clean").log_probabilities;
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
        let edited = execute(&s.weights, &s.stats, &model, &s.base, &s.rows, &BTreeMap::new(), false).expect("edited").log_probabilities;
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
    let donor = execute(&s.weights, &s.stats, &model, &s.donor, &s.rows, &BTreeMap::new(), false).expect("donor").log_probabilities;
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
    std::fs::remove_dir_all(&dir).ok();
}

/// The behavior's half of the draws depends on the seed alone: two programs face the same
/// experiments there, so `M`'s cached outcomes serve both.
#[test]
fn the_behaviors_half_is_the_same_for_every_program() {
    let (weights, _) = model("graph_sites_uniform");
    let pool = vec![draw(Family::Zero, &[(SharedSite::Head(0), Operation::Scale(0))], 0, true), draw(Family::Push, &[(SharedSite::Stream(1), Operation::Push { direction: 0, size: 0 })], 3, false)];
    let full = Graph::parse(&full_program(), &weights).expect("parse");
    let units = SiteUnits { pool, ..SiteUnits::default() };
    let a = sample(&weights, &Graph::empty(), false, 40, 11, &[], &units);
    let b = sample(&weights, &full, false, 40, 11, &[], &units);
    // Clean first, then the behavior's draws at even k.
    let fixed = |v: &[crate::graph::Experiment]| v.iter().skip(1).step_by(2).cloned().collect::<Vec<_>>();
    assert_eq!(fixed(&a), fixed(&b));
    assert!(fixed(&a).iter().any(|e| e.family().starts_with("site_")));
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
        let on_partner = run_sites(&s.weights, &s.stats, &model, (&s.donor, &s.rows), None, &d, &s.units).expect("M on x'");
        let mut base = Batch::new(&s.base.sequences()).expect("batch");
        base.reference = Some(std::sync::Arc::new(reference_under(&s.weights, &s.stats, &s.donor, &d, &s.units).expect("reference")));
        let program = run_sites(&s.weights, &s.stats, &empty, (&base, &s.rows), None, &d, &s.units).expect("empty program");
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
    let zeroed = reference_under(&s.weights, &s.stats, &s.donor, &draw(Family::Zero, &[(SharedSite::Head(heads), Operation::Scale(0))], 0, true), &s.units).expect("reference");
    let restore = WeightEdit::Head { layer: 1, head: 0, factor: 0.0 }.apply(&mut s.weights).expect("edit");
    let removed = reference(&s.weights, &s.stats, &s.donor).expect("reference");
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
    let typical = SiteUnits::measure_typical(&s.weights, &s.stats, &s.base.sequences()).expect("typical");
    let (sites, _) = SiteUnits::shared(&s.weights);
    assert_eq!(typical.keys().copied().collect::<Vec<_>>(), sites);
    assert!(typical.values().all(|v| v.is_finite() && *v > 0.0), "{typical:?}");
    let dir = std::env::temp_dir().join(format!("graph_sites_write_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let path = dir.join("MANIFEST_tiny.json");
    let families = [Family::Swap, Family::Zero, Family::Scale, Family::Push, Family::Cut];
    let written = SiteUnits::write_manifest(&path, "tiny", &s.weights, &s.stats, &s.base.sequences(), &families, 40, 5, 64).expect("write");
    assert!(SiteUnits::write_manifest(&path, "tiny", &s.weights, &s.stats, &s.base.sequences(), &families, 40, 5, 64).is_err(), "a manifest is immutable");
    let read = SiteUnits::manifest(&path, 512, s.weights.embedding.ncols()).expect("read");
    assert_eq!(read.pool, written.pool);
    assert_eq!(read.typical, written.typical);
    assert_eq!(read.directions, written.directions);
    assert!(read.pool.iter().all(|d| d.length == 64 && !d.ops.is_empty()));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn aimed_sites_are_the_programs() {
    let (weights, _) = model("graph_sites_aimed");
    let heads = weights.layers[0].heads.len();
    let mut program = Program { model: "tiny".into(), valid: true, ..Program::default() };
    program.nodes = vec![NodeIr { id: "h".into(), pieces: vec![PieceIr { view: "native".into(), layer: 1, kind: "head".into(), index: Some(crate::graph::Index::One(0)) }], rule: None }];
    let graph = Graph::parse(&program, &weights).expect("parse");
    let aimed = SiteUnits::default().aimed_sites(&weights, &graph);
    assert_eq!(aimed, vec![SharedSite::Stream(2), SharedSite::Head(heads), SharedSite::Attention(1), SharedSite::Input(2)]);
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
    let node = |id: &str, layer: usize, kind: &str, index: usize| NodeIr { id: id.into(), pieces: vec![PieceIr { view: "native".into(), layer, kind: kind.into(), index: Some(crate::graph::Index::One(index)) }], rule: None };
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

/// With `GRAPH_DISK_CACHE` set, a second checker of the same behavior reads `M`'s outcomes from
/// the first one's files and scores the same.
#[test]
fn the_disk_cache_serves_a_second_checker() {
    let (weights, sequences) = model("graph_sites_disk");
    let dir = std::env::temp_dir().join(format!("graph_sites_disk_{}", std::process::id()));
    // SAFETY: the variable is read by checkers this test makes; tests that read it run nowhere else.
    unsafe { std::env::set_var("GRAPH_DISK_CACHE", &dir) };
    let mut first = Checker::new(weights.clone(), behavior(&sequences)).expect("checker");
    let (a, _) = first.score(&Program { model: "tiny".into(), valid: true, ..Program::default() }, 12, 4, true, None).expect("score");
    let files = std::fs::read_dir(&dir).map(|d| d.flatten().flat_map(|e| std::fs::read_dir(e.path()).into_iter().flatten()).count()).unwrap_or(0);
    let mut second = Checker::new(weights, behavior(&sequences)).expect("checker");
    let (b, _) = second.score(&Program { model: "tiny".into(), valid: true, ..Program::default() }, 12, 4, true, None).expect("score");
    unsafe { std::env::remove_var("GRAPH_DISK_CACHE") };
    std::fs::remove_dir_all(&dir).ok();
    assert!(files > 0, "no outcome written");
    assert_eq!(a.exec_error_bits, b.exec_error_bits);
}

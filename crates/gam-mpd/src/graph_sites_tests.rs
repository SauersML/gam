//! The graph checker's site operations (interchange's `SiteOp`, `graph::Interventions`) on the tiny
//! export (#2951): identical for `M` and a program that declares every piece, and equal to `M`'s
//! own weight edits and donor runs where those define the same experiment.
use crate::{
    graph::{Batch, Circuit, EdgeIr, Graph, NodeIr, PieceIr, Program, SiteDraw, SiteUnits, Stats, WeightEdit, Weights, execute, kl_bits, op_rows, run_sites, sample},
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
    Program { model: "tiny".into(), nodes, edges, python_tokens: 0, token_types: 0, source: String::new(), valid: true, error: None }
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
    let a = sample(&weights, &Graph::empty(), false, 40, 11, &[], &pool);
    let b = sample(&weights, &full, false, 40, 11, &[], &pool);
    // Clean first, then the behavior's draws at even k.
    let fixed = |v: &[crate::graph::Experiment]| v.iter().skip(1).step_by(2).cloned().collect::<Vec<_>>();
    assert_eq!(fixed(&a), fixed(&b));
    assert!(fixed(&a).iter().any(|e| e.family().starts_with("site_")));
}

//! The graph checker of the graph oracle (#2951): a program the oracle writes, parsed into nodes
//! bound to pieces of `M`'s weights and declared edges between them, executed against `M` under
//! experiments applied verbatim to both, and scored in bits.
//!
//! # Semantics
//!
//! A piece is a slice of `M`'s matrices: a head (its rows of the query, key and value maps and its
//! columns of the output map) or an MLP neuron (its gate and up rows and its down column). A node
//! is a set of pieces at one site (one layer's attention or one layer's MLP). A writer `w` (a node,
//! or `embed`) feeds a reader's route (a head's query, key or value input, an MLP's input, or the
//! logits) through a declared edge.
//!
//! * Stand-ins. Every piece has a stored average input over the behavior's prompts, measured once
//!   on `M`: a neuron its layer's normalized stream `x / r` (`r = √(mean(x²) + ε)`), a head its
//!   attention-weighted normalized stream `Σ_j a_tj x_j / r_j`, both averaged over every token of
//!   the prompts; `embed` the token frequencies. A piece's stand-in write is the piece applied with
//!   the current (possibly edited) weights to that input, so weight edits reach stand-ins verbatim.
//! * Execution. Undeclared pieces write their stand-ins. A declared node's route input is the
//!   stand-in stream at its site (`embed`'s and every earlier piece's stand-in write) plus, over its
//!   declared incoming edges on that route, the writer's actual write minus its stand-in write;
//!   the node computes on its inputs with each norm at its input's own RMS. With every edge kept
//!   (node-level routing, and `M` itself) the input is the actual stream.
//! * `M` is the same circuit with every piece computing and every edge kept, its units listing the
//!   program's nodes first, so a node swap or an edge cut names the same pieces in both.
//!
//! # Score (bits)
//!
//! Execution error `N · mean KL(M_e ‖ P_e)` over the experiments' target tokens (full vocabulary,
//! log base 2); code `python_tokens · log2(token_types)`; opaque numbers (every weight a declared
//! node reads and every stand-in average) at `½ log2 N` each. The reader term is added by
//! `bench/oracle/graph/score.py`.
//!
//! Every run is the host's float64 execution of the blocks `Library` holds for the start library
//! (`library_mdl::explanation`), which equals `M`.
use crate::{
    library_readout::Library,
    operator_program::{Law, Rotary, rms_scale},
    tiled_attention::{probabilities, rotate},
};
use ndarray::{Array1, Array2, Axis, s};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

// ------------------------------------------------------------------------------ the IR

/// The program IR the `mech` tracer writes (`~/mpd-data/team/graph/design.txt` section 5).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Program {
    pub model: String,
    #[serde(default)]
    pub nodes: Vec<NodeIr>,
    #[serde(default)]
    pub edges: Vec<EdgeIr>,
    #[serde(default)]
    pub python_tokens: usize,
    #[serde(default)]
    pub token_types: usize,
    #[serde(default)]
    pub source: String,
    #[serde(default = "yes")]
    pub valid: bool,
    #[serde(default)]
    pub error: Option<String>,
}

fn yes() -> bool {
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeIr {
    pub id: String,
    pub pieces: Vec<PieceIr>,
    #[serde(default)]
    pub rule: Option<serde_json::Value>,
}

/// One address: `index` a unit, a list of units, or absent for every unit of `kind` in the layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PieceIr {
    pub view: String,
    pub layer: usize,
    pub kind: String,
    #[serde(default)]
    pub index: Option<Index>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Index {
    One(usize),
    Many(Vec<usize>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeIr {
    pub from: String,
    pub to: String,
    pub route: String,
}

/// A reader's route: a head's query, key or value input, an MLP's input, the logits' input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Route {
    Query,
    Key,
    Value,
    Input,
}

impl Route {
    fn parse(name: &str) -> Result<Self, String> {
        Ok(match name {
            "query" => Self::Query,
            "key" => Self::Key,
            "value" => Self::Value,
            "input" => Self::Input,
            other => return Err(format!("unknown route {other}")),
        })
    }

    /// The slot of the route in a unit's inputs: heads read 0, 1, 2; an MLP and the logits read 0.
    fn slot(self) -> usize {
        match self {
            Self::Query | Self::Input => 0,
            Self::Key => 1,
            Self::Value => 2,
        }
    }
}

// ------------------------------------------------------------------------------ weights

/// An RMS norm: its gain and epsilon.
#[derive(Clone, Debug)]
pub struct Norm {
    pub gain: Array1<f64>,
    pub epsilon: f64,
}

impl Norm {
    /// `x / r` per row, `r` the row's RMS: the normalized stream before the gain.
    fn unit(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut out = x.clone();
        for (mut row, x) in out.outer_iter_mut().zip(x.outer_iter()) {
            row *= rms_scale(x, self.epsilon);
        }
        out
    }

    /// `γ ⊙ x / r` per row.
    fn apply(&self, x: &Array2<f64>) -> Array2<f64> {
        self.unit(x) * &self.gain.view().insert_axis(Axis(0))
    }
}

/// One head: its query and key maps with their head norms, its value map (head width × width,
/// reading the normed stream) and its output columns (width × head width).
#[derive(Clone, Debug)]
pub struct HeadWeights {
    pub query: Array2<f64>,
    pub query_norm: Option<(Array1<f64>, f64)>,
    pub key: Array2<f64>,
    pub key_norm: Option<(Array1<f64>, f64)>,
    pub value: Array2<f64>,
    pub output: Array2<f64>,
    pub scale: f64,
    pub rotary: Option<Rotary>,
    pub causal: bool,
}

/// One MLP: `h = φ(G x̂ + b)` (times `U x̂ + c` when gated), written by `D h`.
#[derive(Clone, Debug)]
pub struct MlpWeights {
    pub gate: Array2<f64>,
    pub bias: Array1<f64>,
    pub up: Option<Array2<f64>>,
    pub up_bias: Array1<f64>,
    pub out: Array2<f64>,
    pub law: Law,
}

#[derive(Clone, Debug)]
pub struct LayerWeights {
    pub attention: Norm,
    pub heads: Vec<HeadWeights>,
    pub mlp_norm: Norm,
    pub mlp: Option<MlpWeights>,
}

/// `M`'s weights on the host: per layer its blocks, the final norm's epsilon, the unembedding with
/// the final norm's gain folded in (vocabulary × width) and the embedding (vocabulary × width).
#[derive(Clone, Debug)]
pub struct Weights {
    pub layers: Vec<LayerWeights>,
    pub final_norm: Norm,
    pub unembedding: Array2<f64>,
    pub embedding: Array2<f64>,
}

impl Weights {
    /// The blocks of `library` (the start library of `M` equals `M`).
    pub fn of(library: &Library) -> Self {
        library.graph_weights()
    }

    fn width(&self) -> usize {
        self.embedding.ncols()
    }

    fn neurons(&self, layer: usize) -> usize {
        self.layers[layer].mlp.as_ref().map_or(0, |m| m.gate.nrows())
    }
}

// ------------------------------------------------------------------------------ resolved graph

/// The pieces of one site: some heads of a layer's attention, or some neurons of its MLP.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub enum Block {
    Heads { layer: usize, heads: Vec<usize> },
    Neurons { layer: usize, neurons: Vec<usize> },
}

impl Block {
    /// The read site: `2l` a layer's attention, `2l + 1` its MLP; the logits read at `2L`.
    fn site(&self) -> usize {
        match self {
            Self::Heads { layer, .. } => 2 * layer,
            Self::Neurons { layer, .. } => 2 * layer + 1,
        }
    }

    fn routes(&self) -> &'static [Route] {
        match self {
            Self::Heads { .. } => &[Route::Query, Route::Key, Route::Value],
            Self::Neurons { .. } => &[Route::Input],
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Heads { heads, .. } => heads.is_empty(),
            Self::Neurons { neurons, .. } => neurons.is_empty(),
        }
    }
}

/// A writer: `embed` or a unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Writer {
    Embed,
    Unit(usize),
}

/// The writers a route reads beyond the stand-in stream: exactly these, or all writers before it
/// but these.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Incoming {
    Only(BTreeSet<Writer>),
    AllBut(BTreeSet<Writer>),
}

impl Incoming {
    fn all() -> Self {
        Self::AllBut(BTreeSet::new())
    }

    /// The route without writer `w`'s actual write (it reads `w`'s stand-in).
    fn cut(&mut self, w: Writer) {
        match self {
            Self::Only(set) => {
                set.remove(&w);
            }
            Self::AllBut(set) => {
                set.insert(w);
            }
        }
    }
}

/// One unit of a circuit: its pieces and whether it computes (else it writes its stand-in).
#[derive(Clone, Debug)]
pub struct Unit {
    pub block: Block,
    pub computes: bool,
    /// Per route slot, what it reads.
    pub routes: [Incoming; 3],
}

/// A program or `M` as units in site order of computation plus the logits' input. The program's
/// declared nodes are units `0..nodes` in both.
#[derive(Clone, Debug)]
pub struct Circuit {
    pub units: Vec<Unit>,
    pub logits: Incoming,
    pub nodes: usize,
}

/// A parsed program: its nodes' blocks by id and its edges as (writer, reader, route), the reader
/// `None` for the logits.
#[derive(Clone, Debug)]
pub struct Graph {
    pub ids: Vec<String>,
    pub blocks: Vec<Block>,
    pub edges: Vec<(Writer, Option<usize>, Route)>,
}

fn indices(index: &Option<Index>, count: usize, what: &str) -> Result<Vec<usize>, String> {
    let out = match index {
        None => (0..count).collect(),
        Some(Index::One(i)) => vec![*i],
        Some(Index::Many(v)) => v.clone(),
    };
    if let Some(i) = out.iter().find(|i| **i >= count) {
        return Err(format!("{what} {i} out of range (there are {count})"));
    }
    let unique: BTreeSet<usize> = out.iter().copied().collect();
    if unique.len() != out.len() {
        return Err(format!("{what}: a unit listed twice"));
    }
    Ok(unique.into_iter().collect())
}

impl Graph {
    /// The program's nodes resolved against `weights` (native view: heads and neurons) and its
    /// edges checked: a writer writes before its reader reads, a route the reader has, no piece in
    /// two nodes.
    pub fn parse(program: &Program, weights: &Weights) -> Result<Self, String> {
        if !program.valid {
            return Err(program.error.clone().unwrap_or_else(|| "the program is marked invalid".into()));
        }
        let layers = weights.layers.len();
        let mut ids = Vec::new();
        let mut blocks = Vec::new();
        let mut owned: BTreeSet<(usize, bool, usize)> = BTreeSet::new();
        for node in &program.nodes {
            if node.id == "embed" || node.id == "logits" || ids.contains(&node.id) {
                return Err(format!("node id {} is reserved or repeated", node.id));
            }
            if node.rule.as_ref().is_some_and(|r| !r.is_null()) {
                return Err(format!("{}: rules are not executed yet", node.id));
            }
            let mut block: Option<Block> = None;
            for piece in &node.pieces {
                if piece.view != "native" {
                    return Err(format!("{}: view {} is not resolved yet (native only)", node.id, piece.view));
                }
                if piece.layer >= layers {
                    return Err(format!("{}: layer {} out of range (there are {layers})", node.id, piece.layer));
                }
                let l = piece.layer;
                let next = match piece.kind.as_str() {
                    "head" => Block::Heads { layer: l, heads: indices(&piece.index, weights.layers[l].heads.len(), "head")? },
                    "mlp" | "neuron" => Block::Neurons { layer: l, neurons: indices(&piece.index, weights.neurons(l), "neuron")? },
                    other => return Err(format!("{}: kind {other} is not a native piece", node.id)),
                };
                block = Some(match (block, next) {
                    (None, b) => b,
                    (Some(Block::Heads { layer, mut heads }), Block::Heads { layer: m, heads: more }) if layer == m => {
                        heads.extend(more);
                        Block::Heads { layer, heads }
                    }
                    (Some(Block::Neurons { layer, mut neurons }), Block::Neurons { layer: m, neurons: more }) if layer == m => {
                        neurons.extend(more);
                        Block::Neurons { layer, neurons }
                    }
                    _ => return Err(format!("{}: a node's pieces must lie at one site (one layer's heads or one layer's MLP)", node.id)),
                });
            }
            let Some(mut block) = block else { return Err(format!("{}: a node of no pieces", node.id)) };
            match &mut block {
                Block::Heads { layer, heads } => {
                    heads.sort_unstable();
                    for &h in heads.iter() {
                        if !owned.insert((*layer, true, h)) {
                            return Err(format!("{}: head {layer}.{h} is in two nodes", node.id));
                        }
                    }
                }
                Block::Neurons { layer, neurons } => {
                    neurons.sort_unstable();
                    for &i in neurons.iter() {
                        if !owned.insert((*layer, false, i)) {
                            return Err(format!("{}: neuron {layer}.{i} is in two nodes", node.id));
                        }
                    }
                }
            }
            if block.is_empty() {
                return Err(format!("{}: a node of no pieces", node.id));
            }
            ids.push(node.id.clone());
            blocks.push(block);
        }
        let final_site = 2 * layers;
        let mut edges = Vec::new();
        for e in &program.edges {
            let route = Route::parse(&e.route)?;
            let writer = match e.from.as_str() {
                "embed" => Writer::Embed,
                id => Writer::Unit(ids.iter().position(|i| i == id).ok_or_else(|| format!("edge from unknown node {id}"))?),
            };
            let reader = match e.to.as_str() {
                "logits" => None,
                id => Some(ids.iter().position(|i| i == id).ok_or_else(|| format!("edge to unknown node {id}"))?),
            };
            let read_site = reader.map_or(final_site, |r| blocks[r].site());
            if let Writer::Unit(w) = writer
                && blocks[w].site() >= read_site
            {
                return Err(format!("edge {} >> {}: the writer does not write before the reader reads", e.from, e.to));
            }
            let routes = reader.map_or(&[Route::Input][..], |r| blocks[r].routes());
            if !routes.contains(&route) {
                return Err(format!("edge {} >> {}: the reader has no {} route", e.from, e.to, e.route));
            }
            if !edges.contains(&(writer, reader, route)) {
                edges.push((writer, reader, route));
            }
        }
        Ok(Self { ids, blocks, edges })
    }

    /// The empty program: every piece a stand-in.
    pub fn empty() -> Self {
        Self { ids: Vec::new(), blocks: Vec::new(), edges: Vec::new() }
    }

    /// Every piece of `weights` not in a node, per site: the heads of each layer, then its neurons.
    fn complement(&self, weights: &Weights) -> Vec<Block> {
        let mut out = Vec::new();
        for (l, layer) in weights.layers.iter().enumerate() {
            let taken = |heads: bool| -> BTreeSet<usize> {
                self.blocks
                    .iter()
                    .flat_map(|b| match b {
                        Block::Heads { layer, heads: hs } if heads && *layer == l => hs.clone(),
                        Block::Neurons { layer, neurons } if !heads && *layer == l => neurons.clone(),
                        _ => Vec::new(),
                    })
                    .collect()
            };
            let (heads, neurons) = (taken(true), taken(false));
            let rest: Vec<usize> = (0..layer.heads.len()).filter(|h| !heads.contains(h)).collect();
            if !rest.is_empty() {
                out.push(Block::Heads { layer: l, heads: rest });
            }
            let rest: Vec<usize> = (0..weights.neurons(l)).filter(|i| !neurons.contains(i)).collect();
            if !rest.is_empty() {
                out.push(Block::Neurons { layer: l, neurons: rest });
            }
        }
        out
    }

    /// The program as a circuit: its nodes computing, routed by the declared edges (`edges`) or with
    /// every edge kept (node-level), every other piece a stand-in.
    pub fn program(&self, weights: &Weights, edges: bool) -> Circuit {
        let routed = |reader: Option<usize>, route: Route| -> Incoming {
            if !edges {
                return Incoming::all();
            }
            Incoming::Only(self.edges.iter().filter(|(_, r, k)| *r == reader && *k == route).map(|(w, _, _)| *w).collect())
        };
        let mut units: Vec<Unit> = self
            .blocks
            .iter()
            .enumerate()
            .map(|(n, block)| {
                let mut routes = [Incoming::all(), Incoming::all(), Incoming::all()];
                for &route in block.routes() {
                    routes[route.slot()] = routed(Some(n), route);
                }
                Unit { block: block.clone(), computes: true, routes }
            })
            .collect();
        units.extend(self.complement(weights).into_iter().map(|block| Unit { block, computes: false, routes: [Incoming::all(), Incoming::all(), Incoming::all()] }));
        Circuit { units, logits: routed(None, Route::Input), nodes: self.blocks.len() }
    }

    /// `M` as a circuit with the program's nodes as its first units: every piece computing, every
    /// edge kept.
    pub fn model(&self, weights: &Weights) -> Circuit {
        let mut circuit = self.program(weights, false);
        for unit in &mut circuit.units {
            unit.computes = true;
        }
        circuit
    }

    /// Opaque numbers: every weight a declared node reads (a head's query, key, value and output
    /// maps and head norm gains, keys and values shared by query heads counted once; a neuron's gate
    /// and up rows with their biases and its down column) and every stand-in average (`embed`'s,
    /// one per undeclared head, one per layer with an undeclared neuron), each of width `d`.
    pub fn opaque_numbers(&self, weights: &Weights) -> usize {
        let d = weights.width();
        let mut count = d;
        for block in &self.blocks {
            match block {
                Block::Heads { layer, heads } => {
                    let all = &weights.layers[*layer].heads;
                    let mut keys: Vec<usize> = Vec::new();
                    for &h in heads {
                        let w = &all[h];
                        count += w.query.len() + w.output.len() + w.query_norm.as_ref().map_or(0, |(g, _)| g.len());
                        if !keys.iter().any(|&k| all[k].key == w.key && all[k].value == w.value) {
                            keys.push(h);
                            count += w.key.len() + w.value.len() + w.key_norm.as_ref().map_or(0, |(g, _)| g.len());
                        }
                    }
                }
                Block::Neurons { layer, neurons } => {
                    let m = weights.layers[*layer].mlp.as_ref().expect("a resolved neuron has an MLP");
                    let per = 2 * m.gate.ncols() + 1 + m.up.as_ref().map_or(0, |u| u.ncols() + 1);
                    count += neurons.len() * per;
                }
            }
        }
        for block in self.complement(weights) {
            count += match block {
                Block::Heads { heads, .. } => heads.len() * d,
                Block::Neurons { .. } => d,
            };
        }
        count
    }
}

// ------------------------------------------------------------------------------ stand-ins

/// The stored average inputs of a behavior's prompts, measured on `M`: the token frequencies, per
/// layer and head its mean attention-weighted normalized stream, per layer its MLP's mean
/// normalized stream.
#[derive(Clone, Debug)]
pub struct Stats {
    pub tokens: BTreeMap<u32, f64>,
    pub heads: Vec<Vec<Array1<f64>>>,
    pub mlps: Vec<Array1<f64>>,
}

impl Stats {
    /// Measured on `M` (unedited `weights`) over `sequences`.
    pub fn measure(weights: &Weights, sequences: &[Vec<u32>]) -> Result<Self, String> {
        let batch = Batch::new(sequences)?;
        let rows = batch.tokens.len() as f64;
        let mut tokens = BTreeMap::new();
        for &t in &batch.tokens {
            *tokens.entry(t).or_insert(0.0) += 1.0 / rows;
        }
        let graph = Graph::empty();
        let circuit = graph.model(weights);
        let mut stats = Self { tokens, heads: Vec::new(), mlps: Vec::new() };
        let run = execute(weights, &stats, &circuit, &batch, &[], &BTreeMap::new(), true)?;
        let recorded = run.recorded.ok_or("no recorded inputs")?;
        stats.heads = recorded.0;
        stats.mlps = recorded.1;
        Ok(stats)
    }
}

// ------------------------------------------------------------------------------ execution

/// Sequences end to end, each its own causal span (start, length).
pub struct Batch {
    pub tokens: Vec<u32>,
    pub spans: Vec<(usize, usize)>,
}

impl Batch {
    pub fn new(sequences: &[Vec<u32>]) -> Result<Self, String> {
        let mut tokens = Vec::new();
        let mut spans = Vec::new();
        for s in sequences {
            if s.is_empty() {
                return Err("an empty sequence".into());
            }
            spans.push((tokens.len(), s.len()));
            tokens.extend_from_slice(s);
        }
        if tokens.is_empty() {
            return Err("no sequences".into());
        }
        Ok(Self { tokens, spans })
    }

    /// The batch row of (sequence, position).
    pub fn row(&self, sequence: usize, position: usize) -> Result<usize, String> {
        let &(start, length) = self.spans.get(sequence).ok_or("no such sequence")?;
        if position >= length {
            return Err(format!("position {position} past the sequence's {length} tokens"));
        }
        Ok(start + position)
    }
}

/// One run: the logits' log-probabilities at the scored rows (rows × vocabulary) and every
/// computing unit's actual write (rows × width).
pub struct Execution {
    pub log_probabilities: Array2<f64>,
    pub writes: Vec<Option<Array2<f64>>>,
    recorded: Option<(Vec<Vec<Array1<f64>>>, Vec<Array1<f64>>)>,
}

fn project(x: &Array2<f64>, map: &Array2<f64>, norm: Option<&(Array1<f64>, f64)>) -> Array2<f64> {
    let mut out = x.dot(&map.t());
    if let Some((gain, epsilon)) = norm {
        for mut row in out.outer_iter_mut() {
            let r = rms_scale(row.view(), *epsilon);
            row.zip_mut_with(gain, |v, g| *v *= r * g);
        }
    }
    out
}

/// Heads `heads` of `layer` on their query, key and value inputs (residual streams). With
/// `record`, each head's mean attention-weighted normalized value input as well.
fn heads_write(layer: &LayerWeights, heads: &[usize], inputs: [&Array2<f64>; 3], spans: &[(usize, usize)], record: bool) -> (Array2<f64>, Vec<Array1<f64>>) {
    let norm = &layer.attention;
    let (q_hat, k_hat) = (norm.apply(inputs[0]), norm.apply(inputs[1]));
    let v_unit = norm.unit(inputs[2]);
    let v_hat = &v_unit * &norm.gain.view().insert_axis(Axis(0));
    let (rows, d) = v_hat.dim();
    let mut out = Array2::<f64>::zeros((rows, d));
    let mut recorded = Vec::new();
    for &h in heads {
        let w = &layer.heads[h];
        let (q, k, v) = (project(&q_hat, &w.query, w.query_norm.as_ref()), project(&k_hat, &w.key, w.key_norm.as_ref()), v_hat.dot(&w.value.t()));
        let mut z = Array2::<f64>::zeros(v.dim());
        let mut mixed = Array1::<f64>::zeros(d);
        for &(start, length) in spans {
            let positions: Vec<u32> = (0..length as u32).collect();
            let span = start..start + length;
            let (qs, ks) = (q.slice(s![span.clone(), ..]).to_owned(), k.slice(s![span.clone(), ..]).to_owned());
            let (qs, ks) = (rotate(&qs, w.rotary, &positions, false), rotate(&ks, w.rotary, &positions, false));
            let a = probabilities(qs.view(), ks.view(), &positions, 0, w.scale, w.causal);
            z.slice_mut(s![span.clone(), ..]).assign(&a.dot(&v.slice(s![span.clone(), ..])));
            if record {
                mixed += &a.dot(&v_unit.slice(s![span, ..])).sum_axis(Axis(0));
            }
        }
        if record {
            recorded.push(mixed / rows as f64);
        }
        out += &z.dot(&w.output.t());
    }
    (out, recorded)
}

/// Neurons `neurons` of an MLP on the normed stream `x_hat` (rows × width).
fn neurons_write(mlp: &MlpWeights, neurons: &[usize], x_hat: &Array2<f64>) -> Array2<f64> {
    let gate = mlp.gate.select(Axis(0), neurons);
    let mut h = x_hat.dot(&gate.t());
    let bias = mlp.bias.select(Axis(0), neurons);
    h += &bias.view().insert_axis(Axis(0));
    h.mapv_inplace(|g| mlp.law.apply(g));
    if let Some(up) = &mlp.up {
        let mut u = x_hat.dot(&up.select(Axis(0), neurons).t());
        u += &mlp.up_bias.select(Axis(0), neurons).view().insert_axis(Axis(0));
        h *= &u;
    }
    h.dot(&mlp.out.select(Axis(1), neurons).t())
}

/// A block's stand-in write (width): its pieces applied with `weights` to their stored inputs.
fn stand_in(weights: &Weights, stats: &Stats, block: &Block) -> Array1<f64> {
    match block {
        Block::Heads { layer, heads } => {
            let lw = &weights.layers[*layer];
            let mut out = Array1::<f64>::zeros(weights.width());
            for &h in heads {
                let w = &lw.heads[h];
                let input = &stats.heads[*layer][h] * &lw.attention.gain;
                out += &w.output.dot(&w.value.dot(&input));
            }
            out
        }
        Block::Neurons { layer, neurons } => {
            let lw = &weights.layers[*layer];
            let x_hat = (&stats.mlps[*layer] * &lw.mlp_norm.gain).insert_axis(Axis(0));
            neurons_write(lw.mlp.as_ref().expect("a neuron block has an MLP"), neurons, &x_hat).row(0).to_owned()
        }
    }
}

/// Executes `circuit` on `batch`: per unit in site order its route inputs (the stand-in stream
/// plus the declared writers' actual minus stand-in writes, or the actual stream minus the cut
/// writers'), its write (actual, `swaps`' value, or its stand-in), and the logits at `scored` rows.
/// With `record`, `M`'s stand-in inputs are measured on the way (every unit must compute and read
/// the actual stream).
pub fn execute(weights: &Weights, stats: &Stats, circuit: &Circuit, batch: &Batch, scored: &[usize], swaps: &BTreeMap<usize, Array2<f64>>, record: bool) -> Result<Execution, String> {
    let (rows, d) = (batch.tokens.len(), weights.width());
    let vocabulary = weights.embedding.nrows();
    if let Some(t) = batch.tokens.iter().find(|t| **t as usize >= vocabulary) {
        return Err(format!("token {t} outside the vocabulary of {vocabulary}"));
    }
    let embed = weights.embedding.select(Axis(0), &batch.tokens.iter().map(|t| *t as usize).collect::<Vec<_>>());
    let mut embed_standin = Array1::<f64>::zeros(d);
    for (&t, &f) in &stats.tokens {
        embed_standin.scaled_add(f, &weights.embedding.row(t as usize));
    }
    let standins: Vec<Array1<f64>> = if record { vec![Array1::zeros(d); circuit.units.len()] } else { circuit.units.iter().map(|u| stand_in(weights, stats, &u.block)).collect() };
    let mut order: Vec<usize> = (0..circuit.units.len()).collect();
    order.sort_by_key(|&u| circuit.units[u].block.site());
    // The actual stream and the stand-in stream entering the current site.
    let mut stream = embed.clone();
    let mut standin_stream = embed_standin.clone();
    let mut writes: Vec<Option<Array2<f64>>> = vec![None; circuit.units.len()];
    let (mut head_stats, mut mlp_stats) = (vec![vec![Array1::<f64>::zeros(d); 0]; weights.layers.len()], vec![Array1::<f64>::zeros(d); weights.layers.len()]);
    if record {
        for (l, layer) in weights.layers.iter().enumerate() {
            head_stats[l] = vec![Array1::zeros(d); layer.heads.len()];
        }
    }
    let delta = |w: Writer, writes: &[Option<Array2<f64>>]| -> Option<Array2<f64>> {
        match w {
            Writer::Embed => Some(&embed - &embed_standin.view().insert_axis(Axis(0))),
            Writer::Unit(u) => writes[u].as_ref().map(|a| a - &standins[u].view().insert_axis(Axis(0))),
        }
    };
    let input = |incoming: &Incoming, stream: &Array2<f64>, standin_stream: &Array1<f64>, writes: &[Option<Array2<f64>>]| -> Array2<f64> {
        match incoming {
            Incoming::AllBut(cut) => {
                let mut x = stream.clone();
                for &w in cut {
                    if let Some(dw) = delta(w, writes) {
                        x -= &dw;
                    }
                }
                x
            }
            Incoming::Only(kept) => {
                let mut x = Array2::from_shape_fn((rows, d), |(_, c)| standin_stream[c]);
                for &w in kept {
                    if let Some(dw) = delta(w, writes) {
                        x += &dw;
                    }
                }
                x
            }
        }
    };
    let mut at = 0;
    while at < order.len() {
        let site = circuit.units[order[at]].block.site();
        let end = order[at..].iter().position(|&u| circuit.units[u].block.site() != site).map_or(order.len(), |k| at + k);
        for &u in &order[at..end] {
            let unit = &circuit.units[u];
            if !unit.computes {
                continue;
            }
            if let Some(value) = swaps.get(&u) {
                writes[u] = Some(value.clone());
                continue;
            }
            let routes = unit.block.routes();
            let inputs: Vec<Array2<f64>> = routes.iter().map(|r| input(&unit.routes[r.slot()], &stream, &standin_stream, &writes)).collect();
            writes[u] = Some(match &unit.block {
                Block::Heads { layer, heads } => {
                    let (w, recorded) = heads_write(&weights.layers[*layer], heads, [&inputs[0], &inputs[1], &inputs[2]], &batch.spans, record);
                    for (h, r) in heads.iter().zip(recorded) {
                        head_stats[*layer][*h] = r;
                    }
                    w
                }
                Block::Neurons { layer, neurons } => {
                    let lw = &weights.layers[*layer];
                    if record {
                        mlp_stats[*layer] = lw.mlp_norm.unit(&inputs[0]).mean_axis(Axis(0)).ok_or("no rows")?;
                    }
                    neurons_write(lw.mlp.as_ref().ok_or("a neuron block without an MLP")?, neurons, &lw.mlp_norm.apply(&inputs[0]))
                }
            });
        }
        for &u in &order[at..end] {
            match &writes[u] {
                Some(w) => stream += w,
                None => stream += &standins[u].view().insert_axis(Axis(0)),
            }
            standin_stream += &standins[u];
        }
        at = end;
    }
    let last = input(&circuit.logits, &stream, &standin_stream, &writes).select(Axis(0), scored);
    let log_probabilities = log_probabilities(weights, &last)?;
    Ok(Execution { log_probabilities, writes, recorded: record.then_some((head_stats, mlp_stats)) })
}

/// Next-token log-probabilities of final streams (rows × width) through the final norm (its own
/// RMS) and the unembedding, normalized in float64.
pub fn log_probabilities(weights: &Weights, last: &Array2<f64>) -> Result<Array2<f64>, String> {
    let mut logits = last.dot(&weights.unembedding.t());
    for (mut row, x) in logits.outer_iter_mut().zip(last.outer_iter()) {
        row *= rms_scale(x, weights.final_norm.epsilon);
        let values = gam_math::categorical::log_softmax(row.as_slice().ok_or("a contiguous row")?).map_err(|e| e.to_string())?;
        row.assign(&Array1::from(values));
    }
    Ok(logits)
}

/// `KL(p ‖ q)` in bits per row of two log-probability tables.
pub fn kl_bits(p: &Array2<f64>, q: &Array2<f64>) -> Vec<f64> {
    p.outer_iter().zip(q.outer_iter()).map(|(p, q)| p.iter().zip(q.iter()).map(|(a, b)| if a.is_finite() { a.exp() * (a - b) } else { 0.0 }).sum::<f64>() / std::f64::consts::LN_2).collect()
}

// ------------------------------------------------------------------------------ experiments

/// A matrix of one block a low-rank perturbation adds to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Matrix {
    Query,
    Key,
    Value,
    Output,
    Gate,
    Down,
}

/// A native weight edit, applied to `M` and the program's weights alike.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WeightEdit {
    /// A head's output columns times `factor` (0 removes the head).
    Head { layer: usize, head: usize, factor: f64 },
    /// Neurons' down columns times `factor`.
    Neurons { layer: usize, neurons: Vec<usize>, factor: f64 },
    /// `W + u vᵀ` on one matrix (a head's for `head`, else the layer's MLP's).
    RankOne { layer: usize, head: Option<usize>, matrix: Matrix, u: Vec<f64>, v: Vec<f64> },
}

impl WeightEdit {
    fn matrix<'a>(weights: &'a mut Weights, layer: usize, head: Option<usize>, matrix: Matrix) -> Result<&'a mut Array2<f64>, String> {
        let lw = weights.layers.get_mut(layer).ok_or("no such layer")?;
        Ok(match (head, matrix) {
            (Some(h), m) => {
                let w = lw.heads.get_mut(h).ok_or("no such head")?;
                match m {
                    Matrix::Query => &mut w.query,
                    Matrix::Key => &mut w.key,
                    Matrix::Value => &mut w.value,
                    Matrix::Output => &mut w.output,
                    _ => return Err("an MLP matrix of a head".into()),
                }
            }
            (None, Matrix::Gate) => &mut lw.mlp.as_mut().ok_or("no MLP")?.gate,
            (None, Matrix::Down) => &mut lw.mlp.as_mut().ok_or("no MLP")?.out,
            _ => return Err("a head matrix without a head".into()),
        })
    }

    /// Applies the edit; returns what restores the weights.
    pub fn apply(&self, weights: &mut Weights) -> Result<Restore, String> {
        let (layer, head, matrix) = match self {
            Self::Head { layer, head, .. } => (*layer, Some(*head), Matrix::Output),
            Self::Neurons { layer, .. } => (*layer, None, Matrix::Down),
            Self::RankOne { layer, head, matrix, .. } => (*layer, *head, *matrix),
        };
        let m = Self::matrix(weights, layer, head, matrix)?;
        let saved = m.clone();
        match self {
            Self::Head { factor, .. } => *m *= *factor,
            Self::Neurons { neurons, factor, .. } => {
                for &i in neurons {
                    if i >= m.ncols() {
                        return Err(format!("neuron {i} out of range"));
                    }
                    m.column_mut(i).mapv_inplace(|v| v * factor);
                }
            }
            Self::RankOne { u, v, .. } => {
                if u.len() != m.nrows() || v.len() != m.ncols() {
                    return Err("a rank-one edit of the wrong shape".into());
                }
                for (i, ui) in u.iter().enumerate() {
                    for (j, vj) in v.iter().enumerate() {
                        m[[i, j]] += ui * vj;
                    }
                }
            }
        }
        Ok(Restore { layer, head, matrix, saved })
    }
}

/// A matrix's value before an edit.
pub struct Restore {
    layer: usize,
    head: Option<usize>,
    matrix: Matrix,
    saved: Array2<f64>,
}

impl Restore {
    pub fn restore(self, weights: &mut Weights) -> Result<(), String> {
        *WeightEdit::matrix(weights, self.layer, self.head, self.matrix)? = self.saved;
        Ok(())
    }
}

/// One experiment, chosen by the checker and applied verbatim to `M` and the program.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Experiment {
    Clean,
    /// Every prompt's counterfactual in its place.
    Counterfactual,
    Edit { edit: WeightEdit, aimed: bool },
    /// Node `node`'s write replaced by its write on the donor prompt (the next prompt of the same
    /// length).
    Swap { node: usize },
    /// The edge from `from` to `to` (a node, `None` the logits) on `route` cut: the reader gets the
    /// writer's stand-in write.
    Cut { from: Writer, to: Option<usize>, route: Route, declared: bool },
}

impl Experiment {
    pub fn family(&self) -> &'static str {
        match self {
            Self::Clean => "clean",
            Self::Counterfactual => "counterfactual",
            Self::Edit { edit: WeightEdit::RankOne { .. }, .. } => "rank_one",
            Self::Edit { aimed: true, .. } => "edit_aimed",
            Self::Edit { aimed: false, .. } => "edit_uniform",
            Self::Swap { .. } => "swap",
            Self::Cut { declared: true, .. } => "cut_declared",
            Self::Cut { declared: false, .. } => "cut_undeclared",
        }
    }

    /// The experiment in words, for the reader.
    pub fn describe(&self, graph: &Graph) -> String {
        let node = |u: usize| graph.ids.get(u).cloned().unwrap_or_else(|| format!("unit {u}"));
        let writer = |w: &Writer| match w {
            Writer::Embed => "embed".to_string(),
            Writer::Unit(u) => node(*u),
        };
        match self {
            Self::Clean => "the prompt as given".into(),
            Self::Counterfactual => "the counterfactual prompt in place of the prompt".into(),
            Self::Edit { edit, .. } => match edit {
                WeightEdit::Head { layer, head, factor } => format!("L[{layer}].head[{head}]'s output weights times {factor}"),
                WeightEdit::Neurons { layer, neurons, factor } => format!("L[{layer}].mlp{neurons:?}'s down weights times {factor}"),
                WeightEdit::RankOne { layer, head, matrix, .. } => format!("a random rank-one change of L[{layer}]{}'s {matrix:?} weights", head.map_or(".mlp".to_string(), |h| format!(".head[{h}]"))),
            },
            Self::Swap { node: n } => format!("{}'s output replaced by its output on another prompt", node(*n)),
            Self::Cut { from, to, route, .. } => format!("the connection {} >> {}.{route:?} cut (the reader gets its average)", writer(from), to.map_or("logits".to_string(), node)),
        }
    }
}

/// The checker's draw of `count` experiments beyond clean (and counterfactual when the prompts
/// have them): weight edits (half uniform over pieces at random granularity, half aimed at the
/// program's pieces and pieces it omits), rank-one perturbations, node swaps, edge cuts (declared
/// edges and undeclared pairs).
pub fn sample(weights: &Weights, graph: &Graph, counterfactual: bool, count: usize, seed: u64) -> Vec<Experiment> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = vec![Experiment::Clean];
    if counterfactual {
        out.push(Experiment::Counterfactual);
    }
    let layers = weights.layers.len();
    let factors = [0.0, 0.5, 2.0];
    let omitted = graph.complement(weights);
    let random_edit = |rng: &mut StdRng, block: Option<&Block>| -> WeightEdit {
        let factor = factors[rng.random_range(0..factors.len())];
        let block = match block {
            Some(b) => b.clone(),
            None => {
                let l = rng.random_range(0..layers);
                if rng.random_bool(0.5) || weights.neurons(l) == 0 {
                    Block::Heads { layer: l, heads: (0..weights.layers[l].heads.len()).collect() }
                } else {
                    Block::Neurons { layer: l, neurons: (0..weights.neurons(l)).collect() }
                }
            }
        };
        match block {
            Block::Heads { layer, heads } => WeightEdit::Head { layer, head: heads[rng.random_range(0..heads.len())], factor },
            Block::Neurons { layer, neurons } => {
                // Granularity: one neuron, a group of 8, or every neuron of the block.
                let picked = match rng.random_range(0..3) {
                    0 => vec![neurons[rng.random_range(0..neurons.len())]],
                    1 => (0..8.min(neurons.len())).map(|_| neurons[rng.random_range(0..neurons.len())]).collect::<BTreeSet<_>>().into_iter().collect(),
                    _ => neurons.clone(),
                };
                WeightEdit::Neurons { layer, neurons: picked, factor }
            }
        }
    };
    let declared_edges: Vec<(Writer, Option<usize>, Route)> = graph.edges.clone();
    let mut kinds = vec!["edit_uniform", "edit_aimed", "rank_one", "cut_undeclared"];
    if !graph.blocks.is_empty() {
        kinds.push("swap");
    }
    if !declared_edges.is_empty() {
        kinds.push("cut_declared");
    }
    for _ in 0..count {
        let kind = kinds[rng.random_range(0..kinds.len())];
        out.push(match kind {
            "edit_uniform" => Experiment::Edit { edit: random_edit(&mut rng, None), aimed: false },
            "edit_aimed" => {
                let own = !graph.blocks.is_empty() && (omitted.is_empty() || rng.random_bool(0.5));
                let pool = if own { &graph.blocks } else { &omitted };
                let block = pool[rng.random_range(0..pool.len())].clone();
                Experiment::Edit { edit: random_edit(&mut rng, Some(&block)), aimed: true }
            }
            "rank_one" => {
                let layer = rng.random_range(0..layers);
                let heads = weights.layers[layer].heads.len();
                let (head, matrix) = if rng.random_bool(0.5) || weights.neurons(layer) == 0 {
                    (Some(rng.random_range(0..heads)), [Matrix::Query, Matrix::Key, Matrix::Value, Matrix::Output][rng.random_range(0..4)])
                } else {
                    (None, [Matrix::Gate, Matrix::Down][rng.random_range(0..2)])
                };
                let probe = weights.clone_matrix(layer, head, matrix);
                // u vᵀ of Frobenius norm half the matrix's, unit random directions.
                let size = 0.5 * probe.iter().map(|v| v * v).sum::<f64>().sqrt();
                let mut unit = |n: usize| -> Vec<f64> {
                    let v: Vec<f64> = (0..n).map(|_| rng.random::<f64>() - 0.5).collect();
                    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
                    v.into_iter().map(|x| x / norm).collect()
                };
                let (rows, cols) = probe.dim();
                let u: Vec<f64> = unit(rows).into_iter().map(|x| x * size).collect();
                let v = unit(cols);
                Experiment::Edit { edit: WeightEdit::RankOne { layer, head, matrix, u, v }, aimed: false }
            }
            "swap" => Experiment::Swap { node: rng.random_range(0..graph.blocks.len()) },
            "cut_declared" => {
                let (from, to, route) = declared_edges[rng.random_range(0..declared_edges.len())];
                Experiment::Cut { from, to, route, declared: true }
            }
            _ => {
                // An undeclared pair among embed, the nodes and the logits, in causal order.
                let readers: Vec<Option<usize>> = (0..graph.blocks.len()).map(Some).chain([None]).collect();
                let to = readers[rng.random_range(0..readers.len())];
                let site = to.map_or(2 * layers, |r| graph.blocks[r].site());
                let writers: Vec<Writer> = [Writer::Embed].into_iter().chain((0..graph.blocks.len()).filter(|&w| graph.blocks[w].site() < site).map(Writer::Unit)).collect();
                let from = writers[rng.random_range(0..writers.len())];
                let routes = to.map_or(&[Route::Input][..], |r| graph.blocks[r].routes());
                let route = routes[rng.random_range(0..routes.len())];
                let declared = declared_edges.contains(&(from, to, route));
                Experiment::Cut { from, to, route, declared }
            }
        });
    }
    out
}

impl Weights {
    fn clone_matrix(&self, layer: usize, head: Option<usize>, matrix: Matrix) -> Array2<f64> {
        let lw = &self.layers[layer];
        match (head, matrix) {
            (Some(h), Matrix::Query) => lw.heads[h].query.clone(),
            (Some(h), Matrix::Key) => lw.heads[h].key.clone(),
            (Some(h), Matrix::Value) => lw.heads[h].value.clone(),
            (Some(h), _) => lw.heads[h].output.clone(),
            (None, Matrix::Gate) => lw.mlp.as_ref().map(|m| m.gate.clone()).unwrap_or_default(),
            (None, _) => lw.mlp.as_ref().map(|m| m.out.clone()).unwrap_or_default(),
        }
    }
}

// ------------------------------------------------------------------------------ behaviors

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Counterfactual {
    #[serde(default)]
    pub text: String,
    pub token_ids: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prompt {
    #[serde(default)]
    pub text: String,
    pub token_ids: Vec<u32>,
    pub target_positions: Vec<usize>,
    #[serde(default)]
    pub counterfactual: Option<Counterfactual>,
}

/// A behavior file (`~/mpd-data/graph_oracle/behaviors/<model>/<id>.json`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Behavior {
    pub id: String,
    pub model: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub frequency: Option<f64>,
    pub prompts: Vec<Prompt>,
    #[serde(default)]
    pub split: String,
    #[serde(default)]
    pub model_accuracy: Option<f64>,
}

impl Behavior {
    /// The declared size in scored tokens: `2^24`, times the behavior's frequency when it states one.
    pub fn size(&self) -> f64 {
        16_777_216.0 * self.frequency.unwrap_or(1.0)
    }
}

/// A behavior prepared for scoring: its prompts and counterfactuals as batches, the scored rows,
/// the swap donors, the stand-in averages, and `M`'s outcome per experiment (cached by its key).
pub struct Checker {
    pub weights: Weights,
    pub behavior: Behavior,
    pub stats: Stats,
    clean: (Batch, Vec<usize>),
    counterfactual: Option<(Batch, Vec<usize>)>,
    /// Per prompt with a same-length donor: (prompt, donor).
    donors: Vec<(usize, usize)>,
    cache: BTreeMap<String, Array2<f64>>,
}

/// Every score term (bits) and the counts behind them.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Score {
    pub total_bits: f64,
    pub exec_error_bits: f64,
    pub reader_error_bits: f64,
    pub code_bits: f64,
    pub python_tokens: usize,
    pub opaque_numbers: usize,
    pub opaque_bits: f64,
    #[serde(rename = "N")]
    pub n: f64,
    pub experiments: usize,
    pub valid: bool,
    pub error: Option<String>,
    pub per_family: BTreeMap<String, Family>,
}

/// One experiment family's share of the execution error.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Family {
    pub experiments: usize,
    pub tokens: usize,
    /// Mean `KL(M_e ‖ P_e)` per scored token, bits.
    pub mean_kl_bits: f64,
}

fn scored_rows(batch: &Batch, prompts: &[(usize, &[usize])]) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for &(s, positions) in prompts {
        for &p in positions {
            out.push(batch.row(s, p)?);
        }
    }
    Ok(out)
}

impl Checker {
    pub fn new(weights: Weights, behavior: Behavior) -> Result<Self, String> {
        if behavior.prompts.is_empty() {
            return Err("a behavior of no prompts".into());
        }
        let sequences: Vec<Vec<u32>> = behavior.prompts.iter().map(|p| p.token_ids.clone()).collect();
        let batch = Batch::new(&sequences)?;
        let targets: Vec<(usize, &[usize])> = behavior.prompts.iter().enumerate().map(|(i, p)| (i, p.target_positions.as_slice())).collect();
        let rows = scored_rows(&batch, &targets)?;
        let stats = Stats::measure(&weights, &sequences)?;
        // A counterfactual's targets: the prompt's, counted from the end of the sequence.
        let counterfactual = if behavior.prompts.iter().all(|p| p.counterfactual.is_some()) {
            let cf: Vec<Vec<u32>> = behavior.prompts.iter().map(|p| p.counterfactual.as_ref().map(|c| c.token_ids.clone()).unwrap_or_default()).collect();
            let b = Batch::new(&cf)?;
            let shifted: Vec<Vec<usize>> = behavior.prompts.iter().zip(&cf).map(|(p, c)| p.target_positions.iter().filter_map(|&t| (t + c.len()).checked_sub(p.token_ids.len())).collect()).collect();
            if shifted.iter().zip(&behavior.prompts).any(|(s, p)| s.len() != p.target_positions.len()) {
                return Err("a counterfactual too short for its prompt's targets".into());
            }
            let r = scored_rows(&b, &shifted.iter().enumerate().map(|(i, s)| (i, s.as_slice())).collect::<Vec<_>>())?;
            Some((b, r))
        } else {
            None
        };
        let mut donors = Vec::new();
        let mut by_length: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, p) in behavior.prompts.iter().enumerate() {
            by_length.entry(p.token_ids.len()).or_default().push(i);
        }
        for group in by_length.values().filter(|g| g.len() > 1) {
            donors.extend(group.iter().enumerate().map(|(k, &i)| (i, group[(k + 1) % group.len()])));
        }
        donors.sort_unstable();
        Ok(Self { weights, behavior, stats, clean: (batch, rows), counterfactual, donors, cache: BTreeMap::new() })
    }

    /// The experiment's key for `M`'s cache: what it does to which pieces.
    fn key(graph: &Graph, e: &Experiment) -> String {
        let block = |u: usize| serde_json::to_string(&graph.blocks[u]).unwrap_or_default();
        match e {
            Experiment::Swap { node } => format!("swap {}", block(*node)),
            Experiment::Cut { from, to, route, .. } => {
                let from = match from {
                    Writer::Embed => "embed".to_string(),
                    Writer::Unit(u) => block(*u),
                };
                format!("cut {from} {} {route:?}", to.map_or("logits".to_string(), block))
            }
            other => serde_json::to_string(other).unwrap_or_default(),
        }
    }

    /// One circuit under one experiment: its log-probabilities at the scored rows (the swap's
    /// prompts only, for a swap).
    fn outcome(&mut self, circuit: &Circuit, e: &Experiment) -> Result<Array2<f64>, String> {
        let mut circuit = circuit.clone();
        let (batch, rows) = match e {
            Experiment::Counterfactual => self.counterfactual.as_ref().ok_or("the behavior has no counterfactuals")?,
            _ => &self.clean,
        };
        let restore = match e {
            Experiment::Edit { edit, .. } => Some(edit.apply(&mut self.weights)?),
            _ => None,
        };
        let result = (|| -> Result<Array2<f64>, String> {
            match e {
                Experiment::Cut { from, to, route, .. } => {
                    match to {
                        Some(r) => circuit.units[*r].routes[route.slot()].cut(*from),
                        None => circuit.logits.cut(*from),
                    }
                    Ok(execute(&self.weights, &self.stats, &circuit, batch, rows, &BTreeMap::new(), false)?.log_probabilities)
                }
                Experiment::Swap { node } => {
                    if self.donors.is_empty() {
                        return Err("no two prompts of one length to swap between".into());
                    }
                    let prompts: Vec<Vec<u32>> = self.donors.iter().map(|(i, _)| self.behavior.prompts[*i].token_ids.clone()).collect();
                    let donors: Vec<Vec<u32>> = self.donors.iter().map(|(_, j)| self.behavior.prompts[*j].token_ids.clone()).collect();
                    let (base, donor) = (Batch::new(&prompts)?, Batch::new(&donors)?);
                    let donor_run = execute(&self.weights, &self.stats, &circuit, &donor, &[], &BTreeMap::new(), false)?;
                    let value = donor_run.writes[*node].clone().ok_or("the swapped node does not compute")?;
                    let targets: Vec<(usize, &[usize])> = self.donors.iter().enumerate().map(|(k, (i, _))| (k, self.behavior.prompts[*i].target_positions.as_slice())).collect();
                    let rows = scored_rows(&base, &targets)?;
                    Ok(execute(&self.weights, &self.stats, &circuit, &base, &rows, &[(*node, value)].into(), false)?.log_probabilities)
                }
                _ => Ok(execute(&self.weights, &self.stats, &circuit, batch, rows, &BTreeMap::new(), false)?.log_probabilities),
            }
        })();
        if let Some(r) = restore {
            r.restore(&mut self.weights)?;
        }
        result
    }

    /// `M`'s outcome under `e`, cached.
    pub fn model_outcome(&mut self, graph: &Graph, e: &Experiment) -> Result<Array2<f64>, String> {
        let key = Self::key(graph, e);
        if let Some(hit) = self.cache.get(&key) {
            return Ok(hit.clone());
        }
        let circuit = graph.model(&self.weights);
        let out = self.outcome(&circuit, e)?;
        self.cache.insert(key, out.clone());
        Ok(out)
    }

    /// The program's score under `count` sampled experiments (seed `seed`); `edges` routes by the
    /// declared edges, else every edge among the nodes is kept. `n` overrides the behavior's size.
    /// An invalid program is scored as the empty program, flagged.
    pub fn score(&mut self, program: &Program, count: usize, seed: u64, edges: bool, n: Option<f64>) -> Result<(Score, Vec<(Experiment, Array2<f64>, Array2<f64>)>), String> {
        let n = n.unwrap_or_else(|| self.behavior.size());
        let (graph, valid, error) = match Graph::parse(program, &self.weights) {
            Ok(g) => (g, true, None),
            Err(e) => (Graph::empty(), false, Some(e)),
        };
        let experiments = sample(&self.weights, &graph, self.counterfactual.is_some(), count, seed);
        let circuit = graph.program(&self.weights, edges);
        let mut per_family: BTreeMap<String, Family> = BTreeMap::new();
        let mut total = (0.0, 0usize);
        let mut outcomes = Vec::new();
        for e in &experiments {
            if matches!(e, Experiment::Swap { .. }) && self.donors.is_empty() {
                continue;
            }
            let m = self.model_outcome(&graph, e)?;
            let p = self.outcome(&circuit, e)?;
            let kl = kl_bits(&m, &p);
            let sum: f64 = kl.iter().sum();
            let entry = per_family.entry(e.family().to_string()).or_default();
            entry.experiments += 1;
            entry.tokens += kl.len();
            entry.mean_kl_bits += sum;
            total.0 += sum;
            total.1 += kl.len();
            outcomes.push((e.clone(), m, p));
        }
        for v in per_family.values_mut() {
            v.mean_kl_bits /= v.tokens.max(1) as f64;
        }
        let exec_error_bits = n * total.0 / total.1.max(1) as f64;
        let code_bits = if valid && program.token_types > 1 { program.python_tokens as f64 * (program.token_types as f64).log2() } else { 0.0 };
        let opaque_numbers = graph.opaque_numbers(&self.weights);
        let opaque_bits = 0.5 * n.log2() * opaque_numbers as f64;
        let score = Score {
            total_bits: exec_error_bits + code_bits + opaque_bits,
            exec_error_bits,
            reader_error_bits: 0.0,
            code_bits,
            python_tokens: if valid { program.python_tokens } else { 0 },
            opaque_numbers,
            opaque_bits,
            n,
            experiments: outcomes.len(),
            valid,
            error,
            per_family,
        };
        Ok((score, outcomes))
    }

    /// Each swapped prompt with its donor.
    pub fn donors(&self) -> &[(usize, usize)] {
        &self.donors
    }

    /// A swap's scored tokens as (prompt, position), in its rows' order.
    pub fn swap_targets(&self) -> Vec<(usize, usize)> {
        self.donors.iter().flat_map(|&(i, _)| self.behavior.prompts[i].target_positions.iter().map(move |&t| (i, t))).collect()
    }

    /// The graph a program parses to (for describing its experiments), or the empty graph.
    pub fn graph(&self, program: &Program) -> Graph {
        Graph::parse(program, &self.weights).unwrap_or_else(|_| Graph::empty())
    }
}

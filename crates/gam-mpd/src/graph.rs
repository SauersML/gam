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
    interchange::{self, Operation, SharedSite, SiteOp},
    library_readout::Library,
    operator_program::{Law, Rotary, rms_scale},
    tiled_attention::{probabilities, rotate},
};
use ndarray::{Array1, Array2, Axis, s};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

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
    /// What undeclared pieces and edges carry: "delete", nothing (they contribute zero; the default
    /// when the model has a VPD view attached), or "counterfactual", the same model's values on the
    /// prompt's counterfactual (the default otherwise).
    #[serde(default)]
    pub standin: Option<String>,
    /// Ids of nodes from a shared base library (parts every behavior's program may use): their
    /// precision is reported as `Score::base_bits`, outside the total, for the caller to charge once
    /// across behaviors.
    #[serde(default)]
    pub base: Vec<String>,
    /// The English explanation's tokens and token types (the reader's tokenizer), counted by the
    /// program's writer: `Score::explanation_bits`.
    #[serde(default)]
    pub explanation_tokens: usize,
    #[serde(default)]
    pub explanation_token_types: usize,
    /// The algorithm's variables aligned to parts ([`AlignmentIr`]), checked by interchange.
    #[serde(default)]
    pub alignments: Vec<AlignmentIr>,
    /// The named groups of parts the program uses ([`GroupIr`]).
    #[serde(default)]
    pub groups: Vec<GroupIr>,
}

/// A named group of parts a program uses (mech's `G.<name>`, from the model's shared library of
/// groups): its part tokens (`<p:L.S.I>`, `<p:L.S.rest>` for VPD site `S` of layer `L`) and the
/// nodes holding them. A use costs one name in `Score::structure_bits`, its parts none there; the
/// group's definition, its parts' names, is `Score::group_bits`, outside the total (a library is
/// paid once per model).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GroupIr {
    pub name: String,
    pub parts: Vec<String>,
    #[serde(default)]
    pub nodes: Vec<String>,
}

/// A VPD part a group names: layer, matrix (`q_proj` 0, `k_proj` 1, `v_proj` 2, `o_proj` 3, `c_fc`
/// 4, `down_proj` 5) and subcomponent (the matrix's subcomponent count: its remainder).
type GroupPart = (usize, usize, usize);

/// Part token `<p:L.S.I>` or `<p:L.S.rest>` of a VPD site as a [`GroupPart`].
fn group_part(weights: &Weights, token: &str) -> Result<GroupPart, String> {
    let inner = token.strip_prefix("<p:").and_then(|t| t.strip_suffix('>')).ok_or_else(|| format!("group part {token}: not a part token"))?;
    let fields: Vec<&str> = inner.split('.').collect();
    let [layer, site, index] = fields.as_slice() else { return Err(format!("group part {token}: not a VPD part <p:L.S.I>")) };
    let layer: usize = layer.parse().map_err(|_| format!("group part {token}: layer"))?;
    let matrix = ["q", "k", "v", "o", "fc", "down"].iter().position(|m| m == site).ok_or_else(|| format!("group part {token}: site {site} is not a VPD site"))?;
    let count = if matrix < 4 {
        weights.vpd_attention.get(&layer).map(|a| [a.q.0.nrows(), a.k.0.nrows(), a.v.0.nrows(), a.o.0.nrows()][matrix])
    } else {
        weights.vpd.get(&layer).map(|v| if matrix == 4 { v.fc_u.nrows() } else { v.down_u.nrows() })
    }
    .ok_or_else(|| format!("group part {token}: layer {layer} has no VPD view of that site"))?;
    let index = if *index == "rest" { count } else { index.parse::<usize>().ok().filter(|&i| i < count).ok_or_else(|| format!("group part {token}: index outside 0..{count}"))? };
    Ok((layer, matrix, index))
}

/// The code tokens [`Graph::structure`] counts for a program's structure, as format v3 writes it:
/// a node (`name = node( … )`), a label on it (`means = function`, or a claim), an edge (the writer
/// in the reader's list and its separator), and a route other than the input (`. query`).
pub const NODE_TOKENS: usize = 5;
pub const LABEL_TOKENS: usize = 3;
pub const EDGE_TOKENS: usize = 2;
pub const ROUTE_TOKENS: usize = 2;

/// The bits of one code token of a program of `types` token types (none below two types, as its
/// code bits).
pub fn token_bits(types: usize) -> f64 {
    if types > 1 { (types as f64).log2() } else { 0.0 }
}

/// A named group a program uses: its name, parts and the nodes (indices) holding them.
#[derive(Clone, Debug)]
pub struct GroupUse {
    pub name: String,
    pub parts: BTreeSet<GroupPart>,
    pub nodes: Vec<usize>,
}

/// An alignment (design_v2 section 2): the variable `variable` of the program's algorithm (an
/// intermediate one: the answer variable is aligned by construction and not tested) is held by the
/// parts of nodes `nodes`, tested by interchange on the prompt pairs `pairs`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AlignmentIr {
    pub variable: String,
    pub nodes: Vec<String>,
    #[serde(default)]
    pub pairs: Vec<PairIr>,
}

/// One interchange of an alignment: prompts `base` and `source` (indices into the behavior, of one
/// length), or with `changed` the base prompt and its own changed prompt (its counterfactual), so
/// that the pair differs only where the behavior's change does (format v3's label test). Its target
/// is `M`'s own distribution on the source at the base's targets (an older IR's algorithm answer
/// tokens, keys "answer"/"answers", are ignored).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PairIr {
    pub base: usize,
    #[serde(default)]
    pub source: usize,
    #[serde(default)]
    pub changed: bool,
}

fn yes() -> bool {
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeIr {
    pub id: String,
    pub pieces: Vec<PieceIr>,
    /// An attention claim about the node's parts ([`Claim`]; "rule" in older programs).
    #[serde(default, alias = "rule")]
    pub claim: Option<serde_json::Value>,
    /// Where the node acts ([`Positions`]): per sequence (a prompt or a counterfactual, by its
    /// tokens) the positions; empty, everywhere.
    #[serde(default)]
    pub at: Vec<SequenceAt>,
}

/// One sequence's positions of a node: its tokens and the positions (from 0) the node acts at.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SequenceAt {
    pub tokens: Vec<u32>,
    pub positions: Vec<usize>,
}

/// Where a program's node acts: per sequence, by its tokens, the positions. At the others its parts
/// write their stand-ins (their counterfactual values, or zero when deleted), and so do its query,
/// key, value and MLP-input contributions there: a reader at another position reads the node's
/// stand-in from those positions. A sequence it does not list (an experiment's edited prompt) it
/// acts at everywhere.
#[derive(Clone, Debug, Default)]
pub struct Positions {
    pub by_tokens: std::collections::HashMap<Vec<u32>, Vec<bool>>,
}

impl Positions {
    /// Per row of `batch` whether the node acts there; `invert`: everywhere but at its positions
    /// (nowhere in a sequence it does not list).
    pub fn rows(&self, batch: &Batch, invert: bool) -> Vec<bool> {
        let mut out = vec![!invert; batch.tokens.len()];
        for &(start, length) in &batch.spans {
            if let Some(mask) = self.by_tokens.get(&batch.tokens[start..start + length]) {
                for (row, &acts) in out[start..start + length].iter_mut().zip(mask) {
                    *row = acts != invert;
                }
            }
        }
        out
    }
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
    /// "rest": a VPD matrix's remainder `W − Σ U Vᵀ`, its own piece.
    Name(String),
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
    pub(crate) fn slot(self) -> usize {
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

/// A stored weight matrix: float32. Every model the checker reads stores float32 or bfloat16
/// weights, so float32 holds them exactly at half the memory of float64; an edit rounds its result
/// to float32, as the device computes anyway. The host executor widens to float64 to multiply.
pub type Stored = Array2<f32>;

/// A stored matrix (or view) widened to float64.
pub(crate) fn wide(m: ndarray::ArrayView2<f32>) -> Array2<f64> {
    m.mapv(f64::from)
}

/// `m += d`, added in float64 and rounded to float32.
fn add_wide(mut m: ndarray::ArrayViewMut2<f32>, d: ndarray::ArrayView2<f64>) {
    m.zip_mut_with(&d, |w, x| *w = (f64::from(*w) + x) as f32);
}

/// One head: its query and key maps with their head norms, its value map (head width × width,
/// reading the normed stream) and its output columns (width × head width).
#[derive(Clone, Debug)]
pub struct HeadWeights {
    pub query: Stored,
    pub query_norm: Option<(Array1<f64>, f64)>,
    /// The key and value maps, one allocation for the heads of a key/value group (grouped-query
    /// attention); an edit of one head's map gives that head its own copy (`Arc::make_mut`).
    pub key: Arc<Stored>,
    pub key_norm: Option<(Array1<f64>, f64)>,
    pub value: Arc<Stored>,
    pub output: Stored,
    pub scale: f64,
    pub rotary: Option<Rotary>,
    pub causal: bool,
}

/// One MLP: `h = φ(G x̂ + b)` (times `U x̂ + c` when gated), written by `D h`.
#[derive(Clone, Debug)]
pub struct MlpWeights {
    pub gate: Stored,
    pub bias: Array1<f64>,
    pub up: Option<Stored>,
    pub up_bias: Array1<f64>,
    pub out: Stored,
    pub law: Law,
}

#[derive(Clone, Debug)]
pub struct LayerWeights {
    pub attention: Norm,
    pub heads: Vec<HeadWeights>,
    pub mlp_norm: Norm,
    pub mlp: Option<MlpWeights>,
}

/// `M`'s weights on the host: per layer its blocks, the final norm (applied with its gain before
/// the unembedding), the unembedding (vocabulary × width) and the embedding (vocabulary × width),
/// one matrix when the model ties them.
#[derive(Clone, Debug)]
pub struct Weights {
    pub layers: Vec<LayerWeights>,
    pub final_norm: Norm,
    pub unembedding: Arc<Stored>,
    pub embedding: Arc<Stored>,
    /// Per layer with a transcoder view, its features (`attach_transcoders`).
    pub transcoders: BTreeMap<usize, Features>,
    /// Per layer with a VPD view of its MLP, VPD's subcomponents of `c_fc` and `down_proj`
    /// (`attach_vpd`).
    pub vpd: BTreeMap<usize, VpdMlp>,
    /// Per layer with a VPD view of its attention, VPD's subcomponents of `q_proj`, `k_proj`,
    /// `v_proj` and `o_proj` (`attach_vpd`).
    pub vpd_attention: BTreeMap<usize, VpdAttention>,
    /// Our library's parts (decomp's exact start, `attach_library`): per (layer, attention?) its
    /// parts in file order, each the VPD subcomponents it holds per matrix (attention: q, k, v, o;
    /// MLP: c_fc, down_proj). Parts execute through VPD's factors.
    pub library: BTreeMap<(usize, bool), Vec<[Vec<usize>; 4]>>,
    /// The device's resident copies of these weights are keyed by this (with each matrix's address
    /// and shape) and dropped when the last clone goes.
    generation: Arc<Generation>,
}

/// A `Weights`' identity on the device ([`crate::graph_device::forget`] on drop).
#[derive(Debug)]
struct Generation(u64);

impl Drop for Generation {
    fn drop(&mut self) {
        crate::graph_device::forget(self.0);
    }
}

/// VPD's subcomponents of one attention's four matrices, each `U` (subcomponents × out) and `V`
/// (in × subcomponents) as in [`VpdMlp`]; `q`, `k`, `v` read the normed stream and write the heads'
/// concatenated queries, keys and values, `o` reads the heads' concatenated reads.
#[derive(Clone, Debug)]
pub struct VpdAttention {
    pub q: (Array2<f64>, Array2<f64>),
    pub k: (Array2<f64>, Array2<f64>),
    pub v: (Array2<f64>, Array2<f64>),
    pub o: (Array2<f64>, Array2<f64>),
}

/// Subcomponents `picked` of a matrix `w` (out × in) with factors `(U, V)` applied to the rows of
/// `x` (rows × in); with `rest`, every other subcomponent and the remainder, `x wᵀ − named`.
fn sliced(factors: &(Array2<f64>, Array2<f64>), w: &Array2<f64>, picked: &[usize], rest: bool, x: &Array2<f64>) -> Array2<f64> {
    sliced_parts(&factors.0, &factors.1, w, picked, rest, x)
}

/// [`sliced`] with the factors apart: `U` (subcomponents × out), `V` (in × subcomponents); index
/// `U`'s row count is the remainder `W − Σ U Vᵀ`.
fn sliced_parts(u: &Array2<f64>, v: &Array2<f64>, w: &Array2<f64>, picked: &[usize], rest: bool, x: &Array2<f64>) -> Array2<f64> {
    let count = u.nrows();
    let subs: Vec<usize> = picked.iter().copied().filter(|&i| i < count).collect();
    let mut named = x.dot(&v.select(Axis(1), &subs)).dot(&u.select(Axis(0), &subs));
    if picked.contains(&count) {
        named += &(x.dot(&w.t()) - x.dot(v).dot(u));
    }
    if rest { x.dot(&w.t()) - named } else { named }
}

/// A layer's heads as the four native matrices VPD decomposes: the query, key and value maps
/// stacked over heads (heads × head width, by width) and the output columns side by side.
fn attention_maps(layer: &LayerWeights) -> [Array2<f64>; 4] {
    let stack = |f: &dyn Fn(&HeadWeights) -> &Stored| {
        let views: Vec<_> = layer.heads.iter().map(|h| f(h).view()).collect();
        wide(ndarray::concatenate(Axis(0), &views).expect("heads of one width").view())
    };
    let outputs: Vec<_> = layer.heads.iter().map(|h| h.output.view()).collect();
    [stack(&|h| &h.query), stack(&|h| &*h.key), stack(&|h| &*h.value), wide(ndarray::concatenate(Axis(1), &outputs).expect("heads of one width").view())]
}

/// VPD's subcomponents of one MLP's two matrices (`explanation_battery::Factors`): subcomponent `i`
/// of `c_fc` is `U_fc[i] ⊗ V_fc[:, i]` (`U_fc` subcomponents × hidden, `V_fc` width ×
/// subcomponents), of `down_proj` `U_down[j] ⊗ V_down[:, j]` (`U_down` subcomponents × width,
/// `V_down` hidden × subcomponents). Each matrix is its subcomponents plus a remainder, the
/// current weights minus their sum, so native edits land in the remainder.
#[derive(Clone, Debug)]
pub struct VpdMlp {
    pub fc_u: Array2<f64>,
    pub fc_v: Array2<f64>,
    pub down_u: Array2<f64>,
    pub down_v: Array2<f64>,
}

impl VpdMlp {
    /// The hidden pre-activation `c_fc` subcomponents `fc` write from normed inputs `x_hat`
    /// (rows × hidden); with `rest` every other subcomponent and the remainder, `W x̂ − Σ`.
    fn fc(&self, mlp: &MlpWeights, fc: &[usize], rest: bool, x_hat: &Array2<f64>) -> Array2<f64> {
        sliced_parts(&self.fc_u, &self.fc_v, &wide(mlp.gate.view()), fc, rest, x_hat)
    }

    /// The residual write of `down_proj` subcomponents `down` from hidden activations `h`; with
    /// `rest` every other subcomponent and the remainder.
    fn down(&self, mlp: &MlpWeights, down: &[usize], rest: bool, h: &Array2<f64>) -> Array2<f64> {
        sliced_parts(&self.down_u, &self.down_v, &wide(mlp.out.view()), down, rest, h)
    }
}

/// One layer's transcoder (circuit-tracer's single-layer ReLU transcoder, `library_transcoder`):
/// feature `i` reads the MLP's normed input `x̂` and writes `relu(g_i·x̂ + c_i) u_i`. The file, its
/// feature count, and the rows of the features programs declare (read on demand,
/// [`Weights::load_features`]).
#[derive(Clone, Debug)]
pub struct Features {
    pub path: std::path::PathBuf,
    pub count: usize,
    rows: BTreeMap<usize, (Array1<f64>, f64, Array1<f64>)>,
}

impl Features {
    /// Features `features`' encoder rows (features × width), biases and decoder rows (features ×
    /// width), loaded by [`Weights::load_features`].
    pub(crate) fn stacked(&self, features: &[usize], width: usize) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>), String> {
        let (mut g, mut c, mut u) = (Array2::zeros((features.len(), width)), Array1::zeros(features.len()), Array2::zeros((features.len(), width)));
        for (i, f) in features.iter().enumerate() {
            let (gi, ci, ui) = self.rows.get(f).ok_or_else(|| format!("feature {f} not loaded (Weights::load_features)"))?;
            g.row_mut(i).assign(gi);
            c[i] = *ci;
            u.row_mut(i).assign(ui);
        }
        Ok((g, c, u))
    }

    /// The write of features `features` on normed inputs `x_hat` (rows × width).
    fn write(&self, features: &[usize], x_hat: &Array2<f64>) -> Result<Array2<f64>, String> {
        let mut out = Array2::<f64>::zeros(x_hat.dim());
        for f in features {
            let (g, c, u) = self.rows.get(f).ok_or_else(|| format!("feature {f} not loaded (Weights::load_features)"))?;
            let a = x_hat.dot(g).mapv(|t| (t + c).max(0.0));
            out += &(a.insert_axis(Axis(1)) * &u.view().insert_axis(Axis(0)));
        }
        Ok(out)
    }
}

impl Weights {
    /// `M`'s blocks with no decomposition views attached; a tied unembedding (equal to the
    /// embedding) is kept once.
    pub fn new(layers: Vec<LayerWeights>, final_norm: Norm, unembedding: Stored, embedding: Stored) -> Self {
        let embedding = Arc::new(embedding);
        let unembedding = if unembedding == *embedding { embedding.clone() } else { Arc::new(unembedding) };
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let generation = Arc::new(Generation(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        Self { layers, final_norm, unembedding, embedding, transcoders: BTreeMap::new(), vpd: BTreeMap::new(), vpd_attention: BTreeMap::new(), library: BTreeMap::new(), generation }
    }

    /// Attaches the transcoder view: `dir/layer_{l}.safetensors` per layer that has one.
    pub fn attach_transcoders(&mut self, dir: &std::path::Path) -> Result<usize, String> {
        let mut attached = 0;
        for l in 0..self.layers.len() {
            let path = dir.join(format!("layer_{l}.safetensors"));
            if !path.exists() {
                continue;
            }
            let t = crate::library_transcoder::Transcoder::open(&path)?;
            if t.width != self.width() {
                return Err(format!("{}: width {} for a model of width {}", path.display(), t.width, self.width()));
            }
            self.transcoders.insert(l, Features { path, count: t.features, rows: BTreeMap::new() });
            attached += 1;
        }
        Ok(attached)
    }

    /// Attaches VPD's view of the MLPs from a decomposition export (`explanation_battery::load_factors`).
    pub fn attach_vpd(&mut self, dir: &std::path::Path) -> Result<usize, String> {
        use crate::explanation_battery::Kind;
        let factors = crate::explanation_battery::load_factors(dir)?;
        let mut attached = 0;
        for l in 0..self.layers.len() {
            let find = |kind: Kind| factors.iter().find(|f| f.layer == l && f.kind == kind);
            if let (Some(q), Some(k), Some(v), Some(o)) = (find(Kind::Query), find(Kind::Key), find(Kind::Value), find(Kind::Output)) {
                let lw = &self.layers[l];
                if lw.heads.iter().any(|h| h.query_norm.is_some() || h.key_norm.is_some()) {
                    return Err(format!("layer {l}: the VPD view reads attention without head norms"));
                }
                let [wq, wk, wv, wo] = attention_maps(lw);
                for (name, f, w) in [("q", q, &wq), ("k", k, &wk), ("v", v, &wv), ("o", o, &wo)] {
                    if f.u.ncols() != w.nrows() || f.v.nrows() != w.ncols() {
                        return Err(format!("layer {l}: VPD's {name}_proj factors do not fit its {} × {} matrix", w.nrows(), w.ncols()));
                    }
                }
                let pair = |f: &crate::explanation_battery::Factors| (f.u.clone(), f.v.clone());
                self.vpd_attention.insert(l, VpdAttention { q: pair(q), k: pair(k), v: pair(v), o: pair(o) });
            }
            let (Some(fc), Some(down), Some(mlp)) = (find(Kind::Up), find(Kind::Down), self.layers[l].mlp.as_ref()) else { continue };
            if mlp.up.is_some() {
                return Err(format!("layer {l}: the VPD view reads a plain (ungated) MLP"));
            }
            let (n, d) = mlp.gate.dim();
            if fc.u.ncols() != n || fc.v.nrows() != d || down.u.ncols() != d || down.v.nrows() != n {
                return Err(format!("layer {l}: VPD's MLP factors do not fit the MLP ({n} × {d})"));
            }
            self.vpd.insert(l, VpdMlp { fc_u: fc.u.clone(), fc_v: fc.v.clone(), down_u: down.u.clone(), down_v: down.v.clone() });
            attached += 1;
        }
        Ok(attached)
    }

    /// Attaches our library's parts from decomp's start (`start.components.json`, arm `arm`): each
    /// component's slices `[site, index]` (site `6 l + k`, `k` in VPD's order q, k, v, o, c_fc,
    /// down_proj) name VPD subcomponents, so the VPD view must be attached. Part `i` of layer `l`'s
    /// attention or MLP is the `i`-th such component in file order (mech's `PD.lib[l].attn[i]`,
    /// `.mlp[i]`).
    pub fn attach_library(&mut self, start: &std::path::Path, arm: &str) -> Result<usize, String> {
        #[derive(Deserialize)]
        struct Component {
            slices: Vec<[usize; 2]>,
        }
        #[derive(Deserialize)]
        struct Arm {
            arm: String,
            components: Vec<Component>,
        }
        let arms: Vec<Arm> = serde_json::from_slice(&std::fs::read(start).map_err(|e| format!("{}: {e}", start.display()))?).map_err(|e| e.to_string())?;
        let components = arms.into_iter().find(|a| a.arm == arm).ok_or_else(|| format!("{}: no arm {arm}", start.display()))?.components;
        let mut library: BTreeMap<(usize, bool), Vec<[Vec<usize>; 4]>> = BTreeMap::new();
        for c in &components {
            let Some(&[first, _]) = c.slices.first() else { return Err("a library part of no slices".into()) };
            let (layer, attention) = (first / 6, first % 6 < 4);
            let mut lists: [Vec<usize>; 4] = Default::default();
            for &[site, index] in &c.slices {
                if site / 6 != layer || (site % 6 < 4) != attention {
                    return Err(format!("a library part spans sites {first} and {site}"));
                }
                let (kind, count) = match site % 6 {
                    k @ 0..4 => {
                        let a = self.vpd_attention.get(&layer).ok_or_else(|| format!("layer {layer}: attach VPD's attention view first"))?;
                        (k, [&a.q, &a.k, &a.v, &a.o][k].0.nrows())
                    }
                    k => {
                        let m = self.vpd.get(&layer).ok_or_else(|| format!("layer {layer}: attach VPD's MLP view first"))?;
                        (k - 4, if k == 4 { m.fc_u.nrows() } else { m.down_u.nrows() })
                    }
                };
                if index >= count {
                    return Err(format!("library slice [{site}, {index}] past {count} subcomponents"));
                }
                lists[kind].push(index);
            }
            library.entry((layer, attention)).or_default().push(lists);
        }
        let parts = components.len();
        self.library = library;
        Ok(parts)
    }

    /// Reads the rows of every transcoder feature `graph`'s nodes declare.
    pub fn load_features(&mut self, graph: &Graph) -> Result<(), String> {
        for block in &graph.blocks {
            let Block::Features { layer, features, .. } = block else { continue };
            let t = self.transcoders.get_mut(layer).ok_or_else(|| format!("layer {layer} has no transcoder"))?;
            let missing: Vec<usize> = features.iter().copied().filter(|f| !t.rows.contains_key(f)).collect();
            if missing.is_empty() {
                continue;
            }
            let file = crate::safetensors::SafetensorsFile::open(&t.path).map_err(|e| e.to_string())?;
            let d = file.vector("b_dec", self.embedding.ncols()).map_err(|e| e.to_string())?.len();
            let (encoder, decoder) = (file.stored("W_enc", t.count, d).map_err(|e| e.to_string())?, file.stored("W_dec", t.count, d).map_err(|e| e.to_string())?);
            let bias = file.vector("b_enc", t.count).map_err(|e| e.to_string())?;
            for f in missing {
                let row = |m: &crate::safetensors::Stored| m.rows(f..f + 1).map(|r| r.matrix().row(0).to_owned()).ok_or_else(|| format!("feature {f} out of range"));
                t.rows.insert(f, (row(&encoder)?, bias[f], row(&decoder)?));
            }
        }
        Ok(())
    }

    /// `M`'s blocks read straight from its split native program (`run_check::split_sites`) and
    /// its `layers` (`run_check::layer_nodes`), as [`Weights::of`] reads them from the start
    /// library but without building one (a library on the host holds several more copies of a
    /// model's matrices: Qwen3-0.6B's load went past 24 GiB).
    pub fn from_native(native: &crate::operator_program::OperatorProgram, layers: &[crate::run_check::LayerNodes]) -> Result<Self, String> {
        use crate::operator_program::Node;
        let norm_of = |normed: usize| -> Result<Norm, String> {
            let Node::Affine { terms, .. } = &native.nodes[normed] else { return Err(format!("node {normed} is not a normed stream")) };
            let [(rms, gain)] = terms[..] else { return Err(format!("node {normed} is not one gain of a norm")) };
            let Node::RmsNorm { epsilon, .. } = native.nodes[rms] else { return Err(format!("node {normed} does not read an RMS norm")) };
            Ok(Norm { gain: native.operators[gain].matrix().diag().to_owned(), epsilon })
        };
        let narrow = |m: Array2<f64>| m.mapv(|v| v as f32);
        let map = |node: usize, input: usize| -> Result<(Stored, Option<Array1<f64>>), String> {
            match &native.nodes[node] {
                Node::Affine { terms, bias } if terms.len() == 1 && terms[0].0 == input => Ok((narrow(native.operators[terms[0].1].matrix()), bias.map(|b| native.operators[b].matrix().column(0).to_owned()))),
                other => Err(format!("node {node} is not an affine map of node {input}: {other:?}")),
            }
        };
        let head_norm = |node: usize| -> Result<Option<(Array1<f64>, f64)>, String> {
            if crate::run_check::head_projection(native, node) == node {
                return Ok(None);
            }
            let Node::Affine { terms, .. } = &native.nodes[node] else { return Err(format!("node {node}: a head norm")) };
            let Node::RmsNorm { epsilon, .. } = native.nodes[terms[0].0] else { return Err(format!("node {node}: a head norm")) };
            Ok(Some((native.operators[terms[0].1].matrix().diag().to_owned(), epsilon)))
        };
        let mut out = Vec::with_capacity(layers.len());
        for (l, layer) in layers.iter().enumerate() {
            let Node::Affine { terms: outputs, .. } = &native.nodes[layer.attention] else { return Err(format!("layer {l}: the attention output is not a map")) };
            let mut heads = Vec::with_capacity(layer.reads.len());
            // A key/value group's heads read one key and one value node: one map each.
            let mut shared: BTreeMap<usize, Arc<Stored>> = BTreeMap::new();
            let mut map_of = |node: usize, x: usize| -> Result<Arc<Stored>, String> {
                if let Some(m) = shared.get(&node) {
                    return Ok(m.clone());
                }
                let m = Arc::new(map(node, x)?.0);
                shared.insert(node, m.clone());
                Ok(m)
            };
            for (h, &read) in layer.reads.iter().enumerate() {
                let Node::Attend { query, key, value, scale, rotary, causal } = native.nodes[read].clone() else { return Err(format!("layer {l} head {h}: the read is not an attention")) };
                let x = layer.normed_stream;
                let output = outputs.iter().find(|(n, _)| *n == read).map(|(_, op)| narrow(native.operators[*op].matrix())).ok_or_else(|| format!("layer {l} head {h}: no output columns"))?;
                heads.push(HeadWeights {
                    query: map(crate::run_check::head_projection(native, query), x)?.0,
                    query_norm: head_norm(query)?,
                    key: map_of(crate::run_check::head_projection(native, key), x)?,
                    key_norm: head_norm(key)?,
                    value: map_of(value, x)?,
                    output,
                    scale: scale.value(),
                    rotary,
                    causal,
                });
            }
            let x = layer.normed;
            let zeros = |n: usize| Array1::<f64>::zeros(n);
            let (gate_node, up_node, laws) = match &native.nodes[layer.active] {
                Node::Hadamard { left, right } => {
                    let Node::Pointwise { input, laws } = &native.nodes[*left] else { return Err(format!("layer {l}: the gated product's left factor is not one law")) };
                    (*input, Some(*right), laws.clone())
                }
                Node::Pointwise { input, laws } => (*input, None, laws.clone()),
                other => return Err(format!("layer {l}: the MLP's activations are {other:?}")),
            };
            let law = *laws.first().ok_or("an MLP of no units")?;
            if laws.iter().any(|w| *w != law) {
                return Err(format!("layer {l}: the MLP's units have different laws"));
            }
            let (gate, bias) = map(gate_node, x)?;
            let n = gate.nrows();
            let (up, up_bias) = match up_node {
                Some(u) => {
                    let (m, b) = map(u, x)?;
                    (Some(m), b.unwrap_or_else(|| zeros(n)))
                }
                None => (None, zeros(n)),
            };
            out.push(LayerWeights {
                attention: norm_of(layer.normed_stream)?,
                heads,
                mlp_norm: norm_of(x)?,
                mlp: Some(MlpWeights { bias: bias.unwrap_or_else(|| zeros(n)), gate, up, up_bias, out: map(layer.mlp, layer.active)?.0, law }),
            });
        }
        let head = crate::resident_causal_fit::fixed_head_target::Head::of(native)?;
        let final_norm = norm_of(head.hidden)?;
        let unembedding = head.embedding().mapv(|v| v as f32);
        let feature = native.nodes.iter().position(|n| matches!(n, Node::Feature { .. })).ok_or("no token feature")?;
        let embedding = native
            .nodes
            .iter()
            .find_map(|n| match n {
                Node::Affine { terms, bias: None } if terms.len() == 1 && terms[0].0 == feature => Some(native.operators[terms[0].1].matrix()),
                _ => None,
            })
            .ok_or("no token embedding")?;
        let embedding = if embedding.nrows() == unembedding.ncols() { narrow(embedding.t().to_owned()) } else { narrow(embedding) };
        Ok(Self::new(out, final_norm, unembedding, embedding))
    }

    /// The blocks of `library` (the start library of `M` equals `M`).
    pub fn of(library: &Library) -> Self {
        library.graph_weights()
    }

    /// The device's key for these weights.
    pub(crate) fn generation(&self) -> u64 {
        self.generation.0
    }

    pub(crate) fn width(&self) -> usize {
        self.embedding.ncols()
    }

    fn neurons(&self, layer: usize) -> usize {
        self.layers[layer].mlp.as_ref().map_or(0, |m| m.gate.nrows())
    }

    /// Whether a VPD view is attached: programs then delete what they leave undeclared by default
    /// (`Program::standin`).
    pub fn has_vpd(&self) -> bool {
        !self.vpd.is_empty() || !self.vpd_attention.is_empty()
    }

    /// The parts a program may name: every head and neuron, every VPD subcomponent and remainder of
    /// an attached view, every transcoder feature and every library part.
    pub fn vocabulary(&self) -> usize {
        let native: usize = (0..self.layers.len()).map(|l| self.layers[l].heads.len() + self.neurons(l)).sum();
        let mlp: usize = self.vpd.values().map(|v| v.fc_u.nrows() + v.down_u.nrows() + 2).sum();
        let attention: usize = self.vpd_attention.values().map(|a| [&a.q, &a.k, &a.v, &a.o].iter().map(|f| f.0.nrows() + 1).sum::<usize>()).sum();
        let features: usize = self.transcoders.values().map(|t| t.count).sum();
        let library: usize = self.library.values().map(Vec::len).sum();
        native + mlp + attention + features + library
    }
}

// ------------------------------------------------------------------------------ resolved graph

/// The pieces of one site: some heads of a layer's attention, some neurons of its MLP, or some
/// transcoder features of its MLP (`rest`: the MLP minus those features, i.e. every other feature
/// and the transcoder's exact error piece).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Block {
    Heads { layer: usize, heads: Vec<usize> },
    Neurons { layer: usize, neurons: Vec<usize> },
    Features { layer: usize, features: Vec<usize>, rest: bool },
    /// VPD subcomponents of the MLP's `c_fc` (`fc`) and `down_proj` (`down`); with `rest`, every
    /// other subcomponent of both and the remainders.
    Slices { layer: usize, fc: Vec<usize>, down: Vec<usize>, rest: bool },
    /// VPD subcomponents of the attention's `q_proj`, `k_proj`, `v_proj` and `o_proj`; with `rest`,
    /// every other subcomponent of the four and the remainders.
    AttnSlices { layer: usize, q: Vec<usize>, k: Vec<usize>, v: Vec<usize>, o: Vec<usize>, rest: bool },
}

impl Block {
    /// The read site: `2l` a layer's attention, `2l + 1` its MLP; the logits read at `2L`.
    pub(crate) fn site(&self) -> usize {
        match self {
            Self::Heads { layer, .. } | Self::AttnSlices { layer, .. } => 2 * layer,
            Self::Neurons { layer, .. } | Self::Features { layer, .. } | Self::Slices { layer, .. } => 2 * layer + 1,
        }
    }

    /// Whether the block writes the residual stream (a VPD node of `c_fc` subcomponents alone writes
    /// only its MLP's hidden pre-activation).
    fn writes_residual(&self) -> bool {
        match self {
            Self::Slices { down, rest, .. } => *rest || !down.is_empty(),
            Self::AttnSlices { o, rest, .. } => *rest || !o.is_empty(),
            _ => true,
        }
    }

    /// The residual routes the block reads (a VPD attention node: those its q, k, v subcomponents
    /// read; a down_proj-only VPD node: none).
    fn reads(&self) -> Vec<Route> {
        match self {
            Self::AttnSlices { q, k, v, rest, .. } => [(Route::Query, q), (Route::Key, k), (Route::Value, v)].into_iter().filter(|(_, p)| *rest || !p.is_empty()).map(|(r, _)| r).collect(),
            Self::Slices { fc, rest, .. } if fc.is_empty() && !rest => Vec::new(),
            other => other.routes().to_vec(),
        }
    }

    pub(crate) fn routes(&self) -> &'static [Route] {
        match self {
            Self::Heads { .. } | Self::AttnSlices { .. } => &[Route::Query, Route::Key, Route::Value],
            Self::Neurons { .. } | Self::Features { .. } | Self::Slices { .. } => &[Route::Input],
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Heads { heads, .. } => heads.is_empty(),
            Self::Neurons { neurons, .. } => neurons.is_empty(),
            Self::Features { features, rest, .. } => features.is_empty() && !rest,
            Self::Slices { fc, down, rest, .. } => fc.is_empty() && down.is_empty() && !rest,
            Self::AttnSlices { q, k, v, o, rest, .. } => q.is_empty() && k.is_empty() && v.is_empty() && o.is_empty() && !rest,
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

    /// Whether the route reads writer `w`'s write.
    fn reads(&self, w: &Writer) -> bool {
        match self {
            Self::AllBut(cut) => !cut.contains(w),
            Self::Only(kept) => kept.contains(w),
        }
    }

}

/// One unit of a circuit: its pieces and whether it computes (else it writes its stand-in).
#[derive(Clone, Debug)]
pub struct Unit {
    pub block: Block,
    pub computes: bool,
    /// A computing unit that acts only at some positions ([`Positions`], and whether inverted: a
    /// complement model's node, computing everywhere but at its positions); none, everywhere.
    pub at: Option<(Arc<Positions>, bool)>,
    /// Per route slot, what it reads.
    pub routes: [Incoming; 3],
    /// For a VPD MLP block: the units of its site whose `c_fc` writes its `down_proj` subcomponents
    /// read (the hidden pre-activation; the rest at their counterfactual values).
    pub hidden: Incoming,
}

/// A program or `M` as units in site order of computation plus the logits' input. The program's
/// declared nodes are units `0..nodes` in both.
#[derive(Clone, Debug)]
pub struct Circuit {
    pub units: Vec<Unit>,
    pub logits: Incoming,
    pub nodes: usize,
    /// Stand-ins are zero ([`Graph::delete`]): `Checker::referenced` attaches [`Reference::zeros`].
    pub delete: bool,
    /// Per node unit its positions ([`Positions`]; none: everywhere): a swap of a node's writes
    /// ([`execute`]'s `swaps`) replaces them at those positions only.
    pub places: Vec<Option<Arc<Positions>>>,
}


impl Circuit {
    /// Whether this is `M` itself: every unit computing, every edge kept (no stand-in is read).
    pub fn is_model(&self) -> bool {
        let all = |i: &Incoming| matches!(i, Incoming::AllBut(cut) if cut.is_empty());
        all(&self.logits) && self.units.iter().all(|u| u.computes && u.routes.iter().all(all) && all(&u.hidden))
    }
}

/// A parsed program: its nodes' blocks by id and its edges as (writer, reader, route), the reader
/// `None` for the logits.
#[derive(Clone, Debug)]
pub struct Graph {
    /// Undeclared pieces and edges contribute nothing (`Program::standin` "delete"); else they carry
    /// their values on the counterfactual.
    pub delete: bool,
    pub ids: Vec<String>,
    pub blocks: Vec<Block>,
    /// Per node its attention claim ([`Claim`]): checked, never executed.
    pub claims: Vec<Option<Claim>>,
    pub edges: Vec<(Writer, Option<usize>, Route)>,
    /// Edges within one MLP site, (writer node, reader node): `c_fc` subcomponents to `down_proj`
    /// subcomponents through the hidden pre-activation.
    pub internal: Vec<(usize, usize)>,
    /// The program's alignments with their nodes as indices ([`Checker::alignment_error`]).
    pub alignments: Vec<(AlignmentIr, Vec<usize>)>,
    /// Per node where it acts ([`Positions`]; none: everywhere).
    pub positions: Vec<Option<Arc<Positions>>>,
    /// The shared base's nodes (`Program::base`): always on, connected to every node, kept by
    /// necessity's deletion, priced apart.
    pub base: BTreeSet<usize>,
    /// How many of the last `edges` and `internal` the base implies (not declared): they cost no
    /// structure bits.
    pub implied: (usize, usize),
    /// The named groups the program uses ([`GroupIr`]).
    pub groups: Vec<GroupUse>,
}

/// Whether a same-site edge from `writer` to `reader` joins two parts of one VPD site: c_fc
/// subcomponents feeding down_proj subcomponents, or q/k/v subcomponents feeding o_proj ones.
fn joins(writer: &Block, reader: &Block) -> bool {
    match (writer, reader) {
        (Block::Slices { fc, .. }, Block::Slices { down, .. }) => !fc.is_empty() && !down.is_empty(),
        (Block::AttnSlices { q, k, v, .. }, Block::AttnSlices { o, .. }) => !(q.is_empty() && k.is_empty() && v.is_empty()) && !o.is_empty(),
        _ => false,
    }
}

/// A built-in attention rule (design.txt section 5, "Rules"): uniform over the positions `j ≤ t` it
/// picks, position 0 when it picks none. A [`Claim`]'s convenience form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeadRule {
    /// `j = t − k`.
    Offset(usize),
    /// Every `j < t` with `key(j) == query(t)`.
    Match { query: TokenExpr, key: TokenExpr },
    /// `j = 0`.
    First,
}

/// A token expression of a match rule: the tokens, or an expression read `k` positions earlier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenExpr {
    Tokens,
    Shift(Box<TokenExpr>, usize),
}

impl TokenExpr {
    fn parse(v: &serde_json::Value) -> Result<Self, String> {
        match v.get("op").and_then(serde_json::Value::as_str) {
            Some("tokens") => Ok(Self::Tokens),
            Some("shift") => {
                let by = v.get("by").and_then(serde_json::Value::as_u64).ok_or("shift: a whole number of positions \"by\"")?;
                Ok(Self::Shift(Box::new(Self::parse(v.get("arg").ok_or("shift: an \"arg\"")?)?), by as usize))
            }
            _ => Err(format!("token expression {v}: tokens or shift")),
        }
    }

    /// The expression's value at position `t` of `tokens` (none before the sequence starts).
    fn at(&self, tokens: &[u32], t: usize) -> Option<u32> {
        match self {
            Self::Tokens => tokens.get(t).copied(),
            Self::Shift(e, k) => t.checked_sub(*k).and_then(|u| e.at(tokens, u)),
        }
    }
}

impl HeadRule {
    pub fn parse(v: &serde_json::Value) -> Result<Self, String> {
        if v.get("op").and_then(serde_json::Value::as_str) != Some("attend") {
            return Err(format!("rule {v}: only attend rules"));
        }
        if let Some(k) = v.get("offset") {
            return Ok(Self::Offset(k.as_u64().ok_or("attend: a whole-number offset")? as usize));
        }
        if v.get("first").and_then(serde_json::Value::as_bool) == Some(true) {
            return Ok(Self::First);
        }
        match (v.get("query"), v.get("key")) {
            (Some(q), Some(k)) => Ok(Self::Match { query: TokenExpr::parse(q)?, key: TokenExpr::parse(k)? }),
            _ => Err(format!("attend rule {v}: offset, first, or query and key")),
        }
    }

    /// The attention weights over one sequence of `tokens` (positions × positions).
    pub fn pattern(&self, tokens: &[u32]) -> Array2<f64> {
        let n = tokens.len();
        let mut a = Array2::<f64>::zeros((n, n));
        for t in 0..n {
            let picked: Vec<usize> = match self {
                Self::Offset(k) => t.checked_sub(*k).into_iter().collect(),
                Self::First => vec![0],
                Self::Match { query, key } => match query.at(tokens, t) {
                    Some(q) => (0..t).filter(|&j| key.at(tokens, j) == Some(q)).collect(),
                    None => Vec::new(),
                },
            };
            let picked = if picked.is_empty() { vec![0] } else { picked };
            let w = 1.0 / picked.len() as f64;
            for j in picked {
                a[[t, j]] = w;
            }
        }
        a
    }
}

/// An attention claim about a node's parts (design_v2 section 1): the pattern their queries and
/// keys produce, as a built-in rule or as data (a Python function of the program, evaluated by mech
/// on each prompt and counterfactual of the behavior). The parts run as the model's weights; the
/// checker scores the claim against the pattern they produce ([`Checker::claim_error`]).
#[derive(Clone, Debug, PartialEq)]
pub enum Claim {
    Rule(HeadRule),
    /// Per prompt of the behavior and per counterfactual, in prompt order, the claimed pattern
    /// (positions × positions; row `t` a distribution over positions `0..=t`).
    Pattern { prompts: Vec<Array2<f64>>, counterfactuals: Vec<Array2<f64>> },
}

impl Claim {
    pub fn parse(v: &serde_json::Value) -> Result<Self, String> {
        match v.get("op").and_then(serde_json::Value::as_str) {
            Some("attend") => Ok(Self::Rule(HeadRule::parse(v)?)),
            Some("pattern") => {
                let tables = |key: &str| -> Result<Vec<Array2<f64>>, String> {
                    match v.get(key) {
                        None => Ok(Vec::new()),
                        Some(list) => list.as_array().ok_or_else(|| format!("pattern claim: \"{key}\" is a list of patterns"))?.iter().map(pattern_table).collect(),
                    }
                };
                Ok(Self::Pattern { prompts: tables("prompts")?, counterfactuals: tables("counterfactuals")? })
            }
            _ => Err(format!("claim {v}: an attend rule or a pattern")),
        }
    }

    /// The claimed pattern on `tokens`, the `index`-th prompt of the behavior or its counterfactual.
    pub fn pattern(&self, tokens: &[u32], index: usize, counterfactual: bool) -> Result<Array2<f64>, String> {
        match self {
            Self::Rule(r) => Ok(r.pattern(tokens)),
            Self::Pattern { prompts, counterfactuals } => {
                let what = if counterfactual { "counterfactual" } else { "prompt" };
                let table = if counterfactual { counterfactuals } else { prompts }.get(index).ok_or_else(|| format!("the claimed pattern has no {what} {index}"))?;
                if table.nrows() != tokens.len() {
                    return Err(format!("the claimed pattern of {what} {index} has {} positions for {} tokens", table.nrows(), tokens.len()));
                }
                Ok(table.clone())
            }
        }
    }
}

/// One sequence's claimed pattern from rows of weights: row `t` weighs positions `0..=t` (or every
/// position, zero past `t`); each row is normalized.
fn pattern_table(v: &serde_json::Value) -> Result<Array2<f64>, String> {
    let rows = v.as_array().ok_or("a pattern is a list of rows")?;
    let n = rows.len();
    let mut out = Array2::<f64>::zeros((n, n));
    for (t, row) in rows.iter().enumerate() {
        let w: Vec<f64> = row.as_array().ok_or("a pattern row is a list of weights")?.iter().map(|x| x.as_f64().ok_or("a pattern weight is a number")).collect::<Result<_, _>>()?;
        if w.len() != t + 1 && w.len() != n {
            return Err(format!("pattern row {t}: {} weights for positions 0..={t}", w.len()));
        }
        if w.iter().any(|x| !x.is_finite() || *x < 0.0) || w.iter().skip(t + 1).any(|x| *x != 0.0) {
            return Err(format!("pattern row {t}: weights are nonnegative and zero past position {t}"));
        }
        let total: f64 = w.iter().sum();
        if total <= 0.0 {
            return Err(format!("pattern row {t}: no weight"));
        }
        for (j, x) in w.iter().enumerate().take(t + 1) {
            out[[t, j]] = x / total;
        }
    }
    Ok(out)
}

/// The attention patterns `block`'s queries and keys produce on sequences `spans` of a batch whose
/// normed attention inputs at the block's layer are `x_hat` (`M`'s): per head with its weight in
/// the node's claim, per sequence (positions × positions). Native heads: each head's own queries
/// and keys, equal weights. VPD or library parts: queries from the node's q_proj subcomponents and
/// keys from its k_proj ones (the model's own of a kind it names none of), in each head they reach,
/// weighed by the attention logits they produce there (`Σ |q_t · k_j|` over `j ≤ t`, scaled), so a
/// head the subcomponents barely write counts little.
pub(crate) fn produced_patterns(weights: &Weights, block: &Block, x_hat: &Array2<f64>, spans: &[(usize, usize)], blocks: &[Vec<[usize; 4]>]) -> Result<Vec<(f64, Vec<Array2<f64>>)>, String> {
    let heads: Vec<(Array2<f64>, Array2<f64>, &HeadWeights, bool)> = match block {
        Block::Heads { layer, heads } => {
            let lw = weights.layers.get(*layer).ok_or("no such layer")?;
            heads.iter().map(|&h| {
                let w = &lw.heads[h];
                (project(x_hat, &w.query, w.query_norm.as_ref()), project(x_hat, &w.key, w.key_norm.as_ref()), w, false)
            }).collect()
        }
        Block::AttnSlices { layer, q, k, .. } => {
            let lw = weights.layers.get(*layer).ok_or("no such layer")?;
            let a = weights.vpd_attention.get(layer).ok_or_else(|| format!("layer {layer}'s attention has no VPD view"))?;
            let maps = attention_maps(lw);
            let side = |list: &[usize], factors: &(Array2<f64>, Array2<f64>), map: &Array2<f64>| if list.is_empty() { x_hat.dot(&map.t()) } else { sliced(factors, map, list, false, x_hat) };
            let (qs, ks) = (side(q, &a.q, &maps[0]), side(k, &a.k, &maps[1]));
            let mut out = Vec::new();
            let (mut qc, mut kc) = (0, 0);
            for w in &lw.heads {
                let (qw, kw) = (w.query.nrows(), w.key.nrows());
                out.push((qs.slice(s![.., qc..qc + qw]).to_owned(), ks.slice(s![.., kc..kc + kw]).to_owned(), w, true));
                (qc, kc) = (qc + qw, kc + kw);
            }
            out
        }
        _ => return Err("an attention claim on a block without queries and keys".into()),
    };
    let mut out = Vec::with_capacity(heads.len());
    for (qh, kh, w, weighed) in heads {
        let mut patterns = Vec::with_capacity(spans.len());
        let mut logits = 0.0;
        for (n, &(start, length)) in spans.iter().enumerate() {
            let positions: Vec<u32> = (0..length as u32).collect();
            let span = start..start + length;
            let (qs, ks) = (qh.slice(s![span.clone(), ..]).to_owned(), kh.slice(s![span, ..]).to_owned());
            let (qs, ks) = (rotate(&qs, w.rotary, &positions, false), rotate(&ks, w.rotary, &positions, false));
            if weighed {
                let scores = qs.dot(&ks.t());
                logits += scores.indexed_iter().filter(|((t, j), _)| j <= t).map(|(_, x)| x.abs()).sum::<f64>() * w.scale;
            }
            let mut a = probabilities(qs.view(), ks.view(), &positions, 0, w.scale, w.causal);
            if let Some(b) = blocks.get(n).filter(|b| !b.is_empty()) {
                block_attention(&mut a, b);
            }
            patterns.push(a);
        }
        out.push((if weighed { logits } else { 1.0 }, patterns));
    }
    Ok(out)
}

/// A VPD matrix's subcomponents `index` names ([`indices`]), or with "rest" its remainder
/// `W − Σ U Vᵀ`, numbered `count` (one past the last subcomponent).
fn subcomponents(index: &Option<Index>, count: usize, what: &str) -> Result<Vec<usize>, String> {
    match index {
        Some(Index::Name(n)) if n == "rest" => Ok(vec![count]),
        other => indices(other, count, what),
    }
}

fn indices(index: &Option<Index>, count: usize, what: &str) -> Result<Vec<usize>, String> {
    let out = match index {
        None => (0..count).collect(),
        Some(Index::One(i)) => vec![*i],
        Some(Index::Many(v)) => v.clone(),
        Some(Index::Name(n)) => return Err(format!("{what}: index {n} names no unit (\"rest\" names a VPD matrix's remainder)")),
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
        let mut claims: Vec<Option<Claim>> = Vec::new();
        let mut owned: BTreeSet<(usize, bool, usize)> = BTreeSet::new();
        let mut featured: BTreeSet<(usize, usize)> = BTreeSet::new();
        let mut sliced: BTreeSet<(usize, usize, usize)> = BTreeSet::new();
        let mut views: BTreeMap<(usize, bool), String> = BTreeMap::new();
        for node in &program.nodes {
            if node.id == "embed" || node.id == "logits" || ids.contains(&node.id) {
                return Err(format!("node id {} is reserved or repeated", node.id));
            }
            let claim = match node.claim.as_ref().filter(|r| !r.is_null()) {
                Some(r) => Some(Claim::parse(r).map_err(|e| format!("{}: {e}", node.id))?),
                None => None,
            };
            let mut block: Option<Block> = None;
            for piece in &node.pieces {
                if piece.layer >= layers {
                    return Err(format!("{}: layer {} out of range (there are {layers})", node.id, piece.layer));
                }
                let l = piece.layer;
                let next = match (piece.view.as_str(), piece.kind.as_str()) {
                    ("native", "head") => Block::Heads { layer: l, heads: indices(&piece.index, weights.layers[l].heads.len(), "head")? },
                    ("native", "mlp" | "neuron") => Block::Neurons { layer: l, neurons: indices(&piece.index, weights.neurons(l), "neuron")? },
                    ("transcoder", "feature") => {
                        let t = weights.transcoders.get(&l).ok_or_else(|| format!("{}: layer {l} has no transcoder view", node.id))?;
                        if piece.index.is_none() {
                            return Err(format!("{}: name the transcoder features of layer {l}", node.id));
                        }
                        Block::Features { layer: l, features: indices(&piece.index, t.count, "feature")?, rest: false }
                    }
                    ("library", kind @ ("attn" | "mlp")) => {
                        let attention = kind == "attn";
                        let parts = weights.library.get(&(l, attention)).ok_or_else(|| format!("{}: layer {l}'s {kind} has no library view", node.id))?;
                        if piece.index.is_none() {
                            return Err(format!("{}: name the library parts of layer {l}'s {kind}", node.id));
                        }
                        let mut lists: [Vec<usize>; 4] = Default::default();
                        for p in indices(&piece.index, parts.len(), "library part")? {
                            for (all, more) in lists.iter_mut().zip(&parts[p]) {
                                all.extend(more.iter().copied());
                            }
                        }
                        let [a, b, c, d] = lists;
                        if attention { Block::AttnSlices { layer: l, q: a, k: b, v: c, o: d, rest: false } } else { Block::Slices { layer: l, fc: a, down: b, rest: false } }
                    }
                    ("vpd", kind @ ("q_proj" | "k_proj" | "v_proj" | "o_proj")) => {
                        let a = weights.vpd_attention.get(&l).ok_or_else(|| format!("{}: layer {l}'s attention has no VPD view", node.id))?;
                        if piece.index.is_none() {
                            return Err(format!("{}: name the {kind} subcomponents of layer {l}", node.id));
                        }
                        let (q, k, v, o) = (&a.q.0, &a.k.0, &a.v.0, &a.o.0);
                        let count = match kind {
                            "q_proj" => q.nrows(),
                            "k_proj" => k.nrows(),
                            "v_proj" => v.nrows(),
                            _ => o.nrows(),
                        };
                        let picked = subcomponents(&piece.index, count, kind)?;
                        let mut lists: [Vec<usize>; 4] = Default::default();
                        lists[["q_proj", "k_proj", "v_proj", "o_proj"].iter().position(|k| *k == kind).unwrap_or(3)] = picked;
                        let [q, k, v, o] = lists;
                        Block::AttnSlices { layer: l, q, k, v, o, rest: false }
                    }
                    ("vpd", kind @ ("c_fc" | "down_proj")) => {
                        let v = weights.vpd.get(&l).ok_or_else(|| format!("{}: layer {l}'s MLP has no VPD view", node.id))?;
                        let (count, fc) = if kind == "c_fc" { (v.fc_u.nrows(), true) } else { (v.down_u.nrows(), false) };
                        if piece.index.is_none() {
                            return Err(format!("{}: name the {kind} subcomponents of layer {l}", node.id));
                        }
                        let picked = subcomponents(&piece.index, count, kind)?;
                        if fc { Block::Slices { layer: l, fc: picked, down: Vec::new(), rest: false } } else { Block::Slices { layer: l, fc: Vec::new(), down: picked, rest: false } }
                    }
                    (view, kind) => return Err(format!("{}: {view} piece of kind {kind} is not resolved (native heads and neurons, transcoder features, VPD c_fc and down_proj subcomponents)", node.id)),
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
                    (Some(Block::Features { layer, mut features, .. }), Block::Features { layer: m, features: more, .. }) if layer == m => {
                        features.extend(more);
                        Block::Features { layer, features, rest: false }
                    }
                    (Some(Block::Slices { layer, mut fc, mut down, .. }), Block::Slices { layer: m, fc: f, down: dn, .. }) if layer == m => {
                        fc.extend(f);
                        down.extend(dn);
                        Block::Slices { layer, fc, down, rest: false }
                    }
                    (Some(Block::AttnSlices { layer, mut q, mut k, mut v, mut o, .. }), Block::AttnSlices { layer: m, q: q2, k: k2, v: v2, o: o2, .. }) if layer == m => {
                        q.extend(q2);
                        k.extend(k2);
                        v.extend(v2);
                        o.extend(o2);
                        Block::AttnSlices { layer, q, k, v, o, rest: false }
                    }
                    _ => return Err(format!("{}: a node's pieces must lie at one site (one layer's heads or one layer's MLP)", node.id)),
                });
            }
            let Some(mut block) = block else { return Err(format!("{}: a node of no pieces", node.id)) };
            for piece in &node.pieces {
                let site = (piece.layer, matches!(piece.kind.as_str(), "head" | "attn" | "q_proj" | "k_proj" | "v_proj" | "o_proj"));
                if let Some(other) = views.insert(site, piece.view.clone()).filter(|v| *v != piece.view) {
                    return Err(format!("{}: layer {}'s {} appears in views {other} and {}", node.id, site.0, if site.1 { "attention" } else { "MLP" }, piece.view));
                }
            }
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
                Block::Features { layer, features, .. } => {
                    features.sort_unstable();
                    if !features.windows(2).all(|w| w[0] < w[1]) {
                        return Err(format!("{}: a feature listed twice", node.id));
                    }
                    if featured.iter().any(|(l, f)| l == layer && features.contains(f)) {
                        return Err(format!("{}: a feature of layer {layer} is in two nodes", node.id));
                    }
                    featured.extend(features.iter().map(|&f| (*layer, f)));
                }
                Block::Slices { layer, fc, down, .. } => {
                    fc.sort_unstable();
                    down.sort_unstable();
                    for (kind, list) in [(0usize, &*fc), (1, &*down)] {
                        if !list.windows(2).all(|w| w[0] < w[1]) {
                            return Err(format!("{}: a subcomponent listed twice", node.id));
                        }
                        for &i in list {
                            if !sliced.insert((*layer, kind, i)) {
                                return Err(format!("{}: subcomponent {i} of layer {layer} is in two nodes", node.id));
                            }
                        }
                    }
                }
                Block::AttnSlices { layer, q, k, v, o, .. } => {
                    for (kind, list) in [(2usize, q), (3, k), (4, v), (5, o)] {
                        list.sort_unstable();
                        if !list.windows(2).all(|w| w[0] < w[1]) {
                            return Err(format!("{}: a subcomponent listed twice", node.id));
                        }
                        for &i in list.iter() {
                            if !sliced.insert((*layer, kind, i)) {
                                return Err(format!("{}: subcomponent {i} of layer {layer} is in two nodes", node.id));
                            }
                        }
                    }
                }
            }
            if let Some((l, _, _)) = sliced.iter().find(|(l, kind, _)| *kind < 2 && (owned.iter().any(|(m, head, _)| m == l && !head) || featured.iter().any(|(m, _)| m == l))) {
                return Err(format!("{}: layer {l}'s MLP appears in more than one view", node.id));
            }
            if let Some((l, _, _)) = sliced.iter().find(|(l, kind, _)| *kind >= 2 && owned.iter().any(|(m, head, _)| m == l && *head)) {
                return Err(format!("{}: layer {l}'s attention appears both as heads and as VPD subcomponents", node.id));
            }
            // One view per site: a layer's MLP is read through its neurons or its transcoder.
            if let Some((l, _)) = featured.iter().find(|(l, _)| owned.iter().any(|(m, head, _)| m == l && !head)) {
                return Err(format!("{}: layer {l}'s MLP appears both as neurons and as transcoder features", node.id));
            }
            if block.is_empty() {
                return Err(format!("{}: a node of no pieces", node.id));
            }
            // An attention claim is about the pattern the node's queries and keys produce: native heads
            // or one layer's attention parts with q_proj or k_proj subcomponents.
            let attends = match &block {
                Block::Heads { .. } => true,
                Block::AttnSlices { q, k, .. } => !q.is_empty() || !k.is_empty(),
                _ => false,
            };
            if claim.is_some() && !attends {
                return Err(format!("{}: an attention claim needs native heads or q_proj/k_proj subcomponents", node.id));
            }
            ids.push(node.id.clone());
            blocks.push(block);
            claims.push(claim);
        }
        let final_site = 2 * layers;
        let mut edges = Vec::new();
        let mut internal = Vec::new();
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
            // Within one VPD MLP: c_fc subcomponents to down_proj subcomponents (the hidden stream).
            if let (Writer::Unit(w), Some(r)) = (writer, reader)
                && blocks[w].site() == read_site
            {
                if !joins(&blocks[w], &blocks[r]) || route != Route::Input {
                    return Err(format!("edge {} >> {}: within one site only c_fc subcomponents feed down_proj subcomponents and q/k/v subcomponents feed o_proj subcomponents", e.from, e.to));
                }
                if !internal.contains(&(w, r)) {
                    internal.push((w, r));
                }
                continue;
            }
            if let Writer::Unit(w) = writer
                && (blocks[w].site() >= read_site || !blocks[w].writes_residual())
            {
                return Err(format!("edge {} >> {}: the writer does not write the residual stream before the reader reads", e.from, e.to));
            }
            let routes: Vec<Route> = reader.map_or(vec![Route::Input], |r| blocks[r].reads());
            if routes.is_empty() {
                return Err(format!("edge {} >> {}: the reader's subcomponents read only their site's own stream (down_proj, o_proj)", e.from, e.to));
            }
            // `a >> head` (route input) feeds all of a head's inputs.
            let expanded: Vec<Route> = if route == Route::Input && !routes.contains(&Route::Input) { routes.to_vec() } else { vec![route] };
            for route in expanded {
                if !routes.contains(&route) {
                    return Err(format!("edge {} >> {}: the reader has no {} route", e.from, e.to, e.route));
                }
                if !edges.contains(&(writer, reader, route)) {
                    edges.push((writer, reader, route));
                }
            }
        }
        let delete = match program.standin.as_deref() {
            None => weights.has_vpd(),
            Some("delete") => true,
            Some("counterfactual") => false,
            Some(other) => return Err(format!("stand-in {other}: \"delete\" (undeclared pieces contribute nothing) or \"counterfactual\" (their values on the counterfactual)")),
        };
        if let Some(b) = program.base.iter().find(|b| !ids.contains(b)) {
            return Err(format!("base node {b} is not a node"));
        }
        // A shared base's nodes are generic machinery, always on: each reads every earlier write
        // (`embed` and every residual writer before its site), every later node and the logits read
        // its write, and within its site it joins every other node either way (c_fc to down_proj,
        // q/k/v to o_proj); the logits read `embed` too, as in the model's residual stream. These
        // edges are implied by the base, not declared.
        let base: BTreeSet<usize> = ids.iter().enumerate().filter(|(_, id)| program.base.contains(id)).map(|(k, _)| k).collect();
        let declared = (edges.len(), internal.len());
        if !base.is_empty() && !edges.contains(&(Writer::Embed, None, Route::Input)) {
            edges.push((Writer::Embed, None, Route::Input));
        }
        for &b in &base {
            let site = blocks[b].site();
            let mut implied: Vec<(Writer, Option<usize>, Route)> = blocks[b].reads().into_iter().map(|route| (Writer::Embed, Some(b), route)).collect();
            if blocks[b].writes_residual() {
                implied.push((Writer::Unit(b), None, Route::Input));
            }
            for (w, block) in blocks.iter().enumerate().filter(|&(w, _)| w != b) {
                if block.site() == site {
                    for pair in [(w, b), (b, w)] {
                        if joins(&blocks[pair.0], &blocks[pair.1]) && !internal.contains(&pair) {
                            internal.push(pair);
                        }
                    }
                }
                if block.site() < site && block.writes_residual() {
                    implied.extend(blocks[b].reads().into_iter().map(|route| (Writer::Unit(w), Some(b), route)));
                }
                if block.site() > site && blocks[b].writes_residual() {
                    implied.extend(block.reads().into_iter().map(|route| (Writer::Unit(b), Some(w), route)));
                }
            }
            for e in implied {
                if !edges.contains(&e) {
                    edges.push(e);
                }
            }
        }
        // An aligned variable is read from its nodes' writes into the residual stream (an interchange
        // swaps them).
        let mut alignments = Vec::with_capacity(program.alignments.len());
        for b in &program.alignments {
            let nodes: Vec<usize> = b.nodes.iter().map(|id| ids.iter().position(|i| i == id).ok_or_else(|| format!("alignment {}: unknown node {id}", b.variable))).collect::<Result<_, _>>()?;
            // No nodes: a variable of the behavior the program does not place; its test swaps nothing
            // and costs the signal.
            if let Some(&n) = nodes.iter().find(|&&n| !blocks[n].writes_residual()) {
                return Err(format!("alignment {}: node {} writes no residual stream (an interchange swaps a node's write)", b.variable, ids[n]));
            }
            alignments.push((b.clone(), nodes));
        }
        let implied = (edges.len() - declared.0, internal.len() - declared.1);
        let mut groups = Vec::with_capacity(program.groups.len());
        for g in &program.groups {
            let nodes: Vec<usize> = g.nodes.iter().map(|id| ids.iter().position(|i| i == id).ok_or_else(|| format!("group {}: unknown node {id}", g.name))).collect::<Result<_, _>>()?;
            let parts = g.parts.iter().map(|t| group_part(weights, t)).collect::<Result<BTreeSet<_>, _>>()?;
            groups.push(GroupUse { name: g.name.clone(), parts, nodes });
        }
        let mut positions = Vec::with_capacity(program.nodes.len());
        for node in &program.nodes {
            if node.at.is_empty() {
                positions.push(None);
                continue;
            }
            let mut by_tokens = std::collections::HashMap::new();
            for seq in &node.at {
                let mut mask = vec![false; seq.tokens.len()];
                for &t in &seq.positions {
                    *mask.get_mut(t).ok_or_else(|| format!("{}: position {t} past a sequence of {} tokens", node.id, seq.tokens.len()))? = true;
                }
                by_tokens.insert(seq.tokens.clone(), mask);
            }
            positions.push(Some(Arc::new(Positions { by_tokens })));
        }
        Ok(Self { delete, ids, blocks, claims, edges, internal, alignments, positions, base, implied, groups })
    }

    /// The empty program: every piece a stand-in.
    pub fn empty() -> Self {
        Self { delete: false, ids: Vec::new(), blocks: Vec::new(), claims: Vec::new(), edges: Vec::new(), internal: Vec::new(), alignments: Vec::new(), positions: Vec::new(), base: BTreeSet::new(), implied: (0, 0), groups: Vec::new() }
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
            // A VPD-view attention: every undeclared subcomponent of its four matrices and the
            // remainders.
            let attention: Vec<[&Vec<usize>; 4]> = self.blocks.iter().filter_map(|b| match b {
                Block::AttnSlices { layer, q, k, v, o, .. } if *layer == l => Some([q, k, v, o]),
                _ => None,
            }).collect();
            if !attention.is_empty() {
                let gather = |m: usize| -> Vec<usize> {
                    let mut all: Vec<usize> = attention.iter().flat_map(|a| a[m].iter().copied()).collect();
                    all.sort_unstable();
                    all
                };
                out.push(Block::AttnSlices { layer: l, q: gather(0), k: gather(1), v: gather(2), o: gather(3), rest: true });
            } else {
                let rest: Vec<usize> = (0..layer.heads.len()).filter(|h| !heads.contains(h)).collect();
                if !rest.is_empty() {
                    out.push(Block::Heads { layer: l, heads: rest });
                }
            }
            // A transcoder-view MLP: the MLP minus the declared features (every other feature and
            // the exact error piece).
            let features: Vec<usize> = self.blocks.iter().flat_map(|b| match b {
                Block::Features { layer, features, .. } if *layer == l => features.clone(),
                _ => Vec::new(),
            }).collect();
            if !features.is_empty() {
                let mut features = features;
                features.sort_unstable();
                out.push(Block::Features { layer: l, features, rest: true });
                continue;
            }
            // A VPD-view MLP: every undeclared subcomponent of both matrices and the remainders.
            let slices: Vec<(&Vec<usize>, &Vec<usize>)> = self.blocks.iter().filter_map(|b| match b {
                Block::Slices { layer, fc, down, .. } if *layer == l => Some((fc, down)),
                _ => None,
            }).collect();
            if !slices.is_empty() {
                let mut fc: Vec<usize> = slices.iter().flat_map(|(f, _)| f.iter().copied()).collect();
                let mut down: Vec<usize> = slices.iter().flat_map(|(_, d)| d.iter().copied()).collect();
                fc.sort_unstable();
                down.sort_unstable();
                out.push(Block::Slices { layer: l, fc, down, rest: true });
                continue;
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
                // A node reads its own c_fc subcomponents; others' through declared same-site edges.
                let hidden = if edges { Incoming::Only(self.internal.iter().filter(|(_, r)| *r == n).map(|(w, _)| Writer::Unit(*w)).chain([Writer::Unit(n)]).collect()) } else { Incoming::all() };
                Unit { block: block.clone(), computes: true, at: self.positions[n].clone().map(|p| (p, false)), routes, hidden }
            })
            .collect();
        units.extend(self.complement(weights).into_iter().map(|block| Unit { block, computes: false, at: None, routes: [Incoming::all(), Incoming::all(), Incoming::all()], hidden: Incoming::all() }));
        Circuit { units, logits: routed(None, Route::Input), nodes: self.blocks.len(), delete: self.delete, places: self.positions.clone() }
    }

    /// `M` with the program's nodes writing their stand-ins (their values on the counterfactual, or
    /// zero when deleted) at their positions (everywhere for a node without positions) and every
    /// other piece computing on the prompt, every edge kept; the logits read `embed`'s stand-in when
    /// the program routes `embed` to them. The necessity experiments' `M`.
    pub fn complement_model(&self, weights: &Weights) -> Circuit {
        let mut circuit = self.model(weights);
        circuit.delete = self.delete;
        // The shared base stays: necessity takes out the program's own nodes.
        for (k, unit) in circuit.units.iter_mut().take(self.blocks.len()).enumerate() {
            unit.computes = self.base.contains(&k) || self.positions[k].is_some();
            if !self.base.contains(&k) {
                unit.at = self.positions[k].clone().map(|p| (p, true));
            }
        }
        if self.edges.iter().any(|(w, r, _)| *w == Writer::Embed && r.is_none()) {
            circuit.logits = Incoming::AllBut([Writer::Embed].into());
        }
        circuit
    }

    /// `M` as a circuit with the program's nodes as its first units: every piece computing at every
    /// position, every edge kept (the nodes' positions kept for swaps, `Circuit::places`).
    pub fn model(&self, weights: &Weights) -> Circuit {
        let mut circuit = self.program(weights, false);
        circuit.delete = false;
        for unit in &mut circuit.units {
            unit.computes = true;
            unit.at = None;
        }
        circuit
    }

    /// What a reader takes in from the program's structure (design_v2 section 2; format v3's pricing
    /// parity): each part a node lists costs `log2 V` bits to name (`V` = [`Weights::vocabulary`]), a
    /// VPD remainder its matrix's rank in names (it holds that many directions), a named group's use
    /// one name; the rest is code, `token` bits per code token (the program's `log2` token types),
    /// counted from the graph whatever the syntax wrote it: [`NODE_TOKENS`] per node,
    /// [`LABEL_TOKENS`] per label (an alignment with pairs, or a claim), [`EDGE_TOKENS`] per edge and
    /// [`ROUTE_TOKENS`] more for a query, key or value route. So the same subcomponents with the same
    /// data flow cost the same in every format (mech's code tokens leave these statements out). The
    /// edges a base implies cost nothing. Returns the part names, the bits of every part, node,
    /// label and edge outside `base`, and the bits of `base`'s nodes and the declared edges that
    /// touch them.
    pub fn structure(&self, weights: &Weights, base: &BTreeSet<usize>, token: f64) -> (usize, f64, f64) {
        let name = (weights.vocabulary().max(2) as f64).log2();
        // Names in one site's list (matrix `m` of `layer`, at node `k`): one per subcomponent, the
        // rank for the remainder (index `count`), none for a part a named group the node holds
        // names.
        let names = |k: usize, (layer, m): (usize, usize), list: &[usize], (u, v): (&Array2<f64>, &Array2<f64>)| -> usize {
            list.iter().filter(|&&i| !self.groups.iter().any(|g| g.nodes.contains(&k) && g.parts.contains(&(layer, m, i)))).map(|&i| if i < u.nrows() { 1 } else { u.ncols().min(v.nrows()) }).sum()
        };
        let mut parts = 0;
        let (mut bits, mut base_bits) = (0.0, 0.0);
        for (k, block) in self.blocks.iter().enumerate() {
            let count = match block {
                Block::Heads { heads, .. } => heads.len(),
                Block::Neurons { neurons, .. } => neurons.len(),
                Block::Features { features, .. } => features.len(),
                Block::Slices { layer, fc, down, .. } => weights.vpd.get(layer).map_or(fc.len() + down.len(), |v| names(k, (*layer, 4), fc, (&v.fc_u, &v.fc_v)) + names(k, (*layer, 5), down, (&v.down_u, &v.down_v))),
                Block::AttnSlices { layer, q, k: key, v, o, .. } => match weights.vpd_attention.get(layer) {
                    Some(a) => names(k, (*layer, 0), q, (&a.q.0, &a.q.1)) + names(k, (*layer, 1), key, (&a.k.0, &a.k.1)) + names(k, (*layer, 2), v, (&a.v.0, &a.v.1)) + names(k, (*layer, 3), o, (&a.o.0, &a.o.1)),
                    None => q.len() + key.len() + v.len() + o.len(),
                },
            };
            parts += count;
            let claim = if self.claims.get(k).is_some_and(Option::is_some) { LABEL_TOKENS } else { 0 };
            *(if base.contains(&k) { &mut base_bits } else { &mut bits }) += count as f64 * name + (NODE_TOKENS + claim) as f64 * token;
        }
        // Each named group used: one name; each label (an alignment tested by pairs; the answer's,
        // aligned by construction, has none): its code.
        parts += self.groups.len();
        let labels = self.alignments.iter().filter(|(a, nodes)| !a.pairs.is_empty() && !nodes.is_empty()).count();
        bits += self.groups.len() as f64 * name + (LABEL_TOKENS * labels) as f64 * token;
        let touches = |w: &Writer, r: Option<usize>| matches!(w, Writer::Unit(u) if base.contains(u)) || r.is_some_and(|r| base.contains(&r));
        for (w, r, route) in &self.edges[..self.edges.len() - self.implied.0] {
            let tokens = EDGE_TOKENS + if *route == Route::Input { 0 } else { ROUTE_TOKENS };
            *(if touches(w, *r) { &mut base_bits } else { &mut bits }) += tokens as f64 * token;
        }
        for &(w, r) in &self.internal[..self.internal.len() - self.implied.1] {
            *(if base.contains(&w) || base.contains(&r) { &mut base_bits } else { &mut bits }) += EDGE_TOKENS as f64 * token;
        }
        (parts, bits, base_bits)
    }

    /// The definitions of the named groups the program uses: their parts' names (one per
    /// subcomponent, the rank for a remainder), at `log2 V` bits each as in [`Graph::structure`].
    pub fn group_bits(&self, weights: &Weights) -> f64 {
        let name = (weights.vocabulary().max(2) as f64).log2();
        let rank = |&(layer, m, i): &GroupPart| -> usize {
            let factors = if m < 4 { weights.vpd_attention.get(&layer).map(|a| [&a.q, &a.k, &a.v, &a.o][m]).map(|(u, v)| (u, v)) } else { weights.vpd.get(&layer).map(|v| if m == 4 { (&v.fc_u, &v.fc_v) } else { (&v.down_u, &v.down_v) }) };
            factors.map_or(1, |(u, v)| if i < u.nrows() { 1 } else { u.ncols().min(v.nrows()) })
        };
        self.groups.iter().map(|g| g.parts.iter().map(rank).sum::<usize>() as f64 * name).sum()
    }

    /// Opaque numbers: every weight a declared node reads (a head's query, key, value and output
    /// maps and head norm gains, keys and values shared by query heads counted once; a neuron's gate
    /// and up rows with their biases and its down column; a transcoder feature's encoder row, bias
    /// and decoder row) and, under average stand-ins, every stand-in average (`embed`'s, one per
    /// undeclared head, one per layer with an undeclared neuron), each of width `d`.
    pub fn opaque_numbers(&self, weights: &Weights) -> usize {
        let d = weights.width();
        let mut count = 0;
        for block in &self.blocks {
            match block {
                Block::Heads { layer, heads } => {
                    let all = &weights.layers[*layer].heads;
                    let mut keys: Vec<usize> = Vec::new();
                    for &h in heads {
                        let w = &all[h];
                        count += w.query.len() + w.output.len() + w.query_norm.as_ref().map_or(0, |(g, _)| g.len());
                        if !keys.iter().any(|&k| (Arc::ptr_eq(&all[k].key, &w.key) || all[k].key == w.key) && (Arc::ptr_eq(&all[k].value, &w.value) || all[k].value == w.value)) {
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
                // A feature's encoder row, bias and decoder row.
                Block::Features { features, .. } => count += features.len() * (2 * d + 1),
                // A subcomponent's two vectors: width and hidden.
                // A subcomponent's two vectors; a remainder (index = the subcomponent count) its matrix.
                Block::Slices { layer, fc, down, .. } => {
                    let hidden = weights.neurons(*layer);
                    let counts = weights.vpd.get(layer).map_or((0, 0), |v| (v.fc_u.nrows(), v.down_u.nrows()));
                    for (list, n) in [(fc, counts.0), (down, counts.1)] {
                        count += list.iter().map(|&i| if i == n { hidden * d } else { d + hidden }).sum::<usize>();
                    }
                }
                Block::AttnSlices { layer, q, k, v, o, .. } => {
                    let width: usize = weights.layers[*layer].heads.iter().map(|h| h.query.nrows()).sum();
                    let counts = weights.vpd_attention.get(layer).map_or([0; 4], |a| [a.q.0.nrows(), a.k.0.nrows(), a.v.0.nrows(), a.o.0.nrows()]);
                    for (list, n) in [q, k, v, o].into_iter().zip(counts) {
                        count += list.iter().map(|&i| if i == n { width * d } else { d + width }).sum::<usize>();
                    }
                }
            }
        }
        count
    }
}

// ------------------------------------------------------------------------------ stand-ins


// ------------------------------------------------------------------------------ execution

/// Sequences end to end, each its own causal span (start, length), and, for counterfactual
/// stand-ins, the same model's run on each sequence's counterfactual ([`Reference`]).
#[derive(Clone)]
pub struct Batch {
    pub tokens: Vec<u32>,
    pub spans: Vec<(usize, usize)>,
    pub reference: Option<std::sync::Arc<Reference>>,
    /// Per sequence its attention blocks ([`Prompt::attention_block`]), positions within it.
    pub blocks: Vec<Vec<[usize; 4]>>,
    /// Per sequence, a batch made from a behavior's items: its partner (a prompt's counterfactual,
    /// a counterfactual's prompt; empty where it has none), whose run its stand-ins read. Empty:
    /// partners by the sequences' tokens (`Checker::partners`). Two items with one prompt text and
    /// different counterfactuals keep their own.
    pub partners: Vec<Vec<u32>>,
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
        let blocks = vec![Vec::new(); spans.len()];
        Ok(Self { tokens, spans, reference: None, blocks, partners: Vec::new() })
    }

    /// The batch with each sequence's partner ([`Batch::partners`]).
    pub fn with_partners(mut self, partners: Vec<Vec<u32>>) -> Self {
        self.partners = partners;
        self
    }

    /// The batch's sequences.
    pub fn sequences(&self) -> Vec<Vec<u32>> {
        self.spans.iter().map(|&(start, length)| self.tokens[start..start + length].to_vec()).collect()
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

/// `M`'s run on the counterfactual sequences (the stand-ins' source), per piece so that any unit
/// partition reads it: the embeddings, per layer each head's attention read `z_h` (rows × head
/// width), its MLP's activations (rows × neurons) and its MLP's write (rows × width). A unit's
/// stand-in write is assembled with the current weights, the same (possibly edited) weights the
/// run used.
#[derive(Clone, Debug)]
pub struct Reference {
    /// The run's identity (`next_reference_id`; a clone keeps it): the device keeps uploaded copies
    /// of its arrays under it.
    pub id: u64,
    pub embed: Array2<f64>,
    pub reads: Vec<Vec<Array2<f64>>>,
    pub active: Vec<Array2<f64>>,
    pub mlp: Vec<Array2<f64>>,
    /// Per layer its MLP's normed input `x̂` (rows × width): transcoder features read it.
    pub inputs: Vec<Array2<f64>>,
    /// Per layer its attention's normed input (rows × width): VPD q, k, v subcomponents read it.
    pub attention_inputs: Vec<Array2<f64>>,
    /// Every array is zero ([`Reference::zeros`]): every stand-in is zero, without computing it.
    pub zero: bool,
}

/// A fresh [`Reference::id`].
pub(crate) fn next_reference_id() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl Reference {
    /// A run of `rows` tokens where every piece is deleted: every array zero, so every stand-in
    /// write is zero (deletion semantics, [`Graph::delete`]).
    pub fn zeros(weights: &Weights, rows: usize) -> Self {
        let d = weights.width();
        let per_layer = || vec![Array2::zeros((rows, d)); weights.layers.len()];
        Self {
            id: next_reference_id(),
            embed: Array2::zeros((rows, d)),
            reads: weights.layers.iter().map(|l| l.heads.iter().map(|h| Array2::zeros((rows, h.query.nrows()))).collect()).collect(),
            active: (0..weights.layers.len()).map(|l| Array2::zeros((rows, weights.neurons(l)))).collect(),
            mlp: per_layer(),
            inputs: per_layer(),
            attention_inputs: per_layer(),
            zero: true,
        }
    }

    /// The run at rows `rows` alone (arrays it did not record stay empty).
    fn select_rows(&self, rows: &[usize]) -> Self {
        let pick = |a: &Array2<f64>| if a.nrows() == 0 { a.clone() } else { a.select(Axis(0), rows) };
        Self {
            id: next_reference_id(),
            embed: pick(&self.embed),
            reads: self.reads.iter().map(|l| l.iter().map(pick).collect()).collect(),
            active: self.active.iter().map(pick).collect(),
            mlp: self.mlp.iter().map(pick).collect(),
            inputs: self.inputs.iter().map(pick).collect(),
            attention_inputs: self.attention_inputs.iter().map(pick).collect(),
            zero: self.zero,
        }
    }

    /// Block `block`'s write in the run (rows × width).
    fn write(&self, weights: &Weights, block: &Block) -> Result<Array2<f64>, String> {
        let rows = self.embed.nrows();
        match block {
            Block::Heads { layer, heads } => {
                let mut out = Array2::<f64>::zeros((rows, weights.width()));
                for &h in heads {
                    let z = self.reads.get(*layer).and_then(|r| r.get(h)).ok_or("a head the reference did not record")?;
                    out += &par_dot(z, wide(weights.layers[*layer].heads[h].output.view()).t());
                }
                Ok(out)
            }
            Block::Neurons { layer, neurons } => {
                let mlp = weights.layers[*layer].mlp.as_ref().ok_or("a neuron block without an MLP")?;
                let active = self.active.get(*layer).ok_or("an MLP the reference did not record")?;
                let n = mlp.gate.nrows();
                // The few neurons outside a large block are subtracted from the MLP's whole write.
                if 2 * neurons.len() > n {
                    let inside: BTreeSet<usize> = neurons.iter().copied().collect();
                    let rest: Vec<usize> = (0..n).filter(|i| !inside.contains(i)).collect();
                    Ok(&self.mlp[*layer] - &active.select(Axis(1), &rest).dot(&wide(mlp.out.select(Axis(1), &rest).view()).t()))
                } else {
                    Ok(active.select(Axis(1), neurons).dot(&wide(mlp.out.select(Axis(1), neurons).view()).t()))
                }
            }
            Block::Slices { layer, down, rest, .. } => {
                let mlp = weights.layers[*layer].mlp.as_ref().ok_or("a VPD view of a layer without an MLP")?;
                let vpd = weights.vpd.get(layer).ok_or_else(|| format!("layer {layer} has no VPD view"))?;
                Ok(vpd.down(mlp, down, *rest, self.active.get(*layer).ok_or("an MLP the reference did not record")?))
            }
            Block::AttnSlices { layer, o, rest, .. } => {
                let a = weights.vpd_attention.get(layer).ok_or_else(|| format!("layer {layer}'s attention has no VPD view"))?;
                let reads = self.reads.get(*layer).ok_or("an attention the reference did not record")?;
                let z = ndarray::concatenate(Axis(1), &reads.iter().map(|r| r.view()).collect::<Vec<_>>()).map_err(|e| e.to_string())?;
                Ok(sliced(&a.o, &attention_maps(&weights.layers[*layer])[3], o, *rest, &z))
            }
            Block::Features { layer, features, rest } => {
                let t = weights.transcoders.get(layer).ok_or_else(|| format!("layer {layer} has no transcoder"))?;
                let named = t.write(features, self.inputs.get(*layer).ok_or("an MLP the reference did not record")?)?;
                Ok(if *rest { &self.mlp[*layer] - &named } else { named })
            }
        }
    }
}

/// `M`'s run on `batch` with `weights` recorded per piece ([`Reference`]).
pub fn reference(weights: &Weights, batch: &Batch) -> Result<Reference, String> {
    let circuit = Graph::empty().model(weights);
    let mut plain = batch.clone();
    plain.reference = None;
    run(weights, &circuit, &plain, &[], &BTreeMap::new(), &Interventions::default(), true)?.captured().ok_or_else(|| "no captured run".to_string())
}

/// `M`'s run on `batch` recorded per piece ([`Reference`]) under the site operations `draw`
/// that act on a run without a donor (a swap or a cut reads the run's own donor, the
/// counterfactual, which on the counterfactual's own run changes nothing): the counterfactual run
/// of a site experiment, the same experiment applied to it.
pub fn reference_under(weights: &Weights, batch: &Batch, draw: &SiteDraw, units: &SiteUnits) -> Result<Reference, String> {
    let own = SiteDraw { ops: draw.ops.iter().filter(|o| !matches!(o.operation, Operation::Swap | Operation::Cut { .. })).copied().collect(), ..draw.clone() };
    let heads: BTreeSet<(usize, usize)> = own.ops.iter().filter_map(|o| if let SharedSite::Head(h) = o.site { head_of(weights, h).ok() } else { None }).collect();
    let circuit = Graph::empty().model(weights).split_heads(&heads);
    let mut plain = batch.clone();
    plain.reference = None;
    let ops = Interventions::resolve(&own, &circuit, weights, &plain, units, None, None)?;
    run(weights, &circuit, &plain, &[], &BTreeMap::new(), &ops, true)?.captured().ok_or_else(|| "no captured run".to_string())
}

/// One run: the logits' log-probabilities at the scored rows (rows × vocabulary) and every
/// computing unit's actual write (rows × width).
pub struct Execution {
    pub log_probabilities: Array2<f64>,
    pub writes: Vec<Option<Array2<f64>>>,
    /// Per (unit, route slot) its normed input, at the sites `Interventions::record` lists.
    pub normed: BTreeMap<(usize, usize), Array2<f64>>,
    /// Per VPD-view attention unit its heads' reads `z` (rows × heads' value widths), at the sites
    /// `Interventions::record_reads` lists.
    pub reads: BTreeMap<usize, Array2<f64>>,
    captured: Option<Reference>,
}

impl Execution {
    /// A run's outcome from the device path (`graph_device`).
    pub(crate) fn of(log_probabilities: Array2<f64>, writes: Vec<Option<Array2<f64>>>, captured: Option<Reference>, normed: BTreeMap<(usize, usize), Array2<f64>>) -> Self {
        Self { log_probabilities, writes, normed, reads: BTreeMap::new(), captured }
    }

    /// The run's capture ([`Reference`]), when it made one.
    pub(crate) fn captured(self) -> Option<Reference> {
        self.captured
    }
}

fn project(x: &Array2<f64>, map: &Stored, norm: Option<&(Array1<f64>, f64)>) -> Array2<f64> {
    let mut out = par_dot(x, wide(map.view()).t());
    if let Some((gain, epsilon)) = norm {
        for mut row in out.outer_iter_mut() {
            let r = rms_scale(row.view(), *epsilon);
            row.zip_mut_with(gain, |v, g| *v *= r * g);
        }
    }
    out
}

/// Heads `heads` of `layer` on their query, key and value inputs (residual streams), `normed`
/// applied to each normed input (route slot, value); with `capture`, each head's read as well.
fn heads_write(layer: &LayerWeights, heads: &[usize], inputs: [&Array2<f64>; 3], spans: &[(usize, usize)], blocks: &[Vec<[usize; 4]>], capture: bool, normed: &mut dyn FnMut(usize, &mut Array2<f64>)) -> (Array2<f64>, Vec<Array2<f64>>) {
    let norm = &layer.attention;
    let (mut q_hat, mut k_hat, mut v_hat) = (norm.apply(inputs[0]), norm.apply(inputs[1]), norm.apply(inputs[2]));
    normed(0, &mut q_hat);
    normed(1, &mut k_hat);
    normed(2, &mut v_hat);
    let (rows, d) = v_hat.dim();
    let mut out = Array2::<f64>::zeros((rows, d));
    let mut reads = Vec::new();
    for &h in heads {
        let w = &layer.heads[h];
        let (q, k, v) = (project(&q_hat, &w.query, w.query_norm.as_ref()), project(&k_hat, &w.key, w.key_norm.as_ref()), par_dot(&v_hat, wide(w.value.view()).t()));
        let mut z = Array2::<f64>::zeros(v.dim());
        for (n, &(start, length)) in spans.iter().enumerate() {
            let positions: Vec<u32> = (0..length as u32).collect();
            let span = start..start + length;
            let (qs, ks) = (q.slice(s![span.clone(), ..]).to_owned(), k.slice(s![span.clone(), ..]).to_owned());
            let (qs, ks) = (rotate(&qs, w.rotary, &positions, false), rotate(&ks, w.rotary, &positions, false));
            let mut a = probabilities(qs.view(), ks.view(), &positions, 0, w.scale, w.causal);
            if let Some(b) = blocks.get(n).filter(|b| !b.is_empty()) {
                block_attention(&mut a, b);
            }
            z.slice_mut(s![span.clone(), ..]).assign(&a.dot(&v.slice(s![span, ..])));
        }
        out += &par_dot(&z, wide(w.output.view()).t());
        if capture {
            reads.push(z);
        }
    }
    (out, reads)
}

/// Attention weights (queries × keys of one sequence) with the blocked query-key ranges removed and
/// each query's remaining weights renormalized: the softmax with those scores at minus infinity. A
/// query with nothing left attends to nothing.
fn block_attention(a: &mut Array2<f64>, blocks: &[[usize; 4]]) {
    let (queries, keys) = a.dim();
    for &[q0, q1, k0, k1] in blocks {
        for q in q0..q1.min(queries) {
            for k in k0..k1.min(keys) {
                a[[q, k]] = 0.0;
            }
        }
    }
    for mut row in a.outer_iter_mut() {
        let total: f64 = row.sum();
        if total > 0.0 {
            row /= total;
        }
    }
}

/// Neurons `neurons` of an MLP on the normed stream `x_hat` (rows × width).
fn neurons_write(mlp: &MlpWeights, neurons: &[usize], x_hat: &Array2<f64>) -> Array2<f64> {
    par_dot(&neurons_active(mlp, neurons, x_hat), wide(mlp.out.select(Axis(1), neurons).view()).t())
}

/// The activations of neurons `neurons` (rows × neurons).
fn neurons_active(mlp: &MlpWeights, neurons: &[usize], x_hat: &Array2<f64>) -> Array2<f64> {
    let gate = wide(mlp.gate.select(Axis(0), neurons).view());
    let mut h = par_dot(x_hat, gate.t());
    let bias = mlp.bias.select(Axis(0), neurons);
    h += &bias.view().insert_axis(Axis(0));
    h.mapv_inplace(|g| mlp.law.apply(g));
    if let Some(up) = &mlp.up {
        let mut u = par_dot(x_hat, wide(up.select(Axis(0), neurons).view()).t());
        u += &mlp.up_bias.select(Axis(0), neurons).view().insert_axis(Axis(0));
        h *= &u;
    }
    h
}

/// Transcoder features `features` of `layer`'s MLP on its normed inputs `x_hat` (rows × width),
/// or with `rest` the MLP minus them (every other feature and the transcoder's exact error).
fn features_write(weights: &Weights, layer: usize, features: &[usize], rest: bool, x_hat: &Array2<f64>) -> Result<Array2<f64>, String> {
    let t = weights.transcoders.get(&layer).ok_or_else(|| format!("layer {layer} has no transcoder"))?;
    let named = t.write(features, x_hat)?;
    if !rest {
        return Ok(named);
    }
    let mlp = weights.layers[layer].mlp.as_ref().ok_or("a transcoder on a layer without an MLP")?;
    let all: Vec<usize> = (0..mlp.gate.nrows()).collect();
    Ok(neurons_write(mlp, &all, x_hat) - named)
}


/// Executes `circuit` on `batch`: per unit in site order its route inputs (the stand-in stream
/// plus the declared writers' actual minus stand-in writes, or the actual stream minus the cut
/// writers'), its write (actual, `swaps`' value, or its stand-in), and the logits at `scored` rows.
pub fn execute(weights: &Weights, circuit: &Circuit, batch: &Batch, scored: &[usize], swaps: &BTreeMap<usize, Array2<f64>>) -> Result<Execution, String> {
    execute_with(weights, circuit, batch, scored, swaps, &Interventions::default())
}

/// The most rows a stacked run holds ([`execute_stacked`]): its hidden activations take rows × the
/// MLP's width, and vpd4l's behaviors hold about a thousand rows (at 2^14 rows a score of chunk
/// programs passed the 12 GiB lease score.py gives a vpd4l checker, at 2^13 it peaked at 15 GB).
const STACKED_ROWS: usize = 1 << 12;

/// Whether `circuit` computes as the model's stream with every part it leaves out deleted or at its
/// counterfactual value, so that it can run stacked with others ([`execute_stacked`]): its
/// computing units are VPD units without `rest`, each reads (on every route its parts read) the
/// embedding and every earlier computing unit's write, each MLP or attention unit with `down_proj`
/// or `o_proj` subcomponents reads the hidden writes of every computing unit of its site, and the
/// logits read every write; with counterfactual stand-ins every layer has both VPD views (the
/// stand-ins are then the counterfactual writes of the parts it does not name) and no transcoder
/// features stand in. Under these routes a unit's input is the stream itself either way.
pub(crate) fn stacks(weights: &Weights, circuit: &Circuit) -> bool {
    if circuit.units.iter().any(|u| u.at.is_some()) {
        return false;
    }
    if !circuit.delete && ((0..weights.layers.len()).any(|l| !weights.vpd.contains_key(&l) || !weights.vpd_attention.contains_key(&l)) || circuit.units.iter().any(|u| matches!(u.block, Block::Features { .. }))) {
        return false;
    }
    let computing: Vec<usize> = (0..circuit.units.len()).filter(|&u| circuit.units[u].computes).collect();
    let all_read = |incoming: &Incoming, before: usize| incoming.reads(&Writer::Embed) && computing.iter().filter(|&&w| circuit.units[w].block.site() < before).all(|&w| incoming.reads(&Writer::Unit(w)));
    computing.iter().all(|&u| {
        let unit = &circuit.units[u];
        let site = unit.block.site();
        let (vpd, writes) = match &unit.block {
            Block::Slices { down, rest: false, .. } => (true, !down.is_empty()),
            Block::AttnSlices { o, rest: false, .. } => (true, !o.is_empty()),
            _ => (false, false),
        };
        vpd && unit.block.reads().iter().all(|r| all_read(&unit.routes[r.slot()], site))
            && (!writes || computing.iter().filter(|&&w| circuit.units[w].block.site() == site).all(|&w| unit.hidden.reads(&Writer::Unit(w))))
    }) && all_read(&circuit.logits, usize::MAX)
}

/// The runs of `circuits` (each one that [`stacks`], all deleting or all with counterfactual
/// stand-ins) on `batch` (with its counterfactual run attached for the latter, as
/// `Checker::referenced` attaches it) at rows `scored`, as one batch of copies
/// ([`crate::graph_device::execute_copies`]): per site one unit naming the union of the circuits'
/// subcomponents (every site with counterfactual stand-ins), each copy weighing them by how many of
/// its units name them. Each copy's log-probabilities equal its own run's ([`execute`]) up to
/// summation order. `None` when there is no device, a circuit does not stack, the stand-ins differ,
/// the batch has attention blocks or the copies would pass [`STACKED_ROWS`].
pub(crate) fn execute_stacked(weights: &Weights, circuits: &[&Circuit], batch: &Batch, scored: &[usize]) -> Option<Result<Vec<Array2<f64>>, String>> {
    execute_stacked_with(weights, circuits, (batch, scored), &mut |c, job, copies| crate::graph_device::execute_copies(weights, c, job, copies))
}

/// [`execute_stacked`] with the stacked runs made by `run`.
pub(crate) fn execute_stacked_with(weights: &Weights, circuits: &[&Circuit], (batch, scored): (&Batch, &[usize]), run: &mut CopiesRun) -> Option<Result<Vec<Array2<f64>>, String>> {
    run_stack(stack(weights, circuits, batch, scored)?, batch, run)
}

/// The complement models (`Graph::complement_model`: every part computing but the program's own,
/// which carry their counterfactual values) of programs with counterfactual stand-ins whose
/// circuits stack (`circuits`, none with a shared base), run as one batch of copies like
/// [`execute_stacked_with`]: the merged circuit's computing unit at each site is the matrix less each
/// copy's named parts (its `rest`, by the copy's counts) and its stand-in unit the named parts at
/// their counterfactual values; `embed_routed[j]`: program `j` routes the embedding to the logits,
/// so its complement's logits read the embedding's stand-in. Each copy equals its complement
/// model's own run up to summation order (a layer a copy names nothing of computes through the VPD
/// path, equal to the native one up to rounding).
pub(crate) fn execute_complements_with(weights: &Weights, (circuits, embed_routed): (&[&Circuit], &[bool]), (batch, scored): (&Batch, &[usize]), run: &mut CopiesRun) -> Option<Result<Vec<Array2<f64>>, String>> {
    if circuits.iter().any(|c| c.delete) || embed_routed.len() != circuits.len() {
        return None;
    }
    let mut stacked = stack_sites(weights, circuits, batch, scored, (true, true))?;
    stacked.copies.embed_out = embed_routed.to_vec();
    run_stack(stacked, batch, run)
}

/// A stacked run of `stacked` on `batch` (its shared prefix made once), each copy's log-probabilities.
fn run_stack(mut stacked: Stacked, batch: &Batch, run: &mut CopiesRun) -> Option<Result<Vec<Array2<f64>>, String>> {
    // Sites before the first where the copies' parts differ compute alike on every copy: one copy's
    // run makes that prefix's stream, and the copies resume from it (one-part neighbours of an
    // answer share every site before the edited one).
    if let Some(site) = stacked.shared_prefix() {
        let one = crate::graph_device::Copies { count: 1, rows: stacked.copies.rows, masks: stacked.copies.masks.iter().map(|(&k, m)| (k, crate::graph_device::CopyMask { counts: m.counts.slice(ndarray::s![0..1, ..]).to_owned(), remainder: m.remainder.iter().take(1).copied().collect() })).collect(), standin: stacked.copies.standin.clone(), resume: None, halt: Some(site), embed_out: Vec::new() };
        let (swaps, ops) = (BTreeMap::new(), Interventions::default());
        let job = crate::graph_device::Run { tokens: &batch.tokens, spans: &batch.spans, scored: &[], swaps: &swaps, capture: false, reference: None, ops: &ops };
        match run(&stacked.merged, &job, &one)? {
            Ok(prefix) => stacked.copies.resume = Some((site, crate::graph_device::prefix_stream(prefix))),
            Err(e) => return Some(Err(e)),
        }
    }
    let count = stacked.copies.count;
    run(&stacked.merged, &stacked.job(), &stacked.copies).map(|out| out.and_then(|execution| crate::graph_device::per_copy(execution, count)))
}

/// A stacked run's merged circuit, copies and rows ([`stack`]).
pub(crate) struct Stacked {
    pub merged: Circuit,
    pub copies: crate::graph_device::Copies,
    tokens: Vec<u32>,
    spans: Vec<(usize, usize)>,
    scored: Vec<usize>,
    /// No swaps and no site operations.
    swaps: BTreeMap<usize, Array2<f64>>,
    ops: Interventions,
}

impl Stacked {
    /// The device's job: the batch's rows once per copy, each copy's scored rows in turn.
    pub(crate) fn job(&self) -> crate::graph_device::Run<'_> {
        crate::graph_device::Run { tokens: &self.tokens, spans: &self.spans, scored: &self.scored, swaps: &self.swaps, capture: false, reference: None, ops: &self.ops }
    }

    /// The first site where the copies' parts differ, when some site before it has units (the
    /// copies compute alike up to it); `None` for one copy or copies that differ at the first site.
    fn shared_prefix(&self) -> Option<usize> {
        if self.copies.count < 2 {
            return None;
        }
        let alike = |u: usize| self.copies.masks.iter().filter(|((v, _), _)| *v == u).all(|(_, m)| m.counts.outer_iter().all(|row| row == m.counts.row(0)) && m.remainder.iter().all(|&r| r == m.remainder[0]));
        let mut sites: Vec<usize> = self.merged.units.iter().map(|u| u.block.site()).collect();
        sites.dedup();
        let differs = (0..self.merged.units.len()).find(|&u| !alike(u)).map(|u| self.merged.units[u].block.site())?;
        (sites.first().is_some_and(|&first| first < differs)).then_some(differs)
    }

    /// The batch of every copy's rows (no counterfactual run attached, no attention blocks): where
    /// site operations land ([`Interventions::resolve`]).
    fn batch(&self) -> Batch {
        Batch { tokens: self.tokens.clone(), spans: self.spans.clone(), reference: None, blocks: vec![Vec::new(); self.spans.len()], partners: Vec::new() }
    }
}

/// [`run_sites`] for circuits that stack, as one batch of copies: the base and the donor stacked
/// alike (each with its counterfactual run, under counterfactual stand-ins), the donor run made
/// stacked with the recording `run_sites` makes, the operations resolved on the merged circuit over
/// every copy's rows (a scaled counterfactual run from one copy's rows), then the stacked run.
/// Each copy's log-probabilities equal its own `run_sites` up to summation order: the merged
/// circuit has a computing unit and (with counterfactual stand-ins) a stand-in unit at every site,
/// so every operation finds the writers, inputs and reads it finds in a program's own circuit.
pub(crate) fn stacked_sites(weights: &Weights, circuits: &[&Circuit], (base, rows): (&Batch, &[usize]), donor: Option<&Batch>, draw: &SiteDraw, units: &SiteUnits) -> Option<Result<Vec<Array2<f64>>, String>> {
    stacked_sites_with(weights, circuits, (base, rows), donor, (draw, units), &mut |c, job, copies| crate::graph_device::execute_copies(weights, c, job, copies))
}

/// A stacked run's execution on the device (`graph_device::execute_copies`; the tests run one on
/// the host backend).
pub(crate) type CopiesRun<'a> = dyn FnMut(&Circuit, &crate::graph_device::Run, &crate::graph_device::Copies) -> Option<Result<Execution, String>> + 'a;

/// [`stacked_sites`] with the stacked runs made by `run`.
pub(crate) fn stacked_sites_with(weights: &Weights, circuits: &[&Circuit], (base, rows): (&Batch, &[usize]), donor: Option<&Batch>, (draw, units): (&SiteDraw, &SiteUnits), run: &mut CopiesRun) -> Option<Result<Vec<Array2<f64>>, String>> {
    let mut stacked = stack_sites(weights, circuits, base, rows, (true, false))?;
    let merged = stacked.merged.split_heads(&BTreeSet::new());
    let donor = match donor {
        Some(b) if Interventions::needs_donor(draw) => {
            let d = stack_sites(weights, circuits, b, &[], (true, false))?;
            let record_reads = draw.ops.iter().filter(|o| o.operation == Operation::Swap).filter_map(|o| if let SharedSite::Head(h) = o.site { head_of(weights, h).ok().map(|(l, _)| 2 * l) } else { None }).collect();
            let recording = Interventions { record_reads, ..Interventions::recording(Interventions::donor_record(draw, weights)) };
            let job = crate::graph_device::Run { ops: &recording, ..d.job() };
            match run(&d.merged, &job, &d.copies)? {
                Ok(run) => Some((run, d.batch())),
                Err(e) => return Some(Err(e)),
            }
        }
        _ => None,
    };
    let (donor_run, donor_batch) = donor.as_ref().map(|(r, b)| (r, b)).unzip();
    let ops = match Interventions::resolve(draw, &merged, weights, &stacked.batch(), units, donor_run, donor_batch) {
        Ok(ops) => ops,
        Err(e) => return Some(Err(e)),
    };
    // The counterfactual run under the operations on the heads' reads, from one copy's rows.
    if let Some(r) = base.reference.as_deref().filter(|r| !r.zero) {
        let one = match Interventions::resolve(draw, &merged, weights, base, units, donor_run, donor_batch) {
            Ok(ops) => ops,
            Err(e) => return Some(Err(e)),
        };
        if let Some(scaled) = one.scaled_reference(r) {
            stacked.copies.standin = Some(std::sync::Arc::new(scaled));
        }
    }
    let job = crate::graph_device::Run { ops: &ops, ..stacked.job() };
    let count = stacked.copies.count;
    run(&merged, &job, &stacked.copies).map(|out| out.and_then(|execution| crate::graph_device::per_copy(execution, count)))
}

/// [`execute_stacked`]'s merged circuit and copies, `None` where it returns `None` before the device.
pub(crate) fn stack(weights: &Weights, circuits: &[&Circuit], batch: &Batch, scored: &[usize]) -> Option<Stacked> {
    stack_sites(weights, circuits, batch, scored, (false, false))
}

/// [`stack`], with a computing unit at every site also for deleting copies when `every_site`
/// (where site operations look for a site's units, as on a program's own circuit, whose undeclared
/// pieces are units there); with `complement`, the copies' complement models
/// ([`execute_complements_with`]): each site's computing unit the `rest` of the named parts, its
/// stand-in unit the named parts.
fn stack_sites(weights: &Weights, circuits: &[&Circuit], batch: &Batch, scored: &[usize], (every_site, complement): (bool, bool)) -> Option<Stacked> {
    let rows = batch.tokens.len();
    if circuits.is_empty() || circuits.len() * rows > STACKED_ROWS || !batch.blocks.iter().all(Vec::is_empty) || !circuits.iter().all(|c| stacks(weights, c)) {
        return None;
    }
    let delete = circuits[0].delete;
    if circuits.iter().any(|c| c.delete != delete) {
        return None;
    }
    let standin = match &batch.reference {
        Some(r) if !delete && !r.zero && r.embed.nrows() == rows => Some(r.clone()),
        _ if delete => None,
        _ => return None,
    };
    // Per site and slot, each copy's subcomponents (with repeats) and remainder.
    let mut lists: BTreeMap<(usize, usize), Vec<(Vec<usize>, bool)>> = BTreeMap::new();
    let slots = |block: &Block| -> Vec<(usize, Vec<usize>)> {
        match block {
            Block::Slices { fc, down, .. } => vec![(0, fc.clone()), (1, down.clone())],
            Block::AttnSlices { q, k, v, o, .. } => vec![(0, q.clone()), (1, k.clone()), (2, v.clone()), (3, o.clone())],
            _ => Vec::new(),
        }
    };
    // Deleting copies need a unit where some copy names parts; counterfactual ones at every site,
    // each adding the counterfactual write of the parts its copy leaves out.
    let mut layers: BTreeMap<usize, usize> = if delete && !every_site { BTreeMap::new() } else { (0..2 * weights.layers.len()).map(|site| (site, site / 2)).collect() };
    for (j, circuit) in circuits.iter().enumerate() {
        for unit in circuit.units.iter().filter(|u| u.computes) {
            let site = unit.block.site();
            layers.insert(site, site / 2);
            for (slot, list) in slots(&unit.block) {
                let count = counts_of(weights, site / 2, site, slot);
                let per = lists.entry((site, slot)).or_insert_with(|| vec![(Vec::new(), false); circuits.len()]);
                per[j].0.extend(list.iter().copied().filter(|&i| i < count));
                per[j].1 |= list.contains(&count);
            }
        }
    }
    // The merged circuit: one unit per site, in site order, each slot's list the union (sorted) and
    // the remainder last when any copy names it.
    let mut units = Vec::new();
    let mut masks = BTreeMap::new();
    for (&site, &layer) in &layers {
        let slots = if site % 2 == 0 { 4 } else { 2 };
        let mut merged: Vec<Vec<usize>> = vec![Vec::new(); slots];
        for (slot, list) in merged.iter_mut().enumerate() {
            let Some(per) = lists.get(&(site, slot)) else { continue };
            let mut union: Vec<usize> = per.iter().flat_map(|(l, _)| l.iter().copied()).collect();
            union.sort_unstable();
            union.dedup();
            let mut counts = Array2::<f64>::zeros((circuits.len(), union.len()));
            for (j, (l, _)) in per.iter().enumerate() {
                for i in l {
                    if let Ok(at) = union.binary_search(i) {
                        counts[[j, at]] += 1.0;
                    }
                }
            }
            let remainder: Vec<bool> = per.iter().map(|(_, r)| *r).collect();
            let total = counts_of(weights, layer, site, slot);
            *list = union;
            if remainder.iter().any(|&r| r) {
                list.push(total);
            }
            masks.insert((units.len(), slot), crate::graph_device::CopyMask { counts, remainder });
        }
        let block = if site % 2 == 0 {
            let [q, k, v, o]: [Vec<usize>; 4] = merged.try_into().ok()?;
            Block::AttnSlices { layer, q, k, v, o, rest: false }
        } else {
            let [fc, down]: [Vec<usize>; 2] = merged.try_into().ok()?;
            Block::Slices { layer, fc, down, rest: false }
        };
        // With counterfactual stand-ins, the site's stand-in unit: its `rest` (not computing), on each
        // copy's rows the counterfactual write of the parts the copy does not name (the computing
        // unit's `down_proj` or `o_proj` counts), as a program's remainder unit.
        let flipped = |rest: bool| match &block {
            Block::AttnSlices { layer, q, k, v, o, .. } => Block::AttnSlices { layer: *layer, q: q.clone(), k: k.clone(), v: v.clone(), o: o.clone(), rest },
            Block::Slices { layer, fc, down, .. } => Block::Slices { layer: *layer, fc: fc.clone(), down: down.clone(), rest },
            other => other.clone(),
        };
        let standin_unit = (!delete).then(|| flipped(!complement));
        let block = if complement { flipped(true) } else { block.clone() };
        let at = units.len();
        units.push(Unit { block, computes: true, at: None, routes: [Incoming::all(), Incoming::all(), Incoming::all()], hidden: Incoming::all() });
        if let Some(block) = standin_unit {
            let slot = if site % 2 == 0 { 3 } else { 1 };
            if let Some(m) = masks.get(&(at, slot)).cloned() {
                masks.insert((at + 1, slot), m);
            }
            units.push(Unit { block, computes: false, at: None, routes: [Incoming::all(), Incoming::all(), Incoming::all()], hidden: Incoming::all() });
        }
    }
    let merged = Circuit { nodes: units.len(), units, logits: Incoming::all(), delete: true, places: Vec::new() };
    let copies = crate::graph_device::Copies { count: circuits.len(), rows, masks, standin, resume: None, halt: None, embed_out: Vec::new() };
    let tokens: Vec<u32> = (0..circuits.len()).flat_map(|_| batch.tokens.iter().copied()).collect();
    let spans: Vec<(usize, usize)> = (0..circuits.len()).flat_map(|j| batch.spans.iter().map(move |&(start, n)| (j * rows + start, n))).collect();
    let scored: Vec<usize> = (0..circuits.len()).flat_map(|j| scored.iter().map(move |&r| j * rows + r)).collect();
    Some(Stacked { merged, copies, tokens, spans, scored, swaps: BTreeMap::new(), ops: Interventions::default() })
}

/// Per site, the computing units' blocks of `circuit` (as text): two circuits whose signatures agree
/// at every site before one compute alike up to it.
fn site_signatures(circuit: &Circuit) -> BTreeMap<usize, String> {
    let mut out: BTreeMap<usize, String> = BTreeMap::new();
    for unit in circuit.units.iter().filter(|u| u.computes) {
        out.entry(unit.block.site()).or_default().push_str(&format!("{:?};", unit.block));
    }
    out
}

/// The number of VPD subcomponents of a slot (an MLP's `c_fc` 0 or `down_proj` 1; an attention's
/// `q_proj` 0 to `o_proj` 3) at `layer`: the remainder's index.
fn counts_of(weights: &Weights, layer: usize, site: usize, slot: usize) -> usize {
    if site % 2 == 0 {
        weights.vpd_attention.get(&layer).map_or(0, |a| [a.q.0.nrows(), a.k.0.nrows(), a.v.0.nrows(), a.o.0.nrows()][slot.min(3)])
    } else {
        weights.vpd.get(&layer).map_or(0, |v| if slot == 0 { v.fc_u.nrows() } else { v.down_u.nrows() })
    }
}

/// [`execute`] under the row interventions `ops` (site operations, [`Interventions`]): after a
/// site, writers' writes scaled or swapped and vectors pushed into the stream; at a site, its
/// units' normed inputs scaled, pushed or swapped and cut writers read on the donor.
pub fn execute_with(weights: &Weights, circuit: &Circuit, batch: &Batch, scored: &[usize], swaps: &BTreeMap<usize, Array2<f64>>, ops: &Interventions) -> Result<Execution, String> {
    run(weights, circuit, batch, scored, swaps, ops, false)
}

/// The stand-ins a run of `circuit` on `batch` reads (`embed`'s and each unit's, rows × width):
/// each unit's write in `batch.reference`, `M`'s run on the counterfactuals; zeros for a circuit
/// whose every unit computes (`M` itself reads none); any other circuit without the counterfactual
/// run is an error (`Checker::referenced` attaches it).
pub(crate) fn standins_of(weights: &Weights, circuit: &Circuit, batch: &Batch) -> Result<(Array2<f64>, Vec<Array2<f64>>), String> {
    let (rows, d, units) = (batch.tokens.len(), weights.width(), circuit.units.len());
    Ok(match &batch.reference {
        None if circuit.units.iter().all(|u| u.computes) => (Array2::zeros((rows, d)), vec![Array2::zeros((rows, d)); units]),
        None => return Err("a program's undeclared pieces take their values from the counterfactual run, which this batch lacks".into()),
        Some(r) if r.zero && r.embed.nrows() == rows => (Array2::zeros((rows, d)), vec![Array2::zeros((rows, d)); units]),
        Some(r) => {
            if r.embed.nrows() != rows {
                return Err(format!("a counterfactual run of {} tokens for a batch of {rows}", r.embed.nrows()));
            }
            (r.embed.clone(), circuit.units.iter().map(|u| r.write(weights, &u.block)).collect::<Result<_, _>>()?)
        }
    })
}

/// [`execute_with`], with `capture` recording every head's read and every MLP's activations and
/// write ([`Reference`]; every unit must compute, each layer's heads and MLP one unit each).
/// Stand-ins: `standins_of`.
fn run(weights: &Weights, circuit: &Circuit, batch: &Batch, scored: &[usize], swaps: &BTreeMap<usize, Array2<f64>>, ops: &Interventions, capture: bool) -> Result<Execution, String> {
    let (rows, d) = (batch.tokens.len(), weights.width());
    let vocabulary = weights.embedding.nrows();
    if let Some(t) = batch.tokens.iter().find(|t| **t as usize >= vocabulary) {
        return Err(format!("token {t} outside the vocabulary of {vocabulary}"));
    }
    let embed = wide(weights.embedding.select(Axis(0), &batch.tokens.iter().map(|t| *t as usize).collect::<Vec<_>>()).view());
    let units = circuit.units.len();
    // The device runs what it covers (`graph_device`, stand-ins assembled there); everything else
    // runs here.
    if batch.blocks.iter().all(Vec::is_empty) {
        let job = crate::graph_device::Run { tokens: &batch.tokens, spans: &batch.spans, scored, swaps, capture, reference: batch.reference.as_deref(), ops };
        if let Some(out) = crate::graph_device::run(weights, circuit, &job) {
            return out;
        }
    }
    let (embed_standin, standins) = standins_of(weights, circuit, batch)?;
    let mut captured = capture.then(|| Reference {
        id: next_reference_id(),
        embed: embed.clone(),
        reads: weights.layers.iter().map(|l| vec![Array2::zeros((0, 0)); l.heads.len()]).collect(),
        active: vec![Array2::zeros((0, 0)); weights.layers.len()],
        mlp: vec![Array2::zeros((0, 0)); weights.layers.len()],
        inputs: vec![Array2::zeros((0, 0)); weights.layers.len()],
        attention_inputs: vec![Array2::zeros((0, 0)); weights.layers.len()],
        zero: false,
    });
    let mut order: Vec<usize> = (0..circuit.units.len()).collect();
    order.sort_by_key(|&u| circuit.units[u].block.site());
    // The actual stream and the stand-in stream entering the current site, every unit's write and
    // the per-row factors of the stand-ins interventions scaled (`embed`'s last).
    let mut st = Streams {
        stream: embed.clone(),
        standin_stream: embed_standin.clone(),
        embed,
        writes: vec![None; units],
        factors: vec![None; units + 1],
    };
    let mut normed_kept = BTreeMap::new();
    let mut reads_kept = BTreeMap::new();
    // Positions ([`Positions`]): per unit the rows it acts at (none: every row), and per swapped
    // node the rows its swap replaces (none: every row).
    let acts: Vec<Option<Vec<bool>>> = circuit.units.iter().map(|u| u.at.as_ref().map(|(p, invert)| p.rows(batch, *invert))).collect();
    let swapped_at: Vec<Option<Vec<bool>>> = (0..units).map(|u| if swaps.contains_key(&u) { circuit.places.get(u).and_then(Option::as_ref).map(|p| p.rows(batch, false)) } else { None }).collect();
    let full_swap = |u: usize| swaps.contains_key(&u) && swapped_at[u].is_none();
    // A unit's contribution to queries, keys, values or a hidden pre-activation (actual minus
    // stand-in) is zero at the rows it does not act at.
    let at_rows = |u: usize, mut delta: Array2<f64>| -> Array2<f64> {
        if let Some(rows) = &acts[u] {
            for (mut row, &on) in delta.outer_iter_mut().zip(rows) {
                if !on {
                    row.fill(0.0);
                }
            }
        }
        delta
    };
    // A unit's write: the swap's rows where it is swapped, its stand-in's where it does not act.
    let finish = |u: usize, mut write: Array2<f64>| -> Array2<f64> {
        if let (Some(rows), Some(value)) = (&swapped_at[u], swaps.get(&u)) {
            for ((mut row, &on), v) in write.outer_iter_mut().zip(rows).zip(value.outer_iter()) {
                if on {
                    row.assign(&v);
                }
            }
        }
        if let Some(rows) = &acts[u] {
            for ((mut row, &on), v) in write.outer_iter_mut().zip(rows).zip(standins[u].outer_iter()) {
                if !on {
                    row.assign(&v);
                }
            }
        }
        write
    };
    let input = |incoming: &Incoming, st: &Streams| -> Array2<f64> {
        match incoming {
            Incoming::AllBut(cut) => {
                let mut x = st.stream.clone();
                for &w in cut {
                    if let Some(dw) = st.delta(w, &embed_standin, &standins) {
                        x -= &dw;
                    }
                }
                x
            }
            Incoming::Only(kept) => {
                let mut x = st.standin_stream.clone();
                for &w in kept {
                    if let Some(dw) = st.delta(w, &embed_standin, &standins) {
                        x += &dw;
                    }
                }
                x
            }
        }
    };
    ops.after(None, &mut st, &embed_standin, &standins)?;
    let mut at = 0;
    while at < order.len() {
        let site = circuit.units[order[at]].block.site();
        let end = order[at..].iter().position(|&u| circuit.units[u].block.site() != site).map_or(order.len(), |k| at + k);
        // A VPD-view MLP computes its units together: each reader's hidden pre-activation is the
        // counterfactual one plus the c_fc writes it reads (on x minus on x').
        let mut slice_writes: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
        let slices: Vec<usize> = order[at..end].iter().copied().filter(|&u| matches!(circuit.units[u].block, Block::Slices { .. })).collect();
        if let Some(&first) = slices.first() {
            let Block::Slices { layer, .. } = circuit.units[first].block else { return Err("a VPD-view unit of another block".into()) };
            let mlp = weights.layers[layer].mlp.as_ref().ok_or("a VPD view of a layer without an MLP")?;
            let vpd = weights.vpd.get(&layer).ok_or_else(|| format!("layer {layer} has no VPD view"))?;
            let norm = &weights.layers[layer].mlp_norm;
            let x_ref = match &batch.reference {
                Some(r) => r.inputs.get(layer).cloned().ok_or("an MLP the reference did not record")?,
                None => Array2::zeros((rows, d)),
            };
            let fc_of = |b: &Block, x: &Array2<f64>| match b {
                Block::Slices { fc, rest, .. } => Ok(vpd.fc(mlp, fc, *rest, x)),
                _ => Err("a VPD-view MLP unit of another block".to_string()),
            };
            let mut pre_ref = x_ref.dot(&wide(mlp.gate.view()).t());
            pre_ref += &mlp.bias.view().insert_axis(Axis(0));
            let mut deltas: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
            for &u in &slices {
                let unit = &circuit.units[u];
                if !unit.computes || full_swap(u) {
                    continue;
                }
                let routes = unit.block.routes();
                let mut inputs: Vec<Array2<f64>> = routes.iter().map(|r| input(&unit.routes[r.slot()], &st)).collect();
                ops.cut_inputs(site, unit, routes, &mut inputs, &st)?;
                let mut x_hat = norm.apply(&inputs[0]);
                ops.normed(site, u, 0, &mut x_hat, &mut normed_kept);
                deltas.insert(u, at_rows(u, fc_of(&unit.block, &x_hat)? - fc_of(&unit.block, &x_ref)?));
            }
            for &u in &slices {
                let unit = &circuit.units[u];
                if !unit.computes || full_swap(u) {
                    continue;
                }
                let Block::Slices { down, rest, .. } = &unit.block else { return Err("a VPD-view unit of another block".into()) };
                if down.is_empty() && !rest {
                    slice_writes.insert(u, Array2::zeros((rows, d)));
                    continue;
                }
                let mut pre = pre_ref.clone();
                for (&w, delta) in &deltas {
                    let reads = match &unit.hidden {
                        Incoming::AllBut(cut) => !cut.contains(&Writer::Unit(w)),
                        Incoming::Only(kept) => kept.contains(&Writer::Unit(w)),
                    };
                    if reads {
                        pre += delta;
                    }
                }
                pre.mapv_inplace(|t| mlp.law.apply(t));
                slice_writes.insert(u, vpd.down(mlp, down, *rest, &pre));
            }
        }
        // A VPD-view attention likewise: each reader of the queries, keys and values (a unit with
        // o_proj subcomponents) takes the counterfactual ones plus the q/k/v writes it reads, runs
        // the heads' attention on them and writes through its o_proj subcomponents.
        let attention: Vec<usize> = order[at..end].iter().copied().filter(|&u| matches!(circuit.units[u].block, Block::AttnSlices { .. })).collect();
        if let Some(&first) = attention.first() {
            let Block::AttnSlices { layer, .. } = circuit.units[first].block else { return Err("a VPD-view unit of another block".into()) };
            let lw = &weights.layers[layer];
            let vpd = weights.vpd_attention.get(&layer).ok_or_else(|| format!("layer {layer}'s attention has no VPD view"))?;
            let maps = attention_maps(lw);
            let x_ref = match &batch.reference {
                Some(r) => r.attention_inputs.get(layer).cloned().ok_or("an attention the reference did not record")?,
                // `M` (every unit computing and read) needs no reference: the deltas sum to its own
                // queries, keys and values.
                None => Array2::zeros((rows, d)),
            };
            let factors = [&vpd.q, &vpd.k, &vpd.v];
            let refs: Vec<Array2<f64>> = (0..3).map(|m| x_ref.dot(&maps[m].t())).collect();
            let mut deltas: BTreeMap<usize, Vec<Array2<f64>>> = BTreeMap::new();
            for &u in &attention {
                let unit = &circuit.units[u];
                if !unit.computes || full_swap(u) {
                    continue;
                }
                let Block::AttnSlices { q, k, v, rest, .. } = &unit.block else { return Err("a VPD-view unit of another block".into()) };
                let routes = unit.block.routes();
                let mut inputs: Vec<Array2<f64>> = routes.iter().map(|r| input(&unit.routes[r.slot()], &st)).collect();
                ops.cut_inputs(site, unit, routes, &mut inputs, &st)?;
                let mut ds = Vec::with_capacity(3);
                for (m, list) in [q, k, v].into_iter().enumerate() {
                    let mut x_hat = lw.attention.apply(&inputs[m]);
                    ops.normed(site, u, m, &mut x_hat, &mut normed_kept);
                    ds.push(at_rows(u, sliced(factors[m], &maps[m], list, *rest, &x_hat) - sliced(factors[m], &maps[m], list, *rest, &x_ref)));
                }
                deltas.insert(u, ds);
            }
            for &u in &attention {
                let unit = &circuit.units[u];
                if !unit.computes || full_swap(u) {
                    continue;
                }
                let Block::AttnSlices { o, rest, .. } = &unit.block else { return Err("a VPD-view unit of another block".into()) };
                if o.is_empty() && !rest {
                    slice_writes.insert(u, Array2::zeros((rows, d)));
                    continue;
                }
                let mut qkv = refs.clone();
                for (&w, ds) in &deltas {
                    let reads = match &unit.hidden {
                        Incoming::AllBut(cut) => !cut.contains(&Writer::Unit(w)),
                        Incoming::Only(kept) => kept.contains(&Writer::Unit(w)),
                    };
                    if reads {
                        for (x, dx) in qkv.iter_mut().zip(ds) {
                            *x += dx;
                        }
                    }
                }
                let mut z = Array2::<f64>::zeros((rows, maps[3].ncols()));
                let (mut qc, mut kc, mut vc) = (0, 0, 0);
                for hw in &lw.heads {
                    let (qw, kw, vw) = (hw.query.nrows(), hw.key.nrows(), hw.value.nrows());
                    for (n, &(start, length)) in batch.spans.iter().enumerate() {
                        let positions: Vec<u32> = (0..length as u32).collect();
                        let span = start..start + length;
                        let qs = qkv[0].slice(s![span.clone(), qc..qc + qw]).to_owned();
                        let ks = qkv[1].slice(s![span.clone(), kc..kc + kw]).to_owned();
                        let (qs, ks) = (rotate(&qs, hw.rotary, &positions, false), rotate(&ks, hw.rotary, &positions, false));
                        let mut a = probabilities(qs.view(), ks.view(), &positions, 0, hw.scale, hw.causal);
                        if let Some(b) = batch.blocks.get(n).filter(|b| !b.is_empty()) {
                            block_attention(&mut a, b);
                        }
                        z.slice_mut(s![span.clone(), vc..vc + vw]).assign(&a.dot(&qkv[2].slice(s![span, vc..vc + vw])));
                    }
                    (qc, kc, vc) = (qc + qw, kc + kw, vc + vw);
                }
                ops.head_reads_of(site, u, lw, &mut z, &mut reads_kept);
                slice_writes.insert(u, sliced(&vpd.o, &maps[3], o, *rest, &z));
            }
        }
        for &u in &order[at..end] {
            let unit = &circuit.units[u];
            if !unit.computes {
                continue;
            }
            if full_swap(u) {
                st.writes[u] = Some(swaps[&u].clone());
                continue;
            }
            if let Some(w) = slice_writes.remove(&u) {
                st.writes[u] = Some(finish(u, w));
                continue;
            }
            let routes = unit.block.routes();
            let mut inputs: Vec<Array2<f64>> = routes.iter().map(|r| input(&unit.routes[r.slot()], &st)).collect();
            ops.cut_inputs(site, unit, routes, &mut inputs, &st)?;
            let mut normed = |slot: usize, x: &mut Array2<f64>| ops.normed(site, u, slot, x, &mut normed_kept);
            let write = match &unit.block {
                Block::Heads { layer, heads } => {
                    let (w, reads) = heads_write(&weights.layers[*layer], heads, [&inputs[0], &inputs[1], &inputs[2]], &batch.spans, &batch.blocks, capture, &mut normed);
                    if let Some(c) = captured.as_mut() {
                        for (h, z) in heads.iter().zip(reads) {
                            c.reads[*layer][*h] = z;
                        }
                        c.attention_inputs[*layer] = weights.layers[*layer].attention.apply(&inputs[0]);
                    }
                    w
                }
                Block::Neurons { layer, neurons } => {
                    let lw = &weights.layers[*layer];
                    let mlp = lw.mlp.as_ref().ok_or("a neuron block without an MLP")?;
                    let mut x_hat = lw.mlp_norm.apply(&inputs[0]);
                    normed(0, &mut x_hat);
                    let active = neurons_active(mlp, neurons, &x_hat);
                    let write = par_dot(&active, wide(mlp.out.select(Axis(1), neurons).view()).t());
                    if let Some(c) = captured.as_mut() {
                        if neurons.len() != mlp.gate.nrows() {
                            return Err("a capture needs each MLP whole in one unit".into());
                        }
                        c.mlp[*layer] = write.clone();
                        c.active[*layer] = active;
                        c.inputs[*layer] = x_hat;
                    }
                    write
                }
                Block::Features { layer, features, rest } => {
                    let mut x_hat = weights.layers[*layer].mlp_norm.apply(&inputs[0]);
                    normed(0, &mut x_hat);
                    features_write(weights, *layer, features, *rest, &x_hat)?
                }
                Block::Slices { .. } | Block::AttnSlices { .. } => return Err("a VPD block outside its site's pass".into()),
            };
            st.writes[u] = Some(finish(u, write));
        }
        for &u in &order[at..end] {
            match &st.writes[u] {
                Some(w) => st.stream += w,
                None => st.stream += &standins[u],
            }
            st.standin_stream += &standins[u];
        }
        ops.after(Some(site), &mut st, &embed_standin, &standins)?;
        at = end;
    }
    let last = input(&circuit.logits, &st).select(Axis(0), scored);
    let log_probabilities = log_probabilities(weights, &last)?;
    Ok(Execution { log_probabilities, writes: st.writes, normed: normed_kept, reads: reads_kept, captured })
}

/// `a · b`, row blocks of `a` on parallel threads when called outside rayon's pool (a run the
/// batch makes alone: a weight edit's group of one, a counterfactual run every program waits
/// for). Inside the pool it multiplies on its own thread: a worker that waits on nested parallel
/// work steals other runs, and a stolen run that waits on the counterfactual run this worker is
/// computing (a `OnceLock`) never returns, which deadlocked a vpd4l batch.
fn par_dot(a: &Array2<f64>, b: ndarray::ArrayView2<f64>) -> Array2<f64> {
    const BLOCK: usize = 64;
    if let Some(out) = crate::graph_device::dot(a, b) {
        return out;
    }
    if a.nrows() <= BLOCK || rayon::current_thread_index().is_some() {
        return a.dot(&b);
    }
    let mut out = Array2::<f64>::zeros((a.nrows(), b.ncols()));
    out.axis_chunks_iter_mut(Axis(0), BLOCK).into_par_iter().zip(a.axis_chunks_iter(Axis(0), BLOCK).into_par_iter()).for_each(|(mut o, x)| o.assign(&x.dot(&b)));
    out
}

pub use crate::graph_device::use_device;

/// Next-token log-probabilities of final streams (rows × width) through the final norm (its own
/// RMS, its gain) and the unembedding, normalized in float64.
pub fn log_probabilities(weights: &Weights, last: &Array2<f64>) -> Result<Array2<f64>, String> {
    let gained = last * &weights.final_norm.gain.view().insert_axis(Axis(0));
    let mut logits = match crate::graph_device::logits(weights, &gained) {
        Some(l) => l,
        None => {
            // Widened a block of the vocabulary at a time (Qwen3's whole unembedding is 1.2 GB in
            // float64).
            let mut out = Array2::<f64>::zeros((gained.nrows(), weights.unembedding.nrows()));
            for (mut o, u) in out.axis_chunks_iter_mut(Axis(1), 8192).zip(weights.unembedding.axis_chunks_iter(Axis(0), 8192)) {
                o.assign(&par_dot(&gained, wide(u).t()));
            }
            out
        }
    };
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

/// What an error compares in each scored row ([`Checker::metric`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// The whole next-token distribution: `KL(p ‖ q)`.
    #[default]
    Full,
    /// The behavior's answers: per scored row the distribution over the prompt's answer `a`, the
    /// counterfactual's answer `a'` and any other token under `p` and `q`, compared by `KL` (a VPD
    /// subnetwork's standard, the target prediction kept rather than the whole distribution, with the
    /// contrast circuit metrics compare, the logit difference of `a` and `a'`, in it). The answers are
    /// the behavior's ([`Checker::answers_at`]); a row without them takes the token `by` puts first.
    Answer,
    /// The behavior's choice: per scored row the distribution over the prompt's answer `a` and the
    /// counterfactual's answer `a'` alone (renormalized) under `p` and `q`, compared by `KL`: how far
    /// the log odds of `a` against `a'` move, the contrast circuit metrics compare. A row whose answers
    /// coincide or are missing scores 0.
    Choice,
}

/// Per row, `KL` in bits between the distributions of `p` and `q` restricted to the two tokens of
/// `picks[row]` (renormalized); 0 for a row without two distinct picks.
pub fn choice_kl_bits(p: &Array2<f64>, q: &Array2<f64>, picks: &[Vec<usize>]) -> Vec<f64> {
    let pair = |l: f64, m: f64| {
        let total = l.max(m) + (-(l - m).abs()).exp().ln_1p();
        (l - total, m - total)
    };
    (0..p.nrows())
        .map(|row| match picks.get(row).map(Vec::as_slice) {
            Some(&[a, b, ..]) if a != b => {
                let (pa, pb) = pair(p[[row, a]], p[[row, b]]);
                let (qa, qb) = pair(q[[row, a]], q[[row, b]]);
                (pa.exp() * (pa - qa) + pb.exp() * (pb - qb)) / std::f64::consts::LN_2
            }
            _ => 0.0,
        })
        .collect()
}

/// Per row, `KL` in bits between the distributions of `p` and `q` over the outcomes `picks[row]`
/// (distinct tokens) and any other token; a row with no picks takes the token `by` puts first.
pub fn answer_kl_bits(p: &Array2<f64>, q: &Array2<f64>, by: &Array2<f64>, picks: &[Vec<usize>]) -> Vec<f64> {
    let term = |a: f64, b: f64| if a > 0.0 { a * (a / b.max(1e-300)).ln() } else { 0.0 };
    p.outer_iter()
        .zip(q.outer_iter())
        .zip(by.outer_iter())
        .enumerate()
        .map(|(row, ((p, q), by))| {
            let mut tokens: Vec<usize> = picks.get(row).cloned().unwrap_or_default();
            tokens.dedup();
            if tokens.is_empty() {
                tokens.push(by.iter().enumerate().fold((0, f64::NEG_INFINITY), |best, (i, &v)| if v > best.1 { (i, v) } else { best }).0);
            }
            let (pa, qa): (Vec<f64>, Vec<f64>) = tokens.iter().map(|&t| (p[t].exp(), q[t].exp())).unzip();
            let (p_rest, q_rest) = ((1.0 - pa.iter().sum::<f64>()).max(0.0), (1.0 - qa.iter().sum::<f64>()).max(1e-300));
            (pa.iter().zip(&qa).map(|(&a, &b)| term(a, b)).sum::<f64>() + term(p_rest, q_rest)) / std::f64::consts::LN_2
        })
        .collect()
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
    /// VPD subcomponents `indices` of the MLP's `c_fc` (`down` false) or `down_proj` times `factor`:
    /// the matrix changes by `(factor − 1) Σ_i U_i ⊗ V_i` and each subcomponent's `U_i` by the
    /// factor, so the remainder piece stays as it was (the layer's VPD view, `Weights::vpd`).
    Subcomponents { layer: usize, down: bool, indices: Vec<usize>, factor: f64 },
    /// VPD subcomponents `indices` of the attention's `q_proj`, `k_proj`, `v_proj` or `o_proj`
    /// (`map` 0, 1, 2, 3) times `factor`, as [`WeightEdit::Subcomponents`]: each head's rows (its
    /// columns of `o_proj`) of `(factor − 1) Σ_i U_i ⊗ V_i` added to its map, `U_i` scaled.
    AttnSubcomponents { layer: usize, map: usize, indices: Vec<usize>, factor: f64 },
}

impl WeightEdit {
    fn matrix<'a>(weights: &'a mut Weights, layer: usize, head: Option<usize>, matrix: Matrix) -> Result<&'a mut Stored, String> {
        let lw = weights.layers.get_mut(layer).ok_or("no such layer")?;
        let m = match (head, matrix) {
            (Some(h), m) => {
                let w = lw.heads.get_mut(h).ok_or("no such head")?;
                match m {
                    Matrix::Query => &mut w.query,
                    Matrix::Key => Arc::make_mut(&mut w.key),
                    Matrix::Value => Arc::make_mut(&mut w.value),
                    Matrix::Output => &mut w.output,
                    _ => return Err("an MLP matrix of a head".into()),
                }
            }
            (None, Matrix::Gate) => &mut lw.mlp.as_mut().ok_or("no MLP")?.gate,
            (None, Matrix::Down) => &mut lw.mlp.as_mut().ok_or("no MLP")?.out,
            _ => return Err("a head matrix without a head".into()),
        };
        // The device's resident copies of matrices of this shape are uploaded again.
        crate::graph_device::edited(m);
        Ok(m)
    }

    /// Applies the edit; returns what restores the weights.
    pub fn apply(&self, weights: &mut Weights) -> Result<Restore, String> {
        if let Self::AttnSubcomponents { layer, map, indices, factor } = self {
            return Self::attention_subcomponents(weights, *layer, *map, indices, *factor);
        }
        let (layer, head, matrix) = match self {
            Self::Head { layer, head, .. } => (*layer, Some(*head), Matrix::Output),
            Self::Neurons { layer, .. } => (*layer, None, Matrix::Down),
            Self::RankOne { layer, head, matrix, .. } => (*layer, *head, *matrix),
            Self::Subcomponents { layer, down, .. } => (*layer, None, if *down { Matrix::Down } else { Matrix::Gate }),
            Self::AttnSubcomponents { .. } => return Err("handled above".into()),
        };
        // A subcomponent edit's change of the matrix, and its factors' saved values.
        let mut factors = None;
        let mut delta = None;
        if let Self::Subcomponents { layer, down, indices, factor } = self {
            let vpd = weights.vpd.get_mut(layer).ok_or_else(|| format!("layer {layer} has no VPD view"))?;
            let (u, v) = if *down { (&mut vpd.down_u, &vpd.down_v) } else { (&mut vpd.fc_u, &vpd.fc_v) };
            if let Some(i) = indices.iter().find(|&&i| i >= u.nrows()) {
                return Err(format!("subcomponent {i} out of range"));
            }
            // c_fc: (hidden × width) += U_fc[i] ⊗ V_fc[:, i]; down_proj: (width × hidden) += U_down[i] ⊗ V_down[:, i].
            let picked_u = u.select(Axis(0), indices);
            let picked_v = v.select(Axis(1), indices);
            delta = Some(picked_u.t().dot(&picked_v.t()) * (factor - 1.0));
            factors = Some((*down, u.clone()));
            crate::graph_device::edited(u);
            for &i in indices {
                u.row_mut(i).mapv_inplace(|x| x * factor);
            }
        }
        let maps = key_value_maps(weights, layer, matrix);
        let m = Self::matrix(weights, layer, head, matrix)?;
        let saved = m.clone();
        match self {
            Self::AttnSubcomponents { .. } => return Err("an attention subcomponent edit goes through attention_subcomponents".into()),
            Self::Subcomponents { .. } => {
                let d = delta.ok_or("no change")?;
                if d.dim() != m.dim() {
                    return Err("VPD factors of another shape than the matrix".into());
                }
                add_wide(m.view_mut(), d.view());
            }
            Self::Head { factor, .. } => m.mapv_inplace(|w| (f64::from(w) * factor) as f32),
            Self::Neurons { neurons, factor, .. } => {
                for &i in neurons {
                    if i >= m.ncols() {
                        return Err(format!("neuron {i} out of range"));
                    }
                    m.column_mut(i).mapv_inplace(|w| (f64::from(w) * factor) as f32);
                }
            }
            Self::RankOne { u, v, .. } => {
                if u.len() != m.nrows() || v.len() != m.ncols() {
                    return Err("a rank-one edit of the wrong shape".into());
                }
                for (i, ui) in u.iter().enumerate() {
                    for (j, vj) in v.iter().enumerate() {
                        m[[i, j]] = (f64::from(m[[i, j]]) + ui * vj) as f32;
                    }
                }
            }
        }
        // Keys and values a group of query heads shares (grouped-query attention) are one matrix of
        // M: a rank-one edit of one head's applies to every head that shares it.
        let mut shared = Vec::new();
        if let (Self::RankOne { .. }, Some(h), Matrix::Key | Matrix::Value) = (self, head, matrix) {
            let edited = Self::matrix(weights, layer, head, matrix)?.clone();
            for g in (0..weights.layers[layer].heads.len()).filter(|&g| g != h) {
                let m = Self::matrix(weights, layer, Some(g), matrix)?;
                if *m == saved {
                    *m = edited.clone();
                    shared.push(g);
                }
            }
        }
        Ok(Restore { layer, head, matrix, saved, factors, shared, heads: Vec::new(), attention: None, maps })
    }

    /// [`WeightEdit::AttnSubcomponents`]: the change split over the heads (rows `h·d..(h+1)·d` of
    /// the stacked query, key or value map, columns of the output map), every head's map saved.
    fn attention_subcomponents(weights: &mut Weights, layer: usize, map: usize, indices: &[usize], factor: f64) -> Result<Restore, String> {
        let matrix = [Matrix::Query, Matrix::Key, Matrix::Value, Matrix::Output].get(map).copied().ok_or("an attention map is 0..4")?;
        let vpd = weights.vpd_attention.get_mut(&layer).ok_or_else(|| format!("layer {layer} has no VPD view of attention"))?;
        let factors = match map {
            0 => &mut vpd.q,
            1 => &mut vpd.k,
            2 => &mut vpd.v,
            _ => &mut vpd.o,
        };
        if let Some(i) = indices.iter().find(|&&i| i >= factors.0.nrows()) {
            return Err(format!("subcomponent {i} out of range"));
        }
        // (out × in) change: U rows are outputs, V columns inputs.
        let delta = factors.0.select(Axis(0), indices).t().dot(&factors.1.select(Axis(1), indices).t()) * (factor - 1.0);
        let saved_u = factors.0.clone();
        crate::graph_device::edited(&factors.0);
        for &i in indices {
            factors.0.row_mut(i).mapv_inplace(|x| x * factor);
        }
        let mut heads = Vec::new();
        let maps = key_value_maps(weights, layer, matrix);
        let mut at = 0;
        for h in 0..weights.layers[layer].heads.len() {
            let m = Self::matrix(weights, layer, Some(h), matrix)?;
            heads.push((h, m.clone()));
            let part = if matrix == Matrix::Output { delta.slice(s![.., at..at + m.ncols()]) } else { delta.slice(s![at..at + m.nrows(), ..]) };
            if part.dim() != m.dim() {
                return Err("VPD attention factors of another shape than the heads' maps".into());
            }
            at += if matrix == Matrix::Output { m.ncols() } else { m.nrows() };
            add_wide(m.view_mut(), part);
        }
        let (first, saved) = heads.first().cloned().ok_or("a layer without heads")?;
        Ok(Restore { layer, head: Some(first), matrix, saved, factors: None, shared: Vec::new(), heads: heads.into_iter().skip(1).collect(), attention: Some((map, saved_u)), maps })
    }
}

/// Every head's key and value maps of `layer` before an edit of `matrix` changes them (none for
/// other matrices): [`Restore`] puts these back so a key/value group shares one map again.
fn key_value_maps(weights: &Weights, layer: usize, matrix: Matrix) -> Vec<(Arc<Stored>, Arc<Stored>)> {
    match (matrix, weights.layers.get(layer)) {
        (Matrix::Key | Matrix::Value, Some(lw)) => lw.heads.iter().map(|w| (w.key.clone(), w.value.clone())).collect(),
        _ => Vec::new(),
    }
}

/// A matrix's value before an edit.
pub struct Restore {
    layer: usize,
    head: Option<usize>,
    matrix: Matrix,
    saved: Stored,
    /// A subcomponent edit's VPD factors before it (`down_proj`'s when true).
    factors: Option<(bool, Array2<f64>)>,
    /// The heads sharing the edited key or value matrix, edited alike.
    shared: Vec<usize>,
    /// Further heads' maps before an attention subcomponent edit, and its factors' `U`.
    heads: Vec<(usize, Stored)>,
    attention: Option<(usize, Array2<f64>)>,
    /// A key or value edit's heads' maps before it (`key_value_maps`), restored as they were shared.
    maps: Vec<(Arc<Stored>, Arc<Stored>)>,
}

impl Restore {
    pub fn restore(self, weights: &mut Weights) -> Result<(), String> {
        for g in &self.shared {
            *WeightEdit::matrix(weights, self.layer, Some(*g), self.matrix)? = self.saved.clone();
        }
        for (h, m) in self.heads {
            *WeightEdit::matrix(weights, self.layer, Some(h), self.matrix)? = m;
        }
        if let Some((map, u)) = self.attention {
            let vpd = weights.vpd_attention.get_mut(&self.layer).ok_or("the VPD view of attention went missing")?;
            let factors = match map {
                0 => &mut vpd.q,
                1 => &mut vpd.k,
                2 => &mut vpd.v,
                _ => &mut vpd.o,
            };
            crate::graph_device::edited(&factors.0);
            factors.0 = u;
        }
        *WeightEdit::matrix(weights, self.layer, self.head, self.matrix)? = self.saved;
        if let Some((down, u)) = self.factors {
            let vpd = weights.vpd.get_mut(&self.layer).ok_or("the VPD view went missing")?;
            let target = if down { &mut vpd.down_u } else { &mut vpd.fc_u };
            crate::graph_device::edited(target);
            *target = u;
        }
        if let Some(lw) = weights.layers.get_mut(self.layer).filter(|_| !self.maps.is_empty()) {
            for (w, (key, value)) in lw.heads.iter_mut().zip(self.maps) {
                // The restored copies' allocations are freed: no resident copy may outlive them.
                crate::graph_device::edited(&w.key);
                crate::graph_device::edited(&w.value);
                (w.key, w.value) = (key, value);
            }
        }
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
    /// A weight edit, of a random piece (uniform) or of one of `M`'s strongest pieces for the
    /// behavior (targeted).
    Edit { edit: WeightEdit, targeted: bool },
    /// Operations on sites `M` and every program share (`interchange::SiteOp`: heads', attentions'
    /// and MLPs' outputs, the embeddings, the stream after a block, a block's normed input; swaps
    /// and cuts read the donor, the prompt's counterfactual), [`Interventions`].
    Sites { draw: SiteDraw },
}

impl Experiment {
    pub fn family(&self) -> &'static str {
        match self {
            Self::Clean => "clean",
            Self::Counterfactual => "counterfactual",
            Self::Edit { edit: WeightEdit::RankOne { .. }, .. } => "rank_one",
            Self::Edit { targeted: true, .. } => "edit_targeted",
            Self::Edit { targeted: false, .. } => "edit_uniform",
            Self::Sites { draw } => match draw.family {
                interchange::Family::Swap => "site_swap",
                interchange::Family::Zero => "site_zero",
                interchange::Family::Scale => "site_scale",
                interchange::Family::Push => "site_push",
                interchange::Family::Cut => "site_cut",
                interchange::Family::Read => "site_read",
                interchange::Family::Weight => "site_weight",
            },
        }
    }

    /// The experiment in words, for the reader.
    pub fn describe(&self) -> String {
        match self {
            Self::Clean => "the prompt as given".into(),
            Self::Counterfactual => "the counterfactual prompt in place of the prompt".into(),
            Self::Edit { edit, .. } => match edit {
                WeightEdit::Head { layer, head, factor } => format!("L[{layer}].head[{head}]'s output weights times {factor}"),
                WeightEdit::Neurons { layer, neurons, factor } => format!("L[{layer}].mlp{neurons:?}'s down weights times {factor}"),
                WeightEdit::RankOne { layer, head, matrix, .. } => format!("a random rank-one change of L[{layer}]{}'s {matrix:?} weights", head.map_or(".mlp".to_string(), |h| format!(".head[{h}]"))),
                WeightEdit::Subcomponents { layer, down, indices, factor } => format!("PD.vpd[{layer}].{}{indices:?} times {factor}", if *down { "down_proj" } else { "c_fc" }),
                WeightEdit::AttnSubcomponents { layer, map, indices, factor } => format!("PD.vpd[{layer}].{}{indices:?} times {factor}", ["q_proj", "k_proj", "v_proj", "o_proj"][(*map).min(3)]),
            },
            Self::Sites { draw } => describe_sites(draw),
        }
    }
}

/// Pieces and connections of `M` by measured effect on the behavior (`Checker::targets`, measured
/// once): each piece (a head, a whole MLP, a group of an MLP's neurons, a group of VPD
/// subcomponents) with the mean `KL(M ‖ M_e)` at the targets when it is removed, and each
/// connection (an attention's or an MLP's output into a later block's read, a site cut from the
/// counterfactual) with that of its cut, strongest first.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Targets {
    pub pieces: Vec<(Block, f64)>,
    pub cuts: Vec<(SiteOp, f64)>,
}

/// How many of the strongest pieces and of the strongest connections targeted draws take.
const TARGETED: usize = 8;

/// The version of `M`'s execution semantics in disk-cache names: raised whenever a change alters what
/// a cached outcome of `M` would be, so an old cache is never read as a new one.
const DISK_SEMANTICS: u64 = 2;

/// The experiments every program of a behavior is scored on (design.txt section 2), drawn from
/// `seed` and `M`'s own pieces alone, never from a program or an attached view (VPD, library,
/// transcoder), so every checker configuration scores the same set: clean, the counterfactual prompts (`counterfactual`),
/// then `count` draws, half uniform (weight edits of random pieces at random granularity,
/// rank-one perturbations and, with a manifest, its site operations, a third each) and half
/// targeted at `targets`' strongest pieces and connections (a piece removed or scaled, a head's or
/// an MLP's output swapped from the counterfactual at every token, a connection cut).
/// The complement experiments of a score's experiment set (necessity, [`Checker::necessity_runs`]),
/// the same for every program: the clean prompts, the set's first [`COMPLEMENT_EDITS`] weight edits
/// and its first [`COMPLEMENT_SITES`] site operations that read no donor.
pub fn complements(experiments: &[Experiment]) -> Vec<Experiment> {
    let edits = experiments.iter().filter(|e| matches!(e, Experiment::Edit { .. })).take(COMPLEMENT_EDITS);
    let sites = experiments.iter().filter(|e| matches!(e, Experiment::Sites { draw } if !Interventions::needs_donor(draw))).take(COMPLEMENT_SITES);
    std::iter::once(Experiment::Clean).chain(edits.cloned()).chain(sites.cloned()).collect()
}

/// Weight edits and site operations of the experiment set each program's necessity is also
/// measured under ([`complements`]).
pub const COMPLEMENT_EDITS: usize = 1;
pub const COMPLEMENT_SITES: usize = 1;

pub fn sample(weights: &Weights, counterfactual: bool, count: usize, seed: u64, targets: &Targets, units: &SiteUnits) -> Vec<Experiment> {
    let sites = &units.pool;
    let mut out = vec![Experiment::Clean];
    if counterfactual {
        out.push(Experiment::Counterfactual);
    }
    let layers = weights.layers.len();
    let factors = [0.0, 0.5, 2.0];
    let random_edit = |rng: &mut StdRng, block: Option<&Block>| -> WeightEdit {
        let factor = factors[rng.random_range(0..factors.len())];
        let block = match block {
            Some(b) => b.clone(),
            None => {
                // M's own pieces only: an attached view must not change the draw (the same seed gives
                // the same experiments whatever views a checker carries).
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
            // VPD attention subcomponents: one, 8 or all of one of the node's maps (a remainder node:
            // any of the layer's).
            Block::AttnSlices { layer, q, k, v, o, rest } => {
                let lists = [q, k, v, o];
                let total = |m: usize| weights.vpd_attention.get(&layer).map_or(0, |a| [&a.q, &a.k, &a.v, &a.o][m].0.nrows());
                let maps: Vec<usize> = (0..4).filter(|&m| if rest { total(m) > 0 } else { !lists[m].is_empty() }).collect();
                if maps.is_empty() {
                    WeightEdit::Head { layer, head: rng.random_range(0..weights.layers[layer].heads.len()), factor }
                } else {
                    let map = maps[rng.random_range(0..maps.len())];
                    let pool: Vec<usize> = if rest { (0..total(map)).collect() } else { lists[map].clone() };
                    let indices = match rng.random_range(0..3) {
                        0 => vec![pool[rng.random_range(0..pool.len())]],
                        1 => (0..8.min(pool.len())).map(|_| pool[rng.random_range(0..pool.len())]).collect::<BTreeSet<_>>().into_iter().collect(),
                        _ => pool,
                    };
                    WeightEdit::AttnSubcomponents { layer, map, indices, factor }
                }
            }
            // Transcoder features are not M's weights: an edit takes their layer's MLP.
            Block::Features { layer, .. } => WeightEdit::Neurons { layer, neurons: (0..weights.neurons(layer)).collect(), factor },
            // VPD subcomponents: one, a group of 8, or all of the node's c_fc or down_proj ones
            // (a remainder node: any of the layer's).
            Block::Slices { layer, fc, down, rest } => {
                let total = |d: bool| weights.vpd.get(&layer).map_or(0, |v| if d { v.down_u.nrows() } else { v.fc_u.nrows() });
                let d = if rest || (!fc.is_empty() && !down.is_empty()) { rng.random_bool(0.5) } else { fc.is_empty() };
                let pool: Vec<usize> = if rest { (0..total(d)).collect() } else if d { down } else { fc };
                if pool.is_empty() {
                    WeightEdit::Neurons { layer, neurons: (0..weights.neurons(layer)).collect(), factor }
                } else {
                    let indices = match rng.random_range(0..3) {
                        0 => vec![pool[rng.random_range(0..pool.len())]],
                        1 => (0..8.min(pool.len())).map(|_| pool[rng.random_range(0..pool.len())]).collect::<BTreeSet<_>>().into_iter().collect(),
                        _ => pool,
                    };
                    WeightEdit::Subcomponents { layer, down: d, indices, factor }
                }
            }
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
    let pieces: Vec<&Block> = targets.pieces.iter().take(TARGETED).map(|(b, _)| b).collect();
    let mut first_head = vec![0usize; layers + 1];
    for (l, layer) in weights.layers.iter().enumerate() {
        first_head[l + 1] = first_head[l] + layer.heads.len();
    }
    // The strongest pieces with a shared site of their own: one head, a whole MLP.
    let swappable: Vec<SharedSite> = pieces
        .iter()
        .filter_map(|b| match b {
            Block::Heads { layer, heads } if heads.len() == 1 => Some(SharedSite::Head(first_head[*layer] + heads[0])),
            Block::Neurons { layer, neurons } if neurons.len() == weights.neurons(*layer) => Some(SharedSite::Mlp(*layer)),
            _ => None,
        })
        .collect();
    let cuts: Vec<SiteOp> = targets.cuts.iter().take(TARGETED).map(|(c, _)| *c).collect();
    let mut targeted_kinds = Vec::new();
    if !pieces.is_empty() {
        targeted_kinds.push("edit");
    }
    if !swappable.is_empty() {
        targeted_kinds.push("swap");
    }
    if !cuts.is_empty() {
        targeted_kinds.push("cut");
    }
    let uniform_kinds: &[&str] = if sites.is_empty() { &["edit_uniform", "rank_one"] } else { &["edit_uniform", "rank_one", "sites"] };
    let (mut uniform, mut targeted) = (StdRng::seed_from_u64(seed), StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15));
    let every = |site: SharedSite, operation: Operation, family: interchange::Family| SiteDraw { family, ops: vec![SiteOp { site, operation, onward: true }], position: 0, length: 1 };
    for k in 0..count {
        let take_targeted = k % 2 == 1 && !targeted_kinds.is_empty();
        let rng = if take_targeted { &mut targeted } else { &mut uniform };
        let kind = if take_targeted { targeted_kinds[rng.random_range(0..targeted_kinds.len())] } else { uniform_kinds[rng.random_range(0..uniform_kinds.len())] };
        out.push(match kind {
            "sites" => Experiment::Sites { draw: sites[rng.random_range(0..sites.len())].clone() },
            "edit_uniform" => Experiment::Edit { edit: random_edit(rng, None), targeted: false },
            "edit" => {
                let factor = factors[rng.random_range(0..factors.len())];
                let edit = match pieces[rng.random_range(0..pieces.len())].clone() {
                    Block::Heads { layer, heads } => WeightEdit::Head { layer, head: heads[0], factor },
                    Block::Slices { layer, fc, down, .. } if fc.is_empty() => WeightEdit::Subcomponents { layer, down: true, indices: down, factor },
                    Block::Slices { layer, fc, .. } => WeightEdit::Subcomponents { layer, down: false, indices: fc, factor },
                    Block::Neurons { layer, neurons } => WeightEdit::Neurons { layer, neurons, factor },
                    other => random_edit(rng, Some(&other)),
                };
                Experiment::Edit { edit, targeted: true }
            }
            "swap" => Experiment::Sites { draw: every(swappable[rng.random_range(0..swappable.len())], Operation::Swap, interchange::Family::Swap) },
            "cut" => {
                let c = cuts[rng.random_range(0..cuts.len())];
                Experiment::Sites { draw: every(c.site, c.operation, interchange::Family::Cut) }
            }
            // "rank_one"
            _ => {
                let layer = rng.random_range(0..layers);
                let heads = weights.layers[layer].heads.len();
                let (head, matrix) = if rng.random_bool(0.5) || weights.neurons(layer) == 0 {
                    (Some(rng.random_range(0..heads)), [Matrix::Query, Matrix::Key, Matrix::Value, Matrix::Output][rng.random_range(0..4)])
                } else {
                    (None, [Matrix::Gate, Matrix::Down][rng.random_range(0..2)])
                };
                let probe = weights.clone_matrix(layer, head, matrix);
                // u vᵀ of Frobenius norm half the matrix's, unit random directions.
                let size = 0.5 * probe.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
                let mut unit = |n: usize| -> Vec<f64> {
                    let v: Vec<f64> = (0..n).map(|_| rng.random::<f64>() - 0.5).collect();
                    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
                    v.into_iter().map(|x| x / norm).collect()
                };
                let (rows, cols) = probe.dim();
                let u: Vec<f64> = unit(rows).into_iter().map(|x| x * size).collect();
                let v = unit(cols);
                Experiment::Edit { edit: WeightEdit::RankOne { layer, head, matrix, u, v }, targeted: false }
            }
        });
    }
    out
}

impl Weights {
    fn clone_matrix(&self, layer: usize, head: Option<usize>, matrix: Matrix) -> Stored {
        let lw = &self.layers[layer];
        match (head, matrix) {
            (Some(h), Matrix::Query) => lw.heads[h].query.clone(),
            (Some(h), Matrix::Key) => Array2::clone(&lw.heads[h].key),
            (Some(h), Matrix::Value) => Array2::clone(&lw.heads[h].value),
            (Some(h), _) => lw.heads[h].output.clone(),
            (None, Matrix::Gate) => lw.mlp.as_ref().map(|m| m.gate.clone()).unwrap_or_default(),
            (None, _) => lw.mlp.as_ref().map(|m| m.out.clone()).unwrap_or_default(),
        }
    }
}

/// What a score's runs need ([`Checker::measure_runs`]): the runs (program, experiment, `M`'s cache
/// key), the runs grouped by weight edit, each program parsed with its circuit, the reader's candidate
/// count and `M`'s clean outcome.
#[derive(Clone, Copy)]
struct Plan<'a> {
    runs: &'a [(usize, Experiment, String)],
    groups: &'a BTreeMap<Option<String>, Vec<usize>>,
    parsed: &'a [(Graph, bool, Option<String>)],
    circuits: &'a [Circuit],
    top: usize,
    clean: Option<&'a Array2<f64>>,
}

// ------------------------------------------------------------------------------ site operations

/// The mutable state of a run (`execute_with`): the actual stream and the stand-in stream entering
/// the current site, `embed`'s actual write, every unit's actual write (`None` while it writes its
/// stand-in), and per writer (units, then `embed`) the per-row factor interventions scaled its
/// stand-in by (`None`: 1 at every row).
struct Streams {
    stream: Array2<f64>,
    standin_stream: Array2<f64>,
    embed: Array2<f64>,
    writes: Vec<Option<Array2<f64>>>,
    factors: Vec<Option<Array1<f64>>>,
}

impl Streams {
    fn slot(&self, w: Writer) -> usize {
        match w {
            Writer::Embed => self.writes.len(),
            Writer::Unit(u) => u,
        }
    }

    /// Writer `w`'s stand-in write at every row (rows × width).
    fn standin(&self, w: Writer, embed_standin: &Array2<f64>, standins: &[Array2<f64>]) -> Array2<f64> {
        let s = match w {
            Writer::Embed => embed_standin,
            Writer::Unit(u) => &standins[u],
        };
        let mut out = s.clone();
        if let Some(f) = &self.factors[self.slot(w)] {
            out *= &f.view().insert_axis(Axis(1));
        }
        out
    }

    /// Writer `w`'s actual write minus its stand-in write, `None` for a unit writing its stand-in.
    fn delta(&self, w: Writer, embed_standin: &Array2<f64>, standins: &[Array2<f64>]) -> Option<Array2<f64>> {
        let actual = match w {
            Writer::Embed => &self.embed,
            Writer::Unit(u) => self.writes[u].as_ref()?,
        };
        Some(actual - &self.standin(w, embed_standin, standins))
    }

    /// Writer `w`'s actual write, `None` for a unit writing its stand-in.
    fn actual_mut(&mut self, w: Writer) -> Option<&mut Array2<f64>> {
        match w {
            Writer::Embed => Some(&mut self.embed),
            Writer::Unit(u) => self.writes[u].as_mut(),
        }
    }
}

/// What happens after a site ([`Interventions`]).
#[derive(Clone, Debug)]
pub(crate) enum After {
    /// The writers' writes, actual and stand-in, times the factor at the rows.
    Scale(Vec<Writer>, Vec<usize>, f64),
    /// The writers' actual writes at the rows replaced by their writes on the donor (stand-ins do
    /// not depend on the prompt).
    Swap(Vec<Writer>, Vec<usize>),
    /// The vector added to the stream at the rows, the actual and the stand-in stream alike, so
    /// every later reader of either model receives it.
    Push(Vec<usize>, Array1<f64>),
}

/// What happens to every normed input of a site's units ([`Interventions`]).
#[derive(Clone, Debug)]
pub(crate) enum OnInput {
    Scale(Vec<usize>, f64),
    Push(Vec<usize>, Array1<f64>),
    /// The normed input at the rows replaced by the same unit's on the donor.
    Swap(Vec<usize>),
}

/// The same circuit's run on the donor sequences: `embed`'s write, every unit's write, and the
/// normed inputs at the sites an input swap names.
#[derive(Clone, Debug)]
pub struct Donor {
    pub(crate) embed: Array2<f64>,
    pub(crate) writes: Vec<Option<Array2<f64>>>,
    pub(crate) normed: BTreeMap<(usize, usize), Array2<f64>>,
    pub(crate) reads: BTreeMap<usize, Array2<f64>>,
}

/// Row interventions of one run: site operations (`interchange::SiteOp`) resolved against a
/// circuit's units and a batch's rows ([`Interventions::resolve`]), applied identically to `M`
/// and the program by [`execute_with`]. A writer site's operation acts on the pieces' writes,
/// actual and stand-in alike (a stand-in is the piece applied to its average input, so its output
/// scales and moves with the piece's); an operation on the stream after block `b` acts on every
/// writer up to `b`; an operation on a block's input acts on each of its units' normed inputs; a cut
/// gives the readers of block `to` the writers' values on the donor.
#[derive(Clone, Debug, Default)]
pub struct Interventions {
    pub(crate) after: Vec<(Option<usize>, After)>,
    pub(crate) inputs: Vec<(usize, OnInput)>,
    /// Readers at a site: the writers' writes on the donor in place of their actual writes at the
    /// rows, through the routes that read the writers' actual writes.
    pub(crate) cuts: Vec<(usize, Vec<Writer>, Vec<usize>)>,
    pub(crate) donor: Option<Donor>,
    /// Sites whose units' normed inputs a run keeps (`Execution::normed`).
    pub(crate) record: BTreeSet<usize>,
    /// Operations on a head of a VPD-view attention, where no unit is the head alone: they act on
    /// the head's read `z_h` before `o_proj` (site, head in its layer, operation, rows), which for a
    /// native head is the same as acting on its write.
    pub(crate) head_reads: Vec<(usize, usize, HeadRead, Vec<usize>)>,
    /// Sites whose VPD-view attention units' reads a run keeps (`Execution::reads`).
    pub(crate) record_reads: BTreeSet<usize>,
}

/// What a site operation does to a head's read in a VPD-view attention ([`Interventions`]).
#[derive(Clone, Copy, Debug)]
pub(crate) enum HeadRead {
    Scale(f64),
    /// The read at the rows replaced by the same unit's on the donor.
    Swap,
}

impl Interventions {
    /// The operations after `point` (a site, `None` before every site, on `embed`).
    fn after(&self, point: Option<usize>, st: &mut Streams, embed_standin: &Array2<f64>, standins: &[Array2<f64>]) -> Result<(), String> {
        for (_, op) in self.after.iter().filter(|(p, _)| *p == point) {
            match op {
                After::Scale(writers, rows, f) => {
                    for &w in writers {
                        let standin = st.standin(w, embed_standin, standins);
                        let slot = st.slot(w);
                        let n = st.stream.nrows();
                        let factors = st.factors[slot].get_or_insert_with(|| Array1::ones(n));
                        for &r in rows {
                            factors[r] *= f;
                        }
                        for &r in rows {
                            let contribution = match st.actual_mut(w) {
                                Some(a) => {
                                    let old = a.row(r).to_owned();
                                    a.row_mut(r).mapv_inplace(|v| v * f);
                                    old
                                }
                                None => standin.row(r).to_owned(),
                            };
                            st.stream.row_mut(r).scaled_add(f - 1.0, &contribution);
                            st.standin_stream.row_mut(r).scaled_add(f - 1.0, &standin.row(r));
                        }
                    }
                }
                After::Swap(writers, rows) => {
                    let donor = self.donor.as_ref().ok_or("a swap without a donor run")?;
                    for &w in writers {
                        let value = match w {
                            Writer::Embed => Some(&donor.embed),
                            Writer::Unit(u) => donor.writes.get(u).and_then(Option::as_ref),
                        };
                        let (Some(value), Some(actual)) = (value.cloned(), st.actual_mut(w)) else { continue };
                        let mut change = Array2::<f64>::zeros(actual.dim());
                        for &r in rows {
                            change.row_mut(r).assign(&(&value.row(r) - &actual.row(r)));
                            actual.row_mut(r).assign(&value.row(r));
                        }
                        st.stream += &change;
                    }
                }
                After::Push(rows, v) => {
                    for &r in rows {
                        st.stream.row_mut(r).scaled_add(1.0, v);
                        st.standin_stream.row_mut(r).scaled_add(1.0, v);
                    }
                }
            }
        }
        Ok(())
    }

    /// Unit `unit`'s route inputs at `site` (one per route of `routes`) under the cuts into it: a
    /// writer it reads actually (an edge kept) is read on the donor at the cut rows.
    fn cut_inputs(&self, site: usize, unit: &Unit, routes: &[Route], inputs: &mut [Array2<f64>], st: &Streams) -> Result<(), String> {
        for (_, writers, rows) in self.cuts.iter().filter(|(to, _, _)| *to == site) {
            let donor = self.donor.as_ref().ok_or("a cut without a donor run")?;
            for &w in writers {
                let (actual, value) = match w {
                    Writer::Embed => (Some(&st.embed), Some(&donor.embed)),
                    Writer::Unit(u) => (st.writes[u].as_ref(), donor.writes.get(u).and_then(Option::as_ref)),
                };
                let (Some(actual), Some(value)) = (actual, value) else { continue };
                for (x, route) in inputs.iter_mut().zip(routes) {
                    let reads = match &unit.routes[route.slot()] {
                        Incoming::AllBut(cut) => !cut.contains(&w),
                        Incoming::Only(kept) => kept.contains(&w),
                    };
                    if reads {
                        for &r in rows {
                            x.row_mut(r).scaled_add(1.0, &(&value.row(r) - &actual.row(r)));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Unit `u`'s normed input on route slot `slot` at `site` under the input operations there,
    /// kept in `kept` when the site is recorded.
    fn normed(&self, site: usize, u: usize, slot: usize, x: &mut Array2<f64>, kept: &mut BTreeMap<(usize, usize), Array2<f64>>) {
        for (_, op) in self.inputs.iter().filter(|(s, _)| *s == site) {
            match op {
                OnInput::Scale(rows, f) => rows.iter().for_each(|&r| x.row_mut(r).mapv_inplace(|v| v * f)),
                OnInput::Push(rows, v) => rows.iter().for_each(|&r| x.row_mut(r).scaled_add(1.0, v)),
                OnInput::Swap(rows) => {
                    if let Some(value) = self.donor.as_ref().and_then(|d| d.normed.get(&(u, slot))) {
                        rows.iter().for_each(|&r| x.row_mut(r).assign(&value.row(r)));
                    }
                }
            }
        }
        if self.record.contains(&site) {
            kept.insert((u, slot), x.clone());
        }
    }

    /// Unit `u`'s heads' reads `z` at `site` under the head operations there (a head's columns
    /// scaled, or taken from the same unit's reads on the donor), kept in `kept` when the site is
    /// recorded.
    pub(crate) fn head_reads_of(&self, site: usize, u: usize, layer: &LayerWeights, z: &mut Array2<f64>, kept: &mut BTreeMap<usize, Array2<f64>>) {
        for (_, h, read, rows) in self.head_reads.iter().filter(|(s, ..)| *s == site) {
            let from: usize = layer.heads.iter().take(*h).map(|w| w.value.nrows()).sum();
            let cols = from..from + layer.heads.get(*h).map_or(0, |w| w.value.nrows());
            match read {
                HeadRead::Scale(f) => rows.iter().for_each(|&r| z.slice_mut(s![r, cols.clone()]).mapv_inplace(|v| v * f)),
                HeadRead::Swap => {
                    if let Some(donor) = self.donor.as_ref().and_then(|d| d.reads.get(&u)) {
                        rows.iter().for_each(|&r| z.slice_mut(s![r, cols.clone()]).assign(&donor.slice(s![r, cols.clone()])));
                    }
                }
            }
        }
        if self.record_reads.contains(&site) {
            kept.insert(u, z.clone());
        }
    }

    /// The counterfactual run with a VPD-view attention's head scalings applied to its recorded
    /// reads, so undeclared units' stand-ins scale with the head as `M`'s run does.
    pub(crate) fn scaled_reference(&self, reference: &Reference) -> Option<Reference> {
        let scalings: Vec<&(usize, usize, HeadRead, Vec<usize>)> = self.head_reads.iter().filter(|(_, _, r, _)| matches!(r, HeadRead::Scale(_))).collect();
        if scalings.is_empty() {
            return None;
        }
        // A new identity: the device keeps uploaded copies of a run's arrays by it.
        let mut out = Reference { id: next_reference_id(), ..reference.clone() };
        for (site, h, read, rows) in scalings {
            if let (HeadRead::Scale(f), Some(z)) = (read, out.reads.get_mut(site / 2).and_then(|l| l.get_mut(*h))) {
                let n = z.nrows();
                rows.iter().filter(|&&r| r < n).for_each(|&r| z.row_mut(r).mapv_inplace(|v| v * f));
            }
        }
        Some(out)
    }

    /// Whether the operations read a donor run (swaps and cuts).
    pub fn needs_donor(draw: &SiteDraw) -> bool {
        draw.ops.iter().any(|o| matches!(o.operation, Operation::Swap | Operation::Cut { .. }))
    }

    /// The sites whose normed inputs the donor run must keep (input swaps).
    pub fn donor_record(draw: &SiteDraw, weights: &Weights) -> BTreeSet<usize> {
        draw.ops.iter().filter(|o| o.operation == Operation::Swap).filter_map(|o| match o.site {
            SharedSite::Input(b) if b < 2 * weights.layers.len() => Some(b),
            _ => None,
        }).collect()
    }

    /// A donor-run's interventions: none, its normed inputs at `record` kept.
    pub fn recording(record: BTreeSet<usize>) -> Self {
        Self { record, ..Self::default() }
    }

    /// `draw`'s operations on `circuit` (split by [`Circuit::split_heads`] at the heads it names)
    /// over `batch`'s rows, with the donor run `donor` when it swaps or cuts.
    pub fn resolve(draw: &SiteDraw, circuit: &Circuit, weights: &Weights, batch: &Batch, units: &SiteUnits, donor: Option<&Execution>, donor_batch: Option<&Batch>) -> Result<Self, String> {
        let blocks = 2 * weights.layers.len();
        let mut out = Self::default();
        if let (Some(run), Some(b)) = (donor, donor_batch) {
            let tokens: Vec<usize> = b.tokens.iter().map(|t| *t as usize).collect();
            out.donor = Some(Donor { embed: wide(weights.embedding.select(Axis(0), &tokens).view()), writes: run.writes.clone(), normed: run.normed.clone(), reads: run.reads.clone() });
        }
        let heads_at = |pred: &dyn Fn(&Block) -> bool| -> Vec<Writer> { circuit.units.iter().enumerate().filter(|(_, u)| pred(&u.block)).map(|(i, _)| Writer::Unit(i)).collect() };
        for op in &draw.ops {
            let rows = op_rows(batch, draw.position, draw.length, op.onward);
            let (point, writers): (Option<usize>, Vec<Writer>) = match op.site {
                SharedSite::Head(h) => {
                    let (l, hh) = head_of(weights, h)?;
                    (Some(2 * l), heads_at(&|b| matches!(b, Block::Heads { layer, heads } if *layer == l && heads.as_slice() == [hh])))
                }
                // Every unit writing at the site, whatever its view.
                SharedSite::Attention(l) => (Some(2 * l), heads_at(&|b| b.site() == 2 * l && b.writes_residual())),
                SharedSite::Mlp(l) => (Some(2 * l + 1), heads_at(&|b| b.site() == 2 * l + 1 && b.writes_residual())),
                SharedSite::Embedding => (None, vec![Writer::Embed]),
                SharedSite::Stream(b) => (Some(b), [Writer::Embed].into_iter().chain(heads_at(&|x| x.site() <= b)).collect()),
                SharedSite::Input(_) => (None, Vec::new()),
            };
            if let SharedSite::Head(h) = op.site
                && writers.len() != 1
            {
                let (l, hh) = head_of(weights, h)?;
                let vpd_view = circuit.units.iter().any(|u| matches!(u.block, Block::AttnSlices { layer, .. } if layer == l));
                if !vpd_view {
                    return Err(format!("head {h} is not its own unit (split the circuit first)"));
                }
                let read = match op.operation {
                    Operation::Scale(i) => HeadRead::Scale(interchange::SCALES.get(i).copied().ok_or_else(|| format!("no scale {i}"))?),
                    Operation::Swap => HeadRead::Swap,
                    other => return Err(format!("{other:?} on head {h} of a VPD-view attention")),
                };
                if matches!(read, HeadRead::Swap) && out.donor.is_none() {
                    return Err("swaps and cuts need a donor run".into());
                }
                out.head_reads.push((2 * l, hh, read, rows));
                continue;
            }
            if point.is_some_and(|p| p >= blocks) {
                return Err(format!("{:?} past the last block", op.site));
            }
            let push = |direction: usize, size: usize| -> Result<Array1<f64>, String> {
                let v = units.directions.get(direction).ok_or_else(|| format!("no pushed direction {direction}"))?;
                let typical = units.typical.get(&op.site).ok_or_else(|| format!("no typical norm of {:?}", op.site))?;
                let scale = interchange::SIZES.get(size).ok_or("no such push size")? * typical;
                if v.len() != weights.width() {
                    return Err("a pushed direction of another width".into());
                }
                Ok(Array1::from_iter(v.iter().map(|x| x * scale)))
            };
            let factor = |i: usize| interchange::SCALES.get(i).copied().ok_or_else(|| format!("no scale {i}"));
            match (op.site, op.operation) {
                (SharedSite::Input(b), operation) => {
                    if b >= blocks {
                        return Err(format!("input {b} past the last block"));
                    }
                    out.inputs.push((
                        b,
                        match operation {
                            Operation::Scale(i) => OnInput::Scale(rows, factor(i)?),
                            Operation::Push { direction, size } => OnInput::Push(rows, push(direction, size)?),
                            Operation::Swap => OnInput::Swap(rows),
                            Operation::Cut { .. } => return Err("a cut from a block's input".into()),
                        },
                    ));
                }
                (_, Operation::Scale(i)) => out.after.push((point, After::Scale(writers, rows, factor(i)?))),
                (_, Operation::Push { direction, size }) => out.after.push((point, After::Push(rows, push(direction, size)?))),
                (_, Operation::Swap) => out.after.push((point, After::Swap(writers, rows))),
                (_, Operation::Cut { to }) => {
                    if to >= blocks || point.is_some_and(|p| p >= to) {
                        return Err(format!("a cut from {:?} into block {to}", op.site));
                    }
                    out.cuts.push((to, writers, rows));
                }
            }
        }
        if out.donor.is_none() && (!out.cuts.is_empty() || out.after.iter().any(|(_, a)| matches!(a, After::Swap(..))) || out.inputs.iter().any(|(_, i)| matches!(i, OnInput::Swap(_)))) {
            return Err("swaps and cuts need a donor run".into());
        }
        Ok(out)
    }
}

/// `circuit` on `base` (its batch and scored rows) under site operations `draw`: the circuit split
/// at the heads the operations name, the donor run of the same circuit on `donor` when they swap
/// or cut, the operations resolved ([`Interventions::resolve`]) and the run's log-probabilities.
pub fn run_sites(weights: &Weights, circuit: &Circuit, base: (&Batch, &[usize]), donor: Option<&Batch>, draw: &SiteDraw, units: &SiteUnits) -> Result<Array2<f64>, String> {
    let heads: BTreeSet<(usize, usize)> = draw.ops.iter().filter_map(|o| if let SharedSite::Head(h) = o.site { head_of(weights, h).ok() } else { None }).collect();
    let circuit = circuit.split_heads(&heads);
    let donor_run = match donor {
        Some(b) if Interventions::needs_donor(draw) => {
            let record_reads = draw.ops.iter().filter(|o| o.operation == Operation::Swap).filter_map(|o| if let SharedSite::Head(h) = o.site { head_of(weights, h).ok().map(|(l, _)| 2 * l) } else { None }).collect();
            let recording = Interventions { record_reads, ..Interventions::recording(Interventions::donor_record(draw, weights)) };
            Some(execute_with(weights, &circuit, b, &[], &BTreeMap::new(), &recording)?)
        }
        _ => None,
    };
    let ops = Interventions::resolve(draw, &circuit, weights, base.0, units, donor_run.as_ref(), donor)?;
    let scaled = base.0.reference.as_deref().and_then(|r| ops.scaled_reference(r));
    let mut batch = base.0.clone();
    if let Some(r) = scaled {
        batch.reference = Some(std::sync::Arc::new(r));
    }
    Ok(execute_with(weights, &circuit, &batch, base.1, &BTreeMap::new(), &ops)?.log_probabilities)
}

/// Head `h` numbered layer by layer (`SharedSite::Head`) as (layer, head).
fn head_of(weights: &Weights, h: usize) -> Result<(usize, usize), String> {
    let mut rest = h;
    for (l, layer) in weights.layers.iter().enumerate() {
        if rest < layer.heads.len() {
            return Ok((l, rest));
        }
        rest -= layer.heads.len();
    }
    Err(format!("no head {h}"))
}

/// The rows an operation drawn at `position` for sequences of `length` tokens acts on in each of
/// `batch`'s sequences: the position mapped onto a sequence of `n` tokens as
/// `1 + (position − 1)(n − 1) / (length − 1)` (position 0, the attention sink, stays 0), alone or
/// with every row after it (`onward`).
pub fn op_rows(batch: &Batch, position: usize, length: usize, onward: bool) -> Vec<usize> {
    let mut out = Vec::new();
    for &(start, n) in &batch.spans {
        let p = if position == 0 || n == 1 || length <= 1 { 0 } else { (1 + (position - 1) * (n - 1) / (length - 1)).min(n - 1) };
        if onward {
            out.extend(start + p..start + n);
        } else {
            out.push(start + p);
        }
    }
    out
}

impl Circuit {
    /// The circuit with each of `heads` (layer, head) its own unit, computing and routed as its unit
    /// was (a unit's heads compute independently of each other, so no output changes): the units a
    /// head's site operation acts on. Split units keep their index for the first part; the rest
    /// are appended, and every route reading the unit reads all of its parts.
    pub fn split_heads(&self, heads: &BTreeSet<(usize, usize)>) -> Circuit {
        let mut out = self.clone();
        let mut parts: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for u in 0..self.units.len() {
            let Block::Heads { layer, heads: hs } = &self.units[u].block else { continue };
            let singled: Vec<usize> = hs.iter().copied().filter(|h| heads.contains(&(*layer, *h))).collect();
            if singled.is_empty() || hs.len() == 1 {
                continue;
            }
            let rest: Vec<usize> = hs.iter().copied().filter(|h| !singled.contains(h)).collect();
            let mut groups: Vec<Vec<usize>> = singled.into_iter().map(|h| vec![h]).collect();
            if !rest.is_empty() {
                groups.push(rest);
            }
            out.units[u].block = Block::Heads { layer: *layer, heads: groups[0].clone() };
            for g in &groups[1..] {
                parts.entry(u).or_default().push(out.units.len());
                let mut unit = self.units[u].clone();
                unit.block = Block::Heads { layer: *layer, heads: g.clone() };
                out.units.push(unit);
            }
        }
        let widen = |incoming: &mut Incoming| {
            let set = match incoming {
                Incoming::Only(s) | Incoming::AllBut(s) => s,
            };
            for (u, more) in &parts {
                if set.contains(&Writer::Unit(*u)) {
                    set.extend(more.iter().map(|&m| Writer::Unit(m)));
                }
            }
        };
        for unit in &mut out.units {
            unit.routes.iter_mut().for_each(widen);
        }
        widen(&mut out.logits);
        out
    }
}

/// One drawn site-operation experiment: its family, operations and position, drawn for sequences
/// of `length` tokens ([`op_rows`] maps the position onto each prompt).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SiteDraw {
    pub family: interchange::Family,
    pub ops: Vec<SiteOp>,
    pub position: usize,
    pub length: usize,
}

/// What site operations are drawn from and measured in: each site's typical norm (the unit of a
/// push, `Interchange::measure_typical`), the pushed directions (`interchange::seeded_directions`),
/// and the pool of drawn experiments the uniform half takes from (a manifest's).
#[derive(Clone, Debug, Default)]
pub struct SiteUnits {
    pub typical: BTreeMap<SharedSite, f64>,
    pub directions: Vec<Vec<f64>>,
    pub pool: Vec<SiteDraw>,
}

impl SiteUnits {
    /// Every shared site of `weights`' model in ascending order (`SharedSite`'s order), and each
    /// head's block (`interchange::draw_site_ops`'s arguments).
    pub fn shared(weights: &Weights) -> (Vec<SharedSite>, Vec<usize>) {
        let layers = weights.layers.len();
        let head_blocks: Vec<usize> = weights.layers.iter().enumerate().flat_map(|(l, layer)| std::iter::repeat_n(2 * l, layer.heads.len())).collect();
        let mut sites: Vec<SharedSite> = (0..2 * layers).map(SharedSite::Stream).chain((0..head_blocks.len()).map(SharedSite::Head)).collect();
        sites.extend((0..layers).map(SharedSite::Attention).chain((0..layers).map(SharedSite::Mlp)).chain((0..2 * layers).map(SharedSite::Input)));
        sites.push(SharedSite::Embedding);
        (sites, head_blocks)
    }

    /// Each shared site's typical norm on `M` (unedited `weights`) over `sequences`: the root mean
    /// square of its rows' norms after each sequence's first token, as
    /// `Interchange::measure_typical` measures it (heads' and attentions' outputs and MLPs' outputs
    /// as their writes into the stream, a block's input as its normed input, the stream after a block,
    /// the embeddings).
    pub fn measure_typical(weights: &Weights, sequences: &[Vec<u32>]) -> Result<BTreeMap<SharedSite, f64>, String> {
        let batch = Batch::new(sequences)?;
        let all: BTreeSet<(usize, usize)> = weights.layers.iter().enumerate().flat_map(|(l, layer)| (0..layer.heads.len()).map(move |h| (l, h))).collect();
        let circuit = Graph::empty().model(weights).split_heads(&all);
        let blocks = 2 * weights.layers.len();
        let run = execute_with(weights, &circuit, &batch, &[], &BTreeMap::new(), &Interventions::recording((0..blocks).collect()))?;
        let later: Vec<usize> = batch.spans.iter().flat_map(|&(start, n)| start + 1..start + n).collect();
        let typical = |x: &Array2<f64>| -> f64 { (later.iter().map(|&r| x.row(r).dot(&x.row(r))).sum::<f64>() / later.len().max(1) as f64).sqrt() };
        let mut out = BTreeMap::new();
        let mut stream = wide(weights.embedding.select(Axis(0), &batch.tokens.iter().map(|t| *t as usize).collect::<Vec<_>>()).view());
        out.insert(SharedSite::Embedding, typical(&stream));
        let (_, head_blocks) = Self::shared(weights);
        for b in 0..blocks {
            let at: Vec<usize> = (0..circuit.units.len()).filter(|&u| circuit.units[u].block.site() == b).collect();
            let first = at.first().ok_or("a block without units")?;
            out.insert(SharedSite::Input(b), typical(run.normed.get(&(*first, 0)).ok_or("an unrecorded input")?));
            let mut total = Array2::<f64>::zeros(stream.dim());
            for &u in &at {
                let w = run.writes[u].as_ref().ok_or("a unit that did not compute")?;
                total += w;
                if let Block::Heads { layer, heads } = &circuit.units[u].block
                    && let [h] = heads.as_slice()
                {
                    let index = head_blocks.iter().take_while(|&&hb| hb < 2 * layer).count() + h;
                    out.insert(SharedSite::Head(index), typical(w));
                }
            }
            out.insert(if b % 2 == 0 { SharedSite::Attention(b / 2) } else { SharedSite::Mlp(b / 2) }, typical(&total));
            stream += &total;
            out.insert(SharedSite::Stream(b), typical(&stream));
        }
        Ok(out)
    }

    /// An immutable manifest for a model without one, drawn by interchange's code: typical norms
    /// measured on `sequences` ([`SiteUnits::measure_typical`]), `interchange::DIRECTIONS` seeded
    /// directions, and `count` experiments, each of a family uniform in `families`
    /// (`interchange::draw_site_ops` over every shared site on sequences of `length` tokens). The
    /// file has the keys of `mpd_library_mdl_2951`'s `Manifest` (one batch of experiments) plus
    /// `context` (`length`), and is refused if it exists.
    pub fn write_manifest(path: &std::path::Path, export: &str, weights: &Weights, sequences: &[Vec<u32>], families: &[interchange::Family], count: usize, seed: u64, length: usize) -> Result<Self, String> {
        if path.exists() {
            return Err(format!("{} exists and is immutable", path.display()));
        }
        let typical = Self::measure_typical(weights, sequences)?;
        let directions = interchange::seeded_directions(interchange::DIRECTIONS, weights.width(), seed);
        let (sites, head_blocks) = Self::shared(weights);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut experiments = Vec::with_capacity(count);
        let mut pool = Vec::with_capacity(count);
        let blocks = 2 * weights.layers.len();
        for _ in 0..count {
            let family = families[rng.random_range(0..families.len())];
            let (patch, position) = interchange::draw_site_ops(&mut rng, family, length, &sites, &head_blocks, &typical, directions.len(), blocks)?;
            if let interchange::Patch::Ops { family, ops } = &patch {
                pool.push(SiteDraw { family: *family, ops: ops.clone(), position, length });
            }
            experiments.push(interchange::Experiment { base: 0, source: 1, explained: vec![true; blocks], patch: Some(patch), position });
        }
        let manifest = serde_json::json!({
            "export": export,
            "sequences": [0, sequences.len()],
            "rows": format!("{} sequences of {} tokens", sequences.len(), sequences.first().map_or(0, Vec::len)),
            "seed": seed,
            "families": families,
            "edits_per_sequence": count,
            "batch_sequences": sequences.len(),
            "binary": option_env!("GIT_HASH"),
            "directions": "seeded",
            "push": directions,
            "typical": typical.iter().collect::<Vec<_>>(),
            "experiments": [experiments],
            "context": length,
        });
        std::fs::write(path, serde_json::to_vec(&manifest).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self { typical, directions, pool })
    }

    /// The experiments of an immutable experiment manifest (`mpd_library_mdl_2951`'s `Manifest`,
    /// e.g. `~/mpd-data/compare/manifest/MANIFEST_vpd4l_s1.json`) drawn on sequences of `length`
    /// tokens of a model of width `width`: its typical norms, its pushed directions (stored, or
    /// seeded from its seed as `Interchange::set_directions` draws them), and every experiment of
    /// operations on shared sites in its order.
    pub fn manifest(path: &std::path::Path, length: usize, width: usize) -> Result<Self, String> {
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?).map_err(|e| format!("{}: {e}", path.display()))?;
        let typical: Vec<(SharedSite, f64)> = serde_json::from_value(value["typical"].clone()).map_err(|e| format!("manifest typical: {e}"))?;
        let stored: Vec<Vec<f64>> = serde_json::from_value(value.get("push").cloned().unwrap_or(serde_json::Value::Array(Vec::new()))).map_err(|e| format!("manifest push: {e}"))?;
        let seed = value["seed"].as_u64().ok_or("manifest seed")?;
        let directions = if stored.is_empty() { interchange::seeded_directions(interchange::DIRECTIONS, width, seed) } else { stored };
        let batches: Vec<Vec<interchange::Experiment>> = serde_json::from_value(value["experiments"].clone()).map_err(|e| format!("manifest experiments: {e}"))?;
        let length = value.get("context").and_then(serde_json::Value::as_u64).map_or(length, |c| c as usize);
        if length == 0 {
            return Err(format!("{}: the manifest states no context and none was given", path.display()));
        }
        let pool = batches
            .into_iter()
            .flatten()
            .filter_map(|e| match e.patch {
                Some(interchange::Patch::Ops { family, ops }) => Some(SiteDraw { family, ops, position: e.position, length }),
                _ => None,
            })
            .collect();
        Ok(Self { typical: typical.into_iter().collect(), directions, pool })
    }
}

/// One scored experiment: the experiment, per scored token `KL(M_e ‖ P_e)` in bits, and the
/// reader's candidates when asked for.
#[derive(Clone, Debug)]
pub struct Measured(pub Experiment, pub Vec<f64>, pub Option<Candidates>);

/// Per scored token of an experiment (in [`Checker::rows_of`] order): `M`'s `k` most probable
/// clean tokens at that (prompt, position), their clean probabilities, `M_e`'s and `P_e`'s
/// probabilities of them, and each distribution's rest (what reader_score.py reads).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Candidates {
    pub tokens: Vec<Vec<usize>>,
    pub clean: Vec<Vec<f64>>,
    pub model: Vec<Vec<f64>>,
    pub program: Vec<Vec<f64>>,
    pub clean_other: Vec<f64>,
    pub model_other: Vec<f64>,
    pub program_other: Vec<f64>,
}

impl Candidates {
    /// From `M`'s clean log-probabilities `clean` (rows `clean_rows`), `M_e`'s `m` and `P_e`'s `p`
    /// (rows `rows`).
    fn of(clean: &Array2<f64>, m: &Array2<f64>, p: &Array2<f64>, rows: &[(usize, usize)], clean_rows: &[(usize, usize)], k: usize) -> Self {
        let mut out = Self::default();
        let rest = |values: &[f64]| (1.0 - values.iter().sum::<f64>()).max(0.0);
        for (r, at) in rows.iter().enumerate().take(m.nrows().min(p.nrows())) {
            let c = clean_rows.iter().position(|t| t == at).unwrap_or(r).min(clean.nrows().saturating_sub(1));
            let row = clean.row(c);
            let mut order: Vec<usize> = (0..row.len()).collect();
            order.sort_by(|a, b| row[*b].total_cmp(&row[*a]));
            order.truncate(k);
            let clean_p: Vec<f64> = order.iter().map(|&t| row[t].exp()).collect();
            let model_p: Vec<f64> = order.iter().map(|&t| m[[r, t]].exp()).collect();
            let program_p: Vec<f64> = order.iter().map(|&t| p[[r, t]].exp()).collect();
            out.clean_other.push(rest(&clean_p));
            out.model_other.push(rest(&model_p));
            out.program_other.push(rest(&program_p));
            out.tokens.push(order);
            out.clean.push(clean_p);
            out.model.push(model_p);
            out.program.push(program_p);
        }
        out
    }
}

/// The byte budget of `M`'s cached outcomes a checker starts with (`Checker::cache_bytes`): 2 GiB.
const CACHE_BYTES: usize = 2 << 30;

/// Site operations in words (the reader's description; the same for `M` and every program).
fn describe_sites(draw: &SiteDraw) -> String {
    let block = |b: usize| format!("layer {}'s {}", b / 2, if b % 2 == 0 { "attention" } else { "MLP" });
    let site = |s: &SharedSite| match s {
        SharedSite::Head(h) => format!("the output of head {h} (heads numbered layer by layer)"),
        SharedSite::Attention(l) => format!("the output of layer {l}'s attention (all its heads)"),
        SharedSite::Mlp(l) => format!("the output of L[{l}].mlp"),
        SharedSite::Embedding => "the token embeddings".to_string(),
        SharedSite::Stream(b) => format!("the residual stream after {}", block(*b)),
        SharedSite::Input(b) => format!("the normalized input of {}", block(*b)),
    };
    let ops: Vec<String> = draw
        .ops
        .iter()
        .map(|o| match o.operation {
            Operation::Scale(i) if interchange::SCALES[i] == 0.0 => format!("{} set to zero", site(&o.site)),
            Operation::Scale(i) => format!("{} times {}", site(&o.site), interchange::SCALES[i]),
            Operation::Push { size, .. } => format!("{} plus a fixed random direction of {} times its typical size", site(&o.site), interchange::SIZES[size]),
            Operation::Swap => format!("{} replaced by its value on the counterfactual text", site(&o.site)),
            Operation::Cut { to } => format!("{} reads {} as computed on the counterfactual text", block(to), site(&o.site)),
        })
        .collect();
    let onward = draw.ops.first().is_some_and(|o| o.onward);
    let rows = match (draw.position, onward) {
        (0, true) => "at every token".to_string(),
        (p, true) => format!("from the token {:.0}% of the way into the text on", 100.0 * p as f64 / draw.length.max(1) as f64),
        (p, false) => format!("at the token {:.0}% of the way into the text", 100.0 * p as f64 / draw.length.max(1) as f64),
    };
    format!("{}: {}", rows, ops.join("; "))
}

// ------------------------------------------------------------------------------ behaviors

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Counterfactual {
    #[serde(default)]
    pub text: String,
    pub token_ids: Vec<u32>,
    /// Per target position the counterfactual's answer token (the behavior's, [`Metric::Answer`]).
    #[serde(default)]
    pub answer_ids: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prompt {
    #[serde(default)]
    pub text: String,
    pub token_ids: Vec<u32>,
    pub target_positions: Vec<usize>,
    /// Per target position the prompt's answer token (the behavior's, [`Metric::Answer`]).
    #[serde(default)]
    pub answer_ids: Vec<u32>,
    #[serde(default)]
    pub counterfactual: Option<Counterfactual>,
    /// Token ranges whose attention is masked, `[query_start, query_end, key_start, key_end]`:
    /// queries in the first range do not attend to keys in the second (the prompt's and its
    /// counterfactual's, for `M` and every program, in every experiment).
    #[serde(default)]
    pub attention_block: Vec<[usize; 4]>,
}

impl Prompt {
    /// Its counterfactual's tokens where that is as long as the prompt (stand-ins read the
    /// partner's run position by position), else none.
    fn partner(&self) -> Vec<u32> {
        self.counterfactual.as_ref().filter(|c| c.token_ids.len() == self.token_ids.len()).map(|c| c.token_ids.clone()).unwrap_or_default()
    }
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

/// A counterfactual run computed once by whichever run asks first.
type RunCell = Arc<std::sync::OnceLock<Result<Arc<Reference>, String>>>;

/// The bytes a computed run holds (none while uncomputed or failed).
fn run_bytes(cell: &RunCell) -> usize {
    match cell.get() {
        Some(Ok(r)) => r.bytes(),
        _ => 0,
    }
}

/// The cell under `key` in `cache` (made when absent, then the newest), the oldest computed runs
/// dropped while the computed ones hold more than `budget` bytes.
fn cached_run(cache: &std::sync::Mutex<Vec<(String, RunCell)>>, key: String, budget: usize) -> Result<RunCell, String> {
    let mut cache = cache.lock().map_err(|e| e.to_string())?;
    if let Some(at) = cache.iter().position(|(k, _)| *k == key) {
        let entry = cache.remove(at);
        let cell = entry.1.clone();
        cache.push(entry);
        return Ok(cell);
    }
    let mut held: usize = cache.iter().map(|(_, c)| run_bytes(c)).sum();
    while held > budget && !cache.is_empty() {
        let (_, old) = cache.remove(0);
        held -= run_bytes(&old);
    }
    let cell: RunCell = Arc::new(std::sync::OnceLock::new());
    cache.push((key, cell.clone()));
    Ok(cell)
}

impl Reference {
    /// The bytes its tables hold.
    pub fn bytes(&self) -> usize {
        let tables = std::iter::once(&self.embed).chain(self.reads.iter().flatten()).chain(&self.active).chain(&self.mlp).chain(&self.inputs).chain(&self.attention_inputs);
        8 * tables.map(|a| a.len()).sum::<usize>()
    }
}

/// A behavior prepared for scoring: its prompts and counterfactuals as batches, the scored rows,
/// the swap donors, the stand-in averages, and `M`'s outcome per experiment (cached by its key).
pub struct Checker {
    pub weights: Weights,
    pub behavior: Behavior,
    clean: (Batch, Vec<usize>),
    counterfactual: Option<(Batch, Vec<usize>)>,
    /// Per prompt with a same-length donor: (prompt, donor).
    donors: Vec<(usize, usize)>,
    /// `M`'s outcome per experiment key (log-probabilities in float64: exact scores, an empty or full
    /// program's error to the last bits) and the keys in the order
    /// they were stored: the oldest are dropped past `cache_bytes`.
    cache: BTreeMap<String, Arc<Array2<f64>>>,
    cached: std::collections::VecDeque<String>,
    pub cache_bytes: usize,
    /// The directory of the disk cache of `M`'s outcomes, which every checker process given the
    /// same directory shares; `None` (the default) keeps them in memory only.
    pub disk_cache: Option<std::path::PathBuf>,
    /// A directory of small per-behavior memos shared across checker processes and runs: the
    /// targets (`Checker::targets`, hundreds of runs of `M`), named like the disk cache's files
    /// (behavior, prompts, weights, semantics).
    pub memo_dir: Option<std::path::PathBuf>,
    /// Heads by their measured removal effect on `M` (mean `KL(M ‖ M without the head)` at the
    /// targets), strongest first; measured on first use.
    targets: Option<Targets>,
    /// The units and pool of site operations (`SiteUnits::manifest`); empty draws none.
    pub sites: SiteUnits,
    /// When set to `m`, a score draws its experiments from seed `seed mod m`: `m` collections of
    /// experiments recur across the seeds a caller steps through (an RL run's steps), so `M`'s
    /// cached outcomes serve them all.
    pub uniform_seeds: Option<u64>,
    /// Each prompt's tokens with its counterfactual's and back: a sequence's stand-in source.
    partners: std::collections::HashMap<Vec<u32>, Vec<u32>>,
    /// Each prompt's and counterfactual's attention blocks by their tokens.
    blocks: std::collections::HashMap<Vec<u32>, Vec<[usize; 4]>>,
    /// The weight edit applied now, if any (its JSON): part of a counterfactual run's cache key.
    edit: Option<String>,
    /// Counterfactual runs by (edit, sequences), each computed once by whichever run asks first.
    pub(crate) references: std::sync::Mutex<Vec<(String, RunCell)>>,
    /// Counterfactual runs under site operations (`reference_under`) by experiment, for one
    /// [`Checker::score_batch`] (cleared at its start).
    site_references: std::sync::Mutex<Vec<(String, RunCell)>>,
    /// The bytes of counterfactual runs kept across scores (each cache), newest dropped first: an
    /// experiment set's first runs are computed once per behavior and serve every later score of it.
    pub reference_bytes: usize,
    /// Whether scores measure necessity (`Score::necessity_error_bits`; true by default): a search
    /// ranking candidates by sufficiency alone may skip its runs.
    pub necessity: bool,
    /// What every error compares per scored row ([`Metric`]; the whole distribution by default).
    pub metric: Metric,
    /// The model's shared base (generic machinery, design_v2 section 2): its nodes join every scored
    /// program ([`Checker::with_base`]), always on and priced apart (`Score::base_bits`).
    pub base: Option<Program>,
    /// [`Reference::zeros`] by row count, for deleting programs' runs.
    zeros: std::sync::Mutex<BTreeMap<usize, Arc<Reference>>>,
    /// `M`'s normed attention input per (layer, on the counterfactuals?), for attention claims
    /// ([`Checker::claim_error`]).
    claim_inputs: BTreeMap<(usize, bool), Arc<Array2<f64>>>,
}

/// One scored token of [`Checker::complement_tokens`]: its prompt and position, `M`'s top token on
/// the prompt and on the counterfactual, and both tokens' natural log-probabilities (clean top,
/// counterfactual top) under `M(x)`, `M(x')` and the complement run `M_c`.
#[derive(Clone, Debug, Serialize)]
pub struct ComplementToken {
    pub prompt: usize,
    pub position: usize,
    pub clean_top: usize,
    pub counterfactual_top: usize,
    pub clean: [f64; 2],
    pub counterfactual: [f64; 2],
    pub complement: [f64; 2],
}

/// Every score term (bits) and the counts behind them.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Score {
    pub total_bits: f64,
    pub exec_error_bits: f64,
    /// `N` times the mean, over the complement experiments the program's semantics measure
    /// ([`complements`], equal weights), of the per-token necessity error ([`Checker::necessity_runs`]):
    /// with counterfactual stand-ins, how far `M` with the program's nodes at their counterfactual
    /// values (every other piece on the prompt) stays from `M` on the counterfactual (collapse),
    /// at most the signal, which the empty program pays; deleting, `KL(M_c ‖ P_c)` of `M` with the
    /// nodes deleted against the program's prediction, zero for a program of no nodes outside its
    /// shared base.
    pub necessity_error_bits: f64,
    /// `N` times the program's attention-claim error ([`Checker::claim_error`]): zero for true
    /// claims and for a program that makes none.
    pub claim_error_bits: f64,
    /// `N` times the program's alignment error ([`Checker::alignment_error`]): zero where every
    /// interchange gives the algorithm's answer, and for a program that aligns nothing.
    pub alignment_error_bits: f64,
    pub reader_error_bits: f64,
    /// What a reader takes in (design_v2 section 2): `structure_bits + code_bits + explanation_bits`.
    pub complexity_bits: f64,
    /// The program's parts, nodes, labels and edges ([`Graph::structure`]), its shared base's apart.
    pub structure_bits: f64,
    /// The algorithm's tokens without comments (`Program::python_tokens` times `log2` of its token
    /// types).
    pub code_bits: f64,
    /// The English explanation's tokens times `log2` of their types (`Program::explanation_tokens`).
    pub explanation_bits: f64,
    /// Part names the program's nodes list ([`Graph::structure`]; a remainder counts its rank).
    pub parts: usize,
    pub python_tokens: usize,
    /// The weight numbers the declared nodes read (reported only: weights are not in the score).
    pub opaque_numbers: usize,
    /// The structure bits of the program's shared base nodes (`Program::base`) and the edges that
    /// touch them, not in `total_bits`: the caller charges a base once across behaviors.
    pub base_bits: f64,
    /// The definitions of the named groups the program uses ([`Graph::group_bits`]), not in
    /// `total_bits`: a model's library of groups is paid once.
    pub group_bits: f64,
    /// What undeclared pieces carried: "delete" or "counterfactual" ([`Graph::delete`]).
    pub standin: String,
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
        let batch = Batch::new(&sequences)?.with_partners(behavior.prompts.iter().map(Prompt::partner).collect());
        let targets: Vec<(usize, &[usize])> = behavior.prompts.iter().enumerate().map(|(i, p)| (i, p.target_positions.as_slice())).collect();
        let rows = scored_rows(&batch, &targets)?;
        // A counterfactual's targets: the prompt's, counted from the end of the sequence.
        let counterfactual = if behavior.prompts.iter().all(|p| p.counterfactual.is_some()) {
            let cf: Vec<Vec<u32>> = behavior.prompts.iter().map(|p| p.counterfactual.as_ref().map(|c| c.token_ids.clone()).unwrap_or_default()).collect();
            let b = Batch::new(&cf)?.with_partners(behavior.prompts.iter().zip(&cf).map(|(p, c)| if c.len() == p.token_ids.len() { p.token_ids.clone() } else { Vec::new() }).collect());
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
        let mut partners = std::collections::HashMap::new();
        let mut blocks = std::collections::HashMap::new();
        for p in behavior.prompts.iter().filter(|p| !p.attention_block.is_empty()) {
            blocks.insert(p.token_ids.clone(), p.attention_block.clone());
            if let Some(c) = &p.counterfactual {
                blocks.insert(c.token_ids.clone(), p.attention_block.clone());
            }
        }
        for p in &behavior.prompts {
            if let Some(c) = p.counterfactual.as_ref().filter(|c| c.token_ids.len() == p.token_ids.len()) {
                partners.insert(p.token_ids.clone(), c.token_ids.clone());
                partners.entry(c.token_ids.clone()).or_insert_with(|| p.token_ids.clone());
            }
        }
        Ok(Self {
            weights,
            behavior,
            clean: (batch, rows),
            counterfactual,
            donors,
            cache: BTreeMap::new(),
            cached: Default::default(),
            cache_bytes: CACHE_BYTES,
            disk_cache: None,
            memo_dir: None,
            targets: None,
            sites: SiteUnits::default(),
            uniform_seeds: None,
            partners,
            blocks,
            edit: None,
            references: std::sync::Mutex::new(Vec::new()),
            site_references: std::sync::Mutex::new(Vec::new()),
            reference_bytes: 3 << 30,
            necessity: true,
            metric: Metric::Full,
            base: None,
            zeros: std::sync::Mutex::new(BTreeMap::new()),
            claim_inputs: BTreeMap::new(),
        })
    }

    /// `program` with the checker's shared base ([`Checker::base`]) added: the base's nodes, less
    /// the parts the program names itself (a part belongs to one node; a base node left empty is
    /// dropped), listed in `Program::base`, with the base's declared edges among the nodes kept. A
    /// program that reuses a base node's id is marked invalid.
    pub fn with_base(&self, program: &Program) -> Program {
        let mut out = program.clone();
        if !out.valid {
            return out;
        }
        // Base parts are priced outside the total, so a program may list as its own base only parts
        // of the model's shared base (else it could move every part there for free).
        if let Some(e) = self.foreign_base_part(program) {
            out.valid = false;
            out.error = Some(e);
            return out;
        }
        let Some(base) = &self.base else { return out };
        if let Some(n) = base.nodes.iter().find(|n| program.nodes.iter().any(|m| m.id == n.id)) {
            out.valid = false;
            out.error = Some(format!("node id {} is the shared base's", n.id));
            return out;
        }
        // Each (view, layer, kind) the program names, with its units ("rest" as None), or every unit.
        let mut named: BTreeMap<(String, usize, String), Option<BTreeSet<Option<usize>>>> = BTreeMap::new();
        for p in program.nodes.iter().flat_map(|n| &n.pieces) {
            let key = (p.view.clone(), p.layer, p.kind.clone());
            let units: Option<Vec<Option<usize>>> = match &p.index {
                None => None,
                Some(Index::One(i)) => Some(vec![Some(*i)]),
                Some(Index::Many(v)) => Some(v.iter().map(|&i| Some(i)).collect()),
                Some(Index::Name(_)) => Some(vec![None]),
            };
            let entry = named.entry(key).or_insert_with(|| Some(BTreeSet::new()));
            match units {
                None => *entry = None,
                Some(units) => {
                    if let Some(set) = entry.as_mut() {
                        set.extend(units);
                    }
                }
            }
        }
        let mut kept = Vec::new();
        for node in &base.nodes {
            let mut pieces = Vec::new();
            for p in &node.pieces {
                let piece = match (named.get(&(p.view.clone(), p.layer, p.kind.clone())), &p.index) {
                    (None, _) => Some(p.clone()),
                    (Some(None), _) => None,
                    (Some(Some(set)), Some(Index::One(i))) => (!set.contains(&Some(*i))).then(|| p.clone()),
                    (Some(Some(set)), Some(Index::Many(v))) => {
                        let left: Vec<usize> = v.iter().copied().filter(|&i| !set.contains(&Some(i))).collect();
                        (!left.is_empty()).then(|| PieceIr { index: Some(Index::Many(left)), ..p.clone() })
                    }
                    (Some(Some(set)), Some(Index::Name(_))) => (!set.contains(&None)).then(|| p.clone()),
                    // A base piece of every unit the program also names: the program's parts move.
                    (Some(Some(_)), None) => None,
                };
                pieces.extend(piece);
            }
            if !pieces.is_empty() {
                kept.push(node.id.clone());
                out.nodes.push(NodeIr { id: node.id.clone(), pieces, claim: None, at: Vec::new() });
            }
        }
        let present = |id: &str| id == "embed" || id == "logits" || kept.iter().any(|k| k == id) || program.nodes.iter().any(|n| n.id == id);
        out.edges.extend(base.edges.iter().filter(|e| present(&e.from) && present(&e.to)).cloned());
        out.base.extend(kept);
        out
    }

    /// The first part a node of `program`'s own base (`Program::base`) holds outside the model's
    /// shared base ([`Checker::base`]; with none, every part is outside), as an error.
    fn foreign_base_part(&self, program: &Program) -> Option<String> {
        // A piece's units ("rest" as None), or None for every unit of its kind in the layer.
        let units = |p: &PieceIr| -> Option<Vec<Option<usize>>> {
            match &p.index {
                None => None,
                Some(Index::One(i)) => Some(vec![Some(*i)]),
                Some(Index::Many(v)) => Some(v.iter().map(|&i| Some(i)).collect()),
                Some(Index::Name(_)) => Some(vec![None]),
            }
        };
        let mut library: BTreeMap<(String, usize, String), Option<BTreeSet<Option<usize>>>> = BTreeMap::new();
        for p in self.base.iter().flat_map(|b| &b.nodes).flat_map(|n| &n.pieces) {
            let entry = library.entry((p.view.clone(), p.layer, p.kind.clone())).or_insert_with(|| Some(BTreeSet::new()));
            match (units(p), entry.as_mut()) {
                (None, _) => *entry = None,
                (Some(u), Some(set)) => set.extend(u),
                (Some(_), None) => {}
            }
        }
        for node in program.nodes.iter().filter(|n| program.base.contains(&n.id)) {
            for p in &node.pieces {
                let inside = match (library.get(&(p.view.clone(), p.layer, p.kind.clone())), units(p)) {
                    (None, _) => false,
                    (Some(None), _) => true,
                    (Some(Some(_)), None) => false,
                    (Some(Some(set)), Some(u)) => u.iter().all(|x| set.contains(x)),
                };
                if !inside {
                    return Some(format!("base node {}: {} {} layer {} holds parts outside the model's shared base", node.id, p.view, p.kind, p.layer));
                }
            }
        }
        None
    }

    /// Sets (or with `None` clears) the shared base, checked by parsing it alone; returns its
    /// structure bits (`Score::base_bits` of the empty program) and its part count.
    pub fn set_base(&mut self, base: Option<Program>) -> Result<(f64, usize), String> {
        let previous = std::mem::replace(&mut self.base, base);
        match self.empty_graph() {
            Ok(g) => {
                let token = token_bits(self.base.as_ref().map_or(0, |b| b.token_types));
                let (parts, _, bits) = g.structure(&self.weights, &g.base, token);
                Ok((bits, parts))
            }
            Err(e) => {
                self.base = previous;
                Err(e)
            }
        }
    }

    /// The empty program's graph: the shared base alone when the checker has one.
    fn empty_graph(&self) -> Result<Graph, String> {
        if self.base.is_none() {
            return Ok(Graph { delete: self.weights.has_vpd(), ..Graph::empty() });
        }
        let empty = Program { model: self.behavior.model.clone(), valid: true, ..Program::default() };
        Graph::parse(&self.with_base(&empty), &self.weights).map_err(|e| format!("the shared base: {e}"))
    }

    /// The error of `graph`'s attention claims, bits per row: for each claimed node the mean, over
    /// the rows of the behavior's prompts and counterfactuals, of `KL(claimed ‖ produced)` (the
    /// pattern its parts produce on `M`'s own inputs, [`produced_patterns`], heads weighed as there),
    /// summed over nodes. A true claim costs nothing; the parts run as weights either way.
    pub fn claim_error(&mut self, graph: &Graph) -> Result<f64, String> {
        let mut total = 0.0;
        for (block, claim) in graph.blocks.iter().zip(&graph.claims) {
            let Some(claim) = claim else { continue };
            let layer = match block {
                Block::Heads { layer, .. } | Block::AttnSlices { layer, .. } => *layer,
                _ => return Err("an attention claim on a block without queries and keys".into()),
            };
            let (mut kl, mut rows) = (0.0, 0usize);
            for counterfactual in [false, true] {
                let Some(batch) = (if counterfactual { self.counterfactual.as_ref().map(|c| c.0.clone()) } else { Some(self.clean.0.clone()) }) else { continue };
                let x_hat = self.claim_input(layer, counterfactual, &batch)?;
                let mut masked = batch.clone();
                self.mask(&mut masked);
                let produced = produced_patterns(&self.weights, block, &x_hat, &batch.spans, &masked.blocks)?;
                let total_weight: f64 = produced.iter().map(|(w, _)| w).sum();
                for (n, &(start, length)) in batch.spans.iter().enumerate() {
                    let claimed = claim.pattern(&batch.tokens[start..start + length], n, counterfactual)?;
                    for t in 0..length {
                        let row: f64 = produced.iter().map(|(w, patterns)| {
                            let actual = patterns[n].row(t);
                            w * claimed.row(t).iter().zip(actual).filter(|(c, _)| **c > 0.0).map(|(c, a)| c * (c / a.max(f64::MIN_POSITIVE)).log2()).sum::<f64>()
                        }).sum();
                        kl += if total_weight > 0.0 { row / total_weight } else { 0.0 };
                    }
                    rows += length;
                }
            }
            total += kl / rows.max(1) as f64;
        }
        Ok(total)
    }

    /// `M`'s normed attention input at `layer` on `batch` (the prompts, or their counterfactuals),
    /// recorded once.
    fn claim_input(&mut self, layer: usize, counterfactual: bool, batch: &Batch) -> Result<Arc<Array2<f64>>, String> {
        if let Some(x) = self.claim_inputs.get(&(layer, counterfactual)) {
            return Ok(x.clone());
        }
        let circuit = Graph::empty().model(&self.weights);
        let unit = circuit.units.iter().position(|u| matches!(u.block, Block::Heads { layer: l, .. } if l == layer)).ok_or("a layer without heads")?;
        let mut masked = batch.clone();
        self.mask(&mut masked);
        let run = execute_with(&self.weights, &circuit, &masked, &[], &BTreeMap::new(), &Interventions::recording([2 * layer].into()))?;
        let x = Arc::new(run.normed.get(&(unit, 0)).cloned().ok_or("an unrecorded attention input")?);
        self.claim_inputs.insert((layer, counterfactual), x.clone());
        Ok(x)
    }

    /// Whether `graph`'s necessity is collapse toward the counterfactual: the behavior has
    /// counterfactuals, and the program takes counterfactual values or is scored on the answer.
    pub fn collapses(&self, graph: &Graph) -> bool {
        self.counterfactual.is_some() && (!graph.delete || self.metric != Metric::Full)
    }

    /// The error of `q` against `p` at the scored rows `rows` ((prompt, position), in the tables' row
    /// order) under [`Checker::metric`]: `KL(p ‖ q)`, or the `KL` over the behavior's answers (the
    /// prompt's and the counterfactual's at each target, else the token `by` puts first) and any
    /// other token.
    pub fn divergence(&self, p: &Array2<f64>, q: &Array2<f64>, by: &Array2<f64>, rows: &[(usize, usize)]) -> Vec<f64> {
        match self.metric {
            Metric::Full => kl_bits(p, q),
            Metric::Answer => answer_kl_bits(p, q, by, &self.answers_at(rows)),
            Metric::Choice => choice_kl_bits(p, q, &self.answers_at(rows)),
        }
    }

    /// Per scored row (prompt, position) the behavior's answers there: the prompt's and its
    /// counterfactual's answer tokens ([`Prompt::answer_ids`], else the token that follows the
    /// target in each sequence: a behavior's prompts continue with their answers).
    pub fn answers_at(&self, rows: &[(usize, usize)]) -> Vec<Vec<usize>> {
        let answer = |ids: &[u32], given: &[u32], k: usize, t: usize| given.get(k).or_else(|| ids.get(t + 1)).map(|&a| a as usize);
        rows.iter()
            .map(|&(i, t)| {
                let Some(p) = self.behavior.prompts.get(i) else { return Vec::new() };
                let Some(k) = p.target_positions.iter().position(|&x| x == t) else { return Vec::new() };
                let changed = p.counterfactual.as_ref().and_then(|c| answer(&c.token_ids, &c.answer_ids, k, t));
                answer(&p.token_ids, &p.answer_ids, k, t).into_iter().chain(changed).collect()
            })
            .collect()
    }

    /// The error of `graph`'s alignments, bits per target: for each alignment the mean, over its pairs'
    /// base targets, of `KL(M(source) ‖ M_swap)`, where `M_swap` is `M` on the base with the aligned
    /// nodes' writes taken from its run on the source (every other piece computing on the base,
    /// `M`'s circuit, [`Graph::model`]) and `M(source)` is `M`'s own distribution on the source at
    /// the same positions (the lead, 10-08 00:48: a variable's whole carrier moves the base's output
    /// to the source's). Each target's error is at most the signal `KL(M(source) ‖ M(base))`, what
    /// swapping no parts costs; summed over alignments.
    pub fn alignment_error(&mut self, graph: &Graph) -> Result<f64, String> {
        let mut total = 0.0;
        let circuit = graph.model(&self.weights);
        for (alignment, nodes) in graph.alignments.iter().filter(|(b, _)| !b.pairs.is_empty()) {
            let prompts = &self.behavior.prompts;
            let mut bases = Vec::with_capacity(alignment.pairs.len());
            let mut sources = Vec::with_capacity(alignment.pairs.len());
            for pair in &alignment.pairs {
                let Some(base) = prompts.get(pair.base) else {
                    return Err(format!("alignment {}: prompt {} outside the behavior", alignment.variable, pair.base));
                };
                let source = if pair.changed {
                    base.partner()
                } else {
                    prompts.get(pair.source).map(|p| p.token_ids.clone()).ok_or_else(|| format!("alignment {}: prompt {} outside the behavior", alignment.variable, pair.source))?
                };
                if base.token_ids.len() != source.len() {
                    let what = if pair.changed { "its changed prompt".to_string() } else { format!("prompt {}", pair.source) };
                    return Err(format!("alignment {}: prompt {} and {what} differ in length", alignment.variable, pair.base));
                }
                bases.push(base.token_ids.clone());
                sources.push(source);
            }
            let (mut base, mut source) = (Batch::new(&bases)?, Batch::new(&sources)?);
            self.mask(&mut base);
            self.mask(&mut source);
            // The base's targets, and the same positions in the source (prompts of one length).
            let (mut rows, mut source_rows) = (Vec::new(), Vec::new());
            for ((pair, &(start, _)), &(from, _)) in alignment.pairs.iter().zip(&base.spans).zip(&source.spans) {
                rows.extend(prompts[pair.base].target_positions.iter().map(|&t| start + t));
                source_rows.extend(prompts[pair.base].target_positions.iter().map(|&t| from + t));
            }
            let written = execute(&self.weights, &circuit, &source, &[], &BTreeMap::new())?.writes;
            let target = execute(&self.weights, &circuit, &source, &source_rows, &BTreeMap::new())?.log_probabilities;
            let unswapped = execute(&self.weights, &circuit, &base, &rows, &BTreeMap::new())?.log_probabilities;
            let swaps: BTreeMap<usize, Array2<f64>> = nodes.iter().map(|&u| written.get(u).cloned().flatten().map(|w| (u, w)).ok_or_else(|| format!("alignment {}: an aligned node wrote nothing", alignment.variable))).collect::<Result<_, _>>()?;
            let swapped = if swaps.is_empty() { unswapped.clone() } else { execute(&self.weights, &circuit, &base, &rows, &swaps)?.log_probabilities };
            let scored: Vec<(usize, usize)> = alignment.pairs.iter().flat_map(|pair| prompts[pair.base].target_positions.iter().map(move |&t| (pair.base, t))).collect();
            let (errors, signal) = (self.divergence(&target, &swapped, &target, &scored), self.divergence(&target, &unswapped, &target, &scored));
            let cost: f64 = errors.iter().zip(&signal).map(|(e, s)| e.min(*s)).sum();
            total += cost / rows.len().max(1) as f64;
        }
        Ok(total)
    }

    /// Attaches the behavior's attention blocks to every sequence of `b`. Every native batch goes
    /// through it (the program's runs and every stand-in run), so a run never drops the
    /// behavior's execution constraints.
    fn mask(&self, b: &mut Batch) {
        b.blocks = b.sequences().iter().map(|s| self.blocks.get(s).cloned().unwrap_or_default()).collect();
    }

    /// `batch` with its stand-in source attached: under counterfactual stand-ins, `M`'s run (with
    /// the current weights) on each sequence's partner (a prompt's counterfactual, a
    /// counterfactual's prompt). `M` itself (every unit computing, every edge kept) reads no
    /// stand-in and gets none.
    pub fn referenced(&self, circuit: &Circuit, batch: &Batch) -> Result<Batch, String> {
        let mut out = batch.clone();
        out.reference = None;
        self.mask(&mut out);
        if circuit.is_model() {
            return Ok(out);
        }
        if circuit.delete {
            let rows = out.tokens.len();
            let mut zeros = self.zeros.lock().map_err(|e| e.to_string())?;
            out.reference = Some(zeros.entry(rows).or_insert_with(|| Arc::new(Reference::zeros(&self.weights, rows))).clone());
            return Ok(out);
        }
        let partner = self.partners_of(batch)?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&partner, &mut hasher);
        let key = format!("{} {}", self.edit.as_deref().unwrap_or(""), std::hash::Hasher::finish(&hasher));
        let cell = cached_run(&self.references, key, self.reference_bytes)?;
        let r = cell
            .get_or_init(|| {
                let mut b = Batch::new(&partner)?;
                self.mask(&mut b);
                reference(&self.weights, &b).map(Arc::new)
            })
            .clone()?;
        out.reference = Some(r);
        Ok(out)
    }

    /// Each sequence's partner: the batch's own ([`Batch::partners`], by item), else by its tokens.
    fn partners_of(&self, batch: &Batch) -> Result<Vec<Vec<u32>>, String> {
        let missing = || "counterfactual stand-ins need each prompt's counterfactual of the same length".to_string();
        if !batch.partners.is_empty() {
            if batch.partners.len() != batch.spans.len() {
                return Err(format!("{} partners for {} sequences", batch.partners.len(), batch.spans.len()));
            }
            return batch.partners.iter().map(|p| if p.is_empty() { Err(missing()) } else { Ok(p.clone()) }).collect();
        }
        batch.sequences().iter().map(|s| self.partners.get(s).cloned().ok_or_else(missing)).collect()
    }


    /// Sets the weight edit applied now (`None` after restoring): part of a counterfactual run's
    /// key, so each edit's runs are kept apart.
    fn set_edit(&mut self, e: &Experiment) {
        self.edit = match e {
            Experiment::Edit { edit, .. } => serde_json::to_string(edit).ok(),
            _ => None,
        };
    }

    /// The experiment's key for `M`'s cache: what it does to which pieces.
    fn key(e: &Experiment) -> String {
        match e {
            Experiment::Edit { edit, .. } => format!("edit {}", serde_json::to_string(edit).unwrap_or_default()),
            other => serde_json::to_string(other).unwrap_or_default(),
        }
    }

    /// One circuit under one experiment: its log-probabilities at the scored rows (a site
    /// operation's base prompts only, when they read a same-length donor).
    fn outcome(&mut self, circuit: &Circuit, e: &Experiment) -> Result<Array2<f64>, String> {
        let restore = match e {
            Experiment::Edit { edit, .. } => Some(edit.apply(&mut self.weights)?),
            _ => None,
        };
        self.set_edit(e);
        let result = self.run(circuit, e);
        self.set_edit(&Experiment::Clean);
        if let Some(r) = restore {
            r.restore(&mut self.weights)?;
        }
        result
    }

    /// [`Checker::outcome`] on the current weights (an edit's already applied).
    fn run(&self, circuit: &Circuit, e: &Experiment) -> Result<Array2<f64>, String> {
        let circuit = circuit.clone();
        let (batch, rows) = match e {
            Experiment::Counterfactual => self.counterfactual.as_ref().ok_or("the behavior has no counterfactuals")?,
            _ => &self.clean,
        };
        {
            match e {
                Experiment::Sites { draw } => self.sites_outcome(&circuit, draw),
                _ => Ok(execute(&self.weights, &circuit, &self.referenced(&circuit, batch)?, rows, &BTreeMap::new())?.log_probabilities),
            }
        }
    }

    /// [`Checker::run`] of several circuits under one experiment as one stacked batch
    /// ([`execute_stacked`]): `None` where they do not stack, a site experiment among them.
    fn run_stacked(&self, circuits: &[&Circuit], e: &Experiment) -> Option<Result<Vec<Array2<f64>>, String>> {
        let (batch, rows) = match e {
            Experiment::Sites { draw } => {
                let (base, rows, donor) = match self.site_inputs(circuits.first()?, draw) {
                    Ok(inputs) => inputs,
                    Err(e) => return Some(Err(e)),
                };
                return stacked_sites(&self.weights, circuits, (&base, &rows), donor.as_ref(), draw, &self.sites);
            }
            Experiment::Counterfactual => self.counterfactual.as_ref()?,
            _ => &self.clean,
        };
        let first = circuits.first()?;
        let batch = match self.referenced(first, batch) {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };
        execute_stacked(&self.weights, circuits, &batch, rows)
    }

    /// The complement models of programs that stack ([`execute_complements_with`]) on the prompts,
    /// with the current weights (a complement experiment's edit applied): `None` where they do not
    /// stack.
    fn run_complements(&self, programs: &[(&Graph, &Circuit)]) -> Option<Result<Vec<Array2<f64>>, String>> {
        let (batch, rows) = &self.clean;
        let batch = match self.referenced(programs.first()?.1, batch) {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };
        let circuits: Vec<&Circuit> = programs.iter().map(|(_, c)| *c).collect();
        let routed: Vec<bool> = programs.iter().map(|(g, _)| g.edges.iter().any(|(w, r, _)| *w == Writer::Embed && r.is_none())).collect();
        execute_complements_with(&self.weights, (&circuits, &routed), (&batch, rows), &mut |c, job, copies| crate::graph_device::execute_copies(&self.weights, c, job, copies))
    }

    /// Stores `M`'s outcome under `key`, dropping the oldest past the byte budget.
    fn keep(&mut self, key: String, outcome: Arc<Array2<f64>>) {
        if self.cache.insert(key.clone(), outcome).is_none() {
            self.cached.push_back(key);
        }
        let mut bytes: usize = self.cache.values().map(|a| a.len() * 8).sum();
        while bytes > self.cache_bytes && self.cached.len() > 1 {
            let Some(old) = self.cached.pop_front() else { break };
            if let Some(a) = self.cache.remove(&old) {
                bytes -= a.len() * 8;
            }
        }
    }

    /// `M`'s outcome under `e`, cached.
    pub fn model_outcome(&mut self, graph: &Graph, e: &Experiment) -> Result<Array2<f64>, String> {
        let key = Self::key(e);
        if let Some(hit) = self.cache.get(&key) {
            return Ok((**hit).clone());
        }
        let out = match self.disk_get(&key) {
            Some(m) => m,
            None => {
                let circuit = graph.model(&self.weights);
                let m = self.outcome(&circuit, e)?;
                self.disk_put(&key, &m);
                m
            }
        };
        self.keep(key, Arc::new(out.clone()));
        Ok(out)
    }

    /// The file of `M`'s outcome under `key` in the disk cache (`Checker::disk_cache`, shared by
    /// every checker process given it): named by the behavior, a fingerprint of its prompts, attention
    /// blocks and of `M`'s weights, the semantics version ([`DISK_SEMANTICS`]) and the key; `None` when
    /// the cache is off.
    fn disk_path(&self, key: &str) -> Option<std::path::PathBuf> {
        let dir = self.disk_cache.as_ref()?;
        let (sub, h) = self.fingerprint(key);
        Some(dir.join(sub).join(format!("{h:016x}.f64")))
    }

    /// The memo file of `name` (`Checker::memo_dir`), named like the disk cache's files.
    fn memo_path(&self, name: &str) -> Option<std::path::PathBuf> {
        let dir = self.memo_dir.as_ref()?;
        let (sub, h) = self.fingerprint(name);
        Some(dir.join(sub).join(format!("{h:016x}.json")))
    }

    /// A memo of `name` written by an earlier run, when there is one.
    fn memo_get<T: serde::de::DeserializeOwned>(&self, name: &str) -> Option<T> {
        serde_json::from_slice(&std::fs::read(self.memo_path(name)?).ok()?).ok()
    }

    /// Writes the memo of `name` (whole, then renamed into place).
    fn memo_put<T: Serialize>(&self, name: &str, value: &T) {
        let Some(path) = self.memo_path(name) else { return };
        let (Some(dir), Ok(bytes)) = (path.parent(), serde_json::to_vec(value)) else { return };
        let tmp = path.with_extension(format!("{}.partial", std::process::id()));
        if std::fs::create_dir_all(dir).is_ok() && std::fs::write(&tmp, bytes).is_ok() && std::fs::rename(&tmp, &path).is_err()
            && let Err(e) = std::fs::remove_file(&tmp)
        {
            eprintln!("memo {}: {e}", tmp.display());
        }
    }

    /// The behavior's directory name and a hash of its prompts, attention blocks, `M`'s weights, the
    /// semantics version and `key`.
    fn fingerprint(&self, key: &str) -> (String, u64) {
        // FNV-1a over little-endian words.
        let fnv = |mut h: u64, words: &mut dyn Iterator<Item = u64>| -> u64 {
            for w in words {
                for b in w.to_le_bytes() {
                    h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
                }
            }
            h
        };
        let mut words: Vec<u64> = Vec::new();
        for p in &self.behavior.prompts {
            words.extend(p.token_ids.iter().map(|&t| u64::from(t)));
            words.extend(p.target_positions.iter().map(|&t| t as u64));
            if let Some(c) = &p.counterfactual {
                words.extend(c.token_ids.iter().map(|&t| u64::from(t)));
            }
            words.push(u64::MAX);
        }
        let prompts = fnv(0xcbf2_9ce4_8422_2325, &mut words.into_iter());
        let w = &self.weights;
        let mut sample: Vec<f64> = w.embedding.row(0).iter().map(|&v| f64::from(v)).collect();
        sample.extend(w.unembedding.row(w.unembedding.nrows() - 1).iter().map(|&v| f64::from(v)));
        sample.extend(w.final_norm.gain.iter());
        // A row of every matrix (attention's maps as well as the MLPs'), so two checkpoints that differ in
        // any block do not share outcomes; the behavior's attention blocks and the execution semantics'
        // version are part of the name too.
        for l in &w.layers {
            sample.extend(l.mlp.iter().flat_map(|m| m.out.row(0).to_vec()).map(f64::from));
            for h in &l.heads {
                for m in [&h.query, &*h.key, &*h.value, &h.output] {
                    sample.extend(m.row(0).iter().map(|&v| f64::from(v)));
                }
            }
        }
        let blocks = self.behavior.prompts.iter().flat_map(|p| p.attention_block.iter().flatten().map(|&b| b as u64).chain([u64::MAX - 1]));
        let h = fnv(prompts, &mut sample.iter().map(|v| v.to_bits()).chain(blocks).chain(std::iter::once(DISK_SEMANTICS)).chain(key.bytes().map(u64::from)));
        (format!("{}_{prompts:016x}", self.behavior.id), h)
    }

    /// `M`'s outcome under `key` from the disk cache, when it holds it.
    fn disk_get(&self, key: &str) -> Option<Array2<f64>> {
        let bytes = std::fs::read(self.disk_path(key)?).ok()?;
        let (rows, cols) = (u64::from_le_bytes(bytes.get(..8)?.try_into().ok()?) as usize, u64::from_le_bytes(bytes.get(8..16)?.try_into().ok()?) as usize);
        let values: Vec<f64> = bytes.get(16..)?.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap_or_default())).collect();
        Array2::from_shape_vec((rows, cols), values).ok()
    }

    /// Writes `M`'s outcome under `key` to the disk cache (written whole, then renamed into place).
    fn disk_put(&self, key: &str, m: &Array2<f64>) {
        let Some(path) = self.disk_path(key) else { return };
        let (Some(dir), true) = (path.parent(), m.is_standard_layout()) else { return };
        let mut bytes = Vec::with_capacity(16 + 8 * m.len());
        bytes.extend_from_slice(&(m.nrows() as u64).to_le_bytes());
        bytes.extend_from_slice(&(m.ncols() as u64).to_le_bytes());
        m.iter().for_each(|v| bytes.extend_from_slice(&v.to_le_bytes()));
        let partial = path.with_extension(format!("partial{}", std::process::id()));
        if std::fs::create_dir_all(dir).is_ok() && std::fs::write(&partial, bytes).is_ok() {
            if let Err(e) = std::fs::rename(&partial, &path) {
                log::warn!("graph disk cache: {} not stored: {e}", path.display());
            }
        }
    }

    /// The program's score under `count` sampled experiments (seed `seed`); `edges` routes by the
    /// declared edges, else every edge among the nodes is kept. `n` overrides the behavior's size.
    /// An invalid program is scored as the empty program, flagged. [`Checker::score_batch`] of one.
    pub fn score(&mut self, program: &Program, count: usize, seed: u64, edges: bool, n: Option<f64>) -> Result<(Score, Vec<Measured>), String> {
        let mut out = self.score_batch(std::slice::from_ref(program), count, seed, edges, n, 0)?;
        out.pop().ok_or_else(|| "no score".into())
    }

    /// Every program's score under its `count` sampled experiments (seed `seed`; the behavior's
    /// half is the same for all of them), with per experiment its tokens' `KL(M_e ‖ P_e)` and, for
    /// `top` > 0, the reader's candidates ([`Candidates`]). `M`'s outcomes are computed once per
    /// experiment and cached; runs go in parallel threads, experiments that edit weights grouped
    /// by their edit (the edit applied once, then every run that needs it).
    pub fn score_batch(&mut self, programs: &[Program], count: usize, seed: u64, edges: bool, n: Option<f64>, top: usize) -> Result<Vec<(Score, Vec<Measured>)>, String> {
        let merged: Vec<Program> = programs.iter().map(|p| self.with_base(p)).collect();
        let programs = merged.as_slice();
        let empty = self.empty_graph()?;
        let seed = self.uniform_seeds.map_or(seed, |m| seed % m.max(1));
        // Targets aim the drawn experiments; a score of none (the clean and counterfactual runs alone)
        // does not measure them.
        let targets = if count > 0 { self.targets()? } else { Targets::default() };
        let parsed: Vec<(Graph, bool, Option<String>)> = programs
            .iter()
            .map(|program| match Graph::parse(program, &self.weights) {
                Ok(g) => (g, true, None),
                Err(e) => (empty.clone(), false, Some(e)),
            })
            .collect();
        for (g, _, _) in &parsed {
            self.weights.load_features(g)?;
        }
        let circuits: Vec<Circuit> = parsed.iter().map(|(g, _, _)| g.program(&self.weights, edges)).collect();
        // One experiment set for every program (design.txt section 2), drawn from the seed alone;
        // site operations that read a donor are left out when the behavior has none.
        let donors = self.counterfactual_donors() || !self.donors.is_empty();
        let experiments: Vec<Experiment> = sample(&self.weights, self.counterfactual.is_some(), count, seed, &targets, &self.sites)
            .into_iter()
            .filter(|e| donors || !matches!(e, Experiment::Sites { draw } if Interventions::needs_donor(draw)))
            .collect();
        // N: the tokens the experiment set scores, so the errors are its total code length in bits.
        let n = n.unwrap_or_else(|| experiments.iter().map(|e| self.rows_of(e).len()).sum::<usize>().max(1) as f64);
        // Per run (program, experiment) the cache key of M's outcome.
        let mut runs: Vec<(usize, Experiment, String)> = Vec::new();
        let mut drawn = vec![0usize; programs.len()];
        for i in 0..parsed.len() {
            for e in &experiments {
                runs.push((i, e.clone(), Self::key(e)));
                drawn[i] += 1;
            }
        }
        // Runs grouped by the weight edit they apply (none first).
        let mut groups: BTreeMap<Option<String>, Vec<usize>> = BTreeMap::new();
        for (r, (_, e, _)) in runs.iter().enumerate() {
            let edit = match e {
                Experiment::Edit { edit, .. } => Some(serde_json::to_string(edit).map_err(|e| e.to_string())?),
                _ => None,
            };
            groups.entry(edit).or_default().push(r);
        }
        let mut measured: Vec<Option<(Vec<f64>, Option<Candidates>)>> = vec![None; runs.len()];
        let clean = if top > 0 { Some(self.model_outcome(&Graph::empty(), &Experiment::Clean)?) } else { None };
        let plan = Plan { runs: &runs, groups: &groups, parsed: &parsed, circuits: &circuits, top, clean: clean.as_ref() };
        self.measure_runs(plan, &mut measured)?;
        // Necessity, for each program with nodes: the same complement experiments for every program.
        let complements = complements(&experiments);
        // (Programs of no nodes too where necessity is collapse: their parts' collapse is the whole signal.)
        let named: Vec<usize> = (0..parsed.len()).filter(|&i| self.necessity && (self.collapses(&parsed[i].0) || parsed[i].0.blocks.len() > parsed[i].0.base.len())).collect();
        let necessity_kl = self.necessity_runs(&named.iter().map(|&i| (&parsed[i].0, &circuits[i])).collect::<Vec<_>>(), &complements)?;
        let mut necessity = vec![(0.0, BTreeMap::new()); parsed.len()];
        for (&i, kls) in named.iter().zip(&necessity_kl) {
            let mut families: BTreeMap<String, Family> = BTreeMap::new();
            let measured: Vec<(&Experiment, &Vec<f64>)> = complements.iter().zip(kls).filter_map(|(e, kl)| kl.as_ref().map(|kl| (e, kl))).collect();
            let mut mean = 0.0;
            for &(e, kl) in &measured {
                let entry = families.entry(format!("necessity_{}", e.family())).or_default();
                entry.experiments += 1;
                entry.tokens += kl.len();
                entry.mean_kl_bits += kl.iter().sum::<f64>();
                mean += kl.iter().sum::<f64>() / kl.len().max(1) as f64 / measured.len() as f64;
            }
            for v in families.values_mut() {
                v.mean_kl_bits /= v.tokens.max(1) as f64;
            }
            necessity[i] = (n * mean, families);
        }
        let mut out = Vec::with_capacity(programs.len());
        let mut runs_and_measures = runs.into_iter().zip(measured);
        for (i, program) in programs.iter().enumerate() {
            let (graph, valid, error) = &parsed[i];
            let mut per_family: BTreeMap<String, Family> = BTreeMap::new();
            let mut total = (0.0, 0usize);
            let mut outcomes = Vec::new();
            for ((_, e, _), m) in runs_and_measures.by_ref().take(drawn[i]) {
                let (kl, candidates) = m.ok_or("an unmeasured run")?;
                let sum: f64 = kl.iter().sum();
                let entry = per_family.entry(e.family().to_string()).or_default();
                entry.experiments += 1;
                entry.tokens += kl.len();
                entry.mean_kl_bits += sum;
                total.0 += sum;
                total.1 += kl.len();
                outcomes.push(Measured(e, kl, candidates));
            }
            for v in per_family.values_mut() {
                v.mean_kl_bits /= v.tokens.max(1) as f64;
            }
            let (necessity_error_bits, necessity_families) = std::mem::take(&mut necessity[i]);
            per_family.extend(necessity_families);
            let exec_error_bits = n * total.0 / total.1.max(1) as f64;
            let token = if *valid { token_bits(program.token_types) } else { 0.0 };
            let code_bits = program.python_tokens as f64 * token;
            let explanation_bits = if *valid && program.explanation_token_types > 1 { program.explanation_tokens as f64 * (program.explanation_token_types as f64).log2() } else { 0.0 };
            let opaque_numbers = graph.opaque_numbers(&self.weights);
            // A shared base library's nodes are priced apart, for the caller to charge once.
            let (parts, structure_bits, base_bits) = graph.structure(&self.weights, &graph.base, token);
            let complexity_bits = structure_bits + code_bits + explanation_bits;
            let claim_error_bits = n * self.claim_error(graph)?;
            let alignment_error_bits = n * self.alignment_error(graph)?;
            let score = Score {
                total_bits: exec_error_bits + necessity_error_bits + claim_error_bits + alignment_error_bits + complexity_bits,
                exec_error_bits,
                necessity_error_bits,
                claim_error_bits,
                alignment_error_bits,
                reader_error_bits: 0.0,
                complexity_bits,
                structure_bits,
                code_bits,
                explanation_bits,
                parts,
                python_tokens: if *valid { program.python_tokens } else { 0 },
                opaque_numbers,
                base_bits,
                group_bits: graph.group_bits(&self.weights),
                standin: if graph.delete { "delete" } else { "counterfactual" }.into(),
                n,
                experiments: outcomes.len(),
                valid: *valid,
                error: error.clone(),
                per_family,
            };
            out.push((score, outcomes));
        }
        Ok(out)
    }

    /// The necessity experiments' errors, per program per complement experiment (`None` where the
    /// program's semantics have no such experiment), per scored token, from `M_c`: `M` with the
    /// program's nodes taken out and every other piece computing on the prompt
    /// ([`Graph::complement_model`]).
    ///
    /// Where necessity is collapse ([`Checker::collapses`]), the mirror of sufficiency: taking out
    /// the named parts together (at their counterfactual values, or deleted for a deleting program
    /// scored on the answer) should make `M` behave as on the counterfactual. The error is
    /// `min(D(M(x'), M_c), D(M(x'), M(x)))` ([`Checker::divergence`]: the whole distribution, or the
    /// prompt's answer) with `M(x')` `M`'s own output on the counterfactual and `M(x)` on the prompt
    /// under the same experiment (weight edit or site operation): at most the signal, which the
    /// empty program pays (its `M_c` is `M`), about zero for a complete mechanism. Without
    /// counterfactuals, or scored on the whole distribution (deleting every part makes noise, not the
    /// counterfactual's distribution), deleting programs ([`Graph::delete`]; clean and weight edits)
    /// compare `M_c` with their prediction, `M`'s own run less the program's parts, nothing
    /// recomputed ([`Checker::without_parts`]).
    pub(crate) fn necessity_runs(&mut self, programs: &[(&Graph, &Circuit)], complements: &[Experiment]) -> Result<Vec<Vec<Option<Vec<f64>>>>, String> {
        let mut out = vec![Vec::with_capacity(complements.len()); programs.len()];
        if programs.is_empty() {
            return Ok(out);
        }
        let native = Graph::empty().model(&self.weights);
        for e in complements {
            let measured: Vec<bool> = programs.iter().map(|(g, _)| self.collapses(g) || (g.delete && !matches!(e, Experiment::Sites { .. }))).collect();
            self.set_edit(e);
            // Every run with the experiment's edit applied to M's weights.
            let restore = match e {
                Experiment::Edit { edit, .. } => Some(edit.apply(&mut self.weights)?),
                _ => None,
            };
            let result = (|| -> Result<Vec<Option<Vec<f64>>>, String> {
                // With counterfactual stand-ins: M on the counterfactuals and on the prompts under the
                // experiment, shared by every program.
                let collapse = programs.iter().zip(&measured).any(|((g, _), &m)| m && self.collapses(g));
                let (target, prompt) = if collapse { (Some(self.run_swapped(&native, e)?), Some(self.run(&native, e)?)) } else { (None, None) };
                // The complement models of counterfactual programs that stack, as one batch
                // (`run_complements`), site experiments aside.
                let mut complemented: Vec<Option<Array2<f64>>> = vec![None; programs.len()];
                if !matches!(e, Experiment::Sites { .. }) {
                    let rows = self.clean.0.tokens.len().max(1);
                    let plain: Vec<usize> = (0..programs.len()).filter(|&k| measured[k] && !programs[k].0.delete && programs[k].0.base.is_empty() && !programs[k].0.blocks.is_empty() && stacks(&self.weights, programs[k].1)).collect();
                    for group in plain.chunks((STACKED_ROWS / rows).max(1)).filter(|g| g.len() > 1) {
                        let pairs: Vec<(&Graph, &Circuit)> = group.iter().map(|&k| (programs[k].0, programs[k].1)).collect();
                        if let Some(out) = self.run_complements(&pairs) {
                            for (&k, p) in group.iter().zip(out?) {
                                complemented[k] = Some(p);
                            }
                        }
                    }
                }
                let mut kls = Vec::with_capacity(programs.len());
                for (k, ((g, _), &m)) in programs.iter().zip(&measured).enumerate() {
                    if !m {
                        kls.push(None);
                        continue;
                    }
                    if let (true, Some(target), Some(prompt)) = (self.collapses(g), &target, &prompt) {
                        // A program of no nodes outside its base takes nothing out: M_c is M.
                        let model = match complemented[k].take() {
                            Some(m) => m,
                            None if g.blocks.len() <= g.base.len() => prompt.clone(),
                            None => self.run(&g.complement_model(&self.weights), e)?,
                        };
                        let rows = self.rows_of(e);
                        kls.push(Some(self.divergence(target, &model, prompt, &rows).into_iter().zip(self.divergence(target, prompt, prompt, &rows)).map(|(c, s)| c.min(s)).collect()));
                        continue;
                    }
                    let model = self.run(&g.complement_model(&self.weights), e)?;
                    kls.push(Some(kl_bits(&model, &self.without_parts(g)?)));
                }
                Ok(kls)
            })();
            self.set_edit(&Experiment::Clean);
            let restored = restore.map(|r| r.restore(&mut self.weights)).transpose();
            for (k, kl) in result?.into_iter().enumerate() {
                out[k].push(kl);
            }
            restored?;
        }
        Ok(out)
    }

    /// `M` on the prompts less `graph`'s parts, nothing recomputed: every other piece's write in
    /// `M`'s own run (and `embed`'s, unless the program routes it to the logits) through the final
    /// norm, at the scored rows. A deleting program's prediction of `M` with its parts deleted.
    fn without_parts(&self, graph: &Graph) -> Result<Array2<f64>, String> {
        if graph.positions.iter().any(Option::is_some) {
            return Err("a program whose nodes act at positions needs the behavior's counterfactuals for necessity".into());
        }
        let (batch, rows) = &self.clean;
        let cell = cached_run(&self.references, format!("{} own", self.edit.as_deref().unwrap_or("")), self.reference_bytes)?;
        let r = cell
            .get_or_init(|| {
                let mut b = batch.clone();
                b.reference = None;
                self.mask(&mut b);
                reference(&self.weights, &b).map(Arc::new)
            })
            .clone()?;
        // The writes at the scored rows alone: nothing is recomputed, so no other row enters.
        let r = r.select_rows(rows);
        let routed = graph.edges.iter().any(|(w, reader, _)| *w == Writer::Embed && reader.is_none());
        let mut stream = if routed { Array2::zeros(r.embed.dim()) } else { r.embed.clone() };
        for unit in graph.complement_model(&self.weights).units.iter().filter(|u| u.computes) {
            stream += &r.write(&self.weights, &unit.block)?;
        }
        log_probabilities(&self.weights, &stream)
    }

    /// `circuit` under `e` with the roles of prompt and counterfactual swapped: run on the
    /// counterfactuals, its stand-ins from the prompts (`e`'s weight edit applied by the caller).
    fn run_swapped(&self, circuit: &Circuit, e: &Experiment) -> Result<Array2<f64>, String> {
        match e {
            Experiment::Sites { draw } => {
                let (base, rows, _) = self.site_inputs_on(circuit, draw, true)?;
                run_sites(&self.weights, circuit, (&base, &rows), None, draw, &self.sites)
            }
            _ => self.run(circuit, &Experiment::Counterfactual),
        }
    }

    /// The runs of a score: first `M` (each experiment once, its outcome cached), then per weight
    /// edit (applied once) the counterfactual runs every program reads and every program's runs of
    /// the edit's experiments, on parallel threads. Every run uses `M`'s weights: precision is not in
    /// the score (design_v2 section 2).
    fn measure_runs(&mut self, plan: Plan, measured: &mut [Option<(Vec<f64>, Option<Candidates>)>]) -> Result<(), String> {
        let Plan { runs, groups, parsed, circuits, top, clean } = plan;
        // Where the score's time goes, logged at its end: M's outcomes, the counterfactual runs and
        // the programs' runs.
        let started = std::time::Instant::now();
        let mut seconds = [0.0f64; 3];
        // M's outcomes come from its native circuit, the same for every program whatever its views.
        let native = Graph::empty().model(&self.weights);
        let stacking: Vec<bool> = circuits.iter().map(|c| stacks(&self.weights, c)).collect();
        // The program runs made in stacks (logged with the times).
        let stacked_runs = std::sync::atomic::AtomicUsize::new(0);
        let mut program_runs = 0usize;
        // M's outcomes of this score, held until it ends: the cache's byte budget may drop some
        // while later groups add theirs.
        let mut outcomes: BTreeMap<String, Arc<Array2<f64>>> = BTreeMap::new();
        let distinct = |members: &[usize]| -> Vec<usize> {
            let mut out: Vec<usize> = Vec::new();
            for &r in members {
                if !out.iter().any(|&m| runs[m].2 == runs[r].2) {
                    out.push(r);
                }
            }
            out
        };
        for members in groups.values() {
            let restore = match &runs[members[0]].1 {
                Experiment::Edit { edit, .. } => Some(edit.apply(&mut self.weights)?),
                _ => None,
            };
            let result = (|| -> Result<(), String> {
                let missing = distinct(members);
                let this = &*self;
                // Each outcome with whether it is new to the cache.
                let compute = |&r: &usize| -> Result<(String, Arc<Array2<f64>>, bool), String> {
                    let (_, e, key) = &runs[r];
                    if let Some(m) = this.cache.get(key) {
                        return Ok((key.clone(), m.clone(), false));
                    }
                    if let Some(m) = this.disk_get(key) {
                        return Ok((key.clone(), Arc::new(m), true));
                    }
                    let m = this.run(&native, e)?;
                    this.disk_put(key, &m);
                    Ok((key.clone(), Arc::new(m), true))
                };
                // One run alone goes on this thread, so its products spread over the pool (par_dot).
                let made: Vec<(String, Arc<Array2<f64>>, bool)> = if missing.len() == 1 { missing.iter().map(compute).collect::<Result<_, String>>()? } else { missing.par_iter().map(compute).collect::<Result<_, String>>()? };
                for (key, m, new) in made {
                    if new {
                        self.keep(key.clone(), m.clone());
                    }
                    outcomes.insert(key, m);
                }
                Ok(())
            })();
            if let Some(r) = restore {
                r.restore(&mut self.weights)?;
            }
            result?;
        }
        seconds[0] = started.elapsed().as_secs_f64();
        let mut made_bytes = 0usize;
        // Deleting programs read no counterfactual run (Reference::zeros), so a batch of them makes
        // none, and a behavior without counterfactuals scores them.
        let counterfactual = circuits.iter().any(|c| !c.delete && !c.is_model());
        // The programs' runs, per edit group in chunks of experiments: a chunk's counterfactual runs
        // are made first (on this thread: made inside parallel runs, a run waiting on one could be
        // stolen by the thread making it while that thread waits on the device's pool, and neither
        // returns), then every program's runs of the chunk in parallel. Counterfactual runs past
        // `reference_bytes` are dropped after each chunk (a Qwen3-0.6B run holds 2.8 GB), never
        // during one.
        for members in groups.values() {
            // The group's weight edit, applied to M's weights for its counterfactual runs and the
            // programs' runs alike (an experiment edits the program's own weights).
            let restore = match &runs[members[0]].1 {
                Experiment::Edit { edit, .. } => Some(edit.apply(&mut self.weights)?),
                _ => None,
            };
            self.set_edit(&runs[members[0]].1.clone());
            let budget = self.reference_bytes;
            let result = (|| -> Result<(), String> {
                let experiments = distinct(members);
                let mut at = 0;
                // The most bytes of new counterfactual runs one experiment has made: runs kept from an
                // earlier score make none, so chunks stay single until an experiment makes some.
                let mut per = 0;
                while at < experiments.len() {
                    let size = if per == 0 { 1 } else { (budget / per).clamp(1, 8) };
                    let chunk: Vec<usize> = experiments[at..(at + size).min(experiments.len())].to_vec();
                    self.reference_bytes = usize::MAX;
                    let before = self.held_references();
                    let prewarm = std::time::Instant::now();
                    // Deleting programs read no counterfactual run (their stand-ins are zero).
                    if parsed.iter().any(|(g, _, _)| !g.delete) {
                        if counterfactual {
                        chunk.iter().try_for_each(|&r| self.prewarm(&runs[r].1))?;
                    }
                    }
                    seconds[1] += prewarm.elapsed().as_secs_f64();
                    made_bytes += self.held_references().saturating_sub(before);
                    // Chunks as large as the budget holds.
                    per = per.max(self.held_references().saturating_sub(before) / chunk.len());
                    let keys: BTreeSet<&str> = chunk.iter().map(|&r| runs[r].2.as_str()).collect();
                    let mine: Vec<usize> = members.iter().copied().filter(|&r| keys.contains(runs[r].2.as_str())).collect();
                    let this = &*self;
                    let score_p = |&r: &usize| -> Result<(usize, Vec<f64>, Option<Candidates>), String> {
                        let (i, e, key) = &runs[r];
                        let m = outcomes.get(key).ok_or("M's outcome went missing")?;
                        let p = this.run(&circuits[*i], e)?;
                        let kl = this.divergence(m, &p, m, &this.rows_of(e));
                        let candidates = clean.map(|c| Candidates::of(c, m, &p, &this.rows_of(e), &this.rows_of(&Experiment::Clean), top));
                        Ok((r, kl, candidates))
                    };
                    // The programs that stack ([`stacks`]) run each experiment, site operations
                    // included, as one batch of copies (`run_stacked`, at most `STACKED_ROWS` rows),
                    // the others one by one.
                    let mut jobs: Vec<Vec<usize>> = Vec::new();
                    let mut stacked: BTreeMap<(&str, bool), Vec<usize>> = BTreeMap::new();
                    for &r in &mine {
                        if stacking[runs[r].0] {
                            stacked.entry((runs[r].2.as_str(), circuits[runs[r].0].delete)).or_default().push(r);
                        } else {
                            jobs.push(vec![r]);
                        }
                    }
                    let per_stack = (STACKED_ROWS / this.clean.0.tokens.len().max(1)).max(1);
                    // Each experiment's programs ordered by the first site where their units differ
                    // from the most common ones there, latest first: a stack's copies then share long
                    // prefixes, computed once (`execute_stacked`).
                    jobs.extend(stacked.into_values().flat_map(|mut rs| {
                        let signatures: Vec<BTreeMap<usize, String>> = rs.iter().map(|&r| site_signatures(&circuits[runs[r].0])).collect();
                        let sites: BTreeSet<usize> = signatures.iter().flat_map(|m| m.keys().copied()).collect();
                        let common: BTreeMap<usize, &str> = sites.iter().filter_map(|&site| {
                            let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
                            for m in &signatures {
                                *counts.entry(m.get(&site).map_or("", String::as_str)).or_default() += 1;
                            }
                            counts.into_iter().max_by_key(|&(_, n)| n).map(|(sig, _)| (site, sig))
                        }).collect();
                        let first: Vec<usize> = signatures.iter().map(|m| sites.iter().copied().find(|site| m.get(site).map_or("", String::as_str) != common.get(site).copied().unwrap_or("")).unwrap_or(usize::MAX)).collect();
                        let mut order: Vec<usize> = (0..rs.len()).collect();
                        order.sort_by_key(|&k| std::cmp::Reverse(first[k]));
                        rs = order.iter().map(|&k| rs[k]).collect();
                        rs.chunks(per_stack).map(<[usize]>::to_vec).collect::<Vec<_>>()
                    }));
                    let run_job = |job: &Vec<usize>| -> Result<Vec<(usize, Vec<f64>, Option<Candidates>)>, String> {
                        if job.len() > 1 {
                            let e = &runs[job[0]].1;
                            let group: Vec<&Circuit> = job.iter().map(|&r| &circuits[runs[r].0]).collect();
                            if let Some(out) = this.run_stacked(&group, e) {
                                stacked_runs.fetch_add(job.len(), std::sync::atomic::Ordering::Relaxed);
                                return out?
                                    .into_iter()
                                    .zip(job)
                                    .map(|(p, &r)| {
                                        let (_, e, key) = &runs[r];
                                        let m = outcomes.get(key).ok_or("M's outcome went missing")?;
                                        let candidates = clean.map(|c| Candidates::of(c, m, &p, &this.rows_of(e), &this.rows_of(&Experiment::Clean), top));
                                        Ok((r, this.divergence(m, &p, m, &this.rows_of(e)), candidates))
                                    })
                                    .collect();
                            }
                        }
                        job.iter().map(score_p).collect()
                    };
                    program_runs += mine.len();
                    let scored: Vec<(usize, Vec<f64>, Option<Candidates>)> = if jobs.len() == 1 { jobs.iter().map(run_job).collect::<Result<Vec<_>, _>>()? } else { jobs.par_iter().map(run_job).collect::<Result<Vec<_>, _>>()? }.into_iter().flatten().collect();
                    for (r, kl, candidates) in scored {
                        measured[r] = Some((kl, candidates));
                    }
                    self.reference_bytes = budget;
                    self.drop_references()?;
                    at += chunk.len();
                }
                Ok(())
            })();
            self.reference_bytes = budget;
            self.set_edit(&Experiment::Clean);
            let restored = restore.map(|r| r.restore(&mut self.weights)).transpose();
            result?;
            restored?;
        }
        seconds[2] = started.elapsed().as_secs_f64() - seconds[0] - seconds[1];
        log::info!("graph score of {} programs: M's outcomes {:.1} s, counterfactual runs {:.1} s ({:.1} GB made), program runs {:.1} s ({} of {program_runs} stacked)", parsed.len(), seconds[0], seconds[1], made_bytes as f64 / 1e9, seconds[2], stacked_runs.load(std::sync::atomic::Ordering::Relaxed));
        Ok(())
    }

    /// The bytes of the counterfactual runs both caches hold.
    fn held_references(&self) -> usize {
        [&self.references, &self.site_references].iter().map(|c| c.lock().map_or(0, |c| c.iter().map(|(_, cell)| run_bytes(cell)).sum::<usize>())).sum()
    }

    /// Drops the newest counterfactual runs while each cache holds more than `reference_bytes`.
    /// A score walks its experiments in one order every time, so dropping the newest keeps the
    /// runs its first chunks read on the next score of the behavior; dropping the oldest would
    /// keep the last chunks' and remake every run on every score.
    fn drop_references(&self) -> Result<(), String> {
        for cache in [&self.references, &self.site_references] {
            let mut cache = cache.lock().map_err(|e| e.to_string())?;
            let mut held: usize = cache.iter().map(|(_, c)| run_bytes(c)).sum();
            while held > self.reference_bytes {
                let Some((_, newest)) = cache.pop() else { break };
                held -= run_bytes(&newest);
            }
        }
        Ok(())
    }

    /// `M`'s strongest pieces and connections for the behavior ([`Targets`]), measured once and
    /// kept, from the coarse to the fine: every attention's and MLP's output zeroed at every token
    /// (equal to removing it, graph_sites_tests); then each head of the strongest attentions, the two
    /// strongest MLPs' neurons in 8 contiguous groups (and, with a VPD view, each of their two
    /// matrices' subcomponents in 8 groups), and the strongest attentions' and MLPs' outputs cut
    /// into each of the next 8 blocks from the counterfactual, `TARGETED` of each kind; a piece's or
    /// connection's effect is the mean `KL(M ‖ M_e)` at the targets. A deep model measures a few
    /// hundred runs this way, not one per head and per connection (Qwen3-0.6B: about 930).
    pub fn targets(&mut self) -> Result<Targets, String> {
        if let Some(t) = &self.targets {
            return Ok(t.clone());
        }
        if let Some(t) = self.memo_get::<Targets>("targets") {
            self.targets = Some(t.clone());
            return Ok(t);
        }
        let graph = Graph::empty();
        let clean = self.model_outcome(&graph, &Experiment::Clean)?;
        let effect = |m: &Array2<f64>| {
            let kl = kl_bits(&clean, m);
            kl.iter().sum::<f64>() / kl.len().max(1) as f64
        };
        let layers = self.weights.layers.len();
        let blocks = 2 * layers;
        let every = |site: SharedSite, operation: Operation, family: interchange::Family| Experiment::Sites { draw: SiteDraw { family, ops: vec![SiteOp { site, operation, onward: true }], position: 0, length: 1 } };
        let model = graph.model(&self.weights);
        // Site operations leave the weights alone, so each stage's runs go in parallel.
        let measure = |this: &Self, runs: &[Experiment]| -> Result<Vec<f64>, String> {
            runs.par_iter()
                .map(|e| -> Result<f64, String> {
                    let key = Self::key(e);
                    let m = match this.disk_get(&key) {
                        Some(m) => m,
                        None => {
                            let m = this.run(&model, e)?;
                            this.disk_put(&key, &m);
                            m
                        }
                    };
                    Ok(effect(&m))
                })
                .collect()
        };
        // Stage 1: every attention's and MLP's output.
        let sites: Vec<SharedSite> = (0..layers).flat_map(|l| [SharedSite::Attention(l), SharedSite::Mlp(l)]).filter(|s| !matches!(s, SharedSite::Mlp(l) if self.weights.neurons(*l) == 0)).collect();
        let coarse = measure(self, &sites.iter().map(|&s| every(s, Operation::Scale(0), interchange::Family::Zero)).collect::<Vec<_>>())?;
        let mut targets = Targets::default();
        let mut ranked: Vec<(SharedSite, f64)> = sites.iter().copied().zip(coarse).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        for &(s, e) in &ranked {
            if let SharedSite::Mlp(l) = s {
                targets.pieces.push((Block::Neurons { layer: l, neurons: (0..self.weights.neurons(l)).collect() }, e));
            }
        }
        // Stage 2: each head of the strongest attentions, and the strongest outputs' connections.
        let mut first = vec![0usize; layers + 1];
        for (l, layer) in self.weights.layers.iter().enumerate() {
            first[l + 1] = first[l] + layer.heads.len();
        }
        let mut fine: Vec<(Option<Block>, Option<SiteOp>, Experiment)> = Vec::new();
        for &(s, _) in ranked.iter().filter(|(s, _)| matches!(s, SharedSite::Attention(_))).take(TARGETED) {
            if let SharedSite::Attention(l) = s {
                for head in 0..self.weights.layers[l].heads.len() {
                    fine.push((Some(Block::Heads { layer: l, heads: vec![head] }), None, every(SharedSite::Head(first[l] + head), Operation::Scale(0), interchange::Family::Zero)));
                }
            }
        }
        if self.counterfactual_donors() || !self.donors.is_empty() {
            for &(site, _) in ranked.iter().take(TARGETED) {
                let from = match site {
                    SharedSite::Attention(l) => 2 * l,
                    SharedSite::Mlp(l) => 2 * l + 1,
                    _ => continue,
                };
                for to in from + 1..blocks.min(from + 9) {
                    let op = SiteOp { site, operation: Operation::Cut { to }, onward: true };
                    fine.push((None, Some(op), every(site, op.operation, interchange::Family::Cut)));
                }
            }
        }
        let measured = measure(self, &fine.iter().map(|(_, _, e)| e.clone()).collect::<Vec<_>>())?;
        for ((piece, cut, _), e) in fine.into_iter().zip(measured) {
            if let Some(b) = piece {
                targets.pieces.push((b, e));
            }
            if let Some(c) = cut {
                targets.cuts.push((c, e));
            }
        }
        // Sub-blocks of the two strongest MLPs: weight edits, one at a time.
        let mut mlps: Vec<(usize, f64)> = targets.pieces.iter().filter_map(|(b, e)| if let Block::Neurons { layer, .. } = b { Some((*layer, *e)) } else { None }).collect();
        mlps.sort_by(|a, b| b.1.total_cmp(&a.1));
        for &(layer, _) in mlps.iter().take(2) {
            let n = self.weights.neurons(layer);
            // M's own neurons only, so the targets (and the experiments drawn from them) do not depend on
            // the views a checker carries.
            let groups: Vec<(Block, WeightEdit)> = (0..8).map(|g| (g * n / 8..(g + 1) * n / 8).collect::<Vec<usize>>()).filter(|g| !g.is_empty()).map(|g| (Block::Neurons { layer, neurons: g.clone() }, WeightEdit::Neurons { layer, neurons: g, factor: 0.0 })).collect();
            for (block, edit) in groups {
                let m = self.model_outcome(&graph, &Experiment::Edit { edit, targeted: true })?;
                targets.pieces.push((block, effect(&m)));
            }
        }
        targets.pieces.sort_by(|a, b| b.1.total_cmp(&a.1));
        targets.cuts.sort_by(|a, b| b.1.total_cmp(&a.1));
        self.memo_put("targets", &targets);
        self.targets = Some(targets.clone());
        Ok(targets)
    }

    /// Whether every prompt has a counterfactual of its own length: the donor of site operations.
    fn counterfactual_donors(&self) -> bool {
        self.behavior.prompts.iter().all(|p| p.counterfactual.as_ref().is_some_and(|c| c.token_ids.len() == p.token_ids.len()))
    }

    /// A site operation's base sequences, their scored rows and the donor sequences: every prompt
    /// with its counterfactual, else (when the counterfactuals differ in length) the prompts with a
    /// same-length prompt as donor, as node swaps take them; no donor when the operations read none.
    fn site_batches(&self, donor: bool) -> Result<(Batch, Vec<usize>, Option<Batch>), String> {
        let prompts = &self.behavior.prompts;
        let all: Vec<Vec<u32>> = prompts.iter().map(|p| p.token_ids.clone()).collect();
        // Each sequence's partner by item (`Batch::partners`): a prompt's counterfactual, a
        // counterfactual's prompt.
        let mine = |i: usize| prompts[i].partner();
        let theirs = |i: usize| if prompts[i].partner().is_empty() { Vec::new() } else { prompts[i].token_ids.clone() };
        if !donor || self.counterfactual_donors() {
            let base = Batch::new(&all)?.with_partners((0..prompts.len()).map(mine).collect());
            let rows = self.clean.1.clone();
            let donor = if donor { Some(Batch::new(&prompts.iter().map(|p| p.counterfactual.as_ref().map(|c| c.token_ids.clone()).unwrap_or_default()).collect::<Vec<_>>())?.with_partners((0..prompts.len()).map(theirs).collect())) } else { None };
            return Ok((base, rows, donor));
        }
        if self.donors.is_empty() {
            return Err("site operations with a donor need same-length counterfactuals or two prompts of one length".into());
        }
        let base = Batch::new(&self.donors.iter().map(|(i, _)| prompts[*i].token_ids.clone()).collect::<Vec<_>>())?.with_partners(self.donors.iter().map(|&(i, _)| mine(i)).collect());
        let donors = Batch::new(&self.donors.iter().map(|(_, j)| prompts[*j].token_ids.clone()).collect::<Vec<_>>())?.with_partners(self.donors.iter().map(|&(_, j)| mine(j)).collect());
        let targets: Vec<(usize, &[usize])> = self.donors.iter().enumerate().map(|(k, (i, _))| (k, prompts[*i].target_positions.as_slice())).collect();
        let rows = scored_rows(&base, &targets)?;
        Ok((base, rows, Some(donors)))
    }

    /// `circuit` under site operations `draw` ([`run_sites`]): its log-probabilities at the
    /// scored rows.
    fn sites_outcome(&self, circuit: &Circuit, draw: &SiteDraw) -> Result<Array2<f64>, String> {
        let (base, rows, donor) = self.site_inputs(circuit, draw)?;
        run_sites(&self.weights, circuit, (&base, &rows), donor.as_ref(), draw, &self.sites)
    }

    /// A site experiment's base batch with its counterfactual run under the same operations
    /// (`reference_under`), scored rows and donor batch with its own counterfactual run.
    fn site_inputs(&self, circuit: &Circuit, draw: &SiteDraw) -> Result<(Batch, Vec<usize>, Option<Batch>), String> {
        self.site_inputs_on(circuit, draw, false)
    }

    /// [`Checker::site_inputs`]; `swapped`, the counterfactuals as the base (their stand-ins from
    /// the prompts) and no donor.
    fn site_inputs_on(&self, circuit: &Circuit, draw: &SiteDraw, swapped: bool) -> Result<(Batch, Vec<usize>, Option<Batch>), String> {
        let (base, rows, donor) = if swapped {
            let (batch, rows) = self.counterfactual.as_ref().ok_or("the behavior has no counterfactuals")?;
            (batch.clone(), rows.clone(), None)
        } else {
            self.site_batches(Interventions::needs_donor(draw))?
        };
        let (mut base, donor) = (self.referenced(circuit, &base)?, donor.map(|d| self.referenced(circuit, &d)).transpose()?);
        // The stand-ins' run on the counterfactuals takes the same operations (reference_under);
        // deleted pieces stay zero.
        if base.reference.is_some() && !circuit.delete {
            let partner = self.partners_of(&base)?;
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&partner, &mut hasher);
            let key = format!("{} {}", serde_json::to_string(draw).map_err(|e| e.to_string())?, std::hash::Hasher::finish(&hasher));
            let cell = cached_run(&self.site_references, key, self.reference_bytes)?;
            let r = cell
                .get_or_init(|| {
                    let mut b = Batch::new(&partner)?;
                    self.mask(&mut b);
                    reference_under(&self.weights, &b, draw, &self.sites).map(Arc::new)
                })
                .clone()?;
            base.reference = Some(r);
        }
        Ok((base, rows, donor))
    }

    /// Computes the counterfactual runs a program's run of `e` reads (with the current weights), on
    /// this thread before the programs' parallel runs.
    fn prewarm(&self, e: &Experiment) -> Result<(), String> {
        let circuit = Graph::empty().program(&self.weights, true);
        match e {
            Experiment::Sites { draw } => self.site_inputs(&circuit, draw).map(|_| ()),
            Experiment::Counterfactual => self.referenced(&circuit, &self.counterfactual.as_ref().ok_or("the behavior has no counterfactuals")?.0).map(|_| ()),
            _ => self.referenced(&circuit, &self.clean.0).map(|_| ()),
        }
    }

    /// An experiment's scored tokens as (prompt, position), in its rows' order: a node swap's and a
    /// donor-reading site operation's without same-length counterfactuals are the prompts with a
    /// same-length donor; a counterfactual's positions are the prompt's.
    pub fn rows_of(&self, e: &Experiment) -> Vec<(usize, usize)> {
        let subset = match e {
            Experiment::Sites { draw } => Interventions::needs_donor(draw) && !self.counterfactual_donors(),
            _ => false,
        };
        if subset {
            return self.swap_targets();
        }
        self.behavior.prompts.iter().enumerate().flat_map(|(i, p)| p.target_positions.iter().map(move |&t| (i, t))).collect()
    }

    /// Each swapped prompt with its donor.
    pub fn donors(&self) -> &[(usize, usize)] {
        &self.donors
    }

    /// A swap's scored tokens as (prompt, position), in its rows' order.
    pub fn swap_targets(&self) -> Vec<(usize, usize)> {
        self.donors.iter().flat_map(|&(i, _)| self.behavior.prompts[i].target_positions.iter().map(move |&t| (i, t))).collect()
    }

    /// Per scored token of the clean prompts (the behavior's target positions in prompt order): `M`'s
    /// top token on the prompt and on its counterfactual, and the natural log-probabilities of both
    /// tokens under `M(x)`, `M(x')` and `M_c`, the clean complement run of necessity
    /// ([`Checker::necessity_runs`]: the program's nodes taken out, at their counterfactual values,
    /// every other piece computing on the prompt). The ground truth of the reader's question whether
    /// switching an explanation's parts to their counterfactual values moves `M` to the
    /// counterfactual's answer (bench/oracle/graph/reader_questions.py).
    pub fn complement_tokens(&mut self, program: &Program) -> Result<Vec<ComplementToken>, String> {
        let program = self.with_base(program);
        let graph = Graph::parse(&program, &self.weights)?;
        self.weights.load_features(&graph)?;
        if self.counterfactual.is_none() {
            return Err("the behavior has no counterfactuals".into());
        }
        self.set_edit(&Experiment::Clean);
        let native = Graph::empty().model(&self.weights);
        let prompt = self.run(&native, &Experiment::Clean)?;
        let swapped = self.run_swapped(&native, &Experiment::Clean)?;
        // A program of no nodes outside its base takes nothing out: M_c is M.
        let complement = if graph.blocks.len() <= graph.base.len() { prompt.clone() } else { self.run(&graph.complement_model(&self.weights), &Experiment::Clean)? };
        let top = |row: ndarray::ArrayView1<f64>| row.iter().enumerate().fold((0, f64::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
        let targets: Vec<(usize, usize)> = self.behavior.prompts.iter().enumerate().flat_map(|(i, p)| p.target_positions.iter().map(move |&t| (i, t))).collect();
        Ok((0..prompt.nrows())
            .map(|r| {
                let (clean, counter) = (top(prompt.row(r)), top(swapped.row(r)));
                let (p, position) = targets.get(r).copied().unwrap_or((usize::MAX, usize::MAX));
                ComplementToken {
                    prompt: p,
                    position,
                    clean_top: clean,
                    counterfactual_top: counter,
                    clean: [prompt[[r, clean]], prompt[[r, counter]]],
                    counterfactual: [swapped[[r, clean]], swapped[[r, counter]]],
                    complement: [complement[[r, clean]], complement[[r, counter]]],
                }
            })
            .collect())
    }

    /// The graph a program parses to (for describing its experiments), or the empty graph.
    pub fn graph(&self, program: &Program) -> Graph {
        match Graph::parse(program, &self.weights) {
            Ok(graph) => graph,
            Err(e) => {
                log::info!("graph: an invalid program is described as the empty graph: {e}");
                Graph::empty()
            }
        }
    }
}

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
    /// What undeclared pieces and edges carry: "counterfactual" (the default: the same model's
    /// values on the prompt's counterfactual), or the diagnostic "global" (each piece applied to
    /// its average input over the behavior's prompts).
    #[serde(default)]
    pub standin: Option<String>,
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
    let named = x.dot(&factors.1.select(Axis(1), picked)).dot(&factors.0.select(Axis(0), picked));
    if rest { x.dot(&w.t()) - named } else { named }
}

/// A layer's heads as the four native matrices VPD decomposes: the query, key and value maps
/// stacked over heads (heads × head width, by width) and the output columns side by side.
fn attention_maps(layer: &LayerWeights) -> [Array2<f64>; 4] {
    let stack = |f: &dyn Fn(&HeadWeights) -> &Array2<f64>| {
        let views: Vec<_> = layer.heads.iter().map(|h| f(h).view()).collect();
        ndarray::concatenate(Axis(0), &views).expect("heads of one width")
    };
    let outputs: Vec<_> = layer.heads.iter().map(|h| h.output.view()).collect();
    [stack(&|h| &h.query), stack(&|h| &h.key), stack(&|h| &h.value), ndarray::concatenate(Axis(1), &outputs).expect("heads of one width")]
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
        let named = x_hat.dot(&self.fc_v.select(Axis(1), fc)).dot(&self.fc_u.select(Axis(0), fc));
        if rest { x_hat.dot(&mlp.gate.t()) - named } else { named }
    }

    /// The residual write of `down_proj` subcomponents `down` from hidden activations `h`; with
    /// `rest` every other subcomponent and the remainder.
    fn down(&self, mlp: &MlpWeights, down: &[usize], rest: bool, h: &Array2<f64>) -> Array2<f64> {
        let named = h.dot(&self.down_v.select(Axis(1), down)).dot(&self.down_u.select(Axis(0), down));
        if rest { h.dot(&mlp.out.t()) - named } else { named }
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
    /// `M`'s blocks with no decomposition views attached.
    pub fn new(layers: Vec<LayerWeights>, final_norm: Norm, unembedding: Array2<f64>, embedding: Array2<f64>) -> Self {
        Self { layers, final_norm, unembedding, embedding, transcoders: BTreeMap::new(), vpd: BTreeMap::new(), vpd_attention: BTreeMap::new(), library: BTreeMap::new() }
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

/// The pieces of one site: some heads of a layer's attention, some neurons of its MLP, or some
/// transcoder features of its MLP (`rest`: the MLP minus those features, i.e. every other feature
/// and the transcoder's exact error piece).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
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
    fn site(&self) -> usize {
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

    fn routes(&self) -> &'static [Route] {
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
    /// Stand-ins from the counterfactual run (else from average inputs).
    pub counterfactual: bool,
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
    pub ids: Vec<String>,
    pub blocks: Vec<Block>,
    pub edges: Vec<(Writer, Option<usize>, Route)>,
    /// The program's stand-ins: counterfactual (the default) or global averages (diagnostic).
    pub counterfactual: bool,
    /// Edges within one MLP site, (writer node, reader node): `c_fc` subcomponents to `down_proj`
    /// subcomponents through the hidden pre-activation.
    pub internal: Vec<(usize, usize)>,
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
        let mut featured: BTreeSet<(usize, usize)> = BTreeSet::new();
        let mut sliced: BTreeSet<(usize, usize, usize)> = BTreeSet::new();
        let mut views: BTreeMap<(usize, bool), String> = BTreeMap::new();
        for node in &program.nodes {
            if node.id == "embed" || node.id == "logits" || ids.contains(&node.id) {
                return Err(format!("node id {} is reserved or repeated", node.id));
            }
            if node.rule.as_ref().is_some_and(|r| !r.is_null()) {
                return Err(format!("{}: rules are not executed yet", node.id));
            }
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
                        let picked = indices(&piece.index, count, kind)?;
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
                        let picked = indices(&piece.index, count, kind)?;
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
            ids.push(node.id.clone());
            blocks.push(block);
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
                let joins = match (&blocks[w], &blocks[r]) {
                    (Block::Slices { fc, .. }, Block::Slices { down, .. }) => !fc.is_empty() && !down.is_empty(),
                    (Block::AttnSlices { q, k, v, .. }, Block::AttnSlices { o, .. }) => !(q.is_empty() && k.is_empty() && v.is_empty()) && !o.is_empty(),
                    _ => false,
                };
                if !joins || route != Route::Input {
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
        let counterfactual = match program.standin.as_deref() {
            None | Some("counterfactual") => true,
            Some("global") => false,
            Some(other) => return Err(format!("unknown stand-in {other} (counterfactual or global)")),
        };
        Ok(Self { ids, blocks, edges, counterfactual, internal })
    }

    /// The empty program: every piece a stand-in.
    pub fn empty() -> Self {
        Self { ids: Vec::new(), blocks: Vec::new(), edges: Vec::new(), counterfactual: true, internal: Vec::new() }
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
                Unit { block: block.clone(), computes: true, routes, hidden }
            })
            .collect();
        units.extend(self.complement(weights).into_iter().map(|block| Unit { block, computes: false, routes: [Incoming::all(), Incoming::all(), Incoming::all()], hidden: Incoming::all() }));
        Circuit { units, logits: routed(None, Route::Input), nodes: self.blocks.len(), counterfactual: self.counterfactual }
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
                // A feature's encoder row, bias and decoder row.
                Block::Features { features, .. } => count += features.len() * (2 * d + 1),
                // A subcomponent's two vectors: width and hidden.
                Block::Slices { layer, fc, down, .. } => count += (fc.len() + down.len()) * (d + weights.neurons(*layer)),
                Block::AttnSlices { layer, q, k, v, o, .. } => {
                    let width: usize = weights.layers[*layer].heads.iter().map(|h| h.query.nrows()).sum();
                    count += (q.len() + k.len() + v.len() + o.len()) * (d + width);
                }
            }
        }
        // Counterfactual stand-ins are the model's own values and cost nothing; average stand-ins
        // are numbers the program carries.
        if !self.counterfactual {
            count += d;
            for block in self.complement(weights) {
                count += match block {
                    Block::Heads { heads, .. } => heads.len() * d,
                    Block::Neurons { .. } | Block::Features { .. } | Block::Slices { .. } | Block::AttnSlices { .. } => d,
                };
            }
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
    /// Per layer each neuron's mean activation, and whether a neuron's stand-in is its mean
    /// activation through its current down column (`D h̄`, the mean of its write; edits of its down
    /// column reach it, edits of its gate and up rows do not) instead of the neuron applied to its
    /// layer's mean input.
    pub activations: Vec<Array1<f64>>,
    pub mean_output: bool,
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
        let mut stats = Self { tokens, heads: Vec::new(), mlps: Vec::new(), activations: Vec::new(), mean_output: false };
        let run = execute(weights, &stats, &circuit, &batch, &[], &BTreeMap::new(), true)?;
        let recorded = run.recorded.ok_or("no recorded inputs")?;
        stats.heads = recorded.0;
        stats.mlps = recorded.1;
        stats.activations = recorded.2;
        Ok(stats)
    }
}

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
        Ok(Self { tokens, spans, reference: None, blocks })
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
    pub embed: Array2<f64>,
    pub reads: Vec<Vec<Array2<f64>>>,
    pub active: Vec<Array2<f64>>,
    pub mlp: Vec<Array2<f64>>,
    /// Per layer its MLP's normed input `x̂` (rows × width): transcoder features read it.
    pub inputs: Vec<Array2<f64>>,
    /// Per layer its attention's normed input (rows × width): VPD q, k, v subcomponents read it.
    pub attention_inputs: Vec<Array2<f64>>,
}

impl Reference {
    /// Block `block`'s write in the run (rows × width).
    fn write(&self, weights: &Weights, block: &Block) -> Result<Array2<f64>, String> {
        let rows = self.embed.nrows();
        match block {
            Block::Heads { layer, heads } => {
                let mut out = Array2::<f64>::zeros((rows, weights.width()));
                for &h in heads {
                    let z = self.reads.get(*layer).and_then(|r| r.get(h)).ok_or("a head the reference did not record")?;
                    out += &par_dot(z, weights.layers[*layer].heads[h].output.t());
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
                    Ok(&self.mlp[*layer] - &active.select(Axis(1), &rest).dot(&mlp.out.select(Axis(1), &rest).t()))
                } else {
                    Ok(active.select(Axis(1), neurons).dot(&mlp.out.select(Axis(1), neurons).t()))
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
pub fn reference(weights: &Weights, stats: &Stats, batch: &Batch) -> Result<Reference, String> {
    let circuit = Graph::empty().model(weights);
    let mut plain = batch.clone();
    plain.reference = None;
    run(weights, stats, &circuit, &plain, &[], &BTreeMap::new(), false, &Interventions::default(), true)?.captured.ok_or_else(|| "no captured run".to_string())
}

/// `M`'s run on `batch` recorded per piece ([`Reference`]) under the site operations `draw`
/// that act on a run without a donor (a swap or a cut reads the run's own donor, the
/// counterfactual, which on the counterfactual's own run changes nothing): the counterfactual run
/// of a site experiment, the same experiment applied to it.
pub fn reference_under(weights: &Weights, stats: &Stats, batch: &Batch, draw: &SiteDraw, units: &SiteUnits) -> Result<Reference, String> {
    let own = SiteDraw { ops: draw.ops.iter().filter(|o| !matches!(o.operation, Operation::Swap | Operation::Cut { .. })).copied().collect(), ..draw.clone() };
    let heads: BTreeSet<(usize, usize)> = own.ops.iter().filter_map(|o| if let SharedSite::Head(h) = o.site { head_of(weights, h).ok() } else { None }).collect();
    let circuit = Graph::empty().model(weights).split_heads(&heads);
    let mut plain = batch.clone();
    plain.reference = None;
    let ops = Interventions::resolve(&own, &circuit, weights, &plain, units, None, None)?;
    run(weights, stats, &circuit, &plain, &[], &BTreeMap::new(), false, &ops, true)?.captured.ok_or_else(|| "no captured run".to_string())
}

/// One run: the logits' log-probabilities at the scored rows (rows × vocabulary) and every
/// computing unit's actual write (rows × width).
pub struct Execution {
    pub log_probabilities: Array2<f64>,
    pub writes: Vec<Option<Array2<f64>>>,
    recorded: Option<(Vec<Vec<Array1<f64>>>, Vec<Array1<f64>>, Vec<Array1<f64>>)>,
    /// Per (unit, route slot) its normed input, at the sites `Interventions::record` lists.
    pub normed: BTreeMap<(usize, usize), Array2<f64>>,
    captured: Option<Reference>,
}

fn project(x: &Array2<f64>, map: &Array2<f64>, norm: Option<&(Array1<f64>, f64)>) -> Array2<f64> {
    let mut out = par_dot(x, map.t());
    if let Some((gain, epsilon)) = norm {
        for mut row in out.outer_iter_mut() {
            let r = rms_scale(row.view(), *epsilon);
            row.zip_mut_with(gain, |v, g| *v *= r * g);
        }
    }
    out
}

/// Heads `heads` of `layer` on their query, key and value inputs (residual streams), `normed`
/// applied to each normed input (route slot, value). With `record`, each head's mean
/// attention-weighted normalized value input as well.
fn heads_write(layer: &LayerWeights, heads: &[usize], inputs: [&Array2<f64>; 3], spans: &[(usize, usize)], blocks: &[Vec<[usize; 4]>], record: bool, capture: bool, normed: &mut dyn FnMut(usize, &mut Array2<f64>)) -> (Array2<f64>, Vec<Array1<f64>>, Vec<Array2<f64>>) {
    let norm = &layer.attention;
    let (mut q_hat, mut k_hat) = (norm.apply(inputs[0]), norm.apply(inputs[1]));
    let v_unit = norm.unit(inputs[2]);
    let mut v_hat = &v_unit * &norm.gain.view().insert_axis(Axis(0));
    normed(0, &mut q_hat);
    normed(1, &mut k_hat);
    normed(2, &mut v_hat);
    let (rows, d) = v_hat.dim();
    let mut out = Array2::<f64>::zeros((rows, d));
    let (mut recorded, mut reads) = (Vec::new(), Vec::new());
    for &h in heads {
        let w = &layer.heads[h];
        let (q, k, v) = (project(&q_hat, &w.query, w.query_norm.as_ref()), project(&k_hat, &w.key, w.key_norm.as_ref()), par_dot(&v_hat, w.value.t()));
        let mut z = Array2::<f64>::zeros(v.dim());
        let mut mixed = Array1::<f64>::zeros(d);
        for (n, &(start, length)) in spans.iter().enumerate() {
            let positions: Vec<u32> = (0..length as u32).collect();
            let span = start..start + length;
            let (qs, ks) = (q.slice(s![span.clone(), ..]).to_owned(), k.slice(s![span.clone(), ..]).to_owned());
            let (qs, ks) = (rotate(&qs, w.rotary, &positions, false), rotate(&ks, w.rotary, &positions, false));
            let mut a = probabilities(qs.view(), ks.view(), &positions, 0, w.scale, w.causal);
            if let Some(b) = blocks.get(n).filter(|b| !b.is_empty()) {
                block_attention(&mut a, b);
            }
            z.slice_mut(s![span.clone(), ..]).assign(&a.dot(&v.slice(s![span.clone(), ..])));
            if record {
                mixed += &a.dot(&v_unit.slice(s![span, ..])).sum_axis(Axis(0));
            }
        }
        if record {
            recorded.push(mixed / rows as f64);
        }
        out += &par_dot(&z, w.output.t());
        if capture {
            reads.push(z);
        }
    }
    (out, recorded, reads)
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
    par_dot(&neurons_active(mlp, neurons, x_hat), mlp.out.select(Axis(1), neurons).t())
}

/// The activations of neurons `neurons` (rows × neurons).
fn neurons_active(mlp: &MlpWeights, neurons: &[usize], x_hat: &Array2<f64>) -> Array2<f64> {
    let gate = mlp.gate.select(Axis(0), neurons);
    let mut h = par_dot(x_hat, gate.t());
    let bias = mlp.bias.select(Axis(0), neurons);
    h += &bias.view().insert_axis(Axis(0));
    h.mapv_inplace(|g| mlp.law.apply(g));
    if let Some(up) = &mlp.up {
        let mut u = par_dot(x_hat, up.select(Axis(0), neurons).t());
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

/// A block's stand-in write (width): its pieces applied with `weights` to their stored inputs.
fn stand_in(weights: &Weights, stats: &Stats, block: &Block) -> Result<Array1<f64>, String> {
    Ok(match block {
        Block::AttnSlices { .. } => return Err("average stand-ins do not cover VPD attention subcomponents (use counterfactual stand-ins)".into()),
        Block::Slices { layer, down, rest, .. } => {
            let mlp = weights.layers[*layer].mlp.as_ref().ok_or("a VPD view of a layer without an MLP")?;
            let vpd = weights.vpd.get(layer).ok_or_else(|| format!("layer {layer} has no VPD view"))?;
            let x_hat = (&stats.mlps[*layer] * &weights.layers[*layer].mlp_norm.gain).insert_axis(Axis(0));
            let mut h = x_hat.dot(&mlp.gate.t()) + &mlp.bias.view().insert_axis(Axis(0));
            h.mapv_inplace(|t| mlp.law.apply(t));
            vpd.down(mlp, down, *rest, &h).row(0).to_owned()
        }
        Block::Features { layer, features, rest } => {
            let x_hat = (&stats.mlps[*layer] * &weights.layers[*layer].mlp_norm.gain).insert_axis(Axis(0));
            features_write(weights, *layer, features, *rest, &x_hat)?.row(0).to_owned()
        }
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
        Block::Neurons { layer, neurons } if stats.mean_output => {
            let mlp = weights.layers[*layer].mlp.as_ref().expect("a neuron block has an MLP");
            mlp.out.select(Axis(1), neurons).dot(&stats.activations[*layer].select(Axis(0), neurons))
        }
        Block::Neurons { layer, neurons } => {
            let lw = &weights.layers[*layer];
            let x_hat = (&stats.mlps[*layer] * &lw.mlp_norm.gain).insert_axis(Axis(0));
            neurons_write(lw.mlp.as_ref().expect("a neuron block has an MLP"), neurons, &x_hat).row(0).to_owned()
        }
    })
}

/// Executes `circuit` on `batch`: per unit in site order its route inputs (the stand-in stream
/// plus the declared writers' actual minus stand-in writes, or the actual stream minus the cut
/// writers'), its write (actual, `swaps`' value, or its stand-in), and the logits at `scored` rows.
/// With `record`, `M`'s stand-in inputs are measured on the way (every unit must compute and read
/// the actual stream).
pub fn execute(weights: &Weights, stats: &Stats, circuit: &Circuit, batch: &Batch, scored: &[usize], swaps: &BTreeMap<usize, Array2<f64>>, record: bool) -> Result<Execution, String> {
    execute_with(weights, stats, circuit, batch, scored, swaps, record, &Interventions::default())
}

/// [`execute`] under the row interventions `ops` (site operations, [`Interventions`]): after a
/// site, writers' writes scaled or swapped and vectors pushed into the stream; at a site, its
/// units' normed inputs scaled, pushed or swapped and cut writers read on the donor.
pub fn execute_with(weights: &Weights, stats: &Stats, circuit: &Circuit, batch: &Batch, scored: &[usize], swaps: &BTreeMap<usize, Array2<f64>>, record: bool, ops: &Interventions) -> Result<Execution, String> {
    run(weights, stats, circuit, batch, scored, swaps, record, ops, false)
}

/// [`execute_with`], with `capture` recording every head's read and every MLP's activations and
/// write ([`Reference`]; every unit must compute, each layer's heads and MLP one unit each).
/// Stand-ins: with `batch.reference`, each unit's write in that run (counterfactual stand-ins),
/// else each piece applied to its average input ([`Stats`]).
fn run(weights: &Weights, stats: &Stats, circuit: &Circuit, batch: &Batch, scored: &[usize], swaps: &BTreeMap<usize, Array2<f64>>, record: bool, ops: &Interventions, capture: bool) -> Result<Execution, String> {
    let (rows, d) = (batch.tokens.len(), weights.width());
    let vocabulary = weights.embedding.nrows();
    if let Some(t) = batch.tokens.iter().find(|t| **t as usize >= vocabulary) {
        return Err(format!("token {t} outside the vocabulary of {vocabulary}"));
    }
    let embed = weights.embedding.select(Axis(0), &batch.tokens.iter().map(|t| *t as usize).collect::<Vec<_>>());
    let units = circuit.units.len();
    let broadcast = |v: &Array1<f64>| Array2::from_shape_fn((rows, d), |(_, c)| v[c]);
    let (embed_standin, standins): (Array2<f64>, Vec<Array2<f64>>) = match &batch.reference {
        _ if record => (Array2::zeros((rows, d)), vec![Array2::zeros((rows, d)); units]),
        Some(r) => {
            if r.embed.nrows() != rows {
                return Err(format!("a counterfactual run of {} tokens for a batch of {rows}", r.embed.nrows()));
            }
            (r.embed.clone(), circuit.units.iter().map(|u| r.write(weights, &u.block)).collect::<Result<_, _>>()?)
        }
        None => {
            let mut e = Array1::<f64>::zeros(d);
            for (&t, &f) in &stats.tokens {
                e.scaled_add(f, &weights.embedding.row(t as usize));
            }
            (broadcast(&e), circuit.units.iter().map(|u| stand_in(weights, stats, &u.block).map(|v| broadcast(&v))).collect::<Result<_, _>>()?)
        }
    };
    let mut captured = capture.then(|| Reference {
        embed: embed.clone(),
        reads: weights.layers.iter().map(|l| vec![Array2::zeros((0, 0)); l.heads.len()]).collect(),
        active: vec![Array2::zeros((0, 0)); weights.layers.len()],
        mlp: vec![Array2::zeros((0, 0)); weights.layers.len()],
        inputs: vec![Array2::zeros((0, 0)); weights.layers.len()],
        attention_inputs: vec![Array2::zeros((0, 0)); weights.layers.len()],
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
    let (mut head_stats, mut mlp_stats) = (vec![vec![Array1::<f64>::zeros(d); 0]; weights.layers.len()], vec![Array1::<f64>::zeros(d); weights.layers.len()]);
    let mut active_stats: Vec<Array1<f64>> = (0..weights.layers.len()).map(|l| Array1::zeros(weights.neurons(l))).collect();
    if record {
        for (l, layer) in weights.layers.iter().enumerate() {
            head_stats[l] = vec![Array1::zeros(d); layer.heads.len()];
        }
    }
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
                None => broadcast(&(&stats.mlps[layer] * &norm.gain)),
            };
            let fc_of = |b: &Block, x: &Array2<f64>| match b {
                Block::Slices { fc, rest, .. } => Ok(vpd.fc(mlp, fc, *rest, x)),
                _ => Err("a VPD-view MLP unit of another block".to_string()),
            };
            let mut pre_ref = x_ref.dot(&mlp.gate.t());
            pre_ref += &mlp.bias.view().insert_axis(Axis(0));
            let mut deltas: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
            for &u in &slices {
                let unit = &circuit.units[u];
                if !unit.computes || swaps.contains_key(&u) {
                    continue;
                }
                let routes = unit.block.routes();
                let mut inputs: Vec<Array2<f64>> = routes.iter().map(|r| input(&unit.routes[r.slot()], &st)).collect();
                ops.cut_inputs(site, unit, routes, &mut inputs, &st)?;
                let mut x_hat = norm.apply(&inputs[0]);
                ops.normed(site, u, 0, &mut x_hat, &mut normed_kept);
                deltas.insert(u, fc_of(&unit.block, &x_hat)? - fc_of(&unit.block, &x_ref)?);
            }
            for &u in &slices {
                let unit = &circuit.units[u];
                if !unit.computes || swaps.contains_key(&u) {
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
                // queries, keys and values. A program under average stand-ins fails at its
                // complement's stand-in (`stand_in`).
                None => Array2::zeros((rows, d)),
            };
            let factors = [&vpd.q, &vpd.k, &vpd.v];
            let refs: Vec<Array2<f64>> = (0..3).map(|m| x_ref.dot(&maps[m].t())).collect();
            let mut deltas: BTreeMap<usize, Vec<Array2<f64>>> = BTreeMap::new();
            for &u in &attention {
                let unit = &circuit.units[u];
                if !unit.computes || swaps.contains_key(&u) {
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
                    ds.push(sliced(factors[m], &maps[m], list, *rest, &x_hat) - sliced(factors[m], &maps[m], list, *rest, &x_ref));
                }
                deltas.insert(u, ds);
            }
            for &u in &attention {
                let unit = &circuit.units[u];
                if !unit.computes || swaps.contains_key(&u) {
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
                slice_writes.insert(u, sliced(&vpd.o, &maps[3], o, *rest, &z));
            }
        }
        for &u in &order[at..end] {
            let unit = &circuit.units[u];
            if !unit.computes {
                continue;
            }
            if let Some(value) = swaps.get(&u) {
                st.writes[u] = Some(value.clone());
                continue;
            }
            if let Some(w) = slice_writes.remove(&u) {
                st.writes[u] = Some(w);
                continue;
            }
            let routes = unit.block.routes();
            let mut inputs: Vec<Array2<f64>> = routes.iter().map(|r| input(&unit.routes[r.slot()], &st)).collect();
            ops.cut_inputs(site, unit, routes, &mut inputs, &st)?;
            let mut normed = |slot: usize, x: &mut Array2<f64>| ops.normed(site, u, slot, x, &mut normed_kept);
            let write = match &unit.block {
                Block::Heads { layer, heads } => {
                    let (w, recorded, reads) = heads_write(&weights.layers[*layer], heads, [&inputs[0], &inputs[1], &inputs[2]], &batch.spans, &batch.blocks, record, capture, &mut normed);
                    for (h, r) in heads.iter().zip(recorded) {
                        head_stats[*layer][*h] = r;
                    }
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
                    if record {
                        mlp_stats[*layer] = lw.mlp_norm.unit(&inputs[0]).mean_axis(Axis(0)).ok_or("no rows")?;
                        let mean = active.mean_axis(Axis(0)).ok_or("no rows")?;
                        for (k, &i) in neurons.iter().enumerate() {
                            active_stats[*layer][i] = mean[k];
                        }
                    }
                    let write = par_dot(&active, mlp.out.select(Axis(1), neurons).t());
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
            st.writes[u] = Some(write);
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
    Ok(Execution { log_probabilities, writes: st.writes, recorded: record.then_some((head_stats, mlp_stats, active_stats)), normed: normed_kept, captured })
}

/// `a · b`, row blocks of `a` on parallel threads when called outside rayon's pool (a run the
/// batch makes alone: a weight edit's group of one, a counterfactual run every program waits
/// for). Inside the pool it multiplies on its own thread: a worker that waits on nested parallel
/// work steals other runs, and a stolen run that waits on the counterfactual run this worker is
/// computing (a `OnceLock`) never returns, which deadlocked a vpd4l batch.
fn par_dot(a: &Array2<f64>, b: ndarray::ArrayView2<f64>) -> Array2<f64> {
    const BLOCK: usize = 64;
    if a.nrows() <= BLOCK || rayon::current_thread_index().is_some() {
        return a.dot(&b);
    }
    let mut out = Array2::<f64>::zeros((a.nrows(), b.ncols()));
    out.axis_chunks_iter_mut(Axis(0), BLOCK).into_par_iter().zip(a.axis_chunks_iter(Axis(0), BLOCK).into_par_iter()).for_each(|(mut o, x)| o.assign(&x.dot(&b)));
    out
}

/// Next-token log-probabilities of final streams (rows × width) through the final norm (its own
/// RMS) and the unembedding, normalized in float64.
pub fn log_probabilities(weights: &Weights, last: &Array2<f64>) -> Result<Array2<f64>, String> {
    let mut logits = par_dot(last, weights.unembedding.t());
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
            for &i in indices {
                u.row_mut(i).mapv_inplace(|x| x * factor);
            }
        }
        let m = Self::matrix(weights, layer, head, matrix)?;
        let saved = m.clone();
        match self {
            Self::Subcomponents { .. } => {
                let d = delta.ok_or("no change")?;
                if d.dim() != m.dim() {
                    return Err("VPD factors of another shape than the matrix".into());
                }
                *m += &d;
            }
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
        Ok(Restore { layer, head, matrix, saved, factors, shared, heads: Vec::new(), attention: None })
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
        for &i in indices {
            factors.0.row_mut(i).mapv_inplace(|x| x * factor);
        }
        let mut heads = Vec::new();
        let mut at = 0;
        for h in 0..weights.layers[layer].heads.len() {
            let m = Self::matrix(weights, layer, Some(h), matrix)?;
            heads.push((h, m.clone()));
            let part = if matrix == Matrix::Output { delta.slice(s![.., at..at + m.ncols()]) } else { delta.slice(s![at..at + m.nrows(), ..]) };
            if part.dim() != m.dim() {
                return Err("VPD attention factors of another shape than the heads' maps".into());
            }
            at += if matrix == Matrix::Output { m.ncols() } else { m.nrows() };
            *m += &part;
        }
        let (first, saved) = heads.first().cloned().ok_or("a layer without heads")?;
        Ok(Restore { layer, head: Some(first), matrix, saved, factors: None, shared: Vec::new(), heads: heads.into_iter().skip(1).collect(), attention: Some((map, saved_u)) })
    }
}

/// A matrix's value before an edit.
pub struct Restore {
    layer: usize,
    head: Option<usize>,
    matrix: Matrix,
    saved: Array2<f64>,
    /// A subcomponent edit's VPD factors before it (`down_proj`'s when true).
    factors: Option<(bool, Array2<f64>)>,
    /// The heads sharing the edited key or value matrix, edited alike.
    shared: Vec<usize>,
    /// Further heads' maps before an attention subcomponent edit, and its factors' `U`.
    heads: Vec<(usize, Array2<f64>)>,
    attention: Option<(usize, Array2<f64>)>,
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
            [&mut vpd.q, &mut vpd.k, &mut vpd.v, &mut vpd.o][map.min(3)].0 = u;
        }
        *WeightEdit::matrix(weights, self.layer, self.head, self.matrix)? = self.saved;
        if let Some((down, u)) = self.factors {
            let vpd = weights.vpd.get_mut(&self.layer).ok_or("the VPD view went missing")?;
            *(if down { &mut vpd.down_u } else { &mut vpd.fc_u }) = u;
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
    Edit { edit: WeightEdit, aimed: bool },
    /// Node `node`'s write replaced by its write on the donor prompt (the next prompt of the same
    /// length).
    Swap { node: usize },
    /// The edge from `from` to `to` (a node, `None` the logits) on `route` cut: the reader gets the
    /// writer's stand-in write.
    Cut { from: Writer, to: Option<usize>, route: Route, declared: bool },
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
            Self::Edit { aimed: true, .. } => "edit_aimed",
            Self::Edit { aimed: false, .. } => "edit_uniform",
            Self::Swap { .. } => "swap",
            Self::Cut { declared: true, .. } => "cut_declared",
            Self::Cut { declared: false, .. } => "cut_undeclared",
            Self::Sites { draw } => match draw.family {
                interchange::Family::Swap => "site_swap",
                interchange::Family::Zero => "site_zero",
                interchange::Family::Scale => "site_scale",
                interchange::Family::Push => "site_push",
                interchange::Family::Cut => "site_cut",
                interchange::Family::Read => "site_read",
            },
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
                WeightEdit::Subcomponents { layer, down, indices, factor } => format!("PD.vpd[{layer}].{}{indices:?} times {factor}", if *down { "down_proj" } else { "c_fc" }),
                WeightEdit::AttnSubcomponents { layer, map, indices, factor } => format!("PD.vpd[{layer}].{}{indices:?} times {factor}", ["q_proj", "k_proj", "v_proj", "o_proj"][(*map).min(3)]),
            },
            Self::Swap { node: n } => format!("{}'s output replaced by its output on another prompt", node(*n)),
            Self::Cut { from, to, route, .. } => format!("the connection {} >> {}.{route:?} cut (the reader gets its average)", writer(from), to.map_or("logits".to_string(), node)),
            Self::Sites { draw } => describe_sites(draw),
        }
    }
}

/// The checker's draw of `count` experiments beyond clean (and counterfactual when the prompts
/// have them): weight edits (half uniform over pieces at random granularity, half aimed at the
/// program's pieces and pieces it omits, half of those among the four omitted heads first in
/// `strongest`, heads by measured removal effect), rank-one perturbations, node swaps, edge cuts
/// (declared edges and undeclared pairs).
pub fn sample(weights: &Weights, graph: &Graph, counterfactual: bool, count: usize, seed: u64, strongest: &[(usize, usize)], units: &SiteUnits) -> Vec<Experiment> {
    let sites = &units.pool;
    let aimed_sites = units.aimed_sites(weights, graph);
    let mut out = vec![Experiment::Clean];
    if counterfactual {
        out.push(Experiment::Counterfactual);
    }
    let layers = weights.layers.len();
    let factors = [0.0, 0.5, 2.0];
    let omitted = graph.complement(weights);
    let declared = |l: usize, h: usize| graph.blocks.iter().any(|b| matches!(b, Block::Heads { layer, heads } if *layer == l && heads.contains(&h)));
    let strong: Vec<Block> = strongest.iter().filter(|(l, h)| !declared(*l, *h)).take(4).map(|&(layer, h)| Block::Heads { layer, heads: vec![h] }).collect();
    let random_edit = |rng: &mut StdRng, block: Option<&Block>| -> WeightEdit {
        let factor = factors[rng.random_range(0..factors.len())];
        let block = match block {
            Some(b) => b.clone(),
            None => {
                let l = rng.random_range(0..layers);
                if weights.vpd.contains_key(&l) && rng.random_bool(1.0 / 3.0) {
                    // A layer with a VPD view: its subcomponents a third of the time.
                    if weights.vpd_attention.contains_key(&l) && rng.random_bool(0.5) {
                        Block::AttnSlices { layer: l, q: Vec::new(), k: Vec::new(), v: Vec::new(), o: Vec::new(), rest: true }
                    } else {
                        Block::Slices { layer: l, fc: Vec::new(), down: Vec::new(), rest: true }
                    }
                } else if rng.random_bool(0.5) || weights.neurons(l) == 0 {
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
            // Transcoder features are not M's weights: an aimed edit takes their layer's MLP.
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
    let declared_edges: Vec<(Writer, Option<usize>, Route)> = graph.edges.clone();
    // Half the draws are the behavior's (uniform edits and rank-one perturbations drawn from `seed`
    // alone, the same for every program, so `M`'s outcomes are shared across programs), half are
    // aimed at the program (its pieces and the strongest it omits, its nodes, its edges).
    let (mut fixed, mut aimed) = (StdRng::seed_from_u64(seed), StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15));
    let mut kinds = vec!["edit_aimed", "cut_undeclared"];
    if !graph.blocks.is_empty() {
        kinds.push("swap");
    }
    if !declared_edges.is_empty() {
        kinds.push("cut_declared");
    }
    if !aimed_sites.is_empty() && !units.directions.is_empty() {
        kinds.push("site_aimed");
    }
    for k in 0..count {
        let rng = if k % 2 == 0 { &mut fixed } else { &mut aimed };
        // The behavior's half: uniform weight edits, rank-one perturbations and (when a pool of drawn
        // site operations is given, a manifest's) site operations, a third each.
        let fixed_kinds: &[&str] = if sites.is_empty() { &["edit_uniform", "rank_one"] } else { &["edit_uniform", "rank_one", "sites"] };
        let kind = if k % 2 == 0 { fixed_kinds[rng.random_range(0..fixed_kinds.len())] } else { kinds[rng.random_range(0..kinds.len())] };
        out.push(match kind {
            "sites" => Experiment::Sites { draw: sites[rng.random_range(0..sites.len())].clone() },
            "site_aimed" => match units.draw_at(rng, &aimed_sites, weights) {
                Some(draw) => Experiment::Sites { draw },
                None => Experiment::Edit { edit: random_edit(rng, graph.blocks.first()), aimed: true },
            },
            "edit_uniform" => Experiment::Edit { edit: random_edit(rng, None), aimed: false },
            "edit_aimed" => {
                let own = !graph.blocks.is_empty() && (omitted.is_empty() || rng.random_bool(0.5));
                let pool = if own {
                    &graph.blocks
                } else if !strong.is_empty() && rng.random_bool(0.5) {
                    &strong
                } else {
                    &omitted
                };
                let block = pool[rng.random_range(0..pool.len())].clone();
                Experiment::Edit { edit: random_edit(rng, Some(&block)), aimed: true }
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
enum After {
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
enum OnInput {
    Scale(Vec<usize>, f64),
    Push(Vec<usize>, Array1<f64>),
    /// The normed input at the rows replaced by the same unit's on the donor.
    Swap(Vec<usize>),
}

/// The same circuit's run on the donor sequences: `embed`'s write, every unit's write, and the
/// normed inputs at the sites an input swap names.
#[derive(Clone, Debug)]
pub struct Donor {
    embed: Array2<f64>,
    writes: Vec<Option<Array2<f64>>>,
    normed: BTreeMap<(usize, usize), Array2<f64>>,
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
    after: Vec<(Option<usize>, After)>,
    inputs: Vec<(usize, OnInput)>,
    /// Readers at a site: the writers' writes on the donor in place of their actual writes at the
    /// rows, through the routes that read the writers' actual writes.
    cuts: Vec<(usize, Vec<Writer>, Vec<usize>)>,
    donor: Option<Donor>,
    /// Sites whose units' normed inputs a run keeps (`Execution::normed`).
    record: BTreeSet<usize>,
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
            out.donor = Some(Donor { embed: weights.embedding.select(Axis(0), &tokens), writes: run.writes.clone(), normed: run.normed.clone() });
        }
        let heads_at = |pred: &dyn Fn(&Block) -> bool| -> Vec<Writer> { circuit.units.iter().enumerate().filter(|(_, u)| pred(&u.block)).map(|(i, _)| Writer::Unit(i)).collect() };
        for op in &draw.ops {
            let rows = op_rows(batch, draw.position, draw.length, op.onward);
            let (point, writers): (Option<usize>, Vec<Writer>) = match op.site {
                SharedSite::Head(h) => {
                    let (l, hh) = head_of(weights, h)?;
                    (Some(2 * l), heads_at(&|b| matches!(b, Block::Heads { layer, heads } if *layer == l && heads.as_slice() == [hh])))
                }
                SharedSite::Attention(l) => (Some(2 * l), heads_at(&|b| matches!(b, Block::Heads { layer, .. } if *layer == l))),
                SharedSite::Mlp(l) => (Some(2 * l + 1), heads_at(&|b| matches!(b, Block::Neurons { layer, .. } if *layer == l))),
                SharedSite::Embedding => (None, vec![Writer::Embed]),
                SharedSite::Stream(b) => (Some(b), [Writer::Embed].into_iter().chain(heads_at(&|x| x.site() <= b)).collect()),
                SharedSite::Input(_) => (None, Vec::new()),
            };
            if let SharedSite::Head(h) = op.site
                && writers.len() != 1
            {
                return Err(format!("head {h} is not its own unit (split the circuit first)"));
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
pub fn run_sites(weights: &Weights, stats: &Stats, circuit: &Circuit, base: (&Batch, &[usize]), donor: Option<&Batch>, draw: &SiteDraw, units: &SiteUnits) -> Result<Array2<f64>, String> {
    let heads: BTreeSet<(usize, usize)> = draw.ops.iter().filter_map(|o| if let SharedSite::Head(h) = o.site { head_of(weights, h).ok() } else { None }).collect();
    let circuit = circuit.split_heads(&heads);
    let donor_run = match donor {
        Some(b) if Interventions::needs_donor(draw) => Some(execute_with(weights, stats, &circuit, b, &[], &BTreeMap::new(), false, &Interventions::recording(Interventions::donor_record(draw, weights)))?),
        _ => None,
    };
    let ops = Interventions::resolve(draw, &circuit, weights, base.0, units, donor_run.as_ref(), donor)?;
    Ok(execute_with(weights, stats, &circuit, base.0, base.1, &BTreeMap::new(), false, &ops)?.log_probabilities)
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

    /// The sites of a program's nodes: each declared head's output, the attention or MLP output of
    /// each layer holding a node, a node's block input and the stream after it.
    pub fn aimed_sites(&self, weights: &Weights, graph: &Graph) -> Vec<SharedSite> {
        let mut out = BTreeSet::new();
        let mut first = vec![0usize; weights.layers.len() + 1];
        for (l, layer) in weights.layers.iter().enumerate() {
            first[l + 1] = first[l] + layer.heads.len();
        }
        for block in &graph.blocks {
            match block {
                Block::Heads { layer, heads } => {
                    out.extend(heads.iter().map(|h| SharedSite::Head(first[*layer] + h)));
                    out.insert(SharedSite::Attention(*layer));
                }
                Block::AttnSlices { layer, .. } => {
                    out.insert(SharedSite::Attention(*layer));
                }
                Block::Neurons { layer, .. } | Block::Features { layer, .. } | Block::Slices { layer, .. } => {
                    out.insert(SharedSite::Mlp(*layer));
                }
            }
            out.insert(SharedSite::Input(block.site()));
            out.insert(SharedSite::Stream(block.site()));
        }
        out.into_iter().collect()
    }

    /// One experiment of operations drawn as interchange draws them (`draw_site_ops`, a family
    /// uniform among swap, zero, scale, push, cut) on the sites `sites`, its position on sequences of
    /// the pool's length (512 without a pool); `None` when no operation of the family fits them.
    pub fn draw_at(&self, rng: &mut impl RngExt, sites: &[SharedSite], weights: &Weights) -> Option<SiteDraw> {
        let (_, head_blocks) = Self::shared(weights);
        let families = [interchange::Family::Swap, interchange::Family::Zero, interchange::Family::Scale, interchange::Family::Push, interchange::Family::Cut];
        let family = families[rng.random_range(0..families.len())];
        let length = self.pool.first().map_or(512, |d| d.length);
        match interchange::draw_site_ops(rng, family, length, sites, &head_blocks, &self.typical, self.directions.len(), 2 * weights.layers.len()) {
            Ok((interchange::Patch::Ops { family, ops }, position)) if !ops.is_empty() => Some(SiteDraw { family, ops, position, length }),
            _ => None,
        }
    }

    /// Each shared site's typical norm on `M` (unedited `weights`) over `sequences`: the root mean
    /// square of its rows' norms after each sequence's first token, as
    /// `Interchange::measure_typical` measures it (heads' and attentions' outputs and MLPs' outputs
    /// as their writes into the stream, a block's input as its normed input, the stream after a block,
    /// the embeddings).
    pub fn measure_typical(weights: &Weights, stats: &Stats, sequences: &[Vec<u32>]) -> Result<BTreeMap<SharedSite, f64>, String> {
        let batch = Batch::new(sequences)?;
        let all: BTreeSet<(usize, usize)> = weights.layers.iter().enumerate().flat_map(|(l, layer)| (0..layer.heads.len()).map(move |h| (l, h))).collect();
        let circuit = Graph::empty().model(weights).split_heads(&all);
        let blocks = 2 * weights.layers.len();
        let run = execute_with(weights, stats, &circuit, &batch, &[], &BTreeMap::new(), false, &Interventions::recording((0..blocks).collect()))?;
        let later: Vec<usize> = batch.spans.iter().flat_map(|&(start, n)| start + 1..start + n).collect();
        let typical = |x: &Array2<f64>| -> f64 { (later.iter().map(|&r| x.row(r).dot(&x.row(r))).sum::<f64>() / later.len().max(1) as f64).sqrt() };
        let mut out = BTreeMap::new();
        let mut stream = weights.embedding.select(Axis(0), &batch.tokens.iter().map(|t| *t as usize).collect::<Vec<_>>());
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
    pub fn write_manifest(path: &std::path::Path, export: &str, weights: &Weights, stats: &Stats, sequences: &[Vec<u32>], families: &[interchange::Family], count: usize, seed: u64, length: usize) -> Result<Self, String> {
        if path.exists() {
            return Err(format!("{} exists and is immutable", path.display()));
        }
        let typical = Self::measure_typical(weights, stats, sequences)?;
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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prompt {
    #[serde(default)]
    pub text: String,
    pub token_ids: Vec<u32>,
    pub target_positions: Vec<usize>,
    #[serde(default)]
    pub counterfactual: Option<Counterfactual>,
    /// Token ranges whose attention is masked, `[query_start, query_end, key_start, key_end]`:
    /// queries in the first range do not attend to keys in the second (the prompt's and its
    /// counterfactual's, for `M` and every program, in every experiment).
    #[serde(default)]
    pub attention_block: Vec<[usize; 4]>,
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
    /// `M`'s outcome per experiment key (log-probabilities in float64: exact scores, an empty or full
    /// program's error to the last bits) and the keys in the order
    /// they were stored: the oldest are dropped past `cache_bytes`.
    cache: BTreeMap<String, Arc<Array2<f64>>>,
    cached: std::collections::VecDeque<String>,
    pub cache_bytes: usize,
    /// The directory of the disk cache of `M`'s outcomes, which every checker process given the
    /// same directory shares; `None` (the default) keeps them in memory only.
    pub disk_cache: Option<std::path::PathBuf>,
    /// Heads by their measured removal effect on `M` (mean `KL(M ‖ M without the head)` at the
    /// targets), strongest first; measured on first use.
    strongest: Option<Vec<(usize, usize)>>,
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
    references: std::sync::Mutex<Vec<(String, Arc<std::sync::OnceLock<Result<Arc<Reference>, String>>>)>>,
    /// Counterfactual runs under site operations (`reference_under`) by experiment, for one
    /// [`Checker::score_batch`] (cleared at its start).
    site_references: std::sync::Mutex<BTreeMap<String, Arc<std::sync::OnceLock<Result<Arc<Reference>, String>>>>>,
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
            stats,
            clean: (batch, rows),
            counterfactual,
            donors,
            cache: BTreeMap::new(),
            cached: Default::default(),
            cache_bytes: CACHE_BYTES,
            disk_cache: None,
            strongest: None,
            sites: SiteUnits::default(),
            uniform_seeds: None,
            partners,
            blocks,
            edit: None,
            references: std::sync::Mutex::new(Vec::new()),
            site_references: std::sync::Mutex::new(BTreeMap::new()),
        })
    }

    /// `batch` with its stand-in source attached: under counterfactual stand-ins, `M`'s run (with
    /// the current weights) on each sequence's partner (a prompt's counterfactual, a
    /// counterfactual's prompt). `M` itself (every unit computing, every edge kept) reads no
    /// stand-in and gets none.
    pub fn referenced(&self, circuit: &Circuit, batch: &Batch) -> Result<Batch, String> {
        let mut out = batch.clone();
        out.reference = None;
        let masked = |b: &mut Batch| b.blocks = b.sequences().iter().map(|s| self.blocks.get(s).cloned().unwrap_or_default()).collect();
        masked(&mut out);
        if !circuit.counterfactual || circuit.is_model() {
            return Ok(out);
        }
        let partner: Vec<Vec<u32>> = batch.sequences().iter().map(|s| self.partners.get(s).cloned().ok_or("counterfactual stand-ins need each prompt's counterfactual of the same length")).collect::<Result<_, _>>()?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&partner, &mut hasher);
        let key = format!("{} {}", self.edit.as_deref().unwrap_or(""), std::hash::Hasher::finish(&hasher));
        let cell = {
            let mut cache = self.references.lock().map_err(|e| e.to_string())?;
            match cache.iter().find(|(k, _)| *k == key) {
                Some((_, c)) => c.clone(),
                None => {
                    // A few batches recur per edit (the prompts, their counterfactuals, a swap's subset).
                    if cache.len() >= 6 {
                        cache.remove(0);
                    }
                    let c = Arc::new(std::sync::OnceLock::new());
                    cache.push((key, c.clone()));
                    c
                }
            }
        };
        let r = cell
            .get_or_init(|| {
                let mut b = Batch::new(&partner)?;
                masked(&mut b);
                reference(&self.weights, &self.stats, &b).map(Arc::new)
            })
            .clone()?;
        out.reference = Some(r);
        Ok(out)
    }

    /// Sets the weight edit applied now (`None` after restoring) and drops the counterfactual runs
    /// of any other edit.
    fn set_edit(&mut self, e: &Experiment) {
        self.edit = match e {
            Experiment::Edit { edit, .. } => serde_json::to_string(edit).ok(),
            _ => None,
        };
        if let Ok(mut cache) = self.references.lock() {
            cache.retain(|(k, _)| k.starts_with(' '));
        }
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
            Experiment::Edit { edit, .. } => format!("edit {}", serde_json::to_string(edit).unwrap_or_default()),
            other => serde_json::to_string(other).unwrap_or_default(),
        }
    }

    /// One circuit under one experiment: its log-probabilities at the scored rows (the swap's
    /// prompts only, for a swap).
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
        let mut circuit = circuit.clone();
        let (batch, rows) = match e {
            Experiment::Counterfactual => self.counterfactual.as_ref().ok_or("the behavior has no counterfactuals")?,
            _ => &self.clean,
        };
        {
            match e {
                Experiment::Cut { from, to, route, .. } => {
                    match to {
                        Some(r) => circuit.units[*r].routes[route.slot()].cut(*from),
                        None => circuit.logits.cut(*from),
                    }
                    Ok(execute(&self.weights, &self.stats, &circuit, &self.referenced(&circuit, batch)?, rows, &BTreeMap::new(), false)?.log_probabilities)
                }
                Experiment::Swap { node } => {
                    if self.donors.is_empty() {
                        return Err("no two prompts of one length to swap between".into());
                    }
                    let prompts: Vec<Vec<u32>> = self.donors.iter().map(|(i, _)| self.behavior.prompts[*i].token_ids.clone()).collect();
                    let donors: Vec<Vec<u32>> = self.donors.iter().map(|(_, j)| self.behavior.prompts[*j].token_ids.clone()).collect();
                    let (base, donor) = (Batch::new(&prompts)?, Batch::new(&donors)?);
                    let donor_run = execute(&self.weights, &self.stats, &circuit, &self.referenced(&circuit, &donor)?, &[], &BTreeMap::new(), false)?;
                    let value = donor_run.writes[*node].clone().ok_or("the swapped node does not compute")?;
                    let targets: Vec<(usize, &[usize])> = self.donors.iter().enumerate().map(|(k, (i, _))| (k, self.behavior.prompts[*i].target_positions.as_slice())).collect();
                    let rows = scored_rows(&base, &targets)?;
                    Ok(execute(&self.weights, &self.stats, &circuit, &self.referenced(&circuit, &base)?, &rows, &[(*node, value)].into(), false)?.log_probabilities)
                }
                Experiment::Sites { draw } => self.sites_outcome(&circuit, draw),
                _ => Ok(execute(&self.weights, &self.stats, &circuit, &self.referenced(&circuit, batch)?, rows, &BTreeMap::new(), false)?.log_probabilities),
            }
        }
    }

    /// `M`'s cache key of `e` on `graph`'s units.
    fn model_key(&self, graph: &Graph, e: &Experiment) -> String {
        // A cut hands the reader the writer's stand-in, so the stand-in form is part of the key.
        let form = match (graph.counterfactual, self.stats.mean_output) {
            (true, _) => "counterfactual; ",
            (false, true) => "mean output; ",
            (false, false) => "",
        };
        format!("{form}{}", Self::key(graph, e))
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
        let key = self.model_key(graph, e);
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
    /// every checker process given it): named by the behavior, a fingerprint of its prompts and of `M`'s weights,
    /// and the key; `None` when the cache is off.
    fn disk_path(&self, key: &str) -> Option<std::path::PathBuf> {
        let dir = self.disk_cache.as_ref()?;
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
        let mut sample: Vec<f64> = w.embedding.row(0).to_vec();
        sample.extend(w.unembedding.row(w.unembedding.nrows() - 1).iter());
        for l in &w.layers {
            sample.extend(l.mlp.iter().flat_map(|m| m.out.row(0).to_vec()));
        }
        let h = fnv(prompts, &mut sample.iter().map(|v| v.to_bits()).chain(key.bytes().map(u64::from)));
        Some(std::path::Path::new(&dir).join(format!("{}_{prompts:016x}", self.behavior.id)).join(format!("{h:016x}.f64")))
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
        let n = n.unwrap_or_else(|| self.behavior.size());
        let seed = self.uniform_seeds.map_or(seed, |m| seed % m.max(1));
        let strongest = self.strongest()?;
        self.site_references.lock().map_err(|e| e.to_string())?.clear();
        let parsed: Vec<(Graph, bool, Option<String>)> = programs
            .iter()
            .map(|program| match Graph::parse(program, &self.weights) {
                Ok(g) => (g, true, None),
                Err(e) => (Graph::empty(), false, Some(e)),
            })
            .collect();
        for (g, _, _) in &parsed {
            self.weights.load_features(g)?;
        }
        let circuits: Vec<Circuit> = parsed.iter().map(|(g, _, _)| g.program(&self.weights, edges)).collect();
        let models: Vec<Circuit> = parsed.iter().map(|(g, _, _)| g.model(&self.weights)).collect();
        // Per program its experiments; per run (program, experiment) the cache key of M's outcome.
        let mut runs: Vec<(usize, Experiment, String)> = Vec::new();
        let mut drawn = vec![0usize; programs.len()];
        for (i, (graph, _, _)) in parsed.iter().enumerate() {
            for e in sample(&self.weights, graph, self.counterfactual.is_some(), count, seed, &strongest, &self.sites) {
                if matches!(e, Experiment::Swap { .. }) && self.donors.is_empty() {
                    continue;
                }
                let key = self.model_key(graph, &e);
                runs.push((i, e, key));
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
        for members in groups.values() {
            let restore = match &runs[members[0]].1 {
                Experiment::Edit { edit, .. } => Some(edit.apply(&mut self.weights)?),
                _ => None,
            };
            self.set_edit(&runs[members[0]].1.clone());
            let result = (|| -> Result<(), String> {
                // At most CHUNK distinct experiments at a time: their counterfactual runs under site
                // operations (about 200 MB each on vpd4l) are dropped between chunks.
                const CHUNK: usize = 8;
                let mut order: Vec<&str> = Vec::new();
                let mut chunks: Vec<Vec<usize>> = Vec::new();
                for &r in members {
                    let k = runs[r].2.as_str();
                    let at = order.iter().position(|x| *x == k).unwrap_or_else(|| {
                        order.push(k);
                        order.len() - 1
                    });
                    if chunks.len() <= at / CHUNK {
                        chunks.push(Vec::new());
                    }
                    chunks[at / CHUNK].push(r);
                }
                for members in &chunks {
                    // M once per key not yet cached, in parallel.
                    let mut missing: Vec<usize> = Vec::new();
                    for &r in members {
                        if !self.cache.contains_key(&runs[r].2) && !missing.iter().any(|&m| runs[m].2 == runs[r].2) {
                            missing.push(r);
                        }
                    }
                    let this = &*self;
                    let compute_m = |&r: &usize| -> Result<(String, Arc<Array2<f64>>), String> {
                            let key = &runs[r].2;
                            if let Some(m) = this.disk_get(key) {
                                return Ok((key.clone(), Arc::new(m)));
                            }
                            let m = this.run(&models[runs[r].0], &runs[r].1)?;
                            this.disk_put(key, &m);
                            Ok((key.clone(), Arc::new(m)))
                    };
                    // One run alone goes on this thread, so its products spread over the pool (par_dot).
                    let made: Vec<(String, Arc<Array2<f64>>)> = if missing.len() == 1 { missing.iter().map(compute_m).collect::<Result<_, String>>()? } else { missing.par_iter().map(compute_m).collect::<Result<_, String>>()? };
                    let fresh: BTreeMap<String, Arc<Array2<f64>>> = made.iter().cloned().collect();
                    let this = &*self;
                    let score_p = |&r: &usize| -> Result<(usize, Vec<f64>, Option<Candidates>), String> {
                            let (i, e, key) = &runs[r];
                            let m = fresh.get(key).or_else(|| this.cache.get(key)).ok_or("M's outcome went missing")?;
                            let p = this.run(&circuits[*i], e)?;
                            let kl = kl_bits(m, &p);
                            let candidates = clean.as_ref().map(|c| Candidates::of(c, m, &p, &this.rows_of(e), &this.rows_of(&Experiment::Clean), top));
                            Ok((r, kl, candidates))
                    };
                    let scored: Vec<(usize, Vec<f64>, Option<Candidates>)> = if members.len() == 1 { members.iter().map(score_p).collect::<Result<_, String>>()? } else { members.par_iter().map(score_p).collect::<Result<_, String>>()? };
                    for (key, m) in made {
                        self.keep(key, m);
                    }
                    for (r, kl, candidates) in scored {
                        measured[r] = Some((kl, candidates));
                    }
                    self.site_references.lock().map_err(|e| e.to_string())?.clear();
                }
                Ok(())
            })();
            self.set_edit(&Experiment::Clean);
            if let Some(r) = restore {
                r.restore(&mut self.weights)?;
            }
            result?;
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
            let exec_error_bits = n * total.0 / total.1.max(1) as f64;
            let code_bits = if *valid && program.token_types > 1 { program.python_tokens as f64 * (program.token_types as f64).log2() } else { 0.0 };
            let opaque_numbers = graph.opaque_numbers(&self.weights);
            let opaque_bits = 0.5 * n.log2() * opaque_numbers as f64;
            let score = Score {
                total_bits: exec_error_bits + code_bits + opaque_bits,
                exec_error_bits,
                reader_error_bits: 0.0,
                code_bits,
                python_tokens: if *valid { program.python_tokens } else { 0 },
                opaque_numbers,
                opaque_bits,
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

    /// Heads by measured removal effect, strongest first (cached).
    pub fn strongest(&mut self) -> Result<Vec<(usize, usize)>, String> {
        if let Some(s) = &self.strongest {
            return Ok(s.clone());
        }
        let graph = Graph::empty();
        let clean = self.model_outcome(&graph, &Experiment::Clean)?;
        // A head's output zeroed at every token is the head removed (graph_sites_tests), and site
        // operations leave the weights alone, so every head runs in parallel.
        let heads: Vec<(usize, usize)> = self.weights.layers.iter().enumerate().flat_map(|(l, layer)| (0..layer.heads.len()).map(move |h| (l, h))).collect();
        let model = graph.model(&self.weights);
        let this = &*self;
        let mut effects: Vec<((usize, usize), f64)> = heads
            .par_iter()
            .enumerate()
            .map(|(k, &lh)| {
                let draw = SiteDraw { family: interchange::Family::Zero, ops: vec![SiteOp { site: SharedSite::Head(k), operation: Operation::Scale(0), onward: true }], position: 0, length: 1 };
                let removed = this.run(&model, &Experiment::Sites { draw })?;
                let kl = kl_bits(&clean, &removed);
                Ok((lh, kl.iter().sum::<f64>() / kl.len().max(1) as f64))
            })
            .collect::<Result<_, String>>()?;
        effects.sort_by(|a, b| b.1.total_cmp(&a.1));
        let order: Vec<(usize, usize)> = effects.into_iter().map(|(k, _)| k).collect();
        self.strongest = Some(order.clone());
        Ok(order)
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
        if !donor || self.counterfactual_donors() {
            let base = Batch::new(&all)?;
            let rows = self.clean.1.clone();
            let donor = if donor { Some(Batch::new(&prompts.iter().map(|p| p.counterfactual.as_ref().map(|c| c.token_ids.clone()).unwrap_or_default()).collect::<Vec<_>>())?) } else { None };
            return Ok((base, rows, donor));
        }
        if self.donors.is_empty() {
            return Err("site operations with a donor need same-length counterfactuals or two prompts of one length".into());
        }
        let base = Batch::new(&self.donors.iter().map(|(i, _)| prompts[*i].token_ids.clone()).collect::<Vec<_>>())?;
        let donors = Batch::new(&self.donors.iter().map(|(_, j)| prompts[*j].token_ids.clone()).collect::<Vec<_>>())?;
        let targets: Vec<(usize, &[usize])> = self.donors.iter().enumerate().map(|(k, (i, _))| (k, prompts[*i].target_positions.as_slice())).collect();
        let rows = scored_rows(&base, &targets)?;
        Ok((base, rows, Some(donors)))
    }

    /// `circuit` under site operations `draw` ([`run_sites`]): its log-probabilities at the
    /// scored rows.
    fn sites_outcome(&self, circuit: &Circuit, draw: &SiteDraw) -> Result<Array2<f64>, String> {
        let (base, rows, donor) = self.site_batches(Interventions::needs_donor(draw))?;
        let (mut base, donor) = (self.referenced(circuit, &base)?, donor.map(|d| self.referenced(circuit, &d)).transpose()?);
        // The stand-ins' run on the counterfactuals takes the same operations (reference_under).
        if base.reference.is_some() {
            let partner: Vec<Vec<u32>> = base.sequences().iter().map(|s| self.partners.get(s).cloned().ok_or("a prompt without a counterfactual")).collect::<Result<_, _>>()?;
            let key = format!("{} {}", serde_json::to_string(draw).map_err(|e| e.to_string())?, partner.len());
            let cell = {
                let mut cache = self.site_references.lock().map_err(|e| e.to_string())?;
                cache.entry(key).or_insert_with(|| Arc::new(std::sync::OnceLock::new())).clone()
            };
            let r = cell.get_or_init(|| Batch::new(&partner).and_then(|b| reference_under(&self.weights, &self.stats, &b, draw, &self.sites)).map(Arc::new)).clone()?;
            base.reference = Some(r);
        }
        run_sites(&self.weights, &self.stats, circuit, (&base, &rows), donor.as_ref(), draw, &self.sites)
    }

    /// An experiment's scored tokens as (prompt, position), in its rows' order: a node swap's and a
    /// donor-reading site operation's without same-length counterfactuals are the prompts with a
    /// same-length donor; a counterfactual's positions are the prompt's.
    pub fn rows_of(&self, e: &Experiment) -> Vec<(usize, usize)> {
        let subset = match e {
            Experiment::Swap { .. } => true,
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

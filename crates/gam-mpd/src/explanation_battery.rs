//! The common evaluation battery (#2951): one held-out scoring of an explanation `E` of a language
//! model `M`, with the same definitions whatever `E` is: a library explanation (`library_mdl`) or
//! VPD's decomposition (parameter subcomponents gated by a causal-importance network).
//!
//! Every divergence is `KL(M ‖ E)` of the next-token distributions per token, in bits; cross-entropy
//! is in nats per predicted token; top-1 agreement is the fraction of tokens where `E`'s most
//! probable token is `M`'s. Per-token summaries are the mean and the quantiles over every token.
//!
//! # Protocols
//!
//! A protocol says which of `M`'s layers run as `E`'s (VPD's appendix B.1): every nonempty layer
//! subset `S` with the others `M`'s, each layer reading the stream before it (`S` all layers is
//! error-propagating, one layer is single-layer, a prefix is a cut), and clean-input (every layer of
//! `E` reading `M`'s stream entering it, the final stream `M`'s embedding plus each layer's
//! increment).
//!
//! # VPD's decomposition
//!
//! VPD writes each of `M`'s weight matrices `W` (a site: a layer's query, key, value and output
//! maps, its MLP's up and down maps) as `C` rank-one subcomponents `V_c U_cᵀ` and a remainder
//! `Δ = W − (V U)ᵀ`; the site computes `((x V) ⊙ m) U + δ x Δᵀ`, with a mask `m ∈ [0, 1]^C` per token
//! and `δ ∈ [0, 1]` per token. Its causal-importance network reads `M`'s own inputs of every site
//! (each RMS-normed), runs a bidirectional transformer and writes `g ∈ [0, 1]^C` per site and token
//! (VPD's intended setting). Masks ([`Strategy`]): the CI values `m = g`; rounded `m = 1[g > 0]`;
//! stochastic `m = g + (1 − g) u`, `δ = u′` with `u, u′` uniform; unmasked `m = 1`. Only the
//! stochastic masks keep the remainder (`δ = 0` otherwise, as in VPD's evaluation); a layer run as
//! `M`'s has `m = 1`, `δ = 1`, which is `M`'s map.
//!
//! The decomposition is read from its one-time export (`bench/vpd_2951/vpd_export.py`): per site
//! `{site}.U` (`C × d_out`) and `{site}.V` (`d_in × C`), and the network's weights; `M`'s from its
//! engine export. Both run as operator programs ([`Model`]): `M`'s forward with each site's masks as
//! raw slots, and `M`'s forward followed by the causal-importance network.

use crate::{
    artifact::Artifact,
    device_program::{DeviceProgram, DeviceTrace},
    interchange,
    library_mdl::sequence_family,
    operator_program::{
        Basis, Declarations, Domain, FamilyInputs, Group, Interface, LabelKind, Law, Node, Operator, OperatorProgram, Provenance, Rotary, Scale,
        Slot, exact_precision,
    },
    run_check::LayerNodes,
};
use gam_gpu::tensor::{Arithmetic, Device, Op, Tensor};
use gam_math::categorical::log_softmax;
use ndarray::{Array1, Array2, Axis, s};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    f64::consts::LN_2,
    path::{Path, PathBuf},
    sync::Arc,
};

fn error(e: impl std::fmt::Display) -> String {
    format!("explanation battery: {e}")
}

// ------------------------------------------------------------------------------ exports

/// An engine export: `export.json` and its float64 tensors.
struct Export {
    dir: PathBuf,
    record: Value,
}

impl Export {
    fn open(dir: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(dir.join("export.json")).map_err(|e| error(format!("{}: {e}", dir.display())))?;
        Ok(Self { dir: dir.to_path_buf(), record: serde_json::from_str(&text).map_err(error)? })
    }

    fn tensor(&self, name: &str) -> Result<Array2<f64>, String> {
        let dims: Vec<usize> = self.record["files"][name]["shape"]
            .as_array()
            .ok_or_else(|| error(format!("{name}: no shape")))?
            .iter()
            .map(|v| v.as_u64().map(|v| v as usize).ok_or_else(|| error(format!("{name}: a shape entry"))))
            .collect::<Result<_, _>>()?;
        let (rows, cols) = match dims[..] {
            [n] => (1, n),
            [r, c] => (r, c),
            _ => return Err(error(format!("{name}: shape {dims:?}"))),
        };
        crate::import::read_f64_shaped(&self.dir.join(format!("{name}.f64")), rows, cols)
    }

    fn count(&self, key: &str) -> Result<usize, String> {
        self.record["config"][key].as_u64().map(|v| v as usize).ok_or_else(|| error(format!("config.{key}")))
    }
}

/// The weight matrices of one layer of `M`, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Query,
    Key,
    Value,
    Output,
    Up,
    Down,
}

pub const KINDS: [Kind; 6] = [Kind::Query, Kind::Key, Kind::Value, Kind::Output, Kind::Up, Kind::Down];

impl Kind {
    fn export_name(self) -> &'static str {
        match self {
            Self::Query => "attn.q_proj",
            Self::Key => "attn.k_proj",
            Self::Value => "attn.v_proj",
            Self::Output => "attn.o_proj",
            Self::Up => "mlp.c_fc",
            Self::Down => "mlp.down_proj",
        }
    }

    /// The block whose read the site's map reads: 0 the attention's normed stream (and the
    /// output map, the heads' reads), 1 the MLP's.
    pub fn block(self) -> usize {
        match self {
            Self::Query | Self::Key | Self::Value | Self::Output => 0,
            Self::Up | Self::Down => 1,
        }
    }
}

/// `M`'s configuration: a pre-norm rotary transformer with tied embeddings, one pointwise MLP law
/// and no biases (the 4-layer Pile model's architecture).
#[derive(Clone, Debug)]
struct Config {
    d: usize,
    layers: usize,
    heads: usize,
    head_dim: usize,
    vocab: usize,
    hidden: usize,
    rotary: Rotary,
    epsilon: f64,
    law: Law,
}

impl Config {
    fn of(export: &Export) -> Result<Self, String> {
        let config = &export.record["config"];
        let flag = |key: &str| config[key].as_bool().unwrap_or(false);
        if flag("parallel_residual") || flag("qk_norm") || flag("mlp_gated") || !config["tied_embeddings"].as_bool().unwrap_or(true) {
            return Err(error("the battery's model is sequential, without head norms or gates, with tied embeddings"));
        }
        if config["rope_pairing"].as_str() != Some("rotate_half") || config["norm"].as_str().is_some_and(|n| n != "rms") {
            return Err(error("the battery's model rotates halves and has RMS norms"));
        }
        let (heads, kv) = (export.count("n_heads")?, export.count("n_kv_heads")?);
        if heads != kv {
            return Err(error("the battery's model has one key and value per query head"));
        }
        let law = match config["mlp_act"].as_str() {
            Some("gelu_tanh") => Law::GeluTanh,
            Some("gelu") => Law::Gelu,
            Some("relu") => Law::Relu,
            Some("silu") => Law::Silu,
            other => return Err(error(format!("MLP law {other:?}"))),
        };
        let theta = config["rope_theta"].as_f64().ok_or_else(|| error("config.rope_theta"))?;
        if theta.fract() != 0.0 || theta <= 0.0 || theta > f64::from(u32::MAX) {
            return Err(error("rope_theta is not a positive integer"));
        }
        let head_dim = export.count("head_dim")?;
        let d = export.count("d_model")?;
        let hidden = export.tensor("blocks.0.mlp.c_fc")?.nrows();
        Ok(Self {
            d,
            layers: export.count("n_layers")?,
            heads,
            head_dim,
            vocab: export.count("vocab")?,
            hidden,
            rotary: Rotary { base: theta as u32, dims: head_dim as u32, half_split: true },
            epsilon: config["norm_eps"].as_f64().ok_or_else(|| error("config.norm_eps"))?,
            law,
        })
    }
}

/// One site's subcomponents: `u` (`C × d_out`) and `v` (`d_in × C`).
pub struct Factors {
    pub name: String,
    pub layer: usize,
    pub kind: Kind,
    pub u: Array2<f64>,
    pub v: Array2<f64>,
}

impl Factors {
    pub fn subcomponents(&self) -> usize {
        self.u.nrows()
    }
}

struct CiBlock {
    q: Array2<f64>,
    k: Array2<f64>,
    v: Array2<f64>,
    o: Array2<f64>,
    fc1: (Array2<f64>, Array2<f64>),
    fc2: (Array2<f64>, Array2<f64>),
}

/// VPD's causal-importance network: its sites' inputs in `order` (indices into the sites), an
/// input projection, bidirectional pre-norm blocks with an exact-GELU MLP, and a head whose
/// outputs, in `order`, are clamped to `[0, 1]`.
struct CiNetwork {
    order: Vec<usize>,
    input: (Array2<f64>, Array2<f64>),
    blocks: Vec<CiBlock>,
    head: (Array2<f64>, Array2<f64>),
    heads: usize,
    head_dim: usize,
    rope_base: u32,
    epsilon: f64,
}

/// VPD's decomposition of `M`: per site its subcomponents, in `M`'s order (per layer [`KINDS`]),
/// and its causal-importance network.
pub struct Decomposition {
    pub sites: Vec<Factors>,
    ci: CiNetwork,
}

/// The subcomponents of every site of VPD's exported decomposition, in `M`'s order.
pub fn load_factors(dir: &Path) -> Result<Vec<Factors>, String> {
    factors_of(&Export::open(dir)?)
}

fn factors_of(export: &Export) -> Result<Vec<Factors>, String> {
    let names = site_names(export)?;
    let mut sites = Vec::with_capacity(names.len());
    for (i, name) in names.iter().enumerate() {
        let (layer, kind) = (i / KINDS.len(), KINDS[i % KINDS.len()]);
        if *name != format!("h.{layer}.{}", kind.export_name()) {
            return Err(error(format!("site {i} is {name}, not layer {layer}'s {kind:?} in M's order")));
        }
        let (u, v) = (export.tensor(&format!("{name}.U"))?, export.tensor(&format!("{name}.V"))?);
        if u.nrows() != v.ncols() {
            return Err(error(format!("{name}: U and V disagree on the subcomponents")));
        }
        sites.push(Factors { name: name.clone(), layer, kind, u, v });
    }
    Ok(sites)
}

fn site_names(export: &Export) -> Result<Vec<String>, String> {
    export.record["config"]["sites"]
        .as_array()
        .ok_or_else(|| error("config.sites"))?
        .iter()
        .map(|v| v.as_str().map(str::to_string).ok_or_else(|| error("a site name")))
        .collect()
}

impl Decomposition {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let export = Export::open(dir)?;
        let config = &export.record["config"];
        let names = site_names(&export)?;
        let sites = factors_of(&export)?;
        let ci = &config["ci"];
        let order: Vec<usize> = ci["order"]
            .as_array()
            .ok_or_else(|| error("config.ci.order"))?
            .iter()
            .map(|v| v.as_str().and_then(|n| names.iter().position(|m| m == n)).ok_or_else(|| error("a site of the network's order")))
            .collect::<Result<_, _>>()?;
        let count = |key: &str| ci[key].as_u64().map(|v| v as usize).ok_or_else(|| error(format!("config.ci.{key}")));
        if ci["activation"].as_str() != Some("gelu") || ci["output"].as_str() != Some("clamp01") {
            return Err(error("the network's law is the exact GELU and its output a clamp to [0, 1]"));
        }
        let blocks = (0..count("blocks")?)
            .map(|i| -> Result<CiBlock, String> {
                let t = |part: &str| export.tensor(&format!("ci.blocks.{i}.{part}"));
                Ok(CiBlock { q: t("q")?, k: t("k")?, v: t("v")?, o: t("o")?, fc1: (t("fc1.W")?, t("fc1.b")?), fc2: (t("fc2.W")?, t("fc2.b")?) })
            })
            .collect::<Result<_, _>>()?;
        let rope_base = count("rope_base")?;
        let ci = CiNetwork {
            order,
            input: (export.tensor("ci.input.W")?, export.tensor("ci.input.b")?),
            blocks,
            head: (export.tensor("ci.head.W")?, export.tensor("ci.head.b")?),
            heads: count("heads")?,
            head_dim: count("head_dim")?,
            rope_base: u32::try_from(rope_base).map_err(error)?,
            epsilon: ci["epsilon"].as_f64().ok_or_else(|| error("config.ci.epsilon"))?,
        };
        Ok(Self { sites, ci })
    }
}

// ------------------------------------------------------------------------------ programs

struct Builder {
    operators: Vec<Arc<Operator>>,
    nodes: Vec<Node>,
    slots: Vec<Slot>,
}

impl Builder {
    fn dense(&mut self, name: &str, rows: &Interface, cols: &Interface, values: Array2<f64>) -> Result<usize, String> {
        let precision = exact_precision(values.iter().copied()).map_err(error)?;
        let op = Operator::dense(name, rows.clone(), cols.clone(), values, precision, Provenance::native(name)).map_err(error)?;
        self.operators.push(Arc::new(op));
        Ok(self.operators.len() - 1)
    }

    fn diag(&mut self, name: &str, interface: &Interface, values: Array1<f64>) -> Result<usize, String> {
        let precision = exact_precision(values.iter().copied()).map_err(error)?;
        let op = Operator::diag(name, interface.clone(), values, precision, Provenance::native(name)).map_err(error)?;
        self.operators.push(Arc::new(op));
        Ok(self.operators.len() - 1)
    }

    fn identity(&mut self, name: &str, interface: &Interface) -> usize {
        self.operators.push(Arc::new(Operator::identity(name, interface.clone())));
        self.operators.len() - 1
    }

    fn node(&mut self, node: Node) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    /// A raw slot of `width` and the node reading it.
    fn raw(&mut self, width: usize) -> (usize, usize) {
        self.slots.push(Slot::Raw { width });
        let slot = self.slots.len() - 1;
        (slot, self.node(Node::Raw { slot }))
    }

    fn program(self, vocab: usize, output: usize) -> Result<OperatorProgram, String> {
        let program = OperatorProgram {
            rules: Vec::new(),
            declarations: Declarations { parameters: 0, domains: vec![Domain { size: vocab }], slots: self.slots },
            bases: vec![Basis::Indicator { domain: 0 }],
            operators: self.operators,
            nodes: self.nodes,
            output,
        };
        program.interfaces().map_err(error)?;
        Ok(program)
    }
}

fn uniform(count: usize) -> Result<Interface, String> {
    Interface::uniform(count, 1, LabelKind::Unit, 0).map_err(error)
}

fn native(width: usize) -> Result<Interface, String> {
    Interface::native(width).map_err(error)
}

/// The nodes of a built model program.
#[derive(Clone, Debug)]
pub struct Layout {
    /// The stream entering each layer (layer 0: the embedding), the final residual and the final
    /// normed stream.
    pub streams: Vec<usize>,
    pub residual: usize,
    pub hidden: usize,
    /// Per block (each layer's attention, then its MLP) the normed stream its maps read.
    pub reads: Vec<usize>,
    /// Per site the node its map reads (the output map: the heads' reads, concatenated).
    pub inputs: Vec<usize>,
    /// Per site of a decomposed program: its masked subcomponent activations `(x V) ⊙ m`, and the
    /// raw slots of its mask `m` (width `C`) and of its remainder's `δ` (repeated over the width
    /// it multiplies: a head for the query, key and value maps, the output otherwise).
    pub activations: Vec<usize>,
    pub masks: Vec<usize>,
    pub deltas: Vec<usize>,
    /// The nodes reading those slots.
    pub mask_nodes: Vec<usize>,
    pub delta_nodes: Vec<usize>,
    /// Per layer, each head's attention read (the output map's input), in head order.
    pub head_reads: Vec<Vec<usize>>,
    /// `M`'s read variables as each program computes them: per layer each head's query, key and
    /// value vectors, and the MLP's pre-activations (one column per neuron).
    pub head_variables: Vec<Vec<[usize; 3]>>,
    pub pre_activations: Vec<usize>,
}

/// A model as an operator program through its final normed stream, with its node layout.
pub struct Model {
    pub program: OperatorProgram,
    pub layout: Layout,
}

/// `M`'s forward from its export, each site `M`'s map, or with `factors` VPD's masked
/// subcomponents and remainder (module note), every layer's nodes after the stream entering it.
fn build(export: &Export, config: &Config, factors: Option<&[Factors]>) -> Result<(Builder, Layout), String> {
    let (d, hd, heads) = (config.d, config.head_dim, config.heads);
    let model = uniform(d)?;
    let head = native(hd)?;
    let neurons = uniform(config.hidden)?;
    let tokens = Interface::uniform(config.vocab, 1, LabelKind::Token, 0).map_err(error)?;
    let mut b = Builder { operators: Vec::new(), nodes: Vec::new(), slots: vec![Slot::Token { domain: 0 }] };
    let identity = b.identity("I", &model);
    let head_identity = b.identity("I head", &head);
    let neuron_identity = b.identity("I neurons", &neurons);
    let embedding = b.dense("wte", &model, &tokens, export.tensor("wte")?.t().to_owned())?;
    let norm = |b: &mut Builder, x: usize, name: &str| -> Result<usize, String> {
        let gain = export.tensor(name)?;
        if gain.dim() != (1, d) {
            return Err(error(format!("{name}: not a row of {d}")));
        }
        let op = b.diag(name, &model, gain.row(0).to_owned())?;
        let normed = b.node(Node::RmsNorm { input: x, epsilon: config.epsilon });
        Ok(b.node(Node::Affine { terms: vec![(normed, op)], bias: None }))
    };
    let feature = b.node(Node::Feature { slot: 0, basis: 0 });
    let mut x = b.node(Node::Affine { terms: vec![(feature, embedding)], bias: None });
    let sites = KINDS.len() * config.layers;
    let mut layout = Layout {
        streams: Vec::with_capacity(config.layers),
        residual: 0,
        hidden: 0,
        reads: Vec::with_capacity(2 * config.layers),
        inputs: vec![0; sites],
        activations: Vec::new(),
        masks: Vec::new(),
        deltas: Vec::new(),
        mask_nodes: Vec::new(),
        delta_nodes: Vec::new(),
        head_reads: Vec::new(),
        head_variables: Vec::new(),
        pre_activations: Vec::new(),
    };
    // A decomposed site on `terms` (its input as (node, columns of V's rows) pairs, the columns'
    // interfaces): its masked activations; the caller writes `U` and the remainder.
    struct Decomposed {
        masked: usize,
        delta: usize,
    }
    let decomposed = |b: &mut Builder, layout: &mut Layout, f: &Factors, inputs: &[(usize, std::ops::Range<usize>, Interface)], delta_width: usize| -> Result<Decomposed, String> {
        let c = f.subcomponents();
        let sub = native(c)?;
        let mut terms = Vec::with_capacity(inputs.len());
        for (i, (node, rows, cols)) in inputs.iter().enumerate() {
            let op = b.dense(&format!("{}.V{i}", f.name), &sub, cols, f.v.slice(s![rows.clone(), ..]).t().to_owned())?;
            terms.push((*node, op));
        }
        let inner = b.node(Node::Affine { terms, bias: None });
        let (mask_slot, mask) = b.raw(c);
        let masked = b.node(Node::Hadamard { left: inner, right: mask });
        let (delta_slot, delta) = b.raw(delta_width);
        layout.activations.push(masked);
        layout.masks.push(mask_slot);
        layout.deltas.push(delta_slot);
        layout.mask_nodes.push(mask);
        layout.delta_nodes.push(delta);
        Ok(Decomposed { masked, delta })
    };
    for l in 0..config.layers {
        layout.streams.push(x);
        let prefix = format!("blocks.{l}.");
        let site = |kind: Kind| KINDS.len() * l + KINDS.iter().position(|k| *k == kind).unwrap_or(0);
        let h1 = norm(&mut b, x, &format!("{prefix}rms1.gain"))?;
        layout.reads.push(h1);
        let weight = |kind: Kind| export.tensor(&format!("{prefix}{}", kind.export_name()));
        // The query, key and value of every head.
        let mut per_head: Vec<[usize; 3]> = vec![[0; 3]; heads];
        for (j, kind) in [Kind::Query, Kind::Key, Kind::Value].into_iter().enumerate() {
            let w = weight(kind)?;
            layout.inputs[site(kind)] = h1;
            match factors {
                None => {
                    for (h, out) in per_head.iter_mut().enumerate() {
                        let op = b.dense(&format!("{prefix}{kind:?}{h}"), &head, &model, w.slice(s![h * hd..(h + 1) * hd, ..]).to_owned())?;
                        out[j] = b.node(Node::Affine { terms: vec![(h1, op)], bias: None });
                    }
                }
                Some(factors) => {
                    let f = &factors[site(kind)];
                    let remainder = &w - &f.v.dot(&f.u).t();
                    let site_nodes = decomposed(&mut b, &mut layout, f, &[(h1, 0..d, model.clone())], hd)?;
                    for (h, out) in per_head.iter_mut().enumerate() {
                        let rows = h * hd..(h + 1) * hd;
                        let u = b.dense(&format!("{}.U{h}", f.name), &head, &native(f.subcomponents())?, f.u.slice(s![.., rows.clone()]).t().to_owned())?;
                        let r = b.dense(&format!("{}.R{h}", f.name), &head, &model, remainder.slice(s![rows, ..]).to_owned())?;
                        let rest = b.node(Node::Affine { terms: vec![(h1, r)], bias: None });
                        let kept = b.node(Node::Hadamard { left: rest, right: site_nodes.delta });
                        out[j] = b.node(Node::Affine { terms: vec![(site_nodes.masked, u), (kept, head_identity)], bias: None });
                    }
                }
            }
        }
        let reads: Vec<usize> = per_head
            .iter()
            .map(|[q, k, v]| b.node(Node::Attend { query: *q, key: *k, value: *v, scale: Scale::InverseSqrt(hd as u32), rotary: Some(config.rotary), causal: true }))
            .collect();
        layout.head_reads.push(reads.clone());
        layout.head_variables.push(per_head.clone());
        let concatenated = b.node(Node::Concat { parts: reads.clone() });
        layout.inputs[site(Kind::Output)] = concatenated;
        let wo = weight(Kind::Output)?;
        let attended = match factors {
            None => {
                let mut terms = vec![(x, identity)];
                for (h, read) in reads.iter().enumerate() {
                    let op = b.dense(&format!("{prefix}o{h}"), &model, &head, wo.slice(s![.., h * hd..(h + 1) * hd]).to_owned())?;
                    terms.push((*read, op));
                }
                b.node(Node::Affine { terms, bias: None })
            }
            Some(factors) => {
                let f = &factors[site(Kind::Output)];
                let remainder = &wo - &f.v.dot(&f.u).t();
                let inputs: Vec<(usize, std::ops::Range<usize>, Interface)> = reads.iter().enumerate().map(|(h, r)| (*r, h * hd..(h + 1) * hd, head.clone())).collect();
                let site_nodes = decomposed(&mut b, &mut layout, f, &inputs, d)?;
                let mut rest_terms = Vec::with_capacity(heads);
                for (h, read) in reads.iter().enumerate() {
                    let r = b.dense(&format!("{}.R{h}", f.name), &model, &head, remainder.slice(s![.., h * hd..(h + 1) * hd]).to_owned())?;
                    rest_terms.push((*read, r));
                }
                let rest = b.node(Node::Affine { terms: rest_terms, bias: None });
                let kept = b.node(Node::Hadamard { left: rest, right: site_nodes.delta });
                let u = b.dense(&format!("{}.U", f.name), &model, &native(f.subcomponents())?, f.u.t().to_owned())?;
                b.node(Node::Affine { terms: vec![(x, identity), (site_nodes.masked, u), (kept, identity)], bias: None })
            }
        };
        let h2 = norm(&mut b, attended, &format!("{prefix}rms2.gain"))?;
        layout.reads.push(h2);
        layout.inputs[site(Kind::Up)] = h2;
        let (wc, wd) = (weight(Kind::Up)?, weight(Kind::Down)?);
        let pre = match factors {
            None => {
                let op = b.dense(&format!("{prefix}c_fc"), &neurons, &model, wc)?;
                b.node(Node::Affine { terms: vec![(h2, op)], bias: None })
            }
            Some(factors) => {
                let f = &factors[site(Kind::Up)];
                let remainder = &wc - &f.v.dot(&f.u).t();
                let site_nodes = decomposed(&mut b, &mut layout, f, &[(h2, 0..d, model.clone())], config.hidden)?;
                let r = b.dense(&format!("{}.R", f.name), &neurons, &model, remainder)?;
                let rest = b.node(Node::Affine { terms: vec![(h2, r)], bias: None });
                let kept = b.node(Node::Hadamard { left: rest, right: site_nodes.delta });
                let u = b.dense(&format!("{}.U", f.name), &neurons, &native(f.subcomponents())?, f.u.t().to_owned())?;
                b.node(Node::Affine { terms: vec![(site_nodes.masked, u), (kept, neuron_identity)], bias: None })
            }
        };
        layout.pre_activations.push(pre);
        let active = b.node(Node::Pointwise { input: pre, laws: vec![config.law; neurons.groups().len()] });
        layout.inputs[site(Kind::Down)] = active;
        x = match factors {
            None => {
                let op = b.dense(&format!("{prefix}down_proj"), &model, &neurons, wd)?;
                b.node(Node::Affine { terms: vec![(attended, identity), (active, op)], bias: None })
            }
            Some(factors) => {
                let f = &factors[site(Kind::Down)];
                let remainder = &wd - &f.v.dot(&f.u).t();
                let site_nodes = decomposed(&mut b, &mut layout, f, &[(active, 0..config.hidden, neurons.clone())], d)?;
                let r = b.dense(&format!("{}.R", f.name), &model, &neurons, remainder)?;
                let rest = b.node(Node::Affine { terms: vec![(active, r)], bias: None });
                let kept = b.node(Node::Hadamard { left: rest, right: site_nodes.delta });
                let u = b.dense(&format!("{}.U", f.name), &model, &native(f.subcomponents())?, f.u.t().to_owned())?;
                b.node(Node::Affine { terms: vec![(attended, identity), (site_nodes.masked, u), (kept, identity)], bias: None })
            }
        };
    }
    layout.residual = x;
    layout.hidden = norm(&mut b, x, "final_norm.gain")?;
    Ok((b, layout))
}

/// `M` (`factors` none) or VPD's masked decomposition of it with the sites' subcomponents
/// `factors`, through the final normed stream, and the unembedding (`vocabulary × d`).
pub fn model(export_dir: &Path, factors: Option<&[Factors]>) -> Result<(Model, Array2<f64>), String> {
    let export = Export::open(export_dir)?;
    let config = Config::of(&export)?;
    let (b, layout) = build(&export, &config, factors)?;
    let program = b.program(config.vocab, layout.hidden)?;
    Ok((Model { program, layout }, export.tensor("wte")?))
}

/// `M` followed by VPD's causal-importance network `ci` of the sites `factors` (module note); its
/// output nodes per site, in `M`'s order of the sites. The network's weights are released as their
/// operators are made. VPD's network attends over every position (`causal` false); with `causal`
/// no position reads a later one.
fn importance_model(export_dir: &Path, factors: &[Factors], ci: CiNetwork, causal: bool) -> Result<(Model, Vec<usize>), String> {
    let export = Export::open(export_dir)?;
    let config = Config::of(&export)?;
    let (mut b, layout) = build(&export, &config, None)?;
    let CiNetwork { order, input, blocks, head: (head_w, head_b), heads, head_dim, rope_base, epsilon } = ci;
    let width = input.0.ncols();
    let stream = uniform(width)?;
    let head = native(head_dim)?;
    let concatenated =
        Interface::new(native(config.head_dim)?.groups().iter().copied().cycle().take(config.heads).collect::<Vec<Group>>()).map_err(error)?;
    let interface_of = |site: usize| -> Result<Interface, String> {
        Ok(match KINDS[site % KINDS.len()] {
            Kind::Query | Kind::Key | Kind::Value | Kind::Up => uniform(config.d)?,
            Kind::Output => concatenated.clone(),
            Kind::Down => uniform(config.hidden)?,
        })
    };
    let identity = b.identity("ci I", &stream);
    let bias = |b: &mut Builder, name: &str, rows: &Interface, row: &Array2<f64>| b.dense(name, rows, &Interface::constant(), row.t().to_owned());
    let mut terms = Vec::with_capacity(order.len());
    let mut offset = 0;
    for &site in &order {
        let cols = interface_of(site)?;
        let w = cols.width();
        let normed = b.node(Node::RmsNorm { input: layout.inputs[site], epsilon });
        let op = b.dense(&format!("ci.input.{site}"), &stream, &cols, input.0.slice(s![offset..offset + w, ..]).t().to_owned())?;
        terms.push((normed, op));
        offset += w;
    }
    if offset != input.0.nrows() {
        return Err(error("the network's input projection does not match the sites' widths"));
    }
    let input_bias = bias(&mut b, "ci.input.b", &stream, &input.1)?;
    drop(input);
    let mut x = b.node(Node::Affine { terms, bias: Some(input_bias) });
    let rotary = Rotary { base: rope_base, dims: head_dim as u32, half_split: true };
    for (i, block) in blocks.into_iter().enumerate() {
        let h = b.node(Node::RmsNorm { input: x, epsilon });
        let mut terms = vec![(x, identity)];
        for j in 0..heads {
            let rows = j * head_dim..(j + 1) * head_dim;
            let mut map = |w: &Array2<f64>, part: &str| -> Result<usize, String> {
                let op = b.dense(&format!("ci.{i}.{part}{j}"), &head, &stream, w.slice(s![rows.clone(), ..]).to_owned())?;
                Ok(b.node(Node::Affine { terms: vec![(h, op)], bias: None }))
            };
            let (q, k, v) = (map(&block.q, "q")?, map(&block.k, "k")?, map(&block.v, "v")?);
            let read = b.node(Node::Attend { query: q, key: k, value: v, scale: Scale::InverseSqrt(head_dim as u32), rotary: Some(rotary), causal });
            let o = b.dense(&format!("ci.{i}.o{j}"), &stream, &head, block.o.slice(s![.., rows]).to_owned())?;
            terms.push((read, o));
        }
        let attended = b.node(Node::Affine { terms, bias: None });
        let h = b.node(Node::RmsNorm { input: attended, epsilon });
        let hidden = uniform(block.fc1.0.ncols())?;
        let fc1 = b.dense(&format!("ci.{i}.fc1"), &hidden, &stream, block.fc1.0.t().to_owned())?;
        let fc1_bias = bias(&mut b, &format!("ci.{i}.fc1.b"), &hidden, &block.fc1.1)?;
        let pre = b.node(Node::Affine { terms: vec![(h, fc1)], bias: Some(fc1_bias) });
        let active = b.node(Node::Pointwise { input: pre, laws: vec![Law::Gelu; hidden.groups().len()] });
        let fc2 = b.dense(&format!("ci.{i}.fc2"), &stream, &hidden, block.fc2.0.t().to_owned())?;
        let fc2_bias = bias(&mut b, &format!("ci.{i}.fc2.b"), &stream, &block.fc2.1)?;
        x = b.node(Node::Affine { terms: vec![(attended, identity), (active, fc2)], bias: Some(fc2_bias) });
    }
    // The head, per site in the network's order, clamped to [0, 1]: relu(z) − relu(z − 1).
    let mut outputs = vec![0; factors.len()];
    let mut offset = 0;
    for &site in &order {
        let c = factors[site].subcomponents();
        let sub = native(c)?;
        let op = b.dense(&format!("ci.head.{site}"), &sub, &stream, head_w.slice(s![.., offset..offset + c]).t().to_owned())?;
        let head_bias = bias(&mut b, &format!("ci.head.b.{site}"), &sub, &head_b.slice(s![.., offset..offset + c]).to_owned())?;
        let pre = b.node(Node::Affine { terms: vec![(x, op)], bias: Some(head_bias) });
        // One law per coordinate group: the subcomponents are one native group.
        let lower = b.node(Node::Pointwise { input: pre, laws: vec![Law::Relu; sub.groups().len()] });
        let one = b.dense(&format!("ci.one.{site}"), &sub, &Interface::constant(), Array2::from_elem((c, 1), -1.0))?;
        let shifted_identity = b.identity(&format!("ci.I.{site}"), &sub);
        let shifted = b.node(Node::Affine { terms: vec![(pre, shifted_identity)], bias: Some(one) });
        let upper = b.node(Node::Pointwise { input: shifted, laws: vec![Law::Relu; sub.groups().len()] });
        let negative = b.diag(&format!("ci.minus.{site}"), &sub, Array1::from_elem(c, -1.0))?;
        outputs[site] = b.node(Node::Affine { terms: vec![(lower, shifted_identity), (upper, negative)], bias: None });
        offset += c;
    }
    let output = *outputs.iter().max().ok_or_else(|| error("no sites"))?;
    let program = b.program(config.vocab, output)?;
    Ok((Model { program, layout }, outputs))
}

// ------------------------------------------------------------------------------ running

/// One model on the device, run layer by layer through its final normed stream, with the
/// unembedding for its logits.
pub struct Side {
    pub program: DeviceProgram,
    pub streams: Vec<usize>,
    pub residual: usize,
    pub hidden: usize,
    head: Tensor,
}

/// The input of the final norm before `hidden` (a gain over an RMS norm).
fn final_residual(flat: &OperatorProgram, hidden: usize) -> Result<usize, String> {
    let mut n = hidden;
    loop {
        match &flat.nodes[n] {
            Node::RmsNorm { input, .. } => return Ok(*input),
            Node::Affine { terms, bias: None } if terms.len() == 1 => n = terms[0].0,
            Node::Gain { input, .. } => n = *input,
            other => return Err(error(format!("the final normed stream is not a norm of the residual: {other:?}"))),
        }
    }
}

/// The unembedding (classes × width) read by `flat`'s final normed stream.
fn unembedding_of(flat: &OperatorProgram, hidden: usize) -> Result<Array2<f64>, String> {
    let logits = match &flat.nodes[flat.output] {
        Node::Readout { input, .. } => *input,
        _ => flat.output,
    };
    match &flat.nodes[logits] {
        Node::Transposed { input, operator } if *input == hidden => Ok(flat.operators[*operator].matrix().t().to_owned()),
        Node::Affine { terms, bias: None } if terms.len() == 1 && terms[0].0 == hidden => Ok(flat.operators[terms[0].1].matrix()),
        other => Err(error(format!("the logits are not one dense map of the final normed stream: {other:?}"))),
    }
}

impl Side {
    fn compile(device: &Device, program: &OperatorProgram, numeric_bytes: usize) -> Result<DeviceProgram, String> {
        let mut compiled = DeviceProgram::compile_values_bounded(device, program, numeric_bytes)?;
        compiled.set_arithmetic(if device.float64() { Arithmetic::F64 } else { Arithmetic::F32 });
        Ok(compiled)
    }

    /// A built model (through its final normed stream) with its unembedding.
    pub fn of_model(device: &Device, model: &Model, unembedding: &Array2<f64>, numeric_bytes: usize) -> Result<Self, String> {
        Ok(Self {
            program: Self::compile(device, &model.program, numeric_bytes)?,
            streams: model.layout.streams.clone(),
            residual: model.layout.residual,
            hidden: model.layout.hidden,
            head: device.upload(unembedding.view()).map_err(error)?,
        })
    }

    /// An artifact of the split native program with sites `layers` (`M` itself through
    /// `Artifact::native`, or a library explanation).
    pub fn of_artifact(device: &Device, artifact: &Artifact, layers: &[LayerNodes], numeric_bytes: usize) -> Result<Self, String> {
        let (flat, entries, _) = interchange::sites(artifact, layers)?;
        // The stream entering each layer: the stream entering its attention block.
        let streams = entries.into_iter().step_by(2).collect();
        let program = Self::compile(device, &interchange::prefix(&flat)?, numeric_bytes)?;
        let hidden = program.hidden();
        let head = device.upload(unembedding_of(&flat, hidden)?.view()).map_err(error)?;
        Ok(Self { residual: final_residual(&flat, hidden)?, program, streams, hidden, head })
    }

    pub fn layers(&self) -> usize {
        self.streams.len()
    }

    /// The node layer `l` leaves: the stream entering the next layer, or the final residual.
    pub fn leaving(&self, l: usize) -> usize {
        if l + 1 < self.streams.len() { self.streams[l + 1] } else { self.residual }
    }

    /// Layer `l` on `family` entering at the stream `entry` (none at layer 0), the raw slots in
    /// `given` and `edit` offered every node: its trace.
    pub fn layer_trace(
        &self,
        family: &FamilyInputs,
        l: usize,
        entry: Option<&Tensor>,
        given: BTreeMap<usize, Tensor>,
        edit: impl FnMut(usize, &DeviceTrace) -> Result<Option<Tensor>, String>,
    ) -> Result<DeviceTrace, String> {
        let d = self.program.device();
        let entry = entry.map(|x| d.copy(x).map(|x| (self.streams[l], x))).transpose().map_err(error)?;
        self.program.forward_span_given(family, given, entry, self.leaving(l), edit)
    }

    /// [`Self::layer_trace`] without edits: the stream layer `l` leaves.
    pub fn layer(&self, family: &FamilyInputs, l: usize, entry: Option<&Tensor>, given: BTreeMap<usize, Tensor>) -> Result<Tensor, String> {
        let trace = self.layer_trace(family, l, entry, given, |_, _| Ok(None))?;
        self.program.device().copy(trace.value(self.leaving(l))?).map_err(error)
    }

    /// The final normed stream on `family` from the final residual `x`.
    pub fn hidden_of(&self, family: &FamilyInputs, x: &Tensor) -> Result<Tensor, String> {
        let d = self.program.device();
        let trace = self.program.forward_span(family, Some((self.residual, d.copy(x).map_err(error)?)), self.hidden, |_, _| Ok(None))?;
        d.copy(trace.value(self.hidden)?).map_err(error)
    }

    /// Per sequence of `family` (each `length` rows), the logits of the final residual `x`.
    pub fn logits(&self, family: &FamilyInputs, x: &Tensor, length: usize) -> Result<Vec<Array2<f64>>, String> {
        let d = self.program.device();
        let hidden = self.hidden_of(family, x)?;
        (0..family.rows / length)
            .map(|s| {
                let rows = d.rows_of(&hidden, s * length, length).map_err(error)?;
                let mut logits = d.zeros(length, self.head.rows()).map_err(error)?;
                d.gemm(&mut logits, 1.0, &rows, Op::N, &self.head, Op::T, 0.0, self.program.arithmetic()).map_err(error)?;
                d.download(&logits).map_err(error)
            })
            .collect()
    }
}

// ------------------------------------------------------------------------------ scoring

/// Per-token values and their summary.
#[derive(Clone, Default)]
pub struct Tokens(pub Vec<f64>);

impl Tokens {
    pub fn summary(&self) -> Value {
        if self.0.is_empty() {
            return Value::Null;
        }
        let mut v = self.0.clone();
        v.sort_by(f64::total_cmp);
        let q = |p: f64| v[((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)];
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        json!({"tokens": v.len(), "mean": mean, "q50": q(0.5), "q90": q(0.9), "q99": q(0.99), "max": q(1.0)})
    }

    pub fn mean(&self) -> f64 {
        self.0.iter().sum::<f64>() / self.0.len() as f64
    }
}

/// `M`'s log-probabilities and most probable token per row of each sequence's logits.
pub struct Reference {
    pub log_probabilities: Vec<Array2<f64>>,
    pub top: Vec<Vec<usize>>,
}

impl Reference {
    pub fn of(logits: Vec<Array2<f64>>) -> Result<Self, String> {
        let mut top = Vec::with_capacity(logits.len());
        let mut log_probabilities = Vec::with_capacity(logits.len());
        for mut l in logits {
            top.push((0..l.nrows()).map(|t| argmax(l.row(t).iter().copied())).collect());
            l.axis_iter_mut(Axis(0)).into_par_iter().try_for_each(|mut row| -> Result<(), String> {
                let lp = log_softmax(row.as_slice().ok_or_else(|| error("noncontiguous logits"))?).map_err(error)?;
                row.iter_mut().zip(lp).for_each(|(v, p)| *v = p);
                Ok(())
            })?;
            log_probabilities.push(l);
        }
        Ok(Self { log_probabilities, top })
    }

    /// `M`'s cross-entropy on the text, per predicted token.
    pub fn cross_entropy(&self, sequences: &[&[u32]]) -> Vec<f64> {
        self.log_probabilities.iter().zip(sequences).flat_map(|(lp, tokens)| (0..tokens.len() - 1).map(move |t| -lp[[t, tokens[t + 1] as usize]])).collect()
    }
}

fn argmax(values: impl Iterator<Item = f64>) -> usize {
    values.enumerate().max_by(|a, b| a.1.total_cmp(&b.1)).map_or(0, |(i, _)| i)
}

/// One protocol's per-token KL in bits, its cross-entropy and its top-1 agreement with `M`.
#[derive(Clone, Default)]
pub struct Behaviour {
    pub kl: Tokens,
    pub ce: Vec<f64>,
    pub agree: Vec<bool>,
}

impl Behaviour {
    /// Score the explanation's logits per sequence against `M`'s.
    pub fn add(&mut self, m: &Reference, e: &[Array2<f64>], sequences: &[&[u32]]) -> Result<(), String> {
        for (s, (e, tokens)) in e.iter().zip(sequences).enumerate() {
            let (lp, top) = (&m.log_probabilities[s], &m.top[s]);
            let rows: Vec<(f64, f64, bool)> = (0..e.nrows())
                .into_par_iter()
                .map(|t| -> Result<(f64, f64, bool), String> {
                    let lq = log_softmax(e.row(t).as_slice().ok_or_else(|| error("noncontiguous logits"))?).map_err(error)?;
                    let kl: f64 = lp.row(t).iter().zip(&lq).map(|(p, q)| p.exp() * (p - q)).sum();
                    let ce = if t + 1 < tokens.len() { -lq[tokens[t + 1] as usize] } else { f64::NAN };
                    Ok((kl / LN_2, ce, argmax(lq.iter().copied()) == top[t]))
                })
                .collect::<Result<_, _>>()?;
            for (kl, ce, agree) in rows {
                self.kl.0.push(kl);
                if ce.is_finite() {
                    self.ce.push(ce);
                }
                self.agree.push(agree);
            }
        }
        Ok(())
    }

    pub fn summary(&self) -> Value {
        json!({
            "kl_bits": self.kl.summary(),
            "ce": self.ce.iter().sum::<f64>() / self.ce.len() as f64,
            "top1_agreement": self.agree.iter().filter(|a| **a).count() as f64 / self.agree.len() as f64,
        })
    }
}

/// The name of a layer set: `layers_` and its layers, or `clean_input`.
pub fn set_name(set: &[bool]) -> String {
    format!("layers_{}", set.iter().enumerate().filter(|(_, e)| **e).map(|(l, _)| l.to_string()).collect::<String>())
}

/// Every nonempty layer subset of `layers`, by size then lexicographically.
pub fn subsets(layers: usize) -> Vec<Vec<bool>> {
    let mut out: Vec<Vec<bool>> = (1..1u64 << layers).map(|bits| (0..layers).map(|l| bits >> l & 1 == 1).collect()).collect();
    out.sort_by_key(|set| (set.iter().filter(|e| **e).count(), set.iter().map(|e| !e).collect::<Vec<_>>()));
    out
}

/// The final residual of every protocol (module note) on one batch, from `layer(l, explained,
/// entry)` (layer `l` of `E` or of `M` on the stream `entry`, none at layer 0, returning the stream it
/// leaves) and `M`'s streams entering each layer and its final residual (`m_streams`, `L + 1`).
pub fn protocols(
    d: &Device,
    m_streams: &[Tensor],
    embedding: &Tensor,
    mut layer: impl FnMut(usize, bool, Option<&Tensor>) -> Result<Tensor, String>,
) -> Result<Vec<(String, Tensor)>, String> {
    let layers = m_streams.len() - 1;
    let mut out = Vec::new();
    for set in subsets(layers) {
        let mut x: Option<Tensor> = None;
        for (l, explained) in set.iter().enumerate() {
            // A layer of M after only M's layers continues M's own stream.
            if !explained && set[..l].iter().all(|e| !e) {
                x = Some(d.copy(&m_streams[l + 1]).map_err(error)?);
                continue;
            }
            x = Some(layer(l, *explained, x.as_ref())?);
        }
        out.push((set_name(&set), x.ok_or_else(|| error("no layers"))?));
    }
    let mut x = d.copy(embedding).map_err(error)?;
    for l in 0..layers {
        let entering = if l == 0 { embedding } else { &m_streams[l] };
        let leaving = layer(l, true, (l > 0).then_some(entering))?;
        d.axpy(&mut x, 1.0, &leaving).map_err(error)?;
        d.axpy(&mut x, -1.0, entering).map_err(error)?;
    }
    out.push(("clean_input".into(), x));
    Ok(out)
}

/// `M`'s embedding and streams (entering each layer, then the final residual) on one batch.
pub fn streams(m: &Side, family: &FamilyInputs, given: impl Fn(usize) -> BTreeMap<usize, Tensor>) -> Result<(Tensor, Vec<Tensor>), String> {
    let d = m.program.device();
    let mut out: Vec<Tensor> = Vec::with_capacity(m.layers() + 1);
    let mut embedding = None;
    for l in 0..m.layers() {
        let trace = m.layer_trace(family, l, out.last(), given(l), |_, _| Ok(None))?;
        if l == 0 {
            let e = d.copy(trace.value(m.streams[0])?).map_err(error)?;
            out.push(d.copy(&e).map_err(error)?);
            embedding = Some(e);
        }
        out.push(d.copy(trace.value(m.leaving(l))?).map_err(error)?);
    }
    Ok((embedding.ok_or_else(|| error("no layers"))?, out))
}

/// The protocol summaries, with the single-layer mean and the mean per subset size.
pub fn protocol_summary(protocols: &BTreeMap<String, Behaviour>, layers: usize) -> Value {
    let mut out: serde_json::Map<String, Value> = protocols.iter().map(|(k, v)| (k.clone(), v.summary())).collect();
    for k in 1..layers {
        let names: Vec<String> = subsets(layers).into_iter().filter(|s| s.iter().filter(|e| **e).count() == k).map(|s| set_name(&s)).collect();
        let present: Vec<&Behaviour> = names.iter().filter_map(|n| protocols.get(n)).collect();
        if present.is_empty() {
            continue;
        }
        let mean = |f: &dyn Fn(&Behaviour) -> f64| present.iter().map(|b| f(b)).sum::<f64>() / present.len() as f64;
        out.insert(
            format!("subsets_{k}"),
            json!({
                "kl_bits_mean": mean(&|b| b.kl.mean()),
                "ce": mean(&|b| b.ce.iter().sum::<f64>() / b.ce.len() as f64),
                "top1_agreement": mean(&|b| b.agree.iter().filter(|a| **a).count() as f64 / b.agree.len() as f64),
            }),
        );
    }
    Value::Object(out)
}

// ------------------------------------------------------------------------------ VPD

/// VPD's masks for one batch: per site the mask (`rows × C`) and the remainder's `δ` per row.
pub struct Masks {
    pub mask: Vec<Array2<f64>>,
    pub delta: Vec<Array1<f64>>,
}

/// How VPD's masks are set from the causal importances `g` (module note).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    Ci,
    Rounded,
    Stochastic,
    Unmasked,
}

pub const STRATEGIES: [Strategy; 4] = [Strategy::Ci, Strategy::Rounded, Strategy::Stochastic, Strategy::Unmasked];

impl Strategy {
    pub fn name(self) -> &'static str {
        match self {
            Self::Ci => "ci",
            Self::Rounded => "rounded",
            Self::Stochastic => "stochastic",
            Self::Unmasked => "unmasked",
        }
    }

    pub fn masks(self, g: &[Array2<f64>], rng: &mut StdRng) -> Masks {
        let rows = g.first().map_or(0, Array2::nrows);
        let mask = g
            .iter()
            .map(|g| match self {
                Self::Ci => g.clone(),
                Self::Rounded => g.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
                Self::Stochastic => g.mapv(|v| v + (1.0 - v) * rng.random::<f64>()),
                Self::Unmasked => Array2::ones(g.dim()),
            })
            .collect();
        let delta = g.iter().map(|_| if self == Self::Stochastic { Array1::from_shape_fn(rows, |_| rng.random::<f64>()) } else { Array1::zeros(rows) }).collect();
        Masks { mask, delta }
    }
}

/// VPD's decomposition run on the device: `M`, the masked decomposition `E` and the
/// causal-importance program, with the sites' slots.
pub struct Vpd {
    pub m: Side,
    pub e: Side,
    importance: DeviceProgram,
    outputs: Vec<usize>,
    pub layout: Layout,
    pub m_layout: Layout,
    /// Per site its layer, subcomponents and its remainder's `δ` width, and its subcomponents.
    pub sites: Vec<(usize, usize, usize)>,
    pub factors: Vec<Factors>,
}

impl Vpd {
    /// Each program is built, compiled and released in turn, the network's weights as they are
    /// used, so the host holds one program at a time.
    pub fn new(device: &Device, export: &Path, decomposition: Decomposition, numeric_bytes: usize) -> Result<Self, String> {
        let Decomposition { sites: factors, ci } = decomposition;
        let (importance, outputs) = {
            let (built, outputs) = importance_model(export, &factors, ci, false)?;
            (Side::compile(device, &built.program, numeric_bytes)?, outputs)
        };
        let (m, m_layout, unembedding) = {
            let (built, unembedding) = model(export, None)?;
            (Side::of_model(device, &built, &unembedding, numeric_bytes)?, built.layout, unembedding)
        };
        let (e, layout, sites) = {
            let (built, _) = model(export, Some(&factors))?;
            let sites = built
                .layout
                .deltas
                .iter()
                .zip(&factors)
                .map(|(slot, f)| match built.program.declarations.slots[*slot] {
                    Slot::Raw { width } => Ok((f.layer, f.subcomponents(), width)),
                    Slot::Token { .. } => Err(error("a remainder's slot is not raw")),
                })
                .collect::<Result<_, _>>()?;
            (Side::of_model(device, &built, &unembedding, numeric_bytes)?, built.layout, sites)
        };
        Ok(Self { m, e, importance, outputs, layout, m_layout, sites, factors })
    }

    pub fn layers(&self) -> usize {
        self.m.layers()
    }

    /// The causal importances of every site on `family` (each `rows × C`).
    pub fn importances(&self, family: &FamilyInputs) -> Result<Vec<Array2<f64>>, String> {
        let trace = self.importance.forward(family)?;
        let d = self.importance.device();
        self.outputs.iter().map(|n| d.download(trace.value(*n)?).map_err(error)).collect()
    }

    /// The raw slots of layer `l`'s sites: `masks` when `explained`, else `m = 1`, `δ = 1` (`M`'s
    /// maps).
    pub fn given(&self, d: &Device, l: usize, masks: Option<&Masks>, rows: usize) -> Result<BTreeMap<usize, Tensor>, String> {
        let mut out = BTreeMap::new();
        for (s, &(layer, c, width)) in self.sites.iter().enumerate() {
            if layer != l {
                continue;
            }
            let (mask, delta) = match masks {
                Some(masks) => (d.upload(masks.mask[s].view()).map_err(error)?, {
                    let column = masks.delta[s].view().insert_axis(Axis(1)).to_owned();
                    d.upload(column.broadcast((rows, width)).ok_or_else(|| error("a remainder's δ"))?.view()).map_err(error)?
                }),
                None => (d.upload(Array2::<f64>::ones((rows, c)).view()).map_err(error)?, d.upload(Array2::<f64>::ones((rows, width)).view()).map_err(error)?),
            };
            out.insert(self.layout.masks[s], mask);
            out.insert(self.layout.deltas[s], delta);
        }
        Ok(out)
    }
}

/// Every token's count of active subcomponents (`g > 0`), overall and per layer.
pub fn active_counts(g: &[Array2<f64>], sites: &[(usize, usize, usize)], layers: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let rows = g.first().map_or(0, Array2::nrows);
    let mut per_layer = vec![vec![0.0; rows]; layers];
    for (g, &(layer, _, _)) in g.iter().zip(sites) {
        for (t, row) in g.rows().into_iter().enumerate() {
            per_layer[layer][t] += row.iter().filter(|v| **v > 0.0).count() as f64;
        }
    }
    let total = (0..rows).map(|t| per_layer.iter().map(|l| l[t]).sum()).collect();
    (total, per_layer)
}

/// The battery's behaviour and protocols of VPD on `sequences` (module note), in batches of
/// `batch` sequences: per mask strategy the error-propagating protocol, and (`every_protocol`) for
/// the CI and rounded masks every protocol; `M`'s cross-entropy and the active subcomponents per
/// token.
pub fn vpd_protocols(vpd: &Vpd, sequences: &[Vec<u32>], batch: usize, seed: u64, every_protocol: bool) -> Result<Value, String> {
    let d = vpd.m.program.device().clone();
    let layers = vpd.layers();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rows: BTreeMap<String, Behaviour> = BTreeMap::new();
    let mut m_ce = Vec::new();
    let (mut active, mut active_layers) = (Tokens::default(), vec![Tokens::default(); layers]);
    for chunk in sequences.chunks(batch) {
        let views: Vec<&[u32]> = chunk.iter().map(Vec::as_slice).collect();
        let family = sequence_family(&views)?;
        let length = views[0].len();
        let g = vpd.importances(&family)?;
        let (total, per_layer) = active_counts(&g, &vpd.sites, layers);
        active.0.extend(total);
        for (acc, counts) in active_layers.iter_mut().zip(per_layer) {
            acc.0.extend(counts);
        }
        let (embedding, m_streams) = streams(&vpd.m, &family, |_| BTreeMap::new())?;
        let reference = Reference::of(vpd.m.logits(&family, &m_streams[layers], length)?)?;
        m_ce.extend(reference.cross_entropy(&views));
        for strategy in STRATEGIES {
            let masks = strategy.masks(&g, &mut rng);
            let mut layer = |l: usize, explained: bool, entry: Option<&Tensor>| -> Result<Tensor, String> {
                if explained {
                    vpd.e.layer(&family, l, entry, vpd.given(&d, l, Some(&masks), family.rows)?)
                } else {
                    vpd.m.layer(&family, l, entry, BTreeMap::new())
                }
            };
            let finals = if every_protocol && matches!(strategy, Strategy::Ci | Strategy::Rounded) {
                protocols(&d, &m_streams, &embedding, &mut layer)?
            } else {
                let mut x: Option<Tensor> = None;
                for l in 0..layers {
                    x = Some(layer(l, true, x.as_ref())?);
                }
                vec![(set_name(&vec![true; layers]), x.ok_or_else(|| error("no layers"))?)]
            };
            for (name, x) in finals {
                let e = vpd.m.logits(&family, &x, length)?;
                rows.entry(format!("{}/{name}", strategy.name())).or_default().add(&reference, &e, &views)?;
            }
        }
        log::info!("battery: VPD protocols on {} sequences", chunk.len());
    }
    let mut out = serde_json::Map::new();
    for strategy in STRATEGIES {
        let prefix = format!("{}/", strategy.name());
        let of: BTreeMap<String, Behaviour> = rows.iter().filter_map(|(k, v)| k.strip_prefix(&prefix).map(|k| (k.to_string(), v.clone()))).collect();
        out.insert(strategy.name().into(), protocol_summary(&of, layers));
    }
    Ok(json!({
        "ce_target": m_ce.iter().sum::<f64>() / m_ce.len() as f64,
        "active_subcomponents_per_token": active.summary(),
        "active_subcomponents_per_token_by_layer": active_layers.iter().map(Tokens::mean).collect::<Vec<_>>(),
        "masks": out,
    }))
}

/// VPD's evaluation adversary (`PGDReconLoss`, its sources shared across the batch): one source
/// `s ∈ [0, 1]^{C+1}` per site (its subcomponents' and its remainder's), drawn uniformly, the masks
/// `m = g + (1 − g) s` and `δ = s_δ` at every token, `steps` sign-gradient ascent steps of
/// `step_size` on the mean over every token of `sequences` of `KL(M ‖ VPD)`, each followed by a
/// clamp to `[0, 1]`. The gradient is the program's reverse pass to the masks' raw slots from the
/// logits' `p_E − p_M` through the unembedding. Returns the mean KL in bits after the last step.
pub fn vpd_pgd(vpd: &Vpd, sequences: &[Vec<u32>], batch: usize, steps: usize, step_size: f64, seed: u64) -> Result<Value, String> {
    let d = vpd.e.program.device().clone();
    let layers = vpd.layers();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut source: Vec<Array1<f64>> = vpd.sites.iter().map(|(_, c, _)| Array1::from_shape_fn(c + 1, |_| rng.random::<f64>())).collect();
    // The importances, kept sparse per batch and site: (row, column, value) where g > 0.
    let mut sparse: Vec<Vec<Vec<(u32, u32, f64)>>> = Vec::new();
    let chunks: Vec<&[Vec<u32>]> = sequences.chunks(batch).collect();
    for chunk in &chunks {
        let views: Vec<&[u32]> = chunk.iter().map(Vec::as_slice).collect();
        let g = vpd.importances(&sequence_family(&views)?)?;
        sparse.push(g.iter().map(|g| g.indexed_iter().filter(|(_, v)| **v > 0.0).map(|((r, c), v)| (r as u32, c as u32, *v)).collect()).collect());
    }
    let tokens: usize = sequences.iter().map(Vec::len).sum();
    let head = d.copy(&vpd.e.head).map_err(error)?;
    let sweep = |source: &[Array1<f64>], gradient: bool| -> Result<(f64, Vec<Array1<f64>>), String> {
        let mut total = 0.0;
        let mut grads: Vec<Array1<f64>> = source.iter().map(|s| Array1::zeros(s.len())).collect();
        for (chunk, g) in chunks.iter().zip(&sparse) {
            let views: Vec<&[u32]> = chunk.iter().map(Vec::as_slice).collect();
            let family = sequence_family(&views)?;
            let (rows, length) = (family.rows, views[0].len());
            // The masks: s broadcast over rows, g + (1 − g) s where g > 0.
            let mut given = BTreeMap::new();
            let mut importance: Vec<Array2<f64>> = Vec::with_capacity(g.len());
            for (site, ((_, c, width), entries)) in vpd.sites.iter().zip(g).enumerate() {
                let s = &source[site];
                let mut dense = Array2::<f64>::zeros((rows, *c));
                let mut mask = Array2::from_shape_fn((rows, *c), |(_, j)| s[j]);
                for &(r, j, v) in entries {
                    let (r, j) = (r as usize, j as usize);
                    dense[[r, j]] = v;
                    mask[[r, j]] = v + (1.0 - v) * s[j];
                }
                given.insert(vpd.layout.masks[site], d.upload(mask.view()).map_err(error)?);
                given.insert(vpd.layout.deltas[site], d.upload(Array2::from_elem((rows, *width), s[*c]).view()).map_err(error)?);
                importance.push(dense);
            }
            let trace = vpd.e.program.forward_given(&family, given)?;
            let hidden = trace.value(vpd.e.hidden)?;
            let (_, m_streams) = streams(&vpd.m, &family, |_| BTreeMap::new())?;
            let reference = Reference::of(vpd.m.logits(&family, &m_streams[layers], length)?)?;
            let mut seed_rows = d.zeros(rows, hidden.cols()).map_err(error)?;
            for s in 0..views.len() {
                let h = d.rows_of(hidden, s * length, length).map_err(error)?;
                let mut logits = d.zeros(length, head.rows()).map_err(error)?;
                d.gemm(&mut logits, 1.0, &h, Op::N, &head, Op::T, 0.0, vpd.e.program.arithmetic()).map_err(error)?;
                let logits = d.download(&logits).map_err(error)?;
                let lp = &reference.log_probabilities[s];
                let mut cotangent = Array2::<f64>::zeros(logits.dim());
                let kl: Vec<f64> = cotangent
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .map(|(t, mut row)| -> Result<f64, String> {
                        let lq = log_softmax(logits.row(t).as_slice().ok_or_else(|| error("noncontiguous logits"))?).map_err(error)?;
                        let mut kl = 0.0;
                        for ((out, p), q) in row.iter_mut().zip(lp.row(t)).zip(&lq) {
                            kl += p.exp() * (p - q);
                            *out = (q.exp() - p.exp()) / tokens as f64;
                        }
                        Ok(kl)
                    })
                    .collect::<Result<_, _>>()?;
                total += kl.iter().sum::<f64>() / tokens as f64;
                if gradient {
                    let upstream = d.upload(cotangent.view()).map_err(error)?;
                    let mut part = d.zeros(length, hidden.cols()).map_err(error)?;
                    d.gemm(&mut part, 1.0, &upstream, Op::N, &head, Op::N, 0.0, vpd.e.program.arithmetic()).map_err(error)?;
                    d.set_rows(&mut seed_rows, s * length, &part).map_err(error)?;
                }
            }
            if gradient {
                let keep: Vec<usize> = vpd.layout.mask_nodes.iter().chain(&vpd.layout.delta_nodes).copied().collect();
                let cotangents = vpd.e.program.vjp_values_seeded(&trace, BTreeMap::from([(vpd.e.hidden, seed_rows)]), &keep, vpd.e.program.arithmetic())?;
                for (site, grad) in grads.iter_mut().enumerate() {
                    let c = vpd.sites[site].1;
                    if let Some(t) = cotangents.get(&vpd.layout.mask_nodes[site]) {
                        let cot = d.download(t).map_err(error)?;
                        let weighted = &cot * &importance[site].mapv(|v| 1.0 - v);
                        grad.slice_mut(s![..c]).scaled_add(1.0, &weighted.sum_axis(Axis(0)));
                    }
                    if let Some(t) = cotangents.get(&vpd.layout.delta_nodes[site]) {
                        grad[c] += d.download(t).map_err(error)?.sum();
                    }
                }
            }
        }
        Ok((total, grads))
    };
    let start = sweep(&source, false)?.0;
    for step in 0..steps {
        let (_, grads) = sweep(&source, true)?;
        for (s, g) in source.iter_mut().zip(&grads) {
            s.zip_mut_with(g, |v, g| *v = (*v + step_size * g.signum() * f64::from(u8::from(*g != 0.0))).clamp(0.0, 1.0));
        }
        log::info!("battery: VPD adversary step {}/{steps}", step + 1);
    }
    let (end, _) = sweep(&source, false)?;
    Ok(json!({"steps": steps, "step_size": step_size, "sequences": sequences.len(), "kl_bits_start": start / LN_2, "kl_bits": end / LN_2}))
}

// ------------------------------------------------------------------------------ VPD interchange

/// An orthonormal basis (`d × r`) of the span of the columns `cols` of `v` (`d × C`): the right
/// singular vectors of their rows whose singular values exceed the decomposition's rounding band;
/// none when no direction is resolved from zero.
fn column_span(v: &Array2<f64>, cols: &[usize]) -> Result<Option<Array2<f32>>, String> {
    if cols.is_empty() {
        return Ok(None);
    }
    let rows = v.select(Axis(1), cols).t().to_owned();
    let decomposition = gam_linalg::decompose::svd(rows.view(), false).map_err(error)?;
    let rank = decomposition.singular_values.iter().filter(|s| **s > decomposition.band).count();
    Ok((rank > 0).then(|| decomposition.vt.slice(s![..rank, ..]).t().mapv(|x| x as f32)))
}

/// The sites whose maps read block `block`'s normed stream: a layer's query, key and value (its
/// attention), or its up map (its MLP).
pub fn reading_sites(block: usize) -> Vec<usize> {
    let l = block / 2;
    if block % 2 == 0 { vec![KINDS.len() * l, KINDS.len() * l + 1, KINDS.len() * l + 2] } else { vec![KINDS.len() * l + 4] }
}

/// Per token of a batch, the span of the reads of VPD's subcomponents active there (`g > 0`) at
/// block `block`.
fn active_spans(factors: &[Factors], g: &[Array2<f64>], block: usize) -> Result<Vec<Option<Array2<f32>>>, String> {
    let sites = reading_sites(block);
    let v = ndarray::concatenate(Axis(1), &sites.iter().map(|s| factors[*s].v.view()).collect::<Vec<_>>()).map_err(error)?;
    let rows = g[sites[0]].nrows();
    (0..rows)
        .into_par_iter()
        .map(|t| {
            let mut cols = Vec::new();
            let mut offset = 0;
            for &s in &sites {
                cols.extend(g[s].row(t).iter().enumerate().filter(|(_, x)| **x > 0.0).map(|(c, _)| offset + c));
                offset += g[s].ncols();
            }
            column_span(&v, &cols)
        })
        .collect()
}

/// `s + (h − s) Q Qᵀ` per row (the complement patch's read: the base's component inside each
/// row's span `Q`, the source's outside it; the source whole where the span is empty), the source
/// row of base row `t` being `t mod length`.
fn complement(h: &Array2<f64>, source: &Array2<f64>, spans: &[Option<Array2<f32>>]) -> Array2<f64> {
    let length = source.nrows();
    let mut out = Array2::zeros(h.dim());
    out.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(t, mut row)| {
        let s = source.row(t % length);
        row.assign(&s);
        if let Some(q) = &spans[t] {
            let difference = (&h.row(t) - &s).mapv(|x| x as f32);
            let coordinates = difference.dot(q);
            row.zip_mut_with(&q.dot(&coordinates), |r, k| *r += f64::from(*k));
        }
    });
    out
}

/// One read patch: per base row `rows`, the read direction `q` (unit) at block `block`.
#[derive(Clone)]
struct ReadPatch {
    block: usize,
    row: usize,
    q: Array1<f64>,
}

/// `h + ((s − h)·q) q` on each patched row's tokens.
fn read_patched(h: &Array2<f64>, source: &Array2<f64>, patches: &[&ReadPatch], length: usize) -> Array2<f64> {
    let mut out = h.clone();
    for p in patches {
        for t in 0..length {
            let r = p.row * length + t;
            let along = (&source.row(t) - &h.row(r)).dot(&p.q);
            out.row_mut(r).scaled_add(along, &p.q);
        }
    }
    out
}

/// Per source and family (the read patch, each block's complement, every block's complement
/// at once): `KL(M_e ‖ E_e)` per token in bits.
struct Families {
    names: Vec<String>,
    per_source: Vec<Vec<(f64, usize)>>,
    all: Vec<Tokens>,
}

impl Families {
    fn add(&mut self, family: usize, source: usize, bits: &[f64]) {
        self.per_source[family][source].0 += bits.iter().sum::<f64>();
        self.per_source[family][source].1 += bits.len();
        self.all[family].0.extend_from_slice(bits);
    }

    fn summary(&self, worst_of: &[usize]) -> Value {
        let mut out = serde_json::Map::new();
        for (f, name) in self.names.iter().enumerate() {
            let means: Vec<f64> = self.per_source[f].iter().map(|(bits, n)| bits / *n as f64).collect();
            let worst: serde_json::Map<String, Value> =
                worst_of.iter().filter(|k| **k <= means.len()).map(|&k| (format!("worst_of_{k}"), json!(means[..k].iter().copied().fold(f64::NEG_INFINITY, f64::max)))).collect();
            out.insert(name.clone(), json!({"all_sources": self.all[f].summary(), "per_source_mean_bits": means, "shared_source": worst}));
        }
        Value::Object(out)
    }
}

/// VPD's interchange experiments (VPD alone, the base's CI masks, the remainder dropped), every
/// base patched at every position with one source shared across all bases; each model takes the
/// source's read from its own run on the source (VPD with the source's own CI masks). Families:
/// per base one read patch of one subcomponent's read direction (uniform over the subcomponents of
/// the sites reading a block), the complement patch of each block (at each token the component
/// outside the span of the active subcomponents' reads), and the complement patch at every block
/// at once (each block's read taken under the earlier patches). VPD reads only its active
/// coordinates, which a complement patch keeps, so its prediction there is its unpatched output.
pub fn vpd_interchange(vpd: &Vpd, bases: &[Vec<u32>], sources: &[Vec<u32>], batch: usize, seed: u64, worst_of: &[usize]) -> Result<Value, String> {
    let d = vpd.m.program.device().clone();
    let (layers, blocks) = (vpd.layers(), 2 * vpd.layers());
    let factors = &vpd.factors;
    let mut rng = StdRng::seed_from_u64(seed);
    // The read patches: per base a subcomponent uniform over the reading sites' subcomponents.
    let reading: Vec<(usize, usize)> = (0..blocks).flat_map(|b| reading_sites(b).into_iter().map(move |s| (b, s))).collect();
    let total: usize = reading.iter().map(|(_, s)| factors[*s].subcomponents()).sum();
    let read_of: Vec<(usize, usize, usize)> = (0..bases.len())
        .map(|_| {
            let mut k = rng.random_range(0..total);
            for &(b, s) in &reading {
                if k < factors[s].subcomponents() {
                    return (b, s, k);
                }
                k -= factors[s].subcomponents();
            }
            (0, 0, 0)
        })
        .collect();
    // Each source's reads at every block: M's, and VPD's under the source's own CI masks.
    let mut source_reads: Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)> = Vec::with_capacity(sources.len());
    for source in sources {
        let family = sequence_family(&[source.as_slice()])?;
        let g = vpd.importances(&family)?;
        let masks = Strategy::Ci.masks(&g, &mut rng);
        let capture = |side: &Side, reads: &[usize], masks: Option<&Masks>| -> Result<Vec<Array2<f64>>, String> {
            let mut out = vec![Array2::zeros((0, 0)); blocks];
            let mut x: Option<Tensor> = None;
            for l in 0..layers {
                let given = match masks {
                    Some(m) => vpd.given(&d, l, Some(m), family.rows)?,
                    None => BTreeMap::new(),
                };
                let trace = side.layer_trace(&family, l, x.as_ref(), given, |_, _| Ok(None))?;
                for b in [2 * l, 2 * l + 1] {
                    out[b] = d.download(trace.value(reads[b])?).map_err(error)?;
                }
                x = Some(d.copy(trace.value(side.leaving(l))?).map_err(error)?);
            }
            Ok(out)
        };
        source_reads.push((capture(&vpd.m, &vpd.m_layout.reads, None)?, capture(&vpd.e, &vpd.layout.reads, Some(&masks))?));
    }
    let mut names = vec!["read".to_string()];
    names.extend((0..blocks).map(|b| format!("complement_block_{b}")));
    names.push("complement_every_block".into());
    let mut families = Families { per_source: vec![vec![(0.0, 0); sources.len()]; names.len()], all: vec![Tokens::default(); names.len()], names };
    for (c, chunk) in bases.chunks(batch).enumerate() {
        let views: Vec<&[u32]> = chunk.iter().map(Vec::as_slice).collect();
        let family = sequence_family(&views)?;
        let length = views[0].len();
        let g = vpd.importances(&family)?;
        let masks = Strategy::Ci.masks(&g, &mut rng);
        let (_, m_streams) = streams(&vpd.m, &family, |_| BTreeMap::new())?;
        // VPD's unpatched output, and M's reads at every block.
        let mut x: Option<Tensor> = None;
        for l in 0..layers {
            x = Some(vpd.e.layer(&family, l, x.as_ref(), vpd.given(&d, l, Some(&masks), family.rows)?)?);
        }
        let e_logits = vpd.m.logits(&family, x.as_ref().ok_or_else(|| error("no layers"))?, length)?;
        let mut m_reads = vec![Array2::zeros((0, 0)); blocks];
        {
            let mut x: Option<Tensor> = None;
            for l in 0..layers {
                let trace = vpd.m.layer_trace(&family, l, x.as_ref(), BTreeMap::new(), |_, _| Ok(None))?;
                for b in [2 * l, 2 * l + 1] {
                    m_reads[b] = d.download(trace.value(vpd.m_layout.reads[b])?).map_err(error)?;
                }
                x = Some(d.copy(trace.value(vpd.m.leaving(l))?).map_err(error)?);
            }
        }
        let spans: Vec<Vec<Option<Array2<f32>>>> = (0..blocks).map(|b| active_spans(factors, &g, b)).collect::<Result<_, _>>()?;
        let patches: Vec<ReadPatch> = (0..chunk.len())
            .map(|r| {
                let (block, site, k) = read_of[c * batch + r];
                let v = factors[site].v.column(k).to_owned();
                let norm = v.dot(&v).sqrt();
                ReadPatch { block, row: r, q: v / norm }
            })
            .collect();
        for (s, (m_source, e_source)) in source_reads.iter().enumerate() {
            let score = |families: &mut Families, f: usize, m_final: &Tensor, e_logits: &[Array2<f64>]| -> Result<(), String> {
                let reference = Reference::of(vpd.m.logits(&family, m_final, length)?)?;
                let mut behaviour = Behaviour::default();
                behaviour.add(&reference, e_logits, &views)?;
                families.add(f, s, &behaviour.kl.0);
                Ok(())
            };
            // A forward through `side` from layer `from` whose reads are replaced by `patch(block, h)`.
            let patched_run = |side: &Side, reads: &[usize], from: usize, masks: Option<&Masks>, patch: &dyn Fn(usize, &Array2<f64>) -> Option<Array2<f64>>| -> Result<Tensor, String> {
                let mut x: Option<Tensor> = (from > 0).then(|| d.copy(&m_streams[from])).transpose().map_err(error)?;
                for l in from..layers {
                    let given = match masks {
                        Some(m) => vpd.given(&d, l, Some(m), family.rows)?,
                        None => BTreeMap::new(),
                    };
                    let trace = side.layer_trace(&family, l, x.as_ref(), given, |node, trace| {
                        let Some(b) = [2 * l, 2 * l + 1].into_iter().find(|b| reads[*b] == node) else { return Ok(None) };
                        let h = d.download(trace.value(node)?).map_err(error)?;
                        patch(b, &h).map(|value| d.upload(value.view()).map_err(error)).transpose()
                    })?;
                    x = Some(d.copy(trace.value(side.leaving(l))?).map_err(error)?);
                }
                x.ok_or_else(|| error("no layers"))
            };
            // The read patch, in M and in VPD.
            let reads_at = |b: usize| -> Vec<&ReadPatch> { patches.iter().filter(|p| p.block == b).collect() };
            let m_final = patched_run(&vpd.m, &vpd.m_layout.reads, 0, None, &|b, h| {
                let at = reads_at(b);
                (!at.is_empty()).then(|| read_patched(h, &m_source[b], &at, length))
            })?;
            let e_final = patched_run(&vpd.e, &vpd.layout.reads, 0, Some(&masks), &|b, h| {
                let at = reads_at(b);
                (!at.is_empty()).then(|| read_patched(h, &e_source[b], &at, length))
            })?;
            let e_read_logits = vpd.m.logits(&family, &e_final, length)?;
            score(&mut families, 0, &m_final, &e_read_logits)?;
            // The complement patch of each block, from M's stream entering its layer.
            for b in 0..blocks {
                let value = complement(&m_reads[b], &m_source[b], &spans[b]);
                let m_final = patched_run(&vpd.m, &vpd.m_layout.reads, b / 2, None, &|at, _| (at == b).then(|| value.clone()))?;
                score(&mut families, 1 + b, &m_final, &e_logits)?;
            }
            // The complement patch at every block at once.
            let m_final = patched_run(&vpd.m, &vpd.m_layout.reads, 0, None, &|b, h| Some(complement(h, &m_source[b], &spans[b])))?;
            score(&mut families, 1 + blocks, &m_final, &e_logits)?;
        }
        log::info!("battery: VPD interchange on {} bases", chunk.len());
    }
    let mut complement_blocks = Tokens::default();
    (1..=blocks).for_each(|f| complement_blocks.0.extend_from_slice(&families.all[f].0));
    Ok(json!({"patches": families.summary(worst_of), "complement_all_blocks": complement_blocks.summary()}))
}

// ------------------------------------------------------------------------------ cancellation

/// Per token, `a·b / (|a| |b|)` (zero where either vanishes).
fn cosines(a: &Array2<f64>, b: &Array2<f64>) -> Vec<f64> {
    a.outer_iter()
        .zip(b.outer_iter())
        .map(|(x, y)| {
            let n = (x.dot(&x) * y.dot(&y)).sqrt();
            if n > 0.0 { x.dot(&y) / n } else { 0.0 }
        })
        .collect()
}

/// Why removing VPD's inactive subcomponents from the last layer as well shrinks the error of
/// removing them from an earlier layer `l`, with the CI and the rounded masks (remainder dropped).
/// Per token, at the final residual: `D_l` = the final residual with VPD's layer `l` alone less
/// `M`'s, `D_3` likewise for the last layer; `c_3(x)` = `M`'s last layer on the stream `x` less
/// VPD's (what masking removes); `I_l = c_3(x_l) − c_3(x)`, `x_l` the stream entering the last
/// layer with layer `l` masked and `x` `M`'s. Exactly, masking both gives `D_l + D_3 − I_l`. The
/// intervention "no interaction" holds the last layer's removed part at its value on `M`'s stream,
/// `M + D_l + D_3`; "`l` without `I_l`" is `M + D_l − I_l`. KL in bits per token.
pub fn vpd_cancellation(vpd: &Vpd, bases: &[Vec<u32>], batch: usize, seed: u64) -> Result<Value, String> {
    let d = vpd.m.program.device().clone();
    let layers = vpd.layers();
    let last = layers - 1;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out: BTreeMap<&str, BTreeMap<String, Tokens>> = BTreeMap::new();
    for chunk in bases.chunks(batch) {
        let views: Vec<&[u32]> = chunk.iter().map(Vec::as_slice).collect();
        let family = sequence_family(&views)?;
        let length = views[0].len();
        let g = vpd.importances(&family)?;
        let (_, m_streams) = streams(&vpd.m, &family, |_| BTreeMap::new())?;
        let reference = Reference::of(vpd.m.logits(&family, &m_streams[layers], length)?)?;
        let m_final = d.download(&m_streams[layers]).map_err(error)?;
        for strategy in [Strategy::Ci, Strategy::Rounded] {
            let masks = strategy.masks(&g, &mut rng);
            let stats = out.entry(strategy.name()).or_default();
            let mut add = |key: String, values: Vec<f64>| stats.entry(key).or_default().0.extend(values);
            let kl = |x: &Array2<f64>| -> Result<Vec<f64>, String> {
                let tensor = d.upload(x.view()).map_err(error)?;
                let mut behaviour = Behaviour::default();
                behaviour.add(&reference, &vpd.m.logits(&family, &tensor, length)?, &views)?;
                Ok(behaviour.kl.0)
            };
            let norms = |x: &Array2<f64>| -> Vec<f64> { x.outer_iter().map(|r| r.dot(&r).sqrt()).collect() };
            // The run with VPD at the layers `masked`: its final residual and its stream entering
            // the last layer.
            let run = |masked: &[usize]| -> Result<(Array2<f64>, Tensor), String> {
                let mut x: Option<Tensor> = None;
                let mut entering = None;
                for l in 0..layers {
                    if l == last {
                        entering = Some(match &x {
                            Some(t) => d.copy(t).map_err(error)?,
                            None => d.copy(&m_streams[last]).map_err(error)?,
                        });
                    }
                    x = Some(if masked.contains(&l) {
                        vpd.e.layer(&family, l, x.as_ref(), vpd.given(&d, l, Some(&masks), family.rows)?)?
                    } else {
                        vpd.m.layer(&family, l, x.as_ref(), BTreeMap::new())?
                    });
                }
                let final_residual = d.download(&x.ok_or_else(|| error("no layers"))?).map_err(error)?;
                Ok((final_residual, entering.ok_or_else(|| error("no last layer"))?))
            };
            let removed = |x: &Tensor| -> Result<Array2<f64>, String> {
                let m = d.download(&vpd.m.layer(&family, last, Some(x), BTreeMap::new())?).map_err(error)?;
                let e = d.download(&vpd.e.layer(&family, last, Some(x), vpd.given(&d, last, Some(&masks), family.rows)?)?).map_err(error)?;
                Ok(m - e)
            };
            let (f3, _) = run(&[last])?;
            let d3 = &f3 - &m_final;
            let c3 = removed(&m_streams[last])?;
            add(format!("kl_{last}"), kl(&f3)?);
            add(format!("norm_D{last}"), norms(&d3));
            for l in 0..last {
                let (fl, entering_l) = run(&[l])?;
                let dl = &fl - &m_final;
                let interaction = removed(&entering_l)? - &c3;
                add(format!("kl_{l}"), kl(&fl)?);
                add(format!("kl_{l}{last}"), kl(&run(&[l, last])?.0)?);
                add(format!("kl_{l}{last}_no_interaction"), kl(&(&m_final + &dl + &d3))?);
                add(format!("kl_{l}_without_I"), kl(&(&m_final + &dl - &interaction))?);
                add(format!("norm_D{l}"), norms(&dl));
                add(format!("norm_I{l}"), norms(&interaction));
                add(format!("cos_I{l}_D{l}"), cosines(&interaction, &dl));
                add(format!("cos_D{l}_D{last}"), cosines(&dl, &d3));
            }
        }
        log::info!("battery: VPD cancellation on {} bases", chunk.len());
    }
    Ok(Value::Object(
        out.iter().map(|(k, v)| (k.to_string(), Value::Object(v.iter().map(|(n, t)| (n.clone(), t.summary())).collect()))).collect(),
    ))
}

// ------------------------------------------------------------------------------ circuits

/// A pair of a prompt and its counterfactual of equal token count, with the prompt's next token
/// (`target`) and the counterfactual's (`foil`): the subject-verb agreement pairs of Marks et al.
/// (2025), tokenized once (`bench/vpd_2951/sva_export.py`).
#[derive(Clone, Debug, serde::Deserialize)]
pub struct Pair {
    pub clean: Vec<u32>,
    pub counterfactual: Vec<u32>,
    pub target: u32,
    pub foil: u32,
}

/// A node basis of a model for circuits (Arora, Wu et al. 2026; Marks et al. 2025): node groups,
/// each the columns of one node of a model program on the device, and the raw slots that program
/// takes.
pub struct NodeBasis<'a> {
    pub side: &'a Side,
    pub groups: Vec<usize>,
    pub given: &'a dyn Fn(usize) -> Result<BTreeMap<usize, Tensor>, String>,
    /// The unembedding (`vocabulary × d`) the final normed stream is read by.
    pub unembedding: &'a Array2<f64>,
}

/// Pairs grouped by token count (a family runs sequences of one length).
fn by_length(pairs: &[Pair]) -> BTreeMap<usize, Vec<&Pair>> {
    let mut out: BTreeMap<usize, Vec<&Pair>> = BTreeMap::new();
    for p in pairs {
        out.entry(p.clean.len()).or_default().push(p);
    }
    out
}

impl NodeBasis<'_> {
    /// One forward pass on `sequences` with every group's node replaced by `replace(group, value)`
    /// where it returns one: the trace.
    pub fn run(&self, sequences: &[&[u32]], replace: &dyn Fn(usize, &Array2<f64>) -> Option<Array2<f64>>) -> Result<(FamilyInputs, DeviceTrace), String> {
        let family = sequence_family(sequences)?;
        let d = self.side.program.device();
        let trace = self.side.program.forward_span_given(&family, (self.given)(family.rows)?, None, self.side.hidden, |node, trace| {
            let Some(g) = self.groups.iter().position(|n| *n == node) else { return Ok(None) };
            let value = d.download(trace.value(node)?).map_err(error)?;
            replace(g, &value).map(|v| d.upload(v.view()).map_err(error)).transpose()
        })?;
        Ok((family, trace))
    }

    /// Per group, every node's mean at each position over the prompts and counterfactuals of
    /// `pairs` (positions × width; a position no prompt reaches is zero).
    pub fn means(&self, pairs: &[Pair]) -> Result<Vec<Array2<f64>>, String> {
        let d = self.side.program.device();
        let longest = pairs.iter().map(|p| p.clean.len()).max().unwrap_or(0);
        let mut sums: Vec<Array2<f64>> = Vec::new();
        let mut counts = vec![0.0; longest];
        for (length, group) in by_length(pairs) {
            let sequences: Vec<&[u32]> = group.iter().flat_map(|p| [p.clean.as_slice(), p.counterfactual.as_slice()]).collect();
            let (_, trace) = self.run(&sequences, &|_, _| None)?;
            for (g, node) in self.groups.iter().enumerate() {
                let value = d.download(trace.value(*node)?).map_err(error)?;
                if sums.len() <= g {
                    sums.push(Array2::zeros((longest, value.ncols())));
                }
                for (r, row) in value.outer_iter().enumerate() {
                    let mut at = sums[g].row_mut(r % length);
                    at += &row;
                }
            }
            counts.iter_mut().take(length).for_each(|c| *c += sequences.len() as f64);
        }
        for s in &mut sums {
            for (mut row, c) in s.outer_iter_mut().zip(&counts) {
                if *c > 0.0 {
                    row /= *c;
                }
            }
        }
        Ok(sums)
    }

    /// The mean over `pairs` of the logit difference `target − foil` at the last position, every
    /// node outside `keep` (per group, per column) set to its mean at its position.
    pub fn metric(&self, pairs: &[Pair], keep: Option<&[Vec<bool>]>, means: &[Array2<f64>]) -> Result<f64, String> {
        let mut total = 0.0;
        for (length, group) in by_length(pairs) {
            let sequences: Vec<&[u32]> = group.iter().map(|p| p.clean.as_slice()).collect();
            let replace = |g: usize, value: &Array2<f64>| -> Option<Array2<f64>> {
                let keep = &keep?[g];
                let mut out = value.clone();
                for (r, mut row) in out.outer_iter_mut().enumerate() {
                    for (c, v) in row.iter_mut().enumerate() {
                        if !keep[c] {
                            *v = means[g][[r % length, c]];
                        }
                    }
                }
                Some(out)
            };
            let (_, trace) = self.run(&sequences, &replace)?;
            let hidden = self.side.program.device().download(trace.value(self.side.hidden)?).map_err(error)?;
            for (i, p) in group.iter().enumerate() {
                let h = hidden.row((i + 1) * length - 1);
                total += h.dot(&self.unembedding.row(p.target as usize)) - h.dot(&self.unembedding.row(p.foil as usize));
            }
        }
        Ok(total / pairs.len() as f64)
    }
}

/// Faithfulness and completeness of the circuits of the `k` nodes of largest measured effect
/// `|attribution|` ([`NodeBasis::patch_effects`])
/// (per group, per column) on `test`, for `k` the powers of two up to the node count and the count
/// itself: `(m(C_k) − m(∅)) / (m(M) − m(∅))` with every node outside the circuit at its mean
/// (faithfulness) and with the circuit itself at its mean (completeness), `∅` every node at its
/// mean, means over `train`'s prompts and counterfactuals.
pub fn circuit_curve(basis: &NodeBasis, attribution: &[Array1<f64>], train: &[Pair], test: &[Pair]) -> Result<Value, String> {
    let means = basis.means(train)?;
    let widths: Vec<usize> = attribution.iter().map(Array1::len).collect();
    let mut order: Vec<(usize, usize)> = widths.iter().enumerate().flat_map(|(g, w)| (0..*w).map(move |c| (g, c))).collect();
    order.sort_by(|a, b| attribution[b.0][b.1].abs().total_cmp(&attribution[a.0][a.1].abs()));
    let total = order.len();
    let keep_of = |k: usize, inside: bool| -> Vec<Vec<bool>> {
        let mut keep: Vec<Vec<bool>> = widths.iter().map(|w| vec![!inside; *w]).collect();
        for &(g, c) in &order[..k] {
            keep[g][c] = inside;
        }
        keep
    };
    let full = basis.metric(test, None, &means)?;
    let empty = basis.metric(test, Some(&keep_of(0, true)), &means)?;
    let mut ks: Vec<usize> = (0..).map(|j| 1usize << j).take_while(|k| *k < total).collect();
    ks.push(total);
    let mut faithfulness = Vec::with_capacity(ks.len());
    let mut completeness = Vec::with_capacity(ks.len());
    for &k in &ks {
        faithfulness.push((basis.metric(test, Some(&keep_of(k, true)), &means)? - empty) / (full - empty));
        completeness.push((basis.metric(test, Some(&keep_of(k, false)), &means)? - empty) / (full - empty));
    }
    Ok(json!({"nodes": total, "m_model": full, "m_empty": empty, "k": ks, "faithfulness": faithfulness, "completeness": completeness}))
}

impl NodeBasis<'_> {
    /// Per group and column, the node's measured effect on the logit difference: the mean over
    /// `pairs` of the change of `target − foil` at the last position when that one column is
    /// replaced at every position by its value on the counterfactual, every later node rerun
    /// (activation patching). Columns are patched in copies of the prompts, one column per copy,
    /// as many copies per pass as fit in `rows` rows.
    pub fn patch_effects(&self, pairs: &[Pair], rows: usize) -> Result<Vec<Array1<f64>>, String> {
        let d = self.side.program.device();
        let mut effects: Vec<Array1<f64>> = Vec::new();
        for (length, group) in by_length(pairs) {
            let clean: Vec<&[u32]> = group.iter().map(|p| p.clean.as_slice()).collect();
            let counterfactual: Vec<&[u32]> = group.iter().map(|p| p.counterfactual.as_slice()).collect();
            let (_, trace) = self.run(&counterfactual, &|_, _| None)?;
            let source: Vec<Array2<f64>> = self.groups.iter().map(|n| d.download(trace.value(*n)?).map_err(error)).collect::<Result<_, _>>()?;
            if effects.is_empty() {
                effects = source.iter().map(|v| Array1::zeros(v.ncols())).collect();
            }
            // Per sequence of a pass (copies of `group`'s prompts), its logit difference.
            let differences = |count: usize, trace: &DeviceTrace| -> Result<Vec<f64>, String> {
                let hidden = d.download(trace.value(self.side.hidden)?).map_err(error)?;
                Ok((0..count)
                    .map(|i| {
                        let p = group[i % group.len()];
                        let h = hidden.row((i + 1) * length - 1);
                        h.dot(&self.unembedding.row(p.target as usize)) - h.dot(&self.unembedding.row(p.foil as usize))
                    })
                    .collect())
            };
            let (_, trace) = self.run(&clean, &|_, _| None)?;
            let base = differences(clean.len(), &trace)?;
            let per_copy = group.len() * length;
            let copies = (rows / per_copy).max(1);
            for (g, value) in source.iter().enumerate() {
                let columns: Vec<usize> = (0..value.ncols()).collect();
                for chunk in columns.chunks(copies) {
                    let sequences: Vec<&[u32]> = chunk.iter().flat_map(|_| clean.iter().copied()).collect();
                    let replace = |at: usize, current: &Array2<f64>| -> Option<Array2<f64>> {
                        if at != g {
                            return None;
                        }
                        let mut out = current.clone();
                        for (j, &c) in chunk.iter().enumerate() {
                            for r in 0..per_copy {
                                out[[j * per_copy + r, c]] = value[[r, c]];
                            }
                        }
                        Some(out)
                    };
                    let (_, trace) = self.run(&sequences, &replace)?;
                    let patched = differences(sequences.len(), &trace)?;
                    for (j, &c) in chunk.iter().enumerate() {
                        effects[g][c] += (0..group.len()).map(|i| patched[j * group.len() + i] - base[i]).sum::<f64>() / pairs.len() as f64;
                    }
                }
            }
        }
        Ok(effects)
    }
}

// ------------------------------------------------------------------------------ VPD's code length

/// The sum over a batch's tokens of `KL(M ‖ E)` in nats, from the final normed streams `m` of
/// `M` and `e` of `E` (the batch's sequences of `length` rows each) through the unembedding
/// `head`, on the device; with `seed` the gradient of that sum in `E`'s final normed stream,
/// `(p_E − p_M)` through `head`.
fn divergence_and_seed(d: &Device, m: &Tensor, e: &Tensor, head: &Tensor, length: usize, seed: bool, arithmetic: Arithmetic) -> Result<(f64, Option<Tensor>), String> {
    let sequences = e.rows() / length;
    let mut total = 0.0;
    let mut out = if seed { Some(d.zeros(e.rows(), e.cols()).map_err(error)?) } else { None };
    for s in 0..sequences {
        let logits_of = |x: &Tensor| -> Result<Tensor, String> {
            let rows = d.rows_of(x, s * length, length).map_err(error)?;
            let mut logits = d.zeros(length, head.rows()).map_err(error)?;
            d.gemm(&mut logits, 1.0, &rows, Op::N, head, Op::T, 0.0, arithmetic).map_err(error)?;
            Ok(logits)
        };
        let (target, mut logits) = (logits_of(m)?, logits_of(e)?);
        // The KL per row, and the cotangent `q − p` in place of `E`'s logits.
        total += d.kl_rows(&target, &mut logits, None).map_err(error)?.iter().sum::<f64>();
        if let Some(out) = out.as_mut() {
            let mut part = d.zeros(length, e.cols()).map_err(error)?;
            d.gemm(&mut part, 1.0, &logits, Op::N, head, Op::N, 0.0, arithmetic).map_err(error)?;
            d.set_rows(out, s * length, &part).map_err(error)?;
        }
    }
    Ok((total, out))
}

/// The causal-importance masks of every site on `family`, as the raw slots of the whole program
/// (the remainder dropped: VPD's intended setting), on the device: each mask the importance
/// network's output itself (`Strategy::Ci`) and every `δ` zero.
fn importance_given(vpd: &Vpd, family: &FamilyInputs) -> Result<BTreeMap<usize, Tensor>, String> {
    let d = vpd.e.program.device();
    let trace = vpd.importance.forward(family)?;
    let mut given = BTreeMap::new();
    for (s, &(_, _, width)) in vpd.sites.iter().enumerate() {
        given.insert(vpd.layout.masks[s], d.copy(trace.value(vpd.outputs[s])?).map_err(error)?);
        given.insert(vpd.layout.deltas[s], d.zeros(family.rows, width).map_err(error)?);
    }
    Ok(given)
}

/// VPD's decomposition priced in the library's code length (`library_mdl`, module note there): its
/// subcomponents' means held at VPD's values, a factorized Gaussian posterior `N(μ, σ²)` over every
/// entry of `U` and `V` with one empirical-Bayes prior group per subcomponent's `U` row and per its
/// `V` column, `σ` fitted by IVON with the mean's step zero (its curvature estimate then sets
/// `σ = 1 / √(N (h + δ))`, the width at which `N E_q[ℓ] + KL(q ‖ p)` is stationary for fixed means),
/// on the data term `Σ_t KL(M ‖ VPD_θ)` over the `train` sequences' tokens in VPD's intended setting
/// (the causal-importance masks of `M`'s activations, the remainder dropped). Epochs run until the
/// mean paired improvement of the per-batch objective is below its standard error. Reports per
/// epoch and at the end `F = KL(q ‖ p) + Σ_G ½ log2 |G| + N · data` in bits, and the held-out data
/// term at the posterior mean (VPD itself) and at one sample per batch.
///
/// The converged posterior is written to `save` ([`write_pricing_posterior`]). With `start`, the
/// fit starts from such a file instead of from each group's mean square: its log standard
/// deviations and IVON's curvature estimates, which are per-token quantities, so a posterior
/// converged at one `N` starts the fit at another (the curvature's running average otherwise
/// starts far above the data's and falls by a factor `e` per epoch: at `N = 2^20` the start took
/// 30 epochs, at any `N`).
///
/// With `charge` (VPD's decomposition directory) the causal-importance network is charged too
/// ([`Charged`]): its weights are a posterior of their own, one prior group per operator row,
/// sampled with the subcomponents', and `F` adds its description; the masks still read `M`'s
/// clean activations. Its posterior is written beside `save` (extension `network.f32`).
pub fn vpd_pricing(vpd: &Vpd, export: &Path, train: &[Vec<u32>], held_out: &[Vec<u32>], batch: usize, seed: u64, (start, charge): (Option<&Path>, Option<&Path>), save: &Path, mut report: impl FnMut(&Value) -> Result<(), String>) -> Result<Value, String> {
    let device = vpd.e.program.device().clone();
    let (built, _) = model(export, Some(&vpd.factors))?;
    // The subcomponents' operators: per site its `V` (rows the subcomponents) and `U` (columns).
    let mut trainable = Vec::new();
    let mut groups: Vec<Vec<u32>> = Vec::new();
    let mut means: Vec<Array2<f64>> = Vec::new();
    let mut sizes: Vec<f64> = Vec::new();
    let mut base = 0usize;
    for f in &vpd.factors {
        let c = f.subcomponents();
        sizes.extend(std::iter::repeat_n(f.v.nrows() as f64, c));
        sizes.extend(std::iter::repeat_n(f.u.ncols() as f64, c));
        for (op, operator) in built.program.operators.iter().enumerate() {
            let Some(part) = operator.name.strip_prefix(&format!("{}.", f.name)) else { continue };
            let values = operator.matrix();
            let (rows, cols) = values.dim();
            let ids: Vec<u32> = if part.starts_with('V') {
                (0..rows * cols).map(|e| (base + e / cols) as u32).collect()
            } else if part.starts_with('U') {
                (0..rows * cols).map(|e| (base + c + e % cols) as u32).collect()
            } else {
                continue;
            };
            trainable.push(op);
            groups.push(ids);
            means.push(values);
        }
        base += 2 * c;
    }
    let tokens = (train.iter().map(Vec::len).sum::<usize>()) as f64;
    // The start: each entry's variance its group's mean square over the training tokens.
    let mut squares = vec![(0.0, 0.0); base];
    for (m, ids) in means.iter().zip(&groups) {
        for (v, g) in m.iter().zip(ids) {
            squares[*g as usize].0 += 1.0;
            squares[*g as usize].1 += v * v;
        }
    }
    let started = start.map(|path| read_pricing_posterior(path, &means)).transpose()?;
    let log_sd: Vec<Array2<f64>> = match &started {
        Some(arrays) => arrays.iter().map(|(log_sd, _)| log_sd.clone()).collect(),
        None => means
            .iter()
            .zip(&groups)
            .map(|(m, ids)| {
                let cols = m.ncols();
                Array2::from_shape_fn(m.dim(), |(r, c)| {
                    let (n, s) = squares[ids[r * cols + c] as usize];
                    0.5 * (s / n / tokens).ln()
                })
            })
            .collect(),
    };
    let mut program = Side::compile(&device, &built.program, usize::MAX)?;
    let hidden_node = built.layout.hidden;
    drop(built);
    program.prepare_dense_parameters(&trainable)?;
    let parts = crate::device_posterior::Parts { operators: &trainable, mean: &means, log_sd: &log_sd, groups: &groups, count: base, reference: None };
    let state = started.as_ref().map(|_| crate::device_posterior::State::Zero);
    let mut posterior = crate::device_posterior::DevicePosterior::from_parts(&device, &parts, tokens, state, 0)?;
    // The pass sets the deviations and the curvature only: the means stay where they are.
    posterior.hold_means(true);
    if let Some(arrays) = started {
        for (i, (log_sd, curvature)) in arrays.iter().enumerate() {
            posterior.set_start(i, log_sd, curvature)?;
        }
        posterior.settle()?;
    }
    drop((means, log_sd));
    let variance_nats: f64 = sizes.iter().map(|n| 0.5 * n.ln()).sum();
    let batches: Vec<&[Vec<u32>]> = train.chunks(batch).collect();
    let ivon = crate::device_posterior::Ivon { beta1: 0.0, beta2: 1.0 - 1.0 / batches.len() as f64 };
    let head = d_copy(&device, &vpd.e.head)?;
    let arithmetic = program.arithmetic();
    let mut charged = match charge {
        Some(dir) => {
            posterior.mean_into(&mut program)?;
            Some(Charged::new(vpd, export, dir, (&mut program, hidden_node), &batches, &head, tokens, seed)?)
        }
        None => None,
    };
    let variance_nats = variance_nats + charged.as_ref().map_or(0.0, |c| c.variance_nats);
    // One pass over `sequences` at a sample per batch (with `step`, IVON's step after each): the
    // per-batch data term in nats per token.
    let pass = |sequences: &[&[Vec<u32>]], epoch: u64, step: bool, posterior: &mut crate::device_posterior::DevicePosterior, program: &mut DeviceProgram, charged: &mut Option<Charged>, at_mean: bool| -> Result<Vec<(f64, usize)>, String> {
        let mut out = Vec::with_capacity(sequences.len());
        for (b, chunk) in sequences.iter().enumerate() {
            let key = seed ^ (epoch << 32) ^ b as u64;
            if at_mean { posterior.mean_into(program)? } else { posterior.sample_into(program, key)? }
            if let Some(c) = charged.as_mut() {
                // The network's draw under a key apart from the subcomponents' (both count their
                // operators from zero).
                if at_mean { c.posterior.mean_into(&mut c.program)? } else { c.posterior.sample_into(&mut c.program, gam_linalg::utils::splitmix64_hash(key ^ NETWORK_KEY))? }
            }
            let views: Vec<&[u32]> = chunk.iter().map(Vec::as_slice).collect();
            let family = sequence_family(&views)?;
            let length = views[0].len();
            let network = charged.as_ref().map(|c| c.program.forward(&family)).transpose()?;
            let given = match (charged.as_ref(), network.as_ref()) {
                (Some(c), Some(ci)) => network_given(vpd, &family, ci, &c.outputs)?,
                _ => importance_given(vpd, &family)?,
            };
            let keep: &[usize] = if charged.is_some() { &vpd.layout.mask_nodes } else { &[] };
            let trace = program.forward_given(&family, given)?;
            let (_, m_streams) = streams(&vpd.m, &family, |_| BTreeMap::new())?;
            let m_hidden = vpd.m.hidden_of(&family, &m_streams[vpd.layers()])?;
            let (kl, cotangent) = divergence_and_seed(&device, &m_hidden, trace.value(hidden_node)?, &head, length, step, arithmetic)?;
            if let Some(cotangent) = cotangent {
                let (nodes, gradients) = program.vjp_values_dense(&trace, BTreeMap::from([(hidden_node, cotangent)]), keep, &trainable, arithmetic)?;
                // A draw of the Gauss–Newton factor through the same forward pass: the Fisher
                // probe at VPD's own prediction at every token (`interchange::Factor`), under a
                // key apart from the sample's (both are Philox draws).
                let probe = gam_linalg::utils::splitmix64_hash(key);
                let probed = crate::interchange::fisher_probe_seed(&device, trace.value(hidden_node)?, &head, length, probe, arithmetic)?;
                let (probe_nodes, factor) = program.vjp_values_dense(&trace, BTreeMap::from([(hidden_node, probed)]), keep, &trainable, arithmetic)?;
                let per_token = 1.0 / family.rows as f64;
                posterior.step(&gradients, per_token, (&factor, per_token), &BTreeMap::new(), &ivon)?;
                if let (Some(c), Some(ci)) = (charged.as_mut(), network.as_ref()) {
                    // The masks' cotangents carried on through the network to its weights.
                    let (_, g) = c.program.vjp_values_dense(ci, mask_seeds(vpd, nodes, &c.outputs), &[], &c.trainable, arithmetic)?;
                    let (_, u) = c.program.vjp_values_dense(ci, mask_seeds(vpd, probe_nodes, &c.outputs), &[], &c.trainable, arithmetic)?;
                    c.posterior.step(&g, per_token, (&u, per_token), &BTreeMap::new(), &ivon)?;
                }
            }
            out.push((kl, family.rows));
        }
        Ok(out)
    };
    let mut epochs: Vec<Value> = Vec::new();
    let mut previous: Option<Vec<f64>> = None;
    for epoch in 1u64.. {
        let data = pass(&batches, epoch, true, &mut posterior, &mut program, &mut charged, false)?;
        let network_divergence: f64 = charged.as_ref().map(|c| c.posterior.divergences()).transpose()?.map_or(0.0, |d| d.iter().sum());
        let divergence: f64 = posterior.divergences()?.iter().sum::<f64>() + network_divergence;
        let description = divergence + variance_nats;
        // Each batch's objective estimate `N · data per token + description`, in bits.
        let estimates: Vec<f64> = data.iter().map(|(kl, n)| (tokens * kl / *n as f64 + description) / LN_2).collect();
        let data_bits = data.iter().map(|(kl, _)| kl).sum::<f64>() / tokens / LN_2;
        let (improvement, standard_error) = match &previous {
            Some(before) => {
                let d: Vec<f64> = before.iter().zip(&estimates).map(|(a, b)| a - b).collect();
                let mean = d.iter().sum::<f64>() / d.len() as f64;
                let var = d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (d.len() as f64 - 1.0).max(1.0);
                (Some(mean), Some((var / d.len() as f64).sqrt()))
            }
            None => (None, None),
        };
        epochs.push(json!({
            "epoch": epoch,
            "data_bits_per_token": data_bits,
            "divergence_bits": divergence / LN_2,
            "network_divergence_bits": network_divergence / LN_2,
            "variance_bits": variance_nats / LN_2,
            "objective_bits": tokens * data_bits + description / LN_2,
            "improvement_bits": improvement,
            "standard_error_bits": standard_error,
        }));
        log::info!("pricing epoch {epoch}: {}", epochs[epochs.len() - 1]);
        report(&json!({"epochs": epochs}))?;
        let converged = matches!((improvement, standard_error), (Some(i), Some(s)) if i < s);
        previous = Some(estimates);
        if converged {
            break;
        }
    }
    let held: Vec<&[Vec<u32>]> = held_out.chunks(batch).collect();
    let held_tokens: usize = held_out.iter().map(Vec::len).sum();
    let at = |data: Vec<(f64, usize)>| data.iter().map(|(kl, _)| kl).sum::<f64>() / held_tokens as f64 / LN_2;
    write_pricing_posterior(save, &posterior, trainable.len())?;
    if let Some(c) = charged.as_ref() {
        write_pricing_posterior(&save.with_extension("network.f32"), &c.posterior, c.trainable.len())?;
    }
    let sampled = at(pass(&held, 0, false, &mut posterior, &mut program, &mut charged, false)?);
    let mean = at(pass(&held, 0, false, &mut posterior, &mut program, &mut charged, true)?);
    let network_divergence: f64 = charged.as_ref().map(|c| c.posterior.divergences()).transpose()?.map_or(0.0, |d| d.iter().sum());
    let divergence: f64 = posterior.divergences()?.iter().sum::<f64>() + network_divergence;
    Ok(json!({
        "network": charged.as_ref().map(|c| json!({
            "parameters": c.parameters,
            "groups": c.groups,
            "divergence_bits": network_divergence / LN_2,
            "variance_bits": c.variance_nats / LN_2,
        })),
        "start": start.map(|p| p.display().to_string()),
        "posterior": save.display().to_string(),
        "training_tokens": tokens,
        "subcomponents": base / 2,
        "parameters": groups.iter().map(Vec::len).sum::<usize>(),
        "groups": base,
        "divergence_bits": divergence / LN_2,
        "variance_bits": variance_nats / LN_2,
        "held_out_kl_bits_per_token_at_mean": mean,
        "held_out_kl_bits_per_token_sampled": sampled,
        "epochs": epochs,
    }))
}

/// The key a charged network's weight draw is made apart from the subcomponents' under.
const NETWORK_KEY: u64 = 0x6369_6e65_7477_6f72;

/// VPD's causal-importance network as trainable operators of a program of its own, charged beside
/// the subcomponents ([`vpd_pricing`]): every learned operator of the network (the input
/// projection, the blocks' maps and biases, the head), one prior group per operator row (an
/// output unit's weights; a bias column is one group; a head row is one subcomponent's gate), the
/// means held at VPD's values. The deviations start at the diagonal Gauss–Newton (Laplace)
/// optimum for fixed means: one pass over the training batches at the means sums each weight's
/// squared Fisher-probe gradient (the probe at `E`'s prediction, pulled back through `E` to its
/// masks and through the network), `h` its average per token, and each group solves
/// `σ_j² = 1 / (N h_j + 1 / v_G)`, `v_G = mean(μ² + σ²)` by fixed-point iteration (100 rounds at
/// most: a group no token's curvature reaches has none, its `v_G` growing and its divergence
/// falling toward zero).
struct Charged {
    program: DeviceProgram,
    outputs: Vec<usize>,
    trainable: Vec<usize>,
    posterior: crate::device_posterior::DevicePosterior,
    parameters: usize,
    groups: usize,
    variance_nats: f64,
}

impl Charged {
    #[allow(clippy::too_many_arguments)]
    fn new(vpd: &Vpd, export: &Path, decomposition: &Path, (program, hidden_node): (&mut DeviceProgram, usize), batches: &[&[Vec<u32>]], head: &Tensor, tokens: f64, seed: u64) -> Result<Self, String> {
        let device = program.device().clone();
        let arithmetic = program.arithmetic();
        let Decomposition { sites, ci } = Decomposition::load(decomposition)?;
        let (built, outputs) = importance_model(export, &sites, ci, false)?;
        drop(sites);
        let fixed = |name: &str| name == "ci I" || ["ci.one.", "ci.I.", "ci.minus."].iter().any(|p| name.starts_with(p));
        let (mut trainable, mut groups, mut means, mut sizes): (Vec<usize>, Vec<Vec<u32>>, Vec<Array2<f64>>, Vec<usize>) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (op, operator) in built.program.operators.iter().enumerate() {
            if !operator.name.starts_with("ci.") || fixed(&operator.name) {
                continue;
            }
            let values = operator.matrix();
            let (rows, cols) = values.dim();
            let base = u32::try_from(sizes.len()).map_err(error)?;
            if cols == 1 {
                groups.push(vec![base; rows]);
                sizes.push(rows);
            } else {
                groups.push((0..rows * cols).map(|e| base + (e / cols) as u32).collect());
                sizes.extend(std::iter::repeat_n(cols, rows));
            }
            trainable.push(op);
            means.push(values);
        }
        let mut network = Side::compile(&device, &built.program, usize::MAX)?;
        drop(built);
        network.prepare_dense_parameters(&trainable)?;
        log::info!("charged network: {} operators, {} parameters, {} groups", trainable.len(), groups.iter().map(Vec::len).sum::<usize>(), sizes.len());
        // The Laplace pass at the means: per weight the sum over batches of its squared probe gradient.
        let mut squares: BTreeMap<usize, Tensor> = BTreeMap::new();
        for (b, chunk) in batches.iter().enumerate() {
            let views: Vec<&[u32]> = chunk.iter().map(Vec::as_slice).collect();
            let family = sequence_family(&views)?;
            let length = views[0].len();
            let ci = network.forward(&family)?;
            let trace = program.forward_given(&family, network_given(vpd, &family, &ci, &outputs)?)?;
            let key = gam_linalg::utils::splitmix64_hash(seed ^ NETWORK_KEY ^ b as u64);
            let probed = crate::interchange::fisher_probe_seed(&device, trace.value(hidden_node)?, head, length, key, arithmetic)?;
            let (nodes, _) = program.vjp_values_dense(&trace, BTreeMap::from([(hidden_node, probed)]), &vpd.layout.mask_nodes, &[], arithmetic)?;
            drop(trace);
            let (_, factor) = network.vjp_values_dense(&ci, mask_seeds(vpd, nodes, &outputs), &[], &trainable, arithmetic)?;
            for (op, u) in &factor {
                match squares.get_mut(op) {
                    Some(sum) => device.hadamard(sum, u, u, true).map_err(error)?,
                    None => {
                        let mut sum = device.zeros(u.rows(), u.cols()).map_err(error)?;
                        device.hadamard(&mut sum, u, u, false).map_err(error)?;
                        squares.insert(*op, sum);
                    }
                }
            }
            if b % 256 == 0 {
                log::info!("charged network: Laplace pass, batch {b} of {}", batches.len());
            }
        }
        let mut log_sd = Vec::with_capacity(means.len());
        for (op, mean) in trainable.iter().zip(&means) {
            let h = match squares.remove(op) {
                Some(sum) => device.download(&sum).map_err(error)?.mapv(|v| v / tokens),
                None => Array2::zeros(mean.dim()),
            };
            let deviations = |mu: &[f64], h: &[f64]| -> Vec<f64> {
                let n = mu.len() as f64;
                let second: f64 = mu.iter().map(|x| x * x).sum();
                let mut v = (second / n).max(f64::MIN_POSITIVE);
                for _ in 0..100 {
                    let next = (second + h.iter().map(|hj| 1.0 / (tokens * hj + 1.0 / v)).sum::<f64>()) / n;
                    let settled = (next - v).abs() <= 1e-12 * v;
                    v = next;
                    if settled {
                        break;
                    }
                }
                h.iter().map(|hj| 0.5 * (1.0 / (tokens * hj + 1.0 / v)).ln()).collect()
            };
            let (rows, cols) = mean.dim();
            let flat = |a: &Array2<f64>| a.iter().copied().collect::<Vec<f64>>();
            let values: Vec<f64> = if cols == 1 {
                deviations(&flat(mean), &flat(&h))
            } else {
                (0..rows).flat_map(|r| deviations(&mean.row(r).to_vec(), &h.row(r).to_vec())).collect()
            };
            log_sd.push(Array2::from_shape_vec((rows, cols), values).map_err(error)?);
        }
        let parts = crate::device_posterior::Parts { operators: &trainable, mean: &means, log_sd: &log_sd, groups: &groups, count: sizes.len(), reference: None };
        let mut posterior = crate::device_posterior::DevicePosterior::from_parts(&device, &parts, tokens, None, 0)?;
        posterior.hold_means(true);
        let parameters = groups.iter().map(Vec::len).sum();
        let variance_nats = sizes.iter().map(|n| 0.5 * (*n as f64).ln()).sum();
        let divergence: f64 = posterior.divergences()?.iter().sum();
        log::info!("charged network: Laplace start, divergence {:.0} bits, variance {:.0} bits", divergence / LN_2, variance_nats / LN_2);
        Ok(Self { program: network, outputs, trainable, posterior, parameters, groups: sizes.len(), variance_nats })
    }
}

/// The raw slots of every site for `E` with the masks a network's trace `ci` computed at its
/// output nodes `outputs`, the remainder dropped.
fn network_given(vpd: &Vpd, family: &FamilyInputs, ci: &DeviceTrace, outputs: &[usize]) -> Result<BTreeMap<usize, Tensor>, String> {
    let d = vpd.e.program.device();
    let mut given = BTreeMap::new();
    for (s, &(_, _, width)) in vpd.sites.iter().enumerate() {
        given.insert(vpd.layout.masks[s], d.copy(ci.value(outputs[s])?).map_err(error)?);
        given.insert(vpd.layout.deltas[s], d.zeros(family.rows, width).map_err(error)?);
    }
    Ok(given)
}

/// The cotangents of `E`'s mask nodes (`nodes`, kept by its reverse pass) as seeds of the
/// network's outputs `outputs`.
fn mask_seeds(vpd: &Vpd, mut nodes: BTreeMap<usize, Tensor>, outputs: &[usize]) -> BTreeMap<usize, Tensor> {
    vpd.layout.mask_nodes.iter().zip(outputs).filter_map(|(m, o)| nodes.remove(m).map(|t| (*o, t))).collect()
}

/// A pricing posterior written to `path` ([`vpd_pricing`]): per trainable operator in order, its
/// log standard deviations and then IVON's curvature estimates, row-major little-endian float32
/// (the device holds them in float32 on CUDA). Written to `path` with `.partial` appended and
/// renamed when whole.
pub fn write_pricing_posterior(path: &Path, posterior: &crate::device_posterior::DevicePosterior, operators: usize) -> Result<(), String> {
    use std::io::Write;
    let partial = PathBuf::from(format!("{}.partial", path.display()));
    let mut file = std::io::BufWriter::new(std::fs::File::create(&partial).map_err(error)?);
    for i in 0..operators {
        let (_, log_sd, [_, curvature]) = posterior.operator(i)?;
        for array in [&log_sd, &curvature] {
            let bytes: Vec<u8> = array.iter().flat_map(|v| (*v as f32).to_le_bytes()).collect();
            file.write_all(&bytes).map_err(error)?;
        }
    }
    file.flush().map_err(error)?;
    drop(file);
    std::fs::rename(&partial, path).map_err(error)
}

/// The log standard deviations and curvature estimates of a pricing posterior
/// ([`write_pricing_posterior`]), one pair per array of `shapes_of`'s shapes; a file of another size
/// is refused.
fn read_pricing_posterior(path: &Path, shapes_of: &[Array2<f64>]) -> Result<Vec<(Array2<f64>, Array2<f64>)>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let entries: usize = shapes_of.iter().map(|m| 2 * m.len()).sum();
    if bytes.len() != 4 * entries {
        return Err(format!("{}: {} bytes, a posterior of these operators has {}", path.display(), bytes.len(), 4 * entries));
    }
    let mut values = bytes.chunks_exact(4).map(|c| f64::from(f32::from_le_bytes(c.try_into().expect("four bytes"))));
    let mut next = |dim: (usize, usize)| Array2::from_shape_fn(dim, |_| values.next().expect("sized above"));
    Ok(shapes_of.iter().map(|m| (next(m.dim()), next(m.dim()))).collect())
}

/// Where VPD's masks come from, measured (the fair baseline): held-out `KL(M ‖ VPD)` in bits per
/// token at VPD's own weights (the remainder dropped) and the subcomponents active (mask above
/// zero) per token per layer, with the masks of its causal-importance network computed from
/// `M`'s clean activations (`m_clean`, VPD's setting); from them with the network's attention
/// causal (`causal`), so no position's mask reads a later position; with every mask 1
/// (`all_on`); from VPD's own activations by `k` rounds of `masks ← CI(the sites' inputs of
/// VPD run with masks)` from every mask 1 (`own_k`), which reads nothing of `M`; and one such
/// round through the causal network (`own_causal_1`), autonomous and causal. The network
/// reads every site's input at once (one input projection over all the sites) and attends over
/// every position, so VPD on its own activations has no single pass: a layer's masks read later
/// layers' inputs, and position `t`'s read position `t + 1`'s, whose first sites' input is the
/// next token's embedding.
pub fn vpd_mask_sources(vpd: &Vpd, export: &Path, decomposition: &Path, held_out: &[Vec<u32>], batch: usize, rounds: usize, numeric_bytes: usize) -> Result<Value, String> {
    let d = vpd.e.program.device().clone();
    let layers = vpd.layers();
    let (causal, causal_outputs) = {
        let Decomposition { sites, ci } = Decomposition::load(decomposition)?;
        let (built, outputs) = importance_model(export, &sites, ci, true)?;
        (Side::compile(&d, &built.program, numeric_bytes)?, outputs)
    };
    let arithmetic = vpd.e.program.arithmetic();
    let names: Vec<String> = ["m_clean", "causal", "all_on", "own_causal_1"].iter().map(|s| s.to_string()).chain((1..=rounds).map(|k| format!("own_{k}"))).collect();
    let mut kl = vec![0.0; names.len()];
    let mut active = vec![vec![0.0; layers]; names.len()];
    let mut tokens = 0usize;
    let masks_of = |trace: &DeviceTrace, outputs: &[usize]| -> Result<Vec<Tensor>, String> { outputs.iter().map(|n| Ok(d.copy(trace.value(*n)?).map_err(error)?)).collect() };
    for chunk in held_out.chunks(batch) {
        let views: Vec<&[u32]> = chunk.iter().map(Vec::as_slice).collect();
        let family = sequence_family(&views)?;
        let length = views[0].len();
        tokens += family.rows;
        let (_, m_streams) = streams(&vpd.m, &family, |_| BTreeMap::new())?;
        let m_hidden = vpd.m.hidden_of(&family, &m_streams[layers])?;
        // VPD run with `masks` (δ = 0), scored as setting `k`: its trace.
        let mut score = |k: usize, masks: &[Tensor]| -> Result<DeviceTrace, String> {
            let mut given = BTreeMap::new();
            for (s, &(_, _, width)) in vpd.sites.iter().enumerate() {
                given.insert(vpd.layout.masks[s], d.copy(&masks[s]).map_err(error)?);
                given.insert(vpd.layout.deltas[s], d.zeros(family.rows, width).map_err(error)?);
            }
            let trace = vpd.e.program.forward_given(&family, given)?;
            let (nats, _) = divergence_and_seed(&d, &m_hidden, trace.value(vpd.e.hidden)?, &vpd.e.head, length, false, arithmetic)?;
            kl[k] += nats;
            let g: Vec<Array2<f64>> = masks.iter().map(|m| d.download(m).map_err(error)).collect::<Result<_, _>>()?;
            let (_, per_layer) = active_counts(&g, &vpd.sites, layers);
            for (acc, counts) in active[k].iter_mut().zip(per_layer) {
                *acc += counts.iter().sum::<f64>();
            }
            Ok(trace)
        };
        score(0, &masks_of(&vpd.importance.forward(&family)?, &vpd.outputs)?)?;
        score(1, &masks_of(&causal.forward(&family)?, &causal_outputs)?)?;
        let ones: Vec<Tensor> = vpd.sites.iter().map(|&(_, c, _)| d.upload(Array2::<f64>::ones((family.rows, c)).view()).map_err(error)).collect::<Result<_, _>>()?;
        let mut trace = score(2, &ones)?;
        for k in 0..rounds {
            // The network on VPD's own inputs to the sites: `M`'s inputs replaced by them.
            let inputs: BTreeMap<usize, Tensor> =
                vpd.layout.inputs.iter().zip(&vpd.m_layout.inputs).map(|(own, m)| Ok((*m, d.copy(trace.value(*own)?).map_err(error)?))).collect::<Result<_, String>>()?;
            let replaced = |network: &DeviceProgram, outputs: &[usize]| -> Result<Vec<Tensor>, String> {
                let ci = network.forward_edited(&family, BTreeMap::new(), &std::collections::BTreeSet::new(), |_, _| Ok(()), |node, _| {
                    inputs.get(&node).map(|v| d.copy(v).map_err(error)).transpose()
                })?;
                masks_of(&ci, outputs)
            };
            if k == 0 {
                score(3, &replaced(&causal, &causal_outputs)?)?;
            }
            let masks = replaced(&vpd.importance, &vpd.outputs)?;
            trace = score(4 + k, &masks)?;
        }
    }
    let rows = tokens as f64;
    Ok(Value::Object(
        names
            .iter()
            .enumerate()
            .map(|(k, name)| {
                (
                    name.clone(),
                    json!({
                        "kl_bits_per_token": kl[k] / rows / LN_2,
                        "active_per_token": active[k].iter().sum::<f64>() / rows,
                        "active_per_token_per_layer": active[k].iter().map(|a| a / rows).collect::<Vec<f64>>(),
                    }),
                )
            })
            .collect(),
    ))
}

fn d_copy(d: &Device, t: &Tensor) -> Result<Tensor, String> {
    d.copy(t).map_err(error)
}

// ------------------------------------------------------------------------------ VPD, one position

/// One of `M`'s read variables (`interchange::reads`): head `h`'s query, key or value vector
/// (`part` 0, 1, 2) at layer `layer`'s attention block, or neuron `index`'s pre-activation at its
/// MLP block.
#[derive(Clone, Copy, Debug)]
enum Variable {
    Head { layer: usize, head: usize, part: usize },
    Neuron { layer: usize, index: usize },
}

impl Variable {
    fn block(self) -> usize {
        match self {
            Self::Head { layer, .. } => 2 * layer,
            Self::Neuron { layer, .. } => 2 * layer + 1,
        }
    }

    /// The node computing the variable in a program of `layout`, and its columns there.
    fn place(self, layout: &Layout) -> (usize, std::ops::Range<usize>, bool) {
        match self {
            Self::Head { layer, head, part } => (layout.head_variables[layer][head][part], 0..0, true),
            Self::Neuron { layer, index } => (layout.pre_activations[layer], index..index + 1, false),
        }
    }
}

/// Every read variable of `M` (per layer each head's query, key and value, then each neuron).
fn variables(layout: &Layout, neurons: usize) -> Vec<Variable> {
    let mut out = Vec::new();
    for (layer, heads) in layout.head_variables.iter().enumerate() {
        for head in 0..heads.len() {
            out.extend((0..3).map(|part| Variable::Head { layer, head, part }));
        }
        out.extend((0..neurons).map(|index| Variable::Neuron { layer, index }));
    }
    out
}

/// Row `row` of a read `h` patched with the source's row `s` outside the span `q`: the
/// complement `s + (h − s) Q Qᵀ`; with no span it takes the source whole.
fn complement_row(h: &mut Array2<f64>, row: usize, s: ndarray::ArrayView1<f64>, q: Option<&Array2<f32>>) {
    let base = h.row(row).to_owned();
    let mut out = s.to_owned();
    if let Some(q) = q {
        let difference = (&base - &s).mapv(|x| x as f32);
        let along = q.dot(&difference.dot(q));
        out.zip_mut_with(&along, |o, a| *o += f64::from(*a));
    }
    h.row_mut(row).assign(&out);
}

/// VPD's interchange experiments as the battery asks them of a library explanation
/// (`interchange`, module note there): each patch at one position `t₀` per base (uniform), scored
/// from `t₀` on, the source a sequence shared across all bases; VPD under the base's CI masks, the
/// remainder dropped. Families: `read`, one of `M`'s read variables drawn uniformly per base (a
/// head's query, key or value vector, or a neuron's pre-activation: the same questions asked of
/// every explanation), its value at `t₀` replaced by the value the same model computes on the
/// source (VPD under the source's own CI masks), every other variable reading the base's stream;
/// `read_joint`, a subset of the variables at that variable's block (its size uniform,
/// `interchange::subset`) patched together; `complement_block_b`, at block `b` the component of
/// `M`'s normed stream outside the span of the reads of VPD's subcomponents active at `t₀`; and
/// `complement_joint`, the complements at a set of blocks of size uniform in `2..=2L`
/// (`interchange::hybrid_of`). VPD reads only its active coordinates, which a complement keeps, so
/// its prediction there is its unpatched output.
pub fn vpd_interchange_atomic(vpd: &Vpd, bases: &[Vec<u32>], sources: &[Vec<u32>], batch: usize, seed: u64, worst_of: &[usize]) -> Result<Value, String> {
    let d = vpd.m.program.device().clone();
    let (layers, blocks) = (vpd.layers(), 2 * vpd.layers());
    let length = bases.first().map_or(0, Vec::len);
    let neurons = vpd.m.program.widths()[vpd.m_layout.pre_activations[0]];
    let variables = variables(&vpd.m_layout, neurons);
    let mut rng = StdRng::seed_from_u64(seed);
    let position: Vec<usize> = (0..bases.len()).map(|_| rng.random_range(0..length)).collect();
    let read_of: Vec<usize> = (0..bases.len()).map(|_| rng.random_range(0..variables.len())).collect();
    let joint_of: Vec<Vec<usize>> = read_of
        .iter()
        .map(|&v| crate::interchange::subset(&mut rng, &(0..variables.len()).filter(|i| variables[*i].block() == variables[v].block()).collect::<Vec<_>>()))
        .collect();
    let blocks_of: Vec<Vec<usize>> = (0..bases.len())
        .map(|_| {
            let k = rng.random_range(2..=blocks);
            crate::interchange::hybrid_of(&mut rng, blocks, k).iter().enumerate().filter(|(_, x)| **x).map(|(b, _)| b).collect()
        })
        .collect();
    // The nodes a side's variables and reads live at.
    let nodes_of = |layout: &Layout, reads: bool| -> Vec<usize> {
        let mut nodes: Vec<usize> = layout.head_variables.iter().flatten().flatten().copied().chain(layout.pre_activations.iter().copied()).collect();
        if reads {
            nodes.extend(&layout.reads);
        }
        nodes
    };
    let (m_nodes, e_nodes) = (nodes_of(&vpd.m_layout, true), nodes_of(&vpd.layout, false));
    // A forward through `side` on `family` whose node values among `nodes` pass through `patch`.
    let run = |side: &Side, family: &FamilyInputs, masks: Option<&Masks>, nodes: &[usize], patch: &mut dyn FnMut(usize, &mut Array2<f64>) -> bool| -> Result<Tensor, String> {
        let mut x: Option<Tensor> = None;
        for l in 0..layers {
            let given = match masks {
                Some(m) => vpd.given(&d, l, Some(m), family.rows)?,
                None => BTreeMap::new(),
            };
            let trace = side.layer_trace(family, l, x.as_ref(), given, |node, trace| {
                if !nodes.contains(&node) {
                    return Ok(None);
                }
                let mut h = d.download(trace.value(node)?).map_err(error)?;
                if !patch(node, &mut h) {
                    return Ok(None);
                }
                d.upload(h.view()).map(Some).map_err(error)
            })?;
            x = Some(d.copy(trace.value(side.leaving(l))?).map_err(error)?);
        }
        x.ok_or_else(|| error("no layers"))
    };
    // Each source's values at every variable (and, for M, every read): M's, and VPD's under the
    // source's own CI masks.
    let mut source_values: Vec<(BTreeMap<usize, Array2<f64>>, BTreeMap<usize, Array2<f64>>)> = Vec::with_capacity(sources.len());
    for source in sources {
        let family = sequence_family(&[source.as_slice()])?;
        let masks = Strategy::Ci.masks(&vpd.importances(&family)?, &mut rng);
        let capture = |side: &Side, masks: Option<&Masks>, nodes: &[usize]| -> Result<BTreeMap<usize, Array2<f64>>, String> {
            let mut out = BTreeMap::new();
            run(side, &family, masks, nodes, &mut |node, h| {
                out.insert(node, h.clone());
                false
            })?;
            Ok(out)
        };
        source_values.push((capture(&vpd.m, None, &m_nodes)?, capture(&vpd.e, Some(&masks), &e_nodes)?));
    }
    let mut names = vec!["read".to_string(), "read_joint".to_string()];
    names.extend((0..blocks).map(|b| format!("complement_block_{b}")));
    names.push("complement_joint".into());
    let mut families = Families { per_source: vec![vec![(0.0, 0); sources.len()]; names.len()], all: vec![Tokens::default(); names.len()], names };
    for (c, chunk) in bases.chunks(batch).enumerate() {
        let views: Vec<&[u32]> = chunk.iter().map(Vec::as_slice).collect();
        let family = sequence_family(&views)?;
        let g = vpd.importances(&family)?;
        let masks = Strategy::Ci.masks(&g, &mut rng);
        let index = |r: usize| c * batch + r;
        // The spans of VPD's active reads at each base's position, per block.
        let active: Vec<Vec<Option<Array2<f32>>>> = (0..blocks)
            .map(|b| -> Result<Vec<Option<Array2<f32>>>, String> {
                let sites = reading_sites(b);
                let v = ndarray::concatenate(Axis(1), &sites.iter().map(|s| vpd.factors[*s].v.view()).collect::<Vec<_>>()).map_err(error)?;
                (0..chunk.len())
                    .map(|r| {
                        let t = r * length + position[index(r)];
                        let mut cols = Vec::new();
                        let mut offset = 0;
                        for &s in &sites {
                            cols.extend(g[s].row(t).iter().enumerate().filter(|(_, x)| **x > 0.0).map(|(k, _)| offset + k));
                            offset += g[s].ncols();
                        }
                        column_span(&v, &cols)
                    })
                    .collect()
            })
            .collect::<Result<_, _>>()?;
        // VPD's unpatched output.
        let mut x: Option<Tensor> = None;
        for l in 0..layers {
            x = Some(vpd.e.layer(&family, l, x.as_ref(), vpd.given(&d, l, Some(&masks), family.rows)?)?);
        }
        let e_logits = vpd.m.logits(&family, x.as_ref().ok_or_else(|| error("no layers"))?, length)?;
        for (s, (m_source, e_source)) in source_values.iter().enumerate() {
            // KL(M_e ‖ E_e) per base from its position on.
            let score = |families: &mut Families, f: usize, m_final: &Tensor, e_logits: &[Array2<f64>]| -> Result<(), String> {
                let reference = Reference::of(vpd.m.logits(&family, m_final, length)?)?;
                let mut behaviour = Behaviour::default();
                behaviour.add(&reference, e_logits, &views)?;
                let bits: Vec<f64> = (0..chunk.len()).flat_map(|r| behaviour.kl.0[r * length + position[index(r)]..(r + 1) * length].to_vec()).collect();
                families.add(f, s, &bits);
                Ok(())
            };
            for f in 0..2 {
                // Each base's variable (or its joint set): at its node, its columns at `t₀` take the
                // source's values from the same model.
                let patch_with = |layout: &Layout, source: &BTreeMap<usize, Array2<f64>>, node: usize, h: &mut Array2<f64>| -> bool {
                    let mut any = false;
                    for r in 0..chunk.len() {
                        let chosen = if f == 0 { std::slice::from_ref(&read_of[index(r)]) } else { joint_of[index(r)].as_slice() };
                        for &v in chosen {
                            let (at, columns, whole) = variables[v].place(layout);
                            if at != node {
                                continue;
                            }
                            let (t, value) = (position[index(r)], &source[&node]);
                            let columns = if whole { 0..h.ncols() } else { columns };
                            h.slice_mut(s![r * length + t, columns.clone()]).assign(&value.slice(s![t, columns]));
                            any = true;
                        }
                    }
                    any
                };
                let m_final = run(&vpd.m, &family, None, &m_nodes, &mut |node, h| patch_with(&vpd.m_layout, m_source, node, h))?;
                let e_final = run(&vpd.e, &family, Some(&masks), &e_nodes, &mut |node, h| patch_with(&vpd.layout, e_source, node, h))?;
                let e_patched = vpd.m.logits(&family, &e_final, length)?;
                score(&mut families, f, &m_final, &e_patched)?;
            }
            // The complements, at M's reads.
            let block_of = |node: usize| vpd.m_layout.reads.iter().position(|r| *r == node);
            for b in 0..blocks {
                let m_final = run(&vpd.m, &family, None, &m_nodes, &mut |node, h| {
                    if block_of(node) != Some(b) {
                        return false;
                    }
                    for r in 0..chunk.len() {
                        let t = position[index(r)];
                        complement_row(h, r * length + t, m_source[&node].row(t), active[b][r].as_ref());
                    }
                    true
                })?;
                score(&mut families, 2 + b, &m_final, &e_logits)?;
            }
            let m_final = run(&vpd.m, &family, None, &m_nodes, &mut |node, h| {
                let Some(at) = block_of(node) else { return false };
                let mut any = false;
                for r in 0..chunk.len() {
                    if blocks_of[index(r)].contains(&at) {
                        let t = position[index(r)];
                        complement_row(h, r * length + t, m_source[&node].row(t), active[at][r].as_ref());
                        any = true;
                    }
                }
                any
            })?;
            score(&mut families, 2 + blocks, &m_final, &e_logits)?;
        }
        log::info!("battery: VPD one-position interchange on {} bases", chunk.len());
    }
    Ok(json!({"patches": families.summary(worst_of)}))
}

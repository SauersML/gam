//! A readable account of a library explanation (`library_mdl`, #2951): for every surviving
//! function, where and how strongly it acts on held-out text, what it causes, what it writes and
//! reads in token terms, and how the functions feed one another.
//!
//! # Quantities
//!
//! An MLP function `i` of layer `l` computes `h_i = φ(g_i·x̂ + c_i)` (GELU) or
//! `h_i = φ(g_i·x̂ + c_i) (w_i·x̂)` (SwiGLU) on its layer's normed stream `x̂ = γ ⊙ x / r`,
//! `r = √(mean(x²) + ε)` the RMS of the residual stream `x` at that read, and writes `h_i u_i`.
//! A head `h` of layer `l` reads `x̂` through its query, key and value maps and writes
//! `W_O,h z_h`, `z_h` its attention read and `W_O,h` the native output projection's columns of
//! the head.
//!
//! * RelP attribution. The reverse pass of the network linearized with every norm's denominator,
//!   every law's gate factor `φ(x)/x` and every attention pattern frozen, and half of the gradient
//!   through each factor of a product (SwiGLU's gate times up), seeded with the gradient of a metric
//!   `m` in the final stream (by default the centred logit of the model's predicted token, the
//!   final norm's denominator frozen). A function's attribution of the prediction at target token
//!   `t` is `A_i(t) = Σ_{s ≤ t} h_i(s) ∂m(t)/∂h_i(s)` (a head: `z_h · ∂m(t)/∂z_h`), over every
//!   position it acts at; within a cut (one layer's heads, one MLP) the functions and the residual
//!   stream entering it account for `m(t)`. Targets are drawn per held-out sequence, and their
//!   predictions attributed one at a time.
//! * Importance: the mean of `|A_i(t)|` over held-out tokens, which ranks the functions; per
//!   threshold `τ`, the fraction of held-out tokens with `|A_i(t)| > τ |m(t)|` (causal activity).
//!   Background functions are important at the first threshold on every held-out token.
//! * Positive-gate fraction (MLP functions): the fraction of tokens with a positive gate
//!   pre-activation. It is not activity: GELU and SiLU are nonzero on both sides, and a negative
//!   gate times a large up value is a large output. It is not comparable with VPD's L0.
//! * Usage: the mean output norm `|h_i| ‖u_i‖` (heads `‖W_O,h z_h‖`) over held-out tokens.
//!   Contexts: the held-out tokens with the largest output norm (heads: with the position the
//!   head attends to most there).
//! * Supports and opposes: the held-out tokens with the largest positive and negative attributions,
//!   and the predicted token a majority of each set shares.
//! * Writes: the output direction through the final norm's gain and the unembedding,
//!   `W_U (γ_f ⊙ u_i)`, centred over the vocabulary; the most promoted and suppressed tokens (a
//!   direct path only). Reads: the gate direction `γ ⊙ g_i` against each token's embedding at that
//!   read, `(γ ⊙ g_i)·e_t / √(mean(e_t²) + ε)`.
//! * Flow edges: RelP's direct effect of a writer on a reader, summed over held-out tokens:
//!   `flow(s → t) = Σ_τ h_s(τ) ∂h_t/∂h_s(τ) ∂m/∂h_t(τ)` through the residual stream alone (no other
//!   function between them), in the linearized network with the reader's norm denominator frozen
//!   and the half rule. An MLP reader reads through its gate and, when gated, its up map; a head
//!   reader through its value map (frozen pattern) and its query and key maps (the scores
//!   `c q·k` split half to each factor, through the softmax's derivative and the head's query and
//!   key norms, their denominators frozen). Flows are computed among the `candidates` most
//!   important functions; each keeps its strongest inputs, and the most important (`core`) their
//!   complete wiring.
//! * Attention (heads): the attention mass by query-key offset in binary orders of magnitude, the
//!   source tokens receiving the most mass, the OV map's largest token-to-token entries over the
//!   held-out text's tokens (`self_top1`: the fraction of source tokens whose largest output is
//!   themselves), and the previous-token, induction and duplicate-token scores.
//!
//! The model runs on its device in that device's arithmetic; the vocabulary-wide searches (top
//! tokens, OV entries) run on the products device; every reported token score is recomputed in
//! float64 on the host from the selected tokens.
use crate::{
    artifact::Artifact,
    artifact_device::mapped_inlined_observed,
    device_program::DeviceProgram,
    library_mdl::{Explanation, Posterior, sequence_family},
    operator_program::{Node, OperatorProgram, Rotary, Rule, rms_scale},
    resident_causal_fit::fixed_head_target::Head,
    run_check::LayerNodes,
    tiled_attention::{probabilities, rotate},
};
use gam_gpu::tensor::{Arithmetic, Device, Op, Storage, Tensor};
use ndarray::{Array1, Array2, ArrayView1, Axis, s};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn arithmetic(device: &Device) -> Arithmetic {
    match device.storage() {
        Storage::F64 => Arithmetic::F64,
        Storage::F32 | Storage::Bf16 => Arithmetic::F32,
    }
}

/// How much of the account to report, and the run's resources.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    /// Held-out sequences run at a time, the program's operator buffers, and rows of
    /// vocabulary-wide products formed at once.
    pub batch_sequences: usize,
    pub numeric_bytes: usize,
    pub tile_rows: usize,
    /// Per function: contexts, tokens of each token list, and strongest inputs reported.
    pub contexts: usize,
    pub tokens: usize,
    pub edges: usize,
    /// The functions of largest RelP importance among which flows are computed, and those (the
    /// background apart) whose complete wiring among themselves is reported.
    pub candidates: usize,
    pub core: usize,
    /// The fractions `τ` of `|m(t)|` above which a function's `|A_i(t)|` counts it as causally
    /// important at token `t`.
    pub thresholds: Vec<f64>,
    /// Target tokens drawn per held-out sequence (uniformly, without replacement) whose
    /// predictions are attributed one at a time.
    pub targets: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Context {
    pub sequence: usize,
    pub position: usize,
    pub value: f64,
    /// Heads: the position attended to most.
    pub source: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TokenScore {
    pub token: u32,
    pub score: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Entry {
    pub source: u32,
    pub output: u32,
    pub score: f64,
}

/// An input of a function: the writer (an index into `Readout::functions`), its mean flow per
/// held-out token, and that flow per route of the reader's reads: a head reader's value, query
/// and key maps; an MLP reader's gate and up maps (and zero).
#[derive(Clone, Debug, Serialize)]
pub struct Edge {
    pub from: usize,
    pub flow: f64,
    pub routes: [f64; 3],
}

#[derive(Clone, Debug, Serialize)]
pub struct Attention {
    /// Inclusive offset ranges and the mean attention mass per query in each.
    pub offsets: Vec<[usize; 2]>,
    pub offset_mass: Vec<f64>,
    /// Source tokens by their share of all attention mass.
    pub sources: Vec<TokenScore>,
    pub ov: Vec<Entry>,
    pub self_top1: f64,
    /// Mean attention to the previous token on held-out text; on sequences of random held-out
    /// tokens repeated once, the mean attention from the second copy to the token after the
    /// first occurrence (induction) and to the first occurrence itself (duplicate token).
    pub previous: f64,
    pub induction: f64,
    pub duplicate: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Head,
    Mlp,
}

#[derive(Clone, Debug, Serialize)]
pub struct Function {
    pub name: String,
    pub layer: usize,
    pub kind: Kind,
    /// MLP functions: the fraction of held-out tokens with a positive gate pre-activation (not
    /// activity; module note).
    pub positive_gate_fraction: Option<f64>,
    /// Per threshold `τ`, the fraction of held-out tokens with `|A_i(t)| > τ |m(t)|`.
    pub important_fraction: Vec<f64>,
    pub usage: f64,
    pub contexts: Vec<Context>,
    pub promoted: Vec<TokenScore>,
    pub suppressed: Vec<TokenScore>,
    pub reads: Vec<TokenScore>,
    pub inputs: Vec<Edge>,
    pub attention: Option<Attention>,
    /// Mean over held-out tokens of `|A_i(t)|` and of `A_i(t)`, `A_i(t)` the function's RelP
    /// attribution of the model's predicted-token logit (module note).
    pub importance: f64,
    pub attribution: f64,
    /// The held-out tokens with the largest positive and negative attributions, and the predicted
    /// token a majority of each set shares (none when no token holds a majority).
    pub supports: Vec<Context>,
    pub opposes: Vec<Context>,
    pub supports_token: Option<u32>,
    pub opposes_token: Option<u32>,
    /// Important at the first threshold on every held-out token.
    pub background: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Readout {
    pub held_out_tokens: usize,
    pub functions: Vec<Function>,
    pub removed: Vec<String>,
    /// Indices into `functions` of the most important functions, descending, and `wiring[i][j]` the
    /// mean flow per held-out token from core function `j` to core function `i` (zero when `j`
    /// does not precede `i`'s read).
    pub core: Vec<usize>,
    pub wiring: Vec<Vec<f64>>,
    /// Per held-out token, the model's predicted token, and the mean centred logit `m` of it.
    pub predicted: Vec<u32>,
    pub mean_logit: f64,
    /// The target tokens whose predictions were attributed (`Settings::targets` per sequence);
    /// importance, important fractions, supports and opposes, and the cut summaries are over them.
    pub targets: usize,
    /// Per cut of the reverse pass (one layer's heads, one MLP), its causally important functions
    /// per token.
    pub cuts: Vec<CutSummary>,
    /// Summed over cuts: per threshold `τ`, the mean number of functions per token with
    /// `|A_i(t)| > τ |m(t)|`, and the mean participation number.
    pub important: Vec<[f64; 2]>,
    pub participation: f64,
}

/// One cut's mean per token of its participation number `(Σ_i |A_i(t)|)² / Σ_i A_i(t)²` (the
/// effective number of its functions the prediction rests on) and, per threshold, of its number of
/// functions with `|A_i(t)| > τ |m(t)|`. Within a cut, the functions and the residual stream
/// entering it account for `m` (RelP's completeness), so counts are comparable across cuts.
#[derive(Clone, Debug, Serialize)]
pub struct CutSummary {
    pub name: String,
    pub participation: f64,
    pub counts: Vec<f64>,
}

// ------------------------------------------------------------------------------ the explanation

/// A read of the residual stream: its native norm's input node, gain and epsilon.
struct Site {
    input: usize,
    gain: Array1<f64>,
    epsilon: f64,
}

fn site(native: &OperatorProgram, normed: usize) -> Result<Site, String> {
    let Node::Affine { terms, .. } = &native.nodes[normed] else { return Err(format!("node {normed} is not a normed stream")) };
    let [(rms, gain)] = terms[..] else { return Err(format!("node {normed} is not one gain of a norm")) };
    let Node::RmsNorm { input, epsilon } = native.nodes[rms] else { return Err(format!("node {normed} does not read an RMS norm")) };
    Ok(Site { input, gain: native.operators[gain].matrix().diag().to_owned(), epsilon })
}

struct MlpBlock {
    layer: usize,
    call: usize,
    /// Rule nodes of the activations and of the gate pre-activations.
    activation: usize,
    gate_pre: usize,
    gate: Array2<f64>,
    bias: Array1<f64>,
    out: Array2<f64>,
    /// A gated MLP's up map: its rule node and matrix, and its bias (zero where there is none).
    up: Option<(usize, Array2<f64>)>,
    up_bias: Array1<f64>,
}

/// A head's query or key read: the rule node of its projection, the projection's map, and the
/// head norm after it (gain and epsilon), when there is one; `scored` is the rule node the
/// attention reads.
struct Read {
    scored: usize,
    projection: usize,
    map: Array2<f64>,
    norm: Option<(Array1<f64>, f64)>,
}

fn read_of(program: &OperatorProgram, rule: &Rule, node: usize) -> Result<Read, String> {
    if let Node::Affine { terms, bias: None } = &rule.nodes[node]
        && let [(normed, gain)] = terms[..]
        && let Node::RmsNorm { input, epsilon } = rule.nodes[normed]
    {
        return Ok(Read { scored: node, projection: input, map: operator_of(program, rule, input)?.0, norm: Some((program.operators[gain].matrix().diag().to_owned(), epsilon)) });
    }
    Ok(Read { scored: node, projection: node, map: operator_of(program, rule, node)?.0, norm: None })
}

struct HeadBlock {
    layer: usize,
    head: usize,
    call: usize,
    query: Read,
    key: Read,
    value_node: usize,
    value: Array2<f64>,
    output: Array2<f64>,
    scale: f64,
    rotary: Option<Rotary>,
    causal: bool,
}

fn operator_of(program: &OperatorProgram, rule: &Rule, node: usize) -> Result<(Array2<f64>, Option<usize>), String> {
    match &rule.nodes[node] {
        Node::Affine { terms, bias } if terms.len() == 1 => Ok((program.operators[terms[0].1].matrix(), *bias)),
        other => Err(format!("{}: node {node} is {other:?}, not one map", rule.name)),
    }
}

fn mlp_block(program: &OperatorProgram, rule: &Rule, layer: usize, call: usize) -> Result<MlpBlock, String> {
    let Node::Affine { terms, bias: None } = &rule.nodes[rule.output] else { return Err(format!("{}: the output is not one map", rule.name)) };
    let [(activation, out)] = terms[..] else { return Err(format!("{}: the output reads more than the activations", rule.name)) };
    let (gate_active, up, up_bias) = match &rule.nodes[activation] {
        Node::Pointwise { .. } => (activation, None, None),
        Node::Hadamard { left, right } => {
            let law = [*left, *right].into_iter().find(|n| matches!(rule.nodes[*n], Node::Pointwise { .. })).ok_or("a gated MLP without a law")?;
            let up = if law == *left { *right } else { *left };
            let (map, bias) = operator_of(program, rule, up)?;
            (law, Some((up, map)), bias)
        }
        other => return Err(format!("{}: activations are {other:?}", rule.name)),
    };
    let Node::Pointwise { input: gate_pre, .. } = rule.nodes[gate_active] else { return Err("no gate".into()) };
    let (gate, bias) = operator_of(program, rule, gate_pre)?;
    let bias = match bias {
        Some(op) => program.operators[op].matrix().column(0).to_owned(),
        None => Array1::zeros(gate.nrows()),
    };
    let up_bias = match up_bias {
        Some(op) => program.operators[op].matrix().column(0).to_owned(),
        None => Array1::zeros(gate.nrows()),
    };
    Ok(MlpBlock { layer, call, activation, gate_pre, gate, bias, out: program.operators[out].matrix(), up, up_bias })
}

fn head_block(program: &OperatorProgram, rule: &Rule, layer: usize, head: usize, call: usize, output: Array2<f64>) -> Result<HeadBlock, String> {
    let Node::Attend { query, key, value, scale, rotary, causal } = rule.nodes[rule.output] else {
        return Err(format!("{}: the output is not an attention", rule.name));
    };
    let (query, key) = (read_of(program, rule, query)?, read_of(program, rule, key)?);
    Ok(HeadBlock { layer, head, call, query, key, value_node: value, value: operator_of(program, rule, value)?.0, output, scale: scale.value(), rotary, causal })
}

/// `library.l{l}.h{h}` or `library.l{l}.mlp`: the layer and the head.
fn parse(name: &str) -> Option<(usize, Option<usize>)> {
    let (layer, part) = name.strip_prefix("library.l")?.split_once('.')?;
    let layer = layer.parse().ok()?;
    if part == "mlp" {
        return Some((layer, None));
    }
    Some((layer, Some(part.strip_prefix('h')?.parse().ok()?)))
}

/// The native output projection's columns of each head of `layer`.
fn output_columns(native: &OperatorProgram, layer: &LayerNodes) -> Result<Vec<Array2<f64>>, String> {
    let Node::Affine { terms, .. } = &native.nodes[layer.attention] else { return Err("the attention output is not a map".into()) };
    layer
        .reads
        .iter()
        .map(|read| terms.iter().find(|(n, _)| n == read).map(|(_, op)| native.operators[*op].matrix()).ok_or_else(|| "a head without output columns".to_string()))
        .collect()
}


// ------------------------------------------------------------------------------ accumulation

/// The `k` largest values offered with their ids, descending.
#[derive(Clone, Default)]
struct Best {
    entries: Vec<(f64, usize)>,
}

impl Best {
    fn offer(&mut self, k: usize, value: f64, id: usize) {
        if k == 0 || self.entries.len() == k && self.entries.last().is_some_and(|e| value <= e.0) {
            return;
        }
        let at = self.entries.partition_point(|e| e.0 >= value);
        self.entries.insert(at, (value, id));
        self.entries.truncate(k);
    }
}

#[derive(Clone, Default)]
struct MlpStat {
    positive: f64,
    abs: f64,
    top: Best,
    relp: Relp,
}

/// A function's RelP attributions: `Σ |A|`, `Σ A`, per threshold the tokens with
/// `|A| > τ |m|`, and the rows of the largest and of the most negative.
#[derive(Clone, Default)]
struct Relp {
    absolute: f64,
    signed: f64,
    important: Vec<f64>,
    supports: Best,
    opposes: Best,
}

impl Relp {
    /// The attribution `a` of the prediction `m` at target row `id`.
    fn add(&mut self, k: usize, a: f64, m: f64, thresholds: &[f64], id: usize) {
        self.important.resize(thresholds.len(), 0.0);
        self.absolute += a.abs();
        self.signed += a;
        for (count, tau) in self.important.iter_mut().zip(thresholds) {
            if a.abs() > tau * m.abs() {
                *count += 1.0;
            }
        }
        self.supports.offer(k, a, id);
        self.opposes.offer(k, -a, id);
    }
}

struct HeadStat {
    output: f64,
    top: Best,
    offsets: Vec<f64>,
    sources: HashMap<u32, f64>,
    /// Per held-out row, the position attended to most.
    attended: Vec<usize>,
    relp: Relp,
    /// Summed attention to the induction and the duplicate-token positions.
    induction: f64,
    duplicate: f64,
}

/// The binary order of magnitude of an offset: 0, 1, 2–3, 4–7, …
fn offset_bin(offset: usize) -> usize {
    (usize::BITS - offset.leading_zeros()) as usize
}

// ------------------------------------------------------------------------------ products

/// Per row of `rows`, the `k` columns of `sign · rows · tableᵀ` largest, descending, found on
/// `device` (the table resident there) in tiles of `tile` rows.
fn extreme_columns(device: &Device, table: &Tensor, rows: &Array2<f64>, k: usize, sign: f64, tile: usize) -> Result<Vec<Vec<usize>>, String> {
    let width = table.rows();
    let k = k.min(width);
    let mut out = Vec::with_capacity(rows.nrows());
    for start in (0..rows.nrows()).step_by(tile) {
        let n = tile.min(rows.nrows() - start);
        let block = device.upload(rows.slice(s![start..start + n, ..])).map_err(error)?;
        let mut products = device.zeros(n, width).map_err(error)?;
        device.gemm(&mut products, sign, &block, Op::N, table, Op::T, 0.0, arithmetic(device)).map_err(error)?;
        let mut chosen = vec![Vec::with_capacity(k); n];
        for _ in 0..k {
            let best = device.argmax_rows(&products).map_err(error)?;
            let positions: Vec<u32> = best.iter().enumerate().map(|(r, c)| u32::try_from(r * width + c).map_err(error)).collect::<Result<_, _>>()?;
            device.fill_entries(&mut products, &device.upload_indices(&positions).map_err(error)?, f64::NEG_INFINITY).map_err(error)?;
            for (r, c) in best.into_iter().enumerate() {
                chosen[r].push(c);
            }
        }
        out.extend(chosen);
    }
    Ok(out)
}

/// Each row of `table` divided by `√(mean(row²) + ε)`.
fn normed_rows(table: Array2<f64>, epsilon: f64) -> Array2<f64> {
    let mut out = table;
    out.outer_iter_mut().into_par_iter().for_each(|mut row| {
        let r = (row.iter().map(|v| v * v).sum::<f64>() / row.len() as f64 + epsilon).sqrt();
        row.mapv_inplace(|v| v / r);
    });
    out
}

fn scored(tokens: &[usize], score: impl Fn(usize) -> f64) -> Vec<TokenScore> {
    tokens.iter().map(|&t| TokenScore { token: t as u32, score: score(t) }).collect()
}

fn dot(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.dot(&b)
}

// ------------------------------------------------------------------------------ the library

/// A library explanation prepared for forward passes and RelP attribution: its blocks, the native
/// reads and readout they act through, and its program compiled with every read, activation and
/// attention input observed.
pub struct Library<'a> {
    model: &'a Device,
    wide: &'a Device,
    tile_rows: usize,
    /// Reads in depth order: site `2l` is layer `l`'s attention read, `2l + 1` its MLP read.
    sites: Vec<Site>,
    final_site: Site,
    mlps: Vec<MlpBlock>,
    heads: Vec<HeadBlock>,
    /// Per layer, its heads (indices into `heads`).
    layer_heads: Vec<Vec<usize>>,
    /// The unembedding with the final norm's gain (vocabulary × width), its mean row, and its
    /// copy on `wide`.
    unembedding: Array2<f64>,
    unembedding_mean: Array1<f64>,
    unembedding_table: Tensor,
    program: DeviceProgram,
    observed: Vec<usize>,
    mlp_paths: Vec<usize>,
    head_paths: usize,
}

/// One forward pass's observed values on sequences of one length.
struct Pass {
    rows: usize,
    /// Per site, the residual stream it reads and `1/r` per row; the final stream and its `1/r`.
    streams: Vec<Array2<f64>>,
    inverse: Vec<Array1<f64>>,
    last: Array2<f64>,
    inverse_final: Array1<f64>,
    /// Per MLP: its activations, gate pre-activations and, when gated, up values.
    mlp: Vec<(Array2<f64>, Array2<f64>, Option<Array2<f64>>)>,
    /// Per head: the query and key it scores, its read, its value, and its query and key
    /// projections (before its head norms); per head and sequence, its attention weights.
    head: Vec<[Array2<f64>; 6]>,
    weights: Vec<Vec<Array2<f64>>>,
}

impl Pass {
    /// The first `end` rows of sequence `sequence` (of `length` rows) alone.
    fn prefix(&self, sequence: usize, length: usize, end: usize) -> Pass {
        let rows = sequence * length..sequence * length + end;
        let cut = |x: &Array2<f64>| x.slice(s![rows.clone(), ..]).to_owned();
        Pass {
            rows: end,
            streams: self.streams.iter().map(&cut).collect(),
            inverse: self.inverse.iter().map(|v| v.slice(s![rows.clone()]).to_owned()).collect(),
            last: cut(&self.last),
            inverse_final: self.inverse_final.slice(s![rows.clone()]).to_owned(),
            mlp: self.mlp.iter().map(|(a, p, u)| (cut(a), cut(p), u.as_ref().map(cut))).collect(),
            head: self.head.iter().map(|values| std::array::from_fn(|j| cut(&values[j]))).collect(),
            weights: self.weights.iter().map(|w| vec![w[sequence].slice(s![..end, ..end]).to_owned()]).collect(),
        }
    }
}

/// A cut of the reverse pass: one MLP's functions (an index into the MLPs) or one layer's heads
/// (indices into the heads).
enum Cut<'c> {
    Mlp(usize),
    Heads(usize, &'c [usize]),
}

/// What a cut's functions read, as the gradient of `m` in the linearized network with respect to
/// each read's value per row: an MLP's gate pre-activations and up values (rows × functions); per
/// head of the cut its value, and (when routes are asked for) its query and key projections (rows
/// × head width).
enum Reads<'r> {
    Mlp { gate: &'r Array2<f64>, up: Option<&'r Array2<f64>> },
    Heads(&'r HeadReads),
}

struct HeadReads {
    value: Vec<Array2<f64>>,
    query: Vec<Array2<f64>>,
    key: Vec<Array2<f64>>,
}

/// What the reverse pass hands its visitor at a cut: the cut, its attributions (rows × its
/// functions), its reads, and `∂m/∂x` for the residual stream `x` right after the cut.
struct Visit<'v> {
    cut: Cut<'v>,
    attribution: Array2<f64>,
    reads: Reads<'v>,
    after: &'v Array2<f64>,
}

/// One cut's completeness: the sum of its functions' attributions, the attribution `Σ ∂m/∂x · x`
/// of the residual stream entering it along the skip connection, and the attribution of the
/// biases of the functions after it; in the linearized network the three sum to `m`.
#[derive(Clone, Debug, Serialize)]
pub struct CutCheck {
    pub name: String,
    pub functions: f64,
    pub stream: f64,
    pub biases: f64,
    pub metric: f64,
}

impl CutCheck {
    /// `|functions + stream + biases − m| / |m|`.
    #[must_use]
    pub fn gap(&self) -> f64 {
        ((self.functions + self.stream + self.biases - self.metric) / self.metric.abs()).abs()
    }
}

/// A layer's gradients of the metric in the linearized network (`Prompt::gradients`): with
/// respect to the residual stream after its attention and after its MLP (the attention and MLP
/// outputs), each head's value, and its MLP's gate pre-activations (and up values when gated).
#[derive(Clone, Debug)]
pub struct LayerGradients {
    pub attention_output: Array2<f64>,
    pub mlp_output: Array2<f64>,
    pub values: Vec<Array2<f64>>,
    pub gate: Array2<f64>,
    pub up: Option<Array2<f64>>,
}

/// A prompt to attribute: its tokens, an optional baseline of the same length (attributions are
/// then of the difference of each function's value from its value on the baseline), and the
/// metric.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prompt {
    pub tokens: Vec<u32>,
    #[serde(default)]
    pub baseline: Option<Vec<u32>>,
    pub metric: Metric,
    /// Whether to return each layer's gradients ([`LayerGradients`]).
    #[serde(default)]
    pub gradients: bool,
}

/// What is attributed: the sum over positions of the centred logit of the model's predicted token
/// (a function's attribution at a position then includes its effect on later positions'
/// predictions), or at one position the logit of `target` minus the logit of `foil`.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Metric {
    Predicted,
    Difference { position: usize, target: u32, foil: u32 },
}

/// A prompt's attributions: per position the metric `m` (zero where it is not taken) and the
/// predicted token, and per position and function (columns in [`Library::functions`]'s order)
/// `A_i(t)` and the function's output norm (`|h_i| ‖u_i‖`, a head's `‖W_O,h z_h‖`); per layer its
/// gradients when the prompt asks for them.
#[derive(Clone, Debug)]
pub struct PromptAttribution {
    pub metric: Vec<f64>,
    pub predicted: Vec<u32>,
    pub attributions: Array2<f64>,
    pub outputs: Array2<f64>,
    pub gradients: Option<Vec<LayerGradients>>,
}

/// A function's identity: its name, layer and kind.
#[derive(Clone, Debug, Serialize)]
pub struct FunctionId {
    pub name: String,
    pub layer: usize,
    pub kind: Kind,
}

impl<'a> Library<'a> {
    /// `artifact`, a library explanation of the split native program `native`
    /// (`run_check::split_sites`) with its `layers`, compiled on `model` within `numeric_bytes` of
    /// operator buffers; vocabulary-wide products run on `wide` in tiles of `tile_rows` rows.
    pub fn new(model: &'a Device, wide: &'a Device, native: &OperatorProgram, layers: &[LayerNodes], artifact: &Artifact, numeric_bytes: usize, tile_rows: usize) -> Result<Self, String> {
        if numeric_bytes == 0 || tile_rows == 0 {
            return Err("positive operator buffers and tile rows required".into());
        }
        let program = &artifact.program;
        let sites: Vec<Site> = layers.iter().flat_map(|l| [site(native, l.normed_stream), site(native, l.normed)]).collect::<Result<_, _>>()?;
        let outputs: Vec<Vec<Array2<f64>>> = layers.iter().map(|l| output_columns(native, l)).collect::<Result<_, _>>()?;
        let (mut mlps, mut heads) = (Vec::new(), Vec::new());
        // The library's blocks by their bindings' names (a decoded artifact's rules are unnamed).
        for binding in &artifact.blocks {
            let Some((l, head)) = parse(&binding.name) else { continue };
            let n = binding.write;
            let Node::Call { rule, .. } = &program.nodes[n] else { return Err(format!("{}: its write is not a call", binding.name)) };
            let rule = &program.rules[*rule];
            match head {
                None => mlps.push(mlp_block(program, rule, l, n)?),
                Some(h) => {
                    let columns = outputs.get(l).and_then(|o| o.get(h)).ok_or_else(|| format!("{}: no such native head", binding.name))?;
                    heads.push(head_block(program, rule, l, h, n, columns.clone())?);
                }
            }
        }
        mlps.sort_by_key(|b| b.layer);
        heads.sort_by_key(|b| (b.layer, b.head));
        if mlps.is_empty() && heads.is_empty() {
            return Err("the artifact holds no library functions".into());
        }
        let layer_heads = (0..layers.len()).map(|l| (0..heads.len()).filter(|h| heads[*h].layer == l).collect()).collect();
        let Head { hidden, embedding: mut unembedding, .. } = Head::of(native)?;
        let final_site = site(native, hidden)?;
        unembedding.axis_iter_mut(Axis(0)).for_each(|mut row| row *= &final_site.gain);
        let unembedding_mean = unembedding.mean_axis(Axis(0)).ok_or("an empty vocabulary")?;
        let unembedding_table = wide.upload(unembedding.view()).map_err(error)?;
        // Observed values: each site's stream and the final one, each MLP's activations, gate
        // pre-activations and up values, each head's query, key and read.
        let place = |n: usize| artifact.place(n).ok_or_else(|| format!("the artifact does not hold native node {n}"));
        let mut paths: Vec<Vec<usize>> = sites.iter().map(|s| Ok(vec![place(s.input)?])).collect::<Result<_, String>>()?;
        paths.push(vec![place(final_site.input)?]);
        let mut mlp_paths = Vec::new();
        for b in &mlps {
            mlp_paths.push(paths.len());
            paths.extend([vec![b.call, b.activation], vec![b.call, b.gate_pre]]);
            if let Some((up, _)) = &b.up {
                paths.push(vec![b.call, *up]);
            }
        }
        let head_paths = paths.len();
        for b in &heads {
            paths.extend([vec![b.call, b.query.scored], vec![b.call, b.key.scored], vec![b.call], vec![b.call, b.value_node], vec![b.call, b.query.projection], vec![b.call, b.key.projection]]);
        }
        let (flat, _, observed) = mapped_inlined_observed(program, &paths)?;
        let prefix = Head::of(&flat)?.prefix(&flat);
        drop(flat);
        let mut compiled = DeviceProgram::compile_values_bounded(model, &prefix, numeric_bytes)?;
        drop(prefix);
        compiled.set_arithmetic(arithmetic(model));
        Ok(Self {
            model,
            wide,
            tile_rows,
            sites,
            final_site,
            mlps,
            heads,
            layer_heads,
            unembedding,
            unembedding_mean,
            unembedding_table,
            program: compiled,
            observed,
            mlp_paths,
            head_paths,
        })
    }

    /// Every function in the order of attribution columns: per layer its heads, then its MLP
    /// functions.
    #[must_use]
    pub fn functions(&self) -> Vec<FunctionId> {
        let mut out = Vec::new();
        for (l, members) in self.layer_heads.iter().enumerate() {
            out.extend(members.iter().map(|h| FunctionId { name: format!("L{l}.H{}", self.heads[*h].head), layer: l, kind: Kind::Head }));
            for block in self.mlps.iter().filter(|b| b.layer == l) {
                out.extend((0..block.gate.nrows()).map(|i| FunctionId { name: format!("L{l}.M{i}"), layer: l, kind: Kind::Mlp }));
            }
        }
        out
    }

    /// Per head and sequence of `count` sequences of `length`, the attention weights from the
    /// observed queries and keys.
    fn attend(&self, values: &[[Array2<f64>; 6]], count: usize, length: usize) -> Vec<Vec<Array2<f64>>> {
        let positions: Vec<u32> = (0..length as u32).collect();
        self.heads
            .par_iter()
            .zip(values.par_iter())
            .map(|(block, [q, key, ..])| {
                (0..count)
                    .map(|s| {
                        let span = s * length..(s + 1) * length;
                        let (q, k) = (q.slice(s![span.clone(), ..]).to_owned(), key.slice(s![span, ..]).to_owned());
                        let (q, k) = (rotate(&q, block.rotary, &positions, false), rotate(&k, block.rotary, &positions, false));
                        probabilities(q.view(), k.view(), &positions, 0, block.scale, block.causal)
                    })
                    .collect()
            })
            .collect()
    }

    /// The forward pass on `batch` (sequences of one length).
    fn pass(&self, batch: &[&[u32]]) -> Result<Pass, String> {
        let family = sequence_family(batch)?;
        let trace = self.program.forward(&family)?;
        let get = |i: usize| -> Result<Array2<f64>, String> { self.model.download(trace.value(self.observed[i])?).map_err(error) };
        let inverse_of = |x: &Array2<f64>, epsilon: f64| x.map_axis(Axis(1), |row| rms_scale(row, epsilon));
        let streams: Vec<Array2<f64>> = (0..self.sites.len()).map(get).collect::<Result<_, String>>()?;
        let inverse = streams.iter().zip(&self.sites).map(|(x, s)| inverse_of(x, s.epsilon)).collect();
        let last = get(self.sites.len())?;
        let inverse_final = inverse_of(&last, self.final_site.epsilon);
        let mlp = self
            .mlps
            .iter()
            .enumerate()
            .map(|(b, block)| {
                let i = self.mlp_paths[b];
                Ok((get(i)?, get(i + 1)?, if block.up.is_some() { Some(get(i + 2)?) } else { None }))
            })
            .collect::<Result<_, String>>()?;
        let p = self.head_paths;
        let head: Vec<[Array2<f64>; 6]> = (0..self.heads.len())
            .map(|h| Ok([get(p + 6 * h)?, get(p + 6 * h + 1)?, get(p + 6 * h + 2)?, get(p + 6 * h + 3)?, get(p + 6 * h + 4)?, get(p + 6 * h + 5)?]))
            .collect::<Result<_, String>>()?;
        let weights = self.attend(&head, batch.len(), batch[0].len());
        Ok(Pass { rows: family.rows, streams, inverse, last, inverse_final, mlp, head, weights })
    }

    /// The model's predicted token per row and the gradient of its centred logit in the final
    /// stream (the final norm's denominator frozen), with the centred logit itself.
    fn predicted(&self, pass: &Pass) -> Result<(Vec<usize>, Array2<f64>, Vec<f64>), String> {
        let predicted: Vec<usize> = extreme_columns(self.wide, &self.unembedding_table, &pass.last, 1, 1.0, self.tile_rows)?.into_iter().map(|c| c[0]).collect();
        let seed = Array2::from_shape_fn(pass.last.dim(), |(r, c)| (self.unembedding[[predicted[r], c]] - self.unembedding_mean[c]) * pass.inverse_final[r]);
        let metric = (0..pass.rows).map(|r| seed.row(r).dot(&pass.last.row(r))).collect();
        Ok((predicted, seed, metric))
    }

    /// RelP (module note): the reverse pass from `seed`, the metric's gradient in the final stream,
    /// with every norm's denominator, every law's gate factor `φ(x)/x` and every attention pattern
    /// frozen and half of the gradient through each factor of a product; each cut's attributions
    /// (rows × its functions) and reads go to `visit`, deepest first. With `baseline`, a function's
    /// value is taken less its value on the baseline. With `routes`, each head's reads also carry
    /// the gradient through its scores: `c q·k` split half to each factor, through the softmax's
    /// derivative and the head norms (denominators frozen); the reverse pass itself keeps the
    /// patterns frozen.
    /// Returns `∂m/∂x` for the embedding stream `x` entering the first layer.
    fn relp(&self, pass: &Pass, seed: Array2<f64>, baseline: Option<&Pass>, routes: bool, visit: &mut dyn FnMut(Visit<'_>)) -> Array2<f64> {
        let length = pass.rows / pass.weights.first().map_or(1, |w| w.len().max(1));
        let positions: Vec<u32> = (0..length as u32).collect();
        let mut g = seed;
        for l in (0..self.layer_heads.len()).rev() {
            for (b, block) in self.mlps.iter().enumerate().filter(|(_, b)| b.layer == l) {
                let (activations, pre, up) = &pass.mlp[b];
                let d = g.dot(&block.out);
                let value = match baseline {
                    Some(base) => activations - &base.mlp[b].0,
                    None => activations.clone(),
                };
                let through = |factor: &Array2<f64>, share: f64| {
                    let mut out = &d * activations;
                    ndarray::Zip::from(&mut out).and(factor).for_each(|o, f| *o = if *f == 0.0 { 0.0 } else { share * *o / f });
                    out
                };
                let (gate, up_gradient) = match (up, &block.up) {
                    (Some(up_values), Some(_)) => (through(pre, 0.5), Some(through(up_values, 0.5))),
                    _ => (through(pre, 1.0), None),
                };
                let mut read = gate.dot(&block.gate);
                if let (Some(up_gradient), Some((_, up_map))) = (&up_gradient, &block.up) {
                    read = read + up_gradient.dot(up_map);
                }
                visit(Visit { cut: Cut::Mlp(b), attribution: value * &d, reads: Reads::Mlp { gate: &gate, up: up_gradient.as_ref() }, after: &g });
                let (gain, inv) = (&self.sites[2 * l + 1].gain, &pass.inverse[2 * l + 1]);
                g = g + read * &gain.view().insert_axis(Axis(0)) * &inv.view().insert_axis(Axis(1));
            }
            let members = &self.layer_heads[l];
            if members.is_empty() {
                continue;
            }
            let mut attribution = Array2::<f64>::zeros((pass.rows, members.len()));
            let mut delta = Array2::<f64>::zeros(g.dim());
            let mut reads = HeadReads { value: Vec::new(), query: Vec::new(), key: Vec::new() };
            for (c, &h) in members.iter().enumerate() {
                let block = &self.heads[h];
                let [q, k, z, v, query_projection, key_projection] = &pass.head[h];
                let gz = g.dot(&block.output);
                let value = match baseline {
                    Some(base) => z - &base.head[h][2],
                    None => z.clone(),
                };
                attribution.column_mut(c).assign(&(value * &gz).sum_axis(Axis(1)));
                let mut gv = Array2::<f64>::zeros(gz.dim());
                let (mut gq, mut gk) = (Array2::<f64>::zeros(q.dim()), Array2::<f64>::zeros(k.dim()));
                for (s, weights) in pass.weights[h].iter().enumerate() {
                    let span = s * length..(s + 1) * length;
                    let gzs = gz.slice(s![span.clone(), ..]);
                    gv.slice_mut(s![span.clone(), ..]).assign(&weights.t().dot(&gzs));
                    if !routes {
                        continue;
                    }
                    // ∂m/∂S_τσ = A_τσ (∂m/∂z_τ · (v_σ − z_τ)), then half of `c q·k` to each factor.
                    let own = (&gzs * &z.slice(s![span.clone(), ..])).sum_axis(Axis(1));
                    let mut scores = gzs.dot(&v.slice(s![span.clone(), ..]).t());
                    ndarray::Zip::indexed(&mut scores).and(weights).for_each(|(t, _), d, w| *d = w * (*d - own[t]));
                    let (qs, ks) = (q.slice(s![span.clone(), ..]).to_owned(), k.slice(s![span.clone(), ..]).to_owned());
                    let (qs, ks) = (rotate(&qs, block.rotary, &positions, false), rotate(&ks, block.rotary, &positions, false));
                    let half = 0.5 * block.scale;
                    let (dq, dk) = (scores.dot(&*ks) * half, scores.t().dot(&*qs) * half);
                    let (dq, dk) = (rotate(&dq, block.rotary, &positions, true).into_owned(), rotate(&dk, block.rotary, &positions, true).into_owned());
                    let unnormed = |grad: Array2<f64>, read: &Read, projection: ndarray::ArrayView2<f64>| match &read.norm {
                        Some((gain, epsilon)) => {
                            let mut out = grad * &gain.view().insert_axis(Axis(0));
                            out.outer_iter_mut().zip(projection.outer_iter()).for_each(|(mut row, p)| row *= rms_scale(p, *epsilon));
                            out
                        }
                        None => grad,
                    };
                    gq.slice_mut(s![span.clone(), ..]).assign(&unnormed(dq, &block.query, query_projection.slice(s![span.clone(), ..])));
                    gk.slice_mut(s![span.clone(), ..]).assign(&unnormed(dk, &block.key, key_projection.slice(s![span, ..])));
                }
                delta = delta + gv.dot(&block.value);
                reads.value.push(gv);
                reads.query.push(gq);
                reads.key.push(gk);
            }
            visit(Visit { cut: Cut::Heads(l, members), attribution, reads: Reads::Heads(&reads), after: &g });
            let (gain, inv) = (&self.sites[2 * l].gain, &pass.inverse[2 * l]);
            g = g + delta * &gain.view().insert_axis(Axis(0)) * &inv.view().insert_axis(Axis(1));
        }
        g
    }

    /// The first column of each cut's functions in [`Library::functions`]'s order.
    fn columns(&self) -> (Vec<usize>, Vec<usize>) {
        let (mut heads, mut mlps) = (vec![0; self.layer_heads.len()], vec![0; self.mlps.len()]);
        let mut next = 0;
        for (l, members) in self.layer_heads.iter().enumerate() {
            heads[l] = next;
            next += members.len();
            for (b, block) in self.mlps.iter().enumerate().filter(|(_, b)| b.layer == l) {
                mlps[b] = next;
                next += block.gate.nrows();
            }
        }
        (heads, mlps)
    }

    /// RelP attributions of `prompt`'s metric to every function at every position.
    pub fn attributions(&self, prompt: &Prompt) -> Result<PromptAttribution, String> {
        let rows = prompt.tokens.len();
        if rows == 0 || prompt.baseline.as_ref().is_some_and(|b| b.len() != rows) {
            return Err("a prompt must be nonempty and its baseline of the same length".into());
        }
        let pass = self.pass(&[&prompt.tokens])?;
        let baseline = prompt.baseline.as_ref().map(|b| self.pass(&[b])).transpose()?;
        let (predicted, seed, metric) = match &prompt.metric {
            Metric::Predicted => self.predicted(&pass)?,
            Metric::Difference { position, target, foil } => {
                let (t, f, p) = (*target as usize, *foil as usize, *position);
                if p >= rows || t >= self.unembedding.nrows() || f >= self.unembedding.nrows() {
                    return Err("a difference metric outside the prompt or the vocabulary".into());
                }
                let mut seed = Array2::<f64>::zeros(pass.last.dim());
                let row = (&self.unembedding.row(t) - &self.unembedding.row(f)) * pass.inverse_final[p];
                let mut metric = vec![0.0; rows];
                metric[p] = row.dot(&pass.last.row(p));
                seed.row_mut(p).assign(&row);
                (self.predicted(&pass)?.0, seed, metric)
            }
        };
        let (head_columns, mlp_columns) = self.columns();
        let mut attributions = Array2::<f64>::zeros((rows, self.functions().len()));
        let empty = || Array2::<f64>::zeros((0, 0));
        let mut gradients: Vec<LayerGradients> = (0..self.layer_heads.len())
            .map(|_| LayerGradients { attention_output: empty(), mlp_output: empty(), values: Vec::new(), gate: empty(), up: None })
            .collect();
        self.relp(&pass, seed, baseline.as_ref(), false, &mut |visit| {
            let start = match visit.cut {
                Cut::Mlp(b) => mlp_columns[b],
                Cut::Heads(l, _) => head_columns[l],
            };
            attributions.slice_mut(s![.., start..start + visit.attribution.ncols()]).assign(&visit.attribution);
            if !prompt.gradients {
                return;
            }
            if let (Cut::Mlp(b), Reads::Mlp { gate, up }) = (&visit.cut, &visit.reads) {
                let layer = &mut gradients[self.mlps[*b].layer];
                layer.mlp_output = visit.after.clone();
                layer.gate = (*gate).clone();
                layer.up = up.cloned();
            }
            if let (Cut::Heads(l, _), Reads::Heads(reads)) = (&visit.cut, &visit.reads) {
                gradients[*l].attention_output = visit.after.clone();
                gradients[*l].values = reads.value.clone();
            }
        });
        let mut outputs = Array2::<f64>::zeros(attributions.dim());
        for (b, block) in self.mlps.iter().enumerate() {
            let norms = block.out.map_axis(Axis(0), |u| u.dot(&u).sqrt());
            let start = mlp_columns[b];
            outputs.slice_mut(s![.., start..start + norms.len()]).assign(&(pass.mlp[b].0.mapv(f64::abs) * &norms.view().insert_axis(Axis(0))));
        }
        for (l, members) in self.layer_heads.iter().enumerate() {
            for (c, &h) in members.iter().enumerate() {
                let z = &pass.head[h][2];
                let gram = self.heads[h].output.t().dot(&self.heads[h].output);
                outputs.column_mut(head_columns[l] + c).assign(&(z.dot(&gram) * z).sum_axis(Axis(1)).mapv(|v| v.max(0.0).sqrt()));
            }
        }
        let gradients = prompt.gradients.then_some(gradients);
        Ok(PromptAttribution { metric, predicted: predicted.into_iter().map(|t| t as u32).collect(), attributions, outputs, gradients })
    }

    /// RelP's completeness on `prompt` (no baseline): at every cut, its functions' attributions,
    /// the skip connection's `Σ ∂m/∂x · x` and the biases after it against `m` (summed over
    /// positions), and finally the embedding stream's, named `embedding`. In the linearized network
    /// each total equals `m` up to rounding.
    pub fn completeness(&self, prompt: &Prompt) -> Result<Vec<CutCheck>, String> {
        let pass = self.pass(&[&prompt.tokens])?;
        let (seed, metric) = match &prompt.metric {
            Metric::Predicted => {
                let (_, seed, metric) = self.predicted(&pass)?;
                (seed, metric.iter().sum::<f64>())
            }
            Metric::Difference { position, target, foil } => {
                let (t, f, p) = (*target as usize, *foil as usize, *position);
                if p >= pass.rows || t >= self.unembedding.nrows() || f >= self.unembedding.nrows() {
                    return Err("a difference metric outside the prompt or the vocabulary".into());
                }
                let mut seed = Array2::<f64>::zeros(pass.last.dim());
                seed.row_mut(p).assign(&((&self.unembedding.row(t) - &self.unembedding.row(f)) * pass.inverse_final[p]));
                let m = seed.row(p).dot(&pass.last.row(p));
                (seed, m)
            }
        };
        let (mut checks, mut biases) = (Vec::new(), 0.0);
        let embedding = self.relp(&pass, seed, None, false, &mut |visit| {
            let (name, site) = match visit.cut {
                Cut::Mlp(b) => (format!("L{}.mlp", self.mlps[b].layer), 2 * self.mlps[b].layer + 1),
                Cut::Heads(l, _) => (format!("L{l}.heads"), 2 * l),
            };
            checks.push(CutCheck { name, functions: visit.attribution.sum(), stream: (visit.after * &pass.streams[site]).sum(), biases, metric });
            // A function's pre-activation bias c enters h through the frozen factor as the
            // gradient with respect to the pre-activation times c.
            if let (Cut::Mlp(b), Reads::Mlp { gate, up }) = (&visit.cut, &visit.reads) {
                let block = &self.mlps[*b];
                biases += gate.dot(&block.bias).sum() + up.map_or(0.0, |u| u.dot(&block.up_bias).sum());
            }
        });
        checks.push(CutCheck { name: "embedding".into(), functions: 0.0, stream: (&embedding * &pass.streams[0]).sum(), biases, metric });
        Ok(checks)
    }
}

// ------------------------------------------------------------------------------ the read-out

/// The read-out (module note) of `artifact`, a library explanation of the split native program
/// `native` (`run_check::split_sites`) with its `layers`, on the held-out `sequences` (of equal
/// length). The model runs on `model`; vocabulary-wide searches and wiring products on `wide`.
pub fn read_out(model: &Device, wide: &Device, native: &OperatorProgram, layers: &[LayerNodes], artifact: &Artifact, sequences: &[Vec<u32>], settings: &Settings) -> Result<Readout, String> {
    if settings.batch_sequences == 0 || sequences.is_empty() || settings.thresholds.iter().any(|t| !(t.is_finite() && *t > 0.0)) {
        return Err("invalid read-out settings or no held-out sequences".into());
    }
    let length = sequences[0].len();
    if length == 0 || sequences.iter().any(|s| s.len() != length) {
        return Err("held-out sequences must be nonempty and of equal length".into());
    }
    let library = Library::new(model, wide, native, layers, artifact, settings.numeric_bytes, settings.tile_rows)?;
    let Library { sites, mlps, heads, unembedding, unembedding_mean, unembedding_table, .. } = &library;
    let feature = native.nodes.iter().position(|n| matches!(n, Node::Feature { .. })).ok_or("no token feature")?;
    let embedding = native
        .nodes
        .iter()
        .find_map(|n| match n {
            Node::Affine { terms, bias: None } if terms.len() == 1 && terms[0].0 == feature => Some(native.operators[terms[0].1].matrix_cow()),
            _ => None,
        })
        .ok_or("no token embedding")?;
    let embedding = if embedding.nrows() == unembedding.ncols() { embedding.t().to_owned() } else { embedding.into_owned() };
    let (vocabulary, width) = embedding.dim();
    // Per read epsilon, the embedding's rows through that read's norm (on the host and on `wide`).
    let mut epsilons: Vec<f64> = sites.iter().map(|s| s.epsilon).collect();
    epsilons.sort_by(f64::total_cmp);
    epsilons.dedup();
    let mut embedding = Some(embedding);
    let mut embedding_tables: BTreeMap<u64, (Array2<f64>, Tensor)> = BTreeMap::new();
    for (e, epsilon) in epsilons.iter().enumerate() {
        let raw = if e + 1 == epsilons.len() { embedding.take() } else { embedding.clone() }.ok_or("no embedding")?;
        let table = normed_rows(raw, *epsilon);
        let resident = wide.upload(table.view()).map_err(error)?;
        embedding_tables.insert(epsilon.to_bits(), (table, resident));
    }

    let k = settings.contexts;
    let thresholds = &settings.thresholds;
    let mut mlp_stats: Vec<Vec<MlpStat>> = mlps.iter().map(|b| vec![MlpStat::default(); b.gate.nrows()]).collect();
    let mut head_stats: Vec<HeadStat> = heads
        .iter()
        .map(|_| HeadStat {
            output: 0.0,
            top: Best::default(),
            offsets: Vec::new(),
            sources: HashMap::new(),
            attended: Vec::new(),
            relp: Relp::default(),
            induction: 0.0,
            duplicate: 0.0,
        })
        .collect();
    let grams: Vec<Array2<f64>> = heads.iter().map(|b| b.output.t().dot(&b.output)).collect();
    let out_norms: Vec<Array1<f64>> = mlps.iter().map(|b| b.out.map_axis(Axis(0), |u| u.dot(&u).sqrt())).collect();
    let mut predicted_all: Vec<u32> = Vec::with_capacity(sequences.len() * length);
    let mut logit_sum = 0.0;
    let (mut rng, mut samples) = (StdRng::seed_from_u64(0), 0.0);
    // Per cut, the summed participation number and, per threshold, the summed count of functions
    // with |A_i(t)| > τ |m(t)|.
    let cuts = 2 * layers.len();
    let mut cut_participation = vec![0.0; cuts];
    let mut cut_counts = vec![vec![0.0; thresholds.len()]; cuts];
    let batches: Vec<(usize, &[Vec<u32>])> = sequences.chunks(settings.batch_sequences).enumerate().map(|(c, b)| (c * settings.batch_sequences * length, b)).collect();
    for (first_row, batch) in &batches {
        let first_row = *first_row;
        let pass = library.pass(&batch.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
        let rows = pass.rows;
        for (b, (activations, pre, _)) in pass.mlp.iter().enumerate() {
            let norms = &out_norms[b];
            mlp_stats[b].par_iter_mut().enumerate().for_each(|(i, stat)| {
                for r in 0..rows {
                    let a = activations[[r, i]];
                    if pre[[r, i]] > 0.0 {
                        stat.positive += 1.0;
                    }
                    stat.abs += a.abs();
                    stat.top.offer(k, a.abs() * norms[i], first_row + r);
                }
            });
        }
        head_stats.par_iter_mut().zip(pass.head.par_iter()).zip(grams.par_iter()).zip(pass.weights.par_iter()).for_each(|(((stat, values), gram), weights)| {
            for (sequence, weights) in batch.iter().zip(weights) {
                for (t, row) in weights.outer_iter().enumerate() {
                    let mut best = (0, f64::NEG_INFINITY);
                    for (u, w) in row.iter().enumerate().filter(|(_, w)| **w > 0.0) {
                        let bin = offset_bin(t.abs_diff(u));
                        if stat.offsets.len() <= bin {
                            stat.offsets.resize(bin + 1, 0.0);
                        }
                        stat.offsets[bin] += w;
                        *stat.sources.entry(sequence[u]).or_insert(0.0) += w;
                        if *w > best.1 {
                            best = (u, *w);
                        }
                    }
                    stat.attended.push(best.0);
                }
            }
            let z = &values[2];
            let norms = (z.dot(gram) * z).sum_axis(Axis(1));
            for r in 0..rows {
                let norm = norms[r].max(0.0).sqrt();
                stat.output += norm;
                stat.top.offer(k, norm, first_row + r);
            }
        });
        let (predicted, seed, metric) = library.predicted(&pass)?;
        // Each drawn target's prediction alone: the reverse pass from its row of the seed over the
        // sequence up to it, each function's attribution summed over the positions it acts at.
        for s in 0..batch.len() {
            let mut positions: Vec<usize> = (0..length).collect();
            for j in 0..settings.targets.min(length) {
                let pick = rng.random_range(j..length);
                positions.swap(j, pick);
            }
            for &t in &positions[..settings.targets.min(length)] {
                let (row, id) = (s * length + t, first_row + s * length + t);
                let prefix = pass.prefix(s, length, t + 1);
                let mut target_seed = Array2::<f64>::zeros((t + 1, seed.ncols()));
                target_seed.row_mut(t).assign(&seed.row(row));
                let m = metric[row];
                samples += 1.0;
                library.relp(&prefix, target_seed, None, false, &mut |visit| {
                    let totals = visit.attribution.sum_axis(Axis(0));
                    let index = match visit.cut {
                        Cut::Mlp(b) => {
                            mlp_stats[b].iter_mut().zip(&totals).for_each(|(stat, a)| stat.relp.add(k, *a, m, thresholds, id));
                            2 * mlps[b].layer + 1
                        }
                        Cut::Heads(l, members) => {
                            for (h, a) in members.iter().zip(&totals) {
                                head_stats[*h].relp.add(k, *a, m, thresholds, id);
                            }
                            2 * l
                        }
                    };
                    let (absolute, square) = totals.iter().fold((0.0, 0.0), |(a, q), v| (a + v.abs(), q + v * v));
                    if square > 0.0 {
                        cut_participation[index] += absolute * absolute / square;
                    }
                    for (count, tau) in cut_counts[index].iter_mut().zip(thresholds) {
                        *count += totals.iter().filter(|v| v.abs() > tau * m.abs()).count() as f64;
                    }
                });
            }
        }
        logit_sum += metric.iter().sum::<f64>();
        predicted_all.extend(predicted.iter().map(|t| *t as u32));
    }
    // Head diagnostics on sequences of random held-out tokens repeated once.
    let mut present: Vec<usize> = sequences.iter().flatten().map(|t| *t as usize).filter(|t| *t < vocabulary).collect();
    present.sort_unstable();
    present.dedup();
    let half = length / 2;
    if half >= 2 && !heads.is_empty() {
        let mut rng = StdRng::seed_from_u64(0);
        let repeated: Vec<Vec<u32>> = (0..sequences.len())
            .map(|_| {
                let first: Vec<u32> = (0..half).map(|_| present[rng.random_range(0..present.len())] as u32).collect();
                let mut sequence: Vec<u32> = first.iter().chain(&first).copied().collect();
                sequence.resize(length, first[0]);
                sequence
            })
            .collect();
        for batch in repeated.chunks(settings.batch_sequences) {
            let pass = library.pass(&batch.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
            for (stat, weights) in head_stats.iter_mut().zip(&pass.weights) {
                for weights in weights {
                    for t in half + 1..2 * half {
                        stat.induction += weights[[t, t + 1 - half]];
                        stat.duplicate += weights[[t, t - half]];
                    }
                }
            }
        }
    }
    let repeats = (sequences.len() * (half.max(1) - 1)) as f64;
    let total = (sequences.len() * length) as f64;
    let context = |value: f64, id: usize, source: Option<usize>| Context { sequence: id / length, position: id % length, value, source };
    // The predicted token a majority of a function's top attribution rows share.
    let majority = |best: &Best| {
        let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
        best.entries.iter().for_each(|(_, id)| *counts.entry(predicted_all[*id]).or_insert(0) += 1);
        counts.into_iter().find(|(_, c)| 2 * c > best.entries.len()).map(|(t, _)| t)
    };
    let fill = |f: &mut Function, relp: &Relp| {
        let rows = |best: &Best, sign: f64| best.entries.iter().map(|(v, id)| context(sign * v, *id, None)).collect::<Vec<_>>();
        f.importance = relp.absolute / samples;
        f.attribution = relp.signed / samples;
        f.important_fraction = relp.important.iter().map(|n| n / samples).collect();
        f.background = f.important_fraction.first().is_some_and(|x| *x == 1.0);
        f.supports = rows(&relp.supports, 1.0);
        f.opposes = rows(&relp.opposes, -1.0);
        f.supports_token = majority(&relp.supports);
        f.opposes_token = majority(&relp.opposes);
    };
    let blank = |name: String, layer: usize, kind: Kind| Function {
        name,
        layer,
        kind,
        positive_gate_fraction: None,
        important_fraction: Vec::new(),
        usage: 0.0,
        contexts: Vec::new(),
        promoted: Vec::new(),
        suppressed: Vec::new(),
        reads: Vec::new(),
        inputs: Vec::new(),
        attention: None,
        importance: 0.0,
        attribution: 0.0,
        supports: Vec::new(),
        opposes: Vec::new(),
        supports_token: None,
        opposes_token: None,
        background: false,
    };

    // The functions, surviving ones only: per layer its heads, then its MLP functions.
    let mut functions: Vec<Function> = Vec::new();
    let mut removed = Vec::new();
    let mut head_index = vec![None; heads.len()];
    let mut mlp_index: Vec<Vec<Option<usize>>> = mlps.iter().map(|b| vec![None; b.gate.nrows()]).collect();
    for l in 0..layers.len() {
        for (h, block) in heads.iter().enumerate().filter(|(_, b)| b.layer == l) {
            let name = format!("L{l}.H{}", block.head);
            if block.value.iter().all(|v| *v == 0.0) || block.output.iter().all(|v| *v == 0.0) {
                removed.push(name);
                continue;
            }
            let stat = &head_stats[h];
            let mass: f64 = stat.offsets.iter().sum();
            let mut sources: Vec<(u32, f64)> = stat.sources.iter().map(|(t, m)| (*t, *m / mass)).collect();
            sources.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            sources.truncate(settings.tokens);
            head_index[h] = Some(functions.len());
            let mut f = blank(name, l, Kind::Head);
            fill(&mut f, &stat.relp);
            f.usage = stat.output / total;
            f.contexts = stat.top.entries.iter().map(|(v, id)| context(*v, *id, Some(stat.attended[*id]))).collect();
            f.attention = Some(Attention {
                offsets: (0..stat.offsets.len()).map(|b| if b == 0 { [0, 0] } else { [1 << (b - 1), (1 << b) - 1] }).collect(),
                offset_mass: stat.offsets.iter().map(|m| m / total).collect(),
                sources: sources.into_iter().map(|(token, score)| TokenScore { token, score }).collect(),
                ov: Vec::new(),
                self_top1: 0.0,
                previous: stat.offsets.get(1).copied().unwrap_or(0.0) / (total - sequences.len() as f64).max(1.0),
                induction: stat.induction / repeats.max(1.0),
                duplicate: stat.duplicate / repeats.max(1.0),
            });
            functions.push(f);
        }
        for (b, block) in mlps.iter().enumerate().filter(|(_, b)| b.layer == l) {
            for i in 0..block.gate.nrows() {
                let name = format!("L{l}.M{i}");
                if out_norms[b][i] == 0.0 || (block.gate.row(i).iter().all(|v| *v == 0.0) && block.bias[i] == 0.0) {
                    removed.push(name);
                    continue;
                }
                let stat = &mlp_stats[b][i];
                mlp_index[b][i] = Some(functions.len());
                let mut f = blank(name, l, Kind::Mlp);
                fill(&mut f, &stat.relp);
                f.positive_gate_fraction = Some(stat.positive / total);
                f.usage = stat.abs / total * out_norms[b][i];
                f.contexts = stat.top.entries.iter().map(|(v, id)| context(*v, *id, None)).collect();
                functions.push(f);
            }
        }
    }

    // What MLP functions write and read, in tokens.
    for (b, block) in mlps.iter().enumerate() {
        let kept: Vec<usize> = (0..block.gate.nrows()).filter(|i| mlp_index[b][*i].is_some()).collect();
        if kept.is_empty() {
            continue;
        }
        let writes = Array2::from_shape_fn((kept.len(), block.out.nrows()), |(r, c)| block.out[[c, kept[r]]]);
        let gain = &sites[2 * block.layer + 1].gain;
        let reads = Array2::from_shape_fn((kept.len(), block.gate.ncols()), |(r, c)| block.gate[[kept[r], c]] * gain[c]);
        let (table, resident) = &embedding_tables[&sites[2 * block.layer + 1].epsilon.to_bits()];
        let promoted = extreme_columns(wide, unembedding_table, &writes, settings.tokens, 1.0, settings.tile_rows)?;
        let suppressed = extreme_columns(wide, unembedding_table, &writes, settings.tokens, -1.0, settings.tile_rows)?;
        let read = extreme_columns(wide, resident, &reads, settings.tokens, 1.0, settings.tile_rows)?;
        for (r, &i) in kept.iter().enumerate() {
            let (u, g) = (writes.row(r), reads.row(r));
            let centre = dot(unembedding_mean.view(), u);
            let write = |t: usize| dot(unembedding.row(t), u) - centre;
            let f = &mut functions[mlp_index[b][i].ok_or("an unindexed function")?];
            f.promoted = scored(&promoted[r], write);
            f.suppressed = scored(&suppressed[r], write);
            f.reads = scored(&read[r], |t| dot(table.row(t), g));
        }
    }

    // Heads: the OV map's largest entries over the held-out text's tokens.
    for (h, block) in heads.iter().enumerate() {
        let Some(index) = head_index[h] else { continue };
        let read_site = &sites[2 * block.layer];
        let (table, _) = &embedding_tables[&read_site.epsilon.to_bits()];
        let value_read = &block.value * &read_site.gain.view().insert_axis(Axis(0));
        let sources = Array2::from_shape_fn((present.len(), width), |(r, c)| table[[present[r], c]]).dot(&value_read.t());
        let output = block.output.t().to_owned();
        let resident = {
            let columns = wide.upload(block.output.view()).map_err(error)?;
            let mut out = wide.zeros(unembedding.nrows(), block.output.ncols()).map_err(error)?;
            wide.gemm(&mut out, 1.0, unembedding_table, Op::N, &columns, Op::N, 0.0, arithmetic(wide)).map_err(error)?;
            out
        };
        let best = extreme_columns(wide, &resident, &sources, 1, 1.0, settings.tile_rows)?;
        let centre = output.dot(unembedding_mean);
        let mut entries: Vec<Entry> = best
            .par_iter()
            .enumerate()
            .map(|(r, o)| {
                let o = o[0];
                let written = output.dot(&unembedding.row(o)) - &centre;
                Entry { source: present[r] as u32, output: o as u32, score: dot(sources.row(r), written.view()) }
            })
            .collect();
        let copies = entries.iter().filter(|e| e.source == e.output).count();
        entries.sort_by(|a, b| b.score.total_cmp(&a.score));
        entries.truncate(settings.tokens);
        if let Some(attention) = functions[index].attention.as_mut() {
            attention.ov = entries;
            attention.self_top1 = copies as f64 / present.len().max(1) as f64;
        }
    }

    // Flow edges among the most important functions: a second pass over the held-out sequences.
    let mut ranked: Vec<usize> = (0..functions.len()).collect();
    ranked.sort_by(|a, b| functions[*b].importance.total_cmp(&functions[*a].importance));
    ranked.truncate(settings.candidates);
    let mut candidate_of: Vec<Option<usize>> = vec![None; functions.len()];
    ranked.iter().enumerate().for_each(|(c, f)| candidate_of[*f] = Some(c));
    // Per MLP its candidate functions and their candidate indices; per head its candidate index.
    let mlp_candidates: Vec<Vec<(usize, usize)>> =
        mlp_index.iter().map(|block| block.iter().enumerate().filter_map(|(i, f)| Some((i, candidate_of[(*f)?]?))).collect()).collect();
    let head_candidate: Vec<Option<usize>> = head_index.iter().map(|f| candidate_of[(*f)?]).collect();
    // Per route (a head reader's value, query and key maps; an MLP reader's gate and up maps), the
    // summed flow from each candidate writer (columns) to each candidate reader (rows).
    let mut flows: [Array2<f64>; 3] = std::array::from_fn(|_| Array2::<f64>::zeros((ranked.len(), ranked.len())));
    for (_, batch) in &batches {
        let pass = library.pass(&batch.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
        let (_, seed, _) = library.predicted(&pass)?;
        library.relp(&pass, seed, None, true, &mut |visit| {
            let (cut, reads) = (visit.cut, &visit.reads);
            // The writers before this read: MLPs of earlier layers, heads of earlier layers (and of
            // this layer, for an MLP read).
            let (layer, site_index) = match cut {
                Cut::Mlp(b) => (mlps[b].layer, 2 * mlps[b].layer + 1),
                Cut::Heads(l, _) => (l, 2 * l),
            };
            let (gain, inverse) = (&sites[site_index].gain, &pass.inverse[site_index]);
            let mlp_writers: Vec<usize> = (0..mlps.len()).filter(|b| mlps[*b].layer < layer && !mlp_candidates[*b].is_empty()).collect();
            let head_writers: Vec<usize> =
                (0..heads.len()).filter(|h| head_candidate[*h].is_some() && (heads[*h].layer < layer || heads[*h].layer == layer && site_index % 2 == 1)).collect();
            if let (Cut::Mlp(b), Reads::Mlp { gate, up }) = (&cut, reads) {
                let b = *b;
                let readers = &mlp_candidates[b];
                if !readers.is_empty() {
                    let block = &mlps[b];
                    let mut parts = vec![(*gate, &block.gate)];
                    if let (Some(up), Some((_, map))) = (up, &block.up) {
                        parts.push((*up, map));
                    }
                    for (route, (gradient, map)) in parts.into_iter().enumerate() {
                        let flows = &mut flows[route];
                        // Per token the reader's gradient through its frozen norm, and its direction.
                        let weights = Array2::from_shape_fn((pass.rows, readers.len()), |(r, c)| gradient[[r, readers[c].0]] * inverse[r]);
                        let directions = Array2::from_shape_fn((readers.len(), width), |(c, d)| map[[readers[c].0, d]] * gain[d]);
                        for &w in &mlp_writers {
                            let writers = &mlp_candidates[w];
                            let values = Array2::from_shape_fn((pass.rows, writers.len()), |(r, c)| pass.mlp[w].0[[r, writers[c].0]]);
                            let outputs = Array2::from_shape_fn((width, writers.len()), |(d, c)| mlps[w].out[[d, writers[c].0]]);
                            let flow = weights.t().dot(&values) * directions.dot(&outputs);
                            for (r, reader) in readers.iter().enumerate() {
                                for (c, writer) in writers.iter().enumerate() {
                                    flows[[reader.1, writer.1]] += flow[[r, c]];
                                }
                            }
                        }
                        for &h in &head_writers {
                            let (z, Some(writer)) = (&pass.head[h][2], head_candidate[h]) else { continue };
                            let flow = (weights.t().dot(z) * directions.dot(&heads[h].output)).sum_axis(Axis(1));
                            for (r, reader) in readers.iter().enumerate() {
                                flows[[reader.1, writer]] += flow[r];
                            }
                        }
                    }
                }
            }
            if let (Cut::Heads(_, members), Reads::Heads(head_reads)) = (&cut, reads) {
                    for (c, &h) in members.iter().enumerate() {
                        let Some(reader) = head_candidate[h] else { continue };
                        let block = &heads[h];
                        let routes = [(&head_reads.value[c], &block.value), (&head_reads.query[c], &block.query.map), (&head_reads.key[c], &block.key.map)];
                        for (route, (gradient, map)) in routes.into_iter().enumerate() {
                            let flows = &mut flows[route];
                            let weights = gradient * &inverse.view().insert_axis(Axis(1));
                            let read = map * &gain.view().insert_axis(Axis(0));
                            for &w in &mlp_writers {
                                let writers = &mlp_candidates[w];
                                let values = Array2::from_shape_fn((pass.rows, writers.len()), |(r, c)| pass.mlp[w].0[[r, writers[c].0]]);
                                let outputs = Array2::from_shape_fn((writers.len(), width), |(c, d)| mlps[w].out[[d, writers[c].0]]);
                                let flow = (values.t().dot(&weights) * outputs.dot(&read.t())).sum_axis(Axis(1));
                                for (c, writer) in writers.iter().enumerate() {
                                    flows[[reader, writer.1]] += flow[c];
                                }
                            }
                            for &s in &head_writers {
                                let Some(writer) = head_candidate[s] else { continue };
                                flows[[reader, writer]] += (pass.head[s][2].t().dot(&weights) * heads[s].output.t().dot(&read.t())).sum();
                            }
                        }
                    }
            }
        });
    }
    let summed = &flows[0] + &flows[1] + &flows[2];
    for (c, &f) in ranked.iter().enumerate() {
        let mut best = Best::default();
        for (w, flow) in summed.row(c).iter().enumerate() {
            if *flow != 0.0 {
                best.offer(settings.edges, flow.abs(), w);
            }
        }
        functions[f].inputs =
            best.entries.iter().map(|(_, w)| Edge { from: ranked[*w], flow: summed[[c, *w]] / total, routes: std::array::from_fn(|r| flows[r][[c, *w]] / total) }).collect();
    }
    // The core: the most important candidates, the background apart.
    let core: Vec<usize> = ranked.iter().copied().filter(|f| !functions[*f].background).take(settings.core).collect();
    let wiring = core.iter().map(|i| core.iter().map(|j| candidate_of[*i].zip(candidate_of[*j]).map_or(0.0, |(a, b)| summed[[a, b]] / total)).collect()).collect();
    Ok(Readout {
        held_out_tokens: sequences.len() * length,
        functions,
        removed,
        core,
        wiring,
        predicted: predicted_all,
        mean_logit: logit_sum / total,
        targets: samples as usize,
        important: thresholds.iter().enumerate().map(|(j, tau)| [*tau, cut_counts.iter().map(|c| c[j]).sum::<f64>() / samples]).collect(),
        participation: cut_participation.iter().sum::<f64>() / samples,
        cuts: (0..cuts)
            .map(|c| CutSummary {
                name: format!("L{}.{}", c / 2, if c % 2 == 0 { "heads" } else { "mlp" }),
                participation: cut_participation[c] / samples,
                counts: cut_counts[c].iter().map(|n| n / samples).collect(),
            })
            .collect(),
    })
}

// ------------------------------------------------------------------------------ token text

/// A byte-level BPE vocabulary (`tokenizer.json`): each token's bytes.
pub struct Vocabulary {
    pieces: Vec<Vec<u8>>,
}

impl Vocabulary {
    pub fn from_tokenizer(path: &Path) -> Result<Self, String> {
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path).map_err(error)?).map_err(error)?;
        // The byte-level alphabet: printable bytes stand for themselves, the rest for 256 + n.
        let mut printable: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
        let mut chars = printable.clone();
        let mut extra = 0;
        for byte in 0..256u32 {
            if !printable.contains(&byte) {
                printable.push(byte);
                chars.push(256 + extra);
                extra += 1;
            }
        }
        let byte_of: HashMap<char, u8> = chars.iter().zip(&printable).filter_map(|(c, b)| Some((char::from_u32(*c)?, *b as u8))).collect();
        let mut pieces: Vec<Vec<u8>> = Vec::new();
        let mut put = |id: usize, bytes: Vec<u8>| {
            if pieces.len() <= id {
                pieces.resize(id + 1, Vec::new());
            }
            pieces[id] = bytes;
        };
        let vocab = value["model"]["vocab"].as_object().ok_or("tokenizer.json has no model.vocab")?;
        for (piece, id) in vocab {
            let id = id.as_u64().ok_or("a vocabulary id")? as usize;
            put(id, piece.chars().map(|c| byte_of.get(&c).copied().ok_or_else(|| format!("{piece:?} is not byte-level"))).collect::<Result<_, _>>()?);
        }
        for added in value["added_tokens"].as_array().into_iter().flatten() {
            let id = added["id"].as_u64().ok_or("an added token id")? as usize;
            put(id, added["content"].as_str().ok_or("an added token's content")?.as_bytes().to_vec());
        }
        Ok(Self { pieces })
    }

    /// The text of `tokens`, invalid UTF-8 replaced.
    #[must_use]
    pub fn text(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| self.pieces.get(*t as usize).cloned().unwrap_or_else(|| format!("<{t}>").into_bytes())).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

// ------------------------------------------------------------------------------ where the bits go

/// What one function's parameters cost in the code length `F`: its prior groups' `KL(q_G ‖ p_G)`
/// and, with each group variance's `½ log2 |G|`, its total, in bits, and the total per part (a
/// head's query-key planes and value coordinates; an MLP function's gate, up direction and
/// output).
#[derive(Clone, Debug, Serialize)]
pub struct FunctionCost {
    pub name: String,
    pub layer: usize,
    pub kind: Kind,
    /// Whether any of its groups is still in the explanation.
    pub active: bool,
    pub divergence_bits: f64,
    pub bits: f64,
    pub parts: BTreeMap<String, f64>,
}

/// The posterior of a `library_mdl` fit checkpoint of `explanation`: the checkpoint is the fit's
/// progress as JSON after its length, then per trainable operator its `μ`, `ln σ` and four
/// optimizer arrays as little-endian float64.
pub fn checkpoint_posterior(explanation: &Explanation, path: &Path) -> Result<Posterior, String> {
    let bytes = std::fs::read(path).map_err(error)?;
    let length = u64::from_le_bytes(bytes.get(..8).ok_or("a truncated checkpoint")?.try_into().map_err(error)?) as usize;
    let header: serde_json::Value = serde_json::from_slice(bytes.get(8..8 + length).ok_or("a truncated checkpoint")?).map_err(error)?;
    let tokens = header["tokens"].as_u64().ok_or("a checkpoint without its tokens")? as usize;
    let mut posterior = Posterior::new(explanation, tokens)?;
    let shapes: Vec<(usize, usize)> = serde_json::from_value(header["shapes"].clone()).map_err(error)?;
    let active: Vec<bool> = serde_json::from_value(header["active"].clone()).map_err(error)?;
    if shapes != posterior.mean.iter().map(Array2::dim).collect::<Vec<_>>() || active.len() != posterior.active.len() {
        return Err(format!("{}: a checkpoint of another explanation", path.display()));
    }
    let count: usize = shapes.iter().map(|(r, c)| r * c * 6).sum();
    if bytes.len() != 8 + length + count * 8 {
        return Err(format!("{}: a checkpoint of the wrong size", path.display()));
    }
    let mut at = 8 + length;
    let mut next = |dim: (usize, usize)| {
        let n = dim.0 * dim.1;
        let values: Vec<f64> = bytes[at..at + 8 * n].chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect();
        at += 8 * n;
        Array2::from_shape_vec(dim, values).map_err(error)
    };
    for (i, dim) in shapes.iter().enumerate() {
        posterior.mean[i] = next(*dim)?;
        posterior.log_sd[i] = next(*dim)?;
        for _ in 0..4 {
            next(*dim)?;
        }
    }
    posterior.active = active;
    Ok(posterior)
}

/// Per layer and kind of function (`L{l}.heads`, `L{l}.mlp`), what `posterior`'s groups of it cost
/// in bits, `KL(q‖p)` and total, each group counted once (query heads sharing a key and value read
/// one group).
#[must_use]
pub fn layer_costs(explanation: &Explanation, posterior: &Posterior) -> BTreeMap<String, [f64; 2]> {
    const BITS: f64 = std::f64::consts::LOG2_E;
    let (divergences, costs) = (posterior.divergences(), posterior.costs());
    let mut out = BTreeMap::new();
    for (l, layer) in explanation.layers.iter().enumerate() {
        let heads: std::collections::BTreeSet<usize> = layer.heads.iter().flat_map(|(planes, values)| planes.iter().chain(values).copied()).collect();
        let functions: std::collections::BTreeSet<usize> = layer.functions.iter().flatten().copied().collect();
        for (kind, groups) in [("heads", heads), ("mlp", functions)] {
            let total = |of: &[f64]| groups.iter().map(|g| of[*g] * BITS).sum::<f64>();
            out.insert(format!("L{l}.{kind}"), [total(&divergences), total(&costs)]);
        }
    }
    out
}

/// Per function of `explanation` (per layer its heads, then its MLP functions), what `posterior`'s
/// groups of it cost (`library_mdl::Posterior::divergences` and `costs`). A key and value group
/// shared by several query heads is listed under each; totals over heads come from
/// [`layer_costs`].
#[must_use]
pub fn function_costs(explanation: &Explanation, posterior: &Posterior) -> Vec<FunctionCost> {
    const BITS: f64 = std::f64::consts::LOG2_E;
    let (divergences, costs) = (posterior.divergences(), posterior.costs());
    let function = |name: String, layer: usize, kind: Kind, parts: Vec<(String, &[usize])>| {
        let groups: Vec<usize> = parts.iter().flat_map(|(_, g)| g.iter().copied()).collect();
        FunctionCost {
            name,
            layer,
            kind,
            active: groups.iter().any(|g| posterior.active[*g]),
            divergence_bits: groups.iter().map(|g| divergences[*g] * BITS).sum(),
            bits: groups.iter().map(|g| costs[*g] * BITS).sum(),
            parts: parts.into_iter().map(|(part, g)| (part, g.iter().map(|g| costs[*g] * BITS).sum())).collect(),
        }
    };
    let mut out = Vec::new();
    for (l, layer) in explanation.layers.iter().enumerate() {
        for (h, (planes, values)) in layer.heads.iter().enumerate() {
            out.push(function(format!("L{l}.H{h}"), l, Kind::Head, vec![("query-key planes".into(), planes), ("values".into(), values)]));
        }
        for (i, groups) in layer.functions.iter().enumerate() {
            let parts = groups.iter().map(|g| (explanation.groups[*g].name.rsplit('.').next().unwrap_or("").to_string(), std::slice::from_ref(g))).collect();
            out.push(function(format!("L{l}.M{i}"), l, Kind::Mlp, parts));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{Library, Metric, Prompt};
    use crate::{
        import::import_language_model,
        library_mdl,
        operator_program::SlotValues,
        run_check::{layer_nodes, split_sites},
        test_support::{tiny_export, tiny_qwen3_export},
    };
    use gam_gpu::tensor::Device;
    use std::path::PathBuf;

    /// The largest relative gap `|Σ_i A_i + Σ ∂m/∂x · x − m| / |m|` over every cut (and the
    /// embedding stream) of the tiny decoder exported to `dir`, on its six sequences, for the
    /// predicted tokens' logits and for a logit difference at one position.
    fn largest_gap(dir: PathBuf) -> f64 {
        let imported = import_language_model(&dir, 6, 12).expect("tiny export");
        std::fs::remove_dir_all(dir).expect("remove the tiny export");
        let native = split_sites(&imported.program).expect("split sites");
        let layers = layer_nodes(&native, 2).expect("layer nodes");
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let artifact = library_mdl::explanation(&native, &layers).expect("library").artifact;
        let host = Device::host();
        let library = Library::new(&host, &host, &native, &layers, &artifact, 1 << 30, 64).expect("library on the host");
        let mut largest: f64 = 0.0;
        for (i, sequence) in tokens.chunks(12).enumerate() {
            for metric in [Metric::Predicted, Metric::Difference { position: 4 + i, target: 1, foil: 2 }] {
                let prompt = Prompt { tokens: sequence.to_vec(), baseline: None, metric, gradients: false };
                for check in library.completeness(&prompt).expect("completeness") {
                    largest = largest.max(check.gap());
                }
            }
        }
        largest
    }

    /// RelP's attributions with the skip connection account for the metric at every cut of a GELU
    /// decoder (frozen norm denominators, frozen gate factors, frozen attention patterns).
    #[test]
    fn relp_is_complete_at_every_cut_of_a_gelu_decoder() {
        let gap = largest_gap(tiny_export("readout_complete_gelu", 2));
        eprintln!("readout_complete_gelu: largest relative gap {gap:e}");
        assert!(gap < 1e-10, "largest relative gap {gap}");
    }

    /// ... and of a SiLU-gated decoder with head norms and a shared key and value: half of the
    /// gradient through each factor of gate × up.
    #[test]
    fn relp_is_complete_at_every_cut_of_a_gated_decoder() {
        let gap = largest_gap(tiny_qwen3_export("readout_complete_gated", 2));
        eprintln!("readout_complete_gated: largest relative gap {gap:e}");
        assert!(gap < 1e-10, "largest relative gap {gap}");
    }
}

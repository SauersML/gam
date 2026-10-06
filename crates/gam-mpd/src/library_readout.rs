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
//! * Removal effect: the change in `KL(M ‖ P)` of a token's next-token distributions, in bits,
//!   when one function's write alone (`h_i u_i`, a head's `W_O,h z_h`) is taken out of the
//!   explanation `P` at every position and every later layer rerun ([`Library::removal_effects`]).
//!   Importance is its mean over the measured held-out tokens; its quantiles give the spread.
//!   Participation: per token, the effective number of functions its prediction rests on,
//!   `(Σ_i |ΔKL_i|)² / Σ_i ΔKL_i²`.
//! * Positive-gate fraction (MLP functions): the fraction of tokens with a positive gate
//!   pre-activation. It is not activity: GELU and SiLU are nonzero on both sides, and a negative
//!   gate times a large up value is a large output. It is not comparable with VPD's L0.
//! * Usage: the mean output norm `|h_i| ‖u_i‖` (heads `‖W_O,h z_h‖`) over held-out tokens.
//!   Contexts: the held-out tokens with the largest output norm (heads: with the position the
//!   head attends to most there).
//! * Supports and opposes: the measured tokens where the function's removal raises the divergence
//!   most and lowers it most, and the predicted token a majority of each set shares.
//! * Writes: the output direction through the final norm's gain and the unembedding,
//!   `W_U (γ_f ⊙ u_i)`, centred over the vocabulary; the most promoted and suppressed tokens (a
//!   direct path only). Reads: the gate direction `γ ⊙ g_i` against each token's embedding at that
//!   read, `(γ ⊙ g_i)·e_t / √(mean(e_t²) + ε)`.
//! * Edges: a writer's exact contribution to a reader's reads ([`Library::contributions`], the
//!   writer's write through the reader's norm at the stream's actual RMS and each read map: an
//!   MLP reader's gate and up, a head reader's query, key and value) ranks the candidates among
//!   the `candidates` most important functions; each of the `edges` strongest is measured by path
//!   patching ([`Library::path_patch`]: the write taken out of that reader's reads alone, the
//!   reader recomputed and everything after it rerun), as the reader's output change and
//!   `KL(P ‖ P′)` in bits. The `core` most important functions' wiring among themselves is
//!   measured whole.
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
    device_program::{DeviceProgram, DeviceTrace},
    library_mdl::{Explanation, Posterior, sequence_family},
    operator_program::{Law, Node, OperatorProgram, Rotary, Rule, rms_scale},
    resident_causal_fit::fixed_head_target::{Head, ResidentHead, Target, Teacher},
    run_check::LayerNodes,
    tiled_attention::{probabilities, rotate},
};
use gam_gpu::tensor::{Arithmetic, Device, Op, Storage, Tensor};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Axis, CowArray, Data, Ix1, Ix2, s};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
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
    /// Per function: contexts, tokens of each token list, and inputs measured.
    pub contexts: usize,
    pub tokens: usize,
    pub edges: usize,
    /// The held-out sequences (the first of them) on which removals and path patches are
    /// measured, and the functions removed at a time.
    pub measured_sequences: usize,
    pub removal_batch: usize,
    /// The most important functions among which edge candidates are ranked by exact
    /// contribution, and those whose wiring among themselves is measured whole.
    pub candidates: usize,
    pub core: usize,
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

/// An input of a function: the writer (an index into `Readout::functions`), its exact
/// contribution to the reader's reads per route ([`Library::contributions`]), and the measured
/// effect of path patching it out ([`Library::path_patch`]): the reader's mean output change and
/// `KL(P ‖ P′)` per token in bits.
#[derive(Clone, Debug, Serialize)]
pub struct Edge {
    pub from: usize,
    pub contribution: [f64; 3],
    pub output: f64,
    pub prediction_bits: f64,
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
    pub usage: f64,
    pub contexts: Vec<Context>,
    pub promoted: Vec<TokenScore>,
    pub suppressed: Vec<TokenScore>,
    pub reads: Vec<TokenScore>,
    pub inputs: Vec<Edge>,
    pub attention: Option<Attention>,
    /// The measured removal effect ([`Library::removal_effects`]): the mean change of
    /// `KL(M ‖ P)` per token in bits, and its quantiles 0.5, 0.9, 0.99 and 1 over the tokens.
    pub importance: f64,
    pub effect_quantiles: [f64; 4],
    /// The tokens where its removal raises the divergence most and where it lowers it most, and
    /// the predicted token a majority of each set shares (none when no token holds a majority).
    pub supports: Vec<Context>,
    pub opposes: Vec<Context>,
    pub supports_token: Option<u32>,
    pub opposes_token: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Readout {
    pub held_out_tokens: usize,
    /// The tokens removals and path patches were measured on.
    pub measured_tokens: usize,
    pub functions: Vec<Function>,
    pub removed: Vec<String>,
    /// Indices into `functions` of the most important functions, descending, and `wiring[i][j]`
    /// the measured `KL(P ‖ P′)` per token in bits of path patching core function `j` out of core
    /// function `i`'s reads (zero when `j` does not precede `i`'s read).
    pub core: Vec<usize>,
    pub wiring: Vec<Vec<f64>>,
    /// Per held-out token, the model's predicted token.
    pub predicted: Vec<u32>,
    /// Per measured token, the effective number of functions its prediction rests on,
    /// `(Σ_i |ΔKL_i|)² / Σ_i ΔKL_i²` over the measured removals: its quantiles 0.1, 0.5, 0.9 and
    /// its mean.
    pub participation: [f64; 4],
}

/// One copy of [`Library::edited`]: a sequence, the functions whose writes it scales with their
/// factors, and the rows read.
#[derive(Clone, Debug)]
pub struct Edit {
    pub sequence: usize,
    pub scale: Vec<(usize, f64)>,
    pub rows: Vec<usize>,
}

/// [`Library::edited`]'s reads: per read row (copies in order, each copy's rows in order) the final
/// stream before the final norm (reads × width) and every function's activity (reads × functions).
pub struct Edited {
    pub last: Array2<f64>,
    pub activity: Option<Array2<f64>>,
}

/// One path patch's per-row results ([`Library::path_patched`]).
pub struct PathPatch {
    /// The change of the reader's write (rows × width).
    pub change: Array2<f64>,
    /// The reader's activity before and after the patch.
    pub before: Array1<f64>,
    pub after: Array1<f64>,
    /// The final streams before the final norm, patched and of the run.
    pub last: Array2<f64>,
    pub base: Array2<f64>,
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
    /// The gate's law.
    law: Law,
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
    let Node::Pointwise { input: gate_pre, laws } = &rule.nodes[gate_active] else { return Err("no gate".into()) };
    let (gate_pre, law) = (*gate_pre, *laws.first().ok_or("a law of no units")?);
    if laws.iter().any(|l| *l != law) {
        return Err(format!("{}: units of different laws", rule.name));
    }
    let (gate, bias) = operator_of(program, rule, gate_pre)?;
    let bias = match bias {
        Some(op) => program.operators[op].matrix().column(0).to_owned(),
        None => Array1::zeros(gate.nrows()),
    };
    let up_bias = match up_bias {
        Some(op) => program.operators[op].matrix().column(0).to_owned(),
        None => Array1::zeros(gate.nrows()),
    };
    Ok(MlpBlock { layer, call, activation, gate_pre, gate, bias, out: program.operators[out].matrix(), up, up_bias, law })
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
}

struct HeadStat {
    output: f64,
    top: Best,
    offsets: Vec<f64>,
    sources: HashMap<u32, f64>,
    /// Per held-out row, the position attended to most.
    attended: Vec<usize>,
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
fn extreme_columns<S: Data<Elem = f64>>(device: &Device, table: &Tensor, rows: &ArrayBase<S, Ix2>, k: usize, sign: f64, tile: usize) -> Result<Vec<Vec<usize>>, String> {
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

/// A library explanation prepared for forward passes: its blocks, the native
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
    /// The input embedding (vocabulary × width), and the head (the unembedding after the final
    /// norm) on the model's device, which scores the explanation's distributions against `M`'s.
    embedding: Array2<f64>,
    resident: ResidentHead,
    program: DeviceProgram,
    observed: Vec<usize>,
    mlp_paths: Vec<usize>,
    head_paths: usize,
}

type Values<'a> = CowArray<'a, f64, Ix2>;

/// One forward pass's observed values on sequences of one length (owned, or a view of another
/// pass's rows).
struct Pass<'a> {
    rows: usize,
    /// Per site, the residual stream it reads and `1/r` per row; the final stream and its `1/r`.
    streams: Vec<Values<'a>>,
    inverse: Vec<CowArray<'a, f64, Ix1>>,
    last: Values<'a>,
    inverse_final: CowArray<'a, f64, Ix1>,
    /// Per MLP: its activations, gate pre-activations and, when gated, up values.
    mlp: Vec<(Values<'a>, Values<'a>, Option<Values<'a>>)>,
    /// Per head: the query and key it scores, its read, its value, and its query and key
    /// projections (before its head norms); per head and sequence, its attention weights.
    head: Vec<[Values<'a>; 6]>,
    weights: Vec<Vec<Values<'a>>>,
}

/// One run of the explanation ([`Library::run`]).
#[derive(Clone, Debug)]
pub struct Run {
    /// Per read site, the residual stream entering it (site `2l`: layer `l`'s attention, `2l + 1`:
    /// its MLP), rows × width.
    pub streams: Vec<Array2<f64>>,
    /// Per head ([`Library::heads`]), its read (rows × head width) and its attention weights per
    /// sequence.
    pub reads: Vec<Array2<f64>>,
    pub weights: Vec<Vec<Array2<f64>>>,
    /// The final stream, before the final norm.
    pub last: Array2<f64>,
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
        let head = Head::of(native)?;
        let resident = ResidentHead::new(model, &head, tile_rows)?;
        let Head { hidden, embedding: mut unembedding, .. } = head;
        let final_site = site(native, hidden)?;
        unembedding.axis_iter_mut(Axis(0)).for_each(|mut row| row *= &final_site.gain);
        let unembedding_mean = unembedding.mean_axis(Axis(0)).ok_or("an empty vocabulary")?;
        let unembedding_table = wide.upload(unembedding.view()).map_err(error)?;
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
            embedding,
            resident,
            program: compiled,
            observed,
            mlp_paths,
            head_paths,
        })
    }

    /// The heads, as (layer, head), in the order [`Library::run`] and [`Library::head_on`] index them.
    #[must_use]
    pub fn heads(&self) -> Vec<(usize, usize)> {
        self.heads.iter().map(|b| (b.layer, b.head)).collect()
    }

    /// The explanation run on `sequences` (of one length), each head in `replace` (an index into
    /// [`Library::heads`]) reading the given values (rows × head width) in place of its own, and
    /// everything after it recomputed.
    pub fn run(&self, sequences: &[Vec<u32>], replace: &BTreeMap<usize, Array2<f64>>) -> Result<Run, String> {
        self.run_with(sequences, replace, &BTreeMap::new())
    }

    /// [`Library::run`] with each MLP in `mlps` (an index into the layers' MLPs, in layer order)
    /// taking the given activations (rows × its functions) as well.
    pub fn run_with(&self, sequences: &[Vec<u32>], heads: &BTreeMap<usize, Array2<f64>>, mlps: &BTreeMap<usize, Array2<f64>>) -> Result<Run, String> {
        let pass = self.pass_with(&sequences.iter().map(Vec::as_slice).collect::<Vec<_>>(), heads, mlps)?;
        Ok(Run {
            streams: pass.streams.into_iter().map(CowArray::into_owned).collect(),
            reads: pass.head.into_iter().map(|[_, _, z, ..]| z.into_owned()).collect(),
            weights: pass.weights.into_iter().map(|w| w.into_iter().map(CowArray::into_owned).collect()).collect(),
            last: pass.last.into_owned(),
        })
    }

    /// Head `h`'s write of its read `z` (rows × head width): `z W_O,hᵀ`.
    #[must_use]
    pub fn write(&self, h: usize, z: &Array2<f64>) -> Array2<f64> {
        z.dot(&self.heads[h].output.t())
    }

    /// Head `h` recomputed with its query, key and value reads taken from the given streams entering
    /// its layer (rows × width, sequences of `length` rows): each normed by the layer's norm (its
    /// own RMS), read by the head's map and head norm, the query and key rotated, the causal
    /// softmax of their scores mixing the values. Its read and its attention weights per sequence.
    #[must_use]
    pub fn head_on(&self, h: usize, query: &Array2<f64>, key: &Array2<f64>, value: &Array2<f64>, length: usize) -> (Array2<f64>, Vec<Array2<f64>>) {
        let block = &self.heads[h];
        let site = &self.sites[2 * block.layer];
        let normed = |x: &Array2<f64>| {
            let mut out = x * &site.gain.view().insert_axis(Axis(0));
            out.outer_iter_mut().zip(x.outer_iter()).for_each(|(mut row, x)| row *= rms_scale(x, site.epsilon));
            out
        };
        let read = |x: &Array2<f64>, read: &Read| {
            let mut out = normed(x).dot(&read.map.t());
            if let Some((gain, epsilon)) = &read.norm {
                let projection = out.clone();
                out = out * &gain.view().insert_axis(Axis(0));
                out.outer_iter_mut().zip(projection.outer_iter()).for_each(|(mut row, p)| row *= rms_scale(p, *epsilon));
            }
            out
        };
        let (q, k, v) = (read(query, &block.query), read(key, &block.key), normed(value).dot(&block.value.t()));
        let positions: Vec<u32> = (0..length as u32).collect();
        let mut z = Array2::<f64>::zeros(v.dim());
        let mut weights = Vec::new();
        for s in 0..q.nrows() / length.max(1) {
            let span = s * length..(s + 1) * length;
            let (qs, ks) = (q.slice(s![span.clone(), ..]).to_owned(), k.slice(s![span.clone(), ..]).to_owned());
            let (qs, ks) = (rotate(&qs, block.rotary, &positions, false), rotate(&ks, block.rotary, &positions, false));
            let w = probabilities(qs.view(), ks.view(), &positions, 0, block.scale, block.causal);
            z.slice_mut(s![span.clone(), ..]).assign(&w.dot(&v.slice(s![span, ..])));
            weights.push(w);
        }
        (z, weights)
    }

    /// The next-token log-probabilities of each row of the final stream `last` (rows × vocabulary),
    /// through the final norm (its own RMS) and the unembedding.
    pub fn log_probabilities(&self, last: &Array2<f64>) -> Result<Array2<f64>, String> {
        let mut logits = last.dot(&self.unembedding.t());
        for (mut row, x) in logits.outer_iter_mut().zip(last.outer_iter()) {
            row *= rms_scale(x, self.final_site.epsilon);
            let values = gam_math::categorical::log_softmax(row.as_slice().ok_or("a contiguous row")?).map_err(error)?;
            row.assign(&Array1::from(values));
        }
        Ok(logits)
    }

    /// A function's place: its layer, and its head (an index into the heads) or its MLP (an index
    /// into the MLPs) and function.
    fn place(&self, function: usize) -> Result<(usize, Result<usize, (usize, usize)>), String> {
        let (head_columns, mlp_columns) = self.columns();
        for (l, members) in self.layer_heads.iter().enumerate() {
            if (head_columns[l]..head_columns[l] + members.len()).contains(&function) {
                return Ok((l, Ok(members[function - head_columns[l]])));
            }
        }
        for (b, block) in self.mlps.iter().enumerate() {
            if (mlp_columns[b]..mlp_columns[b] + block.gate.nrows()).contains(&function) {
                return Ok((block.layer, Err((b, function - mlp_columns[b]))));
            }
        }
        Err(format!("no function {function}"))
    }

    /// The measured effect of one edge, by path patching ([`Library::path_patched`]). Returns the
    /// mean over the rows of `sequences` of the reader's output change (`|Δh| ‖u‖`, a head's
    /// `‖W_O Δz‖`) and of `KL(P ‖ P′)` of the next-token distributions in bits.
    pub fn path_patch(&self, sequences: &[Vec<u32>], writer: usize, reader: usize, route: Option<usize>) -> Result<[f64; 2], String> {
        let patched = self.path_patched(sequences, writer, reader, route)?;
        let rows = patched.change.nrows() as f64;
        let output = patched.change.outer_iter().map(|row| row.dot(&row).sqrt()).sum::<f64>() / rows;
        let (base, edited) = (self.log_probabilities(&patched.base)?, self.log_probabilities(&patched.last)?);
        let kl: f64 = base.outer_iter().zip(edited.outer_iter()).map(|(p, q)| p.iter().zip(q.iter()).map(|(a, b)| a.exp() * (a - b)).sum::<f64>()).sum();
        Ok([output, kl / rows / std::f64::consts::LN_2])
    }

    /// One edge by path patching: writer `writer`'s write taken out of reader `reader`'s reads only
    /// (of its read `route` alone, numbered as in [`Library::contributions`], or of every read), the
    /// reader recomputed from them (an MLP function through its gate and law, a head through its
    /// attention, each with its own norm at the stream's actual RMS) and everything after it run.
    /// Per row of `sequences`: the change of the reader's write, its activity before and after (an
    /// MLP function's `h_i`, a head's `‖W_O,h z_h‖`), and the final streams before the final norm
    /// of the run and of the patched run.
    pub fn path_patched(&self, sequences: &[Vec<u32>], writer: usize, reader: usize, route: Option<usize>) -> Result<PathPatch, String> {
        let refs: Vec<&[u32]> = sequences.iter().map(Vec::as_slice).collect();
        let pass = self.pass(&refs)?;
        let length = sequences.first().map_or(0, Vec::len);
        let (written, write) = self.place(writer)?;
        let (layer, read) = self.place(reader)?;
        let site = if read.is_ok() { 2 * layer } else { 2 * layer + 1 };
        let before = match write {
            Ok(_) => 2 * written + 1 <= site,
            Err(_) => 2 * written + 2 <= site,
        };
        if !before {
            return Err("the writer does not write before the reader reads".into());
        }
        let w = match write {
            Ok(h) => self.write(h, &pass.head[h][2].to_owned()),
            Err((b, i)) => {
                let (h, u) = (pass.mlp[b].0.column(i), self.mlps[b].out.column(i));
                Array2::from_shape_fn((pass.rows, u.len()), |(r, c)| h[r] * u[c])
            }
        };
        let x = pass.streams[site].to_owned();
        let patched = &x - &w;
        let takes = |r: usize| route.is_none_or(|k| k == r);
        let norms = |a: &Array2<f64>| a.map_axis(Axis(1), |row| row.dot(&row).sqrt());
        let (change, activity, run) = match read {
            Ok(h) => {
                let pick = |r: usize| if takes(r) { &patched } else { &x };
                let (z, _) = self.head_on(h, pick(0), pick(1), pick(2), length);
                let (old, new) = (self.write(h, &pass.head[h][2].to_owned()), self.write(h, &z));
                ((&new - &old), [norms(&old), norms(&new)], self.run_with(sequences, &[(h, z)].into(), &BTreeMap::new())?)
            }
            Err((b, i)) => {
                let block = &self.mlps[b];
                let site = &self.sites[site];
                let read_of = |stream: &Array2<f64>, row: ArrayView1<f64>, bias: f64| -> Array1<f64> {
                    let direction = &row * &site.gain;
                    stream.outer_iter().map(|x| direction.dot(&x) * rms_scale(x, site.epsilon) + bias).collect()
                };
                let gate = if takes(0) { read_of(&patched, block.gate.row(i), block.bias[i]) } else { pass.mlp[b].1.column(i).to_owned() };
                let h = match (&block.up, &pass.mlp[b].2) {
                    (Some((_, up)), Some(base_up)) => {
                        let up = if takes(1) { read_of(&patched, up.row(i), block.up_bias[i]) } else { base_up.column(i).to_owned() };
                        Array1::from_shape_fn(gate.len(), |r| block.law.apply(gate[r]) * up[r])
                    }
                    _ => gate.mapv(|g| block.law.apply(g)),
                };
                let old = pass.mlp[b].0.column(i).to_owned();
                let u = block.out.column(i);
                let change = Array2::from_shape_fn((pass.rows, u.len()), |(r, c)| (h[r] - old[r]) * u[c]);
                let mut activations = pass.mlp[b].0.to_owned();
                activations.column_mut(i).assign(&h);
                (change, [old, h], self.run_with(sequences, &BTreeMap::new(), &[(b, activations)].into())?)
            }
        };
        let [before, after] = activity;
        Ok(PathPatch { change, before, after, last: run.last, base: pass.last.into_owned() })
    }

    /// The reader `f`'s read maps in residual coordinates with its norm's gain folded in (routes ×
    /// rows of the map × width: an MLP function's gate and up direction, a head's query, key and
    /// value maps, as in [`Library::contributions`]), and the read site (`2l` a layer's attention,
    /// `2l + 1` its MLP).
    pub fn read_maps(&self, f: usize) -> Result<(usize, Vec<Array2<f64>>), String> {
        let (layer, kind) = self.place(f)?;
        let site = if kind.is_ok() { 2 * layer } else { 2 * layer + 1 };
        let gain = &self.sites[site].gain;
        let maps = match kind {
            Ok(h) => {
                let block = &self.heads[h];
                [&block.query.map, &block.key.map, &block.value].iter().map(|m| *m * &gain.view().insert_axis(Axis(0))).collect()
            }
            Err((b, i)) => {
                let block = &self.mlps[b];
                let mut maps = vec![(&block.gate.row(i) * gain).insert_axis(Axis(0))];
                if let Some((_, up)) = &block.up {
                    maps.push((&up.row(i) * gain).insert_axis(Axis(0)));
                }
                maps
            }
        };
        Ok((site, maps))
    }

    /// The exact contribution of each writer's write to each reader's reads on `sequences` (of one
    /// length), as the mean over rows of its magnitude: writers and readers are indices into
    /// [`Library::functions`]. A read is linear in the residual stream before its norm, so at a
    /// row with stream `x` and `r = √(mean(x²) + ε)` (the actual RMS) writer `s`'s write `w_s`
    /// enters reader `t`'s read as `R_t (γ ⊙ w_s) / r`, `R_t` the read map: an MLP function's gate
    /// (route 0) and up direction (route 1), a head's query, key and value maps (routes 0, 1, 2,
    /// each the norm of the head-width vector, before a head norm). Result: readers × writers × 3,
    /// zero where the writer does not write before the reader reads.
    pub fn contributions(&self, sequences: &[Vec<u32>], writers: &[usize], readers: &[usize]) -> Result<ndarray::Array3<f64>, String> {
        let pass = self.pass(&sequences.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
        let (head_columns, mlp_columns) = self.columns();
        // A function's place: (layer, head index) or (layer, (MLP, function)).
        let mut place: BTreeMap<usize, (usize, Result<usize, (usize, usize)>)> = BTreeMap::new();
        for (l, members) in self.layer_heads.iter().enumerate() {
            for (c, &h) in members.iter().enumerate() {
                place.insert(head_columns[l] + c, (l, Ok(h)));
            }
        }
        for (b, block) in self.mlps.iter().enumerate() {
            for i in 0..block.gate.nrows() {
                place.insert(mlp_columns[b] + i, (block.layer, Err((b, i))));
            }
        }
        let find = |f: usize| place.get(&f).cloned().ok_or_else(|| format!("no function {f}"));
        let mut head_writes: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
        for &writer in writers {
            if let (_, Ok(h)) = find(writer)? {
                head_writes.insert(h, self.write(h, &pass.head[h][2].to_owned()));
            }
        }
        let mut out = ndarray::Array3::<f64>::zeros((readers.len(), writers.len(), 3));
        for (ri, &reader) in readers.iter().enumerate() {
            let (layer, kind) = find(reader)?;
            let site = if kind.is_ok() { 2 * layer } else { 2 * layer + 1 };
            let (gain, inverse) = (&self.sites[site].gain, &pass.inverse[site]);
            // The reader's read maps in residual coordinates (route × rows of the map × width).
            let maps: Vec<Array2<f64>> = match kind {
                Ok(h) => {
                    let block = &self.heads[h];
                    [&block.query.map, &block.key.map, &block.value].iter().map(|m| *m * &gain.view().insert_axis(Axis(0))).collect()
                }
                Err((b, i)) => {
                    let block = &self.mlps[b];
                    let mut maps = vec![(&block.gate.row(i) * gain).insert_axis(Axis(0))];
                    if let Some((_, up)) = &block.up {
                        maps.push((&up.row(i) * gain).insert_axis(Axis(0)));
                    }
                    maps
                }
            };
            for (wi, &writer) in writers.iter().enumerate() {
                let (written, write) = find(writer)?;
                // A head writes into the stream its layer's MLP reads; an MLP into the next layer's.
                let before = match write {
                    Ok(_) => 2 * written + 1 <= site,
                    Err(_) => 2 * written + 2 <= site,
                };
                if !before {
                    continue;
                }
                for (route, map) in maps.iter().enumerate() {
                    out[[ri, wi, route]] = match write {
                        Ok(h) => {
                            let read = head_writes[&h].dot(&map.t());
                            read.outer_iter().zip(inverse.iter()).map(|(row, inv)| row.dot(&row).sqrt() * inv).sum::<f64>() / pass.rows as f64
                        }
                        Err((b, i)) => {
                            let read = map.dot(&self.mlps[b].out.column(i));
                            let size = read.dot(&read).sqrt();
                            pass.mlp[b].0.column(i).iter().zip(inverse.iter()).map(|(h, inv)| (h * inv).abs()).sum::<f64>() * size / pass.rows as f64
                        }
                    };
                }
            }
        }
        Ok(out)
    }

    /// The measured effect of removing each function: per function ([`Library::functions`]'s
    /// order) and per row of `sequences` (of one length), the change in `KL(M ‖ P)` of the
    /// next-token distributions, in bits, when that function's write alone is taken out of the
    /// explanation (an MLP function's `h_i u_i`, a head's `W_O,h z_h`) and every layer after it
    /// recomputed. `teacher` gives `M`'s distributions; `batch` functions are removed at a time,
    /// each in its own copy of the sequences, entering at the stream after the function's block.
    /// With `scored`, only those rows are scored (the columns of the result, in that order).
    pub fn removal_effects(&self, teacher: &Teacher, sequences: &[Vec<u32>], scored: Option<&[usize]>, batch: usize) -> Result<Array2<f64>, String> {
        if batch == 0 || sequences.is_empty() {
            return Err("a positive batch and held-out sequences required".into());
        }
        let refs: Vec<&[u32]> = sequences.iter().map(Vec::as_slice).collect();
        let family = sequence_family(&refs)?;
        let rows = family.rows;
        let all: Vec<usize> = (0..rows).collect();
        let scored = scored.unwrap_or(&all);
        if scored.iter().any(|r| *r >= rows) {
            return Err("a scored row outside the sequences".into());
        }
        let resident = &self.resident;
        let arithmetic = arithmetic(self.model);
        let hidden = self.program.hidden();
        // The scored rows of `copies` copies of the sequences (copy `k`'s rows at `k · rows`).
        let selection = |copies: usize| -> Result<_, String> {
            let indices: Vec<u32> = (0..copies).flat_map(|k| scored.iter().map(move |r| (k * rows + r) as u32)).collect();
            self.model.upload_indices(&indices).map_err(error)
        };
        let full = teacher.target(&family, None)?;
        let target = Target {
            mu: std::sync::Arc::new(self.model.gather_rows(&full.mu, &selection(1)?).map_err(error)?),
            entropy: scored.iter().map(|r| full.entropy[*r]).collect(),
            head: std::sync::Arc::clone(&full.head),
            scored: None,
        };
        drop(full);
        let score = |trace: &DeviceTrace, copies: usize, target: &Target| -> Result<Vec<f64>, String> {
            let selected = self.model.gather_rows(trace.value(hidden)?, &selection(copies)?).map_err(error)?;
            Ok(resident.score(self.model, &selected, target, false, None, arithmetic)?.0)
        };
        let base = score(&self.program.forward(&family)?, 1, &target)?;
        let pass = self.pass(&refs)?;
        let n = scored.len();
        let mut effects = Array2::<f64>::zeros((self.functions().len(), n));
        self.sweep(&refs, &pass, None, batch, |columns, trace| {
            let indices = self.model.upload_indices(&(0..columns.len()).flat_map(|_| 0..n as u32).collect::<Vec<_>>()).map_err(error)?;
            let copied_target = Target {
                mu: std::sync::Arc::new(self.model.gather_rows(&target.mu, &indices).map_err(error)?),
                entropy: (0..columns.len()).flat_map(|_| target.entropy.iter().copied()).collect(),
                head: std::sync::Arc::clone(&target.head),
                scored: None,
            };
            let kl = score(trace, columns.len(), &copied_target)?;
            for (k, column) in columns.iter().enumerate() {
                for r in 0..n {
                    effects[[*column, r]] = (kl[k * n + r] - base[r]) / std::f64::consts::LN_2;
                }
            }
            Ok(())
        })?;
        Ok(effects)
    }

    /// Activation patching of every function on prompt pairs: per function ([`Library::functions`]'s
    /// order) and per pair (a clean prompt and a counterfactual of its length) with its tokens
    /// `(target, foil)`, the change of the logit difference `target − foil` at the last position
    /// when the function's write on the clean prompt is replaced at every position by its write on
    /// the counterfactual and every later layer rerun; `batch` functions at a time.
    pub fn patch_effects(&self, pairs: &[(Vec<u32>, Vec<u32>)], tokens: &[(u32, u32)], batch: usize) -> Result<Array2<f64>, String> {
        let vocabulary = self.unembedding.nrows() as u32;
        if batch == 0 || pairs.len() != tokens.len() || tokens.iter().any(|(t, f)| *t >= vocabulary || *f >= vocabulary) {
            return Err("a positive batch and one target and foil in the vocabulary per pair required".into());
        }
        let mut effects = Array2::<f64>::zeros((self.functions().len(), pairs.len()));
        let last_node = self.observed[self.sites.len()];
        for (p, ((clean, counterfactual), (target, foil))) in pairs.iter().zip(tokens).enumerate() {
            let length = clean.len();
            if length == 0 || counterfactual.len() != length {
                return Err("a clean prompt and a counterfactual of its length required".into());
            }
            let direction = &self.unembedding.row(*target as usize) - &self.unembedding.row(*foil as usize);
            let difference = |x: ArrayView1<f64>| direction.dot(&x) * rms_scale(x, self.final_site.epsilon);
            let refs = [clean.as_slice()];
            let pass = self.pass(&refs)?;
            let other = self.pass(&[counterfactual.as_slice()])?;
            let base = difference(pass.last.row(length - 1));
            self.sweep(&refs, &pass, Some(&other), batch, |columns, trace| {
                let rows = self.model.upload_indices(&(0..columns.len()).map(|k| (k * length + length - 1) as u32).collect::<Vec<_>>()).map_err(error)?;
                let last = self.model.download(&self.model.gather_rows(trace.value(last_node)?, &rows).map_err(error)?).map_err(error)?;
                for (k, column) in columns.iter().enumerate() {
                    effects[[*column, p]] = difference(last.row(k)) - base;
                }
                Ok(())
            })?;
        }
        Ok(effects)
    }

    /// Every function's write in `pass` (the run of `refs`) taken out at every position, or
    /// replaced by its write in `replacement` (a run of sequences of the same shape), `batch`
    /// functions at a time, each in its own copy of the sequences entering at the stream after its
    /// block and every later layer rerun; `measure` reads each batch's columns
    /// ([`Library::functions`]'s order, copy `k` at rows `k · rows`) and trace.
    fn sweep(&self, refs: &[&[u32]], pass: &Pass<'_>, replacement: Option<&Pass<'_>>, batch: usize, mut measure: impl FnMut(&[usize], &DeviceTrace) -> Result<(), String>) -> Result<(), String> {
        if batch == 0 || replacement.is_some_and(|r| r.rows != pass.rows) {
            return Err("a positive batch and a replacement run of the same rows required".into());
        }
        let rows = pass.rows;
        let (head_columns, mlp_columns) = self.columns();
        // Per layer: the heads' writes enter the stream before its MLP, the MLP functions' the
        // stream after it (the next layer's, or the final one).
        // (column, entering stream, head or (MLP, function)).
        let mut jobs: Vec<(usize, usize, Result<usize, (usize, usize)>)> = Vec::new();
        for (l, members) in self.layer_heads.iter().enumerate() {
            for (c, &h) in members.iter().enumerate() {
                jobs.push((head_columns[l] + c, 2 * l + 1, Ok(h)));
            }
            for (b, block) in self.mlps.iter().enumerate().filter(|(_, b)| b.layer == l) {
                let entry = if 2 * l + 2 < self.sites.len() { 2 * l + 2 } else { self.sites.len() };
                jobs.extend((0..block.gate.nrows()).map(|i| (mlp_columns[b] + i, entry, Err((b, i)))));
            }
        }
        let write = |pass: &Pass<'_>, job: &Result<usize, (usize, usize)>| -> Array2<f64> {
            match job {
                Ok(h) => self.write(*h, &pass.head[*h][2].to_owned()),
                Err((b, i)) => {
                    let (h, u) = (pass.mlp[*b].0.column(*i), self.mlps[*b].out.column(*i));
                    Array2::from_shape_fn((rows, u.len()), |(r, c)| h[r] * u[c])
                }
            }
        };
        let stream = |entry: usize| if entry < self.sites.len() { &pass.streams[entry] } else { &pass.last };
        let hidden = self.program.hidden();
        let mut start = 0;
        while start < jobs.len() {
            // A batch changes functions entering at one stream.
            let entry = jobs[start].1;
            let end = (start + batch).min(jobs.len());
            let end = start + jobs[start..end].iter().take_while(|j| j.1 == entry).count();
            let chunk = &jobs[start..end];
            let x = stream(entry);
            let mut values = Array2::<f64>::zeros((chunk.len() * rows, x.ncols()));
            for (k, (_, _, job)) in chunk.iter().enumerate() {
                let mut edited = x - &write(pass, job);
                if let Some(other) = replacement {
                    edited += &write(other, job);
                }
                values.slice_mut(s![k * rows..(k + 1) * rows, ..]).assign(&edited);
            }
            let copies: Vec<&[u32]> = (0..chunk.len()).flat_map(|_| refs.iter().copied()).collect();
            let copied = sequence_family(&copies)?;
            let trace = self.program.forward_span(&copied, Some((self.observed[entry], self.model.upload(values.view()).map_err(error)?)), hidden, |_, _| Ok(None))?;
            measure(&chunk.iter().map(|j| j.0).collect::<Vec<_>>(), &trace)?;
            start = end;
        }
        Ok(())
    }

    /// Runs of the explanation with functions' writes scaled: copy `k` is `sequences[edits[k].sequence]`
    /// (all of one length) with each function `f` of `edits[k].scale` ([`Library::functions`]'s
    /// order) writing `α_f` times its own write at every position (an MLP function's `α h_i u_i`
    /// through its activation, a head's `α W_O,h z_h` through its read; `α = 1` is the explanation,
    /// `α = 0` the function's removal) and every later computation run on the edited values, all
    /// copies in one forward pass. Per read row of each copy (`edits[k].rows`, in order): the final
    /// stream before the final norm and, with `activity`, every function's activity there (an MLP
    /// function's activation `h_i`, a head's output norm `‖W_O,h z_h‖`).
    pub fn edited(&self, sequences: &[Vec<u32>], edits: &[Edit], activity: bool) -> Result<Edited, String> {
        let length = sequences.first().map_or(0, Vec::len);
        if edits.is_empty() || length == 0 || edits.iter().any(|e| e.sequence >= sequences.len() || e.rows.iter().any(|r| *r >= length)) {
            return Err("edits of rows inside the sequences required".into());
        }
        let copies: Vec<&[u32]> = edits.iter().map(|e| sequences[e.sequence].as_slice()).collect();
        let family = sequence_family(&copies)?;
        let rows = family.rows;
        // Per edited node (an MLP's activations, a head's read), the factor of every value.
        let mut masks: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
        for (k, edit) in edits.iter().enumerate() {
            for &(f, alpha) in &edit.scale {
                let (node, column, width) = match self.place(f)?.1 {
                    Ok(h) => (self.observed[self.head_paths + 6 * h + 2], None, self.heads[h].output.ncols()),
                    Err((b, i)) => (self.observed[self.mlp_paths[b]], Some(i), self.mlps[b].gate.nrows()),
                };
                let mask = masks.entry(node).or_insert_with(|| Array2::ones((rows, width)));
                let mut span = mask.slice_mut(s![k * length..(k + 1) * length, ..]);
                match column {
                    Some(i) => span.column_mut(i).mapv_inplace(|v| v * alpha),
                    None => span.mapv_inplace(|v| v * alpha),
                }
            }
        }
        let masks: BTreeMap<usize, Tensor> = masks.into_iter().map(|(n, m)| Ok((n, self.model.upload(m.view()).map_err(error)?))).collect::<Result<_, String>>()?;
        let edit = |node: usize, trace: &DeviceTrace| -> Result<Option<Tensor>, String> {
            let Some(mask) = masks.get(&node) else { return Ok(None) };
            let value = trace.value(node)?;
            let mut out = self.model.zeros(value.rows(), value.cols()).map_err(error)?;
            self.model.hadamard(&mut out, value, mask, false).map_err(error)?;
            Ok(Some(out))
        };
        let trace = self.program.forward_edited(&family, BTreeMap::new(), &BTreeSet::new(), |_, _| Ok(()), edit)?;
        let read: Vec<u32> = edits.iter().enumerate().flat_map(|(k, e)| e.rows.iter().map(move |r| (k * length + r) as u32)).collect();
        let indices = self.model.upload_indices(&read).map_err(error)?;
        let gather = |node: usize| -> Result<Array2<f64>, String> { self.model.download(&self.model.gather_rows(trace.value(node)?, &indices).map_err(error)?).map_err(error) };
        let last = gather(self.observed[self.sites.len()])?;
        let activity = if activity {
            let (head_columns, mlp_columns) = self.columns();
            let mut out = Array2::<f64>::zeros((read.len(), self.functions().len()));
            for (l, members) in self.layer_heads.iter().enumerate() {
                for (c, &h) in members.iter().enumerate() {
                    let written = self.write(h, &gather(self.observed[self.head_paths + 6 * h + 2])?);
                    out.column_mut(head_columns[l] + c).assign(&written.map_axis(Axis(1), |row| row.dot(&row).sqrt()));
                }
            }
            for (b, block) in self.mlps.iter().enumerate() {
                out.slice_mut(s![.., mlp_columns[b]..mlp_columns[b] + block.gate.nrows()]).assign(&gather(self.observed[self.mlp_paths[b]])?);
            }
            Some(out)
        } else {
            None
        };
        Ok(Edited { last, activity })
    }

    /// Each function's write at one row of a run of `sequence`: functions × width
    /// ([`Library::functions`]'s order), with the final stream's `1/r` there.
    pub fn writes_at(&self, sequence: &[u32], row: usize) -> Result<(Array2<f64>, f64), String> {
        let (writes, inverse, _) = self.writes_and_reads_at(sequence, row)?;
        Ok((writes, inverse))
    }

    /// [`Library::writes_at`] with each read site's `1/r` at the row.
    pub fn writes_and_reads_at(&self, sequence: &[u32], row: usize) -> Result<(Array2<f64>, f64, Vec<f64>), String> {
        let pass = self.pass(&[sequence])?;
        let mut out = Array2::<f64>::zeros((self.functions().len(), self.unembedding.ncols()));
        let (head_columns, mlp_columns) = self.columns();
        for (l, members) in self.layer_heads.iter().enumerate() {
            for (c, &h) in members.iter().enumerate() {
                out.row_mut(head_columns[l] + c).assign(&self.heads[h].output.dot(&pass.head[h][2].row(row)));
            }
        }
        for (b, block) in self.mlps.iter().enumerate() {
            for i in 0..block.gate.nrows() {
                out.row_mut(mlp_columns[b] + i).assign(&(&block.out.column(i) * pass.mlp[b].0[[row, i]]));
            }
        }
        Ok((out, pass.inverse_final[row], pass.inverse.iter().map(|v| v[row]).collect()))
    }

    /// The unembedding row of token `t` with the final norm's gain (`γ_f ⊙ e_t`).
    #[must_use]
    pub fn unembedding_row(&self, t: usize) -> ArrayView1<'_, f64> {
        self.unembedding.row(t)
    }

    /// The functions' maps, per layer and kind, as named arrays (shape, row-major values): an MLP's
    /// gate directions `g_i` (functions × width, read on the normed stream `x̂`), gate biases
    /// `c_i`, write directions `u_i` (functions × width) and, when gated, up directions; a layer's
    /// heads' query, key and value maps (heads × head width × width, on `x̂`) and output maps
    /// (heads × width × head width).
    #[must_use]
    pub fn maps(&self) -> Vec<(String, Vec<usize>, Vec<f64>)> {
        let mut out = Vec::new();
        for (l, members) in self.layer_heads.iter().enumerate() {
            if !members.is_empty() {
                for key in ["query", "key", "value", "output"] {
                    let map = |b: &HeadBlock| -> Array2<f64> {
                        match key {
                            "query" => b.query.map.clone(),
                            "key" => b.key.map.clone(),
                            "value" => b.value.clone(),
                            _ => b.output.clone(),
                        }
                    };
                    let (r, c) = map(&self.heads[members[0]]).dim();
                    let values: Vec<f64> = members.iter().flat_map(|h| map(&self.heads[*h]).iter().copied().collect::<Vec<_>>()).collect();
                    out.push((format!("h.{l}.attn.head.{key}"), vec![members.len(), r, c], values));
                }
            }
            for block in self.mlps.iter().filter(|b| b.layer == l) {
                let rows = |a: &Array2<f64>| (vec![a.nrows(), a.ncols()], a.iter().copied().collect());
                let (shape, values) = rows(&block.gate);
                out.push((format!("h.{l}.mlp.function.gate"), shape, values));
                out.push((format!("h.{l}.mlp.function.bias"), vec![block.bias.len()], block.bias.to_vec()));
                let (shape, values) = rows(&block.out.t().to_owned());
                out.push((format!("h.{l}.mlp.function.write"), shape, values));
                if let Some((_, up)) = &block.up {
                    let (shape, values) = rows(up);
                    out.push((format!("h.{l}.mlp.function.up"), shape, values));
                }
            }
        }
        out
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
    fn pass(&self, batch: &[&[u32]]) -> Result<Pass<'static>, String> {
        self.pass_with(batch, &BTreeMap::new(), &BTreeMap::new())
    }

    /// [`Library::pass`] with each head in `heads` (an index into the heads) reading the given
    /// values (rows × head width) in place of its own, and each MLP in `mlps` (an index into the
    /// MLPs) taking the given activations (rows × functions); everything after recomputed.
    fn pass_with(&self, batch: &[&[u32]], heads: &BTreeMap<usize, Array2<f64>>, mlps: &BTreeMap<usize, Array2<f64>>) -> Result<Pass<'static>, String> {
        let family = sequence_family(batch)?;
        let p = self.head_paths;
        let trace = if heads.is_empty() && mlps.is_empty() {
            self.program.forward(&family)?
        } else {
            let at: BTreeMap<usize, &Array2<f64>> = heads
                .iter()
                .map(|(h, z)| (self.observed[p + 6 * h + 2], z))
                .chain(mlps.iter().map(|(b, a)| (self.observed[self.mlp_paths[*b]], a)))
                .collect();
            if at.values().any(|z| z.nrows() != family.rows) {
                return Err("a replaced read of another number of rows".into());
            }
            let edit = |node: usize, _: &DeviceTrace| -> Result<Option<Tensor>, String> {
                at.get(&node).map(|z| self.model.upload(z.view()).map_err(error)).transpose()
            };
            self.program.forward_edited(&family, BTreeMap::new(), &BTreeSet::new(), |_, _| Ok(()), edit)?
        };
        let get = |i: usize| -> Result<Array2<f64>, String> { self.model.download(trace.value(self.observed[i])?).map_err(error) };
        let inverse_of = |x: &Array2<f64>, epsilon: f64| x.map_axis(Axis(1), |row| rms_scale(row, epsilon));
        let streams: Vec<Array2<f64>> = (0..self.sites.len()).map(get).collect::<Result<_, String>>()?;
        let inverse = streams.iter().zip(&self.sites).map(|(x, s)| CowArray::from(inverse_of(x, s.epsilon))).collect();
        let last = get(self.sites.len())?;
        let inverse_final = CowArray::from(inverse_of(&last, self.final_site.epsilon));
        let mlp = self
            .mlps
            .iter()
            .enumerate()
            .map(|(b, block)| {
                let i = self.mlp_paths[b];
                Ok((CowArray::from(get(i)?), CowArray::from(get(i + 1)?), if block.up.is_some() { Some(CowArray::from(get(i + 2)?)) } else { None }))
            })
            .collect::<Result<_, String>>()?;
        let head: Vec<[Array2<f64>; 6]> = (0..self.heads.len())
            .map(|h| Ok([get(p + 6 * h)?, get(p + 6 * h + 1)?, get(p + 6 * h + 2)?, get(p + 6 * h + 3)?, get(p + 6 * h + 4)?, get(p + 6 * h + 5)?]))
            .collect::<Result<_, String>>()?;
        let weights = self.attend(&head, batch.len(), batch[0].len()).into_iter().map(|w| w.into_iter().map(CowArray::from).collect()).collect();
        let head = head.into_iter().map(|values| values.map(CowArray::from)).collect();
        Ok(Pass { rows: family.rows, streams: streams.into_iter().map(CowArray::from).collect(), inverse, last: CowArray::from(last), inverse_final, mlp, head, weights })
    }

    /// The model's predicted token per row and the gradient of its centred logit in the final
    /// stream (the final norm's denominator frozen), with the centred logit itself.
    fn predicted(&self, pass: &Pass<'_>) -> Result<(Vec<usize>, Array2<f64>, Vec<f64>), String> {
        let predicted: Vec<usize> = extreme_columns(self.wide, &self.unembedding_table, &pass.last, 1, 1.0, self.tile_rows)?.into_iter().map(|c| c[0]).collect();
        let seed = Array2::from_shape_fn(pass.last.dim(), |(r, c)| (self.unembedding[[predicted[r], c]] - self.unembedding_mean[c]) * pass.inverse_final[r]);
        let metric = (0..pass.rows).map(|r| seed.row(r).dot(&pass.last.row(r))).collect();
        Ok((predicted, seed, metric))
    }

    /// The first column of each layer's heads and of each MLP's functions in [`Library::functions`]'s order.
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
}

// ------------------------------------------------------------------------------ the read-out

/// The read-out (module note) of `library` on the held-out `sequences` (of equal length), its
/// measured removals and path patches on the first `Settings::measured_sequences` of them, `M`'s
/// distributions from `teacher`.
pub fn read_out(library: &Library, teacher: &Teacher, sequences: &[Vec<u32>], settings: &Settings) -> Result<Readout, String> {
    let measured = settings.measured_sequences.min(sequences.len());
    if settings.batch_sequences == 0 || settings.removal_batch == 0 || measured == 0 {
        return Err("invalid read-out settings or no held-out sequences".into());
    }
    let length = sequences[0].len();
    if length == 0 || sequences.iter().any(|s| s.len() != length) {
        return Err("held-out sequences must be nonempty and of equal length".into());
    }
    let Library { sites, mlps, heads, unembedding, unembedding_mean, unembedding_table, embedding, wide, layer_heads, .. } = library;
    let layers = layer_heads.len();
    let (vocabulary, width) = embedding.dim();
    // Per read epsilon, the embedding's rows through that read's norm (on the host and on `wide`).
    let mut epsilons: Vec<f64> = sites.iter().map(|s| s.epsilon).collect();
    epsilons.sort_by(f64::total_cmp);
    epsilons.dedup();
    let mut embedding_tables: BTreeMap<u64, (Array2<f64>, Tensor)> = BTreeMap::new();
    for epsilon in &epsilons {
        let table = normed_rows(embedding.clone(), *epsilon);
        let resident = wide.upload(table.view()).map_err(error)?;
        embedding_tables.insert(epsilon.to_bits(), (table, resident));
    }

    let k = settings.contexts;
    let mut mlp_stats: Vec<Vec<MlpStat>> = mlps.iter().map(|b| vec![MlpStat::default(); b.gate.nrows()]).collect();
    let mut head_stats: Vec<HeadStat> = heads
        .iter()
        .map(|_| HeadStat { output: 0.0, top: Best::default(), offsets: Vec::new(), sources: HashMap::new(), attended: Vec::new(), induction: 0.0, duplicate: 0.0 })
        .collect();
    let grams: Vec<Array2<f64>> = heads.iter().map(|b| b.output.t().dot(&b.output)).collect();
    let out_norms: Vec<Array1<f64>> = mlps.iter().map(|b| b.out.map_axis(Axis(0), |u| u.dot(&u).sqrt())).collect();
    let mut predicted_all: Vec<u32> = Vec::with_capacity(sequences.len() * length);
    for (chunk, batch) in sequences.chunks(settings.batch_sequences).enumerate() {
        let first_row = chunk * settings.batch_sequences * length;
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
        predicted_all.extend(library.predicted(&pass)?.0.iter().map(|t| *t as u32));
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

    // Measured removals on the first sequences: per function (Library::functions' columns) and
    // measured token, the change of KL(M ‖ P) in bits.
    let measured_sequences = &sequences[..measured];
    let effects = library.removal_effects(teacher, measured_sequences, None, settings.removal_batch)?;
    let column: BTreeMap<String, usize> = library.functions().into_iter().enumerate().map(|(c, f)| (f.name, c)).collect();
    // The predicted token a majority of rows share.
    let majority = |best: &Best| {
        let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
        best.entries.iter().for_each(|(_, id)| *counts.entry(predicted_all[*id]).or_insert(0) += 1);
        counts.into_iter().find(|(_, c)| 2 * c > best.entries.len()).map(|(t, _)| t)
    };
    let measure = |f: &mut Function| -> Result<(), String> {
        let row = effects.row(*column.get(&f.name).ok_or_else(|| format!("{}: not measured", f.name))?);
        let mut sorted = row.to_vec();
        sorted.sort_by(f64::total_cmp);
        let quantile = |q: f64| sorted[((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1];
        f.importance = row.mean().unwrap_or(0.0);
        f.effect_quantiles = [quantile(0.5), quantile(0.9), quantile(0.99), quantile(1.0)];
        let (mut supports, mut opposes) = (Best::default(), Best::default());
        for (id, effect) in row.iter().enumerate() {
            supports.offer(k, *effect, id);
            opposes.offer(k, -*effect, id);
        }
        f.supports = supports.entries.iter().map(|(v, id)| context(*v, *id, None)).collect();
        f.opposes = opposes.entries.iter().map(|(v, id)| context(-*v, *id, None)).collect();
        f.supports_token = majority(&supports);
        f.opposes_token = majority(&opposes);
        Ok(())
    };
    let blank = |name: String, layer: usize, kind: Kind| Function {
        name,
        layer,
        kind,
        positive_gate_fraction: None,
        usage: 0.0,
        contexts: Vec::new(),
        promoted: Vec::new(),
        suppressed: Vec::new(),
        reads: Vec::new(),
        inputs: Vec::new(),
        attention: None,
        importance: 0.0,
        effect_quantiles: [0.0; 4],
        supports: Vec::new(),
        opposes: Vec::new(),
        supports_token: None,
        opposes_token: None,
    };

    // The functions, surviving ones only: per layer its heads, then its MLP functions.
    let mut functions: Vec<Function> = Vec::new();
    let mut removed = Vec::new();
    let mut head_index = vec![None; heads.len()];
    let mut mlp_index: Vec<Vec<Option<usize>>> = mlps.iter().map(|b| vec![None; b.gate.nrows()]).collect();
    for l in 0..layers {
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
            measure(&mut f)?;
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
                measure(&mut f)?;
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

    // Edges: among the most important functions, each core reader's writers ranked by exact
    // contribution, the strongest measured by path patching; the core's wiring measured whole.
    let mut ranked: Vec<usize> = (0..functions.len()).collect();
    ranked.sort_by(|a, b| functions[*b].importance.total_cmp(&functions[*a].importance));
    let core: Vec<usize> = ranked.iter().copied().take(settings.core).collect();
    ranked.truncate(settings.candidates.max(settings.core));
    let columns_of = |list: &[usize]| list.iter().map(|f| column[&functions[*f].name]).collect::<Vec<_>>();
    let contributions = library.contributions(measured_sequences, &columns_of(&ranked), &columns_of(&core))?;
    let mut measured_edges: BTreeMap<(usize, usize), (f64, f64)> = BTreeMap::new();
    let mut patch = |writer: usize, reader: usize| -> Result<(f64, f64), String> {
        if let Some(found) = measured_edges.get(&(writer, reader)) {
            return Ok(*found);
        }
        let [output, bits] = library.path_patch(measured_sequences, column[&functions[writer].name], column[&functions[reader].name], None)?;
        measured_edges.insert((writer, reader), (output, bits));
        Ok((output, bits))
    };
    let mut wiring = vec![vec![0.0; core.len()]; core.len()];
    let mut inputs: Vec<(usize, Vec<Edge>)> = Vec::new();
    for (ri, &reader) in core.iter().enumerate() {
        let mut candidates: Vec<(usize, [f64; 3])> = ranked
            .iter()
            .enumerate()
            .map(|(wi, &writer)| (writer, [contributions[[ri, wi, 0]], contributions[[ri, wi, 1]], contributions[[ri, wi, 2]]]))
            .filter(|(_, c)| c.iter().sum::<f64>() > 0.0)
            .collect();
        candidates.sort_by(|a, b| b.1.iter().sum::<f64>().total_cmp(&a.1.iter().sum::<f64>()));
        candidates.truncate(settings.edges);
        let mut edges = Vec::with_capacity(candidates.len());
        for (writer, contribution) in candidates {
            let (output, prediction_bits) = patch(writer, reader)?;
            edges.push(Edge { from: writer, contribution, output, prediction_bits });
        }
        for (wi, &writer) in core.iter().enumerate() {
            let w = ranked.iter().position(|f| *f == writer).ok_or("a core function outside the candidates")?;
            if (0..3).any(|route| contributions[[ri, w, route]] > 0.0) {
                wiring[ri][wi] = patch(writer, reader)?.1;
            }
        }
        inputs.push((reader, edges));
    }
    for (reader, edges) in inputs {
        functions[reader].inputs = edges;
    }
    let mut participation: Vec<f64> = effects
        .columns()
        .into_iter()
        .map(|column| {
            let (absolute, square) = column.iter().fold((0.0, 0.0), |(a, q), v| (a + v.abs(), q + v * v));
            if square > 0.0 { absolute * absolute / square } else { 0.0 }
        })
        .collect();
    participation.sort_by(f64::total_cmp);
    let quantile = |q: f64| participation[((q * participation.len() as f64).ceil() as usize).clamp(1, participation.len()) - 1];
    Ok(Readout {
        held_out_tokens: sequences.len() * length,
        measured_tokens: effects.ncols(),
        functions,
        removed,
        core,
        wiring,
        predicted: predicted_all,
        participation: [quantile(0.1), quantile(0.5), quantile(0.9), participation.iter().sum::<f64>() / participation.len() as f64],
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
    use super::Library;
    use crate::{
        import::import_language_model,
        library_mdl,
        operator_program::SlotValues,
        run_check::{layer_nodes, split_sites},
        test_support::{tiny_export, tiny_qwen3_export},
    };
    use gam_gpu::tensor::Device;

    /// A head recomputed on the run's own streams gives its read, a run whose head reads are
    /// replaced by their own values reproduces the run, and a replaced read changes what follows.
    #[test]
    fn heads_recompute_and_replace_exactly() {
        for dir in [tiny_export("readout_heads_gelu", 2), tiny_qwen3_export("readout_heads_gated", 2)] {
            let imported = import_language_model(&dir, 6, 12).expect("tiny export");
            std::fs::remove_dir_all(dir).expect("remove the tiny export");
            let native = split_sites(&imported.program).expect("split sites");
            let layers = layer_nodes(&native, 2).expect("layer nodes");
            let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
            let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
            let artifact = library_mdl::explanation(&native, &layers).expect("library").artifact;
            let host = Device::host();
            let library = Library::new(&host, &host, &native, &layers, &artifact, 1 << 30, 64).expect("library on the host");
            let base = library.run(&sequences, &std::collections::BTreeMap::new()).expect("run");
            let largest = |a: &ndarray::Array2<f64>, b: &ndarray::Array2<f64>| (a - b).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            for (h, (layer, _)) in library.heads().into_iter().enumerate() {
                let stream = &base.streams[2 * layer];
                let (z, _) = library.head_on(h, stream, stream, stream, 12);
                assert!(largest(&z, &base.reads[h]) < 1e-10, "head {h} recomputed off by {}", largest(&z, &base.reads[h]));
            }
            let same: std::collections::BTreeMap<usize, ndarray::Array2<f64>> = [(0, base.reads[0].clone())].into();
            assert!(largest(&library.run(&sequences, &same).expect("run").last, &base.last) < 1e-12);
            let zero: std::collections::BTreeMap<usize, ndarray::Array2<f64>> = [(0, ndarray::Array2::zeros(base.reads[0].dim()))].into();
            assert!(largest(&library.run(&sequences, &zero).expect("run").last, &base.last) > 1e-6);
            let log_p = library.log_probabilities(&base.last).expect("log probabilities");
            assert!(log_p.outer_iter().all(|row| (row.mapv(f64::exp).sum() - 1.0).abs() < 1e-12));
        }
    }

    /// A copy of [`Library::edited`] scaling a head's write equals the run with its read scaled, one
    /// scaling an MLP function's write equals the run with that activation scaled, and copies in one
    /// call equal their separate calls.
    #[test]
    fn scaled_writes_match_scaled_values() {
        use super::Edit;
        for dir in [tiny_export("readout_scaled_gelu", 2), tiny_qwen3_export("readout_scaled_gated", 2)] {
            let imported = import_language_model(&dir, 6, 12).expect("tiny export");
            std::fs::remove_dir_all(dir).expect("remove the tiny export");
            let native = split_sites(&imported.program).expect("split sites");
            let layers = layer_nodes(&native, 2).expect("layer nodes");
            let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
            let sequences: Vec<Vec<u32>> = tokens.chunks(12).take(2).map(<[u32]>::to_vec).collect();
            let artifact = library_mdl::explanation(&native, &layers).expect("library").artifact;
            let host = Device::host();
            let library = Library::new(&host, &host, &native, &layers, &artifact, 1 << 30, 64).expect("library on the host");
            let functions = library.functions();
            let all: Vec<usize> = (0..12).collect();
            let largest = |a: &ndarray::Array2<f64>, b: &ndarray::Array2<f64>| (a - b).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let none = std::collections::BTreeMap::new();
            let clean = library.edited(&sequences[..1], &[Edit { sequence: 0, scale: vec![], rows: all.clone() }], true).expect("clean");
            let base = library.run(&sequences[..1], &none).expect("run");
            assert!(largest(&clean.last, &base.last) < 1e-12, "an unedited copy is not the run");
            let alpha = 0.37;
            for (h, (layer, head)) in library.heads().into_iter().enumerate() {
                let f = functions.iter().position(|g| g.name == format!("L{layer}.H{head}")).expect("a head's column");
                let scaled = library.edited(&sequences[..1], &[Edit { sequence: 0, scale: vec![(f, alpha)], rows: all.clone() }], false).expect("edited");
                let direct = library.run(&sequences[..1], &[(h, &base.reads[h] * alpha)].into()).expect("run");
                assert!(largest(&scaled.last, &direct.last) < 1e-10, "head {h}: {}", largest(&scaled.last, &direct.last));
                assert!(largest(&scaled.last, &base.last) > 1e-8, "head {h}: the edit changed nothing");
            }
            let activity = clean.activity.expect("activity");
            for (b, f) in [(0, 3), (1, 5)] {
                let (first, _) = functions.iter().enumerate().find(|(_, g)| g.layer == b && matches!(g.kind, super::Kind::Mlp)).expect("an MLP function");
                let count = functions.iter().filter(|g| g.layer == b && matches!(g.kind, super::Kind::Mlp)).count();
                let mut values = activity.slice(ndarray::s![.., first..first + count]).to_owned();
                values.column_mut(f).mapv_inplace(|v| v * alpha);
                let direct = library.run_with(&sequences[..1], &none, &[(b, values)].into()).expect("run");
                let scaled = library.edited(&sequences[..1], &[Edit { sequence: 0, scale: vec![(first + f, alpha)], rows: all.clone() }], false).expect("edited");
                assert!(largest(&scaled.last, &direct.last) < 1e-10, "MLP {b} function {f}: {}", largest(&scaled.last, &direct.last));
            }
            // Two copies with different edits in one call equal their separate calls.
            let edits = [Edit { sequence: 1, scale: vec![(0, 0.0)], rows: vec![3, 11] }, Edit { sequence: 0, scale: vec![(functions.len() - 1, 2.0)], rows: vec![7] }];
            let joint = library.edited(&sequences, &edits, true).expect("joint");
            let one = library.edited(&sequences, &edits[..1], true).expect("first");
            let two = library.edited(&sequences, &edits[1..], true).expect("second");
            let stacked = ndarray::concatenate(ndarray::Axis(0), &[one.last.view(), two.last.view()]).expect("stack");
            assert!(largest(&joint.last, &stacked) < 1e-12);
            let stacked = ndarray::concatenate(ndarray::Axis(0), &[one.activity.as_ref().expect("a").view(), two.activity.as_ref().expect("a").view()]).expect("stack");
            assert!(largest(joint.activity.as_ref().expect("a"), &stacked) < 1e-12);
        }
    }

    /// Patching every function with its own write changes nothing, and patching a head's write
    /// with the counterfactual's equals the run with that head's read replaced by the
    /// counterfactual's.
    #[test]
    fn patching_a_head_matches_replacing_its_read() {
        for dir in [tiny_export("readout_patch_gelu", 2), tiny_qwen3_export("readout_patch_gated", 2)] {
            let imported = import_language_model(&dir, 6, 12).expect("tiny export");
            std::fs::remove_dir_all(dir).expect("remove the tiny export");
            let native = split_sites(&imported.program).expect("split sites");
            let layers = layer_nodes(&native, 2).expect("layer nodes");
            let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
            let (clean, other) = (tokens[..12].to_vec(), tokens[12..24].to_vec());
            let artifact = library_mdl::explanation(&native, &layers).expect("library").artifact;
            let host = Device::host();
            let library = Library::new(&host, &host, &native, &layers, &artifact, 1 << 30, 64).expect("library on the host");
            let same = library.patch_effects(&[(clean.clone(), clean.clone())], &[(1, 2)], 7).expect("patch");
            assert!(same.iter().all(|v| v.abs() < 1e-12), "self-patching changed the metric");
            let effects = library.patch_effects(&[(clean.clone(), other.clone())], &[(1, 2)], 7).expect("patch");
            let none = std::collections::BTreeMap::new();
            let (base, counter) = (library.run(&[clean.clone()], &none).expect("run"), library.run(&[other], &none).expect("run"));
            let difference = |last: &ndarray::Array2<f64>| {
                let log_p = library.log_probabilities(last).expect("log probabilities");
                log_p[[11, 1]] - log_p[[11, 2]]
            };
            let functions = library.functions();
            for (h, (layer, head)) in library.heads().into_iter().enumerate() {
                let column = functions.iter().position(|f| f.name == format!("L{layer}.H{head}")).expect("a head's column");
                let replaced = library.run(&[clean.clone()], &[(h, counter.reads[h].clone())].into()).expect("run");
                let expected = difference(&replaced.last) - difference(&base.last);
                let gap = (effects[[column, 0]] - expected).abs();
                assert!(gap < 1e-9 * (1.0 + expected.abs()), "head {h}: patched {} against {expected}", effects[[column, 0]]);
            }
        }
    }

}

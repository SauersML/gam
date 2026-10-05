//! A readable account of a library explanation (`library_mdl`, #2951): for every surviving
//! function, where and how strongly it acts on held-out text, what it writes and reads in token
//! terms, and how the functions feed one another.
//!
//! # Quantities
//!
//! An MLP function `i` of layer `l` computes `a_i = φ(g_i·x̂ + c_i)` (GELU) or
//! `a_i = φ(g_i·x̂ + c_i) (w_i·x̂)` (SwiGLU) on its layer's normed stream `x̂ = γ ⊙ x / r`,
//! `r = √(mean(x²) + ε)` the RMS of the residual stream `x` at that read, and writes `a_i u_i`.
//! It is *active* at a token when its gate pre-activation `g_i·x̂ + c_i` is positive (the law's
//! linear half; on the other half GELU and SiLU are bounded by their minimum, about `−0.17` and
//! `−0.28`). A head `h` of layer `l` writes `W_O,h z_h`, `z_h` its attention read and `W_O,h` the
//! native output projection's columns of the head; a head computes at every token, so its
//! frequency is one.
//!
//! * Frequency: the fraction of held-out tokens at which the function is active. Mean output when
//!   active: the mean of `‖a_i u_i‖ = |a_i| ‖u_i‖` over those tokens (heads: of `‖W_O,h z_h‖` over all
//!   tokens). Usage: the mean output norm over all held-out tokens, which ranks the functions.
//! * Contexts: the held-out tokens with the largest `a_i` (heads: the largest `‖W_O,h z_h‖`, with
//!   the position the head attends to most there).
//! * Writes: the output direction through the final norm's gain and the unembedding,
//!   `W_U (γ_f ⊙ u_i)`, centred over the vocabulary (a softmax-invariant shift); the most
//!   promoted and most suppressed tokens. The final norm's division by the stream's RMS is a
//!   positive scale per token and leaves the order unchanged.
//! * Reads: the gate direction in residual coordinates `g̃_i = γ ⊙ g_i` against each token's
//!   embedding at that read, `g̃_i·e_t / √(mean(e_t²) + ε)`: the tokens whose embedding alone opens
//!   the gate most.
//! * Wiring: the root-mean-square, over held-out tokens, of writer `j`'s direct contribution to
//!   reader `i`'s input. For an MLP reader the input is its gate pre-activation, so an MLP writer
//!   contributes `(g̃_i·u_j) a_j / r` and the weight is `|g̃_i·u_j| √E[a_j² / r²]` (the alignment
//!   `|g̃_i·u_j|` times how strongly and how often `j` is active); a head writer contributes
//!   `g̃_iᵀ W_O,j z_j / r`, weight `√(g̃_iᵀ W_O,j C_j W_O,jᵀ g̃_i)` with `C_j = E[z_j z_jᵀ / r²]`. For a
//!   head reader the input is its value vector `V_h x̂`, and the weight is the RMS norm of the
//!   change of that vector, the same expressions with the rows of `V_h Γ` in place of `g̃_i`. Only
//!   writers before the read are wired (direct paths through the residual stream).
//! * Attention (heads): the attention mass by query-key offset in binary orders of magnitude
//!   (offset 0, 1, 2–3, 4–7, …), the source tokens receiving the most mass, and the OV map's
//!   largest token-to-token entries: source token `s` (its embedding through the head's read norm
//!   and value map) to output token `o` through `W_O,h`, the final norm's gain and the
//!   unembedding, centred over outputs, the largest per source, sources restricted to the tokens
//!   of the held-out text. `self_top1` is the fraction of those sources whose largest output is
//!   the source token itself (copying).
//!
//! The model runs on its device in that device's arithmetic; the vocabulary-wide searches (top
//! tokens, OV entries) and the wiring products run on the products device; every reported token
//! score is recomputed in float64 on the host from the selected tokens.
use crate::{
    artifact::Artifact,
    artifact_device::mapped_inlined_observed,
    device_program::DeviceProgram,
    library_mdl::sequence_family,
    operator_program::{Node, OperatorProgram, Rotary, Rule},
    resident_causal_fit::fixed_head_target::Head,
    run_check::LayerNodes,
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
    /// The functions of largest RelP importance (background functions apart) whose complete
    /// wiring among themselves is reported.
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

#[derive(Clone, Debug, Serialize)]
pub struct Edge {
    pub from: usize,
    pub weight: f64,
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
    pub frequency: f64,
    pub mean_output_active: f64,
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
    /// Active on every held-out token.
    pub background: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Readout {
    pub held_out_tokens: usize,
    pub functions: Vec<Function>,
    pub removed: Vec<String>,
    /// Indices into `functions` of the most important functions, descending, and `wiring[i][j]` the weight from
    /// core function `j` to core function `i` (zero when `j` does not precede `i`'s read).
    pub core: Vec<usize>,
    pub wiring: Vec<Vec<f64>>,
    /// Per held-out token, the model's predicted token; the mean centred logit `m` of it, the
    /// mean sum of the functions' attributions, and the mean participation number
    /// `(Σ_i |A_i|)² / Σ_i A_i²` (the effective number of functions the prediction rests on).
    pub predicted: Vec<u32>,
    pub mean_logit: f64,
    pub mean_attributed: f64,
    pub participation: f64,
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
    /// A gated MLP's up map: its rule node and matrix.
    up: Option<(usize, Array2<f64>)>,
}

struct HeadBlock {
    layer: usize,
    head: usize,
    call: usize,
    query: usize,
    key: usize,
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
    let (gate_active, up) = match &rule.nodes[activation] {
        Node::Pointwise { .. } => (activation, None),
        Node::Hadamard { left, right } => {
            let law = [*left, *right].into_iter().find(|n| matches!(rule.nodes[*n], Node::Pointwise { .. })).ok_or("a gated MLP without a law")?;
            let up = if law == *left { *right } else { *left };
            (law, Some((up, operator_of(program, rule, up)?.0)))
        }
        other => return Err(format!("{}: activations are {other:?}", rule.name)),
    };
    let Node::Pointwise { input: gate_pre, .. } = rule.nodes[gate_active] else { return Err("no gate".into()) };
    let (gate, bias) = operator_of(program, rule, gate_pre)?;
    let bias = match bias {
        Some(op) => program.operators[op].matrix().column(0).to_owned(),
        None => Array1::zeros(gate.nrows()),
    };
    Ok(MlpBlock { layer, call, activation, gate_pre, gate, bias, out: program.operators[out].matrix(), up })
}

fn head_block(program: &OperatorProgram, rule: &Rule, layer: usize, head: usize, call: usize, output: Array2<f64>) -> Result<HeadBlock, String> {
    let Node::Attend { query, key, value, scale, rotary, causal } = rule.nodes[rule.output] else {
        return Err(format!("{}: the output is not an attention", rule.name));
    };
    let (value, _) = operator_of(program, rule, value)?;
    Ok(HeadBlock { layer, head, call, query, key, value, output, scale: scale.value(), rotary, causal })
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
    active: f64,
    active_abs: f64,
    abs: f64,
    /// Per site from the writer's first visible one, `Σ a² / r²`.
    second: Vec<f64>,
    top: Best,
    relp: Relp,
}

/// A function's RelP attributions: `Σ |A|`, `Σ A`, and the rows of the largest and of the most
/// negative.
#[derive(Clone, Default)]
struct Relp {
    absolute: f64,
    signed: f64,
    supports: Best,
    opposes: Best,
}

impl Relp {
    fn add(&mut self, k: usize, attribution: ArrayView1<f64>, first_row: usize) {
        for (r, a) in attribution.iter().enumerate() {
            self.absolute += a.abs();
            self.signed += a;
            self.supports.offer(k, *a, first_row + r);
            self.opposes.offer(k, -*a, first_row + r);
        }
    }
}

struct HeadStat {
    output: f64,
    /// Per site from the writer's first visible one, `Σ z zᵀ / r²`.
    second: Vec<Array2<f64>>,
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

/// Per row `t` of one sequence, the softmax of `scale q_t·k_s` over `s ≤ t` (all `s` when not
/// causal).
fn attention_rows(query: &Array2<f64>, key: &Array2<f64>, scale: f64, causal: bool) -> Array2<f64> {
    let mut weights = query.dot(&key.t());
    for (t, mut row) in weights.outer_iter_mut().enumerate() {
        let visible = if causal { t + 1 } else { row.len() };
        let max = row.iter().take(visible).fold(f64::NEG_INFINITY, |m, v| m.max(*v * scale));
        let mut total = 0.0;
        for (s, w) in row.iter_mut().enumerate() {
            *w = if s < visible { (*w * scale - max).exp() } else { 0.0 };
            total += *w;
        }
        row.mapv_inplace(|w| w / total);
    }
    weights
}

/// Rows at positions `0, 1, …` rotated to their position.
fn rotated(values: Array2<f64>, rotary: Option<Rotary>) -> Array2<f64> {
    let mut out = values.as_standard_layout().to_owned();
    if let Some(rotary) = rotary {
        for (t, mut row) in out.outer_iter_mut().enumerate() {
            if let Some(slice) = row.as_slice_mut() {
                rotary.rotate(slice, None, t as u32);
            }
        }
    }
    out
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

/// `rows · tableᵀ` on `device`, downloaded.
fn products(device: &Device, rows: &Array2<f64>, table: &Array2<f64>) -> Result<Array2<f64>, String> {
    let (a, b) = (device.upload(rows.view()).map_err(error)?, device.upload(table.view()).map_err(error)?);
    let mut c = device.zeros(rows.nrows(), table.nrows()).map_err(error)?;
    device.gemm(&mut c, 1.0, &a, Op::N, &b, Op::T, 0.0, arithmetic(device)).map_err(error)?;
    device.download(&c).map_err(error)
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

// ------------------------------------------------------------------------------ the read-out

/// The read-out (module note) of `artifact`, a library explanation of the split native program
/// `native` (`run_check::split_sites`) with its `layers`, on the held-out `sequences` (of equal
/// length). The model runs on `model`; vocabulary-wide searches and wiring products on `wide`.
pub fn read_out(model: &Device, wide: &Device, native: &OperatorProgram, layers: &[LayerNodes], artifact: &Artifact, sequences: &[Vec<u32>], settings: &Settings) -> Result<Readout, String> {
    if settings.batch_sequences == 0 || settings.tile_rows == 0 || settings.numeric_bytes == 0 || sequences.is_empty() {
        return Err("invalid read-out settings or no held-out sequences".into());
    }
    let length = sequences[0].len();
    if length == 0 || sequences.iter().any(|s| s.len() != length) {
        return Err("held-out sequences must be nonempty and of equal length".into());
    }
    let program = &artifact.program;
    // Reads in depth order: site 2l is layer l's attention read, 2l + 1 its MLP read.
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
    // The final norm's gain with the unembedding, and the input embedding (vocabulary × width).
    let Head { hidden, embedding: mut unembedding, .. } = Head::of(native)?;
    let final_gain = site(native, hidden)?.gain;
    unembedding.axis_iter_mut(Axis(0)).for_each(|mut row| row *= &final_gain);
    let unembedding_mean = unembedding.mean_axis(Axis(0)).ok_or("an empty vocabulary")?;
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

    // Observed values: each site's stream, each MLP's activations and gate pre-activations, each
    // head's query, key and read.
    let place = |n: usize| artifact.place(n).ok_or_else(|| format!("the artifact does not hold native node {n}"));
    let final_site = site(native, hidden)?;
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
        paths.extend([vec![b.call, b.query], vec![b.call, b.key], vec![b.call]]);
    }
    let (flat, _, observed) = mapped_inlined_observed(program, &paths)?;
    let prefix = Head::of(&flat)?.prefix(&flat);
    drop(flat);
    let mut device_program = DeviceProgram::compile_values_bounded(model, &prefix, settings.numeric_bytes)?;
    drop(prefix);
    device_program.set_arithmetic(arithmetic(model));

    let k = settings.contexts;
    let mlp_first = |l: usize| 2 * l + 2;
    let head_first = |l: usize| 2 * l + 1;
    let mut mlp_stats: Vec<Vec<MlpStat>> = mlps
        .iter()
        .map(|b| vec![MlpStat { second: vec![0.0; sites.len().saturating_sub(mlp_first(b.layer))], ..MlpStat::default() }; b.gate.nrows()])
        .collect();
    let mut head_stats: Vec<HeadStat> = heads
        .iter()
        .map(|b| HeadStat {
            output: 0.0,
            second: vec![Array2::zeros((b.value.nrows(), b.value.nrows())); sites.len().saturating_sub(head_first(b.layer))],
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
    let unembedding_table = wide.upload(unembedding.view()).map_err(error)?;
    let inverse_rms = |x: &Array2<f64>, epsilon: f64| x.map_axis(Axis(1), |x| 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + epsilon).sqrt());
    // Per head and sequence of a batch, the attention weights.
    let attend = |values: &[[Array2<f64>; 3]], count: usize| -> Vec<Vec<Array2<f64>>> {
        heads
            .par_iter()
            .zip(values.par_iter())
            .map(|(block, [q, key, _])| {
                (0..count)
                    .map(|s| {
                        let span = s * length..(s + 1) * length;
                        let query = rotated(q.slice(s![span.clone(), ..]).to_owned(), block.rotary);
                        let keys = rotated(key.slice(s![span, ..]).to_owned(), block.rotary);
                        attention_rows(&query, &keys, block.scale, block.causal)
                    })
                    .collect()
            })
            .collect()
    };
    let mut predicted_all: Vec<u32> = Vec::with_capacity(sequences.len() * length);
    let (mut logit_sum, mut attributed_sum, mut participation_sum) = (0.0, 0.0, 0.0);
    for (chunk, batch) in sequences.chunks(settings.batch_sequences).enumerate() {
        let first_row = chunk * settings.batch_sequences * length;
        let family = sequence_family(&batch.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
        let trace = device_program.forward(&family)?;
        let get = |i: usize| -> Result<Array2<f64>, String> { model.download(trace.value(observed[i])?).map_err(error) };
        let rows = family.rows;
        let inverse: Vec<Array1<f64>> = sites.iter().enumerate().map(|(i, s)| Ok(inverse_rms(&get(i)?, s.epsilon))).collect::<Result<_, String>>()?;
        let last = get(sites.len())?;
        let inverse_final = inverse_rms(&last, final_site.epsilon);
        let predicted: Vec<usize> = extreme_columns(wide, &unembedding_table, &last, 1, 1.0, settings.tile_rows)?.into_iter().map(|c| c[0]).collect();
        // The gradient of the predicted token's centred logit in the final stream, the final
        // norm's denominator frozen.
        let mut g = Array2::from_shape_fn((rows, width), |(r, c)| (unembedding[[predicted[r], c]] - unembedding_mean[c]) * inverse_final[r]);
        let logits: f64 = (0..rows).map(|r| g.row(r).dot(&last.row(r))).sum();
        let mlp_values: Vec<(Array2<f64>, Array2<f64>, Option<Array2<f64>>)> = mlps
            .iter()
            .enumerate()
            .map(|(b, block)| {
                let i = mlp_paths[b];
                Ok((get(i)?, get(i + 1)?, if block.up.is_some() { Some(get(i + 2)?) } else { None }))
            })
            .collect::<Result<_, String>>()?;
        for (b, block) in mlps.iter().enumerate() {
            let (activations, pre, _) = &mlp_values[b];
            let visible = &inverse[mlp_first(block.layer).min(sites.len())..];
            mlp_stats[b].par_iter_mut().enumerate().for_each(|(i, stat)| {
                for r in 0..rows {
                    let a = activations[[r, i]];
                    if pre[[r, i]] > 0.0 {
                        stat.active += 1.0;
                        stat.active_abs += a.abs();
                    }
                    stat.abs += a.abs();
                    for (m, inv) in stat.second.iter_mut().zip(visible) {
                        *m += a * a * inv[r] * inv[r];
                    }
                    stat.top.offer(k, a, first_row + r);
                }
            });
        }
        let values: Vec<[Array2<f64>; 3]> =
            (0..heads.len()).map(|h| Ok([get(head_paths + 3 * h)?, get(head_paths + 3 * h + 1)?, get(head_paths + 3 * h + 2)?])).collect::<Result<_, String>>()?;
        let weights = attend(&values, batch.len());
        head_stats.par_iter_mut().zip(heads.par_iter()).zip(values.par_iter()).zip(grams.par_iter()).zip(weights.par_iter()).for_each(
            |((((stat, block), [_, _, z]), gram), weights)| {
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
                let norms = (z.dot(gram) * z).sum_axis(Axis(1));
                for r in 0..rows {
                    let norm = norms[r].max(0.0).sqrt();
                    stat.output += norm;
                    stat.top.offer(k, norm, first_row + r);
                }
                for (second, inv) in stat.second.iter_mut().zip(&inverse[head_first(block.layer).min(sites.len())..]) {
                    let scaled = z * &inv.view().insert_axis(Axis(1));
                    *second += &scaled.t().dot(&scaled);
                }
            },
        );
        // RelP: the reverse pass of the network with every norm's denominator, every law's gate
        // factor `φ(x)/x` and every attention pattern frozen, and half of the gradient through
        // each factor of a product (module note).
        let (mut absolute, mut square, mut signed) = (Array1::<f64>::zeros(rows), Array1::<f64>::zeros(rows), Array1::<f64>::zeros(rows));
        let mut tally = |attribution: ArrayView1<f64>, r: usize| {
            absolute[r] += attribution[r].abs();
            square[r] += attribution[r] * attribution[r];
            signed[r] += attribution[r];
        };
        for l in (0..layers.len()).rev() {
            for (b, block) in mlps.iter().enumerate().filter(|(_, b)| b.layer == l) {
                let (activations, pre, up) = &mlp_values[b];
                let d = g.dot(&block.out);
                let attribution = activations * &d;
                mlp_stats[b].par_iter_mut().enumerate().for_each(|(i, stat)| stat.relp.add(k, attribution.column(i), first_row));
                for column in attribution.columns() {
                    (0..rows).for_each(|r| tally(column, r));
                }
                let through = |factor: &Array2<f64>, share: f64| {
                    let mut out = &d * activations;
                    ndarray::Zip::from(&mut out).and(factor).for_each(|o, f| *o = if *f == 0.0 { 0.0 } else { share * *o / f });
                    out
                };
                let read = match (up, &block.up) {
                    (Some(up_values), Some((_, up_map))) => through(pre, 0.5).dot(&block.gate) + through(up_values, 0.5).dot(up_map),
                    _ => through(pre, 1.0).dot(&block.gate),
                };
                let (gain, inv) = (&sites[2 * l + 1].gain, &inverse[2 * l + 1]);
                g = g + read * &gain.view().insert_axis(Axis(0)) * &inv.view().insert_axis(Axis(1));
            }
            let mut delta = Array2::<f64>::zeros((rows, width));
            for (h, block) in heads.iter().enumerate().filter(|(_, b)| b.layer == l) {
                let z = &values[h][2];
                let gz = g.dot(&block.output);
                let attribution = (z * &gz).sum_axis(Axis(1));
                head_stats[h].relp.add(k, attribution.view(), first_row);
                (0..rows).for_each(|r| tally(attribution.view(), r));
                let mut gv = Array2::<f64>::zeros(gz.dim());
                for (s, weights) in weights[h].iter().enumerate() {
                    let span = s * length..(s + 1) * length;
                    gv.slice_mut(s![span.clone(), ..]).assign(&weights.t().dot(&gz.slice(s![span, ..])));
                }
                delta = delta + gv.dot(&block.value);
            }
            let (gain, inv) = (&sites[2 * l].gain, &inverse[2 * l]);
            g = g + delta * &gain.view().insert_axis(Axis(0)) * &inv.view().insert_axis(Axis(1));
        }
        logit_sum += logits;
        attributed_sum += signed.sum();
        participation_sum += absolute.iter().zip(&square).map(|(a, q)| if *q > 0.0 { a * a / q } else { 0.0 }).sum::<f64>();
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
            let family = sequence_family(&batch.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
            let trace = device_program.forward(&family)?;
            let values: Vec<[Array2<f64>; 3]> = (0..heads.len())
                .map(|h| {
                    let get = |i: usize| -> Result<Array2<f64>, String> { model.download(trace.value(observed[i])?).map_err(error) };
                    Ok([get(head_paths + 3 * h)?, get(head_paths + 3 * h + 1)?, Array2::zeros((0, 0))])
                })
                .collect::<Result<_, String>>()?;
            for (stat, weights) in head_stats.iter_mut().zip(attend(&values, batch.len())) {
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
    let relp_fields = |relp: &Relp, sign: f64| {
        let rows = |best: &Best, sign: f64| best.entries.iter().map(|(v, id)| context(sign * v, *id, None)).collect::<Vec<_>>();
        (relp.absolute / total, sign * relp.signed / total, rows(&relp.supports, 1.0), rows(&relp.opposes, -1.0), majority(&relp.supports), majority(&relp.opposes))
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
            let (importance, attribution, supports, opposes, supports_token, opposes_token) = relp_fields(&stat.relp, 1.0);
            functions.push(Function {
                importance,
                attribution,
                supports,
                opposes,
                supports_token,
                opposes_token,
                background: false,
                name,
                layer: l,
                kind: Kind::Head,
                frequency: 1.0,
                mean_output_active: stat.output / total,
                usage: stat.output / total,
                contexts: stat.top.entries.iter().map(|(v, id)| context(*v, *id, Some(stat.attended[*id]))).collect(),
                promoted: Vec::new(),
                suppressed: Vec::new(),
                reads: Vec::new(),
                inputs: Vec::new(),
                attention: Some(Attention {
                    offsets: (0..stat.offsets.len()).map(|b| if b == 0 { [0, 0] } else { [1 << (b - 1), (1 << b) - 1] }).collect(),
                    offset_mass: stat.offsets.iter().map(|m| m / total).collect(),
                    sources: sources.into_iter().map(|(token, score)| TokenScore { token, score }).collect(),
                    ov: Vec::new(),
                    self_top1: 0.0,
                    previous: stat.offsets.get(1).copied().unwrap_or(0.0) / (total - sequences.len() as f64).max(1.0),
                    induction: stat.induction / repeats.max(1.0),
                    duplicate: stat.duplicate / repeats.max(1.0),
                }),
            });
        }
        for (b, block) in mlps.iter().enumerate().filter(|(_, b)| b.layer == l) {
            for i in 0..block.gate.nrows() {
                let name = format!("L{l}.M{i}");
                let out_norm = block.out.column(i).dot(&block.out.column(i)).sqrt();
                if out_norm == 0.0 || (block.gate.row(i).iter().all(|v| *v == 0.0) && block.bias[i] == 0.0) {
                    removed.push(name);
                    continue;
                }
                let stat = &mlp_stats[b][i];
                mlp_index[b][i] = Some(functions.len());
                let (importance, attribution, supports, opposes, supports_token, opposes_token) = relp_fields(&stat.relp, 1.0);
                functions.push(Function {
                    importance,
                    attribution,
                    supports,
                    opposes,
                    supports_token,
                    opposes_token,
                    background: stat.active == total,
                    name,
                    layer: l,
                    kind: Kind::Mlp,
                    frequency: stat.active / total,
                    mean_output_active: if stat.active > 0.0 { stat.active_abs / stat.active * out_norm } else { 0.0 },
                    usage: stat.abs / total * out_norm,
                    contexts: stat.top.entries.iter().map(|(v, id)| context(*v, *id, None)).collect(),
                    promoted: Vec::new(),
                    suppressed: Vec::new(),
                    reads: Vec::new(),
                    inputs: Vec::new(),
                    attention: None,
                });
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
        let promoted = extreme_columns(wide, &unembedding_table, &writes, settings.tokens, 1.0, settings.tile_rows)?;
        let suppressed = extreme_columns(wide, &unembedding_table, &writes, settings.tokens, -1.0, settings.tile_rows)?;
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
            wide.gemm(&mut out, 1.0, &unembedding_table, Op::N, &columns, Op::N, 0.0, arithmetic(wide)).map_err(error)?;
            out
        };
        let best = extreme_columns(wide, &resident, &sources, 1, 1.0, settings.tile_rows)?;
        let centre = output.dot(&unembedding_mean);
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

    // Wiring: per read, every surviving reader against every surviving writer before it.
    // The core: the functions of largest RelP importance, background functions apart.
    let mut order: Vec<usize> = (0..functions.len()).filter(|i| !functions[*i].background).collect();
    order.sort_by(|a, b| functions[*b].importance.total_cmp(&functions[*a].importance));
    order.truncate(settings.core);
    let mut core_of: Vec<Option<usize>> = vec![None; functions.len()];
    order.iter().enumerate().for_each(|(p, f)| core_of[*f] = Some(p));
    let mut wiring = vec![vec![0.0; order.len()]; order.len()];
    for (si, read_site) in sites.iter().enumerate() {
        let layer = si / 2;
        // Reader rows and, per reader, its function and row range.
        let mut rows: Vec<Array1<f64>> = Vec::new();
        let mut readers: Vec<(usize, std::ops::Range<usize>)> = Vec::new();
        if si % 2 == 1 {
            for (b, block) in mlps.iter().enumerate().filter(|(_, b)| b.layer == layer) {
                for (i, index) in mlp_index[b].iter().enumerate() {
                    let Some(index) = index else { continue };
                    readers.push((*index, rows.len()..rows.len() + 1));
                    rows.push(&block.gate.row(i) * &read_site.gain);
                }
            }
        } else {
            for (h, block) in heads.iter().enumerate().filter(|(_, b)| b.layer == layer) {
                let Some(index) = head_index[h] else { continue };
                readers.push((index, rows.len()..rows.len() + block.value.nrows()));
                rows.extend(block.value.outer_iter().map(|v| &v * &read_site.gain));
            }
        }
        if readers.is_empty() {
            continue;
        }
        let reader_rows = Array2::from_shape_fn((rows.len(), width), |(r, c)| rows[r][c]);
        let mut best: Vec<Best> = vec![Best::default(); readers.len()];
        // Every reader against a block of writers, `weight(reader, c)` the weight from writer `c`
        // (function `writers[c]`).
        let mut wire = |writers: &[usize], weight: &(dyn Fn(&std::ops::Range<usize>, usize) -> f64 + Sync)| {
            best.par_iter_mut().zip(readers.par_iter()).for_each(|(best, (_, range))| {
                for (c, writer) in writers.iter().enumerate() {
                    best.offer(settings.edges, weight(range, c), *writer);
                }
            });
            for (reader, range) in &readers {
                let Some(i) = core_of[*reader] else { continue };
                for (c, writer) in writers.iter().enumerate() {
                    if let Some(j) = core_of[*writer] {
                        wiring[i][j] = weight(range, c);
                    }
                }
            }
        };
        for (b, block) in mlps.iter().enumerate().filter(|(_, b)| mlp_first(b.layer) <= si) {
            let kept: Vec<usize> = (0..block.gate.nrows()).filter(|i| mlp_index[b][*i].is_some()).collect();
            if kept.is_empty() {
                continue;
            }
            let writers: Vec<usize> = kept.iter().filter_map(|i| mlp_index[b][*i]).collect();
            let writes = Array2::from_shape_fn((kept.len(), block.out.nrows()), |(r, c)| block.out[[c, kept[r]]]);
            let p = products(wide, &reader_rows, &writes)?;
            let strength: Vec<f64> = kept.iter().map(|i| (mlp_stats[b][*i].second[si - mlp_first(block.layer)] / total).sqrt()).collect();
            wire(&writers, &|range, c| range.clone().map(|r| p[[r, c]] * p[[r, c]]).sum::<f64>().sqrt() * strength[c]);
        }
        for (h, block) in heads.iter().enumerate().filter(|(_, b)| head_first(b.layer) <= si) {
            let Some(writer) = head_index[h] else { continue };
            let p = products(wide, &reader_rows, &block.output.t().to_owned())?;
            let second = &head_stats[h].second[si - head_first(block.layer)] / total;
            let quadratic = (p.dot(&second) * &p).sum_axis(Axis(1));
            wire(&[writer], &|range, _| range.clone().map(|r| quadratic[r].max(0.0)).sum::<f64>().sqrt());
        }
        for ((index, _), best) in readers.iter().zip(best) {
            functions[*index].inputs = best.entries.into_iter().map(|(weight, from)| Edge { from, weight }).collect();
        }
    }
    Ok(Readout {
        held_out_tokens: sequences.len() * length,
        functions,
        removed,
        core: order,
        wiring,
        predicted: predicted_all,
        mean_logit: logit_sum / total,
        mean_attributed: attributed_sum / total,
        participation: participation_sum / total,
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

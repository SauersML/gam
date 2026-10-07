//! Interchange experiments of causal abstraction (Geiger et al., JMLR 2025) on the device (#2951).
//!
//! The explanation `P` and the model `M` share their residual streams and block inputs: per block
//! (a layer's attention, then its MLP; `2L` blocks in `L` layers) the stream entering it and the
//! normed stream its projections read. The alignment between the two is the identity, so every
//! experiment is defined identically for both models. An experiment `e` draws a base sequence `x`
//! and a source sequence `x′` and consists of
//!
//! * a hybrid: a set `H` of blocks, `P_e` running `P`'s version of the blocks in `H` and `M`'s of
//!   the others, while `M_e` runs `M` alone. `|H| = k` is drawn uniformly in `1..=2L`, then `H`
//!   uniformly among the sets of that size, so the autonomous `P` (`k = 2L`) and each single
//!   replacement keep probability `1/(2L)` and every combination of blocks is tested, not only
//!   prefixes (exact abstraction under the identity alignment requires every hybrid to match `M`);
//! * and at most one patch, at one position `t₀` of one block `B`, applied by both models: a read
//!   patch of one of `M`'s functions at `B` (the variables: an MLP function's gate pre-activation
//!   `a_i·x̂ + c_i`, with its up pre-activation `b_i·x̂ + e_i` for a gated law; a head's query, key
//!   or value vector, a key or value map its group's query heads share being one variable), or a
//!   joint read patch of several of them at once. A patch replaces each variable's value at row
//!   `t₀` by its value on `x′`, and nothing else: the other functions keep reading the base's
//!   stream, so the experiment intervenes on the explanation's variables alone, the interchange
//!   intervention of causal abstraction.
//!
//! Each model patches its own variables: `M` its native functions, `P` the call sites that replaced
//! them (`Artifact::owners`: a head's or an MLP's rule, or the unit of a body at a call). A
//! function `P` no longer computes has no value to patch in `P`, so only `M_e` changes, and the
//! experiment tests that the function does not matter. The source's value is the same model's own
//! on `x′`: `M`'s for `M_e`, the same hybrid's for `P_e`. The gradient passes through it: the
//! cotangent of a patched entry flows to the source's run on `x′`, the base's own entry there
//! receives none.
//!
//! The data term is `Σ_e Σ_{t ≥ t₀} KL(M_e ‖ P_e)` over the tokens `t` of each base from the
//! experiment's position on (an unpatched experiment's position is 0), in bits ([`evaluate`]),
//! scored through the final head the two models share (compact fixed-head targets). Attention is
//! causal, so before `t₀` a patched run equals the unpatched run of the same hybrid, whose
//! divergence the base's unpatched experiment already scores; a patched experiment therefore
//! shares its base's unpatched hybrid and is scored from `t₀` on.
//!
//! # Execution
//!
//! A run of one sequence through a range of blocks is a lane. Block by block, the lanes that run
//! `P` at that block run as one batch through `P`'s block, and the others through `M`'s
//! ([`DeviceProgram::forward_span`]): no block runs twice and every product spans all experiments
//! at once. A patched experiment's base forks from its unpatched run at the patched block, and its
//! source runs up to that block in the same calls, so a patch takes the source's values from rows
//! of the call it edits ([`Edits`]). The reverse pass runs the blocks backwards, the cotangents of
//! the patched entries moving to the sources' rows.
//!
//! Attention is causal, so a patched or edited path's rows before its position `t₀` equal those of
//! the path it forked from at every block both run alike (the same hybrid, no edit of the other):
//! a lane holding such a path is a suffix lane (`Lane::prefix`). At every block its call takes
//! only its rows from `t₀` on, and its rows before `t₀` are copied from that path's lane after the
//! block; the reverse pass adds their cotangents to that lane's rows before the block's reverse,
//! through which they reach the parameters as the copied rows' own computation would. At an
//! attention block the call runs as segments (`device_attention::Segment`,
//! `DeviceProgram::forward_span_segments`): a suffix lane's queries read the keys and values of its
//! positions before `t₀` from its twin's rows in the same call (`Plan::before`), and their
//! cotangents go to those rows.
//!
//! A block's reverse needs its forward's tape (the block's intermediate values), which holds many
//! times the rows of the stream entering it. A call keeps its tape while the tapes kept so far,
//! with room for the reverse passes' gradients, the cotangents and one block run again, fit in the
//! device's free memory at the pass's start ([`BlockEngine::tape_budget`]); past that point a call
//! keeps only the rows of the stream entering it, and each reverse pass runs the block's forward
//! again from them (the same products on the same values, so the same tape) before reversing it.
//! More experiments then fit in one batch than their tapes would allow.

use crate::{
    artifact::Artifact,
    artifact_device::{mapped_inlined, mapped_inlined_observed},
    device_program::{DeviceProgram, DeviceTrace},
    operator_program::{FamilyInputs, Node, OperatorProgram, SequenceLayout, SlotValues},
    resident_causal_fit::fixed_head_target::{Head, ResidentHead, Target},
    run_check::LayerNodes,
};
use gam_gpu::gpu_error::GpuError;
use gam_gpu::tensor::{Arithmetic, ColumnBlocks, Device, Op, RowNorm, Storage, Tensor};
use gam_runtime::resource::{Governed, MemoryGovernor};
use rand::RngExt;
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, HashMap},
    ops::Range,
    path::PathBuf,
    sync::Arc,
};

fn error(e: impl std::fmt::Display) -> String {
    format!("interchange: {e}")
}

/// One model's sites, checked against its program: the stream entering each block, each block's
/// read, the dense operators that receive gradients, per block, and where it holds each read
/// variable's value.
#[derive(Clone)]
struct Sites {
    entries: Vec<usize>,
    trainable: Vec<Vec<usize>>,
    values: Vec<Value>,
    parts: PartSites,
}

impl Sites {
    fn new(program: &DeviceProgram, flat: &OperatorProgram, entries: Vec<usize>, reads: Vec<usize>, trainable: &[usize], values: Vec<Value>) -> Result<Self, String> {
        let blocks = entries.len();
        let hidden = program.hidden();
        let widths = program.widths();
        if blocks == 0 || blocks % 2 != 0 || reads.len() != blocks || widths.len() != hidden + 1 || flat.nodes.len() <= hidden {
            return Err(error("a model needs an entering stream and a read per block, two blocks per layer, and a program through its hidden node"));
        }
        let width = widths[entries[0]];
        if entries.iter().chain(&reads).chain([&hidden]).any(|n| widths[*n] != width) {
            return Err(error("streams, reads and the hidden node differ in width"));
        }
        let wanted: BTreeSet<usize> = trainable.iter().copied().collect();
        let mut per_block = Vec::with_capacity(blocks);
        for b in 0..blocks {
            let end = if b + 1 < blocks { entries[b + 1] } else { hidden };
            let first = if b == 0 { 0 } else { entries[b] + 1 };
            if entries[b] >= end || !(first..end).contains(&reads[b]) {
                return Err(error(format!("block {b}: its entering stream, read and end are not in order")));
            }
            let floor = if b == 0 { 0 } else { entries[b] };
            let mut used = BTreeSet::new();
            for n in first..=end {
                let node = &flat.nodes[n];
                if node.arguments().iter().any(|a| *a < floor && !matches!(flat.nodes[*a], Node::Feature { .. })) {
                    return Err(error(format!("block {b}: node {n} reads a node before the stream entering the block")));
                }
                used.extend(node.operators().into_iter().filter(|op| wanted.contains(op)));
            }
            per_block.push(used.into_iter().collect());
        }
        if values.iter().any(|v| v.block >= blocks) || values.iter().flat_map(|v| &v.sites).any(|s| s.node >= hidden || widths[s.node] != s.width || s.columns.is_empty() || s.columns.end > s.width) {
            return Err(error("a read variable's value outside the program or its node"));
        }
        Ok(Self { entries, trainable: per_block, values, parts: PartSites::default() })
    }
}

/// One model as the experiments run it: its resident-value program through the final normed
/// stream and its sites.
pub struct Model<'a> {
    pub program: &'a DeviceProgram,
    sites: Arc<Sites>,
    /// `M`'s prefixes kept across scorings ([`PrefixStore`]), for the teacher of an
    /// [`Interchange`] that keeps them.
    prefixes: Option<&'a RefCell<PrefixStore>>,
}

impl<'a> Model<'a> {
    /// `program` is the resident-value program of `flat` through its hidden node (the final normed
    /// stream, `Head::prefix`). Per block (each layer's attention, then its MLP) `entries` holds
    /// the stream entering it and `reads` the node its projections read. `trainable` lists the
    /// dense operators that receive gradients (none for `M`). Each block's nodes must read nothing
    /// before the stream entering it except token features, so that one block runs from that
    /// stream alone. `values` says where the model holds each read variable's value ([`values`]).
    pub fn new(program: &'a DeviceProgram, flat: &OperatorProgram, entries: Vec<usize>, reads: Vec<usize>, trainable: &[usize], values: Vec<Value>) -> Result<Self, String> {
        Ok(Self { program, sites: Arc::new(Sites::new(program, flat, entries, reads, trainable, values)?), prefixes: None })
    }

    fn blocks(&self) -> usize {
        self.sites.entries.len()
    }

    fn width(&self) -> usize {
        self.program.widths()[self.program.hidden()]
    }

    fn entry(&self, b: usize) -> usize {
        self.sites.entries[b]
    }

    /// The last node block `b` runs: the stream entering the next block, or the hidden node.
    fn end(&self, b: usize) -> usize {
        if b + 1 < self.blocks() { self.entry(b + 1) } else { self.program.hidden() }
    }
}

/// The model's forward tangent, for the tests of the reverse pass against it.
#[cfg(test)]
mod model_tangent_tests {
    use super::*;

    impl Model<'_> {
        /// The forward tangent of block `block` on rows `ranges` of the stream's tangent
        /// `stream` (replaced by the tangent of the stream after the block), from its forward's
        /// `tape`, with each operator in `tangents` moving along its entry and `edits`' tangents
        /// ([`Edits::tangent`]); the exact directional derivative the reverse pass is the transpose of.
        pub(crate) fn tangent(&self, block: usize, tape: &DeviceTrace, stream: &mut Tensor, ranges: &[Range<usize>], edits: Option<&Edits>, tangents: &BTreeMap<usize, ndarray::Array2<f64>>) -> Result<(), String> {
            let d = self.program.device();
            let rows: usize = ranges.iter().map(ExactSizeIterator::len).sum();
            let entry = if block == 0 { None } else { Some((self.entry(block), gather(d, stream, ranges)?)) };
            let widths = self.program.widths();
            let out = self.program.jvp_span(tape, entry, self.end(block), tangents, self.program.arithmetic(), |n, t, dv| match edits {
                Some(edits) => edits.tangent(d, n, t, dv, (rows, widths[n])),
                None => Ok(()),
            })?;
            let out = match out {
                Some(t) => t,
                None => d.zeros(rows, stream.cols()).map_err(error)?,
            };
            scatter(d, stream, ranges, &out)
        }
    }
}

/// The flat program of `artifact` (`mapped_inlined`), and per block (each layer's attention, then
/// its MLP) its entering stream and its read at the native sites `layers`
/// (`run_check::layer_nodes` of the native program `artifact` was made from): the arguments of
/// [`Model::new`]. `Artifact::native` gives `M`'s.
pub fn sites(artifact: &Artifact, layers: &[LayerNodes]) -> Result<(OperatorProgram, Vec<usize>, Vec<usize>), String> {
    let (flat, entries, reads, _, _, _) = flat_sites(artifact, layers)?;
    Ok((flat, entries, reads))
}

/// [`flat_sites`]' result: the flat program, per block its entering stream and its read, its
/// parts' sites (read and output), its heads' node and the shared sites' nodes.
type FlatSites = (OperatorProgram, Vec<usize>, Vec<usize>, Vec<Option<(usize, usize)>>, Vec<Option<usize>>, BTreeMap<SharedSite, usize>);

/// [`sites`] with, per block, where `artifact` applies edits of parts ([`PartSites`]): for layer
/// `l`'s MLP (block `2l + 1`) its read and the MLP's output (the native `mlp` node or the node
/// that replaced it), none for an attention block or an MLP output the artifact does not hold.
fn flat_sites(artifact: &Artifact, layers: &[LayerNodes]) -> Result<FlatSites, String> {
    let (flat, roots) = mapped_inlined(&artifact.program)?;
    let at = |native: usize| artifact.place(native).map(|n| roots[n]).ok_or_else(|| error(format!("native node {native} is not held")));
    let entries: Vec<usize> = layers.iter().flat_map(|l| [l.stream, l.attended]).map(at).collect::<Result<_, _>>()?;
    let reads: Vec<usize> = layers.iter().flat_map(|l| [l.normed_stream, l.normed]).map(at).collect::<Result<_, _>>()?;
    let blocks = entries.len();
    let parts: Vec<Option<(usize, usize)>> = layers
        .iter()
        .enumerate()
        .flat_map(|(l, layer)| {
            let b = 2 * l + 1;
            let end = if b + 1 < blocks { entries[b + 1] } else { flat.nodes.len() };
            let out = at(layer.mlp).ok().filter(|o| *o > reads[b] && *o <= end);
            [None, out.map(|o| (reads[b], o))]
        })
        .collect();
    // Per head (layer-major) the node of its attention output, inside its attention block.
    let heads: Vec<Option<usize>> = layers
        .iter()
        .enumerate()
        .flat_map(|(l, layer)| layer.reads.iter().map(move |r| (l, *r)))
        .map(|(l, r)| at(r).ok().filter(|n| *n > reads[2 * l] && *n < entries[2 * l + 1]))
        .collect();
    // The shared sites the artifact holds, each inside the block that computes it: the stream after
    // each block (the next block's entering stream; after the last, the last residual), each head's
    // output, each attention's and each MLP's output.
    let mut shared = BTreeMap::new();
    let end_of = |b: usize| if b + 1 < blocks { entries[b + 1] } else { flat.nodes.len() };
    for b in 0..blocks {
        let node = if b + 1 < blocks { Some(entries[b + 1]) } else { layers.last().and_then(|l| at(l.residual).ok()) };
        if let Some(node) = node.filter(|n| *n > reads[b] && *n <= end_of(b)) {
            shared.insert(SharedSite::Stream(b), node);
        }
    }
    for (h, node) in heads.iter().enumerate() {
        if let Some(node) = node {
            shared.insert(SharedSite::Head(h), *node);
        }
    }
    for (b, read) in reads.iter().enumerate() {
        shared.insert(SharedSite::Input(b), *read);
    }
    if entries.first().is_some_and(|e| reads.first().is_some_and(|r| e < r)) {
        shared.insert(SharedSite::Embedding, entries[0]);
    }
    for (l, layer) in layers.iter().enumerate() {
        if let Some(node) = at(layer.attention).ok().filter(|n| *n > reads[2 * l] && *n < entries[2 * l + 1]) {
            shared.insert(SharedSite::Attention(l), node);
        }
        if let Some((_, out)) = parts[2 * l + 1] {
            shared.insert(SharedSite::Mlp(l), out);
        }
    }
    Ok((flat, entries, reads, parts, heads, shared))
}

/// The program of the flat program `flat` through its head's hidden node (the final normed
/// stream), which [`Model::new`] runs.
pub fn prefix(flat: &OperatorProgram) -> Result<OperatorProgram, String> {
    Ok(Head::of(flat)?.prefix(flat))
}

/// One read variable at block `block`: rows `rows` of each operator `operator` applied to the
/// block's read (`parts`), whose value is those rows of the operator's output. [`reads`] gives
/// `M`'s, as native operators; a library's own (`library_reads`) are its operators. A gated MLP
/// function reads through two operators, its gate's and its up map's rows.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReadVariable {
    pub block: usize,
    pub parts: Vec<(usize, Range<usize>)>,
}

/// The read variables of a library explanation of `layers` layers (`library_mdl::explanation`), as
/// its own operators: per head its query, key and value read maps, per MLP function its gate row
/// (with its up row for a gated law).
fn library_reads(program: &OperatorProgram, layers: usize) -> Result<Vec<ReadVariable>, String> {
    let named: BTreeMap<&str, usize> = program.operators.iter().enumerate().map(|(i, op)| (op.name.as_str(), i)).collect();
    let mut out = Vec::new();
    for l in 0..layers {
        // A head reads through the query, key and value maps its rule applies to its input (nodes
        // 1, 2 and 3), its own or a function it shares (`library_sharing`).
        for h in 0.. {
            let Some(rule) = program.rules.iter().find(|r| r.name == format!("library.l{l}.h{h}")) else { break };
            for node in 1..4 {
                let op = match rule.nodes.get(node) {
                    Some(Node::Affine { terms, bias: None }) if terms.len() == 1 && terms[0].0 == 0 => terms[0].1,
                    _ => return Err(error(format!("layer {l} head {h}: node {node} is not a read map"))),
                };
                // A key or value map the query heads of one key-value group share is one variable.
                let variable = ReadVariable { block: 2 * l, parts: vec![(op, 0..program.operators[op].rows.width())] };
                if !out.contains(&variable) {
                    out.push(variable);
                }
            }
        }
        // An MLP function reads through its gate, and a gated law's function through its up
        // direction too.
        let gate = *named.get(format!("library.l{l}.mlp.gate").as_str()).ok_or_else(|| error(format!("layer {l}: no MLP gate")))?;
        let up = named.get(format!("library.l{l}.mlp.up").as_str()).copied();
        out.extend((0..program.operators[gate].rows.width()).map(|i| ReadVariable {
            block: 2 * l + 1,
            parts: std::iter::once(gate).chain(up).map(|op| (op, i..i + 1)).collect(),
        }));
    }
    Ok(out)
}

/// A patch: one read variable (an index into the variables), or the distinct read variables
/// `variables` (ascending, all at one block) jointly. A joint read patch shows what single ones
/// miss: many reads that matter little one at a time and much together.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Patch {
    Read { variable: usize },
    Reads { variables: Vec<usize> },
    /// Operations on sites every explanation shares with `M` ([`SiteOp`]), applied verbatim to both
    /// models, drawn as `family` draws them; a swap reads the experiment's source sequence.
    Ops { family: Family, ops: Vec<SiteOp> },
    /// The native weight edit `edit` of the table [`Interchange::set_weight_edits`] holds, at
    /// every row from the experiment's position on, its matrices all in block `block`: each model
    /// adds `ΔW·x` at its uses of each edited matrix ([`MatrixUse`]).
    Weights { edit: usize, block: usize },
}

/// A site every explanation shares with `M`, each model holding it at its own node: the stream
/// after block `b` (block `2l` is layer `l`'s attention, `2l + 1` its MLP), head `h`'s attention
/// output (heads numbered layer by layer), layer `l`'s attention output (the `o` site's), its MLP
/// output, block `b`'s read (its normed input), and the tokens' embeddings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SharedSite {
    Stream(usize),
    Head(usize),
    Attention(usize),
    Mlp(usize),
    /// Block `b`'s read: its normed input (a layer's heads' query, key and value input, or its
    /// MLP's input).
    Input(usize),
    /// The tokens' embeddings, the stream entering block 0 (a swap there swaps the token).
    Embedding,
}

impl SharedSite {
    /// The block whose run computes the site; `heads` gives each head's block.
    fn block(&self, heads: &[usize]) -> Option<usize> {
        match self {
            SharedSite::Stream(b) => Some(*b),
            SharedSite::Head(h) => heads.get(*h).copied(),
            SharedSite::Attention(l) => Some(2 * l),
            SharedSite::Mlp(l) => Some(2 * l + 1),
            SharedSite::Input(b) => Some(*b),
            SharedSite::Embedding => Some(0),
        }
    }
}

/// What an operation does at its site's rows: the value on the donor (the experiment's source)
/// at the same row (`Swap`), the value scaled by `SCALES[i]` (`Scale`, 0 zeroing it), or unit
/// direction `direction` ([`Interchange::set_directions`]) times `SIZES[size]` times the site's
/// typical norm ([`Interchange::measure_typical`]) added (`Push`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Swap,
    Scale(usize),
    Push { direction: usize, size: usize },
    /// A cut of the connection from the site (an attention's or an MLP's output) to the later block
    /// `to`: that block's read sees the site's value on the donor in place of the base's,
    /// `N(s + o(x′) − o(x))` at the row (`s` the stream entering it, `N` its input norm, recomputed),
    /// and nothing else changes.
    Cut { to: usize },
}

/// The candidate weight edits measured per evaluation ([`Interchange::draw_weight_edits`]).
const WEIGHT_SCREEN_CHUNK: usize = 32;

/// The factors a scale operation multiplies its site's value by.
pub const SCALES: [f64; 4] = [0.0, 0.5, 2.0, 3.0];

/// The seeded unit directions a push draws from ([`Interchange::set_directions`]).
pub const DIRECTIONS: usize = 64;

/// The sizes of a pushed direction, in units of its site's typical norm.
pub const SIZES: [f64; 3] = [0.5, 1.0, 2.0];

/// One operation of an experiment: its site, what it does, and its rows (the experiment's
/// position alone, or with `onward` every row from it on).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SiteOp {
    pub site: SharedSite,
    pub operation: Operation,
    pub onward: bool,
}

impl Patch {
    /// The patched variables (none for an edit of a part).
    pub fn variables(&self) -> &[usize] {
        match self {
            Self::Read { variable } => std::slice::from_ref(variable),
            Self::Reads { variables } => variables,
            Self::Ops { .. } | Self::Weights { .. } => &[],
        }
    }
}

/// A part of the explanation that edits act on: an MLP function `relu(g·x + c) u` of MLP block
/// `block` (block `2l + 1` is layer `l`'s MLP), reading the block's read `x` (the normed stream)
/// through `g` (`read`) and `c` (`bias`) and writing `u` (`write`) into the block's output, as
/// `P`'s posterior mean holds it (a transcoder feature, `library_transcoder`).
///
/// A transcoder feature of a library explanation, `relu(g·x + c) u` of MLP block `block` (block
/// `2l + 1` is layer `l`'s MLP), reading the block's read `x` through `g` (`read`) and `c` (`bias`)
/// and writing `u` (`write`), as the oracle's export reads it ([`parts_of`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Part {
    pub block: usize,
    /// Its row in its block's feature operators (the library's function index in the layer).
    pub index: usize,
    pub read: Vec<f64>,
    pub bias: f64,
    pub write: Vec<f64>,
}

/// The parts of a library explanation `program` of `layers` layers (`library_mdl::explanation_with`):
/// per layer whose MLP is transcoder features (`library_transcoder::mlp`: a ReLU gate with its bias,
/// no up map), each feature `relu(g·x + c) u` with a nonzero write, in the layer's order; layers
/// running `M`'s own functions have none.
pub fn parts_of(program: &OperatorProgram, layers: usize) -> Result<Vec<Part>, String> {
    let named: BTreeMap<&str, usize> = program.operators.iter().enumerate().map(|(i, op)| (op.name.as_str(), i)).collect();
    let mut out = Vec::new();
    for l in 0..layers {
        let name = format!("library.l{l}.mlp");
        let at = |part: &str| named.get(format!("{name}.{part}").as_str()).copied();
        let (Some(gate), Some(bias), Some(write)) = (at("gate"), at("gate_bias"), at("out")) else { continue };
        if at("up").is_some() || at("sink").is_none() {
            continue;
        }
        let (gate, bias, write) = (program.operators[gate].matrix(), program.operators[bias].matrix(), program.operators[write].matrix());
        if bias.dim() != (gate.nrows(), 1) || write.dim() != (gate.ncols(), gate.nrows()) {
            return Err(error(format!("layer {l}: a transcoder MLP's gate, bias and write disagree in shape")));
        }
        for i in 0..gate.nrows() {
            let u = write.column(i);
            if u.iter().any(|v| *v != 0.0) {
                out.push(Part { block: 2 * l + 1, index: i, read: gate.row(i).to_vec(), bias: bias[[i, 0]], write: u.to_vec() });
            }
        }
    }
    Ok(out)
}

/// `count` unit directions of width `width` drawn from `seed` (each coordinate standard normal,
/// then normalized): the directions [`Operation::Push`] adds ([`Interchange::set_directions`]),
/// the same for every model.
pub fn seeded_directions(count: usize, width: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
    let mut normal = || -> f64 {
        let (u, v): (f64, f64) = (rng.random::<f64>().max(f64::MIN_POSITIVE), rng.random());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    };
    (0..count)
        .map(|_| {
            let v: Vec<f64> = (0..width).map(|_| normal()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v.into_iter().map(|x| x / norm).collect()
        })
        .collect()
}

/// One experiment's operations of `family` on sequences of `length` tokens and its position
/// ([`Interchange::draw_ops`]), over the shared sites `shared` (ascending), `head_blocks` each
/// head's block, `typical` the sites' typical norms, `directions` pushed directions and `blocks`
/// blocks: what a model outside an [`Interchange`] (the graph checker, `graph`) draws from.
pub fn draw_site_ops(rng: &mut impl RngExt, family: Family, length: usize, shared: &[SharedSite], head_blocks: &[usize], typical: &BTreeMap<SharedSite, f64>, directions: usize, blocks: usize) -> Result<(Patch, usize), String> {
    let sites: Vec<SharedSite> = shared
        .iter()
        .copied()
        .filter(|s| match family {
            Family::Swap | Family::Scale => true,
            Family::Zero => !matches!(s, SharedSite::Stream(_)),
            Family::Push => !matches!(s, SharedSite::Head(_)) && typical.contains_key(s),
            // A cut starts at an attention's or an MLP's output before the last block.
            Family::Cut => matches!(s, SharedSite::Attention(_) | SharedSite::Mlp(_)) && s.block(head_blocks).is_some_and(|b| b + 1 < blocks),
            Family::Read | Family::Weight => false,
        })
        .collect();
    if sites.is_empty() || length == 0 || (family == Family::Push && directions == 0) {
        return Err(error(format!("{family:?}: no site to operate on, or empty sequences")));
    }
    let k = (1usize << rng.random_range(0..=4)).min(sites.len());
    let chosen = hybrid_of(rng, sites.len(), k);
    // A one-token sequence (a real-valued toy's input, `bench/toys_2951`) is edited at its
    // token; the draw is made all the same, so longer sequences draw as before.
    let (position, onward) = match rng.random_range(0..3) {
        _ if length == 1 => (0, true),
        0 => (rng.random_range(1..length), false),
        1 => (rng.random_range(1..length), true),
        _ => (0, true),
    };
    let mut ops = Vec::with_capacity(k);
    let every = family == Family::Cut && rng.random_range(0..2) == 0;
    for (site, _) in sites.iter().zip(&chosen).filter(|(_, c)| **c) {
        let operation = match family {
            Family::Swap => Operation::Swap,
            Family::Zero => Operation::Scale(0),
            Family::Scale => Operation::Scale(rng.random_range(1..SCALES.len())),
            Family::Push => Operation::Push { direction: rng.random_range(0..directions), size: rng.random_range(0..SIZES.len()) },
            // A cut into one later block uniform after its site's, or (half of the experiments)
            // into every later block's read, so the whole rest of the model sees the site's
            // donor value; one cut into each block's read.
            Family::Cut => {
                let from = site.block(&head_blocks).ok_or_else(|| error("a cut at an unknown site"))?;
                let targets: Vec<usize> = if every { (from + 1..blocks).collect() } else { vec![rng.random_range(from + 1..blocks)] };
                for to in targets {
                    if !ops.iter().any(|o: &SiteOp| o.operation == Operation::Cut { to }) {
                        ops.push(SiteOp { site: *site, operation: Operation::Cut { to }, onward });
                    }
                }
                continue;
            }
            Family::Read | Family::Weight => return Err(error(format!("{family:?}: not an operation on a site"))),
        };
        ops.push(SiteOp { site: *site, operation, onward });
    }
    Ok((Patch::Ops { family, ops }, position))
}

/// The families of patched experiments: a read patch, single or joint ([`sample`]), and
/// operations on shared sites ([`Interchange::sample_ops`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    Read,
    /// Operations on shared sites ([`Interchange::sample_ops`]): swaps from a donor, zeroing a
    /// head's, an attention's or an MLP's output, scaling a site, pushing a direction.
    Swap,
    Zero,
    Scale,
    Push,
    Cut,
    /// Native weight edits ([`Patch::Weights`]), drawn from [`Interchange::set_weight_edits`]'s table.
    Weight,
}

/// Where a model applies edits of parts: per block, for an MLP block, the node its parts read
/// (the block's read) and the node their writes add to (the MLP's output), and the parts.
#[derive(Clone, Debug, Default)]
pub struct PartSites {
    nodes: Vec<Option<(usize, usize)>>,
    /// The model applies no edits ([`Interchange::unedited_explanation`]).
    unedited: bool,
    /// Per head ([`Interchange::heads`]) the node holding its attention output, when the model
    /// holds it, and the node's width.
    heads: Vec<Option<(usize, usize)>>,
    /// Per head its block (`2l` for layer `l`'s).
    head_blocks: Vec<usize>,
    /// Per shared site the node holding it and the node's width.
    shared: BTreeMap<SharedSite, (usize, usize)>,
    /// Per block its read and its input norm, when an affine gain of an RMS norm (a cut's).
    reads: Vec<(usize, Option<Norm>)>,
    /// The sites' typical norms ([`Interchange::measure_typical`]) and the unit directions pushed
    /// ([`Interchange::set_directions`]), the same for every model.
    typical: Arc<BTreeMap<SharedSite, f64>>,
    directions: Arc<Vec<Vec<f64>>>,
    /// Per native matrix (by `M`'s operator name) where the model applies an edit of it
    /// ([`matrix_uses`]), and the native weight edits experiments draw from
    /// ([`Interchange::set_weight_edits`]), the same for every model.
    matrices: Arc<BTreeMap<String, Vec<MatrixUse>>>,
    weights: Arc<Vec<crate::weight_edit::Drawn>>,
}

/// Where a model applies a native weight edit of one block of one of `M`'s matrices: the node
/// holding the map's input and the node its output adds into (an affine node), the native block's
/// rows and columns, the columns of the output and of the input they sit at, the nodes' widths,
/// whether the model holds the block transposed (its output then indexed by the native columns),
/// and the factor the edit is divided by (the product of an owner's scalar factors; an owner with
/// a matrix factor cannot take an edit, and an edit of its block fails when applied). The edit's
/// block `ΔB` joins the output as the term `ΔB·x` on the input, at every row the edit covers.
#[derive(Clone, Debug, PartialEq)]
pub struct MatrixUse {
    pub input: usize,
    pub output: usize,
    pub native_rows: Range<usize>,
    pub native_cols: Range<usize>,
    pub out_at: usize,
    pub in_at: usize,
    pub widths: (usize, usize),
    pub transposed: bool,
    pub factor: Result<f64, String>,
}

/// Per native matrix (by name) the uses of its blocks in `artifact`, whose flat program is `flat`
/// (`mapped_inlined`), for the edits of [`Patch::Weights`]: every node of `flat` applying an
/// operator named as one of `M`'s (`native`'s) of the same shape, and every owner's block of
/// `owners` (the artifact's, `Artifact::owners`, or for an explanation `library_vpd` built, which
/// records none, its maps' uses, `library_vpd::uses`): an operator's, at each node of its rule body
/// applying it, or a summed block's, at its use's input and output nodes (`Owner::uses`). A matrix
/// the artifact holds no block of has no use: an edit of it changes `M` alone.
pub fn matrix_uses(native: &OperatorProgram, (artifact, owners): (&Artifact, &[crate::artifact::Owner]), flat: &OperatorProgram) -> Result<BTreeMap<String, Vec<MatrixUse>>, String> {
    let shapes: HashMap<&str, (usize, usize)> = native.operators.iter().map(|op| (op.name.as_str(), (op.rows.width(), op.cols.width()))).collect();
    let interfaces = flat.interfaces().map_err(error)?;
    let width = |n: usize| interfaces.get(n).map(|i| i.width()).ok_or_else(|| error("a matrix use outside the program"));
    let owned: BTreeSet<&str> = owners.iter().map(|o| o.operator.as_str()).collect();
    let mut out: BTreeMap<String, Vec<MatrixUse>> = BTreeMap::new();
    for (n, node) in flat.nodes.iter().enumerate() {
        let Node::Affine { terms, .. } = node else { continue };
        for (input, op) in terms {
            let o = &flat.operators[*op];
            let shape = (o.rows.width(), o.cols.width());
            if owned.contains(o.name.as_str()) || shapes.get(o.name.as_str()) != Some(&shape) {
                continue;
            }
            out.entry(o.name.clone()).or_default().push(MatrixUse { input: *input, output: n, native_rows: 0..shape.0, native_cols: 0..shape.1, out_at: 0, in_at: 0, widths: (width(n)?, width(*input)?), transposed: false, factor: Ok(1.0) });
        }
    }
    let program = &artifact.program;
    let named: BTreeMap<&str, usize> = program.operators.iter().enumerate().map(|(i, op)| (op.name.as_str(), i)).collect();
    // Per owner's use: the observation paths of its input and output nodes.
    let mut found: Vec<(&crate::artifact::Owner, Result<f64, String>)> = Vec::new();
    let mut paths: Vec<Vec<usize>> = Vec::new();
    for owner in owners {
        if !shapes.contains_key(owner.native.as_str()) {
            continue;
        }
        let call = invocation(program, &owner.body, &owner.site)?;
        let body = call.last().and_then(|n| call_rule(program, &call[..call.len() - 1], *n)).ok_or_else(|| error(format!("{}: no rule {}", owner.site, owner.body)))?;
        let factor = crate::weight_edit::scalar_factor(artifact, owner);
        let at = |node: usize| -> Vec<usize> { call.iter().copied().chain([node]).collect() };
        match owner.uses {
            Some((input, output)) => {
                found.push((owner, factor.clone()));
                paths.extend([at(input), at(output)]);
            }
            None => {
                let op = *named.get(owner.operator.as_str()).ok_or_else(|| error(format!("{}: no operator {}", owner.site, owner.operator)))?;
                for (n, node) in program.rules[body].nodes.iter().enumerate() {
                    let Node::Affine { terms, .. } = node else { continue };
                    for (input, _) in terms.iter().filter(|t| t.1 == op) {
                        found.push((owner, factor.clone()));
                        paths.extend([at(*input), at(n)]);
                    }
                }
            }
        }
    }
    if paths.is_empty() {
        return Ok(out);
    }
    let (observing, _, observed) = mapped_inlined_observed(program, &paths)?;
    if observing.nodes.len() != flat.nodes.len() || observed.len() != paths.len() {
        return Err(error("the observed program differs from the flat program"));
    }
    for ((owner, factor), pair) in found.into_iter().zip(observed.chunks(2)) {
        let (input, output) = (pair[0], pair[1]);
        let use_of = MatrixUse { input, output, native_rows: owner.native_rows.clone(), native_cols: owner.native_cols.clone(), out_at: owner.rows.start, in_at: owner.cols.start, widths: (width(output)?, width(input)?), transposed: owner.transposed, factor };
        out.entry(owner.native.clone()).or_default().push(use_of);
    }
    Ok(out)
}

/// A block's input norm `N(s) = γ ⊙ s · (mean(s²) + ε)^{-1/2} + β` of the stream `s` entering it
/// (node `entry`), as the model applies it: an affine gain of an RMS norm.
#[derive(Clone, Debug)]
pub(crate) struct Norm {
    entry: usize,
    epsilon: f64,
    gain: Vec<f64>,
    bias: Option<Vec<f64>>,
}

impl Norm {
    /// The norm of `read` in `flat` (an affine gain of an RMS norm of `entry`), if it is one.
    fn of(flat: &OperatorProgram, read: usize, entry: usize) -> Option<Self> {
        let Node::Affine { terms, bias } = flat.nodes.get(read)? else { return None };
        let [(normed, gain)] = terms[..] else { return None };
        let Node::RmsNorm { input, epsilon } = flat.nodes.get(normed)? else { return None };
        if *input != entry {
            return None;
        }
        let g = flat.operators.get(gain)?.matrix();
        if !g.is_square() || g.indexed_iter().any(|((r, c), v)| r != c && *v != 0.0) {
            return None;
        }
        let bias = match bias {
            Some(b) => Some(flat.operators.get(*b)?.matrix().column(0).to_vec()),
            None => None,
        };
        Some(Self { entry, epsilon: *epsilon, gain: g.diag().to_vec(), bias })
    }

    /// `N(y)` per row of `y`, its tangent along `along` or its pullback of `along` ([`RowNorm`]),
    /// on `d` (`interchange`'s cuts run it in float64, as the host's arithmetic was).
    fn rows(&self, d: &Device, mode: RowNorm, y: &Tensor, along: Option<&Tensor>) -> Result<Tensor, String> {
        let gain = d.upload_vec(1, self.gain.len(), self.gain.clone()).map_err(error)?;
        let bias = self.bias.as_ref().map(|b| d.upload_vec(1, b.len(), b.clone())).transpose().map_err(error)?;
        d.row_norm(mode, y, along, (&gain, bias.as_ref()), self.epsilon).map_err(error)
    }
}

/// A connection cut's record in a plan: its site's values on the base and on the donor at the row
/// (the forward's), their tangents (a tangent pass's), and per reverse pass the cotangent of their
/// difference, which the cut's read leaves and the site's block, later in the same pass, takes
/// (first in, first out: every pass reverses the calls in the same order).
#[derive(Default)]
pub(crate) struct Record {
    values: [Option<RowRef>; 2],
    tangents: [Option<RowRef>; 2],
    carried: std::collections::VecDeque<RowRef>,
}

/// Row `row` of a call's rows on the device, shared by every record that call took: a record
/// keeps no copy of its own, and a cut gathers its rows from each such tensor at once ([`stacked`]).
#[derive(Clone)]
struct RowRef {
    rows: Arc<Tensor>,
    row: usize,
}

pub(crate) type Records = std::rc::Rc<RefCell<BTreeMap<usize, Record>>>;

/// The cuts replacing one read node's rows ([`Edits`]): the block's input norm, per cut its row
/// and cut, the forward's `y = s + δ` (one row per cut), and the reverse's cotangent of the stream
/// entering the block at the cuts' rows, which the block's reverse adds to the entering stream's.
/// Both stay on the device in float64 (`wide`).
struct CutReads {
    norm: Norm,
    rows: Vec<(usize, usize)>,
    kept: RefCell<Option<Tensor>>,
    entering: RefCell<Option<Arc<Tensor>>>,
}

/// The device a cut computes on: float64 where the backend holds it, so `y = s + b − a` and the
/// norm round as the host's float64 arithmetic did, each sum and product once. The Apple GPU holds
/// no float64 tensors (`Device::with_storage`): there the cut computes in its own f32.
fn wide(d: &Device) -> Result<Device, String> {
    match d.with_storage(Storage::F64) {
        Ok(wide) => Ok(wide),
        Err(GpuError::NoDeviceKernel { .. }) => Ok(d.clone()),
        Err(e) => Err(error(e)),
    }
}

/// Per cut of `cuts` (in order) its record's base (`role` 0) or donor (1) value, or with `tangent`
/// their tangents (zero where none was recorded), as rows of a float64 tensor on `wide`.
fn stacked(d: &Device, wide: &Device, recorded: &Records, cuts: &[(usize, usize)], (role, tangent): (usize, bool), width: usize) -> Result<Tensor, String> {
    let records = recorded.borrow();
    let mut refs = Vec::with_capacity(cuts.len());
    for (_, cut) in cuts {
        let record = records.get(cut).ok_or_else(|| error("a cut whose site was not recorded"))?;
        let row = if tangent { record.tangents[role].clone() } else { record.values[role].clone() };
        if row.is_none() && !tangent {
            return Err(error("a cut whose site was not recorded"));
        }
        refs.push(row);
    }
    wide.convert(&gathered(d, &refs, width)?).map_err(error)
}

/// The rows `refs` (zero where `None`) as one tensor on `d` in their own storage: one gather and
/// one scatter per tensor holding some of them, exact copies.
fn gathered(d: &Device, refs: &[Option<RowRef>], width: usize) -> Result<Tensor, String> {
    let mut out = d.zeros(refs.len(), width).map_err(error)?;
    let mut groups: Vec<(Arc<Tensor>, Vec<u32>, Vec<u32>)> = Vec::new();
    for (k, r) in refs.iter().enumerate() {
        let Some(r) = r else { continue };
        let (from, to) = (u32::try_from(r.row).map_err(error)?, u32::try_from(k).map_err(error)?);
        match groups.iter_mut().find(|(t, _, _)| Arc::ptr_eq(t, &r.rows)) {
            Some(group) => {
                group.1.push(from);
                group.2.push(to);
            }
            None => groups.push((Arc::clone(&r.rows), vec![from], vec![to])),
        }
    }
    for (rows, from, to) in groups {
        let part = d.gather_rows(&rows, &d.upload_indices(&from).map_err(error)?).map_err(error)?;
        d.scatter_rows(&mut out, &d.upload_indices(&to).map_err(error)?, &part, false).map_err(error)?;
    }
    Ok(out)
}

/// Add `rows` (one per entry of `at`) to the rows `at` of `t`, each round of distinct rows at once.
fn add_rows(d: &Device, t: &mut Tensor, at: &[usize], rows: &ndarray::Array2<f64>) -> Result<(), String> {
    add_row_tensor(d, t, at, &d.upload(rows.view()).map_err(error)?)
}

/// [`add_rows`] from rows already on the device.
fn add_row_tensor(d: &Device, t: &mut Tensor, at: &[usize], added: &Tensor) -> Result<(), String> {
    for round in rounds(at) {
        let ranges = single(round.iter().map(|i| at[*i]));
        let mut h = d.gather_ranges(t, &ranges).map_err(error)?;
        let part = picked(d, added, &round)?;
        d.axpy(&mut h, 1.0, part.as_ref().unwrap_or(added)).map_err(error)?;
        d.scatter_ranges(t, &ranges, &h).map_err(error)?;
    }
    Ok(())
}

impl PartSites {

    pub fn nodes(&self, block: usize) -> Option<(usize, usize)> {
        self.nodes.get(block).copied().flatten()
    }
}

/// One experiment: base and source sequences (indices into the batch's), the hybrid (per block
/// whether `P_e` runs `P`'s version of it), at most one patch, and its position: the row the patch
/// replaces and the first scored token (0 for an unpatched experiment, scored everywhere).
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Experiment {
    pub base: usize,
    pub source: usize,
    pub explained: Vec<bool>,
    pub patch: Option<Patch>,
    pub position: usize,
}

impl Experiment {
    /// The patched block of the variables `values`, or the first block operations
    /// act at, `heads` giving each head's block (none for an unpatched experiment).
    fn block(&self, values: &[Value], heads: &[usize]) -> Result<Option<usize>, String> {
        let Some(patch) = &self.patch else { return Ok(None) };
        if let Patch::Ops { ops, .. } = patch {
            return ops.iter().map(|o| o.site.block(heads)).min().flatten().map(Some).ok_or_else(|| error("operations of no site"));
        }
        if let Patch::Weights { block, .. } = patch {
            return Ok(Some(*block));
        }
        let chosen = patch.variables();
        let block = |i: &usize| values.get(*i).map(|v| v.block).ok_or_else(|| error("a patch of an unknown variable"));
        let first = block(chosen.first().ok_or_else(|| error("a joint read patch of no variable"))?)?;
        for (i, w) in chosen.iter().zip(chosen.iter().skip(1)) {
            if i >= w || block(w)? != first {
                return Err(error("a joint read patch needs distinct ascending variables of one block"));
            }
        }
        Ok(Some(first))
    }
}

/// A hybrid of `blocks` blocks: its size `k` uniform in `1..=blocks`, then a uniform set of `k`
/// blocks running `P` (module note).
pub fn hybrid(rng: &mut impl RngExt, blocks: usize) -> Vec<bool> {
    let k = rng.random_range(1..=blocks);
    hybrid_of(rng, blocks, k)
}

/// A hybrid of `blocks` blocks whose `k` blocks running `P` are a uniform set of that size.
pub fn hybrid_of(rng: &mut impl RngExt, blocks: usize, k: usize) -> Vec<bool> {
    let k = k.min(blocks);
    let mut order: Vec<usize> = (0..blocks).collect();
    for i in 0..k {
        let j = rng.random_range(i..blocks);
        order.swap(i, j);
    }
    let mut explained = vec![false; blocks];
    for b in &order[..k] {
        explained[*b] = true;
    }
    explained
}

/// Per base sequence `n < sequences` (each `length` tokens) one clean experiment and one patched
/// with source sequence `n`, drawn in stages, each stage's weight stated:
///
/// * the clean experiment's hybrid: with probability ½ `P` alone (every block `P`'s: the
///   explanation as delivered), else a hybrid drawn by [`hybrid`] (`P`'s parts in `M`'s place);
/// * the patched experiment, under the same hybrid: its block uniform over the blocks holding a
///   read variable (every block of `M`'s own functions; a transcoder's block holds none,
///   `library_transcoder`, and a base has no patched experiment when no block holds one);
///   with probability ½ a read patch of one of the block's `variables`, uniform among them, else
///   a joint read patch of a random subset of them ([`subset`]); its position uniform over the
///   `length` positions.
///
/// A uniform draw over all variables would make nearly every patch an MLP reader's (they are
/// almost all of the variables). The variables are fixed (`M`'s reads), so the joint subsets test
/// the reads a removal drops, many at once, without depending on `P`.
pub fn sample(rng: &mut impl RngExt, sequences: usize, variables: &[ReadVariable], blocks: usize, length: usize) -> Result<Vec<Experiment>, String> {
    let mut at_block: Vec<Vec<usize>> = vec![Vec::new(); blocks];
    for (i, v) in variables.iter().enumerate() {
        at_block.get_mut(v.block).ok_or_else(|| error("a read variable outside the blocks"))?.push(i);
    }
    if blocks == 0 || length == 0 {
        return Err(error("experiments need blocks, and sequences need tokens"));
    }
    let readable: Vec<usize> = (0..blocks).filter(|b| !at_block[*b].is_empty()).collect();
    let mut out = Vec::with_capacity(2 * sequences);
    for n in 0..sequences {
        let explained = if rng.random_range(0..2) == 0 { vec![true; blocks] } else { hybrid(rng, blocks) };
        if readable.is_empty() {
            out.push(Experiment { base: n, source: n, explained, patch: None, position: 0 });
            continue;
        }
        let candidates = &at_block[readable[rng.random_range(0..readable.len())]];
        let patch = if rng.random_range(0..2) == 0 {
            Patch::Read { variable: candidates[rng.random_range(0..candidates.len())] }
        } else {
            Patch::Reads { variables: subset(rng, candidates) }
        };
        let position = rng.random_range(0..length);
        out.push(Experiment { base: n, source: n, explained: explained.clone(), patch: None, position: 0 });
        out.push(Experiment { base: n, source: n, explained, patch: Some(patch), position });
    }
    Ok(out)
}

/// A random nonempty subset of `candidates` (ascending): its size `k` uniform in
/// `1..=candidates.len()`, then a uniform set of `k`, as for hybrids. Small and partial subsets
/// mix the base's and the source's reads; large ones, spanning the stream, swap the whole read.
pub fn subset(rng: &mut impl RngExt, candidates: &[usize]) -> Vec<usize> {
    let k = rng.random_range(1..=candidates.len());
    hybrid_of(rng, candidates.len(), k).iter().zip(candidates).filter(|(chosen, _)| **chosen).map(|(_, v)| *v).collect()
}

/// The realized count of each family among `experiments`: clean with `P` alone, clean under a
/// hybrid, single read patches of attention and of MLP variables, and joint read patches.
pub fn census(experiments: &[Experiment], variables: &[ReadVariable]) -> BTreeMap<&'static str, usize> {
    let mut counts: BTreeMap<&'static str, usize> = ["clean_alone", "clean_hybrid", "read_attention", "read_mlp", "read_joint", "swap", "zero", "scale", "push", "cut", "weight"].into_iter().map(|k| (k, 0)).collect();
    for e in experiments {
        let family = match &e.patch {
            None if e.explained.iter().all(|x| *x) => "clean_alone",
            None => "clean_hybrid",
            Some(Patch::Read { variable }) if variables.get(*variable).is_some_and(|v| v.block % 2 == 0) => "read_attention",
            Some(Patch::Read { .. }) => "read_mlp",
            Some(Patch::Reads { .. }) => "read_joint",
            Some(Patch::Ops { family: Family::Swap, .. }) => "swap",
            Some(Patch::Ops { family: Family::Zero, .. }) => "zero",
            Some(Patch::Ops { family: Family::Scale, .. }) => "scale",
            Some(Patch::Ops { family: Family::Push, .. }) => "push",
            Some(Patch::Ops { family: Family::Cut, .. }) => "cut",
            Some(Patch::Ops { .. }) => "ops",
            Some(Patch::Weights { .. }) => "weight",
        };
        *counts.entry(family).or_default() += 1;
    }
    counts
}

/// Base and source token sequences, all of one length.
pub struct Batch {
    pub base: Vec<Vec<u32>>,
    pub source: Vec<Vec<u32>>,
    length: usize,
}

impl Batch {
    /// The tokens of each sequence.
    pub fn length(&self) -> usize {
        self.length
    }

    pub fn new(base: Vec<Vec<u32>>, source: Vec<Vec<u32>>) -> Result<Self, String> {
        let length = base.first().map_or(0, Vec::len);
        if length == 0 || base.iter().chain(&source).any(|s| s.len() != length) {
            return Err(error("sequences must be nonempty and of one length"));
        }
        Ok(Self { base, source, length })
    }
}

/// The final linear head `M` and `P` share, on the device: compact targets and their scores.
pub struct FixedHead {
    head: Arc<Head>,
    resident: ResidentHead,
    /// The hidden width as one column block (row dots).
    width: ColumnBlocks,
}

impl FixedHead {
    /// The head of the flat programs `native` and `explanation`, which must be the same matrix;
    /// `tile_rows` rows of vocabulary logits are formed at once where a score forms them.
    pub fn new(device: &Device, native: &OperatorProgram, explanation: &OperatorProgram, tile_rows: usize) -> Result<Self, String> {
        let head = Head::of(native)?;
        if !head.same(&Head::of(explanation)?) {
            return Err(error("the explanation's head is not the model's"));
        }
        if tile_rows == 0 {
            return Err(error("positive head tile rows required"));
        }
        let resident = ResidentHead::new(device, &head, tile_rows)?;
        let width = device.column_blocks(&[head.embedding().ncols()]).map_err(error)?;
        Ok(Self { head: Arc::new(head), resident, width })
    }

    /// The compact target of the hidden rows `hidden`: per row `μ = E_p[e]` of the head's rows `e`
    /// under `p = softmax(E h)`, and `Σ p log p = μ·h − log Z`, `Z` the partition; the products in
    /// `arithmetic`. `Device::head_log_partition` gives `log Z` and `μ` without forming the rows ×
    /// vocabulary logits in f32 storage.
    fn target(&self, d: &Device, hidden: &Tensor, arithmetic: Arithmetic) -> Result<Target, String> {
        // A Gaussian head's target is `M`'s outputs (`Head::gaussian_target`), made on the host.
        if self.head.gaussian.is_some() {
            let target = self.head.gaussian_target(&d.download(hidden).map_err(error)?)?;
            let mu = d.upload(target.view()).map_err(error)?;
            return Ok(Target { mu: Arc::new(mu), entropy: vec![0.0; hidden.rows()], head: Arc::clone(&self.head), scored: None });
        }
        let mut mu = d.zeros(hidden.rows(), hidden.cols()).map_err(error)?;
        let partitions = d.head_log_partition(hidden, &self.resident.embedding, false, None, Some(&mut mu), arithmetic).map_err(error)?;
        let dots = d.download(&d.block_products(hidden, &mu, &self.width).map_err(error)?).map_err(error)?;
        let entropy = partitions.iter().enumerate().map(|(r, z)| dots[(r, 0)] - z).collect();
        Ok(Target { mu: Arc::new(mu), entropy, head: Arc::clone(&self.head), scored: None })
    }
}

/// The read variables every explanation of the split native program `native` with its `layers` is
/// asked about: `M`'s functions, in the order of the library started at `M`
/// (`library_mdl::explanation`, `library_reads`), each part an operator of `native` and its rows.
/// The library's ownership map (`Artifact::owners`) names the native operator and rows each of its
/// read rows replaces.
pub fn reads(native: &OperatorProgram, layers: &[LayerNodes]) -> Result<Vec<ReadVariable>, String> {
    Ok(crate::library_mdl::explanation(native, layers)?.reads)
}

/// [`reads`] from `start`, the library explanation of `native`'s `blocks` layers as
/// `library_mdl::explanation` makes it (its program and its owners).
pub(crate) fn reads_of(native: &OperatorProgram, start: &crate::artifact::Artifact, blocks: usize) -> Result<Vec<ReadVariable>, String> {
    let program = &start.program;
    let named: BTreeMap<&str, usize> = native.operators.iter().enumerate().map(|(i, op)| (op.name.as_str(), i)).collect();
    // Each library operator's read-map owners, in the ownership map's order.
    let mut owners: HashMap<&str, Vec<&crate::artifact::Owner>> = HashMap::new();
    for owner in start.owners.iter().filter(|o| READS.contains(&o.role.as_str())) {
        owners.entry(owner.operator.as_str()).or_default().push(owner);
    }
    // Library operators whose read owners are not operators of `M` (a transcoder's features,
    // `library_transcoder`): their read variables have no value in `M` to patch.
    let foreign: std::collections::HashSet<&str> = start.owners.iter().filter(|o| READS.contains(&o.role.as_str()) && !named.contains_key(o.native.as_str())).map(|o| o.operator.as_str()).collect();
    library_reads(program, blocks)?
        .into_iter()
        .filter(|v| !v.parts.iter().all(|(op, _)| foreign.contains(program.operators[*op].name.as_str())))
        .map(|v| {
            let mut parts: Vec<(usize, Range<usize>)> = Vec::new();
            for (op, rows) in &v.parts {
                let name = &program.operators[*op].name;
                for owner in owners.get(name.as_str()).into_iter().flatten() {
                    let (from, to) = (rows.start.max(owner.rows.start), rows.end.min(owner.rows.end));
                    if from >= to {
                        continue;
                    }
                    let at = *named.get(owner.native.as_str()).ok_or_else(|| error(format!("{}: no native operator {}", name, owner.native)))?;
                    let shift = owner.native_rows.start;
                    let part = (at, shift + from - owner.rows.start..shift + to - owner.rows.start);
                    if !parts.contains(&part) {
                        parts.push(part);
                    }
                }
            }
            if parts.is_empty() {
                return Err(error(format!("block {}: a read variable no native operator owns", v.block)));
            }
            Ok(ReadVariable { block: v.block, parts })
        })
        .collect()
}

/// The roles of the owners (`Artifact::owners`) whose blocks are read maps: a head's query, key
/// and value maps, an MLP's gate and up maps.
const READS: [&str; 5] = ["q", "k", "v", "gate", "up"];

/// Where a model holds part of a read variable's value: the node of its flat program, the node's
/// width, and the columns of the node's value the part is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Site {
    pub node: usize,
    pub width: usize,
    pub columns: Range<usize>,
}

/// Where a model holds one read variable's value: the variable's block and the sites of its parts
/// (none for a function the model no longer computes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Value {
    pub block: usize,
    pub sites: Vec<Site>,
}

/// Where `artifact`, an explanation of the split native program `native` (or `Artifact::native`
/// itself) with its `layers`, holds each of the read `variables` ([`reads`]): per variable the
/// nodes of its flat program ([`sites`]) and their columns. With an ownership map, each native
/// part's sites are those of its owners: the node applying the owner's operator in the owner's
/// rule body, at the owner's call site, the columns its rows; a part no owner replaces has no site,
/// the function being gone from `artifact`. Without one, the artifact's operators are the native
/// ones, each applied by one node. Every site lies in its variable's block, after the block's read.
pub fn values(native: &OperatorProgram, artifact: &Artifact, layers: &[LayerNodes], variables: &[ReadVariable]) -> Result<Vec<Value>, String> {
    let (flat, entries, reads) = sites(artifact, layers)?;
    let program = &artifact.program;
    let named: BTreeMap<&str, usize> = program.operators.iter().enumerate().map(|(i, op)| (op.name.as_str(), i)).collect();
    // The flat program's nodes by the operator their first term applies, and the read-map owners by
    // the native operator they replace (in the ownership map's order): each variable looks up its
    // own instead of scanning every node or owner.
    let mut applying: HashMap<usize, Vec<usize>> = HashMap::new();
    for (n, node) in flat.nodes.iter().enumerate() {
        if let Node::Affine { terms, .. } = node
            && let Some(first) = terms.first()
        {
            applying.entry(first.1).or_default().push(n);
        }
    }
    let mut owners: HashMap<&str, Vec<&crate::artifact::Owner>> = HashMap::new();
    for owner in artifact.owners.iter().filter(|o| READS.contains(&o.role.as_str())) {
        owners.entry(owner.native.as_str()).or_default().push(owner);
    }
    // Per variable, per site: the observation path or the node, and the columns.
    let mut found: Vec<Vec<(Result<Vec<usize>, usize>, Range<usize>)>> = Vec::with_capacity(variables.len());
    for v in variables {
        let mut out = Vec::new();
        for (op, rows) in &v.parts {
            let name = native.operators.get(*op).map(|o| o.name.as_str()).ok_or_else(|| error("a read variable of an unknown native operator"))?;
            if artifact.owners.is_empty() {
                let at = *named.get(name).ok_or_else(|| error(format!("{name}: not an operator of the model")))?;
                let node = match applying.get(&at).map(Vec::as_slice) {
                    Some(&[n]) => n,
                    _ => return Err(error(format!("{name}: not applied by exactly one node"))),
                };
                out.push((Err(node), rows.clone()));
                continue;
            }
            for owner in owners.get(name).into_iter().flatten() {
                let (from, to) = (rows.start.max(owner.native_rows.start), rows.end.min(owner.native_rows.end));
                if from >= to {
                    continue;
                }
                let mut path = invocation(program, &owner.body, &owner.site)?;
                let body = path.last().and_then(|n| call_rule(program, &path[..path.len() - 1], *n)).ok_or_else(|| error(format!("{}: no rule {}", owner.site, owner.body)))?;
                // A summed block's use holds the map's output at its output node.
                let node = match owner.uses {
                    Some((_, output)) => output,
                    None => {
                        let at = *named.get(owner.operator.as_str()).ok_or_else(|| error(format!("{}: no operator {}", owner.site, owner.operator)))?;
                        program.rules[body]
                            .nodes
                            .iter()
                            .position(|n| matches!(n, Node::Affine { terms, .. } if terms.first().is_some_and(|t| t.1 == at)))
                            .ok_or_else(|| error(format!("{}: no node of {} applies {}", owner.site, owner.body, owner.operator)))?
                    }
                };
                path.push(node);
                let shift = owner.rows.start;
                out.push((Ok(path), shift + from - owner.native_rows.start..shift + to - owner.native_rows.start));
            }
        }
        found.push(out);
    }
    let paths: Vec<Vec<usize>> = found.iter().flatten().filter_map(|(path, _)| path.as_ref().ok().cloned()).collect();
    let observed = if paths.is_empty() {
        Vec::new()
    } else {
        let (observing, _, observed) = mapped_inlined_observed(program, &paths)?;
        if observing.nodes.len() != flat.nodes.len() {
            return Err(error("the observed program differs from the flat program"));
        }
        observed
    };
    let mut next = observed.into_iter();
    let interfaces = flat.interfaces().map_err(error)?;
    let blocks = entries.len();
    let end = |b: usize| if b + 1 < blocks { entries[b + 1] } else { flat.nodes.len() };
    variables
        .iter()
        .zip(found)
        .map(|(v, parts)| {
            let sites = parts
                .into_iter()
                .map(|(path, columns)| {
                    let node = match path {
                        Ok(_) => next.next().ok_or_else(|| error("an unobserved site"))?,
                        Err(node) => node,
                    };
                    let width = interfaces.get(node).ok_or_else(|| error("a site outside the program"))?.width();
                    if v.block >= blocks || node <= reads[v.block] || node >= end(v.block) || columns.end > width {
                        return Err(error(format!("block {}: a read variable's value outside the block", v.block)));
                    }
                    Ok(Site { node, width, columns })
                })
                .collect::<Result<_, String>>()?;
            Ok(Value { block: v.block, sites })
        })
        .collect()
}

/// The path of call nodes from a root node of `program` to the call of the rule named `body` at
/// call site `site` (outermost first, `mapped_inlined_observed`'s form). A rule called at one site
/// has that site's name; a body called at several sites is told apart by its call's read binding,
/// the operator `{site}.read` applied to the call's argument.
fn invocation(program: &OperatorProgram, body: &str, site: &str) -> Result<Vec<usize>, String> {
    fn walk(program: &OperatorProgram, nodes: &[Node], body: &str, site: &str, path: &mut Vec<usize>, found: &mut Vec<Vec<usize>>) {
        for (n, node) in nodes.iter().enumerate() {
            let Node::Call { rule, arguments } = node else { continue };
            let Some(called) = program.rules.get(*rule) else { continue };
            path.push(n);
            if called.name == body {
                let read = arguments.first().and_then(|a| match nodes.get(*a) {
                    Some(Node::Affine { terms, .. }) => terms.first().and_then(|t| program.operators.get(t.1)).map(|op| op.name.as_str()),
                    _ => None,
                });
                if site == body || read == Some(format!("{site}.read").as_str()) {
                    found.push(path.clone());
                }
            } else {
                walk(program, &called.nodes, body, site, path, found);
            }
            path.pop();
        }
    }
    let mut found = Vec::new();
    walk(program, &program.nodes, body, site, &mut Vec::new(), &mut found);
    match (found.pop(), found.is_empty()) {
        (Some(path), true) => Ok(path),
        (None, _) => Err(error(format!("{site}: no call of {body}"))),
        (Some(_), false) => Err(error(format!("{site}: several calls of {body}"))),
    }
}

/// The rule called by node `node` of the body that the call path `outer` leads into (the root
/// program's nodes when it is empty).
fn call_rule(program: &OperatorProgram, outer: &[usize], node: usize) -> Option<usize> {
    let mut nodes = &program.nodes;
    for n in outer {
        let Node::Call { rule, .. } = nodes.get(*n)? else { return None };
        nodes = &program.rules.get(*rule)?.nodes;
    }
    match nodes.get(node)? {
        Node::Call { rule, .. } => Some(*rule),
        _ => None,
    }
}

/// The patches of one engine call (module note): per node of the model's flat program, the rows
/// whose patched entries take the source row's (both rows of the call), with the 0/1 masks of the
/// node's columns that keep the base's entries and that take the source's.
pub struct Edits {
    nodes: BTreeMap<usize, Patches>,
    /// Pushed directions by node: the rows and the vectors added there (the forward's alone: a
    /// constant has no tangent and passes the cotangent as it is).
    adds: BTreeMap<usize, (Vec<usize>, ndarray::Array2<f64>)>,
    /// Connection cuts: probes by the site node they record (row, cut, role), the reads they
    /// replace, and the plan's record they share ([`Operation::Cut`]).
    probes: BTreeMap<usize, Vec<(usize, usize, usize)>>,
    cuts: BTreeMap<usize, CutReads>,
    recorded: Option<Records>,
    /// Native weight edits by the node their terms add to ([`MatrixUse`]), and per input node of
    /// a term the cotangents the terms carry back to it in a reverse pass (which reaches a term's
    /// output before its input).
    weights: BTreeMap<usize, Vec<WeightTerm>>,
    carried: RefCell<BTreeMap<usize, Vec<(Vec<usize>, Tensor)>>>,
}

/// One edit's term `x R Wᵀ` at a matrix use's output, `x` the use's input at the call's rows
/// `rows`: `R` (input width × rank) holds the edit's input-side factor at the use's input columns
/// and `W` (output width × rank) its output-side factor at the output columns, divided by the
/// use's factor, both zero elsewhere.
struct WeightTerm {
    input: usize,
    rows: Vec<usize>,
    read: Tensor,
    write: Tensor,
}

/// `added` (one row per entry of `rows`, distinct) added to the rows `rows` of `t`, on the device.
fn add_device_rows(d: &Device, t: &mut Tensor, rows: &[usize], added: &Tensor) -> Result<(), String> {
    let ranges = single(rows.iter().copied());
    let mut h = d.gather_ranges(t, &ranges).map_err(error)?;
    d.axpy(&mut h, 1.0, added).map_err(error)?;
    d.scatter_ranges(t, &ranges, &h).map_err(error)
}

/// The arithmetic of a device's products, as [`Interchange::new`] sets the models'.
fn arithmetic_of(d: &Device) -> Arithmetic {
    if d.float64() { Arithmetic::F64 } else { Arithmetic::F32 }
}

/// `x R Wᵀ` for `x` the rows `rows` of `input` (`R`, `W` a [`WeightTerm`]'s factors).
fn weight_term(d: &Device, input: &Tensor, rows: &[usize], read: &Tensor, write: &Tensor) -> Result<Tensor, String> {
    let x = d.gather_ranges(input, &single(rows.iter().copied())).map_err(error)?;
    let mut low = d.empty(rows.len(), read.cols()).map_err(error)?;
    d.gemm(&mut low, 1.0, &x, Op::N, read, Op::N, 0.0, arithmetic_of(d)).map_err(error)?;
    let mut out = d.empty(rows.len(), write.rows()).map_err(error)?;
    d.gemm(&mut out, 1.0, &low, Op::N, write, Op::T, 0.0, arithmetic_of(d)).map_err(error)?;
    Ok(out)
}

/// One node's patches ([`Edits`]) in the order of their (patched row, source row): the patched
/// rows, the source rows, and one row per patch of the 0/1 masks of the node's columns that keep
/// the patched row's entries and that take the source's.
struct Patches {
    rows: Vec<usize>,
    sources: Vec<usize>,
    keep: Tensor,
    take: Tensor,
}

/// The positions of `rows` split into rounds of distinct rows, in order: the `k`-th occurrence of
/// a row is in round `k`, so writing (or adding to) each round's rows at once does what writing
/// them one at a time in order does.
fn rounds(rows: &[usize]) -> Vec<Vec<usize>> {
    let mut seen: BTreeMap<usize, usize> = BTreeMap::new();
    let mut out: Vec<Vec<usize>> = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let k = seen.entry(*row).or_insert(0);
        if out.len() <= *k {
            out.push(Vec::new());
        }
        out[*k].push(i);
        *k += 1;
    }
    out
}

/// One-row ranges at `rows`.
fn single(rows: impl IntoIterator<Item = usize>) -> Vec<Range<usize>> {
    rows.into_iter().map(|r| r..r + 1).collect()
}

/// The rows `round` of `t`, or `None` when the round is every row of `t` in order (`t` itself).
fn picked(d: &Device, t: &Tensor, round: &[usize]) -> Result<Option<Tensor>, String> {
    if round.len() == t.rows() && round.iter().enumerate().all(|(i, r)| i == *r) {
        return Ok(None);
    }
    Ok(Some(d.gather_ranges(t, &single(round.iter().copied())).map_err(error)?))
}

impl Edits {
    /// The patches of a call at block rows `patches` (the patched row, the source's row, and the
    /// variables) at the sites `values` of the model that runs the call, and its edits `edits`
    /// (the edited row and the edit: an operation on a shared site or a cut's step) at the
    /// model's edit sites `sites`.
    pub(crate) fn with_parts(d: &Device, patches: &[(usize, usize, &[usize])], values: &[Value], edits: &[(usize, Edit)], sites: Option<&PartSites>) -> Result<Self, String> {
        let mut zeroed = Vec::new();
        let mut adds: BTreeMap<usize, (Vec<usize>, Vec<Vec<f64>>)> = BTreeMap::new();
        let mut probes: BTreeMap<usize, Vec<(usize, usize, usize)>> = BTreeMap::new();
        let mut cuts: BTreeMap<usize, CutReads> = BTreeMap::new();
        let mut weight_rows: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (row, edit) in edits {
            let sites = sites.ok_or_else(|| error("an edit in a model without edit sites"))?;
            if sites.unedited {
                continue;
            }
            match edit {
                Edit::OpAt { site, operation, from } => {
                    let (node, width) = sites.shared.get(site).copied().ok_or_else(|| error(format!("{site:?}: not held by the model")))?;
                    match operation {
                        Operation::Swap => zeroed.push((*row, *from, node, width, 0.0, 1.0)),
                        Operation::Scale(i) => zeroed.push((*row, *row, node, width, *SCALES.get(*i).ok_or_else(|| error("a scale outside SCALES"))?, 0.0)),
                        Operation::Push { direction, size } => {
                            let unit = sites.directions.get(*direction).ok_or_else(|| error("a push of an unknown direction"))?;
                            let norm = sites.typical.get(site).copied().ok_or_else(|| error(format!("{site:?}: no typical norm")))?;
                            let size = SIZES.get(*size).ok_or_else(|| error("a push outside SIZES"))?;
                            if unit.len() != width {
                                return Err(error(format!("{site:?}: a direction of another width")));
                            }
                            let entry = adds.entry(node).or_default();
                            entry.0.push(*row);
                            entry.1.push(unit.iter().map(|u| size * norm * u).collect());
                        }
                        Operation::Cut { .. } => return Err(error("a cut without its probes")),
                    }
                }
                Edit::Op { .. } => return Err(error("an operation without its call's rows")),
                Edit::Probe { cut, site, role, .. } => {
                    let (node, _) = sites.shared.get(site).copied().ok_or_else(|| error(format!("{site:?}: not held by the model")))?;
                    probes.entry(node).or_default().push((*row, *cut, *role));
                }
                Edit::Weight { edit, .. } => weight_rows.entry(*edit).or_default().push(*row),
                Edit::CutRead { cut, block, .. } => {
                    let (read, norm) = sites.reads.get(*block).ok_or_else(|| error(format!("block {block}: no read")))?;
                    let norm = norm.clone().ok_or_else(|| error(format!("block {block}: an input norm a cut cannot recompute")))?;
                    cuts.entry(*read).or_insert_with(|| CutReads { norm, rows: Vec::new(), kept: RefCell::new(None), entering: RefCell::new(None) }).rows.push((*row, *cut));
                }
            }
        }
        // Each weight edit's coordinate edits, entry-wise: at every use of the edited map in the
        // model, the edited rows' entries of the use's output (its columns indexing the map's rows)
        // or the edited columns' entries of its input, scaled at the edit's rows, so each part and
        // leftover computing the map computes it with those rows or columns scaled.
        let mut scaled: BTreeMap<(usize, usize), Vec<f64>> = BTreeMap::new();
        if let Some(sites) = sites.filter(|s| !weight_rows.is_empty() && !s.unedited) {
            for (e, rows) in &weight_rows {
                let drawn = sites.weights.get(*e).ok_or_else(|| error(format!("weight edit {e}: not in the table")))?;
                for x in &drawn.entries {
                    let mut done: BTreeSet<(usize, usize)> = BTreeSet::new();
                    for u in sites.matrices.get(&x.native).into_iter().flatten() {
                        // The output indexes the map's rows, the input its columns (the other way
                        // round for a block held transposed).
                        let output = x.rows != u.transposed;
                        let native = if x.rows { &u.native_rows } else { &u.native_cols };
                        let (node, at, width) = if output { (u.output, u.out_at, u.widths.0) } else { (u.input, u.in_at, u.widths.1) };
                        let columns: Vec<usize> = x.units.iter().filter(|j| native.contains(j)).map(|j| at + j - native.start).filter(|c| done.insert((node, *c))).collect();
                        if columns.iter().any(|c| *c >= width) {
                            return Err(error(format!("{}: an edited unit outside its use", x.native)));
                        }
                        for &row in rows {
                            let mask = scaled.entry((node, row)).or_insert_with(|| vec![1.0; width]);
                            columns.iter().for_each(|c| mask[*c] *= x.alpha);
                        }
                    }
                }
            }
        }
        let scaled: Vec<(usize, usize, Vec<f64>)> = scaled.into_iter().map(|((node, row), mask)| (row, node, mask)).collect();
        let mut out = Self::patches(d, patches, values, (&zeroed, &scaled))?;
        out.adds = adds.into_iter().map(|(node, (rows, vectors))| (node, (rows, ndarray::Array2::from_shape_fn((vectors.len(), vectors[0].len()), |(i, c)| vectors[i][c])))).collect();
        out.probes = probes;
        out.cuts = cuts;
        // Each weight edit's term at every use of each matrix it edits, over its rows.
        if let Some(sites) = sites.filter(|s| !weight_rows.is_empty() && !s.unedited) {
            for (e, rows) in weight_rows {
                let drawn = sites.weights.get(e).ok_or_else(|| error(format!("weight edit {e}: not in the table")))?;
                for f in &drawn.factors {
                    for u in sites.matrices.get(&f.native).into_iter().flatten() {
                        let rank = f.left.ncols();
                        // The input side indexes the native columns, the output side the rows (the
                        // other way round for a block held transposed).
                        let ((inward, in_native), (outward, out_native)) = if u.transposed { ((&f.left, &u.native_rows), (&f.right, &u.native_cols)) } else { ((&f.right, &u.native_cols), (&f.left, &u.native_rows)) };
                        let (out_width, in_width) = u.widths;
                        if inward.ncols() != rank || u.in_at + in_native.len() > in_width || u.out_at + out_native.len() > out_width || in_native.end > inward.nrows() || out_native.end > outward.nrows() {
                            return Err(error(format!("{}: an edit's factors do not fit its use", f.native)));
                        }
                        let mut read = ndarray::Array2::zeros((in_width, rank));
                        read.slice_mut(ndarray::s![u.in_at..u.in_at + in_native.len(), ..]).assign(&inward.slice(ndarray::s![in_native.clone(), ..]));
                        let mut write = ndarray::Array2::zeros((out_width, rank));
                        let factor = u.factor.as_ref().map_err(|e| error(format!("{}: {e}", f.native)))?;
                        write.slice_mut(ndarray::s![u.out_at..u.out_at + out_native.len(), ..]).assign(&(&outward.slice(ndarray::s![out_native.clone(), ..]) / *factor));
                        out.weights.entry(u.output).or_default().push(WeightTerm { input: u.input, rows: rows.clone(), read: d.upload(read.view()).map_err(error)?, write: d.upload(write.view()).map_err(error)? });
                    }
                }
            }
        }
        Ok(out)
    }

    /// The read patches `patches` and the rows `zeroed` (a row, a node and its width) of nodes
    /// whose value is zeroed there (a head's removal: the patch keeping none of the row and taking
    /// none of its source, the row itself).
    fn patches(d: &Device, patches: &[(usize, usize, &[usize])], values: &[Value], (zeroed, scaled): (&[(usize, usize, usize, usize, f64, f64)], &[(usize, usize, Vec<f64>)])) -> Result<Self, String> {
        let mut masks: BTreeMap<(usize, usize, usize), (Vec<f64>, Vec<f64>)> = BTreeMap::new();
        for (row, source, variables) in patches {
            for v in *variables {
                for site in &values.get(*v).ok_or_else(|| error("a patch of an unknown variable"))?.sites {
                    let mask = masks.entry((site.node, *row, *source)).or_insert_with(|| (vec![1.0; site.width], vec![0.0; site.width]));
                    mask.0[site.columns.clone()].iter_mut().for_each(|m| *m = 0.0);
                    mask.1[site.columns.clone()].iter_mut().for_each(|m| *m = 1.0);
                }
            }
        }
        for (row, source, node, width, keep, take) in zeroed {
            masks.insert((*node, *row, *source), (vec![*keep; *width], vec![*take; *width]));
        }
        // A row's entries scaled column by column (a weight edit's coordinate edits), on top of
        // whatever the row already keeps.
        for (row, node, factors) in scaled {
            let mask = masks.entry((*node, *row, *row)).or_insert_with(|| (vec![1.0; factors.len()], vec![0.0; factors.len()]));
            mask.0.iter_mut().zip(factors).for_each(|(k, f)| *k *= f);
            mask.1.iter_mut().zip(factors).for_each(|(t, f)| *t *= f);
        }
        let mut grouped: BTreeMap<usize, (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>)> = BTreeMap::new();
        for ((node, row, source), (keep, take)) in masks {
            let entry = grouped.entry(node).or_default();
            entry.0.push(row);
            entry.1.push(source);
            entry.2.extend(keep);
            entry.3.extend(take);
        }
        let mut nodes = BTreeMap::new();
        for (node, (rows, sources, keep, take)) in grouped {
            let width = keep.len() / rows.len();
            let (keep, take) = (d.upload_vec(rows.len(), width, keep).map_err(error)?, d.upload_vec(rows.len(), width, take).map_err(error)?);
            nodes.insert(node, Patches { rows, sources, keep, take });
        }
        Ok(Self { nodes, adds: BTreeMap::new(), probes: BTreeMap::new(), cuts: BTreeMap::new(), recorded: None, weights: BTreeMap::new(), carried: RefCell::new(BTreeMap::new()) })
    }

    /// The nodes the call edits.
    /// The nodes the call edits, and the inputs of its weight edits' terms (whose cotangents
    /// take the terms' share, [`Edits::transpose`]).
    pub fn nodes(&self) -> BTreeSet<usize> {
        let inputs = self.weights.values().flatten().map(|t| t.input);
        self.nodes.keys().chain(self.adds.keys()).chain(self.probes.keys()).chain(self.cuts.keys()).chain(self.weights.keys()).copied().chain(inputs).collect()
    }

    /// Add the cuts' cotangents of the stream entering the block ([`Edits::transpose`]) to its
    /// cotangent `entering` (the call's rows), once per reverse pass.
    pub fn add_entering(&self, d: &Device, entering: &mut Tensor) -> Result<(), String> {
        for c in self.cuts.values() {
            if let Some(rows) = c.entering.borrow_mut().take() {
                add_row_tensor(d, entering, &c.rows.iter().map(|r| r.0).collect::<Vec<_>>(), &d.convert(&rows).map_err(error)?)?;
            }
        }
        Ok(())
    }

    /// Whether the forward pass changes node `node`'s value or records it (a cut's probe).
    pub fn changes(&self, node: usize) -> bool {
        self.nodes.contains_key(&node) || self.adds.contains_key(&node) || self.probes.contains_key(&node) || self.cuts.contains_key(&node) || self.weights.contains_key(&node)
    }

    /// The additions at node `node` on its value `value` (the call's rows), `value_of` giving the
    /// call's other nodes' values: the pushed directions, the cuts' probes and replaced reads
    /// at its row, a function of the model's own read there.
    pub fn write<'v>(&self, d: &Device, node: usize, value: &mut Tensor, value_of: impl Fn(usize) -> Result<&'v Tensor, String>) -> Result<(), String> {
        if let Some((rows, vectors)) = self.adds.get(&node) {
            add_rows(d, value, rows, vectors)?;
        }
        for t in self.weights.get(&node).into_iter().flatten() {
            let term = weight_term(d, value_of(t.input)?, &t.rows, &t.read, &t.write)?;
            add_device_rows(d, value, &t.rows, &term)?;
        }
        // The probes and cuts stay on the device: a read-back here held the device idle while the
        // host computed the replaced rows (about 110 ms of a 686 ms vpd4l step, decomp's grouped
        // direction fit on an A40).
        let rows_of = |t: &Tensor, at: &[usize]| -> Result<Tensor, String> { d.gather_ranges(t, &single(at.iter().copied())).map_err(error) };
        if let Some(probes) = self.probes.get(&node) {
            let recorded = self.recorded.as_ref().ok_or_else(|| error("a cut outside a plan"))?;
            let x = Arc::new(rows_of(value, &probes.iter().map(|p| p.0).collect::<Vec<_>>())?);
            for (k, (_, cut, role)) in probes.iter().enumerate() {
                recorded.borrow_mut().entry(*cut).or_default().values[*role] = Some(RowRef { rows: Arc::clone(&x), row: k });
            }
        }
        if let Some(c) = self.cuts.get(&node) {
            let recorded = self.recorded.as_ref().ok_or_else(|| error("a cut outside a plan"))?;
            let at: Vec<usize> = c.rows.iter().map(|r| r.0).collect();
            if rounds(&at).len() > 1 {
                return Err(error("two cuts replacing one read's row"));
            }
            // y = s + b − a per cut (`s` the stream entering the block, `b` and `a` the site's donor
            // and base values) and N(y), in float64: each sum and product rounded once, in the
            // host's order, so the rows are its values bit for bit.
            let wide = wide(d)?;
            let width = value_of(c.norm.entry)?.cols();
            let mut y = wide.convert(&rows_of(value_of(c.norm.entry)?, &at)?).map_err(error)?;
            wide.axpy(&mut y, 1.0, &stacked(d, &wide, recorded, &c.rows, (1, false), width)?).map_err(error)?;
            wide.axpy(&mut y, -1.0, &stacked(d, &wide, recorded, &c.rows, (0, false), width)?).map_err(error)?;
            let replaced = d.convert(&c.norm.rows(&wide, RowNorm::Apply, &y, None)?).map_err(error)?;
            *c.kept.borrow_mut() = Some(y);
            d.scatter_ranges(value, &single(at.iter().copied()), &replaced).map_err(error)?;
        }
        Ok(())
    }

    /// Node `node`'s value `value` (the call's rows in order) with each patched row's patched
    /// entries replaced by its source row's: `h ⊙ (1 − m) + s ⊙ m`, both products exact, every
    /// source row read before any row is written, the patches applied as one at a time in order
    /// (a patched row repeated takes its later patches on top of its earlier ones), each round of
    /// distinct patched rows at once.
    pub fn apply(&self, d: &Device, node: usize, value: &mut Tensor) -> Result<(), String> {
        let Some(p) = self.nodes.get(&node) else { return Ok(()) };
        let sources = d.gather_ranges(value, &single(p.sources.iter().copied())).map_err(error)?;
        for round in rounds(&p.rows) {
            let rows = single(round.iter().map(|i| p.rows[*i]));
            let h = d.gather_ranges(value, &rows).map_err(error)?;
            let (keep, take, from) = (picked(d, &p.keep, &round)?, picked(d, &p.take, &round)?, picked(d, &sources, &round)?);
            let mut out = d.empty(round.len(), value.cols()).map_err(error)?;
            d.hadamard(&mut out, &h, keep.as_ref().unwrap_or(&p.keep), false).map_err(error)?;
            d.hadamard(&mut out, from.as_ref().unwrap_or(&sources), take.as_ref().unwrap_or(&p.take), true).map_err(error)?;
            d.scatter_ranges(value, &rows, &out).map_err(error)?;
        }
        Ok(())
    }

    /// The tangent of node `node` (`t`, `None` for zero) under the call's edits: the masks applied
    /// to it; a pushed direction is a constant, whose tangent is zero.
    pub fn tangent(&self, d: &Device, node: usize, t: &mut Option<Tensor>, dv: &[Option<Tensor>], (rows, width): (usize, usize)) -> Result<(), String> {
        if let Some(value) = t.as_mut() {
            self.apply(d, node, value)?;
        }
        for term in self.weights.get(&node).into_iter().flatten() {
            let Some(dx) = dv.get(term.input).and_then(Option::as_ref) else { continue };
            let added = weight_term(d, dx, &term.rows, &term.read, &term.write)?;
            let value = match t {
                Some(value) => value,
                None => t.insert(d.zeros(rows, width).map_err(error)?),
            };
            add_device_rows(d, value, &term.rows, &added)?;
        }
        let rows_of = |t: &Tensor, at: &[usize]| -> Result<Tensor, String> { d.gather_ranges(t, &single(at.iter().copied())).map_err(error) };
        if let Some(probes) = self.probes.get(&node) {
            let recorded = self.recorded.as_ref().ok_or_else(|| error("a cut outside a plan"))?;
            let dx = t.as_ref().map(|t| rows_of(t, &probes.iter().map(|p| p.0).collect::<Vec<_>>())).transpose()?.map(Arc::new);
            for (k, (_, cut, role)) in probes.iter().enumerate() {
                recorded.borrow_mut().entry(*cut).or_default().tangents[*role] = dx.as_ref().map(|dx| RowRef { rows: Arc::clone(dx), row: k });
            }
        }
        if let Some(c) = self.cuts.get(&node) {
            let recorded = self.recorded.as_ref().ok_or_else(|| error("a cut outside a plan"))?;
            let at: Vec<usize> = c.rows.iter().map(|r| r.0).collect();
            // dy = ds + t₁ − t₀ (zero where no tangent reached the entering stream or a site) and
            // N's tangent at y, in float64 on the device as the forward's.
            let wide = wide(d)?;
            let mut dy = match dv.get(c.norm.entry).and_then(Option::as_ref) {
                Some(ds) => wide.convert(&rows_of(ds, &at)?).map_err(error)?,
                None => wide.zeros(at.len(), width).map_err(error)?,
            };
            wide.axpy(&mut dy, 1.0, &stacked(d, &wide, recorded, &c.rows, (1, true), width)?).map_err(error)?;
            wide.axpy(&mut dy, -1.0, &stacked(d, &wide, recorded, &c.rows, (0, true), width)?).map_err(error)?;
            let kept = c.kept.borrow();
            let y = kept.as_ref().ok_or_else(|| error("a cut's tangent before its forward"))?;
            let replaced = d.convert(&c.norm.rows(&wide, RowNorm::Tangent, y, Some(&dy))?).map_err(error)?;
            let value = match t {
                Some(value) => value,
                None => t.insert(d.zeros(rows, width).map_err(error)?),
            };
            d.scatter_ranges(value, &single(at.iter().copied()), &replaced).map_err(error)?;
        }
        Ok(())
    }

    /// The transpose of [`Edits::apply`] on node `node`'s cotangent `g`: each patched row keeps
    /// `ḡ ⊙ (1 − m)` and its source row receives `ḡ ⊙ m`, every patched row read before any row
    /// is written and the sources' shares added in order, each round of distinct rows at once.
    pub fn transpose(&self, d: &Device, node: usize, g: &mut Tensor) -> Result<(), String> {
        // A weight edit's term `x R Wᵀ` added to this node: its input takes `ḡ W Rᵀ` at the term's
        // rows, carried to the input's own visit (later in the pass); the node keeps `ḡ`.
        for t in self.weights.get(&node).into_iter().flatten() {
            let back = weight_term(d, g, &t.rows, &t.write, &t.read)?;
            self.carried.borrow_mut().entry(t.input).or_default().push((t.rows.clone(), back));
        }
        if let Some(carried) = self.carried.borrow_mut().remove(&node) {
            for (rows, back) in carried {
                add_device_rows(d, g, &rows, &back)?;
            }
        }
        // A cut's read row was replaced by N(y): its cotangent goes through N's transpose to the
        // stream entering the block (added by its reverse, `add_entering`) and to the cut site's
        // donor and base values (carried to the site's block), and none to the read's own rule.
        if let Some(c) = self.cuts.get(&node) {
            let recorded = self.recorded.as_ref().ok_or_else(|| error("a cut outside a plan"))?;
            let at: Vec<usize> = c.rows.iter().map(|r| r.0).collect();
            // N's pullback of the read rows' cotangent at y, in float64 on the device.
            let wide = wide(d)?;
            let w = wide.convert(&d.gather_ranges(g, &single(at.iter().copied())).map_err(error)?).map_err(error)?;
            let kept = c.kept.borrow();
            let y = kept.as_ref().ok_or_else(|| error("a cut's reverse before its forward"))?;
            let back = Arc::new(c.norm.rows(&wide, RowNorm::Pullback, y, Some(&w))?);
            for (k, (_, cut)) in c.rows.iter().enumerate() {
                recorded.borrow_mut().get_mut(cut).ok_or_else(|| error("a cut whose site was not recorded"))?.carried.push_back(RowRef { rows: Arc::clone(&back), row: k });
            }
            *c.entering.borrow_mut() = Some(back);
            d.scatter_ranges(g, &single(at.iter().copied()), &d.zeros(at.len(), g.cols()).map_err(error)?).map_err(error)?;
        }
        if let Some(probes) = self.probes.get(&node) {
            let recorded = self.recorded.as_ref().ok_or_else(|| error("a cut outside a plan"))?;
            // The donor's value takes the carried cotangent and the base's its negation (a product
            // by −1, exact), in float64, rounded once into the cotangent's storage.
            let wide = wide(d)?;
            let mut taken: BTreeMap<usize, RowRef> = BTreeMap::new();
            let mut refs = Vec::with_capacity(probes.len());
            for (_, cut, _) in probes {
                if !taken.contains_key(cut) {
                    let v = recorded.borrow_mut().get_mut(cut).and_then(|r| r.carried.pop_front()).ok_or_else(|| error("a cut's site reversed before its read"))?;
                    taken.insert(*cut, v);
                }
                refs.push(Some(taken.get(cut).ok_or_else(|| error("a cut's carried cotangent"))?.clone()));
            }
            let carried = gathered(&wide, &refs, g.cols())?;
            let signs: Vec<f64> = probes.iter().flat_map(|(_, _, role)| std::iter::repeat_n(if *role == 1 { 1.0 } else { -1.0 }, g.cols())).collect();
            let signs = wide.upload_vec(probes.len(), g.cols(), signs).map_err(error)?;
            let mut added = wide.empty(probes.len(), g.cols()).map_err(error)?;
            wide.hadamard(&mut added, &carried, &signs, false).map_err(error)?;
            add_row_tensor(d, g, &probes.iter().map(|p| p.0).collect::<Vec<_>>(), &d.convert(&added).map_err(error)?)?;
        }
        let Some(p) = self.nodes.get(&node) else { return Ok(()) };
        let read = d.gather_ranges(g, &single(p.rows.iter().copied())).map_err(error)?;
        let mut base = d.empty(p.rows.len(), g.cols()).map_err(error)?;
        d.hadamard(&mut base, &read, &p.keep, false).map_err(error)?;
        let mut part = d.empty(p.rows.len(), g.cols()).map_err(error)?;
        d.hadamard(&mut part, &read, &p.take, false).map_err(error)?;
        for round in rounds(&p.rows) {
            let written = picked(d, &base, &round)?;
            d.scatter_ranges(g, &single(round.iter().map(|i| p.rows[*i])), written.as_ref().unwrap_or(&base)).map_err(error)?;
        }
        for round in rounds(&p.sources) {
            let sources = single(round.iter().map(|i| p.sources[*i]));
            let mut total = d.gather_ranges(g, &sources).map_err(error)?;
            let added = picked(d, &part, &round)?;
            d.axpy(&mut total, 1.0, added.as_ref().unwrap_or(&part)).map_err(error)?;
            d.scatter_ranges(g, &sources, &total).map_err(error)?;
        }
        Ok(())
    }
}

/// A block engine: one model's blocks (block `2l` is layer `l`'s attention, `2l + 1` its MLP) run
/// on rows of one stream buffer in place. A call names its rows as ranges of the buffer, each one
/// sequence's positions `0..T` (`tokens` holds each range's sequence); attention is causal within
/// each range. The forward pass replaces each range's rows by the stream after the block (after
/// the last block, the final normed stream); block 0 reads the tokens. `edits`, when given, patch
/// the values of the nodes they name (of the model's flat program; the call's rows in range order)
/// as soon as each is computed ([`Edits::apply`]). The reverse pass replaces the rows' cotangent by
/// the cotangent of the stream entering the block, transposes the edits on those nodes' cotangents
/// ([`Edits::transpose`]), and adds the gradient of the trainable operators the block uses into
/// `gradient`; it leaves the tape as it was, so one forward pass serves several reverse passes.
pub trait BlockEngine {
    type Tape;

    fn device(&self) -> &Device;

    /// The stream's width, and the number of blocks.
    fn width(&self) -> usize;
    fn blocks(&self) -> usize;

    /// The precision of the products.
    fn arithmetic(&self) -> Arithmetic;

    /// The prefixes of this engine's runs kept across scorings ([`PrefixStore`]): `M`'s, as an
    /// [`Interchange`] keeps them; none by default.
    fn prefixes(&self) -> Option<&RefCell<PrefixStore>> {
        None
    }

    /// Where the model holds each read variable's value ([`values`]): the nodes its edits name.
    fn values(&self) -> &[Value];

    /// Where the model applies edits of parts, and the parts ([`PartSites`]); none by default.
    fn part_sites(&self) -> Option<&PartSites> {
        None
    }

    fn forward(
        &self,
        block: usize,
        stream: &mut Tensor,
        ranges: &[Range<usize>],
        tokens: &[&[u32]],
        edits: Option<&Edits>,
        keep: bool,
    ) -> Result<Option<Self::Tape>, String> {
        self.forward_before(block, stream, (ranges, &vec![Vec::new(); ranges.len()]), tokens, edits, keep)
    }

    /// [`BlockEngine::forward`] where a range may hold its sequence's rows from a position on (a
    /// suffix lane, `Plan::range`): `before`, per range, the call's rows (laid out one range after
    /// another) holding the keys of its positions before its first row (`Plan::before`), empty
    /// for a whole sequence.
    fn forward_before(
        &self,
        block: usize,
        stream: &mut Tensor,
        call: (&[Range<usize>], &[Vec<Range<usize>>]),
        tokens: &[&[u32]],
        edits: Option<&Edits>,
        keep: bool,
    ) -> Result<Option<Self::Tape>, String>;

    /// The reverse of block `block` from its tape, its products in `arithmetic` (a pass may run in
    /// another precision than the forward pass that made the tape), adding `P`'s parameter
    /// gradient into `gradient`.
    fn reverse(
        &self,
        block: usize,
        tape: &Self::Tape,
        cotangent: &mut Tensor,
        ranges: &[Range<usize>],
        edits: Option<&Edits>,
        sums: (&mut BTreeMap<usize, Tensor>, Arithmetic),
    ) -> Result<(), String>;

    /// The bytes a tape holds.
    fn tape_bytes(tape: &Self::Tape) -> usize;

    /// Drop what the reverse passes of a call kept on its tape for one another (the bfloat16 copies
    /// of its values, [`DeviceTrace::release_rounded`]), once they all ran; nothing by default.
    fn release(_tape: &Self::Tape) {}

    /// The bytes of one gradient of every trainable operator (a reverse pass's sums).
    fn gradient_bytes(&self) -> Result<usize, String>;

    /// The bytes the tapes of a forward pass over `rows` stream rows may hold before its calls keep
    /// their entering rows instead (module note): the device's free memory less the gradient sums
    /// of the divergence's and the Gauss–Newton factor's reverse passes and three times the stream
    /// (its cotangent, the scored rows and their seed); unbounded on the host.
    fn tape_budget(&self, rows: usize) -> Result<usize, String> {
        let d = self.device();
        let value = match d.storage() {
            Storage::Bf16 => 2,
            Storage::F32 => 4,
            Storage::F64 => 8,
        };
        Ok(match d.memory().map_err(error)? {
            Some((free, _)) => free.saturating_sub(self.gradient_bytes()?.saturating_mul(2).saturating_add(rows.saturating_mul(self.width()).saturating_mul(value).saturating_mul(3))),
            None => usize::MAX,
        })
    }
}

/// The ranges' rows of `t` stacked in order.
fn gather(d: &Device, t: &Tensor, ranges: &[Range<usize>]) -> Result<Tensor, String> {
    d.gather_ranges(t, ranges).map_err(error)
}

/// `values`' rows written back to the ranges' rows of `t`, in order.
fn scatter(d: &Device, t: &mut Tensor, ranges: &[Range<usize>], values: &Tensor) -> Result<(), String> {
    d.scatter_ranges(t, ranges, values).map_err(error)?;
    Ok(())
}

/// The ranges (equal-length sequences `tokens`) as one family.
fn family(tokens: &[&[u32]]) -> FamilyInputs {
    let length = tokens.first().map_or(0, |t| t.len());
    let mut ids = Vec::with_capacity(tokens.len() * length);
    let (mut sequence, mut position) = (Vec::with_capacity(ids.capacity()), Vec::with_capacity(ids.capacity()));
    for (i, t) in tokens.iter().enumerate() {
        ids.extend_from_slice(t);
        sequence.extend(std::iter::repeat_n(i as u32, t.len()));
        position.extend(0..t.len() as u32);
    }
    FamilyInputs { rows: ids.len(), slots: vec![SlotValues::Tokens(ids)], layout: Some(SequenceLayout { sequence, position }) }
}

/// The family and segments of a call's rows where a range holds a sequence's rows from a position
/// on (a suffix lane, [`Plan::range`]): each range its own sequence at its positions, a segment
/// whose earlier keys are its `before` rows; `None` when every range is a whole sequence.
fn segments(ranges: &[Range<usize>], before: &[Vec<Range<usize>>], tokens: &[&[u32]]) -> Result<Option<(FamilyInputs, Vec<crate::device_attention::Segment>)>, String> {
    if ranges.iter().zip(tokens).all(|(r, t)| r.len() == t.len()) && before.iter().all(Vec::is_empty) {
        return Ok(None);
    }
    if before.len() != ranges.len() {
        return Err(error("a call's earlier keys for another number of ranges"));
    }
    let rows: usize = ranges.iter().map(ExactSizeIterator::len).sum();
    let (mut ids, mut sequence, mut position, mut out) = (Vec::with_capacity(rows), Vec::with_capacity(rows), Vec::with_capacity(rows), Vec::with_capacity(ranges.len()));
    let mut at = 0;
    for (i, (r, t)) in ranges.iter().zip(tokens).enumerate() {
        let first = t.len().checked_sub(r.len()).ok_or_else(|| error("a range longer than its sequence"))?;
        ids.extend_from_slice(&t[first..]);
        sequence.extend(std::iter::repeat_n(i as u32, r.len()));
        position.extend(first as u32..t.len() as u32);
        out.push(crate::device_attention::Segment { rows: at..at + r.len(), first, before: before[i].clone() });
        at += r.len();
    }
    Ok(Some((FamilyInputs { rows, slots: vec![SlotValues::Tokens(ids)], layout: Some(SequenceLayout { sequence, position }) }, out)))
}

/// The reference engine: the model's resident-value program, one block a span
/// ([`DeviceProgram::forward_span`], [`DeviceProgram::vjp_values_dense_edited`]) on the call's rows
/// gathered into one tensor.
impl BlockEngine for Model<'_> {
    type Tape = DeviceTrace;

    fn device(&self) -> &Device {
        self.program.device()
    }

    fn width(&self) -> usize {
        Model::width(self)
    }

    fn blocks(&self) -> usize {
        Model::blocks(self)
    }

    fn arithmetic(&self) -> Arithmetic {
        self.program.arithmetic()
    }

    fn prefixes(&self) -> Option<&RefCell<PrefixStore>> {
        self.prefixes
    }

    fn values(&self) -> &[Value] {
        &self.sites.values
    }

    fn part_sites(&self) -> Option<&PartSites> {
        Some(&self.sites.parts)
    }

    fn forward_before(
        &self,
        block: usize,
        stream: &mut Tensor,
        (ranges, before): (&[Range<usize>], &[Vec<Range<usize>>]),
        tokens: &[&[u32]],
        edits: Option<&Edits>,
        keep: bool,
    ) -> Result<Option<DeviceTrace>, String> {
        let d = self.program.device();
        let entry = if block == 0 { None } else { Some((self.entry(block), gather(d, stream, ranges)?)) };
        let edit = |n: usize, trace: &DeviceTrace| -> Result<Option<Tensor>, String> {
            match edits {
                Some(edits) if edits.changes(n) => {
                    let mut value = d.copy(trace.value(n)?).map_err(error)?;
                    edits.apply(d, n, &mut value)?;
                    edits.write(d, n, &mut value, |m| trace.value(m))?;
                    Ok(Some(value))
                }
                _ => Ok(None),
            }
        };
        let end = self.end(block);
        let trace = match segments(ranges, before, tokens)? {
            None => self.program.forward_span(&family(tokens), entry, end, edit)?,
            Some((family, segments)) => self.program.forward_span_segments(&family, entry, end, edit, Arc::new(segments))?,
        };
        scatter(d, stream, ranges, trace.value(end)?)?;
        Ok(keep.then_some(trace))
    }

    fn reverse(
        &self,
        block: usize,
        tape: &DeviceTrace,
        cotangent: &mut Tensor,
        ranges: &[Range<usize>],
        edits: Option<&Edits>,
        (gradient, arithmetic): (&mut BTreeMap<usize, Tensor>, Arithmetic),
    ) -> Result<(), String> {
        let d = self.program.device();
        let seeds = BTreeMap::from([(self.end(block), gather(d, cotangent, ranges)?)]);
        let edited = edits.map(Edits::nodes).unwrap_or_default();
        let mut keep: Vec<usize> = edited.iter().copied().collect();
        if block > 0 {
            keep.push(self.entry(block));
        }
        let mut hook = |n: usize, g: &mut Tensor| -> Result<(), String> {
            match edits {
                Some(edits) => edits.transpose(d, n, g),
                None => Ok(()),
            }
        };
        let mut nodes = self.program.vjp_values_dense_edited(tape, seeds, &keep, &self.sites.trainable[block], arithmetic, (&edited, &mut hook), gradient)?;
        let mut entering = if block > 0 {
            nodes.remove(&self.entry(block)).ok_or_else(|| error("no cotangent of a block's entering stream"))?
        } else {
            d.zeros(ranges.iter().map(ExactSizeIterator::len).sum(), cotangent.cols()).map_err(error)?
        };
        if let Some(edits) = edits {
            edits.add_entering(d, &mut entering)?;
        }
        scatter(d, cotangent, ranges, &entering)
    }

    fn tape_bytes(tape: &DeviceTrace) -> usize {
        tape.bytes()
    }

    fn release(tape: &DeviceTrace) {
        tape.release_rounded();
    }

    fn gradient_bytes(&self) -> Result<usize, String> {
        let operators: BTreeSet<usize> = self.sites.trainable.iter().flatten().copied().collect();
        operators.iter().map(|op| self.program.dense(*op).map(Tensor::bytes)).sum()
    }
}

/// One sequence's way through the blocks: run by the hybrid `explained` (per block whether `P`
/// runs it) up to block `end`, with at most one patch at row `position`: its block, its variables,
/// and the path whose values at that block are the source.
struct Path<'t> {
    tokens: &'t [u32],
    explained: &'t [bool],
    end: usize,
    position: usize,
    patch: Option<(usize, &'t [usize], usize)>,
    /// The edits (of a part, a head's removal, a cut connection's steps) and their blocks.
    edits: Vec<(usize, Edit)>,
}

/// An edit a path applies at one block: an operation on a shared site, or a cut's step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Edit {
    /// An operation at a shared site at row `at` of its path's sequence, a swap from the donor's
    /// path `source`; as a call's edit (`OpAt`), a swap from the call's row `from`.
    Op { site: SharedSite, operation: Operation, at: usize, source: usize },
    OpAt { site: SharedSite, operation: Operation, from: usize },
    /// A connection cut's record of its site's value at row `at` on the base (`role` 0) or the
    /// donor (1), and its replacement of block `block`'s read at row `at` (`Operation::Cut`).
    Probe { cut: usize, site: SharedSite, role: usize, at: usize },
    CutRead { cut: usize, block: usize, at: usize },
    /// The native weight edit `edit` ([`Patch::Weights`]) at row `at`.
    Weight { edit: usize, at: usize },
}

impl<'t> Path<'t> {
    fn patched(&self, block: usize) -> Option<(&'t [usize], usize)> {
        self.patch.filter(|(b, _, _)| *b == block).map(|(_, variables, source)| (variables, source))
    }

    fn edited(&self, block: usize) -> Option<Edit> {
        self.edits.iter().find(|(b, _)| *b == block).map(|(_, edit)| *edit)
    }

    /// The first row the path changes at `block`, if any: a patch at its position, an operation
    /// or a cut's read at its row (a cut's probe records a value and changes none).
    fn first_change(&self, block: usize) -> Option<usize> {
        let patched = self.patched(block).map(|_| self.position);
        let edited = self.edits.iter().filter(|(b, _)| *b == block).filter_map(|(_, edit)| match edit {
            Edit::Op { at, .. } | Edit::CutRead { at, .. } | Edit::Weight { at, .. } => Some(*at),
            Edit::Probe { .. } => None,
            Edit::OpAt { .. } => Some(0),
        });
        patched.into_iter().chain(edited).min()
    }

    /// Whether the path changes no row at any block (a clean experiment's).
    fn clean(&self) -> bool {
        self.patch.is_none() && self.edits.is_empty()
    }
}

/// One lane of rows in the stream buffer: the blocks `start..end` of path `path` (whose blocks
/// before `start` are its parent lane's, copied at `start`).
struct Lane {
    path: usize,
    start: usize,
    end: usize,
    parent: Option<usize>,
    rows: Range<usize>,
    /// For a suffix lane (module note), the path whose rows before `t₀` it shares at every block it
    /// runs, and `t₀` (its own path's position).
    prefix: Option<(usize, usize)>,
}

/// The lanes of a set of paths, sharing prefixes: a path forks from a lane of the same sequence at
/// the first block where their hybrids differ or either is patched (the stream entering that block
/// is the same in both), by one copy of the rows; a path the lane holds whole (no patch, the same
/// hybrid to its end) takes no lane of its own. `holder[p]` is path `p`'s last lane.
struct Plan<'t> {
    paths: Vec<Path<'t>>,
    lanes: Vec<Lane>,
    holder: Vec<usize>,
    length: usize,
    /// Per connection cut the values its probes record ([`Edits::write`]).
    recorded: Records,
}

impl<'t> Plan<'t> {
    fn new(paths: Vec<Path<'t>>, length: usize) -> Self {
        // The longest paths first; among them a clean path, then the later positions first, so a path
        // forks from one whose rows before its position are its own (`Lane::prefix`).
        let mut order: Vec<usize> = (0..paths.len()).collect();
        order.sort_by_key(|p| (std::cmp::Reverse(paths[*p].end), !paths[*p].clean(), std::cmp::Reverse(paths[*p].position)));
        let mut lanes: Vec<Lane> = Vec::new();
        let mut holder = vec![usize::MAX; paths.len()];
        for p in order {
            let path = &paths[p];
            // The longest shared prefix with an existing lane's path.
            let mut best: Option<(usize, usize)> = None;
            for (l, lane) in lanes.iter().enumerate() {
                let other = &paths[lane.path];
                if other.tokens != path.tokens {
                    continue;
                }
                let limit = other.end.min(path.end);
                let shared = (0..limit).find(|&b| other.explained[b] != path.explained[b] || other.patched(b).is_some() || path.patched(b).is_some() || other.edited(b).is_some() || path.edited(b).is_some()).unwrap_or(limit);
                if best.is_none_or(|(_, k)| shared > k) {
                    best = Some((l, shared));
                }
            }
            match best {
                Some((l, shared)) if shared == path.end => holder[p] = l,
                found => {
                    // The lane whose rows hold the stream entering the fork block.
                    let parent = found.filter(|(_, k)| *k > 0).map(|(mut l, k)| {
                        while lanes[l].start > k {
                            l = lanes[l].parent.unwrap_or(l);
                        }
                        (l, k)
                    });
                    let start = parent.map_or(0, |(_, k)| k);
                    let rows = lanes.len() * length..(lanes.len() + 1) * length;
                    holder[p] = lanes.len();
                    lanes.push(Lane { path: p, start, end: path.end, parent: parent.map(|(l, _)| l), rows, prefix: None });
                }
            }
        }
        // A forked lane shares its rows before t₀ with the path it forked from while both run the
        // same hybrid, t₀ the first row either changes at the blocks the lane runs (every edit acts
        // from a row on; a donor changes none, so its rows before its twin's first change are its
        // twin's); that path's lane at each block is an earlier lane.
        for l in 0..lanes.len() {
            let (lane, path) = (&lanes[l], &paths[lanes[l].path]);
            let Some(parent) = lane.parent else { continue };
            let twin = lanes[parent].path;
            let other = &paths[twin];
            if other.tokens != path.tokens || other.end < lane.end || (lane.start..lane.end).any(|b| other.explained[b] != path.explained[b]) {
                continue;
            }
            let t0 = (lane.start..lane.end).flat_map(|b| [path.first_change(b), other.first_change(b)]).flatten().min().unwrap_or(length);
            if t0 > 0 && t0 < length {
                lanes[l].prefix = Some((twin, t0));
            }
        }
        Self { paths, lanes, holder, length, recorded: std::rc::Rc::new(RefCell::new(BTreeMap::new())) }
    }

    /// Lane `l`'s rows a call at `block` takes: from its position on for a suffix lane (module
    /// note), else all.
    fn range(&self, l: usize) -> Range<usize> {
        let lane = &self.lanes[l];
        match lane.prefix {
            Some((_, t0)) => lane.rows.start + t0..lane.rows.end,
            None => lane.rows.clone(),
        }
    }

    /// The rows of a call at `block` over `lanes` holding positions `0..upto` of lane `l`'s
    /// sequence, in position order: a suffix lane's before its position from its twin's lane (in
    /// the same call: it runs the block alike), its own after.
    fn keys(&self, block: usize, lanes: &[usize], l: usize, upto: usize) -> Result<Vec<Range<usize>>, String> {
        let first = self.range(l).start - self.lanes[l].rows.start;
        let mut out = match self.lanes[l].prefix {
            Some((twin, _)) if first > 0 && upto > 0 => self.keys(block, lanes, self.lane(twin, block), first.min(upto))?,
            _ => Vec::new(),
        };
        if upto > first {
            let at = self.row_of(lanes, l, first)?;
            out.push(at..at + upto - first);
        }
        Ok(out)
    }

    /// Per lane of a call at `block` over `lanes`, the call's rows holding the keys of its positions
    /// before its first row (none for a whole lane).
    fn before(&self, block: usize, lanes: &[usize]) -> Result<Vec<Vec<Range<usize>>>, String> {
        lanes
            .iter()
            .map(|&l| match self.lanes[l].prefix {
                Some((twin, t0)) => self.keys(block, lanes, self.lane(twin, block), t0),
                None => Ok(Vec::new()),
            })
            .collect()
    }

    /// The row of a call at `block` over `lanes` (each taking [`Plan::range`]) holding row `row` of
    /// lane `l`'s sequence.
    fn row_of(&self, lanes: &[usize], l: usize, row: usize) -> Result<usize, String> {
        let mut offset = 0;
        for &x in lanes {
            let range = self.range(x);
            if x == l {
                let first = range.start - self.lanes[x].rows.start;
                if row < first || row >= self.length {
                    return Err(error("a row before its lane's rows in a call"));
                }
                return Ok(offset + row - first);
            }
            offset += range.len();
        }
        Err(error("a lane outside its call"))
    }

    /// After block `block`, each suffix lane running it takes its rows before its position from
    /// its twin path's lane, in lane order (a twin's own rows are complete first).
    fn copy_prefixes(&self, d: &Device, stream: &mut Tensor, block: usize) -> Result<(), String> {
        for lane in self.lanes.iter().filter(|l| (l.start..l.end).contains(&block)) {
            if let Some((twin, t0)) = lane.prefix {
                let from = self.lanes[self.lane(twin, block)].rows.start;
                d.copy_rows_within(stream, lane.rows.start, from, t0).map_err(error)?;
            }
        }
        Ok(())
    }

    /// Before the reverse of a call at `block` over `lanes`, each suffix lane's cotangents of its
    /// rows before its position move to its twin's lane, the later lanes first (a twin passes on
    /// what it received).
    fn return_prefixes(&self, d: &Device, cotangent: &mut Tensor, block: usize, lanes: &[usize]) -> Result<(), String> {
        let mut suffix: Vec<usize> = lanes.iter().copied().filter(|l| self.lanes[*l].prefix.is_some()).collect();
        suffix.sort_unstable();
        for l in suffix.into_iter().rev() {
            let Some((twin, t0)) = self.lanes[l].prefix else { continue };
            let (at, to) = (self.lanes[l].rows.start, self.lanes[self.lane(twin, block)].rows.start);
            d.axpy_rows_within(cotangent, to, 1.0, at, t0).map_err(error)?;
            d.set_rows(cotangent, at, &d.zeros(t0, cotangent.cols()).map_err(error)?).map_err(error)?;
        }
        Ok(())
    }

    /// The lane holding path `p`'s rows at block `block`.
    fn lane(&self, p: usize, block: usize) -> usize {
        let mut l = self.holder[p];
        while self.lanes[l].start > block {
            l = self.lanes[l].parent.unwrap_or(l);
        }
        l
    }

    /// The patches applied at `block`, as rows of a call over `lanes` (in order): the patched row,
    /// the source's row, and the variables.
    /// Whether the reverse passes through block `block` of lane `l` reach anything they
    /// differentiate: a block at or below it on the lane's rows (its own from its start, then its
    /// parent's below the fork, and a suffix lane's twin's, which hold its rows before its
    /// position) that `P` runs or a patch or an edit acts on. Where none does, the cotangent there
    /// only flows through `M`'s blocks, which hold nothing trainable, to the stream's entry, so
    /// [`run`] keeps no tape for the call and [`run_reverse`] does not reverse it.
    fn reaches_trainable(&self, l: usize, block: usize) -> bool {
        let lane = &self.lanes[l];
        let path = &self.paths[lane.path];
        let top = block.min(lane.end.saturating_sub(1));
        if (lane.start..=top).any(|c| path.explained[c] || path.patched(c).is_some() || path.edits.iter().any(|(b, _)| *b == c)) {
            return true;
        }
        if let Some((twin, _)) = lane.prefix
            && self.reaches_trainable(self.lane(twin, top), top)
        {
            return true;
        }
        match lane.parent {
            Some(parent) if lane.start > 0 => self.reaches_trainable(parent, lane.start - 1),
            _ => false,
        }
    }

    /// Per lane, the last block of its prefix [`run`] may keep and restore ([`PrefixStore`]): a whole
    /// lane (from block 0, no parent, no twin) through the blocks before anything trainable reaches
    /// it (`Plan::reaches_trainable`: `M` runs them, no patch or edit acts there), and before the
    /// first block another lane reads its rows at (a lane forking from it, a suffix lane's twin, a
    /// patch's source, an operation's donor), which needs them there. None for any other lane.
    fn prefix_ends(&self) -> Vec<Option<usize>> {
        let mut ends: Vec<Option<usize>> = (0..self.lanes.len())
            .map(|l| {
                let lane = &self.lanes[l];
                if lane.start != 0 || lane.parent.is_some() || lane.prefix.is_some() {
                    return None;
                }
                (lane.start..lane.end).take_while(|&b| !self.reaches_trainable(l, b)).last()
            })
            .collect();
        // A lane read at block `c` keeps its prefix only through block c − 1.
        let mut before = |l: usize, c: usize| {
            ends[l] = match (ends[l], c.checked_sub(1)) {
                (Some(k), Some(limit)) => Some(k.min(limit)),
                _ => None,
            };
        };
        for lane in &self.lanes {
            if let Some(parent) = lane.parent {
                before(parent, lane.start);
            }
            let path = &self.paths[lane.path];
            for c in lane.start..lane.end {
                if let Some((twin, _)) = lane.prefix {
                    before(self.lane(twin, c), c);
                }
                if let Some((_, source)) = path.patched(c) {
                    before(self.lane(source, c), c);
                }
                for (_, edit) in path.edits.iter().filter(|(b, _)| *b == c) {
                    // Only a swap reads a donor; another operation's source is `usize::MAX`.
                    if let Edit::Op { operation: Operation::Swap, source, .. } = edit {
                        before(self.lane(*source, c), c);
                    }
                }
            }
        }
        ends
    }

    fn patches(&self, block: usize, lanes: &[usize]) -> Result<Vec<(usize, usize, &'t [usize])>, String> {
        let mut out = Vec::new();
        for &l in lanes {
            let path = &self.paths[self.lanes[l].path];
            if let Some((variables, source)) = path.patched(block) {
                let row = self.row_of(lanes, l, path.position)?;
                let source = self.row_of(lanes, self.lane(source, block), path.position)?;
                out.push((row, source, variables));
            }
        }
        Ok(out)
    }

    /// The edits applied at `block` (of parts, heads' removals), as rows of a call over `lanes`
    /// (in order): the edited row and the edit.
    fn writes(&self, block: usize, lanes: &[usize]) -> Result<Vec<(usize, Edit)>, String> {
        let mut out = Vec::new();
        for &l in lanes {
            let path = &self.paths[self.lanes[l].path];
            for (_, edit) in path.edits.iter().filter(|(b, _)| *b == block) {
                let at = match edit {
                    Edit::Op { at, .. } | Edit::Probe { at, .. } | Edit::CutRead { at, .. } | Edit::Weight { at, .. } => *at,
                    _ => path.position,
                };
                let edit = match edit {
                    Edit::Op { site, operation, at, source } => {
                        let from = match operation {
                            Operation::Swap => self.row_of(lanes, self.lane(*source, block), *at)?,
                            _ => self.row_of(lanes, l, *at)?,
                        };
                        Edit::OpAt { site: *site, operation: *operation, from }
                    }
                    other => *other,
                };
                out.push((self.row_of(lanes, l, at)?, edit));
            }
        }
        Ok(out)
    }
}

/// What a call of a forward pass keeps for the reverse passes (module note): its tape, or the rows
/// of the stream entering it (none at block 0, which reads the tokens).
enum Kept<T> {
    Tape(T),
    Entering(Option<Tensor>),
}

/// One engine call of a forward pass: its block, side (0 for `P`, 1 for `M`), lanes, edits, and
/// what it keeps for the reverse passes (nothing when none follows).
struct Call<T> {
    block: usize,
    side: usize,
    lanes: Vec<usize>,
    edits: Option<Edits>,
    kept: Option<Kept<T>>,
}

/// The edits of the call at block `block` over `lanes` run by `engine` ([`Plan::patches`] at its
/// sites), none when nothing there is patched.
fn edits<E: BlockEngine>(engine: &E, plan: &Plan, block: usize, lanes: &[usize]) -> Result<Option<Edits>, String> {
    let (patches, writes) = (plan.patches(block, lanes)?, plan.writes(block, lanes)?);
    if patches.is_empty() && writes.is_empty() {
        return Ok(None);
    }
    let mut out = Edits::with_parts(engine.device(), &patches, engine.values(), &writes, engine.part_sites())?;
    out.recorded = Some(std::rc::Rc::clone(&plan.recorded));
    Ok(Some(out))
}

/// Run `plan` on the engines `[P, M]` (module note): block by block, the lanes forking there copy
/// their parent's rows, then per side one call over the lanes running that side, the patches
/// applied as its edits from the source rows of the same call. Returns the stream buffer and
/// the calls. With `keep`, each call keeps its tape while the kept bytes, this call's tape and one
/// block run again with its reverse (each estimated by the largest tape so far) stay below `P`'s
/// [`BlockEngine::tape_budget`], and the rows entering it otherwise.
fn run<E: BlockEngine>(engines: [&E; 2], plan: &Plan, keep: bool) -> Result<(Tensor, Vec<Call<E::Tape>>), String> {
    let (d, width, blocks) = (engines[0].device(), engines[0].width(), engines[0].blocks());
    let mut stream = d.zeros(plan.lanes.len() * plan.length, width).map_err(error)?;
    let mut calls = Vec::new();
    let budget = if keep { Some(engines[0].tape_budget(stream.rows())?) } else { None };
    let (mut kept, mut largest) = (0usize, 0usize);
    // `M`'s prefixes kept across scorings (`PrefixStore`), where `P` and `M` are two engines (not a
    // run of `M` alone, its targets') and the stream holds f32: a lane whose prefix is kept skips
    // its calls through the prefix's last block and takes the kept rows after it.
    let store = if std::ptr::eq(engines[0], engines[1]) { None } else { engines[1].prefixes() };
    let ends = if store.is_some() { plan.prefix_ends() } else { vec![None; plan.lanes.len()] };
    let key = |l: usize, k: usize| (plan.paths[plan.lanes[l].path].tokens.to_vec(), k);
    let restored: Vec<bool> = match store {
        Some(store) => {
            store.borrow_mut().settle(d)?;
            let store = store.borrow();
            (0..plan.lanes.len()).map(|l| ends[l].is_some_and(|k| store.rows.contains_key(&key(l, k)))).collect()
        }
        None => vec![false; plan.lanes.len()],
    };
    for b in 0..blocks {
        for lane in plan.lanes.iter().filter(|l| l.start == b) {
            if let Some(parent) = lane.parent {
                let rows = plan.lanes[parent].rows.clone();
                d.copy_rows_within(&mut stream, lane.rows.start, rows.start, rows.len()).map_err(error)?;
            }
        }
        for side in 0..2 {
            let lanes: Vec<usize> = (0..plan.lanes.len())
                .filter(|&l| (plan.lanes[l].start..plan.lanes[l].end).contains(&b) && plan.paths[plan.lanes[l].path].explained[b] == (side == 0))
                .filter(|&l| !(restored[l] && ends[l].is_some_and(|k| b <= k)))
                .collect();
            if lanes.is_empty() {
                continue;
            }
            let ranges: Vec<Range<usize>> = lanes.iter().map(|l| plan.range(*l)).collect();
            let before = plan.before(b, &lanes)?;
            let tokens: Vec<&[u32]> = lanes.iter().map(|l| plan.paths[plan.lanes[*l].path].tokens).collect();
            let edits = edits(engines[side], plan, b, &lanes)?;
            // An `M` call none of whose lanes' cotangents reach anything trainable is not reversed
            // (`Plan::reaches_trainable`): its forward keeps nothing.
            let reversed = side == 0 || edits.is_some() || lanes.iter().any(|&l| plan.reaches_trainable(l, b));
            let kept_here = match budget {
                None => {
                    engines[side].forward_before(b, &mut stream, (&ranges, &before), &tokens, edits.as_ref(), false)?;
                    None
                }
                Some(_) if !reversed => {
                    engines[side].forward_before(b, &mut stream, (&ranges, &before), &tokens, edits.as_ref(), false)?;
                    None
                }
                Some(budget) if kept.saturating_add(largest.saturating_mul(3)) < budget => {
                    let tape = engines[side].forward_before(b, &mut stream, (&ranges, &before), &tokens, edits.as_ref(), true)?.ok_or_else(|| error("a call kept no tape"))?;
                    let bytes = E::tape_bytes(&tape);
                    kept = kept.saturating_add(bytes);
                    largest = largest.max(bytes);
                    Some(Kept::Tape(tape))
                }
                Some(_) => {
                    let entering = if b == 0 { None } else { Some(gather(d, &stream, &ranges)?) };
                    kept = kept.saturating_add(entering.as_ref().map_or(0, Tensor::bytes));
                    engines[side].forward_before(b, &mut stream, (&ranges, &before), &tokens, edits.as_ref(), false)?;
                    Some(Kept::Entering(entering))
                }
            };
            calls.push(Call { block: b, side, lanes, edits, kept: kept_here });
        }
        if let Some(store) = store {
            for l in (0..plan.lanes.len()).filter(|&l| ends[l] == Some(b)) {
                let rows = plan.lanes[l].rows.clone();
                if restored[l] {
                    let store = store.borrow();
                    let held = match &**store.rows.get(&key(l, b)).ok_or_else(|| error("a kept prefix gone"))? {
                        KeptValues::Single(v) => d.upload_f32_overlapped(rows.len(), width, v),
                        KeptValues::Double(v) => d.upload_vec(rows.len(), width, v.clone()),
                    }
                    .map_err(error)?;
                    d.set_rows(&mut stream, rows.start, &held).map_err(error)?;
                } else {
                    let taken = d.rows_of(&stream, rows.start, rows.len()).map_err(error)?;
                    store.borrow_mut().pending.push((key(l, b), taken));
                }
            }
        }
        plan.copy_prefixes(d, &mut stream, b)?;
    }
    Ok((stream, calls))
}

/// The products' precision of the reverse passes (the data term's gradient and the Gauss–Newton
/// factor): bfloat16 on CUDA in f32 storage (whose bfloat16 tensor cores run at least twice its
/// f32 rate), else the forward's `arithmetic`. The forward passes and the scores, the objective and
/// the line step's measurements, keep the forward's arithmetic. Each reverse pass's result is one
/// Monte Carlo draw (the gradient at one weight sample, and the factor whose square estimates the
/// Gauss–Newton diagonal); bfloat16 operands move their entries by about 0.55% (vpd4l, against the
/// float64 reference). Fit A/B speed-revab (vpd4l, N = 2^20, RTX 4090, seeds 1-2): F after epochs
/// 0/1/2 at 44.10/29.00/24.15 and 44.78/29.78/23.21 M bits with the data term's reverse in
/// bfloat16, against 45.04/30.75/24.99 and 43.37/29.51/25.03 in f32, and 0.1231 s per step against
/// 0.1315.
pub(crate) fn factor_arithmetic(d: &Device, arithmetic: Arithmetic) -> Arithmetic {
    if d.storage() == Storage::F32 && d.with_storage(Storage::Bf16).is_ok() { Arithmetic::Bf16 } else { arithmetic }
}

/// One reverse pass of [`run_reverse`]: the stream buffer's cotangent (rows of the paths'
/// outputs), the parameter gradient it adds into, and its products' arithmetic.
struct Pass<'a> {
    cotangent: Tensor,
    gradient: &'a mut BTreeMap<usize, Tensor>,
    arithmetic: Arithmetic,
}

/// The reverses of [`run`] from the cotangents of `passes` together: block by block backwards,
/// each call's tape had once (recomputed from its entering rows when it kept those) and reversed
/// for every pass in turn with the patches' transposed edits (each source's part added into the
/// source's row of the same call), then each fork's rows added into its parent's in every pass.
/// Each pass adds `P`'s parameter gradient into its own map, by the same operations in the same
/// order as a reverse of its own, so each result is that pass's alone bit for bit, with one
/// recomputation of a call's tape for all of them; the calls' tapes stay.
fn run_reverse<E: BlockEngine>(engines: [&E; 2], plan: &Plan, calls: &[Call<E::Tape>], passes: &mut [Pass<'_>]) -> Result<(), String> {
    let d = engines[0].device();
    let width = passes.first().map(|p| p.cotangent.cols()).ok_or_else(|| error("a reverse pass of no cotangent"))?;
    for (index, call) in calls.iter().enumerate().rev() {
        let b = call.block;
        let ranges: Vec<Range<usize>> = call.lanes.iter().map(|l| plan.range(*l)).collect();
        let recomputed;
        // A call that kept nothing is one `run` found the reverse cannot reach anything trainable
        // through; its forks still return their rows below.
        let tape = match call.kept.as_ref() {
            None => None,
            Some(Kept::Tape(tape)) => Some(tape),
            // The block's forward again, from the rows that entered it (laid out one lane after
            // another) with the same patches, keeping its tape.
            Some(Kept::Entering(entering)) => {
                let mut local = Vec::with_capacity(ranges.len());
                let mut at = 0;
                for r in &ranges {
                    local.push(at..at + r.len());
                    at += r.len();
                }
                let mut rows = match entering {
                    Some(rows) => d.copy(rows).map_err(error)?,
                    None if b == 0 => d.zeros(at, width).map_err(error)?,
                    None => return Err(error("a call past the first block without its entering rows")),
                };
                let tokens: Vec<&[u32]> = call.lanes.iter().map(|l| plan.paths[plan.lanes[*l].path].tokens).collect();
                recomputed = engines[call.side].forward_before(b, &mut rows, (&local, &plan.before(b, &call.lanes)?), &tokens, call.edits.as_ref(), true)?.ok_or_else(|| error("a call kept no tape"))?;
                Some(&recomputed)
            }
        };
        for pass in passes.iter_mut() {
            plan.return_prefixes(d, &mut pass.cotangent, b, &call.lanes)?;
            if let Some(tape) = tape {
                engines[call.side].reverse(b, tape, &mut pass.cotangent, &ranges, call.edits.as_ref(), (&mut *pass.gradient, pass.arithmetic))?;
            }
        }
        if let Some(tape) = tape {
            E::release(tape);
        }
        // Once both sides of the block are reversed, the forks made there return their rows, the
        // later lanes first (a lane forked from a lane forked at the same block returns through it).
        if index == 0 || calls[index - 1].block != b {
            for pass in passes.iter_mut() {
                for lane in plan.lanes.iter().rev().filter(|l| l.start == b) {
                    if let Some(parent) = lane.parent {
                        let rows = plan.lanes[parent].rows.clone();
                        if lane.rows.len() != rows.len() {
                            return Err(error("a lane returns rows to a parent of another length"));
                        }
                        d.axpy_rows_within(&mut pass.cotangent, rows.start, 1.0, lane.rows.start, rows.len()).map_err(error)?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Refuse an experiment outside `batch` or with a hybrid not of `blocks` blocks.
fn check(e: &Experiment, batch: &Batch, blocks: usize) -> Result<(), String> {
    if e.base >= batch.base.len() || e.source >= batch.source.len() || e.explained.len() != blocks || e.position >= batch.length {
        return Err(error("an experiment outside the batch or its blocks"));
    }
    Ok(())
}

/// `M`'s compact statistics for a batch's experiments, per experiment from its position on: the
/// targets every score of `P` on them is measured against ([`targets`]). They do not depend on `P`;
/// a fit makes them on the device whenever it scores a batch.
pub struct Targets {
    rows: Vec<Target>,
}

/// The paths of `experiments` on `batch` over the read variables of `values` (their blocks), every
/// block run by `M` when `native`, else by each experiment's hybrid: per patched experiment its
/// source's path (to its patched block) and its base's path; per experiment the index of its base's
/// path.
fn paths<'t>(batch: &'t Batch, experiments: &'t [Experiment], values: &[Value], heads: &[usize], native: Option<&'t [bool]>, blocks: usize) -> Result<(Vec<Path<'t>>, Vec<usize>), String> {
    let (mut paths, mut bases) = (Vec::with_capacity(2 * experiments.len()), Vec::with_capacity(experiments.len()));
    let cuts = &mut 0usize;
    for e in experiments {
        check(e, batch, blocks)?;
        let explained = native.unwrap_or(&e.explained);
        let (patch, edits) = match (&e.patch, e.block(values, heads)?) {
            (Some(Patch::Ops { ops, .. }), Some(_)) => {
                // The donor runs to the last block a swap or a cut reads, its probes recording each
                // cut site's donor value.
                let donor = ops.iter().filter(|o| matches!(o.operation, Operation::Swap | Operation::Cut { .. })).filter_map(|o| o.site.block(heads)).max();
                let source = donor.map(|b| {
                    paths.push(Path { tokens: &batch.source[e.source], explained, end: b + 1, position: 0, patch: None, edits: Vec::new() });
                    paths.len() - 1
                });
                let mut edits = Vec::new();
                for o in ops {
                    let block = o.site.block(heads).ok_or_else(|| error("an operation at an unknown head"))?;
                    let rows = if o.onward { e.position..batch.length } else { e.position..e.position + 1 };
                    for at in rows {
                        match o.operation {
                            Operation::Cut { to } => {
                                if to <= block || to >= blocks {
                                    return Err(error("a cut to a block not after its site's"));
                                }
                                let cut = *cuts;
                                *cuts += 1;
                                let donor = source.ok_or_else(|| error("a cut without its donor"))?;
                                paths[donor].edits.push((block, Edit::Probe { cut, site: o.site, role: 1, at }));
                                edits.push((block, Edit::Probe { cut, site: o.site, role: 0, at }));
                                edits.push((to, Edit::CutRead { cut, block: to, at }));
                            }
                            _ => edits.push((block, Edit::Op { site: o.site, operation: o.operation, at, source: source.unwrap_or(usize::MAX) })),
                        }
                    }
                }
                (None, edits)
            }
            (Some(Patch::Weights { edit, block }), Some(_)) => (None, (e.position..batch.length).map(|at| (*block, Edit::Weight { edit: *edit, at })).collect()),
            (Some(patch), Some(block)) => {
                paths.push(Path { tokens: &batch.source[e.source], explained, end: block + 1, position: 0, patch: None, edits: Vec::new() });
                (Some((block, patch.variables(), paths.len() - 1)), Vec::new())
            }
            _ => (None, Vec::new()),
        };
        bases.push(paths.len());
        paths.push(Path { tokens: &batch.base[e.base], explained, end: blocks, position: e.position, patch, edits });
    }
    Ok((paths, bases))
}

/// Each head's block in `engine`'s edit sites (none without them).
fn engine_heads<E: BlockEngine>(engine: &E) -> &[usize] {
    engine.part_sites().map_or(&[], |s| &s.head_blocks)
}

/// Per experiment, the rows of `stream` its base's path ends on, from the experiment's position on.
fn outputs(plan: &Plan, bases: &[usize], experiments: &[Experiment]) -> Vec<Range<usize>> {
    bases.iter().zip(experiments).map(|(p, e)| plan.lanes[plan.holder[*p]].rows.start + e.position..plan.lanes[plan.holder[*p]].rows.end).collect()
}

/// `M`'s targets for `experiments` on `batch`: `M` on every base and source, the patched runs
/// forking from their base's clean run at the patched block (`Plan`).
pub fn targets<E: BlockEngine>(m: &E, head: &FixedHead, batch: &Batch, experiments: &[Experiment]) -> Result<Targets, String> {
    let d = m.device();
    let blocks = m.blocks();
    let native = vec![false; blocks];
    let (paths, bases) = paths(batch, experiments, m.values(), engine_heads(m), Some(&native), blocks)?;
    let plan = Plan::new(paths, batch.length);
    let (stream, _) = run([m, m], &plan, false)?;
    let rows = outputs(&plan, &bases, experiments);
    let hidden = gather(d, &stream, &rows)?;
    let all = head.target(d, &hidden, m.arithmetic())?;
    let mut out = Vec::with_capacity(experiments.len());
    let mut at = 0;
    for r in &rows {
        let mu = d.rows_of(&all.mu, at, r.len()).map_err(error)?;
        out.push(Target { mu: Arc::new(mu), entropy: all.entropy[at..at + r.len()].to_vec(), head: Arc::clone(&head.head), scored: None });
        at += r.len();
    }
    Ok(Targets { rows: out })
}

/// The exact identity of a batch's experiments, which alone decides `M`'s targets for them: the
/// base and source tokens and every experiment.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Identity {
    base: Vec<Vec<u32>>,
    source: Vec<Vec<u32>>,
    experiments: Vec<Experiment>,
}

/// `M`'s targets of one batch's experiments held on the host, each value as the device held it
/// (f32 values for f32 and bfloat16 storage, float64 for float64), so that putting them back
/// ([`HostTargets::restore`]) gives the same targets bit for bit.
struct HostTargets {
    rows: Vec<KeptTarget>,
}

struct KeptTarget {
    mu: KeptValues,
    shape: (usize, usize),
    storage: Storage,
    entropy: Vec<f64>,
    head: Arc<Head>,
    scored: Option<Vec<bool>>,
}

enum KeptValues {
    Single(Vec<f32>),
    Double(Vec<f64>),
}

impl HostTargets {
    /// The host bytes `targets` take when kept, with their identity's tokens.
    fn bytes(targets: &Targets, identity: &Identity) -> usize {
        let tokens: usize = identity.base.iter().chain(&identity.source).map(Vec::len).sum();
        let rows: usize = targets
            .rows
            .iter()
            .map(|t| {
                let value = if t.mu.storage() == Storage::F64 { 8 } else { 4 };
                value * t.mu.len() + 8 * t.entropy.len() + t.scored.as_ref().map_or(0, Vec::len)
            })
            .sum();
        4 * tokens + rows
    }

    /// `targets` (on the device `d`) on the host.
    fn of(d: &Device, targets: &Targets) -> Result<Self, String> {
        let rows = targets
            .rows
            .iter()
            .map(|t| {
                let shape = (t.mu.rows(), t.mu.cols());
                let storage = t.mu.storage();
                let mu = match storage {
                    Storage::F64 => KeptValues::Double(d.download(&t.mu).map_err(error)?.into_iter().collect()),
                    // f32 and bfloat16 values are f32 values, read as they are (`Device::download_f32`).
                    Storage::F32 | Storage::Bf16 => KeptValues::Single(d.download_f32(&t.mu).map_err(error)?),
                };
                Ok(KeptTarget { mu, shape, storage, entropy: t.entropy.clone(), head: Arc::clone(&t.head), scored: t.scored.clone() })
            })
            .collect::<Result<_, String>>()?;
        Ok(Self { rows })
    }

    /// The kept targets on the device `d` again, each `μ` in the storage it was made in.
    fn restore(&self, d: &Device) -> Result<Targets, String> {
        let rows = self
            .rows
            .iter()
            .map(|t| {
                let held = if d.storage() == t.storage { d.clone() } else { d.with_storage(t.storage).map_err(error)? };
                let mu = match &t.mu {
                    // Copied beside the kernels queued before it (the batch's other half's reverse).
                    KeptValues::Single(v) => held.upload_f32_overlapped(t.shape.0, t.shape.1, v),
                    KeptValues::Double(v) => held.upload_vec(t.shape.0, t.shape.1, v.clone()),
                }
                .map_err(error)?;
                Ok(Target { mu: Arc::new(mu), entropy: t.entropy.clone(), head: Arc::clone(&t.head), scored: t.scored.clone() })
            })
            .collect::<Result<_, String>>()?;
        Ok(Targets { rows })
    }
}

impl HostTargets {
    /// The kept targets as bytes (little-endian): per row its shape, storage and `μ` as held, its
    /// entropies and its scored flags. Every row's head is the interchange's, which a read puts
    /// back.
    fn encode(&self) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        let word = |out: &mut Vec<u8>, n: usize| out.extend_from_slice(&(n as u64).to_le_bytes());
        word(&mut out, self.rows.len());
        for t in &self.rows {
            word(&mut out, t.shape.0);
            word(&mut out, t.shape.1);
            out.push(match t.storage {
                Storage::F64 => 0,
                Storage::F32 => 1,
                Storage::Bf16 => 2,
            });
            match &t.mu {
                KeptValues::Single(v) => {
                    out.push(0);
                    word(&mut out, v.len());
                    v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes()));
                }
                KeptValues::Double(v) => {
                    out.push(1);
                    word(&mut out, v.len());
                    v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes()));
                }
            }
            word(&mut out, t.entropy.len());
            t.entropy.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes()));
            match &t.scored {
                Some(flags) => {
                    out.push(1);
                    word(&mut out, flags.len());
                    out.extend(flags.iter().map(|f| u8::from(*f)));
                }
                None => out.push(0),
            }
        }
        Ok(out)
    }

    /// [`HostTargets::encode`]'s bytes back, each row with `head`.
    fn decode(bytes: &[u8], head: &Arc<Head>) -> Result<Self, String> {
        let mut r = Reader { bytes, at: 0 };
        let count = r.word()?;
        let mut rows = Vec::with_capacity(count);
        for _ in 0..count {
            let shape = (r.word()?, r.word()?);
            let storage = match r.take(1)?[0] {
                0 => Storage::F64,
                1 => Storage::F32,
                2 => Storage::Bf16,
                _ => return Err(error("a kept targets file of an unknown storage")),
            };
            let kind = r.take(1)?[0];
            let n = r.word()?;
            let mu = match kind {
                0 => KeptValues::Single(r.take(4 * n)?.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
                1 => KeptValues::Double(r.doubles(n)?),
                _ => return Err(error("a kept targets file of an unknown value kind")),
            };
            let n = r.word()?;
            let entropy = r.doubles(n)?;
            let scored = match r.take(1)?[0] {
                1 => {
                    let n = r.word()?;
                    Some(r.take(n)?.iter().map(|f| *f != 0).collect())
                }
                _ => None,
            };
            rows.push(KeptTarget { mu, shape, storage, entropy, head: Arc::clone(head), scored });
        }
        if r.at != bytes.len() {
            return Err(error("a kept targets file longer than its rows"));
        }
        Ok(Self { rows })
    }
}

/// A cursor over a kept targets file's bytes.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let part = self.bytes.get(self.at..self.at + n).ok_or_else(|| error("a kept targets file ends early"))?;
        self.at += n;
        Ok(part)
    }

    fn word(&mut self) -> Result<usize, String> {
        let b = self.take(8)?;
        usize::try_from(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])).map_err(error)
    }

    fn doubles(&mut self, n: usize) -> Result<Vec<f64>, String> {
        Ok(self.take(8 * n)?.chunks_exact(8).map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect())
    }
}

/// Bytes free to this process on the file system holding `dir`, and its size.
#[cfg(unix)]
fn disk_space(dir: &std::path::Path) -> Option<(u64, u64)> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `path` is NUL-terminated and `statvfs` fills `stat` when it returns 0.
    if unsafe { libc::statvfs(path.as_ptr(), stat.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: `statvfs` returned 0, so `stat` is written.
    let stat = unsafe { stat.assume_init() };
    let bytes = |blocks: u128| u64::try_from(blocks * stat.f_frsize as u128).unwrap_or(u64::MAX);
    Some((bytes(stat.f_bavail as u128), bytes(stat.f_blocks as u128)))
}

/// Off Unix there is no free-space query here, so nothing is written to disk.
#[cfg(not(unix))]
fn disk_space(dir: &std::path::Path) -> Option<(u64, u64)> {
    std::fs::metadata(dir).ok().and(None)
}

/// Kept targets the memory budget does not admit, on local disk: one file per batch in a directory
/// of this store's own (removed with it), written while the file system keeps a tenth of its size
/// and at least 8 GiB free beside them (room for the run's checkpoints and logs); a batch that
/// does not fit is made on the device each time.
struct DiskTargets {
    dir: PathBuf,
    files: HashMap<Identity, PathBuf>,
    /// Each batch asked for after another (the scorings of a fit repeat their batches in order),
    /// the last asked for, and the read of the batch expected next, started on its own thread
    /// when the one before it was read, so a pass reads one batch while the device scores another.
    next: HashMap<Identity, Identity>,
    last: Option<Identity>,
    ahead: Option<(Identity, std::thread::JoinHandle<std::io::Result<Vec<u8>>>)>,
}

impl DiskTargets {
    fn new() -> Option<Self> {
        static STORES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = STORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("mpd-targets-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).ok()?;
        Some(Self { dir, files: HashMap::new(), next: HashMap::new(), last: None, ahead: None })
    }

    /// Records that `identity` is asked for now, after the last one.
    fn asked(&mut self, identity: &Identity) {
        if let Some(last) = self.last.replace(identity.clone()) {
            self.next.insert(last, identity.clone());
        }
    }

    /// The bytes kept for `identity`, from the read started ahead when it was the batch expected,
    /// and the read of the batch expected after it started.
    fn read(&mut self, identity: &Identity) -> Result<Option<Vec<u8>>, String> {
        let Some(path) = self.files.get(identity) else { return Ok(None) };
        let bytes = match self.ahead.take() {
            Some((expected, reading)) if &expected == identity => reading.join().map_err(|_| error("a kept targets read panicked"))?.map_err(error)?,
            _ => std::fs::read(path).map_err(error)?,
        };
        if let Some(following) = self.next.get(identity)
            && let Some(path) = self.files.get(following)
        {
            let path = path.clone();
            self.ahead = Some((following.clone(), std::thread::spawn(move || std::fs::read(path))));
        }
        Ok(Some(bytes))
    }

    /// `kept` written for `identity` when the disk has room for its `bytes`.
    fn keep(&mut self, identity: Identity, kept: &HostTargets) -> Result<(), String> {
        let bytes = kept.encode()?;
        let Some((free, total)) = disk_space(&self.dir) else { return Ok(()) };
        if free < bytes.len() as u64 + (total / 10).max(8 << 30) {
            return Ok(());
        }
        let path = self.dir.join(format!("{}.targets", self.files.len()));
        let partial = path.with_extension("partial");
        std::fs::write(&partial, &bytes).map_err(error)?;
        std::fs::rename(&partial, &path).map_err(error)?;
        self.files.insert(identity, path);
        Ok(())
    }
}

impl Drop for DiskTargets {
    fn drop(&mut self) {
        // A read still running finishes before its file goes.
        if let Some((_, reading)) = self.ahead.take()
            && reading.join().is_err()
        {
            log::warn!("interchange: a kept targets read panicked");
        }
        if let Err(e) = std::fs::remove_dir_all(&self.dir) {
            log::warn!("interchange: kept targets at {} not removed: {e}", self.dir.display());
        }
    }
}

/// The targets an [`Interchange`] keeps on the host once [`Interchange::keep_targets`] asked for
/// it, by the exact identity of each batch's experiments, each batch's bytes reserved from
/// `governor` for as long as they are held, and those the budget does not admit on local disk.
/// `M`'s prefixes kept on the host while the governor admits them: per whole lane of a scoring, its
/// stream rows after the last block `M` runs on it before a patch, an edit, a `P` block or another
/// lane's read reaches it (`Plan::prefix_ends`), by its tokens and that block. `M` is fixed, so those
/// rows are the same at every scoring of the collection; `run` restores them in place of running
/// the blocks again. The rows are kept in the stream's own precision (f32, else float64). A scoring
/// copies the rows it keeps on the device (`pending`), and the next reads them back
/// (`PrefixStore::settle`): reading each at once would wait for the device in the middle of the
/// forward pass, and the step's own read of its scores waits for the copies anyway.
pub struct PrefixStore {
    governor: MemoryGovernor,
    rows: HashMap<(Vec<u32>, usize), Governed<KeptValues>>,
    pending: Vec<((Vec<u32>, usize), Tensor)>,
}

impl PrefixStore {
    fn new(governor: &MemoryGovernor) -> Self {
        Self { governor: governor.clone(), rows: HashMap::new(), pending: Vec::new() }
    }

    /// The rows the last scoring copied, read back and kept.
    fn settle(&mut self, d: &Device) -> Result<(), String> {
        for (key, taken) in std::mem::take(&mut self.pending) {
            let values = match taken.storage() {
                Storage::F64 => KeptValues::Double(d.download(&taken).map_err(error)?.into_iter().collect()),
                Storage::F32 | Storage::Bf16 => KeptValues::Single(d.download_f32(&taken).map_err(error)?),
            };
            self.keep(key, values);
        }
        Ok(())
    }

    fn keep(&mut self, key: (Vec<u32>, usize), values: KeptValues) {
        let bytes = match &values {
            KeptValues::Single(v) => 4 * v.len(),
            KeptValues::Double(v) => 8 * v.len(),
        } + 4 * key.0.len();
        if let Ok(reservation) = self.governor.try_reserve(bytes, "interchange: M's prefix rows kept on the host") {
            self.rows.insert(key, reservation.bind(values));
        }
    }

    /// The prefixes kept, with those copied and not yet read back.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len() + self.pending.len()
    }

    /// Whether none is kept.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty() && self.pending.is_empty()
    }
}

struct TargetStore {
    governor: MemoryGovernor,
    batches: HashMap<Identity, Governed<HostTargets>>,
    disk: Option<DiskTargets>,
}

impl Targets {
    /// The targets of several lists of experiments, in order: those of the lists' concatenation.
    pub fn joined(parts: impl IntoIterator<Item = Targets>) -> Targets {
        Targets { rows: parts.into_iter().flat_map(|t| t.rows).collect() }
    }

    /// Per experiment its rows' `μ` and `Σ p log p`, on the host.
    pub fn host(&self, d: &Device) -> Result<Vec<(ndarray::Array2<f64>, Vec<f64>)>, String> {
        self.rows.iter().map(|t| Ok((d.download(&t.mu).map_err(error)?, t.entropy.clone()))).collect()
    }
}

/// The stream buffer's cotangent (`buffer` rows) from `seed` on the scored `rows` (in their order),
/// times `weight`.
fn spread(d: &Device, buffer: usize, rows: &[Range<usize>], seed: &Tensor, weight: f64) -> Result<Tensor, String> {
    let mut cotangent = d.zeros(buffer, seed.cols()).map_err(error)?;
    let mut at = 0;
    for r in rows {
        d.axpy_rows(&mut cotangent, r.start, weight, (seed, at), r.len()).map_err(error)?;
        at += r.len();
    }
    Ok(cotangent)
}

/// One batch's scores and gradients ([`evaluate`], [`evaluate_probed`]).
pub struct Evaluation {
    /// Per experiment, per base token from its position on, `KL(M_e ‖ P_e)` in bits (empty when no
    /// targets were given).
    pub bits: Vec<Vec<f64>>,
    /// The gradient of the sum of `bits` in each of `P`'s trainable operators, on the device (empty
    /// when not asked for).
    pub gradient: BTreeMap<usize, Tensor>,
    /// A draw of the Gauss–Newton factor, when asked for ([`evaluate_probed`]).
    pub factor: Option<Factor>,
    /// The forward pass's work ([`Work`]).
    pub work: Work,
}

/// What a forward pass over a batch's experiments ran: its paths, lanes and suffix lanes (module
/// note), and the block rows its calls took against those of every lane whole (lanes × blocks run
/// × the sequence length).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Work {
    pub paths: usize,
    pub lanes: usize,
    pub suffix_lanes: usize,
    pub rows: usize,
    pub whole_rows: usize,
}

/// A draw of the Gauss–Newton factor of a batch's experiments: `u = Σ_t J_tᵀ b_t` in each of `P`'s
/// trainable operators on the device, over every scored token `t` of every experiment `e`, `J_t`
/// the Jacobian of `P_e`'s logits at `t` and `b_t` the Fisher probe there
/// (`Device::fisher_probe_cotangent`): with `p_t` `P_e`'s prediction and `ξ_t` independent random
/// signs over the vocabulary, `b_t = √p_t ⊙ ξ_t − p_t (√p_t · ξ_t)`; and the number of tokens `n`.
/// With `L_t = diag √p_t − p_t √p_tᵀ`, `b_t = L_t ξ_t` and `L_t L_tᵀ = diag p_t − p_t p_tᵀ`, and the
/// tokens' signs are independent, so `E[u uᵀ] = Σ_t J_tᵀ (diag p_t − p_t p_tᵀ) J_t`: `u ⊙ u / n` is
/// an unbiased estimate of the diagonal of the data term's Gauss–Newton curvature per token (the
/// Hessian of `KL(M_e ‖ P_e)` in `P`'s logits is `P`'s softmax Fisher matrix whatever `M_e` is).
/// Each entry is `u_i = a · ξ` over all the batch's signs `ξ`, so `E[u_i⁴] = 3 c² − 2 Σ_j a_j⁴` with
/// `c = E[u_i²] = ‖a‖²`: `u_i²` estimates `c` with relative variance at most 2 whatever the
/// predictions are. A label `y_t` drawn from `p_t` (the score `p_t − e_{y_t}` in place of `b_t`) has
/// the same `E[u uᵀ]`, but its squared score estimates `p (1 − p)` for a class of probability `p`
/// with relative variance `(1 − 2p)² / (p (1 − p))`, `10⁶` at `p = 10⁻⁶`.
pub struct Factor {
    pub gradient: BTreeMap<usize, Tensor>,
    pub tokens: usize,
}

/// The Fisher probe under `key` pulled back to the final normed stream `hidden` (rows × width): per
/// row `r`, `b_r E` with `b_r` the probe's cotangent at `P`'s prediction `softmax(E h_r)` there
/// (`Device::fisher_probe_cotangent`, rows numbered from zero; `embedding` is `E`, classes ×
/// width). The logits are formed `tile` rows at a time.
pub fn fisher_probe_seed(d: &Device, hidden: &Tensor, embedding: &Tensor, tile: usize, key: u64, arithmetic: Arithmetic) -> Result<Tensor, String> {
    let (rows, classes) = (hidden.rows(), embedding.rows());
    if tile == 0 {
        return Err(error("positive tile rows required"));
    }
    // Every buffer below is written whole (products with β = 0, the tiles' rows), so none is zeroed.
    let mut seed = d.empty(rows, hidden.cols()).map_err(error)?;
    for start in (0..rows).step_by(tile) {
        let n = tile.min(rows - start);
        let h = d.rows_of(hidden, start, n).map_err(error)?;
        let mut logits = d.empty(n, classes).map_err(error)?;
        d.gemm(&mut logits, 1.0, &h, Op::N, embedding, Op::T, 0.0, arithmetic).map_err(error)?;
        d.softmax_rows(&mut logits, false).map_err(error)?;
        d.fisher_probe_cotangent(&mut logits, (key, start), None).map_err(error)?;
        let mut part = d.empty(n, hidden.cols()).map_err(error)?;
        d.gemm(&mut part, 1.0, &logits, Op::N, embedding, Op::N, 0.0, arithmetic).map_err(error)?;
        d.set_rows(&mut seed, start, &part).map_err(error)?;
    }
    Ok(seed)
}

/// `KL(M_e ‖ P_e)` per token for each of `experiments` on `batch` from its position on (module
/// note), at `P`'s current parameters, against `M`'s `targets` for them, and with `gradient` its
/// sum's gradient in `P`'s trainable operators. `M` runs here only as the hybrids' blocks it keeps.
pub fn evaluate<E: BlockEngine>(m: &E, p: &E, head: &FixedHead, batch: &Batch, targets: &Targets, experiments: &[Experiment], gradient: bool) -> Result<Evaluation, String> {
    evaluate_probed((m, p), head, (batch, experiments), Some(targets), gradient, None)
}

/// [`evaluate`] of `experiments` on `batch`, with the scores only when `targets` are given, and
/// with the probe key `probe` also a draw of the Gauss–Newton factor ([`Factor`]) under it, the
/// scored rows numbered from zero in experiment order: a second reverse pass through the same
/// forward pass, seeded at every scored token by the probe pulled back through the head. With the
/// divergence's gradient the head's one sweep of the vocabulary makes both seeds
/// (`Device::head_log_partition_probed`); without it, [`fisher_probe_seed`] sweeps for the probe.
/// Either way a key gives the same signs: the two seeds of one key differ only by rounding.
pub fn evaluate_probed<E: BlockEngine>(
    (m, p): (&E, &E),
    head: &FixedHead,
    (batch, experiments): (&Batch, &[Experiment]),
    targets: Option<&Targets>,
    gradient: bool,
    probe: Option<u64>,
) -> Result<Evaluation, String> {
    let d = p.device();
    let (blocks, length, width) = (p.blocks(), batch.length, p.width());
    let blocks_of = |values: &[Value]| values.iter().map(|v| v.block).collect::<Vec<_>>();
    if m.blocks() != blocks || m.width() != width || blocks_of(m.values()) != blocks_of(p.values()) || targets.is_some_and(|t| t.rows.len() != experiments.len()) {
        return Err(error("models, read variables or targets do not match"));
    }
    if gradient && targets.is_none() {
        return Err(error("the divergence's gradient needs targets"));
    }
    for (i, e) in experiments.iter().enumerate() {
        check(e, batch, blocks)?;
        if targets.is_some_and(|t| t.rows[i].entropy.len() != length - e.position) {
            return Err(error("a target of other rows than its experiment's"));
        }
    }
    let (paths, bases) = paths(batch, experiments, p.values(), engine_heads(p), None, blocks)?;
    let plan = Plan::new(paths, length);
    let arithmetic = p.arithmetic();
    let (stream, calls) = run([p, m], &plan, gradient || probe.is_some())?;
    let work = Work {
        paths: plan.paths.len(),
        lanes: plan.lanes.len(),
        suffix_lanes: plan.lanes.iter().filter(|l| l.prefix.is_some()).count(),
        rows: calls.iter().map(|c| c.lanes.iter().map(|l| plan.range(*l).len()).sum::<usize>()).sum(),
        whole_rows: calls.iter().map(|c| c.lanes.len() * plan.length).sum(),
    };
    // Each experiment's scored rows, from its position on.
    let rows = outputs(&plan, &bases, experiments);
    let hidden = gather(d, &stream, &rows)?;
    let spread = |seed: &Tensor, weight: f64| spread(d, stream.rows(), &rows, seed, weight);
    let mut bits = Vec::new();
    let mut total = BTreeMap::new();
    let (mut probed, mut data_seed) = (None, None);
    if let Some(targets) = targets {
        let mut mu = d.zeros(hidden.rows(), width).map_err(error)?;
        let mut entropy = Vec::with_capacity(hidden.rows());
        let mut at = 0;
        for (r, t) in rows.iter().zip(&targets.rows) {
            d.set_rows(&mut mu, at, &t.mu).map_err(error)?;
            entropy.extend_from_slice(&t.entropy);
            at += r.len();
        }
        let target = Target { mu: Arc::new(mu), entropy, head: Arc::clone(&head.head), scored: None };
        let (nats, seed, pulled) = head.resident.score(d, &hidden, &target, gradient, probe.filter(|_| gradient), arithmetic)?;
        probed = pulled;
        bits.reserve(experiments.len());
        let mut at = 0;
        for r in &rows {
            bits.push(nats[at..at + r.len()].iter().map(|v| v / std::f64::consts::LN_2).collect());
            at += r.len();
        }
        if gradient {
            let seed = seed.ok_or_else(|| error("the head returned no cotangent"))?;
            data_seed = Some(spread(&seed, 1.0 / std::f64::consts::LN_2)?);
        }
    }
    // The divergence's gradient and the factor's draw (with a sweep of its own for the probe, its
    // logits and its seed, in the factor's arithmetic, when the divergence's sweep made none)
    // reverse together, each call's tape had once for both.
    let factor = factor_arithmetic(d, arithmetic);
    let factor_seed = match probe {
        Some(key) => Some(spread(
            &match probed {
                Some(seed) => seed,
                None if head.head.gaussian.is_some() => {
                    let (_, slope) = head.head.gaussian_outputs(&d.download(&hidden).map_err(error)?).ok_or_else(|| error("not a Gaussian head"))?;
                    crate::resident_causal_fit::fixed_head_target::gaussian_probe(d, &head.head, &slope, key)?
                }
                None => fisher_probe_seed(d, &hidden, head.resident.embedding_in(d, factor)?, head.resident.tile_rows.max(1), key, factor)?,
            },
            1.0,
        )?),
        None => None,
    };
    let mut u = BTreeMap::new();
    let mut passes = Vec::with_capacity(2);
    if let Some(cotangent) = data_seed {
        passes.push(Pass { cotangent, gradient: &mut total, arithmetic: factor });
    }
    let factored = factor_seed.is_some();
    if let Some(cotangent) = factor_seed {
        passes.push(Pass { cotangent, gradient: &mut u, arithmetic: factor });
    }
    if !passes.is_empty() {
        run_reverse([p, m], &plan, &calls, &mut passes)?;
    }
    drop(passes);
    let factor = factored.then(|| Factor { gradient: u, tokens: hidden.rows() });
    Ok(Evaluation { bits, gradient: total, factor, work })
}

/// The draw `u = Σ_t J_tᵀ b_t` of the Gauss–Newton factor ([`Factor`]) in `P`'s trainable operators
/// over every scored token `t` of `experiments` on `batch` (each experiment from its position on, as
/// [`evaluate`] scores it), at `P`'s loaded parameters, `b_t` the Fisher probe under `key` at
/// `P_e`'s own next-token distribution there (rows numbered from zero in experiment order). Its
/// outer product `u uᵀ` is an unbiased estimate of the Gauss–Newton matrix of the experiments'
/// divergence `Σ KL(M_e ‖ P_e)` in nats, `Σ_t J_tᵀ F_t J_t` with `F_t` the Fisher matrix of `P_e`'s
/// softmax at `t`: the matrix that is the divergence's Hessian where `P_e`'s predictions equal
/// `M_e`'s.
pub fn fisher_probe<E: BlockEngine>(m: &E, p: &E, head: &FixedHead, batch: &Batch, experiments: &[Experiment], key: u64) -> Result<BTreeMap<usize, Tensor>, String> {
    let evaluation = evaluate_probed((m, p), head, (batch, experiments), None, false, Some(key))?;
    Ok(evaluation.factor.ok_or_else(|| error("no Gauss–Newton factor"))?.gradient)
}

/// The experiments of one batch scored at `P`'s loaded parameters.
pub struct Scored {
    /// Per experiment, per base token, `KL(M_e ‖ P_e)` in bits.
    pub bits: Vec<Vec<f64>>,
    /// The gradient of the sum of `bits` in each trainable operator, in the order they were given
    /// (empty when not asked for).
    pub gradient: Vec<ndarray::Array2<f64>>,
    /// The forward pass's work ([`Work`]).
    pub work: Work,
}

/// `M` and `P` compiled for interchange experiments, with the head they share and `P`'s read
/// variables: what a fit needs to score `P` on a batch of experiments.
pub struct Interchange {
    m: DeviceProgram,
    p: DeviceProgram,
    m_sites: Arc<Sites>,
    p_sites: Arc<Sites>,
    head: FixedHead,
    variables: Vec<ReadVariable>,
    trainable: Vec<usize>,
    /// `M`'s targets kept on the host ([`Interchange::keep_targets`]); none until asked for.
    kept: RefCell<Option<TargetStore>>,
    /// `M`'s prefixes kept on the host ([`PrefixStore`]), with the targets.
    prefixes: Option<RefCell<PrefixStore>>,
}

impl Interchange {
    /// `M` is the split native program `native` (`run_check::split_sites`) with its `layers`
    /// (`run_check::layer_nodes`); `P` is `explanation`, built from it with the same sites, whose
    /// operators `trainable` receive gradients. The experiments patch the read variables
    /// `variables` ([`reads`]), each model at its own sites ([`values`]). Each program holds at
    /// most `numeric_bytes` of operator values on the device; `tile_rows` rows of vocabulary
    /// logits are formed at once. Products run in the device's storage precision.
    pub fn new(
        device: &Device,
        native: &OperatorProgram,
        layers: &[LayerNodes],
        explanation: &Artifact,
        trainable: &[usize],
        variables: Vec<ReadVariable>,
        numeric_bytes: usize,
        tile_rows: usize,
    ) -> Result<Self, String> {
        let arithmetic = if device.float64() { Arithmetic::F64 } else { Arithmetic::F32 };
        let (m_flat, m_streams, m_reads, m_parts, m_heads, m_shared) = flat_sites(&Artifact::native(native)?, layers)?;
        let (p_flat, p_streams, p_reads, p_parts, p_heads, p_shared) = flat_sites(explanation, layers)?;
        let (m_prefix, p_prefix) = (prefix(&m_flat)?, prefix(&p_flat)?);
        let mut m = DeviceProgram::compile_values_bounded(device, &m_prefix, numeric_bytes)?;
        m.set_arithmetic(arithmetic);
        // Inherited frozen operators retain their native Arc identity through
        // inlining. Reuse their resident buffers; prepare_dense_parameters below
        // detaches every trainable owner before any posterior update can run.
        let mut p = DeviceProgram::compile_values_sharing_bounded(&m, &p_prefix, numeric_bytes)?;
        p.set_arithmetic(arithmetic);
        p.prepare_dense_parameters(trainable)?;
        let head = FixedHead::new(device, &m_flat, &p_flat, tile_rows)?;
        let m_values = values(native, &Artifact::native(native)?, layers, &variables)?;
        let p_values = values(native, explanation, layers, &variables)?;
        let norms = |flat: &OperatorProgram, reads: &[usize], streams: &[usize]| -> Vec<(usize, Option<Norm>)> { reads.iter().zip(streams).map(|(r, s)| (*r, Norm::of(flat, *r, *s))).collect() };
        let (m_norms, p_norms) = (norms(&m_flat, &m_reads, &m_streams), norms(&p_flat, &p_reads, &p_streams));
        let mut m_sites = Sites::new(&m, &m_flat, m_streams, m_reads, &[], m_values)?;
        let mut p_sites = Sites::new(&p, &p_flat, p_streams, p_reads, trainable, p_values)?;
        m_sites.parts.matrices = Arc::new(matrix_uses(native, (&Artifact::native(native)?, &[]), &m_flat)?);
        let p_owners = if explanation.owners.is_empty() { crate::library_vpd::uses(native, layers, explanation)? } else { explanation.owners.clone() };
        p_sites.parts.matrices = Arc::new(matrix_uses(native, (explanation, &p_owners), &p_flat)?);
        m_sites.parts.reads = m_norms;
        p_sites.parts.reads = p_norms;
        let head_blocks: Vec<usize> = layers.iter().enumerate().flat_map(|(l, layer)| layer.reads.iter().map(move |_| 2 * l)).collect();
        for (sites, nodes, heads, program, shared) in [(&mut m_sites, m_parts, m_heads, &m, m_shared), (&mut p_sites, p_parts, p_heads, &p, p_shared)] {
            sites.parts.shared = shared.into_iter().map(|(site, n)| (site, (n, program.widths()[n]))).collect();
            sites.parts.nodes = nodes;
            sites.parts.heads = heads.into_iter().map(|n| n.map(|n| (n, program.widths()[n]))).collect();
            sites.parts.head_blocks.clone_from(&head_blocks);
        }
        let (m_sites, p_sites) = (Arc::new(m_sites), Arc::new(p_sites));
        Ok(Self { m, p, m_sites, p_sites, head, variables, trainable: trainable.to_vec(), kept: RefCell::new(None), prefixes: None })
    }

    /// The shared sites' typical norms ([`Interchange::measure_typical`]), which size a push: the
    /// same for every model, so another program applying the same operations reads them here.
    pub fn typical_norms(&self) -> BTreeMap<SharedSite, f64> {
        (*self.m_sites.parts.typical).clone()
    }

    /// The unit directions a push adds ([`Interchange::set_directions`]).
    pub fn push_directions(&self) -> Vec<Vec<f64>> {
        (*self.m_sites.parts.directions).clone()
    }

    /// The shared sites both models hold ([`SharedSite`]).
    pub fn shared_sites(&self) -> Vec<SharedSite> {
        self.m_sites.parts.shared.keys().filter(|s| self.p_sites.parts.shared.contains_key(s)).copied().collect()
    }

    /// `count` unit directions of the stream's width drawn from `seed` (each coordinate standard
    /// normal, then normalized): the directions [`Operation::Push`] adds, the same for every model.
    pub fn set_directions(&mut self, count: usize, seed: u64) {
        let width = self.m.widths()[self.m_sites.entries[0]];
        self.set_push_directions(seeded_directions(count, width, seed));
    }

    /// `directions` as the directions [`Operation::Push`] adds (index `i` the `i`-th), the same for
    /// every model: an adversarial search's candidates.
    pub fn set_push_directions(&mut self, directions: Vec<Vec<f64>>) {
        let directions = Arc::new(directions);
        for sites in [&mut self.m_sites, &mut self.p_sites] {
            let mut next = (**sites).clone();
            next.parts.directions = Arc::clone(&directions);
            *sites = Arc::new(next);
        }
    }

    /// Set each shared site's typical norm to `typical`'s (an experiment manifest's, measured once by
    /// [`Interchange::measure_typical`]), for both models.
    pub fn set_typical(&mut self, typical: BTreeMap<SharedSite, f64>) {
        let typical = Arc::new(typical);
        for sites in [&mut self.m_sites, &mut self.p_sites] {
            let mut next = (**sites).clone();
            next.parts.typical = Arc::clone(&typical);
            *sites = Arc::new(next);
        }
    }

    /// Each shared site's typical norm, the root mean square of its rows' norms over `batch`'s base
    /// sequences after their first token on `M`'s own runs: the unit of [`SIZES`], the same for
    /// every model.
    pub fn measure_typical(&mut self, batch: &Batch) -> Result<(), String> {
        let typical = {
            let (m, _) = self.models();
            let (d, length) = (m.device(), batch.length());
            let n = batch.base.len();
            let ranges: Vec<Range<usize>> = (0..n).map(|i| i * length..(i + 1) * length).collect();
            let later: Vec<Range<usize>> = (0..n).map(|i| i * length + 1..(i + 1) * length).collect();
            let tokens: Vec<&[u32]> = batch.base.iter().map(Vec::as_slice).collect();
            let heads = self.m_sites.parts.head_blocks.clone();
            let mut stream = d.zeros(n * length, BlockEngine::width(&m)).map_err(error)?;
            let mut typical = BTreeMap::new();
            for b in 0..m.blocks() {
                let trace = m.forward(b, &mut stream, &ranges, &tokens, None, true)?.ok_or_else(|| error("a block kept no tape"))?;
                for (site, (node, _)) in self.m_sites.parts.shared.iter().filter(|(s, _)| s.block(&heads) == Some(b)) {
                    let rows = d.download(&d.gather_ranges(trace.value(*node)?, &later).map_err(error)?).map_err(error)?;
                    let mean = rows.rows().into_iter().map(|r| r.dot(&r)).sum::<f64>() / rows.nrows().max(1) as f64;
                    typical.insert(*site, mean.sqrt());
                }
            }
            Arc::new(typical)
        };
        for sites in [&mut self.m_sites, &mut self.p_sites] {
            let mut next = (**sites).clone();
            next.parts.typical = Arc::clone(&typical);
            *sites = Arc::new(next);
        }
        Ok(())
    }

    /// Per base sequence of `batch`, its clean experiment and `per_base` experiments of operations
    /// on shared sites, each of a family drawn uniformly from `families` (none of `M`'s or an
    /// explanation's parts enter the draw, so every explanation faces the same experiments), the
    /// donor of base `n` its sequence `donors[n]`, all with `P` alone (with `hybrids`, under one
    /// hybrid per base drawn as [`sample`] draws it). An experiment's `k` operations, `k = 2^u` with
    /// `u` uniform in `0..=4` (at most the family's sites), act at distinct sites drawn uniformly
    /// among the family's (a swap and a scale at every shared site, a zeroing at heads', attentions'
    /// and MLPs' outputs, a push at the stream's, attentions' and MLPs' outputs), all at one row, at
    /// every row from it on, or at every row, each a third of the time (the row uniform after the
    /// attention sink); a scale's factor uniform among the nonzero [`SCALES`], a push's direction and
    /// size uniform.
    pub fn sample_ops(&self, rng: &mut impl RngExt, batch: &Batch, families: &[Family], per_base: usize, donors: &[usize], hybrids: bool) -> Result<Vec<Experiment>, String> {
        let blocks = self.m_sites.entries.len();
        if families.is_empty() || donors.len() != batch.base.len() {
            return Err(error("operations need families and a donor per base"));
        }
        let mut out = Vec::new();
        for n in 0..batch.base.len() {
            let explained = if !hybrids || rng.random_range(0..2) == 0 { vec![true; blocks] } else { hybrid(rng, blocks) };
            for _ in 0..per_base {
                let family = families[rng.random_range(0..families.len())];
                let (patch, position) = self.draw_ops(rng, family, batch.length())?;
                out.push(Experiment { base: n, source: donors[n], explained: explained.clone(), patch: Some(patch), position });
            }
            out.push(Experiment { base: n, source: n, explained, patch: None, position: 0 });
        }
        Ok(out)
    }

    /// One experiment's operations of `family` on sequences of `length` tokens and its position, as
    /// [`Interchange::sample_ops`] draws them.
    /// A weight edit ([`Family::Weight`]) is one of the table's ([`Interchange::set_weight_edits`]),
    /// uniform, at every row.
    pub fn draw_ops(&self, rng: &mut impl RngExt, family: Family, length: usize) -> Result<(Patch, usize), String> {
        let parts = &self.m_sites.parts;
        if family == Family::Weight {
            if parts.weights.is_empty() {
                return Err(error("a weight edit drawn before the table was set (set_weight_edits)"));
            }
            let edit = rng.random_range(0..parts.weights.len());
            return Ok((Patch::Weights { edit, block: parts.weights[edit].block }, 0));
        }
        draw_site_ops(rng, family, length, &self.shared_sites(), &parts.head_blocks, &parts.typical, parts.directions.len(), self.m_sites.entries.len())
    }

    /// `edits` as the native weight edits experiments of [`Family::Weight`] draw from (index `i` the
    /// `i`-th, [`Patch::Weights`]), the same for every model. Each edited matrix must be one of
    /// `M`'s, its factors of its shape, and its uses in `M` inside the edit's block.
    pub fn set_weight_edits(&mut self, edits: Vec<crate::weight_edit::Drawn>) -> Result<(), String> {
        let m = self.models().0;
        for (i, e) in edits.iter().enumerate() {
            let (start, end) = (m.entry(e.block), m.end(e.block));
            for x in &e.entries {
                let uses = self.m_sites.parts.matrices.get(&x.native).filter(|u| !u.is_empty()).ok_or_else(|| error(format!("weight edit {i}: {} is not one of M's matrices", x.native)))?;
                for u in uses {
                    let n = if x.rows { u.native_rows.end } else { u.native_cols.end };
                    if x.units.iter().any(|j| *j >= n) || u.input <= start || u.output > end {
                        return Err(error(format!("weight edit {i}: units of {} outside it, or it applied outside block {}", x.native, e.block)));
                    }
                }
            }
            for f in &e.factors {
                let uses = self.m_sites.parts.matrices.get(&f.native).filter(|u| !u.is_empty()).ok_or_else(|| error(format!("weight edit {i}: {} is not one of M's matrices", f.native)))?;
                for u in uses {
                    if (f.left.nrows(), f.right.nrows()) != (u.native_rows.end, u.native_cols.end) || f.left.ncols() != f.right.ncols() {
                        return Err(error(format!("weight edit {i}: factors of {:?} and {:?} for {} of {} x {}", f.left.dim(), f.right.dim(), f.native, u.native_rows.end, u.native_cols.end)));
                    }
                    if u.input <= start || u.output > end {
                        return Err(error(format!("weight edit {i}: {} is applied outside block {}", f.native, e.block)));
                    }
                }
            }
        }
        let table = Arc::new(edits);
        for sites in [&mut self.m_sites, &mut self.p_sites] {
            let mut next = (**sites).clone();
            next.parts.weights = Arc::clone(&table);
            *sites = Arc::new(next);
        }
        Ok(())
    }

    /// The native weight edits experiments draw from ([`Interchange::set_weight_edits`]).
    pub fn weight_edits(&self) -> &[crate::weight_edit::Drawn] {
        &self.m_sites.parts.weights
    }

    /// Set the table of native weight edits ([`Interchange::set_weight_edits`]) to `kept` of
    /// `candidates` edits of `M` (`native`) drawn from `seed` (`weight_edit::candidates`), each
    /// measured on `M` alone on `screen` (`KL(M_e ‖ M)` per token, [`Interchange::weight_effects`]),
    /// kept stratified by that effect (`weight_edit::stratified`, bins at
    /// `weight_edit::EFFECT_EDGES`) so edits that move `M` much are as common as those that barely do;
    /// returns the table. It depends on `M`, the seed and the screen's sequences alone: every
    /// explanation faces the same edits.
    pub fn draw_weight_edits(&mut self, native: &OperatorProgram, screen: &Batch, seed: u64, (candidates, kept): (usize, usize)) -> Result<Vec<crate::weight_edit::Drawn>, String> {
        let mut drawn = crate::weight_edit::candidates(native, seed, candidates)?;
        self.set_weight_edits(drawn.clone())?;
        let all: Vec<usize> = (0..drawn.len()).collect();
        for chunk in all.chunks(WEIGHT_SCREEN_CHUNK) {
            for (i, effect) in chunk.iter().zip(self.weight_effects(screen, chunk)?) {
                drawn[*i].effect = Some(effect);
            }
        }
        let table = crate::weight_edit::stratified(drawn, kept, &crate::weight_edit::EFFECT_EDGES)?;
        self.set_weight_edits(table.clone())?;
        Ok(table)
    }

    /// Per weight edit of the table, `edits` its indices, its effect on `M`: `KL(M_e ‖ M)` in bits
    /// per token over `batch`'s base sequences (every token).
    pub fn weight_effects(&self, batch: &Batch, edits: &[usize]) -> Result<Vec<f64>, String> {
        let (m, _) = self.models();
        let blocks = m.blocks();
        let table = &self.m_sites.parts.weights;
        let mut experiments = Vec::with_capacity(edits.len() * batch.base.len());
        for &e in edits {
            let block = table.get(e).ok_or_else(|| error(format!("weight edit {e}: not in the table")))?.block;
            experiments.extend((0..batch.base.len()).map(|n| Experiment { base: n, source: n, explained: vec![true; blocks], patch: Some(Patch::Weights { edit: e, block }), position: 0 }));
        }
        let targets = targets(&m, &self.head, batch, &experiments)?;
        let mut unedited = (*self.m_sites).clone();
        unedited.parts.unedited = true;
        let clean = Model { program: &self.m, sites: Arc::new(unedited), prefixes: None };
        let evaluation = evaluate(&m, &clean, &self.head, batch, &targets, &experiments, false)?;
        Ok(evaluation.bits.chunks(batch.base.len().max(1)).map(|per| {
            let tokens: usize = per.iter().map(Vec::len).sum();
            per.iter().flatten().sum::<f64>() / tokens.max(1) as f64
        }).collect())
    }

    /// `P` applies no edits of parts from now on, `M` still does: with `P` = `M`
    /// (`Artifact::native`), an edit's score is then `KL(M_e ‖ M)`, the size of the change the edit
    /// makes to `M`, over the same tokens as its `KL(M_e ‖ P_e)`.
    pub fn unedited_explanation(&mut self) {
        let mut next = (*self.p_sites).clone();
        next.parts.unedited = true;
        self.p_sites = Arc::new(next);
    }

    /// Per head (layer-major, a layer's heads in order) its attention block, the heads a
    /// site ([`SharedSite::Head`]) names; a head either model does not hold is not a shared site.
    pub fn heads(&self) -> Vec<(usize, bool)> {
        let held = |s: &Sites, h: usize| s.parts.heads.get(h).copied().flatten().is_some();
        self.m_sites.parts.head_blocks.iter().enumerate().map(|(h, b)| (*b, held(&self.m_sites, h) && held(&self.p_sites, h))).collect()
    }

    /// `M` and `P` as the free functions of this module take them.
    pub fn models(&self) -> (Model<'_>, Model<'_>) {
        (Model { program: &self.m, sites: Arc::clone(&self.m_sites), prefixes: self.prefixes.as_ref() }, Model { program: &self.p, sites: Arc::clone(&self.p_sites), prefixes: None })
    }

    /// `P`'s program, to write its trainable operators on the device (`device_posterior`).
    pub fn explanation_mut(&mut self) -> &mut DeviceProgram {
        &mut self.p
    }

    /// The head `M` and `P` share.
    pub fn head(&self) -> &FixedHead {
        &self.head
    }

    /// The read variables the experiments patch.
    pub fn variables(&self) -> &[ReadVariable] {
        &self.variables
    }

    /// `M`'s targets for `experiments` on `batch` ([`targets`]), on `M`'s engine. Once
    /// [`Interchange::keep_targets`] asked for it, targets made for a batch are kept on the host
    /// while the process's memory budget admits them, on local disk while it has room
    /// (`DiskTargets`), and the same experiments on the same tokens are given those back (bit for
    /// bit) instead of running `M` again.
    pub fn targets(&self, batch: &Batch, experiments: &[Experiment]) -> Result<Targets, String> {
        let teacher = self.models().0;
        let mut store = self.kept.try_borrow_mut().map_err(error)?;
        let identity = store.as_ref().map(|_| Identity { base: batch.base.clone(), source: batch.source.clone(), experiments: experiments.to_vec() });
        if let (Some(store), Some(identity)) = (store.as_mut(), identity.as_ref()) {
            if let Some(disk) = store.disk.as_mut() {
                disk.asked(identity);
            }
            if let Some(kept) = store.batches.get(identity) {
                return kept.restore(self.m.device());
            }
            if let Some(bytes) = store.disk.as_mut().map(|disk| disk.read(identity)).transpose()?.flatten() {
                return HostTargets::decode(&bytes, &self.head.head)?.restore(self.m.device());
            }
        }
        let made = targets(&teacher, &self.head, batch, experiments)?;
        if let (Some(store), Some(identity)) = (store.as_mut(), identity) {
            store.batches.remove(&identity);
            if let Ok(reservation) = store.governor.try_reserve(HostTargets::bytes(&made, &identity), "interchange: M's targets of a batch kept on the host") {
                store.batches.insert(identity, reservation.bind(HostTargets::of(self.m.device(), &made)?));
            } else if let Some(disk) = store.disk.as_mut()
                && made.rows.iter().all(|t| Arc::ptr_eq(&t.head, &self.head.head))
            {
                disk.keep(identity, &HostTargets::of(self.m.device(), &made)?)?;
            }
        }
        Ok(made)
    }

    /// Keep `M`'s targets on the host from now on ([`Interchange::targets`]), each batch's while
    /// `governor`'s budget admits it (made on the device each time otherwise): for a fit that
    /// scores one fixed collection of experiments again and again, where `M`'s forward pass is
    /// otherwise repeated at every scoring.
    pub fn keep_targets(&mut self, governor: &MemoryGovernor) {
        *self.kept.get_mut() = Some(TargetStore { governor: governor.clone(), batches: HashMap::new(), disk: DiskTargets::new() });
        self.prefixes = Some(RefCell::new(PrefixStore::new(governor)));
    }

    /// `M`'s prefixes kept ([`PrefixStore`]): none before [`Interchange::keep_targets`].
    #[must_use]
    pub fn kept_prefixes(&self) -> usize {
        self.prefixes.as_ref().map_or(0, |p| p.borrow().len())
    }

    /// `P`'s program, so that a fit writes each weight sample into its resident parameters.
    pub fn program_mut(&mut self) -> &mut DeviceProgram {
        &mut self.p
    }

    /// `P`'s score on `experiments` against `targets` ([`evaluate`]), the gradient left on the
    /// device per trainable operator.
    pub fn evaluate_resident(&self, batch: &Batch, experiments: &[Experiment], targets: &Targets, gradient: bool) -> Result<Evaluation, String> {
        self.evaluate_probed(batch, experiments, Some(targets), gradient, None)
    }

    /// [`evaluate_probed`] of `P` as it is held, the gradients left on the device.
    pub fn evaluate_probed(&self, batch: &Batch, experiments: &[Experiment], targets: Option<&Targets>, gradient: bool, probe: Option<u64>) -> Result<Evaluation, String> {
        let (teacher, p) = self.models();
        evaluate_probed((&teacher, &p), &self.head, (batch, experiments), targets, gradient, probe)
    }

    /// [`fisher_probe`] at `P`'s loaded parameters, the gradient left on the device per trainable
    /// operator.
    pub fn fisher_probe_resident(&self, batch: &Batch, experiments: &[Experiment], key: u64) -> Result<BTreeMap<usize, Tensor>, String> {
        let evaluation = self.evaluate_probed(batch, experiments, None, false, Some(key))?;
        Ok(evaluation.factor.ok_or_else(|| error("no Gauss–Newton factor"))?.gradient)
    }

    /// Load `P`'s trainable operators, in the order given to [`Interchange::new`].
    /// The first `values.len()` of them (a fit's posterior operators, ahead of operators it writes
    /// itself, such as a shared stage's assignment, `library_mdl::Share`); the rest keep theirs.
    pub fn load<A: std::borrow::Borrow<ndarray::Array2<f64>>>(&mut self, values: &[A]) -> Result<(), String> {
        if values.len() > self.trainable.len() {
            return Err(error("more values than trainable operators"));
        }
        for (&op, value) in self.trainable.iter().zip(values) {
            let tensor = self.p.device().upload(value.borrow().view()).map_err(error)?;
            self.p.replace_dense_parameter(op, tensor)?;
        }
        self.p.refresh_fused()
    }

    /// `KL(M_e ‖ P_e)` per token for each of `experiments` on `batch` from its position on, at
    /// `P`'s loaded parameters, with `M`'s targets made here, and
    /// with `gradient` its sum's gradient downloaded per trainable operator.
    pub fn evaluate(&self, batch: &Batch, experiments: &[Experiment], gradient: bool) -> Result<Scored, String> {
        let targets = self.targets(batch, experiments)?;
        let evaluation = self.evaluate_resident(batch, experiments, &targets, gradient)?;
        if !gradient {
            return Ok(Scored { bits: evaluation.bits, gradient: Vec::new(), work: evaluation.work });
        }
        let d = self.p.device();
        let gradient = self
            .trainable
            .iter()
            .map(|op| match evaluation.gradient.get(op) {
                Some(g) => d.download(g).map_err(error),
                None => {
                    let shape = self.p.dense(*op)?;
                    Ok(ndarray::Array2::zeros((shape.rows(), shape.cols())))
                }
            })
            .collect::<Result<Vec<_>, String>>()?;
        if gradient.iter().any(|g| g.iter().any(|v| !v.is_finite())) {
            return Err(error("a nonfinite parameter gradient"));
        }
        Ok(Scored { bits: evaluation.bits, gradient, work: evaluation.work })
    }

    /// The response diagnostic of `experiments` on `batch`: per experiment, per scored token (from
    /// its position on), `KL(p_e ‖ p̃_e)` in bits, with `p` = `M`, `q` = `P` (the experiment's
    /// hybrid), `0` the clean run of the same base (and hybrid), `e` the edited run, and
    /// `p̃_e ∝ p_0 q_e / q_0`: `M`'s clean prediction moved by `P`'s response to the edit. The head is
    /// linear in the final hidden row `h` (`p ∝ exp(E h)`), so `p̃_e` is the head at
    /// `h_{M,0} + h_{P,e} − h_{P,0}`. It is zero where `P` responds to the edit as `M` does
    /// (`h_{P,e} − h_{P,0} = h_{M,e} − h_{M,0}`) whatever `P`'s clean error, which the gap
    /// `KL(p_e ‖ q_e)` also counts.
    pub fn response(&self, batch: &Batch, experiments: &[Experiment]) -> Result<Vec<Vec<f64>>, String> {
        self.response_from(self, batch, experiments)
    }

    /// [`Interchange::response`] where the edit is compiled into this interchange's models (a native
    /// weight edit, `weight_edit::compile`) and `clean` holds the models before it: the clean runs
    /// `p_0` and `q_0` are `clean`'s, the edited ones this interchange's.
    pub fn response_from(&self, clean: &Interchange, batch: &Batch, experiments: &[Experiment]) -> Result<Vec<Vec<f64>>, String> {
        let (m, p) = self.models();
        let (m_clean, p_clean) = clean.models();
        let d = p.device();
        let unedited: Vec<Experiment> = experiments.iter().map(|e| Experiment { base: e.base, source: e.base, explained: e.explained.clone(), patch: None, position: 0 }).collect();
        let targets = self.targets(batch, experiments)?;
        let (m0, p0, pe) = (final_rows(&m_clean, &p_clean, batch, &unedited, true)?, final_rows(&m_clean, &p_clean, batch, &unedited, false)?, final_rows(&m, &p, batch, experiments, false)?);
        let mut moved = ndarray::Array2::zeros((pe.iter().map(ndarray::Array2::nrows).sum(), BlockEngine::width(&p)));
        let mut at = 0;
        for (i, e) in experiments.iter().enumerate() {
            let rows = pe[i].nrows();
            let tail = ndarray::s![e.position.., ..];
            let h = &m0[i].slice(tail) + &pe[i] - &p0[i].slice(tail);
            moved.slice_mut(ndarray::s![at..at + rows, ..]).assign(&h);
            at += rows;
        }
        let hidden = d.upload(moved.view()).map_err(error)?;
        let mut mu = d.zeros(hidden.rows(), hidden.cols()).map_err(error)?;
        let (mut entropy, mut at) = (Vec::with_capacity(hidden.rows()), 0);
        for (r, t) in pe.iter().zip(&targets.rows) {
            d.set_rows(&mut mu, at, &t.mu).map_err(error)?;
            entropy.extend_from_slice(&t.entropy);
            at += r.nrows();
        }
        let target = Target { mu: Arc::new(mu), entropy, head: Arc::clone(&self.head.head), scored: None };
        let (nats, _, _) = self.head.resident.score(d, &hidden, &target, false, None, p.arithmetic())?;
        let mut out = Vec::with_capacity(experiments.len());
        let mut at = 0;
        for r in &pe {
            out.push(nats[at..at + r.nrows()].iter().map(|v| v / std::f64::consts::LN_2).collect());
            at += r.nrows();
        }
        Ok(out)
    }
}

/// Per experiment, the final hidden rows of its base's path from its position on (on the host):
/// `M` alone when `alone`, else the experiment's hybrid of `P` and `M` with its edits.
fn final_rows<E: BlockEngine>(m: &E, p: &E, batch: &Batch, experiments: &[Experiment], alone: bool) -> Result<Vec<ndarray::Array2<f64>>, String> {
    let d = p.device();
    let blocks = p.blocks();
    let native = vec![false; blocks];
    let engine = if alone { m } else { p };
    let (paths, bases) = paths(batch, experiments, engine.values(), engine_heads(engine), alone.then_some(native.as_slice()), blocks)?;
    let plan = Plan::new(paths, batch.length);
    let (stream, _) = run([engine, m], &plan, false)?;
    outputs(&plan, &bases, experiments).iter().map(|r| d.download(&d.rows_of(&stream, r.start, r.len()).map_err(error)?).map_err(error)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        import::import_language_model,
        library_mdl,
        run_check::{layer_nodes, split_sites},
        test_support::tiny_export,
    };
    use rand::{SeedableRng, rngs::StdRng};

    /// Two cotangents reversed together give each the gradient a reverse of its own gives, bit for
    /// bit, on the host and on an accelerator.
    #[test]
    fn reverses_run_together_are_each_one_alone() {
        let dir = tiny_export("interchange_fused_reverse", 2);
        let imported = import_language_model(&dir, 6, 12).expect("the tiny export imports");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("the native sites");
        let layers = layer_nodes(&native, 2).expect("the layers");
        let explanation = library_mdl::explanation(&native, &layers).expect("the library");
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let blocks: Vec<LayerNodes> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let mut devices = vec![Device::host()];
        devices.extend(Device::accelerator(gam_gpu::GpuPolicy::Auto).expect("a device probe"));
        for device in devices {
            let variables = reads(&native, &blocks).expect("the reads");
            let ic = Interchange::new(&device, &native, &blocks, &explanation.artifact, &explanation.trainable, variables, 1 << 30, 64).expect("the experiments");
            let batch = Batch::new(sequences[..3].to_vec(), sequences[3..].to_vec()).expect("the batch");
            let experiments = sample(&mut StdRng::seed_from_u64(5), 3, ic.variables(), 4, 12).expect("the draw");
            let (m, p) = ic.models();
            let (paths, bases) = paths(&batch, &experiments, p.values(), &[], None, 4).expect("the paths");
            let plan = Plan::new(paths, 12);
            let (stream, calls) = run([&p, &m], &plan, true).expect("the forward pass");
            let rows = outputs(&plan, &bases, &experiments);
            let scored: usize = rows.iter().map(ExactSizeIterator::len).sum();
            let seed = |k: u64| {
                let mut rng = StdRng::seed_from_u64(k);
                let values = ndarray::Array2::from_shape_fn((scored, stream.cols()), |_| rng.random::<f64>() - 0.5);
                spread(&device, stream.rows(), &rows, &device.upload(values.view()).expect("upload"), 1.0).expect("the seed")
            };
            let bits = |gradient: &BTreeMap<usize, Tensor>| -> Vec<(usize, Vec<u64>)> {
                gradient.iter().map(|(op, g)| (*op, device.download(g).expect("download").iter().map(|v| v.to_bits()).collect())).collect()
            };
            let arithmetic = factor_arithmetic(&device, p.arithmetic());
            let (mut first, mut second) = (BTreeMap::new(), BTreeMap::new());
            run_reverse([&p, &m], &plan, &calls, &mut [Pass { cotangent: seed(1), gradient: &mut first, arithmetic }]).expect("the first pass");
            run_reverse([&p, &m], &plan, &calls, &mut [Pass { cotangent: seed(2), gradient: &mut second, arithmetic }]).expect("the second pass");
            let (mut first_together, mut second_together) = (BTreeMap::new(), BTreeMap::new());
            run_reverse(
                [&p, &m],
                &plan,
                &calls,
                &mut [Pass { cotangent: seed(1), gradient: &mut first_together, arithmetic }, Pass { cotangent: seed(2), gradient: &mut second_together, arithmetic }],
            )
            .expect("the passes together");
            assert!(!first.is_empty(), "the passes reach P's operators");
            assert_eq!(bits(&first_together), bits(&first), "the first pass");
            assert_eq!(bits(&second_together), bits(&second), "the second pass");
        }
    }

    /// Batched edits against one patch at a time, with a patched row repeated (2), a source
    /// repeated (4) and a source that is also a patched row (2), on the host: equal bit for bit.
    #[test]
    fn batched_edits_do_what_one_patch_at_a_time_does() {
        let d = Device::host();
        let (rows, sources) = (vec![1, 2, 2, 3], vec![4, 0, 4, 2]);
        let take = ndarray::array![[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let keep = take.mapv(|m: f64| 1.0 - m);
        let patches = Patches { rows: rows.clone(), sources: sources.clone(), keep: d.upload(keep.view()).unwrap(), take: d.upload(take.view()).unwrap() };
        let edits = Edits { nodes: BTreeMap::from([(7, patches)]), adds: BTreeMap::new(), probes: BTreeMap::new(), cuts: BTreeMap::new(), recorded: None, weights: BTreeMap::new(), carried: RefCell::new(BTreeMap::new()) };
        let start = ndarray::Array2::from_shape_fn((6, 3), |(r, c)| 1.0 + r as f64 * 0.37 - c as f64 * 1.9);
        // Applied one at a time, every source read first.
        let mut applied = start.clone();
        let read: Vec<_> = sources.iter().map(|s| start.row(*s).to_owned()).collect();
        for (i, row) in rows.iter().enumerate() {
            let next = &applied.row(*row) * &keep.row(i) + &read[i] * &take.row(i);
            applied.row_mut(*row).assign(&next);
        }
        let mut value = d.upload(start.view()).unwrap();
        edits.apply(&d, 7, &mut value).unwrap();
        assert_eq!(d.download(&value).unwrap(), applied);
        // The transpose one at a time: every patched row read first, then the sources' shares.
        let mut transposed = start.clone();
        let read: Vec<_> = rows.iter().map(|r| start.row(*r).to_owned()).collect();
        for (i, row) in rows.iter().enumerate() {
            transposed.row_mut(*row).assign(&(&read[i] * &keep.row(i)));
        }
        for (i, source) in sources.iter().enumerate() {
            let next = &transposed.row(*source) + &(&read[i] * &take.row(i));
            transposed.row_mut(*source).assign(&next);
        }
        let mut g = d.upload(start.view()).unwrap();
        edits.transpose(&d, 7, &mut g).unwrap();
        assert_eq!(d.download(&g).unwrap(), transposed);
        assert_eq!(rounds(&[2, 1, 2, 2, 1]), vec![vec![0, 1], vec![2, 4], vec![3]]);
    }

    /// Kept targets ([`Interchange::keep_targets`]) are the targets made by `M` bit for bit, and so
    /// is every score against them, on the host and on the accelerator when there is one; a budget
    /// that does not admit a batch's targets keeps none and still gives them.
    #[test]
    fn kept_targets_are_the_made_targets_bit_for_bit() {
        let dir = tiny_export("interchange_kept_targets", 2);
        let imported = import_language_model(&dir, 6, 12).expect("the tiny export imports");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("the native sites");
        let layers = layer_nodes(&native, 2).expect("the layers");
        let explanation = library_mdl::explanation(&native, &layers).expect("the library");
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let blocks: Vec<LayerNodes> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let mut devices = vec![Device::host()];
        devices.extend(Device::accelerator(gam_gpu::GpuPolicy::Auto).expect("a device probe"));
        for (d, device) in devices.into_iter().enumerate() {
            let made = |budget: Option<usize>| {
                let variables = reads(&native, &blocks).expect("the reads");
                let mut ic = Interchange::new(&device, &native, &blocks, &explanation.artifact, &explanation.trainable, variables, 1 << 30, 64).expect("the experiments");
                if let Some(budget) = budget {
                    ic.keep_targets(&MemoryGovernor::with_budget_bytes(budget));
                }
                ic
            };
            let batch = Batch::new(sequences[..3].to_vec(), sequences[3..].to_vec()).expect("the batch");
            let experiments = sample(&mut StdRng::seed_from_u64(3), 3, made(None).variables(), 4, 12).expect("the draw");
            let other = sample(&mut StdRng::seed_from_u64(4), 3, made(None).variables(), 4, 12).expect("the draw");
            let fresh = made(None);
            let reference = fresh.targets(&batch, &experiments).expect("the targets");
            let scores = |ic: &Interchange, targets: &Targets| ic.evaluate_resident(&batch, &experiments, targets, false).expect("a score").bits;
            let bits = |values: Vec<(ndarray::Array2<f64>, Vec<f64>)>| -> Vec<Vec<u64>> {
                values.into_iter().map(|(mu, entropy)| mu.iter().chain(&entropy).map(|v| v.to_bits()).collect()).collect()
            };
            let score_bits = |scores: Vec<Vec<f64>>| -> Vec<Vec<u64>> { scores.into_iter().map(|s| s.into_iter().map(f64::to_bits).collect()).collect() };
            let expected = bits(reference.host(&device).expect("download"));
            let expected_scores = scores(&fresh, &reference);
            for (budget, kept) in [(1 << 34, 2), (0, 0)] {
                let ic = made(Some(budget));
                for _ in 0..2 {
                    let targets = ic.targets(&batch, &experiments).expect("the targets");
                    assert_eq!(bits(targets.host(&device).expect("download")), expected, "device {d}, budget {budget}: the targets");
                    assert_eq!(score_bits(scores(&ic, &targets)), score_bits(expected_scores.clone()), "device {d}, budget {budget}: the scores");
                    // Other experiments on the same tokens are another identity.
                    let targets = ic.targets(&batch, &other).expect("the targets");
                    assert_eq!(bits(targets.host(&device).expect("download")), bits(fresh.targets(&batch, &other).expect("the targets").host(&device).expect("download")));
                }
                assert_eq!(ic.kept.borrow().as_ref().map(|s| s.batches.len()), Some(kept), "device {d}, budget {budget}: the batches kept");
                // A budget that admits nothing keeps both batches on disk, read back bit for bit above.
                let on_disk = ic.kept.borrow().as_ref().and_then(|s| s.disk.as_ref().map(|disk| disk.files.len()));
                let room = disk_space(&std::env::temp_dir()).is_some_and(|(free, total)| free > (total / 10).max(8 << 30) + (1 << 30));
                assert!(!room || on_disk == Some(2 - kept), "device {d}, budget {budget}: {on_disk:?} batches on disk");
            }
        }
    }

    /// The squared Gauss–Newton factor ([`Factor`]) estimates the diagonal of `Σ_r J_rᵀ F_r J_r`
    /// over a batch's scored rows `r`, `F_r = diag p_r − p_r p_rᵀ` at `P`'s prediction `p_r`. The
    /// exact diagonal is `Σ_r Σ_c p_rc (J_rᵀ (p_r − e_c))²`: one reverse pass per row and class
    /// through the same forward pass, seeded at that row alone. The draws' mean of `u²` must lie
    /// within five standard errors of it, for each operator's trace and for its largest entry: for
    /// the probe made by its own sweep, and for the one made in the divergence's sweep with its
    /// gradient. On the host the two make one key's factor bit for bit (the same probabilities,
    /// signs and products, and reverses run together are each one alone).
    #[test]
    fn the_squared_factor_estimates_the_gauss_newton_diagonal() {
        let dir = tiny_export("interchange_factor", 2);
        let imported = import_language_model(&dir, 6, 12).expect("the tiny export imports");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("the native sites");
        let layers = layer_nodes(&native, 2).expect("the layers");
        let explanation = library_mdl::explanation(&native, &layers).expect("the library");
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let device = Device::host();
        let blocks: Vec<LayerNodes> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let variables = reads(&native, &blocks).expect("the reads");
        let ic = Interchange::new(&device, &native, &blocks, &explanation.artifact, &explanation.trainable, variables, 1 << 30, 64).expect("the experiments");
        let batch = Batch::new(sequences[..3].to_vec(), sequences[3..].to_vec()).expect("the batch");
        let experiments = sample(&mut StdRng::seed_from_u64(3), 3, ic.variables(), 4, 12).expect("the draw");
        // The exact diagonal.
        let (m, p) = ic.models();
        let (paths, bases) = paths(&batch, &experiments, p.values(), &[], None, 4).expect("the paths");
        let plan = Plan::new(paths, 12);
        let (stream, calls) = run([&p, &m], &plan, true).expect("the forward pass");
        let rows = outputs(&plan, &bases, &experiments);
        let hidden = device.download(&gather(&device, &stream, &rows).expect("the scored rows")).expect("download");
        let embedding = device.download(&ic.head().resident.embedding).expect("download");
        let logits = hidden.dot(&embedding.t());
        let mut exact: BTreeMap<usize, ndarray::Array2<f64>> = BTreeMap::new();
        for r in 0..hidden.nrows() {
            let top = logits.row(r).fold(f64::NEG_INFINITY, |a, b| a.max(*b));
            let weights = logits.row(r).mapv(|z| (z - top).exp());
            let prediction = &weights / weights.sum();
            for c in 0..embedding.nrows() {
                let mut residual = prediction.clone();
                residual[c] -= 1.0;
                let mut seed = ndarray::Array2::zeros(hidden.dim());
                seed.row_mut(r).assign(&residual.dot(&embedding));
                let cotangent = spread(&device, stream.rows(), &rows, &device.upload(seed.view()).expect("upload"), 1.0).expect("the seed");
                let mut gradient = BTreeMap::new();
                run_reverse([&p, &m], &plan, &calls, &mut [Pass { cotangent, gradient: &mut gradient, arithmetic: p.arithmetic() }]).expect("the reverse pass");
                for (op, g) in gradient {
                    let g = device.download(&g).expect("download");
                    let term = g.mapv(|v| prediction[c] * v * v);
                    match exact.get_mut(&op) {
                        Some(total) => *total += &term,
                        None => {
                            exact.insert(op, term);
                        }
                    }
                }
            }
        }
        // The draws.
        let targets = ic.targets(&batch, &experiments).expect("the targets");
        let factor_bits = |factor: &Factor| -> Vec<(usize, Vec<u64>)> {
            factor.gradient.iter().map(|(op, u)| (*op, device.download(u).expect("download").iter().map(|v| v.to_bits()).collect())).collect()
        };
        for key in [1, 2, 3] {
            let fused = ic.evaluate_probed(&batch, &experiments, Some(&targets), true, Some(key)).expect("a draw").factor.expect("the factor");
            let alone = ic.evaluate_probed(&batch, &experiments, None, false, Some(key)).expect("a draw").factor.expect("the factor");
            assert_eq!(factor_bits(&fused), factor_bits(&alone), "key {key}: the divergence's sweep and the probe's own");
        }
        for fused in [false, true] {
            let draws = 400;
            let mut sums: BTreeMap<usize, (ndarray::Array2<f64>, ndarray::Array2<f64>, f64, f64)> = BTreeMap::new();
            let mut rng = StdRng::seed_from_u64(7);
            for _ in 0..draws {
                let key = rng.random::<u64>();
                let evaluation = if fused {
                    ic.evaluate_probed(&batch, &experiments, Some(&targets), true, Some(key))
                } else {
                    ic.evaluate_probed(&batch, &experiments, None, false, Some(key))
                };
                let factor = evaluation.expect("a draw").factor.expect("the factor");
                assert_eq!(factor.tokens, hidden.nrows());
                for (op, u) in factor.gradient {
                    let squared = device.download(&u).expect("download").mapv(|v| v * v);
                    let trace = squared.sum();
                    let entry = sums.entry(op).or_insert_with(|| (ndarray::Array2::zeros(squared.dim()), ndarray::Array2::zeros(squared.dim()), 0.0, 0.0));
                    entry.0 += &squared;
                    entry.1 += &squared.mapv(|v| v * v);
                    entry.2 += trace;
                    entry.3 += trace * trace;
                }
            }
            let n = draws as f64;
            let check = |what: String, sum: f64, square: f64, expected: f64| {
                let mean = sum / n;
                let error = ((square / n - mean * mean) / (n - 1.0)).max(0.0).sqrt();
                assert!((mean - expected).abs() <= 5.0 * error + 1e-12 * expected.abs(), "{what}: mean {mean:e} of the draws against {expected:e} (standard error {error:e})");
            };
            assert_eq!(sums.keys().collect::<Vec<_>>(), exact.keys().collect::<Vec<_>>(), "the same operators receive the factor");
            for (op, expected) in &exact {
                let (sum, square, trace, trace_square) = &sums[op];
                let name = &explanation.artifact.program.operators[*op].name;
                check(format!("{name} trace (fused {fused})"), *trace, *trace_square, expected.sum());
                let (at, largest) = expected.indexed_iter().fold(((0, 0), 0.0), |best, (at, v)| if *v > best.1 { (at, *v) } else { best });
                check(format!("{name} entry {at:?} (fused {fused})"), sum[at], square[at], largest);
            }
        }
    }

    /// The forward tangents through blocks and edits ([`Model::tangent`]) of a batch with a swap of
    /// the stream from a donor, a head's zeroing from a position on, an MLP's output scaled at every
    /// row, a pushed direction and connection cuts (onward and at one row), on the scoped starting
    /// library of the tiny Qwen3 export (its MLPs P's, its attention M's): for random tangents `v` of P's
    /// operators and a random cotangent `ḡ` of the scored rows, `⟨ḡ, J v⟩` from the tangent pass
    /// equals `Σ ⟨∇, v⟩` from the reverse pass to 1e-10 relative (host, float64).
    #[test]
    fn the_reverse_pass_is_the_transpose_of_the_tangent_pass() {
        use rand::RngExt;
        let dir = crate::test_support::tiny_qwen3_export("interchange_tangent", 2);
        let imported = import_language_model(&dir, 6, 12).expect("the tiny export imports");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("the native sites");
        let layers = layer_nodes(&native, 2).expect("the layers");
        let explanation = library_mdl::scoped(&library_mdl::explanation(&native, &layers).expect("the library"), &[1, 3]).expect("scoped");
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let device = Device::host();
        let blocks: Vec<LayerNodes> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let variables = reads(&native, &blocks).expect("the reads");
        let mut ic = Interchange::new(&device, &native, &blocks, &explanation.artifact, &explanation.trainable, variables, 1 << 30, 64).expect("the experiments");
        let mut rng = StdRng::seed_from_u64(21);
        let head = ic.shared_sites().into_iter().find(|s| matches!(s, SharedSite::Head(_))).expect("a shared head");
        let batch = Batch::new(sequences[..3].to_vec(), sequences[3..6].to_vec()).expect("the batch");
        ic.set_directions(4, 2);
        ic.measure_typical(&batch).expect("the typical norms");
        let e = |base: usize, source: usize, family: Family, ops: Vec<SiteOp>, position: usize| Experiment { base, source, explained: vec![true; 4], patch: Some(Patch::Ops { family, ops }), position };
        let op = |site: SharedSite, operation: Operation, onward: bool| SiteOp { site, operation, onward };
        let experiments = vec![
            e(0, 1, Family::Swap, vec![op(SharedSite::Stream(0), Operation::Swap, false), op(SharedSite::Mlp(0), Operation::Swap, true)], 3),
            e(1, 1, Family::Zero, vec![op(head, Operation::Scale(0), true)], 2),
            e(2, 2, Family::Scale, vec![op(SharedSite::Mlp(1), Operation::Scale(3), true)], 0),
            e(0, 0, Family::Push, vec![op(SharedSite::Stream(1), Operation::Push { direction: 1, size: 2 }, false)], 6),
            e(1, 2, Family::Cut, vec![op(SharedSite::Mlp(0), Operation::Cut { to: 3 }, true), op(SharedSite::Attention(0), Operation::Cut { to: 2 }, false)], 4),
            // One row of an MLP's output, forked from its base's clean run: a lane whose rows before
            // its position its twin serves (`Lane::prefix`).
            Experiment { base: 2, source: 2, explained: vec![true; 4], patch: None, position: 0 },
            e(2, 2, Family::Scale, vec![op(SharedSite::Mlp(1), Operation::Scale(2), false)], 5),
        ];
        let (m, p) = ic.models();
        let (paths, bases) = paths(&batch, &experiments, p.values(), engine_heads(&p), None, 4).expect("the paths");
        let plan = Plan::new(paths, 12);
        let (stream, calls) = run([&p, &m], &plan, true).expect("the forward pass");
        let rows = outputs(&plan, &bases, &experiments);
        let scored: usize = rows.iter().map(ExactSizeIterator::len).sum();
        let seed_values = ndarray::Array2::from_shape_fn((scored, stream.cols()), |_| rng.random::<f64>() - 0.5);
        let seed = spread(&device, stream.rows(), &rows, &device.upload(seed_values.view()).expect("upload"), 1.0).expect("the seed");
        let mut gradient = BTreeMap::new();
        run_reverse([&p, &m], &plan, &calls, &mut [Pass { cotangent: device.copy(&seed).expect("copy"), gradient: &mut gradient, arithmetic: Arithmetic::F64 }]).expect("the reverse pass");
        let tangents: BTreeMap<usize, ndarray::Array2<f64>> = explanation
            .trainable
            .iter()
            .map(|op| {
                let shape = p.program.dense(*op).expect("a dense operator").dim();
                (*op, ndarray::Array2::from_shape_fn(shape, |_| rng.random::<f64>() - 0.5))
            })
            .collect();
        let along: f64 = tangents.iter().map(|(op, v)| device.download(&gradient[op]).expect("download").iter().zip(v).map(|(a, b)| a * b).sum::<f64>()).sum();
        // The tangent pass: block by block, the lanes forking there copying their parent's rows.
        let mut t = device.zeros(stream.rows(), stream.cols()).expect("zeros");
        let none = BTreeMap::new();
        for b in 0..4 {
            for lane in plan.lanes.iter().filter(|l| l.start == b) {
                if let Some(parent) = lane.parent {
                    let from = plan.lanes[parent].rows.clone();
                    device.copy_rows_within(&mut t, lane.rows.start, from.start, from.len()).expect("a fork");
                }
            }
            for call in calls.iter().filter(|c| c.block == b) {
                let ranges: Vec<Range<usize>> = call.lanes.iter().map(|l| plan.range(*l)).collect();
                let Some(Kept::Tape(tape)) = &call.kept else { panic!("a call kept no tape") };
                let model = if call.side == 0 { &p } else { &m };
                model.tangent(b, tape, &mut t, &ranges, call.edits.as_ref(), if call.side == 0 { &tangents } else { &none }).expect("the tangent");
            }
            plan.copy_prefixes(&device, &mut t, b).expect("the prefixes");
        }
        assert!(plan.lanes.iter().any(|l| l.prefix.is_some()), "some lane runs its MLP blocks from its position on");
        let projected: f64 = device.download(&t).expect("download").iter().zip(device.download(&seed).expect("download").iter()).map(|(a, b)| a * b).sum();
        assert!(projected.abs() > 1e-6, "the tangent reaches the scored rows: {projected}");
        assert!((projected - along).abs() <= 1e-10 * projected.abs().max(along.abs()), "tangent pass {projected}, reverse pass {along}");
    }


    /// The response diagnostic ([`Interchange::response`]) on the tiny Qwen3 export's scoped
    /// starting library with its MLPs perturbed (so `P` differs from `M`), `P` autonomous: with `P`
    /// applying no edit, `q_e = q_0`, so `p̃_e = p_0` and the diagnostic is the edit's effect
    /// `KL(p_e ‖ p_0)`, which a reference with `P` = `M` unedited scores as its gap (1e-9); applying
    /// the edit, it differs. (In a hybrid, `M`'s blocks still apply the edit on `P_e`'s path, so
    /// `q_e ≠ q_0` there even with `P` unedited.)
    #[test]
    fn the_response_diagnostic_of_an_unedited_explanation_is_the_edits_effect() {
        use rand::RngExt;
        let dir = crate::test_support::tiny_qwen3_export("interchange_response", 2);
        let imported = import_language_model(&dir, 6, 12).expect("the tiny export imports");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("the native sites");
        let layers = layer_nodes(&native, 2).expect("the layers");
        let explanation = library_mdl::scoped(&library_mdl::explanation(&native, &layers).expect("the library"), &[1, 3]).expect("scoped");
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let device = Device::host();
        let blocks: Vec<LayerNodes> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let variables = reads(&native, &blocks).expect("the reads");
        let mut ic = Interchange::new(&device, &native, &blocks, &explanation.artifact, &explanation.trainable, variables.clone(), 1 << 30, 64).expect("the experiments");
        let mut rng = StdRng::seed_from_u64(43);
        let program = &explanation.artifact.program;
        let perturbed: Vec<ndarray::Array2<f64>> = explanation.trainable.iter().map(|op| program.operators[*op].matrix().mapv(|v| v * (1.0 + 0.3 * (rng.random::<f64>() - 0.5)))).collect();
        ic.load(&perturbed).expect("a P that differs from M");
        let batch = Batch::new(sequences[..3].to_vec(), sequences[3..6].to_vec()).expect("the batch");
        let op = |site: SharedSite, operation: Operation, onward: bool| SiteOp { site, operation, onward };
        let experiments = vec![
            Experiment { base: 0, source: 1, explained: vec![true; 4], patch: Some(Patch::Ops { family: Family::Scale, ops: vec![op(SharedSite::Mlp(0), Operation::Scale(3), true)] }), position: 2 },
            Experiment { base: 1, source: 2, explained: vec![true; 4], patch: Some(Patch::Ops { family: Family::Swap, ops: vec![op(SharedSite::Stream(0), Operation::Swap, false)] }), position: 4 },
        ];
        let edited = ic.response(&batch, &experiments).expect("the response");
        ic.unedited_explanation();
        let unedited = ic.response(&batch, &experiments).expect("the response");
        let mut reference = Interchange::new(&device, &native, &blocks, &Artifact::native(&native).expect("M"), &[], variables, 1 << 30, 64).expect("the reference");
        reference.unedited_explanation();
        let effect = reference.evaluate(&batch, &experiments, false).expect("the effect").bits;
        for (a, b) in unedited.iter().flatten().zip(effect.iter().flatten()) {
            assert!((a - b).abs() <= 1e-9 * b.abs().max(1.0), "unedited P: the diagnostic {a}, the effect {b}");
        }
        let (sum_edited, sum_unedited): (f64, f64) = (edited.iter().flatten().sum(), unedited.iter().flatten().sum());
        assert!(sum_unedited > 1e-6 && (sum_edited - sum_unedited).abs() > 1e-6, "the edit moves M ({sum_unedited}) and P's response changes the diagnostic ({sum_edited})");
    }

    /// Swaps and cuts take the donor's values from each model's own run of the donor, never `M`'s:
    /// with `P` the scoped starting library of the tiny Qwen3 export with its MLPs perturbed (so `P`
    /// differs from `M`), swapping the whole stream after block 1 from the donor at every row makes
    /// `P_e` and `M_e` the two models' clean runs of the donor from block 2 on, so the swap scores
    /// as the donor's clean experiment does (1e-12); and a cut's donor record in `P`'s run is `P`'s
    /// own MLP output on the donor at the row (a run of `P` alone), which differs from `M`'s.
    #[test]
    fn swaps_and_cuts_read_each_models_own_donor_run() {
        use rand::RngExt;
        let dir = crate::test_support::tiny_qwen3_export("interchange_own_donor", 2);
        let imported = import_language_model(&dir, 6, 12).expect("the tiny export imports");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("the native sites");
        let layers = layer_nodes(&native, 2).expect("the layers");
        let explanation = library_mdl::scoped(&library_mdl::explanation(&native, &layers).expect("the library"), &[1, 3]).expect("scoped");
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let device = Device::host();
        let blocks: Vec<LayerNodes> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let variables = reads(&native, &blocks).expect("the reads");
        let mut ic = Interchange::new(&device, &native, &blocks, &explanation.artifact, &explanation.trainable, variables, 1 << 30, 64).expect("the experiments");
        let mut rng = StdRng::seed_from_u64(41);
        let program = &explanation.artifact.program;
        let perturbed: Vec<ndarray::Array2<f64>> = explanation.trainable.iter().map(|op| program.operators[*op].matrix().mapv(|v| v * (1.0 + 0.3 * (rng.random::<f64>() - 0.5)))).collect();
        ic.load(&perturbed).expect("a P that differs from M");
        let batch = Batch::new(sequences[..3].to_vec(), sequences[3..6].to_vec()).expect("the batch");
        // Base 0 with donor 1 (`batch.source[1]` is the donor's sequence): the whole stream after
        // block 1 swapped at every row, against the donor's clean experiment.
        let donor = Batch::new(vec![batch.base[0].clone(), batch.source[1].clone()], vec![batch.source[1].clone(), batch.source[1].clone()]).expect("the batch");
        let swap = Experiment { base: 0, source: 1, explained: vec![true; 4], patch: Some(Patch::Ops { family: Family::Swap, ops: vec![SiteOp { site: SharedSite::Stream(1), operation: Operation::Swap, onward: true }] }), position: 0 };
        let clean = Experiment { base: 1, source: 1, explained: vec![true; 4], patch: None, position: 0 };
        let bits = ic.evaluate(&donor, &[swap, clean], false).expect("evaluate").bits;
        assert!(bits[1].iter().sum::<f64>() > 1e-6, "P differs from M on the donor");
        for (a, b) in bits[0].iter().zip(&bits[1]) {
            assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "the swap scores {a}, the donor's clean run {b}");
        }
        // A cut from layer 0's MLP output into block 3's read at row 5: the donor record in P's run.
        let cut = Experiment { base: 0, source: 1, explained: vec![true; 4], patch: Some(Patch::Ops { family: Family::Cut, ops: vec![SiteOp { site: SharedSite::Mlp(0), operation: Operation::Cut { to: 3 }, onward: false }] }), position: 5 };
        let (m, p) = ic.models();
        let (paths, _) = paths(&batch, std::slice::from_ref(&cut), p.values(), engine_heads(&p), None, 4).expect("the paths");
        let plan = Plan::new(paths, 12);
        run([&p, &m], &plan, false).expect("the forward pass");
        let recorded = plan.recorded.borrow().get(&0).and_then(|r| r.values[1].as_ref().map(|v| device.download(&v.rows).expect("download").row(v.row).to_vec())).expect("the donor's record");
        let own = |model: &Model| -> Vec<f64> {
            let mut stream = device.zeros(12, BlockEngine::width(model)).expect("zeros");
            let tok = vec![batch.source[1].as_slice()];
            model.forward(0, &mut stream, &[0..12], &tok, None, false).expect("attention");
            let trace = model.forward(1, &mut stream, &[0..12], &tok, None, true).expect("the MLP").expect("a tape");
            let (node, _) = model.part_sites().expect("edit sites").shared[&SharedSite::Mlp(0)];
            device.download(trace.value(node).expect("the MLP output")).expect("download").row(5).to_vec()
        };
        let (from_p, from_m) = (own(&p), own(&m));
        let scale = from_p.iter().fold(1.0f64, |a, v| a.max(v.abs()));
        assert!(recorded.iter().zip(&from_p).all(|(a, b)| (a - b).abs() <= 1e-12 * scale), "the record is P's own donor value");
        assert!(from_p.iter().zip(&from_m).any(|(a, b)| (a - b).abs() > 1e-6 * scale), "P's donor value differs from M's");
    }

}

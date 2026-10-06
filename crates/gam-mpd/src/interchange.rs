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
use gam_gpu::tensor::{Arithmetic, ColumnBlocks, Device, Op, Storage, Tensor};
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
}

impl<'a> Model<'a> {
    /// `program` is the resident-value program of `flat` through its hidden node (the final normed
    /// stream, `Head::prefix`). Per block (each layer's attention, then its MLP) `entries` holds
    /// the stream entering it and `reads` the node its projections read. `trainable` lists the
    /// dense operators that receive gradients (none for `M`). Each block's nodes must read nothing
    /// before the stream entering it except token features, so that one block runs from that
    /// stream alone. `values` says where the model holds each read variable's value ([`values`]).
    pub fn new(program: &'a DeviceProgram, flat: &OperatorProgram, entries: Vec<usize>, reads: Vec<usize>, trainable: &[usize], values: Vec<Value>) -> Result<Self, String> {
        Ok(Self { program, sites: Arc::new(Sites::new(program, flat, entries, reads, trainable, values)?) })
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

/// The flat program of `artifact` (`mapped_inlined`), and per block (each layer's attention, then
/// its MLP) its entering stream and its read at the native sites `layers`
/// (`run_check::layer_nodes` of the native program `artifact` was made from): the arguments of
/// [`Model::new`]. `Artifact::native` gives `M`'s.
pub fn sites(artifact: &Artifact, layers: &[LayerNodes]) -> Result<(OperatorProgram, Vec<usize>, Vec<usize>), String> {
    let (flat, entries, reads, _, _) = flat_sites(artifact, layers)?;
    Ok((flat, entries, reads))
}

/// [`sites`] with, per block, where `artifact` applies edits of parts ([`PartSites`]): for layer
/// `l`'s MLP (block `2l + 1`) its read and the MLP's output (the native `mlp` node or the node
/// that replaced it), none for an attention block or an MLP output the artifact does not hold.
fn flat_sites(artifact: &Artifact, layers: &[LayerNodes]) -> Result<(OperatorProgram, Vec<usize>, Vec<usize>, Vec<Option<(usize, usize)>>, Vec<Option<usize>>), String> {
    let (flat, roots) = mapped_inlined(&artifact.program)?;
    let at = |native: usize| artifact.place(native).map(|n| roots[n]).ok_or_else(|| error(format!("native node {native} is not held")));
    let entries: Vec<usize> = layers.iter().flat_map(|l| [l.stream, l.attended]).map(at).collect::<Result<_, _>>()?;
    let reads: Vec<usize> = layers.iter().flat_map(|l| [l.normed_stream, l.normed]).map(at).collect::<Result<_, _>>()?;
    let blocks = entries.len();
    let parts = layers
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
    let heads = layers
        .iter()
        .enumerate()
        .flat_map(|(l, layer)| layer.reads.iter().map(move |r| (l, *r)))
        .map(|(l, r)| at(r).ok().filter(|n| *n > reads[2 * l] && *n < entries[2 * l + 1]))
        .collect();
    Ok((flat, entries, reads, parts, heads))
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
///
/// An edit of a part (`Part`): part `part` (an index into the parts, [`Interchange::set_parts`])
/// with its write scaled by `FACTORS[factor]` at the experiment's position. A head's removal
/// (`Head`): head `head` (an index into the heads, [`Interchange::heads`]) writes nothing at the
/// experiment's position, its attention output (the `o` site's input, the head's columns) zeroed
/// there in both models. A cut connection (`Cut`): part `to`'s read of part `from`'s write (`from`
/// in an earlier MLP block) takes `from`'s write on the source sequence in place of the base's at
/// the experiment's position, nothing else changing: the path patch of the connection, the
/// substitution made inside `to`'s read alone ([`Edits`]). The other edits patch no read variable
/// and have no source. A joint edit (`Parts`): the distinct parts `parts` (ascending, of one
/// block) each scaled by `FACTORS[factor]` at once, as one edit of each at the same row. A swap of
/// a part's activation (`Swap`): `(a(x′) − a(x))·u` added to the part's block output at the row,
/// `a` the part's activation read on the model's own stream on the source `x′` and on the base `x`
/// there (the transcoder features' read patch, which `M`'s read variables do not hold).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Patch {
    Read { variable: usize },
    Reads { variables: Vec<usize> },
    Part { part: usize, factor: usize },
    Head { head: usize },
    Cut { from: usize, to: usize },
    Parts { parts: Vec<usize>, factor: usize },
    Swap { part: usize },
}

impl Patch {
    /// The patched variables (none for an edit of a part).
    pub fn variables(&self) -> &[usize] {
        match self {
            Self::Read { variable } => std::slice::from_ref(variable),
            Self::Reads { variables } => variables,
            Self::Part { .. } | Self::Head { .. } | Self::Cut { .. } | Self::Parts { .. } | Self::Swap { .. } => &[],
        }
    }
}

/// A part of the explanation that edits act on: an MLP function `relu(g·x + c) u` of MLP block
/// `block` (block `2l + 1` is layer `l`'s MLP), reading the block's read `x` (the normed stream)
/// through `g` (`read`) and `c` (`bias`) and writing `u` (`write`) into the block's output, as
/// `P`'s posterior mean holds it (a transcoder feature, `library_transcoder`).
///
/// An edit of a part is one fixed function of a block's read, so it is defined identically on `M`
/// and on `P`: scaling the part's write by `α` adds `(α − 1)·relu(g·x + c)·u` to the block's
/// output at the edited row, `x` being that model's own read there. On `P` at its posterior mean
/// it is exactly `P` with the part's write scaled by `α` (`α = 0` removes the part); on `M` it
/// tests the claim that `M`'s MLP output holds the part's write `u` with the coefficient `P`
/// computes from `M`'s own input, so that subtracting it changes `M` as removing the part changes
/// `P`.
#[derive(Clone, Debug, PartialEq)]
pub struct Part {
    pub block: usize,
    pub read: Vec<f64>,
    pub bias: f64,
    pub write: Vec<f64>,
}

/// The factors `α` an edit of a part scales its write by: 0 removes the part, 0.5 halves it, 2
/// and 3 amplify it.
pub const FACTORS: [f64; 4] = [0.0, 0.5, 2.0, 3.0];

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
        if at("up").is_some() || at("m_gate").is_none() {
            continue;
        }
        let (gate, bias, write) = (program.operators[gate].matrix(), program.operators[bias].matrix(), program.operators[write].matrix());
        if bias.dim() != (gate.nrows(), 1) || write.dim() != (gate.ncols(), gate.nrows()) {
            return Err(error(format!("layer {l}: a transcoder MLP's gate, bias and write disagree in shape")));
        }
        for i in 0..gate.nrows() {
            let u = write.column(i);
            if u.iter().any(|v| *v != 0.0) {
                out.push(Part { block: 2 * l + 1, read: gate.row(i).to_vec(), bias: bias[[i, 0]], write: u.to_vec() });
            }
        }
    }
    Ok(out)
}

/// The families of patched experiments: a read patch, single or joint ([`sample`]); removing a
/// part (`FACTORS[0]`); scaling a part's write by one of the other factors; removing a random
/// subset of the parts firing at a row of one block at once (as [`subset`] draws a joint read
/// patch's: a single part's removal moves `M` little); removing a head; and cutting a connection
/// between two parts; swapping a part's activation for its value on the source sequence ([`Interchange::draw_edits`]; scored, not trained: it has
/// no reverse pass).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    Read,
    RemovePart,
    AmplifyPart,
    RemoveHead,
    CutConnection,
    RemoveParts,
    SwapPart,
}

/// Where a model applies edits of parts: per block, for an MLP block, the node its parts read
/// (the block's read) and the node their writes add to (the MLP's output), and the parts.
#[derive(Clone, Debug, Default)]
pub struct PartSites {
    nodes: Vec<Option<(usize, usize)>>,
    parts: Arc<Vec<Part>>,
    /// The model applies no edits ([`Interchange::unedited_explanation`]).
    unedited: bool,
    /// Per head ([`Interchange::heads`]) the node holding its attention output, when the model
    /// holds it, and the node's width.
    heads: Vec<Option<(usize, usize)>>,
    /// Per head its block (`2l` for layer `l`'s).
    head_blocks: Vec<usize>,
    /// Per block, for an MLP block, its input norm when the model applies it as an RMS norm of the
    /// stream entering the block with a gain (a cut connection recomputes it).
    norms: Vec<Option<Norm>>,
}

/// A block's input norm `N(s) = γ ⊙ s · (mean(s²) + ε)^{-1/2} + β` of the stream `s` entering it
/// (node `entry`), as the model applies it.
#[derive(Clone, Debug)]
struct Norm {
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

    fn apply(&self, s: &[f64]) -> Vec<f64> {
        let r = 1.0 / (s.iter().map(|v| v * v).sum::<f64>() / s.len() as f64 + self.epsilon).sqrt();
        s.iter().enumerate().map(|(k, v)| self.gain[k] * v * r + self.bias.as_ref().map_or(0.0, |b| b[k])).collect()
    }
}

/// A part's activation `relu(g·x + c)` at a read `x`.
fn activation(part: &Part, x: &[f64]) -> f64 {
    (x.iter().zip(&part.read).map(|(a, g)| a * g).sum::<f64>() + part.bias).max(0.0)
}

/// The cut connections' steps at one block of a call ([`Edits`]): per probe its row, cut, role (0
/// the base, 1 the source) and part, whose activation it records; per cut its row, cut and parts.
struct Cuts {
    read: usize,
    norm: Option<Norm>,
    probes: Vec<(usize, usize, usize, usize)>,
    cuts: Vec<(usize, usize, usize, usize)>,
}

impl PartSites {
    /// The parts.
    pub fn parts(&self) -> &[Part] {
        &self.parts
    }

    /// Block `block`'s node its parts read and the node their writes add to, if it has parts.
    pub fn nodes(&self, block: usize) -> Option<(usize, usize)> {
        self.nodes.get(block).copied().flatten()
    }
}

/// One experiment: base and source sequences (indices into the batch's), the hybrid (per block
/// whether `P_e` runs `P`'s version of it), at most one patch, and its position: the row the patch
/// replaces and the first scored token (0 for an unpatched experiment, scored everywhere).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Experiment {
    pub base: usize,
    pub source: usize,
    pub explained: Vec<bool>,
    pub patch: Option<Patch>,
    pub position: usize,
}

impl Experiment {
    /// The patched block of the variables `values` or of the parts `parts` (none for an unpatched
    /// experiment).
    fn block(&self, values: &[Value], parts: &[Part], heads: &[usize]) -> Result<Option<usize>, String> {
        let Some(patch) = &self.patch else { return Ok(None) };
        if let Patch::Cut { from, to } = patch {
            let (a, b) = (parts.get(*from).ok_or_else(|| error("a cut of an unknown part"))?, parts.get(*to).ok_or_else(|| error("a cut of an unknown part"))?);
            if a.block >= b.block {
                return Err(error("a cut connection from a part of a later block"));
            }
            return Ok(Some(b.block));
        }
        if let Patch::Head { head } = patch {
            return heads.get(*head).map(|b| Some(*b)).ok_or_else(|| error("a removal of an unknown head"));
        }
        if let Patch::Swap { part } = patch {
            return parts.get(*part).map(|p| Some(p.block)).ok_or_else(|| error("a swap of an unknown part"));
        }
        if let Patch::Part { part, factor } = patch {
            if *factor >= FACTORS.len() {
                return Err(error("an edit's factor outside FACTORS"));
            }
            return parts.get(*part).map(|p| Some(p.block)).ok_or_else(|| error("an edit of an unknown part"));
        }
        if let Patch::Parts { parts: chosen, factor } = patch {
            let block = |i: &usize| parts.get(*i).map(|p| p.block).ok_or_else(|| error("an edit of an unknown part"));
            let first = block(chosen.first().ok_or_else(|| error("a joint edit of no part"))?)?;
            if *factor >= FACTORS.len() || chosen.windows(2).any(|w| w[0] >= w[1]) || chosen.iter().any(|i| block(i).ok() != Some(first)) {
                return Err(error("a joint edit needs distinct ascending parts of one block and a factor in FACTORS"));
            }
            return Ok(Some(first));
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
    let mut counts: BTreeMap<&'static str, usize> = ["clean_alone", "clean_hybrid", "read_attention", "read_mlp", "read_joint", "remove_part", "amplify_part", "remove_head", "cut_connection", "remove_parts", "swap_part"].into_iter().map(|k| (k, 0)).collect();
    for e in experiments {
        let family = match &e.patch {
            None if e.explained.iter().all(|x| *x) => "clean_alone",
            None => "clean_hybrid",
            Some(Patch::Read { variable }) if variables.get(*variable).is_some_and(|v| v.block % 2 == 0) => "read_attention",
            Some(Patch::Read { .. }) => "read_mlp",
            Some(Patch::Reads { .. }) => "read_joint",
            Some(Patch::Part { factor: 0, .. }) => "remove_part",
            Some(Patch::Part { .. }) => "amplify_part",
            Some(Patch::Head { .. }) => "remove_head",
            Some(Patch::Cut { .. }) => "cut_connection",
            Some(Patch::Parts { .. }) => "remove_parts",
            Some(Patch::Swap { .. }) => "swap_part",
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
                let at = *named.get(owner.operator.as_str()).ok_or_else(|| error(format!("{}: no operator {}", owner.site, owner.operator)))?;
                let mut path = invocation(program, &owner.body, &owner.site)?;
                let body = path.last().and_then(|n| call_rule(program, &path[..path.len() - 1], *n)).ok_or_else(|| error(format!("{}: no rule {}", owner.site, owner.body)))?;
                let node = program.rules[body]
                    .nodes
                    .iter()
                    .position(|n| matches!(n, Node::Affine { terms, .. } if terms.first().is_some_and(|t| t.1 == at)))
                    .ok_or_else(|| error(format!("{}: no node of {} applies {}", owner.site, owner.body, owner.operator)))?;
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
    writes: BTreeMap<usize, Writes>,
    /// The cut connections' probes and cuts by the node they act at (the MLP's output), the parts,
    /// and per cut the activations of its `from` part on the base and on the source, recorded at
    /// `from`'s block for `to`'s block, which a later call runs (shared by a plan's calls).
    cuts: BTreeMap<usize, Cuts>,
    parts: Arc<Vec<Part>>,
    recorded: Option<std::rc::Rc<RefCell<BTreeMap<usize, [f64; 2]>>>>,
}

/// The edits of parts at one block's output node ([`Edits`]): per edit its row, the part's read
/// `g`, bias `c` and write `u`, and `α − 1`; the node the parts read; and, kept between a call's
/// passes, per edit the slope `(α − 1)·1[g·x + c > 0]` of its forward and the cotangent `u·ḡ` of
/// its coefficient in the reverse.
struct Writes {
    read: usize,
    rows: Vec<usize>,
    /// Per edit the row its activation is read at (its own row, or a swap's source row).
    from: Vec<usize>,
    gates: ndarray::Array2<f64>,
    biases: Vec<f64>,
    outs: ndarray::Array2<f64>,
    scales: Vec<f64>,
    slopes: RefCell<Vec<f64>>,
    carried: RefCell<Vec<f64>>,
}

impl Writes {
    /// Add `rows` (one per edit, in the edits' order) to the edited rows of `t`, each round of
    /// distinct rows at once ([`rounds`]).
    fn add(&self, d: &Device, t: &mut Tensor, rows: ndarray::Array2<f64>, targets: &[usize]) -> Result<(), String> {
        let added = d.upload(rows.view()).map_err(error)?;
        for round in rounds(targets) {
            let at = single(round.iter().map(|i| targets[*i]));
            let mut value = d.gather_ranges(t, &at).map_err(error)?;
            let part = picked(d, &added, &round)?;
            d.axpy(&mut value, 1.0, part.as_ref().unwrap_or(&added)).map_err(error)?;
            d.scatter_ranges(t, &at, &value).map_err(error)?;
        }
        Ok(())
    }

    /// The edited rows of `t` on the host.
    fn rows_of(&self, d: &Device, t: &Tensor, rows: &[usize]) -> Result<ndarray::Array2<f64>, String> {
        d.download(&d.gather_ranges(t, &single(rows.iter().copied())).map_err(error)?).map_err(error)
    }
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
    /// variables), at the sites `values` of the model that runs the call, and its edits of parts
    /// `edits` (the edited row, the part and the index of its factor in [`FACTORS`]) at the
    /// model's part sites `sites`.
    pub(crate) fn with_parts(
        d: &Device,
        patches: &[(usize, usize, &[usize])],
        values: &[Value],
        edits: &[(usize, Edit)],
        sites: Option<&PartSites>,
        recorded: Option<std::rc::Rc<RefCell<BTreeMap<usize, [f64; 2]>>>>,
    ) -> Result<Self, String> {
        let mut writes: BTreeMap<usize, (usize, Vec<(usize, usize, &Part, f64)>)> = BTreeMap::new();
        let mut zeroed = Vec::new();
        let mut cuts: BTreeMap<usize, Cuts> = BTreeMap::new();
        for (row, edit) in edits {
            let sites = sites.ok_or_else(|| error("an edit in a model without edit sites"))?;
            if sites.unedited {
                continue;
            }
            let at_block = |cuts: &mut BTreeMap<usize, Cuts>, part: usize| -> Result<usize, String> {
                let block = sites.parts.get(part).ok_or_else(|| error("a cut of an unknown part"))?.block;
                let (read, out) = sites.nodes.get(block).copied().flatten().ok_or_else(|| error(format!("block {block}: no part sites")))?;
                let norm = sites.norms.get(block).cloned().flatten();
                cuts.entry(out).or_insert_with(|| Cuts { read, norm, probes: Vec::new(), cuts: Vec::new() });
                Ok(out)
            };
            let (part, factor, from) = match edit {
                Edit::Part { part, factor } => (part, *factor, None),
                // A swap: the part's activation read at the source's row, less the base's.
                Edit::SwapAt { part, from } => (part, usize::MAX, Some(*from)),
                Edit::Swap { .. } => return Err(error("a swap without its source's row")),
                Edit::Probe { cut, part, role, .. } => {
                    let out = at_block(&mut cuts, *part)?;
                    cuts.get_mut(&out).ok_or_else(|| error("a cut's block"))?.probes.push((*row, *cut, *role, *part));
                    continue;
                }
                Edit::Cut { cut, from, to } => {
                    let out = at_block(&mut cuts, *to)?;
                    cuts.get_mut(&out).ok_or_else(|| error("a cut's block"))?.cuts.push((*row, *cut, *from, *to));
                    continue;
                }
                Edit::Head { head } => {
                    let (node, width) = sites.heads.get(*head).copied().flatten().ok_or_else(|| error(format!("head {head}: not held by the model")))?;
                    zeroed.push((*row, node, width));
                    continue;
                }
            };
            let p = sites.parts.get(*part).ok_or_else(|| error("an edit of an unknown part"))?;
            let (read, out) = sites.nodes.get(p.block).copied().flatten().ok_or_else(|| error(format!("block {}: no part sites", p.block)))?;
            let list = &mut writes.entry(out).or_insert_with(|| (read, Vec::new())).1;
            match from {
                Some(from) => {
                    list.push((*row, from, p, 1.0));
                    list.push((*row, *row, p, -1.0));
                }
                None => list.push((*row, *row, p, FACTORS.get(factor).ok_or_else(|| error("an edit's factor outside FACTORS"))? - 1.0)),
            }
        }
        let writes = writes
            .into_iter()
            .map(|(out, (read, list))| {
                let width = list[0].2.read.len();
                if list.iter().any(|(_, _, p, _)| p.read.len() != width || p.write.len() != width) {
                    return Err(error("parts of one block differ in width"));
                }
                let gates = ndarray::Array2::from_shape_fn((list.len(), width), |(i, c)| list[i].2.read[c]);
                let outs = ndarray::Array2::from_shape_fn((list.len(), width), |(i, c)| list[i].2.write[c]);
                let rows = list.iter().map(|(r, _, _, _)| *r).collect();
                let from = list.iter().map(|(_, f, _, _)| *f).collect();
                let biases = list.iter().map(|(_, _, p, _)| p.bias).collect();
                let scales: Vec<f64> = list.iter().map(|(_, _, _, s)| *s).collect();
                let k = scales.len();
                Ok((out, Writes { read, rows, from, gates, biases, outs, scales, slopes: RefCell::new(vec![0.0; k]), carried: RefCell::new(vec![0.0; k]) }))
            })
            .collect::<Result<_, String>>()?;
        let mut out = Self::patches(d, patches, values, &zeroed)?;
        out.writes = writes;
        if !cuts.is_empty() {
            out.parts = Arc::clone(&sites.ok_or_else(|| error("a cut in a model without edit sites"))?.parts);
            out.cuts = cuts;
            out.recorded = Some(recorded.ok_or_else(|| error("a cut connection outside a plan"))?);
        }
        Ok(out)
    }

    /// The read patches `patches` and the rows `zeroed` (a row, a node and its width) of nodes
    /// whose value is zeroed there (a head's removal: the patch keeping none of the row and taking
    /// none of its source, the row itself).
    fn patches(d: &Device, patches: &[(usize, usize, &[usize])], values: &[Value], zeroed: &[(usize, usize, usize)]) -> Result<Self, String> {
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
        for (row, node, width) in zeroed {
            masks.insert((*node, *row, *row), (vec![0.0; *width], vec![0.0; *width]));
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
        Ok(Self { nodes, writes: BTreeMap::new(), cuts: BTreeMap::new(), parts: Arc::new(Vec::new()), recorded: None })
    }

    /// The nodes the call edits, with the nodes edited parts read (whose cotangents take the
    /// edits' share).
    pub fn nodes(&self) -> BTreeSet<usize> {
        self.nodes.keys().chain(self.writes.keys()).chain(self.cuts.keys()).copied().chain(self.writes.values().map(|w| w.read)).collect()
    }

    /// Whether the forward pass changes node `node`'s value (a read of parts only is not changed).
    pub fn changes(&self, node: usize) -> bool {
        self.nodes.contains_key(&node) || self.writes.contains_key(&node) || self.cuts.contains_key(&node)
    }

    /// The edits of parts and the cut connections' steps at node `node` on its value `value` (the
    /// call's rows), `value_of` giving the call's other nodes' values: each edited row adds
    /// `(α − 1)·relu(g·x + c)·u`, `x` its row of the parts' read, the slopes kept for the
    /// transpose; a probe records its part's activation at its row; a cut adds
    /// `(relu(g_B·N(s + δ) + c_B) − relu(g_B·N(s) + c_B))·u_B` at its row, `s` the stream entering
    /// the block there, `N` the block's input norm and `δ = (a′ − a)·u_A` the change of the `from`
    /// part's write from the base's activation `a` to the source's `a′`.
    pub fn write<'v>(&self, d: &Device, node: usize, value: &mut Tensor, value_of: impl Fn(usize) -> Result<&'v Tensor, String>) -> Result<(), String> {
        if let Some(c) = self.cuts.get(&node) {
            let rows = |t: &Tensor, at: &[usize]| -> Result<ndarray::Array2<f64>, String> { d.download(&d.gather_ranges(t, &single(at.iter().copied())).map_err(error)?).map_err(error) };
            let recorded = self.recorded.as_ref().ok_or_else(|| error("a cut connection outside a plan"))?;
            let at: Vec<usize> = c.probes.iter().map(|p| p.0).collect();
            if !at.is_empty() {
                let x = rows(value_of(c.read)?, &at)?;
                for (k, (_, cut, role, part)) in c.probes.iter().enumerate() {
                    let a = activation(&self.parts[*part], x.row(k).as_slice().ok_or_else(|| error("a row"))?);
                    recorded.borrow_mut().entry(*cut).or_insert([f64::NAN; 2])[*role] = a;
                }
            }
            let at: Vec<usize> = c.cuts.iter().map(|p| p.0).collect();
            if !at.is_empty() {
                let norm = c.norm.as_ref().ok_or_else(|| error("a cut connection at a block whose input norm is not an RMS norm with a gain"))?;
                let streams = rows(value_of(norm.entry)?, &at)?;
                let mut added = ndarray::Array2::zeros((at.len(), value.cols()));
                for (k, (_, cut, from, to)) in c.cuts.iter().enumerate() {
                    let [a, source] = recorded.borrow().get(cut).copied().ok_or_else(|| error("a cut connection whose activations were not recorded"))?;
                    let (from, to) = (&self.parts[*from], &self.parts[*to]);
                    let s = streams.row(k).to_vec();
                    let moved: Vec<f64> = s.iter().zip(&from.write).map(|(v, u)| v + (source - a) * u).collect();
                    let change = activation(to, &norm.apply(&moved)) - activation(to, &norm.apply(&s));
                    added.row_mut(k).iter_mut().zip(&to.write).for_each(|(v, u)| *v = change * u);
                }
                let added = d.upload(added.view()).map_err(error)?;
                for round in rounds(&at) {
                    let rows = single(round.iter().map(|i| at[*i]));
                    let mut h = d.gather_ranges(value, &rows).map_err(error)?;
                    let part = picked(d, &added, &round)?;
                    d.axpy(&mut h, 1.0, part.as_ref().unwrap_or(&added)).map_err(error)?;
                    d.scatter_ranges(value, &rows, &h).map_err(error)?;
                }
            }
        }
        let Some(w) = self.writes.get(&node) else { return Ok(()) };
        let x = w.rows_of(d, value_of(w.read)?, &w.from)?;
        let mut added = ndarray::Array2::zeros((w.rows.len(), value.cols()));
        let mut slopes = vec![0.0; w.rows.len()];
        for i in 0..w.rows.len() {
            let pre = x.row(i).dot(&w.gates.row(i)) + w.biases[i];
            if pre > 0.0 {
                slopes[i] = w.scales[i];
                added.row_mut(i).assign(&(&w.outs.row(i) * (w.scales[i] * pre)));
            }
        }
        *w.slopes.borrow_mut() = slopes;
        w.add(d, value, added, &w.rows)
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

    /// The transpose of [`Edits::apply`] on node `node`'s cotangent `g`: each patched row keeps
    /// `ḡ ⊙ (1 − m)` and its source row receives `ḡ ⊙ m`, every patched row read before any row
    /// is written and the sources' shares added in order, each round of distinct rows at once.
    ///
    /// An edit of a part leaves its output node's cotangent as it is and keeps the cotangent `u·ḡ`
    /// of its coefficient there; at the node the part reads (later in the reverse), the edited row
    /// receives `(α − 1)·1[g·x + c > 0]·(u·ḡ)·g`.
    pub fn transpose(&self, d: &Device, node: usize, g: &mut Tensor) -> Result<(), String> {
        if self.cuts.contains_key(&node) {
            return Err(error("a cut connection has no reverse pass: it scores explanations, it does not train them"));
        }
        if let Some(w) = self.writes.get(&node) {
            let rows = w.rows_of(d, g, &w.rows)?;
            *w.carried.borrow_mut() = (0..w.rows.len()).map(|i| rows.row(i).dot(&w.outs.row(i))).collect();
        }
        for w in self.writes.values().filter(|w| w.read == node) {
            let (slopes, carried) = (w.slopes.borrow(), w.carried.borrow());
            let added = ndarray::Array2::from_shape_fn((w.rows.len(), g.cols()), |(i, c)| slopes[i] * carried[i] * w.gates[[i, c]]);
            w.add(d, g, added, &w.from)?;
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

    fn values(&self) -> &[Value] {
        &self.sites.values
    }

    fn part_sites(&self) -> Option<&PartSites> {
        Some(&self.sites.parts)
    }

    fn forward(
        &self,
        block: usize,
        stream: &mut Tensor,
        ranges: &[Range<usize>],
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
        let trace = self.program.forward_span(&family(tokens), entry, end, edit)?;
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
        let entering = if block > 0 {
            nodes.remove(&self.entry(block)).ok_or_else(|| error("no cotangent of a block's entering stream"))?
        } else {
            d.zeros(ranges.iter().map(ExactSizeIterator::len).sum(), cotangent.cols()).map_err(error)?
        };
        scatter(d, cotangent, ranges, &entering)
    }

    fn tape_bytes(tape: &DeviceTrace) -> usize {
        tape.bytes()
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

/// An edit a path applies at one block: of part `part` by `FACTORS[factor]`, or a head's removal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Edit {
    Part { part: usize, factor: usize },
    Head { head: usize },
    /// A cut connection's record of part `part`'s activation at row `at` on the base (`role` 0)
    /// or the source (1), and its change to `to`'s read (`Patch::Cut`).
    Probe { cut: usize, part: usize, role: usize, at: usize },
    Cut { cut: usize, from: usize, to: usize },
    /// A swap of part `part`'s activation from the source's path `source`, and as a call's edit,
    /// from the call's row `from`.
    Swap { part: usize, source: usize },
    SwapAt { part: usize, from: usize },
}

impl<'t> Path<'t> {
    fn patched(&self, block: usize) -> Option<(&'t [usize], usize)> {
        self.patch.filter(|(b, _, _)| *b == block).map(|(_, variables, source)| (variables, source))
    }

    fn edited(&self, block: usize) -> Option<Edit> {
        self.edits.iter().find(|(b, _)| *b == block).map(|(_, edit)| *edit)
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
    /// Per cut connection the activations its probes record ([`Edits::write`]).
    recorded: std::rc::Rc<RefCell<BTreeMap<usize, [f64; 2]>>>,
}

impl<'t> Plan<'t> {
    fn new(paths: Vec<Path<'t>>, length: usize) -> Self {
        let mut order: Vec<usize> = (0..paths.len()).collect();
        order.sort_by_key(|p| std::cmp::Reverse(paths[*p].end));
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
                    lanes.push(Lane { path: p, start, end: path.end, parent: parent.map(|(l, _)| l), rows });
                }
            }
        }
        Self { paths, lanes, holder, length, recorded: std::rc::Rc::new(RefCell::new(BTreeMap::new())) }
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
    fn patches(&self, block: usize, lanes: &[usize]) -> Result<Vec<(usize, usize, &'t [usize])>, String> {
        let at = |l: usize| lanes.iter().position(|x| *x == l).map(|i| i * self.length);
        let mut out = Vec::new();
        for &l in lanes {
            let path = &self.paths[self.lanes[l].path];
            if let Some((variables, source)) = path.patched(block) {
                let row = at(l).ok_or_else(|| error("a patched lane outside its call"))? + path.position;
                let source = at(self.lane(source, block)).ok_or_else(|| error("a patch's source outside its call"))? + path.position;
                out.push((row, source, variables));
            }
        }
        Ok(out)
    }

    /// The edits applied at `block` (of parts, heads' removals), as rows of a call over `lanes`
    /// (in order): the edited row and the edit.
    fn writes(&self, block: usize, lanes: &[usize]) -> Result<Vec<(usize, Edit)>, String> {
        let mut out = Vec::new();
        for (i, &l) in lanes.iter().enumerate() {
            let path = &self.paths[self.lanes[l].path];
            for (_, edit) in path.edits.iter().filter(|(b, _)| *b == block) {
                let at = match edit {
                    Edit::Probe { at, .. } => *at,
                    _ => path.position,
                };
                let edit = match edit {
                    Edit::Swap { part, source } => {
                        let lane = self.lane(*source, block);
                        let k = lanes.iter().position(|x| *x == lane).ok_or_else(|| error("a swap's source outside its call"))?;
                        Edit::SwapAt { part: *part, from: k * self.length + path.position }
                    }
                    other => *other,
                };
                out.push((i * self.length + at, edit));
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
    Ok(Some(Edits::with_parts(engine.device(), &patches, engine.values(), &writes, engine.part_sites(), Some(std::rc::Rc::clone(&plan.recorded)))?))
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
    for b in 0..blocks {
        for lane in plan.lanes.iter().filter(|l| l.start == b) {
            if let Some(parent) = lane.parent {
                let rows = plan.lanes[parent].rows.clone();
                d.copy_rows_within(&mut stream, lane.rows.start, rows.start, rows.len()).map_err(error)?;
            }
        }
        for side in 0..2 {
            let lanes: Vec<usize> = (0..plan.lanes.len()).filter(|&l| (plan.lanes[l].start..plan.lanes[l].end).contains(&b) && plan.paths[plan.lanes[l].path].explained[b] == (side == 0)).collect();
            if lanes.is_empty() {
                continue;
            }
            let ranges: Vec<Range<usize>> = lanes.iter().map(|l| plan.lanes[*l].rows.clone()).collect();
            let tokens: Vec<&[u32]> = lanes.iter().map(|l| plan.paths[plan.lanes[*l].path].tokens).collect();
            let edits = edits(engines[side], plan, b, &lanes)?;
            let kept_here = match budget {
                None => {
                    engines[side].forward(b, &mut stream, &ranges, &tokens, edits.as_ref(), false)?;
                    None
                }
                Some(budget) if kept.saturating_add(largest.saturating_mul(3)) < budget => {
                    let tape = engines[side].forward(b, &mut stream, &ranges, &tokens, edits.as_ref(), true)?.ok_or_else(|| error("a call kept no tape"))?;
                    let bytes = E::tape_bytes(&tape);
                    kept = kept.saturating_add(bytes);
                    largest = largest.max(bytes);
                    Some(Kept::Tape(tape))
                }
                Some(_) => {
                    let entering = if b == 0 { None } else { Some(gather(d, &stream, &ranges)?) };
                    kept = kept.saturating_add(entering.as_ref().map_or(0, Tensor::bytes));
                    engines[side].forward(b, &mut stream, &ranges, &tokens, edits.as_ref(), false)?;
                    Some(Kept::Entering(entering))
                }
            };
            calls.push(Call { block: b, side, lanes, edits, kept: kept_here });
        }
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
fn factor_arithmetic(d: &Device, arithmetic: Arithmetic) -> Arithmetic {
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
        let ranges: Vec<Range<usize>> = call.lanes.iter().map(|l| plan.lanes[*l].rows.clone()).collect();
        let recomputed;
        let tape = match call.kept.as_ref().ok_or_else(|| error("a call kept nothing for the reverse pass"))? {
            Kept::Tape(tape) => tape,
            // The block's forward again, from the rows that entered it (laid out one lane after
            // another) with the same patches, keeping its tape.
            Kept::Entering(entering) => {
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
                recomputed = engines[call.side].forward(b, &mut rows, &local, &tokens, call.edits.as_ref(), true)?.ok_or_else(|| error("a call kept no tape"))?;
                &recomputed
            }
        };
        for pass in passes.iter_mut() {
            engines[call.side].reverse(b, tape, &mut pass.cotangent, &ranges, call.edits.as_ref(), (&mut *pass.gradient, pass.arithmetic))?;
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
fn paths<'t>(batch: &'t Batch, experiments: &'t [Experiment], values: &[Value], (parts, heads): (&[Part], &[usize]), native: Option<&'t [bool]>, blocks: usize) -> Result<(Vec<Path<'t>>, Vec<usize>), String> {
    let (mut paths, mut bases) = (Vec::with_capacity(2 * experiments.len()), Vec::with_capacity(experiments.len()));
    for (i, e) in experiments.iter().enumerate() {
        check(e, batch, blocks)?;
        let explained = native.unwrap_or(&e.explained);
        let (patch, edits) = match (&e.patch, e.block(values, parts, heads)?) {
            (Some(Patch::Part { part, factor }), Some(block)) => (None, vec![(block, Edit::Part { part: *part, factor: *factor })]),
            (Some(Patch::Head { head }), Some(block)) => (None, vec![(block, Edit::Head { head: *head })]),
            (Some(Patch::Swap { part }), Some(block)) => {
                paths.push(Path { tokens: &batch.source[e.source], explained, end: block + 1, position: 0, patch: None, edits: Vec::new() });
                (None, vec![(block, Edit::Swap { part: *part, source: paths.len() - 1 })])
            }
            (Some(Patch::Parts { parts: chosen, factor }), Some(block)) => (None, chosen.iter().map(|part| (block, Edit::Part { part: *part, factor: *factor })).collect()),
            (Some(Patch::Cut { from, to }), Some(block)) => {
                // The source runs to `from`'s block, where its probe records `from`'s activation.
                let a = parts[*from].block;
                let probe = |role: usize| Edit::Probe { cut: i, part: *from, role, at: e.position };
                paths.push(Path { tokens: &batch.source[e.source], explained, end: a + 1, position: 0, patch: None, edits: vec![(a, probe(1))] });
                (None, vec![(a, probe(0)), (block, Edit::Cut { cut: i, from: *from, to: *to })])
            }
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

/// The parts of `engine`'s edits and its heads' blocks (none without part sites).
fn engine_parts<E: BlockEngine>(engine: &E) -> (&[Part], &[usize]) {
    engine.part_sites().map_or((&[], &[]), |s| (s.parts(), &s.head_blocks))
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
    let (paths, bases) = paths(batch, experiments, m.values(), engine_parts(m), Some(&native), blocks)?;
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
                    KeptValues::Single(v) => held.upload_f32(t.shape.0, t.shape.1, v),
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
struct TargetStore {
    governor: MemoryGovernor,
    batches: HashMap<Identity, Governed<HostTargets>>,
    disk: Option<DiskTargets>,
}

impl Targets {
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
    let (paths, bases) = paths(batch, experiments, p.values(), engine_parts(p), None, blocks)?;
    let plan = Plan::new(paths, length);
    let arithmetic = p.arithmetic();
    let (stream, calls) = run([p, m], &plan, gradient || probe.is_some())?;
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
                None => fisher_probe_seed(d, &hidden, &head.resident.embedding, head.resident.tile_rows.max(1), key, factor)?,
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
    Ok(Evaluation { bits, gradient: total, factor })
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
        let (m_flat, m_streams, m_reads, m_parts, m_heads) = flat_sites(&Artifact::native(native)?, layers)?;
        let (p_flat, p_streams, p_reads, p_parts, p_heads) = flat_sites(explanation, layers)?;
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
        let mut m_sites = Sites::new(&m, &m_flat, m_streams, m_reads, &[], m_values)?;
        let mut p_sites = Sites::new(&p, &p_flat, p_streams, p_reads, trainable, p_values)?;
        let head_blocks: Vec<usize> = layers.iter().enumerate().flat_map(|(l, layer)| layer.reads.iter().map(move |_| 2 * l)).collect();
        for (sites, nodes, heads, program, flat) in [(&mut m_sites, m_parts, m_heads, &m, &m_flat), (&mut p_sites, p_parts, p_heads, &p, &p_flat)] {
            sites.parts.norms = nodes.iter().enumerate().map(|(b, n)| n.and_then(|(read, _)| Norm::of(flat, read, sites.entries[b]))).collect();
            sites.parts.nodes = nodes;
            sites.parts.heads = heads.into_iter().map(|n| n.map(|n| (n, program.widths()[n]))).collect();
            sites.parts.head_blocks.clone_from(&head_blocks);
        }
        let (m_sites, p_sites) = (Arc::new(m_sites), Arc::new(p_sites));
        Ok(Self { m, p, m_sites, p_sites, head, variables, trainable: trainable.to_vec(), kept: RefCell::new(None) })
    }

    /// The parts that edits of parts act on ([`Part`], [`Patch::Part`]), the same for `M` and `P`:
    /// each in an MLP block both models hold the output of, reading and writing the stream's
    /// width. A fit sets them from `P`'s posterior mean.
    pub fn set_parts(&mut self, parts: Vec<Part>) -> Result<(), String> {
        let width = self.m.widths()[self.m_sites.entries[0]];
        for p in &parts {
            let held = |s: &Sites| s.parts.nodes.get(p.block).copied().flatten().is_some();
            if !held(&self.m_sites) || !held(&self.p_sites) {
                return Err(error(format!("block {}: a part outside an MLP block both models hold", p.block)));
            }
            if p.read.len() != width || p.write.len() != width {
                return Err(error("a part's read or write is not of the stream's width"));
            }
        }
        let parts = Arc::new(parts);
        for sites in [&mut self.m_sites, &mut self.p_sites] {
            let mut next = (**sites).clone();
            next.parts.parts = Arc::clone(&parts);
            *sites = Arc::new(next);
        }
        Ok(())
    }

    /// `P` applies no edits of parts from now on, `M` still does: with `P` = `M`
    /// (`Artifact::native`), an edit's score is then `KL(M_e ‖ M)`, the size of the change the edit
    /// makes to `M`, over the same tokens as its `KL(M_e ‖ P_e)`.
    pub fn unedited_explanation(&mut self) {
        let mut next = (*self.p_sites).clone();
        next.parts.unedited = true;
        self.p_sites = Arc::new(next);
    }

    /// The parts ([`Interchange::set_parts`]).
    pub fn parts(&self) -> &[Part] {
        &self.m_sites.parts.parts
    }

    /// Per head (layer-major, a layer's heads in order) its attention block, the heads a
    /// removal ([`Patch::Head`]) names; a head either model does not hold cannot be removed.
    pub fn heads(&self) -> Vec<(usize, bool)> {
        let held = |s: &Sites, h: usize| s.parts.heads.get(h).copied().flatten().is_some();
        self.m_sites.parts.head_blocks.iter().enumerate().map(|(h, b)| (*b, held(&self.m_sites, h) && held(&self.p_sites, h))).collect()
    }

    /// Per `(base, block, position)` of `wanted` (a base sequence of `batch`, an MLP block, a row),
    /// the parts of that block that fire on `M`'s own read there (`g·x + c > 0`, `M` running alone
    /// on the base).
    pub fn active_parts(&self, batch: &Batch, wanted: &[(usize, usize, usize)]) -> Result<Vec<Vec<usize>>, String> {
        let (m, _) = self.models();
        let (d, length) = (m.device(), batch.length());
        let bases: Vec<usize> = wanted.iter().map(|w| w.0).collect::<BTreeSet<_>>().into_iter().collect();
        if wanted.iter().any(|w| w.0 >= batch.base.len() || w.1 >= m.blocks() || w.2 >= length) {
            return Err(error("an activity query outside the batch or the blocks"));
        }
        let mut by_block: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, p) in self.parts().iter().enumerate() {
            by_block.entry(p.block).or_default().push(i);
        }
        let ranges: Vec<Range<usize>> = (0..bases.len()).map(|i| i * length..(i + 1) * length).collect();
        let tokens: Vec<&[u32]> = bases.iter().map(|b| batch.base[*b].as_slice()).collect();
        let mut stream = d.zeros(bases.len() * length, BlockEngine::width(&m)).map_err(error)?;
        let mut out = vec![Vec::new(); wanted.len()];
        let Some(last) = wanted.iter().map(|w| w.1).max() else { return Ok(out) };
        for b in 0..=last {
            let needed: Vec<usize> = (0..wanted.len()).filter(|i| wanted[*i].1 == b).collect();
            let Some(trace) = m.forward(b, &mut stream, &ranges, &tokens, None, !needed.is_empty())? else { continue };
            let (read, _) = self.m_sites.parts.nodes(b).ok_or_else(|| error(format!("block {b}: no parts")))?;
            let rows = needed.iter().map(|i| bases.binary_search(&wanted[*i].0).map(|k| k * length + wanted[*i].2).map_err(|_| error("a base outside the run")));
            let rows: Vec<usize> = rows.collect::<Result<_, _>>()?;
            let x = d.download(&d.gather_ranges(trace.value(read)?, &single(rows)).map_err(error)?).map_err(error)?;
            let parts = by_block.get(&b).map(Vec::as_slice).unwrap_or_default();
            for (k, i) in needed.iter().enumerate() {
                let row = x.row(k);
                out[*i] = parts.iter().copied().filter(|j| row.iter().zip(&self.parts()[*j].read).map(|(a, g)| a * g).sum::<f64>() + self.parts()[*j].bias > 0.0).collect();
            }
        }
        Ok(out)
    }

    /// Per slot `(base, family)` of `slots` (a base sequence of `batch`, an edit family), an edit of a
    /// part and its position: its block uniform over the MLP blocks holding parts, its position
    /// uniform over the rows after the first (the first is the attention sink, where a transcoder
    /// block runs `M`'s own MLP), its factor 0 for a removal and uniform over the other factors for
    /// an amplification, and its part uniform over the parts firing on `M`'s read there with one
    /// part of the block drawn uniformly added (mostly a silent one: its edit tests that the part
    /// is silent in `P` where it is in `M`).
    pub fn draw_edits(&self, rng: &mut impl RngExt, batch: &Batch, slots: &[(usize, Family)]) -> Result<Vec<(Patch, usize)>, String> {
        let length = batch.length();
        let mut by_block: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, p) in self.parts().iter().enumerate() {
            by_block.entry(p.block).or_default().push(i);
        }
        let held: Vec<usize> = by_block.keys().copied().collect();
        if length < 2 {
            return Err(error("edits need sequences of two tokens or more"));
        }
        let heads: Vec<usize> = self.heads().iter().enumerate().filter(|(_, (_, held))| *held).map(|(h, _)| h).collect();
        // Per slot: a head's removal, or the parts' draws (one block's or a cut's two) and the
        // activity queries they need.
        enum Drawn {
            Head(usize, usize),
            Part(usize, usize, usize),
            Parts(usize, usize),
            Swap(usize, usize),
            Cut(usize, usize, usize),
        }
        let mut drawn = Vec::with_capacity(slots.len());
        let mut wanted = Vec::with_capacity(slots.len());
        let silent = |rng: &mut _, block: usize| -> usize {
            let candidates = &by_block[&block];
            candidates[RngExt::random_range(rng, 0..candidates.len())]
        };
        for (n, family) in slots {
            let position = rng.random_range(1..length);
            match family {
                Family::RemoveHead => {
                    if heads.is_empty() {
                        return Err(error("a head's removal, but no head both models hold"));
                    }
                    drawn.push(Drawn::Head(heads[rng.random_range(0..heads.len())], position));
                }
                Family::RemovePart | Family::AmplifyPart => {
                    if held.is_empty() {
                        return Err(error("an edit of a part, but no parts"));
                    }
                    let block = held[rng.random_range(0..held.len())];
                    let factor = if *family == Family::RemovePart { 0 } else { rng.random_range(1..FACTORS.len()) };
                    drawn.push(Drawn::Part(position, factor, silent(rng, block)));
                    wanted.push((*n, block, position));
                }
                Family::SwapPart => {
                    if held.is_empty() {
                        return Err(error("a swap of a part, but no parts"));
                    }
                    let block = held[rng.random_range(0..held.len())];
                    drawn.push(Drawn::Swap(position, silent(rng, block)));
                    wanted.push((*n, block, position));
                }
                Family::RemoveParts => {
                    if held.is_empty() {
                        return Err(error("an edit of parts, but no parts"));
                    }
                    let block = held[rng.random_range(0..held.len())];
                    drawn.push(Drawn::Parts(position, silent(rng, block)));
                    wanted.push((*n, block, position));
                }
                Family::CutConnection => {
                    // `to`'s block uniform over the blocks with parts after the first, `from`'s
                    // uniform over those before it.
                    if held.len() < 2 {
                        return Err(error("a cut connection needs parts in two blocks"));
                    }
                    let to = held[rng.random_range(1..held.len())];
                    let from = held[rng.random_range(0..held.iter().position(|b| *b == to).unwrap_or(1))];
                    drawn.push(Drawn::Cut(position, silent(rng, from), silent(rng, to)));
                    wanted.push((*n, from, position));
                    wanted.push((*n, to, position));
                }
                Family::Read => return Err(error("a read patch is not an edit")),
            }
        }
        let mut active = if wanted.is_empty() { Vec::new() } else { self.active_parts(batch, &wanted)? }.into_iter();
        let mut next = || active.next().ok_or_else(|| error("an edit without its activity"));
        let mut out = Vec::with_capacity(drawn.len());
        for d in drawn {
            // A part uniform over the firing parts and the one drawn uniformly.
            let choose = |rng: &mut _, silent: usize, mut candidates: Vec<usize>| -> usize {
                if !candidates.contains(&silent) {
                    candidates.push(silent);
                }
                candidates[RngExt::random_range(rng, 0..candidates.len())]
            };
            out.push(match d {
                Drawn::Head(head, position) => (Patch::Head { head }, position),
                Drawn::Part(position, factor, silent) => (Patch::Part { part: choose(rng, silent, next()?), factor }, position),
                Drawn::Swap(position, silent) => (Patch::Swap { part: choose(rng, silent, next()?) }, position),
                Drawn::Parts(position, silent) => {
                    // A uniform subset of the firing parts (the drawn one when none fires).
                    let firing = next()?;
                    let candidates = if firing.is_empty() { vec![silent] } else { firing };
                    (Patch::Parts { parts: subset(rng, &candidates), factor: 0 }, position)
                }
                Drawn::Cut(position, from, to) => {
                    let from = choose(rng, from, next()?);
                    (Patch::Cut { from, to: choose(rng, to, next()?) }, position)
                }
            });
        }
        Ok(out)
    }

    /// Per base sequence of `batch`, its clean experiment and `per_base` edits of parts
    /// ([`Interchange::draw_edits`]), each of a family drawn uniformly from `families`, all with
    /// `P` alone (or, with `hybrids`, under one hybrid per base drawn as [`sample`] draws it).
    pub fn sample_edits(&self, rng: &mut impl RngExt, batch: &Batch, families: &[Family], per_base: usize, hybrids: bool) -> Result<Vec<Experiment>, String> {
        let blocks = self.m_sites.entries.len();
        if families.is_empty() {
            return Err(error("edits need families"));
        }
        let mut slots = Vec::new();
        let mut hybrid_of_base = Vec::new();
        for n in 0..batch.base.len() {
            hybrid_of_base.push(if !hybrids || rng.random_range(0..2) == 0 { vec![true; blocks] } else { hybrid(rng, blocks) });
            for _ in 0..per_base {
                slots.push((n, families[rng.random_range(0..families.len())]));
            }
        }
        let mut edits = self.draw_edits(rng, batch, &slots)?.into_iter();
        let mut out = Vec::with_capacity(slots.len() + batch.base.len());
        for (n, explained) in hybrid_of_base.into_iter().enumerate() {
            for _ in 0..per_base {
                let (patch, position) = edits.next().ok_or_else(|| error("an edit not drawn"))?;
                // A cut connection takes the `from` part's write on the next base sequence.
                let source = if matches!(patch, Patch::Cut { .. } | Patch::Swap { .. }) { (n + 1) % batch.base.len() } else { n };
                out.push(Experiment { base: n, source, explained: explained.clone(), patch: Some(patch), position });
            }
            out.push(Experiment { base: n, source: n, explained, patch: None, position: 0 });
        }
        Ok(out)
    }

    /// `M` and `P` as the free functions of this module take them.
    pub fn models(&self) -> (Model<'_>, Model<'_>) {
        (Model { program: &self.m, sites: Arc::clone(&self.m_sites) }, Model { program: &self.p, sites: Arc::clone(&self.p_sites) })
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
    pub fn load<A: std::borrow::Borrow<ndarray::Array2<f64>>>(&mut self, values: &[A]) -> Result<(), String> {
        if values.len() != self.trainable.len() {
            return Err(error("one value per trainable operator required"));
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
            return Ok(Scored { bits: evaluation.bits, gradient: Vec::new() });
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
        Ok(Scored { bits: evaluation.bits, gradient })
    }
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
            let (paths, bases) = paths(&batch, &experiments, p.values(), (&[], &[]), None, 4).expect("the paths");
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
        let edits = Edits { nodes: BTreeMap::from([(7, patches)]), writes: BTreeMap::new(), cuts: BTreeMap::new(), parts: Arc::new(Vec::new()), recorded: None };
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
        let (paths, bases) = paths(&batch, &experiments, p.values(), (&[], &[]), None, 4).expect("the paths");
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
}

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
    decoder::{self, Decoder},
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
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet, HashMap},
    ops::Range,
    sync::Arc,
};

fn error(e: impl std::fmt::Display) -> String {
    format!("interchange: {e}")
}

/// One model's sites, checked against its program: the stream entering each block, each block's
/// read, the dense operators that receive gradients, per block, and where it holds each read
/// variable's value.
struct Sites {
    entries: Vec<usize>,
    reads: Vec<usize>,
    trainable: Vec<Vec<usize>>,
    values: Vec<Value>,
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
        Ok(Self { entries, reads, trainable: per_block, values })
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
    let (flat, roots) = mapped_inlined(&artifact.program)?;
    let at = |native: usize| artifact.place(native).map(|n| roots[n]).ok_or_else(|| error(format!("native node {native} is not held")));
    let entries = layers.iter().flat_map(|l| [l.stream, l.attended]).map(at).collect::<Result<_, _>>()?;
    let reads = layers.iter().flat_map(|l| [l.normed_stream, l.normed]).map(at).collect::<Result<_, _>>()?;
    Ok((flat, entries, reads))
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Patch {
    Read { variable: usize },
    Reads { variables: Vec<usize> },
}

impl Patch {
    /// The patched variables.
    pub fn variables(&self) -> &[usize] {
        match self {
            Self::Read { variable } => std::slice::from_ref(variable),
            Self::Reads { variables } => variables,
        }
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
    /// The patched block of the variables `values` (none for an unpatched experiment).
    fn block(&self, values: &[Value]) -> Result<Option<usize>, String> {
        let Some(patch) = &self.patch else { return Ok(None) };
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
/// * the patched experiment, under the same hybrid: its block uniform over the `blocks` blocks;
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
    if blocks == 0 || length == 0 || at_block.iter().any(Vec::is_empty) {
        return Err(error("every block needs a read variable to patch, and sequences need tokens"));
    }
    let mut out = Vec::with_capacity(2 * sequences);
    for n in 0..sequences {
        let explained = if rng.random_range(0..2) == 0 { vec![true; blocks] } else { hybrid(rng, blocks) };
        let candidates = &at_block[rng.random_range(0..blocks)];
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
    let mut counts: BTreeMap<&'static str, usize> = ["clean_alone", "clean_hybrid", "read_attention", "read_mlp", "read_joint"].into_iter().map(|k| (k, 0)).collect();
    for e in experiments {
        let family = match &e.patch {
            None if e.explained.iter().all(|x| *x) => "clean_alone",
            None => "clean_hybrid",
            Some(Patch::Read { variable }) if variables.get(*variable).is_some_and(|v| v.block % 2 == 0) => "read_attention",
            Some(Patch::Read { .. }) => "read_mlp",
            Some(Patch::Reads { .. }) => "read_joint",
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
        let width = device.column_blocks(&[head.embedding.ncols()]).map_err(error)?;
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
    let start = crate::library_mdl::explanation(native, layers)?;
    let program = &start.artifact.program;
    let named: BTreeMap<&str, usize> = native.operators.iter().enumerate().map(|(i, op)| (op.name.as_str(), i)).collect();
    library_reads(program, layers.len())?
        .into_iter()
        .map(|v| {
            let mut parts: Vec<(usize, Range<usize>)> = Vec::new();
            for (op, rows) in &v.parts {
                let name = &program.operators[*op].name;
                for owner in start.artifact.owners.iter().filter(|o| &o.operator == name && READS.contains(&o.role.as_str())) {
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
    // Per variable, per site: the observation path or the node, and the columns.
    let mut found: Vec<Vec<(Result<Vec<usize>, usize>, Range<usize>)>> = Vec::with_capacity(variables.len());
    for v in variables {
        let mut out = Vec::new();
        for (op, rows) in &v.parts {
            let name = native.operators.get(*op).map(|o| o.name.as_str()).ok_or_else(|| error("a read variable of an unknown native operator"))?;
            if artifact.owners.is_empty() {
                let at = *named.get(name).ok_or_else(|| error(format!("{name}: not an operator of the model")))?;
                let mut applying = flat.nodes.iter().enumerate().filter(|(_, n)| matches!(n, Node::Affine { terms, .. } if terms.first().is_some_and(|t| t.1 == at)));
                let node = match (applying.next(), applying.next()) {
                    (Some((n, _)), None) => n,
                    _ => return Err(error(format!("{name}: not applied by exactly one node"))),
                };
                out.push((Err(node), rows.clone()));
                continue;
            }
            for owner in artifact.owners.iter().filter(|o| o.native == name && READS.contains(&o.role.as_str())) {
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
    nodes: BTreeMap<usize, Vec<(usize, usize, Tensor, Tensor)>>,
}

impl Edits {
    /// The patches of a call at block rows `patches` (the patched row, the source's row, and the
    /// variables), at the sites `values` of the model that runs the call.
    fn new(d: &Device, patches: &[(usize, usize, &[usize])], values: &[Value]) -> Result<Self, String> {
        let mut masks: BTreeMap<(usize, usize, usize), Vec<f64>> = BTreeMap::new();
        for (row, source, variables) in patches {
            for v in *variables {
                for site in &values.get(*v).ok_or_else(|| error("a patch of an unknown variable"))?.sites {
                    let mask = masks.entry((site.node, *row, *source)).or_insert_with(|| vec![0.0; site.width]);
                    mask[site.columns.clone()].iter_mut().for_each(|m| *m = 1.0);
                }
            }
        }
        let mut nodes: BTreeMap<usize, Vec<(usize, usize, Tensor, Tensor)>> = BTreeMap::new();
        for ((node, row, source), take) in masks {
            let keep = take.iter().map(|m| 1.0 - m).collect();
            let (keep, take) = (d.upload_vec(1, take.len(), keep).map_err(error)?, d.upload_vec(1, take.len(), take).map_err(error)?);
            nodes.entry(node).or_default().push((row, source, keep, take));
        }
        Ok(Self { nodes })
    }

    /// The nodes the call edits.
    pub fn nodes(&self) -> BTreeSet<usize> {
        self.nodes.keys().copied().collect()
    }

    /// Node `node`'s value `value` (the call's rows in order) with each patched row's patched
    /// entries replaced by its source row's: `h ⊙ (1 − m) + s ⊙ m`, both products exact, every
    /// source row read before any row is written.
    pub fn apply(&self, d: &Device, node: usize, value: &mut Tensor) -> Result<(), String> {
        let Some(patches) = self.nodes.get(&node) else { return Ok(()) };
        let sources = patches.iter().map(|(_, s, _, _)| d.rows_of(value, *s, 1).map_err(error)).collect::<Result<Vec<_>, _>>()?;
        for ((row, _, keep, take), s) in patches.iter().zip(&sources) {
            let h = d.rows_of(value, *row, 1).map_err(error)?;
            let mut out = d.zeros(1, value.cols()).map_err(error)?;
            d.scale_columns(&mut out, &h, keep, false).map_err(error)?;
            d.scale_columns(&mut out, s, take, true).map_err(error)?;
            d.set_rows(value, *row, &out).map_err(error)?;
        }
        Ok(())
    }

    /// The transpose of [`Edits::apply`] on node `node`'s cotangent `g`: each patched row keeps
    /// `ḡ ⊙ (1 − m)` and its source row receives `ḡ ⊙ m`.
    pub fn transpose(&self, d: &Device, node: usize, g: &mut Tensor) -> Result<(), String> {
        let Some(patches) = self.nodes.get(&node) else { return Ok(()) };
        let rows = patches.iter().map(|(r, _, _, _)| d.rows_of(g, *r, 1).map_err(error)).collect::<Result<Vec<_>, _>>()?;
        let mut parts = Vec::with_capacity(patches.len());
        for ((row, source, keep, take), gr) in patches.iter().zip(&rows) {
            let mut base = d.zeros(1, g.cols()).map_err(error)?;
            d.scale_columns(&mut base, gr, keep, false).map_err(error)?;
            d.set_rows(g, *row, &base).map_err(error)?;
            let mut part = d.zeros(1, g.cols()).map_err(error)?;
            d.scale_columns(&mut part, gr, take, false).map_err(error)?;
            parts.push((*source, part));
        }
        for (source, part) in parts {
            let mut total = d.rows_of(g, source, 1).map_err(error)?;
            d.axpy(&mut total, 1.0, &part).map_err(error)?;
            d.set_rows(g, source, &total).map_err(error)?;
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
        let edited = edits.map(Edits::nodes).unwrap_or_default();
        let edit = |n: usize, trace: &DeviceTrace| -> Result<Option<Tensor>, String> {
            match edits {
                Some(edits) if edited.contains(&n) => {
                    let mut value = d.copy(trace.value(n)?).map_err(error)?;
                    edits.apply(d, n, &mut value)?;
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
        let (nodes, gradients) = self.program.vjp_values_dense_edited(tape, seeds, &keep, &self.sites.trainable[block], arithmetic, &edited, &mut hook)?;
        for (op, g) in gradients {
            match gradient.get_mut(&op) {
                Some(total) => d.axpy(total, 1.0, &g).map_err(error)?,
                None => {
                    gradient.insert(op, g);
                }
            }
        }
        let entering = if block > 0 {
            d.copy(nodes.get(&self.entry(block)).ok_or_else(|| error("no cotangent of a block's entering stream"))?).map_err(error)?
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
}

impl<'t> Path<'t> {
    fn patched(&self, block: usize) -> Option<(&'t [usize], usize)> {
        self.patch.filter(|(b, _, _)| *b == block).map(|(_, variables, source)| (variables, source))
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
                let shared = (0..limit).find(|&b| other.explained[b] != path.explained[b] || other.patched(b).is_some() || path.patched(b).is_some()).unwrap_or(limit);
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
        Self { paths, lanes, holder, length }
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
    let patches = plan.patches(block, lanes)?;
    if patches.is_empty() {
        return Ok(None);
    }
    Ok(Some(Edits::new(engine.device(), &patches, engine.values())?))
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
                let copied = d.rows_of(&stream, rows.start, rows.len()).map_err(error)?;
                d.set_rows(&mut stream, lane.rows.start, &copied).map_err(error)?;
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

/// The products' precision of the reverse passes (the data term's gradient and the sampled-label
/// factor): bfloat16 on CUDA in f32 storage (whose bfloat16 tensor cores run at least twice its
/// f32 rate), else the forward's `arithmetic`. The forward pass and the scores, the objective,
/// keep the forward's arithmetic, so `F` is unchanged. Each pass's result is one Monte Carlo draw:
/// the gradient at one weight sample, whose noise from the sample dominates its entries, and the
/// factor, whose square estimates the Gauss–Newton diagonal with a relative standard deviation near
/// one; bfloat16 operands move their entries by a few percent (`Interchange::fuse`), differently
/// at every sample.
fn factor_arithmetic(d: &Device, arithmetic: Arithmetic) -> Arithmetic {
    if d.storage() == Storage::F32 && d.with_storage(Storage::Bf16).is_ok() { Arithmetic::Bf16 } else { arithmetic }
}

/// The reverse of [`run`] from the stream buffer's cotangent `cotangent` (rows of the paths'
/// outputs): block by block backwards, each call's reverse with the patches' transposed edits
/// (each source's part added into the source's row of the same call), then each fork's rows added
/// into its parent's. Adds `P`'s parameter gradient into `gradient`; the calls' tapes stay. The
/// blocks' products run in `arithmetic`.
fn run_reverse<E: BlockEngine>(engines: [&E; 2], plan: &Plan, calls: &[Call<E::Tape>], mut cotangent: Tensor, gradient: &mut BTreeMap<usize, Tensor>, arithmetic: Arithmetic) -> Result<(), String> {
    let d = engines[0].device();
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
                    None if b == 0 => d.zeros(at, cotangent.cols()).map_err(error)?,
                    None => return Err(error("a call past the first block without its entering rows")),
                };
                let tokens: Vec<&[u32]> = call.lanes.iter().map(|l| plan.paths[plan.lanes[*l].path].tokens).collect();
                recomputed = engines[call.side].forward(b, &mut rows, &local, &tokens, call.edits.as_ref(), true)?.ok_or_else(|| error("a call kept no tape"))?;
                &recomputed
            }
        };
        engines[call.side].reverse(b, tape, &mut cotangent, &ranges, call.edits.as_ref(), (&mut *gradient, arithmetic))?;
        // Once both sides of the block are reversed, the forks made there return their rows, the
        // later lanes first (a lane forked from a lane forked at the same block returns through it).
        if index == 0 || calls[index - 1].block != b {
            for lane in plan.lanes.iter().rev().filter(|l| l.start == b) {
                if let Some(parent) = lane.parent {
                    let rows = plan.lanes[parent].rows.clone();
                    let mut total = d.rows_of(&cotangent, rows.start, rows.len()).map_err(error)?;
                    d.axpy(&mut total, 1.0, &d.rows_of(&cotangent, lane.rows.start, lane.rows.len()).map_err(error)?).map_err(error)?;
                    d.set_rows(&mut cotangent, rows.start, &total).map_err(error)?;
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
/// a fit makes them on the device whenever it scores a batch. [`Interchange::targets`] records how
/// `M` ran ([`Execution`]), and [`Interchange::evaluate_labelled`] refuses targets made another way.
pub struct Targets {
    rows: Vec<Target>,
    teacher: Option<Execution>,
}

/// How one model's blocks run in the experiments: its program block by block, or its fused decoder
/// ([`Decoder`]), and the arithmetic of the products outside the blocks (the head, the patches).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Execution {
    Program(Arithmetic),
    Fused(Arithmetic),
}

/// One model's engine in the experiments: its program block by block, or its fused decoder. Each
/// model of an [`Interchange`] takes its own, so `M`'s blocks and targets run the same way whatever
/// explanation they are compared with.
pub enum Engine<'a> {
    Program(Model<'a>),
    Fused(&'a Decoder),
}

/// A tape of an [`Engine`].
pub enum EngineTape {
    Program(DeviceTrace),
    Fused(decoder::Tape),
}

impl Engine<'_> {
    /// How it runs.
    #[must_use]
    pub fn execution(&self) -> Execution {
        match self {
            Self::Program(m) => Execution::Program(BlockEngine::arithmetic(m)),
            Self::Fused(d) => Execution::Fused(BlockEngine::arithmetic(*d)),
        }
    }
}

impl BlockEngine for Engine<'_> {
    type Tape = EngineTape;

    fn device(&self) -> &Device {
        match self {
            Self::Program(m) => BlockEngine::device(m),
            Self::Fused(d) => BlockEngine::device(*d),
        }
    }

    fn width(&self) -> usize {
        match self {
            Self::Program(m) => BlockEngine::width(m),
            Self::Fused(d) => BlockEngine::width(*d),
        }
    }

    fn blocks(&self) -> usize {
        match self {
            Self::Program(m) => BlockEngine::blocks(m),
            Self::Fused(d) => BlockEngine::blocks(*d),
        }
    }

    fn arithmetic(&self) -> Arithmetic {
        match self {
            Self::Program(m) => BlockEngine::arithmetic(m),
            Self::Fused(d) => BlockEngine::arithmetic(*d),
        }
    }

    fn values(&self) -> &[Value] {
        match self {
            Self::Program(m) => BlockEngine::values(m),
            Self::Fused(d) => BlockEngine::values(*d),
        }
    }

    fn forward(
        &self,
        block: usize,
        stream: &mut Tensor,
        ranges: &[Range<usize>],
        tokens: &[&[u32]],
        edits: Option<&Edits>,
        keep: bool,
    ) -> Result<Option<EngineTape>, String> {
        Ok(match self {
            Self::Program(m) => BlockEngine::forward(m, block, stream, ranges, tokens, edits, keep)?.map(EngineTape::Program),
            Self::Fused(d) => BlockEngine::forward(*d, block, stream, ranges, tokens, edits, keep)?.map(EngineTape::Fused),
        })
    }

    fn reverse(
        &self,
        block: usize,
        tape: &EngineTape,
        cotangent: &mut Tensor,
        ranges: &[Range<usize>],
        edits: Option<&Edits>,
        sums: (&mut BTreeMap<usize, Tensor>, Arithmetic),
    ) -> Result<(), String> {
        match (self, tape) {
            (Self::Program(m), EngineTape::Program(t)) => BlockEngine::reverse(m, block, t, cotangent, ranges, edits, sums),
            (Self::Fused(d), EngineTape::Fused(t)) => BlockEngine::reverse(*d, block, t, cotangent, ranges, edits, sums),
            _ => Err(error("a tape of another engine")),
        }
    }

    fn tape_bytes(tape: &EngineTape) -> usize {
        match tape {
            EngineTape::Program(t) => <Model<'_> as BlockEngine>::tape_bytes(t),
            EngineTape::Fused(t) => <Decoder as BlockEngine>::tape_bytes(t),
        }
    }

    fn gradient_bytes(&self) -> Result<usize, String> {
        match self {
            Self::Program(m) => BlockEngine::gradient_bytes(m),
            Self::Fused(d) => BlockEngine::gradient_bytes(*d),
        }
    }
}

/// The paths of `experiments` on `batch` over the read variables of `values` (their blocks), every
/// block run by `M` when `native`, else by each experiment's hybrid: per patched experiment its
/// source's path (to its patched block) and its base's path; per experiment the index of its base's
/// path.
fn paths<'t>(batch: &'t Batch, experiments: &'t [Experiment], values: &[Value], native: Option<&'t [bool]>, blocks: usize) -> Result<(Vec<Path<'t>>, Vec<usize>), String> {
    let (mut paths, mut bases) = (Vec::with_capacity(2 * experiments.len()), Vec::with_capacity(experiments.len()));
    for e in experiments {
        check(e, batch, blocks)?;
        let explained = native.unwrap_or(&e.explained);
        let patch = match (&e.patch, e.block(values)?) {
            (Some(patch), Some(block)) => {
                paths.push(Path { tokens: &batch.source[e.source], explained, end: block + 1, position: 0, patch: None });
                Some((block, patch.variables(), paths.len() - 1))
            }
            _ => None,
        };
        bases.push(paths.len());
        paths.push(Path { tokens: &batch.base[e.base], explained, end: blocks, position: e.position, patch });
    }
    Ok((paths, bases))
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
    let (paths, bases) = paths(batch, experiments, m.values(), Some(&native), blocks)?;
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
    Ok(Targets { rows: out, teacher: None })
}

/// The exact identity of a batch's experiments, which alone decides `M`'s targets for them (with
/// how `M` runs, [`Execution`]): the base and source tokens and every experiment.
#[derive(PartialEq, Eq, Hash)]
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
    teacher: Option<Execution>,
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
                let values = d.download(&t.mu).map_err(error)?;
                let shape = values.dim();
                let storage = t.mu.storage();
                let mu = match storage {
                    Storage::F64 => KeptValues::Double(values.into_iter().collect()),
                    // f32 and bfloat16 values are f32 values: narrowing their widened copies is exact.
                    Storage::F32 | Storage::Bf16 => KeptValues::Single(values.iter().map(|v| *v as f32).collect()),
                };
                Ok(KeptTarget { mu, shape, storage, entropy: t.entropy.clone(), head: Arc::clone(&t.head), scored: t.scored.clone() })
            })
            .collect::<Result<_, String>>()?;
        Ok(Self { rows, teacher: targets.teacher })
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
        Ok(Targets { rows, teacher: self.teacher })
    }
}

/// The targets an [`Interchange`] keeps on the host once [`Interchange::keep_targets`] asked for
/// it, by the exact identity of each batch's experiments, each batch's bytes reserved from
/// `governor` for as long as they are held.
struct TargetStore {
    governor: MemoryGovernor,
    batches: HashMap<Identity, Governed<HostTargets>>,
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
        let mut total = d.rows_of(&cotangent, r.start, r.len()).map_err(error)?;
        d.axpy(&mut total, weight, &d.rows_of(seed, at, r.len()).map_err(error)?).map_err(error)?;
        d.set_rows(&mut cotangent, r.start, &total).map_err(error)?;
        at += r.len();
    }
    Ok(cotangent)
}

/// One batch's scores and gradients ([`evaluate`], [`evaluate_labelled`]).
pub struct Evaluation {
    /// Per experiment, per base token from its position on, `KL(M_e ‖ P_e)` in bits (empty when no
    /// targets were given).
    pub bits: Vec<Vec<f64>>,
    /// The gradient of the sum of `bits` in each of `P`'s trainable operators, on the device (empty
    /// when not asked for).
    pub gradient: BTreeMap<usize, Tensor>,
    /// A draw of the Gauss–Newton factor, when asked for ([`evaluate_labelled`]).
    pub factor: Option<Factor>,
}

/// A draw of the Gauss–Newton factor of a batch's experiments: the gradient `u`, in each of `P`'s
/// trainable operators on the device, of `Σ_t log P_e(y_t)` over every scored token `t` of every
/// experiment `e`, each label `y_t` drawn from `P_e`'s own prediction there, and the number of
/// tokens `n`. The labels are independent across tokens with `E[e_y] = p`, so
/// `E[u uᵀ] = Σ_t J_tᵀ (diag p_t − p_t p_tᵀ) J_t`: `u ⊙ u / n` is an unbiased estimate of the
/// diagonal of the data term's Gauss–Newton curvature per token (the Hessian of `KL(M_e ‖ P_e)` in
/// `P`'s logits is `P`'s softmax Fisher matrix whatever `M_e` is).
pub struct Factor {
    pub gradient: BTreeMap<usize, Tensor>,
    pub tokens: usize,
}

/// The cotangent at the final normed stream `hidden` (rows × width) of `−Σ_r log P(y_r)` for one
/// label `y_r` per row drawn from `P`'s prediction `softmax(E h_r)` there with that row's entry of
/// `uniforms` (`embedding` is `E`, classes × width): per row `(p_r − e_{y_r}) E`. The logits are
/// formed `tile` rows at a time.
pub fn sampled_label_seed(d: &Device, hidden: &Tensor, embedding: &Tensor, tile: usize, uniforms: &[f64], arithmetic: Arithmetic) -> Result<Tensor, String> {
    let (rows, classes) = (hidden.rows(), embedding.rows());
    if uniforms.len() != rows || tile == 0 {
        return Err(error("one uniform per row and positive tile rows required"));
    }
    // Every buffer below is written whole (products with β = 0, the tiles' rows), so none is zeroed.
    let mut seed = d.empty(rows, hidden.cols()).map_err(error)?;
    for start in (0..rows).step_by(tile) {
        let n = tile.min(rows - start);
        let h = d.rows_of(hidden, start, n).map_err(error)?;
        let mut logits = d.empty(n, classes).map_err(error)?;
        d.gemm(&mut logits, 1.0, &h, Op::N, embedding, Op::T, 0.0, arithmetic).map_err(error)?;
        let uniforms = d.upload_vec(n, 1, uniforms[start..start + n].to_vec()).map_err(error)?;
        d.sampled_cotangent(&mut logits, &uniforms, None).map_err(error)?;
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
    evaluate_labelled((m, p), head, (batch, experiments), Some(targets), gradient, None)
}

/// [`evaluate`] of `experiments` on `batch`, with the scores only when `targets` are
/// given, and with `labels` also a draw of the Gauss–Newton factor ([`Factor`]), each scored row's
/// label drawn with its entry of `labels` (rows in experiment order): a second reverse pass through
/// the same forward pass, seeded at every scored token by [`sampled_label_seed`].
pub fn evaluate_labelled<E: BlockEngine>(
    (m, p): (&E, &E),
    head: &FixedHead,
    (batch, experiments): (&Batch, &[Experiment]),
    targets: Option<&Targets>,
    gradient: bool,
    labels: Option<&[f64]>,
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
    let (paths, bases) = paths(batch, experiments, p.values(), None, blocks)?;
    let plan = Plan::new(paths, length);
    let arithmetic = p.arithmetic();
    let (stream, calls) = run([p, m], &plan, gradient || labels.is_some())?;
    // Each experiment's scored rows, from its position on.
    let rows = outputs(&plan, &bases, experiments);
    let hidden = gather(d, &stream, &rows)?;
    let spread = |seed: &Tensor, weight: f64| spread(d, stream.rows(), &rows, seed, weight);
    let mut bits = Vec::new();
    let mut total = BTreeMap::new();
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
        let (nats, seed) = head.resident.score(d, &hidden, &target, gradient, arithmetic)?;
        bits.reserve(experiments.len());
        let mut at = 0;
        for r in &rows {
            bits.push(nats[at..at + r.len()].iter().map(|v| v / std::f64::consts::LN_2).collect());
            at += r.len();
        }
        if gradient {
            let seed = seed.ok_or_else(|| error("the head returned no cotangent"))?;
            run_reverse([p, m], &plan, &calls, spread(&seed, 1.0 / std::f64::consts::LN_2)?, &mut total, factor_arithmetic(d, arithmetic))?;
        }
    }
    let factor = match labels {
        Some(uniforms) => {
            // The label draw's logits, its seed and the pass in the factor's arithmetic.
            let factor = factor_arithmetic(d, arithmetic);
            let seed = sampled_label_seed(d, &hidden, &head.resident.embedding, head.resident.tile_rows.max(1), uniforms, factor)?;
            let mut u = BTreeMap::new();
            run_reverse([p, m], &plan, &calls, spread(&seed, 1.0)?, &mut u, factor)?;
            Some(Factor { gradient: u, tokens: hidden.rows() })
        }
        None => None,
    };
    Ok(Evaluation { bits, gradient: total, factor })
}

/// The gradient in `P`'s trainable operators of `Σ log P_e(y)` over every scored token of
/// `experiments` on `batch` (each experiment from its position on, as [`evaluate`] scores it), at
/// `P`'s loaded parameters, each `y` drawn from `P_e`'s own next-token distribution at its row with
/// that row's entry of `uniforms` (rows in experiment order). Its outer product `u uᵀ` is an unbiased
/// estimate of the Gauss–Newton matrix of the experiments' divergence `Σ KL(M_e ‖ P_e)` in nats,
/// `Σ_t J_tᵀ F_t J_t` with `F_t` the Fisher matrix of `P_e`'s softmax at `t`: the matrix that is the
/// divergence's Hessian where `P_e`'s predictions equal `M_e`'s.
pub fn sampled_label<E: BlockEngine>(m: &E, p: &E, head: &FixedHead, batch: &Batch, experiments: &[Experiment], uniforms: &[f64]) -> Result<BTreeMap<usize, Tensor>, String> {
    let evaluation = evaluate_labelled((m, p), head, (batch, experiments), None, false, Some(uniforms))?;
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
    /// `M`'s and `P`'s programs through their hidden nodes, which [`Interchange::fuse`] compiles.
    prefixes: (OperatorProgram, OperatorProgram),
    /// Once [`Interchange::fuse`] asked for fused engines, `M`'s when `M` is of the decoder family
    /// and `P`'s when `P` is, each decided by its own program alone; a model without one runs its
    /// program. `P`'s is refreshed from `P`'s program before an evaluation that follows a write to
    /// it (`stale`).
    teacher: Option<Decoder>,
    candidate: Option<RefCell<Decoder>>,
    fuse_asked: bool,
    stale: Cell<bool>,
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
        let (m_flat, m_streams, m_reads) = sites(&Artifact::native(native)?, layers)?;
        let (p_flat, p_streams, p_reads) = sites(explanation, layers)?;
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
        let m_sites = Arc::new(Sites::new(&m, &m_flat, m_streams, m_reads, &[], m_values)?);
        let p_sites = Arc::new(Sites::new(&p, &p_flat, p_streams, p_reads, trainable, p_values)?);
        Ok(Self { m, p, m_sites, p_sites, head, variables, trainable: trainable.to_vec(), prefixes: (m_prefix, p_prefix), teacher: None, candidate: None, fuse_asked: false, stale: Cell::new(false), kept: RefCell::new(None) })
    }

    /// Run each model on its fused engine from now on, its products in `arithmetic`
    /// (`Decoder::with_arithmetic`), where the device holds f32 or is the host (whose decoder rounds
    /// as `arithmetic` says) and the model is of the decoder family, each decided by its own
    /// program: `M`'s execution does not depend on the explanation it is compared with. Returns
    /// whether `P` runs fused. Not the default: in bfloat16 the decoder's products differ from the
    /// program engine by as much as the divergence itself (`mpd_engine_parity_2951`, vpd4l on an
    /// RTX 4090: up to 0.045 bits per token against means of 0.02 to 0.05, gradients 6 to 8% off).
    pub fn fuse(&mut self, arithmetic: Arithmetic) -> Result<bool, String> {
        if self.fuse_asked || (self.p.device().float64() && !self.p.device().is_host()) {
            return Ok(self.candidate.is_some());
        }
        self.fuse_asked = true;
        let device = self.p.device();
        match Decoder::new(device, &self.prefixes.0, (&self.m_sites.entries, &self.m_sites.reads, self.m.hidden()), &[]) {
            Ok(engine) => self.teacher = Some(engine.with_arithmetic(arithmetic).with_values(self.m_sites.values.clone())),
            Err(reason) => log::info!("interchange: M runs its program ({reason})"),
        }
        match Decoder::new(device, &self.prefixes.1, (&self.p_sites.entries, &self.p_sites.reads, self.p.hidden()), &self.trainable).map(|d| d.with_arithmetic(arithmetic).with_values(self.p_sites.values.clone())) {
            Ok(mut engine) => {
                engine.refresh(&self.p)?;
                self.candidate = Some(RefCell::new(engine));
                self.stale.set(false);
            }
            Err(reason) => log::info!("interchange: P runs its program ({reason})"),
        }
        Ok(self.candidate.is_some())
    }

    /// Whether `P` runs on its fused engine.
    pub fn fused(&self) -> bool {
        self.candidate.is_some()
    }

    /// `M`'s engine.
    fn teacher(&self) -> Engine<'_> {
        match &self.teacher {
            Some(engine) => Engine::Fused(engine),
            None => Engine::Program(self.models().0),
        }
    }

    /// How `M` runs: the execution every target of this interchange is made with.
    #[must_use]
    pub fn teacher_execution(&self) -> Execution {
        self.teacher().execution()
    }

    /// `M` and `P` as the free functions of this module take them.
    pub fn models(&self) -> (Model<'_>, Model<'_>) {
        (Model { program: &self.m, sites: Arc::clone(&self.m_sites) }, Model { program: &self.p, sites: Arc::clone(&self.p_sites) })
    }

    /// `P`'s program, to write its trainable operators on the device (`device_posterior`).
    pub fn explanation_mut(&mut self) -> &mut DeviceProgram {
        self.stale.set(true);
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
    /// while the process's memory budget admits them, and the same experiments on the same tokens
    /// are given those back (bit for bit) instead of running `M` again.
    pub fn targets(&self, batch: &Batch, experiments: &[Experiment]) -> Result<Targets, String> {
        let teacher = self.teacher();
        let execution = teacher.execution();
        let mut store = self.kept.try_borrow_mut().map_err(error)?;
        let identity = store.as_ref().map(|_| Identity { base: batch.base.clone(), source: batch.source.clone(), experiments: experiments.to_vec() });
        if let (Some(store), Some(identity)) = (store.as_ref(), identity.as_ref())
            && let Some(kept) = store.batches.get(identity)
            && kept.teacher == Some(execution)
        {
            return kept.restore(self.m.device());
        }
        let mut made = targets(&teacher, &self.head, batch, experiments)?;
        made.teacher = Some(execution);
        if let (Some(store), Some(identity)) = (store.as_mut(), identity) {
            store.batches.remove(&identity);
            if let Ok(reservation) = store.governor.try_reserve(HostTargets::bytes(&made, &identity), "interchange: M's targets of a batch kept on the host") {
                store.batches.insert(identity, reservation.bind(HostTargets::of(self.m.device(), &made)?));
            }
        }
        Ok(made)
    }

    /// Keep `M`'s targets on the host from now on ([`Interchange::targets`]), each batch's while
    /// `governor`'s budget admits it (made on the device each time otherwise): for a fit that
    /// scores one fixed collection of experiments again and again, where `M`'s forward pass is
    /// otherwise repeated at every scoring.
    pub fn keep_targets(&mut self, governor: &MemoryGovernor) {
        *self.kept.get_mut() = Some(TargetStore { governor: governor.clone(), batches: HashMap::new() });
    }

    /// `P`'s program, so that a fit writes each weight sample into its resident parameters.
    pub fn program_mut(&mut self) -> &mut DeviceProgram {
        self.stale.set(true);
        &mut self.p
    }

    /// `P`'s score on `experiments` against `targets` ([`evaluate`]), the gradient left on the
    /// device per trainable operator.
    pub fn evaluate_resident(&self, batch: &Batch, experiments: &[Experiment], targets: &Targets, gradient: bool) -> Result<Evaluation, String> {
        self.evaluate_labelled(batch, experiments, Some(targets), gradient, None)
    }

    /// [`evaluate_labelled`] of `P` as it is held, the gradients left on the device.
    pub fn evaluate_labelled(&self, batch: &Batch, experiments: &[Experiment], targets: Option<&Targets>, gradient: bool, labels: Option<&[f64]>) -> Result<Evaluation, String> {
        let teacher = self.teacher();
        if let Some(made) = targets.and_then(|t| t.teacher)
            && made != teacher.execution()
        {
            return Err(error(format!("targets made with M run as {made:?}, compared with M run as {:?}", teacher.execution())));
        }
        let held;
        let p = match &self.candidate {
            Some(engine) => {
                if self.stale.replace(false) {
                    engine.try_borrow_mut().map_err(error)?.refresh(&self.p)?;
                }
                held = engine.try_borrow().map_err(error)?;
                Engine::Fused(&*held)
            }
            None => Engine::Program(self.models().1),
        };
        evaluate_labelled((&teacher, &p), &self.head, (batch, experiments), targets, gradient, labels)
    }

    /// [`sampled_label`] at `P`'s loaded parameters, the gradient left on the device per trainable
    /// operator.
    pub fn sampled_label_resident(&self, batch: &Batch, experiments: &[Experiment], uniforms: &[f64]) -> Result<BTreeMap<usize, Tensor>, String> {
        let evaluation = self.evaluate_labelled(batch, experiments, None, false, Some(uniforms))?;
        Ok(evaluation.factor.ok_or_else(|| error("no Gauss–Newton factor"))?.gradient)
    }

    /// Load `P`'s trainable operators, in the order given to [`Interchange::new`].
    pub fn load(&mut self, values: &[ndarray::Array2<f64>]) -> Result<(), String> {
        if values.len() != self.trainable.len() {
            return Err(error("one value per trainable operator required"));
        }
        for (&op, value) in self.trainable.iter().zip(values) {
            let tensor = self.p.device().upload(value.view()).map_err(error)?;
            self.p.replace_dense_parameter(op, tensor)?;
        }
        self.stale.set(true);
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
            }
        }
    }

    /// The squared Gauss–Newton factor ([`Factor`]) estimates the diagonal of `Σ_r J_rᵀ F_r J_r`
    /// over a batch's scored rows `r`, `F_r = diag p_r − p_r p_rᵀ` at `P`'s prediction `p_r`. The
    /// exact diagonal is `Σ_r Σ_c p_rc (J_rᵀ (p_r − e_c))²`: one reverse pass per row and class
    /// through the same forward pass, seeded at that row alone. The draws' mean of `u²` must lie
    /// within five standard errors of it, for each operator's trace and for its largest entry.
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
        let (paths, bases) = paths(&batch, &experiments, p.values(), None, 4).expect("the paths");
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
                run_reverse([&p, &m], &plan, &calls, cotangent, &mut gradient, p.arithmetic()).expect("the reverse pass");
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
        let draws = 400;
        let mut sums: BTreeMap<usize, (ndarray::Array2<f64>, ndarray::Array2<f64>, f64, f64)> = BTreeMap::new();
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..draws {
            let uniforms: Vec<f64> = (0..hidden.nrows()).map(|_| rng.random::<f64>()).collect();
            let evaluation = ic.evaluate_labelled(&batch, &experiments, None, false, Some(&uniforms)).expect("a draw");
            let factor = evaluation.factor.expect("the factor");
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
            check(format!("{name} trace"), *trace, *trace_square, expected.sum());
            let (at, largest) = expected.indexed_iter().fold(((0, 0), 0.0), |best, (at, v)| if *v > best.1 { (at, *v) } else { best });
            check(format!("{name} entry {at:?}"), sum[at], square[at], largest);
        }
    }
}

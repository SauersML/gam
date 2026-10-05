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
//! * and patches at one position `t₀` of distinct blocks (none, one read patch, a joint read
//!   patch of several variables of one block, or the complements at a set of blocks), applied by
//!   both models; at a block `B`, on the read's row `t₀` only (the variables are a direction at a
//!   position):
//!   - a read patch of one of the read variables at `B` (an MLP function's gate direction `g_i`,
//!     or an attention function's query, key or value read map, a subspace), with orthonormal
//!     basis `Q`: the read becomes `h + (s − h) Q Qᵀ`, the source's coordinates in the variable;
//!     a joint read patch of several variables takes `Q` a basis of their span, all of them at
//!     once;
//!   - a complement patch at `B`, with `Q` an orthonormal basis of the span of all of `P`'s read
//!     directions at `B`: the read becomes `s + (h − s) Q Qᵀ`, the source's coordinates outside
//!     every read of `P`. `P` predicts no effect, so `M` must show none.
//!
//! `h` is the model's read value at `B` and `t₀` on `x` and `s` its own read value at `B` and `t₀`
//! on `x′`: `M`'s for `M_e`, the same hybrid's for `P_e`. Only the block's projections read the read
//! node, so the patch is path-specific: the residual passthrough keeps the base's stream. `Q` is
//! data, computed from `P`'s current reads ([`design`]): the right singular vectors whose singular
//! values are resolved from zero by their rounding band. No gradient passes through it. The
//! gradient does pass through `s`: a read cotangent `ḡ` splits into the base's `ḡ − ḡ Q Qᵀ` and
//! the source's `ḡ Q Qᵀ` (the complement's the other way round), and the source's part flows back
//! through the hybrid's run on `x′`.
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
//! at once. An evaluation makes three passes: `M` under each patch, from the patched block on and
//! entering at `M`'s clean stream; the hybrids on the sources, up to the patched block; the
//! hybrids on the bases. `M`'s clean runs are made once per batch ([`Teacher`]). The reverse pass
//! runs the bases' blocks backwards, then the sources' from their patched reads.

use crate::{
    artifact::Artifact,
    artifact_device::mapped_inlined,
    device_program::{DeviceProgram, DeviceTrace},
    operator_program::{FamilyInputs, Node, OperatorProgram, SequenceLayout, SlotValues},
    resident_causal_fit::fixed_head_target::{Head, ResidentHead, Target},
    run_check::LayerNodes,
};
use gam_gpu::tensor::{Arithmetic, ColumnBlocks, Device, Op, Tensor};
use gam_linalg::decompose::svd;
use ndarray::{ArrayView2, Axis, s};
use rand::RngExt;
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    sync::Arc,
};

fn error(e: impl std::fmt::Display) -> String {
    format!("interchange: {e}")
}

/// One model's sites, checked against its program: the stream entering each block, each block's
/// read, and the dense operators that receive gradients, per block.
struct Sites {
    entries: Vec<usize>,
    reads: Vec<usize>,
    trainable: Vec<Vec<usize>>,
}

impl Sites {
    fn new(program: &DeviceProgram, flat: &OperatorProgram, entries: Vec<usize>, reads: Vec<usize>, trainable: &[usize]) -> Result<Self, String> {
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
        Ok(Self { entries, reads, trainable: per_block })
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
    /// stream alone.
    pub fn new(program: &'a DeviceProgram, flat: &OperatorProgram, entries: Vec<usize>, reads: Vec<usize>, trainable: &[usize]) -> Result<Self, String> {
        Ok(Self { program, sites: Arc::new(Sites::new(program, flat, entries, reads, trainable)?) })
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

    fn read(&self, block: usize) -> usize {
        self.sites.reads[block]
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

/// One of `P`'s read variables at block `block`'s input: the span of the rows `rows` of each of its
/// dense operators `operator` reading that input (`parts`). A gated MLP function reads through two
/// operators, its gate's and its input's rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadVariable {
    pub block: usize,
    pub parts: Vec<(usize, Range<usize>)>,
}

/// The read variables of a library explanation of `layers` layers (`library_mdl::explanation`):
/// per head its query, key and value read maps, per MLP function its gate direction (with its up
/// direction for a gated law).
pub fn library_reads(program: &OperatorProgram, layers: usize) -> Result<Vec<ReadVariable>, String> {
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

/// A patch: one read variable (an index into the variables); the distinct read variables
/// `variables` (ascending, all at one block) jointly; or at each of the distinct blocks `blocks`
/// (ascending) the complement of every read there. The complements at different blocks are
/// distinct variables, so patching several at once is one joint intervention whose order does not
/// matter; cancellation between blocks shows only under such joint patches. A joint read patch
/// shows what single ones miss: many reads that matter little one at a time and much together.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Patch {
    Read { variable: usize },
    Reads { variables: Vec<usize> },
    Complement { blocks: Vec<usize> },
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
    /// The patched blocks, ascending (none for an unpatched experiment).
    fn blocks(&self, variables: &[ReadVariable]) -> Result<Vec<usize>, String> {
        Ok(match &self.patch {
            None => Vec::new(),
            Some(Patch::Read { variable }) => vec![variables.get(*variable).ok_or_else(|| error("a patch of an unknown variable"))?.block],
            Some(Patch::Reads { variables: chosen }) => {
                let block = |i: &usize| variables.get(*i).map(|v| v.block).ok_or_else(|| error("a patch of an unknown variable"));
                let first = block(chosen.first().ok_or_else(|| error("a joint read patch of no variable"))?)?;
                for (i, w) in chosen.iter().zip(chosen.iter().skip(1)) {
                    if i >= w || block(w)? != first {
                        return Err(error("a joint read patch needs distinct ascending variables of one block"));
                    }
                }
                vec![first]
            }
            Some(Patch::Complement { blocks }) => {
                if blocks.is_empty() || blocks.windows(2).any(|w| w[0] >= w[1]) {
                    return Err(error("a complement patch needs distinct ascending blocks"));
                }
                blocks.clone()
            }
        })
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
/// hybrid, single read patches of attention and of MLP variables, joint read patches, and
/// complement patches.
pub fn census(experiments: &[Experiment], variables: &[ReadVariable]) -> BTreeMap<&'static str, usize> {
    let mut counts: BTreeMap<&'static str, usize> =
        ["clean_alone", "clean_hybrid", "read_attention", "read_mlp", "read_joint", "complement"].into_iter().map(|k| (k, 0)).collect();
    for e in experiments {
        let family = match &e.patch {
            None if e.explained.iter().all(|x| *x) => "clean_alone",
            None => "clean_hybrid",
            Some(Patch::Read { variable }) if variables.get(*variable).is_some_and(|v| v.block % 2 == 0) => "read_attention",
            Some(Patch::Read { .. }) => "read_mlp",
            Some(Patch::Reads { .. }) => "read_joint",
            Some(Patch::Complement { .. }) => "complement",
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

/// The span of a patch's directions in the `d`-dimensional stream: no direction, every direction,
/// or a proper subspace with an orthonormal basis `Q` (`d × r`, `0 < r < d`).
enum Span<T> {
    Empty,
    Whole,
    Part(T),
}

/// A patch's directions: the block, the kind, and their span.
struct Basis {
    block: usize,
    complement: bool,
    span: Span<Tensor>,
}

impl Basis {
    /// Whether the patch takes the source's rows whole (`Some(true)`: a read of every direction,
    /// a complement of none) or keeps the base's (`Some(false)`: a read of none, a complement of
    /// every direction); `None` when it mixes them through `Q`.
    fn extreme(&self) -> Option<bool> {
        match self.span {
            Span::Empty => Some(self.complement),
            Span::Whole => Some(!self.complement),
            Span::Part(_) => None,
        }
    }
}

/// The patch directions of each experiment at `P`'s reads when it was made, per patched block in
/// ascending order: the experiment design, which is data and carries no gradient.
pub struct Design {
    bases: Vec<Vec<Arc<Basis>>>,
}

/// The span of the rows of `rows`: their right singular vectors whose singular values exceed the
/// decomposition's rounding band, as an orthonormal basis `d × r` unless none or all `d` are
/// resolved from zero.
fn span(rows: ArrayView2<'_, f64>) -> Result<Span<ndarray::Array2<f64>>, String> {
    let decomposition = svd(rows, false).map_err(error)?;
    let rank = decomposition.singular_values.iter().filter(|s| **s > decomposition.band).count();
    Ok(match rank {
        0 => Span::Empty,
        r if r == rows.ncols() => Span::Whole,
        r => Span::Part(decomposition.vt.slice(s![..r, ..]).t().to_owned()),
    })
}

/// The patch directions of `experiments` at `P`'s current values of its read `variables`.
pub fn design(p: &Model, variables: &[ReadVariable], experiments: &[Experiment]) -> Result<Design, String> {
    let d = p.program.device();
    let rows_of = |op: usize, rows: &Range<usize>| -> Result<ndarray::Array2<f64>, String> {
        let all = p.program.dense(op)?;
        if rows.end > all.rows() || rows.is_empty() || all.cols() != p.width() {
            return Err(error("a read variable outside its operator"));
        }
        d.download(&d.rows_of(all, rows.start, rows.len()).map_err(error)?).map_err(error)
    };
    design_with(d, p.blocks(), variables, experiments, rows_of)
}

/// The patch directions of `experiments` over `blocks` blocks, with `rows_of(operator, rows)`
/// giving those rows of one of `P`'s operators.
fn design_with(
    d: &Device,
    blocks: usize,
    variables: &[ReadVariable],
    experiments: &[Experiment],
    rows_of: impl Fn(usize, &Range<usize>) -> Result<ndarray::Array2<f64>, String>,
) -> Result<Design, String> {
    let rows_of = |v: &ReadVariable| -> Result<ndarray::Array2<f64>, String> {
        let parts = v.parts.iter().map(|(op, rows)| rows_of(*op, rows)).collect::<Result<Vec<_>, String>>()?;
        let views: Vec<_> = parts.iter().map(|m| m.view()).collect();
        ndarray::concatenate(Axis(0), &views).map_err(error)
    };
    let upload = |q: Span<ndarray::Array2<f64>>| -> Result<Span<Tensor>, String> {
        Ok(match q {
            Span::Empty => Span::Empty,
            Span::Whole => Span::Whole,
            Span::Part(q) => Span::Part(d.upload(q.view()).map_err(error)?),
        })
    };
    let mut reads: BTreeMap<usize, Arc<Basis>> = BTreeMap::new();
    let mut complements: BTreeMap<usize, Arc<Basis>> = BTreeMap::new();
    let mut bases = Vec::with_capacity(experiments.len());
    for e in experiments {
        let mut basis = Vec::new();
        match &e.patch {
            None => {}
            Some(Patch::Read { variable }) => basis.push(match reads.get(variable) {
                Some(b) => Arc::clone(b),
                None => {
                    let v = variables.get(*variable).ok_or_else(|| error("a patch of an unknown variable"))?;
                    let b = Arc::new(Basis { block: v.block, complement: false, span: upload(span(rows_of(v)?.view())?)? });
                    reads.insert(*variable, Arc::clone(&b));
                    b
                }
            }),
            Some(Patch::Reads { variables: chosen }) => {
                e.blocks(variables)?;
                let parts = chosen.iter().map(|i| rows_of(&variables[*i])).collect::<Result<Vec<_>, _>>()?;
                let views: Vec<_> = parts.iter().map(|m| m.view()).collect();
                let q = span(ndarray::concatenate(Axis(0), &views).map_err(error)?.view())?;
                basis.push(Arc::new(Basis { block: variables[chosen[0]].block, complement: false, span: upload(q)? }));
            }
            Some(Patch::Complement { .. }) => {
                for block in e.blocks(variables)? {
                    basis.push(match complements.get(&block) {
                        Some(b) => Arc::clone(b),
                        None => {
                            if block >= blocks {
                                return Err(error("a complement patch of an unknown block"));
                            }
                            let parts = variables.iter().filter(|v| v.block == block).map(&rows_of).collect::<Result<Vec<_>, _>>()?;
                            let q = if parts.is_empty() {
                                Span::Empty
                            } else {
                                let views: Vec<_> = parts.iter().map(|m| m.view()).collect();
                                span(ndarray::concatenate(Axis(0), &views).map_err(error)?.view())?
                            };
                            let b = Arc::new(Basis { block, complement: true, span: upload(q)? });
                            complements.insert(block, Arc::clone(&b));
                            b
                        }
                    });
                }
            }
        }
        bases.push(basis);
    }
    Ok(Design { bases })
}

/// Rows `at..at + s.rows()` of the read `value` patched with the source's `s`: `h + (s − h) Q Qᵀ`
/// (read) or `s + (h − s) Q Qᵀ` (complement).
fn exchange(d: &Device, value: &mut Tensor, at: usize, s: &Tensor, basis: &Basis, arithmetic: Arithmetic) -> Result<(), String> {
    let rows = s.rows();
    let Span::Part(q) = &basis.span else {
        return if basis.extreme() == Some(true) { d.set_rows(value, at, s).map_err(error) } else { Ok(()) };
    };
    let h = d.rows_of(value, at, rows).map_err(error)?;
    let mut difference = d.copy(s).map_err(error)?;
    d.axpy(&mut difference, -1.0, &h).map_err(error)?;
    let mut coordinates = d.zeros(rows, q.cols()).map_err(error)?;
    d.gemm(&mut coordinates, 1.0, &difference, Op::N, q, Op::N, 0.0, arithmetic).map_err(error)?;
    let (mut out, sign) = if basis.complement { (d.copy(s).map_err(error)?, -1.0) } else { (h, 1.0) };
    d.gemm(&mut out, sign, &coordinates, Op::N, q, Op::T, 1.0, arithmetic).map_err(error)?;
    d.set_rows(value, at, &out).map_err(error)
}

/// The transpose of [`exchange`] on rows `at..at + rows` of the read cotangent `g`: the base's
/// part stays in `g`, the source's is returned (none when it is zero).
fn exchange_cotangent(d: &Device, g: &mut Tensor, at: usize, rows: usize, basis: &Basis, arithmetic: Arithmetic) -> Result<Option<Tensor>, String> {
    let Span::Part(q) = &basis.span else {
        if basis.extreme() != Some(true) {
            return Ok(None);
        }
        let source = d.rows_of(g, at, rows).map_err(error)?;
        d.set_rows(g, at, &d.zeros(rows, g.cols()).map_err(error)?).map_err(error)?;
        return Ok(Some(source));
    };
    let gr = d.rows_of(g, at, rows).map_err(error)?;
    let mut coordinates = d.zeros(rows, q.cols()).map_err(error)?;
    d.gemm(&mut coordinates, 1.0, &gr, Op::N, q, Op::N, 0.0, arithmetic).map_err(error)?;
    let mut inside = d.zeros(rows, g.cols()).map_err(error)?;
    d.gemm(&mut inside, 1.0, &coordinates, Op::N, q, Op::T, 0.0, arithmetic).map_err(error)?;
    let mut outside = gr;
    d.axpy(&mut outside, -1.0, &inside).map_err(error)?;
    let (base, source) = if basis.complement { (inside, outside) } else { (outside, inside) };
    d.set_rows(g, at, &base).map_err(error)?;
    Ok(Some(source))
}

/// A block engine: one model's blocks (block `2l` is layer `l`'s attention, `2l + 1` its MLP) run
/// on rows of one stream buffer in place. A call names its rows as ranges of the buffer, each one
/// sequence's positions `0..T` (`tokens` holds each range's sequence); attention is causal within
/// each range. The forward pass replaces each range's rows by the stream after the block (after
/// the last block, the final normed stream); block 0 reads the tokens. `read`, when given, edits
/// the block's read (the normed stream its projections read; the call's rows in range order)
/// before the projections. The reverse pass replaces the rows' cotangent by the cotangent of the
/// stream entering the block, calls `read` on the read's cotangent (the transposed edit), and adds
/// the gradient of the trainable operators the block uses into `gradient`.
pub trait BlockEngine {
    type Tape;

    fn device(&self) -> &Device;

    /// The stream's width, and the number of blocks.
    fn width(&self) -> usize;
    fn blocks(&self) -> usize;

    /// The precision of the products, which the patches use too.
    fn arithmetic(&self) -> Arithmetic;

    fn forward(
        &self,
        block: usize,
        stream: &mut Tensor,
        ranges: &[Range<usize>],
        tokens: &[&[u32]],
        read: Option<&mut dyn FnMut(&mut Tensor) -> Result<(), String>>,
        keep: bool,
    ) -> Result<Option<Self::Tape>, String>;

    fn reverse(
        &self,
        block: usize,
        tape: Self::Tape,
        cotangent: &mut Tensor,
        ranges: &[Range<usize>],
        read: Option<&mut dyn FnMut(&mut Tensor) -> Result<(), String>>,
        gradient: &mut BTreeMap<usize, Tensor>,
    ) -> Result<(), String>;
}

/// The ranges' rows of `t` stacked in order.
fn gather(d: &Device, t: &Tensor, ranges: &[Range<usize>]) -> Result<Tensor, String> {
    let mut out = d.zeros(ranges.iter().map(ExactSizeIterator::len).sum(), t.cols()).map_err(error)?;
    let mut at = 0;
    for r in ranges {
        d.set_rows(&mut out, at, &d.rows_of(t, r.start, r.len()).map_err(error)?).map_err(error)?;
        at += r.len();
    }
    Ok(out)
}

/// `values`' rows written back to the ranges' rows of `t`, in order.
fn scatter(d: &Device, t: &mut Tensor, ranges: &[Range<usize>], values: &Tensor) -> Result<(), String> {
    let mut at = 0;
    for r in ranges {
        d.set_rows(t, r.start, &d.rows_of(values, at, r.len()).map_err(error)?).map_err(error)?;
        at += r.len();
    }
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

    fn forward(
        &self,
        block: usize,
        stream: &mut Tensor,
        ranges: &[Range<usize>],
        tokens: &[&[u32]],
        mut read: Option<&mut dyn FnMut(&mut Tensor) -> Result<(), String>>,
        keep: bool,
    ) -> Result<Option<DeviceTrace>, String> {
        let d = self.program.device();
        let entry = if block == 0 { None } else { Some((self.entry(block), gather(d, stream, ranges)?)) };
        let node = self.read(block);
        let edit = |n: usize, trace: &DeviceTrace| -> Result<Option<Tensor>, String> {
            match read.as_mut() {
                Some(read) if n == node => {
                    let mut value = d.copy(trace.value(n)?).map_err(error)?;
                    read(&mut value)?;
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
        tape: DeviceTrace,
        cotangent: &mut Tensor,
        ranges: &[Range<usize>],
        mut read: Option<&mut dyn FnMut(&mut Tensor) -> Result<(), String>>,
        gradient: &mut BTreeMap<usize, Tensor>,
    ) -> Result<(), String> {
        let d = self.program.device();
        let node = self.read(block);
        let seeds = BTreeMap::from([(self.end(block), gather(d, cotangent, ranges)?)]);
        let edited: BTreeSet<usize> = read.as_ref().map(|_| node).into_iter().collect();
        let mut keep: Vec<usize> = edited.iter().copied().collect();
        if block > 0 {
            keep.push(self.entry(block));
        }
        let mut hook = |n: usize, g: &mut Tensor| -> Result<(), String> {
            match read.as_mut() {
                Some(read) if n == node => read(g),
                _ => Ok(()),
            }
        };
        let arithmetic = self.program.arithmetic();
        let (nodes, gradients) = self.program.vjp_values_dense_edited(&tape, seeds, &keep, &self.sites.trainable[block], arithmetic, &edited, &mut hook)?;
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
}

/// One sequence's way through the blocks: run by the hybrid `explained` (per block whether `P`
/// runs it) up to block `end`, with patches at row `position` of distinct blocks, each with its
/// directions and the path whose read at that block is the source.
struct Path<'t> {
    tokens: &'t [u32],
    explained: &'t [bool],
    end: usize,
    position: usize,
    patches: Vec<(Arc<Basis>, usize)>,
}

impl Path<'_> {
    fn patched(&self, block: usize) -> Option<&(Arc<Basis>, usize)> {
        self.patches.iter().find(|(basis, _)| basis.block == block)
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
    /// the source's row, and the directions.
    fn patches(&self, block: usize, lanes: &[usize]) -> Result<Vec<(usize, usize, Arc<Basis>)>, String> {
        let at = |l: usize| lanes.iter().position(|x| *x == l).map(|i| i * self.length);
        let mut out = Vec::new();
        for &l in lanes {
            let path = &self.paths[self.lanes[l].path];
            if let Some((basis, source)) = path.patched(block) {
                let row = at(l).ok_or_else(|| error("a patched lane outside its call"))? + path.position;
                let source = at(self.lane(*source, block)).ok_or_else(|| error("a patch's source outside its call"))? + path.position;
                out.push((row, source, Arc::clone(basis)));
            }
        }
        Ok(out)
    }
}

/// One engine call of a forward pass: its block, side (0 for `P`, 1 for `M`), lanes and tape.
struct Call<T> {
    block: usize,
    side: usize,
    lanes: Vec<usize>,
    tape: Option<T>,
}

/// Run `plan` on the engines `[P, M]` (module note): block by block, the lanes forking there copy
/// their parent's rows, then per side one call over the lanes running that side, the patches
/// applied in the read hook from the source rows of the same call. Returns the stream buffer and
/// the calls (their tapes when `keep`).
fn run<E: BlockEngine>(engines: [&E; 2], plan: &Plan, keep: bool) -> Result<(Tensor, Vec<Call<E::Tape>>), String> {
    let (d, width, blocks, arithmetic) = (engines[0].device(), engines[0].width(), engines[0].blocks(), engines[0].arithmetic());
    let mut stream = d.zeros(plan.lanes.len() * plan.length, width).map_err(error)?;
    let mut calls = Vec::new();
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
            let patches = plan.patches(b, &lanes)?;
            let mut edit = |read: &mut Tensor| -> Result<(), String> {
                let sources = patches.iter().map(|(_, s, _)| d.rows_of(read, *s, 1).map_err(error)).collect::<Result<Vec<_>, _>>()?;
                for ((row, _, basis), s) in patches.iter().zip(&sources) {
                    exchange(d, read, *row, s, basis, arithmetic)?;
                }
                Ok(())
            };
            let read: Option<&mut dyn FnMut(&mut Tensor) -> Result<(), String>> = if patches.is_empty() { None } else { Some(&mut edit) };
            let tape = engines[side].forward(b, &mut stream, &ranges, &tokens, read, keep)?;
            calls.push(Call { block: b, side, lanes, tape });
        }
    }
    Ok((stream, calls))
}

/// The reverse of [`run`] from the stream buffer's cotangent `cotangent` (rows of the paths'
/// outputs): block by block backwards, each call's reverse with the patches' transposed edits
/// (each source's part added into the source's row of the same call), then each fork's rows added
/// into its parent's. Adds `P`'s parameter gradient into `gradient`.
fn run_reverse<E: BlockEngine>(engines: [&E; 2], plan: &Plan, calls: Vec<Call<E::Tape>>, mut cotangent: Tensor, gradient: &mut BTreeMap<usize, Tensor>) -> Result<(), String> {
    let (d, arithmetic) = (engines[0].device(), engines[0].arithmetic());
    let mut calls = calls;
    while let Some(call) = calls.pop() {
        let b = call.block;
        let ranges: Vec<Range<usize>> = call.lanes.iter().map(|l| plan.lanes[*l].rows.clone()).collect();
        let patches = plan.patches(b, &call.lanes)?;
        let mut transpose = |g: &mut Tensor| -> Result<(), String> {
            for (row, source, basis) in &patches {
                if let Some(part) = exchange_cotangent(d, g, *row, 1, basis, arithmetic)? {
                    let mut total = d.rows_of(g, *source, 1).map_err(error)?;
                    d.axpy(&mut total, 1.0, &part).map_err(error)?;
                    d.set_rows(g, *source, &total).map_err(error)?;
                }
            }
            Ok(())
        };
        let read: Option<&mut dyn FnMut(&mut Tensor) -> Result<(), String>> = if patches.is_empty() { None } else { Some(&mut transpose) };
        let tape = call.tape.ok_or_else(|| error("a call kept no tape"))?;
        engines[call.side].reverse(b, tape, &mut cotangent, &ranges, read, gradient)?;
        // Once both sides of the block are reversed, the forks made there return their rows, the
        // later lanes first (a lane forked from a lane forked at the same block returns through it).
        if calls.last().is_none_or(|c| c.block != b) {
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
/// targets every score of `P` on them is measured against. They do not depend on `P`, so a fit
/// makes them once per batch ([`targets`]) and keeps them ([`Targets::write`], [`Targets::read`]).
pub struct Targets {
    rows: Vec<Target>,
}

/// The paths of `experiments` on `batch` with the directions of `design`, every block run by `M`
/// when `native`, else by each experiment's hybrid: per patched experiment its source's path (to
/// its last patched block) and its base's path; per experiment the index of its base's path.
fn paths<'t>(batch: &'t Batch, experiments: &'t [Experiment], design: &Design, native: Option<&'t [bool]>, blocks: usize) -> Result<(Vec<Path<'t>>, Vec<usize>), String> {
    if design.bases.len() != experiments.len() {
        return Err(error("the design does not match the experiments"));
    }
    let (mut paths, mut bases) = (Vec::with_capacity(2 * experiments.len()), Vec::with_capacity(experiments.len()));
    for (e, directions) in experiments.iter().zip(&design.bases) {
        check(e, batch, blocks)?;
        let explained = native.unwrap_or(&e.explained);
        let mut patches = Vec::with_capacity(directions.len());
        if let Some(last) = directions.last() {
            paths.push(Path { tokens: &batch.source[e.source], explained, end: last.block + 1, position: 0, patches: Vec::new() });
            patches.extend(directions.iter().map(|basis| (Arc::clone(basis), paths.len() - 1)));
        }
        bases.push(paths.len());
        paths.push(Path { tokens: &batch.base[e.base], explained, end: blocks, position: e.position, patches });
    }
    Ok((paths, bases))
}

/// Per experiment, the rows of `stream` its base's path ends on, from the experiment's position on.
fn outputs(plan: &Plan, bases: &[usize], experiments: &[Experiment]) -> Vec<Range<usize>> {
    bases.iter().zip(experiments).map(|(p, e)| plan.lanes[plan.holder[*p]].rows.start + e.position..plan.lanes[plan.holder[*p]].rows.end).collect()
}

/// `M`'s targets for `experiments` on `batch` under the patch directions of `design`: `M` on every
/// base and source, the patched runs forking from their base's clean run at the first patched
/// block ([`Plan`]).
pub fn targets<E: BlockEngine>(m: &E, head: &FixedHead, batch: &Batch, experiments: &[Experiment], design: &Design) -> Result<Targets, String> {
    let d = m.device();
    let blocks = m.blocks();
    let native = vec![false; blocks];
    let (paths, bases) = paths(batch, experiments, design, Some(&native), blocks)?;
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

/// A fingerprint of `experiments` on `batch` (their tokens, hybrids, patches and positions), so that
/// kept targets are read back only for the experiments they were made for.
pub fn fingerprint(batch: &Batch, experiments: &[Experiment]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (&batch.base, &batch.source, experiments).hash(&mut hasher);
    hasher.finish()
}

impl Targets {
    /// Write the targets to `path` with the `fingerprint` of their experiments: the fingerprint, then
    /// per experiment its rows and width, `μ` row by row in the device's storage precision (four
    /// bytes per value in f32 storage, eight in float64) and the rows' `Σ p log p` in float64, all
    /// little-endian.
    pub fn write(&self, d: &Device, path: &std::path::Path, fingerprint: u64) -> Result<(), String> {
        use std::io::Write;
        let partial = path.with_extension("partial");
        let mut file = std::io::BufWriter::new(std::fs::File::create(&partial).map_err(error)?);
        let wide = d.float64();
        file.write_all(&fingerprint.to_le_bytes()).map_err(error)?;
        file.write_all(&[u8::from(wide)]).map_err(error)?;
        file.write_all(&(self.rows.len() as u64).to_le_bytes()).map_err(error)?;
        for t in &self.rows {
            let mu = d.download(&t.mu).map_err(error)?;
            file.write_all(&(mu.nrows() as u64).to_le_bytes()).map_err(error)?;
            file.write_all(&(mu.ncols() as u64).to_le_bytes()).map_err(error)?;
            for v in mu.iter() {
                if wide {
                    file.write_all(&v.to_le_bytes()).map_err(error)?;
                } else {
                    file.write_all(&(*v as f32).to_le_bytes()).map_err(error)?;
                }
            }
            for v in &t.entropy {
                file.write_all(&v.to_le_bytes()).map_err(error)?;
            }
        }
        file.into_inner().map_err(error)?.sync_all().map_err(error)?;
        std::fs::rename(&partial, path).map_err(error)
    }

    /// The targets [`Targets::write`] wrote to `path`, on the device `d` of `head`; none when they
    /// were made for other experiments than those of `fingerprint`.
    pub fn read(d: &Device, head: &FixedHead, path: &std::path::Path, fingerprint: u64) -> Result<Option<Self>, String> {
        let bytes = std::fs::read(path).map_err(|e| error(format!("{}: {e}", path.display())))?;
        let mut at = 0usize;
        let mut take = |n: usize| -> Result<&[u8], String> {
            let part = bytes.get(at..at + n).ok_or_else(|| error(format!("{}: truncated targets", path.display())))?;
            at += n;
            Ok(part)
        };
        let word = |b: &[u8]| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
        if word(take(8)?) != fingerprint {
            return Ok(None);
        }
        let wide = take(1)?[0] == 1;
        let count = word(take(8)?) as usize;
        let mut rows = Vec::with_capacity(count);
        for _ in 0..count {
            let (r, c) = (word(take(8)?) as usize, word(take(8)?) as usize);
            let values: Vec<f64> = if wide {
                take(8 * r * c)?.chunks_exact(8).map(word).map(f64::from_bits).collect()
            } else {
                take(4 * r * c)?.chunks_exact(4).map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))).collect()
            };
            let entropy = take(8 * r)?.chunks_exact(8).map(word).map(f64::from_bits).collect();
            let mu = d.upload_vec(r, c, values).map_err(error)?;
            rows.push(Target { mu: Arc::new(mu), entropy, head: Arc::clone(&head.head), scored: None });
        }
        Ok(Some(Self { rows }))
    }
}

/// The experiments' data term and its gradient.
pub struct Evaluation {
    /// Per experiment, per base token from its position on, `KL(M_e ‖ P_e)` in bits.
    pub bits: Vec<Vec<f64>>,
    /// The gradient of the sum of `bits` in each of `P`'s trainable operators, on the device (empty
    /// when not asked for).
    pub gradient: BTreeMap<usize, Tensor>,
}

/// `KL(M_e ‖ P_e)` per token for each of `experiments` on `batch` from its position on (module
/// note), at `P`'s current parameters and the patch directions of `design`, against `M`'s
/// `targets` for them, and with `gradient` its sum's gradient in `P`'s trainable operators. `M`
/// runs here only as the hybrids' blocks it keeps.
pub fn evaluate<E: BlockEngine>(m: &E, p: &E, head: &FixedHead, batch: &Batch, targets: &Targets, experiments: &[Experiment], design: &Design, gradient: bool) -> Result<Evaluation, String> {
    let d = p.device();
    let (blocks, length, width) = (p.blocks(), batch.length, p.width());
    if m.blocks() != blocks || m.width() != width || design.bases.len() != experiments.len() || targets.rows.len() != experiments.len() {
        return Err(error("models, design or targets do not match"));
    }
    for (e, t) in experiments.iter().zip(&targets.rows) {
        check(e, batch, blocks)?;
        if t.entropy.len() != length - e.position {
            return Err(error("a target of other rows than its experiment's"));
        }
    }
    let (paths, bases) = paths(batch, experiments, design, None, blocks)?;
    let plan = Plan::new(paths, length);
    let arithmetic = p.arithmetic();
    let (stream, calls) = run([p, m], &plan, gradient)?;
    // Scores against the targets, each experiment from its position on.
    let rows = outputs(&plan, &bases, experiments);
    let hidden = gather(d, &stream, &rows)?;
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
    let mut bits = Vec::with_capacity(experiments.len());
    let mut at = 0;
    for r in &rows {
        bits.push(nats[at..at + r.len()].iter().map(|v| v / std::f64::consts::LN_2).collect());
        at += r.len();
    }
    if !gradient {
        return Ok(Evaluation { bits, gradient: BTreeMap::new() });
    }
    let seed = seed.ok_or_else(|| error("the head returned no cotangent"))?;
    let mut cotangent = d.zeros(stream.rows(), width).map_err(error)?;
    let mut at = 0;
    for r in &rows {
        let mut total = d.rows_of(&cotangent, r.start, r.len()).map_err(error)?;
        d.axpy(&mut total, 1.0 / std::f64::consts::LN_2, &d.rows_of(&seed, at, r.len()).map_err(error)?).map_err(error)?;
        d.set_rows(&mut cotangent, r.start, &total).map_err(error)?;
        at += r.len();
    }
    let mut total = BTreeMap::new();
    run_reverse([p, m], &plan, calls, cotangent, &mut total)?;
    Ok(Evaluation { bits, gradient: total })
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
}

impl Interchange {
    /// `M` is the split native program `native` (`run_check::split_sites`) with its `layers`
    /// (`run_check::layer_nodes`); `P` is `explanation`, built from it with the same sites, whose
    /// operators `trainable` receive gradients and whose read variables are `variables` (for a
    /// library explanation, [`library_reads`]). Each program holds at most `numeric_bytes` of
    /// operator values on the device; `tile_rows` rows of vocabulary logits are formed at once.
    /// Products run in the device's storage precision.
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
        let mut m = DeviceProgram::compile_values_bounded(device, &prefix(&m_flat)?, numeric_bytes)?;
        m.set_arithmetic(arithmetic);
        // Inherited frozen operators retain their native Arc identity through
        // inlining. Reuse their resident buffers; prepare_dense_parameters below
        // detaches every trainable owner before any posterior update can run.
        let mut p = DeviceProgram::compile_values_sharing_bounded(&m, &prefix(&p_flat)?, numeric_bytes)?;
        p.set_arithmetic(arithmetic);
        p.prepare_dense_parameters(trainable)?;
        let head = FixedHead::new(device, &m_flat, &p_flat, tile_rows)?;
        let m_sites = Arc::new(Sites::new(&m, &m_flat, m_streams, m_reads, &[])?);
        let p_sites = Arc::new(Sites::new(&p, &p_flat, p_streams, p_reads, trainable)?);
        if variables.iter().any(|v| v.block >= 2 * layers.len() || v.parts.is_empty() || v.parts.iter().any(|(op, _)| !trainable.contains(op))) {
            return Err(error("a read variable outside the blocks or the trainable operators"));
        }
        Ok(Self { m, p, m_sites, p_sites, head, variables, trainable: trainable.to_vec() })
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

    /// `P`'s read variables.
    pub fn variables(&self) -> &[ReadVariable] {
        &self.variables
    }

    /// The patch directions of `experiments` over `variables` at `P`'s trainable operators'
    /// values `values` (in the order given to [`Interchange::new`]), read on the host: no load.
    pub fn design_at(&self, variables: &[ReadVariable], experiments: &[Experiment], values: &[ndarray::Array2<f64>]) -> Result<Design, String> {
        if values.len() != self.trainable.len() {
            return Err(error("one value per trainable operator required"));
        }
        design_with(self.p.device(), self.m_sites.entries.len(), variables, experiments, |op, rows| self.host_rows(values, op, rows))
    }

    /// Rows `rows` of trainable operator `op` in `values` (in the order given to
    /// [`Interchange::new`]).
    fn host_rows(&self, values: &[ndarray::Array2<f64>], op: usize, rows: &Range<usize>) -> Result<ndarray::Array2<f64>, String> {
        let width = self.p.widths()[self.p.hidden()];
        let at = self.trainable.iter().position(|t| *t == op).ok_or_else(|| error("a read variable outside the trainable operators"))?;
        let all = &values[at];
        if rows.end > all.nrows() || rows.is_empty() || all.ncols() != width {
            return Err(error("a read variable outside its operator"));
        }
        Ok(all.slice(s![rows.clone(), ..]).to_owned())
    }

    /// `M`'s targets for `experiments` on `batch` under `design` ([`targets`]).
    pub fn targets(&self, batch: &Batch, experiments: &[Experiment], design: &Design) -> Result<Targets, String> {
        let (m, _) = self.models();
        targets(&m, &self.head, batch, experiments, design)
    }

    /// `P`'s program, so that a fit writes each weight sample into its resident parameters.
    pub fn program_mut(&mut self) -> &mut DeviceProgram {
        &mut self.p
    }

    /// `P`'s score on `experiments` against `targets` ([`evaluate`]), the gradient left on the
    /// device per trainable operator.
    pub fn evaluate_resident(&self, batch: &Batch, experiments: &[Experiment], design: &Design, targets: &Targets, gradient: bool) -> Result<Evaluation, String> {
        let (m, p) = self.models();
        evaluate(&m, &p, &self.head, batch, targets, experiments, design, gradient)
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
        self.p.refresh_fused()
    }

    /// `KL(M_e ‖ P_e)` per token for each of `experiments` on `batch` from its position on, at
    /// `P`'s loaded parameters and the directions of `design`, with `M`'s targets made here, and
    /// with `gradient` its sum's gradient downloaded per trainable operator.
    pub fn evaluate(&self, batch: &Batch, experiments: &[Experiment], design: &Design, gradient: bool) -> Result<Scored, String> {
        let targets = self.targets(batch, experiments, design)?;
        let evaluation = self.evaluate_resident(batch, experiments, design, &targets, gradient)?;
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

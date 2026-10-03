//! The human-facing reading of an operator program (#2951): the small rules it reuses with their
//! bindings, its message bits split by where they are stored, the KL bits of the behaviour it does
//! not reproduce, and which operators are still the unchanged native ones.
//!
//! # Rules
//!
//! The program is read at the level of coordinate groups: a vertex is one group of one node's
//! interface, and an edge is a present operator block or a node's own dependence (a pointwise law
//! reads its group, a softmax every score, a mix its payloads' group and every weight, a bilinear
//! score every group of both sides). Every vertex has a shape: its node kind, law and scale, its
//! group's width and label kind, and, per argument node, the set of its argument vertices' shapes.
//! Label indices, node identities and how many groups of one argument node are read are not part
//! of a shape, so plane 5 and plane 7, head 0 and head 3, a sum over three units and one over
//! nine have one shape when they compute one expression. A rule is a shape class. Its instances
//! are the class's vertices, each with its body: the part of its cone that no other instance
//! reaches. What an instance reads outside its body is shared. Repeated classes are taken first,
//! largest shape first, then the classes of one vertex (the glue between rules); a vertex already
//! inside a body is never a root, so every vertex belongs to one rule.
//! An instance's bindings are its root, its labels, its operator blocks and the shared vertices it
//! reads. The program's own rules are read the same way: every call is expanded into its body
//! (a body stored once is applied at each call), so a rule called `k` times is one shape class of
//! `k` instances whose blocks are all shared, and each vertex a call produced names that call.
//!
//! Instances of one rule compute one expression, with their own reals. A change of basis here is
//! a relabelling of groups, an orthogonal change of each group's coordinates, and a positive
//! rescaling of a unit under a positively homogeneous law (ReLU, the identity, zero), which moves
//! the scale from the unit's incoming blocks to its outgoing ones. Each instance's fingerprint is
//! the sorted singular values of its body blocks, after each such unit is rescaled so that its
//! pre-activation's incoming blocks have unit norm; it is invariant under every such change.
//! Equal fingerprints are therefore necessary for two instances to be one rule up to change of
//! basis: instances whose fingerprints differ beyond the fingerprints' rounding bands are proven
//! distinct, the others are grouped as not distinguished. The spread, the largest relative
//! distance of an instance's fingerprint from the medoid's, says how far the instances are from
//! being one rule.
//!
//! # Bits
//!
//! The program's message (`OperatorProgram::code_account`) splits three ways by storage: table
//! storage, the whole message of every operator whose rows or columns are all tokens of a domain
//! (an embedding, an unembedding, a token-indexed constant); other operator reals, the lattice
//! indices of every other operator's reals; and structure, the rest (the header, the bases, the
//! rules, the other operators' interfaces and present-block subsets, and the nodes). The split is
//! bookkeeping by storage layout, not a split into memorized data and algorithm. A rule's bits are
//! the exact signed Elias δ lengths of its body blocks' lattice indices; a block several instances
//! read (a tied operator) is counted once, at the rule. The KL bits are the behaviour's, per input
//! `n KL(model ‖ program)/ln 2`, as the caller supplies them.
//!
//! # Unchanged operators and per-input KL
//!
//! In the program: the bits of operators whose provenance records no rewrite (native, possibly
//! restricted). Having a rewrite in its provenance says only that an operator was rewritten, not
//! that it is understood. In the behaviour: the KL bits per input, the fewest inputs carrying half
//! and nine tenths of them, the worst inputs, and the totals by each token slot's value.

use super::codec::signed_delta_len_bits;
use super::dense::svd;
use super::operator_program::{
    Basis, FamilyInputs, LabelKind, Law, Node, Operator, OperatorBody, OperatorProgram, ProgramError, Scale, SlotValues,
    remap_node,
};
use super::precision::LatticeCode;
use gam_linalg::roundoff::accumulation_growth;
use ndarray::{Array2, s};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

/// The behaviour the program is scored against: per input of the family, its KL bits
/// `n KL/ln 2` (an upper end; `+∞` where no bound was obtained) and whether its argmax agrees with
/// the model's.
pub struct Behaviour<'a> {
    pub inputs: &'a FamilyInputs,
    pub row_bits: &'a [f64],
    pub argmax_agrees: &'a [bool],
}

/// The two-part code, split by storage.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BitSplit {
    /// Header, bases, rules, nodes, and the interfaces and block subsets of operators that are not
    /// token-indexed tables.
    pub structure: u64,
    /// The lattice indices of the reals of operators that are not token-indexed tables.
    pub other_operator_reals: u64,
    /// Token-indexed tables, whole.
    pub table_storage: u64,
    /// The behaviour's KL bits, when supplied.
    pub kl: Option<f64>,
}

impl BitSplit {
    pub fn program(&self) -> u64 {
        self.structure + self.other_operator_reals + self.table_storage
    }
}

/// One use of a rule.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Instance {
    /// The root vertex: node and group label, e.g. `n14 Plane5`.
    pub root: String,
    /// The body's groups by label kind, e.g. `Plane{5} Unit{3,17,40}`.
    pub labels: String,
    /// Per operator, the blocks the body reads: `W_in Unit{3,17} ← Plane{5}`.
    pub operators: Vec<String>,
    /// The shared vertices the body reads, per node.
    pub reads: Vec<String>,
    /// The body's vertices.
    pub size: usize,
    /// The reals and constant bits of the blocks only this instance reads.
    pub reals: usize,
    pub bits: u64,
    pub fingerprint: Vec<f64>,
}

/// A shape class with its instances.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Rule {
    /// The expression every instance computes: named subexpressions first, the root last.
    pub template: Vec<String>,
    pub instances: Vec<Instance>,
    /// Instances grouped by fingerprint within its rounding band: different groups are proven
    /// different up to change of basis, one group is not distinguished.
    pub distinct: Vec<Vec<usize>>,
    /// The largest relative distance of an instance's fingerprint from the medoid's.
    pub spread: f64,
    /// Operators whose blocks more than one instance reads.
    pub shared: Vec<String>,
    /// All distinct blocks the rule's bodies read.
    pub reals: usize,
    pub bits: u64,
}

/// What the behaviour still costs, input by input.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BehaviourMap {
    pub kl_bits: f64,
    pub row_bits: Vec<f64>,
    /// Inputs by decreasing KL bits.
    pub order: Vec<usize>,
    /// The fewest inputs carrying half, and nine tenths, of the KL bits.
    pub half: usize,
    pub most: usize,
    /// Per slot, for a token slot, the KL bits summed by token value.
    pub by_slot: Vec<Option<Vec<f64>>>,
    /// The inputs' token values per slot (`None` for a raw slot).
    pub tokens: Vec<Option<Vec<u32>>>,
    pub disagreements: usize,
}

/// The program as rules, bits, unchanged native operators and per-input KL.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Printout {
    pub bits: BitSplit,
    pub bases: Vec<String>,
    /// The program's own rules: name, calls, body nodes.
    pub calls: Vec<String>,
    /// By decreasing bits.
    pub rules: Vec<Rule>,
    /// Bits of operators with no rewrite in their provenance.
    pub unchanged_native_bits: u64,
    /// Those operators, by decreasing bits.
    pub unchanged_native: Vec<(String, u64)>,
    pub behaviour: Option<BehaviourMap>,
}

// ------------------------------------------------------------------------------------ the graph

/// One present block of one operator; a low-rank operator's reals are one block, `(MAX, MAX)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Block {
    operator: usize,
    row: usize,
    col: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Tag {
    Term,
    Transposed,
    Bias,
    Left,
    Right,
    Score,
    Weight,
    Payload,
    Law,
    Part,
    Basis,
}

struct Edge {
    from: Option<usize>,
    tag: Tag,
    block: Option<Block>,
}

struct Graph<'p> {
    program: &'p OperatorProgram,
    /// Per node, the call that produced it, when it came from a rule body.
    origins: Vec<Option<String>>,
    offsets: Vec<usize>,
    owner: Vec<(usize, usize)>,
    labels: Vec<(LabelKind, u32)>,
    widths: Vec<usize>,
    edges: Vec<Vec<Edge>>,
}

fn present(op: &Operator, row: usize, col: usize) -> bool {
    match &op.body {
        OperatorBody::Dense { present, .. } => present[(row, col)],
        OperatorBody::LowRank { .. } => true,
        OperatorBody::Identity | OperatorBody::Diagonal { .. } => row == col,
    }
}

impl<'p> Graph<'p> {
    fn new(program: &'p OperatorProgram, origins: Vec<Option<String>>) -> Result<Self, ProgramError> {
        let interfaces = program.interfaces()?;
        let mut offsets = Vec::with_capacity(interfaces.len() + 1);
        let mut owner = Vec::new();
        let mut labels = Vec::new();
        let mut widths = Vec::new();
        offsets.push(0);
        for (node, interface) in interfaces.iter().enumerate() {
            for (group, g) in interface.groups().iter().enumerate() {
                owner.push((node, group));
                labels.push((g.label.kind, g.label.index));
                widths.push(g.width);
            }
            offsets.push(owner.len());
        }
        let v = |node: usize, group: usize| offsets[node] + group;
        let all = |node: usize| offsets[node]..offsets[node + 1];
        let ops = &program.operators;
        let mut edges: Vec<Vec<Edge>> = Vec::with_capacity(owner.len());
        for &(index, g) in &owner {
            let mut e = Vec::new();
            let mut push = |from: Option<usize>, tag: Tag, block: Option<Block>| e.push(Edge { from, tag, block });
            match &program.nodes[index] {
                Node::Feature { .. } | Node::Raw { .. } => {}
                Node::Constant { operator } => {
                    if present(&ops[*operator], g, 0) {
                        push(None, Tag::Bias, Some(Block { operator: *operator, row: g, col: 0 }));
                    }
                }
                Node::Affine { terms, bias } => {
                    for &(argument, operator) in terms {
                        for c in 0..ops[operator].cols.group_count() {
                            if present(&ops[operator], g, c) {
                                push(Some(v(argument, c)), Tag::Term, Some(Block { operator, row: g, col: c }));
                            }
                        }
                    }
                    if let Some(operator) = *bias
                        && present(&ops[operator], g, 0)
                    {
                        push(None, Tag::Bias, Some(Block { operator, row: g, col: 0 }));
                    }
                }
                Node::Bilinear { left, right, .. } => {
                    all(*left).for_each(|u| push(Some(u), Tag::Left, None));
                    all(*right).for_each(|u| push(Some(u), Tag::Right, None));
                }
                Node::Softmax { scores } => scores.iter().for_each(|s| push(Some(v(*s, 0)), Tag::Score, None)),
                Node::Mix { weights, payloads } => {
                    payloads.iter().for_each(|(_, p)| push(Some(v(*p, g)), Tag::Payload, None));
                    all(*weights).for_each(|u| push(Some(u), Tag::Weight, None));
                }
                Node::Pointwise { input, .. } | Node::Gain { input, .. } => push(Some(v(*input, g)), Tag::Law, None),
                Node::RmsNorm { input, .. } => all(*input).for_each(|u| push(Some(u), Tag::Law, None)),
                Node::Attend { query, key, value, .. } => {
                    push(Some(v(*value, g)), Tag::Payload, None);
                    all(*query).for_each(|u| push(Some(u), Tag::Left, None));
                    all(*key).for_each(|u| push(Some(u), Tag::Right, None));
                }
                Node::Transposed { input, operator } => {
                    for r in 0..ops[*operator].rows.group_count() {
                        if present(&ops[*operator], r, g) {
                            push(Some(v(*input, r)), Tag::Transposed, Some(Block { operator: *operator, row: r, col: g }));
                        }
                    }
                }
                // Expanded by `inline` before the graph is built.
                Node::Param { .. } | Node::Call { .. } => {}
                Node::Hadamard { left, right } => {
                    push(Some(v(*left, g)), Tag::Left, None);
                    push(Some(v(*right, g)), Tag::Right, None);
                }
                Node::Readout { input, .. } => all(*input).for_each(|u| push(Some(u), Tag::Basis, None)),
                Node::Outer { left, right } => {
                    let width = interfaces[*right].group_count();
                    push(Some(v(*left, g / width)), Tag::Left, None);
                    push(Some(v(*right, g % width)), Tag::Right, None);
                }
                Node::Concat { parts } => {
                    let mut start = 0;
                    for part in parts {
                        let count = interfaces[*part].group_count();
                        if g < start + count {
                            push(Some(v(*part, g - start)), Tag::Part, None);
                            break;
                        }
                        start += count;
                    }
                }
            }
            edges.push(e);
        }
        Ok(Self { program, origins, offsets, owner, labels, widths, edges })
    }

    fn node(&self, vertex: usize) -> &Node {
        &self.program.nodes[self.owner[vertex].0]
    }

    /// The reals key of a block: the block itself, the whole operator for a low-rank one, none for
    /// the identity.
    fn reals_key(&self, block: Block) -> Option<Block> {
        match self.program.operators[block.operator].body {
            OperatorBody::Identity => None,
            OperatorBody::LowRank { .. } => Some(Block { operator: block.operator, row: usize::MAX, col: usize::MAX }),
            OperatorBody::Dense { .. } | OperatorBody::Diagonal { .. } => Some(block),
        }
    }

    /// The reals of a reals key and their signed Elias δ bits.
    fn reals_bits(&self, key: Block) -> Result<(usize, u64), ProgramError> {
        let op = &self.program.operators[key.operator];
        let (reals, precision): (Vec<f64>, _) = match &op.body {
            OperatorBody::Identity => return Ok((0, 0)),
            OperatorBody::LowRank { left, right, precision } => (left.iter().chain(right.iter()).copied().collect(), *precision),
            OperatorBody::Dense { values, precision, .. } => (
                values.slice(s![op.rows.range(key.row), op.cols.range(key.col)]).iter().copied().collect(),
                *precision,
            ),
            OperatorBody::Diagonal { values, precision } => {
                let reals = if key.row == key.col { values.slice(s![op.rows.range(key.row)]).to_vec() } else { Vec::new() };
                (reals, *precision)
            }
        };
        let code = LatticeCode::encode(&reals, precision).map_err(ProgramError::Code)?;
        let mut bits = 0;
        for &index in code.indices() {
            bits += signed_delta_len_bits(index)?;
        }
        Ok((reals.len(), bits))
    }

    fn block_matrix(&self, block: Block, cache: &mut HashMap<usize, Array2<f64>>) -> Array2<f64> {
        let op = &self.program.operators[block.operator];
        if let OperatorBody::Diagonal { values, .. } = &op.body {
            let (rows, cols) = (op.rows.range(block.row), op.cols.range(block.col));
            return if block.row == block.col {
                Array2::from_diag(&values.slice(s![rows]))
            } else {
                Array2::zeros((rows.len(), cols.len()))
            };
        }
        let matrix = cache.entry(block.operator).or_insert_with(|| op.matrix());
        matrix.slice(s![op.rows.range(block.row), op.cols.range(block.col)]).to_owned()
    }
}

// ----------------------------------------------------------------------------------------- shapes

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Head {
    kind: &'static str,
    law: &'static str,
    label: LabelKind,
    width: usize,
    detail: u32,
}

/// Per argument node (and tag), the set of (child shape, operator kind) it contributes.
type Entry = (Tag, Vec<(Option<usize>, u8)>);

fn law_name(law: Law) -> &'static str {
    match law {
        Law::Relu => "relu",
        Law::Identity => "id",
        Law::Zero => "zero",
        Law::Silu => "silu",
        Law::Gelu => "gelu",
        Law::GeluTanh => "gelu_tanh",
    }
}

fn homogeneous(law: Law) -> bool {
    matches!(law, Law::Relu | Law::Identity | Law::Zero)
}

fn kind_word(kind: LabelKind) -> &'static str {
    match kind {
        LabelKind::Native => "vector",
        LabelKind::Const => "1",
        LabelKind::Plane => "plane",
        LabelKind::Unit => "unit",
        LabelKind::Token => "token",
        LabelKind::Position => "position",
        LabelKind::Control => "control",
        LabelKind::Pair => "pair",
        LabelKind::Factor => "factor",
    }
}

fn head(graph: &Graph<'_>, vertex: usize) -> Head {
    let (node, group) = graph.owner[vertex];
    let (label, _) = graph.labels[vertex];
    let (kind, law, detail) = match &graph.program.nodes[node] {
        Node::Feature { basis, .. } => ("feature", "", matches!(graph.program.bases[*basis], Basis::Characters { .. }) as u32),
        Node::Raw { .. } => ("raw", "", 0),
        Node::Constant { .. } => ("constant", "", 0),
        Node::Affine { .. } => ("affine", "", 0),
        Node::Bilinear { scale, .. } => (
            "bilinear",
            "",
            match scale {
                Scale::One => 0,
                Scale::InverseSqrt(n) => *n,
            },
        ),
        Node::Softmax { .. } => ("softmax", "", 0),
        Node::Mix { .. } => ("mix", "", 0),
        Node::Pointwise { laws, .. } => ("pointwise", law_name(laws[group]), 0),
        Node::Hadamard { .. } => ("hadamard", "", 0),
        Node::Readout { .. } => ("readout", "", 0),
        Node::Outer { .. } => ("outer", "", 0),
        Node::Concat { .. } => ("concat", "", 0),
        Node::Gain { .. } => ("gain", "", 0),
        Node::RmsNorm { .. } => ("rmsnorm", "", 0),
        Node::Attend { scale, rotary, causal, .. } => (
            "attend",
            if rotary.is_some() { "rotary" } else { "" },
            2 * (match scale {
                Scale::One => 0,
                Scale::InverseSqrt(n) => *n,
            }) + u32::from(*causal),
        ),
        Node::Transposed { .. } => ("transposed", "", 0),
        Node::Param { .. } => ("param", "", 0),
        Node::Call { .. } => ("call", "", 0),
    };
    Head { kind, law, label, width: graph.widths[vertex], detail }
}

fn body_kind(graph: &Graph<'_>, block: Option<Block>) -> u8 {
    match block.map(|b| &graph.program.operators[b.operator].body) {
        None => 0,
        Some(OperatorBody::Identity) => 1,
        Some(OperatorBody::Dense { .. }) => 2,
        Some(OperatorBody::LowRank { .. }) => 3,
        Some(OperatorBody::Diagonal { .. }) => 4,
    }
}

/// Every vertex's shape id, and each shape's tree size (saturating).
fn shapes(graph: &Graph<'_>) -> (Vec<usize>, Vec<u64>) {
    let mut interned: HashMap<(Head, Vec<Entry>), usize> = HashMap::new();
    let mut sizes: Vec<u64> = Vec::new();
    let mut shape = vec![0usize; graph.owner.len()];
    for vertex in 0..graph.owner.len() {
        let mut grouped: BTreeMap<(Tag, usize), BTreeSet<(Option<usize>, u8)>> = BTreeMap::new();
        for edge in &graph.edges[vertex] {
            let source = edge.from.map_or(usize::MAX, |u| graph.owner[u].0);
            grouped
                .entry((edge.tag, source))
                .or_default()
                .insert((edge.from.map(|u| shape[u]), body_kind(graph, edge.block)));
        }
        let mut entries: Vec<Entry> = grouped.into_iter().map(|((tag, _), set)| (tag, set.into_iter().collect())).collect();
        entries.sort();
        let size = entries
            .iter()
            .flat_map(|(_, set)| set.iter())
            .fold(1u64, |acc, (child, _)| acc.saturating_add(child.map_or(1, |c| sizes[c])));
        let next = interned.len();
        let id = *interned.entry((head(graph, vertex), entries)).or_insert(next);
        if id == sizes.len() {
            sizes.push(size);
        }
        shape[vertex] = id;
    }
    (shape, sizes)
}

// -------------------------------------------------------------------------------------- rendering

/// `Plane{1,5,7} Unit{0..127}`: each label kind's indices, consecutive runs compressed.
fn summarize(labels: impl Iterator<Item = (LabelKind, u32)>) -> String {
    let mut by_kind: BTreeMap<LabelKind, Vec<u32>> = BTreeMap::new();
    for (kind, index) in labels {
        by_kind.entry(kind).or_default().push(index);
    }
    let mut parts = Vec::new();
    for (kind, mut indices) in by_kind {
        indices.sort_unstable();
        indices.dedup();
        let mut runs: Vec<String> = Vec::new();
        let mut start = 0;
        while start < indices.len() {
            let mut end = start;
            while end + 1 < indices.len() && indices[end + 1] == indices[end] + 1 {
                end += 1;
            }
            runs.push(match end - start {
                0 => indices[start].to_string(),
                1 => format!("{},{}", indices[start], indices[end]),
                _ => format!("{}..{}", indices[start], indices[end]),
            });
            start = end + 1;
        }
        parts.push(match kind {
            LabelKind::Native | LabelKind::Const => format!("{kind:?}"),
            _ => format!("{kind:?}{{{}}}", runs.join(",")),
        });
    }
    parts.join(" ")
}

/// A vertex outside a body, as a template argument.
fn argument(graph: &Graph<'_>, vertex: usize) -> String {
    let (kind, _) = graph.labels[vertex];
    match graph.node(vertex) {
        Node::Feature { slot, .. } => format!("{}(x{slot})", kind_word(kind)),
        Node::Raw { slot } => format!("x{slot}"),
        _ => format!("n{}:{}", graph.owner[vertex].0, kind_word(kind)),
    }
}

struct Renderer<'g, 'p> {
    graph: &'g Graph<'p>,
    body: &'g BTreeSet<usize>,
    named: HashMap<usize, String>,
    definitions: Vec<String>,
}

impl Renderer<'_, '_> {
    /// Per (tag, argument node): the distinct child shapes, one representative each, with how
    /// many vertices of that node the vertex reads.
    fn groups(&self, vertex: usize, shape: &[usize]) -> Vec<(Tag, Option<Block>, Vec<(Option<usize>, usize)>)> {
        let mut out: Vec<(Tag, usize, Option<Block>, Vec<(Option<usize>, usize)>)> = Vec::new();
        for edge in &self.graph.edges[vertex] {
            let source = edge.from.map_or(usize::MAX, |u| self.graph.owner[u].0);
            let slot = match out.iter().position(|(tag, s, _, _)| *tag == edge.tag && *s == source) {
                Some(i) => i,
                None => {
                    out.push((edge.tag, source, edge.block, Vec::new()));
                    out.len() - 1
                }
            };
            let children = &mut out[slot].3;
            match children.iter_mut().find(|(c, _)| c.map(|u| shape[u]) == edge.from.map(|u| shape[u])) {
                Some((_, count)) => *count += 1,
                None => children.push((edge.from, 1)),
            }
        }
        out.into_iter().map(|(tag, _, block, children)| (tag, block, children)).collect()
    }

    fn child(&mut self, vertex: Option<usize>, count: usize, shape: &[usize], parents: &HashMap<usize, usize>) -> String {
        let Some(u) = vertex else { return "b".to_string() };
        let text = if !self.body.contains(&u) {
            argument(self.graph, u)
        } else if let Some(name) = self.named.get(&u) {
            name.clone()
        } else if parents.get(&u).copied().unwrap_or(0) > 1 && !self.graph.edges[u].is_empty() {
            let expression = self.render(u, shape, parents);
            let name = format!("s{}", self.named.len() + 1);
            self.definitions.push(format!("{name} = {expression}"));
            self.named.insert(u, name.clone());
            name
        } else {
            self.render(u, shape, parents)
        };
        if count > 1 { format!("Σ_{} {text}", kind_word(self.graph.labels[u].0)) } else { text }
    }

    fn render(&mut self, vertex: usize, shape: &[usize], parents: &HashMap<usize, usize>) -> String {
        let groups = self.groups(vertex, shape);
        let mut parts: Vec<(Tag, String)> = Vec::new();
        for (tag, block, children) in groups {
            let identity = block.is_some_and(|b| matches!(self.graph.program.operators[b.operator].body, OperatorBody::Identity));
            let texts: Vec<String> = children.into_iter().map(|(c, n)| self.child(c, n, shape, parents)).collect();
            for text in texts {
                let text = match tag {
                    Tag::Term if identity => text,
                    Tag::Term if text.starts_with('Σ') => text.replacen(' ', " W·", 1),
                    Tag::Term => format!("W·{text}"),
                    Tag::Transposed => format!("Wᵀ·{text}"),
                    _ => text,
                };
                parts.push((tag, text));
            }
        }
        let join = |tag: Tag, sep: &str| parts.iter().filter(|(t, _)| *t == tag).map(|(_, s)| s.as_str()).collect::<Vec<_>>().join(sep);
        let (kind, index) = self.graph.labels[vertex];
        match self.graph.node(vertex) {
            Node::Feature { slot, .. } => format!("{}(x{slot})", kind_word(kind)),
            Node::Raw { slot } => format!("x{slot}"),
            Node::Constant { .. } => "b".to_string(),
            Node::Affine { .. } => {
                let mut terms: Vec<&str> = parts.iter().filter(|(t, _)| *t == Tag::Term).map(|(_, s)| s.as_str()).collect();
                if parts.iter().any(|(t, _)| *t == Tag::Bias) {
                    terms.push("b");
                }
                if terms.is_empty() { "0".to_string() } else { terms.join(" + ") }
            }
            Node::Pointwise { laws, .. } => {
                let inner = join(Tag::Law, "");
                match laws[self.graph.owner[vertex].1] {
                    Law::Identity => inner,
                    Law::Zero => "0".to_string(),
                    law => format!("{}({inner})", law_name(law)),
                }
            }
            Node::Bilinear { .. } => format!("⟨{}, {}⟩", join(Tag::Left, " + "), join(Tag::Right, " + ")),
            Node::Softmax { .. } => format!("softmax_{index}({})", join(Tag::Score, ", ")),
            Node::Mix { .. } => format!("mix({}; {})", join(Tag::Weight, ", "), join(Tag::Payload, ", ")),
            Node::Hadamard { .. } => format!("{} ⊙ {}", join(Tag::Left, ""), join(Tag::Right, "")),
            Node::Outer { .. } => format!("{} ⊗ {}", join(Tag::Left, ""), join(Tag::Right, "")),
            Node::Concat { .. } => join(Tag::Part, ""),
            Node::Readout { .. } => format!("logits({})", join(Tag::Basis, ", ")),
            Node::Gain { .. } => format!("c·{}", join(Tag::Law, "")),
            Node::RmsNorm { .. } => format!("rmsnorm({})", join(Tag::Law, " + ")),
            Node::Attend { .. } => format!(
                "attend({}; {}; {})",
                join(Tag::Left, " + "),
                join(Tag::Right, " + "),
                join(Tag::Payload, "")
            ),
            Node::Transposed { .. } => join(Tag::Transposed, " + "),
            Node::Param { index } => format!("arg{index}"),
            Node::Call { rule, .. } => format!("call{rule}"),
        }
    }
}

// ------------------------------------------------------------------------------------ fingerprint

/// The instance's gauge-invariant fingerprint (see the module note) with its per-entry band,
/// both sorted by decreasing value.
fn fingerprint(
    graph: &Graph<'_>,
    body: &BTreeSet<usize>,
    cache: &mut HashMap<usize, Array2<f64>>,
) -> Result<(Vec<f64>, Vec<f64>), ProgramError> {
    let mut scale_in: HashMap<usize, f64> = HashMap::new();
    let mut scale_out: HashMap<usize, f64> = HashMap::new();
    for &vertex in body {
        let (_, group) = graph.owner[vertex];
        let Node::Pointwise { input, laws } = graph.node(vertex) else { continue };
        let pre = graph.offsets[*input] + group;
        if !homogeneous(laws[group]) || !body.contains(&pre) {
            continue;
        }
        let mut squared = 0.0;
        for edge in &graph.edges[pre] {
            if let Some(block) = edge.block {
                squared += graph.block_matrix(block, cache).iter().map(|v| v * v).sum::<f64>();
            }
        }
        if squared > 0.0 {
            let norm = squared.sqrt();
            scale_in.insert(pre, 1.0 / norm);
            scale_out.insert(vertex, norm);
        }
    }
    let mut entries: Vec<(f64, f64)> = Vec::new();
    for &vertex in body {
        for edge in &graph.edges[vertex] {
            let Some(block) = edge.block else { continue };
            let factor = scale_in.get(&vertex).copied().unwrap_or(1.0)
                * edge.from.and_then(|u| scale_out.get(&u).copied()).unwrap_or(1.0);
            let matrix = graph.block_matrix(block, cache) * factor;
            let decomposed =
                svd(matrix.view(), false).map_err(|error| ProgramError::Shape(format!("fingerprint: {error:?}")))?;
            let (values, band) = (decomposed.singular_values, decomposed.band);
            let rounding = accumulation_growth(matrix.len() + 4);
            entries.extend(values.iter().map(|&value| (value, band + rounding * value)));
        }
    }
    entries.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(entries.into_iter().unzip())
}

fn relative_distance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().max(b.len());
    let at = |v: &[f64], i: usize| v.get(i).copied().unwrap_or(0.0);
    let difference = (0..n).map(|i| (at(a, i) - at(b, i)).powi(2)).sum::<f64>().sqrt();
    let scale = a.iter().map(|v| v * v).sum::<f64>().sqrt().max(b.iter().map(|v| v * v).sum::<f64>().sqrt());
    if scale == 0.0 { 0.0 } else { difference / scale }
}

fn within_bands(a: &(Vec<f64>, Vec<f64>), b: &(Vec<f64>, Vec<f64>)) -> bool {
    a.0.len() == b.0.len() && (0..a.0.len()).all(|i| (a.0[i] - b.0[i]).abs() <= a.1[i] + b.1[i])
}

/// Instances grouped by fingerprint equality within bands (transitively), and the spread.
fn distinct_and_spread(prints: &[(Vec<f64>, Vec<f64>)]) -> (Vec<Vec<usize>>, f64) {
    let n = prints.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], i: usize) -> usize {
        let mut root = i;
        while parent[root] != root {
            root = parent[root];
        }
        parent[i] = root;
        root
    }
    let mut spread = f64::INFINITY;
    for i in 0..n {
        let mut farthest: f64 = 0.0;
        for j in 0..n {
            if i != j {
                farthest = farthest.max(relative_distance(&prints[i].0, &prints[j].0));
                if j > i && within_bands(&prints[i], &prints[j]) {
                    let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                    parent[a] = b;
                }
            }
        }
        spread = spread.min(farthest);
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }
    let mut distinct: Vec<Vec<usize>> = groups.into_values().collect();
    distinct.sort();
    (distinct, if n == 0 { 0.0 } else { spread })
}

// ------------------------------------------------------------------------------------------ print

/// For a character basis, `a(t) = k t + c mod p` on the cycle's tokens in token order, if so.
fn affine_labelling(positions: &[Option<u32>]) -> Option<(u64, u64)> {
    let placed: Vec<(u64, u64)> =
        positions.iter().enumerate().filter_map(|(t, a)| a.map(|a| (t as u64, u64::from(a)))).collect();
    let p = placed.len() as u64;
    let &(t0, a0) = placed.first()?;
    (1..p).find_map(|k| {
        let c = (a0 + p * p - (k * t0) % p) % p;
        placed.iter().all(|&(t, a)| (k * t + c) % p == a).then_some((k, c))
    })
}

fn describe_basis(program: &OperatorProgram, basis: &Basis) -> String {
    match basis {
        Basis::Indicator { domain } => {
            format!("domain {domain} ({} tokens): one-hot", program.declarations.domains[*domain].size)
        }
        Basis::Characters { domain, positions, declared } => {
            let period = basis.period().unwrap_or(0);
            let origin = if *declared { "declared" } else { "recovered" };
            let labelling = match affine_labelling(positions) {
                Some((1, 0)) => "a(t) = t".to_string(),
                Some((k, 0)) => format!("a(t) = {k}t"),
                Some((k, c)) => format!("a(t) = {k}t + {c}"),
                None => "a(t) not affine in t".to_string(),
            };
            format!(
                "domain {domain} ({} tokens): characters of a {origin} {period}-cycle, {labelling}",
                program.declarations.domains[*domain].size
            )
        }
    }
}

fn behaviour_map(behaviour: &Behaviour<'_>) -> Result<BehaviourMap, ProgramError> {
    let rows = behaviour.inputs.rows;
    if behaviour.row_bits.len() != rows || behaviour.argmax_agrees.len() != rows {
        return Err(ProgramError::Shape(format!(
            "print: {} row bits and {} argmax flags for {rows} inputs",
            behaviour.row_bits.len(),
            behaviour.argmax_agrees.len()
        )));
    }
    let kl_bits: f64 = behaviour.row_bits.iter().sum();
    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by(|&a, &b| behaviour.row_bits[b].total_cmp(&behaviour.row_bits[a]).then(a.cmp(&b)));
    let carrying = |share: f64| {
        let mut sum = 0.0;
        order
            .iter()
            .position(|&row| {
                sum += behaviour.row_bits[row];
                sum >= share * kl_bits
            })
            .map_or(rows, |i| i + 1)
    };
    let tokens: Vec<Option<Vec<u32>>> = behaviour
        .inputs
        .slots
        .iter()
        .map(|slot| match slot {
            SlotValues::Tokens(values) => Some(values.clone()),
            SlotValues::Raw(_) => None,
        })
        .collect();
    let by_slot = tokens
        .iter()
        .map(|values| {
            values.as_ref().map(|values| {
                let mut sums = vec![0.0; values.iter().copied().max().map_or(0, |m| m as usize + 1)];
                for (row, &token) in values.iter().enumerate() {
                    sums[token as usize] += behaviour.row_bits[row];
                }
                sums
            })
        })
        .collect();
    Ok(BehaviourMap {
        kl_bits,
        row_bits: behaviour.row_bits.to_vec(),
        half: carrying(0.5),
        most: carrying(0.9),
        order,
        by_slot,
        tokens,
        disagreements: behaviour.argmax_agrees.iter().filter(|agree| !**agree).count(),
    })
}

/// Append `nodes` (a program's or a rule body's, with `arguments` for its parameters) to `out` with
/// every call expanded into its rule's body; returns each node's index in `out`.
fn inline_nodes(
    program: &OperatorProgram,
    nodes: &[Node],
    arguments: &[usize],
    origin: Option<&str>,
    out: &mut Vec<Node>,
    origins: &mut Vec<Option<String>>,
    calls: &mut BTreeMap<usize, usize>,
) -> Result<Vec<usize>, ProgramError> {
    let operators: Vec<usize> = (0..program.operators.len()).collect();
    let bases: Vec<usize> = (0..program.bases.len()).collect();
    let rules: Vec<usize> = (0..program.rules.len()).collect();
    let mut map = Vec::with_capacity(nodes.len());
    for node in nodes {
        match node {
            Node::Param { index } => {
                map.push(*arguments.get(*index).ok_or(ProgramError::Reference { what: "call argument", index: *index })?)
            }
            Node::Call { rule, arguments: args } => {
                let body = program.rules.get(*rule).ok_or(ProgramError::Reference { what: "rule", index: *rule })?;
                let flat: Vec<usize> = args.iter().map(|a| map[*a]).collect();
                let count = calls.entry(*rule).or_default();
                let label = match origin {
                    Some(outer) => format!("{outer}/{}#{count}", body.name),
                    None => format!("{}#{count}", body.name),
                };
                *count += 1;
                let inner = inline_nodes(program, &body.nodes, &flat, Some(&label), out, origins, calls)?;
                map.push(inner[body.output]);
            }
            other => {
                let mut node = other.clone();
                remap_node(&mut node, &map, &operators, &bases, &rules);
                out.push(node);
                origins.push(origin.map(str::to_string));
                map.push(out.len() - 1);
            }
        }
    }
    Ok(map)
}

/// `program` with every call expanded, and per node the call that produced it.
fn inline(program: &OperatorProgram) -> Result<(OperatorProgram, Vec<Option<String>>), ProgramError> {
    let (mut nodes, mut origins) = (Vec::new(), Vec::new());
    let map = inline_nodes(program, &program.nodes, &[], None, &mut nodes, &mut origins, &mut BTreeMap::new())?;
    let flat = OperatorProgram {
        declarations: program.declarations.clone(),
        bases: program.bases.clone(),
        operators: program.operators.clone(),
        rules: Vec::new(),
        nodes,
        output: map[program.output],
    };
    Ok((flat, origins))
}

/// The program as rules with bindings, its bit split, and its unchanged native operators.
pub fn print(program: &OperatorProgram, behaviour: Option<&Behaviour<'_>>) -> Result<Printout, ProgramError> {
    let account = program.code_account()?;
    let tokens = |side: &super::operator_program::Interface| side.groups().iter().all(|g| g.label.kind == LabelKind::Token);
    let (mut table_storage, mut other_operator_reals) = (0u64, 0u64);
    for (op, (structure, reals)) in program.operators.iter().zip(&account.operator_bits) {
        if tokens(&op.rows) || tokens(&op.cols) {
            table_storage += structure + reals;
        } else {
            other_operator_reals += reals;
        }
    }
    let behaviour = behaviour.map(behaviour_map).transpose()?;
    let bits = BitSplit {
        structure: account.total_bits - other_operator_reals - table_storage,
        other_operator_reals,
        table_storage,
        kl: behaviour.as_ref().map(|b| b.kl_bits),
    };
    let (flat, origins) = inline(program)?;
    let graph = Graph::new(&flat, origins)?;
    let (shape, sizes) = shapes(&graph);

    // Shape classes of the vertices that compute something, largest shape first.
    let mut classes: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for vertex in 0..graph.owner.len() {
        let basis = matches!(graph.node(vertex), Node::Feature { .. } | Node::Raw { .. } | Node::Readout { .. });
        if !basis && !graph.edges[vertex].is_empty() {
            classes.entry(shape[vertex]).or_default().push(vertex);
        }
    }
    let mut order: Vec<(usize, Vec<usize>)> = classes.into_iter().collect();
    order.sort_by(|a, b| {
        (b.1.len() > 1)
            .cmp(&(a.1.len() > 1))
            .then(sizes[b.0].cmp(&sizes[a.0]))
            .then(b.1.len().cmp(&a.1.len()))
            .then(a.0.cmp(&b.0))
    });

    let mut covered = vec![false; graph.owner.len()];
    let mut reach = vec![0u32; graph.owner.len()];
    let mut seen = vec![usize::MAX; graph.owner.len()];
    let mut cache: HashMap<usize, Array2<f64>> = HashMap::new();
    let mut rules = Vec::new();
    for (_, members) in order {
        let roots: Vec<usize> = members.into_iter().filter(|v| !covered[*v]).collect();
        if roots.is_empty() {
            continue;
        }
        // How many roots reach each uncovered vertex.
        let mut touched = Vec::new();
        for (r, &root) in roots.iter().enumerate() {
            let mut stack = vec![root];
            while let Some(u) = stack.pop() {
                if seen[u] == r || covered[u] {
                    continue;
                }
                seen[u] = r;
                if reach[u] == 0 {
                    touched.push(u);
                }
                reach[u] += 1;
                stack.extend(graph.edges[u].iter().filter_map(|e| e.from));
            }
        }
        let bodies: Vec<BTreeSet<usize>> = roots
            .iter()
            .map(|&root| {
                let mut body = BTreeSet::new();
                let mut stack = vec![root];
                while let Some(u) = stack.pop() {
                    if covered[u] || reach[u] != 1 || !body.insert(u) {
                        continue;
                    }
                    stack.extend(graph.edges[u].iter().filter_map(|e| e.from));
                }
                body
            })
            .collect();
        for u in touched {
            reach[u] = 0;
        }
        for &u in bodies.iter().flatten() {
            covered[u] = true;
        }
        seen.iter_mut().for_each(|s| *s = usize::MAX);
        rules.push(rule(&graph, &shape, &roots, &bodies, &mut cache)?);
    }
    rules.sort_by(|a, b| b.bits.cmp(&a.bits).then(b.instances.len().cmp(&a.instances.len())));

    let mut unchanged_native: Vec<(String, u64)> = program
        .operators
        .iter()
        .zip(&account.operator_bits)
        .filter(|(op, _)| op.provenance.derivation.is_empty() && !op.provenance.sources.is_empty())
        .map(|(op, (structure, reals))| (op.name.clone(), structure + reals))
        .collect();
    unchanged_native.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok(Printout {
        bits,
        bases: program.bases.iter().map(|basis| describe_basis(program, basis)).collect(),
        calls: program
            .rules
            .iter()
            .enumerate()
            .map(|(index, rule)| {
                let calls = program
                    .nodes
                    .iter()
                    .chain(program.rules.iter().flat_map(|r| r.nodes.iter()))
                    .filter(|node| matches!(node, Node::Call { rule, .. } if *rule == index))
                    .count();
                format!("{} ×{calls}, body of {} nodes", rule.name, rule.nodes.len())
            })
            .collect(),
        rules,
        unchanged_native_bits: unchanged_native.iter().map(|(_, b)| b).sum(),
        unchanged_native,
        behaviour,
    })
}

fn rule(
    graph: &Graph<'_>,
    shape: &[usize],
    roots: &[usize],
    bodies: &[BTreeSet<usize>],
    cache: &mut HashMap<usize, Array2<f64>>,
) -> Result<Rule, ProgramError> {
    // Which instances read each reals key.
    let mut readers: BTreeMap<Block, BTreeSet<usize>> = BTreeMap::new();
    for (i, body) in bodies.iter().enumerate() {
        for &v in body {
            for block in graph.edges[v].iter().filter_map(|e| e.block) {
                if let Some(key) = graph.reals_key(block) {
                    readers.entry(key).or_default().insert(i);
                }
            }
        }
    }
    let mut key_bits: BTreeMap<Block, (usize, u64)> = BTreeMap::new();
    for &key in readers.keys() {
        key_bits.insert(key, graph.reals_bits(key)?);
    }
    let mut shared: BTreeSet<String> = BTreeSet::new();
    for (key, who) in &readers {
        if who.len() > 1 {
            shared.insert(graph.program.operators[key.operator].name.clone());
        }
    }
    let mut instances = Vec::new();
    let mut prints = Vec::new();
    for (i, (&root, body)) in roots.iter().zip(bodies).enumerate() {
        let (node, _) = graph.owner[root];
        let (kind, index) = graph.labels[root];
        let mut blocks: BTreeMap<usize, (BTreeSet<(LabelKind, u32)>, BTreeSet<(LabelKind, u32)>)> = BTreeMap::new();
        let mut reads: BTreeMap<usize, BTreeSet<(LabelKind, u32)>> = BTreeMap::new();
        for &v in body {
            for edge in &graph.edges[v] {
                if let Some(block) = edge.block {
                    let op = &graph.program.operators[block.operator];
                    let entry = blocks.entry(block.operator).or_default();
                    let (r, c) = (op.rows.groups()[block.row].label, op.cols.groups()[block.col].label);
                    entry.0.insert((r.kind, r.index));
                    entry.1.insert((c.kind, c.index));
                }
                if let Some(u) = edge.from
                    && !body.contains(&u)
                {
                    reads.entry(graph.owner[u].0).or_default().insert(graph.labels[u]);
                }
            }
        }
        let operators = blocks
            .into_iter()
            .map(|(op, (rows, cols))| {
                let name = &graph.program.operators[op].name;
                match graph.program.operators[op].body {
                    OperatorBody::LowRank { ref left, .. } => format!("{name} rank {}", left.ncols()),
                    _ => format!("{name} {} ← {}", summarize(rows.into_iter()), summarize(cols.into_iter())),
                }
            })
            .collect();
        let reads = reads
            .into_iter()
            .map(|(node, labels)| {
                let site = match &graph.program.nodes[node] {
                    Node::Feature { slot, .. } | Node::Raw { slot } => format!("x{slot}"),
                    _ => format!("n{node}"),
                };
                format!("{site} {}", summarize(labels.into_iter()))
            })
            .collect();
        let (reals, bits) = readers
            .iter()
            .filter(|(_, who)| who.len() == 1 && who.contains(&i))
            .map(|(key, _)| key_bits[key])
            .fold((0, 0), |acc, (r, b)| (acc.0 + r, acc.1 + b));
        let print = fingerprint(graph, body, cache)?;
        instances.push(Instance {
            root: match &graph.origins[node] {
                Some(call) => format!("n{node} {kind:?}{index} in {call}"),
                None => format!("n{node} {kind:?}{index}"),
            },
            labels: summarize(body.iter().map(|&v| graph.labels[v])),
            operators,
            reads,
            size: body.len(),
            reals,
            bits,
            fingerprint: print.0.clone(),
        });
        prints.push(print);
    }
    let (distinct, spread) = distinct_and_spread(&prints);
    let template = {
        let body = &bodies[0];
        let mut parents: HashMap<usize, usize> = HashMap::new();
        for &v in body {
            let sources: BTreeSet<usize> = graph.edges[v].iter().filter_map(|e| e.from).filter(|u| body.contains(u)).collect();
            for u in sources {
                *parents.entry(u).or_default() += 1;
            }
        }
        let mut renderer = Renderer { graph, body, named: HashMap::new(), definitions: Vec::new() };
        let root = renderer.render(roots[0], shape, &parents);
        let mut lines = renderer.definitions;
        lines.push(format!("{} ← {root}", kind_word(graph.labels[roots[0]].0)));
        lines
    };
    let (reals, bits) = key_bits.values().fold((0, 0), |acc, (r, b)| (acc.0 + r, acc.1 + b));
    Ok(Rule { template, instances, distinct, spread, shared: shared.into_iter().collect(), reals, bits })
}

// ---------------------------------------------------------------------------------------- display

/// How many instances, worst inputs and token values a printout lists before summarizing the rest.
const LISTED: usize = 6;

impl fmt::Display for Printout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = &self.bits;
        write!(
            f,
            "program {} bits = structure {} + other operator reals {} + table storage {}",
            b.program(),
            b.structure,
            b.other_operator_reals,
            b.table_storage
        )?;
        match b.kl {
            Some(kl) => writeln!(f, "; KL {kl:.1} bits; total {:.1} bits", b.program() as f64 + kl)?,
            None => writeln!(f)?,
        }
        if !self.bases.is_empty() {
            writeln!(f, "bases")?;
            for basis in &self.bases {
                writeln!(f, "  {basis}")?;
            }
        }
        if !self.calls.is_empty() {
            writeln!(f, "calls")?;
            for call in &self.calls {
                writeln!(f, "  {call}")?;
            }
        }
        writeln!(f, "rules")?;
        for (r, rule) in self.rules.iter().enumerate() {
            let n = rule.instances.len();
            let lines = &rule.template;
            writeln!(f, "  R{} ×{n}  {}   [{} bits, {} reals]", r + 1, lines[lines.len() - 1], rule.bits, rule.reals)?;
            for line in &lines[..lines.len() - 1] {
                writeln!(f, "        where {line}")?;
            }
            if n > 1 {
                write!(f, "        {} distinct up to change of basis, spread {:.3}", rule.distinct.len(), rule.spread)?;
                if !rule.shared.is_empty() {
                    write!(f, "; shared {}", rule.shared.join(", "))?;
                }
                writeln!(f)?;
            }
            for instance in rule.instances.iter().take(LISTED) {
                writeln!(
                    f,
                    "        {}: {}; {}{}  [{} bits]",
                    instance.root,
                    instance.labels,
                    instance.operators.join("; "),
                    if instance.reads.is_empty() { String::new() } else { format!("; reads {}", instance.reads.join(", ")) },
                    instance.bits
                )?;
            }
            if n > LISTED {
                writeln!(f, "        … {} more", n - LISTED)?;
            }
        }
        write!(f, "unchanged native operators: {} bits", self.unchanged_native_bits)?;
        let listed: Vec<String> =
            self.unchanged_native.iter().take(LISTED).map(|(name, bits)| format!("{name} {bits}")).collect();
        writeln!(f, "{}{}", if listed.is_empty() { "" } else { ": " }, listed.join(", "))?;
        if let Some(m) = &self.behaviour {
            writeln!(
                f,
                "behaviour: {:.1} KL bits over {} inputs; half in {}, nine tenths in {}; {} argmax disagreements",
                m.kl_bits,
                m.row_bits.len(),
                m.half,
                m.most,
                m.disagreements
            )?;
            let input = |row: usize| {
                let values: Vec<String> =
                    m.tokens.iter().map(|t| t.as_ref().map_or("·".to_string(), |t| t[row].to_string())).collect();
                format!("({})", values.join(", "))
            };
            let worst: Vec<String> =
                m.order.iter().take(LISTED).map(|&row| format!("{} {:.2}", input(row), m.row_bits[row])).collect();
            writeln!(f, "    worst: {}", worst.join("; "))?;
            for (slot, sums) in m.by_slot.iter().enumerate() {
                let Some(sums) = sums else { continue };
                let mut ranked: Vec<(usize, f64)> = sums.iter().copied().enumerate().filter(|(_, s)| *s > 0.0).collect();
                if ranked.len() < 2 {
                    continue;
                }
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                let top: Vec<String> = ranked.iter().take(LISTED).map(|(t, s)| format!("{t}: {s:.1}")).collect();
                writeln!(f, "    by x{slot}: {}", top.join("  "))?;
            }
        }
        Ok(())
    }
}

//! The component-level view of an operator program (#2951): what each operator reads, the law that
//! applies it, what it writes, how often it is used, and what it costs, with the share of the
//! message still spent on native, unrewritten operators.
//!
//! A node-level graph is the model in another notation; this view is the compact object a reader
//! checks. Each [`ComponentView`] names an operator's present input groups and output groups (by
//! label, runs of indices compressed), the node kinds that apply it, the number of places it is
//! used, its message length and its native sources. An operator whose provenance records no rewrite
//! is native and unresolved: its bits are the program's unresolved share.

use super::operator_program::{Interface, LabelKind, Node, Operator, OperatorBody, OperatorProgram, ProgramError};
use std::collections::BTreeMap;

/// One operator, as a component.
#[derive(Clone, Debug, PartialEq)]
pub struct ComponentView {
    pub name: String,
    /// The present input groups, e.g. `Plane{1,5,7} Const`.
    pub reads: String,
    /// The present output groups.
    pub writes: String,
    /// The node kinds that apply it, with each downstream pointwise law.
    pub laws: Vec<String>,
    /// How many node terms reference it.
    pub uses: usize,
    pub reals: usize,
    pub bits: u64,
    pub sources: Vec<String>,
    /// No rewrite produced it: it is the native operator, possibly restricted.
    pub unresolved: bool,
    /// A lookup table: its columns or its rows are indexed by the tokens of a domain (an embedding,
    /// an unembedding, a token-indexed constant). Its bits are data; every other operator's are
    /// algorithm.
    pub data: bool,
}

/// The whole program's view.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgramView {
    pub bits: u64,
    pub components: Vec<ComponentView>,
    pub unresolved_bits: u64,
    /// The bits of lookup-table components; the rest of the operators' bits are algorithm.
    pub data_bits: u64,
    pub algorithm_bits: u64,
}

impl ProgramView {
    pub fn unresolved_fraction(&self) -> f64 {
        self.unresolved_bits as f64 / self.bits as f64
    }
}

/// `Plane{1,5,7} Unit{0..127}`: each label kind's present indices, consecutive runs compressed.
fn summarize(labels: impl Iterator<Item = (LabelKind, u32)>) -> String {
    let mut by_kind: BTreeMap<LabelKind, Vec<u32>> = BTreeMap::new();
    for (kind, index) in labels {
        by_kind.entry(kind).or_default().push(index);
    }
    let mut out = String::new();
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
            runs.push(if end > start + 1 {
                format!("{}..{}", indices[start], indices[end])
            } else if end == start + 1 {
                format!("{},{}", indices[start], indices[end])
            } else {
                indices[start].to_string()
            });
            start = end + 1;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        match kind {
            LabelKind::Native | LabelKind::Const => out.push_str(&format!("{kind:?}")),
            _ => out.push_str(&format!("{kind:?}{{{}}}", runs.join(","))),
        }
    }
    out
}

fn present_labels(op: &Operator, rows: bool) -> String {
    let side = if rows { &op.rows } else { &op.cols };
    let groups: Vec<usize> = match &op.body {
        OperatorBody::Identity | OperatorBody::LowRank { .. } => (0..side.group_count()).collect(),
        OperatorBody::Dense { present, .. } => (0..side.group_count())
            .filter(|&g| if rows { present.row(g).iter().any(|k| *k) } else { present.column(g).iter().any(|k| *k) })
            .collect(),
    };
    summarize(groups.into_iter().map(|g| (side.groups()[g].label.kind, side.groups()[g].label.index)))
}

fn node_kind(node: &Node) -> &'static str {
    match node {
        Node::Feature { .. } => "feature",
        Node::Raw { .. } => "raw",
        Node::Constant { .. } => "constant",
        Node::Affine { .. } => "affine",
        Node::Bilinear { .. } => "bilinear",
        Node::Softmax { .. } => "softmax",
        Node::Mix { .. } => "mix",
        Node::Pointwise { .. } => "pointwise",
        Node::Hadamard { .. } => "hadamard",
        Node::Readout { .. } => "readout",
        Node::Outer { .. } => "outer",
        Node::Concat { .. } => "concat",
        Node::Param { .. } => "param",
        Node::Call { .. } => "call",
        Node::Gain { .. } => "gain",
        Node::Attend { .. } => "attend",
        Node::RmsNorm { .. } => "rms_norm",
        Node::Transposed { .. } => "transposed",
    }
}

/// The view of `program`.
pub fn view(program: &OperatorProgram) -> Result<ProgramView, ProgramError> {
    let account = program.code_account()?;
    let mut uses = vec![0usize; program.operators.len()];
    let mut laws: Vec<Vec<String>> = vec![Vec::new(); program.operators.len()];
    for (index, node) in program.nodes.iter().enumerate() {
        for op in node.operators() {
            uses[op] += 1;
            let downstream: Vec<String> = program
                .nodes
                .iter()
                .filter(|reader| reader.arguments().contains(&index))
                .map(|reader| match reader {
                    Node::Pointwise { laws, .. } => {
                        let mut kinds: Vec<String> = laws.iter().map(|law| format!("{law:?}")).collect();
                        kinds.sort();
                        kinds.dedup();
                        format!("pointwise[{}]", kinds.join("|"))
                    }
                    other => node_kind(other).to_string(),
                })
                .collect();
            let entry = format!("{} -> {}", node_kind(node), if downstream.is_empty() { "output".to_string() } else { downstream.join(",") });
            if !laws[op].contains(&entry) {
                laws[op].push(entry);
            }
        }
    }
    let mut components = Vec::new();
    let mut unresolved_bits = 0;
    let (mut data_bits, mut algorithm_bits) = (0u64, 0u64);
    for (index, op) in program.operators.iter().enumerate() {
        let bits = account.operator_bits[index].0 + account.operator_bits[index].1;
        let unresolved = op.provenance.derivation.is_empty() && !op.provenance.sources.is_empty();
        if unresolved {
            unresolved_bits += bits;
        }
        let tokens = |side: &Interface| side.groups().iter().all(|g| g.label.kind == LabelKind::Token);
        let data = tokens(&op.rows) || tokens(&op.cols);
        if data {
            data_bits += bits;
        } else {
            algorithm_bits += bits;
        }
        components.push(ComponentView {
            name: op.name.clone(),
            reads: present_labels(op, false),
            writes: present_labels(op, true),
            laws: laws[index].clone(),
            uses: uses[index],
            reals: op.real_count(),
            bits,
            sources: op.provenance.sources.clone(),
            unresolved,
            data,
        });
    }
    Ok(ProgramView { bits: account.total_bits, components, unresolved_bits, data_bits, algorithm_bits })
}

/// A compact text rendering: one line per component with bits, sorted by bits, and the totals.
pub fn render(view: &ProgramView) -> String {
    let mut components: Vec<&ComponentView> = view.components.iter().filter(|c| c.bits > 0).collect();
    components.sort_by(|a, b| b.bits.cmp(&a.bits).then_with(|| a.name.cmp(&b.name)));
    let mut out = format!(
        "{} bits: {} algorithm, {} data; {:.1}% unresolved\n",
        view.bits,
        view.algorithm_bits,
        view.data_bits,
        100.0 * view.unresolved_fraction()
    );
    for c in components {
        out.push_str(&format!(
            "{:>10} bits {:>8} reals  {:<24} reads [{}] writes [{}] x{} {}{}\n",
            c.bits,
            c.reals,
            c.name,
            c.reads,
            c.writes,
            c.uses,
            c.laws.join("; "),
            if c.unresolved { "  (native)" } else { "" }
        ));
    }
    out
}

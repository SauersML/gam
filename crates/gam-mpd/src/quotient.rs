//! Exact invariance quotients of an operator program (#2951): the program's message never pays
//! for a coordinate the network's function cannot see.
//!
//! [`quotient`] maps a program to a canonical representative of its orbit under the exact
//! invariances below and returns the bits each one removed, measured on the program's own message
//! ([`OperatorProgram::code_bits`]). A step is kept only when the message is strictly shorter, so
//! the saving is exact and the procedure terminates. The invariances are read from the program's
//! node laws and its operator sharing alone; nothing is declared about the task.
//!
//! # Invariances
//!
//! * **Ties** ([`Invariance::Tie`]). Operators with equal interfaces and equal reals are one
//!   operator read twice; every use reads the first. The executed output is bit-identical.
//! * **Single-key softmax** ([`Invariance::SingleKey`]). A softmax over one score is `1` for every
//!   input (`exp(0)/exp(0)`, exact in binary64), so a mix read through it is its one payload; this
//!   is the first query position under a causal mask. The mix is replaced by its payload and the
//!   score path, now unread, is pruned. The executed output is bit-identical.
//! * **Softmax row shift** ([`Invariance::SoftmaxShift`]). A summand common to every score of a
//!   softmax does not move its weights. For affine scores that is a shared term or bias; for
//!   bilinear scores `c q·k_j` with one query `q`, it is a term or bias shared by every key `k_j`
//!   (a key bias, for one). Keys and scores are shared across softmaxes (every query reads every
//!   earlier key), so a summand is removed from a set of nodes only when every softmax reached
//!   through them has all of its scores in the set, found as the largest closed subset. An
//!   attention node reads every key of a sequence from one node, so there only a key bias is
//!   common, and only without a rotary turn.
//! * **Logit shift** ([`Invariance::LogitShift`]). The behaviour is `readouts` categorical
//!   distributions per input, each invariant under a common shift of its logits. At an output
//!   readout the shift is the direction `u` of the readout's input interface with `Φ u = 1`: every
//!   coordinate of an indicator basis (per readout block), and for a character basis the constant
//!   coordinate plus the off-cycle tokens. Every operator writing only to that node moves as
//!   `A ↦ A − u δᵀ`; per column, `δ` is the lattice integer minimising the column's exact code
//!   length (a sweep over the breakpoints of the Elias δ length), so the constant row of a character
//!   readout is removed outright.
//! * **Power-of-two diagonal gauge** ([`Invariance::UnitScale`]). Scaling a coordinate by `2^k`
//!   is exact in binary64 (away from overflow and subnormals), and it commutes with every
//!   positively homogeneous law: the identity and ReLU, a mix's payloads, a concat, one side of a
//!   bilinear product against `2^{−k}` on the other, one side of a Hadamard or outer product whose
//!   other side is fixed. The admissible exponents are the integer solutions of these equalities
//!   over every node coordinate and every operator row, column and low-rank inner coordinate, with
//!   inputs, softmax scores and weights, the readout, non-homogeneous laws (GELU, SiLU) and the
//!   output fixed at zero. Shared operators contribute one variable per coordinate however many
//!   nodes read them, so tied weights and heads sharing a key/value operator are covered by the
//!   same equations. The equations are pairwise (`x = ±y`, `x = 0`) and are solved by a signed
//!   union-find; a product whose sides are both free fixes its left side, which keeps a subgroup
//!   and so stays exact. Each free class is one generator. The representative takes, generator by
//!   generator, the exponent that shortens the exact message most while every moved real stays on
//!   its operator's lattice; the executed output is bit-identical.
//! * **Zero blocks** ([`Invariance::ZeroBlocks`]). A present block of zeros is sent absent.
//! * **Lattice** ([`Invariance::Lattice`]). An operator's reals are re-sent on the coarsest lattice
//!   that holds them exactly, when that is shorter; the reals do not move.
//!
//! * **Coordinate permutation** ([`Invariance::Permutation`]). Two free classes of the diagonal
//!   gauge whose members are the same node coordinates and operator rows, columns and inner
//!   coordinates, with the same signs, interface groups and pointwise laws, can be exchanged
//!   everywhere at once without moving the function: every law above is permutation-equivariant in
//!   such coordinates (hidden units, residual coordinates, a head's query/key or value/output
//!   coordinates). The representative sorts each set of interchangeable classes by one dense
//!   operator's rows, so that operator is sent with ordered rows, whose order costs nothing
//!   (`operator_program`'s ordered kind); the operator is the one whose ordered message is shortest.
//!   Summation orders change, so the output moves within its rounding band.
//!
//! Norms and rotary attention: an RMS norm sees a common scale of its input only through `ε`, so
//! its input carries one shared exponent when `ε = 0` and none otherwise, while a permutation of
//! coordinates passes through it (and through every pointwise law) unchanged. A rotary plane turns
//! its two query (and key) coordinates into each other, so they share one exponent and do not
//! permute. The rotary commutant's continuous part and the orthogonal gauge of a normed residual
//! stream are not lattice-exact; `canonical` treats them on native layers.

use super::codec::{signed_delta_len_bits, signed_prefix_integer_len_bits};
use super::engine::{EngineError, Edit, Exactness, Primitive, Proposal, SearchContext};
use super::fit::ProposalKind;
use super::gated_rewrite::rms_input_scale_is_symmetry;
use super::operator_program::{
    Basis, Interface, Label, LabelKind, Law, Node, Operator, OperatorBody, OperatorProgram, ProgramError, remap_node,
};
use super::precision::DeclaredPrecision;
use ndarray::s;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// An exact invariance [`quotient`] removes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Invariance {
    /// Nodes, operators and bases the output never reads.
    Unread,
    /// Operators equal in interfaces and reals, read as one shared operator.
    Tie,
    SingleKey,
    SoftmaxShift,
    LogitShift,
    ZeroBlocks,
    UnitScale,
    Permutation,
    Lattice,
}

const STEPS: [Invariance; 8] = [
    Invariance::Tie,
    Invariance::SingleKey,
    Invariance::SoftmaxShift,
    Invariance::LogitShift,
    Invariance::ZeroBlocks,
    Invariance::UnitScale,
    Invariance::Permutation,
    Invariance::Lattice,
];

/// A canonical representative and what reaching it saved.
#[derive(Clone, Debug)]
pub struct Quotient {
    pub program: OperatorProgram,
    pub bits_before: u64,
    pub bits_after: u64,
    /// Bits each invariance removed, summed over rounds; they add to `bits_before − bits_after`.
    pub saved: BTreeMap<Invariance, u64>,
    /// Free generators of the power-of-two diagonal gauge that move at least one real, on the
    /// representative.
    pub scale_generators: usize,
}

/// The quotient as a search move: the current program's canonical representative, proposed when
/// reaching it shortens the message. Every invariance is exact in exact arithmetic; the engine
/// certifies the representative like any other proposal.
pub struct Canonical;

impl Primitive for Canonical {
    fn name(&self) -> &'static str {
        "quotient"
    }

    fn propose(&self, context: &SearchContext<'_>) -> Result<Vec<Proposal>, EngineError> {
        let reached = quotient(context.program, context.contract.readouts)?;
        if reached.bits_after >= reached.bits_before {
            return Ok(Vec::new());
        }
        let removed: Vec<String> = reached.saved.iter().map(|(invariance, bits)| format!("{invariance:?} {bits}")).collect();
        Ok(vec![Proposal {
            primitive: "quotient",
            kind: ProposalKind::Reduce,
            exactness: Exactness::Exact { derivation: "the program's exact invariances (quotient module note)".to_string() },
            description: format!("the canonical representative under exact invariances: {} bits ({})", reached.bits_before - reached.bits_after, removed.join(", ")),
            edit: Edit::Program(Box::new(reached.program)),
        }])
    }
}

/// The canonical representative of `program` under the exact invariances of the module note, for
/// a behaviour of `readouts` distributions per input (the output's width in equal blocks).
pub fn quotient(program: &OperatorProgram, readouts: usize) -> Result<Quotient, ProgramError> {
    program.interfaces()?;
    let bits_before = program.code_bits()?;
    let mut current = program.clone();
    current.prune();
    let mut bits = current.code_bits()?;
    let mut saved = BTreeMap::new();
    if bits < bits_before {
        *saved.entry(Invariance::Unread).or_insert(0) += bits_before - bits;
    }
    loop {
        let start = bits;
        for invariance in STEPS {
            let candidate = match invariance {
                Invariance::Unread => None,
                Invariance::Tie => ties(&current),
                Invariance::SingleKey => single_keys(&current),
                Invariance::SoftmaxShift => softmax_shift(&current)?,
                Invariance::LogitShift => logit_shift(&current, readouts)?,
                Invariance::ZeroBlocks => zero_blocks(&current),
                Invariance::UnitScale => scale_gauge(&current)?.0,
                Invariance::Permutation => permutation(&current)?,
                Invariance::Lattice => lattice(&current)?,
            };
            let Some(candidate) = candidate else { continue };
            let candidate_bits = candidate.code_bits()?;
            if candidate_bits < bits {
                *saved.entry(invariance).or_insert(0) += bits - candidate_bits;
                bits = candidate_bits;
                current = candidate;
            }
        }
        if bits == start {
            break;
        }
    }
    let scale_generators = scale_gauge(&current)?.1;
    Ok(Quotient { program: current, bits_before, bits_after: bits, saved, scale_generators })
}

// ------------------------------------------------------------------------------------ program graph

/// For each node, the nodes that read it.
fn readers(program: &OperatorProgram) -> Vec<BTreeSet<usize>> {
    let mut out = vec![BTreeSet::new(); program.nodes.len()];
    for (index, node) in program.nodes.iter().enumerate() {
        for argument in node.arguments() {
            out[argument].insert(index);
        }
    }
    out
}

/// For each operator, the nodes that reference it, directly or through a called rule.
fn operator_uses(program: &OperatorProgram) -> Vec<BTreeSet<usize>> {
    let mut out = vec![BTreeSet::new(); program.operators.len()];
    for (index, node) in program.nodes.iter().enumerate() {
        for operator in program.node_operators(node) {
            out[operator].insert(index);
        }
    }
    out
}

/// `program` with every read of node `a` redirected to `redirect[a]`, then pruned.
fn redirected(program: &OperatorProgram, redirect: &[usize]) -> OperatorProgram {
    let mut out = program.clone();
    let operators: Vec<usize> = (0..program.operators.len()).collect();
    let bases: Vec<usize> = (0..program.bases.len()).collect();
    let rules: Vec<usize> = (0..program.rules.len()).collect();
    for node in &mut out.nodes {
        remap_node(node, redirect, &operators, &bases, &rules);
    }
    out.output = redirect[out.output];
    out.prune();
    out
}

// ------------------------------------------------------------------------------------ ties

/// Every use of an operator equal to an earlier one redirected to the earlier one.
fn ties(program: &OperatorProgram) -> Option<OperatorProgram> {
    let ops = &program.operators;
    let mut map: Vec<usize> = (0..ops.len()).collect();
    let mut changed = false;
    for i in 0..ops.len() {
        if let Some(j) = (0..i).find(|&j| map[j] == j && ops[j].rows == ops[i].rows && ops[j].cols == ops[i].cols && ops[j].body == ops[i].body) {
            map[i] = j;
            changed = true;
        }
    }
    if !changed {
        return None;
    }
    let mut out = program.clone();
    let bases: Vec<usize> = (0..program.bases.len()).collect();
    let rules: Vec<usize> = (0..program.rules.len()).collect();
    let nodes: Vec<usize> = (0..program.nodes.len()).collect();
    for node in &mut out.nodes {
        remap_node(node, &nodes, &map, &bases, &rules);
    }
    for rule in &mut out.rules {
        let local: Vec<usize> = (0..rule.nodes.len()).collect();
        for node in &mut rule.nodes {
            remap_node(node, &local, &map, &bases, &rules);
        }
    }
    out.prune();
    Some(out)
}

// ------------------------------------------------------------------------------------ single keys

/// Mixes read through a one-score softmax replaced by their payload.
pub(super) fn single_keys(program: &OperatorProgram) -> Option<OperatorProgram> {
    let readers = readers(program);
    let mut redirect: Vec<usize> = (0..program.nodes.len()).collect();
    let mut changed = false;
    for (index, node) in program.nodes.iter().enumerate() {
        let Node::Softmax { scores } = node else { continue };
        if scores.len() != 1 || index == program.output || readers[index].is_empty() {
            continue;
        }
        let all_mixes = readers[index].iter().all(|&reader| {
            matches!(&program.nodes[reader], Node::Mix { weights, payloads } if *weights == index && payloads.len() == 1)
        });
        if !all_mixes {
            continue;
        }
        for &reader in &readers[index] {
            if let Node::Mix { payloads, .. } = &program.nodes[reader] {
                redirect[reader] = payloads[0].1;
                changed = true;
            }
        }
    }
    if !changed {
        return None;
    }
    // A payload may itself be a collapsed mix; payloads precede their readers.
    for index in 0..redirect.len() {
        redirect[index] = redirect[redirect[index]];
    }
    Some(redirected(program, &redirect))
}

// ------------------------------------------------------------------------------------ softmax shift

/// A summand of an affine node: a term `(argument, operator)` or a bias operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Summand {
    Term(usize, usize),
    Bias(usize),
}

fn summands(node: &Node) -> Vec<Summand> {
    match node {
        Node::Affine { terms, bias } => {
            terms.iter().map(|&(a, o)| Summand::Term(a, o)).chain(bias.iter().map(|&o| Summand::Bias(o))).collect()
        }
        _ => Vec::new(),
    }
}

/// The largest subset of `carriers` from which one common summand can be removed without moving
/// any softmax's weights (module note, "Softmax row shift").
fn closed_carriers(
    program: &OperatorProgram,
    readers: &[BTreeSet<usize>],
    carriers: &BTreeSet<usize>,
    summand: Summand,
) -> BTreeSet<usize> {
    let mut set = carriers.clone();
    loop {
        let mut bad = BTreeSet::new();
        'carrier: for &carrier in &set {
            if carrier == program.output {
                bad.insert(carrier);
                continue;
            }
            for &reader in &readers[carrier] {
                match &program.nodes[reader] {
                    // One key node holds every key of the sequence, so only a bias is common to
                    // them all, and a rotary turn would make even that depend on the offset.
                    Node::Attend { query, key, value, rotary: None, .. }
                        if *key == carrier && *query != carrier && *value != carrier && matches!(summand, Summand::Bias(_)) =>
                    {
                        continue;
                    }
                    Node::Softmax { scores } => {
                        if !scores.iter().all(|score| set.contains(score)) {
                            bad.insert(carrier);
                            continue 'carrier;
                        }
                    }
                    Node::Bilinear { left, right, scale } => {
                        let varying_right = match (*left == carrier, *right == carrier) {
                            (false, true) => true,
                            (true, false) => false,
                            _ => {
                                bad.insert(carrier);
                                continue 'carrier;
                            }
                        };
                        let fixed = if varying_right { *left } else { *right };
                        if set.contains(&fixed) || reader == program.output || readers[reader].is_empty() {
                            bad.insert(carrier);
                            continue 'carrier;
                        }
                        for &softmax in &readers[reader] {
                            let Node::Softmax { scores } = &program.nodes[softmax] else {
                                bad.insert(carrier);
                                continue 'carrier;
                            };
                            let common = scores.iter().all(|&score| match &program.nodes[score] {
                                Node::Bilinear { left: l, right: r, scale: c } => {
                                    c == scale
                                        && if varying_right { *l == fixed && set.contains(r) } else { *r == fixed && set.contains(l) }
                                }
                                _ => false,
                            });
                            if !common {
                                bad.insert(carrier);
                                continue 'carrier;
                            }
                        }
                    }
                    _ => {
                        bad.insert(carrier);
                        continue 'carrier;
                    }
                }
            }
        }
        if bad.is_empty() {
            return set;
        }
        for carrier in bad {
            set.remove(&carrier);
        }
    }
}

/// Every summand common to the scores (or keys) of the softmaxes that read them removed.
fn softmax_shift(program: &OperatorProgram) -> Result<Option<OperatorProgram>, ProgramError> {
    let mut current = program.clone();
    let mut changed = false;
    loop {
        let readers = readers(&current);
        let mut carriers: BTreeMap<Summand, BTreeSet<usize>> = BTreeMap::new();
        for (index, node) in current.nodes.iter().enumerate() {
            let list = summands(node);
            if list.len() < 2 {
                continue;
            }
            for summand in list {
                carriers.entry(summand).or_default().insert(index);
            }
        }
        let found = carriers.iter().find_map(|(summand, set)| {
            let closed = closed_carriers(&current, &readers, set, *summand);
            let reaches_softmax = closed.iter().any(|&c| !readers[c].is_empty());
            (reaches_softmax && !closed.is_empty()).then_some((*summand, closed))
        });
        let Some((summand, closed)) = found else { break };
        for index in closed {
            if let Node::Affine { terms, bias } = &mut current.nodes[index] {
                match summand {
                    Summand::Term(argument, operator) => {
                        let at = terms.iter().position(|t| *t == (argument, operator));
                        if let Some(at) = at {
                            terms.remove(at);
                        }
                    }
                    Summand::Bias(operator) => {
                        if *bias == Some(operator) {
                            *bias = None;
                        }
                    }
                }
            }
        }
        changed = true;
    }
    if !changed {
        return Ok(None);
    }
    current.prune();
    current.interfaces()?;
    Ok(Some(current))
}

// ------------------------------------------------------------------------------------ logit shift

/// The logit-shift targets below the output: affine or constant nodes, each with its directions,
/// the sets of its coordinates on which a direction is one (it is zero elsewhere). A node's
/// directions are the blocks it holds; a concat hands each part the blocks inside it, a gain passes
/// them to its input, and a readout pulls its one block back through `Φ`. A node read by anything
/// but its parent is not a target.
fn shift_targets(program: &OperatorProgram, readouts: usize) -> Result<BTreeMap<usize, Vec<Vec<usize>>>, ProgramError> {
    let interfaces = program.interfaces()?;
    let readers = readers(program);
    let mut targets = BTreeMap::new();
    let width = interfaces[program.output].width();
    if readouts == 0 || width % readouts != 0 || !readers[program.output].is_empty() {
        return Ok(targets);
    }
    let size = width / readouts;
    let blocks: Vec<Vec<usize>> = (0..readouts).map(|b| (b * size..(b + 1) * size).collect()).collect();
    let mut stack = vec![(program.output, blocks)];
    while let Some((node, blocks)) = stack.pop() {
        match &program.nodes[node] {
            Node::Affine { .. } | Node::Constant { .. } => {
                targets.insert(node, blocks);
            }
            Node::Concat { parts } => {
                let mut offset = 0;
                for &part in parts {
                    let w = interfaces[part].width();
                    let inside: Vec<Vec<usize>> = blocks
                        .iter()
                        .filter(|b| b.iter().all(|&c| c >= offset && c < offset + w))
                        .map(|b| b.iter().map(|c| c - offset).collect())
                        .collect();
                    if !inside.is_empty() && readers[part] == BTreeSet::from([node]) {
                        stack.push((part, inside));
                    }
                    offset += w;
                }
            }
            // `c y` moves along the same directions as `y`.
            Node::Gain { input, .. } => {
                if readers[*input] == BTreeSet::from([node]) {
                    stack.push((*input, blocks));
                }
            }
            Node::Readout { input, basis } => {
                if readers[*input] != BTreeSet::from([node]) {
                    continue;
                }
                let classes = interfaces[node].width();
                let whole = blocks.len() == 1 && blocks[0].len() == classes;
                let directions = match &program.bases[*basis] {
                    Basis::Indicator { .. } => Some(blocks),
                    Basis::Characters { positions, .. } if whole => {
                        let interface = &interfaces[*input];
                        let mut coordinates = Vec::new();
                        if let Some(g) = interface.find(Label::new(LabelKind::Const, 0)) {
                            coordinates.extend(interface.range(g));
                        }
                        for (token, position) in positions.iter().enumerate() {
                            if position.is_none() {
                                let g = interface
                                    .find(Label::new(LabelKind::Token, token as u32))
                                    .ok_or_else(|| ProgramError::Interface(format!("off-cycle token {token} has no group")))?;
                                coordinates.extend(interface.range(g));
                            }
                        }
                        Some(vec![coordinates])
                    }
                    Basis::Characters { .. } => None,
                };
                if let Some(directions) = directions {
                    stack.push((*input, directions));
                }
            }
            // No other law carries a common shift of its value to its reals.
            Node::Feature { .. }
            | Node::Raw { .. }
            | Node::Bilinear { .. }
            | Node::Softmax { .. }
            | Node::Mix { .. }
            | Node::Pointwise { .. }
            | Node::Hadamard { .. }
            | Node::Outer { .. }
            | Node::Param { .. }
            | Node::Call { .. }
            | Node::Attend { .. }
            | Node::RmsNorm { .. }
            | Node::Transposed { .. } => continue,
        }
    }
    Ok(targets)
}

/// Every operator written only to shift targets that share its directions moved along them,
/// column by column, to its shortest lattice representative.
fn logit_shift(program: &OperatorProgram, readouts: usize) -> Result<Option<OperatorProgram>, ProgramError> {
    let targets = shift_targets(program, readouts)?;
    let uses = operator_uses(program);
    let mut current = program.clone();
    let mut changed = false;
    let operators: BTreeSet<usize> = targets.keys().flat_map(|&t| program.node_operators(&program.nodes[t])).collect();
    for operator in operators {
        let mut users = uses[operator].iter();
        let Some(directions) = users.next().and_then(|u| targets.get(u)) else { continue };
        if !users.all(|u| targets.get(u) == Some(directions)) {
            continue;
        }
        if !matches!(current.operators[operator].body, OperatorBody::Dense { .. }) {
            continue;
        }
        let op = Arc::make_mut(&mut current.operators[operator]);
        let (rows, cols) = (op.rows.clone(), op.cols.clone());
        let OperatorBody::Dense { values, present, precision } = &mut op.body else { continue };
        let scale = 1.0 / precision.step();
        for direction in directions {
            for column in 0..cols.width() {
                let col_group = cols.group_of(column);
                if !direction.iter().all(|&r| present[[rows.group_of(r), col_group]]) {
                    continue;
                }
                let indices: Vec<i64> = direction.iter().map(|&r| (values[[r, column]] * scale) as i64).collect();
                let delta = shortest_shift(&indices)?;
                if delta == 0 {
                    continue;
                }
                for (&r, &n) in direction.iter().zip(&indices) {
                    values[[r, column]] = (n - delta) as f64 * precision.step();
                }
                changed = true;
            }
        }
    }
    Ok(changed.then_some(current))
}

/// The length of a lattice index in the operator message.
pub(super) fn index_bits(index: i64) -> Result<u64, ProgramError> {
    Ok(signed_delta_len_bits(index)?)
}

/// The integer `δ` minimising `Σ_i len(n_i − δ)` for the signed Elias δ length, which depends on
/// `|m|` only and steps up at `|m| = 2^j`; ties go to the `δ` nearest zero, and `0` is kept unless
/// another value is strictly shorter. With `len(m) = len(0) + Σ_j Δ_j [|m| ≥ 2^j]`, the length is
/// `const − Σ_j Δ_j #{i : |n_i − δ| < 2^j}`, a sum of window counts maximised by one sweep over the
/// window ends.
pub(super) fn shortest_shift(indices: &[i64]) -> Result<i64, ProgramError> {
    if indices.len() < 2 {
        return Ok(indices.first().copied().unwrap_or(0));
    }
    let (lo, hi) = indices.iter().fold((i64::MAX, i64::MIN), |(a, b), &n| (a.min(n), b.max(n)));
    let spread = (hi - lo) as u64;
    let rings = (64 - spread.leading_zeros()) as usize + 1;
    let mut steps = Vec::with_capacity(rings);
    for j in 0..rings {
        let at = 1i64 << j;
        steps.push(index_bits(at)? - index_bits(at - 1)?);
    }
    let gain = |delta: i64| -> u64 {
        let mut total = 0;
        for &n in indices {
            let distance = n.abs_diff(delta);
            for (j, &step) in steps.iter().enumerate() {
                if distance < (1u64 << j) {
                    total += step;
                }
            }
        }
        total
    };
    let mut events: Vec<(i64, i64)> = Vec::with_capacity(2 * rings * indices.len());
    for &n in indices {
        for (j, &step) in steps.iter().enumerate() {
            if step == 0 {
                continue;
            }
            let half = (1i64 << j) - 1;
            events.push((n - half, step as i64));
            events.push((n + half + 1, -(step as i64)));
        }
    }
    events.sort_unstable();
    let mut best = (gain(0) as i64, 0i64);
    let mut level = 0i64;
    let mut k = 0;
    while k < events.len() {
        let at = events[k].0;
        while k < events.len() && events[k].0 == at {
            level += events[k].1;
            k += 1;
        }
        let end = events.get(k).map_or(at, |e| e.0 - 1);
        // Constant on `[at, end]`: its point nearest zero, kept inside the indices' hull.
        let (a, b) = (at.max(lo), end.min(hi));
        if a > b {
            continue;
        }
        let point = if a > 0 { a } else if b < 0 { b } else { 0 };
        if level > best.0 || (level == best.0 && point.unsigned_abs() < best.1.unsigned_abs()) {
            best = (level, point);
        }
    }
    Ok(best.1)
}

// ------------------------------------------------------------------------------------ zero blocks

fn zero_blocks(program: &OperatorProgram) -> Option<OperatorProgram> {
    let mut current = program.clone();
    let mut changed = false;
    for shared in &mut current.operators {
        let OperatorBody::Dense { values, present, .. } = &shared.body else { continue };
        let zero = |(r, c): (usize, usize)| values.slice(s![shared.rows.range(r), shared.cols.range(c)]).iter().all(|v| *v == 0.0);
        let zeros: Vec<(usize, usize)> = present.indexed_iter().filter(|(at, keep)| **keep && zero(*at)).map(|(at, _)| at).collect();
        if zeros.is_empty() {
            continue;
        }
        let op = Arc::make_mut(shared);
        let OperatorBody::Dense { present, .. } = &mut op.body else { continue };
        for at in zeros {
            present[at] = false;
        }
        changed = true;
    }
    changed.then_some(current)
}

// ------------------------------------------------------------------------------------ lattices

/// An operator's present reals as lattice indices, with where each sits.
#[derive(Clone, Copy, Debug)]
enum Slot {
    Dense(usize, usize),
    Left(usize, usize),
    Right(usize, usize),
    Diagonal(usize),
}

fn lattice_entries(body: &OperatorBody, rows: &Interface, cols: &Interface) -> Vec<(Slot, f64)> {
    match body {
        OperatorBody::Identity => Vec::new(),
        OperatorBody::Dense { values, present, .. } => {
            let mut out = Vec::new();
            for ((r, c), &keep) in present.indexed_iter() {
                if keep {
                    for i in rows.range(r) {
                        for j in cols.range(c) {
                            out.push((Slot::Dense(i, j), values[[i, j]]));
                        }
                    }
                }
            }
            out
        }
        OperatorBody::LowRank { left, right, .. } => left
            .indexed_iter()
            .map(|((i, m), v)| (Slot::Left(i, m), *v))
            .chain(right.indexed_iter().map(|((m, j), v)| (Slot::Right(m, j), *v)))
            .collect(),
        OperatorBody::Diagonal { values, .. } => values.indexed_iter().map(|(i, v)| (Slot::Diagonal(i), *v)).collect(),
    }
}

fn precision_of(body: &OperatorBody) -> Option<DeclaredPrecision> {
    match body {
        OperatorBody::Dense { precision, .. } | OperatorBody::LowRank { precision, .. } | OperatorBody::Diagonal { precision, .. } => {
            Some(*precision)
        }
        OperatorBody::Identity => None,
    }
}

/// Write `value` at `slot`, which [`lattice_entries`] took from this body.
fn set_entry(body: &mut OperatorBody, slot: Slot, value: f64) {
    let target = match (body, slot) {
        (OperatorBody::Dense { values, .. }, Slot::Dense(i, j)) => values.get_mut((i, j)),
        (OperatorBody::LowRank { left, .. }, Slot::Left(i, m)) => left.get_mut((i, m)),
        (OperatorBody::LowRank { right, .. }, Slot::Right(m, j)) => right.get_mut((m, j)),
        (OperatorBody::Diagonal { values, .. }, Slot::Diagonal(i)) => values.get_mut(i),
        (OperatorBody::Identity | OperatorBody::Dense { .. } | OperatorBody::LowRank { .. } | OperatorBody::Diagonal { .. }, _) => None,
    };
    if let Some(entry) = target {
        *entry = value;
    }
}

fn set_precision(body: &mut OperatorBody, to: DeclaredPrecision) {
    if let OperatorBody::Dense { precision, .. } | OperatorBody::LowRank { precision, .. } | OperatorBody::Diagonal { precision, .. } = body {
        *precision = to;
    }
}

/// The coarsening `t ≤ min trailing zeros` of `indices`' lattice that minimises the lattice
/// message (the fraction bits' code plus every index's), and its length saving.
fn best_coarsening(indices: &[i64], fraction_bits: i32) -> Result<(u32, u64), ProgramError> {
    let trailing = indices.iter().filter(|n| **n != 0).map(|n| n.trailing_zeros()).min();
    let limit = match trailing {
        Some(t) => t,
        // All zero: any lattice holds them; the one with the shortest fraction-bits code.
        None => fraction_bits.unsigned_abs(),
    };
    let cost = |t: u32| -> Result<u64, ProgramError> {
        let mut total = signed_prefix_integer_len_bits(i64::from(fraction_bits) - i64::from(t))?;
        for &n in indices {
            total += index_bits(n.checked_shr(t).unwrap_or(0))?;
        }
        Ok(total)
    };
    let base = cost(0)?;
    let mut best = (0, base);
    for t in 1..=limit {
        if DeclaredPrecision::new(fraction_bits - t as i32).is_err() {
            break;
        }
        let c = cost(t)?;
        if c < best.1 {
            best = (t, c);
        }
    }
    Ok((best.0, base - best.1))
}

fn coarsen_operator(body: &mut OperatorBody, rows: &Interface, cols: &Interface) -> Result<bool, ProgramError> {
    let Some(precision) = precision_of(body) else { return Ok(false) };
    let scale = 1.0 / precision.step();
    let entries = lattice_entries(body, rows, cols);
    let indices: Vec<i64> = entries.iter().map(|(_, v)| (v * scale) as i64).collect();
    let (t, saving) = best_coarsening(&indices, precision.fraction_bits())?;
    if t == 0 || saving == 0 {
        return Ok(false);
    }
    let coarse = DeclaredPrecision::new(precision.fraction_bits() - t as i32).map_err(ProgramError::Code)?;
    set_precision(body, coarse);
    Ok(true)
}

/// Every operator on its shortest exact lattice.
fn lattice(program: &OperatorProgram) -> Result<Option<OperatorProgram>, ProgramError> {
    let mut current = program.clone();
    let mut changed = false;
    for shared in &mut current.operators {
        let mut body = shared.body.clone();
        if coarsen_operator(&mut body, &shared.rows, &shared.cols)? {
            Arc::make_mut(shared).body = body;
            changed = true;
        }
    }
    Ok(changed.then_some(current))
}

// ------------------------------------------------------------------------------------ scale gauge

/// Pairwise equalities `σ_x = ±σ_y` and `σ_x = 0` over integer exponents.
struct SignedUnion {
    parent: Vec<usize>,
    /// `σ_x = sign[x] σ_parent[x]`.
    sign: Vec<i8>,
    pinned: Vec<bool>,
}

impl SignedUnion {
    fn new(size: usize) -> Self {
        Self { parent: (0..size).collect(), sign: vec![1; size], pinned: vec![false; size] }
    }

    /// The root of `x` and `s` with `σ_x = s σ_root`.
    fn find(&mut self, x: usize) -> (usize, i8) {
        let mut path = Vec::new();
        let mut at = x;
        while self.parent[at] != at {
            path.push(at);
            at = self.parent[at];
        }
        let root = at;
        // Compress from the top: each node's sign relative to the root.
        for &node in path.iter().rev() {
            let parent = self.parent[node];
            let parent_sign = if parent == root { 1 } else { self.sign[parent] };
            self.sign[node] *= parent_sign;
            self.parent[node] = root;
        }
        (root, if x == root { 1 } else { self.sign[x] })
    }

    /// `σ_x = s σ_y`.
    fn join(&mut self, x: usize, y: usize, s: i8) {
        let (rx, sx) = self.find(x);
        let (ry, sy) = self.find(y);
        let relative = sx * s * sy;
        if rx == ry {
            if relative != 1 {
                self.pinned[rx] = true;
            }
            return;
        }
        self.parent[rx] = ry;
        self.sign[rx] = relative;
        self.pinned[ry] |= self.pinned[rx];
    }

    fn pin(&mut self, x: usize) {
        let (root, _) = self.find(x);
        self.pinned[root] = true;
    }

    fn is_pinned(&mut self, x: usize) -> bool {
        let (root, _) = self.find(x);
        self.pinned[root]
    }
}

/// The variables of the diagonal gauge: node coordinates, then operator rows, columns and
/// low-rank inner coordinates.
struct Variables {
    node: Vec<usize>,
    row: Vec<usize>,
    col: Vec<usize>,
    inner: Vec<usize>,
    count: usize,
}

impl Variables {
    fn new(program: &OperatorProgram, interfaces: &[Interface]) -> Self {
        let mut count = 0;
        let mut take = |width: usize| {
            count += width;
            count - width
        };
        let node = interfaces.iter().map(|i| take(i.width())).collect();
        let row = program.operators.iter().map(|o| take(o.rows.width())).collect();
        let col = program.operators.iter().map(|o| take(o.cols.width())).collect();
        let inner = program
            .operators
            .iter()
            .map(|o| take(if let OperatorBody::LowRank { left, .. } = &o.body { left.ncols() } else { 0 }))
            .collect();
        Self { node, row, col, inner, count }
    }
}

/// The equations of the module note's power-of-two diagonal gauge, or, with `permutation`, of the
/// coordinate classes a permutation may exchange: there every coordinate-wise law (any pointwise
/// law, an RMS norm, a Hadamard product) carries a coordinate to itself, where a scale would
/// have to be fixed.
fn gauge_equations(program: &OperatorProgram, interfaces: &[Interface], v: &Variables, permutation: bool) -> SignedUnion {
    let mut u = SignedUnion::new(v.count);
    let width = |n: usize| interfaces[n].width();
    let mut products: Vec<(usize, usize, usize)> = Vec::new();
    for (index, node) in program.nodes.iter().enumerate() {
        let at = |i: usize| v.node[index] + i;
        match node {
            Node::Feature { .. } | Node::Raw { .. } => (0..width(index)).for_each(|i| u.pin(at(i))),
            Node::Constant { operator } => {
                (0..width(index)).for_each(|i| u.join(at(i), v.row[*operator] + i, 1));
                u.pin(v.col[*operator]);
            }
            Node::Affine { terms, bias } => {
                for &(argument, operator) in terms {
                    let op = &program.operators[operator];
                    for c in 0..op.cols.width() {
                        u.join(v.col[operator] + c, v.node[argument] + c, 1);
                    }
                    for r in 0..op.rows.width() {
                        u.join(at(r), v.row[operator] + r, 1);
                    }
                    if matches!(op.body, OperatorBody::Identity) {
                        for r in 0..op.rows.width() {
                            u.join(v.row[operator] + r, v.col[operator] + r, 1);
                        }
                    }
                }
                if let Some(operator) = bias {
                    for r in 0..width(index) {
                        u.join(at(r), v.row[*operator] + r, 1);
                    }
                    u.pin(v.col[*operator]);
                }
            }
            // `y = x A`: an entry `A[r, c]` carries `2^{ρ_r − κ_c}`, so `ρ_r = −σ_x,r` and
            // `κ_c = −σ_y,c`.
            Node::Transposed { input, operator } => {
                let op = &program.operators[*operator];
                for r in 0..op.rows.width() {
                    u.join(v.row[*operator] + r, v.node[*input] + r, -1);
                }
                for c in 0..op.cols.width() {
                    u.join(v.col[*operator] + c, at(c), -1);
                }
            }
            Node::Bilinear { left, right, .. } => {
                u.pin(at(0));
                for i in 0..width(*left) {
                    u.join(v.node[*left] + i, v.node[*right] + i, -1);
                }
            }
            // Scores `c (R q)·(R k)` pair each query coordinate with its key coordinate; a rotary
            // plane turns its two coordinates into each other. The output reads the values.
            Node::Attend { query, key, value, rotary, .. } => {
                for i in 0..width(*query) {
                    u.join(v.node[*query] + i, v.node[*key] + i, -1);
                }
                if let Some(rotary) = rotary {
                    let half = rotary.dims as usize / 2;
                    for i in 0..half {
                        let (a, b) = if rotary.half_split { (i, i + half) } else { (2 * i, 2 * i + 1) };
                        u.join(v.node[*query] + a, v.node[*query] + b, 1);
                    }
                }
                (0..width(index)).for_each(|i| u.join(at(i), v.node[*value] + i, 1));
            }
            Node::Softmax { scores } => {
                for &score in scores {
                    u.pin(v.node[score]);
                }
                (0..width(index)).for_each(|i| u.pin(at(i)));
            }
            Node::Mix { weights, payloads } => {
                (0..width(*weights)).for_each(|i| u.pin(v.node[*weights] + i));
                for &(_, payload) in payloads {
                    (0..width(index)).for_each(|i| u.join(at(i), v.node[payload] + i, 1));
                }
            }
            Node::Pointwise { input, laws } => {
                let interface = &interfaces[*input];
                for (group, law) in laws.iter().enumerate() {
                    for i in interface.range(group) {
                        match law {
                            Law::Relu | Law::Identity => u.join(at(i), v.node[*input] + i, 1),
                            Law::Zero if !permutation => continue,
                            Law::Silu | Law::Gelu | Law::GeluTanh if !permutation => {
                                u.pin(at(i));
                                u.pin(v.node[*input] + i);
                            }
                            Law::Zero | Law::Silu | Law::Gelu | Law::GeluTanh => u.join(at(i), v.node[*input] + i, 1),
                        }
                    }
                }
            }
            // `x / √(mean x² + ε)` sees a common scale of its input only through `ε`: with `ε = 0`
            // every input coordinate shares one exponent and the output none.
            Node::RmsNorm { input, epsilon } => {
                if permutation {
                    (0..width(index)).for_each(|i| u.join(at(i), v.node[*input] + i, 1));
                } else {
                    (0..width(index)).for_each(|i| u.pin(at(i)));
                    if rms_input_scale_is_symmetry(*epsilon) {
                        (1..width(*input)).for_each(|i| u.join(v.node[*input] + i, v.node[*input], 1));
                    } else {
                        (0..width(*input)).for_each(|i| u.pin(v.node[*input] + i));
                    }
                }
            }
            Node::Hadamard { left, right } => {
                for i in 0..width(index) {
                    if permutation {
                        u.join(at(i), v.node[*left] + i, 1);
                        u.join(at(i), v.node[*right] + i, 1);
                    } else {
                        products.push((at(i), v.node[*left] + i, v.node[*right] + i));
                    }
                }
            }
            Node::Outer { left, right } => {
                let (li, ri) = (&interfaces[*left], &interfaces[*right]);
                let mut offset = 0;
                for g1 in 0..li.group_count() {
                    for g2 in 0..ri.group_count() {
                        for i in li.range(g1) {
                            for j in ri.range(g2) {
                                if permutation {
                                    u.pin(at(offset));
                                    u.pin(v.node[*left] + i);
                                    u.pin(v.node[*right] + j);
                                } else {
                                    products.push((at(offset), v.node[*left] + i, v.node[*right] + j));
                                }
                                offset += 1;
                            }
                        }
                    }
                }
            }
            Node::Concat { parts } => {
                let mut offset = 0;
                for &part in parts {
                    for i in 0..width(part) {
                        u.join(at(offset + i), v.node[part] + i, 1);
                    }
                    offset += width(part);
                }
            }
            Node::Readout { input, .. } => {
                (0..width(*input)).for_each(|i| u.pin(v.node[*input] + i));
                (0..width(index)).for_each(|i| u.pin(at(i)));
            }
            Node::Gain { input, .. } => (0..width(index)).for_each(|i| u.join(at(i), v.node[*input] + i, 1)),
            // A rule body is shared by every call and not quotiented here: its arguments, its
            // value and every operator it reads are fixed.
            Node::Call { arguments, .. } => {
                for &argument in arguments {
                    (0..width(argument)).for_each(|i| u.pin(v.node[argument] + i));
                }
                (0..width(index)).for_each(|i| u.pin(at(i)));
                for operator in program.node_operators(node) {
                    let op = &program.operators[operator];
                    (0..op.rows.width()).for_each(|i| u.pin(v.row[operator] + i));
                    (0..op.cols.width()).for_each(|i| u.pin(v.col[operator] + i));
                    if let OperatorBody::LowRank { left, .. } = &op.body {
                        (0..left.ncols()).for_each(|i| u.pin(v.inner[operator] + i));
                    }
                }
            }
            Node::Param { .. } => (0..width(index)).for_each(|i| u.pin(at(i))),
        }
    }
    (0..width(program.output)).for_each(|i| u.pin(v.node[program.output] + i));
    // `σ_out = σ_l + σ_r`: with one side fixed it is an equality; otherwise the left side is fixed.
    for (out, l, r) in products {
        if u.is_pinned(l) {
            u.join(out, r, 1);
        } else if u.is_pinned(r) {
            u.join(out, l, 1);
        } else {
            u.pin(l);
            u.join(out, r, 1);
        }
    }
    u
}

/// One real of the gauge: operator, slot, lattice index, and the (row, column) variables whose
/// exponents it carries as `2^{σ_row − σ_col}`.
struct GaugeEntry {
    operator: usize,
    slot: Slot,
    index: i64,
    row: usize,
    col: usize,
}

/// The shortest representative under the power-of-two diagonal gauge, and the number of free
/// generators that move a real.
pub(super) fn scale_gauge(program: &OperatorProgram) -> Result<(Option<OperatorProgram>, usize), ProgramError> {
    let interfaces = program.interfaces()?;
    let v = Variables::new(program, &interfaces);
    let mut u = gauge_equations(program, &interfaces, &v, false);
    let mut entries: Vec<GaugeEntry> = Vec::new();
    for (operator, op) in program.operators.iter().enumerate() {
        let Some(precision) = precision_of(&op.body) else { continue };
        let scale = 1.0 / precision.step();
        for (slot, value) in lattice_entries(&op.body, &op.rows, &op.cols) {
            let (row, col) = match slot {
                Slot::Dense(i, j) => (v.row[operator] + i, v.col[operator] + j),
                Slot::Left(i, m) => (v.row[operator] + i, v.inner[operator] + m),
                Slot::Right(m, j) => (v.inner[operator] + m, v.col[operator] + j),
                Slot::Diagonal(i) => (v.row[operator] + i, v.col[operator] + i),
            };
            entries.push(GaugeEntry { operator, slot, index: (value * scale) as i64, row, col });
        }
    }
    // Per free generator, its entries and their exponent coefficients.
    let mut generators: BTreeMap<usize, Vec<(usize, i32)>> = BTreeMap::new();
    for (k, entry) in entries.iter().enumerate() {
        let (rr, rs) = u.find(entry.row);
        let (cr, cs) = u.find(entry.col);
        let mut coefficients: BTreeMap<usize, i32> = BTreeMap::new();
        if !u.pinned[rr] {
            *coefficients.entry(rr).or_insert(0) += i32::from(rs);
        }
        if !u.pinned[cr] {
            *coefficients.entry(cr).or_insert(0) -= i32::from(cs);
        }
        for (g, a) in coefficients {
            if a != 0 {
                generators.entry(g).or_default().push((k, a));
            }
        }
    }
    let free = generators.values().filter(|list| list.iter().any(|(k, _)| entries[*k].index != 0)).count();
    let mut precisions: Vec<Option<DeclaredPrecision>> = program.operators.iter().map(|o| precision_of(&o.body)).collect();
    let mut changed = false;
    loop {
        let mut improved = false;
        for list in generators.values() {
            let (mut lo, mut hi) = (i64::MIN, i64::MAX);
            let mut any = false;
            for &(k, a) in list {
                let n = entries[k].index;
                if n == 0 {
                    continue;
                }
                any = true;
                let (down, up) = (-i64::from(n.trailing_zeros()), 53 - i64::from(64 - n.unsigned_abs().leading_zeros()));
                let a = i64::from(a);
                // `down ≤ a Δ ≤ up`.
                let (l, h) = if a > 0 { (down.div_euclid(a) + i64::from(down.rem_euclid(a) != 0), up.div_euclid(a)) } else {
                    let b = -a;
                    ((-up).div_euclid(b) + i64::from((-up).rem_euclid(b) != 0), (-down).div_euclid(b))
                };
                lo = lo.max(l);
                hi = hi.min(h);
            }
            if !any || lo > hi || (lo == 0 && hi == 0) {
                continue;
            }
            let cost = |delta: i64| -> Result<u64, ProgramError> {
                let mut total = 0;
                for &(k, a) in list {
                    let n = entries[k].index;
                    let shift = i64::from(a) * delta;
                    let moved = if shift >= 0 { n << shift } else { n >> -shift };
                    total += index_bits(moved)?;
                }
                Ok(total)
            };
            let base = cost(0)?;
            let mut best = (base, 0i64);
            for delta in lo..=hi {
                let c = cost(delta)?;
                if c < best.0 || (c == best.0 && delta.abs() < best.1.abs()) {
                    best = (c, delta);
                }
            }
            if best.1 == 0 || best.0 >= base {
                continue;
            }
            for &(k, a) in list {
                let shift = i64::from(a) * best.1;
                let n = entries[k].index;
                entries[k].index = if shift >= 0 { n << shift } else { n >> -shift };
            }
            improved = true;
            changed = true;
        }
        // Coarsen every lattice the moves allow, then look again.
        let mut by_operator: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (k, entry) in entries.iter().enumerate() {
            by_operator.entry(entry.operator).or_default().push(k);
        }
        for (operator, list) in by_operator {
            let Some(precision) = precisions[operator] else { continue };
            let indices: Vec<i64> = list.iter().map(|&k| entries[k].index).collect();
            let (t, saving) = best_coarsening(&indices, precision.fraction_bits())?;
            if t > 0 && saving > 0 && indices.iter().any(|n| *n != 0) {
                for &k in &list {
                    entries[k].index >>= t;
                }
                precisions[operator] =
                    Some(DeclaredPrecision::new(precision.fraction_bits() - t as i32).map_err(ProgramError::Code)?);
                improved = true;
                changed = true;
            }
        }
        if !improved {
            break;
        }
    }
    if !changed {
        return Ok((None, free));
    }
    let mut current = program.clone();
    for (operator, precision) in precisions.iter().enumerate() {
        if let Some(precision) = precision {
            set_precision(&mut Arc::make_mut(&mut current.operators[operator]).body, *precision);
        }
    }
    for entry in &entries {
        let step = precisions[entry.operator].map_or(1.0, DeclaredPrecision::step);
        set_entry(&mut Arc::make_mut(&mut current.operators[entry.operator]).body, entry.slot, entry.index as f64 * step);
    }
    Ok((Some(current), free))
}

// ------------------------------------------------------------------------------------ permutation

/// What a gauge variable is a coordinate of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Member {
    Node(usize),
    Row(usize),
    Col(usize),
    Inner(usize),
}

/// Where coordinate `i` of an interface may move without changing the interface: anywhere among
/// single-coordinate groups of one label kind, or within its own wider group.
fn coordinate_context(interface: &Interface, i: usize) -> (LabelKind, usize) {
    let group = interface.group_of(i);
    let g = interface.groups()[group];
    (g.label.kind, if g.width == 1 { usize::MAX } else { group })
}

/// The pointwise law applied to coordinate `i` of node `node`, as an index, when a pointwise node
/// applies one.
fn law_at(program: &OperatorProgram, interfaces: &[Interface], node: usize, i: usize) -> Option<u8> {
    match &program.nodes[node] {
        Node::Pointwise { input, laws } => Some(match laws[interfaces[*input].group_of(i)] {
            Law::Relu => 0,
            Law::Identity => 1,
            Law::Zero => 2,
            Law::Silu => 3,
            Law::Gelu => 4,
            Law::GeluTanh => 5,
        }),
        _ => None,
    }
}

/// One member of a class: the member, the sign, where its coordinate may move, and its law.
type Signature = Vec<(Member, i8, (LabelKind, usize), Option<u8>)>;

/// Interchangeable free classes, sorted by one dense operator's rows (module note, "Coordinate
/// permutation").
fn permutation(program: &OperatorProgram) -> Result<Option<OperatorProgram>, ProgramError> {
    let interfaces = program.interfaces()?;
    let v = Variables::new(program, &interfaces);
    let mut u = gauge_equations(program, &interfaces, &v, true);
    let mut owner: Vec<(Member, usize)> = Vec::with_capacity(v.count);
    for (n, interface) in interfaces.iter().enumerate() {
        owner.extend((0..interface.width()).map(|i| (Member::Node(n), i)));
    }
    for (o, op) in program.operators.iter().enumerate() {
        owner.extend((0..op.rows.width()).map(|i| (Member::Row(o), i)));
    }
    for (o, op) in program.operators.iter().enumerate() {
        owner.extend((0..op.cols.width()).map(|i| (Member::Col(o), i)));
    }
    for (o, op) in program.operators.iter().enumerate() {
        let inner = if let OperatorBody::LowRank { left, .. } = &op.body { left.ncols() } else { 0 };
        owner.extend((0..inner).map(|i| (Member::Inner(o), i)));
    }
    // Per free class: its signature and each member's coordinate.
    let mut classes: BTreeMap<usize, (Signature, BTreeMap<Member, usize>)> = BTreeMap::new();
    let mut refused: BTreeSet<usize> = BTreeSet::new();
    for (variable, &(member, i)) in owner.iter().enumerate() {
        let (root, sign) = u.find(variable);
        if u.pinned[root] {
            continue;
        }
        let (context, law) = match member {
            Member::Node(n) => (coordinate_context(&interfaces[n], i), law_at(program, &interfaces, n, i)),
            Member::Row(o) => (coordinate_context(&program.operators[o].rows, i), None),
            Member::Col(o) => (coordinate_context(&program.operators[o].cols, i), None),
            Member::Inner(_) => ((LabelKind::Native, usize::MAX), None),
        };
        let entry = classes.entry(root).or_default();
        if entry.1.insert(member, i).is_some() {
            refused.insert(root);
        }
        entry.0.push((member, sign, context, law));
    }
    let mut sets: BTreeMap<Signature, Vec<usize>> = BTreeMap::new();
    for (root, (mut signature, _)) in classes.clone() {
        if refused.contains(&root) {
            continue;
        }
        signature.sort_by(|a, b| a.0.cmp(&b.0));
        // Signs are relative to an arbitrary root: normalise to the first member's.
        let first = signature[0].1;
        signature.iter_mut().for_each(|entry| entry.1 *= first);
        sets.entry(signature).or_default().push(root);
    }
    let mut current = program.clone();
    let mut bits = current.code_bits()?;
    let mut changed = false;
    for (signature, roots) in sets {
        if roots.len() < 2 {
            continue;
        }
        let candidates: Vec<usize> = signature
            .iter()
            .filter_map(|(member, ..)| match member {
                Member::Row(o) => Some(*o),
                Member::Node(_) | Member::Col(_) | Member::Inner(_) => None,
            })
            .filter(|&o| matches!(current.operators[o].body, OperatorBody::Dense { .. }))
            .collect();
        // Only the members' operators change, so the message changes by their lengths alone.
        let touched: BTreeSet<usize> = signature
            .iter()
            .filter_map(|(member, ..)| match member {
                Member::Row(o) | Member::Col(o) | Member::Inner(o) => Some(*o),
                Member::Node(_) => None,
            })
            .collect();
        let length = |program: &OperatorProgram| -> Result<u64, ProgramError> {
            touched.iter().map(|&o| program.operators[o].code_bits().map(|(a, b)| a + b)).sum()
        };
        let base = length(&current)?;
        let mut best: Option<(u64, OperatorProgram)> = None;
        for sort_by in candidates {
            let candidate = sorted_classes(&current, &roots, &classes, sort_by)?;
            let candidate_bits = bits - base + length(&candidate)?;
            if candidate_bits < best.as_ref().map_or(bits, |b| b.0) {
                best = Some((candidate_bits, candidate));
            }
        }
        if let Some((candidate_bits, candidate)) = best {
            bits = candidate_bits;
            current = candidate;
            changed = true;
        }
    }
    Ok(changed.then_some(current))
}

/// `program` with the classes `roots` exchanged so that operator `sort_by`'s rows at their
/// coordinates ascend lexicographically in their lattice indices.
fn sorted_classes(
    program: &OperatorProgram,
    roots: &[usize],
    classes: &BTreeMap<usize, (Signature, BTreeMap<Member, usize>)>,
    sort_by: usize,
) -> Result<OperatorProgram, ProgramError> {
    let op = &program.operators[sort_by];
    let OperatorBody::Dense { values, present, precision } = &op.body else {
        return Err(ProgramError::Input(format!("operator {} is not dense", op.name)));
    };
    let scale = 1.0 / precision.step();
    let row_key = |r: usize| -> Vec<i64> {
        let mut key = Vec::new();
        for c in 0..op.cols.group_count() {
            if present[[op.rows.group_of(r), c]] {
                key.extend(op.cols.range(c).map(|j| (values[[r, j]] * scale) as i64));
            }
        }
        key
    };
    let coordinate = |root: usize, member: Member| classes[&root].1[&member];
    // The slots, in the sorting operator's row order, and the classes in key order.
    let mut occupants: Vec<usize> = roots.to_vec();
    occupants.sort_by_key(|&root| coordinate(root, Member::Row(sort_by)));
    let mut ordered: Vec<usize> = roots.to_vec();
    ordered.sort_by_cached_key(|&root| (row_key(coordinate(root, Member::Row(sort_by))), coordinate(root, Member::Row(sort_by))));
    let members: Vec<Member> = classes[&roots[0]].1.keys().copied().collect();
    let mut out = program.clone();
    for member in members {
        // Coordinate map old → new for this member.
        let moves: Vec<(usize, usize)> = ordered
            .iter()
            .zip(&occupants)
            .map(|(&class, &slot)| (coordinate(class, member), coordinate(slot, member)))
            .collect();
        // An operator's rows and columns may both move (a square map of the stream to itself):
        // each move reads the operator as the previous one left it.
        match member {
            Member::Node(_) => {}
            Member::Row(o) => {
                let source = out.operators[o].clone();
                permute_rows(Arc::make_mut(&mut out.operators[o]), &source, &moves);
            }
            Member::Col(o) => {
                let source = out.operators[o].clone();
                permute_cols(Arc::make_mut(&mut out.operators[o]), &source, &moves);
            }
            Member::Inner(o) => {
                let source = out.operators[o].body.clone();
                if let (OperatorBody::LowRank { left, right, .. }, OperatorBody::LowRank { left: l0, right: r0, .. }) =
                    (&mut Arc::make_mut(&mut out.operators[o]).body, &source)
                {
                    for &(from, to) in &moves {
                        left.column_mut(to).assign(&l0.column(from));
                        right.row_mut(to).assign(&r0.row(from));
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Rows `from → to` of `source` written into `target` (whose other rows are already in place),
/// with their present blocks when the rows are single-row groups.
fn permute_rows(target: &mut Operator, source: &Operator, moves: &[(usize, usize)]) {
    let rows = source.rows.clone();
    if let (OperatorBody::Dense { values, present, .. }, OperatorBody::Dense { values: v0, present: p0, .. }) =
        (&mut target.body, &source.body)
    {
        for &(from, to) in moves {
            values.row_mut(to).assign(&v0.row(from));
            let (gf, gt) = (rows.group_of(from), rows.group_of(to));
            if rows.groups()[gf].width == 1 && rows.groups()[gt].width == 1 {
                present.row_mut(gt).assign(&p0.row(gf));
            }
        }
    } else if let (OperatorBody::LowRank { left, .. }, OperatorBody::LowRank { left: l0, .. }) =
        (&mut target.body, &source.body)
    {
        for &(from, to) in moves {
            left.row_mut(to).assign(&l0.row(from));
        }
    }
}

/// Columns `from → to` of `source` written into `target`, as [`permute_rows`].
fn permute_cols(target: &mut Operator, source: &Operator, moves: &[(usize, usize)]) {
    let cols = source.cols.clone();
    if let (OperatorBody::Dense { values, present, .. }, OperatorBody::Dense { values: v0, present: p0, .. }) =
        (&mut target.body, &source.body)
    {
        for &(from, to) in moves {
            values.column_mut(to).assign(&v0.column(from));
            let (gf, gt) = (cols.group_of(from), cols.group_of(to));
            if cols.groups()[gf].width == 1 && cols.groups()[gt].width == 1 {
                present.column_mut(gt).assign(&p0.column(gf));
            }
        }
    } else if let (OperatorBody::LowRank { right, .. }, OperatorBody::LowRank { right: r0, .. }) =
        (&mut target.body, &source.body)
    {
        for &(from, to) in moves {
            right.column_mut(to).assign(&r0.column(from));
        }
    }
}

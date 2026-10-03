//! Exact path decomposition of an operator program's value, linearized over its trace (#2951).
//!
//! # Conditioning
//!
//! Every node of an [`OperatorProgram`] that can carry a path is linear in some of its arguments
//! once the others and its own executed values are fixed at the [`Trace`]: an affine node in each
//! term; a readout, a transposed read, a gain and a concatenation in their input; an RMS norm at its
//! executed per-row scale `(mean x² + ε)^{-1/2}`; a pointwise law `σ(z) = γ(z) z` at its executed gate
//! `γ(z) = σ(z)/z` (`1[z > 0]` for ReLU, `Φ(z)` for the exact GELU, the logistic for SiLU, any
//! value where `z = 0`, which contributes nothing); a mix in its payloads at the executed softmax
//! weights; an attend node in its value at the attention its traced query and key fix; a Hadamard
//! product in one side at the executed other side (the gate side: the one argument that is a
//! non-identity pointwise law, or the side [`PathOptions::hadamard`] declares). Each is an identity
//! at the traced input, not a linearization: no derivative enters, and the conditioned program
//! returns the traced value.
//!
//! What the conditioning fixes is listed beside the result ([`Condition`]): attention queries and
//! keys, mix weights, gates and norm scales. Each conditioned argument is itself a node, so its own
//! value splits exactly over paths by [`decompose`] with that node as the target (the QK dependence
//! of a head is the decomposition of its query and key nodes, and the score is bilinear in them).
//! A node that cannot be split exactly is a source of its own and says why ([`SourceKind::Opaque`]):
//! a bilinear score, a softmax, an outer product, a rule call, a Hadamard product with no gate side,
//! a node that reads one argument both as a condition and as a value.
//!
//! # Paths
//!
//! A path is a source node followed by nodes each read linearly by the next, ending at the target.
//! A source is an input (a feature, a raw slot, a constant operator), a bias an affine node emits,
//! or an opaque node's whole value. A path's contribution is the composed conditioned maps applied
//! to its source; since every linear node's value is the sum of its conditioned reads of its
//! arguments plus what it emits, the contributions of all paths sum to the target's traced value.
//!
//! # Certified enumeration
//!
//! A prefix whose value `v` sits at node `n` roots every path extending it; their total mass
//! `Σ ‖c_P‖_F` is at most `Γ(n) ‖v‖_F`, with `Γ(target) = 1` and `Γ(n) = Σ_{c reads n} ‖M_{c,n}‖ Γ(c)`
//! from certified upper bounds of the conditioned maps (a certified spectral bound of an operator,
//! the largest gate, scale or weight, `√L` for an attention read over at most `L` rows of a
//! sequence, whose weights sum to one per row). Prefixes are expanded best first; a node with one
//! reader is passed through without a queue step. [`PathOptions::expansions`] is the declared
//! number of expansions. Every prefix left is a [`PathRemainder`]: the exact net value of its
//! subtree (one conditioned propagation) and its certified mass bound. Paths and remainders sum to
//! the traced value, so the cut is certified, never a threshold.
//!
//! # Rounding
//!
//! The identity holds exactly for the conditioned program. The traced value and every computed item
//! err from it by at most `γ_K A`, `A` the absolute conditioned forward (every map and source in
//! absolute value) and `K` the inner dimensions along the longest chain;
//! [`PathDecomposition::rounding_band`] adds the summation of the items.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use std::fmt;
use std::ops::Range;

use faer::Side;
use gam_linalg::faer_ndarray::{FaerArrayView, FaerLinalgError, fast_ata, self_adjoint_eigenvalues};
use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth, symmetric_spectrum_rounding_band_at_dim};
use gam_runtime::resource::{MemoryGovernor, MemoryReservation, MemoryReservationError};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2, s};

use crate::operator_program::{
    FamilyInputs, Interface, Law, Node, OperatorBody, OperatorProgram, ProgramError, Trace, rms_scale,
};

/// Why a path decomposition refused.
#[derive(Debug)]
pub enum PathError {
    Program(ProgramError),
    /// A dense matrix the decomposition forms does not fit the memory budget.
    Memory { context: &'static str, source: MemoryReservationError },
    /// A self-adjoint eigendecomposition failed.
    Eigen { context: &'static str, source: FaerLinalgError },
    /// A matrix whose norm is bounded is not finite.
    NonFiniteMatrix { context: &'static str },
    /// The target is not a node, or the trace does not cover it.
    Target { node: usize, nodes: usize },
    NonFinite { node: usize },
}

impl fmt::Display for PathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Program(error) => write!(formatter, "paths: {error}"),
            Self::Memory { context, source } => write!(formatter, "{context}: {source}"),
            Self::Eigen { context, source } => write!(formatter, "{context}: self-adjoint eigendecomposition failed: {source}"),
            Self::NonFiniteMatrix { context } => write!(formatter, "{context}: non-finite value"),
            Self::Target { node, nodes } => write!(formatter, "paths: target {node} of a {nodes}-node trace"),
            Self::NonFinite { node } => write!(formatter, "paths: node {node} is not finite"),
        }
    }
}

impl std::error::Error for PathError {}

impl From<ProgramError> for PathError {
    fn from(error: ProgramError) -> Self {
        Self::Program(error)
    }
}

/// The argument of a Hadamard product that carries paths; the other is its executed gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HadamardSide {
    Left,
    Right,
}

/// The declared number of best-first expansions, and Hadamard sides declared by node.
#[derive(Clone, Debug)]
pub struct PathOptions {
    pub expansions: usize,
    pub hadamard: BTreeMap<usize, HadamardSide>,
}

impl PathOptions {
    pub fn expansions(expansions: usize) -> Self {
        Self { expansions, hadamard: BTreeMap::new() }
    }
}

/// What a source node is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceKind {
    /// A feature, a raw slot, a constant operator or a rule argument.
    Input,
    /// The bias of an affine node.
    Emitted,
    /// A node whose value is not split by path, with the reason.
    Opaque(&'static str),
}

/// What a conditioned argument fixes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConditionKind {
    AttentionQuery,
    AttentionKey,
    MixWeights,
    HadamardGate,
    /// A pointwise law's gate `σ(z)/z`, a function of the node's own input.
    PointwiseGate,
    /// An RMS norm's scale, a function of the node's own input.
    NormScale,
}

/// An argument the linearization holds at its traced value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Condition {
    pub consumer: usize,
    pub argument: usize,
    pub kind: ConditionKind,
}

/// One enumerated path: its nodes (source first, target last), its exact contribution, its share
/// `⟨c, y⟩/⟨y, y⟩` of the target's value `y`, and its prefix's bound when it was expanded.
#[derive(Clone, Debug)]
pub struct TracePath {
    pub nodes: Vec<usize>,
    pub contribution: Array2<f64>,
    pub share: f64,
    pub bound: f64,
}

/// An unexpanded prefix: the exact net contribution of every path extending it, and a certified
/// bound on their total mass `Σ ‖c_P‖_F`.
#[derive(Clone, Debug)]
pub struct PathRemainder {
    pub nodes: Vec<usize>,
    pub net: Array2<f64>,
    pub share: f64,
    pub mass_bound: f64,
}

/// One hop of a path: `reader` reads `writer`'s value linearly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PathHop {
    pub writer: usize,
    pub reader: usize,
}

/// The enumerated paths through one hop: their summed contribution, mass `Σ ‖c_P‖_F` and count.
/// What the remainders carry is bounded by [`PathDecomposition::remainder_mass_bound`].
#[derive(Clone, Debug)]
pub struct HopFlow {
    pub hop: PathHop,
    pub net: Array2<f64>,
    pub mass: f64,
    pub paths: usize,
}

impl TracePath {
    /// The path's hops in order.
    pub fn hops(&self) -> Vec<PathHop> {
        self.nodes.windows(2).map(|pair| PathHop { writer: pair[0], reader: pair[1] }).collect()
    }

    pub fn mass(&self) -> f64 {
        frobenius(&self.contribution)
    }
}

/// A target's paths, remainders, sources and conditions.
#[derive(Clone, Debug)]
pub struct PathDecomposition {
    pub target: usize,
    /// The traced value `y` of the target.
    pub value: Array2<f64>,
    /// Paths in expansion order.
    pub paths: Vec<TracePath>,
    /// Remainders, largest bound first.
    pub remainders: Vec<PathRemainder>,
    pub remainder_mass_bound: f64,
    /// `‖Σ items − y‖_F` is at most this.
    pub rounding_band: f64,
    /// Every source node that reaches the target, with its kind.
    pub sources: Vec<(usize, SourceKind)>,
    /// Every conditioned argument of a node that reaches the target.
    pub conditions: Vec<Condition>,
}

impl PathDecomposition {
    /// The enumerated paths grouped by hop, largest mass first. Every path takes exactly one hop into
    /// the target (or is the target itself, a source with no hop).
    pub fn hop_flows(&self) -> Vec<HopFlow> {
        let mut flows: BTreeMap<PathHop, HopFlow> = BTreeMap::new();
        for path in &self.paths {
            let mass = path.mass();
            for hop in path.hops() {
                let flow = flows.entry(hop).or_insert_with(|| HopFlow {
                    hop,
                    net: Array2::zeros(path.contribution.dim()),
                    mass: 0.0,
                    paths: 0,
                });
                flow.net += &path.contribution;
                flow.mass += mass;
                flow.paths += 1;
            }
        }
        let mut flows: Vec<HopFlow> = flows.into_values().collect();
        flows.sort_by(|a, b| b.mass.total_cmp(&a.mass));
        flows
    }

    /// The fewest items (paths and remainders, largest mass first) whose mass reaches `fraction` of
    /// the items' total mass.
    pub fn mass_count(&self, fraction: f64) -> usize {
        let mut masses: Vec<f64> =
            self.paths.iter().map(TracePath::mass).chain(self.remainders.iter().map(|r| frobenius(&r.net))).collect();
        masses.sort_by(|a, b| b.total_cmp(a));
        let total: f64 = masses.iter().sum();
        let mut running = 0.0;
        for (index, mass) in masses.iter().enumerate() {
            running += mass;
            if running >= fraction * total {
                return index + 1;
            }
        }
        masses.len()
    }

    /// `Σ` paths `+ Σ` remainders.
    pub fn items_sum(&self) -> Array2<f64> {
        let mut sum = Array2::<f64>::zeros(self.value.dim());
        for path in &self.paths {
            sum += &path.contribution;
        }
        for remainder in &self.remainders {
            sum += &remainder.net;
        }
        sum
    }
}

fn frobenius(matrix: &Array2<f64>) -> f64 {
    matrix.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// A conditioned linear map from an argument's value to its reader's value.
#[derive(Clone, Debug)]
enum EdgeMap {
    /// `x ↦ k x`: identity terms, `k` of them.
    Identity(f64),
    /// `x ↦ x Mᵀ`.
    Dense(Array2<f64>),
    /// `x ↦ c x`.
    Scalar(f64),
    /// `x` placed at each column range of an output this wide.
    Place { ranges: Vec<Range<usize>>, width: usize },
    /// Row `r` times `s_r`.
    RowScale(Array1<f64>),
    /// `x ⊙ E`.
    Entrywise(Array2<f64>),
    /// The attend node's read of `x` as its value at the traced attention; `rows` bounds the rows of
    /// a sequence.
    Attend { reader: usize, value: usize, rows: usize },
}

impl EdgeMap {
    /// Inner operations per output entry, for the rounding band.
    fn inner(&self) -> usize {
        match self {
            Self::Identity(_) | Self::Scalar(_) | Self::Place { .. } => 1,
            Self::Dense(matrix) => matrix.ncols(),
            Self::RowScale(_) | Self::Entrywise(_) => 2,
            Self::Attend { rows, .. } => rows + 2,
        }
    }
}

/// How a node is split.
#[derive(Clone, Debug)]
enum Split {
    Source(SourceKind),
    /// Its linear arguments with their maps, and what it emits.
    Linear { edges: Vec<(usize, EdgeMap)>, emitted: Option<Array2<f64>> },
}

/// The program linearized at one trace.
struct Linearized<'a> {
    program: &'a OperatorProgram,
    inputs: &'a FamilyInputs,
    trace: &'a Trace,
    interfaces: Vec<Interface>,
    target: usize,
    splits: Vec<Split>,
    /// Per node, its readers through a linear edge that reach the target.
    readers: Vec<Vec<usize>>,
    reaches: Vec<bool>,
}

fn gate(law: Law, z: f64) -> f64 {
    if z != 0.0 {
        return law.apply(z) / z;
    }
    match law {
        Law::Relu | Law::Zero => 0.0,
        Law::Identity => 1.0,
        Law::Silu | Law::Gelu | Law::GeluTanh => 0.5,
    }
}

impl<'a> Linearized<'a> {
    fn new(
        program: &'a OperatorProgram,
        inputs: &'a FamilyInputs,
        trace: &'a Trace,
        target: usize,
        options: &PathOptions,
    ) -> Result<(Self, Vec<Condition>), PathError> {
        let interfaces = program.interfaces()?;
        let rows = inputs.rows;
        let sequence_rows = match &inputs.layout {
            Some(layout) => {
                let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
                for &sequence in &layout.sequence {
                    *counts.entry(sequence).or_default() += 1;
                }
                counts.values().copied().max().unwrap_or(1)
            }
            None => 1,
        };
        let mut splits = Vec::with_capacity(target + 1);
        let mut conditions = Vec::new();
        for (index, node) in program.nodes.iter().enumerate().take(target + 1) {
            let value = move |node: usize| &trace.values[node];
            let split = match node {
                Node::Feature { .. } | Node::Raw { .. } | Node::Constant { .. } | Node::Param { .. } => Split::Source(SourceKind::Input),
                Node::Call { .. } => Split::Source(SourceKind::Opaque("a rule call is not split")),
                Node::Bilinear { .. } => Split::Source(SourceKind::Opaque("a bilinear score")),
                Node::Softmax { .. } => Split::Source(SourceKind::Opaque("a softmax")),
                Node::Outer { .. } => Split::Source(SourceKind::Opaque("an outer product")),
                Node::Affine { terms, bias } => {
                    let mut grouped: BTreeMap<usize, EdgeMap> = BTreeMap::new();
                    for &(argument, operator) in terms {
                        let op = &program.operators[operator];
                        let map = match &op.body {
                            OperatorBody::Identity => EdgeMap::Identity(1.0),
                            _ => EdgeMap::Dense(op.matrix()),
                        };
                        let merged = match (grouped.remove(&argument), map) {
                            (None, map) => map,
                            (Some(EdgeMap::Identity(a)), EdgeMap::Identity(b)) => EdgeMap::Identity(a + b),
                            (Some(EdgeMap::Identity(a)), EdgeMap::Dense(m)) | (Some(EdgeMap::Dense(m)), EdgeMap::Identity(a)) => {
                                EdgeMap::Dense(m + Array2::<f64>::eye(op.rows.width()) * a)
                            }
                            (Some(EdgeMap::Dense(a)), EdgeMap::Dense(b)) => EdgeMap::Dense(a + b),
                            (Some(other), _) => other,
                        };
                        grouped.insert(argument, merged);
                    }
                    let emitted = bias.map(|op| {
                        let column = program.operators[op].matrix().column(0).to_owned();
                        Array2::from_shape_fn((rows, column.len()), |(_, c)| column[c])
                    });
                    Split::Linear { edges: grouped.into_iter().collect(), emitted }
                }
                Node::Readout { input, basis } => {
                    let base = &program.bases[*basis];
                    let size = program.declarations.domains[base.domain()].size;
                    let classes: Vec<u32> = (0..size as u32).collect();
                    let phi = base.evaluate(&program.declarations, &classes)?.values;
                    Split::Linear { edges: vec![(*input, EdgeMap::Dense(phi))], emitted: None }
                }
                Node::Transposed { input, operator } => {
                    let a = program.operators[*operator].matrix();
                    Split::Linear { edges: vec![(*input, EdgeMap::Dense(a.t().to_owned()))], emitted: None }
                }
                Node::Gain { input, .. } => {
                    // The gain's coefficient, read off the node's own law at a field of ones.
                    let ones = Array2::<f64>::ones(value(*input).dim());
                    let patch = BTreeMap::from([(*input, ones)]);
                    let scaled = program.evaluate_with(index, inputs, trace, &patch, &interfaces)?;
                    let coefficient = scaled.first().copied().unwrap_or(0.0);
                    Split::Linear { edges: vec![(*input, EdgeMap::Scalar(coefficient))], emitted: None }
                }
                Node::Concat { parts } => {
                    let width: usize = parts.iter().map(|&part| value(part).ncols()).sum();
                    let mut placed: BTreeMap<usize, Vec<Range<usize>>> = BTreeMap::new();
                    let mut offset = 0;
                    for &part in parts {
                        let w = value(part).ncols();
                        placed.entry(part).or_default().push(offset..offset + w);
                        offset += w;
                    }
                    let edges = placed.into_iter().map(|(part, ranges)| (part, EdgeMap::Place { ranges, width })).collect();
                    Split::Linear { edges, emitted: None }
                }
                Node::RmsNorm { input, epsilon } => {
                    conditions.push(Condition { consumer: index, argument: *input, kind: ConditionKind::NormScale });
                    let scales = value(*input).rows().into_iter().map(|row| rms_scale(row, *epsilon)).collect();
                    Split::Linear { edges: vec![(*input, EdgeMap::RowScale(scales))], emitted: None }
                }
                Node::Pointwise { input, laws } => {
                    conditions.push(Condition { consumer: index, argument: *input, kind: ConditionKind::PointwiseGate });
                    let z = value(*input);
                    let interface = &interfaces[*input];
                    let mut gates = Array2::<f64>::zeros(z.dim());
                    for (group, law) in laws.iter().enumerate() {
                        for column in interface.range(group) {
                            for row in 0..z.nrows() {
                                gates[[row, column]] = gate(*law, z[[row, column]]);
                            }
                        }
                    }
                    Split::Linear { edges: vec![(*input, EdgeMap::Entrywise(gates))], emitted: None }
                }
                Node::Mix { weights, payloads } => {
                    if payloads.iter().any(|(_, payload)| payload == weights) {
                        Split::Source(SourceKind::Opaque("a mix reads its weights as a payload"))
                    } else {
                        conditions.push(Condition { consumer: index, argument: *weights, kind: ConditionKind::MixWeights });
                        let alpha = value(*weights);
                        let mut grouped: BTreeMap<usize, Array1<f64>> = BTreeMap::new();
                        for &(column, payload) in payloads {
                            let entry = grouped.entry(payload).or_insert_with(|| Array1::zeros(rows));
                            *entry += &alpha.column(column);
                        }
                        Split::Linear { edges: grouped.into_iter().map(|(p, a)| (p, EdgeMap::RowScale(a))).collect(), emitted: None }
                    }
                }
                Node::Attend { query, key, value: read, .. } => {
                    if read == query || read == key {
                        Split::Source(SourceKind::Opaque("an attend node reads its value as a query or key"))
                    } else {
                        conditions.push(Condition { consumer: index, argument: *query, kind: ConditionKind::AttentionQuery });
                        conditions.push(Condition { consumer: index, argument: *key, kind: ConditionKind::AttentionKey });
                        let map = EdgeMap::Attend { reader: index, value: *read, rows: sequence_rows };
                        Split::Linear { edges: vec![(*read, map)], emitted: None }
                    }
                }
                Node::Hadamard { left, right } => {
                    let is_gate = |node: usize| {
                        matches!(&program.nodes[node], Node::Pointwise { laws, .. } if laws.iter().any(|law| *law != Law::Identity))
                    };
                    let side = options.hadamard.get(&index).copied().or(match (is_gate(*left), is_gate(*right)) {
                        (true, false) => Some(HadamardSide::Right),
                        (false, true) => Some(HadamardSide::Left),
                        _ => None,
                    });
                    match side {
                        _ if left == right => Split::Source(SourceKind::Opaque("a Hadamard square")),
                        None => Split::Source(SourceKind::Opaque("a Hadamard product with no gate side")),
                        Some(side) => {
                            let (carried, gated) = match side {
                                HadamardSide::Left => (*left, *right),
                                HadamardSide::Right => (*right, *left),
                            };
                            conditions.push(Condition { consumer: index, argument: gated, kind: ConditionKind::HadamardGate });
                            Split::Linear { edges: vec![(carried, EdgeMap::Entrywise(value(gated).clone()))], emitted: None }
                        }
                    }
                }
            };
            splits.push(split);
        }
        let mut readers = vec![Vec::new(); target + 1];
        for (index, split) in splits.iter().enumerate() {
            if let Split::Linear { edges, .. } = split {
                for (argument, _) in edges {
                    readers[*argument].push(index);
                }
            }
        }
        let mut reaches = vec![false; target + 1];
        reaches[target] = true;
        for index in (0..target).rev() {
            reaches[index] = readers[index].iter().any(|&reader| reaches[reader]);
        }
        for list in &mut readers {
            list.retain(|&reader| reaches[reader]);
            list.dedup();
        }
        conditions.retain(|condition| reaches[condition.consumer]);
        Ok((Self { program, inputs, trace, interfaces, target, splits, readers, reaches }, conditions))
    }

    fn edge(&self, reader: usize, argument: usize) -> Option<&EdgeMap> {
        match &self.splits[reader] {
            Split::Linear { edges, .. } => edges.iter().find(|(a, _)| *a == argument).map(|(_, map)| map),
            Split::Source(_) => None,
        }
    }

    /// `M_{reader, argument}` applied to `field` (with `abs`, `|M|` applied to `field ≥ 0`).
    fn apply(&self, map: &EdgeMap, field: &Array2<f64>, abs: bool) -> Result<Array2<f64>, PathError> {
        Ok(match map {
            EdgeMap::Identity(k) => field * if abs { k.abs() } else { *k },
            EdgeMap::Dense(matrix) => {
                if abs {
                    field.dot(&matrix.mapv(f64::abs).t())
                } else {
                    field.dot(&matrix.t())
                }
            }
            EdgeMap::Scalar(c) => field * if abs { c.abs() } else { *c },
            EdgeMap::Place { ranges, width } => {
                let mut out = Array2::<f64>::zeros((field.nrows(), *width));
                for range in ranges {
                    let mut target = out.slice_mut(s![.., range.clone()]);
                    target += field;
                }
                out
            }
            EdgeMap::RowScale(scales) => {
                let column = if abs { scales.mapv(f64::abs) } else { scales.clone() };
                field * &column.insert_axis(Axis(1))
            }
            EdgeMap::Entrywise(factors) => {
                if abs {
                    field * &factors.mapv(f64::abs)
                } else {
                    field * factors
                }
            }
            // The attention weights are nonnegative, so the read of `|x|` is the read `|M|` makes.
            EdgeMap::Attend { reader, value, .. } => {
                let patch = BTreeMap::from([(*value, field.clone())]);
                self.program.evaluate_with(*reader, self.inputs, self.trace, &patch, &self.interfaces)?
            }
        })
    }

    /// A certified upper bound on `‖M‖` as a map of Frobenius norms.
    fn norm_bound(&self, governor: &MemoryGovernor, map: &EdgeMap) -> Result<f64, PathError> {
        let bound = match map {
            EdgeMap::Identity(k) | EdgeMap::Scalar(k) => k.abs(),
            EdgeMap::Dense(matrix) => {
                let formation = accumulation_growth(matrix.ncols() + 2) * frobenius(&matrix.mapv(f64::abs));
                spectral_norm_bounds(governor, matrix, formation, "paths: operator")?.upper
            }
            EdgeMap::Place { ranges, .. } => (ranges.len() as f64).sqrt(),
            EdgeMap::RowScale(scales) => scales.iter().fold(0.0_f64, |acc, v| acc.max(v.abs())),
            EdgeMap::Entrywise(factors) => factors.iter().fold(0.0_f64, |acc, v| acc.max(v.abs())),
            // Rows of weights sum to one, so ‖P‖₁ ≤ L and ‖P‖∞ = 1: ‖P‖₂ ≤ √L.
            EdgeMap::Attend { rows, .. } => (*rows as f64).sqrt(),
        };
        Ok(bound * (1.0 + accumulation_growth(8)))
    }

    /// The target's value from `field` entering at `node`: every path extending it, summed.
    fn propagate(&self, node: usize, field: &Array2<f64>) -> Result<Array2<f64>, PathError> {
        let mut values: BTreeMap<usize, Array2<f64>> = BTreeMap::from([(node, field.clone())]);
        for reader in node + 1..=self.target {
            if !self.reaches[reader] {
                continue;
            }
            let Split::Linear { edges, .. } = &self.splits[reader] else { continue };
            let mut sum: Option<Array2<f64>> = None;
            for (argument, map) in edges {
                if let Some(input) = values.get(argument) {
                    let read = self.apply(map, input, false)?;
                    sum = Some(match sum {
                        Some(total) => total + read,
                        None => read,
                    });
                }
            }
            if let Some(sum) = sum {
                values.insert(reader, sum);
            }
        }
        Ok(values.remove(&self.target).unwrap_or_else(|| Array2::zeros(self.trace.values[self.target].dim())))
    }
}

/// A prefix in the best-first queue.
struct Prefix {
    bound: f64,
    order: usize,
    nodes: Vec<usize>,
    value: Array2<f64>,
}

impl PartialEq for Prefix {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Prefix {}

impl PartialOrd for Prefix {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Prefix {
    /// Larger bound first; among equal bounds, the earlier queued.
    fn cmp(&self, other: &Self) -> Ordering {
        self.bound.total_cmp(&other.bound).then_with(|| other.order.cmp(&self.order))
    }
}

/// The paths of node `target`'s traced value (see the module note). `trace` is `program`'s
/// execution on `inputs`.
pub fn decompose(
    governor: &MemoryGovernor,
    program: &OperatorProgram,
    inputs: &FamilyInputs,
    trace: &Trace,
    target: usize,
    options: &PathOptions,
) -> Result<PathDecomposition, PathError> {
    if target >= program.nodes.len() || target >= trace.values.len() {
        return Err(PathError::Target { node: target, nodes: trace.values.len().min(program.nodes.len()) });
    }
    for (node, value) in trace.values.iter().enumerate().take(target + 1) {
        if value.iter().any(|v| !v.is_finite()) {
            return Err(PathError::NonFinite { node });
        }
    }
    let (linear, conditions) = Linearized::new(program, inputs, trace, target, options)?;
    let value = program.node_value(trace, inputs, target)?.into_owned();

    // Γ(n): a certified bound on the summed norms of the suffix maps from n to the target.
    let mut gamma = vec![0.0; target + 1];
    gamma[target] = 1.0;
    for node in (0..target).rev() {
        if !linear.reaches[node] {
            continue;
        }
        let mut total = 0.0;
        for &reader in &linear.readers[node] {
            if let Some(map) = linear.edge(reader, node) {
                total += linear.norm_bound(governor, map)? * gamma[reader];
            }
        }
        gamma[node] = total * (1.0 + accumulation_growth(linear.readers[node].len() + 2));
    }

    let mut sources = Vec::new();
    let mut queue = BinaryHeap::new();
    let mut reservations: Vec<MemoryReservation> = Vec::new();
    let mut order = 0;
    let rows = inputs.rows;
    let mut push = |queue: &mut BinaryHeap<Prefix>, nodes: Vec<usize>, value: Array2<f64>| -> Result<(), PathError> {
        if value.iter().all(|&v| v == 0.0) {
            return Ok(());
        }
        let node = *nodes.last().unwrap_or(&target);
        let bound = gamma[node] * frobenius(&value) * (1.0 + accumulation_growth(value.len()));
        reservations.push(reserve(governor, rows.max(1), value.ncols().max(1), 1, "paths: prefix")?);
        queue.push(Prefix { bound, order, nodes, value });
        order += 1;
        Ok(())
    };
    for node in 0..=target {
        if !linear.reaches[node] {
            continue;
        }
        match &linear.splits[node] {
            Split::Source(kind) => {
                sources.push((node, *kind));
                push(&mut queue, vec![node], program.node_value(trace, inputs, node)?.into_owned())?;
            }
            Split::Linear { emitted: Some(emitted), .. } => {
                sources.push((node, SourceKind::Emitted));
                push(&mut queue, vec![node], emitted.clone())?;
            }
            Split::Linear { .. } => {}
        }
    }
    let denominator = value.iter().map(|v| v * v).sum::<f64>();
    let share = |contribution: &Array2<f64>| {
        if denominator > 0.0 { (contribution * &value).sum() / denominator } else { 0.0 }
    };
    let mut paths = Vec::new();
    let mut expansions = 0;
    while expansions < options.expansions {
        let Some(prefix) = queue.pop() else { break };
        expansions += 1;
        let (mut nodes, mut field) = (prefix.nodes, prefix.value);
        // Pass through nodes with one reader: no branching, no queue step.
        loop {
            let node = *nodes.last().unwrap_or(&target);
            if node == target || linear.readers[node].len() != 1 {
                break;
            }
            let reader = linear.readers[node][0];
            let map = linear.edge(reader, node).ok_or(PathError::Target { node: reader, nodes: target + 1 })?;
            field = linear.apply(map, &field, false)?;
            nodes.push(reader);
        }
        let node = *nodes.last().unwrap_or(&target);
        if node == target {
            paths.push(TracePath { share: share(&field), bound: prefix.bound, nodes, contribution: field });
            continue;
        }
        for &reader in &linear.readers[node] {
            let map = linear.edge(reader, node).ok_or(PathError::Target { node: reader, nodes: target + 1 })?;
            let read = linear.apply(map, &field, false)?;
            let mut extended = nodes.clone();
            extended.push(reader);
            push(&mut queue, extended, read)?;
        }
    }
    let mut remainders = Vec::with_capacity(queue.len());
    for prefix in queue.into_sorted_vec().into_iter().rev() {
        let node = *prefix.nodes.last().unwrap_or(&target);
        let net = if node == target { prefix.value } else { linear.propagate(node, &prefix.value)? };
        remainders.push(PathRemainder { share: share(&net), mass_bound: prefix.bound, nodes: prefix.nodes, net });
    }
    let remainder_mass_bound = remainders.iter().map(|r| r.mass_bound).sum();
    let rounding_band = rounding_band(&linear, &paths, &remainders, &value)?;
    drop(push);
    drop(reservations);
    Ok(PathDecomposition { target, value, paths, remainders, remainder_mass_bound, rounding_band, sources, conditions })
}

/// `2 γ_K ‖A‖_F` for the absolute conditioned forward `A` and the inner dimensions `K` along the
/// longest chain, plus the summation of the items.
fn rounding_band(
    linear: &Linearized<'_>,
    paths: &[TracePath],
    remainders: &[PathRemainder],
    value: &Array2<f64>,
) -> Result<f64, PathError> {
    let target = linear.target;
    let mut absolute: Vec<Option<Array2<f64>>> = vec![None; target + 1];
    let mut depth = vec![0usize; target + 1];
    for node in 0..=target {
        if !linear.reaches[node] {
            continue;
        }
        let (field, inner) = match &linear.splits[node] {
            Split::Source(_) => (linear.program.node_value(linear.trace, linear.inputs, node)?.mapv(f64::abs), 0),
            Split::Linear { edges, emitted } => {
                let mut total: Option<Array2<f64>> = emitted.as_ref().map(|e| e.mapv(f64::abs));
                let mut inner = 1;
                for (argument, map) in edges {
                    inner += map.inner();
                    depth[node] = depth[node].max(depth[*argument]);
                    if let Some(input) = &absolute[*argument] {
                        let read = linear.apply(map, input, true)?;
                        total = Some(match total {
                            Some(sum) => sum + read,
                            None => read,
                        });
                    }
                }
                (total.unwrap_or_else(|| Array2::zeros(linear.trace.values[node].dim())), inner + 2)
            }
        };
        depth[node] += inner;
        absolute[node] = Some(field);
    }
    let reach = absolute[target].as_ref().map_or(0.0, frobenius);
    let chains = 2.0 * accumulation_growth(depth[target] + 4) * reach;
    let items = paths.len() + remainders.len() + 1;
    let summed: f64 = paths.iter().map(TracePath::mass).chain(remainders.iter().map(|r| frobenius(&r.net))).sum::<f64>()
        + frobenius(value);
    Ok(chains + accumulation_growth(items) * summed)
}

/// Bounds on the spectral norm of an exact matrix, read off its computed value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralNormBounds {
    /// `max(0, σ̂₁ − band)`. A positive value certifies a nonzero exact matrix.
    pub lower: f64,
    /// `σ̂₁ + band`. The exact norm is at most this.
    pub upper: f64,
}

/// Bounds on `‖R‖₂` for an exact matrix `R` whose computed value `R̂` (`m × n`)
/// is within `formation` of it in spectral norm, read off the largest
/// eigenvalue of the Gram of `R̂`'s smaller side. No singular value
/// decomposition is formed.
///
/// With `k = min(m, n)`, `t = max(m, n)`, `u` the unit roundoff,
/// `γ_j = j·u/(1 − j·u)` and `η = 2⁻¹⁰⁷⁴`:
///
/// 1. **Scale.** `B = 2^{−e}·R̂`, with `e` taking `max|r̂ᵢⱼ|` to about `[½, 1)`,
///    so the Gram below neither underflows nor overflows. A power of two is
///    exact except where an entry lands subnormal, which moves it by at most
///    `η`, so by Weyl `|σ_max(B) − 2^{−e}‖R̂‖₂| ≤ β = √(mn)·η`.
/// 2. **Gram.** `Ĝ = fl(BᵀB)` (or `fl(BBᵀ)`), `k × k`, with one triangle
///    mirrored, so `Ĝ` is exactly symmetric. Each entry is an inner product of
///    length `t`, so `|Ĝ − G| ≤ γ_t·|B|ᵀ|B|` entrywise, in any summation order and
///    with or without fused multiply-adds (Higham, *ASNA* 2nd ed., §3.5). The
///    majorant is entrywise non-negative and positive semidefinite, so
///    `‖Ĝ − G‖₂ ≤ γ_t·‖|B|ᵀ|B|‖₂ ≤ γ_t·tr(|B|ᵀ|B|) = γ_t·‖B‖²_F`. A diagonal entry
///    sums squares, so `Ĝᵢᵢ ≥ (1 − γ_t)·Gᵢᵢ`, and the trace sums `k`
///    non-negative terms, so `‖B‖²_F ≤ fl(tr Ĝ)/((1 − γ_t)(1 − γ_k))` and
///    `δ = γ_t·fl(tr Ĝ)/((1 − γ_t)(1 − γ_k))` bounds `‖Ĝ − G‖₂`.
/// 3. **Spectrum.** The self-adjoint eigensolver is backward stable: its
///    computed `λ̂` are the exact eigenvalues of `Ĝ + E` with
///    `‖E‖₂ ≤ ρ = k·(ε·max|λ̂| + η)`
///    ([`symmetric_spectrum_rounding_band_at_dim`]; the same convention as the
///    [`gam_linalg::roundoff::factor_singular_band`] a full SVD reads). By Weyl,
///    `σ_max(B)² = λ_max(G) ∈ [λ̂_max − ρ − δ, λ̂_max + ρ + δ]`.
/// 4. **Root.** `√·` is monotone. The endpoints are widened for the handful of
///    rounded operations that form them, each root is taken one ulp outward,
///    `β` is added on each side, and the result is multiplied back by `2^e`
///    (exact, one more ulp outward for a subnormal landing).
/// 5. **Exact matrix.** `|‖R‖₂ − ‖R̂‖₂| ≤ formation`, so
///    `lower = max(0, σ_lo − formation)` and `upper = σ_hi + formation` bracket
///    `‖R‖₂`.
///
/// Since `‖B‖²_F ≤ k·σ_max(B)²`, `δ ≤ γ_t·k·λ_max`, so the bracket on `‖R̂‖₂` is
/// within about `(t + 1)·k·u/2` of it relatively (`2.3e-10` at `t = k = 2048`),
/// where a full SVD's band is `t·ε`. The Gram is one `k × k × t` product at pool parallelism and the spectrum an
/// eigenvalue-only decomposition at the fixed EVD degree, so the result is
/// identical at every pool width.
pub fn spectral_norm_bounds<S: Data<Elem = f64>>(
    governor: &MemoryGovernor,
    matrix: &ArrayBase<S, Ix2>,
    formation: f64,
    context: &'static str,
) -> Result<SpectralNormBounds, PathError> {
    if matrix.is_empty() {
        return Ok(SpectralNormBounds {
            lower: 0.0,
            upper: formation,
        });
    }
    let (rows, cols) = matrix.dim();
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(PathError::NonFiniteMatrix { context });
    }
    let largest = matrix.iter().fold(0.0_f64, |largest, value| largest.max(value.abs()));
    if largest == 0.0 {
        return Ok(SpectralNormBounds {
            lower: 0.0,
            upper: formation,
        });
    }
    let short = rows.min(cols);
    let long = rows.max(cols);
    // The scaled copy, then the Gram and the eigensolver's working copy of it.
    let working = reserve(governor, rows, cols, 1, context)?;
    let gram_reservation = reserve(governor, short, short, 2, context)?;
    // `2^e` overflows for `e > 1023` and `2^{−e}` for a subnormal `e`, so each
    // power is applied as two representable halves.
    let exponent = largest.log2().floor() as i32 + 1;
    let (shrink, shrink_tail) = (
        2.0_f64.powi(-(exponent / 2)),
        2.0_f64.powi(-(exponent - exponent / 2)),
    );
    let scaled = matrix.mapv(|value| value * shrink * shrink_tail);
    let gram = if cols <= rows {
        fast_ata(&scaled)
    } else {
        fast_ata(&scaled.t())
    };
    drop(scaled);
    drop(working);
    let gram_view = FaerArrayView::new(&gram);
    let spectrum = self_adjoint_eigenvalues(gram_view.as_ref(), Side::Lower).map_err(|source| {
        PathError::Eigen {
            context,
            source: FaerLinalgError::SelfAdjointEigen(source),
        }
    })?;
    drop(gram_view);
    let spectrum = spectrum.as_ref().column_vector();
    let eigenvalues: Vec<f64> = (0..short).map(|index| spectrum[index]).collect();
    let trace: f64 = (0..short).map(|index| gram[[index, index]]).sum();
    drop(gram);
    drop(gram_reservation);
    if eigenvalues.iter().any(|value| !value.is_finite()) {
        return Err(PathError::NonFiniteMatrix { context });
    }
    let largest_eigenvalue = eigenvalues
        .iter()
        .fold(f64::NEG_INFINITY, |largest, &value| largest.max(value));
    let gram_formation = accumulation_growth(long) * trace
        / ((1.0 - accumulation_growth(long)) * (1.0 - accumulation_growth(short)));
    let spectrum_band = symmetric_spectrum_rounding_band_at_dim(short, &eigenvalues);
    // The slack and each endpoint take a handful of rounded operations.
    let slack = (spectrum_band + gram_formation) * (1.0 + accumulation_growth(8));
    let widen = 4.0 * UNIT_ROUNDOFF;
    let squared_upper = (largest_eigenvalue + slack) * (1.0 + widen);
    let squared_lower = ((largest_eigenvalue - slack) * (1.0 - widen)).max(0.0);
    let subnormal_shift = ((rows as f64).sqrt() * (cols as f64).sqrt() * f64::from_bits(1)).next_up();
    let scaled_upper = (squared_upper.sqrt().next_up() + subnormal_shift).next_up();
    let scaled_lower = (squared_lower.sqrt().next_down() - subnormal_shift).next_down().max(0.0);
    let (grow, grow_tail) = (
        2.0_f64.powi(exponent / 2),
        2.0_f64.powi(exponent - exponent / 2),
    );
    let sigma_upper = (scaled_upper * grow * grow_tail).next_up();
    let sigma_lower = (scaled_lower * grow * grow_tail).next_down().max(0.0);
    Ok(SpectralNormBounds {
        lower: (sigma_lower - formation).max(0.0),
        upper: sigma_upper + formation,
    })
}

/// Reserves `copies` dense `rows × cols` matrices on `governor` before they are formed.
fn reserve(
    governor: &MemoryGovernor,
    rows: usize,
    cols: usize,
    copies: usize,
    context: &'static str,
) -> Result<MemoryReservation, PathError> {
    governor
        .try_reserve_dense_f64_copies(rows, cols, copies, context)
        .map_err(|source| PathError::Memory { context, source })
}

#[cfg(test)]
#[path = "paths_tests.rs"]
mod tests;

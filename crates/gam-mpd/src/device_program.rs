//! An operator program executed on a device, values resident (#2951).
//!
//! [`DeviceProgram`] lowers a program's nodes onto `gam_gpu::tensor`'s operations once, keeps its
//! operators on the device, and then runs forward passes ([`DeviceProgram::forward`]), the KL of
//! the logits against a target with its cotangent ([`DeviceProgram::kl`]), sampled-label
//! cotangents ([`DeviceProgram::sampled`]), reverse passes ([`DeviceProgram::vjp`]) and tangents
//! in the operators' reals ([`DeviceProgram::jvp`], [`DeviceProgram::quadratic`]) with every node
//! value staying on the device: only what a caller asks for comes back.
//!
//! The rules are the CPU's (`OperatorProgram::execute`, `derivatives::{vjp, jvp}`) term for term:
//! an affine node adds its terms in order and then its bias; attention rotates its queries and
//! keys by the CPU's own angles (`Rotary::turn`, tabulated on the host), scores `c q·k`, takes the
//! max-shifted softmax over the keys at or before each position and reads the values; a norm is
//! `x / √(mean(x²) + ε)`. Each sequence is one block of rows, so the attention of every sequence
//! runs as one strided-batched product: a family of many sequences fills the device in one launch.
//! On the host backend the same lowering runs on the CPU; that is how it is tested everywhere.
//!
//! The logits are never formed whole. The program's output must be a linear head on a hidden
//! node (`h A` for a tied unembedding read transposed, or `h Aᵀ`, then an indicator readout), and
//! the head runs in row tiles of at most [`TILE_BYTES`]: each tile's logits, its KL against the
//! target's tile and its cotangent, pulled straight back to the hidden node.
//!
//! What is lowered: features of an indicator basis read only through affine terms (a gather of the
//! operator's columns), raw slots, constants, affine nodes over identity, diagonal, dense and
//! low-rank operators, pointwise laws, Hadamard products, RMS norms, attention, and the head.
//! Any other node refuses the whole program with its reason; its caller then runs on the CPU.

use super::device_heads::{self, Buffer, Heads, Stacked};
use super::operator_program::{Basis, FamilyInputs, Law, Node, Operator, OperatorBody, OperatorProgram, Rotary, SlotValues};
use gam_gpu::tensor::{Arithmetic, Device, Indices, Op, PointwiseLaw, Tensor};
use ndarray::{Array1, Array2};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, OnceLock};

/// The largest logits tile the head forms at once, in bytes.
pub const TILE_BYTES: usize = 1 << 30;

fn error(e: impl std::fmt::Display) -> String {
    format!("device: {e}")
}

/// The roles an operator is held on the device in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Role {
    /// Products `x Aᵀ` (an affine term), `g A` (its cotangent), or `h A` (a transposed read).
    Product,
    /// Its transpose as a table of rows, gathered by token (an affine term on a one-hot feature).
    Table,
    /// Its single column as a row (a bias or a constant).
    Column,
}

/// An operator on the device in one role.
enum Held {
    Identity,
    Diagonal(Tensor),
    Dense(Tensor),
    /// `left` (rows × r) and `right` (r × cols).
    LowRank(Tensor, Tensor),
    Table(Tensor),
    Column(Tensor),
}

struct HeldOperator {
    source: Arc<Operator>,
    /// False after a resident training update: pointer identity with the original
    /// host operator must no longer license sharing these changed values.
    source_matches: bool,
    /// Shared with every program compiled from this one ([`DeviceProgram::compile_sharing`]).
    held: Arc<Held>,
}

fn hold(device: &Device, op: &Operator, role: Role) -> Result<Held, String> {
    Ok(match role {
        Role::Column => Held::Column(device.upload(op.matrix().t().view()).map_err(error)?),
        Role::Table => Held::Table(device.upload(op.matrix_cow().t()).map_err(error)?),
        Role::Product => match &op.body {
            OperatorBody::Identity => Held::Identity,
            OperatorBody::Diagonal { values, .. } => Held::Diagonal(device.upload_vec(1, values.len(), values.to_vec()).map_err(error)?),
            OperatorBody::LowRank { left, right, .. } => Held::LowRank(device.upload(left.view()).map_err(error)?, device.upload(right.view()).map_err(error)?),
            OperatorBody::Dense { values, .. } => match op.diagonal() {
                Some(d) => Held::Diagonal(device.upload_vec(1, d.len(), d.to_vec()).map_err(error)?),
                None => Held::Dense(device.upload(values.view()).map_err(error)?),
            },
        },
    })
}

/// How one node executes.
enum Step {
    /// A one-hot feature of an indicator basis: only its token ids (its readers gather).
    Feature {
        slot: usize,
    },
    Raw {
        slot: usize,
    },
    Constant {
        operator: usize,
    },
    Affine {
        terms: Vec<(usize, usize)>,
        bias: Option<usize>,
    },
    Pointwise {
        input: usize,
        codes: Indices,
    },
    /// Fixed scalar expression from the serialized rule, evaluated once at compilation.
    Gain { input: usize, factor: f64 },
    Hadamard {
        left: usize,
        right: usize,
    },
    RmsNorm {
        input: usize,
        epsilon: f64,
    },
    Attend {
        query: usize,
        key: usize,
        value: usize,
        scale: f64,
        rotary: Option<Rotary>,
        causal: bool,
    },
    Transposed {
        input: usize,
        operator: usize,
    },
    Concat { parts: Vec<usize> },
    /// An indicator readout is an exact value copy in resident-value mode.
    Readout { input: usize },
    /// The head's logits or their readout: never formed whole.
    Head,
}

/// The output's linear head: logits `h A` (`transposed`) or `h Aᵀ` from the hidden node `h`.
#[derive(Clone, Copy, Debug)]
struct Head {
    hidden: usize,
    /// None for ordinary resident-value execution, with hidden = output.
    operator: Option<usize>,
    transposed: bool,
    classes: usize,
}

/// The constant of the tanh GELU as the CPU forms it (`operator_program`'s `gelu_tanh`).
fn gelu_tanh_constant() -> f64 {
    std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2
}

fn law_of(law: Law) -> PointwiseLaw {
    match law {
        Law::Relu => PointwiseLaw::Relu,
        Law::Identity => PointwiseLaw::Identity,
        Law::Zero => PointwiseLaw::Zero,
        Law::Silu => PointwiseLaw::Silu,
        Law::Gelu => PointwiseLaw::Gelu,
        Law::GeluTanh => PointwiseLaw::GeluTanh,
    }
}

/// The nodes a step reads.
fn step_arguments(step: &Step) -> Vec<usize> {
    match step {
        Step::Feature { .. } | Step::Raw { .. } | Step::Constant { .. } | Step::Head => Vec::new(),
        Step::Affine { terms, .. } => terms.iter().map(|t| t.0).collect(),
        Step::Pointwise { input, .. } | Step::Gain { input, .. } | Step::RmsNorm { input, .. } | Step::Readout { input } | Step::Transposed { input, .. } => vec![*input],
        Step::Hadamard { left, right } => vec![*left, *right],
        Step::Attend { query, key, value, .. } => vec![*query, *key, *value],
        Step::Concat { parts } => parts.clone(),
    }
}

/// A program lowered onto a device (module note).
pub struct DeviceProgram {
    device: Device,
    steps: Vec<Step>,
    /// Each node's width.
    widths: Vec<usize>,
    head: Head,
    edited_head_nodes: BTreeMap<usize, Option<usize>>,
    operators: BTreeMap<(usize, Role), HeldOperator>,
    batch: Mutex<Option<Arc<PreparedBatch>>>,
    /// The arithmetic of every product in the forward pass and the head (float64 by default; a
    /// training step's proposals may run in TF32, its accepted point is scored again in float64).
    arithmetic: Arithmetic,
    /// Sibling attention heads run as one computation ([`device_heads`]).
    fused: Vec<Fused>,
    /// Per node, the group it is a member or the output of.
    grouped: Vec<Option<usize>>,
    /// Per operator, how many times its device copy was replaced ([`Frozen`] values stay valid while
    /// no operator they read changes).
    revisions: BTreeMap<usize, u64>,
}

/// A group of sibling heads and its stacked operators.
struct Fused {
    heads: Heads,
    stacked: Arc<Stacked>,
    /// The host operators the stacked copies were made from, in order (shared by identity with the
    /// programs compiled from this one).
    sources: Vec<usize>,
    /// False once a training step replaces one of its operators: it then runs node by node.
    live: bool,
}

/// The identities of the host operators `heads` stacks, in order.
fn sources(program: &OperatorProgram, heads: &Heads) -> Vec<usize> {
    let id = |op: usize| Arc::as_ptr(&program.operators[op]) as usize;
    heads
        .projection_operators
        .iter()
        .flat_map(|(op, bias)| [id(*op), bias.map_or(0, id)])
        .chain(heads.attends.iter().map(|(_, op)| id(*op)))
        .collect()
}

struct PreparedBatch {
    rows: usize,
    layout: Option<super::operator_program::SequenceLayout>,
    tokens: BTreeMap<usize, Vec<u32>>,
    ids: BTreeMap<usize, Arc<Indices>>,
    blocks: usize,
    rotations: Arc<Vec<(Rotary, Tensor, Tensor)>>,
}

/// A fused group's buffers in a trace ([`device_heads`]): `P`, `N` and `G` when its queries and
/// keys are normed, and `A`.
#[derive(Clone, Copy)]
struct Buffers {
    projections: usize,
    normed: Option<(usize, usize)>,
    reads: usize,
}

/// A node's value in a trace.
enum Slot {
    /// No value: a feature, the streamed head, or a node not computed yet.
    Empty,
    Value(Tensor),
    /// Another node's value (an indicator readout of it).
    Alias(usize),
    /// Columns `start..start + width` of the trace's buffer `buffer` (a fused group's), copied out
    /// when first read.
    Columns { buffer: usize, start: usize, width: usize, copy: OnceLock<Tensor> },
    /// A value computed once for every forward on a family ([`Frozen`]).
    Shared(Arc<Tensor>),
}

/// One forward pass's node values on the device (none for features and the head), with the
/// family's blocks and rotation tables.
pub struct DeviceTrace {
    slots: Vec<Slot>,
    /// The fused groups' buffers, and per group which are its own when it ran fused.
    buffers: Vec<Tensor>,
    fused: Vec<Option<Buffers>>,
    /// Per node, whether its value came from a [`Frozen`] (a reverse pass leaves those alone).
    frozen: Option<Arc<Vec<bool>>>,
    device: Device,
    pub rows: usize,
    ids: BTreeMap<usize, Arc<Indices>>,
    /// Sequences (equal row blocks) and their length.
    blocks: usize,
    /// Per rotary configuration, `(cos, sin)` per row and plane.
    rotations: Arc<Vec<(Rotary, Tensor, Tensor)>>,
}

impl DeviceTrace {
    /// Node `n`'s value.
    pub fn value(&self, n: usize) -> Result<&Tensor, String> {
        match self.slots.get(n) {
            Some(Slot::Value(t)) => Ok(t),
            Some(Slot::Shared(t)) => Ok(t),
            Some(Slot::Alias(m)) => self.value(*m),
            Some(Slot::Columns { buffer, start, width, copy }) => match copy.get() {
                Some(t) => Ok(t),
                None => {
                    let t = self.device.columns_of(&self.buffers[*buffer], *start..*start + *width).map_err(error)?;
                    Ok(copy.get_or_init(|| t))
                }
            },
            _ => Err(format!("device: node {n} has no resident value")),
        }
    }

    /// Whether node `n` has a value.
    #[must_use]
    pub fn has(&self, n: usize) -> bool {
        !matches!(self.slots.get(n), None | Some(Slot::Empty))
    }

    /// The program's node count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Node `n`'s value to change in place (made the node's own first when it is another node's
    /// value or a block of a fused buffer).
    pub fn value_mut(&mut self, n: usize) -> Result<&mut Tensor, String> {
        // Nodes reading this value as their own keep it as it is now.
        for reader in 0..self.slots.len() {
            if matches!(self.slots[reader], Slot::Alias(m) if m == n) {
                let own = self.device.copy(self.value(n)?).map_err(error)?;
                self.slots[reader] = Slot::Value(own);
            }
        }
        if !matches!(self.slots.get(n), Some(Slot::Value(_))) {
            let own = self.device.copy(self.value(n)?).map_err(error)?;
            self.slots[n] = Slot::Value(own);
        }
        match self.slots.get_mut(n) {
            Some(Slot::Value(t)) => Ok(t),
            _ => Err(format!("device: node {n} has no resident value")),
        }
    }

    /// Node `n`'s value moved out of the trace, or a copy of it when another node reads it as its
    /// own or it is a block of a fused buffer.
    pub fn take(&mut self, n: usize) -> Result<Tensor, String> {
        let aliased = self.slots.iter().any(|s| matches!(s, Slot::Alias(m) if *m == n));
        match self.slots.get_mut(n) {
            Some(slot @ Slot::Value(_)) if !aliased => match std::mem::replace(slot, Slot::Empty) {
                Slot::Value(t) => Ok(t),
                _ => Err(format!("device: node {n} has no resident value")),
            },
            _ => self.device.copy(self.value(n)?).map_err(error),
        }
    }
}

/// `out`, zeros of `rows × width` first when it is empty.
fn ensure<'a>(d: &Device, out: &'a mut Option<Tensor>, rows: usize, width: usize) -> Result<&'a mut Tensor, String> {
    if out.is_none() {
        *out = Some(d.zeros(rows, width).map_err(error)?);
    }
    out.as_mut().ok_or_else(|| "device: tangent slot".to_string())
}

/// The hooks a forward pass offers node values: `before` at the nodes of its set, `edit` at every
/// node or none.
#[derive(Clone, Copy)]
struct Hooks<'a> {
    before: Option<&'a BTreeSet<usize>>,
    edit: bool,
}

impl Hooks<'_> {
    fn before(&self, node: usize) -> bool {
        self.before.is_some_and(|nodes| nodes.contains(&node))
    }
}

/// The values of every node no trainable operator reaches, on one family: computed once by
/// [`DeviceProgram::freeze`] and shared by every [`DeviceProgram::forward_frozen`] on that family,
/// which computes only the other nodes. Only the frozen nodes the others read, and the ones asked
/// for, keep their values.
pub struct Frozen {
    batch: Arc<PreparedBatch>,
    frozen: Arc<Vec<bool>>,
    values: BTreeMap<usize, Arc<Tensor>>,
    trainable: BTreeSet<usize>,
    /// The program's operator revisions the values were computed at.
    revisions: BTreeMap<usize, u64>,
}

impl Frozen {
    /// Whether node `n` is frozen.
    #[must_use]
    pub fn is_frozen(&self, n: usize) -> bool {
        self.frozen.get(n).copied().unwrap_or(false)
    }

    /// The bytes of the values it keeps.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.values.values().map(|t| t.bytes()).sum()
    }
}

/// How a forward pass uses frozen values: not at all, computing only the frozen nodes, or taking
/// them from a [`Frozen`].
#[derive(Clone, Copy)]
enum Reuse<'a> {
    Nothing,
    Freeze(&'a [bool]),
    From(&'a Frozen),
}

/// The nodes a forward pass runs: those after `entry`'s node, which takes the given value (none
/// before it runs), up to `end`.
struct Span {
    entry: Option<(usize, Tensor)>,
    end: usize,
}

impl Span {
    /// Every node.
    fn all() -> Self {
        Self { entry: None, end: usize::MAX }
    }
}

/// The rotation tables of `rotary` among a trace's, with its pairing.
fn turn(rotations: &[(Rotary, Tensor, Tensor)], rotary: Option<Rotary>) -> Result<device_heads::Turn<'_>, String> {
    rotary
        .map(|r| rotations.iter().find(|(x, _, _)| *x == r).map(|(_, c, s)| (c, s, r.half_split)).ok_or_else(|| "device: no rotation table".to_string()))
        .transpose()
}

fn rotations_of(trace: &DeviceTrace, rotary: Rotary) -> Result<(&Tensor, &Tensor), String> {
    trace.rotations.iter().find(|(r, _, _)| *r == rotary).map(|(_, c, s)| (c, s)).ok_or_else(|| "device: no rotation table".to_string())
}

impl DeviceProgram {
    /// Lower `program` onto `device`, or the reason it cannot be.
    pub fn compile(device: &Device, program: &OperatorProgram) -> Result<Self, String> {
        Self::lower(device, program, None, false, None)
    }

    /// [`Self::compile`] on `from`'s device, every operator `program` shares with `from` (the same
    /// `Arc`, in the same role) held by both instead of uploaded again: the programs of one model
    /// with different sites replaced hold the model's weights once.
    pub fn compile_sharing(from: &Self, program: &OperatorProgram) -> Result<Self, String> {
        Self::lower(&from.device, program, Some(from), false, None)
    }

    /// Materialize the actual output expression as ordinary resident values.
    /// No streamed head, synthetic operator, or extra output arithmetic is added.
    /// Head scoring and derivative helpers explicitly refuse this mode.
    pub fn compile_values(device: &Device, program: &OperatorProgram) -> Result<Self, String> {
        Self::lower(device, program, None, true, None)
    }

    /// Value execution sharing exact operator Arcs in the same executable roles.
    pub fn compile_values_sharing(from: &Self, program: &OperatorProgram) -> Result<Self, String> {
        Self::lower(&from.device, program, Some(from), true, None)
    }

    /// Bound retained numeric operator buffers before uploading any operator.
    /// Excludes node indices, activations, workspaces, allocator overhead and host values.
    pub fn compile_values_bounded(device: &Device, program: &OperatorProgram, numeric_bytes_limit: usize) -> Result<Self, String> {
        if numeric_bytes_limit == 0 { return Err("positive operator numeric byte limit required".into()); }
        Self::lower(device, program, None, true, Some(numeric_bytes_limit))
    }

    /// Shared resident-value compilation with the same retained numeric-buffer
    /// preflight as compile_values_bounded. Shared buffers still count toward
    /// this program's declared retained bound; this is not an incremental limit.
    pub fn compile_values_sharing_bounded(from: &Self, program: &OperatorProgram, numeric_bytes_limit: usize) -> Result<Self, String> {
        if numeric_bytes_limit == 0 { return Err("positive operator numeric byte limit required".into()); }
        Self::lower(&from.device, program, Some(from), true, Some(numeric_bytes_limit))
    }

    /// Bytes of unique retained float64 operator buffers; excludes all other storage.
    pub fn operator_numeric_bytes(&self) -> Result<usize, String> {
        let mut seen = std::collections::BTreeSet::new();
        let mut total = 0usize;
        for held in self.operators.values() {
            if !seen.insert(Arc::as_ptr(&held.held) as usize) { continue; }
            let count = match &*held.held {
                Held::Identity => 0,
                Held::Diagonal(t) | Held::Dense(t) | Held::Table(t) | Held::Column(t) => t.len(),
                Held::LowRank(a, b) => a.len().checked_add(b.len()).ok_or("operator size overflow")?,
            };
            total = total.checked_add(count.checked_mul(8).ok_or("operator byte overflow")?).ok_or("operator byte overflow")?;
        }
        for group in &self.fused {
            if seen.insert(Arc::as_ptr(&group.stacked) as usize) {
                total = total.checked_add(group.stacked.len().checked_mul(8).ok_or("operator byte overflow")?).ok_or("operator byte overflow")?;
            }
        }
        Ok(total)
    }

    fn lower(device: &Device, program: &OperatorProgram, from: Option<&Self>, values: bool, numeric_bytes_limit: Option<usize>) -> Result<Self, String> {
        let interfaces = program.interfaces().map_err(|e| e.to_string())?;
        let widths: Vec<usize> = interfaces.iter().map(|i| i.width()).collect();
        let head = if values {
            Head { hidden: program.output, operator: None, transposed: false, classes: widths[program.output] }
        } else { Self::head_of(program)? };
        let head_nodes: Vec<usize> = if values { Vec::new() } else { match &program.nodes[program.output] {
            Node::Readout { input, .. } => vec![program.output, *input],
            _ => vec![program.output],
        }};
        let edited_head_nodes = head_nodes
            .iter()
            .map(|&node| {
                (
                    node,
                    match &program.nodes[node] {
                        Node::Readout { input, .. } => Some(*input),
                        _ => None,
                    },
                )
            })
            .collect();
        // Which nodes read each node: a feature may only be read by affine terms.
        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); program.nodes.len()];
        for (index, node) in program.nodes.iter().enumerate() {
            for argument in node.arguments() {
                readers[argument].push(index);
            }
        }
        let mut steps = Vec::with_capacity(program.nodes.len());
        let mut wanted: Vec<(usize, Role)> = head.operator.into_iter().map(|op| (op, Role::Product)).collect();
        for (index, node) in program.nodes.iter().enumerate() {
            if head_nodes.contains(&index) {
                steps.push(Step::Head);
                continue;
            }
            let refuse = |what: &str| Err(format!("node {index}: {what} has no device rule"));
            let step = match node {
                Node::Feature { slot, basis } => {
                    if !matches!(program.bases[*basis], Basis::Indicator { .. }) {
                        return refuse("a feature of a character basis");
                    }
                    if readers[index].iter().any(|r| !matches!(program.nodes[*r], Node::Affine { .. })) {
                        return refuse("a feature read outside an affine term");
                    }
                    Step::Feature { slot: *slot }
                }
                Node::Raw { slot } => Step::Raw { slot: *slot },
                Node::Constant { operator } => {
                    wanted.push((*operator, Role::Column));
                    Step::Constant { operator: *operator }
                }
                Node::Affine { terms, bias } => {
                    for (argument, operator) in terms {
                        let feature = matches!(program.nodes[*argument], Node::Feature { .. });
                        wanted.push((*operator, if feature { Role::Table } else { Role::Product }));
                    }
                    if let Some(b) = bias {
                        wanted.push((*b, Role::Column));
                    }
                    Step::Affine { terms: terms.clone(), bias: *bias }
                }
                Node::Pointwise { input, laws } => {
                    let interface = &interfaces[*input];
                    let mut codes = vec![PointwiseLaw::Identity.code(); widths[*input]];
                    for (group, law) in laws.iter().enumerate() {
                        for c in interface.range(group) {
                            codes[c] = law_of(*law).code();
                        }
                    }
                    Step::Pointwise { input: *input, codes: device.upload_indices(&codes).map_err(error)? }
                }
                Node::Hadamard { left, right } => Step::Hadamard { left: *left, right: *right },
                Node::RmsNorm { input, epsilon } => Step::RmsNorm { input: *input, epsilon: *epsilon },
                Node::Attend { query, key, value, scale, rotary, causal } => {
                    if let Some(r) = rotary
                        && 2 * r.pairs().len() > widths[*query]
                    {
                        return refuse("a rotation wider than its head");
                    }
                    Step::Attend { query: *query, key: *key, value: *value, scale: scale.value(), rotary: *rotary, causal: *causal }
                }
                Node::Transposed { input, operator } => {
                    wanted.push((*operator, Role::Product));
                    Step::Transposed { input: *input, operator: *operator }
                }
                Node::Readout { input, basis } if values && matches!(program.bases[*basis], Basis::Indicator { .. }) => Step::Readout { input: *input },
                Node::Readout { .. } => return refuse("a readout before the output"),
                Node::Bilinear { .. } => return refuse("a bilinear node"),
                Node::Softmax { .. } => return refuse("a softmax node"),
                Node::Mix { .. } => return refuse("a mix node"),
                Node::Outer { .. } => return refuse("an outer product"),
                Node::Concat { parts } => Step::Concat { parts: parts.clone() },
                Node::Param { .. } | Node::Call { .. } => return refuse("a rule"),
                Node::Gain { input, coefficient } => {
                    // Parameters supplied at execution time have no resident representation.
                    // Constant polynomial expressions retain their ordinary IR semantics.
                    let (factor, _, _) = coefficient.evaluate(&[]).map_err(error)?;
                    if !factor.is_finite() { return refuse("a nonfinite fixed gain"); }
                    Step::Gain { input: *input, factor }
                }
            };
            steps.push(step);
        }
        let groups = device_heads::find(program, &readers, &widths, &head_nodes);
        // `from`'s operators by their source and role.
        let held_by: BTreeMap<(usize, Role), &Arc<Held>> =
            from.map(|f| f.operators.iter().filter(|(_, h)| h.source_matches)
                .map(|((_, role), h)| ((Arc::as_ptr(&h.source) as usize, *role), &h.held)).collect()).unwrap_or_default();
        if let Some(limit) = numeric_bytes_limit {
            let mut total = 0usize;
            for &(index, role) in &wanted.iter().copied().collect::<std::collections::BTreeSet<_>>() {
                let operator = &program.operators[index];
                let count = match role {
                    Role::Table | Role::Column => operator.rows.width().checked_mul(operator.cols.width()).ok_or("operator size overflow")?,
                    Role::Product => match &operator.body {
                        OperatorBody::Identity => 0,
                        OperatorBody::Diagonal { values, .. } => values.len(),
                        OperatorBody::LowRank { left, right, .. } => left.len().checked_add(right.len()).ok_or("operator size overflow")?,
                        OperatorBody::Dense { values, .. } => operator.diagonal().map_or(values.len(), |d| d.len()),
                    },
                };
                total = total.checked_add(count.checked_mul(8).ok_or("operator byte overflow")?).ok_or("operator byte overflow")?;
            }
            for heads in &groups {
                let (input, output) = (widths[heads.input], widths[heads.output]);
                let biased = heads.projection_operators.iter().any(|(_, b)| b.is_some());
                let gains = heads.norms.as_ref().map_or(0, |_| heads.normed_columns());
                let count = heads.columns().checked_mul(input + usize::from(biased)).and_then(|n| n.checked_add(output.checked_mul(heads.heads * heads.width)?)?.checked_add(gains)).ok_or("operator size overflow")?;
                total = total.checked_add(count.checked_mul(8).ok_or("operator byte overflow")?).ok_or("operator byte overflow")?;
            }
            if total > limit { return Err(format!("operator numeric buffers {total} exceed declared source limit {limit}; excludes indices/activations/workspaces/allocator/host")); }
        }
        let mut operators = BTreeMap::new();
        for key in wanted {
            if operators.contains_key(&key) {
                continue;
            }
            let source = Arc::clone(&program.operators[key.0]);
            let held = match held_by.get(&(Arc::as_ptr(&source) as usize, key.1)) {
                Some(h) => Arc::clone(h),
                None => Arc::new(hold(device, &source, key.1)?),
            };
            operators.insert(key, HeldOperator { source, source_matches: true, held });
        }
        let mut grouped = vec![None; program.nodes.len()];
        let mut fused = Vec::with_capacity(groups.len());
        for heads in groups {
            for node in heads.members().chain([heads.output]) {
                grouped[node] = Some(fused.len());
            }
            let sources = sources(program, &heads);
            let stacked = match from.and_then(|f| f.fused.iter().find(|g| g.live && g.sources == sources)) {
                Some(shared) => Arc::clone(&shared.stacked),
                None => Arc::new(Stacked::upload(device, program, &heads).map_err(error)?),
            };
            fused.push(Fused { heads, stacked, sources, live: true });
        }
        Ok(Self { device: device.clone(), steps, widths, head, edited_head_nodes, operators, batch: Mutex::new(None), arithmetic: Arithmetic::F64, fused, grouped, revisions: BTreeMap::new() })
    }

    /// Operator `op`'s device copy changed: every group reading it runs node by node from now on
    /// (its stacked copy is stale), and frozen values reading it are stale.
    fn dissolve(&mut self, op: usize) {
        *self.revisions.entry(op).or_default() += 1;
        for group in &mut self.fused {
            if group.heads.operators().contains(&op) {
                group.live = false;
            }
        }
    }

    /// The number of sibling-head groups that run as one computation.
    #[must_use]
    pub fn fused_groups(&self) -> usize {
        self.fused.iter().filter(|g| g.live).count()
    }

    /// Run every group of sibling heads node by node from now on (the same values, rounded in
    /// another order; for timing and parity against the fused computation).
    pub fn unfuse(&mut self) {
        for group in &mut self.fused {
            group.live = false;
        }
    }

    fn head_of(program: &OperatorProgram) -> Result<Head, String> {
        let logits = match &program.nodes[program.output] {
            Node::Readout { input, basis } => {
                if !matches!(program.bases[*basis], Basis::Indicator { .. }) {
                    return Err("the output reads a character basis".to_string());
                }
                *input
            }
            _ => program.output,
        };
        let (hidden, operator, transposed) = match &program.nodes[logits] {
            Node::Transposed { input, operator } => (*input, *operator, true),
            Node::Affine { terms, bias: None } if terms.len() == 1 => (terms[0].0, terms[0].1, false),
            _ => return Err("the output is not a linear head on a hidden node".to_string()),
        };
        if !matches!(program.operators[operator].body, OperatorBody::Dense { .. }) || program.operators[operator].diagonal().is_some() {
            return Err("the head's operator is not a dense matrix".to_string());
        }
        let a = program.operators[operator].matrix_cow();
        let classes = if transposed { a.ncols() } else { a.nrows() };
        Ok(Head { hidden, operator: Some(operator), transposed, classes })
    }

    fn linear_operator(&self) -> Result<usize, String> {
        self.head.operator.ok_or_else(|| "device: resident-value mode has no linear head; head scoring and derivatives are unsupported".into())
    }

    /// The device it runs on.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The node the linear head reads, or the output in resident-value mode.
    #[must_use]
    pub fn hidden(&self) -> usize {
        self.head.hidden
    }

    /// The number of head classes, or the output width in resident-value mode.
    #[must_use]
    pub fn classes(&self) -> usize {
        self.head.classes
    }

    /// Every node's width.
    #[must_use]
    pub fn widths(&self) -> &[usize] {
        &self.widths
    }

    /// A dense operator's device copy (an affine term's `A` of `x Aᵀ`).
    pub fn dense(&self, op: usize) -> Result<&Tensor, String> {
        match self.held(op, Role::Product)? {
            Held::Dense(a) => Ok(a),
            _ => Err(format!("device: operator {op} is not held dense")),
        }
    }

    /// A dense operator's device copy to change in place (a trained library); the program's
    /// host operator no longer describes it until [`Self::refresh`] from a program holding it.
    pub fn dense_mut(&mut self, op: usize) -> Result<&mut Tensor, String> {
        if self.operators.contains_key(&(op, Role::Column)) {
            return Err(format!("device: operator {op} also has column uses; use replace_dense_parameter"));
        }
        self.dissolve(op);
        let held = self.operators.get_mut(&(op, Role::Product))
            .ok_or_else(|| format!("device: operator {op} is not held dense"))?;
        match Arc::get_mut(&mut held.held) {
            Some(Held::Dense(a)) => { held.source_matches = false; Ok(a) },
            None => Err(format!("device: operator {op} is shared with another program")),
            _ => Err(format!("device: operator {op} is not held dense")),
        }
    }

    /// Prepare explicitly selected Dense literal parameters for resident fitting.
    /// Dense zero/diagonal fast paths are expanded once, since training may move
    /// off-diagonal entries. Bias columns gain a canonical rows-by-one copy.
    /// This initial preparation may upload the original literal matrix; subsequent
    /// replacements transfer no numerical parameters through the host.
    /// Callers must budget canonical storage, column copies and optimizer state.
    pub fn prepare_dense_parameters(&mut self, trainable: &[usize]) -> Result<(), String> {
        let requested: std::collections::BTreeSet<_> = trainable.iter().copied().collect();
        if requested.len() != trainable.len() { return Err("device: duplicate trainable operator".into()); }
        for &op in trainable { self.trainable_dense_source(op)?; }
        for &op in trainable {
            let source = self.trainable_dense_source(op)?;
            let value = if let Some(held) = self.operators.get(&(op, Role::Product)) {
                match held.held.as_ref() {
                    Held::Dense(value) => self.device.copy(value).map_err(error)?,
                    // Only a Dense literal's immutable diagonal fast path reaches
                    // here. Later replacements always materialize Held::Dense.
                    Held::Diagonal(_) => self.device.upload(source.matrix_cow().view()).map_err(error)?,
                    _ => return Err("device: unsupported trainable dense storage".into()),
                }
            } else {
                self.device.copy(self.column(op)?).map_err(error)?
                    .reshape(source.rows.width(), 1).map_err(error)?
            };
            self.replace_dense_parameter(op, value)?;
        }
        Ok(())
    }

    fn trainable_dense_source(&self, op: usize) -> Result<Arc<Operator>, String> {
        if self.operators.contains_key(&(op, Role::Table)) {
            return Err("device: trainable table roles are unsupported".into());
        }
        let held = self.operators.get(&(op, Role::Product))
            .or_else(|| self.operators.get(&(op, Role::Column)))
            .ok_or("device: unknown trainable operator")?;
        if !matches!(held.source.body, OperatorBody::Dense { .. }) {
            return Err("device: trainable parameter must be a Dense literal".into());
        }
        Ok(Arc::clone(&held.source))
    }

    /// Canonical resident parameter after [`Self::prepare_dense_parameters`].
    pub fn dense_parameter(&self, op: usize) -> Result<&Tensor, String> { self.dense(op) }

    /// Replace a canonical Dense parameter and every execution role coherently.
    /// `value` must be on this program's device. Column uses receive an exact
    /// resident copy reshaped as one row; transpose uses read the canonical copy.
    /// Programs sharing the old buffers retain their previous parameter values.
    /// The host program remains unchanged; export the final canonical values before
    /// encoding or pricing a fitted candidate. No floating-point rounding occurs here.
    pub fn replace_dense_parameter(&mut self, op: usize, value: Tensor) -> Result<(), String> {
        let source = self.trainable_dense_source(op)?;
        if value.dim() != (source.rows.width(), source.cols.width()) {
            return Err("device: replacement parameter shape mismatch".into());
        }
        self.dissolve(op);
        let column = if self.operators.contains_key(&(op, Role::Column)) {
            if value.cols() != 1 { return Err("device: column parameter is not one column".into()); }
            Some(self.device.copy(&value).map_err(error)?.reshape(1, value.rows()).map_err(error)?)
        } else { None };
        self.operators.insert((op, Role::Product), HeldOperator { source: Arc::clone(&source), source_matches: false, held: Arc::new(Held::Dense(value)) });
        if let Some(column) = column {
            self.operators.insert((op, Role::Column), HeldOperator { source, source_matches: false, held: Arc::new(Held::Column(column)) });
        }
        Ok(())
    }

    /// The logits of every row, on the device (a target).
    pub fn logits_on_device(&self, trace: &DeviceTrace) -> Result<Tensor, String> {
        self.linear_operator()?;
        let hidden = trace.value(self.head.hidden)?;
        let mut out = self.device.zeros(trace.rows, self.head.classes).map_err(error)?;
        let tile = self.tile_rows();
        for start in (0..trace.rows).step_by(tile) {
            let n = tile.min(trace.rows - start);
            let logits = self.logits_tile(hidden, start, n, self.arithmetic)?;
            self.device.set_rows(&mut out, start, &logits).map_err(error)?;
        }
        Ok(out)
    }

    /// Materialize only the requested rows of the actual compiled dense head.
    /// The caller owns its tile budget; no normalization or precision change is added.
    pub fn logits_rows(&self, trace: &DeviceTrace, start: usize, rows: usize) -> Result<Tensor, String> {
        if rows == 0 || start.checked_add(rows).is_none_or(|end| end > trace.rows) {
            return Err("device: streamed logits row range outside trace".into());
        }
        self.logits_tile(trace.value(self.head.hidden)?, start, rows, self.arithmetic)
    }

    /// The arithmetic of the forward pass's and the head's products.
    #[must_use]
    pub fn arithmetic(&self) -> Arithmetic {
        self.arithmetic
    }

    /// Run the forward pass's and the head's products in `arithmetic` from now on.
    pub fn set_arithmetic(&mut self, arithmetic: Arithmetic) {
        self.arithmetic = arithmetic;
    }

    /// Re-upload every operator `program` now holds a different copy of (a stepped library).
    pub fn refresh(&mut self, program: &OperatorProgram) -> Result<(), String> {
        let mut changed = BTreeSet::new();
        for ((op, role), held) in &mut self.operators {
            if !held.source_matches || !Arc::ptr_eq(&held.source, &program.operators[*op]) {
                held.source = Arc::clone(&program.operators[*op]);
                held.held = Arc::new(hold(&self.device, &held.source, *role)?);
                held.source_matches = true;
                changed.insert(*op);
            }
        }
        for op in changed {
            self.dissolve(op);
        }
        Ok(())
    }

    fn held(&self, op: usize, role: Role) -> Result<&Held, String> {
        self.operators.get(&(op, role)).map(|h| h.held.as_ref()).ok_or_else(|| format!("device: operator {op} not held as {role:?}"))
    }

    /// The bytes a forward pass keeps per row (every resident node value), for sizing batches.
    #[must_use]
    pub fn bytes_per_row(&self) -> usize {
        self.steps.iter().zip(&self.widths).filter(|(s, _)| !matches!(s, Step::Feature { .. } | Step::Head)).map(|(_, w)| w * 8).sum()
    }

    /// The family's blocks and rotation tables: every sequence one contiguous block of equal
    /// length, positions strictly increasing within it.
    fn prepared_batch(&self, family: &FamilyInputs) -> Result<Arc<PreparedBatch>, String> {
        let mut cached = self.batch.lock().map_err(|_| "device: poisoned batch cache".to_string())?;
        if let Some(saved) = cached.as_ref() {
            let same_layout = match (&saved.layout, &family.layout) {
                (None, None) => true,
                (Some(a), Some(b)) => a.sequence == b.sequence && a.position == b.position,
                _ => false,
            };
            if saved.rows == family.rows
                && same_layout
                && saved.tokens.iter().all(|(slot, tokens)| matches!(family.slots.get(*slot), Some(SlotValues::Tokens(current)) if current == tokens))
            {
                return Ok(Arc::clone(saved));
            }
        }
        let (blocks, rotations) = self.layout(family)?;
        let mut tokens = BTreeMap::new();
        let mut ids = BTreeMap::new();
        for step in &self.steps {
            if let Step::Feature { slot } = step {
                let Some(SlotValues::Tokens(values)) = family.slots.get(*slot) else {
                    return Err(format!("device: slot {slot} holds no tokens"));
                };
                if values.len() != family.rows {
                    return Err("device: token count does not match batch".to_string());
                }
                if !ids.contains_key(slot) {
                    ids.insert(*slot, Arc::new(self.device.upload_indices(values).map_err(error)?));
                    tokens.insert(*slot, values.clone());
                }
            }
        }
        let saved = Arc::new(PreparedBatch { rows: family.rows, layout: family.layout.clone(), tokens, ids, blocks, rotations: Arc::new(rotations) });
        *cached = Some(Arc::clone(&saved));
        Ok(saved)
    }

    fn layout(&self, family: &FamilyInputs) -> Result<(usize, Vec<(Rotary, Tensor, Tensor)>), String> {
        let rotaries: Vec<Rotary> = {
            let mut list: Vec<Rotary> = Vec::new();
            for step in &self.steps {
                if let Step::Attend { rotary: Some(r), .. } = step
                    && !list.contains(r)
                {
                    list.push(*r);
                }
            }
            list
        };
        let attends = self.steps.iter().any(|s| matches!(s, Step::Attend { .. }));
        let Some(layout) = &family.layout else {
            if attends {
                return Err("device: attention needs a sequence layout".to_string());
            }
            return Ok((1, Vec::new()));
        };
        let rows = family.rows;
        if layout.sequence.len() != rows || layout.position.len() != rows {
            return Err("device: layout length does not match batch".to_string());
        }
        let length = (1..=rows).find(|&l| l == rows || layout.sequence[l] != layout.sequence[0]).unwrap_or(rows);
        if attends {
            if length == 0 || rows % length != 0 {
                return Err(format!("device: {rows} rows are not equal sequence blocks of {length}"));
            }
            for b in 0..rows / length {
                let block = b * length..(b + 1) * length;
                let id = layout.sequence[block.start];
                if layout.sequence[block.clone()].iter().any(|s| *s != id)
                    || (b > 0 && layout.sequence[block.start - 1] == id)
                    || layout.position[block].windows(2).any(|w| w[1] <= w[0])
                {
                    return Err("device: a sequence is not one block of increasing positions".to_string());
                }
            }
        }
        let mut rotations = Vec::new();
        for rotary in rotaries {
            let planes = rotary.pairs().len();
            let (mut cos, mut sin) = (Vec::with_capacity(rows * planes), Vec::with_capacity(rows * planes));
            for &position in &layout.position {
                for plane in 0..planes {
                    let (c, s) = rotary.turn(plane, position);
                    cos.push(c);
                    sin.push(s);
                }
            }
            rotations.push((rotary, self.device.upload_vec(rows, planes, cos).map_err(error)?, self.device.upload_vec(rows, planes, sin).map_err(error)?));
        }
        Ok((if attends { rows / length } else { 1 }, rotations))
    }

    /// `out ← out + x op(A)` for an affine term (`x Aᵀ`) or a transposed read (`x A`).
    fn add_product(&self, out: &mut Tensor, x: &Tensor, op: usize, transposed: bool, arithmetic: Arithmetic) -> Result<(), String> {
        let d = &self.device;
        match self.held(op, Role::Product)? {
            Held::Identity => d.axpy(out, 1.0, x).map_err(error),
            Held::Diagonal(diag) => d.scale_columns(out, x, diag, true).map_err(error),
            Held::Dense(a) => d.gemm(out, 1.0, x, Op::N, a, if transposed { Op::N } else { Op::T }, 1.0, arithmetic).map_err(error),
            Held::LowRank(left, right) => {
                // x (L R)ᵀ = (x Rᵀ) Lᵀ;  x (L R) = (x L) R.
                let (first, first_op, second, second_op) = if transposed { (left, Op::N, right, Op::N) } else { (right, Op::T, left, Op::T) };
                let inner = if transposed { left.cols() } else { right.rows() };
                let mut middle = d.zeros(x.rows(), inner).map_err(error)?;
                d.gemm(&mut middle, 1.0, x, Op::N, first, first_op, 0.0, arithmetic).map_err(error)?;
                d.gemm(out, 1.0, &middle, Op::N, second, second_op, 1.0, arithmetic).map_err(error)
            }
            Held::Table(_) | Held::Column(_) => Err("device: an operator held in the wrong role".to_string()),
        }
    }

    fn column(&self, op: usize) -> Result<&Tensor, String> {
        match self.held(op, Role::Column)? {
            Held::Column(c) => Ok(c),
            _ => Err("device: an operator held in the wrong role".to_string()),
        }
    }

    /// One forward pass on `family` (module note).
    pub fn forward(&self, family: &FamilyInputs) -> Result<DeviceTrace, String> {
        self.forward_given(family, BTreeMap::new())
    }

    /// One forward pass on `family`, the raw slots in `given` taking those device values instead
    /// of the family's (which may then be empty).
    pub fn forward_given(&self, family: &FamilyInputs, given: BTreeMap<usize, Tensor>) -> Result<DeviceTrace, String> {
        self.forward_gated(family, given, &[], |_, _| Err("device: no gate decides".to_string()))
    }

    /// Resident forward values when edited execution materializes logits and Readout.
    /// Attention workspaces still require additional memory; callers should batch conservatively.
    pub fn edited_bytes_per_row(&self) -> usize {
        self.bytes_per_row().saturating_add(self.edited_head_nodes.keys().map(|&n| 8 * self.widths[n]).sum::<usize>())
    }

    /// An autonomous forward pass (`OperatorProgram::execute_with_gates` on the device): for each
    /// `(amplitude, mask)` of `gated` (a raw mask node read only after its amplitude node), once
    /// the amplitude is computed `decide(amplitude, trace)` returns the mask's value (rows × its
    /// width) from the trace so far, and the pass goes on with it. The other raw slots are as in
    /// [`Self::forward_given`]; a gated mask's slot needs no value in `family`.
    pub fn forward_gated(
        &self,
        family: &FamilyInputs,
        given: BTreeMap<usize, Tensor>,
        gated: &[(usize, usize)],
        decide: impl FnMut(usize, &DeviceTrace) -> Result<Tensor, String>,
    ) -> Result<DeviceTrace, String> {
        self.forward_hooks(Some(family), given, gated, decide, false, Hooks { before: None, edit: false }, |_, _| Ok(()), |_, _| Ok(None), Reuse::Nothing, Span::all())
    }

    /// Per-node edits on materialized values, preserving exception-before-intervention order:
    /// `before(node, value)` changes the value of each node in `before_at` in place, then
    /// `edit(node, trace)` may replace any node's value. In a group of sibling heads run as one
    /// computation ([`device_heads`]; one with a node in `before_at` runs node by node) `edit` is
    /// offered the group's projections first, then its heads, each replacement written back before
    /// the attention or the output reads it: there a hook reads only its own node's value. An
    /// unsupported unmaterialized head/feature is not offered to either callback.
    pub fn forward_edited(
        &self,
        family: &FamilyInputs,
        given: BTreeMap<usize, Tensor>,
        before_at: &BTreeSet<usize>,
        before: impl FnMut(usize, &mut Tensor) -> Result<(), String>,
        edit: impl FnMut(usize, &DeviceTrace) -> Result<Option<Tensor>, String>,
    ) -> Result<DeviceTrace, String> {
        self.forward_hooks(Some(family), given, &[], |_, _| Err("no gate".into()), true, Hooks { before: Some(before_at), edit: true }, before, edit, Reuse::Nothing, Span::all())
    }

    /// [`Self::forward_edited`] with the streamed dense head left unmaterialized.
    /// Callers must not request edits or exceptions at streamed head nodes.
    pub fn forward_edited_intermediates(
        &self,
        family: &FamilyInputs,
        before_at: &BTreeSet<usize>,
        before: impl FnMut(usize, &mut Tensor) -> Result<(), String>,
        edit: impl FnMut(usize, &DeviceTrace) -> Result<Option<Tensor>, String>,
    ) -> Result<DeviceTrace, String> {
        self.forward_hooks(Some(family), BTreeMap::new(), &[], |_, _| Err("no gate".into()), false, Hooks { before: Some(before_at), edit: true }, before, edit, Reuse::Nothing, Span::all())
    }

    /// [`Self::forward_edited_intermediates`] over a span of nodes: node `entry.0` takes the value
    /// `entry.1` and no node before it runs (none after it may read one, except token features),
    /// and no node after `end` runs. `edit` is offered every node that runs. One layer of a model
    /// thus runs on given rows of the stream entering it (`interchange`).
    pub fn forward_span(
        &self,
        family: &FamilyInputs,
        entry: Option<(usize, Tensor)>,
        end: usize,
        edit: impl FnMut(usize, &DeviceTrace) -> Result<Option<Tensor>, String>,
    ) -> Result<DeviceTrace, String> {
        let hooks = Hooks { before: None, edit: true };
        self.forward_hooks(Some(family), BTreeMap::new(), &[], |_, _| Err("no gate".into()), false, hooks, |_, _| Ok(()), edit, Reuse::Nothing, Span { entry, end })
    }

    pub fn is_streamed_head(&self, node: usize) -> bool {
        self.edited_head_nodes.contains_key(&node)
    }

    /// Which nodes no operator of `trainable` reaches.
    fn frozen_nodes(&self, trainable: &BTreeSet<usize>) -> Vec<bool> {
        let mut frozen = vec![true; self.steps.len()];
        for (index, step) in self.steps.iter().enumerate() {
            // The streamed head is never frozen (it has no resident value to keep).
            let operators: Vec<usize> = match step {
                Step::Head => vec![usize::MAX],
                Step::Constant { operator } | Step::Transposed { operator, .. } => vec![*operator],
                Step::Affine { terms, bias } => terms.iter().map(|t| t.1).chain(*bias).collect(),
                _ => Vec::new(),
            };
            frozen[index] = step_arguments(step).iter().all(|r| frozen[*r]) && operators.iter().all(|op| *op != usize::MAX && !trainable.contains(op));
        }
        frozen
    }

    /// The values on `family` of every node no operator of `trainable` reaches (module note of
    /// [`Frozen`]), the raw slots in `given` as in [`Self::forward_given`]; of those, the ones the
    /// other nodes read and the ones in `keep` keep their values.
    pub fn freeze(&self, family: &FamilyInputs, given: BTreeMap<usize, Tensor>, trainable: &[usize], keep: &[usize]) -> Result<Frozen, String> {
        let trainable: BTreeSet<usize> = trainable.iter().copied().collect();
        let frozen = self.frozen_nodes(&trainable);
        let batch = self.prepared_batch(family)?;
        let mut trace = self.forward_hooks(Some(family), given, &[], |_, _| Err("no gate".into()), false, Hooks { before: None, edit: false }, |_, _| Ok(()), |_, _| Ok(None), Reuse::Freeze(&frozen), Span::all())?;
        let mut read = vec![false; self.steps.len()];
        for (index, step) in self.steps.iter().enumerate() {
            if !frozen[index] {
                for argument in step_arguments(step) {
                    read[argument] = true;
                }
            }
        }
        let mut values = BTreeMap::new();
        let kept: Vec<usize> = (0..self.steps.len()).filter(|n| frozen[*n] && (read[*n] || keep.contains(n)) && trace.has(*n)).collect();
        for node in kept {
            values.insert(node, Arc::new(trace.take(node)?));
        }
        Ok(Frozen { batch, frozen: Arc::new(frozen), values, trainable, revisions: self.revisions.clone() })
    }

    /// A forward pass on [`Self::freeze`]'s family computing only the nodes a trainable operator
    /// reaches; the frozen nodes' values are `frozen`'s, shared. A reverse pass on it leaves the
    /// frozen nodes alone (no cotangent reaches them unless one is kept).
    pub fn forward_frozen(&self, frozen: &Frozen) -> Result<DeviceTrace, String> {
        if frozen.frozen.len() != self.steps.len() {
            return Err("device: frozen values of another program".into());
        }
        for (op, revision) in &self.revisions {
            if !frozen.trainable.contains(op) && frozen.revisions.get(op) != Some(revision) {
                return Err(format!("device: operator {op} changed since its frozen values were computed"));
            }
        }
        self.forward_hooks(None, BTreeMap::new(), &[], |_, _| Err("no gate".into()), false, Hooks { before: None, edit: false }, |_, _| Ok(()), |_, _| Ok(None), Reuse::From(frozen), Span::all())
    }

    fn forward_hooks(
        &self,
        family: Option<&FamilyInputs>,
        mut given: BTreeMap<usize, Tensor>,
        gated: &[(usize, usize)],
        mut decide: impl FnMut(usize, &DeviceTrace) -> Result<Tensor, String>,
        materialize_heads: bool,
        hooks: Hooks<'_>,
        mut before: impl FnMut(usize, &mut Tensor) -> Result<(), String>,
        mut edit: impl FnMut(usize, &DeviceTrace) -> Result<Option<Tensor>, String>,
        reuse: Reuse<'_>,
        span: Span,
    ) -> Result<DeviceTrace, String> {
        let d = &self.device;
        for &(amplitude, mask) in gated {
            if mask >= amplitude
                || amplitude >= self.steps.len()
                || !matches!(self.steps[mask], Step::Raw { .. })
                || self.widths[mask] != self.widths[amplitude]
            {
                return Err(format!("device: gate ({amplitude}, {mask}) is not an amplitude after its raw mask of the same width"));
            }
        }
        let batch = match (reuse, family) {
            (Reuse::From(frozen), _) => Arc::clone(&frozen.batch),
            (_, Some(family)) => self.prepared_batch(family)?,
            (_, None) => return Err("device: a forward pass without its family".into()),
        };
        let rows = batch.rows;
        let (entry, mut entered) = match span.entry {
            Some((node, value)) => (Some(node), Some(value)),
            None => (None, None),
        };
        if entry.is_some_and(|node| node >= self.steps.len() || node >= span.end) {
            return Err("device: a span's entry is not before its end".into());
        }
        // Which nodes this pass computes.
        let computes = |n: usize| n <= span.end && entry.is_none_or(|e| n > e) && match reuse {
            Reuse::Nothing => true,
            Reuse::Freeze(frozen) => frozen[n],
            Reuse::From(frozen) => !frozen.frozen[n],
        };
        // A group runs fused unless an in-place hook or a gate may change one of its nodes, its
        // sequences are long enough for the tiled attention, or this pass does not compute its heads.
        let tiled = Self::tiled(rows, batch.blocks);
        let fusing: Vec<bool> = self
            .fused
            .iter()
            .map(|g| g.live && !tiled && computes(g.heads.first()) && g.heads.members().all(|m| !hooks.before(m) && !gated.iter().any(|&(a, k)| a == m || k == m)))
            .collect();
        let mut trace = DeviceTrace {
            slots: (0..self.steps.len()).map(|_| Slot::Empty).collect(),
            buffers: Vec::new(),
            fused: vec![None; self.fused.len()],
            frozen: match reuse {
                Reuse::From(frozen) => Some(Arc::clone(&frozen.frozen)),
                _ => None,
            },
            device: d.clone(),
            rows,
            ids: batch.ids.clone(),
            blocks: batch.blocks,
            rotations: Arc::clone(&batch.rotations),
        };
        for (index, step) in self.steps.iter().enumerate() {
            if Some(index) == entry {
                let value = entered.take().ok_or("device: a span's entry value")?;
                if value.dim() != (rows, self.widths[index]) {
                    return Err(format!("device: a {:?} entry value for node {index} of {rows} x {}", value.dim(), self.widths[index]));
                }
                trace.slots[index] = Slot::Value(value);
                continue;
            }
            if index > span.end {
                break;
            }
            if !computes(index) {
                if let Reuse::From(frozen) = reuse
                    && let Some(value) = frozen.values.get(&index)
                {
                    trace.slots[index] = Slot::Shared(Arc::clone(value));
                }
                continue;
            }
            let group = self.grouped[index].filter(|g| fusing[*g]);
            if let Some(g) = group
                && index != self.fused[g].heads.output
            {
                if index == self.fused[g].heads.first() {
                    let mut offer = |trace: &DeviceTrace, node: usize| if hooks.edit { edit(node, trace) } else { Ok(None) };
                    self.run_heads(&mut trace, g, &mut offer)?;
                }
                continue;
            }
            let width = self.widths[index];
            let hook = hooks.before(index);
            let value = |t: Tensor| Slot::Value(t);
            let mut slot = if let Some(g) = group {
                value(self.heads_output(&trace, g)?)
            } else { match step {
                Step::Head if materialize_heads => match self.edited_head_nodes.get(&index) {
                    Some(Some(input)) if !hook => Slot::Alias(*input),
                    Some(Some(input)) => value(d.copy(trace.value(*input)?).map_err(error)?),
                    Some(None) => value(self.logits_on_device(&trace)?),
                    None => return Err(format!("device: missing edited head node {index}")),
                },
                Step::Head => Slot::Empty,
                Step::Feature { .. } => Slot::Empty,
                // A gated mask is filled once its amplitude is known.
                Step::Raw { .. } if gated.iter().any(|(_, mask)| *mask == index) => Slot::Empty,
                Step::Raw { slot } => match given.remove(slot) {
                    Some(given) if given.dim() == (rows, width) => value(given),
                    Some(given) => {
                        return Err(format!("device: a {:?} value for slot {slot} of {rows} × {width}", given.dim()));
                    }
                    None => {
                        let Some(SlotValues::Raw(values)) = family.map(|f| &f.slots[*slot]) else {
                            return Err(format!("device: slot {slot} holds no raw rows"));
                        };
                        value(d.upload(values.view()).map_err(error)?)
                    }
                },
                Step::Constant { operator } => value(d.broadcast_rows(self.column(*operator)?, rows).map_err(error)?),
                Step::Affine { terms, bias } => {
                    let mut out = d.zeros(rows, width).map_err(error)?;
                    for (argument, operator) in terms {
                        self.add_term(&mut out, &trace, *argument, *operator)?;
                    }
                    if let Some(b) = bias {
                        d.add_row(&mut out, 1.0, self.column(*b)?).map_err(error)?;
                    }
                    value(out)
                }
                Step::Pointwise { input, codes } => value(d.law_values(trace.value(*input)?, codes, gelu_tanh_constant()).map_err(error)?),
                Step::Gain { input, factor } => {
                    let mut out = d.zeros(rows, width).map_err(error)?;
                    d.axpy(&mut out, *factor, trace.value(*input)?).map_err(error)?;
                    value(out)
                }
                Step::Hadamard { left, right } => {
                    let mut out = d.zeros(rows, width).map_err(error)?;
                    d.hadamard(&mut out, trace.value(*left)?, trace.value(*right)?, false).map_err(error)?;
                    value(out)
                }
                Step::RmsNorm { input, epsilon } => value(d.rms_norm(trace.value(*input)?, *epsilon).map_err(error)?),
                Step::Attend { query, key, value: v, scale, rotary, causal } => {
                    let (q, k) = self.rotated(&trace, *query, *key, *rotary)?;
                    let v = trace.value(*v)?;
                    if tiled {
                        value(super::device_attention::forward(d, (&q, &k, v), trace.blocks, *scale, *causal, self.arithmetic).map_err(error)?)
                    } else {
                        let alpha = self.attention(&trace, &q, &k, *scale, *causal)?;
                        let mut out = d.zeros(rows, v.cols()).map_err(error)?;
                        d.gemm_batched(trace.blocks, &mut out, 1.0, &alpha, Op::N, v, Op::N, 0.0, self.arithmetic).map_err(error)?;
                        value(out)
                    }
                }
                // An indicator readout is its input's value; a hook may change it, so it gets its own.
                Step::Readout { input } if !hook => Slot::Alias(*input),
                Step::Readout { input } => value(d.copy(trace.value(*input)?).map_err(error)?),
                Step::Concat { parts } => {
                    let mut out = d.zeros(rows, width).map_err(error)?;
                    let mut start = 0;
                    for &part in parts {
                        let part = trace.value(part)?;
                        d.set_columns(&mut out, start, part).map_err(error)?;
                        start += part.cols();
                    }
                    value(out)
                }
                Step::Transposed { input, operator } => {
                    let mut out = d.zeros(rows, width).map_err(error)?;
                    self.add_product(&mut out, trace.value(*input)?, *operator, true, self.arithmetic)?;
                    value(out)
                }
            }};
            if hook && let Slot::Value(value) = &mut slot {
                before(index, value)?;
            }
            trace.slots[index] = slot;
            if hooks.edit && let Some(replacement) = edit(index, &trace)? {
                if !trace.has(index) {
                    return Err(format!("device: edit of unmaterialized node {index} is unsupported"));
                }
                if replacement.dim() != (rows, width) {
                    return Err(format!("device: edited node {index} has {:?}, expected {rows} x {width}", replacement.dim()));
                }
                trace.slots[index] = Slot::Value(replacement);
            }
            if let Some(&(_, mask)) = gated.iter().find(|(amplitude, _)| *amplitude == index) {
                let decided = decide(index, &trace)?;
                if decided.dim() != (rows, self.widths[mask]) {
                    return Err(format!("device: a {:?} mask for gate {index} of {rows} × {}", decided.dim(), self.widths[mask]));
                }
                trace.slots[mask] = Slot::Value(decided);
            }
        }
        Ok(trace)
    }

    /// `out ← out + x op(A)` for the affine term `(argument, operator)`: a gather of the operator's
    /// columns when `argument` is a one-hot feature.
    fn add_term(&self, out: &mut Tensor, trace: &DeviceTrace, argument: usize, operator: usize) -> Result<(), String> {
        let d = &self.device;
        if let Step::Feature { slot } = &self.steps[argument] {
            let Held::Table(table) = self.held(operator, Role::Table)? else {
                return Err("device: an operator held in the wrong role".to_string());
            };
            let ids = trace.ids.get(slot).ok_or("device: feature ids missing")?;
            let gathered = d.gather_rows(table, ids).map_err(error)?;
            d.axpy(out, 1.0, &gathered).map_err(error)
        } else {
            self.add_product(out, trace.value(argument)?, operator, false, self.arithmetic)
        }
    }

    /// Group `g`'s heads (module note of [`device_heads`]): `P` into the trace's buffers, each
    /// projection offered to `edit` in node order (a replacement written back into `P`), then `N`
    /// and `G` from it the same way when the queries and keys are normed, then `A`; every member's
    /// value a block of one of them.
    fn run_heads(
        &self,
        trace: &mut DeviceTrace,
        g: usize,
        edit: &mut impl FnMut(&DeviceTrace, usize) -> Result<Option<Tensor>, String>,
    ) -> Result<(), String> {
        let d = &self.device;
        let group = &self.fused[g];
        let heads = &group.heads;
        let rotations = Arc::clone(&trace.rotations);
        let turn = turn(&rotations, heads.rotary)?;
        let p = device_heads::project(d, heads, &group.stacked, trace.value(heads.input)?, self.arithmetic).map_err(error)?;
        let projections = Self::offer_blocks(trace, heads, (Buffer::Projections, p), edit)?;
        let normed = match &heads.norms {
            Some(norms) => {
                let n = device_heads::normalize(d, heads, &trace.buffers[projections], norms.epsilon).map_err(error)?;
                let normed = Self::offer_blocks(trace, heads, (Buffer::Normed, n), edit)?;
                let gained = device_heads::gain(d, &group.stacked, &trace.buffers[normed]).map_err(error)?;
                Some((normed, Self::offer_blocks(trace, heads, (Buffer::Gained, gained), edit)?))
            }
            None => None,
        };
        let qk = &trace.buffers[normed.map_or(projections, |(_, gained)| gained)];
        let a = device_heads::attend(d, heads, (qk, &trace.buffers[projections]), trace.blocks, turn, self.arithmetic).map_err(error)?;
        let reads = Self::offer_blocks(trace, heads, (Buffer::Reads, a), edit)?;
        trace.fused[g] = Some(Buffers { projections, normed, reads });
        Ok(())
    }

    /// `values` into the trace's buffers, each member of `buffer` a block of it, offered to `edit`
    /// in node order; a replacement becomes the node's value and is written into the buffer.
    fn offer_blocks(
        trace: &mut DeviceTrace,
        heads: &Heads,
        (buffer, values): (Buffer, Tensor),
        edit: &mut impl FnMut(&DeviceTrace, usize) -> Result<Option<Tensor>, String>,
    ) -> Result<usize, String> {
        let index = trace.buffers.len();
        trace.buffers.push(values);
        for node in heads.nodes_of(buffer) {
            let (_, start) = heads.block(node).ok_or("device: a fused node outside its group")?;
            trace.slots[node] = Slot::Columns { buffer: index, start, width: heads.width, copy: OnceLock::new() };
            if let Some(replacement) = edit(trace, node)? {
                if replacement.dim() != (trace.rows, heads.width) {
                    return Err(format!("device: edited node {node} has {:?}, expected {} x {}", replacement.dim(), trace.rows, heads.width));
                }
                let device = trace.device.clone();
                device.set_columns(&mut trace.buffers[index], start, &replacement).map_err(error)?;
                trace.slots[node] = Slot::Value(replacement);
            }
        }
        Ok(index)
    }

    /// Group `g`'s output node: its other terms, `A Oᵀ` and its bias.
    fn heads_output(&self, trace: &DeviceTrace, g: usize) -> Result<Tensor, String> {
        let d = &self.device;
        let group = &self.fused[g];
        let heads = &group.heads;
        let buffers = trace.fused[g].ok_or("device: a fused output before its heads")?;
        let mut out = d.zeros(trace.rows, self.widths[heads.output]).map_err(error)?;
        for (argument, operator) in &heads.rest {
            self.add_term(&mut out, trace, *argument, *operator)?;
        }
        d.gemm(&mut out, 1.0, &trace.buffers[buffers.reads], Op::N, &group.stacked.reads, Op::T, 1.0, self.arithmetic).map_err(error)?;
        if let Some(b) = heads.bias {
            d.add_row(&mut out, 1.0, self.column(b)?).map_err(error)?;
        }
        Ok(out)
    }

    /// The query and key of an attend node turned to their positions.
    fn rotated(&self, trace: &DeviceTrace, query: usize, key: usize, rotary: Option<Rotary>) -> Result<(Tensor, Tensor), String> {
        let d = &self.device;
        let (q, k) = (trace.value(query)?, trace.value(key)?);
        match rotary {
            None => Ok((d.copy(q).map_err(error)?, d.copy(k).map_err(error)?)),
            Some(r) => {
                let (cos, sin) = rotations_of(trace, r)?;
                Ok((d.rotate(q, cos, sin, r.half_split, false).map_err(error)?, d.rotate(k, cos, sin, r.half_split, false).map_err(error)?))
            }
        }
    }

    /// Whether the attention of `blocks` sequences in `rows` runs in query tiles
    /// ([`super::device_attention`]) rather than with every block's weights at once.
    fn tiled(rows: usize, blocks: usize) -> bool {
        let length = rows / blocks;
        length > 1024 || rows.saturating_mul(length) > 8 * 1024 * 1024
    }

    fn tile_attention(trace: &DeviceTrace) -> bool {
        Self::tiled(trace.rows, trace.blocks)
    }

    fn attention(&self, trace: &DeviceTrace, q: &Tensor, k: &Tensor, scale: f64, causal: bool) -> Result<Tensor, String> {
        let d = &self.device;
        let length = trace.rows / trace.blocks;
        let mut scores = d.zeros(trace.rows, length).map_err(error)?;
        d.gemm_batched(trace.blocks, &mut scores, scale, q, Op::N, k, Op::T, 0.0, self.arithmetic).map_err(error)?;
        d.softmax_rows(&mut scores, causal).map_err(error)?;
        Ok(scores)
    }

    /// Rows `start..start + n` of the logits.
    fn logits_tile(&self, hidden: &Tensor, start: usize, n: usize, arithmetic: Arithmetic) -> Result<Tensor, String> {
        self.linear_operator()?;
        let d = &self.device;
        let h = d.rows_of(hidden, start, n).map_err(error)?;
        let mut logits = d.zeros(n, self.head.classes).map_err(error)?;
        self.add_product(&mut logits, &h, self.linear_operator()?, self.head.transposed, arithmetic)?;
        Ok(logits)
    }

    /// `g_h ← g_h + g_logits · ∂logits/∂h` for one tile, written at rows `start..`.
    fn pull_tile(&self, g_hidden: &mut Tensor, start: usize, cotangent: &Tensor, arithmetic: Arithmetic) -> Result<(), String> {
        let d = &self.device;
        let Held::Dense(a) = self.held(self.linear_operator()?, Role::Product)? else {
            return Err("device: the head is not dense".to_string());
        };
        let mut part = d.zeros(cotangent.rows(), self.widths[self.head.hidden]).map_err(error)?;
        let op = if self.head.transposed { Op::T } else { Op::N };
        d.gemm(&mut part, 1.0, cotangent, Op::N, a, op, 0.0, arithmetic).map_err(error)?;
        d.set_rows(g_hidden, start, &part).map_err(error)
    }

    fn tile_rows(&self) -> usize {
        (TILE_BYTES / (8 * self.head.classes.max(1))).max(1)
    }

    fn flags(&self, scored: Option<&[bool]>, start: usize, n: usize) -> Result<Option<Indices>, String> {
        scored.map(|s| self.device.upload_indices(&s[start..start + n].iter().map(|b| u32::from(*b)).collect::<Vec<_>>()).map_err(error)).transpose()
    }

    /// Per row, `KL(softmax(target) ‖ softmax(logits))` (zero on rows `scored` leaves out), and
    /// its cotangent pulled back to the hidden node.
    pub fn kl(&self, trace: &DeviceTrace, target: &Tensor, scored: Option<&[bool]>) -> Result<(Array1<f64>, Tensor), String> {
        let (values, gradient) = self.kl_impl(trace, target, scored, true)?;
        Ok((values, gradient.ok_or("device: missing KL gradient")?))
    }

    /// Per-row KL for candidate acceptance, without a vocabulary cotangent or head pullback.
    pub fn score_only(&self, trace: &DeviceTrace, target: &Tensor, scored: Option<&[bool]>) -> Result<Array1<f64>, String> {
        self.kl_impl(trace, target, scored, false).map(|(values, _)| values)
    }

    fn kl_impl(&self, trace: &DeviceTrace, target: &Tensor, scored: Option<&[bool]>, gradient: bool) -> Result<(Array1<f64>, Option<Tensor>), String> {
        self.linear_operator()?;
        let d = &self.device;
        let hidden = trace.value(self.head.hidden)?;
        if target.dim() != (trace.rows, self.head.classes) {
            return Err(format!("device: a {:?} target for {} rows of {} classes", target.dim(), trace.rows, self.head.classes));
        }
        if scored.is_some_and(|s| s.len() != trace.rows) {
            return Err("device: scored flags do not match trace rows".to_string());
        }
        let mut kl = Vec::with_capacity(trace.rows);
        let mut g = gradient.then(|| d.zeros(trace.rows, hidden.cols()).map_err(error)).transpose()?;
        let tile = self.tile_rows();
        for start in (0..trace.rows).step_by(tile) {
            let n = tile.min(trace.rows - start);
            let mut logits = self.logits_tile(hidden, start, n, self.arithmetic)?;
            let t = d.rows_of(target, start, n).map_err(error)?;
            let flags = self.flags(scored, start, n)?;
            if let Some(g) = &mut g {
                kl.extend(d.kl_rows(&t, &mut logits, flags.as_ref()).map_err(error)?);
                self.pull_tile(g, start, &logits, self.arithmetic)?;
            } else {
                kl.extend(d.kl_score_rows(&t, &mut logits, flags.as_ref()).map_err(error)?);
            }
        }
        Ok((Array1::from(kl), g))
    }

    /// The cotangent of `−log q_y` pulled back to the hidden node, `y` drawn per row from the
    /// logits' softmax by `uniforms[r]` (rows `scored` leaves out stay zero).
    pub fn sampled(&self, trace: &DeviceTrace, uniforms: &[f64], scored: Option<&[bool]>, arithmetic: Arithmetic) -> Result<Tensor, String> {
        self.linear_operator()?;
        let d = &self.device;
        let hidden = trace.value(self.head.hidden)?;
        let mut g = d.zeros(trace.rows, hidden.cols()).map_err(error)?;
        let tile = self.tile_rows();
        for start in (0..trace.rows).step_by(tile) {
            let n = tile.min(trace.rows - start);
            let mut logits = self.logits_tile(hidden, start, n, self.arithmetic)?;
            let u = d.upload_vec(n, 1, uniforms[start..start + n].to_vec()).map_err(error)?;
            d.sampled_cotangent(&mut logits, &u, self.flags(scored, start, n)?.as_ref()).map_err(error)?;
            self.pull_tile(&mut g, start, &logits, arithmetic)?;
        }
        Ok(g)
    }

    /// Sample several hidden cotangents from one forward. Each vocabulary tile is projected,
    /// normalized, and pulled through the head only once. Seeds use `E_q[o] − o_y` in F64;
    /// subsequent reverse passes can still use proposal arithmetic. Only hidden-width seeds
    /// survive each tile, so no full-batch vocabulary distribution is retained.
    pub fn sampled_many(&self, trace: &DeviceTrace, uniforms: &[Vec<f64>], scored: Option<&[bool]>) -> Result<Vec<Tensor>, String> {
        self.linear_operator()?;
        if uniforms.iter().any(|u| u.len() != trace.rows) || scored.is_some_and(|s| s.len() != trace.rows) {
            return Err("device: sampled uniforms or flags do not match trace rows".to_string());
        }
        if uniforms.is_empty() {
            return Ok(Vec::new());
        }
        let d = &self.device;
        let hidden = trace.value(self.head.hidden)?;
        let Held::Dense(head) = self.held(self.linear_operator()?, Role::Product)? else {
            return Err("device: the head is not dense".to_string());
        };
        let mut seeds = uniforms.iter().map(|_| d.zeros(trace.rows, hidden.cols()).map_err(error)).collect::<Result<Vec<_>, _>>()?;
        let tile = self.tile_rows();
        for start in (0..trace.rows).step_by(tile) {
            let n = tile.min(trace.rows - start);
            let mut probabilities = self.logits_tile(hidden, start, n, self.arithmetic)?;
            d.softmax_rows(&mut probabilities, false).map_err(error)?;
            let mut mean = d.zeros(n, hidden.cols()).map_err(error)?;
            self.pull_tile(&mut mean, 0, &probabilities, self.arithmetic)?;
            let flags = self.flags(scored, start, n)?;
            for (uniforms, seed) in uniforms.iter().zip(&mut seeds) {
                let u = d.upload_vec(n, 1, uniforms[start..start + n].to_vec()).map_err(error)?;
                let part = d.sampled_head_cotangent(&probabilities, &mean, head, self.head.transposed, &u, flags.as_ref()).map_err(error)?;
                d.set_rows(seed, start, &part).map_err(error)?;
            }
        }
        Ok(seeds)
    }

    /// `Σ_rows Σ_c q_c (t_c − Σ_j q_j t_j)²`: the output Fisher's quadratic form on the logits'
    /// tangent `t = t_h · ∂logits/∂h` from the hidden node's tangent (rows `scored` leaves out
    /// add nothing).
    pub fn quadratic(&self, trace: &DeviceTrace, tangent: &Tensor, scored: Option<&[bool]>, arithmetic: Arithmetic) -> Result<f64, String> {
        self.linear_operator()?;
        let hidden = trace.value(self.head.hidden)?;
        let tile = self.tile_rows();
        let mut total = 0.0;
        for start in (0..trace.rows).step_by(tile) {
            let n = tile.min(trace.rows - start);
            let logits = self.logits_tile(hidden, start, n, self.arithmetic)?;
            let t = self.logits_tile(tangent, start, n, arithmetic)?;
            let per_row = self.device.softmax_quadratic(&logits, &t).map_err(error)?;
            total += per_row.iter().enumerate().filter(|(r, _)| scored.is_none_or(|s| s[start + r])).map(|(_, v)| v).sum::<f64>();
        }
        Ok(total)
    }

    /// The logits of rows `start..start + n`, on the host.
    pub fn logits(&self, trace: &DeviceTrace, start: usize, n: usize) -> Result<Array2<f64>, String> {
        let tile = self.logits_tile(trace.value(self.head.hidden)?, start, n, self.arithmetic)?;
        self.device.download(&tile).map_err(error)
    }

    /// Reverse pass (`derivatives::vjp`'s rules) from the hidden node's cotangent `seed`: the
    /// cotangents of the nodes in `keep` (those any cotangent reaches), each node's dropped once
    /// its own rule has run. Products run in `arithmetic`.
    pub fn vjp(&self, trace: &DeviceTrace, seed: Tensor, keep: &[usize], arithmetic: Arithmetic) -> Result<BTreeMap<usize, Tensor>, String> {
        self.vjp_seeded(trace, seed, BTreeMap::new(), keep, arithmetic)
    }

    /// [`Self::vjp`] of the hidden node's `seed` plus, at each node of `extra`, its cotangent
    /// there (a term of the scalar that reads that node directly).
    pub fn vjp_seeded(
        &self,
        trace: &DeviceTrace,
        seed: Tensor,
        extra: BTreeMap<usize, Tensor>,
        keep: &[usize],
        arithmetic: Arithmetic,
    ) -> Result<BTreeMap<usize, Tensor>, String> {
        self.linear_operator()?;
        let mut seeds = BTreeMap::from([(self.head.hidden, seed)]);
        for (node, term) in extra {
            match seeds.get_mut(&node) {
                Some(existing) => self.device.axpy(existing, 1.0, &term).map_err(error)?,
                None => {
                    seeds.insert(node, term);
                }
            }
        }
        self.reverse_seeds(trace, seeds, keep, arithmetic, &BTreeSet::new(), &mut |_, _| Ok(()))
    }

    /// Reverse resident-value expressions from explicitly declared node seeds.
    /// No vocabulary head or synthetic operator is required. Cotangents stay on
    /// the device; this differentiates fixed executed values, not serialization,
    /// rounding, structural search, or an acceptance verdict.
    pub fn vjp_values_seeded(
        &self,
        trace: &DeviceTrace,
        seeds: BTreeMap<usize, Tensor>,
        keep: &[usize],
        arithmetic: Arithmetic,
    ) -> Result<BTreeMap<usize, Tensor>, String> {
        if self.head.operator.is_some() {
            return Err("device: values VJP requires resident-value compilation".into());
        }
        self.reverse_seeds(trace, seeds, keep, arithmetic, &BTreeSet::new(), &mut |_, _| Ok(()))
    }

    /// The reverse pass from `seeds` down to the lowest node of `keep`; at each node of `edited`,
    /// `hook` changes its cotangent before the node's own rule reads it (the transpose of a forward
    /// edit there).
    fn reverse_seeds(
        &self,
        trace: &DeviceTrace,
        seeds: BTreeMap<usize, Tensor>,
        keep: &[usize],
        arithmetic: Arithmetic,
        edited: &BTreeSet<usize>,
        hook: &mut dyn FnMut(usize, &mut Tensor) -> Result<(), String>,
    ) -> Result<BTreeMap<usize, Tensor>, String> {
        if keep
            .iter()
            .chain(seeds.keys())
            .any(|node| *node >= self.steps.len())
        {
            return Err("device: retained or seeded node out of range".into());
        }
        for (&node, term) in &seeds {
            if term.dim() != (trace.rows, self.widths[node]) {
                return Err(format!(
                    "device: a {:?} cotangent at node {node} of width {}",
                    term.dim(),
                    self.widths[node]
                ));
            }
        }
        let Some(first) = keep.iter().copied().min() else {
            return Ok(BTreeMap::new());
        };
        let d = &self.device;
        // A group that ran fused reverses fused unless a cotangent is wanted or seeded inside it.
        let fused: Vec<bool> = self
            .fused
            .iter()
            .enumerate()
            .map(|(i, f)| f.live && trace.fused.get(i).is_some_and(Option::is_some) && f.heads.members().all(|m| !keep.contains(&m) && !seeds.contains_key(&m) && !edited.contains(&m)))
            .collect();
        let mut g: Vec<Option<Tensor>> = (0..self.steps.len()).map(|_| None).collect();
        for (node, term) in seeds {
            g[node] = Some(term);
        }
        // A frozen node's cotangent is wanted only when it is kept.
        let needed: Vec<bool> = (0..self.steps.len()).map(|n| keep.contains(&n) || trace.frozen.as_ref().is_none_or(|f| !f[n])).collect();
        let mut kept = BTreeMap::new();
        // Adds `term` into node `n`'s cotangent.
        let add = |g: &mut Vec<Option<Tensor>>, n: usize, term: Tensor| -> Result<(), String> {
            match g[n].as_mut() {
                Some(existing) => d.axpy(existing, 1.0, &term).map_err(error),
                None => {
                    g[n] = Some(term);
                    Ok(())
                }
            }
        };
        let slot = |g: &mut Vec<Option<Tensor>>, n: usize, rows: usize| -> Result<(), String> {
            if g[n].is_none() {
                g[n] = Some(d.zeros(rows, self.widths[n]).map_err(error)?);
            }
            Ok(())
        };
        for index in (first..self.steps.len()).rev() {
            let Some(mut cot) = g[index].take() else { continue };
            if edited.contains(&index) {
                hook(index, &mut cot)?;
            }
            if index == first {
                kept.insert(index, cot);
                break;
            }
            if let Some(group) = self.grouped[index].filter(|f| fused[*f] && self.fused[*f].heads.output == index) {
                self.heads_reverse(trace, group, &cot, (&mut g, &needed), arithmetic)?;
                if keep.contains(&index) {
                    kept.insert(index, cot);
                }
                continue;
            }
            match &self.steps[index] {
                Step::Concat { parts } => {
                    let mut column = 0usize;
                    for part in parts {
                        let end = column.checked_add(self.widths[*part]).ok_or("device: Concat cotangent width overflow")?;
                        if needed[*part] {
                            let term = d.columns_of(&cot, column..end).map_err(error)?;
                            add(&mut g, *part, term)?;
                        }
                        column = end;
                    }
                    if column != cot.cols() { return Err("device: Concat cotangent width mismatch".into()); }
                }
                Step::Readout { input } | Step::Gain { input, .. } | Step::Transposed { input, .. } | Step::Pointwise { input, .. } | Step::RmsNorm { input, .. }
                    if !needed[*input] => {}
                Step::Readout { input } => add(&mut g, *input, d.copy(&cot).map_err(error)?)?,
                Step::Gain { input, factor } => {
                    let mut term = d.zeros(cot.rows(), cot.cols()).map_err(error)?;
                    d.axpy(&mut term, *factor, &cot).map_err(error)?;
                    add(&mut g, *input, term)?;
                }
                Step::Head | Step::Feature { .. } | Step::Raw { .. } | Step::Constant { .. } => {}
                Step::Affine { terms, .. } => {
                    for (argument, operator) in terms {
                        self.pull_term((&mut g, &needed), trace.rows, &cot, *argument, *operator, arithmetic)?;
                    }
                }
                Step::Transposed { input, operator } => {
                    slot(&mut g, *input, trace.rows)?;
                    let target = g[*input].as_mut().ok_or("device: cotangent slot")?;
                    self.add_product(target, &cot, *operator, false, arithmetic)?;
                }
                Step::Pointwise { input, codes } => {
                    let term = d
                        .law_slopes(&cot, trace.value(*input)?, codes, gelu_tanh_constant())
                        .map_err(error)?;
                    add(&mut g, *input, term)?;
                }
                Step::Hadamard { left, right } => {
                    // A raw input's cotangent goes nowhere unless it is kept (a mask's).
                    let wanted = |n: usize| {
                        keep.contains(&n)
                            || (needed[n] && !matches!(self.steps[n], Step::Raw { .. } | Step::Constant { .. }))
                    };
                    if wanted(*left) {
                        let mut gl = d.zeros(trace.rows, self.widths[*left]).map_err(error)?;
                        d.hadamard(&mut gl, &cot, trace.value(*right)?, false)
                            .map_err(error)?;
                        add(&mut g, *left, gl)?;
                    }
                    if wanted(*right) {
                        let mut gr = d.zeros(trace.rows, self.widths[*right]).map_err(error)?;
                        d.hadamard(&mut gr, &cot, trace.value(*left)?, false)
                            .map_err(error)?;
                        add(&mut g, *right, gr)?;
                    }
                }
                Step::RmsNorm { input, epsilon } => {
                    let term = d
                        .rms_norm_backward(trace.value(*input)?, &cot, *epsilon)
                        .map_err(error)?;
                    add(&mut g, *input, term)?;
                }
                Step::Attend {
                    query,
                    key,
                    value,
                    scale,
                    rotary,
                    causal,
                } if [*query, *key, *value].iter().any(|n| needed[*n]) => {
                    let (gq, gk, gv) = self.attend_cotangent(
                        trace,
                        (*query, *key, *value),
                        &cot,
                        *scale,
                        *rotary,
                        *causal,
                        arithmetic,
                    )?;
                    for (node, term) in [(*query, gq), (*key, gk), (*value, gv)] {
                        if needed[node] {
                            add(&mut g, node, term)?;
                        }
                    }
                }
                Step::Attend { .. } => {}
            }
            if keep.contains(&index) {
                kept.insert(index, cot);
            }
        }
        Ok(kept)
    }

    /// `g[argument] ← g[argument] + cot · ∂(x op(A)ᵀ)/∂x` for the affine term `(argument, operator)`
    /// (nothing for a one-hot feature).
    fn pull_term(&self, (g, needed): (&mut [Option<Tensor>], &[bool]), rows: usize, cot: &Tensor, argument: usize, operator: usize, arithmetic: Arithmetic) -> Result<(), String> {
        let d = &self.device;
        if matches!(self.steps[argument], Step::Feature { .. }) || !needed[argument] {
            return Ok(());
        }
        if g[argument].is_none() {
            g[argument] = Some(d.zeros(rows, self.widths[argument]).map_err(error)?);
        }
        let target = g[argument].as_mut().ok_or("device: cotangent slot")?;
        match self.held(operator, Role::Product)? {
            Held::Identity => d.axpy(target, 1.0, cot).map_err(error),
            Held::Diagonal(diag) => d.scale_columns(target, cot, diag, true).map_err(error),
            Held::Dense(a) => d.gemm(target, 1.0, cot, Op::N, a, Op::N, 1.0, arithmetic).map_err(error),
            Held::LowRank(left, right) => {
                // g (L R) = (g L) R.
                let mut middle = d.zeros(cot.rows(), left.cols()).map_err(error)?;
                d.gemm(&mut middle, 1.0, cot, Op::N, left, Op::N, 0.0, arithmetic).map_err(error)?;
                d.gemm(target, 1.0, &middle, Op::N, right, Op::N, 1.0, arithmetic).map_err(error)
            }
            Held::Table(_) | Held::Column(_) => Err("device: an operator held in the wrong role".to_string()),
        }
    }

    /// Group `g`'s reverse rule from its output node's cotangent `cot`: the other terms as an
    /// affine node's, then `A`'s cotangent `cot O`, the heads backwards to `P`'s, and the input's
    /// `g_P W`.
    fn heads_reverse(&self, trace: &DeviceTrace, g: usize, cot: &Tensor, (grads, needed): (&mut [Option<Tensor>], &[bool]), arithmetic: Arithmetic) -> Result<(), String> {
        let d = &self.device;
        let group = &self.fused[g];
        let heads = &group.heads;
        let buffers = trace.fused[g].ok_or("device: a fused reverse without its forward")?;
        for (argument, operator) in &heads.rest {
            self.pull_term((&mut *grads, needed), trace.rows, cot, *argument, *operator, arithmetic)?;
        }
        if !needed[heads.input] {
            return Ok(());
        }
        let mut g_a = d.zeros(trace.rows, heads.heads * heads.width).map_err(error)?;
        d.gemm(&mut g_a, 1.0, cot, Op::N, &group.stacked.reads, Op::N, 0.0, arithmetic).map_err(error)?;
        let turn = turn(&trace.rotations, heads.rotary)?;
        let gained = buffers.normed.map(|(_, gained)| &trace.buffers[gained]);
        let g_p = device_heads::backward(d, (heads, &group.stacked), (&trace.buffers[buffers.projections], gained), &g_a, trace.blocks, turn, (self.arithmetic, arithmetic)).map_err(error)?;
        if grads[heads.input].is_none() {
            grads[heads.input] = Some(d.zeros(trace.rows, self.widths[heads.input]).map_err(error)?);
        }
        let target = grads[heads.input].as_mut().ok_or("device: cotangent slot")?;
        d.gemm(target, 1.0, &g_p, Op::N, &group.stacked.weights, Op::N, 1.0, arithmetic).map_err(error)
    }

    /// Resident cotangents for explicitly trainable dense operators, including
    /// bias and Constant columns. Every shared use contributes once in its actual
    /// orientation, including operators used in both column and product roles.
    /// Unsupported trainable roles (tables, diagonal/low-rank/identity bodies)
    /// fail explicitly. Parameters and source traces are never changed.
    pub fn vjp_values_dense(
        &self,
        trace: &DeviceTrace,
        seeds: BTreeMap<usize, Tensor>,
        keep: &[usize],
        trainable: &[usize],
        arithmetic: Arithmetic,
    ) -> Result<(BTreeMap<usize, Tensor>, BTreeMap<usize, Tensor>), String> {
        self.vjp_values_dense_edited(trace, seeds, keep, trainable, arithmetic, &BTreeSet::new(), &mut |_, _| Ok(()))
    }

    /// [`Self::vjp_values_dense`] of a forward pass whose nodes `edited` were edited: at each,
    /// `hook` maps the cotangent of the edited value to that of the value the node computed (the
    /// transpose of the edit) before the node's own rule reads it; the hook keeps any other part
    /// of the edit's transpose itself (`interchange`).
    pub fn vjp_values_dense_edited(
        &self,
        trace: &DeviceTrace,
        seeds: BTreeMap<usize, Tensor>,
        keep: &[usize],
        trainable: &[usize],
        arithmetic: Arithmetic,
        edited: &BTreeSet<usize>,
        hook: &mut dyn FnMut(usize, &mut Tensor) -> Result<(), String>,
    ) -> Result<(BTreeMap<usize, Tensor>, BTreeMap<usize, Tensor>), String> {
        if self.head.operator.is_some() {
            return Err("device: dense values VJP requires resident-value compilation".into());
        }
        let requested: std::collections::BTreeSet<_> = trainable.iter().copied().collect();
        if requested.len() != trainable.len() {
            return Err("device: duplicate trainable operator".into());
        }
        let mut gradients = BTreeMap::new();
        for &op in &requested {
            if self
                .operators
                .keys()
                .any(|(index, role)| *index == op && *role == Role::Table)
            {
                return Err(
                    "device: trainable operator has an unsupported table role".into(),
                );
            }
            let held = self.operators.get(&(op, Role::Product))
                .or_else(|| self.operators.get(&(op, Role::Column)))
                .ok_or("device: trainable operator has no product or column role")?;
            if !matches!(held.source.body, OperatorBody::Dense { .. }) {
                return Err("device: trainable operator must have a dense literal body".into());
            }
            // Dense literals may be executed by an exact diagonal fast path;
            // their parameter space still contains every matrix entry.
            gradients.insert(op, self.device.zeros(held.source.rows.width(), held.source.cols.width()).map_err(error)?);
        }
        let mut retained = keep.to_vec();
        for (node, step) in self.steps.iter().enumerate() {
            let uses = match step {
                Step::Affine { terms, bias } => terms.iter().any(|(_, op)| requested.contains(op))
                    || bias.is_some_and(|op| requested.contains(&op)),
                Step::Constant { operator } => requested.contains(operator),
                Step::Transposed { operator, .. } => requested.contains(operator),
                _ => false,
            };
            if uses && !retained.contains(&node) {
                retained.push(node);
            }
        }
        let mut nodes = self.reverse_seeds(trace, seeds, &retained, arithmetic, edited, hook)?;
        // One scalar constant is uploaded; all reductions and gradient arrays stay
        // on the device. Reuse the same broadcast across every column occurrence.
        let has_columns = requested.iter().any(|op| self.operators.contains_key(&(*op, Role::Column)));
        let ones = if has_columns {
            let one = self.device.upload_vec(1, 1, vec![1.0]).map_err(error)?;
            Some(self.device.broadcast_rows(&one, trace.rows).map_err(error)?)
        } else { None };
        for (node, step) in self.steps.iter().enumerate() {
            let Some(cot) = nodes.get(&node) else {
                continue;
            };
            if let Step::Affine { terms, .. } = step {
                for (input, op) in terms {
                    if let Some(gradient) = gradients.get_mut(op) {
                        self.device.gemm(gradient, 1.0, cot, Op::T, trace.value(*input)?, Op::N, 1.0, arithmetic).map_err(error)?;
                    }
                }
            } else if let Step::Transposed { input, operator } = step {
                if let Some(gradient) = gradients.get_mut(operator) {
                    self.device.gemm(gradient, 1.0, trace.value(*input)?, Op::T, cot, Op::N, 1.0, arithmetic).map_err(error)?;
                }
            }
            let column = match step {
                Step::Affine { bias, .. } => *bias,
                Step::Constant { operator } => Some(*operator),
                _ => None,
            };
            if let Some(gradient) = column.and_then(|op| gradients.get_mut(&op)) {
                self.device.gemm(gradient, 1.0, cot, Op::T,
                    ones.as_ref().ok_or("device: missing column reduction workspace")?, Op::N,
                    1.0, arithmetic).map_err(error)?;
            }
        }
        nodes.retain(|node, _| keep.contains(node));
        Ok((nodes, gradients))
    }

    fn attend_cotangent(
        &self,
        trace: &DeviceTrace,
        (query, key, value): (usize, usize, usize),
        cot: &Tensor,
        scale: f64,
        rotary: Option<Rotary>,
        causal: bool,
        arithmetic: Arithmetic,
    ) -> Result<(Tensor, Tensor, Tensor), String> {
        let d = &self.device;
        let blocks = trace.blocks;
        let (q, k) = self.rotated(trace, query, key, rotary)?;
        if Self::tile_attention(trace) {
            let (gq, gk, gv) = super::device_attention::backward(d, (&q, &k, trace.value(value)?), cot, blocks, scale, causal, (self.arithmetic, arithmetic))
                .map_err(error)?;
            return match rotary {
                None => Ok((gq, gk, gv)),
                Some(r) => {
                    let (cos, sin) = rotations_of(trace, r)?;
                    Ok((d.rotate(&gq, cos, sin, r.half_split, true).map_err(error)?, d.rotate(&gk, cos, sin, r.half_split, true).map_err(error)?, gv))
                }
            };
        }
        let alpha = self.attention(trace, &q, &k, scale, causal)?;
        let v = trace.value(value)?;
        let length = trace.rows / blocks;
        let mut dalpha = d.zeros(trace.rows, length).map_err(error)?;
        d.gemm_batched(blocks, &mut dalpha, 1.0, cot, Op::N, v, Op::T, 0.0, arithmetic).map_err(error)?;
        let mut gv = d.zeros(trace.rows, v.cols()).map_err(error)?;
        d.gemm_batched(blocks, &mut gv, 1.0, &alpha, Op::T, cot, Op::N, 0.0, arithmetic).map_err(error)?;
        let ds = d.softmax_backward(&alpha, &dalpha).map_err(error)?;
        drop((alpha, dalpha));
        let mut gq = d.zeros(trace.rows, q.cols()).map_err(error)?;
        d.gemm_batched(blocks, &mut gq, scale, &ds, Op::N, &k, Op::N, 0.0, arithmetic).map_err(error)?;
        let mut gk = d.zeros(trace.rows, k.cols()).map_err(error)?;
        d.gemm_batched(blocks, &mut gk, scale, &ds, Op::T, &q, Op::N, 0.0, arithmetic).map_err(error)?;
        let (gq, gk) = match rotary {
            None => (gq, gk),
            Some(r) => {
                let (cos, sin) = rotations_of(trace, r)?;
                (d.rotate(&gq, cos, sin, r.half_split, true).map_err(error)?, d.rotate(&gk, cos, sin, r.half_split, true).map_err(error)?)
            }
        };
        Ok((gq, gk, gv))
    }

    /// Forward tangent (`derivatives::jvp`'s rules) of the hidden node when each operator in
    /// `tangents` moves along its entry (a host matrix of the operator's shape); `None` when no
    /// tangent reaches it. Products run in `arithmetic`.
    pub fn jvp(&self, trace: &DeviceTrace, tangents: &BTreeMap<usize, Array2<f64>>, arithmetic: Arithmetic) -> Result<Option<Tensor>, String> {
        self.linear_operator()?;
        let d = &self.device;
        let rows = trace.rows;
        let upload = |m: &Array2<f64>| d.upload(m.view()).map_err(error);
        let mut dv: Vec<Option<Tensor>> = (0..self.steps.len()).map(|_| None).collect();
        for index in 0..=self.head.hidden {
            let width = self.widths[index];
            let t = match &self.steps[index] {
                Step::Concat { .. } | Step::Readout { .. } => return Err("device: resident Concat/readout derivatives are unsupported".into()),
                Step::Head | Step::Feature { .. } | Step::Raw { .. } => None,
                Step::Constant { operator } => match tangents.get(operator) {
                    Some(dc) => Some(d.broadcast_rows(&upload(&dc.t().to_owned())?, rows).map_err(error)?),
                    None => None,
                },
                Step::Affine { terms, bias } => {
                    let mut out: Option<Tensor> = None;
                    for (argument, operator) in terms {
                        if let Some(dx) = dv[*argument].as_ref() {
                            self.add_product(ensure(d, &mut out, rows, width)?, dx, *operator, false, arithmetic)?;
                        }
                        if let Some(da) = tangents.get(operator) {
                            let o = ensure(d, &mut out, rows, width)?;
                            let diagonal_row = self
                                .operators
                                .get(&(*operator, Role::Product))
                                .is_some_and(|held| matches!(held.source.body, OperatorBody::Diagonal { .. }) && da.dim() == (1, width));
                            if let Step::Feature { slot } = &self.steps[*argument] {
                                let ids = trace.ids.get(slot).ok_or("device: feature ids missing")?;
                                let gathered = d.gather_rows(&upload(&da.t().to_owned())?, ids).map_err(error)?;
                                d.axpy(o, 1.0, &gathered).map_err(error)?;
                            } else if diagonal_row {
                                // A diagonal's tangent given as its diagonal: a column scale.
                                d.scale_columns(o, trace.value(*argument)?, &upload(da)?, true).map_err(error)?;
                            } else {
                                d.gemm(o, 1.0, trace.value(*argument)?, Op::N, &upload(da)?, Op::T, 1.0, arithmetic).map_err(error)?;
                            }
                        }
                    }
                    if let Some(db) = bias.and_then(|b| tangents.get(&b)) {
                        d.add_row(ensure(d, &mut out, rows, width)?, 1.0, &upload(&db.t().to_owned())?).map_err(error)?;
                    }
                    out
                }
                Step::Transposed { input, operator } => {
                    let mut out: Option<Tensor> = None;
                    if let Some(dx) = dv[*input].as_ref() {
                        let mut o = d.zeros(rows, width).map_err(error)?;
                        self.add_product(&mut o, dx, *operator, true, arithmetic)?;
                        out = Some(o);
                    }
                    if let Some(da) = tangents.get(operator) {
                        d.gemm(ensure(d, &mut out, rows, width)?, 1.0, trace.value(*input)?, Op::N, &upload(da)?, Op::N, 1.0, arithmetic).map_err(error)?;
                    }
                    out
                }
                Step::Pointwise { input, codes } => match dv[*input].as_ref() {
                    Some(dx) => Some(d.law_slopes(dx, trace.value(*input)?, codes, gelu_tanh_constant()).map_err(error)?),
                    None => None,
                },
                Step::Gain { input, factor } => match dv[*input].as_ref() {
                    Some(dx) => {
                        let mut out = d.zeros(rows, width).map_err(error)?;
                        d.axpy(&mut out, *factor, dx).map_err(error)?;
                        Some(out)
                    }
                    None => None,
                },
                Step::Hadamard { left, right } => match (dv[*left].as_ref(), dv[*right].as_ref()) {
                    (None, None) => None,
                    (dl, dr) => {
                        let mut out = d.zeros(rows, width).map_err(error)?;
                        if let Some(dl) = dl {
                            d.hadamard(&mut out, dl, trace.value(*right)?, true).map_err(error)?;
                        }
                        if let Some(dr) = dr {
                            d.hadamard(&mut out, trace.value(*left)?, dr, true).map_err(error)?;
                        }
                        Some(out)
                    }
                },
                Step::RmsNorm { input, epsilon } => match dv[*input].as_ref() {
                    Some(dx) => Some(d.rms_norm_tangent(trace.value(*input)?, dx, *epsilon).map_err(error)?),
                    None => None,
                },
                Step::Attend { query, key, value, scale, rotary, causal } => {
                    if dv[*query].is_none() && dv[*key].is_none() && dv[*value].is_none() {
                        None
                    } else {
                        Some(self.attend_tangent(trace, (*query, *key, *value), &dv, *scale, *rotary, *causal, arithmetic)?)
                    }
                }
            };
            dv[index] = t;
        }
        Ok(dv[self.head.hidden].take())
    }

    fn attend_tangent(
        &self,
        trace: &DeviceTrace,
        (query, key, value): (usize, usize, usize),
        dv: &[Option<Tensor>],
        scale: f64,
        rotary: Option<Rotary>,
        causal: bool,
        arithmetic: Arithmetic,
    ) -> Result<Tensor, String> {
        let d = &self.device;
        let blocks = trace.blocks;
        let (q, k) = self.rotated(trace, query, key, rotary)?;
        let turn = |t: &Tensor| -> Result<Tensor, String> {
            match rotary {
                None => d.copy(t).map_err(error),
                Some(r) => {
                    let (cos, sin) = rotations_of(trace, r)?;
                    d.rotate(t, cos, sin, r.half_split, false).map_err(error)
                }
            }
        };
        if Self::tile_attention(trace) {
            let dq = dv[query].as_ref().map(&turn).transpose()?;
            let dk = dv[key].as_ref().map(&turn).transpose()?;
            return super::device_attention::tangent(
                d,
                (&q, &k, trace.value(value)?),
                (dq.as_ref(), dk.as_ref(), dv[value].as_ref()),
                blocks,
                scale,
                causal,
                (self.arithmetic, arithmetic),
            )
            .map_err(error);
        }
        let alpha = self.attention(trace, &q, &k, scale, causal)?;
        let length = trace.rows / blocks;
        let mut ds = d.zeros(trace.rows, length).map_err(error)?;
        if let Some(dq) = dv[query].as_ref() {
            d.gemm_batched(blocks, &mut ds, scale, &turn(dq)?, Op::N, &k, Op::T, 1.0, arithmetic).map_err(error)?;
        }
        if let Some(dk) = dv[key].as_ref() {
            d.gemm_batched(blocks, &mut ds, scale, &q, Op::N, &turn(dk)?, Op::T, 1.0, arithmetic).map_err(error)?;
        }
        let dalpha = d.softmax_backward(&alpha, &ds).map_err(error)?;
        let v = trace.value(value)?;
        let mut out = d.zeros(trace.rows, v.cols()).map_err(error)?;
        d.gemm_batched(blocks, &mut out, 1.0, &dalpha, Op::N, v, Op::N, 0.0, arithmetic).map_err(error)?;
        if let Some(dvv) = dv[value].as_ref() {
            d.gemm_batched(blocks, &mut out, 1.0, &alpha, Op::N, dvv, Op::N, 1.0, arithmetic).map_err(error)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod value_sharing_tests {
    use super::*;
    use crate::operator_program::{Declarations, Interface, Slot, exact_precision};
    use ndarray::array;
    #[test]
    fn value_sharing_preserves_native_f64_and_index_shifted_parameter_buffers() {
        let interface = Interface::native(2).unwrap();
        let values = array![[1.0 + 2.0_f64.powi(-40), 1.0], [0.0, 2.0]];
        let operator = Arc::new(Operator::dense("actual-native", interface.clone(), interface.clone(), values.clone(),
            exact_precision(values.iter().copied()).unwrap(), Default::default()).unwrap());
        let native = OperatorProgram { declarations: Declarations { domains: vec![], slots: vec![Slot::Raw { width: 2 }], parameters: 0 },
            bases: vec![], rules: vec![], operators: vec![operator.clone()],
            nodes: vec![Node::Raw { slot: 0 }, Node::Affine { terms: vec![(0, 0)], bias: None }, Node::Concat { parts: vec![1, 0] }], output: 2 };
        assert!(!crate::artifact::Artifact::native(&native).unwrap().has_f32_literals());
        let family = FamilyInputs { rows: 1, slots: vec![SlotValues::Raw(array![[1.0, 2.0]])], layout: None };
        let device = Device::host();
        assert!(DeviceProgram::compile_values_bounded(&device, &native, 31).is_err());
        let source = DeviceProgram::compile_values_bounded(&device, &native, 32).unwrap();
        assert_eq!(source.operator_numeric_bytes().unwrap(), 32);
        let mut shifted = native.clone();
        shifted.operators.insert(0, Arc::new(Operator::identity("unused", interface)));
        shifted.nodes[1] = Node::Affine { terms: vec![(0, 1)], bias: None };
        let shared = DeviceProgram::compile_values_sharing(&source, &shifted).unwrap();
        let retained = source.operator_numeric_bytes().unwrap();
        let bounded = DeviceProgram::compile_values_sharing_bounded(&source, &shifted, retained).unwrap();
        assert!(Arc::ptr_eq(&source.operators[&(0, Role::Product)].held, &bounded.operators[&(1, Role::Product)].held));
        assert!(DeviceProgram::compile_values_sharing_bounded(&source, &shifted, retained-1).is_err());
        assert!(DeviceProgram::compile_values_sharing_bounded(&source, &shifted, 0).is_err());
        let fresh = DeviceProgram::compile_values(&device, &shifted).unwrap();
        assert!(Arc::ptr_eq(&source.operators[&(0, Role::Product)].held, &shared.operators[&(1, Role::Product)].held));
        let a = shared.forward(&family).unwrap(); let b = fresh.forward(&family).unwrap();
        let output = |p: &DeviceProgram, t: &DeviceTrace| device.download(t.value(p.hidden()).unwrap()).unwrap();
        assert_eq!(output(&shared, &a), output(&fresh, &b));
        assert_eq!(output(&shared, &a), native.execute(&family, false).unwrap().values[native.output]);
        let before = output(&source, &source.forward(&family).unwrap());
        let mut altered = shifted.clone();
        let OperatorBody::Dense { values, .. } = &mut Arc::make_mut(&mut altered.operators[1]).body else { panic!("dense source") };
        values[[0, 0]] += 2.0_f64.powi(-40);
        let changed = DeviceProgram::compile_values_sharing(&source, &altered).unwrap();
        assert!(!Arc::ptr_eq(&source.operators[&(0, Role::Product)].held, &changed.operators[&(1, Role::Product)].held));
        assert_eq!(before, output(&source, &source.forward(&family).unwrap()));
        assert_eq!(operator.matrix(), native.operators[0].matrix());
    }
}

#[cfg(test)]
mod values_vjp_tests {
    use super::*;
    use crate::operator_program::{Declarations, FamilyInputs, Interface, Rule, Slot, SlotValues, exact_precision};
    use ndarray::{Array2, array};

    fn fixture() -> (OperatorProgram, FamilyInputs) {
        let interface = Interface::native(2).unwrap();
        let values = array![[0.8, -0.3], [0.2, 1.1]];
        let operator = Arc::new(Operator::dense("shared", interface.clone(), interface.clone(), values.clone(),
            exact_precision(values.iter().copied()).unwrap(), Default::default()).unwrap());
        let program = OperatorProgram {
            declarations: Declarations { domains: vec![], slots: vec![Slot::Raw { width: 2 }], parameters: 0 },
            bases: vec![], operators: vec![operator],
            rules: vec![Rule { name: "nonlinear shared body".into(), inputs: vec![interface],
                nodes: vec![Node::Param { index: 0 }, Node::Pointwise { input: 0, laws: vec![Law::GeluTanh] },
                    Node::Hadamard { left: 1, right: 0 }], output: 2 }],
            nodes: vec![Node::Raw { slot: 0 }, Node::Affine { terms: vec![(0, 0)], bias: None },
                Node::Call { rule: 0, arguments: vec![1] }, Node::Call { rule: 0, arguments: vec![0] },
                Node::Hadamard { left: 2, right: 3 }, Node::Transposed { input: 4, operator: 0 },
                Node::Affine { terms: vec![(5, 0), (1, 0)], bias: None }], output: 6,
        };
        let family = FamilyInputs { rows: 2, slots: vec![SlotValues::Raw(array![[0.4, -0.7], [1.2, 0.3]])], layout: None };
        (program, family)
    }
    fn seeds(program: &OperatorProgram) -> BTreeMap<usize, Array2<f64>> {
        BTreeMap::from([(program.output, array![[0.7, -0.2], [-0.4, 0.9]]), (2, Array2::from_elem((2, 2), 0.13))])
    }
    fn objective(program: &OperatorProgram, family: &FamilyInputs) -> f64 {
        let trace = program.execute(family, false).unwrap();
        seeds(program).into_iter().map(|(node, seed)| (&trace.values[node] * &seed).sum()).sum()
    }
    #[test]
    fn fixed_vector_gains_preserve_shared_calls_values_and_cotangents() {
        use crate::operator_program::Coefficient;
        let (mut called, family) = fixture();
        called.rules[0].nodes.push(Node::Gain { input: 2, coefficient: Coefficient::Product(vec![
            Coefficient::Number(-0.5), Coefficient::Sum(vec![Coefficient::Number(1.0), Coefficient::Number(0.5)])]) });
        called.rules[0].output = 3;
        let artifact = crate::artifact::Artifact::native(&called).unwrap();
        let decoded = crate::artifact::Artifact::from_bytes(&artifact.to_bytes().unwrap(), &called.declarations).unwrap();
        let (program, _) = crate::artifact_device::mapped_inlined(&decoded.program).unwrap();
        let cpu = program.execute(&family, false).unwrap();
        let reference = crate::derivatives::vjp_seeded(&program, &family, &cpu, seeds(&program), Some(&[0, 1])).unwrap();
        let mut devices = vec![Device::host()];
        if let Some(device) = Device::accelerator(gam_gpu::GpuPolicy::Auto).expect("device probe") {
            if device.float64() { devices.push(device); }
        }
        for device in devices {
            let lowered = DeviceProgram::compile_values(&device, &program).unwrap();
            let trace = lowered.forward(&family).unwrap();
            let values = device.download(trace.value(program.output).unwrap()).unwrap();
            assert!((&values - &cpu.values[program.output]).iter().all(|v| v.abs() < 2e-12));
            let upload = seeds(&program).into_iter().map(|(n, v)| (n, device.upload(v.view()).unwrap())).collect();
            let actual = lowered.vjp_values_seeded(&trace, upload, &[0, 1], Arithmetic::F64).unwrap();
            for node in [0, 1] {
                let cot = device.download(&actual[&node]).unwrap();
                assert!((&cot - reference[node].as_ref().unwrap()).iter().all(|v| v.abs() < 2e-12));
            }
        }
    }
    #[test]
    fn fixed_gain_tangents_match_cpu_and_external_parameters_are_refused() {
        use crate::operator_program::Coefficient;
        let (mut program, family) = fixture();
        program.rules.clear();
        program.nodes = vec![Node::Raw { slot: 0 }, Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Gain { input: 1, coefficient: Coefficient::Number(-0.75) },
            Node::Affine { terms: vec![(2, 0)], bias: None }];
        program.output = 3;
        let tangents = BTreeMap::from([(0, array![[0.1, 0.4], [-0.7, 0.2]])]);
        let mut prefix = program.clone(); prefix.output = 2; prefix.nodes.truncate(3);
        let reference = crate::derivatives::jvp(&prefix, &family, &prefix.execute(&family, false).unwrap(), &tangents).unwrap();
        let device = Device::host();
        let lowered = DeviceProgram::compile(&device, &program).unwrap();
        let trace = lowered.forward(&family).unwrap();
        let actual = device.download(&lowered.jvp(&trace, &tangents, Arithmetic::F64).unwrap().unwrap()).unwrap();
        assert!((&actual - reference).iter().all(|v| v.abs() < 2e-12));
        program.declarations.parameters = 1;
        program.nodes[2] = Node::Gain { input: 1, coefficient: Coefficient::Parameter(0) };
        assert!(DeviceProgram::compile(&device, &program).is_err());
    }
    #[test]
    fn nonlinear_rule_shared_dense_and_transposed_cotangents_match_cpu_and_finite_differences() {
        let (called, family) = fixture();
        let (program, _) = crate::artifact_device::mapped_inlined(&called).unwrap();
        let cpu = program.execute(&family, false).unwrap();
        assert_eq!(called.execute(&family, false).unwrap().values[called.output], cpu.values[program.output]);
        let reference = crate::derivatives::vjp_seeded(&program, &family, &cpu, seeds(&program), Some(&[0, 1])).unwrap();
        let mut devices = vec![Device::host()];
        if let Some(device) = Device::accelerator(gam_gpu::GpuPolicy::Auto).expect("device probe") {
            if device.float64() { devices.push(device); }
        }
        for device in devices {
            let lowered = DeviceProgram::compile_values(&device, &program).unwrap();
            let trace = lowered.forward(&family).unwrap();
            let upload_seeds = || seeds(&program).into_iter().map(|(node, value)| (node, device.upload(value.view()).unwrap())).collect();
            let nodes = lowered.vjp_values_seeded(&trace, upload_seeds(), &[0, 1], Arithmetic::F64).unwrap();
            for node in [0, 1] {
                let actual = device.download(&nodes[&node]).unwrap();
                assert!((&actual - reference[node].as_ref().unwrap()).iter().all(|v| v.abs() < 2e-12), "{} node {node}", device.name());
            }
            let (nodes, operators) = lowered.vjp_values_dense(&trace, upload_seeds(), &[0], &[0], Arithmetic::F64).unwrap();
            assert_eq!(nodes.len(), 1);
            let gradient = device.download(&operators[&0]).unwrap();
            for row in 0..2 { for col in 0..2 {
                let mut shifted = program.clone();
                let delta = 1e-6;
                let set = |p: &mut OperatorProgram, step| {
                    let OperatorBody::Dense { values, .. } = &mut Arc::make_mut(&mut p.operators[0]).body else { panic!("dense fixture") };
                    values[[row,col]] += step;
                };
                set(&mut shifted, delta);
                let plus = objective(&shifted, &family);
                set(&mut shifted, -2.0 * delta);
                let minus = objective(&shifted, &family);
                assert!((gradient[[row,col]] - (plus-minus)/(2.0*delta)).abs() < 2e-8, "{} shared [{row},{col}]", device.name());
            }}
            assert_eq!(device.download(trace.value(0).unwrap()).unwrap(), cpu.values[0]);
            assert_eq!(device.download(lowered.dense(0).unwrap()).unwrap(), program.operators[0].matrix());
            assert!(lowered.vjp_values_seeded(&trace, BTreeMap::from([(usize::MAX, device.zeros(2,2).unwrap())]), &[], Arithmetic::F64).is_err());
            assert!(lowered.vjp_values_seeded(&trace, BTreeMap::from([(program.output, device.zeros(1,2).unwrap())]), &[0], Arithmetic::F64).is_err());
            assert!(lowered.vjp_values_dense(&trace, upload_seeds(), &[], &[0,0], Arithmetic::F64).is_err());
            assert!(lowered.vjp(&trace, device.zeros(2,2).unwrap(), &[0], Arithmetic::F64).is_err());
        }
    }
    #[test]
    fn shared_nonlinear_columns_sum_all_bias_constant_and_product_uses() {
        let scalar = Interface::native(1).unwrap();
        let constant = Interface::constant();
        let pair = Interface::native(2).unwrap();
        let column = |name: &str, values: Array2<f64>| Arc::new(Operator::dense(
            name, pair.clone(), constant.clone(), values.clone(),
            exact_precision(values.iter().copied()).unwrap(), Default::default()).unwrap());
        let binding = Arc::new(Operator::dense("bind raw scalar", constant.clone(), scalar,
            array![[1.0]], exact_precision([1.0]).unwrap(), Default::default()).unwrap());
        let called = OperatorProgram {
            declarations: Declarations { domains: vec![], slots: vec![Slot::Raw { width: 1 }], parameters: 0 },
            bases: vec![],
            operators: vec![column("shared offset and reader", array![[0.3],[-0.4]]),
                column("constant only", array![[0.2],[0.7]]), binding],
            rules: vec![Rule { name: "learned offset in reused nonlinear body".into(), inputs: vec![constant],
                nodes: vec![Node::Param { index: 0 },
                    Node::Affine { terms: vec![(0,0)], bias: Some(0) },
                    Node::Pointwise { input: 1, laws: vec![Law::GeluTanh] }], output: 2 }],
            nodes: vec![Node::Raw { slot: 0 },
                Node::Affine { terms: vec![(0,2)], bias: None },
                Node::Call { rule: 0, arguments: vec![1] },
                Node::Transposed { input: 2, operator: 0 },
                Node::Call { rule: 0, arguments: vec![3] },
                Node::Constant { operator: 0 }, Node::Constant { operator: 1 },
                Node::Hadamard { left: 4, right: 5 },
                Node::Hadamard { left: 7, right: 6 }], output: 8,
        };
        let family = FamilyInputs { rows: 3, slots: vec![SlotValues::Raw(array![[-0.6],[0.2],[1.1]])], layout: None };
        let (program, _) = crate::artifact_device::mapped_inlined(&called).unwrap();
        let output_seed = array![[0.7,-0.2],[-0.4,0.9],[0.6,0.3]];
        let objective = |p: &OperatorProgram| {
            let values = p.execute(&family, false).unwrap();
            (&values.values[p.output] * &output_seed).sum()
        };
        let mut devices = vec![Device::host()];
        if let Some(device) = Device::accelerator(gam_gpu::GpuPolicy::Auto).unwrap() {
            if device.float64() { devices.push(device); }
        }
        for device in devices {
            let lowered = DeviceProgram::compile_values(&device, &program).unwrap();
            let trace = lowered.forward(&family).unwrap();
            let (_, gradients) = lowered.vjp_values_dense(&trace,
                BTreeMap::from([(program.output, device.upload(output_seed.view()).unwrap())]),
                &[], &[0,1], Arithmetic::F64).unwrap();
            for op in 0..2 {
                let actual = device.download(&gradients[&op]).unwrap();
                for row in 0..2 {
                    let mut moved = program.clone();
                    let delta = 1e-6;
                    let set = |p: &mut OperatorProgram, step: f64| {
                        let OperatorBody::Dense { values, .. } = &mut Arc::make_mut(&mut p.operators[op]).body else { panic!("dense") };
                        values[[row,0]] += step;
                    };
                    set(&mut moved, delta);
                    let plus = objective(&moved);
                    set(&mut moved, -2.0*delta);
                    let reference = (plus-objective(&moved))/(2.0*delta);
                    assert!((actual[[row,0]]-reference).abs() < 2e-9,
                        "{} op {op} row {row}: {} != {reference}", device.name(), actual[[row,0]]);
                }
            }
            assert_eq!(device.download(trace.value(0).unwrap()).unwrap(), array![[-0.6],[0.2],[1.1]]);
            let peer = DeviceProgram::compile_values_sharing(&lowered, &program).unwrap();
            let before = device.download(trace.value(program.output).unwrap()).unwrap();
            let mut trained = lowered;
            trained.prepare_dense_parameters(&[0,1]).unwrap();
            assert_eq!(device.download(trained.dense_parameter(1).unwrap()).unwrap(), program.operators[1].matrix());
            assert!(trained.dense_mut(0).is_err(), "must not leave bias and product copies inconsistent");
            let changed = array![[0.9],[-0.8]];
            trained.replace_dense_parameter(0, device.upload(changed.view()).unwrap()).unwrap();
            let changed_constant = array![[0.4],[0.1]];
            trained.replace_dense_parameter(1, device.upload(changed_constant.view()).unwrap()).unwrap();
            let mut reference = program.clone();
            for (op, value) in [(0, changed), (1, changed_constant)] {
                let OperatorBody::Dense { values, .. } = &mut Arc::make_mut(&mut reference.operators[op]).body else { panic!("dense") };
                *values = value;
            }
            let expected = reference.execute(&family, false).unwrap();
            let after = trained.forward(&family).unwrap();
            let actual = device.download(after.value(program.output).unwrap()).unwrap();
            assert!((&actual - &expected.values[program.output]).iter().all(|v| v.abs()<2e-12));
            let peer_trace = peer.forward(&family).unwrap();
            assert_eq!(device.download(peer_trace.value(program.output).unwrap()).unwrap(), before);
            assert_ne!(actual, before);
            // Compiling the original host program must not borrow trained values
            // just because its old parameter Arcs still have the same addresses.
            let original = DeviceProgram::compile_values_sharing(&trained, &program).unwrap();
            let original_trace = original.forward(&family).unwrap();
            assert_eq!(device.download(original_trace.value(program.output).unwrap()).unwrap(), before);
            trained.refresh(&program).unwrap();
            let restored = trained.forward(&family).unwrap();
            assert_eq!(device.download(restored.value(program.output).unwrap()).unwrap(), before);
        }
    }

    #[test]
    fn dense_zero_literals_have_full_matrix_cotangents_despite_diagonal_execution() {
        let device = Device::host();
        let interface = Interface::native(2).unwrap();
        let values = Array2::zeros((2,2));
        let operator = Operator::dense("zero coefficient", interface.clone(), interface, values.clone(),
            exact_precision(values.iter().copied()).unwrap(), Default::default()).unwrap();
        let program = OperatorProgram { declarations: Declarations { domains: vec![], slots: vec![Slot::Raw { width: 2 }], parameters: 0 },
            operators: vec![Arc::new(operator)], bases: vec![], rules: vec![],
            nodes: vec![Node::Raw { slot: 0 }, Node::Affine { terms: vec![(0,0)], bias: None }], output: 1 };
        let family = FamilyInputs { rows: 1, slots: vec![SlotValues::Raw(array![[2.,3.]])], layout: None };
        let lowered = DeviceProgram::compile_values(&device,&program).unwrap();
        let trace = lowered.forward(&family).unwrap();
        let (_, gradients) = lowered.vjp_values_dense(&trace, BTreeMap::from([(1,device.upload(array![[5.,7.]].view()).unwrap())]), &[], &[0], Arithmetic::F64).unwrap();
        assert_eq!(device.download(&gradients[&0]).unwrap(),array![[10.,15.],[14.,21.]]);
        let mut trained = lowered;
        trained.prepare_dense_parameters(&[0]).unwrap();
        let moved = device.upload(array![[0.,2.],[3.,0.]].view()).unwrap();
        trained.replace_dense_parameter(0,moved).unwrap();
        let trace = trained.forward(&family).unwrap();
        assert_eq!(device.download(trace.value(1).unwrap()).unwrap(),array![[6.,6.]]);
    }

    #[test]
    fn concatenated_cotangents_accumulate_repeated_parts_and_storage_refusals() {
        let (called, family) = fixture();
        let (mut program, _) = crate::artifact_device::mapped_inlined(&called).unwrap();
        let device = Device::host();
        program.nodes.push(Node::Concat { parts: vec![program.output, 0, 0] });
        program.output = program.nodes.len()-1;
        let lowered = DeviceProgram::compile_values(&device, &program).unwrap();
        let trace = lowered.forward(&family).unwrap();
        let seeds = BTreeMap::from([(program.output, Array2::ones((2,6)))]);
        let cpu = program.execute(&family,false).unwrap();
        let reference = crate::derivatives::vjp_seeded(&program,&family,&cpu,seeds.clone(),Some(&[0])).unwrap();
        let actual = lowered.vjp_values_seeded(&trace,seeds.into_iter().map(|(n,a)|(n,device.upload(a.view()).unwrap())).collect(),&[0],Arithmetic::F64).unwrap();
        assert!((&device.download(&actual[&0]).unwrap()-reference[0].as_ref().unwrap()).iter().all(|v|v.abs()<2e-12));
        let interface = Interface::native(2).unwrap();
        program.operators[0] = Arc::new(Operator::identity("unsupported identity", interface));
        program.nodes.pop(); program.output = program.nodes.len()-1;
        let lowered = DeviceProgram::compile_values(&device, &program).unwrap();
        // Refuse unsupported trainable storage before any reverse pass.
        let trace = lowered.forward(&family).unwrap();
        assert!(lowered.vjp_values_dense(&trace, BTreeMap::new(), &[0], &[0], Arithmetic::F64).is_err());
    }
}

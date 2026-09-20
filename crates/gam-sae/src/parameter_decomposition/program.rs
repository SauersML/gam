//! The mechanism program (#2951): a typed graph of sums, compositions, native
//! primitives, reads and writes, and calls to shared bodies, executed against
//! its native reference.
//!
//! # The graph
//!
//! A [`Program`] is a set of [`Body`]s and an entry body. A body is a DAG whose
//! nodes are listed in topological order: a node may only read nodes listed
//! before it. Values are row-valued matrices (rows are positions, columns are
//! widths). A body has input ports, one output node, and may read and write the
//! program's named state slots (a residual stream, a cache).
//!
//! * [`Node::Native`] runs one of the source's primitives on the source's
//!   tensors, bound through a [`ParameterSource`].
//! * [`Node::Sum`] is `y = sum_i m_i x_i`: mechanisms that add.
//! * [`Node::Compose`] applies stage bodies in sequence, each stage deleted along
//!   the straight path `x -> x + m (f(x) - x)`. For linear stages
//!   `f_k = I + Delta_k` the output is `prod_k (I + m_k Delta_k) x`, so the cross
//!   term `m_A m_B Delta_B Delta_A` is induced by execution and has no control of
//!   its own.
//! * [`Node::Call`] invokes a shared body. Every call site is an invocation, and a
//!   control inside the body can be set at every invocation or at one.
//! * [`Node::Refine`] pairs a mechanism body with the native body it decomposes.
//!   When every control the mechanism reaches resolves to exactly `1` at this
//!   invocation, the native body runs on its original path, so the all-on program
//!   is bit-identical to the source; otherwise the mechanism runs.
//!
//! # Controls, mask groups and macros
//!
//! A control is one masking site: a sum term, a composition stage, one component
//! coordinate of a [`NativePrimitive::CoordinateMask`], or the component anchor
//! of a parameter. Every control belongs to exactly one [`MaskGroup`], and a
//! group is the unit of assignment: tied controls take one value. A macro is a
//! different object, a called body whose internal controls stay independent: a
//! [`MaskAssignment`] scoped to one invocation changes that call only. A group
//! ties controls; a call shares structure without tying anything.
//!
//! A mask value is any finite real (continuous, binary or signed). An unassigned
//! group is `1`, the identity intervention, not a declared mask domain: the
//! domain an experiment searches over is declared by that experiment.

use gam_math::gaussian_activation::{
    GaussianActivation, GaussianActivationError, gaussian_smoothing_derivatives,
};
use gam_math::gaussian_gated::silu_derivatives;
use super::codec::{
    BitReader, BitString, CodecError, DagNode, decode_fixed_index, decode_ordered_dag,
    decode_prefix_integer, encode_fixed_index, encode_ordered_dag, encode_prefix_integer,
};
use super::gated_rewrite::{GatedRewriteError, MaskedNorm, swiglu_hidden};
use super::attention::{
    AttentionGeometry, AttentionProgramError, ProjectedRows, RotaryCausalAttention, RotaryEmbedding,
    RotaryPairing, head_rms_norm,
};
use super::precision::{DecodableArtifact, DeclaredPrecision, LatticeCode};
use super::supports::{EvidenceStatus, EvidenceStatusError, ExactBasis};
use super::apply::{ApplyError, native_linear};
use super::lift::{
    AnchorMask, LiftError, ResidualAnchor, StorageTensor, TensorId, TensorRegistry, TieOrientation, UseMap,
    UseSiteId,
};
use gam_linalg::roundoff::accumulation_growth;
use gam_runtime::resource::{Governed, MemoryGovernor, MemoryReservation, MemoryReservationError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// The serialized schema tag of a [`Program`] document.
pub const MECHANISM_PROGRAM_SCHEMA: &str = "gamfit.MechanismProgram/v1";

/// Index of a body in the program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BodyId(pub u32);

/// Index of a node inside its body.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

/// Index of a named state slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SlotId(pub u32);

/// Index of a formal parameter: a tensor the program reads, bound to the source's
/// tensors by a [`ParameterSource`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ParameterSlot(pub u32);

/// Index of a control (one masking site).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ControlId(pub u32);

/// Index of a mask group (tied controls).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MaskGroupId(pub u32);

impl BodyId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl NodeId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl SlotId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl ParameterSlot {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl ControlId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl MaskGroupId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// One step of an invocation path: the call site `node` in `body`, and for a
/// composition the stage index (`0` for a call or a refinement).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CallSite {
    pub body: BodyId,
    pub node: NodeId,
    pub stage: u32,
}

/// The source's elementwise activation, evaluated by the gam-math owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NativeActivation {
    /// `max(t, 0)`.
    Relu,
    /// The exact GELU `t Phi(t)`.
    ExactGelu,
    /// SiLU (swish) `t sigma(t)`, the gate of a SwiGLU block.
    Silu,
}

impl NativeActivation {
    /// The gam-math owner's activation. This match and the one in
    /// [`Self::from_owner`] are exhaustive both ways, so an activation the owner adds
    /// fails to compile until it has a node.
    pub fn to_owner(self) -> GaussianActivation {
        match self {
            Self::Relu => GaussianActivation::Relu,
            Self::ExactGelu => GaussianActivation::ExactGelu,
            Self::Silu => GaussianActivation::Silu,
        }
    }

    /// The node for an owner activation, e.g. one named by
    /// `GaussianActivation::from_hidden_act`.
    pub fn from_owner(owner: GaussianActivation) -> Self {
        match owner {
            GaussianActivation::Relu => Self::Relu,
            GaussianActivation::ExactGelu => Self::ExactGelu,
            GaussianActivation::Silu => Self::Silu,
        }
    }

    /// The activation's value at `t`, from its gam-math owner: the zero-variance
    /// Gaussian smoothing for ReLU and GELU, and the SiLU jet for SiLU, whose
    /// closed-form smoothing entries refuse it. A non-finite `t` is refused.
    fn forward(self, t: f64) -> Result<f64, GaussianActivationError> {
        match self.to_owner() {
            GaussianActivation::Silu => {
                if !t.is_finite() {
                    return Err(GaussianActivationError::NonFiniteArgument { value: t });
                }
                Ok(silu_derivatives(t)[0])
            }
            owner @ (GaussianActivation::Relu | GaussianActivation::ExactGelu) => {
                let mut value = [0.0];
                gaussian_smoothing_derivatives(owner, t, 0.0, &mut value)?;
                Ok(value[0])
            }
        }
    }
}

/// A primitive of the source network.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NativePrimitive {
    /// Rows mapped through a parameter matrix: `y = x A^T`, with `A = Theta` for an
    /// identity use and `A = Theta^T` for a transposed one (`lift::TieOrientation`).
    /// The orientation is the map the consuming operation applies, and it must equal
    /// the registered use site's `UseMap::Linear` orientation ([`Program::bind_use_sites`]).
    Linear { weight: ParameterSlot, orientation: TieOrientation },
    /// A parameter vector added to every row.
    AddBias { bias: ParameterSlot },
    /// The source's elementwise activation.
    Activation { activation: NativeActivation },
    /// The elementwise product of two values of one shape.
    Hadamard,
    /// A diagonal mask on component coordinates: column `c` is multiplied by the
    /// value of `controls[c]`. By the mask gauge (P1) only a group mask `cI` on a
    /// declared group, not independent diagonal entries in an arbitrary basis, is
    /// an intrinsic intervention; tie the coordinates with a [`MaskGroup`].
    CoordinateMask { controls: Vec<ControlId> },
    /// `Qwen3RMSNorm`, `gain ⊙ (h (mean(h^2) + epsilon)^(-1/2))` per row, evaluated by
    /// `gated_rewrite::MaskedNorm`. `epsilon` is the source configuration's value.
    RmsNorm { epsilon: f64, gain: ParameterSlot },
    /// The source's per-head query/key norm (Qwen3 `q_norm`, `k_norm`): the RMS norm of
    /// [`Self::RmsNorm`] on each contiguous `head_dim` block of a row, with one gain of
    /// length `head_dim` shared by every head, evaluated by `attention::head_rms_norm`, the
    /// owner [`super::attention::NativeAttention::with_query_key_norm`] runs. `epsilon` is the
    /// source configuration's value.
    HeadRmsNorm {
        head_dim: usize,
        epsilon: f64,
        gain: ParameterSlot,
    },
    /// `torch.nn.LayerNorm` over the population variance, with gain and bias,
    /// evaluated by `gated_rewrite::MaskedNorm`.
    LayerNorm {
        epsilon: f64,
        gain: ParameterSlot,
        bias: ParameterSlot,
    },
    /// The source's causal rotary self-attention on already-projected rows
    /// `(queries, keys, values)`, evaluated by `attention::RotaryCausalAttention` at
    /// the execution's absolute positions: rotate, score, mask causally, softmax
    /// jointly, read the values. The output is the head-mixed rows before the
    /// output projection, which is a separate Linear node.
    CausalSelfAttention {
        geometry: AttentionGeometry,
        rotary: RotaryEmbedding,
        score_scale: f64,
    },
    /// A SwiGLU gate `s(gate) ⊙ up`, with `s` the SiLU, evaluated by
    /// `gated_rewrite::swiglu_hidden` on the rows `(gate, up)`.
    SwiGlu,
}

impl NativePrimitive {
    fn arity(&self) -> usize {
        match self {
            Self::CausalSelfAttention { .. } => 3,
            Self::Hadamard | Self::SwiGlu => 2,
            Self::Linear { .. }
            | Self::AddBias { .. }
            | Self::Activation { .. }
            | Self::CoordinateMask { .. }
            | Self::RmsNorm { .. }
            | Self::HeadRmsNorm { .. }
            | Self::LayerNorm { .. } => 1,
        }
    }

    /// The formal parameters this primitive reads.
    pub fn parameters(&self) -> Vec<ParameterSlot> {
        self.parameter_maps().into_iter().map(|(parameter, _)| parameter).collect()
    }

    /// Each formal parameter this primitive reads, in [`Self::parameters`] order, with
    /// what the node does with it: a Linear node multiplies by its matrix in its
    /// orientation, and every other reader uses the stored values.
    pub fn parameter_maps(&self) -> Vec<(ParameterSlot, UseMap)> {
        match self {
            Self::Linear { weight, orientation } => vec![(*weight, UseMap::Linear(*orientation))],
            Self::AddBias { bias: parameter }
            | Self::RmsNorm { gain: parameter, .. }
            | Self::HeadRmsNorm { gain: parameter, .. } => vec![(*parameter, UseMap::Stored)],
            Self::LayerNorm { gain, bias, .. } => vec![(*gain, UseMap::Stored), (*bias, UseMap::Stored)],
            Self::Activation { .. }
            | Self::Hadamard
            | Self::CoordinateMask { .. }
            | Self::CausalSelfAttention { .. }
            | Self::SwiGlu => Vec::new(),
        }
    }
}

/// One term of a [`Node::Sum`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SumTerm {
    pub value: NodeId,
    pub control: Option<ControlId>,
}

/// One stage of a [`Node::Compose`]: a one-input body and its deletion control.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComposeStage {
    pub body: BodyId,
    pub control: Option<ControlId>,
}

/// A node of a body.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Node {
    /// The body's input `port`.
    Input { port: u32 },
    /// The current value of a state slot.
    Read { slot: SlotId },
    /// Stores `value` in a state slot; the node's value is `value`.
    Write { slot: SlotId, value: NodeId },
    /// A source primitive applied to `arguments`.
    Native {
        primitive: NativePrimitive,
        arguments: Vec<NodeId>,
    },
    /// `sum_i m_i x_i` over the terms, in term order.
    Sum { terms: Vec<SumTerm> },
    /// The stages applied in order to `value`, stage `k` as
    /// `x -> x + m_k (f_k(x) - x)`.
    Compose {
        value: NodeId,
        stages: Vec<ComposeStage>,
    },
    /// An invocation of a shared body.
    Call {
        body: BodyId,
        arguments: Vec<NodeId>,
    },
    /// A mechanism body and the native body it decomposes, on the same arguments.
    Refine {
        native: BodyId,
        mechanism: BodyId,
        arguments: Vec<NodeId>,
    },
}

impl Node {
    /// The nodes this node reads, in argument order.
    pub fn arguments(&self) -> Vec<NodeId> {
        match self {
            Self::Input { .. } | Self::Read { .. } => Vec::new(),
            Self::Write { value, .. } | Self::Compose { value, .. } => vec![*value],
            Self::Native { arguments, .. }
            | Self::Call { arguments, .. }
            | Self::Refine { arguments, .. } => arguments.clone(),
            Self::Sum { terms } => terms.iter().map(|term| term.value).collect(),
        }
    }
}

/// A DAG of nodes with input ports and one output node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Body {
    pub name: String,
    pub inputs: u32,
    pub nodes: Vec<Node>,
    pub output: NodeId,
}

/// A formal parameter and the controls of its component anchor. A parameter with
/// controls is affine in them (the anchor `m_Delta Theta_* + B sum_c (m_c -
/// m_Delta) v_c`); the binding applies it. [`LiftSource`] reads them in the order
/// `[m_Delta, m_1, ..., m_C]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterDecl {
    pub name: String,
    pub controls: Vec<ControlId>,
}

/// A named state slot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlotDecl {
    pub name: String,
}

/// A named control.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlDecl {
    pub name: String,
}

/// Controls tied to one assigned value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskGroup {
    pub name: String,
    pub controls: Vec<ControlId>,
}

/// The unvalidated parts of a program; [`Program::new`] validates them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramParts {
    pub parameters: Vec<ParameterDecl>,
    pub slots: Vec<SlotDecl>,
    pub controls: Vec<ControlDecl>,
    pub mask_groups: Vec<MaskGroup>,
    pub bodies: Vec<Body>,
    pub entry: BodyId,
}

/// A refused program, assignment, execution or document.
#[derive(Clone, Debug, PartialEq)]
pub enum ProgramError {
    UnknownBody { body: BodyId },
    UnknownSlot { slot: SlotId },
    UnknownParameter { parameter: ParameterSlot },
    UnknownControl { control: ControlId },
    UnknownMaskGroup { group: MaskGroupId },
    UnknownInputPort { body: BodyId, node: NodeId, port: u32 },
    UnknownOutput { body: BodyId, output: NodeId },
    /// A node reads a node that is not listed before it.
    ForwardReference { body: BodyId, node: NodeId, argument: NodeId },
    PrimitiveArity { body: BodyId, node: NodeId, expected: usize, found: usize },
    CallArity { body: BodyId, node: NodeId, callee: BodyId, expected: usize, found: usize },
    EmptySum { body: BodyId, node: NodeId },
    EmptyCompose { body: BodyId, node: NodeId },
    /// The call graph has a cycle through `body`.
    RecursiveCall { body: BodyId },
    /// A declared parameter no Linear or AddBias node reads.
    ParameterUnused { parameter: ParameterSlot },
    /// A declared slot no Read or Write node names.
    SlotUnused { slot: SlotId },
    ControlUnused { control: ControlId },
    ControlUsedTwice { control: ControlId },
    ControlUngrouped { control: ControlId },
    ControlInTwoGroups { control: ControlId },
    EmptyMaskGroup { group: MaskGroupId },
    /// A composition stage or refinement body writes a state slot, so deleting the
    /// stage or choosing the native path would change which writes happen.
    ImpureBody { body: BodyId, node: NodeId, callee: BodyId },
    /// A refinement's native body reaches a control or another refinement.
    NativeReferenceNotNative { body: BodyId, node: NodeId, native: BodyId },
    InputCount { body: BodyId, expected: usize, found: usize },
    SlotCount { expected: usize, found: usize },
    SlotUnwritten { slot: SlotId },
    ShapeMismatch { body: BodyId, node: NodeId, left: (usize, usize), right: (usize, usize) },
    WidthMismatch { body: BodyId, node: NodeId, expected: usize, found: usize },
    RowCountChanged { body: BodyId, node: NodeId, expected: usize, found: usize },
    Activation { body: BodyId, node: NodeId, error: GaussianActivationError },
    /// A normalization or SwiGLU node refused by its owner in `gated_rewrite`.
    GatedRewrite { body: BodyId, node: NodeId, error: GatedRewriteError },
    /// An attention node refused by its owner, `attention::RotaryCausalAttention`.
    Attention { body: BodyId, node: NodeId, error: AttentionProgramError },
    /// A refinement residual has no evidence status, e.g. a non-finite output entry.
    Evidence { body: BodyId, node: NodeId, error: EvidenceStatusError },
    /// The real constants have no code at the declared precision (`precision.rs`).
    Precision { message: String },
    NonFiniteMask { group: MaskGroupId, value: f64 },
    DuplicateMaskScope { group: MaskGroupId },
    /// An executor invariant: a value was released before its last use.
    ValueReleased { body: BodyId, node: NodeId },
    SchemaMismatch { found: String },
    Json { message: String },
    /// A program codeword could not be written or read.
    Codec { error: CodecError },
}

impl fmt::Display for ProgramError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownBody { body } => write!(formatter, "no body {}", body.0),
            Self::UnknownSlot { slot } => write!(formatter, "no state slot {}", slot.0),
            Self::UnknownParameter { parameter } => {
                write!(formatter, "no formal parameter {}", parameter.0)
            }
            Self::UnknownControl { control } => write!(formatter, "no control {}", control.0),
            Self::UnknownMaskGroup { group } => write!(formatter, "no mask group {}", group.0),
            Self::UnknownInputPort { body, node, port } => write!(
                formatter,
                "node {} of body {} reads input port {port}, which the body does not declare",
                node.0, body.0
            ),
            Self::UnknownOutput { body, output } => {
                write!(formatter, "body {} outputs missing node {}", body.0, output.0)
            }
            Self::ForwardReference { body, node, argument } => write!(
                formatter,
                "node {} of body {} reads node {}, which is not listed before it",
                node.0, body.0, argument.0
            ),
            Self::PrimitiveArity { body, node, expected, found } => write!(
                formatter,
                "native node {} of body {} takes {expected} arguments, got {found}",
                node.0, body.0
            ),
            Self::CallArity { body, node, callee, expected, found } => write!(
                formatter,
                "node {} of body {} passes {found} values to body {}, which takes {expected}",
                node.0, body.0, callee.0
            ),
            Self::EmptySum { body, node } => {
                write!(formatter, "sum node {} of body {} has no terms", node.0, body.0)
            }
            Self::EmptyCompose { body, node } => {
                write!(formatter, "compose node {} of body {} has no stages", node.0, body.0)
            }
            Self::RecursiveCall { body } => {
                write!(formatter, "the call graph has a cycle through body {}", body.0)
            }
            Self::ControlUnused { control } => {
                write!(formatter, "control {} is used at no site", control.0)
            }
            Self::ControlUsedTwice { control } => write!(
                formatter,
                "control {} is used at two sites; tie two sites with a mask group instead",
                control.0
            ),
            Self::ControlUngrouped { control } => {
                write!(formatter, "control {} belongs to no mask group", control.0)
            }
            Self::ControlInTwoGroups { control } => {
                write!(formatter, "control {} belongs to two mask groups", control.0)
            }
            Self::EmptyMaskGroup { group } => {
                write!(formatter, "mask group {} ties no controls", group.0)
            }
            Self::ImpureBody { body, node, callee } => write!(
                formatter,
                "node {} of body {} deletes or refines body {}, which writes a state slot",
                node.0, body.0, callee.0
            ),
            Self::NativeReferenceNotNative { body, node, native } => write!(
                formatter,
                "refinement node {} of body {} names body {} as native, but it reaches a control or a refinement",
                node.0, body.0, native.0
            ),
            Self::InputCount { body, expected, found } => write!(
                formatter,
                "body {} takes {expected} inputs, got {found}",
                body.0
            ),
            Self::SlotCount { expected, found } => write!(
                formatter,
                "the program declares {expected} state slots, got {found} initial values"
            ),
            Self::SlotUnwritten { slot } => {
                write!(formatter, "state slot {} is read before any value is written", slot.0)
            }
            Self::ShapeMismatch { body, node, left, right } => write!(
                formatter,
                "node {} of body {} combines shapes {left:?} and {right:?}",
                node.0, body.0
            ),
            Self::WidthMismatch { body, node, expected, found } => write!(
                formatter,
                "node {} of body {} needs width {expected}, got {found}",
                node.0, body.0
            ),
            Self::RowCountChanged { body, node, expected, found } => write!(
                formatter,
                "native node {} of body {} returned {found} rows for {expected} input rows",
                node.0, body.0
            ),
            Self::Activation { body, node, error } => write!(
                formatter,
                "activation node {} of body {}: {error}",
                node.0, body.0
            ),
            Self::GatedRewrite { body, node, error } => write!(
                formatter,
                "normalization or SwiGLU node {} of body {}: {error}",
                node.0, body.0
            ),
            Self::Attention { body, node, error } => write!(
                formatter,
                "attention node {} of body {}: {error}",
                node.0, body.0
            ),
            Self::Evidence { body, node, error } => write!(
                formatter,
                "refinement node {} of body {} has no residual status: {error:?}",
                node.0, body.0
            ),
            Self::Precision { message } => write!(formatter, "program real constants: {message}"),
            Self::NonFiniteMask { group, value } => {
                write!(formatter, "mask group {} was assigned non-finite {value}", group.0)
            }
            Self::DuplicateMaskScope { group } => write!(
                formatter,
                "mask group {} is assigned twice at one invocation scope",
                group.0
            ),
            Self::ValueReleased { body, node } => write!(
                formatter,
                "executor invariant: node {} of body {} was released before its last use",
                node.0, body.0
            ),
            Self::SchemaMismatch { found } => write!(
                formatter,
                "expected schema {MECHANISM_PROGRAM_SCHEMA}, found {found}"
            ),
            Self::Json { message } => write!(formatter, "program document: {message}"),
            Self::Codec { error } => write!(formatter, "program codeword: {error}"),
            Self::ParameterUnused { parameter } => {
                write!(formatter, "formal parameter {} is read by no node", parameter.0)
            }
            Self::SlotUnused { slot } => {
                write!(formatter, "state slot {} is named by no read or write", slot.0)
            }
        }
    }
}

impl std::error::Error for ProgramError {}

impl From<CodecError> for ProgramError {
    fn from(error: CodecError) -> Self {
        Self::Codec { error }
    }
}

/// A validated program.
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    parts: ProgramParts,
    control_group: Vec<MaskGroupId>,
    use_counts: Vec<Vec<u32>>,
    controls_below: Vec<Vec<ControlId>>,
}

impl Program {
    /// Validates the graph: ids in range, topological node order, arities, an
    /// acyclic call graph, every control used at exactly one site and in exactly
    /// one group, composition stages and refinements that write no slots, and
    /// native bodies that reach no control.
    pub fn new(parts: ProgramParts) -> Result<Self, ProgramError> {
        let body_count = parts.bodies.len();
        callee_body(&parts, parts.entry)?;
        let mut parameter_used = vec![false; parts.parameters.len()];
        let mut slot_used = vec![false; parts.slots.len()];
        let mut control_sites = vec![0u32; parts.controls.len()];
        for decl in &parts.parameters {
            for &control in &decl.controls {
                record_control_site(&mut control_sites, control)?;
            }
        }
        let mut use_counts = Vec::with_capacity(body_count);
        let mut callees: Vec<Vec<BodyId>> = Vec::with_capacity(body_count);
        let mut controls_below: Vec<Vec<ControlId>> = Vec::with_capacity(body_count);
        let mut writes_below = Vec::with_capacity(body_count);
        let mut refines_below = Vec::with_capacity(body_count);
        for (b, body) in parts.bodies.iter().enumerate() {
            let body_id = BodyId(b as u32);
            let mut counts = vec![0u32; body.nodes.len()];
            let mut body_callees = Vec::new();
            let mut controls = Vec::new();
            let mut writes = false;
            let mut refines = false;
            for (i, node) in body.nodes.iter().enumerate() {
                let node_id = NodeId(i as u32);
                for argument in node.arguments() {
                    if argument.index() >= i {
                        return Err(ProgramError::ForwardReference {
                            body: body_id,
                            node: node_id,
                            argument,
                        });
                    }
                    counts[argument.index()] += 1;
                }
                match node {
                    Node::Input { port } => {
                        if *port >= body.inputs {
                            return Err(ProgramError::UnknownInputPort {
                                body: body_id,
                                node: node_id,
                                port: *port,
                            });
                        }
                    }
                    Node::Read { slot } => {
                        check_slot(&parts, *slot)?;
                        slot_used[slot.index()] = true;
                    }
                    Node::Write { slot, .. } => {
                        check_slot(&parts, *slot)?;
                        slot_used[slot.index()] = true;
                        writes = true;
                    }
                    Node::Native { primitive, arguments } => {
                        if arguments.len() != primitive.arity() {
                            return Err(ProgramError::PrimitiveArity {
                                body: body_id,
                                node: node_id,
                                expected: primitive.arity(),
                                found: arguments.len(),
                            });
                        }
                        for parameter in primitive.parameters() {
                            let decl = parts
                                .parameters
                                .get(parameter.index())
                                .ok_or(ProgramError::UnknownParameter { parameter })?;
                            controls.extend(decl.controls.iter().copied());
                            parameter_used[parameter.index()] = true;
                        }
                        if let NativePrimitive::CoordinateMask { controls: mask } = primitive {
                            for &control in mask {
                                record_control_site(&mut control_sites, control)?;
                                controls.push(control);
                            }
                        }
                        if let NativePrimitive::CausalSelfAttention { geometry, rotary, score_scale } =
                            primitive
                        {
                            RotaryCausalAttention::new(*geometry, rotary.clone(), *score_scale).map_err(
                                |error| ProgramError::Attention { body: body_id, node: node_id, error },
                            )?;
                        }
                    }
                    Node::Sum { terms } => {
                        if terms.is_empty() {
                            return Err(ProgramError::EmptySum { body: body_id, node: node_id });
                        }
                        for control in terms.iter().filter_map(|term| term.control) {
                            record_control_site(&mut control_sites, control)?;
                            controls.push(control);
                        }
                    }
                    Node::Compose { stages, .. } => {
                        if stages.is_empty() {
                            return Err(ProgramError::EmptyCompose {
                                body: body_id,
                                node: node_id,
                            });
                        }
                        for stage in stages {
                            let inputs = callee_body(&parts, stage.body)?.inputs as usize;
                            if inputs != 1 {
                                return Err(ProgramError::CallArity {
                                    body: body_id,
                                    node: node_id,
                                    callee: stage.body,
                                    expected: inputs,
                                    found: 1,
                                });
                            }
                            body_callees.push(stage.body);
                            if let Some(control) = stage.control {
                                record_control_site(&mut control_sites, control)?;
                                controls.push(control);
                            }
                        }
                    }
                    Node::Call { body: callee, arguments } => {
                        let inputs = callee_body(&parts, *callee)?.inputs as usize;
                        if inputs != arguments.len() {
                            return Err(ProgramError::CallArity {
                                body: body_id,
                                node: node_id,
                                callee: *callee,
                                expected: inputs,
                                found: arguments.len(),
                            });
                        }
                        body_callees.push(*callee);
                    }
                    Node::Refine { native, mechanism, arguments } => {
                        for callee in [*native, *mechanism] {
                            let inputs = callee_body(&parts, callee)?.inputs as usize;
                            if inputs != arguments.len() {
                                return Err(ProgramError::CallArity {
                                    body: body_id,
                                    node: node_id,
                                    callee,
                                    expected: inputs,
                                    found: arguments.len(),
                                });
                            }
                            body_callees.push(callee);
                        }
                        refines = true;
                    }
                }
            }
            let output_count = counts.get_mut(body.output.index()).ok_or(
                ProgramError::UnknownOutput { body: body_id, output: body.output },
            )?;
            *output_count += 1;
            use_counts.push(counts);
            callees.push(body_callees);
            controls_below.push(controls);
            writes_below.push(writes);
            refines_below.push(refines);
        }
        if let Some(unused) = parameter_used.iter().position(|&used| !used) {
            return Err(ProgramError::ParameterUnused { parameter: ParameterSlot(unused as u32) });
        }
        if let Some(unused) = slot_used.iter().position(|&used| !used) {
            return Err(ProgramError::SlotUnused { slot: SlotId(unused as u32) });
        }
        if let Some(unused) = control_sites.iter().position(|&sites| sites == 0) {
            return Err(ProgramError::ControlUnused { control: ControlId(unused as u32) });
        }
        let mut group_of: Vec<Option<MaskGroupId>> = vec![None; parts.controls.len()];
        for (g, group) in parts.mask_groups.iter().enumerate() {
            let group_id = MaskGroupId(g as u32);
            if group.controls.is_empty() {
                return Err(ProgramError::EmptyMaskGroup { group: group_id });
            }
            for &control in &group.controls {
                let entry = group_of
                    .get_mut(control.index())
                    .ok_or(ProgramError::UnknownControl { control })?;
                if entry.is_some() {
                    return Err(ProgramError::ControlInTwoGroups { control });
                }
                *entry = Some(group_id);
            }
        }
        let control_group = group_of
            .iter()
            .enumerate()
            .map(|(c, group)| {
                group.ok_or(ProgramError::ControlUngrouped { control: ControlId(c as u32) })
            })
            .collect::<Result<Vec<_>, _>>()?;
        for body in call_graph_post_order(&callees)? {
            let b = body.index();
            for callee in callees[b].iter().map(|callee| callee.index()) {
                writes_below[b] = writes_below[b] || writes_below[callee];
                refines_below[b] = refines_below[b] || refines_below[callee];
                let below = controls_below[callee].clone();
                controls_below[b].extend(below);
            }
            controls_below[b].sort_unstable();
            controls_below[b].dedup();
        }
        for (b, body) in parts.bodies.iter().enumerate() {
            let body_id = BodyId(b as u32);
            for (i, node) in body.nodes.iter().enumerate() {
                let node_id = NodeId(i as u32);
                let pure_callees: Vec<BodyId> = match node {
                    Node::Compose { stages, .. } => stages.iter().map(|stage| stage.body).collect(),
                    Node::Refine { native, mechanism, .. } => {
                        if !controls_below[native.index()].is_empty()
                            || refines_below[native.index()]
                        {
                            return Err(ProgramError::NativeReferenceNotNative {
                                body: body_id,
                                node: node_id,
                                native: *native,
                            });
                        }
                        vec![*native, *mechanism]
                    }
                    Node::Input { .. }
                    | Node::Read { .. }
                    | Node::Write { .. }
                    | Node::Native { .. }
                    | Node::Sum { .. }
                    | Node::Call { .. } => Vec::new(),
                };
                if let Some(callee) = pure_callees.into_iter().find(|callee| writes_below[callee.index()]) {
                    return Err(ProgramError::ImpureBody { body: body_id, node: node_id, callee });
                }
            }
        }
        Ok(Self { parts, control_group, use_counts, controls_below })
    }

    /// The validated parts.
    pub fn parts(&self) -> &ProgramParts {
        &self.parts
    }

    /// The mask group a control belongs to.
    pub fn mask_group_of(&self, control: ControlId) -> Option<MaskGroupId> {
        self.control_group.get(control.index()).copied()
    }

    /// Every control a body reaches, directly or through the bodies it calls,
    /// composes or refines, in increasing order.
    pub fn controls_reached(&self, body: BodyId) -> Option<&[ControlId]> {
        self.controls_below.get(body.index()).map(Vec::as_slice)
    }
}

fn callee_body(parts: &ProgramParts, body: BodyId) -> Result<&Body, ProgramError> {
    parts.bodies.get(body.index()).ok_or(ProgramError::UnknownBody { body })
}

fn check_slot(parts: &ProgramParts, slot: SlotId) -> Result<(), ProgramError> {
    if slot.index() < parts.slots.len() {
        Ok(())
    } else {
        Err(ProgramError::UnknownSlot { slot })
    }
}

fn record_control_site(sites: &mut [u32], control: ControlId) -> Result<(), ProgramError> {
    let count = sites
        .get_mut(control.index())
        .ok_or(ProgramError::UnknownControl { control })?;
    if *count > 0 {
        return Err(ProgramError::ControlUsedTwice { control });
    }
    *count = 1;
    Ok(())
}

/// Callees before callers, refusing a cycle.
fn call_graph_post_order(callees: &[Vec<BodyId>]) -> Result<Vec<BodyId>, ProgramError> {
    const UNVISITED: u8 = 0;
    const ON_STACK: u8 = 1;
    let mut state = vec![UNVISITED; callees.len()];
    let mut order = Vec::with_capacity(callees.len());
    for root in 0..callees.len() {
        if state[root] != UNVISITED {
            continue;
        }
        state[root] = ON_STACK;
        let mut stack: Vec<(usize, usize)> = vec![(root, 0)];
        while let Some(frame) = stack.last_mut() {
            let (body, next) = *frame;
            if next < callees[body].len() {
                frame.1 += 1;
                let callee = callees[body][next].index();
                if state[callee] == ON_STACK {
                    return Err(ProgramError::RecursiveCall { body: BodyId(callee as u32) });
                }
                if state[callee] == UNVISITED {
                    state[callee] = ON_STACK;
                    stack.push((callee, 0));
                }
            } else {
                state[body] = ON_STACK + 1;
                order.push(BodyId(body as u32));
                stack.pop();
            }
        }
    }
    Ok(order)
}

/// A formal parameter at one use: the node that reads it, the invocation it runs
/// in, and the values its anchor controls resolve to there, in the order of
/// [`ParameterDecl::controls`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParameterUse<'a> {
    pub parameter: ParameterSlot,
    pub body: BodyId,
    pub node: NodeId,
    pub invocation: &'a [CallSite],
    pub controls: &'a [f64],
}

/// Binds formal parameters to the source's tensors. The rows handed to
/// [`ParameterSource::apply_linear`] are the current input of that use, after
/// every upstream intervention, and an implementation applies its tensor to them
/// in the node's orientation without materializing an edited copy, returning the
/// product under the memory reservation that accounts for it. At controls all `1`
/// it executes the original tensor.
pub trait ParameterSource {
    type Error: std::error::Error;

    /// `x A^T` for the matrix parameter at this use, with `A = Theta` for an
    /// identity use and `A = Theta^T` for a transposed one.
    fn apply_linear(
        &self,
        parameter: ParameterUse<'_>,
        orientation: TieOrientation,
        rows: ArrayView2<'_, f64>,
    ) -> Result<Governed<Array2<f64>>, Self::Error>;

    /// The vector parameter at this use.
    fn vector(&self, parameter: ParameterUse<'_>) -> Result<Array1<f64>, Self::Error>;
}

/// A refused execution: the program itself, the parameter binding at one use, or
/// the memory reservation of a value the execution forms.
#[derive(Debug)]
pub enum ExecutionError<E> {
    Program(ProgramError),
    Source {
        body: BodyId,
        node: NodeId,
        parameter: ParameterSlot,
        error: E,
    },
    /// A value's footprint does not fit the process memory budget.
    Memory { error: MemoryReservationError },
}

impl<E> From<ProgramError> for ExecutionError<E> {
    fn from(error: ProgramError) -> Self {
        Self::Program(error)
    }
}

impl<E: fmt::Display> fmt::Display for ExecutionError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Program(error) => write!(formatter, "{error}"),
            Self::Source { body, node, parameter, error } => write!(
                formatter,
                "parameter {} at node {} of body {}: {error}",
                parameter.0, node.0, body.0
            ),
            Self::Memory { error } => write!(formatter, "mechanism program value: {error}"),
        }
    }
}

impl<E: std::error::Error> std::error::Error for ExecutionError<E> {}

/// One assigned group value at an invocation scope. The empty scope is global. A
/// scope applies at every invocation path it prefixes, and the deepest applicable
/// scope wins.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScopedMask {
    pub group: MaskGroupId,
    pub scope: Vec<CallSite>,
    pub value: f64,
}

/// Mask group values by invocation scope. A group with no applicable entry is
/// `1`, the identity intervention.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskAssignment {
    entries: Vec<ScopedMask>,
}

impl MaskAssignment {
    /// No intervention.
    pub fn all_on() -> Self {
        Self { entries: Vec::new() }
    }

    /// Assigns `value` to `group` at `scope`, refusing a non-finite value and a
    /// second value at the same scope.
    pub fn set(
        &mut self,
        group: MaskGroupId,
        scope: Vec<CallSite>,
        value: f64,
    ) -> Result<(), ProgramError> {
        if !value.is_finite() {
            return Err(ProgramError::NonFiniteMask { group, value });
        }
        if self
            .entries
            .iter()
            .any(|entry| entry.group == group && entry.scope == scope)
        {
            return Err(ProgramError::DuplicateMaskScope { group });
        }
        self.entries.push(ScopedMask { group, scope, value });
        Ok(())
    }

    pub fn entries(&self) -> &[ScopedMask] {
        &self.entries
    }
}

/// The entry body's output, under the memory reservation that accounts for it, and
/// the final state slots.
#[derive(Debug)]
pub struct Execution {
    pub output: Governed<Array2<f64>>,
    pub slots: Vec<Option<Array2<f64>>>,
}

/// The status of a residual figure at one evaluated invocation: the witness is the
/// `(row, column)` of the extremal entry and the domain is the invocation path.
pub type ResidualStatus = EvidenceStatus<(usize, usize), Vec<CallSite>>;

/// At one refinement invocation, the mechanism body against the native body on
/// the same arguments.
#[derive(Clone, Debug, PartialEq)]
pub struct RefinementResidual {
    pub invocation: Vec<CallSite>,
    /// Every control the mechanism reaches resolved to `1`, so the native output
    /// was carried downstream.
    pub all_on: bool,
    /// `max |mechanism - native|` over every entry of the two computed outputs at
    /// this invocation: Exact over the one-member family, witnessed by its entry. Its
    /// numerical error is `gamma_1 * value`, the rounding of the one subtraction that
    /// forms each entry, since `abs` and `max` are exact. It says nothing about how
    /// far either computed output is from its real function.
    pub difference: ResidualStatus,
    /// `max |native|` over every entry, the scale of the difference: Exact, with no
    /// rounding.
    pub native_scale: ResidualStatus,
}

impl Program {
    /// Runs the entry body on `inputs` with state slots starting at `slots`.
    pub fn execute<S: ParameterSource>(
        &self,
        source: &S,
        masks: &MaskAssignment,
        inputs: Vec<Array2<f64>>,
        slots: Vec<Option<Array2<f64>>>,
        positions: &[i64],
    ) -> Result<Execution, ExecutionError<S::Error>> {
        let mut executor = Executor::new(self, source, masks, slots, positions, false)?;
        let output = executor.run(inputs)?;
        Ok(Execution { output, slots: executor.slots })
    }

    /// Runs like [`Program::execute`], and at every refinement invocation also runs
    /// the body that was not carried downstream, reporting the residual between the
    /// mechanism and its native reference.
    pub fn refinement_residuals<S: ParameterSource>(
        &self,
        source: &S,
        masks: &MaskAssignment,
        inputs: Vec<Array2<f64>>,
        slots: Vec<Option<Array2<f64>>>,
        positions: &[i64],
    ) -> Result<(Execution, Vec<RefinementResidual>), ExecutionError<S::Error>> {
        let mut executor = Executor::new(self, source, masks, slots, positions, true)?;
        let output = executor.run(inputs)?;
        let residuals = executor.residuals.take().unwrap_or_default();
        Ok((Execution { output, slots: executor.slots }, residuals))
    }
}

/// A node's value in the executor, under the memory reservation that accounts for it:
/// a Linear product under the reservation of the source that formed it, and every
/// other value under one the executor takes before the value is formed. Values are
/// moved and read in place, never copied out of their reservations.
type NodeValue = Governed<Array2<f64>>;

/// Reserves the footprint of a `rows x cols` value the executor is about to form.
fn admit<E>(rows: usize, cols: usize) -> Result<MemoryReservation, ExecutionError<E>> {
    MemoryGovernor::global()
        .try_reserve_dense_f64(rows, cols, "mechanism program value")
        .map_err(|error| ExecutionError::Memory { error })
}

/// A copy of `rows` under a reservation taken before the copy is made.
fn governed_copy<E>(rows: &Array2<f64>) -> Result<NodeValue, ExecutionError<E>> {
    Ok(admit(rows.nrows(), rows.ncols())?.bind(rows.clone()))
}

struct Executor<'a, S> {
    program: &'a Program,
    source: &'a S,
    masks: BTreeMap<MaskGroupId, Vec<(&'a [CallSite], f64)>>,
    slots: Vec<Option<Array2<f64>>>,
    /// The reservation of each slot's executor-written value, held while the executor
    /// owns it; the caller's own starting values and the final slots it receives back
    /// are its own.
    slot_reservations: Vec<Option<MemoryReservation>>,
    positions: &'a [i64],
    residuals: Option<Vec<RefinementResidual>>,
}

impl<'a, S: ParameterSource> Executor<'a, S> {
    fn new(
        program: &'a Program,
        source: &'a S,
        masks: &'a MaskAssignment,
        slots: Vec<Option<Array2<f64>>>,
        positions: &'a [i64],
        record_residuals: bool,
    ) -> Result<Self, ProgramError> {
        if slots.len() != program.parts.slots.len() {
            return Err(ProgramError::SlotCount {
                expected: program.parts.slots.len(),
                found: slots.len(),
            });
        }
        let mut index: BTreeMap<MaskGroupId, Vec<(&'a [CallSite], f64)>> = BTreeMap::new();
        for entry in &masks.entries {
            if entry.group.index() >= program.parts.mask_groups.len() {
                return Err(ProgramError::UnknownMaskGroup { group: entry.group });
            }
            if !entry.value.is_finite() {
                return Err(ProgramError::NonFiniteMask { group: entry.group, value: entry.value });
            }
            let scoped = index.entry(entry.group).or_default();
            if scoped.iter().any(|assigned| assigned.0 == entry.scope.as_slice()) {
                return Err(ProgramError::DuplicateMaskScope { group: entry.group });
            }
            scoped.push((entry.scope.as_slice(), entry.value));
        }
        let slot_reservations = std::iter::repeat_with(|| None).take(slots.len()).collect();
        Ok(Self {
            program,
            source,
            masks: index,
            slots,
            slot_reservations,
            positions,
            residuals: record_residuals.then(Vec::new),
        })
    }

    fn run(&mut self, inputs: Vec<Array2<f64>>) -> Result<Governed<Array2<f64>>, ExecutionError<S::Error>> {
        // The caller's inputs are moved under reservations for the run, not copied.
        let inputs = inputs
            .into_iter()
            .map(|rows| Ok(admit(rows.nrows(), rows.ncols())?.bind(rows)))
            .collect::<Result<Vec<_>, ExecutionError<S::Error>>>()?;
        let mut path = Vec::new();
        self.body(self.program.parts.entry, inputs, &mut path)
    }

    /// The value of a control at an invocation path: its group's deepest
    /// applicable scoped value, or `1`.
    fn resolve(&self, control: ControlId, path: &[CallSite]) -> f64 {
        let group = self.program.control_group[control.index()];
        let mut value = 1.0;
        let mut deepest: Option<usize> = None;
        if let Some(scoped) = self.masks.get(&group) {
            for &(scope, assigned) in scoped {
                if path.starts_with(scope) && deepest.is_none_or(|depth| scope.len() > depth) {
                    value = assigned;
                    deepest = Some(scope.len());
                }
            }
        }
        value
    }

    /// Whether every control `body_id` reaches resolves to exactly `1` where it runs:
    /// a node's own controls at `path`, and a called, composed or refined body's at the
    /// path its invocation pushes, so a scope set on a deeper invocation is seen.
    fn all_on_below(&self, body_id: BodyId, path: &mut Vec<CallSite>) -> bool {
        let program = self.program;
        let within = |site: CallSite, callee: BodyId, path: &mut Vec<CallSite>| {
            path.push(site);
            let reached = self.all_on_below(callee, path);
            path.pop();
            reached
        };
        program.parts.bodies[body_id.index()].nodes.iter().enumerate().all(|(i, node)| {
            let site = CallSite { body: body_id, node: NodeId(i as u32), stage: 0 };
            match node {
                Node::Input { .. } | Node::Read { .. } | Node::Write { .. } => true,
                Node::Native { primitive, .. } => {
                    let coordinates: &[ControlId] = match primitive {
                        NativePrimitive::CoordinateMask { controls } => controls,
                        _ => &[],
                    };
                    coordinates.iter().all(|&control| self.resolve(control, path) == 1.0)
                        && primitive.parameters().iter().all(|parameter| {
                            program.parts.parameters[parameter.index()]
                                .controls
                                .iter()
                                .all(|&control| self.resolve(control, path) == 1.0)
                        })
                }
                Node::Sum { terms } => terms
                    .iter()
                    .all(|term| term.control.is_none_or(|control| self.resolve(control, path) == 1.0)),
                Node::Compose { stages, .. } => stages.iter().enumerate().all(|(k, stage)| {
                    stage.control.is_none_or(|control| self.resolve(control, path) == 1.0)
                        && within(CallSite { stage: k as u32, ..site }, stage.body, path)
                }),
                Node::Call { body: callee, .. } | Node::Refine { mechanism: callee, .. } => {
                    within(site, *callee, path)
                }
            }
        })
    }

    fn parameter_controls(&self, parameter: ParameterSlot, path: &[CallSite]) -> Vec<f64> {
        self.program.parts.parameters[parameter.index()]
            .controls
            .iter()
            .map(|&control| self.resolve(control, path))
            .collect()
    }

    fn parameter_vector(
        &self,
        parameter: ParameterSlot,
        site: CallSite,
        path: &[CallSite],
    ) -> Result<Array1<f64>, ExecutionError<S::Error>> {
        let controls = self.parameter_controls(parameter, path);
        let parameter_use = ParameterUse {
            parameter,
            body: site.body,
            node: site.node,
            invocation: path,
            controls: &controls,
        };
        self.source.vector(parameter_use).map_err(|error| ExecutionError::Source {
            body: site.body,
            node: site.node,
            parameter,
            error,
        })
    }

    fn body(
        &mut self,
        body_id: BodyId,
        inputs: Vec<NodeValue>,
        path: &mut Vec<CallSite>,
    ) -> Result<NodeValue, ExecutionError<S::Error>> {
        let program = self.program;
        let body = &program.parts.bodies[body_id.index()];
        if inputs.len() != body.inputs as usize {
            return Err(ProgramError::InputCount {
                body: body_id,
                expected: body.inputs as usize,
                found: inputs.len(),
            }
            .into());
        }
        let mut remaining = program.use_counts[body_id.index()].clone();
        let mut values: Vec<Option<NodeValue>> =
            std::iter::repeat_with(|| None).take(body.nodes.len()).collect();
        for (i, node) in body.nodes.iter().enumerate() {
            let node_id = NodeId(i as u32);
            let site = CallSite { body: body_id, node: node_id, stage: 0 };
            let value = match node {
                Node::Input { port } => governed_copy(&inputs[*port as usize])?,
                Node::Read { slot } => governed_copy(
                    self.slots[slot.index()]
                        .as_ref()
                        .ok_or(ProgramError::SlotUnwritten { slot: *slot })?,
                )?,
                Node::Write { slot, value } => {
                    let written = governed_copy(argument(&values, body_id, *value)?)?;
                    let reservation = admit(written.nrows(), written.ncols())?;
                    self.slots[slot.index()] = Some((*written).clone());
                    self.slot_reservations[slot.index()] = Some(reservation);
                    written
                }
                Node::Native { primitive, arguments } => {
                    self.native(site, primitive, arguments, &values, path)?
                }
                Node::Sum { terms } => self.sum(site, terms, &values, path)?,
                Node::Compose { value, stages } => {
                    let mut current = governed_copy(argument(&values, body_id, *value)?)?;
                    for (k, stage) in stages.iter().enumerate() {
                        let m = stage
                            .control
                            .map_or(1.0, |control| self.resolve(control, path));
                        if m == 0.0 {
                            continue;
                        }
                        let stage_input = governed_copy(&current)?;
                        path.push(CallSite { stage: k as u32, ..site });
                        let staged = self.body(stage.body, vec![stage_input], path);
                        path.pop();
                        let mut staged = staged?;
                        if staged.dim() != current.dim() {
                            return Err(ProgramError::ShapeMismatch {
                                body: body_id,
                                node: node_id,
                                left: current.dim(),
                                right: staged.dim(),
                            }
                            .into());
                        }
                        if m != 1.0 {
                            *staged -= &*current;
                            *staged *= m;
                            *staged += &*current;
                        }
                        current = staged;
                    }
                    current
                }
                Node::Call { body: callee, arguments } => {
                    let passed = collect_arguments(&values, body_id, arguments)?;
                    path.push(site);
                    let called = self.body(*callee, passed, path);
                    path.pop();
                    called?
                }
                Node::Refine { native, mechanism, arguments } => {
                    let passed = collect_arguments(&values, body_id, arguments)?;
                    path.push(site);
                    let refined = self.refine(site, *native, *mechanism, passed, path);
                    path.pop();
                    refined?
                }
            };
            for argument in node.arguments() {
                let count = &mut remaining[argument.index()];
                *count -= 1;
                if *count == 0 {
                    values[argument.index()] = None;
                }
            }
            if remaining[i] > 0 {
                values[i] = Some(value);
            }
        }
        values[body.output.index()].take().ok_or(ExecutionError::Program(
            ProgramError::ValueReleased { body: body_id, node: body.output },
        ))
    }

    fn native(
        &self,
        site: CallSite,
        primitive: &NativePrimitive,
        arguments: &[NodeId],
        values: &[Option<NodeValue>],
        path: &[CallSite],
    ) -> Result<NodeValue, ExecutionError<S::Error>> {
        let (body, node) = (site.body, site.node);
        let x = argument(values, body, arguments[0])?;
        // Every primitive but Linear forms a value of its first argument's shape, reserved
        // before it is formed: by the executor, or by the owner the primitive calls.
        let formed: Result<(MemoryReservation, Array2<f64>), ExecutionError<S::Error>> = match primitive {
            NativePrimitive::Linear { weight, orientation } => {
                let controls = self.parameter_controls(*weight, path);
                let parameter_use = ParameterUse {
                    parameter: *weight,
                    body,
                    node,
                    invocation: path,
                    controls: &controls,
                };
                let rows = self
                    .source
                    .apply_linear(parameter_use, *orientation, x.view())
                    .map_err(|error| ExecutionError::Source { body, node, parameter: *weight, error })?;
                if rows.nrows() != x.nrows() {
                    return Err(ProgramError::RowCountChanged {
                        body,
                        node,
                        expected: x.nrows(),
                        found: rows.nrows(),
                    }
                    .into());
                }
                return Ok(rows);
            }
            NativePrimitive::AddBias { bias } => {
                let vector = self.parameter_vector(*bias, site, path)?;
                if vector.len() != x.ncols() {
                    return Err(ProgramError::WidthMismatch {
                        body,
                        node,
                        expected: x.ncols(),
                        found: vector.len(),
                    }
                    .into());
                }
                let reservation = admit(x.nrows(), x.ncols())?;
                let mut rows = x.clone();
                rows += &vector;
                Ok((reservation, rows))
            }
            NativePrimitive::SwiGlu => {
                let up = argument(values, body, arguments[1])?;
                let reservation = admit(x.nrows(), x.ncols())?;
                swiglu_hidden(x.view(), up.view()).map(|rows| (reservation, rows)).map_err(|error| {
                    ExecutionError::Program(ProgramError::GatedRewrite { body, node, error })
                })
            }
            NativePrimitive::CausalSelfAttention { geometry, rotary, score_scale } => {
                let attention_error =
                    |error| ExecutionError::Program(ProgramError::Attention { body, node, error });
                let keys = argument(values, body, arguments[1])?;
                let value_rows = argument(values, body, arguments[2])?;
                let attention = RotaryCausalAttention::new(*geometry, rotary.clone(), *score_scale)
                    .map_err(attention_error)?;
                let reservation = admit(x.nrows(), x.ncols())?;
                let attended = attention
                    .attend_projected(
                        ProjectedRows::exact(x.view()),
                        ProjectedRows::exact(keys.view()),
                        ProjectedRows::exact(value_rows.view()),
                        self.positions,
                    )
                    .map_err(attention_error)?;
                Ok((reservation, attended.mixed))
            }
            NativePrimitive::RmsNorm { epsilon, gain } => {
                let gain_vector = self.parameter_vector(*gain, site, path)?;
                let reservation = admit(x.nrows(), x.ncols())?;
                MaskedNorm::Rms { epsilon: *epsilon, gain: gain_vector.view() }
                    .apply(x.view())
                    .map(|rows| (reservation, rows))
                    .map_err(|error| ExecutionError::Program(ProgramError::GatedRewrite { body, node, error }))
            }
            NativePrimitive::HeadRmsNorm { head_dim, epsilon, gain } => {
                let gain_vector = self.parameter_vector(*gain, site, path)?;
                let reservation = admit(x.nrows(), x.ncols())?;
                head_rms_norm(x.view(), *head_dim, *epsilon, gain_vector.view())
                    .map(|rows| (reservation, rows))
                    .map_err(|error| ExecutionError::Program(ProgramError::GatedRewrite { body, node, error }))
            }
            NativePrimitive::LayerNorm { epsilon, gain, bias } => {
                let gain_vector = self.parameter_vector(*gain, site, path)?;
                let bias_vector = self.parameter_vector(*bias, site, path)?;
                let reservation = admit(x.nrows(), x.ncols())?;
                MaskedNorm::Layer {
                    epsilon: *epsilon,
                    gain: gain_vector.view(),
                    bias: bias_vector.view(),
                }
                .apply(x.view())
                .map(|rows| (reservation, rows))
                .map_err(|error| ExecutionError::Program(ProgramError::GatedRewrite { body, node, error }))
            }
            NativePrimitive::Activation { activation } => {
                let reservation = admit(x.nrows(), x.ncols())?;
                let mut rows = x.clone();
                for entry in rows.iter_mut() {
                    *entry = activation
                        .forward(*entry)
                        .map_err(|error| ProgramError::Activation { body, node, error })?;
                }
                Ok((reservation, rows))
            }
            NativePrimitive::Hadamard => {
                let y = argument(values, body, arguments[1])?;
                if x.dim() != y.dim() {
                    return Err(ProgramError::ShapeMismatch {
                        body,
                        node,
                        left: x.dim(),
                        right: y.dim(),
                    }
                    .into());
                }
                let reservation = admit(x.nrows(), x.ncols())?;
                Ok((reservation, x * y))
            }
            NativePrimitive::CoordinateMask { controls } => {
                if x.ncols() != controls.len() {
                    return Err(ProgramError::WidthMismatch {
                        body,
                        node,
                        expected: controls.len(),
                        found: x.ncols(),
                    }
                    .into());
                }
                let reservation = admit(x.nrows(), x.ncols())?;
                let mut rows = x.clone();
                for (c, &control) in controls.iter().enumerate() {
                    let m = self.resolve(control, path);
                    if m != 1.0 {
                        let mut column = rows.column_mut(c);
                        column *= m;
                    }
                }
                Ok((reservation, rows))
            }
        };
        let (reservation, rows) = formed?;
        // The reservation was taken for the first argument's shape; an owner that forms
        // another shape breaks that contract.
        if rows.dim() != x.dim() {
            return Err(ProgramError::ShapeMismatch { body, node, left: x.dim(), right: rows.dim() }.into());
        }
        Ok(reservation.bind(rows))
    }

    fn sum(
        &self,
        site: CallSite,
        terms: &[SumTerm],
        values: &[Option<NodeValue>],
        path: &[CallSite],
    ) -> Result<NodeValue, ExecutionError<S::Error>> {
        let first = argument(values, site.body, terms[0].value)?;
        let reservation = admit(first.nrows(), first.ncols())?;
        let mut total: Option<Array2<f64>> = None;
        for term in terms {
            let x = argument(values, site.body, term.value)?;
            if x.dim() != first.dim() {
                return Err(ProgramError::ShapeMismatch {
                    body: site.body,
                    node: site.node,
                    left: first.dim(),
                    right: x.dim(),
                }
                .into());
            }
            let m = term.control.map_or(1.0, |control| self.resolve(control, path));
            if m == 0.0 {
                continue;
            }
            total = Some(match total.take() {
                None if m == 1.0 => x.clone(),
                None => x * m,
                Some(mut accumulated) => {
                    if m == 1.0 {
                        accumulated += x;
                    } else {
                        accumulated.scaled_add(m, x);
                    }
                    accumulated
                }
            });
        }
        Ok(reservation.bind(total.unwrap_or_else(|| Array2::zeros(first.dim()))))
    }

    fn refine(
        &mut self,
        site: CallSite,
        native: BodyId,
        mechanism: BodyId,
        arguments: Vec<NodeValue>,
        path: &mut Vec<CallSite>,
    ) -> Result<NodeValue, ExecutionError<S::Error>> {
        let all_on = self.all_on_below(mechanism, path);
        if self.residuals.is_none() {
            let chosen = if all_on { native } else { mechanism };
            return self.body(chosen, arguments, path);
        }
        let native_arguments = arguments.iter().map(|rows| governed_copy(rows)).collect::<Result<Vec<_>, _>>()?;
        let native_output = self.body(native, native_arguments, path)?;
        let mechanism_output = self.body(mechanism, arguments, path)?;
        if native_output.dim() != mechanism_output.dim() {
            return Err(ProgramError::ShapeMismatch {
                body: site.body,
                node: site.node,
                left: native_output.dim(),
                right: mechanism_output.dim(),
            }
            .into());
        }
        let columns = native_output.ncols().max(1);
        let witness = |flat: usize| (flat / columns, flat % columns);
        let evidence_error =
            |error| ExecutionError::Program(ProgramError::Evidence { body: site.body, node: site.node, error });
        let (difference_value, difference_at) =
            extremal_entry(
                mechanism_output
                    .iter()
                    .zip(native_output.iter())
                    .map(|(mechanism_entry, native_entry)| (mechanism_entry - native_entry).abs()),
            );
        let (scale_value, scale_at) = extremal_entry(native_output.iter().map(|entry| entry.abs()));
        // The bound is rounded up, so the computed error band is never below the real
        // `gamma_1 * value`.
        let difference = ResidualStatus::exact(
            difference_value,
            (accumulation_growth(1) * difference_value).next_up(),
            ExactBasis::Exhaustive { cardinality: 1 },
            difference_at.map(witness),
            path.clone(),
        )
        .map_err(evidence_error)?;
        let native_scale = ResidualStatus::exact(
            scale_value,
            0.0,
            ExactBasis::Exhaustive { cardinality: 1 },
            scale_at.map(witness),
            path.clone(),
        )
        .map_err(evidence_error)?;
        if let Some(residuals) = self.residuals.as_mut() {
            residuals.push(RefinementResidual {
                invocation: path.clone(),
                all_on,
                difference,
                native_scale,
            });
        }
        Ok(if all_on { native_output } else { mechanism_output })
    }
}

fn argument<E>(
    values: &[Option<NodeValue>],
    body: BodyId,
    node: NodeId,
) -> Result<&Array2<f64>, ExecutionError<E>> {
    values[node.index()]
        .as_deref()
        .ok_or(ExecutionError::Program(ProgramError::ValueReleased { body, node }))
}

fn collect_arguments<E>(
    values: &[Option<NodeValue>],
    body: BodyId,
    arguments: &[NodeId],
) -> Result<Vec<NodeValue>, ExecutionError<E>> {
    arguments
        .iter()
        .map(|&node| governed_copy(argument(values, body, node)?))
        .collect()
}

/// The largest value and its flat position. The first NaN is returned with its
/// position, so the status constructor refuses it instead of hiding it; `(0, None)`
/// for no values.
fn extremal_entry(values: impl Iterator<Item = f64>) -> (f64, Option<usize>) {
    let mut largest = (0.0, None);
    for (position, value) in values.enumerate() {
        if value.is_nan() {
            return (value, Some(position));
        }
        if largest.1.is_none() || value > largest.0 {
            largest = (value, Some(position));
        }
    }
    largest
}

/// How a value depends on the mask values: a polynomial through linear operations,
/// or through a nonlinearity. Both degrees are upper bounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MaskDependence {
    /// A polynomial of total degree at most `degree` in the mask values, and of
    /// degree at most `group_degrees[g]` in the value of mask group `g`. A group ties
    /// its controls into one variable; an absent group has degree zero.
    Polynomial {
        degree: u32,
        group_degrees: BTreeMap<MaskGroupId, u32>,
    },
    /// A mask-dependent value reaches a nonlinearity.
    Nonlinear,
}

impl MaskDependence {
    const FREE: Self = Self::Polynomial { degree: 0, group_degrees: BTreeMap::new() };

    /// One mask group's value, to the first power.
    fn group(group: MaskGroupId) -> Self {
        Self::Polynomial { degree: 1, group_degrees: BTreeMap::from([(group, 1)]) }
    }

    /// The total degree bound, or `None` through a nonlinearity.
    pub fn degree(&self) -> Option<u32> {
        match self {
            Self::Polynomial { degree, .. } => Some(*degree),
            Self::Nonlinear => None,
        }
    }

    /// The degree bound in one group's value, or `None` through a nonlinearity.
    pub fn group_degree(&self, group: MaskGroupId) -> Option<u32> {
        match self {
            Self::Polynomial { group_degrees, .. } => Some(group_degrees.get(&group).copied().unwrap_or(0)),
            Self::Nonlinear => None,
        }
    }

    /// Affine in the mask values: total degree at most one.
    pub fn is_affine(&self) -> bool {
        matches!(self, Self::Polynomial { degree, .. } if *degree <= 1)
    }

    /// Multilinear in the mask groups: degree at most one in each group's value, with
    /// no nonlinear path. This is where the logit vertex adversary (P9, `moments.rs`)
    /// runs rather than refuses: a product of distinct groups, such as two masked
    /// layers or a composition's induced cross term, is inside, and a group read by
    /// two multiplied factors is outside. Inside, P9 is Exact only when every control
    /// moves at most one declared factor; a control moving two factors couples their
    /// zonotopes and P9 returns a UniformBound. Multilinear does not mean exact.
    pub fn is_multilinear(&self) -> bool {
        matches!(self, Self::Polynomial { group_degrees, .. } if group_degrees.values().all(|&degree| degree <= 1))
    }

    fn product(&self, other: &Self) -> Self {
        match (self, other) {
            (
                Self::Polynomial { degree: left, group_degrees: left_groups },
                Self::Polynomial { degree: right, group_degrees: right_groups },
            ) => {
                let mut group_degrees = left_groups.clone();
                for (&group, &degree) in right_groups {
                    let entry = group_degrees.entry(group).or_insert(0);
                    *entry = entry.saturating_add(degree);
                }
                Self::Polynomial { degree: left.saturating_add(*right), group_degrees }
            }
            (Self::Nonlinear, _) | (_, Self::Nonlinear) => Self::Nonlinear,
        }
    }

    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (
                Self::Polynomial { degree: left, group_degrees: left_groups },
                Self::Polynomial { degree: right, group_degrees: right_groups },
            ) => {
                let mut group_degrees = left_groups.clone();
                for (&group, &degree) in right_groups {
                    let entry = group_degrees.entry(group).or_insert(0);
                    *entry = (*entry).max(degree);
                }
                Self::Polynomial { degree: (*left).max(*right), group_degrees }
            }
            (Self::Nonlinear, _) | (_, Self::Nonlinear) => Self::Nonlinear,
        }
    }
}

impl Program {
    /// The mask dependence of the entry body's output, with inputs and initial
    /// slots mask-free. A refinement contributes its mechanism body: the native
    /// body it selects at all-on reaches no control.
    pub fn output_mask_dependence(&self) -> MaskDependence {
        let entry = self.parts.entry;
        let inputs = vec![MaskDependence::FREE; self.parts.bodies[entry.index()].inputs as usize];
        let mut slots = vec![MaskDependence::FREE; self.parts.slots.len()];
        self.body_mask_dependence(entry, &inputs, &mut slots)
    }

    fn body_mask_dependence(
        &self,
        body_id: BodyId,
        inputs: &[MaskDependence],
        slots: &mut [MaskDependence],
    ) -> MaskDependence {
        let body = &self.parts.bodies[body_id.index()];
        let mut values: Vec<MaskDependence> = Vec::with_capacity(body.nodes.len());
        for node in &body.nodes {
            let value = match node {
                Node::Input { port } => inputs[*port as usize].clone(),
                Node::Read { slot } => slots[slot.index()].clone(),
                Node::Write { slot, value } => {
                    slots[slot.index()] = values[value.index()].clone();
                    values[value.index()].clone()
                }
                Node::Native { primitive, arguments } => {
                    let x = values[arguments[0].index()].clone();
                    match primitive {
                        NativePrimitive::Linear { weight, .. } => {
                            x.product(&self.parameter_mask_dependence(*weight))
                        }
                        NativePrimitive::AddBias { bias } => {
                            x.join(&self.parameter_mask_dependence(*bias))
                        }
                        NativePrimitive::Activation { .. } => {
                            if x == MaskDependence::FREE { x } else { MaskDependence::Nonlinear }
                        }
                        NativePrimitive::Hadamard => x.product(&values[arguments[1].index()]),
                        NativePrimitive::CoordinateMask { controls } => {
                            if controls.is_empty() {
                                x
                            } else {
                                controls.iter().fold(MaskDependence::FREE, |total, control| {
                                    total.join(&self.controlled(&x, Some(*control)))
                                })
                            }
                        }
                        NativePrimitive::RmsNorm { gain, .. } | NativePrimitive::HeadRmsNorm { gain, .. } => {
                            if x == MaskDependence::FREE {
                                self.parameter_mask_dependence(*gain)
                            } else {
                                MaskDependence::Nonlinear
                            }
                        }
                        NativePrimitive::LayerNorm { gain, bias, .. } => {
                            if x == MaskDependence::FREE {
                                self.parameter_mask_dependence(*gain)
                                    .join(&self.parameter_mask_dependence(*bias))
                            } else {
                                MaskDependence::Nonlinear
                            }
                        }
                        NativePrimitive::CausalSelfAttention { .. } => {
                            // The weights are a softmax of the query-key scores; the
                            // values enter linearly under mask-free weights.
                            let keys = &values[arguments[1].index()];
                            if x == MaskDependence::FREE && *keys == MaskDependence::FREE {
                                values[arguments[2].index()].clone()
                            } else {
                                MaskDependence::Nonlinear
                            }
                        }
                        NativePrimitive::SwiGlu => {
                            // SiLU(gate) multiplies `up` elementwise: linear in `up`
                            // under a mask-free gate.
                            if x == MaskDependence::FREE {
                                values[arguments[1].index()].clone()
                            } else {
                                MaskDependence::Nonlinear
                            }
                        }
                    }
                }
                Node::Sum { terms } => terms.iter().fold(MaskDependence::FREE, |total, term| {
                    total.join(&self.controlled(&values[term.value.index()], term.control))
                }),
                Node::Compose { value, stages } => {
                    let mut current = values[value.index()].clone();
                    for stage in stages {
                        let staged =
                            self.body_mask_dependence(stage.body, std::slice::from_ref(&current), slots);
                        current = match stage.control {
                            Some(control) => self.controlled(&current.join(&staged), Some(control)),
                            None => staged,
                        };
                    }
                    current
                }
                Node::Call { body: callee, arguments }
                | Node::Refine { mechanism: callee, arguments, .. } => {
                    let passed: Vec<MaskDependence> =
                        arguments.iter().map(|argument| values[argument.index()].clone()).collect();
                    self.body_mask_dependence(*callee, &passed, slots)
                }
            };
            values.push(value);
        }
        values[body.output.index()].clone()
    }

    /// A parameter's anchor is affine in its controls jointly: every monomial has
    /// degree one, in one group's value.
    fn parameter_mask_dependence(&self, parameter: ParameterSlot) -> MaskDependence {
        self.parts.parameters[parameter.index()]
            .controls
            .iter()
            .fold(MaskDependence::FREE, |total, control| {
                total.join(&MaskDependence::group(self.control_group[control.index()]))
            })
    }

    /// A value multiplied by one control's group value, or unchanged.
    fn controlled(&self, value: &MaskDependence, control: Option<ControlId>) -> MaskDependence {
        match control {
            Some(control) => value.product(&MaskDependence::group(self.control_group[control.index()])),
            None => value.clone(),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProgramDocument {
    schema: String,
    program: ProgramParts,
}

fn json_error(error: serde_json::Error) -> ProgramError {
    ProgramError::Json { message: error.to_string() }
}

impl Program {
    /// The program as a [`MECHANISM_PROGRAM_SCHEMA`] JSON document.
    pub fn to_json(&self) -> Result<String, ProgramError> {
        let document = ProgramDocument {
            schema: MECHANISM_PROGRAM_SCHEMA.to_string(),
            program: self.parts.clone(),
        };
        serde_json::to_string(&document).map_err(json_error)
    }

    /// Reads a document, refusing any other schema tag, unknown fields and an
    /// invalid program. Older schemas are refused, not migrated.
    pub fn from_json(text: &str) -> Result<Self, ProgramError> {
        let value: serde_json::Value = serde_json::from_str(text).map_err(json_error)?;
        let found = value
            .get("schema")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if found != MECHANISM_PROGRAM_SCHEMA {
            return Err(ProgramError::SchemaMismatch { found: found.to_string() });
        }
        let document: ProgramDocument = serde_json::from_value(value).map_err(json_error)?;
        Self::new(document.program)
    }
}

/// A dense tensor bound to a formal parameter.
#[derive(Clone, Debug, PartialEq)]
pub enum DenseTensor {
    Matrix(Array2<f64>),
    Vector(Array1<f64>),
}

/// Formal parameters bound to dense tensors with no component structure, the
/// native binding of a small program such as a planted teacher. A parameter with
/// anchor controls needs a component binding and is refused here.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseParameters {
    tensors: Vec<DenseTensor>,
}

/// A refused dense parameter use.
#[derive(Clone, Debug, PartialEq)]
pub enum DenseParameterError {
    Unbound { parameter: ParameterSlot },
    NotAMatrix { parameter: ParameterSlot },
    NotAVector { parameter: ParameterSlot },
    /// The matrix-free owner refused the product, e.g. rows whose width is not the
    /// matrix's width in the node's orientation.
    Apply { parameter: ParameterSlot, error: ApplyError },
    /// A dense tensor has no component anchor for controls to act on.
    Controlled { parameter: ParameterSlot },
}

impl fmt::Display for DenseParameterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unbound { parameter } => {
                write!(formatter, "formal parameter {} has no dense tensor", parameter.0)
            }
            Self::NotAMatrix { parameter } => {
                write!(formatter, "formal parameter {} is bound to a vector, not a matrix", parameter.0)
            }
            Self::NotAVector { parameter } => {
                write!(formatter, "formal parameter {} is bound to a matrix, not a vector", parameter.0)
            }
            Self::Apply { parameter, error } => {
                write!(formatter, "formal parameter {}: {error}", parameter.0)
            }
            Self::Controlled { parameter } => write!(
                formatter,
                "formal parameter {} declares anchor controls, which a dense tensor cannot apply",
                parameter.0
            ),
        }
    }
}

impl std::error::Error for DenseParameterError {}

impl DenseParameters {
    /// Binds formal parameter `k` to `tensors[k]`.
    pub fn new(tensors: Vec<DenseTensor>) -> Self {
        Self { tensors }
    }

    fn tensor(&self, parameter_use: ParameterUse<'_>) -> Result<&DenseTensor, DenseParameterError> {
        let parameter = parameter_use.parameter;
        if !parameter_use.controls.is_empty() {
            return Err(DenseParameterError::Controlled { parameter });
        }
        self.tensors
            .get(parameter.index())
            .ok_or(DenseParameterError::Unbound { parameter })
    }
}

impl ParameterSource for DenseParameters {
    type Error = DenseParameterError;

    fn apply_linear(
        &self,
        parameter_use: ParameterUse<'_>,
        orientation: TieOrientation,
        rows: ArrayView2<'_, f64>,
    ) -> Result<Governed<Array2<f64>>, Self::Error> {
        let parameter = parameter_use.parameter;
        match self.tensor(parameter_use)? {
            DenseTensor::Matrix(matrix) => match orientation {
                TieOrientation::Identity => native_linear(matrix.view(), rows),
                TieOrientation::Transpose => native_linear(matrix.t(), rows),
            }
            .map_err(|error| DenseParameterError::Apply { parameter, error }),
            DenseTensor::Vector(..) => Err(DenseParameterError::NotAMatrix { parameter }),
        }
    }

    fn vector(&self, parameter_use: ParameterUse<'_>) -> Result<Array1<f64>, Self::Error> {
        let parameter = parameter_use.parameter;
        match self.tensor(parameter_use)? {
            DenseTensor::Vector(vector) => Ok(vector.clone()),
            DenseTensor::Matrix(..) => Err(DenseParameterError::NotAVector { parameter }),
        }
    }
}

/// The node-label alphabet of a program codeword: Input, Read, Write, Linear,
/// AddBias, Relu, ExactGelu, Silu, Hadamard, CoordinateMask, Sum, Compose, Call,
/// Refine, RmsNorm, LayerNorm, CausalSelfAttention, SwiGlu, transposed Linear and
/// HeadRmsNorm. A Linear node's orientation is its label, as an activation's kind is.
pub const PROGRAM_LABEL_ALPHABET: usize = 20;

fn node_label(node: &Node) -> usize {
    match node {
        Node::Input { .. } => 0,
        Node::Read { .. } => 1,
        Node::Write { .. } => 2,
        Node::Native { primitive, .. } => match primitive {
            NativePrimitive::Linear { orientation: TieOrientation::Identity, .. } => 3,
            NativePrimitive::Linear { orientation: TieOrientation::Transpose, .. } => 18,
            NativePrimitive::AddBias { .. } => 4,
            NativePrimitive::Activation { activation } => match activation {
                NativeActivation::Relu => 5,
                NativeActivation::ExactGelu => 6,
                NativeActivation::Silu => 7,
            },
            NativePrimitive::Hadamard => 8,
            NativePrimitive::CoordinateMask { .. } => 9,
            NativePrimitive::RmsNorm { .. } => 14,
            NativePrimitive::LayerNorm { .. } => 15,
            NativePrimitive::CausalSelfAttention { .. } => 16,
            NativePrimitive::SwiGlu => 17,
            NativePrimitive::HeadRmsNorm { .. } => 19,
        },
        Node::Sum { .. } => 10,
        Node::Compose { .. } => 11,
        Node::Call { .. } => 12,
        Node::Refine { .. } => 13,
    }
}

/// A primitive's real constants, in the order the program codeword carries them.
fn real_constants(primitive: &NativePrimitive) -> Vec<f64> {
    match primitive {
        NativePrimitive::RmsNorm { epsilon, .. }
        | NativePrimitive::HeadRmsNorm { epsilon, .. }
        | NativePrimitive::LayerNorm { epsilon, .. } => vec![*epsilon],
        NativePrimitive::CausalSelfAttention { rotary, score_scale, .. } => {
            let mut reals = rotary.inverse_frequencies.clone();
            reals.extend([rotary.attention_scaling, *score_scale]);
            reals
        }
        NativePrimitive::Linear { .. }
        | NativePrimitive::AddBias { .. }
        | NativePrimitive::Activation { .. }
        | NativePrimitive::Hadamard
        | NativePrimitive::CoordinateMask { .. }
        | NativePrimitive::SwiGlu => Vec::new(),
    }
}

fn precision_error(message: String) -> ProgramError {
    ProgramError::Precision { message }
}

/// The next real constant of the lattice code.
fn next_real(reals: &mut std::vec::IntoIter<f64>) -> Result<f64, CodecError> {
    reals.next().ok_or(CodecError::InvalidCodeword(
        "a node reads a real constant the lattice code does not carry".to_string(),
    ))
}

fn encode_optional_control(
    out: &mut BitString,
    control: Option<ControlId>,
    controls: usize,
) -> Result<(), CodecError> {
    out.push_bit(control.is_some());
    match control {
        Some(control) => encode_fixed_index(out, control.index(), controls),
        None => Ok(()),
    }
}

fn decode_optional_control(
    reader: &mut BitReader<'_>,
    controls: usize,
) -> Result<Option<ControlId>, CodecError> {
    if reader.read_bit()? {
        Ok(Some(ControlId(decode_fixed_index(reader, controls)? as u32)))
    } else {
        Ok(None)
    }
}

fn encode_control_list(
    out: &mut BitString,
    list: &[ControlId],
    controls: usize,
) -> Result<(), CodecError> {
    encode_prefix_integer(out, list.len() as u64 + 1)?;
    for control in list {
        encode_fixed_index(out, control.index(), controls)?;
    }
    Ok(())
}

/// A count of items that each spend at least one bit of the rest of the message
/// ([`Program::encode`]). A larger count is not the codeword of a valid program and
/// is refused before anything is allocated.
fn decode_count(reader: &mut BitReader<'_>) -> Result<usize, CodecError> {
    let count = decode_prefix_integer(reader)? - 1;
    if count > reader.remaining_bits() {
        return Err(CodecError::InvalidCodeword(format!(
            "{count} items cannot fit in {} remaining bits",
            reader.remaining_bits()
        )));
    }
    usize::try_from(count)
        .map_err(|error| CodecError::InvalidCodeword(format!("a count does not fit usize: {error}")))
}

/// An ordered control list. Every control sits at exactly one site, so no list is
/// longer than the declared controls, even where an entry costs zero bits.
fn decode_control_list(
    reader: &mut BitReader<'_>,
    controls: usize,
) -> Result<Vec<ControlId>, CodecError> {
    let length = decode_prefix_integer(reader)? - 1;
    if length > controls as u64 {
        return Err(CodecError::InvalidCodeword(format!(
            "a list of {length} controls exceeds the {controls} declared"
        )));
    }
    let mut list = Vec::with_capacity(length as usize);
    for _ in 0..length {
        list.push(ControlId(decode_fixed_index(reader, controls)? as u32));
    }
    Ok(list)
}

fn arity_refusal(label: usize, found: usize) -> CodecError {
    CodecError::InvalidCodeword(format!("node label {label} cannot take {found} arguments"))
}

impl Program {
    /// The program's codeword, through the mpd-codec encoders. It codes the
    /// executable structure only: names are not coded, and a decoded program has
    /// empty names.
    ///
    /// The header is the parameter, slot, control, group and body counts (each as
    /// count + 1 in the prefix integer code), the entry as a fixed index, each
    /// parameter's ordered anchor controls, and each control's group as a fixed
    /// index (the groups partition the controls). Each body is its input count + 1,
    /// its ordered DAG of node labels and arguments ([`encode_ordered_dag`]), its
    /// output as a fixed index, then each node's payload in node order.
    ///
    /// Validation makes every declared parameter, slot, control, group and body
    /// spend at least one bit, which bounds every count by the message length.
    pub fn encode(&self, precision: DeclaredPrecision) -> Result<BitString, ProgramError> {
        let parts = &self.parts;
        let mut out = BitString::new();
        let reals: Vec<f64> = parts
            .bodies
            .iter()
            .flat_map(|body| body.nodes.iter())
            .flat_map(|node| match node {
                Node::Native { primitive, .. } => real_constants(primitive),
                _ => Vec::new(),
            })
            .collect();
        LatticeCode::encode(&reals, precision)
            .map_err(precision_error)?
            .write(&mut out)
            .map_err(precision_error)?;
        for count in [
            parts.parameters.len(),
            parts.slots.len(),
            parts.controls.len(),
            parts.mask_groups.len(),
            parts.bodies.len(),
        ] {
            encode_prefix_integer(&mut out, count as u64 + 1)?;
        }
        encode_fixed_index(&mut out, parts.entry.index(), parts.bodies.len())?;
        for decl in &parts.parameters {
            encode_control_list(&mut out, &decl.controls, parts.controls.len())?;
        }
        for group in &self.control_group {
            encode_fixed_index(&mut out, group.index(), parts.mask_groups.len())?;
        }
        for body in &parts.bodies {
            encode_prefix_integer(&mut out, u64::from(body.inputs) + 1)?;
            let dag: Vec<DagNode> = body
                .nodes
                .iter()
                .map(|node| DagNode {
                    label: node_label(node),
                    arguments: node.arguments().iter().map(|argument| argument.index()).collect(),
                })
                .collect();
            encode_ordered_dag(&mut out, PROGRAM_LABEL_ALPHABET, &dag)?;
            encode_fixed_index(&mut out, body.output.index(), body.nodes.len())?;
            for node in &body.nodes {
                self.encode_payload(&mut out, body, node)?;
            }
        }
        Ok(out)
    }

    fn encode_payload(&self, out: &mut BitString, body: &Body, node: &Node) -> Result<(), CodecError> {
        let parts = &self.parts;
        let (controls, bodies) = (parts.controls.len(), parts.bodies.len());
        match node {
            Node::Input { port } => encode_fixed_index(out, *port as usize, body.inputs as usize),
            Node::Read { slot } | Node::Write { slot, .. } => {
                encode_fixed_index(out, slot.index(), parts.slots.len())
            }
            Node::Native { primitive, .. } => match primitive {
                NativePrimitive::Linear { weight: parameter, .. }
                | NativePrimitive::AddBias { bias: parameter } => {
                    encode_fixed_index(out, parameter.index(), parts.parameters.len())
                }
                NativePrimitive::CoordinateMask { controls: mask } => {
                    encode_control_list(out, mask, controls)
                }
                NativePrimitive::RmsNorm { gain, .. } => {
                    encode_fixed_index(out, gain.index(), parts.parameters.len())
                }
                NativePrimitive::HeadRmsNorm { head_dim, gain, .. } => {
                    encode_prefix_integer(out, *head_dim as u64 + 1)?;
                    encode_fixed_index(out, gain.index(), parts.parameters.len())
                }
                NativePrimitive::LayerNorm { gain, bias, .. } => {
                    encode_fixed_index(out, gain.index(), parts.parameters.len())?;
                    encode_fixed_index(out, bias.index(), parts.parameters.len())
                }
                NativePrimitive::CausalSelfAttention { geometry, rotary, .. } => {
                    for dimension in
                        [geometry.model_dim, geometry.n_heads, geometry.n_kv_heads, geometry.head_dim]
                    {
                        encode_prefix_integer(out, dimension as u64 + 1)?;
                    }
                    out.push_bit(rotary.pairing == RotaryPairing::Interleaved);
                    encode_prefix_integer(out, rotary.inverse_frequencies.len() as u64 + 1)
                }
                NativePrimitive::Activation { .. } | NativePrimitive::Hadamard | NativePrimitive::SwiGlu => {
                    Ok(())
                }
            },
            Node::Sum { terms } => {
                for term in terms {
                    encode_optional_control(out, term.control, controls)?;
                }
                Ok(())
            }
            Node::Compose { stages, .. } => {
                encode_prefix_integer(out, stages.len() as u64 + 1)?;
                for stage in stages {
                    encode_fixed_index(out, stage.body.index(), bodies)?;
                    encode_optional_control(out, stage.control, controls)?;
                }
                Ok(())
            }
            Node::Call { body: callee, .. } => encode_fixed_index(out, callee.index(), bodies),
            Node::Refine { native, mechanism, .. } => {
                encode_fixed_index(out, native.index(), bodies)?;
                encode_fixed_index(out, mechanism.index(), bodies)
            }
        }
    }

    /// Reads one codeword ([`Program::encode`]), refusing trailing bits, and
    /// validates the decoded program through [`Program::new`].
    pub fn decode(message: &BitString) -> Result<Self, ProgramError> {
        let mut reader = message.reader();
        let lattice = LatticeCode::read(&mut reader).map_err(precision_error)?;
        let mut reals = lattice.decode().map_err(precision_error)?.into_iter();
        let parameter_count = decode_count(&mut reader)?;
        let slot_count = decode_count(&mut reader)?;
        let control_count = decode_count(&mut reader)?;
        let group_count = decode_count(&mut reader)?;
        let body_count = decode_count(&mut reader)?;
        let entry = BodyId(decode_fixed_index(&mut reader, body_count)? as u32);
        let mut parameters = Vec::with_capacity(parameter_count);
        for _ in 0..parameter_count {
            let controls = decode_control_list(&mut reader, control_count)?;
            parameters.push(ParameterDecl { name: String::new(), controls });
        }
        let mut mask_groups: Vec<MaskGroup> =
            std::iter::repeat_with(|| MaskGroup { name: String::new(), controls: Vec::new() })
                .take(group_count)
                .collect();
        for control in 0..control_count {
            let group = decode_fixed_index(&mut reader, group_count)?;
            mask_groups[group].controls.push(ControlId(control as u32));
        }
        let mut bodies = Vec::with_capacity(body_count);
        for _ in 0..body_count {
            let inputs = u32::try_from(decode_prefix_integer(&mut reader)? - 1).map_err(|error| {
                CodecError::InvalidCodeword(format!("a body input count does not fit u32: {error}"))
            })?;
            let dag = decode_ordered_dag(&mut reader, PROGRAM_LABEL_ALPHABET)?;
            let output = NodeId(decode_fixed_index(&mut reader, dag.len())? as u32);
            let mut nodes = Vec::with_capacity(dag.len());
            for dag_node in dag {
                let arguments: Vec<NodeId> =
                    dag_node.arguments.iter().map(|&argument| NodeId(argument as u32)).collect();
                nodes.push(decode_node(
                    &mut reader,
                    &mut reals,
                    dag_node.label,
                    arguments,
                    inputs as usize,
                    [parameter_count, slot_count, control_count, body_count],
                )?);
            }
            bodies.push(Body { name: String::new(), inputs, nodes, output });
        }
        if reals.next().is_some() {
            return Err(ProgramError::Codec {
                error: CodecError::InvalidCodeword(
                    "the lattice code carries real constants no node reads".to_string(),
                ),
            });
        }
        reader.finish()?;
        let slots = std::iter::repeat_with(|| SlotDecl { name: String::new() })
            .take(slot_count)
            .collect();
        let controls = std::iter::repeat_with(|| ControlDecl { name: String::new() })
            .take(control_count)
            .collect();
        Self::new(ProgramParts { parameters, slots, controls, mask_groups, bodies, entry })
    }
}

fn decode_node(
    reader: &mut BitReader<'_>,
    reals: &mut std::vec::IntoIter<f64>,
    label: usize,
    arguments: Vec<NodeId>,
    inputs: usize,
    [parameters, slots, controls, bodies]: [usize; 4],
) -> Result<Node, CodecError> {
    let single = |arguments: &[NodeId]| match arguments {
        [value] => Ok(*value),
        other => Err(arity_refusal(label, other.len())),
    };
    let none = |arguments: &[NodeId]| {
        if arguments.is_empty() { Ok(()) } else { Err(arity_refusal(label, arguments.len())) }
    };
    let native = |primitive: NativePrimitive, arguments: Vec<NodeId>| Node::Native { primitive, arguments };
    Ok(match label {
        0 => {
            none(&arguments)?;
            Node::Input { port: decode_fixed_index(reader, inputs)? as u32 }
        }
        1 => {
            none(&arguments)?;
            Node::Read { slot: SlotId(decode_fixed_index(reader, slots)? as u32) }
        }
        2 => {
            let value = single(&arguments)?;
            Node::Write { slot: SlotId(decode_fixed_index(reader, slots)? as u32), value }
        }
        3 | 18 => {
            let orientation = if label == 3 { TieOrientation::Identity } else { TieOrientation::Transpose };
            let weight = ParameterSlot(decode_fixed_index(reader, parameters)? as u32);
            native(NativePrimitive::Linear { weight, orientation }, arguments)
        }
        4 => native(
            NativePrimitive::AddBias { bias: ParameterSlot(decode_fixed_index(reader, parameters)? as u32) },
            arguments,
        ),
        5 => native(NativePrimitive::Activation { activation: NativeActivation::Relu }, arguments),
        6 => native(NativePrimitive::Activation { activation: NativeActivation::ExactGelu }, arguments),
        7 => native(NativePrimitive::Activation { activation: NativeActivation::Silu }, arguments),
        8 => native(NativePrimitive::Hadamard, arguments),
        9 => native(
            NativePrimitive::CoordinateMask { controls: decode_control_list(reader, controls)? },
            arguments,
        ),
        10 => {
            let mut terms = Vec::with_capacity(arguments.len());
            for &value in &arguments {
                terms.push(SumTerm { value, control: decode_optional_control(reader, controls)? });
            }
            Node::Sum { terms }
        }
        11 => {
            let value = single(&arguments)?;
            let stage_count = decode_count(reader)?;
            let mut stages = Vec::with_capacity(stage_count);
            for _ in 0..stage_count {
                let body = BodyId(decode_fixed_index(reader, bodies)? as u32);
                stages.push(ComposeStage { body, control: decode_optional_control(reader, controls)? });
            }
            Node::Compose { value, stages }
        }
        12 => Node::Call { body: BodyId(decode_fixed_index(reader, bodies)? as u32), arguments },
        17 => native(NativePrimitive::SwiGlu, arguments),
        16 => {
            let mut dimensions = [0_usize; 4];
            for dimension in &mut dimensions {
                *dimension = usize::try_from(decode_prefix_integer(reader)? - 1).map_err(|error| {
                    CodecError::InvalidCodeword(format!("an attention dimension does not fit usize: {error}"))
                })?;
            }
            let [model_dim, n_heads, n_kv_heads, head_dim] = dimensions;
            let pairing = if reader.read_bit()? { RotaryPairing::Interleaved } else { RotaryPairing::HalfSplit };
            let frequency_count = decode_prefix_integer(reader)? - 1;
            // Every frequency is a real of the lattice code, so no count above the
            // reals that remain is a codeword.
            if frequency_count > reals.len() as u64 {
                return Err(CodecError::InvalidCodeword(format!(
                    "{frequency_count} rotary frequencies exceed the {} real constants left",
                    reals.len()
                )));
            }
            let mut inverse_frequencies = Vec::with_capacity(frequency_count as usize);
            for _ in 0..frequency_count {
                inverse_frequencies.push(next_real(reals)?);
            }
            let attention_scaling = next_real(reals)?;
            let score_scale = next_real(reals)?;
            native(
                NativePrimitive::CausalSelfAttention {
                    geometry: AttentionGeometry { model_dim, n_heads, n_kv_heads, head_dim },
                    rotary: RotaryEmbedding { pairing, inverse_frequencies, attention_scaling },
                    score_scale,
                },
                arguments,
            )
        }
        14 => {
            let epsilon = next_real(reals)?;
            let gain = ParameterSlot(decode_fixed_index(reader, parameters)? as u32);
            native(NativePrimitive::RmsNorm { epsilon, gain }, arguments)
        }
        15 => {
            let epsilon = next_real(reals)?;
            let gain = ParameterSlot(decode_fixed_index(reader, parameters)? as u32);
            let bias = ParameterSlot(decode_fixed_index(reader, parameters)? as u32);
            native(NativePrimitive::LayerNorm { epsilon, gain, bias }, arguments)
        }
        19 => {
            let head_dim = usize::try_from(decode_prefix_integer(reader)? - 1).map_err(|error| {
                CodecError::InvalidCodeword(format!("a head dimension does not fit usize: {error}"))
            })?;
            let epsilon = next_real(reals)?;
            let gain = ParameterSlot(decode_fixed_index(reader, parameters)? as u32);
            native(NativePrimitive::HeadRmsNorm { head_dim, epsilon, gain }, arguments)
        }
        13 => {
            let native_body = BodyId(decode_fixed_index(reader, bodies)? as u32);
            let mechanism = BodyId(decode_fixed_index(reader, bodies)? as u32);
            Node::Refine { native: native_body, mechanism, arguments }
        }
        other => return Err(CodecError::InvalidCodeword(format!("no node label {other}"))),
    })
}

/// One read of a formal parameter on the teacher's path, bound to its registered use
/// site.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundRead {
    pub parameter: ParameterSlot,
    pub body: BodyId,
    pub node: NodeId,
    pub invocation: Vec<CallSite>,
    pub site: UseSiteId,
    pub map: UseMap,
}

/// A refused binding of the program's reads to a tensor registry.
#[derive(Clone, Debug, PartialEq)]
pub enum BindingError {
    /// A declared parameter's name, or a registered site's name, is not registered.
    Registry { parameter: ParameterSlot, error: LiftError },
    /// A read on the teacher's path has no registered use site at its ordinal.
    UnregisteredRead { parameter: ParameterSlot, body: BodyId, node: NodeId, site: UseSiteId },
    /// The registered site reads different storage than the parameter names.
    StorageMismatch { site: UseSiteId, declared: TensorId, registered: TensorId },
    /// The registered site does something else with the parameter than the node does: a
    /// Linear node in the other orientation, a Linear node on a stored read, or a stored
    /// reader on a linear site.
    MapMismatch { site: UseSiteId, declared: UseMap, registered: UseMap },
    /// A registered use site of a storage the program reads is reached by no read, so a
    /// global edit of that storage would reach a site the program does not execute.
    UnreadSite { storage: TensorId, site: UseSiteId },
}

impl fmt::Display for BindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Registry { parameter, error } => {
                write!(formatter, "formal parameter {}: {error}", parameter.0)
            }
            Self::UnregisteredRead { parameter, body, node, site } => write!(
                formatter,
                "formal parameter {} read at node {} of body {} has no registered use site {}",
                parameter.0, node.0, body.0, site.0
            ),
            Self::StorageMismatch { site, declared, registered } => write!(
                formatter,
                "use site {} reads {}, not the declared storage {}",
                site.0, registered.0, declared.0
            ),
            Self::MapMismatch { site, declared, registered } => write!(
                formatter,
                "use site {} is registered as {registered:?}, but its node does {declared:?}",
                site.0
            ),
            Self::UnreadSite { storage, site } => write!(
                formatter,
                "use site {} of {} is reached by no read of the program",
                site.0, storage.0
            ),
        }
    }
}

impl std::error::Error for BindingError {}

impl Program {
    /// Binds every parameter read on the teacher's path to its registered use site.
    ///
    /// The teacher's path is the all-on execution: every composition stage, the native
    /// body of every refinement, and every call. A declared parameter names storage or an
    /// alias of it, and the `k`-th read of a storage tensor along the path, in execution
    /// order, binds to `UseSiteId::read(storage, k)`, which must be registered, read that
    /// storage and do what the node does. Every registered site of a storage the program
    /// reads must be reached, so a global edit of that storage reaches exactly the bound
    /// reads. Two reads of one storage with the same map and swapped ordinals pass every
    /// check here; only an end-to-end receipt against the teacher (A12) detects them. A
    /// mechanism body's reads are not on the teacher's path and are not bound.
    pub fn bind_use_sites(&self, registry: &TensorRegistry) -> Result<Vec<BoundRead>, BindingError> {
        let storage = self
            .parts
            .parameters
            .iter()
            .enumerate()
            .map(|(p, decl)| {
                registry
                    .storage_of(&TensorId(decl.name.clone()))
                    .cloned()
                    .map_err(|error| BindingError::Registry { parameter: ParameterSlot(p as u32), error })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut raw = Vec::new();
        let mut path = Vec::new();
        self.teacher_reads(self.parts.entry, &mut path, &mut raw);
        let mut ordinals: BTreeMap<TensorId, usize> = BTreeMap::new();
        let mut reads = Vec::with_capacity(raw.len());
        for (parameter, body, node, invocation, map) in raw {
            let read_storage = &storage[parameter.index()];
            let ordinal = ordinals.entry(read_storage.clone()).or_insert(0);
            let site = UseSiteId::read(read_storage, *ordinal);
            *ordinal += 1;
            let registered = match registry.resolve_use_site(&site) {
                Ok(registered) => registered,
                Err(LiftError::UnknownUseSite(..)) => {
                    return Err(BindingError::UnregisteredRead { parameter, body, node, site });
                }
                Err(error) => return Err(BindingError::Registry { parameter, error }),
            };
            if registered.storage != *read_storage {
                return Err(BindingError::StorageMismatch {
                    site,
                    declared: read_storage.clone(),
                    registered: registered.storage,
                });
            }
            if registered.map != map {
                return Err(BindingError::MapMismatch { site, declared: map, registered: registered.map });
            }
            reads.push(BoundRead { parameter, body, node, invocation, site, map });
        }
        for read_storage in ordinals.keys() {
            let reached: BTreeSet<&UseSiteId> = reads
                .iter()
                .filter(|read| storage[read.parameter.index()] == *read_storage)
                .map(|read| &read.site)
                .collect();
            if let Some(site) = registry
                .use_sites_of(read_storage)
                .into_iter()
                .find(|site| !reached.contains(site))
            {
                return Err(BindingError::UnreadSite { storage: read_storage.clone(), site: site.clone() });
            }
        }
        Ok(reads)
    }

    /// Every parameter read of `body_id` on the teacher's path, in execution order.
    fn teacher_reads(
        &self,
        body_id: BodyId,
        path: &mut Vec<CallSite>,
        out: &mut Vec<(ParameterSlot, BodyId, NodeId, Vec<CallSite>, UseMap)>,
    ) {
        for (i, node) in self.parts.bodies[body_id.index()].nodes.iter().enumerate() {
            let site = CallSite { body: body_id, node: NodeId(i as u32), stage: 0 };
            match node {
                Node::Native { primitive, .. } => {
                    for (parameter, map) in primitive.parameter_maps() {
                        out.push((parameter, site.body, site.node, path.clone(), map));
                    }
                }
                Node::Compose { stages, .. } => {
                    for (k, stage) in stages.iter().enumerate() {
                        path.push(CallSite { stage: k as u32, ..site });
                        self.teacher_reads(stage.body, path, out);
                        path.pop();
                    }
                }
                Node::Call { body: callee, .. } | Node::Refine { native: callee, .. } => {
                    path.push(site);
                    self.teacher_reads(*callee, path, out);
                    path.pop();
                }
                Node::Input { .. } | Node::Read { .. } | Node::Write { .. } | Node::Sum { .. } => {}
            }
        }
    }
}

/// Binds a program's formal parameters to the teacher's registered tensors at the
/// use sites [`Program::bind_use_sites`] binds: a linear read applies its storage's
/// residual anchor at the controls of that use, and a stored read returns the
/// storage's registered values.
///
/// A parameter with anchor controls declares them as `[m_Delta, m_1, ..., m_C]`, the
/// order of [`AnchorMask`]'s residual and components, so its reads execute
/// `Theta(m) = m_Delta Theta_* + B sum_c (m_c - m_Delta) v_c` without forming it. A
/// parameter with no controls executes the teacher's tensor at every use. Only bound
/// reads execute, so the reads a global edit of a storage reaches are exactly the ones
/// this source runs; a mechanism body's read is not on the teacher's path and is refused.
/// Each anchor must have been built against the same registry: [`ResidualAnchor::new`]
/// is where its values are checked against the registered fingerprint.
pub struct LiftSource<'a> {
    storage: Vec<TensorId>,
    reads: BTreeMap<(ParameterSlot, BodyId, NodeId, Vec<CallSite>), BoundRead>,
    anchors: BTreeMap<TensorId, &'a ResidualAnchor<'a>>,
    stored: BTreeMap<TensorId, ArrayView1<'a, f64>>,
}

/// A refused [`LiftSource`] binding or read.
#[derive(Clone, Debug, PartialEq)]
pub enum LiftSourceError {
    /// The program's reads do not bind to the registry.
    Binding(BindingError),
    /// A declared name, or the name of a value vector, that the registry refuses.
    Registry { name: TensorId, error: LiftError },
    /// Two anchors, or two value vectors, for one storage tensor.
    Duplicate { storage: TensorId },
    /// A storage tensor the program reads linearly has no anchor.
    MissingAnchor { storage: TensorId },
    /// A storage tensor the program reads as stored values has no values.
    MissingValues { storage: TensorId },
    /// An anchor, or values, for a storage tensor no bound read uses that way.
    UnreadStorage { storage: TensorId },
    /// The values are not the registered ones: another shape or another fingerprint.
    ValuesMismatch { storage: TensorId, registered: StorageTensor, found: StorageTensor },
    /// A parameter's control count is neither zero nor one residual plus one per
    /// component of its storage's anchor.
    ControlCount { parameter: ParameterSlot, controls: usize, components: usize },
    /// A stored read of a parameter with anchor controls, whose mask no stored read
    /// carries.
    ControlledStoredRead { parameter: ParameterSlot, controls: usize },
    /// A read the teacher's path does not reach, such as a mechanism body's.
    UnboundRead { parameter: ParameterSlot, body: BodyId, node: NodeId },
    /// The read applies another map than its bound site registers.
    MapMismatch { site: UseSiteId, registered: UseMap, applied: UseMap },
    /// The anchor refused the read.
    Lift { site: UseSiteId, error: LiftError },
}

impl fmt::Display for LiftSourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binding(error) => write!(formatter, "{error}"),
            Self::Registry { name, error } => write!(formatter, "{}: {error}", name.0),
            Self::Duplicate { storage } => write!(formatter, "storage {} is bound twice", storage.0),
            Self::MissingAnchor { storage } => {
                write!(formatter, "storage {} is read linearly but has no anchor", storage.0)
            }
            Self::MissingValues { storage } => {
                write!(formatter, "storage {} is read as stored values but has none", storage.0)
            }
            Self::UnreadStorage { storage } => {
                write!(formatter, "storage {} is bound, but no bound read uses it that way", storage.0)
            }
            Self::ValuesMismatch { storage, registered, found } => write!(
                formatter,
                "the values of storage {} have shape {:?} and fingerprint {:#x}, not the registered {:?} and {:#x}",
                storage.0, found.shape, found.fingerprint, registered.shape, registered.fingerprint
            ),
            Self::ControlCount { parameter, controls, components } => write!(
                formatter,
                "formal parameter {} declares {controls} controls; its anchor takes none or {}",
                parameter.0,
                components + 1
            ),
            Self::ControlledStoredRead { parameter, controls } => write!(
                formatter,
                "formal parameter {} has {controls} anchor controls and a stored read, which carries no mask",
                parameter.0
            ),
            Self::UnboundRead { parameter, body, node } => write!(
                formatter,
                "formal parameter {} read at node {} of body {} is not on the teacher's path",
                parameter.0, node.0, body.0
            ),
            Self::MapMismatch { site, registered, applied } => write!(
                formatter,
                "use site {} is registered as {registered:?}, but the read applies {applied:?}",
                site.0
            ),
            Self::Lift { site, error } => write!(formatter, "use site {}: {error}", site.0),
        }
    }
}

impl std::error::Error for LiftSourceError {}

impl<'a> LiftSource<'a> {
    /// Binds `program`'s reads to `registry`, each linearly read storage to its anchor
    /// and each storage read as stored values to its values. Refuses values that are not
    /// the registered ones, a read without a tensor and a tensor without a read.
    pub fn new(
        program: &Program,
        registry: &TensorRegistry,
        anchors: Vec<&'a ResidualAnchor<'a>>,
        stored: Vec<(TensorId, ArrayView1<'a, f64>)>,
    ) -> Result<Self, LiftSourceError> {
        let bound = program.bind_use_sites(registry).map_err(LiftSourceError::Binding)?;
        let storage = program
            .parts
            .parameters
            .iter()
            .map(|decl| {
                let name = TensorId(decl.name.clone());
                registry
                    .storage_of(&name)
                    .cloned()
                    .map_err(|error| LiftSourceError::Registry { name, error })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut anchor_map = BTreeMap::new();
        for anchor in anchors {
            if anchor_map.insert(anchor.storage().clone(), anchor).is_some() {
                return Err(LiftSourceError::Duplicate { storage: anchor.storage().clone() });
            }
        }
        let mut stored_map = BTreeMap::new();
        for (id, values) in stored {
            // The owner's shape and fingerprint of the values, from a registry holding only them.
            let mut probe = TensorRegistry::default();
            probe
                .register_storage(id.clone(), values.view().into_dyn())
                .map_err(|error| LiftSourceError::Registry { name: id.clone(), error })?;
            match (registry.storage(&id), probe.storage(&id)) {
                (Some(registered), Some(found)) if registered == found => {}
                (Some(registered), Some(found)) => {
                    return Err(LiftSourceError::ValuesMismatch {
                        storage: id,
                        registered: registered.clone(),
                        found: found.clone(),
                    });
                }
                _ => return Err(LiftSourceError::UnreadStorage { storage: id }),
            }
            if stored_map.insert(id.clone(), values).is_some() {
                return Err(LiftSourceError::Duplicate { storage: id });
            }
        }
        let mut linear = BTreeSet::new();
        let mut plain = BTreeSet::new();
        for read in &bound {
            let parameter = read.parameter;
            let read_storage = &storage[parameter.index()];
            let controls = program.parts.parameters[parameter.index()].controls.len();
            match read.map {
                UseMap::Linear(..) => {
                    let anchor = anchor_map
                        .get(read_storage)
                        .ok_or_else(|| LiftSourceError::MissingAnchor { storage: read_storage.clone() })?;
                    let components = anchor.component_count();
                    if controls != 0 && controls != components + 1 {
                        return Err(LiftSourceError::ControlCount { parameter, controls, components });
                    }
                    linear.insert(read_storage.clone());
                }
                UseMap::Stored => {
                    if controls != 0 {
                        return Err(LiftSourceError::ControlledStoredRead { parameter, controls });
                    }
                    if !stored_map.contains_key(read_storage) {
                        return Err(LiftSourceError::MissingValues { storage: read_storage.clone() });
                    }
                    plain.insert(read_storage.clone());
                }
            }
        }
        if let Some(unread) = anchor_map.keys().find(|id| !linear.contains(*id)) {
            return Err(LiftSourceError::UnreadStorage { storage: unread.clone() });
        }
        if let Some(unread) = stored_map.keys().find(|id| !plain.contains(*id)) {
            return Err(LiftSourceError::UnreadStorage { storage: unread.clone() });
        }
        let reads = bound
            .into_iter()
            .map(|read| ((read.parameter, read.body, read.node, read.invocation.clone()), read))
            .collect();
        Ok(Self { storage, reads, anchors: anchor_map, stored: stored_map })
    }

    /// The bound read at this use, which must apply the map its site registers.
    fn bound_read(&self, parameter: &ParameterUse<'_>, applied: UseMap) -> Result<&BoundRead, LiftSourceError> {
        let key = (parameter.parameter, parameter.body, parameter.node, parameter.invocation.to_vec());
        let read = self.reads.get(&key).ok_or(LiftSourceError::UnboundRead {
            parameter: parameter.parameter,
            body: parameter.body,
            node: parameter.node,
        })?;
        if read.map != applied {
            return Err(LiftSourceError::MapMismatch { site: read.site.clone(), registered: read.map, applied });
        }
        Ok(read)
    }
}

impl ParameterSource for LiftSource<'_> {
    type Error = LiftSourceError;

    fn apply_linear(
        &self,
        parameter: ParameterUse<'_>,
        orientation: TieOrientation,
        rows: ArrayView2<'_, f64>,
    ) -> Result<Governed<Array2<f64>>, Self::Error> {
        let read = self.bound_read(&parameter, UseMap::Linear(orientation))?;
        let storage = &self.storage[parameter.parameter.index()];
        let anchor = self
            .anchors
            .get(storage)
            .ok_or_else(|| LiftSourceError::MissingAnchor { storage: storage.clone() })?;
        match parameter.controls {
            [] => anchor.native_apply(rows, orientation),
            [residual, components @ ..] => anchor.apply(
                &AnchorMask { residual: *residual, components: components.to_vec() },
                rows,
                orientation,
            ),
        }
        .map_err(|error| LiftSourceError::Lift { site: read.site.clone(), error })
    }

    fn vector(&self, parameter: ParameterUse<'_>) -> Result<Array1<f64>, Self::Error> {
        self.bound_read(&parameter, UseMap::Stored)?;
        let storage = &self.storage[parameter.parameter.index()];
        self.stored
            .get(storage)
            .map(|values| values.to_owned())
            .ok_or_else(|| LiftSourceError::MissingValues { storage: storage.clone() })
    }
}

#[cfg(test)]
mod tests {
    //! Fixtures whose entries and mask values are dyadic rationals with small
    //! numerators: every product and sum they reach is representable, so no
    //! operation rounds and the comparisons are exact.

    use super::*;
    use crate::parameter_decomposition::apply::FactoredEdit;
    use crate::parameter_decomposition::lift::ComponentCoefficients;
    use ndarray::array;

    fn controls(indices: &[u32]) -> Vec<ControlId> {
        indices.iter().map(|&index| ControlId(index)).collect()
    }

    fn named_controls(count: u32) -> Vec<ControlDecl> {
        (0..count).map(|index| ControlDecl { name: format!("c{index}") }).collect()
    }

    fn singleton_groups(count: u32) -> Vec<MaskGroup> {
        (0..count)
            .map(|index| MaskGroup { name: format!("g{index}"), controls: vec![ControlId(index)] })
            .collect()
    }

    fn free_parameters(count: u32) -> Vec<ParameterDecl> {
        (0..count)
            .map(|index| ParameterDecl { name: format!("p{index}"), controls: Vec::new() })
            .collect()
    }

    fn native(primitive: NativePrimitive, arguments: &[u32]) -> Node {
        Node::Native { primitive, arguments: arguments.iter().map(|&index| NodeId(index)).collect() }
    }

    fn linear(parameter: u32, argument: u32) -> Node {
        native(
            NativePrimitive::Linear { weight: ParameterSlot(parameter), orientation: TieOrientation::Identity },
            &[argument],
        )
    }

    fn mask(indices: &[u32], argument: u32) -> Node {
        native(NativePrimitive::CoordinateMask { controls: controls(indices) }, &[argument])
    }

    fn term(value: u32, control: Option<u32>) -> SumTerm {
        SumTerm { value: NodeId(value), control: control.map(ControlId) }
    }

    fn body(name: &str, inputs: u32, nodes: Vec<Node>) -> Body {
        let output = NodeId(nodes.len() as u32 - 1);
        Body { name: name.to_string(), inputs, nodes, output }
    }

    fn row_map(rows: &Array2<f64>, matrix: &Array2<f64>) -> Array2<f64> {
        rows.dot(&matrix.t())
    }

    fn global(values: &[(u32, f64)]) -> MaskAssignment {
        let mut masks = MaskAssignment::all_on();
        for &(group, value) in values {
            masks.set(MaskGroupId(group), Vec::new(), value).expect("a finite mask at a new scope");
        }
        masks
    }

    fn row_positions(rows: &Array2<f64>) -> Vec<i64> {
        (0..rows.nrows() as i64).collect()
    }

    fn run<S: ParameterSource>(program: &Program, source: &S, masks: &MaskAssignment, x: &Array2<f64>) -> Array2<f64> {
        program
            .execute(source, masks, vec![x.clone()], Vec::new(), &row_positions(x))
            .expect("execution of a valid program")
            .output
            .to_owned()
    }

    #[test]
    fn tied_coordinate_masks_and_a_controlled_sum_execute_exactly() {
        let r = array![[1.0, 0.5], [-0.25, 2.0], [0.75, -1.0]];
        let u = array![[0.5, -1.0, 0.25], [2.0, 0.125, -0.5]];
        let x = array![[1.0, -2.0], [0.5, 4.0]];
        let parts = ProgramParts {
            parameters: free_parameters(2),
            slots: Vec::new(),
            controls: named_controls(4),
            mask_groups: vec![
                MaskGroup { name: "tied".to_string(), controls: controls(&[0, 1]) },
                MaskGroup { name: "third".to_string(), controls: controls(&[2]) },
                MaskGroup { name: "residual".to_string(), controls: controls(&[3]) },
            ],
            bodies: vec![body(
                "entry",
                1,
                vec![
                    Node::Input { port: 0 },
                    linear(0, 0),
                    mask(&[0, 1, 2], 1),
                    linear(1, 2),
                    Node::Sum { terms: vec![term(0, None), term(3, Some(3))] },
                ],
            )],
            entry: BodyId(0),
        };
        let program = Program::new(parts).expect("a valid program");
        let source = DenseParameters::new(vec![DenseTensor::Matrix(r.clone()), DenseTensor::Matrix(u.clone())]);
        let expected = |coordinates: [f64; 3], residual: f64| {
            let diagonal = Array2::from_diag(&Array1::from(coordinates.to_vec()));
            &x + &(row_map(&row_map(&row_map(&x, &r), &diagonal), &u) * residual)
        };

        let all_on = run(&program, &source, &MaskAssignment::all_on(), &x);
        assert_eq!(all_on, expected([1.0, 1.0, 1.0], 1.0));

        let masked = run(&program, &source, &global(&[(0, 0.5), (1, -1.25), (2, 0.75)]), &x);
        assert_eq!(masked, expected([0.5, 0.5, -1.25], 0.75));
        // Positive controls: the mask moved the output, and the group set BOTH tied
        // coordinates (setting only the first gives a different output).
        assert_ne!(masked, all_on);
        assert_ne!(masked, expected([0.5, 1.0, -1.25], 0.75));
        assert_eq!(program.output_mask_dependence().degree(), Some(2));
    }

    #[test]
    fn composition_induces_the_cross_term_without_a_control_of_its_own() {
        let delta_a = array![[0.5, -0.25], [0.0, 1.0]];
        let delta_b = array![[-1.0, 0.5], [0.25, 0.0]];
        let identity = Array2::<f64>::eye(2);
        let x = array![[1.0, 2.0], [-0.5, 0.25]];
        let parts = ProgramParts {
            parameters: free_parameters(2),
            slots: Vec::new(),
            controls: named_controls(2),
            mask_groups: singleton_groups(2),
            bodies: vec![
                body(
                    "entry",
                    1,
                    vec![
                        Node::Input { port: 0 },
                        Node::Compose {
                            value: NodeId(0),
                            stages: vec![
                                ComposeStage { body: BodyId(1), control: Some(ControlId(0)) },
                                ComposeStage { body: BodyId(2), control: Some(ControlId(1)) },
                            ],
                        },
                    ],
                ),
                body("a", 1, vec![Node::Input { port: 0 }, linear(0, 0)]),
                body("b", 1, vec![Node::Input { port: 0 }, linear(1, 0)]),
            ],
            entry: BodyId(0),
        };
        let program = Program::new(parts).expect("a valid program");
        let source = DenseParameters::new(vec![
            DenseTensor::Matrix(&identity + &delta_a),
            DenseTensor::Matrix(&identity + &delta_b),
        ]);
        let (m_a, m_b) = (0.5, -0.25);
        let output = run(&program, &source, &global(&[(0, m_a), (1, m_b)]), &x);
        // Rows transform by transposes: stage A then stage B is
        // (I + m_B Delta_B)(I + m_A Delta_A) = I + m_A Delta_A + m_B Delta_B + m_A m_B Delta_B Delta_A.
        let first_order = &x + &(row_map(&x, &delta_a) * m_a) + &(row_map(&x, &delta_b) * m_b);
        let cross = row_map(&row_map(&x, &delta_a), &delta_b) * (m_a * m_b);
        assert_eq!(output, &first_order + &cross);
        // Negative control: the cross term is nonzero, so a sum of the two
        // first-order mechanisms is refuted.
        assert_ne!(output, first_order);
        assert_eq!(program.output_mask_dependence().degree(), Some(2));
        assert!(!program.output_mask_dependence().is_affine());
        // The induced cross term multiplies two distinct groups, so P9 runs on it
        // instead of refusing.
        assert!(program.output_mask_dependence().is_multilinear());
        // A deleted stage is skipped exactly.
        let deleted = run(&program, &source, &global(&[(0, 0.0), (1, 1.0)]), &x);
        assert_eq!(deleted, row_map(&x, &(&identity + &delta_b)));
    }

    fn two_control_program(nodes: Vec<Node>) -> Program {
        Program::new(ProgramParts {
            parameters: free_parameters(2),
            slots: Vec::new(),
            controls: named_controls(2),
            mask_groups: singleton_groups(2),
            bodies: vec![body("entry", 1, nodes)],
            entry: BodyId(0),
        })
        .expect("a valid program")
    }

    #[test]
    fn mask_dependence_separates_affine_sums_from_products_and_nonlinearities() {
        let affine_sum = vec![
            Node::Input { port: 0 },
            linear(0, 0),
            linear(1, 0),
            Node::Sum { terms: vec![term(0, None), term(1, Some(0)), term(2, Some(1))] },
        ];
        let affine = two_control_program(affine_sum.clone());
        assert_eq!(affine.output_mask_dependence().degree(), Some(1));
        assert!(affine.output_mask_dependence().is_affine());

        let mut through_activation = affine_sum;
        through_activation.push(native(NativePrimitive::Activation { activation: NativeActivation::ExactGelu }, &[3]));
        assert_eq!(two_control_program(through_activation).output_mask_dependence(), MaskDependence::Nonlinear);

        let multiplied_layers = two_control_program(vec![
            Node::Input { port: 0 },
            linear(0, 0),
            Node::Sum { terms: vec![term(1, Some(0))] },
            linear(1, 2),
            Node::Sum { terms: vec![term(3, Some(1))] },
        ]);
        assert_eq!(multiplied_layers.output_mask_dependence().degree(), Some(2));
        assert!(!multiplied_layers.output_mask_dependence().is_affine());
        assert!(multiplied_layers.output_mask_dependence().is_multilinear());

        // Positive control: one group tying the controls of two multiplied layers has
        // degree two in that group's value, so the value is not multilinear.
        let tied_layers = Program::new(ProgramParts {
            parameters: free_parameters(2),
            slots: Vec::new(),
            controls: named_controls(2),
            mask_groups: vec![MaskGroup { name: "tied".to_string(), controls: controls(&[0, 1]) }],
            bodies: vec![body(
                "entry",
                1,
                vec![
                    Node::Input { port: 0 },
                    linear(0, 0),
                    Node::Sum { terms: vec![term(1, Some(0))] },
                    linear(1, 2),
                    Node::Sum { terms: vec![term(3, Some(1))] },
                ],
            )],
            entry: BodyId(0),
        })
        .expect("a valid program");
        let tied_dependence = tied_layers.output_mask_dependence();
        assert_eq!(tied_dependence.degree(), Some(2));
        assert_eq!(tied_dependence.group_degree(MaskGroupId(0)), Some(2));
        assert!(!tied_dependence.is_multilinear());

        let activation_before_masks = two_control_program(vec![
            Node::Input { port: 0 },
            linear(0, 0),
            native(NativePrimitive::Activation { activation: NativeActivation::Relu }, &[1]),
            linear(1, 2),
            Node::Sum { terms: vec![term(3, Some(0)), term(0, Some(1))] },
        ]);
        assert_eq!(activation_before_masks.output_mask_dependence().degree(), Some(1));
    }

    #[test]
    fn a_refinement_runs_its_native_body_exactly_when_every_reached_control_is_one() {
        let w = array![[1.0, -0.5], [0.25, 2.0]];
        let r = Array2::<f64>::eye(2);
        // A deliberately wrong factor, U R = 2 W, so the two paths are told apart.
        let u = &w * 2.0;
        let x = array![[1.0, -2.0], [0.5, 4.0]];
        let parts = ProgramParts {
            parameters: free_parameters(3),
            slots: Vec::new(),
            controls: named_controls(2),
            mask_groups: singleton_groups(2),
            bodies: vec![
                body(
                    "entry",
                    1,
                    vec![
                        Node::Input { port: 0 },
                        Node::Refine { native: BodyId(1), mechanism: BodyId(2), arguments: vec![NodeId(0)] },
                    ],
                ),
                body("native", 1, vec![Node::Input { port: 0 }, linear(0, 0)]),
                body("mechanism", 1, vec![Node::Input { port: 0 }, linear(1, 0), mask(&[0, 1], 1), linear(2, 2)]),
            ],
            entry: BodyId(0),
        };
        let program = Program::new(parts).expect("a valid program");
        let source = DenseParameters::new(vec![
            DenseTensor::Matrix(w.clone()),
            DenseTensor::Matrix(r.clone()),
            DenseTensor::Matrix(u.clone()),
        ]);
        let native_output = row_map(&x, &w);
        let mechanism_all_on = row_map(&row_map(&x, &r), &u);
        assert_ne!(mechanism_all_on, native_output);

        assert_eq!(run(&program, &source, &MaskAssignment::all_on(), &x), native_output);
        assert_eq!(run(&program, &source, &global(&[(0, 1.0), (1, 1.0)]), &x), native_output);

        let half = global(&[(0, 0.5)]);
        let diagonal = array![[0.5, 0.0], [0.0, 1.0]];
        assert_eq!(run(&program, &source, &half, &x), row_map(&row_map(&row_map(&x, &r), &diagonal), &u));

        let (execution, residuals) = program
            .refinement_residuals(&source, &MaskAssignment::all_on(), vec![x.clone()], Vec::new(), &row_positions(&x))
            .expect("residual execution");
        assert_eq!(*execution.output, native_output);
        assert_eq!(residuals.len(), 1);
        let residual = &residuals[0];
        assert!(residual.all_on);
        assert_eq!(residual.invocation, vec![CallSite { body: BodyId(0), node: NodeId(1), stage: 0 }]);
        // U R - W = W, so the residual equals the native scale. The difference carries
        // the rounding of its one subtraction; the scale is exact.
        let native_scale = native_output.iter().fold(0.0_f64, |largest, entry| largest.max(entry.abs()));
        assert!(matches!(
            &residual.difference,
            EvidenceStatus::Exact { value, numerical_error, witness: Some(..), .. }
                if *value == native_scale
                    && *numerical_error == (accumulation_growth(1) * native_scale).next_up()
                    && *numerical_error > accumulation_growth(1) * native_scale
        ));
        assert!(matches!(
            &residual.native_scale,
            EvidenceStatus::Exact { value, numerical_error, .. }
                if *value == native_scale && *numerical_error == 0.0
        ));

        let (half_execution, half_residuals) = program
            .refinement_residuals(&source, &half, vec![x.clone()], Vec::new(), &row_positions(&x))
            .expect("residual execution");
        assert!(!half_residuals[0].all_on);
        assert_eq!(*half_execution.output, run(&program, &source, &half, &x));
    }

    #[test]
    fn a_refinement_resolves_each_reached_control_at_the_invocation_where_it_runs() {
        // The native body is `2 x`. The mechanism reaches the shared body's one term
        // control `m`, `m x`, through a call or through a composition's second stage, so
        // that control runs one invocation below the refinement.
        let build = |mechanism: Body, stage: Vec<Body>| {
            let mut bodies = vec![
                body(
                    "entry",
                    1,
                    vec![
                        Node::Input { port: 0 },
                        Node::Refine { native: BodyId(1), mechanism: BodyId(2), arguments: vec![NodeId(0)] },
                    ],
                ),
                body("native", 1, vec![Node::Input { port: 0 }, Node::Sum { terms: vec![term(0, None), term(0, None)] }]),
                mechanism,
                body("shared", 1, vec![Node::Input { port: 0 }, Node::Sum { terms: vec![term(0, Some(0))] }]),
            ];
            bodies.extend(stage);
            Program::new(ProgramParts {
                parameters: Vec::new(),
                slots: Vec::new(),
                controls: named_controls(1),
                mask_groups: singleton_groups(1),
                bodies,
                entry: BodyId(0),
            })
            .expect("a valid program")
        };
        let called = build(
            body("mechanism", 1, vec![Node::Input { port: 0 }, Node::Call { body: BodyId(3), arguments: vec![NodeId(0)] }]),
            Vec::new(),
        );
        let composed = build(
            body(
                "mechanism",
                1,
                vec![
                    Node::Input { port: 0 },
                    Node::Compose {
                        value: NodeId(0),
                        stages: vec![
                            ComposeStage { body: BodyId(4), control: None },
                            ComposeStage { body: BodyId(3), control: None },
                        ],
                    },
                ],
            ),
            vec![body("identity", 1, vec![Node::Input { port: 0 }, Node::Sum { terms: vec![term(0, None)] }])],
        );
        let source = DenseParameters::new(Vec::new());
        let x = array![[1.0, -2.0]];
        let refinement = CallSite { body: BodyId(0), node: NodeId(1), stage: 0 };
        let mechanism_node = CallSite { body: BodyId(2), node: NodeId(1), stage: 0 };
        for (program, inner) in [(&called, mechanism_node), (&composed, CallSite { stage: 1, ..mechanism_node })] {
            let deep = vec![refinement, inner];
            assert_eq!(run(program, &source, &MaskAssignment::all_on(), &x), &x * 2.0);
            // A scope on the refinement reaches the control below it.
            let mut shallow = MaskAssignment::all_on();
            shallow.set(MaskGroupId(0), vec![refinement], 0.5).expect("a new scope");
            assert_eq!(run(program, &source, &shallow, &x), &x * 0.5);
            // A scope on the invocation where the control runs deletes it there, so the
            // mechanism runs, and the residual records the refinement as not all-on.
            let mut deleted = MaskAssignment::all_on();
            deleted.set(MaskGroupId(0), deep.clone(), 0.0).expect("a new scope");
            assert_eq!(run(program, &source, &deleted, &x), array![[0.0, 0.0]]);
            let (execution, residuals) = program
                .refinement_residuals(&source, &deleted, vec![x.clone()], Vec::new(), &row_positions(&x))
                .expect("residual execution");
            assert_eq!(*execution.output, array![[0.0, 0.0]]);
            assert!(!residuals[0].all_on);
            // A deeper scope of `1` overrides the global value where the control runs,
            // so every reached control is on and the native body runs.
            let mut restored = global(&[(0, 0.5)]);
            restored.set(MaskGroupId(0), deep, 1.0).expect("a new scope");
            assert_eq!(run(program, &source, &restored, &x), &x * 2.0);
        }
    }

    #[test]
    fn an_invocation_scope_edits_one_call_and_a_group_ties_every_use() {
        let build = |mask_groups: Vec<MaskGroup>| {
            Program::new(ProgramParts {
                parameters: Vec::new(),
                slots: Vec::new(),
                controls: named_controls(2),
                mask_groups,
                bodies: vec![
                    body(
                        "entry",
                        1,
                        vec![
                            Node::Input { port: 0 },
                            Node::Call { body: BodyId(1), arguments: vec![NodeId(0)] },
                            Node::Call { body: BodyId(1), arguments: vec![NodeId(1)] },
                            Node::Call { body: BodyId(2), arguments: vec![NodeId(2)] },
                        ],
                    ),
                    body("shared", 1, vec![Node::Input { port: 0 }, mask(&[0], 0)]),
                    body("other", 1, vec![Node::Input { port: 0 }, mask(&[1], 0)]),
                ],
                entry: BodyId(0),
            })
            .expect("a valid program")
        };
        let source = DenseParameters::new(Vec::new());
        let x = array![[2.0]];
        let first = CallSite { body: BodyId(0), node: NodeId(1), stage: 0 };
        let second = CallSite { body: BodyId(0), node: NodeId(2), stage: 0 };
        let separate = build(singleton_groups(2));

        assert_eq!(run(&separate, &source, &MaskAssignment::all_on(), &x), array![[2.0]]);
        // A global value reaches both invocations of the shared body.
        assert_eq!(run(&separate, &source, &global(&[(0, 0.5)]), &x), array![[0.5]]);
        // A scoped value reaches only its invocation.
        let mut scoped = MaskAssignment::all_on();
        scoped.set(MaskGroupId(0), vec![first], 0.5).expect("a new scope");
        assert_eq!(run(&separate, &source, &scoped, &x), array![[1.0]]);
        // The deepest applicable scope wins over the global value.
        let mut layered = global(&[(0, 0.25)]);
        layered.set(MaskGroupId(0), vec![second], 0.5).expect("a new scope");
        assert_eq!(run(&separate, &source, &layered, &x), array![[0.25]]);

        let tied = build(vec![MaskGroup { name: "both".to_string(), controls: controls(&[0, 1]) }]);
        assert_eq!(run(&tied, &source, &global(&[(0, 0.5)]), &x), array![[0.25]]);
        // Positive control: untied, the same value leaves the other body's control on.
        assert_ne!(run(&separate, &source, &global(&[(0, 0.5)]), &x), array![[0.25]]);
    }

    fn activation_program(activation: NativeActivation) -> Program {
        Program::new(ProgramParts {
            parameters: Vec::new(),
            slots: Vec::new(),
            controls: Vec::new(),
            mask_groups: Vec::new(),
            bodies: vec![body(
                "entry",
                1,
                vec![Node::Input { port: 0 }, native(NativePrimitive::Activation { activation }, &[0])],
            )],
            entry: BodyId(0),
        })
        .expect("a valid program")
    }

    #[test]
    fn activation_nodes_evaluate_the_gam_math_owner() {
        let x = array![[-1.5, 0.0, 2.25]];
        let smoothed = |owner: GaussianActivation, t: f64| {
            let mut value = [0.0];
            gaussian_smoothing_derivatives(owner, t, 0.0, &mut value).expect("a finite point");
            value[0]
        };
        let source = DenseParameters::new(Vec::new());
        let mut outputs = Vec::new();
        for activation in [NativeActivation::Relu, NativeActivation::ExactGelu, NativeActivation::Silu] {
            let output = run(&activation_program(activation), &source, &MaskAssignment::all_on(), &x);
            let expected = x.mapv(|t| match activation {
                NativeActivation::Relu => smoothed(GaussianActivation::Relu, t),
                NativeActivation::ExactGelu => smoothed(GaussianActivation::ExactGelu, t),
                NativeActivation::Silu => silu_derivatives(t)[0],
            });
            assert_eq!(output, expected);
            outputs.push(output);
        }
        assert_eq!(outputs[0], array![[0.0, 0.0, 2.25]]);
        // The exact GELU is negative at -1.5 and strictly between 0 and t at 2.25,
        // so the kinds are not routed to one function.
        assert!(outputs[1][[0, 0]] < 0.0);
        assert!(outputs[1][[0, 2]] > 0.0 && outputs[1][[0, 2]] < 2.25);
        // SiLU is t sigma(t): exactly zero at 0, negative at -1.5, and not the GELU there.
        assert_eq!(outputs[2][[0, 1]], 0.0);
        assert!(outputs[2][[0, 0]] < 0.0);
        assert_ne!(outputs[2][[0, 0]], outputs[1][[0, 0]]);
        // The node and owner enumerations map onto each other exactly, both ways.
        for activation in [NativeActivation::Relu, NativeActivation::ExactGelu, NativeActivation::Silu] {
            assert_eq!(NativeActivation::from_owner(activation.to_owner()), activation);
        }
        for owner in [GaussianActivation::Relu, GaussianActivation::ExactGelu, GaussianActivation::Silu] {
            assert_eq!(NativeActivation::from_owner(owner).to_owner(), owner);
        }
        // The SiLU jet returns NaN on a non-finite input; the node refuses it instead.
        assert!(silu_derivatives(f64::NAN)[0].is_nan());
        assert!(matches!(
            activation_program(NativeActivation::Silu).execute(
                &source,
                &MaskAssignment::all_on(),
                vec![array![[f64::NAN]]],
                Vec::new(),
                &[0],
            ),
            Err(ExecutionError::Program(ProgramError::Activation {
                error: GaussianActivationError::NonFiniteArgument { .. },
                ..
            }))
        ));
    }

    fn validation_base() -> ProgramParts {
        ProgramParts {
            parameters: free_parameters(1),
            slots: Vec::new(),
            controls: named_controls(1),
            mask_groups: singleton_groups(1),
            bodies: vec![
                body(
                    "entry",
                    1,
                    vec![
                        Node::Input { port: 0 },
                        Node::Compose {
                            value: NodeId(0),
                            stages: vec![ComposeStage { body: BodyId(1), control: Some(ControlId(0)) }],
                        },
                    ],
                ),
                body("stage", 1, vec![Node::Input { port: 0 }, linear(0, 0)]),
            ],
            entry: BodyId(0),
        }
    }

    #[test]
    fn validation_refuses_each_malformed_graph() {
        assert!(Program::new(validation_base()).is_ok());

        let mut forward = validation_base();
        forward.bodies[1].nodes[1] = linear(0, 1);
        assert!(matches!(Program::new(forward), Err(ProgramError::ForwardReference { .. })));

        let mut recursive = validation_base();
        recursive.bodies[1].nodes.push(Node::Call { body: BodyId(1), arguments: vec![NodeId(1)] });
        assert!(matches!(
            Program::new(recursive),
            Err(ProgramError::RecursiveCall { body: BodyId(1) })
        ));

        let mut twice = validation_base();
        twice.bodies[1].nodes.push(mask(&[0], 1));
        assert!(matches!(
            Program::new(twice),
            Err(ProgramError::ControlUsedTwice { control: ControlId(0) })
        ));

        let mut two_groups = validation_base();
        two_groups.mask_groups.push(MaskGroup { name: "again".to_string(), controls: controls(&[0]) });
        assert!(matches!(Program::new(two_groups), Err(ProgramError::ControlInTwoGroups { .. })));

        let mut ungrouped = validation_base();
        ungrouped.mask_groups.clear();
        assert!(matches!(Program::new(ungrouped), Err(ProgramError::ControlUngrouped { .. })));

        let mut unused = validation_base();
        unused.controls.push(ControlDecl { name: "extra".to_string() });
        assert!(matches!(
            Program::new(unused),
            Err(ProgramError::ControlUnused { control: ControlId(1) })
        ));

        let mut impure = validation_base();
        impure.slots.push(SlotDecl { name: "stream".to_string() });
        impure.bodies[1].nodes.push(Node::Write { slot: SlotId(0), value: NodeId(1) });
        assert!(matches!(
            Program::new(impure),
            Err(ProgramError::ImpureBody { callee: BodyId(1), .. })
        ));

        let refinement = |native: u32, mechanism: u32| ProgramParts {
            parameters: free_parameters(1),
            slots: Vec::new(),
            controls: named_controls(1),
            mask_groups: singleton_groups(1),
            bodies: vec![
                body(
                    "entry",
                    1,
                    vec![
                        Node::Input { port: 0 },
                        Node::Refine {
                            native: BodyId(native),
                            mechanism: BodyId(mechanism),
                            arguments: vec![NodeId(0)],
                        },
                    ],
                ),
                body("masked", 1, vec![Node::Input { port: 0 }, mask(&[0], 0)]),
                body("plain", 1, vec![Node::Input { port: 0 }, linear(0, 0)]),
            ],
            entry: BodyId(0),
        };
        assert!(Program::new(refinement(2, 1)).is_ok());
        assert!(matches!(
            Program::new(refinement(1, 2)),
            Err(ProgramError::NativeReferenceNotNative { native: BodyId(1), .. })
        ));
    }

    #[test]
    fn a_program_document_round_trips_and_refuses_other_schemas() {
        let program = Program::new(validation_base()).expect("a valid program");
        let text = program.to_json().expect("a serializable program");
        assert_eq!(Program::from_json(&text).expect("a round trip"), program);

        let older = text.replace(MECHANISM_PROGRAM_SCHEMA, "gamfit.MechanismProgram/v0");
        assert_ne!(older, text);
        assert!(matches!(
            Program::from_json(&older),
            Err(ProgramError::SchemaMismatch { found }) if found == "gamfit.MechanismProgram/v0"
        ));

        let extra_field = text.replacen('{', "{\"extra\":1,", 1);
        assert!(matches!(Program::from_json(&extra_field), Err(ProgramError::Json { .. })));

        let mut invalid = validation_base();
        invalid.mask_groups.clear();
        let invalid_text = serde_json::to_string(&ProgramDocument {
            schema: MECHANISM_PROGRAM_SCHEMA.to_string(),
            program: invalid,
        })
        .expect("a serializable document");
        assert!(matches!(Program::from_json(&invalid_text), Err(ProgramError::ControlUngrouped { .. })));
    }

    #[test]
    fn slots_carry_state_and_refusals_name_their_cause() {
        let w = array![[0.5, 0.0], [-0.25, 1.0]];
        let x = array![[2.0, -1.0]];
        let program = Program::new(ProgramParts {
            parameters: free_parameters(1),
            slots: vec![SlotDecl { name: "stream".to_string() }],
            controls: Vec::new(),
            mask_groups: Vec::new(),
            bodies: vec![body(
                "entry",
                0,
                vec![
                    Node::Read { slot: SlotId(0) },
                    linear(0, 0),
                    Node::Sum { terms: vec![term(0, None), term(1, None)] },
                    Node::Write { slot: SlotId(0), value: NodeId(2) },
                ],
            )],
            entry: BodyId(0),
        })
        .expect("a valid program");
        let source = DenseParameters::new(vec![DenseTensor::Matrix(w.clone())]);
        let execution = program
            .execute(&source, &MaskAssignment::all_on(), Vec::new(), vec![Some(x.clone())], &[0])
            .expect("a written slot");
        let updated = &x + &row_map(&x, &w);
        assert_eq!(*execution.output, updated);
        assert_eq!(execution.slots, vec![Some(updated)]);
        assert!(matches!(
            program.execute(&source, &MaskAssignment::all_on(), Vec::new(), vec![None], &[0]),
            Err(ExecutionError::Program(ProgramError::SlotUnwritten { .. }))
        ));
        assert!(matches!(
            program.execute(&source, &MaskAssignment::all_on(), Vec::new(), Vec::new(), &[0]),
            Err(ExecutionError::Program(ProgramError::SlotCount { expected: 1, found: 0 }))
        ));

        let mut masks = MaskAssignment::all_on();
        assert!(matches!(
            masks.set(MaskGroupId(0), Vec::new(), f64::NAN),
            Err(ProgramError::NonFiniteMask { .. })
        ));
        assert!(masks.set(MaskGroupId(0), Vec::new(), 0.5).is_ok());
        assert!(matches!(
            masks.set(MaskGroupId(0), Vec::new(), 0.25),
            Err(ProgramError::DuplicateMaskScope { .. })
        ));
        assert!(matches!(
            program.execute(&source, &masks, Vec::new(), vec![Some(x.clone())], &[0]),
            Err(ExecutionError::Program(ProgramError::UnknownMaskGroup { .. }))
        ));

        let anchored = Program::new(ProgramParts {
            parameters: vec![ParameterDecl { name: "anchored".to_string(), controls: controls(&[0]) }],
            slots: Vec::new(),
            controls: named_controls(1),
            mask_groups: singleton_groups(1),
            bodies: vec![body("entry", 1, vec![Node::Input { port: 0 }, linear(0, 0)])],
            entry: BodyId(0),
        })
        .expect("a valid program");
        assert_eq!(anchored.output_mask_dependence().degree(), Some(1));
        assert!(matches!(
            anchored.execute(&source, &MaskAssignment::all_on(), vec![x.clone()], Vec::new(), &[0]),
            Err(ExecutionError::Source { error: DenseParameterError::Controlled { .. }, .. })
        ));
    }

    #[test]
    fn validation_refuses_declarations_no_node_uses() {
        assert!(Program::new(validation_base()).is_ok());
        let mut unused_slot = validation_base();
        unused_slot.slots.push(SlotDecl { name: "idle".to_string() });
        assert!(matches!(
            Program::new(unused_slot),
            Err(ProgramError::SlotUnused { slot: SlotId(0) })
        ));
        let mut unused_parameter = validation_base();
        unused_parameter.parameters.push(ParameterDecl { name: "idle".to_string(), controls: Vec::new() });
        assert!(matches!(
            Program::new(unused_parameter),
            Err(ProgramError::ParameterUnused { parameter: ParameterSlot(1) })
        ));
    }

    #[test]
    fn normalization_nodes_evaluate_the_gated_rewrite_owner() {
        let x = array![[1.0, -2.0, 0.5], [3.0, 0.25, -1.0]];
        let gain = array![0.5, 2.0, -1.0];
        let bias = array![0.125, 0.0, -0.25];
        // The source configuration's epsilon, e.g. Qwen3's rms_norm_eps.
        let epsilon = 1e-6;
        let build = |primitive: NativePrimitive, parameters: u32| {
            Program::new(ProgramParts {
                parameters: free_parameters(parameters),
                slots: Vec::new(),
                controls: Vec::new(),
                mask_groups: Vec::new(),
                bodies: vec![body("entry", 1, vec![Node::Input { port: 0 }, native(primitive, &[0])])],
                entry: BodyId(0),
            })
            .expect("a valid program")
        };
        let source = DenseParameters::new(vec![DenseTensor::Vector(gain.clone()), DenseTensor::Vector(bias.clone())]);
        let rms = run(
            &build(NativePrimitive::RmsNorm { epsilon, gain: ParameterSlot(0) }, 1),
            &source,
            &MaskAssignment::all_on(),
            &x,
        );
        let expected_rms = MaskedNorm::Rms { epsilon, gain: gain.view() }
            .apply(x.view())
            .expect("a finite normalization");
        assert_eq!(rms, expected_rms);
        let layer = run(
            &build(NativePrimitive::LayerNorm { epsilon, gain: ParameterSlot(0), bias: ParameterSlot(1) }, 2),
            &source,
            &MaskAssignment::all_on(),
            &x,
        );
        let expected_layer = MaskedNorm::Layer { epsilon, gain: gain.view(), bias: bias.view() }
            .apply(x.view())
            .expect("a finite normalization");
        assert_eq!(layer, expected_layer);
        // Positive control: the two kinds are different functions of the same rows.
        assert_ne!(rms, layer);
        // The owner's refusal of an invalid epsilon reaches the caller unchanged.
        assert!(matches!(
            build(NativePrimitive::RmsNorm { epsilon: -1.0, gain: ParameterSlot(0) }, 1).execute(
                &source,
                &MaskAssignment::all_on(),
                vec![x.clone()],
                Vec::new(),
                &row_positions(&x),
            ),
            Err(ExecutionError::Program(ProgramError::GatedRewrite {
                error: GatedRewriteError::InvalidEpsilon { .. },
                ..
            }))
        ));
    }

    /// A `HeadRmsNorm` node is the attention owner's per-head norm, the function a Qwen3 block's
    /// `q_norm` and `k_norm` run: each contiguous `head_dim` block of a row normalized with the
    /// one shared gain. Controls: over two heads it differs from the row-wide `RmsNorm` of the
    /// same rows with the gain tiled; a width that is not a whole number of heads reaches the
    /// caller as the owner's typed refusal; the codeword carries the head width.
    #[test]
    fn head_norm_nodes_evaluate_the_attention_owner_head_by_head() {
        let x = array![[1.0, -2.0, 0.5, 3.0], [0.25, -1.0, 2.0, 0.125]];
        let gain = array![0.5, -1.5];
        // epsilon = 2^-20, a point of the codeword lattice.
        let epsilon = 9.5367431640625e-7;
        let parts = |head_dim: usize| ProgramParts {
            parameters: free_parameters(1),
            slots: Vec::new(),
            controls: Vec::new(),
            mask_groups: Vec::new(),
            bodies: vec![body(
                "entry",
                1,
                vec![
                    Node::Input { port: 0 },
                    native(NativePrimitive::HeadRmsNorm { head_dim, epsilon, gain: ParameterSlot(0) }, &[0]),
                ],
            )],
            entry: BodyId(0),
        };
        let program = Program::new(parts(2)).expect("a valid program");
        let source = DenseParameters::new(vec![DenseTensor::Vector(gain.clone())]);
        let normed = run(&program, &source, &MaskAssignment::all_on(), &x);
        assert_eq!(normed, head_rms_norm(x.view(), 2, epsilon, gain.view()).expect("two whole heads"));
        let second_head = MaskedNorm::Rms { epsilon, gain: gain.view() }
            .apply(x.slice(ndarray::s![.., 2..]))
            .expect("finite head rows");
        assert_eq!(normed.slice(ndarray::s![.., 2..]), second_head, "each head is its own RMS norm");
        let tiled = array![0.5, -1.5, 0.5, -1.5];
        let row_wide = MaskedNorm::Rms { epsilon, gain: tiled.view() }.apply(x.view()).expect("finite rows");
        assert_ne!(normed, row_wide, "the per-head norm is not the row-wide norm");
        let odd = array![[1.0, 2.0, 3.0]];
        assert!(matches!(
            program.execute(&source, &MaskAssignment::all_on(), vec![odd.clone()], Vec::new(), &row_positions(&odd)),
            Err(ExecutionError::Program(ProgramError::GatedRewrite {
                error: GatedRewriteError::ShapeMismatch { .. },
                ..
            }))
        ));
        let encode = |head_dim: usize| {
            Program::new(parts(head_dim))
                .expect("a valid program")
                .encode(lattice())
                .expect("an encodable program")
        };
        let decoded = Program::decode(&encode(2)).expect("the codeword of a valid program");
        assert_eq!(decoded.parts(), &unnamed(&parts(2)));
        let other = Program::decode(&encode(4)).expect("the codeword of a valid program");
        assert_eq!(other.parts(), &unnamed(&parts(4)));
        assert_ne!(decoded.parts(), other.parts(), "the codeword carries the head width");
    }

    #[test]
    fn swiglu_nodes_evaluate_the_gated_rewrite_owner() {
        let gate = array![[1.0, -2.0], [0.5, 0.0]];
        let up = array![[0.25, 3.0], [-1.0, 2.0]];
        let program = Program::new(ProgramParts {
            parameters: Vec::new(),
            slots: Vec::new(),
            controls: Vec::new(),
            mask_groups: Vec::new(),
            bodies: vec![body(
                "entry",
                2,
                vec![Node::Input { port: 0 }, Node::Input { port: 1 }, native(NativePrimitive::SwiGlu, &[0, 1])],
            )],
            entry: BodyId(0),
        })
        .expect("a valid program");
        let source = DenseParameters::new(Vec::new());
        let output = program
            .execute(&source, &MaskAssignment::all_on(), vec![gate.clone(), up.clone()], Vec::new(), &row_positions(&gate))
            .expect("SwiGLU execution")
            .output;
        assert_eq!(*output, swiglu_hidden(gate.view(), up.view()).expect("finite rows of one shape"));
        // Positive control: the gate passes through SiLU, so the node is not the plain product.
        assert_ne!(*output, &gate * &up);
        // The owner's shape refusal reaches the caller unchanged.
        assert!(matches!(
            program.execute(
                &source,
                &MaskAssignment::all_on(),
                vec![gate.clone(), array![[0.0], [1.0]]],
                Vec::new(),
                &row_positions(&gate),
            ),
            Err(ExecutionError::Program(ProgramError::GatedRewrite { .. }))
        ));
    }

    fn attention_rotary() -> RotaryEmbedding {
        RotaryEmbedding { pairing: RotaryPairing::HalfSplit, inverse_frequencies: vec![1.0], attention_scaling: 1.0 }
    }

    fn attention_fixture(geometry: AttentionGeometry) -> ProgramParts {
        ProgramParts {
            parameters: free_parameters(3),
            slots: Vec::new(),
            controls: Vec::new(),
            mask_groups: Vec::new(),
            bodies: vec![body(
                "entry",
                1,
                vec![
                    Node::Input { port: 0 },
                    linear(0, 0),
                    linear(1, 0),
                    linear(2, 0),
                    native(
                        NativePrimitive::CausalSelfAttention {
                            geometry,
                            rotary: attention_rotary(),
                            score_scale: 0.75,
                        },
                        &[1, 2, 3],
                    ),
                ],
            )],
            entry: BodyId(0),
        }
    }

    #[test]
    fn causal_self_attention_nodes_evaluate_the_attention_owner() {
        let geometry = AttentionGeometry { model_dim: 4, n_heads: 2, n_kv_heads: 1, head_dim: 2 };
        let x = array![[1.0, -0.5, 0.25, 2.0], [0.5, 1.5, -1.0, 0.0], [-2.0, 0.75, 1.0, -0.25]];
        let w_q = array![
            [0.5, 0.0, 1.0, -0.5],
            [0.25, 1.0, 0.0, 0.5],
            [-1.0, 0.5, 0.25, 0.0],
            [0.0, -0.25, 0.5, 1.0]
        ];
        let w_k = array![[1.0, 0.5, -0.5, 0.0], [0.0, 0.25, 1.0, -1.0]];
        let w_v = array![[0.5, -1.0, 0.0, 0.25], [1.0, 0.0, 0.5, -0.5]];
        let parts = attention_fixture(geometry);
        let program = Program::new(parts.clone()).expect("a valid program");
        let source = DenseParameters::new(vec![
            DenseTensor::Matrix(w_q.clone()),
            DenseTensor::Matrix(w_k.clone()),
            DenseTensor::Matrix(w_v.clone()),
        ]);
        let positions = [0_i64, 1, 2];
        let output = program
            .execute(&source, &MaskAssignment::all_on(), vec![x.clone()], Vec::new(), &positions)
            .expect("attention execution")
            .output;
        let owner = RotaryCausalAttention::new(geometry, attention_rotary(), 0.75)
            .expect("a valid attention");
        let (queries, keys, value_rows) = (row_map(&x, &w_q), row_map(&x, &w_k), row_map(&x, &w_v));
        let expected = owner
            .attend_projected(
                ProjectedRows::exact(queries.view()),
                ProjectedRows::exact(keys.view()),
                ProjectedRows::exact(value_rows.view()),
                &positions,
            )
            .expect("owner attention")
            .mixed;
        assert_eq!(*output, expected);
        // Positive control: moving one position changes the rotary scores, so the
        // execution's positions reach the node.
        let shifted = program
            .execute(&source, &MaskAssignment::all_on(), vec![x.clone()], Vec::new(), &[0, 1, 5])
            .expect("attention execution")
            .output;
        assert_ne!(*shifted, *output);
        // Construction refuses key/value heads that do not divide the query heads.
        assert!(matches!(
            Program::new(attention_fixture(AttentionGeometry { model_dim: 4, n_heads: 3, n_kv_heads: 2, head_dim: 2 })),
            Err(ProgramError::Attention { error: AttentionProgramError::KeyValueHeadsDoNotDivide { .. }, .. })
        ));
        // Positions that do not name every row are refused by the owner.
        assert!(matches!(
            program.execute(&source, &MaskAssignment::all_on(), vec![x.clone()], Vec::new(), &[0, 1]),
            Err(ExecutionError::Program(ProgramError::Attention { .. }))
        ));
        assert_eq!(program.output_mask_dependence().degree(), Some(0));
        let decoded = Program::decode(&program.encode(lattice()).expect("an encodable program"))
            .expect("the codeword of a valid program");
        assert_eq!(decoded.parts(), &unnamed(&parts));
    }

    fn unnamed(parts: &ProgramParts) -> ProgramParts {
        let mut parts = parts.clone();
        for decl in &mut parts.parameters {
            decl.name.clear();
        }
        for decl in &mut parts.slots {
            decl.name.clear();
        }
        for decl in &mut parts.controls {
            decl.name.clear();
        }
        for group in &mut parts.mask_groups {
            group.name.clear();
        }
        for body in &mut parts.bodies {
            body.name.clear();
        }
        parts
    }

    fn copy_bits(message: &BitString, count: u64) -> BitString {
        let mut reader = message.reader();
        let mut copy = BitString::new();
        for _ in 0..count {
            copy.push_bit(reader.read_bit().expect("a bit of the message"));
        }
        copy
    }

    /// The declared precision of the codeword tests: the step `2^-24`, on which every
    /// fixture constant lies.
    fn lattice() -> DeclaredPrecision {
        DeclaredPrecision::new(24).expect("a normal dyadic step")
    }

    fn codeword_fixture() -> ProgramParts {
        ProgramParts {
            parameters: vec![
                ParameterDecl { name: "anchored".to_string(), controls: controls(&[1, 0]) },
                ParameterDecl { name: "gain".to_string(), controls: Vec::new() },
            ],
            slots: vec![SlotDecl { name: "stream".to_string() }],
            controls: named_controls(3),
            mask_groups: vec![
                MaskGroup { name: "tied".to_string(), controls: controls(&[0, 2]) },
                MaskGroup { name: "anchor".to_string(), controls: controls(&[1]) },
            ],
            bodies: vec![
                body(
                    "entry",
                    1,
                    vec![
                        Node::Input { port: 0 },
                        Node::Refine { native: BodyId(1), mechanism: BodyId(2), arguments: vec![NodeId(0)] },
                        Node::Sum { terms: vec![term(0, None), term(1, Some(2))] },
                        Node::Write { slot: SlotId(0), value: NodeId(2) },
                    ],
                ),
                body(
                    "native",
                    1,
                    vec![
                        Node::Input { port: 0 },
                        // epsilon = 2^-20, a point of the codeword lattice.
                        native(NativePrimitive::RmsNorm { epsilon: 9.5367431640625e-7, gain: ParameterSlot(1) }, &[0]),
                        native(NativePrimitive::Activation { activation: NativeActivation::Silu }, &[1]),
                    ],
                ),
                body(
                    "mechanism",
                    1,
                    vec![Node::Input { port: 0 }, linear(0, 0), native(NativePrimitive::Hadamard, &[1, 1])],
                ),
            ],
            entry: BodyId(0),
        }
    }

    #[test]
    fn a_program_codeword_decodes_to_the_same_executable_program() {
        let mut lengths = Vec::new();
        for parts in [validation_base(), codeword_fixture()] {
            let program = Program::new(parts.clone()).expect("a valid program");
            let message = program.encode(lattice()).expect("an encodable program");
            let decoded = Program::decode(&message).expect("the codeword of a valid program");
            assert_eq!(decoded.parts(), &unnamed(&parts));
            lengths.push(message.len_bits());
        }
        // Positive control that the code carries structure: one more controlled term
        // lengthens it.
        let mut longer = codeword_fixture();
        longer.controls.push(ControlDecl { name: "c3".to_string() });
        longer.mask_groups.push(MaskGroup { name: "extra".to_string(), controls: controls(&[3]) });
        longer.bodies[0].nodes[2] =
            Node::Sum { terms: vec![term(0, None), term(1, Some(2)), term(1, Some(3))] };
        let longer_bits = Program::new(longer)
            .expect("a valid program")
            .encode(lattice())
            .expect("an encodable program")
            .len_bits();
        assert!(longer_bits > lengths[1]);

        // Real constants go through the declared-precision lattice: an off-lattice
        // epsilon decodes to its nearest lattice point, a different program, whose
        // distortion is measured after decoding.
        let mut off_lattice = codeword_fixture();
        off_lattice.bodies[1].nodes[1] =
            native(NativePrimitive::RmsNorm { epsilon: 1e-6, gain: ParameterSlot(1) }, &[0]);
        let coarse = DeclaredPrecision::new(8).expect("a normal dyadic step");
        let quantized = Program::decode(
            &Program::new(off_lattice.clone())
                .expect("a valid program")
                .encode(coarse)
                .expect("an encodable program"),
        )
        .expect("the codeword of a valid program");
        assert_ne!(quantized.parts(), &unnamed(&off_lattice));

        let message = Program::new(codeword_fixture())
            .expect("a valid program")
            .encode(lattice())
            .expect("an encodable program");
        // The decoder follows the encoder's path, so a message missing its last bit
        // runs out exactly at the field that bit belongs to.
        assert!(matches!(
            Program::decode(&copy_bits(&message, message.len_bits() - 1)),
            Err(ProgramError::Codec { error: CodecError::UnexpectedEnd { .. } })
        ));
        let mut trailing = copy_bits(&message, message.len_bits());
        trailing.push_bit(false);
        assert!(matches!(
            Program::decode(&trailing),
            Err(ProgramError::Codec { error: CodecError::TrailingBits { .. } })
        ));
        // A valid empty lattice message, then a hostile parameter count: the count is
        // refused before anything is allocated.
        let mut hostile = BitString::new();
        LatticeCode::encode(&[], lattice())
            .expect("an empty lattice code")
            .write(&mut hostile)
            .expect("a lattice message");
        encode_prefix_integer(&mut hostile, 1_u64 << 40).expect("a prefix integer");
        assert!(matches!(
            Program::decode(&hostile),
            Err(ProgramError::Codec { error: CodecError::InvalidCodeword(..) })
        ));
    }

    fn oriented(parameter: u32, orientation: TieOrientation, argument: u32) -> Node {
        native(NativePrimitive::Linear { weight: ParameterSlot(parameter), orientation }, &[argument])
    }

    #[test]
    fn a_transposed_linear_node_multiplies_by_the_stored_matrix_through_the_owner() {
        let w = array![[1.0, -0.5, 0.25], [0.5, 2.0, -1.0]];
        let x = array![[1.0, -2.0], [0.5, 4.0]];
        let build = |orientation: TieOrientation| {
            Program::new(ProgramParts {
                parameters: free_parameters(1),
                slots: Vec::new(),
                controls: Vec::new(),
                mask_groups: Vec::new(),
                bodies: vec![body("entry", 1, vec![Node::Input { port: 0 }, oriented(0, orientation, 0)])],
                entry: BodyId(0),
            })
            .expect("a valid program")
        };
        let source = DenseParameters::new(vec![DenseTensor::Matrix(w.clone())]);
        let transposed = build(TieOrientation::Transpose);
        // `x A^T` with `A = Theta^T` is `x Theta`.
        assert_eq!(run(&transposed, &source, &MaskAssignment::all_on(), &x), x.dot(&w));
        // Positive control: an identity use of the same matrix takes rows of width 3, and
        // the owner refuses rows of width 2.
        assert!(matches!(
            build(TieOrientation::Identity).execute(
                &source,
                &MaskAssignment::all_on(),
                vec![x.clone()],
                Vec::new(),
                &row_positions(&x),
            ),
            Err(ExecutionError::Source { error: DenseParameterError::Apply { .. }, .. })
        ));
        // The orientation travels in the document and, as its own label, in the codeword.
        let text = transposed.to_json().expect("a serializable program");
        assert!(text.contains("\"orientation\":\"Transpose\""));
        assert_eq!(Program::from_json(&text).expect("a round trip"), transposed);
        let decoded = Program::decode(&transposed.encode(lattice()).expect("an encodable program"))
            .expect("the codeword of a valid program");
        assert_eq!(decoded.parts(), &unnamed(transposed.parts()));
        assert_ne!(decoded.parts(), &unnamed(build(TieOrientation::Identity).parts()));
    }

    /// The registry of a tied embedding `embed` (4, 3) and a bias (4), with the use sites
    /// `(storage, ordinal, map)`.
    fn registry_with(sites: &[(&str, usize, UseMap)]) -> TensorRegistry {
        let mut registry = TensorRegistry::default();
        registry
            .register_storage(TensorId("embed".to_string()), Array2::<f64>::zeros((4, 3)).view().into_dyn())
            .expect("a new storage name");
        registry
            .register_storage(TensorId("bias".to_string()), Array1::<f64>::zeros(4).view().into_dyn())
            .expect("a new storage name");
        for &(storage, ordinal, map) in sites {
            let storage = TensorId(storage.to_string());
            registry
                .register_use_site(UseSiteId::read(&storage, ordinal), storage, map)
                .expect("a new use site of registered storage");
        }
        registry
    }

    fn named_program(names: &[&str], nodes: Vec<Node>) -> Program {
        Program::new(ProgramParts {
            parameters: names
                .iter()
                .map(|name| ParameterDecl { name: name.to_string(), controls: Vec::new() })
                .collect(),
            slots: Vec::new(),
            controls: Vec::new(),
            mask_groups: Vec::new(),
            bodies: vec![body("entry", 1, nodes)],
            entry: BodyId(0),
        })
        .expect("a valid program")
    }

    /// `embed` read as `x embed^T`, a bias added, then the head read as `x embed`.
    fn tied_nodes(head: u32) -> Vec<Node> {
        vec![
            Node::Input { port: 0 },
            oriented(0, TieOrientation::Identity, 0),
            native(NativePrimitive::AddBias { bias: ParameterSlot(1) }, &[1]),
            oriented(head, TieOrientation::Transpose, 2),
        ]
    }

    const TIED_SITES: [(&str, usize, UseMap); 3] = [
        ("embed", 0, UseMap::Linear(TieOrientation::Identity)),
        ("bias", 0, UseMap::Stored),
        ("embed", 1, UseMap::Linear(TieOrientation::Transpose)),
    ];

    #[test]
    fn use_sites_bind_teacher_reads_in_execution_order() {
        let registry = registry_with(&TIED_SITES);
        let reads = named_program(&["embed", "bias"], tied_nodes(0))
            .bind_use_sites(&registry)
            .expect("the tied fixture binds");
        let bound: Vec<(String, UseMap)> = reads.iter().map(|read| (read.site.0.clone(), read.map)).collect();
        assert_eq!(
            bound,
            vec![
                ("embed#0".to_string(), UseMap::Linear(TieOrientation::Identity)),
                ("bias#0".to_string(), UseMap::Stored),
                ("embed#1".to_string(), UseMap::Linear(TieOrientation::Transpose)),
            ]
        );
        // A global edit of the storage reaches exactly its bound reads.
        let global: BTreeSet<&UseSiteId> =
            registry.use_sites_of(&TensorId("embed".to_string())).into_iter().collect();
        let bound_embed: BTreeSet<&UseSiteId> =
            reads.iter().filter(|read| read.parameter == ParameterSlot(0)).map(|read| &read.site).collect();
        assert_eq!(global, bound_embed);
    }

    #[test]
    fn use_site_binding_refuses_every_map_disagreement() {
        let tied = named_program(&["embed", "bias"], tied_nodes(0));
        let mut flipped = TIED_SITES;
        flipped[2].2 = UseMap::Linear(TieOrientation::Identity);
        assert_eq!(
            tied.bind_use_sites(&registry_with(&flipped)),
            Err(BindingError::MapMismatch {
                site: UseSiteId("embed#1".to_string()),
                declared: UseMap::Linear(TieOrientation::Transpose),
                registered: UseMap::Linear(TieOrientation::Identity),
            })
        );
        let mut stored = TIED_SITES;
        stored[0].2 = UseMap::Stored;
        assert_eq!(
            tied.bind_use_sites(&registry_with(&stored)),
            Err(BindingError::MapMismatch {
                site: UseSiteId("embed#0".to_string()),
                declared: UseMap::Linear(TieOrientation::Identity),
                registered: UseMap::Stored,
            })
        );
        // A matrix read by a stored use (a bias here, a lookup in a model) where the registry
        // records a linear use.
        let lookup = named_program(
            &["embed"],
            vec![Node::Input { port: 0 }, native(NativePrimitive::AddBias { bias: ParameterSlot(0) }, &[0])],
        );
        assert_eq!(
            lookup.bind_use_sites(&registry_with(&[("embed", 0, UseMap::Linear(TieOrientation::Identity))])),
            Err(BindingError::MapMismatch {
                site: UseSiteId("embed#0".to_string()),
                declared: UseMap::Stored,
                registered: UseMap::Linear(TieOrientation::Identity),
            })
        );
    }

    #[test]
    fn use_site_binding_refuses_a_missing_or_an_unreached_site() {
        let tied = named_program(&["embed", "bias"], tied_nodes(0));
        assert!(matches!(
            tied.bind_use_sites(&registry_with(&TIED_SITES[..2])),
            Err(BindingError::UnregisteredRead { site, .. }) if site == UseSiteId("embed#1".to_string())
        ));
        let mut extra = TIED_SITES.to_vec();
        extra.push(("embed", 2, UseMap::Linear(TieOrientation::Identity)));
        assert_eq!(
            tied.bind_use_sites(&registry_with(&extra)),
            Err(BindingError::UnreadSite {
                storage: TensorId("embed".to_string()),
                site: UseSiteId("embed#2".to_string()),
            })
        );
    }

    #[test]
    fn a_parameter_named_by_an_alias_shares_its_storage_ordinals() {
        let mut registry = registry_with(&TIED_SITES);
        registry
            .register_alias(TensorId("lm_head".to_string()), TensorId("embed".to_string()))
            .expect("an alias of registered storage");
        let reads = named_program(&["embed", "bias", "lm_head"], tied_nodes(2))
            .bind_use_sites(&registry)
            .expect("an alias binds to its storage");
        // The head's read is the storage's second read, not the alias's first.
        assert_eq!(reads[2].parameter, ParameterSlot(2));
        assert_eq!(reads[2].site, UseSiteId("embed#1".to_string()));
        // Positive control: an unregistered name is refused as such.
        assert!(matches!(
            named_program(&["embed", "bias", "decoder"], tied_nodes(2)).bind_use_sites(&registry),
            Err(BindingError::Registry { parameter: ParameterSlot(2), error: LiftError::UnknownTensor(..) })
        ));
    }

    /// The teacher of the lift-source fixtures: `w` (2, 3) read as `x w^T`, a bias (2) added, then `w` read
    /// again as `x w`, with a two-component anchor on `w` whose basis matrices are `u_c r_c^T`.
    struct LiftFixture {
        w: Array2<f64>,
        bias: Array1<f64>,
        left: Array2<f64>,
        right: Array2<f64>,
        registry: TensorRegistry,
    }

    impl LiftFixture {
        fn new() -> Self {
            let w = array![[1.0, -0.5, 0.25], [0.5, 2.0, -1.0]];
            let bias = array![0.25, -0.5];
            let mut registry = TensorRegistry::default();
            for (name, values) in [("w", w.view().into_dyn()), ("bias", bias.view().into_dyn())] {
                registry.register_storage(TensorId(name.to_string()), values).expect("a new storage name");
            }
            registry
                .register_storage(TensorId("unread".to_string()), Array1::<f64>::zeros(2).view().into_dyn())
                .expect("a new storage name");
            for (storage, ordinal, map) in [
                ("w", 0, UseMap::Linear(TieOrientation::Identity)),
                ("bias", 0, UseMap::Stored),
                ("w", 1, UseMap::Linear(TieOrientation::Transpose)),
            ] {
                let storage = TensorId(storage.to_string());
                registry
                    .register_use_site(UseSiteId::read(&storage, ordinal), storage, map)
                    .expect("a new use site of registered storage");
            }
            Self {
                w,
                bias,
                left: array![[1.0, 0.5], [-0.5, 1.0]],
                right: array![[0.5, 0.0], [0.0, 1.0], [0.25, -0.5]],
                registry,
            }
        }

        fn anchor(&self) -> ResidualAnchor<'_> {
            ResidualAnchor::new(
                &self.registry,
                TensorId("w".to_string()),
                self.w.view(),
                FactoredEdit::new(self.left.clone(), self.right.clone()).expect("finite factors of equal rank"),
                vec![1, 1],
                ComponentCoefficients::Basis,
            )
            .expect("the registered teacher anchors")
        }

        /// `Theta(m) = m_Delta w + sum_c (m_c - m_Delta) u_c r_c^T`, formed densely for the reference.
        fn edited(&self, masks: [f64; 3]) -> Array2<f64> {
            let mut theta = &self.w * masks[0];
            for c in 0..2 {
                let left = self.left.column(c).insert_axis(ndarray::Axis(1));
                let basis = left.dot(&self.right.column(c).insert_axis(ndarray::Axis(0)));
                theta = theta + basis * (masks[c + 1] - masks[0]);
            }
            theta
        }
    }

    /// The tied program over the fixture: `w` declares the controls `w_controls`, the bias declares `bias_controls`.
    fn lift_program(w_controls: &[u32], bias_controls: &[u32], control_count: u32) -> Program {
        Program::new(ProgramParts {
            parameters: vec![
                ParameterDecl { name: "w".to_string(), controls: controls(w_controls) },
                ParameterDecl { name: "bias".to_string(), controls: controls(bias_controls) },
            ],
            slots: Vec::new(),
            controls: named_controls(control_count),
            mask_groups: singleton_groups(control_count),
            bodies: vec![body(
                "entry",
                1,
                vec![
                    Node::Input { port: 0 },
                    oriented(0, TieOrientation::Identity, 0),
                    native(NativePrimitive::AddBias { bias: ParameterSlot(1) }, &[1]),
                    oriented(0, TieOrientation::Transpose, 2),
                ],
            )],
            entry: BodyId(0),
        })
        .expect("a valid program")
    }

    #[test]
    fn a_lift_source_executes_the_residual_anchor_at_every_bound_read() {
        let fixture = LiftFixture::new();
        let anchor = fixture.anchor();
        let program = lift_program(&[0, 1, 2], &[], 3);
        let source = LiftSource::new(
            &program,
            &fixture.registry,
            vec![&anchor],
            vec![(TensorId("bias".to_string()), fixture.bias.view())],
        )
        .expect("every read has a tensor and every tensor a read");
        let x = array![[1.0, -2.0, 0.5], [0.5, 4.0, -1.0]];
        let expected = |masks: [f64; 3]| {
            let theta = fixture.edited(masks);
            (x.dot(&theta.t()) + &fixture.bias).dot(&theta)
        };

        // All on, the source executes the teacher: the dense binding of the same tensors, which takes
        // no anchor controls, runs the same program without them bit for bit.
        let all_on = run(&program, &source, &MaskAssignment::all_on(), &x);
        let teacher = DenseParameters::new(vec![
            DenseTensor::Matrix(fixture.w.clone()),
            DenseTensor::Vector(fixture.bias.clone()),
        ]);
        assert_eq!(all_on, run(&lift_program(&[], &[], 0), &teacher, &MaskAssignment::all_on(), &x));
        assert_eq!(all_on, expected([1.0, 1.0, 1.0]));

        // A global edit: both tied reads of `w` execute `Theta(m)`, the residual scaled by `m_Delta`.
        let masked = run(&program, &source, &global(&[(0, 0.5), (1, 0.25), (2, -1.0)]), &x);
        assert_eq!(masked, expected([0.5, 0.25, -1.0]));
        // Positive controls: the mask moved the output, and the residual control is not ignored.
        assert_ne!(masked, all_on);
        assert_ne!(masked, expected([1.0, 0.25, -1.0]));
    }

    #[test]
    fn a_lift_source_refuses_a_read_without_a_tensor_and_a_tensor_without_a_read() {
        let fixture = LiftFixture::new();
        let anchor = fixture.anchor();
        let program = lift_program(&[0, 1, 2], &[], 3);
        let id = |name: &str| TensorId(name.to_string());
        let bias = || vec![(id("bias"), fixture.bias.view())];
        assert!(LiftSource::new(&program, &fixture.registry, vec![&anchor], bias()).is_ok());
        assert_eq!(
            LiftSource::new(&program, &fixture.registry, Vec::new(), bias()).err(),
            Some(LiftSourceError::MissingAnchor { storage: id("w") })
        );
        assert_eq!(
            LiftSource::new(&program, &fixture.registry, vec![&anchor], Vec::new()).err(),
            Some(LiftSourceError::MissingValues { storage: id("bias") })
        );
        assert_eq!(
            LiftSource::new(&program, &fixture.registry, vec![&anchor, &anchor], bias()).err(),
            Some(LiftSourceError::Duplicate { storage: id("w") })
        );
        let unread = Array1::<f64>::zeros(2);
        let mut extra = bias();
        extra.push((id("unread"), unread.view()));
        assert_eq!(
            LiftSource::new(&program, &fixture.registry, vec![&anchor], extra).err(),
            Some(LiftSourceError::UnreadStorage { storage: id("unread") })
        );
        // Values other than the registered ones: one entry changed.
        let other = array![0.25, 0.5];
        assert!(matches!(
            LiftSource::new(&program, &fixture.registry, vec![&anchor], vec![(id("bias"), other.view())]),
            Err(LiftSourceError::ValuesMismatch { storage, .. }) if storage == id("bias")
        ));
        // Two controls on a two-component anchor, which takes a residual control too.
        assert_eq!(
            LiftSource::new(&lift_program(&[0, 1], &[], 2), &fixture.registry, vec![&anchor], bias()).err(),
            Some(LiftSourceError::ControlCount { parameter: ParameterSlot(0), controls: 2, components: 2 })
        );
        assert_eq!(
            LiftSource::new(&lift_program(&[0, 1, 2], &[3], 4), &fixture.registry, vec![&anchor], bias()).err(),
            Some(LiftSourceError::ControlledStoredRead { parameter: ParameterSlot(1), controls: 1 })
        );
    }

    #[test]
    fn a_lift_source_refuses_a_read_off_the_teachers_path_and_a_map_its_site_does_not_register() {
        let fixture = LiftFixture::new();
        let anchor = fixture.anchor();
        let x = array![[1.0, -2.0, 0.5], [0.5, 4.0, -1.0]];
        // A refinement whose mechanism reads `w` too; only the native body is on the teacher's path.
        let mut registry = TensorRegistry::default();
        registry
            .register_storage(TensorId("w".to_string()), fixture.w.view().into_dyn())
            .expect("a new storage name");
        let w = TensorId("w".to_string());
        registry
            .register_use_site(UseSiteId::read(&w, 0), w.clone(), UseMap::Linear(TieOrientation::Identity))
            .expect("a new use site of registered storage");
        let refined = Program::new(ProgramParts {
            parameters: vec![ParameterDecl { name: "w".to_string(), controls: Vec::new() }],
            slots: Vec::new(),
            controls: named_controls(2),
            mask_groups: singleton_groups(2),
            bodies: vec![
                body(
                    "entry",
                    1,
                    vec![
                        Node::Input { port: 0 },
                        Node::Refine { native: BodyId(1), mechanism: BodyId(2), arguments: vec![NodeId(0)] },
                    ],
                ),
                body("native", 1, vec![Node::Input { port: 0 }, linear(0, 0)]),
                body("mechanism", 1, vec![Node::Input { port: 0 }, linear(0, 0), mask(&[0, 1], 1)]),
            ],
            entry: BodyId(0),
        })
        .expect("a valid program");
        let refined_anchor = ResidualAnchor::new(
            &registry,
            w.clone(),
            fixture.w.view(),
            FactoredEdit::new(fixture.left.clone(), fixture.right.clone()).expect("finite factors of equal rank"),
            vec![1, 1],
            ComponentCoefficients::Basis,
        )
        .expect("the registered teacher anchors");
        let source = LiftSource::new(&refined, &registry, vec![&refined_anchor], Vec::new())
            .expect("the native body's read binds");
        assert_eq!(run(&refined, &source, &MaskAssignment::all_on(), &x), x.dot(&fixture.w.t()));
        assert!(matches!(
            refined.execute(&source, &global(&[(0, 0.5)]), vec![x.clone()], Vec::new(), &row_positions(&x)),
            Err(ExecutionError::Source {
                error: LiftSourceError::UnboundRead { parameter: ParameterSlot(0), body: BodyId(2), node: NodeId(1) },
                ..
            })
        ));

        // A direct read in the other orientation, or as stored values, of the identity site `w#0`.
        let program = lift_program(&[0, 1, 2], &[], 3);
        let tied = LiftSource::new(
            &program,
            &fixture.registry,
            vec![&anchor],
            vec![(TensorId("bias".to_string()), fixture.bias.view())],
        )
        .expect("every read has a tensor and every tensor a read");
        let first = ParameterUse {
            parameter: ParameterSlot(0),
            body: BodyId(0),
            node: NodeId(1),
            invocation: &[],
            controls: &[1.0, 1.0, 1.0],
        };
        let site = UseSiteId::read(&w, 0);
        let identity = UseMap::Linear(TieOrientation::Identity);
        assert_eq!(
            tied.apply_linear(first, TieOrientation::Transpose, x.slice(ndarray::s![.., ..2])).err(),
            Some(LiftSourceError::MapMismatch {
                site: site.clone(),
                registered: identity,
                applied: UseMap::Linear(TieOrientation::Transpose),
            })
        );
        assert_eq!(
            tied.vector(first).err(),
            Some(LiftSourceError::MapMismatch { site, registered: identity, applied: UseMap::Stored })
        );
        // Positive control: the registered orientation reads the teacher.
        assert_eq!(
            tied.apply_linear(first, TieOrientation::Identity, x.view()).expect("the bound read").to_owned(),
            x.dot(&fixture.w.t())
        );
    }

    /// A dense source that records the process's admissible budget at every product, so a
    /// test reads which values the executor holds under reservations while a source runs.
    struct RecordingSource {
        dense: DenseParameters,
        remaining: std::cell::RefCell<Vec<usize>>,
    }

    impl ParameterSource for RecordingSource {
        type Error = DenseParameterError;

        fn apply_linear(
            &self,
            parameter: ParameterUse<'_>,
            orientation: TieOrientation,
            rows: ArrayView2<'_, f64>,
        ) -> Result<Governed<Array2<f64>>, Self::Error> {
            self.remaining.borrow_mut().push(MemoryGovernor::global().remaining_bytes());
            self.dense.apply_linear(parameter, orientation, rows)
        }

        fn vector(&self, parameter: ParameterUse<'_>) -> Result<Array1<f64>, Self::Error> {
            self.dense.vector(parameter)
        }
    }

    /// Every value the executor forms is reserved before it is formed and released when
    /// its last reader has run. The process-wide ledger is read inside the source, so the
    /// run is alone in its process (nextest runs each test in its own). Before the
    /// executor reserved its own values, both products read nothing reserved.
    #[test]
    fn every_value_the_executor_forms_is_reserved_while_it_is_live() {
        let w1 = array![[1.0, -0.5, 0.25], [0.5, 2.0, -1.0]];
        let w2 = array![[1.0, 0.5], [-1.0, 0.25], [0.5, 0.5], [2.0, -0.25], [0.0, 1.0]];
        let x = array![[1.0, -2.0, 0.5], [0.5, 4.0, -1.0], [0.25, 0.0, 2.0], [-1.0, 1.0, 1.0]];
        let program = Program::new(ProgramParts {
            parameters: free_parameters(2),
            slots: Vec::new(),
            controls: Vec::new(),
            mask_groups: Vec::new(),
            bodies: vec![body(
                "entry",
                1,
                vec![
                    Node::Input { port: 0 },
                    linear(0, 0),
                    native(NativePrimitive::Activation { activation: NativeActivation::Relu }, &[1]),
                    linear(1, 2),
                ],
            )],
            entry: BodyId(0),
        })
        .expect("a valid program");
        let source = RecordingSource {
            dense: DenseParameters::new(vec![DenseTensor::Matrix(w1.clone()), DenseTensor::Matrix(w2.clone())]),
            remaining: std::cell::RefCell::new(Vec::new()),
        };
        let bytes = |rows: usize, cols: usize| rows * cols * std::mem::size_of::<f64>();
        let before = MemoryGovernor::global().remaining_bytes();
        let execution = program
            .execute(&source, &MaskAssignment::all_on(), vec![x.clone()], Vec::new(), &row_positions(&x))
            .expect("execution of a valid program");
        assert_eq!(*execution.output, row_map(&row_map(&x, &w1).mapv(|t| t.max(0.0)), &w2));
        let recorded = source.remaining.borrow().clone();
        assert_eq!(recorded.len(), 2, "one reading per product");
        // At the first product the caller's input and the Input node's copy are live.
        assert_eq!(before - recorded[0], 2 * bytes(4, 3));
        // At the second, the input and the Relu output; the copy and the first product
        // were released once their last reader ran.
        assert_eq!(before - recorded[1], bytes(4, 3) + bytes(4, 2));
        // After the run only the output the caller holds is reserved, and dropping it
        // releases the last of it.
        assert_eq!(before - MemoryGovernor::global().remaining_bytes(), bytes(4, 5));
        drop(execution);
        assert_eq!(MemoryGovernor::global().remaining_bytes(), before);
    }

    /// A value that does not fit the process's memory budget is refused as a typed
    /// `ExecutionError::Memory` before it is formed, and the values reserved before it are
    /// released. The test holds the ledger (bookkeeping, no allocation) down to less than two
    /// input footprints: the caller's input is admitted, and the Input node's copy is refused.
    /// Released, the same run executes.
    #[test]
    fn a_value_beyond_the_memory_budget_is_refused_typed_before_it_is_formed() {
        let w = array![[1.0, -0.5, 0.25], [0.5, 2.0, -1.0]];
        let x = array![[1.0, -2.0, 0.5], [0.5, 4.0, -1.0]];
        let program = Program::new(ProgramParts {
            parameters: free_parameters(1),
            slots: Vec::new(),
            controls: Vec::new(),
            mask_groups: Vec::new(),
            bodies: vec![body("entry", 1, vec![Node::Input { port: 0 }, linear(0, 0)])],
            entry: BodyId(0),
        })
        .expect("a valid program");
        let source = DenseParameters::new(vec![DenseTensor::Matrix(w.clone())]);
        let governor = MemoryGovernor::global();
        let input_bytes = x.len() * std::mem::size_of::<f64>();
        let left = 2 * input_bytes - 1;
        let hold = governor
            .try_reserve(governor.remaining_bytes() - left, "test: hold the budget below two inputs")
            .expect("a ledger hold within the budget");
        assert_eq!(governor.remaining_bytes(), left);
        let refused = program.execute(&source, &MaskAssignment::all_on(), vec![x.clone()], Vec::new(), &row_positions(&x));
        assert!(matches!(refused, Err(ExecutionError::Memory { .. })), "the Input node's copy does not fit");
        assert_eq!(governor.remaining_bytes(), left, "the admitted input was released with the refusal");
        drop(hold);
        assert_eq!(run(&program, &source, &MaskAssignment::all_on(), &x), row_map(&x, &w));
    }
}

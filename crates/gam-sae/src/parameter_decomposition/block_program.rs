//! Mechanism programs over [`super::block`]'s attention-only layers and gated decoder blocks
//! (#2951).
//!
//! [`attention_layer_program`] writes a norm-free, MLP-free decoder layer
//! `h' = h + concat_h(z_h) W_Oᵀ` as one [`Program`]: the residual input, the query,
//! key and value `Linear` nodes on it, `CausalSelfAttention` on the three projected
//! rows, the output `Linear` node on the head-mixed rows, and the residual `Sum`.
//! Formal parameter `k` is projection `k` of [`PROJECTIONS`], and a `Write` slot
//! keeps each stage in [`LayerStage`] order.
//!
//! The layer binds its own program. [`NativeAttentionLayer`] is a dense
//! [`ParameterSource`] on its stored tensors, and [`ComponentAttentionLayer`] binds
//! each projection's anchor controls to its component coordinates: every control
//! at `1` reads the stored tensor, and any other setting reads `U diag(m) R` through
//! apply.rs without forming it. [`ComponentLayerProgram`] declares one control per
//! component and one mask group per control, so a mask on the program is exactly a
//! [`super::block::ProjectionRead::Components`] read of the layer.
//!
//! # Exact replay
//!
//! The program's nodes run the owners the layer runs, in the layer's order:
//! apply.rs's `native_linear` or `apply_anchored_linear` for each read,
//! attention.rs's `attend_projected` on exact rows, and the residual add
//! `h.clone() += write`. So an executed program is bit-identical to
//! [`NativeAttentionLayer::execute`] with native reads, and a masked program to
//! [`ComponentAttentionLayer::execute`] with component reads, at every stage.
//!
//! # A whole gated block
//!
//! [`GatedBlockProgram`] writes a gated (SwiGLU) decoder block, the A12 block shape, as one program:
//! norm, the query, key and value projections with their biases, `CausalSelfAttention`, the output
//! projection, the residual `Sum`, norm, the gate and up projections, `SwiGlu`, the down projection
//! and the residual `Sum`, in the block's layout, with a slot per stage. Its formal parameters
//! are named after the source's tensors, so a registry binding can read them, and
//! [`GatedBlockSource`] binds a [`NativeGatedBlock`]'s own tensors, borrowed. Its projections run
//! through apply.rs, where the native block runs its owners' kernels, so the two agree stage by
//! stage within the sum of their rounding bands, and the norm stages bit for bit. A block that
//! normalizes each head's queries and keys (Qwen3 `q_norm`, `k_norm`) gets a `HeadRmsNorm` node
//! after each of those projections' bias, the owner its native attention runs, with the two gains
//! as formal parameters.
//!
//! # Validity domain
//!
//! A program reads each projection globally. A read at declared positions
//! ([`super::block::ProjectionRead::Edited`]) has no program node, since program masks are scoped
//! by invocation, not by sequence position.

use super::apply::{ApplyError, FactorView, apply_anchored_linear, native_linear};
use super::block::{
    AttentionProjection, BlockError, ComponentAttentionLayer, NativeAttentionLayer, NativeGatedBlock, NativeNorm,
};
use super::gated_rewrite::ResidualLayout;
use super::lift::TieOrientation;
use super::program::{
    Body, BodyId, ControlDecl, ControlId, MaskAssignment, MaskGroup, MaskGroupId, NativePrimitive, Node,
    NodeId, ParameterDecl, ParameterSlot, ParameterSource, ParameterUse, Program, ProgramError, ProgramParts,
    SlotDecl, SlotId, SumTerm,
};
use gam_runtime::resource::Governed;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::fmt;

/// The formal parameters of a layer program, in slot order.
pub const PROJECTIONS: [AttentionProjection; 4] = [
    AttentionProjection::Query,
    AttentionProjection::Key,
    AttentionProjection::Value,
    AttentionProjection::Output,
];

/// A stage a layer program keeps in a `Write` slot, in slot order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerStage {
    /// `x W_Qᵀ`.
    Queries,
    /// `x W_Kᵀ`.
    Keys,
    /// `x W_Vᵀ`.
    Values,
    /// The head-mixed rows `concat_h z_h` before the output projection.
    Mixed,
    /// `concat_h(z_h) W_Oᵀ`, the layer's write with no residual.
    Write,
}

impl LayerStage {
    pub const ALL: [Self; 5] = [Self::Queries, Self::Keys, Self::Values, Self::Mixed, Self::Write];

    /// The stage's slot in a layer program.
    pub fn slot(self) -> SlotId {
        SlotId(match self {
            Self::Queries => 0,
            Self::Keys => 1,
            Self::Values => 2,
            Self::Mixed => 3,
            Self::Write => 4,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::Queries => "queries",
            Self::Keys => "keys",
            Self::Values => "values",
            Self::Mixed => "mixed",
            Self::Write => "write",
        }
    }
}

/// The formal parameter of one projection in a layer program.
pub fn parameter_of(projection: AttentionProjection) -> ParameterSlot {
    ParameterSlot(match projection {
        AttentionProjection::Query => 0,
        AttentionProjection::Key => 1,
        AttentionProjection::Value => 2,
        AttentionProjection::Output => 3,
    })
}

/// A refused layer program, program binding or mask.
#[derive(Debug)]
pub enum LayerProgramError {
    /// The program graph or a mask assignment was refused.
    Program(ProgramError),
    /// A formal parameter outside the layer's four projections.
    Unbound { parameter: ParameterSlot },
    /// A layer program reads no vector parameter.
    NotAVector { parameter: ParameterSlot },
    /// A dense layer has no component anchor for controls to act on.
    DenseControls { projection: AttentionProjection },
    /// The controls at a use do not match the projection's component count.
    ControlCount {
        projection: AttentionProjection,
        expected: usize,
        found: usize,
    },
    /// apply.rs refused a read.
    Apply {
        projection: AttentionProjection,
        error: ApplyError,
    },
    /// The layer refused its component factors.
    Block(BlockError),
}

impl From<ProgramError> for LayerProgramError {
    fn from(error: ProgramError) -> Self {
        Self::Program(error)
    }
}

impl From<BlockError> for LayerProgramError {
    fn from(error: BlockError) -> Self {
        Self::Block(error)
    }
}

impl fmt::Display for LayerProgramError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Program(error) => write!(formatter, "the layer program was refused: {error}"),
            Self::Unbound { parameter } => write!(
                formatter,
                "formal parameter {} is not one of an attention layer's four projections",
                parameter.0
            ),
            Self::NotAVector { parameter } => write!(
                formatter,
                "formal parameter {} is a projection matrix, not a vector",
                parameter.0
            ),
            Self::DenseControls { projection } => write!(
                formatter,
                "the {projection} projection of a dense layer has no component anchor for controls"
            ),
            Self::ControlCount {
                projection,
                expected,
                found,
            } => write!(
                formatter,
                "the {projection} projection has {expected} components, and its use carries {found} controls"
            ),
            Self::Apply { projection, error } => write!(formatter, "the {projection} read was refused: {error}"),
            Self::Block(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for LayerProgramError {}

fn projection_of(parameter: ParameterSlot) -> Result<AttentionProjection, LayerProgramError> {
    PROJECTIONS
        .get(parameter.index())
        .copied()
        .ok_or(LayerProgramError::Unbound { parameter })
}

/// The entry body of a layer program: every node reads the residual input or an
/// earlier stage, and each stage is kept in its slot.
fn layer_body(layer: &NativeAttentionLayer) -> Body {
    let linear = |projection: AttentionProjection, argument: u32| Node::Native {
        primitive: NativePrimitive::Linear {
            weight: parameter_of(projection),
            orientation: TieOrientation::Identity,
        },
        arguments: vec![NodeId(argument)],
    };
    let write = |stage: LayerStage, value: u32| Node::Write {
        slot: stage.slot(),
        value: NodeId(value),
    };
    Body {
        name: "attention layer".to_string(),
        inputs: 1,
        nodes: vec![
            // 0: the residual stream h.
            Node::Input { port: 0 },
            // 1-6: the three projections of h, each kept.
            linear(AttentionProjection::Query, 0),
            write(LayerStage::Queries, 1),
            linear(AttentionProjection::Key, 0),
            write(LayerStage::Keys, 3),
            linear(AttentionProjection::Value, 0),
            write(LayerStage::Values, 5),
            // 7-8: joint causal attention on the exact projected rows.
            Node::Native {
                primitive: NativePrimitive::CausalSelfAttention {
                    geometry: layer.geometry(),
                    rotary: layer.rotary().clone(),
                    score_scale: layer.score_scale(),
                },
                arguments: vec![NodeId(2), NodeId(4), NodeId(6)],
            },
            write(LayerStage::Mixed, 7),
            // 9-10: the output projection of the head-mixed rows.
            linear(AttentionProjection::Output, 8),
            write(LayerStage::Write, 9),
            // 11: h + write.
            Node::Sum {
                terms: vec![
                    SumTerm {
                        value: NodeId(0),
                        control: None,
                    },
                    SumTerm {
                        value: NodeId(10),
                        control: None,
                    },
                ],
            },
        ],
        output: NodeId(11),
    }
}

fn layer_parts(layer: &NativeAttentionLayer, parameters: Vec<ParameterDecl>) -> ProgramParts {
    ProgramParts {
        parameters,
        slots: LayerStage::ALL
            .iter()
            .map(|stage| SlotDecl {
                name: stage.name().to_string(),
            })
            .collect(),
        controls: Vec::new(),
        mask_groups: Vec::new(),
        bodies: vec![layer_body(layer)],
        entry: BodyId(0),
    }
}

/// The native program of `layer`, bound by `layer` itself as a dense
/// [`ParameterSource`]. Executed with [`MaskAssignment::all_on`], empty slots and
/// the layer's positions, it is bit-identical to [`NativeAttentionLayer::execute`]
/// with native reads.
pub fn attention_layer_program(layer: &NativeAttentionLayer) -> Result<Program, LayerProgramError> {
    let parameters = PROJECTIONS
        .iter()
        .map(|projection| ParameterDecl {
            name: projection.to_string(),
            controls: Vec::new(),
        })
        .collect();
    Ok(Program::new(layer_parts(layer, parameters))?)
}

/// The component program of a [`ComponentAttentionLayer`]: projection `p` carries
/// one anchor control per component of its exact factor, and each control is its
/// own mask group.
#[derive(Clone, Debug)]
pub struct ComponentLayerProgram {
    program: Program,
    groups: [Vec<MaskGroupId>; 4],
}

impl ComponentLayerProgram {
    /// The program of `layer`, bound by `layer` itself.
    pub fn new(layer: &ComponentAttentionLayer) -> Result<Self, LayerProgramError> {
        let mut parts = layer_parts(layer.native(), Vec::new());
        let mut groups: [Vec<MaskGroupId>; 4] = Default::default();
        for projection in PROJECTIONS {
            let mut controls = Vec::new();
            for component in 0..layer.factor(projection).components() {
                let control = ControlId(parts.controls.len() as u32);
                let name = format!("{projection}.{component}");
                parts.controls.push(ControlDecl { name: name.clone() });
                groups[parameter_of(projection).index()].push(MaskGroupId(parts.mask_groups.len() as u32));
                parts.mask_groups.push(MaskGroup {
                    name,
                    controls: vec![control],
                });
                controls.push(control);
            }
            parts.parameters.push(ParameterDecl {
                name: projection.to_string(),
                controls,
            });
        }
        Ok(Self {
            program: Program::new(parts)?,
            groups,
        })
    }

    pub fn program(&self) -> &Program {
        &self.program
    }

    /// The mask group of one component of one projection.
    pub fn group(&self, projection: AttentionProjection, component: usize) -> Option<MaskGroupId> {
        self.groups[parameter_of(projection).index()].get(component).copied()
    }

    /// The global assignment of the component masks `masks[k]` on projection
    /// `PROJECTIONS[k]`: the program's reading of [`super::block::ProjectionRead::Components`].
    pub fn masks(&self, masks: [ArrayView1<'_, f64>; 4]) -> Result<MaskAssignment, LayerProgramError> {
        let mut assignment = MaskAssignment::all_on();
        for (projection, mask) in PROJECTIONS.into_iter().zip(masks) {
            let groups = &self.groups[parameter_of(projection).index()];
            if mask.len() != groups.len() {
                return Err(LayerProgramError::ControlCount {
                    projection,
                    expected: groups.len(),
                    found: mask.len(),
                });
            }
            for (&group, &value) in groups.iter().zip(mask.iter()) {
                assignment.set(group, Vec::new(), value)?;
            }
        }
        Ok(assignment)
    }
}

/// The stored tensor in the use's orientation.
fn oriented(weight: ArrayView2<'_, f64>, orientation: TieOrientation) -> ArrayView2<'_, f64> {
    match orientation {
        TieOrientation::Identity => weight,
        TieOrientation::Transpose => weight.reversed_axes(),
    }
}

impl ParameterSource for NativeAttentionLayer {
    type Error = LayerProgramError;

    fn apply_linear(
        &self,
        parameter: ParameterUse<'_>,
        orientation: TieOrientation,
        rows: ArrayView2<'_, f64>,
    ) -> Result<Governed<Array2<f64>>, LayerProgramError> {
        let projection = projection_of(parameter.parameter)?;
        if !parameter.controls.is_empty() {
            return Err(LayerProgramError::DenseControls { projection });
        }
        native_linear(oriented(self.weight(projection), orientation), rows)
            .map_err(|error| LayerProgramError::Apply { projection, error })
    }

    fn vector(&self, parameter: ParameterUse<'_>) -> Result<Array1<f64>, LayerProgramError> {
        Err(LayerProgramError::NotAVector {
            parameter: parameter.parameter,
        })
    }
}

impl ParameterSource for ComponentAttentionLayer {
    type Error = LayerProgramError;

    fn apply_linear(
        &self,
        parameter: ParameterUse<'_>,
        orientation: TieOrientation,
        rows: ArrayView2<'_, f64>,
    ) -> Result<Governed<Array2<f64>>, LayerProgramError> {
        let projection = projection_of(parameter.parameter)?;
        let refused = |error| LayerProgramError::Apply { projection, error };
        let weight = oriented(self.native().weight(projection), orientation);
        let components = self.factor(projection).components();
        if parameter.controls.len() != components {
            return Err(LayerProgramError::ControlCount {
                projection,
                expected: components,
                found: parameter.controls.len(),
            });
        }
        // Every control at 1 is the stored tensor on its original path.
        if parameter.controls.iter().all(|&control| control == 1.0) {
            return native_linear(weight, rows).map_err(refused);
        }
        let stored = self.components(projection)?;
        let factor = match orientation {
            TieOrientation::Identity => stored,
            TieOrientation::Transpose => FactorView::new(stored.right(), stored.left()).map_err(refused)?,
        };
        apply_anchored_linear(weight, 0.0, factor, ArrayView1::from(parameter.controls), rows).map_err(refused)
    }

    fn vector(&self, parameter: ParameterUse<'_>) -> Result<Array1<f64>, LayerProgramError> {
        Err(LayerProgramError::NotAVector {
            parameter: parameter.parameter,
        })
    }
}

/// A formal parameter of a gated block's program. A block whose norms are RMSNorms
/// declares no norm bias.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatedBlockParameter {
    AttentionNormGain,
    AttentionNormBias,
    QueryWeight,
    QueryBias,
    KeyWeight,
    KeyBias,
    ValueWeight,
    ValueBias,
    OutputWeight,
    OutputBias,
    MlpNormGain,
    MlpNormBias,
    Gate,
    Up,
    Down,
    /// The per-head query norm's gain, declared only by a block that has one.
    QueryNormGain,
    /// The per-head key norm's gain, declared only by a block that has one.
    KeyNormGain,
}

impl fmt::Display for GatedBlockParameter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::AttentionNormGain => "input_layernorm.weight",
            Self::AttentionNormBias => "input_layernorm.bias",
            Self::QueryWeight => "self_attn.q_proj.weight",
            Self::QueryBias => "self_attn.q_proj.bias",
            Self::KeyWeight => "self_attn.k_proj.weight",
            Self::KeyBias => "self_attn.k_proj.bias",
            Self::ValueWeight => "self_attn.v_proj.weight",
            Self::ValueBias => "self_attn.v_proj.bias",
            Self::OutputWeight => "self_attn.o_proj.weight",
            Self::OutputBias => "self_attn.o_proj.bias",
            Self::MlpNormGain => "post_attention_layernorm.weight",
            Self::MlpNormBias => "post_attention_layernorm.bias",
            Self::Gate => "mlp.gate_proj.weight",
            Self::Up => "mlp.up_proj.weight",
            Self::Down => "mlp.down_proj.weight",
            Self::QueryNormGain => "self_attn.q_norm.weight",
            Self::KeyNormGain => "self_attn.k_norm.weight",
        })
    }
}

/// A stage a gated block's program keeps in a `Write` slot, in slot order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatedBlockStage {
    /// `N₁(h)`.
    AttentionInput,
    /// `x W_Qᵀ + b_Q` on the attention input, before a per-head query norm.
    Queries,
    /// `x W_Kᵀ + b_K`, before a per-head key norm.
    Keys,
    Values,
    /// The head-mixed rows before the output projection.
    Mixed,
    /// `concat_h(z_h) W_Oᵀ + b_O`, the attention write with no residual.
    AttentionWrite,
    /// `N₂` of the stream the MLP reads: `h + A` sequential, `h` parallel.
    MlpInput,
    /// `gate_proj` of the MLP input.
    Gate,
    /// `up_proj` of the MLP input.
    Up,
    /// `s(gate) ⊙ up`, the `down_proj` input.
    Hidden,
    /// `down_proj`, the MLP write with no residual.
    MlpWrite,
}

impl GatedBlockStage {
    pub const ALL: [Self; 11] = [
        Self::AttentionInput,
        Self::Queries,
        Self::Keys,
        Self::Values,
        Self::Mixed,
        Self::AttentionWrite,
        Self::MlpInput,
        Self::Gate,
        Self::Up,
        Self::Hidden,
        Self::MlpWrite,
    ];

    /// The stage's slot in a gated block's program.
    pub fn slot(self) -> SlotId {
        SlotId(match self {
            Self::AttentionInput => 0,
            Self::Queries => 1,
            Self::Keys => 2,
            Self::Values => 3,
            Self::Mixed => 4,
            Self::AttentionWrite => 5,
            Self::MlpInput => 6,
            Self::Gate => 7,
            Self::Up => 8,
            Self::Hidden => 9,
            Self::MlpWrite => 10,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::AttentionInput => "attention input",
            Self::Queries => "queries",
            Self::Keys => "keys",
            Self::Values => "values",
            Self::Mixed => "mixed",
            Self::AttentionWrite => "attention write",
            Self::MlpInput => "MLP input",
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Hidden => "hidden",
            Self::MlpWrite => "MLP write",
        }
    }
}

/// A refused gated-block program or binding.
#[derive(Debug)]
pub enum GatedBlockProgramError {
    /// The program graph was refused.
    Program(ProgramError),
    /// A formal parameter the program does not declare.
    Unbound { parameter: ParameterSlot },
    /// A vector parameter read as a matrix.
    NotAMatrix { parameter: GatedBlockParameter },
    /// A matrix parameter read as a vector.
    NotAVector { parameter: GatedBlockParameter },
    /// A native block has no component anchor for controls to act on.
    Controlled { parameter: GatedBlockParameter },
    /// apply.rs refused a read.
    Apply {
        parameter: GatedBlockParameter,
        error: ApplyError,
    },
}

impl From<ProgramError> for GatedBlockProgramError {
    fn from(error: ProgramError) -> Self {
        Self::Program(error)
    }
}

impl fmt::Display for GatedBlockProgramError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Program(error) => write!(formatter, "the gated block program was refused: {error}"),
            Self::Unbound { parameter } => {
                write!(formatter, "formal parameter {} is not one of the block program's", parameter.0)
            }
            Self::NotAMatrix { parameter } => write!(formatter, "{parameter} is a vector, not a matrix"),
            Self::NotAVector { parameter } => write!(formatter, "{parameter} is a matrix, not a vector"),
            Self::Controlled { parameter } => {
                write!(formatter, "{parameter} is a native tensor with no component anchor for controls")
            }
            Self::Apply { parameter, error } => write!(formatter, "the {parameter} read was refused: {error}"),
        }
    }
}

impl std::error::Error for GatedBlockProgramError {}

/// A body's nodes, pushed in topological order.
struct BodyNodes(Vec<Node>);

impl BodyNodes {
    fn push(&mut self, node: Node) -> NodeId {
        self.0.push(node);
        NodeId(self.0.len() as u32 - 1)
    }

    fn write(&mut self, stage: GatedBlockStage, value: NodeId) -> NodeId {
        self.push(Node::Write {
            slot: stage.slot(),
            value,
        })
    }

    fn native(&mut self, primitive: NativePrimitive, arguments: Vec<NodeId>) -> NodeId {
        self.push(Node::Native { primitive, arguments })
    }

    fn sum(&mut self, values: &[NodeId]) -> NodeId {
        self.push(Node::Sum {
            terms: values
                .iter()
                .map(|&value| SumTerm { value, control: None })
                .collect(),
        })
    }

    fn linear(&mut self, weight: ParameterSlot, input: NodeId) -> NodeId {
        self.native(
            NativePrimitive::Linear {
                weight,
                orientation: TieOrientation::Identity,
            },
            vec![input],
        )
    }

    /// `x Wᵀ + b`: a `Linear` node, then an `AddBias` node.
    fn affine(&mut self, (weight, bias): (ParameterSlot, ParameterSlot), input: NodeId) -> NodeId {
        let product = self.linear(weight, input);
        self.native(NativePrimitive::AddBias { bias }, vec![product])
    }
}

/// Declares the next formal parameter.
fn declare(parameters: &mut Vec<GatedBlockParameter>, parameter: GatedBlockParameter) -> ParameterSlot {
    parameters.push(parameter);
    ParameterSlot(parameters.len() as u32 - 1)
}

/// The norm primitive of `norm`, declaring its gain and, for a LayerNorm, its bias.
fn norm_primitive(
    parameters: &mut Vec<GatedBlockParameter>,
    norm: &NativeNorm,
    gain: GatedBlockParameter,
    bias: GatedBlockParameter,
) -> NativePrimitive {
    match norm {
        NativeNorm::Rms { epsilon, .. } => NativePrimitive::RmsNorm {
            epsilon: *epsilon,
            gain: declare(parameters, gain),
        },
        NativeNorm::Layer { epsilon, .. } => {
            let gain = declare(parameters, gain);
            NativePrimitive::LayerNorm {
                epsilon: *epsilon,
                gain,
                bias: declare(parameters, bias),
            }
        }
    }
}

/// A gated (SwiGLU) decoder block written as one mechanism program: norm, the query, key
/// and value `Linear` and `AddBias` nodes, `CausalSelfAttention`, the output projection,
/// the residual `Sum`, norm, the gate and up `Linear` nodes, `SwiGlu`, the down `Linear`
/// node and the residual `Sum`, in the block's residual layout. Each stage is kept in its
/// [`GatedBlockStage`] slot, and the formal parameters are named after the source's
/// tensors ([`GatedBlockParameter`]), in the order [`Self::parameters`] lists them.
///
/// The residual sums run in the source's order: `(h + a) + m` sequential and
/// `(m + a) + h` parallel, as [`super::gated_rewrite::decoder_layer`] adds them.
#[derive(Clone, Debug)]
pub struct GatedBlockProgram {
    program: Program,
    parameters: Vec<GatedBlockParameter>,
}

impl GatedBlockProgram {
    /// The program of `block`'s structure: its norms, attention geometry and layout. Its
    /// tensors are bound at execution ([`Self::source`]).
    pub fn new(block: &NativeGatedBlock) -> Result<Self, GatedBlockProgramError> {
        let attention = block.attention();
        let mut parameters = Vec::new();
        let attention_norm = norm_primitive(
            &mut parameters,
            block.attention_norm(),
            GatedBlockParameter::AttentionNormGain,
            GatedBlockParameter::AttentionNormBias,
        );
        let query = (
            declare(&mut parameters, GatedBlockParameter::QueryWeight),
            declare(&mut parameters, GatedBlockParameter::QueryBias),
        );
        let key = (
            declare(&mut parameters, GatedBlockParameter::KeyWeight),
            declare(&mut parameters, GatedBlockParameter::KeyBias),
        );
        let value = (
            declare(&mut parameters, GatedBlockParameter::ValueWeight),
            declare(&mut parameters, GatedBlockParameter::ValueBias),
        );
        let output = (
            declare(&mut parameters, GatedBlockParameter::OutputWeight),
            declare(&mut parameters, GatedBlockParameter::OutputBias),
        );
        let mlp_norm = norm_primitive(
            &mut parameters,
            block.mlp_norm(),
            GatedBlockParameter::MlpNormGain,
            GatedBlockParameter::MlpNormBias,
        );
        let gate = declare(&mut parameters, GatedBlockParameter::Gate);
        let up = declare(&mut parameters, GatedBlockParameter::Up);
        let down = declare(&mut parameters, GatedBlockParameter::Down);
        let head_dim = attention.geometry().head_dim;
        let head_norms = attention.query_key_norm().map(|norm| {
            let head_norm = |gain| NativePrimitive::HeadRmsNorm {
                head_dim,
                epsilon: norm.epsilon(),
                gain,
            };
            (
                head_norm(declare(&mut parameters, GatedBlockParameter::QueryNormGain)),
                head_norm(declare(&mut parameters, GatedBlockParameter::KeyNormGain)),
            )
        });

        let mut nodes = BodyNodes(Vec::new());
        let residual = nodes.push(Node::Input { port: 0 });
        let normalized = nodes.native(attention_norm, vec![residual]);
        let attention_input = nodes.write(GatedBlockStage::AttentionInput, normalized);
        let queries = nodes.affine(query, attention_input);
        let mut queries = nodes.write(GatedBlockStage::Queries, queries);
        let keys = nodes.affine(key, attention_input);
        let mut keys = nodes.write(GatedBlockStage::Keys, keys);
        if let Some((query_norm, key_norm)) = head_norms {
            queries = nodes.native(query_norm, vec![queries]);
            keys = nodes.native(key_norm, vec![keys]);
        }
        let values = nodes.affine(value, attention_input);
        let values = nodes.write(GatedBlockStage::Values, values);
        let mixed = nodes.native(
            NativePrimitive::CausalSelfAttention {
                geometry: attention.geometry(),
                rotary: attention.rotary().clone(),
                score_scale: attention.score_scale(),
            },
            vec![queries, keys, values],
        );
        let mixed = nodes.write(GatedBlockStage::Mixed, mixed);
        let attention_write = nodes.affine(output, mixed);
        let attention_write = nodes.write(GatedBlockStage::AttentionWrite, attention_write);
        let stream = match block.layout() {
            ResidualLayout::Sequential => nodes.sum(&[residual, attention_write]),
            ResidualLayout::Parallel => residual,
        };
        let normalized = nodes.native(mlp_norm, vec![stream]);
        let mlp_input = nodes.write(GatedBlockStage::MlpInput, normalized);
        let gated = nodes.linear(gate, mlp_input);
        let gated = nodes.write(GatedBlockStage::Gate, gated);
        let lifted = nodes.linear(up, mlp_input);
        let lifted = nodes.write(GatedBlockStage::Up, lifted);
        let hidden = nodes.native(NativePrimitive::SwiGlu, vec![gated, lifted]);
        let hidden = nodes.write(GatedBlockStage::Hidden, hidden);
        let mlp_write = nodes.linear(down, hidden);
        let mlp_write = nodes.write(GatedBlockStage::MlpWrite, mlp_write);
        let block_output = match block.layout() {
            ResidualLayout::Sequential => nodes.sum(&[stream, mlp_write]),
            ResidualLayout::Parallel => nodes.sum(&[mlp_write, attention_write, residual]),
        };
        let parts = ProgramParts {
            parameters: parameters
                .iter()
                .map(|parameter| ParameterDecl {
                    name: parameter.to_string(),
                    controls: Vec::new(),
                })
                .collect(),
            slots: GatedBlockStage::ALL
                .iter()
                .map(|stage| SlotDecl {
                    name: stage.name().to_string(),
                })
                .collect(),
            controls: Vec::new(),
            mask_groups: Vec::new(),
            bodies: vec![Body {
                name: "gated decoder block".to_string(),
                inputs: 1,
                nodes: nodes.0,
                output: block_output,
            }],
            entry: BodyId(0),
        };
        Ok(Self {
            program: Program::new(parts)?,
            parameters,
        })
    }

    pub fn program(&self) -> &Program {
        &self.program
    }

    /// The source tensor each formal parameter reads, in slot order.
    pub fn parameters(&self) -> &[GatedBlockParameter] {
        &self.parameters
    }

    /// `block`'s tensors bound to this program's formal parameters, borrowed, not copied.
    pub fn source<'a>(&'a self, block: &'a NativeGatedBlock) -> GatedBlockSource<'a> {
        GatedBlockSource {
            block,
            parameters: &self.parameters,
        }
    }
}

/// A native gated block's tensors as a [`GatedBlockProgram`]'s dense
/// [`ParameterSource`]: it refuses anchor controls and reads a Transpose use through
/// the stored tensor's transposed view.
#[derive(Clone, Copy, Debug)]
pub struct GatedBlockSource<'a> {
    block: &'a NativeGatedBlock,
    parameters: &'a [GatedBlockParameter],
}

impl GatedBlockSource<'_> {
    fn parameter(&self, parameter: ParameterUse<'_>) -> Result<GatedBlockParameter, GatedBlockProgramError> {
        let declared = self
            .parameters
            .get(parameter.parameter.index())
            .copied()
            .ok_or(GatedBlockProgramError::Unbound {
                parameter: parameter.parameter,
            })?;
        if parameter.controls.is_empty() {
            Ok(declared)
        } else {
            Err(GatedBlockProgramError::Controlled { parameter: declared })
        }
    }
}

impl ParameterSource for GatedBlockSource<'_> {
    type Error = GatedBlockProgramError;

    fn apply_linear(
        &self,
        parameter: ParameterUse<'_>,
        orientation: TieOrientation,
        rows: ArrayView2<'_, f64>,
    ) -> Result<Governed<Array2<f64>>, GatedBlockProgramError> {
        let declared = self.parameter(parameter)?;
        let attention = self.block.attention();
        let weight = match declared {
            GatedBlockParameter::QueryWeight => attention.query().weight.view(),
            GatedBlockParameter::KeyWeight => attention.key().weight.view(),
            GatedBlockParameter::ValueWeight => attention.value().weight.view(),
            GatedBlockParameter::OutputWeight => attention.output().weight.view(),
            GatedBlockParameter::Gate => self.block.mlp().gate(),
            GatedBlockParameter::Up => self.block.mlp().up(),
            GatedBlockParameter::Down => self.block.mlp().down(),
            GatedBlockParameter::AttentionNormGain
            | GatedBlockParameter::AttentionNormBias
            | GatedBlockParameter::QueryBias
            | GatedBlockParameter::KeyBias
            | GatedBlockParameter::ValueBias
            | GatedBlockParameter::OutputBias
            | GatedBlockParameter::MlpNormGain
            | GatedBlockParameter::MlpNormBias
            | GatedBlockParameter::QueryNormGain
            | GatedBlockParameter::KeyNormGain => {
                return Err(GatedBlockProgramError::NotAMatrix { parameter: declared });
            }
        };
        native_linear(oriented(weight, orientation), rows)
            .map_err(|error| GatedBlockProgramError::Apply { parameter: declared, error })
    }

    fn vector(&self, parameter: ParameterUse<'_>) -> Result<Array1<f64>, GatedBlockProgramError> {
        let declared = self.parameter(parameter)?;
        let attention = self.block.attention();
        let not_a_vector = GatedBlockProgramError::NotAVector { parameter: declared };
        match declared {
            GatedBlockParameter::AttentionNormGain => Ok(norm_gain(self.block.attention_norm())),
            GatedBlockParameter::AttentionNormBias => norm_bias(self.block.attention_norm()).ok_or(not_a_vector),
            GatedBlockParameter::QueryBias => Ok(attention.query().bias.clone()),
            GatedBlockParameter::KeyBias => Ok(attention.key().bias.clone()),
            GatedBlockParameter::ValueBias => Ok(attention.value().bias.clone()),
            GatedBlockParameter::OutputBias => Ok(attention.output().bias.clone()),
            GatedBlockParameter::MlpNormGain => Ok(norm_gain(self.block.mlp_norm())),
            GatedBlockParameter::MlpNormBias => norm_bias(self.block.mlp_norm()).ok_or(not_a_vector),
            GatedBlockParameter::QueryNormGain => attention
                .query_key_norm()
                .map(|norm| norm.query_gain().to_owned())
                .ok_or(not_a_vector),
            GatedBlockParameter::KeyNormGain => attention
                .query_key_norm()
                .map(|norm| norm.key_gain().to_owned())
                .ok_or(not_a_vector),
            GatedBlockParameter::QueryWeight
            | GatedBlockParameter::KeyWeight
            | GatedBlockParameter::ValueWeight
            | GatedBlockParameter::OutputWeight
            | GatedBlockParameter::Gate
            | GatedBlockParameter::Up
            | GatedBlockParameter::Down => Err(not_a_vector),
        }
    }
}

fn norm_gain(norm: &NativeNorm) -> Array1<f64> {
    match norm {
        NativeNorm::Rms { gain, .. } | NativeNorm::Layer { gain, .. } => gain.clone(),
    }
}

fn norm_bias(norm: &NativeNorm) -> Option<Array1<f64>> {
    match norm {
        NativeNorm::Rms { .. } => None,
        NativeNorm::Layer { bias, .. } => Some(bias.clone()),
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter_decomposition::attention::{
        AffineProjection, AttentionGeometry, NativeAttention, ProjectedRows, QueryKeyNorm, RotaryCausalAttention,
        RotaryEmbedding, RotaryPairing, head_rms_norm_with_radius,
    };
    use crate::parameter_decomposition::gated_rewrite::{NativeSwiglu, swiglu_hidden};
    use crate::parameter_decomposition::receipts::affine_stage_band;
    use gam_linalg::roundoff::accumulation_growth;
    use crate::parameter_decomposition::block::{
        AttentionLayerExecution, AttentionLayerReads, ComponentMasks, ProjectionRead,
    };
    use crate::parameter_decomposition::program::{Execution, ExecutionError};
    use crate::parameter_decomposition::rewrite::ComponentRead;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    const WIDTH: usize = 8;
    const TOKENS: usize = 6;
    const COMPONENTS: usize = 9;

    fn geometry() -> AttentionGeometry {
        AttentionGeometry {
            model_dim: WIDTH,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
        }
    }

    /// One rotated plane per head, so the program's attention node carries a real
    /// rotary embedding.
    fn rotary() -> RotaryEmbedding {
        RotaryEmbedding {
            pairing: RotaryPairing::HalfSplit,
            inverse_frequencies: vec![1.0, 0.25],
            attention_scaling: 1.0,
        }
    }

    fn eighths(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_simple_fn((rows, cols), || rng.random_range(-16..=16) as f64 / 8.0)
    }

    /// A layer whose four weights factor exactly through overcomplete reads in eighths,
    /// over a residual in thirds of eighths: not dyadic, so every linear read rounds,
    /// and a changed route changes bits.
    struct Fixture {
        candidates: [Array2<f64>; 4],
        reads: [Array2<f64>; 4],
        residual: Array2<f64>,
        positions: Vec<i64>,
    }

    impl Fixture {
        fn new(seed: u64) -> Self {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut candidate = || eighths(&mut rng, WIDTH, COMPONENTS);
            let candidates = [candidate(), candidate(), candidate(), candidate()];
            let mut read = || eighths(&mut rng, COMPONENTS, WIDTH);
            let reads = [read(), read(), read(), read()];
            Self {
                candidates,
                reads,
                residual: Array2::from_shape_simple_fn((TOKENS, WIDTH), || rng.random_range(-48..=48) as f64 / 24.0),
                positions: (3..3 + TOKENS as i64).collect(),
            }
        }

        fn native(&self) -> NativeAttentionLayer {
            let [query, key, value, output] = [0, 1, 2, 3].map(|k| self.candidates[k].dot(&self.reads[k]));
            NativeAttentionLayer::new(geometry(), rotary(), 0.5, query, key, value, output)
                .expect("fixture weights match the geometry")
        }

        fn component(&self) -> ComponentAttentionLayer {
            let read = |k: usize| ComponentRead {
                read: self.reads[k].view(),
                candidate_write: self.candidates[k].view(),
            };
            ComponentAttentionLayer::new(self.native(), read(0), read(1), read(2), read(3))
                .expect("random overcomplete reads are resolved")
        }

        fn run<S: ParameterSource>(
            &self,
            program: &Program,
            source: &S,
            masks: &MaskAssignment,
        ) -> Result<Execution, ExecutionError<S::Error>> {
            program.execute(
                source,
                masks,
                vec![self.residual.clone()],
                vec![None; LayerStage::ALL.len()],
                &self.positions,
            )
        }
    }

    fn bits(values: &Array2<f64>) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    fn slot<'a>(execution: &'a Execution, stage: LayerStage) -> &'a Array2<f64> {
        execution.slots[stage.slot().index()]
            .as_ref()
            .expect("every stage slot is written")
    }

    /// Every stage of an executed program against the same stage of an executed layer.
    fn stage_bits_agree(execution: &Execution, layer: &AttentionLayerExecution) -> Vec<(LayerStage, bool)> {
        LayerStage::ALL
            .iter()
            .map(|&stage| {
                let expected: &Array2<f64> = match stage {
                    LayerStage::Queries => &*layer.queries,
                    LayerStage::Keys => &*layer.keys,
                    LayerStage::Values => &*layer.values,
                    LayerStage::Mixed => &layer.attention.mixed,
                    LayerStage::Write => &*layer.write,
                };
                (stage, bits(slot(execution, stage)) == bits(expected))
            })
            .collect()
    }

    /// Eighths per projection: continuous in `[0, 1]`, binary, and signed in `[-3/2, 3/2]`.
    fn mask_family(rng: &mut StdRng) -> [(&'static str, [Array1<f64>; 4]); 3] {
        let mut draw = |low: i32, high: i32, denominator: f64| {
            let mut vector =
                || Array1::from_shape_simple_fn(COMPONENTS, || rng.random_range(low..=high) as f64 / denominator);
            [vector(), vector(), vector(), vector()]
        };
        [
            ("continuous", draw(0, 8, 8.0)),
            ("binary", draw(0, 1, 1.0)),
            ("signed", draw(-12, 12, 8.0)),
        ]
    }

    fn all_ones() -> [Array1<f64>; 4] {
        [
            Array1::ones(COMPONENTS),
            Array1::ones(COMPONENTS),
            Array1::ones(COMPONENTS),
            Array1::ones(COMPONENTS),
        ]
    }

    fn views(masks: &[Array1<f64>; 4]) -> [ArrayView1<'_, f64>; 4] {
        [0, 1, 2, 3].map(|k| masks[k].view())
    }

    fn component_reads(masks: &[Array1<f64>; 4]) -> AttentionLayerReads<'_> {
        AttentionLayerReads {
            query: ProjectionRead::Components(ComponentMasks::uniform(masks[0].view())),
            key: ProjectionRead::Components(ComponentMasks::uniform(masks[1].view())),
            value: ProjectionRead::Components(ComponentMasks::uniform(masks[2].view())),
            output: ProjectionRead::Components(ComponentMasks::uniform(masks[3].view())),
        }
    }

    /// The native program of a layer runs the layer's owners in the layer's order, so
    /// its output and every kept stage are bit-identical to the layer's native
    /// execution. Positive control: a program whose query and key nodes read each
    /// other's tensors (every parameter still read once, so the program is valid)
    /// changes the output bits.
    #[test]
    fn a_native_layer_program_replays_the_native_layer_bit_for_bit() {
        let fixture = Fixture::new(2991);
        let layer = fixture.native();
        let program = attention_layer_program(&layer).expect("the layer program is valid");
        let native = layer
            .execute(AttentionLayerReads::native(), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
            .expect("native layer");
        let executed = fixture
            .run(&program, &layer, &MaskAssignment::all_on())
            .expect("native layer program");
        assert!(
            bits(&executed.output) == bits(&native.output),
            "the native program must replay the native layer's output bit for bit"
        );
        for (stage, agrees) in stage_bits_agree(&executed, &native) {
            assert!(agrees, "the native program's {stage:?} slot must equal the layer's stage bit for bit");
        }

        let mut rewired = program.parts().clone();
        let read = |projection: AttentionProjection| Node::Native {
            primitive: NativePrimitive::Linear {
                weight: parameter_of(projection),
                orientation: TieOrientation::Identity,
            },
            arguments: vec![NodeId(0)],
        };
        rewired.bodies[0].nodes[1] = read(AttentionProjection::Key);
        rewired.bodies[0].nodes[3] = read(AttentionProjection::Query);
        let rewired = Program::new(rewired).expect("a program with its query and key reads swapped is still valid");
        let control = fixture
            .run(&rewired, &layer, &MaskAssignment::all_on())
            .expect("swapped program");
        assert!(
            bits(&control.output) != bits(&native.output),
            "control: the bit comparison must see query and key nodes that read each other's tensors"
        );
    }

    /// For continuous, binary and signed component masks on all four projections, the
    /// masked component program is the layer's component reads, bit for bit at every
    /// stage. Positive control: one value mask entry moved by 1/8 moves the output bits.
    #[test]
    fn a_masked_component_program_replays_component_reads_bit_for_bit() {
        let fixture = Fixture::new(2992);
        let layer = fixture.component();
        let program = ComponentLayerProgram::new(&layer).expect("the component program is valid");
        let mut rng = StdRng::seed_from_u64(2993);
        for (kind, masks) in mask_family(&mut rng) {
            for (k, mask) in masks.iter().enumerate() {
                assert!(
                    mask.iter().any(|&entry| entry != 1.0),
                    "{kind} masks, {}: an all-ones mask reads the stored tensor, not the component read",
                    PROJECTIONS[k]
                );
            }
            let expected = layer
                .execute(component_reads(&masks), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
                .expect("component reads");
            let assignment = program.masks(views(&masks)).expect("masks of the right lengths");
            let executed = fixture
                .run(program.program(), &layer, &assignment)
                .expect("masked component program");
            assert!(
                bits(&executed.output) == bits(&expected.output),
                "{kind} masks: the masked program must replay the component reads' output bit for bit"
            );
            for (stage, agrees) in stage_bits_agree(&executed, &expected) {
                assert!(agrees, "{kind} masks: the {stage:?} slot must equal the component reads' stage bit for bit");
            }

            let mut moved = masks.clone();
            moved[2][0] += 0.125;
            let control = fixture
                .run(program.program(), &layer, &program.masks(views(&moved)).expect("moved masks"))
                .expect("moved-mask program");
            assert!(
                bits(&control.output) != bits(&executed.output),
                "{kind} masks: control: a value mask entry moved by 1/8 must move the output bits"
            );
        }
    }

    /// With no mask assigned every control is 1, so the component program reads the
    /// stored tensors on their original paths and replays the native layer. Positive
    /// control: the layer's factored all-ones reads, the same algebra, give other bits.
    #[test]
    fn an_all_on_component_program_executes_the_stored_tensors() {
        let fixture = Fixture::new(2994);
        let layer = fixture.component();
        let program = ComponentLayerProgram::new(&layer).expect("the component program is valid");
        let native = layer
            .execute(AttentionLayerReads::native(), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
            .expect("native reads");
        let executed = fixture
            .run(program.program(), &layer, &MaskAssignment::all_on())
            .expect("all-on component program");
        assert!(
            bits(&executed.output) == bits(&native.output),
            "the all-on component program must execute the stored tensors on their original paths"
        );
        for (stage, agrees) in stage_bits_agree(&executed, &native) {
            assert!(agrees, "the all-on {stage:?} slot must equal the native layer's stage bit for bit");
        }
        let ones = all_ones();
        let factored = layer
            .execute(component_reads(&ones), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
            .expect("factored all-ones reads");
        assert!(
            bits(&factored.output) != bits(&native.output),
            "control: the bit comparison must distinguish the factored all-ones reads"
        );
    }

    /// Typed refusals, each beside the accepted neighbour: a dense layer bound to a
    /// program with anchor controls, a mask of another length, and a non-finite mask.
    #[test]
    fn layer_programs_refuse_bindings_and_masks_outside_their_domain() {
        let fixture = Fixture::new(2995);
        let layer = fixture.component();
        let program = ComponentLayerProgram::new(&layer).expect("the component program is valid");
        let dense = attention_layer_program(layer.native()).expect("the dense program is valid");
        assert!(
            fixture.run(&dense, layer.native(), &MaskAssignment::all_on()).is_ok(),
            "control: the dense layer binds its own program"
        );
        match fixture.run(program.program(), layer.native(), &MaskAssignment::all_on()) {
            Err(ExecutionError::Source {
                error: LayerProgramError::DenseControls { projection },
                ..
            }) => assert_eq!(projection, AttentionProjection::Query, "the first read refuses"),
            other => panic!("a dense layer must refuse anchor controls, got {other:?}"),
        }

        let ones = all_ones();
        assert!(program.masks(views(&ones)).is_ok(), "control: masks of the component count are accepted");
        let mut short = ones.clone();
        short[1] = Array1::ones(COMPONENTS - 1);
        match program.masks(views(&short)) {
            Err(LayerProgramError::ControlCount {
                projection,
                expected,
                found,
            }) => assert_eq!(
                (projection, expected, found),
                (AttentionProjection::Key, COMPONENTS, COMPONENTS - 1),
                "the short key mask is named"
            ),
            other => panic!("a mask of another length must be refused, got {other:?}"),
        }
        let mut infinite = ones;
        infinite[3][4] = f64::INFINITY;
        assert!(
            matches!(program.masks(views(&infinite)), Err(LayerProgramError::Program(_))),
            "a non-finite mask must be refused by the program's assignment"
        );
    }

    const HIDDEN: usize = 12;

    /// A gated block on eighths weights with biases, over a residual in thirds of eighths:
    /// Llama-style (RMSNorm, sequential) or parallel with LayerNorm. `shift` moves the query
    /// weight's and the gate weight's first entries, for the positive controls.
    /// A gated block in eighths. With `head_norm` its attention normalizes each head's queries
    /// and keys (Qwen3 `q_norm`, `k_norm`) with gains in eighths, drawn after every other
    /// tensor, so the other tensors match the norm-free block of the same seed.
    fn gated_block(
        layout: ResidualLayout,
        seed: u64,
        shift: f64,
        head_norm: bool,
    ) -> (NativeGatedBlock, Array2<f64>, Vec<i64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut matrix = |rows: usize, cols: usize| eighths(&mut rng, rows, cols);
        let (mut query, key, value, output) =
            (matrix(WIDTH, WIDTH), matrix(WIDTH, WIDTH), matrix(WIDTH, WIDTH), matrix(WIDTH, WIDTH));
        let (mut gate, up, down) = (matrix(HIDDEN, WIDTH), matrix(HIDDEN, WIDTH), matrix(WIDTH, HIDDEN));
        let mut vector = |len: usize| Array1::from_shape_simple_fn(len, || rng.random_range(-8..=8) as f64 / 8.0);
        let (query_bias, key_bias, value_bias, output_bias) = (vector(WIDTH), vector(WIDTH), vector(WIDTH), vector(WIDTH));
        let (attention_gain, attention_bias, mlp_gain, mlp_bias) =
            (vector(WIDTH), vector(WIDTH), vector(WIDTH), vector(WIDTH));
        query[[0, 0]] += shift;
        gate[[0, 0]] += shift;
        let attention = NativeAttention::new(
            geometry(),
            rotary(),
            0.5,
            AffineProjection {
                weight: query,
                bias: query_bias,
            },
            AffineProjection {
                weight: key,
                bias: key_bias,
            },
            AffineProjection {
                weight: value,
                bias: value_bias,
            },
            AffineProjection {
                weight: output,
                bias: output_bias,
            },
        )
        .expect("fixture attention tensors match the geometry");
        let residual = Array2::from_shape_simple_fn((TOKENS, WIDTH), || rng.random_range(-48..=48) as f64 / 24.0);
        let attention = if head_norm {
            let head_dim = geometry().head_dim;
            let mut gain = || Array1::from_shape_simple_fn(head_dim, || rng.random_range(4..=12) as f64 / 8.0);
            let (query_gain, key_gain) = (gain(), gain());
            attention
                .with_query_key_norm(1.0e-6, query_gain, key_gain)
                .expect("head-width gains")
        } else {
            attention
        };
        let norm = |gain: Array1<f64>, bias: Array1<f64>| match layout {
            ResidualLayout::Sequential => NativeNorm::Rms { epsilon: 1.0e-6, gain },
            ResidualLayout::Parallel => NativeNorm::Layer {
                epsilon: 1.0e-5,
                gain,
                bias,
            },
        };
        let block = NativeGatedBlock::new(
            layout,
            norm(attention_gain, attention_bias),
            attention,
            norm(mlp_gain, mlp_bias),
            NativeSwiglu::new(gate, up, down).expect("fixture SwiGLU shapes compose"),
        );
        (block, residual, (3..3 + TOKENS as i64).collect())
    }

    fn violations(left: &Array2<f64>, right: &Array2<f64>, band: &Array2<f64>) -> usize {
        left.iter()
            .zip(right.iter())
            .zip(band.iter())
            .filter(|((a, b), bound)| (*a - *b).abs() > **bound)
            .count()
    }

    fn gated_slot(execution: &Execution, stage: GatedBlockStage) -> &Array2<f64> {
        execution.slots[stage.slot().index()]
            .as_ref()
            .expect("every stage slot is written")
    }

    fn run_gated(
        program: &GatedBlockProgram,
        block: &NativeGatedBlock,
        residual: &Array2<f64>,
        positions: &[i64],
    ) -> Execution {
        program
            .program()
            .execute(
                &program.source(block),
                &MaskAssignment::all_on(),
                vec![residual.clone()],
                vec![None; GatedBlockStage::ALL.len()],
                positions,
            )
            .expect("gated block program")
    }

    /// The program route's radius at its attention write, against the exact sublayer at the
    /// program's attention input: each projection's band `affine_stage_band` (`Linear` then
    /// `AddBias`, `d + 1` roundings) enters the attention core as its input radius, through
    /// the owner's per-head norm radius when the block normalizes queries and keys, and the
    /// output projection adds its own band plus `|W_O|` times the mixed radius.
    fn program_attention_radius(block: &NativeGatedBlock, execution: &Execution, positions: &[i64]) -> Array2<f64> {
        let attention = block.attention();
        let input = gated_slot(execution, GatedBlockStage::AttentionInput);
        let band = |projection: &AffineProjection, rows: &Array2<f64>| {
            affine_stage_band(projection.weight.view(), Some(projection.bias.view()), rows.view())
                .expect("fixture shapes")
        };
        let (query_radius, key_radius, value_radius) =
            (band(attention.query(), input), band(attention.key(), input), band(attention.value(), input));
        let core = RotaryCausalAttention::new(attention.geometry(), attention.rotary().clone(), attention.score_scale())
            .expect("the fixture attention core");
        let normed = |stage, radius: &Array2<f64>, gain: fn(&QueryKeyNorm) -> ArrayView1<'_, f64>| {
            let rows = gated_slot(execution, stage);
            match attention.query_key_norm() {
                None => (rows.clone(), radius.clone()),
                Some(norm) => head_rms_norm_with_radius(
                    ProjectedRows {
                        values: rows.view(),
                        radius: radius.view(),
                    },
                    attention.geometry().head_dim,
                    norm.epsilon(),
                    gain(norm),
                )
                .expect("finite head rows"),
            }
        };
        let (queries, query_radius) = normed(GatedBlockStage::Queries, &query_radius, QueryKeyNorm::query_gain);
        let (keys, key_radius) = normed(GatedBlockStage::Keys, &key_radius, QueryKeyNorm::key_gain);
        let with_radius = core
            .attend_projected(
                ProjectedRows {
                    values: queries.view(),
                    radius: query_radius.view(),
                },
                ProjectedRows {
                    values: keys.view(),
                    radius: key_radius.view(),
                },
                ProjectedRows {
                    values: gated_slot(execution, GatedBlockStage::Values).view(),
                    radius: value_radius.view(),
                },
                positions,
            )
            .expect("finite projected rows");
        let mixed = gated_slot(execution, GatedBlockStage::Mixed);
        assert!(
            bits(&with_radius.mixed) == bits(mixed),
            "the radius arithmetic must leave the program's mixed rows' bits alone"
        );
        let carried = with_radius.mixed_radius.dot(&attention.output().weight.mapv(f64::abs).t())
            / (1.0 - accumulation_growth(mixed.ncols() + 1));
        band(attention.output(), mixed) + &carried
    }

    /// A gated block's program is a receipt-grade executor of the block. Fed the program's own
    /// previous stage, the native block's owners agree with every program stage: the norm stages
    /// bit for bit (the same `MaskedNorm`); the attention write within the sum of the native
    /// attention's radius and the program route's radius; the gate, up and down projections
    /// within the two routes' affine bands (apply.rs on one side, the native SwiGLU's products on
    /// the other); the SwiGLU hidden rows and the residual sums bit for bit. Both layouts, and a
    /// Qwen3-style block whose attention normalizes each head's queries and keys, where the
    /// program's `HeadRmsNorm` nodes run the native attention's own norm owner.
    /// Positive controls: a query weight entry and a gate weight entry moved by 1e-9 each leave
    /// their stage's band.
    #[test]
    fn a_gated_block_program_agrees_with_the_native_block_stage_by_stage() {
        for (layout, seed, head_norm) in [
            (ResidualLayout::Sequential, 3001, false),
            (ResidualLayout::Parallel, 3002, false),
            (ResidualLayout::Sequential, 3004, true),
        ] {
            let (block, residual, positions) = gated_block(layout, seed, 0.0, head_norm);
            let program = GatedBlockProgram::new(&block).expect("the block program is valid");
            let executed = run_gated(&program, &block, &residual, &positions);
            let native = block.execute(residual.view(), &positions).expect("native block");
            let slot = |stage| gated_slot(&executed, stage);
            assert!(
                bits(slot(GatedBlockStage::AttentionInput)) == bits(&native.attention_input),
                "{layout:?}: the attention norm is the same MaskedNorm on the same rows"
            );
            let attention_band =
                program_attention_radius(&block, &executed, &positions) + &native.attention.output_radius;
            assert_eq!(
                violations(slot(GatedBlockStage::AttentionWrite), &native.attention.output, &attention_band),
                0,
                "{layout:?}: the program's attention write left the two routes' summed radii"
            );
            let stream = match layout {
                ResidualLayout::Sequential => &residual + slot(GatedBlockStage::AttentionWrite),
                ResidualLayout::Parallel => residual.clone(),
            };
            let mlp_input = block.mlp_norm().as_masked_norm().apply(stream.view()).expect("finite stream");
            assert!(
                bits(slot(GatedBlockStage::MlpInput)) == bits(&mlp_input),
                "{layout:?}: the MLP norm is the same MaskedNorm on the program's stream"
            );
            let native_mlp = block
                .mlp()
                .execute_stages(slot(GatedBlockStage::MlpInput).view())
                .expect("native SwiGLU stages");
            for (stage, weight, native_rows) in [
                (GatedBlockStage::Gate, block.mlp().gate(), &native_mlp.gate),
                (GatedBlockStage::Up, block.mlp().up(), &native_mlp.up),
            ] {
                let band =
                    affine_stage_band(weight, None, slot(GatedBlockStage::MlpInput).view()).expect("fixture shapes");
                assert_eq!(
                    violations(slot(stage), native_rows, &(&band * 2.0)),
                    0,
                    "{layout:?}: the program's {stage:?} left the two routes' affine bands"
                );
            }
            let hidden = swiglu_hidden(slot(GatedBlockStage::Gate).view(), slot(GatedBlockStage::Up).view())
                .expect("finite gate and up rows");
            assert!(
                bits(slot(GatedBlockStage::Hidden)) == bits(&hidden),
                "{layout:?}: the SwiGLU node is the owner's swiglu_hidden on the program's rows"
            );
            let native_write = slot(GatedBlockStage::Hidden).dot(&block.mlp().down().t());
            let band = affine_stage_band(block.mlp().down(), None, slot(GatedBlockStage::Hidden).view())
                .expect("fixture shapes");
            assert_eq!(
                violations(slot(GatedBlockStage::MlpWrite), &native_write, &(&band * 2.0)),
                0,
                "{layout:?}: the program's MLP write left the two routes' affine bands"
            );
            let composed = match layout {
                ResidualLayout::Sequential => &stream + slot(GatedBlockStage::MlpWrite),
                ResidualLayout::Parallel => {
                    &(slot(GatedBlockStage::MlpWrite) + slot(GatedBlockStage::AttentionWrite)) + &residual
                }
            };
            assert!(
                bits(&executed.output) == bits(&composed),
                "{layout:?}: the residual sums run in the source's order"
            );

            // Positive controls: the program of a block whose query and gate weights moved by 1e-9.
            let (moved, residual, positions) = gated_block(layout, seed, 1.0e-9, head_norm);
            let shifted = run_gated(&program, &moved, &residual, &positions);
            assert!(
                violations(
                    gated_slot(&shifted, GatedBlockStage::AttentionWrite),
                    &native.attention.output,
                    &attention_band
                ) > 0,
                "{layout:?}: the attention band must resolve a 1e-9 query weight move"
            );
            let shifted_input = gated_slot(&shifted, GatedBlockStage::MlpInput);
            let unmoved_gate = block
                .mlp()
                .execute_stages(shifted_input.view())
                .expect("native SwiGLU stages")
                .gate;
            let band = affine_stage_band(block.mlp().gate(), None, shifted_input.view()).expect("fixture shapes");
            assert!(
                violations(gated_slot(&shifted, GatedBlockStage::Gate), &unmoved_gate, &(&band * 2.0)) > 0,
                "{layout:?}: the gate band must resolve a 1e-9 gate weight move"
            );
        }
    }

    /// Typed refusals beside the accepted binding: the source refuses an undeclared parameter, a
    /// vector read as a matrix, a matrix read as a vector, and anchor controls. A block with a
    /// per-head query/key norm declares the two gains after every other parameter, reads them as
    /// vectors and refuses them as matrices.
    #[test]
    fn a_gated_block_program_refuses_what_it_cannot_bind() {
        let (block, residual, positions) = gated_block(ResidualLayout::Sequential, 3003, 0.0, false);
        let program = GatedBlockProgram::new(&block).expect("the block program is valid");
        assert!(
            program
                .program()
                .execute(
                    &program.source(&block),
                    &MaskAssignment::all_on(),
                    vec![residual.clone()],
                    vec![None; GatedBlockStage::ALL.len()],
                    &positions,
                )
                .is_ok(),
            "control: the block binds its own program"
        );
        let (normed, _, _) = gated_block(ResidualLayout::Sequential, 3003, 0.0, true);
        let normed_program = GatedBlockProgram::new(&normed).expect("a normed block has a program");
        let declared = normed_program.parameters();
        assert_eq!(
            &declared[..program.parameters().len()],
            program.parameters(),
            "the norm gains come after every parameter the norm-free block declares"
        );
        assert_eq!(
            &declared[program.parameters().len()..],
            [GatedBlockParameter::QueryNormGain, GatedBlockParameter::KeyNormGain],
            "a normed block declares its query and key norm gains"
        );
        let normed_source = normed_program.source(&normed);
        let normed_at = |parameter: GatedBlockParameter| ParameterUse {
            parameter: ParameterSlot(declared.iter().position(|d| *d == parameter).expect("declared") as u32),
            body: BodyId(0),
            node: NodeId(0),
            invocation: &[],
            controls: &[],
        };
        let query_key_norm = normed.attention().query_key_norm().expect("the normed block's norm");
        for (parameter, gain) in [
            (GatedBlockParameter::QueryNormGain, query_key_norm.query_gain()),
            (GatedBlockParameter::KeyNormGain, query_key_norm.key_gain()),
        ] {
            assert_eq!(
                normed_source.vector(normed_at(parameter)).expect("a gain reads as a vector"),
                gain,
                "{parameter} reads the native norm's own gain"
            );
            assert!(
                matches!(
                    normed_source.apply_linear(normed_at(parameter), TieOrientation::Identity, residual.view()),
                    Err(GatedBlockProgramError::NotAMatrix { parameter: refused }) if refused == parameter
                ),
                "{parameter} read as a matrix must be refused"
            );
        }

        let source = program.source(&block);
        let slot_of = |parameter: GatedBlockParameter| {
            ParameterSlot(
                program
                    .parameters()
                    .iter()
                    .position(|declared| *declared == parameter)
                    .expect("declared parameter") as u32,
            )
        };
        let rows = Array2::<f64>::ones((1, WIDTH));
        let at = |parameter: ParameterSlot, controls: &'static [f64]| ParameterUse {
            parameter,
            body: BodyId(0),
            node: NodeId(0),
            invocation: &[],
            controls,
        };
        let gate = slot_of(GatedBlockParameter::Gate);
        assert!(
            matches!(
                source.apply_linear(at(ParameterSlot(99), &[]), TieOrientation::Identity, rows.view()),
                Err(GatedBlockProgramError::Unbound {
                    parameter: ParameterSlot(99)
                })
            ),
            "an undeclared parameter must be refused"
        );
        assert!(
            matches!(
                source.apply_linear(
                    at(slot_of(GatedBlockParameter::AttentionNormGain), &[]),
                    TieOrientation::Identity,
                    rows.view()
                ),
                Err(GatedBlockProgramError::NotAMatrix {
                    parameter: GatedBlockParameter::AttentionNormGain
                })
            ),
            "a gain read as a matrix must be refused"
        );
        assert!(
            matches!(
                source.vector(at(gate, &[])),
                Err(GatedBlockProgramError::NotAVector {
                    parameter: GatedBlockParameter::Gate
                })
            ),
            "a weight read as a vector must be refused"
        );
        assert!(
            matches!(
                source.apply_linear(at(gate, &[0.5]), TieOrientation::Identity, rows.view()),
                Err(GatedBlockProgramError::Controlled {
                    parameter: GatedBlockParameter::Gate
                })
            ),
            "anchor controls on a native tensor must be refused"
        );
        assert!(
            source.apply_linear(at(gate, &[]), TieOrientation::Identity, rows.view()).is_ok(),
            "control: the gate weight reads the rows"
        );
    }
}

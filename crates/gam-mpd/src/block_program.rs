//! Mechanism programs over [`super::block`]'s attention-only layers (#2951).
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
//! each projection's anchor controls to its component coordinates: every setting,
//! all ones included, reads `U diag(m) R` through apply.rs without forming it. The
//! stored tensors on their original paths are [`attention_layer_program`]'s, never an
//! all-ones mask's. [`ComponentLayerProgram`] declares one control per
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
//! # Validity domain
//!
//! A program reads each projection globally. A read at declared positions
//! ([`super::block::ProjectionRead::Edited`]) has no program node, since program masks are scoped
//! by invocation, not by sequence position.

use super::apply::{ApplyError, FactorView, apply_anchored_linear, native_linear};
use super::block::{
    AttentionProjection, BlockError, ComponentAttentionLayer, NativeAttentionLayer,
};
use super::lift::TieOrientation;
use super::program::{
    Body, BodyId, ControlDecl, ControlId, MaskAssignment, MaskGroup, MaskGroupId, NativePrimitive, Node,
    NodeId, ParameterDecl, ParameterSlot, ParameterSource, ParameterUse, Program, ProgramError, ProgramParts,
    SlotDecl, SlotId, SumTerm,
};
use gam_runtime::resource::{Governed, MemoryGovernor};
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
        governor: &MemoryGovernor,
        parameter: ParameterUse<'_>,
        orientation: TieOrientation,
        rows: ArrayView2<'_, f64>,
    ) -> Result<Governed<Array2<f64>>, LayerProgramError> {
        let projection = projection_of(parameter.parameter)?;
        if !parameter.controls.is_empty() {
            return Err(LayerProgramError::DenseControls { projection });
        }
        native_linear(governor, oriented(self.weight(projection), orientation), rows)
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
        governor: &MemoryGovernor,
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
        let stored = self.components(projection)?;
        let factor = match orientation {
            TieOrientation::Identity => stored,
            TieOrientation::Transpose => FactorView::new(stored.right(), stored.left()).map_err(refused)?,
        };
        apply_anchored_linear(governor, weight, 0.0, factor, ArrayView1::from(parameter.controls), rows).map_err(refused)
    }

    fn vector(&self, parameter: ParameterUse<'_>) -> Result<Array1<f64>, LayerProgramError> {
        Err(LayerProgramError::NotAVector {
            parameter: parameter.parameter,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::test_governor;
    use crate::attention::{AttentionGeometry, ProjectedRows, RotaryEmbedding, RotaryPairing};
    use crate::block::{
        AttentionLayerExecution, AttentionLayerReads, ComponentMasks, ProjectionRead,
    };
    use crate::program::{Execution, ExecutionError};
    use crate::rewrite::ComponentRead;
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
                test_governor(), source,
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

    fn slot(execution: &Execution, stage: LayerStage) -> &Array2<f64> {
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
                    LayerStage::Queries => &layer.queries,
                    LayerStage::Keys => &layer.keys,
                    LayerStage::Values => &layer.values,
                    LayerStage::Mixed => &layer.attention.mixed,
                    LayerStage::Write => &layer.write,
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
            .execute(test_governor(), AttentionLayerReads::native(), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
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
            let expected = layer
                .execute(test_governor(), component_reads(&masks), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
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

    /// With no mask assigned every control is 1, and the component program still reads the
    /// factors: it replays the layer's factored all-ones reads bit for bit, not the stored
    /// tensors, which only the native program reads. Positive control: the native layer, the
    /// same algebra, gives other bits.
    #[test]
    fn an_all_on_component_program_reads_the_factors() {
        let fixture = Fixture::new(2994);
        let layer = fixture.component();
        let program = ComponentLayerProgram::new(&layer).expect("the component program is valid");
        let ones = all_ones();
        let factored = layer
            .execute(test_governor(), component_reads(&ones), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
            .expect("factored all-ones reads");
        let executed = fixture
            .run(program.program(), &layer, &MaskAssignment::all_on())
            .expect("all-on component program");
        assert!(
            bits(&executed.output) == bits(&factored.output),
            "the all-on component program must replay the factored all-ones reads"
        );
        for (stage, agrees) in stage_bits_agree(&executed, &factored) {
            assert!(agrees, "the all-on {stage:?} slot must equal the factored reads' stage bit for bit");
        }
        let native = layer
            .execute(test_governor(), AttentionLayerReads::native(), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
            .expect("native reads");
        assert!(
            bits(&factored.output) != bits(&native.output),
            "control: the bit comparison must distinguish the native reads"
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
}

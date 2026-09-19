//! The component-coordinate MLP block of [`super::rewrite`] as a mechanism
//! [`Program`] (#2951 P5).
//!
//! [`ComponentMlp::program`] states the block as a [`Program`]. Each weight is a
//! [`Node::Refine`] whose native body is `Linear W` and whose mechanism body is
//! `Linear R → CoordinateMask → Linear U`, followed by the native bias, the
//! activation and the residual sum. One mask group per component carries its
//! mask ([`ComponentMlp::mask_assignment`]). A refinement runs its native tensor
//! when every component of that weight resolves to exactly `1`, which is
//! [`ComponentMask::AllOn`] here, and its factors otherwise.
//!
//! The block binds the program's eight formal parameters itself
//! ([`MlpParameter`], [`ParameterSource`]), borrowing its own tensors. Each
//! product is formed by the block's own kernel, the `rows · Aᵀ` its methods use,
//! and not by `apply::native_linear`, whose faer tiles round differently. The
//! program then repeats the block's rounded operations one for one: a coordinate
//! mask multiplies a column only when its value is not `1`, which changes no bit,
//! and the residual sum adds the write to `h`, which floating-point addition
//! commutes bitwise. So the program executes [`ComponentMlp::execute`] bit for
//! bit, with each factor whose mask is all ones executed as all on. That covers
//! ReLU and exact GELU. A SiLU block's program runs the program's SiLU owner,
//! while [`super::rewrite::NativeMlp::activate`] refuses SiLU, whose Gaussian
//! smoothing has no closed form.
//!
//! This module sits downstream of both [`super::rewrite`] and [`super::program`],
//! so the rewrite itself imports no program type.

use super::lift::TieOrientation;
use super::program::{
    Body, BodyId, ControlDecl, ControlId, MaskAssignment, MaskGroup, MaskGroupId, NativeActivation,
    NativePrimitive, Node, NodeId, ParameterDecl, ParameterSlot, ParameterSource, ParameterUse,
    Program, ProgramError, ProgramParts, SumTerm,
};
use super::rewrite::{ComponentMask, ComponentMlp, MlpFactor, MlpMask, ShapeMismatch, check};
use gam_runtime::resource::{Governed, MemoryGovernor, MemoryReservationError};
use ndarray::{Array1, Array2, ArrayView2};
use std::fmt;

/// The formal parameters of a [`ComponentMlp`]'s mechanism program, in slot
/// order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlpParameter {
    /// `W₁`.
    ReadInWeight,
    /// `b₁`.
    ReadInBias,
    /// The read-in factor's read `R`.
    ReadInRead,
    /// The read-in factor's write `U`.
    ReadInWrite,
    /// `W₂`.
    WriteOutWeight,
    /// `b₂`.
    WriteOutBias,
    /// The write-out factor's read `R̄`.
    WriteOutRead,
    /// The write-out factor's write `Ū`.
    WriteOutWrite,
}

impl MlpParameter {
    /// Every parameter, in slot order.
    pub const ALL: [Self; 8] = [
        Self::ReadInWeight,
        Self::ReadInBias,
        Self::ReadInRead,
        Self::ReadInWrite,
        Self::WriteOutWeight,
        Self::WriteOutBias,
        Self::WriteOutRead,
        Self::WriteOutWrite,
    ];

    /// The program's parameter slot of this tensor.
    pub fn slot(self) -> ParameterSlot {
        ParameterSlot(self as u32)
    }

    /// The parameter at `slot`, if the block's program declares one there.
    pub fn from_slot(slot: ParameterSlot) -> Option<Self> {
        Self::ALL.get(slot.index()).copied()
    }

    /// The parameter's name in the program's declarations.
    pub fn name(self) -> &'static str {
        match self {
            Self::ReadInWeight => "read-in weight",
            Self::ReadInBias => "read-in bias",
            Self::ReadInRead => "read-in component read",
            Self::ReadInWrite => "read-in component write",
            Self::WriteOutWeight => "write-out weight",
            Self::WriteOutBias => "write-out bias",
            Self::WriteOutRead => "write-out component read",
            Self::WriteOutWrite => "write-out component write",
        }
    }
}

/// A refused mask assignment of a [`ComponentMlp`]'s mechanism program.
#[derive(Debug)]
pub enum MlpMaskError {
    /// A factor's mask length is not its component count.
    Shape(ShapeMismatch),
    /// The assignment refused a mask value, e.g. a non-finite one.
    Program(ProgramError),
}

impl From<ShapeMismatch> for MlpMaskError {
    fn from(mismatch: ShapeMismatch) -> Self {
        Self::Shape(mismatch)
    }
}

impl From<ProgramError> for MlpMaskError {
    fn from(error: ProgramError) -> Self {
        Self::Program(error)
    }
}

impl fmt::Display for MlpMaskError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(mismatch) => {
                write!(formatter, "an MLP mask does not fit its factor: {mismatch}")
            }
            Self::Program(error) => {
                write!(formatter, "the MLP block's program refused a mask value: {error}")
            }
        }
    }
}

impl std::error::Error for MlpMaskError {}

/// A formal parameter at one use that a [`ComponentMlp`] cannot bind.
#[derive(Debug)]
pub enum MlpBindingError {
    /// A slot the block's program does not declare.
    UnknownParameter { parameter: ParameterSlot },
    /// A bias read as a matrix.
    NotAMatrix { parameter: MlpParameter },
    /// A matrix read as a vector.
    NotAVector { parameter: MlpParameter },
    /// A use with anchor controls. The block's tensors are fixed factors with no
    /// component anchor for controls to act on; its masks act through the
    /// program's coordinate masks.
    Controlled { parameter: ParameterSlot },
    /// Rows whose width is not the matrix's input width in the node's
    /// orientation.
    Shape(ShapeMismatch),
    /// The product's footprint does not fit the process memory budget.
    Memory(MemoryReservationError),
}

impl From<ShapeMismatch> for MlpBindingError {
    fn from(mismatch: ShapeMismatch) -> Self {
        Self::Shape(mismatch)
    }
}

impl fmt::Display for MlpBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownParameter { parameter } => write!(
                formatter,
                "formal parameter {} is not one of the MLP block's eight",
                parameter.0
            ),
            Self::NotAMatrix { parameter } => write!(
                formatter,
                "the MLP block's {} is a vector, not a matrix",
                parameter.name()
            ),
            Self::NotAVector { parameter } => write!(
                formatter,
                "the MLP block's {} is a matrix, not a vector",
                parameter.name()
            ),
            Self::Controlled { parameter } => write!(
                formatter,
                "formal parameter {} declares anchor controls, which the MLP block's fixed factors cannot apply",
                parameter.0
            ),
            Self::Shape(mismatch) => {
                write!(formatter, "an MLP block product's shapes do not compose: {mismatch}")
            }
            Self::Memory(error) => write!(formatter, "an MLP block product: {error}"),
        }
    }
}

impl std::error::Error for MlpBindingError {}

impl ComponentMlp {
    /// The block as a mechanism [`Program`] (see the module documentation). The
    /// read-in components are controls `0..C` and the write-out components
    /// `C..C + C̄`, each alone in the mask group of its own index.
    pub fn program(&self) -> Result<Program, ProgramError> {
        let read_in_components = self.read_in().components();
        let write_out_components = self.write_out().components();
        let names: Vec<String> = (0..read_in_components)
            .map(|c| format!("{} component {c}", MlpFactor::ReadIn))
            .chain((0..write_out_components).map(|c| format!("{} component {c}", MlpFactor::WriteOut)))
            .collect();
        let controls = names
            .iter()
            .map(|name| ControlDecl { name: name.clone() })
            .collect();
        let mask_groups = names
            .into_iter()
            .enumerate()
            .map(|(c, name)| MaskGroup {
                name,
                controls: vec![ControlId(c as u32)],
            })
            .collect();
        // Every read of the block is `x Θᵀ` on the tensor as stored.
        let linear = |weight: MlpParameter, argument: u32| Node::Native {
            primitive: NativePrimitive::Linear {
                weight: weight.slot(),
                orientation: TieOrientation::Identity,
            },
            arguments: vec![NodeId(argument)],
        };
        let native = |weight: MlpParameter| Body {
            name: weight.name().to_string(),
            inputs: 1,
            nodes: vec![Node::Input { port: 0 }, linear(weight, 0)],
            output: NodeId(1),
        };
        let mechanism = |read: MlpParameter, write: MlpParameter, first: usize, count: usize| Body {
            name: format!("{} under masks", write.name()),
            inputs: 1,
            nodes: vec![
                Node::Input { port: 0 },
                linear(read, 0),
                Node::Native {
                    primitive: NativePrimitive::CoordinateMask {
                        controls: (first..first + count).map(|c| ControlId(c as u32)).collect(),
                    },
                    arguments: vec![NodeId(1)],
                },
                linear(write, 2),
            ],
            output: NodeId(3),
        };
        let entry = Body {
            name: "residual MLP block".to_string(),
            inputs: 1,
            nodes: vec![
                Node::Input { port: 0 },
                Node::Refine {
                    native: BodyId(1),
                    mechanism: BodyId(2),
                    arguments: vec![NodeId(0)],
                },
                Node::Native {
                    primitive: NativePrimitive::AddBias {
                        bias: MlpParameter::ReadInBias.slot(),
                    },
                    arguments: vec![NodeId(1)],
                },
                Node::Native {
                    primitive: NativePrimitive::Activation {
                        activation: NativeActivation::from_owner(self.native().activation()),
                    },
                    arguments: vec![NodeId(2)],
                },
                Node::Refine {
                    native: BodyId(3),
                    mechanism: BodyId(4),
                    arguments: vec![NodeId(3)],
                },
                Node::Native {
                    primitive: NativePrimitive::AddBias {
                        bias: MlpParameter::WriteOutBias.slot(),
                    },
                    arguments: vec![NodeId(4)],
                },
                Node::Sum {
                    terms: vec![
                        SumTerm {
                            value: NodeId(0),
                            control: None,
                        },
                        SumTerm {
                            value: NodeId(5),
                            control: None,
                        },
                    ],
                },
            ],
            output: NodeId(6),
        };
        let bodies = vec![
            entry,
            native(MlpParameter::ReadInWeight),
            mechanism(
                MlpParameter::ReadInRead,
                MlpParameter::ReadInWrite,
                0,
                read_in_components,
            ),
            native(MlpParameter::WriteOutWeight),
            mechanism(
                MlpParameter::WriteOutRead,
                MlpParameter::WriteOutWrite,
                read_in_components,
                write_out_components,
            ),
        ];
        let parameters = MlpParameter::ALL
            .iter()
            .map(|parameter| ParameterDecl {
                name: parameter.name().to_string(),
                controls: Vec::new(),
            })
            .collect();
        Program::new(ProgramParts {
            parameters,
            slots: Vec::new(),
            controls,
            mask_groups,
            bodies,
            entry: BodyId(0),
        })
    }

    /// The assignment of [`Self::program`] for `mask`: each component's value on
    /// its own group at global scope. A factor left all on assigns nothing, so its
    /// refinement runs the native tensor.
    pub fn mask_assignment(&self, mask: MlpMask<'_>) -> Result<MaskAssignment, MlpMaskError> {
        let mut assignment = MaskAssignment::all_on();
        for (first, factor, factor_mask) in [
            (0, self.read_in(), mask.read_in),
            (self.read_in().components(), self.write_out(), mask.write_out),
        ] {
            if let ComponentMask::Components(values) = factor_mask {
                check("component mask length", factor.components(), values.len())?;
                for (c, &value) in values.iter().enumerate() {
                    assignment.set(MaskGroupId((first + c) as u32), Vec::new(), value)?;
                }
            }
        }
        Ok(assignment)
    }
}

/// The block's parameter at a use, refusing a use with anchor controls and a slot
/// the block's program does not declare.
fn bound_parameter(parameter_use: ParameterUse<'_>) -> Result<MlpParameter, MlpBindingError> {
    let parameter = parameter_use.parameter;
    if !parameter_use.controls.is_empty() {
        return Err(MlpBindingError::Controlled { parameter });
    }
    MlpParameter::from_slot(parameter).ok_or(MlpBindingError::UnknownParameter { parameter })
}

impl ParameterSource for ComponentMlp {
    type Error = MlpBindingError;

    /// `x Aᵀ` on the block's own tensor, through the block's own kernel (see the
    /// module documentation), with the product's footprint reserved before it is
    /// formed.
    fn apply_linear(
        &self,
        parameter_use: ParameterUse<'_>,
        orientation: TieOrientation,
        rows: ArrayView2<'_, f64>,
    ) -> Result<Governed<Array2<f64>>, Self::Error> {
        let stored = match bound_parameter(parameter_use)? {
            MlpParameter::ReadInWeight => self.native().read_in(),
            MlpParameter::ReadInRead => self.read_in().read(),
            MlpParameter::ReadInWrite => self.read_in().write(),
            MlpParameter::WriteOutWeight => self.native().write_out(),
            MlpParameter::WriteOutRead => self.write_out().read(),
            MlpParameter::WriteOutWrite => self.write_out().write(),
            parameter @ (MlpParameter::ReadInBias | MlpParameter::WriteOutBias) => {
                return Err(MlpBindingError::NotAMatrix { parameter });
            }
        };
        let applied = match orientation {
            TieOrientation::Identity => stored,
            TieOrientation::Transpose => stored.reversed_axes(),
        };
        check("linear input width", applied.ncols(), rows.ncols())?;
        let reservation = MemoryGovernor::global()
            .try_reserve_dense_f64(rows.nrows(), applied.nrows(), "component MLP linear map")
            .map_err(MlpBindingError::Memory)?;
        Ok(reservation.bind(rows.dot(&applied.t())))
    }

    fn vector(&self, parameter_use: ParameterUse<'_>) -> Result<Array1<f64>, Self::Error> {
        match bound_parameter(parameter_use)? {
            MlpParameter::ReadInBias => Ok(self.native().bias_in().to_owned()),
            MlpParameter::WriteOutBias => Ok(self.native().bias_out().to_owned()),
            parameter @ (MlpParameter::ReadInWeight
            | MlpParameter::ReadInRead
            | MlpParameter::ReadInWrite
            | MlpParameter::WriteOutWeight
            | MlpParameter::WriteOutRead
            | MlpParameter::WriteOutWrite) => Err(MlpBindingError::NotAVector { parameter }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter_decomposition::test_support::component_mlp::{
        HIDDEN, READ_IN_COMPONENTS, ROWS, WIDTH, WRITE_OUT_COMPONENTS, bitwise_equal, mask_family,
        random_block, uniform,
    };
    use gam_math::gaussian_activation::GaussianActivation;
    use ndarray::ArrayView1;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// The refinement's own selection: a factor whose every component is exactly
    /// `1` runs its native tensor.
    fn all_on_or_components(mask: ArrayView1<'_, f64>) -> ComponentMask<'_> {
        if mask.iter().all(|&value| value == 1.0) {
            ComponentMask::AllOn
        } else {
            ComponentMask::Components(mask)
        }
    }

    fn positions() -> Vec<i64> {
        (0..ROWS as i64).collect()
    }

    /// The block's program on `inputs` under the assignment of `mask`.
    fn execute_program(
        block: &ComponentMlp,
        program: &Program,
        inputs: &Array2<f64>,
        mask: MlpMask<'_>,
    ) -> Governed<Array2<f64>> {
        let assignment = block.mask_assignment(mask).expect("mask assignment");
        program
            .execute(block, &assignment, vec![inputs.clone()], Vec::new(), &positions())
            .expect("program execution")
            .output
    }

    /// Every pair of read-in and write-out mask kinds, including one factor all on
    /// and the other masked.
    #[test]
    fn the_mechanism_program_executes_the_component_block_bit_for_bit() {
        for (activation, seed) in [
            (GaussianActivation::Relu, 2961),
            (GaussianActivation::ExactGelu, 2962),
        ] {
            let (block, inputs) = random_block(activation, seed);
            let program = block.program().expect("the block's program validates");
            let mut rng = StdRng::seed_from_u64(seed);
            let read_in_masks = mask_family(&mut rng, READ_IN_COMPONENTS);
            let write_out_masks = mask_family(&mut rng, WRITE_OUT_COMPONENTS);
            for (kind, read_in_mask) in &read_in_masks {
                for (write_out_kind, write_out_mask) in &write_out_masks {
                    let executed = execute_program(
                        &block,
                        &program,
                        &inputs,
                        MlpMask {
                            read_in: ComponentMask::Components(read_in_mask.view()),
                            write_out: ComponentMask::Components(write_out_mask.view()),
                        },
                    );
                    let reference = block
                        .execute(
                            inputs.view(),
                            MlpMask {
                                read_in: all_on_or_components(read_in_mask.view()),
                                write_out: all_on_or_components(write_out_mask.view()),
                            },
                        )
                        .expect("component block");
                    assert!(
                        bitwise_equal(&executed, &reference),
                        "{activation:?} block, {kind} read-in and {write_out_kind} write-out masks: the program must execute the component block bit for bit"
                    );
                }
            }

            // Positive controls: the comparison sees the masks, and it sees the
            // refinements route all-ones factors to their native tensors.
            let continuous = &read_in_masks[0].1;
            let read_in_ones = &read_in_masks[3].1;
            let write_out_ones = &write_out_masks[3].1;
            let masked = execute_program(
                &block,
                &program,
                &inputs,
                MlpMask {
                    read_in: ComponentMask::Components(continuous.view()),
                    write_out: ComponentMask::Components(write_out_ones.view()),
                },
            );
            let native = block.native().execute(inputs.view()).expect("native block");
            assert!(
                !bitwise_equal(&masked, &native),
                "{activation:?} block: the comparison must distinguish a masked program from the native block"
            );
            let all_ones = MlpMask {
                read_in: ComponentMask::Components(read_in_ones.view()),
                write_out: ComponentMask::Components(write_out_ones.view()),
            };
            let routed = execute_program(&block, &program, &inputs, all_ones);
            let factored = block
                .execute(inputs.view(), all_ones)
                .expect("all-ones factored block");
            assert!(
                !bitwise_equal(&routed, &factored),
                "{activation:?} block: the all-ones program must run the native tensors, not the factored all-ones path"
            );
        }
    }

    #[test]
    fn the_all_on_program_runs_the_native_tensors_and_a_masked_one_runs_the_factors() {
        let (block, inputs) = random_block(GaussianActivation::ExactGelu, 2963);
        let program = block.program().expect("the block's program validates");
        let native = block.native().execute(inputs.view()).expect("native block");
        let (all_on, all_on_residuals) = program
            .refinement_residuals(
                &block,
                &MaskAssignment::all_on(),
                vec![inputs.clone()],
                Vec::new(),
                &positions(),
            )
            .expect("all-on program");
        assert!(
            bitwise_equal(&all_on.output, &native),
            "the all-on program must execute the original tensors"
        );
        assert!(
            all_on_residuals.len() == 2 && all_on_residuals.iter().all(|residual| residual.all_on),
            "both refinements of the all-on program must carry their native bodies, got {all_on_residuals:?}"
        );

        let mut rng = StdRng::seed_from_u64(11);
        let read_in_mask =
            Array1::from_shape_simple_fn(READ_IN_COMPONENTS, || rng.random_range(0.0..1.0));
        let write_out_mask =
            Array1::from_shape_simple_fn(WRITE_OUT_COMPONENTS, || rng.random_range(0.0..1.0));
        let partial = block
            .mask_assignment(MlpMask {
                read_in: ComponentMask::Components(read_in_mask.view()),
                write_out: ComponentMask::Components(write_out_mask.view()),
            })
            .expect("partial mask assignment");
        let (masked, masked_residuals) = program
            .refinement_residuals(&block, &partial, vec![inputs.clone()], Vec::new(), &positions())
            .expect("partial program");
        assert!(
            masked_residuals.len() == 2 && masked_residuals.iter().all(|residual| !residual.all_on),
            "under partial masks both refinements must carry their mechanism bodies, got {masked_residuals:?}"
        );

        // Positive control: the partial program is not the native block, so the
        // bit-identity check above sees the masks.
        assert!(
            !bitwise_equal(&masked.output, &native),
            "the comparison must distinguish a masked program from the native block"
        );
    }

    #[test]
    fn the_binding_applies_each_tensor_in_its_orientation_and_refuses_what_the_block_does_not_hold() {
        let (block, inputs) = random_block(GaussianActivation::ExactGelu, 2964);
        let mut rng = StdRng::seed_from_u64(12);
        let hidden = uniform(&mut rng, ROWS, HIDDEN, 1.0);
        let at = |parameter: MlpParameter| ParameterUse {
            parameter: parameter.slot(),
            body: BodyId(0),
            node: NodeId(0),
            invocation: &[],
            controls: &[],
        };
        for parameter in MlpParameter::ALL {
            assert_eq!(
                MlpParameter::from_slot(parameter.slot()),
                Some(parameter),
                "{parameter:?} must round-trip through its slot"
            );
        }

        let identity = block
            .apply_linear(at(MlpParameter::ReadInWeight), TieOrientation::Identity, inputs.view())
            .expect("an identity use of W₁");
        assert!(
            bitwise_equal(&identity, &inputs.dot(&block.native().read_in().t())),
            "an identity use must apply x W₁ᵀ"
        );
        let transposed = block
            .apply_linear(at(MlpParameter::ReadInWeight), TieOrientation::Transpose, hidden.view())
            .expect("a transposed use of W₁");
        assert!(
            bitwise_equal(&transposed, &hidden.dot(&block.native().read_in())),
            "a transposed use must apply x W₁"
        );
        let crossed = block.apply_linear(
            at(MlpParameter::ReadInWeight),
            TieOrientation::Identity,
            hidden.view(),
        );
        assert!(
            matches!(
                crossed,
                Err(MlpBindingError::Shape(ShapeMismatch { expected, found, .. }))
                    if expected == WIDTH && found == HIDDEN
            ),
            "hidden rows must be refused by an identity use of W₁, got {crossed:?}"
        );
        let bias_as_matrix = block.apply_linear(
            at(MlpParameter::ReadInBias),
            TieOrientation::Identity,
            inputs.view(),
        );
        assert!(
            matches!(
                bias_as_matrix,
                Err(MlpBindingError::NotAMatrix {
                    parameter: MlpParameter::ReadInBias
                })
            ),
            "a bias must be refused as a matrix, got {bias_as_matrix:?}"
        );
        let matrix_as_bias = block.vector(at(MlpParameter::WriteOutWrite));
        assert!(
            matches!(
                matrix_as_bias,
                Err(MlpBindingError::NotAVector {
                    parameter: MlpParameter::WriteOutWrite
                })
            ),
            "a write must be refused as a vector, got {matrix_as_bias:?}"
        );
        let unknown = block.vector(ParameterUse {
            parameter: ParameterSlot(MlpParameter::ALL.len() as u32),
            ..at(MlpParameter::ReadInBias)
        });
        assert!(
            matches!(unknown, Err(MlpBindingError::UnknownParameter { .. })),
            "a slot past the eight must be refused, got {unknown:?}"
        );
        let controlled = block.vector(ParameterUse {
            controls: &[0.5],
            ..at(MlpParameter::ReadInBias)
        });
        assert!(
            matches!(controlled, Err(MlpBindingError::Controlled { .. })),
            "a use with anchor controls must be refused, got {controlled:?}"
        );

        // Positive control: the same bias with no controls binds.
        let bias = block
            .vector(at(MlpParameter::ReadInBias))
            .expect("the read-in bias");
        assert_eq!(bias, block.native().bias_in(), "the read-in bias must bind to b₁");
    }
}

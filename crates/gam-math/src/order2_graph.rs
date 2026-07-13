//! Compiled value/gradient/Hessian lowering for runtime row expressions.
//!
//! [`Order2Graph`] implements the same [`RuntimeJetScalar`] algebra as the eager
//! packed jets, but records a small scalar DAG instead of propagating a dense
//! Hessian through every intermediate. Each node carries its value and primary
//! gradient. Once the scalar output is known, one reverse sweep computes node
//! adjoints and accumulates only the local curvature of nonlinear nodes:
//!
//! `H = sum_node adjoint[node] * J_parent' * local_H[node] * J_parent`.
//!
//! This is the universal second-order chain rule for a scalar DAG. Families own
//! only their one generic row expression and certified unary derivative stacks;
//! this module owns the compiled lowering schedule.

use std::cell::RefCell;

use crate::jet_scalar::{Order2, RuntimeJetScalar, SymmetricQuadraticCoefficients};

#[derive(Clone, Copy, Debug)]
struct GraphTerm {
    node: usize,
    first: f64,
    second: f64,
}

#[derive(Clone, Copy, Debug)]
enum GraphNodeKind {
    Constant,
    Variable,
    Add {
        left: usize,
        right: usize,
    },
    Sub {
        left: usize,
        right: usize,
    },
    Scale {
        input: usize,
        scale: f64,
    },
    Product {
        left: usize,
        right: usize,
    },
    MultiplyAdd {
        left: usize,
        right: usize,
        addend: usize,
    },
    Unary {
        input: usize,
        first: f64,
        second: f64,
    },
    LinearTerms {
        start: usize,
        len: usize,
    },
    DiagonalTerms {
        start: usize,
        len: usize,
    },
    Quadratic {
        start: usize,
        len: usize,
        coefficient_start: usize,
    },
}

#[derive(Clone, Copy, Debug)]
struct GraphNode {
    value: f64,
    kind: GraphNodeKind,
}

#[derive(Debug, Default)]
struct GraphTape {
    dimension: usize,
    nodes: Vec<GraphNode>,
    gradients: Vec<f64>,
    terms: Vec<GraphTerm>,
    coefficients: Vec<f64>,
}

/// Reusable storage for a compiled scalar DAG.
///
/// Reset between rows. All vectors retain their capacity, so a warmed worker
/// performs no tape or reverse-adjoint allocation.
#[derive(Debug, Default)]
pub struct Order2GraphWorkspace {
    tape: RefCell<GraphTape>,
    adjoints: RefCell<Vec<f64>>,
}

impl Order2GraphWorkspace {
    /// Empty reusable graph storage.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reclaim the prior row while retaining every allocation.
    pub fn reset(&mut self, dimension: usize) {
        let tape = self.tape.get_mut();
        tape.dimension = dimension;
        tape.nodes.clear();
        tape.gradients.clear();
        tape.terms.clear();
        tape.coefficients.clear();
        self.adjoints.get_mut().clear();
    }

    /// Bytes retained by the reusable graph and reverse sweep buffers.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        let tape = self.tape.borrow();
        tape.nodes.capacity() * std::mem::size_of::<GraphNode>()
            + tape.gradients.capacity() * std::mem::size_of::<f64>()
            + tape.terms.capacity() * std::mem::size_of::<GraphTerm>()
            + tape.coefficients.capacity() * std::mem::size_of::<f64>()
            + self.adjoints.borrow().capacity() * std::mem::size_of::<f64>()
    }

    #[inline(always)]
    fn push<const K: usize>(
        tape: &mut GraphTape,
        value: f64,
        kind: GraphNodeKind,
        gradient: [f64; K],
    ) -> usize {
        assert_eq!(tape.dimension, K, "compiled graph dimension mismatch");
        assert_eq!(tape.gradients.len(), tape.nodes.len() * K);
        let node = tape.nodes.len();
        tape.nodes.push(GraphNode { value, kind });
        tape.gradients.extend_from_slice(&gradient);
        node
    }

    fn lower<const K: usize>(&self, output: usize) -> Order2<K> {
        let tape = self.tape.borrow();
        assert_eq!(tape.dimension, K, "compiled graph dimension mismatch");
        assert!(output < tape.nodes.len(), "compiled graph output is absent");
        assert_eq!(tape.gradients.len(), tape.nodes.len() * K);

        let mut adjoints = self.adjoints.borrow_mut();
        adjoints.resize(tape.nodes.len(), 0.0);
        adjoints.fill(0.0);
        adjoints[output] = 1.0;

        let mut out = crate::jet_tower::Tower2::zero();
        out.v = tape.nodes[output].value;
        for primary in 0..K {
            out.g[primary] = tape.gradients[output * K + primary];
        }

        for node_index in (0..=output).rev() {
            let adjoint = adjoints[node_index];
            if adjoint == 0.0 {
                continue;
            }
            match tape.nodes[node_index].kind {
                GraphNodeKind::Constant | GraphNodeKind::Variable => {}
                GraphNodeKind::Add { left, right } => {
                    adjoints[left] += adjoint;
                    adjoints[right] += adjoint;
                }
                GraphNodeKind::Sub { left, right } => {
                    adjoints[left] += adjoint;
                    adjoints[right] -= adjoint;
                }
                GraphNodeKind::Scale { input, scale } => {
                    adjoints[input] += adjoint * scale;
                }
                GraphNodeKind::Product { left, right } => {
                    adjoints[left] += adjoint * tape.nodes[right].value;
                    adjoints[right] += adjoint * tape.nodes[left].value;
                    for primary in 0..K {
                        let left_primary = tape.gradients[left * K + primary];
                        let right_primary = tape.gradients[right * K + primary];
                        for other in primary..K {
                            let curvature = left_primary * tape.gradients[right * K + other]
                                + right_primary * tape.gradients[left * K + other];
                            out.h[primary][other] += adjoint * curvature;
                        }
                    }
                }
                GraphNodeKind::MultiplyAdd {
                    left,
                    right,
                    addend,
                } => {
                    adjoints[left] += adjoint * tape.nodes[right].value;
                    adjoints[right] += adjoint * tape.nodes[left].value;
                    adjoints[addend] += adjoint;
                    for primary in 0..K {
                        let left_primary = tape.gradients[left * K + primary];
                        let right_primary = tape.gradients[right * K + primary];
                        for other in primary..K {
                            let curvature = left_primary * tape.gradients[right * K + other]
                                + right_primary * tape.gradients[left * K + other];
                            out.h[primary][other] += adjoint * curvature;
                        }
                    }
                }
                GraphNodeKind::Unary {
                    input,
                    first,
                    second,
                } => {
                    adjoints[input] += adjoint * first;
                    let curvature_scale = adjoint * second;
                    for primary in 0..K {
                        let input_primary = tape.gradients[input * K + primary];
                        for other in primary..K {
                            out.h[primary][other] +=
                                curvature_scale * input_primary * tape.gradients[input * K + other];
                        }
                    }
                }
                GraphNodeKind::LinearTerms { start, len } => {
                    for term in &tape.terms[start..start + len] {
                        adjoints[term.node] += adjoint * term.first;
                    }
                }
                GraphNodeKind::DiagonalTerms { start, len } => {
                    for term in &tape.terms[start..start + len] {
                        adjoints[term.node] += adjoint * term.first;
                        let curvature_scale = adjoint * term.second;
                        for primary in 0..K {
                            let input_primary = tape.gradients[term.node * K + primary];
                            for other in primary..K {
                                out.h[primary][other] += curvature_scale
                                    * input_primary
                                    * tape.gradients[term.node * K + other];
                            }
                        }
                    }
                }
                GraphNodeKind::Quadratic {
                    start,
                    len,
                    coefficient_start,
                } => {
                    let terms = &tape.terms[start..start + len];
                    for term in terms {
                        adjoints[term.node] += adjoint * term.first;
                    }
                    for primary in 0..K {
                        for other in primary..K {
                            let mut curvature = 0.0;
                            for row in 0..len {
                                let row_gradient = tape.gradients[terms[row].node * K + primary];
                                let mut projected = 0.0;
                                for column in 0..len {
                                    projected += tape.coefficients
                                        [coefficient_start + row * len + column]
                                        * tape.gradients[terms[column].node * K + other];
                                }
                                curvature += row_gradient * projected;
                            }
                            out.h[primary][other] += 2.0 * adjoint * curvature;
                        }
                    }
                }
            }
        }

        for primary in 0..K {
            for other in primary + 1..K {
                out.h[other][primary] = out.h[primary][other];
            }
        }
        Order2(out)
    }
}

/// Const-primary scalar handle into an [`Order2GraphWorkspace`].
///
/// The handle is two machine words. Clone/copy duplicates only the graph node
/// index; all derivative storage remains in the reusable workspace.
#[derive(Clone, Copy, Debug)]
pub struct Order2Graph<'arena, const K: usize> {
    workspace: &'arena Order2GraphWorkspace,
    node: usize,
}

impl<'arena, const K: usize> Order2Graph<'arena, K> {
    /// Lower this scalar output to the ordinary packed order-2 channels.
    #[must_use]
    pub fn into_order2(self) -> Order2<K> {
        self.workspace.lower(self.node)
    }

    #[inline(always)]
    fn assert_compatible(&self, other: &Self) {
        assert!(
            std::ptr::eq(self.workspace, other.workspace),
            "compiled graph scalars belong to different workspaces"
        );
    }

    #[inline(always)]
    fn unary(&self, value: f64, first: f64, second: f64) -> Self {
        let mut tape = self.workspace.tape.borrow_mut();
        let mut gradient = [0.0; K];
        for primary in 0..K {
            gradient[primary] = first * tape.gradients[self.node * K + primary];
        }
        let node = Order2GraphWorkspace::push(
            &mut tape,
            value,
            GraphNodeKind::Unary {
                input: self.node,
                first,
                second,
            },
            gradient,
        );
        Self {
            workspace: self.workspace,
            node,
        }
    }
}

impl<'arena, const K: usize> RuntimeJetScalar<'arena> for Order2Graph<'arena, K> {
    type Workspace = Order2GraphWorkspace;

    #[inline(always)]
    fn constant(c: f64, dimension: usize, workspace: &'arena Self::Workspace) -> Self {
        assert_eq!(dimension, K, "compiled graph dimension mismatch");
        let mut tape = workspace.tape.borrow_mut();
        let node = Order2GraphWorkspace::push(&mut tape, c, GraphNodeKind::Constant, [0.0; K]);
        Self { workspace, node }
    }

    #[inline(always)]
    fn variable(x: f64, axis: usize, dimension: usize, workspace: &'arena Self::Workspace) -> Self {
        assert_eq!(dimension, K, "compiled graph dimension mismatch");
        assert!(axis < K, "compiled graph variable axis out of bounds");
        let mut gradient = [0.0; K];
        gradient[axis] = 1.0;
        let mut tape = workspace.tape.borrow_mut();
        let node = Order2GraphWorkspace::push(&mut tape, x, GraphNodeKind::Variable, gradient);
        Self { workspace, node }
    }

    #[inline(always)]
    fn symmetric_quadratic_form<C: SymmetricQuadraticCoefficients>(
        inputs: &[Self],
        coefficients: &C,
        dimension: usize,
        workspace: &'arena Self::Workspace,
    ) -> Self {
        assert_eq!(dimension, K, "compiled graph dimension mismatch");
        assert_eq!(inputs.len(), coefficients.dimension());
        assert!(inputs.len() <= K);
        assert!(
            inputs
                .iter()
                .all(|input| std::ptr::eq(input.workspace, workspace)),
            "compiled graph quadratic inputs belong to different workspaces"
        );

        let input_dimension = inputs.len();
        let mut values = [0.0; K];
        for axis in 0..input_dimension {
            values[axis] = inputs[axis].value();
        }
        let mut projected = [0.0; K];
        coefficients.multiply(
            &values[..input_dimension],
            &mut projected[..input_dimension],
        );
        let mut value = 0.0;
        for axis in 0..input_dimension {
            value += values[axis] * projected[axis];
        }

        let mut tape = workspace.tape.borrow_mut();
        let start = tape.terms.len();
        let mut gradient = [0.0; K];
        for axis in 0..input_dimension {
            let first = 2.0 * projected[axis];
            tape.terms.push(GraphTerm {
                node: inputs[axis].node,
                first,
                second: 0.0,
            });
            for primary in 0..K {
                gradient[primary] += first * tape.gradients[inputs[axis].node * K + primary];
            }
        }
        let coefficient_start = tape.coefficients.len();
        for row in 0..input_dimension {
            for column in 0..input_dimension {
                tape.coefficients
                    .push(coefficients.coefficient(row, column));
            }
        }
        let node = Order2GraphWorkspace::push(
            &mut tape,
            value,
            GraphNodeKind::Quadratic {
                start,
                len: input_dimension,
                coefficient_start,
            },
            gradient,
        );
        Self { workspace, node }
    }

    #[inline(always)]
    fn linear_combination(
        inputs: &[Self],
        weights: &[f64],
        dimension: usize,
        workspace: &'arena Self::Workspace,
    ) -> Self {
        assert_eq!(dimension, K, "compiled graph dimension mismatch");
        assert_eq!(inputs.len(), weights.len());
        assert!(
            inputs
                .iter()
                .all(|input| std::ptr::eq(input.workspace, workspace)),
            "compiled graph linear inputs belong to different workspaces"
        );
        let mut tape = workspace.tape.borrow_mut();
        let start = tape.terms.len();
        let mut value = 0.0;
        let mut gradient = [0.0; K];
        for (input, &weight) in inputs.iter().zip(weights) {
            value += weight * tape.nodes[input.node].value;
            tape.terms.push(GraphTerm {
                node: input.node,
                first: weight,
                second: 0.0,
            });
            for primary in 0..K {
                gradient[primary] += weight * tape.gradients[input.node * K + primary];
            }
        }
        let node = Order2GraphWorkspace::push(
            &mut tape,
            value,
            GraphNodeKind::LinearTerms {
                start,
                len: inputs.len(),
            },
            gradient,
        );
        Self { workspace, node }
    }

    #[inline(always)]
    fn add_constant(&self, constant: f64, workspace: &'arena Self::Workspace) -> Self {
        assert!(std::ptr::eq(self.workspace, workspace));
        self.unary(self.value() + constant, 1.0, 0.0)
    }

    #[inline(always)]
    fn multiply_add(&self, right: &Self, addend: &Self) -> Self {
        self.assert_compatible(right);
        self.assert_compatible(addend);
        let mut tape = self.workspace.tape.borrow_mut();
        let left_value = tape.nodes[self.node].value;
        let right_value = tape.nodes[right.node].value;
        let mut gradient = [0.0; K];
        for primary in 0..K {
            gradient[primary] = left_value * tape.gradients[right.node * K + primary]
                + tape.gradients[self.node * K + primary] * right_value
                + tape.gradients[addend.node * K + primary];
        }
        let value = left_value * right_value + tape.nodes[addend.node].value;
        let node = Order2GraphWorkspace::push(
            &mut tape,
            value,
            GraphNodeKind::MultiplyAdd {
                left: self.node,
                right: right.node,
                addend: addend.node,
            },
            gradient,
        );
        Self {
            workspace: self.workspace,
            node,
        }
    }

    #[inline(always)]
    fn composed_sum(
        inputs: &[Self],
        derivative_stacks: &[[f64; 5]],
        dimension: usize,
        workspace: &'arena Self::Workspace,
    ) -> Self {
        assert_eq!(dimension, K, "compiled graph dimension mismatch");
        assert_eq!(inputs.len(), derivative_stacks.len());
        assert!(
            inputs
                .iter()
                .all(|input| std::ptr::eq(input.workspace, workspace)),
            "compiled graph composed inputs belong to different workspaces"
        );
        let mut tape = workspace.tape.borrow_mut();
        let start = tape.terms.len();
        let mut value = 0.0;
        let mut gradient = [0.0; K];
        for (input, stack) in inputs.iter().zip(derivative_stacks) {
            value += stack[0];
            tape.terms.push(GraphTerm {
                node: input.node,
                first: stack[1],
                second: stack[2],
            });
            for primary in 0..K {
                gradient[primary] += stack[1] * tape.gradients[input.node * K + primary];
            }
        }
        let node = Order2GraphWorkspace::push(
            &mut tape,
            value,
            GraphNodeKind::DiagonalTerms {
                start,
                len: inputs.len(),
            },
            gradient,
        );
        Self { workspace, node }
    }

    #[inline(always)]
    fn product(&self, right: &Self) -> Self {
        self.mul(right)
    }

    #[inline(always)]
    fn affine_compose(
        &self,
        input_scale: f64,
        input_shift: f64,
        derivative_stack: [f64; 5],
        workspace: &'arena Self::Workspace,
    ) -> Self {
        assert!(std::ptr::eq(self.workspace, workspace));
        assert!(input_shift.is_finite(), "affine input shift must be finite");
        self.unary(
            derivative_stack[0],
            derivative_stack[1] * input_scale,
            derivative_stack[2] * input_scale * input_scale,
        )
    }

    #[inline(always)]
    fn affine_composed_sum(
        inputs: &[Self],
        input_scales: &[f64],
        derivative_stacks: &[[f64; 5]],
        dimension: usize,
        workspace: &'arena Self::Workspace,
    ) -> Self {
        assert_eq!(dimension, K, "compiled graph dimension mismatch");
        assert_eq!(inputs.len(), input_scales.len());
        assert_eq!(inputs.len(), derivative_stacks.len());
        assert!(
            inputs
                .iter()
                .all(|input| std::ptr::eq(input.workspace, workspace)),
            "compiled graph composed inputs belong to different workspaces"
        );
        let mut tape = workspace.tape.borrow_mut();
        let start = tape.terms.len();
        let mut value = 0.0;
        let mut gradient = [0.0; K];
        for ((input, &input_scale), stack) in inputs.iter().zip(input_scales).zip(derivative_stacks)
        {
            let first = stack[1] * input_scale;
            let second = stack[2] * input_scale * input_scale;
            value += stack[0];
            tape.terms.push(GraphTerm {
                node: input.node,
                first,
                second,
            });
            for primary in 0..K {
                gradient[primary] += first * tape.gradients[input.node * K + primary];
            }
        }
        let node = Order2GraphWorkspace::push(
            &mut tape,
            value,
            GraphNodeKind::DiagonalTerms {
                start,
                len: inputs.len(),
            },
            gradient,
        );
        Self { workspace, node }
    }

    #[inline(always)]
    fn dimension(&self) -> usize {
        K
    }

    #[inline(always)]
    fn value(&self) -> f64 {
        self.workspace.tape.borrow().nodes[self.node].value
    }

    #[inline(always)]
    fn add(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let mut tape = self.workspace.tape.borrow_mut();
        let mut gradient = [0.0; K];
        for primary in 0..K {
            gradient[primary] =
                tape.gradients[self.node * K + primary] + tape.gradients[other.node * K + primary];
        }
        let value = tape.nodes[self.node].value + tape.nodes[other.node].value;
        let node = Order2GraphWorkspace::push(
            &mut tape,
            value,
            GraphNodeKind::Add {
                left: self.node,
                right: other.node,
            },
            gradient,
        );
        Self {
            workspace: self.workspace,
            node,
        }
    }

    #[inline(always)]
    fn sub(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let mut tape = self.workspace.tape.borrow_mut();
        let mut gradient = [0.0; K];
        for primary in 0..K {
            gradient[primary] =
                tape.gradients[self.node * K + primary] - tape.gradients[other.node * K + primary];
        }
        let value = tape.nodes[self.node].value - tape.nodes[other.node].value;
        let node = Order2GraphWorkspace::push(
            &mut tape,
            value,
            GraphNodeKind::Sub {
                left: self.node,
                right: other.node,
            },
            gradient,
        );
        Self {
            workspace: self.workspace,
            node,
        }
    }

    #[inline(always)]
    fn mul(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let mut tape = self.workspace.tape.borrow_mut();
        let left_value = tape.nodes[self.node].value;
        let right_value = tape.nodes[other.node].value;
        let mut gradient = [0.0; K];
        for primary in 0..K {
            gradient[primary] = left_value * tape.gradients[other.node * K + primary]
                + tape.gradients[self.node * K + primary] * right_value;
        }
        let node = Order2GraphWorkspace::push(
            &mut tape,
            left_value * right_value,
            GraphNodeKind::Product {
                left: self.node,
                right: other.node,
            },
            gradient,
        );
        Self {
            workspace: self.workspace,
            node,
        }
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        self.unary(-self.value(), -1.0, 0.0)
    }

    #[inline(always)]
    fn scale(&self, scale: f64) -> Self {
        let mut tape = self.workspace.tape.borrow_mut();
        let mut gradient = [0.0; K];
        for primary in 0..K {
            gradient[primary] = scale * tape.gradients[self.node * K + primary];
        }
        let value = scale * tape.nodes[self.node].value;
        let node = Order2GraphWorkspace::push(
            &mut tape,
            value,
            GraphNodeKind::Scale {
                input: self.node,
                scale,
            },
            gradient,
        );
        Self {
            workspace: self.workspace,
            node,
        }
    }

    #[inline(always)]
    fn compose_unary(&self, derivatives: [f64; 5]) -> Self {
        self.unary(derivatives[0], derivatives[1], derivatives[2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jet_scalar::{FixedRuntimeJet, JetScalar};
    use crate::nested_dual::JetField;

    struct DenseSymmetric3([[f64; 3]; 3]);

    impl SymmetricQuadraticCoefficients for DenseSymmetric3 {
        fn dimension(&self) -> usize {
            3
        }

        fn multiply(&self, input: &[f64], output: &mut [f64]) {
            for row in 0..3 {
                output[row] = (0..3)
                    .map(|column| self.0[row][column] * input[column])
                    .sum();
            }
        }

        fn coefficient(&self, row: usize, column: usize) -> f64 {
            self.0[row][column]
        }
    }

    fn expression<'arena, S: RuntimeJetScalar<'arena>>(
        vars: &[S; 6],
        coefficients: &DenseSymmetric3,
        scales: &[f64; 4],
        stacks: &[[f64; 5]; 4],
        workspace: &'arena S::Workspace,
    ) -> S {
        let nonlinear = [
            vars[0].product(&vars[1]),
            vars[2].affine_compose(scales[0], scales[1], stacks[0], workspace),
            vars[3].multiply_add(&vars[4], &vars[5]),
        ];
        let quadratic = S::symmetric_quadratic_form(&nonlinear, coefficients, 6, workspace);
        let linear = S::linear_combination(vars, &[0.2, -0.7, 1.1, 0.4, -0.3, 0.8], 6, workspace);
        let product = quadratic.product(&linear);
        S::affine_composed_sum(
            &[quadratic, linear, product, nonlinear[1].clone()],
            &[scales[0], scales[1], scales[2], scales[3]],
            stacks,
            6,
            workspace,
        )
    }

    #[test]
    fn compiled_graph_matches_eager_order2_randomized_full_vgh() {
        fn sample(state: &mut u64) -> f64 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            let unit = (*state >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64));
            2.0 * unit - 1.0
        }

        fn close(actual: f64, expected: f64, case: usize, label: &str) {
            let tolerance = 5.0e-12 * actual.abs().max(expected.abs()).max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "case {case} {label}: graph={actual:+.16e}, eager={expected:+.16e}, tolerance={tolerance:.3e}"
            );
        }

        let mut state = 0x932d_a660_5eed_f00d_u64;
        let mut workspace = Order2GraphWorkspace::new();
        for case in 0..256 {
            let values: [f64; 6] = std::array::from_fn(|_| sample(&mut state));
            let scales: [f64; 4] = std::array::from_fn(|_| sample(&mut state));
            let stacks: [[f64; 5]; 4] =
                std::array::from_fn(|_| std::array::from_fn(|_| sample(&mut state)));
            let raw: [[f64; 3]; 3] =
                std::array::from_fn(|_| std::array::from_fn(|_| sample(&mut state)));
            let coefficients = DenseSymmetric3([
                [raw[0][0], raw[0][1], raw[0][2]],
                [raw[0][1], raw[1][1], raw[1][2]],
                [raw[0][2], raw[1][2], raw[2][2]],
            ]);

            let eager_vars: [FixedRuntimeJet<Order2<6>, 6>; 6] = std::array::from_fn(|axis| {
                FixedRuntimeJet::from_inner(Order2::variable(values[axis], axis))
            });
            let eager = expression(&eager_vars, &coefficients, &scales, &stacks, &()).into_inner();

            workspace.reset(6);
            let graph_vars: [Order2Graph<'_, 6>; 6] = std::array::from_fn(|axis| {
                Order2Graph::variable(values[axis], axis, 6, &workspace)
            });
            let graph =
                expression(&graph_vars, &coefficients, &scales, &stacks, &workspace).into_order2();

            close(graph.value(), eager.value(), case, "value");
            for primary in 0..6 {
                close(graph.g()[primary], eager.g()[primary], case, "gradient");
                for other in 0..6 {
                    close(
                        graph.h()[primary][other],
                        eager.h()[primary][other],
                        case,
                        "Hessian",
                    );
                }
            }
        }
    }
}

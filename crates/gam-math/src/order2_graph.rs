//! Compiled value/gradient/Hessian lowering for runtime row expressions.
//!
//! [`Order2Graph`] implements the same [`RuntimeJetScalar`] algebra as the eager
//! packed jets, but records a small scalar DAG instead of propagating a dense
//! Hessian through every intermediate. Each node carries its value, primary
//! gradient, and coefficients over a small basis of local curvature atoms.
//! Scalar operations propagate those coefficients directly; the final output
//! materializes each rank-one, diagonal, cross, or projected-quadratic atom
//! exactly once.
//!
//! This is the universal second-order chain rule for a scalar DAG. Families own
//! only their one generic row expression and certified unary derivative stacks;
//! this module owns the compiled lowering schedule.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

use crate::jet_scalar::{Order2, RuntimeJetScalar, SymmetricQuadraticCoefficients};

#[derive(Clone, Copy, Debug)]
enum CurvatureAtom {
    Empty,
    RankOne {
        input: usize,
    },
    Cross {
        left: usize,
        right: usize,
    },
    Diagonal {
        term_start: usize,
        len: usize,
        support: u128,
    },
    Dense {
        curvature_start: usize,
        dimension: usize,
        support: u128,
    },
}

#[derive(Clone, Copy, Debug)]
struct GraphNode {
    value: f64,
    support: u128,
}

const MAX_PRIMARY_DIMENSION: usize = 16;
const MAX_GRAPH_NODES: usize = 32;
const MAX_CURVATURE_ATOMS: usize = MAX_GRAPH_NODES;
const MAX_DIAGONAL_TERMS: usize = MAX_GRAPH_NODES * MAX_GRAPH_NODES;
const MAX_DENSE_CURVATURE_VALUES: usize =
    MAX_GRAPH_NODES * MAX_PRIMARY_DIMENSION * MAX_PRIMARY_DIMENSION;
const EMPTY_NODE: GraphNode = GraphNode {
    value: 0.0,
    support: 0,
};
const EMPTY_ATOM: CurvatureAtom = CurvatureAtom::Empty;

/// Fixed-capacity scalar scratch with only its logical prefix initialized.
///
/// The graph's active atom count and quadratic arity are usually far below the
/// hard node capacity. Keeping the inactive suffix uninitialized prevents a
/// full 32-scalar memset in every primitive while retaining allocation-free,
/// checked storage for the general graph contract.
struct InlineScalars {
    slots: [MaybeUninit<f64>; MAX_GRAPH_NODES],
    len: usize,
}

impl InlineScalars {
    #[inline(always)]
    fn initialize_zeros(storage: &mut MaybeUninit<Self>, len: usize) -> &mut Self {
        assert!(len <= MAX_GRAPH_NODES, "inline scalar capacity exceeded");
        let scalars = storage.as_mut_ptr();
        // SAFETY: every bit pattern is valid for `MaybeUninit<f64>`, so writing
        // `len` makes the enclosing `InlineScalars` initialized. Only the
        // logical prefix is then exposed as `f64` by the accessors below.
        unsafe {
            std::ptr::addr_of_mut!((*scalars).len).write(len);
            let slots = &mut *std::ptr::addr_of_mut!((*scalars).slots);
            for slot in &mut slots[..len] {
                slot.write(0.0);
            }
            &mut *scalars
        }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn push(&mut self, value: f64) {
        assert!(
            self.len < MAX_GRAPH_NODES,
            "inline scalar capacity exceeded"
        );
        self.slots[self.len].write(value);
        self.len += 1;
    }

    #[inline(always)]
    fn add(&mut self, index: usize, value: f64) {
        debug_assert!(index < self.len);
        // SAFETY: `zeros` initializes the entire logical prefix, `push`
        // initializes one slot before extending it, and callers only address
        // indices below `len`.
        unsafe {
            *self.slots.get_unchecked_mut(index).assume_init_mut() += value;
        }
    }

    #[inline(always)]
    fn as_slice(&self) -> &[f64] {
        // SAFETY: exactly the prefix `0..len` is initialized by `zeros`/`push`;
        // `MaybeUninit<f64>` has the same layout and alignment as `f64`.
        unsafe { std::slice::from_raw_parts(self.slots.as_ptr().cast::<f64>(), self.len) }
    }

    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [f64] {
        // SAFETY: the logical prefix is initialized and uniquely borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.slots.as_mut_ptr().cast::<f64>(), self.len) }
    }
}

#[derive(Debug)]
struct GraphTape {
    dimension: usize,
    node_len: usize,
    atom_len: usize,
    diagonal_len: usize,
    dense_curvature_len: usize,
    nodes: [GraphNode; MAX_GRAPH_NODES],
    gradients: [f64; MAX_GRAPH_NODES * MAX_PRIMARY_DIMENSION],
    hessian_weights: [f64; MAX_GRAPH_NODES * MAX_CURVATURE_ATOMS],
    atoms: [CurvatureAtom; MAX_CURVATURE_ATOMS],
    diagonal_inputs: [usize; MAX_DIAGONAL_TERMS],
    diagonal_coefficients: [f64; MAX_DIAGONAL_TERMS],
    dense_curvatures: [f64; MAX_DENSE_CURVATURE_VALUES],
}

impl GraphTape {
    fn new() -> Self {
        Self {
            dimension: 0,
            node_len: 0,
            atom_len: 0,
            diagonal_len: 0,
            dense_curvature_len: 0,
            nodes: [EMPTY_NODE; MAX_GRAPH_NODES],
            gradients: [0.0; MAX_GRAPH_NODES * MAX_PRIMARY_DIMENSION],
            hessian_weights: [0.0; MAX_GRAPH_NODES * MAX_CURVATURE_ATOMS],
            atoms: [EMPTY_ATOM; MAX_CURVATURE_ATOMS],
            diagonal_inputs: [0; MAX_DIAGONAL_TERMS],
            diagonal_coefficients: [0.0; MAX_DIAGONAL_TERMS],
            dense_curvatures: [0.0; MAX_DENSE_CURVATURE_VALUES],
        }
    }
}

/// Reusable storage for a compiled scalar DAG.
///
/// Reset between rows. The boxed tape has checked fixed capacities, so every
/// row after worker construction performs no tape or curvature-basis allocation
/// and no growth branches.
#[derive(Debug)]
pub struct Order2GraphWorkspace {
    tape: UnsafeCell<Box<GraphTape>>,
}

impl Default for Order2GraphWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

impl Order2GraphWorkspace {
    /// Empty reusable graph storage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tape: UnsafeCell::new(Box::new(GraphTape::new())),
        }
    }

    /// Reclaim the prior row in constant time.
    pub fn reset(&mut self, dimension: usize) {
        assert!(
            dimension <= MAX_PRIMARY_DIMENSION,
            "compiled graph supports at most {MAX_PRIMARY_DIMENSION} primaries"
        );
        let tape = self.tape.get_mut().as_mut();
        tape.dimension = dimension;
        tape.node_len = 0;
        tape.atom_len = 0;
        tape.diagonal_len = 0;
        tape.dense_curvature_len = 0;
    }

    /// Bytes retained by the fixed graph and curvature-basis tape.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<GraphTape>()
    }

    /// Number of active nodes in the current row schedule.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.tape().node_len
    }

    /// Shared tape access. `UnsafeCell` removes the dynamic borrow state that a
    /// `RefCell` would place in every scalar operation. The workspace is not
    /// `Sync`, every mutation is completed before a scalar handle is returned,
    /// and lowering starts only after expression construction, so no aliases to
    /// the tape contents escape these two private accessors.
    #[inline(always)]
    fn tape(&self) -> &GraphTape {
        unsafe { (&*self.tape.get()).as_ref() }
    }

    #[inline(always)]
    fn tape_mut(&self) -> &mut GraphTape {
        unsafe { (&mut *self.tape.get()).as_mut() }
    }

    #[inline(always)]
    fn push<const K: usize>(
        tape: &mut GraphTape,
        value: f64,
        support: u128,
        gradient: [f64; K],
        hessian_weights: &InlineScalars,
    ) -> usize {
        assert_eq!(tape.dimension, K, "compiled graph dimension mismatch");
        assert_eq!(
            hessian_weights.len(),
            tape.atom_len,
            "compiled graph curvature dimension mismatch"
        );
        assert!(
            tape.node_len < MAX_GRAPH_NODES,
            "compiled graph node capacity exceeded"
        );
        let node = tape.node_len;
        tape.nodes[node] = GraphNode { value, support };
        tape.gradients[node * K..(node + 1) * K].copy_from_slice(&gradient);
        let weight_start = node * MAX_CURVATURE_ATOMS;
        tape.hessian_weights[weight_start..weight_start + tape.atom_len]
            .copy_from_slice(hessian_weights.as_slice());
        tape.node_len += 1;
        node
    }

    #[inline(always)]
    fn inherit_hessian(tape: &GraphTape, output: &mut InlineScalars, input: usize, scale: f64) {
        debug_assert_eq!(output.len(), tape.atom_len);
        for atom in 0..tape.atom_len {
            output.add(
                atom,
                scale * tape.hessian_weights[input * MAX_CURVATURE_ATOMS + atom],
            );
        }
    }

    #[inline(always)]
    fn push_atom(tape: &mut GraphTape, atom: CurvatureAtom) -> usize {
        assert!(
            tape.atom_len < MAX_CURVATURE_ATOMS,
            "compiled graph curvature capacity exceeded"
        );
        let index = tape.atom_len;
        for node in 0..tape.node_len {
            tape.hessian_weights[node * MAX_CURVATURE_ATOMS + index] = 0.0;
        }
        tape.atoms[index] = atom;
        tape.atom_len += 1;
        index
    }

    #[inline(always)]
    fn push_weighted_atom(
        tape: &mut GraphTape,
        weights: &mut InlineScalars,
        atom: CurvatureAtom,
        coefficient: f64,
    ) {
        let index = Self::push_atom(tape, atom);
        debug_assert_eq!(index, weights.len());
        weights.push(coefficient);
    }

    #[inline(always)]
    fn push_diagonal_atom(
        tape: &mut GraphTape,
        coefficients_by_node: &InlineScalars,
    ) -> Option<usize> {
        debug_assert_eq!(coefficients_by_node.len(), tape.node_len);
        let term_start = tape.diagonal_len;
        let mut support = 0_u128;
        for (input, &coefficient) in coefficients_by_node.as_slice().iter().enumerate() {
            if coefficient == 0.0 {
                continue;
            }
            assert!(
                tape.diagonal_len < MAX_DIAGONAL_TERMS,
                "compiled graph diagonal-term capacity exceeded"
            );
            tape.diagonal_inputs[tape.diagonal_len] = input;
            tape.diagonal_coefficients[tape.diagonal_len] = coefficient;
            tape.diagonal_len += 1;
            support |= tape.nodes[input].support;
        }
        let len = tape.diagonal_len - term_start;
        (len != 0).then(|| {
            Self::push_atom(
                tape,
                CurvatureAtom::Diagonal {
                    term_start,
                    len,
                    support,
                },
            )
        })
    }

    #[inline(always)]
    fn push_dense_curvature(tape: &mut GraphTape, curvature: &[f64]) -> usize {
        assert!(
            curvature.len() <= MAX_DENSE_CURVATURE_VALUES - tape.dense_curvature_len,
            "compiled graph dense-curvature capacity exceeded"
        );
        let start = tape.dense_curvature_len;
        tape.dense_curvatures[start..start + curvature.len()].copy_from_slice(curvature);
        tape.dense_curvature_len += curvature.len();
        start
    }

    #[inline(always)]
    fn lower<const K: usize>(&self, output: usize) -> Order2<K> {
        let tape = self.tape();
        assert_eq!(tape.dimension, K, "compiled graph dimension mismatch");
        assert!(output < tape.node_len, "compiled graph output is absent");

        let mut out = crate::jet_tower::Tower2::zero();
        out.v = tape.nodes[output].value;
        for primary in 0..K {
            out.g[primary] = tape.gradients[output * K + primary];
        }

        for atom_index in 0..tape.atom_len {
            let weight = tape.hessian_weights[output * MAX_CURVATURE_ATOMS + atom_index];
            if weight == 0.0 {
                continue;
            }
            match tape.atoms[atom_index] {
                CurvatureAtom::Empty => {}
                CurvatureAtom::RankOne { input } => {
                    for_each_supported_upper(tape.nodes[input].support, |primary, other| {
                        out.h[primary][other] += weight
                            * tape.gradients[input * K + primary]
                            * tape.gradients[input * K + other];
                    });
                }
                CurvatureAtom::Cross { left, right } => {
                    for_each_supported_upper(
                        tape.nodes[left].support | tape.nodes[right].support,
                        |primary, other| {
                            let left_primary = tape.gradients[left * K + primary];
                            let right_primary = tape.gradients[right * K + primary];
                            let curvature = left_primary * tape.gradients[right * K + other]
                                + right_primary * tape.gradients[left * K + other];
                            out.h[primary][other] += weight * curvature;
                        },
                    );
                }
                CurvatureAtom::Diagonal {
                    term_start,
                    len,
                    support,
                } => {
                    for_each_supported_upper(support, |primary, other| {
                        let mut curvature = 0.0;
                        for term in term_start..term_start + len {
                            let input = tape.diagonal_inputs[term];
                            curvature += tape.diagonal_coefficients[term]
                                * tape.gradients[input * K + primary]
                                * tape.gradients[input * K + other];
                        }
                        out.h[primary][other] += weight * curvature;
                    });
                }
                CurvatureAtom::Dense {
                    curvature_start,
                    dimension,
                    support,
                } => {
                    debug_assert_eq!(dimension, K);
                    for_each_supported_upper(support, |primary, other| {
                        out.h[primary][other] += weight
                            * tape.dense_curvatures[curvature_start + primary * dimension + other];
                    });
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

#[inline(always)]
fn for_each_supported_upper(mut rows: u128, mut visit: impl FnMut(usize, usize)) {
    while rows != 0 {
        let primary = rows.trailing_zeros() as usize;
        rows &= rows - 1;
        let mut columns = rows | (1_u128 << primary);
        while columns != 0 {
            let other = columns.trailing_zeros() as usize;
            columns &= columns - 1;
            visit(primary, other);
        }
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
        let tape = self.workspace.tape_mut();
        let mut gradient = [0.0; K];
        for primary in 0..K {
            gradient[primary] = first * tape.gradients[self.node * K + primary];
        }
        let support = tape.nodes[self.node].support;
        let mut hessian_weights_storage = MaybeUninit::uninit();
        let hessian_weights =
            InlineScalars::initialize_zeros(&mut hessian_weights_storage, tape.atom_len);
        Order2GraphWorkspace::inherit_hessian(tape, hessian_weights, self.node, first);
        if second != 0.0 {
            Order2GraphWorkspace::push_weighted_atom(
                tape,
                hessian_weights,
                CurvatureAtom::RankOne { input: self.node },
                second,
            );
        }
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, hessian_weights);
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
        let tape = workspace.tape_mut();
        let mut hessian_weights_storage = MaybeUninit::uninit();
        let hessian_weights =
            InlineScalars::initialize_zeros(&mut hessian_weights_storage, tape.atom_len);
        let node = Order2GraphWorkspace::push(tape, c, 0, [0.0; K], hessian_weights);
        Self { workspace, node }
    }

    #[inline(always)]
    fn variable(x: f64, axis: usize, dimension: usize, workspace: &'arena Self::Workspace) -> Self {
        assert_eq!(dimension, K, "compiled graph dimension mismatch");
        assert!(axis < K, "compiled graph variable axis out of bounds");
        let mut gradient = [0.0; K];
        gradient[axis] = 1.0;
        let tape = workspace.tape_mut();
        let mut hessian_weights_storage = MaybeUninit::uninit();
        let hessian_weights =
            InlineScalars::initialize_zeros(&mut hessian_weights_storage, tape.atom_len);
        let node = Order2GraphWorkspace::push(tape, x, 1_u128 << axis, gradient, hessian_weights);
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
        assert!(
            inputs.len() <= MAX_GRAPH_NODES,
            "compiled graph quadratic arity exceeds graph capacity"
        );
        assert!(
            inputs
                .iter()
                .all(|input| std::ptr::eq(input.workspace, workspace)),
            "compiled graph quadratic inputs belong to different workspaces"
        );

        let input_dimension = inputs.len();
        let mut values_storage = MaybeUninit::uninit();
        let values = InlineScalars::initialize_zeros(&mut values_storage, input_dimension);
        {
            let tape = workspace.tape();
            for (value, input) in values.as_mut_slice().iter_mut().zip(inputs) {
                *value = tape.nodes[input.node].value;
            }
        }
        let mut projected_values_storage = MaybeUninit::uninit();
        let projected_values =
            InlineScalars::initialize_zeros(&mut projected_values_storage, input_dimension);
        coefficients.multiply(values.as_slice(), projected_values.as_mut_slice());
        let value = values
            .as_slice()
            .iter()
            .zip(projected_values.as_slice())
            .map(|(&input, &projected)| input * projected)
            .sum();

        // Project each primary-space input direction through the operator while
        // no tape borrow is live. Besides preserving structured `multiply`
        // implementations, this prevents arbitrary coefficient callbacks from
        // aliasing the workspace's `UnsafeCell` accessors.
        let mut local_curvature = [[0.0; K]; K];
        let mut direction_storage = MaybeUninit::uninit();
        let direction = InlineScalars::initialize_zeros(&mut direction_storage, input_dimension);
        let mut projected_direction_storage = MaybeUninit::uninit();
        let projected_direction =
            InlineScalars::initialize_zeros(&mut projected_direction_storage, input_dimension);
        for other in 0..K {
            {
                let tape = workspace.tape();
                for (channel, input) in direction.as_mut_slice().iter_mut().zip(inputs) {
                    *channel = tape.gradients[input.node * K + other];
                }
            }
            coefficients.multiply(direction.as_slice(), projected_direction.as_mut_slice());
            let tape = workspace.tape();
            for primary in 0..=other {
                let mut curvature = 0.0;
                for (input, &projected) in inputs.iter().zip(projected_direction.as_slice()) {
                    curvature += tape.gradients[input.node * K + primary] * projected;
                }
                local_curvature[primary][other] = 2.0 * curvature;
            }
        }

        let tape = workspace.tape_mut();
        let mut gradient = [0.0; K];
        let mut hessian_weights_storage = MaybeUninit::uninit();
        let hessian_weights =
            InlineScalars::initialize_zeros(&mut hessian_weights_storage, tape.atom_len);
        let mut support = 0_u128;
        for axis in 0..input_dimension {
            let first = 2.0 * projected_values.as_slice()[axis];
            Order2GraphWorkspace::inherit_hessian(tape, hessian_weights, inputs[axis].node, first);
            support |= tape.nodes[inputs[axis].node].support;
            for primary in 0..K {
                gradient[primary] += first * tape.gradients[inputs[axis].node * K + primary];
            }
        }
        let local_curvature = local_curvature.as_flattened();
        if local_curvature.iter().any(|&curvature| curvature != 0.0) {
            let curvature_start = Order2GraphWorkspace::push_dense_curvature(tape, local_curvature);
            Order2GraphWorkspace::push_weighted_atom(
                tape,
                hessian_weights,
                CurvatureAtom::Dense {
                    curvature_start,
                    dimension: K,
                    support,
                },
                1.0,
            );
        }
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, hessian_weights);
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
        let tape = workspace.tape_mut();
        let mut value = 0.0;
        let mut gradient = [0.0; K];
        let mut hessian_weights_storage = MaybeUninit::uninit();
        let hessian_weights =
            InlineScalars::initialize_zeros(&mut hessian_weights_storage, tape.atom_len);
        let mut support = 0_u128;
        for (input, &weight) in inputs.iter().zip(weights) {
            value += weight * tape.nodes[input.node].value;
            Order2GraphWorkspace::inherit_hessian(tape, hessian_weights, input.node, weight);
            support |= tape.nodes[input.node].support;
            for primary in 0..K {
                gradient[primary] += weight * tape.gradients[input.node * K + primary];
            }
        }
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, hessian_weights);
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
        let tape = self.workspace.tape_mut();
        let left_value = tape.nodes[self.node].value;
        let right_value = tape.nodes[right.node].value;
        let mut gradient = [0.0; K];
        for primary in 0..K {
            gradient[primary] = left_value * tape.gradients[right.node * K + primary]
                + tape.gradients[self.node * K + primary] * right_value
                + tape.gradients[addend.node * K + primary];
        }
        let value = left_value * right_value + tape.nodes[addend.node].value;
        let support = tape.nodes[self.node].support
            | tape.nodes[right.node].support
            | tape.nodes[addend.node].support;
        let mut hessian_weights_storage = MaybeUninit::uninit();
        let hessian_weights =
            InlineScalars::initialize_zeros(&mut hessian_weights_storage, tape.atom_len);
        Order2GraphWorkspace::inherit_hessian(tape, hessian_weights, self.node, right_value);
        Order2GraphWorkspace::inherit_hessian(tape, hessian_weights, right.node, left_value);
        Order2GraphWorkspace::inherit_hessian(tape, hessian_weights, addend.node, 1.0);
        Order2GraphWorkspace::push_weighted_atom(
            tape,
            hessian_weights,
            CurvatureAtom::Cross {
                left: self.node,
                right: right.node,
            },
            1.0,
        );
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, hessian_weights);
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
        let tape = workspace.tape_mut();
        let mut value = 0.0;
        let mut gradient = [0.0; K];
        let mut hessian_weights_storage = MaybeUninit::uninit();
        let hessian_weights =
            InlineScalars::initialize_zeros(&mut hessian_weights_storage, tape.atom_len);
        let mut diagonal_coefficients_storage = MaybeUninit::uninit();
        let diagonal_coefficients =
            InlineScalars::initialize_zeros(&mut diagonal_coefficients_storage, tape.node_len);
        let mut support = 0_u128;
        for (input, stack) in inputs.iter().zip(derivative_stacks) {
            value += stack[0];
            Order2GraphWorkspace::inherit_hessian(tape, hessian_weights, input.node, stack[1]);
            diagonal_coefficients.add(input.node, stack[2]);
            support |= tape.nodes[input.node].support;
            for primary in 0..K {
                gradient[primary] += stack[1] * tape.gradients[input.node * K + primary];
            }
        }
        if let Some(atom) = Order2GraphWorkspace::push_diagonal_atom(tape, diagonal_coefficients) {
            debug_assert_eq!(atom, hessian_weights.len());
            hessian_weights.push(1.0);
        }
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, hessian_weights);
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
        let tape = workspace.tape_mut();
        let mut value = 0.0;
        let mut gradient = [0.0; K];
        let mut hessian_weights_storage = MaybeUninit::uninit();
        let hessian_weights =
            InlineScalars::initialize_zeros(&mut hessian_weights_storage, tape.atom_len);
        let mut diagonal_coefficients_storage = MaybeUninit::uninit();
        let diagonal_coefficients =
            InlineScalars::initialize_zeros(&mut diagonal_coefficients_storage, tape.node_len);
        let mut support = 0_u128;
        for ((input, &input_scale), stack) in inputs.iter().zip(input_scales).zip(derivative_stacks)
        {
            let first = stack[1] * input_scale;
            value += stack[0];
            Order2GraphWorkspace::inherit_hessian(tape, hessian_weights, input.node, first);
            diagonal_coefficients.add(input.node, stack[2] * input_scale * input_scale);
            support |= tape.nodes[input.node].support;
            for primary in 0..K {
                gradient[primary] += first * tape.gradients[input.node * K + primary];
            }
        }
        if let Some(atom) = Order2GraphWorkspace::push_diagonal_atom(tape, diagonal_coefficients) {
            debug_assert_eq!(atom, hessian_weights.len());
            hessian_weights.push(1.0);
        }
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, hessian_weights);
        Self { workspace, node }
    }

    #[inline(always)]
    fn dimension(&self) -> usize {
        K
    }

    #[inline(always)]
    fn value(&self) -> f64 {
        self.workspace.tape().nodes[self.node].value
    }

    #[inline(always)]
    fn add(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let tape = self.workspace.tape_mut();
        let mut gradient = [0.0; K];
        for primary in 0..K {
            gradient[primary] =
                tape.gradients[self.node * K + primary] + tape.gradients[other.node * K + primary];
        }
        let value = tape.nodes[self.node].value + tape.nodes[other.node].value;
        let support = tape.nodes[self.node].support | tape.nodes[other.node].support;
        let mut hessian_weights_storage = MaybeUninit::uninit();
        let hessian_weights =
            InlineScalars::initialize_zeros(&mut hessian_weights_storage, tape.atom_len);
        Order2GraphWorkspace::inherit_hessian(tape, hessian_weights, self.node, 1.0);
        Order2GraphWorkspace::inherit_hessian(tape, hessian_weights, other.node, 1.0);
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, hessian_weights);
        Self {
            workspace: self.workspace,
            node,
        }
    }

    #[inline(always)]
    fn sub(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let tape = self.workspace.tape_mut();
        let mut gradient = [0.0; K];
        for primary in 0..K {
            gradient[primary] =
                tape.gradients[self.node * K + primary] - tape.gradients[other.node * K + primary];
        }
        let value = tape.nodes[self.node].value - tape.nodes[other.node].value;
        let support = tape.nodes[self.node].support | tape.nodes[other.node].support;
        let mut hessian_weights_storage = MaybeUninit::uninit();
        let hessian_weights =
            InlineScalars::initialize_zeros(&mut hessian_weights_storage, tape.atom_len);
        Order2GraphWorkspace::inherit_hessian(tape, hessian_weights, self.node, 1.0);
        Order2GraphWorkspace::inherit_hessian(tape, hessian_weights, other.node, -1.0);
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, hessian_weights);
        Self {
            workspace: self.workspace,
            node,
        }
    }

    #[inline(always)]
    fn mul(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let tape = self.workspace.tape_mut();
        let left_value = tape.nodes[self.node].value;
        let right_value = tape.nodes[other.node].value;
        let mut gradient = [0.0; K];
        for primary in 0..K {
            gradient[primary] = left_value * tape.gradients[other.node * K + primary]
                + tape.gradients[self.node * K + primary] * right_value;
        }
        let support = tape.nodes[self.node].support | tape.nodes[other.node].support;
        let mut hessian_weights_storage = MaybeUninit::uninit();
        let hessian_weights =
            InlineScalars::initialize_zeros(&mut hessian_weights_storage, tape.atom_len);
        Order2GraphWorkspace::inherit_hessian(tape, hessian_weights, self.node, right_value);
        Order2GraphWorkspace::inherit_hessian(tape, hessian_weights, other.node, left_value);
        Order2GraphWorkspace::push_weighted_atom(
            tape,
            hessian_weights,
            CurvatureAtom::Cross {
                left: self.node,
                right: other.node,
            },
            1.0,
        );
        let node = Order2GraphWorkspace::push(
            tape,
            left_value * right_value,
            support,
            gradient,
            hessian_weights,
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
        let tape = self.workspace.tape_mut();
        let mut gradient = [0.0; K];
        for primary in 0..K {
            gradient[primary] = scale * tape.gradients[self.node * K + primary];
        }
        let value = scale * tape.nodes[self.node].value;
        let support = tape.nodes[self.node].support;
        let mut hessian_weights_storage = MaybeUninit::uninit();
        let hessian_weights =
            InlineScalars::initialize_zeros(&mut hessian_weights_storage, tape.atom_len);
        Order2GraphWorkspace::inherit_hessian(tape, hessian_weights, self.node, scale);
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, hessian_weights);
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
    use crate::jet_scalar::{DynamicJetArena, DynamicOrder2, FixedRuntimeJet, JetScalar};
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

    struct MatrixFreeDense3<'arena> {
        matrix: [[f64; 3]; 3],
        workspace: &'arena Order2GraphWorkspace,
    }

    impl SymmetricQuadraticCoefficients for MatrixFreeDense3<'_> {
        fn dimension(&self) -> usize {
            3
        }

        fn multiply(&self, input: &[f64], output: &mut [f64]) {
            assert!(self.workspace.node_count() != 0);
            for row in 0..3 {
                output[row] = (0..3)
                    .map(|column| self.matrix[row][column] * input[column])
                    .sum();
            }
        }

        fn coefficient(&self, _row: usize, _column: usize) -> f64 {
            panic!("compiled graph quadratic lowering must preserve matrix-free multiply")
        }
    }

    #[test]
    fn compiled_graph_quadratic_arity_is_independent_and_matrix_free() {
        let mut workspace = Order2GraphWorkspace::new();
        workspace.reset(2);
        let x = Order2Graph::<2>::variable(0.4, 0, 2, &workspace);
        let y = Order2Graph::<2>::variable(-0.7, 1, 2, &workspace);
        let xy = x.product(&y);
        let coefficients = MatrixFreeDense3 {
            matrix: [[1.2, -0.3, 0.25], [-0.3, 0.8, 0.17], [0.25, 0.17, 1.4]],
            workspace: &workspace,
        };
        let graph =
            Order2Graph::<2>::symmetric_quadratic_form(&[x, y, xy], &coefficients, 2, &workspace)
                .into_order2();

        let arena = DynamicJetArena::new();
        let eager_x = DynamicOrder2::variable(0.4, 0, 2, &arena);
        let eager_y = DynamicOrder2::variable(-0.7, 1, 2, &arena);
        let eager_xy = eager_x.product(&eager_y);
        let eager = DynamicOrder2::symmetric_quadratic_form(
            &[eager_x, eager_y, eager_xy],
            &coefficients,
            2,
            &arena,
        );

        let close = |actual: f64, expected: f64| {
            let tolerance = 2.0e-13 * actual.abs().max(expected.abs()).max(1.0);
            assert!((actual - expected).abs() <= tolerance);
        };
        close(graph.value(), eager.v);
        for primary in 0..2 {
            close(graph.g()[primary], eager.g()[primary]);
        }
        for row in 0..2 {
            for column in 0..2 {
                close(graph.h()[row][column], eager.h_at(row, column));
            }
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

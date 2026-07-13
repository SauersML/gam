//! Compiled value/gradient/Hessian lowering for runtime row expressions.
//!
//! [`Order2Graph`] implements the same [`RuntimeJetScalar`] algebra as the eager
//! packed jets, but records a small scalar DAG instead of propagating a dense
//! Hessian through every intermediate. Each node carries its value and primary
//! gradient. Once the scalar output is known, one reverse sweep computes node
//! adjoints and accumulates each nonlinear node's local curvature exactly once.
//!
//! This is the universal second-order chain rule for a scalar DAG. Families own
//! only their one generic row expression and certified unary derivative stacks;
//! this module owns the compiled lowering schedule.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

use crate::jet_scalar::{Order2, RuntimeJetScalar, SymmetricQuadraticCoefficients};

#[derive(Clone, Copy, Debug)]
struct GraphNode {
    value: f64,
    support: u16,
    edge_start: u16,
    edge_len: u16,
}

#[derive(Clone, Copy, Debug)]
enum CurvatureEvent {
    RankOne {
        owner: u8,
        input: u8,
        second: f64,
    },
    Cross {
        owner: u8,
        left: u8,
        right: u8,
    },
    Diagonal {
        owner: u8,
        term_start: u16,
        len: u16,
    },
    Projected {
        owner: u8,
        curvature_start: u16,
        support: u16,
    },
}

const MAX_PRIMARY_DIMENSION: usize = 16;
const MAX_QUADRATIC_ARITY: usize = 32;
const MAX_GRAPH_NODES: usize = 64;
const MAX_GRAPH_EDGES: usize = MAX_GRAPH_NODES * MAX_GRAPH_NODES;
const MAX_PROJECTED_CURVATURE_VALUES: usize =
    MAX_GRAPH_NODES * MAX_PRIMARY_DIMENSION * (MAX_PRIMARY_DIMENSION + 1) / 2;
const EMPTY_NODE: GraphNode = GraphNode {
    value: 0.0,
    support: 0,
    edge_start: 0,
    edge_len: 0,
};
const EMPTY_CURVATURE_EVENT: CurvatureEvent = CurvatureEvent::RankOne {
    owner: 0,
    input: 0,
    second: 0.0,
};

/// Fixed-capacity scalar scratch with only its logical prefix initialized.
///
/// Quadratic arity is usually far below the hard node capacity. Keeping the
/// inactive suffix uninitialized prevents full-capacity scratch clears while
/// retaining allocation-free, checked storage for the general graph contract.
struct InlineScalars {
    slots: [MaybeUninit<f64>; MAX_QUADRATIC_ARITY],
    len: usize,
}

impl InlineScalars {
    #[inline(always)]
    fn initialize_zeros(storage: &mut MaybeUninit<Self>, len: usize) -> &mut Self {
        assert!(
            len <= MAX_QUADRATIC_ARITY,
            "inline scalar capacity exceeded"
        );
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
    fn as_slice(&self) -> &[f64] {
        // SAFETY: exactly the prefix `0..len` is initialized by `initialize_zeros`;
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
    edge_len: usize,
    event_len: usize,
    diagonal_len: usize,
    projected_curvature_len: usize,
    nodes: [GraphNode; MAX_GRAPH_NODES],
    gradients: [f64; MAX_GRAPH_NODES * MAX_PRIMARY_DIMENSION],
    adjoints: [f64; MAX_GRAPH_NODES],
    edge_parents: [u8; MAX_GRAPH_EDGES],
    edge_firsts: [f64; MAX_GRAPH_EDGES],
    events: [CurvatureEvent; MAX_GRAPH_NODES],
    diagonal_inputs: [u8; MAX_GRAPH_EDGES],
    diagonal_seconds: [f64; MAX_GRAPH_EDGES],
    projected_curvatures: [f64; MAX_PROJECTED_CURVATURE_VALUES],
}

impl GraphTape {
    fn new() -> Self {
        Self {
            dimension: 0,
            node_len: 0,
            edge_len: 0,
            event_len: 0,
            diagonal_len: 0,
            projected_curvature_len: 0,
            nodes: [EMPTY_NODE; MAX_GRAPH_NODES],
            gradients: [0.0; MAX_GRAPH_NODES * MAX_PRIMARY_DIMENSION],
            adjoints: [0.0; MAX_GRAPH_NODES],
            edge_parents: [0; MAX_GRAPH_EDGES],
            edge_firsts: [0.0; MAX_GRAPH_EDGES],
            events: [EMPTY_CURVATURE_EVENT; MAX_GRAPH_NODES],
            diagonal_inputs: [0; MAX_GRAPH_EDGES],
            diagonal_seconds: [0.0; MAX_GRAPH_EDGES],
            projected_curvatures: [0.0; MAX_PROJECTED_CURVATURE_VALUES],
        }
    }
}

/// Reusable storage for a compiled scalar DAG.
///
/// Reset between rows. The boxed tape has checked fixed capacities, so every
/// row after worker construction performs no tape or reverse-sweep allocation
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
        tape.edge_len = 0;
        tape.event_len = 0;
        tape.diagonal_len = 0;
        tape.projected_curvature_len = 0;
    }

    /// Shared tape access. `UnsafeCell` removes the dynamic borrow state that a
    /// `RefCell` would place in every scalar operation. The workspace is not
    /// `Sync`, every mutation is completed before a scalar handle is returned,
    /// and lowering starts only after expression construction, so no aliases to
    /// the tape contents escape these two private accessors.
    #[inline(always)]
    fn tape(&self) -> &GraphTape {
        // SAFETY: the workspace is not Sync, scalar construction completes each
        // mutation before returning, and lowering starts only after construction,
        // so no mutable reference aliases this shared tape reference.
        unsafe { (&*self.tape.get()).as_ref() }
    }

    #[inline(always)]
    fn tape_mut(&self) -> &mut GraphTape {
        // SAFETY: the workspace is not Sync and every tape mutation is serialized
        // by the expression-construction/lowering protocol described on `tape`.
        unsafe { (&mut *self.tape.get()).as_mut() }
    }

    #[inline(always)]
    fn push<const K: usize>(
        tape: &mut GraphTape,
        value: f64,
        support: u16,
        gradient: [f64; K],
        edge_start: usize,
    ) -> usize {
        assert_eq!(tape.dimension, K, "compiled graph dimension mismatch");
        assert!(
            tape.node_len < MAX_GRAPH_NODES,
            "compiled graph node capacity exceeded"
        );
        let node = tape.node_len;
        let edge_len = tape.edge_len - edge_start;
        tape.nodes[node] = GraphNode {
            value,
            support,
            edge_start: edge_start as u16,
            edge_len: edge_len as u16,
        };
        tape.gradients[node * K..(node + 1) * K].copy_from_slice(&gradient);
        tape.node_len += 1;
        node
    }

    #[inline(always)]
    fn push_edge(tape: &mut GraphTape, parent: usize, first: f64) {
        assert!(
            tape.edge_len < MAX_GRAPH_EDGES,
            "compiled graph edge capacity exceeded"
        );
        assert!(
            parent < tape.node_len,
            "compiled graph edge parent must name an existing node"
        );
        tape.edge_parents[tape.edge_len] = parent as u8;
        tape.edge_firsts[tape.edge_len] = first;
        tape.edge_len += 1;
    }

    #[inline(always)]
    fn push_event(tape: &mut GraphTape, event: CurvatureEvent) {
        assert!(
            tape.event_len < MAX_GRAPH_NODES,
            "compiled graph curvature-event capacity exceeded"
        );
        tape.events[tape.event_len] = event;
        tape.event_len += 1;
    }

    #[inline(always)]
    fn push_diagonal_term(tape: &mut GraphTape, input: usize, second: f64) {
        assert!(
            tape.diagonal_len < MAX_GRAPH_EDGES,
            "compiled graph diagonal-term capacity exceeded"
        );
        assert!(
            input < tape.node_len,
            "compiled graph diagonal input must name an existing node"
        );
        tape.diagonal_inputs[tape.diagonal_len] = input as u8;
        tape.diagonal_seconds[tape.diagonal_len] = second;
        tape.diagonal_len += 1;
    }

    #[inline(always)]
    fn lower<const K: usize>(&self, output: usize) -> Order2<K> {
        let tape = self.tape_mut();
        assert_eq!(tape.dimension, K, "compiled graph dimension mismatch");
        assert!(output < tape.node_len, "compiled graph output is absent");
        tape.adjoints[..tape.node_len].fill(0.0);
        tape.adjoints[output] = 1.0;

        let mut out = crate::jet_tower::Tower2::zero();
        out.v = tape.nodes[output].value;
        for primary in 0..K {
            out.g[primary] = tape.gradients[output * K + primary];
        }

        for node_index in (0..=output).rev() {
            let adjoint = tape.adjoints[node_index];
            if adjoint == 0.0 {
                continue;
            }
            let node = tape.nodes[node_index];
            let edge_start = node.edge_start as usize;
            for edge in edge_start..edge_start + node.edge_len as usize {
                let parent = tape.edge_parents[edge] as usize;
                tape.adjoints[parent] += adjoint * tape.edge_firsts[edge];
            }
        }

        for event_index in (0..tape.event_len).rev() {
            match tape.events[event_index] {
                CurvatureEvent::RankOne {
                    owner,
                    input,
                    second,
                } => {
                    let owner_adjoint = tape.adjoints[owner as usize];
                    if owner_adjoint == 0.0 {
                        continue;
                    }
                    let input = input as usize;
                    let curvature_scale = owner_adjoint * second;
                    for_each_supported_upper(tape.nodes[input].support, |primary, other| {
                        out.h[primary][other] += curvature_scale
                            * tape.gradients[input * K + primary]
                            * tape.gradients[input * K + other];
                    });
                }
                CurvatureEvent::Cross { owner, left, right } => {
                    let owner_adjoint = tape.adjoints[owner as usize];
                    if owner_adjoint == 0.0 {
                        continue;
                    }
                    let left = left as usize;
                    let right = right as usize;
                    for_each_supported_upper(
                        tape.nodes[left].support | tape.nodes[right].support,
                        |primary, other| {
                            let left_primary = tape.gradients[left * K + primary];
                            let right_primary = tape.gradients[right * K + primary];
                            let curvature = left_primary * tape.gradients[right * K + other]
                                + right_primary * tape.gradients[left * K + other];
                            out.h[primary][other] += owner_adjoint * curvature;
                        },
                    );
                }
                CurvatureEvent::Diagonal {
                    owner,
                    term_start,
                    len,
                } => {
                    let owner_adjoint = tape.adjoints[owner as usize];
                    if owner_adjoint == 0.0 {
                        continue;
                    }
                    let term_start = term_start as usize;
                    for term in term_start..term_start + len as usize {
                        let input = tape.diagonal_inputs[term] as usize;
                        let curvature_scale = owner_adjoint * tape.diagonal_seconds[term];
                        if curvature_scale == 0.0 {
                            continue;
                        }
                        for_each_supported_upper(tape.nodes[input].support, |primary, other| {
                            out.h[primary][other] += curvature_scale
                                * tape.gradients[input * K + primary]
                                * tape.gradients[input * K + other];
                        });
                    }
                }
                CurvatureEvent::Projected {
                    owner,
                    curvature_start,
                    support,
                } => {
                    let owner_adjoint = tape.adjoints[owner as usize];
                    if owner_adjoint == 0.0 {
                        continue;
                    }
                    let mut curvature_offset = 0;
                    let curvature_start = curvature_start as usize;
                    for_each_supported_upper_by_column(support, |primary, other| {
                        out.h[primary][other] += owner_adjoint
                            * tape.projected_curvatures[curvature_start + curvature_offset];
                        curvature_offset += 1;
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
fn for_each_supported_upper_by_column(support: u16, mut visit: impl FnMut(usize, usize)) {
    let mut columns = support;
    while columns != 0 {
        let other = columns.trailing_zeros() as usize;
        columns &= columns - 1;
        let mut rows = support & (u16::MAX >> (MAX_PRIMARY_DIMENSION - other - 1));
        while rows != 0 {
            let primary = rows.trailing_zeros() as usize;
            rows &= rows - 1;
            visit(primary, other);
        }
    }
}

#[inline(always)]
fn for_each_supported_upper(mut rows: u16, mut visit: impl FnMut(usize, usize)) {
    while rows != 0 {
        let primary = rows.trailing_zeros() as usize;
        rows &= rows - 1;
        let mut columns = rows | (1_u16 << primary);
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
        let owner = tape.node_len;
        let edge_start = tape.edge_len;
        Order2GraphWorkspace::push_edge(tape, self.node, first);
        Order2GraphWorkspace::push_event(
            tape,
            CurvatureEvent::RankOne {
                owner: owner as u8,
                input: self.node as u8,
                second,
            },
        );
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, edge_start);
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
        let edge_start = tape.edge_len;
        let node = Order2GraphWorkspace::push(tape, c, 0, [0.0; K], edge_start);
        Self { workspace, node }
    }

    #[inline(always)]
    fn variable(x: f64, axis: usize, dimension: usize, workspace: &'arena Self::Workspace) -> Self {
        assert_eq!(dimension, K, "compiled graph dimension mismatch");
        assert!(axis < K, "compiled graph variable axis out of bounds");
        let mut gradient = [0.0; K];
        gradient[axis] = 1.0;
        let tape = workspace.tape_mut();
        let edge_start = tape.edge_len;
        let node = Order2GraphWorkspace::push(tape, x, 1_u16 << axis, gradient, edge_start);
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
            inputs.len() <= MAX_QUADRATIC_ARITY,
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
        let mut support = 0_u16;
        {
            let tape = workspace.tape();
            for (value, input) in values.as_mut_slice().iter_mut().zip(inputs) {
                *value = tape.nodes[input.node].value;
                support |= tape.nodes[input.node].support;
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
        let mut direction_storage = MaybeUninit::uninit();
        let direction = InlineScalars::initialize_zeros(&mut direction_storage, input_dimension);
        let mut projected_direction_storage = MaybeUninit::uninit();
        let projected_direction =
            InlineScalars::initialize_zeros(&mut projected_direction_storage, input_dimension);
        let supported_dimension = support.count_ones() as usize;
        let curvature_len = supported_dimension * (supported_dimension + 1) / 2;
        let curvature_start = {
            let tape = workspace.tape_mut();
            assert!(
                curvature_len <= MAX_PROJECTED_CURVATURE_VALUES - tape.projected_curvature_len,
                "compiled graph projected-curvature capacity exceeded"
            );
            let start = tape.projected_curvature_len;
            tape.projected_curvature_len += curvature_len;
            start
        };
        let mut curvature_offset = 0;
        let mut supported_columns = support;
        while supported_columns != 0 {
            let other = supported_columns.trailing_zeros() as usize;
            supported_columns &= supported_columns - 1;
            {
                let tape = workspace.tape();
                for (channel, input) in direction.as_mut_slice().iter_mut().zip(inputs) {
                    *channel = tape.gradients[input.node * K + other];
                }
            }
            coefficients.multiply(direction.as_slice(), projected_direction.as_mut_slice());
            let mut supported_rows = support & (u16::MAX >> (MAX_PRIMARY_DIMENSION - other - 1));
            while supported_rows != 0 {
                let primary = supported_rows.trailing_zeros() as usize;
                supported_rows &= supported_rows - 1;
                let curvature = {
                    let tape = workspace.tape();
                    inputs
                        .iter()
                        .zip(projected_direction.as_slice())
                        .map(|(input, &projected)| {
                            tape.gradients[input.node * K + primary] * projected
                        })
                        .sum::<f64>()
                };
                workspace.tape_mut().projected_curvatures[curvature_start + curvature_offset] =
                    2.0 * curvature;
                curvature_offset += 1;
            }
        }
        assert_eq!(
            curvature_offset, curvature_len,
            "compiled graph projected-curvature schedule must fill its exact packed range"
        );

        let tape = workspace.tape_mut();
        let owner = tape.node_len;
        let edge_start = tape.edge_len;
        let mut gradient = [0.0; K];
        for axis in 0..input_dimension {
            let first = 2.0 * projected_values.as_slice()[axis];
            Order2GraphWorkspace::push_edge(tape, inputs[axis].node, first);
            for primary in 0..K {
                gradient[primary] += first * tape.gradients[inputs[axis].node * K + primary];
            }
        }
        Order2GraphWorkspace::push_event(
            tape,
            CurvatureEvent::Projected {
                owner: owner as u8,
                curvature_start: curvature_start as u16,
                support,
            },
        );
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, edge_start);
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
        let edge_start = tape.edge_len;
        let mut value = 0.0;
        let mut gradient = [0.0; K];
        let mut support = 0_u16;
        for (input, &weight) in inputs.iter().zip(weights) {
            value += weight * tape.nodes[input.node].value;
            Order2GraphWorkspace::push_edge(tape, input.node, weight);
            support |= tape.nodes[input.node].support;
            for primary in 0..K {
                gradient[primary] += weight * tape.gradients[input.node * K + primary];
            }
        }
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, edge_start);
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
        let owner = tape.node_len;
        let edge_start = tape.edge_len;
        Order2GraphWorkspace::push_edge(tape, self.node, right_value);
        Order2GraphWorkspace::push_edge(tape, right.node, left_value);
        Order2GraphWorkspace::push_edge(tape, addend.node, 1.0);
        Order2GraphWorkspace::push_event(
            tape,
            CurvatureEvent::Cross {
                owner: owner as u8,
                left: self.node as u8,
                right: right.node as u8,
            },
        );
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, edge_start);
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
        let owner = tape.node_len;
        let edge_start = tape.edge_len;
        let term_start = tape.diagonal_len;
        let mut value = 0.0;
        let mut gradient = [0.0; K];
        let mut support = 0_u16;
        for (input, stack) in inputs.iter().zip(derivative_stacks) {
            value += stack[0];
            Order2GraphWorkspace::push_edge(tape, input.node, stack[1]);
            Order2GraphWorkspace::push_diagonal_term(tape, input.node, stack[2]);
            support |= tape.nodes[input.node].support;
            for primary in 0..K {
                gradient[primary] += stack[1] * tape.gradients[input.node * K + primary];
            }
        }
        Order2GraphWorkspace::push_event(
            tape,
            CurvatureEvent::Diagonal {
                owner: owner as u8,
                term_start: term_start as u16,
                len: inputs.len() as u16,
            },
        );
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, edge_start);
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
        let owner = tape.node_len;
        let edge_start = tape.edge_len;
        let term_start = tape.diagonal_len;
        let mut value = 0.0;
        let mut gradient = [0.0; K];
        let mut support = 0_u16;
        for ((input, &input_scale), stack) in inputs.iter().zip(input_scales).zip(derivative_stacks)
        {
            let first = stack[1] * input_scale;
            let second = stack[2] * input_scale * input_scale;
            value += stack[0];
            Order2GraphWorkspace::push_edge(tape, input.node, first);
            Order2GraphWorkspace::push_diagonal_term(tape, input.node, second);
            support |= tape.nodes[input.node].support;
            for primary in 0..K {
                gradient[primary] += first * tape.gradients[input.node * K + primary];
            }
        }
        Order2GraphWorkspace::push_event(
            tape,
            CurvatureEvent::Diagonal {
                owner: owner as u8,
                term_start: term_start as u16,
                len: inputs.len() as u16,
            },
        );
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, edge_start);
        Self { workspace, node }
    }

    #[inline(always)]
    fn scaled_multiply_add_affine_composed_sum<const N: usize>(
        lefts: &[&Self; N],
        rights: &[&Self; N],
        addends: &[&Self; N],
        addend_scales: &[f64; N],
        input_scales: &[f64; N],
        derivative_stacks: &[[f64; 5]; N],
        dimension: usize,
        workspace: &'arena Self::Workspace,
    ) -> Self {
        assert_eq!(dimension, K, "compiled graph dimension mismatch");
        assert!(
            lefts
                .iter()
                .chain(rights)
                .all(|input| std::ptr::eq(input.workspace, workspace)),
            "compiled fused product-composition inputs belong to different workspaces"
        );
        assert!(
            addends
                .iter()
                .zip(addend_scales)
                .all(|(input, &scale)| scale == 0.0 || std::ptr::eq(input.workspace, workspace)),
            "live compiled fused addends belong to different workspaces"
        );

        let tape = workspace.tape_mut();
        let mut term_gradients = [[0.0; K]; N];
        let mut firsts = [0.0; N];
        let mut seconds = [0.0; N];
        let mut value = 0.0;
        let mut gradient = [0.0; K];
        let mut support = 0_u16;
        for term in 0..N {
            let left = lefts[term].node;
            let right = rights[term].node;
            let addend = addends[term].node;
            let left_value = tape.nodes[left].value;
            let right_value = tape.nodes[right].value;
            let first = derivative_stacks[term][1] * input_scales[term];
            let second = derivative_stacks[term][2] * input_scales[term] * input_scales[term];
            firsts[term] = first;
            seconds[term] = second;
            value += derivative_stacks[term][0];
            support |= tape.nodes[left].support | tape.nodes[right].support;
            if addend_scales[term] != 0.0 {
                support |= tape.nodes[addend].support;
            }
            for primary in 0..K {
                let product_gradient = left_value * tape.gradients[right * K + primary]
                    + tape.gradients[left * K + primary] * right_value;
                let inner_gradient = if addend_scales[term] == 0.0 {
                    product_gradient
                } else if addend_scales[term] == 1.0 {
                    product_gradient + tape.gradients[addend * K + primary]
                } else {
                    product_gradient + addend_scales[term] * tape.gradients[addend * K + primary]
                };
                term_gradients[term][primary] = inner_gradient;
                gradient[primary] += first * inner_gradient;
            }
        }

        let supported_dimension = support.count_ones() as usize;
        let curvature_len = supported_dimension * (supported_dimension + 1) / 2;
        assert!(
            curvature_len <= MAX_PROJECTED_CURVATURE_VALUES - tape.projected_curvature_len,
            "compiled graph projected-curvature capacity exceeded"
        );
        let curvature_start = tape.projected_curvature_len;
        tape.projected_curvature_len += curvature_len;
        let mut curvature_offset = 0;
        for_each_supported_upper_by_column(support, |primary, other| {
            let mut channel = 0.0;
            for term in 0..N {
                let left = lefts[term].node;
                let right = rights[term].node;
                let cross = tape.gradients[left * K + primary] * tape.gradients[right * K + other]
                    + tape.gradients[left * K + other] * tape.gradients[right * K + primary];
                channel += firsts[term] * cross
                    + seconds[term] * term_gradients[term][primary] * term_gradients[term][other];
            }
            tape.projected_curvatures[curvature_start + curvature_offset] = channel;
            curvature_offset += 1;
        });
        assert_eq!(
            curvature_offset, curvature_len,
            "compiled fused product-composition must fill its packed curvature"
        );

        let owner = tape.node_len;
        let edge_start = tape.edge_len;
        for term in 0..N {
            let left = lefts[term].node;
            let right = rights[term].node;
            let addend = addends[term].node;
            let left_first = firsts[term] * tape.nodes[right].value;
            let right_first = firsts[term] * tape.nodes[left].value;
            Order2GraphWorkspace::push_edge(tape, left, left_first);
            Order2GraphWorkspace::push_edge(tape, right, right_first);
            if addend_scales[term] != 0.0 {
                Order2GraphWorkspace::push_edge(tape, addend, firsts[term] * addend_scales[term]);
            }
        }
        Order2GraphWorkspace::push_event(
            tape,
            CurvatureEvent::Projected {
                owner: owner as u8,
                curvature_start: curvature_start as u16,
                support,
            },
        );
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, edge_start);
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
        let edge_start = tape.edge_len;
        Order2GraphWorkspace::push_edge(tape, self.node, 1.0);
        Order2GraphWorkspace::push_edge(tape, other.node, 1.0);
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, edge_start);
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
        let edge_start = tape.edge_len;
        Order2GraphWorkspace::push_edge(tape, self.node, 1.0);
        Order2GraphWorkspace::push_edge(tape, other.node, -1.0);
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, edge_start);
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
        let owner = tape.node_len;
        let edge_start = tape.edge_len;
        Order2GraphWorkspace::push_edge(tape, self.node, right_value);
        Order2GraphWorkspace::push_edge(tape, other.node, left_value);
        Order2GraphWorkspace::push_event(
            tape,
            CurvatureEvent::Cross {
                owner: owner as u8,
                left: self.node as u8,
                right: other.node as u8,
            },
        );
        let node = Order2GraphWorkspace::push(
            tape,
            left_value * right_value,
            support,
            gradient,
            edge_start,
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
        let edge_start = tape.edge_len;
        Order2GraphWorkspace::push_edge(tape, self.node, scale);
        let node = Order2GraphWorkspace::push(tape, value, support, gradient, edge_start);
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
    use std::cell::Cell;

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
            assert!(self.workspace.tape().node_len != 0);
            for row in 0..3 {
                output[row] = (0..3)
                    .map(|column| self.matrix[row][column] * input[column])
                    .sum();
            }
        }

        fn coefficient(&self, _: usize, _: usize) -> f64 {
            panic!("compiled graph quadratic lowering must preserve matrix-free multiply")
        }
    }

    struct MatrixFreeIdentity32<'arena> {
        workspace: &'arena Order2GraphWorkspace,
    }

    impl SymmetricQuadraticCoefficients for MatrixFreeIdentity32<'_> {
        fn dimension(&self) -> usize {
            MAX_QUADRATIC_ARITY
        }

        fn multiply(&self, input: &[f64], output: &mut [f64]) {
            assert!(self.workspace.tape().node_len != 0);
            output.copy_from_slice(input);
        }

        fn coefficient(&self, _: usize, _: usize) -> f64 {
            panic!("compiled graph quadratic lowering must preserve matrix-free multiply")
        }
    }

    struct CountingIdentity3<'arena> {
        workspace: &'arena Order2GraphWorkspace,
        multiply_calls: Cell<usize>,
    }

    impl SymmetricQuadraticCoefficients for CountingIdentity3<'_> {
        fn dimension(&self) -> usize {
            3
        }

        fn multiply(&self, input: &[f64], output: &mut [f64]) {
            assert!(self.workspace.tape().node_len != 0);
            self.multiply_calls.set(self.multiply_calls.get() + 1);
            output.copy_from_slice(input);
        }

        fn coefficient(&self, _: usize, _: usize) -> f64 {
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

    #[test]
    fn compiled_graph_accepts_maximum_quadratic_arity_plus_output_node() {
        let mut workspace = Order2GraphWorkspace::new();
        workspace.reset(MAX_PRIMARY_DIMENSION);
        let values: [f64; MAX_QUADRATIC_ARITY] =
            std::array::from_fn(|axis| 0.01 * (axis + 1) as f64);
        let vars: [Order2Graph<'_, MAX_PRIMARY_DIMENSION>; MAX_QUADRATIC_ARITY] =
            std::array::from_fn(|axis| {
                Order2Graph::variable(
                    values[axis],
                    axis % MAX_PRIMARY_DIMENSION,
                    MAX_PRIMARY_DIMENSION,
                    &workspace,
                )
            });
        let coefficients = MatrixFreeIdentity32 {
            workspace: &workspace,
        };
        let graph = Order2Graph::symmetric_quadratic_form(
            &vars,
            &coefficients,
            MAX_PRIMARY_DIMENSION,
            &workspace,
        )
        .into_order2();

        let expected_value = values.iter().map(|value| value * value).sum::<f64>();
        let tolerance = 2.0e-13;
        assert!((graph.value() - expected_value).abs() <= tolerance);
        for primary in 0..MAX_PRIMARY_DIMENSION {
            let expected_gradient = 2.0 * (values[primary] + values[primary + 16]);
            assert!((graph.g()[primary] - expected_gradient).abs() <= tolerance);
            for other in 0..MAX_PRIMARY_DIMENSION {
                let expected_hessian = if primary == other { 4.0 } else { 0.0 };
                assert!((graph.h()[primary][other] - expected_hessian).abs() <= tolerance);
            }
        }
    }

    #[test]
    fn compiled_graph_projects_only_sparse_supported_primary_directions() {
        let mut workspace = Order2GraphWorkspace::new();
        workspace.reset(MAX_PRIMARY_DIMENSION);
        let x = Order2Graph::<MAX_PRIMARY_DIMENSION>::variable(
            0.4,
            0,
            MAX_PRIMARY_DIMENSION,
            &workspace,
        );
        let y = Order2Graph::<MAX_PRIMARY_DIMENSION>::variable(
            -0.7,
            7,
            MAX_PRIMARY_DIMENSION,
            &workspace,
        );
        let z = Order2Graph::<MAX_PRIMARY_DIMENSION>::variable(
            1.1,
            15,
            MAX_PRIMARY_DIMENSION,
            &workspace,
        );
        let coefficients = CountingIdentity3 {
            workspace: &workspace,
            multiply_calls: Cell::new(0),
        };
        let graph = Order2Graph::symmetric_quadratic_form(
            &[x, y, z],
            &coefficients,
            MAX_PRIMARY_DIMENSION,
            &workspace,
        )
        .into_order2();

        assert_eq!(coefficients.multiply_calls.get(), 4);
        for primary in 0..MAX_PRIMARY_DIMENSION {
            let active = matches!(primary, 0 | 7 | 15);
            let expected_gradient = match primary {
                0 => 0.8,
                7 => -1.4,
                15 => 2.2,
                _ => 0.0,
            };
            assert_eq!(graph.g()[primary], expected_gradient);
            for other in 0..MAX_PRIMARY_DIMENSION {
                assert_eq!(
                    graph.h()[primary][other],
                    if active && primary == other { 2.0 } else { 0.0 }
                );
            }
        }
    }

    #[test]
    fn compiled_fused_addend_support_obeys_structural_coefficient() {
        let mut workspace = Order2GraphWorkspace::new();
        {
            workspace.reset(3);
            let left = Order2Graph::<3>::variable(0.4, 0, 3, &workspace);
            let right = Order2Graph::<3>::variable(-0.7, 1, 3, &workspace);
            let omitted = Order2Graph::<3>::variable(1.1, 2, 3, &workspace);
            let output = Order2Graph::scaled_multiply_add_affine_composed_sum(
                &[&left],
                &[&right],
                &[&omitted],
                &[-0.0],
                &[1.0],
                &[[0.2, 0.0, 1.0, 0.0, 0.0]],
                3,
                &workspace,
            );
            assert_eq!(workspace.tape().nodes[output.node].support, 0b011);
            let channels = output.into_order2();
            assert!(channels.g()[2] == 0.0);
            assert!((0..3).all(|axis| channels.h()[axis][2] == 0.0));
        }

        workspace.reset(3);
        let left = Order2Graph::<3>::variable(0.4, 0, 3, &workspace);
        let right = Order2Graph::<3>::variable(-0.7, 1, 3, &workspace);
        let live = Order2Graph::<3>::variable(1.1, 2, 3, &workspace);
        let output = Order2Graph::scaled_multiply_add_affine_composed_sum(
            &[&left],
            &[&right],
            &[&live],
            &[1.0],
            &[1.0],
            &[[0.2, 0.0, 1.0, 0.0, 0.0]],
            3,
            &workspace,
        );
        assert_eq!(workspace.tape().nodes[output.node].support, 0b111);
        let channels = output.into_order2();
        assert_eq!(channels.g()[2], 0.0);
        assert_eq!(channels.h()[2][2], 1.0);
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

    fn fused_expression<'arena, S: RuntimeJetScalar<'arena>>(
        vars: &[S; 6],
        scales: &[f64; 4],
        stacks: &[[f64; 5]; 4],
        workspace: &'arena S::Workspace,
    ) -> S {
        const N: usize = 10;
        let upstream = [
            vars[0].product(&vars[1]),
            vars[2].affine_compose(scales[0], scales[1], stacks[0], workspace),
            vars[3].multiply_add(&vars[4], &vars[5]),
        ];
        let lefts: [&S; N] = std::array::from_fn(|term| &upstream[term % upstream.len()]);
        let rights: [&S; N] =
            std::array::from_fn(|term| &upstream[(3 * term + 1) % upstream.len()]);
        let addends: [&S; N] = std::array::from_fn(|term| &vars[(5 * term + 2) % vars.len()]);
        let addend_scales: [f64; N] = std::array::from_fn(|term| match term % 4 {
            0 => 0.0,
            1 => 1.0,
            2 => -0.75,
            _ => 0.35,
        });
        let input_scales: [f64; N] = std::array::from_fn(|term| match term {
            0 => 0.0,
            1 => -1.25,
            _ => scales[term % scales.len()],
        });
        let derivative_stacks: [[f64; 5]; N] =
            std::array::from_fn(|term| stacks[term % stacks.len()]);
        S::scaled_multiply_add_affine_composed_sum(
            &lefts,
            &rights,
            &addends,
            &addend_scales,
            &input_scales,
            &derivative_stacks,
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
        let mut fused_workspace = Order2GraphWorkspace::new();

        fused_workspace.reset(6);
        let empty_terms: [&Order2Graph<'_, 6>; 0] = [];
        let empty_scales: [f64; 0] = [];
        let empty_stacks: [[f64; 5]; 0] = [];
        let empty = Order2Graph::scaled_multiply_add_affine_composed_sum(
            &empty_terms,
            &empty_terms,
            &empty_terms,
            &empty_scales,
            &empty_scales,
            &empty_stacks,
            6,
            &fused_workspace,
        )
        .into_order2();
        assert_eq!(empty.value().to_bits(), 0.0_f64.to_bits());
        assert!(empty.g().iter().all(|&channel| channel == 0.0));
        assert!(empty.h().iter().flatten().all(|&channel| channel == 0.0));

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
            let eager_fused = fused_expression(&eager_vars, &scales, &stacks, &()).into_inner();

            workspace.reset(6);
            let graph_vars: [Order2Graph<'_, 6>; 6] = std::array::from_fn(|axis| {
                Order2Graph::variable(values[axis], axis, 6, &workspace)
            });
            let graph =
                expression(&graph_vars, &coefficients, &scales, &stacks, &workspace).into_order2();
            fused_workspace.reset(6);
            let fused_graph_vars: [Order2Graph<'_, 6>; 6] = std::array::from_fn(|axis| {
                Order2Graph::variable(values[axis], axis, 6, &fused_workspace)
            });
            let graph_fused =
                fused_expression(&fused_graph_vars, &scales, &stacks, &fused_workspace)
                    .into_order2();

            close(graph.value(), eager.value(), case, "value");
            close(
                graph_fused.value(),
                eager_fused.value(),
                case,
                "fused value",
            );
            for primary in 0..6 {
                close(graph.g()[primary], eager.g()[primary], case, "gradient");
                close(
                    graph_fused.g()[primary],
                    eager_fused.g()[primary],
                    case,
                    "fused gradient",
                );
                for other in 0..6 {
                    close(
                        graph.h()[primary][other],
                        eager.h()[primary][other],
                        case,
                        "Hessian",
                    );
                    close(
                        graph_fused.h()[primary][other],
                        eager_fused.h()[primary][other],
                        case,
                        "fused Hessian",
                    );
                }
            }
        }
    }
}

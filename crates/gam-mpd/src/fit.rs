//! The kinds of structural proposal the program decomposition searches over (#2951).
//!
//! Each proposal is scored by the two-part code of the program it yields; fidelity alone
//! never accepts, so an operator that interpolates the teacher still loses when its code is
//! longer.

/// A structural proposal on the current artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProposalKind {
    /// Two families of one shape class and label-manifold type become one field, or
    /// tied uses call one shared body.
    Share,
    /// A component becomes two whose sum is that component at the start.
    Split,
    /// One more basis function or label dimension, starting at its prior mean.
    Refine,
    /// A component, a rank or a basis function is removed, or a tensor returns to its
    /// native primitive.
    Reduce,
    /// A component is born from the residual at a violating witness, or a recovered
    /// structured coordinate becomes a program node.
    Expose,
}

//! Leaf error type for basis construction.
//!
//! `BasisError` lives in the neutral `gam-problem` crate (not `gam-terms`) so
//! that downstream consumers — families, design assembly, the terms cluster —
//! can resolve it without dragging in a `gam-terms` dependency cycle.

use gam_linalg::faer_ndarray::FaerLinalgError;
use thiserror::Error;

/// A comprehensive error type for all operations within the basis module.
#[derive(Error, Debug)]
pub enum BasisError {
    #[error("Spline degree must be at least 1, but was {0}.")]
    InvalidDegree(usize),

    #[error(
        "Spline degree {degree} is too low for derivative order {derivative_order}; need degree >= {minimum_degree}."
    )]
    InsufficientDegreeForDerivative {
        degree: usize,
        derivative_order: usize,
        minimum_degree: usize,
    },

    #[error("Data range is invalid: start ({0}) must be less than or equal to end ({1}).")]
    InvalidRange(f64, f64),

    #[error(
        "Data range has zero width (min equals max), which collapses the B-spline knot domain; requested {0} internal knots."
    )]
    DegenerateRange(usize),

    #[error(
        "Penalty order ({order}) must be positive and less than the number of basis functions ({num_basis})."
    )]
    InvalidPenaltyOrder { order: usize, num_basis: usize },

    #[error(
        "Insufficient knots for degree {degree} spline: need at least {required} knots but only {provided} were provided."
    )]
    InsufficientKnotsForDegree {
        degree: usize,
        required: usize,
        provided: usize,
    },

    #[error(
        "Cannot apply sum-to-zero constraint: requires at least 2 basis functions, but only {found} were provided."
    )]
    InsufficientColumnsForConstraint { found: usize },

    #[error(
        "Constraint matrix must have the same number of rows as the basis: basis has {basisrows}, constraint has {constraintrows}."
    )]
    ConstraintMatrixRowMismatch {
        basisrows: usize,
        constraintrows: usize,
    },

    #[error(
        "Weights dimension mismatch: expected {expected} weights to match basis matrix rows, but got {found}."
    )]
    WeightsDimensionMismatch { expected: usize, found: usize },

    #[error("QR decomposition failed while applying constraints: {0}")]
    LinalgError(#[from] FaerLinalgError),

    #[error(
        "Failed to identify a constraint nullspace basis at {site}: \
         coefficient dim {coeff_dim}, cross-rank {cross_rank}, \
         constraint Frobenius {cross_frobenius:.3e}, \
         constrained Gram spectrum {gram_spectrum}. \
         The smooth basis collapses onto the parametric block — typical causes: \
         (a) the smooth's evaluated kernel underflows after projecting out the \
         polynomial nullspace, leaving only floating-point noise (Duchon hybrid \
         in moderate-to-high d with length_scale near pairwise center distances); \
         (b) the parametric block already spans the smooth's column space \
         (over-restrictive identifiability constraint); \
         (c) the smooth has effective rank ≤ parametric-block size on this data."
    )]
    ConstraintNullspaceCollapsed {
        site: &'static str,
        cross_rank: usize,
        coeff_dim: usize,
        cross_frobenius: f64,
        /// Pre-formatted constrained-Gram spectrum summary. The structural
        /// early-return sites bail at the cross-rank check before the Gram is
        /// ever eigendecomposed, so they report `not computed` rather than a
        /// misleading NaN; only the spectral-rank-deficiency site fills in real
        /// max/min eigenvalues and tolerance.
        gram_spectrum: String,
    },

    #[error(
        "Knot vector is degenerate: all Greville abscissae are equal, so linear constraint cannot be applied."
    )]
    DegenerateKnots,

    #[error(
        "The provided knot vector is invalid: {0}. It must be non-decreasing and contain only finite values."
    )]
    InvalidKnotVector(String),

    #[error("Failed to build sparse basis matrix: {0}")]
    SparseCreation(String),

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error(
        "Indefinite penalty matrix in {context}: minimum eigenvalue {min_eigenvalue:.3e} is below tolerance {tolerance:.3e}. {guidance}"
    )]
    IndefinitePenalty {
        context: String,
        min_eigenvalue: f64,
        tolerance: f64,
        guidance: String,
    },

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("{0}")]
    DenseDerivativeMaterializationRefused(String),

    #[error(
        "Radial basis derivative is undefined at center collision (r = 0) for {kernel} \
         with dim = {dim}, m = {m}: {message}. The first/second derivative of the \
         underlying φ(r) does not have a finite limit as r → 0+, so the design-row \
         gradient and Hessian have no well-defined value at coincident points."
    )]
    DegenerateAtCollision {
        kernel: &'static str,
        dim: usize,
        m: f64,
        message: &'static str,
    },

    #[error(
        "Periodic radial basis derivative is undefined at the wrap branch cut \
         (signed displacement = ±period/2) for raw delta = {raw}, period = {period}: \
         the wrapped displacement jumps between ±period/2 and the first derivative \
         w.r.t. the input has a one-sided discontinuity. Move the evaluation point \
         off the branch cut or define a one-sided convention."
    )]
    PeriodicWrapBranchCut { raw: f64, period: f64 },

    #[error("{0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_degree_mentions_degree_in_message() {
        let err = BasisError::InvalidDegree(0);
        let msg = err.to_string();
        assert!(msg.contains("0"), "expected degree in message, got: {msg}");
        assert!(msg.to_lowercase().contains("degree"));
    }

    #[test]
    fn invalid_range_mentions_start_and_end() {
        let err = BasisError::InvalidRange(2.5, 1.0);
        let msg = err.to_string();
        assert!(
            msg.contains("2.5") || msg.contains("start"),
            "message: {msg}"
        );
    }

    #[test]
    fn degenerate_range_mentions_zero_width() {
        let err = BasisError::DegenerateRange(4);
        let msg = err.to_string().to_lowercase();
        assert!(msg.contains("zero"), "message: {msg}");
    }

    #[test]
    fn invalid_penalty_order_mentions_order_and_num_basis() {
        let err = BasisError::InvalidPenaltyOrder {
            order: 5,
            num_basis: 3,
        };
        let msg = err.to_string();
        assert!(msg.contains("5") && msg.contains("3"), "message: {msg}");
    }

    #[test]
    fn insufficient_knots_mentions_degree() {
        let err = BasisError::InsufficientKnotsForDegree {
            degree: 3,
            required: 10,
            provided: 5,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("3") && msg.contains("10") && msg.contains("5"),
            "message: {msg}"
        );
    }

    #[test]
    fn invalid_knot_vector_includes_reason() {
        let err = BasisError::InvalidKnotVector("decreasing knots".to_string());
        let msg = err.to_string();
        assert!(msg.contains("decreasing knots"), "message: {msg}");
    }

    #[test]
    fn invalid_input_passthrough() {
        let err = BasisError::InvalidInput("bad value".to_string());
        assert!(err.to_string().contains("bad value"));
    }

    #[test]
    fn other_passthrough() {
        let err = BasisError::Other("catch-all".to_string());
        assert_eq!(err.to_string(), "catch-all");
    }
}

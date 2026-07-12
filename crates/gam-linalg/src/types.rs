use serde::{Deserialize, Serialize};

/// Determinant semantics of an objective-accounted ridge.
///
/// There is deliberately no `Auto`: callers must decide whether they are
/// evaluating the exact SPD determinant or a named positive-part
/// approximation before constructing the policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RidgeDeterminantMode {
    /// Exact full log-determinant of the ridged SPD matrix.
    Full,
    /// Smooth positive-part spectral determinant approximation. This changes
    /// the estimand and is therefore explicitly named as an approximation.
    PositivePartApproximation,
}

/// Structurally valid ways a diagonal ridge may participate in a computation.
///
/// The former public boolean matrix admitted contradictory states such as a
/// quadratic penalty without the corresponding Hessian. This enum has only
/// the three coherent inhabitants used by the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RidgePolicy {
    /// Ridge is an explicit part of the exact objective: quadratic, penalty
    /// normalizer, and Laplace Hessian all include it, using a full SPD logdet.
    ExactFullObjective,
    /// Ridge participates in every objective term, but determinant evaluation
    /// is the explicitly named positive-part approximation.
    PositivePartApproximateObjective,
    /// Ridge changes only an inner linear solve and never the fitted objective,
    /// exported Hessian, determinant, covariance, or serialized model.
    SolverOnly,
}

impl RidgePolicy {
    pub const fn exact_full_objective() -> Self {
        Self::ExactFullObjective
    }

    pub const fn positive_part_approximate_objective() -> Self {
        Self::PositivePartApproximateObjective
    }

    pub const fn solver_only() -> Self {
        Self::SolverOnly
    }

    #[inline]
    pub const fn accounts_for_objective(self) -> bool {
        !matches!(self, Self::SolverOnly)
    }

    #[inline]
    pub const fn determinant_mode(self) -> RidgeDeterminantMode {
        match self {
            Self::ExactFullObjective | Self::SolverOnly => RidgeDeterminantMode::Full,
            Self::PositivePartApproximateObjective => {
                RidgeDeterminantMode::PositivePartApproximation
            }
        }
    }

    #[inline]
    pub const fn is_approximation(self) -> bool {
        matches!(self, Self::PositivePartApproximateObjective)
    }
}

#[cfg(test)]
mod ridge_policy_tests {
    use super::*;

    #[test]
    fn exact_policy_is_homogeneous_and_full() {
        let policy = RidgePolicy::exact_full_objective();
        assert!(policy.accounts_for_objective());
        assert_eq!(policy.determinant_mode(), RidgeDeterminantMode::Full);
        assert!(!policy.is_approximation());
    }

    #[test]
    fn positive_part_policy_is_explicitly_approximate() {
        let policy = RidgePolicy::positive_part_approximate_objective();
        assert!(policy.accounts_for_objective());
        assert_eq!(
            policy.determinant_mode(),
            RidgeDeterminantMode::PositivePartApproximation
        );
        assert!(policy.is_approximation());
    }

    #[test]
    fn solver_only_policy_cannot_enter_objective_accounting() {
        let policy = RidgePolicy::solver_only();
        assert!(!policy.accounts_for_objective());
        assert_eq!(policy.determinant_mode(), RidgeDeterminantMode::Full);
    }
}

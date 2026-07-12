use serde::{Deserialize, Serialize};

/// Determinant semantics of an objective-accounted ridge.
///
/// There is deliberately no `Auto`: callers must decide whether they are
/// evaluating the exact SPD determinant or a named positive-part
/// approximation before constructing the policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RidgeDeterminantMode {
    /// Exact full log-determinant of the ridged SPD matrix.
    Full,
    /// Positive-part pseudo-determinant. This changes the estimand and is
    /// therefore explicitly an approximation.
    PositivePartApproximation,
}

/// Structurally valid ways a diagonal ridge may participate in a computation.
///
/// The former public boolean matrix admitted contradictory states such as a
/// quadratic penalty without the corresponding Hessian. This enum has only
/// the three coherent inhabitants used by the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

    /// Stabilization is selected independently of smoothing parameters in all
    /// supported modes. There is no public state that can contradict this.
    #[inline]
    pub const fn is_rho_independent(self) -> bool {
        true
    }

    #[inline]
    pub const fn includes_objective(self) -> bool {
        !matches!(self, Self::SolverOnly)
    }

    #[inline]
    pub const fn includes_quadratic_penalty(self) -> bool {
        self.includes_objective()
    }

    #[inline]
    pub const fn includes_penalty_logdet(self) -> bool {
        self.includes_objective()
    }

    #[inline]
    pub const fn includes_laplace_hessian(self) -> bool {
        self.includes_objective()
    }

    #[inline]
    pub const fn determinant_mode(self) -> Option<RidgeDeterminantMode> {
        match self {
            Self::ExactFullObjective => Some(RidgeDeterminantMode::Full),
            Self::PositivePartApproximateObjective => {
                Some(RidgeDeterminantMode::PositivePartApproximation)
            }
            Self::SolverOnly => None,
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
        assert!(policy.is_rho_independent());
        assert!(policy.includes_quadratic_penalty());
        assert!(policy.includes_penalty_logdet());
        assert!(policy.includes_laplace_hessian());
        assert_eq!(policy.determinant_mode(), Some(RidgeDeterminantMode::Full));
        assert!(!policy.is_approximation());
    }

    #[test]
    fn positive_part_policy_is_explicitly_approximate() {
        let policy = RidgePolicy::positive_part_approximate_objective();
        assert!(policy.includes_objective());
        assert_eq!(
            policy.determinant_mode(),
            Some(RidgeDeterminantMode::PositivePartApproximation)
        );
        assert!(policy.is_approximation());
    }

    #[test]
    fn solver_only_policy_cannot_enter_objective_accounting() {
        let policy = RidgePolicy::solver_only();
        assert!(!policy.includes_objective());
        assert!(!policy.includes_quadratic_penalty());
        assert!(!policy.includes_penalty_logdet());
        assert!(!policy.includes_laplace_hessian());
        assert_eq!(policy.determinant_mode(), None);
    }
}

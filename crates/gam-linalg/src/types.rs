use serde::{Deserialize, Serialize};

/// Structurally valid ways a diagonal ridge may participate in a computation.
///
/// The former public boolean matrix admitted contradictory states such as a
/// quadratic penalty without the corresponding Hessian. This enum has only
/// the coherent inhabitants the engine actually selects.
///
/// # Why there is no approximate-determinant inhabitant (#2670)
///
/// A third variant, `PositivePartApproximateObjective`, used to route the ridged
/// log-determinant through a smooth positive-part spectral approximation
/// (`log|A|_reg = Σ log r_ε(σ_j)`), which is a DIFFERENT estimand from the exact
/// SPD determinant and was named as such. Nothing in production ever selected
/// it: every construction of it lived under `#[cfg(test)]`, so the only thing it
/// bought the library was a second, worse answer a user could opt into by
/// mistake. It is deleted rather than kept as a fallback — a preserved fallback
/// is a second implementation of the same quantity, and this one changed the
/// estimand while doing it.
///
/// The smooth regulariser itself is NOT deleted and was never this enum's
/// business: `spectral_regularize` / `spectral_epsilon` stay live in the REML
/// outer engine's `DenseSpectralOperator`, where a caller genuinely wants the
/// smooth surrogate together with its matching analytic gradient.
///
/// With one determinant semantics left there is no `determinant_mode()` and no
/// `RidgeDeterminantMode`: a query whose answer is a constant is not a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RidgePolicy {
    /// Ridge is an explicit part of the exact objective: quadratic, penalty
    /// normalizer, and Laplace Hessian all include it, using a full SPD logdet.
    ExactFullObjective,
    /// Ridge changes only an inner linear solve and never the fitted objective,
    /// exported Hessian, determinant, covariance, or serialized model.
    SolverOnly,
}

impl RidgePolicy {
    pub const fn exact_full_objective() -> Self {
        Self::ExactFullObjective
    }

    pub const fn solver_only() -> Self {
        Self::SolverOnly
    }

    #[inline]
    pub const fn accounts_for_objective(self) -> bool {
        !matches!(self, Self::SolverOnly)
    }
}

#[cfg(test)]
mod ridge_policy_tests {
    use super::*;

    #[test]
    fn exact_policy_accounts_for_the_objective() {
        assert!(RidgePolicy::exact_full_objective().accounts_for_objective());
    }

    #[test]
    fn solver_only_policy_cannot_enter_objective_accounting() {
        assert!(!RidgePolicy::solver_only().accounts_for_objective());
    }

    /// #2670 — the inhabitants are exactly the two the engine selects. A third
    /// would have to be a second answer to the same question, which is what the
    /// deleted positive-part variant was. The binding below is irrefutable only
    /// while that holds: re-adding a variant makes the pattern refutable, and a
    /// refutable `let` fails to compile here first.
    #[test]
    fn the_policy_has_no_third_inhabitant() {
        for policy in [
            RidgePolicy::exact_full_objective(),
            RidgePolicy::solver_only(),
        ] {
            let (RidgePolicy::ExactFullObjective | RidgePolicy::SolverOnly) = policy;
        }
    }
}

//! # Outer-objective contract (lower shared layer)
//!
//! The interface types that the `families` layer must *name, implement, and
//! return* to participate in outer smoothing-parameter optimization, hosted
//! below both `families` and `solver` so families stop importing *up* into
//! `crate::solver::rho_optimizer` (#1135).
//!
//! What lives here is exactly the **family ↔ solver contract**: the matrix-free
//! [`HessianOperator`] trait that families implement, the [`OuterEval`] result
//! they return, the [`EfsEval`] step bundle, and the capability enums
//! ([`Derivative`], [`DeclaredHessianForm`], [`HessianMaterialization`]) plus
//! GAM-specific outer-strategy errors ([`OuterStrategyError`]). The generic
//! Hessian contract and payload are owned by `opt` and re-exported here.
//!
//! What does *not* live here is the solver's *use* of the contract — the outer
//! runner, ARC/trust-region planning, seeding, caching, barrier configuration,
//! and `OuterProblem` — all of which stay in `crate::solver::rho_optimizer` and
//! depend downward on this module. `crate::solver::rho_optimizer` re-exports
//! these names so existing `crate::solver::rho_optimizer::*` paths keep working.

use ndarray::Array1;
pub use opt::{HessianMaterialization, HessianOperator, HessianValue, ObjectiveEvalError};

/// Typed error for the outer-strategy Hessian-operator surface.
///
/// All construction sites inside `outer_strategy` build one of these variants
/// instead of an ad-hoc `String`; the historical `Result<_, String>` boundary
/// at the family/solver boundary.
#[derive(Debug, Clone)]
pub enum OuterStrategyError {
    /// Shape / dimension violation of a rho-block additive Hessian update.
    RhoBlockShape { reason: String },
}

impl_reason_error_boilerplate! {
    OuterStrategyError {
        RhoBlockShape,
    }
}

/// Whether an analytic derivative is available for a given order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Derivative {
    /// Exact analytic derivative implemented and available.
    Analytic,
    /// No analytic derivative; must be approximated or skipped.
    Unavailable,
}

/// Capability-time declaration of what shape the outer Hessian takes.
/// Replaces the binary `Derivative` for the Hessian field on
/// `OuterCapability`: callers that know the shape upfront declare
/// it here, and the planner routes between dense ARC and matrix-free
/// trust-region *before* seed evaluation rather than dynamically
/// branching on `seed_eval.hessian` at runtime.
///
/// Variants:
/// - `Dense`: the family always returns `HessianValue::Dense(_)`.
///   The planner picks dense ARC; matrix-free TR is never engaged.
/// - `Operator { materialization, estimated_materialization_cost }`:
///   the family always returns `HessianValue::Operator(_)`. The
///   planner picks matrix-free TR unless `materialization` advertises
///   `Explicit`/`BatchedHvp` cheaply enough that materializing once
///   per outer iter (opt 0.4.2 `with_materialize_when_cheap`) wins.
///   `estimated_materialization_cost` is reserved for a future cost
///   model; today it is purely informational.
/// - `Either`: the family may return either shape; the runner inspects
///   the seed eval and locks the route then. This is the historical
///   default for code paths where `Derivative::Analytic` made the
///   declaration and the seed loop branched on `seed_eval.hessian`.
/// - `Unavailable`: no analytic Hessian. The planner picks BFGS / EFS
///   per the gradient declaration and the rest of the capability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeclaredHessianForm {
    Dense,
    Operator {
        materialization: HessianMaterialization,
        estimated_materialization_cost: Option<f64>,
    },
    Either,
    Unavailable,
}

impl DeclaredHessianForm {
    /// Coarse "is an analytic Hessian declared?" projection. `true`
    /// for `Dense` / `Operator` / `Either`; `false` for `Unavailable`.
    /// Used by `plan` to keep the existing `Derivative`-based match
    /// arms while richer routing decisions consult the form directly.
    pub const fn is_analytic(self) -> bool {
        !matches!(self, DeclaredHessianForm::Unavailable)
    }

    /// True when the declaration commits to a matrix-free path.
    pub const fn is_operator_only(self) -> bool {
        matches!(self, DeclaredHessianForm::Operator { .. })
    }

    /// True when the declaration commits to a dense path.
    pub const fn is_dense_only(self) -> bool {
        matches!(self, DeclaredHessianForm::Dense)
    }
}

/// Shared outer-objective result used by optimizer-facing objective
/// implementations.
pub struct OuterEval {
    pub cost: f64,
    pub gradient: Array1<f64>,
    pub hessian: HessianValue,
    /// Optional inner-solver iterate at this rho. Families whose inner solve
    /// produces a PIRLS beta populate this so the persistent-cache layer can
    /// store `(rho, beta)` together.
    pub inner_beta_hint: Option<Array1<f64>>,
}

impl OuterEval {
    /// Conventional representation of an infeasible trial point.
    pub fn infeasible(n_params: usize) -> Self {
        Self {
            cost: f64::INFINITY,
            gradient: Array1::zeros(n_params),
            hessian: HessianValue::Unavailable,
            inner_beta_hint: None,
        }
    }

    pub fn value_only(cost: f64, n_params: usize, inner_beta_hint: Option<Array1<f64>>) -> Self {
        Self {
            cost,
            gradient: Array1::zeros(n_params),
            hessian: HessianValue::Unavailable,
            inner_beta_hint,
        }
    }
}

impl Clone for OuterEval {
    fn clone(&self) -> Self {
        Self {
            cost: self.cost,
            gradient: self.gradient.clone(),
            hessian: self.hessian.clone(),
            inner_beta_hint: self.inner_beta_hint.clone(),
        }
    }
}

impl std::fmt::Debug for OuterEval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OuterEval")
            .field("cost", &self.cost)
            .field("gradient", &self.gradient)
            .field("hessian", &self.hessian)
            .finish()
    }
}

/// Result bundle returned by the EFS (extended Fellner–Schall) evaluation
/// path. Pure data: families compute the additive step and the optional
/// curvature/gradient diagnostics; the solver consumes them.
#[derive(Clone, Debug)]
pub struct EfsEval {
    /// REML/LAML cost at the current rho (for convergence monitoring and
    /// comparing candidates).
    pub cost: f64,
    /// Additive steps. Length = n_rho + n_ext_coords.
    ///
    /// For pure EFS: steps for non-penalty-like coordinates are 0.0.
    /// For hybrid EFS: ρ-coords get standard EFS multiplicative steps,
    /// ψ-coords get preconditioned gradient steps `Δψ = -α G⁺ g_ψ`.
    pub steps: Vec<f64>,
    /// Current coefficient vector β̂ from the inner P-IRLS solve.
    /// Used by the EFS loop for the runtime barrier-curvature significance
    /// check when monotonicity constraints are present.
    pub beta: Option<Array1<f64>>,
    /// Raw REML/LAML gradient restricted to the ψ block (design-moving coords).
    ///
    /// Present only when the hybrid EFS strategy is active. Used by the
    /// outer iteration for backtracking on the ψ step: if the combined
    /// (ρ-EFS, ψ-gradient) step does not decrease V(θ), the ψ step size
    /// α is halved while keeping the ρ-EFS step fixed.
    ///
    /// This avoids re-evaluating the gradient during backtracking since
    /// the gradient was already computed as part of the hybrid EFS eval.
    pub psi_gradient: Option<Array1<f64>>,
    /// Indices into the full θ vector that correspond to ψ (design-moving)
    /// coordinates. Used by the backtracking logic to selectively scale
    /// only the ψ portion of the step.
    pub psi_indices: Option<Vec<usize>>,
    /// Inner-Hessian curvature scale captured during the EFS eval, used to
    /// condition the ψ preconditioner across outer iterations.
    pub inner_hessian_scale: Option<f64>,
    /// Logdet enclosure gap diagnostic (lower/upper bound spread) captured at
    /// this EFS evaluation when the bounded-logdet path is active.
    pub logdet_enclosure_gap: Option<f64>,
    /// Number of consecutive successful inner solves that returned to the same
    /// banked incumbent after a non-monotone boundary mutation.
    ///
    /// `None` means the objective has no restored-incumbent certificate.  `Some`
    /// is reset to zero whenever the objective banks a genuinely better model.
    /// Two consecutive restorations are the minimal evidence of recurrence: one
    /// restoration can be a one-off repair, while the second establishes that
    /// changing the outer coordinate has returned to the same stationary fitted
    /// state again.  Fixed-point runners may terminate on that certificate even
    /// when the raw update keeps moving along an objective-flat parameter ridge.
    pub consecutive_restored_incumbents: Option<usize>,
}

impl EfsEval {
    pub fn with_logdet_enclosure_gap(mut self, gap: Option<f64>) -> Self {
        self.logdet_enclosure_gap = gap;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── DeclaredHessianForm ───────────────────────────────────────────────────

    #[test]
    fn declared_unavailable_is_not_analytic() {
        assert!(!DeclaredHessianForm::Unavailable.is_analytic());
    }

    #[test]
    fn declared_dense_is_analytic() {
        assert!(DeclaredHessianForm::Dense.is_analytic());
    }

    #[test]
    fn declared_either_is_analytic() {
        assert!(DeclaredHessianForm::Either.is_analytic());
    }

    #[test]
    fn declared_operator_is_analytic() {
        let form = DeclaredHessianForm::Operator {
            materialization: HessianMaterialization::Explicit,
            estimated_materialization_cost: None,
        };
        assert!(form.is_analytic());
    }

    #[test]
    fn only_operator_variant_is_operator_only() {
        let form = DeclaredHessianForm::Operator {
            materialization: HessianMaterialization::RepeatedHvp,
            estimated_materialization_cost: Some(1.0),
        };
        assert!(form.is_operator_only());
        assert!(!DeclaredHessianForm::Dense.is_operator_only());
        assert!(!DeclaredHessianForm::Either.is_operator_only());
        assert!(!DeclaredHessianForm::Unavailable.is_operator_only());
    }

    #[test]
    fn only_dense_variant_is_dense_only() {
        assert!(DeclaredHessianForm::Dense.is_dense_only());
        assert!(!DeclaredHessianForm::Either.is_dense_only());
        assert!(!DeclaredHessianForm::Unavailable.is_dense_only());
    }

    // ── OuterEval ─────────────────────────────────────────────────────────────

    #[test]
    fn infeasible_eval_has_infinity_cost() {
        let eval = OuterEval::infeasible(3);
        assert_eq!(eval.cost, f64::INFINITY);
        assert_eq!(eval.gradient.len(), 3);
    }

    #[test]
    fn value_only_eval_has_specified_cost() {
        let eval = OuterEval::value_only(42.5, 2, None);
        assert_eq!(eval.cost, 42.5);
        assert_eq!(eval.gradient.len(), 2);
        assert!(eval.inner_beta_hint.is_none());
    }
}

//! Invariants for `StabilizationLedger`.
//!
//! These tests pin the canonical taxonomy of stabilization ridges:
//!
//!   * `SolverDampingOnly`     — never enters objective/grad/Hessian/logdet.
//!   * `NumericalPerturbation` — never enters objective/grad/Hessian/logdet.
//!   * `ExplicitPrior`         — enters every accounting pass consistently
//!                               (objective up by ½ δ ‖β‖², gradient gains
//!                               δ β, Hessian gains δ I, logdet gains the
//!                               appropriate term).
//!
//! Other agents in the cross-cutting ridge cleanup (penalty_strict,
//! pirls_curvature, covariance_strict, resource_serialize, custom_family)
//! coordinate by reading these invariants, so the asserts here are the
//! stable contract the rest of the codebase keys off.
//!
//! The former matrix of public inclusion flags is now enforced statically:
//! [`RidgePolicy`] admits only coherent objective or solver-only policies,
//! while the ledger's delta accessors derive accounting participation from
//! [`StabilizationKind`]. Heterogeneous combinations are unrepresentable.


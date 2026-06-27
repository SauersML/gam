//! Central authority for outer smoothing-parameter optimization strategy.
//!
//! Every path that optimizes smoothing parameters (standard REML, link-wiggle,
//! GAMLSS custom family, spatial kappa, etc.) declares its derivative
//! capability here and receives an [`OuterPlan`] that determines which solver
//! and Hessian source to use.
//!
//! # Design invariant
//!
//! The planner never synthesizes numerical Hessians. If a path cannot provide
//! an analytic Hessian, that fact is visible in its
//! [`OuterCapability`] declaration and in the resulting [`OuterPlan`], which
//! falls back to BFGS or an EFS variant instead of synthesizing second-order
//! curvature numerically.

use gam_runtime::warm_start::{LoadSource, Session as CacheSession};

use crate::estimate::EstimationError;

use crate::estimate::reml::reml_outer_engine::BarrierConfig;

use crate::priority_selection::{
    PriorityBudgetStage, PriorityStageSummary, rank_indices_with_budget_cascade,
};

use crate::startup_stats::{
    SeedRejection, StartupStats, format_no_seeds_passed, uniform_structural_key,
};

use ::opt::{
    Arc as ArcOptimizer, ArcError, Bfgs, BfgsError, Bounds, FallbackPolicy as OptFallbackPolicy,
    FirstOrderObjective, FirstOrderSample, FixedPoint, FixedPointError, FixedPointObjective,
    FixedPointSample, FixedPointStatus, GradientTolerance, HessianFallbackPolicy,
    HessianMaterialization, HessianOperator, HessianValue, InitialMetric, MatrixFreeTrustRegion,
    MaxIterations, ObjectiveEvalError, OperatorObjective, OperatorSample, OptimizationStatus,
    OptimizerObserver, SecondOrderObjective, SecondOrderSample, Solution, StepInfo, Tolerance,
    ZerothOrderObjective,
};

use ndarray::{Array1, Array2, ArrayView2};

use std::sync::Arc;

use std::sync::Mutex;

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

mod bridges;
mod capability;
mod fd_audit; // fd-ok: FD-audit oracle module; verifies analytic gradients against FD, never in math path
mod hessian_operator;
mod objective;
mod run;
mod run_plan;
mod seed_screening;

pub(crate) use crate::model_types::CERTIFICATE_RAIL_MARGIN;
pub use crate::model_types::CriterionCertificate;
pub(crate) use bridges::*;
pub use capability::*;
// #1521 carve: the spatial-optimization driver runs the FD-audit oracle as
// `gam_solve::rho_optimizer::outer_gradient_fd_audit` and inspects its
// `OuterGradientFdAudit`/`OuterGradientFdComponent` result. Those three are the
// only fd_audit items consumed outside the module, so they are its entire
// re-export surface (a whole-module `pub(crate) use fd_audit::*` glob brought in
// nothing else and was dead). The oracle is audit-only, never on the math path.
pub use fd_audit::{OuterGradientFdAudit, OuterGradientFdComponent, outer_gradient_fd_audit}; // fd-ok: audit-only oracle, not in the math path
pub use gam_problem::{HessianResult, OuterEval};
pub use hessian_operator::*;
pub use objective::*;
pub(crate) use run::*;
// Re-export the outer-problem driver at `pub` (not just `pub(crate)`) so the
// gam-pyffi crate can construct it directly for the SAE joint-fit FFI path.
pub use run::OuterProblem;
// Re-export the outer-loop result struct at `pub` (the blanket `run` re-export
// above is `pub(crate)`) so the lifted gam-models fit-orchestration driver can
// name `gam_solve::rho_optimizer::OuterResult` (#1521).
pub use run::OuterResult;
pub(crate) use run_plan::*;
// Re-export the outer wall-clock deadline arming at `pub` (the blanket
// `run_plan` re-export above is `pub(crate)`) so the gam-pyffi SAE fit entry can
// bound its outer search the same way the in-crate survival entry does.
pub use run_plan::{
    arm_outer_wall_clock_deadline, clear_outer_wall_clock_deadline,
    outer_wall_clock_deadline_exceeded,
};
pub(crate) use seed_screening::*;

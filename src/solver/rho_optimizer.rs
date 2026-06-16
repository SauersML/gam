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

use crate::warm_start::{LoadSource, Session as CacheSession};

use crate::estimate::EstimationError;

use crate::solver::estimate::reml::reml_outer_engine::BarrierConfig;

use crate::solver::priority_selection::{
    PriorityBudgetStage, PriorityStageSummary, rank_indices_with_budget_cascade,
};

use crate::solver::startup_stats::{
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
mod fd_audit;
mod hessian_operator;
mod objective;
mod run;
mod run_plan;
mod seed_screening;

pub use crate::model_types::CriterionCertificate;
pub(crate) use crate::model_types::{CERTIFICATE_RAIL_MARGIN, CERTIFICATE_Z_GATE};
pub use crate::solver::objective_base::{HessianResult, OuterEval};
pub(crate) use bridges::*;
pub use capability::*;
pub(crate) use fd_audit::*;
pub use hessian_operator::*;
pub use objective::*;
pub(crate) use run::*;
// Re-export the outer-problem driver at `pub` (not just `pub(crate)`) so the
// gam-pyffi crate can construct it directly for the SAE joint-fit FFI path.
pub use run::OuterProblem;
pub(crate) use run_plan::*;
pub(crate) use seed_screening::*;

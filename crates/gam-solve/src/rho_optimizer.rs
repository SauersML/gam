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

use gam_runtime::warm_start::Session as CacheSession;

use crate::estimate::EstimationError;

use crate::estimate::reml::reml_outer_engine::BarrierConfig;

use crate::startup_stats::{
    SeedRejection, StartupStats, format_no_seeds_passed, uniform_structural_key,
};

use ::opt::{
    Arc as ArcOptimizer, ArcError, Bfgs, BfgsError, Bounds, FallbackPolicy as OptFallbackPolicy,
    FirstOrderObjective, FirstOrderSample, FixedPoint, FixedPointError, FixedPointObjective,
    FixedPointSample, FixedPointStatus, GradientTolerance,
    HessianMaterialization, HessianOperator, InitialMetric, LineSearchFailureReason,
    MatrixFreeTrustRegion, MaxIterations,
    ObjectiveEvalError, ObjectiveEvalKind, OperatorObjective, OperatorSample, OptimizationStatus,
    OptimizerObserver, TerminationReason,
    SecondOrderObjective, SecondOrderSample, Solution, StepInfo, Tolerance, ZerothOrderObjective,
};

use ndarray::{Array1, Array2, ArrayView2};

use std::sync::Arc;

use std::sync::Mutex;

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

pub mod asymptote_certificate;
mod bridges;
mod capability;
mod decrement_bands;
#[cfg(test)]
#[path = "rho_optimizer/efs_fallback_routing_tests.rs"]
mod efs_fallback_routing_tests;
#[cfg(test)]
#[path = "rho_optimizer/efs_step_domain_2902_tests.rs"]
mod efs_step_domain_2902_tests;
mod hessian_operator;
#[cfg(test)]
#[path = "rho_optimizer/ladder_incumbent_resume_3306_tests.rs"]
mod ladder_incumbent_resume_3306_tests;
#[cfg(test)]
#[path = "rho_optimizer/logdet_forward_error_1b_tests.rs"]
mod logdet_forward_error_1b_tests;
mod newton_polish;
mod objective;
mod outer_measurement;
mod rail;
pub mod rail_face;
#[cfg(test)]
#[path = "rho_optimizer/rail_projection_tests.rs"]
mod rail_projection_tests;
mod run;
mod run_plan;
mod inner_progress;
mod saddle_escape_latch;
pub mod zero_smoothing_face;

pub(crate) use crate::model_types::CERTIFICATE_RAIL_MARGIN;
pub use crate::model_types::{
    CriterionErrorBound, CurvatureFloorClearance, OuterCriterionCertificate,
    OuterStationarityCertificate, CertifiedRung, CurvatureEvidence, FacePositivityRoute, RailCoordinate, RailTailEvidence,
    RailedCoordinateFact,
};
pub(crate) use bridges::*;
pub use capability::*;
pub use gam_problem::{DeclaredHessianForm, Derivative, HessianValue, OuterEval};
pub(crate) use hessian_operator::*;
pub use objective::*;
pub(crate) use rail::*;
pub(crate) use run::*;
// Re-export the outer-problem driver at `pub` (not just `pub(crate)`) so the
// gam-pyffi crate can construct it directly for the SAE joint-fit FFI path.
pub use run::OuterProblem;
// Re-export the outer-loop result struct at `pub` (the blanket `run` re-export
// above is `pub(crate)`) so the lifted gam-models fit-orchestration driver can
// name `gam_solve::rho_optimizer::OuterResult` (#1521).
pub use outer_measurement::OuterFirstOrderMeasurement;
pub use run::{CertifiedOuterResult, MultistartOutcome, OuterResult, OuterResultOrigin};
// Re-export the converged-via certificate vocabulary (#2235/#2241) so callers
// that thread the termination verdict into their own payloads (gam-sae's
// SaeOuterTermination) can name the variants.
pub use run::OuterConvergedVia;
pub use run::{
    OuterStationaryPointRejection, audit_stationary_point, criterion_statistical_resolution,
    outer_value_agreement_bound,
};
pub(crate) use run_plan::*;
pub(crate) use inner_progress::*;

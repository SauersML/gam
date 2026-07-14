//! Shared state and evaluation bridge for typed reactive continuation.
//!
//! The ordinary outer optimizer starts directly at its explicit or generated
//! seed. Continuation exists only as [`crate::continuation_path::ContinuationPath`],
//! which is created after the literal exact target has returned a non-finite
//! criterion and then moves the objective's coupled rho/assignment/isometry
//! state transactionally back to that literal target.

use ndarray::Array1;

use crate::estimate::EstimationError;
use crate::inner_status::{InnerFailure, classify_inner_error};
use crate::rho_optimizer::{OuterEvalOrder, OuterObjective};
use gam_problem::OuterEval;

/// Accepted state carried between solved reactive-domain waypoints.
#[derive(Debug, Clone)]
pub(crate) struct ContinuationState {
    pub last_rho: Array1<f64>,
    pub last_eval: OuterEval,
    pub last_beta: Array1<f64>,
    pub steps_accepted: usize,
}

pub(crate) fn inner_failure_from(err: EstimationError) -> InnerFailure {
    match err {
        EstimationError::RemlOptimizationFailed(msg) => classify_inner_error(msg),
        other => InnerFailure::Other(other.to_string()),
    }
}

/// Install a non-empty coefficient hint and evaluate one exact reactive
/// waypoint. Empty hints preserve the objective-owned state established by
/// the preceding transaction.
pub(crate) fn eval_step(
    obj: &mut dyn OuterObjective,
    rho: &Array1<f64>,
    beta_seed: &Array1<f64>,
    order: OuterEvalOrder,
) -> Result<OuterEval, InnerFailure> {
    if !beta_seed.is_empty() {
        obj.seed_inner_state(beta_seed)
            .map_err(inner_failure_from)?;
    }
    obj.eval_with_order(rho, order).map_err(inner_failure_from)
}

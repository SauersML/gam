use super::outer_strategy::{
    Derivative, OuterCapability, OuterEval, OuterObjective,
};
use crate::estimate::EstimationError;
use ndarray::Array1;

/// Lightweight adapter for cost-only outer objectives.
///
/// The optimizer computes gradients via finite differences on the cost.
/// Used for auxiliary optimization problems (e.g., survival inverse-link)
/// that don't have analytic gradients.
pub(crate) struct CostOnlyObjective<F> {
    pub n_params: usize,
    pub cost_fn: F,
}

impl<F> OuterObjective for CostOnlyObjective<F>
where
    F: FnMut(&Array1<f64>) -> Result<f64, EstimationError>,
{
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::FiniteDifference,
            hessian: Derivative::Unavailable,
            n_params: self.n_params,
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
        }
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        (self.cost_fn)(rho)
    }

    fn eval(&mut self, _: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        Err(EstimationError::InvalidInput(
            "cost-only auxiliary optimization must run through the cost bridge".to_string(),
        ))
    }

    fn reset(&mut self) {}
}

use super::outer_strategy::{
    Derivative, FallbackPolicy, OuterCapability, OuterConfig, OuterEval, OuterObjective,
    OuterResult, run_outer,
};
use crate::estimate::EstimationError;
use ndarray::Array1;

/// High-level request for an auxiliary cost-only optimization problem.
///
/// Callers specify the search geometry and an initial guess. The solver layer
/// owns the derivative declaration and delegates the concrete optimizer choice
/// to `outer_strategy`.
#[derive(Clone, Debug)]
pub(crate) struct CostOnlyOptimizationRequest {
    /// Reference point used to seed the search.
    pub(crate) initial_guess: Array1<f64>,
    /// Optimizer convergence tolerance.
    pub(crate) tolerance: f64,
    /// Maximum outer iterations per seed candidate.
    pub(crate) max_iter: usize,
    /// Finite-difference step size used by the chosen solver.
    pub(crate) fd_step: f64,
    /// Optional per-coordinate lower/upper bounds for the outer coordinates.
    pub(crate) bounds: Option<(Array1<f64>, Array1<f64>)>,
    /// Seed generation and screening configuration.
    pub(crate) seed_config: crate::seeding::SeedConfig,
    /// Symmetric bound applied when `bounds` is absent.
    pub(crate) rho_bound: f64,
}

impl CostOnlyOptimizationRequest {
    pub(crate) fn new(initial_guess: Array1<f64>) -> Self {
        Self {
            initial_guess,
            tolerance: 1e-5,
            max_iter: 200,
            fd_step: 1e-4,
            bounds: None,
            seed_config: crate::seeding::SeedConfig::default(),
            rho_bound: 30.0,
        }
    }

    fn into_outer_config(self) -> OuterConfig {
        OuterConfig {
            tolerance: self.tolerance,
            max_iter: self.max_iter,
            fd_step: self.fd_step,
            bounds: self.bounds,
            seed_config: self.seed_config,
            rho_bound: self.rho_bound,
            heuristic_lambdas: Some(self.initial_guess.to_vec()),
            initial_rho: None,
            fallback_policy: FallbackPolicy::Automatic,
            screening_cap: None,
        }
    }
}

struct CostOnlyObjective<F> {
    n_params: usize,
    cost_fn: F,
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
            all_penalty_like: false,
            has_psi_coords: false,
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

pub(crate) fn optimize_cost_only<F>(
    request: CostOnlyOptimizationRequest,
    context: &str,
    cost_fn: F,
) -> Result<OuterResult, EstimationError>
where
    F: FnMut(&Array1<f64>) -> Result<f64, EstimationError>,
{
    let n_params = request.initial_guess.len();
    let outer_config = request.into_outer_config();
    let mut objective = CostOnlyObjective { n_params, cost_fn };
    run_outer(&mut objective, &outer_config, context)
}

use crate::survival::{
    MonotonicityPenalty, PenaltyBlocks, SurvivalEngineInputs, SurvivalSpec, WorkingModelSurvival,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Flattened engine inputs for Royston-Parmar likelihood evaluation.
pub struct RoystonParmarInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub weights: ArrayView1<'a, f64>,
    pub x_entry: ArrayView2<'a, f64>,
    pub x_exit: ArrayView2<'a, f64>,
    pub x_derivative: ArrayView2<'a, f64>,
}

/// Build an engine survival working model from flattened arrays.
pub fn working_model_from_flattened(
    penalties: PenaltyBlocks,
    monotonicity: MonotonicityPenalty,
    spec: SurvivalSpec,
    inputs: RoystonParmarInputs<'_>,
) -> Result<WorkingModelSurvival, crate::survival::SurvivalError> {
    WorkingModelSurvival::from_engine_inputs(
        SurvivalEngineInputs {
            age_entry: inputs.age_entry,
            age_exit: inputs.age_exit,
            event_target: inputs.event_target,
            event_competing: inputs.event_competing,
            sample_weight: inputs.weights,
            x_entry: inputs.x_entry,
            x_exit: inputs.x_exit,
            x_derivative: inputs.x_derivative,
        },
        penalties,
        monotonicity,
        spec,
    )
}

/// Compute expected Hessian directly from flattened inputs.
pub fn expected_hessian_from_flattened(
    penalties: PenaltyBlocks,
    monotonicity: MonotonicityPenalty,
    spec: SurvivalSpec,
    beta: ArrayView1<'_, f64>,
    inputs: RoystonParmarInputs<'_>,
) -> Result<Array2<f64>, crate::estimate::EstimationError> {
    let model = working_model_from_flattened(penalties, monotonicity, spec, inputs)
        .map_err(|e| crate::estimate::EstimationError::InvalidSpecification(e.to_string()))?;
    let state = model
        .update_state(&beta.to_owned())
        .map_err(|e| crate::estimate::EstimationError::InvalidSpecification(e.to_string()))?;
    Ok(state.hessian)
}

/// Options for survival smoothing-parameter optimization over `rho = log(lambda)`.
#[derive(Clone, Debug)]
pub struct SurvivalLambdaOptimizerOptions {
    pub max_iter: usize,
    pub tol: f64,
    pub finite_diff_step: f64,
    pub seed_config: crate::seeding::SeedConfig,
}

impl Default for SurvivalLambdaOptimizerOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-5,
            finite_diff_step: 1e-3,
            seed_config: crate::seeding::SeedConfig::default(),
        }
    }
}

/// Result of survival smoothing-parameter optimization.
#[derive(Clone, Debug)]
pub struct SurvivalLambdaOptimizerResult {
    pub rho: Array1<f64>,
    pub lambdas: Array1<f64>,
    pub final_value: f64,
    pub iterations: usize,
    pub final_grad_norm: f64,
    pub stationary: bool,
}

/// Optimize survival smoothing parameters via multi-start BFGS.
///
/// This wraps the engine-level optimizer with a survival-family specific contract.
///
/// The caller provides the exact survival objective and exact gradient in rho-space:
///   `(value, grad_rho) = (V(rho), dV/drho)`.
/// The optimizer in `estimate.rs` then runs multi-start BFGS in unconstrained
/// coordinates and applies the chain rule internally.
pub fn optimize_survival_lambdas_with_multistart<F>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    objective_with_gradient: F,
    options: &SurvivalLambdaOptimizerOptions,
) -> Result<SurvivalLambdaOptimizerResult, crate::estimate::EstimationError>
where
    F: FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>), crate::estimate::EstimationError>,
{
    // Default path is exact-gradient-first for survival models.
    // This avoids repeated inner re-solves required by finite differences.
    let core_opts = crate::estimate::SmoothingBfgsOptions {
        max_iter: options.max_iter,
        tol: options.tol,
        finite_diff_step: options.finite_diff_step,
        seed_config: options.seed_config,
    };
    let result = crate::estimate::optimize_log_smoothing_with_multistart_with_gradient(
        num_penalties,
        heuristic_lambdas,
        objective_with_gradient,
        &core_opts,
    )?;
    let lambdas = result.rho.mapv(f64::exp);
    Ok(SurvivalLambdaOptimizerResult {
        rho: result.rho,
        lambdas,
        final_value: result.final_value,
        iterations: result.iterations,
        final_grad_norm: result.final_grad_norm,
        stationary: result.stationary,
    })
}

/// Explicit finite-difference fallback for callers that only expose an
/// objective value `V(rho)` and not an exact rho-gradient.
pub fn optimize_survival_lambdas_with_multistart_fd<F>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    objective: F,
    options: &SurvivalLambdaOptimizerOptions,
) -> Result<SurvivalLambdaOptimizerResult, crate::estimate::EstimationError>
where
    F: Fn(&Array1<f64>) -> Result<f64, crate::estimate::EstimationError>,
{
    // Explicit fallback path for callers that only provide V(rho).
    // Gradient is approximated numerically in estimate.rs.
    let core_opts = crate::estimate::SmoothingBfgsOptions {
        max_iter: options.max_iter,
        tol: options.tol,
        finite_diff_step: options.finite_diff_step,
        seed_config: options.seed_config,
    };
    let result = crate::estimate::optimize_log_smoothing_with_multistart(
        num_penalties,
        heuristic_lambdas,
        objective,
        &core_opts,
    )?;
    let lambdas = result.rho.mapv(f64::exp);
    Ok(SurvivalLambdaOptimizerResult {
        rho: result.rho,
        lambdas,
        final_value: result.final_value,
        iterations: result.iterations,
        final_grad_norm: result.final_grad_norm,
        stationary: result.stationary,
    })
}

/// Backward-compatible alias for explicit-gradient survival optimization.
pub fn optimize_survival_lambdas_with_multistart_with_gradient<F>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    objective_with_gradient: F,
    options: &SurvivalLambdaOptimizerOptions,
) -> Result<SurvivalLambdaOptimizerResult, crate::estimate::EstimationError>
where
    F: FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>), crate::estimate::EstimationError>,
{
    optimize_survival_lambdas_with_multistart(
        num_penalties,
        heuristic_lambdas,
        objective_with_gradient,
        options,
    )
}

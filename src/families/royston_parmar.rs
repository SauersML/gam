use crate::survival::{
    MonotonicityPenalty, PenaltyBlocks, SurvivalEngineInputs, SurvivalSpec, WorkingModelSurvival,
};
use ndarray::{Array2, ArrayView1, ArrayView2};

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

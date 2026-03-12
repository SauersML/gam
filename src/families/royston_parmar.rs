use crate::survival::{
    MonotonicityPenalty, PenaltyBlocks, SurvivalBaselineOffsets, SurvivalEngineInputs,
    SurvivalSpec, SurvivalTimeCovarInputs, WorkingModelSurvival,
};
use ndarray::{ArrayView1, ArrayView2};

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
    pub eta_offset_entry: Option<ArrayView1<'a, f64>>,
    pub eta_offset_exit: Option<ArrayView1<'a, f64>>,
    pub derivative_offset_exit: Option<ArrayView1<'a, f64>>,
}

pub struct RoystonParmarSharedTimeCovariateInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub weights: ArrayView1<'a, f64>,
    pub time_entry: ArrayView2<'a, f64>,
    pub time_exit: ArrayView2<'a, f64>,
    pub time_derivative: ArrayView2<'a, f64>,
    pub covariates: ArrayView2<'a, f64>,
    pub eta_offset_entry: Option<ArrayView1<'a, f64>>,
    pub eta_offset_exit: Option<ArrayView1<'a, f64>>,
    pub derivative_offset_exit: Option<ArrayView1<'a, f64>>,
}

/// Build an engine survival working model from flattened arrays.
pub fn working_model_from_flattened(
    penalties: PenaltyBlocks,
    monotonicity: MonotonicityPenalty,
    spec: SurvivalSpec,
    inputs: RoystonParmarInputs<'_>,
) -> Result<WorkingModelSurvival, crate::survival::SurvivalError> {
    let offsets = match (
        inputs.eta_offset_entry,
        inputs.eta_offset_exit,
        inputs.derivative_offset_exit,
    ) {
        (Some(eta_entry), Some(eta_exit), Some(derivative_exit)) => Some(SurvivalBaselineOffsets {
            eta_entry,
            eta_exit,
            derivative_exit,
        }),
        (None, None, None) => None,
        _ => {
            return Err(crate::survival::SurvivalError::DimensionMismatch);
        }
    };

    WorkingModelSurvival::from_engine_inputswith_offsets(
        SurvivalEngineInputs {
            age_entry: inputs.age_entry,
            age_exit: inputs.age_exit,
            event_target: inputs.event_target,
            event_competing: inputs.event_competing,
            sampleweight: inputs.weights,
            x_entry: inputs.x_entry,
            x_exit: inputs.x_exit,
            x_derivative: inputs.x_derivative,
        },
        offsets,
        penalties,
        monotonicity,
        spec,
    )
}

pub fn working_model_from_time_covariateshared(
    penalties: PenaltyBlocks,
    monotonicity: MonotonicityPenalty,
    spec: SurvivalSpec,
    inputs: RoystonParmarSharedTimeCovariateInputs<'_>,
) -> Result<WorkingModelSurvival, crate::survival::SurvivalError> {
    let offsets = match (
        inputs.eta_offset_entry,
        inputs.eta_offset_exit,
        inputs.derivative_offset_exit,
    ) {
        (Some(eta_entry), Some(eta_exit), Some(derivative_exit)) => Some(SurvivalBaselineOffsets {
            eta_entry,
            eta_exit,
            derivative_exit,
        }),
        (None, None, None) => None,
        _ => {
            return Err(crate::survival::SurvivalError::DimensionMismatch);
        }
    };
    WorkingModelSurvival::from_time_covariate_inputswith_offsets(
        SurvivalTimeCovarInputs {
            age_entry: inputs.age_entry,
            age_exit: inputs.age_exit,
            event_target: inputs.event_target,
            event_competing: inputs.event_competing,
            sampleweight: inputs.weights,
            time_entry: inputs.time_entry,
            time_exit: inputs.time_exit,
            time_derivative: inputs.time_derivative,
            covariates: inputs.covariates,
        },
        offsets,
        penalties,
        monotonicity,
        spec,
    )
}



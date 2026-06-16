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
    pub monotonicity_constraint_rows: Option<ArrayView2<'a, f64>>,
    pub monotonicity_constraint_offsets: Option<ArrayView1<'a, f64>>,
    pub eta_offset_entry: Option<ArrayView1<'a, f64>>,
    pub eta_offset_exit: Option<ArrayView1<'a, f64>>,
    pub derivative_offset_exit: Option<ArrayView1<'a, f64>>,
}

/// Per-row inputs for a Royston–Parmar survival model whose log-cumulative-hazard
/// design factorises into a shared time block `time_*` and a row-aligned
/// covariate block `covariates`.  The final design is `[time | covariates]`
/// (column-wise concatenation per row).  `age_entry/age_exit` carry the
/// delayed-entry/exit interval; `event_target` is the binary primary-event
/// indicator and `event_competing` the competing-cause indicator.  The
/// `monotonicity_constraint_*` views describe the linear constraint
/// `A·β + offset ≥ 0` enforcing dη/da ≥ 0; the `*_offset_*` views carry the
/// fixed (non-coefficient) addends to η at entry, η at exit, and dη/da at
/// exit.  All views are zero-copy borrows of caller-owned storage.
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
    pub monotonicity_constraint_rows: Option<ArrayView2<'a, f64>>,
    pub monotonicity_constraint_offsets: Option<ArrayView1<'a, f64>>,
    pub eta_offset_entry: Option<ArrayView1<'a, f64>>,
    pub eta_offset_exit: Option<ArrayView1<'a, f64>>,
    pub derivative_offset_exit: Option<ArrayView1<'a, f64>>,
}

fn survival_baseline_offsets<'a>(
    eta_offset_entry: Option<ArrayView1<'a, f64>>,
    eta_offset_exit: Option<ArrayView1<'a, f64>>,
    derivative_offset_exit: Option<ArrayView1<'a, f64>>,
) -> Result<Option<SurvivalBaselineOffsets<'a>>, crate::survival::SurvivalError> {
    match (eta_offset_entry, eta_offset_exit, derivative_offset_exit) {
        (Some(eta_entry), Some(eta_exit), Some(derivative_exit)) => {
            Ok(Some(SurvivalBaselineOffsets {
                eta_entry,
                eta_exit,
                derivative_exit,
            }))
        }
        (None, None, None) => Ok(None),
        _ => Err(crate::survival::SurvivalError::DimensionMismatch),
    }
}

/// Build an engine survival working model from flattened arrays.
pub fn working_model_from_flattened(
    penalties: PenaltyBlocks,
    monotonicity: MonotonicityPenalty,
    spec: SurvivalSpec,
    inputs: RoystonParmarInputs<'_>,
) -> Result<WorkingModelSurvival, crate::survival::SurvivalError> {
    let offsets = survival_baseline_offsets(
        inputs.eta_offset_entry,
        inputs.eta_offset_exit,
        inputs.derivative_offset_exit,
    )?;

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
            monotonicity_constraint_rows: inputs.monotonicity_constraint_rows,
            monotonicity_constraint_offsets: inputs.monotonicity_constraint_offsets,
        },
        offsets,
        penalties,
        monotonicity,
        spec,
    )
}

/// Build a Royston–Parmar `WorkingModelSurvival` from a shared time block
/// and a row-aligned covariate block.  The composite design η = X β
/// concatenates `[time_entry|covariates]` (and analogously for exit and
/// derivative); penalties apply only to the time block, the covariate
/// columns being unpenalized.  `monotonicity` records the dη/da ≥ 0
/// linear-inequality penalty.  Returns the assembled working model that
/// the engine drives during fitting; errors are `SurvivalError` cases
/// (`DimensionMismatch` if the offset views are partially present).
pub fn working_model_from_time_covariateshared(
    penalties: PenaltyBlocks,
    monotonicity: MonotonicityPenalty,
    spec: SurvivalSpec,
    inputs: RoystonParmarSharedTimeCovariateInputs<'_>,
) -> Result<WorkingModelSurvival, crate::survival::SurvivalError> {
    let offsets = survival_baseline_offsets(
        inputs.eta_offset_entry,
        inputs.eta_offset_exit,
        inputs.derivative_offset_exit,
    )?;
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
            monotonicity_constraint_rows: inputs.monotonicity_constraint_rows,
            monotonicity_constraint_offsets: inputs.monotonicity_constraint_offsets,
        },
        offsets,
        penalties,
        monotonicity,
        spec,
    )
}

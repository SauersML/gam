use crate::custom_family::{
    BlockWorkingSet, CustomFamily, FamilyEvaluation, ParameterBlockState,
    projected_linear_constraint_stationarity_vector,
};
use crate::model_types::EstimationError;
use gam_linalg::faer_ndarray::{fast_atv, fast_av, fast_xt_diag_x, fast_xt_diag_y};
use gam_linalg::matrix::SymmetricMatrix;
use gam_problem::{Coefficients, LinearPredictor};
use gam_row_macros::row_atom;
use gam_solve::pirls::{
    ConstraintSet, LinearInequalityConstraints, WorkingModel as PirlsWorkingModel, WorkingState,
    array1_l2_norm,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, Axis};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::Range;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SurvivalError {
    #[error("input dimensions are inconsistent")]
    DimensionMismatch,
    #[error("inputs contain non-finite values")]
    NonFiniteInput,
    #[error("survival spec '{0}' is not supported by the one-hazard survival engine")]
    UnsupportedSpec(&'static str),
    #[error("crude risk integration setup is invalid")]
    InvalidIntegrationSetup,
    #[error("survival time grid must be finite, non-negative, and strictly increasing")]
    InvalidTimeGrid,
    #[error("cumulative hazard must be nondecreasing")]
    NonMonotoneCumulativeHazard,
    #[error("instantaneous hazard must stay strictly positive during integration")]
    NonPositiveHazard,
    #[error("{reason}")]
    InvalidInput { reason: String },
    #[error("{reason}")]
    CauseSpecificDimensionMismatch { reason: String },
    #[error("{reason}")]
    NumericalFailure { reason: String },
    #[error("{reason}")]
    EventCodeInvalid { reason: String },
    #[error("cause-specific survival block {block}: {source}")]
    CauseSpecificBlock {
        block: usize,
        #[source]
        source: Box<SurvivalError>,
    },
}

impl From<SurvivalError> for String {
    fn from(err: SurvivalError) -> Self {
        err.to_string()
    }
}

impl From<crate::block_layout::block_count::BlockCountMismatch> for SurvivalError {
    fn from(err: crate::block_layout::block_count::BlockCountMismatch) -> SurvivalError {
        SurvivalError::CauseSpecificDimensionMismatch {
            reason: err.message(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum SurvivalSpec {
    #[default]
    Net,
    Crude,
}

#[derive(Debug, Clone)]
pub struct SurvivalEngineInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub sampleweight: ArrayView1<'a, f64>,
    pub x_entry: ArrayView2<'a, f64>,
    pub x_exit: ArrayView2<'a, f64>,
    pub x_derivative: ArrayView2<'a, f64>,
    /// Optional global monotonicity collocation rows for the full coefficient vector.
    /// Non-structural survival models should pass these explicitly instead of
    /// relying on observed derivative rows.
    pub monotonicity_constraint_rows: Option<ArrayView2<'a, f64>>,
    /// Baseline offsets corresponding to `monotonicity_constraint_rows`.
    pub monotonicity_constraint_offsets: Option<ArrayView1<'a, f64>>,
}

#[derive(Debug, Clone)]
pub struct SurvivalTimeCovarInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub sampleweight: ArrayView1<'a, f64>,
    pub time_entry: ArrayView2<'a, f64>,
    pub time_exit: ArrayView2<'a, f64>,
    pub time_derivative: ArrayView2<'a, f64>,
    pub covariates: ArrayView2<'a, f64>,
    /// Optional global monotonicity collocation rows for the full coefficient vector.
    /// Non-structural survival models should pass these explicitly instead of
    /// relying on observed derivative rows.
    pub monotonicity_constraint_rows: Option<ArrayView2<'a, f64>>,
    /// Baseline offsets corresponding to `monotonicity_constraint_rows`.
    pub monotonicity_constraint_offsets: Option<ArrayView1<'a, f64>>,
}

#[derive(Debug, Clone)]
pub struct SurvivalBaselineOffsets<'a> {
    /// Baseline target contribution to eta at entry time: eta_target(t_entry).
    pub eta_entry: ArrayView1<'a, f64>,
    /// Baseline target contribution to eta at exit time: eta_target(t_exit).
    pub eta_exit: ArrayView1<'a, f64>,
    /// Baseline target contribution to d eta / d t at exit: eta_target'(t_exit).
    ///
    /// This is used in event terms where log-hazard requires
    /// log(d eta / d t). By threading this as an explicit offset, we get
    /// "parametric default + spline deviation" behavior:
    /// - strong penalty => deviation ~ 0 => model collapses to baseline target,
    /// - weak penalty   => deviation can bend away where data supports it.
    pub derivative_exit: ArrayView1<'a, f64>,
}

#[derive(Debug, Clone)]
pub struct PenaltyBlock {
    pub matrix: Array2<f64>,
    pub lambda: f64,
    pub range: Range<usize>,
    /// Structural nullspace dimension of this penalty matrix.
    /// Used for exact pseudo-logdet computation. 0 means full rank.
    pub nullspace_dim: usize,
}

#[derive(Debug, Clone)]
pub struct PenaltyBlocks {
    pub blocks: Vec<PenaltyBlock>,
}

impl PenaltyBlocks {
    pub fn new(blocks: Vec<PenaltyBlock>) -> Self {
        Self { blocks }
    }

    pub fn gradient(&self, beta: &Array1<f64>) -> Array1<f64> {
        let mut grad = Array1::zeros(beta.len());
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let b = beta.slice(ndarray::s![block.range.clone()]);
            let g = block.matrix.dot(&b);
            let mut dst = grad.slice_mut(ndarray::s![block.range.clone()]);
            dst += &(block.lambda * g);
        }
        grad
    }

    pub fn hessian(&self, dim: usize) -> Array2<f64> {
        let mut h = Array2::zeros((dim, dim));
        self.addhessian_inplace(&mut h);
        h
    }

    pub fn deviance(&self, beta: &Array1<f64>) -> f64 {
        let mut value = 0.0;
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let b = beta.slice(ndarray::s![block.range.clone()]);
            value += 0.5 * block.lambda * b.dot(&block.matrix.dot(&b));
        }
        value
    }

    pub fn addhessian_inplace(&self, h: &mut Array2<f64>) {
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let start = block.range.start;
            let end = block.range.end;
            h.slice_mut(ndarray::s![start..end, start..end])
                .scaled_add(block.lambda, &block.matrix);
        }
    }
}

/// Entry ages at or below this value are treated as left-truncation at the time
/// origin, i.e. "no delayed-entry interval" — the cumulative-hazard term
/// `exp(η_entry)` is dropped because `H(0) = 0`. The Royston-Parmar baseline is
/// `η(t) = log H(t)` with `H(t) → 0` as `t → 0`, so `log H` diverges at the
/// origin; this small positive floor lets a row that genuinely enters at time
/// zero skip the entry contribution instead of evaluating `log H` at a
/// degenerate point. Shared so every entry-detection site stays in lockstep.
///
/// Public so the fit-orchestration layer can classify a dataset as genuinely
/// left-truncated (`entry > threshold`) with the SAME origin convention the
/// likelihood engines use, and pick the left-truncation-robust time anchor
/// accordingly (issue #1790).
pub const ENTRY_AT_ORIGIN_THRESHOLD: f64 = 1e-8;

/// Fraction-to-the-boundary factor for the cause-specific feasible-step search.
/// When a Newton direction would drive a row's derivative down to the
/// monotonicity floor, the step is capped at this fraction of the distance to
/// the boundary rather than landing exactly on it. Staying strictly inside the
/// feasible region (the standard interior-point fraction-to-boundary rule)
/// keeps the next `1/deriv` / `deriv.ln()` evaluation away from the singular
/// boundary where curvature blows up.
const DERIVATIVE_FRACTION_TO_BOUNDARY: f64 = 0.995;

/// Relative projected-KKT accuracy required before evaluating survival LAML.
///
/// LAML is an envelope at the fitted inner mode. A looser or decrement-only
/// certificate is not sufficient: differentiating the criterion away from the
/// mode requires higher-order residual-response terms that the survival family
/// deliberately does not approximate.
pub(crate) const SURVIVAL_LAML_STATIONARITY_RELATIVE_TOL: f64 = 1.0e-8;

#[derive(Debug, Clone)]
pub struct CauseSpecificRoystonParmarBlock {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,
    pub sampleweight: Array1<f64>,
    pub x_entry: Array2<f64>,
    pub x_exit: Array2<f64>,
    pub x_derivative: Array2<f64>,
    pub offset_eta_entry: Array1<f64>,
    pub offset_eta_exit: Array1<f64>,
    pub offset_derivative_exit: Array1<f64>,
    pub derivative_floor: f64,
    /// Number of leading columns that are structural monotone-I-spline time
    /// columns (`0` for a non-structural block, e.g. a parametric Weibull
    /// `log t` baseline). When `> 0`, `block_linear_constraints` emits the
    /// coefficient cone `β_j ≥ 0` for `j in 0..structural_time_columns` in
    /// addition to the per-row derivative guard. Each such column is a monotone
    /// non-decreasing I-spline basis (its M-spline derivative is non-negative
    /// everywhere), so the cone is the exact DOMAIN-WIDE monotonicity
    /// certificate — the per-row guard only pins `q'(t_i) ≥ floor` at the
    /// training rows and leaves tail columns (M-spline support beyond the
    /// largest training exit time, ≈0 at every training row) free to go
    /// negative, producing a non-monotone `q'(t)` at prediction horizons the
    /// Royston-Parmar predictor then refuses. Covariate columns
    /// (`structural_time_columns..p`) are excluded — their coefficients carry
    /// covariate effects and legitimately take any sign; the constant column is
    /// already dropped by the I-spline `keep_cols` at construction, so every
    /// column in `0..structural_time_columns` is a genuine shape column.
    pub structural_time_columns: usize,
}

/// Cause-specific competing-risks survival as a blockwise custom family.
///
/// Each cause is represented by one `ParameterBlockState`, so endpoint-specific
/// coefficients, shared smoothing labels, and user-defined coefficient groups
/// stay on the existing `CustomFamily` / `BlockwiseFitOptions` joint-fit path.
#[derive(Debug, Clone)]
pub struct CauseSpecificRoystonParmarFamily {
    blocks: Vec<CauseSpecificRoystonParmarBlock>,
}

impl CauseSpecificRoystonParmarFamily {
    pub fn new(blocks: Vec<CauseSpecificRoystonParmarBlock>) -> Result<Self, String> {
        if blocks.is_empty() {
            return Err(SurvivalError::InvalidInput {
                reason: "cause-specific survival family requires at least one endpoint".to_string(),
            }
            .into());
        }
        for (idx, block) in blocks.iter().enumerate() {
            validate_cause_specific_block(block).map_err(|err| {
                SurvivalError::CauseSpecificBlock {
                    block: idx + 1,
                    source: Box::new(err),
                }
                .to_string()
            })?;
        }
        Ok(Self { blocks })
    }

}

fn validate_cause_specific_block(
    block: &CauseSpecificRoystonParmarBlock,
) -> Result<(), SurvivalError> {
    let n = block.event_target.len();
    let p = block.x_exit.ncols();
    if n == 0 || p == 0 {
        bail_invalid_surv!("empty event vector or coefficient block");
    }
    if block.age_entry.len() != n
        || block.age_exit.len() != n
        || block.sampleweight.len() != n
        || block.x_entry.nrows() != n
        || block.x_exit.nrows() != n
        || block.x_derivative.nrows() != n
        || block.x_entry.ncols() != p
        || block.x_derivative.ncols() != p
        || block.offset_eta_entry.len() != n
        || block.offset_eta_exit.len() != n
        || block.offset_derivative_exit.len() != n
    {
        return Err(SurvivalError::CauseSpecificDimensionMismatch {
            reason: "dimension mismatch".to_string(),
        });
    }
    // A cause-specific block's `event_target` is the binary cause-k indicator
    // produced by `cause_specific_event_indicator`; a label > 1 here means the
    // caller passed raw multi-cause codes instead of projecting per cause. That
    // is a valid finite label, not non-finite input, so it gets its own clear
    // error rather than the misleading "non-finite input".
    if let Some(&label) = block.event_target.iter().find(|&&v| v > 1) {
        return Err(SurvivalError::EventCodeInvalid {
            reason: format!(
                "cause-specific block event_target must be the binary cause indicator {{0, 1}}, got multi-cause label {label}; project raw codes per cause via cause_specific_event_indicator"
            ),
        });
    }
    if block.age_entry.iter().any(|v| !v.is_finite())
        || block.age_exit.iter().any(|v| !v.is_finite())
        || block
            .sampleweight
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
        || block.x_entry.iter().any(|v| !v.is_finite())
        || block.x_exit.iter().any(|v| !v.is_finite())
        || block.x_derivative.iter().any(|v| !v.is_finite())
        || block.offset_eta_entry.iter().any(|v| !v.is_finite())
        || block.offset_eta_exit.iter().any(|v| !v.is_finite())
        || block.offset_derivative_exit.iter().any(|v| !v.is_finite())
        || !block.derivative_floor.is_finite()
        || block.derivative_floor < 0.0
    {
        bail_invalid_surv!("non-finite input");
    }
    Ok(())
}

row_atom! {
    fn cause_specific_row [generic, order2, third, fourth](
        eta_exit,
        eta_entry,
        derivative;
        weight: scale,
        entry_active: bool,
        event: bool
    ) {
        weight
            * (exp(eta_exit)
                - entry_active * exp(eta_entry)
                - event * (eta_exit + ln(derivative)))
    }
}

/// Frozen local coordinates for exact saved-model ALO replay of one
/// cause-specific transformation/Weibull survival row.
pub struct CauseSpecificSurvivalAloRowInput {
    pub eta_exit: f64,
    pub eta_entry: f64,
    pub derivative_exit: f64,
    pub prior_weight: f64,
    pub entry_active: bool,
    pub event: bool,
}

/// Negative-log-likelihood geometry in local coordinates
/// `[eta_exit, eta_entry, derivative_exit]`.
#[derive(Clone, Debug, PartialEq)]
pub struct CauseSpecificSurvivalAloRowGeometry {
    pub negative_log_likelihood: f64,
    pub nll_score: [f64; 3],
    pub observed_hessian: [[f64; 3]; 3],
}

/// Evaluate the exact order-two row atom shared with survival fitting.
///
/// Inactive entry/event channels are canonicalised before evaluation so their
/// score and curvature are exactly zero and cannot form `0 * overflow` or
/// `0 * log(nonpositive)` intermediates.
pub fn cause_specific_survival_alo_row_geometry(
    input: CauseSpecificSurvivalAloRowInput,
) -> Result<CauseSpecificSurvivalAloRowGeometry, String> {
    if !input.prior_weight.is_finite() || input.prior_weight < 0.0 {
        return Err(format!(
            "cause-specific saved ALO prior weight must be finite and non-negative, got {}",
            input.prior_weight
        ));
    }
    // The live fit drops zero-weight rows before it evaluates any predictor
    // channel. Saved replay must preserve that exact measure: a structurally
    // invalid derivative or overflowing inactive predictor on a row carrying
    // no likelihood mass is irrelevant, and must not turn an exact zero
    // score/curvature row into a diagnostic failure.
    if input.prior_weight == 0.0 {
        return Ok(CauseSpecificSurvivalAloRowGeometry {
            negative_log_likelihood: 0.0,
            nll_score: [0.0; 3],
            observed_hessian: [[0.0; 3]; 3],
        });
    }
    if !input.eta_exit.is_finite() {
        return Err(format!(
            "cause-specific saved ALO exit index must be finite, got {}",
            input.eta_exit
        ));
    }
    let eta_entry = if input.entry_active {
        if !input.eta_entry.is_finite() {
            return Err(format!(
                "cause-specific saved ALO active entry index must be finite, got {}",
                input.eta_entry
            ));
        }
        input.eta_entry
    } else {
        0.0
    };
    let derivative_exit = if input.event {
        if !input.derivative_exit.is_finite() || input.derivative_exit <= 0.0 {
            return Err(format!(
                "cause-specific saved ALO event derivative must be positive and finite, got {}",
                input.derivative_exit
            ));
        }
        input.derivative_exit
    } else {
        1.0
    };
    let atom = cause_specific_row_order2(
        input.eta_exit,
        eta_entry,
        derivative_exit,
        input.prior_weight,
        input.entry_active,
        input.event,
    );
    let gradient = atom.gradient();
    let observed_hessian =
        std::array::from_fn(|row| std::array::from_fn(|column| atom.hessian_at(row, column)));
    if !atom.value().is_finite()
        || gradient.iter().any(|value| !value.is_finite())
        || observed_hessian
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(format!(
            "cause-specific saved ALO row geometry is non-finite: nll={}, score={gradient:?}, hessian={observed_hessian:?}",
            atom.value(),
        ));
    }
    Ok(CauseSpecificSurvivalAloRowGeometry {
        negative_log_likelihood: atom.value(),
        nll_score: gradient,
        observed_hessian,
    })
}

#[derive(Clone, Copy)]
struct CauseSpecificAtomInput {
    primary: [f64; 3],
    weight: f64,
    entry_active: bool,
    event: bool,
}

/// Production [`gam_math::jet_tower::RowProgram`] for one cause-specific
/// Royston-Parmar row.
///
/// The generic evaluator and the live specialized order-2/third/fourth
/// lowerings are all emitted from `cause_specific_row`. Predictor values are
/// ordered as exit index, entry index, and positive spline derivative; entry
/// and event activity remain row constants.
pub struct CauseSpecificRowProgram {
    primary: [f64; 3],
    weight: f64,
    entry_active: bool,
    event: bool,
}

impl CauseSpecificRowProgram {
    /// Construct one canonical predictor-space row program.
    pub fn new(primary: [f64; 3], weight: f64, entry_active: bool, event: bool) -> Self {
        Self {
            primary,
            weight,
            entry_active,
            event,
        }
    }

    fn require_row(row: usize) -> Result<(), String> {
        if row != 0 {
            return Err(format!(
                "CauseSpecificRowProgram holds exactly one row; got row {row}"
            ));
        }
        Ok(())
    }
}

impl gam_math::jet_tower::RowProgram<3> for CauseSpecificRowProgram {
    fn n_rows(&self) -> usize {
        1
    }

    fn primaries(&self, row: usize) -> Result<[f64; 3], String> {
        Self::require_row(row)?;
        Ok(self.primary)
    }

    fn eval<S: gam_math::jet_scalar::JetScalar<3>>(
        &self,
        row: usize,
        p: &[S; 3],
    ) -> Result<S, String> {
        Self::require_row(row)?;
        Ok(cause_specific_row(
            &p[0],
            &p[1],
            &p[2],
            self.weight,
            self.entry_active,
            self.event,
        ))
    }
}

/// Validate one live row and canonicalise inactive entry/event primaries to
/// finite neutral values. The symbolic atom still receives the activity flags,
/// so every derivative of an inactive term is exactly zero without evaluating
/// `0 * exp(overflow)` or `0 * ln(nonpositive)`.
fn cause_specific_atom_input(
    block: &CauseSpecificRoystonParmarBlock,
    row: usize,
    eta_entry: f64,
    eta_exit: f64,
    derivative: f64,
) -> Result<Option<CauseSpecificAtomInput>, SurvivalError> {
    let weight = block.sampleweight[row];
    if weight <= 0.0 {
        return Ok(None);
    }
    if block.age_exit[row] < block.age_entry[row] {
        bail_invalid_surv!("age_exit < age_entry at row {row}");
    }
    let entry_active = block.age_entry[row] > ENTRY_AT_ORIGIN_THRESHOLD;
    let event = block.event_target[row] > 0;
    let eta_entry = if entry_active { eta_entry } else { 0.0 };
    let derivative = if event {
        if !(derivative.is_finite() && derivative > 0.0) {
            return Err(SurvivalError::NumericalFailure {
                reason: format!(
                    "cause-specific survival derivative must be positive at row {row}, got {derivative}"
                ),
            });
        }
        derivative
    } else {
        1.0
    };
    let h_exit = eta_exit.exp();
    let h_entry = eta_entry.exp();
    if !(h_exit.is_finite() && h_entry.is_finite()) {
        return Err(SurvivalError::NumericalFailure {
            reason: format!("non-finite cumulative hazard at row {row}"),
        });
    }
    Ok(Some(CauseSpecificAtomInput {
        primary: [eta_exit, eta_entry, derivative],
        weight,
        entry_active,
        event,
    }))
}

const CAUSE_SPECIFIC_PRIMARY_PAIRS: [(usize, usize); 6] =
    [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];

/// Pull all six primary-Hessian channels through the three coefficient-design
/// Jacobians. Current cross channels are symbolically zero, so their matrix
/// products are skipped; if the row expression later gains a coupling, that
/// generated channel becomes live automatically rather than being dropped by a
/// hand-maintained diagonal assumption.
fn cause_specific_pullback_hessian(
    block: &CauseSpecificRoystonParmarBlock,
    weights: &[Array1<f64>; 6],
) -> Array2<f64> {
    let designs = [&block.x_exit, &block.x_entry, &block.x_derivative];
    let p = block.x_exit.ncols();
    let mut hessian = Array2::<f64>::zeros((p, p));
    for (slot, &(left, right)) in CAUSE_SPECIFIC_PRIMARY_PAIRS.iter().enumerate() {
        let channel = &weights[slot];
        if channel.iter().all(|&value| value == 0.0) {
            continue;
        }
        if left == right {
            hessian += &fast_xt_diag_x(designs[left], channel);
        } else {
            let cross = fast_xt_diag_y(designs[left], channel, designs[right]);
            hessian += &cross;
            hessian += &cross.t();
        }
    }
    hessian
}

fn evaluate_cause_specific_block(
    block: &CauseSpecificRoystonParmarBlock,
    beta: &Array1<f64>,
) -> Result<(f64, Array1<f64>, Array2<f64>), SurvivalError> {
    let n = block.event_target.len();
    let p = block.x_exit.ncols();
    if beta.len() != p {
        return Err(SurvivalError::CauseSpecificDimensionMismatch {
            reason: format!("beta length mismatch: got {}, expected {p}", beta.len()),
        });
    }
    let eta_entry = fast_av(&block.x_entry, beta) + &block.offset_eta_entry;
    let eta_exit = fast_av(&block.x_exit, beta) + &block.offset_eta_exit;
    let derivative = fast_av(&block.x_derivative, beta) + &block.offset_derivative_exit;
    let mut log_likelihood = 0.0;
    let mut gradient_weights: [Array1<f64>; 3] = std::array::from_fn(|_| Array1::<f64>::zeros(n));
    let mut hessian_weights: [Array1<f64>; 6] = std::array::from_fn(|_| Array1::<f64>::zeros(n));

    for i in 0..n {
        let Some(input) =
            cause_specific_atom_input(block, i, eta_entry[i], eta_exit[i], derivative[i])?
        else {
            continue;
        };
        let atom = cause_specific_row_order2(
            input.primary[0],
            input.primary[1],
            input.primary[2],
            input.weight,
            input.entry_active,
            input.event,
        );
        log_likelihood -= atom.value();
        let gradient = atom.gradient();
        for axis in 0..3 {
            gradient_weights[axis][i] = -gradient[axis];
        }
        for (slot, &(left, right)) in CAUSE_SPECIFIC_PRIMARY_PAIRS.iter().enumerate() {
            hessian_weights[slot][i] = atom.hessian_at(left, right);
        }
    }

    let designs = [&block.x_exit, &block.x_entry, &block.x_derivative];
    let mut gradient = Array1::<f64>::zeros(p);
    for axis in 0..3 {
        gradient += &fast_atv(designs[axis], &gradient_weights[axis]);
    }
    let hessian = cause_specific_pullback_hessian(block, &hessian_weights);
    Ok((log_likelihood, gradient, hessian))
}

impl CustomFamily for CauseSpecificRoystonParmarFamily {
    // Preserve the pre-gam#1395 behavior: the trait default flipped to OFF (the
    // flat-prior exact-Newton objective carries no Jeffreys term), so families
    // that historically armed the term by default opt back in explicitly.
    fn joint_jeffreys_term_required(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        crate::block_layout::block_count::validate_block_count::<SurvivalError>(
            "cause-specific survival",
            self.blocks.len(),
            block_states.len(),
        )?;
        let mut log_likelihood = 0.0;
        let mut blockworking_sets = Vec::with_capacity(self.blocks.len());
        for (block, state) in self.blocks.iter().zip(block_states.iter()) {
            let (ll, gradient, hessian) = evaluate_cause_specific_block(block, &state.beta)?;
            log_likelihood += ll;
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            });
        }
        Ok(FamilyEvaluation {
            log_likelihood,
            blockworking_sets,
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        crate::block_layout::block_count::validate_block_count::<SurvivalError>(
            "cause-specific survival",
            self.blocks.len(),
            block_states.len(),
        )?;
        let mut log_likelihood = 0.0;
        for (block, state) in self.blocks.iter().zip(block_states.iter()) {
            let (ll, _, _) = evaluate_cause_specific_block(block, &state.beta)?;
            log_likelihood += ll;
        }
        Ok(log_likelihood)
    }

    fn likelihood_blocks_uncoupled(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn output_channel_assignment(
        &self,
        specs: &[crate::custom_family::ParameterBlockSpec],
    ) -> Option<Vec<usize>> {
        if specs.len() != self.blocks.len() {
            return Some((0..self.blocks.len()).collect());
        }
        Some((0..specs.len()).collect())
    }

    fn coefficient_hessian_cost(&self, specs: &[crate::custom_family::ParameterBlockSpec]) -> u64 {
        crate::custom_family::default_coefficient_hessian_cost(specs)
    }

    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        spec: &crate::custom_family::ParameterBlockSpec,
    ) -> Result<Option<ConstraintSet>, String> {
        // The constraint rows built below live in block `block_idx`'s coefficient
        // space, so the index and the spec must agree on which block that is —
        // otherwise the returned cone would be imposed on the wrong coordinates.
        let state = block_states.get(block_idx).ok_or_else(|| {
            SurvivalError::CauseSpecificDimensionMismatch {
                reason: format!(
                    "cause-specific survival expected block index < {}, got {block_idx}",
                    block_states.len()
                ),
            }
            .to_string()
        })?;
        if state.beta.len() != spec.design.ncols() {
            return Err(SurvivalError::CauseSpecificDimensionMismatch {
                reason: format!(
                    "cause-specific survival block {block_idx} carries {} coefficient(s) but spec \
                     '{}' has {} design column(s)",
                    state.beta.len(),
                    spec.name,
                    spec.design.ncols()
                ),
            }
            .into());
        }
        let block = self.blocks.get(block_idx).ok_or_else(|| {
            SurvivalError::CauseSpecificDimensionMismatch {
                reason: format!(
                    "cause-specific survival expected block index < {}, got {block_idx}",
                    self.blocks.len()
                ),
            }
            .to_string()
        })?;
        if block.x_derivative.ncols() != spec.design.ncols() {
            return Err(SurvivalError::CauseSpecificDimensionMismatch {
                reason: format!(
                    "cause-specific survival derivative design has {} columns but block '{}' has {}",
                    block.x_derivative.ncols(),
                    spec.name,
                    spec.design.ncols()
                ),
            }
            .into());
        }
        let rhs = block
            .offset_derivative_exit
            .mapv(|offset| block.derivative_floor - offset);
        let p = block.x_derivative.ncols();
        let n_rows = block.x_derivative.nrows();
        // Structural monotone-I-spline time columns get the coefficient cone
        // `β_j ≥ 0` appended to the per-row derivative guard. The per-row guard
        // only pins `q'(t_i) ≥ floor` at training rows; a tail I-spline column
        // whose M-spline support sits beyond the largest training exit time is
        // ≈0 at every training row and so escapes it, letting the penalized fit
        // drive its coefficient negative and make `q'(t)` negative at a
        // prediction horizon in that column's support (the Royston-Parmar
        // predictor then refuses the invalid log-cumulative-hazard derivative).
        // Because each such column is monotone non-decreasing over the whole
        // axis, `β_j ≥ 0` is the exact domain-wide monotonicity certificate.
        let structural_cols = block.structural_time_columns.min(p);
        if structural_cols == 0 {
            return Ok(Some(ConstraintSet::Dense(LinearInequalityConstraints {
                a: block.x_derivative.clone(),
                b: rhs,
            })));
        }
        let mut a = Array2::<f64>::zeros((n_rows + structural_cols, p));
        a.slice_mut(ndarray::s![..n_rows, ..])
            .assign(&block.x_derivative);
        for j in 0..structural_cols {
            a[[n_rows + j, j]] = 1.0;
        }
        let mut b = Array1::<f64>::zeros(n_rows + structural_cols);
        b.slice_mut(ndarray::s![..n_rows]).assign(&rhs);
        Ok(Some(ConstraintSet::Dense(LinearInequalityConstraints {
            a,
            b,
        })))
    }

    fn max_feasible_step_size(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        let block = self.blocks.get(block_idx).ok_or_else(|| {
            SurvivalError::CauseSpecificDimensionMismatch {
                reason: format!(
                    "cause-specific survival expected block index < {}, got {block_idx}",
                    self.blocks.len()
                ),
            }
            .to_string()
        })?;
        let state = block_states.get(block_idx).ok_or_else(|| {
            SurvivalError::CauseSpecificDimensionMismatch {
                reason: format!(
                    "cause-specific survival expected {} block states, got {}",
                    self.blocks.len(),
                    block_states.len()
                ),
            }
            .to_string()
        })?;
        if delta.len() != state.beta.len() || block.x_derivative.ncols() != delta.len() {
            return Err(SurvivalError::CauseSpecificDimensionMismatch {
                reason: "cause-specific survival feasible-step dimension mismatch".to_string(),
            }
            .into());
        }
        let derivative = fast_av(&block.x_derivative, &state.beta) + &block.offset_derivative_exit;
        let derivative_delta = fast_av(&block.x_derivative, delta);
        let mut alpha_max = 1.0_f64;
        for i in 0..derivative.len() {
            if block.sampleweight[i] <= 0.0 {
                continue;
            }
            let current = derivative[i] - block.derivative_floor;
            let slope = derivative_delta[i];
            if slope < 0.0 {
                if current <= 0.0 {
                    return Ok(Some(0.0));
                }
                alpha_max = alpha_max.min(DERIVATIVE_FRACTION_TO_BOUNDARY * current / -slope);
            }
        }
        Ok(Some(alpha_max.clamp(0.0, 1.0)))
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let block = self.blocks.get(block_idx).ok_or_else(|| {
            SurvivalError::CauseSpecificDimensionMismatch {
                reason: format!(
                    "cause-specific survival expected block index < {}, got {block_idx}",
                    self.blocks.len()
                ),
            }
            .to_string()
        })?;
        let state = block_states.get(block_idx).ok_or_else(|| {
            SurvivalError::CauseSpecificDimensionMismatch {
                reason: format!(
                    "cause-specific survival expected {} block states, got {}",
                    self.blocks.len(),
                    block_states.len()
                ),
            }
            .to_string()
        })?;
        Ok(Some(cause_specific_hessian_directional_derivative(
            block,
            &state.beta,
            d_beta,
        )?))
    }

    fn exact_newton_hessian_second_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let block = self.blocks.get(block_idx).ok_or_else(|| {
            SurvivalError::CauseSpecificDimensionMismatch {
                reason: format!(
                    "cause-specific survival expected block index < {}, got {block_idx}",
                    self.blocks.len()
                ),
            }
            .to_string()
        })?;
        let state = block_states.get(block_idx).ok_or_else(|| {
            SurvivalError::CauseSpecificDimensionMismatch {
                reason: format!(
                    "cause-specific survival expected {} block states, got {}",
                    self.blocks.len(),
                    block_states.len()
                ),
            }
            .to_string()
        })?;
        Ok(Some(cause_specific_hessian_second_directional_derivative(
            block,
            &state.beta,
            d_beta_u,
            d_beta_v,
        )?))
    }
}

/// The live third-order coefficient Hessian, lowered at build time from the same
/// [`cause_specific_row`] expression as value/gradient/Hessian.
fn cause_specific_hessian_directional_derivative(
    block: &CauseSpecificRoystonParmarBlock,
    beta: &Array1<f64>,
    d_beta: &Array1<f64>,
) -> Result<Array2<f64>, SurvivalError> {
    let p = block.x_exit.ncols();
    if beta.len() != p || d_beta.len() != p {
        return Err(SurvivalError::CauseSpecificDimensionMismatch {
            reason: "cause-specific survival Hessian derivative dimension mismatch".to_string(),
        });
    }
    let eta_entry = fast_av(&block.x_entry, beta) + &block.offset_eta_entry;
    let eta_exit = fast_av(&block.x_exit, beta) + &block.offset_eta_exit;
    let derivative = fast_av(&block.x_derivative, beta) + &block.offset_derivative_exit;
    let d_eta_entry = fast_av(&block.x_entry, d_beta);
    let d_eta_exit = fast_av(&block.x_exit, d_beta);
    let d_derivative = fast_av(&block.x_derivative, d_beta);
    let n = block.event_target.len();
    let mut weights: [Array1<f64>; 6] = std::array::from_fn(|_| Array1::zeros(n));

    for i in 0..n {
        let Some(input) =
            cause_specific_atom_input(block, i, eta_entry[i], eta_exit[i], derivative[i])?
        else {
            continue;
        };
        let direction = [
            d_eta_exit[i],
            d_eta_entry[i] * f64::from(input.entry_active),
            d_derivative[i] * f64::from(input.event),
        ];
        let matrix = cause_specific_row_third_contracted(
            input.primary[0],
            input.primary[1],
            input.primary[2],
            input.weight,
            input.entry_active,
            input.event,
            &direction,
        );
        for (slot, &(left, right)) in CAUSE_SPECIFIC_PRIMARY_PAIRS.iter().enumerate() {
            weights[slot][i] = matrix[left][right];
        }
    }
    Ok(cause_specific_pullback_hessian(block, &weights))
}

/// The live fourth-order coefficient Hessian, lowered at build time from the
/// same [`cause_specific_row`] expression as every lower channel.
fn cause_specific_hessian_second_directional_derivative(
    block: &CauseSpecificRoystonParmarBlock,
    beta: &Array1<f64>,
    d_beta_u: &Array1<f64>,
    d_beta_v: &Array1<f64>,
) -> Result<Array2<f64>, SurvivalError> {
    let p = block.x_exit.ncols();
    if beta.len() != p || d_beta_u.len() != p || d_beta_v.len() != p {
        return Err(SurvivalError::CauseSpecificDimensionMismatch {
            reason: "cause-specific survival second Hessian derivative dimension mismatch"
                .to_string(),
        });
    }
    let eta_entry = fast_av(&block.x_entry, beta) + &block.offset_eta_entry;
    let eta_exit = fast_av(&block.x_exit, beta) + &block.offset_eta_exit;
    let derivative = fast_av(&block.x_derivative, beta) + &block.offset_derivative_exit;
    let u_eta_entry = fast_av(&block.x_entry, d_beta_u);
    let u_eta_exit = fast_av(&block.x_exit, d_beta_u);
    let u_derivative = fast_av(&block.x_derivative, d_beta_u);
    let v_eta_entry = fast_av(&block.x_entry, d_beta_v);
    let v_eta_exit = fast_av(&block.x_exit, d_beta_v);
    let v_derivative = fast_av(&block.x_derivative, d_beta_v);
    let n = block.event_target.len();
    let mut weights: [Array1<f64>; 6] = std::array::from_fn(|_| Array1::zeros(n));

    for i in 0..n {
        let Some(input) =
            cause_specific_atom_input(block, i, eta_entry[i], eta_exit[i], derivative[i])?
        else {
            continue;
        };
        let direction_u = [
            u_eta_exit[i],
            u_eta_entry[i] * f64::from(input.entry_active),
            u_derivative[i] * f64::from(input.event),
        ];
        let direction_v = [
            v_eta_exit[i],
            v_eta_entry[i] * f64::from(input.entry_active),
            v_derivative[i] * f64::from(input.event),
        ];
        let matrix = cause_specific_row_fourth_contracted(
            input.primary[0],
            input.primary[1],
            input.primary[2],
            input.weight,
            input.entry_active,
            input.event,
            &direction_u,
            &direction_v,
        );
        for (slot, &(left, right)) in CAUSE_SPECIFIC_PRIMARY_PAIRS.iter().enumerate() {
            weights[slot][i] = matrix[left][right];
        }
    }
    Ok(cause_specific_pullback_hessian(block, &weights))
}

pub fn survival_event_code_from_value(value: f64, row_index: usize) -> Result<u8, String> {
    const INTEGER_TOL: f64 = 1e-8;
    const MAX_AUTO_CAUSES: u8 = 32;
    if !value.is_finite() {
        return Err(SurvivalError::EventCodeInvalid {
            reason: format!(
                "survival event value at row {} is non-finite",
                row_index + 1
            ),
        }
        .into());
    }
    if value < 0.0 {
        return Err(SurvivalError::EventCodeInvalid {
            reason: format!(
                "survival event value at row {} is negative: {value}",
                row_index + 1
            ),
        }
        .into());
    }
    let rounded = value.round();
    if (value - rounded).abs() > INTEGER_TOL {
        return Err(SurvivalError::EventCodeInvalid {
            reason: format!(
                "survival event value at row {} must be an integer code with 0=censored, got {value}",
                row_index + 1
            ),
        }
        .into());
    }
    if rounded > f64::from(MAX_AUTO_CAUSES) {
        return Err(SurvivalError::EventCodeInvalid {
            reason: format!(
                "survival event value at row {} has code {rounded}; automatic competing-risks detection supports codes 0..={MAX_AUTO_CAUSES}",
                row_index + 1
            ),
        }
        .into());
    }
    Ok(rounded as u8)
}

pub fn cause_count_from_event_codes(
    event_codes: ArrayView1<'_, u8>,
) -> Result<usize, SurvivalError> {
    let max_code = event_codes.iter().copied().max().map_or(0, usize::from);
    if max_code == 0 {
        return Ok(1);
    }

    let mut present = vec![false; max_code + 1];
    for code in event_codes.iter().copied() {
        present[usize::from(code)] = true;
    }
    if (1..=max_code).any(|code| !present[code]) {
        let actual = present
            .iter()
            .enumerate()
            .skip(1)
            .filter_map(|(code, &seen)| seen.then_some(code.to_string()))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(SurvivalError::EventCodeInvalid {
            reason: format!(
                "survival competing-risks event codes must use contiguous positive codes; observed nonzero codes are {{{actual}}}. Remap event codes contiguously (for example, {{0,1,3}} -> {{0,1,2}}), otherwise a phantom cause is fit with no events and pollutes CIF assembly."
            ),
        });
    }

    Ok(max_code)
}

fn compress_positive_collinear_constraints(
    a: &Array2<f64>,
    b: &Array1<f64>,
) -> LinearInequalityConstraints {
    const SCALE_TOL: f64 = 1e-14;
    const KEY_TOL: f64 = 1e-8;

    let mut grouped: BTreeMap<Vec<i64>, (Vec<f64>, f64)> = BTreeMap::new();
    let mut fallbackrows: Vec<(Vec<f64>, f64)> = Vec::new();

    for i in 0..a.nrows() {
        let row = a.row(i);
        let scale = row.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        if !scale.is_finite() || scale <= SCALE_TOL {
            if b[i] > 0.0 {
                fallbackrows.push((row.to_vec(), b[i]));
            }
            continue;
        }

        let normalizedrow: Vec<f64> = row
            .iter()
            .map(|&v| {
                let scaled = v / scale;
                if scaled.abs() <= KEY_TOL { 0.0 } else { scaled }
            })
            .collect();
        let normalized_rhs = b[i] / scale;
        let key: Vec<i64> = normalizedrow
            .iter()
            .map(|&v| (v / KEY_TOL).round() as i64)
            .collect();

        match grouped.get_mut(&key) {
            Some((_, rhs_max)) => {
                if normalized_rhs > *rhs_max {
                    *rhs_max = normalized_rhs;
                }
            }
            None => {
                grouped.insert(key, (normalizedrow, normalized_rhs));
            }
        }
    }

    let nrows = grouped.len() + fallbackrows.len();
    let n_cols = a.ncols();
    let mut a_out = Array2::<f64>::zeros((nrows, n_cols));
    let mut b_out = Array1::<f64>::zeros(nrows);

    let mut outrow = 0usize;
    for (_, (row, rhs)) in grouped {
        for (j, value) in row.into_iter().enumerate() {
            a_out[[outrow, j]] = value;
        }
        b_out[outrow] = rhs;
        outrow += 1;
    }
    for (row, rhs) in fallbackrows {
        for (j, value) in row.into_iter().enumerate() {
            a_out[[outrow, j]] = value;
        }
        b_out[outrow] = rhs;
        outrow += 1;
    }

    LinearInequalityConstraints { a: a_out, b: b_out }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SurvivalMonotonicityPenalty {
    pub tolerance: f64,
}

#[derive(Debug, Clone)]
enum SurvivalDesign {
    Flat {
        x_entry: Array2<f64>,
        x_exit: Array2<f64>,
        x_derivative: Array2<f64>,
    },
    TimeCovariateShared {
        time_entry: Array2<f64>,
        time_exit: Array2<f64>,
        time_derivative: Array2<f64>,
        covariates: Array2<f64>,
    },
}

impl SurvivalDesign {
    fn p_total(&self) -> usize {
        match self {
            Self::Flat { x_exit, .. } => x_exit.ncols(),
            Self::TimeCovariateShared {
                time_exit,
                covariates,
                ..
            } => time_exit.ncols() + covariates.ncols(),
        }
    }

    fn design_dot(&self, time_mat: &Array2<f64>, beta: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Flat { .. } => time_mat.dot(beta),
            Self::TimeCovariateShared { covariates, .. } => {
                let p_time = time_mat.ncols();
                let mut out = time_mat.dot(&beta.slice(ndarray::s![..p_time]));
                if covariates.ncols() > 0 {
                    out += &covariates.dot(&beta.slice(ndarray::s![p_time..]));
                }
                out
            }
        }
    }

    fn fill_row(&self, time_mat: &Array2<f64>, i: usize, out: &mut [f64]) {
        match self {
            Self::Flat { .. } => {
                for (dst, &src) in out.iter_mut().zip(time_mat.row(i).iter()) {
                    *dst = src;
                }
            }
            Self::TimeCovariateShared { covariates, .. } => {
                let p_time = time_mat.ncols();
                for j in 0..p_time {
                    out[j] = time_mat[[i, j]];
                }
                for j in 0..covariates.ncols() {
                    out[p_time + j] = covariates[[i, j]];
                }
            }
        }
    }
}

/// Pre-allocated workspace buffers for `update_state` to avoid per-iteration allocations.
#[derive(Debug, Clone)]
struct SurvivalWorkspace {
    w_event: Array1<f64>,
    w_event_inv_deriv: Array1<f64>,
    w_event_outer: Array1<f64>,
    w_hess_exit: Array1<f64>,
    w_hess_entry: Array1<f64>,
}

impl SurvivalWorkspace {
    fn new(n: usize) -> Self {
        Self {
            w_event: Array1::zeros(n),
            w_event_inv_deriv: Array1::zeros(n),
            w_event_outer: Array1::zeros(n),
            w_hess_exit: Array1::zeros(n),
            w_hess_entry: Array1::zeros(n),
        }
    }

    fn reset(&mut self, n: usize) {
        if self.w_event.len() != n {
            *self = Self::new(n);
        } else {
            self.w_event.fill(0.0);
            self.w_event_inv_deriv.fill(0.0);
            self.w_event_outer.fill(0.0);
            self.w_hess_exit.fill(0.0);
            self.w_hess_entry.fill(0.0);
        }
    }
}

/// Per-observation gradients of the unpenalized survival NLL with respect
/// to each additive offset channel, at a given β. See
/// [`WorkingModelSurvival::offset_channel_residuals`] for the algebra.
///
/// Contract: all four arrays have length `n` = number of observations.
/// Rows with non-positive sampleweight are 0 in every channel. The
/// `derivative` channel is 0 in all non-event rows. The `right` channel is
/// the interval upper-bound (`R`) η-offset sensitivity and is exactly 0 for
/// every NON-interval-censored model and every non-interval row of the latent
/// interval model (only the dedicated `SurvInterval(L, R, event)` latent fit
/// populates it); the baseline-θ chain rule contracts it against the
/// `age_right`-evaluated η-partial.
#[derive(Clone, Debug)]
pub struct OffsetChannelResiduals {
    /// ∂NLL/∂o_X: w·(exp(η_exit) − δ) per row.
    pub exit: Array1<f64>,
    /// ∂NLL/∂o_E: −w·exp(η_entry) if row has a positive entry interval else 0.
    pub entry: Array1<f64>,
    /// ∂NLL/∂o_D: −w·δ / s (event-row only).
    pub derivative: Array1<f64>,
    /// ∂NLL/∂o_R: interval upper-bound (`R`) η-offset sensitivity,
    /// `−w·∂(log-lik)/∂q_right`. Nonzero only for interval-censored latent
    /// rows; exactly 0 for every other channel/model.
    pub right: Array1<f64>,
}

/// Per-observation Hessians of the unpenalized survival NLL with respect
/// to additive offset channels in `(entry, exit, derivative)` order.
#[derive(Clone, Debug)]
pub struct OffsetChannelCurvatures {
    pub rows: Vec<[[f64; 3]; 3]>,
}

#[derive(Debug)]
pub struct WorkingModelSurvival {
    age_entry: Array1<f64>,
    age_exit: Array1<f64>,
    entry_at_origin: Array1<bool>,
    event_target: Array1<u8>,
    sampleweight: Array1<f64>,
    design: SurvivalDesign,
    offset_eta_entry: Array1<f64>,
    offset_eta_exit: Array1<f64>,
    offset_derivative_exit: Array1<f64>,
    penalties: PenaltyBlocks,
    monotonicity: SurvivalMonotonicityPenalty,
    structurally_monotonic: bool,
    structural_time_columns: usize,
    monotonicity_constraint_rows: Option<Array2<f64>>,
    monotonicity_constraint_offsets: Option<Array1<f64>>,
    workspace: std::sync::Mutex<SurvivalWorkspace>,
}

impl Clone for WorkingModelSurvival {
    fn clone(&self) -> Self {
        let workspace = self
            .workspace
            .lock()
            .expect("survival workspace mutex was not poisoned by a panicking holder")
            .clone();
        Self {
            age_entry: self.age_entry.clone(),
            age_exit: self.age_exit.clone(),
            entry_at_origin: self.entry_at_origin.clone(),
            event_target: self.event_target.clone(),
            sampleweight: self.sampleweight.clone(),
            design: self.design.clone(),
            offset_eta_entry: self.offset_eta_entry.clone(),
            offset_eta_exit: self.offset_eta_exit.clone(),
            offset_derivative_exit: self.offset_derivative_exit.clone(),
            penalties: self.penalties.clone(),
            monotonicity: self.monotonicity,
            structurally_monotonic: self.structurally_monotonic,
            structural_time_columns: self.structural_time_columns,
            monotonicity_constraint_rows: self.monotonicity_constraint_rows.clone(),
            monotonicity_constraint_offsets: self.monotonicity_constraint_offsets.clone(),
            workspace: std::sync::Mutex::new(workspace),
        }
    }
}

impl WorkingModelSurvival {
    const LOG_F64_MAX: f64 = 709.782712893384;

    #[inline]
    fn scaled_exp_component(log_scale: f64, base: f64) -> Result<f64, EstimationError> {
        if base == 0.0 {
            return Ok(0.0);
        }
        let log_abs = log_scale + base.abs().ln();
        if !log_abs.is_finite() {
            crate::bail_invalid_estim!("survival interval term produced non-finite log-magnitude");
        }
        if log_abs > Self::LOG_F64_MAX {
            crate::bail_invalid_estim!(
                "survival interval term exceeds f64 range (log-magnitude={log_abs:.3e})"
            );
        }
        Ok(base.signum() * log_abs.exp())
    }

    fn coefficient_dim(&self) -> usize {
        self.design.p_total()
    }

    fn nrows(&self) -> usize {
        self.sampleweight.len()
    }

    fn entry_dot(&self, beta: &Array1<f64>) -> Array1<f64> {
        let time_mat = match &self.design {
            SurvivalDesign::Flat { x_entry, .. } => x_entry,
            SurvivalDesign::TimeCovariateShared { time_entry, .. } => time_entry,
        };
        self.design.design_dot(time_mat, beta)
    }

    fn exit_dot(&self, beta: &Array1<f64>) -> Array1<f64> {
        let time_mat = match &self.design {
            SurvivalDesign::Flat { x_exit, .. } => x_exit,
            SurvivalDesign::TimeCovariateShared { time_exit, .. } => time_exit,
        };
        self.design.design_dot(time_mat, beta)
    }

    fn derivative_dot(&self, beta: &Array1<f64>) -> Array1<f64> {
        match &self.design {
            SurvivalDesign::Flat { x_derivative, .. } => x_derivative.dot(beta),
            SurvivalDesign::TimeCovariateShared {
                time_derivative, ..
            } => time_derivative.dot(&beta.slice(ndarray::s![..time_derivative.ncols()])),
        }
    }

    fn fill_entry_row(&self, i: usize, out: &mut [f64]) {
        let time_mat = match &self.design {
            SurvivalDesign::Flat { x_entry, .. } => x_entry,
            SurvivalDesign::TimeCovariateShared { time_entry, .. } => time_entry,
        };
        self.design.fill_row(time_mat, i, out);
    }

    fn fill_exit_row(&self, i: usize, out: &mut [f64]) {
        let time_mat = match &self.design {
            SurvivalDesign::Flat { x_exit, .. } => x_exit,
            SurvivalDesign::TimeCovariateShared { time_exit, .. } => time_exit,
        };
        self.design.fill_row(time_mat, i, out);
    }

    fn fill_derivative_row(&self, i: usize, out: &mut [f64]) {
        match &self.design {
            SurvivalDesign::Flat { x_derivative, .. } => {
                for (dst, &src) in out.iter_mut().zip(x_derivative.row(i).iter()) {
                    *dst = src;
                }
            }
            SurvivalDesign::TimeCovariateShared {
                time_derivative, ..
            } => {
                let p_time = time_derivative.ncols();
                for j in 0..p_time {
                    out[j] = time_derivative[[i, j]];
                }
                for dst in out.iter_mut().skip(p_time) {
                    *dst = 0.0;
                }
            }
        }
    }

    fn derivative_xt_diag_x(&self, weights: &Array1<f64>) -> Array2<f64> {
        match &self.design {
            SurvivalDesign::Flat { x_derivative, .. } => fast_xt_diag_x(x_derivative, weights),
            SurvivalDesign::TimeCovariateShared {
                time_derivative,
                covariates,
                ..
            } => {
                let p_time = time_derivative.ncols();
                let p_cov = covariates.ncols();
                let mut out = Array2::<f64>::zeros((p_time + p_cov, p_time + p_cov));
                let time_block = fast_xt_diag_x(time_derivative, weights);
                out.slice_mut(ndarray::s![..p_time, ..p_time])
                    .assign(&time_block);
                out
            }
        }
    }

    /// Compute the full p×p Hessian contribution for the interval terms:
    ///   H = X_exit^T diag(w_exit) X_exit - X_entry^T diag(w_entry) X_entry
    /// using faer-accelerated BLAS on the stored design matrix blocks.
    fn interval_hessian_blas(&self, w_exit: &Array1<f64>, w_entry: &Array1<f64>) -> Array2<f64> {
        match &self.design {
            SurvivalDesign::Flat {
                x_entry, x_exit, ..
            } => {
                let mut h = fast_xt_diag_x(x_exit, w_exit);
                h -= &fast_xt_diag_x(x_entry, w_entry);
                h
            }
            SurvivalDesign::TimeCovariateShared {
                time_entry,
                time_exit,
                covariates,
                ..
            } => {
                let p_time = time_exit.ncols();
                let p_cov = covariates.ncols();
                let p = p_time + p_cov;
                let mut h = Array2::<f64>::zeros((p, p));
                // time-time block: T_exit^T W_exit T_exit - T_entry^T W_entry T_entry
                let tt = {
                    let mut block = fast_xt_diag_x(time_exit, w_exit);
                    block -= &fast_xt_diag_x(time_entry, w_entry);
                    block
                };
                h.slice_mut(ndarray::s![..p_time, ..p_time]).assign(&tt);
                if p_cov > 0 {
                    // time-cov block: T_exit^T W_exit C - T_entry^T W_entry C
                    let tc = {
                        let mut block = fast_xt_diag_y(time_exit, w_exit, covariates);
                        block -= &fast_xt_diag_y(time_entry, w_entry, covariates);
                        block
                    };
                    h.slice_mut(ndarray::s![..p_time, p_time..]).assign(&tc);
                    h.slice_mut(ndarray::s![p_time.., ..p_time]).assign(&tc.t());
                    // cov-cov block: C^T (W_exit - W_entry) C
                    let w_diff = w_exit - w_entry;
                    let cc = fast_xt_diag_x(covariates, &w_diff);
                    h.slice_mut(ndarray::s![p_time.., p_time..]).assign(&cc);
                }
                h
            }
        }
    }

    /// Clamped structural derivative: `max(deriv, floor)` for derivatives
    /// above the roundoff tolerance, `None` outside structural monotonicity
    /// or for genuinely negative derivatives.
    ///
    /// Returns `(value, slope)` where `slope` is the exact derivative of the
    /// clamp itself: 1 on the identity branch, 0 on the floored branch.
    /// Every consumer that differentiates through the structural derivative
    /// MUST scale its derivative-channel terms by `slope` — the floored
    /// branch is locally constant in β, so gradients and Hessians of
    /// `ln(value)` and `1/value` are exactly zero there. Emitting the
    /// reciprocal-scale terms of the UNclamped expression (≈1e12 gradient,
    /// ≈1e24 curvature at deriv = 5e-13) against a value branch that is flat
    /// desynchronizes the objective from its derivatives.
    fn stabilized_structural_derivative(&self, deriv: f64) -> Option<(f64, f64)> {
        const STRUCTURAL_MONO_ROUNDOFF_TOL: f64 = 1e-7;
        const STRUCTURAL_DERIV_FLOOR: f64 = 1e-12;
        if !self.structurally_monotonic {
            return None;
        }
        if deriv >= STRUCTURAL_DERIV_FLOOR {
            return Some((deriv, 1.0));
        }
        if deriv >= -STRUCTURAL_MONO_ROUNDOFF_TOL {
            return Some((STRUCTURAL_DERIV_FLOOR, 0.0));
        }
        None
    }

    fn validate_penalties(
        penalties: &PenaltyBlocks,
        coefficient_dim: usize,
    ) -> Result<(), SurvivalError> {
        for block in &penalties.blocks {
            if !block.lambda.is_finite() || block.lambda < 0.0 {
                return Err(SurvivalError::NonFiniteInput);
            }
            if block.range.start > block.range.end || block.range.end > coefficient_dim {
                return Err(SurvivalError::DimensionMismatch);
            }
            let block_dim = block.range.end - block.range.start;
            if block.matrix.nrows() != block_dim || block.matrix.ncols() != block_dim {
                return Err(SurvivalError::DimensionMismatch);
            }
            if block.matrix.iter().any(|v| !v.is_finite()) {
                return Err(SurvivalError::NonFiniteInput);
            }
        }
        Ok(())
    }

    fn derivative_guard(&self) -> f64 {
        if self.structurally_monotonic {
            // I-spline basis is monotone by construction when coefficients ≥ 0.
            // A derivative of zero (flat hazard) is valid, so the guard only
            // rejects genuinely negative derivatives from floating-point noise.
            return 0.0;
        }
        self.monotonicity.tolerance.max(0.0)
    }

    fn derivative_guard_numerical(&self) -> f64 {
        let derivative_guard = self.derivative_guard();
        if derivative_guard <= 0.0 {
            // For structural monotonicity (guard = 0), tiny negative derivs are
            // tolerated because `stabilized_structural_derivative` lifts the
            // value back to a small positive floor before any `ln`/`1/deriv`
            // use. For *non-structural* monotonicity with tolerance == 0 the
            // raw derivative flows straight through into the event-row
            // `deriv.ln()` and `1.0 / deriv`, so any non-positive value would
            // produce NaN / huge negative weights. Keep the slack only when
            // the structural stabilizer is active.
            if self.structurally_monotonic {
                -1e-10
            } else {
                1e-12
            }
        } else {
            (derivative_guard - (1e-10_f64).min(0.01 * derivative_guard)).max(1e-12)
        }
    }

    fn interval_increment_guard(&self, h_entry: f64, h_exit: f64) -> f64 {
        let scale = h_entry.abs().max(h_exit.abs()).max(1.0);
        1e-10 * scale
    }

    fn structural_time_coefficient_constraints(&self) -> Option<LinearInequalityConstraints> {
        if !self.structurally_monotonic {
            return None;
        }
        let p = self.coefficient_dim();
        let time_columns = self.structural_time_columns.min(p);
        if time_columns == 0 {
            return None;
        }
        // Constrain EVERY time-block coefficient `γ_j ≥ 0`, not only the columns
        // whose derivative basis is active at a training row. Each I-spline time
        // column is monotone non-decreasing across the whole log-time axis (its
        // derivative basis is a non-negative M-spline, verified element-wise in
        // `set_structural_monotonicity`), so the cumulative hazard is monotone at
        // EVERY evaluation time — including prediction horizons beyond the training
        // exit times — iff every time coefficient is non-negative. This whole-block
        // constraint is therefore the exact domain-wide structural-monotonicity
        // certificate.
        //
        // An earlier version restricted the constraint to columns with
        // `derivative_row[j] > tol` at some training row. Tail I-spline columns —
        // whose M-spline support sits beyond the largest training exit time — are
        // ≈0 at every training row and were left UNCONSTRAINED. The penalized fit
        // could then drive those `γ_j < 0`: monotonicity still held at every
        // training row (their basis is ≈0 there) but broke at prediction times in
        // the tail column's support, yielding a negative log-cumulative-hazard
        // derivative `d(logΛ)/dt` that the Royston-Parmar predictor correctly
        // refuses. Constraining the full block closes that tail blind spot at the
        // fit rather than clamping the derivative at predict time.
        let mut a = Array2::<f64>::zeros((time_columns, p));
        let b = Array1::<f64>::zeros(time_columns);
        for j in 0..time_columns {
            a[[j, j]] = 1.0;
        }
        Some(LinearInequalityConstraints { a, b })
    }

    pub fn monotonicity_linear_constraints(&self) -> Option<LinearInequalityConstraints> {
        let p = self.coefficient_dim();
        const DERIVATIVE_ROW_NORM_TOL: f64 = 1e-12;
        if p == 0 {
            return None;
        }
        if self.structurally_monotonic {
            return self.structural_time_coefficient_constraints();
        }
        if let (Some(rows), Some(offsets)) = (
            self.monotonicity_constraint_rows.as_ref(),
            self.monotonicity_constraint_offsets.as_ref(),
        ) {
            let activerows: Vec<usize> = (0..rows.nrows())
                .filter(|&i| {
                    rows.row(i).iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()))
                        > DERIVATIVE_ROW_NORM_TOL
                })
                .collect();
            if activerows.is_empty() {
                return None;
            }
            let mut a = Array2::<f64>::zeros((activerows.len(), p));
            let mut b = Array1::<f64>::zeros(activerows.len());
            for (r, &i) in activerows.iter().enumerate() {
                a.row_mut(r).assign(&rows.row(i));
                b[r] = self.derivative_guard() - offsets[i];
            }
            return Some(compress_positive_collinear_constraints(&a, &b));
        }
        None
    }

    fn validate_offsets(
        offsets: Option<SurvivalBaselineOffsets<'_>>,
        n: usize,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), SurvivalError> {
        if let Some(off) = offsets {
            if off.eta_entry.len() != n || off.eta_exit.len() != n || off.derivative_exit.len() != n
            {
                return Err(SurvivalError::DimensionMismatch);
            }
            if off.eta_entry.iter().any(|v| !v.is_finite())
                || off.eta_exit.iter().any(|v| !v.is_finite())
                || off.derivative_exit.iter().any(|v| !v.is_finite())
            {
                return Err(SurvivalError::NonFiniteInput);
            }
            Ok((
                off.eta_entry.to_owned(),
                off.eta_exit.to_owned(),
                off.derivative_exit.to_owned(),
            ))
        } else {
            Ok((Array1::zeros(n), Array1::zeros(n), Array1::zeros(n)))
        }
    }

    fn validate_common_inputs(
        age_entry: &ArrayView1<f64>,
        age_exit: &ArrayView1<f64>,
        event_target: &ArrayView1<u8>,
        event_competing: &ArrayView1<u8>,
        sampleweight: &ArrayView1<f64>,
    ) -> Result<(), SurvivalError> {
        if age_entry.iter().any(|v| !v.is_finite())
            || age_exit.iter().any(|v| !v.is_finite())
            || sampleweight.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(SurvivalError::NonFiniteInput);
        }
        // The single-hazard engine's `event_target` contract is binary {0, 1}.
        // A code > 1 is a *valid finite multi-cause label* that simply must be
        // projected first (any-event for the pooled baseline, cause-specific for
        // each block); it is NOT a non-finite input. Report it as such so the
        // failure is actionable and never surfaces as the misleading "inputs
        // contain non-finite values".
        if let Some(&label) = event_target.iter().find(|&&v| v > 1) {
            return Err(SurvivalError::EventCodeInvalid {
                reason: format!(
                    "single-hazard survival engine requires a binary {{0, 1}} event_target, got multi-cause label {label}; competing-risks codes must be projected via pooled_any_event_indicator / cause_specific_event_indicator before construction"
                ),
            });
        }
        if let Some(&label) = event_competing.iter().find(|&&v| v > 1) {
            return Err(SurvivalError::EventCodeInvalid {
                reason: format!(
                    "single-hazard survival engine requires a binary {{0, 1}} event_competing, got multi-cause label {label}"
                ),
            });
        }
        if event_target
            .iter()
            .zip(event_competing.iter())
            .any(|(&target, &competing)| target > 0 && competing > 0)
        {
            return Err(SurvivalError::EventCodeInvalid {
                reason: "a row cannot be simultaneously a target event and a competing event"
                    .to_string(),
            });
        }
        // The "must have at least one target event" requirement is a
        // *fittability* check, not a structural one: with all rows censored the
        // likelihood has no event score, so any subsequent fit cannot identify
        // the hazard and the optimizer spins on a flat landscape.  But the
        // structural integrity of the engine — its derivative-guard rejection
        // of decreasing cumulative hazards, its monotonicity-collocation
        // bookkeeping, its update_state numerics — is well-defined on
        // all-censored inputs, and unit tests legitimately exercise those
        // structural paths on censored fixtures.  Move the fittability check
        // out of construction; production fit dispatchers (e.g.
        // `solver::fit_orchestration::materialize_survival`) enforce it on the
        // single chokepoint that actually starts an optimization, where
        // the failure mode it guards against is reachable.
        if age_entry
            .iter()
            .zip(age_exit.iter())
            .any(|(&entry, &exit)| entry < 0.0 || exit <= 0.0)
        {
            return Err(SurvivalError::NonFiniteInput);
        }
        Ok::<(), _>(())
    }

    fn validate_monotonicity_constraints(
        rows: Option<ArrayView2<'_, f64>>,
        offsets: Option<ArrayView1<'_, f64>>,
        coefficient_dim: usize,
    ) -> Result<(Option<Array2<f64>>, Option<Array1<f64>>), SurvivalError> {
        match (rows, offsets) {
            (None, None) => Ok((None, None)),
            (Some(rows), Some(offsets)) => {
                if rows.ncols() != coefficient_dim
                    || rows.nrows() != offsets.len()
                    || rows.iter().any(|v| !v.is_finite())
                    || offsets.iter().any(|v| !v.is_finite())
                {
                    return Err(SurvivalError::DimensionMismatch);
                }
                Ok((Some(rows.to_owned()), Some(offsets.to_owned())))
            }
            _ => Err(SurvivalError::DimensionMismatch),
        }
    }

    fn finish_construction(
        age_entry: ArrayView1<f64>,
        age_exit: ArrayView1<f64>,
        event_target: ArrayView1<u8>,
        sampleweight: ArrayView1<f64>,
        design: SurvivalDesign,
        offset_eta_entry: Array1<f64>,
        offset_eta_exit: Array1<f64>,
        offset_derivative_exit: Array1<f64>,
        penalties: PenaltyBlocks,
        monotonicity: SurvivalMonotonicityPenalty,
        monotonicity_constraint_rows: Option<Array2<f64>>,
        monotonicity_constraint_offsets: Option<Array1<f64>>,
    ) -> Self {
        let n = age_entry.len();
        Self {
            age_entry: age_entry.to_owned(),
            age_exit: age_exit.to_owned(),
            entry_at_origin: age_entry.mapv(|t| t <= ENTRY_AT_ORIGIN_THRESHOLD),
            event_target: event_target.to_owned(),
            sampleweight: sampleweight.to_owned(),
            design,
            offset_eta_entry,
            offset_eta_exit,
            offset_derivative_exit,
            penalties,
            monotonicity,
            structurally_monotonic: false,
            structural_time_columns: 0,
            monotonicity_constraint_rows,
            monotonicity_constraint_offsets,
            workspace: std::sync::Mutex::new(SurvivalWorkspace::new(n)),
        }
    }

    pub fn from_engine_inputswith_offsets(
        inputs: SurvivalEngineInputs<'_>,
        offsets: Option<SurvivalBaselineOffsets<'_>>,
        penalties: PenaltyBlocks,
        monotonicity: SurvivalMonotonicityPenalty,
        spec: SurvivalSpec,
    ) -> Result<Self, SurvivalError> {
        if spec == SurvivalSpec::Crude {
            return Err(SurvivalError::UnsupportedSpec("crude"));
        }
        let n = inputs.age_entry.len();
        let p = inputs.x_entry.ncols();
        if inputs.age_exit.len() != n
            || inputs.event_target.len() != n
            || inputs.event_competing.len() != n
            || inputs.sampleweight.len() != n
            || inputs.x_entry.nrows() != n
            || inputs.x_exit.nrows() != n
            || inputs.x_derivative.nrows() != n
            || inputs.x_entry.ncols() != inputs.x_exit.ncols()
            || inputs.x_entry.ncols() != inputs.x_derivative.ncols()
        {
            return Err(SurvivalError::DimensionMismatch);
        }
        Self::validate_penalties(&penalties, p)?;
        Self::validate_common_inputs(
            &inputs.age_entry,
            &inputs.age_exit,
            &inputs.event_target,
            &inputs.event_competing,
            &inputs.sampleweight,
        )?;
        if inputs.x_entry.iter().any(|v| !v.is_finite())
            || inputs.x_exit.iter().any(|v| !v.is_finite())
            || inputs.x_derivative.iter().any(|v| !v.is_finite())
        {
            return Err(SurvivalError::NonFiniteInput);
        }
        let (offset_eta_entry, offset_eta_exit, offset_derivative_exit) =
            Self::validate_offsets(offsets, n)?;
        let (monotonicity_constraint_rows, monotonicity_constraint_offsets) =
            Self::validate_monotonicity_constraints(
                inputs.monotonicity_constraint_rows,
                inputs.monotonicity_constraint_offsets,
                p,
            )?;

        Ok(Self::finish_construction(
            inputs.age_entry,
            inputs.age_exit,
            inputs.event_target,
            inputs.sampleweight,
            SurvivalDesign::Flat {
                x_entry: inputs.x_entry.to_owned(),
                x_exit: inputs.x_exit.to_owned(),
                x_derivative: inputs.x_derivative.to_owned(),
            },
            offset_eta_entry,
            offset_eta_exit,
            offset_derivative_exit,
            penalties,
            monotonicity,
            monotonicity_constraint_rows,
            monotonicity_constraint_offsets,
        ))
    }

    pub fn from_time_covariate_inputswith_offsets(
        inputs: SurvivalTimeCovarInputs<'_>,
        offsets: Option<SurvivalBaselineOffsets<'_>>,
        penalties: PenaltyBlocks,
        monotonicity: SurvivalMonotonicityPenalty,
        spec: SurvivalSpec,
    ) -> Result<Self, SurvivalError> {
        if spec == SurvivalSpec::Crude {
            return Err(SurvivalError::UnsupportedSpec("crude"));
        }
        let n = inputs.age_entry.len();
        let p_time = inputs.time_entry.ncols();
        let p_cov = inputs.covariates.ncols();
        let p = p_time + p_cov;
        if inputs.age_exit.len() != n
            || inputs.event_target.len() != n
            || inputs.event_competing.len() != n
            || inputs.sampleweight.len() != n
            || inputs.time_entry.nrows() != n
            || inputs.time_exit.nrows() != n
            || inputs.time_derivative.nrows() != n
            || inputs.covariates.nrows() != n
            || inputs.time_entry.ncols() != inputs.time_exit.ncols()
            || inputs.time_entry.ncols() != inputs.time_derivative.ncols()
        {
            return Err(SurvivalError::DimensionMismatch);
        }
        Self::validate_penalties(&penalties, p)?;
        Self::validate_common_inputs(
            &inputs.age_entry,
            &inputs.age_exit,
            &inputs.event_target,
            &inputs.event_competing,
            &inputs.sampleweight,
        )?;
        if inputs.time_entry.iter().any(|v| !v.is_finite())
            || inputs.time_exit.iter().any(|v| !v.is_finite())
            || inputs.time_derivative.iter().any(|v| !v.is_finite())
            || inputs.covariates.iter().any(|v| !v.is_finite())
        {
            return Err(SurvivalError::NonFiniteInput);
        }
        let (offset_eta_entry, offset_eta_exit, offset_derivative_exit) =
            Self::validate_offsets(offsets, n)?;
        let (monotonicity_constraint_rows, monotonicity_constraint_offsets) =
            Self::validate_monotonicity_constraints(
                inputs.monotonicity_constraint_rows,
                inputs.monotonicity_constraint_offsets,
                p,
            )?;

        Ok(Self::finish_construction(
            inputs.age_entry,
            inputs.age_exit,
            inputs.event_target,
            inputs.sampleweight,
            SurvivalDesign::TimeCovariateShared {
                time_entry: inputs.time_entry.to_owned(),
                time_exit: inputs.time_exit.to_owned(),
                time_derivative: inputs.time_derivative.to_owned(),
                covariates: inputs.covariates.to_owned(),
            },
            offset_eta_entry,
            offset_eta_exit,
            offset_derivative_exit,
            penalties,
            monotonicity,
            monotonicity_constraint_rows,
            monotonicity_constraint_offsets,
        ))
    }

    /// Enable/disable monotonic time-block enforcement metadata.
    ///
    /// Monotonicity is enforced through linear inequality constraints on the
    /// derivative design; enabling this records how many leading time columns
    /// belong to that constrained block.
    /// Overwrite the per-block smoothing parameters `λ_k` in place.
    ///
    /// Used by the REML smoothing-parameter selection for transformation
    /// survival fits (issue #563): the outer optimizer proposes a `ρ = log λ`
    /// vector, sets the smoothing blocks' `λ_k` here, and re-runs the inner
    /// constrained PIRLS, so the monotone I-spline baseline can adapt its
    /// wiggliness instead of being pinned at a fixed seed. `lambdas` must have
    /// one entry per penalty block.
    pub fn set_penalty_lambdas(&mut self, lambdas: &[f64]) -> Result<(), EstimationError> {
        if lambdas.len() != self.penalties.blocks.len() {
            crate::bail_invalid_estim!(
                "set_penalty_lambdas expects {} lambdas, got {}",
                self.penalties.blocks.len(),
                lambdas.len()
            );
        }
        for (block, &lambda) in self.penalties.blocks.iter_mut().zip(lambdas.iter()) {
            if !lambda.is_finite() || lambda < 0.0 {
                crate::bail_invalid_estim!("penalty lambda must be finite and >= 0, got {lambda}");
            }
            block.lambda = lambda;
        }
        Ok(())
    }

    pub fn set_structural_monotonicity(
        &mut self,
        enabled: bool,
        time_columns: usize,
    ) -> Result<(), EstimationError> {
        let p = self.coefficient_dim();
        if time_columns > p {
            crate::bail_invalid_estim!(
                "structural time columns {} exceed coefficient dimension {}",
                time_columns,
                p
            );
        }
        if enabled && time_columns == 0 {
            crate::bail_invalid_estim!("structural monotonicity requires at least one time column");
        }
        if enabled {
            const STRUCTURAL_DERIV_TOL: f64 = 1e-12;
            for (i, &offset) in self.offset_derivative_exit.iter().enumerate() {
                if offset < -STRUCTURAL_DERIV_TOL {
                    crate::bail_invalid_estim!(
                        "structural monotonicity requires nonnegative derivative offsets; found offset_derivative_exit[{i}]={offset:.3e}"
                    );
                }
            }
            let mut derivative_row = vec![0.0_f64; p];
            for i in 0..self.nrows() {
                self.fill_derivative_row(i, &mut derivative_row);
                for j in 0..time_columns {
                    let v = derivative_row[j];
                    if v < -STRUCTURAL_DERIV_TOL {
                        crate::bail_invalid_estim!(
                            "structural monotonicity requires nonnegative time-derivative basis entries; found x_derivative[{i},{j}]={v:.3e}"
                        );
                    }
                }
                for j in time_columns..p {
                    let v = derivative_row[j];
                    if v.abs() > STRUCTURAL_DERIV_TOL {
                        crate::bail_invalid_estim!(
                            "structural monotonicity requires zero derivative contribution outside the time block; found x_derivative[{i},{j}]={v:.3e}"
                        );
                    }
                }
            }
            if let (Some(rows), Some(offsets)) = (
                self.monotonicity_constraint_rows.as_ref(),
                self.monotonicity_constraint_offsets.as_ref(),
            ) {
                for (i, &offset) in offsets.iter().enumerate() {
                    if offset < -STRUCTURAL_DERIV_TOL {
                        crate::bail_invalid_estim!(
                            "structural monotonicity requires nonnegative collocation derivative offsets; found monotonicity_constraint_offsets[{i}]={offset:.3e}"
                        );
                    }
                }
                for i in 0..rows.nrows() {
                    for j in 0..time_columns {
                        let v = rows[[i, j]];
                        if v < -STRUCTURAL_DERIV_TOL {
                            crate::bail_invalid_estim!(
                                "structural monotonicity requires nonnegative collocation derivative basis entries; found monotonicity_constraint_rows[{i},{j}]={v:.3e}"
                            );
                        }
                    }
                    for j in time_columns..p {
                        let v = rows[[i, j]];
                        if v.abs() > STRUCTURAL_DERIV_TOL {
                            crate::bail_invalid_estim!(
                                "structural monotonicity requires zero collocation derivative contribution outside the time block; found monotonicity_constraint_rows[{i},{j}]={v:.3e}"
                            );
                        }
                    }
                }
            }
        }
        self.structurally_monotonic = enabled;
        self.structural_time_columns = if enabled { time_columns } else { 0 };
        Ok(())
    }

    pub fn update_state(&self, beta: &Array1<f64>) -> Result<WorkingState, EstimationError> {
        if beta.len() != self.coefficient_dim() {
            crate::bail_invalid_estim!("survival beta dimension mismatch");
        }

        let n = self.nrows();
        let p = self.coefficient_dim();

        // Royston-Parmar contract used throughout the engine:
        //   eta(t) = log(H(t)), where H(t) is cumulative hazard.
        //
        // With row-vectors (per subject i):
        //   a1_i^T := x_exit_i^T,  a0_i^T := x_entry_i^T,  d_i^T := x_derivative_i^T
        // and scalars:
        //   eta1_i = a1_i^T beta,  eta0_i = a0_i^T beta,  s_i = d_i^T beta.
        //
        // The per-subject negative log-likelihood used below is
        //   NLL_i(beta) = exp(eta1_i) - exp(eta0_i) - delta_i * (eta1_i + log(s_i)),
        // with delta_i = event_target_i.
        //
        // This is exactly the form whose derivatives are:
        //   grad_i = exp(eta1_i) a1_i - exp(eta0_i) a0_i - delta_i * (a1_i + d_i / s_i)
        //   Hess_i = exp(eta1_i) a1_i a1_i^T - exp(eta0_i) a0_i a0_i^T
        //            + delta_i * (d_i d_i^T) / s_i^2.
        //
        // Monotonicity is enforced through linear inequality constraints on the
        // derivative design. This keeps the baseline smoothing penalty on the
        // actual spline coefficients and preserves zero-deviation as beta=0.
        //
        // The loop below computes exact beta-space derivatives and then adds penalties.
        // Total predictor = target offset + learned deviation.
        // This is the same architecture used for flexible binary links:
        // principled default, plus penalized wiggle/deviation.
        let eta_entry = self.entry_dot(beta) + &self.offset_eta_entry;
        let eta_exit = self.exit_dot(beta) + &self.offset_eta_exit;
        let derivative_raw = self.derivative_dot(beta) + &self.offset_derivative_exit;

        let mut nll = 0.0;
        let derivative_guard = self.derivative_guard();
        let derivative_guard_numerical = self.derivative_guard_numerical();
        let mut workspace = self
            .workspace
            .lock()
            .expect("survival workspace mutex was not poisoned by a panicking holder");
        workspace.reset(n);
        let SurvivalWorkspace {
            w_event,
            w_event_inv_deriv,
            w_event_outer,
            w_hess_exit,
            w_hess_entry,
        } = &mut *workspace;

        // Phase 1: Scalar loop — compute per-observation weights, NLL, validation.
        for i in 0..n {
            let w = self.sampleweight[i];
            if w <= 0.0 {
                continue;
            }
            let entry_age = self.age_entry[i];
            let exit_age = self.age_exit[i];
            if !entry_age.is_finite() || !exit_age.is_finite() || exit_age < entry_age {
                crate::bail_invalid_estim!(
                    "survival ages must be finite with age_exit >= age_entry"
                );
            }
            let d = f64::from(self.event_target[i]);

            let has_entry_interval = !self.entry_at_origin[i];
            let interval_scale = if has_entry_interval {
                eta_exit[i].max(eta_entry[i])
            } else {
                eta_exit[i]
            };
            let h_e_scaled = (eta_exit[i] - interval_scale).exp();
            let h_s_scaled = if has_entry_interval {
                (eta_entry[i] - interval_scale).exp()
            } else {
                0.0
            };
            let interval_scaled = h_e_scaled - h_s_scaled;
            let interval = Self::scaled_exp_component(interval_scale, interval_scaled)?;
            let (deriv, deriv_slope) = self
                .stabilized_structural_derivative(derivative_raw[i])
                .unwrap_or((derivative_raw[i], 1.0));
            // Monotonicity of η(t) = log H(t) is a structural property of the
            // whole Royston-Parmar spline. If d_eta/dt is *strictly negative*
            // at any observed exit time, the cumulative hazard H(t) decreases
            // there and S(t) is not a valid survival function — both event
            // and censored rows have to refuse that case. Event rows further
            // need deriv strictly above the numerical guard because their
            // NLL contains `deriv.ln()` and `1.0 / deriv`; censored rows do
            // not, so a boundary value of exactly zero is feasible there.
            let mono_floor = if d > 0.0 {
                derivative_guard_numerical
            } else {
                0.0
            };
            if !deriv.is_finite() || deriv < mono_floor {
                return Err(EstimationError::ParameterConstraintViolation(format!(
                    "survival monotonicity violated at row {}: d_eta/dt={:.3e} <= tolerance={:.3e}",
                    i, deriv, derivative_guard
                )));
            }
            if has_entry_interval {
                let increment_guard = self.interval_increment_guard(h_s_scaled, h_e_scaled);
                if interval_scaled + increment_guard < 0.0 {
                    return Err(EstimationError::ParameterConstraintViolation(format!(
                        "survival cumulative hazard decreased over row {}: H(exit)-H(entry)={:.6e}",
                        i, interval
                    )));
                }
            }
            nll += w * interval;

            // Per-observation weights for BLAS phase.
            // scaled_exp_component(interval_scale, h_e_scaled * x[r]) = exp(interval_scale) * h_e_scaled * x[r]
            // so the Hessian weight is w * exp(interval_scale) * h_e_scaled = w * exp(eta_exit).
            let w_exit_i = w * eta_exit[i].exp();
            let w_entry_i = if has_entry_interval {
                w * eta_entry[i].exp()
            } else {
                0.0
            };
            if !w_exit_i.is_finite() {
                crate::bail_invalid_estim!(
                    "survival interval term exceeds f64 range at row {i} (w*exp(eta_exit)={w_exit_i:.3e})"
                );
            }
            w_hess_exit[i] = w_exit_i;
            w_hess_entry[i] = w_entry_i;

            if d > 0.0 {
                // `deriv_slope` is the derivative of the structural clamp: on
                // the floored branch the NLL's `ln(deriv)` term is locally
                // constant in β, so its score and curvature channels vanish.
                let inv_deriv = deriv_slope / deriv;
                nll += -w * (eta_exit[i] + deriv.ln());
                w_event[i] = w;
                w_event_inv_deriv[i] = w * inv_deriv;
                w_event_outer[i] = w * inv_deriv * inv_deriv;
            }
        }

        // Phase 2: BLAS-accelerated Hessian and gradient via faer.
        //   H_interval = X_exit^T diag(w_exit) X_exit - X_entry^T diag(w_entry) X_entry
        //   grad_interval = X_exit^T w_exit - X_entry^T w_entry
        let mut h = self.interval_hessian_blas(w_hess_exit, w_hess_entry);
        // At large smoothing penalties the event-Jacobian score nearly cancels
        // the interval score. Compensated row accumulation keeps the final KKT
        // residual accurate enough for the outer LAML envelope check.
        let mut grad = Array1::<f64>::zeros(p);
        let mut grad_comp = Array1::<f64>::zeros(p);
        let mut row_exit = vec![0.0_f64; p];
        let mut row_entry = vec![0.0_f64; p];
        let mut row_derivative = vec![0.0_f64; p];
        for i in 0..n {
            let w_interval_exit = w_hess_exit[i];
            let w_interval_entry = w_hess_entry[i];
            let w_event_exit = w_event[i];
            let w_event_derivative = w_event_inv_deriv[i];
            if w_interval_exit == 0.0
                && w_interval_entry == 0.0
                && w_event_exit == 0.0
                && w_event_derivative == 0.0
            {
                continue;
            }
            self.fill_exit_row(i, &mut row_exit);
            self.fill_entry_row(i, &mut row_entry);
            self.fill_derivative_row(i, &mut row_derivative);
            for j in 0..p {
                let contribution = w_interval_exit * row_exit[j]
                    - w_interval_entry * row_entry[j]
                    - w_event_exit * row_exit[j]
                    - w_event_derivative * row_derivative[j];
                let t = grad[j] + contribution;
                if grad[j].abs() >= contribution.abs() {
                    grad_comp[j] += (grad[j] - t) + contribution;
                } else {
                    grad_comp[j] += (contribution - t) + grad[j];
                }
                grad[j] = t;
            }
        }
        grad += &grad_comp;

        h += &self.derivative_xt_diag_x(w_event_outer);

        // Norm of the unpenalized score, captured before adding the penalty
        // contribution, for the scale-invariant convergence certificate
        // (||score||_2 + ||S*beta||_2).
        let score_norm = array1_l2_norm(&grad);

        let penaltygrad = self.penalties.gradient(beta);
        // The WorkingState contract (`gam_solve::pirls::WorkingState`) defines
        // `penalty_term` as the FULL quadratic form βᵀSβ: the shared LM objective
        // is `½(deviance + penalty_term)` (reweight.rs), so the penalty ENERGY is
        // `½·penalty_term`. `PenaltyBlocks::deviance` returns the energy `½βᵀSβ`, so
        // the full quadratic is twice it. Storing the energy directly here (as this
        // path historically did) made the LM's accept/reject objective read
        // `½·deviance + ¼βᵀSβ` — under-penalized by 2× against the `Sβ` step gradient
        // in `gradient` below — desyncing objective and step into a non-convergent
        // Levenberg–Marquardt crawl at large λ (#2301 defect B). Downstream survival
        // consumers that want the energy read `½·penalty_term`.
        let penalty_quadratic_form = 2.0 * self.penalties.deviance(beta);
        let penaltygrad_norm = array1_l2_norm(&penaltygrad);

        let mut totalgrad = grad;
        totalgrad += &penaltygrad;

        self.penalties.addhessian_inplace(&mut h);
        // No coefficient ridge is fused into this objective. Indefinite or
        // rank-deficient curvature along the Newton path is the SOLVER's
        // problem: the working-model driver applies Levenberg–Marquardt
        // damping (H + λD²)δ = −g that vanishes at convergence, so the
        // converged estimator is a stationary point of the exact penalized
        // likelihood and is invariant under coefficient rescaling.
        let log_likelihood = -nll;
        let deviance = 2.0 * nll;

        Ok(WorkingState {
            eta: LinearPredictor::new(eta_exit),
            gradient: totalgrad,
            hessian: gam_linalg::matrix::SymmetricMatrix::Dense(h),
            log_likelihood,
            deviance,
            penalty_term: penalty_quadratic_form,
            firth: gam_solve::pirls::FirthDiagnostics::Inactive,
            ridge_used: 0.0,
            hessian_curvature: gam_solve::pirls::HessianCurvatureKind::Observed,
            gradient_natural_scale: score_norm + penaltygrad_norm,
        })
    }

    /// Compute the third-derivative correction matrix for a given mode response `u_k`.
    ///
    /// This is the directional derivative of the unpenalized NLL Hessian w.r.t.
    /// beta along direction `u_k = -H^{-1} A_k beta_hat`. The returned matrix B
    /// satisfies `dH/drho_k = A_k + B`.
    ///
    /// Called via [`SurvivalDerivProvider`] which adapts the sign convention
    /// from the unified `HessianDerivativeProvider` trait (positive `v_k`) to
    /// the negated `u_k` used here.
    pub(crate) fn survival_hessian_derivative_correction(
        &self,
        beta: &Array1<f64>,
        u_k: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        let p = beta.len();
        let n = self.nrows();

        let eta_entry = self.entry_dot(beta) + &self.offset_eta_entry;
        let eta_exit = self.exit_dot(beta) + &self.offset_eta_exit;
        let deriv_raw = self.derivative_dot(beta) + &self.offset_derivative_exit;
        let exp_entry = eta_entry.mapv(f64::exp);
        let exp_exit = eta_exit.mapv(f64::exp);
        let guard = self.derivative_guard();
        let guard_numerical = self.derivative_guard_numerical();

        let jac = Array1::<f64>::ones(p);
        let curvature = Array1::<f64>::zeros(p);
        let third = Array1::<f64>::zeros(p);

        let mut row_exit = vec![0.0_f64; p];
        let mut row_entry = vec![0.0_f64; p];
        let mut row_derivative = vec![0.0_f64; p];
        let mut ge = vec![0.0_f64; p];
        let mut gs = vec![0.0_f64; p];
        let mut gsd = vec![0.0_f64; p];
        let mut he = vec![0.0_f64; p];
        let mut hs = vec![0.0_f64; p];
        let mut hsd = vec![0.0_f64; p];
        let mut te = vec![0.0_f64; p];
        let mut ts = vec![0.0_f64; p];
        let mut tsd = vec![0.0_f64; p];

        let mut b_dir = Array2::<f64>::zeros((p, p));

        for i in 0..n {
            let w_i = self.sampleweight[i];
            if w_i <= 0.0 {
                continue;
            }
            let has_entry = !self.entry_at_origin[i];
            let mut deta_e = 0.0_f64;
            let mut deta_s = 0.0_f64;
            let mut ds = 0.0_f64;
            self.fill_exit_row(i, &mut row_exit);
            self.fill_entry_row(i, &mut row_entry);
            self.fill_derivative_row(i, &mut row_derivative);
            for j in 0..p {
                ge[j] = row_exit[j] * jac[j];
                gs[j] = row_entry[j] * jac[j];
                gsd[j] = row_derivative[j] * jac[j];
                he[j] = row_exit[j] * curvature[j];
                hs[j] = row_entry[j] * curvature[j];
                hsd[j] = row_derivative[j] * curvature[j];
                te[j] = row_exit[j] * third[j];
                ts[j] = row_entry[j] * third[j];
                tsd[j] = row_derivative[j] * third[j];
                deta_e += ge[j] * u_k[j];
                if has_entry {
                    deta_s += gs[j] * u_k[j];
                }
                ds += gsd[j] * u_k[j];
            }

            // Interval part: d/dbeta [ exp(eta) * (g g^T + diag(h)) ][u_k]
            for r in 0..p {
                let dge_r = he[r] * u_k[r];
                let dgs_r = hs[r] * u_k[r];
                let dhe_r = te[r] * u_k[r];
                let dhs_r = ts[r] * u_k[r];
                for c in 0..p {
                    let dge_c = he[c] * u_k[c];
                    let dgs_c = hs[c] * u_k[c];
                    let mut d_h_rc =
                        exp_exit[i] * (deta_e * ge[r] * ge[c] + dge_r * ge[c] + ge[r] * dge_c);
                    if r == c {
                        d_h_rc += exp_exit[i] * (deta_e * he[r] + dhe_r);
                    }
                    if has_entry {
                        d_h_rc -=
                            exp_entry[i] * (deta_s * gs[r] * gs[c] + dgs_r * gs[c] + gs[r] * dgs_c);
                        if r == c {
                            d_h_rc -= exp_entry[i] * (deta_s * hs[r] + dhs_r);
                        }
                    }
                    b_dir[[r, c]] += w_i * d_h_rc;
                }
            }

            // Event part: d/dbeta [ gsd gsd^T / s^2 - diag(he) - diag(hsd / s) ][u_k]
            let (s_i, s_slope) = self
                .stabilized_structural_derivative(deriv_raw[i])
                .unwrap_or((deriv_raw[i], 1.0));
            if !s_i.is_finite() {
                return Err(EstimationError::ParameterConstraintViolation(format!(
                    "survival monotonicity violated in unified trace contraction at row {i}: \
                     d_eta/dt={s_i:.3e} <= tolerance={guard:.3e}",
                )));
            }
            if self.event_target[i] > 0 && s_slope != 0.0 {
                // On the floored clamp branch (slope 0) the event Hessian
                // block is identically zero in a neighborhood of β, so its
                // directional derivative vanishes and the whole event part is
                // skipped.
                if s_i < guard_numerical {
                    return Err(EstimationError::ParameterConstraintViolation(format!(
                        "survival monotonicity violated in unified trace contraction at row {i}: \
                         d_eta/dt={s_i:.3e} <= tolerance={guard:.3e}",
                    )));
                }
                let inv_s = 1.0 / s_i;
                let inv_s2 = inv_s * inv_s;
                let inv_s3 = inv_s2 * inv_s;
                for r in 0..p {
                    let dgd_r = hsd[r] * u_k[r];
                    let dtsd_r = tsd[r] * u_k[r];
                    let dte_r = te[r] * u_k[r];
                    for c in 0..p {
                        let dgd_c = hsd[c] * u_k[c];
                        let mut d_h_rc = (dgd_r * gsd[c] + gsd[r] * dgd_c) * inv_s2
                            - 2.0 * gsd[r] * gsd[c] * ds * inv_s3;
                        if r == c {
                            d_h_rc += -dte_r;
                            d_h_rc += -(dtsd_r * inv_s - hsd[r] * ds * inv_s2);
                        }
                        b_dir[[r, c]] += w_i * d_h_rc;
                    }
                }
            }
        }

        Ok(b_dir)
    }

    /// Per-observation gradients of the unpenalized survival NLL with respect
    /// to each additive offset channel, at the given β.
    ///
    /// Contract (Royston-Parmar, eta = log H(t)):
    ///
    ///   NLL_i(β; o_E, o_X, o_D) = w_i · [
    ///       exp(η1_i) − 1{has_entry}·exp(η0_i)
    ///       − δ_i · (η1_i + log s_i)
    ///   ]
    ///
    /// with η1_i = a1_iᵀβ + o_X\[i\], η0_i = a0_iᵀβ + o_E\[i\],
    ///      s_i  = d_iᵀβ + o_D\[i\].
    ///
    /// The additive offsets enter each of the three η channels linearly, so
    ///   ∂NLL_i/∂o_X\[i\] = w_i · (exp(η1_i) − δ_i)
    ///   ∂NLL_i/∂o_E\[i\] = −w_i · exp(η0_i) · 1{has_entry_interval}
    ///   ∂NLL_i/∂o_D\[i\] = −w_i · δ_i / s_i         (event-row only)
    ///
    /// These three arrays are the sampleweight-scaled residuals used to chain
    /// `∂NLL/∂offset` into `∂NLL/∂θ` via any closed-form `∂offset/∂θ` map
    /// (see `baseline_offset_theta_partials` for parametric baselines). At
    /// converged β*, the envelope theorem on the penalized objective gives
    ///
    ///   d[0.5·(deviance + β*ᵀS_λβ*)] / dθ
    ///     = Σᵢ r_X_i·∂o_X_i/∂θ + r_E_i·∂o_E_i/∂θ + r_D_i·∂o_D_i/∂θ
    ///
    /// exactly (no IFT back-solve required), because β* is a stationary point
    /// of the penalized objective wrt β and the penalty has no θ dependence.
    ///
    /// Rows with `sampleweight[i] ≤ 0` and non-event rows for `r_D` are
    /// returned as exact 0.0 so the output can be dot-producted against a
    /// per-obs baseline-partials array without a mask.
    ///
    /// Structural-monotonicity stabilization on `s_i` (see
    /// `stabilized_structural_derivative`) is applied identically to the
    /// existing `update_state` path so the residual agrees with the
    /// NLL that `update_state` evaluates.
    pub fn offset_channel_residuals(
        &self,
        beta: &Array1<f64>,
    ) -> Result<OffsetChannelResiduals, EstimationError> {
        if beta.len() != self.coefficient_dim() {
            crate::bail_invalid_estim!(
                "survival beta dimension mismatch in offset_channel_residuals"
            );
        }
        let n = self.nrows();
        let eta_entry = self.entry_dot(beta) + &self.offset_eta_entry;
        let eta_exit = self.exit_dot(beta) + &self.offset_eta_exit;
        let derivative_raw = self.derivative_dot(beta) + &self.offset_derivative_exit;

        let derivative_guard_numerical = self.derivative_guard_numerical();
        let mut r_exit = Array1::<f64>::zeros(n);
        let mut r_entry = Array1::<f64>::zeros(n);
        let mut r_deriv = Array1::<f64>::zeros(n);

        for i in 0..n {
            let w = self.sampleweight[i];
            if w <= 0.0 {
                continue;
            }
            let entry_age = self.age_entry[i];
            let exit_age = self.age_exit[i];
            if !entry_age.is_finite() || !exit_age.is_finite() || exit_age < entry_age {
                crate::bail_invalid_estim!(
                    "survival ages must be finite with age_exit >= age_entry"
                );
            }
            let has_entry_interval = !self.entry_at_origin[i];
            let d = f64::from(self.event_target[i]);
            // Phase-1 values matching update_state:
            //   w_exit_i  = w · exp(eta_exit)                    → ∂NLL/∂o_X before − δ·w term
            //   w_entry_i = w · exp(eta_entry) · 1{has_entry}    → matches −∂NLL/∂o_E sign
            let w_exit_i = w * eta_exit[i].exp();
            let w_entry_i = if has_entry_interval {
                w * eta_entry[i].exp()
            } else {
                0.0
            };
            if !w_exit_i.is_finite() {
                crate::bail_invalid_estim!(
                    "offset_channel_residuals: w*exp(eta_exit)={w_exit_i:.3e} non-finite at row {i}"
                );
            }
            r_exit[i] = w_exit_i - d * w;
            r_entry[i] = -w_entry_i;
            // Same per-row monotonicity rule as `update_state`: a strictly
            // negative derivative at any observed exit time (event or
            // censored) falsifies S(t); event rows additionally need
            // `deriv > guard` because `1/deriv` enters their score.
            let deriv_raw = derivative_raw[i];
            let (deriv, deriv_slope) = self
                .stabilized_structural_derivative(deriv_raw)
                .unwrap_or((deriv_raw, 1.0));
            let mono_floor = if d > 0.0 {
                derivative_guard_numerical
            } else {
                0.0
            };
            if !deriv.is_finite() || deriv < mono_floor {
                return Err(EstimationError::ParameterConstraintViolation(format!(
                    "offset_channel_residuals: derivative ≤ numerical guard at row {i}: {deriv:.3e}"
                )));
            }
            if d > 0.0 {
                // The clamp slope zeroes the residual on the floored branch,
                // matching update_state's flat `ln(deriv)` value there.
                r_deriv[i] = -w * d * deriv_slope / deriv;
            }
        }

        let right = Array1::<f64>::zeros(r_exit.len());
        Ok(OffsetChannelResiduals {
            exit: r_exit,
            entry: r_entry,
            derivative: r_deriv,
            right,
        })
    }

    /// Build an [`InnerSolution`](gam_solve::estimate::reml::reml_outer_engine::InnerSolution) from
    /// the survival working state, suitable for the unified REML/LAML evaluator.
    ///
    /// Evaluate the survival outer objective and gradient via the unified REML/LAML
    /// evaluator, using the canonical assembly module.
    pub fn unified_lamlobjective_and_rhogradient(
        &self,
        beta: &Array1<f64>,
        state: &WorkingState,
        rho: &Array1<f64>,
    ) -> Result<(f64, Array1<f64>), EstimationError> {
        use gam_problem::{EvalMode, PseudoLogdetMode};
        use gam_solve::estimate::reml::assembly::InnerAssembly;
        use gam_solve::estimate::reml::reml_outer_engine::{
            DenseSpectralOperator, DispersionHandling,
        };
        use gam_solve::estimate::reml::reparameterized_inner::{
            RawInnerReparamContext, assemble_reparameterized_inner,
        };
        use gam_terms::construction::{
            canonicalize_penalty_specs, precompute_reparam_invariant_from_canonical,
            stable_reparameterizationwith_invariant,
        };
        use gam_terms::penalty_spec::PenaltySpec;

        let p = beta.len();
        let active_penalty_blocks: Vec<&PenaltyBlock> = self
            .penalties
            .blocks
            .iter()
            .filter(|b| b.lambda > 0.0)
            .collect();
        if rho.len() != active_penalty_blocks.len() {
            crate::bail_invalid_estim!(
                "survival LAML rho dimension {} does not match active penalty block count {}",
                rho.len(),
                active_penalty_blocks.len()
            );
        }
        let k_count = active_penalty_blocks.len();

        // The Laplace/LAML envelope is defined at a stationary inner mode.
        // Certify that primary precondition before constructing penalty roots,
        // reparameterizing, or factorizing the observed Hessian. An arbitrary
        // off-mode point can also be indefinite, but it is not yet a candidate
        // Laplace mode; reporting or computing the secondary spectral condition
        // first both obscures the root refusal and wastes the expensive setup. A
        // one-step Newton residual surrogate is not an interchangeable
        // criterion when the likelihood Hessian moves with beta: its scalar
        // moving-Hessian response requires higher-order rho derivatives that
        // the survival provider does not emit. Attaching that partial surrogate
        // made the value and gradient different functions at rho=4 (#2491).
        //
        // Certify the active-set-projected KKT condition in the RAW frame. A
        // binding monotonicity constraint contributes r = A^T lambda and must
        // be projected out before the stationarity decision. Once certified,
        // the exact envelope has no residual term; the transformed assembly
        // therefore carries kkt_residual=None deliberately.
        let projected_norm = {
            let raw = state.gradient.clone();
            let projected = match self.monotonicity_linear_constraints() {
                Some(constraints) => {
                    let constraints = ConstraintSet::Dense(constraints);
                    projected_linear_constraint_stationarity_vector(&raw, beta, &constraints, None)
                        .ok_or_else(|| {
                            EstimationError::InvalidInput(
                                "survival LAML could not project the monotonicity KKT residual"
                                    .to_string(),
                            )
                        })?
                }
                None => raw,
            };
            array1_l2_norm(&projected)
        };
        // Accept exactly what the inner solver certified. `certifies_kkt` is the
        // producer's own convergence predicate: it accepts under EITHER the
        // dimension bound `tol·√(n·p)` or the natural-scale bound
        // `tol·(1+‖g‖_scale)`. Re-deriving only the natural-scale half here made
        // this gate strictly stronger than the contract PIRLS was run under, so
        // every mode that converged via the dimension branch — the binding one
        // whenever √(n·p) > 1+scale, i.e. the normal case for these baselines —
        // was certified by the solver and then refused here, fatally. Same
        // tolerance, same rule, one owner.
        if !projected_norm.is_finite()
            || !state.certifies_kkt(projected_norm, SURVIVAL_LAML_STATIONARITY_RELATIVE_TOL)
        {
            // DECLINING is right and stays: a one-step residual surrogate is
            // not a differentiable substitute for a Laplace mode, so this
            // trial has no LAML value to report. What was wrong is the KIND of
            // refusal. `InvalidInput` is a statement about the configuration,
            // and `is_trial_point_infeasible` answers `false` for it, so the
            // outer lambda-search — whose documented response to "no value
            // here" is to mark the point infeasible and step away — aborted
            // the whole fit at the first such rho instead (#2531, and #1123's
            // own closed guard went red on it). The inner mode's stationarity
            // is a property of THIS rho: the same design at a neighbouring
            // lambda converges perfectly well, which is why the CLI fits it.
            return Err(EstimationError::TrialPointRefused {
                reason: format!(
                    "survival LAML requires a stationary inner mode: projected KKT residual \
                     {projected_norm:.3e} (relative {:.3e}) is not certified by the inner \
                     solver's convergence test at tolerance \
                     {SURVIVAL_LAML_STATIONARITY_RELATIVE_TOL:.3e}; a one-step residual \
                     surrogate is not a differentiable substitute for the Laplace mode",
                    state.relative_gradient_norm(projected_norm)
                ),
            });
        }

        // λ_k = e^{ρ_k}, in active-block (== ρ) order. Shared by the joint
        // normalizer and the Wood reparameterization below.
        let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();

        // --- Raw penalized Hessian + its LAML logdet mode -------------------
        // Survival LAML is a Laplace approximation at the fitted inner mode.
        // Its observed penalized Hessian must therefore be positive definite on
        // the fitted coefficient space. Delayed entry can make trial Hessians
        // indefinite; such a trial is not a Laplace mode and is refused instead
        // of being converted into a positive-subspace pseudo-objective. The
        // exact positive spectrum is also the full IFT system for beta_rho, so
        // value, trace, and mode response now use one Hessian (#2491).
        let h_dense = state.hessian.to_dense();
        let hessian_logdet_mode = PseudoLogdetMode::PositiveDefinite;

        // --- Raw per-block penalties, embedded p×p, in ρ order --------------
        // Feeds the joint pseudo-logdet `log|Σ_k λ_k S_k|₊` (frame-invariant,
        // computed INSIDE the reparam helper from these raw blocks — #2331 R7:
        // value/det1/det2 from a single W-factor eigendecomposition, not a
        // per-block sum; the survival stabilization ridge is a full-span block
        // that overlaps the smoothing blocks, so the joint normalizer is the
        // real objective, #2331 Finding 3a).
        let s_k_embedded: Vec<Array2<f64>> = active_penalty_blocks
            .iter()
            .map(|b| {
                let mut s = Array2::<f64>::zeros((p, p));
                let (rs, re) = (b.range.start, b.range.end);
                s.slice_mut(ndarray::s![rs..re, rs..re]).assign(&b.matrix);
                s
            })
            .collect();

        // --- Wood (2011) stable reparameterization at the outer LAML seam ---
        // (#2331 Inc 2). Build the orthogonal Q_s from the SAME active penalty
        // set, in ρ order, so `canonical_transformed[k] ↔ ρ[k]`.
        //
        // λ-DEPENDENCE / recompute policy:
        //   * `canonical_penalties` and `reparam_invariant` (the subspace split /
        //     q_pen|q_null basis `qs_base`) depend ONLY on the penalty block
        //     matrices+ranges — λ-INDEPENDENT, fixed across the whole outer loop
        //     (a future WorkingModelSurvival cache can memoize them; see the cost
        //     note in the Inc-2 handoff).
        //   * `stable_reparameterizationwith_invariant` embeds λ = e^{ρ} into
        //     `e_transformed` / `canonical_transformed`, so it MUST be recomputed
        //     at every outer iterate. This is the only λ-dependent piece here.
        //
        // canonicalize_penalty_spec drops a block only when it is numerically
        // rank-0; a λ>0 survival penalty (I-spline 2nd-difference, incl. the
        // rank-deficient nullspace ones) is rank>0, so no active block drops. A
        // drop would desync the transformed penalty coordinates from ρ, so we
        // assert the count is preserved rather than let it slip silently.
        let penalty_specs: Vec<PenaltySpec> = active_penalty_blocks
            .iter()
            .map(|b| PenaltySpec::Block {
                local: b.matrix.clone(),
                col_range: b.range.clone(),
                prior_mean: gam_problem::CoefficientPriorMean::Zero,
                structure_hint: None,
                op: None,
            })
            .collect();
        let nullspace_dims: Vec<usize> = active_penalty_blocks
            .iter()
            .map(|b| b.nullspace_dim)
            .collect();
        let (canonical_penalties, _canonical_nullspace) = canonicalize_penalty_specs(
            &penalty_specs,
            &nullspace_dims,
            p,
            "survival LAML seam-A reparameterization",
        )
        .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;
        if canonical_penalties.len() != k_count {
            return Err(EstimationError::InvalidInput(format!(
                "survival LAML reparameterization dropped {} of {} active (λ>0) penalty \
                 block(s) as numerically rank-0; cannot align transformed penalty \
                 coordinates with ρ",
                k_count - canonical_penalties.len(),
                k_count
            )));
        }
        let reparam_invariant =
            precompute_reparam_invariant_from_canonical(&canonical_penalties, p)
                .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;
        let reparam = stable_reparameterizationwith_invariant(
            &canonical_penalties,
            &lambdas,
            p,
            &reparam_invariant,
        )
        .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;

        // --- Conjugate the raw inner solution into the Q_s frame ------------
        // H′ = Q_sᵀ H Q_s, β̂′ = Q_sᵀ β̂, the joint logdet triple (value/det1/det2),
        // and the SurvivalDerivProvider conjugated into the transformed frame.
        // The provider is built on the RAW β̂ (it evaluates raw-frame family
        // curvature); the helper wraps it so every consumed trace is invariant.
        let provider = SurvivalDerivProvider::new(self.clone(), beta.clone());
        let ctx = RawInnerReparamContext {
            hessian: &h_dense,
            beta,
            penalties_embedded: &s_k_embedded,
            lambdas: &lambdas,
        };
        let reparam_inner = assemble_reparameterized_inner(
            &ctx,
            Some(Box::new(provider)),
            &reparam,
        )
        .map_err(EstimationError::InvalidInput)?;

        // Hessian operator on the transformed H′. Orthogonal similarity
        // preserves strict positive definiteness and the exact spectrum.
        // An indefinite inner Hessian here is a property of THIS rho, not of the
        // problem: `H_pen = H + S(lambda)` and the active penalty set includes a
        // full-span stabilization ridge block, so a neighbouring lambda adds PSD
        // curvature on the offending direction and the same design evaluates
        // fine. `InvalidInput` is classified FATAL by `is_trial_point_infeasible`,
        // so raising it aborted the entire fit at the first indefinite trial
        // instead of letting the outer lambda-search mark the point infeasible
        // and step away -- on a 6-row delayed-entry design that is the whole
        // difference between a fit and
        //
        //   positive-definite Hessian required for Laplace evaluation:
        //   eigenvalue 0 is -1.034960e-2
        //
        // The sibling refusal 130 lines above already draws this distinction for
        // exactly the same reason (#2531); this call site was left behind. The
        // positive-definiteness REQUIREMENT is unchanged -- a saddle is still not
        // a Laplace mode and is still refused. Only the blast radius of the
        // refusal is corrected.
        let hop = DenseSpectralOperator::from_symmetric_with_mode(
            &reparam_inner.hessian_transformed,
            hessian_logdet_mode,
        )
        .map_err(|reason| EstimationError::TrialPointRefused {
            reason: format!(
                "survival LAML requires a positive-definite inner Hessian at this rho: {reason}"
            ),
        })?;

        // Penalty coordinates: the per-block TRANSFORMED roots (Q_s frame), the
        // single source of truth for penalty roots there. One coordinate per ρ
        // block, in order (canonical_transformed is built root-by-root from the
        // input penalties, so its order matches ρ). This is exactly the standard
        // lane's `reparam_result.canonical_transformed → to_penalty_coordinate`
        // path (gam-solve reml/objective.rs build_penalty_coords).
        let penalty_coords = reparam
            .canonical_transformed
            .iter()
            .map(|cp| cp.to_penalty_coordinate())
            .collect::<Vec<_>>();

        // `penalty_term` is the full quadratic βᵀSβ (WorkingState contract),
        // which IS the unified evaluator's `penalty_quadratic` (#2301 defect B).
        // It is a scalar and invariant under the orthogonal transform
        // (β′ᵀS′β′ = βᵀSβ), so it carries over unchanged.
        let penalty_quadratic = state.penalty_term;

        let result = InnerAssembly {
            log_likelihood: state.log_likelihood,
            penalty_quadratic,
            beta: reparam_inner.beta_transformed,
            n_observations: self.nrows(),
            hessian_op: std::sync::Arc::new(hop),
            // The survival lane prices its logdet on the operator its inner
            // solve converged against (#2612).
            mode_response_op: None,
            penalty_coords,
            penalty_logdet: reparam_inner.penalty_logdet,
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            },
            rho_curvature_scale: 1.0,
            rho_prior: gam_problem::RhoPrior::Flat,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            deriv_provider: reparam_inner.deriv_provider,
            firth: None,
            nullspace_dim: None,
            barrier_config: None,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            kkt_residual: None,
            active_constraints: None,
        }
        .evaluate(
            rho.as_slice().expect("rho must be contiguous"),
            EvalMode::ValueAndGradient,
            None,
        )
        .map_err(EstimationError::InvalidInput)?;

        let gradient = result.gradient.unwrap_or_else(|| Array1::zeros(rho.len()));
        Ok((result.cost, gradient))
    }

}

/// Derivative provider that adapts survival third-derivative Hessian corrections
/// to the unified [`HessianDerivativeProvider`](gam_solve::estimate::reml::reml_outer_engine::HessianDerivativeProvider)
/// trait.
///
/// The unified trait supplies `v_k = H^{-1}(A_k beta_hat)` (positive sign),
/// whereas the survival engine's
/// [`survival_hessian_derivative_correction`](WorkingModelSurvival::survival_hessian_derivative_correction)
/// expects `u_k = -v_k`. This provider handles the sign conversion.
pub(crate) struct SurvivalDerivProvider {
    model: WorkingModelSurvival,
    beta: Array1<f64>,
}

impl SurvivalDerivProvider {
    pub(crate) fn new(model: WorkingModelSurvival, beta: Array1<f64>) -> Self {
        Self { model, beta }
    }
}

impl gam_solve::estimate::reml::reml_outer_engine::HessianDerivativeProvider
    for SurvivalDerivProvider
{
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // The trait provides v_k = H^{-1}(A_k beta_hat) (positive).
        // The survival method expects u_k = -H^{-1} A_k beta_hat = -v_k.
        let u_k = -v_k;
        match self
            .model
            .survival_hessian_derivative_correction(&self.beta, &u_k)
        {
            Ok(correction) => Ok(Some(correction)),
            Err(e) => Err(e.to_string()),
        }
    }

    fn has_corrections(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone)]
pub struct CrudeRiskResult {
    pub risk: f64,
    pub diseasegradient: Array1<f64>,
    pub mortalitygradient: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct CompetingRisksCifResult {
    /// Cumulative incidence per endpoint. `cif[ep][[row, time_idx]]` is the
    /// probability that cause `ep` occurred by `times[time_idx]` for sample `row`.
    /// Stored one matrix per endpoint so it is ergonomic to index per-cause and
    /// natural to construct from the per-endpoint cumulative-hazard inputs.
    pub cif: Vec<Array2<f64>>,
    pub overall_survival: Array2<f64>,
}

/// Subject-count threshold below which competing-risks CIF assembly stays on the
/// serial path. The per-row work (a `n_times`-long prefix-sum recurrence with a
/// handful of `exp`/`exp_m1` per element) is cheap, so small panels avoid rayon
/// fan-out overhead; large panels (the #1082 quality-test sizes) amortize it.
const COMPETING_RISKS_CIF_PARALLEL_ROW_MIN: usize = 256;

pub fn assemble_competing_risks_cif(
    times: ArrayView1<'_, f64>,
    cumulative_hazard: ArrayView3<'_, f64>,
) -> Result<CompetingRisksCifResult, SurvivalError> {
    let (n_endpoints, n_rows, n_times) = cumulative_hazard.dim();
    if n_endpoints == 0 {
        return Err(SurvivalError::DimensionMismatch);
    }
    let endpoint_hazards = cumulative_hazard
        .axis_iter(Axis(0))
        .map(|view| view.to_owned())
        .collect::<Vec<_>>();
    assemble_competing_risks_cif_from_endpoints(times, &endpoint_hazards).and_then(|result| {
        if result.overall_survival.dim() != (n_rows, n_times) {
            Err(SurvivalError::DimensionMismatch)
        } else {
            Ok(result)
        }
    })
}

pub fn assemble_competing_risks_cif_from_endpoints(
    times: ArrayView1<'_, f64>,
    cumulative_hazards: &[Array2<f64>],
) -> Result<CompetingRisksCifResult, SurvivalError> {
    let n_endpoints = cumulative_hazards.len();
    if n_endpoints == 0 || times.is_empty() {
        return Err(SurvivalError::DimensionMismatch);
    }
    let (n_rows, n_times) = cumulative_hazards[0].dim();
    if n_rows == 0 || n_times == 0 || times.len() != n_times {
        return Err(SurvivalError::DimensionMismatch);
    }
    if times.iter().any(|time| !time.is_finite() || *time < 0.0) {
        return Err(SurvivalError::InvalidTimeGrid);
    }
    if times
        .iter()
        .zip(times.iter().skip(1))
        .any(|(previous, current)| current <= previous)
    {
        return Err(SurvivalError::InvalidTimeGrid);
    }
    for endpoint_hazard in cumulative_hazards {
        if endpoint_hazard.dim() != (n_rows, n_times) {
            return Err(SurvivalError::DimensionMismatch);
        }
        if endpoint_hazard.iter().any(|value| !value.is_finite()) {
            return Err(SurvivalError::NonFiniteInput);
        }
    }

    let max_abs_hazard = cumulative_hazards
        .iter()
        .flat_map(|endpoint_hazard| endpoint_hazard.iter())
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let monotone_tolerance = 1.0e-10_f64 * max_abs_hazard.max(1.0);
    let mut cif: Vec<Array2<f64>> = (0..n_endpoints)
        .map(|_| Array2::<f64>::zeros((n_rows, n_times)))
        .collect();
    let mut overall_survival = Array2::<f64>::zeros((n_rows, n_times));

    // Per-row CIF assembly. The TIME axis is a sequential prefix-sum recurrence
    // (`previous_*` carry forward across `time_idx`) and MUST stay ordered, so it
    // is left as the inner serial loop. The ROW (subject) axis is fully
    // independent: every `previous_*`/`increments` buffer is allocated fresh per
    // row, no state crosses rows, and each row writes only its own disjoint
    // output slices. The per-row result is byte-identical regardless of which
    // thread runs it, so we fan the outer row loop out over rayon and write the
    // owned per-row buffers back serially in row order (deterministic, bit-exact
    // vs. the serial implementation).
    //
    // `cif_flat` is endpoint-major: `cif_flat[endpoint * n_times + time_idx]`.
    let assemble_row = |row: usize| -> Result<(Vec<f64>, Vec<f64>), SurvivalError> {
        let mut cif_flat = vec![0.0_f64; n_endpoints * n_times];
        let mut surv_row = vec![0.0_f64; n_times];
        let mut previous_cif = vec![0.0_f64; n_endpoints];
        let mut previous_cumulative = vec![0.0_f64; n_endpoints];
        let mut increments = vec![0.0_f64; n_endpoints];
        let mut previous_total_cumulative = 0.0_f64;
        for time_idx in 0..n_times {
            let mut total_increment = 0.0_f64;
            for endpoint in 0..n_endpoints {
                let current = cumulative_hazards[endpoint][[row, time_idx]];
                if current < -monotone_tolerance {
                    return Err(SurvivalError::NonMonotoneCumulativeHazard);
                }
                let raw_increment = current - previous_cumulative[endpoint];
                if raw_increment < -monotone_tolerance {
                    return Err(SurvivalError::NonMonotoneCumulativeHazard);
                }
                let increment = raw_increment.max(0.0);
                increments[endpoint] = increment;
                total_increment += increment;
                previous_cumulative[endpoint] += increment;
            }

            let survival_left = (-previous_total_cumulative).exp();
            let interval_failure = -(-total_increment).exp_m1();
            for endpoint in 0..n_endpoints {
                if total_increment > 0.0 {
                    previous_cif[endpoint] +=
                        survival_left * interval_failure * increments[endpoint] / total_increment;
                }
                cif_flat[endpoint * n_times + time_idx] = previous_cif[endpoint].clamp(0.0, 1.0);
            }
            previous_total_cumulative += total_increment;
            // Derive `S(t)` from the stored cause-specific CIFs at this time so
            // that the competing-risks closure identity
            //   Σ_k F_k(t) + S(t) = 1
            // holds bit-exactly. Computing `S` independently as
            // `exp(-Σ_k H_k(t))` and then comparing against the (clamped, ratio-
            // split) Σ F_k introduces O(machine-eps) closure error because the
            // float increments
            //   ΔF_k = S_left·(1-exp(-ΔH))·ΔH_k/ΔH_total
            // do not sum to `S_left - S_new` bit-exactly. By summing the stored
            // CIFs in the same left-fold order as `slice.iter().sum::<f64>()`
            // and defining `S := 1.0 - Σ F_k`, the IEEE-754 round-trip
            //   (1.0 - f) + f
            // restores the identity for finite f ∈ [0, 1]. The mathematically
            // consistent survival value `exp(-H_total)` is still tracked up to
            // ulp-level precision because the ΔF_k construction matches
            // `S_left - S_new` to leading order.
            let mut fsum_at_t = 0.0_f64;
            for endpoint in 0..n_endpoints {
                fsum_at_t += cif_flat[endpoint * n_times + time_idx];
            }
            surv_row[time_idx] = (1.0_f64 - fsum_at_t).clamp(0.0, 1.0);
        }
        Ok((cif_flat, surv_row))
    };

    // Nesting guard (`rayon::current_thread_index().is_none()`) keeps us from
    // oversubscribing when this routine is itself called from inside a rayon
    // worker, and the row-count gate keeps small inputs on the serial path.
    let rows: Vec<(Vec<f64>, Vec<f64>)> = if n_rows >= COMPETING_RISKS_CIF_PARALLEL_ROW_MIN
        && rayon::current_thread_index().is_none()
    {
        use rayon::prelude::*;
        (0..n_rows)
            .into_par_iter()
            .map(assemble_row)
            .collect::<Result<_, _>>()?
    } else {
        (0..n_rows).map(assemble_row).collect::<Result<_, _>>()?
    };

    for (row, (cif_flat, surv_row)) in rows.into_iter().enumerate() {
        for endpoint in 0..n_endpoints {
            for time_idx in 0..n_times {
                cif[endpoint][[row, time_idx]] = cif_flat[endpoint * n_times + time_idx];
            }
        }
        for time_idx in 0..n_times {
            overall_survival[[row, time_idx]] = surv_row[time_idx];
        }
    }

    Ok(CompetingRisksCifResult {
        cif,
        overall_survival,
    })
}

impl PirlsWorkingModel for WorkingModelSurvival {
    fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
        self.update_state(beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, Array3, array, s};

    #[test]
    fn saved_cause_specific_alo_matches_independent_closed_form() {
        let eta_exit = 0.4_f64;
        let eta_entry = -0.3_f64;
        let derivative_exit = 1.7_f64;
        let weight = 2.2_f64;
        let geometry = cause_specific_survival_alo_row_geometry(CauseSpecificSurvivalAloRowInput {
            eta_exit,
            eta_entry,
            derivative_exit,
            prior_weight: weight,
            entry_active: true,
            event: true,
        })
        .expect("valid cause-specific row");
        let expected_nll =
            weight * (eta_exit.exp() - eta_entry.exp() - eta_exit - derivative_exit.ln());
        let expected_score = [
            weight * (eta_exit.exp() - 1.0),
            -weight * eta_entry.exp(),
            -weight / derivative_exit,
        ];
        let expected_hessian = [
            [weight * eta_exit.exp(), 0.0, 0.0],
            [0.0, -weight * eta_entry.exp(), 0.0],
            [0.0, 0.0, weight / derivative_exit.powi(2)],
        ];
        assert!((geometry.negative_log_likelihood - expected_nll).abs() <= 2.0e-14);
        for row in 0..3 {
            assert!((geometry.nll_score[row] - expected_score[row]).abs() <= 2.0e-14);
            for column in 0..3 {
                assert!(
                    (geometry.observed_hessian[row][column] - expected_hessian[row][column]).abs()
                        <= 2.0e-14
                );
            }
        }
        let score_meat = geometry.nll_score[0] * geometry.nll_score[0];
        assert!(
            (geometry.observed_hessian[0][0] - score_meat).abs() > 1.0e-2,
            "survival observed W and empirical score meat C must remain separate"
        );
    }

    /// #932 production single-source parity for the cause-specific Royston-Parmar
    /// derivative tower. The earlier cutover added only a gam-math oracle that
    /// replicated the production `w_exit`/`w_entry`/`w_derivative` weight formulas
    /// verbatim; this module INVOKES production
    /// (`evaluate_cause_specific_block`, `cause_specific_hessian_directional_derivative`,
    /// `cause_specific_hessian_second_directional_derivative`) and pins each
    /// channel against the universal gam-math jet at ≤1e-9, plus an independent
    /// central-difference witness of the live third/fourth against the live lower
    /// order. The generic production program and every specialized channel are
    /// emitted from the same row declaration.
    mod jet_cause_specific_production_parity {
        use super::*;
        use gam_math::jet_tower::{
            program_fourth_contracted, program_row_kernel, program_third_contracted,
        };

        /// A single-row cause-specific block with the design collapsed to the 3×3
        /// identity (`x_exit = e0`, `x_entry = e1`, `x_derivative = e2`, zero
        /// offsets), so β directly parameterises `(η1, η0, s)` and a coefficient
        /// direction IS the predictor-space direction — pinning the per-row
        /// β-space kernels against the jet's predictor-space contractions with no
        /// design projection in the way.
        fn identity_block(w: f64, has_entry: bool, event: bool) -> CauseSpecificRoystonParmarBlock {
            let age_entry = if has_entry { 1.0 } else { 0.0 };
            CauseSpecificRoystonParmarBlock {
                age_entry: array![age_entry],
                age_exit: array![2.0],
                event_target: array![if event { 1u8 } else { 0u8 }],
                sampleweight: array![w],
                x_entry: array![[0.0, 1.0, 0.0]],
                x_exit: array![[1.0, 0.0, 0.0]],
                x_derivative: array![[0.0, 0.0, 1.0]],
                offset_eta_entry: array![0.0],
                offset_eta_exit: array![0.0],
                offset_derivative_exit: array![0.0],
                derivative_floor: 0.0,
                structural_time_columns: 0,
            }
        }

        fn close(hand: f64, jet: f64, tol: f64, label: &str) {
            let band = tol + tol * hand.abs().max(jet.abs());
            assert!(
                (hand - jet).abs() <= band,
                "{label}: hand {hand:+.15e} vs jet {jet:+.15e} (|Δ|={:.3e} band {band:.3e})",
                (hand - jet).abs()
            );
        }

        const JET_TOL: f64 = 1e-9;

        fn run_corner(has_entry: bool, event: bool) {
            // β = (η1, η0, s); s > 0 for the event ln-derivative term.
            let beta = array![0.4_f64, -0.3_f64, 1.3_f64];
            let d_beta = array![0.7_f64, -0.5_f64, 0.6_f64];
            let v_beta = array![-0.2_f64, 0.8_f64, -0.4_f64];
            let w = 1.4_f64;
            let block = identity_block(w, has_entry, event);
            let prog = crate::survival::CauseSpecificRowProgram::new(
                [beta[0], beta[1], beta[2]],
                w,
                has_entry,
                event,
            );
            let label = format!("entry={has_entry} event={event}");

            // ── Value / gradient / Hessian: LIVE evaluate vs jet ──────────────
            let (ll, grad, hess) =
                evaluate_cause_specific_block(&block, &beta).expect("evaluate block");
            let (jet_v, jet_g, jet_h) = program_row_kernel(&prog, 0).expect("jet kernel");
            close(jet_v, -ll, JET_TOL, &format!("{label} value"));
            for a in 0..3 {
                close(jet_g[a], -grad[a], JET_TOL, &format!("{label} grad[{a}]"));
                for b in 0..3 {
                    close(
                        jet_h[a][b],
                        hess[[a, b]],
                        JET_TOL,
                        &format!("{label} H[{a}][{b}]"),
                    );
                }
            }

            // ── Third: LIVE directional derivative vs jet ─────────────────────
            let dh = cause_specific_hessian_directional_derivative(&block, &beta, &d_beta)
                .expect("live third");
            let dir = [d_beta[0], d_beta[1], d_beta[2]];
            let jet_t3 = program_third_contracted(&prog, 0, &dir).expect("jet third");
            for a in 0..3 {
                for b in 0..3 {
                    close(
                        jet_t3[a][b],
                        dh[[a, b]],
                        JET_TOL,
                        &format!("{label} third[{a}][{b}]"),
                    );
                }
            }

            // ── Fourth: LIVE second directional derivative vs jet ─────────────
            let d2h = cause_specific_hessian_second_directional_derivative(
                &block, &beta, &d_beta, &v_beta,
            )
            .expect("live fourth");
            let uu = [d_beta[0], d_beta[1], d_beta[2]];
            let vv = [v_beta[0], v_beta[1], v_beta[2]];
            let jet_t4 = program_fourth_contracted(&prog, 0, &uu, &vv).expect("jet fourth");
            for a in 0..3 {
                for b in 0..3 {
                    close(
                        jet_t4[a][b],
                        d2h[[a, b]],
                        JET_TOL,
                        &format!("{label} fourth[{a}][{b}]"),
                    );
                }
            }

            // ── Independent FD witness (NO jet) ───────────────────────────────
            // ∂_d_beta H via central difference of the LIVE evaluate Hessian.
            let h_fd = 1e-5;
            let bp = &beta + &(&d_beta * h_fd);
            let bm = &beta - &(&d_beta * h_fd);
            let (_, _, hp) = evaluate_cause_specific_block(&block, &bp).expect("evaluate +");
            let (_, _, hm) = evaluate_cause_specific_block(&block, &bm).expect("evaluate -");
            for a in 0..3 {
                for b in 0..3 {
                    let fd = (hp[[a, b]] - hm[[a, b]]) / (2.0 * h_fd);
                    close(dh[[a, b]], fd, 1e-5, &format!("{label} FD third[{a}][{b}]"));
                }
            }
            // ∂_v of the LIVE third (fixed direction d_beta) vs the LIVE fourth.
            let dhp = cause_specific_hessian_directional_derivative(
                &block,
                &bp_along(&beta, &v_beta, h_fd),
                &d_beta,
            )
            .expect("live third +");
            let dhm = cause_specific_hessian_directional_derivative(
                &block,
                &bm_along(&beta, &v_beta, h_fd),
                &d_beta,
            )
            .expect("live third -");
            for a in 0..3 {
                for b in 0..3 {
                    let fd = (dhp[[a, b]] - dhm[[a, b]]) / (2.0 * h_fd);
                    close(
                        d2h[[a, b]],
                        fd,
                        1e-5,
                        &format!("{label} FD fourth[{a}][{b}]"),
                    );
                }
            }
        }

        fn bp_along(beta: &Array1<f64>, v: &Array1<f64>, h: f64) -> Array1<f64> {
            beta + &(v * h)
        }
        fn bm_along(beta: &Array1<f64>, v: &Array1<f64>, h: f64) -> Array1<f64> {
            beta - &(v * h)
        }

        /// The LIVE cause-specific value / gradient / Hessian / third / fourth hand
        /// tower reproduces the universal gam-math jet at ≤1e-9, and the live
        /// third/fourth reproduce an independent central-difference of the live
        /// lower order — across all four (event × entry) corners that gate the
        /// entry and event predictor channels on and off.
        #[test]
        fn cause_specific_live_tower_matches_jet_and_fd() {
            for &has_entry in &[false, true] {
                for &event in &[false, true] {
                    run_corner(has_entry, event);
                }
            }
        }

    }

    #[test]
    fn competing_risks_cif_constant_hazard_matches_closed_form() {
        let times = array![0.0, 2.0, 5.0, 10.0];
        let disease_rates = [0.12, 0.06];
        let death_rates = [0.05, 0.02];
        let cumulative = Array3::from_shape_fn((2, 2, times.len()), |(endpoint, row, time_idx)| {
            let rate = if endpoint == 0 {
                disease_rates[row]
            } else {
                death_rates[row]
            };
            rate * times[time_idx]
        });

        let result =
            assemble_competing_risks_cif(times.view(), cumulative.view()).expect("assemble CIF");

        for row in 0..2 {
            let total_rate = disease_rates[row] + death_rates[row];
            for time_idx in 0..times.len() {
                let failure = 1.0 - (-total_rate * times[time_idx]).exp();
                let expected_disease = disease_rates[row] / total_rate * failure;
                let expected_death = death_rates[row] / total_rate * failure;
                assert!((result.cif[0][[row, time_idx]] - expected_disease).abs() < 1e-12);
                assert!((result.cif[1][[row, time_idx]] - expected_death).abs() < 1e-12);
                assert!(
                    (result.cif[0][[row, time_idx]]
                        + result.cif[1][[row, time_idx]]
                        + result.overall_survival[[row, time_idx]]
                        - 1.0)
                        .abs()
                        < 1e-12
                );
            }
        }
    }

    #[test]
    fn competing_risks_cif_rejects_nonmonotone_hazards() {
        let times = array![0.0, 1.0, 2.0];
        let cumulative = Array3::from_shape_vec((1, 1, 3), vec![0.0, 0.2, 0.1]).expect("shape");
        let err = assemble_competing_risks_cif(times.view(), cumulative.view())
            .expect_err("nonmonotone cumulative hazard should be rejected");
        assert!(matches!(err, SurvivalError::NonMonotoneCumulativeHazard));
    }

    #[test]
    fn competing_risks_cif_plateaus_and_three_causes_conserve_probability() {
        let times = array![0.0, 1.0, 3.0, 7.0, 12.0];
        let cumulative = Array3::from_shape_vec(
            (3, 2, 5),
            vec![
                // cause 1
                0.0, 0.2, 0.2, 0.5, 1.1, 0.0, 0.0, 0.4, 0.4, 0.9, // cause 2
                0.0, 0.1, 0.3, 0.3, 0.7, 0.0, 0.2, 0.2, 0.8, 0.8, // cause 3
                0.0, 0.0, 0.2, 0.6, 0.6, 0.0, 0.1, 0.5, 0.5, 1.5,
            ],
        )
        .expect("shape");

        let result =
            assemble_competing_risks_cif(times.view(), cumulative.view()).expect("assemble CIF");

        for row in 0..2 {
            for time_idx in 0..times.len() {
                let total_cif = result.cif[0][[row, time_idx]]
                    + result.cif[1][[row, time_idx]]
                    + result.cif[2][[row, time_idx]];
                assert!(
                    (total_cif + result.overall_survival[[row, time_idx]] - 1.0).abs() < 1e-12,
                    "probability mass mismatch at row={row}, time_idx={time_idx}"
                );
                assert!((0.0..=1.0).contains(&result.overall_survival[[row, time_idx]]));
                for cause in 0..3 {
                    assert!((0.0..=1.0).contains(&result.cif[cause][[row, time_idx]]));
                    if time_idx > 0 {
                        assert!(
                            result.cif[cause][[row, time_idx]] + 1e-12
                                >= result.cif[cause][[row, time_idx - 1]],
                            "CIF decreased for cause={cause}, row={row}, time_idx={time_idx}"
                        );
                    }
                }
            }
        }

        // Cause 1 is flat between t=1 and t=3 for row 0, but other causes
        // fail in that interval; its CIF must remain exactly flat.
        assert_eq!(result.cif[0][[0, 1]], result.cif[0][[0, 2]]);
        // All causes are flat between t=3 and t=7 for row 1 except cause 2;
        // causes 1 and 3 must not move.
        assert_eq!(result.cif[0][[1, 2]], result.cif[0][[1, 3]]);
        assert_eq!(result.cif[2][[1, 2]], result.cif[2][[1, 3]]);
    }

    #[test]
    fn competing_risks_cif_rejects_bad_time_grids_and_nonfinite_hazards() {
        let cumulative = Array3::zeros((2, 1, 2));

        for times in [array![0.0, 0.0], array![1.0, 0.5], array![-1.0, 1.0]] {
            let err = assemble_competing_risks_cif(times.view(), cumulative.view())
                .expect_err("bad time grid should be rejected");
            assert!(matches!(err, SurvivalError::InvalidTimeGrid));
        }

        let times = array![0.0, 1.0];
        let nonfinite = Array3::from_shape_vec((1, 1, 2), vec![0.0, f64::NAN]).expect("shape");
        let err = assemble_competing_risks_cif(times.view(), nonfinite.view())
            .expect_err("nonfinite hazard should be rejected");
        assert!(matches!(err, SurvivalError::NonFiniteInput));
    }

    #[test]
    fn competing_risks_cif_extreme_hazards_remain_bounded() {
        let times = array![0.0, 1.0, 2.0];
        let cumulative =
            Array3::from_shape_vec((2, 1, 3), vec![0.0, 500.0, 1000.0, 0.0, 250.0, 1000.0])
                .expect("shape");

        let result =
            assemble_competing_risks_cif(times.view(), cumulative.view()).expect("assemble CIF");

        for value in result
            .cif
            .iter()
            .flat_map(|m| m.iter())
            .chain(result.overall_survival.iter())
        {
            assert!(value.is_finite());
            assert!((0.0..=1.0).contains(value));
        }
        assert!((result.cif[0][[0, 2]] + result.cif[1][[0, 2]] - 1.0).abs() < 1e-12);
        assert_eq!(result.overall_survival[[0, 2]], 0.0);
    }

    fn toy_penalties() -> PenaltyBlocks {
        let s = array![[2.0, 0.5], [0.5, 3.0]];
        PenaltyBlocks::new(vec![PenaltyBlock {
            matrix: s,
            lambda: 1.7,
            range: 1..3,
            nullspace_dim: 0,
        }])
    }

    fn survival_inputs<'a>(
        age_entry: &'a Array1<f64>,
        age_exit: &'a Array1<f64>,
        event_target: &'a Array1<u8>,
        event_competing: &'a Array1<u8>,
        sampleweight: &'a Array1<f64>,
        x_entry: &'a Array2<f64>,
        x_exit: &'a Array2<f64>,
        x_derivative: &'a Array2<f64>,
    ) -> SurvivalEngineInputs<'a> {
        SurvivalEngineInputs {
            age_entry: age_entry.view(),
            age_exit: age_exit.view(),
            event_target: event_target.view(),
            event_competing: event_competing.view(),
            sampleweight: sampleweight.view(),
            x_entry: x_entry.view(),
            x_exit: x_exit.view(),
            x_derivative: x_derivative.view(),
            monotonicity_constraint_rows: None,
            monotonicity_constraint_offsets: None,
        }
    }

    fn survival_model_with_offsets(
        inputs: SurvivalEngineInputs<'_>,
        offsets: Option<SurvivalBaselineOffsets<'_>>,
        penalties: PenaltyBlocks,
        monotonicity: SurvivalMonotonicityPenalty,
        spec: SurvivalSpec,
    ) -> Result<WorkingModelSurvival, SurvivalError> {
        WorkingModelSurvival::from_engine_inputswith_offsets(
            inputs,
            offsets,
            penalties,
            monotonicity,
            spec,
        )
    }

    #[test]
    fn penaltyhessian_matchesgradient_jacobian() {
        let penalties = toy_penalties();
        let beta = array![10.0, -0.3, 1.2, 7.0];

        let grad = penalties.gradient(&beta);
        let h = penalties.hessian(beta.len());
        let b_block = beta.slice(s![1..3]).to_owned();
        let expected = 1.7 * array![[2.0, 0.5], [0.5, 3.0]].dot(&b_block);

        assert!((grad[1] - expected[0]).abs() < 1e-12);
        assert!((grad[2] - expected[1]).abs() < 1e-12);
        assert!((h[[1, 1]] - 1.7 * 2.0).abs() < 1e-12);
        assert!((h[[1, 2]] - 1.7 * 0.5).abs() < 1e-12);
        assert!((h[[2, 1]] - 1.7 * 0.5).abs() < 1e-12);
        assert!((h[[2, 2]] - 1.7 * 3.0).abs() < 1e-12);
    }

    #[test]
    fn penaltygradient_matches_deviance_finite_difference() {
        let penalties = toy_penalties();
        let beta = array![10.0, -0.3, 1.2, 7.0];
        let grad = penalties.gradient(&beta);
        let eps = 1e-7;

        for idx in 0..beta.len() {
            let mut plus = beta.clone();
            let mut minus = beta.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let fd = (penalties.deviance(&plus) - penalties.deviance(&minus)) / (2.0 * eps);
            assert_eq!(
                grad[idx].signum(),
                fd.signum(),
                "gradient/deviance sign mismatch at idx={idx}: grad={} fd={fd}",
                grad[idx]
            );
            assert!(
                (grad[idx] - fd).abs() < 1e-6,
                "gradient/deviance mismatch at idx={idx}: grad={} fd={fd}",
                grad[idx]
            );
        }
    }

    #[test]
    fn offset_channel_residuals_match_central_fd_of_nll() {
        // Three observations: two events (non-origin entry and origin entry)
        // and one censored row. This exercises every nonzero channel at least
        // once: r_exit from all rows, r_entry only from the first (has entry
        // interval), r_derivative only from events.
        let age_entry = array![0.5_f64, 0.0, 0.3];
        let age_exit = array![1.4_f64, 1.0, 2.0];
        let event_target = array![1u8, 1u8, 0u8];
        let event_competing = array![0u8, 0u8, 0u8];
        let sampleweight = array![1.0_f64, 2.5, 0.7];
        let x_entry = array![
            [1.0, age_entry[0].ln()],
            [1.0, age_entry[1].max(1e-8).ln()],
            [1.0, age_entry[2].ln()]
        ];
        let x_exit = array![
            [1.0, age_exit[0].ln()],
            [1.0, age_exit[1].ln()],
            [1.0, age_exit[2].ln()]
        ];
        let x_derivative = array![
            [0.0, 1.0 / age_exit[0]],
            [0.0, 1.0 / age_exit[1]],
            [0.0, 1.0 / age_exit[2]]
        ];
        // Baseline offsets chosen so η_entry, η_exit, s are all comfortably
        // away from overflow / monotonicity-violation boundaries.
        let o_entry = array![0.2_f64, 0.0, 0.1];
        let o_exit = array![0.4_f64, 0.5, 0.7];
        let o_deriv = array![0.3_f64, 0.8, 0.5];
        let penalties = PenaltyBlocks::new(Vec::new());
        let mono = SurvivalMonotonicityPenalty { tolerance: 1e-8 };
        let beta = array![-0.7_f64, 0.6];

        let build = |o_e: &Array1<f64>, o_x: &Array1<f64>, o_d: &Array1<f64>| {
            survival_model_with_offsets(
                survival_inputs(
                    &age_entry,
                    &age_exit,
                    &event_target,
                    &event_competing,
                    &sampleweight,
                    &x_entry,
                    &x_exit,
                    &x_derivative,
                ),
                Some(SurvivalBaselineOffsets {
                    eta_entry: o_e.view(),
                    eta_exit: o_x.view(),
                    derivative_exit: o_d.view(),
                }),
                penalties.clone(),
                mono,
                SurvivalSpec::Net,
            )
            .expect("model build")
        };

        let base = build(&o_entry, &o_exit, &o_deriv);
        let resid = base
            .offset_channel_residuals(&beta)
            .expect("offset residuals");
        assert_eq!(resid.exit.len(), 3);
        assert_eq!(resid.entry.len(), 3);
        assert_eq!(resid.derivative.len(), 3);

        // NLL equals half the deviance returned by update_state; that is the
        // exact unpenalized loss whose offset partials r_{X,E,D} encode.
        let nll = |m: &WorkingModelSurvival| 0.5 * m.update_state(&beta).expect("state").deviance;
        let h = 1e-6;

        // Row 1 (origin entry, event=1) has no entry interval, so r_entry[1]
        // must be exactly 0. Row 2 (censored) has r_deriv[2] exactly 0. Check
        // those identities before FD comparison on the nonzero elements.
        assert_eq!(resid.entry[1], 0.0);
        assert_eq!(resid.derivative[2], 0.0);

        for i in 0..3 {
            // exit channel: perturb o_exit[i] alone.
            {
                let mut op = o_exit.clone();
                let mut om = o_exit.clone();
                op[i] += h;
                om[i] -= h;
                let fd = (nll(&build(&o_entry, &op, &o_deriv))
                    - nll(&build(&o_entry, &om, &o_deriv)))
                    / (2.0 * h);
                assert!(
                    (resid.exit[i] - fd).abs() < 1e-6,
                    "∂NLL/∂o_X[{i}]: analytic={:.6e} fd={:.6e}",
                    resid.exit[i],
                    fd
                );
            }
            // entry channel: only row 0 has an entry interval; for rows with
            // entry_at_origin the offset contributes nothing to NLL and FD
            // must also be exactly 0 to numerical precision.
            {
                let mut op = o_entry.clone();
                let mut om = o_entry.clone();
                op[i] += h;
                om[i] -= h;
                let fd = (nll(&build(&op, &o_exit, &o_deriv))
                    - nll(&build(&om, &o_exit, &o_deriv)))
                    / (2.0 * h);
                assert!(
                    (resid.entry[i] - fd).abs() < 1e-6,
                    "∂NLL/∂o_E[{i}]: analytic={:.6e} fd={:.6e}",
                    resid.entry[i],
                    fd
                );
            }
            // derivative channel: only event rows contribute.
            {
                let mut op = o_deriv.clone();
                let mut om = o_deriv.clone();
                op[i] += h;
                om[i] -= h;
                let fd = (nll(&build(&o_entry, &o_exit, &op))
                    - nll(&build(&o_entry, &o_exit, &om)))
                    / (2.0 * h);
                assert!(
                    (resid.derivative[i] - fd).abs() < 1e-6,
                    "∂NLL/∂o_D[{i}]: analytic={:.6e} fd={:.6e}",
                    resid.derivative[i],
                    fd
                );
            }
        }
    }

    #[test]
    fn offset_channel_residuals_respect_zero_sampleweight() {
        let age_entry = array![1.0_f64, 2.0];
        let age_exit = array![2.0_f64, 3.5];
        let event_target = array![1u8, 1u8];
        let event_competing = array![0u8, 0u8];
        let sampleweight = array![0.0_f64, 1.2]; // row 0 is excluded by weight
        let x_entry = array![[1.0, age_entry[0].ln()], [1.0, age_entry[1].ln()]];
        let x_exit = array![[1.0, age_exit[0].ln()], [1.0, age_exit[1].ln()]];
        let x_derivative = array![[0.0, 1.0 / age_exit[0]], [0.0, 1.0 / age_exit[1]]];
        let penalties = PenaltyBlocks::new(Vec::new());
        let mono = SurvivalMonotonicityPenalty { tolerance: 1e-8 };
        let beta = array![-1.0_f64, 0.8];

        let model = survival_model_with_offsets(
            survival_inputs(
                &age_entry,
                &age_exit,
                &event_target,
                &event_competing,
                &sampleweight,
                &x_entry,
                &x_exit,
                &x_derivative,
            ),
            Some(SurvivalBaselineOffsets {
                eta_entry: array![0.0_f64, 0.1].view(),
                eta_exit: array![0.0_f64, 0.2].view(),
                derivative_exit: array![0.0_f64, 0.1].view(),
            }),
            penalties,
            mono,
            SurvivalSpec::Net,
        )
        .expect("model");
        let r = model.offset_channel_residuals(&beta).expect("resid");
        // Row 0 (sampleweight=0) must contribute zero in every channel.
        assert_eq!(r.exit[0], 0.0);
        assert_eq!(r.entry[0], 0.0);
        assert_eq!(r.derivative[0], 0.0);
        // Row 1 must still carry a nonzero exit-channel residual.
        assert!(r.exit[1] != 0.0);
    }

    #[test]
    fn derivative_offset_must_clear_nonstructural_monotonicity_threshold() {
        let age_entry = array![1.0_f64];
        let age_exit = array![2.0_f64];
        let event_target = array![1u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[1.0, 0.0]];
        let x_exit = array![[1.0, 0.0]];
        let x_derivative = array![[0.0, 0.0]];
        let penalties = PenaltyBlocks::new(Vec::new());
        let monotonicity = SurvivalMonotonicityPenalty { tolerance: 3.0 };
        let eta_entry_offset = array![0.0];
        let eta_exit_offset = array![0.0];
        let derivative_offset_below_guard = array![2.0];
        let derivative_offset_above_guard = array![3.1];
        let offsets_below_guard = SurvivalBaselineOffsets {
            eta_entry: eta_entry_offset.view(),
            eta_exit: eta_exit_offset.view(),
            derivative_exit: derivative_offset_below_guard.view(),
        };
        let offsets_above_guard = SurvivalBaselineOffsets {
            eta_entry: eta_entry_offset.view(),
            eta_exit: eta_exit_offset.view(),
            derivative_exit: derivative_offset_above_guard.view(),
        };

        let model_below_guard = survival_model_with_offsets(
            survival_inputs(
                &age_entry,
                &age_exit,
                &event_target,
                &event_competing,
                &sampleweight,
                &x_entry,
                &x_exit,
                &x_derivative,
            ),
            Some(offsets_below_guard),
            penalties.clone(),
            monotonicity,
            SurvivalSpec::Net,
        )
        .expect("construct model with derivative offset below guard");
        let err = model_below_guard
            .update_state(&array![0.0, 0.0])
            .expect_err("derivative offset below guard should be rejected");
        let err_text = err.to_string();
        assert!(
            err_text.contains("d_eta/dt=2.000e0") && err_text.contains("tolerance=3.000e0"),
            "expected derivative guard rejection to report the offset-driven derivative: {err_text}"
        );

        let model_above_guard = survival_model_with_offsets(
            survival_inputs(
                &age_entry,
                &age_exit,
                &event_target,
                &event_competing,
                &sampleweight,
                &x_entry,
                &x_exit,
                &x_derivative,
            ),
            Some(offsets_above_guard),
            penalties,
            SurvivalMonotonicityPenalty { tolerance: 3.0 },
            SurvivalSpec::Net,
        )
        .expect("construct model with derivative offset above guard");
        let state = model_above_guard
            .update_state(&array![0.0, 0.0])
            .expect("derivative offset above guard should remain feasible");
        assert!(state.deviance.is_finite());
    }

    #[test]
    fn structural_monotonicity_rejects_negative_derivative_offsets() {
        let age_entry = array![1.0_f64];
        let age_exit = array![2.0_f64];
        let event_target = array![1u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[0.0]];
        let x_exit = array![[0.2]];
        let x_derivative = array![[1.0]];
        let eta_entry = array![0.0];
        let eta_exit = array![0.0];
        let derivative_exit = array![-1e-3];
        let offsets = SurvivalBaselineOffsets {
            eta_entry: eta_entry.view(),
            eta_exit: eta_exit.view(),
            derivative_exit: derivative_exit.view(),
        };

        let mut model = survival_model_with_offsets(
            survival_inputs(
                &age_entry,
                &age_exit,
                &event_target,
                &event_competing,
                &sampleweight,
                &x_entry,
                &x_exit,
                &x_derivative,
            ),
            Some(offsets),
            PenaltyBlocks::new(Vec::new()),
            SurvivalMonotonicityPenalty { tolerance: 0.0 },
            SurvivalSpec::Net,
        )
        .expect("construct structural survival model");
        let err = model
            .set_structural_monotonicity(true, 1)
            .expect_err("negative derivative offsets must be rejected");
        assert!(
            err.to_string()
                .contains("structural monotonicity requires nonnegative derivative offsets"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn monotonicity_constraints_collapse_positive_collinearrows() {
        let a = array![[0.0, 0.5, 0.0], [0.0, 0.25, 0.0], [0.0, 0.125, 0.0]];
        let b = array![1e-8, 1e-8, 1e-8];

        let compressed = compress_positive_collinear_constraints(&a, &b);

        assert_eq!(compressed.a.nrows(), 1);
        assert_eq!(compressed.a.ncols(), 3);
        assert!(compressed.a[[0, 0]].abs() <= 1e-12);
        assert!((compressed.a[[0, 1]] - 1.0).abs() <= 1e-12);
        assert!(compressed.a[[0, 2]].abs() <= 1e-12);
        assert!((compressed.b[0] - 8e-8).abs() <= 1e-18);
    }

    #[test]
    fn monotonicity_constraints_preserve_distinct_directions() {
        let a = array![[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]];
        let b = array![0.2, 0.3, 0.1];

        let compressed = compress_positive_collinear_constraints(&a, &b);

        assert_eq!(compressed.a.nrows(), 2);
        let mut saw_x = false;
        let mut saw_y = false;
        for i in 0..compressed.a.nrows() {
            if (compressed.a[[i, 0]] - 1.0).abs() <= 1e-12 && compressed.a[[i, 1]].abs() <= 1e-12 {
                saw_x = true;
                assert!((compressed.b[i] - 0.2).abs() <= 1e-12);
            }
            if compressed.a[[i, 0]].abs() <= 1e-12 && (compressed.a[[i, 1]] - 1.0).abs() <= 1e-12 {
                saw_y = true;
                assert!((compressed.b[i] - 0.3).abs() <= 1e-12);
            }
        }
        assert!(saw_x);
        assert!(saw_y);
    }

    #[test]
    fn monotonicity_constraints_cluster_near_collinearrows() {
        let a = array![
            [0.0, 0.5, 0.0],
            [0.0, 0.50000000003, 0.0],
            [0.0, 0.49999999997, 0.0]
        ];
        let b = array![1e-8, 1.00000000005e-8, 0.99999999995e-8];

        let compressed = compress_positive_collinear_constraints(&a, &b);

        assert_eq!(compressed.a.nrows(), 1);
        assert_eq!(compressed.a.ncols(), 3);
        assert!(compressed.a[[0, 0]].abs() <= 1e-12);
        assert!((compressed.a[[0, 1]] - 1.0).abs() <= 1e-12);
        assert!(compressed.a[[0, 2]].abs() <= 1e-12);
        assert!((compressed.b[0] - 2.0e-8).abs() <= 1e-18);
    }

    #[test]
    fn monotonicity_constraints_cluster_spline_like_near_duplicates() {
        let a = array![
            [0.0, 0.401, 0.302, 0.197],
            [0.0, 0.40100000003, 0.30199999998, 0.19700000001],
            [0.0, 0.40099999997, 0.30200000002, 0.19699999999],
            [0.0, 0.125, 0.500, 0.375]
        ];
        let b = array![2.0e-8, 2.00000000004e-8, 1.99999999996e-8, 3.0e-8];

        let compressed = compress_positive_collinear_constraints(&a, &b);

        assert_eq!(compressed.a.nrows(), 2);
        let mut clustered_face = false;
        let mut distinct_face = false;
        for i in 0..compressed.a.nrows() {
            let row = compressed.a.row(i);
            if row[1] > 0.99 && row[2] > 0.7 && row[3] > 0.49 {
                clustered_face = true;
                assert!((compressed.b[i] - (2.0e-8 / 0.401)).abs() <= 1e-12);
            } else {
                distinct_face = true;
                assert!((row[1] - 0.25).abs() <= 1e-12);
                assert!((row[2] - 1.0).abs() <= 1e-12);
                assert!((row[3] - 0.75).abs() <= 1e-12);
                assert!((compressed.b[i] - 6.0e-8).abs() <= 1e-18);
            }
        }
        assert!(clustered_face);
        assert!(distinct_face);
    }

    #[test]
    fn compressed_monotonicity_constraints_preserve_uncompressed_feasible_region() {
        let uncompressed_constraints = LinearInequalityConstraints {
            a: array![
                [0.0, 0.5, 0.0],
                [0.0, 1.0 / 3.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.125, 0.0]
            ],
            b: Array1::from_elem(4, 1e-8),
        };
        let compressed_constraints = compress_positive_collinear_constraints(
            &uncompressed_constraints.a,
            &uncompressed_constraints.b,
        );

        let candidates = [
            array![0.0, 1e-9, 0.0],
            array![0.0, 4e-8, 0.0],
            array![0.0, 8e-8, 0.0],
            array![0.0, 2e-7, 1.5],
        ];
        for beta in candidates {
            let uncompressed_ok = (0..uncompressed_constraints.a.nrows()).all(|i| {
                uncompressed_constraints.a.row(i).dot(&beta) >= uncompressed_constraints.b[i]
            });
            let compressed_ok = (0..compressed_constraints.a.nrows())
                .all(|i| compressed_constraints.a.row(i).dot(&beta) >= compressed_constraints.b[i]);
            assert_eq!(compressed_ok, uncompressed_ok);
        }
    }

}

//! Error and preflight-block enums for survival marginal-slope fitting,
//! with their `Display`/`Error`/`From` conversions. Self-contained: no
//! dependency on the fitting machinery.

#[derive(Debug, Clone)]
pub enum SurvivalMarginalSlopeError {
    /// Spec, data, or runtime configuration failed input validation
    /// (finite/non-negative weights, derivative_guard > 0, supported
    /// base_link, frailty constraints, missing block state, etc.).
    InvalidInput { reason: String },
    /// Lengths, row/column counts, basis widths, or coefficient block
    /// sizes do not agree (covariance dim vs z, design rows vs n,
    /// basis/beta length mismatch, post-update beta length, time
    /// constraints A vs b, hessian_matvec dim mismatch, ...).
    IncompatibleDimensions { reason: String },
    /// A row's transformed time derivative or structural slack fell
    /// below `derivative_guard` (`qd1 < guard`), violating the
    /// monotonicity contract.
    MonotonicityViolation { reason: String },
    /// A numerical step produced a non-finite, non-positive, or
    /// internally inconsistent quantity that downstream code cannot
    /// consume (e.g. non-positive `D`, non-positive `chi1`, calibration
    /// derivative disagrees with the direct evaluation, transformed
    /// derivative not strictly positive).
    NumericalFailure { reason: String },
    /// An integration / outer-optimization step failed to converge to
    /// the requested tolerance (intercept residual, REML outer loop).
    IntegrationFailed { reason: String },
    /// The requested combination of options is not implemented (non-
    /// probit base link, flexible row calculus with K > 1, spatial psi
    /// for unsupported block roles, ...).
    UnsupportedConfiguration { reason: String },
}

/// Block tag used by the joint training-row preflight diagnostic.
/// Names a single block in the joint design layout
/// `[time | marginal | logslope | score_warp? | link_dev?]`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum JointPreflightBlock {
    Time,
    Marginal,
    Logslope,
    ScoreWarp,
    LinkDev,
}

impl std::fmt::Display for JointPreflightBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            JointPreflightBlock::Time => "time",
            JointPreflightBlock::Marginal => "marginal",
            JointPreflightBlock::Logslope => "logslope",
            JointPreflightBlock::ScoreWarp => "score_warp",
            JointPreflightBlock::LinkDev => "link_dev",
        };
        f.write_str(name)
    }
}

impl std::fmt::Display for SurvivalMarginalSlopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SurvivalMarginalSlopeError::InvalidInput { reason }
            | SurvivalMarginalSlopeError::IncompatibleDimensions { reason }
            | SurvivalMarginalSlopeError::MonotonicityViolation { reason }
            | SurvivalMarginalSlopeError::NumericalFailure { reason }
            | SurvivalMarginalSlopeError::IntegrationFailed { reason }
            | SurvivalMarginalSlopeError::UnsupportedConfiguration { reason } => {
                f.write_str(reason)
            }
        }
    }
}

impl std::error::Error for SurvivalMarginalSlopeError {}

impl From<SurvivalMarginalSlopeError> for String {
    fn from(err: SurvivalMarginalSlopeError) -> String {
        err.to_string()
    }
}

impl From<String> for SurvivalMarginalSlopeError {
    /// Inbound conversion from helpers in this module (and adjacent
    /// families) that still surface `Result<_, String>`. The text is
    /// preserved verbatim; `InvalidInput` is the catch-all category for
    /// strings produced outside this module.
    fn from(reason: String) -> SurvivalMarginalSlopeError {
        SurvivalMarginalSlopeError::InvalidInput { reason }
    }
}

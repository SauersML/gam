//! [#780] Auxiliary type and constant declarations for SAE manifold
//! construction, split out of `construction.rs` verbatim to keep that file
//! under the per-file line-count gate. These are the free-standing items that
//! precede the main `impl SaeManifoldTerm` block: the outer-gradient analytic
//! error taxonomy, the forced-row-layout override alias, the co-training
//! penalty weight constants, and the amortized-encoder consistency report. The
//! parent module re-exports this module with a glob, so every reference (here
//! and in `construction.rs`) resolves exactly as before.

use super::*;

/// Typed error from the SAE outer-gradient analytic assembly path (#1436).
///
/// Every variant propagates when the rank-revealing projected solve cannot
/// produce a reliable implicit derivative. The taxonomy distinguishes genuine
/// conditioning/non-identifiability from assembly invariant defects without
/// authorizing a degraded fallback direction.
#[derive(Clone, Debug)]
pub(crate) enum OuterGradientError {
    /// Near-singular or ill-conditioned joint Hessian at a feasible ρ.
    IllConditioned { reason: String },
    /// A non-identifiable / gauge-degenerate direction at this ρ.
    NonIdentifiable { reason: String },
    /// Unexpected: shape/dimension mismatch, non-finite intermediate, or a
    /// violated internal invariant.
    InternalInvariant { reason: String },
}

impl OuterGradientError {
    /// Construct an [`OuterGradientError::InternalInvariant`] from any error
    /// displayable — the default classification for unexpected assembly failures
    /// (shape mismatches, non-finite intermediates, violated invariants).
    pub(crate) fn internal<E: std::fmt::Display>(err: E) -> Self {
        Self::InternalInvariant {
            reason: err.to_string(),
        }
    }

    /// #1451 — classify a `String` error surfaced by the deflation linear-algebra
    /// path (`apply_cached_arrow_hessian`, `DeflatedArrowSolver::from_orthonormal_gauges`)
    /// into the correct [`OuterGradientError`] class.
    ///
    /// A genuine rank-deficiency / near-singularity failure (a back-solve or
    /// Cholesky/Woodbury factor that tripped on a finite, correctly-shaped input)
    /// is a legitimate #1273 conditioning failure and keeps `conditioning_err`
    /// (`IllConditioned`). A
    /// shape/dimension mismatch or a non-finite intermediate is an
    /// internal-invariant defect and MUST propagate ([`Self::internal`]) instead
    /// of being masked as a plausible-but-wrong descent direction — exactly the
    /// #1436 contract.
    ///
    /// The two solver helpers return `String` (not a typed error), so the
    /// distinction is drawn from the stable markers those helpers emit for their
    /// shape/non-finite guards (`vector shapes`, `gauge length`, `must be finite`,
    /// `non-finite`). Everything else — including the `cholesky`/back-solve
    /// near-singular failures — is treated as a genuine conditioning trip. Both
    /// typed classes propagate if the projected solve cannot complete.
    pub(crate) fn classify_arrow_solver_error(message: &str, conditioning_err: Self) -> Self {
        let lower = message.to_ascii_lowercase();
        let is_internal = lower.contains("vector shapes")
            || lower.contains("gauge length")
            || lower.contains("solution length")
            || lower.contains("!= cache")
            || lower.contains("must be finite")
            || lower.contains("non-finite")
            || lower.contains("not finite")
            || lower.contains("nan")
            || lower.contains("inf");
        if is_internal {
            Self::internal(message)
        } else {
            conditioning_err
        }
    }
}

impl std::fmt::Display for OuterGradientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IllConditioned { reason } => write!(f, "ill-conditioned: {reason}"),
            Self::NonIdentifiable { reason } => write!(f, "non-identifiable: {reason}"),
            Self::InternalInvariant { reason } => write!(f, "internal invariant: {reason}"),
        }
    }
}

/// Preserve the producer's rho-locality verdict at the generic outer-objective
/// boundary.
///
/// Conditioning and identifiability belong to the evaluated inner state at one
/// particular rho: a neighbouring rho can have a perfectly regular implicit
/// derivative, so a line search must reject this trial and contract. An
/// internal invariant is independent of rho and remains fatal. Converting every
/// variant to `RemlOptimizationFailed(String)` erased that distinction and made
/// a legitimate non-identifiable BFGS trial abort the whole fit (#2653).
impl From<OuterGradientError> for EstimationError {
    fn from(error: OuterGradientError) -> Self {
        let reason = error.to_string();
        match error {
            OuterGradientError::IllConditioned { .. }
            | OuterGradientError::NonIdentifiable { .. } => {
                EstimationError::TrialPointRefused { reason }
            }
            OuterGradientError::InternalInvariant { .. } => {
                EstimationError::RemlOptimizationFailed(reason)
            }
        }
    }
}

/// String-returning reconstruction diagnostics inside the manifold also
/// consume the typed solver failure with `?`. They do not cross the
/// outer-optimizer boundary, so rendering there does not erase a recoverability
/// decision.
impl From<OuterGradientError> for String {
    fn from(error: OuterGradientError) -> Self {
        error.to_string()
    }
}

/// Active-set layout override for [`SaeManifoldTerm::assemble_arrow_schur_inner`].
///
/// `None` is the production path: TopK derives its exact support layout and all
/// smooth modes remain dense. `Some(layout_opt)` pins a specific layout — dense
/// (`Some(None)`) or a chosen compact `SaeRowLayout` (`Some(Some(..))`) — so the
/// compact-vs-dense Riemannian-geometry equality regression can drive both code
/// paths on identical data.
pub(crate) type ForcedRowLayout = Option<Option<SaeRowLayout>>;

/// #1154 — amortized-encoder consistency of a fitted dictionary against its own
/// fit-time target. The co-training signal of the joint amortized-encoder +
/// REML loop: how faithfully the cheap initializer plus joint refinement invert
/// the dictionary the inner solve converged to.
#[derive(Debug, Clone, Copy)]
pub struct AmortizedEncoderConsistency {
    /// Mean per-element squared gap between the amortized reconstruction and the
    /// exact fitted reconstruction (`‖x̂_amortized − x̂_exact‖² / (n·p)`). Zero ⇒
    /// the IFT predictor reproduces the encode map exactly to first order.
    pub recon_consistency: f64,
    /// Fraction of rowwise shared-residual solves that did not meet the exact
    /// first-order stationarity tolerance.
    pub unconverged_fraction: f64,
    /// Count of unconverged joint row solves (numerator of the fraction).
    pub n_unconverged: usize,
    /// Total joint row solves scored (`n`).
    pub n_encodes: usize,
}

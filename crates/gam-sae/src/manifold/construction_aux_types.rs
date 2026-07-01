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
/// The `eval()` analytic fallback (#1273/#1440: the plain undeflated analytic
/// outer gradient, NOT a finite difference) must fire ONLY for the genuine
/// conditioning/identifiability failure modes it was designed for — a
/// near-singular-but-valid joint Hessian or a gauge-degenerate direction.
/// Shape/indexing bugs, non-finite intermediates, and violated internal
/// invariants are [`OuterGradientError::InternalInvariant`] and MUST propagate
/// as hard errors so regressions surface instead of being silently masked by a
/// degraded descent direction.
#[derive(Clone, Debug)]
pub(crate) enum OuterGradientError {
    /// Expected: near-singular or ill-conditioned joint Hessian at a feasible ρ
    /// (the genuine #1273 flat-valley case). Eligible for the FD fallback.
    IllConditioned { reason: String },
    /// Expected: a non-identifiable / gauge-degenerate direction at this ρ.
    /// Eligible for the FD fallback.
    NonIdentifiable { reason: String },
    /// Unexpected: shape/dimension mismatch, non-finite intermediate, or a
    /// violated internal invariant. MUST propagate — never fall back to FD.
    InternalInvariant { reason: String },
}

impl OuterGradientError {
    /// Whether this error class is recoverable by the #1273/#1440 analytic
    /// plain-solver fallback (i.e. it represents a legitimate
    /// conditioning/identifiability failure, not a programming/invariant defect).
    pub(crate) fn is_conditioning_recoverable(&self) -> bool {
        matches!(
            self,
            Self::IllConditioned { .. } | Self::NonIdentifiable { .. }
        )
    }

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
    /// (`IllConditioned`), so it stays recoverable by the analytic fallback. A
    /// shape/dimension mismatch or a non-finite intermediate is an
    /// internal-invariant defect and MUST propagate ([`Self::internal`]) instead
    /// of being masked as a plausible-but-wrong descent direction — exactly the
    /// #1436 contract.
    ///
    /// The two solver helpers return `String` (not a typed error), so the
    /// distinction is drawn from the stable markers those helpers emit for their
    /// shape/non-finite guards (`vector shapes`, `gauge length`, `must be finite`,
    /// `non-finite`). Everything else — including the `cholesky`/back-solve
    /// near-singular failures — is treated as a genuine conditioning trip.
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

    /// The exact gate the gradient lane (`SaeManifoldOuterObjective::eval`) uses
    /// to decide whether to descend with the #1273/#1440 analytic plain-solver
    /// fallback instead of propagating the error as a hard failure.
    ///
    /// The fallback is admissible ONLY when BOTH hold:
    /// * the REML cost at this rho is finite (a genuinely feasible point -- the
    ///   plain analytic solver supplies a descent direction for a value the
    ///   analytic path already produced), and
    /// * the error is a legitimate conditioning/identifiability failure
    ///   ([`Self::is_conditioning_recoverable`]) -- the genuine #1273 flat-valley
    ///   case.
    ///
    /// A non-finite cost or an [`OuterGradientError::InternalInvariant`] must
    /// propagate: masking a shape/indexing bug, a non-finite intermediate, or a
    /// violated invariant behind a plausible-but-wrong step is exactly the
    /// regression #1436 closes. Centralising the decision here (rather than
    /// inlining the boolean at the call site) makes the `cost x error-class`
    /// contract a single, directly unit-testable predicate.
    pub(crate) fn admits_plain_solver_fallback(&self, cost: f64) -> bool {
        cost.is_finite() && self.is_conditioning_recoverable()
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

impl From<OuterGradientError> for String {
    fn from(e: OuterGradientError) -> String {
        e.to_string()
    }
}

/// Active-set layout override for [`SaeManifoldTerm::assemble_arrow_schur_inner`].
///
/// `None` is the production path: the layout is derived from the assignment mode
/// and `sparse_active_plan`. `Some(layout_opt)` pins a specific layout — dense
/// (`Some(None)`) or a chosen compact `SaeRowLayout` (`Some(Some(..))`) — so the
/// compact-vs-dense Riemannian-geometry equality regression can drive both code
/// paths on identical data without depending on the host/device memory budget
/// that gates the compact path in production.
pub(crate) type ForcedRowLayout = Option<Option<SaeRowLayout>>;

/// #1154 — base co-training weight for the amortized-encoder reconstruction
/// consistency penalty, as a fraction of the REML criterion magnitude. The
/// effective weight is `COTRAIN_RECON_WEIGHT · max(|REML|, 1)`, so the penalty
/// is a bounded, scale-free share of the objective and needs no caller knob.
pub(crate) const COTRAIN_RECON_WEIGHT: f64 = 0.1;

/// #1154 — base co-training weight for the encoder's certifiable-coverage
/// penalty (the fraction of (row, atom) encodes the Kantorovich certificate
/// rejected). Scaled like [`COTRAIN_RECON_WEIGHT`].
pub(crate) const COTRAIN_CERT_WEIGHT: f64 = 0.05;

/// #1154 — amortized-encoder consistency of a fitted dictionary against its own
/// fit-time target. The co-training signal of the joint amortized-encoder +
/// REML loop: how faithfully (and how certifiably) the cheap one-mat-vec
/// encoder inverts the dictionary the inner solve converged to.
#[derive(Debug, Clone, Copy)]
pub struct AmortizedEncoderConsistency {
    /// Mean per-element squared gap between the amortized reconstruction and the
    /// exact fitted reconstruction (`‖x̂_amortized − x̂_exact‖² / (n·p)`). Zero ⇒
    /// the IFT predictor reproduces the encode map exactly to first order.
    pub recon_consistency: f64,
    /// Fraction of (row, atom) amortized encodes whose Kantorovich certificate
    /// failed (`h > ½`) and fell back to the certified Newton encode.
    pub uncertified_fraction: f64,
    /// Count of uncertified (row, atom) encodes (numerator of the fraction).
    pub n_uncertified: usize,
    /// Total (row, atom) encodes scored (`n · K`).
    pub n_encodes: usize,
}

//! The #2566 `log sigma` curvature certificate: a production curvature value
//! paired with a measured bound on its own error, so a consumer that needs a
//! definite Hessian can refuse on a ratio rather than on a scale cutoff.
//!
//! Split out of `survival/mod.rs` so the #2566 diagnostic can carry its source
//! scanner exemption over ~150 lines of certificate machinery instead of the
//! 6,300-line latent-survival fit math. Allowlisting the parent would exempt the
//! entire fit path, which is precisely the hole the finite-difference ban exists
//! to close.

// FD-OK: #2566 sanctioned diagnostic authority -- differences the GRADIENT to
// CERTIFY the analytic curvature on a separate entry point, never on the fit
// path. The production value comes from the analytic jet; the authority only
// measures the error on it.

/// A `log σ` curvature together with a measured bound on its own error (#2566).
///
/// The curvature is the production value; `estimated_absolute_error` is what an
/// independent authority says it could be wrong by. A consumer that needs a
/// definite Hessian refuses on the ratio rather than on a scale cutoff.
#[derive(Clone, Copy, Debug)]
pub struct CertifiedLogSigmaCurvature {
    /// The production negative-Hessian `∂²/∂(log σ)²` entry.
    pub curvature: f64,
    /// Measured bound on `|curvature − truth|`: the observed disagreement with an
    /// independent finite-difference authority, plus that authority's own
    /// numerical error. Not a proved bound — an estimate, from measurement.
    pub estimated_absolute_error: f64,
}

impl CertifiedLogSigmaCurvature {

}

// END-FD-OK

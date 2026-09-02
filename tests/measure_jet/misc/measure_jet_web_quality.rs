//! Measure-jet spline quality gates on a filament web (#904 paradigm: assert
//! against self-constructed truth).
//!
//! Geometry: a Y-junction web in latent R² — strand A (-1,0)→(0,0), strand B
//! (0,0)→(0.8,0.6), strand C (0,0)→(0.8,-0.6) — embedded into ambient R⁸ by a
//! fixed orthonormal linear map plus small ambient coordinate noise. The
//! response is a trend in arc-length, continuous at the junction, with a
//! slope change onto strand C. Training deletes the middle third of strand B
//! (the gap). One integrated gate checks five contracts in file order, sharing
//! fitted models where the contracts intentionally use the same data-generating
//! process:
//!
//! 1. **Truth recovery off-gap**:
//!    at d = 8 ambient with 1-D intrinsic structure, the measure-learned
//!    geometry must recover the strand signal within 2.5× the observation
//!    noise.
//! 2. **Support diagnostic**:
//!    the support curve must be computable from the FITTED model alone and
//!    must separate an on-web query from a far off-web query at the finest
//!    band scale.
//! 3. **Gap bridging with the trend, not the mean**: inside the deleted
//!    stretch of strand B the fit must continue the flank-attested slope
//!    (same sign, within 60% of truth) rather than collapse toward the
//!    global training mean. This is the no-mass-term/N2 contract observed
//!    end to end through REML.
//! 4. **GLM composition**: the same web with a Poisson count response — the
//!    measure-jet penalty must
//!    compose with PIRLS/REML for a non-gaussian family and recover the
//!    log-intensity off-gap.
//! 5. **Interval honesty**: 95% pointwise bands built from the fit's
//!    smoothing-corrected coefficient
//!    covariance must approximately cover the true mean at held-out
//!    on-web points.

// ---------------------------------------------------------------------------
// Data generation: latent web geometry, R²→R⁸ embedding, response encoders.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fit + readout: formula fits, frozen-spec design replay, error metrics.
// ---------------------------------------------------------------------------


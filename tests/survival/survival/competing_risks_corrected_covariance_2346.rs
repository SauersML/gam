//! #2346 acceptance witness — a unified competing-risks (cause-specific) fit
//! must carry the smoothing-corrected joint coefficient covariance with typed
//! provenance, so interval requests at the DEFAULT covariance mode
//! (`SmoothingCorrected`, which per the #2296 provenance contract never falls
//! back) stop hard-erroring on competing-risks models.
//!
//! The fit-side chain under test (gam-custom-family):
//! `joint_smoothing_correction` builds the first-order ρ-uncertainty inflation
//! `C = A·V_ρ·Aᵀ` (`A = V_cond·U`, `U[:,o] = (∂S_λ/∂ρ_o)·β̂`) from the SAME
//! analytic outer ρ-Hessian the criterion certificate judged, excludes rail
//! coordinates (#2337 Thm 2.3), lifts `C` through the identifiability gauge
//! alongside the conditional covariance, and publishes
//! `beta_covariance_corrected = V_cond + C` with
//! `FirstOrderIdentifiedSubspace` provenance on the fit inference.
//!
//! Assertions:
//! 1. presence — the corrected covariance and its typed method are on the fit;
//! 2. shape — corrected matches the conditional covariance dimensions
//!    (cross-cause blocks retained, nothing collapsed per-cause);
//! 3. inflation — `C = V_c − V_cond` is the PSD ρ-uncertainty term, so every
//!    diagonal of the corrected matrix is ≥ the conditional diagonal (within
//!    roundoff) and at least one strictly grows (the smoothing parameters of a
//!    REML fit on finite data are not known exactly);
//! 4. symmetry + finiteness of the corrected matrix.


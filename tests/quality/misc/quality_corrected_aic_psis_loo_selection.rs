//! End-to-end OBJECTIVE quality for the #946 model-comparison channels: exact
//! Wood–Pya–Säfken corrected conditional AIC and zero-refit PSIS-LOO, computed
//! from the fit-retained exact pieces (smoothing-parameter covariance Σ_ρ and
//! the ALO leave-one-out predictions).
//!
//! TRUTH IS CONSTRUCTED, NOT BORROWED. Each arm simulates data from a *known*
//! generative structure and asserts the comparison channels recover the right
//! ordering / sign — a peer tool is never the oracle here.
//!
//!   A. PREDICTIVE SELECTION (PSIS-LOO). Generate data where one of two nested
//!      models is the genuine generator (a real smooth signal vs an
//!      over-flexible competitor on pure noise). The paired Δelpd from
//!      `compare` must point at the model that actually predicts held-out data
//!      better, with the sign the issue defines (positive Δelpd favours `a`).
//!
//!   B. CORRECTED-EDF DIRECTION (Wood–Pya–Säfken). The corrected effective df
//!      must account for smoothing-parameter uncertainty: `τ ≥ tr(F)` exactly
//!      (the ρ-uncertainty contribution `tr(X'WX·Σ_ρ)` is a PSD trace), and the
//!      corrected AIC must therefore penalise complexity at least as much as the
//!      naive conditional AIC. On a fit with a genuinely penalised smooth the
//!      correction must be strictly positive — there IS λ-uncertainty — so the
//!      corrected AIC must exceed the conditional one.
//!
//! Bounds are not weakened to force a pass; a genuine shortfall failing is the
//! intended behaviour.


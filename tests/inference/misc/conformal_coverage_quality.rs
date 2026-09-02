//! End-to-end OBJECTIVE-quality test for distribution-free conformal
//! calibration of prediction intervals (`gam::conformal`).
//!
//! The primary assertion is *realized marginal coverage* on a fresh held-out
//! set drawn from the same DGP: a conformal interval calibrated for nominal
//! 1 − α must cover at least (1 − α) of held-out responses, within
//! finite-sample slack, REGARDLESS of model misspecification.
//!
//! The misspecification is deliberate: the data are HETEROSCEDASTIC (noise
//! standard deviation grows with the covariate), which a homoscedastic
//! Gaussian-identity GAM gets wrong. The test asserts:
//!
//!   1. the conformal interval (calibrated from the model's own
//!      approximate-leave-one-out held-out residuals) covers ≥ nominal on a
//!      fresh draw, while
//!   2. the plain model-based 90% confidence interval UNDER-covers,
//!
//! demonstrating the safety-net value: conformal restores valid coverage on
//! top of a misspecified likelihood. A second arm checks the homoscedastic
//! case still covers (no spurious over/under behavior), and a third checks the
//! exact-order-statistic multiplier is honest about a too-small calibration
//! set (returns +∞ → unbounded interval).


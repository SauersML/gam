//! Regression test for issue #682: distribution-free split-conformal
//! calibration must consume a genuinely held-out calibration fold whose size
//! differs from the training set.
//!
//! Before the fix, `predict_full_uncertainty_conformal` bound the held-out
//! calibration fold to the FROZEN TRAINING fit geometry (via
//! `ConformalCalibrator::from_fit` → ALO over the training `FitGeometry`),
//! which required `hessian_weights.len() == n_cal` and aborted for essentially
//! every realistic fold with:
//!
//!   "ALO diagnostics require hessian_weights length 200; got 500"
//!
//! For a genuinely held-out fold, split-conformal needs NO leave-one-out
//! correction: the fitted predictor is already independent of every
//! calibration point, so the honest nonconformity score is the plain held-out
//! residual `r_i = y_cal_i − μ̂(x_cal_i)` (normalized by the model's
//! predict-time response-scale SE). The exact order-statistic multiplier then
//! gives finite-sample marginal coverage `P(Y ∈ interval) ≥ 1 − α`.
//!
//! This test asserts BOTH well-posed properties on a Gaussian fit with
//! n_train = 600 and a HELD-OUT calibration fold of n_cal = 200 (a different
//! size):
//!
//!   1. the held-out fold of a different size is ACCEPTED (no error), and
//!   2. the resulting conformal interval achieves at least nominal coverage on
//!      a fresh draw from the same DGP (within small finite-sample slack).


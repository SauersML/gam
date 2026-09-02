//! End-to-end quality: gam's ALO sandwich standard error on the linear
//! predictor must be **well CALIBRATED** — the 95% confidence interval it builds
//! around each in-sample linear predictor must cover the *known true* linear
//! predictor `eta_true_i = x_i' beta_true` at (close to) the nominal 95% rate.
//!
//! OBJECTIVE METRIC ASSERTED: empirical coverage of `eta_true` by the
//! gam-derived 95% CI `eta_hat_i +/- t_{0.975, n-p} * SE(eta_i)`, aggregated over
//! many independent simulation replicates drawn from a fixed known parameter
//! vector. A standard-error estimator is *good* exactly when its intervals cover
//! the truth at their stated rate; this is the property a practitioner actually
//! relies on. The pass criterion is
//!
//!   | empirical_coverage(gam) - 0.95 | <= 0.02
//!
//! i.e. gam's intervals are neither anti-conservative (too narrow) nor wasteful
//! (too wide) against ground truth. This is a TRUTH-RELATIVE calibration claim,
//! not a "reproduce a peer tool" claim: the SEs are judged against the true data
//! generating process, not against another fitted output.
//!
//! BASELINE TO MATCH-OR-BEAT: Python `statsmodels` (OLS / Gaussian GLM), the
//! standard regression stack, fits the identical data and builds its own
//! textbook OLS prediction interval `SE(eta_i) = sqrt(sigma^2 x_i'(X'X)^{-1} x_i)`,
//! `sigma^2 = RSS/(n-p)`. We compute statsmodels' own empirical coverage on the
//! same replicates and require gam to be at least as well calibrated:
//!
//!   | coverage(gam) - 0.95 | <= | coverage(statsmodels) - 0.95 | + 0.01
//!
//! For an unpenalized Gaussian linear model both engines are estimating the same
//! closed-form prediction variance, so gam should track the OLS optimum closely;
//! the calibration bar is the real quality claim and the baseline is only a sanity
//! floor. We still print the per-engine coverage and the worst-case SE magnitude
//! with `eprintln!` for context.


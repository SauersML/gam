//! Standing simulation-based-calibration / coverage suite (issue #1891).
//!
//! The calibration harness is the "ban-scanner for uncertainty": every surface
//! the library reports a coverage / credibility / size claim on is enumerated in
//! [`registry`] and gated by an audit mode chosen by its kind.
//!
//! This module holds the CONTRACT half — the registry and the completeness lint
//! that walks the public result payloads field by field and asserts each
//! uncertainty-bearing field maps to a registered target. The audit GATES
//! themselves are standing integration binaries so a single miscompiled gate
//! cannot take down the whole grouped `quality` binary:
//!   * the credible-band / predictive gates: `tests/sbc_*.rs`, including the
//!     parameterized per-family predictive-interval sweep
//!     `tests/sbc_family_predictive_interval_coverage.rs` (Poisson, Negative-
//!     Binomial, Gamma, Beta, Tweedie, Binomial);
//!   * conformal intervals: `tests/misc/predict/full_conformal_predict_route_quality.rs`;
//!   * the ALO/LOO predictive SE (#1869): `tests/quality/families/quality_vs_loo_psis_gaussian_smooth.rs`,
//!     `quality_vs_brute_force_loo_binomial_logit.rs`, `quality_vs_brute_force_loo_poisson_log.rs`,
//!     and `quality_vs_scipy_sandwich_glm_gaussian.rs`;
//!   * smooth-test size under the null (#1872/#1873): the standing
//!     `bug_hunt_smooth_significance_*` gates.
//!
//! The reusable engine (`run_sbc`/`audit_sbc_uniformity`, `run_coverage`/
//! `audit_coverage`, the Wilson verdict, the registry types) lives in
//! `gam_test_support::calibration`; this file is its consumer.

mod registry;

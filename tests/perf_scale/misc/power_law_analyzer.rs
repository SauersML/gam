//! Unit tests for `tests/common::{fit_power_law, report_power_law}`.
//!
//! The shared power-law analyzer is the gatekeeper for every scaling-law
//! probe in the suite (`standard_gam_scaling.rs`,
//! `margslope_inner_pirls_scaling.rs`, and any future probe). It enforces
//! the mission's "MEASURE FIRST" rule by refusing to extrapolate when the
//! fit is poor — which means a regression in this analyzer (e.g. swapping
//! `<` and `<=` on the R² gate, or letting NaN through the residual
//! computation) silently lets bad data drive large-scale-budget verdicts.
//! Pin the policy down with these tests.


//! Standing simulation-based-calibration / coverage suite (issue #1891).
//!
//! The calibration harness is the "ban-scanner for uncertainty": every surface
//! the library reports a coverage / credibility / size claim on is enumerated in
//! [`registry`] and gated by an audit mode chosen by its kind. This suite holds
//! (a) the registry + completeness lint and (b) the gates for the surfaces that
//! were NOT already covered by the six standing `tests/sbc_*.rs` gates —
//! coefficient Wald intervals, ALO/LOO predictive SEs (#1869), frequentist
//! test-size curves (#1872/#1873), and the ρ-posterior certificate SBC (#1810).
//!
//! The reusable engine (`run_sbc`/`audit_sbc_uniformity`, `run_coverage`/
//! `audit_coverage`, the Wilson verdict, the registry types) lives in
//! `gam_test_support::calibration`; these files are its consumers.

mod alo_se;
mod coefficient_wald;
mod registry;
mod rho_posterior;
mod test_size;

//! Dedicated integration-test target for the issue #979 survival fixture.
//!
//! Keeping this regression out of the broad `survival` test binary makes the
//! root-cause development loop independent of unrelated survival fixtures: a
//! targeted run compiles and executes exactly the N=2500, centers=12 public-API
//! contract below.

#[path = "survival/survival/survival_marginal_slope_1040_convergence.rs"]
mod survival_marginal_slope_1040_convergence;

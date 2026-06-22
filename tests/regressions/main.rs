//! Grouped integration-test crate root for regression / bug-hunt tests (issue #1146).
//!
//! The formerly-top-level regression, repro, issue, and bug-hunt crates are
//! included here as modules so they link as ONE binary instead of one linker
//! invocation each. Add new regression-family tests as a module here.

mod factors;
mod families;
mod gpu;
mod manifolds;
mod misc;
mod optimization;
mod predict;
mod sae;
mod smooths;
mod survival;

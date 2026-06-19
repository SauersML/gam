//! Micro-repro crate root — fast, small-n bug reproductions for iteration.
//!
//! WHY THIS CRATE EXISTS
//! Many gate tests run a full DGP at large n for hundreds of seconds. That is
//! the right *final* confirmation, but it makes the edit→test loop painfully
//! slow. A micro-repro is a tiny, deterministic test (same bug, same DGP
//! *shape*, n on the order of 100–300, reduced iterations) that reproduces the
//! defect in a few seconds so an owner can iterate. Reserve the slow gate test
//! for the final green.
//!
//! HOW TO ADD ONE (per-issue, non-build-breaking)
//!   1. Create `tests/micro/issue_<N>_<slug>.rs` containing ONE `#[test]`.
//!   2. Register it below with a `#[path = "micro/..."]` `mod` line.
//!   3. Keep it cheap to COMPILE (only pull `gam`/`ndarray`; no heavy fixtures)
//!      and cheap to RUN (target < 5s). Assert the SAME observable the gate
//!      test asserts, just at small n.
//!
//! This is its OWN integration-test crate (`tests/micro.rs`), so a broken or
//! WIP micro-repro here links/compiles independently and can NEVER block the
//! other test crates' builds — the failure mode that retired the old ad-hoc
//! `diag_1082` file. Build/run just this crate fast with:
//!   cargo test --no-run --test micro           # build only
//!   <target>/debug/deps/micro-<hash> --exact <test>   # run the binary directly
//! or via the harness:  GAM_ISO_TAG=<you> bash msi_iso.sh --fast=micro <test>

#[path = "micro/example_bspline_open_knot_derivative.rs"]
mod example_bspline_open_knot_derivative;

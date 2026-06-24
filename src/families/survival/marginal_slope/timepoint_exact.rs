//! Exact piecewise-cubic hazard timepoint integrals.
//!
//! One cached partition feeds a ladder of exact evaluations, split here by
//! integral kind — each an `impl SurvivalMarginalSlopeFamily` block reaching
//! the family internals through `use super::*;`:
//!
//! - [`partition`]: the cached cells + moment states + fixed partials shared by
//!   every pass (F, D, D_uv).
//! - [`first_full`]: the first-order and full second-order timepoint
//!   evaluations.
//! - [`directional`]: the single-direction extension of the full evaluation.
//! - [`bidirectional`]: the mixed `D_{d1} D_{d2}` bidirectional extension.
//! - [`contracted`]: the flex primary third/fourth contracted exact tensors and
//!   the general (flex-vs-rigid) entry points built on the above.
//!
//! The methods are inherent on `SurvivalMarginalSlopeFamily`, so they remain
//! callable wherever the type is in scope; the `pub(crate) use *;` globs below
//! also re-flatten any free items, preserving the parent's
//! `pub(crate) use timepoint_exact::*;`.

use super::*;

// #932-2 cutover: the hand directional / bidirectional timepoint producers are now
// test-only oracles — the production contracted path (`contracted.rs`) drives the
// single-source `flex_timepoint_inputs_generic` jet builder at `Jet3`/`Jet4`
// (`compute_survival_timepoint_{directional,bidirectional}_jet_from_cached`). The
// hand modules stay as the cross-check that pins the jet path (`flex_jet`'s
// `flex_timepoint_inputs_jet{3,4}_*_matches_hand_932` gates + the `tests.rs` FD
// witnesses), so they are gated to the test build (`*_oracle_tests`) rather than
// deleted.
#[cfg(test)]
mod bidirectional_oracle_tests;
mod contracted;
#[cfg(test)]
mod directional_oracle_tests;
mod first_full;
pub(crate) mod flex_jet;
mod partition;

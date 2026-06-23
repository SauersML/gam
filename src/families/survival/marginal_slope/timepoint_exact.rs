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

mod bidirectional;
mod contracted;
mod directional;
mod first_full;
mod flex_jet;
mod partition;

//! Outer-Hessian derivatives for the unified REML/LAML engine.
//!
//! The outer Hessian ∂²V/∂θᵢ∂θⱼ over the smoothing/extended hyperparameters is
//! delivered through two interchangeable representations — cost, never
//! capability, picks between them — split here by concern:
//!
//! - [`routing`]: representation routing and `(n, p, K)` scale decisions that
//!   select the dense assembly versus the matrix-free operator.
//! - [`traces`]: the derivative-trace computers shared by both paths (adjoint
//!   shortcut, fourth-derivative traces, IFT correction, base/cross logdet
//!   traces, dense-spectral and stochastic variants).
//! - [`kkt`]: the KKT-residual ρ corrections and the shared
//!   `RemlDerivativeWorkspace` gradient→Hessian intermediates.
//! - [`dense`]: the dense `K × K` assembled outer Hessian
//!   ([`compute_outer_hessian`]).
//! - [`operator`]: the matrix-free assembled outer-Hessian operator
//!   ([`UnifiedHessianOperator`] and `build_outer_hessian_operator`).
//!
//! Every submodule reaches its cross-concern dependencies (and the parent
//! `reml_outer_engine` namespace) through `use super::*;`; each item keeps the exact
//! visibility it carried before the split, and the globs below re-flatten the
//! submodules back into this module so external `…::outer_derivatives::<Name>`
//! paths and the parent's `pub use outer_derivatives::*;` are unchanged.

use super::*;

#[path = "outer_derivatives/dense.rs"]
mod dense_assembly;
mod kkt;
mod operator;
mod routing;
mod traces;

pub(crate) use dense_assembly::*;
pub(crate) use kkt::*;
pub(crate) use operator::*;
// Re-flatten `routing` at each item's own declared visibility: the outer-Hessian
// routing predicates/thresholds (`OuterHessianRoutePlan`, `outer_hessian_route_plan`,
// `prefer_outer_hessian_operator`, `MATRIX_FREE_OUTER_HESSIAN_{LARGE_N_THRESHOLD,
// DIM_AT_LARGE_N}`) are declared `pub` so the relocated families' cross-crate
// regression tests (#1521 carve) reach them through the flat
// `gam_solve::estimate::reml::reml_outer_engine::<Name>` path; everything else in
// `routing` is `pub(crate)`. A single `pub use` glob carries those visibilities
// verbatim. A capped `pub(crate) use routing::*` plus a separate `pub use
// routing::{…}` un-cap would import the same routing names at two visibilities —
// `pub(crate)` here and `pub` via the `use super::*` round-trip through the parent's
// re-export — which the compiler rejects as an ambiguous-visibility glob import.
pub use routing::*;
pub(crate) use traces::*;

// ═══════════════════════════════════════════════════════════════════════════
//  Extended Fellner–Schall (EFS) update for all hyperparameters

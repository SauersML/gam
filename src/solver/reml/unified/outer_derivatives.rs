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
//!   ([`UnifiedOuterHessianOperator`] and `build_outer_hessian_operator`).
//!
//! Every submodule reaches its cross-concern dependencies (and the parent
//! `unified` namespace) through `use super::*;`; each item keeps the exact
//! visibility it carried before the split, and the globs below re-flatten the
//! submodules back into this module so external `…::outer_derivatives::<Name>`
//! paths and the parent's `pub use outer_derivatives::*;` are unchanged.

use super::*;

mod dense;
mod kkt;
mod operator;
mod routing;
mod traces;

pub(crate) use dense::*;
pub(crate) use kkt::*;
pub(crate) use operator::*;
pub(crate) use routing::*;
pub(crate) use traces::*;

// ═══════════════════════════════════════════════════════════════════════════
//  Extended Fellner–Schall (EFS) update for all hyperparameters

//! Binomial location-scale family concern, decomposed into genuine sub-concern
//! modules.
//!
//! This was historically a single ~9k-line file. The numeric kernel, the two
//! families, and their exact-Newton Hessian workspaces share one tightly-coupled
//! namespace, so each sub-concern module opens with `use super::*;` and this root
//! forwards the parent gamlss flat namespace down via `pub(crate) use super::*;`
//! and re-exports each sub-concern flat via `pub use <module>::*`. Item
//! visibility is preserved exactly: originally-`pub` family/spec types stay
//! crate-public, and `pub(crate)` helpers stay crate-internal.
//!
//! - [`kernel`]                  — the binomial location-scale numeric kernel:
//!                                 the core row/derivative carriers, the q-algebra
//!                                 (`nonwiggle_q_*`, log-likelihood, expected
//!                                 information), the NLL towers, and the
//!                                 directional-coefficient free functions every
//!                                 family builds on.
//! - [`location_scale`]          — the `BinomialLocationScaleFamily`: its inherent
//!                                 math impl and the `CustomFamily` /
//!                                 `CustomFamilyGenerative` implementations.
//! - [`location_scale_workspace`]— the `BinomialLocationScaleHessianWorkspace`
//!                                 (direction keys/eta/coeff carriers and the
//!                                 exact-Newton joint-Hessian workspace impl).
//! - [`wiggle`]                  — the `BinomialLocationScaleWiggleFamily` variant:
//!                                 the struct, its inherent math impls, the
//!                                 Hessian row pieces, and the wiggle dh row
//!                                 coefficient carriers.
//! - [`wiggle_custom_family`]    — the wiggle family's `CustomFamily`
//!                                 implementation.
//! - [`wiggle_workspace`]        — the wiggle family's exact-Newton Hessian
//!                                 workspace and its `CustomFamilyGenerative` impl.

// Forward the parent gamlss flat namespace down to every sub-concern so their
// `use super::*;` resolves the cross-stack symbols, exactly as this module's
// own `use super::*;` did before the split.
pub(crate) use super::*;

mod kernel;
pub use kernel::*;

mod location_scale;
pub use location_scale::*;

mod location_scale_workspace;
pub use location_scale_workspace::*;

mod wiggle;
pub use wiggle::*;

mod wiggle_custom_family;
pub use wiggle_custom_family::*;

mod wiggle_workspace;
pub use wiggle_workspace::*;

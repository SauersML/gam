//! Gaussian (and adjacent log-link) location-scale family concern, decomposed
//! into genuine sub-concern modules.
//!
//! This was historically a single ~9k-line file. The math is one tightly-coupled
//! namespace (every family shares the location-scale joint-ψ machinery, the
//! shared weight/Hessian assembly, and the matrix-free operator), so each
//! sub-concern module opens with `use super::*;` and this root forwards the
//! parent gamlss flat namespace down via `pub(crate) use super::*;` and
//! re-exports each sub-concern flat via `pub use <module>::*`. Item visibility is
//! preserved exactly: originally-`pub` family/spec types stay crate-public, and
//! `pub(crate)` helpers stay crate-internal.
//!
//! - [`joint_psi`]              — the shared location-scale joint-ψ trait,
//!                                direction/drift carriers, the per-family
//!                                workspace, both Gaussian
//!                                `LocationScaleJointPsiFamily` impls, the
//!                                Gaussian joint row-scalar / directional-weight /
//!                                joint-Hessian assembly free functions, and the
//!                                `exp_sigma` derivative helper.
//! - [`location_scale`]         — the `GaussianLocationScaleFamily` itself: its
//!                                inherent math impl, the channel-Hessian view,
//!                                and the `CustomFamily` / `CustomFamilyGenerative`
//!                                implementations.
//! - [`row_coeff_operator`]     — the matrix-free `RowCoeffOperator` /
//!                                `DesignTwoBlockRowCoeffOperator` HyperOperators,
//!                                the Gaussian exact-Newton Hessian workspace, and
//!                                the two-block operator constructors.
//! - [`wiggle`]                 — the `GaussianLocationScaleWiggleFamily` variant
//!                                (geometry, row pieces, inherent math, directional
//!                                coefficients, `CustomFamily` /
//!                                `CustomFamilyGenerative` impls, and its Hessian
//!                                workspace).
//! - [`binomial_mean_wiggle`]   — the `BinomialMeanWiggleFamily` mean-only wiggle
//!                                family and its workspace.
//! - [`log_link`]               — the `PoissonLogFamily` and `GammaLogFamily`
//!                                log-link families and the shared diagonal-IRLS
//!                                evaluation kernel.
//! - [`binomial_locscale_decl`] — the `BinomialLocationScaleFamily` struct
//!                                declaration plus the macro that wires both
//!                                Binomial location-scale families into the shared
//!                                joint-ψ trait and their workspace type aliases
//!                                (the families' math lives in the sibling
//!                                `binomial` concern).

// Forward the parent gamlss flat namespace down to every sub-concern so their
// `use super::*;` resolves the cross-stack symbols, exactly as this module's
// own `use super::*;` did before the split.
pub(crate) use super::*;

mod joint_psi;
pub use joint_psi::*;

mod location_scale;
pub use location_scale::*;

mod row_coeff_operator;
pub use row_coeff_operator::*;

mod wiggle;
pub use wiggle::*;

mod binomial_mean_wiggle;
pub use binomial_mean_wiggle::*;

mod log_link;
pub use log_link::*;

mod binomial_locscale_decl;
pub use binomial_locscale_decl::*;

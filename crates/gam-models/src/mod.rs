// The shared `impl_reason_error_boilerplate!` derive was carved down into the
// base kernel crate (`gam-model-kernels`) under #1521. `#[macro_use]` brings
// its `#[macro_export]` macro into crate-wide textual scope so the unqualified
// `impl_reason_error_boilerplate! { .. }` call sites across the families that
// stayed here (survival / bms / gamlss / transformation_normal / inference)
// resolve unchanged.
#[macro_use]
extern crate gam_model_kernels;

// Declarative `bail_*` macros whose target error enums live in this crate.
// `#[macro_export]` places them at the crate root so the `bail_invalid_surv!`
// call sites across the relocated families resolve unchanged. `#[macro_use]`
// puts them in textual scope so same-crate call sites use the unqualified name
// (rustc forbids `crate::`-absolute paths to same-crate `macro_export` macros).
#[macro_use]
mod macros;
#[cfg(test)]
mod matern_collapse_1629_tests;
#[cfg(test)]
mod probe_1561_locscale_lambda_tests;

// `bail_*` shorthands whose error types were relocated to the neutral
// `gam-problem` crate. Re-exporting the `#[macro_export]` macros here makes
// `crate::bail_invalid_estim!` / `crate::bail_dim_custom!` resolve unchanged.
pub use gam_problem::{bail_dim_custom, bail_invalid_estim};

// ---------------------------------------------------------------------------
// Facade shims — thin `pub use` re-export modules that reconstruct the
// pre-carve flat namespace so the relocated families' `crate::X::*` paths
// resolve to their new homes in the dependency crates. No code is duplicated.
// ---------------------------------------------------------------------------

/// `crate::probability` → distributional primitives now in `gam-math`.
pub mod probability {
    pub use gam_math::probability::*;
}

/// `crate::util` → progress/span utilities now in `gam-runtime`.
pub mod util {
    pub use gam_runtime::{loop_progress, span};
}

/// `crate::quadrature` → quadrature rules now in `gam-solve`.
pub mod quadrature {
    pub use gam_solve::quadrature::*;
}

/// `crate::seeding` → seed-config carriers now in `gam-problem` (the `seeding`
/// module is private there; its items are surfaced at the crate root).
pub mod seeding {
    pub use gam_problem::{
        SeedConfig, SeedRiskProfile, clamp_seed_rho_to_bounds, normalize_seed_bounds,
    };
}

/// `crate::model_types` → fit-result / penalty-spec carriers now in `gam-solve`,
/// plus the finite-validation helper from `gam-problem` that `gam-solve` only
/// re-exports crate-internally.
pub mod model_types {
    pub use gam_solve::model_types::*;
    pub(crate) use gam_problem::validate_all_finite_estimation;
    // Finite-scalar guard consumed as `crate::model_types::ensure_finite_scalar_estimation`
    // (survival/location_scale); it lives in `gam-problem::finite_validation`.
    pub use gam_problem::ensure_finite_scalar_estimation;
}

/// `crate::inference::{quadrature, probability}` → the carved homes of the two
/// inference submodules that already live in leaf crates. The remaining
/// `crate::inference::{model, formula_dsl, smooth_test, predict_io, generative}`
/// submodules are still in the uncarved monolith (see blocker report) and are
/// intentionally *not* shimmed here — there is no leaf crate to re-export them
/// from yet.
pub mod inference;

pub mod fit_orchestration;
pub mod latent_coordinate;
pub mod protocol;
pub mod response_geometry;

pub mod binomial_multi;
pub mod block_layout;
pub mod bms;
pub(crate) mod coefficient_cost;
pub mod gpu_kernels;
pub mod custom_family;
pub mod family_runtime;
pub(crate) mod fast_channel;
pub(crate) mod fnv1a;
pub mod gamlss;
pub mod joint_penalty;
pub(crate) mod location_scale_engine;
pub mod marginal_slope_orthogonal;
pub mod marginal_slope_shared;
pub mod multinomial;
pub mod multinomial_posterior;
pub(crate) mod multinomial_reml;
pub use multinomial_reml::MultinomialLogitRowProgram;
pub mod outer_subsample;
pub mod parameter_block;
pub mod penalized_vector_glm;
pub(crate) mod row_kernel;
pub mod spatial_psi_bridge;
pub mod survival;
pub mod transformation_normal;
pub mod vector_response;
pub mod wiggle;

// The base kernel layer was carved into `gam-model-kernels` under #1521. These
// re-exports reconstruct the pre-carve flat namespace so every `crate::X::*`
// call site across the families that stayed here (and external `gam_models::X`
// consumers) resolves unchanged — no code is duplicated.
pub use gam_model_kernels::{
    cell_moment_family, cubic_cell_kernel, inverse_link, monotone_root, penalized_projection,
    scale_design, sigma_link,
};

pub use gam_identifiability::families::compiler::{
    BlockOrder, CompiledBlock, CompiledBlocks, CompilerError, RowHessian, RowJacobianOperator,
    compile,
};
pub use vector_response::{
    GaussianVectorLikelihood, MultinomialLogitLikelihood, VectorLikelihood, VectorNoise,
    VectorResponseTarget,
};

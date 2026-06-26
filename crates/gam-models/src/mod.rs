// Declarative `bail_*` / error-boilerplate macros whose target error enums live
// in this crate. `#[macro_export]` places them at the crate root so the
// `crate::bail_invalid_surv!` / `crate::impl_reason_error_boilerplate!` call sites
// across the relocated families resolve unchanged.
mod macros;

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
}

/// `crate::inference::{quadrature, probability}` → the carved homes of the two
/// inference submodules that already live in leaf crates. The remaining
/// `crate::inference::{model, formula_dsl, smooth_test, predict_io, generative}`
/// submodules are still in the uncarved monolith (see blocker report) and are
/// intentionally *not* shimmed here — there is no leaf crate to re-export them
/// from yet.
pub mod inference {
    pub use gam_math::probability;
    pub use gam_solve::quadrature;
}

pub mod binomial_multi;
pub mod block_layout;
pub mod bms;
pub mod cell_moment_family;
pub(crate) mod coefficient_cost;
pub mod gpu_kernels;
pub mod cubic_cell_kernel;
pub mod custom_family;
pub mod family_runtime;
pub(crate) mod fast_channel;
pub(crate) mod fnv1a;
pub mod gamlss;
pub mod inverse_link;
pub mod joint_penalty;
pub(crate) mod location_scale_engine;
pub mod marginal_slope_orthogonal;
pub mod marginal_slope_shared;
pub mod monotone_root;
pub mod multinomial;
pub(crate) mod multinomial_reml;
pub mod outer_subsample;
pub mod parameter_block;
pub mod penalized_vector_glm;
pub(crate) mod row_kernel;
pub mod scale_design;
pub mod sigma_link;
pub mod spatial_psi_bridge;
pub mod survival;
pub mod transformation_normal;
pub mod vector_response;
pub mod wiggle;

pub use gam_identifiability::families::compiler::{
    BlockOrder, CompiledBlock, CompiledBlocks, CompilerError, RowHessian, RowJacobianOperator,
    compile,
};
pub use vector_response::{
    GaussianVectorLikelihood, MultinomialLogitLikelihood, VectorLikelihood, VectorNoise,
    VectorResponseTarget,
};

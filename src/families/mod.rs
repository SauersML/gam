pub mod bms;
/// Backward-compatible path alias: all callers of
/// `crate::families::bernoulli_marginal_slope::*` continue to resolve
/// because every public symbol is re-exported here.
pub mod bernoulli_marginal_slope {
    pub use crate::families::bms::*;
    pub mod deviation_runtime {
        pub use crate::families::bms::deviation_runtime::*;
    }
    pub mod exact_kernel {
        pub use crate::families::bms::exact_kernel::*;
    }
}
pub mod bernoulli_marginal_slope_identifiability;
pub mod binomial_multi;
pub mod coefficient_cost;
pub mod cubic_cell_kernel;
pub mod custom_family;
pub mod gamlss;
pub mod identifiability_compiler;
pub mod inverse_link;
pub mod jet_partitions;
pub mod joint_penalty;
pub mod latent_interval;
pub mod latent_survival;
pub mod location_scale_engine;
pub mod lognormal_kernel;
pub mod marginal_slope_shared;
pub mod monotone_root;
pub mod multinomial;
pub mod multinomial_reml;
pub mod penalized_vector_glm;
pub mod row_kernel;
pub mod royston_parmar;
pub mod scale_design;
pub mod sigma_link;
pub mod strategy;
pub mod survival;
pub mod survival_construction;
pub mod survival_location_scale;
pub mod survival_marginal_slope;
pub mod survival_marginal_slope_identifiability;
pub mod survival_predict;
pub mod survival_time_constraints;
pub mod transformation_normal;
pub mod vector_response;

pub use identifiability_compiler::{
    AnchorRowEvaluator, BlockOrder, CompiledBlock, CompiledBlocks, CompilerError, RowHessian,
    RowJacobianOperator, compile,
};
pub use vector_response::{
    GaussianVectorLikelihood, MultinomialLogitLikelihood, VectorLikelihood, VectorNoise,
    VectorResponseTarget,
};

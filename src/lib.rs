#![deny(dead_code)]
#![deny(unused_variables)]
#![deny(unused_imports)]

include!(concat!(env!("OUT_DIR"), "/lint_errors.rs"));

/// Initialize faer's global parallelism backend to a Rayon pool sized at
/// `rayon::current_num_threads()`. Rayon's pool itself honors the standard
/// `RAYON_NUM_THREADS` environment variable on first use, so callers that
/// need to constrain the worker count (e.g. the benchmark harnesses) set it
/// once on the spawned subprocess and rayon picks it up natively.
///
/// Idempotent: only the first call has effect (guarded by `std::sync::Once`).
/// Without this, faer's global default is `Par::Seq` and matmul/factorizations
/// run single-threaded even when the host has many cores.
pub fn init_parallelism() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        faer::set_global_parallelism(faer::Par::rayon(0));
    });
}

pub mod families;
pub mod gpu;
pub mod inference;
pub mod linalg;
pub mod report;
pub mod resource;
pub mod solver;
mod span;
pub mod terms;
#[cfg(test)]
pub mod testing;
pub mod types;

pub use data::{
    encode_recordswith_inferred_schema, load_csvwith_inferred_schema, load_csvwith_schema,
};
pub use gpu::{GpuDecision, GpuPolicy, GpuWorkloadKind, GpuWorkloadShape};
pub use inference::{
    alo, data, diagnostics, generative, hmc, predict, probability, quadrature, sample,
};
pub use linalg::{faer_ndarray, matrix, utils};
pub use resource::{
    ByteLruCache, DerivativeStorageMode, MaterializationPolicy, MatrixMaterializationError,
    ProblemHints, ResidentBytes, ResourcePolicy,
};
pub use solver::{estimate, mixture_link, pirls, seeding, visualizer};
pub use terms::{basis, construction, hull, layout, smooth, term_builder};

pub use families::bernoulli_marginal_slope;
pub use families::custom_family;
pub use families::gamlss;
pub use families::survival;
pub use families::survival_construction;
pub use families::survival_location_scale;
pub use families::survival_marginal_slope;
pub use families::survival_predict;
pub use families::transformation_normal;
pub use solver::protocol::{
    LatentScoreSemantics, MarginalSlopeCalibrationProtocol, SurvivalMarginalSlopeProtocol,
};
pub use solver::workflow::{
    BernoulliMarginalSlopeFitRequest, BinomialLocationScaleFitRequest, FitConfig, FitRequest,
    FitResult, GaussianLocationScaleFitRequest, LatentBinaryFitRequest, LatentSurvivalFitRequest,
    LinkWiggleConfig, MaterializedModel, StandardBinomialWiggleConfig, StandardFitRequest,
    StandardFitResult, SurvivalLocationScaleFitRequest, SurvivalLocationScaleFitResult,
    SurvivalMarginalSlopeFitRequest, SurvivalTransformationFitRequest,
    SurvivalTransformationFitResult, TransformationNormalFitRequest, fit_from_formula, fit_model,
    is_binary_response, materialize, resolve_family, resolve_offset_column, resolve_weight_column,
};

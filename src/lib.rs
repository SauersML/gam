//! `gam` is a formula-first generalized additive model engine.
//!
//! Models are specified with a Wilkinson-style formula DSL and fit by
//! REML / LAML over Gaussian, binomial, Poisson, and Gamma GLMs, plus
//! location-scale, survival, marginal-slope, and response-geometry
//! extensions. Smoothing parameters are selected automatically;
//! posterior sampling uses NUTS where supported with a Gaussian Laplace
//! fallback elsewhere.
//!
//! ## Two interfaces
//!
//! - **Rust CLI (`gam`)** — fit, predict, report, diagnose, sample,
//!   generate. Built from `src/main.rs`.
//! - **Python library (`gamfit`)** — PyO3 bindings on top of this
//!   crate. See <https://gamfit.readthedocs.io/>.
//!
//! ## Smooth zoo
//!
//! Univariate P-splines, multivariate thin-plate, Matérn, and Duchon
//! radial bases, tensor products, and a family of **geometric smooths**
//! for predictor spaces that are not flat ℝᵈ:
//!
//! - 1-D cyclic / periodic B-splines and periodic Duchon
//! - Tensor products with one or more periodic margins (cylinder,
//!   torus, Möbius)
//! - Intrinsic S² smooths (Wahba reproducing kernel + spherical
//!   harmonics)
//! - Boundary-conditioned (clamped / anchored) 1-D B-splines
//!
//! `scripts/geometric_shapes_demo.py` showcases six topologies
//! (trefoil knot, latent-free loop, wobbly cylinder, lumpy sphere,
//! bumpy torus, Möbius strip) recovered from noisy 3-D point clouds,
//! including a self-validating quality report against analytic truth.
//!
//! ## Crate layout
//!
//! - [`families`] — likelihoods + their analytic gradients / Hessians
//! - [`solver`] — PIRLS, REML/LAML, and the joint blockwise optimiser
//! - [`terms`] — formula terms, basis construction, smooth specs
//! - [`inference`] — prediction, posterior sampling, diagnostics
//! - [`linalg`] — faer ↔ ndarray bridges + numerics helpers
//! - [`gpu`] — runtime CUDA dispatch for hot linear algebra paths
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

pub mod cache;
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
pub use gpu::{gpu_available, selected_gpu_info};
pub use inference::{
    alo, data, diagnostics, generative, hmc, predict, probability, quadrature, sample, smooth_test,
};
pub use linalg::{faer_ndarray, matrix, utils};
pub use resource::{
    ByteLruCache, DerivativeStorageMode, MaterializationPolicy, MatrixMaterializationError,
    ProblemHints, ResidentBytes, ResourcePolicy,
};
pub use solver::{estimate, gaussian_reml, mixture_link, pirls, seeding, visualizer};
pub use terms::{basis, construction, hull, latent_coord, layout, smooth, term_builder};

pub use gpu::{gpu_available, selected_gpu_info};

pub use families::bernoulli_marginal_slope;
pub use families::custom_family;
pub use families::gamlss;
pub use families::survival;
pub use families::survival_construction;
pub use families::survival_location_scale;
pub use families::survival_marginal_slope;
pub use families::survival_predict;
pub use families::transformation_normal;
pub use gpu::{GpuDeviceInfo, gpu_available, selected_gpu_info};
pub use solver::protocol::{
    LatentScoreSemantics, MarginalSlopeCalibrationProtocol, SurvivalMarginalSlopeProtocol,
};
pub use solver::workflow::{
    BernoulliMarginalSlopeFitRequest, BinomialLocationScaleFitRequest, FitConfig, FitRequest,
    FitResult, GaussianLocationScaleFitRequest, LatentBinaryFitRequest, LatentSurvivalFitRequest,
    LinkWiggleConfig, MaterializedModel, StandardBinomialWiggleConfig, StandardFitRequest,
    StandardFitResult, SurvivalLocationScaleFitRequest, SurvivalLocationScaleFitResult,
    SurvivalMarginalSlopeFitRequest, SurvivalTransformationFitRequest,
    SurvivalTransformationFitResult, SurvivalTransformationTermSpec,
    TransformationNormalFitRequest, WorkflowError, fit_from_formula, fit_model, is_binary_response,
    materialize, resolve_family, resolve_offset_column, resolve_weight_column,
};

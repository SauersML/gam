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

include!(concat!(env!("OUT_DIR"), "/lint_errors.rs"));

#[macro_use]
mod macros;

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

pub mod config_resolve;
pub mod families;
pub mod geometry;
pub mod gpu;
pub mod identifiability;
pub mod inference;
pub mod linalg;
pub mod report;
pub mod solver;
pub mod terms;
pub mod test_support;
pub mod types;
pub mod util;
pub mod warm_start;

#[path = "heartbeat.rs"]
pub mod process_monitor;

pub use data::{encode_recordswith_inferred_schema, load_csvwith_inferred_schema};
pub use geometry::{
    CircleManifold, EuclideanManifold, GeodesicIntegrator, GeometryError, GeometryResult,
    GrassmannManifold, ManifoldSpec, ProductManifold, RiemannianLBFGS, RiemannianManifold,
    RiemannianObjective, RiemannianTrustRegion, SpdManifold, SphereManifold, StiefelManifold,
    TorusManifold,
};
pub use gpu::GpuPolicy;
pub use inference::{
    alo, conformal, data, generative, higher_order, hmc, model_comparison, polya_gamma, predict,
    probability, psis, quadrature, rho_posterior, rho_uncertainty, sample, smooth_test,
};
pub use linalg::{faer_ndarray, matrix, utils};
// #931-#935 criterion calculus: the profiled-criterion abstraction
// (CriterionAtom / CriterionSum / Sensitivity) that kills the objective↔gradient
// desync class. Exposed as the staged public criterion-calculus interface it is
// designed to be; the #935 calculus that consumes it inside the inner REML path
// lands per the module's Migration law (one term per pass, FD-verified, old code
// deleted in the same commit). `PenaltySubspaceTrace` is the #901 spectral kernel
// the logdet atom's `Sensitivity` is built from.
pub use solver::estimate::reml::atoms::{
    BetaChannel, CriterionAtom, CriterionSum, HessianLogdetAtom, SampledBlockAtom, Sensitivity,
    StratumFingerprint, ThetaDirection,
};
pub use solver::estimate::reml::unified::PenaltySubspaceTrace;
// #986 frontier ρ-scaling: the per-atom decoupled EFS outer engine. `run_outer`
// auto-routes to it at frontier rho dimension; callers with a known
// arrow-border overlap drive `run_per_atom_efs` directly with an explicit
// `SharedBorderTopology` (`new` for a named border set, `disjoint` /
// `fully_coupled` for the two extremes).
pub use solver::resource::{
    ByteLruCache, DerivativeStorageMode, MaterializationPolicy, MatrixMaterializationError,
    ProblemHints, ResidentBytes, ResourcePolicy,
};
pub use solver::estimate::reml::per_atom_efs::{
    PerAtomEfsConfig, SharedBorderTopology, run_per_atom_efs,
};
pub use solver::{
    estimate, gaussian_reml, mixture_link, pirls, seeding, topology_selector, visualizer,
};
pub use terms::{basis, construction, hull, smooth, term_builder};

pub use families::custom_family;
pub use families::gamlss;
pub use families::survival;
pub use families::survival_location_scale;
pub use families::survival_marginal_slope;
pub use families::survival_predict;
pub use families::transformation_normal;
pub use gpu::GpuDeviceInfo;
pub use solver::protocol::{
    LatentScoreSemantics, MarginalSlopeCalibrationProtocol, SurvivalMarginalSlopeProtocol,
};
pub use solver::workflow::{
    BernoulliMarginalSlopeFitRequest, BinomialLocationScaleFitRequest, CrossFitScoreCalibration,
    CtnStage1Recipe, DispersionLocationScaleFitRequest, DispersionLocationScaleFitResult,
    FitConfig, FitRequest, FitResult, GaussianLocationScaleFitRequest, LatentBinaryFitRequest,
    LatentSurvivalFitRequest, LinkWiggleConfig, MaterializedModel, PreparedSurvivalTimeStack,
    ResidualCascadeInputs, SplineScanInputs, StandardBinomialWiggleConfig, StandardFitRequest,
    StandardFitResult, SurvivalLocationScaleFitRequest, SurvivalLocationScaleFitResult,
    SurvivalMarginalSlopeFitRequest, SurvivalTransformationFitRequest,
    SurvivalTransformationFitResult, SurvivalTransformationTermSpec,
    TransformationNormalFitRequest, WorkflowError, fit_from_formula, fit_model,
    fit_residual_cascade_from_formula, fit_spline_scan_from_formula, is_binary_response,
    materialize, prepare_survival_time_stack, residual_cascade_fast_path, resolve_family,
    resolve_offset_column, resolve_weight_column, spline_scan_fast_path,
};

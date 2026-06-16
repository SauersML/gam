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

// `config_resolve` was extracted from `src/main/` so the CLI driver and the
// Python FFI (gam-pyffi) can share the same JSON → FitConfig resolver; pull
// the current crate in under the `gam` alias so the file can keep using
// `gam::…` paths and stay drop-in for both compilation units.
extern crate self as gam;

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

#[path = "main/config_resolve.rs"]
pub mod config_resolve;
pub mod families;
pub mod geometry;
pub mod gpu;
pub mod identifiability;
pub mod inference;
pub mod linalg;
pub mod model_types;
/// Lower-layer Pareto-smoothed importance-sampling primitive. Self-contained
/// (no solver/inference deps); hosted at the crate root so `solver` and a
/// relocated `rho_uncertainty` can depend on it downward (#1135).
/// `crate::inference::psis` remains a back-compat re-export.
pub mod psis;
/// Lower-layer ρ-uncertainty (PSIS-on-ρ) diagnostic. Depends only on the
/// lower-layer `crate::psis`; hosted at the crate root so `solver` (its primary
/// consumer) can depend on it downward instead of importing *up* into
/// `inference` (#1135). `crate::inference::rho_uncertainty` remains a
/// back-compat re-export.
pub mod rho_uncertainty;
pub mod reml_contracts;
pub mod report;
/// Lower-layer resource-policy/materialization-budget types. Hosted at the
/// crate root (not under `solver`) so the `families` layer can name them
/// without importing *up* into `solver` (#1135). `crate::solver::resource`
/// remains a back-compat re-export.
pub mod resource;
/// Lower-layer outer-iteration row-subsampling/chunking primitives (RowSet,
/// ARROW_ROW_CHUNK). Hosted at the crate root so `families` can name them
/// without importing up into `solver`; `crate::solver::outer_subsample` is a
/// back-compat re-export.
pub mod outer_subsample;
pub(crate) mod rho_prior_eval;
pub mod solver;
/// Lower-layer outer-objective contract (the `OuterHessianOperator` trait,
/// `OuterEval`/`HessianResult`/`EfsEval`, and the capability enums) that the
/// `families` layer implements and returns. Hosted below `solver` so families
/// do not import *up* into `crate::solver::rho_optimizer` (#1135).
pub mod solver_contract;
pub mod terms;
pub mod test_support;
pub mod types;
pub mod util;
pub mod warm_start;

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
    probability, quadrature, rho_posterior, sample, smooth_test,
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
    BetaChannel, CriterionAtom, CriterionSum, HessianLogdetAtom, JeffreysLogdetAtom,
    PenaltyQuadAtom, SampledBlockAtom, Sensitivity, StratumFingerprint, ThetaDirection,
};
pub use solver::estimate::reml::reml_outer_engine::PenaltySubspaceTrace;
// #986 frontier ρ-scaling: the per-atom decoupled EFS outer engine. `run_outer`
// auto-routes to it at frontier rho dimension; callers with a known
// arrow-border overlap drive `run_per_atom_efs` directly with an explicit
// `SharedBorderTopology` (`new` for a named border set, `disjoint` /
// `fully_coupled` for the two extremes).
pub use solver::estimate::reml::per_atom_efs::{
    PerAtomEfsConfig, SharedBorderTopology, run_per_atom_efs,
};
pub use solver::resource::{
    ByteLruCache, DerivativeStorageMode, MaterializationPolicy, MatrixMaterializationError,
    ProblemHints, ResidentBytes, ResourcePolicy,
};
pub use solver::{
    estimate, gaussian_reml, mixture_link, pirls, seeding, topology_selector, visualizer,
};
pub use terms::{basis, construction, smooth, term_builder};

pub use families::custom_family;
pub use families::gamlss;
pub use families::transformation_normal;
pub use gpu::GpuDeviceInfo;
pub use solver::fit_orchestration::{
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
pub use solver::protocol::{
    LatentScoreSemantics, MarginalSlopeCalibrationProtocol, SurvivalMarginalSlopeProtocol,
};

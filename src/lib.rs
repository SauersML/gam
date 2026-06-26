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

/// Stack reserved for each worker in the global Rayon pool.
///
/// The deep numerical kernels run on Rayon workers, not just the calling
/// thread: the survival location-scale row Hessian/derivative operators
/// contract a `Tower4<9>` jet program (9⁴ fourth-order entries, ≈59 KiB per
/// scalar held by value, several towers live at once — see
/// `families::survival::location_scale::row_kernel::row_nll_tower`) inside
/// `RowSet::par_reduce_fold` and `into_par_iter` reductions, and faer's matmul
/// / factorization recursions fan out over the pool too. A single
/// `[Tower4<9>; 9]` array is already ≈0.5 MiB, so one row evaluation plus its
/// temporaries overruns Rayon's default ~2 MiB worker stack and aborts with
/// "thread '…' has overflowed its stack". The CLI entry point already drives
/// the *serial* path on a wide-stack worker (`CLI_WORKER_STACK_SIZE` in
/// `main.rs`); the Rayon pool that evaluates the identical kernel in parallel
/// for `n ≥ EVALUATE_PARALLEL_ROW_THRESHOLD` models must get the same headroom
/// or the parallel path overflows where the serial path no longer does. The
/// reservation is virtual address space — pages commit lazily, so the headroom
/// costs nothing until the deep jet paths actually use it.
const RAYON_WORKER_STACK_SIZE: usize = 64 << 20;

/// Initialize faer's global parallelism backend to a Rayon pool sized at
/// `rayon::current_num_threads()`. Rayon's pool itself honors the standard
/// `RAYON_NUM_THREADS` environment variable on first use, so callers that
/// need to constrain the worker count (e.g. the benchmark harnesses) set it
/// once on the spawned subprocess and rayon picks it up natively.
///
/// The global pool is built explicitly here with a wide per-worker stack
/// (`RAYON_WORKER_STACK_SIZE`) so the survival-LS `Tower4<9>` jet kernel — and
/// every other deep recursion that dispatches onto Rayon — has the same stack
/// headroom on a pool worker as it does on the CLI's wide-stack driver thread.
/// `build_global` only succeeds on the first thread to touch the pool; if some
/// earlier caller already initialized it we keep that pool rather than failing,
/// because the CLI drives `init_parallelism` before any Rayon use and so always
/// wins the race that matters.
///
/// Idempotent: only the first call has effect (guarded by `std::sync::Once`).
/// Without this, faer's global default is `Par::Seq` and matmul/factorizations
/// run single-threaded even when the host has many cores.
pub fn init_parallelism() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        gam_linalg::gpu_hook::register_gpu_dispatch(Box::new(
            crate::gpu::linalg_dispatch::CudaGemmDispatch,
        ));
        // Ignore the error returned when the global pool was already built by
        // an earlier caller: we cannot resize an existing pool, and the only
        // path that strictly needs the wide stack (the CLI) reaches this first.
        drop(
            rayon::ThreadPoolBuilder::new()
                .stack_size(RAYON_WORKER_STACK_SIZE)
                .build_global(),
        );
        faer::set_global_parallelism(faer::Par::rayon(0));
    });
}

#[cfg(test)]
mod gpu_dispatch_registration_tests {
    #[test]
    fn init_parallelism_registers_gpu_dispatch_hook() {
        crate::init_parallelism();
        assert!(gam_linalg::gpu_hook::gpu_dispatch().is_some());
    }
}

pub mod config_resolve;
pub mod families;
pub mod geometry;
pub mod gpu;
pub mod identifiability;
pub mod inference;
pub use gam_linalg as linalg;
pub mod model_types;
/// Lower-layer outer-iteration row-subsampling/chunking primitives (RowSet,
/// ARROW_ROW_CHUNK). Hosted at the crate root so `families` can name them
/// without importing up into `solver`.
pub mod outer_subsample;
/// Lower-layer Pareto-smoothed importance-sampling primitive. Self-contained
/// (no solver/inference deps); hosted at the crate root so `solver` and a
/// relocated `rho_uncertainty` can depend on it downward (#1135).
pub mod psis;
pub mod report;
pub(crate) mod rho_prior_eval;
/// Lower-layer ρ-uncertainty (PSIS-on-ρ) diagnostic. Depends only on the
/// lower-layer `crate::psis`; hosted at the crate root so `solver` (its primary
/// consumer) can depend on it downward instead of importing *up* into
/// `inference` (#1135).
pub mod rho_uncertainty;
pub mod solver;
pub mod terms;
pub mod test_support;
pub mod types;
pub mod util;

pub use gam_data as data;
pub use gam_data::{encode_recordswith_inferred_schema, load_csvwith_inferred_schema};
pub use geometry::{
    CircleManifold, EuclideanManifold, GeodesicIntegrator, GeometryError, GeometryResult,
    GrassmannManifold, ManifoldSpec, ProductManifold, RiemannianLBFGS, RiemannianManifold,
    RiemannianObjective, RiemannianTrustRegion, SpdManifold, SphereManifold, StiefelManifold,
    TorusManifold,
};
pub use gpu::GpuPolicy;
pub use inference::{
    alo, generative, higher_order, model_comparison, polya_gamma, probability, quadrature,
    rho_posterior, sample, smooth_test,
};
// The NUTS/HMC engine module was renamed `inference::hmc` -> `inference::hmc_io`.
// Its public types/functions (NutsConfig, NutsResult, FamilyNutsInputs,
// run_nuts_sampling_flattened_family, ...) are still consumed as `gam::hmc::*`
// by the sampling integration tests and downstream callers, so keep that path
// stable by re-exporting the renamed module under its old name.
pub use gam_linalg::{faer_ndarray, matrix, utils};
pub use inference::hmc_io as hmc;
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
pub use gam_problem::{
    DeclaredHessianForm, Derivative, EfsEval, HessianResult, OuterEval,
    OuterHessianMaterialization, OuterHessianOperator, OuterStrategyError,
};
pub use gam_runtime::resource::{
    ByteLruCache, DerivativeStorageMode, MaterializationPolicy, MatrixMaterializationError,
    ProblemHints, ResidentBytes, ResourcePolicy,
};
// Also keep the module path `gam::resource::…` reachable (it was previously
// `gam::resource`, before the move into the `gam-runtime` foundation crate).
// Integration tests and downstream code import the policy surface by module
// path (e.g. `gam::resource::STRICT_POLICY_NROWS_THRESHOLD`, which is not in
// the flattened re-export above); mirror the `warm_start` module re-export
// below so both the flattened items and the module path stay stable.
pub use gam_runtime::resource;
// The warm-start store (WarmStartStore, Fingerprinter, StoreOptions, ...) was
// relocated into the `gam-runtime` foundation crate. It was previously reachable
// as `gam::warm_start`; keep that path stable by re-exporting the module from
// the crate that now owns it (a normal, non-cyclic dependency).
pub use gam_runtime::warm_start;
pub use outer_subsample::{OuterScoreSubsample, RowSet, WeightedOuterRow};
pub use solver::estimate::reml::per_atom_efs::{
    PerAtomEfsConfig, SharedBorderTopology, run_per_atom_efs,
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
    TransformationNormalFitRequest, WorkflowError, constant_curvature_profiled_reml_scores,
    fit_from_formula, fit_model, fit_residual_cascade_from_formula, fit_spline_scan_from_formula,
    is_binary_response, materialize, prepare_survival_time_stack, residual_cascade_fast_path,
    resolve_family, resolve_offset_column, resolve_weight_column, spline_scan_fast_path,
};
pub use solver::protocol::{
    LatentScoreSemantics, MarginalSlopeCalibrationProtocol, SurvivalMarginalSlopeProtocol,
};

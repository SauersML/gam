pub mod active_set;
pub mod arrow_schur;
pub mod constrained_gaussian_reml;
pub mod constrained_posterior;
pub mod cone_reduction;
pub mod continuation_path;
// The custom-family blockwise carrier (`custom_family` + its persistent
// warm-start cache) was extracted into the `gam-custom-family` crate (#1521),
// which sits ABOVE gam-solve. gam-solve core no longer references it; consumers
// reach it as `gam_custom_family::*` (via the gam-models facade).
pub mod estimate;
pub mod evidence;
pub mod gauge;
pub mod gaussian_marginal;
pub mod gaussian_reml;
pub mod gaussian_reml_multi_penalty;
pub mod glm_sufficient_lane;
pub mod gpu;
pub mod gpu_kernels;
// Descended inference-tier numerics (#1521): ALO REML-evidence diagnostics whose
// deps are all ≤ gam-solve. Re-exported at the monolith root as
// `gam::inference::*` so cross-crate callers resolve unchanged.
pub mod inference;
pub mod inner_status;
pub mod latent_cache;
pub mod loop_guard;
pub mod mixture_link;
// #1521 carve: promoted for `gam-custom-family` (consumes
// `add_rho_block_dense_to_hessian`).
pub mod objective_base;
pub mod parallel_strategy;
pub mod penalty_invariance;
pub mod persistent_warm_start;
pub mod pirls;
pub(crate) mod priority_selection;
pub mod progress_log;
pub mod psi_gram_tensor;
// Pareto-smoothed importance sampling (descended #1521): leaf numerics with no
// crate-internal dependencies, consumed by `gam-inference`'s `rho_posterior`
// adequacy diagnostic and `model_comparison`, and by `gam-problem`.
pub mod psis;
// Rho-prior penalty/barrier evaluation (descended #1521): depends only on
// `gam_spec::RhoPrior`; consumed by `reml::atoms` and (after #1521) the
// extracted `gam-custom-family` crate — promoted `pub(crate)` -> `pub`.
pub mod rho_prior_eval;
pub mod residual_cascade;
pub mod rho_optimizer;
// The `#[macro_export]` error-bail macros live in `gam-problem` (its crate
// root). Importing `bail_invalid_estim` here makes `crate::bail_invalid_estim!`
// resolve at every gam-solve call site exactly as it did when these macros
// lived at the monolith crate root.
pub(crate) use gam_problem::bail_invalid_estim;
pub mod row_measure;
pub mod row_sampling_measure;
pub mod seeding;
pub mod sensitivity;
pub mod spline_scan;
pub(crate) mod startup_stats;
pub mod structure_search;
pub mod topology_formula;
pub mod topology_selector;
pub mod topology_stack_gaussian;
// #1521 carve: promoted `pub(crate)` -> `pub` so the extracted
// `gam-custom-family` crate (above gam-solve) can reach the warm-start
// artifact/transfer carriers it consumes.
pub mod warm_start_artifact;
pub mod warm_start_transfer;

pub use evidence::{
    CircularGaussianFit2d, GaussianMixtureCertificate, GaussianMixtureCheckpoint,
    GaussianMixtureConfig, GaussianMixtureFit, RingGaussianMixtureFit,
    StackingCertificate, StackingCheckpoint, StackingConfig, StackingError, StackingWeights,
    TopologyScoreScale, UnionStructure, solve_stacking_weights,
};
pub use topology_selector::{
    AdaptiveRungError, AdaptiveRungFailureStage, AdaptiveRungKind, AdaptiveRungOrderFailure,
    AutoTopologyKind, EvidenceCertification, Headline, HeldOutDensityProvider,
    MixtureRungFit, MixtureRungResult, PredictiveCandidateKind, PredictiveRaceCandidate,
    PredictiveRaceVerdict, RingOfClustersRungFit, RingOfClustersRungResult, STACKING_CV_FOLDS,
    STACKING_CV_SEED, TopologyAutoFitEvidence, TopologyAutoRankedFit, TopologyAutoSelector,
    TopologyAutoSelectorResult, TopologyCandidateEvidence, TopologyCandidateFailure,
    TopologyCandidateFailureStage, TopologyCandidateOutcome, TopologyCandidateRanked,
    TopologyCandidateSelectionResult, TopologyRaceParallelCandidate, TopologySelectionScoreKind,
    TopologySelectionScoreScale, adjudicate_predictive_race,
    deterministic_cv_folds_seeded, fit_free_cluster_rung,
    fit_ring_of_clusters_rung, run_topology_race_parallel,
    select_topology_candidate_lifecycle, select_topology_with_fit, tk_normalized_score,
};

/// Public re-export of the log-barrier configuration used by the REML/LAML
/// evaluators for monotonicity-constrained coefficients. Exposed so callers
/// (and integration tests) can construct and probe barrier objectives without
/// reaching through the private `estimate::reml::reml_outer_engine` path.
pub use estimate::reml::reml_outer_engine::BarrierConfig;
/// Re-exported for the Python bindings (`gam-pyffi`), which build their
/// analytic-penalty registry through the single shared descriptor parser that
/// also serves the in-process workflow pipeline. Exposed here so PyFFI can name
/// it without the (crate-private) `workflow` module being publicly reachable.
pub mod model_types;
pub mod quadrature;

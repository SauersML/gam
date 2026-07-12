pub mod active_set;
pub mod arrow_schur;
pub mod constrained_gaussian_reml;
pub mod continuation_path;
pub mod cross_node;
// The custom-family blockwise carrier (`custom_family` + its persistent
// warm-start cache) was extracted into the `gam-custom-family` crate (#1521),
// which sits ABOVE gam-solve. gam-solve core no longer references it; consumers
// reach it as `gam_custom_family::*` (via the gam-models facade).
pub mod estimate;
pub mod evidence;
pub mod gauge;
pub mod gaussian_reml;
pub mod glm_sufficient_lane;
pub mod gpu;
pub mod gpu_kernels;
// Descended inference-tier numerics (#1521): ALO REML-evidence diagnostics whose
// deps are all ≤ gam-solve. Re-exported at the monolith root as
// `gam::inference::*` so cross-crate callers resolve unchanged.
pub mod inference;
pub mod inner_status;
pub mod latent_cache;
pub mod latent_inner;
pub mod logdet_bounds;
pub mod loop_guard;
pub mod measure_jet_glm_sufficient;
pub mod measure_jet_gram_cache;
pub mod mixture_link;
// #1521 carve: promoted for `gam-custom-family` (consumes
// `add_rho_block_dense_to_hessian`).
pub mod objective_base;
pub mod orthogonal_reparam;
pub mod parallel_strategy;
pub mod persistent_warm_start;
pub mod pirls;
pub(crate) mod priority_selection;
pub mod progress_log;
pub mod psi_gram_tensor;
// Pareto-smoothed importance sampling (descended #1521): leaf numerics with no
// crate-internal dependencies, consumed by `reml::objective`, `rho_uncertainty`,
// and the monolith's `inference::{rho_posterior, model_comparison}` (which now
// reach it via the `gam-solve` re-export at the monolith crate root).
pub mod psis;
// Rho-prior penalty/barrier evaluation (descended #1521): depends only on
// `gam_spec::RhoPrior`; consumed by `reml::atoms` and (after #1521) the
// extracted `gam-custom-family` crate — promoted `pub(crate)` -> `pub`.
pub mod rho_prior_eval;
// Rho-uncertainty (Pareto-k heavy-tail) diagnostics (descended #1521): depends
// only on the descended `crate::psis`; consumed by `rho_optimizer::run`.
pub mod residual_cascade;
pub mod rho_optimizer;
pub mod rho_uncertainty;
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
pub mod streaming_border;
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
    CircularGaussianFit2d, EvidenceHvpLogDet, EvidenceIftGradientTerms, EvidenceLogDetSource,
    GaussianMixtureCertificate, GaussianMixtureCheckpoint, GaussianMixtureConfig,
    GaussianMixtureError, GaussianMixtureFit, RingGaussianMixtureFit, SelectedTopology,
    StackingCertificate, StackingCheckpoint, StackingConfig, StackingError, StackingWeights,
    TopologyCandidate, TopologyKind, TopologyScoreScale, TopologySelectOptions,
    UNION_STRUCTURE_LADDER, UnionComponentFit, UnionComponentKind, UnionStructure,
    UnionStructureFit, evidence_grad_rho, evidence_hessian_log_det,
    evidence_ift_gradient_correction, fit_gaussian_mixture, fit_ring_gaussian_mixture,
    fit_union_ladder, fit_union_structure, hessian_log_det_from_hvp, laplace_evidence,
    resume_gaussian_mixture, resume_stacking_weights, select_topology, solve_stacking_weights,
    union_per_point_log_density, union_responsibility_split,
};
pub use topology_selector::{
    AutoTopologyKind, EvidenceCertification, Headline, HeldOutDensityProvider, MIXTURE_K_LADDER,
    MixtureRungFit, MixtureRungResult, PredictiveCandidateKind, PredictiveRaceCandidate,
    PredictiveRaceVerdict, RingOfClustersRungFit, RingOfClustersRungResult, STACKING_CV_FOLDS,
    STACKING_CV_SEED, TopologyAutoFitEvidence, TopologyAutoRankedFit, TopologyAutoSelector,
    TopologyAutoSelectorResult, TopologyCandidateEvidence, TopologyCandidateFailure,
    TopologyCandidateFailureStage, TopologyCandidateOutcome, TopologyCandidateRanked,
    TopologyCandidateSelectionResult, TopologyRaceParallelCandidate, TopologySelectionScoreKind,
    TopologySelectionScoreScale, UnionRungFit, UnionRungResult, adjudicate_predictive_race,
    build_cv_log_density_table, deterministic_cv_folds, deterministic_cv_folds_seeded,
    fit_mixture_rung, fit_ring_of_clusters_rung, fit_union_candidate, fit_union_rung,
    mixture_density_provider, parse_union_name, ring_of_clusters_density_provider,
    run_topology_race_parallel, select_topology_candidate_lifecycle, select_topology_with_fit,
    select_topology_with_fit_parallel, tk_normalized_score, union_density_provider,
};

/// Process-wide counter of smoothing-corrections that took the sigma-cubature
/// (second-order) branch in
/// `estimate::reml::eval::RemlState::compute_smoothing_correction_auto`.
/// Re-exported so integration tests can snapshot it before/after a fit and
/// prove the cubature path (rather than the first-order linearization) was
/// actually exercised — see the #582 response-scale-equivariance regression.
pub use estimate::reml::eval::SMOOTHING_CORRECTION_CUBATURE_COUNT;
/// Public re-export of the log-barrier configuration used by the REML/LAML
/// evaluators for monotonicity-constrained coefficients. Exposed so callers
/// (and integration tests) can construct and probe barrier objectives without
/// reaching through the private `estimate::reml::reml_outer_engine` path.
pub use estimate::reml::reml_outer_engine::BarrierConfig;
/// Re-exported for the Python bindings (`gam-pyffi`), which must name the
/// covariance-correction error type without reaching through the private
/// `estimate::reml::reml_outer_engine` path.
pub use estimate::reml::reml_outer_engine::CorrectedCovarianceError;
/// Re-exported for the Python bindings (`gam-pyffi`), which build their
/// analytic-penalty registry through the single shared descriptor parser that
/// also serves the in-process workflow pipeline. Exposed here so PyFFI can name
/// it without the (crate-private) `workflow` module being publicly reachable.
pub mod model_types;
pub mod quadrature;

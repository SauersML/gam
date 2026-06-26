pub(crate) mod active_set;
pub mod arrow_schur;
pub mod continuation_path;
pub mod cross_node;
pub mod custom_family;
pub(crate) mod custom_family_persistent_warm_start;
pub mod estimate;
pub mod evidence;
pub mod gauge;
pub mod gaussian_reml;
pub mod glm_sufficient_lane;
pub mod gpu_kernels;
pub mod gpu;
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
pub(crate) mod objective_base;
pub mod orthogonal_reparam;
pub(crate) mod parallel_strategy;
pub mod persistent_warm_start;
pub mod pirls;
pub(crate) mod priority_selection;
pub mod psi_gram_tensor;
// Pareto-smoothed importance sampling (descended #1521): leaf numerics with no
// crate-internal dependencies, consumed by `reml::objective`, `rho_uncertainty`,
// and the monolith's `inference::{rho_posterior, model_comparison}` (which now
// reach it via the `gam-solve` re-export at the monolith crate root).
pub mod psis;
// Rho-prior penalty/barrier evaluation (descended #1521): depends only on
// `gam_spec::RhoPrior`; consumed by `custom_family::blockwise_solve` and
// `reml::atoms`.
pub(crate) mod rho_prior_eval;
// Rho-uncertainty (Pareto-k heavy-tail) diagnostics (descended #1521): depends
// only on the descended `crate::psis`; consumed by `rho_optimizer::run`.
pub mod rho_uncertainty;
pub mod residual_cascade;
pub mod rho_optimizer;
pub(crate) use gam_problem::riemannian_retraction;
// The `#[macro_export]` error-bail macros live in `gam-problem` (its crate
// root). Importing them here makes `crate::bail_invalid_estim!` /
// `crate::bail_dim_custom!` resolve at every gam-solve call site exactly as
// they did when these macros lived at the monolith crate root.
pub(crate) use gam_problem::{bail_dim_custom, bail_invalid_estim};
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
pub mod visualizer;
pub(crate) mod warm_start_artifact;
pub(crate) mod warm_start_transfer;

// The custom-family blockwise carrier (`fit`, `blockwise_solve`, `joint_newton`,
// …). Its prelude (`crate::custom_family`) re-exports the persistent
// warm-start entry points from `custom_family_persistent_warm_start`, the
// descended block data model + `CustomFamilyError` + `PenaltyMatrix` from
// `gam-problem`, and the `CustomFamily` trait + fit options + ψ design operators
// from `gam-model-api`, so each carrier submodule resolves them through its
// `use super::*` prelude exactly as it did when these lived in
// `families/custom_family/mod.rs`.

pub use evidence::{
    EvidenceHvpLogDet, EvidenceIftGradientTerms, EvidenceLogDetSource, GaussianMixtureConfig,
    GaussianMixtureFit, SelectedTopology, StackingConfig, StackingWeights, TopologyCandidate,
    TopologyKind, TopologyScoreScale, TopologySelectOptions, UNION_STRUCTURE_LADDER,
    UnionComponentFit, UnionComponentKind, UnionStructure, UnionStructureFit, evidence_grad_rho,
    evidence_hessian_log_det, evidence_ift_gradient_correction, fit_gaussian_mixture,
    fit_union_ladder, fit_union_structure, hessian_log_det_from_hvp, laplace_evidence,
    select_topology, solve_stacking_weights, union_per_point_log_density,
    union_responsibility_split,
};
pub use topology_selector::{
    AutoTopologyKind, CrossClassCandidate, CrossClassRaceVerdict, EvidenceCertification, Headline,
    HeldOutDensityProvider, MIXTURE_K_LADDER, MixtureRungFit, MixtureRungResult, STACKING_CV_FOLDS,
    STACKING_CV_SEED, TopologyAutoFitEvidence, TopologyAutoRankedFit, TopologyAutoSelector,
    TopologyAutoSelectorResult, TopologyRaceParallelCandidate, UnionRungFit, UnionRungResult,
    adjudicate_cross_class_race, build_cv_log_density_table, deterministic_cv_folds,
    deterministic_cv_folds_seeded, fit_mixture_rung, fit_union_candidate, fit_union_rung,
    mixture_density_provider, parse_union_name, run_topology_race_parallel,
    select_topology_with_fit, select_topology_with_fit_parallel, tk_normalized_score,
    union_density_provider,
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

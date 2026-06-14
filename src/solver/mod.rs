pub(crate) mod active_set;
pub mod arrow_schur;
pub mod continuation_path;
pub mod cross_node;
pub mod estimate;
pub mod evidence;
pub mod gauge;
pub mod gaussian_reml;
pub mod glm_sufficient_lane;
pub mod gpu;
pub mod grid_spline_2d;
pub mod identifiability_audit;
pub mod identifiability_canonical;
pub mod inner_status;
pub(crate) mod latent_cache;
pub mod latent_inner;
pub mod logdet_bounds;
pub mod loop_guard;
pub mod measure_jet_glm_sufficient;
pub mod measure_jet_gram_cache;
pub mod mixture_link;
pub mod orthogonal_reparam;
pub mod outer_strategy;
pub(crate) mod persistent_warm_start;
pub mod pirls;
pub(crate) mod priority_selection;
pub mod protocol;
pub mod psi_gram_tensor;
pub mod residual_cascade;
pub(crate) mod riemannian_retraction;
pub mod row_measure;
pub mod seeding;
pub mod sensitivity;
pub mod spline_scan;
pub(crate) mod startup_stats;
pub mod streaming_border;
pub mod structure_harvest;
pub mod structure_search;
pub mod topology_formula;
pub mod topology_selector;
pub mod visualizer;
pub mod workflow;

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
    AutoTopologyKind, CrossClassCandidate, CrossClassRaceVerdict, Headline, HeldOutDensityProvider,
    MIXTURE_K_LADDER, MixtureRungFit, MixtureRungResult, STACKING_CV_FOLDS,
    TopologyAutoFitEvidence, TopologyAutoRankedFit, TopologyAutoSelector,
    TopologyAutoSelectorResult, TopologyRaceParallelCandidate, UnionRungFit, UnionRungResult,
    adjudicate_cross_class_race, build_cv_log_density_table, deterministic_cv_folds,
    fit_mixture_rung, fit_union_candidate, fit_union_rung, mixture_density_provider,
    parse_union_name, run_topology_race_parallel, select_topology_with_fit,
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
/// reaching through the private `estimate::reml::unified` path.
pub use estimate::reml::unified::BarrierConfig;
/// Re-exported for the Python bindings (`gam-pyffi`), which must name the
/// covariance-correction error type without reaching through the private
/// `estimate::reml::unified` path.
pub use estimate::reml::unified::CorrectedCovarianceError;
/// Re-exported for the Python bindings (`gam-pyffi`), which build their
/// analytic-penalty registry through the single shared descriptor parser that
/// also serves the in-process workflow pipeline. Exposed here so PyFFI can name
/// it without the (crate-private) `workflow` module being publicly reachable.
pub use workflow::descriptors::build_analytic_penalty_registry_from_descriptors;

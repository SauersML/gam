pub(crate) mod active_set;
pub mod arrow_schur;
pub mod estimate;
pub mod evidence;
pub mod gpu;
pub mod identifiability_audit;
pub mod identifiability_canonical;
pub mod inner_status;
pub(crate) mod latent_cache;
pub mod latent_inner;
pub mod mixture_link;
pub mod orthogonal_reparam;
pub mod outer_strategy;
pub(crate) mod persistent_warm_start;
pub mod pirls;
pub(crate) mod priority_selection;
pub mod protocol;
pub(crate) mod riemannian_retraction;
pub mod row_measure;
pub mod seeding;
pub(crate) mod startup_stats;
pub mod topology_selector;
pub mod visualizer;
pub(crate) mod workflow;

pub use evidence::{
    EvidenceHvpLogDet, EvidenceIftGradientTerms, EvidenceLogDetSource, SelectedTopology,
    TopologyCandidate, TopologyKind, TopologyScoreScale, TopologySelectOptions, evidence_grad_rho,
    evidence_hessian_log_det, evidence_ift_gradient_correction, hessian_log_det_from_hvp,
    laplace_evidence, select_topology,
};
pub use topology_selector::{
    AutoTopologyKind, TopologyAutoFitEvidence, TopologyAutoRankedFit, TopologyAutoSelector,
    TopologyAutoSelectorResult, select_topology_with_fit, tk_normalized_score,
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

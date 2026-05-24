pub(crate) mod active_set;
pub mod arrow_schur;
pub mod arrow_schur_convergence_check;
pub mod estimate;
pub mod evidence;
pub mod gaussian_reml;
pub mod gpu;
pub(crate) mod latent_cache;
pub mod latent_inner;
pub mod mixture_link;
pub mod outer_strategy;
pub(crate) mod persistent_warm_start;
pub mod pirls;
pub mod protocol;
pub mod riemannian;
pub mod riemannian_retraction;
pub mod seeding;
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

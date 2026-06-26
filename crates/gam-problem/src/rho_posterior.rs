//! `ρ`-posterior certificate / escalation DATA types (contract-down #1521).
//!
//! These are the plain-data carriers that a fit result STORES
//! (`UnifiedFitResult::rho_posterior_{certificate,escalation}`) and that the
//! gam-solve REML evaluator returns. The COMPUTATION that produces them — the
//! PSIS certificate, the Tier-1 Gauss-Hermite quadrature, and the Tier-2 NUTS
//! escalation (which pulls the gam-inference `hmc_io` sampler) — stays UP in the
//! monolith `inference::rho_posterior`, which re-exports these types so its
//! construction sites name them unchanged. Contract-downed here (the neutral
//! criterion-contract crate) so gam-solve can store/return them without a
//! back-edge into gam-inference.

use ndarray::{Array1, Array2};
use std::sync::OnceLock;

/// Reliability tier read off the Pareto tail-shape `k̂` of the `ρ`-importance
/// weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RhoCertificate {
    /// `k̂ < 0.5`: the Laplace proposal is excellent — the plug-in (REML
    /// conditional) intervals plus the first-order `V_ρ` correction are
    /// certified adequate; `ρ`-uncertainty does not need a heavier treatment.
    PlugInCertified,
    /// `0.5 ≤ k̂ ≤ 0.7`: the proposal is usable but the self-normalized
    /// importance weights should be used to correct moments.
    ImportanceCorrect,
    /// `k̂ > 0.7`: the Laplace proposal poorly captures `π(ρ|y)`; escalate to
    /// quadrature (small `K`) or NUTS over `ρ`.
    Escalate,
}

impl RhoCertificate {
    pub fn from_k_hat(k_hat: f64) -> Self {
        if !k_hat.is_finite() || k_hat > 0.7 {
            RhoCertificate::Escalate
        } else if k_hat < 0.5 {
            RhoCertificate::PlugInCertified
        } else {
            RhoCertificate::ImportanceCorrect
        }
    }
}

/// The Tier-0 `ρ`-uncertainty certificate for a fit.
#[derive(Debug, Clone)]
pub struct RhoPosteriorCertificate {
    /// Pareto tail-shape of the importance weights — the reliability diagnostic.
    pub k_hat: f64,
    /// The reliability tier derived from `k_hat`.
    pub certificate: RhoCertificate,
    /// Number of proposal draws `M`.
    pub n_samples: usize,
    /// Self-normalized importance weights (length `M`), Pareto-smoothed. These
    /// turn the `M` conditional Gaussians into a free self-normalized mixture
    /// when the tier is `ImportanceCorrect`.
    pub weights: Array1<f64>,
    /// Kish effective sample size `(Σw)² / Σw²` — how many of the `M` draws are
    /// "really" contributing after importance weighting.
    pub effective_sample_size: f64,
}

/// One node of the criterion-closure Tier-1 mixture (#938): a `ρ` location, its
/// normalized posterior mass, and the exact profiled criterion value there.
#[derive(Debug, Clone)]
pub struct RhoMixtureNode {
    /// Smoothing parameters at this node.
    pub rho: Array1<f64>,
    /// Normalized node probability `w_m ∝ exp(−criterion(ρ_m) + criterion(ρ̂)) ×
    /// GH weight × exp(½‖z_m‖²)`.
    pub weight: f64,
    /// Normalized log node probability.
    pub log_weight: f64,
    /// Exact profiled criterion value at the node (`+∞` for infeasible nodes,
    /// which carry zero weight).
    pub cost: f64,
}

/// Tier-1 deliverable (#938): `π(ρ|y)` as a discrete mixture of conditional
/// Gaussians, with the posterior moment summary of `ρ` itself.
///
/// The conditional Gaussian at each node is exactly what the engine already
/// produces at fixed `ρ`; this struct owns the node locations and weights, and
/// `mixture_coefficient_covariance` (monolith `inference::rho_posterior`)
/// assembles the mixture-corrected coefficient covariance from per-node
/// conditionals supplied by the caller.
#[derive(Debug, Clone)]
pub struct RhoPosteriorMixture {
    /// Quadrature nodes with normalized weights (weights sum to 1).
    pub nodes: Vec<RhoMixtureNode>,
    /// Posterior mean of `ρ`: `Σ_m w_m ρ_m`.
    pub mean: Array1<f64>,
    /// Posterior covariance of `ρ`: `Σ_m w_m (ρ_m−ρ̄)(ρ_m−ρ̄)ᵀ`.
    pub covariance: Array2<f64>,
    /// Kish ESS of the node weights `(Σw)²/Σw²` — how non-Gaussian the exact
    /// posterior is relative to the Laplace proposal (max = node count).
    pub effective_sample_size: f64,
}

/// Tier-2 deliverable (#938): `π(ρ|y)` draws from NUTS with the exact profiled
/// gradient, whitened by the exact outer Hessian at `ρ̂`.
#[derive(Debug, Clone)]
pub struct RhoPosteriorSamples {
    /// Draws in ρ space: `(n_draws, K)`.
    pub samples: Array2<f64>,
    /// Posterior mean of `ρ`.
    pub mean: Array1<f64>,
    /// Posterior covariance of `ρ` (sample covariance of the draws).
    pub covariance: Array2<f64>,
    /// Split-chain R̂ mixing diagnostic.
    pub rhat: f64,
    /// Effective sample size.
    pub ess: f64,
    /// Whether the chains mixed (R̂ < 1.1).
    pub converged: bool,
}

/// The auto-selected escalation outcome when the Tier-0 certificate reads
/// [`RhoCertificate::Escalate`] (#938): Tier 1 (deterministic quadrature) for
/// `K ≤ 4`, Tier 2 (NUTS over `ρ`) for `K ≤ 16`, and an HONEST report that
/// escalation is unavailable beyond that — never a silently-degraded answer.
#[derive(Debug, Clone)]
pub enum RhoPosteriorEscalation {
    /// Tier 1: deterministic Gauss-Hermite mixture (`K ≤ 4`).
    Quadrature(RhoPosteriorMixture),
    /// Tier 2: NUTS draws with the exact profiled gradient (`5 ≤ K ≤ 16`).
    Nuts(RhoPosteriorSamples),
    /// Escalation could not run (dimension beyond the NUTS cap, or the chosen
    /// tier failed); intervals remain plug-in + first-order corrected, and the
    /// fit reports WHY.
    Unavailable { n_params: usize, reason: String },
}

// ───────────────────────── injected escalator trait (#1521) ──────────────────

/// The gam-inference-tier producer of the Tier-0 `ρ`-certificate and the
/// auto-selected Tier-1/Tier-2 escalation (trait-inversion #1521).
///
/// The COMPUTATION — the PSIS certificate, the Gauss-Hermite quadrature, and
/// the Tier-2 NUTS over `ρ` — pulls the gam-inference `hmc_io` sampler, so it
/// STAYS UP in the monolith `inference::rho_posterior`. That module implements
/// this trait over its real `rho_posterior_certificate` / `escalate_rho_posterior`
/// functions and injects the impl DOWN via [`set_rho_posterior_escalator`];
/// gam-solve's REML evaluator calls THROUGH [`rho_posterior_escalator`]. Only
/// neutral types (ndarray + the contract-downed `ρ`-posterior carriers) and
/// caller-supplied criterion closures cross this surface — no gam-inference type
/// is threaded, so the trait can live in this neutral crate.
///
/// When no impl is registered (a build that never links the sampler tier) the
/// getter returns `None` and gam-solve declines the certificate/escalation
/// entirely (`(None, None)`), leaving the plug-in + first-order intervals — its
/// existing decline outcome, no behavioral cliff and no stub.
pub trait RhoPosteriorEscalator: Send + Sync {
    /// Tier-0 PSIS `ρ`-certificate. `criterion` evaluates the outer criterion
    /// `−log π(ρ|y)` at a trial `ρ` (`None` for infeasible `ρ`). Returns `None`
    /// when the certificate cannot be formed (see the monolith implementation).
    fn rho_posterior_certificate(
        &self,
        rho_hat: &Array1<f64>,
        outer_hessian: &Array2<f64>,
        criterion: &dyn Fn(&Array1<f64>) -> Option<f64>,
        n_samples: Option<usize>,
    ) -> Option<RhoPosteriorCertificate>;

    /// Auto-selected escalation (Tier-1 quadrature / Tier-2 NUTS / honest
    /// `Unavailable`). `criterion` returns the exact profiled criterion value,
    /// `criterion_and_grad` the value plus the exact LAML `ρ`-gradient; both are
    /// `None` for infeasible `ρ`.
    fn escalate_rho_posterior(
        &self,
        rho_hat: &Array1<f64>,
        outer_hessian: &Array2<f64>,
        criterion: &mut dyn FnMut(&Array1<f64>) -> Option<f64>,
        criterion_and_grad: &mut (dyn FnMut(&Array1<f64>) -> Option<(f64, Array1<f64>)> + Send),
    ) -> RhoPosteriorEscalation;
}

static RHO_POSTERIOR_ESCALATOR: OnceLock<Box<dyn RhoPosteriorEscalator>> = OnceLock::new();

/// Register the monolith's `hmc_io`-backed `ρ`-posterior certificate/escalation
/// producer. Called once at process init by the gam-inference tier. First writer
/// wins; a later call is ignored (returns `Err` with the boxed value) so a
/// re-init can never swap a live producer mid-run.
pub fn set_rho_posterior_escalator(
    escalator: Box<dyn RhoPosteriorEscalator>,
) -> Result<(), Box<dyn RhoPosteriorEscalator>> {
    RHO_POSTERIOR_ESCALATOR.set(escalator)
}

/// The registered `ρ`-posterior certificate/escalation producer, or `None` when
/// the sampler tier is not linked / not yet initialized (gam-solve then declines
/// the certificate and escalation — a safe no-op leaving plug-in intervals).
pub fn rho_posterior_escalator() -> Option<&'static dyn RhoPosteriorEscalator> {
    RHO_POSTERIOR_ESCALATOR.get().map(|b| b.as_ref())
}

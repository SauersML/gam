//! `ρ`-posterior adequacy / escalation DATA types (contract-down #1521).
//!
//! These are the plain-data carriers that a fit result STORES
//! (`FitArtifacts::{rho_posterior, rho_posterior_escalation}`) and that the
//! gam-solve REML evaluator returns. The COMPUTATION that produces them — the
//! PSIS adequacy diagnostic, the Tier-1 Gauss-Hermite quadrature, and the Tier-2 NUTS
//! escalation (which pulls the gam-inference `hmc_io` sampler) — stays UP in the
//! monolith `inference::rho_posterior`, which re-exports these types so its
//! construction sites name them unchanged. Contract-downed here (the neutral
//! criterion-contract crate) so gam-solve can store/return them without a
//! back-edge into gam-inference.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::OnceLock;

/// Adequacy grade of the Laplace proposal, read off the Pareto tail-shape `k̂`
/// of the `ρ`-importance weights (#2946 T2).
///
/// It is a diagnostic with a sampling error, not a certificate: `k̂` comes from
/// `⌈√M⌉` tail excesses and carries the standard error
/// `gam_solve::psis::shape_standard_error`, so a grade whose `k̂` sits within a
/// few standard errors of a cutoff says nothing about which side the true shape
/// is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RhoProposalAdequacy {
    /// `k̂ < 0.5`: the importance weights have a finite second moment, so the
    /// plug-in (REML conditional) intervals plus the first-order `V_ρ`
    /// correction are adequate by this diagnostic; `ρ`-uncertainty does not need
    /// a heavier treatment.
    ///
    /// `PlugInCertified` is its legacy token: accepted when reading a payload
    /// written before #2946 T2, never written.
    #[serde(alias = "PlugInCertified")]
    PlugInAdequate,
    /// `0.5 ≤ k̂ ≤ 0.7`: the proposal is usable but the self-normalized
    /// importance weights should be used to correct moments.
    ImportanceCorrect,
    /// `k̂ > 0.7`: the Laplace proposal poorly captures `π(ρ|y)`; escalate to
    /// quadrature (small `K`) or NUTS over `ρ`.
    Escalate,
}

/// `k̂` above which the self-normalized importance estimator's central limit
/// theorem no longer applies and the proposal must be replaced rather than
/// reweighted (Vehtari, Simpson, Gelman, Yao & Gabry, *Pareto smoothed
/// importance sampling*, JMLR 25(72), 2024, §3): for `k > 0.7` the practical
/// pre-asymptotic convergence rate of PSIS collapses and the estimate is
/// declared unreliable. How finely a given draw count can resolve this
/// boundary is NOT a property of the cutoff: the fitted shape's standard error at
/// tail sample `n` is `√n(1+k)/(n+10)`, and a verdict against this
/// cutoff only means something when the truth sits several such standard errors
/// away from it.
pub const ESCALATE_K_HAT: f64 = 0.7;
/// `k̂` below which the proposal's importance weights have finite variance
/// (`k < 1/2` ⇒ the generalized-Pareto tail has a second moment), so the
/// plug-in answer plus the first-order correction needs no reweighting.
pub const PLUG_IN_ADEQUATE_K_HAT: f64 = 0.5;

impl RhoProposalAdequacy {
    pub fn from_k_hat(k_hat: f64) -> Self {
        if !k_hat.is_finite() || k_hat > ESCALATE_K_HAT {
            RhoProposalAdequacy::Escalate
        } else if k_hat < PLUG_IN_ADEQUATE_K_HAT {
            RhoProposalAdequacy::PlugInAdequate
        } else {
            RhoProposalAdequacy::ImportanceCorrect
        }
    }
}

/// The Tier-0 `ρ`-uncertainty adequacy diagnostic for a fit: the PSIS tail
/// shape of the Laplace proposal's importance weights and the grade read off it.
///
/// Every field persists with the fit, so `k_hat` and `effective_sample_size`
/// are finite by construction: the producer refuses a non-finite tail shape
/// ([`RhoPosteriorRefusal::TailShapeNotFinite`]) and weights that do not
/// normalize ([`RhoPosteriorRefusal::SmoothedWeightsNotNormalizable`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RhoPosteriorAdequacy {
    /// Pareto tail-shape of the importance weights — the reliability diagnostic.
    pub k_hat: f64,
    /// The adequacy grade read off `k_hat`. `certificate` is its legacy key:
    /// accepted when reading a payload written before #2946 T2, never written.
    #[serde(alias = "certificate")]
    pub adequacy: RhoProposalAdequacy,
    /// Number of proposal draws `M`.
    pub n_samples: usize,
    /// Kish effective sample size `(Σw)² / Σw²` — how many of the `M` draws are
    /// "really" contributing after importance weighting.
    pub effective_sample_size: f64,
}

/// Why the Tier-0 `ρ`-adequacy diagnostic could not be formed at `ρ̂` (#2627).
///
/// One arm per refusal site of the adequacy computation. It is a post-fit
/// diagnostic, so a refusal still publishes the fit, and the arm travels
/// with it: a caller asserts on the reason rather than reading a log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RhoPosteriorRefusal {
    /// The outer Hessian's shape does not match `ρ̂`.
    HessianShape { rows: usize, cols: usize, k: usize },
    /// The outer Hessian has no strict Cholesky factor, so there is no Gaussian
    /// proposal with covariance `H_ρ⁻¹`.
    HessianNotPositiveDefinite { detail: String },
    /// The criterion is infeasible at `ρ̂`.
    CriterionInfeasibleAtRhoHat,
    /// The criterion at `ρ̂` is not finite.
    CriterionNotFiniteAtRhoHat,
    /// No proposal draw has a finite criterion.
    NoFiniteProposal,
    /// Too few finite importance weights for the Pareto tail fit.
    TooFewFiniteWeights,
    /// The Pareto tail fit returned a non-finite shape, which grades nothing.
    TailShapeNotFinite,
    /// The smoothed importance weights do not sum to a positive finite total.
    SmoothedWeightsNotNormalizable,
}

impl fmt::Display for RhoPosteriorRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HessianShape { rows, cols, k } => {
                write!(f, "outer Hessian is {rows}x{cols}, expected {k}x{k}")
            }
            Self::HessianNotPositiveDefinite { detail } => {
                write!(f, "outer Hessian is not positive definite: {detail}")
            }
            Self::CriterionInfeasibleAtRhoHat => f.write_str("criterion is infeasible at rho_hat"),
            Self::CriterionNotFiniteAtRhoHat => f.write_str("criterion at rho_hat is not finite"),
            Self::NoFiniteProposal => f.write_str("no proposal draw has a finite criterion"),
            Self::TooFewFiniteWeights => {
                f.write_str("too few finite importance weights for the Pareto tail fit")
            }
            Self::TailShapeNotFinite => f.write_str("the Pareto tail shape is not finite"),
            Self::SmoothedWeightsNotNormalizable => {
                f.write_str("smoothed importance weights do not sum to a positive finite total")
            }
        }
    }
}

/// Why a fit carries no attempt at the Tier-0 `ρ`-adequacy diagnostic (#2627).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RhoPosteriorNotComputed {
    /// Only the REML evaluator's post-fit seam forms the diagnostic, and this
    /// fit was assembled on a route that does not pass through it.
    NotFormedOnThisRoute,
    /// The fit was run without inference (`FitOptions::compute_inference` is
    /// false), and the seam runs only inside the inference pass.
    InferenceNotRequested,
    /// No adequacy producer is registered (a build that never links the
    /// sampler tier).
    EscalatorUnregistered,
    /// The outer Hessian at `ρ̂` could not be formed.
    OuterHessianUnavailable { reason: String },
}

impl fmt::Display for RhoPosteriorNotComputed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFormedOnThisRoute => {
                f.write_str("this fit's route does not pass through the adequacy seam")
            }
            Self::InferenceNotRequested => {
                f.write_str("the fit was run without inference, where the adequacy seam runs")
            }
            Self::EscalatorUnregistered => f.write_str("no rho-posterior producer is registered"),
            Self::OuterHessianUnavailable { reason } => {
                write!(f, "the outer Hessian at rho_hat is unavailable: {reason}")
            }
        }
    }
}

/// What the Tier-0 `ρ`-adequacy diagnostic seam concluded for a fit (#2627).
///
/// It replaces an `Option` whose `None` merged four different facts: nothing to
/// grade, no producer, no outer Hessian, and a refusal whose reason reached only
/// a log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RhoPosteriorOutcome {
    /// No smoothing coordinate: there is nothing to grade.
    NotApplicable,
    /// The diagnostic was never attempted, and why.
    NotComputed(RhoPosteriorNotComputed),
    /// The diagnostic was attempted and refused, and why.
    Refused(RhoPosteriorRefusal),
    /// The diagnostic was formed. Its grade may still say the plug-in is
    /// inadequate. `Certified` is its legacy tag: accepted when reading a payload
    /// written before #2946 T2, never written.
    #[serde(alias = "Certified")]
    Assessed(RhoPosteriorAdequacy),
}

impl Default for RhoPosteriorOutcome {
    /// A fit assembled away from the REML evaluator's post-fit seam never formed
    /// the diagnostic.
    fn default() -> Self {
        Self::NotComputed(RhoPosteriorNotComputed::NotFormedOnThisRoute)
    }
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
/// produces at fixed `ρ`; this struct owns the node locations and weights.
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

/// The auto-selected escalation outcome when the Tier-0 grade reads
/// [`RhoProposalAdequacy::Escalate`] (#938): Tier 1 (deterministic quadrature) for
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

/// The gam-inference-tier producer of the Tier-0 `ρ`-adequacy diagnostic and the
/// auto-selected Tier-1/Tier-2 escalation (trait-inversion #1521).
///
/// The COMPUTATION — the PSIS adequacy diagnostic, the Gauss-Hermite quadrature, and
/// the Tier-2 NUTS over `ρ` — pulls the gam-inference `hmc_io` sampler, so it
/// STAYS UP in the monolith `inference::rho_posterior`. That module implements
/// this trait over its real `rho_posterior_adequacy` / `escalate_rho_posterior`
/// functions and injects the impl DOWN via [`set_rho_posterior_escalator`];
/// gam-solve's REML evaluator calls THROUGH [`rho_posterior_escalator`]. Only
/// neutral types (ndarray + the contract-downed `ρ`-posterior carriers) and
/// caller-supplied criterion closures cross this surface — no gam-inference type
/// is threaded, so the trait can live in this neutral crate.
///
/// When no impl is registered (a build that never links the sampler tier) the
/// getter returns `None`, and gam-solve records
/// [`RhoPosteriorNotComputed::EscalatorUnregistered`] and runs no escalation,
/// leaving the plug-in + first-order intervals.
pub trait RhoPosteriorEscalator: Send + Sync {
    /// Tier-0 PSIS `ρ`-adequacy diagnostic. `criterion` evaluates the outer criterion
    /// `−log π(ρ|y)` at a trial `ρ` (`None` for infeasible `ρ`). Returns
    /// `Ok(None)` when there is nothing to grade (`K = 0`) and the typed
    /// [`RhoPosteriorRefusal`] when the diagnostic cannot be formed.
    fn rho_posterior_adequacy(
        &self,
        rho_hat: &Array1<f64>,
        outer_hessian: &Array2<f64>,
        criterion: &dyn Fn(&Array1<f64>) -> Option<f64>,
        n_samples: Option<usize>,
    ) -> Result<Option<RhoPosteriorAdequacy>, RhoPosteriorRefusal>;

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

/// Register the monolith's `hmc_io`-backed `ρ`-posterior adequacy/escalation
/// producer. Called once at process init by the gam-inference tier. First writer
/// wins; a later call is ignored (returns `Err` with the boxed value) so a
/// re-init can never swap a live producer mid-run.
pub fn set_rho_posterior_escalator(
    escalator: Box<dyn RhoPosteriorEscalator>,
) -> Result<(), Box<dyn RhoPosteriorEscalator>> {
    RHO_POSTERIOR_ESCALATOR.set(escalator)
}

/// The registered `ρ`-posterior adequacy/escalation producer, or `None` when
/// the sampler tier is not linked / not yet initialized (gam-solve then records
/// `EscalatorUnregistered` and runs no escalation, leaving plug-in intervals).
pub fn rho_posterior_escalator() -> Option<&'static dyn RhoPosteriorEscalator> {
    RHO_POSTERIOR_ESCALATOR.get().map(|b| b.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_k_hat_below_half_is_plug_in_adequate() {
        assert_eq!(
            RhoProposalAdequacy::from_k_hat(0.0),
            RhoProposalAdequacy::PlugInAdequate
        );
        assert_eq!(
            RhoProposalAdequacy::from_k_hat(0.499),
            RhoProposalAdequacy::PlugInAdequate
        );
    }

    #[test]
    fn from_k_hat_between_half_and_point_seven_is_importance_correct() {
        assert_eq!(
            RhoProposalAdequacy::from_k_hat(0.5),
            RhoProposalAdequacy::ImportanceCorrect
        );
        assert_eq!(
            RhoProposalAdequacy::from_k_hat(0.7),
            RhoProposalAdequacy::ImportanceCorrect
        );
        assert_eq!(
            RhoProposalAdequacy::from_k_hat(0.65),
            RhoProposalAdequacy::ImportanceCorrect
        );
    }

    #[test]
    fn from_k_hat_above_point_seven_is_escalate() {
        assert_eq!(RhoProposalAdequacy::from_k_hat(0.701), RhoProposalAdequacy::Escalate);
        assert_eq!(RhoProposalAdequacy::from_k_hat(10.0), RhoProposalAdequacy::Escalate);
    }

    #[test]
    fn from_k_hat_nan_is_escalate() {
        assert_eq!(
            RhoProposalAdequacy::from_k_hat(f64::NAN),
            RhoProposalAdequacy::Escalate
        );
    }

    #[test]
    fn from_k_hat_infinity_is_escalate() {
        assert_eq!(
            RhoProposalAdequacy::from_k_hat(f64::INFINITY),
            RhoProposalAdequacy::Escalate
        );
    }

    #[test]
    fn rho_posterior_escalator_returns_none_when_unregistered() {
        // In tests, the monolith escalator is never injected — expect None.
        assert!(rho_posterior_escalator().is_none());
    }

    /// #2627: the outcome persists with the fit, so every arm must round-trip
    /// through the saved-model encoding unchanged, and a refusal must keep the
    /// reason a caller asserts on.
    #[test]
    fn every_rho_posterior_outcome_arm_round_trips_through_json_2627() {
        let outcomes = [
            RhoPosteriorOutcome::NotApplicable,
            RhoPosteriorOutcome::default(),
            RhoPosteriorOutcome::NotComputed(RhoPosteriorNotComputed::InferenceNotRequested),
            RhoPosteriorOutcome::NotComputed(RhoPosteriorNotComputed::EscalatorUnregistered),
            RhoPosteriorOutcome::NotComputed(RhoPosteriorNotComputed::OuterHessianUnavailable {
                reason: "singular".to_string(),
            }),
            RhoPosteriorOutcome::Refused(RhoPosteriorRefusal::HessianNotPositiveDefinite {
                detail: "Cholesky(NonPositivePivot { index: 0 })".to_string(),
            }),
            RhoPosteriorOutcome::Assessed(RhoPosteriorAdequacy {
                k_hat: 0.25,
                adequacy: RhoProposalAdequacy::PlugInAdequate,
                n_samples: 64,
                effective_sample_size: 61.5,
            }),
        ];
        for outcome in outcomes {
            let json = serde_json::to_string(&outcome).expect("outcome serializes");
            let back: RhoPosteriorOutcome = serde_json::from_str(&json).expect("outcome parses");
            assert_eq!(back, outcome, "{json}");
        }
        assert_eq!(
            RhoPosteriorOutcome::default(),
            RhoPosteriorOutcome::NotComputed(RhoPosteriorNotComputed::NotFormedOnThisRoute)
        );
    }

    /// #2946 T2: the saved-model encoding names the diagnostic for what it is.
    /// The written tokens are `Assessed`, `adequacy` and the grade names, and none
    /// of the words that called a sampled tail-shape grade a certificate.
    #[test]
    fn assessed_outcome_writes_adequacy_tokens_not_certificate_words_2946() {
        let json = serde_json::to_string(&RhoPosteriorOutcome::Assessed(RhoPosteriorAdequacy {
            k_hat: 0.25,
            adequacy: RhoProposalAdequacy::PlugInAdequate,
            n_samples: 64,
            effective_sample_size: 61.5,
        }))
        .expect("outcome serializes");
        assert_eq!(
            json,
            r#"{"Assessed":{"k_hat":0.25,"adequacy":"PlugInAdequate","n_samples":64,"effective_sample_size":61.5}}"#
        );
        for grade in [
            RhoProposalAdequacy::PlugInAdequate,
            RhoProposalAdequacy::ImportanceCorrect,
            RhoProposalAdequacy::Escalate,
        ] {
            let token = serde_json::to_string(&grade).expect("grade serializes");
            assert!(!token.to_ascii_lowercase().contains("certif"), "{token}");
        }
    }

    /// #2946 T2: a payload written before the rename (the v18 tokens `Certified`,
    /// `certificate` and `PlugInCertified`) still reads, as the same outcome, and
    /// writes back only the new tokens. The reader accepts the older payload
    /// version by design, so the legacy tokens are read-only aliases.
    #[test]
    fn legacy_certificate_tokens_read_and_rewrite_as_adequacy_2946() {
        let legacy = r#"{"Certified":{"k_hat":0.25,"certificate":"PlugInCertified","n_samples":64,"effective_sample_size":61.5}}"#;
        let read: RhoPosteriorOutcome = serde_json::from_str(legacy).expect("a legacy payload reads");
        assert_eq!(
            read,
            RhoPosteriorOutcome::Assessed(RhoPosteriorAdequacy {
                k_hat: 0.25,
                adequacy: RhoProposalAdequacy::PlugInAdequate,
                n_samples: 64,
                effective_sample_size: 61.5,
            })
        );
        assert_eq!(
            serde_json::to_string(&read).expect("outcome serializes"),
            r#"{"Assessed":{"k_hat":0.25,"adequacy":"PlugInAdequate","n_samples":64,"effective_sample_size":61.5}}"#
        );

        // Control: the same shapes without the aliases refuse the legacy payload,
        // so the read above rests on the aliases and not on a lenient parser.
        #[derive(Debug, Deserialize)]
        enum GradeWithoutAlias {
            PlugInAdequate,
            ImportanceCorrect,
            Escalate,
        }
        #[derive(Debug, Deserialize)]
        struct AdequacyWithoutAlias {
            k_hat: f64,
            adequacy: GradeWithoutAlias,
            n_samples: usize,
            effective_sample_size: f64,
        }
        #[derive(Debug, Deserialize)]
        enum OutcomeWithoutAlias {
            Assessed(AdequacyWithoutAlias),
        }
        let refused = serde_json::from_str::<OutcomeWithoutAlias>(legacy)
            .expect_err("without the aliases the legacy payload must not read");
        assert!(refused.to_string().contains("unknown variant `Certified`"), "{refused}");
        let current = serde_json::to_string(&read).expect("outcome serializes");
        let OutcomeWithoutAlias::Assessed(mirror) =
            serde_json::from_str::<OutcomeWithoutAlias>(&current).expect("the mirror reads new tokens");
        assert!(matches!(mirror.adequacy, GradeWithoutAlias::PlugInAdequate));
        assert_eq!(
            (mirror.k_hat, mirror.n_samples, mirror.effective_sample_size),
            (0.25, 64, 61.5)
        );
    }
}

use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

pub use crate::hull::PeeledHull;

/// Shared default for monotone wiggle/deviation blocks. Formula DSL defaults,
/// workflow configs, and runtime deviation blocks should all derive from this
/// type so reproducible presets do not drift across layers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WigglePenaltyConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_orders: Vec<usize>,
    pub double_penalty: bool,
    pub monotonicity_eps: f64,
}

impl WigglePenaltyConfig {
    pub fn cubic_triple_operator_default() -> Self {
        Self {
            degree: 3,
            num_internal_knots: 8,
            penalty_orders: vec![1, 2, 3],
            double_penalty: true,
            monotonicity_eps: 1e-4,
        }
    }
}

/// Shared engine-level link selector for generalized models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinkFunction {
    Logit,
    Probit,
    CLogLog,
    Sas,
    BetaLogistic,
    Identity,
    Log,
}

impl LinkFunction {
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            Self::Logit => "logit",
            Self::Probit => "probit",
            Self::CLogLog => "cloglog",
            Self::Sas => "sas",
            Self::BetaLogistic => "beta-logistic",
            Self::Identity => "identity",
            Self::Log => "log",
        }
    }
}

/// Supported inverse-link components for convex blended inverse links.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinkComponent {
    Probit,
    Logit,
    CLogLog,
    LogLog,
    Cauchit,
}

impl LinkComponent {
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            Self::Probit => "probit",
            Self::Logit => "logit",
            Self::CLogLog => "cloglog",
            Self::LogLog => "loglog",
            Self::Cauchit => "cauchit",
        }
    }
}

/// User-facing configuration for a blended inverse link.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MixtureLinkSpec {
    pub components: Vec<LinkComponent>,
    /// Free logits for components [0..K-2]. The final component logit is fixed at 0.
    pub initial_rho: Array1<f64>,
}

/// Runtime blended-link state with precomputed softmax weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MixtureLinkState {
    pub components: Vec<LinkComponent>,
    /// Free logits for components [0..K-2]. The final component logit is fixed at 0.
    pub rho: Array1<f64>,
    /// Softmax-normalized component weights (length K).
    pub pi: Array1<f64>,
}

impl MixtureLinkState {
    /// Construct with the size invariants enforced. A `MixtureLinkState` is
    /// only well-formed when `components.len() == pi.len()` and
    /// `rho.len() + 1 == components.len()` (the last logit is fixed at zero).
    /// This constructor surfaces those constraints at the type boundary so
    /// downstream consumers can rely on them without re-checking.
    #[inline]
    pub fn new(
        components: Vec<LinkComponent>,
        rho: Array1<f64>,
        pi: Array1<f64>,
    ) -> Result<Self, String> {
        let k = components.len();
        if k < 2 {
            return Err(format!(
                "MixtureLinkState requires at least two components, got {k}"
            ));
        }
        if pi.len() != k {
            return Err(format!(
                "MixtureLinkState pi has length {}, expected {}",
                pi.len(),
                k
            ));
        }
        if rho.len() + 1 != k {
            return Err(format!(
                "MixtureLinkState rho has length {}, expected {}",
                rho.len(),
                k - 1
            ));
        }
        Ok(Self {
            components,
            rho,
            pi,
        })
    }

    /// Number of components in the mixture. Always >= 2 by invariant.
    #[inline]
    pub fn num_components(&self) -> usize {
        debug_assert_eq!(self.components.len(), self.pi.len());
        debug_assert_eq!(self.rho.len() + 1, self.components.len());
        self.components.len()
    }
}

/// User-facing configuration for the continuous sinh-arcsinh inverse link.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SasLinkSpec {
    pub initial_epsilon: f64,
    pub initial_log_delta: f64,
}

/// Runtime SAS link state with cached positive tail parameter.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SasLinkState {
    pub epsilon: f64,
    /// Raw optimization parameter.
    pub log_delta: f64,
    /// Effective tail parameter delta used in evaluation.
    /// With current bounded parameterization:
    /// delta = exp(B * tanh(log_delta / B)), B = SAS_LOG_DELTA_BOUND.
    pub delta: f64,
}

/// Fixed latent Gaussian scale for the exact marginal cloglog family.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LatentCLogLogState {
    pub latent_sd: f64,
}

impl LatentCLogLogState {
    #[inline]
    pub fn new(latent_sd: f64) -> Result<Self, String> {
        if !latent_sd.is_finite() || latent_sd < 0.0 {
            return Err(format!(
                "latent cloglog standard deviation must be finite and >= 0, got {latent_sd}"
            ));
        }
        Ok(Self { latent_sd })
    }
}

/// Parameterized inverse-link selector used where mu/derivatives are evaluated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InverseLink {
    Standard(LinkFunction),
    LatentCLogLog(LatentCLogLogState),
    Sas(SasLinkState),
    BetaLogistic(SasLinkState),
    Mixture(MixtureLinkState),
}

impl InverseLink {
    #[inline]
    pub fn link_function(&self) -> LinkFunction {
        match self {
            Self::Standard(link) => *link,
            Self::LatentCLogLog(_) => LinkFunction::CLogLog,
            Self::Sas(_) => LinkFunction::Sas,
            Self::BetaLogistic(_) => LinkFunction::BetaLogistic,
            Self::Mixture(_) => LinkFunction::Logit,
        }
    }

    #[inline]
    pub fn mixture_state(&self) -> Option<&MixtureLinkState> {
        match self {
            Self::Mixture(state) => Some(state),
            _ => None,
        }
    }

    #[inline]
    pub fn sas_state(&self) -> Option<&SasLinkState> {
        match self {
            Self::Sas(state) | Self::BetaLogistic(state) => Some(state),
            _ => None,
        }
    }

    #[inline]
    pub fn latent_cloglog_state(&self) -> Option<&LatentCLogLogState> {
        match self {
            Self::LatentCLogLog(state) => Some(state),
            _ => None,
        }
    }

}

/// Fixed prior family for smoothing parameters in joint HMC refinement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RhoPrior {
    Flat,
    Normal {
        mean: f64,
        sd: f64,
    },
    /// Gamma(shape, rate) conjugate hyperprior on the precision lambda = exp(rho).
    ///
    /// The REML/LAML objective is minimized, so this contributes
    /// `rate * exp(rho) - (shape - 1) * rho` up to an additive constant. For a
    /// block with effective dimension n_p and centered quadratic
    /// `(beta - mu)'S_p(beta - mu)`, the conditional posterior is
    /// `Gamma(shape + n_p/2, rate + quadratic/2)` and the closed-form MAP
    /// precision is `(shape + n_p/2 - 1) / (rate + quadratic/2)`.
    /// `Gamma(1, 0)` is the explicit flat/default case and reproduces the
    /// current MacKay/Tipping fixed point.
    GammaPrecision {
        shape: f64,
        rate: f64,
    },
    /// Coordinate-specific priors for models whose smoothing parameters do
    /// not share one prior family, such as nested coefficient groups.
    Independent(Vec<RhoPrior>),
}

impl Default for RhoPrior {
    fn default() -> Self {
        Self::Normal { mean: 0.0, sd: 3.0 }
    }
}

/// Engine-level likelihood selector used by generic APIs.
/// Some families remain restricted to domain-specific entrypoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LikelihoodFamily {
    GaussianIdentity,
    BinomialLogit,
    BinomialProbit,
    BinomialCLogLog,
    BinomialLatentCLogLog,
    BinomialSas,
    BinomialBetaLogistic,
    BinomialMixture,
    PoissonLog,
    GammaLog,
    RoystonParmar,
}

impl LikelihoodFamily {
    #[inline]
    pub fn link_function(self) -> LinkFunction {
        match self {
            Self::GaussianIdentity | Self::RoystonParmar => LinkFunction::Identity,
            Self::PoissonLog | Self::GammaLog => LinkFunction::Log,
            Self::BinomialLogit | Self::BinomialMixture => LinkFunction::Logit,
            Self::BinomialProbit => LinkFunction::Probit,
            Self::BinomialCLogLog | Self::BinomialLatentCLogLog => LinkFunction::CLogLog,
            Self::BinomialSas => LinkFunction::Sas,
            Self::BinomialBetaLogistic => LinkFunction::BetaLogistic,
        }
    }

    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            Self::GaussianIdentity => "gaussian",
            Self::BinomialLogit => "binomial-logit",
            Self::BinomialProbit => "binomial-probit",
            Self::BinomialCLogLog => "binomial-cloglog",
            Self::BinomialLatentCLogLog => "latent-cloglog-binomial",
            Self::BinomialSas => "binomial-sas",
            Self::BinomialBetaLogistic => "binomial-beta-logistic",
            Self::BinomialMixture => "binomial-blended-inverse-link",
            Self::PoissonLog => "poisson-log",
            Self::GammaLog => "gamma-log",
            Self::RoystonParmar => "royston-parmar",
        }
    }

    #[inline]
    pub fn pretty_name(self) -> &'static str {
        match self {
            Self::GaussianIdentity => "Gaussian Identity",
            Self::BinomialLogit => "Binomial Logit",
            Self::BinomialProbit => "Binomial Probit",
            Self::BinomialCLogLog => "Binomial CLogLog",
            Self::BinomialLatentCLogLog => "Latent CLogLog Binomial",
            Self::BinomialSas => "Binomial SAS",
            Self::BinomialBetaLogistic => "Binomial Beta-Logistic",
            Self::BinomialMixture => "Binomial Blended Inverse-Link",
            Self::PoissonLog => "Poisson Log",
            Self::GammaLog => "Gamma Log",
            Self::RoystonParmar => "Royston Parmar",
        }
    }

    /// Whether the shared Jeffreys/Firth implementation is available for this
    /// likelihood family.
    #[inline]
    pub fn supports_firth(self) -> bool {
        matches!(self, Self::BinomialLogit)
    }

    /// `true` for the latent-cloglog binomial family — checked at many sites
    /// to short-circuit the latent-Gaussian quadrature path.
    #[inline]
    pub fn is_latent_cloglog(self) -> bool {
        matches!(self, Self::BinomialLatentCLogLog)
    }

    /// `true` for the blended/mixture-of-inverse-links binomial family.
    #[inline]
    pub fn is_binomial_mixture(self) -> bool {
        matches!(self, Self::BinomialMixture)
    }

    /// `true` for the SAS sinh-arcsinh binomial family.
    #[inline]
    pub fn is_binomial_sas(self) -> bool {
        matches!(self, Self::BinomialSas)
    }

    /// `true` for the beta-logistic binomial family.
    #[inline]
    pub fn is_binomial_beta_logistic(self) -> bool {
        matches!(self, Self::BinomialBetaLogistic)
    }

    #[inline]
    pub fn is_binomial(self) -> bool {
        matches!(
            self,
            Self::BinomialLogit
                | Self::BinomialProbit
                | Self::BinomialCLogLog
                | Self::BinomialLatentCLogLog
                | Self::BinomialSas
                | Self::BinomialBetaLogistic
                | Self::BinomialMixture
        )
    }

    #[inline]
    pub fn default_scale_metadata(self) -> LikelihoodScaleMetadata {
        match self {
            Self::GaussianIdentity => LikelihoodScaleMetadata::ProfiledGaussian,
            Self::GammaLog => LikelihoodScaleMetadata::EstimatedGammaShape { shape: 1.0 },
            Self::BinomialLogit
            | Self::BinomialProbit
            | Self::BinomialCLogLog
            | Self::BinomialLatentCLogLog
            | Self::BinomialSas
            | Self::BinomialBetaLogistic
            | Self::BinomialMixture
            | Self::PoissonLog => LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 },
            Self::RoystonParmar => LikelihoodScaleMetadata::Unspecified,
        }
    }
}

/// GLM-compatible likelihood families (survival families excluded by type).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlmLikelihoodFamily {
    GaussianIdentity,
    BinomialLogit,
    BinomialProbit,
    BinomialCLogLog,
    BinomialSas,
    BinomialBetaLogistic,
    BinomialMixture,
    PoissonLog,
    GammaLog,
}

impl GlmLikelihoodFamily {
    #[inline]
    pub fn link_function(self) -> LinkFunction {
        LikelihoodFamily::from(self).link_function()
    }

    #[inline]
    pub fn supports_firth(self) -> bool {
        LikelihoodFamily::from(self).supports_firth()
    }
}

/// How a likelihood's scale parameter is handled by the fit/result contract.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LikelihoodScaleMetadata {
    /// Gaussian identity fits profile sigma outside the fixed-scale GLM machinery.
    ProfiledGaussian,
    /// Fixed exponential-dispersion parameter `phi`.
    FixedDispersion { phi: f64 },
    /// Fixed Gamma shape `k`, equivalent to `phi = 1 / k`.
    FixedGammaShape { shape: f64 },
    /// Gamma shape `k` estimated jointly with the mean model.
    EstimatedGammaShape { shape: f64 },
    /// The engine does not expose fixed-scale semantics for this family.
    Unspecified,
}

impl LikelihoodScaleMetadata {
    #[inline]
    pub fn fixed_phi(self) -> Option<f64> {
        match self {
            Self::FixedDispersion { phi } => Some(phi),
            Self::FixedGammaShape { shape } | Self::EstimatedGammaShape { shape } => {
                Some(1.0 / shape)
            }
            Self::ProfiledGaussian | Self::Unspecified => None,
        }
    }

    #[inline]
    pub fn gamma_shape(self) -> Option<f64> {
        match self {
            Self::FixedGammaShape { shape } | Self::EstimatedGammaShape { shape } => Some(shape),
            _ => None,
        }
    }

    #[inline]
    pub fn gamma_shape_is_estimated(self) -> bool {
        matches!(self, Self::EstimatedGammaShape { .. })
    }
}

/// Whether a stored log-likelihood includes response-only normalization constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLikelihoodNormalization {
    Full,
    OmittingResponseConstants,
    UserProvided,
}

/// Explicit GLM likelihood specification: family plus scale semantics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GlmLikelihoodSpec {
    pub family: GlmLikelihoodFamily,
    pub scale: LikelihoodScaleMetadata,
}

impl GlmLikelihoodSpec {
    #[inline]
    pub fn canonical(family: GlmLikelihoodFamily) -> Self {
        // GLM families are a strict subset of LikelihoodFamily, so the
        // canonical scale comes from the unified table to avoid drift.
        let scale = LikelihoodFamily::from(family).default_scale_metadata();
        Self { family, scale }
    }

    #[inline]
    pub fn link_function(self) -> LinkFunction {
        self.family.link_function()
    }

    #[inline]
    pub fn response_family(self) -> LikelihoodFamily {
        self.family.into()
    }

    #[inline]
    pub fn fixed_phi(self) -> Option<f64> {
        self.scale.fixed_phi()
    }

    #[inline]
    pub fn gamma_shape(self) -> Option<f64> {
        self.scale.gamma_shape()
    }

    #[inline]
    pub fn with_gamma_shape(mut self, shape: f64) -> Self {
        self.scale = match self.scale {
            LikelihoodScaleMetadata::FixedGammaShape { .. } => {
                LikelihoodScaleMetadata::FixedGammaShape { shape }
            }
            LikelihoodScaleMetadata::EstimatedGammaShape { .. } => {
                LikelihoodScaleMetadata::EstimatedGammaShape { shape }
            }
            _ if self.family == GlmLikelihoodFamily::GammaLog => {
                LikelihoodScaleMetadata::EstimatedGammaShape { shape }
            }
            other => other,
        };
        self
    }
}

impl TryFrom<LikelihoodFamily> for GlmLikelihoodFamily {
    type Error = &'static str;

    fn try_from(value: LikelihoodFamily) -> Result<Self, Self::Error> {
        match value {
            LikelihoodFamily::GaussianIdentity => Ok(Self::GaussianIdentity),
            LikelihoodFamily::BinomialLogit => Ok(Self::BinomialLogit),
            LikelihoodFamily::BinomialProbit => Ok(Self::BinomialProbit),
            LikelihoodFamily::BinomialCLogLog | LikelihoodFamily::BinomialLatentCLogLog => {
                Ok(Self::BinomialCLogLog)
            }
            LikelihoodFamily::BinomialSas => Ok(Self::BinomialSas),
            LikelihoodFamily::BinomialBetaLogistic => Ok(Self::BinomialBetaLogistic),
            LikelihoodFamily::BinomialMixture => Ok(Self::BinomialMixture),
            LikelihoodFamily::PoissonLog => Ok(Self::PoissonLog),
            LikelihoodFamily::GammaLog => Ok(Self::GammaLog),
            LikelihoodFamily::RoystonParmar => {
                Err("RoystonParmar is survival-specific and not a GLM likelihood")
            }
        }
    }
}

impl From<GlmLikelihoodFamily> for LikelihoodFamily {
    fn from(value: GlmLikelihoodFamily) -> Self {
        match value {
            GlmLikelihoodFamily::GaussianIdentity => Self::GaussianIdentity,
            GlmLikelihoodFamily::BinomialLogit => Self::BinomialLogit,
            GlmLikelihoodFamily::BinomialProbit => Self::BinomialProbit,
            GlmLikelihoodFamily::BinomialCLogLog => Self::BinomialCLogLog,
            GlmLikelihoodFamily::BinomialSas => Self::BinomialSas,
            GlmLikelihoodFamily::BinomialBetaLogistic => Self::BinomialBetaLogistic,
            GlmLikelihoodFamily::BinomialMixture => Self::BinomialMixture,
            GlmLikelihoodFamily::PoissonLog => Self::PoissonLog,
            GlmLikelihoodFamily::GammaLog => Self::GammaLog,
        }
    }
}

/// How ridge-adjusted determinants should be evaluated for outer criteria.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RidgeDeterminantMode {
    /// Use exact full logdet.
    Auto,
    /// Use full log-determinant of the ridged matrix (requires SPD in practice).
    Full,
    /// Use positive-part pseudo-determinant (sum log ev for ev > floor).
    PositivePart,
}

/// Storage form of the ridge penalty matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RidgeMatrixForm {
    /// Ridge matrix is `delta * I`.
    ScaledIdentity,
}

/// Global policy governing how a stabilization ridge participates in objectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RidgePolicy {
    /// Must remain independent of smoothing parameters (`rho`) for smooth outer derivatives.
    pub rho_independent: bool,
    /// Include ridge in quadratic penalty term: `0.5 * delta * ||beta||^2`.
    pub include_quadratic_penalty: bool,
    /// Include ridge in penalty determinant term (e.g. `log|S_lambda + delta I|`).
    pub include_penalty_logdet: bool,
    /// Include ridge in Hessian used by Laplace term / implicit differentiation.
    pub include_laplacehessian: bool,
    /// Determinant evaluation mode when ridge participates in logdet terms.
    pub determinant_mode: RidgeDeterminantMode,
}

impl RidgePolicy {
    /// Default policy used by PIRLS/REML path:
    /// treat stabilization ridge as an explicit `delta I` prior contribution
    /// with adaptive logdet evaluation.
    pub fn explicit_stabilization_full() -> Self {
        Self {
            rho_independent: true,
            include_quadratic_penalty: true,
            include_penalty_logdet: true,
            include_laplacehessian: true,
            determinant_mode: RidgeDeterminantMode::Auto,
        }
    }

    pub fn explicit_stabilization_full_exact() -> Self {
        Self {
            determinant_mode: RidgeDeterminantMode::Full,
            ..Self::explicit_stabilization_full()
        }
    }

    /// Variant used when pseudo-determinants are required for indefinite matrices.
    pub fn explicit_stabilization_pospart() -> Self {
        Self {
            determinant_mode: RidgeDeterminantMode::PositivePart,
            ..Self::explicit_stabilization_full()
        }
    }
}

/// Concrete ridge metadata stamped into a fitted PIRLS result.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RidgePassport {
    /// Stabilization magnitude for matrix form `delta * I`.
    pub delta: f64,
    pub matrix_form: RidgeMatrixForm,
    pub policy: RidgePolicy,
}

impl RidgePassport {
    pub fn scaled_identity(delta: f64, policy: RidgePolicy) -> Self {
        Self {
            delta,
            matrix_form: RidgeMatrixForm::ScaledIdentity,
            policy,
        }
    }

    #[inline]
    pub fn penalty_logdet_ridge(self) -> f64 {
        if self.policy.include_penalty_logdet {
            self.delta
        } else {
            0.0
        }
    }

    #[inline]
    pub fn laplacehessianridge(self) -> f64 {
        if self.policy.include_laplacehessian {
            self.delta
        } else {
            0.0
        }
    }
}

// ============================================================================
// StabilizationLedger: canonical accounting for every fixed/heuristic ridge
// added anywhere in the solver, linear-algebra, or family code paths.
//
// Three semantically distinct ridge uses must NEVER be conflated:
//   1. SolverDampingOnly      — Levenberg/trust-region damping; never enters
//                               objective, gradient, logdet, Hessian, or any
//                               saved/serialized model artifact.
//   2. NumericalPerturbation  — added strictly so a linear solve is well-
//                               posed (e.g. Cholesky of a near-singular
//                               matrix). Carries an optional backward-error
//                               bound. Does NOT change the objective.
//   3. ExplicitPrior          — model-level `delta * I` (or block-diagonal)
//                               prior. Appears in quadratic, log normalizer,
//                               Laplace Hessian, serialization, diagnostics.
//
// `RidgePassport` above already encodes the inclusion-flag matrix for the
// PIRLS Laplace ridge specifically; this ledger is the broader sibling that
// every other call site (RidgePlanner, matrix_inverse_with_regularization,
// LAML rho-Hessian inversion, survival stabilization, custom-family
// `ridge_floor`) routes through, so a downstream consumer can ask
// `ledger.included_in_quadratic()` rather than rediscovering the policy.
// ============================================================================

/// Inertia of a symmetric matrix (count of positive / zero / negative
/// eigenvalues). Used by `bump_with_matrix` and other indefinite-aware
/// stabilization rules to drive δ from spectral evidence rather than a
/// condition-number heuristic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Inertia {
    pub positive: usize,
    pub zero: usize,
    pub negative: usize,
}

/// Why a stabilization δ was chosen at this site.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StabilizationRule {
    /// δ is a hard-coded constant in the source.
    FixedConstant,
    /// δ chosen so the SPD floor τ is met: δ = max(0, τ - λ_min(H)).
    InertiaTarget { spd_floor: f64 },
    /// δ chosen via a condition-number / sqrt-ratio heuristic.
    Heuristic,
    /// User- or family-specified prior precision.
    UserSpecified,
    /// δ derived from a back-off escalation after a factorization failure.
    BackoffEscalation { attempts: usize },
}

/// Three semantically distinct flavours a ridge δ can have.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StabilizationKind {
    None,
    /// LM/TR damping. NEVER enters the objective, gradient, logdet, Hessian,
    /// or any saved model artifact. Lives only inside the trust-region step.
    SolverDampingOnly,
    /// Added strictly so a linear solve succeeds. The objective/Hessian the
    /// caller sees is unchanged; the perturbation is a property of the
    /// solver, not the model. `backward_error_bound` is the max change to
    /// the solution norm imputable to the perturbation, when known.
    NumericalPerturbation {
        backward_error_bound: Option<f64>,
    },
    /// Part of the model. Enters quadratic, log normalizer, Hessian,
    /// serialization, and user-visible summaries.
    ExplicitPrior,
}

/// Canonical record of a single stabilization δ applied at a single site.
///
/// Construct via the helper constructors (`solver_damping`,
/// `numerical_perturbation`, `explicit_prior`) so the `included_in_*`
/// invariants are guaranteed to match `kind`. Direct field construction is
/// public for serialization round-trips only.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StabilizationLedger {
    pub kind: StabilizationKind,
    pub delta: f64,
    pub matrix_form: RidgeMatrixForm,
    pub chosen_by: StabilizationRule,
    pub inertia_before: Option<Inertia>,
    pub inertia_after: Option<Inertia>,
    /// True iff δ contributes ½ δ ‖β‖² to the objective.
    pub included_in_quadratic: bool,
    /// True iff δ contributes δ I to the Laplace Hessian used for
    /// covariance / smoothing-parameter inference.
    pub included_in_laplace_hessian: bool,
    /// True iff δ contributes log|S + δ I| to the penalty log-determinant.
    pub included_in_penalty_logdet: bool,
}

impl StabilizationLedger {
    /// "No stabilization applied at this site" sentinel.
    pub fn none() -> Self {
        Self {
            kind: StabilizationKind::None,
            delta: 0.0,
            matrix_form: RidgeMatrixForm::ScaledIdentity,
            chosen_by: StabilizationRule::FixedConstant,
            inertia_before: None,
            inertia_after: None,
            included_in_quadratic: false,
            included_in_laplace_hessian: false,
            included_in_penalty_logdet: false,
        }
    }

    /// LM/TR damping. δ is invisible to the objective, gradient, and any
    /// saved artifact. Asserting this invariant at every read site is the
    /// whole reason the ledger exists.
    pub fn solver_damping(delta: f64, chosen_by: StabilizationRule) -> Self {
        Self {
            kind: StabilizationKind::SolverDampingOnly,
            delta,
            matrix_form: RidgeMatrixForm::ScaledIdentity,
            chosen_by,
            inertia_before: None,
            inertia_after: None,
            included_in_quadratic: false,
            included_in_laplace_hessian: false,
            included_in_penalty_logdet: false,
        }
    }

    /// Solver-only perturbation that leaves the objective unchanged. The
    /// caller may attach a backward-error bound when one is available
    /// (e.g. from iterative refinement / Wilkinson-style analysis).
    pub fn numerical_perturbation(
        delta: f64,
        chosen_by: StabilizationRule,
        backward_error_bound: Option<f64>,
    ) -> Self {
        Self {
            kind: StabilizationKind::NumericalPerturbation {
                backward_error_bound,
            },
            delta,
            matrix_form: RidgeMatrixForm::ScaledIdentity,
            chosen_by,
            inertia_before: None,
            inertia_after: None,
            included_in_quadratic: false,
            included_in_laplace_hessian: false,
            included_in_penalty_logdet: false,
        }
    }

    /// Model-level explicit prior. δ enters every accounting pass: the
    /// quadratic penalty, the Laplace Hessian, the penalty log-determinant,
    /// and serialization.
    pub fn explicit_prior(delta: f64, matrix_form: RidgeMatrixForm) -> Self {
        Self {
            kind: StabilizationKind::ExplicitPrior,
            delta,
            matrix_form,
            chosen_by: StabilizationRule::UserSpecified,
            inertia_before: None,
            inertia_after: None,
            included_in_quadratic: true,
            included_in_laplace_hessian: true,
            included_in_penalty_logdet: true,
        }
    }

    /// Bridge from the existing `RidgePassport` so PIRLS-side code (which
    /// already passes a `RidgePassport` through every call) can hand a
    /// ledger to anything that wants the new uniform view.
    pub fn from_passport(passport: RidgePassport) -> Self {
        let any_included = passport.policy.include_quadratic_penalty
            || passport.policy.include_laplacehessian
            || passport.policy.include_penalty_logdet;
        let kind = if !any_included {
            // A `RidgePassport` whose policy excludes every accounting term
            // is morally a numerical perturbation: the ridge is there to
            // make the solve work but the objective ignores it.
            StabilizationKind::NumericalPerturbation {
                backward_error_bound: None,
            }
        } else {
            StabilizationKind::ExplicitPrior
        };
        Self {
            kind,
            delta: passport.delta,
            matrix_form: passport.matrix_form,
            chosen_by: StabilizationRule::FixedConstant,
            inertia_before: None,
            inertia_after: None,
            included_in_quadratic: passport.policy.include_quadratic_penalty,
            included_in_laplace_hessian: passport.policy.include_laplacehessian,
            included_in_penalty_logdet: passport.policy.include_penalty_logdet,
        }
    }

    /// δ value to fold into the quadratic penalty term, or 0.0 if this
    /// ledger entry is not part of the model.
    #[inline]
    pub fn quadratic_delta(&self) -> f64 {
        if self.included_in_quadratic {
            self.delta
        } else {
            0.0
        }
    }

    /// δ value to add to the Laplace Hessian, or 0.0 if not included.
    #[inline]
    pub fn laplace_hessian_delta(&self) -> f64 {
        if self.included_in_laplace_hessian {
            self.delta
        } else {
            0.0
        }
    }

    /// δ value to add inside log|S + δ I|, or 0.0 if not included.
    #[inline]
    pub fn penalty_logdet_delta(&self) -> f64 {
        if self.included_in_penalty_logdet {
            self.delta
        } else {
            0.0
        }
    }

    /// Invariant check: kind must be consistent with the inclusion flags.
    /// Used by the ledger-invariants test in `tests/ridge_ledger_invariants.rs`.
    pub fn invariants_hold(&self) -> bool {
        match self.kind {
            StabilizationKind::None => {
                self.delta == 0.0
                    && !self.included_in_quadratic
                    && !self.included_in_laplace_hessian
                    && !self.included_in_penalty_logdet
            }
            StabilizationKind::SolverDampingOnly
            | StabilizationKind::NumericalPerturbation { .. } => {
                !self.included_in_quadratic
                    && !self.included_in_laplace_hessian
                    && !self.included_in_penalty_logdet
            }
            StabilizationKind::ExplicitPrior => {
                self.included_in_quadratic
                    && self.included_in_laplace_hessian
                    && self.included_in_penalty_logdet
            }
        }
    }
}
/// Generate a `#[repr(transparent)]` `Array1<f64>` newtype with the
/// `new`/`Deref`/`DerefMut`/`AsRef`/`From` boilerplate every wrapper in this
/// module needs. Keeping the three semantic types behind one macro both
/// removes ~100 lines of duplication and guarantees they cannot drift apart.
macro_rules! array1_f64_newtype {
    ($name:ident $(, $extra:ident)*) => {
        #[repr(transparent)]
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name(pub Array1<f64>);

        impl $name {
            #[inline]
            pub fn new(values: Array1<f64>) -> Self {
                Self(values)
            }

            #[inline]
            pub fn zeros(len: usize) -> Self {
                Self(Array1::zeros(len))
            }
        }

        impl Deref for $name {
            type Target = Array1<f64>;
            #[inline]
            fn deref(&self) -> &Self::Target { &self.0 }
        }

        impl DerefMut for $name {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
        }

        impl AsRef<Array1<f64>> for $name {
            #[inline]
            fn as_ref(&self) -> &Array1<f64> { &self.0 }
        }

        impl From<Array1<f64>> for $name {
            #[inline]
            fn from(values: Array1<f64>) -> Self { Self(values) }
        }

        impl From<$name> for Array1<f64> {
            #[inline]
            fn from(values: $name) -> Self { values.0 }
        }

        $( array1_f64_newtype!(@extra $name $extra); )*
    };
    (@extra $name:ident exp) => {
        impl $name {
            #[inline]
            pub fn exp(&self) -> Array1<f64> { self.0.mapv(f64::exp) }
        }
    };
}

array1_f64_newtype!(Coefficients);
array1_f64_newtype!(LinearPredictor);
array1_f64_newtype!(LogSmoothingParams, exp);

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct LogSmoothingParamsView<'a>(pub ArrayView1<'a, f64>);

impl<'a> LogSmoothingParamsView<'a> {
    pub fn new(values: ArrayView1<'a, f64>) -> Self {
        Self(values)
    }

    pub fn exp(&self) -> Array1<f64> {
        self.0.mapv(f64::exp)
    }
}

impl<'a> Deref for LogSmoothingParamsView<'a> {
    type Target = ArrayView1<'a, f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

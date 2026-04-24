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

/// Supported inverse-link components for convex blended inverse links.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinkComponent {
    Probit,
    Logit,
    CLogLog,
    LogLog,
    Cauchit,
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
    Normal { mean: f64, sd: f64 },
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

    #[inline]
    pub(crate) fn is_binomial(self) -> bool {
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
        let scale = match family {
            GlmLikelihoodFamily::GaussianIdentity => LikelihoodScaleMetadata::ProfiledGaussian,
            GlmLikelihoodFamily::GammaLog => {
                LikelihoodScaleMetadata::EstimatedGammaShape { shape: 1.0 }
            }
            GlmLikelihoodFamily::BinomialLogit
            | GlmLikelihoodFamily::BinomialProbit
            | GlmLikelihoodFamily::BinomialCLogLog
            | GlmLikelihoodFamily::BinomialSas
            | GlmLikelihoodFamily::BinomialBetaLogistic
            | GlmLikelihoodFamily::BinomialMixture
            | GlmLikelihoodFamily::PoissonLog => {
                LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 }
            }
        };
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
    /// Use exact full logdet for smaller systems and SLQ above a size threshold.
    Auto,
    /// Use full log-determinant of the ridged matrix (requires SPD in practice).
    Full,
    /// Use stochastic Lanczos quadrature on the ridged SPD surface.
    StochasticLanczos,
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

    pub fn explicit_stabilization_full_slq() -> Self {
        Self {
            determinant_mode: RidgeDeterminantMode::StochasticLanczos,
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
#[repr(transparent)]
#[derive(Clone, Debug, PartialEq)]
pub struct Coefficients(pub Array1<f64>);

impl Coefficients {
    pub fn new(values: Array1<f64>) -> Self {
        Self(values)
    }

    pub fn zeros(len: usize) -> Self {
        Self(Array1::zeros(len))
    }
}

impl Deref for Coefficients {
    type Target = Array1<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Coefficients {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AsRef<Array1<f64>> for Coefficients {
    fn as_ref(&self) -> &Array1<f64> {
        &self.0
    }
}

impl From<Array1<f64>> for Coefficients {
    fn from(values: Array1<f64>) -> Self {
        Self(values)
    }
}

impl From<Coefficients> for Array1<f64> {
    fn from(values: Coefficients) -> Self {
        values.0
    }
}

#[repr(transparent)]
#[derive(Clone, Debug, PartialEq)]
pub struct LinearPredictor(pub Array1<f64>);

impl LinearPredictor {
    pub fn new(values: Array1<f64>) -> Self {
        Self(values)
    }

    pub fn zeros(len: usize) -> Self {
        Self(Array1::zeros(len))
    }
}

impl Deref for LinearPredictor {
    type Target = Array1<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LinearPredictor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AsRef<Array1<f64>> for LinearPredictor {
    fn as_ref(&self) -> &Array1<f64> {
        &self.0
    }
}

impl From<Array1<f64>> for LinearPredictor {
    fn from(values: Array1<f64>) -> Self {
        Self(values)
    }
}

impl From<LinearPredictor> for Array1<f64> {
    fn from(values: LinearPredictor) -> Self {
        values.0
    }
}

#[repr(transparent)]
#[derive(Clone, Debug, PartialEq)]
pub struct LogSmoothingParams(pub Array1<f64>);

impl LogSmoothingParams {
    pub fn new(values: Array1<f64>) -> Self {
        Self(values)
    }
}

impl Deref for LogSmoothingParams {
    type Target = Array1<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LogSmoothingParams {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Array1<f64>> for LogSmoothingParams {
    fn from(values: Array1<f64>) -> Self {
        Self(values)
    }
}

impl From<LogSmoothingParams> for Array1<f64> {
    fn from(values: LogSmoothingParams) -> Self {
        values.0
    }
}

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

use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

pub use crate::hull::PeeledHull;

/// Shared engine-level link selector for generalized models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinkFunction {
    Logit,
    Probit,
    CLogLog,
    Sas,
    BetaLogistic,
    Identity,
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

/// Parameterized inverse-link selector used where mu/derivatives are evaluated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InverseLink {
    Standard(LinkFunction),
    Sas(SasLinkState),
    BetaLogistic(SasLinkState),
    Mixture(MixtureLinkState),
}

impl InverseLink {
    #[inline]
    pub fn link_function(&self) -> LinkFunction {
        match self {
            Self::Standard(link) => *link,
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
}

/// Engine-level likelihood selector used by generic APIs.
/// Some families remain restricted to domain-specific entrypoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LikelihoodFamily {
    GaussianIdentity,
    BinomialLogit,
    BinomialProbit,
    BinomialCLogLog,
    BinomialSas,
    BinomialBetaLogistic,
    BinomialMixture,
    RoystonParmar,
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
}

impl TryFrom<LikelihoodFamily> for GlmLikelihoodFamily {
    type Error = &'static str;

    fn try_from(value: LikelihoodFamily) -> Result<Self, Self::Error> {
        match value {
            LikelihoodFamily::GaussianIdentity => Ok(Self::GaussianIdentity),
            LikelihoodFamily::BinomialLogit => Ok(Self::BinomialLogit),
            LikelihoodFamily::BinomialProbit => Ok(Self::BinomialProbit),
            LikelihoodFamily::BinomialCLogLog => Ok(Self::BinomialCLogLog),
            LikelihoodFamily::BinomialSas => Ok(Self::BinomialSas),
            LikelihoodFamily::BinomialBetaLogistic => Ok(Self::BinomialBetaLogistic),
            LikelihoodFamily::BinomialMixture => Ok(Self::BinomialMixture),
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
    pub fn quadratic_penalty_ridge(self) -> f64 {
        if self.policy.include_quadratic_penalty {
            self.delta
        } else {
            0.0
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointLinkModel {
    pub knot_range: (f64, f64),
    pub knot_vector: Array1<f64>,
    pub link_transform: ndarray::Array2<f64>,
    pub beta_link: Array1<f64>,
    pub degree: usize,
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

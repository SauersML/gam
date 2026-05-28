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

/// Shared engine-level link selector for generalized models. This is the
/// "wide" link descriptor: CLI parsing, formula DSL, and the projection from
/// `InverseLink::link_function()` all live in this enum, so it carries every
/// link kind the engine knows about — including the state-bearing
/// `Sas` / `BetaLogistic` cases.
///
/// `LinkFunction` is *not* the right type for the state-less `InverseLink::Standard`
/// cell. Use [`StandardLink`] there: the type system then refuses to construct
/// a state-less `Standard(Sas)` / `Standard(BetaLogistic)` placeholder.
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
    pub const fn name(self) -> &'static str {
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

/// Legal-only link descriptor for the state-less `InverseLink::Standard` cell.
///
/// `Sas` / `BetaLogistic` are state-bearing and live in their own
/// `InverseLink::Sas(_)` / `InverseLink::BetaLogistic(_)` variants. The type
/// system enforces that fact by omitting them here, so the historical
/// "state-less placeholder" pattern (`InverseLink::Standard(LinkFunction::Sas)`)
/// no longer compiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StandardLink {
    Logit,
    Probit,
    CLogLog,
    Identity,
    Log,
}

impl StandardLink {
    #[inline]
    pub const fn name(self) -> &'static str {
        self.as_link_function().name()
    }

    #[inline]
    pub const fn as_link_function(self) -> LinkFunction {
        match self {
            Self::Logit => LinkFunction::Logit,
            Self::Probit => LinkFunction::Probit,
            Self::CLogLog => LinkFunction::CLogLog,
            Self::Identity => LinkFunction::Identity,
            Self::Log => LinkFunction::Log,
        }
    }
}

impl From<StandardLink> for LinkFunction {
    #[inline]
    fn from(link: StandardLink) -> Self {
        link.as_link_function()
    }
}

/// Error returned when narrowing a wide [`LinkFunction`] into a [`StandardLink`].
/// `Sas` and `BetaLogistic` are state-bearing and have no legal `Standard(_)`
/// representation; they must be routed through `InverseLink::Sas` /
/// `InverseLink::BetaLogistic`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateBearingLinkInStandardSlot(pub LinkFunction);

impl std::fmt::Display for StateBearingLinkInStandardSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "state-bearing link `{}` cannot be carried by `InverseLink::Standard`; \
             route through `InverseLink::Sas` / `InverseLink::BetaLogistic`",
            self.0.name()
        )
    }
}

impl std::error::Error for StateBearingLinkInStandardSlot {}

impl TryFrom<LinkFunction> for StandardLink {
    type Error = StateBearingLinkInStandardSlot;

    #[inline]
    fn try_from(link: LinkFunction) -> Result<Self, Self::Error> {
        match link {
            LinkFunction::Logit => Ok(Self::Logit),
            LinkFunction::Probit => Ok(Self::Probit),
            LinkFunction::CLogLog => Ok(Self::CLogLog),
            LinkFunction::Identity => Ok(Self::Identity),
            LinkFunction::Log => Ok(Self::Log),
            LinkFunction::Sas | LinkFunction::BetaLogistic => {
                Err(StateBearingLinkInStandardSlot(link))
            }
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
    pub const fn name(self) -> &'static str {
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
    Standard(StandardLink),
    LatentCLogLog(LatentCLogLogState),
    Sas(SasLinkState),
    BetaLogistic(SasLinkState),
    Mixture(MixtureLinkState),
}

impl InverseLink {
    #[inline]
    pub const fn link_function(&self) -> LinkFunction {
        match self {
            Self::Standard(link) => link.as_link_function(),
            Self::LatentCLogLog(_) => LinkFunction::CLogLog,
            Self::Sas(_) => LinkFunction::Sas,
            Self::BetaLogistic(_) => LinkFunction::BetaLogistic,
            Self::Mixture(_) => LinkFunction::Logit,
        }
    }

    #[inline]
    pub const fn mixture_state(&self) -> Option<&MixtureLinkState> {
        match self {
            Self::Mixture(state) => Some(state),
            _ => None,
        }
    }

    #[inline]
    pub const fn sas_state(&self) -> Option<&SasLinkState> {
        match self {
            Self::Sas(state) | Self::BetaLogistic(state) => Some(state),
            _ => None,
        }
    }

    #[inline]
    pub const fn latent_cloglog_state(&self) -> Option<&LatentCLogLogState> {
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

// ---------------------------------------------------------------------------
// Unified likelihood specification
// ---------------------------------------------------------------------------
//
// `LikelihoodSpec { response: ResponseFamily, link: InverseLink }` is the
// canonical likelihood selector. `ResponseFamily` is a pure response-
// distribution selector that carries the per-family scalars
// (`Tweedie { p }`, `NegativeBinomial { theta }`, `Beta { phi }`); `InverseLink`
// is the parameterized inverse-link selector. Splitting (response, link)
// removes the drift bug that the former flat likelihood enum allowed
// when its variant disagreed with a separately-stored `InverseLink`.

/// Pure response distribution selector — no link information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResponseFamily {
    Gaussian,
    Binomial,
    Poisson,
    Tweedie { p: f64 },
    NegativeBinomial { theta: f64 },
    Beta { phi: f64 },
    Gamma,
    RoystonParmar,
}

impl ResponseFamily {
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::Binomial => "binomial",
            Self::Poisson => "poisson",
            Self::Tweedie { .. } => "tweedie",
            Self::NegativeBinomial { .. } => "negative-binomial",
            Self::Beta { .. } => "beta",
            Self::Gamma => "gamma",
            Self::RoystonParmar => "royston-parmar",
        }
    }

    /// Closed-interval bounds for the mean (response-scale) of this family.
    ///
    /// Used by predict-side CI clamps that need to keep transformed bounds
    /// within the support of the response. Beta uses strict-open `(1e-10, 1 − 1e-10)`
    /// to avoid logit singularities; Binomial / Royston-Parmar use the closed
    /// `[0, 1]` since they are evaluated post-transformation. Unbounded
    /// (continuous-real or non-negative-real) families return `None` — the
    /// caller should not clamp.
    #[inline]
    pub fn mean_clamp_bounds(&self) -> Option<(f64, f64)> {
        match self {
            Self::Binomial | Self::RoystonParmar => Some((0.0, 1.0)),
            Self::Beta { .. } => Some((1e-10, 1.0 - 1e-10)),
            Self::Gaussian
            | Self::Poisson
            | Self::Tweedie { .. }
            | Self::NegativeBinomial { .. }
            | Self::Gamma => None,
        }
    }

    /// Per-family textual description of the response-support requirement.
    /// `None` means the family is supported on the entire real line at the
    /// validation layer (Gaussian) or has its support enforced by a downstream
    /// pathway (Binomial via `is_binary_response` and the PIRLS row kernel;
    /// RoystonParmar via the survival pipeline).
    #[inline]
    pub fn response_support_requirement(&self) -> Option<&'static str> {
        match self {
            Self::Gamma => Some("strictly positive response values (y > 0)"),
            Self::Poisson | Self::NegativeBinomial { .. } | Self::Tweedie { .. } => {
                Some("non-negative response values (y ≥ 0)")
            }
            Self::Beta { .. } => Some("response values strictly in the open interval (0, 1)"),
            Self::Gaussian | Self::Binomial | Self::RoystonParmar => None,
        }
    }

    /// Predicate that returns `true` iff `yi` lies in this family's response
    /// support. Only meaningful for families with a non-trivial domain
    /// constraint at the validation layer; `validate_response_support` calls
    /// this only after `response_support_requirement` returns `Some`, so the
    /// "unconstrained" families (Gaussian / Binomial / RoystonParmar) never
    /// hit this code path.
    #[inline]
    fn response_support_contains(&self, yi: f64) -> bool {
        match self {
            Self::Gamma => yi.is_finite() && yi > 0.0,
            Self::Poisson | Self::NegativeBinomial { .. } | Self::Tweedie { .. } => {
                yi.is_finite() && yi >= 0.0
            }
            Self::Beta { .. } => yi.is_finite() && yi > 0.0 && yi < 1.0,
            Self::Gaussian | Self::Binomial | Self::RoystonParmar => true,
        }
    }

    /// Human-readable family label used in domain-violation error messages
    /// (capitalised to match user-facing prose, distinct from `name()` which
    /// returns the lowercase canonical identifier).
    #[inline]
    fn response_support_label(&self) -> &'static str {
        match self {
            Self::Gaussian => "Gaussian",
            Self::Binomial => "Binomial",
            Self::Poisson => "Poisson",
            Self::Tweedie { .. } => "Tweedie",
            Self::NegativeBinomial { .. } => "Negative-Binomial",
            Self::Beta { .. } => "Beta",
            Self::Gamma => "Gamma",
            Self::RoystonParmar => "Royston-Parmar",
        }
    }

    /// Validate that every element of `y` lies in this family's response
    /// support. The check is the upfront, fit-blocking enforcement of the
    /// family's distributional support — e.g. Gamma rejects `y ≤ 0` because
    /// the log-likelihood contains `log(y)`, Poisson rejects `y < 0` because
    /// the log-mass contains `log(y!)`.
    ///
    /// Returns `Ok(())` for families whose support is the entire real line at
    /// this layer (Gaussian) or whose support is enforced by a downstream
    /// pathway (Binomial via `is_binary_response` + PIRLS, RoystonParmar via
    /// the survival pipeline).
    ///
    /// Up to `ResponseSupportViolation::MAX_REPORTED` offending row indices
    /// are returned in the violation so the message stays bounded on large
    /// datasets while still identifying offending rows.
    pub fn validate_response_support(
        &self,
        y: ArrayView1<'_, f64>,
    ) -> Result<(), ResponseSupportViolation> {
        let requirement = match self.response_support_requirement() {
            Some(r) => r,
            None => return Ok(()),
        };
        let mut offending: Vec<(usize, f64)> = Vec::new();
        let mut total_violations: usize = 0;
        for (i, &yi) in y.iter().enumerate() {
            if !self.response_support_contains(yi) {
                total_violations += 1;
                if offending.len() < ResponseSupportViolation::MAX_REPORTED {
                    offending.push((i, yi));
                }
            }
        }
        if total_violations == 0 {
            Ok(())
        } else {
            Err(ResponseSupportViolation {
                family_label: self.response_support_label(),
                requirement,
                offending,
                total_violations,
            })
        }
    }

    /// Auto-infer a likelihood family when the user did not specify one.
    ///
    /// Policy:
    ///   * A string-valued (`Categorical`) response column is refused —
    ///     numeric-encoded level indices (e.g. `"yes"`/`"no"` → `0.0`/`1.0`)
    ///     would otherwise be silently interpreted as a binary outcome,
    ///     producing a probability model the user never asked for.
    ///   * A strictly-binary numeric response (`Binary` kind, or `Numeric`
    ///     with only `{0, 1}` values) maps to `Binomial`.
    ///   * Anything else maps to `Gaussian`.
    ///
    /// The fallback to `is_binary_response` inside the `Numeric` arm is what
    /// historically lived directly inside `resolve_family`; centralising the
    /// policy here means every entry point (formula API, CLI, future bindings)
    /// gets the same default-inference behaviour.
    pub fn infer_from_response(
        y: ArrayView1<'_, f64>,
        y_kind: ResponseColumnKind,
    ) -> Result<Self, ResponseInferenceRefusal> {
        match y_kind {
            ResponseColumnKind::Categorical { levels } => Err(ResponseInferenceRefusal {
                reason: ResponseInferenceRefusalReason::NonNumericResponse,
                levels,
            }),
            ResponseColumnKind::Binary => Ok(Self::Binomial),
            ResponseColumnKind::Numeric => {
                let binary = !y.is_empty()
                    && y.iter().all(|v| {
                        v.is_finite() && ((*v - 0.0).abs() < 1e-12 || (*v - 1.0).abs() < 1e-12)
                    });
                if binary {
                    Ok(Self::Binomial)
                } else {
                    Ok(Self::Gaussian)
                }
            }
        }
    }
}

/// Domain-violation detail produced by [`ResponseFamily::validate_response_support`].
///
/// Owns its own `Display` impl so call sites in the workflow, the CLI, and the
/// external-design GLM path produce identical user-facing prose. The
/// `total_violations` counter is kept distinct from `offending.len()` so the
/// message can honestly say `(N total)` even when only the first
/// `MAX_REPORTED` indices are surfaced.
#[derive(Debug, Clone)]
pub struct ResponseSupportViolation {
    pub family_label: &'static str,
    pub requirement: &'static str,
    pub offending: Vec<(usize, f64)>,
    pub total_violations: usize,
}

impl ResponseSupportViolation {
    /// Maximum number of offending row indices reported in the error message.
    /// Keeps the message bounded on biobank-scale data while still pointing
    /// the user at concrete bad rows to inspect.
    pub const MAX_REPORTED: usize = 5;

    /// Format the violation against a specific response column name. The
    /// column name is supplied by the caller because [`ResponseFamily`] does
    /// not know which column the user pointed at.
    pub fn message_for(&self, response_name: &str) -> String {
        let shown = self
            .offending
            .iter()
            .map(|(i, v)| format!("y[{i}]={v}"))
            .collect::<Vec<_>>()
            .join(", ");
        let more = if self.total_violations > self.offending.len() {
            format!(", ... ({} total)", self.total_violations)
        } else {
            String::new()
        };
        format!(
            "{family} family requires {req}; response column '{name}' violates this constraint at row(s) [{shown}{more}]",
            family = self.family_label,
            req = self.requirement,
            name = response_name,
        )
    }
}

impl std::fmt::Display for ResponseSupportViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message_for("y"))
    }
}

impl std::error::Error for ResponseSupportViolation {}

/// Caller-supplied description of the response column's *source* kind.
///
/// `Categorical { levels }` flags a column that arrived as non-numeric strings
/// (the ingest layer encoded its levels to `0.0, 1.0, ...` indices) — the
/// `levels` list is preserved so the auto-inference refusal can echo them
/// back to the user verbatim. `Binary` is the ingest-layer signal that a
/// numeric column already contains only `{0, 1}` (used to short-circuit the
/// scan inside [`ResponseFamily::infer_from_response`]). `Numeric` is the
/// generic continuous case.
#[derive(Debug, Clone)]
pub enum ResponseColumnKind {
    Numeric,
    Binary,
    Categorical { levels: Vec<String> },
}

/// Reason [`ResponseFamily::infer_from_response`] refused to pick a default
/// family. Kept as an enum so future policy extensions (e.g. "refuse on
/// constant response" — currently a separate CLI-side check) can be added
/// without breaking the call site's match arms.
#[derive(Debug, Clone)]
pub enum ResponseInferenceRefusalReason {
    NonNumericResponse,
}

/// Auto-inference refusal carrying the levels seen in the source column so
/// the workflow error can echo them in its message.
#[derive(Debug, Clone)]
pub struct ResponseInferenceRefusal {
    pub reason: ResponseInferenceRefusalReason,
    pub levels: Vec<String>,
}

impl ResponseInferenceRefusal {
    /// Format the refusal against a specific response column name.
    pub fn message_for(&self, response_name: &str) -> String {
        match self.reason {
            ResponseInferenceRefusalReason::NonNumericResponse => {
                let n = self.levels.len().min(5);
                let head = self
                    .levels
                    .iter()
                    .take(n)
                    .map(|s| format!("'{s}'"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let preview = if self.levels.len() > n {
                    format!("[{head}, ...]")
                } else {
                    format!("[{head}]")
                };
                format!(
                    "response column '{name}' contains non-numeric values {preview}. \
                     Did you mean to use family='binomial' for a binary outcome, \
                     or does '{name}' contain categorical labels that should be encoded first?",
                    name = response_name,
                    preview = preview,
                )
            }
        }
    }
}

impl std::fmt::Display for ResponseInferenceRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message_for("y"))
    }
}

impl std::error::Error for ResponseInferenceRefusal {}

/// Unified likelihood specification: response distribution + parameterized link.
///
/// `ResponseFamily` carries the per-family scalars (Tweedie p, NegBin theta,
/// Beta phi); `InverseLink` carries the parameterized link state. Together
/// they replace the former flat likelihood enum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LikelihoodSpec {
    pub response: ResponseFamily,
    pub link: InverseLink,
}

/// Legal-only enumeration of the `(ResponseFamily, InverseLink)` cells the
/// engine recognises. `LikelihoodSpec` is the product type with ~40 nominal
/// cells (8 response variants × 5 inverse-link variants), but only the cells
/// listed here are honoured by the family math; the rest are silently masked
/// by fallback arms. `FamilySpecKind` is the canonical projection used by
/// naming, predicates, and dispatch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FamilySpecKind {
    GaussianIdentity,
    PoissonLog,
    GammaLog,
    TweedieLog { p: f64 },
    NegativeBinomialLog { theta: f64 },
    BetaLogit { phi: f64 },
    RoystonParmar,
    BinomialLogit,
    BinomialProbit,
    BinomialCLogLog,
    BinomialLatentCLogLog(LatentCLogLogState),
    BinomialSas(SasLinkState),
    BinomialBetaLogistic(SasLinkState),
    BinomialMixture(MixtureLinkState),
}

impl FamilySpecKind {
    /// Short identifier matching the legacy `LikelihoodSpec::name()` strings.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::GaussianIdentity => "gaussian",
            Self::PoissonLog => "poisson-log",
            Self::TweedieLog { .. } => "tweedie-log",
            Self::NegativeBinomialLog { .. } => "negative-binomial-log",
            Self::BetaLogit { .. } => "beta-regression-logit",
            Self::GammaLog => "gamma-log",
            Self::RoystonParmar => "royston-parmar",
            Self::BinomialLogit => "binomial-logit",
            Self::BinomialProbit => "binomial-probit",
            Self::BinomialCLogLog => "binomial-cloglog",
            Self::BinomialLatentCLogLog(_) => "latent-cloglog-binomial",
            Self::BinomialSas(_) => "binomial-sas",
            Self::BinomialBetaLogistic(_) => "binomial-beta-logistic",
            Self::BinomialMixture(_) => "binomial-blended-inverse-link",
        }
    }

    /// Human-readable label matching the legacy `LikelihoodSpec::pretty_name()` strings.
    #[inline]
    pub const fn pretty_name(&self) -> &'static str {
        match self {
            Self::GaussianIdentity => "Gaussian Identity",
            Self::PoissonLog => "Poisson Log",
            Self::TweedieLog { .. } => "Tweedie Log",
            Self::NegativeBinomialLog { .. } => "Negative-Binomial Log",
            Self::BetaLogit { .. } => "Beta Regression Logit",
            Self::GammaLog => "Gamma Log",
            Self::RoystonParmar => "Royston Parmar",
            Self::BinomialLogit => "Binomial Logit",
            Self::BinomialProbit => "Binomial Probit",
            Self::BinomialCLogLog => "Binomial CLogLog",
            Self::BinomialLatentCLogLog(_) => "Latent CLogLog Binomial",
            Self::BinomialSas(_) => "Binomial SAS",
            Self::BinomialBetaLogistic(_) => "Binomial Beta-Logistic",
            Self::BinomialMixture(_) => "Binomial Blended Inverse-Link",
        }
    }

    #[inline]
    pub const fn is_binomial(&self) -> bool {
        matches!(
            self,
            Self::BinomialLogit
                | Self::BinomialProbit
                | Self::BinomialCLogLog
                | Self::BinomialLatentCLogLog(_)
                | Self::BinomialSas(_)
                | Self::BinomialBetaLogistic(_)
                | Self::BinomialMixture(_)
        )
    }

    #[inline]
    pub const fn is_gaussian_identity(&self) -> bool {
        matches!(self, Self::GaussianIdentity)
    }

    #[inline]
    pub const fn is_royston_parmar(&self) -> bool {
        matches!(self, Self::RoystonParmar)
    }

    #[inline]
    pub const fn is_latent_cloglog(&self) -> bool {
        matches!(self, Self::BinomialLatentCLogLog(_))
    }

    #[inline]
    pub const fn is_binomial_mixture(&self) -> bool {
        matches!(self, Self::BinomialMixture(_))
    }

    #[inline]
    pub const fn is_binomial_sas(&self) -> bool {
        matches!(self, Self::BinomialSas(_))
    }

    #[inline]
    pub const fn is_binomial_beta_logistic(&self) -> bool {
        matches!(self, Self::BinomialBetaLogistic(_))
    }

    #[inline]
    pub const fn supports_firth(&self) -> bool {
        matches!(self, Self::BinomialLogit)
    }
}

impl LikelihoodSpec {
    #[inline]
    pub const fn new(response: ResponseFamily, link: InverseLink) -> Self {
        Self { response, link }
    }

    #[inline]
    pub const fn gaussian_identity() -> Self {
        Self::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        )
    }

    #[inline]
    pub const fn binomial_logit() -> Self {
        Self::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        )
    }

    #[inline]
    pub const fn binomial_probit() -> Self {
        Self::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Probit),
        )
    }

    #[inline]
    pub const fn binomial_cloglog() -> Self {
        Self::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::CLogLog),
        )
    }

    #[inline]
    pub const fn binomial_latent_cloglog(state: LatentCLogLogState) -> Self {
        Self::new(ResponseFamily::Binomial, InverseLink::LatentCLogLog(state))
    }

    #[inline]
    pub const fn binomial_sas(state: SasLinkState) -> Self {
        Self::new(ResponseFamily::Binomial, InverseLink::Sas(state))
    }

    #[inline]
    pub const fn binomial_beta_logistic(state: SasLinkState) -> Self {
        Self::new(ResponseFamily::Binomial, InverseLink::BetaLogistic(state))
    }

    #[inline]
    pub fn binomial_mixture(state: MixtureLinkState) -> Self {
        Self::new(ResponseFamily::Binomial, InverseLink::Mixture(state))
    }

    #[inline]
    pub const fn poisson_log() -> Self {
        Self::new(
            ResponseFamily::Poisson,
            InverseLink::Standard(StandardLink::Log),
        )
    }

    #[inline]
    pub const fn tweedie_log(p: f64) -> Self {
        Self::new(
            ResponseFamily::Tweedie { p },
            InverseLink::Standard(StandardLink::Log),
        )
    }

    #[inline]
    pub const fn negative_binomial_log(theta: f64) -> Self {
        Self::new(
            ResponseFamily::NegativeBinomial { theta },
            InverseLink::Standard(StandardLink::Log),
        )
    }

    #[inline]
    pub const fn beta_logit(phi: f64) -> Self {
        Self::new(
            ResponseFamily::Beta { phi },
            InverseLink::Standard(StandardLink::Logit),
        )
    }

    #[inline]
    pub const fn gamma_log() -> Self {
        Self::new(
            ResponseFamily::Gamma,
            InverseLink::Standard(StandardLink::Log),
        )
    }

    #[inline]
    pub const fn royston_parmar() -> Self {
        Self::new(
            ResponseFamily::RoystonParmar,
            InverseLink::Standard(StandardLink::Identity),
        )
    }

    #[inline]
    pub const fn link_function(&self) -> LinkFunction {
        self.link.link_function()
    }

    /// Once-and-for-all classification into the legal-only `FamilySpecKind`.
    ///
    /// `(ResponseFamily, InverseLink)` is a 35-cell product (7 response × 5
    /// inverse-link); only the cells listed here are recognised by the family
    /// math. With `InverseLink::Standard` carrying `StandardLink` (not
    /// `LinkFunction`), the historical "state-less Sas/BetaLogistic
    /// placeholder" cells are no longer representable, so the match is
    /// exhaustive over the legal cells. `Standard(Identity)` / `Standard(Log)`
    /// for the binomial family are structurally inert (no construction site
    /// reaches them) and are routed to `BinomialLogit` as the nearest legal
    /// classification.
    pub fn kind(&self) -> FamilySpecKind {
        match (&self.response, &self.link) {
            (ResponseFamily::Gaussian, _) => FamilySpecKind::GaussianIdentity,
            (ResponseFamily::Poisson, _) => FamilySpecKind::PoissonLog,
            (ResponseFamily::Tweedie { p }, _) => FamilySpecKind::TweedieLog { p: *p },
            (ResponseFamily::NegativeBinomial { theta }, _) => {
                FamilySpecKind::NegativeBinomialLog { theta: *theta }
            }
            (ResponseFamily::Beta { phi }, _) => FamilySpecKind::BetaLogit { phi: *phi },
            (ResponseFamily::Gamma, _) => FamilySpecKind::GammaLog,
            (ResponseFamily::RoystonParmar, _) => FamilySpecKind::RoystonParmar,
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Logit)) => {
                FamilySpecKind::BinomialLogit
            }
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Probit)) => {
                FamilySpecKind::BinomialProbit
            }
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::CLogLog)) => {
                FamilySpecKind::BinomialCLogLog
            }
            (
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Identity | StandardLink::Log),
            ) => FamilySpecKind::BinomialLogit,
            (ResponseFamily::Binomial, InverseLink::LatentCLogLog(state)) => {
                FamilySpecKind::BinomialLatentCLogLog(*state)
            }
            (ResponseFamily::Binomial, InverseLink::Sas(state)) => {
                FamilySpecKind::BinomialSas(*state)
            }
            (ResponseFamily::Binomial, InverseLink::BetaLogistic(state)) => {
                FamilySpecKind::BinomialBetaLogistic(*state)
            }
            (ResponseFamily::Binomial, InverseLink::Mixture(state)) => {
                FamilySpecKind::BinomialMixture(state.clone())
            }
        }
    }

    #[inline]
    pub fn is_binomial(&self) -> bool {
        self.kind().is_binomial()
    }

    #[inline]
    pub fn is_gaussian_identity(&self) -> bool {
        self.kind().is_gaussian_identity()
    }

    #[inline]
    pub fn is_royston_parmar(&self) -> bool {
        self.kind().is_royston_parmar()
    }

    #[inline]
    pub fn is_latent_cloglog(&self) -> bool {
        self.kind().is_latent_cloglog()
    }

    #[inline]
    pub fn is_binomial_mixture(&self) -> bool {
        self.kind().is_binomial_mixture()
    }

    #[inline]
    pub fn is_binomial_sas(&self) -> bool {
        self.kind().is_binomial_sas()
    }

    #[inline]
    pub fn is_binomial_beta_logistic(&self) -> bool {
        self.kind().is_binomial_beta_logistic()
    }

    /// Default scale metadata for this (response, link).
    #[inline]
    pub fn default_scale_metadata(&self) -> LikelihoodScaleMetadata {
        match &self.response {
            ResponseFamily::Gaussian => LikelihoodScaleMetadata::ProfiledGaussian,
            ResponseFamily::Gamma => LikelihoodScaleMetadata::EstimatedGammaShape { shape: 1.0 },
            ResponseFamily::Binomial
            | ResponseFamily::Poisson
            | ResponseFamily::Tweedie { .. }
            | ResponseFamily::NegativeBinomial { .. } => {
                LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 }
            }
            ResponseFamily::Beta { phi } => LikelihoodScaleMetadata::FixedDispersion { phi: *phi },
            ResponseFamily::RoystonParmar => LikelihoodScaleMetadata::Unspecified,
        }
    }

    /// Human-readable label, routed through `FamilySpecKind`.
    #[inline]
    pub fn pretty_name(&self) -> &'static str {
        self.kind().pretty_name()
    }

    /// Short identifier, routed through `FamilySpecKind`.
    #[inline]
    pub fn name(&self) -> &'static str {
        self.kind().name()
    }

    #[inline]
    pub fn supports_firth(&self) -> bool {
        self.kind().supports_firth()
    }

    /// Family-level fixed-dispersion contract. Returns the dispersion parameter
    /// `phi` that the GLM log-likelihood / weight expressions treat as fixed
    /// for the given `ResponseFamily`, or `None` when the family carries no
    /// fixed scale (profiled or jointly estimated).
    ///
    /// - `Gaussian` and `Gamma` profile/estimate the scale jointly with the
    ///   mean, so no fixed `phi` is exposed here.
    /// - `Binomial`, `Poisson`, `Tweedie`, and `NegativeBinomial` are
    ///   unit-scale exponential-family fits (overdispersion in NB is encoded
    ///   in `theta`, not in `phi`), so the contract is `Some(1.0)`.
    /// - `Beta { phi }` carries its precision parameter directly on the family
    ///   variant; the contract returns that exact value rather than the
    ///   placeholder used elsewhere for unit-scale GLMs.
    /// - `RoystonParmar` has no GLM-style dispersion slot.
    #[inline]
    pub const fn fixed_dispersion(&self) -> Option<f64> {
        match self.response {
            ResponseFamily::Gaussian | ResponseFamily::Gamma | ResponseFamily::RoystonParmar => {
                None
            }
            ResponseFamily::Binomial
            | ResponseFamily::Poisson
            | ResponseFamily::Tweedie { .. }
            | ResponseFamily::NegativeBinomial { .. } => Some(1.0),
            ResponseFamily::Beta { phi } => Some(phi),
        }
    }
}

#[inline]
pub const fn is_valid_tweedie_power(p: f64) -> bool {
    p.is_finite() && p > 1.0 && p < 2.0
}

/// Error returned when an `InverseLink` cannot be paired with a particular
/// response family because the link is structurally unsupported for that
/// family. Carries the link name so call sites can produce a useful message
/// without losing the offending variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedLinkError {
    pub family: &'static str,
    pub link_name: String,
}

impl UnsupportedLinkError {
    /// Construct an `UnsupportedLinkError` tagged with the response-family
    /// name (`"binomial"`, `"gaussian"`, ...) and a printable name for the
    /// offending `InverseLink` variant (extracted via the module-private
    /// `inverse_link_diagnostic_name`). No allocation beyond the link name.
    #[inline]
    pub fn new(family: &'static str, link: &InverseLink) -> Self {
        Self {
            family,
            link_name: inverse_link_diagnostic_name(link),
        }
    }
}

impl std::fmt::Display for UnsupportedLinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "inverse link `{}` is not supported by the {} response family",
            self.link_name, self.family
        )
    }
}

impl std::error::Error for UnsupportedLinkError {}

#[inline]
fn inverse_link_diagnostic_name(link: &InverseLink) -> String {
    match link {
        InverseLink::Standard(lf) => lf.name().to_string(),
        InverseLink::LatentCLogLog(_) => "latent-cloglog".to_string(),
        InverseLink::Sas(_) => "sas".to_string(),
        InverseLink::BetaLogistic(_) => "beta-logistic".to_string(),
        InverseLink::Mixture(_) => "mixture".to_string(),
    }
}

/// Resolve a binomial-flavoured `LikelihoodSpec` from an `InverseLink`.
///
/// `StandardLink::Logit | Probit | CLogLog` and the state-bearing
/// `LatentCLogLog / Sas / BetaLogistic / Mixture` variants are accepted as
/// binomial-compatible. `StandardLink::Log | Identity` have no canonical
/// binomial meaning and return `UnsupportedLinkError`. Since
/// `InverseLink::Standard` carries `StandardLink` (not `LinkFunction`), the
/// previously-required `Standard(LinkFunction::Sas | BetaLogistic)` arm is
/// structurally impossible and has been removed.
#[inline]
pub fn inverse_link_to_binomial_spec(
    link: &InverseLink,
) -> Result<LikelihoodSpec, UnsupportedLinkError> {
    match link {
        InverseLink::Standard(StandardLink::Logit)
        | InverseLink::Standard(StandardLink::Probit)
        | InverseLink::Standard(StandardLink::CLogLog) => {
            Ok(LikelihoodSpec::new(ResponseFamily::Binomial, link.clone()))
        }
        InverseLink::LatentCLogLog(_)
        | InverseLink::Sas(_)
        | InverseLink::BetaLogistic(_)
        | InverseLink::Mixture(_) => {
            Ok(LikelihoodSpec::new(ResponseFamily::Binomial, link.clone()))
        }
        InverseLink::Standard(StandardLink::Log)
        | InverseLink::Standard(StandardLink::Identity) => {
            Err(UnsupportedLinkError::new("binomial", link))
        }
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
    pub const fn fixed_phi(self) -> Option<f64> {
        match self {
            Self::FixedDispersion { phi } => Some(phi),
            Self::FixedGammaShape { shape } | Self::EstimatedGammaShape { shape } => {
                Some(1.0 / shape)
            }
            Self::ProfiledGaussian | Self::Unspecified => None,
        }
    }

    #[inline]
    pub const fn gamma_shape(self) -> Option<f64> {
        match self {
            Self::FixedGammaShape { shape } | Self::EstimatedGammaShape { shape } => Some(shape),
            _ => None,
        }
    }

    #[inline]
    pub const fn gamma_shape_is_estimated(self) -> bool {
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

/// Explicit GLM likelihood specification: response/link spec plus scale semantics.
///
/// `spec` is the canonical `(ResponseFamily, InverseLink)` selector. `scale`
/// records how the scale parameter is handled (profiled Gaussian sigma, fixed
/// dispersion, fixed/estimated Gamma shape). The Gamma shape is mutated in
/// place during PIRLS via `with_gamma_shape`; preserving that field on this
/// struct is what lets the inner solver thread the estimated shape into
/// deviance / log-likelihood / weight evaluation without a separate side
/// channel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlmLikelihoodSpec {
    pub spec: LikelihoodSpec,
    pub scale: LikelihoodScaleMetadata,
}

impl GlmLikelihoodSpec {
    /// Build a `GlmLikelihoodSpec` from a `LikelihoodSpec`, deriving the
    /// canonical default scale metadata for the response family.
    #[inline]
    pub fn canonical(spec: LikelihoodSpec) -> Self {
        let scale = spec.default_scale_metadata();
        Self { spec, scale }
    }

    #[inline]
    pub fn link_function(&self) -> LinkFunction {
        self.spec.link_function()
    }

    #[inline]
    pub fn fixed_phi(&self) -> Option<f64> {
        self.scale.fixed_phi()
    }

    #[inline]
    pub fn gamma_shape(&self) -> Option<f64> {
        self.scale.gamma_shape()
    }

    /// Mutate the Gamma shape parameter in place while preserving the rest of
    /// the spec. The shape only takes effect for Gamma families; for other
    /// families the scale metadata is left untouched.
    #[inline]
    pub fn with_gamma_shape(mut self, shape: f64) -> Self {
        self.scale = match self.scale {
            LikelihoodScaleMetadata::FixedGammaShape { .. } => {
                LikelihoodScaleMetadata::FixedGammaShape { shape }
            }
            LikelihoodScaleMetadata::EstimatedGammaShape { .. } => {
                LikelihoodScaleMetadata::EstimatedGammaShape { shape }
            }
            other => match &self.spec.response {
                ResponseFamily::Gamma => LikelihoodScaleMetadata::EstimatedGammaShape { shape },
                _ => other,
            },
        };
        self
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
    pub const fn explicit_stabilization_full() -> Self {
        Self {
            rho_independent: true,
            include_quadratic_penalty: true,
            include_penalty_logdet: true,
            include_laplacehessian: true,
            determinant_mode: RidgeDeterminantMode::Auto,
        }
    }

    pub const fn explicit_stabilization_full_exact() -> Self {
        Self {
            rho_independent: true,
            include_quadratic_penalty: true,
            include_penalty_logdet: true,
            include_laplacehessian: true,
            determinant_mode: RidgeDeterminantMode::Full,
        }
    }

    /// Variant used when pseudo-determinants are required for indefinite matrices.
    pub const fn explicit_stabilization_pospart() -> Self {
        Self {
            rho_independent: true,
            include_quadratic_penalty: true,
            include_penalty_logdet: true,
            include_laplacehessian: true,
            determinant_mode: RidgeDeterminantMode::PositivePart,
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
    pub const fn scaled_identity(delta: f64, policy: RidgePolicy) -> Self {
        Self {
            delta,
            matrix_form: RidgeMatrixForm::ScaledIdentity,
            policy,
        }
    }

    #[inline]
    pub const fn penalty_logdet_ridge(self) -> f64 {
        if self.policy.include_penalty_logdet {
            self.delta
        } else {
            0.0
        }
    }

    #[inline]
    pub const fn laplacehessianridge(self) -> f64 {
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
// `ledger.quadratic_delta()` rather than rediscovering the policy. The three
// inclusion bits were lifted into the `StabilizationKind` discriminant so the
// (kind, inclusion-flags) invariant is enforced statically — heterogeneous
// combinations like "ExplicitPrior with quadratic excluded" no longer typecheck.
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
}

impl StabilizationLedger {
    /// "No stabilization applied at this site" sentinel.
    pub const fn none() -> Self {
        Self {
            kind: StabilizationKind::None,
            delta: 0.0,
            matrix_form: RidgeMatrixForm::ScaledIdentity,
            chosen_by: StabilizationRule::FixedConstant,
            inertia_before: None,
            inertia_after: None,
        }
    }

    /// LM/TR damping. δ is invisible to the objective, gradient, and any
    /// saved artifact. Asserting this invariant at every read site is the
    /// whole reason the ledger exists.
    pub const fn solver_damping(delta: f64, chosen_by: StabilizationRule) -> Self {
        Self {
            kind: StabilizationKind::SolverDampingOnly,
            delta,
            matrix_form: RidgeMatrixForm::ScaledIdentity,
            chosen_by,
            inertia_before: None,
            inertia_after: None,
        }
    }

    /// Solver-only perturbation that leaves the objective unchanged. The
    /// caller may attach a backward-error bound when one is available
    /// (e.g. from iterative refinement / Wilkinson-style analysis).
    pub const fn numerical_perturbation(
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
        }
    }

    /// Model-level explicit prior. δ enters every accounting pass: the
    /// quadratic penalty, the Laplace Hessian, the penalty log-determinant,
    /// and serialization.
    pub const fn explicit_prior(delta: f64, matrix_form: RidgeMatrixForm) -> Self {
        Self {
            kind: StabilizationKind::ExplicitPrior,
            delta,
            matrix_form,
            chosen_by: StabilizationRule::UserSpecified,
            inertia_before: None,
            inertia_after: None,
        }
    }

    /// Bridge from the existing `RidgePassport` so PIRLS-side code (which
    /// already passes a `RidgePassport` through every call) can hand a
    /// ledger to anything that wants the new uniform view.
    ///
    /// `RidgePolicy` is homogeneous-by-construction: every constructor sets
    /// the three inclusion flags identically. A passport whose policy
    /// excludes every accounting term is morally a numerical perturbation
    /// (the ridge is there to make the solve work but the objective ignores
    /// it); a passport whose policy includes every accounting term is an
    /// explicit prior. Heterogeneous flag combinations cannot be produced
    /// by the public `RidgePolicy` API and have no inhabitants downstream.
    pub const fn from_passport(passport: RidgePassport) -> Self {
        let any_included = passport.policy.include_quadratic_penalty
            || passport.policy.include_laplacehessian
            || passport.policy.include_penalty_logdet;
        let kind = if any_included {
            StabilizationKind::ExplicitPrior
        } else {
            StabilizationKind::NumericalPerturbation {
                backward_error_bound: None,
            }
        };
        Self {
            kind,
            delta: passport.delta,
            matrix_form: passport.matrix_form,
            chosen_by: StabilizationRule::FixedConstant,
            inertia_before: None,
            inertia_after: None,
        }
    }

    /// δ value to fold into the quadratic penalty term, or 0.0 if this
    /// ledger entry is not part of the model. Derived from `kind`: only
    /// [`StabilizationKind::ExplicitPrior`] contributes.
    #[inline]
    pub const fn quadratic_delta(&self) -> f64 {
        match self.kind {
            StabilizationKind::ExplicitPrior => self.delta,
            StabilizationKind::None
            | StabilizationKind::SolverDampingOnly
            | StabilizationKind::NumericalPerturbation { .. } => 0.0,
        }
    }

    /// δ value to add to the Laplace Hessian, or 0.0 if not included.
    /// Derived from `kind`: only [`StabilizationKind::ExplicitPrior`]
    /// contributes.
    #[inline]
    pub const fn laplace_hessian_delta(&self) -> f64 {
        match self.kind {
            StabilizationKind::ExplicitPrior => self.delta,
            StabilizationKind::None
            | StabilizationKind::SolverDampingOnly
            | StabilizationKind::NumericalPerturbation { .. } => 0.0,
        }
    }

    /// δ value to add inside log|S + δ I|, or 0.0 if not included.
    /// Derived from `kind`: only [`StabilizationKind::ExplicitPrior`]
    /// contributes.
    #[inline]
    pub const fn penalty_logdet_delta(&self) -> f64 {
        match self.kind {
            StabilizationKind::ExplicitPrior => self.delta,
            StabilizationKind::None
            | StabilizationKind::SolverDampingOnly
            | StabilizationKind::NumericalPerturbation { .. } => 0.0,
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

/// Index into `TermCollectionSpec::smooth_terms` (and the parallel
/// `TermCollectionDesign::smooth.terms` slice produced from it).
///
/// This is **not** a penalty/ρ index, **not** a column index, and **not** a
/// coefficient-offset index. Keeping it behind a `#[repr(transparent)]`
/// newtype makes those confusables a compile error: a `SmoothTermIdx` cannot
/// be silently used to index `rho`, `beta`, or a design column.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SmoothTermIdx(usize);

impl SmoothTermIdx {
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    /// Sentinel used by transient builders that must allocate a coord config
    /// before the smooth term it references has been positioned in the spec.
    /// Every code path that constructs a sentinel must overwrite it before
    /// the value escapes the builder.
    #[inline]
    pub const fn placeholder() -> Self {
        Self(usize::MAX)
    }

    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }

    #[inline]
    pub const fn is_placeholder(self) -> bool {
        self.0 == usize::MAX
    }
}

impl std::fmt::Display for SmoothTermIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Index into the canonical penalty list `&[CanonicalPenalty]` — equivalently,
/// the position of a smoothing parameter in the ρ / λ vector.
///
/// Penalty/ρ indices are not interchangeable with `SmoothTermIdx` (a smooth
/// term can carry multiple canonical penalties — e.g. tensor-product double
/// penalties — and structural penalties don't correspond to any smooth term).
/// Keeping them as separate newtypes makes the historical bug pattern
/// "indexed `rho` with a smooth-term ordinal" impossible to express.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PenaltyIdx(usize);

impl PenaltyIdx {
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for PenaltyIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Index into a single smooth term's set of basis functions — i.e. the `k`
/// in "the k-th basis function `B_k(x)` of this term".
///
/// Distinct from:
///   * [`SmoothTermIdx`] — selects *which* smooth term in the spec.
///   * [`PenaltyIdx`]    — selects *which* ρ/λ entry / canonical penalty.
///   * A design-matrix column index — which lives in the *combined* layout
///     after intercept/parametric blocks and per-term offsets are applied;
///     a `BasisIdx` is term-local, a column index is model-global.
///
/// Keeping this as its own `#[repr(transparent)]` newtype makes the
/// historically-easy confusion "indexed a global column slice with a
/// term-local basis ordinal" (or vice versa) a compile error.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BasisIdx(usize);

impl BasisIdx {
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for BasisIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Index into the user-facing design matrix `data: Array2<f64>` — i.e. the
/// position of a covariate column in the raw input frame, *before* any
/// per-family basis expansion or intercept/parametric layout is applied.
///
/// Distinct from:
///   * [`BasisIdx`] — term-local basis-function ordinal `k` of `B_k(x)`.
///   * [`SmoothTermIdx`] — position in `TermCollectionSpec::smooth_terms`.
///   * A coefficient-vector offset `β[i]` — spans the combined design after
///     expansion, which is much wider than the user-facing data matrix.
///
/// Keeping this as its own `#[repr(transparent)]` newtype rules out the easy
/// confusion of indexing the raw data frame with an expanded-column offset.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ColIdx(usize);

impl ColIdx {
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for ColIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Index of an observation (row) in the user-facing data frame / design
/// matrix — i.e. the `i` in "the i-th observation".
///
/// Distinct from every column-type index in this module ([`ColIdx`],
/// [`BasisIdx`], [`SmoothTermIdx`], [`PenaltyIdx`]) and from coefficient
/// offsets. Keeping rows behind their own `#[repr(transparent)]` newtype
/// makes the classic `data[[col, row]]` transposition a compile error.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RowIdx(usize);

impl RowIdx {
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for RowIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
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

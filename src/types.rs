use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

/// Lower floor on positive working weights shared by likelihood families and
/// PIRLS row assembly so weighted normal equations stay numerically well posed.
pub const MIN_WEIGHT: f64 = 1e-12;

/// Hyperprior placed on a coefficient group's precision / log-precision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CoefficientGroupPrior {
    Flat,
    NormalLogPrecision {
        mean: f64,
        sd: f64,
    },
    GammaPrecision {
        shape: f64,
        rate: f64,
    },
    /// Penalized-complexity prior calibrated by `P(exp(-rho/2) > upper) =
    /// tail_prob`; see [`RhoPrior::PenalizedComplexity`].
    PenalizedComplexity {
        upper: f64,
        tail_prob: f64,
    },
}

impl CoefficientGroupPrior {
    pub fn to_rho_prior(&self) -> RhoPrior {
        match *self {
            Self::Flat => RhoPrior::Flat,
            Self::NormalLogPrecision { mean, sd } => RhoPrior::Normal { mean, sd },
            Self::GammaPrecision { shape, rate } => RhoPrior::GammaPrecision { shape, rate },
            Self::PenalizedComplexity { upper, tail_prob } => {
                RhoPrior::PenalizedComplexity { upper, tail_prob }
            }
        }
    }

    pub fn validate(&self, context: &str) -> Result<(), String> {
        match *self {
            Self::Flat => Ok(()),
            Self::NormalLogPrecision { mean, sd } => {
                if !mean.is_finite() {
                    return Err(format!(
                        "{context} Normal log-precision prior requires finite mean, got {mean}"
                    ));
                }
                if !sd.is_finite() || sd <= 0.0 {
                    return Err(format!(
                        "{context} Normal log-precision prior requires sd > 0, got {sd}"
                    ));
                }
                Ok(())
            }
            Self::GammaPrecision { shape, rate } => {
                if !shape.is_finite() || shape <= 0.0 {
                    return Err(format!(
                        "{context} Gamma precision prior requires shape > 0, got {shape}"
                    ));
                }
                if !rate.is_finite() || rate < 0.0 {
                    return Err(format!(
                        "{context} Gamma precision prior requires rate >= 0, got {rate}"
                    ));
                }
                Ok(())
            }
            Self::PenalizedComplexity { upper, tail_prob } => {
                if !upper.is_finite() || upper <= 0.0 {
                    return Err(format!(
                        "{context} penalized-complexity prior requires upper > 0, got {upper}"
                    ));
                }
                if !tail_prob.is_finite() || tail_prob <= 0.0 || tail_prob >= 1.0 {
                    return Err(format!(
                        "{context} penalized-complexity prior requires tail probability in (0, 1), got {tail_prob}"
                    ));
                }
                Ok(())
            }
        }
    }
}

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

/// Runtime state shared by the two-parameter `Sas` and `BetaLogistic` links:
/// an `epsilon` skew/asymmetry term plus a raw log-scale parameter (`log_delta`)
/// and its derived positive companion (`delta`). The `delta` field's meaning is
/// link-specific — see its doc — so derivative kernels must consume `log_delta`,
/// never `delta`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SasLinkState {
    pub epsilon: f64,
    /// Raw optimization parameter. For `Sas` this is the pre-bound log-tail; for
    /// `BetaLogistic` it is the unconstrained log geometric-mean beta shape (the
    /// `log_shape_center` the beta-logistic kernels expect).
    pub log_delta: f64,
    /// Derived positive companion of `log_delta`. Its meaning depends on the link:
    /// - `Sas`: effective tail parameter `delta = exp(B * tanh(log_delta / B))`,
    ///   `B = SAS_LOG_DELTA_BOUND`.
    /// - `BetaLogistic`: geometric-mean beta shape `exp(log_delta) = sqrt(a*b)`.
    /// The beta-logistic derivative kernels take `log_delta` (the log center), so
    /// passing this exponentiated `delta` to them would be off by an `exp`.
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

/// Whether the inverse link exposes the Fisher-weight jet that the Firth
/// penalty's higher-order correction consumes. Inlined here (issue #1521) so
/// `LikelihoodSpec::supports_firth` has no upward dependency on
/// `solver::mixture_link` — the match is over link variants all defined in this
/// module, so the predicate is self-contained. The canonical jet evaluation
/// still lives in `solver::mixture_link`; this is purely the classifier.
#[inline]
fn inverse_link_has_fisher_weight_jet(link: &InverseLink) -> bool {
    matches!(
        link,
        InverseLink::Standard(StandardLink::Logit | StandardLink::Probit | StandardLink::CLogLog,)
            | InverseLink::LatentCLogLog(_)
            | InverseLink::Sas(_)
            | InverseLink::BetaLogistic(_)
            | InverseLink::Mixture(_)
    )
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
    /// The deterministic REML/LAML objective uses the MAP-in-lambda convention
    /// and is minimized, so this contributes `rate * exp(rho) - (shape - 1) * rho`
    /// up to an additive constant. Samplers over rho include the +rho Jacobian
    /// from lambda = exp(rho), so their log-density contribution is
    /// `shape * rho - rate * exp(rho)`. For a block with effective dimension n_p
    /// and centered quadratic
    /// `(beta - mu)'S_p(beta - mu)`, the conditional posterior is
    /// `Gamma(shape + n_p/2, rate + quadratic/2)` and the closed-form MAP
    /// precision is `(shape + n_p/2 - 1) / (rate + quadratic/2)`.
    /// `Gamma(1, 0)` is the explicit flat/default case and reproduces the
    /// current MacKay/Tipping fixed point.
    GammaPrecision {
        shape: f64,
        rate: f64,
    },
    /// Penalized-complexity (PC) prior on the smoothing parameter
    /// (Simpson, Rue, Riebler, Martins, Sørbye, *Statistical Science* 2017).
    ///
    /// A PC prior fixes a *base* model (here the infinitely-smooth limit, where
    /// the penalized component collapses to its null space) and puts an
    /// exponential prior on the distance away from it. For a Gaussian smooth
    /// with precision `λ = exp(ρ)` the relevant distance is the marginal
    /// standard-deviation scale `d = λ^{-1/2} = exp(-ρ/2)`, and a constant-rate
    /// penalization `p(d) = θ exp(-θ d)` induces the closed-form log-prior
    ///
    /// ```text
    /// log p(ρ) = ln(θ/2) − ρ/2 − θ exp(−ρ/2).
    /// ```
    ///
    /// The rate `θ` is calibrated by the single interpretable tail statement
    /// `P(d > upper) = tail_prob`, i.e. `θ = −ln(tail_prob) / upper`. The prior
    /// is reparameterization-invariant and shrinks toward the simpler model
    /// (an exponential wall against under-smoothing, only a gentle linear pull
    /// toward over-smoothing), which is exactly the Occam behaviour wanted for
    /// high-variance flexible components. The REML/LAML objective is minimized,
    /// so this contributes `ρ/2 + θ exp(−ρ/2)` (up to an additive constant) to
    /// the cost, with gradient `1/2 − (θ/2) exp(−ρ/2)` and (always positive)
    /// curvature `(θ/4) exp(−ρ/2)`.
    PenalizedComplexity {
        /// Upper bound `U` on the distance scale `d = exp(-ρ/2)` (the marginal
        /// SD scale of the penalized component) in the tail statement
        /// `P(d > U) = tail_prob`. Must be finite and strictly positive.
        upper: f64,
        /// Tail probability `α` in `P(d > U) = α`. Must satisfy `0 < α < 1`.
        tail_prob: f64,
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
    Tweedie {
        p: f64,
    },
    NegativeBinomial {
        theta: f64,
        /// `true` when `theta` was supplied by the user as a held-fixed value
        /// (`--negative-binomial-theta`, issue #983): the fit must use exactly
        /// this overdispersion — `Var(y) = μ + μ²/θ`, IRLS weight
        /// `W = μθ/(θ+μ)`, coefficients/covariance/SEs all reflect it — and
        /// the inner solver must never overwrite it. `false` means `theta` is
        /// the running seed/estimate refined from the data each inner solve
        /// (the #802 default). Carried on the family variant — the canonical
        /// `theta` store — so the estimated-vs-fixed contract can never desync
        /// from the value itself; `default_scale_metadata` derives the
        /// matching scale variant.
        theta_fixed: bool,
    },
    Beta {
        phi: f64,
    },
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

    /// Closed numeric bounds of the **response support** — the closure of the
    /// set of values a single observation `Y` can take — used to clamp the
    /// *observation (prediction) interval* so a predictive band never reports
    /// values the response can never attain.
    ///
    /// This is deliberately distinct from [`Self::mean_clamp_bounds`], which
    /// governs the *mean* (confidence) interval. `mean_clamp_bounds` returns
    /// `None` for the non-negative-real families (Poisson / Tweedie /
    /// NegativeBinomial / Gamma) because their default mean interval is built
    /// by transforming the η endpoints through a positive inverse link, which
    /// cannot escape the support. The observation interval, by contrast, is the
    /// symmetric response-scale band `μ ± z·σ_pred`; for a small fitted mean its
    /// lower endpoint crosses below the support floor (e.g. a Poisson count band
    /// going negative), so it must be floored at the response support here.
    ///
    /// The lower edge is the infimum of the support (`0` for every non-negative
    /// family, including the open-at-zero Gamma, whose predictive lower bound is
    /// reported at the boundary `0`). The upper edge is `+∞` where the response
    /// is unbounded above, which leaves the upper band untouched, or `1` for the
    /// `[0, 1]`-valued families. `None` means the response is supported on the
    /// whole real line (Gaussian) or has its support enforced downstream
    /// (Royston–Parmar), and the predictive band is passed through unclamped.
    ///
    /// The match arms mirror [`Self::response_support_contains`]: a new family
    /// must update both together so the support a value is validated against and
    /// the support a predictive band is clamped to stay consistent.
    #[inline]
    pub fn response_support_bounds(&self) -> Option<(f64, f64)> {
        match self {
            Self::Gamma | Self::Poisson | Self::NegativeBinomial { .. } | Self::Tweedie { .. } => {
                Some((0.0, f64::INFINITY))
            }
            Self::Beta { .. } | Self::Binomial => Some((0.0, 1.0)),
            Self::Gaussian | Self::RoystonParmar => None,
        }
    }

    /// Per-family textual description of the response-support requirement.
    /// `None` means the family is supported on the entire real line at the
    /// validation layer (Gaussian) or has its support enforced by a downstream
    /// pathway (RoystonParmar via the survival pipeline).
    ///
    /// `Binomial` is the scalar Bernoulli-logit family: with no per-row trial
    /// count `mᵢ`, the log-likelihood is `ℓ(η) = y·η − log(1 + eη)`, which is
    /// unbounded above for `y ∉ {0, 1}` (as `η → ∞`, `ℓ ~ (y − 1)·η`), and the
    /// binomial deviance term `(1 − y)·log((1 − y)/(1 − μ))` leaves its domain
    /// for `y > 1`. The family is therefore only well-posed for `y ∈ {0, 1}`,
    /// which the support check enforces up front rather than deferring to the
    /// downstream binarity heuristic.
    #[inline]
    pub fn response_support_requirement(&self) -> Option<&'static str> {
        match self {
            Self::Gamma => Some("strictly positive response values (y > 0)"),
            Self::Poisson | Self::NegativeBinomial { .. } | Self::Tweedie { .. } => {
                Some("non-negative response values (y ≥ 0)")
            }
            Self::Beta { .. } => Some("response values strictly in the open interval (0, 1)"),
            Self::Binomial => Some("binary response values (y ∈ {0, 1})"),
            Self::Gaussian | Self::RoystonParmar => None,
        }
    }

    /// Predicate that returns `true` iff `yi` lies in this family's response
    /// support. Only meaningful for families with a non-trivial domain
    /// constraint at the validation layer; `validate_response_support` calls
    /// this only after `response_support_requirement` returns `Some`, so the
    /// "unconstrained" families (Gaussian / RoystonParmar) never hit this code
    /// path.
    ///
    /// `Binomial` accepts only an (exactly) binary value: `y` must equal `0` or
    /// `1` to within `BINOMIAL_BINARY_TOL` — the same `{0, 1}` test used by the
    /// auto-inference and degeneracy paths — because the scalar Bernoulli-logit
    /// likelihood is unbounded for any other value (see
    /// `response_support_requirement`).
    #[inline]
    fn response_support_contains(&self, yi: f64) -> bool {
        match self {
            Self::Gamma => yi.is_finite() && yi > 0.0,
            Self::Poisson | Self::NegativeBinomial { .. } | Self::Tweedie { .. } => {
                yi.is_finite() && yi >= 0.0
            }
            Self::Beta { .. } => yi.is_finite() && yi > 0.0 && yi < 1.0,
            Self::Binomial => {
                yi.is_finite()
                    && ((yi - 0.0).abs() < BINOMIAL_BINARY_TOL
                        || (yi - 1.0).abs() < BINOMIAL_BINARY_TOL)
            }
            Self::Gaussian | Self::RoystonParmar => true,
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
    /// pathway (RoystonParmar via the survival pipeline). `Binomial` is
    /// enforced here: only `y ∈ {0, 1}` keeps the scalar Bernoulli-logit
    /// likelihood bounded.
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

    /// Detect a *degenerate* response: one whose value distribution makes the
    /// family's REML log-likelihood non-finite even though every individual
    /// `y_i` lies inside the family's distributional support.
    ///
    /// Symmetric counterpart to [`Self::validate_response_support`]: support
    /// rejects out-of-domain *values* (e.g. a negative Poisson count); this
    /// rejects *distributions* that send the saturated MLE to a boundary at
    /// which the score diverges. Each family answers the question for itself
    /// — adding a new family does not require touching workflow.rs.
    ///
    /// Concretely:
    /// * `Binomial` — refuses an all-zero or all-one response: the saturated
    ///   logit is ±∞ and the REML score is +∞ (issue #331).
    /// * Every other family currently returns `Ok(())` at this layer — the
    ///   support check already guarantees enough variation to make the
    ///   log-likelihood finite.
    pub fn validate_response_degeneracy(
        &self,
        y: ArrayView1<'_, f64>,
    ) -> Result<(), ResponseDegeneracy> {
        match self {
            Self::Binomial => {
                let mut saw_zero = false;
                let mut saw_one = false;
                for &yi in y.iter() {
                    if (yi - 0.0).abs() < BINOMIAL_BINARY_TOL {
                        saw_zero = true;
                    } else if (yi - 1.0).abs() < BINOMIAL_BINARY_TOL {
                        saw_one = true;
                    }
                    if saw_zero && saw_one {
                        return Ok(());
                    }
                }
                let kind = if saw_one {
                    ResponseDegeneracyKind::BinomialAllOnes
                } else if saw_zero {
                    ResponseDegeneracyKind::BinomialAllZeros
                } else {
                    // Reachable only for an empty response: the support check
                    // (`validate_response_support`) has already rejected any
                    // non-binary value, so every present `yᵢ` is exactly 0 or 1
                    // and at least one of `saw_zero`/`saw_one` is set whenever
                    // `y` is non-empty. An empty response carries no
                    // saturated-boundary degeneracy, so accept it here.
                    return Ok(());
                };
                Err(ResponseDegeneracy {
                    family_label: self.response_support_label(),
                    kind,
                })
            }
            Self::Gaussian => Ok(()),
            Self::Poisson
            | Self::Tweedie { .. }
            | Self::NegativeBinomial { .. }
            | Self::Beta { .. }
            | Self::Gamma
            | Self::RoystonParmar => Ok(()),
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
    ///   * A non-negative integer-valued count response (every value finite,
    ///     `>= 0`, and within [`COUNT_INTEGER_TOL`] of an integer) that reaches
    ///     beyond the binary `{0, 1}` window (i.e. carries at least one value
    ///     `>= 2`) maps to `Poisson` (log link). This is the "magic-by-default"
    ///     count detection: mgcv/statsmodels users expect `0,1,2,3,...` to fit a
    ///     Poisson GLM, not an identity-link Gaussian.
    ///   * Anything else (any fractional or negative value) maps to `Gaussian`.
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
                        v.is_finite()
                            && ((*v - 0.0).abs() < BINOMIAL_BINARY_TOL
                                || (*v - 1.0).abs() < BINOMIAL_BINARY_TOL)
                    });
                if binary {
                    return Ok(Self::Binomial);
                }
                // Count signature: every value finite, non-negative, and an
                // integer within `COUNT_INTEGER_TOL`, with at least one value
                // `>= 2` so it is not the (already-handled) binary case and not
                // a degenerate all-zero column. A single fractional or negative
                // value disqualifies the whole response, keeping continuous and
                // signed data on the conservative Gaussian default.
                let count = !y.is_empty()
                    && y.iter().all(|v| {
                        v.is_finite() && *v >= 0.0 && (*v - v.round()).abs() <= COUNT_INTEGER_TOL
                    })
                    && y.iter().any(|v| *v >= 2.0 - COUNT_INTEGER_TOL);
                if count {
                    Ok(Self::Poisson)
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
    /// Keeps the message bounded on large-scale data while still pointing
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

/// Absolute tolerance for the exact-`{0, 1}` test that defines the scalar
/// Bernoulli (`Binomial`) response support.
///
/// The scalar `Binomial` family carries no per-row trial count, so its
/// log-likelihood is the Bernoulli/soft-label cross-entropy
/// `ℓ(η) = y·η − log(1 + eη)`, which is unbounded above for `y ∉ {0, 1}`.
/// Both the auto-inference (`infer_from_response`) and degeneracy
/// (`validate_response_degeneracy`) paths classify a value as binary by the
/// same `1e-12` window; the support check shares this single threshold so the
/// three layers agree on exactly which responses are admissible.
pub const BINOMIAL_BINARY_TOL: f64 = 1.0e-12;

/// Round tolerance for recognising an integer-valued (count) response.
///
/// `infer_from_response` classifies a numeric response as a Poisson count when
/// every value is finite, non-negative, and within this window of its nearest
/// non-negative integer. The threshold is looser than [`BINOMIAL_BINARY_TOL`]
/// because count columns frequently arrive as `f64` round-trips of integers
/// (CSV parse, integer→double promotion) that accumulate ULP-scale error well
/// above `1e-12`; `1e-9` admits those without ever matching genuinely
/// continuous data, whose fractional parts are O(1).
pub const COUNT_INTEGER_TOL: f64 = 1.0e-9;

/// Classifier for a [`ResponseDegeneracy`]. Each variant carries the family-
/// specific evidence the caller needs to format a useful message without
/// having to re-derive the diagnostic.
#[derive(Debug, Clone)]
pub enum ResponseDegeneracyKind {
    /// Bernoulli / Binomial response with every observed value equal to 0.
    BinomialAllZeros,
    /// Bernoulli / Binomial response with every observed value equal to 1.
    BinomialAllOnes,
}

/// Degenerate-response detail produced by
/// [`ResponseFamily::validate_response_degeneracy`].
///
/// Mirrors [`ResponseSupportViolation`]: it owns its own `Display` and
/// `message_for(column_name)` so call sites in the workflow, the CLI, and
/// any future binding produce identical user-facing prose without coupling
/// each one to the family-internal classifier.
#[derive(Debug, Clone)]
pub struct ResponseDegeneracy {
    pub family_label: &'static str,
    pub kind: ResponseDegeneracyKind,
}

impl ResponseDegeneracy {
    /// Format the degeneracy against a specific response column name. The
    /// column name is supplied by the caller because [`ResponseFamily`] does
    /// not know which column the user pointed at.
    pub fn message_for(&self, response_name: &str) -> String {
        match self.kind {
            ResponseDegeneracyKind::BinomialAllZeros => format!(
                "{family} response '{name}' is degenerate: all values are 0 (no events). \
                 The maximum-likelihood logit is −∞ at this boundary, so the REML score \
                 is not finite. Fix: ensure the response contains at least one 0 and \
                 at least one 1 (e.g. drop the offending subgroup, or refit on a pooled \
                 sample that includes both classes).",
                family = self.family_label,
                name = response_name,
            ),
            ResponseDegeneracyKind::BinomialAllOnes => format!(
                "{family} response '{name}' is degenerate: all values are 1 (no non-events). \
                 The maximum-likelihood logit is +∞ at this boundary, so the REML score \
                 is not finite. Fix: ensure the response contains at least one 0 and \
                 at least one 1 (e.g. drop the offending subgroup, or refit on a pooled \
                 sample that includes both classes).",
                family = self.family_label,
                name = response_name,
            ),
        }
    }
}

impl std::fmt::Display for ResponseDegeneracy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message_for("y"))
    }
}

impl std::error::Error for ResponseDegeneracy {}

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
///
/// Only the legal `(response, link)` cells enumerated by [`LikelihoodSpec::kind`]
/// are representable through the public surface: [`LikelihoodSpec::try_new`]
/// validates the legal matrix on construction, and deserialization routes
/// through [`LikelihoodSpecWire`] (`#[serde(try_from / into)]`) so saved bytes
/// cannot resurrect an illegal cell. The on-wire shape is byte-identical to the
/// historical `{ response, link }` struct, so legal saved models load unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "LikelihoodSpecWire", into = "LikelihoodSpecWire")]
pub struct LikelihoodSpec {
    pub response: ResponseFamily,
    pub link: InverseLink,
}

/// Transparent serde shadow of [`LikelihoodSpec`] with the identical wire shape
/// (`response`, `link`). All (de)serialization of `LikelihoodSpec` routes
/// through this type so the legal-matrix check in
/// [`TryFrom<LikelihoodSpecWire>`] runs on every load, closing the
/// saved-bytes hole: an illegal `(response, link)` cell deserializes into a
/// serde error instead of a silently-masked spec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LikelihoodSpecWire {
    pub response: ResponseFamily,
    pub link: InverseLink,
}

impl From<LikelihoodSpec> for LikelihoodSpecWire {
    #[inline]
    fn from(spec: LikelihoodSpec) -> Self {
        Self {
            response: spec.response,
            link: spec.link,
        }
    }
}

impl TryFrom<LikelihoodSpecWire> for LikelihoodSpec {
    type Error = IllegalLikelihoodCell;

    #[inline]
    fn try_from(wire: LikelihoodSpecWire) -> Result<Self, Self::Error> {
        Self::try_new(wire.response, wire.link)
    }
}

/// Error returned when an illegal `(ResponseFamily, InverseLink)` cell is
/// presented to [`LikelihoodSpec::try_new`] or surfaced during
/// deserialization. Only the cells enumerated by [`LikelihoodSpec::kind`] are
/// legal; every other product cell would silently mask a wrong response
/// transformation (e.g. `Poisson + Identity` predicting `μ = η`, which can go
/// negative).
#[derive(Debug, Clone, PartialEq)]
pub struct IllegalLikelihoodCell {
    pub response: &'static str,
    pub link: &'static str,
}

impl std::fmt::Display for IllegalLikelihoodCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "illegal likelihood cell: response `{}` does not admit inverse link `{}`. \
             Each non-binomial family is pinned to one link (Gaussian/Royston-Parmar→identity, \
             Poisson/Gamma/Tweedie/Negative-Binomial→log, Beta→logit); the binomial family \
             admits logit/probit/cloglog and the latent-cloglog/SAS/beta-logistic/blended \
             links, but not identity/log.",
            self.response, self.link
        )
    }
}

impl std::error::Error for IllegalLikelihoodCell {}

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

    /// Coarse kind-level Firth eligibility: every binomial inverse link this
    /// enum can represent (Logit/Probit/CLogLog and the stateful
    /// LatentCLogLog/SAS/Beta-Logistic/Mixture links) carries a Fisher-weight
    /// jet, so kind-level Firth support is exactly binomial membership.
    ///
    /// The authoritative, link-resolved gate is
    /// [`LikelihoodSpec::supports_firth`], which routes through
    /// `inverse_link_has_fisher_weight_jet`. Keep this in agreement with that
    /// predicate: a future binomial link without a Fisher-weight jet would make
    /// this approximation diverge and must be handled at both sites.
    #[inline]
    pub const fn supports_firth(&self) -> bool {
        self.is_binomial()
    }
}

impl LikelihoodSpec {
    /// Unchecked constructor: assembles a `(response, link)` cell *without*
    /// validating the legal matrix. Reserved for the in-crate named const
    /// constructors below (`gaussian_identity`, `poisson_log`, `beta_logit`,
    /// the `binomial_*` family, …), every one of which builds a cell that is
    /// legal by construction. The public, fallible entry point for an arbitrary
    /// `(response, link)` pair is [`LikelihoodSpec::try_new`]; the serde path
    /// also validates via [`LikelihoodSpecWire`]. Do not expose illegal cells
    /// through this method.
    #[inline]
    pub const fn new(response: ResponseFamily, link: InverseLink) -> Self {
        Self { response, link }
    }

    /// Returns `true` when the `(response, link)` pair is one of the legal cells
    /// the family math honours — exactly the cells enumerated by
    /// [`LikelihoodSpec::kind`] before any masking. Each non-binomial response
    /// is pinned to a single inverse link; the binomial family admits its full
    /// set of probability links but never the identity/log standard links.
    #[inline]
    pub fn is_legal_cell(response: &ResponseFamily, link: &InverseLink) -> bool {
        match response {
            // Pure-identity families.
            ResponseFamily::Gaussian | ResponseFamily::RoystonParmar => {
                matches!(link, InverseLink::Standard(StandardLink::Identity))
            }
            // Log-link families.
            ResponseFamily::Poisson
            | ResponseFamily::Gamma
            | ResponseFamily::Tweedie { .. }
            | ResponseFamily::NegativeBinomial { .. } => {
                matches!(link, InverseLink::Standard(StandardLink::Log))
            }
            // Logit-link family.
            ResponseFamily::Beta { .. } => {
                matches!(link, InverseLink::Standard(StandardLink::Logit))
            }
            // Binomial admits every probability link except the inert
            // identity/log standard links.
            ResponseFamily::Binomial => match link {
                InverseLink::Standard(
                    StandardLink::Logit | StandardLink::Probit | StandardLink::CLogLog,
                ) => true,
                InverseLink::Standard(StandardLink::Identity | StandardLink::Log) => false,
                InverseLink::LatentCLogLog(_)
                | InverseLink::Sas(_)
                | InverseLink::BetaLogistic(_)
                | InverseLink::Mixture(_) => true,
            },
        }
    }

    /// Fallible constructor over an arbitrary `(response, link)` pair. Validates
    /// the legal matrix ([`LikelihoodSpec::is_legal_cell`]) so that an illegal
    /// cell — one whose stored link would drive a wrong response transformation
    /// — is rejected instead of silently masked by [`LikelihoodSpec::kind`].
    #[inline]
    pub fn try_new(
        response: ResponseFamily,
        link: InverseLink,
    ) -> Result<Self, IllegalLikelihoodCell> {
        if Self::is_legal_cell(&response, &link) {
            Ok(Self::new(response, link))
        } else {
            Err(IllegalLikelihoodCell {
                response: response.name(),
                link: link.link_function().name(),
            })
        }
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

    /// Estimated-theta NB spec: `theta` is the seed, refined by the inner
    /// solver (#802 default).
    #[inline]
    pub const fn negative_binomial_log(theta: f64) -> Self {
        Self::new(
            ResponseFamily::NegativeBinomial {
                theta,
                theta_fixed: false,
            },
            InverseLink::Standard(StandardLink::Log),
        )
    }

    /// Fixed-theta NB spec: the fit holds `theta` at exactly this value
    /// (`--negative-binomial-theta`, issue #983).
    #[inline]
    pub const fn negative_binomial_log_fixed(theta: f64) -> Self {
        Self::new(
            ResponseFamily::NegativeBinomial {
                theta,
                theta_fixed: true,
            },
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
    /// `(ResponseFamily, InverseLink)` is a 40-cell product (8 response × 5
    /// inverse-link); only the cells listed here are legal. Construction
    /// ([`LikelihoodSpec::try_new`]) and deserialization (the
    /// [`LikelihoodSpecWire`] `try_from`) both enforce
    /// [`LikelihoodSpec::is_legal_cell`], so an illegal cell can never reach
    /// this method. Each link-pinned family therefore matches its *one* legal
    /// link explicitly; the remaining (now-unreachable) illegal combinations
    /// are `unreachable!()` so the historical silent masking — collapsing e.g.
    /// `Poisson + Identity` to `PoissonLog` while the transform predicted
    /// `μ = η` — can never silently happen again.
    pub fn kind(&self) -> FamilySpecKind {
        // `legal_cell_kind` returns `Some` for every legal cell and `None`
        // for the (by-construction-unreachable) illegal ones. Construction
        // (`try_new`) and deserialization (`LikelihoodSpecWire` try_from)
        // both enforce `is_legal_cell`, so the `None` branch can never fire
        // on a value that exists — `.expect` is the idiomatic loud-on-
        // impossible-state assertion (a banned `unreachable!`/`panic!` macro
        // would be the same panic with worse provenance). If it ever does
        // fire, the message names the offending cell so the silent-masking
        // regression this guards against (e.g. `Poisson + Identity`
        // collapsing to `PoissonLog`) stays impossible.
        self.legal_cell_kind().expect(
            "illegal likelihood cell reached kind(): construction (try_new) and \
             deserialization (LikelihoodSpecWire) guarantee legality",
        )
    }

    fn legal_cell_kind(&self) -> Option<FamilySpecKind> {
        Some(match (&self.response, &self.link) {
            (ResponseFamily::Gaussian, InverseLink::Standard(StandardLink::Identity)) => {
                FamilySpecKind::GaussianIdentity
            }
            (ResponseFamily::RoystonParmar, InverseLink::Standard(StandardLink::Identity)) => {
                FamilySpecKind::RoystonParmar
            }
            (ResponseFamily::Poisson, InverseLink::Standard(StandardLink::Log)) => {
                FamilySpecKind::PoissonLog
            }
            (ResponseFamily::Gamma, InverseLink::Standard(StandardLink::Log)) => {
                FamilySpecKind::GammaLog
            }
            (ResponseFamily::Tweedie { p }, InverseLink::Standard(StandardLink::Log)) => {
                FamilySpecKind::TweedieLog { p: *p }
            }
            (
                ResponseFamily::NegativeBinomial { theta, .. },
                InverseLink::Standard(StandardLink::Log),
            ) => FamilySpecKind::NegativeBinomialLog { theta: *theta },
            (ResponseFamily::Beta { phi }, InverseLink::Standard(StandardLink::Logit)) => {
                FamilySpecKind::BetaLogit { phi: *phi }
            }
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Logit)) => {
                FamilySpecKind::BinomialLogit
            }
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Probit)) => {
                FamilySpecKind::BinomialProbit
            }
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::CLogLog)) => {
                FamilySpecKind::BinomialCLogLog
            }
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
            // Every remaining product cell is illegal. `try_new` /
            // `LikelihoodSpecWire::try_from` reject these, so construction and
            // deserialization guarantee they are unreachable here; `None`
            // surfaces that to `kind()`, which aborts loudly via `.expect`
            // rather than misclassify the family (a wrong `FamilySpecKind`
            // would silently corrupt every downstream likelihood/gradient
            // evaluation). A banned `panic!`/`unreachable!` macro would be the
            // same divergence with worse provenance.
            _ => return None,
        })
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
            // Binomial and Poisson have `phi ≡ 1` (variance fully pinned by the
            // mean), so a fixed unit dispersion is correct.
            ResponseFamily::Binomial | ResponseFamily::Poisson => {
                LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 }
            }
            // Negative-Binomial's overdispersion `theta` (`Var(y)=mu+mu^2/theta`)
            // is a genuine free parameter estimated jointly with the mean by
            // default — the family-variant `theta` is only the seed, refined from
            // the converged-η ML score during fitting, exactly like the Gamma
            // shape / Beta precision / Tweedie φ. Freezing it at the seed made
            // every variance-derived output (coefficient/η SEs, Wald and credible
            // intervals, predictive intervals, `generate` draws) ignore the
            // data's overdispersion (issue #802). `phi` itself stays `≡ 1`.
            //
            // A user-supplied `--negative-binomial-theta` is the opposite
            // contract (issue #983): `theta_fixed = true` routes to the
            // non-estimated scale variant, so the inner solver's refresh gate
            // (`negbin_theta_is_estimated()`) stays closed and the fit honours
            // the held value everywhere it enters.
            ResponseFamily::NegativeBinomial { theta, theta_fixed } => {
                if *theta_fixed {
                    LikelihoodScaleMetadata::FixedNegBinTheta { theta: *theta }
                } else {
                    LikelihoodScaleMetadata::EstimatedNegBinTheta { theta: *theta }
                }
            }
            // Tweedie's dispersion `phi` is a genuine free parameter
            // (`Var(y) = phi · mu^p`) and is estimated jointly with the mean by
            // default, exactly like the Gamma shape and Beta precision. The seed
            // `phi = 1` is refined from the converged-η Pearson residuals during
            // fitting (issue #771). Freezing it at 1 made every variance-derived
            // output (SEs, intervals, generate draws) ignore the data's spread.
            ResponseFamily::Tweedie { .. } => {
                LikelihoodScaleMetadata::EstimatedTweediePhi { phi: 1.0 }
            }
            // Beta precision is estimated jointly with the mean by default
            // (magic-by-default, issue #567): the family-variant `phi` is the
            // seed, refined from the working residuals during fitting.
            ResponseFamily::Beta { phi } => LikelihoodScaleMetadata::EstimatedBetaPhi { phi: *phi },
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
        matches!(self.response, ResponseFamily::Binomial)
            && inverse_link_has_fisher_weight_jet(&self.link)
    }

    /// Family-level fixed-dispersion contract. Returns the dispersion parameter
    /// `phi` that the GLM log-likelihood / weight expressions treat as fixed
    /// for the given `ResponseFamily`, or `None` when the family carries no
    /// fixed scale (profiled or jointly estimated).
    ///
    /// - `Gaussian` and `Gamma` profile/estimate the scale jointly with the
    ///   mean, so no fixed `phi` is exposed here.
    /// - `Binomial` and `Poisson` are unit-scale exponential-family fits, so the
    ///   contract is `Some(1.0)`. NegativeBinomial's overdispersion lives in
    ///   `theta` (a separate parameter / flag), not in a free `phi`, so it also
    ///   returns `Some(1.0)`.
    /// - `Tweedie { p }` carries its variance power on the family variant. Its
    ///   free dispersion `phi` lives in `LikelihoodScaleMetadata` and is
    ///   estimated by default (`EstimatedTweediePhi`, issue #771), so this
    ///   family-level contract only exposes the unit seed used when callers ask
    ///   the response family without scale metadata.
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
pub fn inverse_link_diagnostic_name(link: &InverseLink) -> String {
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
    /// Beta-regression precision `phi` estimated jointly with the mean model.
    /// `Var(y) = mu(1-mu)/(1+phi)`; larger `phi` means less noise. Estimated
    /// from the working residuals after each mean fit and refreshed across outer
    /// iterations, exactly like the Gamma shape (issue #567).
    EstimatedBetaPhi { phi: f64 },
    /// Tweedie exponential-dispersion `phi` estimated jointly with the mean
    /// model. `Var(y) = phi · mu^p` with `phi` a genuine free parameter (unlike
    /// Binomial/Poisson, where `phi ≡ 1`). Estimated by the Pearson moment
    /// estimator `phî = Σ wᵢ (yᵢ − μᵢ)² / μᵢ^p / Σ wᵢ` at the converged η and
    /// refreshed across outer iterations, exactly like the Gamma shape and the
    /// Beta precision. `phi` enters the IRLS working weight `prior·μ^{2−p}/phi`,
    /// so the coefficient covariance `Vb = H⁻¹` already scales as `phi` and the
    /// reported SEs track `√phi` (issue #771).
    EstimatedTweediePhi { phi: f64 },
    /// Negative-Binomial overdispersion `theta` estimated jointly with the mean
    /// model. `Var(y) = mu + mu^2 / theta`; larger `theta` means less
    /// overdispersion (the Poisson limit is `theta → ∞`). Estimated by the
    /// maximum-likelihood `theta` score
    /// `Σ wᵢ[ψ(yᵢ+θ) − ψ(θ) + lnθ + 1 − ln(θ+μᵢ) − (yᵢ+θ)/(μᵢ+θ)] = 0` at the
    /// converged η (MASS `glm.nb`'s `theta.ml`) and refreshed across outer
    /// iterations, exactly like the Gamma shape / Beta precision / Tweedie φ.
    /// Unlike those, `theta` is *not* a dispersion scale `phi`: it enters only
    /// the IRLS working weight `W = μθ/(θ+μ)` (the full NB2 Fisher information),
    /// so the stored penalized Hessian is already the true one and the
    /// coefficient covariance `Vb = H⁻¹` takes no post-hoc multiply — `phi ≡ 1`
    /// for NB, the overdispersion lives in the variance function. The `theta`
    /// carried here mirrors `ResponseFamily::NegativeBinomial { theta }` (the
    /// canonical store every weight/deviance expression reads), kept in sync by
    /// `with_negbin_theta`, exactly as `EstimatedBetaPhi` mirrors `Beta { phi }`
    /// (issue #802).
    EstimatedNegBinTheta { theta: f64 },
    /// Negative-Binomial overdispersion `theta` held fixed at a user-supplied
    /// value (`--negative-binomial-theta`, issue #983). Identical role to
    /// `EstimatedNegBinTheta` in every weight / variance / covariance
    /// expression (`W = μθ/(θ+μ)`, `Var(y) = μ + μ²/θ`, `phi ≡ 1`), but the
    /// inner solver's ML refresh is gated off: the recorded `theta` is the
    /// user's, by construction. The fixed/estimated split mirrors
    /// `FixedGammaShape` vs `EstimatedGammaShape`.
    FixedNegBinTheta { theta: f64 },
    /// The engine does not expose fixed-scale semantics for this family.
    Unspecified,
}

impl LikelihoodScaleMetadata {
    #[inline]
    pub const fn fixed_phi(self) -> Option<f64> {
        match self {
            Self::FixedDispersion { phi }
            | Self::EstimatedBetaPhi { phi }
            | Self::EstimatedTweediePhi { phi } => Some(phi),
            Self::FixedGammaShape { shape } | Self::EstimatedGammaShape { shape } => {
                Some(1.0 / shape)
            }
            // NB's dispersion scale is `phi ≡ 1` (the overdispersion is carried
            // by `theta` inside the variance function, not a scale multiply), so
            // the fixed-`phi` contract is `Some(1.0)` — NOT `theta`.
            Self::EstimatedNegBinTheta { .. } | Self::FixedNegBinTheta { .. } => Some(1.0),
            Self::ProfiledGaussian | Self::Unspecified => None,
        }
    }

    /// Whether the Negative-Binomial overdispersion `theta` is estimated from
    /// data (the default for NB families, issue #802).
    #[inline]
    pub const fn negbin_theta_is_estimated(self) -> bool {
        matches!(self, Self::EstimatedNegBinTheta { .. })
    }

    /// The Negative-Binomial `theta` carried in the scale metadata (estimated
    /// or user-fixed), or `None` for non-NB families.
    #[inline]
    pub const fn negbin_theta(self) -> Option<f64> {
        match self {
            Self::EstimatedNegBinTheta { theta } | Self::FixedNegBinTheta { theta } => Some(theta),
            _ => None,
        }
    }

    /// Whether the Beta-regression precision `phi` is estimated from data.
    #[inline]
    pub const fn beta_phi_is_estimated(self) -> bool {
        matches!(self, Self::EstimatedBetaPhi { .. })
    }

    /// Whether the Tweedie exponential-dispersion `phi` is estimated from data.
    #[inline]
    pub const fn tweedie_phi_is_estimated(self) -> bool {
        matches!(self, Self::EstimatedTweediePhi { .. })
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

    /// Multiplier converting the stored unscaled inverse penalized Hessian
    /// `H⁻¹` into the reported coefficient covariance `Vb = H⁻¹ · scale`.
    ///
    /// # Invariant
    ///
    /// `Vb` is the inverse of the Hessian of the *actual penalized objective the
    /// inner solver minimizes*. The stored Hessian is always assembled as
    /// `H = XᵀWX + S_λ`, with the penalty `S_λ` added **unscaled** (see
    /// `pirls::penalty::add_to_hessian`). Whether `H` is already that true
    /// objective Hessian — and hence whether any post-hoc dispersion multiply is
    /// warranted — is decided entirely by what the IRLS working weight `W`
    /// carries:
    ///
    /// * **Working weight already carries the reciprocal dispersion / full
    ///   Fisher information.** Then `H = Xᵀ(W_sf/φ)X + S_λ` already equals the
    ///   true penalized Hessian (e.g. mgcv's `XᵀW_sfX/φ + S_λ` for Gamma), so
    ///   `Vb = H⁻¹` and the scale is exactly `1.0`. This is the case for Gamma
    ///   (`W = prior·shape = prior/φ`), Tweedie (`W = prior·μ^{2−p}/φ`), Beta
    ///   and Negative-Binomial (the working weight is the complete fixed-scale
    ///   Fisher information), and the fixed-scale exponential families
    ///   Poisson/Binomial (`φ ≡ 1`). Multiplying `H⁻¹` by the dispersion again
    ///   for any of these double-counts it and shrinks every SE by `√dispersion`.
    ///
    /// * **Working weight is scale-free** (`W = priorweights`, the profiled
    ///   Gaussian convention). Then the data term carries an implicit unit scale
    ///   and `H = XᵀPX + S_λ` is the Hessian of `½·(scaled deviance)·σ²⁻¹`
    ///   *without* the `σ²`. The correct covariance restores it:
    ///   `Vb = H⁻¹ · σ̂²`. Only this branch returns a non-unit scale.
    ///
    /// `profiled_gaussian_phi` is the profiled residual variance `σ̂²` and is
    /// consulted **only** for the scale-free profiled-Gaussian branch; every
    /// other family ignores it. This deliberately does NOT touch
    /// `dispersion()` / `dispersion_from_likelihood`, which still report the
    /// response-level observation noise (`1/shape` for Gamma, `1/(1+φ)` for
    /// Beta, …) used by predictive-interval construction — a distinct quantity
    /// from the coefficient-covariance scale defined here.
    #[inline]
    pub fn coefficient_covariance_scale(&self, profiled_gaussian_phi: f64) -> f64 {
        match self.scale {
            // Scale-free working weight: restore the profiled variance.
            LikelihoodScaleMetadata::ProfiledGaussian => profiled_gaussian_phi,
            // Working weight already carries the dispersion / full Fisher
            // information, so the stored H is the true penalized Hessian and no
            // further dispersion multiply is warranted.
            //
            // FixedDispersion covers the explicitly-scaled Gaussian submodel
            // (W·=1/φ above) and Negative-Binomial; the Gamma, Beta and Tweedie
            // variants fold their reciprocal-dispersion / precision / φ into W
            // (Tweedie W = prior·μ^{2−p}/φ, so the SE already scales as √φ); and
            // Unspecified families never expose a separate post-hoc scale.
            LikelihoodScaleMetadata::FixedDispersion { .. }
            | LikelihoodScaleMetadata::FixedGammaShape { .. }
            | LikelihoodScaleMetadata::EstimatedGammaShape { .. }
            | LikelihoodScaleMetadata::EstimatedBetaPhi { .. }
            | LikelihoodScaleMetadata::EstimatedTweediePhi { .. }
            // Negative-Binomial folds `theta` into the working weight
            // `W = μθ/(θ+μ)` (the full NB2 Fisher information), so the stored
            // `H = XᵀWX + S_λ` is already the true penalized Hessian and the
            // covariance scale is `1.0` (`phi ≡ 1`). The reported SEs respond to
            // the data's overdispersion entirely through that `theta`-dependent
            // weight (issue #802) — multiplying again would double-count it.
            // The same holds verbatim for a user-fixed `theta` (issue #983).
            | LikelihoodScaleMetadata::EstimatedNegBinTheta { .. }
            | LikelihoodScaleMetadata::FixedNegBinTheta { .. }
            | LikelihoodScaleMetadata::Unspecified => 1.0,
        }
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

    /// Whether the Beta-regression precision `phi` is estimated from data.
    #[inline]
    pub fn beta_phi_is_estimated(&self) -> bool {
        self.scale.beta_phi_is_estimated()
    }

    /// Mutate the Beta precision `phi` in place, on BOTH the family variant
    /// (where every PIRLS weight / deviance / log-likelihood expression reads it
    /// via `ResponseFamily::Beta { phi }`) and the scale metadata (the
    /// estimated-vs-fixed contract). No-op for non-Beta families. The inner
    /// solver calls this once per inner solve after a moment estimate of `phi`
    /// from the working residuals, so the IRLS weights `Var(y)=mu(1-mu)/(1+phi)`
    /// reflect the true precision rather than the `phi=1` seed (issue #567).
    #[inline]
    pub fn with_beta_phi(mut self, phi: f64) -> Self {
        if let ResponseFamily::Beta { phi: family_phi } = &mut self.spec.response {
            *family_phi = phi;
            self.scale = LikelihoodScaleMetadata::EstimatedBetaPhi { phi };
        }
        self
    }

    /// Whether the Tweedie exponential-dispersion `phi` is estimated from data.
    #[inline]
    pub fn tweedie_phi_is_estimated(&self) -> bool {
        self.scale.tweedie_phi_is_estimated()
    }

    /// Mutate the Tweedie dispersion `phi` in place. Unlike Beta, the Tweedie
    /// power `p` (not `phi`) is what is carried on the `ResponseFamily::Tweedie`
    /// variant; the dispersion lives purely in the scale metadata and is read by
    /// the IRLS weight (`prior·μ^{2−p}/phi`) through `fixed_phi()`. So updating
    /// the metadata here is sufficient to thread the estimated `phi` into every
    /// weight / covariance expression. No-op for non-Tweedie families (issue
    /// #771).
    #[inline]
    pub fn with_tweedie_phi(mut self, phi: f64) -> Self {
        if matches!(self.spec.response, ResponseFamily::Tweedie { .. }) {
            self.scale = LikelihoodScaleMetadata::EstimatedTweediePhi { phi };
        }
        self
    }

    /// Whether the Negative-Binomial overdispersion `theta` is estimated from
    /// data (issue #802).
    #[inline]
    pub fn negbin_theta_is_estimated(&self) -> bool {
        self.scale.negbin_theta_is_estimated()
    }

    /// Mutate the Negative-Binomial overdispersion `theta` in place, on BOTH the
    /// family variant (where every PIRLS weight / deviance / log-likelihood
    /// expression reads it via `ResponseFamily::NegativeBinomial { theta }`) and
    /// the scale metadata (the estimated-vs-fixed contract). No-op for non-NB
    /// families. The inner solver calls this once per inner solve after a
    /// maximum-likelihood estimate of `theta` from the working residuals, so the
    /// IRLS weight `W = μθ/(θ+μ)` and the variance `Var(y)=mu+mu^2/theta` reflect
    /// the data's overdispersion rather than the seed `theta` (issue #802). This
    /// mirrors `with_beta_phi` exactly — both keep the family variant and the
    /// scale metadata as two synchronized views of one estimated parameter.
    /// No-op for a user-fixed `theta` (`theta_fixed = true` /
    /// `FixedNegBinTheta`, issue #983): the held value is the contract, and
    /// this mutator must never let an estimation path overwrite it — the
    /// PIRLS refresh gate (`negbin_theta_is_estimated()`) already skips the
    /// call, this enforces the same invariant at the data itself.
    #[inline]
    pub fn with_negbin_theta(mut self, theta: f64) -> Self {
        if let ResponseFamily::NegativeBinomial {
            theta: family_theta,
            theta_fixed,
        } = &mut self.spec.response
            && !*theta_fixed
        {
            *family_theta = theta;
            self.scale = LikelihoodScaleMetadata::EstimatedNegBinTheta { theta };
        }
        self
    }

    /// The estimated Negative-Binomial `theta`, read from the family variant
    /// (the canonical store), or `None` for non-NB families.
    #[inline]
    pub fn negbin_theta(&self) -> Option<f64> {
        match self.spec.response {
            ResponseFamily::NegativeBinomial { theta, .. } => Some(theta),
            _ => None,
        }
    }

    /// Produce a copy of this spec with the Negative-Binomial overdispersion
    /// `theta` PINNED at `theta` for the duration of the smoothing-parameter
    /// (λ) search (#1082). Converts an `EstimatedNegBinTheta` spec into the
    /// statistically-identical `FixedNegBinTheta` form (`theta_fixed = true`),
    /// which gates off the per-inner-solve ML refresh in
    /// `GamWorkingModel::update_with_curvature` (its guard is
    /// `negbin_theta_is_estimated()`).
    ///
    /// Rationale: with θ estimated, the inner solver re-derives θ from each
    /// outer iterate's *warm-start* η, so θ — and hence the NB working response,
    /// deviance and penalty-logdet that feed the REML criterion — drifts every
    /// outer evaluation. The outer optimizer then chases a moving target and the
    /// projected-gradient convergence test never trips, grinding the loop to
    /// `max_iter` (the #1082 negative-binomial tensor timeout). Holding θ fixed
    /// across the λ-search makes the REML objective `F(ρ) = REML(ρ, θ_frozen)` a
    /// genuine stationary function of ρ, so the loop converges in a handful of
    /// iterations — and θ is still ML-refreshed at the single final, reported fit
    /// (the `refine_dispersion_at_converged_eta = true` accept-fit), exactly as
    /// the function-level docs require ("estimate the scale at the converged fit,
    /// not inside the λ search; mgcv likewise"). No-op for non-NB families and
    /// for an already user-fixed θ.
    #[inline]
    pub fn with_negbin_theta_frozen_for_search(mut self, theta: f64) -> Self {
        if let ResponseFamily::NegativeBinomial {
            theta: family_theta,
            theta_fixed,
        } = &mut self.spec.response
        {
            *family_theta = theta;
            *theta_fixed = true;
            self.scale = LikelihoodScaleMetadata::FixedNegBinTheta { theta };
        }
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

    /// Solver-only stabilization: the ridge `δI` stabilizes the inner linear
    /// solve (it bounds the Newton step `(H+δI)⁻¹∇`) but is **excluded** from
    /// the REML/LAML objective — no `½·δ·‖β‖²` quadratic-penalty term, no
    /// `δ`-shift of the penalty log-determinant, no `δ`-shift of the Laplace
    /// Hessian. Use this when a numerical floor is needed purely to keep the
    /// linear algebra finite during screening and must NOT bias the
    /// smoothing-parameter selection or shrink identified coefficients off the
    /// MLE. With every `include_*` false the optimized objective equals the
    /// true penalized REML criterion, so the value surface and its analytic
    /// gradient describe the same objective (gam#747/#748).
    pub const fn solver_only() -> Self {
        Self {
            rho_independent: true,
            include_quadratic_penalty: false,
            include_penalty_logdet: false,
            include_laplacehessian: false,
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

#[cfg(test)]
mod ridge_policy_tests {
    use super::{RidgePassport, RidgePolicy, StabilizationKind, StabilizationLedger};

    #[test]
    fn solver_only_ridge_policy_stays_off_objective_accounting() {
        let passport = RidgePassport::scaled_identity(1.0e-4, RidgePolicy::solver_only());

        assert!(
            !passport.policy.include_quadratic_penalty,
            "solver-only ridge must not add a quadratic prior"
        );
        assert_eq!(
            passport.penalty_logdet_ridge(),
            0.0,
            "solver-only ridge must not shift the penalty logdet"
        );
        assert_eq!(
            passport.laplacehessianridge(),
            0.0,
            "solver-only ridge must not shift the Laplace Hessian"
        );

        let ledger = StabilizationLedger::from_passport(passport);
        assert!(
            matches!(
                ledger.kind,
                StabilizationKind::NumericalPerturbation {
                    backward_error_bound: None
                }
            ),
            "solver-only ridge is a numerical perturbation, not an explicit prior"
        );
        assert_eq!(
            ledger.quadratic_delta(),
            0.0,
            "solver-only ridge must not contribute to the optimized objective"
        );
        assert_eq!(
            ledger.laplace_hessian_delta(),
            0.0,
            "solver-only ridge must not contribute to REML curvature accounting"
        );
        assert_eq!(
            ledger.penalty_logdet_delta(),
            0.0,
            "solver-only ridge must not contribute to determinant accounting"
        );
    }
}

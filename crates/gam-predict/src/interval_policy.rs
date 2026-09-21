//! Shared interval / posterior-mean policy engine.
//!
//! Every `PredictableModel` predictor computes its own linear predictor(s),
//! response transform, and standard errors using domain-specific math. What
//! they all share — and previously re-implemented inline — is the *policy*
//! layer that turns those quantities into confidence intervals and result
//! structs:
//!
//!   1. the central multiplier `z = F⁻¹(½ + ½·level)` of the fit's
//!      [`IntervalReference`] (normal, or Student-t for an estimated scale),
//!   2. the η-scale interval `η ± z·SE(η)`,
//!   3. the response-scale interval, the image of an index interval
//!      `u ± z·SE(u)` under the monotone response map `h` (the inverse link of
//!      `η`, or a two-block family's own response index), intersected with the
//!      map's feasible argument set, and
//!   4. assembling [`PredictUncertaintyResult`] / [`PredictPosteriorMeanResult`].
//!
//! Every mean band is such an image, so it lies in the response support by
//! construction and nothing clamps it (#3140). Only the observation band, a
//! symmetric predictive set, is clamped to the support.
//!
//! Centralizing this here means the confidence-level convention, quantile
//! routine and covariance-mode plumbing live in one place. A predictor defines
//! only its linear state + response transform and delegates the policy to the
//! helpers below, so interval/posterior-mean behaviour cannot drift between
//! families.

use crate::{
    InferenceCovarianceMode, IntervalReference, PointCovarianceProvenance, PosteriorMeanOptions, PredictInput,
    PredictPosteriorMeanResult, PredictResult, PredictUncertaintyOptions, PredictUncertaintyResult,
    PredictionWithSE, family_observation_band, refuse_declined_covariance,
};
use gam_problem::EstimationError;
use gam_solve::model_types::UnifiedFitResult;
use gam_spec::{EtaFeasibility, InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::Array1;

/// Closed response-scale support `[lo, hi]` used to clamp a symmetric
/// observation (predictive) band. `None` means the response is unbounded and
/// must not be clamped. Mean bands are never clamped: they are images of an
/// index interval under the response map and lie in the support by
/// construction.
///
/// This is the predict-side policy mirror of
/// [`ResponseFamily::mean_clamp_bounds`]: families expose their natural mean
/// support and the engine applies it uniformly. Predictors whose response is a
/// probability (survival tail, binomial location-scale) report the closed
/// `[0, 1]` bounds directly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResponseBounds(Option<(f64, f64)>);

impl ResponseBounds {
    /// Unbounded response — endpoints are passed through unclamped.
    pub(crate) const UNBOUNDED: Self = Self(None);
    /// Closed unit interval `[0, 1]` (probabilities, survival tails).
    pub(crate) const UNIT_PROBABILITY: Self = Self(Some((0.0, 1.0)));

    /// Explicit closed bounds.
    pub fn closed(lo: f64, hi: f64) -> Self {
        Self(Some((lo, hi)))
    }

    /// The response-support clamp for a [`ResponseFamily`], matching
    /// [`ResponseFamily::mean_clamp_bounds`].
    pub(crate) fn for_family(response: &ResponseFamily) -> Self {
        Self(response.mean_clamp_bounds())
    }

    /// Clamp a single value into the support, leaving it untouched when the
    /// response is unbounded.
    #[inline]
    pub fn clamp_value(&self, v: f64) -> f64 {
        match self.0 {
            Some((lo, hi)) => v.clamp(lo, hi),
            None => v,
        }
    }

    /// Clamp every entry of `values` in place into the support.
    pub(crate) fn clamp_in_place(&self, values: &mut Array1<f64>) {
        if let Some((lo, hi)) = self.0 {
            values.mapv_inplace(|v| v.clamp(lo, hi));
        }
    }
}

/// The symmetric interval `center ± z·se`, returned as `(lower, upper)`.
#[inline]
pub(crate) fn symmetric_interval(
    center: &Array1<f64>,
    se: &Array1<f64>,
    z: f64,
) -> (Array1<f64>, Array1<f64>) {
    let half_width = se.mapv(|s| z * s);
    (center - &half_width, center + &half_width)
}

/// The feasible argument set of a response map, with the map's limit at the
/// set's finite boundary.
///
/// A cell whose mean map is defined on a half-line only (`√`, `1/η`, `η^{-1/2}`,
/// identity on a positive family, `exp(η) < 1` for a log-link probability; see
/// [`LikelihoodSpec::eta_feasibility`]) has an η interval that may cross the
/// origin although its point is feasible. The mean band is the image of the
/// feasible part of that interval, so the infeasible part is cut at the
/// boundary and the boundary endpoint takes the map's limit there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EtaDomain {
    /// The map is defined on every finite argument.
    Unrestricted,
    /// The map is defined on the open half-line `feasibility` (never
    /// [`EtaFeasibility::Unrestricted`]); `boundary_mean` is its limit as the
    /// argument tends to the half-line's finite end `0` from inside.
    HalfLine {
        feasibility: EtaFeasibility,
        boundary_mean: f64,
    },
}

impl EtaDomain {
    /// The feasible argument set of `spec`'s inverse link over its family.
    ///
    /// Every restricted cell changes admissibility at `η = 0`, and the limit
    /// of its inverse link there is read off the link itself: `0` for the
    /// identity and `η²`, `1` for `exp`, and `+∞` for `1/η` and `η^{-1/2}`.
    pub(crate) fn of_spec(spec: &LikelihoodSpec) -> Result<Self, EstimationError> {
        let feasibility = spec.eta_feasibility();
        if feasibility == EtaFeasibility::Unrestricted {
            return Ok(Self::Unrestricted);
        }
        let boundary_mean = match &spec.link {
            InverseLink::Standard(StandardLink::Identity | StandardLink::Sqrt) => 0.0,
            InverseLink::Standard(StandardLink::Log) => 1.0,
            InverseLink::Standard(StandardLink::Inverse | StandardLink::InverseSquared) => {
                f64::INFINITY
            }
            other => {
                return Err(EstimationError::InvalidInput(format!(
                    "response-scale interval: the {} cell restricts η to {feasibility:?} but its \
                     link {other:?} has no boundary limit on record",
                    spec.response.name()
                )));
            }
        };
        Ok(Self::HalfLine {
            feasibility,
            boundary_mean,
        })
    }
}

/// Response-scale interval: the image of the index interval
/// `[lower, upper]` under a monotone response map, restricted to the map's
/// feasible argument set `domain`.
///
/// `response_map` is the predictor's inverse-link / response transform, and
/// every one routed here is monotone. The base inverse links are CDF-shaped
/// (logit, probit, cloglog, SAS, beta-logistic, mixtures) or monotone on their
/// half-line (`exp`, `√`, `1/η`). The Bernoulli marginal-slope mean is `Φ(η)`.
/// A learnable link wiggle is a monotone warp that already sits inside `η`.
/// Survival tails decrease in their index. The image of `[lower, upper]` is
/// therefore exactly the span of the two endpoint images, so the endpoints are
/// transformed and ordered, and nothing is scanned (SPEC rule 18, #2902). The
/// ordering also handles a decreasing map. The image lies in the map's range,
/// so it lies in the response support and nothing clamps it (#3140).
///
/// On a half-line domain an endpoint at or past the boundary is cut there and
/// takes the map's boundary limit, which may be infinite (the image of
/// `(0, u]` under `1/η` is `[1/u, ∞)`). An interval entirely outside the
/// domain has an infeasible point and is a typed error.
///
/// Every endpoint image evaluated inside the domain must be finite. A
/// non-finite one is a typed prediction failure: changing that row to a
/// delta-method interval would silently substitute a different uncertainty
/// estimand. Degenerate all-zero count responses are rejected at the
/// family-validation boundary before a fit is minted (#2255).
pub(crate) fn transform_eta_interval<F>(
    eta_lower: &Array1<f64>,
    eta_upper: &Array1<f64>,
    domain: EtaDomain,
    response_map: F,
) -> Result<(Array1<f64>, Array1<f64>), EstimationError>
where
    F: Fn(&Array1<f64>) -> Result<Array1<f64>, EstimationError>,
{
    let n = eta_lower.len();
    // Each row's evaluation points, with the side cut at the boundary (if any).
    // A half-line has one finite end, so at most one side of a row is cut, and
    // that side is evaluated at the row's other, feasible endpoint and then
    // replaced by the boundary limit.
    let mut lower_arg = eta_lower.clone();
    let mut upper_arg = eta_upper.clone();
    let mut lower_at_boundary = vec![false; n];
    let mut upper_at_boundary = vec![false; n];
    if let EtaDomain::HalfLine { feasibility, .. } = domain {
        let (a, b) = feasibility.interval();
        for i in 0..n {
            let (lo, hi) = (eta_lower[i], eta_upper[i]);
            if !(lo <= hi && hi > a && lo < b) {
                return Err(EstimationError::InvalidInput(format!(
                    "response-scale interval: row {i} has the index interval [{lo}, {hi}], \
                     which does not meet the feasible set ({a}, {b}) of the response map"
                )));
            }
            if lo <= a {
                lower_at_boundary[i] = true;
                lower_arg[i] = hi;
            }
            if hi >= b {
                upper_at_boundary[i] = true;
                upper_arg[i] = lo;
            }
        }
    }
    let boundary_mean = match domain {
        EtaDomain::HalfLine { boundary_mean, .. } => boundary_mean,
        EtaDomain::Unrestricted => f64::NAN,
    };
    let at_lower = response_map(&lower_arg)?;
    let at_upper = response_map(&upper_arg)?;
    let mut mean_lower = Array1::<f64>::zeros(n);
    let mut mean_upper = Array1::<f64>::zeros(n);
    for i in 0..n {
        // `f64::min`/`max` return the non-NaN argument, so a single `+inf`/NaN
        // endpoint would otherwise slip through as a finite-but-meaningless bound.
        if !(at_lower[i].is_finite() && at_upper[i].is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "response-scale interval transform produced a non-finite value at row {i}"
            )));
        }
        let first = if lower_at_boundary[i] {
            boundary_mean
        } else {
            at_lower[i]
        };
        let second = if upper_at_boundary[i] {
            boundary_mean
        } else {
            at_upper[i]
        };
        mean_lower[i] = first.min(second);
        mean_upper[i] = first.max(second);
    }
    Ok((mean_lower, mean_upper))
}

/// A two-block family's response index `u` and its posterior SD, the argument
/// its mean band is formed on.
///
/// A two-block response `μ = h(u(η₁, η₂))` is a monotone map of one scalar
/// index `u` built from both linear predictors (survival: the standardized
/// threshold `q0 = −η_t·e^{−η_σ}`, with `S = 1 − F(q0)`). Its band is the image
/// of `u ± z·SD(u)` under `h`, which lies in `h`'s range; no single linear
/// predictor's interval maps onto the response.
#[derive(Clone, Debug)]
pub struct ResponseIndex {
    /// The index `u` per row.
    pub index: Array1<f64>,
    /// Posterior SD of `u` per row, under the pass's covariance.
    pub index_se: Array1<f64>,
}

/// How a predictor maps its interval onto the response scale.
///
/// Every variant forms the band as the image of an index interval under the
/// response map, so the band lies in the support by construction.
pub(crate) enum MeanBoundMethod<'a> {
    /// The image of `η ± z·SE(η)` under the supplied monotone response map,
    /// restricted to `domain`. Non-finite interior images are errors; this
    /// path never substitutes a delta-method interval.
    TransformEta {
        domain: EtaDomain,
        response_map: &'a (dyn Fn(&Array1<f64>) -> Result<Array1<f64>, EstimationError> + 'a),
    },
    /// The image of the response index interval `u ± z·SD(u)` under the
    /// supplied monotone response map of `u`.
    TransformIndex {
        index: &'a ResponseIndex,
        response_map: &'a (dyn Fn(&Array1<f64>) -> Result<Array1<f64>, EstimationError> + 'a),
    },
    /// The response equals the linear predictor (identity link); the response
    /// bounds are exactly the η bounds.
    IdentityEta,
}

/// Compute response-scale `(mean_lower, mean_upper)` for the requested method.
pub(crate) fn mean_bounds(
    eta_lower: &Array1<f64>,
    eta_upper: &Array1<f64>,
    z: f64,
    method: MeanBoundMethod<'_>,
) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
    match method {
        MeanBoundMethod::TransformEta {
            domain,
            response_map,
        } => transform_eta_interval(eta_lower, eta_upper, domain, response_map),
        MeanBoundMethod::TransformIndex {
            index,
            response_map,
        } => {
            let (index_lower, index_upper) = symmetric_interval(&index.index, &index.index_se, z);
            transform_eta_interval(
                &index_lower,
                &index_upper,
                EtaDomain::Unrestricted,
                response_map,
            )
        }
        MeanBoundMethod::IdentityEta => Ok((eta_lower.clone(), eta_upper.clone())),
    }
}

/// Observation (prediction) interval construction selected by a predictor.
/// Keeping the analytic override as its own variant matters for families such
/// as Royston–Parmar: their fresh-response law has an exact discrete predictive
/// set but no additive response-noise standard deviation.
pub(crate) enum ObservationInterval<'a> {
    /// Symmetric `μ ± z·√(SE(μ̂)² + σ²)` band from a per-row
    /// response-scale noise standard deviation.
    Symmetric {
        noise_sd: &'a Array1<f64>,
        /// Response-support clamp; `ResponseBounds::UNBOUNDED` for real-line
        /// responses.
        bounds: ResponseBounds,
    },
    /// Precomputed family-aware predictive endpoints (for example a skewed
    /// equal-tailed interval or the discrete Bernoulli set on `{0, 1}`).
    Override {
        lower: Array1<f64>,
        upper: Array1<f64>,
    },
}

/// Static metadata threaded into every [`PredictUncertaintyResult`].
///
/// This field is pure provenance: the exact covariance definition consumed by
/// the uncertainty calculation. It is copied verbatim into the result so the
/// engine, not each predictor, owns the struct shape.
pub(crate) struct UncertaintyProvenance {
    pub covariance_source: InferenceCovarianceMode,
}

/// The symmetric predictive (observation) band `μ ± z·√(SE(μ̂)² + σ²)`, clamped
/// to the response support.
///
/// A prediction interval covers a *future* response `Y = μ + ε` at the query
/// point. The point `μ̂` is itself estimated (`Var(μ̂) = SE(μ̂)²`) and the
/// observation noise has `Var(Y|μ) = σ²`; the two are independent, so the
/// predictive variance is their sum and the half-width is `z·√(SE(μ̂)² + σ²)`,
/// **not** `z·σ` — dropping the estimation term under-covers wherever the fit is
/// uncertain. The support clamp is not cosmetic either: a symmetric band on a
/// bounded or half-bounded response otherwise reports impossible values, such as
/// a count band going negative.
///
/// Both prediction-interval entry points call this, so they cannot disagree on
/// the same fit. They previously held that agreement — and this
/// `√(mean_se² + obsvar)` convention, shared with `family_observation_band` — by
/// asserting it in a comment in each copy rather than by sharing a call.
pub(crate) fn symmetric_predictive_band(
    mean: &Array1<f64>,
    mean_standard_error: &Array1<f64>,
    noise_sd: &Array1<f64>,
    z: f64,
    bounds: &ResponseBounds,
) -> (Array1<f64>, Array1<f64>) {
    let predictive_se = Array1::from_iter(
        mean_standard_error
            .iter()
            .zip(noise_sd.iter())
            .map(|(&mse, &sd)| (mse * mse + sd * sd).max(0.0).sqrt()),
    );
    let half = predictive_se.mapv(|s| z * s);
    let mut lower = mean - &half;
    let mut upper = mean + &half;
    bounds.clamp_in_place(&mut lower);
    bounds.clamp_in_place(&mut upper);
    (lower, upper)
}

/// Assemble a [`PredictUncertaintyResult`] from a predictor's already-computed
/// linear-predictor / response state.
///
/// This is the shared tail every `predict_full_uncertainty` impl used to inline:
/// validate the confidence level, form the η interval with the multiplier of
/// `reference`, map it onto the response scale via `method`, optionally attach
/// an observation interval, and populate the result struct. Predictors supply
/// only the family-specific quantities (`eta`, `mean`, the two standard errors)
/// plus the policy choice `method`; the engine owns everything else so interval
/// construction cannot drift between families.
pub(crate) fn assemble_uncertainty_result(
    confidence_level: f64,
    reference: IntervalReference,
    eta: Array1<f64>,
    mean: Array1<f64>,
    eta_standard_error: Array1<f64>,
    mean_standard_error: Array1<f64>,
    method: MeanBoundMethod<'_>,
    observation: Option<ObservationInterval<'_>>,
    provenance: UncertaintyProvenance,
) -> Result<PredictUncertaintyResult, EstimationError> {
    let z = reference.central_multiplier(confidence_level)?;
    let (eta_lower, eta_upper) = symmetric_interval(&eta, &eta_standard_error, z);
    let (mean_lower, mean_upper) = mean_bounds(&eta_lower, &eta_upper, z, method)?;
    let (observation_lower, observation_upper) = match observation {
        // A skew-aware predictor (dispersion location-scale) supplies its
        // equal-tailed band directly; use it verbatim (already support-clamped).
        Some(ObservationInterval::Override { lower, upper }) => (Some(lower), Some(upper)),
        Some(ObservationInterval::Symmetric { noise_sd, bounds }) => {
            let (lower, upper) =
                symmetric_predictive_band(&mean, &mean_standard_error, noise_sd, z, &bounds);
            (Some(lower), Some(upper))
        }
        None => (None, None),
    };
    Ok(PredictUncertaintyResult {
        eta,
        mean,
        eta_standard_error,
        mean_standard_error,
        eta_lower,
        eta_upper,
        mean_lower,
        mean_upper,
        observation_lower,
        observation_upper,
        covariance_source: provenance.covariance_source,
    })
}

/// Attach response-scale confidence bounds to a [`PredictPosteriorMeanResult`].
///
/// When a confidence level is supplied, this is the shared tail every
/// `predict_posterior_mean` impl used to inline: validate the level, form the η
/// interval, map it onto the response scale via `method`, and set
/// `mean_lower` / `mean_upper`. When no level is supplied the bounds are left
/// `None`. `eta` / `eta_se` are taken from `result`.
pub(crate) fn assemble_posterior_mean_bounds(
    result: &mut PredictPosteriorMeanResult,
    confidence_level: Option<f64>,
    reference: IntervalReference,
    method: MeanBoundMethod<'_>,
) -> Result<(), EstimationError> {
    let Some(level) = confidence_level else {
        return Ok(());
    };
    let z = reference.central_multiplier(level)?;
    let (eta_lower, eta_upper) = symmetric_interval(&result.eta, &result.eta_standard_error, z);
    let (mean_lower, mean_upper) = mean_bounds(&eta_lower, &eta_upper, z, method)?;
    result.mean_lower = Some(mean_lower);
    result.mean_upper = Some(mean_upper);
    Ok(())
}

/// Which of the two interval-producing pipelines a [`PredictionTransform`] is
/// being driven through.
///
/// A few families compute their response point and standard errors differently
/// in the two passes (notably the threshold-scale probability families, whose
/// posterior mean is a bivariate Gauss–Hermite integral rather than the plug-in
/// delta evaluation used for full uncertainty). The pass is threaded into
/// [`PredictionTransform::linear_state`] so the family can branch its numerics
/// while the assembly stays unified. The interval policy does not depend on the
/// pass: every mean band is the image of an index interval under the family's
/// monotone response map (#3140), so
/// [`PredictionTransform::response_jacobian_rows`] takes no pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictPass {
    /// `predict_full_uncertainty`: η/μ point + η- and mean-scale SEs, with the
    /// η interval reported on the response scale per the family's policy.
    FullUncertainty,
    /// `predict_posterior_mean`: coefficient-uncertainty-integrated response
    /// mean with optional response-scale confidence bounds.
    PosteriorMean,
}

/// How a transform forms the *response-scale* confidence interval from the
/// η-scale state. This is the per-family policy split the predictors used to
/// inline directly into `assemble_uncertainty_result` / `mean_bounds`; a
/// [`PredictionTransform`] now declares it once and the generic drivers thread
/// it through both the full-uncertainty and posterior-mean pipelines.
///
/// Every policy forms the band as the image of an index interval under the
/// monotone [`PredictionTransform::response`], so it lies in the response
/// support by construction and nothing clamps it (#3140).
pub enum ResponseInterval {
    /// The image of `η ± z·SE(η)` under the monotone
    /// [`PredictionTransform::response`], restricted to the map's feasible
    /// argument set. Used by families whose response is an inverse-link image
    /// of one linear predictor (standard, link wiggle, Bernoulli
    /// marginal-slope, binomial and dispersion location-scale, whose `η` is
    /// the inverse link's argument).
    TransformEta(EtaDomain),
    /// Identity link: the response equals the linear predictor, so the response
    /// interval is exactly the η interval (Gaussian location-scale, PIT).
    IdentityEta,
    /// The image of the response index interval `u ± z·SD(u)` under the
    /// monotone [`PredictionTransform::response`] of `u`, with `u` carried in
    /// [`LinearState::response_index`]. Used by two-block families whose
    /// response is a map of an index built from both linear predictors
    /// (survival tail).
    TransformIndex,
}

/// The η-scale state a [`PredictionTransform`] produces for one prediction
/// batch: the linear predictor, its response-scale image, and (when a
/// covariance is available) the η- and mean-scale standard errors.
///
/// Each predictor computes these via whatever bespoke gradient backend its
/// parameterisation requires (dense matvec, link-wiggle chain rule, projected
/// two-block covariance, bivariate GHQ). The generic drivers below consume the
/// finished arrays and own the *policy* layer — interval construction and
/// result assembly — so that layer cannot drift between families.
pub struct LinearState {
    /// Linear predictor η.
    pub eta: Array1<f64>,
    /// Response-scale prediction μ = T(η).
    pub mean: Array1<f64>,
    /// Standard error of η (delta-method base). `None` when no covariance.
    pub eta_se: Option<Array1<f64>>,
    /// Standard error of μ (delta-method, response scale). `None` when no
    /// covariance.
    pub mean_se: Option<Array1<f64>>,
    /// Exact covariance definition consumed by `eta_se` / `mean_se`. The
    /// full-uncertainty driver propagates it into the public result. Point-state
    /// and posterior-mean construction use conditional covariance.
    pub covariance_source: InferenceCovarianceMode,
    /// The response index and its SD under the same covariance as `eta_se`,
    /// for a [`ResponseInterval::TransformIndex`] transform; `None` for every
    /// other policy and whenever no covariance is available.
    pub response_index: Option<ResponseIndex>,
}

/// Family-specific supplier for the shared predict pipeline.
///
/// A predictor implements this trait to describe *only* the parts that differ
/// between families:
///
///   * [`linear_state`](PredictionTransform::linear_state) — the η-scale
///     predictor, its response image, and the standard errors, computed with
///     the predictor's own gradient backend (issue #422 keeps these bespoke;
///     they are genuine numerics, not boilerplate);
///   * [`response`](PredictionTransform::response) — the monotone response
///     map, whose image of the index interval is the mean band;
///   * [`response_jacobian_rows`](PredictionTransform::response_jacobian_rows) —
///     which [`ResponseInterval`] policy maps the index interval onto the
///     response scale;
///   * [`bounds`](PredictionTransform::bounds) — the response-scale support
///     the symmetric observation band is clamped to;
///   * [`observation_noise`](PredictionTransform::observation_noise) — the
///     optional response-scale observation-noise σ.
///
/// Everything else — confidence-level validation, η/mean interval construction,
/// observation intervals, and result-struct assembly — lives
/// in the generic drivers `predict_full_uncertainty_generic` and
/// `predict_posterior_mean_generic`, so the pipeline is one source of truth.
pub trait PredictionTransform {
    /// The fit-free point state: η, μ, and the covariance-derived standard
    /// errors (`None` when no predictor covariance is available). This is the
    /// state behind the point-prediction drivers
    /// `predict_plugin_response_generic` and `predict_with_uncertainty_generic`,
    /// and the default source for the full-uncertainty pass of
    /// [`linear_state`](PredictionTransform::linear_state).
    fn point_state(&self, input: &PredictInput) -> Result<LinearState, EstimationError>;

    /// Compute η, μ, and the standard errors for the requested `pass`. `fit`
    /// carries the posterior covariance / penalized Hessian some predictors
    /// need, and `covariance_mode` selects which covariance (conditional vs.
    /// smoothing-corrected) the full-uncertainty SEs are built from. The
    /// returned [`LinearState::covariance_source`] records which covariance was
    /// actually consumed.
    ///
    /// The default services the full-uncertainty pass from the fit-free
    /// [`point_state`](PredictionTransform::point_state); predictors whose
    /// posterior-mean (or fit-backed full-uncertainty) numerics differ override
    /// this and branch on `pass`. The posterior-mean pass always integrates the
    /// conditional posterior, so `covariance_mode` is only consulted for the
    /// full-uncertainty pass.
    fn linear_state(
        &self,
        input: &PredictInput,
        _: &UnifiedFitResult,
        pass: PredictPass,
        _: InferenceCovarianceMode,
    ) -> Result<LinearState, EstimationError> {
        match pass {
            PredictPass::FullUncertainty => self.point_state(input),
            PredictPass::PosteriorMean => Err(EstimationError::InvalidInput(
                "this transform does not implement the posterior-mean pass".to_string(),
            )),
        }
    }

    /// The monotone response map whose image of the index interval is the
    /// mean band: `μ = T(η)` for [`ResponseInterval::TransformEta`], and the
    /// map of the response index `u` for [`ResponseInterval::TransformIndex`].
    fn response(&self, argument: &Array1<f64>) -> Result<Array1<f64>, EstimationError>;

    /// Which [`ResponseInterval`] policy maps the index interval onto the
    /// response scale. It is the same on both passes: the band is the image of
    /// an index interval under [`response`](PredictionTransform::response).
    /// An error when the transform's link has no recorded feasible argument
    /// set.
    fn response_jacobian_rows(&self) -> Result<ResponseInterval, EstimationError>;

    /// Response-scale support `[lo, hi]` the symmetric observation band is
    /// clamped to. The mean band never consults it: it is an image of the
    /// response map and lies in the support by construction.
    fn bounds(&self) -> ResponseBounds;

    /// The response distribution family. Used by the generic posterior-mean
    /// driver to build the per-family observation (prediction) interval via
    /// `family_observation_band`; `RoystonParmar` yields the discrete
    /// Bernoulli predictive set for the horizon indicator `1{T > t}`.
    fn response_family(&self) -> ResponseFamily;

    /// Optional response-scale observation-noise σ for the requested batch.
    /// `None` (the default) for families without an observation-scale noise
    /// term. Only consulted by the full-uncertainty driver and only when the
    /// caller requested observation intervals.
    fn observation_noise(&self, _: &PredictInput) -> Result<Option<Array1<f64>>, EstimationError> {
        Ok(None)
    }

    /// Optional skew-aware **equal-tailed** observation band, built per row.
    ///
    /// The default (`None`) leaves the observation interval to the generic
    /// symmetric `μ ± z·√(SE(μ̂)² + σ²)` construction (correct for symmetric
    /// response families and the Gaussian location-scale identity link). A
    /// heteroscedastic *dispersion* location-scale predictor whose response is
    /// skewed (Gamma/NB/Beta/Tweedie + `noise_formula`) overrides this to return
    /// equal-tailed quantiles of its per-row moment-matched predictive — the
    /// two-block sibling of `family_observation_band` (#817/#1193/#1194). When
    /// `Some`, the returned `(lower, upper)` replaces the symmetric band in both
    /// the full-uncertainty and posterior-mean drivers; when `None`, the symmetric
    /// path is used.
    ///
    /// `eta` / `eta_se` are the per-row linear predictor and its standard error
    /// under the pass's covariance; the band reads its predictive moments off that
    /// η law itself, so its mean, variance and complement are one law's (#3140).
    /// `z_lower` / `z_upper` are the per-row tail multipliers (the same masses the
    /// symmetric band would target).
    fn observation_band(
        &self,
        input: &PredictInput,
        eta: &Array1<f64>,
        eta_se: &Array1<f64>,
        z_lower: &Array1<f64>,
        z_upper: &Array1<f64>,
    ) -> Result<Option<(Array1<f64>, Array1<f64>)>, EstimationError> {
        // Default: no skew-aware band. The generic symmetric construction is
        // used instead. Validate the per-row inputs the driver hands every
        // transform so an overriding impl and this default agree on shape:
        // one entry per row of the prediction design.
        assert_eq!(eta.len(), input.design.nrows());
        assert_eq!(eta.len(), eta_se.len());
        assert_eq!(eta.len(), z_lower.len());
        assert_eq!(eta.len(), z_upper.len());
        Ok(None)
    }

    /// Optional response-scale posterior SD `√Var[T(η)]`, `η ~ N(eta, eta_se²)`
    /// per row, from the same Gaussian η integral as the posterior-mean point.
    ///
    /// The posterior-mean driver's smoothing-corrected arm reports this, built
    /// on the corrected η SE, as the response-scale SE whenever it is `Some`,
    /// instead of the full-uncertainty pass's delta-method `|dT/dη̂|·SE(η)` —
    /// which collapses to zero wherever the inverse link saturates although the
    /// posterior of μ stays wide. The default `None` keeps the full-uncertainty
    /// pass's `mean_se`.
    fn posterior_response_sd(
        &self,
        eta: &Array1<f64>,
        eta_se: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        // One SE per row, as every overriding impl consumes them.
        assert_eq!(eta.len(), eta_se.len());
        Ok(None)
    }
}

/// Build the `MeanBoundMethod` selected by a transform's [`ResponseInterval`]
/// policy, borrowing the response closure and the response index as needed.
fn mean_bound_method_for<'a>(
    policy: &ResponseInterval,
    response_map: &'a (dyn Fn(&Array1<f64>) -> Result<Array1<f64>, EstimationError> + 'a),
    response_index: Option<&'a ResponseIndex>,
) -> Result<MeanBoundMethod<'a>, EstimationError> {
    Ok(match policy {
        ResponseInterval::TransformEta(domain) => MeanBoundMethod::TransformEta {
            domain: *domain,
            response_map,
        },
        ResponseInterval::IdentityEta => MeanBoundMethod::IdentityEta,
        ResponseInterval::TransformIndex => MeanBoundMethod::TransformIndex {
            index: response_index.ok_or_else(|| {
                EstimationError::InvalidInput(
                    "response-scale interval: the transform bands its response index but \
                     supplied none for this pass"
                        .to_string(),
                )
            })?,
            response_map,
        },
    })
}

/// Refuse a measure-jet extrapolation variance a generic driver cannot carry.
///
/// Only `predict_gamwith_uncertainty` adds a per-row η variance to its band
/// after the covariance's own `Var(η)`. The generic drivers take the transform's
/// η and response standard errors as they come, so a supplied extrapolation
/// variance would silently drop out of the band instead of widening it.
fn refuse_unfused_extrapolation_variance(
    extrapolation_variance: Option<&Array1<f64>>,
) -> Result<(), EstimationError> {
    if extrapolation_variance.is_some() {
        return Err(EstimationError::InvalidInput(
            "this predictor's interval driver cannot add a measure-jet extrapolation variance \
             to its band; only the standard prediction engine fuses it"
                .to_string(),
        ));
    }
    Ok(())
}

/// The single full-uncertainty driver. Runs the predict pipeline once for any
/// [`PredictionTransform`]: compute the η-scale state, require its standard
/// errors, attach the optional observation interval, and assemble the result
/// through `assemble_uncertainty_result`.
pub(crate) fn predict_full_uncertainty_generic<T: PredictionTransform>(
    transform: &T,
    input: &PredictInput,
    fit: &UnifiedFitResult,
    options: &PredictUncertaintyOptions,
) -> Result<PredictUncertaintyResult, EstimationError> {
    refuse_unfused_extrapolation_variance(options.extrapolation_variance.as_ref())?;
    let response_family = transform.response_family();
    let mut state = transform.linear_state(
        input,
        fit,
        PredictPass::FullUncertainty,
        options.covariance_mode,
    )?;
    // Royston–Parmar's reported point is the survival probability S(t).  The
    // default estimand is its conditional-posterior mean E[S(t) | D], not the
    // plug-in S(t; β̂).  Keep the requested covariance mode for the attached
    // uncertainty, but source the point itself from the transform's conditional
    // posterior-mean pass so full prediction and conformal calibration target
    // the same marginal Bernoulli law as posterior-mean prediction.
    if matches!(&response_family, ResponseFamily::RoystonParmar) {
        let posterior_state = transform.linear_state(
            input,
            fit,
            PredictPass::PosteriorMean,
            InferenceCovarianceMode::Conditional,
        )?;
        state.mean = posterior_state.mean;
    }
    let covariance_source = state.covariance_source;
    let eta_se = state.eta_se.ok_or_else(|| {
        EstimationError::InvalidInput(
            "full uncertainty requires covariance (eta_se unavailable)".to_string(),
        )
    })?;
    let mean_se = state.mean_se.ok_or_else(|| {
        EstimationError::InvalidInput(
            "full uncertainty requires covariance (mean_se unavailable)".to_string(),
        )
    })?;
    let policy = transform.response_jacobian_rows()?;
    let response_index = state.response_index;
    let response_map = move |argument: &Array1<f64>| transform.response(argument);
    let reference = IntervalReference::of_fit(fit)?;
    let observation = if options.includeobservation_interval {
        transform.observation_noise(input)?
    } else {
        None
    };
    // A skew-aware predictor (the dispersion location-scale families) builds an
    // equal-tailed band per row from its moment-matched predictive; when present
    // it replaces the symmetric `μ ± z·σ` construction below (#817/#1193/#1194).
    let mut override_band = if options.includeobservation_interval {
        let z = reference.central_multiplier(options.confidence_level)?;
        let z_row = Array1::from_elem(state.mean.len(), z);
        transform.observation_band(input, &state.eta, &eta_se, &z_row, &z_row)?
    } else {
        None
    };
    // A single-distribution transform can have no separate per-row noise
    // channel while still possessing a family-defined predictive law.  This is
    // exactly Royston–Parmar: the point is S(t), and a fresh response at the
    // requested horizon is the Bernoulli indicator 1{T > t}.  Mirror the
    // posterior-mean driver's dispatch instead of gating the family band on
    // `observation_noise` being present.
    if options.includeobservation_interval && override_band.is_none() && observation.is_none() {
        let z = reference.central_multiplier(options.confidence_level)?;
        let z_row = Array1::from_elem(state.mean.len(), z);
        // No transform reaching this driver has a Beta response (a link wiggle is
        // binomial-only, and the dispersion location-scale Beta builds its own band
        // above), so none needs the carried complement.
        let (lower, upper) = family_observation_band(
            &response_family,
            &state.mean,
            None,
            &mean_se,
            &z_row,
            &z_row,
            reference,
            fit,
            options.observation_prior_weights.as_ref(),
        )?;
        override_band = match (lower, upper) {
            (Some(lower), Some(upper)) => Some((lower, upper)),
            (None, None) => None,
            _ => {
                return Err(EstimationError::InvalidInput(
                    "family observation band returned only one endpoint".to_string(),
                ));
            }
        };
    }
    let observation_interval = match override_band {
        Some((lower, upper)) => Some(ObservationInterval::Override { lower, upper }),
        None => observation
            .as_ref()
            .map(|noise_sd| ObservationInterval::Symmetric {
                noise_sd,
                // The transform's response-scale support is exactly the clamp
                // the observation band must respect (unbounded for Gaussian
                // location-scale identity, `[0, 1]` for probability families).
                bounds: transform.bounds(),
            }),
    };
    assemble_uncertainty_result(
        options.confidence_level,
        reference,
        state.eta,
        state.mean,
        eta_se,
        mean_se,
        mean_bound_method_for(&policy, &response_map, response_index.as_ref())?,
        observation_interval,
        UncertaintyProvenance { covariance_source },
    )
}

/// The single posterior-mean driver. Runs the predict pipeline once for any
/// [`PredictionTransform`]: compute the η-scale state and attach response-scale
/// confidence bounds (when a level is supplied) through
/// `assemble_posterior_mean_bounds`.
pub(crate) fn predict_posterior_mean_generic<T: PredictionTransform>(
    transform: &T,
    input: &PredictInput,
    fit: &UnifiedFitResult,
    options: &PosteriorMeanOptions,
) -> Result<PredictPosteriorMeanResult, EstimationError> {
    refuse_unfused_extrapolation_variance(options.extrapolation_variance.as_ref())?;
    // POINT: the posterior-mean pass always integrates the *conditional*
    // posterior, so the reported point is invariant to the uncertainty request
    // (issue #398). `covariance_mode` only shapes the uncertainty attached below.
    let state = transform.linear_state(
        input,
        fit,
        PredictPass::PosteriorMean,
        InferenceCovarianceMode::Conditional,
    )?;
    let policy = transform.response_jacobian_rows()?;
    let has_covariance = state.eta_se.is_some();
    let cond_eta_se = state
        .eta_se
        .clone()
        .unwrap_or_else(|| Array1::zeros(state.eta.len()));
    // The response-scale SE must come from the transform itself: copying the
    // η-scale SE across a nonlinear inverse link omits the link Jacobian and is
    // dimensionally wrong (a logistic fit at η = 10 with SE(η) = 1 has response
    // SE ≈ 4.5e-5, not 1). Only the identity link may reuse SE(η) — there the
    // response IS the linear predictor. Every other transform must supply a
    // genuine response-scale `mean_se`; a missing one is a producer bug, not a
    // fallback opportunity. The no-covariance degrade (η SE also absent) keeps
    // its zero-SE point-only behaviour.
    let cond_mean_se = match (state.mean_se.clone(), &policy) {
        (Some(se), _) => se,
        (None, ResponseInterval::IdentityEta) => cond_eta_se.clone(),
        (None, _) if !has_covariance => cond_eta_se.clone(),
        (None, _) => {
            return Err(EstimationError::InvalidInput(
                "posterior-mean prediction: transform supplied an η-scale SE but no \
                 response-scale SE; a non-identity response interval requires the \
                 delta-method mean SE"
                    .to_string(),
            ));
        }
    };
    let cond_response_index = state.response_index;
    let mut result = PredictPosteriorMeanResult {
        eta: state.eta,
        eta_standard_error: cond_eta_se.clone(),
        mean: state.mean,
        mean_standard_error: None,
        mean_lower: None,
        mean_upper: None,
        observation_lower: None,
        observation_upper: None,
        point_covariance_source: InferenceCovarianceMode::Conditional,
        uncertainty_covariance_source: None,
        point_covariance_provenance: PointCovarianceProvenance::of_fit(fit),
    };

    let Some(level) = options.confidence_level else {
        return Ok(result);
    };
    // The conditional SE above is the point's own; as an interval it would be
    // one the fit declared inadmissible.
    refuse_declined_covariance(fit, "posterior-mean prediction interval")?;

    // UNCERTAINTY: the reported SE / credible bounds / observation band honour
    // `covariance_mode` (issues #811/#812: this path previously hardwired the
    // conditional covariance). The posterior-mean *point* above stays
    // conditional; only the uncertainty responds.
    //
    // `Conditional` keeps the posterior pass's own SE. `SmoothingCorrected`
    // re-derives the SEs from the full-uncertainty pass and errors if the fit
    // cannot supply that exact covariance definition.
    let (eta_se, mean_se, response_index) = match options.covariance_mode {
        InferenceCovarianceMode::Conditional => (cond_eta_se, cond_mean_se, cond_response_index),
        InferenceCovarianceMode::SmoothingCorrected => {
            let unc = transform.linear_state(
                input,
                fit,
                PredictPass::FullUncertainty,
                InferenceCovarianceMode::SmoothingCorrected,
            )?;
            let eta_se = unc.eta_se.ok_or_else(|| {
                EstimationError::InvalidInput(
                    "smoothing-corrected posterior-mean uncertainty requires eta SE".to_string(),
                )
            })?;
            // The full-uncertainty pass supplies the corrected η SE; the
            // response-scale SE is the posterior SD over that same η posterior
            // when the transform can integrate it, as the conditional
            // posterior-mean pass already reports.
            let mean_se = match transform.posterior_response_sd(&result.eta, &eta_se)? {
                Some(sd) => sd,
                None => unc.mean_se.ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "smoothing-corrected posterior-mean uncertainty requires mean SE"
                            .to_string(),
                    )
                })?,
            };
            if unc.covariance_source != InferenceCovarianceMode::SmoothingCorrected {
                return Err(EstimationError::InvalidInput(
                    "smoothing-corrected posterior-mean uncertainty resolved a conditional covariance"
                        .to_string(),
                ));
            }
            (eta_se, mean_se, unc.response_index)
        }
    };
    result.uncertainty_covariance_source = Some(options.covariance_mode);
    result.eta_standard_error = eta_se;
    // Record the response-scale SE used to build the credible band so the FFI/CLI
    // predict tables can report it as `std_error` instead of the link-scale σ_η
    // (#1536).
    result.mean_standard_error = Some(mean_se.clone());

    let reference = IntervalReference::of_fit(fit)?;
    {
        let response_map = |argument: &Array1<f64>| transform.response(argument);
        assemble_posterior_mean_bounds(
            &mut result,
            Some(level),
            reference,
            mean_bound_method_for(&policy, &response_map, response_index.as_ref())?,
        )?;
    }

    if options.include_observation_interval {
        let z = reference.central_multiplier(level)?;
        let z_row = Array1::from_elem(result.mean.len(), z);
        // A skew-aware dispersion location-scale predictor builds an equal-tailed
        // band per row from its moment-matched predictive (#817/#1193/#1194). When
        // present it replaces the symmetric `μ ± z·σ(x)` band below, so the
        // posterior-mean API matches the full-uncertainty API on the same skewed
        // fit instead of emitting a symmetric band.
        let skew_band = transform.observation_band(
            input,
            &result.eta,
            &result.eta_standard_error,
            &z_row,
            &z_row,
        )?;
        match (skew_band, transform.observation_noise(input)?) {
            (Some((lower, upper)), _) => {
                result.observation_lower = Some(lower);
                result.observation_upper = Some(upper);
            }
            // Heteroscedastic location-scale / dispersion predictors carry a
            // *per-row* observation noise σ(x) driven by their second linear
            // predictor (the scale / log-precision submodel). The fit-level
            // scalar dispersion read by `family_observation_band` collapses
            // that to a single constant, which is wrong for exactly the
            // families whose purpose is non-constant variance (Gaussian-LS,
            // and the NB/Gamma/Beta/Tweedie dispersion-LS models). Build the
            // predictive band from the per-row noise instead, using the same
            // `μ ± z·√(SE(μ̂)² + σ(x)²)` convention and response-support clamp
            // the full-uncertainty driver uses, so the two prediction-interval
            // APIs agree on the same fit. (The Gaussian location-scale band is
            // genuinely symmetric, so it keeps this arm.)
            (None, Some(noise_sd)) => {
                let bounds = transform.bounds();
                let (lower, upper) =
                    symmetric_predictive_band(&result.mean, &mean_se, &noise_sd, z, &bounds);
                result.observation_lower = Some(lower);
                result.observation_upper = Some(upper);
            }
            // Single-distribution families with no per-row noise submodel:
            // the fit-level scalar dispersion is the correct observation noise,
            // and `family_observation_band` additionally applies the skew-aware
            // Gamma predictive arm for the right-skewed positive families.
            (None, None) => {
                let (obs_lower, obs_upper) = family_observation_band(
                    &transform.response_family(),
                    &result.mean,
                    // No transform on this driver has a Beta response (see the
                    // full-uncertainty driver), so none needs the complement.
                    None,
                    &mean_se,
                    &z_row,
                    &z_row,
                    reference,
                    fit,
                    // A weighted Gaussian band carries `σ̂²/w_i` (#2077, #3957).
                    options.observation_prior_weights.as_ref(),
                )?;
                result.observation_lower = obs_lower;
                result.observation_upper = obs_upper;
            }
        }
    }

    Ok(result)
}

/// The single plug-in response driver: the transform's fit-free point state,
/// keeping only η and μ.
pub(crate) fn predict_plugin_response_generic<T: PredictionTransform>(
    transform: &T,
    input: &PredictInput,
) -> Result<PredictResult, EstimationError> {
    let state = transform.point_state(input)?;
    Ok(PredictResult {
        eta: state.eta,
        mean: state.mean,
    })
}

/// The single point-with-SE driver: the transform's fit-free point state,
/// carrying the (optional) η/mean standard errors through unchanged.
pub(crate) fn predict_with_uncertainty_generic<T: PredictionTransform>(
    transform: &T,
    input: &PredictInput,
) -> Result<PredictionWithSE, EstimationError> {
    let state = transform.point_state(input)?;
    Ok(PredictionWithSE {
        eta: state.eta,
        mean: state.mean,
        eta_se: state.eta_se,
        mean_se: state.mean_se,
    })
}

/// What a prediction surface asks for, independent of how it renders the answer.
pub struct PredictionRequest {
    /// `Some(level)` requests uncertainty at that central confidence level;
    /// `None` is point-only.
    pub interval: Option<f64>,
    /// Covariance source for the reported SE / bounds / observation band.
    pub covariance_mode: InferenceCovarianceMode,
    /// Emit the response-scale observation (prediction) band in addition to the
    /// credible band, for families exposing a conditional response variance.
    pub observation_interval: bool,
    /// Per-row prior weights for the heteroscedastic observation band
    /// `Var(y_i) = σ̂²/w_i` (#2077). `None` is the unweighted case and keeps the
    /// band byte-identical to a pooled scalar `σ̂²`.
    pub observation_prior_weights: Option<Array1<f64>>,
    /// Per-row η-scale variance the band adds to `Var(η)` (V∞ §5): the priced
    /// off-support ignorance of the model's measure-jet terms,
    /// `FittedModel::measure_jet_extrapolation_variance` over the RAW query rows.
    /// `None` when the model prices none or no interval is requested.
    pub extrapolation_variance: Option<Array1<f64>>,
}

/// The estimands every standard prediction surface publishes before
/// serialization.
///
/// The plug-in pair is deliberately complete: `mean_plugin` is the response
/// transform of `linear_predictor_plugin`. `posterior_mean` is the separate
/// response-scale estimand `E[T(eta) | data]`, and is absent only when a caller
/// explicitly forces a curved-link plug-in prediction. Keeping all three
/// quantities distinct here prevents presenters from pairing a plug-in linear
/// predictor with a posterior response mean under generic names (#2785).
pub struct PredictionColumns {
    pub linear_predictor_plugin: Array1<f64>,
    pub mean_plugin: Array1<f64>,
    pub posterior_mean: Option<Array1<f64>>,
    /// Link-scale posterior SD `SE(η) = √diag(X V Xᵀ)` under the covariance
    /// the band was built from; the response-scale credible bounds are the
    /// inverse link applied to the η quantiles this SD defines.
    pub linear_predictor_standard_error: Option<Array1<f64>>,
    /// Response-scale SE — the SE the response-scale band is built from, never
    /// the link-scale `σ_η` (#1536).
    pub posterior_mean_standard_error: Option<Array1<f64>>,
    pub posterior_mean_lower: Option<Array1<f64>>,
    pub posterior_mean_upper: Option<Array1<f64>>,
    pub observation_lower: Option<Array1<f64>>,
    pub observation_upper: Option<Array1<f64>>,
    /// Covariance consulted to form the point. `None` for a plug-in point,
    /// which consults none.
    pub point_covariance_source: Option<InferenceCovarianceMode>,
    /// Covariance consulted to form the reported uncertainty. `None` for a
    /// point-only request.
    pub uncertainty_covariance_source: Option<InferenceCovarianceMode>,
    /// What the posterior-mean point is conditional on when its covariance is not
    /// one the fit published (gam#2985). `None` for a plug-in point.
    pub point_covariance_provenance: Option<PointCovarianceProvenance>,
}

/// Resolve a [`PredictionRequest`] against a model: the `(interval × curved
/// link)` decision table that selects among the four predictor entry points.
///
/// This table is the last piece of predict *policy* that was not centralized
/// here, and it is the piece that drifted. The CLI and the Python FFI each held
/// their own copy — the CLI's even carried comments reading "Mirror the Python
/// FFI arm" and naming the sibling it was supposed to agree with — and they had
/// diverged in three ways: the CLI hardcoded the observation-interval switch off
/// so `gam predict --uncertainty` could never produce a prediction interval; the
/// CLI never forwarded the #2077 per-row prior weights; and where the CLI refused
/// a missing response-scale SE, the FFI silently substituted the *link-scale* SE
/// under the response-scale `std_error` column name. A comment asserting parity
/// with code it cannot call is not parity.
///
/// The refusal is now the single rule. `eta_standard_error` is a different
/// quantity from `mean_standard_error` — it lives on the link scale, beside a
/// response-scale `mean`/`mean_lower`/`mean_upper` — so emitting it under the
/// response-scale column name is worse than reporting that the backend could not
/// propagate coefficient uncertainty to the response scale.
pub fn resolve_prediction_request(
    predictor: &dyn crate::PredictableModel,
    input: &PredictInput,
    fit: &UnifiedFitResult,
    uses_posterior_mean: bool,
    request: &PredictionRequest,
) -> Result<PredictionColumns, EstimationError> {
    match (request.interval, uses_posterior_mean) {
        // Curved inverse link + interval: add SE and bounds on top of the same
        // posterior-mean point the no-interval branch reports.
        (Some(level), true) => {
            let plugin = predictor.predict_plugin_response(input)?;
            let options = PosteriorMeanOptions {
                confidence_level: Some(level),
                covariance_mode: request.covariance_mode,
                include_observation_interval: request.observation_interval,
                extrapolation_variance: request.extrapolation_variance.clone(),
                // A curved-link weighted Gaussian band is heteroscedastic in the
                // prior weight exactly as the identity-link arm below (#3957).
                observation_prior_weights: request.observation_prior_weights.clone(),
            };
            let prediction = predictor.predict_posterior_mean(input, fit, &options)?;
            let mean_standard_error = prediction.mean_standard_error.ok_or_else(|| {
                EstimationError::InvalidInput(
                    "posterior-mean prediction returned no response-scale standard error, so \
                     this model cannot report response-scale SE columns; the link-scale SE is \
                     a different quantity and is not substituted for it"
                        .to_string(),
                )
            })?;
            let (mean_lower, mean_upper) = prediction
                .mean_lower
                .zip(prediction.mean_upper)
                .ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "posterior-mean prediction did not return confidence bounds".to_string(),
                    )
                })?;
            Ok(PredictionColumns {
                linear_predictor_plugin: plugin.eta,
                mean_plugin: plugin.mean,
                posterior_mean: Some(prediction.mean),
                linear_predictor_standard_error: Some(prediction.eta_standard_error),
                posterior_mean_standard_error: Some(mean_standard_error),
                posterior_mean_lower: Some(mean_lower),
                posterior_mean_upper: Some(mean_upper),
                observation_lower: prediction.observation_lower,
                observation_upper: prediction.observation_upper,
                point_covariance_source: Some(prediction.point_covariance_source),
                uncertainty_covariance_source: prediction.uncertainty_covariance_source,
                point_covariance_provenance: prediction.point_covariance_provenance,
            })
        }
        // Effectively-linear model + interval: the plug-in equals the posterior
        // mean, so the uncertainty path reports the same point as the plain
        // branch and only adds the band; the point never moves because an
        // interval was requested (#398, #2115).
        (Some(level), false) => {
            let options = PredictUncertaintyOptions {
                confidence_level: level,
                covariance_mode: request.covariance_mode,
                includeobservation_interval: request.observation_interval,
                observation_prior_weights: request.observation_prior_weights.clone(),
                extrapolation_variance: request.extrapolation_variance.clone(),
                ..PredictUncertaintyOptions::default()
            };
            let prediction = predictor.predict_full_uncertainty(input, fit, &options)?;
            let mean_plugin = prediction.mean.clone();
            Ok(PredictionColumns {
                linear_predictor_plugin: prediction.eta,
                mean_plugin: mean_plugin.clone(),
                posterior_mean: Some(mean_plugin),
                linear_predictor_standard_error: Some(prediction.eta_standard_error),
                posterior_mean_standard_error: Some(prediction.mean_standard_error),
                posterior_mean_lower: Some(prediction.mean_lower),
                posterior_mean_upper: Some(prediction.mean_upper),
                observation_lower: prediction.observation_lower,
                observation_upper: prediction.observation_upper,
                // A linear-link plug-in point consults no coefficient
                // covariance; only the band does.
                point_covariance_source: None,
                uncertainty_covariance_source: Some(prediction.covariance_source),
                point_covariance_provenance: None,
            })
        }
        // Point-only. A curved link integrates the posterior mean — the only
        // point estimand a prediction surface publishes (SPEC: never MAP; the
        // plug-in pair is carried beside it as explicit columns) — but it must
        // not ask the backend for interval quantities: passing a confidence
        // level is the switch that populates SE/bounds, and the surfaces emit
        // whatever optionals come back (#2136).
        (None, true) => {
            let plugin = predictor.predict_plugin_response(input)?;
            let prediction = predictor.predict_posterior_mean(
                input,
                fit,
                &PosteriorMeanOptions::point_only(),
            )?;
            Ok(PredictionColumns {
                linear_predictor_plugin: plugin.eta,
                mean_plugin: plugin.mean,
                posterior_mean: Some(prediction.mean),
                linear_predictor_standard_error: None,
                posterior_mean_standard_error: None,
                posterior_mean_lower: None,
                posterior_mean_upper: None,
                observation_lower: None,
                observation_upper: None,
                point_covariance_source: Some(prediction.point_covariance_source),
                uncertainty_covariance_source: None,
                point_covariance_provenance: prediction.point_covariance_provenance,
            })
        }
        // Effectively linear response: the posterior mean and plug-in response
        // are the same estimand, so both explicit columns are populated.
        (None, false) => {
            let prediction = predictor.predict_plugin_response(input)?;
            let mean_plugin = prediction.mean.clone();
            Ok(PredictionColumns {
                linear_predictor_plugin: prediction.eta,
                mean_plugin: mean_plugin.clone(),
                posterior_mean: Some(mean_plugin),
                linear_predictor_standard_error: None,
                posterior_mean_standard_error: None,
                posterior_mean_lower: None,
                posterior_mean_upper: None,
                observation_lower: None,
                observation_upper: None,
                point_covariance_source: None,
                uncertainty_covariance_source: None,
                point_covariance_provenance: None,
            })
        }
    }
}

#[cfg(test)]
mod parity_tests {
    //! Parity of the shared engine against the per-predictor inline assembly it
    //! replaced (issue #422). Each test reconstructs — by hand, with the same
    //! confidence-level convention and arithmetic the predictors used inline —
    //! the `(point, SE, η-CI, mean-CI, observation interval)` quantities, then
    //! asserts the engine reproduces them field-for-field. The assertions are
    //! exact (`==`) wherever the engine and the hand path share the same
    //! floating-point operations, and bit-tight (`< 1e-12`) only where ordering
    //! of identical operations could differ.

    use super::*;
    use ndarray::array;

    const LEVEL: f64 = 0.95;

    /// The exact central multiplier both paths route through.
    fn z95() -> f64 {
        IntervalReference::Normal
            .central_multiplier(LEVEL)
            .expect("0.95 is a valid level")
    }

    fn assert_close(a: &Array1<f64>, b: &Array1<f64>, tag: &str) {
        assert_eq!(a.len(), b.len(), "{tag}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-12,
                "{tag}: row {i} mismatch: engine={x}, reference={y}"
            );
        }
    }

    /// SurvivalPredictor shape: a symmetric η interval, and the response band
    /// is the image of the response-index interval `u ± z·SD(u)` under the
    /// decreasing survival tail, so its endpoints swap and it lies in `[0, 1]`.
    #[test]
    fn transform_index_decreasing_map_matches_inline() {
        let eta = array![0.2, -0.5, 1.3];
        let eta_se = array![0.1, 0.2, 0.15];
        let index = ResponseIndex {
            index: array![-1.0, 0.4, 2.5],
            index_se: array![0.3, 0.5, 0.2],
        };
        let tail = |q: &Array1<f64>| -> Result<Array1<f64>, EstimationError> {
            Ok(q.mapv(|x| 1.0 / (1.0 + x.exp())))
        };
        let mean = tail(&index.index).unwrap();
        let mean_se = array![0.04, 0.06, 0.05];
        let z = z95();

        let ref_eta_lower = &eta - &eta_se.mapv(|s| z * s);
        let ref_eta_upper = &eta + &eta_se.mapv(|s| z * s);
        let ref_mean_lower = tail(&(&index.index + &index.index_se.mapv(|s| z * s))).unwrap();
        let ref_mean_upper = tail(&(&index.index - &index.index_se.mapv(|s| z * s))).unwrap();

        let out = assemble_uncertainty_result(
            LEVEL,
            IntervalReference::Normal,
            eta.clone(),
            mean.clone(),
            eta_se.clone(),
            mean_se.clone(),
            MeanBoundMethod::TransformIndex {
                index: &index,
                response_map: &tail,
            },
            None,
            UncertaintyProvenance {
                covariance_source: InferenceCovarianceMode::Conditional,
            },
        )
        .expect("engine assembly");

        assert_close(&out.eta, &eta, "eta point");
        assert_close(&out.mean, &mean, "mean point");
        assert_close(&out.eta_standard_error, &eta_se, "eta SE");
        assert_close(&out.mean_standard_error, &mean_se, "mean SE");
        assert_close(&out.eta_lower, &ref_eta_lower, "eta lower");
        assert_close(&out.eta_upper, &ref_eta_upper, "eta upper");
        assert_close(&out.mean_lower, &ref_mean_lower, "mean lower");
        assert_close(&out.mean_upper, &ref_mean_upper, "mean upper");
        for i in 0..mean.len() {
            assert!(0.0 < out.mean_lower[i] && out.mean_lower[i] <= mean[i]);
            assert!(mean[i] <= out.mean_upper[i] && out.mean_upper[i] < 1.0);
        }
        assert!(out.observation_lower.is_none());
        assert!(out.observation_upper.is_none());
        assert_eq!(out.covariance_source, InferenceCovarianceMode::Conditional);
    }

    /// BernoulliMarginalSlopePredictor shape: symmetric η interval, response
    /// interval by transforming the η endpoints through the inverse link. Here
    /// the response map is the logistic, which is monotone increasing, so the
    /// transformed endpoints stay ordered.
    #[test]
    fn transform_eta_symmetric_matches_inline() {
        let eta = array![0.2, -0.5, 1.3];
        let logistic = |e: &Array1<f64>| -> Result<Array1<f64>, EstimationError> {
            Ok(e.mapv(|x| 1.0 / (1.0 + (-x).exp())))
        };
        let mean = logistic(&eta).unwrap();
        let eta_se = array![0.1, 0.2, 0.15];
        let mean_se = array![0.02, 0.05, 0.03];
        let z = z95();

        let ref_eta_lower = &eta - &eta_se.mapv(|s| z * s);
        let ref_eta_upper = &eta + &eta_se.mapv(|s| z * s);
        let ref_mean_lower = logistic(&ref_eta_lower).unwrap();
        let ref_mean_upper = logistic(&ref_eta_upper).unwrap();

        let out = assemble_uncertainty_result(
            LEVEL,
            IntervalReference::Normal,
            eta.clone(),
            mean.clone(),
            eta_se.clone(),
            mean_se.clone(),
            MeanBoundMethod::TransformEta {
                domain: EtaDomain::Unrestricted,
                response_map: &logistic,
            },
            None,
            UncertaintyProvenance {
                covariance_source: InferenceCovarianceMode::Conditional,
            },
        )
        .expect("engine assembly");

        assert_close(&out.eta_lower, &ref_eta_lower, "eta lower");
        assert_close(&out.eta_upper, &ref_eta_upper, "eta upper");
        assert_close(&out.mean_lower, &ref_mean_lower, "mean lower");
        assert_close(&out.mean_upper, &ref_mean_upper, "mean upper");
    }

    #[test]
    fn transform_eta_nonfinite_image_is_a_typed_error() {
        let eta_lower = array![0.0_f64];
        let eta_upper = array![800.0_f64];
        let exponential =
            |eta: &Array1<f64>| -> Result<Array1<f64>, EstimationError> { Ok(eta.mapv(f64::exp)) };
        let error =
            transform_eta_interval(&eta_lower, &eta_upper, EtaDomain::Unrestricted, exponential)
                .expect_err("a non-finite transformed interval must not switch estimands");
        assert!(
            error.to_string().contains("non-finite value at row 0"),
            "unexpected error: {error}"
        );
    }

    /// GaussianLocationScalePredictor shape: identity link (mean interval ==
    /// η interval) plus a prediction (observation) interval
    /// `μ ± z·√(SE(μ̂)² + σ²)` — the predictive variance combines estimation
    /// uncertainty with the response-scale observation noise.
    #[test]
    fn identity_eta_with_observation_matches_inline() {
        let eta = array![1.0, 2.0, -1.0];
        let mean = eta.clone();
        let eta_se = array![0.3, 0.1, 0.25];
        let sigma = array![0.5, 0.4, 0.6];
        let z = z95();

        let ref_eta_lower = &eta - &eta_se.mapv(|s| z * s);
        let ref_eta_upper = &eta + &eta_se.mapv(|s| z * s);
        // Predictive SE folds the mean SE (== eta_se here) into the noise σ.
        let predictive_se = Array1::from_iter(
            eta_se
                .iter()
                .zip(sigma.iter())
                .map(|(&mse, &sd)| (mse * mse + sd * sd).sqrt()),
        );
        let ref_obs_lower = &mean - &predictive_se.mapv(|s| z * s);
        let ref_obs_upper = &mean + &predictive_se.mapv(|s| z * s);

        let out = assemble_uncertainty_result(
            LEVEL,
            IntervalReference::Normal,
            eta.clone(),
            mean.clone(),
            eta_se.clone(),
            eta_se.clone(),
            MeanBoundMethod::IdentityEta,
            Some(ObservationInterval::Symmetric {
                noise_sd: &sigma,
                bounds: ResponseBounds::UNBOUNDED,
            }),
            UncertaintyProvenance {
                covariance_source: InferenceCovarianceMode::Conditional,
            },
        )
        .expect("engine assembly");

        // Identity link: mean interval is exactly the η interval, and mean SE
        // equals the η SE.
        assert_close(&out.mean_standard_error, &eta_se, "mean SE == eta SE");
        assert_close(&out.eta_lower, &ref_eta_lower, "eta lower");
        assert_close(&out.eta_upper, &ref_eta_upper, "eta upper");
        assert_close(&out.mean_lower, &ref_eta_lower, "mean lower == eta lower");
        assert_close(&out.mean_upper, &ref_eta_upper, "mean upper == eta upper");
        assert_close(
            out.observation_lower.as_ref().expect("obs lower"),
            &ref_obs_lower,
            "observation lower",
        );
        assert_close(
            out.observation_upper.as_ref().expect("obs upper"),
            &ref_obs_upper,
            "observation upper",
        );
    }

    /// A saturated logistic band is the image of the η interval: it lies in
    /// `[0, 1]` because the logistic does, with no clamp in the path, where a
    /// delta band `μ ± z·SE(μ)` would leave the support.
    #[test]
    fn saturated_logistic_band_is_an_image_inside_the_support() {
        let eta = array![6.0, -7.5];
        let eta_se = array![3.0, 4.0];
        let logistic = |e: &Array1<f64>| -> Result<Array1<f64>, EstimationError> {
            Ok(e.mapv(|x| 1.0 / (1.0 + (-x).exp())))
        };
        let z = z95();
        let (eta_lower, eta_upper) = symmetric_interval(&eta, &eta_se, z);
        let (lower, upper) = mean_bounds(
            &eta_lower,
            &eta_upper,
            z,
            MeanBoundMethod::TransformEta {
                domain: EtaDomain::Unrestricted,
                response_map: &logistic,
            },
        )
        .expect("image band");
        assert_close(&lower, &logistic(&eta_lower).unwrap(), "lower is the image");
        assert_close(&upper, &logistic(&eta_upper).unwrap(), "upper is the image");
        for i in 0..eta.len() {
            assert!(0.0 < lower[i] && lower[i] < upper[i] && upper[i] < 1.0);
        }
    }

    /// The inverse link's jet refuses a non-positive argument, as the real
    /// reciprocal-power jets do, so a boundary endpoint must never reach it.
    fn reciprocal(e: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        if e.iter().any(|&x| x <= 0.0) {
            return Err(EstimationError::InvalidInput(
                "reciprocal evaluated off its half-line".to_string(),
            ));
        }
        Ok(e.mapv(|x| 1.0 / x))
    }

    /// An inverse-link Gamma η interval that crosses the origin is cut at the
    /// boundary: the band is the image `[1/u, ∞)` of its feasible part `(0, u]`,
    /// never a negative mean.
    #[test]
    fn half_line_inverse_link_band_crossing_origin_is_unbounded_above() {
        let domain = EtaDomain::of_spec(&LikelihoodSpec {
            response: ResponseFamily::Gamma,
            link: InverseLink::Standard(StandardLink::Inverse),
        })
        .expect("inverse link has a boundary limit");
        assert_eq!(
            domain,
            EtaDomain::HalfLine {
                feasibility: EtaFeasibility::Positive,
                boundary_mean: f64::INFINITY,
            }
        );
        let (lower, upper) =
            transform_eta_interval(&array![-0.5, 0.5], &array![2.0, 4.0], domain, reciprocal)
                .expect("feasible part of each row is non-empty");
        assert_close(&lower, &array![0.5, 0.25], "lower is 1/u");
        assert_eq!(upper[0], f64::INFINITY);
        assert!((upper[1] - 2.0).abs() < 1e-12);
    }

    /// A log-link probability interval crossing `η = 0` is cut at the boundary,
    /// where `exp(η)` tends to `1`, so the band stays inside `[0, 1]`.
    #[test]
    fn half_line_log_link_probability_band_stops_at_one() {
        let domain = EtaDomain::of_spec(&LikelihoodSpec {
            response: ResponseFamily::Binomial,
            link: InverseLink::Standard(StandardLink::Log),
        })
        .expect("log link has a boundary limit");
        let exp_below_zero = |e: &Array1<f64>| -> Result<Array1<f64>, EstimationError> {
            if e.iter().any(|&x| x >= 0.0) {
                return Err(EstimationError::InvalidInput(
                    "log-link probability evaluated at η ≥ 0".to_string(),
                ));
            }
            Ok(e.mapv(f64::exp))
        };
        let (lower, upper) =
            transform_eta_interval(&array![-0.5], &array![0.3], domain, exp_below_zero)
                .expect("the lower endpoint is feasible");
        assert!((lower[0] - (-0.5_f64).exp()).abs() < 1e-12);
        assert_eq!(upper[0], 1.0);
    }

    /// An interval with no feasible part has no feasible point, which is a
    /// typed error rather than a band.
    #[test]
    fn half_line_interval_outside_the_domain_is_a_typed_error() {
        let domain = EtaDomain::HalfLine {
            feasibility: EtaFeasibility::Positive,
            boundary_mean: f64::INFINITY,
        };
        let error = transform_eta_interval(&array![-3.0], &array![-1.0], domain, reciprocal)
            .expect_err("an infeasible interval has no image");
        assert!(
            error.to_string().contains("does not meet the feasible set"),
            "unexpected error: {error}"
        );
    }

    /// Posterior-mean bounds: `None` level leaves bounds unset; a `Some` level
    /// fills them via the same policy as the uncertainty path.
    #[test]
    fn posterior_mean_bounds_match_inline() {
        let eta = array![0.2, -0.4];
        let mean = array![0.55, 0.40];
        let eta_se = array![0.1, 0.2];
        let z = z95();
        let logistic = |e: &Array1<f64>| -> Result<Array1<f64>, EstimationError> {
            Ok(e.mapv(|x| 1.0 / (1.0 + (-x).exp())))
        };

        // No level: bounds stay None.
        let mut none_result = PredictPosteriorMeanResult {
            eta: eta.clone(),
            eta_standard_error: eta_se.clone(),
            mean: mean.clone(),
            mean_standard_error: None,
            mean_lower: None,
            mean_upper: None,
            observation_lower: None,
            observation_upper: None,
            point_covariance_source: InferenceCovarianceMode::Conditional,
            uncertainty_covariance_source: None,
            point_covariance_provenance: None,
        };
        assemble_posterior_mean_bounds(
            &mut none_result,
            None,
            IntervalReference::Normal,
            MeanBoundMethod::TransformEta {
                domain: EtaDomain::Unrestricted,
                response_map: &logistic,
            },
        )
        .expect("engine assembly");
        assert!(none_result.mean_lower.is_none());
        assert!(none_result.mean_upper.is_none());

        // With level: TransformEta bounds matching the inline path.
        let ref_eta_lower = &eta - &eta_se.mapv(|s| z * s);
        let ref_eta_upper = &eta + &eta_se.mapv(|s| z * s);
        let ref_mean_lower = logistic(&ref_eta_lower).unwrap();
        let ref_mean_upper = logistic(&ref_eta_upper).unwrap();

        let mut some_result = PredictPosteriorMeanResult {
            eta: eta.clone(),
            eta_standard_error: eta_se.clone(),
            mean: mean.clone(),
            mean_standard_error: None,
            mean_lower: None,
            mean_upper: None,
            observation_lower: None,
            observation_upper: None,
            point_covariance_source: InferenceCovarianceMode::Conditional,
            uncertainty_covariance_source: None,
            point_covariance_provenance: None,
        };
        assemble_posterior_mean_bounds(
            &mut some_result,
            Some(LEVEL),
            IntervalReference::Normal,
            MeanBoundMethod::TransformEta {
                domain: EtaDomain::Unrestricted,
                response_map: &logistic,
            },
        )
        .expect("engine assembly");
        assert_close(
            some_result.mean_lower.as_ref().expect("mean lower"),
            &ref_mean_lower,
            "posterior mean lower",
        );
        assert_close(
            some_result.mean_upper.as_ref().expect("mean upper"),
            &ref_mean_upper,
            "posterior mean upper",
        );
    }

    /// A decreasing response map (survival tail) must still yield ordered
    /// `(lower, upper)` bounds — the engine takes the per-row min/max of the
    /// transformed endpoints.
    #[test]
    fn transform_eta_non_monotone_orders_bounds() {
        let eta = array![0.0, 0.5];
        let decreasing = |e: &Array1<f64>| -> Result<Array1<f64>, EstimationError> {
            // Survival-like decreasing tail in (0, 1).
            Ok(e.mapv(|x| 1.0 / (1.0 + x.exp())))
        };
        let mean = decreasing(&eta).unwrap();
        let eta_se = array![0.2, 0.3];

        let out = assemble_uncertainty_result(
            LEVEL,
            IntervalReference::Normal,
            eta.clone(),
            mean.clone(),
            eta_se.clone(),
            eta_se.clone(),
            MeanBoundMethod::TransformEta {
                domain: EtaDomain::Unrestricted,
                response_map: &decreasing,
            },
            None,
            UncertaintyProvenance {
                covariance_source: InferenceCovarianceMode::Conditional,
            },
        )
        .expect("engine assembly");

        for (lo, hi) in out.mean_lower.iter().zip(out.mean_upper.iter()) {
            assert!(
                lo <= hi,
                "decreasing map must still return ordered bounds: {lo} > {hi}"
            );
            assert!((0.0..=1.0).contains(lo) && (0.0..=1.0).contains(hi));
        }
    }
}

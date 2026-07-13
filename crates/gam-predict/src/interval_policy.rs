//! Shared interval / posterior-mean policy engine.
//!
//! Every `PredictableModel` predictor computes its own linear predictor(s),
//! response transform, and standard errors using domain-specific math. What
//! they all share — and previously re-implemented inline — is the *policy*
//! layer that turns those quantities into confidence intervals and result
//! structs:
//!
//!   1. the central normal multiplier `z = Φ⁻¹(½ + ½·level)`,
//!   2. the η-scale interval `η ± z·SE(η)`,
//!   3. the response-scale interval, either by transforming the η endpoints
//!      through the inverse link (handling non-monotone maps) or by the
//!      delta-method `μ ± z·SE(μ)`,
//!   4. clamping the response-scale bounds to the family support, and
//!   5. assembling [`PredictUncertaintyResult`] / [`PredictPosteriorMeanResult`].
//!
//! Centralizing this here means the confidence-level convention, quantile
//! routine, covariance-mode plumbing, and response-bound clamps live in one
//! place. A predictor defines only its linear state + response transform and
//! delegates the policy to the helpers below, so interval/posterior-mean
//! behaviour cannot drift between families.

use crate::{
    InferenceCovarianceMode, PosteriorMeanOptions, PredictInput, PredictPosteriorMeanResult,
    PredictResult, PredictUncertaintyOptions, PredictUncertaintyResult, PredictionWithSE,
    family_observation_band,
};
use gam_problem::EstimationError;
use gam_solve::model_types::UnifiedFitResult;
use gam_spec::ResponseFamily;
use ndarray::Array1;

/// Closed response-scale support `[lo, hi]` used to clamp transformed interval
/// endpoints. `None` means the response is unbounded and must not be clamped.
///
/// This is the predict-side policy mirror of
/// [`ResponseFamily::mean_clamp_bounds`]: families expose their natural mean
/// support and the engine applies it uniformly. Predictors whose response is a
/// probability that is *already* evaluated post-transformation (survival tail,
/// binomial location-scale) report the closed `[0, 1]` bounds directly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResponseBounds(Option<(f64, f64)>);

impl ResponseBounds {
    /// Unbounded response — endpoints are passed through unclamped.
    pub const UNBOUNDED: Self = Self(None);
    /// Closed unit interval `[0, 1]` (probabilities, survival tails).
    pub const UNIT_PROBABILITY: Self = Self(Some((0.0, 1.0)));

    /// Explicit closed bounds.
    pub fn closed(lo: f64, hi: f64) -> Self {
        Self(Some((lo, hi)))
    }

    /// The response-support clamp for a [`ResponseFamily`], matching
    /// [`ResponseFamily::mean_clamp_bounds`].
    pub fn for_family(response: &ResponseFamily) -> Self {
        Self(response.mean_clamp_bounds())
    }

    /// The clamp applied to the **observation (prediction) interval** of a
    /// [`ResponseFamily`], matching [`ResponseFamily::response_support_bounds`].
    ///
    /// Distinct from [`Self::for_family`] (the *mean*-interval clamp): the
    /// observation band (symmetric `μ ± z·σ_pred` for most families, equal-tailed
    /// Gamma quantiles for the skewed Gamma arm, see `family_observation_band`)
    /// crosses the support floor for a small fitted mean even when the
    /// mean-interval clamp is `None`. See
    /// [`ResponseFamily::response_support_bounds`].
    pub fn response_support(response: &ResponseFamily) -> Self {
        Self(response.response_support_bounds())
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
    pub fn clamp_in_place(&self, values: &mut Array1<f64>) {
        if let Some((lo, hi)) = self.0 {
            values.mapv_inplace(|v| v.clamp(lo, hi));
        }
    }
}

/// The central two-sided normal multiplier `z = Φ⁻¹(½ + ½·level)` for a
/// confidence `level ∈ (0, 1)`.
///
/// This is the single source of truth for the confidence-level convention used
/// throughout the predict path; every predictor's interval construction routes
/// its quantile through here so the convention cannot diverge.
pub fn central_z(level: f64) -> Result<f64, EstimationError> {
    gam_math::probability::standard_normal_quantile(0.5 + 0.5 * level)
        .map_err(EstimationError::InvalidInput)
}

/// Validate that a confidence level is a usable probability in the open unit
/// interval, returning the corresponding central multiplier.
pub fn validated_central_z(level: f64) -> Result<f64, EstimationError> {
    if !(level.is_finite() && level > 0.0 && level < 1.0) {
        return Err(EstimationError::InvalidInput(format!(
            "confidence_level must be in (0,1), got {level}"
        )));
    }
    central_z(level)
}

/// The symmetric interval `center ± z·se`, returned as `(lower, upper)`.
#[inline]
pub fn symmetric_interval(
    center: &Array1<f64>,
    se: &Array1<f64>,
    z: f64,
) -> (Array1<f64>, Array1<f64>) {
    let half_width = se.mapv(|s| z * s);
    (center - &half_width, center + &half_width)
}

/// Number of evaluation nodes (endpoints included) used to scan the response
/// map over each η interval in [`transform_eta_interval`]. Odd, so the interval
/// midpoint — the point predictor for a symmetric η interval — is always a
/// node. Monotone maps attain their extrema at the endpoint nodes, so for them
/// the scan is exact and reproduces the endpoint-only construction bit-for-bit;
/// the interior nodes exist to catch extrema of non-monotone response maps
/// (learnable link wiggles), for which endpoint-only transformation is false
/// (e.g. `η²` on `[-1, 1]` has image `[0, 1]`, not `[1, 1]`).
const TRANSFORM_INTERVAL_SCAN_NODES: usize = 17;

/// Response-scale interval built by transforming the η-scale interval through
/// a (possibly non-monotone) response map, then clamping to `bounds`.
///
/// `response_map` is the predictor's inverse-link / response transform. The map
/// is evaluated on [`TRANSFORM_INTERVAL_SCAN_NODES`] evenly spaced nodes across
/// each row's η interval and the per-row min/max over the scan is returned, so
/// interior extrema of a non-monotone transform (link wiggles) are captured
/// rather than silently cut off by an endpoint-only image. Because some
/// transforms (notably survival tails) are decreasing, the min/max also keeps
/// the returned `(lower, upper)` genuinely ordered.
///
/// Every transformed node must be finite. A non-finite response image is a
/// typed prediction failure: changing that row to a delta-method interval would
/// silently substitute a different uncertainty estimand. Degenerate all-zero
/// count responses are rejected at the family-validation boundary before a fit
/// is minted (#2255).
pub fn transform_eta_interval<F>(
    eta_lower: &Array1<f64>,
    eta_upper: &Array1<f64>,
    bounds: ResponseBounds,
    response_map: F,
) -> Result<(Array1<f64>, Array1<f64>), EstimationError>
where
    F: Fn(&Array1<f64>) -> Result<Array1<f64>, EstimationError>,
{
    let n = eta_lower.len();
    // Scan the response map over each row's η interval. Node 0 is the lower
    // endpoint and the last node is the upper endpoint, so a monotone map's
    // extrema are reproduced exactly; interior nodes capture non-monotone
    // extrema. All nodes must be finite for a row's scan to count — `f64::min`/
    // `max` return the non-NaN argument, so a single `+inf`/NaN node would
    // otherwise slip through as a finite-but-meaningless bound.
    const K: usize = TRANSFORM_INTERVAL_SCAN_NODES;
    let mut scan_min = Array1::<f64>::from_elem(n, f64::INFINITY);
    let mut scan_max = Array1::<f64>::from_elem(n, f64::NEG_INFINITY);
    let mut scan_finite = vec![true; n];
    for k in 0..K {
        let t = (k as f64) / ((K - 1) as f64);
        let eta_node =
            Array1::from_shape_fn(n, |i| eta_lower[i] + t * (eta_upper[i] - eta_lower[i]));
        let transformed = response_map(&eta_node)?;
        for i in 0..n {
            let v = transformed[i];
            if v.is_finite() {
                scan_min[i] = scan_min[i].min(v);
                scan_max[i] = scan_max[i].max(v);
            } else {
                scan_finite[i] = false;
            }
        }
    }
    let mut mean_lower = Array1::<f64>::zeros(n);
    let mut mean_upper = Array1::<f64>::zeros(n);
    for i in 0..n {
        if !scan_finite[i] {
            return Err(EstimationError::InvalidInput(format!(
                "response-scale interval transform produced a non-finite value at row {i}"
            )));
        }
        mean_lower[i] = scan_min[i].min(scan_max[i]);
        mean_upper[i] = scan_max[i].max(scan_min[i]);
    }
    bounds.clamp_in_place(&mut mean_lower);
    bounds.clamp_in_place(&mut mean_upper);
    Ok((mean_lower, mean_upper))
}

/// Response-scale interval built by the delta method `μ ± z·SE(μ)`, then
/// clamped to `bounds`.
pub fn delta_mean_interval(
    mean: &Array1<f64>,
    mean_se: &Array1<f64>,
    z: f64,
    bounds: ResponseBounds,
) -> (Array1<f64>, Array1<f64>) {
    let (mut mean_lower, mut mean_upper) = symmetric_interval(mean, mean_se, z);
    bounds.clamp_in_place(&mut mean_lower);
    bounds.clamp_in_place(&mut mean_upper);
    (mean_lower, mean_upper)
}

/// How a predictor maps its η-scale interval onto the response scale.
///
/// This is the policy split the predictors used to inline: probability /
/// survival families transform the η endpoints through their response map
/// (well-behaved for nonlinear links), Gaussian-identity families reuse the
/// η interval directly, and dispersion families take the delta-method route.
pub enum MeanBoundMethod<'a> {
    /// Transform `η ± z·SE(η)` through the supplied response map (non-monotone
    /// safe) and clamp to `bounds`. Non-finite transformed values are errors;
    /// this path never substitutes a delta-method interval.
    TransformEta {
        bounds: ResponseBounds,
        response_map: &'a (dyn Fn(&Array1<f64>) -> Result<Array1<f64>, EstimationError> + 'a),
    },
    /// `μ ± z·SE(μ)` clamped to `bounds`.
    Delta {
        mean_se: &'a Array1<f64>,
        bounds: ResponseBounds,
    },
    /// The response equals the linear predictor (identity link); the response
    /// bounds are exactly the η bounds.
    IdentityEta,
}

/// Compute response-scale `(mean_lower, mean_upper)` for the requested method.
pub fn mean_bounds(
    eta_lower: &Array1<f64>,
    eta_upper: &Array1<f64>,
    mean: &Array1<f64>,
    z: f64,
    method: MeanBoundMethod<'_>,
) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
    match method {
        MeanBoundMethod::TransformEta {
            bounds,
            response_map,
        } => transform_eta_interval(eta_lower, eta_upper, bounds, response_map),
        MeanBoundMethod::Delta { mean_se, bounds } => {
            Ok(delta_mean_interval(mean, mean_se, z, bounds))
        }
        MeanBoundMethod::IdentityEta => Ok((eta_lower.clone(), eta_upper.clone())),
    }
}

/// How the η-scale confidence interval is produced for a predictor.
///
/// Most families form the central interval `η ± z·SE(η)`. Threshold-scale
/// families (binomial location-scale) have an η predictor whose interval is not
/// directly meaningful on the response scale, so they collapse the η interval
/// onto the point predictor and carry all uncertainty through the delta-method
/// response interval instead.
pub enum EtaInterval {
    /// Central interval `η ± z·SE(η)`.
    Symmetric,
    /// η interval collapsed to the point predictor (`η_lower = η_upper = η`);
    /// response uncertainty is carried entirely by the mean-bound method.
    Collapsed,
}

impl EtaInterval {
    fn endpoints(
        &self,
        eta: &Array1<f64>,
        eta_se: &Array1<f64>,
        z: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        match self {
            EtaInterval::Symmetric => symmetric_interval(eta, eta_se, z),
            EtaInterval::Collapsed => (eta.clone(), eta.clone()),
        }
    }
}

/// Observation (prediction) interval construction selected by a predictor.
/// Keeping the analytic override as its own variant matters for families such
/// as Royston–Parmar: their fresh-response law has an exact discrete predictive
/// set but no additive response-noise standard deviation.
pub enum ObservationInterval<'a> {
    /// Symmetric `μ ± z·√(SE(μ̂)² + σ²)` band from a per-row
    /// response-scale noise standard deviation.
    Symmetric {
        noise_sd: &'a Array1<f64>,
        /// Response-support clamp; [`ResponseBounds::UNBOUNDED`] for real-line
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
pub struct UncertaintyProvenance {
    pub covariance_source: InferenceCovarianceMode,
}

/// Assemble a [`PredictUncertaintyResult`] from a predictor's already-computed
/// linear-predictor / response state.
///
/// This is the shared tail every `predict_full_uncertainty` impl used to inline:
/// validate the confidence level, form the η interval, map it onto the response
/// scale via `method`, optionally attach an observation interval, and populate
/// the result struct. Predictors supply only the family-specific quantities
/// (`eta`, `mean`, the two standard errors) plus the policy choices
/// (`eta_interval`, `method`); the engine owns everything else so interval
/// construction cannot drift between families.
pub fn assemble_uncertainty_result(
    confidence_level: f64,
    eta: Array1<f64>,
    mean: Array1<f64>,
    eta_standard_error: Array1<f64>,
    mean_standard_error: Array1<f64>,
    eta_interval: EtaInterval,
    method: MeanBoundMethod<'_>,
    observation: Option<ObservationInterval<'_>>,
    provenance: UncertaintyProvenance,
) -> Result<PredictUncertaintyResult, EstimationError> {
    let z = validated_central_z(confidence_level)?;
    let (eta_lower, eta_upper) = eta_interval.endpoints(&eta, &eta_standard_error, z);
    let (mean_lower, mean_upper) = mean_bounds(&eta_lower, &eta_upper, &mean, z, method)?;
    let (observation_lower, observation_upper) = match observation {
        // A skew-aware predictor (dispersion location-scale) supplies its
        // equal-tailed band directly; use it verbatim (already support-clamped).
        Some(ObservationInterval::Override { lower, upper }) => (Some(lower), Some(upper)),
        Some(ObservationInterval::Symmetric { noise_sd, bounds }) => {
            // A prediction (observation) interval covers a *future* response
            // `Y = μ + ε` at the query point. The point `μ̂` is itself estimated
            // (`Var(μ̂) = mean_standard_error²`) and the observation noise has
            // `Var(Y|μ) = noise_sd²`; the two are independent, so the predictive
            // variance is the sum and the band half-width is
            //   z·√(SE(μ̂)² + σ²),
            // NOT `z·σ`. Dropping the estimation term under-covers wherever the
            // fit is uncertain. This matches `family_observation_band`'s
            // `√(mean_se² + obsvar)` convention used by the dedicated engine.
            let predictive_se = Array1::from_iter(
                mean_standard_error
                    .iter()
                    .zip(noise_sd.iter())
                    .map(|(&mse, &sd)| (mse * mse + sd * sd).max(0.0).sqrt()),
            );
            let half = predictive_se.mapv(|s| z * s);
            let mut lower = &mean - &half;
            let mut upper = &mean + &half;
            // The predictive band must lie within the response support; a
            // symmetric band on a bounded/half-bounded response otherwise
            // reports impossible values (a count band going negative).
            bounds.clamp_in_place(&mut lower);
            bounds.clamp_in_place(&mut upper);
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
/// `None`. `eta` / `eta_se` are taken from `result`, so a predictor whose η
/// interval is not meaningful supplies `EtaInterval::Collapsed`.
pub fn assemble_posterior_mean_bounds(
    result: &mut PredictPosteriorMeanResult,
    confidence_level: Option<f64>,
    eta_interval: EtaInterval,
    method: MeanBoundMethod<'_>,
) -> Result<(), EstimationError> {
    let Some(level) = confidence_level else {
        return Ok(());
    };
    let z = validated_central_z(level)?;
    let (eta_lower, eta_upper) = eta_interval.endpoints(&result.eta, &result.eta_standard_error, z);
    let (mean_lower, mean_upper) = mean_bounds(&eta_lower, &eta_upper, &result.mean, z, method)?;
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
/// [`PredictionTransform::linear_state`] and
/// [`PredictionTransform::response_jacobian_rows`] so the family can branch its
/// numerics and its interval policy while the assembly stays unified.
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
/// inline directly into [`assemble_uncertainty_result`] / [`mean_bounds`]; a
/// [`PredictionTransform`] now declares it once and the generic drivers thread
/// it through both the full-uncertainty and posterior-mean pipelines.
pub enum ResponseInterval {
    /// Transform the η endpoints `η ± z·SE(η)` through [`PredictionTransform::response`]
    /// (non-monotone safe), then clamp to [`PredictionTransform::bounds`].
    /// Used by families whose response is a smooth inverse-link image of η
    /// (standard link-wiggle, Bernoulli marginal-slope).
    TransformEta,
    /// Identity link: the response equals the linear predictor, so the response
    /// interval is exactly the η interval (Gaussian location-scale, PIT).
    IdentityEta,
    /// Delta method `μ ± z·SE(μ)` clamped to [`PredictionTransform::bounds`],
    /// with the η interval collapsed onto the point predictor. Used by
    /// threshold-scale probability families whose η interval is not directly
    /// meaningful on the response scale (binomial location-scale, survival
    /// tail).
    CollapsedDelta,
    /// Delta method `μ ± z·SE(μ)` clamped to [`PredictionTransform::bounds`]
    /// with a symmetric η interval retained. Used by families that report a
    /// genuine η interval *and* a response-scale delta SE (survival full
    /// uncertainty).
    SymmetricDelta,
}

/// The η-scale state a [`PredictionTransform`] produces for one prediction
/// batch: the linear predictor, its response-scale image, and (when a
/// covariance is available) the η- and mean-scale standard errors.
///
/// Each predictor computes these via whatever bespoke gradient backend its
/// parameterisation requires (dense matvec, link-wiggle chain rule, projected
/// two-block covariance, bivariate GHQ). The generic drivers below consume the
/// finished arrays and own the *policy* layer — interval construction, support
/// clamping, and result assembly — so that layer cannot drift between families.
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
///   * [`response`](PredictionTransform::response) — the response map μ = T(η),
///     used to transform η-interval endpoints onto the response scale;
///   * [`response_jacobian_rows`](PredictionTransform::response_jacobian_rows) —
///     whether the mean SE supplied by `linear_state` is consumed via the
///     delta method or is the η SE itself (identity link);
///   * [`bounds`](PredictionTransform::bounds) — the response-scale support
///     clamp;
///   * [`response_interval`](PredictionTransform::response_interval) — which
///     [`ResponseInterval`] policy maps the η interval onto the response scale;
///   * [`observation_noise`](PredictionTransform::observation_noise) — the
///     optional response-scale observation-noise σ.
///
/// Everything else — confidence-level validation, η/mean interval construction,
/// support clamping, observation intervals, and result-struct assembly — lives
/// in the generic drivers [`predict_full_uncertainty_generic`] and
/// [`predict_posterior_mean_generic`], so the pipeline is one source of truth.
pub trait PredictionTransform {
    /// The fit-free point state: η, μ, and the covariance-derived standard
    /// errors (`None` when no predictor covariance is available). This is the
    /// state behind the point-prediction drivers
    /// [`predict_plugin_response_generic`] and [`predict_with_uncertainty_generic`],
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

    /// Response map μ = T(η), used to transform η-interval endpoints onto the
    /// response scale (non-monotone safe via [`transform_eta_interval`]).
    fn response(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError>;

    /// Which [`ResponseInterval`] policy maps the η interval onto the response
    /// scale for `pass`. This is the policy mirror of supplying explicit
    /// response-scale Jacobian rows to the SE backend vs. transforming η
    /// endpoints; it selects between the `Delta`/`TransformEta`/`IdentityEta`
    /// arms and the collapsed/symmetric η interval.
    fn response_jacobian_rows(&self, pass: PredictPass) -> ResponseInterval;

    /// Response-scale support `[lo, hi]` clamp.
    fn bounds(&self) -> ResponseBounds;

    /// The response distribution family. Used by the generic posterior-mean
    /// driver to build the per-family observation (prediction) interval via
    /// [`family_observation_band`]; `RoystonParmar` yields the discrete
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
    /// two-block sibling of [`family_observation_band`] (#817/#1193/#1194). When
    /// `Some`, the returned `(lower, upper)` replaces the symmetric band in both
    /// the full-uncertainty and posterior-mean drivers; when `None`, the symmetric
    /// path is used.
    ///
    /// `mean` / `mean_se` are the per-row point and its standard error already
    /// computed by the driver; `z_lower` / `z_upper` are the per-row tail
    /// multipliers (the same masses the symmetric band would target).
    fn observation_band(
        &self,
        _: &PredictInput,
        mean: &Array1<f64>,
        mean_se: &Array1<f64>,
        z_lower: &Array1<f64>,
        z_upper: &Array1<f64>,
    ) -> Result<Option<(Array1<f64>, Array1<f64>)>, EstimationError> {
        // Default: no skew-aware band. The generic symmetric construction is
        // used instead. Validate the per-row inputs the driver hands every
        // transform so an overriding impl and this default agree on shape.
        assert_eq!(mean.len(), mean_se.len());
        assert_eq!(mean.len(), z_lower.len());
        assert_eq!(mean.len(), z_upper.len());
        Ok(None)
    }
}

/// Build the [`MeanBoundMethod`] selected by a transform's [`ResponseInterval`]
/// policy, borrowing the response closure / mean SE as needed.
fn mean_bound_method_for<'a, T: PredictionTransform>(
    transform: &'a T,
    policy: &ResponseInterval,
    response_map: &'a (dyn Fn(&Array1<f64>) -> Result<Array1<f64>, EstimationError> + 'a),
    mean_se: &'a Array1<f64>,
) -> MeanBoundMethod<'a> {
    match policy {
        ResponseInterval::TransformEta => MeanBoundMethod::TransformEta {
            bounds: transform.bounds(),
            response_map,
        },
        ResponseInterval::IdentityEta => MeanBoundMethod::IdentityEta,
        ResponseInterval::CollapsedDelta | ResponseInterval::SymmetricDelta => {
            MeanBoundMethod::Delta {
                mean_se,
                bounds: transform.bounds(),
            }
        }
    }
}

/// The η-interval policy implied by a [`ResponseInterval`].
fn eta_interval_for(policy: &ResponseInterval) -> EtaInterval {
    match policy {
        ResponseInterval::CollapsedDelta => EtaInterval::Collapsed,
        ResponseInterval::TransformEta
        | ResponseInterval::IdentityEta
        | ResponseInterval::SymmetricDelta => EtaInterval::Symmetric,
    }
}

/// The single full-uncertainty driver. Runs the predict pipeline once for any
/// [`PredictionTransform`]: compute the η-scale state, require its standard
/// errors, attach the optional observation interval, and assemble the result
/// through [`assemble_uncertainty_result`].
pub fn predict_full_uncertainty_generic<T: PredictionTransform>(
    transform: &T,
    input: &PredictInput,
    fit: &UnifiedFitResult,
    options: &PredictUncertaintyOptions,
) -> Result<PredictUncertaintyResult, EstimationError> {
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
    let policy = transform.response_jacobian_rows(PredictPass::FullUncertainty);
    let response_map = move |eta: &Array1<f64>| transform.response(eta);
    let observation = if options.includeobservation_interval {
        transform.observation_noise(input)?
    } else {
        None
    };
    // A skew-aware predictor (the dispersion location-scale families) builds an
    // equal-tailed band per row from its moment-matched predictive; when present
    // it replaces the symmetric `μ ± z·σ` construction below (#817/#1193/#1194).
    let mut override_band = if options.includeobservation_interval {
        let z = validated_central_z(options.confidence_level)?;
        let z_row = Array1::from_elem(state.mean.len(), z);
        transform.observation_band(input, &state.mean, &mean_se, &z_row, &z_row)?
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
        let z = validated_central_z(options.confidence_level)?;
        let z_row = Array1::from_elem(state.mean.len(), z);
        let eta_variance = eta_se.mapv(|standard_error| standard_error * standard_error);
        let (lower, upper) = family_observation_band(
            &response_family,
            &state.eta,
            &eta_variance,
            &state.mean,
            &mean_se,
            &z_row,
            &z_row,
            fit,
            None,
        );
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
        state.eta,
        state.mean,
        eta_se,
        mean_se.clone(),
        eta_interval_for(&policy),
        mean_bound_method_for(transform, &policy, &response_map, &mean_se),
        observation_interval,
        UncertaintyProvenance { covariance_source },
    )
}

/// The single posterior-mean driver. Runs the predict pipeline once for any
/// [`PredictionTransform`]: compute the η-scale state and attach response-scale
/// confidence bounds (when a level is supplied) through
/// [`assemble_posterior_mean_bounds`].
pub fn predict_posterior_mean_generic<T: PredictionTransform>(
    transform: &T,
    input: &PredictInput,
    fit: &UnifiedFitResult,
    options: &PosteriorMeanOptions,
) -> Result<PredictPosteriorMeanResult, EstimationError> {
    // POINT: the posterior-mean pass always integrates the *conditional*
    // posterior, so the reported point is invariant to the uncertainty request
    // (issue #398). `covariance_mode` only shapes the uncertainty attached below.
    let state = transform.linear_state(
        input,
        fit,
        PredictPass::PosteriorMean,
        InferenceCovarianceMode::Conditional,
    )?;
    let policy = transform.response_jacobian_rows(PredictPass::PosteriorMean);
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
    // genuine delta-method `mean_se`; a missing one is a producer bug, not a
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
    };

    let Some(level) = options.confidence_level else {
        return Ok(result);
    };

    // UNCERTAINTY: the reported SE / credible bounds / observation band honour
    // `covariance_mode` (issues #811/#812: this path previously hardwired the
    // conditional covariance). The posterior-mean *point* above stays
    // conditional; only the uncertainty responds.
    //
    // `Conditional` keeps the posterior pass's own SE. `SmoothingCorrected`
    // re-derives the SEs from the full-uncertainty pass and errors if the fit
    // cannot supply that exact covariance definition.
    let (eta_se, mean_se) = match options.covariance_mode {
        InferenceCovarianceMode::Conditional => (cond_eta_se, cond_mean_se),
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
            let mean_se = unc.mean_se.ok_or_else(|| {
                EstimationError::InvalidInput(
                    "smoothing-corrected posterior-mean uncertainty requires mean SE".to_string(),
                )
            })?;
            if unc.covariance_source != InferenceCovarianceMode::SmoothingCorrected {
                return Err(EstimationError::InvalidInput(
                    "smoothing-corrected posterior-mean uncertainty resolved a conditional covariance"
                        .to_string(),
                ));
            }
            (eta_se, mean_se)
        }
    };
    result.uncertainty_covariance_source = Some(options.covariance_mode);
    result.eta_standard_error = eta_se;
    // Record the response-scale SE used to build the credible band so the FFI/CLI
    // predict tables can report it as `std_error` instead of the link-scale σ_η
    // (#1536).
    result.mean_standard_error = Some(mean_se.clone());

    {
        let response_map = |eta: &Array1<f64>| transform.response(eta);
        assemble_posterior_mean_bounds(
            &mut result,
            Some(level),
            eta_interval_for(&policy),
            mean_bound_method_for(transform, &policy, &response_map, &mean_se),
        )?;
    }

    if options.include_observation_interval {
        let z = validated_central_z(level)?;
        let z_row = Array1::from_elem(result.mean.len(), z);
        // A skew-aware dispersion location-scale predictor builds an equal-tailed
        // band per row from its moment-matched predictive (#817/#1193/#1194). When
        // present it replaces the symmetric `μ ± z·σ(x)` band below, so the
        // posterior-mean API matches the full-uncertainty API on the same skewed
        // fit instead of emitting a symmetric band.
        let skew_band =
            transform.observation_band(input, &result.mean, &mean_se, &z_row, &z_row)?;
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
                let predictive_se = Array1::from_iter(
                    mean_se
                        .iter()
                        .zip(noise_sd.iter())
                        .map(|(&mse, &sd)| (mse * mse + sd * sd).max(0.0).sqrt()),
                );
                let half = predictive_se.mapv(|s| z * s);
                let mut lower = &result.mean - &half;
                let mut upper = &result.mean + &half;
                bounds.clamp_in_place(&mut lower);
                bounds.clamp_in_place(&mut upper);
                result.observation_lower = Some(lower);
                result.observation_upper = Some(upper);
            }
            // Single-distribution families with no per-row noise submodel:
            // the fit-level scalar dispersion is the correct observation noise,
            // and `family_observation_band` additionally applies the skew-aware
            // Gamma predictive arm for the right-skewed positive families.
            (None, None) => {
                let etavar = result.eta_standard_error.mapv(|s| s * s);
                let (obs_lower, obs_upper) = family_observation_band(
                    &transform.response_family(),
                    &result.eta,
                    &etavar,
                    &result.mean,
                    &mean_se,
                    &z_row,
                    &z_row,
                    fit,
                    // Generic transform posterior-mean band: analytic prior
                    // weights (#2077) are threaded through the dedicated
                    // full-uncertainty Gaussian path, not this driver (None ⇒
                    // unchanged for the families reaching here).
                    None,
                );
                result.observation_lower = obs_lower;
                result.observation_upper = obs_upper;
            }
        }
    }

    Ok(result)
}

/// The single plug-in response driver: the transform's fit-free point state,
/// keeping only η and μ.
pub fn predict_plugin_response_generic<T: PredictionTransform>(
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
pub fn predict_with_uncertainty_generic<T: PredictionTransform>(
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
        central_z(LEVEL).expect("0.95 is a valid level")
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

    /// StandardPredictor (link-wiggle) / SurvivalPredictor shape:
    /// symmetric η interval, delta-method response interval.
    #[test]
    fn delta_symmetric_matches_inline() {
        let eta = array![0.2, -0.5, 1.3];
        let mean = array![0.55, 0.38, 0.78];
        let eta_se = array![0.1, 0.2, 0.15];
        let mean_se = array![0.04, 0.06, 0.05];
        let z = z95();

        // Hand-built reference (the old inline path).
        let ref_eta_lower = &eta - &eta_se.mapv(|s| z * s);
        let ref_eta_upper = &eta + &eta_se.mapv(|s| z * s);
        let ref_mean_lower = (&mean - &mean_se.mapv(|s| z * s)).mapv(|v| v.clamp(0.0, 1.0));
        let ref_mean_upper = (&mean + &mean_se.mapv(|s| z * s)).mapv(|v| v.clamp(0.0, 1.0));

        let out = assemble_uncertainty_result(
            LEVEL,
            eta.clone(),
            mean.clone(),
            eta_se.clone(),
            mean_se.clone(),
            EtaInterval::Symmetric,
            MeanBoundMethod::Delta {
                mean_se: &mean_se,
                bounds: ResponseBounds::UNIT_PROBABILITY,
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
            eta.clone(),
            mean.clone(),
            eta_se.clone(),
            mean_se.clone(),
            EtaInterval::Symmetric,
            MeanBoundMethod::TransformEta {
                bounds: ResponseBounds::UNIT_PROBABILITY,
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
        let error = transform_eta_interval(
            &eta_lower,
            &eta_upper,
            ResponseBounds::UNBOUNDED,
            exponential,
        )
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
            eta.clone(),
            mean.clone(),
            eta_se.clone(),
            eta_se.clone(),
            EtaInterval::Symmetric,
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

    /// BinomialLocationScalePredictor shape: the threshold-scale η interval is
    /// collapsed onto the point predictor, so the η bounds equal η exactly and
    /// the response interval is the delta method on `[0, 1]`.
    #[test]
    fn delta_collapsed_eta_matches_inline() {
        let eta = array![-0.3, 0.7, 0.1];
        let mean = array![0.42, 0.66, 0.51];
        let mean_se = array![0.05, 0.08, 0.04];
        let z = z95();

        let ref_mean_lower = (&mean - &mean_se.mapv(|s| z * s)).mapv(|v| v.clamp(0.0, 1.0));
        let ref_mean_upper = (&mean + &mean_se.mapv(|s| z * s)).mapv(|v| v.clamp(0.0, 1.0));

        let out = assemble_uncertainty_result(
            LEVEL,
            eta.clone(),
            mean.clone(),
            mean_se.clone(),
            mean_se.clone(),
            EtaInterval::Collapsed,
            MeanBoundMethod::Delta {
                mean_se: &mean_se,
                bounds: ResponseBounds::UNIT_PROBABILITY,
            },
            None,
            UncertaintyProvenance {
                covariance_source: InferenceCovarianceMode::Conditional,
            },
        )
        .expect("engine assembly");

        // Collapsed η interval: both endpoints equal the point predictor.
        assert_close(&out.eta_lower, &eta, "eta lower == eta");
        assert_close(&out.eta_upper, &eta, "eta upper == eta");
        assert_close(&out.mean_lower, &ref_mean_lower, "mean lower");
        assert_close(&out.mean_upper, &ref_mean_upper, "mean upper");
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
        };
        assemble_posterior_mean_bounds(
            &mut none_result,
            None,
            EtaInterval::Symmetric,
            MeanBoundMethod::TransformEta {
                bounds: ResponseBounds::UNIT_PROBABILITY,
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
        };
        assemble_posterior_mean_bounds(
            &mut some_result,
            Some(LEVEL),
            EtaInterval::Symmetric,
            MeanBoundMethod::TransformEta {
                bounds: ResponseBounds::UNIT_PROBABILITY,
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

    /// A genuinely non-monotone response map must have its interior extrema
    /// captured: `f(η) = η²` on the symmetric interval `[-c, c]` has image
    /// `[0, c²]`, while endpoint-only transformation would report the
    /// degenerate `[c², c²]`.
    #[test]
    fn transform_eta_non_monotone_captures_interior_extrema() {
        let eta = array![0.0];
        let square =
            |e: &Array1<f64>| -> Result<Array1<f64>, EstimationError> { Ok(e.mapv(|x| x * x)) };
        let mean = square(&eta).unwrap();
        let eta_se = array![1.0];
        let z = z95();
        let c = z * eta_se[0];

        let out = assemble_uncertainty_result(
            LEVEL,
            eta.clone(),
            mean.clone(),
            eta_se.clone(),
            eta_se.clone(),
            EtaInterval::Symmetric,
            MeanBoundMethod::TransformEta {
                bounds: ResponseBounds::UNBOUNDED,
                response_map: &square,
            },
            None,
            UncertaintyProvenance {
                covariance_source: InferenceCovarianceMode::Conditional,
            },
        )
        .expect("engine assembly");

        // The interval midpoint η = 0 is a scan node, so the true interior
        // minimum f(0) = 0 is found exactly; the maximum is at the endpoints.
        assert!(
            out.mean_lower[0].abs() < 1e-12,
            "interior minimum not captured: lower = {}",
            out.mean_lower[0]
        );
        assert!(
            (out.mean_upper[0] - c * c).abs() < 1e-12,
            "endpoint maximum wrong: upper = {}, expected {}",
            out.mean_upper[0],
            c * c
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
            eta.clone(),
            mean.clone(),
            eta_se.clone(),
            eta_se.clone(),
            EtaInterval::Symmetric,
            MeanBoundMethod::TransformEta {
                bounds: ResponseBounds::UNIT_PROBABILITY,
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

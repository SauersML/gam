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

use crate::estimate::EstimationError;
use crate::types::ResponseFamily;
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
    crate::probability::standard_normal_quantile(0.5 + 0.5 * level)
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

/// Response-scale interval built by transforming the η-scale endpoints through
/// a (possibly non-monotone) response map, then clamping to `bounds`.
///
/// `response_map` is the predictor's inverse-link / response transform applied
/// to a vector of linear-predictor endpoints. Because some transforms (notably
/// survival tails) are decreasing, the per-row min/max of the two transformed
/// endpoints is taken so the returned `(lower, upper)` are genuinely ordered.
pub fn transform_eta_interval<F>(
    eta_lower: &Array1<f64>,
    eta_upper: &Array1<f64>,
    bounds: ResponseBounds,
    response_map: F,
) -> Result<(Array1<f64>, Array1<f64>), EstimationError>
where
    F: Fn(&Array1<f64>) -> Result<Array1<f64>, EstimationError>,
{
    let transformed_lower = response_map(eta_lower)?;
    let transformed_upper = response_map(eta_upper)?;
    let mut mean_lower = Array1::from_iter(
        transformed_lower
            .iter()
            .zip(transformed_upper.iter())
            .map(|(&lo, &hi)| lo.min(hi)),
    );
    let mut mean_upper = Array1::from_iter(
        transformed_lower
            .iter()
            .zip(transformed_upper.iter())
            .map(|(&lo, &hi)| lo.max(hi)),
    );
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
    /// safe) and clamp to `bounds`.
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

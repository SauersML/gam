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
use crate::inference::predict::{
    InferenceCovarianceMode, PredictPosteriorMeanResult, PredictUncertaintyResult,
};
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

/// Optional observation (prediction) interval half-width `z·σ` on the response
/// scale, added to / subtracted from the point prediction. `None` for families
/// that do not expose an observation-scale noise term.
pub struct ObservationInterval<'a> {
    /// Per-row response-scale noise standard deviation.
    pub noise_sd: &'a Array1<f64>,
}

/// Static metadata threaded into every [`PredictUncertaintyResult`].
///
/// These fields are pure provenance — the requested covariance mode and whether
/// a smoothing-corrected covariance was actually used — and are copied verbatim
/// into the result so the engine, not each predictor, owns the struct shape.
pub struct UncertaintyProvenance {
    pub covariance_mode_requested: InferenceCovarianceMode,
    pub covariance_corrected_used: bool,
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
#[allow(clippy::too_many_arguments)]
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
        Some(obs) => {
            let half = obs.noise_sd.mapv(|s| z * s);
            (Some(&mean - &half), Some(&mean + &half))
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
        covariance_mode_requested: provenance.covariance_mode_requested,
        covariance_corrected_used: provenance.covariance_corrected_used,
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
                covariance_mode_requested: InferenceCovarianceMode::Conditional,
                covariance_corrected_used: false,
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
        assert!(!out.covariance_corrected_used);
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
                covariance_mode_requested: InferenceCovarianceMode::Conditional,
                covariance_corrected_used: false,
            },
        )
        .expect("engine assembly");

        assert_close(&out.eta_lower, &ref_eta_lower, "eta lower");
        assert_close(&out.eta_upper, &ref_eta_upper, "eta upper");
        assert_close(&out.mean_lower, &ref_mean_lower, "mean lower");
        assert_close(&out.mean_upper, &ref_mean_upper, "mean upper");
    }

    /// GaussianLocationScalePredictor shape: identity link (mean interval ==
    /// η interval) plus an observation interval `μ ± z·σ`.
    #[test]
    fn identity_eta_with_observation_matches_inline() {
        let eta = array![1.0, 2.0, -1.0];
        let mean = eta.clone();
        let eta_se = array![0.3, 0.1, 0.25];
        let sigma = array![0.5, 0.4, 0.6];
        let z = z95();

        let ref_eta_lower = &eta - &eta_se.mapv(|s| z * s);
        let ref_eta_upper = &eta + &eta_se.mapv(|s| z * s);
        let ref_obs_lower = &mean - &sigma.mapv(|s| z * s);
        let ref_obs_upper = &mean + &sigma.mapv(|s| z * s);

        let out = assemble_uncertainty_result(
            LEVEL,
            eta.clone(),
            mean.clone(),
            eta_se.clone(),
            eta_se.clone(),
            EtaInterval::Symmetric,
            MeanBoundMethod::IdentityEta,
            Some(ObservationInterval { noise_sd: &sigma }),
            UncertaintyProvenance {
                covariance_mode_requested: InferenceCovarianceMode::Conditional,
                covariance_corrected_used: false,
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
                covariance_mode_requested: InferenceCovarianceMode::Conditional,
                covariance_corrected_used: false,
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
            mean_lower: None,
            mean_upper: None,
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
            mean_lower: None,
            mean_upper: None,
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
                covariance_mode_requested: InferenceCovarianceMode::Conditional,
                covariance_corrected_used: false,
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

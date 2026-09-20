//! Reference law of Wald pivots built from a fit's published covariance.
//!
//! Every confidence surface that prices a band as `estimate ± critical·SE`
//! (predict intervals, difference-smooth and other effect bands) reads its
//! critical value from this one type, so a fit with an estimated dispersion is
//! never reported against a normal reference on one surface and a Student-t
//! reference on another.

use gam_problem::EstimationError;
use gam_solve::model_types::UnifiedFitResult;

/// Reference law of the interval pivot `(θ − θ̂) / SE(θ̂)`.
///
/// When the covariance carries a known scale the pivot is standard normal.
/// When it carries a scale `φ̂` estimated from the same data, `SE(θ̂)²` is
/// `φ̂·c` and the pivot is a normal divided by `√(φ̂/φ)`. Marginalizing `φ`
/// under its reference prior `1/φ` (in the unpenalized Gaussian linear model
/// this is the exact sampling law `(n − p)·φ̂/φ ~ χ²_{n−p}`, with `edf` in the
/// role of `p` for a penalized fit) turns the normal pivot into Student-t on
/// `ν = n − edf` degrees of freedom. A normal quantile there ignores the
/// sampling variability of `φ̂` and is too narrow at small `n`.
///
/// This is the same reference the fit's own Wald summary reads — the t/F
/// tests use `wald_scale_is_estimated` and `wald_residual_degrees_of_freedom`
/// — so a reported interval and its test stay dual: a coefficient's 95%
/// interval excludes zero exactly when its two-sided p-value is below 0.05.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IntervalReference {
    /// Known scale: `Φ`.
    Normal,
    /// Estimated scale: Student-t on the fit's residual degrees of freedom.
    StudentT { degrees_of_freedom: f64 },
}

impl IntervalReference {
    /// The reference owned by a fitted model: Student-t on `n − edf` when the
    /// fit's covariance is scaled by an estimated dispersion, normal otherwise.
    ///
    /// An estimated scale without positive residual degrees of freedom has no
    /// interval reference (`edf ≥ n` leaves nothing to estimate `φ` from); that
    /// is reported rather than silently read as a known scale.
    ///
    /// A fit with a modeled noise block (location-scale) has no such scalar:
    /// `σ(x)` is a block of the joint posterior whose uncertainty is already in
    /// the covariance, and no single residual-profiled `φ̂` multiplies that
    /// covariance, so its pivot is normal whatever the family's scale tag.
    pub fn of_fit(fit: &UnifiedFitResult) -> Result<Self, EstimationError> {
        let noise_is_modeled = fit
            .blocks
            .iter()
            .any(|block| block.role == gam_problem::BlockRole::Scale);
        if noise_is_modeled || !fit.likelihood_scale.wald_scale_is_estimated() {
            return Ok(Self::Normal);
        }
        let degrees_of_freedom = fit.wald_residual_degrees_of_freedom().ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "interval reference: the fitted scale is estimated ({:?}) but the fit has no \
                 positive residual degrees of freedom n - edf (n = {}, edf = {:?}), so the \
                 Student-t reference for its intervals is undefined",
                fit.likelihood_scale,
                fit.training_sample_size(),
                fit.edf_total(),
            ))
        })?;
        Ok(Self::StudentT { degrees_of_freedom })
    }

    /// Reference quantile `F⁻¹(p)` for `p ∈ (0, 1)`.
    pub fn quantile(self, p: f64) -> Result<f64, EstimationError> {
        match self {
            Self::Normal => gam_math::probability::standard_normal_quantile(p),
            Self::StudentT { degrees_of_freedom } => {
                gam_math::probability::student_t_quantile(p, degrees_of_freedom)
            }
        }
        .map_err(EstimationError::InvalidInput)
    }

    /// Reference CDF `F(x)`. Each tail is read from the function that computes
    /// it, never as one minus the opposite tail.
    pub fn cdf(self, x: f64) -> f64 {
        match self {
            Self::Normal => gam_math::probability::normal_cdf(x),
            Self::StudentT { degrees_of_freedom } => {
                let tail = 0.5
                    * gam_math::probability::student_t_two_sided_probability(x, degrees_of_freedom);
                if x <= 0.0 { tail } else { 1.0 - tail }
            }
        }
    }

    /// The central two-sided multiplier `F⁻¹(½ + ½·level)` for a confidence
    /// `level ∈ (0, 1)`.
    ///
    /// This is the single source of truth for the confidence-level convention:
    /// every predictor's interval and every pointwise effect band routes its
    /// quantile through here so the convention cannot diverge.
    pub fn central_multiplier(self, level: f64) -> Result<f64, EstimationError> {
        if !(level.is_finite() && level > 0.0 && level < 1.0) {
            return Err(EstimationError::InvalidInput(format!(
                "confidence_level must be in (0,1), got {level}"
            )));
        }
        self.quantile(0.5 + 0.5 * level)
    }
}

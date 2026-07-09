//! Distribution-free conformal calibration of prediction intervals.
//!
//! Split-conformal (and its heteroscedastic generalization, conformalized
//! scale regression — Romano, Patterson & Candès 2019) turns any point
//! predictor into intervals with *finite-sample marginal coverage* on a
//! genuinely held-out calibration fold:
//!
//!   P(Y ∈ interval) ≥ 1 − α
//!
//! that holds regardless of model misspecification, given only that the
//! calibration scores are exchangeable with the test score.
//!
//! # The math
//!
//! Given held-out residuals `r_i` and nonnegative raw per-point scales
//! `s_i`, first apply the shared effective-scale map
//!
//!   s_eff(x) = max(s(x), f64::MIN_POSITIVE),
//!
//! after rejecting non-finite or negative scales. Then form nonconformity scores
//!
//!   e_i = |r_i| / s_eff_i.
//!
//! The conformal multiplier is the EXACT order statistic
//!
//!   q̂ = the ⌈(n+1)(1−α)⌉-th smallest of {e_i}        (1-based rank)
//!
//! and if `rank > n` we return `+∞` — the calibration set is too small to
//! certify coverage at the requested level, so the only honest interval is
//! the unbounded one (never a silently under-covering finite interval). The
//! calibrated interval for a test point with point prediction `μ̂(x)` and
//! raw scale `s(x)` is
//!
//!   μ̂(x) ± q̂ · s_eff(x).
//!
//! With `s_i ≡ 1` this is classic absolute-residual split conformal; with
//! heteroscedastic `s_i` it is conformalized scale regression.
//!
//! CRITICAL: this uses the EXACT k-th order statistic from
//! [`gam_math::quantile::order_statistic`]. It deliberately does NOT use
//! the interpolating [`gam_math::quantile::quantile_from_sorted`] — linear
//! interpolation between order statistics would void the finite-sample coverage
//! proof.
//!
//! # Where the calibration lives, and how it is wired
//!
//! Conformal calibration is a *post-fit* operation: a single scalar `q̂` is
//! derived once and then applied per prediction. There are two ways to build
//! it, matching the two exchangeability regimes:
//!
//! * **Held-out fold (the predict-path default).** When the calibration data
//!   were NOT used to fit the model — a genuinely held-out, labeled fold — the
//!   fitted predictor is already independent of every calibration point, so no
//!   leave-one-out correction is needed. The score is the plain held-out
//!   residual `r_i = y_cal_i − μ̂(x_cal_i)` normalized by the effective scale
//!   derived from the model's predict-time response-scale SE `s(x_cal_i)`.
//!   This is
//!   [`ConformalCalibrator::from_held_out_fold`], driven by
//!   [`gam_predict::predict_full_uncertainty_conformal`] over a
//!   [`gam_predict::ConformalCalibrationFold`]. The fold carries
//!   its own design and may be of ANY size, fully decoupled from the training
//!   rows.
//! * **In-sample (no held-out fold available).** When the only data are the
//!   training set, [`ConformalCalibrator::from_fit`] uses the
//!   first-order approximate-leave-one-out diagnostics in
//!   [`gam_solve::inference::alo`] to manufacture leave-one-out residuals from the
//!   training rows. This is a calibrated heuristic: it inherits the split
//!   conformal finite-sample guarantee only to the extent that the approximate
//!   ALO scores match true leave-one-out exchangeable scores; there is no
//!   separate distribution-free finite-sample theorem for the approximation.
//!
//! Either way the predict path consumes `q̂` through the opt-in
//! `conformal_level` field on
//! [`gam_predict::PredictUncertaintyOptions`], which calls
//! [`ConformalCalibrator::apply_to_uncertainty_result`] to overwrite the
//! model-based response-scale bounds with the conformal ones.
//!
//! # Response scale vs. link scale
//!
//! The interval the caller consumes is on the *response* scale
//! (`μ̂ ± q̂·s_eff`), and the held-out split-conformal coverage guarantee is a
//! statement about `Y` on the response scale. We therefore build the
//! calibration scores on the response scale:
//!
//!   r_i = y_i − g⁻¹(η̃_i)
//!
//! where `η̃_i` is the leave-one-out linear predictor and `g⁻¹` the family
//! inverse link, and the per-point scale is the response-scale SE
//! `s_i = |dμ/dη|·se_bayes_i`. At predict time the matching raw scale `s(x)` is
//! the response-scale standard error the predict engine already produces
//! (`mean_standard_error`), passed through the same effective-scale map.
//! Calibration and prediction thus draw their scale from the *same* source and
//! transform it identically, which is exactly what conformalized scale
//! regression requires. For Gaussian-identity the response scale is exact; for
//! the monotone links gam fits (logit/probit/cloglog/log) the response map is
//! monotone so the symmetric `±q̂·s_eff(x)` interval is well defined, and it is
//! finally clamped to the family support — keeping the held-out split guarantee
//! about `Y` directly rather than relying on a delta-method linearization that
//! the coverage proof does not need.

use crate::PredictUncertaintyResult;
use crate::interval_policy::ResponseBounds;
use gam_math::quantile::order_statistic;
use gam_models::family_runtime::FamilyStrategy;
use gam_models::family_runtime::strategy_for_spec;
use gam_problem::EstimationError;
use gam_solve::inference::alo::compute_alo_diagnostics_from_unified;
use gam_solve::model_types::UnifiedFitResult;
use gam_spec::{LikelihoodSpec, LinkFunction};
use ndarray::{Array1, Array2, ArrayView1};

fn effective_scale(scale: f64, idx: usize, role: &str) -> Result<f64, EstimationError> {
    if !(scale.is_finite() && scale >= 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "{role}[{idx}] must be finite and nonnegative, got {scale}"
        )));
    }
    Ok(scale.max(f64::MIN_POSITIVE))
}

/// The conformal nonconformity scores `e_i = |r_i| / s_eff_i` for held-out
/// residuals `r_i` and nonnegative per-point raw scales `s_i`.
///
/// Validates that the two slices have equal length, that every raw scale is
/// finite and nonnegative, and that every residual is finite. Zero raw scales
/// are mapped to `f64::MIN_POSITIVE` by the same effective-scale transform used
/// at prediction time. Returns an `EstimationError::InvalidInput` otherwise —
/// never a silently degenerate score vector.
pub fn nonconformity_scores(
    residuals: ArrayView1<'_, f64>,
    scales: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, EstimationError> {
    if residuals.len() != scales.len() {
        return Err(EstimationError::InvalidInput(format!(
            "conformal calibration requires residuals and scales of equal length, \
             got {} residuals and {} scales",
            residuals.len(),
            scales.len()
        )));
    }
    if residuals.is_empty() {
        return Err(EstimationError::InvalidInput(
            "conformal calibration requires at least one held-out residual".to_string(),
        ));
    }
    let mut scores = Array1::<f64>::zeros(residuals.len());
    for (idx, (&r, &s)) in residuals.iter().zip(scales.iter()).enumerate() {
        if !r.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "conformal residual[{idx}] must be finite, got {r}"
            )));
        }
        let s_eff = effective_scale(s, idx, "conformal scale")?;
        scores[idx] = r.abs() / s_eff;
    }
    Ok(scores)
}

/// The EXACT conformal multiplier `q̂` at miscoverage `α ∈ (0, 1)`.
///
/// `q̂` is the `⌈(n+1)(1−α)⌉`-th smallest score (1-based rank). When that rank
/// exceeds `n` the calibration set is too small to certify `1 − α` coverage,
/// and the only honest multiplier is `+∞` (the unbounded interval). Uses the
/// exact order statistic — no interpolation — so the finite-sample coverage
/// guarantee is preserved.
pub fn conformal_multiplier(
    scores: ArrayView1<'_, f64>,
    alpha: f64,
) -> Result<f64, EstimationError> {
    if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
        return Err(EstimationError::InvalidInput(format!(
            "conformal miscoverage alpha must be in (0,1), got {alpha}"
        )));
    }
    let n = scores.len();
    if n == 0 {
        return Err(EstimationError::InvalidInput(
            "conformal multiplier requires at least one nonconformity score".to_string(),
        ));
    }
    for (idx, &e) in scores.iter().enumerate() {
        if !e.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "conformal score[{idx}] must be finite, got {e}"
            )));
        }
    }
    // 1-based rank ⌈(n+1)(1−α)⌉.
    let rank = ((n as f64 + 1.0) * (1.0 - alpha)).ceil() as usize;
    if rank > n {
        // Too few calibration points to certify coverage at this level.
        return Ok(f64::INFINITY);
    }
    let values: Vec<f64> = scores.iter().copied().collect();
    Ok(order_statistic(&values, rank))
}

/// A fitted conformal calibrator: the single scalar `q̂` and the miscoverage
/// `α` it was computed at. Built once from a fit + training data, then applied
/// per prediction.
#[derive(Clone, Copy, Debug)]
pub struct ConformalCalibrator {
    q_hat: f64,
    alpha: f64,
    n_calibration: usize,
}

impl ConformalCalibrator {
    /// The conformal multiplier `q̂`.
    pub fn q_hat(&self) -> f64 {
        self.q_hat
    }

    /// The nominal miscoverage `α` (so the nominal coverage is `1 − α`).
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// The number of held-out calibration points behind `q̂`.
    pub fn n_calibration(&self) -> usize {
        self.n_calibration
    }

    /// Whether the calibration set was large enough to certify finite
    /// intervals at the requested level (`q̂` finite). When `false` the
    /// honest interval is unbounded.
    pub fn certifies_finite(&self) -> bool {
        self.q_hat.is_finite()
    }

    /// Build a calibrator directly from held-out residuals and per-point
    /// raw scales. This is the pure core both [`ConformalCalibrator::from_fit`]
    /// and the e2e tests route through.
    pub fn from_residuals_and_scales(
        residuals: ArrayView1<'_, f64>,
        scales: ArrayView1<'_, f64>,
        alpha: f64,
    ) -> Result<Self, EstimationError> {
        let scores = nonconformity_scores(residuals, scales)?;
        let q_hat = conformal_multiplier(scores.view(), alpha)?;
        Ok(Self {
            q_hat,
            alpha,
            n_calibration: scores.len(),
        })
    }

    /// Build a calibrator from a genuinely held-out calibration fold.
    ///
    /// This is the correct split-conformal path when the calibration data were
    /// NOT used to fit the model: a held-out fold needs no leave-one-out
    /// correction, because the fitted predictor is already independent of every
    /// calibration point. The honest nonconformity score is the *plain*
    /// held-out residual on the response scale,
    ///
    ///   r_i = y_cal_i − μ̂(x_cal_i),
    ///
    /// normalized (for conformalized scale regression) by the effective scale
    /// derived from the model's own predict-time response-scale standard error
    /// `s_i = s(x_cal_i)` — the SAME scale source and transform applied at test
    /// time by
    /// [`Self::apply_to_uncertainty_result`]. With those scores the exact
    /// order-statistic multiplier `q̂` gives finite-sample marginal coverage
    /// `P(Y ∈ μ̂(x) ± q̂·s_eff(x)) ≥ 1 − α` (Vovk et al.; Romano, Patterson &
    /// Candès 2019), provided the calibration and test points are exchangeable
    /// — which they are for an i.i.d. held-out fold.
    ///
    /// `mu_cal` and `scale_cal` are the model's predict-time response mean and
    /// response-scale SE at the calibration design points (produced by exactly
    /// the same predict engine used for the test points), and `y_cal` is the
    /// held-out response. No fit geometry, no ALO, and no binding of the fold
    /// to the training rows is involved — so a calibration fold of any size is
    /// accepted.
    pub fn from_held_out_fold(
        y_cal: ArrayView1<'_, f64>,
        mu_cal: ArrayView1<'_, f64>,
        scale_cal: ArrayView1<'_, f64>,
        alpha: f64,
    ) -> Result<Self, EstimationError> {
        if y_cal.len() != mu_cal.len() || y_cal.len() != scale_cal.len() {
            return Err(EstimationError::InvalidInput(format!(
                "conformal held-out calibration requires y, mean, and scale of equal length, \
                 got {} responses, {} means, {} scales",
                y_cal.len(),
                mu_cal.len(),
                scale_cal.len()
            )));
        }
        let n = y_cal.len();
        let mut residuals = Array1::<f64>::zeros(n);
        let mut scales = Array1::<f64>::zeros(n);
        for i in 0..n {
            residuals[i] = y_cal[i] - mu_cal[i];
            scales[i] = effective_scale(scale_cal[i], i, "conformal calibration scale")?;
        }
        Self::from_residuals_and_scales(residuals.view(), scales.view(), alpha)
    }

    /// Build a calibrator from a fitted model and its training data.
    ///
    /// Computes first-order approximate-leave-one-out diagnostics (held-out
    /// linear predictors `η̃_i` and per-point posterior SE `se_bayes_i`), maps
    /// both onto the response scale through the fitted family's inverse link,
    /// forms the response-scale held-out residuals `r_i = y_i − g⁻¹(η̃_i)` and
    /// scales `s_i = |dμ/dη(η̃_i)| · se_bayes_i`, applies the shared
    /// effective-scale transform, and returns the resulting `q̂`.
    ///
    /// This ALO path is a calibrated heuristic. It has the split-conformal
    /// finite-sample marginal coverage guarantee only insofar as these
    /// first-order ALO scores match true leave-one-out exchangeable scores; it
    /// is not itself a distribution-free finite-sample guarantee.
    ///
    /// `design`, `offset`, `phi` mirror the arguments
    /// [`compute_alo_diagnostics_from_unified`] requires; `eta` is the fitted
    /// in-sample linear predictor `Xβ̂ + offset`.
    pub fn from_fit(
        fit: &UnifiedFitResult,
        family: &LikelihoodSpec,
        design: &Array2<f64>,
        eta: &Array1<f64>,
        offset: &Array1<f64>,
        y: ArrayView1<'_, f64>,
        phi: f64,
        alpha: f64,
    ) -> Result<Self, EstimationError> {
        let link: LinkFunction = family.link.link_function();
        let alo = compute_alo_diagnostics_from_unified(fit, design, eta, offset, link, phi)?;
        if alo.eta_tilde.len() != y.len() {
            return Err(EstimationError::InvalidInput(format!(
                "conformal calibration: ALO produced {} held-out predictors but y has length {}",
                alo.eta_tilde.len(),
                y.len()
            )));
        }
        let strategy = strategy_for_spec(family);
        let n = y.len();
        let mut residuals = Array1::<f64>::zeros(n);
        let mut scales = Array1::<f64>::zeros(n);
        for i in 0..n {
            let eta_tilde = alo.eta_tilde[i];
            let jet = strategy.inverse_link_jet(eta_tilde)?;
            let mu_tilde = jet.mu;
            // Response-scale held-out residual.
            residuals[i] = y[i] - mu_tilde;
            // Response-scale held-out SE: |dμ/dη| · se_bayes (delta method on
            // the held-out posterior SE), then the same effective-scale map
            // used by the prediction interval.
            let dmu_deta = jet.d1.abs();
            let scale = effective_scale(
                dmu_deta * alo.se_bayes[i],
                i,
                "conformal ALO response-scale SE",
            )?;
            scales[i] = scale;
        }
        Self::from_residuals_and_scales(residuals.view(), scales.view(), alpha)
    }

    /// Calibrated response-scale interval `μ̂(x) ± q̂·s_eff(x)`, clamped to the
    /// family support `bounds`.
    ///
    /// `mean` is the point prediction `μ̂(x)` and `scale` the raw response-scale
    /// SE `s(x)` at each test point. Each raw scale must be finite and
    /// nonnegative; zero is mapped to `f64::MIN_POSITIVE`, exactly as in
    /// calibration. When `q̂ = +∞` the interval is unbounded (`(−∞, +∞)`, then
    /// clamped to the support) — the honest answer when the calibration set
    /// could not certify coverage.
    pub fn calibrated_interval(
        &self,
        mean: &Array1<f64>,
        scale: &Array1<f64>,
        bounds: ResponseBounds,
    ) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
        if mean.len() != scale.len() {
            return Err(EstimationError::InvalidInput(format!(
                "conformal interval requires mean and scale of equal length, \
                 got {} means and {} scales",
                mean.len(),
                scale.len()
            )));
        }
        let mut lower = Array1::<f64>::zeros(mean.len());
        let mut upper = Array1::<f64>::zeros(mean.len());
        for i in 0..mean.len() {
            let s_eff = effective_scale(scale[i], i, "conformal prediction scale")?;
            let half = self.q_hat * s_eff;
            // q̂·s_eff with q̂ = +∞ yields ±∞; bounds.clamp_value then maps that
            // onto the support (or leaves it unbounded).
            lower[i] = bounds.clamp_value(mean[i] - half);
            upper[i] = bounds.clamp_value(mean[i] + half);
        }
        Ok((lower, upper))
    }

    /// Overwrite the response-scale bounds of a model-based
    /// [`PredictUncertaintyResult`] with the conformal interval, using the
    /// result's own `mean` and `mean_standard_error` (the same response-scale
    /// SE source the calibration scales came from). This is the real
    /// predict-path application: the model-based point/SE are kept, only the
    /// `mean_lower` / `mean_upper` bounds become the conformal ones.
    pub fn apply_to_uncertainty_result(
        &self,
        result: &mut PredictUncertaintyResult,
        bounds: ResponseBounds,
    ) -> Result<(), EstimationError> {
        let (lower, upper) =
            self.calibrated_interval(&result.mean, &result.mean_standard_error, bounds)?;
        result.mean_lower = lower;
        result.mean_upper = upper;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn scores_are_abs_residual_over_scale() {
        let r = array![2.0, -4.0, 1.0];
        let s = array![1.0, 2.0, 0.5];
        let e = nonconformity_scores(r.view(), s.view()).expect("valid scores");
        assert_eq!(e, array![2.0, 2.0, 2.0]);
    }

    #[test]
    fn floors_zero_scale_and_rejects_invalid_scale() {
        let r = array![1.0, 2.0];
        let s = array![1.0, 0.0];
        let e = nonconformity_scores(r.view(), s.view()).expect("zero scale is floored");
        assert_eq!(e[0], 1.0);
        assert_eq!(e[1], 2.0 / f64::MIN_POSITIVE);

        let negative = array![1.0, -1.0];
        assert!(nonconformity_scores(r.view(), negative.view()).is_err());
        let nonfinite = array![1.0, f64::NAN];
        assert!(nonconformity_scores(r.view(), nonfinite.view()).is_err());
    }

    #[test]
    fn rejects_nonfinite_residual() {
        let r = array![1.0, f64::NAN];
        let s = array![1.0, 1.0];
        assert!(nonconformity_scores(r.view(), s.view()).is_err());
    }

    #[test]
    fn multiplier_is_exact_order_statistic() {
        // n = 9 scores 1..=9. With alpha = 0.1, rank = ceil(10 * 0.9) = 9,
        // so q̂ is the 9th smallest = 9.
        let scores = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let q = conformal_multiplier(scores.view(), 0.1).expect("valid");
        assert_eq!(q, 9.0);

        // alpha = 0.25, rank = ceil(10 * 0.75) = 8 → 8th smallest = 8.
        let q2 = conformal_multiplier(scores.view(), 0.25).expect("valid");
        assert_eq!(q2, 8.0);
    }

    #[test]
    fn multiplier_does_not_interpolate() {
        // Unequally spaced scores: the exact order statistic must be one of the
        // observed values, never an interpolated value between two of them.
        let scores = array![0.0, 10.0, 100.0, 1000.0];
        // n = 4, alpha = 0.4 → rank = ceil(5 * 0.6) = 3 → 3rd smallest = 100.
        let q = conformal_multiplier(scores.view(), 0.4).expect("valid");
        assert_eq!(q, 100.0);
    }

    #[test]
    fn too_few_points_returns_infinity() {
        // n = 4, alpha = 0.05 → rank = ceil(5 * 0.95) = 5 > 4 → +∞.
        let scores = array![1.0, 2.0, 3.0, 4.0];
        let q = conformal_multiplier(scores.view(), 0.05).expect("valid");
        assert!(q.is_infinite());

        let calib =
            ConformalCalibrator::from_residuals_and_scales(scores.view(), scores.view(), 0.05)
                .expect("valid");
        assert!(!calib.certifies_finite());
    }

    #[test]
    fn rejects_alpha_out_of_range() {
        let scores = array![1.0, 2.0, 3.0];
        assert!(conformal_multiplier(scores.view(), 0.0).is_err());
        assert!(conformal_multiplier(scores.view(), 1.0).is_err());
        assert!(conformal_multiplier(scores.view(), -0.1).is_err());
    }

    #[test]
    fn calibrated_interval_is_symmetric_and_clamped() {
        let calib = ConformalCalibrator::from_residuals_and_scales(
            array![1.0].view(),
            array![1.0].view(),
            0.5,
        )
        .expect("valid");
        // n = 1, alpha = 0.5 → rank = ceil(2 * 0.5) = 1 → q̂ = score = 1.0.
        assert_eq!(calib.q_hat(), 1.0);
        let mean = array![0.5, 2.0];
        let scale = array![0.1, 0.2];
        let (lo, hi) = calib
            .calibrated_interval(&mean, &scale, ResponseBounds::UNBOUNDED)
            .expect("interval");
        assert!((lo[0] - 0.4).abs() < 1e-12);
        assert!((hi[0] - 0.6).abs() < 1e-12);

        // Clamp to [0, 1].
        let (lo_c, hi_c) = calib
            .calibrated_interval(&mean, &scale, ResponseBounds::UNIT_PROBABILITY)
            .expect("interval");
        assert!(lo_c.iter().all(|&v| v >= 0.0));
        assert!(hi_c.iter().all(|&v| v <= 1.0));
    }

    #[test]
    fn zero_scale_uses_same_effective_scale_for_calibration_and_prediction() {
        let calib = ConformalCalibrator::from_held_out_fold(
            array![1.0].view(),
            array![0.0].view(),
            array![0.0].view(),
            0.5,
        )
        .expect("zero calibration scale is floored");
        assert_eq!(calib.q_hat(), 1.0 / f64::MIN_POSITIVE);

        let mean = array![10.0];
        let scale = array![0.0];
        let (lo, hi) = calib
            .calibrated_interval(&mean, &scale, ResponseBounds::UNBOUNDED)
            .expect("zero prediction scale is floored");
        assert!((lo[0] - 9.0).abs() < 1e-12);
        assert!((hi[0] - 11.0).abs() < 1e-12);
    }

    #[test]
    fn calibrated_interval_rejects_negative_prediction_scale() {
        let calib = ConformalCalibrator::from_residuals_and_scales(
            array![1.0].view(),
            array![1.0].view(),
            0.5,
        )
        .expect("valid");
        let mean = array![0.0];
        let scale = array![-0.1];
        assert!(
            calib
                .calibrated_interval(&mean, &scale, ResponseBounds::UNBOUNDED)
                .is_err()
        );
    }
}

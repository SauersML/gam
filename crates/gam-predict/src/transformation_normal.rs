use super::*;
use crate::input::{TRANSFORMATION_NORMAL_BAND_Z_MAX, TRANSFORMATION_NORMAL_BAND_Z_NODES};

/// Predictor for transformation-normal (CTM) models.
///
/// `build_predict_input_for_model` (issue #1612) precomputes the plug-in
/// response-scale conditional mean `E_{Z~N(0,1)}[h⁻¹(Z|x; β̂)]` into the
/// PredictInput offset, and its Laplace-order posterior-mean correction into
/// `auxiliary_scalar` (SPEC rule 3). Both are functions of the covariates alone,
/// so prediction is covariate-only and does not require the outcome column. The
/// default posterior-mean pass reports their sum as both the linear predictor and
/// the mean (eta = mean = E[Y|x]). The explicit plug-in pass reports the offset
/// alone.
///
/// ## Uncertainty contract
///
/// * **Epistemic (coefficient) uncertainty is reported as unavailable, never
///   as zero.** Propagating `Cov(β)` into `E[Y|x]` requires the Jacobian of
///   the inverse transform `∂h⁻¹/∂β`, which needs the I-spline basis partials
///   that are not part of the persisted quantile grid. A zero SE claims exact
///   knowledge of `E[Y|x]` the posterior does not have, so the point paths
///   return `None` SEs and `predict_full_uncertainty` errors instead of
///   emitting zero-width intervals.
/// * **Observation (predictive) intervals are posterior-predictive quantiles.**
///   Under the coefficient posterior the transform at a response is Gaussian,
///   `h(t) ~ N(ĥ(t), s²(t))`, so the predictive CDF is `Φ(g)` with
///   `g = ĥ/√(1 + s²)` and its `p`-quantile is `g⁻¹(Φ⁻¹(p)|x)`. The input builder
///   tabulates `g⁻¹` on a fixed latent-z ladder (`PredictInput::auxiliary_matrix`);
///   the band interpolates that ladder, and refuses a level the predictive CDF
///   does not reach or a row where it is not a distribution function. Adding
///   standard-normal quantiles to `E[Y|x]` directly would be off by exactly the
///   (row-dependent) scale of `h⁻¹` — for `h(y) = 10·y` the plug-in 95% band is
///   `±0.196`, not `±1.96`.
pub(crate) struct TransformationNormalPredictor;

/// Why a posterior-predictive band cannot be read off a ladder row.
#[derive(Debug)]
enum LadderGap {
    /// `Φ(ĥ/√(1 + s²))` is not monotone at the row, so it has no quantiles.
    NotADistribution,
    /// The requested level needs a node whose level the predictive CDF does not
    /// reach.
    Unattainable,
}

fn ladder_gap(nodes: &[f64]) -> Option<LadderGap> {
    if nodes.iter().any(|value| value.is_nan()) {
        Some(LadderGap::NotADistribution)
    } else if nodes.iter().any(|value| value.is_infinite()) {
        Some(LadderGap::Unattainable)
    } else {
        None
    }
}

/// Interpolate one row of the tabulated response-quantile ladder at an arbitrary
/// latent value `z`. The row holds `m` node values `Q[j] = g⁻¹(z_j | x)`, the
/// posterior-predictive quantiles on the fixed even grid from
/// `transformation_normal_band_z_nodes`, followed by their exact latent slopes
/// `Q'[j] = 1 / g′(Q[j] | x)`, which the input builder reads off the same
/// interpolant it inverts. The slope of a function we hold is its derivative,
/// never a difference of its samples (SPEC rule 2).
///
/// A node the builder could not tabulate is published in the row: `±∞` where the
/// predictive CDF never reaches the node's level, NaN across a row where that CDF
/// is not monotone. A cell or continuation that reads one is refused as a
/// [`LadderGap`] instead of interpolated.
///
/// Two things this must not do, both of which it used to (gam#2600):
///
/// * **Clamp past the ends.** A requested level with `|Φ⁻¹(p)| > z_max` used to
///   return the outermost tabulated quantile, so every band beyond 99.994 % was
///   the same interval. Past `±z_max` the band continues at the end node's exact
///   slope, a first-order continuation, which is strictly better than a
///   constant.
/// * **Interpolate a curved quantile function with a chord.** `Q` is `h⁻¹`
///   sampled every `2·z_max/(m−1) = 0.25` in the latent, and `h⁻¹` is as curved
///   as the response is skewed — for a lognormal response `d²y/dz² = y`, so a
///   chord carries `O(Δz²·y/8) ≈ 8e-3·y`. The cubic Hermite through the same
///   nodes with their exact slopes is fourth order (`O(Δz⁴·y/384) ≈ 1e-5·y`). A
///   cell whose end slopes could make that cubic overshoot (either slope above
///   three secants, the Fritsch–Carlson bound) falls back to its chord, so the
///   band stays ordered; the secant decides only that and is never a derivative.
fn ladder_quantile(ladder_row: ndarray::ArrayView1<'_, f64>, z: f64) -> Result<f64, LadderGap> {
    let m = TRANSFORMATION_NORMAL_BAND_Z_NODES;
    assert_eq!(
        ladder_row.len(),
        2 * m,
        "quantile ladder row must hold node values then slopes on the fixed even z grid"
    );
    let values = ladder_row.slice(ndarray::s![..m]);
    let slopes = ladder_row.slice(ndarray::s![m..]);
    let z_max = TRANSFORMATION_NORMAL_BAND_Z_MAX;
    let step = 2.0 * z_max / ((m - 1) as f64);
    let t = (z + z_max) / step;
    if t <= 0.0 {
        if let Some(gap) = ladder_gap(&[values[0], slopes[0]]) {
            return Err(gap);
        }
        return Ok(values[0] + (z + z_max) * slopes[0]);
    }
    if t >= (m - 1) as f64 {
        if let Some(gap) = ladder_gap(&[values[m - 1], slopes[m - 1]]) {
            return Err(gap);
        }
        return Ok(values[m - 1] + (z - z_max) * slopes[m - 1]);
    }
    let j = t.floor() as usize;
    let frac = t - j as f64;
    let (q0, q1) = (values[j], values[j + 1]);
    let (m0, m1) = (slopes[j], slopes[j + 1]);
    if let Some(gap) = ladder_gap(&[q0, q1, m0, m1]) {
        return Err(gap);
    }
    let secant = (q1 - q0) / step;
    if !(secant > 0.0 && m0 <= 3.0 * secant && m1 <= 3.0 * secant) {
        return Ok(q0 + frac * (q1 - q0));
    }
    let (t2, t3) = (frac * frac, frac * frac * frac);
    Ok((2.0 * t3 - 3.0 * t2 + 1.0) * q0
        + (t3 - 2.0 * t2 + frac) * step * m0
        + (-2.0 * t3 + 3.0 * t2) * q1
        + (t3 - t2) * step * m1)
}

/// The Laplace-order posterior mean `E[Y|x]` (SPEC rule 3): the plug-in
/// conditional mean the input builder stores in `offset`, plus the posterior-mean
/// correction it stores in `auxiliary_scalar`. The plug-in pass reads `offset`
/// alone.
fn posterior_mean_response(input: &PredictInput) -> Result<Array1<f64>, EstimationError> {
    let correction = input.auxiliary_scalar.as_ref().ok_or_else(|| {
        EstimationError::InvalidInput(
            "transformation-normal prediction input is missing the posterior-mean correction \
             (auxiliary_scalar)"
                .to_string(),
        )
    })?;
    if correction.len() != input.offset.len() {
        return Err(EstimationError::InvalidInput(format!(
            "transformation-normal posterior-mean correction has {} rows but the plug-in mean \
             has {}",
            correction.len(),
            input.offset.len()
        )));
    }
    Ok(&input.offset + correction)
}

impl PredictionTransform for TransformationNormalPredictor {

    fn response_family(&self) -> ResponseFamily {
        // Only the *latent* `h(y)` is Gaussian. The generic family observation
        // band must never be built from this (its σ lives in latent units);
        // the predictor supplies its own response-scale band from the
        // quantile ladder in `predict_posterior_mean`.
        ResponseFamily::Gaussian
    }

    fn response_jacobian_rows(&self, pass: PredictPass) -> ResponseInterval {
        match pass {
            // `response` is the identity here (the offset already carries the
            // response-scale conditional mean), so there is no link to
            // transform or delta-propagate through in either pass: an η
            // interval already IS the response interval.
            PredictPass::FullUncertainty | PredictPass::PosteriorMean => {
                ResponseInterval::IdentityEta
            }
        }
    }
    fn point_state(&self, input: &PredictInput) -> Result<LinearState, EstimationError> {
        // This is the explicit plug-in state: the offset carries the precomputed
        // plug-in conditional mean `E[Y|x; β̂]`, and the posterior-mean passes add
        // its correction in `posterior_mean_response`. No covariance-propagated
        // SE exists for this quantity (see the struct-level uncertainty contract),
        // so the SEs are `None` — reporting zero would claim certainty the
        // posterior does not have.
        let h = input.offset.clone();
        Ok(LinearState {
            eta: h.clone(),
            mean: h,
            eta_se: None,
            mean_se: None,
            covariance_source: InferenceCovarianceMode::Conditional,
        })
    }

    fn response(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        Ok(eta.clone())
    }

    fn bounds(&self) -> ResponseBounds {
        ResponseBounds::UNBOUNDED
    }

}

impl PredictableModel for TransformationNormalPredictor {
    fn predict_plugin_response(
        &self,
        input: &PredictInput,
    ) -> Result<PredictResult, EstimationError> {
        predict_plugin_response_generic(self, input)
    }

    fn predict_with_uncertainty(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError> {
        // The CTM predictor reports no covariance-derived SEs on the point path;
        // its point is the Laplace-order posterior mean E[Y|x].
        let h = posterior_mean_response(input)?;
        Ok(PredictionWithSE {
            eta: h.clone(),
            mean: h,
            eta_se: None,
            mean_se: None,
        })
    }

    fn predict_full_uncertainty(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError> {
        if !fit.log_likelihood.is_finite() {
            return Err(EstimationError::InvalidInput(
                "transformation-normal predict received a fit with a non-finite log-likelihood"
                    .to_string(),
            ));
        }
        Err(EstimationError::InvalidInput(format!(
            "transformation-normal models cannot report coefficient-uncertainty intervals \
             (level {} requested for {} rows): propagating the coefficient covariance through \
             the inverse transform h⁻¹ requires the I-spline basis Jacobian, which is not part \
             of the persisted quantile grid. Use predict_posterior_mean for the point E[Y|x] \
             and its response-scale observation (predictive) interval.",
            options.confidence_level,
            input.offset.len(),
        )))
    }

    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        options: &PosteriorMeanOptions,
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        // The posterior mean is precomputed by the input builder: the plug-in
        // conditional mean in `offset` plus its Laplace-order correction in
        // `auxiliary_scalar`. The fit contributes no coefficient state here, but a
        // non-finite fitted log-likelihood marks a corrupted payload.
        if !fit.log_likelihood.is_finite() {
            return Err(EstimationError::InvalidInput(
                "transformation-normal predict received a fit with a non-finite log-likelihood"
                    .to_string(),
            ));
        }
        let h = posterior_mean_response(input)?;
        let n = h.len();
        let mut result = PredictPosteriorMeanResult {
            eta: h.clone(),
            // The result struct requires an SE array; epistemic uncertainty is
            // unavailable (see the struct-level contract), so no credible
            // bounds or mean SE are emitted below and this array is inert.
            eta_standard_error: Array1::zeros(n),
            mean: h,
            mean_standard_error: None,
            mean_lower: None,
            mean_upper: None,
            observation_lower: None,
            observation_upper: None,
            point_covariance_source: InferenceCovarianceMode::Conditional,
            uncertainty_covariance_source: None,
            point_covariance_provenance: None,
        };
        if options.include_observation_interval
            && let Some(level) = options.confidence_level
        {
            // The latent `h(Y) − η` is exactly standard normal by construction
            // of the transformation model, so its band is read on `Φ`.
            let z = crate::IntervalReference::Normal.central_multiplier(level)?;
            let ladder = input.auxiliary_matrix.as_ref().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "transformation-normal prediction input is missing the response-scale \
                     quantile ladder (auxiliary_matrix)"
                        .to_string(),
                )
            })?;
            if ladder.nrows() != n || ladder.ncols() != 2 * TRANSFORMATION_NORMAL_BAND_Z_NODES {
                return Err(EstimationError::InvalidInput(format!(
                    "transformation-normal quantile ladder shape mismatch: expected {}x{} \
                     (node values then slopes), got {}x{}",
                    n,
                    2 * TRANSFORMATION_NORMAL_BAND_Z_NODES,
                    ladder.nrows(),
                    ladder.ncols()
                )));
            }
            // Equal-tailed response-scale posterior-predictive band: the p-quantile
            // of `Y|x` is `g⁻¹(Φ⁻¹(p)|x)`, interpolated from the tabulated node
            // values and their exact slopes. `g⁻¹` is monotone increasing wherever
            // it is tabulated and the interpolant is shape-preserving, so the band
            // is ordered by construction.
            let band_limit = |i: usize, target: f64| -> Result<f64, EstimationError> {
                ladder_quantile(ladder.row(i), target).map_err(|gap| {
                    EstimationError::InvalidInput(match gap {
                        LadderGap::NotADistribution => format!(
                            "transformation-normal predictive band at level {level} is undefined \
                             at row {i}: under the Gaussian coefficient posterior the predictive \
                             CDF Φ(ĥ/√(1+s²)) is not monotone there, so it has no quantiles"
                        ),
                        LadderGap::Unattainable => format!(
                            "transformation-normal predictive band at level {level} is not \
                             resolved at row {i}: the predictive CDF Φ(ĥ/√(1+s²)) does not reach \
                             that level within the tabulated ladder, because the Gaussian \
                             coefficient posterior leaves mass on transforms the monotone \
                             likelihood forbids"
                        ),
                    })
                })
            };
            let mut lower = Array1::<f64>::zeros(n);
            let mut upper = Array1::<f64>::zeros(n);
            for i in 0..n {
                lower[i] = band_limit(i, -z)?;
                upper[i] = band_limit(i, z)?;
            }
            result.observation_lower = Some(lower);
            result.observation_upper = Some(upper);
        }
        Ok(result)
    }

    fn n_blocks(&self) -> usize {
        1
    }
    fn block_roles(&self) -> Vec<BlockRole> {
        vec![BlockRole::Mean]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// The ladder a lognormal response actually produces: `h(y) = ln y`, so
    /// `h⁻¹(z) = exp(z)` — smooth, strongly curved, and known in closed form, so
    /// every deviation below is interpolation error and nothing else. The row
    /// holds the node values followed by their exact slopes, which for `exp`
    /// are the values themselves.
    fn exp_ladder() -> Array1<f64> {
        let m = TRANSFORMATION_NORMAL_BAND_Z_NODES;
        let z_max = TRANSFORMATION_NORMAL_BAND_Z_MAX;
        Array1::from_shape_fn(2 * m, |slot| {
            (-z_max + 2.0 * z_max * ((slot % m) as f64) / ((m - 1) as f64)).exp()
        })
    }

    #[test]
    fn band_ladder_does_not_clamp_past_its_own_ends_2600() {
        // A requested level past `z_max` used to return the outermost tabulated
        // quantile, so every band beyond 99.994 % was the same interval.
        let ladder = exp_ladder();
        let m = TRANSFORMATION_NORMAL_BAND_Z_NODES;
        let z_max = TRANSFORMATION_NORMAL_BAND_Z_MAX;
        let end = ladder[m - 1];
        let mut previous = end;
        for &z in &[4.5_f64, 5.0, 6.0] {
            let value = ladder_quantile(ladder.view(), z).expect("a finite ladder resolves every level");
            assert!(
                value > previous,
                "the ladder saturated past its end: q({z}) = {value} <= {previous}"
            );
            previous = value;
        }
        let start = ladder[0];
        let mut previous = start;
        for &z in &[-4.5_f64, -5.0, -6.0] {
            let value = ladder_quantile(ladder.view(), z).expect("a finite ladder resolves every level");
            assert!(
                value < previous,
                "the ladder saturated past its start: q({z}) = {value} >= {previous}"
            );
            previous = value;
        }
        // The continuation is the ladder's own exact end slope.
        let slope = ladder[2 * m - 1];
        let far = ladder_quantile(ladder.view(), z_max + 1.5)
            .expect("a finite ladder continues past its end");
        assert!(
            (far - (end + 1.5 * slope)).abs() < 1e-9,
            "the exterior continuation is not affine at the end slope: {far} vs {}",
            end + 1.5 * slope
        );
    }

    #[test]
    fn band_ladder_interpolation_is_shape_preserving_and_beats_the_chord_2600() {
        let ladder = exp_ladder();
        let m = TRANSFORMATION_NORMAL_BAND_Z_NODES;
        let z_max = TRANSFORMATION_NORMAL_BAND_Z_MAX;
        let step = 2.0 * z_max / ((m - 1) as f64);
        let (mut shaped, mut chord) = (0.0_f64, 0.0_f64);
        let mut previous = f64::NEG_INFINITY;
        for k in 0..=4000 {
            let z = -z_max + 2.0 * z_max * (k as f64) / 4000.0;
            let truth = z.exp();
            let value = ladder_quantile(ladder.view(), z).expect("a finite ladder resolves every level");
            // Monotonicity: the interpolated quantile function IS the band, so
            // an overshoot here is a lower limit above its own upper limit.
            assert!(
                value > previous,
                "the interpolated ladder is not monotone at z={z}: {value} <= {previous}"
            );
            previous = value;
            shaped = shaped.max((value - truth).abs() / truth);
            let t = (z + z_max) / step;
            let j = (t.floor() as usize).min(m - 2);
            let frac = t - (j as f64);
            let linear = ladder[j] + frac * (ladder[j + 1] - ladder[j]);
            chord = chord.max((linear - truth).abs() / truth);
        }
        eprintln!(
            "#2600 ladder: max relative band error  shape-preserving={shaped:.3e}  chord={chord:.3e}"
        );
        assert!(
            shaped * 10.0 < chord,
            "the shape-preserving ladder is not decisively tighter than the chord it replaces: \
             {shaped:.6e} vs {chord:.6e}"
        );
        assert!(
            shaped < 1.0e-3,
            "the reported band carries {shaped:.6e} relative interpolation error"
        );
    }

    #[test]
    fn band_ladder_refuses_unattainable_and_undefined_levels() {
        let m = TRANSFORMATION_NORMAL_BAND_Z_NODES;
        let z_max = TRANSFORMATION_NORMAL_BAND_Z_MAX;
        let mut unattainable = exp_ladder();
        unattainable[m - 1] = f64::INFINITY;
        unattainable[2 * m - 1] = f64::INFINITY;
        assert!(matches!(
            ladder_quantile(unattainable.view(), z_max - 0.1),
            Err(LadderGap::Unattainable)
        ));
        assert!(matches!(
            ladder_quantile(unattainable.view(), z_max + 1.0),
            Err(LadderGap::Unattainable)
        ));
        assert!(ladder_quantile(unattainable.view(), 0.0).is_ok());
        let undefined = Array1::from_elem(2 * m, f64::NAN);
        assert!(matches!(
            ladder_quantile(undefined.view(), 0.0),
            Err(LadderGap::NotADistribution)
        ));
    }

    #[test]
    fn band_ladder_reproduces_its_own_nodes_exactly_2600() {
        let ladder = exp_ladder();
        let m = TRANSFORMATION_NORMAL_BAND_Z_NODES;
        let z_max = TRANSFORMATION_NORMAL_BAND_Z_MAX;
        for j in 0..m {
            let z = -z_max + 2.0 * z_max * (j as f64) / ((m - 1) as f64);
            let value = ladder_quantile(ladder.view(), z).expect("a finite ladder resolves every level");
            assert!(
                (value - ladder[j]).abs() <= 1e-12 * ladder[j].abs().max(1.0),
                "node {j} is not reproduced: {value} vs {}",
                ladder[j]
            );
        }
    }
}

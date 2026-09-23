//! Central posterior bands of the survival surfaces (gam#3560): each band is the pair of
//! quantiles `(1 ∓ level)/2` of the posterior law the published mean integrates, never
//! `mean ± z·sd` clamped to the surface's range. [`super::predict_survival_with_band`] and
//! [`super::predict_competing_risks_with_band`] publish them; this module forms them.
//!
//! - The exact anchored marginal-slope rule inverts the bivariate Gaussian law of its primaries
//!   ([`probit_survival_band`]).
//! - The sigma-point rule carries each cell's index at its posterior mean and standard
//!   deviation and publishes the image of the index's interval ([`SurvivalIndexCell`]).
//! - The competing-risks surfaces invert the joint normal law of every cause's `log H_k` and its
//!   rate ([`competing_risks_bands`]); the cumulative incidence is refused by name
//!   ([`BandRefusal`]).
//! - A cone-truncated posterior is a mixture over its joint rule's nodes, and its band is that
//!   mixture's certified central interval ([`location_scale_truncated_surface_band`],
//!   [`replayed_truncated_surface_band`]).

use super::*;

/// The central interval of the marginal-slope survival `S = Φ(−η(x₀, x₁))` for
/// `(x₀, x₁) ~ N(mu, cov)` at `level` (gam#3560). It is taken on the index `η`
/// and carried through the decreasing `S = Φ(−η)`, which maps the index's
/// quantiles onto the survival's in reverse order and keeps both ends resolved
/// where `S` is within rounding of a rail.
pub(super) fn probit_survival_band(
    quadctx: &gam_solve::quadrature::QuadratureContext,
    mu: [f64; 2],
    cov: [[f64; 2]; 2],
    level: f64,
    index: impl Fn(f64, f64) -> Result<f64, String>,
) -> Result<(f64, f64), String> {
    let (eta_low, eta_high) = bivariate_central_interval(quadctx, mu, cov, level, index)?;
    Ok((probit_survival(eta_high), probit_survival(eta_low)))
}

/// The central interval of `g(x₀, x₁)` for `(x₀, x₁) ~ N(mu, cov)` at `level`,
/// on the support [`gam_solve::quadrature::normal_expectation_2d_projected_result`]
/// integrates the same law over (gam#3560): the two-coordinate rule on the
/// plane, the one-coordinate rule along the major axis of a rank-one
/// covariance, and the point at a point mass. So a band and the mean beside it
/// describe one law.
pub(super) fn bivariate_central_interval(
    quadctx: &gam_solve::quadrature::QuadratureContext,
    mu: [f64; 2],
    cov: [[f64; 2]; 2],
    level: f64,
    g: impl Fn(f64, f64) -> Result<f64, String>,
) -> Result<(f64, f64), String> {
    let interval = match gam_solve::quadrature::BivariateNormalSupport::of(cov) {
        gam_solve::quadrature::BivariateNormalSupport::Plane => {
            gam_solve::quadrature::central_response_interval_on_a_monotone_axis::<2, _, String>(
                quadctx,
                mu,
                cov,
                21,
                level,
                |x| g(x[0], x[1]),
            )?
        }
        gam_solve::quadrature::BivariateNormalSupport::Axis { axis, variance } => {
            gam_solve::quadrature::central_response_interval::<1, _, String>(
                quadctx,
                [0.0],
                [[variance]],
                0,
                21,
                level,
                |t| g(mu[0] + axis[0] * t[0], mu[1] + axis[1] * t[0]),
            )?
        }
        gam_solve::quadrature::BivariateNormalSupport::Point => None,
    };
    match interval {
        Some(interval) => Ok(interval),
        None => {
            let value = g(mu[0], mu[1])?;
            Ok((value, value))
        }
    }
}

/// `S = Φ(−η)`, the marginal-slope survival at index `η`, from the stable
/// signed-probit log-CDF.
pub(super) fn probit_survival(eta: f64) -> f64 {
    signed_probit_logcdf_and_mills_ratio(-eta).0.exp()
}

/// The index a single-event survival cell is a decreasing function of, for a
/// band carried through that function (gam#3560): `S = Φ(−η)` for the
/// marginal-slope probit index and `S = exp(−exp(η))` for `η = log H` of the
/// transformation and Weibull families. Read back from the cell's cumulative
/// hazard, so it is recovered to the resolution `H` carries in either tail.
#[derive(Clone, Copy, Debug)]
pub(super) enum SurvivalBandIndex {
    Probit,
    LogCumulativeHazard,
}

impl SurvivalBandIndex {
    pub(super) fn for_mode(mode: SurvivalLikelihoodMode) -> Result<Self, SurvivalPredictError> {
        match mode {
            SurvivalLikelihoodMode::MarginalSlope => Ok(Self::Probit),
            SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull => {
                Ok(Self::LogCumulativeHazard)
            }
            other => Err(SurvivalPredictError::UnsupportedConfiguration {
                reason: format!(
                    "a central survival band on the sigma-point rule needs a survival that is a \
                     decreasing function of one index per cell; the {} likelihood is not one \
                     this rule reads (gam#3560)",
                    survival_likelihood_modename(other)
                ),
            }),
        }
    }

    /// The cell's index, or `None` at a rail: `H = 0` (`S = 1`, index `−∞`) or
    /// `H = ∞` (`S = 0`, index `+∞`).
    pub(super) fn index(self, cumulative_hazard: f64) -> Result<Option<f64>, SurvivalPredictError> {
        if cumulative_hazard == 0.0 || cumulative_hazard == f64::INFINITY {
            return Ok(None);
        }
        if !(cumulative_hazard.is_finite() && cumulative_hazard > 0.0) {
            return Err(SurvivalPredictError::NumericalFailure {
                reason: format!(
                    "survival band index needs a cumulative hazard in [0, ∞]; got {cumulative_hazard}"
                ),
            });
        }
        Ok(Some(match self {
            // `log Φ(−η) = −H`, so `−η` is the normal quantile of log-CDF `−H`.
            Self::Probit => {
                -gam_math::probability::standard_normal_quantile_from_log_cdf(-cumulative_hazard)?
            }
            Self::LogCumulativeHazard => cumulative_hazard.ln(),
        }))
    }

    /// The survival at `index`.
    fn survival(self, index: f64) -> f64 {
        match self {
            Self::Probit => probit_survival(index),
            Self::LogCumulativeHazard => (-index.exp()).exp(),
        }
    }
}

/// One cell's index over the sigma-point nodes: the moments of the nodes whose
/// index is finite, and whether any node sat at either rail.
#[derive(Clone, Copy, Debug)]
pub(super) struct SurvivalIndexCell {
    pub(super) moment: PosteriorMoment,
    pub(super) resolved: bool,
    pub(super) at_one: bool,
    pub(super) at_zero: bool,
}

impl SurvivalIndexCell {
    pub(super) const EMPTY: Self = Self {
        moment: PosteriorMoment::EMPTY,
        resolved: false,
        at_one: false,
        at_zero: false,
    };

    /// The central interval of the cell's survival at the normal quantile `z`:
    /// the image of the index's interval, or the rail every node sat at. A cell
    /// whose nodes are split between a rail and finite indices has an atom at
    /// the rail that no normal law of the index carries, and it is refused.
    pub(super) fn band(
        self,
        index: SurvivalBandIndex,
        z: f64,
        cell: (usize, usize),
    ) -> Result<(f64, f64), SurvivalPredictError> {
        match (self.resolved, self.at_one, self.at_zero) {
            (true, false, false) => {
                let (mean, variance) = (self.moment.mean(), self.moment.variance());
                if !(mean.is_finite() && variance.is_finite()) {
                    return Err(SurvivalPredictError::NumericalFailure {
                        reason: format!(
                            "survival band index moments are not finite at row {}, time column {}: \
                             mean={mean}, variance={variance}",
                            cell.0, cell.1
                        ),
                    });
                }
                let spread = z * variance.sqrt();
                Ok((index.survival(mean + spread), index.survival(mean - spread)))
            }
            (false, true, false) => Ok((1.0, 1.0)),
            (false, false, true) => Ok((0.0, 0.0)),
            _ => Err(SurvivalPredictError::NumericalFailure {
                reason: format!(
                    "survival band at row {}, time column {}: some posterior nodes put the survival \
                     at a rail and others inside (0, 1), so the index has an atom no normal law \
                     carries and the band is not the image of an index interval (gam#3560)",
                    cell.0, cell.1
                ),
            }),
        }
    }
}

/// Weighted mean and centred co-moment of a small coordinate vector over a node
/// rule: [`PosteriorMoment`] in several coordinates, merged by the same
/// pairwise update so the covariance is a sum of non-negative rank-one terms
/// and never `E[xxᵀ] − E[x]E[x]ᵀ` (gam#4086).
#[derive(Clone, Debug)]
pub(super) struct JointPosteriorMoment {
    weight: f64,
    mean: Vec<f64>,
    /// Row-major `d × d` centred co-moment `Σ wᵢ (xᵢ − x̄)(xᵢ − x̄)ᵀ`.
    centered: Vec<f64>,
}

impl JointPosteriorMoment {
    pub(super) fn new(dimension: usize) -> Self {
        Self {
            weight: 0.0,
            mean: vec![0.0; dimension],
            centered: vec![0.0; dimension * dimension],
        }
    }

    pub(super) fn merge_point(&mut self, weight: f64, point: &[f64]) {
        if weight == 0.0 {
            return;
        }
        if self.weight == 0.0 {
            self.weight = weight;
            self.mean.copy_from_slice(point);
            return;
        }
        let dimension = self.mean.len();
        let total = self.weight + weight;
        let delta: Vec<f64> = point.iter().zip(&self.mean).map(|(x, m)| x - m).collect();
        let factor = self.weight * weight / total;
        for i in 0..dimension {
            for j in 0..dimension {
                self.centered[i * dimension + j] += factor * delta[i] * delta[j];
            }
            self.mean[i] += delta[i] * (weight / total);
        }
        self.weight = total;
    }

    fn covariance(&self, i: usize, j: usize) -> f64 {
        self.centered[i * self.mean.len() + j] / self.weight
    }
}

/// One competing-risks cell's joint law of `(w_1..w_K, w_1′..w_K′)` over the
/// posterior nodes, with `w_k = log H_k` and `w_k′ = h_k/H_k = d w_k/dt`, and
/// whether any node put a cause at `H_k = 0`, where `w_k` is `−∞` (the time
/// origin).
#[derive(Clone, Debug)]
pub(super) struct CompetingRisksBandCell {
    pub(super) coordinates: JointPosteriorMoment,
    pub(super) resolved: bool,
    pub(super) at_origin: bool,
}

/// [`CompetingRisksBands`] of every cell at the normal quantile `z` and the
/// band `level`, from each cell's joint law. A cell every node puts at the
/// time origin is a point mass at `S = 1`, `H = 0`, `h = 0`; one whose nodes
/// are split between the origin and positive cumulative hazards has an atom no
/// normal law of `w` carries, and is refused.
pub(super) fn competing_risks_bands(
    cells: &Array2<CompetingRisksBandCell>,
    cause_count: usize,
    level: f64,
) -> Result<CompetingRisksBands, SurvivalPredictError> {
    let z = gam_math::probability::standard_normal_quantile(0.5 + 0.5 * level)?;
    let (n_rows, n_times) = cells.dim();
    let surfaces = || vec![Array2::<f64>::zeros((n_rows, n_times)); cause_count];
    let mut bands = CompetingRisksBands {
        hazard_lower: surfaces(),
        hazard_upper: surfaces(),
        survival_lower: surfaces(),
        survival_upper: surfaces(),
        cumulative_hazard_lower: surfaces(),
        cumulative_hazard_upper: surfaces(),
        overall_survival_lower: Array2::zeros((n_rows, n_times)),
        overall_survival_upper: Array2::zeros((n_rows, n_times)),
        cif_refusal: BandRefusal::WholeCurveFunctional,
    };
    let quadctx = gam_solve::quadrature::QuadratureContext::new();
    for ((row, time), cell) in cells.indexed_iter() {
        match (cell.resolved, cell.at_origin) {
            (false, true) => {
                for cause in 0..cause_count {
                    bands.survival_lower[cause][[row, time]] = 1.0;
                    bands.survival_upper[cause][[row, time]] = 1.0;
                }
                bands.overall_survival_lower[[row, time]] = 1.0;
                bands.overall_survival_upper[[row, time]] = 1.0;
                continue;
            }
            (true, false) => {}
            _ => {
                return Err(SurvivalPredictError::NumericalFailure {
                    reason: format!(
                        "competing-risks band at row {row}, time column {time}: some posterior \
                         nodes put a cause at zero cumulative hazard and others above it, so its \
                         log cumulative hazard has an atom no normal law carries (gam#3560)"
                    ),
                });
            }
        }
        let law = &cell.coordinates;
        for cause in 0..cause_count {
            let derivative = cause_count + cause;
            let mean = law.mean[cause];
            let spread = z * law.covariance(cause, cause).sqrt();
            let (low, high) = (mean - spread, mean + spread);
            bands.cumulative_hazard_lower[cause][[row, time]] = low.exp();
            bands.cumulative_hazard_upper[cause][[row, time]] = high.exp();
            bands.survival_lower[cause][[row, time]] = (-high.exp()).exp();
            bands.survival_upper[cause][[row, time]] = (-low.exp()).exp();
            let (hazard_low, hazard_high) = bivariate_central_interval(
                &quadctx,
                [law.mean[cause], law.mean[derivative]],
                [
                    [
                        law.covariance(cause, cause),
                        law.covariance(cause, derivative),
                    ],
                    [
                        law.covariance(derivative, cause),
                        law.covariance(derivative, derivative),
                    ],
                ],
                level,
                |w, rate| Ok(w.exp() * rate),
            )?;
            bands.hazard_lower[cause][[row, time]] = hazard_low;
            bands.hazard_upper[cause][[row, time]] = hazard_high;
        }
        let (overall_low, overall_high) =
            overall_survival_central_interval(&quadctx, law, cause_count, level)?;
        bands.overall_survival_lower[[row, time]] = overall_low;
        bands.overall_survival_upper[[row, time]] = overall_high;
    }
    Ok(bands)
}

/// The central interval of the overall survival `exp(−Σ_k exp(w_k))` over the
/// joint normal law of `(w_1..w_K)`, which it is decreasing in along every
/// coordinate. The coordinate count is the cause count, so the rule is
/// dispatched on it; a law with no spread is its point.
pub(super) fn overall_survival_central_interval(
    quadctx: &gam_solve::quadrature::QuadratureContext,
    law: &JointPosteriorMoment,
    cause_count: usize,
    level: f64,
) -> Result<(f64, f64), SurvivalPredictError> {
    fn interval<const K: usize>(
        quadctx: &gam_solve::quadrature::QuadratureContext,
        law: &JointPosteriorMoment,
        level: f64,
    ) -> Result<(f64, f64), String> {
        let mut mu = [0.0_f64; K];
        let mut cov = [[0.0_f64; K]; K];
        for i in 0..K {
            mu[i] = law.mean[i];
            for j in 0..K {
                cov[i][j] = law.covariance(i, j);
            }
        }
        let overall = |w: [f64; K]| -> Result<f64, String> {
            Ok((-w.iter().map(|value| value.exp()).sum::<f64>()).exp())
        };
        match gam_solve::quadrature::central_response_interval_on_a_monotone_axis::<K, _, String>(
            quadctx, mu, cov, 15, level, overall,
        )? {
            Some(interval) => Ok(interval),
            None => {
                let value = overall(mu)?;
                Ok((value, value))
            }
        }
    }
    Ok(match cause_count {
        2 => interval::<2>(quadctx, law, level)?,
        3 => interval::<3>(quadctx, law, level)?,
        4 => interval::<4>(quadctx, law, level)?,
        other => {
            return Err(SurvivalPredictError::UnsupportedConfiguration {
                reason: format!(
                    "the overall-survival band inverts a {other}-coordinate law, and the central \
                     interval rule is instantiated for two to four causes (gam#3560)"
                ),
            });
        }
    })
}

/// Central posterior intervals of the competing-risks surfaces (gam#3560): per
/// cell, the quantiles `(1 ∓ level)/2` of the surface's posterior law, not
/// `mean ± z·sd` clamped to the surface's range.
///
/// Every cause-specific surface is a function of the cause's log cumulative
/// hazard `w_k(t) = log H_k(t)`, which is linear in the coefficients for the
/// transformation and Weibull families this prediction serves, and of its time
/// derivative `w_k′(t)`: `H_k = exp(w_k)`, `S_k = exp(−exp(w_k))`,
/// `h_k = exp(w_k)·w_k′` and the overall survival `exp(−Σ_k exp(w_k))`. The
/// joint law of those coordinates is carried at its posterior mean and
/// covariance from the prediction's own rule, and each band is the central
/// interval of the surface over that law: the image of the one coordinate's
/// interval for `H_k` and `S_k`, and
/// [`gam_solve::quadrature::central_response_interval_on_a_monotone_axis`] for
/// the hazard and the overall survival. The cumulative incidence carries no
/// band: [`Self::cif_refusal`] names why.
#[derive(Clone, Debug)]
pub struct CompetingRisksBands {
    pub hazard_lower: Vec<Array2<f64>>,
    pub hazard_upper: Vec<Array2<f64>>,
    pub survival_lower: Vec<Array2<f64>>,
    pub survival_upper: Vec<Array2<f64>>,
    pub cumulative_hazard_lower: Vec<Array2<f64>>,
    pub cumulative_hazard_upper: Vec<Array2<f64>>,
    pub overall_survival_lower: Array2<f64>,
    pub overall_survival_upper: Array2<f64>,
    /// Why the cumulative-incidence surfaces carry no band. Their point
    /// estimates still publish; a band known to be wrong does not.
    pub cif_refusal: BandRefusal,
}

/// Why a surface's central posterior band is refused rather than published
/// (gam#3560).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BandRefusal {
    /// `CIF_k(t) = ∫_0^t h_k(u)·S(u) du` depends on the whole curve of every
    /// cause's linear predictor over `[0, t]`, not on a fixed small vector of
    /// Gaussian coordinates, so no central interval of its posterior law is
    /// derived. `mean ± z·sd` clamped to `[0, 1]` is not one either.
    WholeCurveFunctional,
}

impl BandRefusal {
    /// The reason, as a presenter publishes it in place of the band.
    pub fn reason(self) -> &'static str {
        match self {
            Self::WholeCurveFunctional => {
                "no central interval is derived for the cumulative incidence: \
                 CIF_k(t) = ∫ h_k S du depends on the whole linear-predictor curve of every \
                 cause, not on a fixed low-dimensional Gaussian coordinate, and mean ± z·sd \
                 clamped to [0, 1] is not an interval of its posterior law (gam#3560)"
            }
        }
    }
}

/// The central survival band of every published location-scale cell under the
/// truncated law (gam#3560): the flattened prediction input's rows are the
/// cells, and each row's band is the certified central interval of its index
/// mixture ([`crate::survival::location_scale::truncated_survival_response_bands`]).
/// A cell at the time origin is `S = 1` under every coefficient vector, so its
/// band is that point.
pub(super) fn location_scale_truncated_surface_band(
    pred_input: &crate::survival::location_scale::SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
    covariance_mode: SurvivalPredictionCovarianceMode,
    level: f64,
    surface_cells: &[(usize, usize, usize)],
    n_rows: usize,
    t_cols: usize,
) -> Result<Array2<(f64, f64)>, String> {
    let covariance = select_survival_prediction_covariance(
        fit.beta_covariance(),
        fit.beta_covariance_corrected(),
        covariance_mode,
    )
    .map_err(String::from)?;
    let x_threshold_dense = pred_input.x_threshold.to_dense_arc();
    let x_log_sigma_dense = pred_input.x_log_sigma.to_dense_arc();
    let (lower, upper) = crate::survival::location_scale::truncated_survival_response_bands(
        pred_input,
        fit,
        covariance,
        &x_threshold_dense,
        &x_log_sigma_dense,
        level,
    )?
    .ok_or_else(|| {
        "survival location-scale band: the truncated posterior's covariance retains no \
         constraint row"
            .to_string()
    })?;
    let mut band = Array2::from_elem((n_rows, t_cols), (1.0, 1.0));
    for &(row, time, k) in surface_cells {
        band[[row, time]] = (lower[k], upper[k]);
    }
    Ok(band)
}

/// The central survival band of every published cell of a Royston-Parmar fit
/// under its cone-truncated law (gam#3560).
///
/// Every cell's index `w = log H` is linear in the coefficients, and given a
/// node's constraint-normal coordinates the coefficients are exactly
/// `N(c_u, Σ_res)` with `c_u` the node's conditional centre and `Σ_res = L Lᵀ`
/// the same residual covariance at every node. So `w | u ~ N(w(c_u), ‖gᵀ‖²)`
/// with `g_j = w(β + L e_j) − w(β)`, which linearity makes the same at every
/// `β`, and the cell's law is the normal mixture
/// [`crate::survival::location_scale::certified_index_mixture_bands`] inverts.
/// The survival `exp(−exp(w))` is decreasing in `w`, so the band is the image of
/// the index interval in reverse order. Rows are taken a chunk at a time so a
/// node's replay covers only the rows whose nodes are held; a cell at the time
/// origin is `S = 1` exactly, and its band is that point.
pub(super) fn replayed_truncated_surface_band(
    req: &SurvivalPredictRequest<'_>,
    covariance_mode: SurvivalPredictionCovarianceMode,
    draws: &TruncatedCoefficientDraws,
    fit: &UnifiedFitResult,
    plugin: &SurvivalPredictResult,
    level: f64,
) -> Result<Array2<(f64, f64)>, SurvivalPredictError> {
    if plugin.likelihood_mode != SurvivalLikelihoodMode::Transformation {
        return Err(SurvivalPredictError::UnsupportedConfiguration {
            reason: format!(
                "the truncated-law survival band reads log H as linear in the coefficients, \
                 which holds for the transformation likelihood; got {}",
                survival_likelihood_modename(plugin.likelihood_mode)
            ),
        });
    }
    let (n_rows, n_times) = plugin.survival.dim();
    let chunk_rows = crate::survival::location_scale::SURVIVAL_ROW_PARALLEL_CHUNK;
    let mut band = Array2::from_elem((n_rows, n_times), (1.0, 1.0));
    for start in (0..n_rows).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n_rows);
        let primary_offset = req.primary_offset.slice(s![start..end]).to_owned();
        let noise_offset = req.noise_offset.slice(s![start..end]).to_owned();
        let chunk_request = |model: &SavedModel| -> Result<SurvivalPredictResult, String> {
            predict_survival_coefficient_law(
                SurvivalPredictRequest {
                    model,
                    data: req.data.slice(s![start..end, ..]),
                    col_map: req.col_map,
                    training_headers: req.training_headers,
                    primary_offset: &primary_offset,
                    noise_offset: &noise_offset,
                    time_grid: req.time_grid,
                    with_uncertainty: false,
                    estimand: SurvivalPredictEstimand::Plugin,
                },
                covariance_mode,
            )
            .map_err(String::from)
        };
        // The cells after the origin, and their log cumulative hazards at `β`.
        let cells: Vec<(usize, usize)> = (start..end)
            .flat_map(|row| (0..n_times).map(move |time| (row, time)))
            .filter(|&(row, time)| plugin.cumulative_hazard[[row, time]] > 0.0)
            .collect();
        let log_cumulative_hazards = |beta: &Array1<f64>| -> Result<Vec<f64>, String> {
            let model = saved_model_with_survival_coefficients(req.model, beta)?;
            let draw = chunk_request(&model)?;
            cells
                .iter()
                .map(|&(row, time)| {
                    let cumulative_hazard = draw.cumulative_hazard[[row - start, time]];
                    if cumulative_hazard.is_finite() && cumulative_hazard > 0.0 {
                        Ok(cumulative_hazard.ln())
                    } else {
                        Err(format!(
                            "truncated-law survival band: row {row}, time column {time} has \
                             cumulative hazard {cumulative_hazard} at a node of the joint rule"
                        ))
                    }
                })
                .collect()
        };
        let reference = log_cumulative_hazards(&fit.beta)?;
        let factor = draws.residual_factor();
        let mut spread = vec![0.0_f64; cells.len()];
        for column in 0..factor.ncols() {
            let shifted = log_cumulative_hazards(&(&fit.beta + &factor.column(column)))?;
            for (slot, (moved, base)) in spread.iter_mut().zip(shifted.iter().zip(&reference)) {
                let step = moved - base;
                *slot += step * step;
            }
        }
        let spread: Vec<f64> = spread.into_iter().map(f64::sqrt).collect();
        let node_index = |normal_coordinates: &Array1<f64>,
                          tangent: &[f64]|
         -> Result<Vec<(f64, f64)>, String> {
            // The tangent is integrated exactly in `spread`; the node's own
            // tangent coordinates only repeat its conditional centre.
            let tangent_width = tangent.len();
            if tangent_width != factor.ncols() {
                return Err(format!(
                    "truncated-law survival band: node has {tangent_width} tangent coordinates, \
                     the law has {}",
                    factor.ncols()
                ));
            }
            let centre = draws.conditional_center(normal_coordinates)?;
            let means = log_cumulative_hazards(&centre)?;
            Ok(means.into_iter().zip(spread.iter().copied()).collect())
        };
        let intervals = crate::survival::location_scale::certified_index_mixture_bands(
            draws.rule(),
            cells.len(),
            level,
            &node_index,
        )?;
        for (&(row, time), &(index_low, index_high)) in cells.iter().zip(&intervals) {
            band[[row, time]] = ((-index_high.exp()).exp(), (-index_low.exp()).exp());
        }
    }
    Ok(band)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The old band this issue retires, `mean ± z·sd` clamped to the surface's
    /// range, kept here only as the control the new bands are compared with.
    fn clamped_symmetric_band(moment: PosteriorMoment, z: f64, range: (f64, f64)) -> (f64, f64) {
        let spread = z * moment.variance().sqrt();
        (
            (moment.mean() - spread).clamp(range.0, range.1),
            (moment.mean() + spread).clamp(range.0, range.1),
        )
    }

    /// A band on a response that saturates at a rail: inside the open range,
    /// around its mean, skewed away from the rail the response is near, and
    /// different from the clamped symmetric band, whose clamp binds at that rail
    /// in the regimes these tests pick (the control that the regime is one where
    /// the old band was wrong).
    fn assert_skewed_band_against_clamp(
        band: (f64, f64),
        moment: PosteriorMoment,
        z: f64,
        range: (f64, f64),
        near_upper_rail: bool,
        label: &str,
    ) {
        let (lower, upper) = band;
        let mean = moment.mean();
        let (clamped_lower, clamped_upper) = clamped_symmetric_band(moment, z, range);
        assert!(
            lower < mean && mean < upper,
            "{label}: band ({lower}, {upper}) does not contain its mean {mean}"
        );
        assert!(
            range.0 < lower && upper < range.1,
            "{label}: band ({lower}, {upper}) reaches a rail of {range:?}"
        );
        if near_upper_rail {
            assert_eq!(
                clamped_upper, range.1,
                "{label}: control — the clamped band's upper end must bind at the rail here"
            );
            assert!(
                upper - mean < mean - lower,
                "{label}: near the upper rail the band must reach further below its mean \
                 than above it: ({lower}, {mean}, {upper})"
            );
        } else {
            assert_eq!(
                clamped_lower, range.0,
                "{label}: control — the clamped band's lower end must bind at the rail here"
            );
            assert!(
                upper - mean > mean - lower,
                "{label}: near the lower rail the band must reach further above its mean \
                 than below it: ({lower}, {mean}, {upper})"
            );
        }
        assert!(
            (lower, upper) != (clamped_lower, clamped_upper),
            "{label}: the central band must not be the clamped symmetric one"
        );
    }

    /// gam#3560: the exact-anchor marginal-slope band is the central interval
    /// of `S = Φ(−η(q, b))` over the Gaussian law of the primaries the posterior
    /// mean integrates. `η = q + b` has unit variance here, at a high-survival
    /// index (`S ≈ Φ(2.5)`) and a low-survival one (`S ≈ Φ(−2.5)`), where `Φ`
    /// is curved and the old band clamped.
    #[test]
    fn marginal_slope_survival_band_is_the_central_interval_not_the_clamped_one_3560() {
        let quadctx = gam_solve::quadrature::QuadratureContext::new();
        let level = 0.95;
        let z = gam_math::probability::standard_normal_quantile(0.5 + 0.5 * level)
            .expect("normal quantile");
        let cov = [[0.5, 0.1], [0.1, 0.3]];
        for (eta_mean, near_upper_rail) in [(-2.5, true), (2.5, false)] {
            let mu = [eta_mean, 0.0];
            let band = probit_survival_band(&quadctx, mu, cov, level, |q, b| Ok(q + b))
                .expect("marginal-slope band");
            let moment = gam_solve::quadrature::normal_expectation_2d_projected_result(
                &quadctx,
                mu,
                cov,
                |q, b| -> Result<PosteriorMoment, String> {
                    Ok(PosteriorMoment::point(probit_survival(q + b)))
                },
            )
            .expect("marginal-slope moments");
            assert_skewed_band_against_clamp(
                band,
                moment,
                z,
                (0.0, 1.0),
                near_upper_rail,
                &format!("marginal slope at eta {eta_mean}"),
            );
        }
    }

    /// gam#3560: on the sigma-point rule the survival band is the image of the
    /// index interval under the decreasing survival map, read from the nodes'
    /// cumulative hazards. A cell every node puts at `S = 1` is that point, and a
    /// cell split between the rail and the interior is refused by name.
    #[test]
    fn sigma_point_survival_band_is_the_image_of_the_index_interval_3560() {
        let level = 0.95;
        let z = gam_math::probability::standard_normal_quantile(0.5 + 0.5 * level)
            .expect("normal quantile");
        for (eta_mean, near_upper_rail) in [(-2.5, true), (2.5, false)] {
            let index = SurvivalBandIndex::Probit;
            let mut cell = SurvivalIndexCell::EMPTY;
            let mut survival = PosteriorMoment::EMPTY;
            for eta in [eta_mean - 1.0, eta_mean + 1.0] {
                let cumulative_hazard = -signed_probit_logcdf_and_mills_ratio(-eta).0;
                let value = index
                    .index(cumulative_hazard)
                    .expect("index")
                    .expect("an interior survival has a finite index");
                cell.moment.merge(0.5, PosteriorMoment::point(value));
                cell.resolved = true;
                survival.merge(0.5, PosteriorMoment::point(probit_survival(eta)));
            }
            let band = cell.band(index, z, (0, 0)).expect("sigma-point band");
            assert_skewed_band_against_clamp(
                band,
                survival,
                z,
                (0.0, 1.0),
                near_upper_rail,
                &format!("sigma point at eta {eta_mean}"),
            );
        }
        let origin = SurvivalIndexCell {
            at_one: true,
            ..SurvivalIndexCell::EMPTY
        };
        assert_eq!(
            origin
                .band(SurvivalBandIndex::LogCumulativeHazard, z, (0, 0))
                .expect("a point mass at S = 1"),
            (1.0, 1.0)
        );
        let split = SurvivalIndexCell {
            resolved: true,
            at_one: true,
            ..SurvivalIndexCell::EMPTY
        };
        assert!(
            split
                .band(SurvivalBandIndex::LogCumulativeHazard, z, (0, 0))
                .is_err(),
            "a cell split between the rail and the interior has no normal index law"
        );
    }

    /// gam#3560: the competing-risks survival, hazard and overall-survival bands
    /// are central intervals of the joint normal law of `(w_k, w_k′)`, `w_k =
    /// log H_k`, carried from the nodes. Two causes, the nodes the symmetric
    /// sigma-point rule places along each of the four coordinates; the published
    /// means and the clamped control are formed from the same nodes.
    #[test]
    fn competing_risks_bands_are_central_intervals_not_the_clamped_ones_3560() {
        let level = 0.95;
        let z = gam_math::probability::standard_normal_quantile(0.5 + 0.5 * level)
            .expect("normal quantile");
        let mean = [0.5, -0.5, 1.0, 1.0];
        let sd = [1.0, 0.5, 0.3, 0.3];
        let dimension = mean.len();
        let scale = (dimension as f64).sqrt();
        let weight = 1.0 / (2 * dimension) as f64;
        let mut cell = CompetingRisksBandCell {
            coordinates: JointPosteriorMoment::new(dimension),
            resolved: true,
            at_origin: false,
        };
        let mut survival = PosteriorMoment::EMPTY;
        let mut hazard = PosteriorMoment::EMPTY;
        let mut overall = PosteriorMoment::EMPTY;
        for axis in 0..dimension {
            for sign in [-1.0, 1.0] {
                let mut node = mean;
                node[axis] += sign * scale * sd[axis];
                cell.coordinates.merge_point(weight, &node);
                survival.merge(weight, PosteriorMoment::point((-node[0].exp()).exp()));
                hazard.merge(weight, PosteriorMoment::point(node[0].exp() * node[2]));
                overall.merge(
                    weight,
                    PosteriorMoment::point((-(node[0].exp() + node[1].exp())).exp()),
                );
            }
        }
        let cells = Array2::from_elem((1, 1), cell);
        let bands = competing_risks_bands(&cells, 2, level).expect("competing-risks bands");
        assert_skewed_band_against_clamp(
            (
                bands.survival_lower[0][[0, 0]],
                bands.survival_upper[0][[0, 0]],
            ),
            survival,
            z,
            (0.0, 1.0),
            false,
            "cause 1 survival",
        );
        assert_skewed_band_against_clamp(
            (bands.hazard_lower[0][[0, 0]], bands.hazard_upper[0][[0, 0]]),
            hazard,
            z,
            (0.0, f64::INFINITY),
            false,
            "cause 1 hazard",
        );
        assert_skewed_band_against_clamp(
            (
                bands.overall_survival_lower[[0, 0]],
                bands.overall_survival_upper[[0, 0]],
            ),
            overall,
            z,
            (0.0, 1.0),
            false,
            "overall survival",
        );
    }
}

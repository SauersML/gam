//! Exact nuisance-scale estimation from a certified linear predictor.
//!
//! Every estimator consumes the same inverse-link surface as the PIRLS working
//! state.  Eta projection, mean floors, neutral fallback values, and silently
//! returned parameter-band endpoints are forbidden: if a required statistic or
//! a finite interior nuisance estimate cannot be represented, the fit fails
//! closed.

use super::*;

pub(crate) const GAMMA_SHAPE_MIN: f64 = 1e-8;
pub(crate) const GAMMA_SHAPE_MAX: f64 = 1e12;
pub(crate) const GAMMA_SHAPE_TARGET_TOL: f64 = 1e-12;

/// Saturation threshold used only by inner-loop separation diagnostics.
pub(super) const PIRLS_ETA_ABS_CAP: f64 = 40.0;

/// Declared finite profiling interval for NB2 theta.  The interval endpoints
/// are not estimates: absence of an interior ML root is a typed refusal.
pub(crate) const NEGBIN_THETA_MIN: f64 = 1e-3;
pub(crate) const NEGBIN_THETA_MAX: f64 = 1e6;

#[inline]
fn row_unrepresentable(
    row: usize,
    quantity: &'static str,
    eta: f64,
    value: f64,
) -> EstimationError {
    EstimationError::PirlsRowGeometryUnrepresentable {
        row,
        quantity,
        eta,
        value,
    }
}

fn certified_log_means(eta: &Array1<f64>) -> Result<Vec<f64>, EstimationError> {
    let rows: Vec<Result<f64, EstimationError>> = eta
        .par_iter()
        .map(|&eta_i| crate::mixture_link::log_link_solver_exp(eta_i))
        .collect();
    rows.into_iter().collect()
}

#[inline]
fn certified_prior_weight(row: usize, eta: f64, weight: f64) -> Result<f64, EstimationError> {
    if weight.is_finite() && weight >= 0.0 {
        Ok(weight)
    } else {
        Err(row_unrepresentable(row, "prior weight", eta, weight))
    }
}

fn certified_pairs_sum(
    rows: Vec<Result<(f64, f64), EstimationError>>,
) -> Result<(f64, f64), EstimationError> {
    let rows: Vec<(f64, f64)> = rows.into_iter().collect::<Result<_, _>>()?;
    let sum = gam_linalg::pairwise_reduce::par_pairwise_map_reduce(
        rows.len(),
        |i| rows[i],
        |(a, b), (c, d)| (a + c, b + d),
        (0.0, 0.0),
    );
    if sum.0.is_finite() && sum.1.is_finite() {
        Ok(sum)
    } else {
        Err(EstimationError::InvalidInput(
            "nuisance-profile reduction exceeded the finite f64 range".to_string(),
        ))
    }
}

#[inline]
pub(crate) fn gamma_shape_score(shape: f64, target: f64) -> f64 {
    shape.ln() - digamma(shape) - target
}

pub(crate) fn estimate_gamma_shape_from_eta(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
) -> Result<f64, EstimationError> {
    let means = certified_log_means(eta)?;
    let rows: Vec<Result<(f64, f64), EstimationError>> = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            let wi = certified_prior_weight(i, eta[i], priorweights[i])?;
            if wi == 0.0 {
                return Ok((0.0, 0.0));
            }
            if !(y[i].is_finite() && y[i] > 0.0) {
                return Err(row_unrepresentable(i, "Gamma response", eta[i], y[i]));
            }
            let ratio = y[i] / means[i];
            let target = ratio - ratio.ln() - 1.0;
            let contribution = wi * target;
            if !(target.is_finite() && target >= 0.0 && contribution.is_finite()) {
                return Err(row_unrepresentable(
                    i,
                    "Gamma shape statistic",
                    eta[i],
                    contribution,
                ));
            }
            Ok((contribution, wi))
        })
        .collect();
    let (weighted_target, total_weight) = certified_pairs_sum(rows)?;
    if !(total_weight > 0.0) {
        crate::bail_invalid_estim!("Gamma shape profiling requires positive total prior weight");
    }
    let target = weighted_target / total_weight;
    if !(target.is_finite() && target > 0.0) {
        crate::bail_invalid_estim!(
            "Gamma shape MLE is not a finite interior value (profile target={target:?})"
        );
    }

    let discriminant = (target - 3.0) * (target - 3.0) + 24.0 * target;
    let approx = ((3.0 - target) + discriminant.sqrt()) / (12.0 * target);
    let mut lo = GAMMA_SHAPE_MIN;
    let mut hi = approx.max(1.0).min(GAMMA_SHAPE_MAX);
    if gamma_shape_score(lo, target) <= 0.0 {
        crate::bail_invalid_estim!(
            "Gamma shape MLE lies below the declared profiling domain ({GAMMA_SHAPE_MIN}, {GAMMA_SHAPE_MAX})"
        );
    }
    while hi < GAMMA_SHAPE_MAX && gamma_shape_score(hi, target) > 0.0 {
        hi = (hi * 2.0).min(GAMMA_SHAPE_MAX);
    }
    if gamma_shape_score(hi, target) > 0.0 {
        crate::bail_invalid_estim!(
            "Gamma shape MLE is not finite inside the declared profiling domain ({GAMMA_SHAPE_MIN}, {GAMMA_SHAPE_MAX})"
        );
    }
    for _ in 0..80 {
        let mid = lo + 0.5 * (hi - lo);
        if gamma_shape_score(mid, target) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) <= GAMMA_SHAPE_TARGET_TOL * hi {
            break;
        }
    }
    let shape = lo + 0.5 * (hi - lo);
    if shape.is_finite() && shape > 0.0 {
        Ok(shape)
    } else {
        crate::bail_invalid_estim!("Gamma shape solve produced {shape:?}")
    }
}

/// Exact method-of-moments Beta precision on the represented logit surface.
pub(crate) fn estimate_beta_phi_from_eta(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
) -> Result<f64, EstimationError> {
    let rows: Vec<Result<(f64, f64), EstimationError>> = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            let wi = certified_prior_weight(i, eta[i], priorweights[i])?;
            if wi == 0.0 {
                return Ok((0.0, 0.0));
            }
            if !(y[i].is_finite() && y[i] > 0.0 && y[i] < 1.0) {
                return Err(row_unrepresentable(i, "Beta response", eta[i], y[i]));
            }
            if !eta[i].is_finite() {
                return Err(EstimationError::InverseLinkDomainViolation {
                    link: "standard logit inverse link",
                    eta: eta[i],
                    lower: -f64::MAX,
                    upper: f64::MAX,
                });
            }
            let jet = logit_inverse_link_jet5(eta[i]);
            if !(jet.mu > 0.0 && jet.mu < 1.0 && jet.d1.is_finite() && jet.d1 > 0.0) {
                return Err(row_unrepresentable(i, "Beta mean/variance", eta[i], jet.d1));
            }
            let resid = y[i] - jet.mu;
            let statistic = wi * resid * resid / jet.d1;
            if !(statistic.is_finite() && statistic >= 0.0) {
                return Err(row_unrepresentable(
                    i,
                    "Beta precision statistic",
                    eta[i],
                    statistic,
                ));
            }
            Ok((statistic, wi))
        })
        .collect();
    let (weighted_pearson, total_weight) = certified_pairs_sum(rows)?;
    if !(total_weight > 0.0 && weighted_pearson > 0.0) {
        crate::bail_invalid_estim!(
            "Beta precision MLE is not finite and positive (Pearson={weighted_pearson:?}, weight={total_weight:?})"
        );
    }
    let phi = total_weight / weighted_pearson - 1.0;
    if phi.is_finite() && phi > 0.0 {
        Ok(phi)
    } else {
        crate::bail_invalid_estim!("Beta precision estimate is not finite and positive: {phi:?}")
    }
}

/// Exact Pearson Tweedie dispersion on the represented log-link surface.
pub(crate) fn estimate_tweedie_phi_from_eta(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
    p: f64,
) -> Result<f64, EstimationError> {
    if !is_valid_tweedie_power(p) {
        crate::bail_invalid_estim!("invalid Tweedie variance power {p:?}");
    }
    let means = certified_log_means(eta)?;
    let rows: Vec<Result<(f64, f64), EstimationError>> = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            let wi = certified_prior_weight(i, eta[i], priorweights[i])?;
            if wi == 0.0 {
                return Ok((0.0, 0.0));
            }
            if !(y[i].is_finite() && y[i] >= 0.0) {
                return Err(row_unrepresentable(i, "Tweedie response", eta[i], y[i]));
            }
            let variance = means[i].powf(p);
            let resid = y[i] - means[i];
            let statistic = wi * resid * resid / variance;
            if !(variance.is_finite()
                && variance > 0.0
                && statistic.is_finite()
                && statistic >= 0.0)
            {
                return Err(row_unrepresentable(
                    i,
                    "Tweedie dispersion statistic",
                    eta[i],
                    statistic,
                ));
            }
            Ok((statistic, wi))
        })
        .collect();
    let (weighted_pearson, total_weight) = certified_pairs_sum(rows)?;
    if !(total_weight > 0.0 && weighted_pearson > 0.0) {
        crate::bail_invalid_estim!(
            "Tweedie dispersion is not finite and positive (Pearson={weighted_pearson:?}, weight={total_weight:?})"
        );
    }
    let phi = weighted_pearson / total_weight;
    if phi.is_finite() && phi > 0.0 {
        Ok(phi)
    } else {
        crate::bail_invalid_estim!("Tweedie dispersion estimate is invalid: {phi:?}")
    }
}

fn negbin_theta_score_and_info_from_means(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    means: &[f64],
    priorweights: ArrayView1<'_, f64>,
    theta: f64,
) -> Result<(f64, f64), EstimationError> {
    if !(theta.is_finite() && theta > 0.0) {
        crate::bail_invalid_estim!("negative-binomial theta must be finite and positive");
    }
    let psi_theta = digamma(theta);
    let trigamma_theta = trigamma(theta);
    let ln_theta = theta.ln();
    let inv_theta = theta.recip();
    let rows: Vec<Result<(f64, f64), EstimationError>> = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            let wi = certified_prior_weight(i, eta[i], priorweights[i])?;
            if wi == 0.0 {
                return Ok((0.0, 0.0));
            }
            let yi = y[i];
            if !valid_count_response(yi) {
                return Err(row_unrepresentable(
                    i,
                    "negative-binomial response",
                    eta[i],
                    yi,
                ));
            }
            let theta_plus_mu = theta + means[i];
            let theta_plus_y = theta + yi;
            let s = digamma(yi + theta) - psi_theta + ln_theta + 1.0
                - theta_plus_mu.ln()
                - theta_plus_y / theta_plus_mu;
            // Avoid forming `(theta + mu)^2`, which can overflow even when the
            // information term itself is representable.
            let info_row = -trigamma(yi + theta) + trigamma_theta - inv_theta + 2.0 / theta_plus_mu
                - (theta_plus_y / theta_plus_mu) / theta_plus_mu;
            let score = wi * s;
            let info = wi * info_row;
            if !(score.is_finite() && info.is_finite()) {
                return Err(row_unrepresentable(
                    i,
                    "negative-binomial theta score/information",
                    eta[i],
                    score,
                ));
            }
            Ok((score, info))
        })
        .collect();
    certified_pairs_sum(rows)
}

pub(crate) fn negbin_theta_score_and_info(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
    theta: f64,
) -> Result<(f64, f64), EstimationError> {
    let means = certified_log_means(eta)?;
    negbin_theta_score_and_info_from_means(y, eta, &means, priorweights, theta)
}

/// Profile a finite interior NB2 theta with safeguarded Newton/bisection.
pub(crate) fn estimate_negbin_theta_from_eta(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
) -> Result<f64, EstimationError> {
    let means = certified_log_means(eta)?;
    let seed_rows: Vec<Result<(f64, f64), EstimationError>> = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            let wi = certified_prior_weight(i, eta[i], priorweights[i])?;
            if wi == 0.0 {
                return Ok((0.0, 0.0));
            }
            if !valid_count_response(y[i]) {
                return Err(row_unrepresentable(
                    i,
                    "negative-binomial response",
                    eta[i],
                    y[i],
                ));
            }
            let resid = y[i] - means[i];
            let pearson = wi * resid * resid / means[i];
            let weighted_mu = wi * means[i];
            if !(pearson.is_finite() && pearson >= 0.0 && weighted_mu.is_finite()) {
                return Err(row_unrepresentable(
                    i,
                    "negative-binomial seed statistic",
                    eta[i],
                    pearson,
                ));
            }
            Ok((weighted_mu, pearson))
        })
        .collect();
    let (wmu, wpearson) = certified_pairs_sum(seed_rows)?;
    let total_weight = priorweights.iter().try_fold(0.0, |sum, &w| {
        if w.is_finite() && w >= 0.0 {
            let next = sum + w;
            if next.is_finite() { Ok(next) } else { Err(()) }
        } else {
            Err(())
        }
    });
    let total_weight = total_weight.map_err(|_| {
        EstimationError::InvalidInput(
            "negative-binomial total prior weight is invalid or unrepresentable".to_string(),
        )
    })?;
    if !(total_weight > 0.0) {
        crate::bail_invalid_estim!("negative-binomial profiling requires positive total weight");
    }
    let mu_bar = wmu / total_weight;
    let pearson_ratio = wpearson / total_weight;
    let mut theta = if pearson_ratio > 1.0 {
        mu_bar / (pearson_ratio - 1.0)
    } else {
        0.5 * (NEGBIN_THETA_MIN + NEGBIN_THETA_MAX)
    };
    // This projection is only a root-solver seed; it is never returned as an
    // estimate and therefore does not alter the profiled model.
    theta = theta.max(NEGBIN_THETA_MIN).min(NEGBIN_THETA_MAX);

    let (score_lo, _) =
        negbin_theta_score_and_info_from_means(y, eta, &means, priorweights, NEGBIN_THETA_MIN)?;
    let (score_hi, _) =
        negbin_theta_score_and_info_from_means(y, eta, &means, priorweights, NEGBIN_THETA_MAX)?;
    if !(score_lo > 0.0 && score_hi < 0.0) {
        crate::bail_invalid_estim!(
            "negative-binomial theta has no finite interior ML root in ({NEGBIN_THETA_MIN}, {NEGBIN_THETA_MAX}); scores=({score_lo:?}, {score_hi:?})"
        );
    }
    let mut lo = NEGBIN_THETA_MIN;
    let mut hi = NEGBIN_THETA_MAX;
    for _ in 0..100 {
        let (score, info) =
            negbin_theta_score_and_info_from_means(y, eta, &means, priorweights, theta)?;
        if score > 0.0 {
            lo = theta;
        } else {
            hi = theta;
        }
        let candidate = theta + score / info;
        let next = if info.is_finite() && info > 0.0 && candidate > lo && candidate < hi {
            candidate
        } else {
            lo + 0.5 * (hi - lo)
        };
        if (next - theta).abs() <= 1e-10 * theta {
            theta = next;
            break;
        }
        theta = next;
    }
    if theta.is_finite() && theta > NEGBIN_THETA_MIN && theta < NEGBIN_THETA_MAX {
        Ok(theta)
    } else {
        crate::bail_invalid_estim!("negative-binomial theta solve produced {theta:?}")
    }
}

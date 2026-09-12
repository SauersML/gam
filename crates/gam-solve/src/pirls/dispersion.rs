//! Exact nuisance-scale estimation from a certified linear predictor.
//!
//! Every estimator consumes the same inverse-link surface as the PIRLS working
//! state.  Eta projection, mean floors, neutral fallback values, and silently
//! returned parameter-band endpoints are forbidden: if a required statistic or
//! a finite interior nuisance estimate cannot be represented, the fit fails
//! closed.

use super::*;

/// Saturation threshold used only by inner-loop separation diagnostics.
pub(super) const PIRLS_ETA_ABS_CAP: f64 = 40.0;

/// Declared finite profiling interval for NB2 theta.  The interval endpoints
/// are not estimates: absence of an interior ML root is a typed refusal.
pub(crate) const NEGBIN_THETA_MIN: f64 = 1e-3;
pub(crate) const NEGBIN_THETA_MAX: f64 = 1e6;


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
        Err(EstimationError::pirls_row_geometry_unrepresentable(row, "prior weight", eta, weight))
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
    let log_minus_digamma = if shape >= 32.0 {
        // ln(a)-psi(a) = 1/(2a)+1/(12a²)-1/(120a⁴)+... .
        // Subtracting the two logarithmic-size values loses the entire
        // score at large a. The first omitted term here is O(a^-12),
        // below one ulp of the result throughout this branch.
        let inv = shape.recip();
        let inv2 = inv * inv;
        0.5 * inv
            + inv2
                * (1.0 / 12.0
                    + inv2
                        * (-1.0 / 120.0
                            + inv2 * (1.0 / 252.0 + inv2 * (-1.0 / 240.0 + inv2 / 132.0))))
    } else {
        shape.ln() - digamma(shape)
    };
    log_minus_digamma - target
}

#[inline]
fn gamma_shape_statistic(response: f64, mean: f64) -> f64 {
    let relative_residual = (response - mean) / mean;
    if relative_residual.abs() <= 0.5 {
        // r-ln(r)-1 = d-ln(1+d), d=(y-mu)/mu. The subtraction
        // y-mu preserves nearby represented responses before division.
        -gam_math::special::log1p_minus_x(relative_residual)
    } else {
        // The log ratio remains finite when y/mu underflows to zero.
        gam_math::special::expm1_minus_x(response.ln() - mean.ln())
    }
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
                return Err(EstimationError::pirls_row_geometry_unrepresentable(i, "Gamma response", eta[i], y[i]));
            }
            let target = gamma_shape_statistic(y[i], means[i]);
            let contribution = wi * target;
            if !(target.is_finite() && target >= 0.0 && contribution.is_finite()) {
                return Err(EstimationError::pirls_row_geometry_unrepresentable(
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
    // Every row is now a nonnegative, cancellation-free statistic. A small
    // positive value is real dispersion; only zero has no finite shape MLE.
    if !(target.is_finite() && target > 0.0) {
        crate::bail_invalid_estim!(
            "Gamma shape MLE is not finite: the dispersion statistic is {target:?}"
        );
    }

    // `ln α − ψ(α)` falls from `+∞` as `α → 0⁺` to `0` as `α → ∞`, so for a
    // positive statistic the score has exactly one root. Bracket it by halving
    // and doubling outward from the closed-form approximation; the only way out
    // of either walk is the representable range itself.
    let approx = if target < 3.0 {
        let delta = 3.0 - target;
        (delta + (delta * delta + 24.0 * target).sqrt()) / 12.0 / target
    } else {
        // Rationalize the numerator and divide the radical by target:
        // 2/(sqrt(t²+18t+9)+t-3). Neither t² nor 12t is formed.
        let inv = target.recip();
        (2.0 * inv) / ((1.0 + 18.0 * inv + 9.0 * inv * inv).sqrt() + 1.0 - 3.0 * inv)
    };
    if !(approx.is_finite() && approx > 0.0) {
        crate::bail_invalid_estim!(
            "Gamma shape approximation is not representable (profile target={target:?}, approximation={approx:?})"
        );
    }
    let mut lo = approx;
    let mut hi = approx;
    while gamma_shape_score(lo, target) <= 0.0 {
        lo *= 0.5;
        if !(lo > 0.0) {
            crate::bail_invalid_estim!(
                "Gamma shape MLE lies below the representable range (profile target={target:?})"
            );
        }
    }
    while gamma_shape_score(hi, target) > 0.0 {
        if hi == f64::MAX {
            crate::bail_invalid_estim!(
                "Gamma shape MLE exceeds the representable range (profile target={target:?})"
            );
        }
        hi = if hi <= 0.5 * f64::MAX { 2.0 * hi } else { f64::MAX };
    }
    // Bisect until no representable shape lies strictly inside the bracket.
    loop {
        let mid = lo + 0.5 * (hi - lo);
        if !(mid > lo && mid < hi) {
            break;
        }
        if gamma_shape_score(mid, target) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
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
                return Err(EstimationError::pirls_row_geometry_unrepresentable(i, "Beta response", eta[i], y[i]));
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
                return Err(EstimationError::pirls_row_geometry_unrepresentable(i, "Beta mean/variance", eta[i], jet.d1));
            }
            let resid = y[i] - jet.mu;
            let statistic = wi * resid * resid / jet.d1;
            if !(statistic.is_finite() && statistic >= 0.0) {
                return Err(EstimationError::pirls_row_geometry_unrepresentable(
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
                return Err(EstimationError::pirls_row_geometry_unrepresentable(i, "Tweedie response", eta[i], y[i]));
            }
            let resid = y[i] - means[i];
            // Form the complete Pearson term before exponentiating. Both
            // residual² and mu^p may exceed the float range while their
            // weighted ratio remains finite and informative.
            let statistic = if resid == 0.0 {
                0.0
            } else {
                (wi.ln() + 2.0 * resid.abs().ln() - p * means[i].ln()).exp()
            };
            if !(statistic.is_finite() && statistic >= 0.0) {
                return Err(EstimationError::pirls_row_geometry_unrepresentable(
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

#[cfg(test)]
mod gamma_tweedie_profile_math_tests {
    use super::*;

    #[test]
    fn gamma_statistic_retains_near_unit_ratios_and_underflowed_ratios() {
        for response in [1.0_f64 - 1.0e-8, 1.0 + 1.0e-8] {
            let delta = response - 1.0;
            let expected = 0.5 * delta * delta - delta.powi(3) / 3.0;
            let actual = gamma_shape_statistic(response, 1.0);
            assert!(actual > 0.0);
            assert!((actual / expected - 1.0).abs() < 1.0e-14);
        }
        let target = gamma_shape_statistic(1.0e-300, 1.0e300);
        assert!((target - (600.0 * std::f64::consts::LN_10 - 1.0)).abs() < 1.0e-12);
        assert_eq!(gamma_shape_statistic(1.0, 1.0), 0.0);
    }

    #[test]
    fn gamma_shape_score_retains_large_shape_information() {
        for shape in [1.0e16_f64, 1.0e200, f64::MAX] {
            let score = gamma_shape_score(shape, 0.0);
            assert!(score > 0.0);
            assert!((score / (0.5 / shape) - 1.0).abs() < 1.0e-14);
        }
    }

    #[test]
    fn gamma_profile_fits_small_nonzero_dispersion_and_large_targets() {
        let y = Array1::from(vec![1.0 - 1.0e-8, 1.0 + 1.0e-8]);
        let eta = Array1::zeros(2);
        let weights = Array1::ones(2);
        let shape = estimate_gamma_shape_from_eta(y.view(), &eta, weights.view())
            .expect("a nonzero dispersion has a finite Gamma shape");
        let mean_square = 0.5 * ((y[0] - 1.0).powi(2) + (y[1] - 1.0).powi(2));
        assert!((shape * mean_square - 1.0).abs() < 1.0e-7);
        let target = 0.5 * (gamma_shape_statistic(y[0], 1.0) + gamma_shape_statistic(y[1], 1.0));
        assert!(gamma_shape_score(0.99 * shape, target) > 0.0);
        assert!(gamma_shape_score(1.01 * shape, target) < 0.0);

        let large_y = Array1::from(vec![1.0e200]);
        let shape = estimate_gamma_shape_from_eta(large_y.view(), &Array1::zeros(1), Array1::ones(1).view())
            .expect("a large profile target has a small finite Gamma shape");
        assert!((shape * large_y[0] - 1.0).abs() < 1.0e-12);

        assert!(estimate_gamma_shape_from_eta(Array1::ones(2).view(), &eta, weights.view()).is_err());
    }

    #[test]
    fn tweedie_pearson_statistic_preserves_representable_extreme_ratios() {
        for log_mean in [-600.0_f64, 600.0] {
            let y = Array1::zeros(1);
            let eta = Array1::from(vec![log_mean]);
            let weights = Array1::ones(1);
            let phi = estimate_tweedie_phi_from_eta(y.view(), &eta, weights.view(), 1.5)
                .expect("Pearson ratio is representable despite overflowed squared terms");
            // With y=0, (y-mu)^2/mu^p = mu^(2-p).
            let expected = (0.5 * log_mean).exp();
            assert!((phi / expected - 1.0).abs() < 1.0e-12);
        }
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
                return Err(EstimationError::pirls_row_geometry_unrepresentable(
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
                return Err(EstimationError::pirls_row_geometry_unrepresentable(
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
                return Err(EstimationError::pirls_row_geometry_unrepresentable(
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
                return Err(EstimationError::pirls_row_geometry_unrepresentable(
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
    if !(score_lo.is_finite() && score_hi.is_finite()) {
        crate::bail_invalid_estim!(
            "negative-binomial theta profile score is unrepresentable at the domain endpoints ({NEGBIN_THETA_MIN}, {NEGBIN_THETA_MAX}); scores=({score_lo:?}, {score_hi:?})"
        );
    }
    // `score` IS d/dtheta of the profiled NB2 log-likelihood, so its SIGN at an
    // endpoint locates the maximizer relative to that endpoint. An interior
    // root needs `score_lo > 0 > score_hi`; the two one-sided failures are not
    // "no maximizer", they are a maximizer the representable domain cannot
    // hold, and the profile's maximum over the domain is then attained AT the
    // endpoint.
    //
    // `score_hi >= 0` is the equidispersed / underdispersed case: the
    // likelihood is still rising at theta = 1e6, i.e. theta-hat is +inf and the
    // NB2 model has collapsed onto its Poisson limit. Measured on this
    // workspace's own fixtures the excess is not even resolvable —
    // `scores=(6948.2, 9.06e-12)` and `scores=(137678.1, 7.45e-13)`, upper
    // scores 15 and 17 orders of magnitude below the lower one — so refusing
    // the WHOLE fit over it discarded a converged model to avoid returning the
    // one value the profile actually maximizes at.
    //
    // `score_lo <= 0` is the mirror image at the lower rail. Both return the
    // endpoint, which is exactly what the safeguarded bisection below would
    // converge to if the domain were open; what it must NOT do is silently
    // report an interior estimate, and it does not: the returned theta is the
    // rail itself.
    //
    // Both endpoints failing at once is a genuinely non-monotone profile score,
    // which no rail can represent, and keeps the fail-loud refusal.
    if score_lo <= 0.0 && score_hi >= 0.0 {
        crate::bail_invalid_estim!(
            "negative-binomial theta profile score is non-monotone across ({NEGBIN_THETA_MIN}, {NEGBIN_THETA_MAX}); scores=({score_lo:?}, {score_hi:?})"
        );
    }
    if score_hi >= 0.0 {
        return Ok(NEGBIN_THETA_MAX);
    }
    if score_lo <= 0.0 {
        return Ok(NEGBIN_THETA_MIN);
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

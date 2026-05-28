use super::*;
use super::family::clamp_bernoulli_link_probability;

pub(crate) fn standardize_latent_z_with_policy(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    context: &str,
    policy: &LatentZPolicy,
) -> Result<(Array1<f64>, LatentZNormalization), String> {
    if z.len() != weights.len() {
        return Err(format!(
            "{context} latent-score normalization length mismatch: z={}, weights={}",
            z.len(),
            weights.len()
        ));
    }
    let weight_sum = weights.iter().copied().sum::<f64>();
    let weight_sq_sum = weights.iter().map(|&w| w * w).sum::<f64>();
    if !(weight_sum.is_finite()
        && weight_sum > 0.0
        && weight_sq_sum.is_finite()
        && weight_sq_sum > 0.0)
    {
        return Err(format!("{context} requires positive finite total weight"));
    }
    let effective_n = weight_sum * weight_sum / weight_sq_sum;
    if !(effective_n.is_finite() && effective_n > 1.0) {
        return Err(format!(
            "{context} requires at least two effective observations for latent-score normalization"
        ));
    }
    let mean = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi)
        .sum::<f64>()
        / weight_sum;
    let var = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * (zi - mean) * (zi - mean))
        .sum::<f64>()
        / weight_sum;
    let sd = var.sqrt();
    if !(sd.is_finite() && sd > BMS_VARIANCE_FLOOR) {
        return Err(format!(
            "{context} requires z with positive finite weighted standard deviation"
        ));
    }
    let target_norm = match policy.normalization {
        LatentZNormalizationMode::None => LatentZNormalization { mean: 0.0, sd: 1.0 },
        LatentZNormalizationMode::FitWeighted => LatentZNormalization { mean, sd },
        LatentZNormalizationMode::Frozen {
            mean: frozen_mean,
            sd: frozen_sd,
        } => LatentZNormalization {
            mean: frozen_mean,
            sd: frozen_sd,
        },
    };
    let mean_tol = policy.mean_tol_multiplier / effective_n.sqrt();
    let sd_tol = policy.sd_tol_multiplier / (2.0 * (effective_n - 1.0).max(1.0)).sqrt();
    let check_msg = || {
        format!(
            "{context} requires z to already be approximately latent N(0,1) before identification normalization; got mean={mean:.6e}, sd={sd:.6e}, effective_n={effective_n:.1}, allowed_mean={mean_tol:.3e}, allowed_sd={sd_tol:.3e}"
        )
    };
    if mean.abs() > mean_tol || (sd - 1.0).abs() > sd_tol {
        match policy.check_mode {
            LatentZCheckMode::Strict => return Err(check_msg()),
            LatentZCheckMode::WarnOnly => log::warn!("{}", check_msg()),
            LatentZCheckMode::Off => {}
        }
    }

    let normalization = target_norm;
    let z_std = normalization.apply(z, context)?;
    let skew = z_std
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi.powi(3))
        .sum::<f64>()
        / weight_sum;
    let kurt = z_std
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi.powi(4))
        .sum::<f64>()
        / weight_sum
        - 3.0;
    if skew.abs() > policy.max_abs_skew || kurt.abs() > policy.max_abs_excess_kurtosis {
        let msg = format!(
            "{context} requires z to be approximately Gaussian after identification normalization; got skewness={skew:.3}, excess_kurtosis={kurt:.3}"
        );
        match policy.check_mode {
            LatentZCheckMode::Strict => return Err(msg),
            LatentZCheckMode::WarnOnly => log::warn!("{}", msg),
            LatentZCheckMode::Off => {}
        }
    }
    if skew.abs() > 0.75 || kurt.abs() > 2.0 {
        log::warn!(
            "{context}: z has skewness={skew:.3} and excess kurtosis={kurt:.3}; latent-measure auto-selection will use empirical calibration unless stricter diagnostics pass"
        );
    }
    Ok((z_std, normalization))
}

pub fn padded_deviation_seed(seed: &Array1<f64>, min_iqr: f64, pad_fraction: f64) -> Array1<f64> {
    let mut sorted = seed.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if sorted.len() < 4 {
        return seed.clone();
    }

    let n = sorted.len();
    let q1 = sorted[n / 4];
    let q3 = sorted[3 * n / 4];
    let iqr = (q3 - q1).max(min_iqr);
    let pad = pad_fraction * iqr;

    let mut out = seed.to_vec();
    out.push(sorted[0] - pad);
    out.push(sorted[n - 1] + pad);
    Array1::from_vec(out)
}

pub(super) fn pooled_probit_baseline(
    y: &Array1<f64>,
    z: &Array1<f64>,
    weights: &Array1<f64>,
) -> Result<(f64, f64), String> {
    if y.len() != z.len() || y.len() != weights.len() {
        return Err(format!(
            "pooled bernoulli-marginal-slope pilot length mismatch: y={}, z={}, weights={}",
            y.len(),
            z.len(),
            weights.len()
        ));
    }
    let weight_sum = weights.iter().copied().sum::<f64>();
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(
            "pooled bernoulli-marginal-slope pilot requires positive finite total weight"
                .to_string(),
        );
    }
    let prevalence = y
        .iter()
        .zip(weights.iter())
        .map(|(&yi, &wi)| yi * wi)
        .sum::<f64>()
        / weight_sum;
    let prevalence = prevalence.clamp(1e-6, 1.0 - 1e-6);
    let z_mean = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| zi * wi)
        .sum::<f64>()
        / weight_sum;
    let z_var = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * (zi - z_mean) * (zi - z_mean))
        .sum::<f64>()
        / weight_sum;
    let yz_cov = y
        .iter()
        .zip(z.iter())
        .zip(weights.iter())
        .map(|((&yi, &zi), &wi)| wi * (yi - prevalence) * (zi - z_mean))
        .sum::<f64>()
        / weight_sum;
    let mut beta0 = standard_normal_quantile(prevalence).map_err(|e| {
        format!("failed to initialize pooled bernoulli-marginal-slope pilot intercept: {e}")
    })?;
    let mut beta1 = if z_var > BMS_VARIANCE_FLOOR {
        yz_cov / z_var
    } else {
        0.0
    };

    let objective_grad_hess =
        |intercept: f64, slope: f64| -> Result<(f64, f64, f64, f64, f64, f64), String> {
            let mut obj = 0.0;
            let mut g0 = 0.0;
            let mut g1 = 0.0;
            let mut h00 = 0.0;
            let mut h01 = 0.0;
            let mut h11 = 0.0;
            for ((&yi, &zi), &wi) in y.iter().zip(z.iter()).zip(weights.iter()) {
                if wi == 0.0 {
                    continue;
                }
                let eta = intercept + slope * zi;
                let s = 2.0 * yi - 1.0;
                let margin = s * eta;
                let (logcdf, lambda) = signed_probit_logcdf_and_mills_ratio(margin);
                let g_eta = -wi * s * lambda;
                let h_eta = wi * lambda * (margin + lambda);
                obj -= wi * logcdf;
                g0 += g_eta;
                g1 += g_eta * zi;
                h00 += h_eta;
                h01 += h_eta * zi;
                h11 += h_eta * zi * zi;
            }
            Ok((obj, g0, g1, h00, h01, h11))
        };

    let mut obj_prev = f64::INFINITY;
    for _ in 0..50 {
        let (obj, g0, g1, h00, h01, h11) = objective_grad_hess(beta0, beta1)?;
        if !obj.is_finite() || !g0.is_finite() || !g1.is_finite() {
            return Err(
                "pooled bernoulli-marginal-slope pilot produced non-finite objective or gradient"
                    .to_string(),
            );
        }
        let grad_max = g0.abs().max(g1.abs());
        if grad_max < BMS_DERIV_TOL {
            break;
        }
        let mut ridge = 1e-8;
        let (step0, step1) = loop {
            let h00_r = h00 + ridge;
            let h11_r = h11 + ridge;
            let det = h00_r * h11_r - h01 * h01;
            if det.is_finite() && det.abs() > 1e-18 {
                let s0 = (h11_r * g0 - h01 * g1) / det;
                let s1 = (-h01 * g0 + h00_r * g1) / det;
                if s0.is_finite() && s1.is_finite() {
                    break (s0, s1);
                }
            }
            ridge *= 10.0;
            if ridge > 1e6 {
                return Err(
                    "pooled bernoulli-marginal-slope pilot Hessian solve failed".to_string()
                );
            }
        };
        let mut accepted = false;
        let mut step_scale = 1.0;
        for _ in 0..25 {
            let cand0 = beta0 - step_scale * step0;
            let cand1 = beta1 - step_scale * step1;
            let (cand_obj, _, _, _, _, _) = objective_grad_hess(cand0, cand1)?;
            if cand_obj.is_finite() && cand_obj <= obj {
                beta0 = cand0;
                beta1 = cand1;
                obj_prev = cand_obj;
                accepted = true;
                break;
            }
            step_scale *= 0.5;
        }
        if !accepted {
            if (obj_prev - obj).abs() < 1e-10 {
                break;
            }
            return Err("pooled bernoulli-marginal-slope pilot line search failed".to_string());
        }
    }
    let a = beta0;
    // Signed slope: preserve direction from pilot probit.
    let b = if beta1.abs() < 1e-6 {
        if beta1.is_sign_negative() {
            -1e-6
        } else {
            1e-6
        }
    } else {
        beta1
    };
    Ok((a / (1.0 + b * b).sqrt(), b))
}

// Compute a non-degenerate pilot η for the link-deviation cross-block
// identifiability orthogonalisation.
//
// The rigid pooled probit pilot from `pooled_probit_baseline` is a scalar
// pair `(a₀, b₀)`, so the rigid observed-scale linear predictor
// `η_rigid[i] = a₀·√(1 + (s_f·b₀)²) + s_f·b₀·z[i]` is **exactly affine in z**
// when the per-row offsets are zero. A degree-3 I-spline of an affine
// function of `z` spans the same column space at training rows as a
// degree-3 I-spline of `z` directly, so evaluating the link-deviation basis
// at `η_rigid` and orthogonalising it against the score-warp basis (built
// on `z`) produces a structurally singular cross-Gram — the candidate is
// fully aliased even though at PIRLS time the link-deviation runtime is
// re-evaluated at the current β-dependent η which carries genuine PC / age
// structure that the score-warp cannot represent.
//
// One probit Gauss-Newton step from the rigid pilot, projected onto the
// full marginal design at the W-IRLS working response, picks up that PC /
// age structure cheaply (one `p_marg × p_marg` Cholesky plus a few matvecs
// — `<<1 s` at biobank scale because `p_marg` is `O(10²)` whereas the
// PIRLS dense Hessian build is `O(n·p²)` per cycle). The resulting
// `η_pilot[i]` has the same row-by-row variation pattern PIRLS will see at
// any non-degenerate β, so the orthogonalisation transform `T` drops only
// the directions that are aliased *across all* β, not those that are
// aliased only at the rigid (rank-1-in-z) pilot.
/// IRLS Hessian row metric for the probit-style data Hessian at a fixed
/// linear predictor `eta`: `w[i] = sample_weights[i] · φ(η_i)² / (μ_i·(1−μ_i))`.
///
/// This is the canonical row metric that the joint penalised Hessian sees
/// during PIRLS for a probit GLM (and the dominant term for
/// BernoulliMarginalSlope's data Hessian). Cross-block orthogonalisation
/// against parametric anchors must use **this** metric — not a uniform
/// W=spec.weights — for the joint Hessian to be block-orthogonal between
/// parametric and flex spans. With a uniform W the orthogonalisation only
/// kills the Euclidean alias; at PIRLS time `Aᵀ W_pirls C̃ ≠ 0` and the
/// joint Hessian carries a near-null direction along the W-metric alias,
/// which REML can drive to arbitrarily small eigenvalue by shrinking the
/// flex block's smoothing parameter — β then runs away along the alias
/// (the failure mode that manifests as `rho≈2.0`, constant `step_inf`,
/// and `beta_inf` growing without bound during PIRLS).
pub(super) fn pilot_irls_hessian_row_metric_at_eta(
    eta_pilot: &Array1<f64>,
    sample_weights: &Array1<f64>,
) -> Array1<f64> {
    let n = eta_pilot.len();
    let mut w = Array1::<f64>::zeros(n);
    for i in 0..n {
        let eta = eta_pilot[i];
        let mu = clamp_bernoulli_link_probability(normal_cdf(eta));
        let phi = normal_pdf(eta).max(1e-300);
        let var = (mu * (1.0 - mu)).max(1e-300);
        w[i] = sample_weights[i] * (phi * phi) / var;
    }
    w
}

/// Per-row rigid pooled-probit pilot η used to seed the IRLS Hessian
/// metric for score-warp cross-block orthogonalisation. Score-warp's
/// basis is evaluated at `z` (β-independent) so there is no GN-stepped
/// pilot to share with the link-deviation path; the rigid pooled-probit
/// pilot is a sensible β-independent reference at which to evaluate
/// `W = p(1−p)·spec.weights` for the W-metric orthogonalisation.
pub(super) fn rigid_pooled_probit_pilot_eta(
    base_link: &InverseLink,
    z: &Array1<f64>,
    marginal_offset: &Array1<f64>,
    logslope_offset: &Array1<f64>,
    baseline_marginal: f64,
    baseline_logslope: f64,
    probit_scale: f64,
) -> Result<Array1<f64>, String> {
    let n = z.len();
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        let a_pre = baseline_marginal + marginal_offset[i];
        let b_pre = baseline_logslope + logslope_offset[i];
        let q_marg = bernoulli_marginal_link_map(base_link, a_pre)
            .map_err(|e| format!("rigid_pooled_probit_pilot_eta marginal link map: {e}"))?
            .q;
        out[i] = rigid_observed_eta(q_marg, b_pre, z[i], probit_scale);
    }
    Ok(out)
}

pub(super) fn pilot_eta_for_link_dev_orthogonalisation(
    base_link: &InverseLink,
    y: &Array1<f64>,
    z: &Array1<f64>,
    weights: &Array1<f64>,
    marginal_design: &DesignMatrix,
    marginal_offset: &Array1<f64>,
    logslope_offset: &Array1<f64>,
    baseline_marginal: f64,
    baseline_logslope: f64,
    probit_scale: f64,
) -> Result<Array1<f64>, String> {
    use crate::faer_ndarray::FaerCholesky;

    let n = y.len();
    if marginal_design.nrows() != n {
        return Err(format!(
            "pilot_eta_for_link_dev_orthogonalisation: marginal design has {} rows, expected {}",
            marginal_design.nrows(),
            n,
        ));
    }
    let mut working_eta = Array1::<f64>::zeros(n);
    let mut w_irls = Array1::<f64>::zeros(n);
    let mut residual = Array1::<f64>::zeros(n);
    for i in 0..n {
        let a_pre = baseline_marginal + marginal_offset[i];
        let b_pre = baseline_logslope + logslope_offset[i];
        let q_marg = bernoulli_marginal_link_map(base_link, a_pre)
            .map_err(|e| {
                format!("pilot_eta_for_link_dev_orthogonalisation marginal link map: {e}")
            })?
            .q;
        let eta = rigid_observed_eta(q_marg, b_pre, z[i], probit_scale);
        working_eta[i] = eta;
        let mu = clamp_bernoulli_link_probability(normal_cdf(eta));
        let phi = normal_pdf(eta).max(1e-300);
        let var = (mu * (1.0 - mu)).max(1e-300);
        w_irls[i] = weights[i] * (phi * phi) / var;
        residual[i] = (y[i] - mu) / phi;
    }
    let p_marg = marginal_design.ncols();
    if p_marg == 0 {
        return Ok(working_eta);
    }
    let xtwr = marginal_design.compute_xtwy(&w_irls, &residual)?;
    let mut xtwx = marginal_design.compute_xtwx(&w_irls)?;
    let trace_diag: f64 = (0..p_marg).map(|i| xtwx[[i, i]]).sum();
    let ridge = (trace_diag / p_marg as f64).max(1e-12) * 1e-6;
    for i in 0..p_marg {
        xtwx[[i, i]] += ridge;
    }
    let factor = xtwx
        .cholesky(faer::Side::Lower)
        .map_err(|e| format!("pilot_eta_for_link_dev_orthogonalisation Cholesky failed: {e}"))?;
    let delta_beta_marg = factor.solvevec(&xtwr);
    let marg_contrib = marginal_design.dot(&delta_beta_marg);
    Ok(&working_eta + &marg_contrib)
}

pub(super) fn joint_setup(
    data: ArrayView2<'_, f64>,
    marginalspec: &TermCollectionSpec,
    logslopespec: &TermCollectionSpec,
    marginal_penalties: usize,
    logslope_penalties: usize,
    extra_rho0: &[f64],
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> ExactJointHyperSetup {
    let marginal_terms = spatial_length_scale_term_indices(marginalspec);
    let logslope_terms = spatial_length_scale_term_indices(logslopespec);
    let rho_dim = marginal_penalties + logslope_penalties + extra_rho0.len();
    let mut rho0vec = Array1::<f64>::zeros(rho_dim);
    for (idx, &value) in extra_rho0.iter().enumerate() {
        rho0vec[marginal_penalties + logslope_penalties + idx] = value;
    }
    let rho_lower = Array1::<f64>::from_elem(rho_dim, -12.0);
    let rho_upper = Array1::<f64>::from_elem(rho_dim, 12.0);
    let marginal_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        marginalspec,
        &marginal_terms,
        kappa_options,
    )
    .reseed_from_data(data, marginalspec, &marginal_terms, kappa_options);
    let logslope_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        logslopespec,
        &logslope_terms,
        kappa_options,
    )
    .reseed_from_data(data, logslopespec, &logslope_terms, kappa_options);
    let mut values = marginal_kappa.as_array().to_vec();
    values.extend(logslope_kappa.as_array().iter());
    let marginal_dims = marginal_kappa.dims_per_term().to_vec();
    let logslope_dims = logslope_kappa.dims_per_term().to_vec();
    let mut dims = marginal_dims.clone();
    dims.extend(logslope_dims.iter().copied());
    let log_kappa0 = SpatialLogKappaCoords::new_with_dims(Array1::from_vec(values), dims.clone());
    // Bounds: concatenate per-block data-aware bounds in the same order.
    let marginal_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        data,
        marginalspec,
        &marginal_terms,
        &marginal_dims,
        kappa_options,
    );
    let logslope_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        data,
        logslopespec,
        &logslope_terms,
        &logslope_dims,
        kappa_options,
    );
    let mut lower_vals = marginal_lower.as_array().to_vec();
    lower_vals.extend(logslope_lower.as_array().iter());
    let log_kappa_lower =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(lower_vals), dims.clone());
    let marginal_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        data,
        marginalspec,
        &marginal_terms,
        &marginal_dims,
        kappa_options,
    );
    let logslope_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        data,
        logslopespec,
        &logslope_terms,
        &logslope_dims,
        kappa_options,
    );
    let mut upper_vals = marginal_upper.as_array().to_vec();
    upper_vals.extend(logslope_upper.as_array().iter());
    let log_kappa_upper = SpatialLogKappaCoords::new_with_dims(Array1::from_vec(upper_vals), dims);
    // Project seed onto bounds in case a user-provided spec.length_scale falls
    // outside the data-derived ψ window; seed was a hint, not a hard constraint.
    let log_kappa0 = log_kappa0.clamp_to_bounds(&log_kappa_lower, &log_kappa_upper);
    ExactJointHyperSetup::new(
        rho0vec,
        rho_lower,
        rho_upper,
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
    )
}

#[inline]
fn signed_probit_neglog_derivatives_up_to_fourth_numeric(
    signed_margin: f64,
    weight: f64,
) -> (f64, f64, f64, f64) {
    if weight == 0.0 || signed_margin == f64::INFINITY {
        return (0.0, 0.0, 0.0, 0.0);
    }
    if signed_margin == f64::NEG_INFINITY {
        return (f64::NEG_INFINITY, weight, 0.0, 0.0);
    }
    if signed_margin.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let (_, lambda) = signed_probit_logcdf_and_mills_ratio(signed_margin);
    let k1 = -lambda;
    let k2 = lambda * (signed_margin + lambda);
    let k3 = lambda
        * (1.0
            - signed_margin * signed_margin
            - 3.0 * signed_margin * lambda
            - 2.0 * lambda * lambda);
    let k4 = lambda
        * ((signed_margin.powi(3) - 3.0 * signed_margin)
            + (7.0 * signed_margin * signed_margin - 4.0) * lambda
            + 12.0 * signed_margin * lambda * lambda
            + 6.0 * lambda.powi(3));
    (weight * k1, weight * k2, weight * k3, weight * k4)
}

/// Exact probit derivative helper used by analytic jet code paths.
///
/// `+inf` is the saturated zero tail and is allowed. `-inf` and `NaN` are
/// rejected instead of being silently collapsed, so exact callers fail fast
/// rather than erasing curvature or domain errors. Numeric boundary behavior
/// that needs to preserve `-inf` / `NaN` values lives in
/// `signed_probit_neglog_derivatives_up_to_fourth_numeric`.
pub(crate) fn signed_probit_neglog_derivatives_up_to_fourth(
    signed_margin: f64,
    weight: f64,
) -> Result<(f64, f64, f64, f64), String> {
    if weight == 0.0 || signed_margin == f64::INFINITY {
        return Ok((0.0, 0.0, 0.0, 0.0));
    }
    if !signed_margin.is_finite() {
        return Err(format!(
            "non-finite signed margin in exact probit derivative helper: {signed_margin}"
        ));
    }
    Ok(signed_probit_neglog_derivatives_up_to_fourth_numeric(
        signed_margin,
        weight,
    ))
}

#[inline]
pub(super) fn rigid_observed_logslope(logslope: f64, probit_scale: f64) -> f64 {
    probit_scale * logslope
}

#[inline]
pub(super) fn rigid_observed_scale(logslope: f64, probit_scale: f64) -> f64 {
    let observed_logslope = rigid_observed_logslope(logslope, probit_scale);
    (1.0 + observed_logslope * observed_logslope).sqrt()
}

#[inline]
pub(super) fn rigid_intercept_from_marginal(marginal_eta: f64, logslope: f64, probit_scale: f64) -> f64 {
    marginal_eta * rigid_observed_scale(logslope, probit_scale)
}

#[inline]
pub(super) fn rigid_prescale_intercept_from_marginal(
    marginal_eta: f64,
    logslope: f64,
    probit_scale: f64,
) -> f64 {
    rigid_intercept_from_marginal(marginal_eta, logslope, probit_scale) / probit_scale
}

#[inline]
pub(super) fn rigid_prescale_intercept_derivative_abs(
    marginal_eta: f64,
    logslope: f64,
    probit_scale: f64,
) -> f64 {
    let c = rigid_observed_scale(logslope, probit_scale);
    probit_scale * normal_pdf(marginal_eta) / c
}

#[inline]
pub(super) fn rigid_observed_eta(marginal_eta: f64, logslope: f64, z: f64, probit_scale: f64) -> f64 {
    marginal_slope_standard_normal_scalar_eta(marginal_eta, logslope, z, probit_scale)
}

#[inline]
pub(super) fn marginal_slope_standard_normal_scalar_eta(q: f64, slope: f64, z: f64, probit_scale: f64) -> f64 {
    let observed_slope = rigid_observed_logslope(slope, probit_scale);
    q * (1.0 + observed_slope * observed_slope).sqrt() + observed_slope * z
}

pub(super) fn unary_derivatives_normal_cdf(x: f64) -> [f64; 5] {
    let pdf = normal_pdf(x);
    [
        normal_cdf(x),
        pdf,
        -x * pdf,
        (x * x - 1.0) * pdf,
        (-x.powi(3) + 3.0 * x) * pdf,
    ]
}

pub(super) fn unary_derivatives_normal_pdf(x: f64) -> [f64; 5] {
    let pdf = normal_pdf(x);
    [
        pdf,
        -x * pdf,
        (x * x - 1.0) * pdf,
        (-x.powi(3) + 3.0 * x) * pdf,
        (x.powi(4) - 6.0 * x * x + 3.0) * pdf,
    ]
}

pub(super) fn unary_derivatives_reciprocal(x: f64) -> [f64; 5] {
    let x1 = x.max(1e-300);
    let x2 = x1 * x1;
    let x3 = x2 * x1;
    let x4 = x3 * x1;
    let x5 = x4 * x1;
    [1.0 / x1, -1.0 / x2, 2.0 / x3, -6.0 / x4, 24.0 / x5]
}

/// Streaming log-sum-exp update: accumulate `exp(log_term)` into a running
/// `(log_max, sum)` pair representing `Σ exp(log_term_i) = exp(log_max) · sum`.
///
/// When `log_term` exceeds the running max, the partial sum is rescaled in
/// place so the new max becomes the reference point. This keeps everything
/// inside the dynamic range of f64 with no allocation.
#[inline]
pub(super) fn lse_accumulate(log_max: &mut f64, sum: &mut f64, log_term: f64) {
    if !log_term.is_finite() {
        return;
    }
    if log_term > *log_max {
        if log_max.is_finite() {
            *sum = *sum * (*log_max - log_term).exp() + 1.0;
        } else {
            *sum = 1.0;
        }
        *log_max = log_term;
    } else {
        *sum += (log_term - *log_max).exp();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarginalSlopeCovarianceShape {
    Diagonal,
    Full,
    LowRank,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MarginalSlopeCovariance {
    Diagonal(Array1<f64>),
    Full(Array2<f64>),
    /// Low-rank factor L with Sigma = L L^T.
    LowRank(Array2<f64>),
}

impl MarginalSlopeCovariance {
    pub fn shape(&self) -> MarginalSlopeCovarianceShape {
        match self {
            Self::Diagonal(_) => MarginalSlopeCovarianceShape::Diagonal,
            Self::Full(_) => MarginalSlopeCovarianceShape::Full,
            Self::LowRank(_) => MarginalSlopeCovarianceShape::LowRank,
        }
    }

    pub fn dim(&self) -> usize {
        match self {
            Self::Diagonal(diag) => diag.len(),
            Self::Full(cov) => cov.nrows(),
            Self::LowRank(factor) => factor.nrows(),
        }
    }

    pub fn validate(&self, context: &str) -> Result<(), String> {
        match self {
            Self::Diagonal(diag) => {
                if diag.is_empty() {
                    return Err(format!("{context} diagonal covariance is empty"));
                }
                for (idx, &value) in diag.iter().enumerate() {
                    if !(value.is_finite() && value >= 0.0) {
                        return Err(format!(
                            "{context} diagonal covariance entry {idx} must be finite and non-negative, got {value}"
                        ));
                    }
                }
            }
            Self::Full(cov) => {
                if cov.nrows() == 0 || cov.nrows() != cov.ncols() {
                    return Err(format!(
                        "{context} full covariance must be non-empty and square, got {}x{}",
                        cov.nrows(),
                        cov.ncols()
                    ));
                }
                for i in 0..cov.nrows() {
                    for j in 0..cov.ncols() {
                        let value = cov[[i, j]];
                        if !value.is_finite() {
                            return Err(format!(
                                "{context} full covariance entry ({i},{j}) is non-finite"
                            ));
                        }
                        if (value - cov[[j, i]]).abs()
                            > 1e-10 * (1.0 + value.abs().max(cov[[j, i]].abs()))
                        {
                            return Err(format!(
                                "{context} full covariance must be symmetric at ({i},{j})"
                            ));
                        }
                    }
                }
            }
            Self::LowRank(factor) => {
                if factor.nrows() == 0 {
                    return Err(format!(
                        "{context} low-rank covariance factor has zero rows"
                    ));
                }
                for ((i, j), &value) in factor.indexed_iter() {
                    if !value.is_finite() {
                        return Err(format!(
                            "{context} low-rank covariance factor entry ({i},{j}) is non-finite"
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn quadratic_form(&self, vector: &[f64]) -> Result<f64, String> {
        self.validate("marginal-slope covariance")?;
        if vector.len() != self.dim() {
            return Err(format!(
                "marginal-slope covariance dimension mismatch: vector={}, covariance={}",
                vector.len(),
                self.dim()
            ));
        }
        if vector.iter().any(|value| !value.is_finite()) {
            return Err("marginal-slope covariance vector contains non-finite values".to_string());
        }
        let value = match self {
            Self::Diagonal(diag) => vector
                .iter()
                .zip(diag.iter())
                .map(|(&v, &sigma)| v * v * sigma)
                .sum::<f64>(),
            Self::Full(cov) => {
                let mut total = 0.0;
                for i in 0..cov.nrows() {
                    let mut row_dot = 0.0;
                    for j in 0..cov.ncols() {
                        row_dot += cov[[i, j]] * vector[j];
                    }
                    total += vector[i] * row_dot;
                }
                total
            }
            Self::LowRank(factor) => {
                // Sigma = L L'. The Gaussian-probit scale only needs
                // r' Sigma r = ||L' r||^2. Equivalently,
                // det(I + L' r r' L) = 1 + ||L' r||^2 by the matrix
                // determinant lemma, so the low-rank path never builds
                // the full K x K covariance.
                let mut total = 0.0;
                for r in 0..factor.ncols() {
                    let mut projection = 0.0;
                    for k in 0..factor.nrows() {
                        projection += factor[[k, r]] * vector[k];
                    }
                    total += projection * projection;
                }
                total
            }
        };
        if value.is_finite() && value >= -1e-10 {
            Ok(value.max(0.0))
        } else {
            Err(format!(
                "marginal-slope covariance quadratic form must be non-negative, got {value}"
            ))
        }
    }
}

// Marginal-slope probit identity.
//
// For a row with latent scores z | a ~ N(0, Sigma(a)) and probit index
//
//     eta = c(a) q(t, a) + r(a)' z,
//
// the preservation target is
//
//     E_z[Phi(-eta) | a] = Phi(-q(t, a)).
//
// If X = r' z is N(0, v) with v = r' Sigma r, then for independent
// E ~ N(0, 1),
//
//     E[Phi(-(c q + X))]
//       = P(E <= -c q - X)
//       = P(E + X <= -c q)
//       = Phi(-c q / sqrt(1 + v)).
//
// Thus the target holds for every q exactly when
//
//     c(a) = sqrt(1 + r(a)' Sigma(a) r(a)).
//
// `probit_scale` maps the raw log-slope surface to the observed probit
// gradient r(a). K=1 with diagonal variance 1 gives the original scalar
// formula sqrt(1 + r^2); full and low-rank covariances differ only in the
// shape-specific evaluation of the same quadratic form.
pub fn marginal_slope_covariance_from_scores(
    scores: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
) -> Result<MarginalSlopeCovariance, String> {
    let (n, k) = scores.dim();
    if k == 0 {
        return Err("marginal-slope score matrix must have at least one column".to_string());
    }
    if weights.len() != n {
        return Err(format!(
            "marginal-slope covariance weight length mismatch: weights={}, rows={n}",
            weights.len()
        ));
    }
    let total_weight = weights.iter().copied().sum::<f64>();
    if !(total_weight.is_finite() && total_weight > 0.0) {
        return Err("marginal-slope covariance needs positive finite total weight".to_string());
    }
    let mut mean = Array1::<f64>::zeros(k);
    for i in 0..n {
        let weight = weights[i];
        if !(weight.is_finite() && weight >= 0.0) {
            return Err(format!(
                "marginal-slope covariance weight {i} must be finite and non-negative, got {weight}"
            ));
        }
        for j in 0..k {
            let score = scores[[i, j]];
            if !score.is_finite() {
                return Err(format!(
                    "marginal-slope covariance score ({i},{j}) is non-finite"
                ));
            }
            mean[j] += weight * score;
        }
    }
    mean.mapv_inplace(|value| value / total_weight);

    let mut cov = Array2::<f64>::zeros((k, k));
    for i in 0..n {
        let weight = weights[i];
        for a in 0..k {
            let da = scores[[i, a]] - mean[a];
            for b in 0..=a {
                let value = weight * da * (scores[[i, b]] - mean[b]) / total_weight;
                cov[[a, b]] += value;
                if a != b {
                    cov[[b, a]] += value;
                }
            }
        }
    }

    // ── Shape classification ──
    //
    // Pick the cheapest representation that preserves r'Σr for arbitrary r.
    //
    //   * K = 1: always Diagonal — LowRank/Full distinctions are meaningless.
    //
    //   * STRICT NUMERICAL DIAGONAL: if every off-diagonal is at machine
    //     precision relative to the diagonal scale, return Diagonal.  This
    //     catches both structurally-orthogonal inputs (post-orthogonalised
    //     production paths) AND degenerate cases like a column of all
    //     zeros (rank-deficient but truly diagonal).
    //
    //   * Otherwise eigendecompose.  positive.len() < K ⇒ the rank
    //     deficiency comes from collinear columns (off-diagonals are
    //     non-trivial) — Diagonal would drop the coupling and break r'Σr
    //     ⇒ LowRank.
    //
    //   * Full rank: apply a 4σ statistical off-diagonal test.  Under H0
    //     (independent population columns) the asymptotic SE of an
    //     off-diagonal sample covariance is √(σ_aa σ_bb / N_eff) with
    //     N_eff = (Σw)² / Σw² (Kish).  Pass ⇒ Diagonal (sample noise was
    //     not real correlation), fail ⇒ Full.  At biobank N_eff the 4σ
    //     statistical floor collapses below the numerical floor, so
    //     production behaviour is unchanged.
    if k == 1 {
        return Ok(MarginalSlopeCovariance::Diagonal(cov.diag().to_owned()));
    }

    let diag: Vec<f64> = (0..k).map(|i| cov[[i, i]]).collect();
    let diag_max = diag.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let numerical_floor = 1e-10 * (1.0 + diag_max);

    let mut is_strict_diagonal = true;
    'strict: for a in 0..k {
        for b in (a + 1)..k {
            if cov[[a, b]].abs() > numerical_floor {
                is_strict_diagonal = false;
                break 'strict;
            }
        }
    }
    if is_strict_diagonal {
        return Ok(MarginalSlopeCovariance::Diagonal(cov.diag().to_owned()));
    }

    use crate::faer_ndarray::FaerEigh;
    let (evals, evecs) = cov
        .eigh(faer::Side::Lower)
        .map_err(|err| format!("marginal-slope covariance eigendecomposition failed: {err}"))?;
    let max_eval = evals
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
    let rank_tol = 1e-10 * max_eval.max(1.0);
    let positive: Vec<(usize, f64)> = evals
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| (value > rank_tol).then_some((idx, value)))
        .collect();

    if positive.len() < k {
        // Rank deficiency with non-trivial off-diagonals ⇒ collinear
        // columns; Diagonal would lose the coupling.
        let mut factor = Array2::<f64>::zeros((k, positive.len()));
        for (col, (idx, value)) in positive.iter().enumerate() {
            let scale = value.sqrt();
            for row in 0..k {
                factor[[row, col]] = evecs[[row, *idx]] * scale;
            }
        }
        return Ok(MarginalSlopeCovariance::LowRank(factor));
    }

    // Full rank.  4σ statistical off-diagonal test.
    let sum_w_sq = weights.iter().map(|&w| w * w).sum::<f64>();
    let n_eff = if sum_w_sq > 0.0 {
        (total_weight * total_weight) / sum_w_sq
    } else {
        1.0
    };
    const OFFDIAG_Z_THRESHOLD: f64 = 4.0;
    let mut is_stat_diagonal = true;
    'stat: for a in 0..k {
        for b in (a + 1)..k {
            let stat_se = (diag[a].max(0.0) * diag[b].max(0.0) / n_eff)
                .max(0.0)
                .sqrt();
            let threshold = numerical_floor.max(OFFDIAG_Z_THRESHOLD * stat_se);
            if cov[[a, b]].abs() > threshold {
                is_stat_diagonal = false;
                break 'stat;
            }
        }
    }
    if is_stat_diagonal {
        Ok(MarginalSlopeCovariance::Diagonal(cov.diag().to_owned()))
    } else {
        Ok(MarginalSlopeCovariance::Full(cov))
    }
}

pub fn marginal_slope_preserving_scale(
    slopes: &[f64],
    covariance: &MarginalSlopeCovariance,
    probit_scale: f64,
) -> Result<f64, String> {
    if !probit_scale.is_finite() {
        return Err(format!(
            "marginal-slope probit scale must be finite, got {probit_scale}"
        ));
    }
    let observed_slopes = slopes
        .iter()
        .map(|&slope| probit_scale * slope)
        .collect::<Vec<_>>();
    let variance = covariance.quadratic_form(&observed_slopes)?;
    Ok((1.0 + variance).sqrt())
}

pub fn marginal_slope_probit_eta(
    q: f64,
    z: &[f64],
    slopes: &[f64],
    covariance: &MarginalSlopeCovariance,
    probit_scale: f64,
) -> Result<f64, String> {
    if z.len() != slopes.len() {
        return Err(format!(
            "marginal-slope score/slope dimension mismatch: z={}, slopes={}",
            z.len(),
            slopes.len()
        ));
    }
    if slopes.len() != covariance.dim() {
        return Err(format!(
            "marginal-slope covariance dimension mismatch: slopes={}, covariance={}",
            slopes.len(),
            covariance.dim()
        ));
    }
    if !q.is_finite() || z.iter().any(|value| !value.is_finite()) {
        return Err("marginal-slope probit eta inputs must be finite".to_string());
    }
    let scale = marginal_slope_preserving_scale(slopes, covariance, probit_scale)?;
    let linear = z
        .iter()
        .zip(slopes.iter())
        .map(|(&score, &slope)| probit_scale * slope * score)
        .sum::<f64>();
    Ok(q * scale + linear)
}

/// Log-space residual evaluator for the empirical-frailty intercept calibration.
///
/// Solves, in log-space, the strictly-increasing equation
///
///   F(a) = log Σᵢ wᵢ Φ(a + b·zᵢ) − log μ★ = 0,
///
/// where `b = rigid_observed_logslope(slope, probit_scale)` and `(zᵢ, wᵢ)` are
/// the supplied quadrature nodes and (positive) weights.
///
/// Mathematical structure of `F`:
///   • `F ∈ C^∞(ℝ)`.
///   • `F` is strictly increasing: `F'(a) = (Σ wᵢ φᵢ) / (Σ wᵢ Φᵢ) > 0` everywhere.
///   • `F(a) → −∞` as `a → −∞`; `F(a) → log(Σ wᵢ) − log μ★ ≥ 0` as `a → +∞`.
///   • Unique root `a★ ∈ ℝ` exists for every `μ★ ∈ (0, 1)`.
///
/// Why log-space: the linear-space residual `Σ wᵢ Φᵢ − μ★` and its derivative
/// `Σ wᵢ φᵢ` are sums of strictly-positive `exp(−η²/2)`-scaled terms. When the
/// seed `a` puts every quadrature node `ηᵢ = a + b·zᵢ` into the deep tail
/// (|ηᵢ| ≳ 38), every term rounds to 0.0 in IEEE-754 and the derivative
/// underflows to exactly zero — destroying Newton's update direction.  The
/// log-space formulation evaluates `log φ(η) = −η²/2 − ½ log 2π` (always finite
/// for any finite η) and `log Φ(η)` via the `erfcx`-based `normal_logcdf`
/// (also always finite for any finite η).  All sums are accumulated by
/// streaming log-sum-exp, so `F`, `F'`, and `F''` are finite for every finite
/// `a` and the global Newton/Halley iteration converges from any seed.
///
/// Returns `(F, F', F'')`.  In the deep left tail Newton converges linearly
/// (Mills ratio: `F'(a) ≈ |a|`, step ≈ `|a|/2`); near the root convergence is
/// quadratic with Newton or cubic with Halley.
pub(super) fn empirical_rigid_calibration_eval(
    intercept: f64,
    log_target_mu: f64,
    slope: f64,
    probit_scale: f64,
    nodes: &[f64],
    weights: &[f64],
) -> Result<(f64, f64, f64), String> {
    if !intercept.is_finite() {
        return Err(format!(
            "empirical latent calibration: non-finite intercept {intercept}"
        ));
    }
    let observed_slope = rigid_observed_logslope(slope, probit_scale);
    const HALF_LOG_2PI: f64 = 0.918_938_533_204_672_8; // 0.5 * ln(2π)

    // Streaming LSE accumulators for log Σ wᵢ φᵢ and log Σ wᵢ Φᵢ.
    let mut log_max_phi = f64::NEG_INFINITY;
    let mut sum_phi = 0.0_f64;
    let mut log_max_cdf = f64::NEG_INFINITY;
    let mut sum_cdf = 0.0_f64;

    // Streaming signed LSE for Σ wᵢ ηᵢ φᵢ, split into positive and negative
    // legs so the cancellation `pos − neg` happens once at the end on a
    // finite, well-scaled remainder.
    let mut log_max_pos = f64::NEG_INFINITY;
    let mut sum_pos = 0.0_f64;
    let mut log_max_neg = f64::NEG_INFINITY;
    let mut sum_neg = 0.0_f64;

    for (&node, &weight) in nodes.iter().zip(weights.iter()) {
        if !(weight.is_finite() && weight > 0.0) {
            continue;
        }
        let eta = intercept + observed_slope * node;
        if !eta.is_finite() {
            return Err(format!(
                "empirical latent calibration: non-finite η at intercept={intercept}, slope={slope}, node={node}"
            ));
        }
        let log_w = weight.ln();
        let log_phi = -0.5 * eta * eta - HALF_LOG_2PI;
        let log_term_phi = log_w + log_phi;
        let log_term_cdf = log_w + normal_logcdf(eta);

        lse_accumulate(&mut log_max_phi, &mut sum_phi, log_term_phi);
        lse_accumulate(&mut log_max_cdf, &mut sum_cdf, log_term_cdf);

        if eta != 0.0 {
            let log_term_eta_phi = log_term_phi + eta.abs().ln();
            if eta > 0.0 {
                lse_accumulate(&mut log_max_pos, &mut sum_pos, log_term_eta_phi);
            } else {
                lse_accumulate(&mut log_max_neg, &mut sum_neg, log_term_eta_phi);
            }
        }
    }

    if !(sum_phi.is_finite() && sum_cdf.is_finite() && sum_phi > 0.0 && sum_cdf > 0.0) {
        return Err(format!(
            "empirical latent calibration: log-space accumulation failed (sum_phi={sum_phi}, sum_cdf={sum_cdf}, intercept={intercept})"
        ));
    }

    let log_s_phi = log_max_phi + sum_phi.ln();
    let log_s_cdf = log_max_cdf + sum_cdf.ln();

    // F = log Σ wᵢ Φᵢ − log μ★
    let f = log_s_cdf - log_target_mu;
    // F' = exp(log Σ wᵢ φᵢ − log Σ wᵢ Φᵢ).
    //
    // F' is mathematically strictly positive everywhere — `Σ wᵢ φᵢ` and
    // `Σ wᵢ Φᵢ` are both sums of strictly-positive terms with positive weights.
    // In the far right tail, Mills ratio gives `φᵢ/Φᵢ → 0` exponentially, so
    // `log F' → −∞` and `(log F').exp()` IEEE-underflows to 0.0. Mathematically
    // it is a tiny positive number; floor it at `f64::MIN_POSITIVE` so the
    // monotone-root solver sees a strictly-positive derivative and routes
    // through its bracket-by-doubling phase (which only needs the *sign* of
    // `F'`, not its magnitude). Newton would propose `Δa = −F/F' = ±∞`, the
    // solver detects that and falls through to bracketing automatically.
    let log_f_prime = log_s_phi - log_s_cdf;
    let f_prime = if log_f_prime > -740.0 {
        log_f_prime.exp()
    } else {
        f64::MIN_POSITIVE
    };

    // F'' = (d/da)(S_φ/S_Φ) = (S_φ' S_Φ − S_φ²)/S_Φ²
    //     = −(Σ wᵢ ηᵢ φᵢ)/S_Φ − (F')²
    // The η-weighted sum is cancellation-prone; combine its positive and
    // negative legs against the same `log_s_cdf` reference so the subtraction
    // happens on dimensionless quantities of bounded magnitude. When the ratio
    // also underflows (deep tail), the result is a clean numerical zero —
    // Halley reduces to Newton, which is what the solver does anyway.
    let exp_safe = |log_x: f64| -> f64 { if log_x > -740.0 { log_x.exp() } else { 0.0 } };
    let pos_over_cdf = if sum_pos > 0.0 {
        exp_safe(log_max_pos + sum_pos.ln() - log_s_cdf)
    } else {
        0.0
    };
    let neg_over_cdf = if sum_neg > 0.0 {
        exp_safe(log_max_neg + sum_neg.ln() - log_s_cdf)
    } else {
        0.0
    };
    let s_etaphi_over_s_cdf = pos_over_cdf - neg_over_cdf;
    let f_double_prime = -s_etaphi_over_s_cdf - f_prime * f_prime;

    if !(f.is_finite() && f_prime.is_finite() && f_prime > 0.0 && f_double_prime.is_finite()) {
        return Err(format!(
            "empirical latent calibration: non-finite log-space state f={f}, f'={f_prime}, f''={f_double_prime} at intercept={intercept}"
        ));
    }
    Ok((f, f_prime, f_double_prime))
}

pub(crate) fn empirical_intercept_from_marginal(
    target_mu: f64,
    target_q: f64,
    slope: f64,
    probit_scale: f64,
    nodes: &[f64],
    weights: &[f64],
    initial: Option<f64>,
) -> Result<f64, String> {
    if !(target_mu.is_finite() && target_mu > 0.0 && target_mu < 1.0) {
        return Err(format!(
            "empirical latent calibration requires target mu in (0,1), got {target_mu}"
        ));
    }
    let log_target_mu = target_mu.ln();
    let closed_form_seed = rigid_intercept_from_marginal(target_q, slope, probit_scale);
    let seed = initial.unwrap_or(closed_form_seed);
    let eval = |a: f64| {
        empirical_rigid_calibration_eval(a, log_target_mu, slope, probit_scale, nodes, weights)
    };
    // Convergence is on the log-space residual |F| = |log Σ wᵢ Φᵢ − log μ★|.
    // Near the root this is the relative error in the calibrated probability,
    // so 1e-13 in log-space corresponds to absolute residual μ★ · 1e-13 in
    // linear space — strictly tighter than the legacy 1e-13 absolute tolerance
    // for every μ★ ∈ (0, 1). The 4·ε floor keeps the contract meaningful when
    // μ★ approaches 1 (where log Σ Φᵢ approaches 0).
    let abs_tol = 1e-13_f64.max(4.0 * f64::EPSILON);
    let solve_from = |s: f64| {
        super::monotone_root::solve_monotone_root(
            eval,
            s,
            "empirical latent intercept",
            abs_tol,
            64,
            48,
        )
        // Enclosing fn emits its own format!() rejection errors as String,
        // so the public return type stays Result<_, String>.
        .map_err(|e| e.to_string())
    };
    // A cached warm start can be poisoned across iterations: the per-row
    // `intercept_warm_starts` slot is shared by reference across line-search
    // trials and across outer-search seed validations, and is written after
    // every successful row-solve — including from rejected line-search trials
    // whose β/slope was wild. When that stale `a` is paired with the current
    // (much smaller) slope, the bracket-by-doubling phase can exhaust its
    // budget without crossing zero. Fall back to the deterministic
    // closed-form seed, which depends only on the current `(target_q, slope)`
    // and is bounded by the analytic rigid-probit geometry, so the cache
    // remains a pure speedup that cannot poison correctness.
    let (root, _, f_best) = match solve_from(seed) {
        Ok(v) => v,
        Err(first_err) => {
            if seed == closed_form_seed {
                return Err(first_err);
            }
            solve_from(closed_form_seed).map_err(|retry_err| {
                format!("{first_err}; closed-form retry from a={closed_form_seed:.6}: {retry_err}")
            })?
        }
    };
    if f_best.abs() > abs_tol {
        return Err(format!(
            "empirical latent intercept solve failed: log-residual={f_best:.3e} at a={root:.6}, target mu={target_mu:.6}"
        ));
    }
    Ok(root)
}

/// Rigid probit scalar kernel: closed-form derivatives up to 4th order.
///
/// η = q·c(g) + s_f·g·z,  c(g) = √(1+(s_f g)²),  s = 2y−1,  m = s·η.
/// u_k absorb weight and sign: u1=w·s·κ₁, u2=w·κ₂, u3=w·s·κ₃, u4=w·κ₄.
pub(super) struct RigidProbitKernel {
    pub(super) logcdf: f64,
    pub(super) u1: f64,
    pub(super) u2: f64,
    pub(super) u3: f64,
    pub(super) u4: f64,
    pub(super) c1: f64,
    pub(super) c2: f64,
    pub(super) c3: f64,
    pub(super) c4: f64,
    pub(super) eta_q: f64,
    pub(super) eta_g: f64,
}

impl RigidProbitKernel {
    #[inline]
    pub(super) fn new(q: f64, g: f64, z: f64, y: f64, w: f64, probit_scale: f64) -> Result<Self, String> {
        let s = 2.0 * y - 1.0;
        let observed_logslope = rigid_observed_logslope(g, probit_scale);
        let g2 = observed_logslope * observed_logslope;
        let c = (1.0 + g2).sqrt();
        let c1 = probit_scale * observed_logslope / c;
        let c_inv3 = 1.0 / (c * c * c);
        let c_inv5 = c_inv3 / (c * c);
        let c_inv7 = c_inv5 / (c * c);
        let eta = marginal_slope_standard_normal_scalar_eta(q, g, z, probit_scale);
        let m = s * eta;
        let (logcdf, _) = signed_probit_logcdf_and_mills_ratio(m);
        let (k1, k2, k3, k4) = signed_probit_neglog_derivatives_up_to_fourth(m, w)?;
        Ok(Self {
            logcdf,
            u1: s * k1,
            u2: k2,
            u3: s * k3,
            u4: k4,
            c1,
            c2: probit_scale * probit_scale * c_inv3,
            c3: -3.0 * probit_scale.powi(3) * observed_logslope * c_inv5,
            c4: probit_scale.powi(4) * (12.0 * g2 - 3.0) * c_inv7,
            eta_q: c,
            eta_g: q * c1 + probit_scale * z,
        })
    }

    /// Objective-only fast path for the rigid probit kernel: returns just
    /// `-w · log Φ(s · η)` (the row negative log-likelihood) without any
    /// derivative-state computation.
    ///
    /// **Mathematically identical** to `RigidProbitKernel::new(...)?.logcdf`
    /// scaled by `-w`: same algebraic form for `η = q·c(g) + s_f·g·z`, same
    /// signed-probit log-CDF kernel. The skipped work is the chain-rule
    /// scaffolding (`u1..u4`, `c1..c4`, `eta_q`, `eta_g`) that downstream
    /// gradient/Hessian assembly needs but the line-search accept/reject
    /// decision never touches.
    ///
    /// Used by [`BernoulliMarginalSlopeFamily::rigid_row_neglog_only`] from
    /// the rigid-path log-likelihood-only loop in
    /// [`log_likelihood_only_with_options`].
    #[inline]
    pub(super) fn neglog_only(
        q: f64,
        g: f64,
        z: f64,
        y: f64,
        w: f64,
        probit_scale: f64,
    ) -> Result<f64, String> {
        let s = 2.0 * y - 1.0;
        let eta = marginal_slope_standard_normal_scalar_eta(q, g, z, probit_scale);
        let m = s * eta;
        let (logcdf, _) = signed_probit_logcdf_and_mills_ratio(m);
        if !logcdf.is_finite() {
            return Err(format!(
                "rigid probit neglog_only: non-finite log Φ at q={q}, g={g}, z={z}, y={y}"
            ));
        }
        Ok(-w * logcdf)
    }

    #[inline]
    pub(super) fn primary_hessian(&self, q: f64) -> [[f64; 2]; 2] {
        let h00 = self.u2 * self.eta_q * self.eta_q;
        let h01 = self.u2 * self.eta_q * self.eta_g + self.u1 * self.c1;
        let h11 = self.u2 * self.eta_g * self.eta_g + self.u1 * q * self.c2;
        [[h00, h01], [h01, h11]]
    }

    #[inline]
    pub(super) fn third_contracted(&self, q: f64, dq: f64, dg: f64) -> [[f64; 2]; 2] {
        let dd = self.eta_q * dq + self.eta_g * dg;
        let dd_q = self.c1 * dg;
        let dd_g = self.c1 * dq + q * self.c2 * dg;
        let dd_qg = self.c2 * dg;
        let dd_gg = self.c2 * dq + q * self.c3 * dg;
        let t00 = self.u3 * self.eta_q * self.eta_q * dd + self.u2 * 2.0 * self.eta_q * dd_q;
        let t01 = self.u3 * self.eta_q * self.eta_g * dd
            + self.u2 * (self.c1 * dd + self.eta_q * dd_g + self.eta_g * dd_q)
            + self.u1 * dd_qg;
        let t11 = self.u3 * self.eta_g * self.eta_g * dd
            + self.u2 * (q * self.c2 * dd + 2.0 * self.eta_g * dd_g)
            + self.u1 * dd_gg;
        [[t00, t01], [t01, t11]]
    }

    #[inline]
    pub(super) fn fourth_contracted(&self, q: f64, uq: f64, ug: f64, vq: f64, vg: f64) -> [[f64; 2]; 2] {
        let du = self.eta_q * uq + self.eta_g * ug;
        let dv = self.eta_q * vq + self.eta_g * vg;
        let du_a = [self.c1 * ug, self.c1 * uq + q * self.c2 * ug];
        let dv_a = [self.c1 * vg, self.c1 * vq + q * self.c2 * vg];
        let du_ab = [
            [0.0, self.c2 * ug],
            [self.c2 * ug, self.c2 * uq + q * self.c3 * ug],
        ];
        let dv_ab = [
            [0.0, self.c2 * vg],
            [self.c2 * vg, self.c2 * vq + q * self.c3 * vg],
        ];
        let dduv = self.c1 * (uq * vg + ug * vq) + q * self.c2 * ug * vg;
        let dduv_a = [
            self.c2 * ug * vg,
            self.c2 * (uq * vg + ug * vq) + q * self.c3 * ug * vg,
        ];
        let dduv_ab = [
            [0.0, self.c3 * ug * vg],
            [
                self.c3 * ug * vg,
                self.c3 * (uq * vg + ug * vq) + q * self.c4 * ug * vg,
            ],
        ];
        let eta_a = [self.eta_q, self.eta_g];
        let eta_ab = [[0.0, self.c1], [self.c1, q * self.c2]];
        let mut f = [[0.0f64; 2]; 2];
        for a in 0..2 {
            for b in a..2 {
                let val = self.u4 * eta_a[a] * eta_a[b] * du * dv
                    + self.u3
                        * (eta_ab[a][b] * du * dv
                            + du_a[a] * eta_a[b] * dv
                            + dv_a[a] * eta_a[b] * du
                            + du_a[b] * eta_a[a] * dv
                            + dv_a[b] * eta_a[a] * du
                            + dduv * eta_a[a] * eta_a[b])
                    + self.u2
                        * (eta_ab[a][b] * dduv
                            + du_a[a] * dv_a[b]
                            + dv_a[a] * du_a[b]
                            + du_ab[a][b] * dv
                            + dv_ab[a][b] * du
                            + eta_a[b] * dduv_a[a]
                            + eta_a[a] * dduv_a[b])
                    + self.u1 * dduv_ab[a][b];
                f[a][b] = val;
                f[b][a] = val;
            }
        }
        f
    }
}

#[inline]
pub(super) fn rigid_transformed_gradient(
    marginal: BernoulliMarginalLinkMap,
    kernel: &RigidProbitKernel,
) -> [f64; 2] {
    [
        kernel.u1 * kernel.eta_q * marginal.q1,
        kernel.u1 * kernel.eta_g,
    ]
}

#[inline]
pub(super) fn rigid_transformed_hessian(
    marginal: BernoulliMarginalLinkMap,
    kernel: &RigidProbitKernel,
) -> [[f64; 2]; 2] {
    let h_q = kernel.primary_hessian(marginal.q);
    let grad_q = kernel.u1 * kernel.eta_q;
    [
        [
            h_q[0][0] * marginal.q1 * marginal.q1 + grad_q * marginal.q2,
            h_q[0][1] * marginal.q1,
        ],
        [h_q[1][0] * marginal.q1, h_q[1][1]],
    ]
}

#[inline]
pub(super) fn rigid_internal_third_components(
    marginal: BernoulliMarginalLinkMap,
    kernel: &RigidProbitKernel,
) -> (f64, f64, f64, f64) {
    let q_dir = kernel.third_contracted(marginal.q, 1.0, 0.0);
    let g_dir = kernel.third_contracted(marginal.q, 0.0, 1.0);
    (q_dir[0][0], q_dir[0][1], q_dir[1][1], g_dir[1][1])
}

#[inline]
/// Closed-form rigid third-derivative tensor — uncontracted in primary space
/// `(η, g)`. Indexed `T[a][b][c]` with a/b/c ∈ {0=η, 1=g}; the tensor is
/// fully symmetric in its three indices, so only four distinct values appear:
/// `T_ηηη`, `T_ηηg`, `T_ηgg`, `T_ggg`.
///
/// This is the axis-invariant building block: a single per-row evaluation
/// produces a tensor that every ψ-axis then contracts cheaply via
/// [`contract_third_full`]. The previous design recomputed the
/// already-contracted matrix per axis, paying the heavy primary-derivative
/// machinery `n_axes` times per row.
pub(super) fn rigid_transformed_third_full(
    marginal: BernoulliMarginalLinkMap,
    kernel: &RigidProbitKernel,
) -> [[[f64; 2]; 2]; 2] {
    let h_q = kernel.primary_hessian(marginal.q);
    let grad_q = kernel.u1 * kernel.eta_q;
    let (f_qqq, f_qqg, f_qgg, f_ggg) = rigid_internal_third_components(marginal, kernel);
    let f_etaetaeta =
        f_qqq * marginal.q1_cu + 3.0 * h_q[0][0] * marginal.q1 * marginal.q2 + grad_q * marginal.q3;
    let f_etaetag = f_qqg * marginal.q1_sq + h_q[0][1] * marginal.q2;
    let f_etagg = f_qgg * marginal.q1;
    third_full_from_symmetric_components(f_etaetaeta, f_etaetag, f_etagg, f_ggg)
}

/// Pack the four independent components of a fully-symmetric 3-tensor on
/// a 2-dim primary space into the dense `[[[f64; 2]; 2]; 2]` representation
/// callers slice. Index ordering follows `T[a][b][c]` = ∂³f/∂p_a ∂p_b ∂p_c.
#[inline]
pub(super) fn third_full_from_symmetric_components(
    t_qqq: f64,
    t_qqg: f64,
    t_qgg: f64,
    t_ggg: f64,
) -> [[[f64; 2]; 2]; 2] {
    let mut t = [[[0.0; 2]; 2]; 2];
    t[0][0][0] = t_qqq;
    t[0][0][1] = t_qqg;
    t[0][1][0] = t_qqg;
    t[1][0][0] = t_qqg;
    t[0][1][1] = t_qgg;
    t[1][0][1] = t_qgg;
    t[1][1][0] = t_qgg;
    t[1][1][1] = t_ggg;
    t
}

/// Contract a symmetric 3-tensor on its third index with a primary-space
/// direction `d = (d_eta, d_g)`, producing the symmetric 2×2 contracted
/// matrix the outer-derivative pipeline consumes:
///   `M[a][b] = Σ_c T[a][b][c] · d[c]`.
#[inline]
pub(super) fn contract_third_full(t: &[[[f64; 2]; 2]; 2], d_eta: f64, d_g: f64) -> [[f64; 2]; 2] {
    [
        [
            t[0][0][0] * d_eta + t[0][0][1] * d_g,
            t[0][1][0] * d_eta + t[0][1][1] * d_g,
        ],
        [
            t[1][0][0] * d_eta + t[1][0][1] * d_g,
            t[1][1][0] * d_eta + t[1][1][1] * d_g,
        ],
    ]
}

/// Closed-form rigid fourth-derivative tensor — uncontracted in primary
/// space `(η, g)`. Indexed `T[a][b][c][d]` with each index ∈ {0=η, 1=g};
/// fully symmetric in all four indices, so only five distinct values appear:
/// `T_ηηηη`, `T_ηηηg`, `T_ηηgg`, `T_ηggg`, `T_gggg`.
///
/// Same structural role as [`rigid_transformed_third_full`] one order up:
/// the 5 axis-invariant components are computed once per row, and every
/// (u, v) ψ-axis pair then folds them into a 2×2 matrix via the cheap
/// [`contract_fourth_full`]. The previous design re-derived all five
/// components per (row, axis-pair), or O(rank²) per row.
pub(super) fn rigid_transformed_fourth_full(
    marginal: BernoulliMarginalLinkMap,
    kernel: &RigidProbitKernel,
) -> [[[[f64; 2]; 2]; 2]; 2] {
    let h_q = kernel.primary_hessian(marginal.q);
    let grad_q = kernel.u1 * kernel.eta_q;
    let (f_qqq, f_qqg, f_qgg, _) = rigid_internal_third_components(marginal, kernel);
    let qq = kernel.fourth_contracted(marginal.q, 1.0, 0.0, 1.0, 0.0);
    let qg = kernel.fourth_contracted(marginal.q, 1.0, 0.0, 0.0, 1.0);
    let gg = kernel.fourth_contracted(marginal.q, 0.0, 1.0, 0.0, 1.0);
    let f_qqqq = qq[0][0];
    let f_qqqg = qq[0][1];
    let f_qqgg = qq[1][1];
    let f_qggg = qg[1][1];
    let f_gggg = gg[1][1];
    let f_eta4 = f_qqqq * marginal.q1_q
        + 6.0 * f_qqq * marginal.q1_sq * marginal.q2
        + 3.0 * h_q[0][0] * marginal.q2 * marginal.q2
        + 4.0 * h_q[0][0] * marginal.q1 * marginal.q3
        + grad_q * marginal.q4;
    let f_eta3g =
        f_qqqg * marginal.q1_cu + 3.0 * f_qqg * marginal.q1 * marginal.q2 + h_q[0][1] * marginal.q3;
    let f_eta2g2 = f_qqgg * marginal.q1_sq + f_qgg * marginal.q2;
    let f_etag3 = f_qggg * marginal.q1;
    fourth_full_from_symmetric_components(f_eta4, f_eta3g, f_eta2g2, f_etag3, f_gggg)
}

/// Pack the five independent components of a fully-symmetric 4-tensor on a
/// 2-dim primary space into the dense `[[[[f64; 2]; 2]; 2]; 2]` form.
/// Index ordering: `T[a][b][c][d]` = ∂⁴f/∂p_a ∂p_b ∂p_c ∂p_d, symmetric in
/// all four indices, so each of the 16 entries is fixed by the multi-set
/// `{#η, #g}` of its index pattern.
#[inline]
pub(super) fn fourth_full_from_symmetric_components(
    t_qqqq: f64,
    t_qqqg: f64,
    t_qqgg: f64,
    t_qggg: f64,
    t_gggg: f64,
) -> [[[[f64; 2]; 2]; 2]; 2] {
    let mut t = [[[[0.0; 2]; 2]; 2]; 2];
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                for d in 0..2 {
                    let g_count = a + b + c + d; // count of `g` indices, 0..=4
                    // a,b,c,d ∈ {0,1} so g_count ∈ 0..=4; the final arm catches
                    // any unexpected value defensively without panicking.
                    t[a][b][c][d] = match g_count {
                        0 => t_qqqq,
                        1 => t_qqqg,
                        2 => t_qqgg,
                        3 => t_qggg,
                        _ => t_gggg,
                    };
                }
            }
        }
    }
    t
}

/// Contract a symmetric 4-tensor on its last two indices with two
/// primary-space directions `u = (u_eta, u_g)` and `v = (v_eta, v_g)`,
/// producing the symmetric 2×2 matrix the outer-Hessian pipeline expects:
///   `M[a][b] = Σ_{c,d} T[a][b][c][d] · u[c] · v[d]`.
#[inline]
pub(super) fn contract_fourth_full(
    t: &[[[[f64; 2]; 2]; 2]; 2],
    u_eta: f64,
    u_g: f64,
    v_eta: f64,
    v_g: f64,
) -> [[f64; 2]; 2] {
    let mut out = [[0.0; 2]; 2];
    for a in 0..2 {
        for b in 0..2 {
            let mut sum = 0.0;
            sum += t[a][b][0][0] * u_eta * v_eta;
            sum += t[a][b][0][1] * u_eta * v_g;
            sum += t[a][b][1][0] * u_g * v_eta;
            sum += t[a][b][1][1] * u_g * v_g;
            out[a][b] = sum;
        }
    }
    out
}

pub(super) fn ensure_finite_third_full_cache_row(t: &[[[f64; 2]; 2]; 2], context: &str) -> Result<(), String> {
    if t.iter().flatten().flatten().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(format!(
            "{context}: warmed third-derivative cache row contains a non-finite value"
        ))
    }
}

pub(super) fn ensure_finite_fourth_full_cache_row(
    t: &[[[[f64; 2]; 2]; 2]; 2],
    context: &str,
) -> Result<(), String> {
    if t.iter()
        .flatten()
        .flatten()
        .flatten()
        .all(|value| value.is_finite())
    {
        Ok(())
    } else {
        Err(format!(
            "{context}: warmed fourth-derivative cache row contains a non-finite value"
        ))
    }
}

pub(crate) fn unary_derivatives_sqrt(x: f64) -> [f64; 5] {
    let s = x.max(1e-300).sqrt();
    let x1 = x.max(1e-300);
    let x2 = x1 * x1;
    let x3 = x2 * x1;
    [
        s,
        0.5 / s,
        -0.25 / (x1 * s),
        3.0 / (8.0 * x2 * s),
        -15.0 / (16.0 * x3 * s),
    ]
}
pub(crate) fn unary_derivatives_neglog_phi(x: f64, weight: f64) -> [f64; 5] {
    if weight == 0.0 || x == f64::INFINITY {
        return [0.0, 0.0, 0.0, 0.0, 0.0];
    }
    if x == f64::NEG_INFINITY {
        return [f64::INFINITY, f64::NEG_INFINITY, weight, 0.0, 0.0];
    }
    if x.is_nan() {
        return [f64::NAN; 5];
    }
    let (d1, d2, d3, d4) = signed_probit_neglog_derivatives_up_to_fourth_numeric(x, weight);
    let (log_cdf, _) = signed_probit_logcdf_and_mills_ratio(x);
    [-weight * log_cdf, d1, d2, d3, d4]
}

/// Derivatives of log(x) through 4th order.
pub(crate) fn unary_derivatives_log(x: f64) -> [f64; 5] {
    let x1 = x.max(1e-300);
    let x2 = x1 * x1;
    let x3 = x2 * x1;
    let x4 = x3 * x1;
    [x1.ln(), 1.0 / x1, -1.0 / x2, 2.0 / x3, -6.0 / x4]
}

/// Derivatives of log φ(x) = -½x² - ½ln(2π) through 4th order.
pub(crate) fn unary_derivatives_log_normal_pdf(x: f64) -> [f64; 5] {
    let c = 0.5 * (2.0 * std::f64::consts::PI).ln();
    [-0.5 * x * x - c, -x, -1.0, 0.0, 0.0]
}

use super::family::clamp_bernoulli_link_probability;
use super::*;
use crate::families::jet_tower::Tower4;
use crate::matrix::{LinearOperator, SignedWeightsView};

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

// ── Pooled 2-D probit pilot Newton solver tuning ─────────────────────────────
//
// `pooled_probit_baseline` solves a 2-parameter (intercept, slope) penalised
// probit by damped Newton. The values below are the standard convergence /
// safeguard knobs; they are deliberately conservative because the pilot is a
// cheap warm-start for the full fit, not the production estimator.

/// Maximum damped-Newton outer iterations for the pooled probit pilot. A 2-D
/// strictly-convex probit converges in well under this; the cap only guards a
/// pathological non-finite data configuration.
const POOLED_PILOT_MAX_NEWTON_ITERS: usize = 50;
/// Initial Levenberg ridge added to the 2×2 Hessian diagonal before the solve.
pub(crate) const POOLED_PILOT_RIDGE_INIT: f64 = 1e-8;
/// Below this absolute determinant the ridged 2×2 system is treated as
/// singular and the ridge is escalated.
pub(crate) const POOLED_PILOT_DET_FLOOR: f64 = 1e-18;
/// Geometric factor by which the ridge grows when the system is singular.
pub(crate) const POOLED_PILOT_RIDGE_GROWTH: f64 = 10.0;
/// Ridge ceiling; exceeding it means the Hessian is unusable and the pilot
/// fails rather than returning a meaningless step.
pub(crate) const POOLED_PILOT_RIDGE_MAX: f64 = 1e6;
/// Maximum backtracking-line-search halvings per Newton step.
const POOLED_PILOT_MAX_BACKTRACKS: usize = 25;
/// Backtracking step contraction factor.
pub(crate) const POOLED_PILOT_BACKTRACK_SHRINK: f64 = 0.5;
/// Objective-change tolerance below which a stalled (rejected) line search is
/// accepted as converged instead of erroring.
pub(crate) const POOLED_PILOT_STALL_TOL: f64 = 1e-10;
/// Minimum-magnitude signed slope returned by the pilot, so the downstream
/// `b/√(1+b²)` rigid seed never collapses to an exactly flat (zero-slope) link.
pub(crate) const POOLED_PILOT_MIN_ABS_SLOPE: f64 = 1e-6;

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
    for _ in 0..POOLED_PILOT_MAX_NEWTON_ITERS {
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
        let mut ridge = POOLED_PILOT_RIDGE_INIT;
        let (step0, step1) = loop {
            let h00_r = h00 + ridge;
            let h11_r = h11 + ridge;
            let det = h00_r * h11_r - h01 * h01;
            if det.is_finite() && det.abs() > POOLED_PILOT_DET_FLOOR {
                let s0 = (h11_r * g0 - h01 * g1) / det;
                let s1 = (-h01 * g0 + h00_r * g1) / det;
                if s0.is_finite() && s1.is_finite() {
                    break (s0, s1);
                }
            }
            ridge *= POOLED_PILOT_RIDGE_GROWTH;
            if ridge > POOLED_PILOT_RIDGE_MAX {
                return Err(
                    "pooled bernoulli-marginal-slope pilot Hessian solve failed".to_string()
                );
            }
        };
        let mut accepted = false;
        let mut step_scale = 1.0;
        for _ in 0..POOLED_PILOT_MAX_BACKTRACKS {
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
            step_scale *= POOLED_PILOT_BACKTRACK_SHRINK;
        }
        if !accepted {
            if (obj_prev - obj).abs() < POOLED_PILOT_STALL_TOL {
                break;
            }
            return Err("pooled bernoulli-marginal-slope pilot line search failed".to_string());
        }
    }
    let a = beta0;
    // Signed slope: preserve direction from pilot probit.
    let b = if beta1.abs() < POOLED_PILOT_MIN_ABS_SLOPE {
        if beta1.is_sign_negative() {
            -POOLED_PILOT_MIN_ABS_SLOPE
        } else {
            POOLED_PILOT_MIN_ABS_SLOPE
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
// — `<<1 s` at large scale because `p_marg` is `O(10²)` whereas the
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

/// Tikhonov ridge for the pilot IRLS marginal solve, as a fraction of the mean
/// Hessian diagonal: `ridge = PILOT_RIDGE_DIAG_FRACTION * max(mean_diag, floor)`.
/// Scaling by the diagonal keeps the ridge scale-invariant; the fraction is
/// small enough to be numerically negligible against a well-conditioned design
/// yet still regularise a near-singular pilot Gram.
pub(crate) const PILOT_RIDGE_DIAG_FRACTION: f64 = 1e-6;
/// Positivity floor on the mean Hessian diagonal used to scale the pilot ridge,
/// so a degenerate (all-zero-diagonal) Gram still receives a tiny ridge.
pub(crate) const PILOT_RIDGE_DIAG_FLOOR: f64 = 1e-12;

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
    let mut xtwx = marginal_design.xt_diag_x_signed_op(SignedWeightsView::from_array(&w_irls))?;
    let trace_diag: f64 = (0..p_marg).map(|i| xtwx[[i, i]]).sum();
    let ridge =
        (trace_diag / p_marg as f64).max(PILOT_RIDGE_DIAG_FLOOR) * PILOT_RIDGE_DIAG_FRACTION;
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
pub(crate) fn signed_probit_neglog_derivatives_up_to_fourth_numeric(
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

/// Fused exact value+derivative stack for the signed-probit negative-log
/// kernel: returns `[-w·logΦ(m), w·k1, w·k2, w·k3, w·k4]` in the `[f64; 5]`
/// shape [`Tower4::compose_unary`] consumes.
///
/// This is the single-source replacement for the two-call pattern
///
/// ```ignore
/// let (logcdf, _) = signed_probit_logcdf_and_mills_ratio(m);
/// let (k1, k2, k3, k4) = signed_probit_neglog_derivatives_up_to_fourth(m, w)?;
/// // → [-w*logcdf, k1, k2, k3, k4]
/// ```
///
/// which evaluated `signed_probit_logcdf_and_mills_ratio` TWICE on the same
/// `m` (once for `logΦ`, once again — discarding `logΦ` — for the Mills ratio
/// `λ` that drives `k1..k4`). On the rigid standard-normal BMS path that pair
/// of `erfcx`/`erfc` transcendentals is the dominant per-row arithmetic across
/// all `n ≈ 356k` rows, so collapsing it to ONE call halves the transcendental
/// budget of the jet build. The result is bit-identical: `logΦ` and `λ` are the
/// exact same values the two-call form produced (same branch, same `ex`), and
/// `k1..k4` are the same polynomials in `(m, λ)`.
///
/// Boundary semantics match [`unary_derivatives_neglog_phi`] (the prior
/// two-call form): `+∞` is the saturated zero tail (all zero); `−∞` returns the
/// `[+∞, −w, w·0, 0, 0]` limit (value `−w·logΦ(−∞)=+∞`, `k1=−λ→−∞` scaled by the
/// `w` already folded by the numeric derivative helper); `NaN` propagates.
#[inline]
pub(crate) fn signed_probit_neglog_unary_stack(signed_margin: f64, weight: f64) -> [f64; 5] {
    if weight == 0.0 || signed_margin == f64::INFINITY {
        return [0.0; 5];
    }
    if signed_margin == f64::NEG_INFINITY {
        // logΦ(−∞) = −∞ ⇒ value −w·(−∞) = +∞; the derivative helper's −∞ limit
        // is (−∞, w, 0, 0) for (k1, k2, k3, k4) before the weight fold below.
        return [f64::INFINITY, f64::NEG_INFINITY, weight, 0.0, 0.0];
    }
    if signed_margin.is_nan() {
        return [f64::NAN; 5];
    }
    // ONE transcendental evaluation feeds both the value (logΦ) and every
    // derivative (through the Mills ratio λ).
    let (logcdf, lambda) = signed_probit_logcdf_and_mills_ratio(signed_margin);
    let m = signed_margin;
    let k1 = -lambda;
    let k2 = lambda * (m + lambda);
    let k3 = lambda * (1.0 - m * m - 3.0 * m * lambda - 2.0 * lambda * lambda);
    let k4 = lambda
        * ((m * m * m - 3.0 * m)
            + (7.0 * m * m - 4.0) * lambda
            + 12.0 * m * lambda * lambda
            + 6.0 * lambda * lambda * lambda);
    [
        -weight * logcdf,
        weight * k1,
        weight * k2,
        weight * k3,
        weight * k4,
    ]
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
pub(super) fn rigid_intercept_from_marginal(
    marginal_eta: f64,
    logslope: f64,
    probit_scale: f64,
) -> f64 {
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
pub(super) fn rigid_observed_eta(
    marginal_eta: f64,
    logslope: f64,
    z: f64,
    probit_scale: f64,
) -> f64 {
    marginal_slope_standard_normal_scalar_eta(marginal_eta, logslope, z, probit_scale)
}

#[inline]
pub(super) fn marginal_slope_standard_normal_scalar_eta(
    q: f64,
    slope: f64,
    z: f64,
    probit_scale: f64,
) -> f64 {
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

/// Negative-side tolerance on the covariance quadratic form `rᵀΣr`. The form
/// is mathematically PSD but finite-precision accumulation in the dense / low-
/// rank sums can produce a tiny negative value at a true zero; results within
/// this tolerance are clamped to zero, anything more negative is a real error.
pub(crate) const COVARIANCE_QUADRATIC_FORM_PSD_TOL: f64 = -1e-10;

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
        if value.is_finite() && value >= COVARIANCE_QUADRATIC_FORM_PSD_TOL {
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
    //     not real correlation), fail ⇒ Full.  At large-scale N_eff the 4σ
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
        crate::families::monotone_root::solve_monotone_root(
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

#[inline]
pub(super) fn rigid_standard_normal_neglog_only(
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

/// The rigid standard-normal Bernoulli row negative log-likelihood, written
/// ONCE over the generic [`JetScalar`] interface (#932 scalar cutover).
///
/// Primaries `p = [q_eta = marginal η, g = slope]`. The body is exactly the
/// production likelihood — `ℓ = −w·logΦ((2y−1)·η)`, `η = q(η_marg)·√(1+(s·g)²)
/// + (s·g)·z` — composed with ONLY [`JetScalar`] ops, so it re-instantiates at
/// whatever order / representation a consumer needs:
///
/// * [`Order2`](super::super::jet_scalar::Order2) → `(v, g, H)`
///   ([`rigid_standard_normal_row_kernel`], the inner-Newton path);
/// * [`OneSeed`](super::super::jet_scalar::OneSeed) → contracted third
///   `Σ_c ℓ_{abc} dir_c` without materialising `t3` (the directional gate);
/// * [`TwoSeed`](super::super::jet_scalar::TwoSeed) → contracted fourth
///   `Σ_{cd} ℓ_{abcd} u_c v_d` without materialising `t4`;
/// * full [`Tower4`] → every uncontracted channel
///   ([`rigid_standard_normal_tower`], feeding the `third_full` / `fourth_full`
///   caches).
///
/// Every consumer derives from THIS one expression, so the value channel and
/// every derivative channel cannot desync (the #736 / #948 bug genus).
///
/// The marginal index `q(η_marg)` enters by composing the hand-certified link
/// derivative stack `[q, q1, q2, q3, q4]` onto the η primary (slot 0); the
/// margin transcendental enters by composing the certified
/// [`signed_probit_neglog_unary_stack`] onto the assembled signed margin — the
/// stability discipline of #932 (humans own primitive stability, the algebra
/// owns combinatorics). The caller MUST guard the signed-margin value against a
/// non-finite (non-`+∞`-excluded) NaN before calling; the seeded-evaluation
/// wrappers below do that.
#[inline]
pub(crate) fn rigid_standard_normal_row_nll_generic<S: crate::families::jet_scalar::JetScalar<2>>(
    p: &[S; 2],
    marginal: BernoulliMarginalLinkMap,
    z: f64,
    y: f64,
    w: f64,
    probit_scale: f64,
) -> Result<S, String> {
    // The order-≤4 signed observed margin `m = (2y−1)·η`, written ONCE in
    // `rigid_standard_normal_signed_margin` over `S: JetScalar<2>` and shared
    // verbatim with the batched builder's Pass-A jet (#932 single source).
    let signed = rigid_standard_normal_signed_margin(p, marginal, z, y, probit_scale);
    // Preserve the production fail-fast: a NaN (non-`+∞`) signed margin is an
    // upstream domain failure, not a tail saturation.
    let m = signed.value();
    if !(m.is_finite() || m == f64::INFINITY) {
        return Err(format!(
            "non-finite signed margin in rigid probit row NLL: {m}"
        ));
    }
    // NLL = −w·logΦ(m) via the fused single-Mills-ratio probit neglog stack.
    Ok(signed.compose_unary(signed_probit_neglog_unary_stack(m, w)))
}

/// The order-≤4 signed observed margin `m = (2y−1)·η` of one rigid
/// standard-normal Bernoulli row, written ONCE over `S: JetScalar<2>`:
/// `q(η_marg)` composed onto the η primary, observed slope `b = s·g`, scale
/// `c = √(1 + b²)`, `η = q·c + b·z`. This is the polynomial part shared by
/// every channel consumer — the per-row / contracted / full-tower generic NLL
/// ([`rigid_standard_normal_row_nll_generic`]) composes the probit-neglog
/// transcendental onto it, and the batched builder's Pass-A jet
/// ([`rigid_standard_normal_signed_jet`]) evaluates it at `Tower4<2>` — so the
/// signed margin has a single source (#932), with no second hand-packed jet.
#[inline]
pub(crate) fn rigid_standard_normal_signed_margin<
    S: crate::families::jet_scalar::JetScalar<2>,
>(
    p: &[S; 2],
    marginal: BernoulliMarginalLinkMap,
    z: f64,
    y: f64,
    probit_scale: f64,
) -> S {
    // q(η_marg): compose the link's q-as-function-of-η stack onto the η primary.
    let q = p[0].compose_unary([
        marginal.q,
        marginal.q1,
        marginal.q2,
        marginal.q3,
        marginal.q4,
    ]);
    let slope = p[1];
    // observed slope b = s·g, scale c = √(1 + b²).
    let observed_slope = slope.scale(probit_scale);
    let b2 = observed_slope.mul(&observed_slope);
    let c = b2.add(&S::constant(1.0)).sqrt();
    // η = q·c + (s·g)·z, signed margin m = (2y−1)·η.
    let eta = q.mul(&c).add(&observed_slope.scale(z));
    eta.scale(2.0 * y - 1.0)
}

/// One row of rigid standard-normal Bernoulli data as a generic
/// [`RowNllProgramGeneric<2>`] (#932 production wiring).
///
/// This is the genuine production consumer of the generic program seam: the row
/// NLL is written ONCE in [`rigid_standard_normal_row_nll_generic`] over
/// `S: JetScalar<2>`, and this single-row program routes it through the
/// [`crate::families::jet_tower`] `generic_*` evaluators
/// ([`generic_full_tower`](crate::families::jet_tower::generic_full_tower) for
/// the uncontracted tensors, and the cheap order-2 / contracted scalars for the
/// value/grad/Hessian and directional channels). Primaries are
/// `[marginal η, slope g]`; the marginal link map and per-row data
/// `(z, y, w, probit_scale)` enter as constants on the body.
pub(crate) struct RigidStandardNormalRow {
    pub(crate) marginal: BernoulliMarginalLinkMap,
    pub(crate) g: f64,
    pub(crate) z: f64,
    pub(crate) y: f64,
    pub(crate) w: f64,
    pub(crate) probit_scale: f64,
}

impl crate::families::jet_tower::RowNllProgramGeneric<2> for RigidStandardNormalRow {
    fn n_rows(&self) -> usize {
        1
    }

    fn primaries(&self, row: usize) -> Result<[f64; 2], String> {
        if row != 0 {
            return Err(format!("RigidStandardNormalRow: row {row} out of range"));
        }
        Ok([self.marginal.eta_value(), self.g])
    }

    fn row_nll_generic<S: crate::families::jet_scalar::JetScalar<2>>(
        &self,
        row: usize,
        p: &[S; 2],
    ) -> Result<S, String> {
        if row != 0 {
            return Err(format!("RigidStandardNormalRow: row {row} out of range"));
        }
        rigid_standard_normal_row_nll_generic(
            p,
            self.marginal,
            self.z,
            self.y,
            self.w,
            self.probit_scale,
        )
    }
}

#[inline]
pub(crate) fn rigid_standard_normal_tower(
    marginal: BernoulliMarginalLinkMap,
    g: f64,
    z: f64,
    y: f64,
    w: f64,
    probit_scale: f64,
) -> Result<Tower4<2>, String> {
    // #932 cutover: the full uncontracted tower comes from the SAME single
    // generic row-NLL expression every other channel consumer derives from,
    // routed through the generic program seam evaluated at the all-channels
    // `Tower4` scalar. `generic_full_tower` seeds `[marginal η, g]` exactly as
    // the previous inline `Tower4::variable` form did, so this is bit-identical
    // while giving `RowNllProgramGeneric` a genuine production consumer.
    let program = RigidStandardNormalRow {
        marginal,
        g,
        z,
        y,
        w,
        probit_scale,
    };
    crate::families::jet_tower::generic_full_tower(&program, 0)
}

/// Branch-free `signed`-margin jet for the rigid standard-normal row kernel.
///
/// This is the order-≤4 polynomial part of [`rigid_standard_normal_tower`]
/// *before* the single transcendental compose: it builds the `Tower4<2>` of the
/// signed observed index `signed = (2y−1)·η`, `η = q·c(g) + g·(s·z)`,
/// `c(g) = √(1 + (s·g)²)`, with no `erfc`/`exp`/`ln` call. Splitting this off
/// lets the batched builder run all the cheap, branch-free jet products in one
/// SIMD-friendly pass and isolate the (branchy, transcendental) Mills-ratio
/// composition into its own tight pass. The returned jet is the SAME expression
/// (`rigid_standard_normal_signed_margin`) the per-row `rigid_standard_normal_tower`
/// signed margin evaluates, here at `Tower4<2>` — bit-identical by construction,
/// not a parallel hand-packed jet (#932 single source).
#[inline]
fn rigid_standard_normal_signed_jet(
    marginal: BernoulliMarginalLinkMap,
    g: f64,
    z: f64,
    y: f64,
    probit_scale: f64,
) -> Tower4<2> {
    // Seed `[marginal η, g]` exactly as the generic program's `primaries()`, then
    // evaluate the one shared signed-margin expression at the all-channels scalar.
    let p = [
        Tower4::<2>::variable(marginal.eta_value(), 0),
        Tower4::<2>::variable(g, 1),
    ];
    rigid_standard_normal_signed_margin(&p, marginal, z, y, probit_scale)
}

/// Batched, two-pass builder of the rigid standard-normal row `Tower4<2>` jets
/// for a contiguous chunk of rows, written for the auto-vectorizer.
///
/// Per row the production path ([`rigid_standard_normal_tower`]) interleaves
/// (1) cheap branch-free jet products to form the `signed` margin jet, (2) ONE
/// branchy transcendental (`erfcx`/`exp`/`ln` via
/// [`signed_probit_neglog_unary_stack`]) that dominates the per-row scalar-ALU
/// budget across all `n ≈ 356k` rows, and (3) the branch-free Faà-di-Bruno
/// `compose_unary` tensor assembly. The interleaving keeps the compiler from
/// vectorizing the loop body because the transcendental's internal branches sit
/// between the two pure-FMA blocks.
///
/// This builder runs the same work as three *separate* loops over the chunk:
///
/// * Pass A — build every `signed` jet (branch-free, [`rigid_standard_normal_signed_jet`]),
///   spilling `signed.v` into a contiguous `margins` scratch buffer.
/// * Pass B — fill the per-row unary derivative stack `[d0..d4]` from
///   `margins`/`weights` (the transcendental, now back-to-back over a flat
///   `&[f64]` so branch prediction and the polynomial `k1..k4` portion stream).
/// * Pass C — `compose_unary` each `signed` jet against its stack (branch-free,
///   pure FMA over the dense tensors → the vectorizable hot block).
///
/// Every scalar operation, and its order, is identical to the per-row path, so
/// the produced jets are bit-for-bit equal; the win is making the n-row build
/// memory-bandwidth-bound rather than scalar-ALU/branch-bound. The `fill`
/// callback writes the consumer's per-row payload (e.g. `.t3` or `.t4`) from the
/// finished jet, so neither tensor is materialized into an intermediate `Vec`.
#[inline]
pub(super) fn rigid_standard_normal_towers_batch<T>(
    marginals: &[BernoulliMarginalLinkMap],
    slopes: &[f64],
    zs: &[f64],
    ys: &[f64],
    weights: &[f64],
    probit_scale: f64,
    out: &mut [T],
    mut fill: impl FnMut(&Tower4<2>) -> Result<T, String>,
) -> Result<(), String> {
    let chunk = marginals.len();
    if slopes.len() != chunk
        || zs.len() != chunk
        || ys.len() != chunk
        || weights.len() != chunk
        || out.len() != chunk
    {
        return Err(format!(
            "rigid_standard_normal_towers_batch length mismatch: marginals={chunk}, \
             slopes={}, zs={}, ys={}, weights={}, out={}",
            slopes.len(),
            zs.len(),
            ys.len(),
            weights.len(),
            out.len()
        ));
    }

    // Pass A: branch-free signed-margin jets + flat margin scratch.
    let mut signed: Vec<Tower4<2>> = Vec::with_capacity(chunk);
    let mut margins: Vec<f64> = Vec::with_capacity(chunk);
    for i in 0..chunk {
        let jet =
            rigid_standard_normal_signed_jet(marginals[i], slopes[i], zs[i], ys[i], probit_scale);
        margins.push(jet.v);
        signed.push(jet);
    }

    // Pass B: the transcendental, isolated over a flat margin slice. Each entry
    // is the exact `[d0..d4]` `compose_unary` consumes; the production path's
    // fail-fast on a non-finite (non-`+∞`) margin is preserved here.
    let mut stacks: Vec<[f64; 5]> = Vec::with_capacity(chunk);
    for i in 0..chunk {
        let m = margins[i];
        if !(m.is_finite() || m == f64::INFINITY) {
            return Err(format!(
                "non-finite signed margin in rigid probit tower batch: {m}"
            ));
        }
        stacks.push(signed_probit_neglog_unary_stack(m, weights[i]));
    }

    // Pass C: branch-free dense compose + consumer fill.
    for i in 0..chunk {
        let tower = signed[i].compose_unary(stacks[i]);
        out[i] = fill(&tower)?;
    }
    Ok(())
}

#[inline]
pub(super) fn rigid_standard_normal_row_kernel(
    marginal: BernoulliMarginalLinkMap,
    g: f64,
    z: f64,
    y: f64,
    w: f64,
    probit_scale: f64,
) -> Result<(f64, [f64; 2], [[f64; 2]; 2]), String> {
    // #932 cutover: value/gradient/Hessian derive from the SAME single generic
    // row-NLL expression (`rigid_standard_normal_row_nll_generic`) every other
    // channel consumer uses, routed through the `RowNllProgramGeneric` seam at the
    // packed `Order2<2>` scalar — there is no longer a hand-assembled `Tower2<2>`
    // here. Seeds `[marginal η, g]` exactly as the deleted inline form did, so it
    // is bit-identical (the `rigid_bernoulli_*_agrees_with_jet_tower_program_all_channels`
    // oracle pins v/g/H ≤ 1e-12), while sharing one definition with the third/
    // fourth/full-tower channels.
    let program = RigidStandardNormalRow {
        marginal,
        g,
        z,
        y,
        w,
        probit_scale,
    };
    crate::families::jet_tower::generic_row_kernel(&program, 0)
}

/// Mixed `(primary, z)` second derivative of the rigid standard-normal row
/// LOG-LIKELIHOOD score: the per-row 2-vector
/// `[∂²(log L)/∂q∂z, ∂²(log L)/∂g∂z]` in the primary coordinates `(q = marginal η,
/// g = slope)`, evaluated at this row's converged `(q, g)` and calibrated
/// latent score `z = ζ`.
///
/// SIGN CONVENTION (#1131). This returns the mixed partial of the
/// LOG-LIKELIHOOD score `score_β,i = ∂(log L_i)/∂β`, NOT of the negative
/// log-likelihood `ℓ = −log L`. Concretely the row jet evaluates the NLL
/// `ℓ = −w·log Φ(sign·η)` and we NEGATE its mixed `(primary, z)` Hessian entries,
/// so the returned 2-vector is `+∂²(log L_i)/∂(q,g)∂ζ_i = −∂²ℓ_i/∂(q,g)∂ζ_i`.
/// This is the convention under which the Murphy–Topel chain
/// `G = Σ_i s_i·(∂ζ_i/∂θ₁)` with `s_i = ∂score_β,i/∂ζ_i` and `Vb = H_β⁻¹`
/// (the NLL-Hessian inverse) gives the SIGNED sensitivity with the right sign:
/// the implicit-function theorem on the stationarity `∂(log L)/∂β = 0` yields
/// `∂β̂/∂θ₁ = −(∂²log L/∂β²)⁻¹·∂²(log L)/∂β∂θ₁ = +H_β⁻¹·G = +Vb·G`. (Had we
/// returned the NLL mixed partial instead, `Vb·G` would equal `−∂β̂/∂θ₁` — a
/// benign sign flip for the PSD quadratic SE `(Vb·G)V₁(Vb·G)ᵀ`, but wrong for
/// any signed consumer of the sensitivity.)
///
/// This is the #1028 Murphy–Topel generated-regressor channel: `score_β,i =
/// ∂(log L_i)/∂β = J_iᵀ·(∂(log L_i)/∂(q,g))`, so the per-row slope-score
/// sensitivity to the calibrated score is
/// `s_i = ∂score_β,i/∂ζ_i = J_iᵀ·(∂²(log L_i)/∂(q,g)∂ζ_i)`, and the primary
/// 2-vector returned here is exactly `∂²(log L_i)/∂(q,g)∂ζ_i`. The block-level
/// contraction `J_iᵀ` (marginal+logslope design rows) is applied by the caller.
///
/// It is computed by seeding `z` as a THIRD jet variable (index 2) in the SAME
/// order-≤2 jet algebra the value/gradient/Hessian path uses, carried by the
/// packed `Order2<3>`/`Tower2<3>` scalar rather than a dense `Tower4<3>`
/// (#932 row-jet machinery, packed-scalar perf cutover): the
/// rigid standard-normal observed index is `η = q·c(g) + g·(s·z)` with
/// `c(g) = √(1 + (s·g)²)`, `s = probit_scale`, and `ℓ = −w·log Φ(sign·η)`. The
/// converged-frame mixed partials of the NLL are the off-diagonal Hessian
/// entries `tower.h[q][z]` and `tower.h[g][z]`, read off in one composition and
/// NEGATED to the log-likelihood-score convention — the only extra cost over the
/// production `Tower4<2>` evaluation is the third jet axis.
#[inline]
pub(super) fn rigid_standard_normal_mixed_z_sensitivity(
    marginal: BernoulliMarginalLinkMap,
    g: f64,
    z: f64,
    y: f64,
    w: f64,
    probit_scale: f64,
) -> Result<[f64; 2], String> {
    // Three jet axes: q = marginal η (0), g = slope (1), z = latent score (2).
    //
    // #932 perf: this consumer reads ONLY the two mixed Hessian channels
    // `h[0][2]`/`h[1][2]`, so it needs only the value/gradient/Hessian stack —
    // the packed `Order2<3>` scalar (operating on its inner `Tower2<3>`), NOT a
    // dense `Tower4<3>` that would materialise the unused `K³`/`K⁴` `t3`/`t4`
    // tensors. The order-≤2 channels are bit-identical to the dense tower
    // (`Tower2::mul`/`compose_unary` match `Tower4` term-for-term), so the read
    // entries are unchanged; the `q3`/`q4` marginal-link channels are dropped
    // because no order-≤2 channel of the composed jet reads them.
    use crate::families::jet_tower::Tower2;
    let mut q = Tower2::<3>::constant(marginal.q);
    q.g[0] = marginal.q1;
    q.h[0][0] = marginal.q2;
    let slope = Tower2::<3>::variable(g, 1);
    let z_var = Tower2::<3>::variable(z, 2);
    let observed_logslope = slope * probit_scale;
    let c = (observed_logslope * observed_logslope + 1.0).sqrt();
    // η = q·c + g·(s·z): z enters linearly through the slope×z product, so the
    // mixed (q,z)/(g,z) curvature is carried entirely by the unary NLL chain and
    // the η-bilinear, exactly as in the Tower4<2> production path.
    let eta = q * c + slope * (z_var * probit_scale);
    let signed = eta * (2.0 * y - 1.0);
    // ONE transcendental per row (see `rigid_standard_normal_tower`).
    if !(signed.v.is_finite() || signed.v == f64::INFINITY) {
        return Err(format!(
            "rigid probit mixed-z sensitivity: non-finite signed margin {} at q={}, g={g}, z={z}, y={y}",
            signed.v, marginal.q
        ));
    }
    let stack = signed_probit_neglog_unary_stack(signed.v, w);
    if !stack[0].is_finite() {
        return Err(format!(
            "rigid probit mixed-z sensitivity: non-finite log Φ at q={}, g={g}, z={z}, y={y}",
            marginal.q
        ));
    }
    // Order-≤2 composition consumes only the leading `[f, f', f'']` of the
    // certified `[f64; 5]` derivative stack.
    let tower = signed.compose_unary([stack[0], stack[1], stack[2]]);
    // #1131: `tower` is the NLL `ℓ = −w·log Φ`, so `tower.h[·][z]` is the mixed
    // partial of the NLL. Negate to the LOG-LIKELIHOOD-score convention
    // `s = ∂²(log L)/∂(primary)∂z = −∂²ℓ/∂(primary)∂z`, under which the
    // downstream Murphy–Topel chain `Vb·G = +∂β̂/∂θ₁` carries the correct sign
    // (see the function doc). The SE is the PSD quadratic `(Vb·G)V₁(Vb·G)ᵀ` and
    // is invariant to this sign, so the reported standard errors are unchanged.
    let s_q = -tower.h[0][2];
    let s_g = -tower.h[1][2];
    if !(s_q.is_finite() && s_g.is_finite()) {
        return Err(format!(
            "rigid probit mixed-z sensitivity: non-finite ∂²(log L)/∂(q,g)∂z = [{s_q}, {s_g}] at q={}, g={g}, z={z}",
            marginal.q
        ));
    }
    Ok([s_q, s_g])
}

/// Assemble the #1028 Murphy–Topel slope-score sensitivity matrix
/// `score_zeta_sensitivity` (`n × p_β`, row `i` = `s_i = ∂score_β,i/∂ζ_i`) for
/// the rigid standard-normal BMS kernel — the kernel the conditional
/// location-scale gate ALWAYS selects (`LatentMeasureKind::StandardNormal`).
///
/// where `s_i = ∂score_β,i/∂ζ_i` is the LOG-LIKELIHOOD-score sensitivity (see
/// the sign convention in [`rigid_standard_normal_mixed_z_sensitivity`], #1131).
/// For each row `i` the primary 2-vector `∂²(log L_i)/∂(q,g)∂ζ_i` is read off the
/// z-augmented row jet ([`rigid_standard_normal_mixed_z_sensitivity`]) at the
/// converged marginal index `q_i` (`marginal_eta[i]`) and slope `g_i`
/// (`slope_eta[i]`) and calibrated score `ζ_i` (`z[i]`), then contracted through
/// the block Jacobian `J_iᵀ` (the same marginal+logslope design-row scatter the
/// row kernel exposes via `jacobian_transpose_action`):
///
/// ```text
///   s_i[marginal_range]  = (∂²(log L_i)/∂q∂ζ_i) · marginal_design.row(i)
///   s_i[logslope_range]  = (∂²(log L_i)/∂g∂ζ_i) · logslope_design.row(i)
/// ```
///
/// `logslope_design` MUST be the reduced-basis design `G·T` actually fitted
/// (so `p_β = p_marginal + r` matches the reduced-frame `covariance_conditional`
/// the correction inflates). The aux deviation blocks (score_warp / link_dev),
/// when present, occupy the trailing columns of `p_beta` and are left zero here:
/// the rigid standard-normal kernel carries no deviation z-dependence, and the
/// conditional gate's canonical (non-flex) kernel has no such blocks — the
/// caller wires the correction only when `p_beta == p_marginal + p_logslope`.
pub(super) fn rigid_standard_normal_score_zeta_sensitivity(
    base_link: &InverseLink,
    marginal_eta: &Array1<f64>,
    slope_eta: &Array1<f64>,
    z: &Array1<f64>,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    probit_scale: f64,
    marginal_design: ArrayView2<'_, f64>,
    logslope_design: ArrayView2<'_, f64>,
    p_beta: usize,
) -> Result<Array2<f64>, String> {
    let n = marginal_eta.len();
    let p_m = marginal_design.ncols();
    let r = logslope_design.ncols();
    if slope_eta.len() != n
        || z.len() != n
        || y.len() != n
        || weights.len() != n
        || marginal_design.nrows() != n
        || logslope_design.nrows() != n
    {
        return Err(format!(
            "score_zeta_sensitivity row mismatch: marginal_eta={n}, slope_eta={}, z={}, y={}, \
             weights={}, marginal_design rows={}, logslope_design rows={}",
            slope_eta.len(),
            z.len(),
            y.len(),
            weights.len(),
            marginal_design.nrows(),
            logslope_design.nrows()
        ));
    }
    if p_m + r > p_beta {
        return Err(format!(
            "score_zeta_sensitivity width overflow: marginal({p_m}) + logslope({r}) > p_beta({p_beta})"
        ));
    }
    let mut s = Array2::<f64>::zeros((n, p_beta));
    for i in 0..n {
        let marginal = bernoulli_marginal_link_map(base_link, marginal_eta[i])?;
        let [s_q, s_g] = rigid_standard_normal_mixed_z_sensitivity(
            marginal,
            slope_eta[i],
            z[i],
            y[i],
            weights[i],
            probit_scale,
        )?;
        // J_iᵀ scatter into the reduced-frame coordinates: marginal block first,
        // then the reduced logslope block.
        if s_q != 0.0 {
            let m_row = marginal_design.row(i);
            for (j, &mij) in m_row.iter().enumerate() {
                s[[i, j]] = s_q * mij;
            }
        }
        if s_g != 0.0 {
            let g_row = logslope_design.row(i);
            for (j, &gij) in g_row.iter().enumerate() {
                s[[i, p_m + j]] = s_g * gij;
            }
        }
    }
    Ok(s)
}

#[inline]
pub(super) fn rigid_standard_normal_third_full(
    marginal: BernoulliMarginalLinkMap,
    g: f64,
    z: f64,
    y: f64,
    w: f64,
    probit_scale: f64,
) -> Result<[[[f64; 2]; 2]; 2], String> {
    Ok(rigid_standard_normal_tower(marginal, g, z, y, w, probit_scale)?.t3)
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

#[inline]
pub(super) fn rigid_standard_normal_fourth_full(
    marginal: BernoulliMarginalLinkMap,
    g: f64,
    z: f64,
    y: f64,
    w: f64,
    probit_scale: f64,
) -> Result<[[[[f64; 2]; 2]; 2]; 2], String> {
    // #932 single-sourcing: the full uncontracted fourth-order primary tensor is
    // the `.t4` channel of the SAME `Tower4<2>` row jet the value/gradient/Hessian
    // and the third-order tensor (`rigid_standard_normal_third_full` → `.t3`) are
    // read from. The marginal latent-coordinate chain `q(η)` is already seeded
    // into axis 0 of the tower (`q.g[0]=q1, q.h[0][0]=q2, q.t3[0][0][0]=q3,
    // q.t4[0][0][0][0]=q4` in `rigid_standard_normal_signed_jet`), so `.t4` is
    // delivered directly in the production `(η, g)` primary space — no separate
    // Faà-di-Bruno q-chain reassembly. This replaces the former hand-written
    // fourth-derivative chain rule with the mechanically-derived tower output,
    // exactly mirroring how `.t3` is consumed;
    // it is cross-checked term-for-term against the independent
    // `HandRigidProbitKernel` witness in
    // `rigid_standard_normal_tower_path_matches_hand_chain_witness`.
    Ok(rigid_standard_normal_tower(marginal, g, z, y, w, probit_scale)?.t4)
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

pub(super) fn ensure_finite_third_full_cache_row(
    t: &[[[f64; 2]; 2]; 2],
    context: &str,
) -> Result<(), String> {
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
    // Single source of truth for the signed-probit value+derivative stack:
    // one Mills-ratio transcendental feeds both logΦ and k1..k4 (the prior
    // body evaluated `signed_probit_logcdf_and_mills_ratio` twice). The
    // ±∞/NaN/zero-weight boundary limits are handled identically inside.
    signed_probit_neglog_unary_stack(x, weight)
}

/// Derivatives of `log(x)` through 4th order.
///
/// # Contract
///
/// `x` must be strictly positive. `log` and its derivatives are undefined at
/// and below the boundary, so this function does NOT clamp: a previous version
/// silently replaced `x` by `x.max(1e-300)`, which fabricated enormous finite
/// derivatives (`1/1e-300` etc.) that are the derivatives of neither `log(x)`
/// nor `log(max(x, floor))`. Such a non-positive argument signals an upstream
/// domain failure (e.g. a monotonicity violation) that must surface, not be
/// masked. Every caller guarantees `x > 0` before invoking this:
/// the survival marginal-slope kernels evaluate `log` of the transformed time
/// derivative `q'(t)·√(1+b²)` only after passing `survival_derivative_guard`
/// (`q'(t) >= derivative_guard > 0`, `√(1+b²) > 0`). A non-positive `x`
/// therefore never reaches here on any supported path; were one to, the
/// function returns the honest IEEE result (`-inf`/`NaN`) — identical in debug
/// and release — rather than a finite fabrication.
pub(crate) fn unary_derivatives_log(x: f64) -> [f64; 5] {
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x3 * x;
    [x.ln(), 1.0 / x, -1.0 / x2, 2.0 / x3, -6.0 / x4]
}

/// Derivatives of log φ(x) = -½x² - ½ln(2π) through 4th order.
pub(crate) fn unary_derivatives_log_normal_pdf(x: f64) -> [f64; 5] {
    let c = 0.5 * (2.0 * std::f64::consts::PI).ln();
    [-0.5 * x * x - c, -x, -1.0, 0.0, 0.0]
}

#[cfg(test)]
mod jet_tower_oracle_tests {
    //! #932 deployment step 2 for the BMS rigid Bernoulli `RowKernel<2>`.
    //!
    //! The production rigid standard-normal row kernel
    //! ([`rigid_standard_normal_row_kernel`] / `_third_full` / `_fourth_full`)
    //! reads value/grad/Hessian/third/fourth straight off ONE
    //! [`rigid_standard_normal_tower`] `Tower4<2>` — the strongest #932 form,
    //! where the production kernel literally *is* the single-expression jet.
    //! What was missing (unlike the two survival `RowKernel` families, which
    //! already carry `verify_kernel_channels` oracles) is an INDEPENDENT
    //! cross-check that this production tower is correct. This module adds it:
    //!
    //! * an independent [`RowNllProgram<2>`] that writes the row NLL
    //!   `ℓ = −w·logΦ((2y−1)·η)`, `η = q·√(1+(s·g)²) + s·g·z` ONCE over generic
    //!   `Tower4` arithmetic (a different composition order than the fused
    //!   production `signed` jet → exercises the Leibniz/Faà-di-Bruno layer
    //!   where the #736 cross-block sign-flip bug genus lives), and
    //! * a special-function-independent central-FD witness of the value channel
    //!   that re-derives `logΦ` from `libm::erfc`, pinning the probit derivative
    //!   stack itself (so the oracle does not merely re-use the production
    //!   transcendental).

    use super::*;
    use crate::families::jet_tower::{
        KernelChannels, RowNllProgram, evaluate_program, verify_kernel_channels,
    };

    /// Independent single-expression row NLL for the rigid standard-normal
    /// Bernoulli kernel, primaries `(q_eta = marginal η, g = slope)`.
    struct BernoulliRigidStandardNormalNllProgram {
        /// `(marginal η, slope g)` per row.
        primaries: Vec<[f64; 2]>,
        /// Per-row `(z latent score, y in {0,1}, w weight)`.
        z: Vec<f64>,
        y: Vec<f64>,
        w: Vec<f64>,
        probit_scale: f64,
    }

    impl RowNllProgram<2> for BernoulliRigidStandardNormalNllProgram {
        fn n_rows(&self) -> usize {
            self.primaries.len()
        }

        fn primaries(&self, row: usize) -> Result<[f64; 2], String> {
            self.primaries
                .get(row)
                .copied()
                .ok_or_else(|| format!("bernoulli rigid nll program: row {row} out of range"))
        }

        fn row_nll(&self, row: usize, p: &[Tower4<2>; 2]) -> Result<Tower4<2>, String> {
            let z = self.z[row];
            let y = self.y[row];
            let w = self.w[row];
            let s = self.probit_scale;
            // q(η) via the family's own marginal link-map derivative stack,
            // composed through generic Leibniz on the η primary (independent of
            // the production signed-jet, which seeds the q tensor slots directly).
            let eta_marginal = p[0];
            let link = bernoulli_marginal_link_map(
                &InverseLink::Standard(crate::types::StandardLink::Probit),
                eta_marginal.v,
            )?;
            let q = eta_marginal.compose_unary([link.q, link.q1, link.q2, link.q3, link.q4]);
            let g = p[1];
            // observed slope b = s·g, scale c = √(1 + b²).
            let observed_slope = g * s;
            let c = (observed_slope * observed_slope + 1.0).compose_unary(unary_derivatives_sqrt(
                observed_slope.v * observed_slope.v + 1.0,
            ));
            // η = q·c + b·z, signed margin m = (2y−1)·η.
            let eta = q * c + observed_slope * z;
            let signed = eta * (2.0 * y - 1.0);
            // NLL = −w·logΦ(m) via the documented probit neglog stack.
            Ok(signed.compose_unary(unary_derivatives_neglog_phi(signed.v, w)))
        }
    }

    /// Special-function-independent scalar row NLL `ℓ(q_eta, g)` using
    /// `libm::erfc`, for the central-FD value-channel witness.
    fn scalar_nll(eta_marginal: f64, g: f64, z: f64, y: f64, w: f64, s: f64) -> f64 {
        let link = bernoulli_marginal_link_map(
            &InverseLink::Standard(crate::types::StandardLink::Probit),
            eta_marginal,
        )
        .unwrap();
        let observed_slope = g * s;
        let c = (observed_slope * observed_slope + 1.0).sqrt();
        let eta = link.q * c + observed_slope * z;
        let signed = (2.0 * y - 1.0) * eta;
        let cdf = 0.5 * libm::erfc(-signed / std::f64::consts::SQRT_2);
        -w * cdf.max(1e-300).ln()
    }

    #[test]
    fn rigid_bernoulli_row_kernel_agrees_with_jet_tower_program_all_channels() {
        // Mixed responses, weights, latent scores, and slope regimes; the last
        // rows push the marginal index toward the normal tails while staying
        // finite. Probit marginal link, standard-normal latent measure.
        let eta = [0.3_f64, -0.7, 0.05, 0.9, -1.2, 2.1, -2.4];
        let g = [0.2_f64, -0.5, 0.35, -0.15, 0.6, 0.45, -0.55];
        let z = [0.4_f64, -1.1, 0.0, 0.7, -0.3, 1.6, -1.4];
        let y = [1.0_f64, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let w = [1.0_f64, 0.8, 1.3, 0.9, 1.1, 0.7, 1.4];
        let n = eta.len();

        // Deterministic direction vectors (no RNG dependency).
        let dirs: [[f64; 2]; 3] = [[0.7, -1.3], [-0.4, 0.6], [1.2, 0.2]];

        for &probit_scale in &[1.0_f64, 0.8] {
            let program = BernoulliRigidStandardNormalNllProgram {
                primaries: (0..n).map(|r| [eta[r], g[r]]).collect(),
                z: z.to_vec(),
                y: y.to_vec(),
                w: w.to_vec(),
                probit_scale,
            };

            for row in 0..n {
                let tower = evaluate_program(&program, row).expect("tower evaluation");

                // Production scalar kernel channels (the hand path under audit).
                let marginal = bernoulli_marginal_link_map(
                    &InverseLink::Standard(crate::types::StandardLink::Probit),
                    eta[row],
                )
                .expect("link map");
                let (value, gradient, hessian) = rigid_standard_normal_row_kernel(
                    marginal,
                    g[row],
                    z[row],
                    y[row],
                    w[row],
                    probit_scale,
                )
                .expect("production row kernel");

                let third_full = rigid_standard_normal_third_full(
                    marginal,
                    g[row],
                    z[row],
                    y[row],
                    w[row],
                    probit_scale,
                )
                .expect("production third");
                let third: Vec<([f64; 2], [[f64; 2]; 2])> = dirs
                    .iter()
                    .map(|d| (*d, contract_third_full(&third_full, d[0], d[1])))
                    .collect();

                let fourth_full = rigid_standard_normal_fourth_full(
                    marginal,
                    g[row],
                    z[row],
                    y[row],
                    w[row],
                    probit_scale,
                )
                .expect("production fourth");
                let fourth: Vec<([f64; 2], [f64; 2], [[f64; 2]; 2])> = dirs
                    .iter()
                    .enumerate()
                    .map(|(i, u)| {
                        let v = dirs[(i + 1) % dirs.len()];
                        (
                            *u,
                            v,
                            contract_fourth_full(&fourth_full, u[0], u[1], v[0], v[1]),
                        )
                    })
                    .collect();

                let claims = KernelChannels {
                    value,
                    gradient,
                    hessian,
                    third,
                    fourth,
                };

                verify_kernel_channels(&tower, &claims, 1e-9).unwrap_or_else(|e| {
                    panic!(
                        "probit_scale {probit_scale} row {row}: production rigid Bernoulli \
                         RowKernel disagrees with #932 jet-tower truth: {e}"
                    )
                });

                // Special-function-independent FD witness of the value channel:
                // re-derives logΦ from `libm::erfc`, pinning the probit derivative
                // stack rather than re-using the production one.
                let h = 1e-3;
                let f = |de: f64, dg: f64| {
                    scalar_nll(
                        eta[row] + de,
                        g[row] + dg,
                        z[row],
                        y[row],
                        w[row],
                        probit_scale,
                    )
                };
                let f0 = f(0.0, 0.0);
                assert!(
                    (f0 - tower.v).abs() <= 1e-9 * f0.abs().max(1.0),
                    "row {row}: independent scalar NLL {f0:+.12e} != tower value {:+.12e}",
                    tower.v
                );
                // 5-point first-derivative stencils.
                let g_eta = (f(-2.0 * h, 0.0) - 8.0 * f(-h, 0.0) + 8.0 * f(h, 0.0)
                    - f(2.0 * h, 0.0))
                    / (12.0 * h);
                let g_g = (f(0.0, -2.0 * h) - 8.0 * f(0.0, -h) + 8.0 * f(0.0, h) - f(0.0, 2.0 * h))
                    / (12.0 * h);
                for (label, fd, ad) in [("∂η", g_eta, tower.g[0]), ("∂g", g_g, tower.g[1])] {
                    assert!(
                        (fd - ad).abs() <= 1e-5 * ad.abs().max(1.0),
                        "row {row} {label}: FD witness {fd:+.6e} != tower grad {ad:+.6e}"
                    );
                }
            }
        }
    }

    /// #932 production wiring: the rigid Bernoulli row, routed through the
    /// generic [`RowNllProgramGeneric<2>`] program seam and its cheap
    /// order-2 / contracted scalar evaluators (`generic_row_kernel`,
    /// `generic_third_contracted`, `generic_fourth_contracted`,
    /// `generic_full_tower`), must agree BIT-FOR-BIT with the dense
    /// `Tower4`-only [`RowNllProgram`] path (`evaluate_program`). Both write the
    /// same single-expression NLL — the contracted scalars fold the direction
    /// into the differentiation, so this pins that the packed channels equal the
    /// corresponding contractions of the dense tower truth, exercising every
    /// `generic_*` evaluator end-to-end through a real production consumer.
    #[test]
    fn rigid_bernoulli_generic_program_matches_tower4_program_all_channels() {
        use crate::families::jet_tower::{
            generic_full_tower, generic_row_kernel, generic_third_contracted,
            generic_fourth_contracted,
        };

        let eta = [0.3_f64, -0.7, 0.05, 0.9, -1.2, 2.1, -2.4];
        let g = [0.2_f64, -0.5, 0.35, -0.15, 0.6, 0.45, -0.55];
        let z = [0.4_f64, -1.1, 0.0, 0.7, -0.3, 1.6, -1.4];
        let y = [1.0_f64, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let w = [1.0_f64, 0.8, 1.3, 0.9, 1.1, 0.7, 1.4];
        let n = eta.len();
        let dirs: [[f64; 2]; 3] = [[0.7, -1.3], [-0.4, 0.6], [1.2, 0.2]];

        let close = |a: f64, b: f64, label: &str| {
            let band = 1e-12 + 1e-12 * a.abs().max(b.abs());
            assert!(
                (a - b).abs() <= band,
                "{label}: generic {a:+.15e} vs Tower4-program {b:+.15e} (band {band:.3e})"
            );
        };

        for &probit_scale in &[1.0_f64, 0.8] {
            // The dense Tower4-only program over all rows (independent path).
            let tower_program = BernoulliRigidStandardNormalNllProgram {
                primaries: (0..n).map(|r| [eta[r], g[r]]).collect(),
                z: z.to_vec(),
                y: y.to_vec(),
                w: w.to_vec(),
                probit_scale,
            };

            for row in 0..n {
                let truth = evaluate_program(&tower_program, row).expect("Tower4 program tower");

                let marginal = bernoulli_marginal_link_map(
                    &InverseLink::Standard(crate::types::StandardLink::Probit),
                    eta[row],
                )
                .expect("link map");
                let program = RigidStandardNormalRow {
                    marginal,
                    g: g[row],
                    z: z[row],
                    y: y[row],
                    w: w[row],
                    probit_scale,
                };

                // generic_full_tower must reproduce the dense tower in EVERY
                // channel (v, g, H, t3, t4).
                let full = generic_full_tower(&program, 0).expect("generic full tower");
                close(full.v, truth.v, "full value");
                for a in 0..2 {
                    close(full.g[a], truth.g[a], "full grad");
                    for b in 0..2 {
                        close(full.h[a][b], truth.h[a][b], "full hess");
                        for c in 0..2 {
                            close(full.t3[a][b][c], truth.t3[a][b][c], "full t3");
                            for d in 0..2 {
                                close(full.t4[a][b][c][d], truth.t4[a][b][c][d], "full t4");
                            }
                        }
                    }
                }

                // generic_row_kernel (Order2) must equal the tower's (v, g, H).
                let (val, grad, hess) =
                    generic_row_kernel(&program, 0).expect("generic row kernel");
                close(val, truth.v, "order2 value");
                for a in 0..2 {
                    close(grad[a], truth.g[a], "order2 grad");
                    for b in 0..2 {
                        close(hess[a][b], truth.h[a][b], "order2 hess");
                    }
                }

                // generic_third_contracted (OneSeed) must equal the dense
                // tower's third contraction for each direction.
                for dir in &dirs {
                    let third = generic_third_contracted(&program, 0, dir)
                        .expect("generic third contracted");
                    let truth3 = truth.third_contracted(dir);
                    for a in 0..2 {
                        for b in 0..2 {
                            close(third[a][b], truth3[a][b], "third contracted");
                        }
                    }
                }

                // generic_fourth_contracted (TwoSeed) must equal the dense
                // tower's fourth contraction for each direction pair.
                for (i, u) in dirs.iter().enumerate() {
                    let v = dirs[(i + 1) % dirs.len()];
                    let fourth = generic_fourth_contracted(&program, 0, u, &v)
                        .expect("generic fourth contracted");
                    let truth4 = truth.fourth_contracted(u, &v);
                    for a in 0..2 {
                        for b in 0..2 {
                            close(fourth[a][b], truth4[a][b], "fourth contracted");
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod flex_primary_hessian_oracle_tests {
    //! #932 correctness gate for the BMS-FLEX per-row primary Hessian assembled
    //! by hand product-rule in
    //! [`super::super::row_primary_hessian::BernoulliMarginalSlopeFamily::compute_row_analytic_flex_from_parts_into`]
    //! (`f_aa += w·φ·(η_aa − η·η_a·η_a)`, the `f_au`/`f_uv`/`a_uv` chain, and the
    //! final `d2_m·η_u·η_v + d1_m·s_y·η_uv` contraction).
    //!
    //! A prior audit found this hand Hessian had NO INDEPENDENT oracle: the only
    //! covering test (`families_bms_joint_hessian_hvp_correction_tests.rs`)
    //! asserts batched-vs-nonbatched self-consistency using the SAME hand code on
    //! both sides, so a dropped product-rule term would pass undetected. This
    //! module closes that gap with a finite-difference witness that NEVER runs the
    //! Hessian-assembly branch: it central-differences the flex GRADIENT — which
    //! is produced by an entirely separate code path (the `need_hessian = false`
    //! value/`eta_u`-scaling lines, none of which read the `f_aa`/`f_au`/`f_uv`
    //! product-rule accumulators) — and pins the analytic Hessian against it.
    //!
    //! The gradient itself is FD-validated transitively: it is the analytic
    //! gradient of the same per-row NLL, evaluated at the converged intercept,
    //! and the FD perturbation re-solves the intercept root per perturbed point
    //! (rebuilding the row context), so the difference quotient is the true
    //! mixed/second partial of the row negative log-likelihood — the independent
    //! truth the hand Hessian must reproduce.

    use super::*;
    use crate::matrix::DenseDesignMatrix;
    use ndarray::Array1;
    use ndarray::Array2;
    use std::sync::Arc;
    use std::sync::Mutex;

    /// Port of the integration-test flex fixture
    /// (`make_flex_hvp_cache_test_family`), kept in-crate so the oracle can run
    /// without the test crate (the family struct is `pub(super)`). Builds a small
    /// flex BMS family with both a score-warp and a link-deviation block so the
    /// flex Hessian assembly exercises every primary block (q, logslope, h, w).
    fn make_flex_oracle_family(n: usize) -> (BernoulliMarginalSlopeFamily, Vec<ParameterBlockState>) {
        let score_seed = Array1::linspace(-2.0, 2.0, n.max(6));
        let link_seed = Array1::linspace(-1.8, 1.8, n.max(6));
        let cfg = DeviationBlockConfig {
            num_internal_knots: 3,
            ..DeviationBlockConfig::default()
        };
        let score_prepared = build_score_warp_deviation_block_from_seed(&score_seed, &cfg)
            .expect("build score warp block");
        let link_prepared = build_link_deviation_block_from_knots_design_seed_and_weights(
            &link_seed, &link_seed, &cfg,
        )
        .expect("build link deviation block");

        let y: Array1<f64> =
            Array1::from_iter((0..n).map(|i| if (i * 17 + 3) % 7 >= 4 { 1.0 } else { 0.0 }));
        let weights: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 0.75 + ((i * 11 + 5) % 5) as f64 * 0.05));
        let z: Array1<f64> =
            Array1::from_iter((0..n).map(|i| -1.7 + 3.4 * (i as f64 + 0.5) / n as f64));
        let marginal_x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                1.0
            } else {
                -0.4 + 0.8 * ((i * 19 + 7) % n) as f64 / n as f64
            }
        });
        let logslope_x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                1.0
            } else {
                0.3 - 0.6 * ((i * 23 + 11) % n) as f64 / n as f64
            }
        });

        let family = BernoulliMarginalSlopeFamily {
            y: Arc::new(y),
            weights: Arc::new(weights),
            z: Arc::new(z.clone()),
            latent_measure: LatentMeasureKind::StandardNormal,
            gaussian_frailty_sd: Some(0.15),
            base_link: InverseLink::Standard(crate::types::StandardLink::Probit),
            marginal_design: DesignMatrix::Dense(DenseDesignMatrix::from(marginal_x.clone())),
            logslope_design: DesignMatrix::Dense(DenseDesignMatrix::from(logslope_x.clone())),
            score_warp: Some(score_prepared.runtime.clone()),
            link_dev: Some(link_prepared.runtime.clone()),
            policy: crate::resource::ResourcePolicy::default_library(),
            cell_moment_lru: Arc::new(exact_kernel::CellMomentLruCache::new(1024)),
            cell_moment_cache_stats: Arc::new(exact_kernel::CellMomentCacheStats::default()),
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };

        let beta_m = Array1::from_vec(vec![0.12, -0.04]);
        let beta_g = Array1::from_vec(vec![0.35, 0.03]);
        let beta_h = Array1::from_iter(
            (0..score_prepared.runtime.basis_dim()).map(|idx| 0.0015 * (idx as f64 + 1.0)),
        );
        let beta_w = Array1::from_iter(
            (0..link_prepared.runtime.basis_dim()).map(|idx| -0.001 * (idx as f64 + 1.0)),
        );
        let states = vec![
            ParameterBlockState {
                eta: marginal_x.dot(&beta_m),
                beta: beta_m,
            },
            ParameterBlockState {
                eta: logslope_x.dot(&beta_g),
                beta: beta_g,
            },
            ParameterBlockState {
                beta: beta_h,
                eta: Array1::zeros(z.len()),
            },
            ParameterBlockState {
                beta: beta_w,
                eta: Array1::zeros(z.len()),
            },
        ];
        (family, states)
    }

    /// The flex primary gradient at a perturbed primary point. Perturbs primary
    /// coordinate `u` by `delta` (mutating the relevant block state — the
    /// marginal/logslope row η or a deviation β plus its design contribution
    /// where applicable), rebuilds the row context FRESH (re-solving the
    /// calibration intercept root at the perturbed point), and returns the
    /// analytic gradient. The Hessian-assembly branch is never run, so this is a
    /// genuinely independent witness for that branch.
    fn flex_gradient_at_perturbed(
        family: &BernoulliMarginalSlopeFamily,
        states: &[ParameterBlockState],
        primary: &super::super::hessian_paths::PrimarySlices,
        row: usize,
        u: usize,
        delta: f64,
    ) -> Array1<f64> {
        let mut states = states.to_vec();
        // Map the primary coordinate `u` onto the parameter that controls it.
        // q / logslope live in the per-row η of blocks 0 / 1; the deviation
        // bases live in the β of blocks 2 (score-warp) / 3 (link-wiggle), which
        // the row context reads via `score_beta` / `link_beta` (their η rows are
        // unused on the flex per-row path, so only β need move).
        if u == primary.q {
            states[0].eta[row] += delta;
        } else if u == primary.logslope {
            states[1].eta[row] += delta;
        } else if let Some(h_range) = primary.h.as_ref()
            && h_range.contains(&u)
        {
            states[2].beta[u - h_range.start] += delta;
        } else if let Some(w_range) = primary.w.as_ref()
            && w_range.contains(&u)
        {
            states[3].beta[u - w_range.start] += delta;
        } else {
            panic!("primary coordinate {u} out of range for flex oracle");
        }
        let row_ctx = family
            .build_row_exact_context_with_stats_and_cell_cache(row, &states, None, false)
            .expect("perturbed row context");
        let (_neglog, grad, _hess) = family
            .compute_row_primary_gradient_hessian(row, &states, primary, &row_ctx)
            .expect("perturbed gradient");
        grad
    }

    /// The hand-assembled BMS-FLEX per-row primary Hessian must equal the
    /// central finite difference of the flex gradient at every fixture row.
    #[test]
    fn flex_primary_hessian_matches_central_fd_of_gradient() {
        let n = 12usize;
        let (family, states) = make_flex_oracle_family(n);
        let cache = family
            .build_exact_eval_cache(&states)
            .expect("flex exact eval cache");
        let primary = &cache.primary;
        let r = primary.total;
        assert!(r >= 4, "flex fixture must carry q + logslope + deviation blocks");

        // Central-difference step. The flex gradient is smooth in every primary
        // coordinate; 1e-4 balances truncation (O(h^2)) against the cancellation
        // floor of the per-perturbation intercept re-solve (~1e-12).
        let h = 1e-4;
        let mut max_rel = 0.0_f64;

        // A handful of interior rows (avoid the strongest-tail endpoints where
        // the FD floor is loosest). Every primary coordinate is differenced.
        for &row in &[2usize, 5, 8] {
            let row_ctx = BernoulliMarginalSlopeFamily::row_ctx(&cache, row);
            let (_neglog, _grad, analytic_hess) = family
                .compute_row_primary_gradient_hessian(row, &states, primary, row_ctx)
                .expect("analytic flex gradient + hessian");

            for u in 0..r {
                let grad_plus = flex_gradient_at_perturbed(&family, &states, primary, row, u, h);
                let grad_minus = flex_gradient_at_perturbed(&family, &states, primary, row, u, -h);
                for v in 0..r {
                    let fd = (grad_plus[v] - grad_minus[v]) / (2.0 * h);
                    let analytic = analytic_hess[[v, u]];
                    let denom = 1.0 + analytic.abs().max(fd.abs());
                    let rel = (analytic - fd).abs() / denom;
                    max_rel = max_rel.max(rel);
                    assert!(
                        rel <= 1e-6,
                        "flex hand Hessian H[{v}][{u}] = {analytic:.6e} disagrees with central \
                         FD of the gradient {fd:.6e} at row {row} (rel {rel:.3e}); a product-rule \
                         term is dropped or mis-signed"
                    );
                }
            }
        }
        // Surface the achieved tightness for the record.
        assert!(max_rel <= 1e-6, "flex Hessian FD oracle max rel {max_rel:.3e}");
    }
}

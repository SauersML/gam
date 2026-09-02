//! Response moments of a location-scale survival fit whose coefficients carry
//! an inequality cone.
//!
//! # What this replaces, and why it is not a refinement of it
//!
//! [`super::moments::exact_survival_response_moments_row`] integrates the
//! predictor against a JOINT NORMAL. That is exact whenever the posterior is
//! Gaussian, and it is what this model's non-wiggle fits get. A fit that
//! carries the monotone link-wiggle block does not have a Gaussian posterior:
//! `β_w ≥ 0` is enforced during the fit
//! (`family_solver::block_linear_constraints`, one row per wiggle coefficient)
//! and the reported posterior is the Gaussian truncated to that cone. The
//! Gaussian rule was reaching it through the moment-matched normal — the normal
//! carrying `E_π[β]` and `Σ_π`.
//!
//! Matching two moments of a cone-truncated law is not an approximation that
//! gets better with work. For `q > 1` retained rows the pushforward of the
//! truncated joint through a basis row is not normal at all, so the error is a
//! FLOOR, and the normal puts a measurable share of its mass — a few percent,
//! measured — on coefficient vectors the fit excluded. Both properties are
//! wrong in kind, not in size: no node count, tolerance, or correction factor
//! moves them.
//!
//! # The rule
//!
//! The truncated posterior factorizes exactly. Write `u = Aβ − b` for the
//! retained constraint-normal coordinates and `G = ΣAᵀW⁻¹` for the lift the
//! correction already stores. Then
//!
//! ```text
//! β = β_unc + G(u − E_untrunc[u]) + ε,   ε ~ N(0, Σ_res),  Σ_res = Σ − GWGᵀ,
//! ```
//!
//! with `ε` exactly Gaussian and exactly independent of `u` — conditioning a
//! Gaussian on `Aβ` leaves a Gaussian whatever the truncation then does to
//! `Aβ`. Note `AΣ_resAᵀ = W − W = 0`: the tangent moves no constraint normal,
//! so a draw built this way is feasible for every `u` the cone admits.
//!
//! So the law needs one rule over `(u, ε)` jointly, which is what
//! [`gam_solve::constrained_posterior::constrained_posterior_joint_cubature`]
//! produces: `points` points of a single low-discrepancy rule whose first `q`
//! coordinates run the separation-of-variables map into the cone and whose
//! remaining coordinates carry the standardized tangent. The per-row cost is
//! `points` evaluations, with no factor of the cubature's node count in it —
//! the nested alternative costs `nodes × outer × inner`, up to `4096 × 3375 ×
//! 21` on the shipped rules, which is why this was never simply cut over.
//!
//! # What this rule gives up, and where that is measured
//!
//! A lattice rule on the smooth tangent block does not have the spectral
//! accuracy of the Gauss-Hermite tensor rule it replaces. That is why the
//! point count is chosen by refinement against the response moments' own
//! resolution rather than fixed, and why the rule is gated against a reference
//! built from the DENSITY rather than from the cubature
//! (`truncated_response_moments_beat_the_moment_matched_normal_2679`).

use super::*;
use gam_solve::constrained_posterior::{
    ConstrainedPosteriorJointPoint, constrained_posterior_joint_cubature,
};

/// Absolute agreement demanded of the response moments between successive
/// point counts before the rule is accepted.
///
/// Denominated in the resolution of the quantity that produces the number, not
/// in a machine epsilon: `E[S]` and `E[S²]` are PROBABILITIES, and the law they
/// are taken against is itself certified only to
/// `ORTHANT_MOMENT_RELATIVE_TOLERANCE = 1e-3` relative on the constraint-normal
/// covariance the fit stored. A probability resolved to `1e-4` absolute is
/// already an order finer than its own input law, and demanding more would be
/// asserting against the lattice's arithmetic rather than against the posterior.
pub(crate) const TRUNCATED_RESPONSE_MOMENT_ABSOLUTE_TOLERANCE: f64 = 1e-4;

/// Point count of the first pass.
pub(crate) const TRUNCATED_RESPONSE_MOMENT_INITIAL_POINTS: usize = 1 << 11;

/// Point count past which the rule is declared non-convergent and the caller
/// gets an error rather than an uncertified moment. Reporting the last iterate
/// would ship an unmeasured number into `response_standard_error`.
pub(crate) const TRUNCATED_RESPONSE_MOMENT_MAXIMUM_POINTS: usize = 1 << 15;

/// Slack allowed when checking the stored removed variance against the
/// constraint-normal variance reconstructed here.
///
/// Both are outputs of the same `1e-3`-relative orthant cubature, one saved at
/// fit time and one rebuilt from the reported covariance, so they can disagree
/// by that much without anything being wrong. This is a frame/scale guard, not
/// a precision claim: a covariance that arrived on the wrong scale misses by a
/// FACTOR, not by a part in a thousand.
const ORTHANT_REMOVED_VARIANCE_SLACK: f64 = 1e-2;

/// How far the truncated law's own mean may sit from the reported coefficient
/// vector before the frames are declared misaligned, in POSTERIOR STANDARD
/// DEVIATIONS of the coefficient in question.
///
/// Denominated in the posterior's own spread because that is the resolution the
/// reported coefficients carry: the two sides are the same estimand computed
/// two ways, and the cubature that produced the stored mean shift is certified
/// to `1e-3` relative. A wrong gauge or a wrong centre misses by an appreciable
/// fraction of a standard deviation, which this catches; quadrature noise does
/// not.
const COEFFICIENT_RECONSTRUCTION_SD_TOLERANCE: f64 = 1e-2;

/// The inequality-truncated coefficient posterior, expressed in the RAW
/// (reported) coefficient frame the response-moment rule works in.
pub(crate) struct TruncatedCoefficientLaw {
    /// `β_unc` in raw coordinates — the centre of the AMBIENT Gaussian, which
    /// is not the reported coefficient vector: the reported one is `E_π[β]`.
    center: Array1<f64>,
    /// `T G`, the raw displacement per unit of `u − E_untrunc[u]`.
    lift: Array2<f64>,
    /// `E_untrunc[u] = Aβ_unc − b` on the retained rows.
    normal_center: Array1<f64>,
    /// `Σ_res` in raw coordinates.
    residual_covariance: Array2<f64>,
    /// Joint rule over `(u, tangent)`; the tangent block has
    /// [`Self::tangent_dimension`] coordinates.
    points: Vec<ConstrainedPosteriorJointPoint>,
    tangent_dimension: usize,
}

/// Build the truncated law for a fit, or `None` when the fit carries no cone
/// the reported covariance was truncated against.
///
/// `covariance` must be the SAME matrix the truncation identity describes — the
/// conditional posterior covariance `Σ_π`. A caller that selected a different
/// reporting covariance (the smoothing-corrected one) is reporting a law the
/// stored cone geometry does not describe, and gets `None` so it keeps the
/// Gaussian rule it had rather than a mixture built around the wrong second
/// moment.
pub(crate) fn build_truncated_coefficient_law(
    fit: &UnifiedFitResult,
    covariance: &Array2<f64>,
    tangent_dimension: usize,
    points: usize,
) -> Result<Option<TruncatedCoefficientLaw>, String> {
    let Some(geometry) = fit.geometry.as_ref() else {
        return Ok(None);
    };
    let Some(constrained) = geometry.constrained_posterior.as_ref() else {
        return Ok(None);
    };
    let Some(correction) = constrained.correction()? else {
        // Every constraint row is slack at f64 resolution: the truncation is
        // invisible and the ambient Gaussian IS the posterior.
        return Ok(None);
    };
    // The cone geometry describes the conditional posterior. If the caller
    // selected another covariance, the mixture below would carry the cone of
    // one law and the spread of another.
    match fit.beta_covariance() {
        Some(conditional) if conditional == covariance => {}
        _ => return Ok(None),
    }

    let transform = &geometry.coefficient_gauge.t_full;
    let raw_dimension = covariance.nrows();
    if transform.nrows() != raw_dimension {
        return Err(format!(
            "survival location-scale truncated response moments: the coefficient gauge lifts \
             into {} raw coordinates but the reported covariance is {raw_dimension}x{raw_dimension}",
            transform.nrows()
        ));
    }
    let active_dimension = transform.ncols();
    let unconstrained_center = constrained.unconstrained_center()?;
    if unconstrained_center.len() != active_dimension {
        return Err(format!(
            "survival location-scale truncated response moments: the ambient centre has {} \
             coordinates but the gauge reduces to {active_dimension}",
            unconstrained_center.len()
        ));
    }
    if correction.lift.nrows() != active_dimension {
        return Err(format!(
            "survival location-scale truncated response moments: the correction lift has {} \
             rows but the gauge reduces to {active_dimension}",
            correction.lift.nrows()
        ));
    }

    let retained = correction.rows.len();
    let lift_matrix = transform.dot(&correction.lift);
    let center = transform.dot(unconstrained_center) + &geometry.coefficient_gauge.affine_shift;

    let mut normal_center = Array1::<f64>::zeros(retained);
    let mut constraint_rows = Array2::<f64>::zeros((retained, active_dimension));
    for (position, &row) in correction.rows.iter().enumerate() {
        let a = constrained.constraints.a.row(row);
        normal_center[position] = a.dot(unconstrained_center) - constrained.constraints.b[row];
        constraint_rows.row_mut(position).assign(&a);
    }

    // The ambient covariance in RAW coordinates. `Σ_π = Σ − GΔGᵀ` is the
    // identity the correction stores, and both `G` and `Δ` push forward through
    // the gauge with the covariance, so this inverts it exactly rather than
    // recomputing a second ambient from a precision that may not be the one the
    // correction was built against.
    let ambient = {
        let scaled = lift_matrix.dot(&correction.removed_normal_variance);
        let mut out = covariance.clone();
        for i in 0..raw_dimension {
            for j in 0..=i {
                let restored = scaled.row(i).dot(&lift_matrix.row(j));
                out[[i, j]] += restored;
                if i != j {
                    out[[j, i]] = out[[i, j]];
                }
            }
        }
        out
    };

    // `W = AΣAᵀ` lives in the ACTIVE frame while the covariance we hold is the
    // raw lift of it. `Tᵀ` is surjective onto the active frame (the gauge has
    // full column rank), so every active row `a` has an exact raw
    // representative `c = T(TᵀT)⁻¹a` with `Tᵀc = a`, and then
    // `aᵀΣ_active a = cᵀ(TΣ_activeTᵀ)c`. This is a change of representative,
    // not an approximation.
    let gram = transform.t().dot(transform);
    let (eigenvalues, eigenvectors) = gram
        .eigh(faer::Side::Lower)
        .map_err(|e| format!("survival location-scale gauge Gram eigendecomposition failed: {e}"))?;
    let max_eigenvalue = eigenvalues
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
    let floor = (max_eigenvalue * PSD_EIGENVALUE_REL_TOL).max(PSD_EIGENVALUE_ABS_FLOOR);
    if eigenvalues.iter().any(|&value| value <= floor) {
        return Err(format!(
            "survival location-scale truncated response moments: the coefficient gauge is rank \
             deficient (smallest Gram eigenvalue {:.3e} against floor {floor:.3e}), so the \
             constraint rows have no raw representative",
            eigenvalues
                .iter()
                .fold(f64::INFINITY, |acc, &value| acc.min(value))
        ));
    }
    let mut gram_inverse = Array2::<f64>::zeros((active_dimension, active_dimension));
    for (column, &eigenvalue) in eigenvalues.iter().enumerate() {
        let vector = eigenvectors.column(column);
        for i in 0..active_dimension {
            for j in 0..active_dimension {
                gram_inverse[[i, j]] += vector[i] * vector[j] / eigenvalue;
            }
        }
    }
    let pulled_rows = constraint_rows.dot(&gram_inverse).dot(&transform.t());
    let normal_covariance = pulled_rows.dot(&ambient).dot(&pulled_rows.t());

    // `Σ_res = Σ − GWGᵀ`. The constraint-normal block is removed exactly, so
    // the tangent moves no constraint normal and every point of the rule below
    // is feasible.
    let residual_covariance = {
        let scaled = lift_matrix.dot(&normal_covariance);
        let mut out = ambient.clone();
        for i in 0..raw_dimension {
            for j in 0..=i {
                let removed = scaled.row(i).dot(&lift_matrix.row(j));
                out[[i, j]] -= removed;
                if i != j {
                    out[[j, i]] = out[[i, j]];
                }
            }
        }
        symmetrize_and_clip_covariance(&out)
    };

    // Two guards that are free to fail, and that fail loudly if the pieces
    // above were assembled in different frames or at different scales.
    //
    // (1) `Δ = W − Cov[u]` with both PSD, so `0 ≤ Δ_kk ≤ W_kk`. `Δ` is READ
    //     from the stored correction while `W` is RECONSTRUCTED here from the
    //     reported covariance, so this compares two independently-produced
    //     numbers: a covariance that reached us on a different scale than the
    //     one the correction was built on breaks it immediately, where a
    //     ratio-shaped check would cancel the scale and read clean.
    for k in 0..retained {
        let removed = correction.removed_normal_variance[[k, k]];
        let total = normal_covariance[[k, k]];
        let slack = ORTHANT_REMOVED_VARIANCE_SLACK * total.abs().max(removed.abs());
        if !(removed >= -slack && removed <= total + slack) {
            return Err(format!(
                "survival location-scale truncated response moments: the stored removed variance \
                 {removed:.6e} on retained row {k} is not within [0, {total:.6e}], the \
                 constraint-normal variance reconstructed from the reported covariance; the \
                 covariance reaching this rule is not the one the cone correction was built \
                 against"
            ));
        }
    }
    // (2) `E_π[β] = β_unc + G·(E[u] − E_untrunc[u])` must reproduce the
    //     coefficient vector the fit reports, in the RAW frame. This is what
    //     certifies the gauge lift of the centre and of the correction lift
    //     together; an absolute magnitude, not a ratio.
    let reported = fit.beta.clone();
    if reported.len() == raw_dimension {
        let reconstructed = &center + &lift_matrix.dot(&correction.normal_mean_shift);
        let mut worst = 0.0f64;
        for j in 0..raw_dimension {
            let scale = covariance[[j, j]].max(0.0).sqrt().max(f64::MIN_POSITIVE);
            worst = worst.max((reported[j] - reconstructed[j]).abs() / scale);
        }
        if worst > COEFFICIENT_RECONSTRUCTION_SD_TOLERANCE {
            return Err(format!(
                "survival location-scale truncated response moments: the truncated law's mean \
                 disagrees with the reported coefficient vector by {worst:.3e} posterior standard \
                 deviations, above {COEFFICIENT_RECONSTRUCTION_SD_TOLERANCE:.1e}; the ambient \
                 centre and the correction lift are not in the frame the reported coefficients are"
            ));
        }
    }

    let upper_limits = correction.upper_limits();
    let cubature = constrained_posterior_joint_cubature(
        &normal_center,
        &normal_covariance,
        &upper_limits,
        tangent_dimension,
        points,
    )?;

    Ok(Some(TruncatedCoefficientLaw {
        center,
        lift: lift_matrix,
        normal_center,
        residual_covariance,
        points: cubature,
        tangent_dimension,
    }))
}

/// Per-row response moments under the truncated law.
///
/// The structure mirrors the Gaussian rule exactly — the same projected
/// covariance on `(h, threshold, log σ)`, the same affine conditional
/// regression of the link-wiggle block onto it — with two differences, and only
/// two: the covariance those blocks are read from is `Σ_res` rather than `Σ_π`,
/// and the location is the node's `β_unc + G(u − E_untrunc[u])` rather than a
/// single moment-matched centre.
pub(crate) fn truncated_survival_response_moments_row(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
    law: &TruncatedCoefficientLaw,
    x_threshold_dense: &Array2<f64>,
    x_log_sigma_dense: &Array2<f64>,
    row: usize,
) -> Result<(f64, f64), String> {
    let beta_time = fit.beta_time();
    let beta_threshold = fit.beta_threshold();
    let beta_log_sigma = fit.beta_log_sigma();
    let beta_link_wiggle = fit.beta_link_wiggle();
    let p_time = beta_time.len();
    let p_t = beta_threshold.len();
    let p_ls = beta_log_sigma.len();
    let pw = beta_link_wiggle.as_ref().map_or(0, |beta| beta.len());
    let (time, threshold, log_sigma, wiggle) =
        survival_response_moment_block_ranges(p_time, p_t, p_ls, pw);

    let a_h = input.x_time_exit.row(row).to_owned();
    let a_t = x_threshold_dense.row(row).to_owned();
    let a_ls = x_log_sigma_dense.row(row).to_owned();

    // The AMBIENT centre of the predictor blocks: `β_unc`, not `E_π[β]`. The
    // node displacement below is measured from it, and adding it to the
    // weighted node displacement reproduces `E_π[β]` by construction.
    let center_time = law.center.slice(s![time.start..time.end]).to_owned();
    let center_threshold = law
        .center
        .slice(s![threshold.start..threshold.end])
        .to_owned();
    let center_log_sigma = law
        .center
        .slice(s![log_sigma.start..log_sigma.end])
        .to_owned();
    let mu = [
        a_h.dot(&center_time) + input.eta_time_offset_exit[row],
        a_t.dot(&center_threshold) + input.eta_threshold_offset[row],
        a_ls.dot(&center_log_sigma) + input.eta_log_sigma_offset[row],
    ];

    // `C G`: how one unit of each constraint-normal coordinate moves each of
    // the three predictor channels.
    let lift = &law.lift;
    let retained = law.normal_center.len();
    let mut channel_lift = Array2::<f64>::zeros((3, retained));
    for k in 0..retained {
        let mut h = 0.0;
        for (j, &value) in a_h.iter().enumerate() {
            h += value * lift[[time.start + j, k]];
        }
        let mut t = 0.0;
        for (j, &value) in a_t.iter().enumerate() {
            t += value * lift[[threshold.start + j, k]];
        }
        let mut l = 0.0;
        for (j, &value) in a_ls.iter().enumerate() {
            l += value * lift[[log_sigma.start + j, k]];
        }
        channel_lift[[0, k]] = h;
        channel_lift[[1, k]] = t;
        channel_lift[[2, k]] = l;
    }

    let cov_htl = projected_survival_response_moment_covariance(
        &law.residual_covariance,
        &a_h,
        &a_t,
        &a_ls,
        p_time,
        p_t,
        p_ls,
    );
    let htl_factor = factorize_psd_covariance(
        &covariance3_to_array2(cov_htl),
        "survival response-moment residual covariance",
    )?;
    let htl_rank = htl_factor.factor.ncols();
    if htl_rank + usize::from(pw > 0) > law.tangent_dimension {
        return Err(format!(
            "survival location-scale truncated response moments: row {row} needs {} tangent \
             coordinates but the joint rule carries {}",
            htl_rank + usize::from(pw > 0),
            law.tangent_dimension
        ));
    }

    // The link-wiggle block, when present, is conditioned on the realized
    // `(h, threshold, log σ)` exactly as the Gaussian rule conditions it — the
    // affine conditional mean of a joint Gaussian — because conditional on `u`
    // the law IS a joint Gaussian with covariance `Σ_res`.
    let wiggle_block = match (beta_link_wiggle.as_ref(), wiggle) {
        (Some(_), Some(wiggle_range)) => {
            let cov_wy = {
                let mut out = Array2::<f64>::zeros((pw, 3));
                let cov_wh = law
                    .residual_covariance
                    .slice(s![
                        wiggle_range.start..wiggle_range.end,
                        time.start..time.end
                    ])
                    .to_owned();
                let cov_wt = law
                    .residual_covariance
                    .slice(s![
                        wiggle_range.start..wiggle_range.end,
                        threshold.start..threshold.end
                    ])
                    .to_owned();
                let cov_wl = law
                    .residual_covariance
                    .slice(s![
                        wiggle_range.start..wiggle_range.end,
                        log_sigma.start..log_sigma.end
                    ])
                    .to_owned();
                out.column_mut(0).assign(&cov_wh.dot(&a_h));
                out.column_mut(1).assign(&cov_wt.dot(&a_t));
                out.column_mut(2).assign(&cov_wl.dot(&a_ls));
                out
            };
            let cov_ww = law
                .residual_covariance
                .slice(s![
                    wiggle_range.start..wiggle_range.end,
                    wiggle_range.start..wiggle_range.end
                ])
                .to_owned();
            let mut regression = cov_wy.dot(&htl_factor.eigenvectors);
            for column in 0..regression.ncols() {
                let scale = htl_factor.inv_sqrt_eigenvalues[column];
                regression
                    .column_mut(column)
                    .mapv_inplace(|value| value * scale);
            }
            let cov_cond = symmetrize_and_clip_covariance(
                &(cov_ww - regression.dot(&regression.t().to_owned())),
            );
            let knots = input
                .link_wiggle_knots
                .as_ref()
                .or(fit.artifacts.survival_link_wiggle_knots.as_ref())
                .ok_or_else(|| {
                    "predict_survival_location_scale: link-wiggle coefficients are missing knot \
                     metadata"
                        .to_string()
                })?
                .clone();
            let degree = input
                .link_wiggle_degree
                .or(fit.artifacts.survival_link_wiggle_degree)
                .ok_or_else(|| {
                    "predict_survival_location_scale: link-wiggle coefficients are missing degree \
                     metadata"
                        .to_string()
                })?;
            let center_wiggle = law
                .center
                .slice(s![wiggle_range.start..wiggle_range.end])
                .to_owned();
            let mut wiggle_lift = Array2::<f64>::zeros((pw, retained));
            for j in 0..pw {
                for k in 0..retained {
                    wiggle_lift[[j, k]] = lift[[wiggle_range.start + j, k]];
                }
            }
            Some((
                regression,
                cov_cond,
                knots,
                degree,
                center_wiggle,
                wiggle_lift,
            ))
        }
        _ => None,
    };

    let mut first = 0.0f64;
    let mut second = 0.0f64;
    let mut displacement = Array1::<f64>::zeros(retained);
    for point in &law.points {
        for k in 0..retained {
            displacement[k] = point.normal_coordinates[k] - law.normal_center[k];
        }
        let mut x = mu;
        for (channel, value) in x.iter_mut().enumerate() {
            for k in 0..retained {
                *value += channel_lift[[channel, k]] * displacement[k];
            }
            for column in 0..htl_rank {
                *value += htl_factor.factor[[channel, column]] * point.tangent[column];
            }
        }
        let q0 = survival_q0_from_eta(x[1], x[2]);
        let eta = match wiggle_block.as_ref() {
            None => x[0] + q0,
            Some((regression, cov_cond, knots, degree, center_wiggle, wiggle_lift)) => {
                let q0_arr = Array1::from_vec(vec![q0]);
                let basis = survival_wiggle_basis_with_options(
                    q0_arr.view(),
                    knots,
                    *degree,
                    BasisOptions::value(),
                )?;
                if basis.ncols() != center_wiggle.len() {
                    return Err(SurvivalLocationScaleError::DimensionMismatch {
                        reason: format!(
                            "predict_survival_location_scale: link-wiggle basis/beta mismatch: \
                             {} vs {}",
                            basis.ncols(),
                            center_wiggle.len()
                        ),
                    }
                    .into());
                }
                let b = basis.row(0).to_owned();
                let mut conditional_mean = center_wiggle.clone();
                for j in 0..conditional_mean.len() {
                    let mut shift = 0.0;
                    for k in 0..retained {
                        shift += wiggle_lift[[j, k]] * displacement[k];
                    }
                    for column in 0..htl_rank {
                        shift += regression[[j, column]] * point.tangent[column];
                    }
                    conditional_mean[j] += shift;
                }
                let w_mean = b.dot(&conditional_mean);
                let w_variance = b.dot(&cov_cond.dot(&b)).max(0.0);
                let w = w_mean + w_variance.sqrt() * point.tangent[law.tangent_dimension - 1];
                x[0] + q0 + w
            }
        };
        let probability = inverse_link_survival_prob_checked(&input.inverse_link, eta)?;
        first += point.weight * probability;
        second += point.weight * probability * probability;
    }
    Ok((first.clamp(0.0, 1.0), second.clamp(0.0, 1.0)))
}

/// Response moments for every row under the truncated law, with the point count
/// chosen by refinement.
pub(crate) fn truncated_survival_response_moments(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
    covariance: &Array2<f64>,
    x_threshold_dense: &Array2<f64>,
    x_log_sigma_dense: &Array2<f64>,
) -> Result<Option<(Array1<f64>, Array1<f64>)>, String> {
    let n = input.x_time_exit.nrows();
    let tangent_dimension = 3 + usize::from(fit.beta_link_wiggle().is_some());
    let mut points = TRUNCATED_RESPONSE_MOMENT_INITIAL_POINTS;
    let Some(law) = build_truncated_coefficient_law(fit, covariance, tangent_dimension, points)?
    else {
        return Ok(None);
    };
    let mut previous = evaluate_truncated_rows(input, fit, &law, x_threshold_dense, x_log_sigma_dense, n)?;
    loop {
        points *= 2;
        let law = build_truncated_coefficient_law(fit, covariance, tangent_dimension, points)?
            .ok_or_else(|| {
                "survival location-scale truncated response moments: the truncated law became \
                 unavailable between refinement passes"
                    .to_string()
            })?;
        let current =
            evaluate_truncated_rows(input, fit, &law, x_threshold_dense, x_log_sigma_dense, n)?;
        let change = previous
            .0
            .iter()
            .zip(current.0.iter())
            .map(|(a, b)| (a - b).abs())
            .chain(
                previous
                    .1
                    .iter()
                    .zip(current.1.iter())
                    .map(|(a, b)| (a - b).abs()),
            )
            .fold(0.0f64, f64::max);
        if change <= TRUNCATED_RESPONSE_MOMENT_ABSOLUTE_TOLERANCE {
            return Ok(Some(current));
        }
        if points >= TRUNCATED_RESPONSE_MOMENT_MAXIMUM_POINTS {
            return Err(format!(
                "survival location-scale truncated response moments did not converge: the \
                 largest response-moment change between {} and {points} joint cubature points is \
                 {change:.3e}, still above {TRUNCATED_RESPONSE_MOMENT_ABSOLUTE_TOLERANCE:.1e}",
                points / 2
            ));
        }
        previous = current;
    }
}

fn evaluate_truncated_rows(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
    law: &TruncatedCoefficientLaw,
    x_threshold_dense: &Array2<f64>,
    x_log_sigma_dense: &Array2<f64>,
    n: usize,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    let mut first = Array1::<f64>::zeros(n);
    let mut second = Array1::<f64>::zeros(n);
    if n >= SURVIVAL_ROW_PARALLEL_THRESHOLD {
        let first_slice = first
            .as_slice_mut()
            .expect("fresh Array1 response moments are contiguous");
        let second_slice = second
            .as_slice_mut()
            .expect("fresh Array1 response moments are contiguous");
        first_slice
            .par_chunks_mut(SURVIVAL_ROW_PARALLEL_CHUNK)
            .zip(second_slice.par_chunks_mut(SURVIVAL_ROW_PARALLEL_CHUNK))
            .enumerate()
            .try_for_each(
                |(chunk_idx, (first_chunk, second_chunk))| -> Result<(), String> {
                    let row_start = chunk_idx * SURVIVAL_ROW_PARALLEL_CHUNK;
                    for offset in 0..first_chunk.len() {
                        let (m1, m2) = truncated_survival_response_moments_row(
                            input,
                            fit,
                            law,
                            x_threshold_dense,
                            x_log_sigma_dense,
                            row_start + offset,
                        )?;
                        first_chunk[offset] = m1;
                        second_chunk[offset] = m2;
                    }
                    Ok(())
                },
            )?;
    } else {
        for row in 0..n {
            let (m1, m2) = truncated_survival_response_moments_row(
                input,
                fit,
                law,
                x_threshold_dense,
                x_log_sigma_dense,
                row,
            )?;
            first[row] = m1;
            second[row] = m2;
        }
    }
    Ok((first, second))
}


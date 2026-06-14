
fn prediction_linear_predictors(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
) -> Result<PredictionLinearPredictors, String> {
    validate_predict_inverse_link(&input.inverse_link)?;
    let components = location_scale_eta_components(
        &input.x_time_exit,
        &input.eta_time_offset_exit,
        input.time_wiggle_knots.as_ref(),
        input.time_wiggle_degree,
        input.time_wiggle_ncols,
        &input.x_threshold,
        &input.eta_threshold_offset,
        &input.x_log_sigma,
        &input.eta_log_sigma_offset,
        fit,
    )?;
    prediction_linear_predictors_from_eta_components(
        components,
        input.link_wiggle_knots.as_ref(),
        input.link_wiggle_degree,
        fit,
    )
}


pub(crate) fn predict_survival_location_scale_from_linear_components(
    x_time_exit: &Array2<f64>,
    eta_time_offset_exit: &Array1<f64>,
    time_wiggle_knots: Option<&Array1<f64>>,
    time_wiggle_degree: Option<usize>,
    time_wiggle_ncols: usize,
    eta_t: &Array1<f64>,
    eta_ls: &Array1<f64>,
    link_wiggle_knots: Option<&Array1<f64>>,
    link_wiggle_degree: Option<usize>,
    inverse_link: &InverseLink,
    fit: &UnifiedFitResult,
) -> Result<SurvivalLocationScalePredictResult, String> {
    validate_predict_inverse_link(inverse_link)?;
    let predictors = prediction_linear_predictors_from_components(
        x_time_exit,
        eta_time_offset_exit,
        time_wiggle_knots,
        time_wiggle_degree,
        time_wiggle_ncols,
        eta_t,
        eta_ls,
        link_wiggle_knots,
        link_wiggle_degree,
        fit,
    )?;
    survival_location_scale_response_from_predictors(inverse_link, predictors)
}


fn prediction_linear_predictors_from_components(
    x_time_exit: &Array2<f64>,
    eta_time_offset_exit: &Array1<f64>,
    time_wiggle_knots: Option<&Array1<f64>>,
    time_wiggle_degree: Option<usize>,
    time_wiggle_ncols: usize,
    eta_t: &Array1<f64>,
    eta_ls: &Array1<f64>,
    link_wiggle_knots: Option<&Array1<f64>>,
    link_wiggle_degree: Option<usize>,
    fit: &UnifiedFitResult,
) -> Result<PredictionLinearPredictors, String> {
    let n = x_time_exit.nrows();
    if eta_time_offset_exit.len() != n || eta_t.len() != n || eta_ls.len() != n {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: "predict_survival_location_scale: row mismatch across inputs".to_string(),
        }
        .into());
    }
    let time_components = location_scale_time_warp_components(
        x_time_exit,
        eta_time_offset_exit,
        time_wiggle_knots,
        time_wiggle_degree,
        time_wiggle_ncols,
        fit,
    )?;
    let inv_sigma = eta_ls.mapv(exp_sigma_inverse_from_eta_scalar);
    prediction_linear_predictors_from_parts(
        time_components.h,
        time_components.time_jac,
        eta_t.clone(),
        eta_ls.clone(),
        inv_sigma,
        link_wiggle_knots,
        link_wiggle_degree,
        fit,
    )
}


fn prediction_linear_predictors_from_eta_components(
    components: LocationScaleEtaComponents,
    link_wiggle_knots: Option<&Array1<f64>>,
    link_wiggle_degree: Option<usize>,
    fit: &UnifiedFitResult,
) -> Result<PredictionLinearPredictors, String> {
    prediction_linear_predictors_from_parts(
        components.h,
        components.time_jac,
        components.eta_t,
        components.eta_ls,
        components.inv_sigma,
        link_wiggle_knots,
        link_wiggle_degree,
        fit,
    )
}


fn prediction_linear_predictors_from_parts(
    h: Array1<f64>,
    time_jac: Array2<f64>,
    eta_t: Array1<f64>,
    eta_ls: Array1<f64>,
    inv_sigma: Array1<f64>,
    link_wiggle_knots: Option<&Array1<f64>>,
    link_wiggle_degree: Option<usize>,
    fit: &UnifiedFitResult,
) -> Result<PredictionLinearPredictors, String> {
    let n = h.len();
    let beta_link_wiggle = fit.beta_link_wiggle();
    if time_jac.nrows() != n || eta_t.len() != n || eta_ls.len() != n || inv_sigma.len() != n {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: "predict_survival_location_scale: row mismatch across inputs".to_string(),
        }
        .into());
    }
    let resolved_wiggle_knots =
        link_wiggle_knots.or(fit.artifacts.survival_link_wiggle_knots.as_ref());
    let resolved_wiggle_degree = link_wiggle_degree.or(fit.artifacts.survival_link_wiggle_degree);
    let q0 = Array1::from_shape_fn(n, |i| survival_q0_from_eta(eta_t[i], eta_ls[i]));
    let (wiggle_design, dq_dq0, etaw) = if let Some(betaw) = beta_link_wiggle.as_ref() {
        let knots = resolved_wiggle_knots.ok_or_else(|| {
            "predict_survival_location_scale: link-wiggle coefficients are missing knot metadata"
                .to_string()
        })?;
        let degree = resolved_wiggle_degree.ok_or_else(|| {
            "predict_survival_location_scale: link-wiggle coefficients are missing degree metadata"
                .to_string()
        })?;
        let design =
            survival_wiggle_basis_with_options(q0.view(), knots, degree, BasisOptions::value())?;
        if design.ncols() != betaw.len() {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "predict_survival_location_scale: link-wiggle design/beta mismatch: {} vs {}",
                    design.ncols(),
                    betaw.len()
                ),
            }
            .into());
        }
        let basis_d1 = survival_wiggle_basis_with_options(
            q0.view(),
            knots,
            degree,
            BasisOptions::first_derivative(),
        )?;
        let dq = Some(fast_av(&basis_d1, betaw) + 1.0);
        let etaw = fast_av(&design, betaw);
        (Some(design), dq, Some(etaw))
    } else {
        (None, None, None)
    };
    Ok(PredictionLinearPredictors {
        h,
        time_jac,
        eta_t,
        inv_sigma,
        etaw,
        wiggle_design,
        dq_dq0,
    })
}


fn survival_response_moment_block_ranges(
    p_time: usize,
    p_t: usize,
    p_ls: usize,
    pw: usize,
) -> (
    std::ops::Range<usize>,
    std::ops::Range<usize>,
    std::ops::Range<usize>,
    Option<std::ops::Range<usize>>,
) {
    let time = 0..p_time;
    let threshold = time.end..time.end + p_t;
    let log_sigma = threshold.end..threshold.end + p_ls;
    let wiggle = (pw > 0).then_some(log_sigma.end..log_sigma.end + pw);
    (time, threshold, log_sigma, wiggle)
}


fn projected_survival_response_moment_covariance(
    covariance: &Array2<f64>,
    a_h: &Array1<f64>,
    a_t: &Array1<f64>,
    a_ls: &Array1<f64>,
    p_time: usize,
    p_t: usize,
    p_ls: usize,
) -> [[f64; 3]; 3] {
    let (time, threshold, log_sigma, _) =
        survival_response_moment_block_ranges(p_time, p_t, p_ls, 0);
    let cov_hh = covariance.slice(s![time.start..time.end, time.start..time.end]);
    let cov_tt = covariance.slice(s![
        threshold.start..threshold.end,
        threshold.start..threshold.end
    ]);
    let cov_ll = covariance.slice(s![
        log_sigma.start..log_sigma.end,
        log_sigma.start..log_sigma.end
    ]);
    let cov_ht = covariance.slice(s![time.start..time.end, threshold.start..threshold.end]);
    let cov_hl = covariance.slice(s![time.start..time.end, log_sigma.start..log_sigma.end]);
    let cov_tl = covariance.slice(s![
        threshold.start..threshold.end,
        log_sigma.start..log_sigma.end
    ]);
    let var_h = a_h.dot(&cov_hh.dot(a_h));
    let var_t = a_t.dot(&cov_tt.dot(a_t));
    let var_ls = a_ls.dot(&cov_ll.dot(a_ls));
    let cov_ht_i = a_h.dot(&cov_ht.dot(a_t));
    let cov_hl_i = a_h.dot(&cov_hl.dot(a_ls));
    let cov_tl_i = a_t.dot(&cov_tl.dot(a_ls));
    [
        [var_h, cov_ht_i, cov_hl_i],
        [cov_ht_i, var_t, cov_tl_i],
        [cov_hl_i, cov_tl_i, var_ls],
    ]
}


fn covariance3_to_array2(cov: [[f64; 3]; 3]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            out[[i, j]] = cov[i][j];
        }
    }
    out
}


fn symmetrize_and_clip_covariance(cov: &Array2<f64>) -> Array2<f64> {
    let mut out = cov.clone();
    for i in 0..out.nrows() {
        out[[i, i]] = out[[i, i]].max(0.0);
        for j in (i + 1)..out.ncols() {
            let avg = 0.5 * (out[[i, j]] + out[[j, i]]);
            out[[i, j]] = avg;
            out[[j, i]] = avg;
        }
    }
    out
}


struct LowRankGaussianFactor {
    factor: Array2<f64>,
    eigenvectors: Array2<f64>,
    inv_sqrt_eigenvalues: Array1<f64>,
}


// Exact projected-Gaussian handling for possibly singular covariance blocks.
// We integrate over the active standard-normal coordinates rather than adding
// jitter or inverting the covariance directly.
fn factorize_psd_covariance(
    covariance: &Array2<f64>,
    label: &str,
) -> Result<LowRankGaussianFactor, String> {
    let covariance = symmetrize_and_clip_covariance(covariance);
    let (eigenvalues, eigenvectors_full) = covariance
        .eigh(faer::Side::Lower)
        .map_err(|e| format!("{label} eigendecomposition failed: {e}"))?;
    let max_abs_eigenvalue = eigenvalues
        .iter()
        .fold(0.0_f64, |acc, &ev| acc.max(ev.abs()));
    let tol = (max_abs_eigenvalue * PSD_EIGENVALUE_REL_TOL).max(PSD_EIGENVALUE_ABS_FLOOR);
    if eigenvalues.iter().any(|&ev| ev < -tol) {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: format!(
                "{label} is not positive semidefinite: minimum eigenvalue {:.3e}",
                eigenvalues
                    .iter()
                    .fold(f64::INFINITY, |acc, &ev| acc.min(ev))
            ),
        }
        .into());
    }

    let active = eigenvalues
        .iter()
        .enumerate()
        .filter_map(|(idx, &ev)| (ev > tol).then_some((idx, ev.sqrt())))
        .collect::<Vec<_>>();
    let mut factor = Array2::<f64>::zeros((covariance.nrows(), active.len()));
    let mut eigenvectors = Array2::<f64>::zeros((covariance.nrows(), active.len()));
    let mut inv_sqrt_eigenvalues = Array1::<f64>::zeros(active.len());
    for (col, (idx, sqrt_ev)) in active.into_iter().enumerate() {
        eigenvectors
            .column_mut(col)
            .assign(&eigenvectors_full.column(idx));
        factor
            .column_mut(col)
            .assign(&(&eigenvectors_full.column(idx) * sqrt_ev));
        inv_sqrt_eigenvalues[col] = 1.0 / sqrt_ev;
    }

    Ok(LowRankGaussianFactor {
        factor,
        eigenvectors,
        inv_sqrt_eigenvalues,
    })
}


fn apply_low_rank_gaussian_factor3(mu: [f64; 3], factor: &Array2<f64>, z: &[f64]) -> [f64; 3] {
    let mut x = mu;
    for row in 0..3 {
        for (col, &latent) in z.iter().enumerate() {
            x[row] += factor[[row, col]] * latent;
        }
    }
    x
}


fn low_rank_normal_expectation_pair_3d_result<F>(
    quadctx: &crate::quadrature::QuadratureContext,
    mu: [f64; 3],
    covariance: [[f64; 3]; 3],
    max_n: usize,
    label: &str,
    integrand: F,
) -> Result<(f64, f64), String>
where
    F: Fn([f64; 3], &[f64]) -> Result<(f64, f64), String>,
{
    let factorization = factorize_psd_covariance(&covariance3_to_array2(covariance), label)?;
    match factorization.factor.ncols() {
        0 => integrand(mu, &[]),
        1 => crate::quadrature::normal_expectation_nd_adaptive_result::<1, _, _, String>(
            quadctx,
            [0.0],
            [[1.0]],
            max_n,
            |z| {
                let latent = [z[0]];
                integrand(
                    apply_low_rank_gaussian_factor3(mu, &factorization.factor, &latent),
                    &latent,
                )
            },
        ),
        2 => crate::quadrature::normal_expectation_nd_adaptive_result::<2, _, _, String>(
            quadctx,
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            max_n,
            |z| {
                let latent = [z[0], z[1]];
                integrand(
                    apply_low_rank_gaussian_factor3(mu, &factorization.factor, &latent),
                    &latent,
                )
            },
        ),
        3 => crate::quadrature::normal_expectation_nd_adaptive_result::<3, _, _, String>(
            quadctx,
            [0.0, 0.0, 0.0],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            max_n,
            |z| {
                let latent = [z[0], z[1], z[2]];
                integrand(
                    apply_low_rank_gaussian_factor3(mu, &factorization.factor, &latent),
                    &latent,
                )
            },
        ),
        rank => Err(SurvivalLocationScaleError::InternalInvariant {
            reason: format!("{label} unexpectedly has rank {rank} > 3"),
        }
        .into()),
    }
}


// Exact response moments must stay in the original Gaussian coordinates:
// [h, threshold, log_sigma] for non-wiggle predictions, with a nested
// conditional Gaussian over the scalar link-wiggle contribution when present.
fn exact_survival_response_moments_row(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
    covariance: &Array2<f64>,
    x_threshold_dense: &Array2<f64>,
    x_log_sigma_dense: &Array2<f64>,
    row: usize,
    quadctx: &crate::quadrature::QuadratureContext,
) -> Result<(f64, f64), String> {
    if input.time_wiggle_ncols > 0 {
        return Err(SurvivalLocationScaleError::InvalidConfiguration { reason: "predict_survival_location_scale: exact response moments are not implemented for time-wiggle models"
                .to_string(), }.into());
    }

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

    let mu_h = a_h.dot(&beta_time) + input.eta_time_offset_exit[row];
    let mu_t = a_t.dot(&beta_threshold) + input.eta_threshold_offset[row];
    let mu_ls = a_ls.dot(&beta_log_sigma) + input.eta_log_sigma_offset[row];
    let mu = [mu_h, mu_t, mu_ls];
    let cov_htl = projected_survival_response_moment_covariance(
        covariance, &a_h, &a_t, &a_ls, p_time, p_t, p_ls,
    );

    if let (Some(beta_w), Some(wiggle_range)) = (beta_link_wiggle.as_ref(), wiggle) {
        let knots = input
            .link_wiggle_knots
            .as_ref()
            .or(fit.artifacts.survival_link_wiggle_knots.as_ref())
            .ok_or_else(|| {
                "predict_survival_location_scale: link-wiggle coefficients are missing knot metadata"
                    .to_string()
            })?;
        let degree = input
            .link_wiggle_degree
            .or(fit.artifacts.survival_link_wiggle_degree)
            .ok_or_else(|| {
                "predict_survival_location_scale: link-wiggle coefficients are missing degree metadata"
                    .to_string()
            })?;

        let htl_factor = factorize_psd_covariance(
            &covariance3_to_array2(cov_htl),
            "survival response-moment projected covariance",
        )?;

        let cov_wy = {
            let mut out = Array2::<f64>::zeros((pw, 3));
            let cov_wh = covariance
                .slice(s![
                    wiggle_range.start..wiggle_range.end,
                    time.start..time.end
                ])
                .to_owned();
            let cov_wt = covariance
                .slice(s![
                    wiggle_range.start..wiggle_range.end,
                    threshold.start..threshold.end
                ])
                .to_owned();
            let cov_wl = covariance
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
        let cov_ww = covariance
            .slice(s![
                wiggle_range.start..wiggle_range.end,
                wiggle_range.start..wiggle_range.end
            ])
            .to_owned();
        let mut regression = cov_wy.dot(&htl_factor.eigenvectors);
        for col in 0..regression.ncols() {
            let scale = htl_factor.inv_sqrt_eigenvalues[col];
            regression
                .column_mut(col)
                .mapv_inplace(|value| value * scale);
        }
        let cov_cond =
            symmetrize_and_clip_covariance(&(cov_ww - regression.dot(&regression.t().to_owned())));

        return low_rank_normal_expectation_pair_3d_result(
            quadctx,
            mu,
            cov_htl,
            15,
            "survival response-moment projected covariance",
            |x, z| {
                let mut cond_mean = beta_w.to_owned();
                for j in 0..pw {
                    for (col, &latent) in z.iter().enumerate() {
                        cond_mean[j] += regression[[j, col]] * latent;
                    }
                }
                let q0 = survival_q0_from_eta(x[1], x[2]);
                let q0_arr = Array1::from_vec(vec![q0]);
                let basis = survival_wiggle_basis_with_options(
                    q0_arr.view(),
                    knots,
                    degree,
                    BasisOptions::value(),
                )?;
                if basis.ncols() != cond_mean.len() {
                    return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                        "predict_survival_location_scale: link-wiggle basis/beta mismatch: {} vs {}",
                        basis.ncols(),
                        cond_mean.len()
                    ) }.into());
                }
                let b = basis.row(0).to_owned();
                let w_mean = b.dot(&cond_mean);
                let w_var = b.dot(&cov_cond.dot(&b)).max(0.0);
                crate::quadrature::normal_expectation_nd_adaptive_result::<1, _, _, String>(
                    quadctx,
                    [x[0] + q0 + w_mean],
                    [[w_var]],
                    21,
                    |eta| {
                        let p = inverse_link_survival_prob_checked(&input.inverse_link, eta[0])?;
                        Ok((p, p * p))
                    },
                )
            },
        )
        .map(|(first, second)| (first.clamp(0.0, 1.0), second.clamp(0.0, 1.0)));
    }

    low_rank_normal_expectation_pair_3d_result(
        quadctx,
        mu,
        cov_htl,
        15,
        "survival response-moment projected covariance",
        |x, _| {
            let p = inverse_link_survival_prob_checked(
                &input.inverse_link,
                x[0] + survival_q0_from_eta(x[1], x[2]),
            )?;
            Ok((p, p * p))
        },
    )
    .map(|(first, second)| (first.clamp(0.0, 1.0), second.clamp(0.0, 1.0)))
}


fn exact_survival_response_moments(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
    covariance: &Array2<f64>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    validate_predict_inverse_link(&input.inverse_link)?;

    let n = input.x_time_exit.nrows();
    let p_time = fit.beta_time().len();
    let p_t = fit.beta_threshold().len();
    let p_ls = fit.beta_log_sigma().len();
    let pw = fit.beta_link_wiggle().map_or(0, |beta| beta.len());
    let p_total = p_time + p_t + p_ls + pw;
    if covariance.nrows() != p_total || covariance.ncols() != p_total {
        return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
            "predict_survival_location_scale: covariance shape mismatch: got {}x{}, expected {}x{}",
            covariance.nrows(),
            covariance.ncols(),
            p_total,
            p_total
        ) }.into());
    }
    if input.x_time_exit.ncols() != p_time {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "predict_survival_location_scale: time design/beta mismatch: {} vs {}",
                input.x_time_exit.ncols(),
                p_time
            ),
        }
        .into());
    }
    if input.eta_time_offset_exit.len() != n
        || input.x_threshold.nrows() != n
        || input.eta_threshold_offset.len() != n
        || input.x_log_sigma.nrows() != n
        || input.eta_log_sigma_offset.len() != n
    {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: "predict_survival_location_scale: row mismatch across inputs".to_string(),
        }
        .into());
    }

    let x_threshold_dense = input.x_threshold.to_dense_arc();
    let x_log_sigma_dense = input.x_log_sigma.to_dense_arc();
    let mut first = Array1::<f64>::zeros(n);
    let mut second = Array1::<f64>::zeros(n);
    // Build a single QuadratureContext up front and share it across all
    // chunks.  Per-chunk construction wastes work (each chunk's first call
    // re-derives the Gauss-Hermite rule from scratch via OnceLock) and risks
    // the OnceLock-inside-rayon deadlock pattern (see repo memory) if the
    // rule init were ever to spawn nested parallel work.  Warm the rule sizes
    // that the per-row evaluator actually uses (15 for the projected 3D
    // path, 21 for the 1D wiggle fallback) so the worker threads only hit
    // the cached rule lookup.
    let quadctx = crate::quadrature::QuadratureContext::new();
    {
        // Warm GH rule caches on the calling thread with cheap probes.
        crate::quadrature::normal_expectation_nd_adaptive_result::<1, _, _, String>(
            &quadctx,
            [0.0_f64],
            [[1.0_f64]],
            21,
            |_x: [f64; 1]| Ok((0.0_f64, 0.0_f64)),
        )?;
        crate::quadrature::normal_expectation_nd_adaptive_result::<1, _, _, String>(
            &quadctx,
            [0.0_f64],
            [[1.0_f64]],
            15,
            |_x: [f64; 1]| Ok((0.0_f64, 0.0_f64)),
        )?;
    }
    if n >= SURVIVAL_ROW_PARALLEL_THRESHOLD {
        let first_slice = first
            .as_slice_mut()
            .expect("fresh Array1 response moments are contiguous");
        let second_slice = second
            .as_slice_mut()
            .expect("fresh Array1 response moments are contiguous");
        let quadctx_ref = &quadctx;
        first_slice
            .par_chunks_mut(SURVIVAL_ROW_PARALLEL_CHUNK)
            .zip(second_slice.par_chunks_mut(SURVIVAL_ROW_PARALLEL_CHUNK))
            .enumerate()
            .try_for_each(
                |(chunk_idx, (first_chunk, second_chunk))| -> Result<(), String> {
                    let row_start = chunk_idx * SURVIVAL_ROW_PARALLEL_CHUNK;
                    for offset in 0..first_chunk.len() {
                        let row = row_start + offset;
                        let (m1, m2) = exact_survival_response_moments_row(
                            input,
                            fit,
                            covariance,
                            &x_threshold_dense,
                            &x_log_sigma_dense,
                            row,
                            quadctx_ref,
                        )?;
                        first_chunk[offset] = m1;
                        second_chunk[offset] = m2;
                    }
                    Ok(())
                },
            )?;
    } else {
        for row in 0..n {
            let (m1, m2) = exact_survival_response_moments_row(
                input,
                fit,
                covariance,
                &x_threshold_dense,
                &x_log_sigma_dense,
                row,
                &quadctx,
            )?;
            first[row] = m1;
            second[row] = m2;
        }
    }
    Ok((first, second))
}


fn lift_conditional_covariance(
    cov_reduced: &Array2<f64>,
    z: &Array2<f64>,
    p_threshold_reduced: usize,
    p_threshold_full: usize,
    threshold_fixed_cols: usize,
    p_log_sigma_reduced: usize,
    p_log_sigma_full: usize,
    log_sigma_fixed_cols: usize,
    p_linkwiggle: usize,
) -> Result<Array2<f64>, String> {
    let p_time_reduced = z.ncols();
    let p_time_full = z.nrows();
    if threshold_fixed_cols + p_threshold_reduced != p_threshold_full {
        return Err(SurvivalLocationScaleError::InvalidConfiguration { reason: format!(
            "survival location-scale covariance lift threshold dimensions are inconsistent: fixed={}, reduced={}, full={}",
            threshold_fixed_cols, p_threshold_reduced, p_threshold_full
        ) }.into());
    }
    if log_sigma_fixed_cols + p_log_sigma_reduced != p_log_sigma_full {
        return Err(SurvivalLocationScaleError::InvalidConfiguration { reason: format!(
            "survival location-scale covariance lift log-sigma dimensions are inconsistent: fixed={}, reduced={}, full={}",
            log_sigma_fixed_cols, p_log_sigma_reduced, p_log_sigma_full
        ) }.into());
    }
    // Raw↔canonical reconciliation at the time sub-block. The time
    // identifiability map `z` lifts the inner solver's ACTIVE (reduced,
    // canonical-gauge) time coefficients back to the RAW time layout, so it must
    // be at least as tall as it is wide — the active block can never carry more
    // columns than the raw block it expands into. If a future canonicalization
    // ever produces a `z` whose active width exceeds the raw width (the
    // raw-vs-active drift behind the historical `[N,N] → [N-1,N-1]` finalization
    // panic, #735), surface it here as a structured DimensionMismatch instead of
    // letting the downstream block assignment fault with a bare ndarray
    // broadcast. The threshold/log_sigma offsets are already validated above via
    // `*_fixed_cols + reduced == full`; this is the matching guard for the one
    // time block whose map is a dense matrix rather than a fixed-column offset.
    if p_time_reduced > p_time_full {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "survival location-scale covariance lift time map is wider than tall: \
             active(reduced)={p_time_reduced} exceeds raw(full)={p_time_full}; \
             the time identifiability transform `z` must map reduced→raw"
            ),
        }
        .into());
    }

    let p_reduced = p_time_reduced + p_threshold_reduced + p_log_sigma_reduced + p_linkwiggle;
    let p_full = p_time_full + p_threshold_full + p_log_sigma_full + p_linkwiggle;
    if cov_reduced.nrows() != p_reduced || cov_reduced.ncols() != p_reduced {
        return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
            "survival location-scale covariance lift expected reduced matrix {p_reduced}x{p_reduced}, got {}x{}",
            cov_reduced.nrows(),
            cov_reduced.ncols()
        ) }.into());
    }

    // The destination slice is sized from `z` itself (`p_time_full` rows,
    // `p_time_reduced` cols), so the assign cannot broadcast-fault as long as the
    // guards above hold and `t_map` is at least that large — which `p_full ≥
    // p_time_full` and `p_reduced ≥ p_time_reduced` guarantee by construction.
    let mut t_map = Array2::<f64>::zeros((p_full, p_reduced));
    t_map
        .slice_mut(s![0..p_time_full, 0..p_time_reduced])
        .assign(z);
    for j in 0..p_threshold_reduced {
        t_map[[p_time_full + threshold_fixed_cols + j, p_time_reduced + j]] = 1.0;
    }
    for j in 0..p_log_sigma_reduced {
        t_map[[
            p_time_full + p_threshold_full + log_sigma_fixed_cols + j,
            p_time_reduced + p_threshold_reduced + j,
        ]] = 1.0;
    }
    for j in 0..p_linkwiggle {
        t_map[[
            p_time_full + p_threshold_full + p_log_sigma_full + j,
            p_time_reduced + p_threshold_reduced + p_log_sigma_reduced + j,
        ]] = 1.0;
    }
    Ok(t_map.dot(cov_reduced).dot(&t_map.t()))
}


impl SurvivalLocationScaleFamily {
    /// Recompute every block's linear predictor `η_b = D_b · β_b + o_b` from
    /// the joint coefficient vector `theta` (block-concatenated) and the block
    /// specs, returning freshly populated [`ParameterBlockState`]s.
    ///
    /// This mirrors the static-geometry branch of the inner solver's
    /// `refresh_all_block_etas`: in the reduced constant-scale parametric-AFT
    /// regime there is no link-wiggle and no monotone time-wiggle, so the
    /// family geometry is static and `solver_design()`/`solver_offset()`
    /// (the stacked `[entry; exit; deriv]` channels) map β to η directly. Each
    /// block's β is passed through `post_update_block_beta` so the time-warp
    /// monotonicity constraints are validated exactly as the coupled path does.
    fn parametric_aft_states_from_theta(
        &self,
        theta: &Array1<f64>,
        specs: &[ParameterBlockSpec],
    ) -> Result<Vec<ParameterBlockState>, String> {
        let offsets = self.joint_block_offsets();
        if theta.len() != *offsets.last().unwrap_or(&0) {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "parametric-AFT direct MLE theta length mismatch: got {}, expected {}",
                    theta.len(),
                    offsets.last().copied().unwrap_or(0)
                ),
            }
            .into());
        }
        let mut states = Vec::with_capacity(specs.len());
        for (b, spec) in specs.iter().enumerate() {
            let beta = theta.slice(s![offsets[b]..offsets[b + 1]]).to_owned();
            let eta = spec.solver_design().matrixvectormultiply(&beta) + spec.solver_offset();
            states.push(ParameterBlockState { beta, eta });
        }
        // Validate (and, for any family that projects, project) each block's β
        // against its constraints — the time block's monotone-derivative guard.
        for b in 0..specs.len() {
            let raw = states[b].beta.clone();
            let projected = self.post_update_block_beta(&states, b, &specs[b], raw)?;
            if projected != states[b].beta {
                states[b].beta.assign(&projected);
                states[b].eta = specs[b]
                    .solver_design()
                    .matrixvectormultiply(&states[b].beta)
                    + specs[b].solver_offset();
            }
        }
        Ok(states)
    }

    /// Direct, robust maximum-likelihood fit of the fully reduced constant-scale
    /// parametric AFT (affine time-warp + location intercept/covariates +
    /// constant log-σ).
    ///
    /// In this regime every block is UNPENALIZED — there are no smoothing
    /// parameters and the REML/LAML outer search is vacuous — so the coupled
    /// exact-joint REML machinery is the wrong tool (issue #736/#735/#721): it
    /// runs an outer ρ search around an inner per-block trust-region Newton that
    /// oscillates and never certifies stationarity on this tiny unpenalized
    /// likelihood. Instead we run a damped, line-searched joint Newton directly
    /// on the negative log-likelihood `−ℓ(θ)`, converging to the gradient-norm
    /// tolerance in a handful of iterations exactly like `survreg`/`lifelines`.
    ///
    /// The step is `δ = H⁻¹ g` with `g = ∇ℓ` (the block-concatenated
    /// log-likelihood gradient) and `H = −∇²ℓ` (the exact joint Hessian, all
    /// cross-blocks included). When `H` is not positive definite at the current
    /// iterate we add Levenberg damping `τ·I` (escalating geometrically) until
    /// the Cholesky factorization succeeds, giving a guaranteed ascent
    /// direction. The step length is first capped to keep the monotone
    /// time-warp feasible (`max_feasible_step_size`) and then Armijo-backtracked
    /// on `−ℓ`, so the time derivative stays `≥ guard` at every observed time
    /// and `ℓ` increases monotonically.
    ///
    /// Returns the converged block states, the log-likelihood at the MLE, and
    /// the joint negative-log-likelihood Hessian `H` (the observed information),
    /// whose inverse is the conditional covariance the caller assembles.
    fn fit_parametric_aft_direct_mle(
        &self,
        specs: &[ParameterBlockSpec],
        max_iter: usize,
        grad_tol: f64,
    ) -> Result<(Vec<ParameterBlockState>, f64, Array2<f64>), String> {
        use crate::faer_ndarray::FaerCholesky;

        self.validate_joint_specs(
            specs,
            "SurvivalLocationScaleFamily direct parametric-AFT MLE",
        )?;
        let offsets = self.joint_block_offsets();
        let p_total = *offsets.last().unwrap_or(&0);
        if p_total == 0 {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: "direct parametric-AFT MLE has no free coefficients".to_string(),
            }
            .into());
        }

        // Cold-start θ from the block specs' (feasible) initial β, falling back
        // to zeros. `parametric_aft_states_from_theta` re-validates feasibility.
        let mut theta = Array1::<f64>::zeros(p_total);
        for (b, spec) in specs.iter().enumerate() {
            if let Some(beta0) = spec.initial_beta.as_ref() {
                if beta0.len() != offsets[b + 1] - offsets[b] {
                    return Err(SurvivalLocationScaleError::DimensionMismatch {
                        reason: format!(
                            "direct parametric-AFT MLE block {b} initial_beta length {} != block width {}",
                            beta0.len(),
                            offsets[b + 1] - offsets[b]
                        ),
                    }
                    .into());
                }
                theta
                    .slice_mut(s![offsets[b]..offsets[b + 1]])
                    .assign(beta0);
            }
        }

        let mut states = self.parametric_aft_states_from_theta(&theta, specs)?;
        // Resync θ to any constraint projection the state builder applied.
        for (b, state) in states.iter().enumerate() {
            theta
                .slice_mut(s![offsets[b]..offsets[b + 1]])
                .assign(&state.beta);
        }
        let mut ll = self.log_likelihood_only(&states)?;
        if !ll.is_finite() {
            return Err(SurvivalLocationScaleError::NumericalFailure {
                reason: format!(
                    "direct parametric-AFT MLE: non-finite initial log-likelihood {ll}"
                ),
            }
            .into());
        }

        // Newton iterations on −ℓ(θ).
        for _ in 0..max_iter {
            let (ll_now, block_gradients) =
                self.evaluate_log_likelihood_and_block_gradients(&states)?;
            ll = ll_now;
            // Concatenate the block log-likelihood gradients g = ∇ℓ.
            let mut g = Array1::<f64>::zeros(p_total);
            if block_gradients.len() != specs.len() {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "direct parametric-AFT MLE gradient block count mismatch: gradients={}, specs={}",
                        block_gradients.len(),
                        specs.len()
                    ),
                }
                .into());
            }
            for (b, gb) in block_gradients.iter().enumerate() {
                if gb.len() != offsets[b + 1] - offsets[b] {
                    return Err(SurvivalLocationScaleError::DimensionMismatch {
                        reason: format!(
                            "direct parametric-AFT MLE block {b} gradient length {} != block width {}",
                            gb.len(),
                            offsets[b + 1] - offsets[b]
                        ),
                    }
                    .into());
                }
                g.slice_mut(s![offsets[b]..offsets[b + 1]]).assign(gb);
            }
            if !g.iter().all(|v| v.is_finite()) {
                return Err(SurvivalLocationScaleError::NumericalFailure {
                    reason: "direct parametric-AFT MLE: non-finite gradient".to_string(),
                }
                .into());
            }
            // Source `g = ∇ℓ` from the SAME jet-tower row kernel that produces the
            // Newton Hessian `H = −∇²ℓ` below, so the step `H δ = g` is solved on a
            // consistent (objective, gradient, Hessian) triple for EVERY residual
            // distribution. The legacy `evaluate_log_likelihood_and_block_gradients`
            // hand-assembly above happens to coincide with the jet tower for the
            // probit (lognormal) residual but diverges for the logit (log-logistic)
            // residual, yielding a wrong Newton direction that pinned the `age`
            // location coefficient to its cold-start 0 (gam#1110). The kernel
            // gradient has the identical block-concatenated layout (its
            // `jacobian_transpose_action` writes per channel into the same
            // `joint_block_offsets` slabs), so it drops in directly; `ll` is still
            // the scalar log-likelihood from the call above (unchanged).
            if let Some(kernel_g) = self.exact_newton_joint_loglik_gradient(&states)? {
                if kernel_g.len() != p_total {
                    return Err(SurvivalLocationScaleError::DimensionMismatch {
                        reason: format!(
                            "direct parametric-AFT MLE: kernel gradient length {} != p_total {}",
                            kernel_g.len(),
                            p_total
                        ),
                    }
                    .into());
                }
                if !kernel_g.iter().all(|v| v.is_finite()) {
                    return Err(SurvivalLocationScaleError::NumericalFailure {
                        reason: "direct parametric-AFT MLE: non-finite kernel gradient".to_string(),
                    }
                    .into());
                }
                g = kernel_g;
            }
            let grad_norm = g.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
            if grad_norm <= grad_tol {
                break;
            }

            // H = −∇²ℓ (positive (semi)definite near the optimum). The exact
            // joint Hessian assembly returns it directly, symmetrized.
            let h = self.exact_newton_joint_hessian(&states)?.ok_or_else(|| {
                SurvivalLocationScaleError::NumericalFailure {
                    reason: "direct parametric-AFT MLE: joint Hessian assembly failed".to_string(),
                }
            })?;
            if !h.iter().all(|v| v.is_finite()) {
                return Err(SurvivalLocationScaleError::NumericalFailure {
                    reason: "direct parametric-AFT MLE: non-finite joint Hessian".to_string(),
                }
                .into());
            }

            // Newton direction δ solving H δ = g (ascent on ℓ). When H is not
            // positive definite, escalate Levenberg damping τ·I until the
            // Cholesky factorization succeeds, guaranteeing an ascent direction.
            let h_scale = h
                .diag()
                .iter()
                .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
                .max(1.0);
            let mut tau = 0.0_f64;
            let delta = loop {
                let mut damped = h.clone();
                if tau > 0.0 {
                    for i in 0..p_total {
                        damped[[i, i]] += tau;
                    }
                }
                match damped.cholesky(faer::Side::Lower) {
                    Ok(chol) => break chol.solvevec(&g),
                    Err(_) => {
                        tau = if tau == 0.0 {
                            LEVENBERG_INITIAL_DAMPING_REL * h_scale
                        } else {
                            tau * LEVENBERG_DAMPING_GROWTH
                        };
                        if tau > LEVENBERG_MAX_DAMPING_REL * h_scale {
                            return Err(SurvivalLocationScaleError::NumericalFailure {
                                reason:
                                    "direct parametric-AFT MLE: Hessian not factorizable even with maximal damping"
                                        .to_string(),
                            }
                            .into());
                        }
                    }
                }
            };
            if !delta.iter().all(|v| v.is_finite()) {
                return Err(SurvivalLocationScaleError::NumericalFailure {
                    reason: "direct parametric-AFT MLE: non-finite Newton step".to_string(),
                }
                .into());
            }

            // Cap the step to keep the monotone time-warp feasible: the family's
            // per-block feasibility barrier reports the largest α that keeps the
            // derivative guard satisfied (only the time block constrains it).
            let mut alpha = 1.0_f64;
            for (b, spec_offset) in offsets.iter().take(specs.len()).enumerate() {
                let block_delta = delta.slice(s![*spec_offset..offsets[b + 1]]).to_owned();
                if let Some(a_max) = self.max_feasible_step_size(&states, b, &block_delta)? {
                    alpha = alpha.min(a_max);
                }
            }

            // Armijo backtracking on −ℓ along the (feasibility-capped) Newton
            // ascent direction. `g·δ > 0` because δ is an ascent direction, so a
            // sufficient-increase condition on ℓ is well posed.
            let directional = g.dot(&delta);
            const ARMIJO_C: f64 = 1e-4;
            const BACKTRACK: f64 = 0.5;
            const MIN_ALPHA: f64 = 1e-12;
            let mut accepted: Option<(Array1<f64>, Vec<ParameterBlockState>, f64)> = None;
            while alpha >= MIN_ALPHA {
                let trial_theta = &theta + &(alpha * &delta);
                if let Ok(cand_states) = self.parametric_aft_states_from_theta(&trial_theta, specs)
                    && let Ok(cand_ll) = self.log_likelihood_only(&cand_states)
                    && cand_ll.is_finite()
                    && cand_ll >= ll + ARMIJO_C * alpha * directional
                {
                    accepted = Some((trial_theta, cand_states, cand_ll));
                    break;
                }
                alpha *= BACKTRACK;
            }
            match accepted {
                Some((new_theta, new_states, new_ll)) => {
                    theta = new_theta;
                    states = new_states;
                    ll = new_ll;
                }
                // No further increase achievable along this direction: the
                // current iterate is (numerically) stationary. Stop here.
                None => break,
            }
        }

        // Observed information at the MLE: the joint negative-log-likelihood
        // Hessian. This is the conditional precision; its inverse is the
        // covariance the caller lifts to the raw coordinate system.
        let h_final = self.exact_newton_joint_hessian(&states)?.ok_or_else(|| {
            SurvivalLocationScaleError::NumericalFailure {
                reason: "direct parametric-AFT MLE: final joint Hessian assembly failed"
                    .to_string(),
            }
        })?;
        if !h_final.iter().all(|v| v.is_finite()) {
            return Err(SurvivalLocationScaleError::NumericalFailure {
                reason: "direct parametric-AFT MLE: non-finite final joint Hessian".to_string(),
            }
            .into());
        }
        Ok((states, ll, h_final))
    }

    /// Block-diagonal-only assembly: returns the four (or three, when no
    /// link-wiggle is configured) principal diagonal blocks of the joint
    /// Hessian without ever materializing the cross blocks. Used by
    /// `evaluate()` so the inner solver gets per-block working sets at
    /// Θ(n · Σ p_b²) instead of Θ(n · (Σ p_b)²) — for large scale
    /// (n ≈ 3·10⁵, Σ p_b ≈ 200) this avoids ~12·10⁹ scalar multiplies and
    /// the corresponding p² dense allocation per evaluate.
    fn assemble_block_diagonal_hessians_from_quantities(
        &self,
        q: &SurvivalJointQuantities,
        block_states: &[ParameterBlockState],
    ) -> Result<Vec<Array2<f64>>, String> {
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let x_threshold_exit_cow = self.x_threshold.to_dense_cow();
        let x_threshold_exit = &*x_threshold_exit_cow;
        let x_threshold_entry_cow = self
            .x_threshold_entry
            .as_ref()
            .map(DesignMatrix::to_dense_cow);
        let x_threshold_entry = x_threshold_entry_cow
            .as_ref()
            .map_or(x_threshold_exit, |c| &**c);
        let x_threshold_deriv_cow = self
            .x_threshold_deriv
            .as_ref()
            .map(DesignMatrix::to_dense_cow);
        let x_threshold_deriv = x_threshold_deriv_cow.as_deref();
        let x_log_sigma_exit_cow = self.x_log_sigma.to_dense_cow();
        let x_log_sigma_exit = &*x_log_sigma_exit_cow;
        let x_log_sigma_entry_cow = self
            .x_log_sigma_entry
            .as_ref()
            .map(DesignMatrix::to_dense_cow);
        let x_log_sigma_entry = x_log_sigma_entry_cow
            .as_ref()
            .map_or(x_log_sigma_exit, |c| &**c);
        let x_log_sigma_deriv_cow = self
            .x_log_sigma_deriv
            .as_ref()
            .map(DesignMatrix::to_dense_cow);
        let x_log_sigma_deriv = x_log_sigma_deriv_cow.as_deref();

        let use_outer_parallel = rayon::current_num_threads() > 1;
        // When multiple independent Hessian blocks are assembled by Rayon tasks,
        // keep each faer GEMM/GEMV sequential.  This trades inner parallelism for
        // coarse block-level parallelism and avoids nested Rayon/faer
        // oversubscription on the same worker pool.
        let product_parallelism = if use_outer_parallel {
            faer::Par::Seq
        } else {
            faer::get_global_parallelism()
        };

        let assemble_h_time = || -> Result<Array2<f64>, String> {
            // Time-time block (mirrors line 5846-5849 in the joint assembly).
            Ok(safe_fast_xt_diag_x_with_parallelism(
                &dynamic.time_jac_entry,
                &(-&q.h_time_h0),
                product_parallelism,
            ) + safe_fast_xt_diag_x_with_parallelism(
                &dynamic.time_jac_exit,
                &(-&q.h_time_h1),
                product_parallelism,
            ) + safe_fast_xt_diag_x_with_parallelism(
                &dynamic.time_jac_deriv,
                &q.h_time_d,
                product_parallelism,
            ))
        };

        let assemble_h_tt = || -> Result<Array2<f64>, String> {
            // Threshold-threshold block.
            if let Some(x_t_deriv) = x_threshold_deriv {
                let h_exit = -(&q.d2_q1 * &q.dq_t.mapv(|v| safe_product(v, v))
                    + &q.d2_qdot1 * &q.dqdot_t.mapv(|v| safe_product(v, v))
                    + &q.d1_qdot1 * &q.d2qdot_tt);
                let h_entry =
                    -(&q.d2_q0 * &q.dq_t_entry.as_ref().unwrap().mapv(|v| safe_product(v, v)));
                let h_deriv = -(&q.d2_qdot1 * &q.dqdot_td.mapv(|v| safe_product(v, v)));
                let h_exit_deriv =
                    -(&q.d2_qdot1 * &(&q.dqdot_t * &q.dqdot_td) + &q.d1_qdot1 * &q.d2qdot_ttd);
                let mut h_tt = weighted_crossprod_dense_with_parallelism(
                    x_threshold_exit,
                    &h_exit,
                    x_threshold_exit,
                    product_parallelism,
                )? + weighted_crossprod_dense_with_parallelism(
                    x_threshold_entry,
                    &h_entry,
                    x_threshold_entry,
                    product_parallelism,
                )? + weighted_crossprod_dense_with_parallelism(
                    x_t_deriv,
                    &h_deriv,
                    x_t_deriv,
                    product_parallelism,
                )?;
                let cross = weighted_crossprod_dense_with_parallelism(
                    x_threshold_exit,
                    &h_exit_deriv,
                    x_t_deriv,
                    product_parallelism,
                )?;
                h_tt += &cross;
                h_tt += &cross.t().to_owned();
                Ok(h_tt)
            } else {
                let h_t = -(&q.d2_q1 * &q.dq_t.mapv(|v| safe_product(v, v))
                    + &q.d2_q0 * &q.dq_t_entry.as_ref().unwrap().mapv(|v| safe_product(v, v))
                    + &q.d2_qdot1 * &q.dqdot_t.mapv(|v| safe_product(v, v))
                    + &q.d1_qdot1 * &q.d2qdot_tt);
                weighted_crossprod_dense_with_parallelism(
                    x_threshold_exit,
                    &h_t,
                    x_threshold_exit,
                    product_parallelism,
                )
            }
        };

        let assemble_h_ll = || -> Result<Array2<f64>, String> {
            // Log-sigma–log-sigma block.
            if let Some(x_ls_deriv) = x_log_sigma_deriv {
                let dq_ls_entry = q.dq_ls_entry.as_ref().unwrap();
                let d2q_ls_entry = q.d2q_ls_entry.as_ref().unwrap();
                let h_exit = -(&q.d2_q1 * &q.dq_ls.mapv(|v| safe_product(v, v))
                    + &(&q.d1_q1 * &q.d2q_ls)
                    + &q.d2_qdot1 * &q.dqdot_ls.mapv(|v| safe_product(v, v))
                    + &(&q.d1_qdot1 * &q.d2qdot_ls));
                let h_entry = -(&q.d2_q0 * &dq_ls_entry.mapv(|v| safe_product(v, v))
                    + &(&q.d1_q0 * d2q_ls_entry));
                let h_deriv = -(&q.d2_qdot1 * &q.dqdot_lsd.mapv(|v| safe_product(v, v)));
                let h_exit_deriv =
                    -(&q.d2_qdot1 * &(&q.dqdot_ls * &q.dqdot_lsd) + &q.d1_qdot1 * &q.d2qdot_lslsd);
                let mut h_ll = weighted_crossprod_dense_with_parallelism(
                    x_log_sigma_exit,
                    &h_exit,
                    x_log_sigma_exit,
                    product_parallelism,
                )? + weighted_crossprod_dense_with_parallelism(
                    x_log_sigma_entry,
                    &h_entry,
                    x_log_sigma_entry,
                    product_parallelism,
                )? + weighted_crossprod_dense_with_parallelism(
                    x_ls_deriv,
                    &h_deriv,
                    x_ls_deriv,
                    product_parallelism,
                )?;
                let cross = weighted_crossprod_dense_with_parallelism(
                    x_log_sigma_exit,
                    &h_exit_deriv,
                    x_ls_deriv,
                    product_parallelism,
                )?;
                h_ll += &cross;
                h_ll += &cross.t().to_owned();
                Ok(h_ll)
            } else {
                let h_ls = -(&q.d2_q1 * &q.dq_ls.mapv(|v| safe_product(v, v))
                    + &(&q.d1_q1 * &q.d2q_ls)
                    + &q.d2_q0 * &q.dq_ls_entry.as_ref().unwrap().mapv(|v| safe_product(v, v))
                    + &(&q.d1_q0 * q.d2q_ls_entry.as_ref().unwrap())
                    + &q.d2_qdot1 * &q.dqdot_ls.mapv(|v| safe_product(v, v))
                    + &(&q.d1_qdot1 * &q.d2qdot_ls));
                weighted_crossprod_dense_with_parallelism(
                    x_log_sigma_exit,
                    &h_ls,
                    x_log_sigma_exit,
                    product_parallelism,
                )
            }
        };

        let assemble_h_wiggle = || -> Result<Option<Array2<f64>>, String> {
            // Optional link-wiggle block.
            if let (Some(xw_exit), Some(xw_entry), Some(xw_qdot)) = (
                dynamic.wiggle_basis_exit.as_ref(),
                dynamic.wiggle_basis_entry.as_ref(),
                dynamic.wiggle_qdot_basis_exit.as_ref(),
            ) {
                Ok(Some(
                    weighted_crossprod_dense_with_parallelism(
                        xw_exit,
                        &(-&q.d2_q1),
                        xw_exit,
                        product_parallelism,
                    )? + weighted_crossprod_dense_with_parallelism(
                        xw_entry,
                        &(-&q.d2_q0),
                        xw_entry,
                        product_parallelism,
                    )? + weighted_crossprod_dense_with_parallelism(
                        xw_qdot,
                        &(-&q.d2_qdot1),
                        xw_qdot,
                        product_parallelism,
                    )?,
                ))
            } else {
                Ok(None)
            }
        };

        let (h_time, h_tt, h_ll, h_wiggle) = if use_outer_parallel {
            let ((h_time, h_tt), (h_ll, h_wiggle)) = rayon::join(
                || rayon::join(assemble_h_time, assemble_h_tt),
                || rayon::join(assemble_h_ll, assemble_h_wiggle),
            );
            (h_time?, h_tt?, h_ll?, h_wiggle?)
        } else {
            (
                assemble_h_time()?,
                assemble_h_tt()?,
                assemble_h_ll()?,
                assemble_h_wiggle()?,
            )
        };

        let mut blocks = vec![h_time, h_tt, h_ll];
        if let Some(hww) = h_wiggle {
            blocks.push(hww);
        }

        Ok(blocks)
    }

    fn assemble_joint_hessian_from_quantities(
        &self,
        q: &SurvivalJointQuantities,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.assemble_joint_hessian_from_quantities_masked(q, block_states, None)
    }

    /// HT-mask-aware variant of [`assemble_joint_hessian_from_quantities`].
    ///
    /// When `row_mask` is `None`, the function is byte-identical to the
    /// pre-refactor implementation (every weight argument is unchanged).
    /// When `row_mask` is `Some(m)`, every row-additive assembly site
    /// replaces the per-row weight `w[i]` with `w[i] * m[i]`. This is the
    /// outer-score Horvitz-Thompson subsample plumbing
    /// (WS4a-survival-LS): every survival-LS assembly site is of the form
    /// `Σ_i x_i y_iᵀ · w_i` (row-additive), so per-row mask multiplication
    /// is unbiased for `Σ_i w_i · x_i y_iᵀ` under HT weighting.
    fn assemble_joint_hessian_from_quantities_masked(
        &self,
        q: &SurvivalJointQuantities,
        block_states: &[ParameterBlockState],
        row_mask: Option<&Array1<f64>>,
    ) -> Result<Option<Array2<f64>>, String> {
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let joint_states = self.validate_joint_states(block_states)?;
        let eta_t_exit = joint_states.3;
        let eta_t_entry = joint_states.5;
        let eta_t_deriv_exit = joint_states.7;
        let eta_ls_deriv_exit = joint_states.8;
        let eta_t_deriv_exit = eta_t_deriv_exit
            .map(|v| v.to_owned())
            .unwrap_or_else(|| Array1::zeros(self.n));
        let eta_ls_deriv_exit = eta_ls_deriv_exit
            .map(|v| v.to_owned())
            .unwrap_or_else(|| Array1::zeros(self.n));
        let offsets = self.joint_block_offsets();
        let p_total = *offsets
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        let x_threshold_exit_cow = self.x_threshold.to_dense_cow();
        let x_threshold_exit = &*x_threshold_exit_cow;
        let x_threshold_entry_cow = self
            .x_threshold_entry
            .as_ref()
            .map(DesignMatrix::to_dense_cow);
        let x_threshold_entry = x_threshold_entry_cow
            .as_ref()
            .map_or(x_threshold_exit, |c| &**c);
        let x_threshold_deriv_cow = self
            .x_threshold_deriv
            .as_ref()
            .map(DesignMatrix::to_dense_cow);
        let x_threshold_deriv = x_threshold_deriv_cow.as_deref();
        let x_log_sigma_exit_cow = self.x_log_sigma.to_dense_cow();
        let x_log_sigma_exit = &*x_log_sigma_exit_cow;
        let x_log_sigma_entry_cow = self
            .x_log_sigma_entry
            .as_ref()
            .map(DesignMatrix::to_dense_cow);
        let x_log_sigma_entry = x_log_sigma_entry_cow
            .as_ref()
            .map_or(x_log_sigma_exit, |c| &**c);
        let x_log_sigma_deriv_cow = self
            .x_log_sigma_deriv
            .as_ref()
            .map(DesignMatrix::to_dense_cow);
        let x_log_sigma_deriv = x_log_sigma_deriv_cow.as_deref();
        let mut joint = Array2::<f64>::zeros((p_total, p_total));
        let add_cross = |acc: &mut Array2<f64>,
                         left: &Array2<f64>,
                         weights: &Array1<f64>,
                         right: &Array2<f64>|
         -> Result<(), String> {
            *acc += &mxtwx(left, weights, right, row_mask)?;
            Ok(())
        };

        let h_time = mxtwxd(&dynamic.time_jac_entry, &(-&q.h_time_h0), row_mask)
            + mxtwxd(&dynamic.time_jac_exit, &(-&q.h_time_h1), row_mask)
            + mxtwxd(&dynamic.time_jac_deriv, &q.h_time_d, row_mask);
        assign_symmetric_block(&mut joint, offsets[0], offsets[0], &h_time);

        if let Some(x_t_deriv) = x_threshold_deriv {
            let h_exit = -(&q.d2_q1 * &q.dq_t.mapv(|v| safe_product(v, v))
                + &q.d2_qdot1 * &q.dqdot_t.mapv(|v| safe_product(v, v))
                + &q.d1_qdot1 * &q.d2qdot_tt);
            let h_entry =
                -(&q.d2_q0 * &q.dq_t_entry.as_ref().unwrap().mapv(|v| safe_product(v, v)));
            let h_deriv = -(&q.d2_qdot1 * &q.dqdot_td.mapv(|v| safe_product(v, v)));
            let h_exit_deriv =
                -(&q.d2_qdot1 * &(&q.dqdot_t * &q.dqdot_td) + &q.d1_qdot1 * &q.d2qdot_ttd);
            let mut h_tt = mxtwx(x_threshold_exit, &h_exit, x_threshold_exit, row_mask)?
                + mxtwx(x_threshold_entry, &h_entry, x_threshold_entry, row_mask)?
                + mxtwx(x_t_deriv, &h_deriv, x_t_deriv, row_mask)?;
            let cross = mxtwx(x_threshold_exit, &h_exit_deriv, x_t_deriv, row_mask)?;
            h_tt += &cross;
            h_tt += &cross.t().to_owned();
            assign_symmetric_block(&mut joint, offsets[1], offsets[1], &h_tt);
        } else {
            let h_t = -(&q.d2_q1 * &q.dq_t.mapv(|v| safe_product(v, v))
                + &q.d2_q0 * &q.dq_t_entry.as_ref().unwrap().mapv(|v| safe_product(v, v))
                + &q.d2_qdot1 * &q.dqdot_t.mapv(|v| safe_product(v, v))
                + &q.d1_qdot1 * &q.d2qdot_tt);
            let h_tt = mxtwx(x_threshold_exit, &h_t, x_threshold_exit, row_mask)?;
            assign_symmetric_block(&mut joint, offsets[1], offsets[1], &h_tt);
        }

        if let Some(x_ls_deriv) = x_log_sigma_deriv {
            let dq_ls_entry = q.dq_ls_entry.as_ref().unwrap();
            let d2q_ls_entry = q.d2q_ls_entry.as_ref().unwrap();
            let h_exit = -(&q.d2_q1 * &q.dq_ls.mapv(|v| safe_product(v, v))
                + &(&q.d1_q1 * &q.d2q_ls)
                + &q.d2_qdot1 * &q.dqdot_ls.mapv(|v| safe_product(v, v))
                + &(&q.d1_qdot1 * &q.d2qdot_ls));
            let h_entry = -(&q.d2_q0 * &dq_ls_entry.mapv(|v| safe_product(v, v))
                + &(&q.d1_q0 * d2q_ls_entry));
            let h_deriv = -(&q.d2_qdot1 * &q.dqdot_lsd.mapv(|v| safe_product(v, v)));
            let h_exit_deriv =
                -(&q.d2_qdot1 * &(&q.dqdot_ls * &q.dqdot_lsd) + &q.d1_qdot1 * &q.d2qdot_lslsd);
            let mut h_ll = mxtwx(x_log_sigma_exit, &h_exit, x_log_sigma_exit, row_mask)?
                + mxtwx(x_log_sigma_entry, &h_entry, x_log_sigma_entry, row_mask)?
                + mxtwx(x_ls_deriv, &h_deriv, x_ls_deriv, row_mask)?;
            let cross = mxtwx(x_log_sigma_exit, &h_exit_deriv, x_ls_deriv, row_mask)?;
            h_ll += &cross;
            h_ll += &cross.t().to_owned();
            assign_symmetric_block(&mut joint, offsets[2], offsets[2], &h_ll);
        } else {
            let h_ls = -(&q.d2_q1 * &q.dq_ls.mapv(|v| safe_product(v, v))
                + &(&q.d1_q1 * &q.d2q_ls)
                + &q.d2_q0 * &q.dq_ls_entry.as_ref().unwrap().mapv(|v| safe_product(v, v))
                + &(&q.d1_q0 * q.d2q_ls_entry.as_ref().unwrap())
                + &q.d2_qdot1 * &q.dqdot_ls.mapv(|v| safe_product(v, v))
                + &(&q.d1_qdot1 * &q.d2qdot_ls));
            let h_ll = mxtwx(x_log_sigma_exit, &h_ls, x_log_sigma_exit, row_mask)?;
            assign_symmetric_block(&mut joint, offsets[2], offsets[2], &h_ll);
        }

        {
            let mut h_tl = Array2::<f64>::zeros((offsets[2] - offsets[1], offsets[3] - offsets[2]));
            let w_exit = -(&q.d2_q1 * &(&q.dq_t * &q.dq_ls) + &(&q.d1_q1 * &q.d2q_tls));
            let w_entry = -(&q.d2_q0
                * &(q.dq_t_entry.as_ref().unwrap() * q.dq_ls_entry.as_ref().unwrap())
                + &(&q.d1_q0 * q.d2q_tls_entry.as_ref().unwrap()));
            add_cross(&mut h_tl, x_threshold_exit, &w_exit, x_log_sigma_exit)?;
            add_cross(&mut h_tl, x_threshold_entry, &w_entry, x_log_sigma_entry)?;
            let w_qdot_exit =
                -(&q.d2_qdot1 * &(&q.dqdot_t * &q.dqdot_ls) + &(&q.d1_qdot1 * &q.d2qdot_tls));
            add_cross(&mut h_tl, x_threshold_exit, &w_qdot_exit, x_log_sigma_exit)?;
            if let Some(x_ls_deriv) = x_log_sigma_deriv {
                let w =
                    -(&q.d2_qdot1 * &(&q.dqdot_t * &q.dqdot_lsd) + &(&q.d1_qdot1 * &q.d2qdot_tlsd));
                add_cross(&mut h_tl, x_threshold_exit, &w, x_ls_deriv)?;
            }
            if let Some(x_t_deriv) = x_threshold_deriv {
                let w =
                    -(&q.d2_qdot1 * &(&q.dqdot_td * &q.dqdot_ls) + &(&q.d1_qdot1 * &q.d2qdot_lstd));
                add_cross(&mut h_tl, x_t_deriv, &w, x_log_sigma_exit)?;
                if let Some(x_ls_deriv) = x_log_sigma_deriv {
                    let wdd = -(&q.d2_qdot1 * &(&q.dqdot_td * &q.dqdot_lsd));
                    add_cross(&mut h_tl, x_t_deriv, &wdd, x_ls_deriv)?;
                }
            }
            assign_symmetric_block(&mut joint, offsets[1], offsets[2], &h_tl);
        }

        let mut h_ht = mxtwx(
            &self.x_time_entry,
            &(-&q.h_time_h0 * q.dq_t_entry.as_ref().unwrap()),
            x_threshold_entry,
            row_mask,
        )? + mxtwx(
            &self.x_time_exit,
            &(-&q.h_time_h1 * &q.dq_t),
            x_threshold_exit,
            row_mask,
        )? + mxtwx(
            &self.x_time_deriv,
            &(&q.h_time_d * &q.dqdot_t),
            x_threshold_exit,
            row_mask,
        )?;
        if let Some(x_t_deriv) = x_threshold_deriv {
            h_ht += &mxtwx(
                &self.x_time_deriv,
                &(&q.h_time_d * &q.dqdot_td),
                x_t_deriv,
                row_mask,
            )?;
        }
        assign_symmetric_block(&mut joint, offsets[0], offsets[1], &h_ht);

        let mut h_hl = mxtwx(
            &self.x_time_entry,
            &(-&q.h_time_h0 * q.dq_ls_entry.as_ref().unwrap()),
            x_log_sigma_entry,
            row_mask,
        )? + mxtwx(
            &self.x_time_exit,
            &(-&q.h_time_h1 * &q.dq_ls),
            x_log_sigma_exit,
            row_mask,
        )? + mxtwx(
            &self.x_time_deriv,
            &(&q.h_time_d * &q.dqdot_ls),
            x_log_sigma_exit,
            row_mask,
        )?;
        if let Some(x_ls_deriv) = x_log_sigma_deriv {
            h_hl += &mxtwx(
                &self.x_time_deriv,
                &(&q.h_time_d * &q.dqdot_lsd),
                x_ls_deriv,
                row_mask,
            )?;
        }
        assign_symmetric_block(&mut joint, offsets[0], offsets[2], &h_hl);

        if let (
            Some(xw_exit),
            Some(xw_entry),
            Some(xw_qdot),
            Some(xw_d1_exit),
            Some(xw_d1_entry),
            Some(xw_d2_exit),
            Some(w_offset),
        ) = (
            dynamic.wiggle_basis_exit.as_ref(),
            dynamic.wiggle_basis_entry.as_ref(),
            dynamic.wiggle_qdot_basis_exit.as_ref(),
            dynamic.wiggle_basis_d1_exit.as_ref(),
            dynamic.wiggle_basis_d1_entry.as_ref(),
            dynamic.wiggle_basis_d2_exit.as_ref(),
            offsets.get(3).copied(),
        ) {
            let hww = mxtwx(xw_exit, &(-&q.d2_q1), xw_exit, row_mask)?
                + mxtwx(xw_entry, &(-&q.d2_q0), xw_entry, row_mask)?
                + mxtwx(xw_qdot, &(-&q.d2_qdot1), xw_qdot, row_mask)?;
            assign_symmetric_block(&mut joint, w_offset, w_offset, &hww);
            let q0_t_entry = Array1::from_iter(dynamic.inv_sigma_entry.iter().map(|&r| -r));
            let q0_t_exit = Array1::from_iter(dynamic.inv_sigma_exit.iter().map(|&r| -r));
            let q0_ls_entry = Array1::from_iter(
                (0..self.n)
                    .map(|i| q_chain_derivs_scalar(eta_t_entry[i], dynamic.eta_ls_entry[i]).1),
            );
            let q0_ls_exit = Array1::from_iter(
                (0..self.n).map(|i| q_chain_derivs_scalar(eta_t_exit[i], dynamic.eta_ls_exit[i]).1),
            );
            let r_base_exit = safe_linear_combo2_arrays(
                &q0_t_exit,
                &eta_t_deriv_exit,
                &q0_ls_exit,
                &eta_ls_deriv_exit,
            )?;
            let r_t_base_exit = Array1::from_iter((0..self.n).map(|i| {
                safe_product(
                    q_chain_derivs_scalar(eta_t_exit[i], dynamic.eta_ls_exit[i]).2,
                    eta_ls_deriv_exit[i],
                )
            }));
            let r_ls_base_exit = Array1::from_iter((0..self.n).map(|i| {
                let (_, _, q_tl, q_ll, _, _) =
                    q_chain_derivs_scalar(eta_t_exit[i], dynamic.eta_ls_exit[i]);
                safe_sum2(
                    safe_product(q_tl, eta_t_deriv_exit[i]),
                    safe_product(q_ll, eta_ls_deriv_exit[i]),
                )
            }));
            let tw_entry_d2 = scale_dense_rows(xw_d1_entry, &q0_t_entry)?;
            let tw_exit_d2 = scale_dense_rows(xw_d1_exit, &q0_t_exit)?;
            let lw_entry_d2 = scale_dense_rows(xw_d1_entry, &q0_ls_entry)?;
            let lw_exit_d2 = scale_dense_rows(xw_d1_exit, &q0_ls_exit)?;
            let qdot_t_w = scale_dense_rows(
                xw_d2_exit,
                &safe_hadamard_product(&q0_t_exit, &r_base_exit)?,
            )? + scale_dense_rows(xw_d1_exit, &r_t_base_exit)?;
            let qdot_ls_w = scale_dense_rows(
                xw_d2_exit,
                &safe_hadamard_product(&q0_ls_exit, &r_base_exit)?,
            )? + scale_dense_rows(xw_d1_exit, &r_ls_base_exit)?;
            let qdot_td_w = scale_dense_rows(xw_d1_exit, &q0_t_exit)?;
            let qdot_lsd_w = scale_dense_rows(xw_d1_exit, &q0_ls_exit)?;

            let mut h_tw = Array2::<f64>::zeros((offsets[2] - offsets[1], offsets[4] - offsets[3]));
            h_tw += &mxtwx(x_threshold_exit, &(-&q.d2_q1 * &q.dq_t), xw_exit, row_mask)?;
            h_tw += &mxtwx(
                x_threshold_exit,
                &(-&q.d1_q1 * &q0_t_exit),
                &tw_exit_d2,
                row_mask,
            )?;
            h_tw += &mxtwx(
                x_threshold_entry,
                &(-&q.d2_q0 * q.dq_t_entry.as_ref().unwrap()),
                xw_entry,
                row_mask,
            )?;
            h_tw += &mxtwx(
                x_threshold_entry,
                &(-&q.d1_q0 * &q0_t_entry),
                &tw_entry_d2,
                row_mask,
            )?;
            h_tw += &mxtwx(
                x_threshold_exit,
                &(-&q.d2_qdot1 * &q.dqdot_t),
                xw_qdot,
                row_mask,
            )?;
            h_tw += &mxtwx(x_threshold_exit, &(-&q.d1_qdot1), &qdot_t_w, row_mask)?;
            if let Some(x_t_deriv) = x_threshold_deriv {
                h_tw += &mxtwx(x_t_deriv, &(-&q.d2_qdot1 * &q.dqdot_td), xw_qdot, row_mask)?;
                h_tw += &mxtwx(x_t_deriv, &(-&q.d1_qdot1), &qdot_td_w, row_mask)?;
            }
            assign_symmetric_block(&mut joint, offsets[1], w_offset, &h_tw);

            let mut h_lw = Array2::<f64>::zeros((offsets[3] - offsets[2], offsets[4] - offsets[3]));
            h_lw += &mxtwx(x_log_sigma_exit, &(-&q.d2_q1 * &q.dq_ls), xw_exit, row_mask)?;
            h_lw += &mxtwx(
                x_log_sigma_exit,
                &(-(&q.d1_q1 * &q0_ls_exit)),
                &lw_exit_d2,
                row_mask,
            )?;
            h_lw += &mxtwx(
                x_log_sigma_entry,
                &(-&q.d2_q0 * q.dq_ls_entry.as_ref().unwrap()),
                xw_entry,
                row_mask,
            )?;
            h_lw += &mxtwx(
                x_log_sigma_entry,
                &(-(&q.d1_q0 * &q0_ls_entry)),
                &lw_entry_d2,
                row_mask,
            )?;
            h_lw += &mxtwx(
                x_log_sigma_exit,
                &(-&q.d2_qdot1 * &q.dqdot_ls),
                xw_qdot,
                row_mask,
            )?;
            h_lw += &mxtwx(x_log_sigma_exit, &(-&q.d1_qdot1), &qdot_ls_w, row_mask)?;
            if let Some(x_ls_deriv) = x_log_sigma_deriv {
                h_lw += &mxtwx(
                    x_ls_deriv,
                    &(-&q.d2_qdot1 * &q.dqdot_lsd),
                    xw_qdot,
                    row_mask,
                )?;
                h_lw += &mxtwx(x_ls_deriv, &(-&q.d1_qdot1), &qdot_lsd_w, row_mask)?;
            }
            assign_symmetric_block(&mut joint, offsets[2], w_offset, &h_lw);

            let h_hw = mxtwx(&self.x_time_entry, &(-&q.h_time_h0), xw_entry, row_mask)?
                + mxtwx(&self.x_time_exit, &(-&q.h_time_h1), xw_exit, row_mask)?
                + mxtwx(&self.x_time_deriv, &q.h_time_d, xw_qdot, row_mask)?;
            assign_symmetric_block(&mut joint, offsets[0], w_offset, &h_hw);
        }

        Ok(Some(joint))
    }

    /// Compute the log-scale shift needed to keep CLogLog survival
    /// derivatives finite.  Returns `L >= 0` such that `exp(u - L) <= exp(500)`
    /// for all row linear predictors `u`.  For non-CLogLog links, returns 0.
    fn hessian_deriv_log_rescale(&self, block_states: &[ParameterBlockState]) -> f64 {
        if !matches!(
            self.inverse_link,
            InverseLink::Standard(StandardLink::CLogLog)
        ) {
            return 0.0;
        }
        let dynamic = match self.build_dynamic_geometry(block_states) {
            Ok(d) => d,
            Err(_) => return 0.0,
        };
        let mut max_u = f64::NEG_INFINITY;
        for i in 0..self.n {
            if self.w[i] <= 0.0 {
                continue;
            }
            let u0 = dynamic.h_entry[i] + dynamic.q_entry[i];
            let u1 = dynamic.h_exit[i] + dynamic.q_exit[i];
            max_u = max_u.max(u0).max(u1);
        }
        // Shift so the largest exp(u - L) ~ exp(500), well within f64 range.
        (max_u - 500.0).max(0.0)
    }

    /// Rescaled joint Hessian for logdet computation.  Returns
    /// `(H_scaled, L)` where `H_scaled = exp(-L) * H_exact` and
    /// `logdet(H_exact) = logdet(H_scaled) + p * L`.
    fn exact_newton_joint_hessian_rescaled(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<(Array2<f64>, f64)>, String> {
        let log_scale = self.hessian_deriv_log_rescale(block_states);
        if log_scale == 0.0 {
            return Ok(self
                .exact_newton_joint_hessian(block_states)?
                .map(|h| (h, 0.0)));
        }
        let q = self.collect_joint_quantities_rescaled(block_states, log_scale)?;
        if self.row_kernel_joint_hessian_supported() {
            let dynamic = self.build_dynamic_geometry(block_states)?;
            let kernel = self.survival_ls_row_kernel(&q, &dynamic);
            let rows = crate::families::row_kernel::RowSet::All;
            let cache = crate::families::row_kernel::build_row_kernel_cache(&kernel, &rows)?;
            return Ok(Some((
                crate::families::row_kernel::row_kernel_hessian_dense(&kernel, &cache, &rows),
                log_scale,
            )));
        }
        Ok(self
            .assemble_joint_hessian_from_quantities(&q, block_states)?
            .map(|h| (h, log_scale)))
    }

    fn exact_newton_joint_hessian_directional_derivative_rescaled(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        log_rescale: f64,
    ) -> Result<Option<Array2<f64>>, String> {
        let q = self.collect_joint_quantities_rescaled(block_states, log_rescale)?;
        let dynamic = self.build_dynamic_geometry(block_states)?;
        self.exact_newton_joint_hessian_directional_derivative_rescaled_from_parts(
            d_beta_flat,
            &q,
            &dynamic,
        )
    }

    /// `_from_parts` variant of
    /// [`Self::exact_newton_joint_hessian_directional_derivative_rescaled`]
    /// that receives the precomputed `q` and `dynamic` instead of recomputing
    /// them on every call. This is the workspace-friendly entry point used by
    /// `SurvivalLocationScaleExactNewtonJointHessianWorkspace` to avoid the
    /// ~300 redundant `collect_joint_quantities_rescaled` /
    /// `build_dynamic_geometry` sweeps the outer Hessian pair loop would
    /// otherwise trigger per evaluation.
    fn exact_newton_joint_hessian_directional_derivative_rescaled_from_parts(
        &self,
        d_beta_flat: &Array1<f64>,
        q: &SurvivalJointQuantities,
        dynamic: &SurvivalDynamicGeometry,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_rescaled_from_parts_masked(
            d_beta_flat,
            q,
            dynamic,
            None,
        )
    }

    /// HT-mask-aware variant of
    /// [`Self::exact_newton_joint_hessian_directional_derivative_rescaled_from_parts`].
    /// `None` is byte-identical to the pre-refactor expression at every site.
    /// See [`Self::assemble_joint_hessian_from_quantities_masked`] for the
    /// row-additivity argument.
    fn exact_newton_joint_hessian_directional_derivative_rescaled_from_parts_masked(
        &self,
        d_beta_flat: &Array1<f64>,
        q: &SurvivalJointQuantities,
        dynamic: &SurvivalDynamicGeometry,
        row_mask: Option<&Array1<f64>>,
    ) -> Result<Option<Array2<f64>>, String> {
        let offsets = self.joint_block_offsets();
        let p_total = *offsets
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        if d_beta_flat.len() != p_total {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "joint d_beta length mismatch: got {}, expected {p_total}",
                    d_beta_flat.len()
                ),
            }
            .into());
        }

        if self.row_kernel_directional_supported() {
            let kernel = self.survival_ls_row_kernel(q, dynamic);
            let rows = row_set_from_survival_mask(row_mask, self.n);
            return crate::families::row_kernel::row_kernel_directional_derivative(
                &kernel,
                &rows,
                d_beta_flat
                    .as_slice()
                    .ok_or_else(|| "joint d_beta must be contiguous".to_string())?,
            )
            .map(Some);
        }

        let time_dir = d_beta_flat.slice(s![offsets[0]..offsets[1]]).to_owned();
        let threshold_dir = d_beta_flat.slice(s![offsets[1]..offsets[2]]).to_owned();
        let log_sigma_dir = d_beta_flat.slice(s![offsets[2]..offsets[3]]).to_owned();
        let wiggle_dir = if self.x_link_wiggle.is_some() {
            Some(d_beta_flat.slice(s![offsets[3]..offsets[4]]).to_owned())
        } else {
            None
        };

        let delta_h0 = dynamic.time_jac_entry.dot(&time_dir);
        let delta_h1 = dynamic.time_jac_exit.dot(&time_dir);
        let delta_d = dynamic.time_jac_deriv.dot(&time_dir);
        let delta_t_exit = self.x_threshold.matrixvectormultiply(&threshold_dir);
        let delta_ls_exit = self.x_log_sigma.matrixvectormultiply(&log_sigma_dir);
        let deltaw = match (self.x_link_wiggle.as_ref(), wiggle_dir.as_ref()) {
            (Some(xw), Some(dir)) => Some(xw.matrixvectormultiply(dir)),
            _ => None,
        };

        let mut delta_q_exit = &q.dq_t * &delta_t_exit + &q.dq_ls * &delta_ls_exit;
        if let Some(dw) = &deltaw {
            delta_q_exit += dw;
        }
        let delta_q_t_exit = &q.d2q_tls * &delta_ls_exit;
        let delta_q_ls_exit = &q.d2q_tls * &delta_t_exit + &q.d2q_ls * &delta_ls_exit;
        let delta_q_tls_exit = &q.d3q_tls_ls * &delta_ls_exit;
        let delta_q_ls_ls_exit = &q.d3q_tls_ls * &delta_t_exit + &q.d3q_ls * &delta_ls_exit;

        let d_d1_q_exit = &q.d2_q1 * &delta_q_exit + &q.h_time_h1 * &delta_h1;
        let d_d2_q_exit = &q.d3_q1 * &delta_q_exit + &q.d_h_h1 * &delta_h1;

        let x_threshold_exit_cow = self.x_threshold.to_dense_cow();
        let x_threshold_exit = &*x_threshold_exit_cow;
        let x_threshold_entry_cow = self
            .x_threshold_entry
            .as_ref()
            .map(DesignMatrix::to_dense_cow);
        let x_threshold_entry = x_threshold_entry_cow.as_deref();
        let x_log_sigma_exit_cow = self.x_log_sigma.to_dense_cow();
        let x_log_sigma_exit = &*x_log_sigma_exit_cow;
        let x_log_sigma_entry_cow = self
            .x_log_sigma_entry
            .as_ref()
            .map(DesignMatrix::to_dense_cow);
        let x_log_sigma_entry = x_log_sigma_entry_cow.as_deref();
        let xw_cow = self.x_link_wiggle.as_ref().map(DesignMatrix::to_dense_cow);
        let xw = xw_cow.as_deref();
        let mut joint = Array2::<f64>::zeros((p_total, p_total));

        struct EntryDeltas {
            delta_q: Array1<f64>,
            delta_q_t: Array1<f64>,
            delta_q_ls: Array1<f64>,
            delta_q_tls: Array1<f64>,
            delta_q_ls_ls: Array1<f64>,
            d_d1_q: Array1<f64>,
            d_d2_q: Array1<f64>,
        }
        let entry_deltas = if x_threshold_entry.is_some() || x_log_sigma_entry.is_some() {
            let dt_en = self
                .x_threshold_entry
                .as_ref()
                .map(|x| x.matrixvectormultiply(&threshold_dir))
                .unwrap_or_else(|| delta_t_exit.clone());
            let dls_en = self
                .x_log_sigma_entry
                .as_ref()
                .map(|x| x.matrixvectormultiply(&log_sigma_dir))
                .unwrap_or_else(|| delta_ls_exit.clone());
            let dq_t_en = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
            let dq_ls_en = q.dq_ls_entry.as_ref().unwrap_or(&q.dq_ls);
            let d2q_tls_en = q.d2q_tls_entry.as_ref().unwrap_or(&q.d2q_tls);
            let d3q_tls_ls_en = q.d3q_tls_ls_entry.as_ref().unwrap_or(&q.d3q_tls_ls);
            let d3q_ls_en = q.d3q_ls_entry.as_ref().unwrap_or(&q.d3q_ls);
            let d2q_ls_en = q.d2q_ls_entry.as_ref().unwrap_or(&q.d2q_ls);
            let mut dq_en = dq_t_en * &dt_en + dq_ls_en * &dls_en;
            if let Some(dw) = &deltaw {
                dq_en += dw;
            }
            EntryDeltas {
                delta_q_t: d2q_tls_en * &dls_en,
                delta_q_ls: d2q_tls_en * &dt_en + d2q_ls_en * &dls_en,
                delta_q_tls: d3q_tls_ls_en * &dls_en,
                delta_q_ls_ls: d3q_tls_ls_en * &dt_en + d3q_ls_en * &dls_en,
                d_d1_q: &q.d2_q0 * &dq_en + &q.h_time_h0 * &delta_h0,
                d_d2_q: &q.d3_q0 * &dq_en + &q.d_h_h0 * &delta_h0,
                delta_q: dq_en,
            }
        } else {
            EntryDeltas {
                delta_q: delta_q_exit.clone(),
                delta_q_t: delta_q_t_exit.clone(),
                delta_q_ls: delta_q_ls_exit.clone(),
                delta_q_tls: delta_q_tls_exit.clone(),
                delta_q_ls_ls: delta_q_ls_ls_exit.clone(),
                d_d1_q: &q.d2_q0 * &delta_q_exit + &q.h_time_h0 * &delta_h0,
                d_d2_q: &q.d3_q0 * &delta_q_exit + &q.d_h_h0 * &delta_h0,
            }
        };

        // Time-time directional derivative of the joint Hessian block.
        //
        // The stored joint "Hessian" H equals -∂²ℓ/∂β². The time-time
        // diagonal base assembly (assemble_joint_hessian_from_quantities)
        // is
        //
        //   H_tt = X_entry'·diag(-h_time_h0)·X_entry
        //        + X_exit' ·diag(-h_time_h1)·X_exit
        //        + X_deriv'·diag(+h_time_d)·X_deriv
        //
        // with h_time_h0 = w·r'(u0)     (= ∂²ℓ/∂h0²),
        //      h_time_h1 = w·event_mix  (= ∂²ℓ/∂h1²),
        //      h_time_d  = -w·d·(d2logg) (= -∂²ℓ/∂d_raw²).
        //
        // Differentiating H_tt along Δβ_t (with Δh0 = X_entry·Δβ_t,
        // Δh1 = X_exit·Δβ_t, Δd = X_deriv·Δβ_t, and q0/q1 invariant in
        // a pure time direction) gives
        //
        //   dH_tt = X_entry'·diag(-w·r''(u0)·Δh0)·X_entry
        //         + X_exit' ·diag(-w·r'''-mixed·Δh1)·X_exit
        //         + X_deriv'·diag(+d_h_d·Δd)·X_deriv
        //         = X_entry'·diag(-d_h_h0·Δh0)·X_entry
        //         + X_exit' ·diag(-d_h_h1·Δh1)·X_exit
        //         + X_deriv'·diag(+d_h_d ·Δd )·X_deriv
        //
        // For non-time directions (Δβ_t = 0) the inner variables Δh0/Δh1/Δd
        // vanish but u0/u1 still shift through q0/q1. Chain rule:
        //   Δu0 = Δh0 + Δq_entry   (for pure time direction: Δq_entry = 0)
        //   Δu1 = Δh1 + Δq_exit
        // so the general form tracks Δu0 = Δh0 + entry_deltas.delta_q and
        // Δu1 = Δh1 + delta_q_exit. (No q-dependence on d_raw ⇒ Δd alone
        // drives the deriv-side weight derivative.)
        let du0 = &delta_h0 + &entry_deltas.delta_q;
        let du1 = &delta_h1 + &delta_q_exit;
        let dh_h0 = &q.d_h_h0 * &du0;
        let dh_h1 = &q.d_h_h1 * &du1;
        let dh_d = &q.d_h_d * &delta_d;
        let d_h_time = mxtwxd(&dynamic.time_jac_entry, &(-&dh_h0), row_mask)
            + mxtwxd(&dynamic.time_jac_exit, &(-&dh_h1), row_mask)
            + mxtwxd(&dynamic.time_jac_deriv, &dh_d, row_mask);
        assign_symmetric_block(&mut joint, offsets[0], offsets[0], &d_h_time);

        if let Some(x_t_en) = x_threshold_entry.as_ref() {
            let dq_t_en = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
            let d_h_exit = -(&d_d2_q_exit * &q.dq_t.mapv(|v| safe_product(v, v))
                + &(&q.d2_q1 * &(2.0 * &delta_q_t_exit * &q.dq_t)));
            let d_h_entry = -(&entry_deltas.d_d2_q * &dq_t_en.mapv(|v| safe_product(v, v))
                + &(&q.d2_q0 * &(2.0 * &entry_deltas.delta_q_t * dq_t_en)));
            let d_h_tt = mxtwx(x_threshold_exit, &d_h_exit, x_threshold_exit, row_mask)?
                + mxtwx(x_t_en, &d_h_entry, x_t_en, row_mask)?;
            assign_symmetric_block(&mut joint, offsets[1], offsets[1], &d_h_tt);
        } else {
            let d_d2_q_ti = &q.d3_q * &delta_q_exit + &q.d_h_h0 * &delta_h0 + &q.d_h_h1 * &delta_h1;
            let d_h_t = -(&d_d2_q_ti * &q.dq_t.mapv(|v| safe_product(v, v))
                + &(&q.d2_q * &(2.0 * &delta_q_t_exit * &q.dq_t)));
            let d_h_tt = mxtwx(x_threshold_exit, &d_h_t, x_threshold_exit, row_mask)?;
            assign_symmetric_block(&mut joint, offsets[1], offsets[1], &d_h_tt);
        }

        {
            let has_t_entry = x_threshold_entry.is_some();
            let has_ls_entry = x_log_sigma_entry.is_some();
            if has_t_entry || has_ls_entry {
                let x_t_en = x_threshold_entry.unwrap_or(x_threshold_exit);
                let x_ls_en = x_log_sigma_entry.unwrap_or(x_log_sigma_exit);
                let dq_t_en = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
                let dq_ls_en = q.dq_ls_entry.as_ref().unwrap_or(&q.dq_ls);
                let d2q_tls_en = q.d2q_tls_entry.as_ref().unwrap_or(&q.d2q_tls);
                let w_exit = -(&d_d2_q_exit * &(&q.dq_t * &q.dq_ls)
                    + &(&q.d2_q1 * &(&delta_q_t_exit * &q.dq_ls + &q.dq_t * &delta_q_ls_exit))
                    + &(&d_d1_q_exit * &q.d2q_tls)
                    + &(&q.d1_q1 * &delta_q_tls_exit));
                let w_entry = -(&entry_deltas.d_d2_q * &(dq_t_en * dq_ls_en)
                    + &(&q.d2_q0
                        * &(&entry_deltas.delta_q_t * dq_ls_en
                            + dq_t_en * &entry_deltas.delta_q_ls))
                    + &(&entry_deltas.d_d1_q * d2q_tls_en)
                    + &(&q.d1_q0 * &entry_deltas.delta_q_tls));
                let d_h_tl = mxtwx(x_threshold_exit, &w_exit, x_log_sigma_exit, row_mask)?
                    + mxtwx(x_t_en, &w_entry, x_ls_en, row_mask)?;
                assign_symmetric_block(&mut joint, offsets[1], offsets[2], &d_h_tl);
            } else {
                let d_d1_q =
                    &q.d2_q * &delta_q_exit + &q.h_time_h0 * &delta_h0 + &q.h_time_h1 * &delta_h1;
                let d_d2_q =
                    &q.d3_q * &delta_q_exit + &q.d_h_h0 * &delta_h0 + &q.d_h_h1 * &delta_h1;
                let d_h_tlweights = -(&d_d2_q * &(&q.dq_t * &q.dq_ls)
                    + &(&q.d2_q * &(&delta_q_t_exit * &q.dq_ls + &q.dq_t * &delta_q_ls_exit))
                    + &(&d_d1_q * &q.d2q_tls)
                    + &(&q.d1_q * &delta_q_tls_exit));
                let d_h_tl = mxtwx(x_threshold_exit, &d_h_tlweights, x_log_sigma_exit, row_mask)?;
                assign_symmetric_block(&mut joint, offsets[1], offsets[2], &d_h_tl);
            }
        }

        if let Some(x_ls_en) = x_log_sigma_entry.as_ref() {
            let dq_ls_en = q.dq_ls_entry.as_ref().unwrap();
            let d2q_ls_en = q.d2q_ls_entry.as_ref().unwrap();
            let d_h_exit = -(&d_d2_q_exit * &q.dq_ls.mapv(|v| safe_product(v, v))
                + &(&q.d2_q1 * &(2.0 * &delta_q_ls_exit * &q.dq_ls))
                + &(&d_d1_q_exit * &q.d2q_ls)
                + &(&q.d1_q1 * &delta_q_ls_ls_exit));
            let d_h_entry = -(&entry_deltas.d_d2_q * &dq_ls_en.mapv(|v| safe_product(v, v))
                + &(&q.d2_q0 * &(2.0 * &entry_deltas.delta_q_ls * dq_ls_en))
                + &(&entry_deltas.d_d1_q * d2q_ls_en)
                + &(&q.d1_q0 * &entry_deltas.delta_q_ls_ls));
            let d_h_ll = mxtwx(x_log_sigma_exit, &d_h_exit, x_log_sigma_exit, row_mask)?
                + mxtwx(x_ls_en, &d_h_entry, x_ls_en, row_mask)?;
            assign_symmetric_block(&mut joint, offsets[2], offsets[2], &d_h_ll);
        } else {
            let d_d1_q =
                &q.d2_q * &delta_q_exit + &q.h_time_h0 * &delta_h0 + &q.h_time_h1 * &delta_h1;
            let d_d2_q = &q.d3_q * &delta_q_exit + &q.d_h_h0 * &delta_h0 + &q.d_h_h1 * &delta_h1;
            let d_h_l = -(&d_d2_q * &q.dq_ls.mapv(|v| safe_product(v, v))
                + &(&q.d2_q * &(2.0 * &delta_q_ls_exit * &q.dq_ls))
                + &(&d_d1_q * &q.d2q_ls)
                + &(&q.d1_q * &delta_q_ls_ls_exit));
            let d_h_ll = mxtwx(x_log_sigma_exit, &d_h_l, x_log_sigma_exit, row_mask)?;
            assign_symmetric_block(&mut joint, offsets[2], offsets[2], &d_h_ll);
        }

        if let (Some(x_t_en), Some(dq_t_en)) = (x_threshold_entry.as_ref(), q.dq_t_entry.as_ref()) {
            let d_h_h0_t = mxtwx(
                &self.x_time_entry,
                &(-(&dh_h0 * dq_t_en + &q.h_time_h0 * &entry_deltas.delta_q_t)),
                x_t_en,
                row_mask,
            )?;
            let d_h_h1_t = mxtwx(
                &self.x_time_exit,
                &(-(&dh_h1 * &q.dq_t + &q.h_time_h1 * &delta_q_t_exit)),
                x_threshold_exit,
                row_mask,
            )?;
            assign_symmetric_block(&mut joint, offsets[0], offsets[1], &(d_h_h0_t + d_h_h1_t));
        } else {
            let delta_q_t = &delta_q_t_exit;
            let d_h_h0_t = mxtwx(
                &self.x_time_entry,
                &(-(&dh_h0 * &q.dq_t + &q.h_time_h0 * delta_q_t)),
                x_threshold_exit,
                row_mask,
            )?;
            let d_h_h1_t = mxtwx(
                &self.x_time_exit,
                &(-(&dh_h1 * &q.dq_t + &q.h_time_h1 * delta_q_t)),
                x_threshold_exit,
                row_mask,
            )?;
            assign_symmetric_block(&mut joint, offsets[0], offsets[1], &(d_h_h0_t + d_h_h1_t));
        }

        if let (Some(x_ls_en), Some(dq_ls_en)) =
            (x_log_sigma_entry.as_ref(), q.dq_ls_entry.as_ref())
        {
            let d_h_h0_l = mxtwx(
                &self.x_time_entry,
                &(-(&dh_h0 * dq_ls_en + &q.h_time_h0 * &entry_deltas.delta_q_ls)),
                x_ls_en,
                row_mask,
            )?;
            let d_h_h1_l = mxtwx(
                &self.x_time_exit,
                &(-(&dh_h1 * &q.dq_ls + &q.h_time_h1 * &delta_q_ls_exit)),
                x_log_sigma_exit,
                row_mask,
            )?;
            assign_symmetric_block(&mut joint, offsets[0], offsets[2], &(d_h_h0_l + d_h_h1_l));
        } else {
            let delta_q_ls = &delta_q_ls_exit;
            let d_h_h0_l = mxtwx(
                &self.x_time_entry,
                &(-(&dh_h0 * &q.dq_ls + &q.h_time_h0 * delta_q_ls)),
                x_log_sigma_exit,
                row_mask,
            )?;
            let d_h_h1_l = mxtwx(
                &self.x_time_exit,
                &(-(&dh_h1 * &q.dq_ls + &q.h_time_h1 * delta_q_ls)),
                x_log_sigma_exit,
                row_mask,
            )?;
            assign_symmetric_block(&mut joint, offsets[0], offsets[2], &(d_h_h0_l + d_h_h1_l));
        }

        if let (Some(xw_dense), Some(w_offset)) = (xw, offsets.get(3).copied()) {
            let d_d2_q_combined = if x_threshold_entry.is_some() || x_log_sigma_entry.is_some() {
                &d_d2_q_exit + &entry_deltas.d_d2_q
            } else {
                &q.d3_q * &delta_q_exit + &q.d_h_h0 * &delta_h0 + &q.d_h_h1 * &delta_h1
            };
            if let (Some(x_t_en), Some(dq_t_en)) =
                (x_threshold_entry.as_ref(), q.dq_t_entry.as_ref())
            {
                let d_h_tw_exit = mxtwx(
                    x_threshold_exit,
                    &(-(&d_d2_q_exit * &q.dq_t + &q.d2_q1 * &delta_q_t_exit)),
                    xw_dense,
                    row_mask,
                )?;
                let d_h_tw_entry = mxtwx(
                    x_t_en,
                    &(-(&entry_deltas.d_d2_q * dq_t_en + &q.d2_q0 * &entry_deltas.delta_q_t)),
                    xw_dense,
                    row_mask,
                )?;
                assign_symmetric_block(
                    &mut joint,
                    offsets[1],
                    w_offset,
                    &(d_h_tw_exit + d_h_tw_entry),
                );
            } else {
                let d_h_tw = mxtwx(
                    x_threshold_exit,
                    &(-(&d_d2_q_combined * &q.dq_t + &q.d2_q * &delta_q_t_exit)),
                    xw_dense,
                    row_mask,
                )?;
                assign_symmetric_block(&mut joint, offsets[1], w_offset, &d_h_tw);
            }

            if let (Some(x_ls_en), Some(dq_ls_en)) =
                (x_log_sigma_entry.as_ref(), q.dq_ls_entry.as_ref())
            {
                let d_h_lw_exit = mxtwx(
                    x_log_sigma_exit,
                    &(-(&d_d2_q_exit * &q.dq_ls + &q.d2_q1 * &delta_q_ls_exit)),
                    xw_dense,
                    row_mask,
                )?;
                let d_h_lw_entry = mxtwx(
                    x_ls_en,
                    &(-(&entry_deltas.d_d2_q * dq_ls_en + &q.d2_q0 * &entry_deltas.delta_q_ls)),
                    xw_dense,
                    row_mask,
                )?;
                assign_symmetric_block(
                    &mut joint,
                    offsets[2],
                    w_offset,
                    &(d_h_lw_exit + d_h_lw_entry),
                );
            } else {
                let d_h_lw = mxtwx(
                    x_log_sigma_exit,
                    &(-(&d_d2_q_combined * &q.dq_ls + &q.d2_q * &delta_q_ls_exit)),
                    xw_dense,
                    row_mask,
                )?;
                assign_symmetric_block(&mut joint, offsets[2], w_offset, &d_h_lw);
            }

            let d_hww = mxtwx(xw_dense, &(-&d_d2_q_combined), xw_dense, row_mask)?;
            assign_symmetric_block(&mut joint, w_offset, w_offset, &d_hww);

            let d_h_h0w = mxtwx(&self.x_time_entry, &(-&dh_h0), xw_dense, row_mask)?;
            let d_h_h1w = mxtwx(&self.x_time_exit, &(-&dh_h1), xw_dense, row_mask)?;
            assign_symmetric_block(&mut joint, offsets[0], w_offset, &(d_h_h0w + d_h_h1w));
        }

        Ok(Some(joint))
    }

    fn evaluate_log_likelihood_and_block_gradients(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Vec<Array1<f64>>), String> {
        self.evaluate_log_likelihood_and_block_gradients_masked(block_states, None)
    }

    /// HT-mask-aware variant of
    /// [`Self::evaluate_log_likelihood_and_block_gradients`]. `None` is
    /// byte-identical to the pre-refactor implementation. `Some(m)`
    /// multiplies each row's likelihood contribution and per-row partial
    /// derivative contributions by `m[i]` before aggregation: the
    /// downstream `X.t().dot(...)` / `transpose_vector_multiply` calls
    /// then automatically produce the HT-weighted gradient.
    fn evaluate_log_likelihood_and_block_gradients_masked(
        &self,
        block_states: &[ParameterBlockState],
        row_mask: Option<&Array1<f64>>,
    ) -> Result<(f64, Vec<Array1<f64>>), String> {
        let n = self.n;
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let mut ll = 0.0;

        let mut grad_time_eta_h0 = Array1::<f64>::zeros(n);
        let mut grad_time_eta_h1 = Array1::<f64>::zeros(n);
        let mut grad_time_eta_d = Array1::<f64>::zeros(n);
        let mut d1_q0 = Array1::<f64>::zeros(n);
        let mut d1_q1 = Array1::<f64>::zeros(n);
        let mut d1_qdot = Array1::<f64>::zeros(n);

        // HT mask lookup: returns m[i] if mask is Some(m) else 1.0. For
        // f64 multiplication, `x * 1.0 == x` exactly (IEEE 754), so the
        // None path is byte-identical to the pre-refactor expression.
        let mask_at = |i: usize| -> f64 { row_mask.map_or(1.0, |m| m[i]) };
        if n >= Self::EVALUATE_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
            const CHUNK: usize = 1024;
            let d1_q0_s = d1_q0
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let d1_q1_s = d1_q1
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let d1_qdot_s = d1_qdot
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let g_h0_s = grad_time_eta_h0
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let g_h1_s = grad_time_eta_h1
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            let g_d_s = grad_time_eta_d
                .as_slice_memory_order_mut()
                .expect("zeros is contiguous");
            ll = d1_q0_s
                .par_chunks_mut(CHUNK)
                .zip(d1_q1_s.par_chunks_mut(CHUNK))
                .zip(d1_qdot_s.par_chunks_mut(CHUNK))
                .zip(g_h0_s.par_chunks_mut(CHUNK))
                .zip(g_h1_s.par_chunks_mut(CHUNK))
                .zip(g_d_s.par_chunks_mut(CHUNK))
                .enumerate()
                .try_fold(
                    || 0.0_f64,
                    |local_ll,
                     (chunk_idx, (((((d1q0_c, d1q1_c), d1qd_c), gh0_c), gh1_c), gd_c))|
                     -> Result<f64, String> {
                        let start = chunk_idx * CHUNK;
                        let mut acc = local_ll;
                        for local in 0..d1q0_c.len() {
                            let i = start + local;
                            let state = self.row_predictor_state(
                                dynamic.h_entry[i],
                                dynamic.h_exit[i],
                                dynamic.hdot_exit[i],
                                dynamic.q_entry[i],
                                dynamic.q_exit[i],
                                dynamic.qdot_exit[i],
                            );
                            if let Some(row) = self.row_derivatives(i, state)? {
                                let w = mask_at(i);
                                acc += row.ll * w;
                                d1q0_c[local] = row.d1_q0 * w;
                                d1q1_c[local] = row.d1_q1 * w;
                                d1qd_c[local] = row.d1_qdot1 * w;
                                gh0_c[local] = row.grad_time_eta_h0 * w;
                                gh1_c[local] = row.grad_time_eta_h1 * w;
                                gd_c[local] = row.grad_time_eta_d * w;
                            }
                        }
                        Ok(acc)
                    },
                )
                .try_reduce(|| 0.0_f64, |a, b| Ok::<_, String>(a + b))?;
        } else {
            for i in 0..n {
                let state = self.row_predictor_state(
                    dynamic.h_entry[i],
                    dynamic.h_exit[i],
                    dynamic.hdot_exit[i],
                    dynamic.q_entry[i],
                    dynamic.q_exit[i],
                    dynamic.qdot_exit[i],
                );
                let Some(row) = self.row_derivatives(i, state)? else {
                    continue;
                };
                let w = mask_at(i);
                ll += row.ll * w;
                d1_q0[i] = row.d1_q0 * w;
                d1_q1[i] = row.d1_q1 * w;
                d1_qdot[i] = row.d1_qdot1 * w;
                grad_time_eta_h0[i] = row.grad_time_eta_h0 * w;
                grad_time_eta_h1[i] = row.grad_time_eta_h1 * w;
                grad_time_eta_d[i] = row.grad_time_eta_d * w;
            }
        }

        let grad_time = dynamic.time_jac_entry.t().dot(&grad_time_eta_h0)
            + dynamic.time_jac_exit.t().dot(&grad_time_eta_h1)
            + dynamic.time_jac_deriv.t().dot(&grad_time_eta_d);

        let mut scratch = Array1::<f64>::zeros(n);

        let grad_t = if let (Some(x_t_entry), Some(x_t_deriv)) = (
            self.x_threshold_entry.as_ref(),
            self.x_threshold_deriv.as_ref(),
        ) {
            // grad_exit[i] = d1_q1[i] * dq_t_exit[i] + d1_qdot[i] * dqdot_t[i]
            ndarray::Zip::from(&mut scratch)
                .and(&d1_q1)
                .and(&dynamic.dq_t_exit)
                .and(&d1_qdot)
                .and(&dynamic.dqdot_t)
                .for_each(|s, &a, &b, &c, &d| *s = a * b + c * d);
            let mut out = self.x_threshold.transpose_vector_multiply(&scratch);
            // grad_entry[i] = d1_q0[i] * dq_t_entry[i]
            ndarray::Zip::from(&mut scratch)
                .and(&d1_q0)
                .and(&dynamic.dq_t_entry)
                .for_each(|s, &a, &b| *s = a * b);
            out = out + x_t_entry.transpose_vector_multiply(&scratch);
            // grad_deriv[i] = d1_qdot[i] * dqdot_td[i]
            ndarray::Zip::from(&mut scratch)
                .and(&d1_qdot)
                .and(&dynamic.dqdot_td)
                .for_each(|s, &a, &b| *s = a * b);
            out + x_t_deriv.transpose_vector_multiply(&scratch)
        } else {
            // combined[i] = d1_q1[i]*dq_t_exit[i] + d1_q0[i]*dq_t_entry[i] + d1_qdot[i]*dqdot_t[i]
            ndarray::Zip::from(&mut scratch)
                .and(&d1_q1)
                .and(&dynamic.dq_t_exit)
                .and(&d1_q0)
                .and(&dynamic.dq_t_entry)
                .for_each(|s, &a, &b, &c, &d| *s = a * b + c * d);
            ndarray::Zip::from(&mut scratch)
                .and(&d1_qdot)
                .and(&dynamic.dqdot_t)
                .for_each(|s, &a, &b| *s += a * b);
            self.x_threshold.transpose_vector_multiply(&scratch)
        };

        let grad_ls = if let (Some(x_ls_entry), Some(x_ls_deriv)) = (
            self.x_log_sigma_entry.as_ref(),
            self.x_log_sigma_deriv.as_ref(),
        ) {
            ndarray::Zip::from(&mut scratch)
                .and(&d1_q1)
                .and(&dynamic.dq_ls_exit)
                .and(&d1_qdot)
                .and(&dynamic.dqdot_ls)
                .for_each(|s, &a, &b, &c, &d| *s = a * b + c * d);
            let mut out = self.x_log_sigma.transpose_vector_multiply(&scratch);
            ndarray::Zip::from(&mut scratch)
                .and(&d1_q0)
                .and(&dynamic.dq_ls_entry)
                .for_each(|s, &a, &b| *s = a * b);
            out = out + x_ls_entry.transpose_vector_multiply(&scratch);
            ndarray::Zip::from(&mut scratch)
                .and(&d1_qdot)
                .and(&dynamic.dqdot_lsd)
                .for_each(|s, &a, &b| *s = a * b);
            out + x_ls_deriv.transpose_vector_multiply(&scratch)
        } else {
            ndarray::Zip::from(&mut scratch)
                .and(&d1_q1)
                .and(&dynamic.dq_ls_exit)
                .and(&d1_q0)
                .and(&dynamic.dq_ls_entry)
                .for_each(|s, &a, &b, &c, &d| *s = a * b + c * d);
            ndarray::Zip::from(&mut scratch)
                .and(&d1_qdot)
                .and(&dynamic.dqdot_ls)
                .for_each(|s, &a, &b| *s += a * b);
            self.x_log_sigma.transpose_vector_multiply(&scratch)
        };

        let mut block_gradients = vec![grad_time, grad_t, grad_ls];
        if let (Some(xw_exit), Some(xw_entry), Some(xw_qdot)) = (
            dynamic.wiggle_basis_exit.as_ref(),
            dynamic.wiggle_basis_entry.as_ref(),
            dynamic.wiggle_qdot_basis_exit.as_ref(),
        ) {
            let gradw =
                xw_exit.t().dot(&d1_q1) + xw_entry.t().dot(&d1_q0) + xw_qdot.t().dot(&d1_qdot);
            block_gradients.push(gradw);
        }

        Ok((ll, block_gradients))
    }

    /// Build the [`BlockEffectiveJacobian`] for block `block_idx` given the
    /// realised block specs.
    ///
    /// Survival location-scale has three linear outputs per row:
    ///   - output 0: η_time       ← time_transform block (block 0)
    ///   - output 1: η_threshold  ← threshold block (block 1)
    ///   - output 2: η_log_sigma  ← log_sigma block (block 2)
    ///
    /// The optional linkwiggle block (block 3) modulates the inverse link
    /// nonlinearly and has an all-zero effective linear Jacobian.
    ///
    /// The stacked Jacobian for block k has shape `(3 * n, p_k)`.
    pub fn block_effective_jacobian(
        specs: &[ParameterBlockSpec],
        block_idx: usize,
    ) -> Result<Box<dyn BlockEffectiveJacobian>, String> {
        crate::util::block_jacobian::AdditiveWiggleBlockLayout {
            family: "SurvivalLocationScaleFamily",
            n_outputs: 3,
            additive_blocks: &[
                Self::BLOCK_TIME,
                Self::BLOCK_THRESHOLD,
                Self::BLOCK_LOG_SIGMA,
            ],
            wiggle_block: Some(Self::BLOCK_LINK_WIGGLE),
        }
        .block_effective_jacobian(specs, block_idx)
    }
}


/// Per-subject 3×3 channel Hessian W_i for survival location-scale.
///
/// The three output channels are:
///   0. η_time   (time-transform, shared entry/exit predictor shift)
///   1. η_thr    (threshold block — shifts both u0 and u1 identically)
///   2. η_ls     (log-scale block — enters the inverse link)
///
/// The full W_i is the second derivative of the row NLL
/// `ρ_i(η_time, η_thr, η_ls)` at the current pilot β:
///
/// ```text
/// W_i[a, b] = ∂²ρ_i / ∂η_a ∂η_b
/// ```
///
/// These are the same second-order scalars computed by
/// `SurvivalLocationScaleFamily::row_derivatives_rescaled` but arranged
/// into the per-channel output-space matrix instead of the per-block
/// raw-coefficient space.
///
/// When the cross-channel curvature is unavailable (e.g. at the
/// canonicalize step before any pilot β is known), the identity metric
/// is used instead — see [`Self::identity`].
pub struct SurvivalLocationScaleChannelHessian {
    /// Row-major `(n × 3 × 3)` PSD-clamped per-subject Hessian.
    h: ndarray::Array3<f64>,
}


impl SurvivalLocationScaleChannelHessian {
    /// Number of output channels for SLS (always 3).
    pub const K: usize = 3;

    /// Construct from a pre-computed `(n × 3 × 3)` tensor.
    /// No PSD clamping is applied — caller is responsible for ensuring PSD.
    pub fn from_full(h: ndarray::Array3<f64>) -> Self {
        assert_eq!(
            h.shape()[1],
            Self::K,
            "SurvivalLocationScaleChannelHessian: expected K={} channels, got {}",
            Self::K,
            h.shape()[1],
        );
        assert_eq!(
            h.shape()[2],
            Self::K,
            "SurvivalLocationScaleChannelHessian: expected K={} channels, got {}",
            Self::K,
            h.shape()[2],
        );
        Self { h }
    }

    /// Structural identity metric: W_i = I₃ for every subject.
    ///
    /// Used at the canonicalize step where no pilot β is available. The
    /// identity metric gives the structurally correct rank answer (a block
    /// with zero Jacobian contributes no information regardless of the
    /// curvature).
    pub fn identity(n: usize) -> Self {
        let mut h = ndarray::Array3::<f64>::zeros((n, Self::K, Self::K));
        for i in 0..n {
            for c in 0..Self::K {
                h[[i, c, c]] = 1.0;
            }
        }
        Self { h }
    }
}


impl FamilyChannelHessian for SurvivalLocationScaleChannelHessian {
    fn n_outputs(&self) -> usize {
        Self::K
    }

    fn n_subjects(&self) -> usize {
        self.h.shape()[0]
    }

    fn fill_subject(&self, i: usize, out: &mut [f64]) {
        assert_eq!(out.len(), Self::K * Self::K);
        let k = Self::K;
        for a in 0..k {
            for b in 0..k {
                out[a * k + b] = self.h[[i, a, b]];
            }
        }
    }

    fn evaluate_full(&self) -> ndarray::Array3<f64> {
        self.h.clone()
    }
}


/// Public entry point for building a [`BlockEffectiveJacobian`] for one block of
/// the survival location-scale model.
///
/// This thin wrapper exposes the otherwise-private `SurvivalLocationScaleFamily`
/// associated function so integration tests and downstream crates can verify the
/// block Jacobian contract without depending on the internal struct.
///
/// See [`SurvivalLocationScaleFamily::block_effective_jacobian`] for the full
/// contract.
pub fn survival_location_scale_block_effective_jacobian(
    specs: &[ParameterBlockSpec],
    block_idx: usize,
) -> Result<Box<dyn BlockEffectiveJacobian>, String> {
    SurvivalLocationScaleFamily::block_effective_jacobian(specs, block_idx)
}


/// Observed vs expected information: The survival location-scale family uses
/// `BlockWorkingSet::ExactNewton` which provides the actual gradient and Hessian
/// (-nabla^2 log L) from the survival likelihood. This is the **observed** Hessian
/// by construction, which is the correct quantity for the outer REML Laplace
/// approximation (see response.md Section 3). No Fisher surrogate is used here.
//
// WS4a-survival-LS staged outer-score subsampling is enabled through
// Horvitz-Thompson row reweighting. The log-likelihood override streams the
// sampled rows through `exact_row_kernel` and multiplies each row contribution by
// `WeightedOuterRow.weight`. The joint-Hessian and ψ workspaces carry a shared
// row mask into the `_masked` assembly variants, where every row-additive
// `Xᵀ diag(W) Y`, `Xᵀ w`, and dot-product site multiplies the final per-row
// contribution by `mask[i]`. This deliberately masks after each row's nonlinear
// survival derivative algebra has produced the final row coefficient, preserving
// the invariant E[Σ_i (mask_i / π_i) contribution_i] = full-data sum.
impl CustomFamily for SurvivalLocationScaleFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    /// Declare the per-block output channel so the pre-fit identifiability
    /// audit routes **channel-aware** (`audit_identifiability_channel_aware`)
    /// instead of the flat n-row Euclidean stack.
    ///
    /// The survival location-scale row NLL `ρ_i(η_time, η_thr, η_ls)` has THREE
    /// output channels (see `SurvivalLocationScaleChannelHessian`):
    ///   - channel 0 — `η_time` (time-transform predictor shift), and the
    ///     link-wiggle correction anchors here (it perturbs the inverse link
    ///     applied on the time/location side; cf. `AdditiveWiggleBlockLayout`
    ///     in `block_effective_jacobian`, which anchors the wiggle at output 0),
    ///   - channel 1 — `η_thr` (threshold / **location** predictor),
    ///   - channel 2 — `η_ls`  (log-σ / **scale** predictor, entering the
    ///     inverse link multiplicatively).
    ///
    /// Without this assignment the flat audit stacks every block's design into
    /// one n-row Euclidean space, so the threshold's **location intercept**
    /// (a `ones` column on channel 1) and the log-σ block's **scale intercept**
    /// (a `ones` column on channel 2) look like two copies of the same constant
    /// and the joint RRQR reports a (spurious) rank deficiency. The audit then
    /// drops one of them by gauge priority, collapsing a genuine free parameter
    /// and pinning a time-invariant covariate's coefficient to exactly 0
    /// (gam#1110: `gam_a_age = 0`). Both intercepts are separately identifiable
    /// — they live on orthogonal likelihood channels — and the channel-aware
    /// audit recognises this, returning a clean (identity-gauge) verdict with no
    /// column surgery, so every block keeps its raw width (no #1068 z-lift /
    /// fixed-col / monotonicity desync) and the location/scale parameters
    /// recover to the survreg/lifelines MLE.
    ///
    /// `wire_output_channels` installs an `AdditiveBlockJacobian` on each block
    /// from this assignment (the blocks carry no explicit `jacobian_callback`);
    /// that callback feeds ONLY the audit — the inner exact-Newton solve maps
    /// β→η through `solver_design()` (the stacked `[exit; entry; deriv]`
    /// operator), which never reads `jacobian_callback`, so the channel wiring
    /// is invisible to the fit itself. This mirrors the survival marginal-slope
    /// family, which wires its own multi-output Jacobian for the same reason.
    fn output_channel_assignment(&self, specs: &[ParameterBlockSpec]) -> Option<Vec<usize>> {
        Some(
            specs
                .iter()
                .map(|spec| match spec.name.as_str() {
                    "time_transform" => 0,
                    "threshold" => 1,
                    "log_sigma" => 2,
                    // The link-wiggle / time-wiggle corrections perturb the
                    // time/location-side inverse link; anchor them on the time
                    // channel exactly as `block_effective_jacobian` does
                    // (`AdditiveWiggleBlockLayout::wiggle_block` → output 0).
                    _ => 0,
                })
                .collect(),
        )
    }

    fn coefficient_hessian_cost(&self, specs: &[crate::custom_family::ParameterBlockSpec]) -> u64 {
        // Survival location-scale couples its blocks (threshold/time/log-σ
        // and any link/time wiggles) through the survival likelihood: every
        // row contributes a dense outer-product over (Σ p_b) coefficients.
        // At large scale the joint outer evaluator routes the coefficient
        // Hessian through its matrix-free HVP path; the cost remains an honest
        // dense-assembly diagnostic, while exact outer derivative order is now
        // driven by the explicit outer-HVP capability below rather than by a
        // first-order downgrade gate.
        crate::custom_family::joint_coupled_coefficient_hessian_cost(self.n as u64, specs)
    }

    fn outer_hyper_hessian_hvp_available(
        &self,
        specs: &[crate::custom_family::ParameterBlockSpec],
    ) -> bool {
        self.validate_joint_specs(
            specs,
            "SurvivalLocationScaleFamily outer hyper Hessian HVP availability",
        )
        .is_ok()
    }

    fn outer_hyper_hessian_dense_available(
        &self,
        specs: &[crate::custom_family::ParameterBlockSpec],
    ) -> bool {
        let p_total: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
        !crate::custom_family::use_joint_matrix_free_path(p_total, self.n)
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let (ll, block_gradients) =
            self.evaluate_log_likelihood_and_block_gradients(block_states)?;

        // Block-diagonal direct path — assemble only the principal blocks
        // the inner solver consumes. The cross blocks (h_ht, h_hl, h_hw,
        // h_tl, h_tw, h_lw) are not required by per-block working sets, so
        // we never materialize them. See `assemble_block_diagonal_hessians_from_quantities`.
        let q = self.collect_joint_quantities(block_states)?;
        let block_hessians =
            self.assemble_block_diagonal_hessians_from_quantities(&q, block_states)?;
        if block_hessians.len() != block_gradients.len() {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "SurvivalLocationScaleFamily evaluate block count mismatch: gradients={}, hessians={}",
                block_gradients.len(),
                block_hessians.len()
            ) }.into());
        }
        let blockworking_sets = block_gradients
            .into_iter()
            .zip(block_hessians)
            .map(|(gradient, hessian)| BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            })
            .collect();
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets,
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        // Fast path for backtracking line search: compute only the scalar
        // log-likelihood, skipping all gradient/Hessian/derivative assembly.
        let n = self.n;
        let dynamic = self.build_dynamic_geometry(block_states)?;

        let row_log_likelihood = |i: usize| -> Result<f64, String> {
            let state = self.row_predictor_state(
                dynamic.h_entry[i],
                dynamic.h_exit[i],
                dynamic.hdot_exit[i],
                dynamic.q_entry[i],
                dynamic.q_exit[i],
                dynamic.qdot_exit[i],
            );
            Ok(self
                .exact_row_kernel(i, state)?
                .map_or(0.0, SurvivalExactRowKernel::log_likelihood))
        };

        const PARALLEL_LOG_LIKELIHOOD_ROW_THRESHOLD: usize = 1024;
        const LOG_LIKELIHOOD_CHUNK_ROWS: usize = 1024;
        if n < PARALLEL_LOG_LIKELIHOOD_ROW_THRESHOLD {
            let mut ll = 0.0;
            for i in 0..n {
                ll += row_log_likelihood(i)?;
            }
            return Ok(ll);
        }

        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let chunk_sums: Vec<Result<f64, String>> = (0..n.div_ceil(LOG_LIKELIHOOD_CHUNK_ROWS))
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * LOG_LIKELIHOOD_CHUNK_ROWS;
                let end = (start + LOG_LIKELIHOOD_CHUNK_ROWS).min(n);
                let mut ll = 0.0;
                for i in start..end {
                    ll += row_log_likelihood(i)?;
                }
                Ok(ll)
            })
            .collect();

        let mut ll = 0.0;
        for chunk_sum in chunk_sums {
            ll += chunk_sum?;
        }
        Ok(ll)
    }

    fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        let Some(subsample) = options.outer_score_subsample.as_ref() else {
            return self.log_likelihood_only(block_states);
        };
        let n = self.n;
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let mut ll = 0.0;
        for row in subsample.rows.as_ref() {
            let i = row.index;
            if i >= n {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "SurvivalLocationScaleFamily outer subsample row index {i} out of bounds for n={n}"
                    ),
                }
                .into());
            }
            let state = self.row_predictor_state(
                dynamic.h_entry[i],
                dynamic.h_exit[i],
                dynamic.hdot_exit[i],
                dynamic.q_entry[i],
                dynamic.q_exit[i],
                dynamic.qdot_exit[i],
            );
            ll += row.weight
                * self
                    .exact_row_kernel(i, state)?
                    .map_or(0.0, SurvivalExactRowKernel::log_likelihood);
        }
        Ok(ll)
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let dims = self.joint_block_dims();
        if block_idx >= dims.len() {
            return Ok(None);
        }
        if d_beta.len() != dims[block_idx] {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "block {block_idx} d_beta length mismatch: got {}, expected {}",
                    d_beta.len(),
                    dims[block_idx]
                ),
            }
            .into());
        }
        let offsets = self.joint_block_offsets();
        let mut d_beta_flat = Array1::<f64>::zeros(*offsets.last().unwrap());
        d_beta_flat
            .slice_mut(s![offsets[block_idx]..offsets[block_idx + 1]])
            .assign(d_beta);
        // The block-level directional derivative must differentiate the
        // UNRESCALED Hessian (from exact_newton_joint_hessian / evaluate()),
        // not the rescaled one used in the outer curvature path.  Pass
        // log_rescale = 0 so quantities match what evaluate() returns.
        let d_joint = self
            .exact_newton_joint_hessian_directional_derivative_rescaled(
                block_states,
                &d_beta_flat,
                0.0,
            )?
            .ok_or_else(|| {
                "missing survival location-scale exact joint directional Hessian".to_string()
            })?;
        Ok(Some(
            d_joint
                .slice(s![
                    offsets[block_idx]..offsets[block_idx + 1],
                    offsets[block_idx]..offsets[block_idx + 1]
                ])
                .to_owned(),
        ))
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let q = self.collect_joint_quantities(block_states)?;
        if self.row_kernel_joint_hessian_supported() {
            let dynamic = self.build_dynamic_geometry(block_states)?;
            let kernel = self.survival_ls_row_kernel(&q, &dynamic);
            let rows = crate::families::row_kernel::RowSet::All;
            let cache = crate::families::row_kernel::build_row_kernel_cache(&kernel, &rows)?;
            return Ok(Some(crate::families::row_kernel::row_kernel_hessian_dense(
                &kernel, &cache, &rows,
            )));
        }
        self.assemble_joint_hessian_from_quantities(&q, block_states)
    }

    /// Block-concatenated log-likelihood gradient `g = ∇ℓ(θ)` assembled from the
    /// SAME per-row jet-tower kernel that [`Self::exact_newton_joint_hessian`]
    /// uses for `H = −∇²ℓ`. Overrides the `CustomFamily` default (which returns
    /// `None`), so the damped Newton `H δ = g` in `fit_parametric_aft_direct_mle`
    /// is solved on a consistent (objective, gradient, Hessian) triple.
    ///
    /// If `g` were assembled by one code path
    /// (`evaluate_log_likelihood_and_block_gradients`) and `H` by another (the
    /// row-kernel jet tower), any divergence between the two — even a single
    /// dropped cross-channel term — would yield a Newton direction that is not
    /// the true ascent step, so a covariate could stall at its cold-start value
    /// and never move (gam#1110: the log-logistic AFT `age` coefficient pinned to
    /// exactly 0 while the lognormal/probit path, whose hand-coded and jet-tower
    /// gradients happen to coincide, recovers it). Sourcing both from
    /// `row_kernel_*` over one cache makes the objective, its gradient, and its
    /// Hessian provably consistent for every residual distribution.
    ///
    /// `row_kernel_gradient` returns `∇(nll) = −∇ℓ` (the cached per-row jets are
    /// of the negative log-likelihood, pulled back by `jacobian_transpose_action`
    /// — exactly the pullback `row_kernel_hessian_dense` consumes), so we negate
    /// it to return `∇ℓ`. Returns `None` only when the row-kernel joint-Hessian
    /// path is unavailable (then the caller keeps the legacy gradient).
    fn exact_newton_joint_loglik_gradient(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array1<f64>>, String> {
        if !self.row_kernel_joint_hessian_supported() {
            return Ok(None);
        }
        let q = self.collect_joint_quantities(block_states)?;
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let kernel = self.survival_ls_row_kernel(&q, &dynamic);
        let rows = crate::families::row_kernel::RowSet::All;
        let cache = crate::families::row_kernel::build_row_kernel_cache(&kernel, &rows)?;
        let nll_grad =
            crate::families::row_kernel::row_kernel_gradient(&kernel, &cache, &rows);
        Ok(Some(-nll_grad))
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        let (log_likelihood, block_gradients) =
            self.evaluate_log_likelihood_and_block_gradients(block_states)?;
        if block_gradients.len() != specs.len() {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "SurvivalLocationScaleFamily joint gradient block count mismatch: gradients={}, specs={}",
                block_gradients.len(),
                specs.len()
            ) }.into());
        }

        let total_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
        let mut gradient = Array1::<f64>::zeros(total_p);
        let mut offset = 0usize;
        for (block_idx, (block_gradient, spec)) in
            block_gradients.iter().zip(specs.iter()).enumerate()
        {
            let width = spec.design.ncols();
            if block_gradient.len() != width {
                return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                    "SurvivalLocationScaleFamily joint gradient length mismatch for block {block_idx}: got {}, expected {}",
                    block_gradient.len(),
                    width
                ) }.into());
            }
            gradient
                .slice_mut(s![offset..offset + width])
                .assign(block_gradient);
            offset += width;
        }

        Ok(Some(ExactNewtonJointGradientEvaluation {
            log_likelihood,
            gradient,
        }))
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn exact_newton_outer_curvature(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<ExactNewtonOuterCurvature>, String> {
        Ok(self
            .exact_newton_joint_hessian_rescaled(block_states)?
            .map(|(hessian, log_scale)| {
                let p = hessian.nrows();
                ExactNewtonOuterCurvature {
                    hessian,
                    rho_curvature_scale: (-log_scale).exp(),
                    hessian_logdet_correction: p as f64 * log_scale,
                }
            }))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // The trait method uses the full rescale for the outer curvature path.
        self.exact_newton_joint_hessian_directional_derivative_rescaled(
            block_states,
            d_beta_flat,
            self.hessian_deriv_log_rescale(block_states),
        )
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.exact_newton_joint_psi_terms_masked(
            block_states,
            specs,
            derivative_blocks,
            psi_index,
            None,
        )
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        if block_states.len() != self.expected_blocks()
            || derivative_blocks.len() != self.expected_blocks()
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "SurvivalLocationScaleFamily joint psi second-order terms expect {} states and derivative blocks, got {} / {}",
                self.expected_blocks(),
                block_states.len(),
                derivative_blocks.len()
            ) }.into());
        }
        self.validate_joint_specs(
            specs,
            "SurvivalLocationScaleFamily joint psi second-order terms",
        )?;
        let psi_dim = derivative_blocks.iter().map(Vec::len).sum::<usize>();
        if psi_i >= psi_dim || psi_j >= psi_dim {
            return Ok(None);
        }
        Ok(None)
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if block_states.len() != self.expected_blocks()
            || specs.len() != self.expected_blocks()
            || derivative_blocks.len() != self.expected_blocks()
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "SurvivalLocationScaleFamily joint psi workspace expects {} states, specs, and derivative blocks, got {} / {} / {}",
                self.expected_blocks(),
                block_states.len(),
                specs.len(),
                derivative_blocks.len()
            ) }.into());
        }
        Ok(Some(Arc::new(SurvivalExactNewtonJointPsiWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            specs.to_vec(),
            derivative_blocks.to_vec(),
        )?)))
    }

    fn exact_newton_joint_psi_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if block_states.len() != self.expected_blocks()
            || specs.len() != self.expected_blocks()
            || derivative_blocks.len() != self.expected_blocks()
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "SurvivalLocationScaleFamily joint psi workspace expects {} states, specs, and derivative blocks, got {} / {} / {}",
                self.expected_blocks(),
                block_states.len(),
                specs.len(),
                derivative_blocks.len()
            ) }.into());
        }
        let mut workspace = SurvivalExactNewtonJointPsiWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            specs.to_vec(),
            derivative_blocks.to_vec(),
        )?;
        if let Some(subsample) = options.outer_score_subsample.as_ref() {
            workspace.apply_outer_subsample(subsample.rows.as_ref());
        }
        Ok(Some(Arc::new(workspace)))
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != self.expected_blocks()
            || derivative_blocks.len() != self.expected_blocks()
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "SurvivalLocationScaleFamily joint psi Hessian directional derivative expects {} states and derivative blocks, got {} / {}",
                self.expected_blocks(),
                block_states.len(),
                derivative_blocks.len()
            ) }.into());
        }
        self.validate_joint_specs(
            specs,
            "SurvivalLocationScaleFamily joint psi Hessian directional derivative",
        )?;
        let p_total = *self
            .joint_block_offsets()
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        if d_beta_flat.len() != p_total {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "joint psi Hessian directional derivative d_beta length mismatch: got {}, expected {p_total}",
                    d_beta_flat.len()
                ),
            }
            .into());
        }
        let psi_dim = derivative_blocks.iter().map(Vec::len).sum::<usize>();
        if psi_index >= psi_dim {
            return Ok(None);
        }
        Ok(None)
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_states.len() != self.expected_blocks() {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "SurvivalLocationScaleFamily joint Hessian second directional derivative expects {} states, got {}",
                    self.expected_blocks(),
                    block_states.len()
                ),
            }
            .into());
        }
        let p_total = *self
            .joint_block_offsets()
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        if d_beta_u_flat.len() != p_total || d_beta_v_flat.len() != p_total {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "joint Hessian second directional derivative length mismatch: got {} / {}, expected {p_total}",
                    d_beta_u_flat.len(),
                    d_beta_v_flat.len()
                ),
            }
            .into());
        }
        Ok(None)
    }

    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        if block_idx == Self::BLOCK_LINK_WIGGLE {
            return Ok(monotone_wiggle_nonnegative_constraints(spec.design.ncols()));
        }
        if block_idx != Self::BLOCK_TIME {
            return Ok(None);
        }
        Ok(self.time_linear_constraints.clone())
    }

    fn max_feasible_step_size(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        if block_idx == Self::BLOCK_TIME {
            return self.max_feasible_time_step(&block_states[Self::BLOCK_TIME].beta, delta);
        }
        if block_idx == Self::BLOCK_LINK_WIGGLE {
            return self
                .max_feasible_link_wiggle_step(&block_states[Self::BLOCK_LINK_WIGGLE].beta, delta);
        }
        Ok(None)
    }

    fn post_update_block_beta(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        assert!(block_states.len() <= isize::MAX as usize);
        assert!(!block_spec.name.is_empty());
        if block_idx == Self::BLOCK_TIME
            && let Some(constraints) = self.time_linear_constraints.as_ref()
        {
            validate_linear_constraints("time post-update", &beta, constraints)?;
        } else if block_idx == Self::BLOCK_LINK_WIGGLE && self.x_link_wiggle.is_some() {
            for j in 0..beta.len() {
                let tol = CONSTRAINT_NONNEGATIVITY_REL_TOL * beta[j].abs().max(1.0);
                if !beta[j].is_finite() || beta[j] < -tol {
                    return Err(SurvivalLocationScaleError::ConstraintViolation {
                        reason: format!(
                            "survival location-scale link-wiggle post-update violates represented nonnegativity at coefficient {j}: value={:.3e}, tol={:.3e}",
                            beta[j], tol
                        ),
                    }
                    .into());
                }
            }
        }
        Ok(beta)
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        self.validate_joint_specs(specs, "SurvivalLocationScaleFamily joint Hessian workspace")?;
        // #921 remaining boundary: do not replace this wrapper with
        // `RowKernelHessianWorkspace` wholesale until the RowKernel adapter
        // stores the fourth index derivatives and the time-varying qdot
        // third-order map tensor. The wrapper routes supported non-wiggle
        // dense Hessian / first directional derivative calls through the
        // generic engine, while unsupported wiggle and time-varying-qdot
        // cases stay on the existing complete algebra instead of pretending the generic
        // fourth-order hook is complete.
        Ok(Some(Arc::new(
            SurvivalLocationScaleExactNewtonJointHessianWorkspace::new(
                self.clone(),
                block_states.to_vec(),
            )?,
        )))
    }

    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        self.validate_joint_specs(
            specs,
            "SurvivalLocationScaleFamily joint Hessian workspace with options",
        )?;
        // See the non-options workspace constructor above. The same boundary
        // applies here; the HT row mask is threaded into the supported
        // RowKernel first-derivative path by `row_set_from_survival_mask`.
        let mut workspace = SurvivalLocationScaleExactNewtonJointHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
        )?;
        if let Some(subsample) = options.outer_score_subsample.as_ref() {
            workspace.apply_outer_subsample(subsample.rows.as_ref());
        }
        Ok(Some(Arc::new(workspace)))
    }

    fn outer_derivative_subsample_capable(&self) -> bool {
        true
    }

    // Inherent `exact_newton_joint_psi_terms_masked` is defined in the
    // `impl SurvivalLocationScaleFamily` block below. It is invoked directly
    // by both this trait method and the ψ workspace's `first_order_terms`
    // override to thread the Horvitz-Thompson row mask through the staged
    // outer-score subsample.
}


impl SurvivalLocationScaleFamily {
    /// HT-mask-aware variant of [`Self::exact_newton_joint_psi_terms`].
    ///
    /// Lives in an inherent impl (not the `impl CustomFamily` trait impl)
    /// because the trait does not declare a `_masked` signature. The survival
    /// ψ workspace overrides `first_order_terms` to invoke this directly with
    /// the workspace's `row_mask`, so the trait dispatch stays on the
    /// pre-refactor `exact_newton_joint_psi_terms` (full data) while staged
    /// outer subsampling threads the HT mask through this side door.
    pub(crate) fn exact_newton_joint_psi_terms_masked(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        row_mask: Option<&Array1<f64>>,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if specs.len() != self.expected_blocks()
            || derivative_blocks.len() != self.expected_blocks()
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "SurvivalLocationScaleFamily joint psi terms expect {} specs and derivative blocks, got {} and {}",
                self.expected_blocks(),
                specs.len(),
                derivative_blocks.len()
            ) }.into());
        }
        let Some(dir) =
            self.exact_newton_joint_psi_direction(block_states, derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let z_t_exit_psi = &dir.z_t_exit_psi;
        let z_t_entry_psi = &dir.z_t_entry_psi;
        let z_ls_exit_psi = &dir.z_ls_exit_psi;
        let z_ls_entry_psi = &dir.z_ls_entry_psi;
        let q = self.collect_joint_quantities(block_states)?;
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let offsets = self.joint_block_offsets();
        let p_total = *offsets
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;

        let x_threshold_exit_cow = self.x_threshold.to_dense_cow();
        let x_threshold_exit = &*x_threshold_exit_cow;
        let x_threshold_entry_cow = self
            .x_threshold_entry
            .as_ref()
            .map(DesignMatrix::to_dense_cow);
        let x_threshold_entry = x_threshold_entry_cow
            .as_ref()
            .map_or(x_threshold_exit, |c| &**c);
        let x_log_sigma_exit_cow = self.x_log_sigma.to_dense_cow();
        let x_log_sigma_exit = &*x_log_sigma_exit_cow;
        let x_log_sigma_entry_cow = self
            .x_log_sigma_entry
            .as_ref()
            .map(DesignMatrix::to_dense_cow);
        let x_log_sigma_entry = x_log_sigma_entry_cow
            .as_ref()
            .map_or(x_log_sigma_exit, |c| &**c);
        let xw_cow = self.x_link_wiggle.as_ref().map(DesignMatrix::to_dense_cow);
        let xw = xw_cow.as_deref();
        let x_t_exit_map = first_psi_linear_map(
            dir.x_t_exit_action.as_ref(),
            dir.x_t_exit_psi.as_ref(),
            self.n,
            x_threshold_exit.ncols(),
        );
        let x_t_entry_map = first_psi_linear_map(
            dir.x_t_entry_action.as_ref(),
            dir.x_t_entry_psi.as_ref(),
            self.n,
            x_threshold_entry.ncols(),
        );
        let x_ls_exit_map = first_psi_linear_map(
            dir.x_ls_exit_action.as_ref(),
            dir.x_ls_exit_psi.as_ref(),
            self.n,
            x_log_sigma_exit.ncols(),
        );
        let x_ls_entry_map = first_psi_linear_map(
            dir.x_ls_entry_action.as_ref(),
            dir.x_ls_entry_psi.as_ref(),
            self.n,
            x_log_sigma_entry.ncols(),
        );

        let dq_t_entry = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
        let dq_ls_entry = q.dq_ls_entry.as_ref().unwrap_or(&q.dq_ls);
        let d2q_tls_entry = q.d2q_tls_entry.as_ref().unwrap_or(&q.d2q_tls);
        let d2q_ls_entry = q.d2q_ls_entry.as_ref().unwrap_or(&q.d2q_ls);
        let d3q_tls_ls_entry = q.d3q_tls_ls_entry.as_ref().unwrap_or(&q.d3q_tls_ls);
        let d3q_ls_entry = q.d3q_ls_entry.as_ref().unwrap_or(&q.d3q_ls);

        let q0_psi = &(dq_t_entry * z_t_entry_psi) + &(dq_ls_entry * z_ls_entry_psi);
        let q1_psi = &(&q.dq_t * z_t_exit_psi) + &(&q.dq_ls * z_ls_exit_psi);
        let dq_t_entry_psi = d2q_tls_entry * z_ls_entry_psi;
        let dq_t_exit_psi = &q.d2q_tls * z_ls_exit_psi;
        let dq_ls_entry_psi = d2q_tls_entry * z_t_entry_psi + d2q_ls_entry * z_ls_entry_psi;
        let dq_ls_exit_psi = &q.d2q_tls * z_t_exit_psi + &q.d2q_ls * z_ls_exit_psi;
        let d2q_tls_entry_psi = d3q_tls_ls_entry * z_ls_entry_psi;
        let d2q_tls_exit_psi = &q.d3q_tls_ls * z_ls_exit_psi;
        let d2q_ls_entry_psi = d3q_tls_ls_entry * z_t_entry_psi + d3q_ls_entry * z_ls_entry_psi;
        let d2q_ls_exit_psi = &q.d3q_tls_ls * z_t_exit_psi + &q.d3q_ls * z_ls_exit_psi;

        let objective_psi = if let Some(m) = row_mask {
            (&(&q.d1_q0 * &q0_psi) * m).sum() + (&(&q.d1_q1 * &q1_psi) * m).sum()
        } else {
            q.d1_q0.dot(&q0_psi) + q.d1_q1.dot(&q1_psi)
        };

        let mut score_psi = Array1::<f64>::zeros(p_total);
        let time_row_entry = -&q.d2_q0 * &q0_psi;
        let time_row_exit = -&q.d2_q1 * &q1_psi;
        let time_score = dynamic
            .time_jac_entry
            .t()
            .dot(&*mask_row_vec(&time_row_entry, row_mask))
            + dynamic
                .time_jac_exit
                .t()
                .dot(&*mask_row_vec(&time_row_exit, row_mask));
        score_psi
            .slice_mut(s![offsets[0]..offsets[1]])
            .assign(&time_score);

        let threshold_score_row_exit = &q.d1_q1 * &q.dq_t;
        let threshold_score_row_entry = &q.d1_q0 * dq_t_entry;
        let d_threshold_score_row_exit = &q.d2_q1 * &q1_psi * &q.dq_t + &q.d1_q1 * &dq_t_exit_psi;
        let d_threshold_score_row_entry =
            &q.d2_q0 * &q0_psi * dq_t_entry + &q.d1_q0 * &dq_t_entry_psi;
        let threshold_score = x_t_exit_map
            .transpose_mul(mask_row_vec(&threshold_score_row_exit, row_mask).view())
            + x_threshold_exit
                .t()
                .dot(&*mask_row_vec(&d_threshold_score_row_exit, row_mask))
            + x_t_entry_map
                .transpose_mul(mask_row_vec(&threshold_score_row_entry, row_mask).view())
            + x_threshold_entry
                .t()
                .dot(&*mask_row_vec(&d_threshold_score_row_entry, row_mask));
        score_psi
            .slice_mut(s![offsets[1]..offsets[2]])
            .assign(&threshold_score);

        let log_sigma_score_row_exit = &q.d1_q1 * &q.dq_ls;
        let log_sigma_score_row_entry = &q.d1_q0 * dq_ls_entry;
        let d_log_sigma_score_row_exit = &q.d2_q1 * &q1_psi * &q.dq_ls + &q.d1_q1 * &dq_ls_exit_psi;
        let d_log_sigma_score_row_entry =
            &q.d2_q0 * &q0_psi * dq_ls_entry + &q.d1_q0 * &dq_ls_entry_psi;
        let log_sigma_score = x_ls_exit_map
            .transpose_mul(mask_row_vec(&log_sigma_score_row_exit, row_mask).view())
            + x_log_sigma_exit
                .t()
                .dot(&*mask_row_vec(&d_log_sigma_score_row_exit, row_mask))
            + x_ls_entry_map
                .transpose_mul(mask_row_vec(&log_sigma_score_row_entry, row_mask).view())
            + x_log_sigma_entry
                .t()
                .dot(&*mask_row_vec(&d_log_sigma_score_row_entry, row_mask));
        score_psi
            .slice_mut(s![offsets[2]..offsets[3]])
            .assign(&log_sigma_score);

        if let (Some(xw_dense), Some(w_offset)) = (xw, offsets.get(3).copied()) {
            let wiggle_row = &q.d2_q0 * &q0_psi + &q.d2_q1 * &q1_psi;
            let wiggle_score = xw_dense.t().dot(&*mask_row_vec(&wiggle_row, row_mask));
            score_psi
                .slice_mut(s![w_offset..offsets[4]])
                .assign(&wiggle_score);
        }

        let h_time_time = mxtwxd(&dynamic.time_jac_entry, &(-&q.d3_q0 * &q0_psi), row_mask)
            + mxtwxd(&dynamic.time_jac_exit, &(-&q.d3_q1 * &q1_psi), row_mask);

        let h_tt_entry = -(&q.d2_q0 * &dq_t_entry.mapv(|v| safe_product(v, v)));
        let h_tt_exit = -(&q.d2_q1 * &q.dq_t.mapv(|v| safe_product(v, v)));
        let dh_tt_entry = -(&q.d3_q0 * &q0_psi * &dq_t_entry.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q0 * dq_t_entry * &dq_t_entry_psi));
        let dh_tt_exit = -(&q.d3_q1 * &q1_psi * &q.dq_t.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q1 * &q.dq_t * &dq_t_exit_psi));

        let h_ll_entry =
            -(&q.d2_q0 * &dq_ls_entry.mapv(|v| safe_product(v, v)) + &(&q.d1_q0 * d2q_ls_entry));
        let h_ll_exit =
            -(&q.d2_q1 * &q.dq_ls.mapv(|v| safe_product(v, v)) + &(&q.d1_q1 * &q.d2q_ls));
        let dh_ll_entry = -(&q.d3_q0 * &q0_psi * &dq_ls_entry.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q0 * dq_ls_entry * &dq_ls_entry_psi)
            + &(&q.d2_q0 * &q0_psi * d2q_ls_entry)
            + &(&q.d1_q0 * &d2q_ls_entry_psi));
        let dh_ll_exit = -(&q.d3_q1 * &q1_psi * &q.dq_ls.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q1 * &q.dq_ls * &dq_ls_exit_psi)
            + &(&q.d2_q1 * &q1_psi * &q.d2q_ls)
            + &(&q.d1_q1 * &d2q_ls_exit_psi));

        let h_tl_entry = -(&q.d2_q0 * &(dq_t_entry * dq_ls_entry) + &(&q.d1_q0 * d2q_tls_entry));
        let h_tl_exit = -(&q.d2_q1 * &(&q.dq_t * &q.dq_ls) + &(&q.d1_q1 * &q.d2q_tls));
        let dh_tl_entry = -(&q.d3_q0 * &q0_psi * &(dq_t_entry * dq_ls_entry)
            + &(&q.d2_q0 * &(&dq_t_entry_psi * dq_ls_entry + dq_t_entry * &dq_ls_entry_psi))
            + &(&q.d2_q0 * &q0_psi * d2q_tls_entry)
            + &(&q.d1_q0 * &d2q_tls_entry_psi));
        let dh_tl_exit = -(&q.d3_q1 * &q1_psi * &(&q.dq_t * &q.dq_ls)
            + &(&q.d2_q1 * &(&dq_t_exit_psi * &q.dq_ls + &q.dq_t * &dq_ls_exit_psi))
            + &(&q.d2_q1 * &q1_psi * &q.d2q_tls)
            + &(&q.d1_q1 * &d2q_tls_exit_psi));

        let h_h0_t = &q.d2_q0 * dq_t_entry;
        let h_h1_t = &q.d2_q1 * &q.dq_t;
        let dh_h0_t = &q.d3_q0 * &q0_psi * dq_t_entry + &q.d2_q0 * &dq_t_entry_psi;
        let dh_h1_t = &q.d3_q1 * &q1_psi * &q.dq_t + &q.d2_q1 * &dq_t_exit_psi;

        let h_h0_ls = &q.d2_q0 * dq_ls_entry;
        let h_h1_ls = &q.d2_q1 * &q.dq_ls;
        let dh_h0_ls = &q.d3_q0 * &q0_psi * dq_ls_entry + &q.d2_q0 * &dq_ls_entry_psi;
        let dh_h1_ls = &q.d3_q1 * &q1_psi * &q.dq_ls + &q.d2_q1 * &dq_ls_exit_psi;
        let h_tw_entry = -(&q.d2_q0 * dq_t_entry);
        let h_tw_exit = -(&q.d2_q1 * &q.dq_t);
        let dh_tw_entry = -(&q.d3_q0 * &q0_psi * dq_t_entry + &q.d2_q0 * &dq_t_entry_psi);
        let dh_tw_exit = -(&q.d3_q1 * &q1_psi * &q.dq_t + &q.d2_q1 * &dq_t_exit_psi);
        let h_lw_entry = -(&q.d2_q0 * dq_ls_entry);
        let h_lw_exit = -(&q.d2_q1 * &q.dq_ls);
        let dh_lw_entry = -(&q.d3_q0 * &q0_psi * dq_ls_entry + &q.d2_q0 * &dq_ls_entry_psi);
        let dh_lw_exit = -(&q.d3_q1 * &q1_psi * &q.dq_ls + &q.d2_q1 * &dq_ls_exit_psi);

        if dir.x_t_exit_action.is_some()
            || dir.x_t_entry_action.is_some()
            || dir.x_ls_exit_action.is_some()
            || dir.x_ls_entry_action.is_some()
        {
            // HT-mask helper. Each per-row pair weight (h_*, dh_*, ±d3·q_psi)
            // is multiplied by the mask before being moved into the deferred
            // operator. `None` is a zero-cost passthrough.
            let mw = |arr: Array1<f64>| -> Array1<f64> {
                match row_mask {
                    Some(m) => &arr * m,
                    None => arr,
                }
            };
            let mut channels = vec![
                CustomFamilyJointDesignChannel::new(
                    offsets[0]..offsets[1],
                    shared_dense_arc(&self.x_time_entry),
                    None,
                ),
                CustomFamilyJointDesignChannel::new(
                    offsets[0]..offsets[1],
                    shared_dense_arc(&self.x_time_exit),
                    None,
                ),
                CustomFamilyJointDesignChannel::new(
                    offsets[1]..offsets[2],
                    shared_dense_arc(x_threshold_exit),
                    dir.x_t_exit_action.clone(),
                ),
                CustomFamilyJointDesignChannel::new(
                    offsets[1]..offsets[2],
                    shared_dense_arc(x_threshold_entry),
                    dir.x_t_entry_action.clone(),
                ),
                CustomFamilyJointDesignChannel::new(
                    offsets[2]..offsets[3],
                    shared_dense_arc(x_log_sigma_exit),
                    dir.x_ls_exit_action.clone(),
                ),
                CustomFamilyJointDesignChannel::new(
                    offsets[2]..offsets[3],
                    shared_dense_arc(x_log_sigma_entry),
                    dir.x_ls_entry_action.clone(),
                ),
            ];
            let mut pairs = vec![
                CustomFamilyJointDesignPairContribution::new(
                    0,
                    0,
                    mw(Array1::zeros(self.x_time_entry.nrows())),
                    mw(-&q.d3_q0 * &q0_psi),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    1,
                    1,
                    mw(Array1::zeros(self.x_time_exit.nrows())),
                    mw(-&q.d3_q1 * &q1_psi),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    2,
                    2,
                    mw(h_tt_exit.clone()),
                    mw(dh_tt_exit.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    3,
                    3,
                    mw(h_tt_entry.clone()),
                    mw(dh_tt_entry.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    4,
                    4,
                    mw(h_ll_exit.clone()),
                    mw(dh_ll_exit.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    5,
                    5,
                    mw(h_ll_entry.clone()),
                    mw(dh_ll_entry.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    2,
                    4,
                    mw(h_tl_exit.clone()),
                    mw(dh_tl_exit.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    4,
                    2,
                    mw(h_tl_exit.clone()),
                    mw(dh_tl_exit.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    3,
                    5,
                    mw(h_tl_entry.clone()),
                    mw(dh_tl_entry.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    5,
                    3,
                    mw(h_tl_entry.clone()),
                    mw(dh_tl_entry.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    0,
                    3,
                    mw(h_h0_t.clone()),
                    mw(dh_h0_t.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    3,
                    0,
                    mw(h_h0_t.clone()),
                    mw(dh_h0_t.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    1,
                    2,
                    mw(h_h1_t.clone()),
                    mw(dh_h1_t.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    2,
                    1,
                    mw(h_h1_t.clone()),
                    mw(dh_h1_t.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    0,
                    5,
                    mw(h_h0_ls.clone()),
                    mw(dh_h0_ls.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    5,
                    0,
                    mw(h_h0_ls.clone()),
                    mw(dh_h0_ls.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    1,
                    4,
                    mw(h_h1_ls.clone()),
                    mw(dh_h1_ls.clone()),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    4,
                    1,
                    mw(h_h1_ls.clone()),
                    mw(dh_h1_ls.clone()),
                ),
            ];
            if let (Some(xw_dense), Some(w_offset)) = (xw, offsets.get(3).copied()) {
                channels.push(CustomFamilyJointDesignChannel::new(
                    w_offset..offsets[4],
                    shared_dense_arc(xw_dense),
                    None,
                ));
                let w_idx = channels.len() - 1;
                let zero_w = Array1::zeros(xw_dense.nrows());
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    w_idx,
                    mw(zero_w.clone()),
                    mw(-&q.d3_q0 * &q0_psi - &q.d3_q1 * &q1_psi),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    2,
                    w_idx,
                    mw(h_tw_exit.clone()),
                    mw(dh_tw_exit.clone()),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    2,
                    mw(h_tw_exit.clone()),
                    mw(dh_tw_exit.clone()),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    3,
                    w_idx,
                    mw(h_tw_entry.clone()),
                    mw(dh_tw_entry.clone()),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    3,
                    mw(h_tw_entry.clone()),
                    mw(dh_tw_entry.clone()),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    4,
                    w_idx,
                    mw(h_lw_exit.clone()),
                    mw(dh_lw_exit.clone()),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    4,
                    mw(h_lw_exit.clone()),
                    mw(dh_lw_exit.clone()),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    5,
                    w_idx,
                    mw(h_lw_entry.clone()),
                    mw(dh_lw_entry.clone()),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    5,
                    mw(h_lw_entry.clone()),
                    mw(dh_lw_entry.clone()),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    0,
                    w_idx,
                    mw(zero_w.clone()),
                    mw(&q.d3_q0 * &q0_psi),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    0,
                    mw(zero_w.clone()),
                    mw(&q.d3_q0 * &q0_psi),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    1,
                    w_idx,
                    mw(zero_w.clone()),
                    mw(&q.d3_q1 * &q1_psi),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    1,
                    mw(zero_w),
                    mw(&q.d3_q1 * &q1_psi),
                ));
            }
            return Ok(Some(ExactNewtonJointPsiTerms {
                objective_psi,
                score_psi,
                hessian_psi: Array2::zeros((0, 0)),
                hessian_psi_operator: Some(std::sync::Arc::new(CustomFamilyJointPsiOperator::new(
                    p_total, channels, pairs,
                ))),
            }));
        }
        let mut hessian_psi = Array2::<f64>::zeros((p_total, p_total));
        assign_symmetric_block(&mut hessian_psi, offsets[0], offsets[0], &h_time_time);
        let h_threshold_threshold =
            mxtwx_psi(
                x_t_exit_map,
                h_tt_exit.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
                row_mask,
            )? + mxtwx_psi(
                CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
                h_tt_exit.view(),
                x_t_exit_map,
                row_mask,
            )? + mxtwx(x_threshold_exit, &dh_tt_exit, x_threshold_exit, row_mask)?
                + mxtwx_psi(
                    x_t_entry_map,
                    h_tt_entry.view(),
                    CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
                    row_mask,
                )?
                + mxtwx_psi(
                    CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
                    h_tt_entry.view(),
                    x_t_entry_map,
                    row_mask,
                )?
                + mxtwx(x_threshold_entry, &dh_tt_entry, x_threshold_entry, row_mask)?;
        assign_symmetric_block(
            &mut hessian_psi,
            offsets[1],
            offsets[1],
            &h_threshold_threshold,
        );
        let h_log_sigma_log_sigma =
            mxtwx_psi(
                x_ls_exit_map,
                h_ll_exit.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
                row_mask,
            )? + mxtwx_psi(
                CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
                h_ll_exit.view(),
                x_ls_exit_map,
                row_mask,
            )? + mxtwx(x_log_sigma_exit, &dh_ll_exit, x_log_sigma_exit, row_mask)?
                + mxtwx_psi(
                    x_ls_entry_map,
                    h_ll_entry.view(),
                    CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
                    row_mask,
                )?
                + mxtwx_psi(
                    CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
                    h_ll_entry.view(),
                    x_ls_entry_map,
                    row_mask,
                )?
                + mxtwx(x_log_sigma_entry, &dh_ll_entry, x_log_sigma_entry, row_mask)?;
        assign_symmetric_block(
            &mut hessian_psi,
            offsets[2],
            offsets[2],
            &h_log_sigma_log_sigma,
        );
        let h_threshold_log_sigma =
            mxtwx_psi(
                x_t_exit_map,
                h_tl_exit.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
                row_mask,
            )? + mxtwx_psi(
                CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
                h_tl_exit.view(),
                x_ls_exit_map,
                row_mask,
            )? + mxtwx(x_threshold_exit, &dh_tl_exit, x_log_sigma_exit, row_mask)?
                + mxtwx_psi(
                    x_t_entry_map,
                    h_tl_entry.view(),
                    CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
                    row_mask,
                )?
                + mxtwx_psi(
                    CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
                    h_tl_entry.view(),
                    x_ls_entry_map,
                    row_mask,
                )?
                + mxtwx(x_threshold_entry, &dh_tl_entry, x_log_sigma_entry, row_mask)?;
        assign_symmetric_block(
            &mut hessian_psi,
            offsets[1],
            offsets[2],
            &h_threshold_log_sigma,
        );
        let h_time_threshold = mxtwx(&self.x_time_entry, &dh_h0_t, x_threshold_entry, row_mask)?
            + mxtwx_psi(
                CustomFamilyPsiLinearMapRef::Dense(&self.x_time_entry),
                h_h0_t.view(),
                x_t_entry_map,
                row_mask,
            )?
            + mxtwx(&self.x_time_exit, &dh_h1_t, x_threshold_exit, row_mask)?
            + mxtwx_psi(
                CustomFamilyPsiLinearMapRef::Dense(&self.x_time_exit),
                h_h1_t.view(),
                x_t_exit_map,
                row_mask,
            )?;
        assign_symmetric_block(&mut hessian_psi, offsets[0], offsets[1], &h_time_threshold);
        let h_time_log_sigma = mxtwx(&self.x_time_entry, &dh_h0_ls, x_log_sigma_entry, row_mask)?
            + mxtwx_psi(
                CustomFamilyPsiLinearMapRef::Dense(&self.x_time_entry),
                h_h0_ls.view(),
                x_ls_entry_map,
                row_mask,
            )?
            + mxtwx(&self.x_time_exit, &dh_h1_ls, x_log_sigma_exit, row_mask)?
            + mxtwx_psi(
                CustomFamilyPsiLinearMapRef::Dense(&self.x_time_exit),
                h_h1_ls.view(),
                x_ls_exit_map,
                row_mask,
            )?;
        assign_symmetric_block(&mut hessian_psi, offsets[0], offsets[2], &h_time_log_sigma);

        if let (Some(xw_dense), Some(w_offset)) = (xw, offsets.get(3).copied()) {
            let h_ww = -(&q.d3_q0 * &q0_psi + &q.d3_q1 * &q1_psi);
            let h_wiggle_wiggle = mxtwx(xw_dense, &h_ww, xw_dense, row_mask)?;
            assign_symmetric_block(&mut hessian_psi, w_offset, w_offset, &h_wiggle_wiggle);
            let h_threshold_wiggle = mxtwx_psi(
                x_t_exit_map,
                h_tw_exit.view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
                row_mask,
            )? + mxtwx(x_threshold_exit, &dh_tw_exit, xw_dense, row_mask)?
                + mxtwx_psi(
                    x_t_entry_map,
                    h_tw_entry.view(),
                    CustomFamilyPsiLinearMapRef::Dense(xw_dense),
                    row_mask,
                )?
                + mxtwx(x_threshold_entry, &dh_tw_entry, xw_dense, row_mask)?;
            assign_symmetric_block(&mut hessian_psi, offsets[1], w_offset, &h_threshold_wiggle);
            let h_log_sigma_wiggle = mxtwx_psi(
                x_ls_exit_map,
                h_lw_exit.view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
                row_mask,
            )? + mxtwx(x_log_sigma_exit, &dh_lw_exit, xw_dense, row_mask)?
                + mxtwx_psi(
                    x_ls_entry_map,
                    h_lw_entry.view(),
                    CustomFamilyPsiLinearMapRef::Dense(xw_dense),
                    row_mask,
                )?
                + mxtwx(x_log_sigma_entry, &dh_lw_entry, xw_dense, row_mask)?;
            assign_symmetric_block(&mut hessian_psi, offsets[2], w_offset, &h_log_sigma_wiggle);
            let h_time_wiggle =
                mxtwx(
                    &self.x_time_entry,
                    &(&q.d3_q0 * &q0_psi),
                    xw_dense,
                    row_mask,
                )? + mxtwx(&self.x_time_exit, &(&q.d3_q1 * &q1_psi), xw_dense, row_mask)?;
            assign_symmetric_block(&mut hessian_psi, offsets[0], w_offset, &h_time_wiggle);
        }

        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator: None,
        }))
    }
}


struct SurvivalExactNewtonJointPsiWorkspace {
    family: SurvivalLocationScaleFamily,
    block_states: Vec<ParameterBlockState>,
    specs: Vec<ParameterBlockSpec>,
    derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    row_mask: Option<Arc<Array1<f64>>>,
}


impl SurvivalExactNewtonJointPsiWorkspace {
    fn new(
        family: SurvivalLocationScaleFamily,
        block_states: Vec<ParameterBlockState>,
        specs: Vec<ParameterBlockSpec>,
        derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    ) -> Result<Self, String> {
        Ok(Self {
            family,
            block_states,
            specs,
            derivative_blocks,
            row_mask: None,
        })
    }

    fn apply_outer_subsample(
        &mut self,
        rows: &[crate::families::marginal_slope_shared::WeightedOuterRow],
    ) {
        let n = self.family.n;
        let mut mask = Array1::<f64>::zeros(n);
        for r in rows {
            if r.index < n {
                mask[r.index] = r.weight;
            }
        }
        self.row_mask = Some(Arc::new(mask));
    }
}


impl ExactNewtonJointPsiWorkspace for SurvivalExactNewtonJointPsiWorkspace {
    fn first_order_terms(
        &self,
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.family.exact_newton_joint_psi_terms_masked(
            &self.block_states,
            &self.specs,
            &self.derivative_blocks,
            psi_index,
            self.row_mask.as_deref(),
        )
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let psi_dim = self.derivative_blocks.iter().map(Vec::len).sum::<usize>();
        if psi_i >= psi_dim || psi_j >= psi_dim {
            return Ok(None);
        }
        Ok(None)
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<crate::solver::estimate::reml::unified::DriftDerivResult>, String> {
        let p_total = *self
            .family
            .joint_block_offsets()
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        if d_beta_flat.len() != p_total {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "joint psi workspace Hessian directional derivative d_beta length mismatch: got {}, expected {p_total}",
                    d_beta_flat.len()
                ),
            }
            .into());
        }
        let psi_dim = self.derivative_blocks.iter().map(Vec::len).sum::<usize>();
        if psi_index >= psi_dim {
            return Ok(None);
        }
        Ok(self
            .family
            .exact_newton_joint_psihessian_directional_derivative(
                &self.block_states,
                &self.specs,
                &self.derivative_blocks,
                psi_index,
                d_beta_flat,
            )?
            .map(crate::solver::estimate::reml::unified::DriftDerivResult::Dense))
    }
}


/// Workspace caching the direction-independent state used by the survival
/// location-scale joint-Hessian directional derivative operators.
struct SurvivalLocationScaleExactNewtonJointHessianWorkspace {
    family: SurvivalLocationScaleFamily,
    q: SurvivalJointQuantities,
    dynamic: SurvivalDynamicGeometry,
    row_mask: Option<Arc<Array1<f64>>>,
}


impl SurvivalLocationScaleExactNewtonJointHessianWorkspace {
    fn new(
        family: SurvivalLocationScaleFamily,
        block_states: Vec<ParameterBlockState>,
    ) -> Result<Self, String> {
        let log_rescale = family.hessian_deriv_log_rescale(&block_states);
        let q = family.collect_joint_quantities_rescaled(&block_states, log_rescale)?;
        let dynamic = family.build_dynamic_geometry(&block_states)?;
        Ok(Self {
            family,
            q,
            dynamic,
            row_mask: None,
        })
    }

    fn apply_outer_subsample(
        &mut self,
        rows: &[crate::families::marginal_slope_shared::WeightedOuterRow],
    ) {
        let n = self.family.n;
        let mut mask = Array1::<f64>::zeros(n);
        for r in rows {
            if r.index < n {
                mask[r.index] = r.weight;
            }
        }
        self.row_mask = Some(Arc::new(mask));
    }
}


impl ExactNewtonJointHessianWorkspace for SurvivalLocationScaleExactNewtonJointHessianWorkspace {
    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative_rescaled_from_parts_masked(
                d_beta_flat,
                &self.q,
                &self.dynamic,
                self.row_mask.as_deref(),
            )
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        Ok(self
            .family
            .exact_newton_joint_hessian_directional_derivative_rescaled_from_parts_masked(
                d_beta_flat,
                &self.q,
                &self.dynamic,
                self.row_mask.as_deref(),
            )?
            .map(|matrix| Arc::new(DenseMatrixHyperOperator { matrix }) as Arc<dyn HyperOperator>))
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let p_total = *self
            .family
            .joint_block_offsets()
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        if d_beta_u_flat.len() != p_total || d_beta_v_flat.len() != p_total {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "joint Hessian workspace second directional derivative length mismatch: got {} / {}, expected {p_total}",
                    d_beta_u_flat.len(),
                    d_beta_v_flat.len()
                ),
            }
            .into());
        }
        Ok(None)
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let p_total = *self
            .family
            .joint_block_offsets()
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        if d_beta_u_flat.len() != p_total || d_beta_v_flat.len() != p_total {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "joint Hessian workspace second directional derivative operator length mismatch: got {} / {}, expected {p_total}",
                    d_beta_u_flat.len(),
                    d_beta_v_flat.len()
                ),
            }
            .into());
        }
        Ok(None)
    }
}


/// Run the direct parametric-AFT MLE for a fully reduced constant-scale model
/// and assemble the same [`UnifiedFitResult`] the coupled path would produce.
///
/// Every block is unpenalized (zero ρ) — the reduced affine time-warp, the
/// location intercept/covariate, and the constant log-σ identify the AFT MLE
/// directly, and `survival_reduced_parametric_aft_regime` has already dropped
/// any default parametric shrinkage ridge — so `log_lambdas`/`lambdas` are
/// empty, the stable penalty term is zero, and the penalized objective is just
/// `−ℓ̂`. The conditional covariance is the inverse of the observed information
/// `H` (the joint negative-log-likelihood Hessian at the MLE), and the
/// geometry's penalized Hessian is `H` itself — matching the exact-Newton joint
/// geometry the coupled survival path stores (`working_weights`/`working_response`
/// are the zero-length convention used by exact-Newton joint families). The
/// shared [`crate::custom_family::blockwise_fit_from_parts`] assembler then
/// computes EDF (= parameter count, since unpenalized) and the inference block
/// exactly as for any custom-family fit.
fn fit_reduced_parametric_aft(
    prepared: &PreparedSurvivalLocationScaleModel,
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, String> {
    use crate::faer_ndarray::FaerCholesky;

    let specs = &prepared.blockspecs;
    let (states, log_likelihood, h) = prepared.family.fit_parametric_aft_direct_mle(
        specs,
        options.inner_max_cycles.max(1),
        options.inner_tol.max(REDUCED_AFT_GRAD_TOL_FLOOR),
    )?;

    let p_total = h.nrows();
    // Conditional covariance Var(θ | λ) = H⁻¹ in the reduced coordinate system.
    // `finalize_survival_location_scale_fit` lifts it back to the raw block
    // coordinates (time null-space expansion + leading-fixed-column padding).
    let identity = Array2::<f64>::eye(p_total);
    let covariance_conditional = match h.cholesky(faer::Side::Lower) {
        Ok(chol) => {
            let cov = chol.solve_mat(&identity);
            if cov.iter().all(|v| v.is_finite()) {
                // Symmetrize away round-off so the lifted covariance is exactly
                // symmetric, as the conditional covariance must be.
                let mut symm = cov.clone();
                for i in 0..p_total {
                    for j in (i + 1)..p_total {
                        let avg = 0.5 * (cov[[i, j]] + cov[[j, i]]);
                        symm[[i, j]] = avg;
                        symm[[j, i]] = avg;
                    }
                }
                Some(symm)
            } else {
                None
            }
        }
        Err(_) => None,
    };

    let geometry = Some(FitGeometry {
        penalized_hessian: h.into(),
        working_weights: Array1::<f64>::zeros(0),
        working_response: Array1::<f64>::zeros(0),
    });

    // The block states carry their η in the family's native row layout — the
    // stacked `[exit; entry; deriv]` channels (`solver_design().nrows()` rows)
    // for the time block, exactly as `refresh_all_block_etas` produces and as
    // the family's `validate_joint_states` / `offset_channel_geometry` require.
    // `blockwise_fit_from_parts` validates each block's `η.len()` against
    // `spec.design.nrows()`, so present it the row-matching `solver_design()`
    // as `design` (same coefficients, penalties, name, role — only the row
    // count differs). All other fields are unchanged, so the assembled result
    // is identical to the coupled path's.
    let assembly_specs: Vec<ParameterBlockSpec> = specs
        .iter()
        .map(|spec| {
            let mut s = spec.clone();
            s.design = spec.solver_design().clone();
            s.offset = spec.solver_offset().clone();
            s.stacked_design = None;
            s.stacked_offset = None;
            s
        })
        .collect();

    crate::custom_family::blockwise_fit_from_parts(
        crate::custom_family::BlockwiseFitResultParts {
            block_states: states,
            log_likelihood,
            log_lambdas: Array1::<f64>::zeros(0),
            lambdas: Array1::<f64>::zeros(0),
            covariance_conditional,
            stable_penalty_term: 0.0,
            // No penalties and no smoothing parameters: the reported objective
            // is the plain negative log-likelihood at the MLE.
            penalized_objective: -log_likelihood,
            outer_iterations: 0,
            outer_gradient_norm: Some(0.0),
            criterion_certificate: None,
            inner_cycles: 0,
            outer_converged: true,
            geometry,
            precomputed_edf: None,
        },
        &assembly_specs,
    )
}


/// Variant that also returns the offset-channel residuals + curvatures at the
/// converged β̂. We have to extract these *before* `finalize_survival_location_scale_fit`
/// runs, because the location-scale finalizer empties `UnifiedFitResult::block_states`
/// (see `survival_fit_from_parts` — `block_states: Vec::new()`), and the family's
/// `offset_channel_geometry` method needs the raw, populated per-block state.
fn fit_survival_location_scale_with_geometry(
    spec: SurvivalLocationScaleSpec,
) -> Result<(UnifiedFitResult, SurvivalLocationScaleConvergedGeometry), String> {
    let prepared = prepare_survival_location_scale_model(&spec)?;
    let options = survival_blockwise_fit_options(&spec);
    // Fully reduced constant-scale PARAMETRIC AFT regime (issue #736/#735/#721):
    // every block is parametric and unpenalized, so REML/LAML smoothing
    // selection is vacuous and the coupled exact-joint REML optimizer is the
    // wrong tool — it oscillates and never certifies stationarity on this tiny
    // unpenalized likelihood. Route directly to a damped, line-searched joint
    // Newton MLE (converges in a handful of iterations like survreg/lifelines),
    // then assemble the identical `UnifiedFitResult` so finalize / predict /
    // CRPS / the `offset_channel_geometry` consumer all work unchanged. Any
    // genuinely flexible or penalized survival LS fit keeps the full coupled
    // path below.
    let fit = if prepared.is_reduced_parametric_aft() {
        fit_reduced_parametric_aft(&prepared, &options)?
    } else {
        fit_custom_family(&prepared.family, &prepared.blockspecs, &options)?
    };
    // Defensive: `fit_custom_family`'s degraded-plan path (after an ARC
    // deterministic-replay stall) can return a fit whose `block_states`
    // were cleared. `offset_channel_geometry` already tolerates this case
    // (returns zero residuals), but `finalize_survival_location_scale_fit`
    // indexes `fit.block_states[BLOCK_TIME]` directly and would panic.
    // Surface a clear sentinel error here; the workflow-level fallback in
    // `fit_survival_location_scale_terms` matches on the sentinel and
    // substitutes zero offset-channel residuals so the outer baseline BFGS
    // can terminate at the current θ instead of propagating a hard panic up
    // to the Python wrapper.
    if fit.block_states.is_empty() {
        return Err(SurvivalLocationScaleError::InternalInvariant {
            reason: SURVIVAL_LOCATION_SCALE_EMPTY_BLOCK_STATES_MARKER.to_string(),
        }
        .into());
    }
    let (residuals, curvatures) =
        prepared.family.offset_channel_geometry(&fit.block_states)?;
    let link_param_data_fit_gradient = prepared
        .family
        .link_param_data_fit_gradient(&fit.block_states)?;
    let finalized = finalize_survival_location_scale_fit(&prepared, &fit)?;
    Ok((
        finalized,
        (residuals, curvatures, link_param_data_fit_gradient),
    ))
}

/// Converged-fit geometry returned alongside the finalized location-scale fit:
/// the offset-channel residuals + curvatures (for the baseline-θ gradient/Hessian)
/// and the exact inverse-link data-fit θ-gradient (`None` when the link has no
/// free parameters).
type SurvivalLocationScaleConvergedGeometry = (
    OffsetChannelResiduals,
    OffsetChannelCurvatures,
    Option<Array1<f64>>,
);


/// Sentinel error string surfaced by `fit_survival_location_scale_with_geometry`
/// when a degraded inner fit returns empty `block_states`. See the matching
/// `match` arm in `fit_survival_location_scale_terms`.
pub(crate) const SURVIVAL_LOCATION_SCALE_EMPTY_BLOCK_STATES_MARKER: &str = "fit_survival_location_scale_with_geometry: fit_custom_family \
     returned a fit with empty block_states (likely ARC \
     deterministic-replay stall in the inner outer solve); \
     cannot finalize this fit";


pub(crate) fn select_survival_link_wiggle_basis_from_pilot(
    pilot: &SurvivalLocationScaleTermFitResult,
    wiggle_cfg: &WiggleBlockConfig,
    wiggle_penalty_orders: &[usize],
) -> Result<SelectedWiggleBasis, String> {
    let eta_threshold = pilot
        .threshold_design
        .design
        .dot(&pilot.fit.beta_threshold());
    let eta_log_sigma = pilot
        .log_sigma_design
        .design
        .dot(&pilot.fit.beta_log_sigma());
    let q_seed = Array1::from_iter(
        eta_threshold
            .iter()
            .zip(eta_log_sigma.iter())
            .map(|(&threshold, &ls)| survival_q0_from_eta(threshold, ls)),
    );
    select_wiggle_basis_from_seed(q_seed.view(), wiggle_cfg, wiggle_penalty_orders)
}


fn linkwiggle_block_input_from_selected_basis(
    selected_wiggle_basis: SelectedWiggleBasis,
) -> LinkWiggleBlockInput {
    let crate::families::wiggle::SelectedWiggleBasis {
        block,
        knots,
        degree,
        ..
    } = selected_wiggle_basis;
    let crate::families::parameter_block::ParameterBlockInput {
        design,
        penalties,
        nullspace_dims,
        initial_log_lambdas,
        initial_beta,
        ..
    } = block;
    LinkWiggleBlockInput {
        design,
        knots,
        degree,
        penalties,
        nullspace_dims,
        initial_log_lambdas,
        initial_beta,
    }
}


pub(crate) fn fit_survival_location_scale_terms_with_selected_wiggle(
    data: ndarray::ArrayView2<'_, f64>,
    mut spec: SurvivalLocationScaleTermSpec,
    selected_wiggle_basis: SelectedWiggleBasis,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<SurvivalLocationScaleTermFitResult, String> {
    spec.linkwiggle_block = Some(linkwiggle_block_input_from_selected_basis(
        selected_wiggle_basis,
    ));
    fit_survival_location_scale_terms(data, spec, kappa_options)
}


pub(crate) fn fit_survival_location_scale_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: SurvivalLocationScaleTermSpec,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<SurvivalLocationScaleTermFitResult, String> {
    let threshold_boot_design =
        build_term_collection_design(data, &spec.thresholdspec).map_err(|e| e.to_string())?;
    let log_sigma_boot_design =
        build_term_collection_design(data, &spec.log_sigmaspec).map_err(|e| e.to_string())?;
    let threshold_bootspec =
        freeze_term_collection_from_design(&spec.thresholdspec, &threshold_boot_design)
            .map_err(|e| e.to_string())?;
    let log_sigma_bootspec =
        freeze_term_collection_from_design(&spec.log_sigmaspec, &log_sigma_boot_design)
            .map_err(|e| e.to_string())?;

    let threshold_boot_derivs = build_survival_covariate_block_psi_derivatives(
        data,
        &threshold_bootspec,
        &threshold_boot_design,
        &spec.threshold_template,
    )?;
    let log_sigma_boot_derivs = build_survival_covariate_block_psi_derivatives(
        data,
        &log_sigma_bootspec,
        &log_sigma_boot_design,
        &spec.log_sigma_template,
    )?;
    let analytic_joint_gradient_available =
        threshold_boot_derivs.is_some() && log_sigma_boot_derivs.is_some();
    let analytic_joint_hessian_available = threshold_boot_derivs
        .as_ref()
        .is_some_and(|derivs| survival_psi_derivatives_support_exact_joint_hessian(derivs))
        && log_sigma_boot_derivs
            .as_ref()
            .is_some_and(|derivs| survival_psi_derivatives_support_exact_joint_hessian(derivs));

    let wiggle_rho0 = spec
        .linkwiggle_block
        .as_ref()
        .and_then(|w| w.initial_log_lambdas.clone())
        .unwrap_or_else(|| Array1::zeros(0));
    // Outer time-warp ρ count. In the reduced constant-scale-AFT regime the
    // time block collapses to its unpenalized affine null space (see
    // `prepare_identified_time_block`), so it carries NO smoothing parameter and
    // must contribute no ρ coordinate to the outer REML search — otherwise the
    // outer optimizer spends a full inner blockwise fit per step crawling a
    // dead-flat time-smoothing dimension until `outer_max_iter` (issue
    // #736/#735/#721). `survival_time_rho_count` is the single source of truth
    // shared with the inner block preparation so the two layouts always agree.
    let constant_scale = log_sigma_boot_design.penalties.is_empty();
    let protected_timewiggle_cols = spec.timewiggle_block.as_ref().map_or(0, |w| w.ncols);
    let k_time = survival_time_rho_count(
        &spec.time_block.penalties,
        spec.time_block.design_exit.ncols(),
        constant_scale,
        protected_timewiggle_cols,
    );
    let time_rho0 = if k_time == 0 {
        // Reduced parametric AFT: the time block is unpenalized, so any caller-
        // supplied per-penalty time seed is irrelevant and the outer search
        // carries no time coordinate.
        Array1::<f64>::zeros(0)
    } else {
        spec.time_block
            .initial_log_lambdas
            .clone()
            .unwrap_or_else(|| Array1::zeros(k_time))
    };
    // Reduced parametric-AFT regime (issue #736/#735/#721): when the location
    // (and scale) carry only full-rank parametric shrinkage ridges
    // (`nullspace_dim == 0`, e.g. the linear-term `LinearTermRidge` on `age`)
    // and the time-warp has reduced to affine with no wiggle, those ridges are
    // dropped — the inner `prepare_survival_location_scale_model` applies the
    // IDENTICAL predicate to the same boot-design penalties, so the inner and
    // outer ρ counts stay provably in lock-step. Dropping them takes the
    // threshold/log_sigma ρ counts to 0, so the outer search carries ZERO
    // coordinates and the fit is a single direct unpenalized parametric-AFT MLE
    // (`fit_parametric_aft_direct_mle`) — milliseconds, and numerically the
    // `survreg`/`lifelines` MLE — instead of crawling a flat, vacuous ρ surface.
    let drop_parametric_ridges = survival_reduced_parametric_aft_regime(
        &spec.time_block.penalties,
        spec.time_block.design_exit.ncols(),
        constant_scale,
        protected_timewiggle_cols,
        &threshold_boot_design.nullspace_dims,
        threshold_boot_design.penalties.len(),
        &log_sigma_boot_design.nullspace_dims,
        log_sigma_boot_design.penalties.len(),
        spec.linkwiggle_block.is_some(),
    );
    let layout = SurvivalLambdaLayout::new(
        k_time,
        if drop_parametric_ridges {
            0
        } else {
            threshold_boot_design.penalties.len()
        },
        if drop_parametric_ridges {
            0
        } else {
            log_sigma_boot_design.penalties.len()
        },
        wiggle_rho0.len(),
    );
    let mut rho0 = Array1::<f64>::zeros(layout.total());
    if layout.k_time > 0 {
        if time_rho0.len() != layout.k_time {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival time initial_log_lambdas length mismatch: got {}, expected {}",
                    time_rho0.len(),
                    layout.k_time
                ),
            }
            .into());
        }
        let range = layout.time_range();
        rho0.slice_mut(s![range.start..range.end])
            .assign(&time_rho0);

        // Parametric-AFT regime: strong-smoothing seed for the time-warp
        // penalty.
        //
        // When the scale block carries no penalties (a single constant σ) the
        // residual distribution `z = (h(t) - η)/σ` is a fixed parametric shape
        // with a single global spread, so the data identifies the baseline
        // *only* through the affine `1 + log t` transform that IS the parametric
        // AFT transform. The flexible deviation of the monotone I-spline
        // time-warp `h(t)` away from its penalty nullspace (that affine
        // baseline) is then statistically unidentified, and the REML/LAML
        // profile in the time smoothing parameter is a long flat ridge that
        // climbs monotonically toward strong smoothing.
        //
        // This unidentifiability is a property of the SCALE block alone, not of
        // the mean. A smooth mean `~ s(z)` adds flexibility in *covariate*
        // space — it bends η as a function of the covariates — but it carries no
        // information about the *time* baseline shape, because the time-warp
        // enters only through `h(t)` and is identified solely by how the event
        // times distribute against a single global σ. So whether the mean is
        // rigid (`~ age`) or smooth (`~ s(z)`), a constant-scale Gaussian AFT
        // leaves the time-warp's non-affine deviation unidentified and the time
        // ridge flat. Gating the seed on `rigid_mean` therefore wrongly excluded
        // the smooth-mean constant-scale case (#735), whose threshold block
        // carries penalties: it fell through to the weak default time seed and
        // its exact-joint outer search crawled the flat time ridge forever.
        //
        // Seeding the weak default (`time_smooth_lambda ≈ 1e-2`) drops the
        // inner REML search into the *interior* of that ridge, where it crawls
        // toward the strong-smoothing boundary one short, ill-conditioned step
        // at a time and never terminates in reasonable time (#736, #735, #721).
        //
        // The previous fix seeded the *interior* point ρ = 8. That did NOT cure
        // the hang: the inner blockwise REML optimizer re-optimizes ρ_time
        // freely from its seed against an inner per-coordinate ρ box bound of
        // ±10 (`fit_custom_family_with_rho_prior`'s `.with_rho_bound(10.0)`).
        // λ = exp(8) ≈ 3·10³ already sits INSIDE the "dead-flat region" that
        // very bound exists to fence off (see the `with_rho_bound` rationale in
        // `custom_family.rs`): with a flat REML gradient and near-singular
        // curvature there, the optimizer wanders between ρ = 8 and the ρ = 10
        // boundary one micro-step at a time and the retry-stall detector spins
        // on the flat surface — producing the >200s no-iteration-log hang. A
        // seed strictly interior to the box can never certify, because the
        // unconstrained projected-gradient stationarity test it would need is
        // exactly the test the flat ridge makes ill-posed.
        //
        // Seed instead at the inner ρ box bound itself. At the bound the
        // box-constraint KKT condition (the REML gradient pushes further into
        // strong smoothing, against an active bound) certifies stationarity
        // *immediately* at iteration 0 for the time coordinate — there is no
        // interior flat region left to wander, because the optimizer is pinned
        // at the wall. λ = exp(10) ≈ 22k is the affine-nullspace limit (the
        // bound's own rationale calls this "statistically indistinguishable
        // from shrunk to nullspace"), i.e. exactly the parametric-AFT affine
        // baseline. This is a regime-specific *initialization*, not a cap or a
        // tolerance change: the I-spline basis dimensions are untouched, so any
        // independent rebuild of the time basis (predictor reconstruction) is
        // unaffected, and a genuinely flexible regime never reaches this branch.
        //
        // The seed is gated on `constant_scale` ONLY. The genuinely flexible
        // time-warp regime is a smooth scale (`noise_formula = s(...)`): a
        // varying σ lets the residual spread change with covariates, which DOES
        // supply identifying information for a non-affine baseline, so those
        // fits carry log_sigma penalties and keep the full weak-seed search.
        // Smooth-mean penalties on the threshold block are still selected
        // normally — only the TIME-WARP block's seed changes here.
        //
        // NOTE: reaching here with `constant_scale == true` already implies the
        // affine reduction did NOT fire (otherwise `k_time == 0` and this whole
        // `if layout.k_time > 0` arm is skipped — the reduced block is
        // unpenalized and carries no ρ at all). This seed therefore only covers
        // the residual constant-scale case where the time penalty has no affine
        // null space to collapse onto (or a timewiggle keeps the flexibility),
        // pinning that surviving time ρ at the strong-smoothing limit.
        if constant_scale {
            // ρ = 10 == the inner blockwise solver's per-coordinate ρ box bound
            // (`custom_family.rs` `with_rho_bound(10.0)`). Seeding AT the bound
            // (not interior, as the prior ρ = 8 seed did) makes the box
            // constraint active from iteration 0, so outer stationarity
            // certifies immediately instead of crawling the flat ridge.
            const PARAMETRIC_AFT_TIME_RHO_SEED: f64 = 10.0;
            let mut time_seed = rho0.slice_mut(s![range.start..range.end]);
            for v in time_seed.iter_mut() {
                *v = PARAMETRIC_AFT_TIME_RHO_SEED;
            }
        }
    }
    // Warm-start: inject converged ρ seeds from a previous fit if supplied. The values are
    // clamped to the outer ρ bounds (±12) so that "dead" coordinates returned at extremes
    // by a prior fit don't crowd the optimizer's box bound on the next probe.
    if layout.k_threshold > 0
        && let Some(seed) = spec.initial_threshold_log_lambdas.as_ref()
    {
        if seed.len() != layout.k_threshold {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival threshold initial_log_lambdas length mismatch: got {}, expected {}",
                    seed.len(),
                    layout.k_threshold
                ),
            }
            .into());
        }
        let range = layout.threshold_range();
        let mut slice = rho0.slice_mut(s![range.start..range.end]);
        for (dst, src) in slice.iter_mut().zip(seed.iter()) {
            if src.is_finite() {
                *dst = src.clamp(-12.0, 12.0);
            }
        }
    }
    if layout.k_log_sigma > 0
        && let Some(seed) = spec.initial_log_sigma_log_lambdas.as_ref()
    {
        if seed.len() != layout.k_log_sigma {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival log_sigma initial_log_lambdas length mismatch: got {}, expected {}",
                    seed.len(),
                    layout.k_log_sigma
                ),
            }
            .into());
        }
        let range = layout.log_sigma_range();
        let mut slice = rho0.slice_mut(s![range.start..range.end]);
        for (dst, src) in slice.iter_mut().zip(seed.iter()) {
            if src.is_finite() {
                *dst = src.clamp(-12.0, 12.0);
            }
        }
    }
    if layout.k_wiggle > 0 {
        let range = layout.wiggle_range();
        rho0.slice_mut(s![range.start..range.end])
            .assign(&wiggle_rho0);
    }
    let joint_setup = build_survival_two_block_exact_joint_setup(
        data.view(),
        &spec.thresholdspec,
        &spec.log_sigmaspec,
        rho0,
        kappa_options,
    );

    let time_beta_hint = std::cell::RefCell::new(spec.time_block.initial_beta.clone());
    let threshold_beta_hint = std::cell::RefCell::new(None::<Array1<f64>>);
    let log_sigma_beta_hint = std::cell::RefCell::new(None::<Array1<f64>>);
    let wiggle_beta_hint = std::cell::RefCell::new(
        spec.linkwiggle_block
            .as_ref()
            .and_then(|w| w.initial_beta.clone()),
    );
    let exact_warm_start = std::cell::RefCell::new(None::<CustomFamilyWarmStart>);
    // Outer ρ-cache β-seed staging slot. See BMS/SMS for the contract: stash
    // the flat β here on cache hit, promote to a real `CustomFamilyWarmStart`
    // once per-block widths are known from `prepare_survival_location_scale_model`.
    let pending_beta_seed = std::cell::RefCell::new(None::<Array1<f64>>);
    // Stash the geometry from the most recent inner fit. Updated on every
    // value-closure call by the spatial optimizer; the last one written
    // corresponds to the converged outer point. This avoids redoing
    // `prepare_survival_location_scale_model` + a second fit pass after the
    // optimizer returns, and (critically) avoids the post-finalize
    // `block_states` wipe that would make the geometry call error out.
    let last_geometry: std::cell::RefCell<
        Option<SurvivalLocationScaleConvergedGeometry>,
    > = std::cell::RefCell::new(None);

    let build_spec = |rho: &Array1<f64>,
                      _: &TermCollectionSpec,
                      _: &TermCollectionSpec,
                      threshold_design: &TermCollectionDesign,
                      log_sigma_design: &TermCollectionDesign|
     -> Result<SurvivalLocationScaleSpec, String> {
        layout.validate_rho(rho, "survival term fit")?;
        let time_beta = filtered_initial_beta(
            time_beta_hint.borrow().as_ref(),
            spec.time_block.design_exit.ncols(),
        );
        // In the reduced parametric-AFT regime the layout carries no
        // threshold/log_sigma ρ (`drop_parametric_ridges`), yet the boot design
        // still carries the parametric ridge as a penalty. Passing the empty
        // layout slice as the seed would mismatch that penalty count; instead
        // pass `None` so the block defaults to a length-matched zero seed, which
        // the inner `prepare_survival_location_scale_model` then drops along with
        // the ridge. Outside the regime the layout slice length equals the
        // design penalty count, so `Some(slice)` is exact.
        let threshold_block = build_survival_covariate_block_from_design(
            threshold_design,
            &spec.threshold_template,
            &spec.threshold_offset,
            if drop_parametric_ridges {
                None
            } else {
                Some(layout.threshold_from(rho))
            },
            filtered_initial_beta(
                threshold_beta_hint.borrow().as_ref(),
                match &spec.threshold_template {
                    SurvivalCovariateTermBlockTemplate::Static => threshold_design.design.ncols(),
                    SurvivalCovariateTermBlockTemplate::TimeVarying {
                        time_basis_exit, ..
                    } => threshold_design.design.ncols() * time_basis_exit.ncols(),
                },
            ),
        )?;
        let log_sigma_block = build_survival_covariate_block_from_design(
            log_sigma_design,
            &spec.log_sigma_template,
            &spec.log_sigma_offset,
            if drop_parametric_ridges {
                None
            } else {
                Some(layout.log_sigma_from(rho))
            },
            filtered_initial_beta(
                log_sigma_beta_hint.borrow().as_ref(),
                match &spec.log_sigma_template {
                    SurvivalCovariateTermBlockTemplate::Static => log_sigma_design.design.ncols(),
                    SurvivalCovariateTermBlockTemplate::TimeVarying {
                        time_basis_exit, ..
                    } => log_sigma_design.design.ncols() * time_basis_exit.ncols(),
                },
            ),
        )?;
        let linkwiggle_block = spec
            .linkwiggle_block
            .as_ref()
            .map(|wiggle| LinkWiggleBlockInput {
                design: wiggle.design.clone(),
                knots: wiggle.knots.clone(),
                degree: wiggle.degree,
                penalties: wiggle.penalties.clone(),
                nullspace_dims: wiggle.nullspace_dims.clone(),
                initial_log_lambdas: layout.wiggle_from(rho),
                initial_beta: filtered_initial_beta(
                    wiggle_beta_hint.borrow().as_ref(),
                    wiggle.design.ncols(),
                ),
            });
        Ok(SurvivalLocationScaleSpec {
            age_entry: spec.age_entry.clone(),
            age_exit: spec.age_exit.clone(),
            event_target: spec.event_target.clone(),
            weights: spec.weights.clone(),
            inverse_link: spec.inverse_link.clone(),
            derivative_guard: spec.derivative_guard,
            max_iter: spec.max_iter,
            tol: spec.tol,
            time_block: TimeBlockInput {
                design_entry: spec.time_block.design_entry.clone(),
                design_exit: spec.time_block.design_exit.clone(),
                design_derivative_exit: spec.time_block.design_derivative_exit.clone(),
                offset_entry: spec.time_block.offset_entry.clone(),
                offset_exit: spec.time_block.offset_exit.clone(),
                derivative_offset_exit: spec.time_block.derivative_offset_exit.clone(),
                time_monotonicity: spec.time_block.time_monotonicity,
                penalties: spec.time_block.penalties.clone(),
                nullspace_dims: spec.time_block.nullspace_dims.clone(),
                // `initial_log_lambdas` is the per-penalty seed for THIS block's
                // (still un-reduced) `penalties`, validated against that list's
                // length by `validate_time_block`. In the flexible regime the
                // outer layout carries one time ρ per penalty, so `time_from`
                // returns exactly `penalties.len()` entries. In the reduced
                // constant-scale-AFT regime (`layout.k_time == 0`) the outer
                // search carries NO time coordinate, so `time_from` is empty —
                // but `penalties` here is the un-reduced length-`k` list (the
                // collapse to the unpenalized affine null space happens later,
                // inside `prepare_identified_time_block`). Emitting the empty
                // outer slice against the un-reduced penalties would make
                // `initial_log_lambdas.len() (0) != penalties.len() (k)` and
                // trip the block's length-consistency check. The downstream
                // reduction re-derives (and drops) this seed for the collapsed
                // block, so any length-`k` value is fine here; carry the
                // caller's original per-penalty seed to stay length-consistent
                // with the un-reduced penalty list (issue #736/#735/#721).
                initial_log_lambdas: if layout.k_time > 0 {
                    Some(layout.time_from(rho))
                } else {
                    spec.time_block.initial_log_lambdas.clone()
                },
                initial_beta: time_beta,
            },
            threshold_block,
            log_sigma_block,
            timewiggle_block: spec.timewiggle_block.clone(),
            linkwiggle_block,
            cache_session: spec.cache_session.clone(),
            cache_mirror_sessions: spec.cache_mirror_sessions.clone(),
        })
    };

    let threshold_terms = spatial_length_scale_term_indices(&spec.thresholdspec);
    let log_sigma_terms = spatial_length_scale_term_indices(&spec.log_sigmaspec);
    // Survival location-scale is a multi-block family with β-dependent
    // joint Hessian: disable EFS/HybridEFS at plan time so the outer never
    // pays for a stalled fixed-point attempt before landing on BFGS.
    let outer_policy = {
        let capability = if analytic_joint_hessian_available {
            crate::families::custom_family::ExactOuterDerivativeOrder::Second
        } else {
            crate::families::custom_family::ExactOuterDerivativeOrder::First
        };
        // Honest per-eval work model so the route planner has a real cost
        // signal for the exact gradient / joint-Hessian routes (#721). The
        // survival likelihood couples every block, so a single coefficient
        // Hessian assembly costs `n · (Σ p_b)²` (matching
        // `joint_coupled_coefficient_hessian_cost`), and each outer
        // coordinate (every penalty ρ, spatial log-κ, and auxiliary axis)
        // propagates one analytic directional derivative through the inner
        // solve. Leaving these at 0 left the planner blind and it never
        // down-routed the heavyweight exact-joint path.
        let n_work = data.nrows() as u64;
        let p_total = (spec.time_block.design_exit.ncols()
            + threshold_boot_design.design.ncols()
            + log_sigma_boot_design.design.ncols()
            + spec
                .linkwiggle_block
                .as_ref()
                .map_or(0, |w| w.design.ncols())) as u64;
        let hess_cost = n_work.saturating_mul(p_total.saturating_mul(p_total));
        let grad_cost = hess_cost / 2;
        let outer_coords =
            (joint_setup.rho_dim() + joint_setup.log_kappa_dim() + joint_setup.auxiliary_dim())
                .max(1) as u128;
        let predicted_hessian_work = (hess_cost as u128).saturating_mul(outer_coords);
        let predicted_gradient_work = (grad_cost as u128).saturating_mul(outer_coords);
        crate::families::custom_family::OuterDerivativePolicy {
            capability,
            predicted_gradient_work,
            predicted_hessian_work,
            // Survival location-scale consumes `outer_score_subsample` on its
            // outer-only LL, joint-Hessian, and ψ workspace paths.
            subsample_capable: true,
        }
    };
    let solved = optimize_spatial_length_scale_exact_joint(
        data,
        &[spec.thresholdspec.clone(), spec.log_sigmaspec.clone()],
        &[threshold_terms, log_sigma_terms],
        kappa_options,
        &joint_setup,
        crate::seeding::SeedRiskProfile::Survival,
        analytic_joint_gradient_available,
        analytic_joint_hessian_available,
        true,
        None,
        outer_policy,
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let (fit, geom) = fit_survival_location_scale_with_geometry(build_spec(
                &rho,
                &specs[0],
                &specs[1],
                &designs[0],
                &designs[1],
            )?)?;
            time_beta_hint.replace(Some(fit.beta_time()));
            threshold_beta_hint.replace(Some(fit.beta_threshold()));
            log_sigma_beta_hint.replace(Some(fit.beta_log_sigma()));
            wiggle_beta_hint.replace(fit.beta_link_wiggle());
            *last_geometry.borrow_mut() = Some(geom);
            Ok(fit)
        },
        |theta,
         specs: &[TermCollectionSpec],
         designs: &[TermCollectionDesign],
         eval_mode,
         row_set: &crate::families::row_kernel::RowSet| {
            use crate::solver::estimate::reml::unified::EvalMode;
            if !analytic_joint_gradient_available {
                return Err(SurvivalLocationScaleError::InvalidConfiguration { reason: "analytic spatial psi derivatives are unavailable for survival exact two-block path"
                        .to_string(), }.into());
            }
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let assembled = build_spec(&rho, &specs[0], &specs[1], &designs[0], &designs[1])?;
            let prepared = prepare_survival_location_scale_model(&assembled)?;
            if let Some(beta_seed) = pending_beta_seed.borrow_mut().take() {
                let widths: Vec<usize> = prepared
                    .blockspecs
                    .iter()
                    .map(|b| b.design.ncols())
                    .collect();
                match CustomFamilyWarmStart::from_cached_beta(&widths, &beta_seed) {
                    Ok(ws) => {
                        exact_warm_start.replace(Some(ws));
                    }
                    Err(e) => {
                        log::warn!(
                            "[survival-LS] outer ρ-cache β-warm-start rejected: {e}; falling back to cold β"
                        );
                    }
                }
            }
            let threshold_derivs = build_survival_covariate_block_psi_derivatives(
                data,
                &specs[0],
                &designs[0],
                &spec.threshold_template,
            )?
            .ok_or_else(|| "missing survival threshold spatial psi derivatives".to_string())?;
            let log_sigma_derivs = build_survival_covariate_block_psi_derivatives(
                data,
                &specs[1],
                &designs[1],
                &spec.log_sigma_template,
            )?
            .ok_or_else(|| "missing survival log-sigma spatial psi derivatives".to_string())?;
            let mut derivative_blocks = vec![Vec::new(), threshold_derivs, log_sigma_derivs];
            if prepared.family.x_link_wiggle.is_some() {
                derivative_blocks.push(Vec::new());
            }
            // If the caller asked for a Hessian but the family can't provide
            // an analytic one, downgrade the request to ValueAndGradient.
            // ValueOnly stays ValueOnly so cost-only line-search probes skip
            // gradient assembly entirely.
            let effective_mode = match eval_mode {
                EvalMode::ValueGradientHessian if !analytic_joint_hessian_available => {
                    EvalMode::ValueAndGradient
                }
                other => other,
            };
            let mut eval_options = survival_blockwise_fit_options(&assembled);
            match row_set {
                crate::families::row_kernel::RowSet::All => {}
                crate::families::row_kernel::RowSet::Subsample { rows, n_full } => {
                    eval_options.outer_score_subsample = Some(Arc::new(
                        crate::families::marginal_slope_shared::OuterScoreSubsample::from_weighted_rows(
                            (**rows).clone(),
                            *n_full,
                            *n_full as u64,
                        ),
                    ));
                }
            }
            let eval = evaluate_custom_family_joint_hyper(
                &prepared.family,
                &prepared.blockspecs,
                &eval_options,
                &rho,
                &derivative_blocks,
                exact_warm_start.borrow().as_ref(),
                effective_mode,
            )
            .map_err(|e| e.to_string())?;
            exact_warm_start.replace(Some(eval.warm_start.clone()));
            if !eval.inner_converged {
                return Err(
                    "survival location-scale exact joint inner solve did not converge".to_string(),
                );
            }
            Ok((eval.objective, eval.gradient, eval.outer_hessian))
        },
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            if !analytic_joint_gradient_available {
                return Err(SurvivalLocationScaleError::InvalidConfiguration { reason: "analytic spatial psi derivatives are unavailable for survival exact two-block path"
                        .to_string(), }.into());
            }
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let assembled = build_spec(&rho, &specs[0], &specs[1], &designs[0], &designs[1])?;
            let prepared = prepare_survival_location_scale_model(&assembled)?;
            if let Some(beta_seed) = pending_beta_seed.borrow_mut().take() {
                let widths: Vec<usize> = prepared
                    .blockspecs
                    .iter()
                    .map(|b| b.design.ncols())
                    .collect();
                match CustomFamilyWarmStart::from_cached_beta(&widths, &beta_seed) {
                    Ok(ws) => {
                        exact_warm_start.replace(Some(ws));
                    }
                    Err(e) => {
                        log::warn!(
                            "[survival-LS] outer ρ-cache β-warm-start rejected (efs): {e}; falling back to cold β"
                        );
                    }
                }
            }
            let threshold_derivs = build_survival_covariate_block_psi_derivatives(
                data,
                &specs[0],
                &designs[0],
                &spec.threshold_template,
            )?
            .ok_or_else(|| "missing survival threshold spatial psi derivatives".to_string())?;
            let log_sigma_derivs = build_survival_covariate_block_psi_derivatives(
                data,
                &specs[1],
                &designs[1],
                &spec.log_sigma_template,
            )?
            .ok_or_else(|| "missing survival log-sigma spatial psi derivatives".to_string())?;
            let mut derivative_blocks = vec![Vec::new(), threshold_derivs, log_sigma_derivs];
            if prepared.family.x_link_wiggle.is_some() {
                derivative_blocks.push(Vec::new());
            }
            let eval = evaluate_custom_family_joint_hyper_efs(
                &prepared.family,
                &prepared.blockspecs,
                &survival_blockwise_fit_options(&assembled),
                &rho,
                &derivative_blocks,
                exact_warm_start.borrow().as_ref(),
            )
            .map_err(|e| e.to_string())?;
            exact_warm_start.replace(Some(eval.warm_start.clone()));
            if !eval.inner_converged {
                return Err(
                    "survival location-scale exact joint EFS inner solve did not converge"
                        .to_string(),
                );
            }
            Ok(eval.efs_eval)
        },
        crate::families::marginal_slope_shared::make_beta_seed_validator(&pending_beta_seed),
    )?;

    let mut resolved_specs = solved.resolved_specs;
    let mut designs = solved.designs;
    // Fast path: the value closure stashed the offset geometry from the
    // *converged* inner fit (computed pre-finalize while `block_states` was
    // still populated). No extra family prep / refit needed here.
    //
    // Fallback: if for some reason no value-closure call ran (or the
    // optimizer's last evaluation happened through the gradient/EFS path
    // without touching the value closure at the final ρ), recompute by
    // redoing one inner fit at the final ρ̂. This pays an extra fit only when
    // the cache is cold — the common location-scale path always populates it.
    let (baseline_offset_residuals, baseline_offset_curvatures, link_param_data_fit_gradient) =
        match last_geometry.borrow_mut().take() {
            Some(geom) => geom,
            None => {
                let rho_final = solved.fit.log_lambdas.clone();
                let final_assembled = build_spec(
                    &rho_final,
                    &resolved_specs[0],
                    &resolved_specs[1],
                    &designs[0],
                    &designs[1],
                )?;
                match fit_survival_location_scale_with_geometry(final_assembled) {
                    Ok((_refit, geom)) => geom,
                    // Degraded-fit fallback: when the refit's inner ARC stalled
                    // into deterministic-replay and produced an empty
                    // `block_states`, `fit_survival_location_scale_with_geometry`
                    // surfaces a sentinel error rather than letting
                    // `finalize_survival_location_scale_fit` panic on the
                    // out-of-bounds `block_states[BLOCK_TIME]` index. Substitute
                    // zero offset-channel residuals so the outer baseline BFGS
                    // sees `‖g‖ = 0` at this candidate and terminates cleanly
                    // at the current θ̂ instead of propagating a hard panic up
                    // through the Python wrapper. Production loss is at most a
                    // slightly suboptimal baseline θ at this BMA parent-set —
                    // strictly better than a `SurvivalLocationScaleFamily
                    // expects 3 blocks, got 0` exception killing the whole
                    // gamfit.fit() call.
                    Err(e)
                        if e == SURVIVAL_LOCATION_SCALE_EMPTY_BLOCK_STATES_MARKER
                            || e.contains("expects 3 blocks, got 0")
                            || e.contains("expects 4 blocks, got 0")
                            || (e.contains("blockwise fit requires at least one block state"))
                            || (e.contains("block_states") && e.contains("got 0")) =>
                    {
                        // Broadened catch: any error indicating an empty
                        // block_states slate maps to the same "degraded BMA
                        // candidate" fallback. The original sentinel catches
                        // the path I instrumented in
                        // `fit_survival_location_scale_with_geometry`, but
                        // `validate_joint_states` and `blockwise_fit_from_parts`
                        // produce structurally similar errors from the ~7 other
                        // call sites of `build_dynamic_geometry` (Hessian
                        // operator routes, EFS fixed-point, etc.) which can also
                        // surface when an ARC deterministic-replay stall
                        // collapses the inner refit's block layout.
                        // Substituting zero residuals here lets the outer
                        // baseline-θ BFGS see `‖g‖ = 0` at the bad candidate
                        // and terminate cleanly rather than panicking the
                        // whole `gam.fit()` call.
                        log::warn!(
                            "fit_survival_location_scale_terms: refit at converged ρ̂ \
                             produced empty block_states ({e}); substituting zero offset \
                             residuals (degraded BMA candidate; outer θ-BFGS will \
                             terminate at this point)"
                        );
                        let n = data.nrows();
                        // Degraded candidate: zero residuals and `None` link
                        // gradient → outer BFGS sees ‖g‖ = 0 and terminates at
                        // the current θ̂ rather than panicking.
                        (
                            OffsetChannelResiduals {
                                exit: Array1::<f64>::zeros(n),
                                entry: Array1::<f64>::zeros(n),
                                derivative: Array1::<f64>::zeros(n),
                                // Location-scale has no interval upper-bound channel.
                                right: Array1::<f64>::zeros(n),
                            },
                            OffsetChannelCurvatures {
                                rows: vec![[[0.0_f64; 3]; 3]; n],
                            },
                            None,
                        )
                    }
                    Err(e) => return Err(e),
                }
            }
        };
    Ok(SurvivalLocationScaleTermFitResult {
        fit: solved.fit,
        resolved_thresholdspec: resolved_specs.remove(0),
        resolved_log_sigmaspec: resolved_specs.remove(0),
        threshold_design: designs.remove(0),
        log_sigma_design: designs.remove(0),
        baseline_offset_residuals,
        baseline_offset_curvatures,
        link_param_data_fit_gradient,
    })
}


pub fn predict_survival_location_scale(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
) -> Result<SurvivalLocationScalePredictResult, String> {
    let predictors = prediction_linear_predictors(input, fit)?;
    survival_location_scale_response_from_predictors(&input.inverse_link, predictors)
}


fn survival_location_scale_response_from_predictors(
    inverse_link: &InverseLink,
    predictors: PredictionLinearPredictors,
) -> Result<SurvivalLocationScalePredictResult, String> {
    use ndarray::Zip;

    let n = predictors.h.len();
    let mut eta = Array1::<f64>::zeros(n);
    match predictors.etaw.as_ref() {
        Some(etaw) => Zip::from(&mut eta)
            .and(&predictors.h)
            .and(&predictors.eta_t)
            .and(&predictors.inv_sigma)
            .and(etaw)
            .par_for_each(|q, &hh, &tt, &r, &w| {
                *q = hh - tt * r + w;
            }),
        None => Zip::from(&mut eta)
            .and(&predictors.h)
            .and(&predictors.eta_t)
            .and(&predictors.inv_sigma)
            .par_for_each(|q, &hh, &tt, &r| {
                *q = hh - tt * r;
            }),
    }
    let survival_values: Result<Vec<f64>, SurvivalLocationScaleError> = {
        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
        eta.as_slice()
            .ok_or_else(|| {
                "predict_survival_location_scale: eta storage is not contiguous".to_string()
            })?
            .par_iter()
            .map(|&v| inverse_link_survival_prob_checked(inverse_link, v))
            .collect()
    };
    let survival_prob = Array1::from_vec(survival_values?);
    Ok(SurvivalLocationScalePredictResult { eta, survival_prob })
}


pub fn predict_survival_location_scale_posterior_mean(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
    covariance: &Array2<f64>,
) -> Result<SurvivalLocationScalePredictResult, String> {
    let pred = predict_survival_location_scale(input, fit)?;
    let (survival_prob, _) = exact_survival_response_moments(input, fit, covariance)?;

    Ok(SurvivalLocationScalePredictResult {
        eta: pred.eta,
        survival_prob,
    })
}


pub fn predict_survival_location_scalewith_uncertainty(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
    covariance: &Array2<f64>,
    posterior_mean: bool,
    include_response_sd: bool,
) -> Result<SurvivalLocationScalePredictUncertaintyResult, String> {
    let base = predict_survival_location_scale(input, fit)?;
    let n = input.x_time_exit.nrows();
    let p_time = fit.beta_time().len();
    let p_t = fit.beta_threshold().len();
    let p_ls = fit.beta_log_sigma().len();
    let beta_link_wiggle = fit.beta_link_wiggle();
    let pw = beta_link_wiggle.as_ref().map_or(0, |b| b.len());
    let resolved_wiggle_knots = input
        .link_wiggle_knots
        .as_ref()
        .or(fit.artifacts.survival_link_wiggle_knots.as_ref());
    let resolved_wiggle_degree = input
        .link_wiggle_degree
        .or(fit.artifacts.survival_link_wiggle_degree);
    let p_total = p_time + p_t + p_ls + pw;
    if covariance.nrows() != p_total || covariance.ncols() != p_total {
        return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
            "predict_survival_location_scalewith_uncertainty: covariance shape mismatch: got {}x{}, expected {}x{}",
            covariance.nrows(),
            covariance.ncols(),
            p_total,
            p_total
        ) }.into());
    }
    if pw > 0
        && (beta_link_wiggle.is_none()
            || resolved_wiggle_knots.is_none()
            || resolved_wiggle_degree.is_none())
    {
        return Err(SurvivalLocationScaleError::InvalidConfiguration { reason: "predict_survival_location_scalewith_uncertainty: dynamic link-wiggle metadata is incomplete"
                .to_string(), }.into());
    }

    let predictors = prediction_linear_predictors(input, fit)?;
    if input.x_threshold.nrows() != n || input.x_log_sigma.nrows() != n {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason:
                "predict_survival_location_scalewith_uncertainty: row mismatch across design views"
                    .to_string(),
        }
        .into());
    }
    let inv_sigma = &predictors.inv_sigma;
    let wiggle_design = predictors.wiggle_design.as_ref();
    let dq_dq0 = predictors.dq_dq0.as_ref();
    let x_t_dense = input.x_threshold.to_dense();
    let x_ls_dense = input.x_log_sigma.to_dense();
    let mut grad = Array2::<f64>::zeros((n, p_total));
    if p_total > 0 && n >= SURVIVAL_ROW_PARALLEL_THRESHOLD {
        let rows_per_chunk = SURVIVAL_ROW_PARALLEL_CHUNK;
        let chunk_len = rows_per_chunk * p_total;
        grad.as_slice_mut()
            .expect("fresh gradient matrix is contiguous")
            .par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(chunk_idx, grad_chunk)| {
                let row_start = chunk_idx * rows_per_chunk;
                for (local_row, row_grad) in grad_chunk.chunks_mut(p_total).enumerate() {
                    let i = row_start + local_row;
                    for j in 0..p_time {
                        row_grad[j] = predictors.time_jac[[i, j]];
                    }
                    let scale = dq_dq0.map_or(1.0, |v| v[i]);
                    for j in 0..p_t {
                        row_grad[p_time + j] = -scale * inv_sigma[i] * x_t_dense[[i, j]];
                    }
                    let coeff_ls = scale * predictors.eta_t[i] * inv_sigma[i];
                    for j in 0..p_ls {
                        row_grad[p_time + p_t + j] = coeff_ls * x_ls_dense[[i, j]];
                    }
                    if let Some(xw) = wiggle_design {
                        for j in 0..pw {
                            row_grad[p_time + p_t + p_ls + j] = xw[[i, j]];
                        }
                    }
                }
            });
    } else {
        for i in 0..n {
            for j in 0..p_time {
                grad[[i, j]] = predictors.time_jac[[i, j]];
            }
            let scale = dq_dq0.map_or(1.0, |v| v[i]);
            for j in 0..p_t {
                grad[[i, p_time + j]] = -scale * inv_sigma[i] * x_t_dense[[i, j]];
            }
            let coeff_ls = scale * predictors.eta_t[i] * inv_sigma[i];
            for j in 0..p_ls {
                grad[[i, p_time + p_t + j]] = coeff_ls * x_ls_dense[[i, j]];
            }
            if let Some(xw) = wiggle_design {
                for j in 0..pw {
                    grad[[i, p_time + p_t + p_ls + j]] = xw[[i, j]];
                }
            }
        }
    }
    let eta_se = linear_predictor_se(grad.view(), covariance);

    let exact_response_moments = if posterior_mean || include_response_sd {
        Some(exact_survival_response_moments(input, fit, covariance)?)
    } else {
        None
    };
    let posterior_mean_response = exact_response_moments
        .as_ref()
        .map(|(mean, _)| mean.clone());
    let posterior_second_moment = exact_response_moments
        .as_ref()
        .map(|(_, second)| second.clone());

    let survival_prob = if posterior_mean {
        posterior_mean_response
            .as_ref()
            .expect("posterior-mean path computes exact response moments")
            .clone()
    } else {
        base.survival_prob.clone()
    };

    let response_standard_error = if include_response_sd {
        let mean = posterior_mean_response
            .as_ref()
            .expect("response-sd path computes exact response moments");
        let second = posterior_second_moment
            .as_ref()
            .expect("response-sd path computes exact response moments");
        let mut sd = Array1::<f64>::zeros(n);
        if n >= SURVIVAL_ROW_PARALLEL_THRESHOLD {
            sd.as_slice_mut()
                .expect("fresh response standard-error array is contiguous")
                .par_chunks_mut(SURVIVAL_ROW_PARALLEL_CHUNK)
                .enumerate()
                .for_each(|(chunk_idx, sd_chunk)| {
                    let row_start = chunk_idx * SURVIVAL_ROW_PARALLEL_CHUNK;
                    for (offset, slot) in sd_chunk.iter_mut().enumerate() {
                        let i = row_start + offset;
                        *slot = (second[i] - mean[i] * mean[i]).max(0.0).sqrt();
                    }
                });
        } else {
            for i in 0..n {
                sd[i] = (second[i] - mean[i] * mean[i]).max(0.0).sqrt();
            }
        }
        Some(sd)
    } else {
        None
    };

    Ok(SurvivalLocationScalePredictUncertaintyResult {
        eta: base.eta,
        survival_prob,
        eta_standard_error: eta_se,
        response_standard_error,
    })
}

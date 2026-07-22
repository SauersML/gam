use super::*;
use gam_solve::gauge::Gauge;

pub(crate) fn survival_response_moment_block_ranges(
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

pub(crate) fn projected_survival_response_moment_covariance(
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

pub(crate) fn covariance3_to_array2(cov: [[f64; 3]; 3]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            out[[i, j]] = cov[i][j];
        }
    }
    out
}

pub(crate) fn symmetrize_and_clip_covariance(cov: &Array2<f64>) -> Array2<f64> {
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

pub(crate) struct LowRankGaussianFactor {
    pub(crate) factor: Array2<f64>,
    pub(crate) eigenvectors: Array2<f64>,
    pub(crate) inv_sqrt_eigenvalues: Array1<f64>,
}

// Exact projected-Gaussian handling for possibly singular covariance blocks.
// We integrate over the active standard-normal coordinates rather than adding
// jitter or inverting the covariance directly.
pub(crate) fn factorize_psd_covariance(
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

pub(crate) fn apply_low_rank_gaussian_factor3(
    mu: [f64; 3],
    factor: &Array2<f64>,
    z: &[f64],
) -> [f64; 3] {
    let mut x = mu;
    for row in 0..3 {
        for (col, &latent) in z.iter().enumerate() {
            x[row] += factor[[row, col]] * latent;
        }
    }
    x
}

pub(crate) fn low_rank_normal_expectation_pair_3d_result<F>(
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

/// Symmetric fraction-to-boundary factor for a realized displacement of a
/// `β ≥ 0` cone-constrained coefficient block (#2390, pattern from #2375).
///
/// Returns the largest `α ∈ [0, 1]` such that BOTH `β̂ + α·d` and `β̂ − α·d`
/// stay in the cone:
///
/// ```text
///   α = min( 1,  min_{j : d_j ≠ 0}  max(β̂_j, 0) / |d_j| )
/// ```
///
/// Depending only on `|d_j|` makes the factor sign-symmetric (`α(d) = α(−d)`),
/// so a symmetric quadrature rule displaced by `α·d` stays symmetric about `β̂`
/// and the posterior mean of every linear functional is exactly unbiased. A
/// coordinate already pinned at its wall (`β̂_j ≤ 0` from round-off, with
/// `d_j ≠ 0`) collapses the displacement to zero rather than admitting an
/// infeasible vector — the same convention as the #2375 survival cubature rule.
pub(crate) fn symmetric_cone_fraction_to_boundary(
    beta: ArrayView1<'_, f64>,
    displacement: ArrayView1<'_, f64>,
) -> f64 {
    let mut alpha = 1.0_f64;
    for (b, d) in beta.iter().zip(displacement.iter()) {
        if *d != 0.0 {
            alpha = alpha.min(b.max(0.0) / d.abs());
        }
    }
    alpha
}

/// `[0, ∞)`-truncated Gaussian expectation of a fallible pair integrand
/// (#2390 layer 2): `E[f(w) | w ≥ 0]` for `w ~ N(mean, sd²)`.
///
/// The feasible image of the monotone I-spline cone under a non-negative
/// basis row is exactly `w ≥ 0`, so the scalar link-wiggle integral must not
/// spend mass on `w < 0` — predictor values no feasible model produces. The
/// integral is computed through the truncated CDF map
/// `w(u) = mean + sd·Φ⁻¹(Φ(−mean/sd) + u·(1 − Φ(−mean/sd)))`, `u ∈ (0, 1)`:
/// the wall becomes the `u = 0` endpoint, the mapped integrand is smooth on
/// the open interval, and fixed-node Gauss–Legendre converges spectrally.
/// Far in the interior (`mean ≫ sd`) the truncated mass underflows and this
/// is the plain Gaussian expectation.
pub(crate) fn truncated_nonnegative_normal_expectation_pair(
    mean: f64,
    sd: f64,
    f: impl Fn(f64) -> Result<(f64, f64), String>,
) -> Result<(f64, f64), String> {
    if !(sd > 0.0) || !sd.is_finite() {
        return f(mean.max(0.0));
    }
    let p0 = gam_math::probability::normal_cdf(-mean / sd);
    let retained = 1.0 - p0;
    if retained <= 1.0e-300 {
        // The unconstrained Gaussian sits essentially entirely below the wall;
        // the truncated law concentrates at the wall itself.
        return f(0.0);
    }
    let (nodes, weights) = gam_math::special::gauss_legendre(32);
    let mut first = 0.0;
    let mut second = 0.0;
    for (t, wgt) in nodes.iter().zip(weights.iter()) {
        let u = 0.5 * (t + 1.0);
        let q = (p0 + u * retained).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        let z = gam_math::probability::standard_normal_quantile(q)
            .map_err(|e| format!("truncated-normal quantile at q={q}: {e}"))?;
        let w = (mean + sd * z).max(0.0);
        let (f1, f2) = f(w)?;
        // Gauss–Legendre on [-1,1] carries a Jacobian ½ into u-space; the
        // CDF map's own Jacobian is absorbed by construction (du IS the
        // truncated probability measure).
        first += 0.5 * wgt * f1;
        second += 0.5 * wgt * f2;
    }
    Ok((first, second))
}

// Exact response moments must stay in the original Gaussian coordinates:
// [h, threshold, log_sigma] for non-wiggle predictions, with a nested
// conditional Gaussian over the scalar link-wiggle contribution when present.
pub(crate) fn exact_survival_response_moments_row(
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
                let mut displacement = Array1::<f64>::zeros(pw);
                for j in 0..pw {
                    for (col, &latent) in z.iter().enumerate() {
                        displacement[j] += regression[[j, col]] * latent;
                    }
                }
                // #2390 (#2385 instance, pattern from #2375): `cond_mean` is a
                // REALIZED coefficient vector for the cone-constrained
                // link-wiggle block (`β_w ≥ 0`, the structural monotone
                // I-spline warp the fit certified). A cone coordinate pinned at
                // its wall has `β̂_w,j = 0` exactly, so an unconstrained
                // conditional displacement manufactures a warp the model does
                // not admit. Scale the displacement by the symmetric
                // fraction-to-boundary factor so every realized vector stays in
                // the cone; the factor depends only on `|d_j|`, so `α(z) =
                // α(−z)` and the rule stays symmetric about `β̂` (the posterior
                // mean of linear functionals stays exactly unbiased). An
                // interior `β̂` with modest spread yields `α = 1`, recovering
                // the unconstrained rule verbatim.
                let alpha = symmetric_cone_fraction_to_boundary(
                    beta_w.view(),
                    displacement.view(),
                );
                for j in 0..pw {
                    cond_mean[j] += alpha * displacement[j];
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
                // #2390 layer 2: the wiggle contribution's feasible image is
                // exactly `w ≥ 0` (non-negative I-spline basis row against the
                // β_w ≥ 0 cone), so the scalar integral runs over the
                // `[0, ∞)`-truncated conditional Gaussian — never over
                // predictor values no feasible model produces.
                truncated_nonnegative_normal_expectation_pair(
                    w_mean,
                    w_var.sqrt(),
                    |w| {
                        let p = inverse_link_survival_prob_checked(
                            &input.inverse_link,
                            x[0] + q0 + w,
                        )?;
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

pub(crate) fn exact_survival_response_moments(
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

/// Exact affine map from the fitted location-scale coefficient frame into the
/// saved/reporting frame.
///
/// The inner fit sees the reduced time block and the active tails of the
/// threshold and log-sigma blocks. Saved coefficients expand all three back to
/// their raw layouts. Keeping this as a single [`Gauge`] is essential: the
/// conditional covariance pushes forward through it, while the penalized
/// Hessian remains in the active frame and is paired with this map for exact
/// post-fit row-Jacobian pullback.
pub(crate) fn survival_location_scale_finalization_gauge(
    time_gauge: &Gauge,
    p_threshold_reduced: usize,
    p_threshold_full: usize,
    threshold_fixed_cols: usize,
    p_log_sigma_reduced: usize,
    p_log_sigma_full: usize,
    log_sigma_fixed_cols: usize,
    p_linkwiggle: Option<usize>,
) -> Result<Gauge, String> {
    if time_gauge.n_blocks() != 1 {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: format!(
                "survival location-scale finalization expected a single-block time gauge, got {} blocks",
                time_gauge.n_blocks()
            ),
        }
        .into());
    }
    let p_time_reduced = time_gauge.reduced_total();
    let p_time_full = time_gauge.raw_total();
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
    // Raw↔canonical reconciliation at the time sub-block. The time Gauge lifts
    // the inner solver's ACTIVE (reduced, canonical-gauge) time coefficients
    // back to the RAW time layout, so its linear map must
    // be at least as tall as it is wide — the active block can never carry more
    // columns than the raw block it expands into. If a future canonicalization
    // ever produces a map whose active width exceeds the raw width (the
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
             the time identifiability Gauge must map reduced→raw"
            ),
        }
        .into());
    }
    time_gauge.validate().map_err(|reason| {
        SurvivalLocationScaleError::InvalidConfiguration {
            reason: format!("survival location-scale time gauge is invalid: {reason}"),
        }
        .to_string()
    })?;

    let fixed_tail_transform = |full: usize, fixed: usize, reduced: usize| {
        let mut t = Array2::<f64>::zeros((full, reduced));
        for j in 0..reduced {
            t[[fixed + j, j]] = 1.0;
        }
        t
    };
    let p_linkwiggle_width = p_linkwiggle.unwrap_or(0);
    let p_full = p_time_full + p_threshold_full + p_log_sigma_full + p_linkwiggle_width;
    let mut affine_shift = Array1::<f64>::zeros(p_full);
    affine_shift
        .slice_mut(s![0..p_time_full])
        .assign(&time_gauge.affine_shift);
    let mut block_transforms = vec![
        time_gauge.block_transform(0),
        fixed_tail_transform(p_threshold_full, threshold_fixed_cols, p_threshold_reduced),
        fixed_tail_transform(p_log_sigma_full, log_sigma_fixed_cols, p_log_sigma_reduced),
    ];
    if let Some(width) = p_linkwiggle {
        block_transforms.push(Array2::<f64>::eye(width));
    }
    let joint_gauge = Gauge::from_block_transforms_with_shift(&block_transforms, affine_shift);
    let p_reduced = p_time_reduced + p_threshold_reduced + p_log_sigma_reduced + p_linkwiggle_width;
    assert_eq!(joint_gauge.raw_total(), p_full);
    assert_eq!(joint_gauge.reduced_total(), p_reduced);
    Ok(joint_gauge)
}

pub(crate) fn lift_conditional_covariance(
    cov_reduced: &Array2<f64>,
    finalization_gauge: &Gauge,
) -> Result<Array2<f64>, String> {
    finalization_gauge.validate().map_err(|reason| {
        SurvivalLocationScaleError::InvalidConfiguration {
            reason: format!("survival location-scale finalization gauge is invalid: {reason}"),
        }
        .to_string()
    })?;
    let p_reduced = finalization_gauge.reduced_total();
    if cov_reduced.nrows() != p_reduced || cov_reduced.ncols() != p_reduced {
        return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
            "survival location-scale covariance lift expected active matrix {p_reduced}x{p_reduced}, got {}x{}",
            cov_reduced.nrows(),
            cov_reduced.ncols()
        ) }.into());
    }
    Ok(finalization_gauge.lift_covariance(cov_reduced))
}

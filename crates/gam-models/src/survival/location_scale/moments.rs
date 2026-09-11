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
    // An eigenvalue inside the eigensolver's rounding band `γ_p·max|λ|` is zero to
    // the resolution this decomposition has: below `−band` the block is genuinely
    // indefinite, and only directions above `band` carry covariance.
    let tol = gam_linalg::roundoff::accumulation_growth(eigenvalues.len()) * max_abs_eigenvalue;
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

// #2446: THE FRACTION-TO-BOUNDARY CLIP ON THE REALIZED CONDITIONAL MEAN OF THE
// `β_w ≥ 0` BLOCK IS GONE (it was `cone_clipped_coordinate_displacement`, #2390
// / #2375). It was the THIRD owner of ONE correction.
//
// The cone is truncated once, upstream: since `0b8611a65` the covariance
// reaching `exact_survival_response_moments_row` is `Σ_π` and
// `beta_link_wiggle` is `E_π[β_w]`. `7a358c067` deleted the second owner — the
// `[0, ∞)` truncation of the scalar warp — for exactly that reason. The clip was
// the same correction a third time, applied to the conditional MEAN, and it is
// the one that cost a MOMENT rather than a tail.
//
// It cannot be defended as a support guarantee. After `7a358c067` the scalar
// warp is integrated over the WHOLE line, so 2–4 % of the law's mass already
// sits at `w < 0`. Pinning the LOCATION inside the cone while the draw around it
// is unconstrained makes no realized warp feasible; it only deletes the spread
// of the location. Feasible location + infeasible draw is not a support
// guarantee, and its price is a wrong second moment in exactly the quantity
// `response_standard_error` reports.
//
// What it cost, in closed form. This function integrates the joint Gaussian
// `N(E_π[β], Σ_π)` as an outer 3-D Gaussian over `y = (h, threshold, log σ)`
// times an inner 1-D Gaussian over `w | y`. That factorization is EXACT for a
// joint Gaussian only when the conditional mean is the affine one,
// `E[β_w] + Σ_wy Σ_yy⁻¹ (y − μ_y)`. The clip was a nonlinear map of it, so the
// nested rule stopped integrating the law it claims. When the wiggle basis row
// `b` is deterministic the whole predictor is one scalar Gaussian and the
// identity is checkable exactly:
//
//     E[η] = μ_h + q0 + bᵀE_π[β_w]
//   Var[η] = aᵀΣ_hh a + 2·aᵀΣ_hw b + bᵀΣ_ww b
//
// The unclipped rule reproduces both to quadrature precision. The clipped rule
// undershot `Var[η]` by essentially the entire cross term — `0.518 → 0.168` at
// the near-wall fixture in
// `nested_response_moment_rule_reproduces_the_scalar_gaussian_law_2446`, a
// factor of three in a reported variance.
//
// Per-coordinate rather than one global `min_j` was the right call about
// global-vs-separable and did not save it: a NEAR-WALL fit has EVERY coordinate
// near its wall, so the per-coordinate form erases every coordinate's
// cross-covariance with `(h, threshold, log σ)` anyway. That is the "silent
// erasure" the clip's own note warned a global factor would cause, reached by a
// different route.

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
                // #2446: `cond_mean` is `E_π[β_w] + Σ_wy Σ_yy⁻¹ (y − μ_y)`, the
                // AFFINE conditional mean of the link-wiggle block given the
                // realized `y = (h, threshold, log σ)`. `regression · z` is that
                // displacement written in the standardized `y` coordinates the
                // outer rule integrates over, so this loop and the constant
                // `cov_cond` below are the exact conditional law of a joint
                // Gaussian — which is what makes the outer×inner factorization
                // below an identity rather than an approximation.
                //
                // Nothing is clipped back into the `β_w ≥ 0` cone here. The cone
                // is truncated ONCE, upstream (see the note above the function):
                // clipping the displacement was a third application of the same
                // correction, it deleted the block's cross-covariance with
                // `(h, threshold, log σ)` at exactly the near-wall fits the cone
                // matters for, and it bought no feasibility because the inner
                // integral already runs over the whole line.
                let mut cond_mean = beta_w.to_owned();
                for j in 0..pw {
                    let mut displacement = 0.0;
                    for (col, &latent) in z.iter().enumerate() {
                        displacement += regression[[j, col]] * latent;
                    }
                    cond_mean[j] += displacement;
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
                // #2446: the cone is accounted for ONCE, upstream. Since
                // `0b8611a65` the covariance reaching here is `Σ_π` and
                // `beta_link_wiggle` is `E_π[β_w]` — both already carry the
                // `β_w ≥ 0` truncation — so `(w_mean, w_var)` are the moments
                // of the constrained law. Truncating the scalar again would
                // apply the same correction twice; measured, that costs a
                // factor of 40 to 300 in `E[S]`, and the ordering does not flip
                // out to thirty times the tolerance the upstream moments are
                // converged to (`ORTHANT_MOMENT_RELATIVE_TOLERANCE = 1e-3`).
                // See `artifacts/issue_2446_double_truncation_robustness.py`.
                //
                // #2679: this is the moment-matched NORMAL, and it is a known
                // approximation rather than the law. The pushforward of a
                // cone-truncated joint through `bᵀ` is not normal for `q > 1`,
                // so matching its first two moments has an error FLOOR. The
                // node mixture `gam_solve::constrained_posterior::
                // constrained_projection_law` produces from the same cubature
                // has an error RATE and puts no mass outside the cone; measured
                // at the shipped 4096-node refinement on a `q = 2` face against
                // a tensor-Simpson reference on the exact truncated density,
                // node sum `2.688e-6` vs this normal's `6.135e-4`, with `0` vs
                // `4.54e-2` infeasible mass.
                //
                // It is not cut over here because a node shifts the mean of the
                // whole coefficient vector — `β = β_unc + t + G(u − E[u])` — so
                // it moves `(h, threshold, log σ)` as well as `w`, the outer
                // rule below cannot be shared across nodes, and the exact
                // cutover costs `nodes × outer × inner` per row. #2679 carries
                // the joint low-discrepancy rule that makes it affordable. The
                // consumer that reads misplaced mass at FIRST order is a
                // quantile, and that one already runs on these nodes
                // (`constrained_projection_equal_tailed_interval`); `E[S]` and
                // `E[S²]` are smooth and do not.
                crate::quadrature::normal_expectation_nd_adaptive_result::<1, _, _, String>(
                    quadctx,
                    [x[0] + q0 + w_mean],
                    [[w_var]],
                    21,
                    |eta| {
                        let p =
                            inverse_link_survival_prob_checked(&input.inverse_link, eta[0])?;
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

    // #2679: when the fit carries an inequality cone, the posterior this rule
    // must integrate is NOT the Gaussian below. The Gaussian below is the
    // moment-matched normal of a cone-truncated law, which is wrong in kind:
    // its error is a floor rather than a rate, and it puts mass on coefficient
    // vectors the fit excluded. `truncated_survival_response_moments` prices
    // the truncated law itself, on a single joint low-discrepancy rule over the
    // constraint-normal and tangent coordinates whose per-row cost carries no
    // factor of the cubature's node count. `None` means there is no cone the
    // reported covariance was truncated against, and then the Gaussian rule
    // below is not an approximation of the posterior but IS the posterior.
    if let Some((first, second)) = truncated_survival_response_moments(
        input,
        fit,
        covariance,
        &x_threshold_dense,
        &x_log_sigma_dense,
    )? {
        return Ok((first, second));
    }

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

use super::inner_strategy::{GeometryBackendKind, HessianEvalStrategyKind};
use super::*;
use crate::linalg::utils::enforce_symmetry;

const DIAGNOSTIC_HESSIAN_FD_REL_STEP: f64 = 1e-4;
const DIAGNOSTIC_HESSIAN_FD_ABS_STEP: f64 = 1e-5;
const DIAGNOSTIC_HESSIAN_BOUNDARY_GUARD: f64 = 1e-8;

impl<'a> RemlState<'a> {
    /// Compute first and second derivatives of the smooth δ-regularized
    /// pseudo-logdet L_δ(S) with respect to ρ.
    ///
    /// # Smooth δ-regularization (response.md Section 7)
    ///
    /// Instead of the hard ε-threshold truncation that creates non-smooth
    /// derivatives at eigenvalue crossings, we use:
    ///
    ///   L_δ(S) = log det(S + δI) − m₀ log δ
    ///
    /// The derivatives are:
    ///
    ///   ∂_k L_δ = tr((S + δI)⁻¹ Aₖ)
    ///   ∂²_kl L_δ = δ_{kl} ∂_k L_δ − λₖ λₗ tr((S+δI)⁻¹ Sₖ (S+δI)⁻¹ Sₗ)
    ///
    /// where Aₖ = λₖ Sₖ.
    ///
    /// Because (S + δI) is always full rank, no eigenspace partitioning is
    /// needed and the leakage/moving-nullspace correction is automatically
    /// captured by the full-rank inverse.
    pub(super) fn structural_penalty_logdet_derivatives(
        &self,
        rs_transformed: &[Array2<f64>],
        lambdas: &Array1<f64>,
        structural_rank: usize,
        ridge: f64,
    ) -> Result<(Array1<f64>, Array2<f64>), EstimationError> {
        let k_count = lambdas.len();
        if rs_transformed.len() != k_count {
            return Err(EstimationError::LayoutError(format!(
                "Penalty root/lambda count mismatch in structural logdet derivatives: roots={}, lambdas={}",
                rs_transformed.len(),
                k_count
            )));
        }
        if k_count == 0 {
            return Ok((Array1::zeros(k_count), Array2::zeros((k_count, k_count))));
        }

        // IMPORTANT: dimensions must follow the *actual* transformed coefficient frame
        // presented by callers (possibly active-constraint projected), not self.p.
        let p_dim = rs_transformed[0].ncols();
        for (k, r_k) in rs_transformed.iter().enumerate() {
            if r_k.ncols() != p_dim {
                return Err(EstimationError::LayoutError(format!(
                    "Inconsistent penalty root width at k={k}: got {}, expected {}",
                    r_k.ncols(),
                    p_dim
                )));
            }
        }
        if p_dim == 0 || structural_rank == 0 {
            return Ok((Array1::zeros(k_count), Array2::zeros((k_count, k_count))));
        }

        let rank = structural_rank.min(p_dim);
        if rank == 0 {
            return Ok((Array1::zeros(k_count), Array2::zeros((k_count, k_count))));
        }

        // Build S(ρ) = Σ λₖ Sₖ in the full p_dim space.
        let mut s_k_full = Vec::with_capacity(k_count);
        let mut s_lambda = Array2::<f64>::zeros((p_dim, p_dim));
        for k in 0..k_count {
            let r_k = &rs_transformed[k];
            let s_k = r_k.t().dot(r_k);
            s_lambda += &s_k.mapv(|v| lambdas[k] * v);
            s_k_full.push(s_k);
        }
        if ridge > 0.0 {
            for i in 0..p_dim {
                s_lambda[[i, i]] += ridge;
            }
        }

        // Eigendecomposition of S(ρ).
        let (evals, _) = s_lambda
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;

        // Choose δ proportional to machine epsilon × spectral scale.
        let delta = super::unified::smooth_logdet_delta(evals.as_slice().unwrap());

        // Add δI to S_lambda to form (S + δI) — full rank, always invertible.
        for i in 0..p_dim {
            s_lambda[[i, i]] += delta;
        }

        // Factorize (S + δI) for solve-based trace computation.
        // No eigenspace partitioning — the full p_dim × p_dim matrix is used.
        let s_reg_factor = self.factorize_faer(&s_lambda);

        // First derivatives: ∂_k L_δ = tr((S + δI)⁻¹ Aₖ) = λₖ tr((S + δI)⁻¹ Sₖ).
        //
        // Compute Y_k = (S + δI)⁻¹ S_k via solve, then det1[k] = λ_k tr(Y_k).
        let mut y_k_mats = Vec::with_capacity(k_count);
        let mut det1 = Array1::<f64>::zeros(k_count);
        for k in 0..k_count {
            let mut y_k = s_k_full[k].clone();
            let mut y_kview = array2_to_matmut(&mut y_k);
            s_reg_factor.solve_in_place(y_kview.as_mut());
            let tr = kahan_sum((0..p_dim).map(|i| y_k[[i, i]]));
            det1[k] = lambdas[k] * tr;
            y_k_mats.push(y_k);
        }

        // Second derivatives:
        //   ∂²_kl L_δ = δ_{kl} ∂_k L_δ − λₖ λₗ tr((S+δI)⁻¹ Sₖ (S+δI)⁻¹ Sₗ)
        //             = δ_{kl} det1[k] − λₖ λₗ tr(Yₖ Yₗ)
        //
        // No leakage correction needed: (S + δI) is full rank, so the
        // moving-nullspace terms that were previously required for the hard
        // ε-threshold are automatically captured.
        let mut det2 = Array2::<f64>::zeros((k_count, k_count));
        for k in 0..k_count {
            for l in 0..=k {
                let tr_ab = Self::trace_product(&y_k_mats[k], &y_k_mats[l]);
                let mut val = -lambdas[k] * lambdas[l] * tr_ab;
                if k == l {
                    val += det1[k];
                }
                det2[[k, l]] = val;
                det2[[l, k]] = val;
            }
        }
        Ok((det1, det2))
    }

    pub(super) fn compute_lamlhessian_analytic_fallback(
        &self,
        rho: &Array1<f64>,
        bundle_hint: Option<&EvalShared>,
    ) -> Result<Array2<f64>, EstimationError> {
        // Deterministic analytic fallback used when the full exact Hessian
        // assembly is unavailable (e.g. active-subspace instability).
        //
        // We keep exact, closed-form pieces that remain robust:
        //   1) Penalty-envelope diagonal term:
        //      0.5 * beta' A_k beta = 0.5 * lambda_k * ||R_k beta||^2.
        //   2) Exact structural penalty smooth pseudo-logdet curvature:
        //      -0.5 * d²/drho² L_δ(S).
        //   3) Soft prior Hessian.
        //
        // We intentionally omit the Laplace log|H|_+ curvature block here,
        // because that block is exactly where unstable active-set boundaries
        // make second derivatives unreliable.
        let bundle_owned;
        let bundle = if let Some(b) = bundle_hint {
            b
        } else {
            bundle_owned = self.obtain_eval_bundle(rho)?;
            &bundle_owned
        };
        let k = rho.len();
        let mut h = Array2::<f64>::zeros((k, k));
        if k == 0 {
            return Ok(h);
        }

        let pirls_result = bundle.pirls_result.as_ref();
        let beta = pirls_result.beta_transformed.as_ref();
        let rs = &pirls_result.reparam_result.rs_transformed;
        let lambdas = rho.mapv(f64::exp);

        for idx in 0..k {
            if idx >= rs.len() {
                break;
            }
            let r_k = &rs[idx];
            if r_k.ncols() == 0 {
                continue;
            }
            let r_beta = r_k.dot(beta);
            let q_diag = 0.5 * lambdas[idx] * r_beta.dot(&r_beta);
            h[[idx, idx]] += q_diag;
        }

        let (structural_rank, _) = self.fixed_subspace_penalty_rank_and_logdet(
            &pirls_result.reparam_result.e_transformed,
            pirls_result.ridge_passport,
        )?;
        let (_, det2) = self.structural_penalty_logdet_derivatives(
            &pirls_result.reparam_result.rs_transformed,
            &lambdas,
            structural_rank,
            pirls_result.ridge_passport.penalty_logdet_ridge(),
        )?;
        h += &det2.mapv(|v| -0.5 * v);

        self.add_soft_priorhessian_in_place(rho, &mut h);

        // Always return a numerically strict PD matrix for downstream Newton/
        // covariance uses in fallback regimes.
        let scale = h
            .diag()
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let jitter = 1e-8 * scale;
        for i in 0..k {
            h[[i, i]] += jitter;
        }
        for i in 0..k {
            for j in 0..i {
                let avg = 0.5 * (h[[i, j]] + h[[j, i]]);
                h[[i, j]] = avg;
                h[[j, i]] = avg;
            }
        }

        if h.iter().any(|v| !v.is_finite()) {
            return Err(EstimationError::RemlOptimizationFailed(
                "Analytic fallback Hessian produced non-finite values".to_string(),
            ));
        }
        Ok(h)
    }

    fn diagnostic_hessian_fd_step(&self, rho_i: f64) -> Result<f64, EstimationError> {
        let base = DIAGNOSTIC_HESSIAN_FD_REL_STEP * (1.0 + rho_i.abs())
            .max(DIAGNOSTIC_HESSIAN_FD_ABS_STEP);
        let symmetric_room = (RHO_BOUND - DIAGNOSTIC_HESSIAN_BOUNDARY_GUARD - rho_i.abs())
            .max(0.0);
        let step = base.min(0.5 * symmetric_room);
        if !step.is_finite() || step <= 0.0 {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "diagnostic numeric Hessian has no symmetric finite-difference room at rho={rho_i:.6e}"
            )));
        }
        Ok(step)
    }

    pub(super) fn compute_lamlhessian_diagnostic_numeric(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        let k = rho.len();
        let mut h = Array2::<f64>::zeros((k, k));
        if k == 0 {
            return Ok(h);
        }

        let f0 = self.compute_cost(rho)?;
        let mut steps = Vec::with_capacity(k);
        for i in 0..k {
            steps.push(self.diagnostic_hessian_fd_step(rho[i])?);
        }

        for i in 0..k {
            let hi = steps[i];

            let mut rho_p = rho.clone();
            rho_p[i] += hi;
            let f_p = self.compute_cost(&rho_p)?;

            let mut rho_m = rho.clone();
            rho_m[i] -= hi;
            let f_m = self.compute_cost(&rho_m)?;

            h[[i, i]] = (f_p - 2.0 * f0 + f_m) / (hi * hi);

            for j in 0..i {
                let hj = steps[j];

                let mut rho_pp = rho.clone();
                rho_pp[i] += hi;
                rho_pp[j] += hj;
                let f_pp = self.compute_cost(&rho_pp)?;

                let mut rho_pm = rho.clone();
                rho_pm[i] += hi;
                rho_pm[j] -= hj;
                let f_pm = self.compute_cost(&rho_pm)?;

                let mut rho_mp = rho.clone();
                rho_mp[i] -= hi;
                rho_mp[j] += hj;
                let f_mp = self.compute_cost(&rho_mp)?;

                let mut rho_mm = rho.clone();
                rho_mm[i] -= hi;
                rho_mm[j] -= hj;
                let f_mm = self.compute_cost(&rho_mm)?;

                let hij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * hi * hj);
                h[[i, j]] = hij;
                h[[j, i]] = hij;
            }
        }

        enforce_symmetry(&mut h);
        if h.iter().any(|v| !v.is_finite()) {
            return Err(EstimationError::RemlOptimizationFailed(
                "diagnostic numeric Hessian produced non-finite values".to_string(),
            ));
        }
        Ok(h)
    }

    fn compute_lamlhessian_exact_from_bundle(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
    ) -> Result<Array2<f64>, EstimationError> {
        let mode = super::unified::EvalMode::ValueGradientHessian;
        let result = if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            self.evaluate_unified_sparse(rho, bundle, mode)?
        } else {
            self.evaluate_unified(rho, bundle, mode)?
        };
        result.hessian.ok_or_else(|| {
            EstimationError::RemlOptimizationFailed(
                "Unified Hessian returned None for VGH mode".into(),
            )
        })
    }

    pub(crate) fn compute_lamlhessian_consistent(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        let bundle = self.obtain_eval_bundle(rho)?;
        let decision = self.selecthessian_strategy_policy(rho, &bundle);
        if decision.reason == "active_subspace_unstable" {
            let rel_gap = bundle.active_subspace_rel_gap.unwrap_or(f64::NAN);
            log::warn!(
                "LAML Hessian strategy selected {:?} (reason={}, rel_gap={:.3e}).",
                decision.strategy,
                decision.reason,
                rel_gap
            );
        } else if decision.strategy != HessianEvalStrategyKind::SpectralExact {
            log::warn!(
                "LAML Hessian strategy selected {:?} (reason={}).",
                decision.strategy,
                decision.reason
            );
        }
        match decision.strategy {
            HessianEvalStrategyKind::SpectralExact => {
                self.compute_lamlhessian_exact_from_bundle(rho, &bundle)
            }
            HessianEvalStrategyKind::AnalyticFallback => {
                self.compute_lamlhessian_analytic_fallback(rho, Some(&bundle))
            }
            HessianEvalStrategyKind::DiagnosticNumeric => {
                self.compute_lamlhessian_diagnostic_numeric(rho)
            }
        }
    }

    pub(crate) fn compute_lamlhessian_exact(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        let bundle = self.obtain_eval_bundle(rho)?;
        self.compute_lamlhessian_exact_from_bundle(rho, &bundle)
    }

    pub(crate) fn compute_smoothing_correction_auto(
        &self,
        final_rho: &Array1<f64>,
        final_fit: &PirlsResult,
        base_covariance: Option<&Array2<f64>>,
        finalgrad_norm: f64,
    ) -> Option<Array2<f64>> {
        // Always compute the fast first-order correction first.
        let first_order = super::compute_smoothing_correction(self, final_rho, final_fit);
        let first_order_correction = first_order.correction.clone();
        let n_rho = final_rho.len();
        if n_rho == 0 {
            // No hyperparameters: the unified corrected covariance equals H^{-1}.
            // Validate the unified path using the spectral operator.
            if let Some(ref base_cov) = base_covariance {
                if let Ok(hop) = super::unified::DenseSpectralOperator::from_symmetric(base_cov) {
                    let outer = Array2::<f64>::zeros((0, 0));
                    let unified_diag = super::unified::compute_corrected_covariance_diagonal(
                        &[],
                        &[],
                        &outer,
                        &hop,
                    );
                    if let Ok(diag) = unified_diag {
                        let p = base_cov.nrows();
                        let max_dev = (0..p)
                            .map(|i| (base_cov[[i, i]] - diag[i]).abs())
                            .fold(0.0_f64, f64::max);
                        log::trace!(
                            "[corrected-cov] unified diagonal validation: max_dev={:.4e}",
                            max_dev,
                        );
                    }
                    let unified_full =
                        super::unified::compute_corrected_covariance(&[], &[], &outer, &hop);
                    if let Ok(full) = unified_full {
                        log::trace!(
                            "[corrected-cov] unified full norm: {:.4e}",
                            full.iter().map(|v| v * v).sum::<f64>().sqrt(),
                        );
                    }
                }
            }
            return first_order_correction;
        }
        if n_rho > AUTO_CUBATURE_MAX_RHO_DIM {
            return first_order_correction;
        }
        if final_fit.beta_transformed.len() > AUTO_CUBATURE_MAX_BETA_DIM {
            return first_order_correction;
        }
        if let Ok(bundle) = self.obtain_eval_bundle(final_rho)
            && bundle.active_subspace_unstable
        {
            // Cubature correction relies on a locally stable outer Hessian.
            // Near active-subspace crossings of H_+, the hard-truncated
            // spectral objective is not described by one smooth quadratic
            // model, so we keep only the first-order correction.
            return first_order_correction;
        }

        let near_boundary = final_rho
            .iter()
            .any(|&v| (RHO_BOUND - v.abs()) <= AUTO_CUBATURE_BOUNDARY_MARGIN);
        let grad_norm = if finalgrad_norm.is_finite() {
            finalgrad_norm
        } else {
            0.0
        };
        let highgrad = grad_norm > 1e-3;
        if !near_boundary && !highgrad {
            // Keep the hot path cheap when the local linearization is likely sufficient.
            return first_order_correction;
        }

        // Build V_rho from the outer Hessian around rho_hat.
        let mut hessian_rho = if let Some(h) = first_order.hessian_rho {
            h
        } else {
            match self.compute_lamlhessian_consistent(final_rho) {
                Ok(h) => h,
                Err(err) => {
                    log::debug!("Auto cubature skipped: rho Hessian unavailable ({}).", err);
                    return first_order_correction;
                }
            }
        };
        enforce_symmetry(&mut hessian_rho);
        let ridge = 1e-8
            * hessian_rho
                .diag()
                .iter()
                .map(|&v| v.abs())
                .fold(0.0, f64::max)
                .max(1e-8);
        for i in 0..n_rho {
            hessian_rho[[i, i]] += ridge;
        }
        let hessian_rho_inv =
            match matrix_inversewith_regularization(&hessian_rho, "auto cubature rho Hessian") {
                Some(v) => v,
                None => return first_order_correction,
            };

        let max_rhovar = hessian_rho_inv
            .diag()
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        if !near_boundary && !highgrad && max_rhovar < 0.1 {
            return first_order_correction;
        }

        use crate::faer_ndarray::FaerEigh;
        use faer::Side;
        let (evals, evecs) = match hessian_rho_inv.eigh(Side::Lower) {
            Ok(x) => x,
            Err(_) => return first_order_correction,
        };
        let mut eig_pairs: Vec<(usize, f64)> = evals
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| v.is_finite() && *v > 1e-12)
            .collect();
        if eig_pairs.is_empty() {
            return first_order_correction;
        }
        eig_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let totalvar: f64 = eig_pairs.iter().map(|(_, v)| *v).sum();
        if !totalvar.is_finite() || totalvar <= 0.0 {
            return first_order_correction;
        }

        let mut rank = 0usize;
        let mut captured = 0.0_f64;
        for (_, eig) in eig_pairs
            .iter()
            .take(AUTO_CUBATURE_MAX_EIGENVECTORS.min(eig_pairs.len()))
        {
            captured += *eig;
            rank += 1;
            if captured / totalvar >= AUTO_CUBATURE_TARGET_VAR_FRAC {
                break;
            }
        }
        if rank == 0 {
            return first_order_correction;
        }

        let base_cov = match base_covariance {
            Some(v) => v,
            None => return first_order_correction,
        };
        let p = base_cov.nrows();
        let radius = (rank as f64).sqrt();
        let mut sigma_points: Vec<Array1<f64>> = Vec::with_capacity(2 * rank);
        for (eig_idx, eigval) in eig_pairs.iter().take(rank) {
            let axis = evecs.column(*eig_idx).to_owned();
            let scale = radius * eigval.sqrt();
            let delta = axis.mapv(|v| v * scale);

            for sign in [1.0_f64, -1.0_f64] {
                let mut rho_point = final_rho.clone();
                for i in 0..n_rho {
                    rho_point[i] =
                        (rho_point[i] + sign * delta[i]).clamp(-RHO_BOUND + 1e-8, RHO_BOUND - 1e-8);
                }
                sigma_points.push(rho_point);
            }
        }
        if sigma_points.is_empty() {
            return first_order_correction;
        }

        // Disable warm-start and PIRLS-cache coupling while evaluating sigma
        // points in parallel. Cache lookups/inserts use an exclusive lock in
        // execute_pirls_if_needed(), so leaving cache enabled serializes this
        // block under contention.
        let point_results: Vec<Option<(Array2<f64>, Array1<f64>)>> = {
            let guards = (
                AtomicFlagGuard::swap(
                    &self.cache_manager.pirls_cache_enabled,
                    false,
                    Ordering::SeqCst,
                ),
                AtomicFlagGuard::swap(&self.warm_start_enabled, false, Ordering::SeqCst),
            );
            (
                guards,
                (0..sigma_points.len())
                    .into_par_iter()
                    .map(|idx| {
                        let fit_point = self.execute_pirls_if_needed(&sigma_points[idx]).ok()?;
                        let h_point = map_hessian_to_original_basis(fit_point.as_ref()).ok()?;
                        let cov_point =
                            matrix_inversewith_regularization(&h_point, "auto cubature point")?;
                        let beta_point = fit_point
                            .reparam_result
                            .qs
                            .dot(fit_point.beta_transformed.as_ref());
                        Some((cov_point, beta_point))
                    })
                    .collect(),
            )
                .1
        };

        if point_results.iter().any(|r| r.is_none()) {
            return first_order_correction;
        }

        let w = 1.0 / (sigma_points.len() as f64);
        let mut mean_hinv = Array2::<f64>::zeros((p, p));
        let mut mean_beta = Array1::<f64>::zeros(p);
        let mut second_beta = Array2::<f64>::zeros((p, p));
        for (cov_point, beta_point) in point_results.into_iter().flatten() {
            mean_hinv += &cov_point.mapv(|v| w * v);
            mean_beta += &beta_point.mapv(|v| w * v);
            let outer = beta_point
                .view()
                .insert_axis(ndarray::Axis(1))
                .dot(&beta_point.view().insert_axis(ndarray::Axis(0)));
            second_beta += &outer.mapv(|v| w * v);
        }

        let mean_outer = mean_beta
            .view()
            .insert_axis(ndarray::Axis(1))
            .dot(&mean_beta.view().insert_axis(ndarray::Axis(0)));
        let var_beta = second_beta - mean_outer;

        let mut total_cov = mean_hinv + var_beta;
        for i in 0..p {
            for j in (i + 1)..p {
                let avg = 0.5 * (total_cov[[i, j]] + total_cov[[j, i]]);
                total_cov[[i, j]] = avg;
                total_cov[[j, i]] = avg;
            }
        }
        if !total_cov.iter().all(|v| v.is_finite()) {
            return first_order_correction;
        }

        let mut corr = total_cov - base_cov;
        for i in 0..p {
            for j in (i + 1)..p {
                let avg = 0.5 * (corr[[i, j]] + corr[[j, i]]);
                corr[[i, j]] = avg;
                corr[[j, i]] = avg;
            }
        }

        log::info!(
            "Using adaptive cubature smoothing correction (rank={}, points={}, near_boundary={}, grad_norm={:.2e}, maxvar={:.2e})",
            rank,
            2 * rank,
            near_boundary,
            grad_norm,
            max_rhovar
        );
        Some(corr)
    }
}

impl<'a> RemlState<'a> {
    pub(super) fn usesobjective_consistentfdgradient(&self, rho: &Array1<f64>) -> bool {
        let _ = rho;
        self.config.link_function() != LinkFunction::Identity
            && self.config.objective_consistentfdgradient
    }
}

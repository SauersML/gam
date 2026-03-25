use super::inner_strategy::{GeometryBackendKind, HessianEvalStrategyKind};
use super::penalty_logdet::PenaltyPseudologdet;
use super::*;
use crate::linalg::utils::enforce_symmetry;

const DIAGNOSTIC_HESSIAN_FD_REL_STEP: f64 = 1e-4;
const DIAGNOSTIC_HESSIAN_FD_ABS_STEP: f64 = 1e-5;
const DIAGNOSTIC_HESSIAN_BOUNDARY_GUARD: f64 = 1e-8;

impl<'a> RemlState<'a> {
    /// Compute first and second derivatives of the exact pseudo-logdet
    /// log|S|₊ with respect to ρ.
    ///
    /// Uses eigendecomposition to identify the positive eigenspace, then
    /// computes exact derivatives on that subspace:
    ///
    ///   ∂_k L = tr(S⁺ Aₖ)
    ///   ∂²_kl L = δ_{kl} ∂_k L − λₖ λₗ tr(S⁺ Sₖ S⁺ Sₗ)
    ///
    /// where Aₖ = λₖ Sₖ and S⁺ is the pseudoinverse on the positive eigenspace.
    pub(super) fn structural_penalty_logdet_derivatives(
        &self,
        rs_transformed: &[Array2<f64>],
        lambdas: &Array1<f64>,
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

        // Build S_k = R_k^T R_k for each penalty component.
        let s_k_matrices: Vec<Array2<f64>> =
            rs_transformed.iter().map(|r_k| r_k.t().dot(r_k)).collect();

        let lambdas_slice = lambdas.as_slice().unwrap();

        let pld = PenaltyPseudologdet::from_components(&s_k_matrices, lambdas_slice, ridge)
            .map_err(|e| EstimationError::LayoutError(e))?;

        let (det1, det2) = pld.rho_derivatives(&s_k_matrices, lambdas_slice);
        Ok((det1, det2))
    }

    /// Block-local penalty logdet derivatives using `CanonicalPenalty`.
    ///
    /// When all penalties are block-disjoint, the eigendecomposition factorizes
    /// per-block at O(block_p³) instead of O(p³). Falls back to the dense path
    /// when blocks overlap.
    pub(super) fn structural_penalty_logdet_derivatives_block_local(
        &self,
        lambdas: &Array1<f64>,
        ridge: f64,
    ) -> Result<(Array1<f64>, Array2<f64>), EstimationError> {
        // Kronecker fast path: compute logdet derivatives directly from the
        // marginal eigenvalue grid.  O(d · ∏q_j) with no coordinate-frame
        // dependence — eigenvalues of Σ_k λ_k (I⊗...⊗S_k⊗...⊗I) are invariant
        // under orthogonal reparameterization, so this is correct regardless of
        // whether P-IRLS uses standard or factored Qs.
        if let Some(ref kron) = self.kronecker_penalty_system {
            let lambdas_slice = lambdas.as_slice().unwrap();
            let (_, det1, det2) = kron.logdet_and_derivatives(lambdas_slice, ridge);
            return Ok((det1, det2));
        }

        let k_count = self.canonical_penalties.len();
        if k_count == 0 || lambdas.len() != k_count {
            return Ok((Array1::zeros(k_count), Array2::zeros((k_count, k_count))));
        }

        let lambdas_slice = lambdas.as_slice().unwrap();

        let pld = PenaltyPseudologdet::from_penalties(
            &self.canonical_penalties,
            lambdas_slice,
            ridge,
            self.p,
        )
        .map_err(|e| EstimationError::LayoutError(e))?;

        let (det1, det2) =
            pld.rho_derivatives_from_penalties(&self.canonical_penalties, lambdas_slice);
        Ok((det1, det2))
    }

    fn diagnostic_hessian_fd_step(&self, rho_i: f64) -> Result<f64, EstimationError> {
        let base = (DIAGNOSTIC_HESSIAN_FD_REL_STEP * (1.0 + rho_i.abs()))
            .max(DIAGNOSTIC_HESSIAN_FD_ABS_STEP);
        let symmetric_room = (RHO_BOUND - DIAGNOSTIC_HESSIAN_BOUNDARY_GUARD - rho_i.abs()).max(0.0);
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

    pub(super) fn compute_lamlhessian_exact_from_bundle(
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
        result
            .hessian
            .materialize_dense()
            .map_err(EstimationError::RemlOptimizationFailed)?
            .ok_or_else(|| {
                EstimationError::RemlOptimizationFailed(
                    "Unified Hessian returned no analytic representation for VGH mode".into(),
                )
            })
    }

    pub(crate) fn compute_lamlhessian_consistent(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        let bundle = self.obtain_eval_bundle(rho)?;
        let decision = self.selecthessian_strategy_policy(rho, &bundle);
        if decision.strategy != HessianEvalStrategyKind::SpectralExact {
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
            HessianEvalStrategyKind::DiagnosticNumeric => {
                self.compute_lamlhessian_diagnostic_numeric(rho)
            }
        }
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
    pub(super) fn usesobjective_consistentfdgradient(&self, _: &Array1<f64>) -> bool {
        self.config.link_function() != LinkFunction::Identity
            && self.config.objective_consistentfdgradient
    }
}

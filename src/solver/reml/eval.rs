use super::inner_strategy::GeometryBackendKind;
use super::penalty_logdet::PenaltyPseudologdet;
use super::*;
use crate::linalg::utils::enforce_symmetry;

// Relative scale of the diagonal ridge added to the ρ-Hessian before
// inverting it for sigma-point construction. Matches the analogous IFT
// regularisation: tiny enough to leave well-conditioned Hessians intact,
// large enough that a near-singular Hessian still yields a usable V_ρ.
const AUTO_CUBATURE_HESSIAN_RIDGE_REL: f64 = 1e-8;
// Absolute floor for the diagonal ridge (prevents zero ridge when the
// Hessian diagonal is degenerate / all-zero).
const AUTO_CUBATURE_HESSIAN_RIDGE_ABS: f64 = 1e-8;
// Eigenvalues of V_ρ below this floor are treated as numerical zero and
// dropped from the sigma-point spectrum. Set well below any meaningful
// posterior variance scale so we don't propagate noise directions.
const AUTO_CUBATURE_EIGENVALUE_FLOOR: f64 = 1e-12;
// Inset from RHO_BOUND when clamping sigma points so the inner PIRLS
// fit at a sigma point is strictly interior to the box constraint
// (the box edge is unreachable by IRLS without barrier intervention).
const AUTO_CUBATURE_RHO_CLAMP_INSET: f64 = 1e-8;
// Skip cubature when the first-order rho-Hessian inverse already shows
// negligible posterior variance on rho (max diag < this threshold) and
// neither boundary contact nor large outer-gradient flags fired.
const AUTO_CUBATURE_RHOVAR_TRIGGER: f64 = 0.1;

impl<'a> RemlState<'a> {
    fn cached_penalty_block_structural_nullities(
        &self,
    ) -> Result<super::penalty_logdet::PenaltyBlockStructuralNullities, EstimationError> {
        if let Some(cached) = self
            .penalty_block_structural_nullities
            .read()
            .unwrap()
            .clone()
        {
            return Ok(cached);
        }
        let computed = PenaltyPseudologdet::structural_block_nullities(&self.canonical_penalties)
            .map_err(EstimationError::LayoutError)?;
        *self.penalty_block_structural_nullities.write().unwrap() = Some(computed.clone());
        Ok(computed)
    }

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
        let s_k_matrices: Vec<Array2<f64>> = rs_transformed
            .iter()
            .map(|r_k| crate::faer_ndarray::fast_atb(r_k, r_k))
            .collect();

        let lambdas_slice = lambdas.as_slice().unwrap();

        let pld = PenaltyPseudologdet::from_components(&s_k_matrices, lambdas_slice, ridge)
            .map_err(EstimationError::LayoutError)?;

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

        let cached_block_nullities = if ridge > 0.0 {
            Some(self.cached_penalty_block_structural_nullities()?)
        } else {
            None
        };
        let pld = PenaltyPseudologdet::from_penalties_with_cached_block_nullities(
            &self.canonical_penalties,
            lambdas_slice,
            ridge,
            self.p,
            cached_block_nullities.as_ref(),
        )
        .map_err(EstimationError::LayoutError)?;

        let (det1, det2) =
            pld.rho_derivatives_from_penalties(&self.canonical_penalties, lambdas_slice);
        Ok((det1, det2))
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
        std::hint::black_box(self.selecthessian_strategy_policy(rho, &bundle));
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
            if let Some(base_cov) = base_covariance
                && let Ok(hop) = super::unified::DenseSpectralOperator::from_symmetric(base_cov) {
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
        // Scale-invariant "high gradient" certificate. The first-order
        // smoothing correction is the local linearization at ρ̂; cubature
        // upgrades it when the linearization is suspect (boundary contact, or
        // the outer gradient is genuinely large). An absolute ‖g‖>1e-3 gate
        // is wrong at every scale: biobank deviance ≈ 10⁵–10⁶ makes ‖g‖≈1
        // perfectly fine but trips the gate unconditionally, while tiny CI
        // problems with deviance ≈ 10–100 stay under 1e-3 even when actually
        // unconverged. Use the same `τ·(1+|F|)` rescaling the OUTER paths use
        // (BFGS / ARC / trust-region via `outer_scaled_tolerance`); deviance
        // is the dominant term in the REML cost at every scale and is the
        // natural cost proxy reachable from `PirlsResult`.
        const HIGHGRAD_REL_TOL: f64 = 1e-3;
        let cost_scale = 1.0 + final_fit.deviance.abs();
        let highgrad = grad_norm > HIGHGRAD_REL_TOL * cost_scale;
        if !near_boundary && !highgrad {
            // Keep the hot path cheap when the local linearization is likely sufficient.
            return first_order_correction;
        }

        // If the first-order path used a rank-deficient pseudo-inverse, the
        // ρ-Hessian was indefinite or near-singular and the matrix-free ridged
        // inverse used below would silently impute spurious variance along the
        // dropped (unidentified) directions. Cubature sigma points propagated
        // through that spurious V_ρ would manufacture higher-order corrections
        // that are not supported by the data. The principled response is to
        // honor the rank deficiency: return the first-order correction (which
        // is already the correct rank-deficient inflation on the identified
        // subspace) and skip cubature entirely.
        if let Some(rank) = first_order.active_rank
            && rank < n_rho {
                log::debug!(
                    "Auto cubature skipped: first-order V_ρ is rank-deficient \
                     ({}/{}); higher-order propagation would impute spurious \
                     variance along unidentified directions.",
                    rank,
                    n_rho,
                );
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
        let ridge = AUTO_CUBATURE_HESSIAN_RIDGE_REL
            * hessian_rho
                .diag()
                .iter()
                .map(|&v| v.abs())
                .fold(0.0, f64::max)
                .max(AUTO_CUBATURE_HESSIAN_RIDGE_ABS);
        for i in 0..n_rho {
            hessian_rho[[i, i]] += ridge;
        }
        let Some(hessian_rho_inv) =
            matrix_inversewith_regularization(&hessian_rho, "auto cubature rho Hessian")
        else {
            return first_order_correction;
        };

        let max_rhovar = hessian_rho_inv
            .diag()
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        if !near_boundary && !highgrad && max_rhovar < AUTO_CUBATURE_RHOVAR_TRIGGER {
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
            .filter(|(_, v)| v.is_finite() && *v > AUTO_CUBATURE_EIGENVALUE_FLOOR)
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

        let Some(base_cov) = base_covariance else {
            return first_order_correction;
        };
        let p = base_cov.nrows();
        let radius = (rank as f64).sqrt();
        let mut sigma_points: Vec<Array1<f64>> = Vec::with_capacity(2 * rank);
        for (eig_idx, eigval) in eig_pairs.iter().take(rank) {
            let axis = evecs.column(*eig_idx).to_owned();
            let scale = radius * eigval.sqrt();
            let delta = axis.mapv(|v| v * scale);

            let lo = -RHO_BOUND + AUTO_CUBATURE_RHO_CLAMP_INSET;
            let hi = RHO_BOUND - AUTO_CUBATURE_RHO_CLAMP_INSET;
            for sign in [1.0_f64, -1.0_f64] {
                let mut rho_point = final_rho.clone();
                rho_point
                    .iter_mut()
                    .zip(delta.iter())
                    .for_each(|(r, &d)| *r = (*r + sign * d).clamp(lo, hi));
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
            // Use scaled_add to avoid allocating intermediate scaled arrays
            // on every sigma-point iteration; numerically equivalent to
            // `mean += &arr.mapv(|v| w * v)`.
            mean_hinv.scaled_add(w, &cov_point);
            mean_beta.scaled_add(w, &beta_point);
            let beta_col = beta_point.view().insert_axis(ndarray::Axis(1));
            let beta_row = beta_point.view().insert_axis(ndarray::Axis(0));
            let outer = beta_col.dot(&beta_row);
            second_beta.scaled_add(w, &outer);
        }

        let mean_outer = mean_beta
            .view()
            .insert_axis(ndarray::Axis(1))
            .dot(&mean_beta.view().insert_axis(ndarray::Axis(0)));
        let var_beta = second_beta - mean_outer;

        let mut total_cov = mean_hinv + var_beta;
        enforce_symmetry(&mut total_cov);
        if !total_cov.iter().all(|v| v.is_finite()) {
            return first_order_correction;
        }

        let mut corr = total_cov - base_cov;
        enforce_symmetry(&mut corr);

        log::debug!(
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
